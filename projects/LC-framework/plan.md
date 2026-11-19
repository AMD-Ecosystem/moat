# LC-framework -- HIP/ROCm port plan

## Project
- Name: LC-framework
- Upstream: https://github.com/burtscher/LC-framework (Martin Burtscher et al., Texas State University)
- Default branch: main (HEAD f72e323, "June 2026 release")
- License: BSD-3-Clause (commercial/derivative OK -- PR-able)
- What it is: a framework that auto-searches and code-generates customized lossless / guaranteed-error-bounded lossy GPU (and CPU/OpenMP) compressors from a pipeline of component + preprocessor algorithms. The GPU code is CUDA C++; the search driver and standalone-codegen are Python.

## Existing AMD support (decisive finding)
LC-framework ALREADY carries an author-maintained HIP path in upstream `main`. It is NOT a fresh CUDA-only project. Evidence in-tree:
- `include/macros.h:77` -- an AMD shim block (`__syncwarp` no-op, `#define __trap() abort()`, `atomicOr_block/atomicAdd_block/atomicSub_block` aliases, `namespace cuda::std { using ::std::numeric_limits; }`) gated on the wavefront macro.
- `framework.h:80`, `compressor-framework.cu:56`, `decompressor-framework.cu:56` -- `#if !defined(__HIPCC__)` guards around `<cuda/atomic>` / `<cuda/std/limits>` ("a CUDA-only library that cannot be automatically replaced by HIPIFY"), with a `volatile`-based `atomicRead/atomicWrite` AMD fallback (`framework.h:165-180`).
- The WS==64 wave64 ballot/popc paths in `components/include/d_zero_elimination.h` and `d_repetition_elimination.h` (`#if defined(WS) && (WS == 64)` -> `unsigned long long` ballot + `__popcll`), plus `__ffsll`/`__clzll` 64-bit variants.
- `.gitignore` lists `*.prehip` -- the HIPIFY backup extension -- confirming the intended workflow is generate -> `hipify-perl` -> `hipcc`.
- README:26: "The GPU code is written in CUDA. A HIP version is also included."

Authoritative-vs-community judgment: AUTHORITATIVE (the original LC authors wrote it; it lives in upstream main, not a fork). No competing AMD fork or PR exists (`gh pr list ... ROCm OR HIP OR AMD` empty; forks scanned, none AMD; no separately-named ROCm project found). Per PORTING_GUIDE "AUTHORITATIVE but incomplete AMD port" -> the value is VALIDATING AND IMPROVING it on a real AMD GPU and upstreaming the small fixes that make it actually build on a current ROCm, NOT a from-scratch re-port.

Decision: PROCEED as a validate-and-fix port. The HIP path was written against an OLDER ROCm where `__AMDGCN_WAVEFRONT_SIZE` was a predefined macro; on ROCm 7.2.1 that macro is GONE, so the entire HIP path silently disables itself and the build breaks / mis-selects wave width. This is a genuine, small, upstreamable delta (details below). This is exactly the gap that also blocks the deferred cuSZ item `cusz-lc-framework-hip`.

### Strategic link to cuSZ
cuSZ bundles LC as `third_party/lc` (PSZ_ACTIVATE_LC, OFF on HIP). That submodule is PINNED to an OLD LC commit (`1cac09c`) that PREDATES any HIP support (verified: no `__HIPCC__`/`__AMDGCN_WAVEFRONT_SIZE`/`*.prehip` in the bundled tree). So the deferred cuSZ task remains valid and is unblocked by this port in two ways: (1) the current upstream LC main already has the HIP scaffolding; (2) this port lands the ROCm-7.2.x fixes that make it actually compile. After this port, cuSZ can bump its submodule to the fixed LC and wire HIPIFY into PSZ_ACTIVATE_LC. (Do not modify cuSZ here; just record the unblock.)

## Build classification
Not pure-CMake and not a torch extension -- there is NO CMakeLists.txt, NO Makefile, NO setup.py / pyproject. The build is a Python CODE-GENERATION step followed by a single hand-run compiler invocation:
- `./generate_Device_LC-Framework.py` copies `framework.cu`->`lc.cu`, `framework.h`->`lc.h`, splices the component/preprocessor switch-cases and the include list into the `##...##`-tagged scaffold, writes `include/consts.h` (with the wave-width `WS` define), and PRINTS an `nvcc ... -o lc lc.cu` command.
- `./generate_Host_LC-Framework.py` is the CPU/OpenMP analogue (g++). `generate_Hybrid_*` is test-only.
- `generate_standalone_{CPU,GPU}_compressor_decompressor.py` emit a fixed-pipeline standalone compressor/decompressor from `compressor-framework.cu`/`decompressor-framework.cu`.

Evidence the generator does NOT emit kernel bodies (so this is NOT a "generator that emits CUDA you must teach to emit HIP" in the kernel sense): the generators only splice `#include` lines, `switch/case` dispatch, `printf` banners, and the WS define (`generate_Device_LC-Framework.py:98-105`, `:143-219`). The actual `__global__`/`__device__` kernels live in the static `framework.cu` + `components/*.h` + `preprocessors/*.h` headers, which are copied verbatim. BUT the generator DOES emit one wave-width guard into `consts.h` (`generate_Device_LC-Framework.py:101`, `generate_Hybrid_LC-Framework.py:138`), so the generators ARE part of the port surface (they must emit a ROCm-7.2.x-valid wave selector).

`ext_type` recorded as `cmake` in upstream.json/status.json as the closest bucket (standalone, non-torch, host-untouched, `.cu` compiled by the GPU toolchain). The real build is "codegen + single hipcc TU"; treat it as the Strategy-A family (host C++ untouched, only the `.cu`/headers see HIP), NOT Strategy B.

## Port strategy
Single-source, HIPIFY-driven, matching the project's existing design (this is what the authors already chose). Do NOT introduce a `cuda_to_hip.h` shim or rename files: the upstream already relies on `hipify-perl` to rewrite `cudaXxx`->`hipXxx`, `<cuda.h>`/`<cuda_runtime.h>`->`<hip/hip_runtime.h>`, `cub`->`hipcub`, `thrust`->`rocThrust`, and keeps a tiny hand-written shim in `macros.h` for what HIPIFY cannot map. The port's job is to make that path build and run correctly on a CURRENT ROCm and to wire the HIPIFY step into the documented build, then upstream the deltas.

Recommended mechanism (lead, gfx90a):
1. Replace the obsolete `__AMDGCN_WAVEFRONT_SIZE` predefined-macro dependency with a ROCm-7.2.x-valid selector (see Risk list). This is the core fix and touches both the static sources and the two generators.
2. Fix the two latent include/namespace gaps that only bite on HIP (missing thrust include; `cuda::std::numeric_limits` not aliased on the path actually taken). See File-by-file.
3. Document + script the HIP build: a generate -> `hipify-perl -inplace` -> `hipcc --offload-arch=$arch` recipe, parallel to the existing `nvcc` recipe in README. The `*.prehip` gitignore is already in place. Prefer an `arch`-parameterized recipe (no hardcoded gfx90a) so followers reuse it with only `--offload-arch=<arch>`.

Mechanical vs AMD-native: a mechanical/correctness-first port is the right call. These are bit-shuffle / RLE / quantizer kernels using warp ballot+popc prefix sums, not GEMM/attention -- there is no CUTLASS/wgmma/tensor-core hot path that would warrant a rocWMMA/CK rewrite. The authors' wave64 ballot paths are already hand-written; we make them activate, not rewrite them.

## CUDA surface inventory
Kernels: many `__global__`/`__device__` across `framework.cu`, `components/*.h` (BIT/CLOG/HCLOG/DIFF*/RLE/RRE/RZE/RARE/RAZE/TCMS/TCNB/TUPL* x word sizes 1/2/4/8), `preprocessors/*.h` (QUANT_* lossy quantizers, LOR1D), `verifiers/*.h`.
- Warp intrinsics (heavy): `__shfl`/`__shfl_up`/`__shfl_down`/`__shfl_xor`, `__ballot`, `__any`, `__all`, `__ffs`/`__ffsll`, `__clz`/`__clzll`, `__popc`/`__popcll`. On CUDA, `macros.h:66-74` maps the bare names to the `_sync(~0,...)` variants (variadic macros already in place); on HIP the bare HIP builtins are used directly. The wave64 ballot results are `unsigned long long` on the `WS==64` branch (correct).
- `__trap()`: used in `framework.h`, `components/include/d_{zero,repetition}_elimination.h`, and 4 `preprocessors/d_QUANT_NOA_*` files. Mapped to `abort()` in the `macros.h` AMD block (must be reachable -- see Risk).
- `__syncwarp()`: no-op shim in the AMD block (must be reachable).
- `cuda::atomic` / `cuda::std::limits`: `<cuda/atomic>` + `<cuda/std/limits>`, `#if !defined(__HIPCC__)` guarded; AMD uses `volatile` read/write fallback. `cuda::std::numeric_limits` aliased to `std::numeric_limits` in the AMD block.
- Thrust: `thrust::minmax_element` + `device_ptr`/`raw_pointer_cast` in several `preprocessors/d_QUANT_*NOA*` and `d_QUANT_I*` files. HIPIFY -> rocThrust (`/opt/rocm/include/thrust/*` present). Verified rewrites cleanly.
- CUB: `cub::DeviceScan::InclusiveSum` in `preprocessors/d_LOR1D_i32.h`. HIPIFY -> `hipcub::DeviceScan` (`/opt/rocm/include/hipcub/hipcub.hpp` present). Verified rewrites cleanly.
- Host runtime API: `cudaMalloc`/`cudaFree`/`cudaMemcpy(Async)`/`cudaMemset`/`cudaDeviceSynchronize`/`cudaSetDevice`/`cudaGetDeviceProperties`/`cudaGetLastError`/`cudaEvent*` (timing). All 1:1 HIPIFY-mapped.
- NOT present (no risk): textures/surfaces, cuRAND, cuBLAS, cuFFT, cuSPARSE, cuDNN, managed memory, layered cudaArray, pinned-via-pitch. `cudaMallocHost`/`cudaFreeHost` (pinned) used twice -> `hipHostMalloc`/`hipHostFree`, trivial.

## Risk list
1. OBSOLETE WAVEFRONT MACRO (the core port issue, HIGH). The HIP path keys everything on `#if defined(__AMDGCN_WAVEFRONT_SIZE)` (and `== 64`). PROVEN on this host: ROCm 7.2.1 hipcc/clang does NOT predefine `__AMDGCN_WAVEFRONT_SIZE` for any arch -- `clang -dM -E` shows only `__GFX9__`/`__gfx90a__` (gfx90a), `__GFX11__`/`__gfx1100__` (gfx1100), `__GFX11__`/`__gfx1151__` (gfx1151). Consequences if unfixed: (a) the `macros.h` AMD shim block never compiles -> `__trap`, `__syncwarp`, `atomic*_block`, `cuda::std::numeric_limits` all undeclared -> build fails; (b) `consts.h` falls to `#define WS 32` on a wave64 device -> the `WS==64` ballot/popc prefix-sum paths in zero/repetition-elimination never activate -> SILENT WRONG COMPRESSION / round-trip corruption on gfx90a even if it built. PORTING_GUIDE confirms: "There is no `__AMDGCN_WAVEFRONT_SIZE__` macro in ROCm 7.2.x; the `__GFX*__` guards are the supported per-arch compile-time selector." Fix: replace the guards with `__GFX9__`(/`__GFX8__`)->wave64 else wave32, gated under `__HIP_PLATFORM_AMD__`/`__HIPCC__`. This is the dietgpu/cuSZ-class wave-width fix done the PORTING_GUIDE way.
   - Multi-arch corollary: because `WS` and the AMD shim both must come from the DEVICE-pass `__GFX*__` selector (constexpr, device-only), the SAME generated source must build a gfx90a;gfx1100 fat binary and pick wave64 on the CDNA code object and wave32 on the RDNA one. Do not bake a single `-DWS=64`. Validate with `llvm-objdump --offloading` showing both code objects.
2. SERIALIZED-FORMAT WAVE-WIDTH COUPLING (verify, MED). LC advertises "bit-for-bit the same result on CPUs and GPUs" and cross-device decompress. The zero/repetition-elimination bitmap layout uses `WS`-wide ballots (e.g. `d_zero_elimination.h:235` writes `bmout_b[... subwarp*8 + sublane/8]` from a 64-bit ballot split into 32-bit halves via `>> (lane & 32)`). MUST confirm the COMPRESSED FORMAT is wave-width-INDEPENDENT (a wave64 gfx90a output decompressible by a wave32 device and by the CPU), per the dietgpu lesson. The `(lane & 32)` half-splitting suggests the authors already pinned the on-wire layout to 32-bit groups regardless of physical wave width -- but this is the highest-value correctness gate: a wave32-vs-wave64 cross-decompress (and GPU-vs-CPU cross-decompress) test is REQUIRED, not just same-device round-trip.
3. MISSING THRUST INCLUDE (LOW, latent upstream bug exposed by HIP). `preprocessors/d_QUANT_INOA_0_f64.h` uses `thrust::minmax_element`/`raw_pointer_cast` with NO local `#include <thrust/...>` (its f32 sibling HAS them -- copy-paste omission). On CUDA it compiles by transitive luck (NVIDIA `<cuda/std>`/`<cuda/atomic>` drag in thrust); on HIP that include is `__HIPCC__`-guarded off, so thrust is never pulled in and it fails ("use of undeclared identifier 'thrust'"). PROVEN: adding the 3 thrust includes (matching the f32 file) fixes it. Upstreamable as a plain correctness fix.
4. cuda::std::numeric_limits ON THE TAKEN PATH (LOW->MED). `preprocessors/d_QUANT_NOA_{0,R}_{f32,f64}.h` call `cuda::std::numeric_limits<...>::min()` (and `__trap()`) at device scope. The `macros.h` alias `namespace cuda::std { using ::std::numeric_limits; }` only exists inside the (currently-dead) `__AMDGCN_WAVEFRONT_SIZE` block, so once Risk 1 is fixed by re-gating that block on `__GFX9__`/`__HIPCC__`, this alias comes alive and these compile. PROVEN failure ("use of undeclared identifier 'cuda'") occurs precisely because the block is dead on 7.2.1. The fix for Risk 1 also fixes this -- but confirm `<limits>` (or `<cuda/std/limits>`'s replacement) is in scope so `std::numeric_limits` resolves; add `#include <limits>` to the AMD shim if needed.
5. nodiscard warnings on hipError_t (COSMETIC). ROCm's `hipMemcpy*`/`hipFree` are `[[nodiscard]]`; the code ignores returns -> warnings, not errors. Leave as-is (matches upstream's CUDA behavior); do not churn.
6. shift-negative-value warnings in QUANT_*_f64 (COSMETIC, pre-existing UB, identical on CUDA). `~0LL << mantissabits`. Out of scope; do not "fix" (would change CUDA byte-output).
7. NVIDIA-build preservation (BC gate). The fix must keep the CUDA path byte-identical: re-gate the AMD block so the `#if defined(__CUDA_ARCH__)`/`__AMDGCN_WAVEFRONT_SIZE` structure still selects the original CUDA branches. PR claim "CUDA build unchanged" -> nvcc compile-check before opening (PORTING_GUIDE PR-prep gate); the missing-thrust-include (Risk 3) is additive and helps CUDA too.
8. No CMake floor / no graphics / no smid pool / no textures -- none of the other PORTING_GUIDE fault classes apply (compute-only, no GL interop -> gfx90a viable; no `__smid()` pools; no resource-handle RAII).

## File-by-file change list (lead, gfx90a)
- `include/macros.h:77` -- change `#if defined(__AMDGCN_WAVEFRONT_SIZE)` to a ROCm-7.2.x-valid AMD guard (`#if defined(__HIP_PLATFORM_AMD__)` for the host-visible shims; the wave-width-specific parts under `__GFX9__`). Ensure `__trap`/`__syncwarp`/`atomic*_block`/`cuda::std::numeric_limits` (and `#include <limits>`) are reachable on AMD. Keep the CUDA branch untouched.
- `framework.cu:59`, `compressor-framework.cu:45`, `decompressor-framework.cu:45` -- replace `__AMDGCN_WAVEFRONT_SIZE == 64` wave-width selection with `__GFX9__` (device pass) -> 64 else 32, under an AMD guard. (The `.cpp` CPU twins at `:45` are OpenMP-only and need no GPU change, but keep them consistent if the same header is shared.)
- `include/consts.h` is GENERATED, so fix the EMITTER: `generate_Device_LC-Framework.py:101-105` and `generate_Hybrid_LC-Framework.py:138` -- emit the `__GFX9__`-based WS selector instead of `__AMDGCN_WAVEFRONT_SIZE == 64`. (Mirror in `generate_Host_*`/standalone generators if they emit the same block -- audit all 5 generators.)
- `preprocessors/d_QUANT_INOA_0_f64.h` -- add `#include <thrust/extrema.h>` / `<thrust/execution_policy.h>` / `<thrust/device_ptr.h>` (match the f32 sibling). Pre-hipify CUDA spelling; HIPIFY rewrites to rocThrust. (Helps CUDA too.)
- README.md + a build note -- document the HIP build (generate -> `hipify-perl -inplace lc.cu lc.h <headers>` -> `hipcc --offload-arch=<arch> -DUSE_GPU -I. -std=c++17 -o lc lc.cu`) parallel to the `nvcc` block, in the project's house style. Document `--offload-arch=<arch>` (not a pinned gfx90a). PR-prep step.
- New-file attribution: only if a new helper file is added (none currently anticipated -- edits are in-place); if a HIP build script is added, carry the AMD copyright + `Jeff Daily` author per CLAUDE.md.

Note on the HIPIFY step: the porter must decide whether to (a) keep relying on `hipify-perl` at build time (document it, smallest source diff -- the authors' model) or (b) make the sources HIPIFY-clean so `hipcc` builds the `.cu` directly. Option (a) matches upstream intent and the `*.prehip` gitignore; recommended. Either way the wave-macro + thrust-include fixes are the same.

## Build commands (gfx90a lead)
```
# in a working copy
./generate_Device_LC-Framework.py
hipify-perl -inplace lc.cu lc.h
for h in framework.h include/macros.h $(find components preprocessors verifiers -name '*.h'); do hipify-perl -inplace "$h"; done
hipcc -O3 --offload-arch=gfx90a -DUSE_GPU -I. -std=c++17 -o lc lc.cu
# multi-arch fat-binary correctness check:
hipcc -O3 --offload-arch=gfx90a --offload-arch=gfx1100 -DUSE_GPU -I. -std=c++17 -o lc_fat lc.cu
llvm-objdump --offloading lc_fat | grep -E 'gfx90a|gfx1100'
```
(Match upstream nvcc flags where they affect numerics: `-fmad=false`-equivalent FMA control. nvcc uses `-fmad=false`/`-ffp-contract=off`; for hipcc add `-ffp-contract=off` to keep the lossy-quantizer numerics matching the CPU/CUDA gold, per PORTING_GUIDE's `-ffp-contract` lesson. Lossless components are integer-only and unaffected.)

A smoke build was DONE during planning in agent_space (lossless BIT_4/RLE_4 pipeline): it surfaced exactly Risks 1, 3, 4 and confirmed thrust/hipcub headers exist; the wave-macro + thrust-include fixes are the gating deltas.

## Test plan (real GPU)
LC has a built-in round-trip verifier -- this is the validation harness, no external test framework.
- Primary GPU correctness (lossless round-trip, bit-for-bit): `./lc <input.dat> AL "" "<pipeline>"`. `AL` mode runs compress+decompress on the GPU and verifies the decompressed output is bit-for-bit equal to the input (README:88). Run for several pipelines covering the warp-collective paths: e.g. `"BIT_4 RLE_4"`, `"RLE_1"`, `".+ .+"` small-input search, and zero/repetition-elimination-exercising data.
- Exhaustive pair self-test: `./lc <input.dat> TS` verifies all component pairs round-trip (test-only mode). Use a modest input to bound runtime.
- Lossy quantizers (exercises thrust/hipcub + cuda::std paths): `./lc <f32.dat> AL "QUANT_ABS_0_f32(0.001)" "<components>"` with a MAXABS verifier (README:149), confirming the error bound holds and INOA/NOA/LOR1D preprocessors run on ROCm.
- CROSS-WAVE / CROSS-DEVICE FORMAT GATE (Risk 2, REQUIRED): compress on gfx90a (wave64), decompress the resulting `LC.encoded` on (a) the CPU build and (b) on gfx1100 (wave32) in the follower stage; the decompressed file must match the original byte-for-bit. This is the consistency gate that a same-device round-trip cannot provide and is the decisive proof the wave-width fix preserves the serialized format. The follower (gfx1100) MUST diff its decode against the gfx90a-produced encode, not just self-round-trip.
- Non-GPU regression set: the CPU/OpenMP build (`generate_Host_LC-Framework.py` + g++) and its `AL`/`TS` round-trip must continue to pass unchanged (our edits are AMD-guarded; verify the host build is byte-identical). nvcc CUDA-build compile-check before the PR (BC gate).

Inputs: use small real datasets (any binary file works; LC is type-agnostic for lossless, f32/f64 arrays for lossy). Generate a few KB-MB test files locally; egress is constrained so avoid large dataset downloads.

## Delivery / upstream note
The fixes (wave-macro modernization + missing thrust include + HIP build doc/script) are a clean upstream PR to burtscher/LC-framework: the project WANTS HIP support (it ships the scaffolding) and merges PRs (one external PR already merged). This is a contribute-upstream target, NOT a link-the-fork reference repo. Open ONE PR after gfx90a + gfx1100 validate. Scope the PR claim to "restores/repairs the HIP build on current ROCm and adds the build recipe", not "adds AMD support from scratch".

## Open questions
- Which ROCm did upstream's HIP path originally target (where `__AMDGCN_WAVEFRONT_SIZE` was predefined)? Affects whether to keep a `defined(__AMDGCN_WAVEFRONT_SIZE) ||` fallback alongside the `__GFX*__` selector for older ROCm back-compat. Default: keep both (additive, no cost).
- Does upstream want the HIPIFY step documented as-is, or sources made hipify-clean so `hipcc` compiles `.cu` directly? Plan recommends documenting the HIPIFY step (matches `*.prehip` design); confirm in the PR description.
- cuSZ submodule bump is a SEPARATE follow-up on the cuSZ project (deferred item `cusz-lc-framework-hip`), not part of this port; this port unblocks it but does not touch cuSZ.
