# foldseek -- porting plan (linux-gfx90a lead)

## Project
- Name: foldseek
- Upstream: https://github.com/steineggerlab/foldseek
- Default branch: main
- Role: fast/sensitive protein-structure search (3Di + AA Gotoh-Smith-Waterman); shares the MMseqs2 code lineage and vendors MMseqs2 under `lib/mmseqs`.

## DISPOSITION: CLEAN PORT (tractable). Strategy A. State -> planned. Dispatch a porter.
Effort class: MEDIUM (single self-contained CUDA library `libmarv`; pure-CMake gate; real wave-size + intrinsic-guard work, but a portable SIMT path already exists and is the default on non-Hopper hardware -- no CK/MFMA reimplementation required).

This is explicitly NOT CK-deferred. The deciding evidence is below ("GPU-path root cause").

## Existing AMD support
- Web search ("foldseek ROCm/AMD GPU/HIP", "MMseqs2 libmarv GPU DPX"): NO AMD/ROCm port of foldseek, MMseqs2-GPU, or libmarv/CUDASW++4.0 exists. Confirms prior research.
- The NVIDIA MMseqs2-GPU paper (Nature Methods 2025, https://www.nature.com/articles/s41592-025-02819-8) and the upstream Readme both state the GPU path targets NVIDIA only (Turing+/Ampere+); the NVIDIA NIM integration is CUDA-only. No ROCm-DS-style separate AMD project, no upstream rocm/hip branch, no community HIP fork found.
- Authoritative-vs-community judgment: N/A (nothing found). From-scratch HIP port our way.

## Build classification: CMake (Strategy A). Evidence
- Top-level `CMakeLists.txt`: `project(foldseek C CXX)`, plain `add_subdirectory` tree, no `find_package(Torch)`, no `torch.utils.cpp_extension`, no setup.py/pyproject. Not a pytorch extension.
- `ENABLE_CUDA` at top-level (`CMakeLists.txt:12`) only drives ProstT5's bundled `ggml` LLM backend (`GGML_CUDA ON`, lines 87-92) -- that is the ProstT5 embedding inference, NOT the structure-search aligner. Do not conflate.
- The structure-search GPU code is `lib/mmseqs` -> `lib/mmseqs/CMakeLists.txt:297-302`: `if (ENABLE_CUDA) add_subdirectory(lib/libmarv/src)`.
- `lib/mmseqs/lib/libmarv/src/CMakeLists.txt`: `project(... LANGUAGES CXX CUDA)`, `add_library(marv ...)`, `enable_language(CUDA)` model, `CUDA_SEPARABLE_COMPILATION ON` + `CUDA_RESOLVE_DEVICE_SYMBOLS ON`, NVCC flags `-rdc=true --extended-lambda --expt-relaxed-constexpr`. Classic pure-CMake CUDA library -> colmap-model Strategy A.

## Port strategy: A (compat header + `enable_language(HIP)` + mark `.cu` LANGUAGE HIP)
Rationale: self-contained CUDA library (`libmarv`) plus a thin host integration in mmseqs (`GpuUtil`, `ungappedprefilter.cpp`, `gpuserver.cpp`). Gate a `USE_HIP`/ROCm option that (a) flips `libmarv/src/CMakeLists.txt` to `enable_language(HIP)` and marks the `.cu`/`.cuh` TUs `LANGUAGE HIP`, (b) maps the CUDA runtime/Thrust/CUB spellings via a single `cuda_to_hip.h` compat header, (c) replaces the NVCC-only flags (`-rdc=true` -> `-fgpu-rdc` + a HIP device-link, `--extended-lambda`/`--expt-relaxed-constexpr` are nvcc-only and dropped on HIP). Keep the NVIDIA path byte-identical behind the option.

Note RDC: libmarv builds with separable compilation. On ROCm this is `-fgpu-rdc` with a device-link step; combined with the heavily-templated half2/short2/float kernel instantiations, add `--offload-compress` (PORTING_GUIDE "Large libraries") pre-emptively if the link bloats.

## GPU-path root cause (THE deciding question) -- PORTABLE SIMT, default non-DPX
foldseek's GPU prefilter/aligner IS the shared MMseqs2 aligner: `lib/mmseqs/lib/libmarv` is CUDASW++4.0 ("marv"), vendored (not a live submodule; it is populated in-tree). mmseqs reaches it via `src/prefiltering/ungappedprefilter.cpp` + `commons/GpuUtil` + `util/gpuserver.cpp`, exposed to foldseek with `--gpu`.

libmarv provides TWO arithmetic paths per alignment mode, selected by a runtime `config.dpx` flag:
- DPX path (`short2`, 16-bit integer SIMD): uses Hopper DPX intrinsics `__vimax3_s16x2(_relu)`, `__viaddmax_s16x2`, `__vibmax_u16x2`, `__vmaxs2`, `__vadd2` (in `mathops.cuh` `MathOps<short2>` and `kernelhelpers.cuh`). 7 DPX call sites. Built only from the `*_instantiation_dpx*.cu` TUs.
- Portable path: gapless uses `MathOps<half2>` (`__hmax2`/`__hadd2`/`__hmax` FP16 SIMD); Smith-Waterman uses `MathOps<float>`/`MathOps<int>`. Built from `*_instantiation_half2*.cu` and `*_instantiation_float.cu`. NO DPX, NO tensor-core, NO dp4a, NO inline-PTX math -- standard SIMT.

DPX is gated on hardware and is NOT the default for non-Hopper:
- `gapless_kernel_config.cuh` / `smithwaterman_kernel_config.cuh`: `getOptimalKernelConfigs_*(deviceId)` switch on compute-capability major.minor; the default for an UNRECOGNIZED device falls through to `*_sm89()` (Ada/L40S), whose every entry sets `dpx=0`. sm80 (Ampere) is also all `dpx=0`. Only sm90/sm103/sm120/sm121 (Hopper+) set `dpx=1`.
- `benchmarking.cuh:1007`: `const bool supportsDPX = ccMajor >= 9;`
- Dispatch (`cudasw4.cuh:4055,4132,4206,4268`): `if(!config.dpx){ half2/float kernels } else { short2 DPX kernels }`.

Therefore on an AMD GPU (which reports an unrecognized cc), libmarv naturally selects the `dpx=0` half2/float SIMT path. The port keeps the DPX path NVIDIA-only behind a guard (or simply never selects it on HIP) and ports the portable path -- a correctness-first mechanical HIP port with no CK/MFMA rewrite. A later AMD-native pass (MFMA / packed-int16) is optional perf, not required for correctness.

ABSENT (scanned, none found): tensor-core/WMMA/`mma_sync`/`nvcuda`, `__dp4a`, CUTLASS/CuTe. Only inline PTX is `mov.u32 %0,%%laneid` (one site).

## CUDA surface inventory (libmarv)
- Kernels: gapless + Smith-Waterman PSSM kernels, `cg::tiled_partition<groupsize>` group-strided (groupsize in {4,8,16,32} typical), `__global__`/`__device__` throughout.
- Warp/lane: `__shfl_down_sync`/`__shfl_up_sync`/`__shfl_sync` (20 sites), all with `0xFFFFFFFF` 32-bit masks; reductions with `offset=16` start (32-lane assumption); `__reduce_max_sync` (sm_80+) guarded by `#if __CUDA_ARCH__ >= 800` with a shuffle `#else`; one inline-PTX `laneid` read (`hpc_helpers/cuda_helpers.cuh:194`).
- Cooperative groups: `cg::tiled_partition`, `cooperative_groups::reduce`.
- Libraries: Thrust + CUB (device sort/scan/reduce, custom thrust allocators in `hpc_helpers/custom_thrust_allocators.cuh`) -> rocThrust/hipCUB. NO cuBLAS/cuFFT/cuRAND/cuSPARSE.
- Memory/exec: streams + events via RAII wrappers (`hpc_helpers/cuda_raiiwrappers.cuh` CudaStream/CudaEvent already move-only with guarded dtor), pinned/managed in `hpc_helpers/simple_allocation.cuh`/`timers.cuh`. No textures/surfaces. NVTX wrapped behind `NO_NVTOOLSEXT` (already disabled in libmarv target) -- map to rocTX or stub.
- char signedness: mmseqs sets `-fsigned-char` globally; libmarv casts chars into `char4`/short2 score packs -- watch the HIP `char4`-is-`char` vs CUDA `signed char` base-type class (PORTING_GUIDE) if any reference-accessor code is touched (low risk here, values are packed via unions).

## Risk list
1. WAVE SIZE (primary). gfx90a is wave64; libmarv is written for 32-lane warps. Concrete hazards:
   - `__shfl_*_sync(0xFFFFFFFF, ...)` 32-bit masks and `offset=16` reduction starts assume a 32-lane warp. The `cg::tiled_partition<N>` groups with N<=32 are logical-warp-width-32 and arch-agnostic (fine per guide), but the bare `0xFFFFFFFF`/offset-16 full-warp reductions in `kernelhelpers.cuh` and `kernels.cuh` must be made `warpSize`-generic or confirmed to run only within a <=32 logical group.
   - `getPaddedQueryLength` (`kernelhelpers.cuh:9`) pads the query layout by `sizeof(char4) * 32` -- a host+device-SHARED warp-width constant baked into the in-memory query format. This is the PORTING_GUIDE "warp-size-dependent SERIALIZED/in-memory FORMAT" class: if a wave64 device strides by 64 while the host pads by 32 (or vice versa), the layout desyncs. Pin the padding to a fixed width that both host and device agree on, independent of the per-arch device warp constant; verify the kernel's actual border-stride matches.
   - Multi-arch build test required: `-DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100"` must emit both code objects AND a single-arch run must be correct; reference-less "looks sane" is NOT sufficient (use the CPU-mode cross-check in the test plan as the oracle).
2. `__reduce_max_sync` guard. The `#if __CUDA_ARCH__ >= 800` selects the intrinsic on CUDA and a shuffle fallback otherwise. Under HIP `__CUDA_ARCH__` is undefined so the `#else` shuffle path is taken (acceptable), but confirm the guard does not accidentally call a HIP-missing `__reduce_max_sync`; make the guard `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800`.
3. Inline PTX `laneid` (`cuda_helpers.cuh:194`) -- replace with HIP `__lane_id()` (or `threadIdx-derived % warpSize`) under a HIP guard.
4. Rule-of-five on stream/event RAII (`cuda_raiiwrappers.cuh`). Wrappers look move-only with guarded dtor (good), but verify default-init `= 0` / no double-destroy under HIP (colmap CuTexObj class).
5. DPX path must be compiled-out or never-selected on HIP. Ensure the `*_instantiation_dpx*.cu` TUs are excluded from the HIP build (or the intrinsics stubbed) so the half2/float path is the only one built; confirm `config.dpx` can never be 1 on an AMD device (default sm89 fallback already gives 0, but a user-supplied tuning config file could force dpx=1 -> guard it).
6. `-ffp-contract` / FP16 numeric drift: half2 scores are integer-valued alignment scores (discrete), so 1-ULP fp drift is unlikely to change integer max-scores, but the SW `float` path could; validate exact score equivalence against CPU mode and pin `-ffp-contract=on` if scores diverge.
7. RDC + template instantiation bloat: `-fgpu-rdc` device-link plus many half2/short2/float instantiations may inflate the object/host image; apply `--offload-compress` if the link approaches the 2 GiB reach wall.
8. Thrust/CUB block-collective TempStorage reuse race on wave64 (PORTING_GUIDE cub class) -- audit any back-to-back block-collective reuse for a missing `__syncthreads()`.
9. Device discovery: `Marv::getDeviceIds()` / `cudaGetDeviceCount` must see the AMD GPU post-hipify; the gpuserver shared-memory handshake (`GpuUtil`) is host-only IPC and should port unchanged.

## File-by-file change list (lead, gfx90a)
- NEW `lib/mmseqs/lib/libmarv/src/cuda_to_hip.h` -- compat header: include `<hip/hip_runtime.h>` (and `<cstring>/<cstdlib>` BEFORE it per gpuRIR lesson), alias the cudaXxx runtime + Thrust/CUB + fp16 symbols the library uses; no-op to `<cuda_runtime.h>` on NVIDIA.
- `lib/mmseqs/lib/libmarv/src/CMakeLists.txt` -- add `USE_HIP` gate: `enable_language(HIP)`, mark `ALIGN_SOURCES` `.cu`/`.cuh` `LANGUAGE HIP`, set `HIP_ARCHITECTURES` from `${CMAKE_HIP_ARCHITECTURES}` (no hardcoded literal), swap NVCC flags for HIP equivalents (`-fgpu-rdc`, drop nvcc-only `--extended-lambda`/`--expt-relaxed-constexpr`), optional `--offload-compress`. Exclude `*_instantiation_dpx*.cu` on HIP.
- `lib/mmseqs/CMakeLists.txt` and top-level `CMakeLists.txt` -- thread the ROCm option down to the libmarv subdir (parallel to existing `ENABLE_CUDA`); decide whether to reuse `ENABLE_CUDA` semantics or add `ENABLE_ROCM`/`USE_HIP`.
- `hpc_helpers/cuda_helpers.cuh` -- `lane_id()` HIP branch.
- `kernelhelpers.cuh` -- guard `__reduce_max_sync` (`defined(__CUDA_ARCH__) &&`), make full-warp reductions warpSize-generic; fix `getPaddedQueryLength` warp-width constant to be arch-stable.
- `mathops.cuh` -- `MathOps<half2>`/`<float>`/`<int>` should port as-is; ensure DPX `MathOps<short2>` is behind a NVIDIA/`!USE_HIP` guard so HIP never instantiates DPX intrinsics.
- `cuda_raiiwrappers.cuh` -- verify default-init/move/dtor under HIP.
- mmseqs host glue (`GpuUtil`, `ungappedprefilter.cpp`, `gpuserver.cpp`) -- likely compile unchanged once the runtime symbols resolve; touch only if they include CUDA headers directly.

## Build commands (gfx90a)
Configure (structure-search GPU path; ProstT5 ggml-CUDA left OFF to isolate the aligner port first):
```
cmake -S projects/foldseek/src -B projects/foldseek/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON -DENABLE_CUDA=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build projects/foldseek/build -j16
```
(Exact option name -- reuse `ENABLE_CUDA` to drive the HIP gate vs add `USE_HIP` -- is an open question below; the porter resolves it when wiring the CMake. Build the `foldseek` executable; libmarv is pulled in via mmseqs when the GPU option is on.)

## Test plan (real GPU correctness gate; egress-feasible)
Oracle = CPU mode vs GPU mode on the SAME bundled inputs (no download).
- Inputs: the in-repo `example/` set (~28 SCOP domain structures `d1*`, plus `1tim.pdb.gz`/`8tim.pdb.gz`), ~90 KB each, already in the clone. Total << 3 MB. Also libmarv's bundled `allqueries.fasta` (45 KB) for a libmarv-level `align` smoke test.
- GPU correctness gate (decisive): build a tiny DB and search the example structures against themselves, once in CPU mode and once with `--gpu 1`, then diff the result tables (hits, scores, alignment endpoints):
  ```
  foldseek createdb example/ exDB
  foldseek search exDB exDB aln_cpu tmp --gpu 0   # CPU reference
  foldseek search exDB exDB aln_gpu tmp --gpu 1   # GPU (libmarv half2 path on gfx90a)
  # convertalis both; diff the .m8 / score columns -- must match within scoring tolerance
  ```
  GPU prefilter requires a padded DB (`makepaddeddb.sh` / `foldseek makepaddeddb`); the gpuserver path may need `foldseek gpuserver`. The porter/validator confirm the exact GPU-search invocation from `data/structuresearch.sh` and `--help`. The wave-size correctness proof is: GPU hits/scores == CPU hits/scores on this set, on gfx90a (wave64) AND on gfx1100 (wave32, follower) -- a divergence is the warp-size bug.
- libmarv-standalone smoke: `makedb allqueries.fasta db && align --query allqueries.fasta --db db` (self-search), score sanity.
- Non-GPU regression set MUST NOT regress: foldseek/MMseqs2 CPU `easy-search`/`search` on the same `example/` inputs (the `--gpu 0` path above doubles as this). The full upstream `regression/` (foldseek-regression + MMseqs2-Regression) are git submodules NOT populated by `--depth=1` and pull external data -- DEFER those; the bundled example cross-check is the primary gate given the ~40-160 KB/s egress limit.
- Egress feasibility: GOOD. All required test inputs ship in the repo; no model/DB download is needed for the aligner correctness gate. (ProstT5 weights and the regression datasets are large downloads -- avoided by scoping validation to the bundled structures and leaving ProstT5/ggml-CUDA out of the first port.)

## Open questions
1. CMake option naming: reuse the existing `ENABLE_CUDA` to also mean "GPU on, HIP backend when USE_HIP" vs introduce a separate `USE_HIP`/`ENABLE_ROCM`. Prefer not to overload `ENABLE_CUDA` (it also gates ProstT5 ggml-CUDA, which we are NOT porting in pass 1) -- likely a dedicated HIP gate scoped to libmarv.
2. Exact foldseek GPU-search CLI for the validation gate (`--gpu`, padded DB, whether a running `gpuserver` is mandatory) -- read `data/structuresearch.sh` and `--help` during bringup.
3. Does the half2 gapless path's score quantization match the CPU 3Di+AA scoring exactly, or only the relative ranking? Determines whether the gate is exact-score or rank/tolerance-based.
4. RDC device-link behavior on ROCm for libmarv's separable compilation + whether `--offload-compress` is needed (decide at first link).
5. ProstT5 ggml-CUDA backend: out of scope for pass 1 (separate ggml-HIP concern); confirm foldseek's structure search works without it (it does -- ProstT5 is for the predict/embedding mode, not the SW aligner).

## Follower platforms
gfx1100/gfx1151 (wave32) reuse this branch and validate first. Wave32 is actually the NATURAL fit for libmarv's 32-lane assumptions, BUT the `getPaddedQueryLength` `*32` format constant and any wave64-specific fix added for gfx90a must stay wave-agnostic (PORTING_GUIDE wave-geometry traps). Cross-arch consistency: gfx1100 GPU results must match gfx90a GPU results (and CPU) on the example set -- deterministic diff, not "looks sane".
