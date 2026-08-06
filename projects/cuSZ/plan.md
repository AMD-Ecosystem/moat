# cuSZ -- ROCm/HIP port plan (re-port against upstream master)

## Project

- Name: cuSZ
- Upstream: https://github.com/szcompressor/cuSZ
- Default branch: master
- Base SHA (fork moat-port, reset to clean master): e1c0135
- Lead platform: linux-gfx90a (CDNA2, wave64), ROCm 7.2.x
- Build class: pure CMake (Strategy A). ext_type=cmake.

This is a FRESH re-port, not a git-rebase of the old commits. The old in-tree HIP
scaffold (base d3cde38, validated e443183) is reference-only. Upstream commit
e5cceb9a ("eradicate HIP-related setup", 2026-04-30) deleted that scaffold and
master is ~30 commits ahead with a heavy refactor.

## Existing AMD support

- Upstream docs: no AMD/ROCm/HIP mention (`grep -rniE 'amd|rocm|hip|gfx[0-9]' README* doc/ changelog` is empty -- e5cceb9a removed the prior mentions).
- Upstream branches/PRs: no rocm/hip branch, no open ROCm/HIP/AMD PR. There is a
  stale historical tag `22sep-hip` (an old experiment), and the `eradicate HIP`
  commit was authored by the maintainer (J. Tian) -- it was a "remove the
  half-built dual-source scaffold" cleanup (renamed `*.cuhip.inl` -> `*.cu.inl`,
  dropped `cmake/hip.cmake` and the HIP examples), NOT a "we reject HIP
  contributions" policy. The repo is NOT a karpathy-style link-the-fork repo; an
  upstream PR re-adding a clean, tested HIP backend is the correct delivery
  vehicle.
- Separate AMD project: `hpdps-group/hipSZ` exists, but it ports **cuSZp**
  (cuSZ-plus, a single-kernel fixed-length-encoding compressor) -- a DIFFERENT,
  simpler algorithm. It does NOT cover this repo's framework (Lorenzo/spline
  prediction + Huffman + the new HFR Huffman-ReVISIT family + FZG). So hipSZ is
  not an authoritative port of szcompressor/cuSZ; a HIP port of THIS framework
  still adds distinct value.
  - URL: https://github.com/hpdps-group/hipSZ -- community/academic, ports cuSZp not cuSZ; NOT a base to adopt.

Decision: PROCEED with a from-scratch HIP port of szcompressor/cuSZ master,
informed (not rebased) by the old-base cheat-sheet in notes.md. No mechanical
AMD-native rewrite is warranted: these are bit-packing / histogram / prefix-scan /
Lorenzo-spline kernels, not GEMM/attention; a correctness-first mechanical HIP
port via the surviving c_cu2hip macros is the right first (and likely only) step.

## Build classification (evidence)

Pure CMake, Strategy A.

- Top-level `CMakeLists.txt:1-10`: `set(PSZ_BACKEND "CUDA" ...)` then
  `project(CUSZ LANGUAGES CUDA CXX C ...)`; backend dispatched at lines 25-33 to
  `cmake/cuda.cmake` or `cmake/sycl.cmake`. No `find_package(Torch)`, no
  `torch.utils.cpp_extension`, no setup.py torch dep.
- `cmake/cuda.cmake` compiles ~73 `.cu` files natively (project LANGUAGES CUDA)
  and explicitly marks two host files `LANGUAGE CUDA`
  (`cmake/cuda.cmake:222`: `set_source_files_properties(psz/src/cli/cli.cc psz/src/cli/executor.cc PROPERTIES LANGUAGE CUDA)`).
- Sub-projects are independent CMake projects: `portable/CMakeLists.txt`,
  `codec/hf/CMakeLists.txt`, `codec/fzg/CMakeLists.txt`, `utils/CMakeLists.txt`
  each `project(... LANGUAGES ... CUDA)` and hardcode `find_package(CUDAToolkit)`,
  `CUDA::cudart`, and their own `*_USE_CUDA` defines. The HIP backend must reach
  each of these (see file-by-file list).

## Port strategy (Strategy A, single-source reuse -- NOT the old dual-source model)

Rationale: the old port (74 files) created separate `*.hip` mirror source files
beside each `*.cu`. Upstream then DELETED exactly that dual-source pattern
(e5cceb9a renamed `*.cuhip.inl` -> `*.cu.inl` and collapsed `.hip` files). The
re-port must align with the new single-source direction: reuse the SAME
`.cu`/`.cu.inl` files marked `LANGUAGE HIP` (the compat-header approach), translating
spellings through the surviving `portable/include/macro/c_cu2hip_*.h` macros.
This yields a far smaller diff than the old approach and matches upstream's own
refactor.

Mechanics:
1. Add `PSZ_BACKEND=HIP` as a third backend option in the top-level CMake and in
   each sub-project CMake, parallel to CUDA and ONEAPI.
2. Add `cmake/hip.cmake` modeled on `cmake/cuda.cmake`: `enable_language(HIP)`,
   define `PSZ_USE_HIP` + `_PORTABLE_USE_HIP` (mirroring cuda.cmake's
   `PSZ_USE_CUDA`/`_PORTABLE_USE_CUDA` at cuda.cmake:5-8), mark every `.cu`
   source (and the two CUDA-marked `.cc`) `LANGUAGE HIP`, swap `CUDA::cudart` ->
   `hip::device`/`hip::host`, cuRAND -> hipRAND, drop/replace nvml/cupti/driver.
   Do NOT pin `CMAKE_HIP_ARCHITECTURES`; rely on `enable_language(HIP)`
   auto-detect, overridable with `-DCMAKE_HIP_ARCHITECTURES=<arch>`.
3. Re-add the `_PORTABLE_USE_HIP` / `PSZ_USE_HIP` include + macro branches that
   e5cceb9a stripped from the portable headers (the surviving 7 files already
   have HIP branches; the central `cxx_backends.h` HIP branch was removed and
   must be restored -- see below).

## CUDA surface inventory (master)

- Kernels / device code: ~73 `.cu` + 32 `.cu.inl`/`.cuh`. Families:
  - Lorenzo predictor: `psz/src/kernel/{lrz_c,lrz_x,proto_lrz_c,proto_lrz_x}.cu*`, `psz/src/compile/*.cu`.
  - Spline (y24/y25): `psz/src/kernel/{spl_y24,spl_y25}.cuh`, `psz/src/compile/spl_y2*_{c,x}_u{1,2}.cu`.
  - Histogram: `psz/src/kernel/{hist_generic,histsp}.cu*`, `psz/src/compile/{hist_generic,histsp}.cu`.
  - Sparse/compaction: `psz/src/kernel/spvn.cu`, `test/src/detail/t_compact.inl`.
  - Huffman + HFR (new family): `codec/hf/src/compile/*.cu` (hfr_r1..r4, hfr_encode_r0..r4, hfr_concat, hfr-pbkc_r0..r4, hfr-pbkgo_r2..r4, hf_kernels.cu) plus `codec/hf/src/*.cuh`/`*.cu.inl` (hfr.cu.inl, hfr_encode_c.cuh, hfr_concat.cuh, hfr-pbkc_c.cuh, hfr-pbkgo_c.cuh, hfr-pbk_decoder.cu, hfr_pick_book.cu).
  - FZG codec: `codec/fzg/src/{fzg_kernel.cu,fzg_c.cu.inl,fzg_x.cu.inl}`.
  - Stat/eval: `utils/src/*.cu.inl` + `utils/src/compile/*.cu` (identical, extrema-f4/f8, calc_err, assess, maxerr, viewer).
  - LC third_party: `third_party/lc_gen/*.cu`.
- Warp intrinsics: `__shfl_up_sync(...,32)` in `psz/src/kernel/wave32.cu.inl`
  (width-32 LOGICAL warp -- arch-agnostic, fine); `__shfl_sync(0xffffffff, x, 0)`
  3-arg form in `codec/hf/src/hfr_encode_c.cuh:56` paired with `(threadIdx.x & 31)==0`
  leader (hfr_encode_c.cuh:54); `__shfl*`/`__popc` in the fzg and hfr concat/encode
  headers; `psz/include/_future/warp_top1.cuh`. NO hardcoded wave64 lane geometry
  found.
- Atomics: `utils/src/atomics.cu.inl` `atomicAddFp<double>` gated on
  `__CUDA_ARCH__>=600` (CAS-loop fallback otherwise); `atomicAdd`/`atomicCAS`/
  `atomicMin` widely. spvn/lrz_c/histsp use integer `atomicAdd`.
- Libraries: cuRAND (`portable/src/utils/rand.cu.cc`, `CUDA::curand`) -> hipRAND;
  `CUDA::nvml` + `CUDA::cuda_driver` (cuda.cmake:178-179, `verinfo_nv.cu`
  deviceQuery sample) -> drop or HIP-equivalent; `CUDA::cupti`
  (example bin_hf, cuda-example.cmake:14) -> drop (profiling-only). NO
  cuBLAS/cuFFT/cuSPARSE. Thrust used via `thrust::device_ptr` in extrema -> rocThrust (header-compatible).
- Textures/surfaces: none found (no tex/surf objects). No texture-pitch risk.
- Pinned/managed memory: `cudaMallocHost`/`cudaMallocManaged` via c_cu2hip macros
  (-> hipHostMalloc/hipMallocManaged, already in c_cu2hip_0_translation.h:17-19).
- Streams/events: `cudaStream_t`/event wrappers in portable `gpu_stream.hh`/
  `gpu_event.hh`/`cxx_backends.h` (HIP branch must be restored).
- Runtime enum: `portable/include/c_type.h:15` `_ptb_runtime` = `{ SEQ, SIMD,
  OPENMP, CUDA, SYCL, THRUST_DPL }` -- NO HIP and NO ROCM value, but
  `portable/include/backend.h:13` references `_ptb_runtime::HIP` under
  `PSZ_USE_HIP`. This is a dangling reference (see risks).

## Risk list (fault classes to expect)

1. **Runtime-enum / PROPER_RUNTIME selection (BLOCKER if unhandled).**
   `backend.h:13` defines `PROPER_RUNTIME _ptb_runtime::HIP` but the enum
   (`c_type.h:15`) has no `HIP` member. The old base hit the same and aliased to
   the (then-present) `ROCM` value; master's enum has neither HIP nor ROCM.
   `PROPER_RUNTIME` is used as a non-type template arg in the test/example path
   (`test/src/detail/t_spv.cu.inl` `spv_*_naive<PROPER_RUNTIME>`,
   `example/src/bin_hist.cc` `hist<PROPER_RUNTIME,T>`), so it must be a valid
   enumerator. Fix: add a `HIP` enumerator to `_ptb_runtime` and ensure every
   `if constexpr (R==CUDA||R==SYCL||R==SEQ)` dispatch in
   `utils/include/compare.hh` (analysis::GPU_probe_extrema, assess_quality) and
   any `kernel`-level runtime switch treats HIP the same as CUDA (reuse the
   `psz::cuda::GPU_*` impls compiled as HIP). Confirm no `switch`/`static_assert`
   rejects the new enumerator. Verify the enum is not serialized into the archive
   format (it is a compile-time dispatch tag, not header data -- confirm during port).

2. **Variadic warp-shuffle macros (3-arg vs 4-arg).**
   `c_cu2hip_1_fix_primitives.h` defines only the 4-arg
   `__shfl_sync(MASK,VAR,SRC,WIDTH)` forms, but `hfr_encode_c.cuh:56` uses the
   3-arg `__shfl_sync(0xffffffff, p_incomp, 0)`. Make the macros variadic so both
   the 3-arg (default-width) and 4-arg forms map to `__shfl(...)`. (Cheat-sheet
   item 8.)

3. **Central portable header HIP branches removed by e5cceb9a.**
   `portable/include/mem/cxx_backends.h` has only `_PORTABLE_USE_CUDA ->
   <cuda_runtime.h>` and the CUDA GPULEN3/stream/event macros; the
   `_PORTABLE_USE_HIP` branch (43 lines per the eradicate-commit stat) is gone.
   Must restore: `#elif defined(_PORTABLE_USE_HIP)` -> `#include
   <hip/hip_runtime.h>` + include the three `c_cu2hip_*` macro headers + the HIP
   GPULEN3/`GPU_BACKEND_SPECIFIC_STREAM`/event-pair macros. This is the primary
   include hook that pulls the translation macros into every TU. The 7 surviving
   HIP-branch files (cxx_mem_ops.h, cxx_smart_ptr.h, timer.hh, query_dev.cu.hh,
   query.hh, extrema.cu.inl, backend.h) are already correct; audit them against
   the restored cxx_backends.h for consistency.

4. **Double-precision atomicAdd.** `atomicAddFp<double>` falls to the CAS loop on
   HIP (`__CUDA_ARCH__` undefined). The CAS loop is correct but slow; prefer the
   native HIP path (`atomicAdd(double*)` works on ROCm; `unsafeAtomicAdd` is the
   fast hardware form on gfx90a). Add a `defined(__HIP_DEVICE_COMPILE__)` branch
   that uses the native/`unsafeAtomicAdd` path. (Cheat-sheet "double atomicAdd".)

5. **Warp size (wave64 vs wave32).** Logical-warp width-32 ops
   (`__shfl_up_sync(...,32)` in wave32.cu.inl, `threadIdx.x%32`/`&31` leaders in
   histsp/lrz_c/hfr_encode) are arch-agnostic and correct on both wave widths.
   No hardcoded wave64 geometry found. Risk is LOW but the followers
   (gfx1100/gfx1201, wave32) must still re-validate -- a 32-lane subgroup on a
   64-wide wavefront is the case to watch. Do NOT introduce any `WARP_SIZE`
   build constant.

6. **CUDA-specific library deps with no 1:1 ROCm need.** `verinfo_nv.cu`
   (NVIDIA deviceQuery sample, `<cuda.h>` + nvml + driver) -- replace with a HIP
   version-info TU (the old port aliased CUDA-named functions in a verinfo.hip)
   or guard it out and provide the HIP query via the surviving query_dev.cu.hh
   HIP branch. `CUDA::cupti` (bin_hf profiling) -> drop on HIP. `CUDA::curand` ->
   `hip::hiprand` (rand.cu.cc needs hipRAND macro coverage: curandGenerator_t,
   curandCreateGenerator, curandGenerateUniform/Double).

7. **histsp tuning-test wrong include path (known).** The old base's
   `test_histsp` referenced a misspelled include; on master the unit is
   `test/src/tune_histsp.cu` + `test/src/detail/tune_histsp.cu.inl`. Verify the
   include path resolves under HIP; if the legacy `test_histsp_*` breaks, it is a
   tuning/perf test (not core) and may be excluded as before -- but first try to
   fix the path rather than exclude.

8. **Per-subproject backend plumbing.** Each of portable/, codec/hf/, codec/fzg/,
   utils/ is a standalone CMake project that hardcodes CUDA. utils/CMakeLists.txt
   already lists HIP in `set_property(CACHE PSZ_BACKEND PROPERTY STRINGS CUDA HIP
   ONEAPI)` and handles `PSZ_BACKEND_UPPER` (lines 30-32) but only implements the
   CUDA branch -- add the HIP branch there and in the other three. Missing any
   one leaves a sub-library compiled as CUDA and the link fails.

9. **Fresh-allocation-not-zero / OOB neighbor reads (general AMD strictness).**
   Audit the spline anchor-block and Lorenzo stencil kernels for edge +/-1 reads
   and partial-write outputs that assume zeroed device memory. (Cheat-sheet
   general items; not previously flagged for cuSZ but worth a pass given the new
   HFR/spline code.)

## File-by-file change list (estimate; verify exact lines during port)

New files:
- `cmake/hip.cmake` -- modeled on cmake/cuda.cmake; `enable_language(HIP)`,
  `PSZ_USE_HIP`/`_PORTABLE_USE_HIP`, mark `.cu`+the two `.cc` `LANGUAGE HIP`,
  hip::device link, hipRAND, drop nvml/cupti/driver. Add `--offload-compress`
  to HIP compile opts if any TU's fatbin is large (HFR template matrix is a
  candidate; measure).

Edited CMake:
- `CMakeLists.txt` (top) -- add HIP branch to the backend `if/elseif`
  (lines 4-10 project(), 25-33 include()), `PROPERTY STRINGS CUDA HIP ONEAPI`.
- `cmake/probe.cmake` -- add the HIP arch/compiler info echo branch (lines 11-20).
- `portable/CMakeLists.txt` -- HIP backend: `_PORTABLE_USE_HIP`, hip::device,
  hipRAND instead of CUDA::curand (lines 22-49).
- `codec/hf/CMakeLists.txt`, `codec/fzg/CMakeLists.txt` -- HIP project lang,
  PHF_USE_HIP/FZG, hip::device, mark `.cu` LANGUAGE HIP.
- `utils/CMakeLists.txt` -- implement the already-stubbed HIP branch (lines
  30-32, 66-100, 129-181): define PSZ_USE_HIP/_PORTABLE_USE_HIP, `enable_language(HIP)`,
  build eval_cu/eval_viewer_cu from the same `.cu` sources as HIP.
- `test/CMakeLists.txt` + a new `test/cmake/hip-test*.cmake` (or generalize
  cuda-test*.cmake) -- the pure-ctest unit tests, bin_hf matrix, and cusz CLI
  matrix, sources marked LANGUAGE HIP.
- `example/CMakeLists.txt` + `example/cmake/hip-example.cmake` -- bin_hf (needed
  by the bin_hf ctest matrix), bin_hist, demo; drop CUDA::cupti.

Edited source (HIP branches / fixes):
- `portable/include/c_type.h` -- add `HIP` to `_ptb_runtime` enum (line 15).
- `portable/include/mem/cxx_backends.h` -- restore the `_PORTABLE_USE_HIP`
  include + macro branch (the central hook).
- `portable/include/macro/c_cu2hip_1_fix_primitives.h` -- variadic shuffle macros.
- `utils/src/atomics.cu.inl` -- HIP double-atomicAdd branch.
- `utils/include/compare.hh` -- analysis dispatch: treat HIP like CUDA
  (GPU_probe_extrema line 56, assess_quality line 70).
- `psz/src/cli/verinfo_nv.cu` (+ a HIP verinfo TU) -- HIP version-info path.
- `portable/src/utils/rand.cu.cc` -- hipRAND coverage (or via c_cu2hip macros).
- Audit the 7 surviving HIP-branch files for consistency post-restore.

Expected scope: smaller than the old 74-file dual-source diff because we reuse
single-source `.cu` files; concentrated in CMake + ~6-10 portable/utils headers.

## Build commands (gfx90a lead)

```bash
cd /var/lib/jenkins/moat/projects/cuSZ/src
rm -rf build-hip && \
cmake -S . -B build-hip \
  -DPSZ_BACKEND=HIP \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++
cmake --build build-hip -j$(nproc)
```

Notes:
- `-DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++` was required at the old base to
  stop the host C++ compiler receiving `-x hip --offload-arch=` from the
  compile-settings interface; expect the same.
- `-DCMAKE_PREFIX_PATH=/opt/rocm` is required on clean ROCm containers for
  `find_package(hip)`/hipRAND (PORTING_GUIDE clean-env rule).
- Followers: same command with `-DCMAKE_HIP_ARCHITECTURES=gfx1100` (or gfx1201 on
  Windows TheRock) and no source change.

## Test plan

Validation gate = the pure-ctest harness on real GPU (gfx90a), no regression in
CPU-only tests.

GPU correctness tests (must pass):
```bash
cd build-hip
HIP_VISIBLE_DEVICES=0 ctest --output-on-failure -j1
```
- Unit: test_l1_compact, test_stat_identical1/2, test_stat_max_error,
  test_mem_unique (GPU); test_hf_revisit_altcode (HFR), test_statfn.
- HFR/codec matrix (synth data, no external files): the `bin_hf` ctest matrix
  (cuda-test-bin_hf.cmake -> hip equivalent): hf/hfr/hfr-pbkc/hfr-pbkgo over
  cauchy-mild, cauchy-sharp, uniform-256, uniform-1024 with CR asserts. These are
  the core HFR-family GPU correctness checks new on master.
- cusz CLI round-trip matrix (cuda-test-cusz.cmake): SKIPs (rc=77) when
  `$CUSZ_TEST_DATA` datasets (RTM/HURR/NYX) are absent. Run if datasets are
  available locally; otherwise document as skipped (not a failure).

CPU-only regression set (must not regress): test_zigzag, test_lrz_seq,
test_hf_cpu_serial_codebook (test_hfserial), portable/test/* (arg_builder,
kv_binder, kv_parse, str2num, check_in, val_eq).

CLI smoke:
```bash
python3 -c "import numpy as np; (np.sin(np.linspace(0,10,10000))).astype('float32').tofile('test.f32')"
HIP_VISIBLE_DEVICES=0 ./cusz -z -i test.f32 -t f32 -m abs -e 0.001 -l 100x100 --codec hfr
HIP_VISIBLE_DEVICES=0 ./cusz -x -i test.f32.cusza --compare test.f32
```

CUDA-path BC check (PR-prep, not lead bring-up): the port adds an additive HIP
backend; the CUDA build must stay intact. Compile-check `PSZ_BACKEND=CUDA` with
nvcc on this GPU-less host before the PR (conda cuda-nvcc), per PORTING_GUIDE.

## Open questions

- Does `_ptb_runtime` (the dispatch enum) ever get serialized into the `.cusza`
  archive header? Confirmed compile-time-only on inspection (used as a template
  tag), but re-verify the header struct in `psz/src/header.c`/`psz/include` during
  the port -- if it IS stored, the HIP enumerator value must be pinned, not the
  arch-default constant (cheat-sheet "warp-size-dependent serialized format"
  analogue).
- HFR template-instantiation fatbin size on gfx90a: the `hfr_r1..r4` /
  `hfr_encode_r0..r4` / `hfr-pbkc/go_r*` matrix may produce large device objects;
  measure and add `--offload-compress` if a TU approaches the link-reach wall.
- `verinfo_nv.cu`: cleanest HIP replacement -- a small HIP deviceQuery TU vs
  aliasing CUDA-named entry points. Decide during port; keep the CUDA file
  untouched for the CUDA build.
- Known upstream functional issues from the old base (--report time,cr crash;
  some -x decode paths) may resurface; they are pre-existing upstream bugs, not
  port regressions -- reproduce on the CUDA build to confirm before reporting.

## Delta plan: linux-gfx1100

Follower platform. RDNA3 (gfx1100, wave32), ROCm 7.2.1. Reuse the SAME
moat-port branch and the SAME build/test commands as the gfx90a lead (Strategy A,
single-source `.cu` marked LANGUAGE HIP); change only `-DCMAKE_HIP_ARCHITECTURES`.
Do NOT re-plan or re-port from scratch. No source change is expected; do not add a
follower commit unless a genuine wave32 build/correctness fix is required.

### Why this needs a fresh GPU run (not a carry-forward)

The current head (`07db1e28`, validated on gfx90a at `5d43e441`, carried forward
to head as doc-only) is the FRESH RE-PORT against upstream master
(`a6d765e8 -> 5d43e441 -> 07db1e28`). gfx1100's only prior PASS (notes "linux-gfx1100
validation", validated_sha `aff8ee6`/`26b1f91`) was on the OLD base
(`d3cde38..e443183`), a DIFFERENT commit set with a different (74-file dual-source)
port shape. The re-port also exercises HFR/FZG codec GPU tests that never ran on the
old base. So gfx1100 starts at `unclaimed` and must validate the re-port on real
wave32 hardware -- there is no validated ancestor to carry from.

### The one wave32-critical delta to confirm

The re-port introduced an arch-unified `__ballot_sync` that the old base lacked,
and it is the single wave-width-sensitive primitive in the build:

    portable/include/macro/c_cu2hip_1_fix_primitives.h:19
    #define __ballot_sync(MASK, PRED) \
      ((unsigned int)(__ballot(PRED) >> (__lane_id() & ~(unsigned)31)))

Rationale (notes "New / confirmed fault classes" item 2; review 2026-06-25): the
FZG codec runs block (32,32) with linear tid = y*32+x, and HFR encode uses 32-lane
logical warps. On wave64 the odd-y rows occupy lanes 32..63, so the old
`(uint32_t)__ballot(pred)` truncation returned the wrong row's mask -- a latent bug
the old port shipped because no FZG/HFR GPU test ran. The half-base shift
`__lane_id() & ~31` selects the correct 32 bits: shift 0 or 32 on wave64, always
shift 0 on wave32 (upper 32 ballot bits are zero). On gfx1100 (wave32) this MUST
reduce to the identity case and the FZG/HFR codec results MUST match gfx90a.
This is the gate: the bin_hf HFR matrix and any FZG-exercising test must pass.

The other `(threadIdx.x & 31)` / `>> 5` sites found in codec/hf are NOT wave
geometry -- they are 32-bit-word bit-packing of the Huffman serialized format
(arch-independent word size), and the width-32 logical-warp `__shfl*(...,32)` /
`__shfl_up_sync(...,32)` sites operate within a 32-lane subgroup regardless of
physical wavefront. No hardcoded wave64 lane geometry exists in the built sources
(the `_future/` files are unbuilt). So the ballot macro is the sole wave32 risk;
everything else is arch-neutral.

### RDNA-specific watch items (low risk, but verify on hardware)

- Native double `atomicAdd` (utils/src/atomics.cu.inl HIP branch) is used by
  test_stat_max_error / extrema; confirm it is correct on RDNA3 (it is a hardware
  path on gfx1100, not the CAS fallback).
- No textures/surfaces, no cuBLAS/cuFFT/cuSPARSE, no layered arrays, no
  linear-filter fp32 texture -- so the CDNA-specific fault classes (256B pitch,
  layered-array collapse, gfx90a linear-texture rejection, `__fsqrt_rn` 1-ULP) do
  NOT apply here. gfx1100 also has a graphics engine (unlike gfx90a), but cuSZ is
  pure compute with no GL/Vulkan interop, so that is moot.

### Cross-arch consistency gate (reference-less determinism)

cuSZ compression is deterministic. Where a GPU test has no CPU reference, diff the
gfx1100 output against the gfx90a output for the SAME input rather than accepting
"deterministic + plausible" (PORTING_GUIDE wave-size cross-arch rule). Concretely:
the CLI round-trip CR/PSNR/max_error for a fixed input, and the bin_hf HFR codec
compression ratios, should match the gfx90a numbers recorded in notes
(1M f32 sine, abs eb 1e-3: CR=27.04, PSNR=70.8, max_error=1.000047e-03). A
divergence in the FZG/HFR path is exactly the wave32 ballot bug this gate is for.

### Build (gfx1100) -- identical to lead, arch swapped

```bash
cd /var/lib/jenkins/moat/projects/cuSZ/src
git checkout moat-port && git pull --ff-only   # ensure head = 07db1e28 re-port
rm -rf build-hip
cmake -S . -B build-hip \
  -DPSZ_BACKEND=HIP \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++
cmake --build build-hip -j$(nproc)
```

The gfx1100 host has 4 GPUs; pin one with HIP_VISIBLE_DEVICES.

### Test (gfx1100) -- the re-port ctest harness

```bash
cd /var/lib/jenkins/moat/projects/cuSZ/src/build-hip
HIP_VISIBLE_DEVICES=0 ctest --output-on-failure -j1
```

Expected (matching the gfx90a re-port result, notes 2026-06-25 @ 5d43e441):
28 PASS / 12 SKIP / 1 known-non-port FAIL out of 41.
- PASS must include the GPU set: test_l1_compact, test_stat_identical1/2,
  test_stat_max_error, test_mem_unique, test_hf_revisit_altcode (HFR), and the
  bin_hf codec matrix (hf/hfr/hfr-pbkc/hfr-pbkgo over cauchy-mild + uniform-256/1024,
  and cauchy-sharp except the one known-fail unit). The HFR/FZG GPU passes are the
  wave32-ballot proof.
- SKIP (rc=77): the cusz CLI round-trip matrix (cusz__rtm/hurr/nyx) -- needs
  $CUSZ_TEST_DATA datasets absent locally; documented skip, not a failure.
- Known non-port FAIL (must fail IDENTICALLY, do not count against the port):
  `hfr__cauchy_sharp__u2` -- `bin_phf: unknown option: --assert-brnum-le`, a
  pre-existing upstream test/CLI desync (the test cmake passes a flag bin_phf.cc
  never implemented); fails identically on the CUDA build, pure CLI arg error, no
  numeric/codec fault.
- `test_histsp_cu` is guarded off on HIP (upstream ctor-signature drift,
  tuning-only); not built. Same exclusion as the lead.

CLI round-trip + cross-arch consistency:

```bash
# IMPORTANT: .astype(np.float32) is REQUIRED. `arange(dtype=float32) * pythonfloat` promotes
# to float64, and a float64 .tofile() read back as -t f32 is garbage (the 2026-06-25 phantom
# "Lorenzo blocker": CR=0.43 was a corrupt INPUT, not a GPU bug -- notes "focused diagnostic").
python3 -c "import numpy as np; np.sin(np.arange(1000000,dtype=np.float32)*2*np.pi/100000).astype(np.float32).tofile('/tmp/test_sine_1m.f32')"
HIP_VISIBLE_DEVICES=0 ./cusz -z -i /tmp/test_sine_1m.f32 -t f32 -m abs -e 1e-3 -l 1000000 --report cr,psnr
HIP_VISIBLE_DEVICES=0 ./cusz -x -i /tmp/test_sine_1m.f32.cusza --compare /tmp/test_sine_1m.f32 --report cr,psnr
# expect CR=27.04, PSNR=70.8, max_error=1.000047e-03 (matches gfx90a; verified on gfx1100 @ 866868f6)
```

### Delta open questions

- If the bin_hf HFR/FZG matrix produces DIFFERENT compression ratios on gfx1100
  vs gfx90a, suspect the `__ballot_sync` half-base-shift macro on wave32 first
  (it should be the shift-0 identity case there). Re-derive against the actual
  `__lane_id()` range on wave32 (0..31) before assuming a deeper bug.
- HFR template-instantiation fatbin size: if a wave32 build TU approaches the
  x86-64 link-reach wall, add `--offload-compress` (it did not on gfx90a; verify).

## Delta plan: windows-gfx1201

Follower platform. RX 9070 XT (gfx1201, RDNA4, wave32), Windows 11, TheRock ROCm
7.14 nightly. Reuse the SAME moat-port branch and the SAME Strategy A single-source
port as the gfx90a lead; this is NOT a fresh plan or a re-port. The existing ROCm
port plan above applies as-is; only the build toolchain (Ninja + amdclang, PE DLL
runtime) and the arch flag differ. Do NOT re-plan from scratch.

### State of play (why this is a build+validate, not a re-plan)

The prior windows-gfx1201 PASS (notes 2026-06-08, validated_sha aff8ee6) was on the
OLD base (d3cde38..aff8ee6, the 74-file dual-source scaffold). Upstream then deleted
that scaffold and master moved on; moat-port was force-reset to the FRESH single-source
re-port (a6d765e8 -> 5d43e44 -> head 07db1e28). The aff8ee6 commit is gone from the
branch (force-update confirmed at fetch). So the prior gfx1201 record does NOT carry
forward: gfx1201 must validate the re-port on real hardware, exactly like gfx1100.

### THE wave32 gate -- gfx1201 shares gfx1100 blocker (read before building)

gfx1201 is wave32 (RDNA4). The re-port head 07db1e28 carries a hardcoded WAVE64
intra-warp scan that FAILED gfx1100 (the other wave32 arch) on 2026-06-25 at
psz/src/kernel/histsp.cu.inl:60-66 (the #ifdef __HIP_PLATFORM_AMD__ branch:
for d=1; d<64; d*=2  with  __shfl_up(sum, d, 64)  and  threadIdx.x % 64 >= d).

This is pre-existing upstream code (J. Tian 5cda177b, 2023), NOT a MOAT-port defect,
but it is on the DEFAULT compression path (GPU_histogram_Cauchy HistSp, feeding the
Huffman codebook). On wave32 the d>=32 shuffles clamp to the 32-lane physical
wavefront and the percent-64 writeback predicate treats a 0..31 lane index as a
64-wide group, so the per-warp histogram totals are wrong. The symptom is NOT a
crash: the 1M f32 sine round-trip on gfx1100 gave CR=0.43 / PSNR=-nan /
max_error=3.4e38 (corrupt) vs the gfx90a reference CR=27.04 / PSNR=70.8 /
max_error=1.000047e-03. gfx1201 (wave32) will reproduce this defect identically; it
builds clean but the default-path round-trip is corrupt.

Consequence for the follower flow: the lead-completed port has a KNOWN wave32 defect.
linux-gfx1100 already diagnosed it and bounced to the porter (notes 2026-06-25). The
fix (generalize the AMD scan to the physical wave width, e.g. drop the wave64
special-case and use the portable 32-lane __shfl_up / percent-32 scan, or gate on
__AMDGCN_WAVEFRONT_SIZE) is a SINGLE shared change on moat-port that re-validates
every wave32 arch. The gfx1201 source delta is therefore expected to be ZERO beyond
that shared fix: once the porter lands histsp and the head advances, gfx1201 builds
the fixed branch and validates. Until then, a gfx1201 validate-first WILL fail the
cross-arch CR gate (this is the documented expectation, not a surprise).

The Windows host MUST NOT make the histsp source fix itself (it is validation-only
per host policy, and the fix belongs to the shared branch driven from a Linux gfx1100
host). The gfx1201 job is: build + GPU-validate the branch once it carries the wave32
fix, and confirm the CR matches the gfx90a reference.

### Windows-gfx1201 build (TheRock venv, Ninja, all-clang) -- proven recipe

From the prior gfx1201 validation (notes 2026-06-08), the Windows toolchain delta vs
the Linux command is: Ninja generator, amdclang(.exe) for C/CXX/HIP, the TheRock
_rocm_sdk_devel tree as CMAKE_PREFIX_PATH, examples OFF, export-all-symbols ON, and a
post-configure sed to strip the -fuse-ld=lld-link flag CMake Windows-Clang module
injects. Re-confirm paths with hipInfo at session start (this host GPU index is not
stable; gfx1201 may be mask 0 or 1).

```bash
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
SRC=B:/develop/moat/projects/cuSZ/src
BUILD=$SRC/build-hip-gfx1201

# Check out the re-port head first (or the porter histsp-fixed head once it lands):
#   cd $SRC && git fetch origin moat-port && git checkout 07db1e28

cmake -S $SRC -B $BUILD -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPSZ_BACKEND=HIP -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/amdclang.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/amdclang++.exe \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/amdclang++.exe \
  -DCMAKE_PREFIX_PATH=$ROCM -DBUILD_TESTING=ON \
  -DPSZ_BUILD_EXAMPLES=OFF -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON

sed -i 's/ -fuse-ld=lld-link//g' $BUILD/build.ninja   # CMake Windows-Clang module fix

cmake --build $BUILD -j64
```

Windows source-fix watch items (the 8 _WIN32-guarded fixes from the OLD gfx1201 port,
notes 2026-06-08): the re-port is a different source tree, so those fixes do NOT carry
as commits. They are HINTS for what Windows/MSVC-STL may still trip in the re-port
(std::to_string needs string header, TypeSym ull dup on Windows, linux/limits.h and
cxxabi.h guards, popen->_popen, asprintf->snprintf, hipMallocHost void** casts, DLL
link-graph explicitness, -DHIP_DISABLE_WARP_SYNC_BUILTINS for the ROCm 7.14 bf16
shfl-sync redefinition). If the re-port already handles these (the lead porter may
have folded equivalents) the build is clean; if a Windows build error appears, these
are the likely classes -- a Windows-only _WIN32-guarded fix is a genuine delta-port
commit (behavior-preserving on Linux, carries forward there). cuSZ is pure CMake
(Strategy A), so the torch BuildExtension .hip Windows regression does NOT apply here.

### DLL runtime staging (required before ctest)

Native .exe loading on Windows: the loader prefers System32 Adrenalin amdhip64_7.dll
over PATH. Copy TheRock runtime DLLs INTO the test dir so the build runtime wins
(notes 2026-06-08): amdhip64_7.dll, amd_comgr.dll, hiprtc*.dll, rocm_kpack.dll from
_rocm_sdk_core/bin; hiprand.dll, rocrand.dll from _rocm_sdk_devel/bin; plus all
project DLLs (hipsz.dll, psz_hip_*.dll, fzg_hip.dll, phf_hip.dll) into $BUILD/test/.

### Test (gfx1201) -- the re-port ctest harness + cross-arch CR gate

```bash
HIP_VISIBLE_DEVICES=<gfx1201 mask> ctest --test-dir $BUILD --output-on-failure -j1
```

Expected once histsp is fixed (mirror the gfx1100 target, notes/plan gfx1100 delta):
28 PASS / 12 SKIP / 1 known-non-port FAIL of 41. PASS must include the GPU set
(test_l1_compact, test_stat_identical1/2, test_stat_max_error, test_mem_unique,
test_hf_revisit_altcode) and the bin_hf HFR/FZG codec matrix. The 12 SKIP are the
cusz CLI dataset tests (no CUSZ_TEST_DATA locally). The 1 known FAIL is the
pre-existing upstream hfr__cauchy_sharp__u2 CLI desync (--assert-brnum-le), which
fails identically on the CUDA build -- not a port regression. test_histsp_cu stays
excluded (tuning-only ctor drift).

THE GATE (cross-arch determinism, do not accept "round-trips + plausible"): the 1M
f32 sine CLI round-trip CR on gfx1201 MUST equal the gfx90a reference (CR=27.04,
PSNR=70.8, max_error=1.000047e-03). If CR diverges (esp. the CR=0.43 corrupt
signature), FIRST verify the INPUT is real float32, not float64-written-as-f32 (the
2026-06-25 gfx1100 phantom -- see notes "focused diagnostic"): the generator MUST end in
`.astype(np.float32).tofile(...)`. Only after a confirmed-valid input does a divergent CR
implicate the histsp wave32 fix not being in the built head (confirm the checkout is the
fixed branch @ 866868f6+, not 07db1e28). Generate the input with numpy
(np.sin(np.arange(1000000,dtype=np.float32)*2*np.pi/100000).astype(np.float32)) and run
cusz -z then cusz -x --compare with --report cr,psnr, same as the gfx1100 delta plan above.

### RDNA4 / Windows watch items (low risk)

- Native double atomicAdd (utils/src/atomics.cu.inl HIP branch) on gfx1201/RDNA4: a
  hardware path, not the CAS fallback; confirm test_stat_max_error / extrema pass.
- No cooperative-kernel launch in cuSZ (the gfx1201 cooperativeLaunch=0 host
  limitation does not apply -- cuSZ is plain grid launches).
- codeobj_diff.py is ELF-only; for any future gfx1201 binary-equiv carry-forward use
  the PE .hip_fat / .hipFatB section-hash method (memory codeobj-diff-windows-pe),
  not codeobj_diff.py directly.
- One-GPU-per-process: pin HIP_VISIBLE_DEVICES to exactly the gfx1201 mask; both GPUs
  in one process crashes the ROCm 7.14 HIP runtime on this host.

### Delta open questions

- Does the porter histsp wave32 fix land before gfx1201 is validated? If the head is
  still 07db1e28 (unfixed) when the validator picks this up, the validate-first WILL
  hit CR=0.43 and bounce to validation-failed -- correct and expected; the fix is a
  shared-branch dependency owned by the gfx1100-driven porter pass, not a Windows-host
  action. Coordinate: gfx1201 validates after the head advances.
- Do any of the 8 prior Windows _WIN32 fix classes resurface in the re-port source?
  If so, that is a legitimate gfx1201 delta commit (Linux-inert, carries forward).
