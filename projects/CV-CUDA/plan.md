# CV-CUDA -- ROCm/HIP port plan (lead: linux-gfx90a)

## Project
- Name: CV-CUDA
- Upstream: https://github.com/CVCUDA/CV-CUDA
- Default branch: main
- Base SHA: ef50300b4a39bd13b4304e9a339b804e49709eab (v0.16.0)
- What it is: NVIDIA's open-source GPU computer-vision operator library. Two C++ layers:
  `src/nvcv` (NVCV core: tensor/image/batch types, allocators, DLPack interop -- 0 .cu, pure host C++)
  and `src/cvcuda` (the operators: a thin `Op*` dispatch layer over `priv/legacy/*.cu` kernel
  implementations + an `nvcv::cuda` device-side tools header library). ~70 .cu, ~39k LOC of kernels,
  8 .cuh, C++ gtest GPU test suite (one TestOp*.cpp per operator).

## Existing AMD support -> decision: PROCEED (none exists; CUDA-only)
- No HIP/ROCm/AMD path anywhere: no `__HIP_*` guards, no gfx references, no rocm branch/PR/fork
  upstream (remote branches are all version/feature branches). AMD is entirely unsupported.
- Not OpenCL/Vulkan/SYCL either -- it is pure CUDA. A correctness-first HIP port of the core
  operators adds clear value and is tractable (see CUDA surface below).
- This is NOT a perf-rewrite target in the CUTLASS/CuTe sense: there is NO CUTLASS, CuTe, wmma/mma,
  tensor-core, Hopper-wgmma, nvjpeg, NVENC, nvImageCodec, cuDNN, or NPP usage anywhere in src
  (greps all 0). The kernels are hand-written elementwise/stencil/reduction CV kernels. A mechanical
  HIP port is the right first step; no AMD-native rewrite is needed for correctness.

## Build classification -> pure CMake (Strategy A)
Evidence:
- Top `CMakeLists.txt:24-36`: `project(cvcuda LANGUAGES C CXX ...)` then `enable_language(CUDA)`.
- `cmake/ConfigCUDA.cmake:21`: `find_package(CUDAToolkit ... REQUIRED)`.
- No `find_package(Torch)`, no `torch.utils.cpp_extension`, no `CUDAExtension` -- the Python bindings
  (`python/`, pybind11) are an optional add-on (`BUILD_PYTHON=ON` by default) layered on the C++ libs,
  not a torch extension. Core builds standalone with `-DBUILD_PYTHON=OFF`.
- C++ standard is already 17 (`cmake/ConfigCompiler.cmake:16`), so the GPUMD rocPRIM-needs-C++17 trap
  does not apply.

## Port strategy: Strategy A, scoped to a validatable operator core
The kernels live as `.cu` in a handful of CMake targets. Plan:
1. Add `option(USE_HIP ...)`; under it `enable_language(HIP)` and gate `enable_language(CUDA)` +
   `find_package(CUDAToolkit)` (ConfigCUDA) behind `if(NOT USE_HIP)`. Set HIP arch from
   `${CMAKE_HIP_ARCHITECTURES}` defaulting to gfx90a ONLY when unset (per the configurable-arch
   lesson -- never a literal, so gfx1100/gfx1151 followers need no CMake edit).
2. Mark the project's `.cu` `LANGUAGE HIP` (the priv, priv/legacy, and the few cvcuda .cu targets).
   Several per-target CMakeLists set sources explicitly (not glob) so this is list-driven; if any
   subdir globs, use the MPPI add_executable/add_library override trick at top scope under USE_HIP.
3. One compat header (e.g. `src/cvcuda/include/cvcuda/cuda_tools/HipCompat.h`, force-included on the
   HIP target via `CMAKE_HIP_FLAGS -include`) handling the cross-cutting items below. Keep `#if
   defined(USE_HIP)` guards rare and localized.
4. nvcc-only flags `-Xfatbin=--compress-all` and `--extended-lambda` (ConfigCUDA.cmake:33,36) live in
   `CMAKE_CUDA_FLAGS`, which does NOT apply to HIP TUs -- so they are naturally inert under USE_HIP.
   hipcc accepts device lambdas natively; no flag needed. Keep `LTO_ENABLED` OFF (HIP+LTO breaks
   linking -- gpuRIR/Fast-Poisson lesson).

### Scope (correctness-first core; defer the rest, documented)
Build + GPU-validate the operators whose full dependency closure is portable. DEFER (scope out of the
gfx90a build via list FILTER under USE_HIP), with reason:
- OpOSD, OpBndBox, OpBoxBlur + `legacy/osd.cu`, `legacy/box_blur.cu`, `legacy/textbackend/`:
  depend on **cuOSD**, a prebuilt CUDA-only static lib (`3rdparty/cuOSD/{x86_64,aarch64}/lib/libcuosd.a`,
  header-only `cuosd.h`, NO SOURCE). Cannot be ported without cuOSD source. Only these 6 files
  reference it (`grep cuosd` = CvCudaLegacy.h, CvCudaOSD.hpp, box_blur.cu, osd.cu, textbackend/*),
  so excluding the OSD/BndBox/BoxBlur trio removes the dependency cleanly. ~45 operators remain.
- Re-include any deferred op later once its blocker is resolved; the scope-down mirrors cupoch
  CUPOCH_CORE_ONLY and STRUMPACK MPI=OFF.
This is the dominant scope decision. Everything else below is portable.

## CUDA surface inventory
- Kernels/operators: ~50 operators, kernels in `src/cvcuda/priv/legacy/*.cu` (69 files) + a few op-level
  `.cu` (OpResize, OpSIFT, OpLabel, OpMinMaxLoc, OpPairwiseMatcher, OpFindHomography, OpRemap,
  OpHQResize, OpColorTwist, OpBrightnessContrast, OpCropFlipNormalizeReformat, OpAdvCvtColor,
  OpNonMaximumSuppression, OpResizeCropConvertReformat). All `__global__`/`__device__`, no `.cu` in nvcv core.
- Device-tools header lib `src/cvcuda/include/cvcuda/cuda_tools/`: SaturateCast, RangeCast, MathWrappers,
  MathOps, Atomics, TensorWrap/BorderWrap/InterpolationWrap (the wrap/border/interp helpers most
  operators build on). These are `__host__ __device__` template headers; the key porting work is here.
- Warp intrinsics: `__shfl_*_sync` (12 down + up/xor), `__activemask` (5), `__syncwarp` (17),
  `warpSize` (25), `__popc` (1). Concentrated in 5 files: OpFindHomography.cu (29), legacy/threshold.cu
  (20), legacy/threshold_var_shape.cu (20), OpLabel.cu (15), legacy/reduce_kernel_utils.cuh (1).
- Textures/surfaces: NONE. No `cudaTextureObject_t`/`tex2D`/`surf*`/`cudaBindTexture`/`texture<` anywhere
  (the colmap/popsift/CudaSift texture, pitch, layered-array, and linear-filter fault classes DO NOT APPLY).
- Inline PTX `asm`: 2 sites only -- `priv/Assert.h:35` (`asm("trap;")`) and
  `cuda_tools/detail/SaturateCastImpl.hpp` (a table of `cvt.sat.*` saturating-conversion PTX). Both have
  portable handling -- see Risk list. No `asm volatile`.
- Libraries: cuBLAS + cuSOLVER (ONLY OpFindHomography.cu/.hpp), cuRAND (gaussian_noise* -- 6 files),
  CUB (`cub::BlockReduce`/`BlockScan`/`BlockRadixSort`/`DeviceReduce` in OpMinMaxLoc, OpPairwiseMatcher,
  threshold*, erase*, inpaint* -- 10 files). NO Thrust, cuFFT, cuSPARSE, NPP, nvjpeg.
- Driver API (`<cuda.h>` / `CU*`): `util/StreamId.cpp` + `util/Stream.cpp` (cuStreamGetId, cuInit,
  cuGetProcAddress, cuGetErrorString/Name, CUstream/CUresult) for per-stream id; `Compat.hpp` and
  `detail/Metaprogramming.hpp` include `<cuda.h>` only for the `CUDA_VERSION` macro.
- fp16: `cuda_fp16.h` include in reduce_kernel_utils.cuh; `float16` elsewhere is just an ImageFormat
  enum label, not `__half` device math. No bf16, no `__hadd`/`__hmul`.
- Memory: `cudaHostAlloc` (2), `cudaMalloc` (2 direct + allocator layer), NO cudaMallocManaged, NO
  cudaMallocAsync, NO IPC. Streams (`cudaStream_t` 934 -- pervasive, trivial 1:1) + events (12).
- `cudaArray` token appears 8x in nvcv ImageFormat descriptors (a memory-layout enum), NOT a texture
  cudaArray bind -- confirmed by the zero texture/surf count.

## Risk list (fault classes, with citations)
1. **SaturateCast PTX table is INERT on HIP for free, but only if `__CUDA_ARCH__` stays undefined.**
   `detail/SaturateCastImpl.hpp:102` gates the `cvt.sat.*` PTX on `#ifdef __CUDA_ARCH__`, with a complete
   portable C++ `#else` fallback (`BaseSaturateCastImpl`, lines 31-88). HIP does NOT define `__CUDA_ARCH__`
   in device compilation, so every specialization auto-takes the C++ fallback -- PTX never compiled.
   ACTION: do NOT define `__CUDA_ARCH__` for HIP (inverse of the cudaKDTree/gsplat trap). This is the single
   most important constraint -- defining `__CUDA_ARCH__` to satisfy some other header would pull in
   un-assemblable PTX. (NVCV math layer follows the same `#ifdef __CUDA_ARCH__` device-intrinsic / `#else`
   std:: pattern in MathWrappersImpl.hpp -- 8 sites, LinAlg.hpp 2, MathWrappers.hpp 3.)
2. **`__CUDA_ARCH__`-gated device math falls to std:: on the HIP device pass.** With (1), every
   `RoundImpl/MinImpl/MaxImpl/PowImpl/ExpImpl/SqrtImpl/AbsImpl` (`detail/MathWrappersImpl.hpp:420-508`)
   takes the `std::` host fallback on the HIP device pass -- compiles and is numerically correct (HIP
   provides device overloads of std:: scalar math), but loses the fast device intrinsics and risks the
   gsplat std::min/max-in-device-code class. PREFERRED FIX: extend each guard to
   `#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)` so HIP uses the device intrinsics
   (`__float2int_rn`, `rintf`, `fminf`, etc. all exist in HIP). Mechanical, ~13 guard sites; keeps CUDA
   byte-for-byte and matches CUDA rounding semantics. (cudaKDTree/gsplat/RXMesh __CUDA_ARCH__ family.)
3. **`__shfl_*_sync` mask literal `0xffffffff` fails to COMPILE on HIP.** ROCm 7.x static_asserts the mask is
   64-bit. Sites: reduce_kernel_utils.cuh `FINAL_MASK 0xffffffff`, OpLabel.cu (8), threshold.cu/threshold_var_shape.cu,
   OpFindHomography `mask`. FIX: a USE_HIP-keyed full mask `0xffffffffffffffffULL` (NOT width-keyed), CUDA keeps
   the 32-bit literal. Surfaces in hipcc's HOST pass of the .cu. (AutoDock-GPU lesson.)
4. **wave64 warp-reduction correctness (split by idiom -- pick per site).**
   - `legacy/reduce_kernel_utils.cuh` warpReduceSum uses `__shfl_xor_sync(FINAL_MASK, val, mask, 32)` with
     EXPLICIT width=32, and stages partials in `shared[32]` indexed by `threadIdx.x>>5`. Explicit-width-32
     butterflies stay within a 32-lane subgroup on wave64 (gsplat tiled_partition<32> lesson), and the block
     reduce treats the block as 32-lane groups throughout -- so this is wave64-TOLERANT once the mask (3) is
     fixed. Validate, do not reflexively rewrite.
   - `OpFindHomography.cu` warp reductions use `warpSize` (runtime, wave-correct) and `warpSize/2` strides,
     with `warpSums[]` shared sized by `blockDim.x/warpSize`. Runtime warpSize makes the reduction itself
     wave-size-agnostic; verify the shared-array sizing uses a 64-safe upper bound, fix mask per (3).
   - `OpLabel.cu` (connected-components) uses `__shfl_up_sync(mask, val, 1)` WITHOUT explicit width -- a
     delta-1 shuffle across the full wavefront. On wave64 the lane-32 boundary semantics differ from a
     32-lane warp (popsift two-32-rows class). HIGHEST-RISK operator: needs per-arch handling (operate on
     32-lane subgroups, or account for the 64-lane row boundary). threshold_var_shape.cu
     `__shfl_xor_sync(0xffffffff, temp, 1/2/4)` (no width) similarly mixes the two 32-lane halves on wave64 --
     add width=32 or wave64-aware reduction. (popsift / RXMesh wave64 ballot/shuffle lessons.)
5. **int atomicMin/atomicMax on gfx90a.** `cuda_tools/Atomics.hpp:84,105` calls integer `atomicMin/atomicMax`
   directly (float path uses an atomicCAS loop already). Per cudaKDTree, int atomicMin/Max are SILENTLY
   DROPPED on coarse-grained/managed memory on gfx90a. CV-CUDA buffers are `hipMalloc` device memory (no
   managed), so likely fine -- but FLAG for MinMaxLoc/Label/threshold and verify with the op's own GPU test;
   if any min/max-accumulating op produces sentinel/empty results, swap to an atomicCAS loop (unsigned cmp
   when sentinel is (uint)-1).
6. **Driver-API per-stream id (`util/StreamId.cpp`).** Uses cuStreamGetId/cuInit/cuGetProcAddress to derive a
   stream identity for per-stream resource caching. The file ALREADY ships a portable fallback
   (`cuStreamGetIdFallback`, lines 57-75: use the stream-handle pointer value). FIX: on HIP take the
   fallback (`id = (uint64_t)stream`) -- correctness-first, the only requirement is a stable per-stream key.
   Alternatively map to `hipStreamGetId` if present in ROCm 7.2.1. Small contained host change. `cuInit`->
   `hipInit`, `CUstream`->`hipStream_t`, `cuGetErrorString/Name`->`hipDrvGetErrorString`/hip equivalents.
7. **`<cuda.h>`/`<vector_types.h>` driver/toolkit includes absent on ROCm.** Compat.hpp, Metaprogramming.hpp,
   StreamId.cpp, Stream.cpp. Retarget to the HIP runtime in the compat header / forwarding shims (MPPI
   hip_compat shim-dir pattern -- per-target BEFORE include dir on the HIP build only, so CUDA path is
   byte-identical). `<cuda.h>` is used in Compat/Metaprogramming ONLY for the `CUDA_VERSION` macro -- define
   a HIP-equivalent guard.
8. **`asm("trap;")` (priv/Assert.h:35).** PTX trap. FIX: on HIP use `__builtin_trap()`/`abort()` (USE_HIP
   guard). One site.
9. **cuBLAS/cuSOLVER swap (OpFindHomography only).** cuBLAS->hipBLAS, cuSOLVER->hipSOLVER. APIs are largely
   1:1 (watch hipBLAS v2 enums and the hipSOLVER handle/workspace signatures). Confined to ONE operator; if
   hipSOLVER coverage is incomplete for the routines used, OpFindHomography can be deferred like the OSD trio.
10. **cuRAND swap (gaussian_noise).** curand_kernel.h device API -> hiprand_kernel.h. The device generator
    (curandState etc.) maps 1:1 in hipRAND; verify the specific distribution calls exist.
11. **CUB -> hipCUB.** `#include <cub/cub.cuh>` / `"cub/cub.cuh"` -> hipCUB headers; `cub::` -> `hipcub::`
    (compat alias `#define cub hipcub` or include path). BlockReduce/BlockScan/BlockRadixSort/DeviceReduce
    are all provided 1:1 (cudaKDTree: hipCUB is a header drop-in on /opt/rocm/include). Watch
    DeviceRadixSort begin_bit!=0 (broken on ROCm) -- PairwiseMatcher uses BlockRadixSort (sorts full key),
    so likely unaffected; verify.
12. **fp16 include.** `cuda_fp16.h` -> `<hip/hip_fp16.h>` (compat header / hipify mapping). Only an include in
    reduce_kernel_utils.cuh; no `__half` device arithmetic to port.
13. **pybind11 + HIP (only if BUILD_PYTHON=ON).** Defer Python for lead validation (`-DBUILD_PYTHON=OFF`);
    the C++ gtest suite is the gate. If Python is enabled later, apply the pybind11 NO_EXTRAS / disable-LTO
    fix (Fast-Poisson/gpuRIR) -- pybind11_add_module injects -flto that strips PyInit_* under HIP.
14. **No textures, no managed memory, no cooperative groups, no cuda::std, no CUTLASS** -- explicitly
    confirmed absent; the corresponding fault classes do not apply.

## File-by-file change list (lead, scoped core)
Build (CMake), additive/guarded -- NVIDIA path byte-for-byte unchanged:
- `CMakeLists.txt`: add `option(USE_HIP ...)`; gate `enable_language(CUDA)` (line 36) under `if(NOT USE_HIP)`;
  under USE_HIP `enable_language(HIP)` + set HIP arch from `${CMAKE_HIP_ARCHITECTURES}` (default gfx90a when
  unset).
- `cmake/ConfigCUDA.cmake`: wrap `find_package(CUDAToolkit REQUIRED)` + nvcc arch list under `if(NOT USE_HIP)`;
  add a USE_HIP branch (find_package(hip), hipBLAS/hipSOLVER/hipRAND/hipCUB as needed by the kept ops).
- `src/cvcuda/priv/CMakeLists.txt`, `src/cvcuda/priv/legacy/CMakeLists.txt`, and any cvcuda .cu target:
  under USE_HIP, `set_source_files_properties(<.cu> PROPERTIES LANGUAGE HIP)` + HIP_ARCHITECTURES; list
  FILTER-EXCLUDE the OSD/BndBox/BoxBlur sources + textbackend + osd.cu/box_blur.cu (cuOSD scope-out).
- Force-include the compat header on the HIP target: `CMAKE_HIP_FLAGS -include .../HipCompat.h`.
Source (guarded by USE_HIP / __HIP_DEVICE_COMPILE__):
- NEW `src/cvcuda/include/cvcuda/cuda_tools/HipCompat.h`: include order (cstring/cstdlib BEFORE
  hip_runtime -- gpuRIR), HIP runtime, full-warp 64-bit mask macro, fp16/cuda.h retargets, cuBLAS/cuSOLVER/
  cuRAND/cub name maps for the kept ops. MUST NOT define `__CUDA_ARCH__`.
- `cuda_tools/detail/MathWrappersImpl.hpp` (+ LinAlg.hpp, MathWrappers.hpp): extend `#ifdef __CUDA_ARCH__`
  -> `#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)` (Risk 2).
- `cuda_tools/Compat.hpp`, `cuda_tools/detail/Metaprogramming.hpp`: retarget `<cuda.h>`/`<vector_types.h>`
  + CUDA_VERSION guard for HIP.
- `priv/Assert.h`: `asm("trap;")` -> `__builtin_trap()` on HIP.
- `priv/legacy/reduce_kernel_utils.cuh`: FINAL_MASK 64-bit on HIP; fp16 include.
- `OpLabel.cu`, `legacy/threshold.cu`, `legacy/threshold_var_shape.cu`, `OpFindHomography.cu`: mask literals
  (Risk 3) + wave64 shuffle handling (Risk 4) -- OpLabel and threshold_var_shape need the per-arch subgroup fix.
- `util/StreamId.cpp`, `util/Stream.cpp`: HIP driver-API path / fallback stream-id (Risk 6).
- `OpFindHomography.{cu,hpp}`: cuBLAS/cuSOLVER -> hipBLAS/hipSOLVER (Risk 9).
- `legacy/gaussian_noise*.cu`, `gaussian_noise_util.cuh`: cuRAND -> hipRAND (Risk 10).
- CUB consumers (OpMinMaxLoc, OpPairwiseMatcher, threshold*, erase*, inpaint*): cub -> hipcub (Risk 11).
(Exact lists to be confirmed by the porter against the per-target CMakeLists; this is the planned surface.)

## Build commands (gfx90a)
Out-of-source, core only (no Python), C++ tests on:
```
cmake -S projects/CV-CUDA/src -B projects/CV-CUDA/src/build-hip -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DBUILD_PYTHON=OFF -DBUILD_TESTS=ON -DBUILD_TESTS_CPP=ON -DBUILD_TESTS_PYTHON=OFF \
  -DBUILD_BENCH=OFF -DBUILD_DOCS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build projects/CV-CUDA/src/build-hip -j$(nproc)
```
(Same command builds gfx1100/gfx1151 with only `-DCMAKE_HIP_ARCHITECTURES=<arch>` -- no source/CMake edit,
because the target reads ${CMAKE_HIP_ARCHITECTURES}.)
Optional CPU-only compile smoketest: `rocm/dev-ubuntu-24.04:7.2.4-complete` (compile check ONLY, never the gate).

## Test plan
Real GPU correctness tests (the validation gate). gtest exes built by `tests/`:
- `tests/cvcuda/system` -> the cvcuda_test_system gtest exe: one TestOp*.cpp per operator, each runs the op
  on GPU and compares to a CPU reference (72 files). PRIMARY gate for the operators.
- `tests/nvcv_types/system` (27) + `tests/nvcv_types/cudatools_system` + `tests/nvcv_types/cudatools_unit`:
  NVCV core + the cuda_tools (SaturateCast/MathWrappers/wraps) device-side tests -- directly exercise the
  Risk 1/2/4 math paths on GPU.
- `tests/nvcv_types/unit` + `tests/cvcuda/unit`: host-side unit tests (must not regress).
Run via ctest in the build dir; run SERIALLY on a single assigned GPU (`ctest --output-on-failure` without
-jN -- MPPI single-GPU contention lesson). Gate filter: run the cvcuda/system + nvcv cudatools/system suites,
EXCLUDING the deferred operators (OpOSD, OpBndBox, OpBoxBlur via gtest --gtest_filter=-, and any deferred
OpFindHomography if hipSOLVER is short). Validation passes when the kept-operator GPU tests + nvcv core
tests pass on gfx90a with no regression in the host unit tests.
Per-operator correctness expectations: CV ops are not chaotic -- expect bitwise or tight-ULP agreement with
the CPU reference (the suite already encodes per-op tolerances). Pay special attention to OpLabel,
OpThreshold (var_shape), OpMinMaxLoc, OpFindHomography on AMD given Risks 4/5/9. Non-GPU regression set:
the unit suites above must stay green.

## Inter-project deps
NONE. CV-CUDA's submodules (pybind11, googletest, dlpack, nvbench) and 3rdparty (cuOSD -- scoped out) are
self-contained; it does not build on any other MOAT project (not a RAPIDS lib, no rmm/raft). No `set-deps`
needed. (Will leave depends_on empty.)

## Open questions
- hipSOLVER coverage for the exact routines OpFindHomography uses (SVD/least-squares/geqrf-class): if a used
  routine is missing/unstable on ROCm 7.2.1, defer OpFindHomography (like the OSD trio) for the lead pass and
  note it; the homography solve is one operator, not core.
- ROCm 7.2.1 `hipStreamGetId` availability -- if absent, use the pointer-value fallback (already in-file).
  Functionally immaterial (only a per-stream cache key).
- Confirm whether any kept operator's CPU reference test itself pulls cuOSD/Python (TestOpBndBox_Smoke /
  TestOpOSD live in the same dir) -- exclude those test sources alongside the deferred ops.
- Exact per-target .cu list per subdir CMakeLists (priv, legacy, cvcuda) to retag LANGUAGE HIP -- porter to
  enumerate; if any subdir globs *.cu, use the MPPI add_executable/add_library top-scope override.
