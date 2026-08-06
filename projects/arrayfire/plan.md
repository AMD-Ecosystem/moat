# Port plan: arrayfire

## Project
- Name: arrayfire
- Upstream: https://github.com/arrayfire/arrayfire
- Default branch: master
- Planning commit (depth-1 clone HEAD): 492718b ("Don't restart automatically after installing VC redistributable (#3685)")
- Version: ArrayFire 3.10.0
- License: BSD-3-Clause

## Existing AMD support -> decision: PROCEED (native HIP backend adds value)
ArrayFire is a multi-backend tensor/array library. `src/backend/` contains exactly four backends: `cpu`, `cuda`, `oneapi`, `opencl`. There is NO `hip` backend and no ROCm code anywhere (a repo-wide grep for hip/rocm/gfx/rocblas/hipblas matches only incidental substrings: "ownership", "ship", "graphics", a cuDNN-dll comment). Upstream has no HIP/ROCm branch and no open/merged HIP PR (`gh search prs ... HIP OR ROCm OR AMD` is empty). No `AMD-Ecosystem/arrayfire` fork exists yet.

AMD GPUs are reachable today ONLY through the OpenCL backend (and CLBlast/clFFT), which is a different programming model and is not ROCm/HIP. Per PORTING_GUIDE "AMD supported only via OpenCL/Vulkan/SYCL, with no HIP path -> a ROCm/HIP port of the CUDA code is still valuable", and reinforced by the AutoDock-GPU lesson that an OpenCL path can itself be broken on the current ROCm OpenCL runtime, a first-class HIP backend is worth adding. Decision: PROCEED with a native HIP backend cloned from the CUDA backend. This is NOT a skip.

Scope reality: this is a LARGE, DEEP port (the CUDA backend is ~42K LOC across 29 `.cu` + 56 `.cuh` + 78 `.cpp` + 198 `.hpp`, and is built on a runtime NVRTC JIT engine). It is bigger than any colmap-style Strategy-A port done in MOAT so far. The plan scopes a correctness-first, phased bring-up with an explicit validatable core, not a big-bang all-ops port.

## Build classification: pure CMake (Strategy A family) -- NOT a pytorch extension
Evidence:
- Top-level `CMakeLists.txt:17` `project(ArrayFire VERSION 3.10.0 LANGUAGES C CXX)`; backends are CMake options `AF_BUILD_CUDA`/`AF_BUILD_OPENCL`/`AF_BUILD_CPU`/`AF_BUILD_ONEAPI` (`CMakeLists.txt:93-97`).
- No `find_package(Torch)`, no `torch.utils.cpp_extension`, no `CUDAExtension`, no setup.py/pyproject torch dependency. The CUDA backend enables CUDA as a first-class CMake language and links the CUDA toolkit libs directly (`src/backend/cuda/CMakeLists.txt`).
So Strategy B (torch hipify) does not apply. This is the Strategy-A world, BUT with a major twist (below) that means the colmap "mark .cu LANGUAGE HIP + one compat header" recipe is necessary-but-far-from-sufficient.

## Port strategy: NEW `src/backend/hip` backend cloned from `src/backend/cuda`, NOT an in-place #ifdef of the CUDA backend
Rationale:
- ArrayFire's architecture is explicitly multi-backend: each backend is a self-contained directory compiled into its own shared library (`afcuda`, `afopencl`, `afcpu`, `afoneapi`) behind a common internal interface, with a `unified` dispatch layer (`src/api/unified`) selecting at runtime. Adding `afhip` as a sibling backend matches the project's own structure, keeps the NVIDIA/CUDA path byte-for-byte unchanged (zero regression risk to the existing backends, which is the dominant MOAT constraint), and is the additive shape the amgcl lesson recommends for mutually-exclusive backends. An in-place `#ifdef USE_HIP` rewrite of the CUDA backend would churn 42K LOC and risk the CUDA path.
- Concretely: `cp -r src/backend/cuda src/backend/hip`, then translate symbols and add an `AF_BUILD_HIP` option + a `src/backend/hip/CMakeLists.txt` deriving from the CUDA one (HIP language, hipBLAS/hipFFT/hipSPARSE/rocSOLVER/rocRAND/hipRTC libs). The bulk of the per-symbol translation (cuda*->hip*, cublas*->hipblas*, etc.) is mechanical and can be seeded by running torch's hipify mapping tables over the copied tree, but it is committed as real source in the new backend (this is the "native HIP backend", not a build-time hipify of CUDA sources).
- Backend identity: the public ABI enum has only `AF_BACKEND_{CPU,CUDA,OPENCL,ONEAPI}` (`include/af/defines.h:418-421`); `getBackend()` returns `AF_BACKEND_CUDA` (`platform.cpp:220`). The HIP backend reports `AF_BACKEND_CUDA` (it IS the CUDA-model backend, now on AMD) so no public ABI/enum change is needed and the unified dispatcher and all 139 tests treat it as the "cuda" backend. (Do NOT invent a new public enum value -- that is an ABI change and would ripple into the unified layer and tests.) The library can still build as `afhip` internally; the test harness's per-backend tag stays "cuda".

### THE defining risk: ArrayFire is a runtime-JIT engine on NVRTC + the CUDA Driver API
This is the single most important fact for planning. Most of ArrayFire's element-wise and many algorithmic kernels are NOT precompiled `.cu` device code. They are generated as CUDA-C++ SOURCE STRINGS at runtime and compiled on the fly:
- `src/backend/cuda/compile_module.cpp` calls `nvrtcCreateProgram`/`nvrtcCompileProgram` with `--gpu-architecture=compute_%d%d` (line ~265, from `getComputeCapability`), `nvrtcGetPTX`, then links via the CUDA Driver API: `cuLinkCreate` / `cuLinkAddData(CU_JIT_INPUT_PTX, ...)` / `cuLinkComplete` / `cuModuleLoadData` and runs kernels with `cuModuleGetFunction` + `cuLaunchKernel` (`Kernel.hpp:43`, `jit.cpp:522`).
- The runtime compiler `#include`s the project's OWN headers, embedded as C-string arrays at build time: `src/backend/cuda/CMakeLists.txt:148-238` lists `nvrtc_src` and a codegen step (`VARNAME nvrtc_files`, `OUTPUT_DIR nvrtc_kernel_headers`) that turns ~30 headers (types.hpp, math.hpp, interp.hpp, traits.hpp, shared.hpp, cuComplex.h, cuda_fp16.h, math_constants.h, vector_types.h, ...) into `nvrtc_kernel_headers/*.hpp` string blobs included by `compile_module.cpp:20-44`.
- 51 of the kernel `.hpp` files dispatch through `common::getKernel(...)` (the NVRTC path), plus the full element-wise JIT in `jit.cpp` (`getKernelString` builds the source).

Port consequence (NOT a mechanical name swap):
- hipRTC EXISTS (`/opt/rocm-7.2.1/include/hip/hiprtc.h`) and hipify maps `nvrtc.h`->`hip/hiprtc.h`, `nvrtc*`->`hiprtc*`, `cuLinkCreate`->`hipLinkCreate`, `cuModuleLoadData`->`hipModuleLoadData`, `CUfunction`->`hipFunction_t` (verified in `torch/utils/hipify/cuda_to_hip_mappings.py`).
- BUT the COMPILE/LINK FLOW differs: NVRTC emits PTX which is then assembled+linked by the driver `cuLink*` path; hipRTC compiles directly to a loadable code object (`hiprtcGetCode`/`hiprtcGetBitcode`) that goes straight into `hipModuleLoadData` -- there is no PTX and the `cuLink*` assemble step is unnecessary/absent. So `compile_module.cpp` must be RESTRUCTURED on HIP: replace `nvrtcGetPTX`+`cuLinkCreate`+`cuLinkAddData`+`cuLinkComplete`+`cuModuleLoadData` with `hiprtcGetCode`+`hipModuleLoadData`. The arch flag becomes `--offload-arch=gfx90a` (queried from device props, not `compute_XX`). This is bespoke per-project work, the bulk of the engineering effort, and the top schedule risk.
- The JIT SOURCE STRINGS themselves are CUDA-C++ that must compile under hipRTC: the embedded `cuComplex.h`, `cuda_fp16.h`, `math_constants.h`, `vector_types.h` headers and intrinsics inside the generated kernels must be HIP-valid. Provide HIP versions of those `nvrtc_kernel_headers` inputs (HIP ships `hip/hip_complex.h`, `hip/hip_fp16.h`, `hip/hip_vector_types.h`; supply a `math_constants.h` shim). Verify hipRTC accepts the project's `--device-as-default-execution-space` / `--std=c++17` / name-expression (`nvrtcAddNameExpression`/`nvrtcGetLoweredName` -> hiprtc equivalents) usage; name-expression support exists in hipRTC but confirm on ROCm 7.2.1 early (it gates templated-kernel instantiation by mangled name).

This JIT restructuring is the reason this port is multi-phase and why the plan does not promise a one-shot full-API port.

## Dependency dispositions (third-party + CUDA libraries)
ArrayFire VENDORS almost nothing (only `extern/half`); it finds host libs and CUDA libs externally. There are NO inter-project MOAT dependencies (it is not RAPIDS; it does not pull rmm/raft/cuCollections). So `set-deps` gets an EMPTY list.

CUDA library -> ROCm mapping (all standard 1:1 swaps per PORTING_GUIDE "Library swaps"):
- cuBLAS / cuBLASLt (`blas.cu`, `cublas.cpp`, `solve.cu`) -> hipBLAS (+ hipBLASLt for the Lt path). Watch hipBLAS v2 enum spellings.
- cuSOLVER dense (`cusolverDn.cpp`, `solve.cu`, `cholesky/lu/qr/svd/inverse`) -> hipSOLVER (rocSOLVER under it). Dense factorizations (getrf/potrf/geqrf/gesvd) are covered; ArrayFire uses only the DENSE cusolverDn API (no cusolverSp sparse-direct), so the RXMesh "cusolverSp has no hipSOLVER equivalent" gap does NOT apply here. Confirm gesdd/gesvdj coverage.
- cuFFT (`cufft.cu`, `fft.cu`, `fftconvolve`) -> hipFFT (rocFFT backend). Note the MPPI lesson: hipFFT status-enum coverage is not 1:1 with cuFFT -- map only the codes that exist and guard orphan `case` labels in any `cufftGetErrorString`-style switch; and hipFFT does not validate a garbage plan the way cuFFT does (robustness diff, not a bug).
- cuSPARSE GENERIC api (`sparse.cu`, `sparse_blas.cu`, `sparse_arith.cu`, `cusparse*.cpp`, `cusparse_descriptor_helpers.hpp`) -> hipSPARSE (per amgcl: hipSPARSE mirrors the cuSPARSE generic api name-for-name and keeps persistent descriptors; prefer it over rocSPARSE). Watch the amgcl hipSPARSE descriptor type-aliasing gotcha (several hipsparse descriptor types are all `typedef void*`, so a single overloaded deleter/visitor over distinct cuSPARSE types fails to compile).
- cuRAND -> rocRAND/hipRAND. ArrayFire's `random_engine.cu` implements its OWN Philox/Threefry/MT generators in device code (it does not lean on cuRAND host API heavily); the grep found no curand symbols in `random_engine.cu`. So this is mostly device-code translation, not a library swap -- validate rng_quality/rng_match closely.
- cuDNN (`cudnn.cpp`, `convolveNN`) -> MIOpen, but gated behind `AF_WITH_CUDNN` (default `${cuDNN_FOUND}`, `CMakeLists.txt:100`). Disposition: build the HIP backend with `-DAF_WITH_CUDNN=OFF` for the initial bring-up (NN convolution is one feature, not core array ops). A MIOpen-backed `convolveNN` is a deferred follow-up, not part of the first validatable core.
- Thrust / CUB (`sort*.cu`, `set.cu`, `scan*`, `ThrustAllocator.cuh`, `<cub/...>`, `<thrust/...>`) -> rocThrust / hipCUB (header drop-ins on hipcc's default `/opt/rocm/include` path per cudaKDTree; `thrust::cuda::par` -> `thrust::hip::par`, and the CUDA-backend `<thrust/system/cuda/...>` includes must swap to `<thrust/system/hip/...>` per rmm). The vendored `thrust::cuda::experimental::pinned_allocator` pattern (if present) -> `thrust::hip::universal_host_pinned_allocator` per cupoch. `ThrustArrayFirePolicy.hpp` is a custom execution policy -- re-target its `thrust::cuda_cub`/`thrust::cuda::par` base to the hip spellings (rmm namespace-delta lesson).
- NVRTC -> hipRTC (see JIT section -- the hard one).
- CUDA Driver API (`cuModule*`, `cuLaunchKernel`, `cuLink*`, `CUcontext`, `CUstream`, `cuMemcpy*Async`) -> HIP module/driver API (`hipModule*`, `hipModuleLaunchKernel`, `hipCtx*`, `hipStream_t`, `hipMemcpy*Async`). The `cuLink*` calls disappear (folded into the hipRTC flow).
- Host deps (spdlog, fmt, boost-compute-ish, FreeImage/Forge for imageio/graphics, gtest for tests) are backend-agnostic and unaffected; build the headless config (no Forge graphics) on the GPU host.

ROCm libraries are present under `/opt/rocm-7.2.1` (ROCm 7.2.1 confirmed).

## CUDA surface inventory
- Precompiled `.cu` device kernels (29): all/any/count/max/min/sum/mean/product (reductions), sort/sort_by_key/sort_index/set/topk (thrust/cub), blas/solve (cublas/cusolver), cufft/fft (cufft), sparse/sparse_arith/sparse_blas (cusparse), random_engine (own RNG), and CV kernels fast/harris/orb/sift/homography/nearest_neighbour/regions.
- JIT/NVRTC-compiled kernels (51 files via `common::getKernel`, plus element-wise `jit.cpp`): index/assign/lookup/approx/morph/meanshift/flood_fill/gradient/unwrap/wrap/resize/rotate/histogram/hsv_rgb/sobel/medfilt/bilateral/match_template/iir/select/range/iota/reorder/triangle/moments/convolve/fftconvolve/scan*/sparse... -- the long tail of array operations.
- Warp/wavefront intrinsics: `__shfl_up`/`__shfl_down` (25 uses), `__ballot`/`__ballot_sync`, `__all_sync`/`__any_sync`, `__popc`/`__popcll`. Chokepoint: `src/backend/cuda/kernel/shfl_intrinsics.hpp` already wraps all shuffles with `FULL_MASK = 0xffffffff` -- the AutoDock-GPU fault class (HIP `__shfl*_sync` static_assert a 64-bit mask). Other warp sites: `reduce_by_key.hpp` (`__shfl*`, hardcoded `% 32`, `/ 32`, `* 32`).
- Hardcoded warp size 32: `reduce.hpp` (`nwarps = THREADS_PER_BLOCK/32`, `warpid = tid/32`, `if (tid < 32)`, `cub::WarpReduce`), `reduce_by_key.hpp` (`threadIdx.x % 32 == 31`, `warpid = threadIdx.x/32`), `fast.hpp` (`cub::BlockReduce<unsigned, 32, BLOCK_REDUCE_WARP_REDUCTIONS, 8>`).
- Driver API: `cuLaunchKernel`, `cuModuleLoadData`/`GetFunction`/`GetGlobal`/`Unload`, `cuLink{Create,AddData,Complete,Destroy}`, `cuMemcpy{Hto D,Dto H,Dto D}Async`, `CUstream`/`CUevent`/`CUmodule`/`CUfunction`/`CUdeviceptr`/`CUcontext`.
- Streams/events: `CUevent` (8), `cuStreamWaitEvent`/`cuStreamSynchronize`; plus runtime-API streams. `Event.cpp`/`Event.hpp` wrap them.
- RAII handle wrapper: `src/backend/common/unique_handle.hpp` -- a generic `unique_handle<T>` with default `handle_(0)`, move-only, guarded `reset()` (already rule-of-five-correct; this is the colmap-class pattern done right). Used for cuFFT/cuSPARSE/cuSOLVER/cuDNN handles via a `DEFINE_HANDLER` macro -- mirror the macro instantiations to the hip* handle types.
- Textures/surfaces: ArrayFire uses `LookupTable1D.hpp` (a 1D texture/`tex1Dfetch` lookup) and some interp paths -- check for `cudaTextureObject_t` (the colmap/popsift texture fault classes: rule-of-five on the handle, no hardware linear filtering on float element-read textures -> manual lerp, 256B pitch for 2D). Limited surface use expected; enumerate precisely during porting.
- fp16: `cuda_fp16.h` / `__half` used in device + JIT; maps to `hip/hip_fp16.h` / `__half`. `extern/half` is the host-side half (Christopher Brumme's half), backend-agnostic.

## Risk list
1. (HIGHEST) NVRTC->hipRTC engine restructuring (`compile_module.cpp`): different compile/link model (PTX+cuLink vs direct code object), arch flag `compute_XX`->`--offload-arch=gfx90a`, name-expression API parity on ROCm 7.2.1, and the embedded JIT source-string headers (cuComplex/cuda_fp16/math_constants/vector_types) must compile under hipRTC. Mitigation: stand up a tiny standalone hipRTC repro FIRST (compile one templated kernel by name expression, load, launch) before touching the backend; this de-risks the whole port. If hipRTC name-expression or `--offload-arch` from a string-include kernel fails on 7.2.1, raise as a blocker early rather than after porting 51 files.
2. Wave64 warp size (gfx90a = 64). Fix host-side warp queries and device-side constants per PORTING_GUIDE: `shfl_intrinsics.hpp` `FULL_MASK` -> 64-bit `0xffffffffffffffffULL` on HIP (AutoDock-GPU). `reduce.hpp`/`reduce_by_key.hpp` hardcoded `/32`,`%32`,`<32`,`==31` and `cub::WarpReduce`/`BlockReduce<...,32,...>` -- decide per kernel whether to go native-64 (atomicAdd-combined reductions, AutoDock axis) or keep 32-lane sub-groups (positionally-packed rows, popsift axis). The `cub::WarpReduce` with `nwarps=THREADS/32` + `if(tid<32)` second stage is the classic wave64 hazard (hipCUB's WarpReduce is wave-width-aware, but the surrounding hand-written `/32` staging and `s_val[nwarps]` sizing are not).
3. CUB warp/block-reduction semantics on wave64: hipCUB `WarpReduce`/`BlockReduce` exist but the project hardcodes the logical warp width (`<unsigned,32,...>` in fast.hpp, `nwarps=THREADS/32` in reduce.hpp). Validate reduction CORRECTNESS + determinism (the MPPI warp-synchronous-reduction-races-on-wave64 lesson: any volatile-shared unrolled tail must be re-checked).
4. Library enum/robustness deltas: hipFFT error-enum non-1:1 (MPPI), hipSPARSE void* descriptor aliasing (amgcl), hipBLAS v2 enums. Mechanical but each can break a TU's compile.
5. rocThrust namespace/header deltas: `<thrust/system/cuda/...>` includes hard-fail on ROCm; swap to `<thrust/system/hip/...>` and alias `thrust::cuda::par`->`thrust::hip::par`, `thrust::cuda_cub`->`thrust::hip_rocprim` (rmm/cupoch). `ThrustArrayFirePolicy.hpp` custom policy is the focal point.
6. Textures: rule-of-five on `cudaTextureObject_t` (colmap), float-element-read linear-filter rejection (popsift/gpuRIR -> manual lerp), 256B 2D pitch (colmap, and CudaSift's "confirm it really is a Pitch2D bind") -- applies to LookupTable1D and any interp texture path.
7. Coarse-grained atomic min/max silently dropped on managed memory (cudaKDTree) and hipCUB DeviceRadixSort nonzero-begin_bit bug (cudaKDTree) -- relevant to sort/ireduce/topk/regions; check whether any sort uses a partial begin_bit and whether bounds use atomicMin/Max on managed allocations.
8. `__CUDA_ARCH__`-keyed host/device branches (cudaKDTree/gsplat/RXMesh): grep the copied backend for `#ifdef __CUDA_ARCH__` device-vs-host data selection and rewrite to `defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)`; this also affects the JIT source strings if they contain such guards.
9. Build-system breadth: the CUDA `CMakeLists.txt` is ~860 lines with static-vs-dynamic CUDA lib loading, cudnn dll shipping, separable compilation, and a codegen step for the nvrtc headers. The HIP CMakeLists must reproduce the nvrtc-header codegen (now feeding hipRTC), drop the static-CUDA-lib machinery, and link hip libs. Risk of over-scoping the CMake -- keep the first cut minimal (shared libs, no static, no cudnn, no graphics).
10. Scale/time risk: ~42K LOC + a bespoke JIT engine is far larger than prior MOAT ports. Mitigation: phase it and define a validatable CORE (below) rather than gating success on all 139 test binaries.

## Phasing and validatable core (breadth-scoping, per cupoch CUPOCH_CORE_ONLY lesson)
Phase 0: standalone hipRTC repro (de-risk #1). Phase 1: CMake `afhip` target + compat/symbol translation so the backend COMPILES and links headless with `-DAF_WITH_CUDNN=OFF -DAF_BUILD_FORGE=OFF`. Phase 2: get the NVRTC->hipRTC JIT engine running so element-wise + JIT ops execute (this lights up the long tail at once). Phase 3: the library-backed ops (blas/fft/solve/sparse via hip libs) and the `.cu` reductions/sort (wave64 fixes). Phase 4: CV kernels (fast/harris/orb/sift) and convolveNN/MIOpen as a deferred extension.
Validatable core target for first "completed": arithmetic/JIT (`jit`, `binary`, `math`, `arith` via assign/index/reduce), `reduce`/`scan`/`ireduce`, `sort`/`sort_by_key`/`set`, `blas`/`dot`/`matmul`, `fft`/`fft_real`, `random`/`rng_quality`, `transpose`/`reorder`/`join`/`moddims`, `index`/`assign`/`lookup`. Defer (documented): cudnn-gated convolveNN, graphics/imageio (Forge/FreeImage), nonfree sift/gloh if `AF_WITH_NONFREE=OFF`.

## File-by-file change list (initial; the new backend is a copy so most files are "translate symbols")
- NEW `src/backend/hip/` = `cp -r src/backend/cuda` then symbol-translate (cuda*->hip*, cublas*->hipblas*, cusolverDn->hipsolver, cusparse->hipsparse, cufft->hipfft, nvrtc->hiprtc, CU*->hip*). Seed with hipify mapping tables, then hand-fix the fault classes.
- `src/backend/hip/compile_module.cpp` (from cuda): RESTRUCTURE the NVRTC+cuLink flow to hipRTC+hipModuleLoadData; arch flag from device props as `--offload-arch=gfx90a`.
- `src/backend/hip/kernel/shfl_intrinsics.hpp`: `FULL_MASK` -> 64-bit on HIP; verify `__shfl_up/down`, `__ballot`, `__all/any` map (HIP provides these; masks 64-bit).
- `src/backend/hip/kernel/reduce.hpp`, `reduce_by_key.hpp`: wave64 warp-size fixes (the `/32`,`%32`,`<32`,`==31`, `nwarps`, `cub::WarpReduce`/`BlockReduce<...,32,...>` staging).
- `src/backend/hip/CMakeLists.txt` (from cuda): HIP language, hip-lib links, reproduce the `nvrtc_kernel_headers` codegen for hipRTC, configurable `CMAKE_HIP_ARCHITECTURES` (default gfx90a only when unset -- never a literal; CudaSift/Gpufit lesson), drop static-CUDA + cudnn + Forge for the first cut.
- `src/backend/hip/ThrustArrayFirePolicy.hpp`, `ThrustAllocator.cuh`: rocThrust namespace/policy spellings.
- The `nvrtc_kernel_headers` SOURCE inputs (`cuComplex.h`, `cuda_fp16.h`, `math_constants.h`, `vector_types.h` blobs): provide HIP-valid equivalents for the runtime compiler.
- Top-level `CMakeLists.txt`: add `option(AF_BUILD_HIP ...)`, `enable_language(HIP)` under it, `add_subdirectory(src/backend/hip)`, and (likely) a `find_package(hip ...)` + hipBLAS/hipFFT/hipSPARSE/rocSOLVER/rocRAND/hiprtc discovery. KEEP the CUDA path untouched.
- `src/api/unified/*`: should need NO change if `afhip` registers as the CUDA-identity backend; verify the unified loader's library-name lookup (it dlopen's `afcuda`/`afopencl`/...). May need to teach it to also find `afhip` (or build `afhip` AS `afcuda` on an AMD-only host). Decide during Phase 1.

## Build commands (gfx90a, headless)
```
cmake -S projects/arrayfire/src -B projects/arrayfire/src/build-hip \
  -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DAF_BUILD_HIP=ON -DAF_BUILD_CUDA=OFF \
  -DAF_BUILD_CPU=ON -DAF_BUILD_OPENCL=OFF -DAF_BUILD_ONEAPI=OFF \
  -DAF_BUILD_UNIFIED=ON -DAF_BUILD_EXAMPLES=OFF -DAF_BUILD_FORGE=OFF \
  -DAF_WITH_CUDNN=OFF -DAF_WITH_IMAGEIO=OFF -DAF_BUILD_DOCS=OFF \
  -DAF_BUILD_TESTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build projects/arrayfire/src/build-hip -j
```
(Keep CPU backend ON so the test suite has a trusted same-process reference backend to diff GPU results against. `-DCMAKE_HIP_ARCHITECTURES` stays a cache var so followers gfx1100/gfx1151 need only change the value -- no source edit.)

## Test plan
- Real test suite: `test/` has 139 gtest `.cpp` files, registered per-backend by `af_add_test(target backend is_serial)` (`test/CMakeLists.txt:71`) with `gtest_discover_tests ... TEST_PREFIX <UPPER_CASE backend>.`. The HIP backend runs under the `CUDA.*` test prefix (it reports `AF_BACKEND_CUDA`).
- Test DATA is fetched (not a submodule in this shallow clone): `af_dep_check_and_populate(af_test_data URI https://github.com/arrayfire/arrayfire-data.git REF 05703a4...)` (`test/CMakeLists.txt:100`). The build will clone it; ensure network access or pre-place it and set `-DAF_USE_RELATIVE_TEST_DIR=ON -DRELATIVE_TEST_DATA_DIR=<path>`.
- GPU run: `cd build-hip && ctest -R '^CUDA\.' --output-on-failure` (the HIP backend's tests). Run SERIALLY on a single assigned GPU (`ctest` not `ctest -jN`) -- the MPPI single-GPU contention lesson.
- Correctness bar: most ArrayFire tests already assert GPU-vs-reference numerics internally; cross-check against the CPU backend (built in the same tree) for any op where a golden value is missing. For RNG (`rng_quality`, `random_practrand`) validate the statistical/quality tests, not bitwise CUDA-equality (different generator hardware path). For reductions/scan assert run-to-run determinism on wave64 (MPPI/popsift).
- Non-GPU regression set that must not regress: the CPU backend tests (`CPU.*`) and the unified-API tests must still pass unchanged; the CUDA-on-NVIDIA path is not built here so "no regression" for it is structural (we add a sibling backend and do not touch `src/backend/cuda`).
- Core-first success: declare the lead platform validated when the Phase-3 validatable-core test set passes on gfx90a with no CPU-backend regression; document deferred test binaries (cudnn/graphics/nonfree) explicitly in notes.md.

## Open questions
1. hipRTC on ROCm 7.2.1: does `nvrtcAddNameExpression`/`nvrtcGetLoweredName` have full hiprtc parity (templated-kernel-by-name), and does `--offload-arch=gfx90a` work for a code object loaded via `hipModuleLoadData` from a string-include compile? (Resolve in Phase 0 repro; this is the make-or-break.)
2. Unified backend loader: does the `unified` layer dlopen a fixed library name (`libafcuda`)? If so, build `afhip` under that name on AMD, or extend the loader. Confirm in Phase 1.
3. hipBLASLt availability/maturity on ROCm 7.2.1 for the cuBLASLt path (else route through plain hipBLAS).
4. Scope of `cudaTextureObject_t` use (LookupTable1D + interp): enumerate exactly to know how much of the colmap/popsift texture fault-class surface applies.
5. Is the full port too large for one MOAT cycle? The phased core makes the FIRST validation tractable; the CV kernels + MIOpen convolveNN are an explicit follow-on. Flag to jeff if Phase 0 reveals hipRTC gaps that would force a from-scratch JIT-engine rewrite.

## Inter-project MOAT dependencies
NONE. ArrayFire vendors only `extern/half` and finds host/GPU libraries externally; it does not depend on any other MOAT project (not a RAPIDS lib). `set-deps` list is empty.
