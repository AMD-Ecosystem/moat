# Gpufit ROCm/HIP Port Plan

## Project

[gpufit/Gpufit](https://github.com/gpufit/Gpufit) -- GPU-accelerated Levenberg-Marquardt
curve fitting in CUDA. A C++/CUDA shared library (`Gpufit`) that fits large batches of
independent nonlinear models (1D/2D Gaussians, Cauchy, splines, etc.) in parallel, one
fit per thread block. Ships Cpufit (CPU reference), Boost.Test known-answer tests, C++
examples, and MATLAB/Python/Java bindings.

Pinned upstream commit: `a0bd66c` (master, shallow clone).

## Existing AMD support

None. No `hip`/`rocm`/`__GFX`/`amdgpu`/`wavefront` tokens anywhere in the tree. AMD is
not supported via OpenCL/Vulkan/SYCL either. Disposition: fresh CUDA-to-HIP port is
valuable.

## Build classification + strategy

Pure CMake project, no PyTorch/Torch dependency -> **Strategy A** (one
compat header, `.cu` marked `LANGUAGE HIP`, minimal guarded fixes, CUDA path untouched).

One wrinkle vs. the CudaSift/popsift templates: Gpufit uses the **legacy `FindCUDA`
module** (`find_package(CUDA REQUIRED)`, `cuda_add_library`, `cuda_add_executable`,
`CUDA_SELECT_NVCC_ARCH_FLAGS`, `CUDA_CHECK`, `${CUDA_LIBRARIES}`), not modern
`enable_language(CUDA)`. The legacy machinery cannot drive HIP. So the `USE_HIP` branch
provides a parallel **modern** path (`enable_language(HIP)` + `add_library`/
`add_executable` + `LANGUAGE HIP` + `HIP_ARCHITECTURES`), while the existing CUDA branch
is left byte-for-byte. Gating is done with `if(USE_HIP) ... else() <original> endif()`.

### Solver path selection: GJ (USE_CUBLAS=OFF), the upstream default

Gpufit has two interchangeable linear-solver backends for the per-fit normal equations,
selected by `USE_CUBLAS`:
- `USE_CUBLAS=ON`: cuBLAS batched LU (`cublasXgetrfBatched` / `cublasXgetrsBatched`).
- `USE_CUBLAS=OFF` (**upstream default**, `DEFAULT_USE_CUBLAS OFF`): a self-contained
  Gauss-Jordan elimination kernel (`cuda_gaussjordan.cu`) with no external library.

This port targets the **default GJ path**. It is fully self-contained device code, needs
no library swap, and -- decisively for the wave64 fault class -- contains no warp
intrinsics (see inventory). hipBLAS is installed on this host, so a `USE_CUBLAS=ON` HIP
build is a possible follow-up (cuBLAS->hipBLAS is a near-1:1 symbol swap), but it is out
of scope for the correctness-first port and not the upstream default.

## CUDA surface inventory

Device TUs (5 `.cu` + `.cuh`/model/estimator headers, all under `Gpufit/`):
`lm_fit_cuda.cu`, `cuda_kernels.cu`, `cuda_gaussjordan.cu`, `gpu_data.cu`, `info.cu`,
plus `examples/c++/CUDA_Interface_Example.cu`.

- **Kernels**: curve-value eval, chi-square / gradient / hessian reduction, parameter
  update, box projection (`cuda_kernels.cu`); Gauss-Jordan solver (`cuda_gaussjordan.cu`).
  All reductions are **block + `__shared__` + `__syncthreads()`/`__threadfence()`** tree
  reductions over the parameter/point dimension. No warp-level primitives.
- **Warp intrinsics**: NONE. No `__shfl*`, `__ballot`, `__any`, `__all`, `__activemask`,
  `__syncwarp`. grep confirms zero hits.
- **Hardcoded 32 / lane masks**: NONE. The only `warpSize` reference is `info.cu:20`
  `warp_size_ = devProp.warpSize;` -- a **runtime query** (correct pattern; yields 64 on
  gfx90a). Its sole consumer is `lm_fit_cuda.cu:306`,
  `while ((n_hessians_per_block + 1) * n_unique_values < info_.warp_size_)`, a block-
  packing heuristic that simply packs more hessians per block on a wider wavefront. No
  32-lane reduction assumption -> **no popsift-class wave64 bug here.**
- **Textures / surfaces / cudaArray**: NONE.
- **Constant memory (`__constant__`)**: NONE. Only `__shared__` (dynamic extern) +
  `__device__`/`__global__`.
- **Libraries**: cuBLAS only, and only on the non-default `USE_CUBLAS=ON` path. The
  default GJ path links no GPU math library. No cuSPARSE/cuRAND/cuFFT/CUB/Thrust.
- **Runtime API used (host)**: `cudaMalloc/Free/Memcpy/Memset/MemGetInfo`,
  `cudaGetDeviceCount/Properties`, `cudaDeviceProp`, `cudaGetLastError/ErrorString`,
  `cudaDriverGetVersion/RuntimeGetVersion`, memcpy-kind enums, `cudaError_t/cudaSuccess`.
  All have 1:1 HIP spellings.
- **Direct CUDA-header includes**: `<cuda_runtime.h>` (info.cu, gpu_data.cu/.cuh,
  CUDA_Interface_Example.cu) and `<device_launch_parameters.h>` (gpu_data.cu,
  cuda_kernels.cuh, cuda_gaussjordan.cuh). Need HIP-only shims so these resolve on ROCm.

## Risk list

1. **Build system (primary risk).** Legacy `FindCUDA` (`cuda_add_library`,
   `CUDA_SELECT_NVCC_ARCH_FLAGS`, `cuda_add_executable`, `CUDA_CHECK`, `CUDA_VERSION`,
   `${CUDA_LIBRARIES}`) is woven through `Gpufit/CMakeLists.txt`, the top CMakeLists, and
   the examples. Each site must get a `USE_HIP` modern-CMake alternative without
   disturbing the CUDA path. Mitigation: gate with `if(USE_HIP)/else()/endif()`; mark the
   existing `.cu` `LANGUAGE HIP`; force-include the compat header.
2. **Header shims.** `<cuda_runtime.h>` and `<device_launch_parameters.h>` don't exist on
   ROCm. Mitigation: `hip_compat/` shim dir (as in CudaSift) added to the include path on
   the HIP build only; both shims redirect to the compat header / HIP runtime.
3. **Wave64 correctness.** LOW. No warp intrinsics; GJ reductions are `__syncthreads()`-
   based and wave-size-agnostic; `warp_size_` is a runtime value feeding only a packing
   heuristic. Validation on gfx90a (wave64) is the proof.
4. **FP strictness / `abs` in device code.** `cuda_gaussjordan.cu` calls `abs()` on `REAL`
   (float/double). Resolves to the correct overload under the HIP device math headers;
   watch for an unexpected integer `abs`. Low risk; the known-answer tests would catch a
   wrong pivot.
5. **`__threadfence()` semantics.** Used in the GJ pivot search. 1:1 in HIP. The reduction
   relies on it plus `__syncthreads()`; behavior is equivalent on AMD.

## File-by-file change list (all HIP-guarded; CUDA path unchanged)

New files (HIP build only):
- `Gpufit/cuda_to_hip.h` -- the single compat header. On `USE_HIP`/`__HIP_PLATFORM_AMD__`
  it includes `<hip/hip_runtime.h>` and `#define`s exactly the cuda* symbols the project
  uses (the inventory list above) to their hip* equivalents; else it's a no-op including
  `<cuda_runtime.h>`. Force-included into every HIP TU.
- `Gpufit/hip_compat/cuda_runtime.h` -- `#pragma once` + `#include "cuda_to_hip.h"` so
  direct `<cuda_runtime.h>` includes resolve on ROCm.
- `Gpufit/hip_compat/device_launch_parameters.h` -- `#pragma once` (+ HIP runtime via the
  compat header); HIP provides `threadIdx`/`blockIdx`/etc. without a separate header.

Edited files:
- `Gpufit/CMakeLists.txt` -- add `option(USE_HIP ...)`. Wrap the `find_package(CUDA)` +
  arch-flag logic and `cuda_add_library(Gpufit ...)` so that under `USE_HIP`:
  `enable_language(HIP)`, `add_library(Gpufit SHARED ${all sources})`,
  `set_source_files_properties(${GpuCudaSources} PROPERTIES LANGUAGE HIP)`,
  `HIP_ARCHITECTURES gfx90a`, `target_compile_definitions(... USE_HIP)`, force-include
  `cuda_to_hip.h`, add `hip_compat/` to the include path. CUDA branch unchanged. Also
  guard the `USE_CUBLAS`/cuBLAS link block to the CUDA path (HIP default = GJ, no cuBLAS).
- top-level `CMakeLists.txt` -- propagate `USE_HIP` (declare the option early so
  subdirectories see it). The Boost-test and example helper functions are CUDA-agnostic;
  the only CUDA coupling is via the subdirectory targets, handled in the module file.
- `examples/c++/CMakeLists.txt` -- guard `add_cuda_example(... CUDA_Interface_Example)`:
  on `USE_HIP`, build it as a HIP TU (`add_executable` + `LANGUAGE HIP` + force-include +
  arch); else keep `cuda_add_executable`. The 3 pure-`.cpp` examples are unchanged (they
  link `Gpufit` and need no device toolchain).

No source-logic edits are anticipated for the GJ path. Any genuinely divergent spot
discovered during the build gets a narrow `#if defined(USE_HIP)` guard.

## Build commands

```
utils/timeit.sh Gpufit compile -- \
  cmake -S projects/Gpufit/src -B projects/Gpufit/src/build-hip \
    -DUSE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx90a \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUBLAS=OFF

utils/timeit.sh Gpufit compile -- \
  cmake --build projects/Gpufit/src/build-hip -j
```

`build-hip/` added to `.git/info/exclude`.

## Test plan (validation on idle gfx90a, MI250X, wave64)

Pick an idle GPU via `rocm-smi`; run with `HIP_VISIBLE_DEVICES=N`.

1. Build the `Gpufit` shared lib + Boost.Test known-answer suites + examples.
2. Run the Boost.Test executables (built into `build-hip/`):
   `Gpufit_Test_Gauss_Fit_1D`, `..._Gauss_Fit_2D`, `..._Gauss_Fit_2D_Elliptic`,
   `..._Gauss_Fit_2D_Rotated`, `..._Cauchy_Fit_2D_Elliptic`, `..._Linear_Fit_1D`,
   `..._Brown_Dennis_Fit`, `..._Fletcher_Powell_Helix_Fit`, `..._Error_Handling`.
   Pass = each Boost suite reports "No errors detected" and exit 0. These assert fitted
   params recover the true params (e.g. Gauss_1D: `{4,2,0.5,1}` to `1e-6`) with
   `chi_square < 1e-6` -- i.e. the LM solver CONVERGES to the correct answer on AMD, not
   NaN/divergent.
3. Run an example (e.g. `Simple_Example`) as an end-to-end smoke check.
4. Determinism: re-run a couple of suites; results must be stable.

Pass criteria: clean HIP build + the full known-answer suite passes on gfx90a +
deterministic. Then leave linux-gfx90a in `ported`.
