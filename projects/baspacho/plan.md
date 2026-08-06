# BaSpaCho Port Plan (linux-gfx90a)

## Project

- **Name**: baspacho
- **Upstream**: https://github.com/facebookresearch/baspacho
- **Default branch**: main
- **Description**: Batched Sparse Cholesky -- a state-of-the-art direct solver for symmetric positive-definite sparse matrices with CUDA GPU support for Levenberg-Marquardt optimization (used by Theseus).

## Existing AMD support

**None found.** Searched:
- Upstream docs: `grep -rniE 'amd|rocm|hip|gfx[0-9]'` returned only references to "Approximate Minimum Degree" (AMD), a sparse matrix reordering algorithm from SuiteSparse, not AMD GPUs.
- Web search: "baspacho ROCm", "baspacho HIP AMD GPU", "baspacho MI300/gfx9" returned no relevant results.
- GitHub forks: `gh api repos/facebookresearch/baspacho/forks` -- 8 forks, none with rocm/hip/amd branches.
- Upstream PRs: No ROCm/HIP PRs. There are open PRs for OpenCL (#7) and Metal (#6) backends, but no HIP backend.
- Upstream branches: No rocm/hip/amd branches.

**Decision**: Proceed with a from-scratch HIP port. No existing AMD effort to build upon.

## Build classification

**Pure CMake** (Strategy A)

Evidence:
- `CMakeLists.txt` line 11: `project(BaSpaCho CXX)` -- standalone CMake project
- `CMakeLists.txt` line 41: `enable_language(CUDA)` -- native CMake CUDA support
- `CMakeLists.txt` line 36-37: `BASPACHO_USE_CUBLAS` option controls CUDA enablement
- No `find_package(Torch)`, no `torch.utils.cpp_extension`, no `CUDAExtension`, no setup.py
- Uses FetchContent for dependencies (Eigen, dispenso, gtest, Sophus)

## Port strategy

**Strategy A: pure CMake, compat-header model**

Rationale:
1. Standalone CMake project with native CUDA language support
2. Small CUDA surface: 1 main `.cu` file (`MatOpsCuda.cu`), 1 helper `.cuh` file (`CudaAtomic.cuh`), 1 defs header (`CudaDefs.h`)
3. No warp intrinsics (no `__shfl*`, `__ballot`, `__activemask`)
4. Library swaps are 1:1: cuBLAS -> hipBLAS, cuSOLVER -> hipSOLVER
5. Custom kernels use simple thread indexing (blockIdx/threadIdx) without warp-level operations

Approach:
1. Add `cuda_to_hip.h` compat header in `baspacho/baspacho/`
2. Add `USE_HIP` CMake option with HIP language enablement
3. Mark `.cu` files as `LANGUAGE HIP` when `USE_HIP=ON`
4. Update `CudaDefs.h` to include HIP headers and hipBLAS/hipSOLVER when `USE_HIP` defined
5. Update `Utils.h` to define `__BASPACHO_HOST_DEVICE__` for `__HIPCC__` as well as `__CUDACC__`

## CUDA surface inventory

### Source files
| File | Type | Content |
|------|------|---------|
| `baspacho/baspacho/MatOpsCuda.cu` | CUDA source | Main GPU implementation (~1450 lines): kernels for factor, elimination, solve, assemble; cuBLAS/cuSOLVER calls |
| `baspacho/baspacho/CudaAtomic.cuh` | Header | Device atomicAdd templates for matrix operations |
| `baspacho/baspacho/CudaDefs.h` | Header | CUDA library includes, error checking macros, DevMirror utility class |
| `cmake/detect_cuda.cu` | Build helper | CUDA arch detection at configure time (N/A for HIP build) |

### CUDA runtime APIs used
| CUDA Symbol | HIP Equivalent | Used in |
|-------------|----------------|---------|
| `cudaMalloc` | `hipMalloc` | CudaDefs.h (DevMirror) |
| `cudaFree` | `hipFree` | CudaDefs.h (DevMirror) |
| `cudaMemcpy` | `hipMemcpy` | CudaDefs.h (DevMirror) |
| `cudaMemcpyHostToDevice` | `hipMemcpyHostToDevice` | CudaDefs.h |
| `cudaMemcpyDeviceToHost` | `hipMemcpyDeviceToHost` | CudaDefs.h |
| `cudaStreamSynchronize` | `hipStreamSynchronize` | MatOpsCuda.cu |
| `cudaError_t` | `hipError_t` | CudaDefs.h |
| `cudaSuccess` | `hipSuccess` | CudaDefs.h |
| `cudaGetErrorString` | `hipGetErrorString` | CudaDefs.h |
| `cudaDeviceReset` | `hipDeviceReset` | CudaDefs.h |

### cuBLAS usage (all have 1:1 hipBLAS equivalents)
- `cublasCreate` / `cublasDestroy` -> `hipblasCreate` / `hipblasDestroy`
- `cublasDtrsm` / `cublasStrsm` -> `hipblasDtrsm` / `hipblasStrsm`
- `cublasDgemm` / `cublasSgemm` -> `hipblasDgemm` / `hipblasSgemm`
- `cublasDsymm` / `cublasSsymm` -> `hipblasDsymm` / `hipblasSsymm`
- `cublasDtrsmBatched` / `cublasStrsmBatched` -> `hipblasDtrsmBatched` / `hipblasStrsmBatched`
- `cublasDgemmBatched` / `cublasSgemmBatched` -> `hipblasDgemmBatched` / `hipblasSgemmBatched`
- Status enums: `CUBLAS_STATUS_*` -> `HIPBLAS_STATUS_*`
- Fill/side enums: `CUBLAS_FILL_MODE_*`, `CUBLAS_SIDE_*`, `CUBLAS_OP_*` -> `HIPBLAS_*`

### cuSOLVER usage (all have 1:1 hipSOLVER equivalents)
- `cusolverDnCreate` / `cusolverDnDestroy` -> `hipsolverDnCreate` / `hipsolverDnDestroy`
- `cusolverDnDpotrf` / `cusolverDnSpotrf` -> `hipsolverDnDpotrf` / `hipsolverDnSpotrf`
- `cusolverDnDpotrf_bufferSize` / `cusolverDnSpotrf_bufferSize` -> `hipsolverDnDpotrf_bufferSize` / `hipsolverDnSpotrf_bufferSize`
- `cusolverDnDpotrfBatched` / `cusolverDnSpotrfBatched` -> `hipsolverDnDpotrfBatched` / `hipsolverDnSpotrfBatched`

### cuSPARSE
- Headers included but no actual cuSPARSE function calls found in the codebase. Only error enum definitions in `CublasError.cpp`. Can be removed from HIP build or mapped to hipSPARSE for forward compatibility.

### Kernel characteristics
- Launch configurations use hardcoded `wgs = 32` workgroup sizes -- these are launch parameters, NOT warp-size-dependent
- No warp intrinsics: no `__shfl*`, `__ballot`, `__activemask`, `__syncwarp`, `warpSize`
- Thread indexing: simple `blockIdx.x * blockDim.x + threadIdx.x` pattern
- atomicAdd: standard usage, no custom warp-level atomics

### Device macros
- `__BASPACHO_HOST_DEVICE__` in `Utils.h` line 143-147: keyed on `__CUDACC__`, needs `__HIPCC__` addition

## Risk list

| Risk | Severity | Mitigation |
|------|----------|------------|
| `__CUDACC__` macro check misses HIP | Low | Add `\|\| defined(__HIPCC__)` to the check in Utils.h |
| cuSOLVER batched Cholesky API differences | Low | hipSOLVER provides 1:1 batched potrf API |
| CMake CUDA arch detection code | Low | Write HIP-specific arch detection or rely on `CMAKE_HIP_ARCHITECTURES` |
| Wave64 vs Wave32 | None | No warp intrinsics used; kernels are warp-agnostic |
| Texture/surface | None | Not used |
| Hardcoded `32` values | None | These are workgroup sizes for launch config, not warp-dependent |

## File-by-file change list

### New files
1. `baspacho/baspacho/cuda_to_hip.h` -- compat header with CUDA->HIP symbol aliases

### Modified files
1. `CMakeLists.txt`
   - Add `BASPACHO_USE_HIP` option
   - When `USE_HIP=ON`: `enable_language(HIP)`, set `CMAKE_HIP_ARCHITECTURES` with default gfx90a
   - Link hipblas, hipsolver instead of CUDA libraries
   - Set `.cu` files to `LANGUAGE HIP`

2. `baspacho/baspacho/CMakeLists.txt`
   - Add HIP build path parallel to CUBLAS path
   - Link `hip::host`, `hip::device`, `hipblas`, `hipsolver`
   - Set `HIP_ARCHITECTURES` on targets

3. `baspacho/baspacho/CudaDefs.h`
   - Add `#if defined(USE_HIP)` branch to include HIP headers (hip_runtime, hipblas, hipsolver)
   - Map error check macros to HIP equivalents

4. `baspacho/baspacho/Utils.h`
   - Update `__BASPACHO_HOST_DEVICE__` macro: `#if defined(__CUDACC__) || defined(__HIPCC__)`

5. `baspacho/baspacho/MatOpsCuda.cu`
   - Include `cuda_to_hip.h` at top
   - Minimal changes: compat header handles symbol mapping

6. `baspacho/baspacho/CublasError.cpp`
   - Add HIP error enum mappings under `#if defined(USE_HIP)` guards

7. `baspacho/tests/CMakeLists.txt`
   - Enable CUDA tests when `USE_HIP=ON` (rename condition or add HIP equivalent)

## Build commands

### Configure (gfx90a)
```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBASPACHO_USE_HIP=ON \
  -DBASPACHO_USE_CUBLAS=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
```

### Build
```bash
cmake --build build -- -j$(nproc)
```

### Follower platforms (gfx1100, gfx1101, gfx1201)
```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBASPACHO_USE_HIP=ON \
  -DBASPACHO_USE_CUBLAS=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100  # or gfx1101, gfx1201
```

## Test plan

### GPU tests (require real AMD GPU)
The project has 5 explicit CUDA/GPU test executables:
- `CudaFactorTest` -- GPU Cholesky factorization correctness
- `CudaSolveTest` -- GPU triangular solve correctness
- `BatchedCudaFactorTest` -- Batched GPU factorization
- `BatchedCudaSolveTest` -- Batched GPU solve
- `CudaPartialTest` -- Partial factorization on GPU

Run GPU tests:
```bash
cd build && ctest -R Cuda --output-on-failure
```

Or run all tests:
```bash
cd build && ctest --output-on-failure
```

### Non-GPU tests (must not regress)
- `AccessorTest`
- `CoalescedBlockMatrixTest`
- `CreateSolverTest`
- `SparseStructureTest`
- `EliminationTreeTest`
- `FactorTest`
- `SolveTest`
- `MathUtilsTest`
- `PartialFactorSolveTest`

Run CPU-only tests:
```bash
cd build && ctest -E Cuda --output-on-failure
```

### Benchmarks (optional, for performance validation)
```bash
build/baspacho/benchmarking/bench -B 1_CHOLMOD  # if CHOLMOD available
```

## Open questions

1. **Naming**: The project uses `BASPACHO_USE_CUBLAS` as the GPU enable flag even though it uses cuBLAS AND cuSOLVER. Should we add a parallel `BASPACHO_USE_HIP` flag or a more generic `BASPACHO_USE_GPU`?
   - Recommendation: Add `BASPACHO_USE_HIP` flag, keep `BASPACHO_USE_CUBLAS` for NVIDIA path unchanged.

2. **cuSPARSE headers**: Included but unused. Remove from HIP build or map to hipSPARSE?
   - Recommendation: Map to hipSPARSE for forward compatibility (library may use it in future).

3. **Legacy CUDA arch workaround**: The project has special handling for CUDA arch < 6.0 (atomicAdd double workaround). This is N/A for HIP (all supported AMD GPUs have native double atomics).
   - Recommendation: Skip legacy handling in HIP path.
