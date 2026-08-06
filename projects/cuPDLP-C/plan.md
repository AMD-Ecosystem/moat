# cuPDLP-C ROCm Port Plan

## Project

- **Name**: cuPDLP-C
- **Upstream**: https://github.com/COPT-Public/cuPDLP-C
- **Default branch**: main
- **Base SHA**: 7b94c41efca42fa176a129a935b59718e1455fad

cuPDLP-C is a GPU-accelerated linear programming solver implementing the Primal-Dual Linear Programming (PDLP) first-order algorithm. It is the C implementation of cuPDLP.jl and integrates with HiGHS for MPS file parsing.

## Existing AMD support

**Assessment**: No existing AMD/ROCm/HIP support.

- Grepped `README.md` and source tree for `amd|rocm|hip|gfx[0-9]` -- no matches.
- No rocm/hip/amd branches in upstream or notable forks.
- Checked forks: yiakwy-xpu-ml-framework-team/OSS-cuPDLP-C, ERGO-Code/cuPDLP-C, KKT-OPT/cuPDLP-C -- no AMD branches.
- Web search for "cuPDLP-C ROCm", "cuPDLP AMD GPU", "cuPDLP HIP" returned no results about an existing port.
- No ROCm/AMD/GPUOpen org repos for cuPDLP found.

**Decision**: Proceed with a from-scratch HIP port. This is valuable as cuPDLP-C is a high-performance first-order LP solver with no existing AMD GPU support.

## Build classification

**Classification**: Pure CMake project (Strategy A)

**Evidence**:
- `/CMakeLists.txt` line 1-9: `cmake_minimum_required(VERSION 3.24)`, `project(CUPDLP ...)`, `enable_language(C CXX)`
- `/CMakeLists.txt` line 5: `option(BUILD_CUDA ...)` controls GPU build
- `/FindCUDAConf.cmake` line 9: `enable_language(CUDA)`
- No `find_package(Torch)`, no `torch.utils.cpp_extension`, no `CUDAExtension`, no setup.py with torch dependency
- Pure C/C++ project with CUDA kernels, optional Python bindings via pybind11

## Port strategy

**Strategy A: Pure CMake, cuda_to_hip.h compat header**

**Rationale**: This is a standalone CMake project with `.cu` sources, cuBLAS, and cuSPARSE usage -- exactly the pattern Strategy A addresses. The project has a clean separation: CUDA code is confined to `cupdlp/cuda/` with two `.cu` files and two `.cuh` headers. The host C code in `cupdlp/*.c` calls CUDA functions through thin wrappers.

## CUDA surface inventory

### Kernel files
- `cupdlp/cuda/cupdlp_cuda_kernels.cu` (339 lines): 22 element-wise kernels + 3 reduction kernels
- `cupdlp/cuda/cupdlp_cudalinalg.cu` (386 lines): SpMV wrappers, kernel launch helpers, movement computation

### __global__ kernels (cupdlp_cuda_kernels.cu)
Simple element-wise kernels (grid-stride loop pattern, no warp intrinsics):
- `element_primal_feas_kernel`, `element_dual_feas_kernel_1/2/3`
- `element_primal_infeas_kernel`, `element_dual_infeas_kernel_lb/ub/constr`
- `element_wise_dot_kernel`, `element_wise_div_kernel`
- `element_wise_projlb/ub_kernel`, `element_wise_projSamelb/ub_kernel`
- `element_wise_initHaslb/ub_kernel`, `element_wise_filterlb/ub_kernel`
- `init_cuda_vec_kernel`
- `primal_grad_step_kernel`, `dual_grad_step_kernel`

Reduction kernels with warp shuffle intrinsics:
- `movement_1_kernel` (256-thread blocks, warp reduction)
- `movement_2_kernel` (256-thread blocks, warp reduction)
- `sum_kernel` (512-thread blocks, warp reduction)

### Warp intrinsics (CRITICAL for AMD port)
Lines 202-281 in `cupdlp_cuda_kernels.cu`:
- `__shfl_down_sync(0xFFFFFFFF, val, offset)` with offsets 1, 2, 4, 8, 16
- Hardcoded warp-size=32 assumptions throughout:
  - `int lane = threadIdx.x % 32`
  - `int wid = threadIdx.x / 32`
  - `__shared__ cupdlp_float shared[32]` (warp count for 256-thread block)
  - Comment: "assumes block size = 256, warp size = 32"
- Explicit warpSize=32 runtime check at `cupdlp_cudalinalg.cu:260-264` that `exit(1)` if not 32

### Library usage
- **cuBLAS**: `cublasDaxpy`, `cublasSaxpy`, `cublasDdot`, `cublasSdot`, `cublasDnrm2`, `cublasSnrm2`, `cublasDscal`, `cublasSscal`, `cublasCreate`, `cublasDestroy`
- **cuSPARSE**: `cusparseSpMV`, `cusparseSpMV_bufferSize`, `cusparseCreateCsc`, `cusparseCreateCsr`, `cusparseCreateDnVec`, `cusparseDestroySpMat`, `cusparseDestroyDnVec`, `cusparseCreate`, `cusparseDestroy`, `cusparseGetVersion`
- `CUSPARSE_SPMV_CSR_ALG2` algorithm enum

### CUDA runtime API
- Memory: `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemset`
- Sync: `cudaDeviceSynchronize`, `cudaGetLastError`
- Device query: `cudaGetDeviceCount`, `cudaGetDeviceProperties`, `cudaDeviceGetAttribute`, `cudaRuntimeGetVersion`, `cudaDriverGetVersion`

### No usage of
- Textures/surfaces
- Streams (beyond default)
- Events
- cudaMallocManaged/cudaMallocHost
- Thrust/CUB
- cuRAND/cuFFT

## Risk list

### High risk: Hardcoded warp size = 32

**Impact**: CDNA (gfx90a) has 64-lane wavefronts; the existing code will crash or compute incorrectly.

**Locations**:
1. `cupdlp_cuda_kernels.cu` lines 202-281: Warp shuffle macros assume 32-wide warps
2. `cupdlp_cuda_kernels.cu` lines 224-311: `int lane = threadIdx.x % 32`, `int wid = threadIdx.x / 32`
3. `cupdlp_cuda_kernels.cu` lines 230, 288, 316: `__shared__ cupdlp_float shared[32]` sized for 256/32=8 warps
4. `cupdlp_cudalinalg.cu` lines 260-264: Runtime check `if (warpSize != 32) exit(1)`

**Fix approach**:
- Device code: Use `__GFX9__` guard to select 64-wide vs 32-wide warp operations
- The shuffle reduction macros must be parameterized for warp width
- Shared memory arrays must be sized for `blockDim.x / warpSize` (constexpr upper bound: 64 for 256/4 case)
- Remove the `exit(1)` check on warpSize != 32; replace with proper multi-arch support

**Multi-arch note**: Per PORTING_GUIDE, build with `-DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100"` to verify both wave64 (CDNA) and wave32 (RDNA) work in a single binary.

### Medium risk: cuSPARSE SpMV algorithm enum

`CUSPARSE_SPMV_CSR_ALG2` is used throughout. hipSPARSE has `HIPSPARSE_SPMV_CSR_ALG2` but verify it exists and behaves identically. The deterministic SpMV algorithm is important for LP solver convergence.

### Low risk: cuBLAS/cuSPARSE handle types

hipBLAS and hipSPARSE have equivalent types but need aliasing in the compat header:
- `cublasHandle_t` -> `hipblasHandle_t`
- `cusparseHandle_t` -> `hipsparseHandle_t`
- `cusparseSpMatDescr_t` -> `hipsparseSpMatDescr_t`
- `cusparseDnVecDescr_t` -> `hipsparseDnVecDescr_t`

### Low risk: FMA intrinsic

`cupdlp_fma_rn` maps to `__fma_rn` (double) or `__fmaf_rn` (float). HIP has identical intrinsics.

## File-by-file change list

### New files
1. `cupdlp/cuda/cuda_to_hip.h` -- compat header with all CUDA->HIP aliases

### Modified files

1. **CMakeLists.txt** (root)
   - Add `option(USE_HIP "Build with HIP for AMD GPUs" OFF)`
   - Add HIP configuration block parallel to CUDA
   - Create `FindHIPConf.cmake` similar to `FindCUDAConf.cmake`

2. **FindHIPConf.cmake** (new)
   - `enable_language(HIP)`
   - Find hipBLAS, hipSPARSE libraries
   - Set `CMAKE_HIP_ARCHITECTURES` default to gfx90a (configurable)

3. **cupdlp/cuda/CMakeLists.txt**
   - Gate `enable_language(CUDA)` vs `enable_language(HIP)`
   - Mark `.cu` files as `LANGUAGE HIP` when USE_HIP
   - Link hipblas, hipsparse instead of cublas, cusparse

4. **cupdlp/CMakeLists.txt**
   - Add HIP library path parallel to CUDA

5. **cupdlp/cuda/cupdlp_cuda_kernels.cuh**
   - Include `cuda_to_hip.h` at top
   - No other changes needed (compat header handles API mapping)

6. **cupdlp/cuda/cupdlp_cuda_kernels.cu**
   - Include `cuda_to_hip.h`
   - Fix warp shuffle macros for wave64:
     - `FULL_WARP_REDUCE` needs wave64 path with offset 32
     - Use `__GFX9__` guard to select 64-wide vs 32-wide code paths
   - Fix `lane = threadIdx.x % warpSize_const` where `warpSize_const` is arch-dependent constexpr
   - Fix shared memory sizing to `blockDim.x / kMinWarpSize` (use 32 as min)

7. **cupdlp/cuda/cupdlp_cudalinalg.cu**
   - Include `cuda_to_hip.h`
   - Remove or fix the `if (warpSize != 32) exit(1)` check
   - No other logic changes needed

8. **cupdlp/cuda/cupdlp_cudalinalg.cuh**
   - Include `cuda_to_hip.h` before cublas/cusparse includes

9. **cupdlp/cupdlp_defs.h**
   - Include `cuda_to_hip.h` at top (before CUDA headers)

10. **cupdlp/cupdlp_linalg.h**, **cupdlp/cupdlp_utils.c**, **cupdlp/cupdlp_solver.c**, **cupdlp/cupdlp_step.c**
    - These include CUDA headers indirectly; compat header propagates through includes

11. **interface/mps_clp.c**, **apps/onlinelp.cpp**
    - Include compat header for `cublasCreate` calls

## Build commands

### Prerequisites
```bash
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export HIGHS_HOME=/path/to/highs  # HiGHS 1.6.0+ with zlib support
```

### Configure (gfx90a)
```bash
mkdir build && cd build
cmake .. \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS_RELEASE="-O2 -DNDEBUG" \
  -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG"
```

### Build
```bash
cmake --build . --target plc -j$(nproc)
```

### Multi-arch validation build
```bash
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100"
cmake --build . -j$(nproc)
# Verify both code objects: llvm-objdump --offloading build/lib/libcudalin.so
```

## Test plan

### Unit tests (GPU)
The project has two CUDA test executables:
```bash
./build/bin/testcudalin   # tests element-wise ops
./build/bin/testcublas    # tests cuBLAS nrm2
```

### Integration test (GPU)
Run the solver on the included example:
```bash
./build/bin/plc -fname ../example/afiro.mps -nIterLim 5000
```

Expected: Solver converges with primal/dual feasibility and gap below default tolerances (1e-4).

### Regression test suite
No formal test suite. Validation should use:
1. The `afiro.mps` example (small LP, ~32 rows x 51 cols)
2. Standard LP benchmarks if available (MIPLIB LP relaxations, Netlib)

### Non-GPU tests
The project has a CPU-only build (`-DBUILD_CUDA=OFF`). Ensure the CPU path still builds and runs correctly after the port.

```bash
mkdir build_cpu && cd build_cpu
cmake .. -DBUILD_CUDA=OFF -DBUILD_APPS=ON
cmake --build . --target plc
./bin/plc -fname ../example/afiro.mps -nIterLim 5000
```

### Wave64/wave32 validation
After the port, build for both gfx90a (wave64) and gfx1100 (wave32) and verify:
1. Both produce correct results on `afiro.mps`
2. Results match between architectures (deterministic algorithm)

## Open questions

1. **HiGHS version compatibility**: The README requires HiGHS 1.6.0. Confirm ROCm environment has a compatible HiGHS or document the build.

2. **CUSPARSE_SPMV_CSR_ALG2**: Verify hipSPARSE has the equivalent `HIPSPARSE_SPMV_CSR_ALG2` and it provides deterministic SpMV.

3. **Python interface**: The pycupdlp module will need the same HIP treatment if Python bindings are wanted. Lower priority than the core solver.

4. **Benchmarking**: After correctness validation, performance comparison against the CUDA build would be valuable but is out of scope for the initial port.
