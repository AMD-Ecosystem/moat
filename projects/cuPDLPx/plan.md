# Plan: cuPDLPx ROCm Port

## Project

- **Name:** cuPDLPx
- **Upstream:** https://github.com/MIT-Lu-Lab/cuPDLPx
- **Default branch:** main
- **Description:** GPU-accelerated first-order linear programming solver based on restarted Halpern PDHG method

## Existing AMD support

**Search results:**
- Grepped upstream docs for AMD/ROCm/HIP references: none found
- WebSearch for "cuPDLPx ROCm", "cuPDLPx AMD GPU", "cuPDLPx HIP": no results
- WebSearch for "PDLP AMD GPU": found torchPDLP (SimplySnap/torchPDLP), a separate PyTorch-based PDLP implementation for AMD, NOT a port of cuPDLPx
- Checked upstream branches: only `main` and `fix_zlib`, no rocm/hip branches
- Checked GitHub forks: one fork (vishalbelsare/cuPDLPx) is a simple mirror, no AMD work

**Decision:** No existing AMD/ROCm/HIP port of cuPDLPx. torchPDLP is a distinct project (pure Python/PyTorch) targeting AMD, not a port of cuPDLPx's CUDA C++ codebase. Proceed with from-scratch port.

## Build classification

**Type:** Pure CMake project (Strategy A)

**Evidence:**
- `CMakeLists.txt` line 7: `project(cupdlpx LANGUAGES C CXX CUDA)`
- No `find_package(Torch)`, no `torch.utils.cpp_extension`, no `CUDAExtension`
- Uses CMake's native CUDA language support with `enable_language(CUDA)`
- Dependencies: zlib, PSLP (FetchContent), CUDA Toolkit (cuBLAS, cuSPARSE)
- Python bindings via pybind11, not torch extensions

## Port strategy

**Strategy A:** Pure CMake with a compat header

**Rationale:**
- Clean separation of .cu files from host code
- cuBLAS/cuSPARSE map directly to hipBLAS/hipSPARSE
- No pytorch dependency, so no hipify-torch needed
- Single compat header covers all CUDA symbols used

## CUDA surface inventory

### Source files (5 .cu files)
| File | Lines | Description |
|------|-------|-------------|
| `src/solver.cu` | 1282 | Main PDHG solver, CUDA graphs |
| `src/utils.cu` | 1320 | SpMV backend, residual computation |
| `src/preconditioner.cu` | 500 | Ruiz/Pock-Chambolle scaling |
| `src/feasibility_polish.cu` | 613 | Feasibility polishing phase |
| `src/spmv_backend.cu` | 259 | cuSPARSE SpMV wrappers |

### Kernels (`__global__`)
- `build_row_ind`, `build_transpose_map`
- `fill_finite_bounds_kernel`, `rescale_solution_kernel`
- `compute_delta_solution_kernel`, `compute_next_primal_solution_kernel`
- `compute_next_primal_solution_major_kernel`, `compute_next_dual_solution_kernel`
- `compute_next_dual_solution_major_kernel`, `compute_and_rescale_reduced_cost_kernel`
- `compute_residual_kernel`, `primal_infeasibility_project_kernel`
- `dual_infeasibility_project_kernel`, `compute_primal_infeasibility_kernel`
- `compute_dual_infeasibility_kernel`, `dual_solution_dual_objective_contribution_kernel`
- `dual_objective_dual_slack_contribution_array_kernel`
- `scale_variables_kernel`, `scale_constraints_kernel`, `scale_csr_nnz_kernel`
- `compute_csr_row_absmax_kernel`, `compute_csr_row_powsum_kernel`
- `clamp_sqrt_and_accum_kernel`, `compute_bound_contrib_kernel`
- `scale_bounds_kernel`, `scale_objective_kernel`, `fill_ones_kernel`
- `zero_finite_value_vectors_kernel`, `compute_delta_primal_solution_kernel`
- `compute_delta_dual_solution_kernel`, `compute_primal_feas_polish_residual_kernel`
- `compute_dual_feas_polish_residual_kernel`

### Device functions
- `__device__ __forceinline__ pow_fast()` in preconditioner.cu

### Warp intrinsics
- **None found.** All kernels use simple thread indexing (`blockIdx.x * blockDim.x + threadIdx.x`), no `__shfl*`, `__ballot`, `warpSize`, or hardcoded 32.

### Textures/Surfaces
- **None used.** All data access is via raw pointers.

### CUDA runtime API
| Symbol | HIP equivalent | Count |
|--------|----------------|-------|
| `cudaMalloc` | `hipMalloc` | ~50 |
| `cudaFree` | `hipFree` | ~50 |
| `cudaMemcpy` | `hipMemcpy` | ~30 |
| `cudaMemcpyAsync` | `hipMemcpyAsync` | ~5 |
| `cudaMemset` | `hipMemset` | ~10 |
| `cudaGetLastError` | `hipGetLastError` | ~5 |
| `cudaStream_t` | `hipStream_t` | ~5 |
| `cudaStreamCreate` | `hipStreamCreate` | 1 |
| `cudaStreamDestroy` | `hipStreamDestroy` | 1 |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` | 1 |
| `cudaError_t` | `hipError_t` | ~10 |
| `cudaSuccess` | `hipSuccess` | ~10 |
| `cudaMemcpyHostToDevice` | `hipMemcpyHostToDevice` | ~20 |
| `cudaMemcpyDeviceToHost` | `hipMemcpyDeviceToHost` | ~15 |
| `cudaMemcpyDeviceToDevice` | `hipMemcpyDeviceToDevice` | ~20 |
| `cudaGetErrorName` | `hipGetErrorName` | 1 |

### CUDA Graph API
| Symbol | HIP equivalent |
|--------|----------------|
| `cudaGraph_t` | `hipGraph_t` |
| `cudaGraphExec_t` | `hipGraphExec_t` |
| `cudaStreamBeginCapture` | `hipStreamBeginCapture` |
| `cudaStreamEndCapture` | `hipStreamEndCapture` |
| `cudaStreamCaptureModeGlobal` | `hipStreamCaptureModeGlobal` |
| `cudaGraphInstantiate` | `hipGraphInstantiate` |
| `cudaGraphDestroy` | `hipGraphDestroy` |
| `cudaGraphLaunch` | `hipGraphLaunch` |
| `cudaGraphExecDestroy` | `hipGraphExecDestroy` |

### cuBLAS usage
| Symbol | HIP equivalent |
|--------|----------------|
| `cublasHandle_t` | `hipblasHandle_t` |
| `cublasCreate` | `hipblasCreate` |
| `cublasDestroy` | `hipblasDestroy` |
| `cublasSetStream` | `hipblasSetStream` |
| `cublasSetPointerMode` | `hipblasSetPointerMode` |
| `CUBLAS_POINTER_MODE_HOST` | `HIPBLAS_POINTER_MODE_HOST` |
| `cublasStatus_t` | `hipblasStatus_t` |
| `CUBLAS_STATUS_SUCCESS` | `HIPBLAS_STATUS_SUCCESS` |
| `cublasGetStatusName` | `hipblasStatusToString` |
| `cublasDnrm2` | `hipblasDnrm2` |
| `cublasDnrm2_v2_64` | `hipblasDnrm2_64` (or use hipblasDnrm2) |
| `cublasDdot` | `hipblasDdot` |
| `cublasDscal` | `hipblasDscal` |
| `cublasDaxpy` | `hipblasDaxpy` |
| `cublasIdamax` | `hipblasIdamax` |

### cuSPARSE usage
| Symbol | HIP equivalent |
|--------|----------------|
| `cusparseHandle_t` | `hipsparseHandle_t` |
| `cusparseCreate` | `hipsparseCreate` |
| `cusparseDestroy` | `hipsparseDestroy` |
| `cusparseSetStream` | `hipsparseSetStream` |
| `cusparseStatus_t` | `hipsparseStatus_t` |
| `CUSPARSE_STATUS_SUCCESS` | `HIPSPARSE_STATUS_SUCCESS` |
| `cusparseGetErrorName` | `hipsparseGetErrorName` |
| `cusparseSpMatDescr_t` | `hipsparseSpMatDescr_t` |
| `cusparseDnVecDescr_t` | `hipsparseDnVecDescr_t` |
| `cusparseCreateCsr` | `hipsparseCreateCsr` |
| `cusparseDestroySpMat` | `hipsparseDestroySpMat` |
| `cusparseCreateDnVec` | `hipsparseCreateDnVec` |
| `cusparseDestroyDnVec` | `hipsparseDestroyDnVec` |
| `cusparseDnVecSetValues` | `hipsparseDnVecSetValues` |
| `cusparseSpMV` | `hipsparseSpMV` |
| `cusparseSpMV_bufferSize` | `hipsparseSpMV_bufferSize` |
| `cusparseSpMV_preprocess` | (may need shim) |
| `cusparseCsr2cscEx2` | `hipsparseCsr2cscEx2` |
| `cusparseCsr2cscEx2_bufferSize` | `hipsparseCsr2cscEx2_bufferSize` |
| `CUSPARSE_INDEX_32I` | `HIPSPARSE_INDEX_32I` |
| `CUSPARSE_INDEX_BASE_ZERO` | `HIPSPARSE_INDEX_BASE_ZERO` |
| `CUSPARSE_OPERATION_NON_TRANSPOSE` | `HIPSPARSE_OPERATION_NON_TRANSPOSE` |
| `CUSPARSE_ACTION_NUMERIC` | `HIPSPARSE_ACTION_NUMERIC` |
| `CUSPARSE_CSR2CSC_ALG_DEFAULT` | `HIPSPARSE_CSR2CSC_ALG_DEFAULT` |
| `CUSPARSE_SPMV_CSR_ALG2` | `HIPSPARSE_SPMV_CSR_ALG2` |

### cuSPARSE SpMVOp (optional, CUDA 13.1+ only)
The project has optional SpMVOp support gated by `CUPDLPX_HAS_SPMVOP` (cusparse version >= 12.7.3). These APIs are NOT available in hipSPARSE, so the port will use the SpMV path only:
- `cusparseSpMVOp_bufferSize` -> N/A
- `cusparseSpMVOp_createDescr` -> N/A
- `cusparseSpMVOp_createPlan` -> N/A
- `cusparseSpMVOp` -> N/A
- `cusparseSpMVOp_destroyDescr` -> N/A
- `cusparseSpMVOp_destroyPlan` -> N/A

### CUB usage (preconditioner.cu)
| Symbol | HIP equivalent |
|--------|----------------|
| `cub::DeviceReduce::Sum` | `hipcub::DeviceReduce::Sum` |

### Data types
| Symbol | HIP equivalent |
|--------|----------------|
| `CUDA_R_64F` | `HIP_R_64F` |

## Risk list

1. **cuSPARSE SpMVOp API unavailable in hipSPARSE:** The project conditionally uses `cusparseSpMVOp` (CUDA 13.1+). hipSPARSE does not have SpMVOp. **Mitigation:** Force `CUPDLPX_HAS_SPMVOP=0` in the HIP build so the SpMV path is used instead. This is already handled by the existing version check macro, just need to ensure it evaluates false on HIP.

2. **cuBLAS `_v2_64` function naming:** The project uses `cublasDnrm2_v2_64` which takes 64-bit integers. hipBLAS has `hipblasDnrm2Ex` with similar semantics, or the standard `hipblasDnrm2` should work since problem sizes are within int32 range for LP problems. **Mitigation:** Map to the standard hipBLAS functions.

3. **`cusparseSpMV_preprocess` availability:** This function may not exist in all hipSPARSE versions. **Mitigation:** hipSPARSE 3.x+ supports this; check ROCm 7.2 compatibility.

4. **`cublasGetStatusName`/`cusparseGetErrorName`:** May have different signatures. **Mitigation:** Map to hipBLAS/hipSPARSE equivalents (hipblasStatusToString, hipsparseGetErrorName).

5. **CUDA Graph API compatibility:** HIP Graph API mirrors CUDA. The project uses basic graph capture/instantiate/launch pattern which is well-supported.

6. **THREADS_PER_BLOCK = 256:** This is arch-agnostic and works on both CUDA and HIP.

7. **No warp-size dependencies:** The code does not use warp intrinsics or hardcode warp size, eliminating wave32/wave64 concerns.

8. **No texture/surface usage:** No texture pitch alignment or layered array issues.

## File-by-file change list

### New files
1. **`internal/cuda_to_hip.h`** - Compat header mapping CUDA -> HIP symbols for:
   - CUDA runtime API (cudaMalloc, cudaFree, etc.)
   - CUDA Graph API (cudaGraph_t, cudaGraphExec_t, etc.)
   - cuBLAS API (cublasHandle_t, cublasDnrm2, etc.)
   - cuSPARSE API (cusparseHandle_t, cusparseSpMV, etc.)
   - CUB -> hipCUB (cub::DeviceReduce -> hipcub::DeviceReduce)
   - Data types (CUDA_R_64F -> HIP_R_64F)

### Modified files
1. **`CMakeLists.txt`**
   - Add `option(USE_HIP "Build with HIP for AMD GPUs" OFF)`
   - Add HIP language enable and architecture configuration
   - Add hipBLAS, hipSPARSE, hipCUB dependencies under USE_HIP
   - Set .cu files to LANGUAGE HIP when USE_HIP is ON

2. **`internal/utils.h`**
   - Include `cuda_to_hip.h` before cuda_runtime.h/cublas/cusparse

3. **`internal/cusparse_compat.h`**
   - Add HIP/hipSPARSE equivalent definitions
   - Force `CUPDLPX_HAS_SPMVOP=0` on HIP (no SpMVOp in hipSPARSE)

4. **`src/preconditioner.cu`**
   - Change `#include <cub/device/device_reduce.cuh>` to include hipCUB on HIP
   - Map `cub::DeviceReduce` -> `hipcub::DeviceReduce`

5. **`src/spmv_backend.cu`**
   - Ensure SpMV path is used (non-SpMVOp) on HIP
   - Include compat header

6. **All .cu files**
   - Include compat header at top

## Build commands

### Configure (gfx90a)
```bash
cmake -B build \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCUPDLPX_BUILD_TESTS=ON \
  -DCUPDLPX_BUILD_PYTHON=OFF
```

### Build
```bash
cmake --build build -j$(nproc)
```

### Follower architectures
```bash
# gfx1100
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 ...

# gfx1151 (Windows)
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 ...
```

## Test plan

### CLI test (smoke test)
```bash
# Download test instance
wget -P test/ https://miplib.zib.de/WebData/instances/2club200v15p5scn.mps.gz

# Run solver
./build/cupdlpx test/2club200v15p5scn.mps.gz test/ -v
```
Expected: Solver converges to OPTIMAL, produces output files.

### C interface test
```bash
./build/tests/test_interface
```
Expected: Test passes.

### Python tests (optional, if building Python bindings)
```bash
pip install -e .
pytest test/ -v
```
Tests include:
- `test_basic.py` - Basic LP solve (minimize/maximize)
- `test_feasibility_polishing.py` - Polishing phase
- `test_infeasible_unbounded.py` - Infeasible/unbounded detection
- `test_limit.py` - Time/iteration limits
- `test_matrix_formats.py` - Dense/CSC/COO input formats
- `test_numerical.py` - Numerical accuracy
- `test_presolve.py` - Presolve functionality
- `test_warm_start.py` - Warm start capability

### Non-GPU tests
- MPS parser tests (CPU-only)
- Presolve tests (mostly CPU)

### Validation criteria
1. CLI solves test LP to OPTIMAL status
2. Primal/dual solutions match reference within tolerance (1e-4 relative)
3. Objective value matches expected
4. No GPU memory errors (run with ROCM_LOG_LEVEL=4 to check)

## Open questions

1. **hipSPARSE SpMV_preprocess availability:** Need to confirm `hipsparseSpMV_preprocess` exists in ROCm 7.2.x. If not, may need to skip preprocessing or use a no-op.

2. **hipBLAS `_v2_64` equivalents:** Confirm hipBLAS has 64-bit integer versions of BLAS functions, or verify 32-bit versions suffice for LP problem scales.

3. **Python bindings on ROCm:** pybind11 builds should work with ROCm, but may need testing. Consider deprioritizing Python bindings for initial port.

4. **CUDA Graph performance on HIP:** CUDA Graphs are used for the main iteration loop. HIP Graph support is mature, but performance may differ.
