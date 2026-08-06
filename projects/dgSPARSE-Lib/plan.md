# dgSPARSE-Lib Porting Plan

## Project

- **Name**: dgSPARSE-Lib
- **Upstream**: https://github.com/dgSPARSE/dgSPARSE-Lib
- **Default branch**: main
- **Description**: Deep Geometric Sparse Library -- high-performance sparse kernel acceleration for GNNs, RecSys, and 3D pointcloud detection

## Existing AMD support

**None found.** No HIP/ROCm/AMD references in README or docs. Web searches for "dgSPARSE ROCm", "dgSPARSE AMD GPU", "dgSPARSE HIP" returned no relevant results. The `xiakubaobaore/dgsparse-hip` repo exists but is a non-authoritative community fork (0 stars, no description, last pushed 2024-01-10, appears abandoned). The upstream has no ROCm-related PRs, issues, or branches. No AMD forks in ROCm/AMD/GPUOpen orgs.

**Decision**: Proceed with a from-scratch port. Do not adopt the community fork as a base.

## Build classification

**PyTorch extension (Strategy B)**.

Evidence (setup.py lines 8-12, 57):
```python
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CUDAExtension,
)
...
Extension = CUDAExtension
```

The project builds via `pip install -e .` using `torch.utils.cpp_extension.CUDAExtension` and `BuildExtension`. Building against a ROCm torch will auto-hipify the extension sources.

## Port strategy

**Strategy B (torch-hipify)**.

Building a CUDAExtension against a ROCm PyTorch wheel automatically runs `torch.utils.hipify` on the `.cu`/`.cuh` sources. The hipify pass handles most cudaXxx -> hipXxx symbol translation. The main work is fixing fault classes (warp size, library swaps) that hipify does not address.

## CUDA surface inventory

### Source files
| File | Type | Content |
|------|------|---------|
| src/cuda/spmm_cuda.cu | Extension | SpMM kernels, CSR2CSC, SDDMM dispatch |
| src/cuda/spconv_cuda.cu | Extension | Sparse convolution using cuBLAS |
| src/cuda/sparse_mapping.cu | Extension | Sparse coordinate mapping with Thrust |
| src/ge-spmm/*.cu | Standalone | GE-SpMM variants (rowcaching, parreduce, seqreduce) |
| src/sddmm/*.cu | Standalone | SDDMM kernels (COO, CSR) |
| src/gspmm-fp/gspmm.cu | Standalone | GSPMM implementation |
| include/cuda/*.cuh | Headers | Kernel definitions and utilities |
| src/util/cuda_util.cuh | Header | Warp primitives and helpers |
| example/ge-spmm/spmm.cu | Example | cuSPARSE-based SpMM benchmark |
| example/sddmm/sddmm.cu | Example | SDDMM benchmark |

### CUDA runtime symbols
- `cudaSetDevice`, `cudaMalloc`, `cudaFree`, `cudaMemcpy` -- standard runtime, 1:1 HIP mapping via hipify
- `cudaGetLastError`, `cudaSuccess` -- error handling, 1:1 HIP mapping
- `c10/cuda/CUDAGuard.h`, `at::cuda::getCurrentCUDABlasHandle` -- PyTorch CUDA helpers, handled by ROCm torch

### Warp intrinsics (CRITICAL)
Extensive use of width-32 warp primitives:

1. **`__shfl_down_sync(FULLMASK, v, delta)`** -- used in SHFL_DOWN_REDUCE macro (cuda_util.cuh lines 26-27, 30-34, etc.)
2. **`__shfl_xor_sync(FULLMASK, v, stride, 32)`** -- used in AllReduce4 helper (cuda_util.cuh lines 258-261) and multiple SDDMM kernels
3. **`__shfl_up_sync(FULLMASK, v, 1)`** -- used in SEG_SHFL_SCAN macro (cuda_util.cuh line 95)
4. **`__shfl_sync(FULLMASK, v, lane)`** -- used in csrspmm_parreduce (lines 304, 322)
5. **`__syncwarp()`** -- used in csrspmm_rowcaching.cu (lines 60, 95, 179, 242)

All shuffle operations use explicit width=32 parameter and FULLMASK (0xffffffff). The 32-lane width is correct for RDNA (wave32) and works as a logical warp on CDNA (wave64) -- HIP supports width-32 shuffles on wave64 as subgroup operations. This is the SAFE pattern per PORTING_GUIDE.

**Lane arithmetic uses 32**:
- `lane_id = (threadIdx.x & (32 - 1))` -- correct for a 32-lane logical warp
- Loop strides: `jj += 32`, `cid += 32` -- iterating in 32-element chunks
- Block sizes: `blockDim(Ndim_warp_per_tb * 32, ...)` -- 32 threads per logical warp

These are width-32 LOGICAL warp ops, not hardcoded physical warp assumptions. Per PORTING_GUIDE, width-32 `__shfl*(...,32)` and `cub::WarpReduce<,32>` are arch-agnostic and operate within a 32-lane subgroup regardless of the physical wavefront. This is correct and needs no change.

### Library usage
| Library | Files | HIP equivalent |
|---------|-------|----------------|
| cuSPARSE | csr2csc.cuh, gspmm.cu, example/ge-spmm/spmm.cu | hipSPARSE |
| cuBLAS | spconv_cuda.cu | hipBLAS |
| Thrust | sparse_mapping.cu | rocThrust (drop-in) |

### Device attributes
- `c10::cuda::CUDAGuard` -- handled by ROCm torch (becomes HIPGuard)
- `at::cuda::getCurrentCUDABlasHandle()` -- handled by ROCm torch (becomes hipBLAS handle)

### Textures/surfaces
None.

### Managed/pinned memory
None explicit; PyTorch tensor allocation is used.

### Streams/events
Uses default stream (0) for Thrust and cuBLAS operations.

## Risk list

1. **cuSPARSE -> hipSPARSE (MODERATE)**: csr2csc.cuh uses `cusparseCsr2cscEx2` and `cusparseCsr2cscEx2_bufferSize`. hipSPARSE provides `hipsparseXcsr2csc` but the Ex2 API may need adaptation. The function signature differs slightly.

2. **cuBLAS -> hipBLAS (LOW)**: spconv_cuda.cu uses `cublasGemmEx` with `CUBLAS_TENSOR_OP_MATH`. hipBLAS provides `hipblasGemmEx` with similar API. May need `HIPBLAS_GEMM_DEFAULT` instead of tensor-op-specific flags.

3. **Width-32 shuffles on wave64 (LOW)**: The code uses explicit width=32 shuffles which HIP supports on wave64 as subgroup ops. Should work correctly without modification.

4. **FULLMASK (0xffffffff) on wave64 (LOW)**: Using a 32-bit mask with width-32 shuffles is correct -- HIP interprets this as "all lanes in the 32-lane subgroup". No change needed.

5. **`lane_id < 31` / `lane_id < 30` guards in SEG_SHFL_SCAN (LOW)**: These lane checks are relative to the 32-lane logical warp, not the physical wavefront. Correct as-is.

6. **Thrust execution policy `thrust::cuda::par` (LOW)**: rocThrust uses the same `thrust::cuda::par` policy on HIP. Should work unchanged.

7. **Test data downloads (EXTERNAL)**: Tests use GraphDataset which downloads graph data (cora, citeseer, pubmed, ppi). Need network access during testing.

## File-by-file change list

### Files modified by torch hipify (automatic)
- src/cuda/spmm_cuda.cu
- src/cuda/spconv_cuda.cu  
- src/cuda/sparse_mapping.cu
- src/ge-spmm/*.cu
- src/sddmm/*.cu
- src/gspmm-fp/gspmm.cu
- include/cuda/*.cuh
- src/util/cuda_util.cuh

### Files requiring manual fixes

1. **include/cuda/csr2csc.cuh**: Replace cuSPARSE calls with hipSPARSE equivalents
   - `cusparseCreate` -> `hipsparseCreate`
   - `cusparseCsr2cscEx2_bufferSize` -> `hipsparseXcsr2csc_bufferSizeExt` or equivalent
   - `cusparseCsr2cscEx2` -> `hipsparseScsr2csc` (float version)
   - `cusparseHandle_t` -> `hipsparseHandle_t`
   - `CUSPARSE_*` enums -> `HIPSPARSE_*`
   - Guard with `#ifdef USE_ROCM` / `#else` to keep CUDA path

2. **src/cuda/spconv_cuda.cu**: Replace cuBLAS calls with hipBLAS equivalents
   - `cublasHandle_t` -> `hipblasHandle_t`
   - `cublasSetStream` -> `hipblasSetStream`
   - `cublasSetMathMode` -> remove or guard (hipBLAS has different math mode API)
   - `cublasGemmEx` -> `hipblasGemmEx`
   - `CUBLAS_*` enums -> `HIPBLAS_*`
   - `cublasComputeType_t` -> `hipblasComputeType_t`
   - Guard with `#ifdef USE_ROCM`

3. **include/cuda/cuda_util.cuh**: Replace cuSPARSE status check
   - `CUSPARSE_STATUS_SUCCESS` -> `HIPSPARSE_STATUS_SUCCESS`
   - `cusparseGetErrorString` -> custom or remove (hipSPARSE has different error API)
   - Guard with `#ifdef USE_ROCM`

4. **setup.py**: Add hipSPARSE/hipBLAS link flags for ROCm build
   - Line 42: `-lcusparse` -> `-lhipsparse` on ROCm
   - Add detection of ROCm build via `torch.version.hip`

5. **src/gspmm-fp/gspmm.cu**: Replace cuSPARSE includes and calls (if any)

6. **example/ge-spmm/spmm.cu, example/util/sp_util.hpp**: cuSPARSE benchmark code
   - Lower priority; examples can be updated or documented as CUDA-only

### Files unchanged
- Python code (dgsparse/*.py, test/*.py)
- CPU code (src/cpu/*.cpp)
- Build configs (Makefile, conda/)

## Build commands

### Configure and build (gfx90a)
```bash
# Ensure ROCm PyTorch is installed
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4

# Build the extension
cd projects/dgSPARSE-Lib/src
pip install -e .
```

### Build for other architectures
The architecture is determined by the ROCm torch installation and `PYTORCH_ROCM_ARCH` environment variable.

```bash
# For gfx1100
export PYTORCH_ROCM_ARCH=gfx1100
pip install -e .
```

## Test plan

### GPU tests
The test suite uses pytest and requires graph datasets:

```bash
cd projects/dgSPARSE-Lib/src/test

# Run all SpMM tests (12 parameterized: 4 datasets x 3 features)
pytest test_spmm.py -v

# Run CSR2CSC tests
pytest test_csr2csr.py -v

# Run DGL integration tests (requires dgl package)
pytest test_dgl.py -v

# Run all tests
pytest -v
```

### Test coverage
- `test_spmm.py`: SpMM sum/max/min/mean forward and backward
- `test_csr2csr.py`: CSR to CSC conversion
- `test_dgl.py`: Integration with DGL library
- `test_spconv.py`: Sparse convolution (uses cuBLAS, will need hipBLAS)
- `test_GIN.py`: GIN layer test

### Datasets used
- cora, citeseer, pubmed, ppi (downloaded automatically by utils.py GraphDataset)

### Non-GPU tests
None identified. All tests require GPU tensors.

## Open questions

1. **hipSPARSE csr2csc API**: Need to verify exact hipSPARSE function signature for CSR2CSC conversion. The `cusparseCsr2cscEx2` has a specific buffer-based API that may map to `hipsparseXcsr2csc` differently.

2. **hipBLAS tensor-op math mode**: cuBLAS `CUBLAS_TENSOR_OP_MATH` enables tensor cores. hipBLAS uses `HIPBLAS_GEMM_FLAGS_*` or `hipblasGemmAlgo_t`. Need to determine correct equivalent or if it can be omitted.

3. **Graph dataset availability**: Tests download datasets from remote. Verify network access during CI/validation.

4. **DGL package compatibility**: `test_dgl.py` requires Deep Graph Library. Need to verify DGL ROCm support or skip those tests.
