# Plan: torch-linear-assignment

## Project

- **Name:** torch-linear-assignment
- **Upstream:** https://github.com/ivan-chai/torch-linear-assignment
- **Default branch:** main
- **Description:** Batch computation of the linear assignment problem on GPU using the Hungarian algorithm (Auction algorithm variant from Crouse 2016).

## Existing AMD support

**None found.** Searches performed:
- Grepped upstream docs (`grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/`): No references except author name "Karpukhin".
- Web search for "torch-linear-assignment ROCm/AMD/HIP": No results specific to this project.
- Checked 20 GitHub forks via `gh api repos/ivan-chai/torch-linear-assignment/forks`: No ROCm/HIP branches (only main and patch-1 branches).
- Searched upstream PRs/issues: No ROCm/HIP related submissions.
- No rocm/hip branches on upstream.

**Decision:** Proceed with a from-scratch port. No existing AMD support to improve or validate.

## Build classification

**pytorch extension (Strategy B)**

Evidence (setup.py lines 6-7, 14-21):
```python
import torch.utils.cpp_extension as torch_cpp_ext
...
torch_cpp_ext.CUDAExtension(
    "torch_linear_assignment._backend",
    [
        "src/torch_linear_assignment_cuda.cpp",
        "src/torch_linear_assignment_cuda_kernel.cu"
    ],
    ...
)
```

The project uses `torch.utils.cpp_extension.CUDAExtension` with `BuildExtension` -- this is the canonical PyTorch extension pattern.

## Port strategy

**Strategy B: pytorch extension (torch hipify)**

Rationale:
- The project is a PyTorch C++/CUDA extension built via setup.py.
- Torch's hipify infrastructure automatically translates `CUDAExtension` sources when building against a ROCm PyTorch.
- The CUDA surface is minimal and clean -- no warp intrinsics, no CUDA libraries, no textures.
- This should be a zero-source-edit port: build against ROCm PyTorch and let torch hipify do the translation.

## CUDA surface inventory

### Kernel attributes
| File | Line | Construct |
|------|------|-----------|
| torch_linear_assignment_cuda_kernel.cu | 56 | `__device__ __forceinline__ array_fill` |
| torch_linear_assignment_cuda_kernel.cu | 65 | `__device__ __forceinline__ augmenting_path_cuda` |
| torch_linear_assignment_cuda_kernel.cu | 128 | `__device__ __forceinline__ solve_cuda_kernel` |
| torch_linear_assignment_cuda_kernel.cu | 178 | `__global__ solve_cuda_kernel_batch` |

### CUDA includes
| File | Line | Include |
|------|------|---------|
| torch_linear_assignment_cuda_kernel.cu | 10 | `#include <cuda.h>` |
| torch_linear_assignment_cuda_kernel.cu | 11 | `#include <cuda_runtime.h>` |

### CUDA runtime calls
| File | Line | Function |
|------|------|----------|
| torch_linear_assignment_cuda_kernel.cu | 25 | `cudaGetDeviceProperties` |
| torch_linear_assignment_cuda_kernel.cu | 213 | `cudaSetDevice` |
| torch_linear_assignment_cuda_kernel.cu | 238 | `at::cuda::getCurrentCUDAStream` (ATen) |
| torch_linear_assignment_cuda_kernel.cu | 251-252 | `cudaGetLastError`, `cudaSuccess` |

### PyTorch/ATen constructs
| File | Line | Construct |
|------|------|-----------|
| torch_linear_assignment_cuda_kernel.cu | 148 | `CUDA_KERNEL_ASSERT` |
| torch_linear_assignment_cuda_kernel.cu | 275 | `AT_DISPATCH_FLOATING_TYPES` |
| torch_linear_assignment_cuda.cpp | 5-7 | `CHECK_CUDA`, `CHECK_CONTIGUOUS` |

### Warp intrinsics
**None.** No `__shfl*`, `__ballot`, `__any`, `__all`, `warpSize`, `__activemask`.

### CUDA libraries
**None.** No cuBLAS, cuFFT, cuRAND, cuSPARSE, Thrust, CUB.

### Textures/Surfaces
**None.**

### Special memory
**None.** No `cudaMallocManaged`, `cudaHostAlloc`, pinned memory.

### Streams/Events
Uses `at::cuda::getCurrentCUDAStream` (ATen abstraction, hipified automatically).

## Risk list

| Risk | Severity | Mitigation |
|------|----------|------------|
| SMPCores function (lines 22-52) | Low | Returns CUDA cores per SM based on `devProp.major/minor`. On AMD this will fall through to default 128. The value is only used for block size selection -- suboptimal but functional. Could add AMD arch detection later for tuning. |
| No warp-size issues | None | Algorithm is per-thread sequential Hungarian/auction; no warp-collective operations, no hardcoded 32. |
| CHECK_CUDA macro | Low | Uses `device().is_cuda()` which returns true for HIP on ROCm PyTorch. No change needed. |
| CUDA_KERNEL_ASSERT | None | Hipified to HIP_KERNEL_ASSERT automatically. |

**This is a low-risk port.** The algorithm is inherently sequential per batch element (Hungarian algorithm), with no warp-level parallelism or CUDA-specific optimizations. Each thread processes one assignment problem independently.

## File-by-file change list

**Expected: Zero source changes required.**

The torch hipify infrastructure (invoked automatically when building `CUDAExtension` against ROCm PyTorch) will translate:
- `#include <cuda.h>` -> `#include <hip/hip_runtime.h>`
- `#include <cuda_runtime.h>` -> (removed, redundant with above)
- `cudaGetDeviceProperties` -> `hipGetDeviceProperties`
- `cudaSetDevice` -> `hipSetDevice`
- `cudaGetLastError` -> `hipGetLastError`
- `cudaSuccess` -> `hipSuccess`
- `cudaDeviceProp` -> `hipDeviceProp_t`
- `CUDA_KERNEL_ASSERT` -> `HIP_KERNEL_ASSERT`

The `at::cuda::*` ATen abstractions are already HIP-aware in ROCm PyTorch.

If manual intervention is needed (unlikely):
1. setup.py line 9: The condition `torch.backends.cuda.is_built()` should work on ROCm PyTorch (returns True when HIP backend is built).

## Build commands

### Configure and build for gfx90a
```bash
# Ensure ROCm PyTorch environment is active
export HIP_VISIBLE_DEVICES=3

# Build with pip (in-place for development)
pip install --no-build-isolation -e .

# Or build wheel
pip wheel --no-build-isolation .
```

### Verify HIP build
```bash
python -c "import torch_linear_assignment._backend as b; print('has_cuda:', b.has_cuda())"
```

## Test plan

### GPU tests (real gfx90a validation required)
```bash
# Run full test suite
pytest tests/

# Individual tests:
# - test_assignment.py::TestAssignment::test_simple - basic correctness
# - test_assignment.py::TestAssignment::test_cuda_equal_to_cpu - GPU vs CPU equivalence
# - test_to_indices.py::TestAssignmentToIndices::test_compare_to_scipy - scipy compatibility

# Benchmark (confirms GPU path is exercised)
python tests/benchmark.py
```

### Non-GPU tests that must not regress
The same tests run on CPU as fallback:
- `test_assignment.py::TestAssignment::test_simple` (on CPU device)
- `test_to_indices.py::TestAssignmentToIndices::test_compare_to_scipy` (on CPU)

### Validation criteria
1. `test_cuda_equal_to_cpu` passes: GPU output matches scipy-validated CPU output.
2. `test_compare_to_scipy` passes: Output format matches scipy's `linear_sum_assignment`.
3. Benchmark shows GPU timing (not fallback warning "No CUDA backend available").
4. Run-to-run determinism (the algorithm is deterministic).

## Open questions

1. **SMPCores tuning:** The `SMPCores()` function returns CUDA cores per SM for NVIDIA archs. On AMD it defaults to 128. Is this block-size selection optimal for gfx90a? Likely fine for correctness; performance tuning could come later.

2. **Force CUDA check in setup.py:** Line 9 checks `torch.backends.cuda.is_built()`. Need to verify this returns True on ROCm PyTorch -- it should, but the porter should confirm.

3. **CI workflows:** The upstream CI tests CPU-only. No GPU CI changes needed for the port.
