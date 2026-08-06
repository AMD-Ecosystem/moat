# FastGeodis ROCm Port Plan

## Project
- **Name:** FastGeodis
- **Upstream:** https://github.com/masadcv/FastGeodis
- **Default branch:** main
- **Description:** Fast Implementation of Generalised Geodesic Distance Transform for CPU (OpenMP) and GPU (CUDA), published in JOSS 2022

## Existing AMD support
- **Status:** None
- **Determination:** No existing AMD/ROCm port found. Searched:
  - Upstream README/docs: no AMD/ROCm/HIP mentions
  - Web search: no FastGeodis ROCm or AMD GPU results
  - GitHub forks: 16 forks, none under ROCm/AMD/GPUOpen orgs, none with rocm/hip/amd in name
  - ROCm/AMD GitHub orgs: no FastGeodis fork exists
- **Decision:** Proceed with a fresh HIP port. This is a valuable addition to the ROCm ecosystem -- geodesic distance transforms are used in medical imaging and interactive segmentation pipelines.

## Build classification
- **Type:** PyTorch extension (torch-extension)
- **Evidence:**
  - `setup.py` uses `torch.utils.cpp_extension` imports: `CUDAExtension`, `CppExtension`, `BuildExtension` (lines 19-23)
  - Sources are `.cu` (CUDA) and `.cpp` (CPU) compiled via PyTorch's extension mechanism
  - CUDA build enabled when `torch.cuda.is_available()` or `FORCE_CUDA=1` (line 24)

**ext_type:** `torch-extension`

## Port strategy
**Strategy B: PyTorch extension**

Rationale:
- PyTorch's `CUDAExtension` automatically invokes `torch.utils.hipify` on `.cu`/`.cuh` sources when built against a ROCm PyTorch
- The CUDA surface is minimal and uses only standard CUDA runtime APIs that have 1:1 HIP equivalents
- No manual symbol renaming needed; hipify handles the `cudaMemcpyToSymbol` -> `hipMemcpyToSymbol` etc.

The port will:
1. Build against ROCm PyTorch (the extension machinery handles hipification)
2. No source modifications expected for basic functionality (the CUDA surface is clean)
3. Minor fixes may be needed if any fault-class issues appear during validation

## CUDA surface inventory

### Kernels (fastgeodis_cuda.cu)
| Item | Lines | HIP equivalent | Notes |
|------|-------|----------------|-------|
| `__constant__ float local_dist2d[3]` | 47 | `__constant__` (identical) | Constant memory arrays |
| `__constant__ float local_dist3d[3*3]` | 48 | `__constant__` (identical) | |
| `__device__ l1distance_cuda()` | 50-53 | `__device__` (identical) | Simple device helper |
| `__global__ geodesic_updown_single_row_pass_kernel` | 56-129 | `__global__` (identical) | 2D row-pass kernel (template) |
| `__global__ geodesic_updown_single_row_pass_ptr_kernel` | 131-202 | `__global__` (identical) | 2D row-pass kernel (raw ptr) |
| `__global__ geodesic_frontback_single_plane_pass_kernel` | 321-405 | `__global__` (identical) | 3D plane-pass kernel (template) |
| `__global__ geodesic_frontback_single_plane_pass_ptr_kernel` | 407-493 | `__global__` (identical) | 3D plane-pass kernel (raw ptr) |

### Runtime API calls
| Call | Lines | HIP equivalent | Notes |
|------|-------|----------------|-------|
| `cudaMemcpyToSymbol(local_dist2d, ...)` | 220 | `hipMemcpyToSymbol` | Auto-hipified |
| `cudaMemcpyToSymbol(local_dist3d, ...)` | 529 | `hipMemcpyToSymbol` | Auto-hipified |

### Synchronization
| Item | Count | Notes |
|------|-------|-------|
| `__syncthreads()` | 4 | Block-level sync, identical in HIP |

### Threading model
- `THREAD_COUNT = 256` (fixed block size)
- 1D blocks for 2D kernels (width direction)
- 2D blocks (16x16) for 3D kernels (width x height)
- No warp-level operations (`__shfl*`, `__ballot`, warp-size assumptions)

### Libraries
- `torch/extension.h` - PyTorch C++ API
- `c10/cuda/CUDAGuard.h` - Device guard (maps to c10/hip/HIPGuard on ROCm)
- `cuda.h`, `cuda_runtime.h` - Auto-mapped to HIP headers

### NOT used (good news)
- No textures/surfaces
- No warp intrinsics (`__shfl*`, `__ballot`, `warpSize`, `__activemask`)
- No cuBLAS/cuFFT/cuRAND/cuSPARSE
- No Thrust/CUB
- No pinned/managed memory beyond what PyTorch handles
- No explicit streams/events (PyTorch manages this)

## Risk list

### Low risk (expected to work out of the box)
1. **Clean CUDA surface** - Only standard kernel launches and `cudaMemcpyToSymbol`. PyTorch hipify handles all of this.
2. **No warp-size dependencies** - No hardcoded 32, no warp intrinsics. The algorithm is a row/plane raster scan with per-thread independent work followed by `__syncthreads()`.
3. **Fixed block sizes** - 256 threads (1D) or 16x16 (2D) are valid on all AMD targets.

### Medium risk (may need attention)
1. **PyTorch ROCm build compatibility** - Need to verify the extension builds cleanly against ROCm PyTorch. The `c10/cuda/CUDAGuard.h` include should auto-map but worth confirming.
2. **`AT_DISPATCH_FLOATING_TYPES` macro** - The templated kernel path (controlled by `#if USE_PTR 0`) uses this macro. Currently `USE_PTR=1` so the raw-pointer path is active, which is simpler.

### No risk
- No FP rounding precision paths (no `__fsqrt_rn`, no FP-equality branching)
- No layered textures or texture filtering
- No `__smid()` usage
- No stream callback issues
- No rule-of-five RAII concerns

## File-by-file change list

### Likely no changes needed
- `FastGeodis/fastgeodis_cuda.cu` - PyTorch hipify should handle all transformations automatically
- `FastGeodis/fastgeodis.cpp` - Host dispatch code, no CUDA-specific content
- `FastGeodis/fastgeodis.h` - Header declarations, no changes needed
- `FastGeodis/common.h` - Validation helpers, no CUDA-specific content
- `FastGeodis/*.cpp` (CPU implementations) - No changes needed
- `setup.py` - No changes needed; CUDAExtension auto-detects ROCm PyTorch

### Potential changes (if issues arise during build)
- May need to add `USE_ROCM` guards if any hipify edge cases appear
- May need to adjust includes if `c10/cuda/CUDAGuard.h` does not auto-map

## Build commands

### Prerequisites
- ROCm 7.2+ installed
- ROCm PyTorch installed (e.g., `pip install torch --index-url https://download.pytorch.org/whl/rocm6.3`)
- Development dependencies: `pip install -r requirements-dev.txt`

### Build for gfx90a
```bash
cd projects/FastGeodis/src
pip install -e . --no-build-isolation
```

The PyTorch extension machinery will:
1. Detect ROCm PyTorch (`torch.version.hip` is set)
2. Invoke hipify on `FastGeodis/fastgeodis_cuda.cu`
3. Compile with hipcc targeting the detected GPU architecture

To force a specific architecture:
```bash
HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx90a pip install -e . --no-build-isolation
```

### Verify build
```bash
python -c "import FastGeodis; print(FastGeodis.__file__)"
python -c "import torch; print(torch.cuda.is_available())"
```

## Test plan

### GPU tests (primary validation)
The test suite in `tests/` exercises both CPU and CUDA paths. With ROCm PyTorch, `torch.cuda.is_available()` returns True and tests run on the AMD GPU.

```bash
cd projects/FastGeodis/src
python -m pytest tests/ -v
# or
python -m unittest discover -s tests -v
```

Key test files:
- `test_fastgeodis.py` - Core geodesic distance transform (2D/3D, signed/unsigned, GSF)
- `test_toivanen.py` - Toivanen algorithm variants (CPU-only)
- `test_pixelqueue.py` - Pixel queue algorithm (CPU-only)
- `test_fastmarch.py` - Fast marching algorithm (CPU-only)

The GPU tests use parameterized configs:
- `CONF_2D_CUDA`: 2D tests on CUDA/HIP at base dimensions 16, 32, 64
- `CONF_3D_CUDA`: 3D tests on CUDA/HIP at base dimensions 16, 32, 64

### Non-GPU tests (must not regress)
CPU implementations (Toivanen, pixelqueue, fastmarch) are exercised by the same test suite. These should pass unchanged.

### Validation criteria
1. All `test_fastgeodis.py` tests pass on CUDA device (now HIP)
2. CPU tests continue to pass
3. Output values match expected (zeros input -> zeros output, ones mask -> v*ones, euclidean distance within tolerance)

### Sample demos
```bash
python samples/demo2d.py
python samples/demo3d.py
python samples/demo2d_signed.py
```

## Open questions

1. **PyTorch ROCm version compatibility** - Need to verify which ROCm PyTorch versions are supported. The extension uses standard PyTorch C++ API so should work with recent versions.

2. **Windows ROCm** - The HIP SDK for Windows may have different PyTorch extension behavior. This is a follower-platform concern (gfx1101, gfx1201).

3. **Performance tuning** - The current block sizes (256, 16x16) are reasonable defaults. AMD-specific tuning (occupancy, wavefront utilization) could improve performance but is not required for correctness.
