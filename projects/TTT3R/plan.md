# TTT3R Port Plan

## Project

- **Name**: TTT3R (3D Reconstruction as Test-Time Training)
- **Upstream**: https://github.com/Inception3D/TTT3R
- **Default branch**: main

## Existing AMD support

**Assessment**: No existing AMD/ROCm/HIP support found.

Searches performed:
- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/` -- no matches
- WebSearch for "TTT3R ROCm", "TTT3R AMD GPU", "CroCo curope ROCm" -- no results
- GitHub fork scan (`gh api repos/Inception3D/TTT3R/forks`) -- no AMD/ROCm/HIP forks
- GitHub issues/PRs search for AMD-related keywords -- no results

The upstream README does not mention AMD support or link to any AMD-related fork. The project merges PRs directly (it does not link to platform forks), so an upstream PR is the appropriate delivery vehicle.

**Decision**: Proceed with a fresh HIP port.

## Build classification

**Type**: PyTorch extension (Strategy B)

**Evidence**:
- `src/croco/models/curope/setup.py` lines 6-7:
  ```python
  from torch.utils.cpp_extension import BuildExtension, CUDAExtension
  ```
- `setup.py` line 20-27: Uses `CUDAExtension` with `torch.utils.cpp_extension.BuildExtension`
- Build command in README: `cd src/croco/models/curope/ && python setup.py build_ext --inplace`

No CMakeLists.txt exists. This is a pure PyTorch extension built via setuptools.

## Port strategy

**Strategy B: PyTorch extension** -- let torch hipify the extension at build time.

**Rationale**:
- The extension uses `torch.utils.cpp_extension.CUDAExtension`, which automatically hipifies `.cu` sources when building against a ROCm PyTorch
- The CUDA kernel is simple (no warp intrinsics, no cuBLAS/cuFFT, no textures)
- The only required changes are:
  1. Replace CUDA-specific nvcc flags (`--ptxas-options`, `--use_fast_math`) with HIP-compatible equivalents
  2. Handle `cudaGetLastError`/`cudaSuccess`/`cudaGetErrorString` in the CHECK_KERNEL macro (torch hipify handles these automatically)
  3. The `torch.cuda.get_gencode_flags()` call needs to be skipped or handled differently on ROCm (it fails on HIP)

The kernel (`kernels.cu`) uses only:
- `__global__` kernel
- `__syncthreads()`
- `extern __shared__` for shared memory
- `torch::PackedTensorAccessor32` (PyTorch API, portable)
- `cosf`, `sinf`, `powf` (standard math, portable)

No warp-level intrinsics, no CUDA libraries, no textures -- this is the cleanest possible extension.

## CUDA surface inventory

### File: `src/croco/models/curope/kernels.cu`

**Includes**:
- `<torch/extension.h>` -- portable via PyTorch
- `<cuda.h>` -- hipified to HIP headers
- `<cuda_runtime.h>` -- hipified to HIP headers
- `<vector>` -- standard C++

**CUDA APIs**:
| Symbol | Line | HIP equivalent | Notes |
|--------|------|----------------|-------|
| `cudaGetLastError` | 14 | `hipGetLastError` | Auto-hipified |
| `cudaSuccess` | 14 | `hipSuccess` | Auto-hipified |
| `cudaGetErrorString` | 14 | `hipGetErrorString` | Auto-hipified |

**Kernel intrinsics**:
| Feature | Line | Notes |
|---------|------|-------|
| `__global__` | 18 | Portable |
| `extern __shared__` | 33 | Portable |
| `__syncthreads()` | 46, 71 | Portable |
| `blockIdx.x` | 36, 53 | Portable |
| `threadIdx.x` | 44, 49, 50, 53, 70, 77, 78, 79, 80 | Portable |

**Math functions**:
- `powf`, `cosf`, `sinf` (lines 45, 54, 55) -- portable device math

### File: `src/croco/models/curope/setup.py`

**CUDA-specific**:
| Line | Code | Issue |
|------|------|-------|
| 5 | `from torch import cuda` | Used for `cuda.get_gencode_flags()` |
| 9 | `cuda.get_gencode_flags()` | Fails on ROCm (no CUDA gencode on AMD) |
| 28 | `nvcc=["-O3", "--ptxas-options=-v", "--use_fast_math"]` | nvcc-specific flags |

### File: `src/croco/models/curope/curope.cpp`

No CUDA APIs -- pure C++ with `torch::Tensor` types. Portable as-is.

## Risk list

1. **Low risk**: `setup.py` uses `torch.cuda.get_gencode_flags()` which fails on ROCm. Fix: conditionally skip or use `PYTORCH_ROCM_ARCH` environment variable for HIP.

2. **Low risk**: nvcc-specific compile flags (`--ptxas-options=-v`, `--use_fast_math`). Fix: use HIP-compatible flags (`-ffast-math` for HIP clang) conditionally.

3. **No warp-size risk**: The kernel uses `__syncthreads()` which is block-level (portable across wave64/wave32). No `__shfl*`, `__ballot`, or `warpSize` usage.

4. **No texture/surface risk**: No texture or surface memory usage.

5. **No library risk**: No cuBLAS, cuFFT, cuRAND, cuSPARSE, Thrust, or CUB usage.

## File-by-file change list

### `src/croco/models/curope/setup.py`

1. Import `torch.version` to detect ROCm
2. Conditionally set `extra_compile_args`:
   - On ROCm: skip `cuda.get_gencode_flags()`, use HIP-compatible flags
   - On CUDA: keep existing nvcc flags
3. Use `HIPExtension` on ROCm (or let `CUDAExtension` handle it automatically with newer PyTorch)

### `src/croco/models/curope/kernels.cu`

No changes required -- PyTorch's hipify will automatically translate:
- `#include <cuda.h>` -> `#include <hip/hip_runtime.h>`
- `#include <cuda_runtime.h>` -> `#include <hip/hip_runtime.h>`
- `cudaGetLastError` -> `hipGetLastError`
- `cudaSuccess` -> `hipSuccess`
- `cudaGetErrorString` -> `hipGetErrorString`

### `src/croco/models/curope/curope.cpp`

No changes required -- no CUDA-specific code.

## Build commands

### Configure + build for gfx90a

```bash
# Create conda environment with ROCm PyTorch
conda create -n ttt3r-rocm python=3.11 cmake=3.14.0
conda activate ttt3r-rocm

# Install ROCm PyTorch (adjust version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3

# Install other dependencies
pip install -r requirements.txt

# Build curope extension
cd src/croco/models/curope/
PYTORCH_ROCM_ARCH="gfx90a" python setup.py build_ext --inplace
cd ../../../../
```

### Alternative: use HIP_VISIBLE_DEVICES if multiple GPUs

```bash
HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH="gfx90a" python setup.py build_ext --inplace
```

## Test plan

### GPU tests (primary validation)

1. **Extension build test**: Verify the curope module builds and imports successfully
   ```bash
   cd src/croco/models/curope/
   python setup.py build_ext --inplace
   python -c "import curope; print('curope imported successfully')"
   ```

2. **Unit test for RoPE kernel**: Create a simple test comparing GPU vs CPU RoPE output
   ```bash
   python -c "
   import torch
   from src.croco.models.curope import cuRoPE2D
   
   # Test on GPU
   rope = cuRoPE2D(freq=100.0, F0=1.0)
   B, N, H, D = 2, 16, 8, 64
   tokens_gpu = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
   positions = torch.randint(0, 10, (B, N, 2), device='cuda', dtype=torch.int64)
   
   # Save input for comparison
   tokens_cpu = tokens_gpu.clone().cpu()
   positions_cpu = positions.cpu()
   
   # Run GPU kernel
   out_gpu = rope(tokens_gpu.clone(), positions)
   
   # Run CPU path (curope.cpp has rope_2d_cpu)
   tokens_cpu_test = tokens_cpu.clone().transpose(1, 2).contiguous()
   import curope
   curope.rope_2d(tokens_cpu_test, positions_cpu, 100.0, 1.0)
   
   # Compare
   diff = (out_gpu.cpu() - tokens_cpu_test.transpose(1, 2)).abs().max().item()
   print(f'Max difference GPU vs CPU: {diff}')
   assert diff < 1e-5, f'GPU vs CPU mismatch: {diff}'
   print('RoPE kernel test PASSED')
   "
   ```

3. **Inference demo**: Run the demo script on example video
   ```bash
   # Download checkpoint first
   cd src && gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link && cd ..
   
   # Run inference (short test)
   python demo.py --model_path src/cut3r_512_dpt_4_64.pth --size 512 \
       --seq_path examples/taylor.mp4 --output_dir tmp/test --port 8080 \
       --model_update_type ttt3r --frame_interval 10 --reset_interval 50 \
       --downsample_factor 1000 --vis_threshold 10.0
   ```

### Non-GPU tests (must not regress)

1. **CPU fallback test**: The extension has a CPU implementation -- verify it works
   ```bash
   python -c "
   import torch
   from src.croco.models.curope import cuRoPE2D
   
   rope = cuRoPE2D(freq=100.0, F0=1.0)
   B, N, H, D = 2, 16, 8, 64
   tokens = torch.randn(B, H, N, D, device='cpu', dtype=torch.float32)
   positions = torch.randint(0, 10, (B, N, 2), device='cpu', dtype=torch.int64)
   
   out = rope(tokens.clone(), positions)
   print('CPU fallback test PASSED')
   "
   ```

2. **Import test**: Verify all Python modules import without GPU
   ```bash
   CUDA_VISIBLE_DEVICES="" python -c "
   # These should import without GPU
   from src.croco.models.curope.curope2d import cuRoPE2D
   print('Module imports OK')
   "
   ```

### Evaluation tests (optional, requires datasets)

The full evaluation suite (eval/) requires external datasets (ScanNet, TUM-dynamics, etc.) and is not suitable for automated validation. Validation focuses on the core curope kernel and inference demo.

## Open questions

1. **PyTorch version**: The ROCm PyTorch wheel selection (rocm6.3 vs rocm6.2) should match the ROCm version on the gfx90a host. Confirm available versions at https://download.pytorch.org/whl/rocm6.3/.

2. **Model checkpoint**: The demo requires downloading a ~2GB checkpoint from Google Drive. For validation, this download step should be automated or the checkpoint cached.

3. **gdown on headless**: The `gdown` tool may require browser authentication for large files. Alternative: use `curl` with direct download link or pre-stage the checkpoint.
