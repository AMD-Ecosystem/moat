# Title: Add AMD GPU (ROCm/HIP) support to the splat_cuda extension

## Compare
https://github.com/joeyan/gaussian_splatting/compare/main...jeffdaily:gaussian_splatting:moat-port

## Body
This adds AMD GPU support to the `splat_cuda` PyTorch extension so the forward and backward rasterizer build and run on ROCm in addition to CUDA. PyTorch's `CUDAExtension` already hipifies the `.cu`/`.cuh` sources at build time, so the same `setup.py` drives both paths and the CUDA build is byte-identical to before. Only two source-level changes were needed.

### Cooperative-groups tile reduction (`src/render_backward.cu`)
The backward rasterizer reduces per-tile partial gradients across a 32-thread tile with `cg::reduce(warp, grad, cg::plus<T>())` from `<cooperative_groups/reduce.h>`. ROCm ships `<hip/hip_cooperative_groups.h>` (`this_thread_block`, `tiled_partition<32>`, `thread_rank`, `sync`, and tile `shfl_xor` all work) but has no `<cooperative_groups/reduce.h>`, `cg::reduce`, or `cg::plus`. Under `USE_ROCM` the missing include is guarded out and the six `cg::reduce` calls go through a `WARP_REDUCE_SUM` macro backed by a templated `warpReduceSum` that folds the value over the tile with `warp.shfl_xor(val, offset)` for `offset = tile_size/2` down to `1`, leaving the tile-wide sum in every lane (the same all-reduce semantics `cg::reduce(plus)` provides). The helper is templated over `T` because the kernel is instantiated for both `float` and `double`, which the float64 gradchecks exercise. A `thread_block_tile<32>` shuffle uses the tile width (32) as the shuffle width, so it stays within the tile's 32 lanes on a 64-wide wavefront; a 256-thread (16x16) block splits into four wavefronts each holding two independent 32-lane tiles, matching the existing tile granularity with no lane-math rework. The CUDA path keeps calling `cg::reduce` unchanged.

### Windows ROCm build (`setup.py`)
On Windows with ROCm, an MSVC-compiled `bindings.cpp` cannot link `c10::ValueError` and related pybind11/c10 exception types from the clang-built `c10.dll` (an inherited-constructor ABI mismatch: clang-cl does not emit `dllexport` symbols for the inherited constructors of `c10::Error` subclasses). When `os.name == 'nt'` and `torch.version.hip is not None`, `setup.py` copies `bindings.cpp` to a generated `bindings_winhip.cu` so `BuildExtension` routes it through the HIP compiler, which shares the same ABI as `c10.dll`. The generated file is gitignored. The Linux and CUDA paths are unchanged. The same root cause was fixed upstream in pytorch/pytorch#175340 (explicit exported constructors for the affected `c10::Error` subclasses); this workaround keeps the extension building on PyTorch releases from before that fix.

### Build
Install a ROCm build of PyTorch (see pytorch.org), then build the extension exactly as on CUDA, selecting the target AMD architecture:
```
PYTORCH_ROCM_ARCH=gfx90a python setup.py build_ext && python setup.py install
```
No CUDA toolkit is required on the ROCm path. The README Installation section documents this alongside the CUDA instructions.

### Validation
Built and tested on AMD Instinct (gfx90a) with a ROCm build of PyTorch, and on AMD Radeon RX 9070 XT (gfx1201) on Windows with a ROCm build of PyTorch. The decisive checks for the reduce replacement are the float64 autograd gradchecks, which compare analytic gradients against central finite differences:
```
cd test
python -m unittest test_rasterize_autograd test_cuda_autograd_functions -v
```
All 16 gradchecks pass deterministically (SH degrees {0,4,9,16} with and without background; projection/sigma/jacobian/conic/SH). The forward no-SH render is bitwise exact to the upstream golden and bitwise-deterministic across runs.
