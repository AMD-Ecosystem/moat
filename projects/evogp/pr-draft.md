# Title: Add AMD GPU (ROCm/HIP) support

## Compare
https://github.com/EMI-Group/evogp/compare/main...jeffdaily:evogp:moat-port

## Body
This PR adds AMD GPU support to EvoGP through ROCm/HIP, so the framework runs on AMD hardware in addition to NVIDIA. The CUDA kernels are compiled with HIP automatically by PyTorch's extension build system (`torch.utils.cpp_extension` hipifies the `.cu` sources at build time), so the port is small and the CUDA path is unchanged.

### What changed

- **Stack allocation in kernels.** The kernels used `alloca` to carve out per-thread scratch buffers (`stack`, `infos`, the generation and mutation stacks). On the AMD device compiler `alloca` is not the right tool for fixed-size on-stack scratch, so these were replaced with fixed-size local arrays sized by the same `MAX_STACK` compile-time constant. The arrays are identical in size and lifetime to the previous `alloca` allocations, so behavior is unchanged on both backends, and the local-array form is also clearer.
- **`device_launch_parameters.h` include.** This NVIDIA-only header (it just declares the `threadIdx`/`blockIdx` built-ins that the toolchain already provides) is guarded with `#ifndef USE_ROCM` so it is included only on the CUDA backend.
- **Device compiler flags.** Several `nvcc`/`ptxas`-only flags (`--ptxas-options`, `-Xptxas`, `-lineinfo`, `-maxrregcount`, `-lcudart`, and the CUDA spellings of relaxed-constexpr and fast math) are not accepted by `hipcc`. `setup.py` now selects the device-compiler flag list based on `torch.version.hip`, keeping the original CUDA flags intact for the NVIDIA build and passing a minimal, HIP-compatible set on ROCm.
- **`.gitignore`.** Ignore the `src/evogp/hip/` directory that the hipify step generates at build time.

### How to build on AMD

Install ROCm (including `hipcc`) and a ROCm build of PyTorch, then install EvoGP as usual:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install git+https://github.com/EMI-Group/evogp.git --no-build-isolation
```

PyTorch detects the ROCm backend and routes the kernels through HIP automatically; no source changes are required, and the `cuda` device strings used throughout the examples select the AMD GPU under ROCm. The same notes are added to the README Installation section.

### Validation

Built and tested on an AMD Instinct MI250X (gfx90a) with ROCm: the kernels compile via HIP and the example/test flows (including `python -m evogp.sr_test`) run on the AMD GPU. The NVIDIA CUDA path is functionally preserved: every HIP-specific change is guarded by `USE_ROCM`/`torch.version.hip`, and the one change shared by both backends (replacing `alloca` scratch with fixed-size local arrays of identical size and lifetime) is behavior-preserving.

This work was prepared with assistance from Claude.
