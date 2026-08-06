# Title: Add AMD GPU support via HIP/ROCm

## Compare
https://github.com/CVCUDA/CV-CUDA/compare/main...jeffdaily:CV-CUDA:moat-port

## Body
This adds the ability to build CV-CUDA's GPU code for AMD GPUs with HIP for ROCm, alongside the existing CUDA build. The port is additive and gated behind a new `USE_HIP` CMake option that defaults to `OFF`, so the NVIDIA build is unchanged when it is not enabled.

### What this does

- Compiles the operators' `.cu` kernels and the CUDA-runtime-using host code with `hipcc`/HIP when `USE_HIP=ON`, while leaving the same sources on the NVIDIA toolchain when it is `OFF`.
- Adds a small ROCm/HIP compatibility layer under `cmake/hip/` that maps the CUDA runtime and library spellings CV-CUDA uses (`cudaStream_t`, `cudaMalloc`, cuBLAS/cuSOLVER/cuRAND status and enum names, the `cub::` namespace, `cudaDataType`, and so on) onto their HIP and ROCm equivalents (hipCUB, hipBLAS, hipSOLVER, rocRAND). The layer is force-included only on the HIP build's translation units and is never on the NVIDIA include path. CV-CUDA's own public `cuda<Op>Submit`/`cuda<Op>Create` API names are deliberately left untouched.
- Handles the AMD GPU differences the operators exercise: the 64-bit wavefront mask for `__shfl*_sync`, two-phase-lookup qualification clang/HIP requires, the math-library substitutions used by `OpFindHomography`, and floating-point contraction settings so HIP results match the CUDA build and the CPU reference within tolerance.

### Building for AMD GPUs

The HIP build reuses the existing `ci/build.sh` flow and CMake options:

```shell
ci/build.sh release build-rel -DUSE_HIP=1 -DCMAKE_HIP_ARCHITECTURES=gfx90a
```

`CMAKE_HIP_ARCHITECTURES` selects the target AMD architecture and defaults to `gfx90a` when unset; set it to your GPU, for example `gfx1100` for RDNA3 desktop cards. No source or CMake edits are needed to retarget. The build produces the same library and test layout as the CUDA build. The documentation is updated in the same place the CUDA build is documented: the Sphinx installation guide gains a "Building for AMD GPUs (ROCm)" section and the `USE_HIP` option, and the README gains a brief AMD-support note linking to it.

### Validation

The HIP build has been validated on Linux on the CDNA2 `gfx90a` (MI200 series) and RDNA3 `gfx1100` architectures, building the library and running the CV-CUDA C++ and Python GPU test suites. The default CUDA build (`USE_HIP=OFF`) has also been compiled with nvcc to confirm the NVIDIA path is unaffected.

This support targets Linux ROCm. CV-CUDA's existing native-Windows limitation is unchanged, so the Windows AMD configurations are out of scope here.

This work was authored with assistance from Claude.
