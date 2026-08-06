# Title: Add AMD GPU (HIP/ROCm) support

## Compare
https://github.com/NVIDIA/barney/compare/main...jeffdaily:barney:moat-port

## Body
This adds the ability to build and run barney's GPU rendering on AMD GPUs with HIP for ROCm, alongside the existing CUDA/OptiX path. The support is additive and gated behind a new `USE_HIP` CMake option that defaults to `OFF`, so the CUDA/OptiX build is unchanged when it is not enabled.

### What this does

Two ROCm backends are added under `rtcore/`, mirroring the existing CUDA backends:

- A software ray-tracing backend that reuses barney's CUDA device code compiled with HIP, traversing the cuBQL BVH (no vendor RT hardware required). The CUDA backend's GPU sources are retagged `LANGUAGE HIP` under `USE_HIP`; OptiX is NVIDIA-only and is not used in this configuration.
- A HIPRT hardware-traversal backend (`rtcore/hiprt/`, enabled with `-DBARNEY_BACKEND_HIPRT=ON`), the AMD analogue of the OptiX backend: it builds HIPRT scenes, runs the single trace kernel, and maps barney's anyHit/closestHit/intersect programs onto HIPRT's filter and custom-geometry function tables.

A small compatibility header (`rtcore/cudaCommon/cuda_to_hip.h`) maps the CUDA runtime spellings the shared code uses onto their HIP equivalents; it is included only on the HIP build and falls through to `<cuda_runtime.h>` otherwise, so the CUDA translation units are unaffected. `cmake/Findhiprt.cmake` locates the HIPRT SDK as a discovered dependency (never vendored), the same way barney finds OptiX.

### Building for AMD GPUs

Add `-DUSE_HIP=ON` and select the target architecture with `-DCMAKE_HIP_ARCHITECTURES=<arch>` (for example `gfx90a` for CDNA2, or `gfx1100` for RDNA3); when unset it defaults to `gfx90a`. To use HIPRT hardware traversal, also add `-DBARNEY_BACKEND_HIPRT=ON -Dhiprt_ROOT=<HIPRT-install>`. The README's "Building and Running" section documents the ROCm path alongside the CUDA/OptiX one.

### Validation

Validated on Linux (`gfx90a` CDNA2 and `gfx1100` RDNA3) and Windows (`gfx1201` RDNA4): both backends build and the bundled validation scenes (triangles, spheres, cylinders, instances, and the opaque/transparent path-traced scenes) render with pixel statistics identical to the CUDA reference; the HIPRT custom-geometry path is pixel-identical to the software backend.

This support targets ROCm. The CUDA/OptiX build is unchanged when `USE_HIP` is off.

This work was authored with assistance from Claude.
