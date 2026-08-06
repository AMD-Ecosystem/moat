# ROCm Port Plan: 3dgrut

## Project
- **Name**: 3dgrut
- **Upstream**: https://github.com/nv-tlabs/3dgrut
- **Default branch**: main
- **Description**: 3D Gaussian Ray Tracing (3DGRT) and 3D Gaussian Unscented Transform (3DGUT) for neural radiance field rendering

## Existing AMD Support

**Status: No ROCm/HIP support exists upstream or in any known fork.**

Searches performed:
- Upstream repo docs grep (`grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/`): No matches for AMD/ROCm/HIP
- WebSearch for "3dgrut ROCm", "3dgrut AMD GPU", "3dgrut HIP": No results
- GitHub forks scan: No AMD/ROCm/HIP-related forks found
- No AMD-related issues or PRs in upstream

The project mentions Vulkan support in a separate project (vk_gaussian_splatting) which could work on AMD via Vulkan RT, but that is a distinct codebase, not this repository.

## Build Classification

**Classification: PyTorch Extension (Strategy B)**

Evidence:
- `setup.py` uses `setuptools` for the Python package wrapper
- `threedgrut/utils/jit.py` uses `torch.utils.cpp_extension.load()` for JIT compilation (lines 156-166)
- `threedgut_tracer/setup_3dgut.py` and `threedgrt_tracer/setup_3dgrt.py` use the jit.load() wrapper
- The project builds `.cu` files via PyTorch's CUDA extension machinery

## Port Strategy

**Decision: BLOCKED - Cannot port. This project has multiple NVIDIA-exclusive dependencies with no ROCm equivalents.**

### Critical Blockers (each independently blocking)

1. **OptiX (NVIDIA RT Cores)**
   - The 3DGRT (ray tracing) component is ENTIRELY built on NVIDIA OptiX 7.5
   - Files: `threedgrt_tracer/src/optixTracer.cpp` (1200+ lines), all kernels in `threedgrt_tracer/src/kernels/cuda/*.cu`
   - Uses OptiX intrinsics throughout: `optixTrace`, `optixGetLaunchIndex`, `optixGetPayload_*`, `optixSetPayload_*`, `optixReportIntersection`, `optixIgnoreIntersection`, `OptixAabb`, `OptixVisibilityMask`
   - 262 references to "optix" across the codebase
   - **HIPRT is NOT a drop-in replacement**: OptiX is deeply integrated with NVRTC runtime compilation and uses PTX; HIPRT uses a completely different API model (geometry/scene objects, different traversal API, bitcode linking instead of PTX)
   - A port would require a complete REWRITE of the ray tracing stack, not a translation

2. **tiny-cuda-nn**
   - Submodule dependency at `thirdparty/tiny-cuda-nn`
   - Contains NVIDIA-specific PTX inline assembly for atomic operations (`asm ("red.relaxed.gpu.global.add.f32...")` in `vec.h`)
   - Known open issue: [ROCm/HIP#3527](https://github.com/ROCm/HIP/issues/3527) documents the conversion attempt and blockers
   - No ROCm port exists or is planned

3. **NVRTC (NVIDIA Runtime Compilation)**
   - Used to JIT-compile OptiX kernels at runtime (`getPtxFromCuString` in optixTracer.cpp)
   - Produces PTX which only NVIDIA hardware can execute
   - No AMD equivalent for this OptiX+NVRTC pattern

4. **Slang Shader Compiler**
   - `slangtorch` dependency (v1.3.18) for compiling .slang kernel files to .cuh
   - Slang outputs CUDA code; it has no HIP/ROCm target
   - Used by both 3DGRT and 3DGUT components

5. **nvidia-ncore**
   - NVIDIA proprietary dataset/framework dependency
   - No AMD equivalent

### Partial Port Assessment: 3DGUT Only?

The 3DGUT (rasterization) component does NOT use OptiX directly. However:

- Still depends on tiny-cuda-nn (blocker #2)
- Still uses Slang-generated CUDA code (blocker #4)
- Has hardcoded warpSize = 32 in `threedgut_tracer/include/3dgut/renderer/gutRendererParameters.h:26`
- Uses warp intrinsics with 32-bit masks (`__shfl_*_sync`, `__ballot_sync`, `__all_sync` with `0xFFFFFFFF`)
- The hybrid "3DGRUT" mode requires both 3DGRT and 3DGUT working together

Even a 3DGUT-only port would require:
1. Porting tiny-cuda-nn (blocked upstream)
2. Adding HIP as a Slang compilation target or rewriting kernel generation
3. Wave64 warp-size abstraction throughout the tiled rendering code

### Alternative: Vulkan Path

The README mentions Vulkan support exists in a SEPARATE project: [vk_gaussian_splatting](https://github.com/nvpro-samples/vk_gaussian_splatting). That project may work on AMD via Vulkan ray tracing (RDNA2+), but:
- It is a distinct codebase, not a branch of 3dgrut
- It is an NVIDIA samples project, not this NVIDIA Research project
- Porting that project is a separate MOAT target, not a port of 3dgrut

## CUDA Surface Inventory (for reference)

### Files
- `threedgrt_tracer/src/optixTracer.cpp`: OptiX context setup, pipeline creation, NVRTC JIT
- `threedgrt_tracer/src/particlePrimitives.cu`: Particle geometry handling
- `threedgrt_tracer/src/kernels/cuda/*.cu`: 6 OptiX ray tracing kernels
- `threedgut_tracer/src/gutRenderer.cu`: Tile-based rasterizer (CUB)
- `threedgrut/optimizers/optimizers.cu`: Adam/MCMC optimizers
- `threedgrut/strategy/src/gaussian_mcmc.cu`: MCMC strategy
- `threedgrut_playground/src/*.cu`: Mesh buffers, hybrid tracer

### Dependencies
- OptiX 7.5 (RT cores)
- NVRTC (runtime PTX compilation)
- CUB (via tiny-cuda-nn)
- cuBLAS (via PyTorch)
- Slang -> CUDA shader generation

### Warp Intrinsics
- `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync` with 32-bit masks
- `__ballot_sync` with 0xFFFFFFFF
- `__all_sync` with 32-bit mask
- Hardcoded `WarpSize = 32`, `WarpMask = 0xFFFFFFFFU`

### Library Usage
- CUB: `cub::DeviceRadixSort::SortPairs`, `cub::DeviceScan::InclusiveSum`
- Streams/Events: Standard CUDA stream usage

## Risk List

1. **OptiX -> HIPRT rewrite**: Not a translation; fundamentally different APIs requiring complete reimplementation of ray tracing
2. **tiny-cuda-nn PTX assembly**: Requires rewriting NVIDIA-specific inline asm to HIP
3. **Slang CUDA output**: No HIP target; kernel generation would need replacement
4. **NVRTC pattern**: No equivalent for OptiX+NVRTC runtime compilation on AMD
5. **Wave64 warp size**: Hardcoded 32 throughout, would need abstraction

## Recommendation

**Set disposition: `cant-port`**

This project is fundamentally built on NVIDIA-exclusive technologies (OptiX RT cores, NVRTC, tiny-cuda-nn PTX, Slang->CUDA) with no feasible path to ROCm. Each blocker would require substantial reimplementation, not translation.

For AMD GPU users interested in Gaussian splatting, alternative projects to consider:
- [gsplat](https://github.com/nerfstudio-project/gsplat) - referenced by 3dgrut authors as "fast, modular, production-ready"; MOAT already ported this
- [OpenSplat](https://github.com/pierotofy/OpenSplat) - explicitly supports AMD/Metal
- [MCGS](https://github.com/MouseChannel/MCGS) - Vulkan compute shader based, cross-vendor
- [vk_gaussian_splatting](https://github.com/nvpro-samples/vk_gaussian_splatting) - Vulkan API, may work on RDNA2+ with Vulkan RT

## Build Commands (N/A)

Port not viable.

## Test Plan (N/A)

Port not viable.

## Open Questions

None - the blockers are clear and fundamental.
