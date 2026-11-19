# faster-gaussian-splatting ROCm/HIP port plan (lead: linux-gfx90a)

## Project
- Upstream: nerficg-project/faster-gaussian-splatting @ main
- CVPR 2026 "Faster-GS: Analyzing and Improving Gaussian Splatting Optimization" -- a highly-optimized, research-friendly 3DGS implementation (2-5x faster than other research codebases, less VRAM)
- This is DISTINCT from gsplat (nerfstudio-project/gsplat) which is already ported -- different organizations, different codebases, different rasterizer implementations
- ext_type: **torch-extension** -> **Strategy B** (torch hipifies .cu/.cuh at build time; do NOT hand-add a compat header or hand-rename symbols)

## Existing AMD support
- **None.** No AMD/ROCm/HIP references in README, docs, or source
- No ROCm-related GitHub issues or PRs
- No AMD-related forks found (checked via gh api repos/.../forks)
- WebSearch found no separate AMD port or ROCm documentation
- Upstream accepts external contributions (standard open-source repo)
- **Decision: proceed with Strategy B port**

## Build classification: torch-extension
Evidence:
- `FasterGSCudaBackend/setup.py` uses `torch.utils.cpp_extension.CUDAExtension` + `BuildExtension`
- `pyproject.toml` lists torch as build dependency
- Builds via `pip install .../FasterGSCudaBackend --no-build-isolation`

## Port strategy: B (torch-hipify)
Rationale: Building a CUDAExtension on a ROCm torch automatically runs torch.utils.hipify on the extension's .cu/.cuh sources. Keep sources in CUDA spelling; hipify translates them. Fix only what hipify cannot (warp size, CUDA intrinsics) in source, guarded by USE_ROCM.

## CUDA surface inventory

### Files (~3720 LOC total)
```
FasterGSCudaBackend/FasterGSCudaBackend/
├── rasterization/
│   ├── src/forward.cu, backward.cu, inference.cu, pruning_scores.cu, rasterization_api.cu
│   └── include/kernels_forward.cuh, kernels_backward.cuh, kernels_inference.cuh,
│              kernels_pruning_scores.cuh, kernel_utils.cuh, sh_utils.cuh,
│              rasterization_config.h, buffer_utils.h, ...
├── adam/src/adam.cu
├── densification/src/densification_api.cu, mcmc.cu
├── filter3d/src/filter3d.cu
├── utils/helper_math.h, utils.h, torch_utils.h
└── torch_bindings/bindings.cpp, *.py
```

### Cooperative Groups (tiled_partition<32>)
Multiple kernels use `cg::tiled_partition<32>(block)` for warp-level operations:
- `kernels_forward.cuh`: preprocess_cu, create_instances_cu
- `kernels_backward.cuh`: blend_backward_cu
- `kernels_inference.cuh`: preprocess_cu, create_instances_cu
- `kernels_pruning_scores.cuh`: preprocess_cu, create_instances_cu
- `kernel_utils.cuh`: compute_exact_n_touched_tiles

Operations used:
- `warp.thread_rank()` -> lane index
- `warp.ballot(pred)` -> ballot
- `warp.shfl(val, lane)` -> shuffle
- `warp.shfl_up(val, delta)` -> shuffle up
- `warp.shfl_xor(val, lane_mask)` -> shuffle xor
- `warp.any(pred)` -> any predicate
- `warp.sync()` -> synchronize tile

**Wave64 safety**: `cg::tiled_partition<32>` creates a 32-lane logical tile regardless of hardware wave width. On gfx90a (wave64), a 256-thread block splits into 8 tiles of 32; CG operations stay within the tile's 32 lanes. Per PORTING_GUIDE: "Width-32 LOGICAL-warp ops (cg::tiled_partition<32>) are arch-agnostic and fine -- they operate within a 32-lane subgroup regardless of the physical wavefront." HIP supports cooperative_groups with explicit tile sizes. **Expectation: no wave64 source fix needed.**

### CUB usage (host-side device algorithms)
- `cub::DeviceRadixSort::SortPairs` - forward.cu, inference.cu, pruning_scores.cu (with begin_bit=0, so no cudaKDTree hipCUB bug applies)
- `cub::DeviceScan::ExclusiveSum`, `InclusiveSum` - forward.cu, inference.cu, pruning_scores.cu
- `cub::BlockReduce<..., cub::BLOCK_REDUCE_WARP_REDUCTIONS>` - kernels_forward.cuh:471

**ROCm mapping**: torch hipify maps `<cub/cub.cuh>` -> `<hipcub/hipcub.hpp>`. hipcub wraps rocPRIM with CUB-compatible API.

### CUDA intrinsics
- `__popc(mask)` -> popcount (HIP: available)
- `__fns(mask, base, offset)` -> find Nth set bit (HIP: available as `__fns`)
- `__saturatef(val)` -> clamp to [0,1] (HIP: available)
- `__uint2float_rn(val)` -> convert (HIP: available)
- `atomicAdd` on float/float2/float3 -> HIP: available

### Shared memory
Used in blend kernels for staging tile data:
- `__shared__ uint collected_last_contributor[32]`
- `__shared__ float4 collected_color_pixel_after_transmittance[32]`
- `__shared__ float2 collected_mean2d[config::block_size_blend]`
- etc.

All sizes are compile-time constants from config (tile_width=16, tile_height=16, block_size_blend=256). No dynamic shared memory. No warp-size-dependent sizing.

### helper_math.h
NVIDIA's standard CUDA helper math file with float2/float3/float4 operations. Includes `<cuda_runtime.h>` directly. torch hipify handles this.

### No usage of:
- cuBLAS, cuFFT, cuRAND, cuDNN, cuSPARSE
- Thrust
- Textures/surfaces
- Managed memory
- Multiple streams/events
- Half precision (__half, bf16)
- GLM in device code

## Risk list

1. **helper_math.h `<cuda_runtime.h>` include**: torch hipify should map this to `<hip/hip_runtime.h>`. If not, may need a USE_ROCM guard.

2. **`constexpr uint warp_size = 32` hardcode**: Used in kernel code alongside `cg::tiled_partition<32>`. Since tiled_partition<32> creates a 32-thread logical tile on any wave width, and the `warp_size` constant is only used for loop bounds over the tile, this should be safe. The constant matches the tile size, not the hardware wave width.

3. **cub::BLOCK_REDUCE_WARP_REDUCTIONS template parameter**: Used in BlockReduce. hipcub should handle this, but verify the reduction works correctly.

4. **Shared memory arrays sized to `[32]`**: In kernels_backward.cuh lines 342-344. These are used for 32-lane tile operations, matching tiled_partition<32>. Wave64-safe because the tile is 32 threads regardless of wave width.

5. **torch hipify incremental build**: Known gotcha -- edits to .cu can recompile stale hipified mirror unless re-hipify runs. May need clean rebuild if edits are made.

## File-by-file change list

Strategy B means NO manual symbol changes. torch hipify handles .cu/.cuh translation. Expected changes:

1. **FasterGSCudaBackend/setup.py** (optional ROCm-aware tweaks):
   - May need to add `-DUSE_ROCM` to extra_compile_args when `torch.version.hip` is present
   - May need to remove NVCC-specific flags like `-use_fast_math` when on ROCm (or check if hipcc accepts it)

2. **Source files**: No changes expected if hipify handles everything. If issues arise:
   - utils/utils.h: guard cuda error checking with USE_ROCM if hipify doesn't translate
   - torch_utils.h: `.is_cuda()` check works for HIP tensors (they report as CUDA)

## Build commands (gfx90a)

Configure ROCm torch environment:
```bash
# Assumes ROCm torch is installed in py_3.12 conda env
conda activate py_3.12
export HIP_VISIBLE_DEVICES=2  # or appropriate GPU ordinal
export ROCM_HOME=/opt/rocm
```

Build:
```bash
cd projects/faster-gaussian-splatting/src/FasterGSCudaBackend
pip install -e . --no-build-isolation
```

Verify import:
```python
import torch
print(f"HIP: {torch.version.hip}")
import FasterGSCudaBackend
print("FasterGSCudaBackend imported successfully")
```

## Test plan

### Tests to locate
This project is designed as part of the NeRFICG framework. It does NOT ship standalone unit tests in this repo. Test options:

1. **Basic import/forward test** (minimal validation):
   ```python
   import torch
   from FasterGSCudaBackend.torch_bindings.rasterization import rasterize, RasterizerSettings
   
   # Create minimal test data
   n_gaussians = 100
   device = torch.device("cuda")
   
   means = torch.randn(n_gaussians, 3, device=device)
   scales = torch.randn(n_gaussians, 3, device=device)
   rotations = torch.randn(n_gaussians, 4, device=device)
   opacities = torch.randn(n_gaussians, device=device)
   sh_coefficients_0 = torch.randn(n_gaussians, 3, device=device)
   sh_coefficients_rest = torch.randn(n_gaussians, 15, 3, device=device)
   
   # Create settings
   w2c = torch.eye(4, device=device).unsqueeze(0)[:, :3, :]  # 1x3x4
   cam_position = torch.zeros(3, device=device)
   bg_color = torch.zeros(3, device=device)
   
   settings = RasterizerSettings(
       w2c=w2c, cam_position=cam_position, bg_color=bg_color,
       active_sh_bases=1, width=256, height=256,
       focal_x=200.0, focal_y=200.0, center_x=128.0, center_y=128.0,
       near_plane=0.1, far_plane=100.0, proper_antialiasing=True
   )
   
   # Run inference
   image = rasterize(means, scales, rotations, opacities,
                     sh_coefficients_0, sh_coefficients_rest.view(n_gaussians, -1),
                     settings, to_chw=True, clamp_output=True)
   print(f"Output shape: {image.shape}, device: {image.device}")
   ```

2. **Gradient test** (verify backward pass):
   ```python
   from FasterGSCudaBackend.torch_bindings.rasterization import diff_rasterize
   # Similar setup with requires_grad=True tensors
   # Run diff_rasterize and call .backward()
   ```

3. **NeRFICG framework integration test** (full validation):
   - Clone NeRFICG framework
   - Install FasterGS method
   - Run training on a small dataset (e.g., synthetic NeRF scene)
   - Compare outputs/metrics

### Non-GPU tests
None -- this is a pure GPU extension.

## Open questions

1. **No standalone tests**: The repo relies on NeRFICG framework for testing. May need to create a minimal validation script or clone NeRFICG for full validation.

2. **setup.py ROCm handling**: Need to verify if torch's CUDAExtension auto-detects ROCm or if explicit flags are needed.

3. **Performance validation**: Mechanical port should be correct, but performance vs CUDA is unknown. Not blocking for correctness validation.
