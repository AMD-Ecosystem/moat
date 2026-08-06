# Port plan: mmcv

## Project
- Name: mmcv
- Upstream: https://github.com/open-mmlab/mmcv
- Default branch: main

## Decision: SKIP (already-supported)

mmcv already ships mature, upstream-merged ROCm/HIP support. A MOAT port would
duplicate AMD's and the upstream maintainers' own work, which the porting guide
explicitly forbids ("Mature ROCm/HIP support upstream -> skip; we do not
duplicate AMD's own work"). Disposition recorded via
`utils/triage.py skip open-mmlab/mmcv --reason already-supported`.

## Existing AMD support (assessment)

Authoritative, upstream-merged ROCm/HIP support -- not a community fork.

Evidence (all in the upstream `main` tree):
- Docs: `docs/en/get_started/build.md:51` and `docs/zh_cn/get_started/build.md:63`
  document the AMD ROCm build path. `docs/en/compatibility.md:131` states the
  `mmcv/ops/csrc` directory was refactored to "flexibly support more backends and
  hardwares like NVIDIA GPUs and AMD GPUs" via upstream PR1206
  (https://github.com/open-mmlab/mmcv/pull/1206).
- Build: `setup.py` (lines 224-272) detects a ROCm PyTorch
  (`torch.version.hip is not None` and `ROCM_HOME is not None`), defines
  `MMCV_WITH_HIP`, and routes through the standard `CUDAExtension`, which torch
  hipifies the `.cu`/`.cuh` sources at build time. This is the canonical
  Strategy-B mechanism -- already implemented upstream.
- Kernels: `MMCV_WITH_HIP` guards are present across the CUDA surface (58 `.cu`
  files under `mmcv/ops/csrc/pytorch/cuda/`). They handle the real AMD fault
  classes already:
  - Warp size: `carafe_cuda_kernel.cuh` defines `WARP_SIZE 64` under
    `MMCV_WITH_HIP` (32 otherwise); same pattern in `correlation_cuda.cuh`,
    `scatter_points_cuda_kernel.cuh`.
  - Warp-shuffle dialect: `__shfl_down` (HIP) vs `__shfl_down_sync(FULL_MASK,...)`
    (CUDA) selected by the guard.
  - Per-op HIP branches in `filtered_lrelu.cu`, `bias_act_cuda.cu`,
    `upfirdn2d_kernel.cu`, `bbox_overlaps_cuda.cu`, `info.cpp`, etc.
- Python: `mmcv/ops/conv2d_gradfix.py` branches on
  `mmengine...parrots_wrapper.is_rocm_pytorch()`; an `is_rocm_pytorch` helper is
  part of the documented API surface (`docs/en/get_started/api_reference.md:255`).

A non-authoritative community fork also exists (`AHNU2019/mmcv_ROCM`), but it is
irrelevant here: the support is already in upstream `main`, so there is nothing
to adopt or improve.

Authoritative vs community judgment: AUTHORITATIVE (upstream-merged, maintained
by open-mmlab with an explicit AMD-support refactor). The correct action is to
skip; there is no differentiator to contribute (unlike the gsplat/llm.c cases
where AMD support lived in a separate fork or PR queue).

## Build classification (for the record)
- torch-extension (Strategy B). Evidence: `setup.py` uses
  `torch.utils.cpp_extension.CUDAExtension` / `BuildExtension`; `ext_type` in
  upstream.json/status.json is already `torch-extension`.

## CUDA surface inventory (for the record)
- 58 `.cu` kernels in `mmcv/ops/csrc/pytorch/cuda/` plus shared `.cuh` in
  `mmcv/ops/csrc/common/cuda/`. Standard PyTorch op kernels (deformable conv,
  RoIAlign, nms, carafe, correlation, voxelization/scatter-points, focal loss,
  the StyleGAN ops upfirdn2d/bias_act/filtered_lrelu, etc.).
- No CUTLASS/CuTe, no Hopper wgmma, no inline PTX requiring an AMD-native rewrite
  observed; the upstream HIP guards already cover the warp-size and
  warp-shuffle fault classes for these kernels.

## Open questions
- None. The skip is unambiguous: upstream already builds and runs on ROCm by its
  own documented path.
