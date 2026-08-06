# fused-ssim ROCm/HIP port plan (linux-gfx90a)

## Project

`rahul-goel/fused-ssim` -- a fully-fused, differentiable SSIM PyTorch CUDA
extension used in Gaussian-splatting training. Forward computes the SSIM map and
(in train mode) the per-pixel partial derivatives in one pass; a second kernel
applies the chain rule for the backward. Both a 2D path (`ssim.cu`) and a 3D path
(`ssim3d.cu`) ship, plus an Apple Metal path (`ssim.mm`) and an Intel SYCL path
(`ssim_sycl.cpp`) that are irrelevant here.

Upstream HEAD at clone: `a7c48d6` ("3D fused ssim documentation and tests update").

## Build classification: Strategy B (pytorch extension)

`setup.py` uses `torch.utils.cpp_extension.CUDAExtension` / `BuildExtension`
(`setup.py:2,69,128`). On a ROCm torch, building a `CUDAExtension` automatically
runs `torch.utils.hipify` over the `.cu` sources and links the HIP runtime. So:
build against a ROCm torch, let torch hipify, and fix only what hipify cannot
(guarded by `USE_ROCM`). Do NOT add a compat header or hand-rename symbols.

Sources compiled for the CUDA/HIP build: `ssim.cu`, `ssim3d.cu`, `ext.cpp`
(`setup.py:69`), with `-DFUSED_SSIM_CUDA` set for both CUDA and HIP.

## Existing AMD support assessment

fused-ssim ALREADY advertises ROCm and ships an AMD code path, but only at the
build-config level:

- `setup.py:29-32` has an `if torch.version.hip:` branch that adds `-ffast-math`
  and skips the NVIDIA-only `--maxrregcount=32 --use_fast_math -gencode` flags.
- README line 10 claims "2D (CUDA, Metal, ROCm) and 3D (CUDA only)"; README has an
  "AMD ROCm" install section; the acknowledgements credit Anton Smirnov for "AMD
  GPU enablement".
- The only AMD-specific commit, `88169c5` ("amd gpu installation fix", #29, by an
  amd.com author), touched `setup.py` ONLY (1 line). No kernel was ever modified
  for AMD.

Per PORTING_GUIDE "assess existing AMD support" + the GPUMD changelog lesson (a
shipped upstream HIP path can bitrot; build + GPU-validate it rather than
auto-skip as already-supported), this is NOT a skip. Two specific reasons to
validate rather than trust the README:

1. The 3D path post-dates the AMD enablement. The README still says 3D is
   "CUDA only", yet `ext.cpp:5,15` gate the 3D bindings on `FUSED_SSIM_CUDA`,
   which `setup.py` defines for HIP too -- so `fusedssim3d` IS compiled and
   exported on ROCm. The 3D path (`ssim3d.cu`, the big #35/#39 rework) has never
   been GPU-validated on AMD. `fused_ssim/__init__.py:6` even imports
   `fusedssim3d` unconditionally on any CUDA-or-HIP device.
2. The README's recommended ROCm wheel is old (rocm6.1); host here is ROCm 7.2.1.

## CUDA surface / fault-class analysis (read before building)

Both kernels are windowed (11x11 Gaussian) separable convolutions with
shared-memory tiling and a per-pixel reduction. Critically, the per-pixel
reduction is done ENTIRELY through `__shared__` arrays plus `block.sync()`
barriers; the Z-axis pass in 3D uses a per-thread register ring buffer. Block is
`(BLOCK_X=16, BLOCK_Y=16)` = 256 threads.

There are NO warp-level primitives anywhere: grep for
`__shfl|__ballot|__activemask|__any|__all|__reduce|__popc|tiled_partition` over
`ssim.cu ssim3d.cu` returns nothing. The two "same warp" strings
(`ssim.cu:167`, `ssim3d.cu:252`) are misleading comments on plain
`ly + BLOCK_Y` shared-memory indexing, not warp ops.

The ONLY literal 32 in the codebase is in the 2D backward load
(`ssim.cu:328-335`): `warp_id=tid/32; lane_id=tid%32; num_warps=(T+31)/32;
col+=32`. This is a benign LOOP-TILING stride that distributes 256 threads over
the `SHARED_Y x SHARED_X` (26x26) shared tile; it does not read hardware
`warpSize` and does not do any cross-lane exchange. On wave64 it still partitions
the 256 threads into groups that fully cover every (row, col), so it is correct
unchanged. (Confirmed by reasoning: with SHARED_X=26 < 32, lanes 0..25 work, rest
idle; with num_warps=8, rows 0..25 are covered by row=warp_id; row+=8.)

=> Expected fault classes for THIS port: NONE from the wave64 class. No texture,
no atomics RMW, no cuBLAS/cuFFT/cuRAND, no rule-of-five handles, no layered
arrays. The only risk is hipify coverage of `cooperative_groups`,
`__constant__`, and `c10/cuda/CUDAGuard.h` (all standard, well-supported by
torch hipify) and possible numerical drift from `-ffast-math` / `__fmaf`
contraction differences vs the reference (a tolerance question, not a
correctness bug). This is the cleanest possible Strategy B case.

## ROCm torch environment

A locally-built ROCm PyTorch is already installed and importable (must run from a
non-source dir to avoid the in-tree `torch/_C` shadow):

- torch `2.13.0a0+gitb5e90ff`, `torch.version.hip = 7.2.53211`, CUDA None.
- `torch.cuda.is_available() == True`, 4 GPUs, all gfx90a (MI250X), warp_size 64.
- Host ROCm 7.2.1, gfx90a (CDNA2, wave64).
- No project `.venv`; the conda env `py_3.12` carries this torch.

GPU assignment for this agent: `HIP_VISIBLE_DEVICES=1` for every build and run.

## Plan of work

1. Build the extension against the ROCm torch from the fork clone:
   `HIP_VISIBLE_DEVICES=1 pip install -e . --no-build-isolation -v` (torch
   hipifies `ssim.cu` / `ssim3d.cu`).
2. If hipify leaves anything broken (not expected), fix in source guarded by
   `USE_ROCM`. No compat header.
3. Validate on GPU 1 FOR REAL:
   - Run the repo's own `tests/test.py` (2D) and `tests/test_3D.py` (3D); they
     compare fused-ssim's forward value AND backward gradients against both a
     pure-PyTorch SSIM and `pytorch_msssim` over many iterations.
   - Add an independent harness over several window/channel/batch configs that
     re-checks forward + gradient vs `pytorch_msssim` within tolerance, and
     checks run-to-run determinism (same input -> identical output bytes).
4. On success: set `linux-gfx90a` to `ported`, record validation in notes.md.
   No fork push (parent delivers); no GitHub Actions; no README/gen_readme.

## Expected outcome

Build classification B; zero kernel edits expected (no wave64 fault). If the 2D
and 3D forward+gradient match the references within tolerance and are
deterministic on gfx90a, this is a clean validation of an already-AMD-aware
project whose 3D path had never actually been GPU-checked on ROCm.
