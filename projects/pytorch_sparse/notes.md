# pytorch_sparse -- porting notes

## Disposition: SKIP (already-supported)

rusty1s/pytorch_sparse (PyPI torch-sparse) has mature, AUTHORITATIVE, merged
upstream ROCm/HIP support. No MOAT port is warranted. Recorded via
`triage.py skip rusty1s/pytorch_sparse --reason already-supported`.

Same call as the three sibling PyG extensions already skipped
(pytorch_scatter, pytorch_cluster, mmcv); the evidence below was verified
directly against this repo, not assumed. See plan.md for the full inventory.

## Evidence (summary)

Merged, AMD/maintainer-authored upstream PRs:
- #282 "Enable ROCm builds" -- dkbhaskaran (AMD), MERGED 2022-10-17.
- #296 "Use macro for `__shfl_*` functions for ROCm" -- miaoneng, MERGED
  2022-11-28.
- #360 "[ROCm] fixes ambiguous calls to shfl*" -- ashwinma (AMD), MERGED
  2024-01-22.
- #405 "Add ROCm 6.4.3+ support" -- Looong01, OPEN.

In-tree ROCm path (Strategy B, torch CUDAExtension):
- setup.py:23,46,98-102,147-148 -- `torch.version.hip` branch, USE_ROCM define,
  `__HIP_NO_HALF_CONVERSIONS__` undef, hipify abs-path workaround.
- csrc/cuda/utils.cuh:42-53 -- USE_ROCM SHFL_*_SYNC maskless-shfl shim +
  `__ldg(at::Half*)`.
- csrc/cuda/atomics.cuh:8, csrc/version.cpp:7-31 -- USE_ROCM guards.

Maintained wheels: Looong01/pyg-rocm-build ships torch-sparse-rocm 0.6.18,
ROCm 7.2.2, PyTorch 2.11, Python 3.10-3.14; latest release 2026-04-21.
`pip install torch-sparse-rocm`.

## One technical note for any future revisit

spmm_cuda.cu spmm_kernel (lines 63-69) does `SHFL_SYNC(FULL_MASK, mat_row, i)`
for i in [0,32). On ROCm SHFL_SYNC -> maskless `__shfl(var, i)` with no explicit
width, reading lane i of the full 64-lane wavefront. With THREADS=256 a
wavefront packs two logical 32-warps, so this is a cross-logical-warp read on
wave64 -- the canonical wave64 hazard. AMD's merged work + the shipped
torch-sparse-rocm wheel are the authoritative resolution; do NOT re-derive it.
If a future ROCm wheel regression surfaces, the fix would be an explicit
width-32 shuffle (`__shfl(var, i, 32)`).

## Conclusion

Authoritative merged upstream support + maintained wheels => skip. Duplicating
AMD's own merged work is out of scope. State left unclaimed; the
already-supported disposition gates re-adoption. See
[[moat-no-duplicate-amd-ports]].
