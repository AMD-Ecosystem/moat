# pytorch_sparse -- port plan

## Project
- Name: pytorch_sparse (PyPI: torch-sparse)
- Upstream: https://github.com/rusty1s/pytorch_sparse
- Default branch: master
- depends_on: pytorch_scatter (itself SKIP/already-supported)

## Disposition: SKIP (already-supported)

rusty1s/pytorch_sparse has mature, AUTHORITATIVE, merged upstream ROCm/HIP
support. No MOAT port is warranted. This mirrors the three sibling PyG
extensions already skipped (pytorch_scatter, mmcv, pytorch_cluster); the
evidence below was verified directly against this repo, not assumed.

Per PORTING_GUIDE "Before porting: assess existing AMD support" -- mature
ROCm/HIP support upstream plus a maintained wheel channel -> skip; we do not
duplicate AMD's own merged work.

## Existing AMD support (authoritative)

Merged, AMD/maintainer-authored upstream PRs (the exact pytorch_scatter
pattern):
- #282 "Enable ROCm builds" -- dkbhaskaran (AMD), MERGED 2022-10-17.
  Foundational: the setup.py `torch.version.hip` path.
- #296 "Use macro for `__shfl_*` functions for ROCm" -- miaoneng, MERGED
  2022-11-28. The SHFL_*_SYNC macro shim in csrc/cuda/utils.cuh.
- #360 "[ROCm] fixes ambiguous calls to shfl* ... c10::Half to __half" --
  ashwinma (AMD), MERGED 2024-01-22. The at::Half __shfl* overloads.
- #405 "Add ROCm 6.4.3+ support" -- Looong01, OPEN 2025-08-15 (same maintainer
  who ships the pyg-rocm-build wheels).

Maintained ROCm wheels:
- Looong01/pyg-rocm-build ships prebuilt torch-sparse-rocm 0.6.18; latest
  release 2026-04-21, ROCm 7.2.2, PyTorch 2.11, Python 3.10-3.14.
  `pip install torch-sparse-rocm`.

Authoritative-vs-community judgment: AUTHORITATIVE. AMD-engineer-authored,
merged into upstream master, plus a maintained wheel channel. Not a one-off
community fork.

Build-time ROCm path already present in setup.py (Strategy B, torch
CUDAExtension):
- setup.py:23  `WITH_CUDA = CUDA_HOME is not None or torch.version.hip`
- setup.py:46  strips stale generated `hip` files on rebuild
- setup.py:98-102 under `if torch.version.hip:` adds `('USE_ROCM', None)` to
  define_macros and undefs `__HIP_NO_HALF_CONVERSIONS__`
- setup.py:147-148 hipify abs-path work-around (`include_package_data=False`)

ROCm-guarded device source (fault classes already handled):
- csrc/cuda/utils.cuh:42-53 -- USE_ROCM-guarded shim: `__ldg(at::Half*)`
  overload and `SHFL_UP_SYNC`/`SHFL_DOWN_SYNC`/`SHFL_SYNC` macros that resolve
  to the MASKLESS `__shfl_up`/`__shfl_down`/`__shfl` on ROCm (avoiding the HIP
  masked-sync mask-mismatch) and to the `_sync` forms on CUDA.
- csrc/cuda/atomics.cuh:8 -- `USE_ROCM`-guarded CAS atomAdd path.
- csrc/version.cpp:7-31 -- `USE_ROCM` -> `<hip/hip_version.h>` / `HIP_VERSION`.

So the Strategy B build path MOAT would otherwise add is already in tree.

## Build classification (evidence)

torch-extension (Strategy B). Evidence:
- setup.py imports `torch.utils.cpp_extension` {CUDA_HOME, BuildExtension,
  CppExtension, CUDAExtension} (setup.py:11-16) and builds each op as a
  `CUDAExtension` (setup.py:120-130).
- `cmdclass={'build_ext': BuildExtension.with_options(...)}` (setup.py:171-173).
A ROCm torch auto-hipifies the `.cu`/`.cuh` and links amdhip64/c10_hip/torch_hip.
(There is also a CMakeLists.txt for the libtorch C++ API consumer path, but the
Python wheel -- the deliverable and what the tests exercise -- builds via
setup.py CUDAExtension, so the classification is torch-extension.)

## Port strategy (would-be)

If a port were warranted it would be Strategy B (torch hipify, no compat header,
no hand-renaming), identical to fused-ssim / torch-linear-assignment. It is not
warranted: the ROCm build path and the warp shim are already merged upstream.

## CUDA surface inventory

- Device sources: csrc/cuda/{spmm,rw,diag,convert}_cuda.cu + reducer.cuh,
  atomics.cuh, utils.cuh.
- Warp intrinsics: only via the SHFL_*_SYNC macros (utils.cuh), already
  USE_ROCM-shimmed to maskless __shfl on ROCm.
- spmm_cuda.cu: 32-relative LOGICAL-warp ops (`thread_idx >> 5` row,
  `thread_idx & 31` lane, reduction loop `for(i=32/2; i>0; i/=2)` with
  SHFL_DOWN; the gather kernel shuffles lanes 0..31 via SHFL_SYNC). `FULL_MASK
  0xffffffff` is unused on ROCm (the macros drop the mask).
- No cub/thrust, no cuBLAS/cuSPARSE/cuFFT/cuRAND, no textures/surfaces, no
  managed/pinned memory beyond ATen, no custom streams/events (uses
  at::cuda::CUDAContext).
- spspmm (test_spspmm.py) dispatches to cuSPARSE only via torch's own
  `torch.sparse.mm` / the cpu path; no in-repo cuSPARSE call.

## Risk list (for the record; not being actioned given SKIP)

- WAVE64 cross-logical-warp shuffle in spmm_cuda.cu spmm_kernel (lines 63-69):
  `mat_rows[i] = SHFL_SYNC(FULL_MASK, mat_row, i)` for i in [0,32). On ROCm
  SHFL_SYNC -> maskless `__shfl(var, i)` with NO explicit width, which reads
  lane i of the full 64-lane wavefront. With THREADS=256 a wavefront holds two
  logical 32-warps (rows), so an upper-half thread reading lane i<32 reads the
  LOWER logical warp's `mat_row`. This is the canonical wave64 cross-logical-
  warp hazard (PORTING_GUIDE warp-size class). It is the ONE thing a from-
  scratch porter would have to prove correct on gfx90a. AMD's merged #282/#296/
  #360 plus the shipped torch-sparse-rocm wheel are the authoritative resolution
  (the sibling pytorch_scatter segment kernels carry the same shim and are
  AMD-validated, wheel-shipped). We defer to that, not re-derive it. If MOAT ever
  revisits, the porter must give the spmm gather an explicit width-32 shuffle
  (`__shfl(var, i, 32)`) or confirm the wheel already does so before claiming a
  delta.
- The diag/rw/convert kernels are elementwise/per-thread; no warp collectives.
- No rule-of-five handles, no OOB neighbor stencils, no 256B texture pitch, no
  library swaps.

## File-by-file change list

None. No fork, no commit. (Were a port warranted, zero source edits -- the
ROCm path is already merged.)

## Build commands (reference, gfx90a)

If ever exercised against a ROCm torch from the fork clone:

    cd projects/pytorch_sparse/src
    HIP_VISIBLE_DEVICES=<n> PYTORCH_ROCM_ARCH=gfx90a \
      python -m pip install -e . --no-build-isolation -v

Requires pytorch_scatter (torch-scatter-rocm) installed first (runtime dep of
the torch_sparse Python package).

## Test plan (reference)

pytest under test/ (20 files): test_spmm.py and test_matmul.py exercise the
spmm CUDA kernel and the wave64 risk above; test_diag/convert/sample/saint/rw
cover the other device kernels; the remaining files are CPU/storage logic that
must not regress. Validation would be `pytest test/ -v` on a real AMD GPU with
torch-scatter-rocm + torch-sparse built against ROCm torch.

## Decision

SKIP, disposition already-supported. Recorded via
`triage.py skip rusty1s/pytorch_sparse --reason already-supported`. Duplicating
AMD's merged ROCm support plus the maintained wheel would re-port already-owned
work. See [[moat-no-duplicate-amd-ports]] and the pytorch_scatter notes.

## Open questions

- None blocking the SKIP. The only technical curiosity (the spmm wave64
  cross-logical-warp shuffle width) is settled by the authoritative merged
  support; re-examine only if a future ROCm wheel regression surfaces.
