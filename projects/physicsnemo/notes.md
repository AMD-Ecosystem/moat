# physicsnemo notes

## Intake screen (2026-11-12)

Origin: Jeff Daily's 2026-11-12 ruling on a third-party "AMD AI software gap" list -- validation project (probably runs on ROCm through PyTorch and Warp, never tested on Instinct, Radeon or Windows).

- Upstream: https://github.com/NVIDIA/physicsnemo, default branch `main`, active (pushed 2026-11-12), not archived. NVIDIA org: no upstream PR expectation beyond a docs/packaging change; licence must be clean file by file.
- Licence: Apache-2.0 (LICENSE.txt), tier 1. `utils/licenses.py scan-nvidia` on a depth-1 clone flagged 6 files (`physicsnemo/datapipes/benchmarks/kernels/*.py`, `examples/minimal/neighbor_list/warp_neighbor_list.py`); read by hand, all six are SPDX Apache-2.0 files whose ImportError message says Warp "is under the NVIDIA Source Code License (NVSCL)" -- a stale remark about Warp's old licence, not a licence on the file. Clean. No submodules.
- CUDA surface: no `.cu`/`.cuh` files. Pure Python on torch>=2.13 plus `warp-lang>=1.17.0` as a core install dependency (Warp kernels in datapipes, mesh, neighbour lists).
- Duplicate effort: no ROCm/AMD/HIP issues, PRs or branches upstream; no AMD-Ecosystem or ROCm fork; no community ROCm repo found.

## Dependencies

- `warp-lang` (NVIDIA/warp) is a hard install dependency, so validation cannot run until Warp has a working ROCm build. MOAT project `warp` (dev line; release lines warp-1.14..1.19) exists; recorded with `set-deps physicsnemo warp`. physicsnemo requires warp>=1.17, so the matching release lines are warp-1.17+ or the dev line.
- Optional CUDA-only extras: `transformer-engine-cu12/cu13` (ROCm substitute: ROCm/TransformerEngine), `natten-cu12/cu13` (no ROCm wheel; NATTEN has a CPU/flex-attention fallback -- gap to check), RAPIDS (cuML/cuDF via pypi.nvidia.com, no ROCm build: gap, out of scope), torch-scatter/torch-cluster (build from source on ROCm). `nvtx` is a hard dependency but is a no-op Python package without NVIDIA tools.

## Validation scope

- Delivery route: fork-only validation fixes on AMD-Ecosystem/physicsnemo, plus at most one upstream docs/packaging PR (ROCm install path in README/pyproject indices) if the suite passes. Most real fixes are expected to land in the Warp port, not here.
- GPU gate: upstream pytest suite `test/` (markers `cuda`, `multigpu_static`, `multigpu_dynamic`; `--multigpu-*` flags for the distributed shards), run with ROCm torch and the MOAT Warp build; supplement with `examples/minimal` Warp examples. Record skips for natten / transformer-engine / RAPIDS tests separately from failures.
- ROCm risks: Warp ROCm maturity (gate), pyproject pins CUDA torch indices (cu129/cu130) so `uv sync` will not pick ROCm torch without an override, natten and RAPIDS gaps.
- Windows: no upstream Windows story (README documents Linux CUDA 12/13 backends only; CUDA 12 backend explicitly Linux-only). Suggest a windows waiver; only Jeff approves one.
- Existing AMD work to back: none found upstream or in the community.
