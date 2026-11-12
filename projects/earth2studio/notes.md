# earth2studio notes

## Intake screen (2026-11-12)

Validation project (lane `validate`, seed `amd-gap-matrix`, NVIDIA-owned upstream), taken up per jeff's 2026-11-12 ruling on the third-party "AMD AI software gap" list.

- Upstream: https://github.com/NVIDIA/earth2studio, default branch `main`, very active (pushed 2026-11-12), not archived. Python framework that wraps prognostic (px), diagnostic (dx) and data-assimilation (da) weather/climate models behind one API; each model is an optional extra in `pyproject.toml`.
- Licence: Apache-2.0 (tier 1), `license = "Apache-2.0"` in `pyproject.toml`. No submodules.
- `utils/licenses.py scan-nvidia` on a depth-1 clone: two hits, both genuinely `SPDX-License-Identifier: LicenseRef-NvidiaProprietary` with the standard "no use without an express licence agreement" text: `docs/_static/css/nvidia-sphinx-theme.css` (docs theme) and `serve/server/openapi.yaml` (the inference-server API spec). Neither is in the model or test path, and validation fixes would not touch them, but they are tier-3 files in the tree and need a person's decision before anything is offered upstream (and arguably before the fork carries them, though they are already published by NVIDIA in the public repo).
- CUDA surface: none of its own (no `.cu`). PyTorch core (`torch>=2.5`); per-model extras pull in the GPU dependencies.
- Tests: `test/` pytest, tox envs `test`, `test-data`, `test-perturb`, `test-stats`, `test-px-models`, `test-dx-models`, `test-da-models`, `test-serve` (`make pytest TOX_ENV=<env>`). Model tests parametrize `device` over `cpu` and `cuda:0` with mock weights; `@pytest.mark.package` tests download and run the real checkpoints on `cuda:0`. `test/_ci/check_gpu.py` prechecks `torch.cuda.is_available()`. Examples gallery under `examples/` (`test/_ci/run_examples.sh`).

## Per-model dependency picture (extras in `pyproject.toml`)

- Pure PyTorch + PhysicsNeMo (expected to work): `fcn`, `dlwp`, `corrdiff` (non-DiT), `stormcast`, `precip-afno(-v2)`, `solarradiation-afno`, `windgust-afno`, `interp-modafno`, `statistics`, `aurora` (microsoft-aurora), `ace2` / `samudrace` (fme), `climatenet`, `persistence`.
- torch-harmonics (has CUDA extension, 17 `.cu` + 6 `.cuh`; check for a pure-PyTorch fallback): `perturbation`, `sfno`, `fcn3` (with `makani`), `atlas`.
- NATTEN (MOAT already declined NATTEN as `cant-port`: libnatten is CUTLASS/CuTe): `atlas`, `interp-crps-dit`, `stormcast-conus`, `stormscope`, `cosmo`, `corrdiff-era5-hrrr`, and `da-cosmo` via `cosmo`. These are the CorrDiff/StormCast DiT variants; expect them to be gaps unless NATTEN's pure-PyTorch reference path (if present in the pinned version) is acceptable for validation.
- flash-attn: `aifs`, `aifs2`, `aifsens`, `aifs2ens` (anemoi-models). Substitute: upstream flash-attn ROCm build.
- `jax[cuda13]`: `gencast`, `graphcast`, `weathernext`. Substitute: JAX ROCm wheels (`jax[rocm]` / ROCm jaxlib plugin).
- `onnxruntime-gpu`: `fengwu`, `fuxi`, `pangu`. Substitute: onnxruntime ROCm / MIGraphX execution provider.
- CuPy / cuCIM / cuDF (CUDA 13 wheels): `cyclone` (cucim, cupy), `da-healda(-v2)`, `da-interp`, `da-cosmo` (cupy, cudf). Substitute: CuPy ROCm build exists; cuCIM and cuDF are MOAT projects (`cucim`, `cupy` on branches); cuDF on ROCm is a gap.
- `earth2grid` (NVlabs, Apache-2.0, 2 `.cu` files for HEALPix padding with a CUDA extension): `utils`, `dlesym`, `cbottle`, `da-healda`. Possible small follow-on port, like torch-harmonics.
- `nvtx`, `pynvml`: NVIDIA-only telemetry; need to be optional / no-op on ROCm (`pynvml` in `sfno`).

## Duplicate effort

- No upstream PR or issue about ROCm/AMD (searched 2026-11-12; only dependabot noise). No fork under ROCm or AMD-Ecosystem. No ROCm mention in docs.
- `NVIDIA/torch-harmonics` (BSD-3-Clause, 23 CUDA source files): no ROCm/HIP PR upstream, no ROCm fork. A candidate small follow-on port; noted here, NOT claimed.
- `physicsnemo` is being screened as its own MOAT project (branch `port/physicsnemo`, claimed 2026-11-12); not recorded as a hard dependency because most earth2studio models need only its pure-PyTorch layers, and its screen is still open.

## Validation scope

- **Delivery route:** fork-only by default (NVIDIA-owned upstream, `nvidia-org`). The fork's `moat-port` carries ROCm install notes and guards (nvtx / pynvml optional, CUDA-only extras excluded); jeff decides whether anything goes upstream.
- **Scope question for jeff (not assumed):** a per-model result is inevitable. Natural cut: validate the core API plus the pure-PyTorch and PhysicsNeMo models (FCN, DLWP, CorrDiff, StormCast, AFNO family, Aurora, ACE2), the flash-attn AIFS models, the JAX models with ROCm JAX, and the ONNX models with ORT-ROCm; record the NATTEN DiT variants and the cuDF data-assimilation models as gaps. Per the standing rule, a partial scope needs jeff's explicit answer.
- **GPU gate:** `make pytest TOX_ENV=test`, `test-perturb`, `test-stats`, `test-px-models`, `test-dx-models`, `test-da-models` on a ROCm torch build (the `cuda:0` parametrization maps to the AMD device), including the `package`-marked real-checkpoint tests for in-scope models; plus `examples/01_getting_started` and one medium-range forecast example (SFNO or FCN) compared against CPU or CUDA output.
- **CUDA-only dependencies:** NATTEN (gap, declined in MOAT as can't-port); torch-harmonics CUDA extension (follow-on port, or its fallback); earth2grid CUDA extension (follow-on port); `jax[cuda13]` (ROCm JAX); `onnxruntime-gpu` (ORT ROCm / MIGraphX EP); `cupy-cuda13x` (CuPy ROCm); `cucim-cu13` (MOAT `cucim`); `cudf-cu13` (gap); `nvtx`, `pynvml` (make optional).
- **Windows:** classifiers say "OS Independent", but the model stack (PhysicsNeMo, JAX, cuDF, NATTEN, Redis serve) is Linux-first and CI is Linux containers. The core API plus a few pure-PyTorch models (FCN, DLWP, AFNO) may run on Windows ROCm torch; a Windows run is possible evidence but not a natural gate. Suggest a `windows` waiver, or keep the gate scoped to the core + pure-PyTorch subset; jeff decides.
- **Risks:** breadth (dozens of model extras, each with its own dependency stack); large checkpoint downloads for `package` tests; NATTEN-dependent DiT models are the headline CorrDiff/StormCast variants the gap list mentions, so the visible result will be partial.

## Declined (2026-11-12)

Jeff Daily ruled earth2studio out of scope for MOAT on 2026-11-12: too much work for a validation project (two LicenseRef-NvidiaProprietary files, NATTEN DiT and cuDF data-assimilation model gaps, per-model validation). Intake is re-recorded as decline and the project is held. Do not request a fork. torch-harmonics and earth2grid stay unclaimed.
