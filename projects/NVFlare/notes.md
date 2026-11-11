# NVFlare notes

## Intake (2026-11-11)

### Fork

The AMD fork already exists: AMD-Ecosystem/NVFlare (GitHub fork of NVIDIA/NVFlare), default branch `main`; the AMD work also sits on branch `rocm-enablement`. Jeff Daily asked the AMD-Ecosystem admins for write access on 2026-11-11; until it is granted nobody in MOAT can push `moat-port` there, so the stage is left at intake for Jeff (no `awaiting-fork`, no `release-forks`). Whether MOAT builds on the AMD branch or ports independently from upstream `main` is Jeff's ruling; the assessment below is the input.

NVIDIA/NVFlare is an NVIDIA-org repository. Per intake.md an `nvidia-org` candidate has no upstream PR to offer and may be ported only on a licence clean file by file; it is (below). The deliverable would be the AMD fork.

### Licence

- Apache-2.0 (LICENSE), tier 1, `utils/licenses.py check NVIDIA/NVFlare`.
- `scan-nvidia` on a shallow upstream clone with submodules (agent_space/NVFlare-screen): no NVIDIA proprietary licence text.
- One submodule: `integration/xgboost/encryption_plugins/cuda_plugin/CGBN` -> NVlabs/CGBN. Its LICENSE is the MIT-style permissive licence ("Copyright (c) 2015, NVIDIA CORPORATION ... Permission is hereby granted, free of charge"), so clean.

### Duplicate effort

- AMD-Ecosystem/NVFlare `main`/`rocm-enablement` is the AMD effort (AMD staff). No ROCm/AMD repo elsewhere (ROCm, ROCm-LS orgs: none).
- Upstream: no open or merged ROCm/HIP PR. The "AMD" search hits are AMD SEV-SNP confidential-computing PRs, unrelated. No hip/rocm branch. The AMD work was not offered upstream.

### GPU surface (upstream main cd17b256, 2026-11-11)

- Core NVFlare is pure Python; GPU compute rides the user's framework (PyTorch in `nvflare/app_opt/pt`, client API). Nothing to port there beyond GPU discovery.
- GPU discovery/resource management: `nvflare/fuel/utils/gpu_utils.py` (nvidia-smi only; `CUDA_VISIBLE_DEVICES` filtering), `nvflare/app_common/resource_managers/gpu_resource_manager.py`, and, new since the AMD fork's base, `nvflare/private/fed/resource_stats/probes/gpu_nvidia.py` (#5639, 2026-11-09), an NVIDIA-only GPU capacity probe built on gpu_utils.
- Real CUDA source: `integration/xgboost/encryption_plugins/cuda_plugin` (`cuda_paillier` shared library for secure federated XGBoost): CMake with `find_program(nvcc)` + `LANGUAGE CUDA`, `src/paillier.h`, `cuda_utils.h`, `delegated_plugin.cc`, and the CGBN multi-precision library (50 `.cu`/`.cuh` files). CGBN is warp-synchronous by design: 328 `__shfl_sync`, 66 `__ballot_sync`, 42 `__shfl_up_sync`, 29 `__shfl_down_sync`, 180 literal `0xFFFFFFFF` masks, and inline PTX `asm` (`arith/asm.cu`, carry-chain adds) with no HIP spelling. It is optional (only the secure-XGBoost HE path uses it) and depends on GMP.
- Tests: `tests/unit_test` (pytest, includes `fuel/utils/gpu_utils_test.py`), integration tests under `tests/integration_test`; the plugin has its own C++ tests in `nvflare_plugin/tests`.
- Supported OS upstream: Linux and macOS only (docs/installation.rst; setup.cfg `Operating System :: POSIX :: Linux`). No Windows support even for CUDA. Active, not archived.

### Dependencies

- None hard. The CUDA plugin is consumed by XGBoost's federated plugin interface; MOAT `xgboost` is unclaimed. Recorded here only, not with `set-deps`.

### Viability

Yes for the GPU-discovery port and Radeon validation of the PyTorch examples (small). The CGBN/paillier plugin is a genuine port with a wave64 problem (32-lane cooperative groups and 32-bit ballot masks) and inline PTX to rewrite; wave32 Radeon is the easier target for it. Windows is not an upstream platform.

## AMD fork assessment

Compared `AMD-Ecosystem/NVFlare@main` (45fdf3ce, 2026-09-22) with `NVIDIA/NVFlare@main` (cd17b256, 2026-11-11):

- **Ahead/behind:** 2 commits ahead (one commit 4cfb5b08 "[ROCm] AMD Instinct (ROCm) enablement for nvflare", 2026-09-16, plus its merge), 183 behind. Merge base 83ec31a8, 2026-09-10. `rocm-enablement` holds the same single commit on the same base.
- **Size:** 3 files, +340 / -11: `nvflare/fuel/utils/gpu_utils.py` (+153: `has_rocm_smi()`, rocm-smi fallbacks for GPU ids and total/free VRAM), `tests/unit_test/fuel/utils/gpu_utils_test.py` (new, 149), `ROCM.md` (new, 49). Clean and minimal; no CI/Docker/README/rename noise.
- **Conflicts with current upstream:** upstream rewrote `gpu_utils.py` twice since the base (#5543 signal-terminated nvidia-smi queries, 2026-10-29; #5639 execution-visible capacity probes, 2026-11-09, which also adds `gpu_nvidia.py` and changes `gpu_resource_manager.py` and the same test file). The fork commit no longer applies cleanly and misses the new NVIDIA-only probe.
- **GPUs claimed:** Instinct only: ROCM.md and the commit message list MI300X (gfx942) and MI350X/MI355X (gfx950), ROCm 10.0.0, Ubuntu 24.04, Python 3.12.
- **Wave64 assumptions:** none (no kernels touched). The fork does not touch the CGBN/paillier CUDA plugin at all, so on AMD the secure-XGBoost GPU path is simply unported.
- **Linux-only pieces:** relies on `rocm-smi` (Linux; not on Windows; superseded by `amd-smi`; absent from pip-only ROCm installs). Upstream itself is Linux/macOS.
- **Offered upstream:** no.

### Radeon (wave32) and Windows risks

- GPU discovery: rocm-smi card numbering versus HIP device order on iGPU + dGPU hosts; `get_visible_gpu_ids()` honours only `CUDA_VISIBLE_DEVICES`, not `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES`. The new `gpu_nvidia.py` probe needs an AMD counterpart.
- CGBN plugin: warp-synchronous 32-lane code with `0xFFFFFFFF` masks and PTX asm. On wave32 the lane model matches; on wave64 every ballot/shuffle mask is wrong-width (fault-class index: `__ballot_sync` cast to 32 bits). Whether this plugin is in scope is a scope question for Jeff.
- Windows: not an upstream platform for NVFlare at all; the Windows gate is a waiver candidate (an agent may only suggest it).

### Recommendation

`fork` (record via `set-intake`). The Python GPU-discovery port is small and the AMD fork's commit is a usable hint but needs re-doing on current upstream; Radeon validation of the PyTorch path is cheap. The optional CGBN/paillier plugin is the only real kernel port and needs a scope ruling. Windows is a waiver candidate.
