# flower notes

## Intake screen (2026-11-11)

Requested by jeff (2026-11-11) for Radeon (wave32) and Windows coverage of AMD's life-science set. The AMD fork already exists at https://github.com/AMD-Ecosystem/flower (default `main`, a GitHub fork of flwrlabs/flower). jeff has asked the AMD-Ecosystem admins for write access; until it is granted, `moat-port` cannot be pushed to the fork. No `awaiting-fork` / `release-forks` step applies; the stage is left for jeff.

### Licence

- `utils/licenses.py check flwrlabs/flower` -> Apache-2.0, tier 1. Recorded `Apache-2.0`.
- No submodules. `utils/licenses.py scan-nvidia agent_space/flower-screen` (upstream main, blobless clone): no NVIDIA proprietary licence text.

### Duplicate effort

- Open upstream PR flwrlabs/flower#8383 "fix(framework): Document AMD GPU setup and report GPU discovery failures" by smahto957 (AMD), head `AMD-Ecosystem/flower:smahto/flower_amd` at 993e2ed (2026-11-11). Opened 2026-10-19. Activity is 18 "address review feedback" rounds driven by Copilot/Codex bot reviews plus repeated pings of maintainers (panh99, tanertopal, danieljanes); no human maintainer review yet. GitHub reports `mergeable`/`mergeStateStatus` UNKNOWN (not computed), which is not "no conflict"; the branch merges main regularly (last merge 2026-11-11, 5 behind now).
- Per the intake role, the recommendation is to contribute to #8383 (Radeon and Windows evidence, docs scope) rather than open a competing PR.
- Upstream docs (`framework/docs/`) carry no ROCm/AMD mention on main. ROCm org: no flower repo.

### CUDA surface / viability

- No CUDA at all (0 `.cu`/`.cuh`): Flower is a pure-Python federated-learning framework. GPU compute comes from the user's ML framework (PyTorch/TensorFlow/JAX ROCm builds); Flower's own GPU touchpoint is the Simulation Runtime's Ray backend, which schedules ClientApps by Ray's GPU resource count. This is a `validate`-lane target: run the simulation examples and the framework test suite on a ROCm torch, fix packaging/discovery gaps, document ROCm support.
- GPU dependencies: `ray==2.56.0` (simulation extra), and on the user side a ROCm PyTorch build. Ray 2.56 discovers AMD GPUs through the `amdsmi` Python package.
- Tests: pytest suite under `framework/py/flwr/**/_test.py`; #8383 adds `ray_actor_test.py` (366 lines, mocked, no GPU needed).
- Hard MOAT dependencies: none.
- Upstream is very active (163 commits since 2026-10-01).

## AMD fork assessment

Compared `AMD-Ecosystem/flower` with `flwrlabs/flower` `main` (f7dcb6d, 2026-11-11).

- Default branch `main` (1435ea5): 0 ahead, 163 behind; it is an unmodified upstream mirror last synced 2026-10-01. No AMD work there.
- Work branch `smahto/flower_amd` (993e2ed, Sujit Kumar Mahto / smahto957): 27 commits ahead (incl. 5 merges of upstream main), 5 behind; merge base 4e5a015 (2026-11-10). Net delta 6 files, +1207/-15:
  - `framework/py/flwr/simulation/ray_transport/ray_actor.py` +210/-9: when Ray reports zero GPUs, log diagnostics based on `/dev/kfd`, `/dev/dri/renderD*` access and sysfs PCI vendor `0x1002`; warn about `amdsmi` importability, `ROCM_PATH` lacking the unversioned `libamd_smi.so`, and `ROCR_VISIBLE_DEVICES`; track GPU presence across all nodes in pool sizing.
  - `ray_actor_test.py` +366 (new, mocked).
  - Docs: new `how-to-run-simulations-on-amd-gpus.rst` (595 lines), edits to `how-to-run-simulations.rst`, `simulate.rst`, `how-to-deploy-supernode-using-slurm.rst`.
  - Noise: none in the tree (no CI/Docker/README churn); the history is noisy (18 bot-review rounds, 5 merge commits), so a squash is expected at merge.
- GPUs claimed: AMD Instinct only (PR description: "Flower runs on AMD Instinct GPUs without modification"). No Radeon or Windows evidence.
- wave32: not applicable (no kernels). Radeon risk is whether Ray's `amdsmi`-based accelerator manager enumerates RDNA cards and maps their accelerator type; unverified, needs a gfx1100/gfx1201 Linux run.
- Linux-only pieces: every diagnostic in the PR is Linux-specific (`/dev/kfd`, `/dev/dri`, `/sys/class/drm`, render/video groups); on Windows these paths do not exist, so the code is a silent no-op there (harmless, but gives no Windows help). Ray's AMD GPU discovery via `amdsmi` is Linux-only, and flower's own `pyproject.toml` excludes `ray` on win32 for Python >= 3.13 (`sys_platform != 'win32'`), so Windows simulation is limited to Python 3.11/3.12 and Ray would see no AMD GPU without manual `num_gpus` resources. Windows coverage likely reduces to "Deployment Engine + ROCm torch ClientApp works; simulation GPU scheduling needs manual resources", or a waiver suggestion (an agent may only suggest it).
- Offered upstream: yes, #8383 is open from this fork branch and awaiting maintainers.
- Judgement: AMD-authored, narrow and clean in content; it is the authoritative effort. MOAT should build on it (contribute Radeon/Windows evidence and doc lines to #8383 or a follow-up after it lands), not port independently. Coordination with smahto957 is a person's call.

## Rulings (2026-11-11, Jeff Daily)

- Validation project on upstream flwrlabs/flower `main`; base is the fork's `main`, a clean upstream mirror (163 behind on 2026-11-11), fast-forwarded with the release-line tooling's mirror command once write access exists.
- Test with a ROCm torch on all three gates. On Linux include Ray simulation on Radeon and confirm Ray enumerates RDNA GPUs (Ray discovers AMD GPUs through `amdsmi`).
- Windows: in scope, no waiver. Flower's deployment runtime (SuperLink/SuperNode clients on a ROCm torch) runs on Windows. Ray simulation is not available on Windows upstream for Python 3.13+ (flower does not install ray on win32), so those tests are skipped there as upstream-unsupported; how to offer simulation on Windows (e.g. manual GPU resources from the torch device count) is a detail Jeff will decide later.
- AMD's open upstream PR #8383 (smahto957; 6 files, +1207/-15) adds zero-GPU diagnostics to `framework/py/flwr/simulation/ray_transport/ray_actor.py`: when Ray reports no GPUs it checks `/dev/kfd` access, `/dev/dri/renderD*` nodes and their sysfs vendor id, and logs hints about the `render`/`video` groups and `amdsmi`. It is logging only (no discovery change) and is a no-op on Windows, where `/dev/kfd` does not exist. No PR competing with #8383; delivery of overlapping findings is Jeff's call.
