# IsaacLab notes

## Intake screen (2026-11-05, windows-gfx1151 host, intake role)

Requested by jeff (Jeff Daily) while he is away; questions for him are in "Open questions for jeff" below. Source: third-party "AMD AI Software Gap Matrix" seed (matrix verdict: not-yet, no AMD path). Screened against upstream `isaac-sim/IsaacLab` `main` @ `eed2d2889a5b963fee8c3594d29fe6d91ec67046` (VERSION 3.0.0), shallow clone in `agent_space/IsaacLab-screen`.

### What it is

NVIDIA-owned robot-learning framework (8.5k stars, active: pushed 2026-11-05). Pure Python: no `.cu`/`.cuh`/C++ in the tree. The GPU work happens in its physics backends and in Warp kernels written in Python (`wp.kernel`), plus torch. Two families of workflow:

- Kit workflows: need Isaac Sim (proprietary, NVIDIA EULA, `isaacsim[all,extscache]==6.2.0.0` from pypi.nvidia.com) and RTX rendering. Not portable by us; out of scope.
- Kit-less workflows: Newton physics (`newton[sim]` pinned to a git dev revision `42f7def1`, "newton 1.8.0.dev0"), MuJoCo Warp (`mujoco-warp~=3.15.0`), `warp-lang==1.17.0`, `torch==2.11.0` from the `pytorch-cu130` index. README: "Kit-less Newton workflows do not require Isaac Sim". Upstream ships a `kitless` docker profile (docs/source/isaaclab/installation.rst: "the only one with no Isaac Sim EULA"). The kitless image also carries OVPhysX/OVRTX (NVIDIA binary wheels, NVIDIA licence); those are optional extras (`[ovphysx]`, `[ovrtx]`), not base deps.

### Licence

- GitHub reports NOASSERTION; `licenses.py check` = UNPARSED. Read by hand: `LICENSE` is BSD-3-Clause ("Copyright (c) 2022-2026, The Isaac Lab Project Developers"); `LICENSE-mimic` is Apache-2.0 and covers `source/isaaclab_mimic`. Both tier 1. Recorded as `BSD-3-Clause AND Apache-2.0`.
- `python3 utils/licenses.py scan-nvidia agent_space/IsaacLab-screen`: 6 hits, each read:
  - `source/isaaclab/isaaclab/envs/mimic_env_cfg.py`: header has `SPDX-License-Identifier: BSD-3-Clause` and `Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES`, then line 7 "Licensed under the NVIDIA Source Code License [see LICENSE for details]", while the LICENSE it points to is BSD-3. A self-contradictory header, probably stale from the mimic code's origin. Recorded, NOT resolved: it is a decision for a person (see questions). A port is unlikely to need to touch this file.
  - `docs/licenses/dependencies/ovphysx-license.txt`, `ovrtx-license.txt`, `ovstage-license.txt`: licence texts of external NVIDIA binary dependencies (optional extras), not Isaac Lab source.
  - `docs/source/isaaclab/installation.rst`: prose telling users Isaac Sim needs the NVIDIA Omniverse EULA / NVIDIA Software License Agreement. Docs, not a licence on the file.
  - `.github/workflows/license-exceptions.json`: CI allow-list of package licences (isaacsim-* packages etc.). Not source.
- No submodules (`.gitmodules` absent). No vendored native code.
- README says `isaaclab_mimic` needs cuRobo under proprietary terms; NVlabs/curobo is now Apache-2.0 upstream (stale README claim; not verified further, cuRobo is a mimic-only extra).
- NVIDIA ownership: upstream is NVIDIA's own repo, so an upstream PR (even docs-only) may be unwelcome. jeff's 2026-11-05 policy allows porting an NVIDIA-owned repo if the licence is clean file by file; it is, apart from the mimic_env_cfg.py header contradiction above.

### Existing AMD effort / duplicate check

- Upstream: `grep -rniE 'amd|rocm|hip|gfx'` over README/docs finds only CPU recommendations ("AMD Ryzen Threadripper") and robot hip joints. `gh pr list --state all --search "rocm OR amd OR hip"`: no ROCm PR (hits are robot "hip" joints). No rocm/hip/amd branches. No AMD/ROCm issue.
- Web search "Isaac Lab ROCm AMD GPU Newton Warp": nothing AMD.
- `AMD-Ecosystem/IsaacLab` exists: created 2026-09-10, GitHub fork of isaac-sim/IsaacLab, single branch `release/3.0.0-beta2` (also its GitHub default branch), 0 ahead / 2 behind upstream, no `main`, no `moat-port`, no MOAT record. jeff ruled 2026-11-05 that MOAT TAKES OVER this fork. Because the fork exists, intake outcome is `screened` (via `awaiting-fork` + `release-forks`, which records `fork_url`).
- GPU dependency AMD forks (none is a MOAT project, so not `set-deps`; recorded here):
  - Warp: `AMD-Ecosystem/warp` (ROCm/warp redirects there), default branch `amd-integration`, `VERSION.md` = `1.13.0+rocm.0`, release branch `release/1.13.0.amd0`. Built from source (`pip install -e .`, auto-detects HIP), README lists gfx942 (MI325X) as the supported GPU, ROCm 7.1.1/7.2.1 and TheRock. Isaac Lab pins `warp-lang==1.17.0`: a four-minor-version gap. AMD-official, active (pushed 2026-11-05).
  - Newton: `AMD-Ecosystem/newton`, branches `main`, `rocm-enablement`. `rocm-enablement` = 1 commit ("[ROCm] AMD Instinct (ROCm) enablement for newton") on top of an upstream base 569 commits behind newton-physics/newton main. Isaac Lab pins a newer dev sha (`42f7def1`).
  - MuJoCo Warp: `AMD-Ecosystem/mujoco_warp`, default `amd-integration`, `release/3.8.1.amd0` plus several perf branches. Isaac Lab wants `mujoco-warp~=3.15.0`.
  - torch: ROCm wheels exist for current torch; Isaac Lab hard-wires the `pytorch-cu130` uv index (`[tool.uv.sources] torch = { index = "pytorch-cu130" }`).
  - Judgement: the Warp/Newton/MuJoCo-Warp forks are authoritative AMD efforts (AMD-Ecosystem, AMD-labelled release branches), but they trail the versions Isaac Lab pins. Whether Isaac Lab's Kit-less path runs on them is unknown until someone tries; the version skew is the first thing the planner must resolve (rebase forks forward vs. pin Isaac Lab back) and is a coordination question with the teams owning those forks, not a race.

### What in Isaac Lab itself is CUDA-specific (cheap grep, not exhaustive)

- No CUDA source. `torch.cuda.*` calls (is_available x52, synchronize, set_device, Stream, CUDAGraph, ...) and `"cuda:0"` device strings (~300) work unchanged on ROCm torch. Whether AMD Warp accepts `"cuda:N"` device strings (and `wp.is_cuda_available`, `wp.get_cuda_device_count`, x18/x11) is a planner question; that is the main seam.
- Graph capture: `wp.ScopedCapture` x39, `wp.capture_launch` x37, `cuda_graph` x122 (config flags). Depends on AMD Warp's HIP graph support (mujoco_warp fork has an `amd/hip-graph-support` branch, suggesting this is known work).
- NVIDIA-only tooling: `nvidia-smi` x30, `pynvml` x28, `cupy` x15, in ~20 files -- benchmark GPU-info recorder (`isaaclab/benchmark/recorders/record_gpu_info.py`, also `torch.version.cuda`), ray multi-GPU scripts, multi-GPU CLI (`CUDA_VISIBLE_DEVICES`), teleop system checks, a few contrib tasks. Mostly peripheral; needs rocm-smi/amdsmi or graceful fallback where on the Kit-less path.
- Packaging: torch/torchvision forced to the cu130 index; `warp-lang` PyPI wheel is CUDA-only; `jax[cuda13]` extra for skrl-jax; base deps also include `omniverseclient==2.74.0` and the `isaaclab-ov` workspace member (check they import cleanly without NVIDIA GPUs). This is the likely bulk of the work: a ROCm install recipe (ROCm torch index, AMD Warp/Newton/MuJoCo-Warp from source or wheels) and a docs page.

### Tests

487 `test_*.py` files under `source/`. Upstream marks tests that pass without Isaac Sim with `pytest.mark.kitless` (contributing.rst: "``kitless`` when the test passes without Isaac Sim"; CI selects files by grepping for the marker). 35 files carry it: isaaclab 12, isaaclab_ov 7 (also need OV wheels, `ov` marker), isaaclab_tasks 6, isaaclab_newton 4, _isaaclab_testing 2, isaaclab_visualizers 2, isaaclab_physx 1, isaaclab_teleop 1. So a defined GPU test subset exists that can run without Isaac Sim; the OV-marked ones need NVIDIA binary wheels and are likely out of reach on AMD.

### Viability

Yes, conditionally. The work is not a CUDA-to-HIP translation: it is (1) getting the Kit-less Newton stack installed on ROCm against the AMD Warp/Newton/MuJoCo-Warp forks (version skew is the first blocker), (2) fixing whatever Isaac Lab Python assumes about CUDA device strings, graph capture and NVIDIA tooling on that path, (3) running the `kitless` (non-`ov`) pytest subset plus a Newton RL training smoke on AMD, (4) a docs/packaging change upstream (ROCm install section), which NVIDIA may or may not accept. Platform note: AMD Warp lists only gfx942; wave32/Windows coverage is unproven and Windows support for Warp-on-HIP is unknown, so the `windows` and possibly `wave32` gates may need waivers later (agent may only suggest).

### Open questions for jeff

1. `source/isaaclab/isaaclab/envs/mimic_env_cfg.py` carries both `SPDX-License-Identifier: BSD-3-Clause` and "Licensed under the NVIDIA Source Code License [see LICENSE for details]" while the referenced LICENSE is BSD-3-Clause. Treat as a stale header (clean, port may proceed) or as an NVIDIA-licensed file needing a clearance decision before anything goes upstream? The port is not expected to touch this file.
2. Fork settings on `AMD-Ecosystem/IsaacLab` (repository settings are a person's job, not done by intake): the fork has only `release/3.0.0-beta2`, lacks the upstream default branch `main`, and its GitHub default branch is not `moat-port`. Queued decision: sync/create `main` as the upstream mirror and, once the porter creates `moat-port`, set it as the GitHub default branch (per the moat-port default-branch rule). Should the stale `release/3.0.0-beta2` branch stay or go?
3. The work depends on AMD-Ecosystem/warp (1.13 vs pinned 1.17), AMD-Ecosystem/newton (rocm-enablement, 569 behind) and AMD-Ecosystem/mujoco_warp (3.8.1 vs ~3.15). None is a MOAT project. Should MOAT intake/adopt them as MOAT dependencies (so `set-deps` can gate on them), or coordinate with whichever AMD team owns those forks and consume their branches as-is?
