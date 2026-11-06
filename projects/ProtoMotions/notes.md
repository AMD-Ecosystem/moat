# ProtoMotions notes

## Intake screen (2026-11-05, windows-gfx1151 host, intake role)

Requested by jeff (Jeff Daily) while he is away; questions for him are in "Open questions for jeff" below. Source: third-party "AMD AI Software Gap Matrix" seed (matrix verdict: unknown, AMD path unknown). Screened against upstream `NVlabs/ProtoMotions` `main` @ `b43fc6fcfebb4996eb651dda8ea8ce981c5433a6` (2026-11-03), shallow clone in `agent_space/ProtoMotions-screen`.

### What it is

NVIDIA-owned (NVlabs) GPU-accelerated humanoid motion RL framework (2.5k stars, active: pushed 2026-11-03, not archived). Python on torch + Lightning. Simulator backends on main (`protomotions/simulator/`): `isaacgym`, `isaaclab`, `newton`, `mujoco`. Genesis was removed in the 2026-10-28 refactor (README). Per backend:

- IsaacGym (Preview 4): NVIDIA closed binary, CUDA-only, deprecated. Not portable; out of scope.
- IsaacLab 3.0: `pyproject.toml` extra pulls `isaaclab==12.0.0` AND `isaacsim[all,extscache]==6.0.0.1`; `simulator/isaaclab/simulator.py` uses the Kit app launcher. That is the Kit path the IsaacLab screen ruled out, not the Kit-less Newton path. Not an AMD route; out of scope.
- Newton: `requirements_newton.txt` pins `newton[examples]==1.0.0`, `warp-lang==1.12.0`, `mujoco-warp==3.5.0.2`, `mujoco==3.5.0`, `newton-actuators==0.1.0`. ProtoMotions calls Newton directly (`protomotions/simulator/newton/`); it does NOT go through Isaac Lab. This is the AMD route.
- MuJoCo: CPU-only, single env (inference / sim2sim). Already vendor-neutral.

### Licence

- GitHub field Apache-2.0; `licenses.py check` = Apache-2.0 tier 1, cleared to contribute. `LICENSE.md` read: Apache-2.0. `deployment/native/LICENSE`: Apache-2.0.
- `python3 utils/licenses.py scan-nvidia agent_space/ProtoMotions-screen`: no NVIDIA proprietary licence text. No `.gitmodules`.
- Unitree robot meshes (`protomotions/data/assets/mesh/{G1,H1_2}/LICENSE`): BSD-3-Clause, Unitree Robotics. Clean.
- Pretrained model weights (`data/pretrained_models/*/LICENSE.pdf`, `legal/MODEL-LICENSING.md`): NVIDIA Open Model License for NVIDIA-released weights/checkpoints only; source stays Apache-2.0 and the policy says it does not apply to models trained with the code. This is a USE question (whether MOAT may download and run those checkpoints in validation), not a contribution question; the scanner does not read PDFs. A port need not touch or redistribute them; validation can train its own short runs instead. Recorded, flagged as a question.
- NVIDIA ownership: an upstream PR (even docs) is unlikely to be accepted; per jeff's 2026-11-05 policy the deliverable would be the AMD-Ecosystem fork, and the licence is clean file by file.

### Existing AMD effort / duplicate check

- Upstream docs/README: `grep -rniE 'amd|rocm|hip|gfx'` finds only robot hip joints. No AMD mention.
- `gh pr list --state all --search "rocm OR amd OR hip"`: no ROCm PR (#199 "Develop", #177 visualizer fix). Branches: none rocm/hip/amd (`feat/newton-1.0` and release branches only).
- No `AMD-Ecosystem/ProtoMotions` or `ROCm/ProtoMotions`; no fork of NVlabs/ProtoMotions with amd/rocm/hip in the name; `gh search repos ProtoMotions` finds none in an AMD org. Web search "ProtoMotions ROCm AMD GPU": nothing. (`amd` and `AMD-AGI` orgs are SAML-gated for this token, so a private effort there cannot be ruled out.)
- GPU dependency forks (none is a MOAT project, so no `set-deps`; reused from the IsaacLab screen on `port/IsaacLab` and re-checked here):
  - Warp: `AMD-Ecosystem/warp`, `amd-integration` = `1.13.0+rocm.0`, branch `release/1.13.0.amd0`; also an `amd-integration-halo` branch (Strix Halo / gfx1151 work, last commit 2026-08-21 "initial fix to cdna3 tests"). ProtoMotions pins `warp-lang==1.12.0`; newton 1.0.0 on PyPI requires only `warp-lang>=1.12.0`. AMD Warp 1.13 is AHEAD of the pin, so the skew is one minor version forward (compatible direction), unlike Isaac Lab (1.17 pin vs 1.13 fork).
  - Newton: `AMD-Ecosystem/newton` `rocm-enablement` is a single commit touching only `pyproject.toml` (+8/-1: a `rocm` extra that drops the CUDA warp wheel). So Newton runs on AMD Warp with no source changes; PyPI `newton==1.0.0` should work unmodified once `warp-lang` is satisfied by AMD Warp (to verify).
  - MuJoCo Warp: ProtoMotions pins `mujoco-warp==3.5.0.2` (requires `warp-lang>=1.11.0`, `mujoco>=3.5.0`). `AMD-Ecosystem/mujoco_warp` carries AMD fixes at 3.8.1 (`release/3.8.1.amd0`, `amd/hip-graph-support`, `amd/solver-convergence-early-exit`), needing `mujoco>=3.8`. Whether upstream 3.5.0.2 runs correctly on AMD Warp without those fixes is the main unknown; otherwise ProtoMotions' Newton pins need bumping to the AMD mujoco_warp line (and newton/mujoco with it).
  - torch: ROCm wheels exist; ProtoMotions core deps leave the torch index to the user (`torch>=2.2`), and `Dockerfile.newton` uses `nvidia/cuda:12.4.0` + the cu124 index.
- Isaac Lab dependency: NOT needed. ProtoMotions' Newton backend imports Newton directly, and its Isaac Lab backend is the Kit path. No `set-deps IsaacLab`. The IsaacLab screen's finding (Kit-less Newton is the AMD route, gated on AMD Warp/Newton/MuJoCo-Warp forks) applies here by analogy and with a smaller version gap.

### CUDA surface

- One `.cu` file: `deployment/native/src/gpc_top_p_plugin.cu` (11.5 KB, Apache-2.0 SPDX, NVIDIA copyright): an exact top-p sampling TensorRT plugin (`NvInfer.h`, `NvInferPlugin.h`), built only with the OFF-by-default CMake option `PROTOMOTIONS_BUILD_GPC_TOP_P_PLUGIN`, alongside the also-OFF `PROTOMOTIONS_BUILD_GPC_TENSORRT` / Metatron TensorRT targets. README calls `deployment/` "prototyping reference code ... not a supported release artifact". OPTIONAL: TensorRT has no HIP equivalent (MIGraphX would be a reimplementation, not a port). Recommend out of scope; the rest of `deployment/native` (MuJoCo + Unitree SDK C++ runtime) is CPU code.
- Python: `torch.cuda` x37 and `"cuda:0"` x22 (fine on ROCm torch); Warp graph capture in `simulator/newton/simulator.py` (`wp.ScopedCapture` L193, `wp.capture_launch` L869) and `tests/test_newton_warmstart.py` -- depends on AMD Warp HIP graph support (a config path to disable capture should exist as fallback; planner to check). No nvidia-smi/pynvml/cupy/tensorrt in Python. `onnxruntime` x36 (export/MuJoCo inference, CPU EP; ONNX Runtime has no ROCm EP, so CPU only).

### Tests

251 `test_*.py` under `protomotions/tests/` (26 mention newton, 12 mention cuda). Upstream CI (`.github/workflows/public-ci.yml`) is CPU-only on `ubuntu-latest`: `pytest protomotions/tests/ -q -m "not needs_lfs"`. GPU evidence would be the newton/cuda tests on an AMD GPU plus a short Newton training smoke (e.g. G1 / H1_2 motion tracking for a few hundred iterations) and checkpoint eval.

### Viability

Yes. Not a CUDA-to-HIP translation; it is the "little or no CUDA source" class under jeff's 2026-11-05 policy: (1) install the Newton backend on ROCm against AMD Warp (1.13 >= pinned 1.12) and resolve mujoco-warp 3.5.0.2 vs the AMD 3.8.1 line, (2) fix any ProtoMotions Python assumptions on that path (graph capture, device strings), (3) run the GPU test subset plus a Newton training smoke on AMD, (4) a ROCm install section / `requirements_newton_rocm.txt` or extra on the fork (upstream unlikely to take it). Smaller and less version-blocked than Isaac Lab. Platform note: AMD Warp lists gfx942 as supported; the `amd-integration-halo` branch suggests gfx1151 work exists, but Windows Warp-on-HIP and wave32 are unproven, so `windows`/`wave32` waivers may be needed later (agent may only suggest).

### Open questions for jeff

1. Pretrained ProtoMotions checkpoints are under the NVIDIA Open Model License (source is Apache-2.0). May MOAT validation download and run those weights, or should validation train its own short Newton runs only? The port does not need to touch or redistribute them.
2. Scope: drop the optional TensorRT top-p plugin (`deployment/native/src/gpc_top_p_plugin.cu`, the only CUDA source; OFF by default, experimental) and the IsaacGym / IsaacLab (Kit) backends, porting only the Newton backend (plus vendor-neutral MuJoCo)? Per the scope-cut rule this needs your explicit answer before a porter starts.
3. Same as IsaacLab's question 3: AMD-Ecosystem/warp, /newton and /mujoco_warp are not MOAT projects. Consume them as-is (coordinate with their owners) or intake them as MOAT dependencies so `set-deps` can gate on them? ProtoMotions needs warp (>=1.12) and mujoco_warp (3.5.0.2 pinned vs AMD 3.8.1).
