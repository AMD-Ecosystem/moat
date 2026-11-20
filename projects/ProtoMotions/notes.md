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

## Planner (2026-11-20, linux-gfx942 fleet host, planner role)

Requested by Jeff Daily as a headless fleet job. Planning only: no GPU used, nothing built. Upstream read at `13de0cccd06f53ce5c9cf9ec7d1932c7da73610f` (2026-11-09), which is also the fork's `main`. `plan.md` and `surface.json` written. `ext_type` = `python-warp`.

### Decisions needed from Jeff before a porter starts

Nothing below has been assumed. The porter must not start until each item has a ruling.

1. **Which AMD Warp line does the port build on?** ProtoMotions pins `warp-lang==1.12.0`. Newton 1.0.0 needs `>=1.12.0` and MuJoCo Warp 3.5.0.2 needs `>=1.11.0`. Both candidates satisfy those floors and both can capture HIP graphs:
   - (a) **MOAT's `warp-1.14`** (`AMD-Ecosystem/warp` `release/1.14.0.amd0`, your 2026-11-19 fork-release ruling), recorded with `python3 utils/moatlib.py set-deps ProtoMotions warp-1.14`. It is the only line with a wave-correct tile width (wave32) and a planned Windows build, and its hipRTC defines `__CUDA_ARCH__`, so Newton's native snippets take their device branch. Cost: the selector holds ProtoMotions until `warp-1.14` completes on a Linux platform. Today it is still `porting` and has completed nowhere.
   - (b) **AMD's own `release/1.13.0.amd0`** @ `5035c8ed` (`1.13.0+rocm.0`; `amd-integration` is the same code). Usable now, AMD-owned, gfx942 Linux only. wave32 would be blocked: the tile width is hardcoded to 64, and MuJoCo Warp's solver reduces at block_dim 64 for G1 (nv=35). There is no Windows build. So wave64 is the only gate it can carry.
   - (c) Both in sequence: build and validate wave64 on (b) now, then move the documented recipe to `warp-1.14` once it completes, as a later round (revalidation on every platform then).
   - Planner recommendation: (a). The ProtoMotions code change is the same under every option, and only (a) can satisfy all three gates without a second round. (c) is the choice if a wave64 result is wanted sooner.
2. **Scope (scope-cut rule): port only the Newton backend?** Proposed: Newton is in scope, plus the shared training, agent and test code it runs through. The vendor-neutral MuJoCo CPU backend is kept as the existing CPU tests cover it. Proposed out, each for a stated reason:
   - the IsaacGym backend (NVIDIA closed binary, no AMD target);
   - the IsaacLab backend (it launches Isaac Sim Kit, `isaacsim[all,extscache]==6.0.0.1`, NVIDIA proprietary; not the Kit-less path our IsaacLab port covers);
   - the TensorRT deployment path: `deployment/native` options `PROTOMOTIONS_BUILD_GPC_TENSORRT`, `PROTOMOTIONS_BUILD_GPC_TRT_PLUGIN` (the repository's only `.cu`, `gpc_top_p_plugin.cu`) and `PROTOMOTIONS_BUILD_METATRON_NATIVE`, plus the TensorRT Python tools and the CUDA/TensorRT EP choices in `deployment/native/examples/gpc_policy_adapter.py`. No HIP target exists to translate into. MIGraphX would be a reimplementation (option: register it with `deferred.py` as later work instead of a plain cut).
   - The CPU-only C++ deployment runtime (MuJoCo, Unitree SDK2, evdev, SDL2, Pico input; default build = library + evdev) contains no GPU code and the port does not touch it. Options: (i) cover it with a configure, build and ctest of the default targets on the AMD host as a no-regression check, or (ii) scope it out as untouched CPU code.
   - The Newton GL viewer: CDNA has no GL. Options: check it on an RDNA host with a display stack, or scope it out (training is headless).
   - These items are deliberately left unaccounted in `surface.json` (see its `planner_note`) until you rule.
3. **Pretrained checkpoints (carried from intake, still open).** The NVIDIA-released weights are under the NVIDIA Open Model License. May validation download and run them (the `needs_lfs` tests that strict-load `inference_last*.ckpt`, and pretrained sim2sim evaluation)? If not, validation trains its own short Newton runs and the `needs_lfs` tests are not run. The plan defaults to not running them.
4. **Windows gate.** Upstream documents Linux only (`pyproject.toml` classifier `Operating System :: POSIX :: Linux`; no Windows in `installation.rst`), and no AMD Warp line has a Windows HIP build yet (`warp-1.14` plans one). Keep the gate and validate windows-gfx1151 last on `warp-1.14`, or have me record a `windows` waiver suggestion (OS obstacle: the upstream is Linux-only) for you to approve or reject? An agent records no waiver.
5. **Delivery and DCO.** Upstream (NVIDIA-owned) takes PRs only with `Signed-off-by` on every commit (`CONTRIBUTING.md:45-62`). Should porter commits on `moat-port` carry `Signed-off-by: Jeff Daily` (a DCO certification in your name), or should signing wait for the PR gate (a squash you sign yourself), or is the fork the deliverable with no PR (intake's expectation for NVIDIA-owned repos)? The plan keeps the diff upstream-shaped either way.

### Findings that shaped the plan (details and line numbers in plan.md)

- **The default Newton path should fail on AMD at construction, on any Warp line that can capture.** ProtoMotions captures `_simulate` whenever `warp_device.is_cuda` (True on HIP) and the mempool is on. MuJoCo Warp defaults `opt.graph_conditional = True` and calls `wp.capture_while` inside that capture. AMD Warp has no conditional graph nodes, so `assert_conditional_graph_support()` raises. Fix: one line in `_setup_sim`, `self.solver.mjw_model.opt.graph_conditional = wp.is_conditional_graph_supported()`. It is a no-op on CUDA 12.4+ and also fixes the CUDA < 12.4 case MuJoCo Warp only warns about. Inferred from source, not run; the porter's first GPU run confirms it.
- Defensive second line: treat `getattr(device, "supports_graph_capture", True) is False` as a no-graph reason, so a capture-less Warp (AMD's 1.18 dev line) falls back to eager stepping instead of raising on `capture_launch(None)`.
- AMD's Warp has moved since the IsaacLab plan. `release/1.13.0.amd0` and `amd-integration` (2026-11-16 / 11-19) now report `supports_graph_capture = True` on HIP. The 1.18 dev line used for IsaacLab still reports False.
- Newton 1.0.0's collision pipeline has `__CUDA_ARCH__`-only native snippets (`contact_reduction.py`: `__syncthreads`, `__shared__` pointer), which break under AMD Warp 1.13's hipRTC. ProtoMotions uses MuJoCo Warp contacts and never calls `model.collide`, so that path should be unreachable. The porter verifies (plan risk 4). It is moot on `warp-1.14`.
- Everything else on the path is unchanged on ROCm torch: `torch.cuda.*`, `torch.compile`, bf16, DDP over RCCL, and onnxruntime on CPU.
- `requirements_newton.txt` and the `newton` extra pin `warp-lang==1.12.0` exactly. The ROCm install uses a new `requirements_newton_rocm.txt` and leaves the CUDA files byte-identical.
