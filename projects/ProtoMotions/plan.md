# ProtoMotions porting plan

Planner run 2026-11-20 on linux-gfx942 (MI300 fleet host), at Jeff Daily's request, planning only: no GPU was used and nothing was built. Upstream read at `NVlabs/ProtoMotions` `main` @ `13de0cccd06f53ce5c9cf9ec7d1932c7da73610f` (2026-11-09; intake screened `b43fc6fc`, 2026-11-03). The fork's `main` mirror is at the same sha. Jeff must rule on the questions in `notes.md` ("Planner (2026-11-20)", top) before a porter starts. Where this plan depends on an answer, it says which one.

## Project

- Name: ProtoMotions (version 3.1.0)
- Upstream: https://github.com/NVlabs/ProtoMotions, default branch `main`
- Fork: https://github.com/AMD-Ecosystem/ProtoMotions, `main` = upstream `13de0ccc`, no `moat-port` yet, push access confirmed (`permissions.push = true`).

## Existing AMD support

Classification: **improvable**. ProtoMotions itself has no AMD support (intake: no ROCm PR, branch, issue or doc mention). Authoritative AMD work exists in the GPU dependency underneath it.

- **Warp** is the only native GPU code on the Newton path. Every kernel ProtoMotions, Newton and MuJoCo Warp launch is a Warp Python kernel that Warp JIT-compiles (hipRTC on ROCm). Three AMD Warp lines exist, all in `AMD-Ecosystem/warp` (`ROCm/warp` redirects there), checked 2026-11-20:
  - `release/1.13.0.amd0` @ `5035c8ed` (`1.13.0+rocm.0`) and `amd-integration` @ `c5bac6df` (`1.13.0+amd0`, 2026-11-19). AMD-official. README support statement: gfx942, Linux, ROCm 7.1.1/7.2.1, build from source (`build_lib.py`); no published wheels. HIP graph capture is supported (`Device.supports_graph_capture` returns True, `context.py:3981-3996`). Conditional graph nodes are not (`is_conditional_graph_supported()` is False on HIP). `WP_TILE_WARP_SIZE` is 64 for every HIP target (`tile_reduce.h`), so tile reductions are wrong on wave32. hipRTC does not define `__CUDA_ARCH__`. HIP build is Linux only.
  - `release/1.14.0.amd0` @ `f6ada527` (`1.14.0+rocm10.2.0`) is MOAT's own `warp-1.14` release-line project. Stage `porting` (Jeff's 2026-11-19 `fork-release` ruling), head moving daily. It already has HIP graph capture with pause/resume, wave-correct tile width (`__GFX9__` gives 64, else 32), and hipRTC defining `__CUDA_ARCH__`. Its plan covers wave32 and Windows. It has completed on no platform yet.
  - `amd-integeration-dev` @ `ac540f59` (`1.18.0.dev2`), the line the IsaacLab port validated on. It reports `supports_graph_capture = False` on HIP. It is not considered further here, since both lines above satisfy ProtoMotions' pins and can capture.
- **Newton** (`AMD-Ecosystem/newton` `rocm-enablement`): one packaging-only commit. Newton is pure Python on Warp. PyPI `newton==1.0.0` requires only `warp-lang>=1.12.0`.
- **MuJoCo Warp** (`AMD-Ecosystem/mujoco_warp`, `release/3.8.1.amd0`, `amd/hip-graph-support`, `amd/solver-convergence-early-exit`): pure Python on Warp. ProtoMotions pins `mujoco-warp==3.5.0.2` (requires `warp-lang>=1.11.0`, `mujoco>=3.5.0`). The AMD branches are performance and graph work on a newer line, and nothing here needs them.

Decision: **consume, do not fork.** Use an AMD Warp build as the GPU backend, and use Newton 1.0.0, MuJoCo Warp 3.5.0.2, MuJoCo 3.5.0 and newton-actuators 0.1.0 unmodified from PyPI at exactly the versions ProtoMotions pins. Which Warp line to use is question 1 for Jeff. MOAT pushes nothing to newton or mujoco_warp. Under the `warp-1.14` option, Warp defects found here go to that project's records, not to this port.

The sibling port `IsaacLab` (review-passed, validated on linux-gfx942 and linux-gfx1100, no regression on a V100) solved the same stack. Its notes give the environment recipe (one ROCm runtime shared by torch and a source-built Warp) and the dependency defects. Read them before starting: `python3 utils/moatlib.py show IsaacLab`, then `git show origin/port/IsaacLab:projects/IsaacLab/notes.md`.

Delivery vehicle: upstream does not link platform forks. It accepts pull requests under a DCO (`CONTRIBUTING.md:45-62`, every commit `Signed-off-by`). Upstream is NVIDIA-owned, so whether to open a PR or keep the fork as the deliverable is decided at the PR gate (intake, and question 5). Keep the diff upstream-shaped either way. Port-versus-rewrite does not arise: the repository has no NVIDIA-tuned kernels, its two Warp kernels are trivial elementwise ones, and Warp compiles them for HIP.

## Build classification

**`python-warp`** (recorded with `set-ext-type`): a pure-Python package (setuptools, `pyproject.toml:1-3`) whose GPU work is torch plus Warp JIT kernels. No `CUDAExtension`, `find_package(Torch)` or compiled extension in the package.

- `surface.py generate`: 1 CUDA file, 77 components. Every generated component comes from `deployment/native/CMakeLists.txt`, a separate optional C++ runtime for robot deployment that the README calls "prototyping reference code ... not a supported release artifact". The Python package has no build-system components, so the planner added them to `surface.json`.
- The one `.cu`, `deployment/native/src/gpc_top_p_plugin.cu`, is a TensorRT plugin (`NvInfer.h`). It is built only under the OFF-by-default `PROTOMOTIONS_BUILD_GPC_TRT_PLUGIN`, next to the also-OFF `PROTOMOTIONS_BUILD_GPC_TENSORRT` and `PROTOMOTIONS_BUILD_METATRON_NATIVE` TensorRT targets.

## Port strategy

Neither A nor B: the "anything else" class, a Python framework on Warp. The port is:

1. an install path that puts ROCm torch, an AMD Warp build and the pinned Newton stack in one environment, without the CUDA `warp-lang==1.12.0` wheel;
2. two small Python fixes where ProtoMotions' CUDA-graph setup assumes features AMD Warp does not have (below);
3. GPU tests and a Newton training smoke on AMD.

No hipify, no compat header.

### Base

`moat-port` is cut from the fork's `main` @ `13de0ccc` (upstream tip as planned). If upstream moves before the porter starts, cut from the new tip and record the sha in notes.

### The two code seams

Both are in `protomotions/simulator/newton/simulator.py`, which is the whole Warp-facing surface of ProtoMotions.

**Seam 1, conditional graph nodes (blocks the default path on every AMD Warp line that can capture).** `_create_simulation` (`simulator.py:162-205`) captures `self._simulate()` with `wp.ScopedCapture` whenever `warp_device.is_cuda`, the Warp mempool is on, and action-latency randomization is off. `is_cuda` is True on HIP devices in AMD Warp, and the mempool is supported there. `_simulate` steps `SolverMuJoCo`. MuJoCo Warp's `io.py:191` defaults `opt.graph_conditional = True`, so `solver.py:3269-3279` calls `wp.capture_while(...)` inside the capture. With no conditional-node support, `capture_while` reaches `assert_conditional_graph_support()` and raises (AMD Warp 1.13 `context.py`, `capture_while` around line 9894). MuJoCo Warp only warns about this (`warp_util.py:145-156`, "graph_conditional should be set to False"), and Newton 1.0.0 never sets the flag (no `graph_conditional` in Newton outside Kamino). So the first `NewtonSimulator` on an AMD GPU should fail during construction.

Fix: in `_setup_sim`, right after the existing `self.solver.mjw_model.geom_margin` / `geom_gap` writes (`simulator.py:439-448`, the same object), set `self.solver.mjw_model.opt.graph_conditional = wp.is_conditional_graph_supported()`. On CUDA 12.4+ that is True, so nothing changes. On HIP, and on a CUDA driver older than 12.4 (the case MuJoCo Warp's warning describes), the solver runs its fixed `iterations` loop inside the graph instead. Converged worlds are marked done per world (`ctx.done`, `nsolving`; `solver.py:3107-3195`) and do no further work, so the cost is extra launches, not wrong physics. ProtoMotions calls `solver.notify_model_changed` for domain randomization (`simulator.py:624, 656`). In Newton 1.0.0 that method updates fields in place and does not reassign `mjw_model` (no `mjw_model =` in its body), so the override should hold. The porter confirms it at runtime. This fix is not ROCm-specific code, so its commit carries no `[ROCm]` prefix.

**Seam 2, Warp builds that cannot capture at all.** If `supports_graph_capture` is False (AMD's 1.18 dev line), `ScopedCapture` runs the body eagerly once and leaves `capture.graph = None`. ProtoMotions would then take one unrequested physics step during construction and raise on the first `wp.capture_launch(None)` (`simulator.py:810-811`). This is IsaacLab's risk 1 in a milder shape (no double stepping per step, because there is no "eager then capture" pattern here). Fix: add one more no-graph reason ahead of the mempool check, `elif not getattr(warp_device, "supports_graph_capture", True): no_cuda_graph_reason = "Warp cannot capture graphs on this device"`. The `getattr` default keeps NVIDIA Warp, which has no such attribute, on its current behaviour. On the Warp lines this plan proposes it is True, so this is defensive. Include it, because it costs one line and turns a crash into a clean eager fallback. If Jeff prefers the smallest possible diff, it is the one item to drop.

Nothing else in ProtoMotions' Python is CUDA-specific on the Newton path. `torch.cuda.*` (8 calls) and `"cuda"` device strings work on ROCm torch. `torch.backends.cudnn` flags are no-ops or map to MIOpen. `torch.compile` (`envs/component_manager.py:281`) uses inductor with Triton, which ROCm torch ships, and already has an eager fallback. The bf16 Muon optimizer runs on any arch with bf16. Lightning Fabric DDP uses the `nccl` backend, which is RCCL on ROCm.

### Dependency set the porter installs (wave64, Linux)

| package | source | note |
|---|---|---|
| torch (2.2 or newer, no exact pin in ProtoMotions) | TheRock wheels from the same ROCm release Warp is built against (`repo.amd.com/rocm/whl/<gfx family>/` or the `whl-next` index `warp-1.14` uses for ROCm 10.2) | one ROCm runtime per process (IsaacLab, Environment recipe). A `download.pytorch.org/whl/rocm*` torch next to a Warp built on another ROCm exposes no GPU |
| `rocm[libraries,devel]` | same index and version as torch | Warp's build and its hipRTC need the headers. Set `ROCM_PATH=$(rocm-sdk path --root)` and prepend `$ROCM_PATH/lib` to `LD_LIBRARY_PATH` (IsaacLab drafts B and C) |
| warp-lang | question 1: `AMD-Ecosystem/warp` `release/1.14.0.amd0` at the sha `warp-1.14` has validated, or `release/1.13.0.amd0` @ `5035c8ed`. Either way: `python build_lib.py --rocm-path $ROCM_PATH --hip-arch=<gfx>`, then install that checkout | if the host also has a CUDA Toolkit, the 1.13 line's `find_cuda_sdk()` takes precedence (IsaacLab draft A; `warp-1.14` errors out and asks for `--rocm-path`) |
| newton[examples]==1.0.0, mujoco-warp==3.5.0.2, mujoco==3.5.0, newton-actuators==0.1.0 | PyPI, unchanged | the AMD Warp build satisfies `warp-lang>=1.12.0` / `>=1.11.0`. Nothing may pull the CUDA `warp-lang` wheel back in |
| ProtoMotions | `pip install -e .` plus a ROCm requirements file (below) | the `newton` extra pins `warp-lang==1.12.0` exactly, so a ROCm install must not use `.[newton]` |

Check after install: `pip list` shows the `+rocm` torch, the AMD Warp version and no `nvidia-*` wheels. `python -c "import warp as wp; wp.init(); d = wp.get_device('cuda:0'); print(wp.__version__, d.is_cuda, getattr(d, 'supports_graph_capture', None), wp.is_conditional_graph_supported())"` prints `True True False` after the version on either proposed line.

## CUDA surface inventory

| item | where | ROCm mapping |
|---|---|---|
| Warp kernels `compute_pd_torques_kernel`, `apply_torques_kernel` | `simulator.py:57-104` | elementwise, no tiles, no native snippets; hipRTC compiles them unchanged |
| Warp graph capture (`wp.ScopedCapture`, `wp.capture_launch`, `use_cuda_graph`) | `simulator.py:162-205, 706, 796-811, 928-943`; `tests/test_newton_warmstart.py:15-21, 81-93` | supported on both proposed Warp lines; seams 1 and 2 above |
| conditional graph nodes (MuJoCo Warp `capture_while`, `graph_conditional`) | reached through `SolverMuJoCo.step` | none on HIP; seam 1 |
| `wp.set_device(str(self.device))`, `"cuda"` / `"cuda:N"` | `simulator.py:147` and tests | AMD Warp keeps the `cuda` alias |
| `wp.from_torch` / `wp.to_torch` zero-copy | `simulator.py` throughout | supported on HIP (IsaacLab uses the same) |
| `torch.cuda.*`, `torch.backends.cudnn`, `torch.compile`, bf16, DDP/NCCL | listed under Port strategy | unchanged on ROCm torch |
| onnxruntime | `deployment/`, export and sim2sim tests | `CPUExecutionProvider` everywhere except `deployment/native/examples/gpc_policy_adapter.py:63-68` (CUDA and TensorRT EPs, deployment layer, question 2) |
| TensorRT, nvcc, `gpc_top_p_plugin.cu` (1 kernel, 1 launch) | `deployment/native` (3 OFF-by-default options) and its Python tools (`tools/*tensorrt*.py`, `bench/gpc_tensorrt_benchmark.py`, `tools/prepare_gpc_policy.py`) | no HIP equivalent to translate into. MIGraphX (or ONNX Runtime's MIGraphX EP) would be a reimplementation, not a port. Question 2 |
| IsaacGym backend | `protomotions/simulator/isaacgym/` | NVIDIA closed binary, CUDA only: no AMD target exists |
| IsaacLab backend | `protomotions/simulator/isaaclab/` | launches Isaac Sim Kit (`isaacsim[all,extscache]==6.0.0.1`), an NVIDIA proprietary binary. Not the Kit-less path MOAT's IsaacLab port covers |
| MuJoCo backend | `protomotions/simulator/mujoco/` | CPU only; vendor-neutral already |

Inside the dependencies (not ProtoMotions code, but on or near its path):

- Newton 1.0.0 `geometry/contact_reduction.py:78-83, 351-360` picks `__syncthreads()` and a `__shared__` buffer with `#if defined(__CUDA_ARCH__)` in `wp.func_native` snippets. Under AMD Warp 1.13's hipRTC that macro is undefined, so those kernels get no barrier and a null shared-memory pointer. They belong to Newton's own collision pipeline (`narrow_phase.py`, `sdf_contact.py`, hydroelastic). ProtoMotions uses MuJoCo Warp's contacts (`SolverMuJoCo` default `use_mujoco_contacts=True`) and never calls `model.collide`, so that path should not be reached. The porter confirms this (below). `warp-1.14` defines `__CUDA_ARCH__` for hipRTC, so the hazard does not exist there.
- Tile reductions: MuJoCo Warp `solver.py`, `forward.py` and `sensor.py` use them with `block_dim = round_up_32(nv)`. G1 has nv = 35 (29 joints plus a free base), giving block_dim 64. SMPL humanoids are larger. On wave32 with AMD Warp 1.13 (tile width hardcoded to 64) those sums are suspect: the IsaacLab gfx1100 validator measured wrong `tile_sum` / `tile_reduce` results at block_dim at least 64 (its Draft F). That run saw no wrong MuJoCo Warp numbers at nv 24-96, but did not trace why, and it is not evidence for G1. On `warp-1.14` the width is wave-correct.

## Risk list

1. **Warp line (question 1).** AMD Warp 1.13 can carry wave64 Linux only. wave32 is at risk from the tile-width defect for every ProtoMotions robot (nv > 32), and there is no Windows HIP build. `warp-1.14` is the only line planned to carry all three gates, but it has not completed anywhere, so a dependency on it delays this port.
2. **Seam 1 incomplete.** Newton may rebuild `mjw_model` (domain randomization calls `notify_model_changed`), dropping the `graph_conditional` override. The test is a captured run with domain randomization on, plus the warm-start test.
3. **Performance without conditional nodes.** The fixed `iterations=100` / `ls_iterations=50` loop is captured as launches, where CUDA exits early on device. Record steps/s and do not gate on it. MuJoCo Warp's `amd/solver-convergence-early-exit` branch is AMD's work on exactly this, for a later round if Jeff wants it.
4. **Newton collision snippets** (above) on the 1.13 line. The porter checks with `WARP_LOG`-style kernel logging or a breakpoint count that no `contact_reduction` / `narrow_phase` kernel launches in a training step. If one does, record it as a Newton/Warp dependency defect (a draft for Jeff, nothing filed) and prefer `warp-1.14`.
5. **Torch and Warp stream ordering.** ProtoMotions writes torch tensors that Warp graphs read, with no explicit stream handoff. It relies on the legacy default stream ordering against Warp's blocking stream. HIP keeps those semantics, but a missing sync shows up as stale actions or observations, not as an error. The warm-start and FK tests compare state after reset and step, which is the check.
6. **Fresh allocations are not zeroed on ROCm.** ProtoMotions allocates only with `wp.zeros` (`simulator.py:395`). Newton and MuJoCo Warp internals are the dependency's concern. Use Warp debug mode (`wp.config.mode = "debug"`) if results drift.
7. **JIT time.** The first run compiles every Newton and MuJoCo Warp kernel through hipRTC: budget minutes, and keep a per-host `WARP_CACHE_PATH`. A hipRTC compile error in a dependency kernel is a dependency finding.
8. **FP divergence** in the iterative MuJoCo solver across archs. Compare against the tolerances the tests already set. Loosen only with a measured, ROCm-gated reason.
9. **Git LFS.** Motion files (`data/motion_for_trackers/*.pt`), meshes and test payloads are LFS. The host may lack `git-lfs`: fetch the needed files from `media.githubusercontent.com/media/NVlabs/ProtoMotions/<sha>/<path>` and check each sha256 against its pointer (as the IsaacLab porter did). Do not fetch `*.ckpt` unless Jeff answers question 3 yes.
10. **Viewer.** `newton.viewer.ViewerGL` runs only when not headless. Compute-only CDNA has no GL, so all runs are `--headless`. A viewer check, if anyone wants one, belongs on an RDNA host with a display stack.
11. **Delivery.** NVIDIA-owned upstream with a DCO (question 5).

## File-by-file change list

On `moat-port`, cut from `13de0ccc`. Python and docs only. This list assumes question 2 is answered "Newton backend only".

1. `protomotions/simulator/newton/simulator.py`: seam 1 (one line in `_setup_sim` plus a short comment) and seam 2 (one `elif` in `_create_simulation`).
2. `protomotions/tests/test_newton_warmstart.py`: change nothing if capture engages on the chosen Warp line, which is expected. Only if seam 2's fallback is exercised (a capture-less Warp), guard the `assert sim.use_cuda_graph` on the device's capture capability, so the eager CUDA-device case still runs. Do not skip the test.
3. A new test is worth adding only if it is cheap: the conditional-node fallback (patch `wp.is_conditional_graph_supported` to False, as MuJoCo Warp's own `io_test.py:1556` does, build a 2-env G1 simulator on the GPU, step it, and assert that capture engaged and the state advances). It exercises seam 1 on NVIDIA too, where it would otherwise be invisible.
4. `requirements_newton_rocm.txt` (new): `requirements_newton.txt` without the `warp-lang==1.12.0` line, with a comment that Warp comes from the AMD build. Do not touch `pyproject.toml` extras or `requirements_newton.txt`: the CUDA install must stay byte-identical.
5. `docs/source/getting_started/installation.rst`: an "AMD GPUs (ROCm)" subsection under Newton. It covers TheRock torch and ROCm SDK wheels for the GPU family, building the chosen AMD Warp from source, `ROCM_PATH` / `LD_LIBRARY_PATH`, installing the pinned Newton stack and `requirements_newton_rocm.txt`, the headless-only note for compute GPUs, and the tested platforms (only those validated). Plus a pointer in the Newton "Requirements" line, which today says "NVIDIA GPU".
6. `docs/source/changelog.rst`: an entry, if upstream keeps one per change (the porter checks recent commits).
7. Unchanged: `Dockerfile.newton` (CUDA base image). A ROCm Dockerfile is optional and out of the minimal port. The IsaacGym, IsaacLab and MuJoCo backends, `deployment/`, and `demos/` stay as they are (question 2).

## Build commands (linux-gfx942 first)

There is no compile step for ProtoMotions. The "build" is the environment plus the Warp build. Wrap each step in `utils/timeit.sh ProtoMotions <phase> --`. Pin a GPU with `HIP_VISIBLE_DEVICES` and use `ulimit -c 0`.

```bash
python3.11 -m venv .venv && . .venv/bin/activate          # Newton docs recommend 3.11+
# 1. torch + ROCm SDK from one TheRock release (index path selects the GPU family)
uv pip install --index-url https://pypi.org/simple --extra-index-url <TheRock index for gfx94X> \
    "torch==<ver>+rocm<R>" "rocm[libraries,devel]==<R>"
export ROCM_PATH=$(rocm-sdk path --root)
export LD_LIBRARY_PATH=$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# 2. AMD Warp (question 1 picks the branch/sha), outside the ProtoMotions tree
git clone https://github.com/AMD-Ecosystem/warp.git ../warp-rocm && git -C ../warp-rocm checkout <sha>
(cd ../warp-rocm && python build_lib.py --rocm-path "$ROCM_PATH" --hip-arch=gfx942 && uv pip install -e .)
# 3. ProtoMotions on the fork's moat-port
git clone -b moat-port https://github.com/AMD-Ecosystem/ProtoMotions.git && cd ProtoMotions
uv pip install -e ".[dev]" -r requirements_newton_rocm.txt
python -c "import torch, warp, newton, mujoco_warp; print(torch.__version__, torch.version.hip, warp.__version__, newton.__version__, mujoco_warp.__version__)"
```

`[dev]` brings pytest and onnx. Make sure the resolver keeps the local Warp: if it tries to replace it, use the same `--override` mechanism the IsaacLab recipe used. For another arch, change only the index family and `--hip-arch`.

## Test plan

**GPU tests (the gate).** Run one pytest process per file from the checkout root, `MUJOCO_GL=egl` or `osmesa` as needed:

- Newton on the GPU: `test_newton_warmstart.py` (the `cuda` param asserts that captured stepping engaged, so it exercises seam 1 directly), `test_newton_contact_matrix.py` (its GPU case, "SolverMuJoCo needs CUDA Warp"), `test_newton_simulator_fk.py` (defaults to `cuda` when available), `test_newton_explicit_control.py`, `test_newton_body_velocity.py`, `test_newton_domain_randomization.py` (`test_cuda_bucket_indexing_after_move`), `test_newton_contact_body_labels.py`, `test_newton_device_selection.py`, `test_newton_python3_compat.py`.
- Other torch-GPU cases: `test_ppo_exploration.py::test_cuda_forward_backward_and_projection`, `test_baked_fk.py`, `test_contact_matrix.py`, `test_self_contacts.py`, `test_cross_simulator_robot_friction.py`. Their GPU or Newton parametrizations run where the skip conditions allow. Read each skip reason from the junit XML.
- Confirm the GPU ran: the warm-start `cuda` case must not skip, and Warp's init banner must show the AMD device. A green result with every `cuda` case skipped is not a pass.

**End-to-end training smoke (Newton, headless).**

```bash
python protomotions/train_agent.py --robot-name g1 --simulator newton \
    --experiment-path examples/experiments/mimic/mlp.py --experiment-name rocm_smoke \
    --motion-file data/motion_for_trackers/g1_random_subset_tiny.pt \
    --num-envs 4096 --batch-size 16384 --ngpu 1 --training-max-steps <a few hundred iterations' worth> --headless true
```

The form comes from `docs/source/getting_started/quickstart.rst:112-120` with `--simulator newton`. Headless is spelled `--headless true` (`train_agent.py:16`). The porter confirms the remaining flags and the step budget. Pass bar: no traceback; the log shows "Using CUDA graph"; tracking reward rises and episode length grows over the run (record the numbers at fixed iterations); steps/s recorded, not gated. Also run one short training with action-latency randomization or domain randomization enabled, if an example config has it. That covers the eager branch and the `notify_model_changed` path (risk 2). A CUDA reference curve for the same command is wanted if an NVIDIA host runs it (below).

**Non-GPU regression set.** `pytest protomotions/tests/ -q -m "not needs_lfs"` on the ROCm environment, compared against the same command on upstream `13de0ccc` in the same environment. That is upstream's own PR CI selection (`.github/workflows/public-ci.yml`, "Tests (CPU)"), which installs `.[mujoco,dev]`. Same pass/skip/fail counts, except tests that newly run because a GPU is present. `needs_lfs` tests read pretrained checkpoints: run them only if Jeff answers question 3 yes.

**CUDA no-regression.** No CUDA-compiled code is touched, so an nvcc compile gate does not apply. The CUDA install files (`pyproject.toml`, `requirements_newton.txt`, `Dockerfile.newton`) stay byte-identical. Seam 1 keeps `graph_conditional = True` where conditional nodes exist. Seam 2's `getattr` default keeps NVIDIA Warp capturing. Owed as runtime evidence, as for IsaacLab: on an NVIDIA host (the V100 fleet host; NVIDIA warp-lang 1.12.0 with sm_70 is fine), run the Newton test files above and the smoke on `moat-port` and on `13de0ccc`. Show identical test ids and counts, "Using CUDA graph" in both, and `graph_conditional` still True.

**Platform order.** linux-gfx942 first (wave64). Then wave32 (linux-gfx1100 or gfx1201), only on a Warp line with the wave-correct tile width (question 1). windows-gfx1151 last and only if question 4 keeps the gate. Never run Newton training at 4096 envs on the laptop. Use a small `--num-envs` there.

## Open questions

Stated for Jeff at the top of the `notes.md` section "Planner (2026-11-20)". No porter starts until they are ruled.

1. Which AMD Warp line (and so whether `depends_on: [warp-1.14]`).
2. Scope: Newton backend only, with the TensorRT deployment path, IsaacGym and the IsaacLab (Kit) backend out.
3. Pretrained checkpoints under the NVIDIA Open Model License in validation (carried from intake).
4. Windows gate: keep it, or a waiver suggestion (upstream documents Linux only).
5. Delivery and DCO sign-off.
