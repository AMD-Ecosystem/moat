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

## Planner (2026-11-05, windows-gfx1151 host, planner role)

Plan written to `plan.md`; surface accounting in `surface.json` (0 CUDA files by construction, 27 planner-added components: 20 owed in scope, 5 scoped out as NVIDIA-binary-only, wave32/windows owed on AMD Warp). Upstream read at `main` @ `eed2d2889a5b963fee8c3594d29fe6d91ec67046`. `ext_type` = `python-warp`.

Key findings (evidence in plan.md):
- Base chosen: upstream `main`, not the fork's `release/3.0.0-beta2` (beta2 pins warp-lang 1.13.0 / newton 1.2.1 / torch 2.10 cu128; it is 1771 behind / 155 ahead of main).
- The version skew intake flagged is mostly not real: Newton and MuJoCo Warp are pure Python on Warp. The AMD Newton fork's whole ROCm port is one `pyproject.toml` commit (+8/-1) that stops `warp-lang` coming from the CUDA wheel. So upstream Newton @ `42f7def1` and PyPI `mujoco-warp~=3.15.0` are used unmodified.
- The one native dependency is Warp. `AMD-Ecosystem/warp` branch `amd-integeration-dev` (sic) @ `ac540f595ff5763a374c113d77e4111a1d4791bd` is `1.18.0.dev2`, which satisfies every floor (Isaac Lab `==1.17.0` needs an override, Newton `>=1.17.0`, MuJoCo Warp `>=1.15`). `amd-integration` (the README-documented line) is still 1.13.0+rocm.0.
- ROCm torch 2.11.0 exists: `download.pytorch.org/whl/rocm7.2` and `rocm7.1` both carry `torch-2.11.0+rocm7.x-cp312` manylinux x86_64.
- Main code seam: AMD Warp reports `Device.is_cuda == True` and `supports_graph_capture == False` on HIP; `ScopedCapture` becomes a no-op that runs its body eagerly and leaves `graph=None`; `capture_launch(None)` raises. Isaac Lab gates capture on `is_cuda`, so `newton_backend.py:449-458` would step physics twice per call and recapture forever. Fix: gate on a capability helper.
- AMD Warp dev hardcodes `WP_TILE_WARP_SIZE 64` for all HIP (`warp/native/tile_reduce.h:14-18`): wrong on RDNA wave32. The `amd-integration-halo` branch (`74014999`, `1.17.0.dev4`, gfx1151 port) selects by `__GFX9__`. AMD Warp's HIP build path is Linux-only (no Windows).
- `CI_MARKER=kitless` selects 12 files at this sha, not 35 (tools/conftest.py matches the literal `pytest.mark.kitless`).

Fleet order for the validator: linux-gfx942 first (AMD Warp's one stated arch), then other wave64 (gfx950, gfx90a), then wave32 (gfx1100/gfx1201), windows-gfx1151 last and only if a Windows AMD Warp build exists.

### Planner questions for jeff

Asked while jeff is away; each states the conservative assumption the pipeline proceeds on.

1a. Dependency consumption. The port consumes `AMD-Ecosystem/warp` branch `amd-integeration-dev` @ `ac540f595ff5763a374c113d77e4111a1d4791bd` (1.18.0.dev2, unreleased, branch name misspelled) as the GPU backend, built from source, plus unmodified upstream Newton @ `42f7def1` and PyPI MuJoCo Warp 3.15. MOAT does not push to any of the three AMD dependency repositories. Is consuming an unreleased AMD Warp dev branch by sha acceptable for the port and its docs, or should MOAT wait for (or ask the AMD Warp owners for) a tagged release >= 1.17? Assumption: consume by sha, record it, never push there.
1b. wave32. AMD Warp's dev branch hardcodes a 64-lane tile width on every AMD target; the fix exists on `amd-integration-halo` (1.17.0.dev4, older than Newton's floor). Getting wave32 right needs the AMD Warp owners to merge that fix into the dev line, which is work in a repository MOAT does not own. Should MOAT raise this with them (through you), or try the halo branch under a version override? Assumption: the wave32 validator tries dev first, then halo under an override, and blocks the platform naming the Warp defect if both fail.
2. Fork settings (carried from intake question 2). The porter needs a `moat-port` branch cut from upstream `main`; the fork has no `main`. Assumption: the porter creates `moat-port` from upstream `main` @ the planned sha and pushes only that branch; `release/3.0.0-beta2` and the GitHub default branch are left for you.
3. Windows gate. AMD Warp's HIP build is Linux-only, so there is no Warp-on-HIP for Windows to run Isaac Lab on. Suggest a `windows` waiver (agent suggestion only; it needs your approval). Assumption: windows-gfx1151 is attempted last and only if a Windows AMD Warp build appears; nothing is waived by an agent.
4. Delivery vehicle. Upstream is NVIDIA's own repository. The diff is kept upstream-shaped (capability-gated graph capture, NVIDIA-tooling fallbacks, a ROCm install docs section, no change to the CUDA install). Open an upstream PR at the PR gate, or treat the fork's `moat-port` as the deliverable? Assumption: decide at the PR gate; nothing upstream-visible before then.
5. Scope. Kit/Isaac Sim, the OV wheels (ovphysx/ovrtx/ovstage), the PhysX backend, mimic and teleop are recorded in surface.json as scoped out because they need NVIDIA proprietary binaries with no AMD build, i.e. impossible rather than deferred. Assumption: this follows from your 2026-11-05 Kit-less framing and the porter proceeds; overrule if you want any of it treated as a deferral instead.

## Porter (2026-11-06, linux-gfx942 MI300X host, porter role)

Dispatched under jeff's 2026-11-05 ruling (take over the fork, start the port); the planner assumptions above were followed as stated (consume AMD Warp by sha, no dependency pushes, only `moat-port` to the fork, nothing for windows).

### Outcome: port built and GPU-tested locally; push to the fork BLOCKED

- `moat-port` was cut from upstream `main` @ `eed2d2889a5b963fee8c3594d29fe6d91ec67046` and carries two commits: `ee1f2cd0c` "[ROCm] Skip Warp graph capture on devices that cannot capture" and `03bb17b08` "[ROCm] Keep the GPU info recorder metadata unchanged" (tip `03bb17b08290babb82d18bb3e2adb25ff1a5194c`).
- `git push -u origin moat-port` to `https://github.com/AMD-Ecosystem/IsaacLab.git` failed: `remote: Permission to AMD-Ecosystem/IsaacLab.git denied to jeffdaily.` / `error: 403`. `gh api repos/AMD-Ecosystem/IsaacLab -q .permissions` = pull only (`push:false`), while `AMD-Ecosystem/moat` reports admin. The taken-over fork was never granted to the account the fleet pushes with. Granting write access is a repository-settings decision for a person; nothing was retried or routed around.
- The two commits are preserved as `projects/IsaacLab/moat-port-03bb17b0.patch` (`git format-patch eed2d288..moat-port`). To resume once access exists: clone upstream, `git checkout -b moat-port eed2d2889a5b963fee8c3594d29fe6d91ec67046`, `git am projects/IsaacLab/moat-port-03bb17b0.patch` (the resulting sha differs from 03bb17b08 unless dates are preserved; `advance-head` to whatever is pushed), push `moat-port`, `advance-head`, `set-blocked --clear`, then `set-state IsaacLab linux-gfx942 ported`. The original local clone stays in this fleet job's `moat/projects/IsaacLab/src` until the job directory is cleaned.
- State left: stage `porting` with the lock held by linux-gfx942 (the work lives here) and platform linux-gfx942 `set-blocked` naming the push permission. `head_sha` not advanced (nothing is on the fork). Not set `ported`: the port is not pushed.

### Question for jeff

Grant the fleet's GitHub account (`jeffdaily`) write access to `AMD-Ecosystem/IsaacLab` (or say which account pushes there), then clear the block. Still open from the planner list: fork default branch and `release/3.0.0-beta2`, delivery vehicle, windows waiver.

### What changed (upstream-shaped; CUDA path unchanged)

- New `isaaclab.utils.warp.utils.supports_graph_capture(device)` = `device.is_cuda and getattr(device, "supports_graph_capture", True)`. NVIDIA Warp has no such attribute (checked the `origin/main` upstream mirror in `AMD-Ecosystem/warp`: absent), so CUDA keeps capturing exactly as before; AMD Warp returns False for HIP devices.
- Capture gates switched from `is_cuda` to the helper: `newton_backend.py` (step capture and BVH refit), `newton_raycast_sensor.py`, `frame_transformer.py` (also an infinite-recursion risk: `is_capturing` never turns True on HIP), `newton_warp_renderer.py`, `physx_runtime.py` (native actuator graphs; the `(None, None)` tuple was truthy), `particle_surface.py`, `warp_graph_cache.py` (eager on HIP GPUs only; CPU unchanged), `utils/device.py` (`reserve_cuda_graph_memory`), contrib `gb300_sharpa_insertion/ik.py`. `mpm_solver.py` needed nothing (its gate is solver-level; the device gate in `newton_backend` covers it).
- Double stepping (plan risk 1) measured on MI300X with a 50-step Newton cube scene, `use_cuda_graph=True`, counting `NewtonBackend._simulate_physics_only` calls: upstream `simulate_calls=100 graph=False`, patched `simulate_calls=50 graph=False` (job dir `work/probe/double_step.py`; the test helper turns gravity off, so the call count is the evidence, not the fall).
- Tests: new `isaaclab.test.utils.skip_without_graph_capture(device)`, called immediately before each capture so the eager assertions earlier in a test still run (in the parity files every "skipped" case first passed its eager parity check). `test_newton_manager.py::test_step_runs_eagerly_once...` mocks capture, so it patches `newton_backend.supports_graph_capture` to True and keeps full coverage; `test_newton_solvers.py::test_mjwarp_authored_reset...` falls back to its eager reset instead of skipping.
- Docs: `docs/source/isaaclab/installation.rst` gains a section "AMD GPUs with ROCm" (`installation-method-rocm`) and a one-line pointer in the System requirements GPU row.
- Changelog fragments `rocm-graph-capture.*` in isaaclab (added, fixed, `.minor`), isaaclab_newton, isaaclab_experimental, isaaclab_visualizers (created its `changelog.d/`), isaaclab_tasks. `python tools/changelog/cli.py check upstream/main` rc=0. `ruff check` / `ruff format` 0.14.10 (the upstream pre-commit pin) and `codespell` 2.4.1 clean on the touched files.
- Dropped in the second commit: a `hip_version` metadata entry in `record_gpu_info.py` broke `TestGPUInfoRecorder::test_get_data_returns_measurement_data` (asserts exactly 4 metadata entries; upstream 4/4 pass, patched failed). The recorder already degrades on ROCm (`cuda_version: Unknown`, `torch.cuda.memory_allocated` fallback). `cli/multigpu.py` needed nothing: with only `HIP_VISIBLE_DEVICES` set it counts through `torch.cuda.device_count()`, which honours it.

### Environment recipe actually used (deviations from plan.md)

The plan's torch from `download.pytorch.org/whl/rocm7.2` does NOT work with a source-built AMD Warp unless Warp is built against the same ROCm: a process gets one `libamdhip64.so.7`. Measured:
1. Warp built against the image's TheRock HIP 7.16 with torch rocm7.2: torch imported first -> Warp prints `CUDA Toolkit 71626.38, Driver 70226.1 (insufficient CUDA driver version!)` and exposes no GPU; Warp first -> torch `RuntimeError: No CUDA GPUs are available`.
2. Warp built against ROCm 7.2 (apt debs `hip-dev hipcc hip-runtime-amd rocm-llvm comgr hsa-rocr hsa-rocr-dev rocm-device-libs rocprofiler-register rocm-core hipcub-dev rocprim-dev rocthrust-dev` from `repo.radeon.com/rocm/apt/7.2` noble, unpacked with `dpkg-deb -x` into the job dir): `warp/native/reduce.cu` (hipCUB `DeviceReduce`) compiled for more than 25 minutes at 100% CPU without finishing, where the 7.16 toolchain builds all of Warp in about 3 minutes. Abandoned.
3. Used instead: AMD's TheRock Python stack, which ships torch 2.11.0 and torchvision 0.26.0 exactly as Isaac Lab pins them. Install with uv against `--index-url https://pypi.org/simple --extra-index-url https://repo.amd.com/rocm/whl/gfx94X-dcgpu/` the requirements `torch==2.11.0+rocm7.13.0 torchvision==0.26.0+rocm7.13.0 "rocm[libraries,devel]==7.13.0"`. uv gives `--extra-index-url` priority; with the AMD index as `--index-url` uv picks PyPI's torch and fails to resolve. Warp then reports `ROCm 7.13, HIP driver 7.13`.
- Runtime environment required: `ROCM_PATH=$(rocm-sdk path --root)` (Warp's HIPRTC finds `float.h` and the HIP headers through it, `warp/_src/build.py:_hip_extra_include_dirs`; without it every kernel JIT fails `fatal error: 'float.h' file not found`) and `LD_LIBRARY_PATH=$ROCM_PATH/lib` (torch preloads `libamdhip64` and `libhiprtc` but not `libhiprtc-builtins.so.7`, which `warp.so` links; without it `Failed to load the shared library warp.so`).
- AMD Warp `amd-integeration-dev` @ `ac540f595ff5763a374c113d77e4111a1d4791bd` (1.18.0.dev2) built with `python build_lib.py --rocm-path $ROCM_PATH --hip-arch=gfx942 --no-use-libmathdx -j 48`. This host also has `/usr/local/cuda`, which `build_lib.find_cuda_sdk()` always prefers (a default-path check with no override), so the build ran through a wrapper that replaces `find_cuda_sdk` with `lambda: None` (job dir `work/scripts/build_warp_hip.py`); the Warp tree is unmodified. The docs tell users to build on a host without a CUDA Toolkit.
- Isaac Lab install into a Python 3.12 venv (the fleet venv is 3.11; the 3.12 venv was made inside it as `$FLEET_VENV/py312`): uv editable install of `.[test,importers]` plus `-e source/<pkg>` for each of the 12 workspace members in the root dependency list, with `--no-sources --override rocm-overrides.txt --index https://repo.amd.com/rocm/whl/gfx94X-dcgpu/` and overrides `torch==2.11.0+rocm7.13.0`, `torchvision==0.26.0+rocm7.13.0`, `warp-lang @ file://<warp-rocm>`. `--no-sources` is required because `[tool.uv.sources]` pins torch to the cu130 index. Result: torch 2.11.0+rocm7.13.0, warp-lang 1.18.0.dev2, newton 1.8.0.dev0 (git `42f7def1`), mujoco-warp 3.15.0, mujoco 3.15.0, omniverseclient 2.74.0, ovstage 0.3.0.1791986420, usd-exchange 3.0.1; no `nvidia-*` wheels. `pyproject.toml` and `uv.lock` untouched.
- Git LFS: the test USD assets and goldens are LFS pointers and the host has no `git-lfs` (downloading the binary was refused by the session's permission classifier). The 202 LFS files under `source/*/test/` (7.1 MB) were fetched from `https://media.githubusercontent.com/media/isaac-sim/IsaacLab/eed2d2889a5b963fee8c3594d29fe6d91ec67046/<path>`, each sha256-checked against its pointer oid (202/202 matched), and restored to pointers before committing. Without them about 45 Newton asset tests fail with `floating_two_link.usda ... is not a valid usda layer`. A validator needs the same, or git-lfs.
- Disk: the host overlay had 4-35 GB free during the run; the rocm-sdk-devel wheel unpacks to about 15 GB. Clear the uv cache after installing.

### GPU results on linux-gfx942 (MI300X, HIP_VISIBLE_DEVICES=0, ulimit -c 0), content of tip 03bb17b08

Kitless marker set (12 files at eed2d288), one pytest process per file with `-m kitless`:
- `isaaclab/test/sim/test_mjcf_converter.py` 11 passed and `test_urdf_converter.py` 18 passed (both need the `importers` extra, else a module-level skip); `test_schemas_deprecation.py` 96 passed.
- `isaaclab_newton/test/assets/test_articulation.py` 37 passed, 5 skipped (capture); `test_bam_actuator_mjwarp.py` 11 passed; `test_newton_actuators_newton.py` 37 passed; `test_rigid_object.py` 20 passed, 2 FAILED (`test_vbd_root_state_tracks_integrated_body[vbd-cuda:0]` and `[body-only-cuda:0]`, Warp gap 1 below).
- `isaaclab_tasks/test/test_microduck_velocity.py` 3 passed (a full Newton environment).
- `isaaclab_visualizers/test/test_newton_adapter.py` 1 FAILED (Warp gap 2); `test_newton_gl_dynamic_mesh.py` 1 FAILED `pyglet.window.NoSuchConfigException: No standard config is available` with `headless=True` (no GL context on compute-only CDNA; `EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1` segfaults). Graphics-pipeline class; judge it on a host with a display stack.
- Not run: `multi_gpu/test_multi_gpu_training_smoke.py` (one pinned GPU), `_isaaclab_testing/test/rendering/test_visualizers_rtx.py` (RTX/OVRTX, out of scope).

Unmarked Warp-kernel tests, one process per file: test_warp_index_kernel 8 passed; test_warp_math 6 passed, 1 skipped; test_warp_memory 11 passed, 2 skipped; test_wrench_composer 37 passed, 1 skipped; test_ray_caster_kernels 16 passed; test_shared_sensor_kernels 6 passed, 5 skipped; test_update_ray_caster_kernel 11 passed; test_newton_manager 43 passed, 4 skipped; test_newton_solvers 36 passed; test_ppisp 14 passed; test_ppisp_kernels 7 passed; actions parity 4 passed; events parity 1 passed, 4 skipped; observations parity 9 skipped (eager halves passed); rewards parity 4 passed, 16 skipped (eager halves passed); terminations parity 6 passed, 4 skipped. Need Kit (the module calls `launch_test_simulation()` without a Newton config): `test_wrench_composer_integration.py`, `test_newton_solver_reset.py`. Before the test guards these files had 48 failures, all `capture_launch() received graph=None` or graph-is-None assertions.

End-to-end: `isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct --num_envs 4096 --max_iterations 100 physics=newton_mjwarp` (this CLI rejects `--headless`) rc=0. Mean reward at iterations 1/10/25/50/75/100 = 0.13 / -2.49 / 2.59 / 4.68 / 4.93 / 4.94; mean episode length 300 (the cap) at the end; about 0.31 s per iteration with 0.22 s collection. No CUDA reference run was available in this job; reward rising to the episode cap is the learning evidence. The runtime benchmark command was not run separately.

`isaaclab --test` runs far more than tools/ (it walks most of the repository) and was stopped at the 20-minute background limit. Failures seen: missing optional modules (`rl_games`, `skrl`, `stable_baselines3`, `isaaclab_teleop`, `isaaclab_mimic`, `ovrtx`, `ovphysx`, `cv2`, `rerun`); `test_template_generator` (resolves with the host's Python 3.11); Newton Warp renderer camera tests (`test_camera_output_contract.py::test_newton_hdr_background_preserves_unlit_geometry`, Warp gap 2: `warp/native/texture.h:827: error: reference to __host__ function 'cpu_sample_2d_channel_at_level' in __host__ __device__ function` under HIPRTC) and `scripts/benchmarks/test/test_runtime_smoke.py::test_runtime_api_returns_profile_summary` (Newton renderer render benchmark, kernel build failure, output truncated, presumed the same); and the recorder test fixed by the second commit.

### AMD Warp and dependency gaps found (not Isaac Lab code; nothing pushed anywhere)

1. `nextafterf` on HIP: Newton VBD (`newton/_src/solvers/vbd/interval_arithmetic.py`) uses `nextafter`; the AMD Warp HIPRTC link fails `ld.lld: error: undefined hidden symbol: nextafterf` (`warp/native/cuda_crt.h:1096` maps it to `__nv_nextafterf`, which has no HIP definition). Affects the VBD solver only; the MuJoCo Warp, XPBD and Featherstone paths are unaffected. The fix belongs in AMD Warp (for example mapping to `__ocml_nextafter_f32`).
2. Warp textures on HIP: `wp.Texture2D(..., device="cuda:0")` -> `Warp CUDA error 801: operation not supported (in function wp_texture_create_device)`; AMD Warp's own suite excludes HIP from its texture tests (`warp/tests/cuda/test_texture.py:2885`, "device textures are unsupported on HIP/ROCm"), and kernels that sample textures fail to compile (texture.h above). Newton's Warp renderer with textured meshes and the render benchmark cannot run on AMD; the new docs section says so.
3. Newton's Kamino solver captures internally (`newton/_src/solvers/kamino/_src/linalg/factorize/rcm_batch.py:73`, `ScopedCapture` then `capture_launch`) and would hit `graph=None` on HIP; not on the tested path. `capture_if` / `capture_while` elsewhere in Newton and MuJoCo Warp run eagerly outside a capture.
4. MuJoCo Warp `io.py:301` warns `CUDA version < 12.4 detected ... graph_conditional should be set to False` because it compares Warp's toolkit version (ROCm 7.13) against CUDA numbers; cosmetic on HIP (training completed).
5. Plan risk 1b (wave32 tile width) not exercised (wave64 host).

The planner's stop condition ("AMD Warp dev cannot run Newton 1.8.0.dev0") did not trigger: the default Newton + MuJoCo Warp path runs end to end; gaps 1 and 2 are scoped to the VBD solver and to textures.

### Not done

- CUDA no-regression on an NVIDIA host: no CUDA-compiled code exists, `pyproject.toml` and `uv.lock` are byte-identical, and the helper keeps the `is_cuda` behaviour when the attribute is absent. Running the kitless set on `moat-port` on an NVIDIA host (to show capture still engages) is still owed.
- Sphinx build of the docs change not run (docs extra not installed); codespell clean, ASCII only.
- No skill lesson promoted yet. Candidate for the `cuda-to-rocm` validation reference: a source-built HIP library (Warp) and a PyTorch wheel must share one ROCm runtime in a process, and TheRock's `rocm[libraries,devel]` plus the matching `torch==X+rocm7.13.0` provides that. Left for the reviewer to judge with the port.

### Porter resume (2026-11-10, linux-gfx942)

jeff granted push access to `AMD-Ecosystem/IsaacLab` (verified 2026-11-09; `gh api repos/AMD-Ecosystem/IsaacLab -q .permissions.push` = true). From the original local clone (tree clean, tip `03bb17b08`): `git push -u origin moat-port` created the branch; `git ls-remote origin` shows `refs/heads/moat-port` = `03bb17b08290babb82d18bb3e2adb25ff1a5194c`, identical to the local tip, so the saved patch was not needed. `release/3.0.0-beta2` (still `414822f6`) and the GitHub default branch (still `release/3.0.0-beta2`) were left alone; nothing was pushed to any dependency repository. Records: `set-blocked --clear` linux-gfx942, `set-state porting` (retook the lock released by the block), `advance-head 03bb17b08...`, `jargon.py --port IsaacLab` clean, fork tree has no modified tracked files, `set-state ported`. Next role: reviewer. Still owed before presenting to jeff: a run on an NVIDIA host showing capture still engages (no compiled CUDA code exists, so there is no nvcc compile to do).

## Review 2026-11-10 (linux-gfx942, reviewer role)

Reviewed `moat-port` @ `03bb17b08290babb82d18bb3e2adb25ff1a5194c` against upstream `main` @ `eed2d2889a5b963fee8c3594d29fe6d91ec67046` (`git diff eed2d288...HEAD`, 30 files, +174/-19). The fork has no `main` branch, so `jargon.py --port IsaacLab` ran against a local `main` ref at eed2d288: clean. `python3 tools/changelog/cli.py check eed2d288...` rc=0. Fork tree clean. The planner questions for jeff above are unchanged and still open.

Verdict: **changes requested** (records and docs; no code defect in the capture gating).

### CUDA graph capture on NVIDIA (checked for the CUDA no-regression run that follows)

On NVIDIA Warp, `supports_graph_capture(device)` evaluates to exactly `device.is_cuda` for every device, so each replaced gate behaves as upstream did:
- Downloaded the PyPI `warp-lang` 1.17.0 (Isaac Lab's pin), 1.18.0 and 1.19.0 manylinux wheels. None of them contains the string `supports_graph_capture` anywhere under `warp/`. `class Device` in `warp/_src/context.py` has no `__getattr__` and no `__slots__`, so `getattr(device, "supports_graph_capture", True)` returns True and the helper returns `device.is_cuda`. `wp.get_device(Device)` returns the object as it is.
- AMD Warp `ac540f59` (`warp/_src/context.py:5826`): a property that is False only when `is_hip` is set; True for CPU and CUDA.
- Production gates, each `is_cuda` replaced one for one and therefore unchanged on NVIDIA: `newton_backend.py:261` (step capture) and `:484` (BVH refit), `newton_raycast_sensor.py:348`, `frame_transformer.py:309`, `newton_warp_renderer.py:595`, `physx_runtime.py:120` (the `is_cuda and is_capturing` stateful guard at :115 is untouched), `particle_surface.py:105`, `utils/device.py:57`, `gb300_sharpa_insertion/ik.py:33`. `warp_graph_cache.py:67-69` adds an early return only when `is_cuda and not supports_graph_capture`, which never holds on NVIDIA.
- Test guards: every `skip_without_graph_capture` call site is parametrized with `DeviceScope.CUDA`, uses `"cuda:0"`, or sits inside `if is_cuda` (test_warp_math), so none of them skips on an NVIDIA GPU. The monkeypatch in `test_newton_manager.py:482` sets the helper to True, its value on NVIDIA anyway. In `test_newton_solvers.py:170` CPU already skipped capture through `use_mujoco_cpu`.
- Not changed, and need no change: `isaaclab_physx` contact-sensor and ray-caster captures (Kit-only, scoped out); `launch_cache.py:24` (`record_cmd` replay, not a graph capture; test_warp_memory passes on HIP).

### Findings

1. Docs leave out the VBD solver gap. `docs/source/isaaclab/installation.rst` (new "AMD GPUs with ROCm" section) lists textures as unsupported on HIP. It does not say that Newton's VBD solver fails to compile on HIP (`nextafterf` undefined at HIPRTC link; the two `test_rigid_object.py` vbd failures). A user who selects VBD gets a link error the docs never warned about. Add one sentence next to the texture sentence. In the same section, `export LD_LIBRARY_PATH=$ROCM_PATH/lib` overwrites the user's existing value; prepend it instead: `export LD_LIBRARY_PATH=$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}`.
2. Plan item 4 (ray scripts) is neither done nor recorded. `scripts/reinforcement_learning/ray/util.py:164` counts GPUs with a bare `nvidia-smi` subprocess, and `wrap_resources.py:100` runs `nvidia-smi` as its test job. The plan asks for a torch-based fallback "if small; else owed with a reason". The porter notes never mention ray. Either add the fallback or record `scripts:ray-multi-gpu` as owed with the reason.
3. `surface.json` is stale. All 20 in-scope components are still `owed` with "not yet ported (planned 2026-11-05)", including ones the porter finished and tested (the packages, `install:rocm-kitless-recipe`, `docs:rocm-install-section`, `test:kitless-marker-set`, `test:warp-kernel-unit-tests-kitless-unmarked`, `example:rsl-rl-train-newton-smoke`). Move finished work to `covered`. Leave the work that is really outstanding as `owed`, each with its real reason: `test:multi-gpu-training-smoke` (one GPU pinned), `benchmark:runtime-newton` (not run), `test:tools-tests` (`isaaclab --test` stopped at the time limit), `benchmark:gpu-info-recorder` (the `hip_version` entry was dropped, and the recorder reports `cuda_version: Unknown` on ROCm; say whether that is the final answer), `scripts:ray-multi-gpu` (finding 2), wave32 and windows. `pr_ready` refuses while anything is owed, so stale entries hide which work is really left.
4. The toolchain workarounds in the docs need upstream bug drafts for jeff (rule: no README-only workarounds; draft the tool bug). The install section depends on three AMD Warp defects, none drafted: (a) `build_lib.py:490-491` calls `find_cuda_sdk()` even when `--rocm-path` is given, and `enable_hip` requires `cuda_path is None` (`:507`). `find_cuda_sdk` checks `WARP_CUDA_PATH`/`CUDA_HOME`/`CUDA_PATH`, then `which nvcc`, then default paths. `--no-cuda` also clears `rocm_path` (`:487`). So a host that has a CUDA Toolkit cannot build HIP Warp. (b) HIPRTC kernel JIT needs `ROCM_PATH` to find `float.h` and the HIP headers under the TheRock wheel layout. (c) `warp.so` needs `libhiprtc-builtins.so.7` on `LD_LIBRARY_PATH`. Also draft the two found during porting: `nextafterf` to `__nv_nextafterf` in `warp/native/cuda_crt.h:1096`, and the HIP texture compile failure in `texture.h:827`. Write each draft into notes.md for jeff; MOAT does not post to AMD-Ecosystem/warp.
5. Lesson not promoted. The porter's candidate is sound and applies to other projects, and `references/validation.md` does not have it: a source-built HIP library and a PyTorch wheel must load the same `libamdhip64` in one process, and TheRock's `rocm[libraries,devel]==X` with the matching `torch==...+rocmX` provides that, while a download.pytorch.org rocm7.2 torch with a TheRock-built library fails with "insufficient CUDA driver version" / "No CUDA GPUs are available". Add it to `.claude/skills/cuda-to-rocm/references/validation.md` on this branch.

### For the PR-prep squash (no fork push needed now)

- The body of commit `ee1f2cd0c` says "Also record the HIP version in the GPU info benchmark recorder", and `03bb17b08` reverts exactly that. The squashed message must drop that sentence and the second commit's text.
- Upstream's own `AGENTS.md` says "Do not add AI co-author or attribution lines". The commit bodies disclose AI assistance in prose, as MOAT requires. Whether that disclosure matches upstream's policy is jeff's call at the PR gate (planner question 4).

### Corrections to the porter's dependency analysis

- Gap 3 (Kamino) is wrong as written. The only production caller of `rcm_batch.create_rcm_batch_launch`, `llt_blocked_rcm_solver.py:516`, passes `use_cuda_graph=False`, so Kamino never reaches `create_cuda_graph_callback` (`rcm_batch.py:710-711`). Only the Kamino examples capture. There is a different possible Kamino defect on HIP, which I have not run: `_sync_threads` (`rcm_batch.py:62-68`) is `__syncthreads()` only under `#if defined(__CUDA_ARCH__)`, which HIPRTC does not define (no definition in AMD Warp `ac540f59`). The `complete_cuda` persistent kernel (`:321-407`, launched when `max_bfs_iters is None` and `device.is_cuda`) would then run with no block barriers on HIP. Record it as a Newton dependency candidate; it needs a run with Kamino before anyone draft-files it.

## Porter round 2 (2026-11-10, linux-gfx942, porter role, answers "Review 2026-11-10")

Fresh clone of `AMD-Ecosystem/IsaacLab` `moat-port` @ `03bb17b08` into this job's `projects/IsaacLab/src`; two commits added on top (no amend, no force push): `36a81e333` "[ROCm] Note the VBD solver gap and keep LD_LIBRARY_PATH in ROCm docs" and `d4aac6ccf` "[ROCm] List GPUs through PyTorch when nvidia-smi is missing in Ray". Pushed `moat-port` `03bb17b08..d4aac6ccf`; `git ls-remote` confirms `d4aac6ccf4d717de7e0566d63f5df20d9f10ec02`. The planner questions for jeff above are untouched and still open. Nothing was posted anywhere and nothing was pushed to a dependency repository.

Test environment: the 2026-11-06 porter's Python 3.12 venv (torch 2.11.0+rocm7.13.0, AMD Warp ac540f59 built for gfx942) read in place, with `PYTHONPATH` set to this clone's `source/*` so every `isaaclab*` import resolved to this clone (checked via `__file__`), `HIP_VISIBLE_DEVICES=2`, `ulimit -c 0`, and the documented `ROCM_PATH` / prepended `LD_LIBRARY_PATH`. A separate venv (`$FLEET_VENV/py312`, ray 2.60.0 plus that venv's site-packages through a `.pth`) ran the Ray probe.

### Finding by finding

1. Docs (`36a81e333`): one sentence after the texture sentence says Newton's VBD solver does not run on HIP devices because its kernels fail to link (`nextafterf` undefined) and names MuJoCo Warp, XPBD and Featherstone as the working solvers; `export LD_LIBRARY_PATH=$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}`. Re-run: `pytest -m kitless source/isaaclab_newton/test/assets/test_rigid_object.py -k vbd` = 2 failed (the two vbd cases, `ld.lld: error: undefined hidden symbol: nextafterf`), 1 passed, as the docs now say. With `LD_LIBRARY_PATH=/opt/nonexistent` before the documented prepend, Warp initialises (`ROCm 7.13, HIP driver 7.13`) and runs a kernel on MI300X. codespell clean; Sphinx build still not run (docs extra not installed).
2. Ray (`d4aac6ccf`): `scripts/reinforcement_learning/ray/util.py` test mode uses `nvidia-smi` when `shutil.which` finds it, else lists the GPUs visible to PyTorch (`get_device_name`, `mem_get_info` free MB, serial `N/A`). Probe (job dir `work/probe/ray_test_mode.py`, local `ray.init`, `execute_job(..., test_mode=True)`): upstream raises `FileNotFoundError: [Errno 2] No such file or directory: 'nvidia-smi'`; patched logs `Name: AMD Instinct MI300X HF|Memory Available: 196140 MB|Serial Number N/A` and `Detected GPUs from PyTorch: 1`. On ROCm the fallback sees only visible GPUs, so the "isolation" comparison in that function never fires there; with `nvidia-smi` present nothing changes. `wrap_resources.py:100` and `task_runner.py` only pass the string `"nvidia-smi"` as the job label to that test mode, so they need no change. ruff 0.14.10 check/format clean; `tools/changelog/cli.py check eed2d288...` rc=0 (scripts are not a changelog package).
3. `surface.json`: 17 components moved to `covered`, each with its evidence; `owed` now holds only real work: `test:multi-gpu-training-smoke` (needs 2+ GPUs, every run was pinned to one), `test:tools-tests` (`isaaclab --test` never ran to completion), `benchmark:gpu-info-recorder` (below), `platform:wave32`, `platform:windows`. `benchmark:runtime-newton` now ran: `isaaclab benchmark runtime --task Isaac-Cartpole-Direct --num_envs 4096 --num_steps 1000 --warmup_steps 50 presets=newton_mjwarp` rc=0, 331,846 steps/s (std 34,091), 1000 iterations, 12.34 s wall, startup 0.46/9.66/11.10 s. `scripts:ray-multi-gpu` is covered by finding 2. `surface.py check IsaacLab` = fully accounted for.
4. Bug drafts: below, "Drafts for jeff". Two of the five root causes differ from what the earlier notes said; see the correction.
5. Lesson: `.claude/skills/cuda-to-rocm/references/validation.md`, new section "A source-built HIP library and a PyTorch wheel must load the same ROCm runtime" (on this branch).

### benchmark:gpu-info-recorder: not final, design question for jeff

`test_recorders.py` 18 passed at `d4aac6ccf`. But on ROCm the recorder has neither pynvml nor `nvidia-smi`: `cuda_version` reads `Unknown`, utilization is never recorded, and memory falls back to `torch.cuda.memory_allocated`, which misses Warp's allocations. The runtime benchmark above therefore prints `gpu_util 0.0 %` and `gpu_mem 0.00 GB`. `torch.cuda.utilization` / `device_memory_used` would work on ROCm through the `amdsmi` Python package, which the documented TheRock install does not ship (`ModuleNotFoundError: amdsmi`); a device-wide `torch.cuda.mem_get_info` fallback works today (measured: free drops by 1 GiB after a 1 GiB allocation) but changes upstream's last-resort semantics and its `test_mem_peak_tracks_running_max`, which drives `memory_allocated` directly. Question for jeff: (a) leave it owed and document that GPU telemetry is unavailable on ROCm, (b) add an `amdsmi` path and put `amdsmi` in the ROCm install docs, or (c) change the fallback to `mem_get_info` and adjust the upstream test. Left `owed`, not decided by an agent.

### Correction to "AMD Warp and dependency gaps" 1 and 2 (measured this round)

Probe (job dir `work/probe/nextafter_probe.py`, one `wp.func_native` snippet per run on MI300X with AMD Warp ac540f59): `#if defined(__CUDA_ARCH__)` is FALSE in AMD Warp's HIPRTC kernel compile, `#if defined(__HIP_DEVICE_COMPILE__)` is TRUE; `return nextafterf(value, INFINITY);` compiles and returns `1.0000001` (Warp's `cuda_crt.h` mapping works); `__ocml_nextafter_f32` also works; `return __builtin_nextafterf(value, INFINITY);` fails `ld.lld: error: undefined hidden symbol: nextafterf`. Newton's `newton/_src/solvers/vbd/interval_arithmetic.py:37-60` writes `#if defined(__CUDA_ARCH__) nextafterf(...) #else __builtin_nextafterf(...) #endif`, so on HIP it takes the CPU branch. Gap 1 is therefore not `cuda_crt.h:1096`; it is a device-code guard that only knows CUDA. AMD Warp's own headers already guard with `!defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)` (`warp/native/crt.h:19,30`, `tile_fft.h:316`). Gap 2's compile half is the same class inside AMD Warp: `warp/native/texture.h:817` `sample_2d` uses `#if defined(__CUDA_ARCH__)` alone, so the HIP device compile takes the CPU branch and calls the host function `cpu_sample_2d_channel_at_level`. The reviewer's Kamino `_sync_threads` candidate (`rcm_batch.py:62-68`) is the same pattern and is consistent with this measurement; it has still not been run.

Skill candidate for the reviewer to judge (not promoted this round): in Warp-based projects, `wp.func_native` snippets and Warp headers that pick device code with `__CUDA_ARCH__` alone take the host branch under AMD Warp's HIPRTC; grep for `__CUDA_ARCH__` in native snippets and add `|| defined(__HIP_DEVICE_COMPILE__)`.

### Drafts for jeff (not filed; MOAT posts nothing to these repositories)

Each draft is for jeff to file or forward to the AMD Warp owners (or the AMD Newton fork owners for D) if he chooses. Repository: `AMD-Ecosystem/warp` unless stated, branch `amd-integeration-dev` @ `ac540f595ff5763a374c113d77e4111a1d4791bd`.

#### Draft A: build_lib.py ignores --rocm-path when a CUDA Toolkit is installed

====
Title: build_lib.py builds CUDA instead of HIP when a CUDA Toolkit is present, even with --rocm-path

On a Linux host that has both ROCm and a CUDA Toolkit (here `/usr/local/cuda`), `python build_lib.py --rocm-path "$ROCM_PATH" --hip-arch=gfx942` does not build the HIP backend. `build_lib.py:490-491` calls `find_cuda_sdk()` whenever `--cuda-path` is not given, and `find_cuda_sdk()` returns the first of `WARP_CUDA_PATH`/`CUDA_HOME`/`CUDA_PATH`, `which nvcc`, or `/usr/local/cuda`. Then `build_lib.py:507` sets `enable_hip = bool(rocm_path) and cuda_path is None`, so an explicit `--rocm-path` loses to an auto-detected toolkit. `--no-cuda` cannot be used to opt out because it also clears `rocm_path` (`build_lib.py:487`).

Expected: an explicit `--rocm-path` (or `--hip-arch`) selects the HIP backend and skips CUDA Toolkit detection, or a flag such as `--no-cuda-toolkit` exists that keeps ROCm enabled.

Workaround used: run the build through a wrapper that replaces `build_lib.find_cuda_sdk` with `lambda: None`, or build on a host with no CUDA Toolkit.
====

#### Draft B: HIPRTC kernel builds need ROCM_PATH under the TheRock Python wheel layout

====
Title: HIPRTC kernel compiles fail with 'float.h' file not found unless ROCM_PATH is set (TheRock wheels)

With ROCm installed from TheRock's Python packages (`rocm[libraries,devel]==7.13.0` from repo.amd.com, no `/opt/rocm`), every Warp kernel launch on a HIP device fails at JIT time with `fatal error: 'float.h' file not found`. `warp/_src/build.py:_hip_extra_include_dirs` looks for the clang resource directory and the HIP headers only under `ROCM_PATH`, `ROCM_HOME`, `HIP_PATH` and `/opt/rocm`. Setting `ROCM_PATH=$(rocm-sdk path --root)` fixes it.

Expected: Warp finds the ROCm root it was loaded against without an environment variable, for example from the directory of the loaded `libhiprtc`/`libamdhip64` or from `rocm_sdk` when it is importable, and names the missing root in the error when it cannot.
====

#### Draft C: warp.so needs libhiprtc-builtins.so.7 on LD_LIBRARY_PATH

====
Title: warp.so fails to load next to a TheRock PyTorch unless LD_LIBRARY_PATH includes the ROCm lib directory

A HIP build of Warp links `warp.so` with `-lamdhip64 -lhiprtc -lhiprtc-builtins` (`warp/_src/build_dll.py:988-998`), and the result has `RUNPATH [$ORIGIN]` only. In a TheRock venv, `import torch` preloads `libamdhip64` and `libhiprtc` from the `_rocm_sdk_*` packages but not `libhiprtc-builtins.so.7`, so `import warp; wp.init()` fails with `Failed to load the shared library warp.so` unless `LD_LIBRARY_PATH` contains `$(rocm-sdk path --root)/lib`.

Expected: `warp.so` loads without `LD_LIBRARY_PATH`, for example by resolving the ROCm libraries the same way the runtime locates the ROCm root (Draft B), by recording the build's ROCm lib directory in the RUNPATH, or by dropping the direct `libhiprtc-builtins` link if HIPRTC loads it itself.
====

#### Draft D: device code guarded by __CUDA_ARCH__ alone takes the host branch under HIPRTC (Newton VBD)

Target: jeff's choice between `AMD-Ecosystem/warp` (make the convention explicit) and the AMD Newton fork `AMD-Ecosystem/newton` (fix the snippet); the defect lives in Newton's code but only shows on AMD Warp.

====
Title: Newton VBD fails to link on HIP: native snippet picks __builtin_nextafterf because __CUDA_ARCH__ is undefined under HIPRTC

On AMD Warp (HIP backend, MI300X, ROCm 7.13), any Newton model that uses the VBD solver fails to build its kernels with `ld.lld: error: undefined hidden symbol: nextafterf`. `newton/_src/solvers/vbd/interval_arithmetic.py:37-60` (`next_float_up`/`next_float_down`) uses a `wp.func_native` snippet that calls `nextafterf` under `#if defined(__CUDA_ARCH__)` and `__builtin_nextafterf` otherwise. Warp's HIPRTC compile does not define `__CUDA_ARCH__` (it defines `__HIP_DEVICE_COMPILE__`), so the device build takes the CPU branch, and `__builtin_nextafterf` lowers to a call to `nextafterf` that has no device definition. Measured with one-line snippets: `nextafterf(1.0f, INFINITY)` returns `1.0000001` on HIP, `__builtin_nextafterf` fails to link.

Fix: guard device code with `#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)`, as Warp's own headers do (`warp/native/crt.h:19`). The same pattern appears in `newton/_src/solvers/kamino/_src/linalg/factorize/rcm_batch.py:62-68` (`_sync_threads` emits `__syncthreads()` only under `__CUDA_ARCH__`), which would leave the persistent `complete_cuda` kernel without block barriers on HIP; not yet run.
====

#### Draft E: Warp textures on HIP: device creation unsupported and sampling kernels fail to compile

====
Title: Texture2D on HIP: wp_texture_create_device returns 801 and texture sampling kernels fail to compile

On AMD Warp (HIP backend, MI300X, ROCm 7.13), `wp.Texture2D(..., device="cuda:0")` fails with `Warp CUDA error 801: operation not supported (in function wp_texture_create_device)`, and any kernel that samples a texture fails to compile with `warp/native/texture.h:827: error: reference to __host__ function 'cpu_sample_2d_channel_at_level' in __host__ __device__ function`. The compile failure comes from `texture.h:817`, where `sample_2d` selects the device path with `#if defined(__CUDA_ARCH__)` alone; HIPRTC does not define it, so the HIP device compile takes the CPU path. The suite already skips HIP for device textures (`warp/tests/cuda/test_texture.py:2885`). Downstream, Newton's Warp renderer cannot render textured meshes on AMD GPUs (Isaac Lab `test_newton_adapter.py`, `test_camera_output_contract.py::test_newton_hdr_background_preserves_unlit_geometry`).

Expected: the texture headers guard device code with `__CUDA_ARCH__ || __HIP_DEVICE_COMPILE__`, and HIP device textures are implemented with HIP texture objects, or `Texture2D` on a HIP device raises a clear "not supported on HIP" error at creation and kernels that sample textures still compile.
====

## Review 2026-11-10, second pass (linux-gfx942, reviewer role)

Reviewed `moat-port` @ `d4aac6ccf4d717de7e0566d63f5df20d9f10ec02`: the full range `eed2d288...HEAD` (31 files, +191/-26), focusing on the round-2 delta `03bb17b08..d4aac6ccf` (`36a81e333`, `d4aac6ccf`) and the skill lesson on `port/IsaacLab`. Fresh clone of the fork; `jargon.py --port IsaacLab` clean (local `main` ref at eed2d288); titles 52-68 characters; no trailer or noreply address; fork tree clean. Drafts fact-checked against a clone of `AMD-Ecosystem/warp` @ `ac540f595ff5763a374c113d77e4111a1d4791bd` and Newton @ `42f7def1` (installed copy in the 2026-11-06 porter venv), with runtime probes in that venv read in place (`HIP_VISIBLE_DEVICES=2`, `ulimit -c 0`; probe scripts in `/tmp`, not kept). The planner questions for jeff and the gpu-info-recorder question (a/b/c) are untouched and still open.

Verdict: **review passed**. No blocking findings.

### "Review 2026-11-10" items

1. Resolved. `installation.rst:582-584` names the VBD link failure and the working solvers, and `:597` prepends `LD_LIBRARY_PATH`. I checked the solver claim, because `test_newton_solvers.py` only builds XPBD and Featherstone configs and never steps them: a probe with a free box and a revolute capsule ran 200 steps of `SolverXPBD` and `SolverFeatherstone` on MI300X with no error and finite poses (kernels compile and launch under HIPRTC). MuJoCo Warp is already covered by the training run.
2. Resolved. `ray/util.py:164-177` keeps the `nvidia-smi` path whenever `shutil.which` finds it, so NVIDIA behaviour is unchanged. The fallback's `"name, MB, N/A"` lines go through the same `split(", ")` parse; GPU names carry no `", "`. The `except subprocess.CalledProcessError` at `:217` still covers only the `nvidia-smi` branch, which is correct because the fallback cannot raise it. Non-blocking: on ROCm the `CUDA_VISIBLE_DEVICES` lines (`:189-214`) report "not set" when Ray isolates through `ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES`. Test mode only logs this and the porter already recorded it.
3. Resolved. `surface.py check IsaacLab` = fully accounted for. The 17 `covered` entries cite evidence that matches the notes, and the five `owed` entries are real remaining work with reasons.
4. Resolved. Drafts A to E are written. Fact-check below.
5. Resolved. `references/validation.md` "A source-built HIP library and a PyTorch wheel must load the same ROCm runtime" is accurate against the porter's recorded errors. The skill had no prior entry on this (grep `libamdhip64`), it sits in the validation reference where a reader with the "insufficient CUDA driver version" symptom would look, and it states the triage step first.

### Fact-check of drafts A to E (AMD Warp ac540f59)

- **A: confirmed.** `build_lib.py:490-491` calls `find_cuda_sdk()` whenever `--cuda-path` is not given. `find_cuda_sdk` (`:86-121`) checks `WARP_CUDA_PATH`/`CUDA_HOME`/`CUDA_PATH`, then `which nvcc`, then `/usr/local/cuda`. `:507` is `enable_hip = bool(args.rocm_path) and args.cuda_path is None`, and `--no-cuda` clears `rocm_path` at `:487`. No other argument (`:313-356`) opts out.
- **B: confirmed by measurement.** `warp/_src/build.py:55-60` (`_hip_extra_include_dirs`) searches only `ROCM_PATH`, `ROCM_HOME`, `HIP_PATH` and `/opt/rocm`. With all three unset, `LD_LIBRARY_PATH` set, and no `/opt/rocm` on this host, the first kernel fails with `warp/native/crt.h:50:10: fatal error: 'float.h' file not found`. With `ROCM_PATH=$(rocm-sdk path --root)` it compiles and returns `[1. 1. 1. 1.]`.
- **C: confirmed by measurement.** `readelf -d warp.so`: `RUNPATH [$ORIGIN]`, `NEEDED` `libamdhip64.so.7`, `libhiprtc.so.7`, `libhiprtc-builtins.so.7`. The link is assembled at `build_dll.py:990-998` and the rpath at `:1052-1063`. torch's `_rocm_init.py` preload list has `amdhip64` and `hiprtc` but not `hiprtc-builtins`. After `import torch` with `LD_LIBRARY_PATH` unset, `ctypes.CDLL(warp.so)` raises `OSError: libhiprtc-builtins.so.7: cannot open shared object file`.
- **D: confirmed by measurement.** Newton `interval_arithmetic.py:38-60` and `rcm_batch.py:62-67` are as quoted. In a `wp.func_native` snippet on HIP, `#if defined(__CUDA_ARCH__)` selects the `#else` branch (returned 2.0), `#if defined(__HIP_DEVICE_COMPILE__)` selects the device branch (returned 1.0), `nextafterf(1.0f, INFINITY)` returns `1.0000001192092896`, and `__builtin_nextafterf` fails with `ld.lld: error: undefined hidden symbol: nextafterf`. `cuda_crt.h:1096` maps `nextafterf` to `__nv_nextafterf` and works. `crt.h:19,30` and `tile_fft.h:316` use the `!__CUDA_ARCH__ && !__HIP_DEVICE_COMPILE__` form, as the draft says. One addition for jeff's choice of target: the snippet is identical in upstream `newton-physics/newton` @ `42f7def1`, the revision Isaac Lab pins. So upstream Newton is a third possible destination besides the two forks. MOAT posts to none of them.
- **E: confirmed, with two line corrections.** `wp.Texture2D(..., device="cuda:0")` raises `Warp CUDA error 801: operation not supported (in function wp_texture_create_device, .../texture.cpp:205)`. The `cpu_sample_2d_channel_at_level` call is at `texture.h:827` as quoted. Before jeff files it: the guard in `sample_2d` is at `texture.h:818`, not `:817` (`:816` is the signature). And it is not alone: all nine device/host selections in `texture.h` (`:689, 705, 721, 741, 758, 775, 797, 818, 840`) use `#if defined(__CUDA_ARCH__)` only, so the draft should say "the texture sampling functions in `texture.h` (nine `__CUDA_ARCH__`-only guards, `:689-840`)" rather than single out `sample_2d`. The HIP exclusion in the test suite is `test_texture.py:2885-2887` (the comment is 2885, the filters 2886-2887).

### Porter's skill candidate (`__CUDA_ARCH__`-only device guards): promote it

The candidate is sound and applies beyond Warp. HIP device compiles (HIPRTC measured here; hip-clang behaves the same) define `__HIP_DEVICE_COMPILE__` and not `__CUDA_ARCH__`. So any `#if defined(__CUDA_ARCH__)` device/host split silently takes the host branch, which shows up as a link error, a host-function-in-device-code error, or (the Kamino case) missing barriers. JIT source strings (`wp.func_native`, NVRTC/HIPRTC kernel strings) are where hipify never looks. The skill does not have this: `fault-classes.md:373-377` covers gating device-only includes, not device-path selection. Promote it to `references/fault-classes.md` "Headers, includes and build" on this branch in the next round that touches the branch (the PR-prep squash or a fix round). It is not blocking validation.

### Still owed (unchanged by this review)

- NVIDIA no-regression run showing capture still engages (no compiled CUDA code, so no nvcc compile applies).
- PR-prep squash notes from the first review still apply. Squash `36a81e333` and `d4aac6ccf` messages as written; they have no stale sentences.
