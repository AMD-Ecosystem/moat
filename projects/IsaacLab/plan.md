# IsaacLab porting plan

Planner run 2026-11-05 on windows-gfx1151 (planning only; no GPU work). Upstream read at `isaac-sim/IsaacLab` `main` @ `eed2d2889a5b963fee8c3594d29fe6d91ec67046` (VERSION 3.0.0), the same sha intake screened. jeff was away during planning; open questions are in `notes.md` and the conservative assumption taken for each is stated here.

## Project

- Name: IsaacLab
- Upstream: https://github.com/isaac-sim/IsaacLab, default branch `main`
- Fork: https://github.com/AMD-Ecosystem/IsaacLab (pre-existing, taken over by jeff's 2026-11-05 ruling). It carries only `release/3.0.0-beta2`; it has no `main` mirror and no `moat-port` yet.

## Existing AMD support

Classification: **improvable** -- no AMD support in Isaac Lab itself, but authoritative AMD work-in-progress exists in the GPU dependency underneath it.

- Isaac Lab: nothing (intake: no ROCm PR/issue/branch, no AMD mention in docs).
- Warp (`AMD-Ecosystem/warp`, AMD-official): this is the real GPU backend and the only native code in the stack. Branches checked 2026-11-05:
  - `amd-integration` / `release/1.13.0.amd0`: `1.13.0+rocm.0`, README support statement gfx942 (MI325X), ROCm 7.1.1/7.2.1, Linux.
  - `amd-integeration-dev` (sic) @ `ac540f595ff5763a374c113d77e4111a1d4791bd` (2026-09-02): `1.18.0.dev2`, 23 commits ahead of / 37 behind its `main` mirror. This is new enough for everything Isaac Lab `main` needs (`warp-lang==1.17.0`, Newton needs `>=1.17.0`, MuJoCo Warp 3.15 needs `>=1.15`).
  - `amd-integration-halo` @ `74014999a31471228e0aaa213bbeb2e5daba95f0` (2026-08-21): `1.17.0.dev4`, carries "feat(amd): add the gfx1151 HIP/ROCm port", including an arch-family wavefront width (`__GFX9__` -> 64, else 32) in `warp/native/tile_reduce.h`.
- Newton (`AMD-Ecosystem/newton`, `rocm-enablement`): ONE commit, `pyproject.toml` only (+8/-1): adds `cuda`/`rocm` extras so `warp-lang` is not pulled from the CUDA wheel. Newton is pure Python on top of Warp; its "ROCm port" is packaging. Its base is `1.7.0.dev0` (MuJoCo Warp 3.12).
- MuJoCo Warp (`AMD-Ecosystem/mujoco_warp`): `amd-integration` is `3.8.1`; branches `amd/hip-graph-support` (3.9.0.1, a `hip_graph_capture()` helper), perf branches. Also pure Python on Warp.

Decision: **consume, do not inherit or fork.** The AMD-Warp dev branch is authoritative and is used as the GPU backend. Newton and MuJoCo Warp are used at exactly the upstream versions Isaac Lab pins, unmodified: neither has native code, and the only thing the AMD Newton fork changed is how `warp-lang` gets installed, which Isaac Lab's ROCm install recipe handles itself. MOAT does not push to any of the three dependency repositories (none is a MOAT project).

Delivery vehicle: upstream does not link platform forks; a normal upstream PR is the plausible vehicle, but upstream is NVIDIA-owned, so whether one is opened is decided at the PR gate (question 4 in notes.md). The plan keeps the diff upstream-shaped regardless: no change to the CUDA default install, no uv.lock regeneration, platform behaviour keyed on a Warp device capability rather than on a vendor check. Port-vs-rewrite does not arise: there are no NVIDIA-tuned kernels in this repository; the Warp kernels are Python and Warp compiles them for HIP itself.

## Build classification

Neither cmake nor torch-extension: **pure Python (uv workspace) on Warp JIT kernels**. `ext_type` recorded as `python-warp`.

Evidence:
- `surface.py generate`: 0 CUDA files, 0 build-system components (no `.cu`/`.cuh`/C++, no `CMakeLists.txt`, no `CUDAExtension`).
- `pyproject.toml:6-9`: setuptools build backend, `packages = []` dependency-only root; workspace members are the 15 `source/*` packages (`pyproject.toml:453-466`).
- GPU work: `wp.kernel` Python kernels compiled at runtime by Warp (HIPRTC on ROCm), plus torch, plus Newton/MuJoCo Warp (also Warp kernels).
- CUDA coupling is in packaging (`pyproject.toml:395-397,467-468`: torch/torchvision forced to the `pytorch-cu130` index; `warp-lang==1.17.0` at line 47 is the CUDA-only PyPI wheel) and in Python capability checks (below).

## Port strategy

Neither A nor B. This is the "anything else" class: a Python framework whose port is (1) an install path that swaps in ROCm torch and AMD Warp and (2) Python-level fixes where Isaac Lab equates "is a CUDA device" with "supports CUDA graph capture" and NVIDIA tooling. No hipify, no compat header.

### Base: upstream `main`, not `release/3.0.0-beta2`

- `release/3.0.0-beta2` (the fork's only branch, upstream tip `414822f6`, 2026-09-10) pins `warp-lang==1.13.0`, `newton[sim]==1.2.1`, torch 2.10 / cu128. That lines up with AMD Warp's release `1.13.0.amd0`, which is the one tempting reason to choose it. Rejected: it is a frozen pre-release line 1771 commits behind `main` (155 commits of its own not on `main`), no upstream PR could target it, and every graph-capture call site this plan fixes would have to be redone on `main`.
- `main` @ `eed2d288` is chosen. Its dependency set is runnable on ROCm today with no pinning backwards: AMD Warp `amd-integeration-dev` (1.18.0.dev2) satisfies every Warp floor, and the Python-only Newton @ `42f7def1` and MuJoCo Warp 3.15 come from upstream unchanged.
- Fallback if AMD Warp dev cannot run Newton 1.8.0.dev0 on wave64 (a Warp API or runtime gap the porter cannot work around in Isaac Lab): STOP and record the gap; do not silently move the port to `release/3.0.0-beta2`. Re-basing to beta2 is a scope change for jeff.

### Dependency set the porter installs (wave64)

| package | source | why |
|---|---|---|
| torch 2.11.0, torchvision 0.26.0 | `https://download.pytorch.org/whl/rocm7.2` (verified `torch-2.11.0+rocm7.2-cp312` manylinux x86_64 exists; `rocm7.1` also) | Isaac Lab pins exact 2.11.0/0.26.0; the `+rocm7.2` local version satisfies `==2.11.0` |
| warp-lang | `AMD-Ecosystem/warp` @ `ac540f595ff5763a374c113d77e4111a1d4791bd` (`amd-integeration-dev`), built from source: `python build_lib.py --hip-arch=<gfx>` then `pip install -e .` (or a built wheel) | the only AMD Warp line with version >= 1.17 |
| newton[sim] | upstream `newton-physics/newton` @ `42f7def118c2c4299ce493457bafd4e8a22a696e` (as pinned), installed so it does not pull the CUDA `warp-lang` wheel | pure Python |
| mujoco-warp | PyPI `~=3.15.0` (as pinned), same no-CUDA-wheel caveat | pure Python |

Version mismatch to resolve in the recipe: AMD Warp reports `1.18.0.dev2`, Isaac Lab requires `warp-lang==1.17.0`. Use a uv override (`uv pip install --override`, the same mechanism `tools/wheel_builder/uv-overrides.txt` already uses) rather than editing the pin, so the CUDA path keeps its exact pin.

### Wave32 and Windows

- AMD Warp dev hardcodes `WP_TILE_WARP_SIZE 64` for every `__HIP_PLATFORM_AMD__` target (`warp/native/tile_reduce.h:14-18`). On RDNA (wave32) any Warp tile reduction is miscompiled. The halo branch has the arch-family fix but is `1.17.0.dev4` (PEP 440 below Newton's `>=1.17.0` floor) and older. Isaac Lab itself uses `wp.tile_*` only in `isaaclab/utils/warp/memory.py`; Newton/MuJoCo Warp usage of tiles is unknown. wave32 is therefore an AMD-Warp question, not an Isaac Lab one (question 1b).
- Windows: AMD Warp dev's HIP build path is Linux-only (`-fPIC`, `libhiprtc.so` probes in `warp/_src/build_dll.py:960-995`). There is no Windows Warp-on-HIP build to run Isaac Lab against. The `windows` gate is a waiver candidate (OS obstacle: a dependency with no Windows build). An agent may only suggest it; question 3.

### Validator platform order

1. **linux-gfx942** (MI300) first: the one arch AMD Warp states support for. Build AMD Warp with `--hip-arch=gfx942`.
2. Further wave64 evidence, additive: linux-gfx950 (MI355), linux-gfx90a (MI200). AMD Warp builds for these via `--hip-arch`; unproven by AMD.
3. **wave32**: linux-gfx1100 or linux-gfx1201, only after wave64 passes. First with AMD Warp dev (if no tile kernel is reached on the tested path the hardcoded 64 is harmless; the validator must check, not assume), else with the halo branch under the same override. If both fail on a Warp-native defect, `set-blocked` the platform with the Warp defect named.
4. **windows-gfx1151** last, per MOAT rule, and only if a Windows AMD Warp build exists by then; otherwise leave it for jeff's waiver decision. Never run heavy Newton training on the laptop.

## CUDA surface inventory

No native CUDA. The surface is Python:

| item | count / where | ROCm mapping |
|---|---|---|
| `torch.cuda.*`, `"cuda:N"` strings | ~50 `is_available`, ~300 strings | unchanged on ROCm torch (HIP masquerades as `cuda`) |
| Warp device strings `"cuda:N"`, `wp.is_cuda_available`, `wp.get_cuda_device_count` | `utils/device.py`, teleop `system_check.py:544` | unchanged: AMD Warp keeps the `cuda` alias, `Device.is_cuda` is True on HIP (`context.py:5795-5802`, `is_hip` = `is_cuda and arch_str.startswith("gfx")`) |
| Warp graph capture (`wp.ScopedCapture`, `wp.capture_launch`, `use_cuda_graph` cfg) | 13 production files, ~25 test files | **the main seam.** AMD Warp: `Device.supports_graph_capture` is False on HIP; `capture_begin` returns False, `ScopedCapture` leaves `capture.graph = None` and RUNS THE BODY EAGERLY; `capture_launch(None)` raises (`context.py:5826-5838, 13045-13051, 13829-13838`). Isaac Lab gates capture on `device.is_cuda`, which is True on HIP. |
| `wp.stream_from_torch(torch.cuda.Stream)` relaxed capture | `newton_backend.py:647` | only reached when capturing; becomes dead on HIP once capture is gated |
| conditional graph nodes | none in Isaac Lab | AMD Warp: unsupported on HIP |
| `nvidia-smi`, `pynvml`, `torch.version.cuda` | `benchmark/recorders/record_gpu_info.py`, `cli/multigpu.py`, ray scripts, teleop | replace with a torch-device-properties fallback (`torch.version.hip`, `torch.cuda.get_device_properties`) where on the Kit-less path |
| `cupy` | contrib tasks, tutorials (`08_mpm`) | cupy has ROCm builds; only touch if a Kit-less test reaches it |
| `CUDA_VISIBLE_DEVICES` | multi-GPU CLI, ray | honoured by the HIP runtime; no change unless a test proves otherwise |
| `wp.tile_*` | `isaaclab/utils/warp/memory.py` | wave32 risk via AMD Warp (above) |

## Risk list

1. **Double-stepping under no-op capture (correctness, silent).** `newton_backend.py:449-458`: when `self.graph is None` it runs `simulate()` eagerly and then calls `self.capture(simulate)`, which on HIP runs `simulate()` AGAIN inside the no-op capture and returns `None`, so every physics step advances twice and recaptures forever. The same "first call eager, then capture" shape is in the ray-caster, frame transformer, renderer, actuator runtime, particle surface and `warp_graph_cache`. Symptom: wrong physics, no error. Fix by gating, never by catching the `capture_launch` exception.
2. Newton/MuJoCo Warp internal graph capture: if those libraries capture internally on `is_cuda`, the same hazard lives in code MOAT does not own. The porter checks `grep -rn ScopedCapture` in the installed Newton/MuJoCo Warp; a hit that misbehaves is recorded as a dependency defect (notes + question), not patched.
3. AMD Warp wave64-only tile width (wave32) and Linux-only HIP build (Windows), above.
4. AMD Warp dev is an unreleased branch with a misspelled name; it can move or vanish. Pin by sha in the recipe and in notes.
5. Warp kernel JIT on HIP: first run compiles every kernel through HIPRTC; budget minutes, and keep the Warp kernel cache per host. A HIPRTC compile error in an Isaac Lab kernel is a real finding (Isaac Lab kernel code that only nvrtc accepts).
6. Fault classes that still apply through Warp: fresh device allocations are not zeroed on ROCm (Isaac Lab code relying on `wp.empty` contents), OOB neighbour reads that CUDA tolerates. Warp's own bounds checks (`wp.config.mode = "debug"`) are the diagnostic.
7. FP divergence in iterative solvers (MuJoCo Warp, Newton) across arches: compare against tolerances the tests already set; do not loosen them without a measured, ROCm-gated reason.
8. Binary NVIDIA wheels in base deps: `omniverseclient==2.74.0`, `usd-exchange`, `isaaclab-ov` workspace member. They are CPU-side, but the porter must confirm a ROCm environment installs and imports them (no CUDA runtime dependency). `ovphysx`/`ovrtx`/`ovstage` are extras and are not installed.
9. Multi-GPU smoke needs 2 GPUs on the validation host.
10. Upstream is NVIDIA-owned; reception of an AMD PR is uncertain (delivery question, not a technical risk).

## File-by-file change list

All changes on the fork's `moat-port`, cut from upstream `main` (the fork needs a `main` mirror first; question 2).

1. `source/isaaclab/isaaclab/utils/warp/` (existing module) -- add one helper, e.g. `supports_graph_capture(device) -> bool` = `device.is_cuda and getattr(device, "supports_graph_capture", True)`. `getattr` default keeps NVIDIA Warp builds that predate the attribute on the current behaviour. Reuse an existing helper if `isaaclab.utils.warp` already has a capture utility.
2. Replace the `is_cuda` capture gate with the helper at every production site:
   - `source/isaaclab_newton/isaaclab_newton/physics/newton_backend.py:260,483` (step capture and BVH refit)
   - `source/isaaclab_newton/isaaclab_newton/sensors/ray_caster/newton_raycast_sensor.py:347`
   - `source/isaaclab_newton/isaaclab_newton/sensors/frame_transformer/frame_transformer.py:308`
   - `source/isaaclab_newton/isaaclab_newton/renderers/newton_warp_renderer.py:594`
   - `source/isaaclab_newton/isaaclab_newton/physics/mpm_solver.py` (capture site)
   - `source/isaaclab/isaaclab/actuators/newton/physx_runtime.py:113,118,171`
   - `source/isaaclab_visualizers/isaaclab_visualizers/newton/particle_surface.py:103`
   - `source/isaaclab_experimental/isaaclab_experimental/utils/warp_graph_cache.py:63-68`
   - `source/isaaclab/isaaclab/utils/device.py:54-58` (`reserve_cuda_graph_memory`: return early when capture is unsupported)
   - PhysX-backend sites (`isaaclab_physx/sensors/ray_caster/ray_caster.py`, `contact_sensor.py`) only if a Kit-less path reaches them; otherwise leave them (they run under Kit only).
3. Tests that capture unconditionally: guard with the same helper (skip the capture-specific assertion with a reason that names the missing capability, run the eager path). Do not skip a whole test that has an eager variant. Sites: `isaaclab/test/utils/test_warp_*`, `test_wrench_composer.py`, `isaaclab_newton/test/physics/test_newton_*`, `isaaclab_experimental/test/envs/mdp/*`, `isaaclab_ppisp/test/test_ppisp_kernels.py`, sensors tests -- only those the validation run actually reaches.
4. `source/isaaclab/isaaclab/benchmark/recorders/record_gpu_info.py`, `source/isaaclab/isaaclab/cli/multigpu.py`: torch-based fallback when `nvidia-smi`/`pynvml` is absent (report `torch.version.hip`). Ray scripts (`scripts/reinforcement_learning/ray/util.py`, `wrap_resources.py`, `task_runner.py`): same fallback for GPU counting, if small; else owed with a reason.
5. Install recipe: a ROCm section in `docs/source/isaaclab/installation.rst` (Kit-less only), plus an override file next to the existing `tools/wheel_builder/uv-overrides.txt` pattern if needed for `warp-lang`. Do NOT add a `rocm` extra/conflict to `pyproject.toml` (forces a uv.lock regeneration and a large diff). Do NOT change the cu130 default.
6. Changelog fragments per package as upstream's contributing guide requires (`tools/changelog/cli.py check`), and the version-bump rules in `docs/source/refs/contributing.rst` ("Maintaining package changelogs and versions").
7. Nothing in Kit-, OV-, PhysX-, mimic- or teleop-only code.

## Build commands (linux-gfx942 first)

There is no compile step for Isaac Lab; the "build" is the environment. Record exact versions in notes.

```bash
# 0. ROCm 7.2.x host, Python 3.12, uv (Isaac Lab requires >=3.12,<3.13)
git clone https://github.com/AMD-Ecosystem/IsaacLab.git && cd IsaacLab && git checkout moat-port
uv venv --python 3.12 .venv && . .venv/bin/activate

# 1. ROCm torch
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/rocm7.2

# 2. AMD Warp from source, pinned (outside the Isaac Lab tree)
git clone https://github.com/AMD-Ecosystem/warp.git ../warp-rocm && git -C ../warp-rocm checkout ac540f595ff5763a374c113d77e4111a1d4791bd
(cd ../warp-rocm && python build_lib.py --hip-arch=gfx942 && uv pip install -e .)
python -c "import warp as wp; wp.init(); d=wp.get_device('cuda:0'); print(wp.__version__, d.arch_str, d.is_hip, d.supports_graph_capture)"

# 3. Isaac Lab workspace without replacing torch/warp
uv pip install -e . --override <overrides file pinning warp-lang to the installed AMD build and torch to +rocm> --extra test
python -c "import warp, torch, newton, mujoco_warp; print(warp.__version__, torch.__version__, torch.version.hip, newton.__version__, mujoco_warp.__version__)"
```

The exact override mechanics (uv `--override` file vs. `--no-deps` for Newton/MuJoCo Warp) are the porter's to settle and record; the invariant is that `pip list` afterwards shows the `+rocm` torch and the AMD Warp build, and no `nvidia-*` CUDA runtime wheels. Wrap commands in `utils/timeit.sh IsaacLab <phase> --`.

For another wave64 arch, change only `--hip-arch`. For wave32, see the platform order above.

## Test plan

GPU tests (the real gate), run with `tools/conftest.py`'s own selection so the set matches upstream CI:

```bash
CI_MARKER=kitless uv run python tools/run_all_tests.py      # or the per-file form below
uv run python -m pytest -m kitless <file>
```

`CI_MARKER=kitless` selects files containing the literal `pytest.mark.kitless` (`tools/conftest.py:1707-1716`). At `eed2d288` that is 12 files (intake's 35 counted every mention of the word):

- In scope (10): `isaaclab/test/sim/test_mjcf_converter.py`, `test_urdf_converter.py`, `test_schemas_deprecation.py`; `isaaclab_newton/test/assets/test_articulation.py`, `test_rigid_object.py`, `test_newton_actuators_newton.py`, `test_bam_actuator_mjwarp.py`; `isaaclab_tasks/test/test_microduck_velocity.py`; `isaaclab_visualizers/test/test_newton_adapter.py`, `test_newton_gl_dynamic_mesh.py` (needs a GL context: run headless/EGL where possible; on compute-only CDNA a GL-only case is scoped per the graphics-pipeline rule, not waived).
- `isaaclab/test/multi_gpu/test_multi_gpu_training_smoke.py`: in scope on a host with >= 2 GPUs.
- `_isaaclab_testing/test/rendering/test_visualizers_rtx.py`: RTX/OVRTX (NVIDIA binary) -- out of scope; record its kitless-but-ov parameters as skipped with that reason.

Beyond the marker: the Warp-kernel unit tests that need neither Kit nor OV (for example `isaaclab/test/utils/test_warp_math.py`, `test_warp_memory.py`, `test_wrench_composer.py`, `isaaclab/test/sensors/test_*_kernels.py`, `isaaclab_newton/test/physics/test_newton_solvers.py`, `test_newton_manager.py`, `isaaclab_ppisp/test/test_ppisp_kernels.py`, `isaaclab_experimental/test/envs/mdp/*parity*.py`). The porter runs each candidate once and classifies it (passes / needs Kit / needs OV / real failure); the passing set becomes the recorded GPU regression set alongside the marker set. A file that needs Kit is not an AMD failure.

End-to-end smoke (training actually learns), Newton + MuJoCo Warp, headless:

```bash
uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct --num_envs 4096 --max_iterations 100 --headless presets=newton_mjwarp
uv run isaaclab benchmark runtime --task Isaac-Cartpole-Direct --num_envs 4096 --num_steps 1000 --warmup_steps 50 presets=newton_mjwarp
```

(Forms from `docs/source/isaaclab/tutorial.rst:305,323` and the CI benchmark matrix `.github/workflows/build.yaml:1005`; exact flags to be confirmed by the porter.) Pass bar: reward rises to the level a CUDA run of the same command reaches (record the number); steps/s recorded, not gated. A double-stepping bug (risk 1) shows up here as a physics-time mismatch even if rewards look plausible, so also assert sim time / step count in one short run.

Non-GPU regression set: `uv run isaaclab --test` (tools/ tests) and the CPU-device parametrizations of the files above must not regress; run them on the ROCm env and compare against the same command on the upstream sha.

CUDA no-regression: no CUDA-compiled code exists, so the nvcc compile gate degenerates to "the CUDA install path is byte-identical in `pyproject.toml`/`uv.lock` and the helper preserves `is_cuda` semantics when `supports_graph_capture` is absent or True". If an NVIDIA host is available, run the kitless marker set there once on `moat-port` to show capture still engages (the `use_cuda_graph` path) -- record whether done.

## Open questions

Recorded verbatim in `notes.md` ("Planner questions for jeff"). Summary and the assumption taken while jeff is away:

1. Dependency forks (1a: consume AMD-Ecosystem/warp `amd-integeration-dev` @ `ac540f59` unmodified and upstream Newton/MuJoCo Warp; 1b: wave32 needs the halo branch's wavefront fix merged into the dev line by its owners). Assumption: consume as-is, never push there; wave32 attempted as described and blocked with the Warp defect named if it fails.
2. Fork settings: create `main` mirror / `moat-port` default. Assumption: the porter cuts `moat-port` from upstream `main` @ the planned sha and pushes only `moat-port`; `release/3.0.0-beta2` untouched.
3. Windows waiver suggestion (AMD Warp has no Windows HIP build). Assumption: windows last; no waiver recorded by an agent.
4. Upstream PR to an NVIDIA-owned repo vs fork-as-deliverable. Assumption: keep the diff upstream-shaped; decide at the PR gate.
5. Scope: Kit/Isaac Sim, OV (`ovphysx`/`ovrtx`/`ovstage`), PhysX backend, mimic and teleop are excluded because they need NVIDIA proprietary binaries, not deferred. Assumption: this follows from jeff's 2026-11-05 Kit-less framing and does not need a separate scope-cut ruling; flagged so he can overrule.
