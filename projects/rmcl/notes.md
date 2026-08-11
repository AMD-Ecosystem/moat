# rmcl notes

## The ROCm port lives in rmagine, not here

rmcl is a thin ROS 2 layer over the rmagine ray-casting library, which it pulls as an external git dependency via `source_dependencies.yaml` rather than vendoring. The GPU compute rmcl uses IS rmagine's CUDA backend, so that is what was ported: rmagine has 21 `.cu` files to rmcl's 3.

That work was done on `AMD-Ecosystem/rmagine` between 2026-06-01 and 2026-06-05 and validated on four architectures, but `projects/rmagine/` did not exist until 2026-08-06, so it was recorded here under rmcl's name. The record has been moved to `projects/rmagine/`, which is where the notes, the plan and the validation history now live. `depends_on: [rmagine]` records the relationship the tooling can act on.

Nothing about the port changed; only where it is filed.

## What rmcl itself still needs

Its own GPU code is unported: `rmcl_ros`'s `particle_motion.cu` and `resampling.cu`, plus the MICP CUDA sensors and the ROS 2 nodes. Both `.cu` files look mechanical -- curand to hiprand, and resampling's reduction already runs the full `__syncthreads` tree, so it is wave-safe. They need ROS 2 jazzy and Embree, which were not available on the host that did the rmagine work, and they should be built through rmagine's HIP toolchain in a colcon workspace.

This needs a fresh plan: the one written in 2026-05 analysed rmcl and concluded the port target was rmagine, and it moved with the work it produced.

## Intake screen 2026-08-11 (linux-gfx942) -- recommend fork

Read-only screen of upstream `uos/rmcl` at `8892cf4` (shallow clone in `agent_space/rmcl-screen`, gitignored). No fork clone was touched and nothing was posted anywhere.

### Licence: BSD-3-Clause, tier 1, uniform

`python3 utils/licenses.py check uos/rmcl` -> `license=BSD-3-Clause tier=1, cleared to contribute`. Not taken on GitHub's word: the repository is a three-package ROS 2 workspace and each package carries its own LICENSE, so all four were read and hashed.

```
77799a4472a34721bb0f689833e85bc6  LICENSE
77799a4472a34721bb0f689833e85bc6  rmcl/LICENSE
77799a4472a34721bb0f689833e85bc6  rmcl_ros/LICENSE
77799a4472a34721bb0f689833e85bc6  rmcl_msgs/LICENSE
```

Identical BSD 3-Clause text, "Copyright (c) 2024, Alexander Mock", matching the `<license>BSD-3-Clause</license>` in each `package.xml`. No mixed or per-part licensing, so nothing here is unresolved.

Per-file checks both clean:

- `python3 utils/licenses.py scan-nvidia agent_space/rmcl-screen` -> no NVIDIA proprietary licence text. This matters more than usual because rmcl compiles OptiX device programs, but it only *consumes* the OptiX SDK -- `rmcl_ros/CMakeLists.txt` fetches the headers at configure time from NVIDIA's public repo rather than vendoring them, so no NVIDIA-licensed file is in the tree we would fork.
- No submodules (`.gitmodules` absent) and no vendored third-party directory. rmagine, the one external source dependency, is cloned by `source_dependencies.yaml` at build time and is separately BSD-3-Clause. So the EnvGS failure mode -- permissive top level over an unlicensed submodule -- does not apply.

Scope note: this clears CONTRIBUTING a port upstream. It says nothing about using rmcl in our own software, and nothing here suggests that is the intent.

### Duplicate effort: none

- `AMD-Ecosystem` contains exactly two `rm*` repositories, `rmcl` and `rmagine`, both ours.
- No `rmcl` or `rmagine` repository in the `ROCm` org.
- All 27 forks of `uos/rmcl` enumerated: personal/student forks plus `AMD-Ecosystem/rmcl` and the maintainer's own `amock/rmcl`. No AMD or ROCm port among them.
- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README.md docs/ CONTRIBUTING.md` returns **nothing**. No notable-forks section, no platform port linked, no ROCm mention anywhere in the docs. The README's only hardware framing is NVIDIA (OptiX) and CPU (Embree).

`AMD-Ecosystem/rmcl` already exists as a plain upstream mirror. Its branches are `main`, `develop`, `noetic`, `humble-v2.0.0`, `rmcl-rmcl`, `ba-fgeers`, `workflow-lyrical` -- all inherited from upstream. There is **no `moat-port` branch**, so no port work has been started in it. Because the fork exists, the intake outcome is `screened` rather than `awaiting-fork`; the fork's existence is the adoption decision and it was already made when the record was first created on 2026-05-30.

### Upstream health

Active, not archived. 254 stars, 27 forks, last push 2026-08-07 (four days before this screen). Maintained by the paper authors at Osnabruck/Nature Robots, with a live `develop` branch and CI across three ROS 2 distros. An upstream PR has a real destination.

### Viability: yes, but the size is not the line count

The `.cu` surface is small -- 508 lines across seven files:

```
 10  rmcl_ros/include/rmcl_ros/rmcl/optix/cuda_rmagine_conversions.cuh
 23  rmcl_ros/include/rmcl_ros/rmcl/particle_motion.cuh
 30  rmcl_ros/include/rmcl_ros/rmcl/optix/cuda_math_helper.cuh
 47  rmcl_ros/include/rmcl_ros/rmcl/resampling.cuh
 48  rmcl_ros/src/rmcl/particle_motion.cu
130  rmcl_ros/src/rmcl/optix/BeamEvaluateProgram.cu
220  rmcl_ros/src/rmcl/resampling.cu
```

Device-side API use is thin: curand (`curandState`, `curand_init`, `curand_normal`), a handful of `cudaMalloc`/`cudaMemcpyAsync`, two `__syncthreads`, no `__shfl`, no ballot, no cuBLAS/cuSOLVER call in rmcl's own code. Taken alone that is a morning's work.

**It is not alone.** The decisive structural fact is that rmcl has no CUDA-only execution path. `rmcl_ros/src/nodes/micp_localization.cpp` selects a correspondence backend by string, and the only three strings are `"embree"` (CPU), `"optix"` and `"vulkan"`. The `MICPSensorCUDA` family is instantiated **only inside `#ifdef RMCL_OPTIX`** (micp_localization.cpp:618-696) -- the CUDA compute classes are passengers of the OptiX ray-casting backend, not a backend of their own. Likewise the global-localization node that consumes `particle_motion.cu` and `resampling.cu` is gated at `rmcl_ros/CMakeLists.txt:701`:

```cmake
if(RMCL_OPTIX AND RMCL_EMBREE)
# The first prototype (v0) for global localization will only compile when both and optix and embree is available.
# So only for NVIDIA devices it is possible to
```

Upstream says it in their own comment: the interesting path is NVIDIA-only. So a port that stops at curand-to-hiprand yields a `rmcl_ros_cuda` that compiles and that no node can reach. Delivering a *usable* AMD path means supplying the ray-casting backend too: `RCCOptix` -> an `RCC*Hiprt`, `BeamEvaluateProgram.cu`'s raygen/closesthit/miss PTX -> a HIPRT trace kernel, and a new backend string wired into the node's selection logic.

That is inventive work, but it is not speculative -- **we have already done the hard half.** rmagine's Stage 2 (`projects/rmagine/notes.md`, fork HEAD `4223818`) reimplemented rmagine's entire OptiX ray-mesh backend against HIPRT and validated all four sensor types on gfx90a. rmcl consumes exactly those four. The rmagine notes name this remainder explicitly as the deferred "rmcl-layer milestone", and they name the blocker: ROS 2 jazzy and Embree, absent on the host that did rmagine.

One API caveat the planner must not miss: rmagine's HIP build did **not** produce a drop-in `rmagine::optix` target. Stage 2 added parallel classes (`PinholeSimulatorHiprt`, `SphericalSimulatorHiprt`, `O1DnSimulatorHiprt`, `OnDnSimulatorHiprt`) under new names. Every rmcl OptiX target is gated on `if(TARGET rmagine::optix)`, so those gates stay dark on a ROCm rmagine no matter what. The CMake wiring is a design question, not a substitution.

### Test surface: zero. This is the real risk.

There are no tests in this repository at all -- no test directory, no `ament_add_gtest`, no `add_test`, no `BUILD_TESTING`, no gtest or Catch2 dependency, no launch file, no `.rviz`, no config, no sample mesh. The four CI workflows (`ros2_humble/jazzy/lyrical` on a shared template) run `colcon build` and then a `colcon test` step with `skip-tests: false` that has nothing to execute.

MOAT requires a real AMD GPU pass, so the validator cannot inherit a suite here and will have to construct the evidence. Two viable routes, both for the planner to weigh:

1. `amock/rmcl_examples` (BSD-3-Clause, maintainer's own) carries the meshes and launch files the README points newcomers to. That is the closest thing to a runnable end-to-end demo, at the cost of a second external repository in the loop.
2. A synthetic harness in the fork driving the MICP sensors and the resampler against an rmagine mesh, mirroring what rmagine's own `test_all_simulators` and `cuda_math_reduction_correctness` did. That follows the pattern that already produced trustworthy evidence on this dependency.

`docs/MICPL.md` shows the node printing an `--- BACKENDS --- / Available raytracing backends:` block at startup, which gives a cheap dispatch check once something runs.

### Host and toolchain reality (linux-gfx942)

- Ubuntu 24.04.4 noble, which is the ROS 2 **jazzy** target -- so the distro half of rmagine's deferral is satisfied on this host. ROS 2 itself is not installed (`/opt/ros` absent) and neither is Embree.
- ROCm 7.14 via the pip `_rocm_sdk_devel` package under `/opt/conda/envs/py_3.12/...`, not `/opt/rocm`. hiprand headers are present (`hiprand.h`, `hiprand_kernel.h`), so the curand half needs no provisioning. Note the non-standard prefix: anything hardcoding `/opt/rocm` will miss.
- HIPRT is **not** present (`HIPRT_PATH` unset, nothing on disk). rmagine's Stage 2 needs `HIPRT_PATH` at runtime for Orochi JIT kernel discovery, so the HIPRT SDK has to be provisioned before the ray-casting half of this port can be built or validated.

### Dependency: rmagine -- satisfied for building, with two caveats

`moatlib.py deps` reports `rmcl: depends_on=['rmagine'] -> READY (deps complete)`. Not unknown and not doomed: `projects/rmagine/` lives on branch `port/rmagine`, is at stage `review-passed`, and is `completed` on four platforms (linux-gfx90a, linux-gfx1100, windows-gfx1101, windows-gfx1201) at fork HEAD `4223818`. `AMD-Ecosystem/rmagine` exists. So no intake request is needed and nothing needs rescoping around a dead dependency.

Two things a planner should carry anyway:

1. **The `## Install as a dependency` section is missing.** `moatlib.py dep-doc-gaps` lists `rmagine -> rmcl` among five such gaps, and orient nags about it. rmagine's notes do contain a validated standalone build recipe (`-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=<arch>`, with `RMAGINE_EMBREE_DISABLE=ON` and friends), but no install-prefix and consume instructions, and the recipe as written *disables* the Embree and ray-tracing backends that rmcl needs turned on. Writing that section is work on rmagine's branch, not here -- DEPENDENCIES.md is explicit that the provider writes it from its own validated build. Flagged rather than fixed, deliberately.
2. **Upstream sequencing.** rmagine is review-passed but has no upstream PR recorded. rmcl's port would build against `AMD-Ecosystem/rmagine@moat-port`, so an upstream rmcl PR cannot land before an upstream rmagine PR does. That constrains publication order, not the port.

### Recommendation

Recommend **fork** (adopt), `--viable yes`. The licence is clean at tier 1, no one else has ported it, upstream is active and can accept a PR, the dependency is ported and validated by us, and upstream has documented in a source comment that its best localization path is NVIDIA-only -- which is precisely the gap worth closing.

The honest counterweight, which belongs in front of whoever decides: this is not the 508-line job the `.cu` count suggests. Scope it as a HIPRT correspondence backend at the rmcl layer plus a from-scratch validation harness, needing ROS 2 jazzy, Embree, and a provisioned HIPRT SDK. A planner who scopes it as "curand to hiprand" will produce code no node can execute.

Nothing here decides anything: adoption stays with the person reading the queue.
