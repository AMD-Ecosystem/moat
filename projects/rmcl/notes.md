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

## Port round 2026-08-13 (linux-gfx942, MI300X, gfx942) -- built, tested, pushed

Executed `plan.md`. Fork `AMD-Ecosystem/rmcl`, branch `moat-port` cut from upstream `main`
`8892cf4` (the fork's own `main` was one commit behind upstream and was left alone). Head
`a8e6f88`.

Two commits:

- `dde091b [ROCm] Compile the rmcl_ros GPU code with HIP`
- `a8e6f88 [ROCm] Add a GPU test for the rmcl_ros kernels`

### What was changed

New `rmcl_ros/include/rmcl_ros/util/cuda_to_hip.h` -- rmcl's own shim (NOT rmagine's, per the
plan: rmagine's exists only on our fork). Maps `curandState`, `curand_init`, `curand`,
`curand_normal` under `USE_HIP || __HIP_PLATFORM_AMD__`, includes the real CUDA headers
otherwise. Host-includable, verified directly: `g++ -std=c++17 -D__HIP_PLATFORM_AMD__ -c` over
`<cstdio>` + `<hiprand/hiprand_kernel.h>` compiles, so `GladiatorResamplerGPU.cpp` stayed a
plain C++ TU and the LANGUAGE HIP set is just the two `.cu` files. Plan risk 3 did not
materialise.

`rmcl_ros/CMakeLists.txt`: `option(USE_HIP ... ${rmagine_cuda_USE_HIP})` -- it defaults from
the variable rmagine's ROCm build exports, so a workspace that found a ROCm rmagine does not
have to pass the flag (verified: a configure with no `-DUSE_HIP` gets `USE_HIP:BOOL=ON`). In
the `RMCL_CUDA` block: `find_package(hip)`, `find_package(hiprand)`, `enable_language(HIP)`
under `USE_HIP`, and the existing CUDA detection gated `if(CMAKE_CUDA_COMPILER AND NOT
USE_HIP)`. After `add_library(rmcl_ros_cuda ...)`: `LANGUAGE HIP` on the two `.cu`,
`USE_HIP` as a PUBLIC compile definition (the installed headers switch spelling with it), and
`hip::host` + `hip::hiprand` PUBLIC.

Include redirects (4 files) to the shim. Two unconditional fixes, correct on CUDA too:
`<climits>` in `resampling.cu` for `UINT_MAX` (plan risk 5, as predicted), and the default
member initializers removed from `SimpleLikelihoodStats` (see gotchas).

README: an AMD paragraph plus a `colcon build --cmake-args -DUSE_HIP=ON
-DCMAKE_HIP_ARCHITECTURES=gfx942` block in the Installation section, matching that section's
terse house style, and stating plainly that ray casting still needs OptiX so correspondence
search on AMD runs on the CPU.

New `rmcl_ros/tests/` (upstream has no tests at all): one executable
`rmcl_ros_tests_gpu_kernels`, `add_test(NAME rmcl_gpu_kernels)`, built under `if(BUILD_TESTING
AND RMCL_CUDA)`. Content per the plan's test plan.

`rmcl/CMakeLists.txt` was NOT touched -- confirmed empirically, as the plan asked: on a
CUDA-free host `CMAKE_CUDA_COMPILER` is unset, `CUDA_LIBRARIES` stays empty and `rmcl`,
`rmcl-cuda`, `rmcl-embree` build with zero changes. The OptiX and Vulkan sources are
untouched (`git diff --stat` is 6 files + 2 new).

### Gotchas (both promoted to the cuda-to-rocm skill, naming rmcl)

1. `__shared__ SimpleLikelihoodStats sdata[blockSize]` where the struct carries default member
   initializers (`float sum = 0.0; float max = -1.0;`): clang errors `initialization is not
   supported for __shared__ variables`; nvcc accepts it and drops the initialization. The
   kernel writes both members before reading them and no host code default-constructs the
   type (`rm::Memory<..., VRAM_CUDA>` never runs constructors), so the initializers were
   already dead. Removed them -> `references/fault-classes.md`, "Memory and lifetime".
2. `hip::device` on `rmcl_ros_cuda` breaks the build: `--offload-arch=gfx942` lands on the five
   `MICP*SensorCUDA.cpp` host compiles (`c++: error: unrecognized command-line option`).
   PRIVATE does not help -- that controls propagation to consumers, not the target's own
   sources -- and `$<COMPILE_LANGUAGE:HIP>` is not allowed in `target_link_libraries`. With
   `enable_language(HIP)` the flag comes from `CMAKE_HIP_ARCHITECTURES` for the HIP sources
   anyway, so `hip::device` is simply not needed. This is a refinement of plan risk 13, which
   expected PRIVATE to be the answer -> `references/strategy-a-cmake.md`, "Build hygiene".
3. The rmagine install path predicted as broken in the plan is FIXED: rmagine at `e7a7b27`
   installs and is consumed cleanly (`find_dependency(hip)` / `find_dependency(hiprand)` in
   its cuda config). Plan risks 1 and 2 no longer apply; the plan's open question 1 is
   answered by rmagine's own delta round.
4. ROS 2 tooling picks up whatever `python3` is first on `PATH`. With the conda python first,
   `rosidl` fails with `ModuleNotFoundError: No module named 'em'`; the apt ROS stack wants
   `/usr/bin/python3`. Fix: `export PATH=/usr/bin:$PATH` before sourcing
   `/opt/ros/jazzy/setup.bash`. Host-specific, so it stays here.

### Environment (linux-gfx942)

- Ubuntu 24.04.4 noble, ROCm 7.14 from the pip `_rocm_sdk_devel` package under
  `/opt/conda/envs/py_3.12/lib/python3.12/site-packages/_rocm_sdk_devel` (there is no
  `/opt/rocm` on this host; pass that prefix on `CMAKE_PREFIX_PATH` and as
  `CMAKE_HIP_COMPILER=<prefix>/llvm/bin/clang++`).
- ROS 2 jazzy installed for this port from `packages.ros.org` (`ros-jazzy-ros-base`, tf2,
  tf2-ros, image-transport, visualization-msgs, std-srvs, rclcpp-components,
  rosidl-default-generators, python3-colcon-common-extensions), plus `libembree-dev`,
  `libassimp-dev`, `libeigen3-dev`, `libtbb-dev`.
- GPU: AMD Instinct MI300X, `amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-`.

### Exact commands

Dependency (rmagine fork `moat-port` at `e7a7b27`), built and installed to a staging prefix:

```
cmake -S projects/rmagine/src -B agent_space/rmcl/rmagine_build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_HIP_COMPILER=$ROCM/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="$ROCM" \
  -DRMAGINE_OPTIX_DISABLE=ON -DRMAGINE_VULKAN_DISABLE=ON \
  -DRMAGINE_VULKAN_CUDA_INTEROP_DISABLE=ON -DRMAGINE_OUSTER_DISABLE=ON \
  -DRMAGINE_BUILD_TESTS=OFF -DRMAGINE_BUILD_TOOLS=OFF \
  -DCMAKE_INSTALL_PREFIX=agent_space/rmcl/rmagine_install
cmake --build agent_space/rmcl/rmagine_build -j && cmake --install agent_space/rmcl/rmagine_build
```

Note the deviation from rmagine's own recipe: `RMAGINE_EMBREE_DISABLE` is left unset on
purpose, so `rmagine::embree` exists and rmcl's CPU backend (`rmcl-embree`,
`rmcl_embree_ros`, the Embree segmentation nodes) is part of the regression set.

Workspace:

```
export PATH=/usr/bin:$PATH
. /opt/ros/jazzy/setup.bash
export CMAKE_PREFIX_PATH="<ws>/rmagine_install:$ROCM:$CMAKE_PREFIX_PATH"
colcon build --base-paths projects/rmcl/src \
  --packages-select rmcl_msgs rmcl rmcl_ros \
  --build-base <ws>/build --install-base <ws>/install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
               -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
               -DCMAKE_HIP_COMPILER=$ROCM/llvm/bin/clang++
```

Result: `Summary: 3 packages finished [4min 46s]`, no errors, no CMake dev warnings. Device
code is present and correctly targeted:

```
llvm-objdump --offloading <ws>/install/rmcl_ros/lib/librmcl_ros_cuda.so
Extracting offload bundle: ... .0.hipv4-amdgcn-amd-amdhsa--gfx942
```

### Test results

```
ctest --test-dir <ws>/build/rmcl_ros --output-on-failure
1/1 Test #1: rmcl_gpu_kernels .................   Passed    0.42 sec
100% tests passed, 0 tests failed out of 1
```

```
particle_move_and_forget: OK (4099 particles)
compute_stats: OK (sum=2051.63 max=1)
gladiator_resample: OK (2043 of 4099 particles replaced, noise mean=-0.00274891 sd=1.01771)
```

`AMD_LOG_LEVEL=3` confirms all four kernels reach the device (not just the copies):

```
ShaderName : rmcl::init_curand_kernel(hiprandState*, unsigned int)
ShaderName : void rmcl::simple_stats_kernel<512u>(rmagine::Transform_<float> const*, ...)
ShaderName : rmcl::particle_move_and_forget_kernel(rmagine::Transform_<float>*, ...)
ShaderName : rmcl::gladiator_resample_kernel(rmagine::Transform_<float> const*, ..., hiprandState*, ...)
```

How the resampler is asserted without a comparable RNG stream (hipRAND's sequence differs
from cuRAND's, plan risk 6): likelihoods are made unique (`mean = i + 1`), so the winner of a
duel is recoverable from the resampled attributes, and with unit translation noise and zero
rotation noise the recovered offset IS the drawn normal triple. That gives a real
distribution check (mean `-0.0027`, sd `1.018` over ~6100 samples) on top of the
stream-independent invariants: a lost duel copies the particle through bitwise, a won duel
must carry a strictly greater likelihood, nothing is NaN, and two runs from freshly seeded
states agree bitwise.

### Not done, deliberately

- The optional `rmcl` package test over `CorrespondencesCUDA::computeCrossStatistics` (plan's
  "optional second test"). The three required kernel tests are in and pass; this one is worth
  adding if a reviewer wants the `rmcl` package to carry evidence of its own.
- Everything about OptiX/HIPRT and the reachable-GPU-backend question. Unchanged from the
  plan: the GPU correspondence path is still unreachable on a ROCm-only configuration, and
  the README and the commit message both say so rather than implying MICP-L now runs on AMD
  GPUs.
- No NVIDIA host exists in the fleet, so the CUDA no-regression claim rests on construction:
  every change is inside `USE_HIP` except `<climits>`, the `SimpleLikelihoodStats`
  initializers and the README. That is exactly the set a reviewer should check.
