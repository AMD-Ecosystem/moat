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

## Review 2026-08-13 (linux-gfx942) -- changes requested

Reviewed `git diff 8892cf4...a8e6f88` on `AMD-Ecosystem/rmcl` `moat-port` (9 files, +396/-12),
both commit messages, `surface.json`, and the two skill lessons this branch promotes. Working
tree clean, `jargon.py --port rmcl` clean, no OptiX/Vulkan source touched, no `cuda[A-Z]` API
call left unshimmed, no warp intrinsic / `warpSize` / hardcoded 32 anywhere in the ported code,
and `simple_stats_kernel` (`rmcl_ros/src/rmcl/resampling.cu:67-74`) runs the full `__syncthreads`
tree with the barrier outside `if(tid < s)`, so it is wave32/wave64 safe as the plan claimed.

### Problems

1. **The rationale in commit `dde091b` states something nvcc does not do.** The body says
   "nvcc dropped the initializers with a warning and clang rejects the declaration outright".
   The clang half is right (reproduced: `initialization is not supported for __shared__
   variables`, gfx942). The nvcc half is not: CUDA 12.8.93 compiles the pre-port
   `resampling.cu` -- NSDMIs present, `__shared__ SimpleLikelihoodStats sdata[blockSize]` at
   `rmcl_ros/src/rmcl/resampling.cu:47` -- with no diagnostic at all, and a four-line repro with
   `-Xcudafe --display_error_number` is equally silent. Reword to say nvcc accepts the
   declaration and ignores the initializers. This is upstream-visible text that a maintainer can
   check in a minute, and it is the paragraph that justifies an unconditional edit to a public
   header. Fix it now rather than later: amending a commit body after validation moves
   `head_sha` and throws away every platform's evidence, whereas today it costs nothing.

2. **The added GPU test has no skip path, so `colcon test` acquires a hard failure on any
   machine that has the toolkit but no usable device.** `rmcl_ros/CMakeLists.txt:815` gates the
   test on `BUILD_TESTING AND RMCL_CUDA`, and `RMCL_CUDA` is a statement about `rmagine::cuda`
   being installed, not about a GPU being present; `rmcl_ros/tests/CMakeLists.txt:9` then
   registers it unconditionally. Upstream CI is safe (the ROS docker images carry no CUDA, so
   `RMCL_CUDA` is false and the subdirectory is never added), but a headless build host or a
   container without device access goes from "colcon test runs nothing" to "colcon test fails",
   in a repository whose README advertises per-distro test badges. Return 77 from `main()` when
   the device count is zero and add
   `set_tests_properties(rmcl_gpu_kernels PROPERTIES SKIP_RETURN_CODE 77)`; the device-count
   call needs two more mappings in `rmcl_ros/include/rmcl_ros/util/cuda_to_hip.h`, which is
   exactly what that header is for.

3. **`rmcl_ros/CMakeLists.txt:233-234` explains `PUBLIC USE_HIP` with a fact that is not true of
   this package.** The comment says "the installed headers switch to the hip spellings of
   curandState and friends with it", but `rmcl_ros` installs no headers: the file contains only
   `install(TARGETS ...)`, there is no `install(DIRECTORY include ...)`, and the installed tree
   from the port's own build contains just `lib/` and `share/`. `PUBLIC` is still the right
   choice, for a different reason -- in-workspace consumers that include these headers, namely
   the new test target and `rmcl_localization` under `RMCL_OPTIX`. Say that instead.

4. **The promoted fault-class lesson hedges where the evidence is now definite.**
   `.claude/skills/cuda-to-rocm/references/fault-classes.md` says nvcc "drops the NSDMIs
   silently (or with a warning)". Measured on CUDA 12.8.93: silent, no diagnostic. Drop the
   parenthesis and name the toolkit version tested, so the next reader does not go looking for a
   warning that is not there. The `strategy-a-cmake.md` lesson about `hip::device` needs no
   change -- it is exactly right, and the mechanism is visible in ROCm's own
   `lib/cmake/hip/hip-config.cmake:77`, which wraps the interface flags in
   `$<$<COMPILE_LANGUAGE:CXX>:...>`, i.e. the `--offload-arch` lands on the target's C++ sources
   and not on its HIP ones.

### Verified here, so it does not have to be re-argued

The plan left "no NVIDIA host in the fleet, so the CUDA path is proven by construction and
checked in review" as an open item. It is now proven by compilation instead. This host carries a
CUDA 12.8 toolkit at `/opt/conda/envs/cuda-12.8`, and all four touched translation units build
against the real CUDA headers at `a8e6f88`, with the `USE_HIP` branch of the shim inactive:

```
CUDAINC=/opt/conda/envs/cuda-12.8/targets/x86_64-linux/include
INC="-Irmcl_ros/include -Irmcl/include -I<ws>/install/rmcl_msgs/include/rmcl_msgs \
     -I<rmagine>/src/rmagine_core/include -I<rmagine>/src/rmagine_cuda/include \
     -I$CUDAINC $(for d in /opt/ros/jazzy/include/*; do echo -I$d; done)"
nvcc -std=c++17 -arch=sm_75 -c rmcl_ros/src/rmcl/resampling.cu      $INC   # ok
nvcc -std=c++17 -arch=sm_75 -c rmcl_ros/src/rmcl/particle_motion.cu $INC   # ok
g++  -std=c++17 -c rmcl_ros/src/rmcl/GladiatorResamplerGPU.cpp      $INC   # ok
g++  -std=c++17 -c rmcl_ros/tests/rmcl_gpu_kernels.cpp              $INC   # ok
```

Only the three pre-existing unused-variable warnings (`resampling.cu:120,121,136`) appear. The
last two lines matter most: the two host translation units that include the new shim still
compile with a plain host compiler against `curand_kernel.h`, so the port does not break the
NVIDIA build of the file it adds. Worth quoting in the PR body -- it is the evidence upstream
will want and it is cheap to reproduce.

Also confirmed rather than taken on trust: rmagine's installed `rmagine-cuda-config.cmake`
really does export `set(rmagine_cuda_USE_HIP ON)` and `find_dependency(hip)` /
`find_dependency(hiprand)`, so the `option(USE_HIP ... ${rmagine_cuda_USE_HIP})` auto-default
works and plan risks 1 and 2 are genuinely retired; `e7a7b27` is an ancestor of the current
rmagine fork tip `1213551`, which adds only a test comment, so the dependency the port was built
against is current. The OptiX and correspondence scoping is honest: nothing under `optix/` or
`vulkan` is in the diff, and both the README paragraph and `dde091b` say plainly that ray casting
still needs OptiX.

### Not blocking, for whoever writes the PR body

The motion test's CPU reference (`rmcl_ros/tests/rmcl_gpu_kernels.cpp:100`) composes poses with
rmagine's own `Transform::operator*`, which is the same shared host/device source the kernel
uses, so it is a reference for indexing, coverage and dispatch but not an independent
implementation of the quaternion math. The reduction and forget-rate references are independent.
Nothing to change; just do not oversell it as "checked against an independent CPU implementation".

## Fix round 2026-08-13 (linux-gfx1100, Radeon Pro W7800, gfx1100) -- review items applied, rebuilt, tested

Applied the four items from the review above. Head `a8e6f88` -> `4f746de`. Nothing was
validated at `a8e6f88` on any platform, so the amend the review asked for was free.

- `493d0f6` is `dde091b` with one sentence reworded: nvcc accepts the `__shared__` declaration
  and ignores the initializers (CUDA 12.8.93, no diagnostic) rather than "dropped them with a
  warning". Message only, tree identical.
- `f3d62d0` is `a8e6f88` replayed unchanged.
- `4f746de` is new: the device-count skip (review item 2) and the corrected `PUBLIC USE_HIP`
  comment (review item 3).
- Review item 4, the hedged fault-class lesson, is fixed on this branch in
  `.claude/skills/cuda-to-rocm/references/fault-classes.md`.

### Review item 2 is only partly fixable inside rmcl -- measured, not assumed

The test now returns 77 and the test carries `SKIP_RETURN_CODE 77`, which is the right shape
and is what the review asked for. It does not actually rescue a device-less machine, and the
reason is worth recording rather than leaving for the next person to rediscover:

`src/rmagine_cuda/src/util/cuda/CudaContext.cpp:198` defines
`CudaContextPtr cuda_def_ctx(new CudaContext(0));` at namespace scope. That global is
constructed when `librmagine-cuda.so` is loaded, i.e. before `main`, and on a machine with no
device it throws out of `cudaGetDeviceProperties` and the process aborts:

```
HIP_VISIBLE_DEVICES=-1 build/rmcl_ros/tests/rmcl_ros_tests_gpu_kernels
[RMagine - CudaContext] CUDA Driver Version / Runtime Version: 70253.21.1 / 70253.21.1
terminate called after throwing an instance of 'std::runtime_error'
  what():  Error calling cudaGetDeviceProperties
Aborted (core dumped)
```

So the skip path cannot be reached: any executable that links `rmagine::cuda` dies at load
time. A standalone probe confirms the runtime itself behaves as the test expects
(`hipGetDeviceCount` -> `err=100 count=0`), so the check is correct, it just runs too late.

Two CTest mechanisms were measured and neither closes the gap, so do not reach for them:

- `SKIP_REGULAR_EXPRESSION` is not consulted for a process that dies by signal. Minimal repro:
  a test that prints the pattern and calls `abort()` is still reported `***Failed (Subprocess
  aborted)`. It was written and then removed rather than shipped with a comment claiming it
  works.
- A `FIXTURES_SETUP` probe does not help either. If the probe skips, the dependent test still
  runs (and aborts); if the probe fails, the dependent test is reported `***Not Run` and both
  count as failures.

The only remaining in-rmcl option is a launcher script that runs the binary and translates the
abort into exit 77, which is a workaround for a dependency's fatal global constructor and does
not belong in an upstream tree. The real fix is making rmagine's default context lazy, which is
rmagine's change, not rmcl's. Registered as deferred work against rmagine; flagged here for the
reviewer to rule on whether the residual exposure (a host with rmagine's GPU backend installed
and no visible device) needs more than this.

Upstream CI is unaffected either way: the ROS docker images carry no GPU backend, `RMCL_CUDA`
is false, and `rmcl_ros/tests/` is never added.

### Environment (linux-gfx1100)

- Ubuntu 24.04 noble, ROCm 7.2.3 at `/opt/rocm` (unlike the gfx942 host, which has no
  `/opt/rocm`), CMake 3.31.6, 64 cores.
- GPU: AMD Radeon Pro W7800 48GB, `amdgcn-amd-amdhsa--gfx1100`.
- ROS 2 jazzy installed for this round from `packages.ros.org` over http (the https endpoint
  presents an `*.osuosl.org` certificate that does not match `packages.ros.org`; the packages
  are GPG-signed, so the http repository line is the working form on this host).
- The conda python is first on `PATH` here too, so `export PATH=/usr/bin:$PATH` before sourcing
  ROS is needed, exactly as the gfx942 round recorded. Alternatively
  `pip install "empy==3.3.4" lark catkin_pkg` into the active environment, which is what was
  done here before that gotcha was read.

### Exact commands

Dependency, rmagine fork `moat-port` at `1213551`, built and installed to a staging prefix
(Embree left enabled on purpose, so rmcl's CPU backend is in the regression set):

```
cmake -S agent_space/deps/rmagine_src -B agent_space/deps/rmagine_build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DRMAGINE_OPTIX_DISABLE=ON -DRMAGINE_VULKAN_DISABLE=ON \
  -DRMAGINE_VULKAN_CUDA_INTEROP_DISABLE=ON -DRMAGINE_OUSTER_DISABLE=ON \
  -DRMAGINE_BUILD_TESTS=OFF -DRMAGINE_BUILD_TOOLS=OFF \
  -DCMAKE_INSTALL_PREFIX=agent_space/deps/rmagine_install
cmake --build agent_space/deps/rmagine_build -j 32
cmake --install agent_space/deps/rmagine_build
```

Workspace (clean build base):

```
export PATH=/usr/bin:$PATH
. /opt/ros/jazzy/setup.bash
colcon build --packages-select rmcl_msgs rmcl rmcl_ros \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
               -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
               -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
               -DCMAKE_PREFIX_PATH="<ws>/rmagine_install;/opt/rocm"
```

`Summary: 3 packages finished [50.3s]`, no errors. Only pre-existing upstream warnings
(unused parameters in the CPU resamplers and converters, and the deprecated
`point_cloud_conversion.hpp` warning from sensor_msgs).

Tests:

```
export LD_LIBRARY_PATH=<ws>/rmagine_install/lib:$LD_LIBRARY_PATH
colcon test --packages-select rmcl_ros --event-handlers console_direct+
```

```
1: particle_move_and_forget: OK (4099 particles)
1: compute_stats: OK (sum=2051.63 max=1)
1: gladiator_resample: OK (2043 of 4099 particles replaced, noise mean=-0.00274891 sd=1.01771)
1/1 Test #1: rmcl_gpu_kernels .................   Passed    0.25 sec
100% tests passed, 0 tests failed out of 1
Summary: 1 test, 0 errors, 0 failures, 0 skipped
```

`LD_LIBRARY_PATH` is not optional under `colcon test`, and this is the one place the staging
prefix leaks: plain `ctest` in the build tree passes without it (build-tree RPATH), but
`colcon test` re-enters with its own environment and the binary fails with
`error while loading shared libraries: librmagine-core.so.2`. rmagine's own
`## Install as a dependency` section says as much; it is easy to read as advice and it is a
requirement.

Device evidence on this platform:

```
llvm-objdump --offloading install/rmcl_ros/lib/librmcl_ros_cuda.so
  -> hipv4-amdgcn-amd-amdhsa--gfx1100
AMD_LOG_LEVEL=3 -> ShaderName : rmcl::particle_move_and_forget_kernel(...)
                   ShaderName : void rmcl::simple_stats_kernel<512u>(...)
                   ShaderName : rmcl::init_curand_kernel(hiprandState*, unsigned int)
                   ShaderName : rmcl::gladiator_resample_kernel(..., hiprandState*, ...)
```

The kernel results are identical to the gfx942 round for the resampler
(`2043 of 4099`, `mean=-0.00274891`, `sd=1.01771`) because the input and the hipRAND stream are
the same; `compute_stats` reports the same `sum=2051.63 max=1`. Wave32 and wave64 agree, which
is what the plan predicted for a reduction that runs the full `__syncthreads` tree.

### Process note, for whoever reads the branch history

This round started from a stale checkout: the local `port/rmcl` had not been pulled, so the
record read `stage: planned, head_sha: null` and the gfx942 port and its review were invisible.
A complete second port was written and built before the first push revealed the existing
`moat-port`. It was discarded, not merged -- the branch here is the gfx942 port with the review
items applied. Pull `port/rmcl` before reading the state, not just at the end.

The discarded work did produce one thing worth weighing: a working, passing test for the `rmcl`
package (`CorrespondencesCUDA::computeCrossStatistics` against rmagine's host
`statistics_p2l`, 1740 of 3001 correspondences inside `max_dist`, mean error 4.8e-08, covariance
error 3.3e-05, needs no ROS). That is the plan's "optional second test" that the gfx942 round
deliberately left out and the review did not ask for. It is not on the branch. If the reviewer
wants `library:rmcl-cuda` to carry evidence of its own, say so and it can be re-added in a
commit of its own rather than being smuggled into a fix round.

## Fix round 2026-08-13 (linux-gfx942, MI300X) -- lost the push race, one measured disagreement

This host was dispatched on the same four review items and did them independently. It took the
fork-write lock at `02:23:06`, built and tested on MI300X, and then found `origin/moat-port`
already at `4f746de`: the gfx1100 round above pushed at `02:31:22`. `git push --force-with-lease`
refused with `stale info`, which is the guard working. Nothing was force-pushed over the gfx1100
tip; the fork clone here is back on `4f746de` and this round's two commits are preserved,
unpushed, on the local branch `gfx942-skip-launcher` in `projects/rmcl/src`.

**For a person: the lock did not keep the two hosts apart.** The record shows this arch entering
`porting` at `02:23:06` (`moatlib.py port-lock rmcl` confirmed the holder at the time) and the
gfx1100 round landing eight minutes later with `porting: null` and `stage: ported`. Both hosts
wrote the same project in the same window. Whether the lock was lost in the status merge or never
seen by the other host is worth checking before the next parallel round; nothing here can decide
it.

Review item 4 landed separately in `be6d289`, a few seconds after the round above; this host had
written the same correction independently and dropped it in favour of the pushed one. Worth
knowing for the next parallel round: `commit-project` stages only `projects/<name>/`, so a skill
edit needs its own commit and does not ride a project transition.

### Measured: a device-less machine can be skipped from inside rmcl after all

The gfx1100 notes conclude the skip cannot be closed without a launcher that translates the abort
into 77, and reject that as a workaround for a dependency's global constructor. The diagnosis is
right and was reproduced here independently (`hipGetDeviceCount` -> `err=100 n=0` under
`HIP_VISIBLE_DEVICES=-1`; the test binary dies at load in rmagine's `cuda_def_ctx`, exit 134, and
ctest reports `Subprocess aborted`, which is exactly the shape the in-`main` check produces).

The conclusion does not follow, and the difference is measurable. A launcher does not have to run
the aborting binary at all:

- `gpu_device_probe.cpp` (22 lines) links `hip::host` only -- not `rmcl_ros_cuda`, so rmagine is
  never loaded -- and exits 77 when the runtime reports no device.
- `run_gpu_test.cmake` (15 lines) is the registered ctest command: it runs the probe, and on 77 it
  stops with `message(FATAL_ERROR "no GPU device available, skipping")` without ever launching the
  test. Otherwise it runs the test and forwards its status.
- `SKIP_REGULAR_EXPRESSION "no GPU device available"` on the test.

The gfx1100 note that `SKIP_REGULAR_EXPRESSION` is not consulted for a process that dies by a
signal is correct, and is why the pattern has to be printed by the launcher rather than by the
test. Measured on this host against a full `-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942` build
of the workspace:

```
ctest --test-dir build/rmcl_ros
1/1 Test #1: rmcl_gpu_kernels .................   Passed    0.65 sec

HIP_VISIBLE_DEVICES=-1 ctest --test-dir build/rmcl_ros
1/1 Test #1: rmcl_gpu_kernels .................***Skipped   0.23 sec
```

So the residual exposure the gfx1100 round flagged for a reviewer to rule on is closable in rmcl
for about 60 lines, without touching rmagine. Whether that machinery is welcome upstream is a
judgement, not a measurement: gfx1100 judged it is not, this host measured that it works. A person
picks. Making rmagine's default context lazy remains the better fix either way, and the deferral
gfx1100 registered against rmagine stands.

### Independently reproduced here

- nvcc 12.8.93 compiles the pre-port `resampling.cu`, NSDMIs on `SimpleLikelihoodStats` and all,
  with no diagnostic, including under `-Xcudafe --display_error_number`; the four-line reduction of
  the same pattern is equally silent, and ROCm clang errors on it. Review item 1's wording is
  right.
- The whole touched CUDA surface still compiles against real CUDA headers with the `USE_HIP` branch
  of the shim inactive: `resampling.cu` and `particle_motion.cu` under nvcc 12.8.93 `-arch=sm_75`,
  and `GladiatorResamplerGPU.cpp`, `rmcl_gpu_kernels.cpp` and the probe under plain `g++ -std=c++17`.
  Only the three pre-existing unused-variable warnings appear.
- gfx942 kernel results match gfx1100 and the earlier gfx942 round exactly: `compute_stats`
  `sum=2051.63 max=1`, resampler `2043 of 4099` replaced, `mean=-0.00274891 sd=1.01771`.
## Review 2026-08-13 (linux-gfx1100, second round) -- changes requested

Re-reviewed the whole port, `git diff 8892cf4...4f746de` on `AMD-Ecosystem/rmcl` `moat-port`
(9 files, +419/-12), all three commit messages, `surface.json`, and the three skill lessons the
branch promotes. Working tree clean, `jargon.py --port rmcl` clean, `check.py` all gates ok.

### Problems

1. **The promoted `hip::device` lesson states a mechanism that ROCm's own CMake config
   contradicts, and contradicts itself two sentences later.**
   `.claude/skills/cuda-to-rocm/references/strategy-a-cmake.md:90-92` says "`hip::device` puts
   `--offload-arch=` in the target's INTERFACE compile options, and a link library applies to
   EVERY source of the target regardless of language". It does not. ROCm wraps every one of
   those flags in a language genex: `hip_add_interface_compile_flags` at
   `/opt/rocm/lib/cmake/hip/hip-config.cmake:91-94` appends
   `INTERFACE_COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:...>"`, and
   `hip-config-amd.cmake:159,166` is what feeds it `-x hip` and `--offload-arch=<gfx>` (ROCm
   7.2.3, read on this host). So the flags land on exactly the target's CXX sources -- which is
   why the five `MICP*SensorCUDA.cpp` compiles fail -- and never on its HIP-language sources.
   The lesson's own closing line, "a target that is 100% HIP sources never shows this", is only
   true under the real mechanism and is false under the one it states. Replace the clause with
   the CXX-genex fact and keep the rest; the advice (do not link `hip::device` on a mixed target;
   `enable_language(HIP)` plus `LANGUAGE HIP` already emits the offload flags) is correct.
   Worth adding while there: the link half is NOT gated -- `hip-config-amd.cmake:162,168` put
   `--hip-link` and `--offload-arch` into `INTERFACE_LINK_LIBRARIES` unconditionally, so
   `hip::device` also reaches the link line of any consumer.
   This is the one blocking item, and it is a doc-only edit: it does not touch the fork, so
   `head_sha` does not move and no evidence is invalidated.

2. **`surface.json` points `test:rmcl_gpu_tests` at a directory that does not exist.**
   The component's `where` reads "added by this port (rmcl_ros/tests, rmcl/tests)", but the
   branch has no `rmcl/tests` -- that was the optional rmcl-package test, deliberately not added
   (the `covered` entry says so correctly). Drop `rmcl/tests` from the `where` so the accounting
   the publisher reads matches the branch.

### Prior round's four items: all genuinely resolved

Checked independently rather than taken from the porter's summary.

1. The nvcc claim in `493d0f6` now reads "nvcc accepts the declaration and ignores the
   initializers (CUDA 12.8.93, no diagnostic) while clang rejects it outright". Reproduced both
   halves here on gfx1100: a 20-line repro of `__shared__ S sdata[blockSize]` over a struct with
   NSDMIs compiles silently under `/opt/conda/envs/cuda-12.8/bin/nvcc -arch=sm_75
   -Xcudafe --display_error_number` (rc 0, no output) and fails under
   `/opt/rocm/llvm/bin/clang++ -x hip --offload-arch=gfx1100` with `error: initialization is not
   supported for __shared__ variables`. The same measurement retires review item 4 for
   `fault-classes.md:189-198`, which now names the toolkit version and drops the "or with a
   warning" hedge.
2. The device skip is as good as it can be made inside rmcl, and the analysis is right, not
   assumed. `rmcl_gpu_kernels.cpp:283-288` returns 77 and `rmcl_ros/tests/CMakeLists.txt:16`
   carries `SKIP_RETURN_CODE 77`. The residual exposure is real and is in the dependency:
   `cuda_def_ctx` is a namespace-scope global at
   `rmagine_src/src/rmagine_cuda/src/util/cuda/CudaContext.cpp:199` (the notes say 198; it is
   199 at rmagine's tip), constructed at library load, so a device-less host aborts before
   `main`. Registered as `rmagine-lazy-default-gpu-context` on `port/rmagine`, confirmed present
   in `deferred.py pending`. Not a blocker: see the ruling below.
3. The `PUBLIC USE_HIP` comment (`rmcl_ros/CMakeLists.txt:233-235`) no longer claims this package
   installs headers. Its replacement is accurate: `rmcl_localization.cpp:11` includes
   `GladiatorResamplerGPU.hpp` and `rmcl_ros/tests/rmcl_gpu_kernels.cpp:12` includes
   `resampling.cuh`.
4. See item 1 above.

### Verified here, so it does not have to be re-argued

- The NVIDIA path still compiles at `4f746de`, including the new `cudaGetDeviceCount` call the
  delta added. Repeated the gfx942 reviewer's method on this host with CUDA 12.8.93:
  `nvcc -std=c++17 -arch=sm_75 -c` on `resampling.cu` and `particle_motion.cu`, and
  `g++ -std=c++17 -fsyntax-only` on `GladiatorResamplerGPU.cpp` and `tests/rmcl_gpu_kernels.cpp`,
  all with the real CUDA headers and the shim's `#else` branch active. All four succeed; only the
  three pre-existing unused-variable warnings appear. `cudaGetDeviceCount`/`cudaSuccess` resolve
  from `<cuda_runtime.h>` via the shim, and cudart reaches the test's link line transitively
  (`rmcl_ros_cuda` -> `rmagine::cuda` -> `${CUDA_LIBRARIES}` = `CUDA::cudart`, rmagine's CUDA
  branch, `src/rmagine_cuda/CMakeLists.txt:143-149`).
- `option(USE_HIP ... ${rmagine_cuda_USE_HIP})` is safe when the variable is undefined: CMake's
  `option()` takes the value argument as optional and defaults to OFF, so a CUDA-only rmagine
  gives `USE_HIP=OFF` and the upstream default is unchanged.
- The shim's `<cstdio>` comment is accurate: `/opt/rocm/include/rocrand/rocrand_mtgp32.h:443`
  calls `printf` and the header includes neither `<cstdio>` nor `<stdio.h>`.
- The macros the shim defines (`cudaError_t`, `cudaSuccess`, `cudaGetDeviceCount`) are
  token-identical to rmagine's own installed shim (`rmagine/util/cuda/cuda_to_hip.h:27,28,32`),
  so a TU that sees both gets a permitted identical redefinition, not a conflict.
- `add_test` really is registered: `ament_cmake_test-extras.cmake:17` calls `enable_testing()`
  when `ament_cmake` is found, so the test's visibility to ctest is not an artefact of this host.
- Fault classes clean, re-checked on the unchanged kernel bodies: no `warpSize`, no `__shfl`, no
  `__ballot`, no literal 32; `simple_stats_kernel` (`resampling.cu:47-79`) initialises all
  `blockSize` LDS slots unconditionally before the loop and runs the full `__syncthreads` tree
  with the barrier outside `if(tid < s)`, so there is no wave-synchronous tail and no
  uninitialised-LDS read on either wave size; `sdata[tid + s]` stays inside the array. Removing
  the NSDMIs is inert on both paths -- the kernel writes both members at `:53-54` before any read,
  and the only other constructions of the type are device-memory (`GladiatorResamplerGPU.cpp:69`)
  or host copies assigned from the device.
- The test's CPU references match the kernels' arithmetic exactly where it asserts equality:
  `particle_move_and_forget_kernel:28` computes `n_meas -= forget_rate * n_meas` in double and
  truncates, and `rmcl_gpu_kernels.cpp:110-113` does the same, so the exact comparison is not
  luck.

### Rulings on the two questions the porter left open

**The residual device-less failure needs nothing further in rmcl.** The 77 path is the right
shape, the abort is entirely inside a dependency's global constructor, the two CTest mechanisms
that look like they would rescue it were measured and do not, and a launcher script that
translates an abort into exit 77 would be rmcl carrying a workaround for rmagine's design. The
real fix is the registered rmagine deferral. Upstream CI cannot hit it (the ROS images carry no
GPU backend, so `RMCL_CUDA` is false and `tests/` is never added). Keep the code and keep the
comment at `rmcl_ros/tests/CMakeLists.txt:11-15` that says plainly why it is not sufficient.

**Leave the optional rmcl-package cross-statistics test out of this port.** `rmcl-cuda` contains
no HIP-compiled code -- `CorrespondencesCUDA.cpp` is plain C++ over rmagine's device functions --
so a test there would mostly re-assert rmagine's kernels, which are already validated on four
architectures, while enlarging the diff into a package this port does not otherwise touch.
`surface.json` already covers `library:rmcl-cuda` with build-and-load evidence. It would be a
genuine gift to upstream and it is cheap (the porter has it written and passing: 1740 of 3001
correspondences, mean error 4.8e-08), so if the person preparing the PR wants the `rmcl` package
to carry evidence of its own, add it as its own commit -- but that is a PR-scope decision, not a
port defect, and nothing here waits on it.

### Addendum, same review run: this verdict was reached while gfx942 was working the same branch

While this review ran, linux-gfx942 was running its own round on rmcl and pushed a hold
(`on_hold`, 02:45) citing a work-lock serialization failure, registered as
`moatlib-port-lock-merge-race` in `data/deferred.json`. This reviewer held the lock from 02:34,
so the lock did not do its job. Two consequences for whoever picks this up:

- The first `commit-project` of this review merged that hold away, because the review's status
  write was based on the pre-hold record. It has been re-applied with `moatlib.py set-hold rmcl
  on` and the original reason preserved. Nothing else in the other host's record was touched.
- **The skip-path ruling above is contested and is now a person's call, not mine.** gfx942
  reports having measured a working launcher pattern (about 60 lines) that does translate the
  device-less abort into a CTest skip, preserved unpushed on a branch `gfx942-skip-launcher` in
  that host's `projects/rmcl/src` (commits `9eaf4f4`, `3544424`). Those commits are not in this
  clone and were not reviewed here, so the ruling above -- keep the 77 path, no launcher upstream
  -- is an argument about what belongs in an upstream tree, made without having read the
  alternative. It is not evidence that the launcher does not work. Whoever rules should read that
  branch first.

The review verdict itself stands on `4f746de`, which is still `head_sha`: the two problems above
are what the next porter round should fix, and neither touches the fork.

## Ruling 2026-08-13 (person): adopt the launcher skip path

Jeff Daily ruled the contested skip-path question: adopt the gfx942 launcher
pattern (probe linking hip::host only + run_gpu_test.cmake + SKIP_REGULAR_EXPRESSION),
preserved on branch gfx942-skip-launcher in projects/rmcl/src (9eaf4f4, 3544424).
It replaces the in-main exit-77 mechanism as the device-less skip. The
rmagine-lazy-default-gpu-context deferral stands as the better fix either way.

## Fix round 2026-08-13 (linux-gfx942, MI300X) -- launcher skip path adopted, head `2cf0e8a`

Executed the person's ruling above plus the two items from the gfx1100 second review. Fork head
`4f746de` -> `2cf0e8a`, still three commits.

### Fork: how the delta was taken

The launcher pieces were taken as a delta onto the reviewed branch, not by resetting to
`gfx942-skip-launcher`: `4f746de` carries gfx1100's reworded commit messages (the corrected nvcc
claim in `493d0f6` above all), which had to survive. Method: with the tree at `4f746de`,
`git checkout gfx942-skip-launcher -- rmcl_ros/tests/ rmcl_ros/include/rmcl_ros/util/cuda_to_hip.h`,
one hand edit to `rmcl_ros/CMakeLists.txt` for the `BUILD_TESTING` comment, then
`git reset --soft f3d62d0` and one fresh commit. The resulting tree is identical to
`gfx942-skip-launcher` except for the `PUBLIC USE_HIP` comment, where gfx1100's wording was kept
because the reviewer had already checked it as accurate (review item 3, resolved).

`4f746de` was replaced rather than reverted-and-re-added. Checked first that this was allowed:
`pr-state rmcl` is `none` and both platform records carry `validated_sha: null`, so nothing was
orphaned. Appending a fourth commit would have left an upstream reader watching the skip mechanism
be built and then replaced within one branch, for no gain.

Files at `2cf0e8a` relative to `f3d62d0` (73 insertions, 4 deletions):

- `rmcl_ros/tests/gpu_device_probe.cpp` (new, 22 lines) -- links the GPU runtime only
  (`hip::host` under `USE_HIP`, `${CUDA_LIBRARIES}` otherwise), never `rmcl_ros_cuda`, so
  rmagine's load-time context construction cannot abort it. Prints `no GPU device available` and
  exits 77 on zero devices.
- `rmcl_ros/tests/run_gpu_test.cmake` (new, 15 lines) -- the registered ctest command: run the
  probe, on nonzero `message(FATAL_ERROR "no GPU device available, skipping")`, else run the test
  and forward its status.
- `rmcl_ros/tests/CMakeLists.txt` -- probe target with the `USE_HIP`/CUDA branches, `add_test` via
  `${CMAKE_COMMAND} -DPROBE=... -DTEST=... -P run_gpu_test.cmake`, and
  `SKIP_REGULAR_EXPRESSION "no GPU device available"` in place of `SKIP_RETURN_CODE 77`. The
  launcher exits normally, so the regex is what fires; the return code never reaches ctest.
- `rmcl_ros/tests/rmcl_gpu_kernels.cpp` -- the in-`main` exit-77 check and the shim include are
  **dropped**, matching the preserved branch. Reason recorded in the commit: a process that
  reaches `main` got there only because the probe found a device, and a device-less machine never
  reaches `main`, so the check could no longer fire. Dead code, not defence in depth.
- `rmcl_ros/include/rmcl_ros/util/cuda_to_hip.h` -- keeps `cudaSuccess` and `cudaGetDeviceCount`
  (the probe uses them) and drops `cudaError_t`, which no longer has a user.

### Measured on this host (the load-bearing evidence for the ruling)

Clean full workspace build, `-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DBUILD_TESTING=ON`,
`Summary: 3 packages finished [4min 54s]`, no errors. `librmcl_ros_cuda.so` carries
`hipv4-amdgcn-amd-amdhsa--gfx942`.

```
ctest --test-dir build/rmcl_ros --output-on-failure
1: particle_move_and_forget: OK (4099 particles)
1: compute_stats: OK (sum=2051.63 max=1)
1: gladiator_resample: OK (2043 of 4099 particles replaced, noise mean=-0.00274891 sd=1.01771)
1/1 Test #1: rmcl_gpu_kernels .................   Passed    0.65 sec

HIP_VISIBLE_DEVICES=-1 ctest --test-dir build/rmcl_ros
1/1 Test #1: rmcl_gpu_kernels .................***Skipped   0.23 sec
The following tests did not run:
	  1 - rmcl_gpu_kernels (Skipped)
```

Kernel results are bit-for-bit what the two earlier rounds recorded on gfx942 and gfx1100. All
four kernels dispatch (`AMD_LOG_LEVEL=3`): `rmcl::init_curand_kernel`,
`void rmcl::simple_stats_kernel<512u>`, `rmcl::particle_move_and_forget_kernel`,
`rmcl::gladiator_resample_kernel`.

Two direction checks the earlier rounds did not record:

- the probe alone: exit 0 with a device, `no GPU device available` and exit 77 under
  `HIP_VISIBLE_DEVICES=-1`;
- the launcher does not swallow failures -- with a device present and `/bin/false` in place of the
  test, it stops with `/bin/false exited with 1` and ctest reports a failure, not a skip, because
  that text does not match the skip regex.

`gfx942-skip-launcher` was deleted from `projects/rmcl/src` once its content was adopted; the
`pre-fixup-backup` branch at `a8e6f88` was left alone.

### The two MOAT-side items from the gfx1100 second review

1. **Blocking, doc-only.** `references/strategy-a-cmake.md` no longer claims a link library's
   compile options apply to every source regardless of language. Re-read the mechanism here on
   ROCm 7.14 rather than taking the review's citation: `hip_add_interface_compile_flags`
   (`hip-config.cmake:75-79`) appends
   `INTERFACE_COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:...>"`, and `hip-config-amd.cmake:143,150`
   feeds it `-x hip` and `--offload-arch=<gfx>`, so the flags hit exactly the target's CXX sources
   and never its HIP-language ones. The advice is unchanged.
   **One correction to the review's own wording:** the link half is not unconditionally ungated.
   `hip_add_interface_link_flags` (`hip-config.cmake:81-90`) wraps `--hip-link` and
   `--offload-arch` in `$<$<LINK_LANGUAGE:CXX>:...>` on CMake 3.20+ and only appends them bare
   below that, on this ROCm. The review read the call sites (`hip-config-amd.cmake:162,168` in
   ROCm 7.2.3) and not the helper's version branch. The lesson says the gating is version-pair
   dependent and to assume `hip::device` reaches a consumer's link line either way, which is the
   safe reading under both.
2. `surface.json` `test:rmcl_gpu_tests` `where` no longer names `rmcl/tests`, which does not
   exist on the branch. Its `covered` evidence now also names the probe, the launcher and the
   skip measurement, since the registration shape changed with this round.

## Review 2026-08-13 (linux-gfx942, third round) -- changes requested

Reviewed `git diff 8892cf4...2cf0e8a` on `AMD-Ecosystem/rmcl` `moat-port` (11 files, +465/-12),
with the weight on the rewritten third commit `2cf0e8a` (+73/-4 over `f3d62d0`). `493d0f6` and
`f3d62d0` are the same git objects the second review read -- same SHAs, and `2cf0e8a` and the
replaced `4f746de` share the parent `f3d62d0` -- so those two commits are byte-identical by
content addressing and were not re-reviewed line by line. Working tree clean, `jargon.py --port
rmcl` clean, `check.py` all gates ok, `prose.py` clean on all three commit bodies, three `[ROCm]`
titles at 45/46/50 chars, AI-assistance disclosure and Test Plan in each body, no
`Co-Authored-By`, no non-ASCII in messages or in the added lines, no internal account references.

### Problems

1. **The launcher reports a broken probe as "no GPU device available", so a GPU test that never
   ran looks like a machine without a GPU.** `rmcl_ros/tests/run_gpu_test.cmake:7` is
   `if(NOT probe_result EQUAL 0)`, which sends every nonzero probe result down the skip path,
   while `rmcl_ros/tests/gpu_device_probe.cpp:6-7` documents the contract as "0 means at least
   one device, 77 means none". The two disagree, and the code loses the distinction the launcher
   exists to make. Measured on this host against the built probe and script:

   ```
   cmake -DPROBE=/bin/false      -DTEST=/bin/true -P run_gpu_test.cmake  -> "no GPU device available, skipping"
   cmake -DPROBE=/nonexistent    -DTEST=/bin/true -P run_gpu_test.cmake  -> "no GPU device available, skipping"
   ```

   Both would be `***Skipped` under the test's `SKIP_REGULAR_EXPRESSION`. A probe that segfaults
   in `hipGetDeviceCount`, or that cannot load `libamdhip64.so` after the binary is relocated out
   of the build tree (the build-tree RUNPATH is what hides this today), is exactly the case where
   the GPU test silently stops running on a machine that has a GPU. Fix in
   `run_gpu_test.cmake:7-9`: skip on 77 and fail on anything else, e.g.

   ```cmake
   if(probe_result EQUAL 77)
       message(FATAL_ERROR "no GPU device available, skipping")
   elseif(NOT probe_result EQUAL 0)
       message(FATAL_ERROR "${PROBE} failed with ${probe_result}")
   endif()
   ```

   The alternative resolution -- keep any-nonzero and delete the "77 means none" sentence from the
   probe -- is worse: it throws away information the probe already produces. The test-side half of
   the launcher is right and does not need touching: with a device present, `/bin/false` in place
   of the test gives `/bin/false exited with 1` and an aborting test gives `exited with Subprocess
   aborted`, neither of which matches the skip regex (both measured here).

2. **The promoted skip lesson recommends a mechanism this port measured to be dead and then
   deleted, and cites this port while doing it.**
   `.claude/skills/cuda-to-rocm/references/validation.md:101-111` establishes that a dependency
   with a namespace-scope GPU context aborts before `main` on a device-less host, and then
   concludes at :108 "So keep the 77 path". In that situation the in-`main` device count check can
   never execute -- which is precisely the argument `2cf0e8a` makes for removing it ("a machine
   without one never reaches main, so the check could no longer fire"), reproduced here: running
   `rmcl_ros_tests_gpu_kernels` directly under `HIP_VISIBLE_DEVICES=-1` dies with
   `terminate called ... Error calling cudaGetDeviceProperties` / `Aborted (core dumped)`, never
   reaching `main`. So the paragraph recommends dead code, and a reader who follows it to the
   named source project (rmcl) finds no 77 path and no `SKIP_RETURN_CODE` on the branch. The
   launcher paragraph at :113-123 still frames itself as an optional extra to "weigh against just
   documenting the gap", which is the position a person overruled in the ruling above. Rewrite
   :96-123 so the split is by cause and not by preference: the in-`main` 77 check plus
   `SKIP_RETURN_CODE 77` is the mechanism where nothing aborts before `main`; where a dependency
   constructs a GPU context at load, that check cannot fire and the probe launcher is the
   mechanism, which is what rmcl ships. Keep the two measured CTest dead ends and the lazy-context
   deferral, both still correct.

### Verified here, so it does not have to be re-argued

- The probe does not pull rmagine in. `readelf -d` on `rmcl_ros_tests_gpu_probe` gives exactly
  two NEEDED entries, `libamdhip64.so.7` and `libc.so.6`; `librmagine-cuda.so.2` and
  `librocrand.so.1` appear only in `rmcl_ros_tests_gpu_kernels`. That is the load-bearing claim
  of the whole design and it holds.
- The acceptance measurements reproduce on this host (MI300X, gfx942, ROCm 7.14, incremental
  colcon build of the three packages at `2cf0e8a`, `Summary: 3 packages finished [5.85s]`):
  `ctest` -> `1/1 Test #1: rmcl_gpu_kernels ... Passed 0.64 sec`;
  `HIP_VISIBLE_DEVICES=-1 ctest` -> `***Skipped 0.23 sec`, `1 - rmcl_gpu_kernels (Skipped)`;
  probe alone -> rc 0 with a device, `no GPU device available` and rc 77 without.
- The CUDA branch of the probe is sound and was compiled, not reasoned about.
  `g++ -std=c++17 -Wall -Wextra -Wpedantic -I rmcl_ros/include -I <cuda-12.8>/targets/x86_64-linux/include
  -c rmcl_ros/tests/gpu_device_probe.cpp` succeeds with no diagnostics through the shim's `#else`
  branch (`cuda_runtime.h`, `curand.h`, `curand_kernel.h` under a plain host compiler), links
  against `-lcudart`, and on this NVIDIA-free host prints `no GPU device available` and exits 77.
  Its CMake branch is also safe when `CUDAToolkit` was the finder: `CUDA_INCLUDE_DIRS` is `""`
  there (`rmcl_ros/CMakeLists.txt:198`) and `target_include_directories(<t> PRIVATE)` with an
  empty expansion is legal CMake (checked), while `CUDA_LIBRARIES` is `CUDA::cudart`, whose
  imported target is visible in the `tests/` subdirectory and carries the include dirs.
- The shim trim is clean: `cudaError_t` has no user anywhere in `rmcl_ros/` or `rmcl/`, and the
  only surviving `cudaMalloc`/`cudaMemcpyAsync`/`cudaMallocHost` uses are in
  `rmcl_ros/src/rmcl/optix/eval_program_groups.cpp`, which no ROCm configuration builds.
  `cudaSuccess` and `cudaGetDeviceCount` have exactly one user each, the probe.
- The ruling was implemented faithfully. `git diff 3544424 HEAD` is one hunk: the `PUBLIC USE_HIP`
  comment at `rmcl_ros/CMakeLists.txt:233-235`, where gfx1100's already-reviewed wording was kept.
  Everything else in the adopted tree is identical to the preserved `gfx942-skip-launcher` tip.
- gfx1100 review item 1 is resolved and its mechanism re-derived here rather than taken from the
  porter: on this host's ROCm 7.14 (`hip-config-version.cmake` `PACKAGE_VERSION "7.14.60850"`),
  `hip_add_interface_compile_flags` (`hip-config.cmake:75-79`) appends
  `INTERFACE_COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:...>"` and is fed `-x hip` and
  `--offload-arch=` at `hip-config-amd.cmake:143,150`; `hip_add_interface_link_flags`
  (`hip-config.cmake:81-90`) is the version branch the porter describes -- bare below CMake 3.20,
  `$<$<LINK_LANGUAGE:CXX>:...>` at or above -- fed `--hip-link` and `--offload-arch=` at
  `hip-config-amd.cmake:146,152`. The lesson's remaining claim that
  `$<COMPILE_LANGUAGE:HIP>` cannot be used in `target_link_libraries` is also true: CMake refuses
  with "may only be used to specify include directories, compile definitions, compile options"
  (checked with a four-line project).
- gfx1100 review item 2 is resolved: `surface.json` `test:rmcl_gpu_tests` `where` is
  "added by this port (rmcl_ros/tests)" and the `covered` evidence names the probe, the launcher
  and the skip measurement.
- Fault classes re-checked on the unchanged kernels: no `warpSize`, `__shfl`, `__ballot`,
  `__activemask`, literal 32, `/32` or `%32` anywhere under `rmcl_ros/src/rmcl/`,
  `rmcl_ros/include/rmcl_ros/rmcl/` or `rmcl_ros/tests/`; `simple_stats_kernel`
  (`resampling.cu:40-79`) writes all `blockSize` LDS slots before the loop and runs the full
  `__syncthreads` tree with the barrier outside `if(tid < s)`, so it is correct on wave32 and
  wave64 alike.

## Fix round 2026-08-13 (linux-gfx942, MI300X) -- launcher distinguishes 77 from failure, head `2b7f439`

Both items of the third review applied. Fork head `2cf0e8a` -> `2b7f439`, still three commits:
the launcher commit is the branch tip, `pr-state rmcl` is `none` and both platforms carry
`validated_sha: null`, so it was amended rather than followed by a fixup, on the same reasoning
as the previous round. `f3d62d0` and `493d0f6` are untouched.

### Fork: `run_gpu_test.cmake` now splits the probe's exit codes

`if(NOT probe_result EQUAL 0)` became

```cmake
if(probe_result EQUAL 77)
    message(FATAL_ERROR "no GPU device available, skipping")
elseif(NOT probe_result EQUAL 0)
    message(FATAL_ERROR "device probe exited with ${probe_result}")
endif()
```

plus two sentences in the file's header comment saying that only 77 means no device. The script's
contract now matches `gpu_device_probe.cpp:6-7`, which is what the review found disagreeing.

Note on the non-numeric case: when the probe cannot be executed at all, `execute_process` puts a
string in `RESULT_VARIABLE`, not a number. `EQUAL 77` is false for it and `NOT ... EQUAL 0` is
true, so it lands in the failure branch and the message carries the string. Measured, not
reasoned about (`device probe exited with No such file or directory`).

### The four required behaviours, measured on this host

Clean full workspace build first (`rm -rf build install`, then `agent_space/rmcl/build_rmcl.sh`,
i.e. `-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DBUILD_TESTING=ON`):
`Summary: 3 packages finished [4min 42s]`, no errors.

The registered test, device present and then hidden:

```
ctest --test-dir build/rmcl_ros --output-on-failure
1: particle_move_and_forget: OK (4099 particles)
1: compute_stats: OK (sum=2051.63 max=1)
1: gladiator_resample: OK (2043 of 4099 particles replaced, noise mean=-0.00274891 sd=1.01771)
1/1 Test #1: rmcl_gpu_kernels .................   Passed    0.65 sec

HIP_VISIBLE_DEVICES=-1 ctest --test-dir build/rmcl_ros
1/1 Test #1: rmcl_gpu_kernels .................***Skipped   0.22 sec
1 - rmcl_gpu_kernels (Skipped)
```

Kernel numbers are bit-for-bit those of all earlier rounds on both platforms.

The probe/test substitutions were run through CTest rather than by invoking `cmake -P` by hand,
so that the `SKIP_REGULAR_EXPRESSION` half is exercised too and skip-vs-failure is CTest's own
verdict. `agent_space/rmcl/launcher_check/` registers the shipped `run_gpu_test.cmake` with the
same skip regex and substituted `PROBE`/`TEST` (scratch, not on the fork). With a device present:

```
1/5 Test #1: a_real_device ....................   Passed    0.57 sec
2/5 Test #2: b_probe_false ....................***Failed    0.02 sec
     device probe exited with 1
3/5 Test #3: c_probe_missing ..................***Failed    0.01 sec
     device probe exited with No such file or directory
4/5 Test #4: d_test_false .....................***Failed    0.22 sec
     /bin/false exited with 1
5/5 Test #5: e_probe_crash ....................***Failed   30.16 sec
     device probe exited with Segmentation fault
```

`e_probe_crash` is a probe that raises SIGSEGV on itself, the review's "segfaults in
`hipGetDeviceCount`" case; it is the slow one because CTest waits out the core dump. None of the
four failure texts matches the skip regex, and the same suite under `HIP_VISIBLE_DEVICES=-1`
gives `a_real_device` and `d_test_false` `***Skipped` (the real probe returns 77 before either
test runs) while `b`, `c` and `e` stay `***Failed` -- a broken probe is not excused by the host
also having no device.

Before this change `b_probe_false` and `c_probe_missing` were `***Skipped`, which is exactly the
review's finding.

### MOAT: the skip lesson is now split by cause

`.claude/skills/cuda-to-rocm/references/validation.md` section "A GPU test that skips on a
device-less machine, and when it cannot" no longer reads as a preference between two options.
It now opens by saying the mechanism follows from whether the process reaches `main`, then gives
two labelled cases: in-`main` device count plus `SKIP_RETURN_CODE 77` where nothing aborts
first, and the probe-launcher pattern where a dependency constructs a GPU context at load. The
second case states plainly that the in-`main` check is dead code there, keeps both measured CTest
dead ends (`SKIP_REGULAR_EXPRESSION` not consulted on a signal death, `FIXTURES_SETUP` giving
either a run-anyway or `***Not Run` plus two failures), requires skipping on 77 alone with every
other nonzero result failing with its status, and names rmcl as the project that ships it. The
lazy-context deferral against rmagine closes the section for both cases. Hard-wrapped to match
the rest of the file; `prose.py` flags the whole file for that and always has.

### Checks

`check.py` all gates ok, `jargon.py --port rmcl` clean, `prose.py` clean on the rewritten commit
body, title still 45 chars, working tree clean, pushed with `--force-with-lease`
(`+ 2cf0e8a...2b7f439 moat-port -> moat-port (forced update)`).

## Review 2026-08-13 (linux-gfx942, fourth round) -- passed

Scope: the amended tip `2cf0e8a` -> `2b7f439` on `AMD-Ecosystem/rmcl` `moat-port` (remote tip
confirmed at `2b7f439`) and the `validation.md` lesson restructure. Both items of the third
review are resolved.

### Problems

None.

### Verified here

- The amendment touches one file. `git diff 2cf0e8a 2b7f439` is `rmcl_ros/tests/run_gpu_test.cmake`
  only, +7/-2, plus the commit message. `f3d62d0` and `493d0f6` are the same git objects as in the
  two previous reviews (identical SHAs and trees, and both tips share the parent `f3d62d0`), so
  they were not re-read.
- `run_gpu_test.cmake:10-14` skips on 77 alone and fails with the probe's own status otherwise.
  Acceptance matrix re-measured by this reviewer, through CTest with the shipped script and the
  shipped `SKIP_REGULAR_EXPRESSION`, in a scratch harness written for this review (not the
  porter's), after an incremental colcon build at `2b7f439`
  (`Summary: 3 packages finished [6.07s]`):

  ```
  ctest --test-dir build/rmcl_ros --output-on-failure
  1/1 Test #1: rmcl_gpu_kernels .................   Passed    0.64 sec

  HIP_VISIBLE_DEVICES=-1 ctest --test-dir build/rmcl_ros
  1/1 Test #1: rmcl_gpu_kernels .................***Skipped   0.23 sec

  1/6 Test #1: a_real ...........................   Passed    0.50 sec
  2/6 Test #2: b_probe_false ....................***Failed    0.02 sec  device probe exited with 1
  3/6 Test #3: c_probe_missing ..................***Failed    0.01 sec  device probe exited with No such file or directory
  4/6 Test #4: d_test_false .....................***Failed    0.22 sec  /bin/false exited with 1
  5/6 Test #5: e_probe_crash ....................***Failed   30.15 sec  device probe exited with Segmentation fault
  6/6 Test #6: f_probe_77 .......................***Skipped   0.02 sec
  ```

  The same six under `HIP_VISIBLE_DEVICES=-1` give `a_real` and `d_test_false` `***Skipped` (the
  real probe returns 77 before either test runs) and leave `b`, `c` and `e` `***Failed`. Skip
  fires on 77 and on nothing else, in both directions; the third review's finding 1 is closed.
  Non-numeric `RESULT_VARIABLE` lands in the failure branch as the porter recorded (case `c`).
- The skip regex is matched against the whole captured output, so a probe that printed
  `no GPU device available` and then exited nonzero-other would still be reported `***Skipped`
  (checked with a stub). Not a defect on this branch: `gpu_device_probe.cpp:18-19` prints that
  line only immediately before `return 77`, and grep shows the phrase in exactly three places --
  the probe, the launcher message, and the skip regex. Recorded so the next reader does not
  re-derive it.
- The probe answering 77 for a `cudaGetDeviceCount` error as well as for a zero count
  (`gpu_device_probe.cpp:16`) was considered and left alone: a runtime that cannot enumerate
  devices is the same "no usable device" answer for a test host, and the loader-level failures
  the third review worried about (relocated binary, missing `libamdhip64.so`) never reach `main`
  and so land in the launcher's failure branch, which case `c` demonstrates.
- The promoted lesson at `validation.md:92-127` splits by cause, not by preference: the opening
  says the mechanism follows from whether the process reaches `main` (:96-97), `**Nothing aborts
  before main**` keeps in-`main` 77 plus `SKIP_RETURN_CODE` (:99-102), and `**A dependency
  constructs a GPU context at load time**` states the in-`main` check "is dead code, not defence
  in depth" (:107-108), keeps both measured CTest dead ends (:109-111), requires skipping on "77,
  and on 77 alone" with every other nonzero result failing with its status (:115-120), and names
  rmcl as the project that ships it (:123-124). The "weigh against just documenting the gap"
  framing is gone from the file. The lazy-context deferral closes the section for both cases
  (:126-127). Finding 2 is closed.
- The lesson's load-bearing measured claim was re-measured rather than taken on trust: a test that
  prints the skip phrase and then dies by `SIGABRT` is `***Failed (Subprocess aborted)`, not
  `***Skipped`, while a sibling returning 77 with `SKIP_RETURN_CODE 77` is `(Skipped)`
  (CMake 3.28.3 here).
- Hygiene: working tree clean in `projects/rmcl/src`, `check.py` all gates ok, `jargon.py --port
  rmcl` clean, `prose.py` clean on the amended body, title `[ROCm] Report a skip when no GPU
  device is present` at 50 chars, no `Co-Authored-By` or noreply trailer, no non-ASCII in the
  message or the added lines, no internal account references. The rewritten Test Plan's four
  quoted failure texts all reproduce here.

Verdict: review-passed. GPU validation at `2b7f439` is the validator's next step on both
platforms (`validated_sha` is null on each).

## Validation 2026-08-13 (linux-gfx942, MI300X) -- completed

Fresh clean build of `AMD-Ecosystem/rmcl` `moat-port` at exactly `2b7f439` (tree confirmed
clean, `git status --porcelain` empty, `HEAD` == `head_sha`). This is the first validation on
this platform (`validated_sha` was `null`).

### Environment

- Ubuntu 24.04.4 noble, ROCm 7.14 from the pip `_rocm_sdk_devel` package under
  `/opt/conda/envs/py_3.12/lib/python3.12/site-packages/_rocm_sdk_devel` (no `/opt/rocm` on this
  host).
- GPU: AMD Instinct MI300X, `amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-` (8 devices visible,
  `rocminfo`).
- ROS 2 jazzy at `/opt/ros/jazzy`. `export PATH=/usr/bin:$PATH` before sourcing it, as prior
  rounds recorded (conda python otherwise shadows the apt ROS tooling).

### Build

Dependency, rmagine fork `moat-port`, built fresh at `1213551` (the current fork tip; an ancestor
of the `e7a7b27` this port's first round built against, and confirmed by that round's review to
differ only by a test-comment commit) into a clean staging prefix, Embree left enabled:

```
cmake -S projects/rmagine/src -B agent_space/rmcl_val/rmagine_build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++ -DCMAKE_PREFIX_PATH="$R" \
  -DRMAGINE_OPTIX_DISABLE=ON -DRMAGINE_VULKAN_DISABLE=ON \
  -DRMAGINE_VULKAN_CUDA_INTEROP_DISABLE=ON -DRMAGINE_OUSTER_DISABLE=ON \
  -DRMAGINE_BUILD_TESTS=OFF -DRMAGINE_BUILD_TOOLS=OFF \
  -DCMAKE_INSTALL_PREFIX=agent_space/rmcl_val/rmagine_install
cmake --build agent_space/rmcl_val/rmagine_build -j && cmake --install agent_space/rmcl_val/rmagine_build
```

(`$R` = the `_rocm_sdk_devel` prefix above.) Installed cleanly, no errors.

Full clean colcon build of the three rmcl packages against that install:

```
export PATH=/usr/bin:$PATH
. /opt/ros/jazzy/setup.bash
export CMAKE_PREFIX_PATH="agent_space/rmcl_val/rmagine_install:$R:$CMAKE_PREFIX_PATH"
colcon build --base-paths projects/rmcl/src --packages-select rmcl_msgs rmcl rmcl_ros \
  --build-base agent_space/rmcl_val/build --install-base agent_space/rmcl_val/install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
               -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
               -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++
```

`Summary: 3 packages finished [4min 37s]`, no errors, only the same pre-existing warnings every
prior round recorded (unused parameters/variables in the CPU resamplers, `resampling.cu`'s three,
the deprecated `point_cloud_conversion.hpp` `#warning`). `librmcl_ros_cuda.so` carries
`amdgcn-amd-amdhsa--gfx942` device code (`strings | grep amdgcn`). Non-GPU regression check: all
expected CPU targets are present in the install tree --
`rmcl_ros/lib/{librmcl_ros.so,librmcl_embree_ros.so,libmicp_localization.so,libmap_segmentation.so,
libconv_pc2_to_scan.so,libconv_pc2_to_o1dn.so,libconv_scan_to_scan.so,
libo1dn_map_segmentation_embree.so,libscan_map_segmentation_embree.so}` and
`rmcl/lib/{librmcl.so,librmcl-cuda.so,librmcl-embree.so}` all built.

### Test results (both runs bit-identical)

```
export PATH=/usr/bin:$PATH; . /opt/ros/jazzy/setup.bash
ctest --test-dir agent_space/rmcl_val/build/rmcl_ros -V
```

Run 1:
```
1: particle_move_and_forget: OK (4099 particles)
1: compute_stats: OK (sum=2051.63 max=1)
1: gladiator_resample: OK (2043 of 4099 particles replaced, noise mean=-0.00274891 sd=1.01771)
1/1 Test #1: rmcl_gpu_kernels .................   Passed    0.43 sec
```

Run 2:
```
1: particle_move_and_forget: OK (4099 particles)
1: compute_stats: OK (sum=2051.63 max=1)
1: gladiator_resample: OK (2043 of 4099 particles replaced, noise mean=-0.00274891 sd=1.01771)
1/1 Test #1: rmcl_gpu_kernels .................   Passed    0.65 sec
```

Identical to every prior round on gfx942 and gfx1100 (bit-for-bit reduction sum/max, resampler
replacement count and noise statistics). 1/1 test passes, 0 non-GPU regressions.

### Skip matrix (launcher probe)

```
HIP_VISIBLE_DEVICES=-1 ctest --test-dir agent_space/rmcl_val/build/rmcl_ros --output-on-failure
1/1 Test #1: rmcl_gpu_kernels .................***Skipped   0.23 sec
The following tests did not run:
	  1 - rmcl_gpu_kernels (Skipped)
```

Device present -> Passed; device hidden -> Skipped (not Failed, not silently absent). Matches the
recorded launcher contract (skip on probe exit 77 alone).

### Kernel dispatch confirmation

```
AMD_LOG_LEVEL=3 agent_space/rmcl_val/build/rmcl_ros/tests/rmcl_ros_tests_gpu_kernels 2>&1 | grep ShaderName
```

All four kernels reach the device:
```
ShaderName : rmcl::particle_move_and_forget_kernel(rmagine::Transform_<float>*, rmcl::ParticleAttributes*, rmagine::Transform_<float>, double, unsigned int)
ShaderName : void rmcl::simple_stats_kernel<512u>(rmagine::Transform_<float> const*, rmcl::ParticleAttributes const*, unsigned int, rmcl::SimpleLikelihoodStats*)
ShaderName : rmcl::init_curand_kernel(hiprandState*, unsigned int)
ShaderName : rmcl::gladiator_resample_kernel(rmagine::Transform_<float> const*, rmcl::ParticleAttributes const*, rmcl::SimpleLikelihoodStats const*, hiprandState*, rmagine::Transform_<float>*, rmcl::ParticleAttributes*, unsigned int, rmcl::GladiatorResamplerConfig)
```

### CUDA no-regression gate -- passed at `2b7f439`

The only delta since the last recorded CUDA compile (at `2cf0e8a`, which the fourth review's diff
`2cf0e8a...2b7f439` shows to be `rmcl_ros/tests/run_gpu_test.cmake` only -- a CMake launcher script
consumed at test-run time, not compiled by nvcc or a host compiler) is not a C++/CUDA source
change. Recompiled the full recorded set anyway against the real CUDA 12.8 toolkit at this head,
per the dispatch instructions, rather than relying on that inference alone:

```
NVCC=/opt/conda/envs/cuda-12.8/bin/nvcc   (release 12.8, V12.8.93)
CUDAINC=/opt/conda/envs/cuda-12.8/targets/x86_64-linux/include
INC="-Iprojects/rmcl/src/rmcl_ros/include -Iprojects/rmcl/src/rmcl/include \
     -Iagent_space/rmcl_val/install/rmcl_msgs/include/rmcl_msgs \
     -Iprojects/rmagine/src/src/rmagine_core/include -Iprojects/rmagine/src/src/rmagine_cuda/include \
     -I$CUDAINC $(for d in /opt/ros/jazzy/include/*; do echo -I$d; done)"
$NVCC -std=c++17 -arch=sm_75 -c rmcl_ros/src/rmcl/resampling.cu      $INC   # ok
$NVCC -std=c++17 -arch=sm_75 -c rmcl_ros/src/rmcl/particle_motion.cu $INC   # ok
g++   -std=c++17 -c rmcl_ros/src/rmcl/GladiatorResamplerGPU.cpp      $INC   # ok
g++   -std=c++17 -c rmcl_ros/tests/rmcl_gpu_kernels.cpp              $INC   # ok
g++   -std=c++17 -c rmcl_ros/tests/gpu_device_probe.cpp              $INC   # ok
```

All five succeed. Only the same three pre-existing unused-variable warnings on `resampling.cu`
(`random_flt`, `L_max`, `L_sum`) appear, exactly as every prior round recorded; no diagnostic on
the shim's `#else` (real CUDA header) branch. This is the CUDA gate for `head_sha 2b7f439`; it is
recorded here so a Windows platform validating this head skips it.

### Jargon and documentation

`python3 utils/jargon.py --port rmcl` -> `jargon: clean`. README `Installation` section documents
the ROCm build in-place (`colcon build --cmake-args -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942`),
next to the CUDA instructions, matching the project's house style.

### Integrity

`git -C projects/rmcl/src status --porcelain` empty at `2b7f439` throughout; no source or build
file was modified to make this validation pass. The rmagine dependency clone
(`projects/rmagine/src`) was checked out to its fork tip `1213551` to build the dependency, then
restored to its prior branch position (`moat-port` at `e7a7b27`, unchanged, unpushed) before
finishing.

### Verdict

Real-GPU pass on linux-gfx942 (MI300X): build clean, 1/1 test passes twice with bit-identical
kernel results, device-less skip matrix correct, all four kernels confirmed dispatched, CUDA
no-regression gate passes at this head, no non-GPU regression, jargon and documentation clean.
`validated_sha` -> `2b7f439c1c7da079b36aae82f95f43f84fbcd932`. This platform carries the `wave64`
gate.

## Validation 2026-08-19 (linux-gfx1100, Radeon Pro W7800) -- completed

First validation on this platform (`validated_sha` was `null`; the platform record still carried
`last_agent: reviewer` and `started_at` from the 2026-08-13 fix round, which was porter/reviewer
work on this arch, not a prior validation attempt). Fresh checkout of `AMD-Ecosystem/rmcl`
`moat-port` reset to exactly `head_sha`, confirmed with `git rev-parse HEAD`:
`2b7f439c1c7da079b36aae82f95f43f84fbcd932`, `git status --porcelain` empty throughout.

### Environment

- Ubuntu 24.04 noble, ROCm 7.2.3 at `/opt/rocm`, CMake 3.31.6, ROS 2 jazzy at `/opt/ros/jazzy`.
- GPU: AMD Radeon Pro W7800 48GB, `amdgcn-amd-amdhsa--gfx1100` (4 devices visible, `rocminfo`).
- `export PATH=/usr/bin:$PATH` before sourcing ROS (conda python is first on `PATH` on this host
  too; matches every prior round's note).

### Dependency

rmagine fork `moat-port` at `1213551f14f64c92c08048f377034b1ee362659d`, matching
`projects/rmagine/status.json`'s `validated_sha` for this platform (`completed`,
2026-08-13). The checkout in `/var/lib/jenkins/moat-worktrees/rmagine/projects/rmagine/src` was
already at this commit with a clean tree, and its install in
`agent_space/rmcl_val`-adjacent `agent_space/deps/rmagine_install` (built by the prior fix round
on this host, same commit, Embree enabled) was reused as-is rather than rebuilt, since the
source tree it was built from is byte-identical to the tip this round checked out (same sha,
clean working tree, nothing to rebuild).

### Build

```
export PATH=/usr/bin:$PATH
. /opt/ros/jazzy/setup.bash
export CMAKE_PREFIX_PATH="agent_space/deps/rmagine_install:/opt/rocm:$CMAKE_PREFIX_PATH"
colcon build --base-paths projects/rmcl/src --packages-select rmcl_msgs rmcl rmcl_ros \
  --build-base agent_space/rmcl_val/build --install-base agent_space/rmcl_val/install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
               -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
               -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
```

`Summary: 3 packages finished [50.8s]`, no errors, only the same pre-existing warnings every
prior round recorded (unused parameters/variables in the CPU resamplers, `resampling.cu`'s three,
the deprecated `point_cloud_conversion.hpp` `#warning`). `librmcl_ros_cuda.so` carries
`hipv4-amdgcn-amd-amdhsa--gfx1100` device code. All expected CPU targets built:
`rmcl_ros/lib/{librmcl_ros.so,librmcl_embree_ros.so,libmicp_localization.so,libmap_segmentation.so,
libconv_pc2_to_scan.so,libconv_pc2_to_o1dn.so,libconv_scan_to_scan.so,
libo1dn_map_segmentation_embree.so,libscan_map_segmentation_embree.so}`.

### Test results (two runs, bit-identical, and identical to every prior gfx942/gfx1100 round)

```
ctest --test-dir agent_space/rmcl_val/build/rmcl_ros -V
1: particle_move_and_forget: OK (4099 particles)
1: compute_stats: OK (sum=2051.63 max=1)
1: gladiator_resample: OK (2043 of 4099 particles replaced, noise mean=-0.00274891 sd=1.01771)
1/1 Test #1: rmcl_gpu_kernels .................   Passed    0.37 sec
100% tests passed, 0 tests failed out of 1
```

Second run: `Passed 0.37 sec`, same kernel output. 1/1 test passes, 0 non-GPU regressions.

### Skip matrix (launcher probe)

```
HIP_VISIBLE_DEVICES=-1 ctest --test-dir agent_space/rmcl_val/build/rmcl_ros --output-on-failure
1/1 Test #1: rmcl_gpu_kernels .................***Skipped   0.11 sec
The following tests did not run:
	  1 - rmcl_gpu_kernels (Skipped)
```

Device present -> Passed; device hidden -> Skipped. Matches the recorded launcher contract.

### Kernel dispatch confirmation

```
AMD_LOG_LEVEL=3 agent_space/rmcl_val/build/rmcl_ros/tests/rmcl_ros_tests_gpu_kernels 2>&1 | grep ShaderName
```

All four kernels reach the device: `rmcl::particle_move_and_forget_kernel`,
`void rmcl::simple_stats_kernel<512u>`, `rmcl::init_curand_kernel`,
`rmcl::gladiator_resample_kernel`.

### CUDA no-regression gate -- already recorded at this head, not re-run

The gfx942 validation above recorded the CUDA gate passing at this exact `head_sha`
(`2b7f439`, "CUDA no-regression gate -- passed at `2b7f439`"). Per the validator's dispatch
instructions this gate tests the code, not the arch, and runs once per `head_sha`; skipped here.

### Jargon and documentation

`python3 utils/jargon.py --port rmcl` -> `jargon: clean`. README `Installation` section documents
the ROCm build in place (`colcon build --cmake-args -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942`,
with `CMAKE_HIP_ARCHITECTURES` noted as settable per target GPU), next to the CUDA instructions.

### Integrity

`git -C projects/rmcl/src status --porcelain` empty at `2b7f439` throughout; no source or build
file was modified to make this validation pass. `projects/rmagine/src` (in this host's separate
`rmagine` worktree) stayed on `moat-port` at `1213551` throughout, clean, untouched.

### Verdict

Real-GPU pass on linux-gfx1100 (Radeon Pro W7800): build clean, 1/1 test passes twice with
bit-identical kernel results (matching every prior gfx942/gfx1100 round exactly), device-less
skip matrix correct, all four kernels confirmed dispatched, CUDA no-regression gate already
recorded at this head, no non-GPU regression, jargon and documentation clean. `validated_sha` ->
`2b7f439c1c7da079b36aae82f95f43f84fbcd932`. This platform carries the `wave32` gate.

## ROS 2 availability and AMD support on Windows (2026-08-20, maintainer question)

Jeff asked whether there is a ROS 2 package that supports AMD hardware, in the context of the
windows-gfx1151 block. Two separate questions are tangled in that, and separating them changes
the answer.

**1. ROS 2 does not "support" GPU vendors -- it has no GPU layer to support them with.** ROS 2
is middleware: discovery, transport, executors, build tooling. It has no GPU abstraction that
could be AMD-enabled or NVIDIA-enabled. rmcl's AMD support IS the rmagine HIP port, already
completed and validated. There is no missing "ROS 2 AMD package" standing between us and a
Windows validation. For completeness: REP 2008 (ROS 2 Hardware Acceleration Architecture)
sketches an `ament_rocm` extension providing a `rocm_acceleration_kernel` macro, and the ROS 2
Hardware Acceleration Working Group lists collaboration with AMD -- but that is a build-system
abstraction over vendor toolchains, illustrative rather than shipped, and rmcl does not depend
on it. Adopting it would be new scope, not a fix for this gate.

**2. ROS 2 on Windows exists and is Tier 1, but the details matter.** Per REP 2000, Jazzy,
Kilted and Rolling all list Windows at `Tier 1`, for `Windows 10 (VS2019)` on amd64. So Windows
is a supported ROS 2 platform in principle. Two frictions against this specific host:
- The supported target is Windows 10 + VS2019 (MSVC). This host is Windows 11 and the HIP
  toolchain is TheRock clang-cl. clang-cl targets the MSVC ABI, so ROS 2's VS2019-built
  binaries should link against clang-cl-built objects -- but that combination is untested here
  and is an assumption, not a verified fact.
- ros2/ros2#1675 reports the Jazzy Windows binary broken since Patch 4 over a Python version
  and path problem (Patch 3 and earlier work). Whichever distro is chosen needs checking against
  that issue rather than assuming the newest patch is good.

**Conclusion.** Nothing about AMD hardware blocks rmcl on Windows. The block is provisioning,
and it is a real project rather than a quick install: ROS 2 Windows binaries at a patch level
that works, plus colcon/ament, plus a Windows rmagine install built WITH Embree (this host's
rmagine validation deliberately ran `RMAGINE_EMBREE_DISABLE=ON`, so the existing artifacts do
not satisfy rmcl). The suggested `windows` waiver stands as the alternative to doing that work;
it remains the weaker kind of waiver, because nobody has shown rmcl CANNOT work on Windows.

Sources: REP 2000 (reps.openrobotics.org/rep-2000/), ROS 2 Kilted Windows binary install docs,
ros2/ros2 issue 1675, REP 2008 (ros.org/reps/rep-2008.html).
