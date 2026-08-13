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
