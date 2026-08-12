# rmcl porting plan

## Project

- Name: `rmcl`
- Upstream: https://github.com/uos/rmcl (default branch `main`, planned against `8892cf4`)
- Fork: https://github.com/AMD-Ecosystem/rmcl (port branch `moat-port`, not yet created)
- Licence: BSD-3-Clause, cleared at intake
- Hard dependency: `rmagine` (MOAT project, stage `review-passed`, fork HEAD `4223818`)

rmcl is a three-package ROS 2 workspace: `rmcl` (plain CMake library, no ROS), `rmcl_msgs`
(message generation) and `rmcl_ros` (ament package with the nodes). All GPU work is expressed
through rmagine, which rmcl consumes with `find_package(rmagine ...)` -- it is never vendored.

## Existing AMD support

None in rmcl itself. Intake enumerated all 27 upstream forks, the ROCm organisation, and grepped
the docs for `amd|rocm|hip|gfx[0-9]` with zero hits; `AMD-Ecosystem/rmcl` is our own unmodified
mirror with no `moat-port` branch. The README's only hardware framing is NVIDIA (OptiX) and CPU
(Embree).

The finer judgement here is not about rmcl but about what already exists one level down. Our own
rmagine port is authoritative for this stack and is the thing to build on:

- `rmagine::cuda` (Stage 1) is a Strategy-A HIP port validated on gfx90a, gfx1100, gfx1101 and
  gfx1201. Its target name and component name are unchanged under `USE_HIP`, so rmcl's existing
  `if(TARGET rmagine::cuda)` gates light up with no rmcl CMake change.
- `rmagine_hiprt` (Stage 2) is a build-tree-only proof of concept. It is NOT the OptiX
  replacement rmcl would need; see "OptiX and why it is not in this port".

Upstream merges platform support rather than linking forks (Embree, OptiX and Vulkan backends all
live in-tree), so an upstream PR is the right vehicle. The PR will however depend on rmagine's
ROCm support existing upstream, which it does not yet. Sequencing note for whoever publishes:
rmcl's change is inert on an upstream rmagine, because `USE_HIP` there has no meaning. This is a
follow-the-dependency PR and should be offered after (or alongside) the rmagine one.

## Build classification: cmake -> Strategy A

Evidence:

- No torch anywhere: no `find_package(Torch)`, no `setup.py`, no `pyproject.toml`, no
  `CUDAExtension`. The repository has three `CMakeLists.txt` and nothing else that builds.
- `rmcl/package.xml` declares `<build_type>cmake</build_type>`; `rmcl_ros/package.xml` declares
  `<build_type>ament_cmake</build_type>` (`rmcl_ros/CMakeLists.txt:42` `find_package(ament_cmake
  REQUIRED)`, `:788` `ament_package()`). ament_cmake is CMake with ROS conventions layered on; the
  language handling is ordinary CMake.
- The CUDA language is enabled the ordinary way: `rmcl/CMakeLists.txt:217-242` and
  `rmcl_ros/CMakeLists.txt:175-201` both run `include(CheckLanguage)` / `check_language(CUDA)` /
  `enable_language(CUDA)`, and `.cu` sources are listed directly in `add_library`
  (`rmcl_ros/CMakeLists.txt:211-212`).

`status.json` `ext_type` was already `cmake` and is confirmed.

Strategy A: one small compat header plus `enable_language(HIP)` and
`set_source_files_properties(... LANGUAGE HIP)`, all inside a new `USE_HIP` branch, mirroring
exactly what rmagine's port did. Sources keep their CUDA spelling; the NVIDIA path is untouched.

Do NOT reuse rmagine's installed `rmagine/util/cuda/cuda_to_hip.h` from rmcl. It is installed and
it would work, but it exists only on our rmagine fork, so an rmcl PR that includes it cannot be
evaluated by an upstream maintainer. rmcl gets its own header covering only the symbols rmcl uses.

## CUDA surface inventory

rmcl's own CUDA surface is small and splits cleanly into a portable half and an OptiX half.

### Portable half (in scope)

| file | what is in it | ROCm mapping |
|---|---|---|
| `rmcl_ros/src/rmcl/particle_motion.cu` (48 lines) | 1 kernel `particle_move_and_forget_kernel`, 1 launch, `#include <cuda_runtime.h>` | HIP language; include redirect |
| `rmcl_ros/src/rmcl/resampling.cu` (220 lines) | 3 kernels (`init_curand_kernel`, `simple_stats_kernel<512>`, `gladiator_resample_kernel`), 3 launches, 2 `__device__` helpers, `__shared__`, 2 `__syncthreads`, `curand_init` / `curand` / `curand_normal` / `curandState` | HIP language; cuRAND -> hipRAND |
| `rmcl_ros/include/rmcl_ros/rmcl/resampling.cuh` | public header, `#include <curand.h>`, `#include <curand_kernel.h>`, `curandState` in the exported signatures | include redirect via the compat header |
| `rmcl_ros/include/rmcl_ros/rmcl/GladiatorResamplerGPU.hpp` | public header, same two cuRAND includes, `rmagine::Memory<curandState, VRAM_CUDA>` member | include redirect; see the risk about host TUs |
| `rmcl_ros/include/rmcl_ros/rmcl/particle_motion.cuh` | declarations only, no CUDA header | none |

Symbol census over the whole tree (regex, no hipify run): `curand`-family 26 hits, 7
`__global__`, 11 `__device__`/`__shared__`, 4 launches, 4 `#include <cuda_runtime.h>`,
`cudaMalloc`/`cudaFree`/`cudaMallocHost`/`cudaFreeHost`/`cudaMemcpyAsync`/`cudaMemcpyHostToDevice`
(all inside the OptiX half). Notably absent: no `__shfl*`, no `__ballot`, no `warpSize`, no
hardcoded 32, no textures or surfaces, no cuBLAS/cuFFT/cuSPARSE/cuSOLVER/cuDNN, no Thrust/CUB/
CUTLASS, no NCCL, no streams or events of rmcl's own, no managed memory. rmcl gets all of that
from rmagine, which is already ported.

The C++ sources of `rmcl_ros_cuda` -- `MICPSensorCUDA.cpp`, `MICPSphericalSensorCUDA.cpp`,
`MICPPinholeSensorCUDA.cpp`, `MICPO1DnSensorCUDA.cpp`, `MICPOnDnSensorCUDA.cpp` -- and the whole
of `rmcl-cuda` (`CorrespondencesCUDA.cpp`) contain no CUDA API calls at all. They use rmagine's
`MemoryView<..., VRAM_CUDA>` and `rm::statistics_p2l`. rmagine's public `types/MemoryCuda.hpp`
pulls in only `util/cuda/cuda_definitions.h`, which is pure `std::shared_ptr` typedefs and no CUDA
header, so these translation units compile with a plain host compiler against a HIP-built rmagine.
That is the single most important structural fact in this plan: it keeps the HIP-compiled surface
down to three translation units.

### OptiX half (out of scope, see below)

`rmcl_ros/src/rmcl/optix/BeamEvaluateProgram.cu` (130 lines of `__raygen__`/`__closesthit__`/
`__miss__` compiled to PTX), `cuda_math_helper.cuh`, `cuda_rmagine_conversions.cuh`,
`eval_modules.cpp`, `eval_program_groups.cpp`, `eval_pipelines.cpp`, `PCDSensorUpdaterOptix.cpp`,
`TFMotionUpdaterGPU.cpp`, `EvaluationDataOptix.hpp`, and `rmcl/src/rmcl/registration/RCCOptix.cpp`.
About 43 OptiX symbols plus the `CUDA_PTX_COMPILATION` codegen path.

## Port strategy

### What gets built on ROCm

With an rmagine built `-DUSE_HIP=ON` and installed, rmcl configures with `rmagine::core` and
`rmagine::cuda` present and `rmagine::optix` / `rmagine::vulkan` absent. That yields:

- `rmcl` and `rmcl-cuda` from the `rmcl` package -- expected to need **zero source and zero CMake
  changes**. `rmcl/CMakeLists.txt:192-243` only enables the CUDA language when
  `CMAKE_CUDA_COMPILER` is found, and on a ROCm-only host it is not, so `CUDA_LIBRARIES` stays
  empty and `rmcl-cuda` links `rmagine::cuda` alone (`:251-254`). The porter must confirm this
  rather than assume it.
- `rmcl_ros`, `rmcl_ros_cuda`, `micp_localization` and the converter/segmentation nodes from the
  `rmcl_ros` package. `rmcl_ros_cuda` is gated on `RMCL_CUDA` alone
  (`rmcl_ros/CMakeLists.txt:171`, set from `if(TARGET rmcl-cuda)` at `:75`), **not** on OptiX. This
  corrects the intake note: the CUDA compute library is reachable at build level with only
  `rmagine::cuda`.

### The honest limitation, stated up front

`rmcl_ros_cuda` builds and its kernels run, but no rmcl **node** instantiates them on a
ROCm-only configuration. `micp_localization.cpp:534-782` selects a correspondence backend from a
string, and the only three are `"embree"` (CPU), `"optix"` and `"vulkan"`. The
`MICPSensorCUDA` family is constructed only inside `#ifdef RMCL_OPTIX` (`:618-696`) and
`#ifdef RMCL_VULKAN` (`:700-778`), because those GPU sensor classes are passengers of a GPU
ray-casting backend rather than a backend in their own right. `rmcl_localization` is gated on
`if(RMCL_OPTIX AND RMCL_EMBREE)` (`rmcl_ros/CMakeLists.txt:701`) with upstream's own comment
saying the path is NVIDIA-only. The "combining unit GPU" line printed at `:342-344` is vestigial;
nothing reads a combining-unit parameter anywhere in the tree.

So the deliverable is: the ROCm build of rmcl compiles every component that does not require
OptiX, its GPU kernels are exercised and asserted correct on an AMD GPU by tests this port adds,
and the CPU (Embree) localization path is unaffected. Making the GPU correspondence path
*reachable* needs a ray-casting backend and that work is not in rmcl -- see below. Say this
plainly in the PR body; do not let the change read as "MICP-L now runs on AMD GPUs".

### OptiX and why it is not in this port (port vs rewrite)

There is no HIP translation target for OptiX; the AMD analogue is HIPRT, which is a
reimplementation and not a translation. rmagine Stage 2 already built such a reimplementation, and
inspecting it decides the question:

- `rmagine_hiprt` has no `install()`, no `install(EXPORT ...)`, no `rmagine::hiprt` alias and no
  `cmake/rmagine-hiprt-config.cmake.in`. It cannot be found by `find_package` from another
  project at all, so rmcl has nothing to link against today.
- Its simulators expose `simulateRanges` / `simulateHits` / `simulatePoints` / `simulate(Tbm,
  HiprtSimulationData&)`. rmcl's `RCCOptix*` classes inherit rmagine's simulator templates and
  call `simulate(Tbm_est, model_buffers_)` with a resizable bundle carrying points, **normals**
  and hits (`rmcl/src/rmcl/registration/RCCOptix.cpp:29-42`), and `CorrespondencesCUDA` feeds
  those normals to `rm::statistics_p2l`. The HIPRT backend computes no normals (its own notes list
  "face normals computation (currently not used)" as remaining work) and does not implement the
  bundle-templated interface.
- It has no map or import layer: rmcl needs an `rm::OptixMap` analogue loadable from a mesh file,
  and `rmagine_hiprt` ships `HiprtMesh`/`HiprtScene` built from raw vertex and index arrays, with
  all meshes merged into one geometry -- rmcl uses scene graphs with instances.
- It needs the HIPRT SDK, which is absent on this host and is not part of ROCm.

Those four gaps are rmagine-side work, not an rmcl wrapper. Once they are closed the rmcl side is
genuinely small: `RCCOptix.cpp` is 157 lines of thin subclassing, so `RCC*Hiprt` would mirror it,
plus a `rmcl-hiprt` library gated on `if(TARGET rmagine::hiprt)` and one more branch in
`micp_localization.cpp`. Register that as deferred work against this project rather than
attempting it here.

A correctness-first mechanical port is therefore the right first step, and the AMD-native
ray-tracing pass is a separate, later, rmagine-led piece of work.

### Cheaper alternative worth a person's attention

The `"vulkan"` backend string already instantiates the same `MICPSensorCUDA` family
(`micp_localization.cpp:700-778`) and rmagine ships a complete cross-vendor Vulkan ray-tracing
backend (`src/rmagine_vulkan`, ~45 files) that is not NVIDIA-specific. The only CUDA-bound piece
is `src/rmagine_vulkan_cuda_interop/src/types/VulkanCudaInterop.cpp`, whose entire CUDA surface is
six symbols -- `cudaImportExternalMemory`, `cudaExternalMemoryGetMappedBuffer`,
`cudaExternalMemoryHandleDesc`, `cudaExternalMemoryBufferDesc`,
`cudaExternalMemoryHandleTypeOpaqueFd`, `cudaExternalMemoryHandleTypeOpaqueWin32` -- each with a
direct `hipImportExternalMemory` / `hipExternalMemory*` equivalent. Porting that one file would
give rmcl a *reachable* GPU correspondence backend on RDNA with **no rmcl source change beyond the
build**. It would not help gfx90a, which has no graphics or ray-tracing pipeline. That work is
rmagine's, not rmcl's, and it is offered here as a routing suggestion, not as part of this plan.

## Consuming the ported rmagine (the missing dependency-install story)

`orient.sh` reports `dep-doc MISSING rmagine`. rmagine's notes have no `## Install as a
dependency` section because rmagine has only ever been built and tested **in its own build tree**;
every recorded validation runs `ctest` from the build directory. rmcl is the first consumer, and
the install path turns out to be broken for a ROCm build. Findings, from reading the fork at
`moat-port`:

**What does work.** `src/rmagine_cuda/CMakeLists.txt` keeps `EXPORT_NAME cuda`, the
`rmagine::cuda` alias, the `install(TARGETS ... EXPORT rmagine-cuda-targets)` and the header
install under `${prefix}/include/rmagine-2.4.2/rmagine/` in the `USE_HIP` branch as well as the
CUDA one. So a `cmake --install` of a `-DUSE_HIP=ON` build produces a normal rmagine package with
`rmagine::core` and `rmagine::cuda`, and rmcl's `find_package(rmagine 2.4 COMPONENTS core
OPTIONAL_COMPONENTS embree cuda optix vulkan vulkan-cuda-interop)` (`rmcl/CMakeLists.txt:63-72`)
is the right call to make against it. `rmagine::optix`, `rmagine::vulkan` and `rmagine::hiprt`
will simply be absent.

**What does not work.** `src/rmagine_cuda/cmake/rmagine-cuda-config.cmake.in` unconditionally
resolves CUDA at consumer time:

```cmake
include(CMakeFindDependencyMacro)
if(@CUDAToolkit_FOUND@)
    find_dependency(CUDAToolkit)
else(@CUDAToolkit_FOUND@)
    find_dependency(CUDA)
endif(@CUDAToolkit_FOUND@)
```

Under `USE_HIP` the CUDA branch of `rmagine_cuda/CMakeLists.txt` never runs, so `CUDAToolkit_FOUND`
is undefined and the generated file reads `if()` -- valid CMake, evaluates false -- and falls
through to `find_dependency(CUDA)` on a machine with no CUDA. I reproduced the consequence with a
minimal package that mirrors rmagine's config structure (CMake 3.31.6, this host):

```
-- Could NOT find CUDA (missing: CUDA_TOOLKIT_ROOT_DIR CUDA_NVCC_EXECUTABLE ...)
CMake Warning at CMakeLists.txt:3 (find_package):
  ... but it set rmagine_FOUND to FALSE so package "rmagine" is considered to be NOT FOUND.
  Reason given by package: rmagine could not be found because dependency CUDA could not be found.
-- rmagine_FOUND=0
```

`find_dependency`'s failure returns out of `rmagine-config.cmake` entirely, so the *whole* package
is reported not found -- `rmagine::core` included, and `OPTIONAL_COMPONENTS` does not save it
because `rmagine-config.cmake.in` includes every component config it finds on disk without
distinguishing optional from required. rmcl then fails at
`target_link_libraries(rmcl rmagine::core ...)`. **This is the first thing the porter will hit,
before any HIP code is compiled.**

Recommended fix, which belongs on `port/rmagine` and not on this branch, in
`src/rmagine_cuda/cmake/rmagine-cuda-config.cmake.in`:

```cmake
include(CMakeFindDependencyMacro)
if("@USE_HIP@")
    find_dependency(hip)
    find_dependency(hiprand)
else()
    if("@CUDAToolkit_FOUND@")
        find_dependency(CUDAToolkit)
    else()
        find_dependency(CUDA)
    endif()
    include(CheckLanguage)
    check_language(CUDA)
endif()
```

The `hip` / `hiprand` half of that is not cosmetic: the exported
`rmagine-cuda-targets.cmake` records `hip::host` and `hip::hiprand` in the public link interface
(`src/rmagine_cuda/CMakeLists.txt`, `USE_HIP` branch), and a consumer that has not run
`find_package(hip)` will fail at generate time on a link to a target that does not exist. The
porter should verify both halves empirically rather than trusting this paragraph.

This finding is also the substance of the `## Install as a dependency` section rmagine is missing.
Recommended content for it, once the fix above is made and revalidated: the configure line with
`-DCMAKE_INSTALL_PREFIX`, `cmake --install`, the resulting layout
(`${prefix}/lib/cmake/rmagine-2.4.2/`, `${prefix}/include/rmagine-2.4.2/`), the exact
`find_package` call a consumer makes, which targets exist in a ROCm build (`rmagine::core`,
`rmagine::cuda`) and which do not (`optix`, `vulkan`, `hiprt`), and the note that
`CMAKE_PREFIX_PATH` must carry the prefix. Do not write it from this branch -- it is an edit to
the rmagine project record.

**Sequencing question for a person.** Fixing that config file advances rmagine's `head_sha` and
therefore invalidates its four completed platform validations, even though the change is
install-side only and cannot affect kernel behaviour. Options: (a) run a small rmagine delta round
first and revalidate, (b) let the rmcl porter make the fix on the rmagine fork as part of this
work and accept the revalidation, (c) something else. This is a routing decision, not an agent's.

## Risk list

1. **`find_package(rmagine)` fails outright on a CUDA-free host.** Reproduced above. Highest
   probability, blocks everything, fix is in rmagine.
2. **`hip::host` / `hip::hiprand` missing from the consumer's scope.** Same file, same fix. Symptom
   is a generate-time error naming a nonexistent target, not a compile error.
3. **`GladiatorResamplerGPU.cpp` is a host C++ translation unit that includes cuRAND device
   headers.** `GladiatorResamplerGPU.hpp` pulls `<curand.h>` and `<curand_kernel.h>` and holds a
   `Memory<curandState, VRAM_CUDA>`; `resampling.cuh` does the same. Under HIP the replacement
   `hiprand_kernel.h` is not host-compilable by plain g++. This is the skill's "a shared compat
   header must be host-includable" class. Preferred fix: mark `GladiatorResamplerGPU.cpp`
   `LANGUAGE HIP` alongside the two `.cu` files, so all three see the HIP toolchain. If that
   proves awkward, the fallback is to make the compat header expose only the `hiprandState` type
   to host TUs and keep the device header behind `__HIPCC__`.
4. **`__HIP_PLATFORM_AMD__` is undefined until `hip_runtime.h` is included.** Gate the compat
   header on the project's own `USE_HIP` define first, exactly as rmagine's does, not on the
   platform macro alone.
5. **Missing includes unmasked by the narrower HIP include graph.** `resampling.cu:104,137` uses
   `UINT_MAX` with no `<climits>`; it currently arrives transitively through the CUDA headers.
   Expect one or two such fixes; they are pre-existing upstream omissions and should be fixed
   unconditionally (they are correct on CUDA too), not hidden behind `USE_HIP`.
6. **cuRAND to hipRAND is not bitwise equivalent.** `curand_init(1234, idx, 0, ...)` produces a
   different stream under hipRAND. The resampler is therefore *not* comparable run-for-run against
   CUDA; validate it statistically (mean and standard deviation of `curand_normal` samples) and by
   determinism across two runs on the same platform. rmagine recorded the same conclusion.
7. **`curand` as a bare token.** `resampling.cu:135` calls `curand(&rstate)`, whose name is a
   prefix of `curand_init` and `curand_normal`. A `#define curand hiprand` is safe (the
   preprocessor matches whole identifiers) but it is easy to get wrong by editing the source
   instead; keep it in the header with the others.
8. **Wavefront size.** rmcl has no warp intrinsics, no `warpSize`, no hardcoded 32 and no
   warp-synchronous reduction tail. `simple_stats_kernel` (`resampling.cu:68-75`) already runs the
   complete `__syncthreads` tree down to `s > 0`, with the barrier outside the `if(tid < s)`, so it
   is correct on wave32 and wave64 alike. This is the pattern rmagine had to be *fixed* into;
   rmcl was written that way already. No wave32-specific change is expected on this host, and the
   rmagine layer underneath is already validated on gfx1100, gfx1101 and gfx1201.
9. **Uninitialised LDS.** `simple_stats_kernel` seeds shared memory with literal zeros
   (`resampling.cu:54-55`), not with the `*= 0.0` idiom that produced NaN on AMD in rmagine. No
   action, but check any new reduction the port adds.
10. **Block size 1024 with a 512-wide template.** `compute_stats` launches
    `simple_stats_kernel<512><<<n_outputs, 512>>>`; the other two kernels launch 1024 threads.
    Keep the template parameter and `blockDim.x` in agreement if anything is touched, and note
    that 1024 is the maximum flat workgroup size on the AMD targets in the fleet.
11. **Upstream oddity, do not "fix" it.** `resampling.cu:51` computes `globId = N * blockIdx.x +
    threadIdx.x`, so with more than one output block every block but the first reads past `N` and
    contributes nothing. It is guarded, so it is not an out-of-bounds read, and the only in-tree
    caller passes a single-element `stats` buffer. Leave it alone and size the test's `stats`
    buffer to one element; an unrelated upstream fix does not belong in a porting PR.
12. **rmagine is built with `-fgpu-rdc` and device-linked into its shared library.** rmcl needs no
    relocatable device code of its own (`merge`, `lcg_rand` and their callers are in the same
    translation unit), so do not copy rmagine's `-fgpu-rdc` / `--hip-link` flags into rmcl. If a
    device-side undefined symbol appears at link time, revisit.
13. **`hip::device` must stay PRIVATE** on `rmcl_ros_cuda` or `--offload-arch=` leaks onto the
    plain C++ compiles of the MICP sensor sources and the ROS nodes. rmagine hit this.
14. **`add_compile_options(-std=c++17)` at `rmcl_ros/CMakeLists.txt:20` applies to every language**,
    HIP included. Harmless for clang, but worth remembering if a flag conflict appears.
15. **Toolchain provisioning is the largest schedule risk, not the code.** ROS 2 jazzy, Embree and
    assimp are all absent on this host. See "Build commands".
16. **Do not touch the OptiX or Vulkan sources.** They must keep compiling on an NVIDIA host. Every
    change belongs inside `if(USE_HIP)` / `#if defined(USE_HIP)`, so the CUDA path stays a pure
    passthrough. There is no NVIDIA host in the fleet to prove this by compiling, so it has to be
    proven by construction and checked in review.

## File-by-file change list

Everything below is additive and `USE_HIP`-gated unless stated.

**New**

- `rmcl_ros/include/rmcl_ros/util/cuda_to_hip.h` -- the compat shim. Under `USE_HIP` (or
  `__HIP_PLATFORM_AMD__`) include `<hip/hip_runtime.h>` and `<hiprand/hiprand_kernel.h>` and map
  only what rmcl uses: `curandState`, `curand_init`, `curand`, `curand_normal`, plus the handful of
  `cudaXxx` runtime spellings if any survive in scope. Otherwise include the real CUDA headers so
  the NVIDIA build is unchanged. Keep it host-includable: no device-only include outside a
  `__CUDACC__`/`__HIPCC__` guard. Follow rmagine's header for shape, including its warning about
  not writing a comment that contains an end-of-comment marker.
- `rmcl_ros/tests/rmcl_ros_cuda_kernels.cpp` (name to taste) -- the GPU test described below.
- `rmcl/tests/rmcl_cuda_cross_statistics.cpp` (optional, second test) -- see the test plan.

**Modified**

- `rmcl_ros/CMakeLists.txt` -- add `option(USE_HIP ...)` defaulting OFF; in the `RMCL_CUDA` block
  (`:171-201`) take a `USE_HIP` branch that runs `enable_language(HIP)`, `find_package(hip)`,
  `find_package(hiprand)` and skips the CUDA language detection; add
  `set_source_files_properties(src/rmcl/particle_motion.cu src/rmcl/resampling.cu
  src/rmcl/GladiatorResamplerGPU.cpp PROPERTIES LANGUAGE HIP)`; force-include the compat header on
  HIP translation units via `CMAKE_HIP_FLAGS` (or include it explicitly from the three sources --
  the explicit form makes a cleaner upstream diff, decide when you see the compile errors); link
  `hip::host` and `hip::hiprand` PUBLIC and `hip::device` PRIVATE on `rmcl_ros_cuda`; default
  `CMAKE_HIP_ARCHITECTURES` only when the caller left it unset. Register the new test.
- `rmcl_ros/include/rmcl_ros/rmcl/resampling.cuh` and
  `rmcl_ros/include/rmcl_ros/rmcl/GladiatorResamplerGPU.hpp` -- replace the two `<curand*.h>`
  include lines with the compat header. Two lines each.
- `rmcl_ros/src/rmcl/particle_motion.cu` and `rmcl_ros/src/rmcl/resampling.cu` -- replace
  `#include <cuda_runtime.h>` / `#include <curand.h>` with the compat header; add `<climits>` if
  `UINT_MAX` turns out to be missing. No kernel body change is anticipated.
- `rmcl/CMakeLists.txt` -- expected untouched. If `rmcl-cuda` needs anything, it will be a guard so
  the CUDA-language block at `:192-243` is skipped under `USE_HIP`; confirm empirically.

**Expected untouched**

`rmcl/src/rmcl/registration/CorrespondencesCUDA.cpp`, all five `MICP*SensorCUDA.cpp`, every CPU
source, every ROS node, and the entire OptiX and Vulkan surface.

## Build commands

Prerequisites on a bare host (this host, linux-gfx1100, has ROCm 7.2.3 at `/opt/rocm`,
Ubuntu 24.04, Eigen3 and TBB; it lacks ROS 2, Embree and assimp):

```
sudo apt-get install -y libassimp-dev libboost-all-dev libeigen3-dev libtbb-dev libembree-dev
# ROS 2 jazzy (Ubuntu 24.04 noble is the jazzy target) per docs.ros.org, then:
sudo apt-get install -y ros-jazzy-ros-base ros-jazzy-tf2 ros-jazzy-tf2-ros \
  ros-jazzy-image-transport ros-jazzy-visualization-msgs ros-jazzy-std-srvs \
  ros-jazzy-rclcpp-components python3-colcon-common-extensions
```

Step 1, build and **install** rmagine from the fork (this is the part no previous run has done):

```
git clone -b moat-port https://github.com/AMD-Ecosystem/rmagine <ws>/rmagine_src
cmake -S <ws>/rmagine_src -B <ws>/rmagine_build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DRMAGINE_OPTIX_DISABLE=ON -DRMAGINE_VULKAN_DISABLE=ON \
  -DRMAGINE_VULKAN_CUDA_INTEROP_DISABLE=ON -DRMAGINE_OUSTER_DISABLE=ON \
  -DRMAGINE_BUILD_TESTS=OFF -DRMAGINE_BUILD_TOOLS=OFF \
  -DCMAKE_INSTALL_PREFIX=<ws>/rmagine_install
cmake --build <ws>/rmagine_build -j$(nproc)
cmake --install <ws>/rmagine_build
```

Use `-DCMAKE_HIP_ARCHITECTURES=gfx90a` on CDNA, `gfx1101`/`gfx1201` elsewhere. Leave
`RMAGINE_EMBREE_DISABLE` unset so `rmagine::embree` exists and rmcl's CPU backend builds; that is
what keeps the non-GPU regression set meaningful.

Step 2, build the rmcl workspace:

```
. /opt/ros/jazzy/setup.bash
colcon build --packages-select rmcl_msgs rmcl rmcl_ros \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
               -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
               -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
               -DCMAKE_PREFIX_PATH=<ws>/rmagine_install
```

Wrap both with `utils/timeit.sh rmcl compile -- ...` and pass absolute build directories, since
`timeit.sh` changes directory to the MOAT repository root.

If ROS 2 provisioning fails or eats the budget, there is a genuinely useful partial target: the
`rmcl` package alone is plain CMake with no ROS dependency at all (`cmake -S rmcl -B build
-DCMAKE_PREFIX_PATH=<ws>/rmagine_install`), which builds `rmcl` and `rmcl-cuda` and can carry the
cross-statistics GPU test. Record that as partial value rather than reporting a total loss.

## Test plan

**Upstream has no tests.** No test directory, no `add_test`, no `ament_add_gtest`, no gtest or
Catch2 dependency, no launch file, no config, no sample mesh. The three CI workflows
(`ros2_humble|jazzy|lyrical` over a shared template) run `colcon build` and then a `colcon test`
step with `skip-tests: false` that has nothing to execute. `rmcl/CMakeLists.txt:7` even declares
`option(BUILD_TESTS "Build tests" ON)` and never uses it.

So the evidence has to be constructed. Build it in the fork as plain executables registered with
`add_test`, matching rmagine's own test style, so no new dependency enters `package.xml` and
`colcon test` gains something real to run -- which is a small genuine gift to upstream and worth a
sentence in the PR body.

Required GPU test, `rmcl_ros`, gated on `RMCL_CUDA` (and `BUILD_TESTING`):

1. `particle_move_and_forget` over 4099 particles (deliberately not a power of two): compare every
   output pose against a CPU reference composition `pose_old * T_bnew_bold` and the forget-rate
   arithmetic on `n_meas`, assert relative error within tolerance, assert no NaN.
2. `compute_stats` over the same 4099 particles with a one-element `stats` buffer: compare `sum`
   against a double-precision CPU reference and `max` exactly; run twice and assert the results are
   bit-identical. This is the reduction gate and the one kernel where a wavefront assumption could
   have hidden.
3. `init_curand` plus `gladiator_resample`: assert no NaN in any output pose or attribute; assert
   the structural invariant that a particle whose sampled opponent is not stronger is copied
   through **exactly** (bitwise equal input and output), which is a real assertion that survives
   the fact that hipRAND's stream differs from cuRAND's; assert the sampled normals have mean near
   zero and standard deviation near one over the full particle set; assert two runs from the same
   freshly initialised states agree.
4. Confirm the kernels actually dispatched on the AMD device with `AMD_LOG_LEVEL=3` and record the
   `ShaderName` lines and the target triple, as rmagine's validations did. A test that passes
   without dispatching anything is not evidence.

Optional second test, `rmcl` package, no ROS needed: `CorrespondencesCUDA::computeCrossStatistics`
over a synthetic dataset and model point cloud whose Umeyama cross-statistics are analytic
(identity correspondence, known mean and covariance), compared against a CPU reference. Worth
doing because it exercises the one GPU entry point in the `rmcl` package and it runs on a host
with no ROS 2. Wire it under the existing unused `BUILD_TESTS` option.

Running:

```
colcon test --packages-select rmcl rmcl_ros --event-handlers console_direct+
# or, per build directory
ctest --test-dir <build>/rmcl_ros --output-on-failure
```

**Non-GPU regression set that must not break:**

- Every CPU target still configures and builds: `rmcl`, `rmcl-embree`, `rmcl_ros`,
  `rmcl_embree_ros`, `micp_localization`, `conv_pc2_to_scan`, `conv_pc2_to_o1dn`,
  `conv_scan_to_scan`, `map_segmentation`, `o1dn_map_segmentation_embree`,
  `scan_map_segmentation_embree`, and the `rmcl_msgs` message generation.
- A `USE_HIP=OFF` configure against the same rmagine install still behaves exactly as before, so
  the option is genuinely opt-in.
- The `micp_localization_node` startup banner (`docs/MICPL.md:54-60`) still prints its
  `--- BACKENDS ---` block listing Embree; that is a cheap end-to-end dispatch check if a ROS
  environment is up, though it is a smoke check and not the GPU gate.
- The OptiX and Vulkan sources are unmodified, verifiable from `git diff` alone.

There is no NVIDIA host in the fleet, so the usual "compile the CUDA path with nvcc" no-regression
gate cannot be run. Substitute: every change is inside a `USE_HIP` branch, and the reviewer checks
that claim against the diff.

## Port surface accounting

`projects/rmcl/surface.json` carries the machine-checked enumeration; `check.py`'s `surface` gate
will refuse a success claim that leaves a component unaccounted for. Additions and removals I made
to the generated floor:

- Added `package:rmcl_msgs` -- the scanner sees no `add_library` there because the package is built
  by `rosidl_generate_interfaces`.
- Added the two benchmark sources under `rmcl_ros/src/benchmarks/`. They exist in the tree and are
  wired into no CMake target at all; recording them stops a reviewer wondering whether they were
  missed.
- Added `codegen:rmcl_ros_optix_ptx`, the `CUDA_PTX_COMPILATION` custom target plus
  `cmake/CompileOptixKernels.cmake` and `cmake/CompileOptixKernelsCudaToolkit.cmake`, which is a
  build path rather than a library and so is invisible to the target scan.
- Added `test:rmcl_gpu_tests` and `option:USE_HIP`, both introduced by this port.
- Removed four floor entries with reasons recorded in `removed_from_floor`: two executables and
  two options that exist only inside comment blocks (`rmcl_ros/CMakeLists.txt:5-6` and `:570-605`)
  and are built by no configuration.

The components the porter must be able to move into `covered` with evidence are exactly:
`library:rmcl-cuda`, `library:rmcl_ros_cuda`, `test:rmcl_gpu_tests`, `option:USE_HIP`. Everything
else is pre-scoped with a reason; if the port ends up touching one of them, move it and say why.

## Open questions

1. **Where does the rmagine config fix go, and who pays for the revalidation?** A person's call;
   options laid out in the dependency section above. Nothing in rmcl can be built until it is
   resolved one way or another.
2. **Is a compute-only rmcl port worth an upstream PR on its own?** It compiles and it is tested,
   but it does not make any node's GPU path reachable, and it depends on an rmagine feature that is
   not upstream yet. A defensible answer is yes-as-groundwork; another is to hold the rmcl PR until
   the ray-casting story exists. Worth deciding before publication rather than after.
3. **Should the next investment be rmagine's Vulkan interop rather than rmcl at all?** Six CUDA
   symbols stand between rmcl and a reachable GPU correspondence backend on RDNA. That may be a
   better use of the next slot than anything in this repository.
4. **HIPRT SDK provisioning.** Absent on this host and not part of ROCm. Only matters if question 3
   is answered with HIPRT instead of Vulkan.
5. **`amock/rmcl_examples`** (BSD-3, the maintainer's own) carries the meshes and launch files the
   README points newcomers to. It would give an end-to-end demo, at the cost of a second external
   repository in the validation loop. Not needed for the tests above; noted in case someone wants a
   qualitative check.
