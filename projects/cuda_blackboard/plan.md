# cuda_blackboard port plan

Planner run 2026-11-20 on linux-gfx950 (no GPU used, nothing built). Upstream analysed at `a4f330e5c228d713774450b0331b83b94c55a466` (main, 2026-10-27, version 0.5.0).

## Project

- Name: cuda_blackboard
- Upstream: https://github.com/autowarefoundation/cuda_blackboard (default branch `main`)
- Fork: https://github.com/AMD-Ecosystem/cuda_blackboard, port branch `moat-port`
- What it is: a ROS 2 package (ament_cmake_auto, C++17) implementing REP 2007 type adaptation plus REP 2009 type negotiation, so GPU-resident `sensor_msgs` (PointCloud2, Image) pass between nodes in one process without a host round trip. Two process-wide singletons: `CudaBlackboard<T>` (ticketed buffer dictionary) and `CudaMemPoolContext` (stream-ordered pool plus an allocation stream and a free stream).
- Consumer: autoware_universe (hard dependency). Its pointcloud preprocessor and other nodes include these headers and pass their own streams in.

## Existing AMD support

None. Intake found no upstream PR, issue, branch or fork doing a HIP port, and no AMD-Ecosystem or ROCm repository. The README says the package "requires the use of NVIDIA GPUs and CUDA". The AMD Silo AI / Autoware Foundation collaboration (2025-12-16) has published nothing that touches this package. Decision: fresh port. Upstream is an actively maintained Autoware Foundation repository that takes PRs, so an upstream PR is the right vehicle.

No performance kernels exist (there are no kernels at all), so the mechanical-port versus AMD-native-rewrite question does not arise.

## Build classification

Pure CMake (ament_cmake_auto) -> Strategy A. Evidence:

- `CMakeLists.txt:5` `find_package(CUDA)` (legacy FindCUDA), `:26-29` returns early with a warning when CUDA is absent, `:33-37` adds `${CUDA_INCLUDE_DIRS}`, `:50-52` and `:59-63` link `${CUDA_LIBRARIES}`.
- No `.cu` or `.cuh` file, no `enable_language(CUDA)`, no torch. Every CUDA call is made from host C++ through `cuda_runtime_api.h` / `cuda_runtime.h`.
- `package.xml` declares no CUDA dependency; colcon builds it with `colcon build --packages-up-to cuda_blackboard`.

Recorded `ext_type` = `cmake`.

## Port strategy

Strategy A, host-only variant: nothing goes through a device compiler. Sources stay `.cpp` compiled as CXX, link `hip::host`, and need only `__HIP_PLATFORM_AMD__` plus the HIP include path. Do NOT mark any source `LANGUAGE HIP` and do not call `enable_language(HIP)`: there is no device code, and that would add a device pass for nothing and require a HIP compiler on consumer hosts.

### The compat header must reach consumers

The installed public headers spell CUDA types and call CUDA functions inline: `cudaStream_t` in the subscriber constructor and `CudaMemPoolContext`, `cudaEvent_t` members in `CudaImage`/`CudaPointCloud2`, inline `cudaMallocFromPoolAsync`/`cudaMemsetAsync`/`cudaFreeAsync`/`cudaMallocHost`/`cudaFreeHost` in `cuda_unique_ptr.hpp`, inline `cudaMemcpyAsync` in `cuda_adaptation.hpp`, the `cuda_check_error` template in `cuda_error.hpp`, and `cudaStreamLegacy` as a default argument in `cuda_blackboard_subscriber.hpp`. A consumer compiled against the installed headers on ROCm therefore needs the same mapping. A private shim directory is not enough.

Decision: one namespaced compat header, not shadow `cuda_runtime.h` / `cuda_runtime_api.h` headers.

- New installed header `include/cuda_blackboard/cuda_runtime_compat.hpp` (porter may pick the exact name). On CUDA it is `#include <cuda_runtime.h>` and nothing else. On HIP (`__HIP_PLATFORM_AMD__`) it includes `<hip/hip_runtime_api.h>` and defines the CUDA names this package uses as macros for their HIP equivalents (list below).
- Replace the CUDA includes with this header: the 10 `#include <cuda_runtime_api.h>` and 2 `#include <cuda_runtime.h>` lines in `include/`, `src/` and `test/`. That is about a dozen one-line edits. Every other line of the sources stays as it is.
- Why not shadow headers: the package's include directory is exported to every consumer through ament. A shadow `cuda_runtime.h` there would win the include search in a consumer's own `.cu`/`.hip` files and hand them only this package's ~45 aliases, breaking any consumer that relies on its own fuller `cuda_runtime.h` shim (autoware_universe's CUDA packages, autoware_cuda_utils). A header under `cuda_blackboard/` cannot shadow anything.
- Make every alias `#ifndef <name>` / `#define <name> <hip name>`, and use the expansion hipify uses (`cudaMallocHost` -> `hipHostMalloc`, `cudaFreeHost` -> `hipHostFree`; `hipMallocHost` is deprecated in HIP 7.16). That way a consumer's own compat header, included before or after ours, either already defines the same token sequence (a legal identical redefinition) or is not overridden by ours. This matters because Autoware packages often build with `-Werror`. Whichever side loses, the expansions must match. Record the alias list in notes.md so the autoware_cuda and autoware_universe porters reuse the same expansions.
- Macros, not `using` aliases, because a consumer shim that `#define`s `cudaStream_t` would turn a `using cudaStream_t = hipStream_t;` into a self-alias. Also because function names (`cudaFreeAsync`) and enumerators (`cudaMemcpyHostToDevice`) cannot be type-aliased.

### Build change (CMakeLists.txt)

Keep the CUDA path byte-for-byte in behaviour. Add the HIP path where CUDA is not found:

- `find_package(CUDA)` stays first. When `NOT CUDA_FOUND`, try `find_package(hip CONFIG QUIET)`. If that is found too, take a HIP path that builds the same targets, linking `hip::host` where the CUDA path links `${CUDA_LIBRARIES}` (library and both GPU tests). The existing "cuda was not found" warning and early `return()` then apply only when neither is found. A host with CUDA builds exactly as today. A ROCm host builds the HIP path with the unchanged documented command (`colcon build`), with no new flag. An explicit opt-in option is not needed and would make the documented command produce a CPU-only build on ROCm.
- Consumers must see `__HIP_PLATFORM_AMD__` and find HIP. `hip::host` carries the definition as an INTERFACE property, but ament_cmake_auto consumers (`ament_auto_find_build_dependencies` / `ament_target_dependencies`) read the package's exported variables, not imported-target interfaces. On the HIP path add `ament_export_dependencies(hip)` and `ament_export_definitions(__HIP_PLATFORM_AMD__)` (ament_cmake_export_definitions). Prove it with a throwaway consumer package in `agent_space/` (an `ament_cmake_auto` package that includes `cuda_blackboard/cuda_adaptation.hpp` and calls `cuda_blackboard::make_unique<std::uint8_t[]>(16)`). Do not commit it.
- `include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})` becomes path-specific. On HIP the include path comes from `hip::host`; never hard-code `/opt/rocm/include` (pip ROCm SDK hosts have no `/opt/rocm`).
- No `--offload-arch` and no `CMAKE_HIP_ARCHITECTURES`: there is no device code.
- `package.xml`: no change (CUDA is not declared there either; ROCm is found by CMake as CUDA is).
- `test/cuda_test_utils.hpp` message strings say "CUDA device"; leave them. They are test diagnostics and editing them adds diff for no behaviour change.

## CUDA surface inventory

The generated census is empty (0 `.cu` files) because every call sits in host C++. surface.json carries the planner-added entries. All of it is CUDA runtime API. There is no driver API, NVRTC, kernel, texture, warp intrinsic, library (cuBLAS, Thrust, CUB, ...), managed memory or graph use. Wavefront width is irrelevant to this package.

| CUDA | HIP | where | note |
|---|---|---|---|
| `cuda_runtime_api.h`, `cuda_runtime.h` | `hip/hip_runtime_api.h` | 12 include lines | via compat header |
| `cudaStream_t`, `cudaEvent_t`, `cudaError_t`, `cudaMemPool_t`, `cudaMemcpyKind`, `cudaMemPoolProps` | `hip*` | public headers | public API spelling stays CUDA |
| `cudaStreamCreate`, `cudaStreamCreateWithFlags(cudaStreamNonBlocking)`, `cudaStreamDestroy`, `cudaStreamSynchronize`, `cudaStreamGetDevice` | `hipStream*`, `hipStreamGetDevice` | mem_pool_context, subscriber node, test fixture | `hipStreamGetDevice` present in HIP 7.16 |
| `cudaStreamLegacy` | `hipStreamLegacy` (`((hipStream_t)1)`) | subscriber header (default arg), subscriber.cpp | risk R1 |
| `cudaEventCreateWithFlags(cudaEventDisableTiming)`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventDestroy`, `cudaStreamWaitEvent(..., cudaEventWaitDefault)` | `hipEvent*`, `hipStreamWaitEvent`, `hipEventWaitDefault` | messages, subscriber, mem pool | risk R2 |
| `cudaMemPoolCreate`, `cudaMemPoolDestroy`, `cudaMemPoolSetAttribute(cudaMemPoolAttrReleaseThreshold)`, `cudaMemAllocationTypePinned`, `cudaMemLocationTypeDevice` | `hipMemPool*`, `hipMemAllocationTypePinned`, `hipMemLocationTypeDevice` | cuda_mem_pool_context.cpp | risk R3 |
| `cudaMallocFromPoolAsync`, `cudaFreeAsync`, `cudaFree`, `cudaMemsetAsync` | `hipMallocFromPoolAsync`, `hipFreeAsync`, `hipFree`, `hipMemsetAsync` | cuda_unique_ptr.hpp (inline, public) | |
| `cudaMallocHost`, `cudaFreeHost` | `hipHostMalloc`, `hipHostFree` | cuda_unique_ptr.hpp | not `hipMallocHost` (deprecated) |
| `cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemcpy{HostToDevice,DeviceToHost,DeviceToDevice}` | `hipMemcpy*` | adaptation, messages, subscriber node | |
| `cudaGetErrorName`, `cudaGetErrorString`, `cudaGetLastError`, `cudaGetDeviceCount`, `cudaSuccess` | `hip*` | cuda_error.hpp, test utils | error text differs (`hipError...`); nothing parses it |

All symbols were checked present in the HIP 7.16 headers of the ROCm SDK on this host. `cudaEventBlockingSync` appears only in comments.

## Risk list

- **R1, legacy default stream.** The deprecated subscriber constructors default `user_stream` to `cudaStreamLegacy`, and `needFallbackToLegacyDefaultStream` records the consumer-completion event on `cudaStreamLegacy` when the user passed `nullptr` or the legacy handle. HIP's `hipStreamLegacy` is the sentinel `(hipStream_t)1`. `hipEventRecord` and `hipStreamWaitEvent` must accept it and give legacy (blocking, synchronizing) semantics against the pool's non-blocking free stream. The upstream gtests only use a created stream, so this path is untested upstream. The validator checks it with the probe below.
- **R2, event ordering carries correctness.** The free is ordered after consumption only by `cudaEventRecord(user/legacy stream)` -> `cudaStreamWaitEvent(free_stream)` -> `cudaFreeAsync(free_stream)`. A broken wait shows up as stale or reused data, not a crash. HIP-specific trap to keep in mind (cuda-to-rocm `faults-memory-streams.md`): on HIP, querying or synchronizing an event after its recording stream was destroyed is a use-after-free. Here events are only recorded on the singleton pool streams (which live for the process) or on the user stream, followed immediately by `StreamWaitEvent`/`EventDestroy`, which do check. No change is expected, but the reviewer should confirm no `EventSynchronize`/`EventQuery` runs on an event whose recording stream can die. `blockCpuUntilStreamCompletion` synchronizes an event on the pool stream, which is safe.
- **R3, stream-ordered pool.** `cudaMemPoolCreate` with `allocType = Pinned` on `location.type = Device`, plus the `ReleaseThreshold` attribute (default 1 GiB, `CUDA_BLACKBOARD_MEM_POOL_RELEASE_THRESHOLD_MB`). HIP supports both, but the props struct must be zero-initialised (it is: `= {}`), and the threshold value is read through a `void*` to `uint64_t`, which matches HIP's signature. Pool memory is not zeroed on ROCm. Upstream already memsets every allocation (`make_unique_impl`), so the fresh-allocation read-modify-write fault class does not apply. Known ROCm behaviour that hits this code exactly (cuda-to-rocm `build-headers-macros.md`, seen on ROCm 10.1 Linux in another port): `hipMemPoolAttrReleaseThreshold` set BEFORE the pool's first allocation is lost, and freed blocks are unmapped at the next synchronize whatever the value. Even when set late, it keeps memory only in 128 MiB chunks. `CudaMemPoolContext` sets it in the constructor, before any allocation, and `make_unique` synchronizes on every allocation. So on ROCm the pool will likely not retain memory, and every message allocation may pay a fresh map. Correctness is unaffected; this is a performance difference. The porter does not work around it by default. The validator measures it (probe item b). If the effect is large, a HIP-guarded workaround (re-apply the threshold after the pool's first allocation) is a separate, measured commit, decided in review.
- **R4, singleton teardown order.** `CudaMemPoolContext` is a function-static singleton destroyed at exit. If the HIP runtime has already torn down by then, `hipStreamSynchronize`/`hipMemPoolDestroy` in the destructor may return errors. They are unchecked, so the visible failure would be a crash or hang at test exit, not a wrong result. Watch the gtest process exit codes. A fix, if needed, is HIP-guarded and minimal.
- **R5, consumer macro collisions.** See "The compat header must reach consumers". The `#ifndef` guards and hipify-identical expansions are the mitigation. The throwaway consumer proves the ament export.
- **R6, ROS 2 on the validators.** Fleet images carry no ROS 2. This is the main cost of the port, not the code.

Not applicable: warpSize / wave32-vs-wave64, textures and their pitch, rule-of-five on texture handles, OOB neighbour reads, library swaps, `__CUDA_ARCH__`. One wave64 run and one wave32 run are still both required by the gates. They will exercise the same host path, and a pass on both is expected.

## File-by-file change list

- `CMakeLists.txt`: HIP fallback branch as above. Targets, test registration and the example components stay as they are. The ROCm path links `hip::host` where CUDA links `${CUDA_LIBRARIES}`, exports `hip` and `__HIP_PLATFORM_AMD__`, and adds no new option.
- `include/cuda_blackboard/cuda_runtime_compat.hpp` (new): CUDA include on CUDA; HIP include plus `#ifndef`-guarded aliases on HIP.
- Include-line swaps (one line each): `include/cuda_blackboard/cuda_adaptation.hpp`, `cuda_error.hpp`, `cuda_blackboard_subscriber.hpp`, `cuda_mem_pool_context.hpp`, `cuda_unique_ptr.hpp` (two includes become one); `src/cuda_message_impl_detail.hpp`, `src/cuda_message_impl_detail.cpp`, `src/cuda_image.cpp`, `src/cuda_pointcloud2.cpp`; `test/cuda_test_utils.hpp`, `test/cuda_blackboard_node_fixture.hpp`.
- `README.md`: one short "AMD GPUs (ROCm)" paragraph beside the NVIDIA requirement sentence: the package also builds on ROCm with the same `colcon build` command when CUDA is absent and HIP is found, and, for a ROCm install outside `/opt/rocm`, it is found with `--cmake-args -DCMAKE_PREFIX_PATH=<rocm prefix>`. No device-selection variables in the example commands.
- No other source change is expected. A fix found under R1 to R4 gets its own commit, HIP-guarded only if CUDA behaviour would otherwise change.

Upstream text uses HIP for the runtime/API and ROCm for the platform; commit title prefix `[ROCm]`.

## Build commands

ROS 2 first (see R6). On Ubuntu 24.04 hosts, ROS 2 Jazzy. Upstream CI covers Humble and Jazzy; Jazzy is enough for the gate. Use an environment inside the job directory so nothing shared on the host changes:

- Preferred: RoboStack (conda-forge ROS 2 Jazzy, `robostack-jazzy` channel, reachable from the fleet) with micromamba or pixi under the job directory. Packages needed: `ros-jazzy-ros-base`, `ros-jazzy-rclcpp-components`, `ros-jazzy-cv-bridge`, `ros-jazzy-sensor-msgs`, `ros-jazzy-std-msgs`, `ros-jazzy-ament-cmake-auto`, `ros-jazzy-ament-cmake-gtest`, `ros-jazzy-ament-lint-auto`, `ros-jazzy-launch-ros`, plus `colcon-common-extensions` and `vcstool`. `negotiated` / `negotiated_interfaces` come from source (pinned in `build_depends.repos`).
- Alternative: the official apt packages (packages.ros.org, `ros-jazzy-ros-base` + the same list). This installs into the host container's system paths, which other jobs share. Use it only where a person has said so for that host.
- If the lint packages are missing, `-DBUILD_TESTING=ON` still needs `ament_lint_auto`. Install it rather than dropping the tests.

Workspace (paths relative to a job-local workspace `ws/`):

```bash
mkdir -p ws/src && cd ws
git clone -b moat-port https://github.com/AMD-Ecosystem/cuda_blackboard src/cuda_blackboard
vcs import src < src/cuda_blackboard/build_depends.repos
source <ros2-jazzy>/setup.bash
colcon build --packages-up-to cuda_blackboard \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=<rocm prefix>
```

`<rocm prefix>` is `/opt/rocm` on a system ROCm, or `$(rocm-sdk path --root)` on a pip ROCm SDK host. Check that `hip-config.cmake` is found under it. Confirm in the configure log that the HIP branch was taken and that the "cuda was not found" warning did not fire. A silently CPU-only build (only `test_publish_mailbox`) is the failure to catch.

## Test plan

GPU gate (both must pass, not skip, on each validated platform; a skip means no device was seen):

```bash
colcon test --packages-select cuda_blackboard --event-handlers console_direct+ \
  --ctest-args -R 'test_cuda_blackboard_(publisher|async_publisher)'
colcon test-result --verbose --test-result-base build/cuda_blackboard
```

- `test_cuda_blackboard_publisher`: 2 tests (counts before negotiation; worker-thread publish reaches the subscriber after negotiation).
- `test_cuda_blackboard_async_publisher`: 1 test (async publisher delivers until destroyed).
- Run each 20 times (`--ctest-args --repeat until-fail:20`) because R2 failures are timing-dependent.

Non-GPU regression set: `test_publish_mailbox` (4 tests) and the ament lint tests registered by `ament_lint_auto_find_test_dependencies()`, both passing as on CUDA. Lint failures that also occur on the unmodified upstream tree are not regressions; compare against an upstream-tip build in the same environment.

The upstream GPU tests cover only PointCloud2 over a created stream. To cover what they miss, the validator also runs:

1. **Example pipeline.** `ros2 launch cuda_blackboard example.launch.py image_path:=<a png>` publishes a CudaImage. The subscriber copies it back and republishes `output_image`. Check that `ros2 topic echo --once /output_image` shows the right resolution, and compare the bytes to the input image (a short rclpy script in `agent_space/`). This covers CudaImage, the ROS -> CUDA type adaptation and a `cudaStreamCreate` default stream.
2. **Probe** (`agent_space/`, never committed) linked against the installed package. It covers: (a) R1: a `CudaBlackboardSubscriber` built through the deprecated constructor (legacy stream) receives data and frees it without error over 1000 messages; (b) R3: with the default 1 GiB threshold, allocate and free 100 x 4 MiB messages and record `hipMemPoolAttrReservedMemCurrent` after the frees plus a synchronize, and the mean `make_unique` latency of the second 50 against the first 50. Record the numbers; a pool that kept nothing is the known ROCm behaviour, not a port failure. Also check that `CUDA_BLACKBOARD_MEM_POOL_RELEASE_THRESHOLD_MB=64` reads back through `hipMemPoolGetAttribute`; (c) a ROS `PointCloud2` -> `CudaPointCloud2` -> ROS round trip is byte-identical for a 1 MiB cloud; (d) the process exits 0 (R4).
3. **Consumer check.** The throwaway ament consumer from the build section configures, builds and runs against the installed package.

Run under `ulimit -c 0`, pinned to one idle GPU.

CUDA path: before the port is presented, compile it with real nvcc on the mi355 fleet host's CUDA toolchain (host C++ against the CUDA runtime is enough here; no GPU run is needed). Use a ROS 2 workspace with `find_package(CUDA)` succeeding, and confirm the CUDA branch is taken and the build is unchanged. A HIP validator cannot see a CUDA regression.

Agnocast variant: see the open question in notes.md. `test_start_worker_thread` needs no GPU and no Agnocast kernel module, but it does need the `agnocast_cie_thread_configurator` and `autoware_agnocast_wrapper` packages at build time with `ENABLE_AGNOCAST=1`.

## Open questions

1. (For Jeff, recorded at the top of the planner section of notes.md) Agnocast build variant: validated on ROCm, or covered by construction only.
2. (Coordination, no ruling needed) The compat-header alias expansions chosen here should be reused by the autoware_cuda and autoware_universe porters, so the macro sets agree when a consumer includes both. Their plans are not written yet. This plan's notes record the list for them to read.
