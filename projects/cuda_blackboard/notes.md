# cuda_blackboard notes

## Intake screen (2026-11-11, intake agent, windows-gfx1151 host)

Context: Jeff Daily ruled on 2026-11-11 that autoware_universe is taken up at full scope and that its two out-of-repo hard dependencies, autowarefoundation/autoware_cuda and this repository, become their own MOAT projects. This screen is the intake for that ruling. Upstream screened at `main` (clone with `--recurse-submodules` in `agent_space/cuda_blackboard-screen`; there are no submodules and no vendored directories).

### Licence

- `python3 utils/licenses.py check autowarefoundation/cuda_blackboard`: Apache-2.0, tier 1, cleared to contribute.
- Top-level `LICENSE` is the Apache License 2.0 text; `package.xml` says `<license>Apache 2</license>`.
- Per-file: most `src/`, `include/` and `test/` files carry no header at all (7 files carry a `Copyright 2024/2026 TIER IV, Inc.` Apache-2.0 header). Headerless files fall under the top-level LICENSE; nothing carries a different licence.
- `python3 utils/licenses.py scan-nvidia agent_space/cuda_blackboard-screen`: no NVIDIA proprietary licence text.
- Recorded with `set-license-spdx cuda_blackboard Apache-2.0`.

### Duplicate effort

- No upstream PRs, issues or branches matching `hip|rocm|amd|migraphx` (all states). README/docs grep: no AMD mention (the README says it "requires the use of NVIDIA GPUs and CUDA").
- No `AMD-Ecosystem/cuda_blackboard` or `ROCm/cuda_blackboard`. The 15 GitHub forks are TIER IV / Autoware contributors and integrators; none is an AMD port. `gh search repos` for autoware + rocm/migraphx: nothing.
- The autoware_universe screen notes an AMD Silo AI collaboration with the Autoware Foundation (2025-12-16); no code from it is visible on GitHub. Coordination is Jeff's call.

### What it is and the CUDA surface

ROS 2 package (C++17, ament_cmake_auto) implementing REP 2007 type adaptation plus REP 2009 type negotiation so GPU-resident `sensor_msgs` (PointCloud2, Image) pass between nodes in one process without a host round trip. Process-wide singletons: `CudaBlackboard<T>` (ticketed device-buffer dictionary) and `CudaMemPoolContext` (a stream-ordered memory pool). Version 0.5.0, actively maintained (last push 2026-10-27).

- No kernels, no `.cu` files, no CUDA libraries: CUDA runtime API only, through `cuda_runtime_api.h` (10 includes) and `cuda_runtime.h` (2). About 40 distinct runtime symbols: streams (`cudaStreamCreateWithFlags`, `cudaStreamNonBlocking`, `cudaStreamLegacy`, `cudaStreamGetDevice`), events (`cudaEventCreateWithFlags` with `DisableTiming`/`BlockingSync`, `cudaEventRecord`, `cudaStreamWaitEvent` with `cudaEventWaitDefault`), async copies and memset, pinned host memory, and the stream-ordered allocator (`cudaMemPoolCreate`, `cudaMemPoolSetAttribute(ReleaseThreshold)`, `cudaMallocFromPoolAsync`, `cudaFreeAsync`).
- ROCm mapping: every symbol has a direct HIP equivalent; checked in the current TheRock HIP headers: `hipStreamLegacy`, `hipEventWaitDefault`, `hipStreamGetDevice`, `hipMallocFromPoolAsync`, `hipMemPoolAttrReleaseThreshold`, `hipEventBlockingSync` are all present. Shape: Strategy A with a compat header (or shim header), since the sources are host C++ that never reach a device compiler; HIP runtime from host C++ needs only `__HIP_PLATFORM_AMD__` and `hip::host`.
- `cudaStream_t` appears in the public headers (consumers in autoware_universe pass their streams in), so the compat header must be installed with the package and consumers must see the same spelling.
- Build change: `find_package(CUDA)` (legacy FindCUDA) gates the whole library; with no CUDA it prints a warning and builds only the CPU mailbox test. The port needs a HIP branch (`find_package(hip)`) beside it, keeping the CUDA path unchanged.

### Risks

- The stream-ordered memory pool and the legacy-default-stream fallback (`cudaStreamLegacy` vs `nullptr`) are the semantics most likely to differ; HIP's pool release-threshold and legacy-stream behaviour should be checked by the GPU tests, not assumed.
- Event-based cross-stream synchronisation (`cudaStreamWaitEvent`) carries the zero-copy correctness argument; a missing wait shows up as stale data, not a crash.
- Validation needs a ROS 2 (Humble or Jazzy) environment plus `negotiated`/`negotiated_interfaces` (osrf/negotiated, pinned in `build_depends.repos`), `cv_bridge`, `rclcpp_components`. Fleet images do not carry ROS 2, so the validator needs a ROS 2 install or container on each Linux host.
- Optional Agnocast path (`ENABLE_AGNOCAST=1`, `autoware_agnocast_wrapper`) is off by default; out of scope unless the planner says otherwise.

### Build and tests

- Build: `colcon build --packages-up-to cuda_blackboard` in a ROS 2 workspace with `build_depends.repos` imported (`vcs import`).
- Tests (`colcon test`): `test_publish_mailbox` (header-only, CPU); `test_cuda_blackboard_publisher` and `test_cuda_blackboard_async_publisher` need a GPU (skip without one); `test_start_worker_thread` only with Agnocast. The two GPU gtests are the gate.

### Dependencies

- Does NOT depend on autoware_cuda: it carries its own `cuda_unique_ptr.hpp` and `cuda_error.hpp` and links only the CUDA runtime. No `set-deps` recorded.
- Non-CUDA ROS dependencies (`negotiated`, `cv_bridge`, `rclcpp`) are not MOAT projects.
- Consumer: autoware_universe depends on this project (recorded on that project, not here).

### Platform

Linux-only ROS 2 package; ROS 2 Autoware targets Ubuntu. The `windows` gate cannot reasonably be satisfied. Jeff approved a windows waiver for autoware_universe; for this project a waiver is only suggested (`suggest-waiver`), not approved.

### Recommendation

Fork. Small (about 25 source and header files, runtime API only, no kernels), Apache-2.0, no duplicate effort, and a hard dependency of the full-scope autoware_universe ruling. Expected port: compat header plus a HIP branch in CMakeLists.txt; the work is mostly in standing up ROS 2 on the validation hosts.

## Windows waiver (2026-11-11)

Jeff Daily approved the suggested `windows` waiver (Linux-only ROS 2 package), with the same ruling as autoware_universe.

## Planner (2026-11-20, planner agent, linux-gfx950 fleet job; no GPU used, nothing built)

### Decision needed from Jeff before a porter starts

**Question 1: is the optional Agnocast build variant in the validated scope?** Upstream has an off-by-default variant (`ENABLE_AGNOCAST=1`, needing the `agnocast_cie_thread_configurator` and `autoware_agnocast_wrapper` ROS packages) that routes the compatible ROS publisher and the async publisher's worker thread through Agnocast. It has no CUDA of its own, and the HIP port touches nothing in it. Its only test, `test_start_worker_thread`, is CPU-only and needs no Agnocast kernel module, only the two packages at build time. The porter's code changes are the same under either answer; only the validators' work differs.

- **Option A (recommended): covered by construction.** Validate the default build only. Record the Agnocast variant and `test_start_worker_thread` as `scoped_out` with the reason "optional, CPU-only, no HIP-specific code; not built on ROCm validators". Cheapest. The risk is that a ROCm-specific break in that variant goes unseen (unlikely, since no code there changes).
- **Option B: validate it too.** Each Linux validator also builds the Agnocast packages from source (Autoware/TIER IV repositories) and runs `test_start_worker_thread` with `ENABLE_AGNOCAST=1`. More ROS setup per host, for one CPU test.

Until this is ruled, both entries are listed in `surface.json` components but neither `covered` nor `scoped_out`, so the surface gate keeps the question open.

The project is recorded `planned`, so the selector may dispatch a porter before this is ruled. The porter's code is the same under A and B. If Jeff wants the porter to wait anyway, `moatlib.py set-hold cuda_blackboard` parks it.

No other ruling is needed: the port covers the whole package (library, installed headers, both example components, all tests). The windows gate is already waived.

### What was decided and why (details in plan.md)

- Build: pure CMake (ament_cmake_auto) with legacy `find_package(CUDA)`, so Strategy A, host-only. No `.cu` files, no kernels, no CUDA libraries: about 45 runtime-API symbols called from host C++. `set-ext-type cmake` recorded. Upstream analysed at `a4f330e5c228`.
- No `enable_language(HIP)` and no `LANGUAGE HIP`: sources stay CXX and link `hip::host`.
- The installed public headers use CUDA types and inline CUDA calls, so consumers need the mapping. Use a namespaced installed compat header (`include/cuda_blackboard/cuda_runtime_compat.hpp`), not shadow `cuda_runtime.h` headers: a shadow header in an ament-exported include directory would hijack consumers' own `cuda_runtime.h` shims.
- HIP is picked when FindCUDA fails and `find_package(hip CONFIG)` succeeds, so the documented `colcon build` works on ROCm with no flag and CUDA hosts are unchanged. Consumers get `hip` and `__HIP_PLATFORM_AMD__` through `ament_export_dependencies` / `ament_export_definitions`. A throwaway consumer package proves this.
- The `surface.py` census is empty (it scans only `.cu`/`.cuh`). The real surface is added by hand under `added_by: planner` (13 entries).

### Compat-header alias list (for this porter and the autoware_cuda / autoware_universe porters)

Each line is `#ifndef X` / `#define X Y`, using hipify's expansion so another package's compat header defines the same token sequence:

- Types: `cudaStream_t`->`hipStream_t`, `cudaEvent_t`->`hipEvent_t`, `cudaError_t`->`hipError_t`, `cudaMemPool_t`->`hipMemPool_t`, `cudaMemPoolProps`->`hipMemPoolProps`, `cudaMemcpyKind`->`hipMemcpyKind`.
- Constants: `cudaSuccess`, `cudaStreamNonBlocking`, `cudaStreamLegacy`, `cudaEventDisableTiming`, `cudaEventWaitDefault`, `cudaMemPoolAttrReleaseThreshold`, `cudaMemAllocationTypePinned`, `cudaMemLocationTypeDevice`, `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, `cudaMemcpyDeviceToDevice` -> the `hip` spelling of each.
- Streams/events: `cudaStreamCreate`, `cudaStreamCreateWithFlags`, `cudaStreamDestroy`, `cudaStreamSynchronize`, `cudaStreamGetDevice`, `cudaStreamWaitEvent`, `cudaEventCreateWithFlags`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventDestroy` -> `hip` spelling.
- Memory: `cudaMemPoolCreate`, `cudaMemPoolDestroy`, `cudaMemPoolSetAttribute`, `cudaMallocFromPoolAsync`, `cudaFreeAsync`, `cudaFree`, `cudaMemsetAsync`, `cudaMemcpy`, `cudaMemcpyAsync` -> `hip` spelling; `cudaMallocHost`->`hipHostMalloc` and `cudaFreeHost`->`hipHostFree` (not `hipMallocHost`/`hipFreeHost`; `hipMallocHost` is deprecated in HIP 7.16).
- Errors/device: `cudaGetErrorName`, `cudaGetErrorString`, `cudaGetLastError`, `cudaGetDeviceCount` -> `hip` spelling.

All present in the HIP 7.16 headers (ROCm SDK on the mi355 host), including `hipStreamLegacy` (`((hipStream_t)1)`), `hipEventWaitDefault`, `hipStreamGetDevice`, `hipMemPoolAttrReleaseThreshold`.

### Validation notes for the validator

- The gate is `test_cuda_blackboard_publisher` (2 tests) and `test_cuda_blackboard_async_publisher` (1). A skip means no device was seen, which is a failure here. Repeat 20 times. Non-GPU set: `test_publish_mailbox` (4) plus ament lint against an upstream-tip baseline.
- Upstream gtests miss CudaImage, ROS<->CUDA adaptation, the legacy-stream subscriber and the pool threshold. plan.md (Test plan) gives the example launch, a probe in `agent_space/` and the consumer check that cover them.
- ROS 2 Jazzy: prefer RoboStack under the job directory, since apt into the shared container is a person's call per host. `negotiated` comes from `build_depends.repos`. Pass `-DCMAKE_PREFIX_PATH=$(rocm-sdk path --root)` on pip ROCm SDK hosts (it holds `lib/cmake/hip/hip-config.cmake`).
- Pool release threshold: `CudaMemPoolContext` sets `ReleaseThreshold` before the pool's first allocation, which is exactly the case of the known HIP defect `hip-vmheap-unmap-threshold-lost-or-overtrimmed` (recorded in projects/LichtFeld-Studio/deferred.json; ROCm 10.1 Linux). Expect the pool to retain nothing. It is performance, not correctness: measure it (plan.md probe b), record the numbers here, and do not work around it unless review decides to.
- Confirm the configure log took the HIP branch. A build that only produced `test_publish_mailbox` fell back to CPU-only silently.

## Porter (2026-11-20, porter agent, linux-gfx942 fleet job; AMD Instinct MI300X, HIP_VISIBLE_DEVICES=4 pinned by the dispatcher)

### Open for a person (nothing below was decided by the porter)

1. **Agnocast variant (planner Question 1) is still open.** As instructed, the default build was ported and the variant was not built. In `surface.json`, `feature:agnocast` and `test:test_start_worker_thread` are recorded under `owed` with the reason "awaiting Jeff's ruling", not `scoped_out`. The surface gate needs every component accounted for once the stage is `ported`, and `owed` keeps the question open. If Jeff rules Option A, they move to `scoped_out`; if Option B, validators build the Agnocast packages.
2. **HIP runtime bug: file it upstream?** `hipEventRecord(ev, hipStreamLegacy)` returns success, and the next `hipStreamWaitEvent` on any stream segfaults. Registered as deferred `hip-event-record-stream-legacy-wait-crash` (rocm-bug-report, component clr). Filing on ROCm/rocm-systems is upstream-visible, so it is Jeff's call. The repro is below. The port works around it in its own HIP-guarded commit (9e0fb86), which the reviewer should judge separately from the main port commit.
3. **CUDA compile gate owed.** This host has no CUDA toolkit (no nvcc, no conda). Fetching micromamba for a job-local environment (for RoboStack or a CUDA toolkit) was refused by this session's permission policy, so I did not pursue it. The CUDA path must be compiled with real nvcc (the mi355 host) at 9e0fb86 before the port is presented. By reading the code, on CUDA: the compat header is `#include <cuda_runtime.h>` only; CMake's CUDA branch differs only by the `GPU_RUNTIME_*` variables, set from `CUDA_INCLUDE_DIRS`/`CUDA_LIBRARIES`; the subscriber fix is `#else cudaStreamLegacy`. That reading is not the gate.
4. **`[[nodiscard]]` warnings (reviewer's call, left as is).** HIP declares `hipError_t` return values `[[nodiscard]]` under C++17, so upstream's deliberately unchecked calls warn on ROCm with `-Wunused-result`: 11 sites in `src/cuda_mem_pool_context.cpp` (destructor), `src/cuda_image.cpp:51-52`, `src/cuda_pointcloud2.cpp:98-99`, `src/cuda_blackboard_subscriber_node.cpp:39` and `test/cuda_test_utils.hpp:28`. None are in a public header that consumers compile, and the package does not build with `-Werror`. Silencing them would touch CUDA-path lines (`(void)` casts), so the minimal port leaves them.

### What changed on the fork (`moat-port`, base upstream `a4f330e`)

- `f6ad66d` `[ROCm] Build with HIP on AMD GPUs when CUDA is not found`: new installed `include/cuda_blackboard/cuda_runtime_compat.hpp` (the planner's alias list exactly, `#ifndef`-guarded, `cudaMallocHost`->`hipHostMalloc` and `cudaFreeHost`->`hipHostFree`). It replaces 12 include lines (11 files). CMakeLists.txt gets a `find_package(hip CONFIG QUIET PATHS $ENV{ROCM_PATH} /opt/rocm)` fallback, `GPU_RUNTIME_INCLUDE_DIRS`/`GPU_RUNTIME_LIBRARIES` (the CUDA ones on CUDA, `hip::host` on HIP), plus `cmake/cuda_blackboard-hip-extras.cmake` registered through `${PROJECT_NAME}_CONFIG_EXTRAS` on HIP only. There is also a README paragraph.
- `9e0fb86` `[ROCm] Do not record the consumer event on hipStreamLegacy`: HIP-guarded. `addConsumerDependencyToFreeStream` records on `nullptr` (HIP's legacy default stream) instead of `hipStreamLegacy`.

Departures from plan.md, with the reasons:
- **No `ament_export_dependencies(hip)` / `ament_export_definitions`.** hip-config sets `hip_LIBRARIES` to `hip::host hip::device`, and ament's export-dependencies extras append it to `cuda_blackboard_LIBRARIES`. A consumer then gets `hip::device`'s `--offload-arch=gfx942`, and g++ fails with `unrecognized command-line option '--offload-arch=gfx942'` (seen with the throwaway consumer). The extras file instead runs `find_package(hip CONFIG REQUIRED ...)` and appends only `hip::host`. That target carries the include path and `__HIP_PLATFORM_AMD__` as interface properties, so consumers using ament_cmake_auto or ament_target_dependencies get both.
- **`include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})` became `${GPU_RUNTIME_INCLUDE_DIRS}`.** When FindCUDA fails, it leaves `CUDA_INCLUDE_DIRS` = `CUDA_TOOLKIT_INCLUDE-NOTFOUND`, and CMake's generate step fails with "Found relative path while evaluating include directories".
- **One source fix beyond the include swaps (R1, below).**

### Environment (job-local; nothing shared on the host was modified)

- ROCm: TheRock 10.2.0 wheels in `$FLEET_VENV`: `$FLEET_VENV/bin/pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple 'rocm[devel,device-gfx942]==10.2.0'`, then `rocm-sdk init`. `ROCM_SDK=$($FLEET_VENV/bin/python -m rocm_sdk path --root)`, HIP 7.17.26434. gcc 13.3, CMake 3.31.6, Python 3.11.17 (job venv).
- ROS 2 Jazzy **built from source** in the job directory. There was no ROS or conda on the host, the micromamba download was refused, and apt into the shared container is not allowed.
  - `vcs import --shallow src < ros2.repos` (jazzy branch of ros2/ros2), plus `vision_opencv` (branch `rolling`, cv_bridge 4.x) and `osrf/negotiated@eac198b5`.
  - Python tooling in the job venv: `colcon-common-extensions vcstool empy==3.3.4 lark catkin_pkg pyyaml pytest flake8 pydocstyle`.
  - Missing system libraries were worked around:
    - Boost 1.83 headers and OpenCV 4.6.0 (core, imgproc, imgcodecs, highgui; no GUI or video backends) built into `<job>/deps/prefix`.
    - Fast-DDS submodules `thirdparty/asio` and `thirdparty/tinyxml2` initialised, with `-DTHIRDPARTY_Asio=FORCE -DTHIRDPARTY_TinyXML2=FORCE`.
    - `COLCON_IGNORE` in `eclipse-iceoryx` (it needs `sys/acl.h`) and `-DENABLE_SHM=OFF` for Cyclone DDS.
    - `-DTRACETOOLS_DISABLED=ON` (no lttng-ust).
    - `-DCV_BRIDGE_DISABLE_PYTHON=ON` (no boost-python).
  - Build command:
    ```
    colcon build --merge-install --packages-up-to rclcpp rclcpp_components sensor_msgs std_msgs cv_bridge negotiated negotiated_interfaces launch_ros ros2launch ros2topic ros2run ros2pkg ament_cmake_auto ament_cmake_gtest ament_lint_auto ros_environment rmw_cyclonedds_cpp rmw_implementation rclpy --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DCV_BRIDGE_DISABLE_PYTHON=ON -DPython3_EXECUTABLE=$FLEET_VENV/bin/python -DTHIRDPARTY_Asio=FORCE -DTHIRDPARTY_TinyXML2=FORCE -DENABLE_SHM=OFF -DTRACETOOLS_DISABLED=ON
    ```
  - Result: 152 packages, about 15 min. Default RMW (Fast-DDS), `ROS_LOCALHOST_ONLY=1`.
- Every colcon build needs `-DPython3_EXECUTABLE=$FLEET_VENV/bin/python`, or ament's `catkin_pkg` import fails under another Python.

### Build

```
source <ros2_jazzy>/install/setup.bash
colcon build --packages-up-to cuda_blackboard --cmake-args -DCMAKE_BUILD_TYPE=Release "-DCMAKE_PREFIX_PATH=$ROCM_SDK;<job>/deps/prefix" -DPython3_EXECUTABLE=$FLEET_VENV/bin/python
```
rc 0. The configure log shows `-- cuda was not found, building cuda_blackboard with HIP 7.17.26434`; the "cuda was not found. the cuda_blackboard package will not be built." warning does not fire. All three libraries and both GPU tests are built (not CPU-only). The only warnings are the 11 `-Wunused-result` sites above, plus FindCUDA's CMP0146 dev warning (upstream's, also present on the upstream tip).

### Tests (all under `ulimit -c 0`, HIP_VISIBLE_DEVICES=4)

`colcon test --packages-select cuda_blackboard --event-handlers console_direct+ --ctest-args --repeat until-fail:20`, then `colcon test-result --verbose`:
- test_publish_mailbox 4/4, test_cuda_blackboard_publisher 2/2, test_cuda_blackboard_async_publisher 1/1, each 20 times: 60 `[  PASSED  ]` blocks, 0 failed, **0 skipped** ("Summary: 10 tests, 0 errors, 0 failures, 0 skipped"). This was run at the final tree (9e0fb86).
- No ament lint test is registered. package.xml has `ament_lint_auto` but no linter `test_depend`, so ctest lists only the 3 gtests, as on upstream.

Extra checks from plan.md's test plan (scripts in `agent_space/`, not committed):
1. **Example pipeline.** `ros2 launch cuda_blackboard example.launch.py image_path:=in.png` with a generated 64x48 RGB PNG. An rclpy subscriber on `/output_image` got 3/3 messages: `64x48 bgr8 step=192 bytes=9216`, byte-identical to the expected BGR of the input. The subscriber logged "Received message with resolution: 48 x 64".
2. **Probe** (`cbb_probe`, an ament package linked against the installed cuda_blackboard):
   - (c) 1 MiB PointCloud2 ROS->CUDA->ROS round trip through the constructor and both TypeAdapter directions: byte-identical.
   - (a) R1: subscriber built through the deprecated stream-less constructor (`cudaStreamLegacy`). Before 9e0fb86 the first delivered message segfaulted in `hipStreamWaitEvent` from `addConsumerDependencyToFreeStream`; gdb showed `pthread_mutex_lock(mutex=0x1d1)`, i.e. the `(hipStream_t)1` sentinel plus a member offset. After the fix: published 1003, received 1000, 0 data mismatches (each callback copies back on the null stream and compares with the seed-derived payload), and no sticky error.
   - (b) R3: default threshold reads back 1073741824 (1 GiB). Allocating 50 x 4 MiB with `make_unique` gives used 200 MiB / reserved 256 MiB while held. After freeing everything plus `hipStreamSynchronize(free_stream)` and `hipDeviceSynchronize`, reserved is **128 MiB** (the pool keeps one 128 MiB chunk, not the 1 GiB asked for). Mean `make_unique` latency: first 50 = 20.3 us, second 50 = 17.6 us. This run follows (a), which had already used the pool, so the first 50 are warm. With `CUDA_BLACKBOARD_MEM_POOL_RELEASE_THRESHOLD_MB=64` the threshold reads back 67108864, reserved after free is 0 MiB, and mean latency is 110.8 us (first 50, cold process) vs 20.3 us (second 50). This matches the known ROCm behaviour (`hip-vmheap-unmap-threshold-lost-or-overtrimmed`): performance only, with no correctness effect, and the planner said not to work around it.
   - (d) Every probe run exited 0 (no teardown crash, R4).
3. **Consumer check.** A throwaway `ament_cmake_auto` package with `-Wall -Wextra -Wpedantic -Werror` includes `cuda_blackboard/cuda_adaptation.hpp`, calls `make_unique<std::uint8_t[]>(16)` and copies it back. It builds and prints `consumer OK` (the memory reads back zeroed).
   - Runtime note: installed ROS executables carry no RUNPATH, and `libcuda_blackboard.so` NEEDs `libamdhip64.so.7`. With the pip ROCm SDK, `export LD_LIBRARY_PATH=$ROCM_SDK/lib:$LD_LIBRARY_PATH` is needed to run installed binaries (a system ROCm's `lib` is normally on the loader path). The colcon tests run from the build tree and need nothing.

### HIP hipStreamLegacy event repro (for the deferred bug report)

```cpp
#include <hip/hip_runtime_api.h>
#include <cstdio>
int main() {
  hipStream_t s; hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
  hipEvent_t ev; hipEventCreateWithFlags(&ev, hipEventDisableTiming);
  printf("record: %d\n", hipEventRecord(ev, hipStreamLegacy));  // 0 (hipSuccess)
  printf("wait: %d\n", hipStreamWaitEvent(s, ev, 0));           // segfault here
  return 0;
}
```
`g++ -std=c++17 -D__HIP_PLATFORM_AMD__ -isystem $R/include repro.cpp -L$R/lib -lamdhip64`:
- Segfaults with ROCm 10.2 (HIP 7.17.26434) and 10.1 (HIP 7.16.26385) on MI300X.
- The same program recording on `nullptr` or `hipStreamPerThread` runs clean.
- With `hipStreamLegacy` as the *waiting* stream it also runs clean, as do `hipMallocAsync`/`hipMemsetAsync`/`hipFreeAsync`/`hipStreamSynchronize` on `hipStreamLegacy`.

So only an event recorded on the legacy sentinel is broken.

### Lesson for cuda-to-rocm: an event recorded on hipStreamLegacy crashes the next hipStreamWaitEvent

For `faults-memory-streams.md`. `cudaStreamLegacy` maps to `hipStreamLegacy` (`(hipStream_t)1`). In HIP 7.16 and 7.17, `hipEventRecord(ev, hipStreamLegacy)` returns success, but a later `hipStreamWaitEvent(other, ev, 0)` dereferences the sentinel and segfaults. Other APIs accept it as a stream. Code that falls back to "record on the legacy default stream" (a typical pattern for consumers that pass no stream) should record on `nullptr` under HIP instead: the HIP null stream is the blocking legacy default stream unless the code is built for per-thread default streams. Upstream tests that always pass a created stream will not reach this path.

### Lesson for cuda-to-rocm: do not export the `hip` package to plain C++ consumers

For `build-cmake-link.md`. hip-config sets `hip_LIBRARIES` to `hip::host;hip::device`. Any mechanism that forwards a dependency's `_LIBRARIES` to consumers (ament's `ament_export_dependencies(hip)`, or hand-written config files that do the same) hands them `hip::device`. Its `--offload-arch=<gfx>` compile and link flags make g++ fail with `unrecognized command-line option`. A host-only package whose installed headers call the HIP runtime should forward only `hip::host`, for example from a config-extras file that runs `find_package(hip CONFIG REQUIRED)` and appends `hip::host`. That target carries the include directory and `__HIP_PLATFORM_AMD__`.
