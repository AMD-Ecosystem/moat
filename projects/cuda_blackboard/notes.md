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

## Review 2026-11-20 (reviewer agent, linux-gfx950 fleet job; no GPU run needed or made)

Scope: `moat-port` at 9e0fb865cd2e38dcc0db5e0e55497f40fb71f733 against upstream `a4f330e` (`git diff a4f330e...9e0fb86`, 16 files, +198/-18), reviewed with the pr-review skill. Every point below was checked again against the code (and, for the HIP defect, against the CLR source).

### Summary

The port maps the CUDA runtime calls of this host-only ROS 2 package to HIP through an installed `#ifndef`-guarded compat header, adds a `find_package(hip)` fallback to CMake that exports only `hip::host` to consumers, and works around a HIP event-record defect on the legacy stream. Verdict: **review-passed**. No code defect was found, and the CUDA path compiles with real nvcc and is unchanged from upstream (CUDA compile check below). One item has to be settled before an upstream PR, and only a person can settle it: DCO sign-off (first item under "Open for a person").

### Second commit (9e0fb86, the hipStreamLegacy event workaround), judged on its own

- **Root cause confirmed in CLR, not only by the porter's black-box repro.** In ROCm/rocm-systems `develop` (21a48eb, 2026-11-20), `projects/clr/hipamd/src/hip_event.cpp` `hipEventRecord_common` stores the raw handle with `e->SetCaptureStream(stream)`, and for `hipStreamLegacy` that is `(hipStream_t)1`. `hip_stream.cpp` `hipStreamWaitEvent_common` then reads that handle back. `hip::isValid()` returns true for `hipStreamLegacy`, so the handle is cast to `hip::Stream*`, and `eventStream->IsEventCaptured(event)` (the capture check after `hip::getStream`, and the earlier one when the waiting stream is null or legacy) dereferences address 1. That is the `mutex=0x1d1` the porter saw in gdb. `hipEventRecord_common` itself and `checkEventCaptureRestrictions` (used by query and synchronize) test for `hipStreamLegacy`; `hipStreamWaitEvent_common` does not. So only waiting on such an event is broken, which matches the repro. The defect is still present at the current develop tip. Add this to the deferred `hip-event-record-stream-legacy-wait-crash` report if it is filed.
- **The workaround is semantically right.** `hip::getStream()` (`hip_context.cpp:110`) resolves `nullptr` and `hipStreamLegacy` to the same `getNullStream()`, so recording on `nullptr` records on exactly the stream the CUDA path names. The library TU is compiled by g++ without `HIP_API_PER_THREAD_DEFAULT_STREAM`, so `hipEventRecord` (not `_spt`) gets `nullptr` = the legacy null stream whatever the consumer's own build mode is. The change is HIP-guarded, touches one function (`src/cuda_blackboard_subscriber.cpp:137-145`), and leaves the CUDA path byte-identical (confirmed by the CUDA build below). The other uses of `user_stream_` == `hipStreamLegacy` (`cudaStreamWaitEvent(user_stream_, ...)` at :90 and :122, and whatever the consumer enqueues on it) pass the sentinel as the *waiting*/*working* stream, which CLR maps to the null stream correctly. Correct as a separate commit; no change requested.
- Residual limitation (not a finding): a build of this library itself with per-thread default streams would make `nullptr` the per-thread stream, where CUDA's explicit `cudaStreamLegacy` stays legacy. Nothing in the package or its documented build does that.

### `[[nodiscard]]` warning sites (porter left them alone): agree, leave them

- Verified: HIP declares `typedef enum __HIP_NODISCARD hipError_t` with `[[nodiscard]]` under C++17 (`hip_runtime_api.h:277,289`); CUDA's `cudaError_t` is not nodiscard. The porter's count of 11 sites is right: `src/cuda_mem_pool_context.cpp:66,69,72,75,78` (destructor), `src/cuda_image.cpp:51-52` and `src/cuda_pointcloud2.cpp:98-99` (destructors), `src/cuda_blackboard_subscriber_node.cpp:39` (example node `cudaMemcpy`), `test/cuda_test_utils.hpp:28` (`cudaGetLastError()` used to clear the error).
- None is in an installed header that consumers compile: a consumer TU that instantiates `make_unique`, `make_host_unique<T[]>` and `make_host_unique<T>` builds clean with `g++ -std=c++17/-std=c++20 -Wall -Wextra -Wpedantic -Werror` against the HIP 7.16 headers, with the HIP include both `-isystem` and `-I`. The package builds without `-Werror`. The destructor calls are deliberately unchecked (they must not throw) and the `cudaGetLastError()` call exists only to reset state. Silencing them with `(void)` would edit nine CUDA-path lines that are warning-free on CUDA, which is unrelated churn in an upstream PR. The only one that hides a real error is the example node's unchecked `cudaMemcpy`, which is an upstream error-handling choice, not a port issue. Keep the minimal port as it is.

### Checks that found nothing to report

Compat-header coverage (every `cuda*` identifier in `include/`, `src/`, `test/` is mapped; `cudaEventBlockingSync` only appears in comments); `cudaMallocHost` -> `hipHostMalloc` resolves through HIP's C++ template overload (default flags) with the 2-argument CUDA call, and `hipHostMalloc` is not deprecated in HIP 7.16; event lifetime (R2): every `ready_event_` and the pool's wait event are recorded on the singleton pool stream, and the subscriber's event is waited on immediately after it is recorded; the CMake fallback cannot fire when `CUDA_FOUND`; `include_directories(SYSTEM ${GPU_RUNTIME_INCLUDE_DIRS})` with an empty list on HIP is harmless; commit titles (56 and 58 chars, `[ROCm]` on ROCm-specific commits), AI-assistance disclosure, no `Co-Authored-By`, `utils/jargon.py --port cuda_blackboard` clean; both lessons above are accurate against the CLR source and the porter's consumer evidence.

### CUDA compile check (CUDA 12.8, mi355 fleet host, at 9e0fb86)

Toolchain: CUDA 12.8 (`/usr/local/cuda`, nvcc V12.8.93, not on PATH by default), gcc 13.3, CMake 3.31.6, Ubuntu 24.04, Python 3.11.17 (job venv). No GPU is used: this is a compile check.

ROS 2 Jazzy, built from source in the job directory following the porter's recipe. This host has system Boost 1.83 and OpenCV 4.6 dev packages, so the job-local Boost/OpenCV prefix was not needed.

```
$FLEET_VENV/bin/pip install colcon-common-extensions vcstool 'empy==3.3.4' lark catkin_pkg pyyaml pytest flake8 pydocstyle numpy setuptools
curl -sSfL https://raw.githubusercontent.com/ros2/ros2/jazzy/ros2.repos -o ros2.repos
vcs import --shallow --retry 3 src < ros2.repos
git clone -b rolling --depth 1 https://github.com/ros-perception/vision_opencv src/vision_opencv   # cv_bridge 4.1.0, a56ad9a
git clone https://github.com/osrf/negotiated src/negotiated && git -C src/negotiated checkout eac198b55dcd052af5988f0f174902913c5f20e7
touch src/eclipse-iceoryx/COLCON_IGNORE
git -C src/eProsima/Fast-DDS submodule update --init --depth 1 thirdparty/asio thirdparty/tinyxml2
MAKEFLAGS=-j7 colcon build --merge-install --parallel-workers 6 --packages-up-to rclcpp rclcpp_components sensor_msgs std_msgs cv_bridge negotiated negotiated_interfaces ament_cmake_auto ament_cmake_gtest ament_lint_auto ros_environment rmw_implementation --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DCV_BRIDGE_DISABLE_PYTHON=ON -DPython3_EXECUTABLE=$FLEET_VENV/bin/python -DTHIRDPARTY_Asio=FORCE -DTHIRDPARTY_TinyXML2=FORCE -DENABLE_SHM=OFF -DTRACETOOLS_DISABLED=ON
```
Result: 138 packages, rc 0.

cuda_blackboard, built twice in separate workspaces with the same command: the fork at 9e0fb86 and, as a baseline, upstream at a4f330e:

```
export PATH=/usr/local/cuda/bin:$FLEET_VENV/bin:$PATH MAKEFLAGS=-j$MAX_JOBS
source <ros2_jazzy>/install/setup.bash
colcon build --event-handlers console_direct+ --packages-select cuda_blackboard --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DPython3_EXECUTABLE=$FLEET_VENV/bin/python -DCMAKE_VERBOSE_MAKEFILE=ON
```

Result: **pass, and the CUDA path is unchanged.**
- Configure: `-- Found CUDA: /usr/local/cuda (found version "12.8")`; the HIP status line and the "cuda was not found" warning do not appear. The ROCm 10.1 SDK in the job venv was present on the host and was not picked up, because `CUDA_FOUND` short-circuits the HIP lookup. The only CMake stderr is FindCUDA's CMP0146 dev warning and the gtest_vendor `cmake_minimum_required` deprecation, both identical on the upstream baseline.
- Compile: all 17 translation units (8 library sources, 2 example components and their 2 generated `node_main` files, 3 tests, gtest, gtest_main) compiled with `-isystem /usr/local/cuda/include`, `-Wall -Wextra -Wpedantic`: **0 compiler warnings** (the baseline also has 0). Link: `libcuda_blackboard.so` and the examples and tests link against `/usr/local/cuda/lib64/libcudart_static.a`, rc 0, the same as the baseline.
- Equivalence with upstream: the installed `share/cuda_blackboard/cmake/` directory is byte-identical to the baseline (no hip extras, nothing named hip), the exported dynamic symbol list of `libcuda_blackboard.so` is identical (1973 symbols), and the NEEDED list contains no HIP library. The installed headers differ only by the include swaps and the added `cuda_runtime_compat.hpp`.
- Consumer TU on CUDA: a file that includes the installed `cuda_blackboard/cuda_unique_ptr.hpp` and instantiates `make_unique<std::uint8_t[]>`, `make_host_unique<float[]>` and `make_host_unique<double>` builds with `g++ -std=c++17` and `-std=c++20 -Wall -Wextra -Wpedantic -Werror -isystem /usr/local/cuda/include`, and as a `.cu` with `nvcc -std=c++17 -Xcompiler -Wall,-Wextra,-Werror`: rc 0 for each.

### Open for a person (added by the reviewer; the porter's items 1, 2 and 4 above are unchanged, and item 3 is now answered by the CUDA compile check)

1. **DCO sign-off is required upstream, and the port commits have none.** Upstream PRs run the DCO app as a required-looking check (`DCO` is among the check runs on PR #29's head commit, together with `semantic-pull-request` and `require-label`), and 59 `Signed-off-by` lines appear in the last 20 upstream commits. Neither f6ad66d nor 9e0fb86 carries `Signed-off-by`, so an upstream PR from this branch would fail DCO. A sign-off is the contributor's own certification, so an agent should not add one in Jeff's name. Jeff's call: sign off (re-create the two commits with `Signed-off-by: Jeff Daily <jeff.daily@amd.com>`; the trees stay identical, so recorded validation can be carried over on the same-tree path, but replacing `moat-port` history needs a forced push, which only a person may authorize), or choose another route. The repository has no `.github/dco.yml`, so DCO remediation commits are presumably not enabled (unverified). Not a code finding, so it does not hold up validation, but it blocks publication.
2. **PR title format (for whoever drafts the upstream PR).** Upstream runs `semantic-pull-request`, and its PR titles are conventional commits (`fix(cuda_blackboard_publisher): ...`, `feat(cuda_blackboard): ...`). The upstream PR title needs that form (for example `feat: support AMD GPUs through HIP on ROCm`), not the `[ROCm]` commit title. `require-label` also expects a label, which is the maintainers' to apply.
3. Agnocast variant ruling (porter item 1) and whether to file the HIP defect (porter item 2, now with the CLR root cause above) remain open. This review decides neither.

## Validation 2026-11-20 (validator agent, linux-gfx942 fleet job; AMD Instinct MI300X, HIP_VISIBLE_DEVICES=3 pinned by the dispatcher)

Result: **pass at 9e0fb865cd2e38dcc0db5e0e55497f40fb71f733** (`moat-port`), built in `projects/cuda_blackboard/src`, tree clean. The Agnocast variant was not built and its two `owed` surface entries are unchanged (still Jeff's ruling).

### Environment (job directory only; nothing shared on the host was modified)

- ROCm: TheRock 10.2.0 wheels in `$FLEET_VENV` (`$FLEET_VENV/bin/pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple 'rocm[devel,device-gfx942]==10.2.0'`, `rocm-sdk init`), HIP 7.17.26434. No system ROCm on the host. gcc 13.3, CMake 3.31.6, Python 3.11.17.
- Boost 1.83 headers (`./b2 install --with-headers`) and OpenCV 4.6.0 (`-DBUILD_LIST=core,imgproc,imgcodecs,highgui`, no GUI/video/OpenCL/IPP backends, bundled png/jpeg/zlib/tiff/webp/openjpeg) into `<job>/deps/prefix`.
- ROS 2 Jazzy from source exactly as in the porter's section (same `vcs import`, vision_opencv `rolling` a56ad9a, negotiated eac198b, iceoryx `COLCON_IGNORE`, Fast-DDS asio/tinyxml2 submodules, the same `colcon build --merge-install --packages-up-to ...` command), with `CMAKE_PREFIX_PATH=<job>/deps/prefix`, `MAKEFLAGS=-j6 --parallel-workers 16`: 152 packages, rc 0, 3 min 26 s.
- Run environment: `source <ros2_jazzy>/install/setup.bash`, `LD_LIBRARY_PATH=$ROCM_SDK/lib:<job>/deps/prefix/lib`, `ROS_LOCALHOST_ONLY=1`; every GPU run under `ulimit -c 0`.
## Validation 2026-11-20 (validator agent, linux-gfx1100 fleet job; AMD Radeon Pro W7800 48GB, gfx1100, wave32, HIP_VISIBLE_DEVICES=2 pinned by the dispatcher)

Result: **pass at 9e0fb865cd2e38dcc0db5e0e55497f40fb71f733** (fork clone checked out at that commit, tracked tree clean). The Agnocast variant was not built and its two `surface.json` entries were left as they are (still Jeff's open question). The CUDA compile check is already recorded at this head (Review section), so it was not repeated. `utils/jargon.py --port cuda_blackboard`: clean. The README documents the ROCm build.

### Environment (all job-local; nothing shared on the host was modified)

- ROCm: TheRock 10.2.0 wheels installed into the job venv (they shadow the base env's 10.1 links): `$FLEET_VENV/bin/pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple 'rocm[devel,device-gfx1100]==10.2.0'`, then `rocm-sdk init`. HIP 7.17.26434. `ldd` on the test binaries resolves `libamdhip64.so.7` to the job venv's 10.2 SDK. gcc 13.3, CMake 3.31.6, Python 3.11.17 (job venv).
- ROS 2 Jazzy built from source in `<job>/ros2_jazzy` with the porter's recipe: `vcs import --shallow` of ros2/ros2 `jazzy` `ros2.repos`, `vision_opencv` `rolling` (a56ad9a), `osrf/negotiated@eac198b5`, `COLCON_IGNORE` in `eclipse-iceoryx`, Fast-DDS `thirdparty/asio` and `thirdparty/tinyxml2` submodules, and the same `colcon build --merge-install --packages-up-to ...` command and `--cmake-args` as in the Porter section (`MAKEFLAGS=-j2 --parallel-workers 5`, so the total stays at MAX_JOBS=10). 152 packages, 6 min 33 s, rc 0.
- This host has the system Boost 1.83 headers but no OpenCV, so OpenCV 4.6.0 (core, imgproc, imgcodecs, highgui; no GUI, video, IPP or OpenCL) was built into `<job>/deps/prefix` and that prefix was put on `CMAKE_PREFIX_PATH`.

### Build

```
colcon build --event-handlers console_direct+ --packages-up-to cuda_blackboard --cmake-args -DCMAKE_BUILD_TYPE=Release "-DCMAKE_PREFIX_PATH=$ROCM_SDK;<job>/deps/prefix" -DPython3_EXECUTABLE=$FLEET_VENV/bin/python
```
rc 0, 28.7 s. Configure: `-- cuda was not found, building cuda_blackboard with HIP 7.17.26434`; the "will not be built" warning does not appear. Built `libcuda_blackboard.so`, both example components, `test_publish_mailbox`, `test_cuda_blackboard_publisher`, `test_cuda_blackboard_async_publisher`. `ldd`: the library and the tests resolve `libamdhip64.so.7` from the job venv's `_rocm_sdk_devel/lib`. Warnings: only the 11 known `-Wunused-result` sites.

### Tests

```
colcon test --packages-select cuda_blackboard --event-handlers console_direct+ --ctest-args --repeat until-fail:20
colcon test-result --verbose
```
rc 0. Each of the 3 gtests 20 times: 60 `[  PASSED  ]` blocks, 0 `[  FAILED  ]`, 0 `[  SKIPPED ]`; "100% tests passed, 0 tests failed out of 3", "Summary: 10 tests, 0 errors, 0 failures, 0 skipped". Total test time 32.1 s.

On the GPU, not a fallback:
- `AMD_LOG_LEVEL=3 ./test_cuda_blackboard_publisher`: 2 tests passed; the log reports `Gfx Major/Minor/Stepping: 9/4/2, Device ID: 0x74a1` (MI300X) and real device work, all returning `hipSuccess`: 2 `hipMemPoolCreate`, 6 `hipMallocFromPoolAsync`, 6 `hipMemsetAsync`, 8 `hipMemcpyAsync`, 16 `hipEventRecord`, 10 `hipStreamWaitEvent`, 6 `hipFreeAsync` (log-line counts).
- The probe reports `count=1 name='AMD Instinct MI300X' arch=gfx942:sramecc+:xnack- pciBus=65`; `amd-smi list` puts BDF 0000:65:00.0 at GPU 3, the pinned device. `hipPointerGetAttributes` on a `CudaPointCloud2::data` gives `hipMemoryTypeDevice`.
- The package has no CPU path: the upstream tip (a4f330e) built in the same environment prints "cuda was not found. the cuda_blackboard package will not be built." and produces only `test_publish_mailbox`.

Non-GPU regression check: upstream a4f330e, same environment, `colcon test`: `test_publish_mailbox` 4/4 ("Summary: 5 tests, 0 errors, 0 failures, 0 skipped"). On the port it also passes 4/4 (20 times). No ament lint test is registered on either.

Extra checks from plan.md (throwaway packages in `agent_space/`, not committed; rewritten for this run since the porter's were container-local):
1. **Example pipeline.** `ros2 launch cuda_blackboard example.launch.py image_path:=in.png` with a generated 64x48 RGB PNG; an rclpy subscriber on `/output_image` got 3/3 messages `64x48 bgr8 step=192 bytes=9216`, each byte-identical to the expected BGR of the input. The subscriber component logged "Received message with resolution: 48 x 64".
2. **Probe** (`cbb_probe`, ament package against the installed cuda_blackboard), exit 0, `probe OK (fails=0)`:
   - (c) 1 MiB PointCloud2: constructor and both TypeAdapter directions, byte-identical.
   - (a) R1: subscriber through the deprecated stream-less constructor (`hipStreamLegacy`): published 1003, received 1000, 0 data mismatches, `hipGetLastError` = hipSuccess. (This is the path 9e0fb86 fixes.)
   - (b) R3: default threshold reads back 1073741824. 50 x 4 MiB held: used 200 MiB, reserved 256 MiB; after free + `hipStreamSynchronize(free_stream)` + `hipDeviceSynchronize`: reserved 128 MiB. Mean `make_unique` 15.4 us (first 50, warm after (a)) vs 15.2 us (second 50). With `CUDA_BLACKBOARD_MEM_POOL_RELEASE_THRESHOLD_MB=64` (cold process, pool test only): threshold reads back 67108864, reserved after free 0 MiB, mean 441.3 us (first 50) vs 17.3 us (second 50). Same picture as on the porter's run (known `hip-vmheap-unmap-threshold-lost-or-overtrimmed`, performance only).
   - (d) every probe run exited 0.
3. **Consumer check.** `cbb_consumer` (`ament_cmake_auto`, `-Wall -Wextra -Wpedantic -Werror`, includes `cuda_blackboard/cuda_adaptation.hpp`, `make_unique<std::uint8_t[]>(16)` and copy back): builds, prints `consumer OK`, rc 0.

### Other gates

- CUDA compile gate: already recorded at 9e0fb86 by the review (CUDA 12.8, mi355 host); not repeated.
- `python3 utils/jargon.py --port cuda_blackboard`: clean (needs local `main` and `moat-port` branches in the fork clone).
- Documentation: the README carries the ROCm build paragraph next to the CUDA requirement.
- `git -C projects/cuda_blackboard/src status --porcelain`: empty.
source <job>/ros2_jazzy/install/setup.bash
export ROCM_SDK=$($FLEET_VENV/bin/python -m rocm_sdk path --root) LD_LIBRARY_PATH=$ROCM_SDK/lib:<job>/deps/prefix/lib:$LD_LIBRARY_PATH ROS_LOCALHOST_ONLY=1
MAKEFLAGS=-j$MAX_JOBS colcon build --event-handlers console_direct+ --packages-select cuda_blackboard --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON "-DCMAKE_PREFIX_PATH=$ROCM_SDK;<job>/deps/prefix" -DPython3_EXECUTABLE=$FLEET_VENV/bin/python
```
rc 0, 24 s. Configure prints `-- cuda was not found, building cuda_blackboard with HIP 7.17.26434`, and the "will not be built" warning does not fire. Built: `libcuda_blackboard.so` (NEEDED `libamdhip64.so.7`), both example component libraries, `test_publish_mailbox`, `test_cuda_blackboard_publisher` and `test_cuda_blackboard_async_publisher`. The only compiler warnings are the 11 known `-Wunused-result` (`hipError_t` nodiscard) sites, plus FindCUDA's CMP0146 dev warning.

### GPU tests (`ulimit -c 0`)

```
colcon test --packages-select cuda_blackboard --event-handlers console_direct+ --ctest-args --repeat until-fail:20
colcon test-result --verbose --test-result-base build/cuda_blackboard
```
- ctest reports `100% tests passed, 0 tests failed out of 3`. Each test ran 20 times: 60 `Passed` ctest lines and 60 gtest `[  PASSED  ]` blocks (20 x 4 test_publish_mailbox, 20 x 2 test_cuda_blackboard_publisher, 20 x 1 test_cuda_blackboard_async_publisher), with **0 skipped** and 0 failed. test-result: `Summary: 10 tests, 0 errors, 0 failures, 0 skipped`.
- **The tests ran on the GPU, not on a fallback.** The package has no CPU path for these tests: without a device, the fixture skips them, and here none was skipped. `AMD_LOG_LEVEL=3 ./test_cuda_blackboard_publisher` (2/2 PASSED) shows `Enumerated GPU agents = 1` and `Gfx Major/Minor/Stepping: 11/0/0, Device ID: 0x7449`, with 3 `hipMallocFromPoolAsync`, 3 `hipMemsetAsync`, 4 `hipMemcpyAsync` (host to device, into device pool addresses), 3 `hipFreeAsync`, 8 `hipEventRecord` and 5 `hipStreamWaitEvent`. That is 62 calls returning `hipSuccess` and no call returning an error.
- Non-GPU baseline: upstream `a4f330e` built in the same environment builds only `test_publish_mailbox`, because CUDA is not found. It passes 20/20, with 0 failures. The port passes it 20/20 too, so there is no regression. No ament lint test is registered, on the port or upstream.

### Extra checks from plan.md's test plan (scripts in `agent_space/`, not committed)

1. **Example pipeline:** `ros2 launch cuda_blackboard example.launch.py image_path:=in.png` with a generated 64x48 RGB PNG. An rclpy subscriber on `/output_image` got 3 of 3 messages, each `64x48 bgr8 step=192 bytes=9216` and byte-identical to the expected BGR of the input. The subscriber node logged `Received message with resolution: 48 x 64` 4 times, and the launch log shows no errors.
2. **Probe** (`cbb_probe`, an ament package linked against the installed cuda_blackboard, built with `-Wall -Wextra -Wpedantic -Werror`). It prints `device: count=1 name='AMD Radeon Pro W7800 48GB' arch=gfx1100 warpSize=32`.
   - (c) 1 MiB PointCloud2 ROS -> CUDA -> ROS, through both the constructor and the TypeAdapter: byte-identical both ways. `hipPointerGetAttributes` reports the payload as device memory.
   - (a) Subscriber through the deprecated stream-less constructor (`cudaStreamLegacy`, the path the 9e0fb86 workaround covers): 1004 published, 1001 received after negotiation. Each payload is 64 KiB and seed-patterned. The callback copies it back with `hipMemcpy` on the null stream and compares it: 0 mismatches, every payload a device pointer, and `hipGetLastError` = hipSuccess at the end. The same run with a created non-blocking user stream (`hipMemcpyAsync` + `hipStreamSynchronize` in the callback): 1004/1001, 0 mismatches, hipSuccess.
   - (b) Pool, cold process, default threshold: reads back 1073741824. 50 x 4 MiB held = used 200 MiB / reserved 256 MiB. After freeing everything plus a sync of `free_stream` and the device, reserved is 128 MiB. Second round: reserved 256 MiB after free. Mean `make_unique` was 247.3 us for the first 50 (cold) and 32.0 us for the second 50. With `CUDA_BLACKBOARD_MEM_POOL_RELEASE_THRESHOLD_MB=64` it reads back 67108864, reserved after free is 0 MiB in both rounds, and the means are 259.7 us and 34.9 us. In the warm run inside the full probe, the default threshold gave 256 then 128 MiB reserved after free, and 78.3 / 35.5 us. So the pool keeps one or two 128 MiB chunks rather than up to 1 GiB, the same performance-only behaviour the porter measured on gfx942 (`hip-vmheap-unmap-threshold-lost-or-overtrimmed`). Correctness is unaffected.
   - (d) Every probe run exited 0 (no teardown crash).
3. **Consumer check:** a consumer TU that includes `cuda_blackboard/cuda_adaptation.hpp`, calls `make_unique<std::uint8_t[]>(16)` and copies it back builds with `-Werror` and prints `consumer OK`. As on any ROCm install outside `/opt/rocm`, the consumer configure needs the ROCm prefix on `CMAKE_PREFIX_PATH`, because the installed hip extras run `find_package(hip CONFIG REQUIRED PATHS $ENV{ROCM_PATH} /opt/rocm)`. The README's prefix note covers that case. Exporting `ROCM_PATH` would also work.

## Ruling on the planner's Question 1 (Jeff Daily, 2026-11-23)

**A, default build only.** The optional `ENABLE_AGNOCAST` variant is scoped out because it has no GPU code. `feature:agnocast` and `test:test_start_worker_thread` moved from `owed` to `scoped_out` in surface.json. (Jeff gave the same ruling for autoware_universe's Agnocast variant.)

## Ruling on DCO sign-off (Jeff Daily, 2026-11-25)

**Option (a).** After the upstream-merge round below passes review and validation, the port is squashed at PR prep into one commit that carries `Signed-off-by: Jeff Daily <jeff.daily@amd.com>`. Its tree must be identical to the validated head, and it is carried with `squash-carry-forward`. Jeff authorizes that one forced update of `moat-port`, since no upstream PR is open. The 2026-11-25 merge job does not squash and does not force-push: it only prepares the message (draft under "Squashed commit message (draft)" below) and records. This answers the reviewer's open item 1 of 2026-11-20.

## Porter 2026-11-25: merge upstream main (linux-gfx942 fleet job; AMD Instinct MI300X in CPX mode, HIP_VISIBLE_DEVICES=8 pinned by the dispatcher)

Upstream `main` moved past the port's base `a4f330e` with five commits of 2026-11-24, up to `e4e1206e30e8f119a833ac8b8d33bc04cfd58e5f`: #33 (sync codecov.yaml), #18 (sync files), #20 (pre-commit quarterly autoupdate), #34 (cppcheck-weekly renamed to cppcheck, sync `.cppcheck_macros.cfg`), #35 (sync files).

### Fork commits (`moat-port`, plain push 9e0fb86..b76dacb, no rebase, amend or force)

- `94e27aa` `Merge upstream main into the HIP port`: `git merge --no-ff upstream/main`, **no conflicts**. Checked: `git diff upstream/main 94e27aa` is exactly `git diff a4f330e 9e0fb86` (the port delta, unchanged; diffed after removing `index` lines and hunk offsets).
- `b76dacb` `lint` (message only, per the lint rule): clang-format include re-ordering in 9 files, +8/-13, include lines only.

### Upstream changes checked against the port

- 17 upstream files changed. 16 are CI/config only (`.github/`, `.pre-commit-config*.yaml`, `.prettierignore`, `codecov.yaml`, `.cppcheck_macros.cfg`). The `.cppcheck_macros.cfg` defines only Autoware agnocast-wrapper macros for cppcheck, so nothing in it concerns HIP.
- The only source change is `include/cuda_blackboard/cuda_unique_ptr.hpp`, the file the freshness check flagged: clang-format v23 reflows the `CUDA_BLACKBOARD_CHECK_CUDA_ERROR(::cudaMallocFromPoolAsync(...))` call (layout only, same tokens). The port touches this file only at the include block (lines 23-26), so git merged both without conflict.
- **No new CUDA runtime API calls.** Upstream added no new `cuda*` identifier, so the compat header needs no new mapping.

### Pre-commit (upstream's updated `.pre-commit-config.yaml`, run locally)

- `pre-commit run --files <the 16 files the port adds or modifies>` at the merge: **clang-format (v23.1.2) failed** on the port's swapped include lines. It re-sorts `#include "cuda_blackboard/cuda_runtime_compat.hpp"` into the quoted-include group, because the port replaced the angle-bracket `<cuda_runtime*.h>` lines in place. All other hooks passed.
- **This failure was already present at 9e0fb86, so the autoupdate did not introduce it.** clang-format 19.1.5 (the pre-update hook version, pip `clang-format==19.1.5`) re-sorts the same include lines in the same 9 files. The 2026-11-20 review missed it. The hook requires the fix either way, hence the `lint` commit.
- markdownlint-cli v0.49.1 crashes under the host's node v16 (`SyntaxError: Invalid regular expression flags` on a `/v` regex). That is an environment failure, not a lint result. With node 22.11.0 (official tarball, job directory) it passes.
- At b76dacb, `pre-commit run --all-files` passes every hook (check-json/xml/yaml, markdownlint, prettier, yamllint, flake8-ros, sort-package-xml, fix-include-guard, isort, black, clang-format, cpplint). Commands: `PRE_COMMIT_HOME=<job>/precommit-cache PATH=<node22>/bin:$FLEET_VENV/bin:$PATH pre-commit run --all-files`. The optional config (markdown-link-check) was not run.

### Environment (job directory only; nothing shared on the host was modified)

- ROCm: TheRock 10.2.0 wheels installed into the job venv, shadowing the base env's 10.1 links: `$FLEET_VENV/bin/pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple 'rocm[devel,device-gfx942]==10.2.0'`, then `rocm-sdk init`. `rocm-sdk path --root` = `$FLEET_VENV/lib/python3.11/site-packages/_rocm_sdk_devel`, HIP 7.17.26434 (`amdclang++` there is AMD clang 24.0.0git, but the build does not use it).
- **Compiler the build actually invokes: `/usr/bin/c++` = GNU g++ 13.3.0** (`CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++` -> `x86_64-linux-gnu-g++-13`) for every translation unit, with `-D__HIP_PLATFORM_AMD__=1` and the SDK include dir from `hip::host`. Configure prints `HIP: Using hipcc from relative path: .../_rocm_sdk_devel/bin/hipcc`. That is hip-config locating hipcc, and hipcc is never invoked (no device code). `libcuda_blackboard.so` NEEDs `libamdhip64.so.7`, which resolves to the job venv's 10.2 SDK.
- cmake 3.31.6, Python 3.11.17 (job venv), Ubuntu 24.04.
- Boost 1.83.0 headers (`./b2 install --with-headers`) and OpenCV 4.6.0 (`-DBUILD_LIST=core,imgproc,imgcodecs,highgui`, no GTK/Qt/FFmpeg/GStreamer/V4L/OpenCL/IPP, bundled png/jpeg/zlib/tiff/webp/openjpeg) built into `<job>/deps/prefix`. This host has neither.
- ROS 2 Jazzy from source in `<job>/ros2_jazzy`, the earlier recipe (ros2/ros2 `jazzy` `ros2.repos` with `vcs import --shallow`, `osrf/negotiated@eac198b5`, `COLCON_IGNORE` in `eclipse-iceoryx`, Fast-DDS `thirdparty/asio`+`tinyxml2` submodules, same `colcon build --merge-install --packages-up-to ...` command and `--cmake-args`, `MAKEFLAGS=-j6 --parallel-workers 16`): 152 packages, rc 0.
  - **vision_opencv must be pinned to `a56ad9a`.** `rolling` has moved on (tip `b4caad4` on 2026-11-25), and its cv_bridge now uses `std::endian` (C++20) and `sensor_msgs::image_encodings::NV12`, which Jazzy's sensor_msgs lacks, so it fails to build. Use `git clone -b rolling https://github.com/ros-perception/vision_opencv && git -C vision_opencv checkout a56ad9a`, not a shallow clone of the tip.
- Run environment: `source <job>/ros2_jazzy/install/setup.bash`, `LD_LIBRARY_PATH=$ROCM_SDK/lib:<job>/deps/prefix/lib`, `ROS_LOCALHOST_ONLY=1`, `MAKEFLAGS=-j$MAX_JOBS` (96), `ulimit -c 0`.

### Build and tests at b76dacb

```
colcon build --event-handlers console_direct+ --packages-up-to cuda_blackboard --cmake-args -DCMAKE_BUILD_TYPE=Release "-DCMAKE_PREFIX_PATH=$ROCM_SDK;<job>/deps/prefix" -DPython3_EXECUTABLE=$FLEET_VENV/bin/python -DCMAKE_VERBOSE_MAKEFILE=ON
timeout 1800 colcon test --packages-select cuda_blackboard --event-handlers console_direct+ --ctest-args --repeat until-fail:20
colcon test-result --verbose --test-result-base build/cuda_blackboard
```

- Build rc 0, 24 s. Configure prints `-- cuda was not found, building cuda_blackboard with HIP 7.17.26434`, and the "will not be built" warning does not fire. All 3 libraries and all 3 tests were built. The warnings are the same 11 known `-Wunused-result` sites (the lint commit moved some by one line: `cuda_image.cpp:50-51`, `cuda_pointcloud2.cpp:97-98`, `cuda_test_utils.hpp:29`).
- Tests rc 0: `100% tests passed, 0 tests failed out of 3`, 60 `[  PASSED  ]` blocks, 0 `[  FAILED  ]`, 0 `[  SKIPPED ]`, `Summary: 10 tests, 0 errors, 0 failures, 0 skipped`, 30.4 s.

### Lesson for moat-porting: `utils/timeit.sh` runs its command from the MOAT root

`timeit.sh` does `cd "$(dirname "$0")/.."` before running the wrapped command. A bare `colcon build` (or any tool that writes relative to the cwd) therefore writes `build/`, `install/` and `log/` into the MOAT checkout. Wrap a script that `cd`s to its workspace first (`timeit.sh <name> compile -- bash build.sh`), or pass absolute `--build-base`/`--install-base`.

## Review 2026-11-25 (reviewer role, linux-gfx942 fleet job; delta 9e0fb865...b76dacb)

Scope: `git diff 9e0fb865...b76dacb` (merge `94e27aa` of upstream `e4e1206`, plus `lint` `b76dacb`), reviewed with the pr-review skill. This job wrote the delta, so an independent child agent reviewed it first. I re-checked each of its claims below against the code. Verdict: **review-passed**. No code defect.

Verified:
- **The merge is faithful.** `94e27aa` has parents 9e0fb86 and e4e1206, and the merge base is a4f330e. `git diff upstream/main 94e27aa` is identical to `git diff a4f330e 9e0fb86` (16 files, +198/-18), and `git diff 9e0fb86 94e27aa` is identical to `git diff a4f330e upstream/main` (17 files, +145/-61). Nothing was dropped on either side.
- **No interaction with the port.** The only upstream source change is the token-identical reflow in `cuda_unique_ptr.hpp`. No new `cuda*` identifier, and CMakeLists.txt, package.xml and cmake/ are untouched. The child preprocessed the 8 library TUs at b76dacb in HIP mode (`g++ -E -D__HIP_PLATFORM_AMD__`, ROCm 10.2 headers). The only `cuda[A-Z]*` tokens left are inside log string literals, so every runtime name is mapped. None of the 42 macro names appears in the Jazzy install headers.
- **The `lint` commit is neutral on HIP.** It changes include and blank lines only, and its message is exactly `lint`. Every file still gets the compat header before its first CUDA name (directly, or through `cuda_error.hpp:4` / `cuda_mem_pool_context.hpp:4`). In 8 of the 9 files the moved line was already a guarded re-include. The one real move is `include/cuda_blackboard/cuda_blackboard_subscriber.hpp:5`, where compat now precedes negotiated/rclcpp/std_msgs. The child compared the preprocessed HIP output of all 8 TUs at 94e27aa and b76dacb: 7 are identical. `cuda_blackboard_subscriber.cpp` differs in 6 libstdc++ attribute lines (`__attribute__((__noinline__))` -> `__attribute__(())`), because HIP's `amd_detail/host_defines.h` does `#define __noinline__` (empty) for a non-HIP-clang compiler, and that TU now parses libstdc++ after it. I confirmed the define in the ROCm 10.2 header. GCC accepts both forms, and the effect is a lost noinline hint on cold libstdc++ paths. The other 7 TUs already had exactly this at the validated head, so this is consistency, not a new behavior. A consumer header's position relative to the compat macros is unchanged, and the `#ifndef` guards behave as before.
- **Commit messages.** The merge title is 37 chars, with an AI disclosure, a fenced Test Plan, no `Co-Authored-By`, and no internal vocabulary. `python3 utils/jargon.py --port cuda_blackboard`: clean.
- **Upstream pre-commit at b76dacb.** All hooks pass (Porter 2026-11-25 above). The child's clang-format 23.1.2 `--dry-run --Werror` over every .hpp/.cpp is also clean, and it confirmed that clang-format 19.1.5 flags the same 9 files at 9e0fb86.

Not a code finding, but owed: **the CUDA compile at b76dacb.** The nvcc check is recorded only at 9e0fb86. The lint commit moves `<cuda_runtime.h>` ahead of rclcpp/libstdc++ in the installed `cuda_blackboard_subscriber.hpp` on the CUDA path. Upstream's own `cuda_error.hpp` / `cuda_mem_pool_context.hpp` already include the CUDA headers first, so it should be neutral, but that is reasoning, not evidence. This host has no CUDA toolkit, so the check is listed under "Owed after this round" for the mi355 host.

Wording nits in the squash message draft (the child's F2-F4), applied to the draft below: "a consumer's own compatibility header" narrowed to a macro-based one that maps the same names (a consumer `using cudaStream_t = hipStream_t;` would become `using hipStream_t = hipStream_t;`); "given no stream" extended to "or the legacy default stream" (`src/cuda_blackboard_subscriber.cpp:157-160` tests `nullptr || cudaStreamLegacy`); "builds exactly as before" on CUDA narrowed to "with the same configuration and libraries" (the headers now include `cuda_runtime.h` where upstream used `cuda_runtime_api.h`).

Untested residual: upstream's cppcheck-differential job (now in a `ros:jazzy` container with `--library=.cppcheck_macros.cfg`) was not run, because this host has no cppcheck. It runs on the upstream PR.

### Squashed commit message (draft, for the PR-prep squash ruled 2026-11-25)

`python3 utils/prose.py` and `python3 utils/jargon.py` on the file: clean. The title is 42 chars. Before the squash is made, the Test Plan's claims must hold at the head being squashed: gfx1100 revalidated, and the CUDA 12.8 compile recorded at that head. If the head moves again, re-check them.

````
feat: support AMD GPUs through HIP on ROCm

cuda_blackboard only builds when CUDA is found, so it cannot be used on AMD GPUs. The package has no device code: every CUDA call is a runtime API call made from host C++, and each one has a direct HIP equivalent.

When find_package(CUDA) fails, CMakeLists.txt now looks for HIP (find_package(hip CONFIG), also searching ROCM_PATH and /opt/rocm) and, if found, builds the same targets linked against hip::host. Nothing goes through a device compiler and no new option is needed, so the documented colcon build works unchanged on a ROCm host, and a host with CUDA builds with the same configuration and libraries as before. On HIP the installed package configuration adds hip::host to the exported libraries, so consumers get the HIP include path and __HIP_PLATFORM_AMD__; it does not export hip::device, whose device-compilation flags a plain C++ compiler rejects.

A new installed header, cuda_blackboard/cuda_runtime_compat.hpp, includes cuda_runtime.h on CUDA. On HIP it includes hip/hip_runtime_api.h and maps the CUDA runtime names this package uses to their HIP equivalents, each as an #ifndef-guarded macro with the same expansion as hipify, so a consumer's own macro-based compatibility header that maps the same names does not conflict with it. It lives under cuda_blackboard/ rather than shadowing cuda_runtime.h, because this package's include directory is exported to every consumer. It replaces the CUDA includes in the headers, sources and tests.

On HIP, an event recorded on hipStreamLegacy crashes the next hipStreamWaitEvent that waits on it. The subscriber records its consumer-completion event on the legacy stream when it was given no stream or the legacy default stream, so on HIP it now records on the null stream, which is HIP's legacy default stream with the same blocking semantics. The upstream tests always pass a created stream and do not reach this path.

The README notes that AMD GPUs are supported through HIP on ROCm.

This change was written with the assistance of an AI coding agent.

Test Plan:

On an AMD Instinct MI300X (gfx942) and an AMD Radeon Pro W7800 (gfx1100), with ROCm 10.2 and ROS 2 Jazzy, the configure log reports the HIP build and all tests pass 20 times in a row:

```
colcon build --packages-up-to cuda_blackboard --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=<rocm prefix>
colcon test --packages-select cuda_blackboard --event-handlers console_direct+ --ctest-args --repeat until-fail:20
colcon test-result --verbose --test-result-base build/cuda_blackboard
```

A subscriber built with the constructor that takes no stream received 1000 PointCloud2 messages in the same process with no data mismatch. With CUDA 12.8 the same colcon build takes the CUDA branch and compiles with no warnings.

Signed-off-by: Jeff Daily <jeff.daily@amd.com>
````
