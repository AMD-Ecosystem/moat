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
