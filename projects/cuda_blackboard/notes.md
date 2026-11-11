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
