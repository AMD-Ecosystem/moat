# autoware_cuda notes

## Intake screen (2026-11-11, intake agent, windows-gfx1151 host)

Context: Jeff Daily ruled on 2026-11-11 that autoware_universe is taken up at full scope (TensorRT engine handling maps to MIGraphX) and that its two out-of-repo hard dependencies, this repository and autowarefoundation/cuda_blackboard, become their own MOAT projects. This screen is the intake for that ruling. Upstream screened at `main` (clone with `--recurse-submodules` in `agent_space/autoware_cuda-screen`; no submodules, no vendored directories).

### Licence

- `python3 utils/licenses.py check autowarefoundation/autoware_cuda`: Apache-2.0, tier 1, cleared to contribute.
- Top-level `LICENSE` is the Apache License 2.0 text; all three `package.xml` files say `Apache License 2.0`.
- Per-file: every `.cpp`/`.hpp`/`.cu` carries an Apache-2.0 header (TIER IV, Inc. 2022-2026; `find_cccl.cmake` The Autoware Foundation 2026). One NVIDIA file, `autoware_tensorrt_common/include/autoware/tensorrt_common/logger.hpp` (`Copyright (c) 2019, NVIDIA CORPORATION`), is under the Apache-2.0 header, which is clean. The four CMake files carry no header and fall under the top-level LICENSE.
- `python3 utils/licenses.py scan-nvidia agent_space/autoware_cuda-screen`: no NVIDIA proprietary licence text.
- Recorded with `set-license-spdx autoware_cuda Apache-2.0`.

### Duplicate effort

- No upstream PRs, issues or branches matching `hip|rocm|amd|migraphx` (all states); the autoware_universe screen ran the same searches with the same result. README/docs grep: no AMD mention (the README says the repo "holds only packages that need CUDA to build").
- No `AMD-Ecosystem/autoware_cuda` or `ROCm/autoware_cuda`; the one GitHub fork (xmfcx, an Autoware maintainer) is not an AMD port. `gh search repos` for autoware + rocm/migraphx: nothing. No MIGraphX backend for Autoware found anywhere.
- The autoware_universe screen notes an AMD Silo AI collaboration with the Autoware Foundation (2025-12-16); no code from it is visible on GitHub. Coordination is Jeff's call.

### What it is

A young repository (split out of autoware_universe per autowarefoundation/autoware#7340; last push 2026-11-10) holding Autoware's shared CUDA/TensorRT base packages. Three ament packages:

1. `autoware_cuda_dependency_meta`: CMake-only virtual package (no sources); its extras add `-fno-lto` to avoid duplicate `fatbinData` symbols. Trivial; the flag is harmless for HIP and may be needed for the same reason with HIP fat binaries.
2. `autoware_cuda_utils` (about 11 KB, header-only INTERFACE library): `cuda_check_error`, `cuda_unique_ptr`, `stream_unique_ptr`, `cuda_memory_pool`, `thrust_utils`, `cuda_gtest_utils`. CUDA runtime API (streams, `cudaMalloc`/`cudaHostAlloc`, `cudaPointerGetAttributes`, stream-ordered pool `cudaMemPoolCreate`/`cudaMallocFromPoolAsync`, `cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)`) plus Thrust (`device_vector`, `fill`, `count`, `thrust::cuda::par.on(stream)`). Links `CCCL::CCCL` through `cmake/find_cccl.cmake` (CUDA 13 CCCL layout).
3. `autoware_tensorrt_common` (about 3.3 KLOC host C++, no kernels): the `TrtCommon` engine wrapper every TensorRT node in autoware_universe uses. Requires TensorRT >= 10.

### CUDA surface and ROCm mapping

- `autoware_cuda_utils`: mechanical HIP port. All runtime symbols exist in HIP (`hipMemPoolCreate`, `hipMallocFromPoolAsync`, `hipPointerGetAttributes`, `hipSetDeviceFlags`); Thrust maps to rocThrust (`thrust::hip::par.on(stream)`), CCCL to rocThrust/hipCUB/libhipcxx. Known fault class to check: `hipPointerAttribute_t` field naming (`type`) and the managed/host memory-type values. Build: the test target uses legacy `find_package(CUDA)` + `cuda_add_executable` on `.cu` files and the library links `CUDA::cudart`; the port needs a HIP branch (`enable_language(HIP)`, `hip::host`, roc::rocthrust) and a replacement for `find_cccl.cmake`, keeping the CUDA path unchanged. Strategy A.
- `autoware_tensorrt_common`: no HIP analogue to translate into; this is a reimplementation with a defined target (MIGraphX), per Jeff's full-scope ruling. What a MIGraphX equivalent needs:
  - ONNX parse and compile: `nvonnxparser::createParser` + `IBuilder::buildSerializedNetwork` -> `migraphx::parse_onnx` + `program::compile(gpu target)`. Direct.
  - Engine cache: `.engine` serialization keyed by TensorRT version + JSON sidecar -> `migraphx::save`/`load` of the compiled program (`.mxr`), keyed by MIGraphX version and gfx target. Direct.
  - Precision: FP16 -> `quantize_fp16`; INT8 with `IInt8Calibrator` (implicit quantization, `tensorrt_conv_calib.hpp`) -> `quantize_int8` with calibration data; per-layer `setPrecision(kHALF)` on conv layers has no direct MIGraphX counterpart (it edits the network definition layer by layer). BF16/FP8 data types: partial.
  - Dynamic shapes: TensorRT optimization profiles (`kMIN/kOPT/kMAX` per input) -> MIGraphX dynamic dimensions (`onnx_options` dyn dims); semantics differ (MIGraphX compiles for ranges, with less mature dynamic-shape support), so the profile validation logic needs care.
  - Execution: `IExecutionContext::setTensorAddress` + `enqueueV3(cudaStream_t)` -> `program::run_async` with parameter maps on a `hipStream_t`. Direct in shape.
  - Plugins: `initLibNvInferPlugins` (standard TensorRT plugin library) has no MIGraphX equivalent; consumer models that rely on built-in TensorRT plugins need MIGraphX custom ops (`migraphx::experimental_custom_op`) or ONNX rewrites. The custom Autoware plugins live in autoware_universe (`autoware_tensorrt_plugins`, `autoware_tensorrt_bevformer`) and are that project's scope.
  - Profiling: `IProfiler` per-layer timing and `IEngineInspector` layer info -> MIGraphX `perf_report` / program print; approximate.
  - DLA (`setDLACore`, `DeviceType::kDLA`, `kGPU_FALLBACK`): Jetson-only, no AMD counterpart; stays CUDA-only.
  - API shape is the main design question: the public header exposes `nvinfer1::Dims`, `DataType`, `TensorIOMode`, `IBuilderConfig`, `INetworkDefinition` and `cudaStream_t`, and consumers in autoware_universe call these directly. Options for the planner: a parallel `autoware_migraphx_common` package with an equivalent API that consumers select at build time, or a backend-neutral layer under `TrtCommon` with TensorRT-free types. Either way the consumer side lands in autoware_universe, so the two projects' plans must agree on the interface.
- MIGraphX support is broadest on Instinct (gfx90a/gfx942/gfx950) and present but thinner on RDNA (gfx1100/gfx1201); the wave32 gate should be checked against MIGraphX's supported-target list before the planner commits to it.

### Risks

- The MIGraphX engine layer is new code, not a port, and the upstream (TIER IV maintainers) may prefer it as a separate package; whether it is offerable upstream as one PR is open.
- Dynamic-shape and INT8-calibration semantics differ from TensorRT; equal outputs on the dummy network do not prove the real perception models work.
- Validation needs ROS 2 (Humble/Jazzy) plus `autoware_cmake`, `autoware_lint_common`, `tensorrt_cmake_module` (the HIP path must not require it), and MIGraphX installed; fleet images carry none of ROS 2.
- `autoware_tensorrt_common` builds with `-Werror`; HIP/MIGraphX headers may trip warnings.

### Build and tests

- Build: `colcon build --packages-up-to autoware_cuda_utils autoware_tensorrt_common` in a ROS 2 workspace (the repo CI imports autoware_universe's `build_depends_*.repos` via `packages_above.repos`).
- Tests: `test_autoware_cuda_utils` (5 `.cu` gtest files: check_error, unique_ptr, utils, stream_unique_ptr, thrust_utils) needs a GPU; `autoware_tensorrt_common_test` (3 CPU `TensorInfoTest` cases plus 10 `TrtCommonIOTest` cases that parse `test/data/dummy_network.onnx` and build an engine, skipped without a GPU). A MIGraphX equivalent would need its own copy of the IO-negotiation tests against the same ONNX file.

### Dependencies

- No MOAT dependencies. Within the repo, `autoware_cuda_utils` and `autoware_tensorrt_common` depend on `autoware_cuda_dependency_meta`; the tensorrt_common test depends on `autoware_cuda_utils`.
- cuda_blackboard does not depend on this repository. autoware_universe depends on both (recorded on that project, not here). `packages_above.repos` names autoware_universe only so CI can build the downstream consumers; it is a reverse dependency, so no `set-deps` here.

### Platform

Linux-only ROS 2 packages; the `windows` gate cannot reasonably be satisfied, and DLA is Jetson-only. Jeff approved a windows waiver for autoware_universe; for this project a waiver is only suggested (`suggest-waiver`), not approved.

### Recommendation

Fork. Apache-2.0, no duplicate effort, and the shared base every GPU package in the full-scope autoware_universe ruling needs. `autoware_cuda_utils` is a small mechanical HIP port (first unit, unblocks the pure-CUDA autoware_universe packages); `autoware_tensorrt_common` needs a MIGraphX engine layer whose API must be designed jointly with the autoware_universe consumers (second unit).

## Windows waiver (2026-11-11)

Jeff Daily approved the suggested `windows` waiver (Linux-only ROS 2 package), with the same ruling as autoware_universe.

## Planning (2026-11-20, planner, linux-gfx950 fleet job; no GPU used, nothing built)

### Decisions for Jeff (rule before a porter starts)

Nobody could answer these during the planning job. plan.md is written for the recommended answer to each and says what changes otherwise.

**Q1. Shape of the MIGraphX engine layer (blocking; also fixes the interface autoware_universe builds against).** `TrtCommon`'s public API takes and returns TensorRT value types (`nvinfer1::Dims`, `DataType`, `TensorIOMode`, `cudaStream_t`), and autoware_universe's 16 consumer packages use them directly (240 uses of `nvinfer1::Dims`; the calibrators in yolox and classifier subclass `nvinfer1::IInt8*Calibrator`). Options:
- **A (recommended): same package, same classes, second backend.** On a ROCm build `autoware_tensorrt_common` compiles a new `src/migraphx_common.cpp` instead of `src/tensorrt_common.cpp` and installs HIP-only stand-in headers named `NvInfer.h` etc. that declare just the value types consumers use (written from scratch, not copied from NVIDIA). Consumers compile unchanged apart from `getNetwork()`/`getBuilderConfig()` users and direct TensorRT code. CUDA path: one `#if` around the class's private members and a code-motion split of the backend-neutral IO checks into `src/tensorrt_common_io.cpp`. Downside: a stand-in `nvinfer1` namespace in an Autoware package may not be welcome upstream.
- **B: new package `autoware_migraphx_common`** in this repository with Autoware-owned types and a mirrored API. CUDA package untouched; every autoware_universe consumer needs per-backend code (types and includes), a much larger diff there.
- **C: backend-neutral refactor:** replace the TensorRT types in `TrtCommon`'s public API with Autoware-owned ones on both backends. Cleanest design, but it changes the CUDA path's public API for every consumer and needs upstream's agreement before any code.

**Q2. INT8 on MIGraphX (blocking for the `int8` precision only).** TensorRT's path (`TrtConvCalib`: implicit-quantization calibrators, calibration-cache files, per-layer FP16 on conv layers, first/last-layer exclusion) has no one-to-one MIGraphX counterpart. Options:
- **a (recommended): implement int8 approximately.** Pull calibration batches from the caller's `IInt8Calibrator` (`getBatch`) into `migraphx::quantize_int8` calibration data, restrict quantized op types where possible with `add_op_name`, ignore per-layer/first/last-layer hints with a logged warning; TensorRT calibration caches are not reusable. Explicit Q/DQ ONNX models work natively. Real-model accuracy checks happen in autoware_universe (yolox, classifier).
- **b: no int8 on MIGraphX in this port.** `fp32`, `fp16` and `strongly-typed` only; `int8` fails with a clear error; int8 recorded as deferred work (`deferred.py add`).

**Q3. Delivery staging (blocking for scheduling).** autoware_universe's pure-CUDA stage (pointcloud preprocessor, ground segmentation, occupancy grid) depends only on `autoware_cuda_utils`, but the selector releases autoware_universe only once this whole project completes on a platform. Options:
- **i (recommended): two rounds, one PR.** Round 1 ports `autoware_cuda_dependency_meta` and `autoware_cuda_utils`, with every `autoware_tensorrt_common` item in surface.json moved to `owed` (accepted by the surface gate; `pr_ready` refuses while anything is owed, so no partial PR can open). Review and validate it, so autoware_universe's pure-CUDA stage unblocks. Round 2 adds the MIGraphX layer on the same branch (a new head, revalidation) before the single upstream PR. Round 2 has to be started deliberately (it will not be selected automatically once platforms show `completed`).
- **ii: one round.** Port both packages before review. Simpler bookkeeping; autoware_universe waits for the MIGraphX layer.
- **iii: two upstream PRs** (cuda_utils first, MIGraphX later). Partial-port PR; needs your explicit approval per the scope-cut rule.

**Q4. Proposed out of scope (confirm).** (a) TensorRT DLA (`dla_core_id`, `setDLACore`, `kGPU_FALLBACK`): Jetson hardware with no AMD counterpart; on MIGraphX a `dla_core_id != -1` request fails with an error. (b) TensorRT's built-in plugin library (`initLibNvInferPlugins`): no MIGraphX counterpart; models that need it are handled per model in autoware_universe (custom ops or ONNX rewrites). The `plugin_paths` dlopen mechanism itself stays, for MIGraphX custom-op libraries. These are listed as components in surface.json with "proposed out, pending Jeff (Q4)"; nothing is recorded as `scoped_out` until you rule.

### What was done

- Resumed from the intake section; no earlier planning existed. Took the planning lock (`set-state autoware_cuda linux-gfx950 planning --agent planner`).
- Read all of upstream at `a5929dd` (three packages, 4.4 KLOC). `set-ext-type autoware_cuda cmake`.
- Consumer census in autoware_universe at `8361f0e` (shallow clone, agent_space): `TrtCommon` methods used are `setTensorAddress` 105, `setInputShape` 78, `getTensorShape` 37, `setup` 29, `enqueueV3` 26, `getIOTensorName` 16, `setTensor` 12, `setTensorsAddresses` 11, `getNbIOTensors` 6, `getOutputDims` 5, `getInputDims` 3, `getTensorIOMode` 1, `getNetwork` 1 (camera_streampetr forces sigmoid/softmax to FP32), `getBuilderConfig` 1 (diffusion_planner). `TrtConvCalib` users: tensorrt_yolox, tensorrt_classifier. autoware_tensorrt_bevformer builds TensorRT engines directly, not through `TrtCommon`.
- Header checks against the ROCm 10.1 pip SDK on this host: rocThrust's HIP backend has no `thrust::cuda` namespace (`thrust::cuda::par` exists only in the CUDA-backend `thrust/system/cuda/detail/par.h`), so `thrust_utils.hpp` needs one guarded line; `hipHostAllocDefault`, `hipDeviceScheduleBlockingSync`, `hipMemPoolAttrReleaseThreshold`, `hipMallocFromPoolAsync` exist; `hipPointerAttribute_t` uses `type`.
- MIGraphX is not in the pip ROCm SDK (no `migraphx.hpp` or `libmigraphx*` on this host). MIGraphX C++ API names checked in `src/api/include/migraphx/migraphx.hpp` (develop): `parse_onnx(file, onnx_options)`, `onnx_options::set_dyn_input_parameter_shape`, `set_default_dyn_dim_value`, `program::compile(target, compile_options)`, `quantize_fp16`, `quantize_int8(prog, target, quantize_int8_options)` with `add_calibration_data`/`add_op_name`, `program::eval`, `program::run_async(params, Stream*)`, `migraphx::save`/`load`, `get_parameter_shapes`, `get_output_shapes`, `argument(shape, void*)`, `register_experimental_custom_op`.
- Wrote plan.md and surface.json (generated floor plus the hand-added tensorrt_common surface, which the census cannot see).

## Rulings on the planner's questions (Jeff Daily, 2026-11-23)

- Q1 (shape of the MIGraphX engine layer): **A**. Same package and classes with a second backend; stand-in `NvInfer.h` value-type headers written from scratch.
- Q2 (INT8 on MIGraphX): **a**. Approximate through `migraphx::quantize_int8` with the caller's calibration batches; per-layer hints ignored with a logged warning.
- Q3 (staging): **i**. Two rounds, one upstream PR. Round 1 ports `autoware_cuda_dependency_meta` and `autoware_cuda_utils` with the `autoware_tensorrt_common` surface owed; round 2 adds the MIGraphX layer on the same branch before the PR.
- Q4 (out of scope): **both cuts confirmed**. TensorRT DLA and TensorRT's built-in plugin library are scoped out; the `plugin_paths` mechanism stays.

## Porter round 1 (2026-11-23, porter agent, linux-gfx950 fleet job; AMD Instinct MI355X, HIP_VISIBLE_DEVICES=2 pinned by the dispatcher)

Scope, per Jeff's Q3 = i ruling and the dispatch: round 1 only. `autoware_cuda_dependency_meta` and `autoware_cuda_utils` are ported; every `autoware_tensorrt_common` surface entry is `owed` (round 2, dispatched separately after round 1 is reviewed and validated); the MIGraphX layer was not started. The two Q4 components are `scoped_out` in surface.json, citing the ruling.

### Open for a person (nothing below was decided by the porter)

1. **DCO sign-off.** Upstream runs the DCO app on every PR (it is listed, and failing, on upstream PR #7's head), and 16 of the last 20 upstream commit bodies carry `Signed-off-by`. Neither port commit (6cef725, 2290174) has one, because a sign-off certifies the contributor's own statement and an agent may not add it in Jeff's name. Options: (a) Jeff signs off when the branch is prepared for the PR (re-creating the commits with identical trees, which needs a history rewrite of `moat-port` that only a person may push, while validation carries over on the same-tree path); (b) Jeff adds sign-off from round 2 onward and handles round 1 the same way at PR prep. This is the same question the cuda_blackboard reviewer raised. It does not block review or validation, only publication.
2. **Copyright holder on the four new files.** `ament_copyright` (part of the package's lint tests) fails a file with no licence header, so the new files need one. They carry `Copyright 2026 The Autoware Foundation` plus the Apache-2.0 header, which matches the newest file upstream (`cmake/find_cccl.cmake`). If Jeff prefers another holder line (for example `Advanced Micro Devices, Inc.` or `The Autoware Contributors`, both seen in Autoware repositories), it is a one-line change per file in a new commit.
3. **Upstream defects seen, not fixed (outside the port; reporting them upstream is Jeff's call).** (a) `cuda_memory_pool.hpp` uses `uint64_t` without `<cstdint>`, so a TU that includes it first fails on CUDA (g++ and nvcc both reproduce it with CUDA 12.8; on HIP, `hip_runtime_api.h` happens to pull in `<stdint.h>`). (b) The pooled `make_unique<T>(n, stream, pool)` does not compile with an array `T` (`make_unique<float[]>(n, stream, pool)`: `T * ptr` with `T = float[]`), unlike the non-pooled overload; scalar `T` works, which is what consumers use. (c) `CMakeLists.txt` passes `SYSTEM ${GTEST_INCLUDE_DIRS}` inside `target_include_directories(... PRIVATE ...)`, so `SYSTEM` is taken as a directory name (ament_cppcheck receives `.../autoware_cuda_utils/SYSTEM`). Harmless; left alone.
4. **Spell check at PR time.** Upstream runs `spell-check-differential` (cspell with Autoware's dictionary). Words like `rocthrust`, `rocprim`, `nodiscard`, `hipcc` and `gfx` in the new files may need a `cspell:ignore` line or a dictionary entry. It could not be run locally, and it is cosmetic.

### What changed on the fork (`moat-port`, base upstream `a5929dd`)

- `6cef725` `fix(autoware_cuda_utils): discard cudaStreamDestroy result explicitly` (shared, no `[ROCm]` prefix): `(void)` on the deliberately unchecked `cudaStreamDestroy` in `StreamDeleter`. HIP declares `hipError_t` `[[nodiscard]]` under C++17, and `autoware_package()` sets `CMAKE_COMPILE_WARNING_AS_ERROR ON`, so every HIP consumer that includes `stream_unique_ptr.hpp` from a `.cpp` failed with `-Werror=unused-result` (reproduced with g++ 13.3 on the upstream header; the patched header compiles clean). No behavior change on CUDA.
- `2290174` `[ROCm] Build autoware_cuda_utils with HIP when CUDA is not found`:
  - `CMakeLists.txt`: `option(AUTOWARE_CUDA_USE_HIP ...)` (default OFF) skips `find_package(CUDA)`. When CUDA is not found, `find_package(hip CONFIG)` + `find_package(rocthrust CONFIG)` (QUIET, `PATHS $ENV{ROCM_PATH} /opt/rocm`); if neither toolkit is found, upstream's warning and `return()`. The CUDA branch keeps upstream's `find_cccl.cmake` + `CCCL::CCCL autoware_cuda_utils::cccl_system CUDA::cudart` and `cuda_add_executable`. The HIP branch adds `hip/include` to the INTERFACE include dirs (BUILD and INSTALL as `include/autoware_cuda_utils/hip`), links `autoware_cuda_utils::rocthrust_system`, installs the shim directory, `enable_language(HIP)`, compiles the 5 test `.cu` as `LANGUAGE HIP`, and adds `-Wno-unused-value -Wno-unused-result` to HIP sources of the test target only (upstream's tests leave `cudaStreamCreate/Destroy/Synchronize` unchecked in `test_thrust_utils.cu`; on CUDA, FindCUDA never passes `-Werror` to `.cu`). `CONFIG_EXTRAS` is a variable chosen per branch.
  - `cmake/find_rocthrust.cmake` (new; HIP CONFIG_EXTRAS): `find_package(hip)` + `find_package(rocthrust)`, and a `autoware_cuda_utils::rocthrust_system` INTERFACE IMPORTED marker holding rocThrust/rocPRIM include dirs (normal and SYSTEM) plus `hip::host`. Deliberately not `roc::rocthrust`, which links `roc::rocprim_hip` -> `hip::device` and puts `--offload-arch` on g++ consumers (the cuda_blackboard lesson).
  - `hip/include/cuda_runtime_api.h`, `cuda_runtime.h` (shim names, include `<hip/hip_runtime_api.h>` / `<hip/hip_runtime.h>`) and `cuda_runtime_hip_compat.h` (`#ifndef`/`#define` for exactly the 34 runtime names the package and its tests use (corrected from 38 after review finding 4); `cudaFreeHost` -> `hipHostFree`, `cudaHostAlloc` -> `hipHostAlloc`, which is not deprecated in HIP 7.17, `cudaPointerAttributes` -> `hipPointerAttribute_t`). The public headers are untouched apart from the two lines above.
  - `thrust_utils.hpp`: `thrust::hip::par.on(stream)` under `#if defined(THRUST_DEVICE_SYSTEM_HIP) && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP`, upstream line otherwise.
  - `README.md` of the package: an "AMD GPUs (ROCm)" section (the repository has no CUDA build doc; the package README is where the build is described).
- `autoware_cuda_dependency_meta`: no change. Its extras add `-fno-lto` to C/C++ flags only (`CMAKE_HIP_FLAGS` untouched); the HIP test and the HIP consumer build and link with it. Whether HIP fat binaries need `-fno-lto` for the duplicate-`fatbinData` reason was not exercised (no LTO in this build).

Departures from plan.md: the shim directory is `autoware_cuda_utils/hip/include/` (outside `include/`, installed only by the HIP branch, so a CUDA install ships nothing new), not `include/autoware/cuda_utils/hip/`. ament_auto installs `include/` wholesale, so the plan's location would have installed the shim on CUDA too. The compat header is `cuda_runtime_hip_compat.h`, not `cuda_to_hip.h`, to avoid a quoted-include clash with consumer compat headers. Option name as planned: `AUTOWARE_CUDA_USE_HIP` (round 2 and autoware_universe can test the same option).

### Environment (job-local; nothing shared on the host was modified)

- ROCm: TheRock 10.2.0 wheels in `$FLEET_VENV`: `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple 'rocm[devel,device-gfx950]==10.2.0'`, `rocm-sdk init`; `ROCM_SDK=$($FLEET_VENV/bin/python -m rocm_sdk path --root)`, HIP 7.17.26434. `export ROCM_PATH=$ROCM_SDK LD_LIBRARY_PATH=$ROCM_SDK/lib:$LD_LIBRARY_PATH`. gcc 13.3, CMake 3.31.6, Python 3.11.17 (job venv). The fleet guard only accepts a pip command that names `$FLEET_VENV` literally.
- ROS 2 Jazzy from source, only what round 1 needs (no rclcpp; round 2 needs it for `autoware_tensorrt_common`): `vcs import --shallow src < ros2.repos` (ros2/ros2 `jazzy`), `COLCON_IGNORE` in `eclipse-iceoryx`, then `colcon build --merge-install --packages-up-to ament_cmake_auto ament_cmake_gtest ament_lint_auto ament_lint_common ament_cmake_copyright ament_cmake_cppcheck ament_cmake_lint_cmake ament_cmake_xmllint ros_environment --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DPython3_EXECUTABLE=$FLEET_VENV/bin/python`: 45 packages, 13 s. Python tooling in the job venv: `colcon-common-extensions vcstool empy==3.3.4 lark catkin_pkg pyyaml pytest flake8 pydocstyle numpy setuptools`.
- `autowarefoundation/autoware_cmake` at `9849ad2` (autoware_cmake 1.4.0 + autoware_lint_common) in the package workspace.
- Lint tools missing on the host, built into `<job>/deps/prefix`: cppcheck 2.13.0 (`cmake -DUSE_MATCHCOMPILER=ON`) and libxml2 2.9.14 `xmllint` (static, no python/lzma/icu). Without them the `cppcheck` test fails ("Could not find 'cppcheck' executable") and `xmllint` is Disabled.

### Build (HIP)

```
colcon build --packages-up-to autoware_cuda_utils --cmake-args -DCMAKE_BUILD_TYPE=Release -DAUTOWARE_CUDA_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx950 -DCMAKE_PREFIX_PATH=$ROCM_SDK -DPython3_EXECUTABLE=$FLEET_VENV/bin/python
```
rc 0 from a clean workspace at 2290174, 0 compiler warnings (`-Wall -Wextra -Wpedantic -Werror` on HIP and CXX, `--offload-arch=gfx950`). Configure prints `-- cuda is not found, building autoware_cuda_utils with HIP 7.17.26434`. `-DAUTOWARE_CUDA_USE_HIP=ON` is needed on this host because FindCUDA finds `/usr/local/cuda` even with nvcc off PATH. Also checked: with `-DCUDA_TOOLKIT_ROOT_DIR=/nonexistent` and no option, the HIP branch is taken automatically; with neither toolkit found, upstream's "won't be built" warning fires. Without `CMAKE_HIP_ARCHITECTURES`, CMake detects gfx950.

### Tests (`ulimit -c 0`, HIP_VISIBLE_DEVICES=2)

```
colcon test --packages-select autoware_cuda_dependency_meta autoware_cuda_utils --event-handlers console_direct+ --ctest-args --repeat until-fail:20
colcon test-result --verbose
```
- `test_autoware_cuda_utils`: `[  PASSED  ] 17 tests.` on each of 20 runs (CudaCheckErrorTest 2, CudaUniquePtrTest 5, CudaUtilsTest 1, StreamUniquePtrTest 5, ThrustUtilsTest 4), 0 `[  SKIPPED ]`, 0 `[  FAILED  ]`. `AMD_LOG_LEVEL=3` shows `Gfx Major/Minor/Stepping: 9/5/0, Device ID: 0x75a3` and real work: 13 `hipLaunchKernel` (Thrust), 11 `hipMalloc`, 3 `hipHostAlloc`, 8 `hipStreamCreateWithFlags`, 2 `hipPointerGetAttributes` (the `type == hipMemoryTypeDevice/Host` checks pass), `hipStreamGetFlags` returns `hipStreamNonBlocking`.
- Lint: copyright 13/13, lint_cmake 3/3, xmllint 1/1. cppcheck: ament_cppcheck skips all 11 files ("Test Skipped due to cppcheck 2.13.0 performance issues"; 2.13 is Ubuntu 24.04's version, so upstream's Jazzy CI skips it too). Running cppcheck 2.13 directly (`-f --inline-suppr -q --language=c++ --std=c++17`) reports nothing on the port's 16 files or on upstream's 13.
- `test-result`: "Summary: 50 tests, 0 errors, 0 failures, 11 skipped" (the 11 are the cppcheck skips).
- No non-GPU regression is possible to measure on HIP against upstream: upstream returns early without CUDA and registers no tests.

Extra checks (throwaway packages under the job directory, not committed):
1. **Consumer package** (`ament_cmake_auto` + `autoware_package()`, so `-Werror`), against the installed HIP package, with a `FATAL_ERROR` if `CUDAToolkit_FOUND` is defined or `CUDA::cudart` exists. It does not fire. Three targets: a g++ `.cpp` via `ament_auto_add_executable`, the same `.cpp` via `ament_target_dependencies`, and a `.hip` via `target_link_libraries(autoware_cuda_utils::autoware_cuda_utils)`. The g++ TUs get `-D__HIP_PLATFORM_AMD__=1` and SYSTEM includes for the ROCm include dir, the package include dir and the shim dir, and no `--offload-arch`. All three run rc 0: a pooled `make_unique<float>(1M, stream, pool)` reads back all zero, `clear_async` zeroes exactly the first half, `make_unique_host` reports `hipMemoryTypeHost`, and `CHECK_CUDA_ERROR(cudaErrorInvalidValue)` throws "hipErrorInvalidValue (1)@...: invalid argument". thrust_stream `fill`/`fill_n`/`count` on a non-blocking stream gives c7=1047576, c5=1000.
2. g++ `-Werror` on a TU including upstream's `stream_unique_ptr.hpp` with the HIP shim: `error: ignoring returned value of type 'hipError_t', declared with attribute 'nodiscard' [-Werror=unused-result]`; with the port's header: rc 0 (the reason for 6cef725).

### CUDA compile check (CUDA 12.8, this host, at 2290174)

Toolchain: `/usr/local/cuda` nvcc V12.8.93 (not on PATH by default), gcc 13.3, CMake 3.31.6. No NVIDIA GPU here, so this is a compile and link check. The port (2290174) and upstream (a5929dd, a git worktree) were built in separate workspaces with the same command:
```
export PATH=/usr/local/cuda/bin:$FLEET_VENV/bin:$PATH
source <ros2_jazzy>/install/setup.bash
colcon build --event-handlers console_direct+ --packages-up-to autoware_cuda_utils --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCMAKE_VERBOSE_MAKEFILE=ON -DPython3_EXECUTABLE=$FLEET_VENV/bin/python
```
Result: **pass, the CUDA path is unchanged.**
- Both: `-- Found CUDA: /usr/local/cuda (found version "12.8")`, `Found CUDAToolkit ... 12.8.93`, CCCL from `/usr/local/cuda/lib64/cmake/cccl`; no HIP line, even though the ROCm 10.2 SDK is installed in the job venv. 10 nvcc invocations each (the 5 test `.cu` files), 0 warnings on both, and `test_autoware_cuda_utils` links (rc 0).
- Install trees: sha256 of every file identical except the two edited headers and the three workspace-path-dependent ament files (`parent_prefix_path`, `local_setup.sh`, `package.sh`). `share/autoware_cuda_utils/cmake/` (Config, export targets, `find_cccl.cmake`) is byte-identical, and the shim directory and `find_rocthrust.cmake` are not installed.
- Consumer TUs against the port headers on CUDA: a `.cpp` including `cuda_unique_ptr.hpp`, `cuda_memory_pool.hpp`, `cuda_utils.hpp`, `stream_unique_ptr.hpp` and instantiating the pooled `make_unique<float>` builds with `g++ -std=c++17` and `-std=c++20 -Wall -Wextra -Wpedantic -Werror -isystem /usr/local/cuda/include`, and the same plus a `thrust_stream::fill` call as `.cu` builds with `nvcc -std=c++17 -Xcompiler -Wall,-Wextra,-Werror`: rc 0 each. (With `cuda_memory_pool.hpp` first, both fail on the upstream `<cstdint>` defect, open item 3a.)

### For the validators

- Gate: `test_autoware_cuda_utils` 17/17 on every run, 0 skipped (a skip means `hipGetDeviceCount` saw no device, which fails the gate). Repeat 20 times. Confirm the configure line `building autoware_cuda_utils with HIP`; a build that only installs headers fell back to nothing.
- A host that also has a CUDA toolkit (like the mi355 host) needs `-DAUTOWARE_CUDA_USE_HIP=ON`.
- Lint needs `cppcheck` and `xmllint` on PATH, or those two tests fail or are disabled for environmental reasons on both the port and upstream.
- `autoware_tensorrt_common` is not part of this round. Do not build it (it returns early without TensorRT anyway); its surface entries are owed.

## Install as a dependency

autoware_universe builds against this repository as source in its colcon workspace (Autoware's `autoware.repos` clones it to `src/core/cuda/autoware_cuda`). For the ROCm build, clone `https://github.com/AMD-Ecosystem/autoware_cuda` at branch `moat-port` next to `autowarefoundation/autoware_cmake` in the workspace `src/`, and build with the ROCm prefix visible:

```
colcon build --packages-up-to autoware_cuda_utils --cmake-args -DCMAKE_PREFIX_PATH=<rocm prefix> -DCMAKE_HIP_ARCHITECTURES=<gfx> [-DAUTOWARE_CUDA_USE_HIP=ON]
```

`-DAUTOWARE_CUDA_USE_HIP=ON` is needed when a CUDA toolkit is also installed. Consumers configure with the same `CMAKE_PREFIX_PATH` (or `ROCM_PATH`), because the installed config extras (`find_rocthrust.cmake`) run `find_package(hip)` and `find_package(rocthrust)`. Linking `autoware_cuda_utils` (with ament_auto, `ament_target_dependencies` or `target_link_libraries(autoware_cuda_utils::autoware_cuda_utils)`) gives a consumer the shim `cuda_runtime_api.h`/`cuda_runtime.h` that map only the 34 runtime names this package uses. Each consumer brings its own compat header for the rest; the shim's macros are `#ifndef`-guarded, so identical definitions coexist. With a pip ROCm SDK, installed binaries need `$ROCM_SDK/lib` on `LD_LIBRARY_PATH`. The installed package records its backend: `share/autoware_cuda_utils/cmake/check_gpu_backend.cmake` sets `autoware_cuda_utils_USE_HIP` (TRUE for a HIP build, FALSE for CUDA) and stops a consumer's configure with a `FATAL_ERROR` when it is set up for the other backend (HIP build: the consumer found CUDA via `CUDA_FOUND`/`CUDAToolkit_FOUND` or enabled the CUDA language; CUDA build: the consumer set `AUTOWARE_CUDA_USE_HIP` or enabled HIP). The check runs at `find_package` and again at the end of the consumer's directory (`cmake_language(DEFER)`), since `autoware_package()` finds dependencies before consumers call `find_package(CUDA)`. So an autoware_universe consumer on a host with both toolkits must honor `-DAUTOWARE_CUDA_USE_HIP=ON` (skip `find_package(CUDA)` when it is ON) or key its backend off `autoware_cuda_utils_USE_HIP`.

`autoware_tensorrt_common` (round 2, head b60c8d7 and later) builds against MIGraphX on ROCm: `colcon build --packages-up-to autoware_tensorrt_common --cmake-args "-DCMAKE_PREFIX_PATH=<rocm prefix>;<migraphx prefix>" [-DAUTOWARE_CUDA_USE_HIP=ON]`, with `tensorrt_cmake_module` (tier4/tensorrt_cmake_module) in the workspace because the package still finds it first. MIGraphX is not in the pip ROCm SDK; build it from source with rocMLIR (recipe in "Porter round 2" below) and put its `lib/` on `LD_LIBRARY_PATH`. Consumers need only HIP from it (the config extras run `find_package(hip CONFIG REQUIRED PATHS $ENV{ROCM_PATH} /opt/rocm)`; MIGraphX is linked privately), and get stand-in `NvInfer.h`, `NvInferRuntime.h`, `NvOnnxParser.h` on their include path declaring `nvinfer1::Dims` (TensorRT 10 layout), `DataType`, `TensorIOMode`, `ILogger`, `IProfiler`, the `IInt8*Calibrator` interfaces, `cudaStream_t` (as `hipStream_t`) and opaque builder/network types. The ROCm install sets `autoware_tensorrt_common_USE_HIP` and runs the same backend check as autoware_cuda_utils; it exports no CUDA, CUDAToolkit or TensorRT dependency. What a consumer has to change: anything calling methods on `getNetwork()` or `getBuilderConfig()` (null on MIGraphX), direct TensorRT use outside `TrtCommon`, TensorRT plugins, `#include <NvInferPlugin.h>` (no stand-in), CMake gates on `TENSORRT_FOUND`/`TENSORRT_VERSION_MAJOR` (use `autoware_tensorrt_common_USE_HIP`), and inputs whose shape changes every frame (each new shape is compiled once; see the round-2 open item on padding).

## Review 2026-11-23 (reviewer, linux-gfx950 fleet job, round 1 at 2290174)

Scope: `git diff a5929dd...2290174` (6cef725, 2290174). Round-1 staging per Jeff's Q3 = i ruling: the owed `autoware_tensorrt_common` surface is intended and not reviewed here. **Verdict: changes requested** (findings 1-2 are code, 3-4 are text). Each finding was re-checked against the code and reproduced where marked.

### Findings

1. **A CUDA-configured consumer linking a HIP-built `autoware_cuda_utils` mixes the two runtimes, and nothing stops it** (`autoware_cuda_utils/cmake/find_rocthrust.cmake:18-37`, the installed config extra; `CMakeLists.txt:7-21`). The backend chosen at build time is not exported, so a consumer decides on its own, and on a host with both toolkits an overlay package configured without `-DAUTOWARE_CUDA_USE_HIP=ON` (an Autoware underlay/overlay setup; colcon only keeps `--cmake-args` consistent within one invocation) finds CUDA while linking the HIP package. Reproduced with a probe consumer (`find_package(CUDA)` + `ament_auto_find_build_dependencies()`, `-Werror` via `autoware_package()`) against the HIP install of 2290174, nvcc 12.8:
   - CUDA language (`enable_language(CUDA)`, `add_library(... k.cu)`): the nvcc TU resolves the real `/usr/local/cuda/.../cuda_runtime.h` (nvcc's implicit include dirs win over the `-isystem` shim dir), but the g++ `.cpp` TU of the same consumer resolves the shim, so `cudaStream_t` is `ihipStream_t*` there: `undefined reference to 'run(ihipStream_t*)'` at link.
   - FindCUDA (`cuda_add_library`, consumer adds `${CUDA_INCLUDE_DIRS}`): real CUDA headers win in both TUs and it links silently, but every TU gets `-D__HIP_PLATFORM_AMD__=1` and the rocThrust/HIP include dirs, and the executable links `libcudart_static.a` and `libamdhip64.so` together.
   So the shim does shadow the real CUDA header, in the host-compiler TUs of a CUDA consumer of a HIP build. Fix: in `find_rocthrust.cmake`, fail with a clear `FATAL_ERROR` when the consuming project has CUDA enabled (`CUDA_FOUND` from FindCUDA, or `CMAKE_CUDA_COMPILER_LOADED`), naming `-DAUTOWARE_CUDA_USE_HIP=ON`; and export the decision (for example `set(autoware_cuda_utils_USE_HIP TRUE)` in the same file) and document it under "Install as a dependency", so autoware_universe consumers key their backend off the package they link instead of re-detecting toolkits. The CUDA build of this package itself cannot pick up the shim: on CUDA it is neither on an include path nor installed (install-tree check below).
2. **An explicit `-DAUTOWARE_CUDA_USE_HIP=ON` with no ROCm found silently skips the package with the message "cuda is not found"** (`CMakeLists.txt:13-19`). Reproduced: option ON, `ROCM_PATH` unset, no ROCm on `CMAKE_PREFIX_PATH`, no `/opt/rocm`: `CMake Warning at CMakeLists.txt:17: cuda is not found, so the autoware_cuda_utils package won't be built.`, configure rc 0, nothing installed; the failure surfaces later in a dependent package. When the user asked for HIP, a missing `hip`/`rocthrust` should be a `FATAL_ERROR` naming HIP/rocThrust and `CMAKE_PREFIX_PATH`/`ROCM_PATH`; upstream's warn-and-skip stays for the automatic (option OFF) path.
3. **The 2290174 commit body overstates its Test Plan.** It says "the copyright, cppcheck, lint_cmake and xmllint tests pass"; cppcheck skips all 11 files ("Test Skipped due to cppcheck 2.13.0 performance issues", reproduced here: `cppcheck.xunit.xml` tests=11 skipped=11). Its build command also omits `-DAUTOWARE_CUDA_USE_HIP=ON`, which the stated test host (MI355X, CUDA also installed) needs; run as written there it takes the CUDA branch. Reword (cppcheck skipped by ament_cppcheck for 2.13; name the option for a host that also has CUDA). Rewording a commit body means re-creating the commit; fold it into the same-tree re-creation already open for the DCO sign-off (porter open item 1), which is Jeff's call, rather than a force-push now; the new commits from findings 1-2 should carry accurate bodies.
4. **Records: the compat header maps 34 runtime names, not 38** (notes "What changed on the fork" and "Install as a dependency"). Checked: the set of `cuda[A-Z]*` identifiers used in `include/` and `test/` equals the 34 `#define`d names in `hip/include/cuda_runtime_hip_compat.h` exactly (none missing, none extra). Correct the count.

### Checked, no finding

- CUDA 12.8 (`/usr/local/cuda`, nvcc V12.8.93, gcc 13.3, CMake 3.31.6, CCCL 2.7.0) at 2290174 and upstream a5929dd in separate workspaces, `colcon build --packages-up-to autoware_cuda_utils --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCMAKE_VERBOSE_MAKEFILE=ON` with `/usr/local/cuda/bin` on PATH: rc 0 both, `Found CUDA ... 12.8`, no HIP line, 10 nvcc invocations each, the same nvcc deprecated-target warning count on both and no compiler warning. Install trees: sha256 identical except the two edited headers and the three workspace-path files; no `hip/` directory and no `find_rocthrust.cmake` installed on CUDA.
- HIP headers (TheRock 10.2, HIP 7.17.26434) contain none of the 34 macro names, so defining them before `<hip/hip_runtime_api.h>` is safe; the only hits under `include/thrust` are rocThrust's CUDA-backend headers (`thrust/system/cuda/...`), which the HIP device system never includes. No HIP, rocThrust or rocPRIM header includes `cuda_runtime*.h`.
- Shim directory installs to `include/autoware_cuda_utils/hip/`; its file names do not collide with any `hip/...` header for a consumer whose include root is `include/autoware_cuda_utils` (merge-install).
- `thrust_utils.hpp`: `THRUST_DEVICE_SYSTEM`/`THRUST_DEVICE_SYSTEM_HIP` are defined by the thrust includes above the `#if`; CCCL defines no `THRUST_DEVICE_SYSTEM_HIP`, so CUDA keeps upstream's line.
- No kernels in the package, so no wavefront-width, texture or OOB fault class applies. Commit titles: 6cef725 shared fix without prefix, 2290174 `[ROCm]`, 64 chars; agent disclosed in prose, no trailer. `utils/jargon.py --port autoware_cuda`: clean.

### Evidence gathered during the review (not a validation record)

ROCm: TheRock `rocm[devel,device-gfx950]==10.2.0` installed into this job's venv (the venv's linked base env carried 10.1.0), HIP 7.17.26434. ROS 2 Jazzy ament tooling from source (45 packages, as in the porter section), cppcheck 2.13.0 and libxml2 2.9.14 xmllint built into `<job>/deps/prefix`, autoware_cmake 9849ad2. HIP build at 2290174 with the porter's command (`-DAUTOWARE_CUDA_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx950 -DCMAKE_PREFIX_PATH=$ROCM_SDK`): rc 0, `-- cuda is not found, building autoware_cuda_utils with HIP 7.17.26434`, 0 compiler warnings. `ulimit -c 0`, `HIP_VISIBLE_DEVICES=2`, `timeout 1800 colcon test --packages-select autoware_cuda_dependency_meta autoware_cuda_utils --ctest-args --repeat until-fail:20 --timeout 300`: `[  PASSED  ] 17 tests.` on 20 of 20 runs, 0 skipped, 0 failed; `colcon test-result`: 50 tests, 0 errors, 0 failures, 11 skipped (cppcheck). The validator was not run because the review requested changes; this run is for the porter's reference only.

### For a person

- Finding 3 needs the commit re-creation Jeff already has to rule on for the DCO sign-off (porter open item 1). Options: (a) reword the body when the branch is re-created with sign-off at PR prep (same tree, validation carries over); (b) leave the body as is and correct the Test Plan in the PR body only. Recommended: (a).

## Porter round 1 fixes (2026-11-23, porter agent, linux-gfx950 fleet job; MI355X, HIP_VISIBLE_DEVICES=2)

Answers "Review 2026-11-23". Still round 1 of Jeff's Q3 = i ruling: no MIGraphX work, `autoware_tensorrt_common` surface stays `owed`. New commits on `moat-port` (no amend/rebase/force): `20be73c`, `0842236`; head 2290174 -> 0842236.

### Per finding

1. **Mixed CUDA/HIP consumer** -> `0842236` `[ROCm] Record the GPU backend autoware_cuda_utils was built with`. New `cmake/check_gpu_backend.cmake.in` (ament configures `.cmake.in` extras with `@ONLY`), added to `ament_auto_package(CONFIG_EXTRAS ...)` on both backends; `CMakeLists.txt` sets `autoware_cuda_utils_USE_HIP` TRUE/FALSE in the branch. Installed file sets `autoware_cuda_utils_USE_HIP` and `FATAL_ERROR`s a consumer configured for the other backend (HIP build: `CUDA_FOUND`, `CUDAToolkit_FOUND` or CUDA in `ENABLED_LANGUAGES`; CUDA build: `AUTOWARE_CUDA_USE_HIP` or HIP in `ENABLED_LANGUAGES`). Checked immediately and deferred to the end of the calling directory (`cmake_language(DEFER CALL ...)`), because `autoware_package()` (autoware_cmake 9849ad2, `autoware_package.cmake:63`) runs `ament_auto_find_build_dependencies()` before a typical consumer's `find_package(CUDA)`, so an immediate check alone would miss the usual Autoware order. Not put in `find_rocthrust.cmake`/`find_cccl.cmake` because the package's own build `include()`s those; the new file only runs for consumers. README: consumers on a host with both toolkits need the option too, and what `autoware_cuda_utils_USE_HIP` means.
2. **Option ON, no ROCm** -> `20be73c` `[ROCm] Fail configure when HIP is requested but not found`: `FATAL_ERROR` naming HIP, rocThrust, `CMAKE_PREFIX_PATH`, `ROCM_PATH` when `AUTOWARE_CUDA_USE_HIP` is ON; the option-OFF path keeps upstream's warn-and-return.
3. **2290174 body overstates the Test Plan**: not reworded (needs a history rewrite, Jeff's call, listed under "For a person" in the review). The two new commit bodies state that ament_cppcheck skips with cppcheck 2.13 and name `-DAUTOWARE_CUDA_USE_HIP=ON` in the build command.
4. **34, not 38**: corrected in "What changed on the fork" and "Install as a dependency" (`grep -o '#define cuda[A-Za-z_]*' hip/include/cuda_runtime_hip_compat.h` lists 34).

### Environment (this job; nothing shared modified)

TheRock `rocm[devel,device-gfx950]==10.2.0` into `$FLEET_VENV` (HIP 7.17.26434), ROS 2 Jazzy ament tooling from source (`<job>/deps/ros2`, same package list as round 1), cppcheck 2.13.0 and libxml2 2.9.14 xmllint into `<job>/deps/prefix`, autoware_cmake 9849ad2. Note for later jobs: `utils/timeit.sh` `cd`s to the moat root, so a wrapped `colcon build` without `--base-paths/--build-base/--install-base/--log-base` builds in the moat checkout; the job wraps colcon with explicit bases. A plain-`cmake` ament probe needs `-DPython3_EXECUTABLE=$FLEET_VENV/bin/python` (catkin_pkg lives there) and a maintainer email catkin_pkg accepts.

### HIP build and tests at 0842236

```
colcon build --packages-up-to autoware_cuda_utils autoware_lint_common --cmake-args -DCMAKE_BUILD_TYPE=Release -DAUTOWARE_CUDA_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx950 -DCMAKE_PREFIX_PATH=$ROCM_SDK -DPython3_EXECUTABLE=$FLEET_VENV/bin/python
timeout 1800 colcon test --packages-select autoware_cuda_dependency_meta autoware_cuda_utils --event-handlers console_direct+ --ctest-args --repeat until-fail:20 --timeout 300
```
rc 0; `-- cuda is not found, building autoware_cuda_utils with HIP 7.17.26434`; installed `check_gpu_backend.cmake` has `set(autoware_cuda_utils_USE_HIP TRUE)`. `[  PASSED  ] 17 tests.` on 20/20 runs, 0 skipped/failed. `colcon test-result`: 51 tests, 0 errors, 0 failures, 11 skipped (cppcheck 2.13 skips; 51 vs 50 is lint_cmake now checking 4 files, the new `.cmake.in` included; copyright 13, cppcheck 11 skipped, xmllint 1, gtest 17). (`autoware_cmake` must be named in `--packages-up-to` or listed through `autoware_lint_common`; colcon did not pull it in for `--packages-up-to autoware_cuda_utils` alone here.)

### CUDA 12.8 compile check at 0842236 (nvcc V12.8.93, no NVIDIA GPU: compile/link only)

`PATH=/usr/local/cuda/bin:$PATH colcon build --packages-up-to autoware_cuda_utils autoware_lint_common --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCMAKE_VERBOSE_MAKEFILE=ON` in separate workspaces for 0842236 and 2290174: rc 0 both, `Found CUDA ... 12.8`, no HIP line, 10 nvcc invocations each, identical warning set (20 nvcc deprecated-target notes, 2 CMake dev warnings), no compiler warning, `test_autoware_cuda_utils` links. Install trees: sha256 identical except the workspace-path files, plus `share/autoware_cuda_utils/cmake/check_gpu_backend.cmake` (new, `set(autoware_cuda_utils_USE_HIP FALSE)`) and `autoware_cuda_utilsConfig.cmake` (lists the new extra).

### Reviewer's mixed-toolkit reproducer, rerun (probe package with `autoware_package()`, so `-Werror`)

Probe: `a.cpp` (makeCudaStream, `make_unique_host<float[]>`, calls `run(cudaStream_t, float*, int)`) + `k.cu` (iota kernel). nvcc 12.8 on PATH.

| autoware_cuda_utils install | consumer | 2290174 | 0842236 |
|---|---|---|---|
| HIP | `find_package(CUDA)` before `autoware_package()`, `enable_language(CUDA)` | links fail: `undefined reference to run(ihipStream_t*, float*, int)` (with `--compile-no-warning-as-error`; with `-Werror` nvcc's `-Wpedantic` line-directive error hits first) | configure `FATAL_ERROR`: "autoware_cuda_utils was built with HIP, but probe_consumer is configured for CUDA. Configure probe_consumer with -DAUTOWARE_CUDA_USE_HIP=ON, or rebuild autoware_cuda_utils with CUDA." |
| HIP | `autoware_package()` then `find_package(CUDA)`, `cuda_add_executable` (deferred path) | builds, links `cudart_static` and `amdhip64`, run aborts `cudaErrorInsufficientDriver` | same configure `FATAL_ERROR` |
| HIP | honors `-DAUTOWARE_CUDA_USE_HIP=ON`, `k.cu` as HIP | -- | builds, links only `amdhip64`, runs on GPU: `probe ok 1000 1023`, rc 0 |
| HIP | same probe without the option on this host (FindCUDA finds /usr/local/cuda) | -- | configure `FATAL_ERROR` (expected) |
| CUDA | CUDA consumer | -- | builds and links `cudart_static` (run needs an NVIDIA GPU) |
| CUDA | `-DAUTOWARE_CUDA_USE_HIP=ON`, HIP | builds to compile errors (`blockIdx` undeclared: real CUDA headers in a HIP TU) | configure `FATAL_ERROR`: "autoware_cuda_utils was built with CUDA, but probe_consumer is configured for HIP. ..." |

Finding 2: `cmake -S autoware_cuda_utils -B <b> -DAUTOWARE_CUDA_USE_HIP=ON` with `ROCM_PATH` unset, no ROCm on `CMAKE_PREFIX_PATH`, no `/opt/rocm`: 2290174 rc 0 with the "cuda is not found ... won't be built" warning; 0842236 rc 1, `CMake Error at CMakeLists.txt:18: AUTOWARE_CUDA_USE_HIP is ON, but HIP or rocThrust is not found. Add the ROCm prefix to CMAKE_PREFIX_PATH or set ROCM_PATH.` Option OFF with `-DCUDA_TOOLKIT_ROOT_DIR=/nonexistent` and no ROCm: rc 0 with upstream's warning, as before.

`utils/jargon.py --port autoware_cuda`: clean. `utils/prose.py` on both commit bodies: clean. `git -C projects/autoware_cuda/src status --porcelain`: clean.

### Open for a person (carried, unchanged)

Porter round 1 items 1-4 (DCO sign-off, copyright holder line, upstream defects, cspell) and the review's finding-3 choice (reword 2290174 at the same-tree re-creation, or correct it in the PR body only) still need Jeff. Nothing in this round depends on them.

## Review 2026-11-23b (reviewer, linux-gfx950 fleet job, delta 2290174...0842236)

Scope: `git diff 2290174...0842236` (20be73c, 0842236), the answer to "Review 2026-11-23". The porter of this delta and the reviewer were the same job, so the review was done by an independent child agent with no part in writing it and every finding was then re-run by the reviewer (configure-only probes, `ulimit -c 0`, against the HIP install of 0842236, nvcc 12.8 on PATH; probe sources in the job's `review-scratch/{fn,fncuda,optcuda}`). **Verdict: changes requested.** The previous findings 2 and 4 are resolved; finding 1 is resolved for the usual Autoware path (`autoware_package()` and the ament `find_*_dependencies` helpers are macros) but not in general (finding A).

### Findings

A. **The deferred check reads `autoware_cuda_utils_USE_HIP` from the wrong scope** (`autoware_cuda_utils/cmake/check_gpu_backend.cmake.in:18` and `:40`). The variable is set where the config file runs; `cmake_language(DEFER CALL ...)` runs at the end of the directory. When a consumer calls `find_package(autoware_cuda_utils)` inside a `function()` (a helper in the consumer, or a dependency's config loaded from one), the variable is gone by then, the deferred check takes the CUDA-build branch, and both directions go wrong. Reproduced:
   - HIP install, `function(f) find_package(autoware_cuda_utils REQUIRED) endfunction()`, `f()`, `enable_language(HIP)`, `-DAUTOWARE_CUDA_USE_HIP=ON` (a correct HIP consumer): rc 1, `check_gpu_backend.cmake:30: autoware_cuda_utils was built with CUDA, but probe_consumer is configured for HIP.` (false positive with a wrong message).
   - HIP install, same function, then `find_package(CUDA REQUIRED)`: rc 0, `CUDA_FOUND=TRUE` (the mixed-runtime case gets through).
   Fix: do not depend on the caller's scope at deferred time, e.g. bake the value into the function body (`if(@autoware_cuda_utils_USE_HIP@)`) or pass it as an argument of the deferred call (`cmake_language(DEFER CALL autoware_cuda_utils_check_gpu_backend ${autoware_cuda_utils_USE_HIP})`), and add the in-function case to the probe matrix.
B. **The HIP-build message gives the wrong remedy when the consumer already set the option** (`check_gpu_backend.cmake.in:22-28`). The HIP branch fires on `CUDA_FOUND`/`CUDAToolkit_FOUND`/CUDA language without regard to `AUTOWARE_CUDA_USE_HIP`, and always says "Configure <pkg> with -DAUTOWARE_CUDA_USE_HIP=ON". Reproduced: HIP install, `find_package(autoware_cuda_utils)` then `find_package(CUDAToolkit)`, configured with `-DAUTOWARE_CUDA_USE_HIP=ON`: rc 1 with that advice. Blocking is right (the consumer did find CUDA), but the message must say what was detected (`CUDA_FOUND`, `CUDAToolkit_FOUND`, or the CUDA language) so the user can find the `find_package(CUDA...)` responsible, e.g. a dependency exporting `CUDAToolkit` (`autoware_tensorrt_common/CMakeLists.txt:94-95` exports `CUDA` and `CUDAToolkit` today; round 2 will need to keep that in mind).
C. **0842236 Test Plan: "nvcc compiles the five test sources with no compiler warnings"** is not exact: the CUDA 12.8 build log has 20 `nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed` lines (same 20 at 2290174, so not a regression). Rewording an existing body is a history rewrite; same handling as the earlier finding 3 (Jeff's call at the same-tree re-creation). The fix commit for A/B should word its own Test Plan as "no warnings beyond nvcc's deprecated-GPU-target notice, as on the unmodified package".

### Checked, no finding

- ament configures the `.cmake.in` extra with `@ONLY` (`ament_package.cmake:103-113`), so `${PROJECT_NAME}`/`${_languages}` survive; installed files read `set(autoware_cuda_utils_USE_HIP TRUE)` (HIP) and `FALSE` (CUDA); the Config lists it after the `find_*` extra.
- The deferred check still runs after a consumer's top-level `return()`. Repeated loads: ament's `_CONFIG_INCLUDED` guard prevents duplicate deferrals in one scope. `cmake_language(DEFER)` needs CMake 3.19; upstream requires 3.22.
- The package's own build and tests are unaffected (the extra runs only in consumers). CUDA build: only addition is the extra; 20be73c fires only with the option ON, option OFF keeps upstream's warn-and-return.
- Commit titles 57 and 64 chars, `[ROCm]` appropriate, AI disclosed in prose, no trailer; `utils/jargon.py --port autoware_cuda` clean. Other body claims match the porter evidence above.

### For a person

- C joins the earlier finding 3 and the DCO sign-off question: options (a) reword both bodies at the same-tree re-creation for sign-off (recommended), or (b) correct them in the PR body only.

Validator not run: the review requested changes. Per the dispatch, the job stops here; the next porter round fixes A and B on top of 0842236.

## Porter round 1 fixes, second pass (2026-11-23, porter agent, linux-gfx950 fleet job; MI355X, HIP_VISIBLE_DEVICES=2)

Answers "Review 2026-11-23b". Still round 1 of Jeff's Q3 = i ruling: no MIGraphX work, `autoware_tensorrt_common` surface stays `owed`. One new commit on `moat-port` (no amend/rebase/force): `992d703` `[ROCm] Check the recorded GPU backend from any consumer scope`; head 0842236 -> 992d703.

### Per finding

- **A (deferred check reads the variable from the wrong scope)**: the check function's branch is now `if(@autoware_cuda_utils_USE_HIP@)`, configured into the body (`if(TRUE)` in the HIP install, `if(FALSE)` in the CUDA install), so the deferred call does not depend on the scope that loaded the config. `set(autoware_cuda_utils_USE_HIP ...)` stays for consumers that read it.
- **B (wrong remedy when the option is already set)**: the HIP-build message names what was detected (`CUDA_FOUND`, `CUDAToolkit_FOUND`, `the CUDA language`). With `AUTOWARE_CUDA_USE_HIP` ON it says to find the `find_package(CUDA)`, `find_package(CUDAToolkit)` or `enable_language(CUDA)` run by the package or a dependency and skip it; with the option OFF, the earlier advice (set the option or rebuild with CUDA). Round 2 must keep this in mind for `autoware_tensorrt_common/CMakeLists.txt:94-95` (exports `CUDA`/`CUDAToolkit`).
- **C (0842236 Test Plan wording)**: not reworded (history rewrite, Jeff's call; see "For a person" below). 992d703's Test Plan says "no warnings beyond nvcc's deprecated-GPU-target notice, as on the unmodified package".

### Environment (this job; nothing shared modified)

TheRock `rocm[devel,device-gfx950]==10.2.0` into `$FLEET_VENV` (HIP 7.17.26434), ROS 2 Jazzy ament tooling from source in `<job>/deps/ros2` (45 packages, 12.4 s), cppcheck 2.13.0 and libxml2 2.9.14 xmllint into `<job>/deps/prefix`, autoware_cmake 9849ad2. Workspaces `<job>/ws` (HIP) and `<job>/ws-cuda` (CUDA) symlink the fork clone and autoware_cmake; colcon run with explicit `--base-paths/--build-base/--install-base/--log-base` (isolated install, so the extra lives at `install/autoware_cuda_utils/share/autoware_cuda_utils/cmake/check_gpu_backend.cmake`).

### Builds at 992d703 (the committed tree; built before commit with the identical file)

- HIP: `colcon build --packages-up-to autoware_cuda_utils autoware_lint_common --cmake-args -DCMAKE_BUILD_TYPE=Release -DAUTOWARE_CUDA_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx950 -DCMAKE_PREFIX_PATH=$ROCM_SDK -DPython3_EXECUTABLE=$FLEET_VENV/bin/python`: rc 0, `-- cuda is not found, building autoware_cuda_utils with HIP 7.17.26434`, 0 compiler warnings.
- CUDA 12.8 (`PATH=/usr/local/cuda/bin:$PATH`, nvcc V12.8.93, no option, `-DBUILD_TESTING=ON -DCMAKE_VERBOSE_MAKEFILE=ON`): rc 0, `Found CUDA ... 12.8`, no HIP line, 10 nvcc invocations, 20 `nvcc warning` lines (deprecated pre-sm_75 targets, same as upstream), no compiler warning, test executable links. Compile/link only (no NVIDIA GPU).
- Tests (`ulimit -c 0`, `HIP_VISIBLE_DEVICES=2`, `timeout 1800 colcon test --packages-select autoware_cuda_dependency_meta autoware_cuda_utils --ctest-args --repeat until-fail:20 --timeout 300`): `[  PASSED  ] 17 tests.` on 20/20 runs, 0 skipped/failed; `colcon test-result`: 51 tests, 0 errors, 0 failures, 11 skipped (cppcheck 2.13).

### Reviewer's probes, rerun against the 992d703 installs (nvcc 12.8 on PATH)

Configure-only probes (the review's `fn`, `fncuda`, `optcuda`, `fnmacro`, `ret`, plus `fnok`: find in a function, `find_package(CUDA)` only when the option is OFF):

| install | probe | 0842236 (review) | 992d703 |
|---|---|---|---|
| HIP | `fn` + option ON (find in function, enable HIP) | rc 1, "built with CUDA" (false positive) | rc 0 |
| HIP | `fncuda` (find in function, then `find_package(CUDA)`) | rc 0 (missed) | rc 1, "configured for CUDA (CUDA_FOUND). Configure probe_consumer with -DAUTOWARE_CUDA_USE_HIP=ON, or rebuild autoware_cuda_utils with CUDA." (deferred) |
| HIP | `optcuda` + option ON (`find_package(CUDAToolkit)`) | rc 1, advice "set -DAUTOWARE_CUDA_USE_HIP=ON" | rc 1, "configured for CUDA (CUDAToolkit_FOUND). Find the find_package(CUDA), find_package(CUDAToolkit) or enable_language(CUDA) that runs for probe_consumer or one of its dependencies, and skip it when AUTOWARE_CUDA_USE_HIP is ON." |
| HIP | `optcuda`, no option | rc 1 | rc 1, "(CUDAToolkit_FOUND). Configure ... with -DAUTOWARE_CUDA_USE_HIP=ON ..." |
| HIP | `fnmacro` + option ON | rc 0 | rc 0 |
| HIP | `ret` (CUDA then top-level `return()`) | rc 1 | rc 1 (deferred still runs) |
| HIP | `fnok` + option ON / no option | -- | rc 0 / rc 1 (CUDA_FOUND) |
| CUDA | `fn` + option ON | -- | rc 1, "built with CUDA, but probe_consumer is configured for HIP" |
| CUDA | `fnok` no option / `fncuda` | -- | rc 0 / rc 0 |
| CUDA | `fnok` + option ON | -- | rc 1, "built with CUDA" |

Build-and-run probe (porter's `probe_consumer`, `autoware_package()` so `-Werror`):

| install | mode | 992d703 |
|---|---|---|
| HIP | `cuda_lang_early` | configure rc 1, "(CUDA_FOUND)" |
| HIP | `findcuda_late` | configure rc 1, "(CUDA_FOUND)" (deferred) |
| HIP | `hip` + option ON | builds, links only `amdhip64`, runs on GPU: `probe ok 1000 1023`, rc 0 |
| HIP | `hip`, no option | configure rc 1 (FindCUDA finds /usr/local/cuda; expected) |
| CUDA | `cuda_plain` | builds, links `cudart_static`; run aborts `cudaErrorInsufficientDriver` (no NVIDIA GPU, expected) |
| CUDA | `cuda_plain` + option ON | configure rc 1, "built with CUDA" |

`utils/jargon.py --port autoware_cuda`: clean. `utils/prose.py` on the 992d703 body: clean. `git -C projects/autoware_cuda/src status --porcelain`: clean.

### Open for a person (carried)

Porter round 1 items 1-4 (DCO sign-off, copyright holder line, upstream defects, cspell) and the reword question for the 2290174 and 0842236 Test Plans (review finding 3 and finding C): options (a) reword both bodies at the same-tree re-creation for sign-off (recommended by the reviewer), (b) correct them in the PR body only. Nothing in this round depends on them.

## Review 2026-11-23c (reviewer, linux-gfx950 fleet job, delta 0842236...992d703)

Scope: `git diff 0842236...992d703` (one commit, `autoware_cuda_utils/cmake/check_gpu_backend.cmake.in`), the answer to "Review 2026-11-23b". The porter and reviewer were the same job, so an independent child agent with no part in writing the delta did the review against the pr-review checklist; the reviewer then re-ran its probes (configure-only, `ulimit -c 0`, nvcc 12.8 on PATH, against the 992d703 HIP and CUDA installs). **Verdict: review passed.** Findings A and B are resolved; C stays with Jeff (wording of the 2290174/0842236 bodies, see "For a person" in the porter section), and 992d703's own Test Plan uses the requested wording.

No findings. Inherent limit, recorded so round 2 knows it (not a defect): the deferred check cannot see `CUDA_FOUND`/`CUDAToolkit_FOUND` (or a non-cache `AUTOWARE_CUDA_USE_HIP`) set only inside some other function scope of the consumer; the enabled-language test is a GLOBAL property and still catches `enable_language(CUDA)` from anywhere. An added probe (`both`: find in a function, then `find_package(CUDA)`, `find_package(CUDAToolkit)`, `enable_language(CUDA)`, option ON, HIP install) gives rc 1 with "configured for CUDA (CUDA_FOUND, CUDAToolkit_FOUND, the CUDA language). Find the find_package(CUDA), find_package(CUDAToolkit) or enable_language(CUDA) that runs for probe_consumer or one of its dependencies, and skip it when AUTOWARE_CUDA_USE_HIP is ON." `ament_lint_cmake` on the file: no problems; `list(JOIN)` needs CMake 3.12 (package requires 3.22); title 61 chars, `[ROCm]` appropriate, AI disclosed in prose, no trailer; `utils/jargon.py --port autoware_cuda` clean. No kernels in the delta, so no wavefront, texture or OOB fault class applies.

## Validation 2026-11-23 (validator, linux-gfx950 fleet job; AMD Instinct MI355X gfx950, HIP_VISIBLE_DEVICES=2) at 992d703

Scope: round 1 (`autoware_cuda_dependency_meta`, `autoware_cuda_utils`); `autoware_tensorrt_common` is owed to round 2 and was not built. Environment as in "Porter round 1 fixes, second pass" (TheRock 10.2.0 in the job venv, HIP 7.17.26434; ROS 2 Jazzy ament tooling, cppcheck 2.13.0, xmllint 2.9.14, autoware_cmake 9849ad2, all built in the job directory). Fork clone checked out at `992d7034b5396900593126e8f19a1982ae7bc682`, tree clean.

- Build from an empty workspace: `colcon build --packages-up-to autoware_cuda_utils autoware_lint_common --cmake-args -DCMAKE_BUILD_TYPE=Release -DAUTOWARE_CUDA_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx950 -DCMAKE_PREFIX_PATH=$ROCM_SDK -DPython3_EXECUTABLE=$FLEET_VENV/bin/python`: rc 0, `-- cuda is not found, building autoware_cuda_utils with HIP 7.17.26434`, 0 compiler warnings.
- Tests (`ulimit -c 0`): `timeout 1800 colcon test --packages-select autoware_cuda_dependency_meta autoware_cuda_utils --ctest-args --repeat until-fail:20 --timeout 300`: `[  PASSED  ] 17 tests.` on 20/20 runs, 0 skipped, 0 failed. `colcon test-result --verbose`: 51 tests, 0 errors, 0 failures, 11 skipped (ament_cppcheck skips with cppcheck 2.13, as on upstream's Jazzy CI). One extra run with `AMD_LOG_LEVEL=3`: `Gfx Major/Minor/Stepping: 9/5/0`, 26 kernel-launch log lines, 17/17 passed. No non-GPU baseline exists upstream to regress against (upstream registers no tests without CUDA).
- CUDA gate at 992d703: `/usr/local/cuda` nvcc V12.8.93 on PATH, `colcon build --packages-up-to autoware_cuda_utils autoware_lint_common --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCMAKE_VERBOSE_MAKEFILE=ON`: rc 0, `Found CUDA ... 12.8`, no HIP line, 10 nvcc invocations, 20 `nvcc warning` lines (deprecated pre-sm_75 targets, same count as upstream), no compiler warning, `test_autoware_cuda_utils` links. Compile/link only (no NVIDIA GPU). Architecture left at the package's FindCUDA default, as upstream builds it.
- Docs: the package README has the "AMD GPUs (ROCm)" section (the repository has no other build doc). `utils/jargon.py --port autoware_cuda`: clean.

Result: completed for linux-gfx950 at 992d703.

## Validation attempt 2026-11-23 (validator, linux-gfx1100 fleet job; AMD Radeon Pro W7800 48GB gfx1100, HIP_VISIBLE_DEVICES=3) for 992d703: not run, GPU wedged

Not a port result. Jeff Daily asked for a GPU health check first on this pinned GPU (wedged for days, driver reset 2026-11-23) and for the run to stop without moving to another GPU if the check failed. It failed, so nothing was built and no state was recorded for linux-gfx1100; the platform still owes its wave32 validation at 992d703.

Check 1, the image's installed torch (2.16.0a0+git6c3af02, HIP 7.16.26385), run from /tmp under `ulimit -c 0`:

```
timeout 120 python3 gpucheck.py   # 4 MiB float32 tensor .to("cuda:0"), sync, (y*2+1).sum(), sync, .cpu()
```

It printed the device (1 visible, W7800 gfx1100) and `H2D done 0.04s`, then hung on the first kernel launch plus synchronize: exit 124 at the 120 s timeout.

Check 2, a minimal HIP program with no torch (`hipcc --offload-arch=gfx1100 -O2 k.hip`: hipMalloc, hipMemcpy H2D, one elementwise kernel of 4096x256 threads, hipDeviceSynchronize):

```
timeout 120 ./k
```

It printed `dev AMD Radeon Pro W7800 48GB gfx1100` and `H2D ok`, then hung in the kernel's hipDeviceSynchronize: exit 124. So torch is not the cause.

amd-smi at the time showed GPU 3 at 0% GFX activity, 29 MB VRAM used, no running processes. dmesg cannot be read from the container. Since the reset the symptom has changed: the host-to-device copy now completes, and the hang moved to the first kernel dispatch. The GPU needs another look from a person before this host can validate.

## Validation 2026-11-23 (validator, linux-gfx1100 fleet job; AMD Radeon Pro W7800 48GB gfx1100, wave32, HIP_VISIBLE_DEVICES=0) at 992d703

Scope: round 1 (`autoware_cuda_dependency_meta`, `autoware_cuda_utils`); `autoware_tensorrt_common` is owed to round 2 and was not built. This is the wave32 run Jeff Daily asked for; the earlier gfx1100 attempt on the wedged GPU 3 is unrelated (this job was pinned to a different GPU). Fork clone checked out at `992d7034b5396900593126e8f19a1982ae7bc682`, tree clean. When this run started, the stage was already `porting` (round 2 lock held by linux-gfx950) with the head still at 992d703.

- GPU health check first (`hipcc --offload-arch=gfx1100`, hipMalloc, H2D, one 4096x256 elementwise kernel, sync, D2H, under `ulimit -c 0` and `timeout 120`): `dev AMD Radeon Pro W7800 48GB gfx1100 warp 32`, `kernel ok`, rc 0.
- Environment, all in the job directory, following the gfx950 recipe: TheRock `rocm[devel,device-gfx1100]==10.2.0` into `$FLEET_VENV` (`--index-url https://stable.repo.amd.com/rocm/whl-next/`), `rocm-sdk init`, HIP 7.17.26434; ROS 2 Jazzy ament tooling from source (`vcs import --shallow`, iceoryx ignored, the same `--packages-up-to` list: 45 packages, 18.3 s); cppcheck 2.13.0 (`-DUSE_MATCHCOMPILER=ON`) and libxml2 2.9.14 static xmllint into `<job>/deps/prefix`; autoware_cmake 9849ad2. gcc 13.3, CMake 3.31.6, Python 3.11 (job venv). Note: colcon's `--log-base` is a global option (`colcon --log-base <dir> build ...`), not a `build` option.
- Build from an empty workspace: `colcon --log-base $W/log build --base-paths $W/src --build-base $W/build --install-base $W/install --packages-up-to autoware_cuda_utils autoware_lint_common --cmake-args -DCMAKE_BUILD_TYPE=Release -DAUTOWARE_CUDA_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH=$ROCM_SDK -DPython3_EXECUTABLE=$FLEET_VENV/bin/python`: rc 0, `-- cuda is not found, building autoware_cuda_utils with HIP 7.17.26434`, 0 compiler warnings, HIP objects compiled with `--offload-arch=gfx1100`. (No CUDA toolkit on this host.)
- Tests (`ulimit -c 0`): `timeout 1800 colcon test --packages-select autoware_cuda_dependency_meta autoware_cuda_utils --ctest-args --repeat until-fail:20 --timeout 300`: rc 0, `[  PASSED  ] 17 tests.` on 20/20 runs, 0 skipped, 0 failed. `colcon test-result --verbose`: 51 tests, 0 errors, 0 failures, 11 skipped (the ament_cppcheck 2.13 skips, as on gfx950). One extra direct run of `test_autoware_cuda_utils` with `AMD_LOG_LEVEL=3`: `Gfx Major/Minor/Stepping: 11/0/0`, 18 `ShaderName` lines, 17/17 passed.
- CUDA gate: already recorded at 992d703 (linux-gfx950 validation), not repeated.
- Docs: `autoware_cuda_utils/README.md` has the "AMD GPUs (ROCm)" section. `utils/jargon.py --port autoware_cuda`: clean.

Result: passed on linux-gfx1100 at 992d703 (wave32).

Not recorded as `completed` in the project state: `moatlib.py set-state autoware_cuda linux-gfx1100 completed --agent validator --sha 992d7034...` refused it ("the project is at stage porting, and `completed` is recorded only once the review has passed"), because round 2's porter took the lock after the review passed. linux-gfx1100 therefore still has no `validated_sha`. If round 2 ends with the head still at 992d703 and the stage back at `review-passed`, the same command records this run without a rerun. If the head moves, this run is evidence only, and wave32 needs a new run at the new head (carry-forward needs a recorded `validated_sha`).

## Porter round 2 (2026-11-23/24, porter agent, linux-gfx942 fleet job; AMD Instinct MI300X gfx942, HIP_VISIBLE_DEVICES=3 pinned by the dispatcher)

Round 2 of Jeff Daily's Q3 = i staging, entered deliberately on his ruling (the porting lock was moved to linux-gfx942 with `port-lock --take`; this job re-stamped it with `set-state autoware_cuda linux-gfx942 porting`). The two earlier gfx950 attempts pushed nothing, so this round started from 992d703. Rulings applied: Q1 = A, Q2 = a, Q4 (DLA and built-in plugins scoped out, `plugin_paths` kept). New commits on `moat-port` on top of 992d703 (no amend, rebase or force): `b90fa9b`, `eba7d90`, `b60c8d7`; head 992d703 -> b60c8d7.

### Open for a person (nothing below was decided by the porter)

1. **Per-frame variable input shapes in autoware_universe (design, affects that project's scope).** MIGraphX compiles a program per concrete input shape. This port compiles the profile's `opt` shape in `setup()` and every other shape inside the profile once, on first use, at full compile cost (seconds to tens of seconds per shape; 10-50 s per model here). That is fine for consumers with fixed shapes or a few batch sizes, but consumers whose shapes change every frame (lidar voxel or pillar counts with profiles spanning thousands of values, e.g. CenterPoint, TransFusion) would compile on most frames. MIGraphX's own dynamic-shape mode was tried and does not help: for one dynamic dimension it compiles one submodule per value of the range (`migraphx-driver run dummy_network.onnx --dyn-input-dim @required_a "[{min:1,max:16,optimals:[8]},4]" ...` built 16 modules `dim_1`..`dim_16`, 8.4 s for that 3-op graph). Options: (a) those autoware_universe consumers pad the variable dimension to the profile max (or a few buckets) before calling `setInputShape` (consumer change, no change here); (b) a follow-up adds an opt-in bucketing mode to this package (compile a few sizes, pad up, slice outputs), which changes semantics and needs its own review; (c) decide per model in autoware_universe. Recommended: (a) or (c), decided when autoware_universe is planned; nothing here depends on it.
2. **Carried from round 1, unchanged:** DCO sign-off (the three new commits also carry no `Signed-off-by`, for the same reason), the copyright holder line (the seven new source files carry `Copyright 2026 The Autoware Foundation` plus Apache-2.0, as in round 1; the ament copyright test is disabled for this package upstream, so lint does not force it), the upstream defects, cspell, and the reword question for the 2290174/0842236 bodies.
3. **More upstream defects seen, not fixed (outside the port; reporting is Jeff's call):** (a) `conv_profiler.hpp:98` narrows TensorRT 10's `int64_t` `Dims::d` into `int` inside a braced initializer (`-Wnarrowing`); consumers do not see it because ament includes the package as a system header, but any TU that includes it with `-I` warns, on CUDA as well (the MIGraphX test target adds `-Wno-narrowing` for it, with a comment). (b) `tensorrt_common.cpp` logs `"Failed to set tensor address for tensor: ", tensor_name` and `"Failed to set input shape for tensor: ", tensor_name` with no `%s`, and `printProfiling()` passes the profiler text without `%s`, so the name and the profile are never printed. The MIGraphX file prints them; the TensorRT file is left as upstream wrote it.

### What changed on the fork

- `b90fa9b` `refactor(autoware_tensorrt_common): split out backend-neutral IO code` (shared, no prefix): pure code motion of the IO reconciliation (`resolveTensorNamesOrThrow`, `dropSkippedIO`, `reconcileOfferedIO`, `acceptOfferedIO`, `getIOTensorNames`, `validateNetworkIO` and the anonymous helpers) and the forwarding setters (by index, vector, map, `setTensor`, `setTensors`) from `tensorrt_common.cpp` into `src/tensorrt_common_io.cpp`, plus the three includes it orphaned. Checked: every moved block is byte-identical to upstream, and the objects define the same 46 `TrtCommon` symbols as upstream's (host compile, see CUDA path below).
- `eba7d90` `[ROCm] Run autoware_tensorrt_common models with MIGraphX`:
  - `CMakeLists.txt`: `option(AUTOWARE_CUDA_USE_HIP ...)` as in autoware_cuda_utils; with it ON, CUDA/CUDAToolkit/TENSORRT are not looked up. When CUDAToolkit is not found, `find_package(hip CONFIG)` + `find_package(migraphx CONFIG)` (QUIET, `PATHS $ENV{ROCM_PATH} /opt/rocm`); found -> MIGraphX branch, option ON and not found -> `FATAL_ERROR`, otherwise upstream's warning and `return()`. A CUDA host without TensorRT keeps upstream's behavior (no HIP fallback). MIGraphX branch: sources `migraphx_common.cpp onnx_model_io.cpp tensorrt_common_io.cpp profiler.cpp`, `PUBLIC hip::host rclcpp::rclcpp`, `PRIVATE migraphx::c stdc++fs` (consumers need no MIGraphX config), same `-Werror` flag set, stand-in include dir on BUILD/INSTALL interfaces and in `ament_export_include_directories`. Exports only `rclcpp` (not CUDA/CUDAToolkit/TENSORRT, which round 1's backend check would have caught in every consumer) plus config extras `cmake/find_hip.cmake` (`find_package(hip CONFIG REQUIRED PATHS ...)`) and `cmake/check_gpu_backend.cmake.in` (autoware_cuda_utils' check with the package name substituted; sets `autoware_tensorrt_common_USE_HIP`). The CUDA branch keeps upstream's sources plus `tensorrt_common_io.cpp`, links and exports; the CUDA install gains only the backend-check extra (as round 1 did). Tests on the MIGraphX branch link `autoware_cuda_utils::autoware_cuda_utils` (its shim is on the target only) and add `autoware_tensorrt_common_migraphx_test`.
  - `migraphx/include/NvInfer.h` (+ `NvInferRuntime.h`, `NvOnnxParser.h` that include it): written from scratch (no NVIDIA header was opened while writing it; names and enum values follow TensorRT's public API documentation). Declares `Dims64`/`Dims` (int32 `nbDims`, int64 `d[8]`, aggregate like TensorRT 10), `Dims2/3/4`, `DataType`, `TensorIOMode`, `LayerType` (no TensorRT values claimed), `LayerInformationFormat`, `CalibrationAlgoType`, `ILogger` (+`Severity` in TensorRT's order, which logger.hpp compares), `IProfiler`, `IInt8Calibrator` + `IInt8EntropyCalibrator`, `IInt8EntropyCalibrator2`, `IInt8MinMaxCalibrator`, `IInt8LegacyCalibrator` (the signatures the yolox and classifier calibrators override), abstract `ITensor`/`ILayer`/`IConvolutionLayer` with exactly the methods conv_profiler.hpp calls, opaque `IBuilderConfig`/`INetworkDefinition`; defines `AUTOWARE_TENSORRT_COMMON_MIGRAPHX` and `#ifndef cudaStream_t` -> `hipStream_t` (same guard as autoware_cuda_utils' compat header, so either order works). Include guards follow the repo's ros-include-guard hook (`NVINFER_H_`; TensorRT's own guard differs). `NvInferPlugin.h` is not provided: no package header includes it, and plugin types are autoware_universe scope.
  - `tensorrt_common.hpp`: one `#ifdef AUTOWARE_TENSORRT_COMMON_MIGRAPHX` around the TensorRT data members only (`friend class TrtConvCalib`, `std::unique_ptr<MigraphxState> migraphx_`, `nvinfer1::IInt8Calibrator * int8_calibrator_`); public section and all member function declarations untouched. The macro comes from the stand-in header, so a consumer that reaches the header any way (target, `_INCLUDE_DIRS`) sees the same layout as the library.
  - `tensorrt_conv_calib.hpp`: MIGraphX branch of `setupWithCalibrator` stores the calibrator, hands it to the backend, warns (int8 only) that per-layer settings and calibration caches are ignored; `quantization()` is TensorRT-only (`#ifndef`).
  - `src/onnx_model_io.cpp`: minimal protobuf wire reader for `ModelProto.graph` input/output/initializer names, elem types and dims (symbolic -> -1), seeking over weights. Needed because MIGraphX exposes no output names and its parameter order is not the ONNX order.
  - `src/migraphx_common.cpp`: the backend. Behavior by method: constructor keeps the `dlopen` loop, `hipSetDeviceFlags(hipDeviceScheduleBlockingSync)`, no `initLibNvInferPlugins`; `initialize()` checks path/extension, refuses `dla_core_id != -1` (constructor throws "Failed to initialize MIGraphX"), queries `gcnArchName`, reads the ONNX IO table, and parses once with MIGraphX so a bad model fails at construction as with TensorRT. `setup()` mirrors upstream's flow (accept/reconcile offered IO, build or load, validate, rebuild on mismatch); NetworkIO dtype overrides are not applied (warning) and `validateNetworkIO` then refuses a mismatch. Build: `parse_onnx` with `set_input_parameter_shape` = profile `opt` (inputs without profile: declared dims, dynamic -> 1), `quantize_fp16` for fp16, `quantize_int8` with host copies of the calibrator's `getBatch` buffers for int8 with a calibrator (no batches -> warning, compiled without int8; no calibrator -> model precisions kept), `compile(gpu, offload_copy=false)`. Execution: per-input shapes from `setInputShape` (validated against declared rank/static dims and the profile range, as TensorRT refuses out-of-profile shapes), program looked up per shape tuple (compiled on first use; int8 programs refuse other shapes because the calibration does not transfer), parameters bound by name (`#output_N` -> ONNX output N), `run_async(params, stream)`. Shapes/types/IO mode reported from the ONNX table (dynamic dims stay -1 as in a TensorRT engine). Cache: `<engine stem>.<gfx base>.<precision>.mxr` + `.mxr.key` (lines `migraphx <version.tweak>`, `target <full gcnArchName>`, `precision <p>`, `profile <ProfileDims::toString()>`...); `validateEngine` compares the first three, `validateProfileDims` the profile lines. `getNetwork()`/`getBuilderConfig()` return null with an error log; `getLayerInformation` returns empty.
  - Tests: `test/migraphx_common_test.cpp` (HIP only, 9 cases: IO order, fp32 into caller buffers, dynamic batch 8/1/16/3/8 and refused shapes, fp16, int8 via `TrtConvCalib` with a test calibrator, cache load and rebuild on key and profile change, dummy network sum, refused setup writes no program, TensorRT-only features refused), `test/data/generate_conv_network.py` -> `conv_network.onnx` (871 B: conv 3x3, ReLU, global average pool, flatten, gemm; outputs `logits` then `features`) and `conv_network_reference.bin` (numpy reference for the test input; matches onnxruntime 1.21 CPU to 3e-8).
  - `README.md`: "AMD GPUs (ROCm)" section in the package README (where the upstream package documents usage; the repository has no other build doc), in the style of round 1's section, listing what differs from TensorRT.
- `b60c8d7` `[ROCm] Do not throw from the MIGraphX test calibrator`: cppcheck `throwInNoexceptFunction` in the test calibrator.

Departures from plan.md: stand-in directory `migraphx/include/` outside `include/` (installed only by the ROCm build, the round-1 reason); no `NvInferPlugin.h` stand-in; the forwarding setters were moved with the IO reconciliation (one shared copy instead of duplicating them); IO order and names come from the ONNX file, not from MIGraphX; dynamic shapes use the per-shape program cache (native dynamic compile evaluated, see open item 1); the backend variable is `autoware_tensorrt_common_USE_HIP` (matches autoware_cuda_utils) instead of the plan's `_BACKEND`; `getLayerInformation` returns empty instead of the program text (MIGraphX's C++ API prints a program only to stdout) and no layer file is written; with `TrtConvCalib` int8 the non-quantized operations stay fp32 (TensorRT's path also enables fp16 for them; MIGraphX's `quantize_fp16` after `quantize_int8` was not tried); `package.xml` unchanged.

### Environment (job-local; nothing shared on the host was modified)

Ubuntu 24.04, gcc 13.3, CMake 3.31.6, Python 3.11.17 (job venv), no root, no system ROCm used, no nvcc on this host. MIOpen user DB and kernel cache pointed into the job directory (`MIOPEN_USER_DB_PATH`, `MIOPEN_CUSTOM_CACHE_DIR`) after the first runs (the first runs wrote MIOpen's usual per-version cache under `~/.cache/miopen` and `~/.config/miopen`, as every ROCm job on the host does).

- ROCm: `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple 'rocm[devel,device-gfx942]==10.2.0'`, `rocm-sdk init`, `ROCM_SDK=$($FLEET_VENV/bin/python -m rocm_sdk path --root)`, HIP 7.17.26434. The SDK has MIOpen, rocBLAS, hipBLASLt, hipBLAS and hiprtc but no MIGraphX, rocMLIR or composable_kernel.
- ROS 2 Jazzy from source with rclcpp (round 1 stopped short of rclcpp): `vcs import --shallow src < ros2.repos` (ros2/ros2 `jazzy`), `COLCON_IGNORE` in `eclipse-iceoryx` and `eclipse-cyclonedds`, then `colcon build --merge-install --packages-up-to rclcpp ament_cmake_auto ament_cmake_gtest ament_lint_auto ament_lint_common ament_cmake_copyright ament_cmake_cppcheck ament_cmake_lint_cmake ament_cmake_xmllint ament_cmake_cpplint ament_cmake_uncrustify ros_environment --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DTRACETOOLS_DISABLED=ON -DCMAKE_PREFIX_PATH=<tinyxml2 prefix> -DAsio_INCLUDE_DIR=<asio>/asio/include -DPython3_EXECUTABLE=$FLEET_VENV/bin/python`: 128 packages, 78 s. Without root, Fast-DDS needs asio headers (`chriskohlhoff/asio` tag `asio-1-28-1`, header-only) and TinyXML2 (`leethomason/tinyxml2` 10.0.0, shared, into a job prefix, which then goes on `LD_LIBRARY_PATH`), and tracetools needs `-DTRACETOOLS_DISABLED=ON` (no lttng-ust).
- Workspace `src/`: the fork clone (symlink), `autowarefoundation/autoware_cmake` 9849ad2, `tier4/tensorrt_cmake_module` c938430 (0.0.5; `autowarefoundation/tensorrt_cmake_module` does not exist). Lint tools in a job prefix: cppcheck 2.13.0 (`-DUSE_MATCHCOMPILER=ON`), libxml2 2.9.14 static `xmllint`. Python tools in the job venv: `colcon-common-extensions vcstool empy==3.3.4 lark catkin_pkg pyyaml pytest flake8 pydocstyle numpy setuptools`, plus `onnxruntime` (CPU reference) and `pre-commit`.

### MIGraphX from source against the TheRock pip SDK (recipe for validators and autoware_universe)

No MIGraphX tag matches ROCm 10.2 (tags end at `rocm-10.1`), so develop was used: `ROCm/AMDMIGraphX` at `f0239c965` (MIGraphX 2.19.0, `MIGRAPHX_VERSION_TWEAK` `20250912-17-889-gf0239c965`). All into one prefix `P=<job>/deps/mgx-prefix`; `R=$ROCM_SDK`; versions from MIGraphX's `requirements.txt` at that commit.

1. abseil-cpp `20250512.0` (`-DABSL_ENABLE_INSTALL=ON -DABSL_PROPAGATE_CXX_STD=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CXX_STANDARD=17`), nlohmann/json `v3.8.0` (`-DJSON_BuildTests=OFF`), msgpack-c `cpp-3.3.0` (`-DMSGPACK_BUILD_TESTS=OFF -DMSGPACK_BUILD_EXAMPLES=OFF -DMSGPACK_CXX17=ON`); gcc is fine for these. sqlite3 dev headers were on the host.
2. protobuf `v30.0` **built with the SDK clang** (`-DCMAKE_C_COMPILER=$R/llvm/bin/clang -DCMAKE_CXX_COMPILER=$R/llvm/bin/clang++ -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -Dprotobuf_ABSL_PROVIDER=package -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CXX_STANDARD=17 -DCMAKE_POLICY_VERSION_MINIMUM=3.5`). Built with gcc, every ONNX parse segfaulted (lesson below).
3. rocm-cmake `6a7c5b73b8882c74f8f7060e2633f230dabb7b63` (`cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$P && cmake --install build`): the SDK's ROCmCMakeBuildTools lacks `rocm_add_version_resource`, which MIGraphX develop calls (`Unknown CMake command "rocm_add_version_resource"`).
4. rocMLIR: `ROCm/rocmlirTriton` at `e0fd0dabcdf377ff32b331e1b116d7b7602867b4` (shallow fetch of that commit), `-DCMAKE_C_COMPILER=$R/llvm/bin/clang -DCMAKE_CXX_COMPILER=$R/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$R -DCMAKE_INSTALL_PREFIX=$P -DBUILD_FAT_LIBROCKCOMPILER=On -DLLVM_INCLUDE_TESTS=Off -DMLIR_ENABLE_ROCM_RUNNER=Off -DMLIR_INCLUDE_TESTS=Off "-DLLVM_TARGETS_TO_BUILD=X86;AMDGPU" -DCMAKE_POLICY_VERSION_MINIMUM=3.5`, 4406 steps, about 11 min with 96 jobs. Without rocMLIR MIGraphX builds and runs most models, but mnist-12 fp32 and int8 fail in `gpu::lowering` ("same_layout: gpu::convolution: Layouts do not match"), reproduced with `migraphx-driver verify mnist-12.onnx --gpu`; with rocMLIR it verifies.
5. MIGraphX: `ROCM_PATH=$R cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$R/llvm/bin/clang++ -DCMAKE_C_COMPILER=$R/llvm/bin/clang "-DCMAKE_PREFIX_PATH=$P;$R" -DCMAKE_INSTALL_PREFIX=$P -DGPU_TARGETS=gfx942 -DMIGRAPHX_ENABLE_MLIR=ON -DMIGRAPHX_USE_COMPOSABLEKERNEL=OFF -DMIGRAPHX_ENABLE_PYTHON=OFF -DMIGRAPHX_ENABLE_TENSORFLOW=OFF -DMIGRAPHX_USE_AMDMLSS=OFF -DBUILD_TESTING=OFF && cmake --build build -j $MAX_JOBS && cmake --install build` (719 steps, about 2 min). `GPU_TARGETS` is the validator's gfx (MIGraphX JIT-compiles most kernels for the running GPU anyway). composable_kernel was left off (not in the SDK; MIGraphX falls back to its own and MLIR kernels).
6. Runtime: `LD_LIBRARY_PATH=$R/lib:$P/lib` (the private libraries in `$P/lib/migraphx/lib` are found through RUNPATH). Check: `migraphx-driver verify <job>/.../test/data/dummy_network.onnx --gpu --input-dim @required_a 8 4 @required_b 8 4 @optional_present 8 4` and `migraphx-driver verify mnist-12.onnx --gpu`: "MIGraphX verification passed successfully". Note `$P/bin` on PATH makes CMake find `$P` as a prefix by itself; to test the not-found path take it off PATH.

### Build and tests at b60c8d7 (HIP, MIGraphX with rocMLIR)

```
colcon --log-base $W/log build --base-paths $W/src --build-base $W/build --install-base $W/install --packages-up-to autoware_cuda_utils autoware_tensorrt_common autoware_lint_common --cmake-args -DCMAKE_BUILD_TYPE=Release -DAUTOWARE_CUDA_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 "-DCMAKE_PREFIX_PATH=$ROCM_SDK;$P" -DPython3_EXECUTABLE=$FLEET_VENV/bin/python
colcon --log-base $W/log test --base-paths $W/src --build-base $W/build --install-base $W/install --packages-select autoware_cuda_dependency_meta autoware_cuda_utils autoware_tensorrt_common --ctest-args --timeout 900
colcon test-result --test-result-base $W/build --verbose
```

- Build from an empty workspace: rc 0, `-- cuda is not found, building autoware_cuda_utils with HIP 7.17.26434`, `-- cuda is not found, building autoware_tensorrt_common with MIGraphX 2.19.0`, 0 compiler warnings.
- `colcon test-result`: 117 tests, 0 errors, 0 failures, 28 skipped (the 11 + 17 ament_cppcheck skips with cppcheck 2.13). autoware_tensorrt_common: `autoware_tensorrt_common_test` 13/13 (3 TensorInfoTest, 10 TrtCommonIOTest, unchanged file, none skipped), `autoware_tensorrt_common_migraphx_test` 9/9, cpplint 17/17, lint_cmake 3/3, xmllint 1/1; autoware_cuda_utils 17/17 gtest as in round 1. `ctest -R "gtest|_test" --repeat until-fail:5` in the tensorrt_common build dir: 5/5 for both binaries (MIGraphX built without rocMLIR, at eba7d90). Final run at b60c8d7 with the rocMLIR build, from an empty workspace and wrapped in `utils/timeit.sh` (compile, test): `colcon test ... --ctest-args --repeat until-fail:3 --timeout 900`: `[  PASSED  ] 13 tests.` and `[  PASSED  ] 9 tests.` on 3 of 3 runs, `test_autoware_cuda_utils` 17/17 on 3 of 3, `colcon test-result` 117 tests, 0 errors, 0 failures, 28 skipped. cppcheck 2.13 directly on the new files: one real finding (fixed in b60c8d7); the rest are upstream-style notes (constructor-body `logger_` assignment mirrors upstream, `utils.hpp` by-value strings are upstream).
- `pre-commit run --files <changed files>` with the repo's config (clang-format 23.1.1, cpplint 2.0.2, ros-include-guard, black, isort, prettier): clean after its autofixes (include guards and clang-format) were taken. `utils/jargon.py --port autoware_cuda`: clean; `utils/prose.py` on the three commit bodies: clean. `git -C projects/autoware_cuda/src status --porcelain`: clean.
- Each TrtCommon `setup()` takes a multiple of 5 s: upstream's `log_throttle` thread sleeps in 5 s steps and `stop_throttle` joins it (same on CUDA).

Public ONNX models (onnx/models `validated/`: mnist-12, squeezenet1.1-7, mobilenetv2-12 with dynamic `batch_size`, resnet18-v1-7 with dynamic `N`), run through a scratch consumer package (`find_package(autoware_tensorrt_common)`, `-Werror`, both `target_link_libraries` and `ament_target_dependencies`, a `FATAL_ERROR` if CUDA leaks in: it does not) that feeds a seeded random input and compares the first output with onnxruntime 1.21 CPU; for dynamic models the profile is min 1 / opt N / max 2N:

| model | precision | batch (opt) | rel. max error | top-1 | setup / run |
|---|---|---|---|---|---|
| mnist-12 | fp32 / fp16 / int8 | 1 | 1.4e-7 / 6.7e-4 / 3.3e-2 | same | 10-20 s / 0.02 ms |
| squeezenet1.1 | fp32 | 1 | 4.7e-7 | same | 45 s, cached 0.35 s / 0.19 ms |
| squeezenet1.1 | fp16 / int8 | 1 | 1.2e-3 / 8.5e-2 | same | 50 s / 35 s, int8 cached 0.37 s |
| mobilenetv2 | fp32 | 4 (4), 1 (4), 8 (4) | 1.3e-6, 9.4e-7, 1.3e-6 | same | 20 s / 2.4, 0.66, 4.8 ms |
| mobilenetv2 | fp16 | 3 (4) | 3.9e-3 | same | 20 s / 1.5 ms |
| resnet18-v1 | fp32 / fp16 / int8 | 2 (2) | 1.1e-6 / 3.4e-3 / 0.25 | same | 25-30 s / 0.83, 0.29, 0.29 ms |
| squeezenet1.1 | fp32, null stream | 1 | 4.7e-7 | same | |

int8 calibration here used 8 random-normal batches, so its error says nothing about real accuracy (the plan's risk list: real-model int8 accuracy is checked in autoware_universe). `migraphx-driver read` on the saved programs: squeezenet int8 has 26 `mlir_quant_convolution*` ops, resnet18 int8 20 plus a `quant_dot`, resnet18 fp16 87 `half_type` values. Batch 1/3/8 with opt 4 exercised the per-shape compile path.

Backend check from consumer packages against the installed HIP packages (configure only): a HIP consumer finding both packages with `-DAUTOWARE_CUDA_USE_HIP=ON`: rc 0, `autoware_tensorrt_common_USE_HIP=TRUE`, no `CUDA_FOUND`/`CUDAToolkit_FOUND`; same plus `CUDA_FOUND` set: rc 1 from the autoware_cuda_utils check; `find_package(autoware_tensorrt_common)` inside a function, then `CUDAToolkit_FOUND`: rc 1 from autoware_tensorrt_common's deferred check ("built with HIP, but probe_consumer is configured for CUDA (CUDAToolkit_FOUND)"). Package configure: option ON with no MIGraphX found: rc 1, "AUTOWARE_CUDA_USE_HIP is ON, but HIP or MIGraphX is not found."; option OFF with neither toolkit: rc 0 with upstream's "cuda, tensorrt libraries are not found".

### CUDA path: not compiled with nvcc in this round

**This host has no nvcc and no TensorRT, so the CUDA build of round 2 has not been run.** It has to be run on a host with nvcc and TensorRT 10 (the mi355 fleet host) at b60c8d7 before this round is presented; `build:cuda-path-nvcc-tensorrt` stays `owed` in surface.json until then. What was checked here instead (host compiler only, which is what compiles this package's CUDA build, since it has no `.cu` files): `g++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -Wno-deprecated-declarations -DTENSORRT_VERSION_MAJOR=10` on `src/tensorrt_common.cpp`, `src/tensorrt_common_io.cpp`, `src/profiler.cpp` and a TU including `tensorrt_conv_calib.hpp`, against the NVIDIA/TensorRT `release/10.13` `include/` (b8db91e), onnx/onnx-tensorrt `NvOnnxParser.h` (10.13-GA, 346bdca) and the CUDA 12.8 runtime headers from the `nvidia-cuda-runtime-cu12==12.8.90` and `nvidia-cuda-nvcc-cu12==12.8.93` wheels (headers only, in a scratch directory): all compile; upstream's `tensorrt_common.cpp` compiles the same way; `nm` shows the port's two objects define exactly upstream's 46 `TrtCommon` symbols. Not checked: CMake configure of the CUDA branch, linking against TensorRT, the CUDA install tree, the tests on an NVIDIA GPU.

### For the reviewer and the validators

- Gate on each platform: `autoware_tensorrt_common_test` 13/13 and `autoware_tensorrt_common_migraphx_test` 9/9, none skipped (a skip means `hipGetDeviceCount` saw no device), plus round 1's 17/17 `test_autoware_cuda_utils`. Confirm the configure line `building autoware_tensorrt_common with MIGraphX`. A validator needs MIGraphX built with rocMLIR (recipe above, `GPU_TARGETS` set to its gfx) and ROS 2 with rclcpp.
- The wave32 gate depends on MIGraphX on the wave32 GPU (gfx1100): not checked here.
- The CUDA compile check of round 2 is owed (previous section).

## Lesson for cuda-to-rocm: protobuf built by gcc crashes a clang-built MIGraphX

For `linux-host-setup.md` (building ROCm libraries from source). Building MIGraphX from source with the ROCm clang against a protobuf (v30) built with the host gcc compiles and links cleanly, but every ONNX parse segfaults inside `google::protobuf::internal::TcParser::FastMtS1` (jump to a garbage address, called from `MessageLite::ParseFromIstream` in `libmigraphx_onnx`). Recent protobuf compiles its table-driven parser functions with a clang-only calling convention (`preserve_most`/`preserve_none` behind `PROTOBUF_CC`), so the clang-compiled generated code (`onnx.pb.cc`) and the gcc-compiled `libprotobuf` disagree on how those functions are called. Build protobuf with the same clang as MIGraphX (MIGraphX's own dependency script uses one compiler for everything). abseil is fine either way. (autoware_cuda)

## Lesson for cuda-to-rocm: MIGraphX names no outputs; take the IO table from the ONNX file

For `no-hip-equivalent.md` (TensorRT -> MIGraphX). TensorRT code addresses IO tensors by name and by index (inputs then outputs, in ONNX order). MIGraphX's API returns outputs by position only (compiled parameters `#output_0`, `#output_1`, ... with `offload_copy` off) and does not promise its parameter order, so the names and order have to come from the model: read `ModelProto.graph` inputs (minus initializers) and outputs, with their declared dims (symbolic -> -1, which is also what TensorRT reports for dynamic dimensions) and elem types; a ~300-line protobuf wire reader that seeks over the weights is enough, with no protobuf or ONNX dependency. Bind each output buffer as the `#output_N` parameter or MIGraphX allocates its own and the caller's buffer stays stale. Also: MIGraphX's dynamic-shape mode compiles one submodule per value of a single dynamic dimension, so for wide TensorRT profiles compile per concrete shape (opt first, others on first use) instead. (autoware_cuda)

## Lesson for moat-porting: MIGraphX recipe for fleet validators

For the fleet build recipes section: the TheRock pip SDK carries no MIGraphX or rocMLIR; the working from-source recipe (MIGraphX develop f0239c965 + rocMLIR + clang-built protobuf + rocm-cmake pin, about 15 min with 96 jobs) is in autoware_cuda's notes, "MIGraphX from source against the TheRock pip SDK". Without rocMLIR MIGraphX builds but some convolution models fail to compile. (autoware_cuda)
