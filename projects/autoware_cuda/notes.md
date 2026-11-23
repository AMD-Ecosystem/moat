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
  - `hip/include/cuda_runtime_api.h`, `cuda_runtime.h` (shim names, include `<hip/hip_runtime_api.h>` / `<hip/hip_runtime.h>`) and `cuda_runtime_hip_compat.h` (`#ifndef`/`#define` for exactly the 38 runtime names the package and its tests use; `cudaFreeHost` -> `hipHostFree`, `cudaHostAlloc` -> `hipHostAlloc`, which is not deprecated in HIP 7.17, `cudaPointerAttributes` -> `hipPointerAttribute_t`). The public headers are untouched apart from the two lines above.
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

`-DAUTOWARE_CUDA_USE_HIP=ON` is needed when a CUDA toolkit is also installed. Consumers configure with the same `CMAKE_PREFIX_PATH` (or `ROCM_PATH`), because the installed config extras (`find_rocthrust.cmake`) run `find_package(hip)` and `find_package(rocthrust)`. Linking `autoware_cuda_utils` (with ament_auto, `ament_target_dependencies` or `target_link_libraries(autoware_cuda_utils::autoware_cuda_utils)`) gives a consumer the shim `cuda_runtime_api.h`/`cuda_runtime.h` that map only the 38 runtime names this package uses. Each consumer brings its own compat header for the rest; the shim's macros are `#ifndef`-guarded, so identical definitions coexist. With a pip ROCm SDK, installed binaries need `$ROCM_SDK/lib` on `LD_LIBRARY_PATH`. As of round 1 (head 2290174), `autoware_tensorrt_common` is not ported yet (round 2, MIGraphX).
