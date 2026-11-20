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
