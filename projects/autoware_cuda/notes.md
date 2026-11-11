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
