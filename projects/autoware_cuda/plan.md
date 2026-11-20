# autoware_cuda port plan

## Project

- Name: autoware_cuda (autowarefoundation/autoware_cuda, maintained by TIER IV engineers for the Autoware Foundation)
- Upstream: https://github.com/autowarefoundation/autoware_cuda, default branch `main`, planned at `a5929dd` ("docs: explain the COLCON_IGNORE loop and name the core CUDA images (#6)")
- Fork: https://github.com/AMD-Ecosystem/autoware_cuda, port branch `moat-port`
- What it is: three ament (ROS 2) packages, about 4.4 KLOC in total, holding the CUDA/TensorRT base that every GPU package in autoware_universe builds on:
  1. `autoware_cuda_dependency_meta`: a CMake-only virtual package whose config extras add `-fno-lto`.
  2. `autoware_cuda_utils`: a header-only INTERFACE library (CUDA runtime RAII helpers, stream-ordered memory pool, Thrust-on-stream helpers, gtest skip helpers) plus a GPU gtest binary.
  3. `autoware_tensorrt_common`: the `TrtCommon` ONNX-to-TensorRT engine wrapper (shared library, about 3.3 KLOC host C++, no kernels), its INT8 calibration subclass `TrtConvCalib`, the per-layer profilers, and a gtest that builds an engine from a dummy ONNX network.
- Reverse dependency: autoware_universe (MOAT project, `depends_on: [autoware_cuda, cuda_blackboard]`). Jeff Daily's scope ruling of 2026-11-11 on autoware_universe takes the whole stack, with TensorRT engine handling mapped to MIGraphX. This plan therefore also fixes the interface autoware_universe's planner has to build against (see "Interface contract for autoware_universe").
- Planned on linux-gfx950 (mi355 fleet host, no GPU used, nothing built). There is no lead platform; this plan is arch-independent.

**Decisions Jeff must make before a porter starts are listed at the top of the `## Planning` section of notes.md (Q1-Q4).** The plan below is written for the recommended answer to each and says where another answer changes it.

## Existing AMD support

None. Intake: `duplicate_effort: none`; no `hip|rocm|amd|migraphx` PRs, issues or branches upstream; no AMD-Ecosystem or ROCm repository; no MIGraphX backend for Autoware anywhere on GitHub. The AMD Silo AI / Autoware Foundation collaboration (2025-12-16) has published no code touching this repository; coordination with it is Jeff's call and he did not make it a precondition.

Upstream is a normal Autoware Foundation repository that takes PRs (CONTRIBUTING.md, semantic-PR CI); it does not link platform forks instead of merging, so an upstream PR is the vehicle. Upstream's README says the repo "holds only packages that need CUDA to build"; a MIGraphX backend inside `autoware_tensorrt_common` stretches that wording, which is part of Q1.

Port-vs-rewrite: there are no performance kernels here (no CUTLASS, no tensor-core code, no warp tuning). `autoware_cuda_utils` is a mechanical HIP port and final in that form. `autoware_tensorrt_common` has nothing to translate into: it is a reimplementation of the engine layer against MIGraphX (the `cuda-to-rocm` skill's "TensorRT -> MIGraphX", the "deep TensorRT" shape: INT8 calibration, version-tied engine cache, plugin loading, optimization profiles).

## Build classification

**cmake** (ament_cmake / ament_cmake_auto inside a colcon workspace). Evidence:

- `autoware_cuda_utils/CMakeLists.txt:7` `find_package(CUDA)` (legacy FindCUDA), `:9-12` returns early when CUDA is absent, `:14` `include(cmake/find_cccl.cmake)`, `:21` `target_link_libraries(... CCCL::CCCL autoware_cuda_utils::cccl_system CUDA::cudart)`, `:33` `cuda_add_executable(test_autoware_cuda_utils ...)` over five `.cu` files, `:60` `ament_auto_package(CONFIG_EXTRAS cmake/find_cccl.cmake)` (the CCCL lookup is re-run in every consumer).
- `autoware_tensorrt_common/CMakeLists.txt:6-15` `find_package(tensorrt_cmake_module REQUIRED)`, `find_package(CUDA)`, `find_package(CUDAToolkit)`, `find_package(TENSORRT)`, early return unless CUDAToolkit and TensorRT are both found; `:17-20` requires TensorRT >= 10; `:27-32` links `CUDA::cudart rclcpp::rclcpp stdc++fs ${TENSORRT_LIBRARIES}`; `:47-49` `-Wall -Wextra -Wpedantic -Werror -Wno-deprecated-declarations`; `:89-95` `ament_export_dependencies(CUDA CUDAToolkit rclcpp tensorrt_cmake_module TENSORRT)`.
- `autoware_cuda_dependency_meta/CMakeLists.txt` is `project(... NONE)` plus `ament_package(CONFIG_EXTRAS ...)`.
- No `find_package(Torch)`, no setup.py, no Python build.

`ext_type`: `cmake` (recorded).

## Port strategy

**Strategy A (CMake HIP language, compat via shim headers)** for both packages, plus a new MIGraphX implementation of the engine layer.

Backend selection, identical in both packages: the existing CUDA/TensorRT branch is tried first and is unchanged; only when it is not taken does a HIP branch look for ROCm (`find_package(hip CONFIG)`, plus rocThrust or MIGraphX). An explicit `-DAUTOWARE_CUDA_USE_HIP=ON` cache option (name to match house style; the porter may pick a shorter one) forces the HIP branch on a machine that has both toolkits. A CUDA-only machine sees no new behaviour.

### autoware_cuda_utils

The headers include `<cuda_runtime_api.h>` / `<cuda_runtime.h>` and use about 35 CUDA runtime names; consumers in autoware_universe include these headers from `.cu` and `.cpp` files. Use the shim-header method so the public headers stay byte-identical to upstream except for one line:

- Add `include/autoware/cuda_utils/hip/cuda_runtime_api.h` and `.../hip/cuda_runtime.h` (directory name is the porter's choice; it must not be on the CUDA include path). Each includes `<hip/hip_runtime_api.h>` (resp. `<hip/hip_runtime.h>`) and a shared `cuda_to_hip.h` that maps exactly the names the package and its tests use (list in "CUDA surface inventory"). It must be host-includable (the skill's "A shared compat header must be host-includable"), because `.cpp` consumers compile it with the host compiler.
- The HIP branch adds that directory to the INTERFACE include directories (BUILD and INSTALL), installs it, and links `hip::host` and `roc::rocthrust` instead of `CCCL::CCCL`, `autoware_cuda_utils::cccl_system` and `CUDA::cudart`.
- `thrust_utils.hpp:29` `thrust::cuda::par.on(stream)`: rocThrust's HIP backend has no `thrust::cuda` namespace (checked in the ROCm 10.1 SDK headers: `thrust::cuda::par` exists only in the CUDA-backend `thrust/system/cuda/detail/par.h`). This is the one guarded source edit: `thrust::hip::par.on(stream)` when `THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP`, upstream line otherwise. Do not try to fix it with a `cuda` macro.
- `cmake/find_cccl.cmake` is exported as CONFIG_EXTRAS and does `find_package(CUDAToolkit REQUIRED)` in every consumer. On the HIP branch it must not run: export a HIP extras file instead (`cmake/find_rocthrust.cmake`: `find_package(hip CONFIG REQUIRED)`, `find_package(rocthrust CONFIG REQUIRED)`), selected at configure time. Check how `ament_auto_package(CONFIG_EXTRAS ...)` behaves when the argument is a variable, and that a consumer configured against the HIP install never touches CUDAToolkit.
- Test target: on the HIP branch `enable_language(HIP)`, `add_executable` over the same sources with `set_source_files_properties(<5 .cu> PROPERTIES LANGUAGE HIP)`, same gtest wiring and `ament_add_gtest_test`. `cuda_add_executable` stays on the CUDA branch.
- `autoware_package()` (autoware_cmake) applies the Autoware warning set; HIP headers must compile clean under it. Put ROCm include directories in SYSTEM where warnings come from them.

### autoware_cuda_dependency_meta

Expected no change: `-fno-lto` is harmless with HIP and may be needed for the same duplicate-fatbin reason with HIP fat binaries. The porter confirms by building; record the result either way.

### autoware_tensorrt_common (MIGraphX engine layer)

Recommended design (Q1 option A, "same API, second backend"): the package keeps its name, headers and `TrtCommon` / `TrtConvCalib` / `Profiler` / `ConvProfiler` / `Logger` classes. On the HIP branch it is built from a new `src/migraphx_common.cpp` (the porter may split it) instead of `src/tensorrt_common.cpp`, against MIGraphX's C++ API (`migraphx/migraphx.hpp`, target `migraphx::c`), `hip::host` and `rclcpp`. Consumers compile unchanged as far as the value types go.

- Value types. `TrtCommon`'s public API and the consumers use `nvinfer1::Dims` (240 uses in autoware_universe), `nvinfer1::DataType`, `nvinfer1::TensorIOMode`, `nvinfer1::ILogger::Severity`, `nvinfer1::IProfiler`, the `IInt8Calibrator` family and `CalibrationAlgoType` (yolox and classifier calibrators), and opaque `IBuilderConfig` / `INetworkDefinition` / `ILayer`. On the HIP branch only, install shim headers named `NvInfer.h`, `NvInferRuntime.h`, `NvInferPlugin.h`, `NvOnnxParser.h` in a HIP-only include directory that declare exactly that subset (written from scratch from TensorRT's public documentation; nothing copied from NVIDIA's headers, which carry NVIDIA's licence), with `Dims` layout matching TensorRT 10 (`int32_t nbDims; int64_t d[8]`). The shim must never be reachable when real TensorRT headers are, which the backend selection guarantees.
- Private state. `tensorrt_common.hpp`'s private members are TensorRT objects. Add one `#if` around the private section: the HIP branch holds a pimpl (`std::unique_ptr<MigraphxState>`), the CUDA branch is exactly upstream. The public section is not touched.
- Shared logic. `resolveTensorNamesOrThrow`, `dropSkippedIO`, `reconcileOfferedIO`, `acceptOfferedIO`, `validateNetworkIO` and the anonymous-namespace helpers are backend-neutral and are what the 10 `TrtCommonIOTest` cases test. Move them, as pure code motion, from `tensorrt_common.cpp` into `src/tensorrt_common_io.cpp` built by both branches, rather than duplicating 250 lines. That is the only edit to the CUDA path's sources; compile it with nvcc and TensorRT 10 headers (see Build) in the same round.
- Mapping, method by method:
  - Constructor: `cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)` -> `hipSetDeviceFlags(hipDeviceScheduleBlockingSync)` (a synonym for Yield on ROCm; fine). `plugin_paths`: keep the `dlopen` loop; a MIGraphX plugin library registers its ops with `register_experimental_custom_op` from a static initialiser (the plugin libraries themselves are autoware_universe scope). `initLibNvInferPlugins`: no counterpart; log once at INFO. `dla_core_id != -1`: refuse with an error (DLA is Jetson hardware; Q4).
  - `initialize()` / ONNX parse: `migraphx::parse_onnx(path, onnx_options)`. Before `setup()` the pre-setup getters (`getNbIOTensors`, `getIOTensorName`, `getTensorShape(index)`, `getInputDims`, `getOutputDims`) answer from the parsed, uncompiled program: `get_parameter_shapes()` for inputs, `get_output_shapes()` for outputs. TensorRT orders IO tensors as network inputs then outputs in ONNX graph order, and consumers index by position (16 `getIOTensorName` uses). MIGraphX does not promise parameter order; read the ONNX graph's input and output name order from the model (MIGraphX's parse keeps output order; inputs may need the ONNX protobuf's `graph.input` minus initializers) and keep an explicit index table. Output names: MIGraphX returns outputs by position, so map ONNX output names to positions once.
  - `setup(profile_dims, network_io)`: the shared reconcile path first (unchanged semantics: rejects before compiling). Profiles: `set_dyn_input_parameter_shape(name, dynamic_dimensions{{min, max, {opt}}...})`, re-parse, compile for `migraphx::target("gpu")` with `offload_copy` off (inputs and outputs are device pointers the caller owns). MIGraphX's dynamic-shape support on the GPU target is narrower than TensorRT profiles; if a dynamic compile is refused or slow, fall back to compiling for the concrete shape on first use of each new shape set by `setInputShape`, caching programs by shape tuple (the opt shape compiled in `setup`). Record which path was taken in notes.md. `validateProfileDims` compares against what the MIGraphX program was compiled for.
  - Precision: `fp32` and `strongly-typed` -> no quantization (types come from the model). `fp16` -> `quantize_fp16(prog)` before compile. `int8` -> Q2. `getPrecision`/`isStronglyTyped` unchanged.
  - Engine cache: TensorRT writes `engine_path` (default `<onnx>.engine`) and validates bytes 24-26 against the TensorRT version. MIGraphX: `migraphx::save(prog, path)` / `load` of the compiled program as `.mxr`; key the file by MIGraphX version, gfx target and precision (encode in the file name next to `engine_path`, for example `<stem>.<gfx>.<precision>.mxr`, or a sidecar), and rebuild on mismatch the way `validateEngine` does. The `.engine` path must not be overwritten with an MXR file.
  - Execution: `setTensorAddress` / `setInputShape` / `setTensor*` record pointer and dims per name (index overloads go through the index table). `enqueueV3(cudaStream_t stream)` (spelled with the shimmed type, which is `hipStream_t`): build `program_parameters` with `argument(shape, ptr)` for every input and for every output buffer, then `program.run_async(params, stream)` (check the stream pointer convention against MIGraphX's own `run_async` example). Keep the host-profiler timing around it as upstream does.
  - `getTensorShape(name)`, `getTensorDataType`, `getTensorIOMode`: from the compiled program; map `migraphx_shape_datatype_t` to `DataType`.
  - `getBuilderConfig()`, `getNetwork()`: no MIGraphX counterpart; return `nullptr` and document it. autoware_universe's two callers (diffusion_planner touches the builder config; camera_streampetr forces sigmoid/softmax layers to FP32 through `getNetwork()`) need their own handling there; for streampetr the MIGraphX equivalent is excluding those ops from `quantize_fp16` (`quantize_op_names`). Offer that as an optional HIP-only `TrtCommonConfig` field only if autoware_universe's plan asks for it.
  - Profiling: per-layer `IProfiler` callbacks have no public MIGraphX API; `getModelProfiler()` returns an empty profiler and `printProfiling` prints the host profile. `getLayerInformation` writes the compiled program's text form instead of TensorRT's JSON layer dump.
- `TrtConvCalib` (`tensorrt_conv_calib.hpp`, header-only, walks `INetworkDefinition` layers to set per-layer FP16 and to exclude first/last layers from INT8): on the HIP branch the layer walk is meaningless. Its constructor/`setup(calibrator, calib_config, ...)` signature stays; behaviour per Q2.
- `conv_profiler.hpp`: header-only, uses `ILayer` and conv layer getters; on the HIP branch `setProfDict` is never called (no layer objects). It must still compile against the shim.
- `package.xml`: `tensorrt_cmake_module` stays a buildtool dependency (tiny, pure CMake, available as a ROS package); the HIP branch must not need TensorRT itself. Do not add a `migraphx` rosdep key unless one exists; document the MIGraphX requirement in the README instead.
- `ament_export_dependencies`: on the HIP branch export `hip` and `migraphx` instead of `CUDA CUDAToolkit TENSORRT`.
- `-Werror`: MIGraphX's headers under `-Wpedantic` may warn; include them as SYSTEM. Keep the flag set unchanged.

### Interface contract for autoware_universe

Whatever Q1 decides is what autoware_universe's planner builds against; record it there when that project is planned:

- `autoware_cuda_utils` installs the shim directory, so a consumer that links `autoware_cuda_utils` gets `cuda_runtime_api.h` -> HIP mapping for free. autoware_universe needs many more runtime names (memcpy, events, cub, `cuda/std/optional`); it should own its own compat header for those and include it alongside, not grow this package's mapping. Identical macro redefinitions across the two headers are legal, but keep each name defined in one place where possible.
- Under Q1 option A, consumer value types (`nvinfer1::Dims`, `DataType`, calibrator interfaces) come from this package's shim; what consumers must change is only what touches `getNetwork()`, `getBuilderConfig()`, direct TensorRT use outside `TrtCommon` (bevformer builds its own engine), and TensorRT plugins.
- Consumers' CMake currently gates on `TENSORRT_VERSION_MAJOR` / `TENSORRT_FOUND`; the HIP install should export a variable consumers can test instead (for example `autoware_tensorrt_common_BACKEND` = `tensorrt` | `migraphx`).

## CUDA surface inventory

Regex census (`utils/surface.py`): 6 files (`thrust_utils.hpp` and the 5 test `.cu` files), no `__global__`, no `<<<`, no warp intrinsics, no textures, Thrust only (22 hits). The census cannot see `autoware_tensorrt_common` (host C++ against TensorRT) or the CUDA runtime API in `.hpp` files; both are added to surface.json by hand.

`autoware_cuda_utils` runtime names (all have HIP equivalents in ROCm 10.1 headers):

| CUDA | HIP | note |
|---|---|---|
| `cudaError_t`, `cudaSuccess`, `cudaErrorInvalidValue`, `cudaGetErrorName`, `cudaGetErrorString` | `hip*` | `CHECK_CUDA_ERROR` streams the enum value; prints a different integer on HIP, which no test compares |
| `cudaMalloc`, `cudaFree`, `cudaMemsetAsync` | `hip*` | |
| `cudaHostAlloc`, `cudaFreeHost`, `cudaHostAllocDefault` | `hipHostAlloc` / `hipHostMalloc`, `hipHostFree`, `hipHostAllocDefault` | `hipHostAlloc` is deprecated in newer HIP; map to `hipHostMalloc` if it warns under `-Werror` |
| `cudaMemPool_t`, `cudaMemPoolProps`, `cudaMemPoolCreate`, `cudaMemPoolSetAttribute`, `cudaMemPoolAttrReleaseThreshold`, `cudaMemAllocationTypePinned`, `cudaMemLocationTypeDevice`, `cudaMallocFromPoolAsync`, `cudaFreeAsync` | `hip*` | stream-ordered allocator; supported on Linux ROCm. The pooled `make_unique` already memsets (ROCm does not zero) |
| `cudaStream_t`, `cudaStreamCreate`, `cudaStreamCreateWithFlags`, `cudaStreamDestroy`, `cudaStreamSynchronize`, `cudaStreamQuery`, `cudaStreamGetFlags`, `cudaStreamDefault`, `cudaStreamNonBlocking` | `hip*` | |
| `cudaPointerGetAttributes`, `cudaPointerAttributes`, `.type`, `cudaMemoryTypeHost`, `cudaMemoryTypeDevice` | `hipPointerGetAttributes`, `hipPointerAttribute_t`, `.type`, `hipMemoryTypeHost`, `hipMemoryTypeDevice` | field is `type` in current HIP (checked); test-only |
| `cudaGetDeviceCount` | `hipGetDeviceCount` | gtest skip helper |
| `thrust::device_vector`, `fill`, `fill_n`, `count`, `thrust::cuda::par.on` | rocThrust; `thrust::hip::par.on` | the one guarded edit |

`autoware_tensorrt_common`: `cudaSetDeviceFlags`, `cudaDeviceScheduleBlockingSync`, `cudaStream_t` (via the cuda_utils shim or its own include), and the TensorRT API: `createInferRuntime`, `createInferBuilder`, `createNetworkV2` (incl. `kSTRONGLY_TYPED`), `createBuilderConfig`, `nvonnxparser::createParser` / `parseFromFile`, `createOptimizationProfile` / `setDimensions` / `addOptimizationProfile`, `BuilderFlag::kFP16/kINT8/kPREFER_PRECISION_CONSTRAINTS/kGPU_FALLBACK`, `setMemoryPoolLimit(kWORKSPACE)`, `buildSerializedNetwork`, `deserializeCudaEngine`, `createExecutionContext`, `setTensorAddress`, `setInputShape`, `enqueueV3`, `getProfileShape`, `IEngineInspector`, `IProfiler`, `setDLACore`, `initLibNvInferPlugins`, `IInt8Calibrator` and `ILayer::setPrecision` (in `TrtConvCalib`). The MIGraphX mapping for each is in "Port strategy".

## Risk list

- **MIGraphX availability on the validators.** The fleet's ROCm is the pip SDK (`rocm-sdk-devel` 10.1 on this host), which carries no MIGraphX (no `migraphx.hpp` or `libmigraphx*` anywhere on this host). Options in order: MIGraphX's pip repository (`migraphx-libs`, which the ROCm docs name as the ONNX Runtime MIGraphX EP's dependency; check whether it ships headers and a CMake config); `apt install migraphx` from repo.radeon.com, which needs a matching apt ROCm and root; building AMDMIGraphX from source in the job directory against the pip SDK (needs MIOpen, rocBLAS, hipBLASLt, rocMLIR, protobuf, msgpack, nlohmann_json, sqlite3; heavy but authorized). Record the working recipe in notes.md under the moat-porting fleet recipes.
- **MIGraphX on wave32.** MIGraphX coverage is broadest on Instinct (gfx90a/gfx942/gfx950). Confirm the wave32 platform's gfx target is supported by the MIGraphX build used before attempting the wave32 gate; `autoware_cuda_utils` is unaffected.
- **ROS 2 on the validators.** No fleet image has ROS 2. Ubuntu 24.04 hosts: ROS 2 Jazzy from packages.ros.org needs root; without root use RoboStack (conda-forge `ros-jazzy-*`) in the job directory. Also needed from source in the workspace: `autoware_cmake`, `autoware_lint_common` (autowarefoundation/autoware_cmake) and `tensorrt_cmake_module` (autowarefoundation/tensorrt_cmake_module) if not packaged.
- **Exported CMake extras run in consumers.** `find_cccl.cmake` runs `find_package(CUDAToolkit REQUIRED)` in every downstream package; a HIP install that still exports it breaks every autoware_universe consumer at configure. Verify with a scratch consumer package, not only with the package's own build.
- **Dynamic shapes.** TensorRT profiles (min/opt/max per input) and MIGraphX dynamic dimensions differ; the dummy network uses a dynamic batch (`[-1, W]`, min 1 / opt 8 / max 16). A per-shape compile cache is the fallback (see strategy); its compile time on first use of a shape is a behaviour difference to document.
- **IO ordering and names.** Positional IO access (`getIOTensorName(i)`, index setters) must match TensorRT's order exactly or consumers bind the wrong buffers silently. Test it explicitly against the dummy network.
- **INT8 semantics** (Q2). TensorRT's implicit-quantization calibrators, calibration caches and per-layer precision constraints do not carry over; equal outputs on the dummy network do not prove a real model's accuracy.
- **Output buffers with `offload_copy` off.** MIGraphX writes outputs into the caller's buffers only when they are passed as output parameters; a missing output argument allocates internally and the caller's buffer stays stale. Test that outputs land in the caller's memory.
- **Shim header collision.** The HIP-only `NvInfer.h` shim must never be found next to a real TensorRT install. Guarded by backend selection; the reviewer should check the include paths on both branches.
- **`-Werror`** in `autoware_tensorrt_common` and Autoware's default warnings in `autoware_cuda_utils`, against HIP and MIGraphX headers.
- **Upstream reception.** A MIGraphX backend is new code in a TensorRT-named package and introduces a stand-in `nvinfer1` namespace on the HIP branch. TIER IV maintainers may prefer a separate package; that is Q1.
- **Wavefront, textures, OOB, texture pitch:** not applicable (no kernels in the package; the five test files launch none either, they use Thrust and runtime calls).

## File-by-file change list

`autoware_cuda_dependency_meta/`: none expected (confirm by build).

`autoware_cuda_utils/`:
- `CMakeLists.txt`: HIP branch (backend option, `find_package(hip)` + `rocthrust`, INTERFACE target links and shim include dir, HIP test executable, HIP extras selection). CUDA branch untouched.
- `cmake/find_rocthrust.cmake` (new): HIP consumer extras.
- `include/autoware/cuda_utils/hip/cuda_runtime_api.h`, `.../hip/cuda_runtime.h`, `.../hip/cuda_to_hip.h` (new): the shim.
- `include/autoware/cuda_utils/thrust_utils.hpp`: guarded `thrust::hip::par`.
- `package.xml`: no change expected (rosdep has no ROCm keys; document the requirement in README).
- `README.md`: a short ROCm build section.

`autoware_tensorrt_common/`:
- `CMakeLists.txt`: MIGraphX branch (find hip + migraphx, sources, links, shim include dir, exports, test wiring with `autoware_cuda_utils`).
- `include/autoware/tensorrt_common/migraphx/NvInfer.h`, `NvInferRuntime.h`, `NvInferPlugin.h`, `NvOnnxParser.h` (new, HIP-only): value-type shim.
- `include/autoware/tensorrt_common/tensorrt_common.hpp`: `#if` around the private section only.
- `src/tensorrt_common.cpp`: backend-neutral IO reconciliation moved out (code motion only).
- `src/tensorrt_common_io.cpp` (new, both branches): the moved code.
- `src/migraphx_common.cpp` (new, HIP branch): the MIGraphX implementation.
- `include/autoware/tensorrt_common/tensorrt_conv_calib.hpp`: HIP-branch body per Q2.
- `src/profiler.cpp`, `profiler.hpp`, `conv_profiler.hpp`, `logger.hpp`, `utils.hpp`: compile against the shim; no change expected.
- `test/tensorrt_common_test.cpp`: unchanged if possible (it already gates on `SKIP_TEST_IF_CUDA_UNAVAILABLE`, which the cuda_utils shim maps to a HIP device query). Add MIGraphX-specific cases (IO order, output buffers land in caller memory, engine-cache reload, fp16 build) in a separate HIP-only test file so the CUDA test file is untouched.
- `README.md`: ROCm/MIGraphX section.

Not touched: `.github/` workflows (MOAT adds no fork Actions; upstream's NVIDIA CI stays as is), `packages_above.repos`, `setup.cfg`, lint configs.

## Build commands

In a ROS 2 Jazzy (or Humble) environment with ROCm and MIGraphX available, from a colcon workspace whose `src/` holds the fork plus `autoware_cmake` and `tensorrt_cmake_module`:

```bash
source /opt/ros/jazzy/setup.bash          # or the RoboStack env activation
export ROCM_PATH=<rocm root>              # pip SDK: python -m rocm_sdk path --root
colcon build --packages-up-to autoware_cuda_utils autoware_tensorrt_common \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DAUTOWARE_CUDA_USE_HIP=ON \
               -DCMAKE_HIP_ARCHITECTURES=<gfx> -DCMAKE_PREFIX_PATH="$ROCM_PATH;<migraphx prefix>"
```

Plus a scratch consumer package that `find_package`s both installed packages, includes their headers from a `.cpp` and a `.hip`/`.cu` file, and links them, to prove the exported extras and include paths (no CUDAToolkit lookup on the HIP install).

CUDA path no-regression (mi355 fleet host, nvcc available): configure and build both packages on the CUDA branch with real nvcc and TensorRT 10 headers. TensorRT's public headers are in NVIDIA/TensorRT on GitHub (`include/`); libraries come from the `tensorrt-cu12`/`tensorrt-cu13` pip wheels. Compiling and linking is the gate there; running needs an NVIDIA GPU and is not required.

## Test plan

GPU gate on every validated platform (each an AMD GPU, Linux):

```bash
colcon test --packages-select autoware_cuda_utils autoware_tensorrt_common --event-handlers console_direct+
colcon test-result --verbose
```

- `test_autoware_cuda_utils`: all cases in the five `.cu` files (check_error 2, unique_ptr, utils, stream_unique_ptr, thrust_utils) pass on the GPU, none skipped. A skip means the HIP device query failed and is a failure of the gate.
- `autoware_tensorrt_common_test`: the 3 `TensorInfoTest` CPU cases and all 10 `TrtCommonIOTest` cases pass on the MIGraphX backend against `test/data/dummy_network.onnx`, none skipped.
- The new HIP-only MIGraphX test file: IO order equals ONNX order; inference output equals a CPU reference for the dummy network (fp32 exact within tolerance, fp16 within a stated tolerance); outputs land in the caller's device buffers; the engine cache is written, reloaded without recompiling, and rebuilt when the key changes; dynamic batch at min, opt and max; int8 per Q2.
- ament lint tests (`ament_lint_auto`) for both packages must not regress: run them on the HIP build (copyright, cpplint, uncrustify where enabled, xmllint, lint_cmake).

Non-GPU regression: the CUDA branch must configure, compile and link unchanged with nvcc + TensorRT 10 (see Build); `TensorInfoTest` passes on any build.

Gates: `wave64` and `wave32` on Linux; `windows` is waived (approved by Jeff Daily 2026-11-11). wave32 for `autoware_tensorrt_common` depends on MIGraphX supporting that target (Risk list).

## Open questions

Q1-Q4 are stated with options at the top of notes.md's `## Planning` section and must be ruled by Jeff before a porter starts. Technical questions the porter resolves and records (no ruling needed): dynamic-shape strategy (native dynamic compile vs per-shape cache), MIGraphX `run_async` stream convention, ONNX input-order source, MIGraphX install recipe per host, whether `-fno-lto` is needed with HIP.
