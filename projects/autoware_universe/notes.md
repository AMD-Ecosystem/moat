# autoware_universe notes

## Intake screen (2026-11-05, intake agent, windows-gfx1151 host)

Source: third-party "AMD AI Software Gap Matrix" seed ("Autoware perception (TensorRT-based nodes)", verdict not-yet, AMD path none). Treated as a lead; findings below are from upstream at `03961e45e0bdc95bab79a49a420b053a5be7bc22` (shallow clone in `agent_space/aw-screen`).

### Licence

- `utils/licenses.py check`: Apache-2.0, tier 1, cleared to contribute. Read LICENSE: plain Apache-2.0.
- `utils/licenses.py scan-nvidia agent_space/aw-screen`: no NVIDIA proprietary licence text.
- Per-file headers in all 16 CUDA-bearing packages read by hand: Apache-2.0 (TIER IV, Autoware Contributors, Multicoreware), BSD-3 (occupancy grid map, ROS-derived), MIT (torch_scatter-derived segment_csr in autoware_tensorrt_plugins, Matthias Fey 2020; NVIDIA 2020 MIT calibrator.hpp in tensorrt_yolox/classifier), CC0 (a small helper in centerpoint/transfusion/streampetr). Every NVIDIA header found is `SPDX-License-Identifier: Apache-2.0`. `autoware_tensorrt_bevformer/TensorRT/` carries code "derived from the BEVFormer TensorRT implementation by Derry Lin", stated Apache-2.0. No submodules, no per-package LICENSE files. Clean.
- External GPU dependencies: `autowarefoundation/autoware_cuda` (Apache-2.0; holds `autoware_cuda_utils`, `autoware_tensorrt_common`, `autoware_cuda_dependency_meta`) and `autowarefoundation/cuda_blackboard` (Apache-2.0).

### Duplicate effort

- Upstream PRs/issues searched (`hip OR rocm OR amd OR migraphx`, all states): nothing relevant. No `hip|rocm|amd|migraphx` branches. README/docs grep: no AMD mention. Same searches on `autoware_cuda` and `cuda_blackboard` PRs: nothing.
- No ROCm/AMD-Ecosystem repo named autoware; no GitHub fork found doing a HIP port.
- AMD presence in the Autoware org exists but elsewhere: `autowarefoundation/auto_speed` has an AMD HPO tutorial (`Models/tutorials/auto_speed_hpo_amd`), `autoware.trucks` VisionPilot and `vision_pilot_model_recipes` mention ROCm in model CMake (inference runtime selection), not universe.
- AMD Silo AI announced a collaboration with the Autoware Foundation on 2025-12-16 (https://www.amd.com/en/blogs/2025/advancing-open-autonomous-driving-perception-at-scale.html): training and deployment on Instinct / Radeon Pro, initial focus the AutoDrive end-to-end model, "open source releases" promised without repos or dates. Nothing public touches autoware_universe's CUDA kernels. This is a coordination question, not a duplicate port (see Questions).
- MOAT: no prior record or disposition for autoware. A stale `origin/port/spconv` branch exists but holds no spconv record (`moatlib.py show spconv`: no record), so it is not a dependency to wire.

### Viability and the realistic portable unit

About 797 KB of `.cu/.cuh` across 16 packages. Split by what the CUDA serves:

Pure CUDA, no TensorRT (the realistic first unit, about 250 KB, 31%):
- `sensing/autoware_cuda_pointcloud_preprocessor` (153 KB): crop box, distortion correction, ring outlier filter, voxel downsample, concatenate-and-time-sync. cub, thrust, `cuda/std/optional` (libhipcxx), legacy `find_package(CUDA)` + `cuda_add_library` with sm_75..sm_120 gencode lists. Has gtests (`test/test_cuda_pointcloud_preprocessor.cpp`, `test/cuda_concatenate_data`).
- `perception/autoware_ground_segmentation_cuda` (33 KB): same pattern, depends on the preprocessor.
- `perception/autoware_probabilistic_occupancy_grid_map` (61 KB): CUDA path is optional (CPU fallback when CUDA is not found); has gtests and launch tests.
- Shared base outside this repo: `autoware_cuda_utils` (11 KB Cuda, helpers: memory pool, unique_ptr, thrust_utils, gtest utils) and `cuda_blackboard` (C++ only, CUDA runtime API for GPU-resident ROS 2 message transport). Both must build on ROCm first. Neither is a MOAT project, so nothing for `set-deps` yet; they need their own intake (or to be carried in the same planner scope as companion forks).

TensorRT-bound (the matrix's actual entry; about 545 KB, 69%):
- Pre/postprocessing kernels around a TensorRT engine: lidar_centerpoint, lidar_transfusion, bevfusion, lidar_frnet, ptv3, camera_streampetr, tensorrt_yolox, tensorrt_classifier, image_projection_based_fusion, calibration_status_classifier, e2e/autoware_tensorrt_vad, plus TensorRT-only nodes with no .cu (traffic_light_classifier/fine_detector, shape_estimation, simpl_prediction, diffusion_planner, lidar_apollo_instance_segmentation).
- Custom TensorRT plugins: `autoware_tensorrt_plugins` (85 KB: spconv implicit-GEMM via cumm/spconv, multi-scale deformable attention, segment_csr, unique, argsort, rotate, quick_cumsum) and `autoware_tensorrt_bevformer` (133 KB, own plugin set).
- All engine handling goes through `autoware_tensorrt_common` (in autoware_cuda): builds engines from ONNX, INT8 calibration, plugin registration. The custom kernels hipify mechanically, but they are useless on ROCm without an inference backend. MIGraphX is the only same-shape option (ONNX in, compiled program out); it would mean a new `autoware_migraphx_common`-style backend plus re-implementing every TensorRT plugin as a MIGraphX custom op, and MIGraphX support for consumer RDNA targets is thinner than for Instinct. Plausible for the plugin-free models (yolox, classifier, centerpoint/transfusion whose ONNX uses standard ops and does voxelization/NMS in our CUDA); not plausible for bevfusion/ptv3/bevformer/VAD without the spconv and deformable-attention plugins. This is a multi-package re-architecture an upstream is unlikely to take as one PR, and it is the planner's call whether to attempt it, as a separate later phase.

Platform: Autoware is a Linux-only ROS 2 (Humble/Jazzy) stack. The `windows` gate is very unlikely to be satisfiable; expect a waiver request if adopted (a person decides).

Upstream: very active (pushed 2026-11-05, 1.8k stars), not archived. The Autoware Foundation's AMD partnership makes a ROCm build path more likely to be welcome than usual, but nothing confirms that.

### Recommendation

Fork (take up), scoped to the pure-CUDA unit: autoware_cuda_utils + cuda_blackboard (companion repos), then cuda_pointcloud_preprocessor, ground_segmentation_cuda and probabilistic_occupancy_grid_map on Linux wave64/wave32, with their gtests as the GPU gate. The TensorRT nodes (what the matrix entry actually names) stay out of scope for the first round; a MIGraphX backend is a separate planner question. Because that is a partial port of what the matrix entry names, the scope cut needs jeff's explicit answer before a porter starts.

### Questions for jeff (left by the intake screen; nobody was available to answer)

1. Scope cut: the only realistic first unit is the non-TensorRT CUDA (pointcloud preprocessor, CUDA ground segmentation, occupancy grid map, about 31% of the CUDA) plus the two companion repos autowarefoundation/autoware_cuda and autowarefoundation/cuda_blackboard. The TensorRT perception nodes the matrix names would need a MIGraphX backend and MIGraphX custom ops for the spconv and deformable-attention plugins. Do you approve starting with the non-TensorRT unit and deferring the TensorRT/MIGraphX work, and should autoware_cuda and cuda_blackboard be screened as their own MOAT projects (dependencies) or carried inside this one?
2. Coordination: AMD Silo AI announced a collaboration with the Autoware Foundation (2025-12-16, AutoDrive end-to-end model on Instinct/Radeon Pro). Nothing public touches autoware_universe's CUDA, but should someone check with that team before we fork, so we are not racing an unpublished internal port?
3. Windows: Autoware is Linux-only ROS 2. If adopted, should the planner request a `windows` gate waiver up front?

## Scope ruling (2026-11-11, Jeff Daily)

Jeff Daily approved the full scope, staged:
1. The two companion repos as their own MOAT projects, recorded as hard dependencies (`set-deps autoware_universe autoware_cuda cuda_blackboard`): https://github.com/autowarefoundation/autoware_cuda and https://github.com/autowarefoundation/cuda_blackboard (both Apache-2.0; claimed and in intake 2026-11-11).
2. The pure-CUDA packages: cuda_pointcloud_preprocessor, ground_segmentation_cuda, probabilistic_occupancy_grid_map, with their gtests as the GPU gate.
3. A MIGraphX backend for the TensorRT perception nodes: an engine layer equivalent to `autoware_tensorrt_common` (ONNX in, compiled program out), plugin-free models first (yolox, classifier, centerpoint, transfusion), then the custom TensorRT plugins (autoware_tensorrt_plugins: spconv implicit GEMM, multi-scale deformable attention, segment_csr, unique, argsort, rotate, quick_cumsum; autoware_tensorrt_bevformer) re-implemented as MIGraphX custom ops, then bevfusion, ptv3, bevformer and VAD. The CUDA/TensorRT path is preserved.

Windows gate: waiver approved by Jeff Daily (Linux-only ROS 2 stack). Coordination with AMD Silo AI (Autoware Foundation collaboration, 2025-12-16) is Jeff's call and was not made a precondition.

## Planning (2026-11-23, planner, linux-gfx942 fleet job; no GPU used, nothing built)

### Decisions for Jeff (rule before the porter of the named stage starts)

Nobody could answer these during the planning job. plan.md is written for the recommended answer to each and says what changes otherwise. The 2026-11-11 full-scope ruling stands; none of these re-opens it.

**Q1. Sparse convolution: how stage 4 gets a ROCm spconv (blocking for stage 4 only: the four spconv plugins in `autoware_tensorrt_plugins`, the bevfusion voxeliser and backbone, ptv3's backbone).** Upstream gets spconv from `autowarefoundation/spconv_cpp` (Apache-2.0), a pre-generated C++ snapshot of spconv 2.3.8 + cumm 0.5.3 for CUDA 13 (`python -m spconv.gencode`; 19k generated files; tensor-core PTX, SM-keyed tuner). It is generated code, so it belongs to its generator: MOAT's `spconv` and `cumm` projects (both `awaiting-fork`, adopted 2026-11-20, HIP via a cumm backend). Options:
- **A (recommended): wait for spconv/cumm, then a ROCm spconv_cpp.** Take `autowarefoundation/spconv_cpp` up as its own MOAT project (intake, then regenerate with the HIP cumm backend, plus its CMake HIP branch), depending on `spconv` and `cumm`. autoware_universe adds `set-deps ... spconv_cpp` only when stage 4 starts (adding it now would hold stages 1-3 behind it). Stage-4 items stay `owed`.
- **B: native HIP sparse conv inside `autoware_tensorrt_plugins`.** A HIP rulebook (hash-table indice pairs) plus gather-GEMM-scatter on hipBLASLt, and a HIP voxeliser for bevfusion (ptv3's sort-based voxeliser is a template). Independent of spconv/cumm, but a large amount of new code for upstream to review, with no tuned implicit GEMM.
- **C: scope stage 4 out** (bevfusion and ptv3 inference and the spconv plugins not ported). A partial port of the ruled scope; listed for completeness only.

**Q2. `autoware_tensorrt_bevdet` (blocking for that package only).** Its GPU code (TensorRT engine, BEVPool/AlignBEV/GatherBEV plugins, IoU3D NMS, preprocessing; about 96 KB CUDA) lives in the external `autowarefoundation/bevdet_vendor`, which has **no licence**: no LICENSE file, GitHub detects none, and its package.xml says `<license>LCH1238</license>` (an author handle). The universe package itself is a thin node. Options:
- **A (recommended): leave bevdet unscheduled** (surface `owed` "awaiting licence ruling") until the licence is clarified; the port does not touch it.
- **B: screen `bevdet_vendor` as a MOAT project**; intake would stop at licence clearance, which is your decision.
- **C: scope it out** with the reason "dependency carries no licence".

**Q3. Where the CUDA-to-HIP compat mapping lives (blocking for stage 1; touches autoware_cuda round 2, now being ported).** `autoware_cuda_utils`' shim maps exactly the 34 runtime names it uses, and its notes say each consumer brings its own header for the rest. autoware_universe needs about 70 runtime names plus `cub/cub.cuh`, `cuda_fp16.h`, `cuda/std/optional`, `cuda/functional`. Options:
- **A (recommended): extend the shim in autoware_cuda round 2** to the full set autoware_universe needs (list in plan.md, "Compat headers"; plus the libcu++ shims), so a consumer that links `autoware_cuda_utils` needs no runtime header of its own. One mapping in one place; costs autoware_cuda a small extra delta in a round that is already open.
- **B: per-package HIP-only compat headers** in autoware_universe, force-included on the HIP branch (`-include`), sources byte-identical. No autoware_cuda change; about 20 near-duplicate headers in the universe diff.
- **C: a new shared package in autoware_universe** (e.g. `autoware_hip_compat`). One copy, but a new package upstream has to accept.

**Q4. Delivery shape (blocking before the first upstream PR, not for porting).** About 25 packages over 4 stages; upstream's norm is one package per PR (semantic titles `feat(autoware_xxx): ...`) with DCO sign-off. Recommended staging either way: one `moat-port` branch, one porting round per stage, later stages `owed` in surface.json (the autoware_cuda Q3 = i pattern), each later stage dispatched deliberately once its gate holds. For the PRs:
- **i: one upstream PR at the end.** Simplest bookkeeping; a 25-package PR is unlikely to be reviewed.
- **ii (recommended): one upstream PR per stage**, stage 1 first (three packages, no TensorRT). Each PR before the last is a partial port of the ruled scope, so this needs your explicit approval; MOAT records one PR per project, so each later stage would go out as a fix/sync-style round or a follow-up PR, which needs a tooling decision at PR time.
- **iii: one PR per package.** Matches upstream's norm best; about 20 PRs and the most tooling work.

**Q5. Agnocast variant (non-blocking for the porter's code).** `autoware_cuda_pointcloud_preprocessor` and the occupancy grid have an optional `ENABLE_AGNOCAST` build (agnocastlib); no CUDA of its own. Same question as cuda_blackboard's open Q1, and the same options: **A (recommended)** default build only, Agnocast `scoped_out` as "optional, no HIP-specific code"; **B** validators also build agnocast. surface.json keeps it `owed` until you rule; a ruling on cuda_blackboard's Q1 can cover both.

### Interface autoware_cuda round 2 must provide (for whoever drives that round; not a question)

From the consumer census below. Under Q1 = A the consumers compile unchanged only if the MIGraphX `TrtCommon` and its stand-in `NvInfer.h` cover all of this:
- Value types: `nvinfer1::Dims`, **`Dims2`, `Dims3`** (transfusion), `DataType` incl. **`kINT64`**, `kINT32`, `kHALF`, `kFLOAT` (ptv3, streampetr), **`TensorIOMode`** (streampetr), the calibrator family incl. **`IInt8LegacyCalibrator`** (yolox and classifier subclass it besides `IInt8EntropyCalibrator2` and `IInt8MinMaxCalibrator`), `CalibrationAlgoType`, `NvInferRuntimeBase.h` and `NvInferRuntime.h` include names (simpl, streampetr `memory.cuh`).
- Methods and overloads: constructors `(config)` and `(config, Profiler, plugin_paths)`; `setup(profiles)` and `setup(profiles, network_io)` incl. ptv3's **optional** NetworkIO/ProfileDims with `std::optional<DataType>`; **integer-keyed** `ProfileDims` and `setInputShape(int, Dims)` (centerpoint); `setTensor(name, ptr[, Dims])` (transfusion); `setTensorsAddresses(vector)`; `getTensorDataType`; profiles with **min 0** dims (bevfusion camera tensors).
- `getNetwork()`/`getBuilderConfig()` return `nullptr` (diffusion_planner already null-checks; streampetr does not and gets a guarded edit here), plus the optional HIP-only `TrtCommonConfig` field to keep named op types (`sigmoid`, `softmax`) out of `quantize_fp16`, which streampetr needs.
- Plugin libraries: the `plugin_paths` dlopen with `register_experimental_custom_op` from a static initialiser in the loaded library, so `autoware_tensorrt_plugins` can ship custom ops at its current install path.
- An exported backend variable (`autoware_tensorrt_common_BACKEND`), and no `CUDA`/`CUDAToolkit` exports on the HIP branch (autoware_cuda review 2026-11-23b, finding B), or every consumer's configure trips `check_gpu_backend.cmake`.
- `autoware::cuda_utils::is_cuda_runtime_available()` and `SKIP_TEST_IF_CUDA_UNAVAILABLE` working on HIP (traffic_light_pipeline and ptv3 tests use them; round 1 already maps `cudaGetDeviceCount`).
- If Q3 = A, the extended runtime/libcu++/CUB shim.

### What was done

- Resumed from the intake section and the 2026-11-11 scope ruling; no earlier planning existed. Took the planning lock (`set-state autoware_universe linux-gfx942 planning --agent planner`), `set-ext-type autoware_universe cmake`.
- Read the dependency records on their branches: autoware_cuda plan, notes and rulings (round 1 at `992d703`, Install as a dependency, the `check_gpu_backend.cmake` contract), cuda_blackboard notes (compat alias list, the `hip::host`-only export lesson, the `hipStreamLegacy` event workaround, ROS 2 Jazzy source recipe).
- Shallow clone of upstream at `6cea1ac` in `projects/autoware_universe/src`. Package census: 21 packages with GPU code or TensorRT (16 with `.cu`, 741 KB), plus bevdet (external code) and bytetrack (declared, unused). Four read-only analysis passes (stage-1 packages; TensorRT plugins and bevformer; lidar consumers; camera and other consumers), and the high-risk claims re-checked by hand: ground segmentation's 32-lane design and masks, segment_csr's mask, select_and_pad's workspace overrun, streampetr's unguarded `getNetwork()`, frnet's raw-pointer `device_vector`, bevfusion's spconv `Point2Voxel`, and the pooled `make_unique` memset in the fork's `cuda_unique_ptr.hpp:76`.
- Header checks against the ROCm 10.1 pip SDK on this host: `thrust::hip::par_nosync` exists (`thrust/system/hip/detail/par.h`); hipCUB has `DeviceMergeSort`, `DeviceRunLengthEncode`, and a transform input iterator; no libhipcxx and no MIGraphX in the SDK.
- Checked `autowarefoundation/spconv_cpp` (Apache-2.0, generated from spconv/cumm) and `autowarefoundation/bevdet_vendor` (no licence) with read-only `gh api` queries. MOAT's `spconv` and `cumm` are `awaiting-fork`; `ccimport` unclaimed.
- Upstream issue #7724 (open): the centerpoint, transfusion and occupancy-grid GPU tests are commented out only because upstream CI has no GPU. The validator builds them from the fork's test sources in a scratch project instead of the port editing them in.

### Package census (6cea1ac)

| package | stage | GPU code | TensorRT / plugins | tests |
|---|---|---|---|---|
| sensing/autoware_cuda_pointcloud_preprocessor | 1 | 8 .cu, 36 kernels; CUB, Thrust `par_nosync`, libcu++ `optional`, `__constant__`, pools | none | 2 gtests (20 GPU + 8 CPU cases) |
| perception/autoware_ground_segmentation_cuda | 1 | 1 .cu + .cuh, 11 kernels; 32-lane warp-synchronous sector kernel | none | none |
| perception/autoware_probabilistic_occupancy_grid_map | 1 | 5 .cu, 13 kernels; 64-bit atomicMin, Eigen in device code, RDC | none | all disabled upstream (#7724) |
| perception/autoware_tensorrt_classifier | 2 | 1 .cu, 10 kernels | TrtConvCalib, 3 calibrator subclasses | none |
| perception/autoware_tensorrt_yolox | 2 | 1 .cu, 3 kernels | TrtConvCalib, calibrators; default model has EfficientNMS_TRT; int8 default for yolox_s_plus_opt | test_label (CPU), 2 GPU+model |
| perception/autoware_traffic_light_classifier | 2 | none | via classifier; TrtCommon direct in lamp recognizer | 4 CPU, 2 GPU+model |
| perception/autoware_traffic_light_fine_detector | 2 | none | via yolox | test_utils (GPU-gated build) |
| perception/autoware_traffic_light_pipeline | 2 | none | via yolox + classifier | 1 GPU+model |
| perception/autoware_lidar_centerpoint | 2 | 4 .cu, 8 kernels; Thrust | 2 TrtCommon, fixed profiles, index overloads | 3 CPU; 3 GPU disabled (#7724) |
| perception/autoware_image_projection_based_fusion | 2 | 1 .cu, 3 kernels | via centerpoint (pointpainting) | 3 CPU, test_pointpainting |
| perception/autoware_lidar_transfusion | 2 | 3 .cu, 6 kernels; Thrust | dynamic profiles, Dims2/Dims3 | 2 CPU; 3 GPU disabled (#7724) |
| perception/autoware_lidar_frnet | 2 | 2 .cu, 3 kernels; half, 64-bit CAS, CUB+Thrust `cuda::par`, `__constant__` | dynamic profiles | lint only |
| sensing/autoware_calibration_status_classifier | 2 | 1 .cu, 4 kernels; 64-bit atomicMin | dynamic H/W profile | test_preprocessing (GPU), 2 GPU+model, 2 CPU |
| perception/autoware_camera_streampetr | 2 | 5 .cu, 8 kernels; Thrust | 3 engines, `getNetwork()` layer walk | none |
| perception/autoware_simpl_prediction | 2 | none | TrtCommon | test_simpl (CPU, 11 sources) |
| perception/autoware_shape_estimation | 2 | none | TrtCommon, `USE_CUDA` macro | test_shape_estimation (CPU) |
| perception/autoware_lidar_apollo_instance_segmentation | 2 | none | TrtCommon | lint only |
| planning/autoware_diffusion_planner | 2 | none | TrtCommon, plugins_path, CUDA graphs, optional ORT CUDA/TRT EP | test_diffusion_planner (mostly CPU) |
| perception/autoware_tensorrt_plugins | 3 (+4) | 7 .cu, 16 kernels; hipCUB; segment_csr mask; select_and_pad overrun | 11 IPluginV3 (4 need spconv) | reference_kernels_test (GPU) |
| e2e/autoware_tensorrt_vad | 3 | 3 .cu, 4 kernels | 3 engines, MSDA/Rotate/SelectAndPad | none |
| perception/autoware_tensorrt_bevformer | 3 | 4 .cu, 20 kernels; half2, dp4a PTX, cuBLAS | own engine code, 3 IPluginV2DynamicExt (x2 variants) | lint only |
| perception/autoware_bevfusion | 4 | 4 .cu, 5 kernels + spconv Point2Voxel | plugins: spconv, QuickCumsumCuda | none |
| perception/autoware_ptv3 | 4 | 4 .cu, 28 kernels; hipCUB, Thrust `cuda::par`, events, pinned | 3 engines, int64 and optional IO; SegmentCSR + spconv | 2 CPU, 5 GPU synthetic |
| perception/autoware_tensorrt_bevdet | Q2 | in bevdet_vendor (no licence) | own engine | none |
| perception/autoware_bytetrack | none | none (unused GPU deps in package.xml) | none | none |

### Notes for the porters

- Stage 1 can start once Q3 is ruled (and with Q3 = A, once autoware_cuda round 2 carries the extended shim; the porter may start the CMake and kernel work earlier, since only the header location moves).
- autoware_cuda's notes: hosts with both toolkits need `-DAUTOWARE_CUDA_USE_HIP=ON`, and the consumer must skip `find_package(CUDA)` then; `utils/timeit.sh` `cd`s to the moat root, so wrap `colcon` with explicit `--base-paths/--build-base/--install-base/--log-base`; every colcon build needs `-DPython3_EXECUTABLE=<venv python>`; installed binaries need `$ROCM_SDK/lib` on `LD_LIBRARY_PATH`.
- cuda_blackboard's `hipStreamLegacy` workaround already covers subscribers built with the deprecated stream-less constructor (ground segmentation uses the deprecated cuda_blackboard API, `-Wno-error=deprecated-declarations`).
- Ground segmentation is the stage-1 item to get right on both widths; plan.md gives the sub-wave fix and the validator's sector-boundary scene.

## Rulings on the planner's questions (Jeff Daily, 2026-11-23)

- Q1 (ROCm spconv for stage 4): **A**. Take `autowarefoundation/spconv_cpp` up as its own project depending on `spconv` and `cumm`; add it to `depends_on` only when stage 4 starts.
- Q2 (bevdet): **B**. Screen `autowarefoundation/bevdet_vendor` as a project; its licence clearance is Jeff's decision when intake reaches it. bevdet stays `owed` here meanwhile.
- Q3 (where the CUDA-to-HIP compat mapping lives): **B** (ruled later the same day). Per-package HIP-only compat headers in autoware_universe, force-included on the HIP branch; no autoware_cuda change for it. This fits Q4 = iii: each package PR carries its own header.
- Q4 (delivery shape): **iii**, one upstream PR per package, following upstream's norm.
- Q5 (Agnocast variant): **A**. Default build only; the Agnocast variant is scoped out because it has no GPU code (same ruling as cuda_blackboard).
