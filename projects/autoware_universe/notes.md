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

## Porter, stage 1 (2026-11-23, porter agent, linux-gfx942 fleet job; AMD Instinct MI300X, HIP_VISIBLE_DEVICES=3 pinned by the dispatcher)

Stage 1 only, as Jeff dispatched it: the three pure-CUDA packages. Every stage 2-4 surface entry stays `owed`; no TensorRT consumer was touched. Fork `moat-port` = upstream `6cea1ac` + 5 commits, head **`9574fe17a463fa3384bfa0aa0f715923deb9787f`**, pushed. Built and tested at that tree.

### Open for a person (nothing below was decided by the porter)

1. **Fork default branch.** `moat-port` now exists on AMD-Ecosystem/autoware_universe; making it the fork's GitHub default branch (the standing practice for MOAT forks) is a repository-settings change a person makes. The recorded `fork_default_branch` stays `main`.
2. **DCO sign-off.** Upstream requires DCO (`Signed-off-by`). The fork commits carry none, matching the autoware_cuda port. Whether the per-package upstream PRs (Q4 = iii) add sign-off, and under whose name, is Jeff's call at PR time; adding it then means new commits or a person-made rewrite.
3. **CUDA compile gate owed.** This host has no nvcc. The CUDA branches are unchanged by reading (the only CUDA-visible edits: shuffle `width = WARP_SIZE`, identical on a 32-thread warp; `[[nodiscard]]` moved before `__device__`; source lists moved into variables; `if(NOT ${CUDA_FOUND})` -> `if(NOT CUDA_FOUND)`), but the rule is a real nvcc compile on the mi355 host before this is brought to Jeff.
4. **libhipcxx requirement.** The pointcloud preprocessor's libcu++ use (`cuda::std::optional` in device code, `cuda::proclaim_return_type`) is served by libhipcxx, present in the TheRock ROCm 10.2 SDK (`include/hipccl`, `lib/cmake/libhipcxx`) but not in the ROCm 10.1 pip SDK the planner checked. The package's HIP branch requires it (configure warns and skips the package without it, or fails under `AUTOWARE_CUDA_USE_HIP=ON`). A `std::` shim was tried and rejected: libstdc++'s `std::optional::value()` calls a host-only throw helper, which hip-clang refuses in device code. Reviewer: confirm this minimum is acceptable, and the wave32 validator needs an SDK with libhipcxx.

### Dependency status as the selector reported it

`moatlib.py deps`: `autoware_universe: depends_on=['autoware_cuda', 'cuda_blackboard'] -> READY (deps complete)`. Built from the forks: `autoware_cuda_utils` at `992d7034b5396900593126e8f19a1982ae7bc682` (configure: `cuda is not found, building autoware_cuda_utils with HIP 7.17.26434`; `autoware_tensorrt_common` COLCON_IGNOREd), `cuda_blackboard` at `moat-port` tip `9e0fb865cd2e38dcc0db5e0e55497f40fb71f733` (configure: `cuda was not found, building cuda_blackboard with HIP 7.17.26434`).

### What changed on the fork (one commit set per package, per Q4 = iii; each package carries its own compat header, per Q3 = B)

- `5fb1c77` `fix(autoware_cuda_pointcloud_preprocessor): put [[nodiscard]] first` -- shared fix, no prefix: `__device__ [[nodiscard]] inline` is rejected by clang ("an attribute list cannot appear here"); nvcc takes both orders.
- `175821e` `[ROCm] Build autoware_cuda_pointcloud_preprocessor with HIP` -- CMake HIP branch (`AUTOWARE_CUDA_USE_HIP` option; skips `find_package(CUDA)` when ON; `find_package(hip/hipcub/libhipcxx)`; `enable_language(HIP)`); `hip/include/cub/{cub.cuh,device/device_radix_sort.cuh,device/device_run_length_encode.cuh}` -> hipCUB with `namespace cub = hipcub`; `hip/include/cuda_pointcloud_preprocessor_hip_compat.hpp` force-included (`$<$<COMPILE_LANGUAGE:HIP>:-include;...>`) with the runtime-name map, `<cuda/std/cmath>` and `<cuda/std/functional>` (CCCL pulls them in transitively on CUDA; the sources use `cuda::std::abs` on floats and `cuda::std::not_equal_to` without including them), and `namespace thrust { namespace cuda = ::thrust::hip; }` for the four `thrust::cuda::par_nosync` sites (sources unchanged). `cudaMemcpyToSymbolAsync` needs no `HIP_SYMBOL`: HIP 7.17 has the `const T&` template overload. All package sources (node `.cpp` and tests too) compile as HIP, because libhipcxx headers do not compile under g++ (`'__host__' does not name a type`) and the installed outlier-filter header exposes `cuda::std::optional` in member signatures. The lib links `autoware_cuda_utils::autoware_cuda_utils` (cuda_runtime.h shim, rocThrust includes, `hip::host`) and `libhipcxx::libhipcxx`; hipCUB comes in as include directories only (no `hip::device`, cuda_blackboard lesson). README section.
- `60bb32c` `[ROCm] Keep ground segmentation warp ops within 32-thread groups` -- the plan's fix: every `__shfl*_sync` gets `width = WARP_SIZE` (13 sites); the two ballots go through a `warpBallot` helper that on HIP shifts `__ballot_sync` right by `__lane_id() & ~(WARP_SIZE - 1)`; on HIP `FULL_MASK` is `0xFFFFFFFFull << (__lane_id() & ~(WARP_SIZE - 1))`, the calling group's 32 lanes (a full 64-bit mask would trap in a debug build when the two groups of a wavefront diverge, `__hip_check_mask`). Correct on wave32 too (offset 0). `WARP_SIZE` stays the logical group size 32.
- `caae48b` `[ROCm] Build autoware_ground_segmentation_cuda with HIP` -- same CMake pattern; `hip/include/cub/cub.cuh`, `hip/include/cuda_ground_segmentation_hip_compat.hpp`; node source compiled as HIP. README section.
- `9574fe1` `[ROCm] Build autoware_probabilistic_occupancy_grid_map with HIP` -- HIP branch builds `<pkg>_cuda` from the 5 `.cu` with `-fgpu-rdc` (compile and link, link with `--hip-link --offload-arch=<CMAKE_HIP_ARCHITECTURES>`): `occupancy_grid_map_{fixed,projective}_kernel.cu` call `raytrace`/`setCellValue` defined in `utils_kernel.cu` (why the CUDA build sets `CUDA_SEPARABLE_COMPILATION`). All package sources compile as HIP with `hip/include/probabilistic_occupancy_grid_map_hip_compat.hpp` force-included. `autoware_package()` turns on `CMAKE_COMPILE_WARNING_AS_ERROR`, which nvcc-built `.cu` never saw: the HIP kernel target sets `COMPILE_WARNING_AS_ERROR OFF` (sign-compare/unused warnings in upstream kernels), and HIP sources get `-Wno-error=unused-value` (`hipError_t` is `[[nodiscard]]`; upstream ignores some results deliberately). The CPU-only fallback (no CUDA, no HIP) is unchanged. README section.

`python3 utils/jargon.py --port autoware_universe`: clean. clang-format 23.1.1 (the repository's pre-commit version) applied to new and edited C++ files.

### Environment (job directory only; nothing shared on the host was modified)

- ROCm: TheRock ROCm 10.2.0 wheels in `$FLEET_VENV` (`$FLEET_VENV/bin/pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple 'rocm[devel,device-gfx942]==10.2.0'`, `rocm-sdk init`), HIP 7.17.26434, AMD clang 24.0.0git. gcc 13.3, CMake 3.31.6, Python 3.11.17.
- A helper agent built the dependency closure; its full report (every repo, commit, flag and timing) was written to `<job>/logs/env_report.md`. Summary:
  - ROS 2 Jazzy from source, merge-install underlay (198 packages; cuda_blackboard recipe plus tier4 glog; iceoryx COLCON_IGNORE; Fast-DDS asio/tinyxml2 submodules; `-DTRACETOOLS_DISABLED=ON -DCV_BRIDGE_DISABLE_PYTHON=ON -DENABLE_SHM=OFF`).
  - System libraries into `<job>/deps/prefix`: Boost 1.83 (compiled libs), Eigen 3.4.0, lz4 1.9.4, FLANN 1.9.2, OpenCV 4.6.0 (core imgproc imgcodecs highgui photo calib3d features2d flann), PCL 1.14.1 (no VTK/OpenNI/realsense/pcap/qhull, no -march=native), GMP 6.3.0, MPFR 4.2.1, CGAL 5.6.1, fmt 9.1.0, Sophus 1.22.10, GeographicLib 2.3, pugixml 1.14, range-v3 0.12.0, tl-expected 1.1.0, tinyxml2 10.0.0, GraphicsMagick 1.3.43.
  - Overlay `<job>/aw_ws` (74 packages). **Autoware repos are not at the autoware.repos tags**: universe main (6cea1ac) needs newer autoware_core APIs (`autoware::point_types::is_data_convertible_to`, autoware_core c3ebc08d, in no tag), so autoware_core `15408eb6`, autoware_utils `055603da`, autoware_internal_msgs `fe5ab82d`, autoware_adapi_msgs `35a7aeb9`, managed_transform_buffer `6c9a65b5`, tier4_autoware_msgs (tier4/universe) `efa41293` are each the last commit before the universe tip's time; autoware_msgs 1.13.0, autoware_cmake 1.4.0, autoware_lanelet2_extension 1.3.0, tier4 pointcloud_to_laserscan `d969ec69` stay on their pins. Third party (Jazzy branches): grid_map `d9614992`, navigation2 `c7046550`, perception_pcl `c5e7e72d`, pcl_msgs `8a925a7c`, diagnostics `20667d1d`, angles `a96224f9`, filters `42c04007`, bond_core `8f024dde`, Lanelet2 1.2.3, mrt_cmake_modules 1.0.11, point_cloud_msg_wrapper `5eabe4cc`, geographic_info `f70b81a4`. Agnocast not cloned (Q5 = A).
  - `env.sh` sets `CPATH`/`LIBRARY_PATH` to `deps/prefix` (autoware_utils_geometry relies on Boost in /usr/include; polygon_remover links bare `gmp`).
- **Environment defect, not the port:** the job-built `libdistortion_corrector_core.so` (autoware_pointcloud_preprocessor) uses `fmt::v9` symbols without a `NEEDED` entry for libfmt, so linking an executable against it fails under ld.lld's `--no-allow-shlib-undefined` (the HIP link driver uses lld). Workaround for this environment only: `-DCMAKE_EXE_LINKER_FLAGS="-Wl,--no-as-needed -L<job>/deps/prefix/lib -lfmt -Wl,--as-needed"` on the stage-1 builds. A validator reusing this recipe needs the same (or an Ubuntu-packaged fmt/spdlog closure).
- The `cppcheck` lint test fails because the cppcheck binary is not installed in this environment; copyright, cpplint and lint_cmake pass on all three packages.

### Build

```
source <job>/env.sh
colcon --log-base log build --base-paths src --build-base build --install-base install \
  --packages-select autoware_cuda_pointcloud_preprocessor autoware_ground_segmentation_cuda autoware_probabilistic_occupancy_grid_map \
  --cmake-args -DCMAKE_BUILD_TYPE=Release "-DCMAKE_PREFIX_PATH=$ROCM_SDK;<job>/deps/prefix" -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DAUTOWARE_CUDA_USE_HIP=ON -DPython3_EXECUTABLE=$FLEET_VENV/bin/python -DBUILD_TESTING=ON "<fmt workaround above>"
```
`src/` holds symlinks to the three fork packages. rc 0. Configure logs: `cuda was not found, building autoware_cuda_pointcloud_preprocessor with HIP 7.17.26434`, `... autoware_ground_segmentation_cuda with HIP 7.17.26434`, `... autoware_probabilistic_occupancy_grid_map with HIP 7.17.26434`. Remaining warnings are upstream's (unused private fields, sign-compare, `-Wreorder-ctor`) plus HIP's `[[nodiscard]]` on ignored `hipError_t`.

Errors met on the way, each fixed in the commits above: the `[[nodiscard]]` order (3 sites); `cuda::std::abs` ambiguous and `::cuda::std::not_equal_to` undeclared (transitive CCCL includes); 64-bit mask static_assert in ground segmentation (fixed by the group mask); `-Werror` from `CMAKE_COMPILE_WARNING_AS_ERROR` on the occupancy-grid kernels and on ignored `hipError_t` results.

### Tests (all under `ulimit -c 0`, with `timeout`, HIP_VISIBLE_DEVICES=3)

- **autoware_cuda_pointcloud_preprocessor:** `colcon test --packages-select autoware_cuda_pointcloud_preprocessor --ctest-args -R test_cuda --repeat until-fail:20`: test_cuda_combine_cloud_handler 8/8 and test_cuda_pointcloud_preprocessor 20/20 (12 device + 8 CPU), each 20 times: 40 `[  PASSED  ]` blocks, 0 failed, 0 skipped. `AMD_LOG_LEVEL=3 ./test_cuda_pointcloud_preprocessor`: `Gfx Major/Minor/Stepping: 9/4/2`, `Device ID: 0x74a1`, 1070 `hipLaunchKernel`. The noise, outlier and downsample filter nodes have no upstream test; they compile and link (validator: run them in the example launches).
- **autoware_ground_segmentation_cuda (no upstream tests):** harness kept in `projects/autoware_universe/harness/ground_segmentation/` (`harness.cpp`, `CMakeLists.txt`, `make_naive.sh`). It builds a synthetic scene (seeded `std::mt19937`, 163,196 points over all 361 one-degree sectors; terrain by `sector % 4`: flat, climbing 0.07, descending 0.03, flat-then-ramp, so the two sectors sharing a 64-wide wavefront always differ; obstacles 0.7-1.5 m above ground; empty-cell gaps in every 7th sector for DISCONTINUOUS/BREAK modes; sparse far cells; points kept clear of sector and cell boundaries), runs `CudaScanGroundSegmentationFilter::classifyPointCloud` from the installed `libcuda_ground_segmentation_lib.so`, and compares every point's ground/non-ground label with a sequential CPU replica of the same algorithm (box filter, polar cells, per-sector queue and slope fit, the four segmentation modes, recheck).
  - Fixed kernel, seeds 7 / 11 / 23: **0 label differences**, 0 missing, 20 runs identical to run 0, each. Label hashes (FNV-1a over per-input-point labels, input order; for the wave32 validator to match): seed 7 `04516cba11e18f95`, seed 11 `7cd4bb48f4483e7f`, seed 23 `957f6d786c501316`. Seed 7: 150,516 ground, 4,848 non-ground, 7,832 removed by the box filter (climbing sectors exceed `max_z`).
  - Naive wave64 variant (`make_naive.sh`: shuffles without width, ballot truncated to 32 bits, full 64-bit mask), same scenes: ~66,900 label differences each, **all in the 180 odd-numbered sectors** (the upper group of each wavefront), ground reference handed across sectors exactly as plan.md predicted. Seed 7 naive hash `53c7d5f9646c3315`. This shows the scene detects the fault.
  - The recheck "keep the last ground point" branch (`32 - __ffs(backup_res)`) is never reached with a positive `non_ground_height_threshold` (the lowest ground point can never exceed min + threshold); the CPU replica prints if it ever is, and it did not. Reviewer note: that expression is not "the last thread with a ground point" (`32 - __ffs` gives 31 when lane 0 voted); upstream, unreachable, left alone.
  - Not guarded (planner note confirmed by reading): `computeCellId` does not bound `point_cell_id`; the box filter does not apply `max_radius`, so a point beyond `max_num_cells_per_sector * cell_divider_size_m` that passes the box writes out of bounds. Not reproduced (the harness keeps radii within range); upstream behaviour, for the reviewer.
- **autoware_probabilistic_occupancy_grid_map:** upstream tests are commented out (#7724). Built out of tree from the fork's own test sources against the installed package (`projects/autoware_universe/harness/occupancy_grid_map/`, no fork edit): `colcon test --packages-select ogm_tests --ctest-args --repeat until-fail:20`: test_utils 1/1, costmap_unit_tests 2/2, fusion_policy_unit_tests 2/2, test_pointcloud_based_method.py (3 tests, GPU node) and test_synchronized_grid_map_fusion_node.py (2 tests), each 20 times, "100% tests passed, 0 tests failed out of 5". Geometry probe `test_ogm_geometry.py` (a wall of points 10.1 m ahead, 5 frames, pointcloud-based node on the GPU): cells 2-9 m ahead = 1 (free), wall cells = 99, cells 12-20 m behind the wall = 50 and cells to the side = 50 (unknown): pass. No CPU path is left for the pointcloud-based maps in this upstream version, so the plan's GPU-vs-CPU comparison is replaced by this by-construction check.

### Surface

All 23 stage-1 components `covered` with evidence; `feature:agnocast` moved to `scoped_out` citing ruling Q5 = A; `library:autoware_tensorrt_bevdet` stays `owed` with the reason updated to ruling Q2 = B; every stage 2-4 entry unchanged under `owed`.

### For the validators

- wave64 (this platform) and wave32 gates. Rebuild with the recipe above (libhipcxx needed). Run the preprocessor gtests 20x, the ground-segmentation harness (`./gs_fixed 20 <seed>` for seeds 7, 11, 23; the wave32 label hashes must equal the ones above), the out-of-tree occupancy-grid tests and the geometry probe, and the example launches for every node (the noise/outlier/downsample filters have no upstream test).

## Lesson for cuda-to-rocm: libhipcxx headers need a HIP compiler

For `references/libraries.md`. libcu++ (`cuda::std::optional`, `cuda::proclaim_return_type`, `<cuda/functional>`) maps to libhipcxx, which recent ROCm SDKs ship under `include/hipccl` (`find_package(libhipcxx)`, target `libhipcxx::libhipcxx`; it provides the `cuda/std/...` spellings). Unlike libcu++ under g++, its headers do not compile with a plain host compiler (`'__host__' does not name a type`), so a host `.cpp` that includes a header exposing `cuda::std` types must be compiled as HIP too, or the ABI of those types would differ between translation units. A `namespace cuda::std = ::std` shim is not a substitute in device code: libstdc++'s `std::optional::value()` calls a host-only throw helper that hip-clang rejects. Also include `<cuda/std/cmath>` / `<cuda/std/functional>` explicitly in a compat header when the sources rely on CCCL including them transitively (`cuda::std::abs(float)` is otherwise ambiguous).

## Lesson for cuda-to-rocm: CMAKE_COMPILE_WARNING_AS_ERROR outranks -Wno-error

For `references/build-cmake-link.md`. A project that sets `CMAKE_COMPILE_WARNING_AS_ERROR` (Autoware's `autoware_package()` does on CMake >= 3.24) gets `-Werror` appended after every target compile option, so `target_compile_options(t PRIVATE -Wno-error)` has no effect. When `.cu` sources move from nvcc (which never saw the flag through `CUDA_NVCC_FLAGS`) to the HIP language, upstream kernel warnings become errors; set the target property `COMPILE_WARNING_AS_ERROR OFF` on the HIP kernel target instead.

## Review 2026-11-23 (reviewer, linux-gfx942 fleet job; stage 1, fork `moat-port` at 9574fe1)

Scope: `git diff 6cea1ac...9574fe1` (5 commits, 16 files), pr-review skill, every claim below re-checked against the code and the ROCm 10.2 (HIP 7.17.26434) headers. Later-stage surface entries `owed` by design (rulings Q1-Q5), not reviewed.

Verdict: **pass, no findings.** What was checked, so a later round does not redo it:

- Ground segmentation warp change (`cuda_common.hpp:35-41`, `cuda_scan_ground_segmentation_filter.cu:273-283` and the 13 shuffle sites). A group is 32 consecutive threads (`sector_id = index / 32`, block 256 or `num_sectors*32` if smaller, so groups never straddle a wavefront and `sector_stride` is even): on wave64, groups 2k and 2k+1 share a wavefront at lane offsets 0 and 32. HIP `__shfl(v, src, w)` indexes `(src & (w-1)) + (self & ~(w-1))` and `__shfl_down` clamps at `(self & (w-1)) + delta >= w` (`amd_warp_functions.h:152-158, 400-402`), so width 32 keeps every shuffle inside the group; the source lanes are always active because each group is converged at these points (the same precondition upstream's FULL_MASK already relies on on CUDA). `__ballot_sync` returns `__ballot(p) & mask` (`amd_warp_sync_functions.h:187-194`), so with the group mask and the shift by `__lane_id() & ~31` the helper yields bit i for group thread i; `__ffs(backup_res)` and `== 0` tests then see exactly the CUDA 32-bit value. `__hip_check_mask` (debug only) waterfalls over distinct masks and asserts each equals the active lanes with that mask, which the per-group mask satisfies even when the two groups diverge (different cell counts, `continue` on empty cells) and which a 64-bit all-ones mask would not. wave32: `__lane_id()` < 32, offset 0, mask `0xFFFFFFFF` as 64-bit, ballot upper half zero: identical to the CUDA semantics. CUDA: the width argument is 32 = warpSize (the default), `FULL_MASK` and `warpBallot` take the unchanged `#else` path, so code generation is unchanged. `__syncwarp()` on HIP is a wavefront fence plus `wave_barrier` (no hardware barrier), so it cannot hang when the other group is elsewhere.
- Porter's synthetic test exercises the shared wavefront: `harness/ground_segmentation/harness.cpp` uses 361 one-degree sectors with terrain chosen by `sector % 4`, so sectors 2k and 2k+1 (the two groups of one wave64) always need different ground references; recheck is on (`recheck_start_distance` 20 m). The naive build (`make_naive.sh` strips the width and shift; CMake defines `FULL_MASK` all-ones) fails only in odd sectors, which is the upper-group signature. It links the installed library, not a copy.
- Force-included compat headers: in all three packages the `-include` sits in `add_compile_options("$<$<COMPILE_LANGUAGE:HIP>:...>")` inside `if(USE_HIP)`, and `hip/include` (with the `cub/*.cuh` stand-ins) is on the include path only in that branch; `USE_HIP` is set only when CUDA was not found or was skipped by `AUTOWARE_CUDA_USE_HIP`. `ament_auto_package` installs `include/` only, so `hip/` never reaches a consumer. A CUDA build cannot see them. The only installed-header change (`cuda_common.hpp`) is behind `__HIP_PLATFORM_AMD__`. Repeated `namespace cub = hipcub;` aliases are legal redeclarations.
- Occupancy grid `-fgpu-rdc`: required, `occupancy_grid_map_{fixed,projective}_kernel.cu` call non-inline `__device__` `raytrace`/`setCellValue` defined in `utils_kernel.cu:77,173`; mirrors upstream's `CUDA_SEPARABLE_COMPILATION ON`; the five `.cu` all get `-fgpu-rdc` and the device link happens inside `<pkg>_cuda` (the only target that launches kernels), with `--offload-arch` taken from `CMAKE_HIP_ARCHITECTURES` (always set after `enable_language(HIP)`).
- Relaxed `-Werror`: checked with CMake 3.31 that `CMAKE_COMPILE_WARNING_AS_ERROR` appends `-Werror` after target options (`CXX_FLAGS = -Wall -Wno-error -Werror`), and with AMD clang 24 that a discarded `[[nodiscard]]` enum is `-Wunused-value`. FindCUDA's `cuda_add_library` never passed `-Werror` to nvcc, so `COMPILE_WARNING_AS_ERROR OFF` on the HIP kernel target and `-Wno-error=unused-value` (HIP sources only; `cudaError_t` is not nodiscard on CUDA) restore CUDA parity and touch no CUDA flag.
- libhipcxx minimum (porter's question 4): acceptable. The HIP branch skips the package with the same warning as a CUDA-less host when libhipcxx is missing; TheRock ROCm 10.2 ships it (`include/hipccl`, `lib/cmake/libhipcxx`); g++ indeed rejects its headers (`__cccl/assert.h`), so compiling the node sources as HIP is needed. The wave32 validator needs an SDK with libhipcxx.
- Commits: titles <= 72, `[ROCm]` on the four ROCm commits and not on the shared `[[nodiscard]]` fix, AI disclosure, Test Plans, no trailer; `jargon.py --port`: clean.
- Lessons: both verified as stated (libhipcxx under g++ fails; `-Wno-error` loses to the appended `-Werror`). For the promoter: say that the specific form `-Wno-error=<group>` still works after the appended `-Werror` (this port relies on it), so the lesson is not read as "no per-warning relaxation is possible".

Not the port's to fix (upstream behaviour, unchanged on every backend, porter already noted): `computeCellId` unbounded cell id beyond `max_radius`; the unreachable `32 - __ffs(backup_res)` recheck branch.

Owed elsewhere: the CUDA compile check. This host has no nvcc; the real nvcc compile of the three packages runs on the mi355 fleet host before the port is brought to Jeff.

## CUDA no-regression compile check (2026-11-23, mi355 fleet job, real nvcc 12.8.93; no GPU used)

Requested by Jeff Daily; closes the "CUDA compile gate owed" item above. Compile and link only, nothing run: CUDA run-time behaviour is `cuda-not-validated`.

**Result: pass, no CUDA regression.** All three stage-1 packages configure on their CUDA branch (`CUDA_VERSION=12.8`, `AUTOWARE_CUDA_USE_HIP=OFF` at the head), compile and link with nvcc at both fork head `9574fe1` and upstream `6cea1ac`, rc 0 each: autoware_cuda_pointcloud_preprocessor (8 `.cu`, plus both gtest executables), autoware_ground_segmentation_cuda (1 `.cu`), autoware_probabilistic_occupancy_grid_map (5 `.cu`; `CUDA_SEPARABLE_COMPILATION` path). No configure message about HIP or skipping a package.

What nvcc and the host compiler see is the same at both commits. Every artifact below was diffed with the workspace and source paths normalized:
- nvcc invocations: the FindCUDA per-object scripts (`*.cu.o.Release.cmake`, 14 files, 4410 lines) are identical: same `CUDA_NVCC_FLAGS` (`--expt-relaxed-constexpr -diag-suppress 20012`, `--extended-lambda` on the preprocessor, `-gencode` sm_75/86/87/89/101/120 plus compute_120 PTX), same include dirs, defines and host flags (`-fno-lto -fPIC -O3 -DNDEBUG`). The occurrence count of `hip` in them is 0.
- Host `.cpp` compile commands (`compile_commands.json`, 33 files) and link lines (`link.txt`, 23 targets): identical. Object lists identical.
- Warnings: identical sets (117 compiler warnings at each commit, all upstream code: nvcc #940-D missing return in `cuda_polar_voxel_{noise,outlier}_filter.hpp`, #550-D `valid_count` unused, #68-D/#191-D in point_cloud_msg_wrapper, g++ `-Wreturn-type` on the same two headers; plus CMake dev warnings, whose only difference is CMakeLists line numbers). Neither `[[nodiscard]]` order draws an nvcc diagnostic.
- Device code: `cuobjdump -sass` (nvdisasm 12.8.90) and `cuobjdump -ptx` of all 14 `.cu.o`, after normalizing nvcc's path-derived `_INTERNAL_<hash>` / `_GLOBAL__N__<hash>` module names: **byte-identical** for every object, every arch (e.g. ground segmentation 90 SASS functions / 46,529 lines, polar voxel outlier filter 240 / 401,807).

So for the shared-source edits: the `[[nodiscard]]`-first order, `width = WARP_SIZE` on the 13 shuffles (the default width on a 32-thread warp), and `warpBallot` / `FULL_MASK` (the HIP code sits behind `__HIP_PLATFORM_AMD__`) leave the generated device code unchanged. Moving the source lists into variables and `if(NOT CUDA_FOUND)` leave the CUDA target sources, flags and link lines unchanged.

Dependencies, all built for CUDA inside the job directory: `autoware_cuda_utils` + `autoware_cuda_dependency_meta` from AMD-Ecosystem/autoware_cuda `992d7034b5396900593126e8f19a1982ae7bc682` (CUDA branch: `CCCL::CCCL`, `CUDA::cudart`), cuda_blackboard `9e0fb865cd2e38dcc0db5e0e55497f40fb71f733` (CUDA, static cudart), negotiated `eac198b5` (cuda_blackboard's `build_depends.repos` pin), autoware_point_types and autoware_agnocast_wrapper from autoware_core `15408eb6` (the released 1.9.0 debs are too old for universe main: no `PointXYZIRCT`; duplicate `FilterBase` instantiation), managed_transform_buffer `6c9a65b`, tier4_debug_msgs (tier4_autoware_msgs `efa4129`), autoware_pointcloud_preprocessor and autoware_pcl_extensions from upstream `6cea1ac`. These are the porter's commits where the porter recorded one.

Environment (job directory only, nothing on the host changed): ROS 2 Jazzy plus PCL/OpenCV/nav2_costmap_2d/grid_map/the released autoware debs, 689 debs fetched with a private `APT_CONFIG` (packages.ros.org noble snapshot 2026-11-05, Ubuntu noble) and unpacked with `dpkg-deb -x` into `<job>/root`, then relocated (see the moat-porting lesson below). colcon 0.21 driven by `/usr/bin/python3` (the ROS debs target Python 3.12), `$FLEET_VENV/bin` first on PATH, CMake 3.31.6, gcc 13.3, nvcc `/usr/local/cuda` 12.8.93. Build, per workspace (`ws_up` = `6cea1ac`, `ws_head` = `9574fe1`, three package symlinks each, `ws_deps` underlay):
```
colcon build --merge-install --base-paths src --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPython3_EXECUTABLE=/usr/bin/python3
```
About 2 min per workspace at `-j21`. Not checked: anything at run time, the lint tests (not run), and other CUDA versions or host compilers.

## Lesson for moat-porting: ROS 2 from debs on a fleet host with no root

The fleet container blocks mounts in user namespaces (`unshare -Urm` then `mount` gives "cannot mount read-only"), so overlaying debs onto `/usr` and `/opt` is out. What works: a private apt (`APT_CONFIG` with `Dir::State::Lists`/`Dir::Cache` in the job dir, `Dir::State::status "/var/lib/dpkg/status"` so only missing packages download, `Debug::NoLocking`, ROS key passed with `signed-by`), `apt-get install --download-only`, `dpkg-deb -x` each deb into `<job>/root`, then repair: (1) rewrite `/opt/ros/jazzy` to the job path in the text files; (2) rewrite `/usr/...` paths that exist only in the job tree; (3) in CMakePackageConfigHelpers configs, where the `_realOrig` path was rewritten, also rewrite the `set(PACKAGE_PREFIX_DIR "/usr")` that follows, and in Qt5 configs the `"/usr/lib/x86_64-linux-gnu/../../"` prefix; (4) point dangling `.so` dev symlinks at the host's runtime libraries; (5) symlink in host entries missing from the job tree's `lib/x86_64-linux-gnu`, `share` and `include` (dev packages such as Qt need their host runtime libraries beside them), **but not** the host's top-level `include/*.h` or the `c++`/`sys`/`linux`/`asm-generic` directories: FindCUDA turns every include dir into `-I`, so a mirrored `math.h` lands ahead of libstdc++'s wrapper and CCCL's `<cmath>` fails ("the global scope has no fpclassify"); do not set `CPATH` to it either; (6) delete any released package that the build replaces from source when its headers are unscoped (autoware_point_types), so the stale copy cannot shadow the new one. Also: colcon hangs (pipe write blocked, 0 CPU) with `CMAKE_VERBOSE_MAKEFILE=ON` on these packages, because one output line exceeds asyncio's 64 KiB reader limit. Take nvcc command lines from FindCUDA's `*.cu.o.<config>.cmake` scripts instead. The CUDA install on the mi355 host has no `nvdisasm`; `apt-get download cuda-nvdisasm-12-8` into the job dir and set `NVDISASM_PATH` for `cuobjdump -sass`.
