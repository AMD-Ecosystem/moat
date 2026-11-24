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
## Validation 2026-11-23 (validator, linux-gfx942 fleet job; AMD Instinct MI300X gfx942, HIP_VISIBLE_DEVICES=3 pinned by the dispatcher; fork `moat-port` at 9574fe1)

Verdict: **validation-failed** (plain, a GPU run failed). `cuda_voxel_grid_downsample_filter_node` aborts on its first point cloud. Everything else passes. Fixing it needs one shared line (below), verified here in a scratch copy. Correction to my review above: I accepted the porter's note that `cudaMemcpyToSymbolAsync` maps directly (it compiles), but a compile proves nothing about the copy kind. This was a runtime fault the review should have asked to see exercised.

### Failure for the porter

- `sensing/autoware_cuda_pointcloud_preprocessor/src/cuda_downsample_filter/cuda_voxel_grid_downsample_filter.cu:395-396`: `cudaMemcpyToSymbolAsync(voxel_info_dev, &voxel_info_, sizeof(VoxelInfo), 0, cudaMemcpyDefault, stream_)` returns `hipErrorInvalidMemcpyDirection (21)` on HIP 7.17.26434, so `CHECK_CUDA_ERROR` throws and the node aborts (`terminate called after throwing an instance of 'std::runtime_error' what(): hipErrorInvalidMemcpyDirection (21)@.../cuda_voxel_grid_downsample_filter.cu#L396`). This is the only `MemcpyToSymbol` call with `Default` in stage 1 (the other `__constant__`, `undistort_kernels.cu:30`, is never copied to).
- Minimal reproducer (`agent_space/sym_repro/sym.hip`, hipcc `--offload-arch=gfx942`): `hipMemcpyToSymbol` and `hipMemcpyToSymbolAsync` with `hipMemcpyDefault` both return `hipErrorInvalidMemcpyDirection`; with `hipMemcpyHostToDevice` both succeed and a kernel reads the value. CUDA documents `cudaMemcpyDefault` as accepted (and recommended) for `cudaMemcpyToSymbol[Async]`, so this is a HIP divergence: registered as `hip-memcpytosymbol-default-kind` (rocm-bug-report, not filed).
- Fix, verified: change the kind to `cudaMemcpyHostToDevice` (the source is the host member `voxel_info_`, so CUDA behaviour is identical; a shared one-line change, no `#ifdef`, and it should not be done by remapping `cudaMemcpyDefault` in the compat header). Built in a scratch copy outside the fork clone (`<job>/fixcheck`, `sed -i '396s/cudaMemcpyDefault/cudaMemcpyHostToDevice/'`), the node then downsamples correctly on the GPU (probe below: 4096/4096 voxels, max centroid error 1.7e-6). Nothing was pushed or committed to the fork; the fork clone is clean at 9574fe1.

### Environment (all inside the job directory)

- TheRock ROCm 10.2.0 into `$FLEET_VENV` (`$FLEET_VENV/bin/pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple 'rocm[devel,device-gfx942]==10.2.0'`, `rocm-sdk init`): HIP 7.17.26434, AMD clang 24.0.0git, libhipcxx, hipCUB, rocThrust. No system ROCm used.
- Dependency closure rebuilt from scratch per the porter's environment report (sources copied from the porter's job, its build products not reused): 17 system libraries into `<job>/deps/prefix` (430 s), ROS 2 Jazzy underlay 198 packages (228 s), overlay 74 packages (353 s), all rc 0. Deviations: Boost, OpenCV, GMP, MPFR, GraphicsMagick re-extracted from the tarballs (the copies had in-source build output); OpenCV also with apps/Java/VTK off; colcon `--parallel-workers 8`, `MAKEFLAGS=-j8` (64 = MAX_JOBS).
- GPU dependencies, fresh clones: AMD-Ecosystem/autoware_cuda at `992d7034b5396900593126e8f19a1982ae7bc682` (autoware_tensorrt_common COLCON_IGNOREd; `-- cuda is not found, building autoware_cuda_utils with HIP 7.17.26434`), AMD-Ecosystem/cuda_blackboard at `9e0fb865cd2e38dcc0db5e0e55497f40fb71f733` (`-- cuda was not found, building cuda_blackboard with HIP 7.17.26434`).
- Same fmt link workaround as the porter (`-DCMAKE_EXE_LINKER_FLAGS="-Wl,--no-as-needed -L<job>/deps/prefix/lib -lfmt -Wl,--as-needed"`), needed only for the environment.

### Build (rc 0, 1 min 39 s)

```
colcon --log-base log build --parallel-workers 3 --base-paths src --build-base build --install-base install \
  --packages-select autoware_cuda_pointcloud_preprocessor autoware_ground_segmentation_cuda autoware_probabilistic_occupancy_grid_map \
  --cmake-args -DCMAKE_BUILD_TYPE=Release "-DCMAKE_PREFIX_PATH=$ROCM_SDK;<job>/deps/prefix" -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DAUTOWARE_CUDA_USE_HIP=ON -DPython3_EXECUTABLE=$FLEET_VENV/bin/python -DBUILD_TESTING=ON "<fmt workaround>"
```
All three configure with `cuda was not found, building <pkg> with HIP 7.17.26434`.

### Tests (all under `ulimit -c 0` and `timeout`)

- **Preprocessor gtests**, `colcon test --packages-select autoware_cuda_pointcloud_preprocessor --ctest-args -R test_cuda --repeat until-fail:20`: test_cuda_combine_cloud_handler 8/8 and test_cuda_pointcloud_preprocessor 20/20, 20 times each (20 + 20 `[  PASSED  ]` blocks, 0 failed, 0 skipped). `AMD_LOG_LEVEL=3`: `Gfx Major/Minor/Stepping: 9/4/2, Device ID: 0x74a1`, 1070 `hipLaunchKernel`.
- **Preprocessor filter nodes without upstream tests** (probes in `agent_space/`, published as plain `PointCloud2` into the nodes' compatible subscriptions):
  - polar voxel outlier and noise filters, GPU node vs upstream's CPU node (`autoware_pointcloud_preprocessor` `polar_voxel_{outlier,noise}_filter_node`, same param file) on an 83,000-point PointXYZIRC scene (4 dense rings, 3,000 isolated points, mixed return types and intensities), seeds 3/11/29: noise filter identical point sets on all three (66,373 / 66,442 / 66,259 kept); outlier filter identical on seeds 3 and 11 (47,064 / 46,925), seed 29 same count 47,007 with one point swapped. Traced: point 60977 lies 3.8e-6 voxel units below the 60 m radial edge (r/0.5 = 119.9999962); the GPU's float (`sqrtf`, FMA-contracted) keeps it in voxel 119 with point 60747, the CPU node's rounding puts it in 120, leaving 60747 alone (count 1 < threshold 2). Re-sending just that pair reproduces it regardless of return type. Float-versus-double edge rounding is upstream's design (the CUDA kernel has the same float path and nvcc contracts FMAs by default), not a ROCm fault. Deterministic over 3 reruns.
  - voxel grid downsample: aborts at 9574fe1 (failure above). With the one-line fix: a 32x16x8 grid of 0.3x0.3x0.1 voxels, 6 points each, shuffled (24,576 points), gives 4096 outputs, one per voxel, max centroid error 1.7e-6 vs a CPU mean: pass. On a 40x25x8 grid (8000 voxels) it returns 27,548 points: an **upstream bug, not the port's**: `searchValidVoxel` passes `most_significant_bit_index = floor(log2(num_voxels))` as `end_bit` to `DeviceRadixSort::SortPairs`, where end_bit is exclusive (CUB and hipCUB alike), so the key's top bit is not sorted and equal voxels split into several runs. A CPU model of a stable sort on the low 12 bits predicts exactly 27,548 runs. Any grid whose voxel count is not a power of two is affected, on CUDA too. Registered as `cuda-voxel-downsample-radix-end-bit` (other; reporting it upstream is a person's call).
- **Ground segmentation reference comparison** (`harness/ground_segmentation`, built against the installed library): `./gs_fixed 20 <seed>`: seeds 7 / 11 / 23 each 0 label differences, 0 missing, 20 runs identical; label hashes `04516cba11e18f95`, `7cd4bb48f4483e7f`, `957f6d786c501316`, identical to the porter's. Naive variant (`gs_naive 3 7`): 66,910 mismatches, all in the 180 odd sectors, hash `53c7d5f9646c3315` (identical to the porter's). Extra check of the mask claim: the fork's kernel source compiled with HIP's debug mask check on (`-UNDEBUG -O1`, which enables `__hip_check_mask`): pass, same hash `04516cba11e18f95`; the same build with `FULL_MASK` forced to `0xFFFFFFFFFFFFFFFF`: `HSA_STATUS_ERROR_EXCEPTION` in `sectorProcessingKernel`, `hipErrorLaunchFailure (719)`. So the group mask is required for debug builds on wave64, as the porter said.
- **Occupancy grid** (upstream tests disabled, #7724; built out of tree from the fork's test sources, `harness/occupancy_grid_map`): `colcon test --packages-select ogm_tests --ctest-args --repeat until-fail:20`: test_utils, costmap_unit_tests, fusion_policy_unit_tests, test_pointcloud_based_method, test_synchronized_grid_map_fusion_node and the geometry probe test_ogm_geometry, 20 times each, "100% tests passed, 0 tests failed out of 6". `AMD_LOG_LEVEL=3` shows the `map_fixed::{prepareTensor,fillEmptySpace,fillObstacles,fillUnknownSpace}Kernel`, `applyBBFKernel` and `copyMapRegionKernel` launches on the MI300X (fillEmptySpace calls `raytrace` across the `-fgpu-rdc` link). The projective map, which no test covers, was run once with a scratch copy of the probe (`grid_map_type: OccupancyGridMapProjectiveBlindSpot`, `scan_origin_frame` 1.8 m above base_link): all five `map_projective::*` kernels run; wall cells 99, behind and side 50, free cells only up to 5.5 m. That matches a hand computation of upstream's formula (`obstacle_z = 0.5 - 1.8`, `dz = 3.1`, ratio -0.419, projected point at 5.86 m, on the sensor side), so the GPU result is upstream's algorithm. That the projection lands sensor-side looks like an upstream quirk. Not registered: it needs a person with the algorithm's intent.
- **Non-GPU tests**: full `colcon test` of the three packages: copyright, cpplint, lint_cmake pass; `cppcheck` fails (binary not installed in the environment) and `xmllint` does not run, the same environment gaps the porter saw; ground segmentation registers no tests.

### Gates

- CUDA compile check: not run on this host (no nvcc; Jeff's instruction puts it elsewhere). It was recorded at 9574fe1 by the nvcc 12.8 job (`## CUDA no-regression compile check` section, commit efd4791b, merged while this validation ran) and is owed again at the head that carries the downsample fix.
- `jargon.py --port autoware_universe`: clean. README ROCm sections present in all three packages. Fork clone clean at 9574fe1 (`git status --porcelain` empty).

## Lesson for cuda-to-rocm: hipMemcpyToSymbol rejects hipMemcpyDefault

For `references/faults-memory-streams.md`. CUDA documents `cudaMemcpyDefault` as accepted (and recommended) for `cudaMemcpyToSymbol` and `cudaMemcpyToSymbolAsync`. HIP 7.17 (ROCm 10.2) returns `hipErrorInvalidMemcpyDirection` for `hipMemcpyDefault` on all three of `hipMemcpyToSymbol`, `hipMemcpyToSymbolAsync` and `hipMemcpyFromSymbol`. The explicit direction works (tested: `hipMemcpyHostToDevice` and `hipMemcpyDeviceToDevice` to a symbol, `hipMemcpyDeviceToHost` from one). The call compiles, so a port that only builds never sees it, and an error-check macro turns it into an abort at the first use. Grep the CUDA sources for `MemcpyToSymbol` with `Default` and change the kind at the call site to the real direction, which is behaviour-preserving on CUDA. Do not remap `cudaMemcpyDefault` in a compat header: other copies rely on it.

## Porter round 2 (2026-11-24, porter agent, linux-gfx942 fleet job; AMD Instinct MI300X, HIP_VISIBLE_DEVICES=3 pinned by the dispatcher)

Requested by Jeff Daily: fix the validator's downsample failure, then search all three stage-1 packages for every other call whose correctness is only checked at run time and run each node end to end on the GPU. Stage 1 only; every stage 2-4 surface entry stays `owed`. Fork `moat-port` = `9574fe1` + 3 commits, head **`fb4072979a73b23cf225e2bb3ecc52300d0be3b4`**, pushed. **This host has no nvcc: the CUDA compile check is owed again at fb40729** (mi355 host). By reading, the CUDA path changes only in `cudaMemcpyDefault` -> `cudaMemcpyHostToDevice` at one call site; the CMake edits are inside `if(USE_HIP)`.

### What changed on the fork

- `bb4cd9a` `fix(autoware_cuda_pointcloud_preprocessor): name the symbol copy kind` -- shared, no prefix: `cuda_voxel_grid_downsample_filter.cu:396` `cudaMemcpyDefault` -> `cudaMemcpyHostToDevice` (the source is always the host member `voxel_info_`, so CUDA does the same copy). Not done by remapping `cudaMemcpyDefault` in the compat header, as the validator advised.
- `41440f4` `[ROCm] Compile the concatenate node with the host compiler` -- **a second run-time fault, found by running the node**: `cuda_concatenate_and_time_sync_node` died on its first frame with `symbol lookup error: libcuda_pointcloud_preprocessor.so: undefined symbol: _ZN24managed_transform_buffer22ManagedTransformBuffer12getTransformIN13geometry_msgs3msg17TransformStamped_ISaIvEEEEENSt9enable_ifIXsr3stdE9is_same_vIT_S6_EESt8optionalIS6_EE4typeE...`. The node source instantiates `concatenate_and_time_sync_node.ipp` (autoware_pointcloud_preprocessor), which calls `managed_tf_buffer_->getTransform<geometry_msgs::msg::TransformStamped>(...)`, whose return type is `std::enable_if_t<std::is_same_v<T, ...>, ...>`. clang mangles the qualified variable template as `Xsr3stdE9is_same_v...`, GCC 13 as `X9is_same_v...` (raw `nm -D` of both above), so the HIP-compiled reference never binds to the g++-built managed_transform_buffer. It links (shared libraries may have undefined symbols) and fails only when the call is first bound. The concatenate node source uses no libcu++/Thrust types, so on the HIP branch it now compiles with the host compiler (`list(REMOVE_ITEM ...)` from the HIP source list), as it does under CUDA; it gets `__HIP_PLATFORM_AMD__` and the `cuda_runtime.h` shim through `autoware_cuda_utils`/`hip::host`, and cuda_blackboard's headers already switch on that macro, so no force-include is needed for it.
- `fb40729` `[ROCm] Say the compat header reaches only HIP sources` -- comment fix for the two comments that said every source.
- Not committed to the fork (MOAT harness): `harness/ground_segmentation/harness.cpp` gains `gs_fixed dump <seed> <file>` (scene + CPU reference labels) for the node probe.

### Run-time-only audit of the three packages

Every CUDA runtime call that compiles whatever its arguments are was listed (`grep -rnoE 'cuda[A-Z]\w*'` over the three packages) and each pattern was exercised on the GPU exactly as the packages call it, in a HIP probe (`agent_space/rt_probe/rt.hip`, hipcc `--offload-arch=gfx942 -O2`):

| Pattern (sites) | HIP 7.17.26434 result |
| --- | --- |
| `MemcpyToSymbol[Async]` with `Default` (1 site, the fixed one) | `hipErrorInvalidMemcpyDirection`, host and device source |
| `MemcpyToSymbolAsync` `HostToDevice` / `DeviceToDevice`; `MemcpyFromSymbol` `DeviceToHost` | success; `MemcpyFromSymbol` `Default` also fails (no site uses it) |
| `__constant__ double time_diff_threshold = 0.1` (undistort, never copied to) | kernel reads 0.1 |
| `cudaMemcpy[Async]` with `Default` host<->device (none in the packages; checked anyway) | success |
| `cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)` (preprocessor node), before and after a stream exists | success both |
| `cudaMemPoolCreate` with `allocType Pinned`, `location {Device, cudaGetDevice()}`; `SetAttribute(ReleaseThreshold)`; `MallocFromPoolAsync`/`FreeAsync` incl. size 0 (downsample, polar filters, ground segmentation) | success; size 0 gives nullptr, freeing it succeeds |
| `cudaMallocAsync`/`FreeAsync` on the default pool, incl. size 0 (ground segmentation `device_vector`) | success; `MemoryPoolsSupported` = 1 |
| `cudaStreamCreate`, `cudaStreamCreateWithFlags(NonBlocking)` (OGM node, return unchecked), `cudaDeviceGetAttribute(MultiProcessorCount)` | success (304 CUs) |
| zero-size grid launch (concat kernel guards it; others launch only with work) | `hipErrorInvalidConfiguration`, as CUDA |

No events, graphs, managed memory, texture objects, `cudaFuncSetAttribute` or host registration in the three packages. Launch configurations: 256-thread blocks everywhere; the only dynamic shared memory is ground segmentation's `8 * (sizeof(Cell)+4) * gnd_cell_buffer_size` (a few hundred bytes). `ldd -r` over every stage-1 library and executable (run-time symbol resolution, the class of the second fault): after the fix 0 undefined symbols, except 4 cuda_blackboard symbols in `libcuda_ground_segmentation_lib.so`, which upstream does not link (its CUDA link line is the same, per the nvcc check) and which every consumer loads first; left alone.

### Minimal reproducer for the HIP symbol-copy behaviour (deferral `hip-memcpytosymbol-default-kind`)

HIP 7.17.26434 (TheRock ROCm 10.2.0 wheels, `rocm[devel,device-gfx942]==10.2.0`), AMD clang 24.0.0git (ROCm llvm-project bdc35841c97f), MI300X gfx942, amdgpu 6.16.13, Linux 6.8.0-134. The validator's reproducer:

```
#include <hip/hip_runtime.h>
#include <cstdio>
struct S { int a[42]; };
__constant__ S s_dev;
__global__ void k(int* o) { *o = s_dev.a[41]; }
int main() {
  S h{}; h.a[41] = 7; hipStream_t st; hipStreamCreate(&st);
  const hipMemcpyKind kinds[] = {hipMemcpyDefault, hipMemcpyHostToDevice};
  const char* names[] = {"hipMemcpyDefault", "hipMemcpyHostToDevice"};
  for (int i = 0; i < 2; ++i) {
    hipError_t e1 = hipMemcpyToSymbolAsync(HIP_SYMBOL(s_dev), &h, sizeof(S), 0, kinds[i], st);
    hipError_t e2 = hipMemcpyToSymbol(HIP_SYMBOL(s_dev), &h, sizeof(S), 0, kinds[i]);
    printf("%s: async=%s sync=%s\n", names[i], hipGetErrorName(e1), hipGetErrorName(e2));
    (void)hipGetLastError();
  }
  int* o; hipMalloc(&o, 4); k<<<1,1,0,st>>>(o); int r = 0; hipMemcpy(&r, o, 4, hipMemcpyDeviceToHost);
  printf("kernel read %d\n", r);
}
```
`hipcc --offload-arch=gfx942 sym.hip -o sym && ./sym` prints `hipMemcpyDefault: async=hipErrorInvalidMemcpyDirection sync=hipErrorInvalidMemcpyDirection`, `hipMemcpyHostToDevice: async=hipSuccess sync=hipSuccess`, `kernel read 7`. This round's wider probe adds: a device source with `hipMemcpyDefault` -> same error; `hipMemcpyFromSymbol(..., hipMemcpyDefault)` -> same error; plain `hipMemcpy[Async]` with `hipMemcpyDefault` -> success. CUDA documents `cudaMemcpyDefault` as accepted for `cudaMemcpyToSymbol[Async]` and `cudaMemcpyFromSymbol`. Filing is Jeff's decision.

### Environment (job directory only; nothing shared modified)

TheRock ROCm 10.2.0 into `$FLEET_VENV` (same pip command as the earlier rounds; HIP 7.17.26434), plus colcon and the ROS Python build tools into the same venv. Dependency closure rebuilt from the validator's source trees (copied; boost, OpenCV, GMP, MPFR, GraphicsMagick re-extracted from the tarballs): system libraries 435 s, ROS 2 Jazzy underlay 232 s, overlay 74 packages 366 s, all rc 0, `MAX_JOBS=48` respected (6 workers x `-j8`). `autoware_cuda_utils` from AMD-Ecosystem/autoware_cuda at `992d7034b5396900593126e8f19a1982ae7bc682` (`-- cuda is not found, building autoware_cuda_utils with HIP 7.17.26434`; autoware_tensorrt_common COLCON_IGNOREd), cuda_blackboard at `9e0fb865cd2e38dcc0db5e0e55497f40fb71f733` (`-- cuda was not found, building cuda_blackboard with HIP 7.17.26434`). Same fmt link workaround as before (environment only).

### Build and tests at fb40729 (all under `ulimit -c 0` and `timeout`)

- Clean build of the three packages (recipe as in the Porter stage 1 section): rc 0, 1 min 39 s, each `cuda was not found, building <pkg> with HIP 7.17.26434`.
- Preprocessor gtests 20x: test_cuda_combine_cloud_handler 8/8 and test_cuda_pointcloud_preprocessor 20/20, 20 `[  PASSED  ]` blocks each, 0 skipped.
- Occupancy grid out-of-tree tests 20x (`harness/occupancy_grid_map`): test_utils, costmap_unit_tests, fusion_policy_unit_tests, test_pointcloud_based_method (GPU node), test_synchronized_grid_map_fusion_node, test_ogm_geometry: "100% tests passed, 0 tests failed out of 6".
- Lint: copyright, cpplint, lint_cmake pass on all three; cppcheck (binary missing) and xmllint (not run) are the environment gaps of the earlier rounds.

### Each node run end to end on the GPU (probes in the job's `agent_space/`, plain `PointCloud2` in and out)

- `cuda_voxel_grid_downsample_filter_node`: 32x16x8 grid of 0.3x0.3x0.1 voxels, 6 points each, shuffled (24,576 points) -> 4096 outputs, one per voxel, max centroid error 1.74e-6: pass. The 40x25x8 grid reproduces the upstream radix `end_bit` bug exactly (27,548 outputs for 8000 voxels), as the validator traced; registered `cuda-voxel-downsample-radix-end-bit`, not the port's.
- `cuda_pointcloud_preprocessor_node`: static TF base_link->lidar (1.0, 0.5, 2.0, yaw 30 deg), twist at 200 Hz, 32 rings x 1800 azimuth steps over a 100 ms sweep (PointXYZIRCAEDT, rings shuffled, time-ordered within a ring) with 15 spikes and 3 zero-distance returns per ring, negative crop box around the vehicle. Compared with a float32 numpy model of the kernels (stable sort by ring, transform, crop box, 2D undistortion, ring outlier filter): vx = 0: 54,831 of 57,600 points (2,111 cropped, 801 ring outliers), vx = 10 m/s: 54,827 (793 ring outliers), max error 3.81e-6 m both: pass. `cudaSetDeviceFlags` raised nothing.
- `cuda_concatenate_and_time_sync_node` vs upstream's CPU `concatenate_and_time_sync_node`, same parameter file, both subscribed to the same two XYZIRC lidars (8192 points each, every point carries a unique id in intensity/channel, static TFs at +-40 deg yaw): at 9574fe1 + the downsample fix it died (symbol lookup error above); after `41440f4`: stationary, both equal the numpy transform within 3.8e-6 m; with vx = 5 m/s and `lidar_r` stamped 40 ms later, both shift `lidar_r` by +0.2000 m and leave `lidar_l` in place, GPU vs CPU max 3.81e-6 m, ids one to one, return types equal: pass.
- `cuda_polar_voxel_{outlier,noise}_filter_node` vs the CPU nodes (validator's probe, 83,000 points, seeds 3/11/29): noise filter identical sets on all three; outlier filter identical on 3 and 11, seed 29 the same one-point float-versus-double voxel-edge swap the validator traced (upstream design).
- `cuda_scan_ground_segmentation_filter_node` (shipped param file, which equals the harness parameters): harness scenes for seeds 7/11/23 (163,196 points) published to the node; its `output/pointcloud` and `output/ground_pointcloud` equal the CPU reference's non-ground and ground sets exactly (seed 7: 150,516 ground, 4,848 non-ground), no duplicates.
- Occupancy grid pointcloud-based node: covered by test_pointcloud_based_method and test_ogm_geometry (above).
- One component container (`launch_ros` ComposableNodeContainer) with the preprocessor and ground segmentation nodes: both pass their probes (ground segmentation twice, seeds 7 and 11). Loaded in the other order the preprocessor component does not load ("Failed to find class with the requested plugin name"): `libcuda_ground_segmentation.so` has a NEEDED entry for `libcuda_pointcloud_preprocessor.so` (upstream `ament_auto_add_library` links every build dependency's libraries, the same on CUDA), so class_loader never sees its factories. Registered `autoware-gs-component-preloads-preprocessor-components` (upstream, a person's call).

### For the reviewer and validators

- Review `9574fe1..fb40729`. The validator re-runs everything above at fb40729 plus the ground-segmentation reference comparison (`./gs_fixed 20 <seed>`, hashes in the Porter stage 1 section).
- The wave32 validator needs the same: the node probes are what found the second fault, and unit tests do not load the concatenate component.

## Lesson for cuda-to-rocm: host sources compiled as HIP can miss symbols in g++-built libraries

For `references/build-cmake-link.md`, beside its undefined-symbol and load-time failures. When a port compiles ordinary host `.cpp` files as HIP (clang) and links them against libraries built with GCC, a few template signatures mangle differently: a return type of `std::enable_if_t<std::is_same_v<T, X>, ...>` is `Xsr3stdE9is_same_v...` from clang and `X9is_same_v...` from GCC 13. The shared library still links (undefined symbols are allowed in `.so` files). Under the default lazy binding the process dies with "symbol lookup error" only when the call is first bound, so unit tests that never reach that path pass; with `-Wl,-z,now` or `LD_BIND_NOW=1` it fails at load instead. Run `ldd -r` on every built library and executable (it resolves all symbols eagerly and lists the missing ones). It also lists symbols a plugin library legitimately leaves to the process that loads it (this port has four such in a library its consumers always load after their provider), so look at each hit rather than fixing them all. Compile as HIP only the sources that need a HIP compiler; leave sources that only call into host libraries on the host compiler, as the CUDA build does. (autoware_universe, ROCm 10.2, GCC 13.3.)

## Review 2026-11-24 (reviewer, linux-gfx942 fleet job; delta `9574fe1...fb40729`, 3 commits)

Same job as Porter round 2, so the review was done by two independent read-only child checks (an ABI/build check and a correctness/records check), with every finding re-verified by hand against the fork. **Verdict: Request Changes** (four small items; nothing affects run-time behaviour or the CUDA path). Validation was not started: Jeff's instruction for this job is to record requested changes and stop.

### Port correctness / minimal footprint
1. `sensing/autoware_cuda_pointcloud_preprocessor/hip/include/cuda_pointcloud_preprocessor_hip_compat.hpp:53-55`: `#define cudaMemcpyDefault hipMemcpyDefault` is now unused (grep over all `.cu/.cpp/.hpp/.cuh/.h` of the three packages finds only the define; cuda_blackboard does not use it either). bb4cd9a left it orphaned, and the header describes itself as mapping the names used here. Remove the three lines in a new `[ROCm]` commit. (Removing it is not the remap the validator warned against.)

### Documentation
2. `sensing/autoware_cuda_pointcloud_preprocessor/README.md:22` says "The sources are unchanged". The branch edits `cuda_voxel_grid_downsample_filter.cu` (bb4cd9a) and `cuda_polar_voxel_outlier_filter.cu` (5fb1c77), both shared CUDA-valid fixes. Reword, e.g. "The sources need no HIP-specific code: ...".

### Lessons (promoted to a published skill, so wording counts)
3. `## Lesson for cuda-to-rocm: hipMemcpyToSymbol rejects hipMemcpyDefault`: "accepts only an explicit `hipMemcpyHostToDevice` / `hipMemcpyDeviceToDevice` (`hipMemcpyDeviceToHost` from a symbol)" implies other FromSymbol kinds are rejected; only Default and DeviceToHost were tested for `hipMemcpyFromSymbol`. Say: Default is rejected on all three; the explicit direction works (tested: HtoD and DtoD to a symbol, DtoH from one).
4. `## Lesson for cuda-to-rocm: host sources compiled as HIP can miss symbols in g++-built libraries`: name one home, `references/build-cmake-link.md` (it already covers undefined-symbol and load-time failures); say the failure is at first call only under lazy binding (`-Wl,-z,now` / `LD_BIND_NOW` makes it a load failure); and say `ldd -r` also lists symbols a plugin library legitimately leaves to its loader (this port has four such in `libcuda_ground_segmentation_lib.so`), so each hit needs a look rather than a blanket fix.

Checked and not a finding (for the next round, so it is not redone): `cudaMemcpyHostToDevice` is the real direction (`voxel_info_` is a host member, `cuda_voxel_grid_downsample_filter.hpp:122`, not rewritten before the stream syncs at `.cu:319`/`.cu:461`); it is the only symbol copy in stage 1. The host-compiled concatenate node and the HIP translation units agree on every shared type: a probe compiled with the real CXX and HIP flags gives identical `sizeof`/`alignof`/member offsets for `CombineCloudHandler<CudaPointCloud2>` (464/8), `CudaConcatStruct`, `CudaPointCloud2` (144/8), `CudaUniquePtr`, the synchronizer component (1488); the only compiler-dependent conditional in its headers is `__HIP_PLATFORM_AMD__`, defined in both (from `hip::host` via cuda_blackboard's hip extras and `autoware_cuda_utils::rocthrust_system`, i.e. declared dependencies, not incidental); the node object's 5 references into `libcuda_pointcloud_preprocessor_lib.so` match exactly. `ldd -r`: no undefined symbol in any library or executable of the preprocessor or occupancy-grid packages; no other `managed_transform_buffer` user in stage 1. Commit titles 69/58/53 chars, prefixes right (the shared fix names no HIP/ROCm target), disclosure, fenced Test Plans, no trailer, ASCII, jargon clean. The Test Plans omit `source install/setup.bash`, as the reviewed 5fb1c77 does; not raised.

## Porter round 3 (2026-11-24, porter agent, linux-gfx942 fleet job; AMD Instinct MI300X, HIP_VISIBLE_DEVICES=3 pinned by the dispatcher)

Requested by Jeff Daily: address the four items of `## Review 2026-11-24` and nothing else. Stage 1 only; every stage 2-4 surface entry stays `owed` (surface.json untouched). Fork `moat-port` = `fb40729` + 2 commits, head **`4f3358ee0ccb8b0e6e6f3fd3ab8651a7399b6783`**, pushed (plain push). **This host has no nvcc**, so no CUDA compile was run here; neither commit touches anything a CUDA build compiles (a define inside the HIP-only force-included header, and the README).

- Item 1: `f45a1ac` `[ROCm] Drop the unused cudaMemcpyDefault name from the compat header` -- removes the three lines from `hip/include/cuda_pointcloud_preprocessor_hip_compat.hpp`. `grep -rn cudaMemcpyDefault` over the three packages (all `.cu/.cpp/.hpp/.cuh/.h`) now finds nothing.
- Item 2: `4f3358e` `[ROCm] Correct the README note on source changes for ROCm builds` -- `README.md:22` now reads "The sources need no HIP-specific code: ...".
- Item 3: lesson `hipMemcpyToSymbol rejects hipMemcpyDefault` reworded: Default rejected on all three calls; explicit direction works, naming exactly what was tested (HtoD and DtoD to a symbol, DtoH from one; round 2's probe table).
- Item 4: lesson `host sources compiled as HIP can miss symbols ...` now names one home (`references/build-cmake-link.md`), says the failure is at first call only under lazy binding (`-Wl,-z,now` / `LD_BIND_NOW` makes it a load failure), and says `ldd -r` also lists symbols a plugin library leaves to its loader, so each hit needs a look.

Environment: rebuilt inside this job directory from the round-2 job's sources (scripts copied with paths repointed): TheRock ROCm 10.2.0 wheels into `$FLEET_VENV` (`rocm[devel,device-gfx942]==10.2.0`, HIP 7.17.26434) plus colcon and the ROS Python tools; system libraries 462 s, ROS 2 Jazzy underlay and the 74-package overlay, all rc 0 (1175 s total). `autoware_cuda_utils` from AMD-Ecosystem/autoware_cuda at `992d7034b5396900593126e8f19a1982ae7bc682` (`-- cuda is not found, building autoware_cuda_utils with HIP 7.17.26434`), cuda_blackboard at `9e0fb865cd2e38dcc0db5e0e55497f40fb71f733` (`-- cuda was not found, building cuda_blackboard with HIP 7.17.26434`), both cloned and built in the job's `aw_ws`. `MAX_JOBS=48` respected (6 workers x `-j8`, then 3 x `-j16`). Same fmt link workaround as earlier rounds (environment only).

Build at 4f3358e (recipe of the Porter stage 1 section): rc 0, 1 min 39 s, all three `cuda was not found, building <pkg> with HIP 7.17.26434`. Pre-push check: `colcon test --packages-select autoware_cuda_pointcloud_preprocessor --ctest-args -R test_cuda` under `ulimit -c 0` and `timeout`: test_cuda_combine_cloud_handler 8/8, test_cuda_pointcloud_preprocessor 20/20. `jargon.py --port autoware_universe`: clean. Fork clone clean.

## Review 2026-11-24, round 3 delta (reviewer, linux-gfx942 fleet job; delta `fb40729...4f3358e`, 2 commits, plus the two corrected lessons)

pr-review skill, local-branch mode, run inline in the same job as Porter round 3 (no child agents); every claim below re-checked by hand against the fork and, where cheap, by probe on the MI300X. **Verdict: Request Changes** (one item, same class as item 1 of the previous review; nothing affects run-time behaviour or the CUDA path). Per Jeff's instruction for this job, validation was not started.

### Port correctness / minimal footprint
1. `sensing/autoware_cuda_pointcloud_preprocessor/hip/include/cuda_pointcloud_preprocessor_hip_compat.hpp:41-43`: `#define cudaErrorInvalidValue hipErrorInvalidValue` maps a name nothing in this package compiles. `grep -rlw cudaErrorInvalidValue` over the package's `.cu/.cpp/.hpp/.cuh/.h/.ipp` (minus the header itself), cuda_blackboard's and autoware_cuda_utils' include trees and the job's overlay `install/*/include` finds no user; its only users are `autoware_ground_segmentation_cuda/include/.../cuda_utility.cuh:57,113` (served by that package's own compat header) and stage-2 `autoware_lidar_frnet`. Orphaned since `175821e` (likely carried over from the ground-segmentation header); earlier reviews missed it. It matters now because `f45a1ac`'s message says the header "maps only the names the package uses", which is false while this line stays. Remove the three lines in a new `[ROCm]` commit (no amend). The same check over the ground-segmentation (30 defines) and occupancy-grid (9 defines) compat headers finds no other unused name, so this is the last one.

Checked and not a finding (so the next round does not redo it): the removed `cudaMemcpyDefault` has no user in any compiled source or included header (grep as above, and the package builds and its gtests pass without it at 4f3358e). README line 22 "need no HIP-specific code" is true: the package's only source edits on the branch are the two shared CUDA-valid fixes (`cuda_voxel_grid_downsample_filter.cu`, `cuda_polar_voxel_outlier_filter.cu`), and `src/`, `include/`, `test/` contain no `__HIP`/`hip*` spelling. Both commit titles <= 72 (68, 64), `[ROCm]` right on each (HIP-only header; README ROCm section), AI disclosure, fenced Test Plans, no trailer, ASCII, `jargon.py --port`: clean. Lesson `hipMemcpyToSymbol rejects hipMemcpyDefault`: probe (`agent_space/rev_probe/sym.hip`, HIP 7.17.26434, gfx942) gives `hipErrorInvalidMemcpyDirection` for Default on `hipMemcpyToSymbol`, `hipMemcpyToSymbolAsync`, `hipMemcpyFromSymbol` and also `hipMemcpyFromSymbolAsync`; `HostToDevice` and `DeviceToDevice` to a symbol succeed sync and async, `DeviceToHost` from one sync and async, kernel reads the copied value; the lesson as written is accurate (for the promoter: the Async FromSymbol variant fails the same way). Lesson `host sources compiled as HIP ...`: probe with a library calling an undefined function: lazy binding starts and dies with "symbol lookup error" only at the call (rc 127); `LD_BIND_NOW=1` and a library linked `-Wl,-z,now` die at load; `ldd -r` lists it; `ldd -r libcuda_ground_segmentation_lib.so` at 4f3358e lists exactly the 4 cuda_blackboard symbols the lesson alludes to. Both lessons now name one reference file each.

## Porter round 4 (2026-11-24, porter agent, linux-gfx942 fleet job; AMD Instinct MI300X, HIP_VISIBLE_DEVICES=3 pinned by the dispatcher)

Requested by Jeff Daily: address the one item of `## Review 2026-11-24, round 3 delta`. Because two reviews in a row found an unused compat-header name, also check every name in all three packages' compat headers against real uses in that package and remove every unused one in the same commit. Nothing else changes. Stage 1 only; every stage 2-4 surface entry stays `owed` (surface.json untouched). Fork `moat-port` = `4f3358e` + 1 commit, head **`129d61b2dc9d63a7cab5d4b1e4bebf5f2c92030e`**, pushed (plain push). **This host has no nvcc**, so no CUDA compile was run here. The commit changes only the two HIP-only force-included headers, which a CUDA build never sees.

### What changed on the fork

- `129d61b` `[ROCm] Drop runtime names the packages never use from compat headers`, 21 lines deleted:
  - `cuda_pointcloud_preprocessor_hip_compat.hpp`: removes `cudaErrorInvalidValue` (the review's finding), plus four more unused names the review missed: `cudaError_t`, `cudaSuccess`, `cudaGetErrorName`, `cudaGetErrorString`.
  - `cuda_ground_segmentation_hip_compat.hpp`: removes `cudaGetErrorName` and `cudaGetErrorString`.

  The occupancy-grid header has no unused name.

### Why the round-3 review undercounted

It counted a name as used when it appeared anywhere the package's code could reach, including the headers of its dependencies. This round's criterion is stricter: a name is used only if some source in the package itself spells it (any `.cu/.cpp/.hpp/.cuh/.h/.ipp` outside `hip/`, macros included). The seven removed names appear only in `autoware_cuda_utils`' `cuda_check_error.hpp` (`::cudaError_t`, `::cudaSuccess`, `::cudaGetErrorName`, `::cudaGetErrorString`) and cuda_blackboard's `cuda_error.hpp`. On HIP, each of those headers maps the names itself before using them: `cuda_check_error.hpp` includes `<cuda_runtime_api.h>`, which resolves to autoware_cuda_utils' `hip/include` shim (`cuda_runtime_hip_compat.h`), and `cuda_error.hpp` includes `cuda_blackboard/cuda_runtime_compat.hpp`. So the package compat headers' copies did no work. `cudaErrorInvalidValue` has no user at all in the preprocessor or its dependencies' headers.

The build confirms it. All three packages build rc 0 at 129d61b with the names gone, and `harness/ground_segmentation` (which force-includes the ground-segmentation header) uses none of them.

### Every remaining name and its use (package source that spells it; count of whole-word occurrences outside `hip/`)

autoware_cuda_pointcloud_preprocessor (27 names):
- Types: `cudaStream_t` (51, every kernel/filter source and header), `cudaMemPool_t` (10: `cub_executor.hpp`, `cuda_pointcloud_preprocessor.hpp`, the polar filters, downsample, `thrust_custom_allocator.hpp`), `cudaMemPoolProps` (1, `cuda_voxel_grid_downsample_filter.cu`).
- Constants:
  - `cudaMemcpyHostToDevice` (8: preprocessor, polar filters, downsample, undistort, the combine-cloud test)
  - `cudaMemcpyDeviceToHost` (13: same files plus `test_cuda_pointcloud_preprocessor.cpp`)
  - `cudaMemcpyDeviceToDevice` (3: downsample, `cuda_combine_cloud_handler.cpp`)
  - `cudaMemAllocationTypePinned`, `cudaMemLocationTypeDevice`, `cudaMemPoolAttrReleaseThreshold` (1 each, downsample pool setup)
  - `cudaDeviceScheduleBlockingSync` (1, `cuda_pointcloud_preprocessor_node.cpp`)
  - `cudaDevAttrMultiProcessorCount` (1, `cuda_pointcloud_preprocessor.cu`)
- Errors and devices: `cudaGetLastError` (11, after kernel launches in common/outlier/undistort/combine kernels and the preprocessor), `cudaGetDevice` (3, polar filters and downsample pool location), `cudaSetDeviceFlags` (1, preprocessor node), `cudaDeviceGetAttribute` (1, preprocessor).
- Memory:
  - `cudaMemcpy` (4: downsample, both gtests)
  - `cudaMemcpyAsync` (21)
  - `cudaMemcpyToSymbolAsync` (1, downsample `voxel_info_dev`)
  - `cudaMemsetAsync` (6: preprocessor, downsample, `thrust_custom_allocator.hpp`)
  - `cudaMemPoolCreate` (1, downsample), `cudaMemPoolDestroy` (1, noise filter), `cudaMemPoolSetAttribute` (1, downsample)
  - `cudaMallocFromPoolAsync` (4), `cudaFreeAsync` (5) (polar filters, downsample, thrust allocator)
- Streams: `cudaStreamCreate` (5: preprocessor, polar filters, downsample, combine-cloud handler), `cudaStreamDestroy` (1, noise filter), `cudaStreamSynchronize` (24).
- Non-macro content, all used:
  - `<cuda/std/cmath>`: `::cuda::std::abs` on floats, `cuda_polar_voxel_{outlier,noise}_filter.cu:174/156`.
  - `<cuda/std/functional>`: `::cuda::std::not_equal_to`, `cuda_voxel_grid_downsample_filter.cu:494`.
  - `thrust::cuda = ::thrust::hip` alias: the four `thrust::cuda::par_nosync` calls, downsample `.cu:356,482,491,498`.
  - The cub stand-ins: `cub/cub.cuh` is included by the preprocessor and both polar filters, and `cub/device/device_radix_sort.cuh` and `cub/device/device_run_length_encode.cuh` by the downsample filter.

autoware_ground_segmentation_cuda (27 names; every use is in the package's own headers `cuda_utility.cuh`, `cuda_mempool_wrapper.hpp`, `cuda_stream_wrapper.hpp`, `device_vector.hpp` or in `cuda_scan_ground_segmentation_filter.cu`):
- `cuda_utility.cuh`: `cudaError_t` (6), `cudaSuccess` (3), `cudaErrorInvalidValue` (2), `cudaErrorLaunchFailure` (1), `cudaGetLastError` (2); `cudaStream_t` (4, also in `cuda_stream_wrapper.hpp`).
- `cuda_mempool_wrapper.hpp`: `cudaMemPool_t` (2), `cudaMemPoolProps`, `cudaMemAllocationTypePinned`, `cudaMemLocationTypeDevice`, `cudaMemPoolAttrReleaseThreshold`, `cudaMemPoolCreate`, `cudaMemPoolDestroy`, `cudaMemPoolSetAttribute` (1 each).
- `cudaGetDevice` (2): `cuda_mempool_wrapper.hpp` and the filter `.cu`.
- `cudaMalloc` (1): the filter `.cu`.
- `device_vector.hpp`:
  - `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, `cudaMemcpyDeviceToDevice` (1 each)
  - `cudaMallocAsync` (1), `cudaMemcpyAsync` (3), `cudaMemsetAsync` (2)
  - `cudaMallocFromPoolAsync`, `cudaFreeAsync`, `cudaStreamSynchronize` (1 each)
- `cuda_stream_wrapper.hpp`: `cudaStreamCreate`, `cudaStreamDestroy` (1 each).
- The `cub/cub.cuh` stand-in: included by `cuda_utility.cuh:23` and the filter `.cu:19`.

autoware_probabilistic_occupancy_grid_map (8 names, no change):
- `cudaStream_t` (31, every kernel header and `.cu`, `occupancy_grid_map_base`, `cuda_pointcloud.hpp`, the node header).
- `cudaMemcpyHostToDevice` (13: `utils.cpp`, the fixed and projective maps, `binary_bayes_filter_updater.cpp`, `cuda_pointcloud.hpp`).
- `cudaMemcpyDeviceToHost` (1, `occupancy_grid_map_base.cpp`).
- `cudaStreamNonBlocking` and `cudaStreamCreateWithFlags` (1 each, `pointcloud_based_occupancy_grid_map_node.cpp`).
- `cudaMemcpyAsync` (14), `cudaMemsetAsync` (8), `cudaStreamSynchronize` (3).

Audit command (each name, per package): `grep -rnw --include='*.cu' --include='*.cpp' --include='*.hpp' --include='*.cuh' --include='*.h' --include='*.ipp' <name> <pkg> --exclude-dir=hip`.

### Environment and checks

- Rebuilt inside this job directory from the round-3 job's sources and scripts, with paths repointed, in 1125 s, all steps rc 0:
  - TheRock ROCm 10.2.0 wheels into `$FLEET_VENV` (`rocm[devel,device-gfx942]==10.2.0`, HIP 7.17.26434); no system ROCm.
  - System libraries in 438 s, then the ROS 2 Jazzy underlay and the 74-package overlay.
  - `autoware_cuda_utils` from AMD-Ecosystem/autoware_cuda at `992d7034b5396900593126e8f19a1982ae7bc682` (`-- cuda is not found, building autoware_cuda_utils with HIP 7.17.26434`).
  - cuda_blackboard at `9e0fb865cd2e38dcc0db5e0e55497f40fb71f733`, the current `moat-port` tip (`-- cuda was not found, building cuda_blackboard with HIP 7.17.26434`).
  - Both dependencies were cloned and built in the job's `aw_ws`, with `MAX_JOBS=48` respected. The fmt link workaround of earlier rounds is still needed (environment only).
- Build at 129d61b (recipe of the Porter stage 1 section): rc 0, 1 min 38 s, all three `cuda was not found, building <pkg> with HIP 7.17.26434`.
- Pre-push gtests under `ulimit -c 0` and `timeout`: test_cuda_combine_cloud_handler 8/8, test_cuda_pointcloud_preprocessor 20/20.
- `jargon.py --port autoware_universe`: clean. Fork clone clean.
- Tooling note: the fleet guard resolves a `git push` remote from the directory the command starts in, so `cd <fork> && git push origin` from the moat checkout is refused as a push to the moat repository. `git -C <fork path> push origin moat-port` is the form it accepts.

## Review 2026-11-24, round 4 delta (reviewer, linux-gfx942 fleet job; delta `4f3358e...129d61b`, 1 commit)

pr-review skill, local-branch mode, run in the same job as Porter round 4. Independence came from a read-only child check (its own read of the delta, the dependency headers, the build's include paths and the commit message), and every point was re-verified by hand. **Verdict: pass.** The code delta has no findings. One commit-message inaccuracy is confirmed. It cannot be fixed by a new commit, so it is recorded below for the person preparing the upstream PRs rather than sent back to the porter.

### For the person preparing the per-package PRs (not a porter action; no amend or force-push is allowed on the branch)

1. **The `129d61b` message body overstates one name.** It says "These names appear only inside the error-check helpers of autoware_cuda_utils and cuda_blackboard, whose headers map them themselves before using them." That is true for `cudaError_t`, `cudaSuccess`, `cudaGetErrorName` and `cudaGetErrorString`. It is not true for `cudaErrorInvalidValue`: no header the preprocessor includes spells it. Its only occurrences are autoware_cuda_utils' own mapping (`hip/include/cuda_runtime_hip_compat.h:41-42`) and `autoware_cuda_utils/test/test_cuda_check_error.cu:31`, and cuda_blackboard does not map it at all. The removal is still right; only the sentence is wrong. Correct wording: "...map them themselves before using them; cudaErrorInvalidValue is used nowhere."
2. **The commit spans two packages.** Ruling Q4 = iii means one upstream PR per package, so preparing those PRs has to split `129d61b` into a preprocessor part and a ground-segmentation part anyway. Jeff asked for the removals in one commit on this branch. The split rewrites history, which is a person's step under the porter rules. Doing the split then is the natural moment to fix item 1.

Options for item 1:
- (a) Reword it when splitting the commit per package at PR preparation (recommended: no branch churn now).
- (b) Have a porter revert `129d61b` and re-apply it as two new per-package commits with corrected messages. This is allowed without amend or force-push, but it adds two commits of history and costs a revalidation.
- (c) Leave it as is. Not recommended: the text is upstream-visible.

### Checked and not a finding (so the next round does not redo it)

- **No removed name is needed.** No source in either package outside `hip/` spells any removed name (any `.cu/.cpp/.hpp/.cuh/.h/.ipp`, macros included). The only macro that expands to some of them is autoware_cuda_utils' `CHECK_CUDA_ERROR`. Its header `cuda_check_error.hpp` includes `<cuda_runtime_api.h>`, which in the build's `flags.make` resolves to the installed `autoware_cuda_utils/hip/cuda_runtime_api.h` shim; no other `cuda_runtime_api.h` is on the host's paths. That shim maps every name the helper uses, to the same HIP targets as the deleted lines. cuda_blackboard's `cuda_error.hpp` includes `cuda_runtime_compat.hpp` first, which maps the same names under `__HIP_PLATFORM_AMD__`. The installed `autoware_pointcloud_preprocessor` headers spell none of the names. Ground segmentation still maps the names its own `cuda_utility.cuh` spells (`cudaError_t`, `cudaSuccess`, `cudaErrorInvalidValue`, `cudaErrorLaunchFailure`).
- **Every remaining name is used.** Each remaining name in all three compat headers is spelled by at least one source of its package outside `hip/`, as tabulated in Porter round 4. So is the non-macro content of the preprocessor header: `cuda::std::abs` and `not_equal_to`, the `thrust::cuda` alias for the four `par_nosync` calls, and the three cub stand-ins. The two ground-segmentation `<cub/cub.cuh>` includes use its stand-in.
- **Binary equivalence on gfx942.** 4f3358e was built in a separate worktree (`<job>/sB`, a path of the same length as `<job>/s1`, because `CHECK_CUDA_ERROR` instantiates `cuda_check_error<char const(&)[N]>` on the `__FILE__` length). `utils/codeobj_diff.py <job>/sB/install/<pkg> <job>/s1/install/<pkg>` gives `verdict=identical` for all three packages (7 + 3 + 8 ELF files; exported symbols and device ISA identical). A first comparison from a path one character longer differed only in those `__FILE__`-length template names, which is how that cause was found.
- **CUDA path.** The headers are force-included only on the `USE_HIP` branch (`$<$<COMPILE_LANGUAGE:HIP>:-include;...>`), so nothing a CUDA build compiles changed. This host has no nvcc, so the CUDA compile was not re-run; by the same reasoning the delta is invisible to nvcc.
- **Commit hygiene.** Title 68 chars, `[ROCm]` right (HIP-only headers), AI disclosure present, fenced Test Plan, no trailer, ASCII, `jargon.py --port autoware_universe` clean.
