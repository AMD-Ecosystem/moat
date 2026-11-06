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
