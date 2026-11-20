# apollo port plan

Planner run 2026-11-20 on linux-gfx942 (no GPU used, nothing built). Analysed at fork `master` d53aa3da47a0 (identical to upstream `master`, last upstream push 2026-04-16).

**Decisions Jeff must make before a porter starts are in `notes.md`, section "Planning (2026-11-20)", questions Q1 to Q5.** This plan is written for the recommended answers and says where another answer changes it.

## Project

- Upstream: https://github.com/ApolloAuto/apollo, default branch `master`.
- Fork: https://github.com/AMD-Ecosystem/apollo, default branch `master`. `status.json` records `fork_default_branch: main`, which does not exist on the fork (Q4). No `moat-port` branch exists yet; the porter cuts it from `master`.
- Size: 2.1 GB checkout, 12.4k files, 567 C++/Python test targets (483 in `modules/`, 81 in `cyber/`, 3 `py_test`).

## Existing AMD support and decision

- https://github.com/ApolloAuto/apollo/pull/14681 (GowthamKudupudi, from `amd/apollo`, merged 2023-01-19) and https://github.com/ApolloAuto/apollo/pull/14850 (maintainer daohu527, merged 2023-03-25). AMD-authored and merged upstream: authoritative. No other AMD effort found (re-checked `gh pr list --state all --search "ROCm OR HIP OR AMD OR migraphx"` on 2026-11-20: nothing after 2023).
- Classification: authoritative but rotted. The Apollo 8.0 AMD path (Bazel `--config=amd`, a TensorFlow-style ROCm toolchain, a MIGraphX backend for the Caffe models, ROCm libtorch, HIP compat macros in the kernels) is still in the tree. It no longer builds, for three independent reasons: the build infrastructure is frozen at ROCm 5.1 / Ubuntu 18.04 / Bazel 5, Apollo 9 to 11 refactors added unconditional CUDA/TensorRT dependencies to the shared perception layer, and the models added since 8.0 use backends with no AMD path at all.
- Decision (per Jeff's 2026-11-10 ruling): validate and repair the existing path; do not re-port. Every kernel the 2023 work HIP-enabled stays as written; new work follows the same idioms (`gpu_library`, `if_cuda`/`if_rocm`, the `#define cudaX hipX` blocks in `modules/perception/common/base/common.h`).
- Kernels are small pre/post-processing kernels (NMS, voxelisation, scatter, resize, ROI pooling), not NVIDIA-tuned GEMM/attention. A mechanical port is right; no AMD-native rewrite.
- Delivery vehicle: the ruling sends fixes to the fork. Whether to offer an upstream PR later is open (Q5). Upstream does not link platform forks, so a PR is a possible vehicle.

## Build classification

`ext_type: bazel`. WORKSPACE-based Bazel (no `MODULE.bazel`, no `.bazelversion`); `WORKSPACE:48` requires >= 3.7.0, and `docker/build/installers/install_bazel.sh:27,92-98` installs 3.7.1 then upgrades to 5.2.0. The build runs inside Apollo's Docker dev image (`docker/scripts/dev_start.sh`, `apollo.sh build_amd` = `bazel build --config=amd`, `scripts/apollo_base.sh:691-764`). It is neither Strategy A (CMake) nor B (torch extension); it is Strategy A's idea expressed in Bazel: one source tree, `.cu` compiled as HIP by `-x hip` through hipcc, a macro compat header, CUDA and ROCm deps selected by `if_cuda`/`if_rocm` (`third_party/gpus/common.bzl`).

## Port strategy

Repair in place, in this order, each layer proven before the next:

1. **Environment.** A dev image the current ROCm can target gfx942, gfx1100 and gfx1201 with (Q2). The 2022 image `rocmapollo/apollo:dev-x86_64-rocm-18.04-20221027_0916` (ROCm 5.1.1, `docker/build/build_docker.sh:199`) cannot emit code for any of them.
2. **Repository rules and toolchain** (`third_party/gpus`, `tools/bootstrap.py`, `tools/bazel.rc`): find ROCm 6/7, configure without a CUDA toolkit, pick gfx targets, compile C++17-clean.
3. **BUILD graph**: make every CUDA/TensorRT/NPP/NVML/Paddle dependency conditional, wire MIGraphX where the code expects it, convert the remaining `@local_config_cuda` `cuda_library` users to `gpu_library`.
4. **Sources**: the few compile errors (syntax error, typo, unguarded TensorRT/CUDA includes, two un-HIPified files).
5. **Tests**: build `//modules/... //cyber/...` under `--config=amd`, run the GPU tests and the regression set on each gate.
6. **Scope per Q1**: models without an AMD backend.

Keep Bazel 5.2 (what the image installs). Moving to Bazel 7 toolchain resolution is not needed for this port and would touch every platform.

## CUDA surface inventory

`surface.json` holds the 61-file regex floor (markers: 125 kernels, 115 launches; libraries: Thrust 65, cuBLAS 97, cuDNN 66 hits) plus 49 planner-added components. Summary by component (W = wired for AMD and HIP-clean, W* = wired but broken, C = CUDA-only build rule, T = TensorRT-only):

| Component | Files | AMD state | Work |
|---|---|---|---|
| perception/common/base (syncedmem, blob, compat macros) | `common/base/common.h` | W | none beyond `<hipblas.h>` -> `<hipblas/hipblas.h>` (ROCm 6 removed the flat header) |
| perception/common/inference/utils | `utils/{cuda_util,gemm,resize,util}.cu` | W | `<hipblas.h>` include |
| perception/common/inference/operators | `operators/roipooling_layer.cu` | W | none |
| perception/common/inference/migraphx + plugins | 5 `.cu`, `mi_net.cc` | W* | link it into consumers; port the two zero-grid guards the TensorRT copies gained (`rcnn_proposal_plugin.cu:282`, `nms_cuda.cu` vs tensorrt `nms_cuda.cu:128-129`); check internal headers `migraphx/gpu/context.hpp`, `gpu/hip.hpp`, `ref/target.hpp` against packaged MIGraphX |
| perception/common/inference (factory, torch, onnx, paddle) | `inference_factory.h:18`, `inference/BUILD:93-124`, `onnx/onnx_single_batch_infer.cc` | W*, blocks every perception target | guard `<NvInferVersion.h>`; `if_cuda` the TensorRT `rt_net` dep (BUILD:100) and Paddle (BUILD:109-111); `if_rocm` `migraphx:mi_net`; fix `class OnnxLogger : {` (onnx_single_batch_infer.cc:55) and the unguarded `<cuda_runtime_api.h>` (:26) |
| perception/common/inference/tensorrt (+ mmdeploy plugins) | 17 files | T | stays NVIDIA-only; must stop being pulled into AMD builds. MIGraphX covers its plugins except grid_sampler / modulated_deform_conv (Q1) |
| camera_detection_multi_stage YOLO postprocess | `detector/yolo/postprocess.cu` | C (BUILD) | `@local_config_cuda` `cuda_library` -> `gpu_library` + `if_cuda`/`if_rocm` (BUILD:3,52-57,61-77,98-103) |
| camera_tracking | `common/similar.cu` | W | none |
| lane_detection (denseline, darkSCNN) | headers | W* | typo `comomn` (darkSCNN_lane_detector.h:34); add `mi_net` dep |
| lidar_detection cnn_segmentation | `feature_generator.cu` | W | none |
| lidar_detection point_pillars | 6 `.cu` | W | lib links Paddle and `@centerpoint_infer_op-x86_64` (CUDA 11.1 prebuilts) unconditionally (lidar_detection/BUILD:194-200) |
| lidar_detection CenterPoint (default lidar model) | Paddle + custom op | C (prebuilt binaries) | Q1 |
| lidar_cpdet_detection | `cpdet.cu` | W* | drop unconditional cudart (BUILD:42); runtime is TensorRT-ONNX (Q1) |
| gpu_down_sample plugin | `voxel_downsample.cu`, `gpu_down_sample.cc` | W* | HIPify: `#include "cuda_runtime.h"` (.cu:19), `cudaHostAlloc`/`cudaHostGetDevicePointer`/`cudaDeviceReset` not in the macro set |
| pointcloud_motion / pointcloud_semantics pointwise_unet | 2 `.cu` | C (BUILD) | `cuda_library` -> `gpu_library`; runtime TensorRT-ONNX (Q1) |
| radar4d_detection point_pillars | 6 `.cu` + lib | W* | copy of lidar pointpillars that never got the AMD shim: `common.h:46` `<cuda_runtime_api.h>`, `point_pillars.h:53` `NvInfer.h`, lib BUILD:208-210 unconditional cudart/TensorRT; mirror the lidar copy's guards |
| planning_open_space trajectory_smoother | `planning_block.cu` | C (BUILD) | source has its shim (planning_block.h:19-34); BUILD:4,87-96 `cuda_library` -> `gpu_library` |
| drivers/camera/format | `yuv2rgb.cu` | C, aarch64 Jetson only | out of AMD scope (only built `if_aarch64`, x86 never compiles it; AMD is x86_64 only) |
| map/pnc_map | `cuda_util.cu` | dead | BUILD target commented out (map/BUILD:110-137); nothing to do |
| perception/common/camera | `image_data_operations*.cc` | W* | `@npp` unconditional (camera/BUILD:71); `<rppi.h>` is the legacy RPP API (check against current RPP); `rpp` target hardcodes `-L/opt/rocm/rpp/lib -lamd_rpp` (rocm/BUILD.tpl:186-200) |
| prediction | `prediction/BUILD:207` | W* | `@npp` unconditional; torch_gpu evaluators use ROCm libtorch; `*_tensorrt` prediction plugins are T (Q1) |
| perception/msg_adapter | BUILD:35-40 | C (stray deps) | drop unconditional cuda/TensorRT deps |
| statistics/resource GPU stats | `gpu_resource_statistic.*` | C (NVML) | `if_cuda` NVML; AMD equivalent (amd-smi lib) or a stub returning "unavailable" (porter's call, the smaller change wins) |
| audio siren detection | libtorch | W | none |
| camera_detection_occupancy BEVFormer | `bevformer_obstacle_detector.cc:24` | C/T | guard `<cuda_runtime.h>`; runtime TensorRT-ONNX (Q1) |
| perception_plugin/inference/onnx_multi_batch | BUILD:15-21 | T | Q1 |

No warp-size hazards: no `__shfl*`, `__ballot*`, `warpSize`, `__activemask`, `0xffffffff` masks, textures or `__constant__` anywhere. The NMS `uint64_t` masks pair with 64-thread blocks (block-level, wave-independent). Block shapes `dim3(32,8)`, 256, 512 are fine on both wave widths.

## Build infrastructure inventory (the bulk of the work)

| Item | Evidence | Problem | Change |
|---|---|---|---|
| CUDA repo always real | `third_party/gpus/cuda_configure.bzl:1170-1186` (dummy branch commented out "always retrun valid cuda lib for cpu compile"); `common.bzl:3` loads `@local_config_cuda` in every GPU BUILD | an AMD box with no CUDA toolkit fails repository fetch | create the dummy repository when `TF_NEED_ROCM=1` (keep NVIDIA and CPU behaviour as is) |
| ROCm discovery | `third_party/gpus/find_rocm_config.py:195-339` | probes `/opt/rocm/{hip,miopen,migraphx,hipblas,rocblas}/include` and top-level `rocm_version.h`, all removed in ROCm 6.0 | probe `include/rocm-core/rocm_version.h`, `include/hip/hip_version.h`, `include/{miopen,migraphx,hipblas,rocblas}/...`; regenerate `find_rocm_config.py.gz.base64` with `compress_find_rocm_config.py` |
| Toolchain paths | `rocm_configure.bzl:175-182,529-546,653,657` | `/hsa/include`, `/hip/include`, `/hip/bin/hipcc`, `/hip/lib`, `llvm/lib/clang/{14.0.0,15.0.0}/include`, per-component include copies | `<rocm>/bin/hipcc`, `<rocm>/lib`, clang resource dir discovered (`<rocm>/llvm/bin/clang -print-resource-dir`) |
| gfx targets | `rocm_configure.bzl:195-209`; `tools/bootstrap.py:55,876-888` | targets only from `rocm_agent_enumerator` at fetch time (fails in a container without the GPU); `bootstrap.py` writes `TF_ROCM_COMPUTE_CAPABILITIES`, which the rule never reads | make bootstrap write `TF_ROCM_AMDGPU_TARGETS`; keep the enumerator as the default; document `TF_ROCM_AMDGPU_TARGETS=gfx942` etc. |
| `NVIDIA=1` leak | `tools/bazel.rc:67` `build:gpu --config=nvidia`; `scripts/apollo_base.sh:673-677` prepends `--config=gpu` then appends `--config=amd` | compile line gets both `-DNVIDIA=1 -DGPU_PLATFORM=NVIDIA` and `-DAMD=1 -DGPU_PLATFORM=AMD`; 22 `#if GPU_PLATFORM == NVIDIA` sites in `modules/` then take the NVIDIA branch while Bazel selects AMD deps | do not add `--config=gpu` when AMD is selected (or have bootstrap's `build:gpu --config=rocm` win); verify with `--announce_rc` |
| hip wrapper | `crosstool/clang/bin/crosstool_wrapper_driver_is_clang_for_hip.tpl:1,21,134-136` | `#!/usr/bin/env python`, `import pipes` (gone in 3.13), only `-std=c++11/14` passed through | `python3`, `shlex`, pass `c++17` (rocThrust/hipCUB on ROCm 6+ need C++17 for `-x hip` units) |
| bootstrap | `tools/bootstrap.py:41,1099` | `distutils` (gone in Python 3.12), missing comma in an env list | fix both |
| libtorch ROCm | `third_party/libtorch/libtorch_gpu.BUILD`, `3rd-libtorch-gpu.BUILD:11,16-17,38`, `install_libtorch.sh:94-117` | AMD libtorch from a Google Drive ID via a cookie scrape, moves `libamdhip64.so.4` into `/opt/rocm/hip/lib`; packaging BUILD has unterminated strings and unconditional `-lnvonnxparser`/cudart; `libtorch_gpu_rocm.BUILD` is an orphan | source ROCm libtorch from download.pytorch.org (verify the ROCm libtorch archive for the chosen ROCm); fix the BUILD files. Risk: libtorch >= 2.1 needs C++17 while `tools/bazel.rc:91-92` builds `-std=c++14` globally; NVIDIA uses libtorch 1.7 |
| MIGraphX | `install_migraphx.sh:28-46` | custom tarball from Google Drive unpacked to `/opt/rocm/migraphx` | the ROCm `migraphx-dev` package |
| magma | `install_gpu_support.sh:30-35`, `install_magma.sh` | CUDA-only build, run on AMD too | skip on AMD (libtorch ROCm brings its own) |
| RPP | `install_rpp.sh` (called by nothing), `rocm/BUILD.tpl:186-200` | pinned 2022 commit, `-lamd_rpp` | ROCm `rpp-dev` package; fix the link line; port `<rppi.h>` calls if the legacy API is gone |
| Paddle Inference / centerpoint op | `third_party/paddleinference/workspace.bzl` (`paddleinference-cu111-x86.tar.gz`), `third_party/centerpoint_infer_op/workspace.bzl:15` | CUDA 11.1 prebuilts linked unconditionally | `if_cuda` on AMD; what replaces them is Q1 |
| Docker | `docker/build/build_docker.sh:5,199,204-206`, `dev.x86_64.amd.dockerfile:21-22,38`, `dev_start.sh:39-41` | ROCm 5.1.1 Ubuntu 18.04 base; `tensorflow==2.3.0` needs Python <= 3.8; prebuilt image on a third-party Docker Hub org | Q2 |
| Detection | `scripts/apollo.bashrc:107-168`, `installer_base.sh:198-224` | "rocm-smi printed something" (rocm-smi is deprecated for amd-smi, and prints a banner without a GPU) | also accept `amd-smi`; check `/dev/kfd` |

## Risk list

- **Environment cost dominates.** Apollo's x86 dev image is Ubuntu 18.04 only (`VERSION_X86_64="dev-x86_64-18.04-20240620_1444"`), and its installers pull many 18.04-built binaries from Baidu's CDN. ROCm 6 and 7 do not support Ubuntu 18.04. Any image that can target gfx942/gfx1100/gfx1201 means bringing Apollo's installer set up on 20.04 or 22.04 (the aarch64 image is already 20.04, so part of that exists). This is the first thing the porter must prove, and the most likely place to block.
- **C++ standard.** `-std=c++14` everywhere vs C++17 for ROCm 6+ device libraries and modern libtorch. Prefer per-target `-std=c++17` on `gpu_library` and libtorch consumers over a global switch; record the choice.
- **MIGraphX API drift.** `mi_net.cc` uses internal headers and the 2022 op API (custom ops, `migraphx::gpu` context); current packaged MIGraphX may differ.
- **Silent wrong branch.** The `NVIDIA=1` leak compiles; it does not fail. Prove the fix by `--announce_rc` and a grep of the hipcc command lines (`HIPCC_VERBOSE=1`), not by a green build.
- **Test data.** `mi_net_test` instantiates 19 cases; only 5 have data in `inference_test_data/migraphx/` (denseline bs1/bs4, darkSCNN bs1/bs4, traffic_light_detection). The other 14 fail on missing files independent of the port. `cnn_segmentation_test` needs `/apollo/modules/perception/common/testdata/lidar/app/...`, absent from the repo, and is `size="large"` (filtered by `--test_size_filters=small,medium`).
- **Model downloads** (`scripts/model_download.sh`) use signed Baidu BOS URLs dated 2024-11-19; they may have expired.
- **Plugin drift** between the TensorRT and MIGraphX copies (zero-grid guards) is a real bug on AMD, not just style.
- **Upstream engagement.** Sparse upstream activity (five commits Feb to Apr 2026); unreviewed community PRs from 2024 remain open.

## File-by-file change list (recommended answers)

- `third_party/gpus/cuda_configure.bzl`: dummy repository when ROCm is selected.
- `third_party/gpus/find_rocm_config.py` (+ regenerated `.gz.base64`): ROCm 6/7 layout.
- `third_party/gpus/rocm_configure.bzl`: include dirs, hipcc and runtime paths, include copies, clang resource dir.
- `third_party/gpus/crosstool/clang/bin/crosstool_wrapper_driver_is_clang_for_hip.tpl`: python3, shlex, c++17.
- `third_party/gpus/rocm/BUILD.tpl`: rpp link line; drop `migraphx_device` if no longer shipped.
- `third_party/gpus/common.bzl`: only if the `-O3` dead line matters (it does not change behaviour today; leave it).
- `tools/bazel.rc`, `scripts/apollo_base.sh`, `tools/bootstrap.py`: AMD config without the NVIDIA defines; target variable name; distutils; comma.
- `scripts/apollo.bashrc`, `docker/build/installers/installer_base.sh`, `docker/scripts/docker_base.sh`: AMD detection.
- `third_party/libtorch/{libtorch_gpu.BUILD,3rd-libtorch-gpu.BUILD}`, `docker/build/installers/{install_libtorch,install_migraphx,install_gpu_support,install_rpp}.sh`.
- `docker/build/build_docker.sh`, `docker/build/{cyber,dev}.x86_64.amd.dockerfile`, `docker/scripts/dev_start.sh`: per Q2.
- `modules/perception/common/inference/{BUILD,inference_factory.h,inference_factory.cc,onnx/onnx_single_batch_infer.cc}`.
- `modules/perception/common/inference/migraphx/plugins/{rcnn_proposal_plugin.cu,nms_cuda.cu}`: zero-grid guards.
- `modules/perception/lane_detection/lib/detector/darkSCNN/darkSCNN_lane_detector.h` (typo) and lane_detection BUILD (mi_net dep).
- `modules/perception/camera_detection_multi_stage/BUILD`, `pointcloud_motion/BUILD`, `pointcloud_semantics/BUILD`, `modules/planning/planning_open_space/BUILD`: `gpu_library`.
- `modules/perception/lidar_cpdet_detection/BUILD`, `lidar_detection/BUILD`, `radar4d_detection/BUILD` + radar4d `common.h`, `point_pillars.h`, `point_pillars_detection.cc`.
- `modules/perception/perception_plugin/lidar_detection/gpu_down_sample/{voxel_downsample.cu,gpu_down_sample.cc}`.
- `modules/perception/common/camera/BUILD`, `modules/prediction/BUILD`, `modules/perception/msg_adapter/BUILD`, `modules/statistics/resource/{BUILD,gpu_resource_statistic.*}`.
- `modules/perception/camera_detection_occupancy/.../bevformer_obstacle_detector.cc`: guard `<cuda_runtime.h>`.
- `<hipblas.h>` -> `<hipblas/hipblas.h>` in `common/base/common.h:27`, `inference/utils/cuda_util.h:23` (and `map/pnc_map/cuda_util.h:25` only if that target is ever re-enabled; it is not).
- `README.md:71` and `docs/01_Installation Instructions/prerequisite_software_installation_guide.md:89-153`: ROCm version and gfx targets that were actually validated.

Every edit keeps the NVIDIA and CPU paths byte-for-byte equivalent in behaviour; the reviewer should check each BUILD edit by reading what `if_cuda` still selects.

## Build commands (inside the AMD dev container, Q2)

```
./apollo.sh config -n
export TF_ROCM_AMDGPU_TARGETS=gfx942        # gfx1100 / gfx1201 on wave32 hosts
bazel build --config=amd --config=opt --announce_rc -- //modules/... //cyber/... -//modules/perception/common/inference/tensorrt/...
```

`./apollo.sh build_opt_amd` is the user-facing equivalent once the `--config=gpu` prepend is fixed; the validator must record both. Pin GPUs for tests with `HIP_VISIBLE_DEVICES` (shared hosts). Cap `--jobs` to the host's physical cores.

## Test plan

GPU tests (all run, counts recorded):

```
bazel test --config=amd --config=unit_test -- \
  //modules/perception/common/base/... \
  //modules/perception/common/inference/migraphx/... \
  //modules/perception/common/inference/utils/... \
  //modules/perception/common/inference:all \
  //modules/planning/planning_open_space/...
```

- `mi_net_test`: run all 19 instances; the 5 with data must pass, the 14 without data are recorded as data-absent (identical result expected on NVIDIA). Compare outputs against the bundled reference blobs the test already checks.
- `syncedmem_test`, `blob_test`, `common_test`: real device malloc/memcpy under `USE_GPU == 1`.
- `distance_approach_ipopt_cuda_interface_test` and the other trajectory_smoother tests: exercise `planning_block` once it compiles as HIP.
- Coverage the repo lacks: the kernels in point_pillars, cnn_segmentation, gpu_down_sample, cpdet, radar4d, yolo postprocess and camera_tracking have no tests. The validator adds a smoke run per component it can reach headless: at minimum, a CPU-vs-GPU differential on `gpu_down_sample` and on the point_pillars preprocess/scatter kernels using a recorded point cloud. If Q1 keeps model runs in scope, run the libtorch PointPillars and SMOKE detectors and the MIGraphX CNNSeg/denseline/darkSCNN models on Apollo's demo record (`cyber_recorder`, downloaded models per `scripts/model_download.sh`), and record detections against the CPU path where one exists.

Non-GPU regression set (must not regress; compiled with the ROCm toolchain and `USE_GPU`):

```
bazel test --config=amd --config=unit_test -- //cyber/... //modules/common/... //modules/planning/... //modules/map/... //modules/control/... //modules/prediction/... //modules/routing/... //modules/canbus_vehicle/...
```

And the full sweep, for the record: `bazel test --config=amd --config=unit_test -- //modules/... //cyber/... -//modules/perception/common/inference/tensorrt/...`. Record pass/fail/flaky counts, not just exit status; compare with the same sweep under `--config=cpu` on the same host and image to separate port regressions from pre-existing failures.

## Open questions

Q1 to Q5 in `notes.md` (scope of models without an AMD backend, build environment, Windows gate, fork default branch, upstream vehicle). The porter does not start until Jeff has answered Q1, Q2 and Q4.
