# apollo notes

## Intake screen (2026-11-05)

Lead: third-party "AMD AI Software Gap Matrix" entry "Baidu Apollo (autonomous driving stack)", verdict unknown, AMD path upstream. Screened at upstream `master` as of 2026-11-05 (last push 2026-04-16).

### Licence

`python3 utils/licenses.py check ApolloAuto/apollo` -> Apache-2.0, tier 1, cleared to contribute. Not NVIDIA-owned, so no per-file NVIDIA scan was run; submodules and vendored `third_party/` were not audited, because the recommendation is a decline.

### Existing AMD support: upstream, AMD-authored, merged

- Upstream PR https://github.com/ApolloAuto/apollo/pull/14681 ("20220926rocm", body "ROCm support added.", 224 files) was opened from the `amd/apollo` fork by GowthamKudupudi and merged 2023-01-19. Follow-up https://github.com/ApolloAuto/apollo/pull/14850 "AMD graphics support" (daohu527, a maintainer) merged 2023-03-25. This is the AMD support that shipped in Apollo 8.0.
- `amd/apollo` (the stale fork named in the lead) is 0 commits ahead and 1266 behind upstream `master` (`gh api repos/ApolloAuto/apollo/compare/master...amd:master`). Everything it carried is upstream; its last push (2023-01-18) is the day before #14681 merged. The `amd` org's private repos and branches are behind SAML SSO for this token, so any unpublished AMD work there remains unknown.
- What is in the tree today:
  - `tools/bazel.rc`: `build:amd --define GPU_PLATFORM=AMD`, `-DGPU_PLATFORM=AMD`, `USE_GPU=true`, alongside `build:nvidia`.
  - `third_party/gpus/rocm_configure.bzl`, `find_rocm_config.py`, `crosstool/hipcc_cc_toolchain_config.bzl.tpl`, `crosstool/clang/bin/crosstool_wrapper_driver_is_clang_for_hip.tpl`, `rocm/build_defs.bzl.tpl`: a full TensorFlow-style ROCm Bazel toolchain; `third_party/gpus/common.bzl` provides `gpu_library`, `if_cuda`, `if_rocm`.
  - `third_party/libtorch/libtorch_gpu_rocm.BUILD`.
  - `modules/perception/common/inference/migraphx/` (MIGraphX inference backend, with HIP plugin kernels and test data), selected in `inference_factory.cc` by `#elif GPU_PLATFORM == AMD`.
  - `docker/build/dev.x86_64.amd.dockerfile`, `cyber.x86_64.amd.dockerfile`, `installers/install_migraphx.sh`.
  - README hardware section: "NVIDIA Turing GPU / AMD GFX9/RDNA/CDNA GPU is strongly recommended", ROCm 5.1+; installation guide has an "Installing ROCm for AMD GPU" section.
- Open/closed PR search (`gh pr list --state all --search "rocm OR hip OR amd"`): only #14681, #14693 (closed duplicate of the same work), #14850, #14847 (closed duplicate). No open competing effort.

Classification per `assess-existing-support.md`: mature, AMD-authored ROCm/HIP support merged upstream -> `already-supported`.

### Residual gap (not a port; a validation question)

The AMD path landed for 8.0 and has seen little care since: `rocm_configure.bzl` last changed 2023-12-05, the AMD dev dockerfile last changed 2024-08-07, and the docs still cite ROCm 5.1. Upstream has since added GPU modules in 9.0 to 11.0 (for example `lidar_cpdet_detection`, `pointcloud_*/pointwise_unet`, `tensorrt/plugins/mmdeploy_plugins`, "Apollo 11.0 bev+occ" on 2026-02-27). Their BUILD files load `gpu_library`/`if_rocm`, but `lidar_cpdet_detection/BUILD` lists `@local_config_cuda//cuda:cudart` unconditionally, so some newer modules may not build under `--config=amd`. 44 `.cu` files in total, most in perception (TensorRT plugins 12, point pillars 12, MIGraphX plugins 5). Whether `master` still builds and runs with `--config=amd` on current ROCm is unknown; finding that out needs the multi-GB Docker/Bazel build, which is far outside a screen.

### Maintenance and build reality

- 26.9k stars, not archived, default branch `master`. Last GitHub release v9.0.0 (2023-12-18); 10.0/11.0 arrive as large squash-style "feat: release/upgrade" commits rather than tagged releases. Commits are sparse (five between 2026-02-09 and 2026-04-16); recent issues are planning-behaviour reports.
- Build: Bazel inside Apollo's Docker dev images (`apollo.sh`, `docker/build/*`), x86_64 and aarch64; repository size about 3.4 GB. Not a light build on any MOAT host, and not Windows-buildable in practice.

### Recommendation

Decline, `SKIP_REASON` `already-supported`: AMD itself upstreamed a ROCm/HIP + MIGraphX path (Bazel `--config=amd`, ROCm toolchain, AMD Docker images) that is merged and documented. MOAT would duplicate AMD's own work. If someone wants the gap closed, the value would be revalidating `--config=amd` on current `master` with current ROCm and porting the newer CUDA-only modules. That is a heavy Bazel/Docker job and a coordination question for whoever owns `amd/apollo`, not a fresh port.

No question for jeff.

## Ruling: validation project (Jeff Daily, 2026-11-10)

Jeff Daily overrode the intake decline: apollo becomes a validation project. Check that the ROCm support merged upstream in 2023 (ApolloAuto/apollo#14681) still builds and works. Validate on every required gate (wave64, wave32, Windows) and fix what breaks; the fixes go on `moat-port` of a new AMD-Ecosystem fork. A person creates that fork, never an agent; the project waits in `awaiting-fork` until it exists, then `release-forks` moves it to `screened` for the planner.


## Planning (2026-11-20, planner on linux-gfx942, no GPU used)

### Questions for Jeff (rule before a porter starts)

Q1 to Q4 need an answer before porting; Q5 can wait until PR prep. Q1 is a scope cut, so per the standing rule nobody assumes it.

**Q1. Scope: what about the default models that have no AMD backend at all?** The 2023 AMD path covered the Apollo 8.0 model set: Caffe nets through MIGraphX (`MINet`), libtorch nets through ROCm libtorch, and HIP kernels. Since then upstream made new backends the defaults, and none of them has an AMD path. These are Paddle Inference (a CUDA 11.1 prebuilt plus a CUDA custom op) and the TensorRT ONNX parser (`Onnx` / `MultiBatchInference`, using mmdeploy grid_sampler and modulated_deform_conv plugins). Affected: lidar CenterPoint (the default lidar detector), camera PETR (BEV) and CaDDN, BEVFormer occupancy, YOLOX3D, lidar CPDet, pointcloud motion and semantics UNets, traffic-light YOLOX detection and EfficientNet recognition, and the prediction `*_tensorrt` plugins. Options:
- (a) **Restore the 8.0 scope (recommended).** `--config=amd` builds all of `//modules/... //cyber/...`. Every HIP kernel compiles and is tested, including the ones whose model has no backend: cpdet, the pointwise_unet kernels, gpu_down_sample, radar4d pointpillars. The MIGraphX and libtorch model paths run. A Paddle or TensorRT-ONNX config fails with a clear "backend not available on AMD" error, and the docs list which models run on AMD. Those models would be recorded as deferred work with your ruling.
- (b) (a) plus a MIGraphX ONNX backend (`migraphx::parse_onnx`) standing in for `Onnx`/`MultiBatchInference`, plus grid_sampler and deformable-conv support. That brings the TensorRT-ONNX models onto AMD. It is feature work of the size of the original 2023 PR, and each model needs its ONNX file to import cleanly into MIGraphX.
- (c) (b) plus Paddle: there is no current Paddle Inference ROCm prebuilt. This means either building Paddle from source with ROCm (a large dependency build of uncertain viability) or running CPU Paddle Inference for CenterPoint/PETR/CaDDN (slow, and named as a CPU fallback).

**Q2. Build environment: on which OS and ROCm should the AMD dev image be rebuilt?** The AMD image is `rocm/dev-ubuntu-18.04:5.1.1-complete` (`docker/build/build_docker.sh:199`; prebuilt as `rocmapollo/apollo:dev-x86_64-rocm-18.04-20221027_0916`, still on Docker Hub). ROCm 5.1 cannot emit code for gfx942, gfx1100 or gfx1201, so no fleet gate can pass on it (gfx90a could, but only for wave64). ROCm 6 and 7 do not support Ubuntu 18.04, and Apollo's x86 dev image is 18.04 only, with installers that fetch 18.04-built binaries from Baidu's CDN. Options:
- (a) **Ubuntu 22.04 base with current ROCm (6.4+ or 7.x; recommended).** This covers gfx942, gfx1100 and gfx1201, and the README lists 22.04 as a supported host. The cost is making Apollo's installer set work on 22.04. The AMD dev image would then differ in OS from the NVIDIA x86 image.
- (b) Ubuntu 20.04 base with the last ROCm that supports 20.04 (verify which release; around 6.1/6.2). It reuses the 20.04 installer work already done for the aarch64 image. It covers gfx942 and gfx1100 but probably not gfx1201, and it pins an aging ROCm.
- (c) Keep the 2022 image and validate wave64 on gfx90a only. This proves nothing for current ROCm and cannot satisfy wave32. Not recommended.
Under (a) or (b), MIGraphX, RPP and libtorch come from ROCm packages and download.pytorch.org instead of the Google Drive tarballs.

**Q3. Windows gate: may an agent suggest a `windows` waiver for you to approve?** Apollo has no Windows build: it is Docker plus Bazel on Linux, CyberRT is POSIX, `rocm_configure.bzl:189-191` and `tools/bootstrap.py` refuse non-Linux, and there is no MSVC or clang-cl toolchain. This is the "obstacle is the operating system" case `config/arches.toml` describes. Options: (a) waive `windows` (recommended; only you can record it), or (b) keep the gate, which makes the project un-shippable.

**Q4. Fork default branch.** `status.json` has `fork_default_branch: main`, but `AMD-Ecosystem/apollo` (like upstream) has only `master`. That is a recorded ruling field, so I did not change it. If you agree: `python3 utils/moatlib.py set-base-branch apollo master --by "Jeff Daily"`.

**Q5. Delivery vehicle (can wait until PR prep).** Your ruling sends the fixes to the fork. Upstream does not link platform forks, so an upstream PR is a possible vehicle. But upstream activity is sparse (five commits Feb to Apr 2026) and community PRs from 2024 sit unreviewed. Options: (a) offer one upstream PR after validation, as the 2023 work did, or (b) keep it fork-only (`set-maintained` style, with your ruling).

### Findings (evidence in plan.md)

Fork `master` d53aa3da47a0 is identical to upstream. The analysis was read-only: the clone is in `projects/apollo/src`, nothing was built, and this host has no `/opt/rocm`.

- The AMD path does not build today, for three independent reasons:
  1. The `@local_config_cuda` repository is always real. Its dummy branch is commented out (`third_party/gpus/cuda_configure.bzl:1172-1174`), and `common.bzl:3` loads it from every GPU BUILD file. So an AMD box without CUDA and cuDNN fails at fetch time.
  2. `find_rocm_config.py` and `rocm_configure.bzl` probe the per-component `/opt/rocm/{hip,miopen,migraphx,hipblas,rocblas}` and `/hip/bin/hipcc` layout that ROCm 6.0 removed. They also hard-code `clang/14.0.0` and `15.0.0`.
  3. `apollo_perception_common_inference` depends unconditionally on the TensorRT `rt_net` (`inference/BUILD:100`) and on the CUDA Paddle prebuilt. It also includes `<NvInferVersion.h>` unguarded (`inference_factory.h:18`) and never links `migraphx:mi_net`.
- Silent bug: `./apollo.sh build_amd` adds `--config=gpu`, which `tools/bazel.rc:67` expands to `--config=nvidia`, then appends `--config=amd`. Both `-DNVIDIA=1 -DGPU_PLATFORM=NVIDIA` and `-DAMD=1 -DGPU_PLATFORM=AMD` reach the compiler. The 22 `#if GPU_PLATFORM == NVIDIA` sites then take the NVIDIA branch while Bazel selects the AMD deps (`scripts/apollo_base.sh:673-677`).
- Source compile breaks on the AMD branch:
  - `class OnnxLogger : {` (`onnx/onnx_single_batch_infer.cc:55`)
  - `modules/perception/comomn/...` (`darkSCNN_lane_detector.h:34`)
  - `gpu_down_sample/voxel_downsample.cu:19` includes `cuda_runtime.h`
  - the radar4d pointpillars copy never got the HIP shim
  - four targets still use the CUDA-only `@local_config_cuda` `cuda_library` (yolo postprocess, both pointwise_unet, planning_block)
  - unconditional `@npp` in two places, NVML in one, and stray cudart/TensorRT deps
- The kernels themselves are wave-safe. No `__shfl`, ballot, `warpSize`, full-mask, texture or constant-memory use. The NMS 64-bit masks pair with 64-thread blocks.
- The MIGraphX plugin copies have drifted from the TensorRT ones. Two zero-grid guards are missing: the `rcnn_proposal_plugin.cu` grid and `nms_cuda.cu`.
- Tests: only `mi_net_test` does real inference, and only 5 of its 19 instances have data in the repo. The other GPU tests are `syncedmem_test`, `blob_test`, `common_test` and the trajectory_smoother tests. Most kernel components have no tests (plan.md, Test plan).
- `surface.json`: the 61-file regex floor plus 49 planner components. No components are scoped out, because every candidate scope-out depends on Q1. The porter records `covered`/`scoped_out` after your ruling.
- `ext_type` recorded as `bazel`.
