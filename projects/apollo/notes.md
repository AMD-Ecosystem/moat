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

