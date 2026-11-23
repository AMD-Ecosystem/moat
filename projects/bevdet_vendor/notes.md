# bevdet_vendor notes

## Origin

Screened by jeff's 2026-11-23 ruling on autoware_universe's planner question Q2 (option B,
recorded in `projects/autoware_universe/notes.md` on `port/autoware_universe`, "## Rulings
on the planner's questions"): screen `autowarefoundation/bevdet_vendor` as a project; its
licence clearance is jeff's decision. `perception/autoware_tensorrt_bevdet` stays `owed`
in autoware_universe meanwhile.

## Intake -- 2026-11-23, linux-gfx942 (fleet job, headless, no GPU used)

Recommendation: **FORK, held at licence clearance** (jeff's decision; see Questions).
Technically small and viable, but the core code carries no licence at all, and this
project is headed for USE (a runtime dependency of Autoware's bevdet node) as much as for
contribution. Upstream: https://github.com/autowarefoundation/bevdet_vendor, default
branch `main`, screened at `dd65290a0` (2026-09-01, "build: make the CCCL::CCCL link
private (#9)"), 55 commits. It is a GitHub fork of
https://github.com/LCH1238/bevdet-tensorrt-cpp (294 stars, last push 2023-09-13, not
archived), the original author's TensorRT C++ inference for BEVDet.

### What it is

A ROS 2 (ament) package building one shared library: BEVDet multi-camera 3D detection
inference on TensorRT. `bevdet.cpp` owns its own engine handling (`createInferRuntime`,
ONNX-to-engine export through `nvonnxparser`), not Autoware's `autoware_tensorrt_common`.
GPU code, 96 KB in 8 `.cu` files, 13 `__global__` kernels: four TensorRT plugins
(Preprocess, BEVPool, AlignBEV, GatherBEV; `REGISTER_TENSORRT_PLUGIN`), IoU3D rotated NMS
(7 kernels), image preprocess and box postprocess (Thrust), plus `tools/export_engine.cu`.
`nvjpegdecoder.cpp` is behind `__HAVE_NVJPEG__`, which the CMake never defines (the CPU
libjpeg decoder is the built path). CMake: `find_package(CUDA)` + `cuda_add_library`,
`CUDA::cudart`, TensorRT via `tensorrt_cmake_module`, CCCL via `cmake/find_cccl.cmake`.
No inline asm, no warp shuffles.

### Licence -- none for the core; recorded "no licence file (unlicensed core; Apache-2.0 plugin and CMake files)", tier 4

    python3 utils/licenses.py check autowarefoundation/bevdet_vendor    # UNPARSED (GitHub: NONE)
    python3 utils/licenses.py check LCH1238/bevdet-tensorrt-cpp         # UNPARSED (GitHub: NONE)
    python3 utils/licenses.py scan-nvidia agent_space/bevdet_vendor-screen  # clean: no NVIDIA proprietary licence text

Recorded with `set-license-spdx` (evidence:
https://github.com/LCH1238/bevdet-tensorrt-cpp/issues/16). `license-gate bevdet_vendor`
is closed (tier 4) until a person runs `record-license-clearance`.

Full scan: full clone (all four branches: `main`, `bevdet_vendor-ros2`, `master`, `one`),
every tracked file read for copyright, SPDX, licence text and provenance comments. No
LICENSE, COPYING or NOTICE file on any branch; no submodules; no commit message mentions a
licence.

| path | licence / notice | origin |
|---|---|---|
| `package.xml` | `<license>LCH1238</license>` (the author's GitHub handle, not a licence); maintainer `cyn-liu` (AutoCore) | Autoware packaging, 2024 |
| `src/{alignbev,bevpool,gatherbev,preprocess}_plugin.cu` and their 4 headers in `include/` | Apache-2.0 header, "Copyright (c) 2021-2023, NVIDIA CORPORATION" | TensorRT OSS plugin template boilerplate; the kernel bodies inside are the author's BEVDet code (BEVPool v2 etc.), so the header's reach over them is unclear |
| `CMakeLists.txt` | Apache-2.0 twice: "Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES" (SPDX) and "Copyright 2024 AutoCore, Inc." | Autoware packaging |
| `cmake/find_cccl.cmake` | `SPDX-License-Identifier: Apache-2.0`, "Copyright 2026 The Autoware Foundation" | Autoware |
| `src/iou3d_nms.cu`, `include/iou3d_nms.h` | "Written by Shaoshuai Shi, All Rights Reserved 2019-2022", no licence named | OpenPCDet's `iou3d_nms_kernel.cu` (OpenPCDet is Apache-2.0); the in-file notice itself names none |
| `tools/open3d_vis.py` | "Copyright (c) OpenMMLab. All rights reserved." | mmdetection3d (Apache-2.0) |
| **everything else**: `src/bevdet.cpp`, `preprocess.cu`, `postprocess.cu`, `data.cpp`, `cpu_jpegdecoder.cpp`, `nvjpegdecoder.cpp`, `precision_config.cpp`, `demo_bevdet.cpp`, the remaining headers, `tools/export_engine.cu`, `export_one_engine.py`, `viewer.py`, `cfgs/`, READMEs | **no notice, no licence**: all rights reserved by default | LCH1238 (15 commits as `lch1238`, 30 as `lyu`), 2023 |
| `sample0/imgs/*.jpg`, `sample0/0.bin` | none stated | appear to be one nuScenes frame (the six nuScenes camera names); nuScenes data is CC BY-NC-SA 4.0 by its own terms, test data only |

The licence history is the decisive evidence. LCH1238/bevdet-tensorrt-cpp#16 "License
information" (2023-08-29): the author replied the same day "ok, I'll add it later." and
never did. Autoware maintainers followed up there twice with no answer:
`mitsudome-r` on 2024-11-11 ("Are there any updates regarding the License term?") and
`xmfcx` on 2026-09-01 ("we vendored this repository under
https://github.com/autowarefoundation/bevdet_vendor to use with Autoware"). So Autoware
itself vendors the code without a licence and knows it.

Scope note (the role's rule): tier 4 is about contributing. Contribution is the smaller
risk here; the project exists to be USED by Autoware, and a MOAT fork would publish a
modified copy of unlicensed code under AMD-Ecosystem. The 2026-08-06 org review jeff
applied to the earlier no-licence projects (DiffPhysDrone, yalm, ...) says "the risk is
in USING these projects, not in contributing to them; covers this project only", so it
does not carry over automatically and this one does carry the use risk.

### Duplicate effort and upstream state

- `gh search repos bevdet --owner AMD-Ecosystem` / `--owner ROCm`: none. Repository
  searches "bevdet rocm|hip|migraphx|amd", "bevpool hip|rocm": none.
- Upstream PRs (all 9, all closed or merged) and branches: nothing HIP/ROCm/AMD. The
  author's repo: no PR mentions HIP/ROCm/AMD; branches `master`, `one`. Docs grep (README,
  `doc/README_zh-CN.md`) for amd/rocm/hip/gfx/migraphx: no hits.
- Forks: 53 of the author's repo, 8 of the Autoware one; the five pushed in 2026 have no
  commits ahead of Autoware's `main` or no HIP-named branch. No AMD effort found.
- Activity: PRs #3 to #9 (Amadeusz Szymko, Ryohsuke Mitsudome, Mete Fatih Cirit) merged
  between 2026-01 and 2026-09; the original author has been silent
  since 2023-09.

### Viability -- yes for the code; the engine path is the real work

- Genuine CUDA, small: 13 kernels, Thrust, no asm, no warp-size assumptions spotted, no
  cuDNN (removed in #3), nvJPEG compiled out. Hipify-level work for the kernels.
- The blocker is TensorRT: the library builds and runs its own TensorRT engine and four
  TensorRT plugins. On ROCm that means the MIGraphX route jeff ruled for autoware_universe
  (stage 3: engine layer plus plugins as MIGraphX custom ops). bevdet_vendor does not go
  through `autoware_tensorrt_common`, so it would either adopt the MIGraphX engine layer
  being built in MOAT's `autoware_cuda` (in `porting`) or carry its own. Planner's call;
  no hard `depends_on` recorded now because the CMake does not consume autoware_cuda
  today.
- Test material: the ONNX export script is in https://github.com/LCH1238/BEVDet (a fork of
  HuangJunJie2017/BEVDet, both Apache-2.0); no released model files found, so a GPU gate
  needs a model exported from BEVDet weights. The repo has no automated tests
  (`ament_lint` only).
- Upstream destination would be `autowarefoundation/bevdet_vendor`, which accepts PRs;
  whether Autoware wants to take changes into a package with no licence is their
  question too.

### Recorded

    python3 utils/moatlib.py scaffold autowarefoundation/bevdet_vendor --branch main
    python3 utils/moatlib.py set-license-spdx bevdet_vendor "no licence file (unlicensed core; Apache-2.0 plugin and CMake files)" --evidence https://github.com/LCH1238/bevdet-tensorrt-cpp/issues/16
    python3 utils/moatlib.py set-intake bevdet_vendor fork ...
    python3 utils/moatlib.py set-state bevdet_vendor linux-gfx942 awaiting-fork --agent intake

### Questions for jeff

1. Licence clearance (`record-license-clearance bevdet_vendor`), or not. The core is
   unlicensed (author promised a licence in 2023 and never added one; Autoware's two
   follow-ups unanswered). Unlike the 2026-08-06 no-licence projects, this one is headed
   for use inside Autoware, so a clearance would cover publishing a modified fork of
   unlicensed code, not only contributing to it.
2. If you do not clear it: the decline I would propose is `license-blocked` ("core code
   has no licence; revisit if LCH1238 licenses bevdet-tensorrt-cpp"), with autoware_universe
   scoping `autoware_tensorrt_bevdet` out under its Q2 option C. Only you can record it.
3. If you do clear it: fork `AMD-Ecosystem/bevdet_vendor` (the adopt decision).
