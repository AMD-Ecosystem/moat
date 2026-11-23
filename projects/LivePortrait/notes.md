# LivePortrait notes

## Intake screen (2026-11-05, windows-gfx1151 host, intake role)

Source: third-party "AMD AI Software Gap Matrix" seed, entry "LivePortrait and talking-head / lip-sync stacks" (verdict unknown, AMD path "upstream"). Upstream screened at its default branch `main`, shallow clone, last commit 2026-06-01.

### Licence

- Top level: MIT (`LICENSE`, "Copyright (c) 2024 Kuaishou Visual Generation and Interaction Center"). GitHub reports NOASSERTION because the file appends a note after the MIT text: "The code of InsightFace is released under the MIT License. The models of InsightFace are for non-commercial research purposes only." Recorded `license_spdx = MIT`.
- Model weights (HF `KwaiVGI/LivePortrait`): card licence `mit`. The InsightFace `buffalo_l` detection models it downloads are non-commercial research only (stated in the LICENSE note). This restricts USE, not contribution; no weights would be modified or redistributed by a port.
- Vendored `src/utils/dependencies/XPose/` (animals mode only): the vendored files carry Apache-2.0 headers from Deformable DETR / DETR / ED-Pose (SenseTime, Facebook, IDEA), and the `.cu` op is Deformable DETR's MultiScaleDeformableAttention (Apache-2.0, SenseTime 2020). However the X-Pose upstream repository (IDEA-Research/X-Pose, `LICENSE.txt`) is "License for Non-commercial Scientific Research Purposes" covering its codes, datasets and models. Mixed per-part: see question below.
- Vendored `src/utils/dependencies/insightface/`: MIT code (per LICENSE note).
- `python3 utils/licenses.py scan-nvidia agent_space/LivePortrait-screen`: no NVIDIA proprietary licence text.

### Where the CUDA is, and whether it is required

- The only compiled CUDA: `src/utils/dependencies/XPose/models/UniPose/ops/src/cuda/{ms_deform_attn_cuda.cu, ms_deform_im2col_cuda.cuh}` (~63 KB) plus `vision.cpp` and a CPU stub. This is the standard Deformable-DETR MSDeformAttn op, built by hand (`python setup.py build` in that directory) and used ONLY by animals mode (X-Pose animal keypoints, `src/utils/animal_landmark_runner.py`).
- `ops/setup.py` gates on `torch.cuda.is_available() and CUDA_HOME is not None`, else `raise NotImplementedError`. On ROCm PyTorch `CUDA_HOME` is normally None, so the op cannot even be attempted today; the gate needs a `ROCM_HOME` / `torch.version.hip` branch, after which `CUDAExtension` hipifies it. The op is widely HIP-ported elsewhere (mmcv, Deformable-DETR forks); expected faults are mild (atomicAdd on half in the backward pass, which inference does not use).
- `ops/functions/ms_deform_attn_func.py` does `import MultiScaleDeformableAttention as MSDA` unconditionally, though a pure-PyTorch `ms_deform_attn_core_pytorch` exists in the same file. A fallback to it would make animals mode run with no extension at all (slower).
- Humans mode (the default, `inference.py`, `app.py`) is pure PyTorch: `F.grid_sample` warping, SPADE generator, optional `torch.compile`. Device selection is `'cuda:' + device_id` with an mps branch, which works unchanged on ROCm PyTorch. `torch.backends.cudnn.benchmark = True` is harmless on ROCm (MIOpen). FP16 autocast is on by default (`flag_use_half_precision`), worth validating on RDNA for the "black boxes" artefact the config comment warns about.

### Other GPU dependencies

- `onnxruntime-gpu==1.18.0` (requirements.txt) for InsightFace face detection (`buffalo_l`, RetinaFace) and the human landmark model. Providers hard-coded to `["CUDAExecutionProvider"]` in `src/utils/cropper.py` and `src/utils/human_landmark_runner.py`; on AMD ORT falls back to CPU with a warning. The ROCm EP is gone and is not an AMD path. Two routes: CPU `onnxruntime` (works, per-frame detection/landmarks on CPU; cropping only, the heavy generator stays on GPU), or AMD's MIGraphX EP (`onnxruntime-migraphx` wheels, Linux only) to be measured. Windows has only CPU (or DirectML, not ROCm).
- No other compiled GPU deps; `transformers` (CLIP text encoder for X-Pose) is pure PyTorch.
- No MOAT project dependencies (`depends_on` stays empty).

### Duplicate effort

- Upstream docs: no AMD/ROCm/HIP mention. Upstream PRs (all states, "hip OR rocm OR amd"): none. Branches: only `main`. Open issues #110 "AMD GPU support?" (maintainer tagged it, no work) and #314 "DirectML/amd gpu support". HF discussion KwaiVGI/LivePortrait/discussions/1: maintainers said they might test on AMD, low priority.
- AMD-Ecosystem / ROCm orgs: no fork or same-named project found. `amd` and `AMD-AGI` orgs are behind SAML SSO for this token, so that part of the check is open.
- Community fork `thebestryan/LivePortrait-AMD` (1 star, pushed 2026-06-18, 3 commits ahead): humans mode only, adds `requirements_amd.txt` (torch 2.3.1 + ROCm 5.7, CPU onnxruntime) and an `onnx_provider.py` that prefers the removed `ROCMExecutionProvider`; states animals mode unsupported. Non-authoritative: hint only, not a base. It does confirm humans mode runs on ROCm PyTorch without code changes beyond ORT provider selection.

### Maintenance

- Not archived, ~19.4k stars. Last commits 2026-06-01, 2026-03-02, 2025-11-16; the last merged PRs are doc/link fixes (latest 2025-11-16), the last code PR merged 2025-02-02 ("fix: cuda 12.8 error"). 11 open PRs. Low-activity but alive; an upstream PR is possible but may sit.

### Assessment

Validate-lane-style job with a small compiled tail. Work:
1. Humans mode: run `inference.py` on ROCm PyTorch (fp16 and fp32) on wave64 / wave32 / Windows; provider selection that picks `MIGraphXExecutionProvider` when present, else CPU, without the CUDA-EP warning; a `requirements_rocm.txt` (no `onnxruntime-gpu`); README install section for AMD.
2. Animals mode: `ops/setup.py` ROCm branch so the MSDeformAttn op builds via hipify; validate forward against `ms_deform_attn_core_pytorch` on GPU; optionally a pure-PyTorch fallback when the extension is absent.
3. No test suite upstream: validation is the inference scripts on the bundled examples plus an op-level parity check against the PyTorch reference.

### Question for jeff

Animals mode vendors X-Pose. The vendored files carry Apache-2.0 headers (the `.cu` is Deformable DETR's, SenseTime, Apache-2.0), but the X-Pose upstream repo is under IDEA's "License for Non-commercial Scientific Research Purposes" for its codes, datasets and models. Is it acceptable to recommend contributing the HIP build fix to the vendored MSDeformAttn op (Apache-2.0 origin) inside that directory, or should the port be scoped to humans mode only (MIT code, no compiled extension) and leave animals mode deferred?

## Planning (2026-11-20, linux-gfx950 host, planner role; no GPU used, nothing built)

### Question 1 for Jeff (must be ruled before a porter starts): is Animals mode in scope?

This is the intake question, still open. Animals mode is the only part of LivePortrait with compiled CUDA: the vendored X-Pose `MultiScaleDeformableAttention` extension (`src/utils/dependencies/XPose/models/UniPose/ops/`). Its files carry Deformable DETR's Apache-2.0 headers (SenseTime 2020), but X-Pose's own repository (IDEA-Research/X-Pose) is under IDEA's "License for Non-commercial Scientific Research Purposes". LivePortrait already ships the subtree under its MIT top-level licence; a port would not add or redistribute anything new, only touch the vendored build.

Options:

- **A. Full port (planner's recommendation).** Humans mode + ONNX provider selection + ROCm requirements/README, AND Animals mode: a ROCm branch in the vendored `ops/setup.py` gate (the only source edit expected in X-Pose; the `.cu` hipifies as is), validated with the op's own `ops/test.py` (forward parity + gradcheck) and `inference_animals.py`. Reason: the edited file and the kernel are Deformable DETR code under Apache-2.0, the change is a build gate, and without it Animals mode cannot run on AMD at all; upstream documents Animals mode as a supported feature.
- **B. Humans mode only; Animals mode deferred** (`deferred.py add` against LivePortrait, README says Animals mode is NVIDIA-only for now). Zero edits under `XPose/`. The surface.json `owed` entries for the four Animals components become `scoped_out` with Jeff's ruling as the reason.
- **C. Humans mode, plus a pure-PyTorch fallback for Animals mode** (`ms_deform_attn_func.py` uses the existing `ms_deform_attn_core_pytorch` when the extension is missing) instead of building the extension. Still edits a vendored X-Pose file, changes behaviour for CUDA users without the extension (today a NameError), and is slower; listed for completeness, not recommended.

Until ruled, `surface.json` lists the four Animals components (`extension:MultiScaleDeformableAttention`, `test:msdeformattn-op`, `mode:animals-inference`, `app:gradio-animals`) under `owed` with this question as the reason, so `pr_ready` refuses; nothing was cut.

No other decision is needed: the ONNX Runtime route follows the standing rule (ROCm library over CPU fallback): MIGraphX EP on Linux, CPU EP on Windows named as a CPU fallback because no ROCm EP ships for Windows (`onnxruntime-migraphx` 1.27.1 on PyPI is manylinux x86_64 only; `pip download --platform win_amd64` finds nothing, 2026-11-20).

### Findings

- Upstream analysed at `9b294b3d0536` (main, 2026-06-02). `ext_type` set to `torch-extension`. Plan: `plan.md`; surface: `surface.json` (floor: 2 CUDA files, 7 kernels, 16 launches, no CUDA libraries; components added by hand since the scan finds none in a setup.py-less Python app).
- Humans mode is pure PyTorch; device strings `'cuda:N'` work on ROCm torch. The only AMD gap is ONNX Runtime: `CUDAExecutionProvider` hard-coded in `src/utils/cropper.py:59,62` and `src/utils/human_landmark_runner.py:36-40` (ORT silently falls back to CPU with a warning), and `requirements.txt` pins `onnxruntime-gpu==1.18.0`.
- MSDeformAttn kernels: no warp intrinsics, `warpSize`, lane masks or textures; backward reductions are `__syncthreads`-separated tree reductions (wavefront-width agnostic); blocks of `channels` threads; max static LDS 24 KB (double, 1024). `#include <THC/THCAtomics.cuh>` maps under torch hipify to `THH/THHAtomics.cuh`, which the ROCm torch install carries (checked in this host's torch 2.16 ROCm build, `include/THH/THHAtomics.cuh`); fallback is `<ATen/cuda/Atomic.cuh>`. `Tensor::data<T>()` / `type()` are deprecated but still present in torch 2.16 headers.
- `ops/setup.py:38` builds only with `torch.cuda.is_available() and CUDA_HOME is not None`; ROCm torch has `CUDA_HOME=None`, so the extension cannot be built on AMD today without the gate change.
- `ops/test.py` is the only real test in the repository: forward double/float vs `ms_deform_attn_core_pytorch`, and gradcheck for channels 30..3096 (covers all six backward kernels). Humans mode has no tests; validation is the inference scripts (plan.md, Test plan).
- `deformable_transformer.py:1052` turns autocast off only around the FFN; the attention op (float/double dispatch only) runs inside the runner's fp16 autocast. Same on CUDA; flagged as risk 7 in plan.md, not a port change.

## Ruling on Question 1 (Jeff Daily, 2026-11-23)

**A, full port.** Humans mode, ONNX provider selection, ROCm requirements and README, and Animals mode: a ROCm branch in the vendored X-Pose `ops/setup.py` gate, validated with the op's own `ops/test.py` (forward parity and gradcheck) and `inference_animals.py`. Standing ruling of the same day: Windows laptop GPU runs go last across all ports; they are queued, not started.
