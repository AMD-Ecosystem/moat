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

## Porting (2026-11-23, linux-gfx1100, porter role)

Fork `moat-port` at `00f208cdf0e7` (4 commits on `main` 9b294b3): provider helper + `requirements_rocm.txt` (60652fc), X-Pose `setup.py` ROCm gate (7436007), README/readme_zh_cn ROCm sections (e9e604e), README workaround wording (00f208c). Pushed; `advance-head` recorded. Dispatched by Jeff as a fleet job; nobody could answer questions during it, so the decisions below are recorded, not taken.

### Question 1 for Jeff: should the code work around the Animals-mode FP16 black boxes on ROCm?

Animals mode in its default half precision produces black boxes on this GPU (Radeon PRO W7800, gfx1100). The cause is a MIOpen defect, not the port: MIOpen's FP16 Winograd forward convolution overflows when inputs are large, although the exact output is far inside the FP16 range (evidence below; registered as `miopen-fp16-winograd-overflow`). Humans mode is not affected (no non-finite values in 32 generator outputs, video diffs below).

- **A. Documentation only (what the branch does now).** README (both languages) says: on some AMD GPUs, Animals mode in half precision shows black boxes; run with `--no_flag_use_half_precision` or set `MIOPEN_DEBUG_CONV_WINOGRAD=0`. No code change; the documented command `python inference_animals.py ...` still gives black boxes on gfx1100 until MIOpen is fixed.
- **B. Guarded code workaround.** On ROCm PyTorch only (`torch.version.hip`), `os.environ.setdefault("MIOPEN_DEBUG_CONV_WINOGRAD", "0")` before the first convolution in the Animals path (e.g. at the top of `inference_animals.py` and `app_animals.py`). Only the process-level variable was tested here; setting it after torch has run a convolution was not. Documented command just works; the cost is that every conv in the process loses Winograd (speed not measured; output within 1.35/255 of FP32). Upstream sees an env-var workaround for a vendor library bug.
- **C. Default Animals mode to FP32 on ROCm.** Change the default of `flag_use_half_precision` for the Animals path when `torch.version.hip` is set. Exact FP32 output, slower, and a behaviour difference users may not expect.

Recommendation: B is the "documented command works" option the standing rule leans towards, but it puts a vendor-bug workaround in upstream code, which is Jeff's call. A is what is pushed. Nothing else in the port depends on this answer.

### Other items needing a person

- **Fork default branch.** The standing rule makes `moat-port` the GitHub default branch of every AMD-Ecosystem fork; `AMD-Ecosystem/LivePortrait`'s default is still `main`. Changing it is a `gh` write a fleet worker may not make. Command for a person: `gh repo edit AMD-Ecosystem/LivePortrait --default-branch moat-port`.
- **CUDA no-regression compile** was not run: this host has no nvcc. The CUDA-side changes are the `setup.py` condition (CUDA clause unchanged, inside an `or`) and the provider helper (returns `CUDAExecutionProvider` whenever `torch.version.hip is None`, as before). Owed on the nvcc host (mi355) before presenting: `cd src/utils/dependencies/XPose/models/UniPose/ops && python setup.py build` against a CUDA torch. Record `cuda-not-validated` if the host's torch is ROCm-only.
- **Windows**: queued last per the 2026-11-23 ruling; not started.
- **`benchmark:speed` is `owed`** in surface.json (speed.py aborts in inductor, below). Scoping it out or accepting a rerun elsewhere is a person's call.
- **Draft MIOpen issue** (for Jeff; not posted). Target ROCm/rocm-libraries with labels `bug` and `project: miopen` (look up exact label names with `gh label list` first). Title: `[MIOpen] FP16 Winograd conv returns inf for in-range output`. Body:

====
MIOpen's FP16 Winograd forward convolution returns inf/NaN for inputs whose exact convolution result is well inside the FP16 range. Seen on a Radeon PRO W7800 (gfx1100) with TheRock ROCm 10.2.0a20261022 (HIP 7.17) and PyTorch 2.16.0a0+rocm10.2.0a20261022.

Reproducer (PyTorch):

```python
import torch, torch.nn.functional as F
torch.manual_seed(0)
x = torch.zeros(1, 512, 64, 64)
sign = (torch.arange(64)[:, None] + torch.arange(64)[None, :]) % 2 * 2 - 1
x[:, 0] = 30000.0 * sign
w = torch.randn(512, 512, 3, 3) * 0.01
xh, wh = x.half().cuda(), w.half().cuda()
ref = F.conv2d(xh.float(), wh.float(), padding=1)
out = F.conv2d(xh, wh, padding=1).float()
print(ref.abs().max().item(), torch.isinf(out).sum().item(), torch.isnan(out).sum().item())
```

| input magnitude | exact output max | default (`ConvWinoFuryRxS<2-3>`) | `MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3=0` (`ConvBinWinogradRxSf2x3g1`) | `MIOPEN_DEBUG_CONV_WINOGRAD=0` |
|---|---|---|---|---|
| 16000 | 1792 | finite, max abs err 1.75 | finite, 1.58 | finite, 0.48 |
| 30000 | 3360 | 2097152 of 2097152 outputs inf | finite, 2.89 | finite, 0.97 |
| 40000 | 4480 | 2097152 NaN | 2097152 inf | finite, 0.99 |

The failures start where the Winograd input transform (sums and differences of up to four inputs) exceeds 65504, so the transformed tiles appear to be held in FP16. Real-world impact: LivePortrait's animal generator has a 3x3 conv with input magnitude 38304 and exact output max 12689; with the default solver 2048 outputs become inf and the rendered video shows black boxes. Expected: an in-range result, as the direct and implicit-GEMM solvers give.
====

### Environment (reproducible recipe)

- Host: AMD Radeon Pro W7800 48GB, gfx1100, `HIP_VISIBLE_DEVICES=0` (job pin). Every GPU run under `ulimit -c 0` and `timeout`.
- ROCm 10.2 exists only as nightlies: `repo.amd.com/rocm/whl/<family>/` stops at 7.13 (whl-multi-arch at 7.14.1); `https://nightly.repo.amd.com/rocm/whl-next/` (redirects to `/rocm/pytorch/whl-next/`) carries 10.1, 10.2 and 10.3. Installed into the job venv (`$FLEET_VENV`, Python 3.11):
  `pip install --pre --index-url https://nightly.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "torch[device-gfx1100]==2.16.0a0+rocm10.2.0a20261022" "torchvision==0.30.0a0+rocm10.2.0a20261022" "rocm[libraries,devel,device-gfx1100]==10.2.0a20261022"` (HIP 7.17.26422). pip warns about the base image's 10.1 device wheels reached through the venv's base-links `.pth`; harmless, the venv's 10.2 copies shadow them (`/proc/self/maps` shows the venv `_rocm_sdk_core/lib/libamdhip64.so.7`).
- `pip install -r requirements_base.txt` (pins resolve with torch 2.16 nightly; numpy 1.26.4, onnx 1.16.1), then `onnxruntime-migraphx==1.27.1`.
- **MIGraphX runtime**: TheRock's SDK has none and the host has no `/opt/rocm`. Built AMDMIGraphX develop `f0239c9` (2.19.0, `libmigraphx_c.so.3`) from source against the venv SDK, about 90 s at -j32: deps nlohmann/json v3.8.0, msgpack-c cpp-3.3.0, rocm-cmake 6a7c5b73 (the SDK's ROCmCMakeBuildTools 0.14.0 lacks `rocm_add_version_resource`) into `$J/deps/prefix`; system protobuf 3.21 and sqlite3. Configure: `CXX=$R/llvm/bin/clang++ CC=$R/llvm/bin/clang cmake .. -G Ninja -DCMAKE_PREFIX_PATH="$J/deps/prefix;$R" -DROCM_PATH=$R -DHIP_PATH=$R -DCMAKE_INSTALL_PREFIX=$J/deps/migraphx-install -DGPU_TARGETS=gfx1100 -DMIGRAPHX_ENABLE_MLIR=OFF -DMIGRAPHX_USE_COMPOSABLEKERNEL=OFF -DMIGRAPHX_ENABLE_PYTHON=OFF -DMIGRAPHX_ENABLE_TENSORFLOW=OFF -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH=<venv _rocm_sdk_core/lib;_rocm_sdk_libraries/lib;install lib>` with `R=$(rocm-sdk path --root)`. MIOpen, rocBLAS, hipBLASLt, hipRTC stayed on. Run time: `LD_LIBRARY_PATH=$J/deps/migraphx-install/lib` (the EP's RUNPATH is `/opt/rocm-7.2.4/lib:/opt/rocm/lib`); when torch is not imported first, the venv `_rocm_sdk_core/lib` and `_rocm_sdk_libraries/lib` must be on it too, else the EP fails to load and ORT silently retries on CPU.
- Extension: `ROCM_HOME=$(rocm-sdk path --root) PATH=$ROCM_HOME/bin:$PATH PYTORCH_ROCM_ARCH=gfx1100 python setup.py build install` in `ops/`. Set `ROCM_HOME` explicitly: `hipcc` first on PATH is the base image's 10.1 SDK. Builds with deprecation warnings only (`Tensor.data<T>()`, `type()`, `atomicAddNoRet`); `THC/THCAtomics.cuh` hipifies to `THH/THHAtomics.cuh`, present. Offload arch gfx1100.
- Weights: `snapshot_download('KwaiVGI/LivePortrait', local_dir='pretrained_weights', ignore_patterns=['*.git*','README.md','docs/*'])`. With huggingface_hub 2.x the README's `huggingface-cli download ... --exclude "*.git*" "README.md" "docs"` downloads only README.md (the CLI reads the trailing arguments as filenames and ignores `--exclude`). Upstream doc drift, not ROCm.
- Gradio needed environment pins that upstream does not make (upstream drift, affects CUDA users equally; not changed in the port): `huggingface_hub<1` (gradio 5.1.0 imports `HfFolder`), `fastapi==0.115.2`, `pydantic==2.10.6` (gradio_client 1.4.0 fails with `TypeError: argument of type 'bool' is not iterable` on pydantic 2.14, HEAD / returns 500), and `no_proxy=127.0.0.1,localhost`.

### Evidence

All on gfx1100 at the code of `e9e604e` (00f208c changes only README wording).

**X-Pose op (`ops/test.py`)**: `check_forward_equal_with_pytorch_double` True (max abs 8.67e-19, rel 2.35e-16); `..._float` True (max abs 4.66e-10, rel 1.13e-07); `check_gradient_numerical` True for D=30, 32, 64, 71, 1025. D=2048 OOMs in the test's slow-mode gradcheck (dense double Jacobians, 7.5 GiB each; 37.5 GiB in use of 45 GiB) in a fresh process too, and D=3096 likewise; a test-size limit, not a port fault. With `gradcheck(fast_mode=True)`: D=30 and 2048 True; D=1025 and 3096 failed only the reentrancy check ("Backward is not reentrant ... although analytical gradient matches numerical"), True with `nondet_tol=1e-12`. Measured run-to-run spread over 20 backward passes: 0 for D<=1024 in double; D=1025: 2.2e-16 (value), 9.9e-16 (loc), 4.2e-16 (attn); D=3096: up to 3.0e-15; float D=3096 up to 1.9e-6. The backward kernels accumulate with atomicAdd (and the >1024 multi-block path adds global atomics), so ordering nondeterminism is inherent on any GPU. Scripts: `agent_space/lp_gradcheck_one.py`, `lp_reentrant.py` (container-local).

**Humans mode** (`inference.py`, MIGraphX EP active, no provider fallback message, model cache warm): default s0/d0 fp16 19 s; s9/d0 19 s; s13.mp4 (video source)/d0 22 s; s9/d12.jpg 11 s; s9/d0.pkl 17 s; `--no_flag_use_half_precision` 34 s; `--flag_do_torch_compile` 548 s cold (one autotune candidate rejected by inductor: "permute(sparse_coo) ... input.dim() = 5 is not equal to len(dims) = 4. Ignoring this choice"), output within mean 0.548/255 (max 29) of eager fp16. `talking.pkl` and `laugh.pkl` fail on every platform with `KeyError: 'c_d_eyes_lst'` (animal templates without the human eye/lip keys); upstream data, not ROCm.

**FP16 vs FP32 vs CPU, Humans mode** (ONNX on CPU EP for all three so the crops are identical):
- Decoded output video (s0/d0, 78 frames, 704x600, values 0..255): GPU FP32 vs CPU FP32 mean 0.062, p99 2, max 22, 0 dark 16x16 blocks; GPU FP16 vs CPU FP32 mean 0.544, p99 3, max 29, 0 dark blocks; GPU FP16 vs GPU FP32 mean 0.544, max 26. ("dark block": 16x16 block mean < 8 in the candidate but > 40 in the reference.)
- Generator output before encoding (4 sources x 8 d0 frames, 32 x 3x512x512; `agent_space/lp_tensorcmp.py`), x255: no NaN/Inf in any mode; GPU FP32 vs CPU mean 0.0003, max 0.040; GPU FP16 vs CPU mean 0.378, p99.9 8.6, max 66.5, 2.73% of values off by >2, 0.12% by >8, 0 dark blocks. The large FP16 differences are thin edge lines: per image at most 42 pixels exceed 32/255, the largest connected region 20 px.
- CPU wall clock for the same video: 364 s vs 61 s (GPU FP32) and 106 s (GPU FP16, cold MIOpen find).
- MIGraphX vs CPU cropping changes the output by mean 0.53-0.54/255 (max 25), the same size as the FP16 effect, 0 dark blocks.

**Animals mode** (`inference_animals.py -s s39.jpg -d wink.pkl --driving_multiplier 1.75 --no_flag_stitching`, 57 frames): FP32 finishes, no non-finite values. FP16 (default) finishes but `parse_output` warns "invalid value encountered in cast"; vs FP32: mean 96.0/255, 69.7% of pixels off by >16, 36307 dark blocks (black boxes). Forward hooks: first non-finite output at `spade_generator.G_middle_5.conv_1` (Conv2d), inputs finite with absmax 38304, output 2048 inf; NaN downstream in the decoder. Per-layer absmax over the run (FP32 / FP16): `G_middle_5.norm_1` 50304 / 50240, `G_middle_5.conv_1` 14540 / 14528 (finite part), so the model stays in FP16 range. Replaying the captured conv (x 1x512x64x64 fp16, w 512x512x3x3): exact output max 12689; default MIOpen (chooses `ConvWinoFuryRxS<2-3>`, also with `cudnn.benchmark`) 2048 inf; `MIOPEN_DEBUG_CONV_WINOGRAD=0` 0 inf, max abs err 3.77 (rel 6e-3); `MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3=0` 0 inf, max abs err 9.07. End to end with either variable: mean 1.35/255 vs FP32, max 25-33, 0 dark blocks. Synthetic reproducer and solver table in the draft issue above (`agent_space/lp_wino_synth.py`). Not checked on wave64; Fury is the RDNA Winograd kernel, so the wave64 validator should still run the FP16 Animals command and the dark-block comparison.

**ONNX provider parity** (`agent_space/lp_ortparity.py`, 26 bundled .jpg sources, 24 with faces; s39 and s41 are animals, no face in either provider): same face counts, boxes and scores equal to 3 decimals, 106-point landmarks within 0.026 px, 203-point landmarks within 0.049 px. Steady per image: CPU EP 86-92 ms, MIGraphX 12.2-12.5 ms. MIGraphX session build + first image: 517 s on the very first run on the host (cold), 57-58 s warm, 2.9 s with `ORT_MIGRAPHX_MODEL_CACHE_PATH` set and populated (132 MB cache). Per model, warm: det_10g compile 10.6 s then 2.2 ms, 2d106det 16.7 s then 0.7 ms, landmark 29.8 s then 6.4 ms. Provider helper branches checked by monkeypatching: ROCm+MIGraphX -> MIGraphX, ROCm without it -> CPU (landmark CPU session keeps 4 intra-op threads), CUDA torch -> CUDA, `flag_force_cpu` -> CPU. With `onnxruntime-migraphx` installed but no MIGraphX runtime, ORT prints "Failed to load library ... libmigraphx_c.so.3" and retries on CPU; the run still completes.

**Gradio**: `app.py` and `app_animals.py` serve `/` with HTTP 200 (after the environment pins above). No request driven through the UI.

**speed.py**: aborts (SIGABRT) while inductor max-autotunes the motion extractor: a Triton conv candidate for `convolution(1x192x32x32, 384x192x2x2)` stride 2 raises `hipErrorLaunchFailure`, then "Error in codegen for ComputedBuffer(name='buf235' ...)". The GPU is healthy afterwards. `inference.py --flag_do_torch_compile` compiles only the warping module and the generator and passes. Not a port change (speed.py is untouched); torch 2.16.0a0+rocm10.2.0a20261022 nightly. Not retried with a different hypothesis.

### Lesson for cuda-to-rocm: MIOpen FP16 Winograd overflows on large in-range inputs

For `faults-numerics.md` (half precision). Symptom: FP16 (autocast) inference gives black or NaN regions on an RDNA3 card, FP32 is clean, and the per-layer FP32 magnitudes are all well under 65504. Cause: MIOpen's FP16 Winograd forward solvers (`ConvWinoFuryRxS<2-3>`, then `ConvBinWinogradRxSf2x3g1`) overflow when inputs reach about 30000 (Fury) or 40000 (the other), apparently holding the input-transform tiles in FP16; the exact convolution output can be under 5000. Diagnose with forward hooks that report the first module whose inputs are finite and output is not, then replay that conv alone with `MIOPEN_LOG_LEVEL=5` (prints "FW Chosen Algorithm") and with `MIOPEN_DEBUG_CONV_WINOGRAD=0`. Workaround: `MIOPEN_DEBUG_CONV_WINOGRAD=0`, or FP32 for that model. Seen on gfx1100, ROCm 10.2.0a20261022 (LivePortrait Animals mode).

### Lesson for cuda-to-rocm: ONNX Runtime's MIGraphX EP from PyPI needs a MIGraphX runtime and a cache

For `libraries.md` (ONNX Runtime). `onnxruntime-migraphx` (PyPI 1.27.1, Linux x86_64 only) bundles only the provider library; it NEEDs `libmigraphx_c.so.3` and `libamdhip64.so.7` with RUNPATH `/opt/rocm-7.2.4/lib:/opt/rocm/lib`, so it needs ROCm's `migraphx` package (or a MIGraphX built against the same ROCm as the PyTorch in the process; the pip-installed TheRock SDK ships none). If the runtime is missing, ORT prints "Failed to load library" and silently retries on CPU, so check `session.get_providers()`. MIGraphX compiles each model on first use (tens of seconds per model, minutes cold); `ORT_MIGRAPHX_MODEL_CACHE_PATH=<dir>` saves the compiled programs and cut the session build from 58 s to 3 s. Select the provider only when `torch.version.hip` is set and `MIGraphXExecutionProvider` is in `get_available_providers()`, so a CUDA build keeps its provider unchanged. Building MIGraphX against a pip SDK takes about two minutes with MLIR and CK off; the SDK's `rocm-cmake` may be too old (`rocm_add_version_resource`), so use the commit MIGraphX's requirements.txt pins. (LivePortrait)

### Lesson for moat-porting: fleet guard reads command text, not intent

For the "Fleet hosts" section. The fleet guard refuses any shell command whose text contains `pip install`, including prose inside a heredoc (a README edit with an install line was refused); write such text with the Write tool and run the script. It judges `git push` by the repository the command starts in, so `cd <moat checkout> && ... && cd projects/<name>/src && git push` is refused as a push to the MOAT origin; run the push as its own command from inside the fork clone. (LivePortrait)

## Review 2026-11-23 (linux-gfx950, reviewer role, fleet job)

Reviewed `moat-port` at `00f208cdf0e7` (4 commits on `main` 9b294b3) with the pr-review skill; every point below was re-checked against the code. Porter Question 1 (FP16 Animals-mode workaround: docs only vs code) is Jeff's and still open; reviewed as the branch stands (option A, documentation only) and not treated as a finding.

Verdict: **no findings; review passed.**

Checked and clean: provider helper `src/utils/human_landmark_runner.py:26-32` returns `CUDAExecutionProvider` whenever `torch.version.hip is None`, so CUDA and CPU-only torch behave as before in both `cropper.py:59,62` and `LandmarkRunner`; ROCm without MIGraphX reaches the existing CPU branch with its 4 intra-op threads; mps and `flag_force_cpu` untouched; InsightFace never passed `device_id` to the detector on CUDA either, so the MIGraphX path matches it. `ops/setup.py:39` CUDA clause byte-identical inside an `or`; the ROCm clause needs no visible GPU (plan risk 6). `requirements_rocm.txt`: `onnxruntime-migraphx` 1.27.1 has cp310-cp314 manylinux_2_34 wheels; `onnxruntime` 1.27.0 has `win_amd64` and requires Python >= 3.11, which the README states. `https://download.pytorch.org/whl/rocm7.2` serves cp311 torch 2.11-2.14.1 and torchvision; torchaudio exists there only as 2.11.0, so the README's three-package command resolves to torch 2.11.0 (consistent, same pattern as upstream's CUDA command; torchaudio is unused by the code). Branch is 0 commits behind `origin/main` (fetched). `utils/jargon.py --port LivePortrait`: clean. Titles <= 72, `[ROCm]` earned on all four, "Authored with Claude.", Test Plans present, no trailer, ASCII. Strategy B (torch hipify) is right for the setup.py extension; kernels have no warp intrinsics or width assumptions (re-read, matches plan).

### CUDA no-regression compile (owed by the porter): done

Real nvcc 12.8 (V12.8.93, `/usr/local/cuda-12.8`) against `torch 2.11.0+cu128` in a job-local venv (`venv-cuda`), on a `git archive 00f208c` export of `ops/`:

```
CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST="8.0;9.0" python setup.py build
```

As-is this raises `NotImplementedError('Cuda is not availabel')` because the host has no NVIDIA GPU and the unchanged upstream CUDA clause requires `torch.cuda.is_available()`; same on upstream `main`. Rerun with `torch.cuda.is_available` patched to `True` in a wrapper that runs `setup.py` (so the unchanged CUDA branch is taken): build rc=0 in 15 s, 59 warnings (unused `q_col`, deprecated `Tensor::type()` / `data<T>()`; no errors), `.so` carries `sm_80` and `sm_90` cubins (`cuobjdump --list-elf`), and `import MultiScaleDeformableAttention` in the CUDA torch exports `ms_deform_attn_forward/backward`. Not checked: running it (no NVIDIA GPU); the Python provider helper on a CUDA host (by code reading only). `ROCM_HOME` imports cleanly from a CUDA torch's `cpp_extension`.

### Porter's MIOpen FP16 Winograd claim, checked on gfx950 (AMD Instinct MI355X, wave64)

TheRock ROCm 10.2.0a20261022 (HIP 7.17.26422), torch 2.16.0a0+rocm10.2.0a20261022 in `$FLEET_VENV`. The porter's synthetic reproducer (512x512x3x3 FP16, one input channel at +-mag, `padding=1`; reference in float64):

| mag | exact max | default (`ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC`, also with `cudnn.benchmark`) | Winograd forced: `ConvWinoRageRxS<2-3>` | `..._RAGE_RXS_F2X3=0` -> `ConvBinWinogradRxSf2x3g1` | also `RXS_F2X3=0 RXS_F2X3_G1=0` -> `ConvBinWinogradRxSf3x2` |
|---|---|---|---|---|---|
| 16000 | 1792 | finite, err 0.95 | finite, 1.41 | finite, 1.58 | finite, 0.76 |
| 30000 | 3360 | finite, 2.00 | 1048576 inf + 1048576 NaN | finite, 2.89 | finite, 1.42 |
| 40000 | 4480 | finite, 3.86 | 2097152 NaN | 2097152 inf | 273861 inf + 1758267 NaN |
| 60000 | 6720 | finite, 3.99 | 2097152 NaN | 2097152 inf | same as 40000 |

"Winograd forced" = `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0 MIOPEN_DEBUG_CONV_DIRECT=0 MIOPEN_DEBUG_CONV_GEMM=0 MIOPEN_DEBUG_CONV_FFT=0`; solver names from `MIOPEN_LOG_LEVEL=5` "FW Chosen Algorithm". Conclusions: the root cause holds and is not RDNA-only: every FP16 Winograd solver MIOpen has for this shape on gfx950 overflows at the same thresholds as on gfx1100 (the f2x3 kernels at 30000-40000, consistent with an FP16 input transform of up to four summed inputs: 4 x 16000 < 65504 < 4 x 30000). On gfx950 MIOpen does not pick Winograd for this shape (it is the same shape as `G_middle_5.conv_1`), so the default FP16 path does not overflow here; the end-to-end Animals FP16 run in validation below is the evidence for the real layer. For the draft MIOpen issue and the `miopen-fp16-winograd-overflow` deferral this adds: reproduced on gfx950 with the solvers forced (`ConvWinoRageRxS<2-3>`, `ConvBinWinogradRxSf2x3g1`, `ConvBinWinogradRxSf3x2`). For the lesson's promotion: the overflow is a property of MIOpen's FP16 Winograd solvers on both RDNA and CDNA; which GPUs hit it by default depends on solver selection. Script: `agent_space/wino_synth.py` (container-local; the code is the draft issue's reproducer looped over magnitudes).

Lesson checks: the MIGraphX EP lesson's NEEDED libraries are right; its RUNPATH is `/opt/rocm-7.2.4/lib:/opt/rocm-7.2.4/lib/migraphx/lib:/opt/rocm/lib` (`readelf -d`), one entry more than quoted, immaterial. The guard lesson matches this job's experience (a `grep` whose text contained the install phrase was refused).

## Ruling on the porter's Question 1 (Jeff Daily, 2026-11-23)

**B, guarded code workaround.** On ROCm PyTorch only (`torch.version.hip`), the Animals entry points default `MIOPEN_DEBUG_CONV_WINOGRAD=0` (`os.environ.setdefault`, so a user's own setting wins) before the first convolution, so the documented Animals command works in half precision. Jeff also asked for the MIOpen issue draft to be prepared (not posted without his approval) and set `moat-port` as the fork's GitHub default branch (done 2026-11-23).
## Validation 2026-11-23 (linux-gfx950, validator role, fleet job)

AMD Instinct MI355X (gfx950:sramecc+:xnack-, wave64), job GPU pin `HIP_VISIBLE_DEVICES=7`, every GPU run under `ulimit -c 0` and `timeout`. Fork `moat-port` at `00f208cdf0e7` (= `head_sha`), built in `projects/LivePortrait/src`. Verdict: **pass**.

### Environment (job-local; nothing in the image's shared paths)

- `$FLEET_VENV` (Python 3.11.17): `torch[device-gfx950]==2.16.0a0+rocm10.2.0a20261022`, `torchvision==0.30.0a0+rocm10.2.0a20261022`, `rocm[libraries,devel,device-gfx950]==10.2.0a20261022` from `https://nightly.repo.amd.com/rocm/whl-next/` (HIP 7.17.26422); then `requirements_rocm.txt` (resolved `onnxruntime-migraphx` 1.27.1, `onnx` 1.16.1, numpy 1.26.4) plus the gradio pins the porter recorded (`huggingface_hub<1`, `fastapi==0.115.2`, `pydantic==2.10.6`).
- MIGraphX runtime: AMDMIGraphX `f0239c9` built per the porter's recipe with `-DGPU_TARGETS=gfx950` (93 s at -j18). This host has no system protobuf headers, so protobuf v3.21.12 was built first into the same deps prefix (`cmake -S protobuf/cmake ... -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON`, its `bin/` first on PATH for `protoc`). Run time: `LD_LIBRARY_PATH=<migraphx-install>/lib:<venv>/_rocm_sdk_core/lib:<venv>/_rocm_sdk_libraries/lib`, `ORT_MIGRAPHX_MODEL_CACHE_PATH=<job dir>`.
- FFmpeg: the host has none. `ffmpeg` = the binary shipped in the `imageio-ffmpeg` wheel (4.2.2) on PATH; `ffprobe` built from FFmpeg `n7.1` source (`./configure --disable-x86asm --disable-doc --disable-ffplay --disable-ffmpeg --enable-ffprobe`). Without `ffprobe` the inference scripts still run (audio detection logs an error and turns audio off) but the Gradio apps die at start (`FFExecutableNotFoundError: Executable 'ffprobe' not found` in gradio's `video_is_playable`); upstream's own install note asks for both.
- Weights: `snapshot_download('KwaiVGI/LivePortrait', ignore_patterns=['*.git*','README.md','docs/*'])` into the job dir, symlinked into `pretrained_weights/` for the runs and removed afterwards.
- ORT prints about 56 `device_discovery.cc ... Failed to open file "/sys/class/drm/cardN/device/vendor"` warnings per process on this container; harmless noise, providers load.

### X-Pose op

`cd src/utils/dependencies/XPose/models/UniPose/ops && ROCM_HOME=$(rocm-sdk path --root) PATH=$ROCM_HOME/bin:$PATH PYTORCH_ROCM_ARCH=gfx950 python setup.py build install`: rc 0, deprecation warnings only, `hipcc` from the venv SDK, `--offload-arch=gfx950`; `llvm-objdump --offloading` shows `hip-amdgcn-amd-amdhsa--gfx950`. Untracked hipify output (`src/hip/`, `*_hip.cpp/h`) is the only tree change.

`python test.py` (unmodified, default slow-mode gradcheck): **9/9 True**. Forward double True (max abs 8.67e-19, rel 2.35e-16), float True (max abs 4.66e-10, rel 1.13e-07); `check_gradient_numerical` True for D = 30, 32, 64, 71, 1025, 2048 and 3096. The 288 GB card fits the dense double Jacobians that ran out of memory on the 48 GB gfx1100, so the multi-block (>1024) backward path passes the test as written here.

### Humans mode (MIGraphX EP active; 0 provider-fallback messages in every log)

| run | rc | wall |
|---|---|---|
| `python inference.py` (s0/d0, fp16), first run, MIGraphX compile cold | 0 | 161 s |
| same, warm cache | 0 | 15.5 s |
| `-s s9.jpg -d d0.mp4` | 0 | 14.9 s |
| `-s s13.mp4 -d d0.mp4` (video source) | 0 | 19.1 s |
| `-s s9.jpg -d d12.jpg` (image driving) | 0 | 13.2 s |
| `-s s9.jpg -d d0.pkl` | 0 | 14.4 s |
| `-s s9.jpg -d talking.pkl` | 1 | `KeyError: 'c_d_eyes_lst'`, upstream data (porter saw the same) |
| `--no_flag_use_half_precision` | 0 | 64.7 s (cold FP32 MIOpen find) |
| `--flag_do_torch_compile` | 0 | 133 s; one autotune candidate rejected (`permute(sparse_coo) ... input.dim() = 5 is not equal to len(dims) = 4. Ignoring this choice`), as on gfx1100 |
| `python speed.py` | 0 | 151 s; no autotune error, no hipError. Appearance 0.68 ms, Motion 1.34 ms, Warping 4.26 ms, SPADE 3.42 ms, Stitching+Retargeting 0.17 ms (100-run means) |

`speed.py` passes on gfx950 where it aborted on gfx1100; `benchmark:speed` stays `owed` in surface.json because changing that record is not the validator's call; this run is the evidence for whoever rules it.

### FP16 vs FP32 vs CPU (Humans)

ONNX forced to the CPU EP for all three modes (identical crops); 4 sources (s0, s5, s7, s9) x `d0.mp4`; generator output captured for the first 8 frames per run (32 tensors of 3x512x512), x255; decoded output video 78 frames. "dark block" = 16x16 block mean < 8 in the candidate and > 40 in the reference.

| source | GPU FP32 vs CPU FP32, generator max / mean | GPU FP16 vs CPU FP32, generator max / mean / p99.9 | video GPU FP32 vs CPU mean / max | video GPU FP16 vs CPU mean / max |
|---|---|---|---|---|
| s0 | 0.004 / 0.0001 | 1.57 / 0.056 / 0.55 | 0.066 / 27 | 0.540 / 27 |
| s5 | 0.004 / 0.0001 | 2.56 / 0.070 / 0.75 | 0.444 / 22 | 0.514 / 20 |
| s7 | 0.023 / 0.0001 | 14.75 / 0.097 / 2.98 | 0.791 / 29 | 0.925 / 34 |
| s9 | 0.020 / 0.0002 | 9.83 / 0.108 / 1.70 | 0.484 / 27 | 0.678 / 31 |

No NaN/Inf and 0 dark blocks in any tensor or video. CPU FP32 wall 70-373 s per video vs 15-20 s on the GPU.

ONNX provider parity (26 bundled `.jpg` sources, 24 with faces; MIGraphX vs CPU EP, `get_providers()` confirmed): same face counts, boxes within 7.6e-5 px, scores within 6.9e-7, 106-point within 0.0088 px, 203-point within 0.067 px; 15.3 ms vs 109 ms per image steady (MIGraphX session build + 26 images 14.2 s with the cache populated).

### Animals mode

`python inference_animals.py -s assets/examples/source/s39.jpg -d assets/examples/driving/wink.pkl --driving_multiplier 1.75 --no_flag_stitching` (57 frames), with forward hooks on all four networks reporting the first module whose inputs are finite and output is not:

| run | rc | wall | non-finite | vs FP32: generator max / mean (16 frames) | vs FP32: video mean / max / dark blocks |
|---|---|---|---|---|---|
| default FP16 | 0 | 58 s | none, no `invalid value encountered in cast` | 13.2 / 0.35 | 1.35 / 29 / 0 |
| `--no_flag_use_half_precision` | 0 | 65 s | none | -- | -- |
| FP16 + `MIOPEN_DEBUG_CONV_WINOGRAD=0` (README workaround) | 0 | 31 s | none | 13.4 / 0.44 | 1.38 / 23 / 0 |

**No black boxes on gfx950 with the default command.** `MIOPEN_LOG_LEVEL=5` over the FP16 run: MIOpen chose `ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC` (11), `ConvHipImplicitGemm3DGroupFwdXdlops` (11), `ConvWinoRageRxS<2-3>` (16), `GemmFwd1x1_0_1` (7), `GemmFwdRest` (5), `ConvDirectNaiveConvFwd` (3), `ConvHipImplicitGemmGroupFwdXdlops` (2). Winograd is used for some layers but not for the overflowing `G_middle_5.conv_1` shape, whose FP16 find-db record (`512-64-64-3x3-512-64-64-...-FP16`) lists only ConvHipConv, GemmFwdRest, ConvHipImplicitGemmGroupFwdXdlops and ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC (no FP16 Winograd records exist in the user find-db at all, so the reviewer's forced-solver experiment left nothing behind). The README text ("on some AMD GPUs (seen on a Radeon PRO W7800)") is accurate for this card too. CPU FP32 for Animals is not possible: the extension's CPU path is `AT_ERROR("Not implement on cpu")` upstream.

### Gradio

`app.py` and `app_animals.py` each serve `/` with HTTP 200 (18 s and 16 s after start) once `ffprobe` is on PATH. No request driven through the UI.

### Gates

- CUDA no-regression compile at `00f208c`: done in the review above (nvcc 12.8, torch 2.11.0+cu128, sm_80/sm_90, imports).
- `python3 utils/jargon.py --port LivePortrait`: clean. Documentation: ROCm install and op build documented beside the CUDA ones in both READMEs.
- Integrity: `git status --porcelain --untracked-files=no` empty; untracked files are hipify build output only.

### Still for a person (unchanged by this run)

Porter Question 1 (FP16 Animals workaround in code or docs; gfx950 needs none), the fork default branch (`moat-port`), `benchmark:speed` owed in surface.json (passes here), the draft MIOpen issue (add the gfx950 forced-solver rows from the review above), and Windows last.

### Addendum: Jeff's ruling B arrived during this job (linux-gfx950, 2026-11-23)

The ruling above (option B, guarded `MIOPEN_DEBUG_CONV_WINOGRAD=0` default in the Animals entry points) merged into `port/LivePortrait` while this job was validating. The review and the gfx950 validation describe `00f208c` (documentation only), as dispatched; this job did not port. Implementing B moves the head, so the change goes back through a porter, a delta review, and revalidation on each platform (a code delta in `inference_animals.py` / `app_animals.py`, not inert).

Evidence for B from this card: what the workaround costs where it is not needed. Warm caches, 3 alternating reps each of `python inference_animals.py -s assets/examples/source/s39.jpg -d assets/examples/driving/wink.pkl --driving_multiplier 1.75 --no_flag_stitching` (FP16): default 15.5 / 15.9 / 16.4 s, with `MIOPEN_DEBUG_CONV_WINOGRAD=0` 15.6 / 15.5 / 15.3 s. No measurable cost end to end, though MIOpen picks `ConvWinoRageRxS<2-3>` for 16 conv configurations here by default; output vs FP32 is the same size either way (table above). For the porter: the FP16 `G_middle_5.conv_1` shape does not use Winograd on gfx950, so a gfx950 revalidation of B cannot show the fix working, only that it costs nothing; the wave32 gfx1100 run is the one that proves it.

## Porting 2026-11-23 (linux-gfx942, porter role, fleet job): Jeff's ruling B

New commit `874215c` on `moat-port` (on top of the reviewed and gfx950-validated `00f208c`; no amend, plain push), `advance-head` recorded. Code: `inference_animals.py` and `app_animals.py` import `torch` and, after their imports and before any module-level code (so before anything can run a convolution; none of the imported `src` modules runs one at import time), do `if torch.version.hip is not None: os.environ.setdefault("MIOPEN_DEBUG_CONV_WINOGRAD", "0")` with the comment "Avoids an FP16 overflow in MIOpen's Winograd convolution; remove once MIOpen fixes it." The `torch.version.hip is not None` spelling matches the provider helper in `human_landmark_runner.py`. CUDA and CPU torch (`torch.version.hip` is None) never touch the environment; Humans entry points (`inference.py`, `app.py`) unchanged. README (both languages): the black-box paragraph now says that with ROCm PyTorch the two Animals entry points default the variable to `0` and that a user's own value takes precedence, instead of telling users to set it. `jargon.py --port LivePortrait`: clean (branch and commit message).

Porter check on AMD Instinct MI300X (gfx942, `HIP_VISIBLE_DEVICES=1`), TheRock 10.2.0a20261022 torch in `$FLEET_VENV`, X-Pose op built with `PYTORCH_ROCM_ARCH=gfx942` (rc 0, deprecation warnings only), ONNX on the CPU EP (MIGraphX was still building): `MIOPEN_LOG_LEVEL=5 python inference_animals.py -s assets/examples/source/s39.jpg -d assets/examples/driving/wink.pkl --driving_multiplier 1.75 --no_flag_stitching` rc 0; "FW Chosen Algorithm" lines: ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC 16, ConvHipImplicitGemm3DGroupFwdXdlops 14, ConvHipImplicitGemmGroupFwdXdlops 13, GemmFwd1x1_0_1 7, ConvDirectNaiveConvFwd 4, GemmFwdRest 1 -- no Winograd, no "invalid value encountered in cast". Same command with the user setting `MIOPEN_DEBUG_CONV_WINOGRAD=1`: `ConvWinoRageRxS<2-3>` chosen 19 times and `parse_output` warns "invalid value encountered in cast" once, so the user's value wins, and gfx942 appears to hit the overflow with Winograd allowed (measured in validation below).
