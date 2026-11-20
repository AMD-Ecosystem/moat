# LivePortrait port plan

Planner run 2026-11-20 on a linux-gfx950 host (no GPU used, nothing built). Upstream analysed at `9b294b3d0536` (`main`, 2026-06-02).

**Scope is not settled.** Whether Animals mode (the vendored X-Pose MSDeformAttn extension) is in this port waits on Jeff's ruling; see notes.md, "Planning (2026-11-20)", Question 1. This plan describes the full port (option A there) and marks which steps fall away under the other options. A porter must not start until that question is answered.

## Project

- Name: LivePortrait
- Upstream: https://github.com/KlingAIResearch/LivePortrait, default branch `main`
- Fork: https://github.com/AMD-Ecosystem/LivePortrait, port branch `moat-port`
- Licence: MIT (intake); vendored X-Pose subtree is mixed (Apache-2.0 file headers, IDEA non-commercial repository licence), which is what Question 1 is about.

## Existing AMD support

- Upstream: none. Issues #110 ("AMD GPU support?") and #314 (DirectML/AMD) are open with no work; no PRs mention HIP/ROCm/AMD.
- https://github.com/thebestryan/LivePortrait-AMD: community, not authoritative (1 star, 3 commits). Humans mode only, pins torch 2.3.1+ROCm 5.7 and prefers `ROCMExecutionProvider`, which ONNX Runtime has since removed. A hint (confirms Humans mode needs no code change beyond ONNX provider selection), not a base.
- Decision: port from upstream. Upstream accepts feature PRs in principle (the last code PR merged 2025-02) but is low-activity; an upstream PR is still the vehicle. It does not link platform forks.
- Performance: no NVIDIA-tuned kernels. The heavy model is plain PyTorch; the one compiled op is a scalar bilinear-gather kernel. Mechanical port; no AMD-native rewrite is warranted.

## Build classification

Two parts, neither a CMake project:

1. **Humans mode (default) and the Gradio apps: pure Python on PyTorch.** No compiled code. `inference.py`, `app.py`, `src/live_portrait_pipeline.py`, `src/live_portrait_wrapper.py` select `'cuda:' + device_id` (line 39/41, 354/356), which is the ROCm device on ROCm PyTorch. Face detection (InsightFace `buffalo_l`) and the 203-point landmark model run through ONNX Runtime with `CUDAExecutionProvider` hard-coded (`src/utils/cropper.py:59,62`, `src/utils/human_landmark_runner.py:36-40`).
2. **Animals mode: a PyTorch extension (torch-extension, Strategy B).** `src/utils/dependencies/XPose/models/UniPose/ops/setup.py` builds `MultiScaleDeformableAttention` with `torch.utils.cpp_extension.CUDAExtension` from `src/vision.cpp`, `src/cpu/ms_deform_attn_cpu.cpp` and `src/cuda/ms_deform_attn_cuda.cu`. Used only by `inference_animals.py` / `app_animals.py` through `src/utils/animal_landmark_runner.py` (X-Pose `UniPose` -> `deformable_transformer.py:30`, `transformer_deformable.py:22` -> `ops/modules/ms_deform_attn*.py` -> `ops/functions/ms_deform_attn_func.py`).

`ext_type` recorded as `torch-extension`.

## Port strategy

- Humans mode: no kernel work. Make the ONNX Runtime provider selection AMD-aware and give ROCm users an install path (requirements file plus README section). The heavy generator, warping and stitching networks stay on the GPU through PyTorch unchanged.
- ONNX Runtime on AMD: ONNX Runtime's ROCm EP is gone. The ROCm equivalent is the MIGraphX EP (`onnxruntime-migraphx`, PyPI, currently 1.27.1, manylinux x86_64 only; no Windows wheel exists on PyPI, checked 2026-11-20). So: Linux prefers `MIGraphXExecutionProvider` when `onnxruntime.get_available_providers()` lists it, else CPU; Windows uses `CPUExecutionProvider`, named in the README as a CPU fallback because no ROCm EP exists there. NVIDIA keeps `CUDAExecutionProvider` exactly as today. Only detection and landmarks (per frame, cropping stage) run in ONNX Runtime, so the CPU fallback is a cropping-speed cost, not a correctness one.
- Animals mode (option A only): Strategy B. Let torch's build-time hipify translate the `.cu`; change only the `setup.py` gate that refuses to build without `CUDA_HOME`. Fix whatever hipify cannot (expected: nothing, see risks).

## CUDA surface inventory

`python3 utils/surface.py generate LivePortrait` floor: 2 CUDA files, 7 kernels, 16 launches, no CUDA libraries.

| file | content | HIP mapping / risk |
|---|---|---|
| `ops/src/cuda/ms_deform_im2col_cuda.cuh` (1326 lines) | 1 forward kernel `ms_deformable_im2col_gpu_kernel`; 6 backward `col2im` kernels (`shm_blocksize_aware_reduce_v1/v2<blockSize>`, `shm_reduce_v1/v2`, `shm_reduce_v2_multi_blocks`, `gm`); grid-stride `CUDA_KERNEL_LOOP`; `CUDA_NUM_THREADS = 1024`; static and `extern __shared__` arrays; tree reductions with `__syncthreads` per step; float/double `atomicAdd` | hipify direct. No warp intrinsics, no `warpSize`, no literal-32 lane logic, no textures. Reductions are barrier-separated (no warp-synchronous tail), so wave64 and wave32 behave the same. Blocks of `channels` threads (30, 71, ...) are legal on HIP. Max static LDS: double, blockSize 1024 -> 1024*3*8 = 24 KB, under 64 KB. `#include <THC/THCAtomics.cuh>`: torch's hipify maps it to `THH/THHAtomics.cuh`, which ROCm torch installs (present in the host's torch 2.16 ROCm build, checked); it is a one-line forwarder to `ATen/cuda/Atomic.cuh`. |
| `ops/src/cuda/ms_deform_attn_cuda.cu` (153 lines) | host wrappers; `AT_DISPATCH_FLOATING_TYPES` (float, double; no half); `at::cuda::getCurrentCUDAStream()`; `#include <cuda.h>`; deprecated `Tensor::type().is_cuda()` and `Tensor::data<T>()` | hipify maps the stream and header. `data<T>()` and `type()` are deprecated but still compile on current torch (TensorBody.h, deprecation warning); not a ROCm issue, leave unless the build fails. |
| `ops/src/ms_deform_attn.h`, `vision.cpp`, `cpu/*` | dispatch on `value.type().is_cuda()` under `WITH_CUDA`; CPU path is `AT_ERROR("Not implement on cpu")` | `WITH_CUDA` stays defined by setup.py on ROCm; nothing to change. |

Python CUDA references outside the extension (all work on ROCm PyTorch unchanged): `torch.backends.cudnn.benchmark` (MIOpen), `torch.autocast(device_type='cuda')`, `torch.compile(mode='max-autotune')` with `torch.compiler.cudagraph_mark_step_begin()` (`live_portrait_wrapper.py:65,286`), `speed.py` `torch.cuda.synchronize`, X-Pose `mask_generate.py:8` `.to('cuda')`, `util/misc.py` distributed helpers (training-only, not reached in inference).

Not CUDA despite the name: `insightface/utils/filesystem.py:146` imports `nvidia.dali` only in an unused data-loading helper.

## Risk list

1. **ONNX Runtime package conflict.** `requirements.txt` pins `onnxruntime-gpu==1.18.0`; installing it beside `onnxruntime-migraphx` or `onnxruntime` leaves two packages owning the `onnxruntime` module. The ROCm requirements file must list exactly one ONNX Runtime package. Also `onnx==1.16.1`/`numpy==1.26.4` pins may not resolve against the ROCm torch and `onnxruntime-migraphx` wheels for the Python the ROCm torch wheels target; the porter records the resolved set.
2. **MIGraphX EP runtime.** The PyPI wheel needs a matching MIGraphX runtime (`libmigraphx`) from the ROCm install; first-session compile time for each ONNX model can be seconds to minutes (MIGraphX compiles per input shape). RetinaFace runs at a fixed 512x512 detection size and the landmark model at 224x224, so shapes are static; verify the compile cost is paid once and that outputs match CPU EP (bounding boxes and landmarks within a small pixel tolerance). If MIGraphX fails to load or mismatches, CPU is the fallback and the reason goes in notes.
3. **InsightFace provider plumbing.** `FaceAnalysisDIY` passes `providers` to `insightface/model_zoo` which calls `set_providers(['CPUExecutionProvider'])` when `ctx_id < 0` (`retinaface.py:132`, etc.). `prepare(ctx_id=device_id)` keeps the passed provider list; check the MIGraphX provider survives and `provider_options` are not CUDA-specific.
4. **FP16 on RDNA / CDNA.** Humans mode runs `torch.autocast(float16)` by default; the config comments warn about "black boxes" on some GPUs. Validate fp16 and `--no_flag_use_half_precision` output on wave64 and wave32 (no NaN/black regions; compare fp16 vs fp32 frames).
5. **`torch.compile` (`--flag_do_torch_compile`).** max-autotune with CUDA graphs on ROCm uses Triton/inductor; Linux only (upstream documents it unsupported on Windows). Run it once per Linux arch as an optional path; a failure there is a ROCm inductor issue to record, not a port blocker.
6. **Animals mode, extension build (option A).** `setup.py:38` requires `torch.cuda.is_available() and CUDA_HOME is not None`; on ROCm torch `CUDA_HOME` is normally `None` and a build host may have no visible GPU. Gate on `CUDA_HOME is not None or ROCM_HOME is not None` (import `ROCM_HOME` from `torch.utils.cpp_extension`) without requiring a visible GPU on the ROCm branch, keeping the CUDA condition byte-equivalent. The nvcc `-D__CUDA_NO_HALF_*` flags reach hipcc through BuildExtension; harmless (no half code), keep. Windows: `hipcc.exe`/setuptools traps in `cuda-to-rocm` `windows.md` apply.
7. **Animals mode, fp16 autocast into a float/double-only op.** `animal_landmark_runner.py:84` wraps X-Pose in fp16 autocast and the extension dispatches only float/double. Upstream behaviour is the same on CUDA (`deformable_transformer.py:1052` disables autocast only around the FFN, not the attention, so whether half reaches the op depends on autocast's casting policy for the custom Function); if Animals mode fails with "not implemented for 'Half'" on ROCm, compare against the CUDA path before changing anything.
8. **Animals mode, missing extension today.** `ms_deform_attn_key_aware.py` swallows the import failure with a warning, then `NameError`s at the first call. Not changed by this plan (upstream behaviour); a pure-PyTorch fallback is option C in Question 1.
9. **No upstream test suite.** Validation relies on the inference scripts and the op's own `ops/test.py`.
10. **Model downloads.** `pretrained_weights/` is fetched from HuggingFace (`KwaiVGI/LivePortrait`, `huggingface-cli download ... --local-dir pretrained_weights`); InsightFace `buffalo_l` is non-commercial (fine for validation, nothing redistributed). X-Pose weights for Animals mode come from the same HF repo (`liveportrait_animals/`, `xpose.pth`).

## File-by-file change list

Humans mode (all options):

- `src/utils/cropper.py` (around lines 46-62): replace the hard-coded `["CUDAExecutionProvider"]` with a provider chosen from `onnxruntime.get_available_providers()`: CUDA if present (unchanged NVIDIA behaviour), else MIGraphX, else CPU. Keep mps and `flag_force_cpu` branches as they are. Put the selection in one small helper (e.g. in `src/utils/helper.py` or a new `src/utils/onnx_providers.py`) shared with the next file.
- `src/utils/human_landmark_runner.py` (lines 36-54): same selection for the `'cuda'` branch; CUDA keeps `('CUDAExecutionProvider', {'device_id': device_id})`, MIGraphX gets `{'device_id': device_id}`, CPU keeps the existing `intra_op_num_threads = 4` session options.
- `requirements_rocm.txt` (new, beside `requirements_macOS.txt`): `-r requirements_base.txt` plus `onnxruntime-migraphx` on Linux and `onnxruntime` on Windows via environment markers (`; sys_platform == "linux"` / `"win32"`), torch installed separately from the ROCm index as the README already does for CUDA. Exact pins chosen by the porter from what resolves; record them.
- `readme.md` (Environment Setup, next to the CUDA and macOS subsections): an "AMD GPUs (ROCm)" install subsection for Linux and Windows: install ROCm PyTorch, `pip install -r requirements_rocm.txt`, the run commands unchanged (no device-selection env vars in the commands). Name the Windows ONNX CPU fallback. `readme_zh_cn.md`: matching short section only if upstream keeps the two in sync for install steps (check at port time; the porter may leave the Chinese README alone and say so).
- `speed.py`: no change (`torch.cuda.synchronize` is the ROCm sync on ROCm PyTorch).

Animals mode (option A only):

- `src/utils/dependencies/XPose/models/UniPose/ops/setup.py` (lines 14-48): import `ROCM_HOME`; extend the gate so a ROCm torch (`torch.version.hip`) with `ROCM_HOME` builds `CUDAExtension`; leave the CUDA condition and error message for neither case as they are.
- `ms_deform_im2col_cuda.cuh` / `ms_deform_attn_cuda.cu`: no change expected. If the hipified build fails on `THC/THCAtomics.cuh`, switch the include to `<ATen/cuda/Atomic.cuh>` (shared fix, no `[ROCm]` prefix: it is what the THC header forwards to on CUDA as well).
- `readme.md` "Fast hands-on (animals)": one line that the same `python setup.py build install` works with ROCm PyTorch on Linux (and Windows if validated).

No change anywhere else: no compat header (Strategy B), no edits to upstream structure.

## Build commands

Humans mode, Linux (porter and validator; Python per ROCm torch wheel support):

```
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm<ver>   # or the host's ROCm torch
pip install -r requirements_rocm.txt
huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"
```

Windows: TheRock ROCm torch wheel for the card (`[device-<gfx>]` extra, `cuda-to-rocm` `windows.md`), then the same `pip install -r requirements_rocm.txt`.

Animals mode extension (option A), from the repository root:

```
cd src/utils/dependencies/XPose/models/UniPose/ops
python setup.py build install
```

On a fleet worker use the job venv (`$FLEET_VENV`), not the image's Python; build output stays in the fork clone.

## Test plan

No upstream test suite exists. Validation evidence per platform (wave64, wave32, Windows):

1. Environment check: `python -c "import torch, onnxruntime; print(torch.version.hip, torch.cuda.get_device_name(0), onnxruntime.get_available_providers())"`.
2. Humans, image-driven and video-driven, fp16 default:
   `python inference.py` (s6.jpg + d0.mp4 defaults), `python inference.py -s assets/examples/source/s9.jpg -d assets/examples/driving/d0.mp4`, `python inference.py -s assets/examples/source/s13.mp4 -d assets/examples/driving/d0.mp4` (video source), `python inference.py -s assets/examples/source/s9.jpg -d assets/examples/driving/d12.jpg` (image driving), and a `.pkl` driving template (`-d assets/examples/driving/talking.pkl`).
   Each must finish, write `animations/*.mp4` and `*_concat.mp4`, and log which ONNX provider ran (no CUDA-EP fallback warning on AMD). Record wall-clock per run.
3. Humans fp32: the first command with `--no_flag_use_half_precision`; compare a few output frames with the fp16 run (mean abs pixel difference recorded; no black/NaN regions in either).
4. ONNX provider parity (Linux): run the cropper's detection + landmarks on 3 source images with MIGraphX EP and with CPU EP; boxes and 203-point landmarks agree within ~1 px. Record the numbers.
5. Optional on Linux: `python inference.py --flag_do_torch_compile` (once; record success or the inductor error).
6. Gradio smoke: `python app.py --server_port <p>` starts and serves one request through the Gradio client or `curl` of the root page; not a gate if the host has no browser path.
7. Animals (option A):
   - op test on GPU: `cd src/utils/dependencies/XPose/models/UniPose/ops && python test.py` -- forward double and float vs `ms_deform_attn_core_pytorch` must print `True`; `gradcheck` for channels 30, 32, 64, 71, 1025, 2048, 3096 (exercises every backward kernel, including blockSize templates and the >1024 multi-block path) must print `True`.
   - end to end: `python inference_animals.py -s assets/examples/source/s39.jpg -d assets/examples/driving/wink.pkl --driving_multiplier 1.75 --no_flag_stitching` writes its outputs.
8. Non-GPU regression: `flag_force_cpu` cropping path still selects CPU EP; macOS branch untouched (code review only, no Mac host). CUDA path: no code-path change for NVIDIA in the provider helper (review), and for option A compile the extension with real nvcc against a CUDA torch (per the CUDA no-regression rule), or record `cuda-not-validated: <error>` if the host's torch is ROCm-only.

## Open questions

See notes.md, "Planning (2026-11-20)": Question 1 (Animals mode scope) must be ruled by Jeff before porting.
