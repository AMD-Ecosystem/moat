# wenet notes

## Intake screen (2026-11-05, windows-gfx1151 host, gap-matrix seed)

Source of lead: third-party "AMD AI Software Gap Matrix" seed (verdict: unknown, AMD path unknown). Screened at upstream main, shallow clone 2026-11-05. Upstream active (pushed 2026-10-26), not archived.

### Licence
- `licenses.py check wenet-e2e/wenet`: Apache-2.0, tier 1 (LICENSE read; GitHub field agrees).
- `licenses.py scan-nvidia agent_space/wenet-screen`: no NVIDIA proprietary licence text. The one `.cu` carries "Copyright (c) 2023, NVIDIA CORPORATION" under Apache-2.0, which is clean. No submodules.

### CUDA surface
- Only CUDA source: `runtime/gpu/tensorrt/LayerNormPlugin/LayerNormPlugin.cu` (4.5 KB), a TensorRT `nvinfer1` plugin for the TensorRT/Triton deployment path. TensorRT has no AMD equivalent API (MIGraphX is a different model), so this file is out of scope, not a port target.
- The training/inference package `wenet/` is plain PyTorch (`--device cuda` style; 12 `torch.cuda` sites). No compiled extension.

### GPU-specific dependencies and deployment paths
- k2 (MOAT project, ROCm support merged upstream via k2-fsa/k2#1353): OPTIONAL. Guarded imports in `wenet/models/k2/model.py` (LF-MMI / HLG decoding) and `wenet/models/transducer/transducer.py` (pruned RNNT loss, `enable_k2` default False; torchaudio RNNT loss is the default). Not a hard dependency, so no set-deps; a good place to exercise the k2 port.
- deepspeed is in requirements.txt (train engine option): upstream DeepSpeed supports ROCm; ROCm/DeepSpeed also exists.
- `runtime/gpu/` (Triton Inference Server): `tensorrt/`, `tensorrt_fastertransformer/` (NVIDIA-only), `cuda_decoders/` uses `riva-asrlib-decoder` (NVIDIA Riva CUDA WFST decoder, NVIDIA-only), and ONNX models via onnxruntime-gpu. `wenet/bin/export_onnx_gpu.py` / `recognize_onnx_gpu.py` need onnxruntime-gpu: ONNX Runtime has no ROCm EP any more, so none of these is an AMD path. Docs should say so.
- `runtime/libtorch` C++ runtime: `-DGPU=ON` downloads a CUDA libtorch and defines `USE_GPU`; a ROCm libtorch option is a possible small CMake + docs change (needs checking that the C++ code only uses `torch::kCUDA`, which maps to HIP in ROCm libtorch).

### Existing AMD effort
- No ROCm/HIP/AMD PR, branch or issue upstream (search hits #1305 "add GPU support to runtime/LibTorch", CUDA, and #2003, unrelated). Upstream has vendor runtimes for Kunlun, Horizon BPU, IPEX, OpenVINO, but none for AMD.
- No ROCm/ or AMD-Ecosystem repo. Web search: nothing for wenet on ROCm. amd / AMD-AGI orgs not searchable with this token (SAML SSO); that part of the duplicate check stays open.

### Recommendation: fork (validate lane, small)
Work: run wenet training/recognition and its GPU tests on ROCm PyTorch (torchaudio RNNT default, k2 options against the merged k2 ROCm support, deepspeed engine), add a ROCm libtorch option to `runtime/libtorch` if the C++ is device-agnostic, and send a docs change naming the ROCm path and marking the TensorRT/Triton/Riva/onnxruntime-gpu paths NVIDIA-only. No CUDA kernel to port. If a person judges the value too low for a pure-PyTorch toolkit with no kernel work, the decline reason would be `not-a-target`.

## Planning (2026-11-20, linux-gfx942 fleet job, planner; no GPU used, nothing built)

**Decisions needed from jeff before a porter starts** (full options and reasoning in `plan.md`, Open questions):

- **Q1 -- scope out the NVIDIA-only deployment layer?** `runtime/gpu/` (Triton, TensorRT incl. the only `.cu` `LayerNormPlugin.cu`, FasterTransformer, Riva CUDA WFST decoder) and the onnxruntime-gpu scripts `wenet/bin/export_onnx_gpu.py` / `recognize_onnx_gpu.py`. (a) recommended: scope all out, docs mark them NVIDIA-only; (b) (a) plus make the two onnx scripts select `MIGraphXExecutionProvider` on AMD (onnxruntime removed its ROCm provider); (c) port the Triton deployment to an onnxruntime/MIGraphX stack (large, not recommended).
- **Q2 -- add a ROCm build of the LibTorch C++ runtime's `GPU` option?** (a) recommended: yes, Linux only like upstream, by letting `runtime/core/cmake/libtorch.cmake` use a local LibTorch (e.g. from the ROCm torch wheel) instead of fetching the pinned cu118 2.2.0 zip; (b) no, docs for the Python package only; (c) (a) plus Windows GPU runtime (not recommended).
- **Q3 -- send an upstream PR when the Python package needs no source change?** (a) recommended: yes, ROCm install docs (+ the Q2 CMake option) backed by three-platform GPU validation, same shape as upstream's NPU support; (b) validate, ship only if a run-time fix is needed; (c) decline as `not-a-target` (a person's call).

Nothing in `surface.json` that depends on Q1/Q2 has been marked covered or scoped out; the porter records those once the rulings exist.

### Findings
- Upstream read at `be6ad97fdb51` (main, 2026-10-26), shallow clone in `projects/wenet/src`. `ext_type` set to `python-torch`.
- Python package: `setup.py` has no extensions; all GPU use is `torch.cuda.*`, `--device cuda`, `nccl`, `torch.cuda.amp`, FSDP, DeepSpeed, `F.scaled_dot_product_attention` -- all served by ROCm PyTorch unchanged. No Python source change expected.
- LibTorch runtime: `runtime/core/decoder/torch_asr_model.cc` uses only `torch::cuda::is_available()` / `at::kCUDA` under `USE_GPU`; the gap is purely `libtorch.cmake` downloading CUDA LibTorch 2.2.0 (`cu118`) and refusing `GPU` off Linux.
- Upstream tests: 17 CPU pytest files under `test/` (CI: `unit_test.yml`, `pytest -q`), runtime ctest targets; no GPU test exists, so GPU evidence comes from a validator harness (plan.md Test plan items 1-7) on `test/resources/dataset`.
- Environment risk, not ROCm: torchaudio 2.9+ (maintenance mode) removed APIs deprecated in 2.8 and routes `load`/`save` through TorchCodec; RNNT loss and `forced_align` were on the keep-or-drop list. wenet uses `torchaudio.info`, `sox_effects`, `rnnt_loss`, `forced_align`. The newest ROCm wheels are torch/torchaudio 2.11 (`rocm7.1`). The fleet host's shared torch is a 2.16 dev build with no torchaudio, so validation needs its own venv. Plan rule: check the APIs first; fall back to the newest ROCm pair that still has them and register the break as a deferral rather than fixing it in the ROCm PR.
- k2 ROCm support is merged upstream (k2-fsa/k2#1353, 2026-07-02); optional for wenet, so no `depends_on`.
- Sources checked: torchaudio 2.9 notes (https://newreleases.io/project/github/pytorch/audio/release/v2.9.0, https://github.com/pytorch/audio/issues/3902); onnxruntime ROCm provider removal and MIGraphX provider (https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html, https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/onnxruntime.html).
