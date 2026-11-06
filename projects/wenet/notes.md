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
