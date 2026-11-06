# FunASR notes

## Intake screen (2026-11-05, windows-gfx1151 host, gap-matrix seed)

Source of lead: third-party "AMD AI Software Gap Matrix" seed (verdict: partial, upstream). Screened at upstream main, shallow clone 2026-11-05.

### Licence
- `licenses.py check modelscope/FunASR`: MIT, tier 1 (LICENSE file read; GitHub field agrees). Model weights carry a separate MODEL_LICENSE, which is about weight use, not code contribution.
- `licenses.py scan-nvidia agent_space/FunASR-screen`: no NVIDIA proprietary licence text. No submodules in the clone.

### CUDA surface
- Only CUDA source: `funasr/models/rwkv_bat/cuda_encoder/wkv_cuda.cu` and `cuda_decoder/wkv_cuda.cu` (4.8 KB each, near-identical RWKV WKV kernel plus `wkv_op.cpp`). JIT-built with `torch.utils.cpp_extension.load` in `funasr/models/rwkv_bat/rwkv_attention.py`, only for the RWKV-BAT / conformer_rwkv models; optional, not on the main Paraformer/SenseVoice/Fun-ASR-Nano paths.
- That loader passes nvcc-only flags (`-res-usage`, `--maxrregcount 60`, `-Xptxas -O3`, `--use_fast_math`) that hipcc will reject, so the RWKV path fails on ROCm torch as written. Kernel itself is a per-channel serial loop with `blockDim = min(C, 32)`, no warp intrinsics: a flag guard is likely the whole fix.
- Everything else is plain PyTorch (51 `torch.cuda`/`.cuda()` sites in `funasr/`), so ROCm PyTorch should run the main toolkit; unverified.

### GPU-specific dependencies (none are MOAT projects; no set-deps)
- No hard dependency on k2, warp-rnnt/warp-transducer, CTC decoders or flash-attn found in setup.py.
- `runtime/python/onnxruntime` and `runtime/python/utils/runtime_eval.py` use `onnxruntime-gpu` / `CUDAExecutionProvider`: ONNX Runtime has no ROCm EP any more, so this deployment path is CPU-only on AMD (not an AMD path; doc it as such).
- `runtime/triton_gpu`: Triton Inference Server + TensorRT deployment, NVIDIA-only; out of scope.
- Fun-ASR-Nano vLLM path (`funasr/auto/auto_model_vllm.py`, `funasr/models/fun_asr_nano/inference_vllm*.py`): vLLM has ROCm support, so this is a validate-only item.
- `runtime/llama.cpp`: upstream already ships ggml Vulkan builds for AMD (README "AMD Windows troubleshooting"); ggml also has a HIP backend, not used by their packaging.

### Existing AMD effort
- No ROCm/HIP PR or branch upstream. AMD-related upstream work is all Vulkan/llama.cpp runtime (issue #3479 AMD Windows Vulkan crash, open; branches `codex/diagnose-amd-vulkan-3479`, `codex/runtime-amd-boundaries-3479`; PR #3482 ggml AMD submission fix).
- Issue #2972 (Hygon DCU, a ROCm-derived stack) closed: maintainer said no validated DCU build; PyTorch path "may work", vLLM path unverified. Shows maintainer interest in non-CUDA accelerators.
- No ROCm/ or AMD-Ecosystem repo for FunASR. Web search: no FunASR-on-ROCm write-up (rocm.blogs.amd.com speech-model blog covers other ASR models). amd / AMD-AGI orgs not searchable with this token (SAML SSO), so that part of the duplicate check stays open.

### Recommendation: fork (validate lane)
Work: install on ROCm PyTorch, run the GPU tests and an AutoModel Paraformer/SenseVoice inference, make the RWKV WKV JIT loader pass hipcc-compatible flags under `torch.version.hip`, validate the vLLM Nano path on ROCm vLLM, and send upstream a docs change (ROCm install line; state the ONNX-GPU and Triton/TensorRT paths are NVIDIA-only). Upstream is very active (pushed 2026-11-05, 21k stars) and responsive to AMD reports.
