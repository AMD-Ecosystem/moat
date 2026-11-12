# CosyVoice notes

## Intake screen (2026-11-12)

Validation project (lane `validate`, seed `amd-gap-matrix`), taken up per jeff's 2026-11-12 ruling on the third-party "AMD AI software gap" list.

- Upstream: the GitHub API resolves `FunAudioLLM/CosyVoice` to the canonical `QwenAudio/CosyVoice` (https://github.com/QwenAudio/CosyVoice), default branch `main`, pushed 2026-05-25, not archived, 24k stars. Project name `CosyVoice`.
- Licence: Apache-2.0 (tier 1). Submodule `third_party/Matcha-TTS` (shivammehta25/Matcha-TTS) is MIT. `utils/licenses.py scan-nvidia` on depth-1 clones of both: no NVIDIA proprietary licence text. Pretrained weights are downloaded from ModelScope / Hugging Face (`Fun-CosyVoice3-0.5B-2512`, `CosyVoice2-0.5B`, `CosyVoice-300M*`); not part of the tree.
- CUDA surface: none of its own (no `.cu`). PyTorch inference (`cosyvoice/cli`), `torch.cuda` streams and AMP, optional `load_jit` (TorchScript), `load_trt` (TensorRT plan for the flow-decoder estimator), `load_vllm` (vLLM for the CosyVoice2/3 LLM), onnxruntime for the speech tokenizer (`CUDAExecutionProvider` in `cosyvoice/cli/frontend.py`) and campplus (CPU), DeepSpeed for training only. `runtime/triton_trtllm` is an NVIDIA-contributed Triton + TensorRT-LLM serving path.
- `requirements.txt` pins the cu121 wheel index, torch 2.3.1, `onnxruntime-gpu==1.18.0` and `tensorrt-cu12==10.13.3.9` on Linux, `deepspeed==0.15.1` on Linux; CPU onnxruntime on win32/darwin.

## Duplicate effort

- Upstream PR #1754 (open, CONFLICTING, no activity since 2025-12-28), https://github.com/QwenAudio/CosyVoice/pull/1754 by alexhegit (AMD): adds `requirements-rocm.txt` (drops the cu121 index, tensorrt and keeps `onnxruntime-gpu`), README "Install with AMD ROCm" (torch rocm6.4 wheels) and a webui model-name fix. Earlier closed PR #1280 by the same author added AMD README steps. This is the AMD community effort to back: refresh it on current `main` rather than file a competing one.
- No fork under ROCm or AMD-Ecosystem (GitHub search 2026-11-12).

## Dependencies

- No MOAT hard dependency. Optional: vLLM (ROCm builds exist upstream), onnxruntime (ROCm/MIGraphX execution providers), TensorRT (no ROCm equivalent).

## Validation scope

- **Delivery route:** upstream PR of fixes (Apache-2.0, non-NVIDIA upstream, large user base). Most likely shape: a ROCm requirements file or extras plus README section (i.e. finish #1754), plus any runtime guards found by validation (TensorRT and `CUDAExecutionProvider` selection must degrade cleanly on ROCm). Upstream activity has slowed (last push 2026-05), so the fork may be the de facto deliverable if PRs sit.
- **GPU gate:** no unit test suite upstream. The gate is `example.py` (CosyVoice-300M SFT / zero-shot / cross-lingual / VC / instruct, CosyVoice2-0.5B, Fun-CosyVoice3-0.5B) on GPU with `fp16` and `load_jit` on, compared against CPU or CUDA output (audio length, speaker similarity / WER as in upstream's eval, or at least non-silent, non-NaN audio); plus `vllm_example.py` with ROCm vLLM on Instinct; plus the `webui.py` smoke start. Training (`examples/libritts`, DeepSpeed) is optional evidence.
- **CUDA-only dependencies:** TensorRT (`load_trt`, `runtime/triton_trtllm`): no ROCm substitute in-tree; gap, keep it disabled on ROCm (MIGraphX would be new work, out of scope). `onnxruntime-gpu` (CUDA EP): substitute `onnxruntime-rocm` / MIGraphX EP or the CPU EP the code already falls back to; needs a provider-selection fix. `deepspeed` 0.15.1: ROCm DeepSpeed builds exist (training only). vLLM: ROCm vLLM exists; version window is 0.9.0 or 0.11+.
- **Windows:** plausible. Requirements already special-case win32 (CPU onnxruntime, no deepspeed / tensorrt), so inference with ROCm torch wheels on Windows is in reach; vLLM and TensorRT paths are Linux-only and excluded. Recommend keeping the `windows` gate (no waiver); the 0.5B models fit gfx1151 easily.
- **Risks:** stale pins (torch 2.3.1, numpy 1.26, onnxruntime 1.18) against current ROCm wheels; `torch.cuda.Stream` / AMP behaviour under HIP; `openai-whisper` and `pyworld` builds on Windows.
