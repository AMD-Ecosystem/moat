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

## Planning (2026-11-20, linux-gfx942, planner)

**Decisions needed from Jeff before a porter starts** (full options and reasoning in plan.md "Open questions"):

- **Q1 Training layer** (`cosyvoice/bin/train.py`, `examples/libritts`, `examples/magicdata-read`, `examples/grpo`). (a) inference-only, training out of scope; (b) add a short single-GPU training smoke for `examples/libritts/cosyvoice2`, grpo out; (c) both recipes in. Recommended: (b).
- **Q2 TensorRT layers** (`load_trt` flow estimator, `runtime/triton_trtllm`, NVIDIA Dockerfiles). (a) guard `load_trt` on HIP and document these as NVIDIA-only, PyTorch estimator is the ROCm path; (b) also build a MIGraphX-compiled estimator as the ROCm counterpart of `load_trt`; (c) also a ROCm serving stack replacing triton_trtllm. Recommended: (a), with (b) as a deferral if wanted.
- **Q3 vLLM backend.** (a) in scope on Linux (wave64, and wave32 where ROCm vLLM supports it), Windows excluded since vLLM has no Windows build; (b) out. Recommended: (a).
- **Q4 Delivery vehicle** (decided at PR time; the plan keeps both (a) and (b) open by reusing #1754's file layout). (a) our own upstream PR crediting #1754; (b) offer the validated change to alexhegit to refresh #1754; (c) fork as deliverable, nothing sent. Context: a maintainer wrote in issue #839 "we use pytorch and tensorrt, we don't do any hardware support"; #1754 has sat CONFLICTING since 2025-12-28, its only conflict being an obsolete `webui.py` hunk (upstream already changed that default to `CosyVoice2-0.5B`).

Nothing is recorded `scoped_out` in surface.json yet; the proposed cuts carry "pending Jeff" notes. After the ruling, move them to `scoped_out` (reason = the ruling) or `owed`.

Findings that resume intake's screen (not redone):

- Upstream `main` at `074ca6dc9e80` (2026-05-26); fork `AMD-Ecosystem/CosyVoice` exists, default branch `main` at the same tip. No new AMD PRs or issues since intake (`gh pr list --state all --search "ROCm OR AMD OR HIP"`: #1754 open, #1280 closed).
- `utils/surface.py generate` floor is empty (0 CUDA files, 0 build components); 15 components added by hand. `ext_type` recorded as `python-torch`.
- On ROCm torch, `torch.cuda.is_available()` is True, so `cosyvoice/cli/frontend.py:46` asks onnxruntime for `CUDAExecutionProvider`; `cosyvoice/utils/onnx.py:15` and `tools/extract_speech_token.py:68` ask for it unconditionally. Fix: select from `onnxruntime.get_available_providers()` (CUDA, then MIGraphX, then CPU). The onnxruntime ROCm EP has been deleted upstream; MIGraphX EP (AMD's `onnxruntime-migraphx` wheel, or the newer `onnxruntime-ep-migraphx` plugin, which must be registered before querying providers) is the target. #1754 keeps `onnxruntime-gpu` (the CUDA wheel) on Linux, which is wrong for ROCm.
- `load_trt=True` on ROCm dies at `import tensorrt` (`cosyvoice/cli/model.py:85`); `vllm_example.py` passes `load_trt=True`. Fix: extend the existing no-GPU downgrade in the three `cosyvoice/cli/cosyvoice.py` constructors with a `torch.version.hip` condition.
- Current torchaudio (2.9+) routes `torchaudio.load/save` (27 call sites) through torchcodec and ignores `backend='soundfile'`; #1754 added `torchcodec==0.9.1` for that reason. Version must match torch.
- No device code exists, so no wavefront, texture or memory-init fault class applies. Runtime risks to watch in validation: masked fp16 SDPA in the CosyVoice3 DiT (`cosyvoice/flow/DiT/modules.py:391,472`), and the LLM thread's side stream (`cosyvoice/cli/model.py:55`) versus the default-stream token2wav in `stream=True` mode.
