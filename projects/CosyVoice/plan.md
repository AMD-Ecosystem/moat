# CosyVoice port plan

Planned 2026-11-20 on linux-gfx942 (planner, no GPU used), against upstream `main` at `074ca6dc9e80` (2026-05-26).

**Before a porter starts, Jeff must rule on the scope questions in "Open questions" (Q1 to Q4).** The plan below is written for the recommended answer to each and says what changes under the alternatives.

## Project

- Name: CosyVoice
- Upstream: https://github.com/QwenAudio/CosyVoice (`FunAudioLLM/CosyVoice` redirects here), default branch `main`, Apache-2.0
- Fork: https://github.com/AMD-Ecosystem/CosyVoice (exists; default branch `main` is the mirror, at upstream's tip). Port work goes on `moat-port`.
- Submodule: `third_party/Matcha-TTS` (MIT) at `dd9105b34bf2`, pure PyTorch, no CUDA code, imported via `sys.path` by every entry point.
- Shape: a validation project. There is no CUDA source of our own to translate; the work is making the documented install and the inference entry points run on ROCm PyTorch, replacing the two NVIDIA-only runtime dependencies (TensorRT, onnxruntime CUDA EP) with their ROCm equivalent or a clean, named fallback, and proving it on real AMD GPUs.

## Existing AMD support

- Upstream PR #1754 (https://github.com/QwenAudio/CosyVoice/pull/1754), alexhegit (AMD), OPEN, `mergeStateStatus` DIRTY / CONFLICTING, untouched since 2025-12-28. Three files: `requirements-rocm.txt` (a copy of `requirements.txt` minus the cu121 index, TensorRT and the torch pins, plus `torchcodec==0.9.1`, but it still installs `onnxruntime-gpu==1.18.0`, the CUDA build, on Linux), a README "Install with AMD ROCm" section (torch from the `rocm6.4` index), and a `webui.py` default-model fix. The webui hunk is what conflicts: upstream since changed that default to `CosyVoice2-0.5B`, so the hunk is obsolete. Judgment: authoritative in intent (an AMD engineer, README-level, the right shape) but incomplete and unvalidated: no code changes, the CUDA onnxruntime wheel kept, no Windows, no vLLM, no evidence. Per the existing-support rule we build on it (same file name, same README placement) rather than invent a competing layout; the delivery question is Q4.
- Earlier closed PR #1280 (same author): README steps for AMD.
- Issue #839 "Why not supporting AMD?": a maintainer replied "we use pytorch and tensorrt, we don't do any hardware support"; another user answered that it works on AMD. Issue #101 (WSL2 + `torch 2.3.1+rocm6.0`, works) and #1189 (RX 6800 XT, `torch 2.6.0+rocm6.2.4`, a HIP OOM later attributed to a local config change) show community users already run it on ROCm by hand.
- No ROCm or AMD-Ecosystem port other than our fork. Classification: improvable community/AMD-engineer effort, no mature support. Decision: validate and finish, in #1754's shape.
- Upstream does not link platform forks; an upstream PR is the natural vehicle, but the maintainer's #839 reply and the 6-month-stale #1754 make acceptance uncertain (Q4).
- No performance kernels exist (no CUTLASS, no custom CUDA), so no port-versus-AMD-native-rewrite decision arises. The TensorRT estimator path is the only "acceleration" layer and is addressed in Q2.

## Build classification

Neither Strategy A nor B applies: pure Python on PyTorch, no `setup.py`/`pyproject.toml`, no `CUDAExtension`, no CMake. Evidence: `find . -name '*.cu*' -o -name 'CMakeLists.txt' -o -name setup.py` (outside `.git`) returns nothing; the install is `pip install -r requirements.txt` (README "Install"). Recorded `ext_type = python-torch`.

What decides the ROCm behaviour is the dependency set in `requirements.txt`:

| line | ROCm status |
|---|---|
| `--extra-index-url .../whl/cu121`, `torch==2.3.1`, `torchaudio==2.3.1` | CUDA wheels. ROCm needs torch/torchaudio from a ROCm index installed first (the 2.3.1 ROCm build is rocm6.0 only, which predates gfx1151/gfx1201 and Windows), so the ROCm requirements must not pin torch. |
| `onnxruntime-gpu==1.18.0` (Linux), extra index `onnxruntime-cuda-12` | CUDA EP. ROCm equivalent is the MIGraphX EP (`onnxruntime-migraphx` wheels from AMD, or the newer `onnxruntime-ep-migraphx` plugin, which needs explicit registration); the ROCm EP has been deleted upstream in onnxruntime, so do not target it. Fallback: plain `onnxruntime` CPU, which win32/darwin already use. |
| `tensorrt-cu12*==10.13.3.9` (Linux) | No ROCm build exists. |
| `deepspeed==0.15.1` (Linux) | Training only. ROCm DeepSpeed exists; ops build lazily. |
| `openai-whisper`, `pyworld`, `numpy==1.26.4`, `onnx==1.16.0`, ... | Platform-neutral but old; the binding constraint is the Python version the ROCm torch wheels need (numpy 1.26.4 has no cp313 wheel; Windows ROCm wheels may be cp312-only). |

## Port strategy

Dependency-and-guard port, minimal footprint, CUDA path byte-identical in behaviour:

1. A new `requirements-rocm.txt` (the name #1754 chose) that drops the CUDA indexes, the torch/torchaudio pins and TensorRT, and installs onnxruntime appropriately per platform. torch/torchaudio come from the ROCm index in a README step before it.
2. ONNX Runtime provider selection that asks onnxruntime what it has instead of keying on `torch.cuda.is_available()`, which is True on ROCm PyTorch and today requests `CUDAExecutionProvider` from a wheel that does not have it. Prefer CUDA EP when present (unchanged on NVIDIA), then MIGraphX EP when present, else CPU.
3. TensorRT guard: on a HIP build of torch (`torch.version.hip` set), `load_trt=True` degrades to the PyTorch estimator with a warning, the same way the code already degrades `load_trt` on a machine with no GPU. Today it dies at `import tensorrt` (and `vllm_example.py` passes `load_trt=True`).
4. README: an "Install with AMD ROCm" subsection for Linux and Windows.

No hipify, no compat header. `torch.cuda.*` streams, AMP and device strings work unchanged under HIP PyTorch and are not to be edited.

## CUDA surface inventory

`utils/surface.py generate` found 0 CUDA files and 0 build components, as expected for a pure-Python project; the components below were added by hand (see surface.json).

GPU-touching Python, file by file:

| site | what | ROCm |
|---|---|---|
| `cosyvoice/cli/model.py:36,113,265` | `torch.device('cuda')` when `torch.cuda.is_available()` | works under HIP torch |
| `cosyvoice/cli/model.py:55,130,277` | `torch.cuda.stream(torch.cuda.Stream(device))` as the LLM-thread context | HIP stream; works. Risk: the LLM runs on a side stream in a thread while token2wav runs on the default stream; correctness depends on the `synchronize()` at `model.py:240,394`. Watch for intermittent garbage audio in stream mode (memory-streams fault class). |
| `cosyvoice/cli/model.py:103,136,293,426` | `torch.cuda.amp.autocast(fp16)` | works (deprecation FutureWarning on new torch, not ours to fix) |
| `cosyvoice/cli/model.py:73-79,138-140` | `torch.jit.load` of shipped TorchScript `.zip` (llm text encoder, llm, flow encoder; fp16/fp32) | TorchScript archives are device-neutral; `map_location` handles it. Risk: the fp16 archives were traced on CUDA and may embed `cuda` device constants; works under HIP because HIP torch answers to `cuda`. Validate `load_jit=True, fp16=True`. |
| `cosyvoice/cli/model.py:81-90`, `cosyvoice/utils/common.py:199-214`, `cosyvoice/flow/flow_matching.py:127-152`, `cosyvoice/utils/file_utils.py:53-89` | TensorRT engine build, deserialize, `execute_async_v3` on `torch.cuda.current_stream().cuda_stream` | no ROCm TensorRT; guard (strategy item 3); Q2 |
| `cosyvoice/cli/model.py:142-150`, `cosyvoice/llm/llm.py:506-534`, `cosyvoice/vllm/cosyvoice2.py`, `cosyvoice/utils/file_utils.py:92-` | vLLM `LLMEngine` with `enable_prompt_embeds`, custom `CosyVoice2ForCausalLM` on `vllm.model_executor.models.qwen2`; bf16; `gpu_memory_utilization=0.2` | ROCm vLLM exists (Linux, Instinct and recent Radeon). Version window per upstream: 0.9.0 or >= 0.11 (V1). Risk: prompt-embeds support in the ROCm build of the chosen version; `export_cosyvoice2_vllm` shells out to `sed -i` (Linux-only, fine since vLLM is Linux-only). Q3. |
| `cosyvoice/cli/frontend.py:46-48` | speech tokenizer ORT session: `CUDAExecutionProvider` if `torch.cuda.is_available()` | wrong provider on ROCm; strategy item 2 |
| `cosyvoice/utils/onnx.py:15` | training-time speech tokenizer: `CUDAExecutionProvider` only, `device_id=LOCAL_RANK` | strategy item 2 (keep per-rank device id for whichever GPU EP is chosen) |
| `cosyvoice/bin/export_onnx.py:93` | ORT check of the exported estimator | strategy item 2 |
| `tools/extract_speech_token.py:68` | data-prep ORT: `CUDAExecutionProvider` only | strategy item 2 |
| `cosyvoice/utils/train_utils.py`, `cosyvoice/bin/train.py` | DDP `nccl` (RCCL under HIP), DeepSpeed, `GradScaler`, `pin_memory` | works under HIP torch; training is Q1 |
| `cosyvoice/flow/DiT/modules.py:391,472` | `F.scaled_dot_product_attention` with an explicit mask (CosyVoice3 DiT) | ROCm SDPA backends (math/mem-efficient/flash via aotriton) differ from CUDA; masked SDPA in fp16 is a numerics risk. Compare against CPU fp32. |
| `cosyvoice/hifigan/generator.py:730`, `cosyvoice/flow/flow.py:418` | `torch.backends.cudnn.*` in `__main__` test stubs | MIOpen under HIP; harmless |
| `runtime/triton_trtllm/**` | NVIDIA Triton + TensorRT-LLM serving (Dockerfile on NVIDIA images, TRT-LLM engines) | no ROCm equivalent of TRT-LLM/Triton-TRT backend; Q2 |
| `docker/Dockerfile`, `runtime/python/Dockerfile` | `nvidia/cuda` and `pytorch:...-cuda` bases | Q2 (ROCm Dockerfile is optional) |
| `examples/grpo/cosyvoice2/**` | GRPO RL training recipe (verl, `onnxruntime-gpu`) | Q1 |

## Risk list

1. **Provider fallback semantics.** With a CPU-only onnxruntime wheel, requesting `CUDAExecutionProvider` logs a warning and falls back to CPU in recent onnxruntime; `onnxruntime-gpu` (the CUDA wheel #1754 keeps) on a machine without CUDA fails to load the EP and also falls back. Verify both; either way the fix is explicit selection, not relying on the warning.
2. **MIGraphX EP on dynamic shapes.** The speech tokenizer (`speech_tokenizer_v1.onnx`, `_v2`, `_v3`) takes variable-length mel input; MIGraphX compiles per input shape, so each new prompt length may pay a compile. Measure first-call and steady-state time against CPU, and compare tokens exactly (integer output, must be identical to CPU EP). If MIGraphX is wrong or impractically slow, keep CPU and record the measured reason; do not silently drop it (Jeff's rule: ROCm library over CPU fallback, CPU only with a named reason). `campplus.onnx` already runs on CPU everywhere upstream; leave it.
3. **Pins versus current ROCm wheels.** torch for current ROCm is >= 2.9; torchaudio >= 2.9 routes `torchaudio.load`/`save` through torchcodec (27 call sites, including `load_wav`'s `backend='soundfile'`, which the new API ignores). #1754 added `torchcodec==0.9.1`; the torchcodec version must match the installed torch, and PyPI torchcodec is a CPU (ffmpeg) build, so confirm it loads against ROCm torch on Linux and Windows. `transformers==4.51.3`, `diffusers==0.29.0`, `lightning==2.2.4` against torch 2.9+: confirm import and inference; relax a pin only on evidence, and only in `requirements-rocm.txt`.
4. **Python version window.** Upstream documents Python 3.10. Windows ROCm torch wheels and `numpy==1.26.4`/`pyworld==0.3.4`/`openai-whisper` wheels must share a Python; record the version that works on each platform and say it in the README.
5. **fp16 numerics.** `fp16=True` with AMP over the flow-matching estimator and masked SDPA; compare GPU fp16 vs CPU fp32 output (see test plan), especially on RDNA where some fp16 kernels fall back.
6. **Stream/thread hazard** in the LLM side-stream (inventory above). Run `stream=True` paths repeatedly.
7. **vLLM ROCm** availability per arch and version, prompt embeds, and bf16 on gfx942 vs RDNA (Q3).
8. **Model downloads** (~1-5 GB each from Hugging Face/ModelScope) on every validating host; keep them in a job directory, never shared paths. `CosyVoice-ttsfrd` is optional (wetext fallback), x86_64 cp310 Linux only: skip it.
9. Windows: `pyworld`, `openai-whisper` build from source when no wheel; `sox` is not required with the soundfile backend; the `sed -i` in `export_cosyvoice2_vllm` is vLLM-only.

No wavefront, texture, warp-intrinsic or uninitialized-device-memory risk exists: there is no device code.

## File-by-file change list (fork `moat-port`)

1. `requirements-rocm.txt` (new): `requirements.txt` without the two extra indexes, `tensorrt-cu12*`, `torch`, `torchaudio`; `onnxruntime` (CPU) as the default on every platform, since the MIGraphX EP wheel is not on PyPI (it comes from AMD's index, documented in the README as an optional accelerator, Linux only); `deepspeed` kept Linux-only (Q1 decides whether it stays); `torchcodec` at the version matching the validated torch, if validation shows it is needed. Keep every other pin identical unless a validated incompatibility forces a change, and record each such change in notes.
2. `cosyvoice/utils/onnx.py` or a small helper in an existing util module: one function returning the provider list (`CUDAExecutionProvider` if in `onnxruntime.get_available_providers()`, else `MIGraphXExecutionProvider` if available, else `CPUExecutionProvider`; carry `device_id` for whichever GPU EP is chosen). Use it at `cosyvoice/cli/frontend.py:46`, `cosyvoice/utils/onnx.py:15`, `cosyvoice/bin/export_onnx.py:93`, `tools/extract_speech_token.py:68`. On NVIDIA with `onnxruntime-gpu` this resolves to exactly today's provider. If the plugin-style `onnxruntime-ep-migraphx` package is what the validated ROCm release ships, registration must happen in the helper before the provider query, guarded by an import try.
3. `cosyvoice/cli/cosyvoice.py` (the three constructors, lines ~47, ~159, ~209): extend the existing "no cuda device" downgrade so that `load_trt` is set False with a warning when `torch.version.hip` is set. One condition per site, no refactor. Do not touch `load_jit`, `fp16` or `load_vllm` there.
4. `README.md`, "Install": a "Install with AMD ROCm" subsection after the conda step: install torch/torchaudio from the ROCm index (Linux) or AMD's Windows wheels, then `pip install -r requirements-rocm.txt`; note that TensorRT (`load_trt`) and the `runtime/triton_trtllm` serving stack are NVIDIA-only; the optional MIGraphX EP install. No device-selection env vars in front of commands.
5. Nothing else unless validation finds a real fault. Explicitly unchanged: `torch.cuda.*` call sites, `requirements.txt`, Dockerfiles, `runtime/triton_trtllm`, the Matcha-TTS submodule.

Every edit must leave the NVIDIA path identical: `requirements.txt` untouched, provider helper resolves to CUDA EP on CUDA builds, TRT guard keyed on `torch.version.hip`. There is no CUDA compile step to run with nvcc; the CUDA no-regression check for this project is reading the diff plus, if an NVIDIA fleet host is free, one `example.py` run there with `requirements.txt`.

## Build commands

No compile. Per platform, in a job-local venv:

Linux (wave64 gfx942/gfx90a/gfx950, wave32 gfx1100/gfx1201):

```bash
git clone --recursive -b moat-port https://github.com/AMD-Ecosystem/CosyVoice.git && cd CosyVoice
python3 -m venv .venv && . .venv/bin/activate          # Python per README (3.10, or the version recorded)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm<ver>   # match host ROCm
pip install -r requirements-rocm.txt
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name())"
# optional MIGraphX EP: per AMD's ONNX Runtime install guide for the host ROCm release
```

Windows (gfx1151): same, with torch/torchaudio from AMD's Windows ROCm wheels; record the exact index and versions in notes.

Models (job-local `pretrained_models/`): `Fun-CosyVoice3-0.5B-2512`, `CosyVoice2-0.5B`, `CosyVoice-300M`, `CosyVoice-300M-SFT`, `CosyVoice-300M-Instruct` via `huggingface_hub.snapshot_download` as in the README.

## Test plan

Upstream has no unit tests and no GPU CI (the only workflow is lint: tabs, trailing whitespace, flake8 with the README's flags). The gate is therefore a scripted inference harness kept in `agent_space/`, not committed, driving upstream's own entry points.

GPU gate, each validating platform:

1. **`example.py` all three functions** (`cosyvoice_example`, `cosyvoice2_example`, `cosyvoice3_example`; `main()` runs only v3 by default, so call each), every mode they exercise: SFT, zero-shot, cross-lingual, VC, instruct (v1), zero-shot + saved speaker + fine-grained + instruct2 + bistream (v2), zero-shot + fine-grained + instruct2 + hotfix + Japanese (v3). Run each model twice: default (fp32, no jit) and `load_jit=True, fp16=True` where the model ships jit archives (v1, v2); v3 with `fp16=True`. Also one `stream=True` run per model, repeated 5 times.
2. **Acceptance per output**: finite, non-silent audio (RMS above a floor), duration within 25% of the CPU reference for the same seed and text (`set_all_random_seed`), and for v2/v3 zero-shot a speaker-similarity or ASR check: transcribe the GPU and CPU outputs with `openai-whisper` (already a dependency) and require matching text up to normalization for the Chinese/English cases. Speech-tokenizer tokens from the chosen GPU EP must equal the CPU EP tokens exactly.
3. **TRT guard**: `AutoModel(..., load_trt=True)` on ROCm logs the warning and synthesizes.
4. **vLLM** (Q3; Linux only): `vllm_example.py` (`cosyvoice3_example` and `cosyvoice2_example`, iterations reduced from 100 to e.g. 5 in the harness, not in the file) with ROCm vLLM; same acceptance as item 2 against the non-vLLM GPU output.
5. **Servers**: `webui.py --model_dir pretrained_models/CosyVoice2-0.5B` starts and one request synthesizes; `runtime/python/fastapi/server.py` starts and `client.py` gets audio. Smoke only.
6. Record RTF (upstream's own log line) per platform as context, not as a gate.

Non-GPU regression set: upstream's flake8 command (`flake8 --max-line-length 180 --ignore B006,B008,B905,C408,E402,E731,E741,W503,W504,F401,F403,F405,F722,F841 --exclude ./third_party/,./runtime/python/grpc/cosyvoice_pb2*py`) and the tab/trailing-space checks on the fork diff; one CPU run of `cosyvoice2_example` (the CPU reference above doubles as this).

Training (Q1, only if ruled in): `examples/libritts/cosyvoice2/run.sh` stages for a few hundred steps on one GPU with `torch_ddp` (and DeepSpeed on Linux), loss decreasing and no NaN; `tools/extract_speech_token.py` on a handful of utterances through the new provider helper.

## Surface accounting

`surface.json` lists the components above by hand (the generator's floor was empty). Recommended answers put in scope: `inference:cli` (all three model families), `onnx:speech-tokenizer-provider`, `guard:tensorrt`, `install:requirements-rocm`, `docs:readme-rocm`, `app:webui`, `server:fastapi`, `server:grpc`, `vllm:llm-backend` (Linux), `platform:windows`. Proposed out pending Jeff: `tensorrt:flow-estimator` (no ROCm TensorRT), `runtime:triton-trtllm`, `docker:nvidia-images`, `training:cosyvoice-recipes`, `training:grpo`. Nothing is recorded `scoped_out` yet, because each cut needs Jeff's ruling; once he rules, the porter moves those entries to `scoped_out` with the ruling as the reason, or to `owed`.

## Open questions

These are for Jeff, repeated at the top of the notes section; the porter must not start until he answers Q1 and Q2.

- **Q1 Training recipes.** `cosyvoice/bin/train.py`, `examples/libritts`, `examples/magicdata-read` (DeepSpeed/torch DDP) and `examples/grpo` (verl RL) are a training layer. Options: (a) inference-only port, training recorded out of scope ("not validated on ROCm"); (b) include a short single-GPU training smoke for `examples/libritts/cosyvoice2` (torch_ddp, plus DeepSpeed on Linux), grpo out; (c) include both. Recommendation: (b), because the only code change touching training is the provider helper and a smoke run is cheap, but it needs the LibriTTS download (~tens of GB for the full set; a subset suffices).
- **Q2 TensorRT layers.** `load_trt` (TensorRT flow estimator), `runtime/triton_trtllm` (Triton + TensorRT-LLM serving) and the NVIDIA Dockerfiles have no ROCm equivalent to translate into. Options: (a) guard `load_trt` and document both as NVIDIA-only, no ROCm replacement (smallest port; the PyTorch estimator is the ROCm path); (b) also add a MIGraphX-compiled estimator as the ROCm counterpart of `load_trt` (new feature work against the exported ONNX, larger and less likely to be accepted upstream); (c) also a ROCm serving stack (vLLM- or MIGraphX-based) replacing triton_trtllm (a new subsystem). Recommendation: (a), with (b) recorded as a deferral if wanted.
- **Q3 vLLM.** Options: (a) in scope on Linux for wave64 and, where ROCm vLLM supports it, wave32, with Windows excluded because vLLM has no Windows build; (b) out of scope. Recommendation: (a); vLLM is upstream's documented fast path and ROCm vLLM exists. If ROCm vLLM lacks a working prompt-embeds path at an upstream-supported version, the porter records that and it comes back to Jeff.
- **Q4 Delivery vehicle.** #1754 by alexhegit is the existing AMD PR, conflicting only on an obsolete webui hunk, and the maintainers said in #839 they "don't do any hardware support". Options: (a) open our own upstream PR with the validated changes, crediting #1754 in the body; (b) offer the validated changes to alexhegit to refresh #1754 (an upstream-visible comment, Jeff's call); (c) keep the fork as the deliverable and send nothing until upstream shows activity. This is decided at PR time, not before porting, but it determines whether the port keeps #1754's exact file layout (the plan does, so (a) and (b) both stay open).
- Not a question, a stated default: the speech tokenizer uses the MIGraphX EP when installed and CPU otherwise; if validation shows MIGraphX wrong or impractical on its dynamic shapes, the porter records the numbers and the CPU path is the named fallback, and that also goes to Jeff.
