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

## Planning (2026-11-20, linux-gfx1100 host, planner; no GPU used, nothing built)

### Decisions needed from Jeff (rule before a porter starts)

The plan (`plan.md`) is written for the recommended answers below; every dependent item there is tagged [Q1]..[Q3]. Nothing has been assumed: until these are ruled, the porter should not start.

**Q1. Which deployment runtimes are in the port?** The Python toolkit (`funasr/`, plain PyTorch plus the WKV JIT kernel and the Triton timestamp kernels) is in scope either way. Upstream also ships four runtimes that have an NVIDIA GPU path:

- **Q1a. llama.cpp/GGUF runtime on ggml's HIP backend.** Upstream ships CUDA and Vulkan builds; AMD is served today by Vulkan. A HIP build needs only a `--backend rocm` value in `funasr-sensevoice.cpp` (ggml's HIP backend registers as `ROCm`, so `--backend cuda` cannot find it), the matching test string, and a README build line. Options: (i) in scope, (ii) out of scope, AMD stays on Vulkan. Recommend (i): small, and it is the HIP equivalent of a shipped CUDA backend.
- **Q1b. ONNX Runtime GPU in `funasr_onnx`.** Only `CUDAExecutionProvider` is selected; ONNX Runtime's ROCm EP is gone and the AMD EP is MIGraphX (`onnxruntime-migraphx` on PyPI). Intake called this path "CPU-only on AMD"; that conflicts with the ROCm-library-over-CPU-fallback rule. Options: (i) add MIGraphX EP selection (two Python files) and validate it, (ii) document as CPU-only on AMD. Recommend (i), with the caveat that MIGraphX compiles per input shape, so it may lose to CPU on short streaming inputs; the validator measures before the docs claim a speedup.
- **Q1c. LibTorch C++ GPU SDK** (`runtime/onnxruntime` / `runtime/websocket` with `-DGPU=ON`, `paraformer-torch.cpp`). Links `torch_cuda c10_cuda` plus Alibaba BladeDISC's `torch_blade` and `ral_base_context`, with `/usr/local/lib/python3.8/...` hardcoded, and loads Blade-optimized TorchScript. The torch part maps to ROCm LibTorch (`torch_hip`, `c10_hip`), but BladeDISC has no maintained ROCm build. Options: (i) scope out and document as NVIDIA-only, (ii) port the LibTorch part and make `torch_blade` optional on ROCm (plain TorchScript models), a larger CMake change to a 2024-vintage SDK. Recommend (i).
- **Q1d. Triton Inference Server + TensorRT** (`runtime/triton_gpu/`). TensorRT has no ROCm counterpart; MIGraphX model repositories would be a rewrite, not a port. Recommend scope out, documented as NVIDIA-only.

**Q2. vLLM validation bar.** The Fun-ASR-Nano / GLM-ASR vLLM paths need no code change but a ROCm vLLM install. ROCm vLLM is Linux-only, so the Windows gate cannot run it. Options: (i) required on both Linux gates (wave64 and wave32), recorded `owed` until run, not applicable on Windows; (ii) required on wave64 only, best-effort on wave32; (iii) best-effort everywhere, not gating. Recommend (i) if the RDNA host can install a ROCm vLLM, else (ii).

**Q3. RWKV WKV loader is dead upstream.** The only CUDA source is the RWKV-BAT WKV kernel, but upstream commented out both calls that build it (`# load_encoder_wkv_kernel(context_size)` at `rwkv_attention.py:561`, decoder at `:488`), so the full-sequence path dereferences `None` on CUDA as well; only a caller that runs the loader itself reaches the kernel. Options: (i) fix only the hipcc flags under `torch.version.hip`, and prove the kernel with a new GPU test that calls the loader explicitly (CUDA behaviour unchanged); (ii) also restore the two loader calls, which fixes a shared-code bug on every platform and belongs in its own commit, or a separate upstream report. Recommend (i), noting the dead loader in the PR body so upstream can choose.

(Delivery vehicle needs no ruling: a normal upstream PR with code, one new GPU test and `docs/rocm.md`. Upstream already merges accelerator docs and device handling for Ascend NPU.)

### What was found (upstream c49474879e7c, funasr 1.4.27)

- Build: pure Python; the WKV kernel is JIT-built by `torch.utils.cpp_extension.load` (`ext_type = torch-extension`, Strategy B). On ROCm torch, `load()` hipifies the sources itself (torch `cpp_extension.py:2484`, local torch 2.16.0a0 / HIP 7.16). The hipified `.hip` may be written next to the sources inside the installed package (plan R4).
- WKV kernel: one thread per (batch, channel), serial over T, `blockDim = min(C, 32)`, no warp intrinsics, private `q[Tmax], r[Tmax]` arrays in backward. Correct on both wavefront widths; the break is only the nvcc flags. Do not map `--use_fast_math` to `-ffast-math`.
- Triton: whisper's `dtw_kernel` / `median_kernel` in `funasr/models/sense_voice/whisper_lib/triton_ops.py`; callers silently fall back to CPU on `RuntimeError`, so validation must show the kernel ran.
- Upstream moved since intake: new `tests_models/` (9 model smoke scripts, 3 on `cuda:0`), a large docs-contract test set in `tests/`, `docs/ascend_npu.md` (accelerator-doc precedent), and the llama.cpp runtime's `tests/test_backend_flags.py` pins the usage string `cpu|cuda|vulkan`.
- In the GGUF runtime only `funasr-sensevoice` has a GPU backend switch; `funasr-encoder` and `funasr-cli`'s encoder are CPU-only, and `funasr-cli` sets `n_gpu_layers=0`.
- Not ours, recorded for the reviewer: `funasr/models/conformer_rwkv/decoder.py:95-102` imports `funasr.models.sense_voice.rwkv_v4/v5/v6`, which no longer exist, so `TransformerRWKVDecoder` (example `conformer_rwkv.yaml`) cannot build on any platform.
- No new AMD effort upstream (`gh search` issues/PRs for rocm/hip/amd, 2026-11-20): only #3479 (Vulkan, open) and #2972 (DCU, closed).
- surface.json: generated floor (2 CUDA files, 31 CMake components) plus 13 planner entries. The 30 CPU-only ONNX Runtime C++ components (executables, libraries, tests and their non-GPU options) are `scoped_out` as having no GPU path; `option:GPU` and every Q-tagged entry are left unaccounted for the ruling and the porter. `surface.py check FunASR` passes (it does not judge a `planned` project's uncovered items).
