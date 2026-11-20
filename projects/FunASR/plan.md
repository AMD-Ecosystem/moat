# FunASR port plan

Planned 2026-11-20 on linux-gfx1100 (planner, no GPU used) at upstream `main` c49474879e7c (funasr 1.4.27).

**Scope is not final.** Three questions in `notes.md` ("Planning (2026-11-20)", "Decisions needed from Jeff") must be ruled before a porter starts. This plan is written for the recommended answers and marks every item that depends on a ruling with **[Q1]**..**[Q3]**.

## Project

- Name: FunASR
- Upstream: https://github.com/modelscope/FunASR (MIT), default branch `main`
- Fork: https://github.com/AMD-Ecosystem/FunASR, port branch `moat-port`
- What it is: Alibaba's speech toolkit. A Python package (`funasr`) on plain PyTorch (Paraformer, SenseVoice, Fun-ASR-Nano, FSMN-VAD, CT-Transformer punctuation, CAM++ speaker, emotion2vec, Qwen3-ASR / GLM-ASR wrappers), plus deployment runtimes under `runtime/` (ONNX Runtime Python and C++, a TorchScript Python package, a LibTorch C++ GPU SDK, a llama.cpp/GGUF runtime, Triton Inference Server + TensorRT).

## Existing AMD support

- Upstream: no ROCm/HIP code, branch or PR. AMD work upstream is Vulkan in the llama.cpp runtime only (issue #3479 open, PR #3482). Issue #2972 (Hygon DCU) was closed with "the PyTorch path may work, unvalidated". Rechecked 2026-11-20 with `gh search issues/prs --repo modelscope/FunASR`: nothing new.
- No AMD-official or community FunASR ROCm port found (intake; amd / AMD-AGI orgs not searchable with this token, so that corner stays open).
- Judgment: **improvable, no authoritative AMD effort to build on.** The value is a validated ROCm path for the Python toolkit, the one compiled-code fix, the ROCm equivalents of the NVIDIA-only runtime backends, and documentation.
- Delivery vehicle: an upstream PR is right. Upstream merges accelerator-specific work (it carries `docs/ascend_npu.md`, NPU device handling in `funasr/utils/amp.py` and `funasr/train_utils/device_funcs.py`) and is active (commits daily).
- Port vs rewrite: nothing performance-critical is NVIDIA-tuned. The WKV kernel is a serial per-channel scan; mechanical port.

## Build classification: torch-extension (Strategy B)

- Pure Python package (`setup.py`, `find_packages(include=["funasr*"])`), no build-time extension. The only compiled CUDA is JIT-built at run time by `torch.utils.cpp_extension.load` in `funasr/models/rwkv_bat/rwkv_attention.py:246` and `:292` from `funasr/models/rwkv_bat/cuda_{encoder,decoder}/{wkv_op.cpp,wkv_cuda.cu}` (shipped as package data, `setup.py:158-161`). On ROCm torch, `load()` hipifies the sources itself (torch `cpp_extension.py:2484`), so Strategy B: rely on torch's hipify, fix only what it cannot.
- Recorded: `moatlib.py set-ext-type FunASR torch-extension`.
- Side builds that are not torch extensions: `runtime/llama.cpp` (CMake, FetchContent of llama.cpp/ggml at 803b7fca, ggml GPU backend chosen by `-DGGML_*`), `runtime/onnxruntime` + `runtime/websocket` (CMake C++, CPU ONNX Runtime; `-DGPU=ON` adds LibTorch + BladeDISC `torch_blade`).

## Port strategy

1. **Python toolkit (validate-first).** ROCm PyTorch reports `"cuda"` for every `torch.cuda.*` / `.cuda()` / `device="cuda"` site (12 `is_available`, 7 `torch.cuda.amp`, memory stats, `set_device`), and `funasr/utils/amp.py` resolves autocast via `torch.accelerator`, which returns `cuda` on ROCm. Expectation: no code change. Prove it by running real inference on GPU and comparing with CPU.
2. **WKV JIT kernel (code change).** The loader passes nvcc-only flags (`-res-usage`, `--maxrregcount 60`, `-Xptxas -O3`, `--use_fast_math`) that hipcc/clang reject. Under `torch.version.hip`, pass `["-O3", f"-DTmax={context_size}"]` only. Do not translate `--use_fast_math` to `-ffast-math` (it implies finite-math-only under clang, and the kernel starts from `MIN_VALUE = -1e38` and divides by exp sums); plain `-O3` keeps CUDA-equivalent semantics without nvcc's approximate intrinsics. Fix the error message ("CUDA is currently a requirement") only if it touches the same lines. CUDA path stays byte-identical. See [Q3] for the commented-out loader calls.
3. **Triton timestamp kernels (validate).** `funasr/models/sense_voice/whisper_lib/triton_ops.py` (whisper's DTW and median-filter kernels) runs on ROCm Triton unchanged in principle; it is only reached through `add_word_timestamps` in the whisper transcribe path. Its callers catch `RuntimeError` and fall back to CPU with a warning, so validation must prove the Triton kernel ran (no fallback warning, or call `dtw_cuda` / `median_filter_cuda` directly and compare with the CPU `dtw_cpu` / sort-based median).
4. **vLLM paths (validate) [Q2].** `funasr/auto/auto_model_vllm.py`, `funasr/models/{fun_asr_nano,glm_asr}/inference_vllm*.py`, `vllm_utils.py`, and the `vllm` backends of `funasr/bin/server.py` / `realtime_ws.py`. Plain vLLM API (`LLM(...)`, `dtype`, `gpu_memory_utilization`, `enforce_eager`); ROCm vLLM should run it. Validate Fun-ASR-Nano via `AutoModelVLLM` on a ROCm vLLM build.
5. **`funasr_torch` TorchScript package (validate).** `runtime/python/libtorch/funasr_torch/*_bin.py` uses `torch.device(f"cuda:{id}")` and `.cuda()`; validate-only.
6. **ONNX Runtime GPU in `funasr_onnx` (code change) [Q1b].** `runtime/python/onnxruntime/funasr_onnx/utils/utils.py:197-224` (`OrtInferSession`) picks `CUDAExecutionProvider` only. ONNX Runtime's ROCm EP is gone; the AMD EP is `MIGraphXExecutionProvider` (`onnxruntime-migraphx` on PyPI, 1.27.1 current). Add: when `device_id != -1` and the CUDA EP is absent but the MIGraphX EP is present, use it with `{"device_id": device_id}`; keep the CPU fallback and its warning. Same treatment for the GPU guard in `runtime/python/utils/runtime_eval.py:81-84` (accept either GPU EP). Risk: FunASR's ONNX graphs have dynamic sequence lengths and MIGraphX compiles per input shape; record compile time and per-utterance latency, and say in the docs if it is only worth it for batched/fixed-length use.
7. **llama.cpp/GGUF runtime on ggml's HIP backend (code change) [Q1a].** Building `runtime/llama.cpp` with `-DGGML_HIP=ON -DGPU_TARGETS=<gfx>` needs no FunASR code change except `--backend` in `sensevoice/funasr-sensevoice/funasr-sensevoice.cpp:139-158`: it matches the string `"cuda"` against the ggml registry/device names, and ggml's HIP backend registers as `ROCm` (devices `ROCm0`...), so `--backend cuda` would exit "no GPU backend". Add a `rocm` backend value (and list it in the usage line). `runtime/llama.cpp/tests/test_backend_flags.py:13` asserts the literal `cpu|cuda|vulkan` in that source, so update the test with the usage string. The other tools need nothing: `funasr-encoder` and `funasr-cli`'s audio encoder compute on `ggml_backend_cpu_init()` only, `funasr-cli` sets `n_gpu_layers=0`, and `funasr-embd` offloads through llama's `-ngl`, which picks the HIP device without a name match. So SenseVoice is the only GGUF tool with a GPU path upstream (CUDA and Vulkan alike), and `funasr-embd -ngl` the only LLM offload to check. Upstream already ships Vulkan binaries for AMD; this adds a native-compute alternative and a documented build line. No release-workflow change (fork Actions stay disabled; adding a ROCm binary package is upstream's choice, mention it in the PR body).
8. **Documentation (code change, upstream-visible).** A new `docs/rocm.md` modelled on `docs/ascend_npu.md` (that page is Chinese; ours in English, with a short pointer from the README GPU install paragraph at `README.md:52-64`): install ROCm PyTorch (`pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm<ver>` on Linux; Windows via the ROCm torch wheels), `device="cuda"` works unchanged, what was validated on which GPUs, the WKV kernel note, the vLLM ROCm pointer, the ONNX MIGraphX EP install line [Q1b], the llama.cpp HIP build line [Q1a], and which paths stay NVIDIA-only (Triton/TensorRT; the LibTorch C++ GPU SDK [Q1c]). Check the docs contract tests (`tests/test_markdown_relative_links.py`, `tests/test_reference_docs_contract.py`, `tests/test_docs_funasr_install_commands.py`) pass with the new page and any README edit.

## CUDA surface inventory

| item | where | HIP/ROCm mapping | risk |
|---|---|---|---|
| `kernel_forward<float>`, `kernel_backward<float>` (x2 copies) | `funasr/models/rwkv_bat/cuda_{encoder,decoder}/wkv_cuda.cu` | torch JIT hipify, unchanged source | low; see risks R1-R4 |
| 4 `<<<>>>` launches, legacy default stream | same, `cuda_forward`/`cuda_backward` | hipLaunchKernelGGL via hipify | R3 |
| `TORCH_LIBRARY` + `PYBIND11_MODULE` op registration | `wkv_op.cpp` | unchanged | none |
| nvcc-only JIT flags | `rwkv_attention.py:267-274`, `:313-320` | guard on `torch.version.hip` | the actual break |
| Triton kernels `dtw_kernel`, generated `median_kernel` | `funasr/models/sense_voice/whisper_lib/triton_ops.py` | ROCm Triton | R5 |
| `torch.cuda.*`, `.cuda()`, `device="cuda"` (~50 sites), `torch.cuda.amp` via `funasr/utils/amp.py` | `funasr/` | ROCm torch masquerades as cuda | low |
| `nccl` default backend | `funasr/train_utils/device_funcs.py:172` | RCCL behind the same name on ROCm torch | multi-GPU only |
| DeepSpeed trainer | `funasr/train_utils/trainer_ds.py`, `funasr/bin/train_ds.py` | DeepSpeed has ROCm builds | optional, see test plan |
| vLLM | 11 files under `funasr/auto`, `funasr/models/{fun_asr_nano,glm_asr,moss_transcribe_diarize}`, `funasr/bin` | ROCm vLLM | [Q2] |
| `CUDAExecutionProvider` | `runtime/python/onnxruntime/funasr_onnx/utils/utils.py`, `runtime/python/utils/runtime_eval.py` | `MIGraphXExecutionProvider` | [Q1b], R6 |
| ggml CUDA backend name match | `runtime/llama.cpp/sensevoice/funasr-sensevoice/funasr-sensevoice.cpp` | ggml HIP backend (`ROCm`) | [Q1a] |
| LibTorch GPU SDK: `at::kCUDA`, links `torch_cuda c10_cuda torch_blade ral_base_context`, hardcoded `/usr/local/lib/python3.8/...` | `runtime/onnxruntime/src/paraformer-torch.cpp`, `runtime/onnxruntime/CMakeLists.txt:61-70`, `runtime/websocket/CMakeLists.txt:27-36`, `runtime/onnxruntime/src/CMakeLists.txt:44-46` | ROCm LibTorch (`torch_hip`, `c10_hip`); BladeDISC has no maintained ROCm build | [Q1c] |
| Triton Inference Server + TensorRT model repos | `runtime/triton_gpu/` | none (TensorRT has no ROCm equivalent; MIGraphX would be a rewrite of the model repos) | [Q1d] |

No cuBLAS/cuFFT/cuRAND/cuDNN/Thrust/CUB/NCCL C++ use, no textures, no warp intrinsics, no driver API, no NVRTC (surface census agrees: 2 CUDA files, 4 kernels, 4 launches, no libraries).

## Risk list

- **R1 wavefront:** `threadsPerBlock = min(C, 32)`; the kernel is one thread per (batch, channel) with no cross-lane operations, so a 32-thread block on wave64 is a half-occupied wavefront, correct on both widths. The Python-side assert `batch * dim % min(dim, 32) == 0` matches the launch and stays.
- **R2 per-thread arrays:** `F q[Tmax], r[Tmax]` in `kernel_backward` are private arrays sized by `context_size` (default configs use hundreds to thousands); on AMD they spill to scratch. Correct, possibly slow; record backward time for the tested `Tmax`.
- **R3 stream:** launches go to the legacy default stream rather than torch's current stream, on CUDA too. Harmless with torch's default stream; not ours to change.
- **R4 hipify write location:** `load()` on ROCm writes `wkv_cuda.hip` (and possibly a hipified `wkv_op.cpp`) next to the sources, i.e. into the installed package directory. A read-only site-packages makes the RWKV path fail on ROCm where CUDA works. Validate with an editable install and with a regular wheel install; if the second fails, the fix is to copy the sources into the build directory before `load()` (ROCm-only branch), and say so in notes.
- **R5 Triton:** whisper's `median_kernel` rewrites `kernel.src` on a `triton.JITFunction`; newer Triton versions changed that API (an upstream openai-whisper issue, not ROCm). If it fails on the ROCm Triton version, check the same Triton version on CUDA before calling it a ROCm fault. The `dtw_kernel` uses `BLOCK_SIZE=1024` in a single program with `tl.debug_barrier()`: fine on ROCm Triton.
- **R6 MIGraphX dynamic shapes:** per-shape compilation can make the GPU EP slower than CPU for streaming/short inputs. Measure before documenting it as a speedup. `onnxruntime.get_device()` may not report `"GPU"` on the MIGraphX build; the provider check must not depend on it.
- **R7 fp16/bf16 numerics:** Fun-ASR-Nano defaults to `dtype="bf16"`; gfx1100 supports bf16 in torch; gfx90a bf16 GEMM goes through hipBLASLt/rocBLAS. Compare transcripts, not bitwise logits.
- **R8 dead upstream code (not ours):** `funasr/models/conformer_rwkv/decoder.py:95-102` imports `funasr.models.sense_voice.rwkv_v4/v5/v6`, which no longer exist, so `TransformerRWKVDecoder` (used by `examples/aishell/conformer/conf/conformer_rwkv.yaml`) cannot build on any platform. Record, do not fix.
- **R9 vLLM availability:** ROCm vLLM is Linux-only and its RDNA support is narrower than CDNA; the Windows gate cannot run the vLLM path at all [Q2].

## File-by-file change list

| file | change | ruling |
|---|---|---|
| `funasr/models/rwkv_bat/rwkv_attention.py` | ROCm flag set in `load_encoder_wkv_kernel` / `load_decoder_wkv_kernel` (one shared helper is fine); R4 fix only if validation shows it is needed | [Q3] for re-enabling the calls |
| `tests/test_rwkv_wkv_kernel.py` (new) | GPU test: build the encoder kernel via `load_encoder_wkv_kernel(T)`, run `WKVLinearAttentionEncoder` forward/backward on small random fp32 inputs (B=2, T=16..64, C=32/64), compare forward with a step-by-step PyTorch reference (the `wkv_linear_attention` recurrence) and backward with `torch.autograd.gradcheck` in fp64-reference form or finite differences; same for the decoder kernel; `skipUnless(torch.cuda.is_available())` and skip without ninja | none |
| `runtime/python/onnxruntime/funasr_onnx/utils/utils.py` | MIGraphX EP selection | [Q1b] |
| `runtime/python/utils/runtime_eval.py` | accept MIGraphX EP in the GPU guard | [Q1b] |
| `runtime/llama.cpp/sensevoice/funasr-sensevoice/funasr-sensevoice.cpp`, `runtime/llama.cpp/tests/test_backend_flags.py` | `rocm` backend value; test's usage-string assert | [Q1a] |
| `runtime/llama.cpp/README.md` | ROCm/HIP build line next to the CUDA/Vulkan ones | [Q1a] |
| `docs/rocm.md` (new), `README.md` (pointer), optionally `docs/deployment_matrix.md` | docs | none |

Nothing in `wkv_cuda.cu` / `wkv_op.cpp` should need to change; if hipify output fails to compile, fix in the `.cu` with a guard that leaves the CUDA path byte-identical.

## Build commands

No package build. In the job venv (ROCm PyTorch already installed, never the image's shared Python):

```
git clone -b moat-port https://github.com/AMD-Ecosystem/FunASR projects/FunASR/src
pip install -e "projects/FunASR/src[test]" ninja
pip install torchaudio --index-url https://download.pytorch.org/whl/rocm<ver>   # if missing; must match torch
```

RWKV kernel: built on first call to `load_encoder_wkv_kernel(T)`; set `TORCH_EXTENSIONS_DIR` to a job-local directory. `PYTORCH_ROCM_ARCH` is not needed for `load()` (it targets the visible GPU) but set it explicitly to the host gfx for a reproducible log.

[Q1a] llama.cpp runtime:

```
cmake -S runtime/llama.cpp -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DGPU_TARGETS=<gfx> -DCMAKE_HIP_COMPILER=$ROCM_PATH/llvm/bin/clang++
cmake --build build-hip -j <cores>
```

(FetchContent clones llama.cpp at the pinned sha; set `FETCHCONTENT_SOURCE_DIR_LLAMA` to a pre-cloned copy on a rate-limited host.)

[Q1b] ONNX: `pip install onnxruntime-migraphx` (needs MIGraphX from the ROCm install) plus `pip install -e runtime/python/onnxruntime`.

CUDA no-regression: the only CUDA-path-affecting edits are Python (WKV flags behind `torch.version.hip`) and, if Q1a, C++ string handling compiled for every backend. Compile the C++ change with `-DGGML_CUDA=ON` on an nvcc host (mi355 fleet host / conda cuda-12.8 env); for the Python change, show that the CUDA flag list is unchanged by diff and, if a CUDA torch is available, build the WKV extension with nvcc once.

## Test plan

Upstream has no GPU test suite: `tests/` is mostly CPU unit and docs-contract tests (unittest-style, run by pytest), and `tests_models/run_all_tests.py` runs 9 model smoke scripts, 6 of them hardcoded to `device="cpu"` and 3 to `cuda:0` (Qwen3-ASR, GLM-ASR, ERes2NetV2). The validator therefore runs:

1. **Non-GPU regression set (must not regress):** `python -m pytest -q tests` on the port branch and on the fork's `main` in the same venv; compare pass/fail sets (network- and optional-dependency-dependent tests will skip or fail identically on both; record the delta, not the absolute count).
2. **New GPU test:** `python -m pytest -q tests/test_rwkv_wkv_kernel.py` (must report passes, not skips; confirm the extension was built by hipcc in the verbose log).
3. **GPU model inference, CPU-vs-GPU differential** (validator-side script in `agent_space/`, recorded in notes with transcripts): `AutoModel(..., device="cuda:0")` for `paraformer-zh` + `fsmn-vad` + `ct-punc` + `cam++` (the README pipeline), `paraformer-zh-streaming` (chunked, with cache), `iic/SenseVoiceSmall`, `FunAudioLLM/Fun-ASR-Nano-2512` (bf16 and fp32), `emotion2vec_plus_large`, and the three `tests_models` GPU scripts as written (`test_qwen3_asr.py`, `test_glm_asr.py`, `test_eres2netv2.py`). Pass = same text as the CPU run for the non-autoregressive models (Paraformer, SenseVoice), and a correct transcript within a small edit distance for the LLM models; timestamps within one frame. Also run the 6 CPU `tests_models` scripts once with `device` patched to `cuda:0` in the validator's copy (not in the fork).
4. **Triton kernels:** call `dtw_cuda` and `median_filter_cuda` on random inputs and compare with `dtw_cpu` and the sort-based median; then one whisper-path transcription with word timestamps and no "Failed to launch Triton kernels" warning.
5. **Training smoke:** `funasr-train` for a few steps of a small Paraformer or SenseVoice finetune on the bundled example data (`examples/industrial_data_pretraining/sense_voice/finetune.sh` pattern) with `use_fp16` and with bf16, confirming loss decreases and autocast is active on `cuda` (no "Disabling autocast" warning). DeepSpeed (`funasr-train-ds`) only if a ROCm DeepSpeed is installable in the job venv; otherwise record it as not run.
6. **vLLM [Q2]:** Fun-ASR-Nano through `AutoModelVLLM` and the `funasr-server --backend vllm` path on a ROCm vLLM install, Linux only; plus the vLLM contract tests (`tests/test_fun_asr_nano_*.py`, `tests/test_glm_asr_vllm_*.py`) with vLLM present.
7. **`funasr_torch`:** export a Paraformer to TorchScript (`funasr-export` / `AutoModel.export(type="torchscript")`) and run `runtime/python/libtorch/demo_paraformer.py` with `device_id=0`.
8. **[Q1b] ONNX MIGraphX:** export Paraformer and FSMN-VAD to ONNX, run `funasr_onnx.Paraformer(..., device_id=0)` and confirm `MIGraphXExecutionProvider` in `session.get_providers()`, same text as the CPU EP; `runtime/python/utils/runtime_eval.py --backend onnx --device_id 0`; the existing ONNX package tests (`tests/test_funasr_onnx_*.py`).
9. **[Q1a] GGUF:** `runtime/llama.cpp/download-funasr-model.sh` for SenseVoice + VAD, then `llama-funasr-sensevoice ... --backend rocm` and `--backend cpu` on the same wav, same text; the runtime's own tests under `runtime/llama.cpp/tests` with the HIP build; `llama-funasr-embd -ngl 99` on Fun-ASR-Nano embeddings (confirm `ROCm0` in the log); `pytest runtime/llama.cpp/tests`.

Count GPU dispatches (`AMD_LOG_LEVEL=1`, or `rocprofv3 --kernel-trace` on one run) for one inference per path, so a silent CPU fallback cannot pass.

Per gate: wave64 and wave32 (Linux) run 1-9 (6 subject to Q2); Windows runs 1-5 and 7 (no vLLM; 8 and 9 only if the ROCm packages exist for Windows, see Q2/Q1).

## Open questions

The three decisions for Jeff are in `notes.md` under "Planning (2026-11-20)". Technical unknowns the porter resolves and records:

- Whether `load()`'s hipify writes into the package directory (R4) and whether a non-editable install breaks.
- `onnxruntime.get_device()` value on the MIGraphX build and MIGraphX compile cost per shape (R6).
- Whether `onnxruntime-migraphx` and ggml-HIP are usable on Windows ROCm.
