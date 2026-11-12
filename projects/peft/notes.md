# peft notes

## Intake screen (2026-11-05, gap-matrix seed; matrix: ready, upstream)

- Licence: Apache-2.0, tier 1 (`utils/licenses.py check huggingface/peft`). No vendored code or submodules.
- CUDA surface: the only CUDA in the tree is `src/peft/tuners/boft/fbd/` (`fbd_cuda.cpp` 815 B, `fbd_cuda_kernel.cu` 3076 B), the optional "fast block diagonal" helper for BOFT. `boft/layer.py:get_fbd_cuda()` JIT-builds it with `torch.utils.cpp_extension.load` inside try/except and falls back to the pure-torch path when the build fails (`fbd_cuda_available = False`). The kernel is two plain elementwise kernels (no warp intrinsics, no `__shfl`, no 32-lane assumption, no CUDA libraries); on a ROCm torch `cpp_extension.load` hipifies it automatically, and if it ever failed the fallback is already correct.
- Everything else is pure Python on PyTorch, which runs on ROCm unchanged.
- Existing AMD support: AMD documents and tests PEFT on ROCm itself -- ROCm docs "Fine-tuning LLMs and inference optimization" (https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/index.html, LoRA via PEFT), the single-GPU fine-tuning guide (https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/fine-tuning/single-gpu-fine-tuning-and-inference.html), and the AI Developer Hub LoRA tutorial tested on MI300X (https://rocm.docs.amd.com/projects/ai-developer-hub/en/v3.0/notebooks/fine_tune/LoRA_Llama-3.2.html, `pip install peft`).
- Upstream: active (pushed 2026-11-05), not archived. No open PR matching hip/rocm/amd; no ROCm/AMD mention in the repo's code search; no open ROCm issue.
- Duplicate effort: none needed; nothing for a fork to carry. The only GPU dependency that matters for QLoRA is bitsandbytes (screened separately).

## Recommendation: decline, `already-supported`

PEFT is a pure-PyTorch library that runs on ROCm through PyTorch's own HIP backend, and AMD's ROCm documentation already ships PEFT/LoRA fine-tuning guides on Instinct GPUs. The 3 KB BOFT helper kernel is an optional JIT extension that torch hipifies on ROCm and that falls back to pure torch on any build failure, so there is no port, no packaging fix and no meaningful docs change to send upstream.
