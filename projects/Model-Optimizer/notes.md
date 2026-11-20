# Model-Optimizer notes

## Intake screen (2026-11-05)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: partial, AMD alternative AMD Quark / ROCm quantization tooling).

### Licence
- `licenses.py check NVIDIA/Model-Optimizer`: Apache-2.0, tier 1. Recorded `license_spdx = Apache-2.0`.
- Because this one is not a perfectly clean decline (see Remaining gap), the per-file scan was run: `git clone --depth 1` into `agent_space/Model-Optimizer-screen`, `python3 utils/licenses.py scan-nvidia agent_space/Model-Optimizer-screen` -> "no NVIDIA proprietary licence text". One submodule, `tools/launcher/modules/Megatron-LM` (NVIDIA/Megatron-LM, BSD-3-Clause, launcher tooling only, not the library).

### What it is
- NVIDIA TensorRT Model Optimizer (`modelopt`): mostly Python over PyTorch (quantization, distillation, pruning, NAS, speculative decoding), exporting for TensorRT-LLM / TensorRT / vLLM. CUDA surface is 10 `.cu` files of optional fake-quant / quant-GEMM extensions in `modelopt/torch/kernels/quantization/{gemm,ggml,conv}`, JIT-built through `modelopt/torch/utils/cpp_extension.py`; a load failure only warns, so the library falls back to PyTorch paths on a machine without them.

### Existing AMD support
- No AMD fork (`ROCm/Model-Optimizer` and `AMD-Ecosystem/Model-Optimizer` do not exist).
- AMD's own quantization toolkit is https://github.com/amd/Quark (AMD Quark), AMD-owned, active (pushed 2026-09-28), AMD's path for quantizing models for AMD hardware. It covers the same role as modelopt's PTQ/QAT for AMD deployment.
- Open upstream PR by an AMD engineer: NVIDIA/Model-Optimizer#1990 (zhihuidu-amd, opened 2026-07-17, rebased 2026-10-26, +5228/-0, 19 files, REVIEW_REQUIRED, maintainer added `stale` / `waiting for feedback` labels 2026-10-26). It adds MI300X FP8 via `torch._scaled_mm` / hipBLASLt (`modelopt/_rocm_compat.py`), a MIGraphX deploy backend, AMD FP8 (e4m3fnuz) / INT8 configs and presets, and 32 MI300X tests. It is purely additive and does NOT give the `.cu` extensions a HIP build. Per intake.md, an open PR for the same work means contribute to it, not compete.

### Remaining gap
- The 10 CUDA extension kernels have no HIP build path, so on ROCm `modelopt` runs those quantizers through the slower PyTorch fallback. That is the only HIP-port-shaped work here, and it overlaps both an AMD engineer's in-flight upstream PR and AMD's own product (Quark).

### Recommendation
Decline, SKIP_REASON `ported-elsewhere`: AMD covers model quantization with its own maintained AMD Quark, and AMD's upstream ROCm enablement of modelopt is already in flight in #1990. A MOAT port would compete with both.

### Question for jeff
Should MOAT instead offer the HIP build of modelopt's JIT CUDA extensions (`modelopt/torch/kernels/quantization/{gemm,ggml,conv}`, 10 `.cu` files; licence clean per `scan-nvidia`) as a follow-on to zhihuidu-amd's NVIDIA/Model-Optimizer#1990, coordinated with that author, or is AMD Quark plus #1990 enough to leave it?
