# TransformerEngine notes

## Intake screen (2026-11-06)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: partial, AMD fork ROCm/TransformerEngine).

### Licence
`utils/licenses.py check NVIDIA/TransformerEngine`: Apache-2.0, tier 1. No per-file `scan-nvidia` run and no clone, because the recommendation is a decline.

### Existing AMD support
- `ROCm/TransformerEngine` is AMD's own port (a standalone repo in the ROCm org, default branch `dev`), committed to daily by AMD engineers (latest 2026-11-05, e.g. #783, #784).
- README "Feature Support Status": activation, cast, fused softmax, layernorm, rmsnorm, transpose, fused rope, FP8 recipe and HipRTC fully supported; GEMM through hipBLASLt for fp32/fp16/bf16/fp8 types; fused attention through AITER/CK and AOTriton backends; FlashAttention-2 without variable sequence length; HipGraph/HipTX partial; tensor, sequence and context parallelism supported; PyTorch and JAX test suites supported. AMD-specific extras: HipKittens and CK grouped GEMM for MoE, FlyDSL GEMM backend (gfx950), MXFP8/MXFP4 recipes.
- The matrix's "partial" reflects the items marked partial above (GEMM type coverage, FA2 varlen, HipGraph). Those are being worked in ROCm/TransformerEngine itself.

### Recommendation
Decline, reason `ported-elsewhere`: AMD maintains a mature, actively developed ROCm port (ROCm/TransformerEngine) with AMD-native backends (hipBLASLt, CK, AITER, AOTriton, HipKittens). The remaining partial features belong to that team's roadmap; a mechanical MOAT port would duplicate and underperform it.
