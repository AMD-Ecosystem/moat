# FlashMLA notes

## Intake screen (2026-11-06)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: partial, AMD alternative via AITER MLA kernels).

- Licence: MIT, tier 1 (`utils/licenses.py check deepseek-ai/FlashMLA`), https://github.com/deepseek-ai/FlashMLA/blob/main/LICENSE. Not cloned; no submodule/NVIDIA-file scan done because the recommendation is a decline.
- What it is: DeepSeek's MLA/sparse-attention (DSA) kernels for DeepSeek V3.2-V4.1 inference. Hand-tuned for NVIDIA SM90/SM100 (CUTLASS/CuTe, wgmma/tcgen05) plus Huawei Ascend; the 2026-09-30 release dropped Hopper and changed the FP8/FP4 KV-cache format. A hipify port has nothing to translate the hot paths to; AMD's equivalent is an AMD-native rewrite.
- Upstream README already points AMD users elsewhere: "AMD Instinct ... The corresponding FlashMLA version can be found at: AITER/MLA" (https://github.com/ROCm/aiter/blob/main/aiter/mla.py), alongside MetaX and Hygon DCU forks. That link IS the AMD port (assess-existing-support step 1). Upstream PR #55 (sunway513, AMD) "Update README.md for AMD MLA" is the origin of it.
- ROCm/aiter: AMD-owned (ROCm org), MIT, not a fork, pushed 2026-11-06. Carries MLA decode (asm `mla_decode_fwd`, `mla_decode_v4.py`), Triton MLA, sparse MLA prefill (`mla_sparse_prefill.py`), gfx950 Gluon sparse MLA incl. FP8, DSv4 sparse MLA training kernels, and gfx1250 FlyDSL MLA kernels. So the DSA/V4 surface the matrix flags is covered by AMD's own library, on CDNA.
- Precedent: deepseek-ai/DeepGEMM was skipped 2026-05-29 as `ported-elsewhere` via ROCm/aiter.
- Residual gap: AITER targets CDNA (gfx942/gfx950/gfx1250); no RDNA/Windows MLA path. That is AITER's scope, not a FlashMLA port, and DeepSeek-scale serving is a datacenter workload.

Recommendation: decline, SKIP_REASON `ported-elsewhere` (AMD-owned ROCm/aiter provides the MLA and sparse-MLA kernels, and upstream's README links it as the AMD version).
