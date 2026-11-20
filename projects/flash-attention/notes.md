# flash-attention notes

## Intake screen (2026-11-06)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: partial, upstream).

- Licence: BSD-3-Clause, tier 1 (`utils/licenses.py check Dao-AILab/flash-attention`), https://github.com/Dao-AILab/flash-attention/blob/main/LICENSE. Not cloned; no submodule/NVIDIA-file scan done because the recommendation is a decline.
- Upstream already supports ROCm. README section "AMD ROCm Support": two backends, Composable Kernel (default; MI200x, MI250x, MI300x, MI355x, RDNA 3/4) and Triton (kernels from ROCm/aiter as `third_party/aiter`; CDNA and RDNA, fwd/bwd, varlen, MQA/GQA, dropout, paged attention, FP8 via the FA3 interface). Install with `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`.
- AMD fork: https://github.com/ROCm/flash-attention (ROCm org, AMD-owned, last pushed 2026-11-02), the staging ground for the CK/Triton work that lands upstream.
- jeff's own merged upstream ROCm PR: #1944 "[ROCm] prepare CK sources for pytorch hipify v2 APIs" (merged 2025-10-18).
- Active AMD-side upstream work in flight (open PRs): #3064 aiter adapter (micmelesse), #2981 AITER FMHA v3 dispatch, #2963 gfx12 BF16 D256 backward kernels, #2930 hdim 192 instances, #2243 gfx1150/gfx1151 in RDNA_ARCHS, #2217 LLC awareness.
- Remaining gaps (the matrix's "partial"): the Hopper FA3 CUDA kernels and the CuTe-DSL FA4 path are NVIDIA-architecture-specific with no hipify target; AMD covers them through the CK/aiter backends rather than a port. RDNA 3.5 (gfx1151) enablement is already an open PR (#2243). None of this is a MOAT port; any gfx1151/Windows help belongs on those existing PRs, through the AMD team.

Recommendation: decline, SKIP_REASON `already-supported` (upstream ships ROCm CK and Triton backends maintained by AMD, with AMD PRs actively landing).
