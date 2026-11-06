# kaolin notes

## Intake screen (2026-11-06)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: partial, AMD fork exists).

### Licence
`utils/licenses.py check NVIDIAGameWorks/kaolin`: Apache-2.0, tier 1. LICENSE on `master` read by hand: the Apache License 2.0 text. Older Kaolin releases were under the NVIDIA Source Code License; the current default branch is not. No per-file `scan-nvidia` run, because the recommendation is a decline; run it before any take-up, since an older-licence file could survive in the tree.

### Existing AMD support
- `AMD-Ecosystem/kaolin` (a GitHub fork of NVIDIAGameWorks/kaolin, in the AMD-Ecosystem org) carries an AMD ROCm enablement: two commits on `master` dated 2026-09-17 by AMD engineers (vcsajjan, soumitra-chatterjee-tech), "[ROCm] AMD Instinct (ROCm) enablement for kaolin"; a `rocm-enablement` branch holds one equivalent commit. It is not a MOAT project (no MOAT records or claim existed; `scaffold` accepted the name).
- Scope per its README: kaolin 0.18.0, full `kaolin._C` GPU extension (87 kernels) via torch build-time hipify plus wave64 fixes; validated on MI300X (gfx942: 15213 passed, 88 failed "all non-port", 1444 skipped) and MI350X/MI355X (gfx950, targeted subset 5618 passed, 1 failed). Diff vs upstream: 35 files, setup.py ROCm detection (`PYTORCH_ROCM_ARCH`, default `gfx942;gfx950`), a `rocm_cub_compat.h`, docs.
- Upstream: no ROCm/HIP/AMD PRs (open or closed) on NVIDIAGameWorks/kaolin; the fork has not been offered upstream. Upstream master has moved 2 commits past the fork's base.

### Gaps left by the AMD fork (for the record, not a reason to compete)
- Instinct-only: no wave32 (RDNA) or Windows validation, and the default arch list excludes RDNA.
- `kaolin.physics` and USD physics materials are wrapped in `try/except ImportError` because `warp-lang` has no ROCm build, so physics (Simplicits) is not covered.
- Footprint: the diff adds 22 new `*_gpu.cu` files (e.g. `metrics/sided_distance_gpu.cu` next to upstream's `sided_distance_cuda.cu`), which look like committed hipify-style copies rather than edits in place; not inspected further.

### Recommendation
Decline, reason `ported-elsewhere`: an AMD-owned fork (AMD-Ecosystem/kaolin) already ports the whole CUDA extension and validates it on MI300X/MI355X. The wave32/Windows coverage and the warp-dependent physics module are differentiators that should be contributed to that fork by coordinating with its owners, not a parallel MOAT port.

### Question for jeff
AMD-Ecosystem/kaolin is in our own org but is not a MOAT project. Should MOAT adopt it as an existing fork (validate wave32/Windows and review the added `*_gpu.cu` copies) in coordination with its authors, or leave it to them?
