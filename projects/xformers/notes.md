# xformers notes

## Intake screen (2026-11-06)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: partial, AMD fork).

- Licence: BSD-3-Clause, tier 1. GitHub reports NOASSERTION (`utils/licenses.py check` returned UNPARSED) because LICENSE opens with a "From xFormers:" preamble before the BSD-3-Clause text; read by hand at https://github.com/facebookresearch/xformers/blob/main/LICENSE. Only submodule is `third_party/cutlass` (NVIDIA/cutlass, BSD-3-Clause). Not cloned; no NVIDIA-file scan done because the recommendation is a decline.
- Upstream facebookresearch/xformers (active, pushed 2026-10-29) already supports ROCm: README publishes "(EXPERIMENTAL) rocm 7.1" wheels at download.pytorch.org/whl/rocm7.1; `xformers/ops/fmha/ck.py` and `ck_splitk.py` are the Composable Kernel FMHA backends; `.github/workflows/rocm_build.yml`, `rocm_ci.yml`, `rocm_docker.yml` exist (gated to the rocm/xformers repository); `tests/readme_test_on_rocm.txt`.
- AMD fork: https://github.com/ROCm/xformers (ROCm org, AMD-owned, default `develop`, pushed 2026-05-29). AMD engineers (Qianfeng Zhang et al.) develop CK FMHA there and merge it upstream in batches: upstream #1257 "Recent updates from ROCm/xformers" (merged 2025-07-09), #1196 "Merge recent changes from ROCm xformers" (2025-03-06). Most recent fork work is RDNA gfx12 FMHA (ROCm/xformers #86/#87/#89, 2026-05-28), so RDNA coverage is being handled by the AMD team.
- jeff's upstream PR #1351 "[ROCm] prepare CK sources for pytorch hipify v2 APIs" was closed unmerged 2026-04-03; jeff's closing comment: pytorch hipify v2 changes were made backward compatible, so it is no longer needed.
- Residual gaps: CUTLASS-based and FA3 paths remain NVIDIA-only (CK covers attention instead); ROCm wheels are Linux-only and labelled experimental; a newer ROCm/xformers develop batch (gfx12) may not be upstream yet. All of these belong to the AMD team that owns the fork.

Recommendation: decline, SKIP_REASON `already-supported` (upstream ships ROCm wheels and the CK FMHA backend, developed by AMD in ROCm/xformers and merged upstream regularly). `ported-elsewhere` would also fit the fork, but upstream itself carries the support.

