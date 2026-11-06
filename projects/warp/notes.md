# warp notes

## Intake screen (2026-11-05)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: partial, AMD fork AMD-Ecosystem/warp).

### Licence
- `licenses.py check NVIDIA/warp`: Apache-2.0, tier 1. Recorded `license_spdx = Apache-2.0`.
- No per-file `scan-nvidia` run: the recommendation is a decline, so nothing is forked or offered upstream.

### Existing AMD support
- https://github.com/AMD-Ecosystem/warp -- fork of NVIDIA/warp, created 2025-12-17, default branch `amd-integration`; `ROCm/warp` redirects to it (the repository was transferred from the ROCm org; history includes "Merge pull request #10 from ROCm/sync/upstream-v1.13"). Branches: `amd-integration`, `amd-integeration-dev` (sic), `amd-integration-halo` (gfx1151), `release/1.13.0.amd0`, `docs/26.10`. Pushed 2026-11-05.
- Owner: an AMD team, not MOAT. Commits and PRs are by AMD engineers (rtmadduri drives most of it; zhihuidu-amd added the rocWMMA MFMA `tile_matmul` path; tomasthoresen has an open PR #21 updating the gfx1151 `amd-integration-halo` branch to Warp v1.18.0 with conditional-graph emulation). It ships ROCm docs for the 26.10 release (issues #25/#27), a Dockerfile, cuBQL integration, HIP graph support, MI355X unit-test fixes. Authoritative and actively maintained.
- Upstream NVIDIA/warp has merged small HIP-friendly fixes from contributors (#1702 HIP mempool for graph capture, #1865 tile warp-scan constants from `WP_TILE_WARP_SIZE`) but closed the WIP HIP shim PRs (#1770, #1932); the HIP port lives in the AMD fork, not upstream.

### Remaining gaps (from the IsaacLab port, `port/IsaacLab` notes "AMD Warp and dependency gaps")
1. Device textures on HIP (`wp.Texture2D` -> Warp CUDA error 801 in `wp_texture_create_device`): KNOWN to the AMD Warp team. Open issue AMD-Ecosystem/warp#20 (shailensobhee, 2026-08-29, "ROCm capability gaps for downstream mujoco_warp/mjlab: hipArray textures and conditional-graph gating via is_cuda"), and the fork's own test suite excludes HIP from texture tests (`warp/tests/cuda/test_texture.py:2885`).
2. `nextafterf` on HIP: NOT found in the fork's issues (searched "nextafter"). Still present on `amd-integeration-dev` (tip 2026-09-02): `warp/native/cuda_crt.h:1096` maps `nextafterf` to `__nv_nextafterf` (and `:1095` `nextafter` to `__nv_nextafter`), which have no HIP definition, so HIPRTC link fails with `undefined hidden symbol: nextafterf`. A likely fix maps them to `__ocml_nextafter_f32` / `__ocml_nextafter_f64`.

Both gaps are fixes to the AMD fork, not a reason for a parallel MOAT port: contribute to AMD-Ecosystem/warp (an issue or PR there, which is a person's call), not compete with it.

### Recommendation
Decline, SKIP_REASON `ported-elsewhere`: AMD's own Warp port (AMD-Ecosystem/warp, formerly ROCm/warp) covers it and is actively maintained. The two gaps IsaacLab found belong to that fork: textures are already tracked there (#20); `nextafterf` is not, and is worth reporting to that team.
