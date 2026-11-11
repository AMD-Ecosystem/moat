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

## Take-up rulings (2026-11-10, Jeff Daily)

Intake recommended decline (ported-elsewhere: AMD-Ecosystem/warp). Jeff Daily ruled on 2026-11-10 that MOAT takes warp up. State is deliberately not yet changed: the release-line tooling below must exist first.

Delivery: no upstream PR. NVIDIA will not accept AMD support, and upstream release branches take no patches. The deliverables are branches in AMD-Ecosystem/warp that MOAT creates and owns.

Structure, one MOAT project per line (each with its own status, lock, gates and validation evidence):
- `warp-1.14`, `warp-1.15`, ... `warp-1.19` (and each later release): the fork carries `release-1.N`, a pristine copy of upstream's branch (same SHAs, recorded `fork_default_branch`), and `release/1.N.0.amd0`, which starts as an exact copy of it and receives the AMD support on top (the line's `fork_branch`). Version `1.N.0+rocm.0`. Later upstream patch tags (`v1.N.1`) arrive as sync rounds.
- `warp` (development line): `amd-integration` against a pristine `main` mirror, kept current with upstream `main` by merges, never rebases.
- Order: walk forward, `warp-1.15` depends on `warp-1.14`, and so on; each release builds on the previous release's port plus the upstream delta. Not port-the-tip-then-backport.
- `release/1.13.0.amd0` (AMD's shipped 1.13, ROCm 26.10) is left alone. MOAT's porting starts at 1.14. The AMD 1.13 delta is reviewed first only to decide whether it is a good starting point for `warp-1.14` (review in progress; result goes in this file).
- Sign-off: a review PR from `release/1.N.0.amd0` into the pristine `release-1.N`, never merged; `/moat approve` there; approval leads to tagging `v1.N.0+rocm.0` and freezing the branch, not to an upstream publish.

Scope for every line: the unreported gaps Isaac Lab found (Warp's runtime compiler does not define `__CUDA_ARCH__`, breaking Newton's `nextafterf` and `texture.h:817`; `WP_TILE_WARP_SIZE` 64 for every target breaks wave32 tile reductions), wave32 Radeon and Windows gfx1151, all gates validated. Textures (hipArray, AMD-Ecosystem/warp#20) are in scope as a separate effort. Packaging is source/Docker build today (no pip package exists); publishing a wheel and its name is for the release owners.

Tooling needed before scaffolding (control-plane work, Jeff merges):
- `scaffold` option creating `warp-1.N` against NVIDIA/warp (project name differs from repo name);
- `set-fork-branch` setter (moatlib already reads `fork_branch` at utils/moatlib.py:121, :1837, :2542; nothing writes it);
- a delivery mode, e.g. `set-delivery <name> fork-release --by <who>`: `pr_ready` gates the in-fork review PR against the mirror; approval triggers tag + freeze instead of `upstream.py --publish`. `set-no-upstream` does not fit (it refuses a review PR and finishes without approval).

## Review of AMD's release/1.13.0.amd0 as a starting point (2026-11-10)

Source review of `v1.13.0..release/1.13.0.amd0` (31 commits). Verdict: not a good base; port `warp-1.14` from pristine `release-1.14`, using AMD's core HIP layer as reference or cherry-picked hunks. Main reasons (spot-checked in source):
- wave64 only: `tile_reduce.h:14-18` sets `WP_TILE_WARP_SIZE 64` and `tile.h:226-232` a 64-bit lane mask for every HIP target;
- Windows broken for every build including CUDA: `warp/_src/build_dll.py:143` uses `hip_enabled` inside `find_host_compiler()`, where it is undefined (it is a local of another function at :543);
- device texture sampling on HIP returns 0 (`texture.h` 672, 686, 718, 733, 768, 786) and the texture tests are dropped on HIP (`test_texture.py:2615`);
- CUDA-path regressions: HIP silently replaces CUDA when ROCm is found (`build_lib.py:449`), `wp_cuda_device_get_arch` ABI change (`warp.h:411`), lost CUDA < 12.3 guard on capture pause/resume (`warp.cu:3276`), non-atomic bf16 `atomic_add` on HIP (`builtin.h:2029`);
- about 40% noise or beyond-port work (CI workflows that delete upstream's, Docker, uv.lock, README; BVH and cuBQL rewrites, rocWMMA fast path that breaks 16x16 fp32 `tile_matmul` at block_dim 256, Cholesky);
- about 35 test files with skips, shrunk sizes or loosened tolerances to re-check.
Reusable subset: commits 527ba15c, da56e495, 7b941c21, 51b16761 and the tile_stack destructor hunk of 5b3130e9 (hip_util.h compat header, HIP branches of warp.cu/cuda_util, hiprtc compile path, `__HIP_DEVICE_COMPILE__` guards, hipcub aliasing, build plumbing). AMD's newer `amd-integration-halo` (16 commits on upstream 1.17.0.dev4) already fixes the wave32 mask, bf16 atomic, device textures and the capture guard; better reference, still Linux-only and not minimal. Carry-forward to v1.14.0: 71 of AMD's non-noise files overlap upstream's 341 changed files; 1.14 split tile.h into tile_matmul/cholesky/solve/fft headers. Full report: plan file warp-113-amd-delta-review.md.

## Base ruling (2026-11-11, Jeff Daily)

- `warp-1.14` is ported independently from the pristine `release-1.14`, not built on `release/1.13.0.amd0`. AMD's reusable 1.13 subset and `amd-integration-halo` are reference material only; credit authorship wherever hunks are taken. Every skip, shrunk size or loosened tolerance is re-derived against the new base, never carried over.
- Release packages are part of the port: each line must build its wheel (`build_lib.py` plus the project's documented packaging path) on every supported platform, Linux and Windows, versioned `1.N.0+rocm.0`, and the validator installs and tests the built wheel, not only an editable checkout. Jeff intends to host the wheels on the AMD-Ecosystem/warp GitHub project later; publishing (a GitHub release under his account) is a separate GitHub-visible action that needs his approval of the exact content.
