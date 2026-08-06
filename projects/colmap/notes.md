# colmap notes

## Colmap -- external community port (tracking entry) 2026-06-10

Colmap is MOAT's FOUNDATIONAL reference port (Strategy A in
PORTING_GUIDE: enable_language(HIP) + a single cuda_to_hip.h compat header). It is NOT a
MOAT-pipeline port -- it is tracked here for visibility.

- Main ROCm PR: colmap/colmap#4420 "Add ROCm/HIP support for patch_match_stereo (AMD GPU)",
  author iShengnan (NOT jeff); jeff contributed substantially. OPEN, base main, +349/-49.
  Validated on MI250 (gfx90a) per the port's testing.
- jeff is a contributor, not the PR author.
- Follow-up (jeff's, more feature support): AMD-Ecosystem/colmap:rocm-sift-gpu @ e41e06e0 -- adds
  SIFT GPU ROCm support; will become a separate PR.
- Lessons already in PORTING_GUIDE from colmap: CuTexObj rule-of-five handle bug, ComputeDOG
  out-of-bounds-read bug.

TODO: (1) address the review questions on PR #4420 (jeff helping); (2) open the rocm-sift-gpu
follow-up PR when ready.

## PR #4420 review status (reviewed 2026-06-10)

PR is MERGEABLE, REVIEW_REQUIRED (needs ahojnnes's approving review). jeff's PR#1 (the
colmap-model rework) + PR#2 (CI + regression fixes), both merged into iShengnan's fork,
resolved nearly all threads:
- ANSWERED: cuda.cc:71 device-sort regression (fixed in #2), gpu_mat.h:195 (void)cudaFree
  rationale (no throw from dtor), CMakeLists.txt:451 HIP_ENABLED option (#2), README.md:31
  reword (#2), NevesLucas rocm-sdk auto-detect (deferred to a follow-up; he was amenable).
- gemini-bot CMake nits (hardcoded /usr/include paths, absolute .so/librocrand paths,
  redundant block): OBSOLETE -- the #1 rework switched to imported targets (hip::host,
  hip::hiprand, roc::rocrand). A one-line "addressed in the rework" would close the bot threads.
- OUTSTANDING (substantive): ahojnnes "tested e2e / CUDA-HIP equivalent?" -- iShengnan answered
  honestly "not directly compared" (HIP runs patch_match_stereo e2e on a 109-image set + suite
  passes, but no side-by-side CUDA numerical diff). Draft reply prepared (pure backend
  substitution, no algo change; COLMAP GPU suite passes on gfx90a MI250 + gfx1100 7900XTX; e2e
  produces valid reconstructions; bit-exact CUDA-vs-HIP is not a meaningful gate since FP
  reductions/atomics reorder across GPUs; offer AMD depth-map stats for the maintainer to compare
  vs a CUDA run). NOT posted (jeff: move on).

NEXT when resumed: (1) post the equivalence reply to ahojnnes (jeff's call) + the gemini "addressed"
closes; (2) the AMD-Ecosystem/colmap:rocm-sift-gpu follow-up PR (SIFT GPU).

## rocm-sdk auto-detect: NevesLucas feedback, requested IN this PR (2026-06-22)

ahojnnes (2026-06-22) asked to address NevesLucas's FindDependencies.cmake:67 comment in PR #4420
itself, not a follow-up. NevesLucas wanted non-default ROCm installs (pip/venv via AMD TheRock)
supported: query `rocm-sdk path --root` for the install root and `rocm-sdk targets` to populate
CMAKE_HIP_ARCHITECTURES automatically.

DONE (local, committed, NOT pushed/PR'd -- awaiting jeff approval + workflow scope):
- Branch AMD-Ecosystem/colmap `rocm-sdk-autodetect` @ f59823d, based on iShengnan:rocm-support 7cc6049
  (the current PR #4420 head). Clone at projects/colmap/src.
- cmake/FindDependencies.cmake HIP block: ROCM_PATH precedence -DROCM_PATH > $ROCM_PATH >
  `rocm-sdk path --root` (best-effort, checks exit 0 + IS_DIRECTORY) > /opt/rocm. Arch detection
  only when CMAKE_HIP_ARCHITECTURES unset after enable_language(HIP)'s own GPU autodetect:
  `rocm-sdk targets` (prints a Python list ['gfx1100', ...]; parsed with REGEX MATCHALL
  gfx[0-9a-fA-F]+, format-agnostic) else the prior static gfx90a;gfx942;gfx1100. Absent rocm-sdk
  the configure is byte-identical to before.
- doc/install.rst: parallel paragraph documenting the auto-detection + precedence.
- rocm-sdk CLI contract verified from TheRock source: `path` needs one of --root/--cmake/--bin
  (requires rocm[devel], else exit 1); `targets` prints a Python-list repr of gfx IDs (needs only
  rocm-sdk-core). Real research summary captured in this session.

VALIDATION: change is build-config only, provably inert where rocm-sdk is absent (all current MOAT
hosts). Verified the real colmap HIP block configures against system ROCm 7.2.1 on gfx90a
(find_package hip/hiprand/rocrand + enable_language(HIP) succeed). Also unit-tested the detection
logic in isolation with a stubbed rocm-sdk across 6 precedence/robustness scenarios (agent_space/
colmap-cmake-test). Full colmap GPU rebuild not run (Ceres/OpenCV/FLANN/glog absent on host;
disproportionate for a config-only, fallback-identical change).

SHIPPED 2026-06-22 (jeff approved push+PR+reply; token refreshed with `workflow` scope):
- Pushed AMD-Ecosystem/colmap:rocm-sdk-autodetect @ f59823d.
- Opened iShengnan/colmap#4 (head jeffdaily:rocm-sdk-autodetect -> base rocm-support); once
  iShengnan merges it, the commit auto-appears in upstream #4420.
- Posted threaded reply on the #4420 FindDependencies.cmake:67 thread
  (discussion_r3456996618) confirming the rocm-sdk auto-detect + precedence, linking #4.

NEXT: watch for iShengnan to merge #4 into rocm-support and ahojnnes to approve/merge #4420.

## PR ownership (2026-07-02)

Upstream PR colmap/colmap#4420 is owned by GitHub user iShengnan, not jeffdaily.
We collaborate by opening PRs against iShengnan's branch (no push access); they
merge, which updates the upstream PR. Do not post replies speaking for the PR
author; only respond on the upstream PR when it is helpful and iShengnan has
not addressed the concern themselves. Status: approved by ahojnnes 2026-06-23;
iShengnan handling final review comments.
