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

## Intake re-screen 2026-08-06 (post-merge of #4420)

Re-screened as a fresh candidate after upstream squash-merged the dense-reconstruction ROCm work. Screened against upstream 711b23a79 (2026-08-06); the sections above are the pre-merge history and stay as provenance.

### Licence (fact, established by reading)

COLMAP itself is **BSD-3-Clause** ("new BSD", Copyright ETH Zurich and UNC Chapel Hill), read from COPYING.txt and repeated verbatim in doc/license.rst. Tier 1, cleared to contribute. GitHub's licence field says NOASSERTION for both upstream and the fork -- `utils/licenses.py check` therefore reports tier 4, which is wrong; it is the NOASSERTION-parse case the intake procedure warns about, not a restrictive licence.

There are no git submodules. `src/thirdparty` is vendored and separately licensed, which COPYING.txt states explicitly ("Building COLMAP with these dependencies may affect the resulting COLMAP license"):

- `SiftGPU` -- UNC Chapel Hill, **non-commercial**: use/copy/modify/distribute "for educational, research and non-profit purposes". Tier 4 read on its own.
- `LSD` -- **AGPL-3.0**. Tier 3. `LSD_ENABLED` defaults ON.
- `Symforce-Caspar` -- Apache-2.0. `VLFeat` -- BSD-2-Clause. `PoissonRecon` -- MIT.

This is per-part licensing that the project documents, not ambiguity, so it is recorded rather than escalated. But the scoping consequence is sharp and is the reason it is spelled out here: **the remaining work targets SiftGPU, the one non-commercial component.** Under MOAT's scope rule that changes nothing about contributing -- a non-commercial licence bars USE while leaving contribution unremarkable, and we would be contributing to COLMAP's vendored copy under COLMAP's own contribution process. It does mean a COLMAP binary built with GPU SIFT is not something AMD may ship or depend on without a separate legal answer, and with LSD_ENABLED ON by default a stock build is AGPL-encumbered besides. `approval_scope` is recorded as `contribute-only` for exactly this reason. If COLMAP is ever headed for USE rather than contribution, that is a new question and this paragraph is the reason.

`utils/licenses.scan_nvidia` over the whole tree returns no file carrying NVIDIA proprietary licence text (matched on licence text, not copyright lines).

### Duplicate effort

The existing AMD port is OURS, and it is already upstream:

- `colmap/colmap#4420` "Add ROCm/HIP support for patch_match_stereo (AMD GPU)" -- **merged 2026-08-05** as squash commit `b09267a21` (+401/-49), author iShengnan, jeff a direct contributor.
- Upstream README.md:31 now says "AMD GPUs are supported via HIP/ROCm when building from source", and doc/install.rst carries a full ROCm build section including the `rocm-sdk` auto-detection we contributed. That README line is the "notable forks"-equivalent signal, and it points at our own work.

No other AMD-Ecosystem or ROCm-org colmap effort exists. `AMD-Ecosystem/colmap` exists (created 2026-08-05, a fork of colmap/colmap) but carries only inherited upstream branches -- no `moat-port`, no ROCm branch. The parked ROCm branches are on the personal fork `jeffdaily/colmap`: `rocm-sift-gpu`, `rocm-sdk-autodetect`, `rocm-support*`, `rocm-cdna-layered-texture-bilinear`.

So this is not "port from scratch" and not "already supported" either. It is **finish a partially-landed port**: dense reconstruction is done and merged, feature extraction/matching is not.

### Viability

Genuinely CUDA, and the gap is real. 489 `.cu`/`.cuh` files, in three groups:

1. `src/colmap/mvs` (4 files) -- **already ported and merged**. Strategy A shape: `src/colmap/util/cuda_to_hip.h` compat header, `.cu` marked `LANGUAGE HIP`, `HIP_ENABLED` CMake option mutually exclusive with `CUDA_ENABLED`.
2. `src/thirdparty/SiftGPU/ProgramCU.cu` (1 file, ~1980 lines) plus its `CuTexImage`/`PyramidCU`/`SiftMatchCU` host companions -- **CUDA-only upstream**, gated `if(CUDA_ENABLED)` with `CUDA::cudart` + `CUDA::curand`; zero occurrences of "hip" in the whole directory. This is GPU SIFT feature extraction and matching, and it is the single most-used GPU path in COLMAP for a typical SfM run. On an AMD build today it silently falls back to CPU/OpenGL SIFT. This is the candidate work, and the prior attempt got far enough to name three concrete faults: a `CuTexObj` rule-of-five handle bug, a `ComputeDOG` out-of-bounds read, and tex2D pitch alignment needing a linear binding on HIP (the first two are already promoted into the `cuda-to-rocm` skill's fault-classes).
3. `src/thirdparty/Symforce-Caspar` (484 generated `.cu`) -- CASPAR-accelerated bundle adjustment, Apache-2.0, `CASPAR_ENABLED` defaults OFF and `CASPAR_ENABLED AND NOT CUDA_ENABLED` is a hard `FATAL_ERROR`. This surface did not exist when #4420 was written, so COLMAP's CUDA footprint GREW after our merge. It uses `cub::` in 12 places and otherwise looks like generated elementwise/reduction kernels. Flagged for the planner as a separate, much larger question -- default-OFF and generated code make it a poor fit for the same change as SIFT.

No MOAT-project dependencies. COLMAP FetchContents faiss v1.14.1 with `FAISS_ENABLE_GPU OFF`, so the MOAT `faiss` port is not needed; Symforce-Caspar is vendored generated code, unrelated to the MOAT `symforce` project. `depends_on` stays empty.

Upstream is emphatically alive: 12.4k stars, 2105 forks, not archived, last push 2026-08-06, 707 open issues, and a maintainer (ahojnnes) who reviewed and merged our ROCm PR. A follow-up PR has a real destination, and the merged #4420 is proof the route works.

### The one thing the planner must not get wrong

The parked branch `jeffdaily/colmap:rocm-sift-gpu` @ `e41e06e0b` **cannot be rebased onto current upstream.** Its first four commits (`658f8b563`, `786e09635`, `e609b9f13`, `def43b234`) are inside the squash `b09267a21`, and #4420 then continued for five more commits the branch never saw. Rebasing replays content that is already upstream and will conflict throughout. The SIFT work must be re-derived: `bf064e920` (enable GPU SIFT under HIP), `e95eb3806` (double-destroy + DoG edge OOB), `3345a9819` (tex2D via linear binding for pitch alignment) touch files #4420 never modified and should transplant cleanly, while `690348f33` (gpu_mat_test + version banner) and the two docs commits (`566e4df7e`, `e41e06e0b`) are likely superseded -- doc/install.rst already documents the ROCm build and says the HIP backend "currently accelerates dense reconstruction (patch_match_stereo)", which is the exact line a SIFT PR would update.

### Verdict

Worth taking up. Tier-1 licence, a live and receptive upstream, a merged precedent PR from the same effort, a well-scoped remaining gap (one 2k-line `.cu` plus three host files), and three of the hard faults already diagnosed. The fork already exists, so the state is `screened` rather than `awaiting-fork` (`unclaimed -> awaiting-fork` is not a legal transition in moatlib, and `release-forks` would advance it immediately anyway).

Carry into planning: re-derive, do not rebase; keep Caspar out of scope unless deliberately chosen; and note the fork's default branch is a plain upstream mirror, so `moat-port` starts from current upstream main.

## Intake verification 2026-08-07 (fresh screen, independent re-check)

Re-ran the whole checklist from scratch against upstream `711b23a7994f9a6b31bf88245b412838370f29c7` (2026-08-07) rather than trusting the section above; every fact below was pulled directly (a shallow clone plus `gh api`), not copied from the earlier write-up. It corroborates that section almost exactly, with two small numeric confirmations it had left unverified.

**Licence.** `utils/licenses.py check colmap/colmap` reports `NOASSERTION` -> `UNPARSED`, confirming this is the GitHub-can't-parse case, not a restrictive licence. Read `COPYING.txt` directly: COLMAP itself is **BSD-3-Clause** ("new BSD", Copyright ETH Zurich and UNC Chapel Hill), also present verbatim in `doc/license.rst`. `config/licenses.toml` places `BSD-3-Clause` in tier 1. Recorded in `status.json.license_spdx`. No `.gitmodules` entries (file exists, empty) -- no git submodules to recurse into. `src/thirdparty` is vendored and separately licensed per `COPYING.txt`'s own disclosure; read each `LICENSE` file: `SiftGPU` is UNC Chapel Hill non-commercial ("educational, research and non-profit purposes", no fee) -- tier 4 standalone; `LSD` is AGPL-3.0 -- tier 3, and `LSD_ENABLED` defaults ON in `src/thirdparty/CMakeLists.txt`; `Symforce-Caspar` is Apache-2.0 (confirmed the actual `Apache License Version 2.0` text, not just a filename guess); `PoissonRecon` and `VLFeat` carry their own `LICENSE`/`LICENSE.txt` files (MIT- and BSD-2-Clause-family respectively per prior review; not re-transcribed here since neither is CUDA-bearing and neither gates this screen). `licenses.scan_nvidia()` run over the full shallow clone (not just `grep NVIDIA`) returns zero hits -- no file anywhere in the tree carries NVIDIA proprietary licence text. Per MOAT's contribution-only scope this is per-part licensing the project documents, not ambiguity, so it does not need escalation; `approval_scope: contribute-only` still applies since a build with GPU SIFT (non-commercial) or default `LSD_ENABLED` (AGPL) is not something AMD may ship/use without a separate legal answer.

**Duplicate effort.** `colmap/colmap#4420` "Add ROCm/HIP support for patch_match_stereo (AMD GPU)" verified via `gh pr view`: state MERGED, merged 2026-08-05T14:08:24Z, merge commit `b09267a21`, author `iShengnan`. `README.md:31` and `doc/install.rst:122-151` (read directly from the clone) document the HIP/ROCm build path we contributed, including the `rocm-sdk` auto-detect precedence, and state plainly that "The HIP backend currently accelerates dense reconstruction" -- upstream's own docs are the signal, and they point at our own prior work, not a third party's. No AMD-Ecosystem or ROCm-org effort exists independent of ours: `AMD-Ecosystem/colmap` is a fork (created 2026-08-05) whose branch list is an unmodified upstream mirror (`release/*`, `sarlinpe/*`, `lpanaf/*`, etc.) -- no `moat-port` branch, no `rocm-*` branch. The parked follow-on work (`rocm-sift-gpu`, `rocm-sdk-autodetect`, `rocm-support*`, `rocm-cdna-layered-texture-bilinear`) lives only on the personal fork `jeffdaily/colmap`, confirmed present via `gh api repos/jeffdaily/colmap/branches`. A `gh search repos colmap` sweep of the top 30 hits surfaces derivative/wrapper projects (pycolmap, colmap-docker, colmap_utils, etc.) and no independent AMD/ROCm port. So: no other team is duplicating this, and what exists is entirely our own prior effort continuing.

**Viability.** Confirmed 489 `.cu`/`.cuh` files in three groups, matching the prior count exactly: (1) `src/colmap/mvs` (4 files, `gpu_mat_test.cu`, `patch_match_cuda.cu`, `gpu_mat_prng.cu`, `gpu_mat_ref_image.cu`) -- already ported, `LANGUAGE HIP` set in `src/colmap/mvs/CMakeLists.txt` under `if(HIP_ENABLED)`, top-level `CMakeLists.txt:41` defines `HIP_ENABLED` mutually exclusive with `CUDA_ENABLED`. (2) `src/thirdparty/SiftGPU/ProgramCU.cu` (the only `.cu` in that directory) -- `grep -r hip` across the whole `SiftGPU` dir returns zero matches, and `CMakeLists.txt:12` gates it strictly `if(CUDA_ENABLED)`; this is the unported gap. (3) `src/thirdparty/Symforce-Caspar` (484 generated `.cu`/`.h` files, `CASPAR_ENABLED` OFF by default at `CMakeLists.txt:77`, hard `FATAL_ERROR` if `CASPAR_ENABLED AND NOT CUDA_ENABLED`) -- confirmed 12 occurrences of `cub::` across 4 generated kernel files, so it is a real (if default-off) NVIDIA-CUB dependency, separate and much larger question than SIFT. No MOAT-project dependency: `src/thirdparty/CMakeLists.txt:56-73` FetchContents faiss v1.14.1 with `set(FAISS_ENABLE_GPU OFF)` explicitly, so the MOAT `faiss` port is not on the critical path; Caspar's vendored generated code has no relation to the MOAT `symforce` project. `depends_on` stays `[]`. Upstream activity confirmed live via `gh api repos/colmap/colmap`: not archived, pushed 2026-08-06T18:15:33Z, 12.4k stars, 2106 forks, 707 open issues.

Nothing in this pass contradicts the 2026-08-06 write-up; the `Verdict` and `The one thing the planner must not get wrong` sections above stand as written and this section only adds independent confirmation of the load-bearing facts (licence texts, PR-merge state, `.cu` counts, `cub::` count, `FAISS_ENABLE_GPU`).
