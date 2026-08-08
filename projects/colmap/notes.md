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

## Port implemented on linux-gfx90a 2026-08-08 (porter)

Re-derived fresh against upstream `d6d2bc8e0d3b4ba37139982d0b80f11af93deaf3` per plan.md; the
parked `jeffdaily/colmap:rocm-sift-gpu` branch was read for WHAT and WHY only, never rebased.
Fork HEAD `512dbe91fbe9321acada2628b1dd80cb8d128990` on `AMD-Ecosystem/colmap:moat-port`.

### Build recipes (this host, ROCm 7.2, MI250X gfx90a)

System packages installed for the build: `libboost-{program-options,graph,system,filesystem,test}-dev
libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev libgflags-dev
libgtest-dev libgmock-dev libsqlite3-dev libglew-dev libsuitesparse-dev libceres-dev
libopenimageio-dev openimageio-tools libcurl4-openssl-dev libssl-dev libcrypto++-dev
qt6-base-dev qt6-svg-dev libqt6opengl6-dev libgl1-mesa-dev libglu1-mesa-dev mesa-utils xvfb
libxkbcommon-dev`.

`openimageio-tools` is easy to miss and the failure is confusing: without it,
`find_package(OpenImageIO)` aborts configure with "imported target OpenImageIO::iconvert
references the file /usr/bin/iconvert but this file does not exist". The `-dev` package alone
is not enough because the exported CMake targets include the command-line tools.

Validation build (GUI, this is the one that can prove GPU SIFT):

    cmake -S . -B build-hip-gui -GNinja \
      -DCUDA_ENABLED=OFF -DHIP_ENABLED=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_BUILD_TYPE=Release -DTESTS_ENABLED=ON -DGUI_ENABLED=ON \
      -DCGAL_ENABLED=OFF -DDOWNLOAD_ENABLED=OFF -DONNX_ENABLED=OFF
    cmake --build build-hip-gui -j"$(nproc)"
    xvfb-run -a ctest --test-dir build-hip-gui -j16 --output-on-failure

Headless build is the same with `-DGUI_ENABLED=OFF` into `build-hip` and needs no `xvfb-run`.

`-DCMAKE_PREFIX_PATH=/opt/rocm` and `-DCMAKE_HIP_COMPILER=...` turned out NOT to be needed: the
`ROCM_PATH` detection already in `cmake/FindDependencies.cmake` resolves `/opt/rocm` and CMake
finds `/opt/rocm/lib/llvm/bin/clang++` on its own. Verified by configuring with neither flag.
plan.md predicted they would be required; they were not, so doc/install.rst was left alone on
that point.

### Results

Both configurations: 159/159 ctest pass, including `feature/sift_test`, `mvs/gpu_mat_test`
(the merged MVS port, no regression) and `util/opengl_utils_test`.

`sift_test` alone: 32/32, of which 15 are GPU tests, including the two in-suite GPU-versus-CPU
comparisons `MatchSiftFeaturesCPUvsGPU.Nominal` and
`MatchGuidedSiftFeaturesCPUvsGPUGuided.EssentialMatrix`.

End to end at two resolutions, one of which (640x480) is the case that used to fail the
256-byte texture pitch check:

| input | keypoints per image | matched pairs / matches | verified |
|---|---|---|---|
| 640x480 | 1037, 1047, 948 | 3 / 1419 | 3 / 1419 |
| 1024x768 | 3124, 3071, 3062 | 3 / 4740 | 3 / 4740 |

### Anti-no-op evidence (plan.md risk 4)

A green `sift_test` is worthless on its own here, so the GPU tests were proven to execute by
counting kernel dispatches rather than by trusting the result. `AMD_LOG_LEVEL=3` on the test
binary names every kernel the HIP runtime launches:

    AMD_LOG_LEVEL=3 ./sift_test --gtest_filter='ExtractSiftFeaturesGPU.Nominal'
      -> FilterH x31, FilterV x31, ComputeDOG x30, ComputeOrientation x5, ComputeDescriptor x5
    AMD_LOG_LEVEL=3 ./sift_test --gtest_filter='MatchSiftFeaturesCPUvsGPU.Nominal:...'
      -> RowMatch x10, ColMatch x9, MultiplyDescriptor x7, MultiplyDescriptorGRay x3

Same check on the end-to-end `feature_extractor` run: ComputeDOG x90, FilterH/FilterV x93,
ComputeKEY x54, InitHist x54, ListGen x113, ComputeOrientation x26, ComputeDescriptor x26.

This is a better signal than wall time. In the HEADLESS build the GPU matcher tests report
0-3 ms each, which looks exactly like the old no-op, but `AMD_LOG_LEVEL=3` shows RowMatch x7,
ColMatch x6 and MultiplyDescriptor x7 for `MatchSiftFeaturesCPUvsGPU.Nominal`: the ~200 ms per
test in the GUI build is Qt building an offscreen GL context, not GPU work. Timing alone would
have mislabelled a real run as a skipped one.

`RunThreadWithOpenGLContext` was the empty inline; it now starts and waits for the thread, and
`opengl_utils_test` (built for every configuration now, not only `GUI_ENABLED`) gained
`RunThreadWithOpenGLContext.RunsThreadBody`, which fails if the body does not run.

### CUDA no-regression

Compile-checked only; there is no NVIDIA GPU on this host, so CUDA results were not measured
and no numerical comparison was made. `nvcc 13.3.73` (conda `-c nvidia cuda-nvcc
cuda-cudart-dev libcurand-dev cuda-cccl`), `-DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=80`:
`colmap_sift_gpu`, the `colmap` executable and `sift_test` all compile and link.

The two unconditional correctness fixes (plan.md Open questions 1) therefore rest on argument
rather than on a measured CUDA run, and the reviewer should weigh them as such:

- The `ComputeDOG` clamp changes only the first/last column and row, where the old code read
  the previous row's last pixel or past the end of the buffer. Interior pixels are
  byte-identical.
- The `tex2D` to `tex1Dfetch` rebind is exact, not approximate: all three fetches are
  `cudaFilterModePoint` and all three kernels clamp x to [1.5, width-1.5] and y to
  [1.5, height-1.5] before fetching, so hardware addressing never applies, and point sampling
  over a pitch2D binding whose pitch is the packed row IS `tex1Dfetch(tex, int(y)*width+int(x))`.

### Left for the reviewer

- `src/pycolmap` guards were widened in source but pycolmap was NOT built here; it is a
  separate CMake project under `python/` consuming `find_package(colmap)`. The export list
  (`CMakeLists.txt:467`) already carried `if(CUDA_ENABLED OR HIP_ENABLED)`, and `GPU_ENABLED`
  now covers HIP, so `colmap_sift_gpu`, `colmap_util_cuda` and `colmap_mvs_cuda` are all
  exported on a ROCm build; the linkage should be fine but is unverified.
- `SiftBackend` (a file-local enum in `feature/sift.cc`) was renamed `CUDA` -> `COMPUTE`, since
  it now names either compute backend. `SiftMatchGPU::SIFTMATCH_CUDA*` is SiftGPU's own API and
  was left spelled as upstream has it.
- `CuTexImage::BindTexture2D` was deleted along with its only two callers. `InitTexture2D` /
  `CopyToTexture2D` / `_cuData2D` were already dead before this change (nothing ever bound
  `_cuData2D`; `BindTexture2D` bound the linear `_cuData` pointer) and were left alone as
  pre-existing dead code.
- plan.md risk 5 (`_IsNvidia == 0` forcing `_UseCUDA = 0` in `GlobalUtil::InitGLParam(0)`) did
  NOT materialise: the compute path reaches `InitGLParam(1)`, and the full in-suite run shows
  GPU kernels dispatching, so no widening of the vendor test was needed. It remains a latent
  hazard if the GLSL and compute paths are ever initialised in the other order.

## Review 2026-08-08 (reviewer, linux-gfx1100 / wave32 host)

Reviewed fork `512dbe91` on `AMD-Ecosystem/colmap:moat-port` against its parent
`d6d2bc8e` (single commit). Verdict: changes-requested. Nothing here says the port is
wrong; five items must land before the pull request is drafted.

Read-only review: no wave64 hardware here, so everything below is from reading the source,
not from running it. The porter's own gfx90a results were not re-measured.

**Reviewed against the fork's default branch as well.** `moat-port..origin/main` is 0, so
the branch is not behind. Note the fork's `main` is itself two upstream commits behind
`moat-port`'s parent, so `git diff origin/main...moat-port` shows unrelated upstream work
(`homography_matrix*`, `two_view_geometry*`); the port diff is `d6d2bc8e..512dbe91` only.
`jargon.py --port colmap` is clean over the whole branch. Title is 59 chars, `[ROCm]`
prefixed, Claude named, no `Co-Authored-By: noreply`, ASCII, no em-dash.

### 1. Compiling out `CuTexImage::CopyFromPBO` turns a transfer failure into silent garbage

`src/thirdparty/SiftGPU/CuTexImage.cpp:257-270`. On ROCm the body is entirely inside
`#if defined(SIFTGPU_GL_INTEROP_ENABLED)`, so the function returns having copied nothing
and having reported nothing. `PyramidCU::ConvertInputToCU`
(`src/thirdparty/SiftGPU/PyramidCU.cpp:936-941`) then does

    if(input->_rgb_converted && input->CopyToPBO(_bufferPBO, ws, hs, GL_LUMINANCE))
    {
        _inputTex->InitTexture(ws, hs, 1);
        _inputTex->CopyFromPBO(ws, hs, _bufferPBO);
    }

`input->CopyToPBO` here is `GLTexImage::CopyToPBO`, which is pure OpenGL and still
succeeds on ROCm, so the branch is taken; `InitTexture` `cudaMalloc`s without zeroing and
`CopyFromPBO` does nothing, so SIFT runs on uninitialized device memory with no
diagnostic. The other three compiled-out sites degrade correctly and are fine:
`CuTexImage::CopyToPBO` returns 0 (`CuTexImage.cpp:272-296`) so `ConvertTexCU2GL` sets
`_bufferTEX` to 0x0, and the 4-arg `CuTexImage(w,h,n,pbo)` constructor leaves `_cuData`
NULL (`CuTexImage.cpp:108-119`) which `ConvertTexCU2GL` explicitly tests
(`PyramidCU.cpp:878,886,895`). `CopyFromPBO` is the only one with no failure signal.

The reachability argument in the comment at `CuTexImage.cpp:35-42` is correct for COLMAP
and I verified it: `sift.cc:694` calls `RunSIFT(pitch, height, data, ...)`, so
`input->_pixel_data` is non-null and `PyramidCU.cpp:930` takes the
`InitTexture`/`CopyFromHost` branch. But this is vendored SiftGPU that upstream ships to
other consumers, and "correct for the one caller in this repository" is a weaker claim
than the code should rest on. Make the ROCm `CopyFromPBO` fail loudly -- a `std::cerr`
line matching the existing "Unable To Convert Intput" style at `PyramidCU.cpp:945` is
enough.

### 2. The Test Plan omits the nvcc check and does not disclose that CUDA was never run

Two of the three fixes are unconditional and therefore change what upstream's CUDA build
computes: the `ComputeDOG_Kernel` border clamp (`ProgramCU.cu:479-494`) and the pitched-to-
linear rebind (`ProgramCU.cu:991`, `ProgramCU.cu:1205`, with the three fetches at
`ProgramCU.cu:862`, `1051`, `1124`). notes.md records honestly that the CUDA path was
compile-checked with nvcc 13.3.73 and never executed, and that the equivalence rests on
argument. The commit's Test Plan lists only the gfx90a run and says nothing about either.

This is the exact question ahojnnes already asked on #4420 (`notes.md:36-38`, "tested e2e /
CUDA-HIP equivalent?"). Put the nvcc configuration in the Test Plan and state plainly that
the CUDA path is compile-checked and not run, and that the clamp changes the first and
last row and column on both backends. The skill's own instruction is to say
"compile-checked with nvcc, not run"; saying it unprompted is much cheaper than saying it
in a review round.

I did verify the two equivalence arguments by reading, and both hold:

- The clamp is exact in the interior. `index = IMUL(row, width) + col` at
  `ProgramCU.cu:475`, and the new `IMUL(row, width) + coln` etc. agree with the old
  `index +/- 1`, `index +/- width` for every non-border pixel. `IMUL` is `__mul24`
  (`ProgramCU.cu:36`), which returns the full 32-bit product of two 24-bit operands, so
  there is no new range limit.
- The `tex2D` to `tex1Dfetch` swap is exact. All three fetches are `cudaFilterModePoint`,
  the pitch was the packed row (`_imgWidth * _numChannel * sizeof(float)` in the deleted
  `BindTexture2D`), `InitTexture` allocates packed with plain `cudaMalloc`
  (`CuTexImage.cpp:163-194`, no `cudaMallocPitch`), and all three kernels clamp x to
  `[1.5, width-1.5]` and y to `[1.5, height-1.5]` before fetching (`ProgramCU.cu:846-849`,
  `1032-1035`, `1105-1108`), so the coordinates are strictly positive and `int()`
  truncation equals `floor()`.

### 3. `cuda_to_hip.h` drags a device RNG header into host translation units

`src/colmap/util/cuda_to_hip.h:48` and `:147` include `<hiprand/hiprand_kernel.h>` and
`<curand_kernel.h>` unconditionally. This change newly routes
`src/thirdparty/SiftGPU/CuTexImage.h:27`, `CuTexImage.cpp:34` and `SiftMatchCU.cpp:34`
through that header, and those are host-compiled `.cpp` files -- only `ProgramCU.cu` is
marked `LANGUAGE HIP` (`src/thirdparty/SiftGPU/CMakeLists.txt:29`). SiftGPU uses no RNG at
all; `grep -rn curand src/thirdparty/SiftGPU/` hits only the CMake `CUDA::curand` line.

The header already carries a scar from exactly this fault class: the `<cstdio>` workaround
at `cuda_to_hip.h:41-45` exists because rocrand's mtgp32 header omits an include and host
TUs tripped over it. Nothing outside a `.cu` names `curandState` (checked across
`src/colmap/`), so gating both includes behind `__HIPCC__` / `__CUDACC__` is safe and
retires that workaround.

I checked whether this is a live break rather than just exposure: CUDA's
`crt/host_defines.h:191-195` empties `__annotate__` when `__CUDACC__` is undefined, so
`curand_kernel.h` is host-includable by design on every host compiler, and a local
`g++ -std=c++17 -c` of `<cuda_runtime.h>` + `<curand_kernel.h>` succeeds. So this is
footprint and fault-class hygiene, not a broken build today.

Related and fixed by the same change: SiftGPU's HIP arm links only `hip::host`
(`src/thirdparty/SiftGPU/CMakeLists.txt:30-32`) while it transitively includes a hiprand
header, whereas `colmap_mvs_cuda` declares `hip::hiprand` and `roc::rocrand` for the same
header (`src/colmap/mvs/CMakeLists.txt:302-306`). It builds on a monolithic `/opt/rocm`,
but `cmake/FindDependencies.cmake:129-151` goes to real trouble to support split
`rocm-sdk` installs where that is not guaranteed. Gate the include and the mismatch is
moot; otherwise add `hip::hiprand`.

### 4. plan.md risk 5 is recorded as closed on evidence that only covers gfx90a

`GlobalUtil.cpp:370` still reads `if(GlobalUtil::_IsNvidia == 0) GlobalUtil::_UseCUDA = 0;`
inside `InitGLParam(0)`, and `PyramidGL.cpp:178` / `SiftMatch.cpp:136` still call it in the
same process that runs the compute tests. notes.md:262-265 records that this "did NOT
materialise". That observation is gfx90a-specific: the only OpenGL context available on a
compute-only CDNA part is Mesa llvmpipe under Xvfb, and an RDNA host has a real AMD GL
driver reporting an AMD vendor string, which reaches the same `_IsNvidia == 0`. The failure
mode is a silent fall back to GLSL with a fully green suite, so it cannot be detected from
the test result.

No code change requested. Write into notes.md that the wave32 validation must confirm the
backend by kernel dispatch (`AMD_LOG_LEVEL=3`, the same check the porter used at
notes.md:206-224), not by ctest passing, and that a GLSL fallback there is a validation
failure and not a pass.

### 5. The commit body calls the `RunThreadWithOpenGLContext` change test infrastructure

`src/colmap/util/opengl_utils.h:96-104`. Seven production call sites route through the same
function: `src/colmap/exe/feature.cc:146,218,269,297,325,353,381` and
`src/colmap/exe/sfm.cc:165`. Their reachability on a non-GUI build is genuinely narrow --
`cmake/FindDependencies.cmake:633-635` forces `OPENGL_ENABLED` off without the GUI, and
`src/colmap/feature/extractor.h:71-75` defaults `use_gpu` to false without
`COLMAP_GPU_ENABLED` -- so in practice the change is test-only. But a maintainer reading
"That is shared test infrastructure" will go and check that themselves, and one sentence
naming `exe/feature.cc` and `exe/sfm.cc` plus the reachability argument saves the round
trip.

### Checked and clean

Recorded so the next reviewer does not repeat it.

- **Wavefront portability, the thing this host is best placed to judge and worst placed to
  test.** `ProgramCU.cu` has no `__shfl*`, no `__ballot`, no `__activemask`, no
  `__syncwarp`, no `warpSize` and no lane mask of any kind. The three literal 32s
  (`ROWMATCH_BLOCK_WIDTH`, `COLMATCH_BLOCK_WIDTH`, `FILTERV_BLOCK_HEIGHT` at
  `ProgramCU.cu:1890`, `1963`, `50`) are block dimensions; no shared array is sized from a
  warp count. `RowMatch_Kernel`'s tree reduction (`ProgramCU.cu:1929-1941`) puts
  `__syncthreads()` outside the `if(threadIdx.x < step)` at every step, so it is correct
  at either width and does not rely on warp-synchronous execution. Every early `return` in
  every kernel follows that kernel's last `__syncthreads()` -- checked one by one at
  `ProgramCU.cu:154/155`, `212/214`, `1456/1459`, `1574,1579/1580`, `1758,1765/1766` -- so
  the intra-wave barrier-divergence class does not apply. Largest static shared allocation
  is FilterV's, about 10 KB, and the largest block is 512 threads (16x32), both fine on
  RDNA. plan.md risk 6 is correct as written.
- Strategy A is implemented as the shape already in the tree: one `cuda_to_hip.h`,
  no second HIP-aware header, `enable_language(HIP)` central, `LANGUAGE HIP` on
  `ProgramCU.cu` rather than a rename, CUDA spelling preserved in the sources.
- `CuTexObj` rule-of-five (`CuTexImage.h:48-71`) is right: NSDMI `handle = 0`, deleted
  copies, move ctor and move assignment that null the source, guarded `Destroy()`. Every
  use in `ProgramCU.cu` is either copy-init from a prvalue or move-assign from one; nothing
  copies. The two default-constructed-and-never-bound cases (`texObjList` on the
  existing-keypoint path, `texObjF4` when `_SubpixelLocalization` is 0) now pass handle 0
  instead of stack garbage, and the kernel does not dereference either under COLMAP's
  settings (`_KeepExtremumSign` is only set by SiftGPU's `-sign` argument, which COLMAP
  does not pass, and `_SubpixelLocalization` defaults to 1).
- Linear-bind sizing is safe: `BindTexture` uses `_numBytes` (`CuTexImage.cpp:76`), and
  `InitTexture` only grows the allocation (`if(size <= _numBytes) return true;`), so the
  binding is never smaller than the current image.
- No hardware-linear-filter assumption anywhere; both static `cudaTextureDesc` singletons
  are `cudaFilterModePoint`.
- No library substitution is needed or made; SiftGPU uses no cuBLAS/cuFFT/cuRAND/CUB.
- Dispatch-site widening is complete. The only remaining `COLMAP_CUDA_ENABLED`-alone sites
  are the deliberate scope-outs (`onnx_utils.cc:145`, `bundle_adjustment_ceres.cc:141`,
  `global_positioning.cc:361`, and their test), the `#endif` label comments in
  `exe/mvs.cc`, the compat header's own `#elif`, and `cuda.cc:56` where the guard covers
  only the CUDA-specific no-device error codes.
- `extractor.cc:139` dropping `&& !defined(COLMAP_CUDA_ENABLED)` is behaviour-preserving:
  `COLMAP_GPU_ENABLED` is defined whenever `CUDA_ENABLED` is
  (`cmake/FindDependencies.cmake:644`).
- Generalizable lessons: already present in the skill on `main` and each one checked
  against this source rather than against the porter's summary. The `AMD_LOG_LEVEL=3`
  dispatch-count method and its "wall time is not reliable" refinement
  (`references/validation.md:12-44`), the pitched-to-linear exactness rule with its two
  preconditions (`references/fault-classes.md:224-234`), the ComputeDOG clamp and the
  CuTexObj rule-of-five entries (`references/fault-classes.md:148-155`) all hold. The
  640x480 arithmetic in the pitch entry checks out: the 80-wide float2 level is a 640-byte
  row, not a multiple of 256. This branch adds no skill entries.

## Review response on linux-gfx1100 2026-08-08 (porter, wave32 host)

All five review items addressed. Everything below was measured on this host (Radeon Pro
W7800, gfx1100, ROCm 7.2.3, GPU index 2) against the amended tree, not carried over from
the gfx90a run. The wavefront analysis was NOT redone; the reviewer's reading stands and
plan.md risk 6 is unchanged.

### Build recipe additions for this host

The gfx90a package list in the section above is complete except for **`libopencv-dev`**,
which is not a COLMAP dependency at all: `find_package(OpenImageIO)` fails configure with
`Imported target "OpenImageIO::OpenImageIO" includes non-existent path
"/usr/include/opencv4"` because Ubuntu's OpenImageIO exports an interface include
directory it does not itself install. Same class of trap as the `openimageio-tools` note.

### Item 1: `CuTexImage::CopyFromPBO` now fails loudly

`CuTexImage.cpp:257-278`. The ROCm arm gained an `#else` that writes
`Unable To Copy From PBO: this build has no pixel buffer object interop` to `std::cerr`
and then `cudaMemset`s `_cuData` to zero, so the caller gets a diagnostic and a defined
buffer rather than uninitialized device memory. The `#else` is inside
`#if defined(SIFTGPU_GL_INTEROP_ENABLED)`, which is defined on every CUDA build, so the
CUDA object code is untouched.

Verified directly rather than by inspection, because COLMAP itself cannot reach the call:
`agent_space/pbo_check.cpp` links `libcolmap_sift_gpu.a`, fills a 64x64 CuTexImage with
12345.0f, calls `CopyFromPBO(64, 64, 0)` and reads back.

    before CopyFromPBO: out[0]=12345 out[n-1]=12345
    Unable To Copy From PBO: this build has no pixel buffer object interop
    after  CopyFromPBO: out[0]=0 out[n-1]=0 sum=0

### Item 4: what `_UseCUDA` actually resolves to on RDNA -- risk 5 is closed by tracing

**`_UseCUDA` is 1 and stays 1 for the whole of `sift_test` on this host.** The result is
much stronger than "did not materialise": the `_IsNvidia == 0` clearing at
`GlobalUtil.cpp:370` is UNREACHABLE from COLMAP, and it is unreachable for a structural
reason, not because of anything about the vendor string or the GPU.

Traced under gdb on the Release binary (no debug info needed; `GlobalParam::_UseCUDA`,
`_IsNvidia` and `_GoodOpenGL` are ordinary global symbols, and `GlobalUtil::InitGLParam`,
`PyramidCU::PyramidCU`, `PyramidGL::PyramidGL` are in the symbol table). Over one full
`xvfb-run` of `sift_test`:

| probe | count | value |
|---|---|---|
| `GlobalUtil::InitGLParam` entered | 1 | `NotTargetGL=1`, `_UseCUDA=1`, `_GoodOpenGL=-1` |
| `PyramidCU::PyramidCU` (compute backend) | 1 | -- |
| `PyramidGL::PyramidGL` (GLSL backend) | **0** | -- |
| `glGetString` | 16 | every one is `GL_VERSION` (0x1f02); `GL_VENDOR` (0x1f00) **never** |
| writes that change `_UseCUDA` (hardware watchpoint) | 1 | `0 -> 1` |

`InitGLParam(1)` takes the early branch at `GlobalUtil.cpp:326-328` (`NotTargetGL &&
!_UseSiftGPUEX`), sets `_GoodOpenGL = 1` and returns before `glewInit()`, so the vendor
string is never read and line 370 never runs. `PyramidGL` is never constructed at all,
because COLMAP always passes `-cuda <index>` (`sift.cc:588`), so the second-call hazard
plan.md risk 5 describes needs a GLSL user that COLMAP does not contain.

Two traps worth recording:

- **Reading a global after the inferior exits gives the ELF initial value, not the last
  live value.** A first pass printed `AT-EXIT: _UseCUDA=0 _GoodOpenGL=-1` and that looked
  exactly like the fallback the reviewer predicted. Those are the initializers at
  `GlobalUtil.cpp:48` and `:99`: gdb was reading the `.data` image from the file after the
  process was gone. The watchpoint is the honest instrument; a post-exit read is not.
- **An `LD_PRELOAD` shim on `glGetString` does not intercept Qt.** Qt resolves GL through
  `QOpenGLFunctions`/`eglGetProcAddress`, so a preloaded `glGetString` is bypassed. It
  would work for SiftGPU, which calls the symbol directly, but a breakpoint proves more
  with less setup and no positive control needed.

For the record on what GL is actually available here: `xvfb-run` gives Mesa llvmpipe
(vendor `Mesa`), same as the gfx90a host. A real AMD driver IS reachable on this machine
via surfaceless EGL -- `EGL_PLATFORM=surfaceless QT_QPA_PLATFORM=eglfs` yields vendor
`AMD`, renderer `AMD Radeon Pro W7800 48GB (radeonsi, navi31)` -- but COLMAP cannot use
it: `OpenGLContextManager` asks for a 2.1 CompatibilityProfile
(`opengl_utils.cc:46-51`), and eglfs offers GLES, so `QOpenGLContext::create()` fails with
EGL 3009. So on this host COLMAP's GL context is llvmpipe either way, and per the trace
above that makes no difference to the backend choice.

### Item 3: the requested gate is NOT safe, and this was proven, not argued

The review asked to gate `<hiprand/hiprand_kernel.h>` / `<curand_kernel.h>` in
`cuda_to_hip.h` behind `__HIPCC__` / `__CUDACC__`. **That breaks the build.** The review's
premise -- "nothing outside a `.cu` names `curandState`" -- holds for `.cc`/`.cpp` files
but not for headers: `gpu_mat.h:137` declares
`FillWithRandomNumbers(..., const GpuMat<curandState>&)` and `gpu_mat_prng.h:37` declares
`class GpuMatPRNG : public GpuMat<curandState>`, and `patch_match.cc` (host-compiled by
`/usr/bin/c++` into `colmap_mvs_cuda`) reaches both through
`patch_match_cuda.h -> cuda_texture.h -> gpu_mat.h`. Built the gated header into an
override include directory and compiled the real TU with the real compile command:

    gpu_mat.h:137:43: error: 'curandState' was not declared in this scope
    gpu_mat_prng.h:37:34: error: 'curandState' was not declared in this scope

(unmodified header: 0 errors). A forward declaration does not rescue it either --
`GpuMat<curandState>` needs `sizeof(T)` in host code, and `hiprandState` is a typedef of
`rocrand_state_xorwow`, not a class that can be forward-declared portably.

So the *other* remedy the review offered was taken, and it is the one that fixes the only
thing that can actually fail: `src/thirdparty/SiftGPU/CMakeLists.txt` now lists
`hip::hiprand` and `roc::rocrand` next to `hip::host`. That mirrors upstream's own CUDA
arm, which has listed `CUDA::curand` for `colmap_sift_gpu` since before this port, and it
matches `colmap_mvs_cuda`. The declared dependency now covers the header the target
actually includes, which is what would have broken on a split `rocm-sdk` install where
hiprand headers are not on `hip::host`'s include path. A comment at `cuda_to_hip.h:47-50`
records why the include cannot be narrowed, so the next reader does not retry it.

Footprint, for the record: the RNG header is 24.6k of the compat header's 56.0k
preprocessed lines. Real, but not worth an opt-out macro plus per-target CMake plumbing
that makes an installed header non-self-contained.

### Items 2 and 5: commit message

Commit amended (no arch had a `validated_sha`, so nothing was orphaned). The Test Plan now
carries the nvcc configuration, run again on THIS host against the amended tree
(nvcc 13.3.73 from conda `-c nvidia cuda-nvcc cuda-cudart-dev libcurand-dev cuda-cccl`
into `agent_space/cuda`; `colmap_sift_gpu`, `colmap_mvs_cuda`, `colmap_feature_sift_test`
and `colmap_main` all compile and link), a plain statement that the CUDA path was compile-
checked and not executed with no numerical comparison made, and the border-only scope of
the ComputeDOG clamp. The `RunThreadWithOpenGLContext` paragraph now names the eight
production call sites (`exe/feature.cc` x7, `exe/sfm.cc` x1) and gives the reachability
argument instead of calling the function test infrastructure.

### Test results on gfx1100

Full suite: **159 of 159 pass**, `xvfb-run -a ctest -j4`, 11.6 s wall. (The 1501 s of the
`-j16` attempt was entirely the Mesa deadlock below, not slowness -- everything except the
hung test finishes in seconds at any `-j`.)

`sift_test` standalone: 32 of 32 in 2.7 s. Kernel dispatches for that run, by
`AMD_LOG_LEVEL=3`: FilterH x31, FilterV x31, ReduceHist x45, ComputeDOG x30, RowMatch x25,
ColMatch x24, ComputeKEY x18, InitHist x18, ListGen x14, MultiplyDescriptorGRay x12,
MultiplyDescriptor x9, NormalizeDescriptor x5, ComputeOrientation x5, ComputeDescriptor x5,
DownsampleKernel x5, MultiplyDescriptorG x4, UpsampleKernel x1.

**`sift_test` deadlocks in Mesa, not in COLMAP, when the suite shares one Xvfb display.**
Under `xvfb-run -a ctest -jN` it hung at `MatchGuidedSiftFeaturesGPU.TypeMismatch` twice
out of two (reported as `Timeout` at ctest's default 1500 s cap the first time), while the
same binary passes standalone 5 times out of 5 in under 3 s. The stack of the hung process
is not COLMAP's:

    #3  __pthread_clockjoin_ex
    #4-#8  libgallium-25.2.8.so
    #9-#10 libGLX_mesa.so.0
    #11 XCloseDisplay

`RunGpuTest` builds a `QApplication` per test, so each test opens and closes the X display,
and llvmpipe's thread pool deadlocks joining its workers on close when other GL clients are
live on the same display. Isolating the variables: sift_test alone on a **pre-existing
shared** `Xvfb :77` passes (2.9 s), so it is concurrency, not the shared display; and the
GPU is not involved at any point. `-j4` is clear of it here and runs the whole suite in
11.6 s; `-j8` and `-j16` both hung.

## Review 2026-08-08 (reviewer, linux-gfx1100 / wave32 host, round 2)

Reviewed fork `4c531f5e` against parent `d6d2bc8e`. Verdict: **changes-requested**, and
none of it is a code change. The code is right: all five previous items are genuinely
addressed, the build is current with the tree, and I re-measured the suite (159/159 at
`-j4`, 12.3 s) and the kernel dispatches (ReduceHist x45, ComputeDOG x30, RowMatch x25,
ColMatch x24, ComputeKEY x18, InitHist x18, ListGen x14, MultiplyDescriptorGRay x12,
MultiplyDescriptor x9, NormalizeDescriptor x5, ComputeOrientation x5, ComputeDescriptor x5,
MultiplyDescriptorG x4), so the GPU bodies really execute.

What must change is the recorded ANALYSIS, in plan.md, in this file, and above all in the
three skill entries this branch adds. Two of the three state something I disproved by
running it. A wrong entry in the skill is worse than no entry, because the next port
inherits it as established fact.

Standing rules are clean and re-checked: `jargon.py --port colmap` clean over the whole
branch; title 59 chars, `[ROCm]` prefix, Claude named, no `Co-Authored-By: noreply`, ASCII,
no em-dash, no AMD-internal account references; fork tree clean
(`git status --porcelain` empty); `moat-port..origin/main` is 0 so the branch is not behind
the fork default; remote `moat-port` == local HEAD == `4c531f5e`. The commit body's
"eight call sites" is accurate (`exe/feature.cc` x7 at 146/218/269/297/325/353/381,
`exe/sfm.cc` x1 at 165, plus `sift_test.cc:136`), and both the nvcc paragraph and the
`RunThreadWithOpenGLContext` paragraph read correctly to an outside maintainer.

### 1. "Line 370 is unreachable from COLMAP" is false -- disproved on this host

`plan.md:305-316` (risk 5 CLOSED), `notes.md:485-509`, and the new
`references/validation.md` entry all rest on the claim that `GlobalUtil.cpp:370` cannot be
reached because "COLMAP always passes `-cuda <index>`". COLMAP does not always pass it.
`feature/sift.cc:583-590` omits `-cuda` when `sift->darkness_adaptivity` is true and
`gpu_index` is negative -- and `gpu_index` defaults to `"-1"` (`feature/extractor.h:80`).
`darkness_adaptivity` is public API: `pycolmap/feature/extraction.cc:163` exposes it as
`SiftExtractionOptions.darkness_adaptivity`.

That path reaches line 370: `sift.cc:668` calls `VerifyContextGL()`, which is
`SiftGPU.cpp:1296-1298` -> `InitSiftGPU()` -> `_UseCUDA == 0` so `new PyramidGL(*this)`
(`SiftGPU.cpp:168ff`) -> `PyramidGL::PyramidGL` calls `InitializeContext()`
(`PyramidGL.cpp:166`) -> `InitGLParam(0)` (`PyramidGL.cpp:177`) -> the `else` branch that
contains line 370.

Measured, not read. A probe driving the real `CreateSiftFeatureExtractor` inside a Qt GL
context, default `gpu_index="-1"`:

    darkness_adaptivity=0:  after Create  _UseCUDA=1 _GoodOpenGL=1  extractor=created
    darkness_adaptivity=1:  after Create  _UseCUDA=0 _GoodOpenGL=0  extractor=nullptr

`_GoodOpenGL=1` is the `NotTargetGL` early return at `GlobalUtil.cpp:326-328` -- that is
the porter's trace, and it is correct for the default configuration, which is what
`sift_test` exercises. `_GoodOpenGL=0` is only reachable through the `else` branch, i.e.
`glewInit()`, the vendor read, and line 370 all ran. So the trace supports **"not reached
in this configuration"**, which is what a single binary with default options can support.
It does not support "unreachable", and it certainly does not support the skill's
"a statement that holds on every GPU and every GL driver rather than on one host".

**The conclusion survives; the reason has to change.** Risk 5's actual fear was a compute
user silently downgraded to GLSL by a cleared `_UseCUDA`. That cannot happen, for a
structural reason that does hold everywhere: `SiftGPU::ParseParam` re-asserts
`GlobalUtil::_UseCUDA = 1` on every fresh `SiftGPU` object (`SiftGPU.cpp:773-776`, guarded
only by `!_initialized`), and `SiftMatchGPU::SetLanguage(SIFTMATCH_CUDA*)` makes
`SiftMatch.cpp:686-691` bypass the `_UseCUDA` test entirely. Measured, GLSL user first,
then a compute extractor in the same process:

    1st: darkness(GLSL)   extractor=nullptr  _UseCUDA=0 _GoodOpenGL=0
    2nd: compute          extractor=nullptr  _UseCUDA=1 _GoodOpenGL=0

`_UseCUDA` is back to 1, as predicted. Rewrite risk 5's closure around that, and drop the
unreachability claim.

Worth one line while you are there, because it is the real residual and it is not what the
risk said: a failed GLSL user sets `_GoodOpenGL = 0`, and `InitSiftGPU` early-returns on
`_GoodOpenGL == 0` for every later extractor in the process, compute included -- so the
second extractor above is `nullptr` too and COLMAP falls back to CPU SIFT. Pre-existing
upstream, identical on a CUDA build, not a port defect, and not something to fix here.

### 2. The forward-declaration claim is false, and `hiprandState` is not a typedef

`notes.md:547-549` and the new `references/strategy-a-cmake.md` entry both say: "A forward
declaration does not save it: `GpuMat<curandState>` needs `sizeof(T)` in host code, and
`hiprandState` is a typedef of `rocrand_state_xorwow` rather than a class you can portably
forward-declare." Both halves are wrong.

`hiprandState` is a class. `/opt/rocm/include/hiprand/hiprand_kernel_rocm.h:43-50` expands
`DEFINE_HIPRAND_STATE(hiprandState, rocrand_state_xorwow)` to
`struct hiprandState : public rocrand_state_xorwow { typedef rocrand_state_xorwow base; };`.
It is `curandState` on the CUDA side that is a typedef
(`curand_kernel.h:302`, `typedef struct curandStateXORWOW curandState;`).

And the forward declaration does work. Compiling the real `patch_match.cc` with its real
command (`/usr/bin/c++ ... -DCOLMAP_HIP_ENABLED`) against an override include dir:

    gated include only                        -> 4 errors (gpu_mat.h:137, gpu_mat_prng.h:37)
    gated include + `struct hiprandState;`    -> 0 errors
    unmodified header (control)               -> 0 errors

**This does not reopen the decision.** The refusal was right and the substitute is right --
see the adjudication below -- so keep the code exactly as it is. Only the justification is
wrong, and it is wrong in the skill, where it will be read as a general fact about ROCm's
RNG headers. Cut that sentence to what holds: the gate as literally requested breaks the
build, reproduced above; the fix that matters is declaring the RNG library on every target
that includes the compat header.

### 3. The Mesa teardown hang is a race, and the entry states it as deterministic

`references/validation.md` records "hung twice out of two" and prescribes lowering `-j`.
The diagnosis is right and I confirmed the stack, with the Qt frames the earlier record did
not show -- these matter because they name the per-test `QApplication` as the trigger:

    #3  __pthread_clockjoin_ex
    #4-#8   libgallium-25.2.8
    #9-#10  libGLX_mesa.so.0
    #11 XCloseDisplay
    #12 QXcbBasicConnection::~QXcbBasicConnection
    #13-14  QXcbIntegration::~QXcbIntegration
    #15 QGuiApplicationPrivate::~QGuiApplicationPrivate
    #16 QApplicationPrivate::~QApplicationPrivate
    #17 colmap::(anonymous namespace)::RunGpuTest

But it is intermittent, not a function of `-j` alone. Four full-suite runs at `-j16` here:
one passed (159/159, 8.76 s), three timed out on `feature/sift_test`. The entry as written
tells a reader that `-j16` hangs; that reader's first `-j16` run may well be green, and
they will conclude the entry is wrong. Say it is a race, give the observed rate, and say
that one green high-`-j` run does not clear it.

### Checked and clean

- **Item 1, `CopyFromPBO`, verified and the remedy is right.** `<iostream>` is included
  (`CuTexImage.cpp:25`) with `using namespace std;`, so `std::cerr` resolves. `_numBytes`
  is the allocation size and `InitTexture` only ever grows it
  (`CuTexImage.cpp:183`, `if(size <= _numBytes) return true;`), so
  `cudaMemset(_cuData, 0, _numBytes)` is always in bounds and never under-zeroes the
  current image. The `#else` sits inside `#if defined(SIFTGPU_GL_INTEROP_ENABLED)`, which
  `CuTexImage.cpp:35-47` defines on every non-HIP build, so this hunk changes no CUDA
  object code. On whether zeroing papers over a caller bug: it does not. The entry point
  returns void into a buffer the caller already allocated, so the only alternatives are
  changing the vendored signature or aborting; a diagnostic plus a defined black image
  matches the existing "Unable To Convert Intput" style at `PyramidCU.cpp:945` and is the
  right weight for vendored third-party code. `esize` stays used on the HIP arm
  (`CuTexImage.cpp:100-105`), so no unused-variable warning.
- **The `ComputeDOG` clamp is interior-exact.** `height` is a kernel parameter,
  `index = IMUL(row, width) + col` (`ProgramCU.cu:475`), and the four clamped fetches
  reduce to the old `index +/- 1` / `index +/- width` for every non-border pixel. The
  second `ComputeDOG_Kernel` overload (`ProgramCU.cu:499-510`) does no neighbour reads and
  correctly needed no change.
- **The pitched-to-linear rebind is exact.** `width`/`height` passed to
  `ComputeOrientation_Kernel` come from `got` itself (`ProgramCU.cu:965`), the same image
  that is bound, so `IMUL(int(y), width) + int(x)` is the element `tex2D` point-sampling
  would have returned over a pitch of `width * sizeof(float2)`. All three kernels clamp x
  to `[1.5, width-1.5]` and y to `[1.5, height-1.5]` (`ProgramCU.cu:848-851`, `1032-1035`,
  `1107-1110`), so coordinates are strictly positive, `int()` equals `floor()`, and the
  largest index is `(height-2)*width + (width-2)`, in bounds.
- **Item 3's substitute is the right remedy** (adjudication): `find_package(hiprand)` and
  `find_package(rocrand)` are already unconditional under `HIP_ENABLED`
  (`FindDependencies.cmake:152-153`), the CUDA arm of SiftGPU has always declared
  `CUDA::curand`, and `colmap_mvs_cuda` already lists `hip::hiprand` + `roc::rocrand`
  (`mvs/CMakeLists.txt:302-306`) for the identical header. The SiftGPU HIP arm now mirrors
  both. It closes a real gap on a split `rocm-sdk` install and keeps the compat header
  self-contained; a forward declaration would trade that for a header whose meaning depends
  on which TU includes it. Correct call.
- Build is current with the tree (`ninja: no work to do`), and `opengl_utils_test` passes
  as a non-GUI-gated test.
