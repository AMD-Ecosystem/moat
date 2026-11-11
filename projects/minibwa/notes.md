# minibwa notes

## Intake screen (2026-11-11)

Requested by jeff (2026-11-11) for Radeon (wave32) and Windows coverage of AMD's life-science set. The AMD fork already exists at https://github.com/AMD-Ecosystem/minibwa (default `master`, a GitHub fork of lh3/minibwa). jeff has asked the AMD-Ecosystem admins for write access; until it is granted, `moat-port` cannot be pushed to the fork. No `awaiting-fork` / `release-forks` step applies; the stage is left for jeff.

### Licence

- GitHub reports NOASSERTION; `utils/licenses.py check lh3/minibwa` -> UNPARSED.
- `LICENSE.txt` read by hand: top-level MIT (Copyright 2025- Dana-Farber Cancer Institute). Recorded `MIT`.
- Third-party notices in the same file: vendored `libsais` (Apache-2.0), vendored `mimalloc/` (MIT, Microsoft). Optionally built and ON by default (disabled with `make gpl=0`): `QSufSort.c` (HPND) and `bwtgen.c` (GPL-2.0). Contributing to the MIT files is unaffected; a default-built binary links GPL-2.0 code. Flagged for jeff, not a blocker for contribution.
- No submodules. `utils/licenses.py scan-nvidia agent_space/minibwa-screen` (upstream master 18bd00a): no NVIDIA proprietary licence text.

### Duplicate effort

- Upstream PRs/branches for hip/rocm/amd/gpu: none (only unrelated perf PRs #8, #9, #78, #79). No AMD-authored PR in lh3/minibwa: the AMD fork's work was never offered upstream.
- Maintainer stance: lh3 declined GPU code for minimap2 (lh3/minimap2#1258, 2024-11): "I would not be able to maintain GPU code in the long term. It is better to keep this as an independent fork". Same maintainer; expect the same answer here. Realistic deliverable is the AMD fork, not an upstream PR (a person would record `set-no-upstream` or `set-maintained` if adopted that way).
- ROCm / AMD-Ecosystem / AMD-AIOSS: only AMD-Ecosystem/minibwa is visible. The fork's README links `AMD-AIOSS/miniBWA`, which is not visible to this account (private or absent).

### CUDA surface / viability

- Upstream has NO CUDA at all (no `.cu`/`.cuh`, no GPU code). It is a CPU short-read aligner (BWT seeding, SSE ksw2 extension). The AMD fork adds a HIP backend from scratch, so the AMD fork IS the GPU port; there is no CUDA path to preserve.
- Upstream is active (last commit 2026-10-30), not archived.
- Tests: upstream has no test target (`Makefile` targets: all, minibwa, clean, depend); `test/` holds chrM reference/read pairs and `api-test/` two API examples. The AMD fork adds `gpu/gpu_selftest.cpp` + `gpu/build_selftest.sh` and a CPU-vs-GPU concordance procedure in BUILD_AND_RUN.md.
- Hard MOAT dependencies: none. Kernel code is transplanted from the AMD minimap2 fork (see the minimap2 project), not a build dependency.
- Windows: upstream itself is POSIX-only (`pthread.h`/`unistd.h` in bseq.c, fastmap.c, kommon.c, kthread.c, l2bit.c, main.c, map-main.c; no `_WIN32` guards). Windows needs a CPU-side port of minibwa first; likely a waiver candidate (an agent may only suggest it).

## AMD fork assessment

Compared `AMD-Ecosystem/minibwa` `master` (ae3f0f5) with `lh3/minibwa` `master` (18bd00a, 2026-10-30).

- Ahead/behind: 2 ahead (c19c414 "[ROCm] AMD Instinct (ROCm) enablement for minibwa" by soumitra-chatterjee-tech, 2026-09-15, plus ae3f0f5, a merge of the same title by vcsajjan), 4 behind. Merge base f0e1174 (2026-08-10, "exit code was always 0 even on error (#67)"). Last push 2026-09-15; idle since.
- Second branch `rocm-enablement` (7646248): one commit with NO common history with upstream (orphan snapshot). The commit message of the master version says "Reconstructed clean ROCm-enablement overlay".
- Delta: 41 files, +3922/-517. New `gpu/` (ksw2_extd2_gpu_multialign.cpp 1029 lines, ksw2_gpu_batch.cpp 770, ksw2_backtrack_gpu.cpp 211, gpu_dispatch.cpp, hip_raii.h, selftest); C-side seam in align.c, map-algo.c, map-main.c, pe.c, options.c, minibwa.h, mbpriv.h; Makefile `GPU=1` path.
- Noise and hazards:
  - The overlay silently REVERTS upstream work. Despite the Aug-10 merge base, several upstream files are byte-identical to older upstream versions: bwt.c = r382 (2026-06-23), l2bit.c = r384 (2026-06-25), NEWS.md = minibwa-0.3 (2026-06-28), kommon.c = 2026-05-10, pe.c = r352 (2026-05-26); bwt.h, kommon.h, l2bit.h, minibwa.1, ketopt.h also lose lines. Any MOAT base on this branch must be rebuilt on current upstream, not merged as-is.
  - Duplicated headers: `gpu/include/{hip_raii.h,ksw2_gpu_batch.h}` and identical copies under `gpu/include/gpu/`.
  - Test data committed at repo root: `reads.fq`, `ref.fa`, binary `ref.fa.l2b`, `ref.fa.mbw`.
  - README rewritten with an "AMD Instinct ROCm 10.1" badge and AMD-first intro; NEWS.md trimmed; `.github/workflows/build.yml` changed; new BUILD_AND_RUN.md.
  - Commit carries a `Co-Authored-By: Claude` trailer.
- GPUs claimed: AMD Instinct only. Validated on MI300X (gfx942) and MI350X/MI355X (gfx950) under ROCm 10.1 (README, BUILD_AND_RUN.md: 10/10 selftests, ~99.997% CPU/GPU concordance). gfx1250 "not yet validated". No Radeon, no Windows.
- wave32 breakers: `gpu/ksw2_extd2_gpu_multialign.cpp` hard-codes `#define WARP_SIZE 64`, `MAX_WARPS 8` for 512-thread blocks, computes `lane_id = tid % 64`, and carries the DP boundary between wavefronts via `lane_id == WARP_SIZE-1` publish + `__shfl_up(x, 1)` + lane-0 fixup. On wave32 the real wavefront is 32 lanes: `__shfl_up` wraps at lane 32, so the carry is missing at every odd 32-lane boundary; and the max-score reduction (`__shfl_down` from offset 32, then lane 0 of each 64-lane group writes `reduce_H[warp_id]`) never folds in the upper 32-lane wave of each group. Silent wrong alignments/scores, not a crash.
- Linux-only pieces: Makefile hipcc path (`ROCM_PATH` from `hipconfig`, `--gcc-install-dir` for the manylinux gcc-toolset, `-no-pie`, `-lamdhip64`), `gpu/build_selftest.sh`; plus the POSIX-only upstream CPU code above.
- Offered upstream: no (no PR in lh3/minibwa), and the maintainer's stated policy for minimap2 says it would not be accepted.
- Judgement: AMD-official but a single-drop, Instinct-only overlay that regresses upstream code. Reusable as a reference for the kernel and the C seam; a MOAT port would re-apply the seam on current upstream, make the kernels wavefront-size-generic, and add Radeon evidence. Whether to build on it or start from the minimap2 kernels is jeff's ruling.
