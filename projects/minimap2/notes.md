# minimap2 notes

## Intake screen (2026-11-11)

Requested by jeff (2026-11-11) for Radeon (wave32) and Windows coverage of AMD's life-science set. The AMD fork already exists at https://github.com/AMD-Ecosystem/minimap2 (default `master`, a GitHub fork of lh3/minimap2). jeff has asked the AMD-Ecosystem admins for write access; until it is granted, `moat-port` cannot be pushed to the fork. No `awaiting-fork` / `release-forks` step applies; the stage is left for jeff.

### Licence

- GitHub reports NOASSERTION; `utils/licenses.py check lh3/minimap2` -> UNPARSED.
- `LICENSE.txt` read by hand: standard MIT text (Copyright 2018- Dana-Farber Cancer Institute, 2017-2018 Broad Institute). Recorded `MIT`.
- Upstream submodule `lib/simde` (MIT). The AMD fork adds submodules cJSON (MIT), simde, spdlog (MIT), googletest (BSD-3-Clause), ittapi (BSD-3-Clause / GPL-2.0 dual); its GPU chaining files carry `SPDX-License-Identifier: MIT` (from mm2-gb).
- `utils/licenses.py scan-nvidia` on upstream master (3c28777) and on the AMD fork master (a37d8fa): no NVIDIA proprietary licence text.

### Duplicate effort

- Upstream PRs for hip/rocm/amd/gpu: lh3/minimap2#1258 "mm2-gb" (Minimap2onGPU/mm2-gb, GPU chaining, HIP-first with CUDA macros) closed 2024-11-15 by the maintainer: "I would not be able to maintain GPU code in the long term. It is better to keep this as an independent fork in your own repo. mm2-fast is doing this way." This is a standing no to GPU code upstream. Realistic deliverable is the AMD fork, not an upstream PR (a person would record `set-no-upstream` or `set-maintained` if adopted that way).
- No AMD-authored PR in lh3/minimap2: the AMD fork's work was never offered upstream.
- ROCm / AMD-Ecosystem / AMD-AIOSS: only AMD-Ecosystem/minimap2 is visible; `AMD-AIOSS/minimap2` (named as the kernel source in the minibwa fork) is not visible to this account.
- Research precursor: mm2-gb (Dong et al., bioRxiv 2024), which the AMD fork's README asks GPU users to cite.

### CUDA surface / viability

- Upstream has NO CUDA at all (CPU-only C with SSE/NEON ksw2). The AMD fork adds the GPU backend from scratch (mm2-gb-derived chaining in `.cu` files compiled as HIP via a `hipify.cuh` macro header, plus AMD-written ksw2 extension kernels), so the AMD fork IS the GPU port. Its CMake supports `GPU=AMD` and `GPU=NONE` only; there is no NVIDIA build to preserve.
- Upstream activity: last commit 2026-05-19 (minimap2-2.31 release); not archived; open unrelated PR #1364.
- Tests: upstream has no test target (sample data in `test/`). The AMD fork adds googletest suites `tests/core` and `tests/gpu` (align, chain, combined, context, mapctx concurrency), `test_suite/{small,medium,skip-large,original}` JSON goldens, `python/tests/test_mappy_gpu.py`, and accuracy/GIAB concordance scripts.
- Hard MOAT dependencies: none. minibwa reuses this fork's ksw2 kernels (copied, not a build dependency).
- Windows: upstream has partial `_WIN32`/`_MSC_VER` guards (index.c, misc.c, kthread.c, ksw2_dispatch.c, khash.h) but still includes `pthread.h`/`unistd.h`/`zlib.h` and is normally built with MinGW/Cygwin; MSVC/clang-cl is not supported upstream.

## AMD fork assessment

Compared `AMD-Ecosystem/minimap2` `master` (a37d8fa) with `lh3/minimap2` `master` (3c28777).

- Ahead/behind: 2 ahead, 0 behind. Merge base = upstream tip 3c28777 (2026-05-19, "Release minimap2-2.31 (r1302)"). Commits: 748ae64 (soumitra-chatterjee-tech, 2026-09-11) and a37d8fa (vcsajjan, 2026-09-15), both titled "[ROCm] AMD Instinct (ROCm) enablement for minimap2"; message says "Reconstructed clean ROCm-enablement overlay on top of master (hipSHIFT-internal artifacts excluded)" and carries a `Co-Authored-By: Claude` trailer. Branch `rocm-enablement` = 748ae64 (1 behind master). Last push 2026-09-15.
- Delta: 278 files, +102004/-561. Upstream root sources are left untouched (only python/cmappy.h changes among root C/H); the fork instead DUPLICATES the whole library into a restructured tree `minimap2/{core,seed,chain,align,src}/` (copies of align.c, map.c, index.c, lchain.c, ksw2_*.c, etc., modified) and builds that tree with a new 754-line CMakeLists.txt + CMakePresets.json. Root Makefile also reworked (+238 lines).
- GPU code: `minimap2/align/src/gpu/` (ksw2_extd2/exts2/extz2 single/multialign/packed kernels, gpu_accum.cpp 1532 lines, batch/backtrack), `minimap2/chain/src/gpu/` (mm2-gb plchain/plmem/plrange/plscore/planalyze `.cu`), `minimap2/seed/src/seed_gpu.hip`.
- Noise: new CLI tools (`cli/mm2_{seed,chain,align}`), spdlog logging layer, JSON/flatbuffer dumps and profiler, `.devcontainer`, Dockerfiles (deb/rpm/sles) and Apptainer packaging, `AGENTS.md`, `CONVENTIONS.md`, `.github/instructions/*`, CODEOWNERS, PR template, `.clang-format/.clang-tidy`, `docs/`, `USAGE.md` (1174 lines), new man pages, ~19k-line golden JSON files, profiling scripts. README rewritten to advertise AMD GPU acceleration. Upstream-unmergeable by design; only a fraction is the GPU port proper.
- GPUs claimed: AMD Instinct. BUILD_GUIDE and presets target gfx942 (MI300X/MI300A, default when not detected) and gfx950 (MI355X); a `gfx1100` (Radeon RX 7900) preset exists but no doc or result claims it was validated. No Windows mention.
- wave32 breakers: `ksw2_extd2_gpu_multialign.cpp` hard-codes `#define WARP_SIZE 64`, `MAX_WARPS 8`, is launched with 512-thread blocks (`ksw2_gpu_batch.cpp:652`), derives `lane_id = tid % 64`, and carries the DP boundary between wavefronts with a last-lane publish + `__shfl_up(x, 1)` + lane-0 fixup: on wave32 `__shfl_up` wraps at lane 32, so the carry is lost at every odd 32-lane boundary, and the max reduction never folds in the upper wave of each 64-lane group. `ksw2_extd2_gpu.cpp` does `__shfl(x, last_lane)` with `last_lane` up to `blockDim.x - 1`, assuming one block is one wavefront. Both are silent wrong-answer bugs on RDNA, not crashes. The mm2-gb chaining code reads `prop->warpSize` (plmem.cu:777), which is the portable pattern; it needs checking, not assuming.
- Linux-only pieces: CMake resolves `/opt/rocm`, packaging is deb/rpm/sles/Apptainer, devcontainer, ittapi; plus upstream's POSIX includes. Windows needs the CMake HIP path working under clang-cl and the CPU core building without MinGW.
- Offered upstream: no, and the maintainer's #1258 answer says it would not be accepted.
- Judgement: AMD-official and substantial (real GPU tests and goldens), but a fork-of-a-restructure with heavy non-port noise, Instinct-only, wave64-hard-coded. Building on it means fixing wavefront-size assumptions and Windows build on the AMD tree; porting independently means re-applying a minimal HIP backend to upstream's flat layout (mm2-gb chaining + the ksw2 kernels). Which one is jeff's ruling; given there is no upstream PR destination, building on the AMD fork is the cheaper path to Radeon/Windows coverage.
