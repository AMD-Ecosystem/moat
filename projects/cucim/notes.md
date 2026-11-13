# cucim notes

## Intake (2026-11-11)

### Disposition override

`data/dispositions.json` carries a 2026-08-06 `skip` for rapidsai/cucim (reason `ported-elsewhere`, note "RAPIDS: the ROCm-DS team owns this domain"). Jeff Daily's 2026-11-11 ruling overrides that disposition: MOAT takes cucim up for the coverage the AMD port lacks, Radeon (wave32) and Windows. The project was scaffolded with `scaffold --force` on that ruling. The skip row itself is left as written; changing it is a person's call.

### Fork

The AMD fork already exists: AMD-Ecosystem/hipCIM (GitHub fork of rapidsai/cucim), default branch `amd-develop`. Jeff asked the AMD-Ecosystem admins for write access on 2026-11-11; until it is granted nobody in MOAT can push `moat-port` there, so the stage is left at intake for Jeff (no `awaiting-fork`, no `release-forks`). Whether MOAT builds on `amd-develop` or ports independently from upstream `main` is Jeff's ruling; the assessment below is the input. `fork_default_branch` was scaffolded as upstream's `main`.

### Licence

- Top level: Apache-2.0 (LICENSE), tier 1, `utils/licenses.py check rapidsai/cucim`.
- `scan-nvidia` on a shallow upstream clone (agent_space/cucim-screen): no NVIDIA proprietary licence text. No submodules. Vendored code under `python/cucim/src/cucim/skimage/_vendored` is CuPy/SciPy-derived (MIT/BSD), and `cpp/include/cucim/3rdparty` (dlpack etc.) is permissive.
- The AMD fork relicensed its own changes "Apache-2.0-only" on 2026-06-04 (commit ab172a9, reverting an earlier MIT dual-licence) and added AMD copyright headers to modified files (3370898).

### Duplicate effort

- AMD-Ecosystem/hipCIM `amd-develop` is the authoritative AMD port (AMD staff, ships the `amd-hipcim` wheel and the rocm.docs.amd.com/projects/hipCIM site). ROCm-LS/hipCIM is archived and its README redirects to AMD-Ecosystem/hipCIM. ROCm-LS/hipCIMwip is archived too. jeffdaily/hipCIM (2026-06-04) is Jeff's own earlier repo.
- Upstream: no open or merged ROCm/HIP/AMD PR, no hip/rocm/amd branch, no AMD mention in README/docs. The AMD work was never offered upstream.
- So this is the "authoritative but incomplete" case of `assess-existing-support.md`: the AMD port is Instinct-only and Linux-only; MOAT's differentiator would be wave32 and Windows.

### CUDA surface (upstream main 27f7540, 2026-11-10)

- 4 `.cu` files (skimage feature/filter kernels loaded by CuPy) plus about 127 CuPy `RawKernel`/`ElementwiseKernel`/`RawModule` sites in Python: most GPU code is CuPy kernels JIT-compiled at runtime, so the port depends on a ROCm CuPy.
- C++ (`cpp/`, `gds/`): cuslide/cuslide2 whole-slide TIFF readers using nvJPEG and, since upstream PR #1148/#1122 (2026-09), nvImageCodec in `cucim.kit.cuslide2`; cuFile (GDS); CUDA runtime memory/staging. Little device code of its own.
- Build: CMake superbuild (C++ libs + plugins) and scikit-build Python bindings, which upstream moved from pybind11 to nanobind in PR #1188 (2026-10-30). Not a torch extension: Strategy-A-shaped for the C++ part, CuPy-on-ROCm for the Python part.
- Tests: pytest suite under `python/cucim/src/cucim/**/tests` and `python/cucim/tests`, gtest/Catch2 C++ tests under `cpp/tests` and plugin `tests/`. Some need downloaded sample slides.
- Upstream is active (pushed 2026-11-11), not archived. Upstream is Linux-only even on CUDA (`Operating System :: POSIX :: Linux`, conda/pip manylinux only).

### Viability

Yes on Linux (proven by the AMD fork on Instinct). Radeon wave32 on Linux looks reachable. Windows is the hard part and may end as a waiver candidate (see risks).

## AMD fork assessment

Compared `AMD-Ecosystem/hipCIM@amd-develop` (f650f03, 2026-10-15) with `rapidsai/cucim@main` (27f7540, 2026-11-10):

- **Ahead/behind:** 114 commits ahead (14 merges), 84 behind. Merge base a0fb9b5, 2026-06-03 ("Update Changelog", the upstream v26.06.00 tag line); the fork merged tag v26.06.00 on 2026-07-08 (ef35ce9) and has not synced since. The port itself started as one squashed commit "AMD ROCm port of cuCIM" (c9e8860, 2026-05-11) on an older upstream.
- **Size:** 208 files, +38457 / -1801 against the merge base (100 added, 95 modified, 10 deleted, 3 renamed). A large share is not port code: `docs/` (ROCm docs-site conversion, about 20% of files), `test_data/baseline` fixtures, a 1516-line `run_amd` build/test/coverage driver, coverage tooling (`scripts/cucim_compare_*.py`, `hipcim_filter_cpp_coverage.py`, `hip_profile_shim.c`), CONTRIBUTING/README rewrites, and a deleted notebook.
- **Port code proper:** `cpp/include/cucim/cuda_runtime_api.h` (494-line CUDA->HIP compat header, Strategy A), nvJPEG replaced by rocJPEG (`cuslide/loader/rocjpeg_processor.cpp` 792 lines, `rocjpeg_jpegtables.h`), cuFile mapped to an AMD shim (`gds/include/cufile_amd.h`, hipFile), ROCm path/RPATH discovery, `python/cucim/src/cucim/clara/_rocm_init.py` (dlopens `libamdhip64`, `librocjpeg`), a new `hipcim` Python package and the package rename to `amd-hipcim` depending on `amd-cupy`, `is_hip` test tolerances, integer interpolation and loop_batch_axis fixes.
- **Feature additions beyond a port:** a new `cucim.kit.cumed` plugin with NIfTI-1 and DICOM readers (743-line cumed.cpp, nifti.h, dicom.h, libopenjpeg/libjpeg-turbo deps), a process-level GPU tile cache, multi-IFD TIFF handling, rocJPEG handle pooling. These are product work, not upstreamable as a port.
- **Noise:** relicensing and AMD copyright headers on modified upstream files, package/distribution rename, docs-site restructure, `[skip ci]`-style changelog merges, internal ticket ids (AIOSS-xxxx) in commit titles.
- **Divergence from upstream since the base:** upstream changed 168 files (+14229/-3555) the fork lacks, including the nanobind binding port (#1188), `cuslide2` + nvImageCodec (#1122, #1145, #1148), the new `cucim.skimage.filters.rank` module with its own `.cu` (#1115), 64-bit label indexing (#1087) and conda rattler-build. Rebasing the fork onto current upstream is a substantial merge (bindings and the slide reader both moved).
- **GPUs claimed:** Instinct only. README and docs list gfx942 (MI300X/MI325X) and gfx950 (MI350X/MI355X); `setup.py` `DEFAULT_GPU_ARCHS = ("gfx942", "gfx950")`, `run_amd` `AMDGPU_TARGETS=gfx942;gfx950`, cuslide CMake `_CUSLIDE_GPU_TARGETS gfx942;gfx950`, trove classifier `Environment :: GPU :: AMD Instinct :: MI300`. All are overridable defaults, not hard wave64 code.
- **Wave64 assumptions seen:** none hard-coded in the C++ or in the fork's Python edits; the CuPy kernels already read `warpSize` from device properties (`_separable_filtering.py`, `_pba_2d.py`). Wave32 correctness of the CuPy kernels has simply never been run.
- **Linux-only pieces:** rocJPEG (VA-API backed, Linux only; the fork even auto-configures the VA-API driver path), hipFile/cuFile (Linux amdgpu only), `_rocm_init.py` loading `.so` names, `run_amd` bash driver, manylinux-only wheels, and the upstream C++ itself (about 37 files using unistd/mman/fcntl/dlopen/pthread).
- **Offered upstream:** no. No PR, issue or branch on rapidsai/cucim; the fork is a downstream product.

### Radeon (wave32) and Windows risks

- CuPy is a hard dependency, recorded with `set-deps cucim cupy`: the MOAT `cupy` project (fork ROCm/cupy, stage review-passed) has validated linux-gfx90a, linux-gfx1201 and windows-gfx1151, so a wave32 and Windows ROCm CuPy exists in MOAT. The fork's `amd-cupy` wheel pin is an AMD product wheel whose Radeon/Windows coverage is unverified; the planner should build against MOAT's cupy. Every `skimage` routine runs through it, and its CuPy kernels have never run on wave32.
- rocJPEG on Radeon: check per-arch support before relying on the GPU decode path.
- Windows: rocJPEG and hipFile do not exist on Windows (CPU decode fallback only, named as such); the C++ tree is POSIX-heavy and upstream never built on Windows even for CUDA. The pure-Python `cucim.skimage` half may be reachable on Windows through MOAT's cupy; the C++ `clara`/cuslide half means porting cucim to Windows first. A partial-scope or waiver question for Jeff, not an agent's call.
- If Jeff rules to build on `amd-develop`, the first job is syncing it with upstream (nanobind, cuslide2/nvImageCodec, the new rank `.cu`).

### Recommendation

`fork` (record via `set-intake`). Worth taking up for wave32 on top of the authoritative AMD port; Windows is plausible for `cucim.skimage` via MOAT's cupy and hard for the C++ slide reader, so the Windows scope needs Jeff's ruling. No upstream PR has been offered by AMD, so any upstream destination is Jeff's call too.
