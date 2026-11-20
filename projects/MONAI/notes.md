# MONAI notes

## Intake screen (2026-11-05, gap-matrix seed)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: partial, AMD fork).

### Licence
- `python3 utils/licenses.py check Project-MONAI/MONAI` -> Apache-2.0, tier 1 (cleared to contribute).

### Existing AMD support (decisive)
- AMD maintains MONAI on ROCm: org fork https://github.com/AMD-Ecosystem/MONAI (forked from Project-MONAI/MONAI, default branch `amd-develop`, tip f9f65c9 dated 2026-09-28, repo pushed 2026-11-05). AMD-official and actively maintained.
- Shipped as `amd-monai` on AMD's PyPI index as part of ROCm-LS (ROCm Toolkit for Life Sciences): https://rocm.docs.amd.com/projects/monai/en/latest/index.html and https://rocm.blogs.amd.com/artificial-intelligence/monai-rocm/README.html. Supported: MI300X / MI325X / MI355X, ROCm 7.0.2 / 7.2.0, Ubuntu 24.04. Labelled early access.
- Upstream is absorbing it too: Project-MONAI/MONAI#9153 "Add ROCm support for AMD Instinct GPUs" MERGED 2026-10-06 (setup.py accepts ROCM_HOME, JIT cache key falls back to torch.version.hip, Dockerfile.rocm). #9161 "Resolve accelerator-vendor dependency variants from pyproject.toml" OPEN since 2026-10-09 (makes `monai[all]` / `monai[cucim]` installable on ROCm). Earlier, #6161 "Add supports for AMD GPU" merged.

### Remaining gap
- The matrix's "partial" is coverage: ROCm-LS documents only Instinct (gfx94x/gfx950) on Linux. Radeon (RDNA, wave32) and Windows are not listed as supported.
- That gap belongs to the ROCm-LS team that owns the fork and the upstream PRs, not to a parallel MOAT port. If wider coverage is wanted, the useful contribution is Radeon/Windows validation evidence handed to that team, coordinated with them.

### Viability
- MONAI is mostly pure PyTorch with a small optional C++/CUDA extension set (monai/csrc), which the merged upstream PR already builds through CUDAExtension hipification. Viable, but already done.

### Recommendation
Decline, SKIP_REASON `ported-elsewhere`: AMD's own team maintains the ROCm port (AMD-Ecosystem/MONAI `amd-develop`, `amd-monai` in ROCm-LS) and is upstreaming it. Note the Radeon/Windows coverage gap for the ROCm-LS team rather than as MOAT work.

## Take-up ruling (2026-11-09)

Jeff Daily ruled on 2026-11-09 that MOAT takes MONAI up as a validation project against the AMD team's existing fork, overriding intake's decline (ported-elsewhere) recommendation:

- The fork is https://github.com/AMD-Ecosystem/MONAI, which the AMD team maintains; its GitHub default branch `amd-develop` is what ships as `amd-monai` in ROCm-LS (Instinct/Linux only).
- Validate it like any port: Linux wave64 and wave32 hosts first, then Windows gfx1151.
- Any fixes Radeon/Windows need go on our `moat-port` branch of that fork, for coordination with the AMD team. The port base is the tip of `amd-develop`, not upstream `dev`: the AMD team's ROCm work lives there and we build on it rather than redo it.
- Delivery is to the AMD team's fork (`amd-develop`), not a new upstream PR to Project-MONAI; the delivery vehicle is Jeff's call later.

How the ruling was recorded (2026-11-10, planner job on linux-gfx942):

- `set-intake MONAI fork` (summary: validation take-up of AMD-Ecosystem/MONAI amd-develop, ruled by Jeff Daily).
- `set-fork-url MONAI https://github.com/AMD-Ecosystem/MONAI --by "Jeff Daily" ...` REFUSED: `no branch 'moat-port' on https://github.com/AMD-Ecosystem/MONAI`. It checks that the port branch already exists on the fork, and this job was not allowed to push anything to the fork, so the branch could not be created.
- The fork URL is the default organization fork (`AMD-Ecosystem/<name>`), so the state machine's existing-fork path records the same fact: `set-state MONAI linux-gfx942 awaiting-fork`, then `release-forks`, which saw that the fork exists and moved the project to `screened` with `fork_url = https://github.com/AMD-Ecosystem/MONAI`. That path does not write a `fork_url_ruling` block. Once `moat-port` exists on the fork, `set-fork-url` can be re-run with the same arguments to attach Jeff's ruling to status.json.
- `fork_default_branch` was the scaffold default `main`, which does not exist on the fork. The fork's upstream mirror is `dev` (fork `dev` 6e358ab is an ancestor of Project-MONAI `dev` c5bddb2; it is stale, 2026-11-10). Recorded with `set-base-branch MONAI dev --by "Jeff Daily"`.
- `moat-port` does not exist on the fork yet. The porter creates it from the `amd-develop` tip (f9f65c9437e131565f75d7a11266540ba51293fe) and not from `dev`. Pushing `moat-port` to the fork is a normal porter action.

## Planner (2026-11-10, linux-gfx942): questions for Jeff before `planned`

The plan (plan.md) is written for the whole project as amd-develop ships it. The stage is still `planning`, with the lock held by linux-gfx942. It was deliberately not moved to `planned`, because the items below touch scope, and under the scope-cut rule only Jeff may decide them. Once they are answered, a planner on linux-gfx942 (the lock holder) records the answers in plan.md and runs `set-state MONAI linux-gfx942 planned`. A planner on a different arch would need `port-lock --take`, which is a person's decision.

1. **TensorRT path (`monai/networks/trt_compiler.py`, `monai/networks/utils.py` `convert_to_trt`/`trt_compile`).** This is NVIDIA-only (tensorrt, torch_tensorrt, polygraphy, cuda-python). amd-develop ships it untouched and it is inert on ROCm, where its tests skip on the missing module. The ROCm equivalent would be MIGraphX (torch_migraphx). By the NVIDIA-library rule, an NVIDIA-only GPU library gets its ROCm equivalent, but here that would be a new backend that neither upstream nor amd-develop has. That is a feature port, not a HIP port. Recommendation: defer it, and leave the path inert and skipped, as amd-develop ships it. Registered as `monai-tensorrt-to-migraphx` (`utils/deferred.py`, feature-port) for Jeff's defer/now ruling.
2. **hipCIM on Windows.** `amd-hipcim` (the cuCIM replacement in the `cucim`/`all` extras) is Linux-only, and setup.cfg already gates it with `platform_system=="Linux"`. No ROCm GPU image-I/O option exists on Windows, so whole-slide-image reading on Windows falls back to the CPU readers (openslide/tifffile), named as such. Does that stand as the Windows scope, or should the Windows gate wait on a hipCIM Windows build?
3. **CuPy-backed transform paths.** MONAI uses CuPy in about 11 modules alongside cucim, as an optional import. CuPy has ROCm wheels on Linux. The plan has the validator install them on Linux, so these paths run instead of skip. Windows has no CuPy ROCm wheel, so those paths skip there. Confirm that this is acceptable and not a cut.
4. **Delivery vehicle.** Recorded as Jeff's call later (a PR into `amd-develop` from `moat-port`, or a handoff to the AMD team). Nothing in the pipeline depends on it before validation.

## Ruling: independent port on upstream dev (Jeff Daily, 2026-11-10)

This supersedes the 2026-11-09 validation take-up against `amd-develop`. Jeff Daily ruled that MOAT ports MONAI as a normal port of upstream Project-MONAI/MONAI `dev`, as if AMD-Ecosystem/MONAI `amd-develop` did not exist. The purpose is an independent port that jeff will compare against amd-develop with the AMD team afterwards, so the port must not be derived from theirs.

For the planner (resume: the stage is `planning`, linux-gfx942 holds the lock) and every later role:
- Do not read amd-develop, its commits or its diff. Rewrite plan.md for the upstream `dev` tip (record the sha). Treat the amd-develop-specific parts of the earlier plan and notes as superseded: the f9f65c94 base, FORCE_CUDA on amd-develop, the MIOpen fresh-process rerun.
- Upstream `dev` already contains the merged upstream ROCm build support (#9153); that is upstream code and in scope as such.
- `moat-port` is cut from the upstream `dev` tip and pushed to AMD-Ecosystem/MONAI as a new branch. The fork's GitHub default branch stays `amd-develop`, because the AMD team owns that fork: the one exception to the moat-port-default rule. `fork_default_branch` stays `dev`.
- Scope is the whole project: monai._C, the runtime-built GMM extension, optional GPU dependencies with their ROCm equivalents where they exist (cuCIM -> hipCIM, CuPy ROCm wheels on Linux), and the GPU, unit and integration suites.
- The three scope questions above (TensorRT/MIGraphX, hipCIM on Windows, CuPy on Windows) still await jeff; carry them forward, revised for the new base if their facts change, and do not mark the project planned until he answers.
- Validation order as usual: Linux wave64 and wave32 first, Windows gfx1151 last.

## Scope rulings (Jeff Daily, 2026-11-18): the three planner questions are answered

These answer the questions under "Planner (2026-11-10, linux-gfx942): questions for Jeff before `planned`", carried forward by the 2026-11-10 independent-port ruling. Nothing else holds the project in `planning`: the planner on linux-gfx942 (the lock holder) writes them into plan.md, revised for the upstream `dev` base, and marks the project planned.

1. **TensorRT path: deferred.** `monai/networks/trt_compiler.py` and `convert_to_trt`/`trt_compile` stay as upstream ships them, inert on ROCm with their tests skipping on the missing module. A MIGraphX (`torch_migraphx`) backend is a feature port, not part of this HIP port; deferral `monai-tensorrt-to-migraphx` is ruled `defer`. The port's documentation names the gap.
2. **Whole-slide image reading on Windows: the CPU readers stand as the Windows scope.** hipCIM is Linux-only and no ROCm GPU image-I/O option exists on Windows (the same fact behind the approved `windows` waiver on MOAT's cucim project), so Windows uses openslide/tifffile, named as a CPU fallback in the documentation. The `windows` gate does not wait on a hipCIM Windows build.
3. **CuPy-backed transform paths on Windows: they run, they do not skip.** There is no Windows CuPy ROCm wheel, but MOAT's cupy project is the Windows ROCm build of CuPy, validated on windows-gfx1151 at a6e01258f (upstream PR cupy/cupy#10467 open). The Windows validator builds CuPy from that port following `projects/cupy/notes.md`, "Install as a dependency", into the venv MONAI runs in, and runs the CuPy-backed paths. `cupy` is recorded as a hard dependency of this project. On Linux the plan is unchanged (upstream CuPy ROCm wheels). If the CuPy source build proves too heavy for the gfx1151 host, record the measured reason and bring it back to jeff; do not skip silently.

## Planner (2026-11-20, linux-gfx942): questions for Jeff before `planned`

**Three questions block the porter.** The project stays in `planning` (lock re-stamped to linux-gfx942 by this job, which has now ended), and nobody may mark it `planned` until Jeff rules on 1-3. Once he has, a planner on linux-gfx942 writes the answers into plan.md (Open questions, the surface table and the change list) and runs `set-state MONAI linux-gfx942 planned`. Questions 1 and 2 are scope cuts under the scope-cut rule; question 3 bears on the independent-port ruling. Everything else in plan.md is settled by the 2026-11-10 and 2026-11-18 rulings.

1. **nvImageCodec GPU DICOM decoding (new upstream since the last plan).** `NvImgCodecPydicomReader` and `monai/data/nvimgcodec_pydicom_plugin.py` (`monai[nvimgcodec]`) decode compressed DICOM on the GPU via NVIDIA's `nvidia.nvimgcodec` pydicom plugin plus CuPy. On ROCm the reader reports itself unavailable and MONAI decodes on the CPU (pydicom/ITK). The ROCm counterpart covers part of it: rocJPEG decodes baseline JPEG only, and its Python binding is rocPyDecode's `rocPyJpegDecode`, a Linux system package. Nothing on ROCm decodes JPEG Lossless, JPEG 2000 or HTJ2K on the GPU.
   - (a) Defer. The reader stays inert and the CPU decoding is named as a CPU fallback in the docs. This is consistent with the TensorRT ruling, because a ROCm reader is a new backend in MONAI that neither upstream nor any AMD branch has.
   - (b) Now. Write a rocJPEG-backed pydicom decoder plugin in MONAI for the baseline-JPEG transfer syntax, keeping CPU for the rest, Linux only.
   Registered as `monai-nvimgcodec-dicom-to-rocjpeg` (feature-port). **Recommendation: (a).** The standing "NVIDIA library -> ROCm equivalent" rule points toward (b). But here MONAI only delegates to NVIDIA's plugin, so (b) would be new feature code covering one transfer syntax of four.
2. **kvikio GPUDirect Storage.** `GDSDataset` (`monai/data/dataset.py`) and the image readers' `to_gpu` option (`monai/data/image_reader.py`) use RAPIDS kvikio. kvikio has no ROCm build, so these paths are inert and their tests skip. ROCm's GDS library is hipFile, which PyTorch exposes on ROCm as `torch.cuda.gds`.
   - (a) Defer.
   - (b) Request an intake of rapidsai/kvikio as its own project, so MONAI picks it up later as an environment dependency.
   - (c) Give MONAI a `torch.cuda.gds` backend (feature work).
   Registered as `monai-kvikio-gds-to-hipfile` (feature-port). **Recommendation: (a)**, with (b) as a separate intake if GDS matters to you.
3. **Overlap with the AMD team's open upstream work.** Two AMD engineers have open work upstream: issue #9285, PR #9161 ("Resolve accelerator-vendor dependency variants", which makes `monai[all]`/`monai[cucim]` resolve to `amd-hipcim`, a ROCm CuPy and so on), and PR #9290 ("Report the HIP version"). Both PRs were updated 2026-11-19. Today `pip install 'monai[cucim]'` on ROCm tries to install `cucim-cu13`, and `print_config` says `CUDA version: None`.
   - (a) Stay clear of both. The port does not touch `pyproject.toml` extras, `print_dependencies.py`, `check_env`/`deviceconfig` reporting, or `Dockerfile.rocm`. The documented route installs the ROCm equivalents by name, as `Dockerfile.rocm` already does. This avoids a competing AMD PR, the gsplat lesson.
   - (b) Cover the same ground independently in the port, as the "as if amd-develop did not exist" ruling could be read.
   **Recommendation: (a).** plan.md is written that way and lists the extras and version reporting under "Not in scope unless Jeff rules otherwise".
4. **(Not blocking.) Delivery.** Should the port go upstream to Project-MONAI as its own PR, given #9161/#9290, or only into your comparison with amd-develop? Your call after validation.

What this round did (2026-11-20):
- Cloned upstream `dev` read-only at a85d75e11cdd9c6d6d13500dc7f3960dd07dd7d3 into `projects/MONAI/src`. amd-develop was not read.
- Rewrote plan.md for that base and regenerated surface.json: the floor is unchanged at 13 files, 24 kernels, 30 launches and 4 warp-intrinsic sites; 14 components; TensorRT scoped out under the 2026-11-18 ruling; `surface.py check MONAI` passes.
- Registered the two deferrals above.

Findings the porter needs (details in plan.md):
- **Real ROCm defect (gfx12): `_compiled_unsupported()`** (`monai/transforms/spatial/functional.py:58`) applies the CUDA rule "compute capability major >= 12 means `_C` lacks this arch". On ROCm the device properties hold the gfx version, and `TORCH_CUDA_ARCH_LIST` is unset, so `max_compute_capability()` is 0. gfx1201 and gfx1250 are therefore declared unsupported, and `Resample`/`SpatialResample`/`Warp` fall back silently to `grid_sample`. Fix: return False on `torch.version.hip`, and adjust `tests/transforms/test_spatial_gpu_support.py` for ROCm. gfx1201 validation proves it.
- Validation traps:
  - `USE_COMPILED` needs `BUILD_MONAI=1` at test time. Without it every `_C` test skips.
  - `test_gmm.py` turns a JIT build failure into `skipTest` and is skipped under `QUICKTEST`.
  - The GMM JIT has a 300 s build timeout.
- GMM shuffle code re-read at the new base: correct on wave64 and wave32. The xor/down offsets never cross a 32-lane half, and only lane 0 of each half is consumed.
- The permutohedral `ranks`/`zeros` un-memset allocations from the old plan's risk 4 are under `LINEAR_D_MEMORY`, which is commented out. They are dead code, so that risk is retired.
- `THC/THCAtomics.cuh` hipifies to `THH/THHAtomics.cuh`, which the ROCm torch wheel ships (checked on torch 2.16 dev, read-only).
- nvsubquadratic (Hyena blocks) needs no port: the base wheel is pure PyTorch. Install it with `--no-deps`, because it pins `torch<2.15`.
