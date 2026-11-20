# kaolin notes

## Intake screen (2026-11-06)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: partial, AMD fork exists).

### Licence
`utils/licenses.py check NVIDIAGameWorks/kaolin`: Apache-2.0, tier 1. LICENSE on `master` read by hand: the Apache License 2.0 text. Older Kaolin releases were under the NVIDIA Source Code License; the current default branch is not. No per-file `scan-nvidia` run, because the recommendation is a decline; run it before any take-up, since an older-licence file could survive in the tree.

### Existing AMD support
- `AMD-Ecosystem/kaolin` (a GitHub fork of NVIDIAGameWorks/kaolin, in the AMD-Ecosystem org) carries an AMD ROCm enablement: two commits on `master` dated 2026-09-17 by AMD engineers (vcsajjan, soumitra-chatterjee-tech), "[ROCm] AMD Instinct (ROCm) enablement for kaolin"; a `rocm-enablement` branch holds one equivalent commit. It is not a MOAT project (no MOAT records or claim existed; `scaffold` accepted the name).
- Scope per its README: kaolin 0.18.0, full `kaolin._C` GPU extension (87 kernels) via torch build-time hipify plus wave64 fixes; validated on MI300X (gfx942: 15213 passed, 88 failed "all non-port", 1444 skipped) and MI350X/MI355X (gfx950, targeted subset 5618 passed, 1 failed). Diff vs upstream: 35 files, setup.py ROCm detection (`PYTORCH_ROCM_ARCH`, default `gfx942;gfx950`), a `rocm_cub_compat.h`, docs.
- Upstream: no ROCm/HIP/AMD PRs (open or closed) on NVIDIAGameWorks/kaolin; the fork has not been offered upstream. Upstream master has moved 2 commits past the fork's base.

### Gaps left by the AMD fork (for the record, not a reason to compete)
- Instinct-only: no wave32 (RDNA) or Windows validation, and the default arch list excludes RDNA.
- `kaolin.physics` and USD physics materials are wrapped in `try/except ImportError` because `warp-lang` has no ROCm build, so physics (Simplicits) is not covered.
- Footprint: the diff adds 22 new `*_gpu.cu` files (e.g. `metrics/sided_distance_gpu.cu` next to upstream's `sided_distance_cuda.cu`), which look like committed hipify-style copies rather than edits in place; not inspected further.

### Recommendation
Decline, reason `ported-elsewhere`: an AMD-owned fork (AMD-Ecosystem/kaolin) already ports the whole CUDA extension and validates it on MI300X/MI355X. The wave32/Windows coverage and the warp-dependent physics module are differentiators that should be contributed to that fork by coordinating with its owners, not a parallel MOAT port.

### Question for jeff
AMD-Ecosystem/kaolin is in our own org but is not a MOAT project. Should MOAT adopt it as an existing fork (validate wave32/Windows and review the added `*_gpu.cu` copies) in coordination with its authors, or leave it to them?

## Take-up ruling (2026-11-10)

Jeff Daily ruled on 2026-11-10 that MOAT takes kaolin up, building on the AMD team's existing ROCm work (same approach as PhysX), overriding intake's decline (ported-elsewhere) recommendation:

- Fork: https://github.com/AMD-Ecosystem/kaolin. `rocm-enablement` is 74a1b18187 (one commit "[ROCm] AMD Instinct (ROCm) enablement for kaolin", 2026-09-17, 35 files). The fork's `master` is fcc9bfcd, a merge of that commit onto upstream d52da9f8 with no further content, so `master` is the AMD team's delivery branch and not a pure upstream mirror (2 ahead / 2 behind NVIDIAGameWorks/kaolin `master` at the time of the ruling). Recorded `fork_default_branch` = `master` (`set-base-branch kaolin master --by "Jeff Daily"`), as with MONAI's `amd-develop`. Neither fork branch is to be modified.
- Port base: the porter creates `moat-port` from the fork `master` tip (carrying 74a1b181 and its authorship), then merges upstream `master`. Our work goes on top as separate commits.
- Minimal viable diff: clean up whitespace-only changes and every other change the port does not need, in commits on top. The ruling keeps only: the `amd-kaolin` distribution rename in setup.py, the `PYTORCH_ROCM_ARCH` default of gfx942;gfx950 when unset (whether to extend it to the wave32 targets is a question for Jeff, not a porter decision), and the ROCm port changes themselves. Known noise to examine first: the 26 added `kaolin/csrc/**/*_gpu.cu` files (~10K lines) that sit beside the existing `*_cuda.cu` files; the one pair compared (metrics/sided_distance) is byte-identical apart from an added AMD copyright line. If they are duplicates, remove them so the port edits the original files in place. Also review the added "Modifications Copyright" lines against the default-no-new-copyright rule, and the README/installation.rst rewrites, for necessity.
- Also in scope: Radeon (wave32, gfx1100/gfx1201) and Windows gfx1151, with every gate validated (Linux wave64 and wave32 first, Windows last). The AMD README reports MI300X 15213 pass / 88 fail ("all non-port") / 1444 skip, and wave64 fixes in sg_inner_product, packed_simple_sum and deftet; re-check those 88 failures against upstream on NVIDIA rather than accepting the label.
- Delivery vehicle is Jeff's call later.

Recorded: `set-intake kaolin fork`, `set-state kaolin linux-gfx942 awaiting-fork`, `release-forks` (stage `screened`, `fork_url` set). Re-run `set-fork-url kaolin https://github.com/AMD-Ecosystem/kaolin --by "Jeff Daily"` once the porter has pushed `moat-port`.

## Licence ruling (2026-11-11, Jeff Daily)

A rescan with the fixed `licenses.py scan-nvidia` (pure Python; the grep-based scan falsely reported this tree clean on the Windows host) found NVIDIA-restricted files in upstream kaolin, unchanged in AMD's `master`/`rocm-enablement` (AMD's delta does not touch them; its `unbatched_mcube_gpu.cu` includes two of them unchanged):
- `kaolin/csrc/ops/spc/point_utils.h`: NVIDIA "proprietary rights ... strictly prohibited" header; declarations only, included by `kaolin/csrc/bindings.cpp`.
- `kaolin/csrc/ops/conversions/unbatched_mcube/defines.h`, `helper_math.h`, `tables.h`: CUDA-samples EULA headers.
- `kaolin/non_commercial/flexicubes/` (`flexicubes.py`, `tables.py`): NVIDIA Source Code License, non-commercial (`LICENSE.NSCL`, `COPYRIGHT`).

Jeff Daily ruled, on `moat-port`, as separate commits:
1. Delete the four headers and replace each: fresh declarations for `point_utils.h`; rewritten `defines.h` and vector-math `helper_math.h` (drop-in for both CUDA and HIP; HIP's built-in vector operators do not exist on CUDA); `tables.h` generated programmatically from the marching-cubes case rules (Lorensen and Cline, 1987), not copied from the NVIDIA-headered file.
2. Delete `kaolin/non_commercial/` (Python, unused by the GPU build) and anything that imports only it.
3. Build and run the CUDA path with real nvcc (mi355 host) as well as the ROCm path, since the replacements change the CUDA build too.
4. Before any delivery, the replacements and `point_utils.h`'s stale proprietary header go to AMD's legal contact (Jeff raises it).

Context: upstream is an NVIDIA-org repo with infrequent releases (latest v0.18.0, 2025-08-08); upstreaming ROCm support is unlikely, so the deliverable is this fork. AMD's fork has produced no release (no tags; `amd-kaolin` is not on public PyPI).

## Planning (2026-11-20, linux-gfx1100 host, planner)

### Decisions needed from Jeff before a porter starts

Full wording and options are in `plan.md` (Open questions). Q1 and Q2 block the porter; Q3 and Q4 have a stated default that applies only if Jeff confirms it.

- **Q1. Warp-dependent features** (`kaolin.physics`, `farthest_point_sampling`, `io.usd.physics_materials`, `experimental.newton`) need a ROCm Warp. MOAT's `warp-1.14` is still in `porting` (Phase 0 done 2026-11-18). Options: (a) port everything else now and record these as `owed`, which blocks `pr_ready` until they validate on the `warp-1.14` build (planner's recommendation); (b) scope them out with a deferral; (c) hold kaolin until `warp-1.14` completes.
- **Q2. `PYTORCH_ROCM_ARCH` default** in setup.py when unset: (a) keep `gfx942;gfx950`; (b) add `gfx1100;gfx1101;gfx1151;gfx1200;gfx1201`; (c) drop the override and use torch's default (the visible GPUs; needs a fallback for GPU-less builds).
- **Q3. nvdiffrast backend**: default is to leave it untouched and record it `scoped_out`, because MOAT declined NVlabs/nvdiffrast as licence-blocked. Confirm.
- **Q4. AMD copyright lines and README/installation text**: default is to remove the "Modifications Copyright ... AMD" lines from upstream files and the AMD README/pip-index prose, and add a short ROCm source-build subsection. Confirm.

### What was done
- Took the work lock (`set-state kaolin linux-gfx1100 planning`), cloned the fork into `projects/kaolin/src` with an `upstream` remote, read the AMD commit file by file, set `ext_type` = `torch-extension`, generated `surface.json` (32 CUDA files from the census at upstream 393740bc) and added 36 components. Nothing built, no GPU used.

### Findings the porter needs (details in plan.md)
- There are 20 `*_gpu.cu` copies, not 26 as the take-up ruling said. 11 are byte-identical to their `_cuda.cu` twin apart from an AMD copyright line. 9 carry the real changes: gs_to_spc, mesh_to_spc, unbatched_mcube, packed_simple_sum, bf, recon, spc, deftet, sg_inner_product.
- Fork `master` as committed looks unbuildable. The `**/*.cu` glob compiles both twins, which define the same strong symbols. The `_cuda.cu` originals also keep nvcc split chevrons (`<< <`), which torch hipify does not rewrite and clang rejects. So the AMD validation numbers cannot come from this commit's tree.
- Of the three AMD "wave64 parity fixes", only deftet is a real wave64 bug: its per-pixel ballot spans two pixels on a 64-lane wavefront. packed_simple_sum and sg_inner_product are correct 32/16-lane logical reductions on wave64, because each leader lane reads only its own group. They only need the 32-bit `__shfl_down_sync` mask fixed, which HIP rejects at compile time. The sg forward mask `0xfffffff` is a 28-bit upstream bug.
- The AMD sg_inner_product file also includes an MI300 performance rewrite in unguarded shared code: a grid split, `atomicAdd` into the outputs, `__expf`, `fmaf`, and `empty_like`. The 2026-11-10 minimal-diff ruling drops it.
- Upstream `master` moved 3 commits past the fork's base (Python and tests only), and the merge is clean.
- Optional module dependencies, which are not `depends_on`: warp-lang (Q1), Newton (Q1), gsplat (use the AMD-Ecosystem port as environment), and nvdiffrast (Q3). The `_C` extension build itself has no hard MOAT dependency.
