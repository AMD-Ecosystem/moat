# spconv notes

## Screen 2 -- 2026-08-13, intake, linux-gfx1100

Recommendation: **DECLINE, reason `already-supported`.** Not recorded -- the
disposition is a person's. `set-intake` carries the row into the queue.

The re-open premise was that FlyDSL makes a cumm AMD codegen backend tractable.
FlyDSL is real, but the premise is now moot for a different reason: while we were
not looking, AMD sparse convolution got solved by other people, in Triton, with
the same API. There is nothing left for a port of this repo to deliver.

### Licence -- Apache-2.0, tier 1, cleared to contribute

Established as fact, not taken from GitHub's field:

    python3 utils/licenses.py check traveller59/spconv    # Apache-2.0, tier 1
    head -5 LICENSE                                       # 201-line standard Apache 2.0 text

- No submodules. `LICENSE` is the only licence/COPYING/NOTICE file in the whole tree,
  so no vendored component carries its own terms. (The EnvGS case -- permissive top
  level over an unlicensed submodule -- does not apply here.) Correction from screen 3:
  this bullet originally read "`.gitmodules` absent". `.gitmodules` is in fact
  *present but zero bytes*, with `git submodule status` empty and no `third_party/`.
  The conclusion is unchanged; only the stated evidence was wrong.
- `python3 utils/licenses.py scan-nvidia agent_space/spconv-screen` -> clean, no
  NVIDIA proprietary licence text.
- `cumm` (FindDefinition/cumm), where a port would actually land, is also Apache-2.0
  tier 1. Checked because it is a separate repo and its licence is not spconv's.

Recorded in `status.json.license_spdx`. Licence is not what decides this one.

### Duplicate effort -- this is what changed since screen 1

Screen 1 (2026-06-11) found no AMD sparse-conv path anywhere. That is no longer true.
Two independent ones now exist, both Triton-based, both permissive:

**1. `L-Reichardt/spconv-triton` -- Apache-2.0, PyPI `spconv-triton` 1.0.0 (2026-07-20)**

A drop-in replacement for spconv 2.3.8, not a fork. One-line adoption:

    # import spconv.pytorch as spconv        # before
    import spconv_triton.pytorch as spconv   # after

- All spconv 2.3.8 layer types, forward and backward, 1d-3d: SubM, regular sparse,
  transposed, inverse, pooling. FP32 / TF32 / FP16.
- **Verified on AMD MI300X.** Benchmark plots committed under `docs/MI300X/`
  (fp16/fp32/tf32 submanifold, C256, 50k voxels).
- End-to-end parity validated on real models: inference (Utonia, Cylinder3D),
  training (Uni3DETR), plus distributed training. Golden-tensor test suite
  (`tests/data/golden_*.pt`) checked against spconv itself.
- `tox` matrix covers both CUDA and ROCm runtimes. Handles the ROCm install trap
  explicitly -- does not declare `triton` as a dependency, because that would drag
  the CUDA wheel onto a ROCm install; relies on torch shipping `pytorch-triton-rocm`.
- Its README states the motivation in our own terms: "spconv is no longer maintained
  and its prebuilt CUDA kernels tie you to NVIDIA hardware", and flatly, "spconv does
  not run on AMD at all."

**2. `JeffreyXiang/FlexGEMM` -- MIT, 143 stars, active 2026-06-25**

"A Cross-Platform Backend for High-Performance Sparse Convolutions", Triton-first by
design. Downstream AMD work already exists on top of it: `Cardboard-box-a/FlexGEMM-rocm`
(HIP/ROCm kernels) and `ATLAS-0321/trellis2-amd`, which ships a FlexGEMM HIP port
targeting **gfx1100** -- our own screen platform. spconv-triton benchmarks itself
against FlexGEMM on MI300X and claims to beat it on submanifold conv while covering
more operators, so the two are live competitors in a space that is now contested
rather than empty.

**Checked and found nothing:**

- `AMD-Ecosystem`: no spconv. (Org holds `Pointcept`, `gtsam_points`, `dgSPARSE-Lib`.)
- `ROCm` org: no spconv, `spconv-rocm`, or `sparse-conv` (direct probes 404; org
  search is SAML-blocked from this host, so probes were by name).
- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/` on upstream: **zero hits.** No
  "notable forks" AMD link.
- Upstream PRs mentioning ROCm/HIP/AMD: none, ever.
- `jiaqiwang969/spconv-rocm`: **not a port.** One commit, README + `.gitignore` only,
  no licence, 0 stars, dead since 2026-02-05. Named like the real thing; is not.

Demand is real and upstream is ignoring it: issue **#780** (2026-06-09), "Request:
Windows HIP/ROCm build path for spconv + cumm (AMD consumer GPU support)", still
unanswered.

### Viability -- screen 1's finding re-verified, unchanged

The blocking fact is structural and has not moved:

- **One `.cu` in the entire repo**: `example/libspconv/main.cu`, a usage example.
- **18 `.py` pccm meta-programs** under `spconv/csrc/`. The "C++ sources" are Python
  that emits CUDA at build time.
- Build is `PCCMExtension` (`setup.py:16,212`), **not** a torch `CUDAExtension`.
  `ext_type: pccm-codegen` in status.json is correct.
- Kernels come from `cumm` (`deps = ["cumm>=0.7.11, <0.8.0"]`, `setup.py:44`), a
  CUTLASS-derived generator emitting NVIDIA tensor-core `mma.sync` tiles and inline PTX.

hipify has nothing to translate. Porting spconv still means writing an AMD codegen
backend for cumm's GEMM/implicit-conv generator -- **in a different repository**, and
unbounded. FlyDSL genuinely narrows that gap versus CK C++ templates, so the re-open
was a fair question to ask. But the answer it would buy is a capability two shipped,
permissively-licensed, AMD-validated libraries already provide.

**Upstream is dormant, so there is no destination either.** Not archived, but:

- Last push **2024-12-15** -- ~20 months.
- 195 open issues; PRs unmerged for a year-plus (#762 "add support for py3.12 and
  py3.13", open since 2025-09-01).

Even if the cumm backend existed, the PR could not land. Nobody should wait for one.

### The Pointcept dependency -- resolved, and it needs a person's edit

`moatlib.py deps` reports `Pointcept: depends_on=['spconv'] -> WAITING on spconv`,
and orient's upkeep flags spconv as a provider missing `## Install as a dependency`.
Both dissolve here:

- **Pointcept is already done.** `stage: review-passed`, upstream PR
  Pointcept/Pointcept#604 **merged 2026-07-06**, validated on `linux-gfx90a`. It
  completed and shipped while spconv sat unported, which is the evidence that spconv
  is an **optional backbone** (SparseUNet/MinkUNet), not a hard build dependency. The
  `depends_on` entry overstates it; per AGENTS.md an optional module dependency belongs
  in notes, not `depends_on`.
- Where a Pointcept user genuinely needs sparse convolution on AMD, the answer is the
  one-line `spconv_triton` import swap above -- no spconv port required.
- The missing `## Install as a dependency` section resolves by the decline: a declined
  project provides nothing to install.

**For a person:** the stale `Pointcept.depends_on = ['spconv']` entry should be cleared
(`moatlib.py set-deps Pointcept` with an empty list) so the dependency graph stops
showing a merged project as WAITING on a declined one. That is another project's record
and outside this screen's scope, so it is left alone here.

### Why `already-supported` and not `cant-port`

Both grounds hold independently, and either alone would decline this:

- `already-supported` -- two mature, permissive, AMD-validated sparse-conv libraries
  with spconv's API. Matches the skill's "a mature separate AMD project (ROCm-DS-style)
  -> skip (already-supported)" classification: the AMD support is a separately-named
  project, not a fork-of, which is exactly why screen 1's fork-oriented search missed
  a thing that did not exist yet.
- `cant-port` -- screen 1's finding, re-verified above, still true.

`already-supported` is the better record because it is the *current* fact and the one
that answers the user's question. `cant-port` says "we could not do this"; a future
screener reading it might reasonably re-open again on the next codegen advance, which
is precisely what happened this time. `already-supported` says "this no longer needs
doing", which is the durable answer and closes the loop.

If it is re-opened a third time, the question to ask first is not "has codegen improved"
but "**do spconv-triton and FlexGEMM still cover AMD?**" If they lapse, the value
returns -- and the tractable move even then is contributing gfx1100/wave32 coverage to
spconv-triton, not porting cumm.

### Screen 1's write-up

The 2026-06-11 analysis is still worth reading and stands unrefuted. The pointer in the
previous version of this file (`git show b40576d53399:...`) is **broken** -- that object
does not exist in this repository. The real location, from before the folder was pruned
in `27f7646`:

    git show 27f7646^:projects/spconv/plan.md
    git show 27f7646^:projects/spconv/notes.md

## Screen 3 -- 2026-08-13, intake, linux-gfx942 (verification of screen 2)

This host was dispatched spconv 13 minutes after screen 2 committed on `linux-gfx1100`
(screen 2 session ended 03:20:39Z; this one started 03:32:26Z). The screen was not
redone. Every load-bearing claim in screen 2 was re-checked independently from a
different host and a fresh shallow clone (`agent_space/spconv-screen-942`, upstream
HEAD `263d6b47425e`). **All of them hold.** The recommendation is unchanged:
**DECLINE, reason `already-supported`** -- still a recommendation, not a decision.

Confirmed independently:

| screen 2 claim | screen 3 result |
|---|---|
| Apache-2.0, tier 1 | `licenses.py check` -> `license=Apache-2.0 tier=1, cleared to contribute`; `LICENSE` is the standard 200-line text |
| no NVIDIA proprietary text | `scan-nvidia` -> clean |
| no vendored/submodule licences | `LICENSE` is the only licence file in the tree; see correction above |
| one `.cu` in the repo | exactly one: `example/libspconv/main.cu` |
| 18 pccm meta-programs | `find spconv/csrc -name '*.py'` -> 18 |
| `PCCMExtension`, not `CUDAExtension` | `setup.py:16` imports `PCCMExtension`; no `CUDAExtension` anywhere; `deps = ["cumm>=0.7.11, <0.8.0"]` at `setup.py:44` |
| zero AMD mentions in upstream docs | `grep -rniE 'amd\|rocm\|hip\|gfx[0-9]' README* docs/` -> no hits |
| upstream dormant since 2024-12-15 | HEAD commit dated `2024-12-15`, subject "change all build back to windows-2019" |
| spconv-triton is a real, AMD-validated drop-in | PyPI `spconv-triton` 1.0.0, released 2026-07-20, Apache-2.0, summary reads "drop-in replacement for spconv on NVIDIA and AMD"; README lists MI300X among verified hardware |
| FlexGEMM MIT / Triton | confirmed MIT, Triton-only (requires Triton >= 3.2.0), 143 stars |
| no AMD-Ecosystem or ROCm effort | `AMD-Ecosystem/spconv`, `ROCm/spconv`, `ROCm/spconv-rocm`, `AMD-Ecosystem/spconv-triton` all 404 |

One nuance worth carrying: FlexGEMM's own README makes no explicit AMD claim -- it
claims cross-platform via Triton, and the AMD specifics live in its downstream forks.
**spconv-triton, not FlexGEMM, is the load-bearing evidence for `already-supported`**,
and it carries the claim on its own (PyPI-released, MI300X-benchmarked, same API).
The decline does not depend on FlexGEMM at all.

### The re-dispatch loop -- for a person, not an agent to fix

A decline **recommendation** correctly leaves `stage: unclaimed`, because only a person
may write the disposition. But the selector treats `unclaimed` as actionable, so every
host that orients onto this project screens it again: three screens now, two of them
inside one hour, each reaching the same answer. The queue row has been correct and
waiting since 03:19Z.

Nothing in the pipeline should change to paper over this -- suppressing re-dispatch
without a recorded disposition would hide genuinely unscreened work. What clears it is
the thing that was always required: **a person answering the intake queue**
(`intake_queue.py publish --apply`, then `apply` to record the answers). spconv is
row 2 of 4 there. Until then, an agent handed spconv should read this file, verify
rather than re-derive, and stop -- as this screen did.
