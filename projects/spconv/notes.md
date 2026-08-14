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

## Screen 4 -- 2026-08-14, intake, linux-gfx90a (verification of screens 2-3)

Fourth dispatch, same cause as screen 3: `stage: unclaimed` is actionable to the
selector and no person has answered the queue yet. Verified rather than re-derived.
Fresh shallow clone `agent_space/spconv-screen4`, upstream HEAD **`263d6b4`
(2024-12-15)** -- byte-identical to screen 3's `263d6b47425e`, so upstream has not
moved in the intervening day or the intervening 20 months.

**Recommendation unchanged: DECLINE, reason `already-supported`.** Still a
recommendation, not a decision. Nothing was written to `dispositions.json`.

Re-verified independently on this host:

| claim | screen 4 result |
|---|---|
| Apache-2.0, tier 1 | `licenses.py check traveller59/spconv` -> `license=Apache-2.0 tier=1, cleared to contribute` |
| cumm Apache-2.0, tier 1 | `licenses.py check FindDefinition/cumm` -> same |
| no NVIDIA proprietary text | `scan-nvidia agent_space/spconv-screen4` -> clean |
| `LICENSE` is the only licence file | confirmed; `.gitmodules` present but **0 bytes**, `git submodule status` empty, no `third_party/` |
| one `.cu` in the repo | exactly one: `example/libspconv/main.cu` |
| 18 pccm meta-programs | `find spconv/csrc -name '*.py'` -> 18 |
| `PCCMExtension`, not `CUDAExtension` | `setup.py:16` imports `PCCMExtension`; `setup.py:212` uses it; no `CUDAExtension`; `setup.py:42/44` pin `cumm-cu*>=0.7.11,<0.8.0` / `cumm>=0.7.11,<0.8.0` |
| zero AMD mentions in upstream docs | `grep -rniE 'amd\|rocm\|hip\|gfx[0-9]' README* docs/` -> no hits |
| upstream dormant | `pushed_at` **2024-12-15**, 195 open issues, 2290 stars, `archived: false` |
| no AMD-Ecosystem / ROCm effort | `AMD-Ecosystem/spconv`, `ROCm/spconv`, `ROCm/spconv-rocm`, `AMD-Ecosystem/cumm`, `ROCm/cumm`, `AMD-Ecosystem/spconv-triton` -> all 404 |
| spconv-triton real and AMD-verified | Apache-2.0, `pushed 2026-07-20`; PyPI 1.0.0 summary reads "drop-in replacement for spconv on NVIDIA and AMD"; `docs/MI300X/` holds `fp16/fp32/tf32_subm_C256_v50000.png` |
| FlexGEMM MIT | confirmed MIT, 143 stars, pushed 2026-06-25 |
| `jiaqiwang969/spconv-rocm` is not a port | confirmed: **no licence**, 0 stars, untouched since 2026-02-05 |
| no disposition, no opt-out | `spconv` absent from all 282 `dispositions.json` entries; no opt-out record |

### Correction to screen 2: issue #780 was answered, and the answer was spconv-triton

Screen 2 wrote that upstream issue #780 ("Request: Windows HIP/ROCm build path for
spconv + cumm") was "still unanswered". That is now wrong in a way that **strengthens**
the recommendation rather than weakening it:

- The issue is **closed**, but not by a maintainer -- the reporter closed it accidentally
  on 2026-06-09, the day he opened it, and said so.
- On **2026-07-09** `L-Reichardt` replied to that exact AMD request:

  > spconv's CUDA dependency inspired me to write a drop-in replacement in Triton. Its
  > still in development, but I verified it on AMD-Server yesterday. It should work on
  > your consumer hardware [...] -> spconv-Triton

So the one recorded instance of a user asking upstream spconv for AMD support was
answered by being handed spconv-triton. That is `already-supported` demonstrated on the
actual demand signal, not inferred from it. **No spconv maintainer ever replied** --
both comments are from non-maintainers, which is the dormancy finding again.

The same thread also confirms screen 2's trellis2 lineage: the reporter notes Trellis2
"appears to use a different convolution library" and worked on his machine, while
stock Trellis (on spconv) did not.

### Honest soft spot: spconv-triton has 1 GitHub star

Recorded because a person deciding this should see it. spconv-triton is real -- PyPI
`spconv-triton` 1.0.0, Apache-2.0, MI300X plots committed, one-line import swap -- but
it is a **single-author, single-release, low-adoption** project: one release
(2026-07-20, no updates since), 1 star, deps only `numpy>=1.24` and `torch>=2.4`.
"Mature" in the sense of covering the API and being AMD-verified; **not** mature in the
sense of a community standing behind it.

This weakens the `already-supported` framing on its own, and is the fair counterargument
to screens 2-3. It does not change the recommendation, because **`cant-port` holds
independently and structurally**: one `.cu`, 18 pccm meta-programs, `PCCMExtension`, and
all real kernels generated by cumm's CUTLASS-derived `mma.sync`/inline-PTX emitter in a
different repository. There is nothing for hipify to translate here no matter what
happens to spconv-triton. If a person prefers the record to say `cant-port` for exactly
that reason, that is a defensible answer to the queue row and the write-up supports it.

### New nuance: cumm is alive, spconv is not

Not noticed by earlier screens. `FindDefinition/cumm` was pushed **2026-03-21** (vs
spconv's 2024-12-15) and is Apache-2.0 tier 1, 86 stars, not archived. So the repo where
an AMD codegen backend would actually land is maintained and could in principle accept a
PR, even though spconv itself cannot.

**Does cumm need its own intake?** Not as a consequence of this screen. It is not in
`data/candidates.json`, has no disposition, and no AMD-Ecosystem/ROCm fork exists. It was
not dispatched here and this screen does not scaffold or claim it. If anyone ever wants
to revisit AMD sparse convolution by the codegen route, **cumm -- not spconv -- is the
candidate to screen**, and it should get its own intake rather than riding on this one.
That work is an AMD backend for a CUTLASS-derived GEMM/implicit-conv generator: unbounded,
and still redundant with spconv-triton today.

### Pointcept dependency -- state unchanged, still needs a person's edit

Re-checked, exactly as screen 2 described:

- `moatlib.py deps` -> `Pointcept: depends_on=['spconv'] -> WAITING on spconv`.
- Pointcept is `stage: review-passed`, PR Pointcept/Pointcept#604 **merged 2026-07-06**,
  validated on `linux-gfx90a` -- it shipped without spconv, which is the proof spconv is
  an optional backbone (SparseUNet/MinkUNet), not a hard build dependency.
- `DEPENDENCIES.md` contains **no** spconv mention at all, so the stale edge lives only
  in `Pointcept/status.json`.
- spconv's own `depends_on` is `[]`, which is correct; no `set-deps` needed here.

Still the recommended person's edit: clear `Pointcept.depends_on` so a merged project
stops showing as WAITING on a declined one. Another project's record, so untouched here.

### Queue status

spconv remains **row 2 of 4** on the single intake queue issue,
`AMD-Ecosystem/moat` issue **#8** (`intake_queue.py publish` reports `would-update`).
The row has been correct and waiting since 2026-08-13 03:19Z. No per-project PR was
opened. Nothing further should happen to this project until a person answers that issue.
