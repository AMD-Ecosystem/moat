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

## Screen 5 -- 2026-08-14, intake, linux-gfx942 (verification of screens 2-4)

Fifth dispatch, same cause as screens 3 and 4: `stage: unclaimed` is actionable to the
selector and no person has answered the queue yet. Verified rather than re-derived, from
a fresh shallow clone `agent_space/spconv-screen5`, upstream HEAD **`263d6b4`
(2024-12-15)** -- identical to screens 3 and 4.

**Recommendation unchanged: DECLINE, reason `already-supported`.** Still a
recommendation, not a decision. Nothing was written to `dispositions.json`.

Re-verified independently on this host:

| claim | screen 5 result |
|---|---|
| Apache-2.0, tier 1 | `licenses.py check traveller59/spconv` -> `license=Apache-2.0 tier=1, cleared to contribute`; GitHub API `license.spdx_id` agrees |
| cumm Apache-2.0, tier 1 | `licenses.py check FindDefinition/cumm` -> same |
| no NVIDIA proprietary text | `scan-nvidia agent_space/spconv-screen5` -> `no NVIDIA proprietary licence text` |
| no vendored/submodule licences | `LICENSE` is the only licence/COPYING/NOTICE file in the tree; `.gitmodules` present but **0 bytes**, `git submodule status` empty, no `third_party/` |
| one `.cu` in the repo | exactly one: `example/libspconv/main.cu` |
| 18 pccm meta-programs | `find spconv/csrc -name '*.py'` -> 18 |
| `PCCMExtension`, not `CUDAExtension` | `setup.py:16` imports it, `setup.py:212` uses it; `grep -rn CUDAExtension` -> **none anywhere**; `setup.py:42/44` pin `cumm-cu*>=0.7.11,<0.8.0` / `cumm>=0.7.11,<0.8.0` |
| zero AMD mentions in upstream docs | `grep -rniE 'amd\|rocm\|hip\|gfx[0-9]' README* docs/` -> no hits |
| upstream dormant | `pushed_at` **2024-12-15T15:41:19Z**, 195 open issues, 2290 stars, `archived: false` |
| no AMD-Ecosystem / ROCm effort | `AMD-Ecosystem/spconv`, `ROCm/spconv`, `ROCm/spconv-rocm`, `AMD-Ecosystem/cumm`, `ROCm/cumm`, `AMD-Ecosystem/spconv-triton`, `ROCm/FlexGEMM` -> all 404 |
| no upstream AMD PRs | GitHub search `repo:traveller59/spconv is:pr ROCm OR HIP OR AMD` -> `total_count: 0` |
| spconv-triton real, Apache-2.0 | pushed 2026-07-20, 1 star, Apache-2.0 |
| FlexGEMM MIT | pushed 2026-06-25, 143 stars, MIT |
| cumm alive | pushed 2026-03-21, 86 stars, not archived |
| `jiaqiwang969/spconv-rocm` is not a port | `license: NONE`, 0 stars, untouched since 2026-02-05 |
| no disposition, no opt-out | `spconv` and `cumm` both absent from all 282 `dispositions.json` entries; no opt-out record |

Screen 4's **correction to screen 2 stands, re-read in full**: issue #780 is `closed`,
closed accidentally by the reporter (`ryanmcandrew`, 2026-06-11), and answered on
2026-07-09 by `L-Reichardt` pointing him at spconv-triton. **No spconv maintainer ever
replied to either comment.** That is the `already-supported` finding demonstrated on the
one recorded demand signal, and the dormancy finding, in the same thread.

### Duplicate-effort search re-run broadly, not just by name

Because duplicate effort is the load-bearing claim, the search was widened rather than
repeating screen 4's direct probes. GitHub repository search for `spconv rocm` and
`spconv triton`, sorted by recency, returns **exactly two repos in the whole of GitHub**:
`L-Reichardt/spconv-triton` (the real one) and `jiaqiwang969/spconv-rocm` (the empty
stub). No new AMD sparse-conv effort has appeared since screen 4, and nothing was missed
by the earlier name-probe method.

### Nothing new, and that is the finding

Five screens on four platforms (gfx1100, gfx942, gfx90a, gfx942) now agree on every
load-bearing fact, and upstream has not moved in 20 months. Screen 4's honest soft spot
-- spconv-triton is single-author, single-release, 1 star -- is unchanged and still the
fair counterargument; `cant-port` still holds independently and structurally, so either
`SKIP_REASON` is defensible and the decline does not turn on that choice. This screen
adds no new argument because there is none to add.

The remaining person's edits are both still outstanding and both still outside this
screen's scope: answer intake queue issue **#8** (spconv is row 2 of 4), and clear the
stale `Pointcept.depends_on = ['spconv']` so a merged project
(Pointcept/Pointcept#604, merged 2026-07-06, validated `linux-gfx90a`) stops showing as
`WAITING on spconv`.

## Screen 6 -- 2026-08-19, intake, linux-gfx1100

Sixth dispatch. **Recommendation unchanged: DECLINE, reason `already-supported`**, still
a recommendation and not a decision -- nothing was written to `dispositions.json`.

The cause of this dispatch is new and worth recording: this host was handed spconv
**with the Pointcept dependency as the stated motivation** ("spconv is recorded as a hard
dependency of Pointcept, which is waiting on it"). Screens 2, 4 and 5 each flagged that
edge as stale bookkeeping for a person to clear. It is now demonstrably more than
cosmetic -- it is *generating* the re-screens it was reported in. That raises the
priority of the edit without changing the recommendation.

Re-verified on this host (all identical to screen 5, five days on):

| claim | screen 6 result |
|---|---|
| Apache-2.0, tier 1 | `licenses.py check traveller59/spconv` -> `license=Apache-2.0 tier=1, cleared to contribute` |
| upstream dormant | `pushed_at` **2024-12-15T15:41:19Z**, 195 open issues, 2290 stars, `archived: false` -- unmoved |
| cumm alive, Apache-2.0 | `pushed_at` 2026-03-21, 87 stars, not archived |
| spconv-triton real, Apache-2.0 | `pushed_at` 2026-07-20, 1 star -- unmoved |
| no AMD-Ecosystem / ROCm effort | `AMD-Ecosystem/spconv`, `ROCm/spconv`, `AMD-Ecosystem/cumm`, `ROCm/cumm` -> all 404 |
| GitHub-wide `spconv rocm` | `total_count: 2` -- `L-Reichardt/spconv-triton` and the empty `jiaqiwang969/spconv-rocm` stub. Unchanged |
| no disposition | `spconv` and `cumm` absent from all 282 `dispositions.json` entries |
| queue unanswered | issue **#8** open, **0 comments**, last updated 2026-08-14T17:41Z; `intake_queue.py publish` reports `would-update` |

The licence and NVIDIA-text findings were established from fresh shallow clones on three
prior hosts (screens 3, 4, 5) against the same upstream HEAD `263d6b4`, which has not
moved since 2024-12-15. No re-clone was made here; re-scanning an unchanged tree a fourth
time is not verification.

### Correction to screens 2/4/5: the Pointcept edge is stale, but not empty

Earlier screens described `Pointcept.depends_on = ['spconv']` as purely stale because
Pointcept shipped without spconv. The first half holds and is now stronger than recorded
-- Pointcept is `stage: review-passed` with PR #604 merged 2026-07-06 and **four**
platforms completed (`linux-gfx90a`, `linux-gfx1100`, `windows-gfx1101`,
`windows-gfx1201`), not the single platform screens 2-5 cited. But "stale" flattened
something real. Pointcept's own `notes.md` says:

> pointseg is a CppExtension (CPU only, no `.cu`) and needs no port. spconv is an
> external dependency (separate MOAT project, unported) and does NOT block these libs --
> the four libs build, install, and pass their op tests without it. Sparse-conv MODEL
> configs (SpUNet/OACNN/PointGroup end-to-end) wait on the spconv ROCm port; that is out
> of scope for this port's validation.

So the edge encodes a **genuine remaining capability gap** -- end-to-end sparse-conv model
configs on AMD -- not a bookkeeping error alone. Two further facts sharpen it:

- `pointcept/models/utils/structure.py:2` does an **unguarded** `import spconv.pytorch as
  spconv` at module top level (contrast `ocnn` on line 4, which is in a `try/except`), and
  that module is used by PointTransformerV3. spconv is not an optional import in the
  Python sense; what made Pointcept's validation possible is that the ops under `libs/`
  are separable from it, not that the import is guarded.
- The gap is recorded **only as prose in Pointcept's notes**. There is no
  `projects/Pointcept/deferred.json` -- the folder holds `notes.md`, `plan.md`,
  `stats.jsonl`, `status.json` and nothing else. An unregistered gap is one nobody
  reviews.

**None of this argues for porting spconv.** It argues that the gap's owner is Pointcept
and its resolution is spconv-triton. Verified from the spconv-triton README today: it
covers submanifold, regular, transposed and inverse sparse convolution plus pooling in
1D-4D, which is the whole operator set SpUNet/OACNN/PointGroup use, and the migration is
the one-line `import spconv_triton.pytorch as spconv` swap. Its own note that it
deliberately does not declare Triton as a dependency, because `pytorch-triton-rocm` is
what a ROCm install already carries, is the detail that makes the swap work on AMD rather
than pulling a CUDA wheel.

The honest gap in that answer, and the concrete opt-in surface: spconv-triton lists
**MI300X** as its verified AMD hardware. **gfx1100/RDNA3 and wave32 are unverified**, and
this dispatch was to a gfx1100 host. Validating the swap for Pointcept's SpUNet configs on
gfx1100 is bounded, testable work with a real recipient, and it is the tractable move that
screen 2 already predicted ("contributing gfx1100/wave32 coverage to spconv-triton, not
porting cumm").

### For a person -- three edits, all outside this screen's scope

1. Answer intake queue issue **#8**. spconv is row 2 of 4 and the row has been correct and
   waiting since 2026-08-13 03:19Z. This is what stops screen 7.
2. Clear `Pointcept.depends_on` (`moatlib.py set-deps Pointcept` with an empty list) so a
   merged, four-platform-validated project stops showing as `WAITING on` a project
   recommended for decline -- and stops sourcing dispatches like this one.
3. Register the sparse-conv gap where it belongs, as a Pointcept deferral rather than as
   a dependency on a declined project: validate the `spconv_triton` import swap for the
   SpUNet/OACNN/PointGroup configs on AMD, gfx1100 included. Not done here because
   `deferred.py add --project Pointcept` writes another project's record from this branch,
   and defer-versus-now is a person's ruling either way.

### Nothing else new

Six screens on four platforms now agree on every load-bearing fact. Screen 4's honest soft
spot -- spconv-triton is single-author, single-release, 1 star -- is unchanged and remains
the fair counterargument; `cant-port` still holds independently and structurally, so the
decline does not turn on which `SKIP_REASON` is chosen. An agent handed spconv again
should read this file, confirm the queue is still unanswered, and stop.

## Screen 7 -- 2026-08-20, intake, linux-gfx90a (verification of screens 2-6, plus three new checks)

Seventh dispatch, same cause as screens 3-6: `stage: unclaimed` is actionable to the
selector and intake queue issue **#8** is still open with **0 comments** (last updated
2026-08-19T04:58Z, which is screen 6's own row refresh -- no person has replied). Verified
rather than re-derived, from a fresh shallow clone `agent_space/spconv-screen7`.

**Recommendation unchanged: DECLINE, reason `already-supported`.** Still a recommendation,
not a decision. Nothing was written to `dispositions.json`; no GitHub write of any kind.

Upstream HEAD is **`263d6b47425ef843c82f997b12d8b714013d216c` (2024-12-15)** -- byte-identical
to screens 3, 4 and 5. Twenty months and seven screens, one commit.

Re-verified on this host:

| claim | screen 7 result |
|---|---|
| Apache-2.0, tier 1 | `licenses.py check traveller59/spconv` -> `license=Apache-2.0 tier=1, cleared to contribute` |
| cumm Apache-2.0, tier 1 | `licenses.py check FindDefinition/cumm` -> same |
| no NVIDIA proprietary text | `scan-nvidia agent_space/spconv-screen7` -> `no NVIDIA proprietary licence text` |
| no vendored/submodule licences | `LICENSE` is the only licence/COPYING/NOTICE file in the tree; `.gitmodules` present but **0 bytes**, `git submodule status` empty, no `third_party/` |
| one `.cu` in the repo | exactly one: `example/libspconv/main.cu`; zero `.cuh` |
| 18 pccm meta-programs | `find spconv/csrc -name '*.py'` -> 18 |
| zero AMD mentions in upstream docs | `grep -rniE 'amd\|rocm\|hip\|gfx[0-9]' README* docs/` -> no hits |
| upstream dormant | `pushed_at` **2024-12-15T15:41:19Z**, 195 open issues, 2290 stars, `archived: false` -- unmoved |
| cumm alive, Apache-2.0 | `pushed_at` 2026-03-21, 87 stars, not archived |
| spconv-triton unmoved | `pushed_at` 2026-07-20, 1 star, 1 fork, Apache-2.0 |
| FlexGEMM | MIT, `pushed_at` 2026-06-25, 146 stars (143 at screen 3) |
| no AMD-Ecosystem / ROCm effort | `AMD-Ecosystem/{spconv,cumm,spconv-triton,FlexGEMM}`, `ROCm/{spconv,spconv-rocm,cumm,FlexGEMM}` -> all 404 |
| GitHub-wide `spconv rocm` | `total_count: 2` -- `L-Reichardt/spconv-triton` and the empty unlicensed `jiaqiwang969/spconv-rocm` stub. Unchanged |
| no disposition, no opt-out | `spconv` and `cumm` absent from all 282 `dispositions.json` entries; spconv absent from `data/candidates.json` |
| spconv `depends_on` | `[]`, which is correct -- no `set-deps` needed on this record |

### New check 1: upstream OPEN pull requests and BRANCHES, enumerated

Screen 5 ran a PR *search* and got `total_count: 0`. This screen enumerated the full open
set and the remote branch list on both repos, so the absence is from a listing rather than
from a query that might have missed an unindexed PR.

    gh pr list --repo traveller59/spconv --state open --search "hip OR rocm OR amd"   # []
    gh pr list --repo traveller59/spconv --state open --limit 100                     # 10 PRs
    git ls-remote --heads https://github.com/traveller59/spconv
    git ls-remote --heads https://github.com/FindDefinition/cumm

- **10 open PRs, none AMD/ROCm/HIP.** In order: #762 py3.12/3.13, #749 version compat,
  #742 conv_relu, #728 `amp.custom_fwd` warning, #714 / #709 SparseGlobalAvgPool, #643
  SparseAvgPool, #634 fuse-BN, #631 conv.py, #616 macOS CPU-only builds. (#728's branch
  `fix/amp_custom_fwd` is *automatic mixed precision*, not AMD -- the near-miss a loose
  grep would report.)
- **spconv branches**: `master`, `develop`, `2.1.x`, `v1.1`, `v1.2.1`, `feature/ampere`,
  `feature/bf16`. **cumm branches**: `main`, `0.2.x`, `feature/ampere`, `feature/debug`.
  No `rocm`, `hip`, `amd` or `gfx*` branch on either. The only architecture branch anyone
  ever cut is `feature/ampere`, on both repos.

### New check 2: upstream has not merged a pull request since 2022

The strongest dormancy evidence yet, and not recorded by screens 1-6, which all cited the
2024-12-15 push date:

    gh pr list --repo traveller59/spconv --state merged --limit 5
    547  2022-12-27  Large kernel for implicit gemm
    515  2022-09-22  Feature/Ampere
    363  2021-11-07  v2.1

**Three years and eight months without merging an external contribution.** The 2024-12-15
push was the maintainer's own CI touch-up ("change all build back to windows-2019"), not a
merge. PR #616, "Enable CPU-only builds on macOS" -- a small, self-contained *platform*
contribution, exactly the shape a ROCm PR would take -- has sat open the whole time.

This settles the "archived upstream still has fork value" clause in the role definition.
spconv is not archived, so a PR is technically openable; the merge record says it would not
land. Nobody should wait for one, and no one should read "not archived" here as a
destination.

### New check 3: functionality search, not name search, in the AMD orgs

Screens 2-6 probed by *name* (`ROCm/spconv`, `AMD-Ecosystem/cumm`, ...). This screen
searched the orgs for the *capability*, which is the method that caught FlashKDA's
duplicate:

    gh api "search/repositories?q=org:ROCm+sparse"          # 7
    gh api "search/repositories?q=org:AMD-Ecosystem+sparse" # 0
    gh api "search/repositories?q=org:ROCm+{submanifold,voxel,point+cloud,spconv}"  # 0 each
    gh api "search/code?q=repo:ROCm/aiter+{spconv,submanifold}"                     # 0 each
    gh api "search/repositories?q=%22sparse+convolution%22+rocm"                    # 1

- ROCm's seven "sparse" repos are all sparse **linear algebra** -- `rocSPARSE`,
  `hipSPARSE`, `hipSPARSELt` (all three now deprecated into `ROCm/rocm-libraries`),
  `rocALUTION`, `hipfort`, and two internal benchmark tools. None is sparse
  **convolution**: no hash-table voxel indexing, no submanifold rule generation. Adjacent
  name, different problem.
- `ROCm/aiter` ("AI Tensor Engine for ROCm", pushed today) has **zero** hits for `spconv`
  or `submanifold`. It is the natural home if AMD ever took this on, and it has not.
- GitHub-wide, the only repo describing itself as sparse convolution on ROCm is
  `L-Reichardt/spconv-triton` -- the same answer the name search gives.

This cuts both ways and the second half is the honest part. It confirms there is **no AMD
first-party effort to duplicate or coordinate with** -- so this is not a race, and the
`already-supported` claim still rests entirely on one outside project. It equally confirms
there is **no AMD-owned place to hand the gap to** if spconv-triton lapses.

### spconv-triton re-read on this host: MI300X only, still the whole AMD claim

From today's README (not from the earlier screens' summaries):

> Verified with: NVIDIA RTX 3060, A100, H100, L4, B200 - **AMD MI300X** - torch 2.4-2.13 - python 3.10-3.14

> spconv does not run on AMD at all. On MI300X, measured against FlexGEMM on submanifold
> convolution, the one common operator, spconv-Triton is faster and additionally covers
> the full operator set.

Operator coverage confirmed as claimed: submanifold and regular sparse convolution 1D-4D,
transposed, inverse, pooling; parity against spconv 2.3.8 via a frozen golden-data suite;
`tox` covers CUDA and ROCm runtimes; Triton is deliberately undeclared as a dependency so a
ROCm install keeps its `pytorch-triton-rocm`. Two AMD-specific operational details worth
carrying to whoever validates the swap: TF32 on torch 2.7+ AMD also needs
`HIPBLASLT_ALLOW_TF32=1`, and the flag must be set on `spconv_triton`, not on the `spconv`
alias, or it silently does nothing.

**The verified-hardware line is MI300X (gfx942) alone.** gfx90a -- this host -- is
unverified, as are gfx1100/wave32. That is the same bounded opt-in surface screens 2 and 6
identified, now confirmed against the source of truth rather than a prior note.

### Dependency context -- unchanged, and the resolution still belongs to Pointcept

`moatlib.py deps` still reports `Pointcept: depends_on=['spconv'] -> WAITING on spconv`.
`DEPENDENCIES.md` still contains no spconv mention, so the edge lives only in
`Pointcept/status.json`. Pointcept is `stage: review-passed`, PR Pointcept/Pointcept#604
merged 2026-07-06, `head_sha` 95f4a51, four platforms completed.

Screen 6's correction stands and is the right framing: the edge is **stale as a blocker but
real as a capability gap**. What providing spconv would actually entail is unchanged and
unbounded -- an AMD codegen backend for cumm's CUTLASS-derived `mma.sync`/inline-PTX
emitter, in a different repository, with nothing for hipify to translate in spconv itself.
What the *gap* needs is the one-line `import spconv_triton.pytorch as spconv` swap,
validated for SpUNet/OACNN/PointGroup on AMD. Those are not the same job, and only the
second one has a bounded scope and a named recipient.

`projects/Pointcept/` on `origin/main` still holds only `notes.md`, `plan.md`,
`stats.jsonl`, `status.json` -- **no `deferred.json`**, so the gap remains unregistered
five days after screen 6 reported it.

### For a person -- the same three edits, none of them this screen's to make

1. Answer intake queue issue **#8**. spconv's row has been correct and waiting since
   2026-08-13 03:19Z. This is the only thing that stops screen 8.
2. Rule on `Pointcept.depends_on = ['spconv']`. Note this is *not* purely cosmetic:
   `pointcept/models/utils/structure.py:2` imports spconv unguarded at module top level,
   so for the sparse-conv model configs it is a genuine hard runtime dependency, not an
   optional one. Clearing it is a scoping ruling, which is why six screens have declined to
   make it.
3. Register the gap as a Pointcept deferral -- validate the `spconv_triton` swap for the
   SpUNet/OACNN/PointGroup configs on AMD, gfx90a and gfx1100 included, neither of which
   spconv-triton claims. `deferred.py add --project Pointcept` writes another project's
   record from this branch, and defer-versus-now is a person's ruling either way.

### Standing conclusion

Seven screens on four platforms agree on every load-bearing fact, and the two new checks
here only strengthen the case: no AMD work exists anywhere upstream (no PR, no branch, no
merge in 44 months) and none exists in the AMD orgs by capability either. Screen 4's honest
soft spot -- spconv-triton is single-author, single-release, 1 star, MI300X-only -- is
unchanged and remains the fair counterargument to `already-supported`; `cant-port` holds
independently and structurally, so the decline does not turn on which `SKIP_REASON` a person
picks. An agent handed spconv again should read this file, confirm issue #8 is still
unanswered, and stop.
