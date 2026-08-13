# diff-surfel-rasterizations notes

A dependency fork: the rasterizer EnvGS's diffuse path renders through.

## Why the platform row is empty

This has no test suite of its own. The code is exercised only through the project
that consumes it, so a GPU run against this repository alone would prove nothing.
The empty row is accurate, not a gap in the record.

The validation lives with **EnvGS**, `completed` on linux-gfx1100, linux-gfx90a, windows-gfx1101, windows-gfx1201.

## Port state

The `moat-port` branch predates this project being tracked here, so the port exists
but its provenance was not recorded: no plan, no dated validation entry, no note of
which commit was tested. Treat it as real work of unverified state rather than as a
validated port.

## Intake screen 2026-08-13 (linux-gfx1100)

### Licence -- the recorded value was wrong, now corrected

`license_spdx` read `no licence file`. That is not what is in the tree.

`utils/licenses.py check xbillowy/diff-surfel-rasterizations` returns
`license=UNPARSED (GitHub returned NONE)`, which is the case the intake role warns
about: GitHub could not classify it, so the field had to be read by hand. Shallow
clone of upstream, then `find -iname '*licen[cs]e*'`:

- There is no top-level `LICENSE`. **Each of the 14 variant directories carries its
  own `LICENSE.md`**, and all 14 are byte-identical (`md5 3aa09743544511a59899c8746ee49a80`).
- That file is the **Gaussian-Splatting License**, Inria + Max Planck Institut fuer
  Informatik (MPII), research/evaluation use only, "The *Software* may be used
  'non-commercially', i.e., for research and/or evaluation purposes only." Section 4.2
  propagates the non-commercial use limitation to derivative works.
- Each `setup.py` repeats it in the header: "Copyright (C) 2023, Inria / GRAPHDECO
  research group ... free for non-commercial, research and evaluation use".

Recorded as `non-commercial (Inria Gaussian-Splatting License)`. This is **not mixed
or ambiguous licensing** -- every code-bearing directory carries the same text, so
there is nothing here to stop and ask about. Uniform, just not at the root, which is
exactly why GitHub returned NONE.

The tier is unchanged: `licenses.tier_of` puts both the old string and the new one at
**tier 4** (not open source / non-commercial), so the licence clearance already on
this record (jeffdaily, 2026-08-06, tier 4, contribute-only, "covers this project
only") still applies untouched. `license-gate` reports `license-ok=True` before and
after. The correction is a fact repair, not a change of standing.

Tier 4 consequence worth stating once, because it decides what this project can ever
be: contributing upstream requires the copyright holder's permission by email, never a
PR or an issue. Related note under "No upstream destination" below.

Per-file scans, both clean:

- `licenses.py scan-nvidia` over the shallow clone: no NVIDIA proprietary licence text.
- Submodule recursed (`git clone --recurse-submodules`). The only submodule is
  `third_party/glm` -> `g-truc/glm`, under The Happy Bunny License **or** MIT
  (`third_party/glm/copying.txt`). Permissive, header-only, and unmodified. This is
  the EnvGS failure mode -- permissive root over an unlicensed submodule -- and it
  does not occur here.

### Duplicate effort -- the existing AMD port is OURS

- Upstream has **two** forks: `rhombus19/diff-surfel-rasterizations` (branch `main`
  only, no port branch) and `AMD-Ecosystem/diff-surfel-rasterizations`. No third-party
  AMD effort.
- `grep -rniE 'amd|rocm|hip|gfx[0-9]'` over the upstream tree, excluding `third_party/`:
  zero hits outside the LICENSE.md text ("Max Planck ..." matching `amd`). No linked
  "notable forks", no AMD-support section.
- Upstream has **no pull requests and no issues, ever** (`gh pr list --state all`,
  `gh issue list --state all` both empty).
- Web search for a separately-named ROCm/HIP surfel rasterizer found nothing.
- `data/dispositions.json`: no entry. `utils/optout.py`: no opt-out for this upstream
  or owner.

So this falls in the `assess-existing-support.md` category "the existing AMD support
IS OURS": `AMD-Ecosystem/diff-surfel-rasterizations` @ `moat-port` is 3 commits ahead
of `main`, and the fork's `main` is at `1aa433c`, identical to upstream HEAD (clean
mirror, as required).

    d7e9f1a [ROCm] Port diff-surfel 2DGS rasterizer to HIP (gfx90a)
    98f0c05 [ROCm] Fix Windows/HIP ext.cpp ABI mismatch in rasterizer setup.py
    4c95346 [ROCm] Document AMD GPU support and add AMD attribution

22 files, +275/-48, covering **3 of the 14** variant directories:
`diff-surfel-rasterization-wet`, `-wet-ch05`, `-wet-ch07`. Those are precisely the
three EnvGS installs (EnvGS notes: byte-identical sources differing only in
`NUM_CHANNELS` = 3/5/7).

This is not a race and not another team's work -- it is MOAT's own stage-1 EnvGS port
that was never given a record of its own.

### Viability

- Genuinely CUDA: 56 `.cu`/`.cuh` files outside `third_party/`, and every variant is a
  torch `CUDAExtension` (`setup.py` -> `torch.utils.cpp_extension.CUDAExtension`).
  `ext_type` should become `torch-extension`; left `unknown` for the planner since it
  is a scope statement, not an intake fact.
- Upstream is **not archived**, but it is quiet and small: 2 stars, 2 forks, last push
  2025-03-30, zero PRs, zero issues, a one-line README.
- Dependencies: `depends_on` stays `[]`. The dependency arrow points the other way --
  **EnvGS consumes this**, as a git submodule. That edge is not currently recorded on
  EnvGS's record (its `status.json` has no `depends_on` key at all, and
  `moatlib.py deps` prints no surfel edge), so `DEPENDENCIES.md` does not show the
  relationship that both notes files describe. Flagged rather than fixed: writing it
  belongs on EnvGS's own branch, not this one.
- The sibling `diff-surfel-tracing` (EnvGS's OptiX reflection path, MIT) is being
  screened concurrently. Not touched here.

### No upstream destination -- the thing that actually decides this

Two independent reasons this port has no realistic upstream PR:

1. **Licence.** Tier 4 non-commercial. Contributing requires the copyright holder's
   written permission by email. The substantive copyright is Inria/MPII GRAPHDECO's,
   not the repository owner's -- `xbillowy/diff-surfel-rasterizations` is an
   aggregation of Inria-derived variants.
2. **The repository.** 2 stars, no PR or issue has ever been opened against it. It is
   one researcher's collection of channel-count variants, not a maintained library
   accepting contributions.

So nobody should wait for a PR here. The value is the fork itself: EnvGS builds
end-to-end from it, and anyone wanting 2DGS surfel rasterization on AMD can build from
`moat-port`. Say so up front rather than letting a later stage discover it.

### Recommendation: fork (take it up), recorded via `set-intake`

The adoption decision was in fact already taken -- the fork exists and carries a real
port. What intake is proposing is the smaller thing: give this project a record of its
own instead of leaving it `unclaimed` behind EnvGS. Concretely, what is left is not a
from-scratch port:

- The 3 EnvGS variants are ported and were exercised on real GPUs through EnvGS at
  `7528e8db` on linux-gfx1100, linux-gfx90a, windows-gfx1101, windows-gfx1201 -- but
  no `head_sha`, plan, or dated validation entry ties any of that to *this* record.
- **11 of the 14 variants are unported.** A user of any other channel count gets
  nothing on AMD. Whether to cover them is a scope question for the planner, and it is
  the only genuinely new porting work here.

State set to `screened` rather than `awaiting-fork`, because the fork already exists.

### The alternative answer, stated plainly

A reviewer could reasonably decline this instead, and the reason would be
**`not-a-target`**: it is a dependency submodule rather than a standalone port target,
it has no test suite of its own so nothing can be validated against this repository
alone, there is no upstream that can receive a PR, and its value has already been
delivered through EnvGS. If that is the call, the port on `moat-port` should stay put
and EnvGS should keep owning it -- a decline here must not be read as a reason to
delete or stop maintaining the fork branch that EnvGS builds from.

Intake does not record either answer. Both are on the queue row.

### Gap in this screen, for the record

`gh search repos ... --owner ROCm` and `gh api search/repositories?q=...+org:ROCm`
both fail from this host with HTTP 422, "Access to the requested organization (ROCm)
is blocked by SAML single sign-on". The ROCm-org half of the duplicate-effort check
could not be run with this token; the finding above rests on the fork list, the
upstream doc grep, the empty PR/issue history, and web search. Given upstream has two
forks total and one of them is ours, the residual risk is low, but the check is a hole
on this host and will be on every screen run here.

## Planning 2026-08-13 (linux-gfx942)

`plan.md` written; `ext_type` set to `torch-extension`; `surface.json` generated and
extended by hand with the 14 extension components and the new test component.

Working clone is the **fork**, not upstream: `gh repo clone AMD-Ecosystem/diff-surfel-rasterizations
projects/diff-surfel-rasterizations/src`. Cloning the fork rather than upstream is what lets
the plan read the existing `moat-port` work; `main` there is byte-identical to upstream
`1aa433c`, and `upstream/main` is configured as a second remote by `gh repo clone`.

Structural facts worth not rediscovering:

- 14 variant directories, 20 files each, but only **three distinct code bodies**. md5
  equivalence classes: `auxiliary.h` identical in all 14; `forward.cu` 2 classes;
  `backward.cu` 2; `rasterizer_impl.cu` 3; `rasterize_points.cu` 3. Per-variant files are
  only `config.h` (NUM_CHANNELS, BLOCK_X/Y) and `setup.py` (package name).
- Families: `base` (`-`, `-ch05`, `-ch11`, `-ch18`, `-ch26`, `-tile1`), `wet` (`-wet`,
  `-wet-ch05/07/11/18/26`), `wet-abs` (`-wet-abs`, `-wet-abs-ch05`). `moat-port` covers
  three of the `wet` family, so both other code bodies are untried on AMD.
- `NUM_WARPS (BLOCK_SIZE/32)` in `auxiliary.h` is the only hardcoded 32 and is **dead** --
  defined in all 14, referenced nowhere. No `__shfl*`, `__ballot`, `warpSize`, `cg::reduce`
  or `tiled_partition` anywhere. The port is wavefront-neutral by inspection.
- No `c10::`, `at::cuda`, `getCurrentCUDAStream`, `CUDAGuard` or `AT_CUDA*` in the tree, so
  the hipify v1-vs-v2 masquerading-API split cannot bite. No `TORCH_HIPIFY_V2` branch needed.
- `rasterize_points.cu` uses the deprecated `x.type().is_cuda()` and `.data<float>()`. Both
  still exist in this fleet's torch 2.14 (`ATen/core/TensorBody.h:230` and `:247`). Checked
  so nobody spends time on it.
- API contracts a harness must respect: `scales` is `(P,2)` (2DGS surfels), `means2D` is
  `(P,4)` for the two `wet-abs` variants and `(P,3)` elsewhere, and any variant with
  `NUM_CHANNELS != 3` must be driven through `colors_precomp` -- upstream throws otherwise,
  because `geomState.rgb` is sized `P*3` regardless of channel count.
