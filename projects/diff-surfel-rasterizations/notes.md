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

## Porting 2026-08-13 (linux-gfx942)

Executed `plan.md`: the remaining 11 variants got the seven-file treatment, and `tests/`
was added. Fork `moat-port` `4c95346` -> `5d567f6` (2 commits, 80 files).

    c579644 [ROCm] Extend the HIP build to all fourteen variants
    5d567f6 [ROCm] Add GPU tests covering every rasterizer variant

Environment: AMD Instinct MI300X (gfx942), ROCm 7.14.60850, torch 2.14.0a0+git7d05abc,
hipify 2.0.0, python 3.12, `PYTORCH_ROCM_ARCH=gfx942`, `MAX_JOBS=32`.

### How the 11 variants were edited, and how completeness was proved

The treatment was applied by a text transform (`agent_space/dsr_port.py`, scratch, not
committed) rather than by hand, and the transform was **validated by regenerating the three
already-ported variants from their upstream files and diffing against `moat-port`: all three
reproduce byte-for-byte**. Only then was it run on the other 11. That is the cheapest way to
guarantee "byte-identical treatment" across 14 copies.

The plan's completeness invariant holds. md5 equivalence classes, upstream vs post-port:

| file | upstream | post-port |
|---|---|---|
| `cuda_rasterizer/auxiliary.h` | 1 | 1 |
| `cuda_rasterizer/forward.cu` | 2 | 2 |
| `cuda_rasterizer/backward.cu` | 2 | 2 |
| `cuda_rasterizer/forward.h` / `backward.h` | 2 / 2 | 2 / 2 |
| `cuda_rasterizer/rasterizer_impl.cu` | 3 | 3 |
| `rasterizer.h` / `rasterize_points.cu` | 3 / 3 | 3 / 3 |
| `rasterizer_impl.h`, `ext.cpp`, `CMakeLists.txt` | 1 each | 1 each |
| `cuda_rasterizer/config.h` (per-variant, untouched) | 11 | 11 |

And the `setup.py` check: all 14 port diffs, with the package name normalised away, hash
identical (`sed -E 's/diff_surfel_rasterization[a-z0-9_]*/PKG/g' | md5sum` -> one class of 14).

### Build: 14/14 for gfx942, 448 s, zero warnings

    git submodule update --init --recursive
    export PYTORCH_ROCM_ARCH=gfx942 MAX_JOBS=32
    for v in diff-surfel-rasterization*/; do
      ( cd "$v" && rm -rf build *.egg-info *.hip hip_rasterizer \
        && pip install -e . --no-build-isolation --no-deps ) || break
    done

`grep -ci warning` over the whole build log: **0**. No spill diagnostic, no "local memory
limit exceeded", at any channel count.

### The one real blocker, and it was the environment: the GLM submodule was never checked out

First build attempt failed in *every* GLM call from device code -- `no matching function for
call to 'dot'`, `call to __host__ function from __device__ function`, `operator[]` const.
The cause is not the port: the planner's `gh repo clone` did not recurse submodules, so
`third_party/glm` was an empty directory. The `-I .../third_party/glm` still existed, so the
compiler silently fell back to **`/usr/include/glm` (GLM 0.9.9.8)**, which predates
`GLM_COMPILER_HIP` and therefore never marks its functions `__device__` under hipcc.

The tell is in the diagnostic paths (`/usr/include/glm/...`, not `third_party/`). After
`git submodule update --init --recursive` the same build passed with no source change.
Promoted to the skill (`fault-classes.md`) because any project bundling a header-only
library as a submodule has this failure mode, and it reads like a broken port.

### hipify 2.x renames the directory; the plan's clean step was written for hipify 1.x

The plan's `rm -rf build *.egg-info *.hip cuda_rasterizer/*.hip` is the stage-1 clean step
and is **incomplete on this fleet**. hipify 2.0.0 does not write `.hip` files beside the
sources; it renames the directory: `cuda_rasterizer/forward.cu` -> `hip_rasterizer/forward.hip`,
with the headers copied alongside. The build commands here add `hip_rasterizer` to the clean
step, and the new `.gitignore` covers `*.hip` and `hip_rasterizer/`. Also promoted to the
skill (`strategy-b-torch.md`), together with the GLM hipify-ignore monkeypatch, which is
confirmed still necessary and still effective on hipify 2.0.0 (`cpp_extension` looks
`hipify_python.hipify` up as a module attribute at call time, so patching after importing
`CUDAExtension` works).

### GPU tests: 95 passed, 5 skipped

    bash utils/timeit.sh diff-surfel-rasterizations test -- \
      python -m pytest projects/diff-surfel-rasterizations/src/tests/test_variants.py -q
    # 95 passed, 5 skipped in 7.38 s

The 5 skips are by design: 4 per-variant checks that `tile1` is too slow to run at 200x150
(it is covered at 32x24 by `test_tile1_matches_base`) plus the cross-variant comparison of
the base variant against itself.

Forward statistics for the three previously-untested code bodies (plus `ch26`, the pressure
case), 4000 surfels, 200x150, `colors_precomp`, opacity finite difference at eps 3e-3:

| variant | image | min / max / mean | nonzero | visible | fd slope |
|---|---|---|---|---|---|
| `diff-surfel-rasterization` (base) | 3x150x200 | 0.0000 / 0.8693 / 0.1123 | 0.411 | 4000/4000 | 0.9789 |
| `-tile1` (32x24, 1500 pts) | 3x24x32 | 0.0000 / 0.8840 / 0.1588 | 0.492 | 1500/1500 | 0.9874 |
| `-wet-abs` | 3x150x200 | 0.0000 / 0.8693 / 0.1123 | 0.411 | 4000/4000 | 0.9789 |
| `-ch26` | 26x150x200 | 0.0000 / 0.9267 / 0.1122 | 0.411 | 4000/4000 | 0.9949 |

Device dispatch confirmed with `AMD_LOG_LEVEL=3`: `void preprocessCUDA<3>(...)`,
`duplicateWithKeys`, `identifyTileRanges`, `void renderCUDA<3u>(...)` all appear as
`ShaderName` in `rocvirtual.cpp`, with `HIP_vector_type<float, 2u>` in the signatures.

### Plan risks, resolved

1. **ch18/ch26 register and LDS pressure -- did not materialise on gfx942.** Both build with
   zero warnings and no spill diagnostic, and `ch26` has the *best* finite-difference slope
   of the four measured. Still open for wave32, where the register file differs.
2. **`tile1` and `__launch_bounds__(1)` -- accepted by the AMD backend.** Compiles clean,
   renders, and its gradients pass. No diagnostic to record.
3. **`wet-abs` `(P,4)` `means2D` -- confirmed, and it is not a port bug.** `dL_dmeans2D` is
   `torch::zeros({P,4})` at `rasterize_points.cu:192` while every other variant uses `{P,3}`.
   Note that `__init__.py` is **byte-identical** to `-wet`'s, so nothing in the Python layer
   hints at it; a `(P,3)` input trips autograd's shape check and looks like a kernel bug.

### Where the plan's test design needed correcting, with evidence

Two assertions in the plan's test plan are too strong. Both were investigated before being
loosened, and neither is a port defect -- both would behave identically under CUDA.

- **Plan test 8, `tile1` vs base "must match within rtol=1e-4": it does not, and cannot.**
  Measured at 32x24: 16.5% of elements differ, max |diff| 0.0092, mean |diff| 2.4e-4,
  correlation 0.9999921. The test now asserts mean |diff| < 1e-3, max |diff| < 0.02,
  correlation > 0.9999.
  **The mechanism recorded here originally -- a coarse tiling compositing 3-sigma tails a
  1x1 tiling drops -- was wrong, and the review caught it. See the fix round below for what
  actually happens (integer truncation of the tile bounds).**
- **Plan test 6, opacity finite difference: the loss is piecewise smooth.** The two hard
  thresholds in the forward (`alpha < 1/255` and `test_T < 0.0001`) do make it so, and the
  eps sweep with the direction this round used (1e-4 -> 0.62, 3e-4 -> 1.34, 1e-3 -> 1.15,
  3e-3 -> 0.98, 1e-2 -> 0.98) is real; float64 loss accumulation changes nothing
  (0.70 / 1.35 / 1.15 / 0.98), so it is not float32 cancellation in the sum.
  **But the threshold noise was not what forced eps 3e-3, and recording it as the cause was
  wrong. The review found the operative cause -- a sign-balanced perturbation direction --
  and the fix round below has the measurement.**

### Test harness design

`tests/test_variants.py` (`tests/README.md` explains how to run it). No reference
implementation exists and no NVIDIA GPU is in this fleet, so the 14 variants are each
other's oracle: three code bodies at ten channel counts whose first three channels
composite identically, so one scene through all of them compared on channels 0-2 turns
coverage into correctness. Per-variant: symbol export, `markVisible` vs an exact CPU frustum
reference (`p_view.z > 0.2`, exact boolean equality), forward finiteness/coverage,
determinism (bit-identical repeat), gradient finiteness, opacity finite difference.
Cross-variant: channels 0-2 agreement, `tile1` vs base, `wet-abs` gradient columns.

A camera at the origin looking down +z with an identity `viewmatrix` is enough; note the
kernel's `transformPoint4x3` reads the matrix in the transposed 3DGS convention
(translation in the last *row* of the flattened tensor).

### Not done, deliberately

- The CMake path is still not built (scoped out by the plan, deferral already registered).
- No CUDA no-regression compile: no NVIDIA GPU or toolkit in this fleet. The argument stays
  structural (plan risk 9), and it is now stronger, since the 11 new variants' source diffs
  are byte-identical to the 3 that EnvGS exercised.
- Plan open question 3 (README install matrix): answered no. The README's ROCm section now
  says "All fourteen" and adds the submodule prerequisite plus a pointer to `tests/`;
  enumerating 14 identical `pip install -e .` invocations would add nothing.

## Review 2026-08-13 (linux-gfx942), `4c95346...5d567f6`

Verdict: **changes-requested**. Problems only, per the review skill. The port itself is
sound -- the invariant was recomputed independently and holds, the 14 variants carry
identical treatment, the CUDA path is preserved, and no ROCm fault class is triggered
(details of what was re-verified are at the end). What follows is what has to change:
two of the recorded justifications are wrong about their own mechanism, one promoted skill
lesson is wrong, and the suite leaves the port's most unusual kernel corner uncovered.

### 1. The `tile1` difference is not the tail effect the code and the record claim

`tests/test_variants.py:308-317`, `notes.md` "Plan test 8" bullet, and the body of commit
`5d567f6` all say the difference is that a coarse tiling composites *tails beyond the
3-sigma radius* that the 1x1 tiling drops, "the radius is a 3-sigma cutoff where
`exp(-4.5)*opacity ~= 0.01`, well above the `alpha < 1/255` reject". That is not what
happens in the scene the test actually renders.

Measured on gfx942 with the committed 32x24 / 1500-point scene: **every** surfel comes back
with `radii == 3` (min = max = 3). Radius is `ceil(max(extent.x, extent.y, cutoff *
FilterSize))` at `forward.cu:230-236` with `FilterSize = 0.707106` (`auxiliary.h:45`), so 3
is the *filter floor*, not a 3-sigma extent: at the edge of that box `alpha` is
`opacity * exp(-9) ~= 1e-4`, an order of magnitude *below* the 1/255 reject, not above it.
The tail argument cannot be the cause here. Confirmed directly: a single surfel with
`radii == 3` renders **bit-identical** images through `tile1` and the base variant
(0 differing pixels at scales 0.005/0.02/0.05); a difference only appears once the surfel is
large enough that `radius > 3` (scale 0.2 -> radius 5, 6 differing pixels).

The real mechanism is integer truncation in the binning, at `auxiliary.h:73-83` combined
with the half-open emit loop at `rasterizer_impl.cu:102-105`. With `BLOCK_X = 1`,
`rect_max.x = (int)((p.x + r + BLOCK_X - 1) / BLOCK_X) = floor(p.x + r)` and the loop is
`x < rect_max.x`, so the pixel column at `floor(p.x + r)` is dropped -- a pixel that is only
`r - frac(p.x)` away from the centre and still above the alpha cutoff when `frac(p.x)` is
large. With 16x16 tiles the same expression rounds *up* to a whole tile, so that column is
kept. Reproduced exactly: one surfel, opacity 0.99, radius 3, sweeping the sub-pixel centre
-- 0 differing pixels for `frac(p.x)` below ~0.75, and 6 differing pixels (`tile1 < base`,
max |diff| 1.0e-2, matching the whole-scene 9.2e-3) for `frac(p.x)` in [0.75, 1.0).

The conclusion is unaffected -- `getRect` is untouched upstream code and a CUDA build does
the same thing, so this is not a port defect and the loosened assertion is the right call.
But the explanation shipped in the repository and in this record is wrong, and it is wrong
in a way that misleads: it implies the tolerance tracks surfel size, when it actually tracks
sub-pixel placement. Fix the docstring at `tests/test_variants.py:308-317` and the notes
bullet; amending the `5d567f6` body is preferable while the branch is unpublished.

### 2. The finite-difference step is not the reason the small-eps slopes were bad

`tests/test_variants.py:254-263` and the notes' "Plan test 6" bullet attribute the eps
sweep (1e-4 -> 0.62, 3e-4 -> 1.34, ... 3e-3 -> 0.98) to the two hard thresholds making the
loss piecewise smooth, and conclude the step must be wide enough to average over crossings.
The threshold noise is real, but it is not what forces eps 3e-3: the committed direction is.

`tests/test_variants.py:273` draws `rand*2-1`, a sign-balanced direction over 4000
opacities, so the directional derivative it probes nearly cancels. Measured on the same
scene and variant: predicted `86.1` for the committed direction versus `~3900` for a
non-negative direction -- a 45x difference in signal against the same crossing noise. With a
non-negative direction (`torch.rand`, no `*2-1`) the slope is:

    eps 1e-5: 0.9969 0.9997 1.0001 1.0002   (4 seeds)
    eps 1e-4: 1.0038 1.0038 1.0000 1.0111
    eps 1e-3: 1.0095 1.0102 1.0089 1.0091
    eps 3e-3: 1.0085 1.0083 1.0076 1.0079

i.e. the analytic opacity gradient is right to ~1% at any step from 1e-5 up, on the very
scene the record says cannot be measured below 1e-3. A sparse 12-surfel scene with no
transmittance saturation gives 1.0007 / 1.0002 / 1.0000 / 1.0000 / 1.0000 over eps 1e-5 to
1e-2, which pins the gradient itself as correct.

Consequence: as committed the check verifies the gradient to only ~10% along a low-signal
direction, when the same two renders buy a ~1% check. Change `tests/test_variants.py:273` to
a non-negative direction, drop `eps` at :274 to 1e-3, and tighten the band at :287; then
correct the docstring and the notes bullet, which currently record a cause that is not the
operative one.

For the record, one claim I could *not* substantiate and am explicitly not making: this is
not a cross-platform flakiness risk. Jittering the scene by a relative 1e-5 (far more than
an FMA-contraction difference between architectures) moves the committed slope only over
0.977-0.994, well inside the 0.9-1.1 band.

### 3. `tile1`'s backward kernels are never launched by the suite

`test_backward_gradients_are_finite` (`tests/test_variants.py:237-238`),
`test_opacity_finite_difference` (`:264-265`), `test_forward_is_deterministic` (`:227-228`)
and `test_forward_is_finite_and_non_trivial` (`:212-213`) all skip on `BLOCK_X == 1`, and
`test_tile1_matches_base` (`:307`) runs forward only. So `BACKWARD::renderCUDA` under
`__launch_bounds__(1)` -- plan risk 2, the single most unusual thing the AMD backend is
asked to do in this port -- has **zero** coverage in the committed gate, on gfx942 and on
every platform still to validate. The record overstates this: "its gradients pass" and the
0.9874 slope in the evidence table are out-of-band measurements a validator cannot
reproduce from the suite.

I ran it here (32x24, 1500 points): gradients are finite, and the opacity slope is 0.9996 at
eps 1e-3 with a non-cancelling direction, so this is a coverage gap and not a defect. Add a
`tile1` backward test at 32x24. One trap to avoid when you do: `means2D.grad` is legitimately
**0.0** in that scene, for the base variant too, because `dL_dmean2D` is written only in the
`rho2d < rho3d` branch at `backward.cu:429-435` and preprocess derives the returned value
from `dL_dtransMats[idx*9+2]`/`[+5]` at `backward.cu:630-631`; assert on the other five
tensors, or use larger surfels.

### 4. The spherical-harmonics path is never executed

`render()` at `tests/test_variants.py:155-162` always passes `colors_precomp`, so
`computeColorFromSH` (`forward.cu:22-72`), its backward (`backward.cu:22-141`), and the
`clamped` buffer never run in the gate. That is a substantial slice of device code, and it
is the one entry point a downstream user reaches without precomputing colours. Four variants
can take it (the 3-channel `base`, `tile1`, `wet`, `wet-abs`); one test on the base variant
covers it. Verified working here (finite image, `dL_dshs` sum 5.7e3), so again coverage, not
a defect.

### 5. Only `color` is asserted; the other rendered outputs are dropped

`render()` returns `out[0], out[1]` (`tests/test_variants.py:164-165`) and every test throws
away `out[2]`, the 7-channel `RENDER_AXUTILITY` buffer (depth, normals, distortion, median
depth), and `out[3]`, the wet family's `out_weight` -- which is the entire reason the wet
variants exist and is accumulated with `atomicAdd` at `forward.cu:422` of that family.
A NaN or garbage in either passes the suite today. Add finiteness and non-triviality
assertions on every returned tensor.

### 6. `ext_winhip.cu` is not ignored

`setup.py:72-74` copies `ext.cpp` to `ext_winhip.cu` inside each variant directory on
Windows+ROCm, and the new `.gitignore` does not list it. A Windows validator gets 14
untracked sources in the tree and can commit them by accident. Add `ext_winhip.cu` to
`.gitignore`.

### 7. Promoted skill lesson: HIP does have cooperative-groups `reduce`

`.claude/skills/cuda-to-rocm/references/fault-classes.md` (commit `051e500`) says
"`#include <cooperative_groups/reduce.h>` (HIP's cooperative groups has no `reduce.h` ...)".
ROCm 7.14 ships `hip/cooperative_groups/hip_reduce.h` and
`hip/amd_detail/amd_hip_cooperative_groups_reduce.h`, which define
`cooperative_groups::reduce`. What is actually true, and checked here, is that the *CUDA
spelling* has no hipify include mapping (`CUDA_INCLUDE_MAP` maps only `cooperative_groups.h`)
and so does not resolve. As written the lesson tells the next porter that `cg::reduce` is
unavailable on ROCm, which is false and would push them into rewriting a reduction that
ports as-is. Reword to: the CUDA spelling has no mapping and must be guarded; ROCm's
equivalent is `<hip/cooperative_groups/hip_reduce.h>`.

The other four lessons check out against their sources: no `__trap` anywhere in the ROCm
include tree; hipify's default `extensions` tuple is `(".cu",".cuh",".c",".cc",".cpp",".h",
".in",".hpp")` with no `.inl`, so the dropped-`.inl` claim is right; the system GLM at
`/usr/include/glm` is 0.9.9.8 with no `GLM_COMPILER_HIP`; hipify 2.0.0 does produce
`hip_rasterizer/*.hip`; and nvcc 12.8 accepts all four respaced chevron spellings that
appear in the diff (compiled here).

### 8. Smaller items

- `.claude/skills/cuda-to-rocm/references/strategy-b-torch.md` (commit `051e500`): "add it to
  `.gitignore` thinking rather than committing it" is a garbled sentence in a document about
  to reach `main`.
- `setup.py:37` re-imports `torch` inside `_patch_hipify_ignore_glm` when it is already
  imported at `setup.py:19`; replicated 14 times.
- `tests/test_variants.py:298-300` renders the reference before deciding to skip the
  reference-against-itself case; wasted render.
- Not a defect, flagged so it does not propagate silently: the 14 `setup.py` headers and
  `tests/test_variants.py:6` add an AMD copyright line, and the `setup.py` copies add an
  author line, against the standing "default to no new copyright or author lines". The base
  commit `4c95346` established this before this project had a record, and undoing it in the
  11 would break the equivalence invariant, so it is a person's call, not the porter's.

### What was re-verified independently (no action needed)

- Completeness invariant: recomputed md5 classes over all 14 variants for 13 files at `main`
  and at `HEAD`. Not just the class *counts* -- the class *partitions* are identical, and
  `rasterizer_impl.h`, `rasterizer.h`, `config.h`, `rasterize_points.cu/.h`, `ext.cpp`,
  `CMakeLists.txt` are byte-identical to upstream. The 14 `setup.py` port diffs collapse to a
  single class once the package name is normalised.
- Treatment uniformity: the per-file diff against upstream is one class per code body across
  all 14 -- so the 11 new variants received exactly what the 3 EnvGS-proven ones carry.
- Fault classes: no `__shfl*`/`__ballot`/`warpSize`/`tiled_partition`/`cg::reduce` anywhere;
  `NUM_WARPS (BLOCK_SIZE/32)` is defined in all 14 and referenced nowhere; every other
  literal 32 is a uint64 key shift or a radix-sort bit range. No textures, no resource
  handles, no OOB neighbour reads, no library swap beyond CUB -> hipCUB (confirmed in the
  generated `hip_rasterizer/rasterizer_impl.hip`).
- CUDA path: every source edit is inside `#if !defined(USE_ROCM)` / `#if defined(USE_ROCM)`
  except the chevron respacing, which nvcc 12.8 accepts (compiled); both `setup.py` helpers
  return early unless `torch.version.hip`.
- `python3 utils/jargon.py --port diff-surfel-rasterizations`: clean. Commit titles are
  `[ROCm]`-prefixed and 51/53 chars, bodies disclose AI assistance and carry Test Plans, no
  `Co-Authored-By` trailer, ASCII throughout, no AMD-internal account reference.
- Suite reproduced on this host: 95 passed, 5 skipped in 7.11 s.
- `git -C projects/diff-surfel-rasterizations/src status --porcelain` is empty.

## Fix round 2026-08-13 (linux-gfx942), `5d567f6` -> `7d6efdc`

Every numbered review item applied. Same environment as the porting round (MI300X gfx942,
ROCm 7.14.60850, torch 2.14.0a0+git7d05abc, hipify 2.0.0, `PYTORCH_ROCM_ARCH=gfx942`,
`MAX_JOBS=32`). All 14 variants rebuilt from clean: **449.6 s, exit 0, `grep -ci warning` = 0**.
Suite after the rebuild: **136 passed, 1 skipped in 7.55 s** (was 95 passed, 5 skipped).

### History was rewritten, deliberately

Nothing had validated `5d567f6` (`validated_sha` null on both rows) and there is no upstream
PR, so the two commits were amended rather than appended to: the wrong `tile1` mechanism and
the wrong finite-difference rationale were *in the commit bodies*, and on a fork whose branch
is itself the deliverable those bodies are what a reader gets. New shas `d4bb1cb` (build) and
`7d6efdc` (tests); the pre-rewrite tip is kept locally as `backup-review-5d567f6` and was not
pushed. `--force-with-lease` only.

### 1. The `tile1` mechanism, corrected and now under test

The review is right and the reproduction is confirmed here. Committed 32x24 scene: every
surfel `radii == 3`, the filter floor. The divergence is integer truncation, not 3-sigma
tails.

A single surfel (opacity 0.99, scale 0.01, radius 3) placed at pixel `(16+f, 12+f)` by
inverting `ndc2Pix`, swept over `f`:

| f | differing pixels | max abs diff |
|---|---|---|
| 0.00 / 0.10 / 0.25 / 0.50 / 0.60 | 0 | 0 |
| 0.70 | 2 | 4.6e-3 |
| 0.80 | 4 | 7.5e-3 |
| 0.90 | 4 | 1.2e-2 |
| 0.95 / 0.99 | 6 | 1.5e-2 / 1.7e-2 |

Every differing pixel lies in the dropped column `floor(p.x + r)` or the dropped row
`floor(p.y + r)` (checked, 0 counterexamples), and `tile1 <= base` at all of them: the 1x1
tiling can only *lose* contributions. `BLOCK_Y` is 1 as well, so the row truncates like the
column -- the review's column-only account is right but half the story.

`tests/test_variants.py` now carries `test_tile1_drops_the_truncated_boundary_pixel`, a
five-point sweep asserting bit-identity below the threshold and, above it, that differences
are one-sided, above 1e-3, and confined to the truncated column and row. The bulk test stays
(it is a different question) with the docstring corrected, as are `tests/README.md` and the
commit body.

### 2. The finite difference: non-negative direction, eps 3e-3 -> 1e-3, band +/-10% -> +/-3%

Reproduced: the committed `rand*2-1` direction predicts `86.1` where a non-negative one
predicts `3983.7` on the same scene. Slope at eps 1e-3 with `torch.rand`, all 14 variants:

    base 1.0091   ch05 1.0095   ch11 1.0088   ch18 1.0093   ch26 1.0089   tile1 1.0060
    wet  1.0091   wet-ch05 1.0095   wet-ch07 1.0080   wet-ch11 1.0088   wet-ch18 1.0093
    wet-ch26 1.0089   wet-abs 1.0091   wet-abs-ch05 1.0095

Range 1.0060-1.0095, so the 0.97-1.03 band has ~2% headroom on both sides. Seed sensitivity
at eps 1e-3 (seeds 11-14): 1.0091 / 1.0081 / 1.0095 / 1.0080. The residual +0.9% is the
second-order term plus the threshold noise the old bullet described -- real, but it was never
what forced the wide step.

### 3. `tile1`'s backward now runs, and so do its other per-variant checks

The four `BLOCK_X == 1` skips are gone. `tile1` runs the whole per-variant set on
`SMALL_SCENE` (32x24, 1500 points) instead of being skipped, so `BACKWARD::renderCUDA` under
`__launch_bounds__(1)` is exercised on every platform that runs the suite. Confirmed on the
device with `AMD_LOG_LEVEL=3`: two distinct `void renderCUDA<3u>(...)` and two distinct
`void preprocessCUDA<3>(...)` signatures dispatch (the forward pair and the backward pair),
with `HIP_vector_type<float, 2u>` in the argument lists.

**The means2D-gradient trap the review flagged is real and has a cause worth recording.**
`dL_dmean2D` is zero in the small scene *because every surfel sits on the filter-size floor*:
it is written only in the `rho2d < rho3d` branch, and while the projected radius is the
`cutoff * FilterSize` floor the 3D term is never the smaller one. Scaling the surfels up
fixes it rather than needing an exemption -- measured on the base variant at 32x24, sum of
`|means2D.grad|`: scale x1 -> 0.0, x3 -> 1240.7, x6 -> 1962.8, x10 -> 1213.4. So `SMALL_SCENE`
carries `scale=3.0` and the uniform "no gradient is identically zero" assertion holds for
`tile1` exactly as for the other thirteen. No per-variant exemption was needed.

### 4-5. SH path, `out_others`, `out_weight`

- `test_spherical_harmonics_colours` covers `computeColorFromSH` and its backward on the
  three 3-channel non-`tile1` variants (base, wet, wet-abs -- all three code bodies). Band 0
  is seeded to reproduce the scene's precomputed colours, so asserting the image *differs*
  from the precomputed render proves the higher bands were evaluated (max abs diff 0.221,
  `|dL_dshs|` sum 6.0e4).
- `render()` now returns every tensor. `test_auxiliary_outputs_are_plausible` asserts on the
  7-channel buffer: shape, finiteness, `alpha` in [0,1] with a max above 0.5, `depth >= 0`
  and `depth/alpha` inside the scene's depth range where `alpha > 0.9` (measured
  2.009-4.278 against a scene z of [2,6] -- depth is accumulated against alpha, not
  normalised, which is the thing to know before writing an assertion on it), median depth
  in range, normal length <= 1.01 and non-zero, distortion non-negative.
- **The auxiliary buffer is a stronger oracle than the colour image and is now used as one.**
  `out_others` depends only on geometry, so it is *bit-identical* across all 13 non-`tile1`
  variants (max abs diff exactly 0.0, measured) -- it is compared against the base variant in
  the cross-variant test at no extra render. `out_weight` likewise depends only on opacity and
  geometry, so it agrees across the whole `wet` family to 1.9e-6 (not bit-identical: it is an
  `atomicAdd` over pixels, so the summation order follows the grid shape). New
  `test_out_weight_matches_wet_reference` plus a per-variant plausibility test that also
  asserts a culled surfel (`radii == 0`) accumulates exactly zero weight.

### 6-9. Smaller items

- `.gitignore`: `ext_winhip.cu` added (the Windows+ROCm copy of `ext.cpp`).
- All 14 `setup.py`: duplicate `import torch` inside `_patch_hipify_ignore_glm` removed. The
  normalised-package-name md5 stays one class of 14.
- `test_first_three_channels_match_base` decides the self-comparison skip before rendering.
- Skill lessons corrected on this branch (both from `051e500`): `fault-classes.md` no longer
  claims HIP lacks cooperative-groups `reduce`, and `strategy-b-torch.md`'s garbled sentence
  is fixed. Details below.
- **Untouched on purpose:** the AMD copyright/author lines in the 14 `setup.py` files and in
  `tests/test_variants.py`. That is a person's call per the review, not the porter's.

### The cooperative-groups lesson was wrong; what is actually true

Checked against the installed ROCm 7.14 tree before rewriting:

- `hip/cooperative_groups/hip_reduce.h` exists and includes
  `hip/amd_detail/amd_hip_cooperative_groups_reduce.h`, which defines
  `cooperative_groups::reduce`. So `cg::reduce` is available on ROCm.
- `hip/hip_cooperative_groups.h` does **not** pull that header in, so the include is needed
  explicitly -- worth stating, since "cooperative groups works" would otherwise imply it.
- hipify's `CUDA_INCLUDE_MAP` maps only `cooperative_groups.h`
  (`torch/utils/hipify/cuda_to_hip_mappings.py:305`), so `<cooperative_groups/reduce.h>` is
  left verbatim and then fails to resolve. That is the real fault, and it is an *include*
  fault, not a missing feature.

The fault-classes entry now says exactly that and names the ROCm header, so the next porter
guards the include instead of rewriting a reduction that ports as-is.

### Invariants re-checked after the edits

- md5 equivalence classes over all 14 variants for 13 source files plus normalised `setup.py`:
  the class *partitions* (not just the counts) are identical at `main`, at the old `5d567f6`,
  and in the new tree. The test files live in `tests/` and are outside the per-variant
  equivalence classes, as expected; the `setup.py` edit was applied by script to all 14.
- `git -C projects/diff-surfel-rasterizations/src status --porcelain` empty after committing.
- `python3 utils/jargon.py --port diff-surfel-rasterizations`: clean. Both titles are
  `[ROCm]`-prefixed, 52 and 54 chars; `utils/prose.py` clean on both bodies.

### Still open for a validator on another platform

The tolerances tightened this round were all measured on gfx942 only. The finite-difference
band is now +/-3% (was +/-10%) and the `tile1` sweep asserts bit-identity below the threshold.
Both should hold anywhere -- the sweep compares two builds of the same source on the same
device, and the slope is a property of the scene -- but they are the two places where a
wave32 or Windows run would first show a difference, so treat a failure there as evidence
about the platform rather than as flakiness to widen away.
