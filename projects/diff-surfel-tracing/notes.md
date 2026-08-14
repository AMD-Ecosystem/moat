# diff-surfel-tracing notes

A dependency fork: the tracer EnvGS's reflection path renders through.

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

## Intake screen (2026-08-13, linux-gfx1100)

Recommendation: **fork** -- i.e. take it up. The fork already exists, so the state
went to `screened` rather than `awaiting-fork`. Recorded with `set-intake`.

This is an unusual intake: it is not a fresh candidate. The port is already written
and GPU-validated, but it was done and recorded entirely under EnvGS's project
record. What is missing is this project's own record and its own route upstream --
`xbillowy/diff-surfel-tracing` is a different repository from `zju3dv/EnvGS`, so
the work can only reach its maintainer as a separate PR. That is what taking this
up buys; it is not a request to port anything from scratch.

### Licence: MIT (tier 1), verified by reading the file

`licenses.py check` returns UNPARSED -- GitHub reports NOASSERTION / "Other" for
this repo. That is a misparse, not a restriction. `LICENSE` at upstream HEAD is the
verbatim MIT text, copyright 2024 3D Vision Group, State Key Lab of CAD&CG,
Zhejiang University. `license_spdx: "MIT"` in status.json was already correct and
is confirmed independently, not inherited on trust.

Worth knowing: upstream had NO licence at all until commit `ef6f24be`
("add: license", 2025-10-14). The `moat-port` branch is cut from a base BEFORE that
commit, so the branch's own tree contains no LICENSE file. Contribution today is
judged against upstream today, which is cleanly MIT.

Scope reminder: MIT clears CONTRIBUTING upstream. It says nothing about the
vendored third-party content below.

### NVIDIA proprietary headers in the OptiX submodule -- needs a person's decision

`.gitmodules` points `third_party/optix` at `NVIDIA/optix-dev`. Of its 14 headers,
**9 carry NVIDIA proprietary licence text** and 5 are BSD-3-Clause;
`third_party/optix/LICENSE.txt` is the NVIDIA DesignWorks SDK EULA. Per the intake
role, NVIDIA-proprietary files need a decision before proceeding, so this is
flagged rather than resolved here.

Mitigating fact for whoever rules on it: the port REPLACES the OptiX path with
HIPRT and does not modify, redistribute, or depend on those headers. They are
upstream's pre-existing submodule, not something the port adds.

**Tooling gap found while checking this.** `licenses.py scan-nvidia` reported the
tree clean and it is not. The markers in `config/licenses.toml` are matched with
`grep -rlF` (case-SENSITIVE, utils/licenses.py:78). The marker reads
`NVIDIA CORPORATION and its licensors retain all intellectual property`; the OptiX
headers read `NVIDIA Corporation ...`. Adding `-i` turns 0 hits into 9. This is a
control-plane bug affecting every screen run to date, so it was NOT fixed from this
port branch -- it needs its own change on the trunk and a re-run of past screens.

### Duplicate effort: none outside MOAT

- No `diff-surfel*` repo in AMD-Ecosystem or ROCm other than our own fork.
- Upstream README has zero matches for amd/rocm/hip/gfx/radeon -- no notable-forks
  link, no existing platform port.
- Upstream forks are `wasahaiah`, `rhombus19`, `piotrmwojcik`, and ours; none are
  AMD ports.
- No entry in `data/candidates.json`, no disposition, no opt-out.
- Upstream has one closed PR (#5, unrelated: setup.py compilation) and no open PRs.

The only existing AMD work is MOAT's own, via EnvGS Stage 2.

### Viability: yes

Genuinely GPU code: `optix_tracer/{forward.cu,backward.cu}` (~1500 lines) plus a
torch `CUDAExtension` (`setup.py`), so `ext_type` is a torch-extension rather than
the recorded `unknown` (intake has no command to set that field).

It is an OptiX ray-tracing pipeline, not ordinary CUDA: `optixAccelBuild` over
`OPTIX_BUILD_INPUT_TYPE_TRIANGLES`, `optixTrace`, module/pipeline/SBT plumbing, and
`__raygen__`/`__anyhit__` programs. OptiX has no ROCm equivalent, so this was a
rewrite onto HIPRT rather than a hipify -- and that rewrite is already done and
validated (EnvGS notes, "Stage 2 port: OptiX -> HIPRT", gfx90a PASS, including a
genuine uninitialized-`cutoff` bug and two return-type UB bugs latent in the
upstream OptiX sources).

Upstream is alive but quiet: not archived, not disabled, 58 stars, 5 forks, last
push 2025-10-14 (~10 months). A PR has a real destination.

`depends_on` stays empty. It is standalone -- the README references
diff-surfel-rasterization only for API similarity, and nothing imports it. The
relationship runs the other way: EnvGS consumes this. The sibling candidate
`diff-surfel-rasterizations` is being screened concurrently by another agent; its
records were not touched here.

### What a planner inherits (not intake's call, but it decides the effort)

1. **Vendored HIPRT/Orochi is the real obstacle to a PR.** `moat-port` adds 110
   files under `third_party/hiprt/`, including a **1.9 MB prebuilt
   `hiprtc0604.dll`** and two more win64 DLLs committed as real binaries. The only
   licence file anywhere under that tree is cuew's; HIPRT's and Orochi's own
   licence files were not vendored with the code. Asking an upstream maintainer to
   take a vendored SDK plus binary blobs is a hard sell, and the missing licence
   files must be resolved regardless.
2. **The branch is 3 upstream commits stale**, including `e0016a27`
   ("update: latest version"), `3b97d5d3`, and the `ef6f24be` licence commit. It
   needs rebasing onto current upstream before any PR.
3. No plan.md, no surface.json, and no validation recorded against this project's
   own `head_sha` (still null). The existing note above explains the empty platform
   row: there is no standalone test suite, so evidence arrives through EnvGS.

## Port round 1 (2026-08-13, linux-gfx942, MI300X, ROCm 7.14) -- PARTIAL

State left at `porting` with the fork-write lock held. The rebase, the additive
CUDA restore, the de-vendoring and the header collapse are all done, committed
and pushed; the GPU gate is NOT met. See "Open defect" below for the one thing
that remains, which is a real finding and not a loose end.

Fork `moat-port` was rewritten from upstream `ef6f24b` rather than rebased over
`5991683`/`415f0a4`: the old branch carried 98 vendored SDK files, so a replay
onto a clean upstream base is both smaller and easier to review. New history:

- `1afb40b` `[ROCm] Fix undefined behavior in quat_to_rotmat_transpose`
- `195d62f` `[ROCm] Add a HIP RT trace back end for AMD GPUs`  (`head_sha`)

### Plan items completed

1. **Rebase onto `ef6f24b`, replaying `e0016a2` + `3b97d5d`.** Done hunk by hunk
   into `hiprt_tracer/kernels.h`: the 2D-projection low-pass path
   (`compute_transmat_xy*`, `rho2d`, `cmd`, `P`, `splat2pixel`, `xy`,
   `FilterInvSquare`) and the distortion accumulation are gone; `DUAL_VISIABLE`
   direction is `ray_d`; `dpt = tmx + payload.dpt` with `payload.dpt = dpt +
   STEP_EPSILON`; the `min_depth` three-way ternary; `ray_ot = E + START_OFFSET *
   ray_dt`; `computeColorFromSH*` take a pre-normalized direction and
   `computeColorFromSHBackward` now assigns `dL_ddir` instead of accumulating
   `dnormvdv`; `numerator = 1 - alpha`; the `power_clamped = 1.0f` override is
   gone; `compute_transmat_uv_backward` is called unconditionally.
   `kernels.h` is 1,304 -> 1,084 lines. Kept: global chunk scratch, the
   `surfelFilter` functor, the traversal loop.
2. **CUDA/OptiX path restored additively.** `git diff ef6f24b -- optix_tracer/
   CMake/ CMakeLists.txt` is now only the unconditional `quat_to_rotmat_transpose`
   fix plus `USE_ROCM`-guarded blocks in `auxiliary.h` and `params.h`; the CUDA
   preprocessed output of both is unchanged. `setup.py` takes upstream's OptiX
   branch verbatim when `torch.version.hip` is falsy. `trace_surfels.cpp` keeps
   upstream's OptiX host glue with the HIP RT bodies under `#ifdef USE_ROCM`.
   Structural verification only -- no NVIDIA GPU and no OptiX SDK on this host.
3. **De-vendored HIP RT.** 98 files and the committed 784 KB DLL are gone,
   replaced by a `third_party/hiprt` submodule pinned at `3.1.0.cb09c56`
   (`8602b8c475255fb922c2792654aae0a6bcdeb0af`), mirroring `third_party/optix`,
   plus an `HIPRT_HOME` fallback mirroring `OPTIX_HOME`. Our patches are extracted
   to `third_party/hiprt-rocm-fixes.patch` with a header explaining each; the
   jargon leak at `Compiler.cpp:655` is gone and `jargon.py --port` is clean.
4. **Duplicate headers collapsed.** `hiprt_tracer/{config,auxiliary,params}.h`
   deleted; their guarded blocks folded into `optix_tracer/`.
5. **Validation harness authored** at
   `projects/diff-surfel-tracing/validation/validate_tracer_rocm.py` (MOAT, not
   the fork). Covers import/API, the README's own `get_triangles` tessellation,
   forward finiteness and hit fraction, all eleven backward gradients including
   the nonzero `grad_grads3D` trap, finite differences, a reflected bounce and a
   cold compilation cache. It runs to the forward stage today; see below.

### Corrections to plan.md, found while executing it

- **The backward returns ELEVEN tensors, not twelve, and there is no
  `dL_dgrads3D_abs`.** Upstream `e0016a2` added exactly one field,
  `Params::dL_dgrads3D`, accumulated as `dL_dmean3D * 0.5f * dpt`
  (`backward.cu:624-626`), and `__init__.py` unpacks eleven names. The plan's
  "12 tensors" and the `_abs` gradient do not exist at `ef6f24b`. The harness
  checks the one that does.
- **There are ten patched HIP RT files, not four.** Beyond the four the plan
  lists, the vendored tree also carried: `Context.cpp` (call the new
  `Compiler::init()` after making the context current), `Compiler.{h,cpp}` (move
  setup out of the constructor), `Orochi.cpp` (`oroSetRawDevice` built the device
  handle from the raw index instead of initializing then setting it), and member
  initializer order / uninitialized loop variable fixes in `MemoryArena.h`,
  `BvhNode.h` and `hiprt_device_impl.h`. All ten are in the patch file.
- **Upstream clobbers `dL_dray_d` in the SH backward.** `e0016a2` changed
  `computeColorFromSHBackward` to assign rather than accumulate its direction
  gradient, so the geometric `dL_dray_d` accumulated earlier in the same hit is
  overwritten (`backward.cu:631`). Mirrored faithfully; worth raising with the
  maintainer, but not ours to change.
- **`W` is never accumulated after the delta.** `e0016a2` deleted the block that
  did `W += w` but kept the `dL_dacc` term that reads `W`. Again mirrored.

### Build recipe (reproducible, this host)

```
git clone --recursive --depth 1 -b 3.1.0.cb09c56 https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT.git /var/lib/jenkins/HIPRT
git -C /var/lib/jenkins/HIPRT apply <fork>/third_party/hiprt-rocm-fixes.patch
HIP_PATH=/opt/rocm cmake -DCMAKE_BUILD_TYPE=Release -DBITCODE=OFF -DNO_UNITTEST=ON \
    -DHIP_PATH=/opt/rocm -S /var/lib/jenkins/HIPRT -B /var/lib/jenkins/HIPRT/build
cmake --build /var/lib/jenkins/HIPRT/build --target hiprt03001 -j16   # ~60 s
HIPRT_HOME=/var/lib/jenkins/HIPRT PYTORCH_ROCM_ARCH=gfx942 \
    pip install -e <fork> --no-build-isolation --no-deps -v            # ~3 min
```

Both succeed. The patch applies cleanly to the pinned tag.

### Build-time gotchas hit and fixed this round

- **The runtime compiler does not define `USE_ROCM`.** Once `params.h` and
  `auxiliary.h` became shared with the CUDA build, the runtime compile of
  `kernels.h` took the CUDA branch and failed on `#include <optix.h>`. Fixed by
  passing `-DUSE_ROCM=1` in the runtime-compiler option list
  (`hiprt_wrapper.cpp`). Anything guarded by a build-system macro has to be told
  to the runtime compiler explicitly; it inherits nothing from the host build.
- **`typedef unsigned long int uint64_t;` clashes under the runtime compiler**
  with `hiprt_common.h`'s `using uint64_t = __hip_internal::uint64_t` (which is
  `unsigned long long`). Guarded out on the ROCm side.
- **torch's hipify writes `*_hip.cpp` / `*_hip.h` beside the sources.** Added to
  `.gitignore`; they are build artifacts, not sources.
- `int_opts` already exists in both trace functions upstream; the chunk scratch
  options had to be named something else.

### Open defect: the traversal hangs on some scenes (BLOCKS the GPU gate)

`forward_kernel` never returns for some surfel scenes. Reproduced with the
harness scene generator: 8, 16, 32, 33, 36 and 40 surfels all trace in about
2 ms at 16x16, 32x32 and 96x96 and produce a plausible image (accumulation
about 0.30, genuine hits and misses); 48 and 64 surfels hang indefinitely at any
resolution, with the GPU pegged at 100%.

Two probes, both on the staged copy of `kernels.h` with the cache cleared:

1. Widening the retrace step from `STEP_EPSILON` (1e-5) to 1e-3 does NOT fix it,
   so it is not the chunk loop crawling forward in tiny increments.
2. Hard-capping the `while (1)` retrace loop at 64 iterations in BOTH kernels
   does NOT fix it either.

Probe 2 is the decisive one: with the outer loop bounded, the hang has to be
inside a single `traceStep`, i.e. inside HIP RT's own
`hiprtGeomTraversalAnyHit::getNextHit()` driving `surfelFilter`. It is a
traversal-side non-termination, not a shading-loop bug, and it is scene
dependent rather than size dependent (40 surfels fine, 48 not). Next steps for
whoever picks this up, in order of cost:

- try `hiprtGeomTraversalAnyHitCustomStack` with an explicit stack, since the
  default global stack is the obvious candidate for a traversal that rejects
  every candidate hit;
- check whether the filter's `p.cnt` overflow (it counts every qualifying hit,
  not just the stored ones) interacts with HIP RT's own restart logic;
- reduce to the minimal failing geometry (bisect the 48-surfel scene down) and
  report it to HIP RT if it reproduces without our filter.

This was NOT observed under EnvGS Stage 2, but that evidence is at the old
tracer commit against a different consumer's geometry, so it does not clear the
current tip either way.

### Verified this round

- Extension builds clean for gfx942; runtime compilation of `kernels.h` succeeds
  and the cache is written (MI300X device names contain no `/`, so the
  cache-filename patch is indeed unnecessary here -- verified, not assumed).
- Forward trace produces a finite, plausible image on real geometry
  (accumulation 0.285 at 8 surfels / 16x16 through 0.312 at 32 surfels / 96x96).
- `python3 utils/jargon.py --port diff-surfel-tracing` clean.
- Working tree clean at `195d62f`; nothing uncommitted.

Backward, finite differences, the reflected bounce and the cold-cache rerun are
all UNVERIFIED, because the harness cannot get past the forward stage on its
default 64-surfel scene.

## Port round 2 (2026-08-13, linux-gfx942, MI300X, ROCm 7.14) -- PARTIAL

The open defect from round 1 is diagnosed and the round-1 diagnosis is **wrong**:
the hang is not in the tracer at all. It is in HIP RT's BVH **build**, and the
port additionally did not compile from its committed sources. Both findings are
below, with the evidence that establishes each.

New fork commit:

- `9722c7b` `[ROCm] Guard the CUDA-only vector operators in auxiliary.h` (`head_sha`)

### Finding 1: round 1's evidence came from a stale staged header

`setup.py::_stage_runtime_files_into_pkg` copies `hiprt_tracer/kernels.h` and
`optix_tracer/{params,config,auxiliary}.h` into the importable package
directory at install time, and the runtime compiler reads *those copies*
(`SurfelTracer.pkg_dir`). Round 1 installed at 04:29 and then did the header
collapse (plan item 4) at ~04:59 without reinstalling, so every "forward works"
result in round 1 was produced by the pre-collapse `hiprt_tracer/auxiliary.h`
that the commit deletes. Reinstalling at the committed tree fails immediately:

```
Runtime compilation failed:
  auxiliary.h:366:4: error: use of undeclared identifier '__trap'
  kernels.h:187:20: error: use of overloaded operator '+=' is ambiguous
      (with operand types 'float3' (aka 'HIP_vector_type<float, 3>') and 'float3')
```

`optix_tracer/auxiliary.h` was never touched by `195d62f` (`git show --stat`
lists `optix_tracer/params.h` only), so the shared header still carried the
CUDA-only free operators for float2/float3/float4 and a `__trap()` call. HIP
declares those operators as members of the vector types, so every use is
ambiguous, and `__trap` is not available to the runtime compiler. `9722c7b`
guards both on `USE_ROCM`; the CUDA preprocessed output is unchanged.

**Rule this establishes for this project:** after ANY edit to a staged header,
reinstall before believing a run. A passing trace proves nothing about the
working tree until the package copies have been refreshed. Promoted to the
`cuda-to-rocm` skill (`references/no-hip-equivalent.md`), because any port that
compiles kernels at runtime from files it copies into its package has this trap.

### Finding 2: the hang is in HIP RT's BVH build, not in the traversal

Round 1 concluded the hang was inside a single
`hiprtGeomTraversalAnyHit::getNextHit()`. That is disproved:

1. Replacing `surfelFilter` with `return true;` as its first statement -- a
   filter that touches no payload, so the retrace loop cannot iterate -- still
   hangs the 48-surfel scene.
2. Iteration guards were added to all three loops of
   `GeomTraversal::getNextHit` in the staged
   `hiprt_root/hiprt/impl/hiprt_device_impl.h` (cap 4096, reporting the loop
   that tripped through `hit.primID` into the image). The guards are live -- at
   cap 3 they fire on 61 of 256 pixels at 40 surfels -- and at cap 4096 the
   48-surfel scene still hangs with no guard ever firing.
3. Splitting `build_acceleration_structure` from the trace shows the hang
   happens **before any trace kernel runs**: `BUILD OK` never prints.
4. `AMD_LOG_LEVEL=3` names the last kernel the HIP runtime is asked for:
   `Collapse_TrianglePairNode_ScratchNode`.

So the hang is HIP RT's own `Collapse` kernel (`hiprt/impl/BvhBuilderKernels.h`,
`__device__ void Collapse`). It is a persistent task-queue kernel whose lanes
spin in `while (hiprt::any(!done))` and whose only exit for a lane that never
receives a task is

```
if ( atomicAdd( &header->m_referenceCount, 0 ) == referenceCount ) done = true;
```

The reference counter is monotonic, so an **exact** comparison is missed
permanently if the emitted count ever passes `referenceCount`. Changing that
one comparison to `>=` in the staged copy makes every previously hanging scene
build and trace (verified on the rotations-perturbed 64-surfel scene: `BUILD OK
5.41s`, `TRACE OK`, acc 0.3154). So the count **overshoots**: the collapse emits
more references than the primitive count it was given.

`>=` is deliberately NOT shipped. `referenceIndices` is allocated with exactly
`primitives.getCount()` entries, so an overshooting count also means
`referenceIndices[rangeAddr]` writes past the end of that temporary buffer, and
relaxing the comparison would trade a hang for silent memory corruption. The
miscount itself has to be found first. That is the next round's task, and the
`>=` result is the lead: instrument which lane emits the extra references.

Scene dependence, measured with `hiprtBuildFlagBitPreferFastBuild` (the
zero-valued default this port uses):

| surfels | 8-41 | 42 | 44 | 46 | 47 | 48 | 50 | 56 | 64 | 96 | 128 |
|---------|------|----|----|----|----|----|----|----|----|----|-----|
| builds  | yes  | NO | yes| NO | NO | NO | NO | NO | NO | NO | NO  |

Two build-flag workarounds were measured and both only move the threshold, so
neither is a fix and neither was committed:

- `hiprtBuildFlagBitDisableTrianglePairing`: 48 surfels builds, 46 and 64 hang.
- `hiprtBuildFlagBitPreferBalancedBuild` (PLOC): every unperturbed scene from 8
  to 200 surfels builds, and its image matches LBVH where LBVH completes
  (acc 0.3000/0.3013/0.3069 vs 0.3000/0.3012/0.3059 at 8/32/40 surfels), but
  the rotations finite-difference scene at 64 surfels still hangs. PLOC and
  LBVH call the *same* `Collapse` kernel, which is what makes the shared
  collapse -- not the hierarchy builder in front of it -- the defect.

### Harness results at `9722c7b` (96x96, 64 surfels, default flags)

The harness gets much further than round 1: the forward completes on the
64-surfel scene that used to hang, and the backward, which had never run at
all, runs end to end. Measured with the Balanced builder in place (the only way
to reach these stages today; the committed tree still hangs in the forward):

- import/API: 5/5 PASS.
- forward: finite rgb/depth/normal PASS, bit-identical rerun PASS, hit-fraction
  and hit-depth checks FAIL (see harness corrections below).
- backward: all eleven gradients finite PASS, all six nonzero checks PASS,
  `grad_grads3D` nonzero wherever `grad_means3D` is PASS (23 surfels), cosine
  vs `grad_means3D` 0.8721 against a 0.9 threshold FAIL.
- finite differences: colors ratio 0.9982 PASS, opacities 0.8156 PASS, means3D
  0.6784 PASS, scales ratio -0.0009 FAIL (fd 1.41e6 vs analytic -1.31e3, three
  orders of magnitude out, so the difference quotient is dominated by a
  discontinuity rather than disagreeing with the gradient), rotations NOT
  REACHED -- its scene hangs the build.
- reflected bounce, cold cache: NOT REACHED.

### Harness corrections made this round (MOAT side, not the fork)

- `others_precomp` was built `(P, 3)`. `AUX_CHANNELS` is 2 (`config.h`) and the
  kernel indexes the buffer with that stride, so the extra column made every
  read misaligned and made autograd reject the `(P, 2)` gradient with
  "invalid gradient at index 7". Now `(P, 2)`. This was masking the whole
  backward stage.
- Still to correct, both mis-calibrated against a scene the harness had never
  actually run: "genuine hits and misses" thresholds on `acc > 1e-3`, but the
  big mirror surfel legitimately contributes above that everywhere, so 1.000 is
  the right answer for this scene and the check needs `acc > 0.5`; and "hit
  depths plausible" reads `dpt` as a distance when it is an unnormalized
  weighted sum, so it should test `dpt / acc` on pixels with real coverage.
  The `grad_grads3D` cosine threshold of 0.9 is likewise invented; the
  densification gradient is depth-scaled, so it is not required to be that
  parallel.

### Also found, not yet fixed

`hiprtSetCacheDirPath(ctx, pkg_dir + "/hiprt_cache")` does not take: the
compiled binaries land in `<cwd>/cache`, HIP RT's default relative path, so the
package writes a `cache/` directory into whatever directory the process was
started in. The harness's cold-cache stage deletes `pkg_dir/hiprt_cache` and
will not observe the real cache.

## Port round 3 (2026-08-14, linux-gfx90a, MI250X, ROCm 7.14) -- GPU GATE MET

Round 2's open defect is root-caused and fixed, and the harness now passes
end to end on this platform. The fix is one token in HIP RT and it is not
scene-specific, not builder-specific and not a workaround.

New fork commit:

- `b7ff795` `[ROCm] Fix a 64-lane wavefront hang in the HIP RT build` (`head_sha`)

### The Collapse miscount: a 32-bit shift in a 64-lane mask

Round 2 established that HIP RT's `Collapse` kernel emits more references than
the primitive count and therefore steps over its own equality exit test, and
left "find the miscount" as this round's task. The miscount is upstream of
`Collapse`, in `openNodes`, and both share the cause.

`hiprt/impl/BvhBuilderKernels.h` builds a per-subgroup lane mask three times
(lines 144, 709, 1104 at tag `3.1.0.cb09c56`):

```c++
const uint64_t subwarpMask = ( ( 1 << BranchingFactor ) - 1 )
                             << static_cast<uint64_t>( ( BranchingFactor * subwarpIndex ) );
```

`( 1 << BranchingFactor ) - 1` has type `int`, and the shift's result type is
the promoted LEFT operand, so the whole expression is evaluated in 32 bits and
only then widened to `uint64_t`. With `BranchingFactor` 4 and `WarpSize` 64
(`hiprt_common.h:202-214`, the CDNA/`HIPRT_RTIP 0` arm covering `__gfx90a__`
and `__gfx942__`), `subwarpIndex` runs 0..15 and the shift count reaches 60.
Shifting an `int` by >= 32 is undefined, and AMD neither traps nor saturates:
the shift instruction uses the low 5 bits of the count, so the count wraps and
lanes 32..63 receive the mask of lanes 0..31.

The consequence is in `openNodes`, which selects the widest child of a
subgroup with
`__ffsll( ballot( maxArea == area ) & subwarpMask ) - 1` and then `shfl`s that
lane's `childIndex`. The upper half of the wavefront reads the LOWER half's
ballot bits, so it opens a subtree belonging to a different task. The same
subtree can be opened into two slots, `Collapse` then emits more leaf
references than there are primitives, `header->m_referenceCount` passes
`referenceCount` without ever equalling it, and every lane spins in
`while ( hiprt::any( !done ) )` forever.

This explains everything round 2 measured and could not explain:

- why it looked size dependent but was not (44 disks built, 42 and 46 did not):
  whether a wave has 8 or more subgroups holding live tasks whose duplicate
  actually changes the reference total is a property of the tree shape.
- why LBVH, LBVH-without-pairing and PLOC all hang: they share `Collapse` AND
  `openNodes`.
- why `>=` "fixed" it while still writing past `referenceIndices`: `>=` treats
  the symptom; the extra references were real writes.
- why only wave64 platforms are affected: a 32-lane wavefront has
  `subwarpIndex <= 7`, so the shift count never reaches 32 and the mask is
  correct. HIP RT is presumably validated on RDNA, where this is invisible.

The fix is `1ull << BranchingFactor` at all three sites, added to
`third_party/hiprt-rocm-fixes.patch`. It is arch-unified by construction: the
wave32 mask is bit-for-bit what it was.

Promoted to the `cuda-to-rocm` skill (`references/fault-classes.md`, wavefront
section) as its own fault class -- "a lane mask built with a 32-bit literal
wraps instead of overflowing" -- because it is not specific to HIP RT and the
existing "masks must be 64-bit" note only covers the mask's TYPE, not the
arithmetic that produces its value.

Not explained, and left as an honest loose end: EnvGS Stage 2 validated this
tracer on gfx90a against the same HIP RT version, which should have tripped
the same bug on its (much larger) scenes. That evidence is at the old tracer
commit and was not reproducible here, so it is recorded rather than
rationalised.

### Harness results at `b7ff795`: 42/42 PASS

```
PYTORCH_ROCM_ARCH=gfx90a, ROCm 7.14.60850, torch 2.14.0a0+git7d05abc
HIP_VISIBLE_DEVICES=0 python3 projects/diff-surfel-tracing/validation/validate_tracer_rocm.py
```

- import/API 5/5; forward finite, covered fraction 0.232, depth range
  [2.159, 4.574], bit-identical rerun.
- backward: all eleven gradients finite, all six nonzero checks pass,
  `grad_grads3D` nonzero wherever `grad_means3D` is (31 surfels), median
  per-surfel cosine against `grad_means3D` 0.9897.
- finite differences over 6 random directions at eps 1e-3: colors cosine
  1.0000 slope 0.9983; opacities 1.0000 / 0.8055; means3D 0.9992 / 0.9322;
  scales 0.9983 / 0.9066; rotations finite.
- reflected bounce (`max_trace_depth=1`, `specular_threshold=0.1`): image
  changes by up to 0.499, all gradients finite.
- cold compilation cache: cleared, rerun bit-identical, cache repopulated.

Both the GPU and the tree are honest: the run above is from a clean
`rm -rf build *.egg-info diff_surfel_tracing/hiprt_root` reinstall against a
HIP RT built from a FRESH clone of the pinned tag with only the committed
`third_party/hiprt-rocm-fixes.patch` applied, and that patch reproduces the
tested tree byte for byte (`git diff` of the verification clone equals the
working clone's).

### Harness corrections made this round (MOAT side, not the fork)

Round 2 listed three mis-calibrations "still to correct" and they are now
corrected, each against a measurement rather than a guess:

- **hits and misses.** Measured `acc` range on this scene is
  [0.012, 0.996] and NO pixel is empty -- the mirror surfel spans the frame,
  so `acc > 1e-3` is 1.000 by construction and proves nothing. The threshold
  is now `acc > 0.5`, which separates pixels a near-opaque surfel covers from
  pixels catching a Gaussian tail: 0.232, inside the (0.05, 0.95) band.
- **hit depths.** `dpt` is the coverage-weighted SUM of hit distances, so it
  is only a depth after dividing by `acc`, and only where `acc` is large. Raw
  `dpt` over `acc > 1e-3` ranged down to 0.073 (a tail pixel), which is what
  failed; `dpt / acc` over `acc > 0.5` is [2.159, 4.574], matching the scene's
  1.5-4.2 z extent.
- **`grad_grads3D` correlation.** `grads3D` is the same per-hit position
  gradient reweighted by hit distance, so per-surfel cosines are high (median
  0.9897, 48 per cent above 0.99) while a few surfels whose near and far hits
  cancel differently go negative (min -0.405). One cosine over the flattened
  stack is dominated by the largest-gradient surfel and read 0.8878. The check
  is now the MEDIAN per-surfel cosine.

And one correction round 2 did not anticipate:

- **The finite-difference step was below the float32 noise floor.** The loss
  is O(1e4) in float32, so a central difference divided by `2*eps` amplifies
  each term's ~1e-3 rounding by `1/(2*eps)`. At the old `eps=1e-4` the
  `means3D` quotient was pure noise: measured ratios 0.43 (eps 1e-4), 1.66
  (3e-5), 0.53 (3e-4) against a stable 0.93 at 1e-3, and the 8-direction
  cosine collapses from 0.9939 at 1e-3 to 0.5116 at 1e-4. All parameters now
  use `eps=1e-3` and SIX random directions, scored by the cosine of the
  6-vector plus a least-squares slope. That is strictly stronger than the old
  single-direction sign-and-ratio test: a ray tracer's loss is piecewise
  smooth, and one direction can land on a silhouette step and decide the gate
  by itself.

`opacities` sits at slope 0.806 with cosine 1.0000, stable across eps 1e-3 and
1e-4 and across all six directions. It is a consistent ~19 per cent bias, not
noise, and is almost certainly the hard alpha cutoff the compositing loop
applies (the difference quotient sees surfels crossing the include threshold;
the analytic gradient correctly does not). It is inside the plan's [0.5, 1.8]
band and identical in character to gfx942's 0.8156, so it is recorded, not
chased.

### Corrections to round 2's remaining note

- **`hiprtSetCacheDirPath` DOES take.** Round 2 recorded that compiled
  binaries land in `<cwd>/cache` and that the cold-cache stage therefore
  observes nothing. Measured here: every binary (tracer kernels AND HIP RT's
  own BVH builder kernels) is written to
  `diff_surfel_tracing/hiprt_cache/`, and the cold-cache stage genuinely
  clears and repopulates it. What remains is cosmetic: an EMPTY `cache/`
  directory is created in the process's working directory, from HIP RT's
  default `Compiler::m_cacheDirectory = "cache"` before the context's path
  override is applied. Worth a follow-up in HIP RT, not a defect in the port.

### Build recipe (this host: no /opt/rocm, ROCm is a pip SDK)

```
export ROCM_PATH=/opt/conda/envs/py_3.12/lib/python3.12/site-packages/_rocm_sdk_devel
export HIP_PATH=$ROCM_PATH PATH=$ROCM_PATH/bin:$PATH
git clone --recursive --depth 1 -b 3.1.0.cb09c56 \
    https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT.git /var/lib/jenkins/HIPRT
git -C /var/lib/jenkins/HIPRT apply <fork>/third_party/hiprt-rocm-fixes.patch
cmake -DCMAKE_BUILD_TYPE=Release -DBITCODE=OFF -DNO_UNITTEST=ON \
    -DHIP_PATH=$ROCM_PATH -S /var/lib/jenkins/HIPRT -B /var/lib/jenkins/HIPRT/build
cmake --build /var/lib/jenkins/HIPRT/build --target hiprt03001 -j32   # ~40 s
HIPRT_HOME=/var/lib/jenkins/HIPRT PYTORCH_ROCM_ARCH=gfx90a \
    pip install -e <fork> --no-build-isolation --no-deps -v           # ~2 min
```

The only deviation from the README is that `HIP_PATH`/`ROCM_PATH` point at the
pip SDK instead of `/opt/rocm`; `setup.py` already reads `ROCM_PATH` with
`/opt/rocm` as the default, so nothing in the fork needed changing for it. The
device name here is "AMD Instinct MI250X / MI250", so the patch's
`getCacheFilename` sanitize is load-bearing on this platform (unlike gfx942,
where round 2 verified it is unnecessary).
