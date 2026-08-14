# diff-surfel-tracing notes

A dependency fork: the tracer EnvGS's reflection path renders through.

## Why the platform row is empty

This has no test suite of its own. The code is exercised only through the project
that consumes it, so a GPU run against this repository alone would prove nothing.
The empty row is accurate, not a gap in the record.

The validation lives with **EnvGS**, `completed` on linux-gfx1100, linux-gfx90a, windows-gfx1101, windows-gfx1201.

Superseded from round 1 on: a GPU gate for this repository alone was authored and
it does prove something. **The harness is `example/validate_rocm.py` in the fork**
(`projects/diff-surfel-tracing/src/example/validate_rocm.py` in a checkout) -- it
ships with the port and MOAT keeps no copy, since round 6. Run it from the fork
checkout root: `HIP_VISIBLE_DEVICES=0 python3 example/validate_rocm.py`. Rounds 1-5
kept it at `projects/diff-surfel-tracing/validation/validate_tracer_rocm.py`, which
is the path every entry before round 6 names.

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

## Review 2026-08-14 (linux-gfx90a, `moat-port` b7ff795 vs upstream ef6f24b) -- CHANGES REQUESTED

Problems only. Round 3's wave64 diagnosis is correct as far as it goes -- the
`subwarpMask` reasoning was re-derived independently and holds (shift result
type is the promoted LEFT operand, so `( ( 1 << BranchingFactor ) - 1 ) <<
static_cast<uint64_t>( ... )` is a 32-bit `int` expression and the
`static_cast` on the right operand changes nothing; `BranchingFactor` 4 with
`WarpSize` 64 at `hiprt_common.h:202-214` gives shift counts to 60; `1ull`
is bit-identical on wave32, where `subwarpIndex <= 7`). The patch also applies
cleanly to the pinned tag and reproduces the tested tree byte for byte
(verified: fresh clone at `8602b8c` + `git apply` diffs equal to the build
clone). The fix is right. It is also incomplete, and the gap is not cosmetic.

### 1. The same 32-bit lane-mask defect is live one screen away, and it is silently dropping half the scene

`third_party/hiprt-rocm-fixes.patch` fixes the three `subwarpMask` sites and
stops. `hiprt/impl/BvhBuilderKernels.h:405`, inside `PairTriangles`, has the
same class:

```c++
uint64_t activeMask = hiprt::ballot( valid );
...
activeMask &= ~( 1u << firstPairedLane );   // firstPairedLane can be 0..63
```

`~( 1u << k )` is an `unsigned int`, so it ZERO-EXTENDS into the 64-bit AND and
clears the entire upper half of `activeMask` -- for every `k`, not just
`k >= 32`. The loop that hands each lane its turn as `broadcastLane` therefore
ends as soon as the low 32 bits drain; lanes 32..63 never get a turn, keep
`pairedIndex == InvalidValue`, and their triangles are never written to
`pairIndices`. `LbvhBuilder.h:261` then calls `primitives.setPairs( pairCount,
pairIndices )`, so those triangles do not exist for the BVH at all. This is
live on this port: `hiprt_tracer/hiprt_wrapper.cpp:235` builds with
`hiprtBuildFlagBitPreferFastBuild` and does not set
`hiprtBuildFlagBitDisableTrianglePairing`, and round 2 already observed
`Collapse_TrianglePairNode_ScratchNode`, so pairing runs.

Measured on this host at `b7ff795`, harness scene (64 surfels -> 128 triangles
-> two 64-lane waves), staged `hiprt_root` copy, cache cleared between runs:

| staged `BvhBuilderKernels.h:405` | surfels with a position gradient | acc mean | acc max |
|---|---|---|---|
| as shipped (`1u`) | **31 / 64** -- indices 0-15 and 32-47 only | 0.3185 | 0.9957 |
| `1ull` | **62 / 64** -- all but two | 0.3285 | 0.9989 |

The surviving indices are exactly the low 32 lanes of each wave. Half the
geometry is missing from the BVH on wave64, and the round-3 "42/42 PASS" was
measured against that half-empty scene: nothing in the harness looks at
geometry completeness, so every check still passed. `surface.json`'s
`covered` entries for `library:optix_tracer_forward` and
`library:optix_tracer_backward` inherit the same overstatement.

Add `1ull` at that site to the patch, rebuild HIP RT from a fresh clone, rerun
the harness, and re-record the forward numbers (`covered fraction`, depth
range) and the finite-difference results -- they will all move.

### 2. Same class again, on the architecture the windows gate will run

`hiprt/impl/BvhBuilderKernels.h:1989` and `:1993` in `PackLeavesWarp`:

```c++
uint64_t packetMask = hiprt::ballot( ... );
...
packetMask ^= 1 << broadcastLane0;
```

`1 << 31` is a signed-int expression; converted to `uint64_t` it sign-extends
to `0xFFFFFFFF80000000` and SETS bits 32..63, so `while ( packetMask )` cannot
terminate once lane 31 is broadcast. `PackLeavesWarp` builds
`TrianglePacketNode`, which `Context.cpp:1073` selects for `getRtip() >= 31`,
i.e. RDNA4 -- gfx1201, one of the windows-gate candidates. Reasoned, not
measured here (no RDNA4 on this host). Fix it in the same pass rather than
rediscovering it as a Windows hang. While in the file, grep the rest of the
dependency for the class: `PlocBuilderKernels.h:283` and
`BvhBuilderUtil.h:169,294` already use `1ull << laneIndex` correctly, which is
what makes the four remaining sites oversights rather than convention.

### 3. The promoted lesson prescribes the check that was not run

`.claude/skills/cuda-to-rocm/references/fault-classes.md` (round 3 hunk) is
accurate and correctly placed, and it says "Grep every `1 <<` and `1u <<`
whose shift count is derived from a lane index" -- the grep that finds items 1
and 2 in the very dependency the port patches. Extend the entry with the
second shape while fixing them, because its failure mode is different and
worse: `mask &= ~( 1u << lane )` on a 64-bit mask wipes the whole upper half
regardless of the shift count (silent wrong answer), and `mask ^= 1 << lane`
at lane 31 sets the upper half (hang). The current text only describes the
wrapped shift count.

### 4. The harness cannot see missing geometry

`projects/diff-surfel-tracing/validation/validate_tracer_rocm.py:236-249`
computes `touched` and reports it, but only asserts consistency between
`grad_grads3D` and `grad_means3D` over the surfels that were hit; 31 of 64
passed. Add a completeness check: every surfel in this scene is in front of
the camera and inside the frustum, so the gate should require essentially all
of them to receive a position gradient (allow a small margin for a genuinely
occluded surfel and print the missing indices when it fails). Without it, any
future geometry loss in the BVH path is invisible again.

The other four recalibrations were fact-checked against the code and stand:
`D += w * dpt` (`hiprt_tracer/kernels.h:522`, upstream
`optix_tracer/forward.cu:284`) confirms `out_dpt` is a coverage-weighted sum,
so `dpt / acc` on `acc > 0.5` is the right depth test; `dL_dgrads3D +=
dL_dmean3D * 0.5f * dpt` (`kernels.h:883`, upstream
`optix_tracer/backward.cu:624-626`) confirms `grads3D` is a per-hit
depth-reweighted copy of the position gradient, so a median per-surfel cosine
is the honest statistic and a global cosine is not; and the `eps` 1e-4 -> 1e-3
change with 6 directions scored by cosine plus least-squares slope is strictly
stronger than the single-direction ratio it replaced, with the acceptance
bands still the plan's. None of them is a loosening. They do all have to be
re-measured after item 1.

### 5. The third-party patch carries four undocumented API additions

`third_party/hiprt-rocm-fixes.patch:370-470` adds
`hiprtGetObjectToWorldFrameSRT`, `hiprtGetWorldToObjectFrameSRT`,
`hiprtGetObjectToWorldFrameMatrix` and `hiprtGetWorldToObjectFrameMatrix` to
`hiprt/impl/hiprt_device_impl.h`. None of them exists at the pinned tag, none
is referenced anywhere in HIP RT, and none is referenced by this port
(`kernels.h` has no `Frame` call). The patch header documents every other
hunk; these are unexplained. Drop them, or say in the header why a user must
apply them.

### 6. `pip install` copies 186 MB of HIP RT into the package

`setup.py::_stage_runtime_files_into_pkg` does
`shutil.copytree( HIPRT_HOME, dst_hiprt )`. Measured here: 186 MB, of which
`.git` is 33 MB, `contrib` 141 MB, `build` 2.9 MB and `dist` 1.2 MB -- and the
runtime JIT only needs `hiprt/` (960 KB) plus the headers it includes.
`package_data` carries the same tree into a wheel. Stage a filtered subset
(`ignore=shutil.ignore_patterns( '.git', 'build', 'dist', ... )` at minimum).

Related, for the PR text rather than the code: the per-launch chunk scratch is
`H * W * CHUNK_SIZE * 8` bytes (`trace_surfels.cpp:274`, `:449`) -- 1.2 MB at
96x96 but 265 MB at 1080p, allocated for both the forward and the backward
launch. It is inherent to the design and correctly explained in
`optix_tracer/params.h`, but a maintainer will want it stated up front.

### 7. Attribution and dangling document references in upstream-visible source

- "Authored with Claude (Anthropic)." at `hiprt_tracer/hiprt_wrapper.h:14`,
  `hiprt_tracer/hiprt_wrapper.cpp:11` and `hiprt_tracer/kernels.h:31`. New
  author lines are against the house rule and the AI-assistance disclosure is
  already where it belongs, in the commit bodies. Remove all three.
- "see PORTING_GUIDE ..." at `hiprt_tracer/kernels.h:12` and `:1057`, and
  "See PORTING_GUIDE OptiX->HIPRT and UPSTREAM_FINDINGS" at
  `hiprt_tracer/hiprt_wrapper.cpp:168`. Neither document exists in this
  repository. Inline the one sentence that matters or drop the pointer.

### 8. Two unguarded shared edits contradict the commit body

`195d62f`'s body says "everything added here is either a new file or guarded
by USE_ROCM". Two changes are neither: `trace_surfels.cpp:276` and `:451`
change `Params params;` to `Params params{};`, and
`diff_surfel_tracing/__init__.py:252` replaces
`site.getsitepackages()[0] + '/diff_surfel_tracing'` with a `__file__`-based
`pkg_dir`. Both are good changes on both back ends (value-init removes reads
of uninitialized `Params` fields; the `__file__` form is what makes an
editable install work), but the body has to say so -- a maintainer checking
that sentence against the diff finds two counterexamples in the first file
they open.

### 9. `b7ff795`'s Test Plan command cannot work as written

```
git -C /tmp/hiprt apply third_party/hiprt-rocm-fixes.patch
```

`git -C` changes directory first, so the patch path resolves to
`/tmp/hiprt/third_party/hiprt-rocm-fixes.patch`. Use an absolute path, or the
form `195d62f` uses.

### The EnvGS loose end: not orthogonal after all

Round 3 recorded, honestly, that EnvGS Stage 2 validated this tracer on gfx90a
against the same HIP RT without tripping the `Collapse` hang. Item 1 supplies a
mechanism rather than leaving it unexplained: `PairTriangles` was dropping the
upper half of every wave there too, so Stage 2's BVHs were built from roughly
half the triangles it thought it had. That changes the primitive count and the
tree shape feeding `openNodes`/`Collapse`, which is exactly what decides
whether the duplicate open ever pushes the reference count past the equality
exit -- so a scene that hangs at full geometry can build at half. It also
implies Stage 2's gfx90a images were rendered with geometry missing and nobody
would have seen it, since the criterion was "plausible reflections". Worth
re-checking EnvGS once item 1 lands; recorded here rather than acted on.

### Checked and clean

Not repeated above: the `kernels.h` shading and gradient math tracks upstream
`forward.cu` + `backward.cu` line for line (normalized-line comparison leaves
only traversal, launch and signature differences); the CUDA path diff is the
single unconditional `quat_to_rotmat_transpose` fix plus `USE_ROCM`-guarded
blocks; both kernels bound-check `h >= H || w >= W`; `IntersectionInfo` is 8
bytes and the `(H, W, CHUNK_SIZE * 2)` int32 scratch matches; no warp
intrinsic, `warpSize` or hardcoded 32 in this port's own device code; commit
titles are `[ROCm]`-prefixed and under 72 characters with no agent
`Co-Authored-By`; `jargon.py --port diff-surfel-tracing` is clean; the working
tree is clean at `b7ff795`; `third_party/optix` is an untouched gitlink and
the `third_party/hiprt` gitlink is the pinned `8602b8c`.

## Port round 4 (2026-08-14, linux-gfx90a, MI250X, ROCm 7.14) -- REVIEW FINDINGS FIXED

All nine review findings are addressed. The substantive one is real and the
reviewer's measurement reproduces exactly: the wave64 lane-mask class had two
more live sites, and one of them was silently building the BVH from half the
scene, which is what round 3's "42/42 PASS" was measured against.

New fork history (the two message-only rewrites are noted below):

- `6847672` `[ROCm] Add a HIP RT trace back end for AMD GPUs`  (was `195d62f`)
- `db2f6ba` `[ROCm] Guard the CUDA-only vector operators in auxiliary.h`  (was `9722c7b`)
- `fa7df21` `[ROCm] Fix a 64-lane wavefront hang in the HIP RT build`  (was `b7ff795`)
- `0eab27b` `[ROCm] Fix two more 32-bit lane masks in the HIP RT patch`
- `c086cde` `[ROCm] Stage only the HIP RT files the runtime compiler reads`
- `5254aa6` `[ROCm] Drop stale pointers and author lines from the back end`  (`head_sha`)

### Finding 1+2: the lane-mask class had two more sites, and both are fixed

`PairTriangles` (`BvhBuilderKernels.h:405`) and `PackLeavesWarp` (`:1989`,
`:1993`) now use `1ull`, in `third_party/hiprt-rocm-fixes.patch`. The reviewer's
mechanism is confirmed on hardware, not just re-derived: reverting ONLY line 405
in the staged `hiprt_root` copy, with the cache cleared, gives

```
[FAIL] the traced geometry is complete -- 31/64 surfels traced,
       missing [5, 16..31, 48..63]
```

i.e. exactly the upper 32 lanes of each of the two wavefronts, and restoring it
gives 62/64. Note the shape of the two failures differs and both matter:
`~( 1u << k )` zero-extends and wipes the upper half for EVERY k (wrong answer,
no diagnostic), while `1 << 31` sign-extends and SETS the upper half (hang).

The grep the promoted lesson prescribes was run over the whole dependency this
time, and it is now clean: the only other lane-derived shifts are
`BvhBuilderKernels.h:1854` and `:1948`, `( 1 << sublaneIndex ) - 1` where
`sublaneIndex = laneIndex % LanesPerLeafPacketTask` is under 4 and the mask it
is combined with is a `uint32_t`. Correct as written.
`PlocBuilderKernels.h:283` and `BvhBuilderUtil.h:169,294` already use `1ull`.

**Corrected in round 5: that paragraph is a false all-clear, do not inherit
it.** The grep it describes cannot find the site that actually faults the GPU,
`RadixSortKernels.h:465-480`, because that one narrows a 64-lane `__ballot`
into a `u32` and contains no shift literal at all. The enumeration also misses
`RadixSortKernels.h:485`, `u32 lowerMask = ( 1u << lane ) - 1` -- benign, since
`lane` is `threadIdx.x % 32` and the mask is a `u32`, but it is a lane-derived
shift in the very file that carried the defect. What survives re-verification
is only the narrow claim about `BvhBuilderKernels.h:1854`/`:1948`,
`PlocBuilderKernels.h:283` and `BvhBuilderUtil.h:169,294`. The sweep that does
find everything is in round 5 below.

Both sites are still present at HIP RT HEAD as of 2026-08-14 (fetched and
checked), so they join the GPUOpen report; registered as the deferral
`hiprt-32bit-lane-masks-pair-and-pack`, pending a person's ruling.

### Finding 3: the fault-class lesson now carries the second shape

`.claude/skills/cuda-to-rocm/references/fault-classes.md` gains a paragraph for
the 32-bit-literal-against-a-64-bit-mask shape, with both failure modes, the
"run the grep over the whole dependency, not just as far as the symptom you are
chasing" instruction, and the harness lesson (finiteness, determinism and
plausible ranges do not see missing geometry).

### Finding 4: the harness can now see missing geometry

`validate_tracer_rocm.py` gains `check_geometry_complete`. It asserts every
surfel centre projects inside the frame, then requires all but
`GEOMETRY_OCCLUSION_MARGIN` (4) of them to receive a position gradient, printing
the missing indices when it fails.

The margin is measured, not invented. The two surfels that legitimately go
untraced (5 and 25) are the two most edge-on disks in the scene:
`|cos(normal, view)|` 0.003 and 0.015 against a median of 0.41 -- a disk seen
edge-on covers no pixel. Any half-wave geometry loss lands at 31/64, far outside
the margin.

### Full 42-case suite RE-MEASURED at the complete scene: 44/44

Not carried forward. The whole suite was re-run at `5254aa6` against a HIP RT
built from a fresh clone of the pinned tag with only the committed patch applied,
after `rm -rf build *.egg-info diff_surfel_tracing/hiprt_root
diff_surfel_tracing/hiprt_cache` and a reinstall. 42 old checks plus the 2 new
geometry checks.

```
PYTORCH_ROCM_ARCH=gfx90a, ROCm 7.14.60850, torch 2.14.0a0+git7d05abc
HIP_VISIBLE_DEVICES=0 python3 projects/diff-surfel-tracing/validation/validate_tracer_rocm.py
44/44 checks passed
```

Round 3's numbers versus this round's, i.e. half-empty scene versus complete
scene. Every recalibration is re-justified against the complete scene here; none
was carried:

| measurement | round 3 (31/64 traced) | round 4 (62/64 traced) |
|---|---|---|
| covered fraction (`acc > 0.5`) | 0.232 | 0.241 |
| `dpt / acc` range | [2.159, 4.574] | [1.820, 4.574] |
| median per-surfel cosine `grads3D` vs `means3D` | 0.9897 | 0.9825 |
| fd colors cosine / slope | 1.0000 / 0.9983 | 1.0000 / 1.0021 |
| fd opacities | 1.0000 / 0.8055 | 0.9999 / 0.7428 |
| fd means3D | 0.9992 / 0.9322 | 0.9993 / 0.9387 |
| fd scales | 0.9983 / 0.9066 | 0.9923 / 0.8856 |
| reflected-bounce max delta rgb | 0.499 | 0.363 |

The bands are unchanged and all four recalibrations still hold on the complete
scene: `acc > 0.5` still separates covered from tail pixels (0.241, inside the
0.05-0.95 band); `dpt / acc` still lands inside the scene's 1.5-4.2 z extent
(the lower end moves to 1.82 because the restored geometry includes nearer
disks); the median per-surfel cosine is still the honest statistic and is now
taken over 62 surfels rather than 31; `eps = 1e-3` over 6 directions still gives
cosines at or above 0.99 on every parameter. The opacities slope moves 0.806 ->
0.743, still a consistent bias rather than noise and still inside the plan's
[0.5, 1.8] band -- the same hard alpha cutoff explanation, now over twice as
much geometry.

### Findings 5-9: hygiene

- **Four undocumented API additions removed** from the patch
  (`hiprtGet{ObjectToWorld,WorldToObject}Frame{SRT,Matrix}`). Confirmed first
  that they exist nowhere at the pinned tag and that nothing in the fork calls
  them.
- **Staging is filtered.** `setup.py` copies only `hiprt/` and
  `contrib/Orochi/ParallelPrimitives`, which are the only two subtrees HIP RT's
  runtime compiler opens (`hiprt/impl/RadixSort.cpp` names the second; the
  device headers include nothing outside those two). 186 MB -> 1016 KB, verified
  by a cold-cache run that compiles every builder and trace kernel from the
  staged copy.
- **Three author lines and three dangling doc pointers removed** from
  `hiprt_tracer/`. The register-pressure comments now state the workaround
  instead of pointing at a document that does not exist here.
- **`195d62f`'s body corrected** (now `6847672`): it claimed everything added
  was new or `USE_ROCM`-guarded. It names the two shared changes instead
  (`Params params{}` at both launch sites, the `__file__`-based `pkg_dir`), says
  they are back-end-neutral fixes, and offers them as separately droppable. The
  chunk-scratch magnitude (1.2 MB at 96x96, 265 MB at 1080p, per launch) is now
  stated in that body too, so a maintainer meets it up front.
- **`b7ff795`'s Test Plan fixed** (now `fa7df21`): `git -C /tmp/hiprt apply
  "$PWD/third_party/hiprt-rocm-fixes.patch"`. Its "42/42" block is replaced by
  what that commit alone establishes -- the build terminates -- because those
  numbers were taken against the half-empty scene.

Both message rewrites are message-only: `git diff` between the pre-rewrite and
post-rewrite branch tips is empty. No platform had a `validated_sha` at the old
shas, so nothing was orphaned.

### The EnvGS loose end: mechanism supplied, and it needs re-checking

Round 3 left it unexplained that EnvGS Stage 2 validated this tracer on gfx90a
against the same HIP RT without tripping the `Collapse` hang. The review supplies
the mechanism and it is now confirmed on hardware here: `PairTriangles` was
dropping the upper half of every wavefront under Stage 2 as well, so those BVHs
were built from roughly half the triangles the tracer thought it had. Half the
primitives is a different primitive count and a different tree shape feeding
`openNodes`/`Collapse`, which is exactly what decides whether the duplicate open
pushes the reference count past the equality exit -- so a scene that hangs at
full geometry can build at half. That resolves the "should have hung and did
not" question.

The consequence for EnvGS is the part that matters: **its gfx90a images were
rendered with geometry missing**, and its acceptance criterion ("plausible
reflections") could not have seen it. EnvGS should be re-checked against the
current patch level. Recorded here, not acted on.

### Build recipe: unchanged from round 3

Same commands, with `HIPRT_HOME=/var/lib/jenkins/HIPRT` and
`ROCM_PATH=/opt/conda/envs/py_3.12/lib/python3.12/site-packages/_rocm_sdk_devel`.
The patch still applies cleanly to a fresh clone of the pinned tag, and the fresh
clone plus patch reproduces the tested tree byte for byte (`git diff` of the
verification clone equals the build clone's, 381 lines each; the only files
present in one and not the other are HIP RT's own gitignored generated headers
`hiprt/hiprt.h` and `hiprt/hiprtew.h`).

## Review 2026-08-14 (round 4, linux-gfx90a, `moat-port` 5254aa6 vs b7ff795) -- CHANGES REQUESTED

Problems only. All nine round-3 findings are genuinely closed and independently
re-verified (list at the end). The round-4 work is correct as far as it goes.
It is also, for the second round running, an incomplete sweep of the same fault
class -- and this time the site it missed is not a silent wrong answer but a
reproducible GPU memory fault, in the second subtree `setup.py` newly stages.

### 1. The port hard-faults the GPU on any scene above ~3072 surfels, same fault class

`contrib/Orochi/ParallelPrimitives/RadixSortKernels.h:465-471` and `:476-480`
(pinned tag `8602b8c`), inside `OnesweepReorder`:

```c++
int warp = threadIdx.x / WARP_SIZE;   // WARP_SIZE is 32 (RadixSortConfigs.h:45)
int lane = threadIdx.x % WARP_SIZE;
...
u32 broThreads = __ballot( itemIndex < numberOfInputs );   // 64-bit result, u32 destination
...
u32 difference = ( 0xFFFFFFFF * bit ) ^ __ballot( bit != 0 );
```

The kernel treats a 256-thread block as eight 32-lane logical warps. On a
64-wide wavefront each physical wave holds two of them, `__ballot` returns all
64 lanes, and storing it in a `u32` keeps bits 0..31 -- so every ODD logical
warp gets the EVEN warp's ballot. Measured directly on this host
(`agent_space/ballot_trunc.hip`, gfx90a, one 256-thread block, predicate
`lane < 8` on even logical warps and `lane < 20` on odd):

```
thread   0 (logical warp 0): u32 ballot=0x000000ff  u64 ballot=0x000fffff000000ff
thread  32 (logical warp 1): u32 ballot=0x000000ff  u64 ballot=0x000fffff000000ff
```

Logical warp 1's own bits are the `0x000fffff` in the upper half; it reads
warp 0's instead. `warpOffsets[k] = digitCount + __popc( broThreads & lowerMask )`
(`:487`) then ranks each key against the wrong peer set, `leaderIdx =
__ffs( broThreads ) - 1` (`:494`) elects a leader from it, and the corrupted
`lpSum` propagates into `pSum[bucketIndex]`, which is the base of the GLOBAL
write `outputKeys[dstIndex]` / `outputValues[dstIndex]`.

This is live on this port. `hiprt/impl/RadixSort.cpp` delegates to
`Oro::RadixSort`, `LbvhBuilder.h:292` calls it for every build, and
`contrib/Orochi/ParallelPrimitives/RadixSort.cpp:243` selects the single-pass
kernel only while `n < SINGLE_SORT_WG_SIZE * SINGLE_SORT_N_ITEMS_PER_WI`
(128 * 24 = 3072). Above that it launches `OnesweepReorderKeyPair64`, which is
the kernel above. Measured on gfx90a at `5254aa6`, harness scene scaled up
(`agent_space/scale_probe.py`, P surfels -> 2P triangles -> ~P triangle pairs
after `PairTriangles`):

| P (surfels) | result |
|---|---|
| 64, 1000, 1500, 1536, 1600, 2048 | completes, finite output |
| 3072 | `Memory access fault ... kernel: OnesweepReorderKeyPair64`, exit 134 |
| 4096 | same fault, reproducible |

Cause pinned by a controlled edit of the staged `hiprt_root` copy with the
kernel cache cleared each time -- shifting the ballot into the logical warp's
half at both sites,

```c++
u32 broThreads = (u32)( __ballot( itemIndex < numberOfInputs )
                        >> ( 32u * ( ( threadIdx.x % warpSize ) / 32u ) ) );
```

makes P=4096 complete with finite output; restoring the file verbatim and
clearing the cache again brings the fault straight back. (That expression is a
probe that isolates the cause, not a reviewed patch -- pick the form HIP RT and
Orochi would take, and check the rest of `OnesweepReorder` for anything else
that assumes the logical lane is the physical lane.)

The consequence is not a corner case. A 3DGS scene is 10^5-10^6 surfels; every
real scene is above the threshold and none of them can build a BVH on gfx90a.
The 64-surfel harness is two orders of magnitude below it, which is why round 3
and round 4 both passed 44/44 over a back end that cannot render a real scene.

Asks:
- Add the fourth entry to `third_party/hiprt-rocm-fixes.patch`, with the same
  header explanation the other entries get. It is a defect in vendored Orochi
  rather than in `hiprt/`, which is worth saying explicitly in the header.
- Add a scale case to `validate_tracer_rocm.py` that crosses 3072 sorted
  elements, because nothing at 64 surfels can reach this code path. Do NOT
  reuse the geometry-completeness criterion there: measured at P=2048 the scene
  traces 1047/2048 with the sort working correctly, because a dense slab
  occludes itself. "Completes without a fault and returns finite output and
  gradients" is the assertion that has meaning at that size.
- Extend the deferral so the Orochi site joins the GPUOpen report. Confirmed
  unchanged at HIP RT HEAD (`e3c01fc`, fetched 2026-08-14).

### 2. The promoted lesson prescribes a grep that cannot find item 1

`.claude/skills/cuda-to-rocm/references/fault-classes.md` (round-4 hunk) is
accurate on both shapes it describes, but its sweep instruction is "every
`1 <<`, `1u <<` and `~( 1u << ... )` in an expression whose other operand is a
64-bit mask". `u32 broThreads = __ballot( ... )` matches none of those patterns,
and it is the shape that faults. The entry above it does say "a `uint32` mask is
wrong on a 64-wide wavefront", but as a statement about mask parameters, not as
something the grep looks for.

Extend the sweep instruction with the destination-type shape -- any 32-bit
variable, struct field or cast that RECEIVES a `__ballot` / `__activemask`
result -- and with the idiom that makes it wrong rather than merely truncated:
code that carves a block into fixed 32-lane logical warps
(`warp = threadIdx.x / 32`) and then uses a wave-wide collective, so the second
logical warp in each physical wave silently reads the first one's answer. Say
that the sweep covers the vendored subtrees the port ships, not just the
dependency's own headers: this one is in `contrib/`, staged by `setup.py`, and
compiled at runtime.

### 3. The round-4 sweep claim in this file is wrong and should be corrected

The round-4 entry says the grep "was run over the whole dependency this time,
and it is now clean: the only other lane-derived shifts are
`BvhBuilderKernels.h:1854` and `:1948`". Two corrections. The enumeration also
misses `RadixSortKernels.h:485`, `u32 lowerMask = ( 1u << lane ) - 1` -- benign,
since `lane` is `threadIdx.x % 32` and the mask is a `u32`, so the conclusion
about it stands, but it is a lane-derived shift and it is in the file that
carries item 1. And "clean" is not the finding; item 1 is. Rewrite it once item
1 is fixed so the next reader does not inherit a false all-clear. The
re-verified part is fine: `BvhBuilderKernels.h:1854` and `:1948` are
`( 1 << sublaneIndex ) - 1` with `sublaneIndex < LanesPerLeafPacketTask` and a
`uint32_t` mask, correct as written, and `PlocBuilderKernels.h:283`,
`BvhBuilderUtil.h:169,294` already use `1ull`.

### Verified closed, independently

Each round-3 finding re-checked against the code rather than the response.

1. `PairTriangles` -- `activeMask &= ~( 1ull << firstPairedLane )` present in the
   patch and in the patched source at `BvhBuilderKernels.h:405`; the recorded
   negative control (31/64, missing `[5, 16..31, 48..63]`) is exactly the surfel
   set that the upper 32 lanes of each of the two waves own, since surfel `i` is
   triangles `2i, 2i+1`. Internally consistent with the round-3 measurement.
2. `PackLeavesWarp` -- `1ull` at both `:1989` and `:1993`. Both sites confirmed
   still defective at HIP RT HEAD, as the deferral states.
3. `check_geometry_complete` -- `validate_tracer_rocm.py:225-242` asserts
   in-frame first, then a margin of 4. The margin holds: `make_scene` puts every
   centre at `|x|,|y| <= 0.6`, `z >= 1.5`, so `in_frame` is a real premise; the
   two untraced disks are scene-specific (a seed-1 scene traces 64/64), which is
   what "edge-on" predicts and a systematic loss does not; and any half-wave loss
   lands at 33 missing, eight times the margin. Full suite re-run here at
   `5254aa6`: 44/44, reflected-bounce delta 3.627e-01, matching the recorded
   0.363. The four recalibrations re-justified on the complete scene are sound.
4. The four `hiprtGet*Frame*` getters are gone from the patch.
5. Staging is 1016 KB (`hiprt/` 956 KB, `contrib/` 56 KB) and demonstrably
   sufficient: the cache was cleared twice during this review and every builder
   kernel recompiled from the staged copy, including `OnesweepReorderKeyPair64`,
   which comes from the `contrib` subtree.
6. `grep -riE "claude|anthropic|PORTING_GUIDE|UPSTREAM_FINDINGS"` over the
   tracked source is empty; the register-pressure comment now states the
   workaround.
7. Message-only rewrites confirmed by tree hash, not by inspection:
   `195d62f`/`6847672` both `c3ec7db`, `9722c7b`/`db2f6ba` both `c00df87`,
   `b7ff795`/`fa7df21` both `63b6ff9`. No platform held a `validated_sha`.
   Both new bodies are accurate, including the honest "the numbers that run
   reports are superseded" paragraph in `fa7df21`.
8. Covered in item 2 above.
9. Deferral `hiprt-32bit-lane-masks-pair-and-pack` registered and accurate;
   EnvGS mechanism recorded.

Also re-checked and clean: `jargon.py --port diff-surfel-tracing`; commit titles
`[ROCm]`-prefixed, 47-61 chars, no agent trailer, ASCII bodies with Test Plans;
the fork tree is clean at `5254aa6`; the CUDA path is unchanged from round 3;
and the committed patch applied to a fresh clone of `8602b8c` reproduces the
build clone's tree byte for byte (381-line diff, identical).

## Port round 5 (2026-08-14, linux-gfx90a, MI250X, ROCm 7.14) -- REVIEW FINDINGS FIXED

All three round-4 findings are addressed. Fork history gains one commit:

- `f35600e` `[ROCm] Fix a truncated wavefront ballot in the radix sort` (`head_sha`)

Nothing earlier was amended or rewritten, so `5254aa6` is still an ancestor.

### Finding 1: the ballot truncation is fixed, and the fault is gone

`third_party/hiprt-rocm-fixes.patch` gains a fourth lane-width entry, in
`contrib/Orochi/ParallelPrimitives/RadixSortKernels.h`. Both ballots in
`OnesweepReorder` now go through one helper placed immediately above it:

```c++
__device__ inline u32 logicalWarpBallot( bool predicate )
{
#if defined( ITS )
	return __ballot_sync( 0xFFFFFFFF, predicate );
#else
	const u32 logicalWarpInWave = ( threadIdx.x % warpSize ) / WARP_SIZE;
	return static_cast<u32>( __ballot( predicate ) >> ( logicalWarpInWave * WARP_SIZE ) );
#endif
}
```

Why this form rather than the reviewer's inline probe. It is arch-unified by
construction, not by an arch guard: on a 32-wide wavefront `threadIdx.x %
warpSize` is under 32, the shift is 0, and the emitted code is what it is
today; the CUDA `ITS` branch keeps `__ballot_sync` over a real 32-lane warp and
is untouched. One helper also replaces the two `#if defined( ITS )` blocks that
were split across the call arguments, so the two call sites are now single
lines and a future third ballot in this kernel picks up the same treatment for
free. `WARP_SIZE` (32, `RadixSortConfigs.h:45`) is the file's own logical warp
width and is used as such rather than as a literal.

Measured on this host at `f35600e`, staged `hiprt_root` copy, cache cleared
before each run:

| staged `RadixSortKernels.h` | P = 4096 |
|---|---|
| verbatim from the pinned tag | `Memory access fault ... kernel: OnesweepReorderKeyPair64` |
| with the patch | completes, 50/50 checks, rgb mean 0.19876 |

The negative control was run in that order, after the passing suite, so the
patched result is not a stale cache: the reverted run rebuilt every kernel from
the reverted source and faulted, and restoring the file plus clearing the cache
gave the same 50/50 as before, byte for byte on the reported numbers.

### The sweep, this time in three patterns

The round-4 sweep looked for shift literals only, which is why it could not see
this. Rerun over both staged subtrees -- `hiprt/` and
`contrib/Orochi/ParallelPrimitives`, i.e. everything `setup.py` ships and the
runtime compiler opens -- for all three shapes:

1. **Shift literal against a 64-bit mask** (`1 <<`, `1u <<`, `~( 1u << ... )`):
   clean. The four fixed sites plus the benign ones already enumerated;
   `RadixSortKernels.h:485` `( 1u << lane ) - 1` is the one the round-4 list
   missed, and it is correct as written (`lane` is `threadIdx.x % 32`, the mask
   is a `u32`, and after this fix `broThreads` is genuinely the logical warp's
   own 32 bits).
2. **32-bit destination receiving a wave-wide collective**
   (`grep -E "(uint32_t|u32|unsigned int|int)[ \t]+[A-Za-z_]+[ \t]*=[^;]*(ballot|activemask)"`):
   `RadixSortKernels.h:465` and `:476` were the only true positives, both fixed.
   The nine hits in `hiprt/` are all `uint32_t` receiving a `__popcll` COUNT or
   an `__ffsll` INDEX, never a mask -- counts and indices over 64 lanes fit in
   32 bits, so they are correct.
3. **Fixed 32-lane logical warps inside a wavefront**
   (`threadIdx.x / 32`, `% 32`, `/ WARP_SIZE`): `RadixSortKernels.h:448` is the
   only occurrence in either subtree that is relative to a hardcoded 32. The
   `% 32` hits in `BvhNode.h:713-789` are 32-bit word/bitfield packing, not
   lanes.

   **Corrected in round 6 (the round-5 wording claimed an absence, and the
   absence is not what was checked).** HIP RT's own kernels DO carve a wavefront
   into fixed logical sublane groups, in four places -- `BvhBuilderKernels.h`
   `:141-145`, `:706-710`, `:1100-1106` (`sublaneIndex = laneIndex %
   BranchingFactor`, `subwarpIndex = laneIndex / BranchingFactor`, and a
   `subwarpMask` over the subgroup) and `:1789-1790` (`subwarpIndex = laneIndex
   / LanesPerLeafPacketTask`, `sublaneIndex = laneIndex %
   LanesPerLeafPacketTask`, where `LanesPerLeafPacketTask` is 4,
   `BvhConfig.h:37`). What makes all four correct is a positive property, and it
   is the property to look for elsewhere rather than an absence: the carve is
   relative to the real `WarpSize` (`laneIndex = threadIdx.x % WarpSize`;
   `hiprt_common.h:204` gives 64 on CDNA and `:206` gives 32 elsewhere -- I
   re-checked every line cited here against the pinned tag), the subgroup mask
   is built in 64 bits (`( 1ull <<
   BranchingFactor ) - 1ull` shifted by the subgroup base, after the patch), a
   wave-absolute ballot result is reduced back to a subgroup index before it is
   compared against one (`:168-169`, `maxIndex = maxLaneIndex %
   BranchingFactor`), and `PackLeavesWarp` already does exactly what
   `logicalWarpBallot` now does -- `hiprt::ballot( ... ) >> (
   LanesPerLeafPacketTask * subwarpIndex )` at `:1844`, over a `uint64_t`. So
   the sweep's conclusion stands, and it stands for a reason a reader can
   re-check.

Also checked and clean in Orochi: `scanExclusive` and `ldsScanExclusive` are
LDS plus `__syncthreads`, with no warp collective at all, and the `warp`/`lane`
indexing in the reorder's phase 2 and 3 only addresses shared memory.

### Which project this one is reported to

Not HIP RT's own code, unlike the first three lane-width entries.
`contrib/Orochi` is vendored into the HIP RT tree (no submodule at this tag),
and the blob is identical to upstream Orochi's:

```
gh api repos/GPUOpen-LibrariesAndSDKs/Orochi/contents/ParallelPrimitives/RadixSortKernels.h --jq .sha
gh api repos/GPUOpen-LibrariesAndSDKs/HIPRT/contents/contrib/Orochi/ParallelPrimitives/RadixSortKernels.h --jq .sha
3fe37293fb4255b190ec21099bd63b0351c71f8b   (both, and the pre-image of our own diff)
```

So the defect is live at Orochi HEAD (`78fb3df`, 2026-08-13) and at HIP RT HEAD
(`e3c01fc`), and the report belongs to the Orochi repository, with HIP RT
picking it up by re-vendoring. Registered separately as the deferral
`orochi-32bit-ballot-onesweep-reorder` (component `orochi`) rather than folded
into `hiprt-32bit-lane-masks-pair-and-pack`: different repository, different
severity, and the tooling has no "append context" verb -- both are pending a
person's ruling and should travel in the same GPUOpen conversation.

### Finding 2: the lesson now prescribes a sweep that finds this

`.claude/skills/cuda-to-rocm/references/fault-classes.md` gains a paragraph
that replaces the single `1 <<` grep with the three patterns above, says the
sweep covers a dependency's vendored `contrib/` copies of other projects
because they are staged and runtime-compiled like its own headers, and adds the
Orochi case with the arch-unified fix form and the reachability trap (a
dependency that dispatches by problem size hides a defect from every small
test scene).

### Full suite at `f35600e`: 50/50

44 previous checks plus 6 new scale checks. Fresh HIP RT clone of the pinned
tag with only the committed patch, `rm -rf build *.egg-info
diff_surfel_tracing/hiprt_root diff_surfel_tracing/hiprt_cache`, reinstall,
then a cold-cache run.

```
PYTORCH_ROCM_ARCH=gfx90a, ROCm 7.14.60850, torch 2.14.0a0+git7d05abc
HIP_VISIBLE_DEVICES=0 python3 projects/diff-surfel-tracing/validation/validate_tracer_rocm.py
50/50 checks passed
```

Every round-4 number reproduced unchanged (covered fraction 0.241, depth range
[1.820, 4.574], geometry 62/64 missing [5, 25], median `grads3D` cosine 0.9825,
fd colors 1.0000/1.0021, opacities 0.9999/0.7428, means3D 0.9993/0.9387, scales
0.9923/0.8856, reflected-bounce delta 3.627e-01), which is the expected result:
at 64 surfels the sort takes the single-pass kernel and this fix cannot reach
it. The new numbers:

| P | covered fraction | surfels with a position gradient | rgb mean |
|---|---|---|---|
| 4096 | 0.386 | 1343 / 4096 | 0.19876 |
| 16384 | 0.457 | 3309 / 16384 | 0.25417 |

`check_scale` deliberately does NOT reuse `check_geometry_complete`. A slab of
this density occludes itself -- 1343 of 4096 is what a correct sort produces
here, and the reviewer measured 1047/2048 at the same scene -- so completeness
carries no information at this size. What it asserts is that the build and both
passes finish (a wrong sort takes the process down with a memory fault, which
no assertion can catch, so the harness surviving IS the check), that every
output and every gradient is finite, and that the image is non-degenerate.
4096 crosses the 3072 threshold with a single sort block; 16384 is four blocks,
which also exercises the cross-block lookback in `OnesweepReorder`.

### Gotchas

- **Rebuilding the HIP RT host library does nothing for a kernel-source edit.**
  `cmake --build ... --target hiprt03001` after editing `RadixSortKernels.h`
  took 0.246 s here -- the file is device source, read from
  `HIPRT_PATH`/`hiprt_root` and compiled at runtime, and it is not in the host
  library at all with `BITCODE=OFF`. What has to happen is the reinstall that
  re-stages the file into `diff_surfel_tracing/hiprt_root` plus
  `rm -rf diff_surfel_tracing/hiprt_cache`. Editing the staged copy directly is
  the fast way to run a controlled experiment, and the cache must be cleared
  each time or nothing you changed is compiled.
- **The patch still reproduces the build tree byte for byte.** Fresh clone of
  `8602b8c` from GitHub, `git apply` of the committed patch, `diff -r` against
  the build clone: identical apart from HIP RT's own gitignored generated
  headers `hiprt/hiprt.h` and `hiprt/hiprtew.h`. 12 files, 133 insertions,
  49 deletions.
- **Open, for the reviewer to rule on:** every Test Plan in this branch's
  commits ends with `python3 validate_tracer_rocm.py`, but the harness lives in
  this repository under `projects/diff-surfel-tracing/validation/`, not in the
  fork, so a maintainer reading the commit cannot run that command. It has been
  that way since round 1 and no round has flagged it. Either the harness ships
  with the port or the Test Plans should say where it comes from.

## Review 2026-08-14 (round 5, linux-gfx90a, `moat-port` f35600e vs 5254aa6) -- CHANGES REQUESTED

Problems only. The round-5 device fix is correct and, this time, the sweep behind
it is complete -- both re-verified against the code and re-measured on this host,
including my own negative control rather than the recorded one (list at the end).
Nothing below is in the port's device code or in its analysis. All three findings
are in text a maintainer or a future reader will take at face value.

### 1. Every Test Plan names a command that cannot be run from this repository

Ruled, since round 5 asked. This is a defect the port must fix, not a person's
PR-shaping call, because it is a factual error in upstream-visible text: seven of
the eight commit bodies on this branch end their Test Plan with a fenced

```
HIP_VISIBLE_DEVICES=0 python3 validate_tracer_rocm.py
```

and `git ls-files` on `moat-port` has no such file -- no test file at all.
`1afb40b`, `6847672`, `db2f6ba`, `0eab27b`, `c086cde`, `5254aa6` and `f35600e`
all carry it; only `fa7df21` does not. A literal command in a Test Plan is a
promise that a reader with the hardware can re-run it. Naming a script that is
not in the tree, from a repository that has no tests at all, reads as "there is a
validation script you have not found" rather than "this lives elsewhere".

The requirement is narrow: every fenced command in every commit body must be
runnable from a clone of the fork. What satisfies it is a choice, and the person
approving the PR will see either outcome, so pick one now rather than deferring:

- **Recommended: ship the harness in the fork, as the single copy.**
  `projects/diff-surfel-tracing/validation/validate_tracer_rocm.py` is 455 lines
  and imports only `os`, `shutil`, `sys`, `torch` and `diff_surfel_tracing`
  (:13-20). It needs no data, no framework and no fixture, its own docstring
  already says `Run: python3 validate_tracer_rocm.py` from the repository root,
  and it is free of MOAT vocabulary. Upstream has no test suite, so this PR asks
  a maintainer to take ~2,400 lines of new back end on trust with no way to check
  it beyond `example/render.py`, which needs the author's Google Drive archive.
  A self-contained ROCm validation script is the answer to "how do I know this
  works on my machine", and it makes the Test Plans true with no rewording.
  plan.md:286 put the harness in MOAT on the reasoning that this repository
  "cannot get one upstream without inventing a test framework for someone else's
  repository" -- one standalone script next to the existing standalone
  `example/render.py` is not a framework, so that reasoning does not hold against
  shipping it. If it ships, keep ONE copy: move it, and point the validation
  command in plan.md and in notes at `projects/diff-surfel-tracing/src/<path>`.
  Two copies will drift, and the drifting one is the one the gate runs.
  Name it for what it is (it asserts `torch.version.hip`, so it is the ROCm
  validation, not a general test) and mention it in the README's AMD section.
- **Otherwise: reword all seven Test Plans** so the fenced block runs. A path
  that does not exist in the repository is not a fix; the block has to name
  something a reader can execute.

Either way this is a message-only rewrite of seven commits. No platform holds a
`validated_sha`, so nothing is invalidated; confirm by tree hash as round 4 did.

### 2. The README still attributes all four patch entries to HIP RT

`README.md:57-58`, upstream-visible, and made wrong by round 5's own
reclassification:

```
# Build HIP RT, applying the fixes it still needs (see the patch header for what
# each one is for; they are HIP RT bugs and are being contributed upstream)
```

The patch header was correctly updated to "bugs in HIP RT and in the copy of
Orochi it vendors" and the whole point of the round-5 attribution work is that
the fourth entry is reported to a different repository. The README, which is the
text a maintainer actually reads, still says all of them are HIP RT bugs.

Second clause in the same sentence: "are being contributed upstream" is not true
of any of them. `hiprt-32bit-lane-masks-pair-and-pack` and
`orochi-32bit-ballot-onesweep-reorder` are both `"status": "open"`,
`"upstream_issue": null`, `"decided": null` -- pending a person's ruling, with
nothing filed. Upstream prose should not claim a report that does not exist. Say
what is true: these are defects in HIP RT and in the Orochi copy it vendors, not
fixed at either project's HEAD, carried here as a patch until they are fixed
there.

### 3. The pattern-3 sweep result is recorded as an absence claim, and the absence is not what you checked

Round 5, "The sweep, this time in three patterns", item 3: "HIP RT's own kernels
use `WarpSize` (the real wavefront width, `hiprt_common.h:202`) throughout and
never carve a wave into logical halves."

The conclusion is right -- I re-derived it -- but the sentence as written will
mislead the next reader, because HIP RT does carve wavefronts into fixed logical
sublane groups, in four places, and a reader who trusts "never carves" will skip
exactly the sites that have to be checked:

- `hiprt/impl/BvhBuilderKernels.h:141-145`, `:706-710`, `:1100-1106`:
  `sublaneIndex = laneIndex % BranchingFactor`,
  `subwarpIndex = laneIndex / BranchingFactor`, `subwarpMask` over the subgroup.
- `hiprt/impl/BvhBuilderKernels.h:1789-1790`:
  `subwarpIndex = laneIndex / LanesPerLeafPacketTask`,
  `sublaneIndex = laneIndex % LanesPerLeafPacketTask` (`LanesPerLeafPacketTask`
  is 4, `BvhConfig.h:37`).

What makes them correct is a positive property, and it is worth recording because
it is the pattern to look for elsewhere: the carve is relative to the real
`WarpSize` (`laneIndex = threadIdx.x % WarpSize`), the subgroup mask is built in
64 bits (`( 1ull << BranchingFactor ) - 1ull` shifted by the subgroup base, after
the patch), a wave-absolute ballot result is reduced back to a subgroup index
before it is compared against a subgroup index (`:168-169`,
`maxIndex = maxLaneIndex % BranchingFactor`), and `PackLeavesWarp` already does
exactly what `logicalWarpBallot` now does -- `hiprt::ballot( ... ) >>
( LanesPerLeafPacketTask * subwarpIndex )` at `:1844`, over a `uint64_t`.

Replace the absence claim with that. This is the third round in which a sweep's
stated rationale has been weaker than its conclusion; "I found no occurrences" and
"the occurrences I found handle it correctly, here is how" are different records,
and only the second one survives the next reader.

### Verified independently

Each round-5 claim re-checked against the code and the hardware, not against the
round-5 write-up.

1. **`logicalWarpBallot` is correct on both wavefront widths.**
   `RadixSortKernels.h:410-418` in the patched tree. `( threadIdx.x % warpSize )
   / WARP_SIZE` is the logical warp's index within the physical wave: the block
   is 1-D and 256 threads (`REORDER_NUMBER_OF_THREADS_PER_BLOCK`,
   `RadixSortConfigs.h:54`; the fault dump below confirms `workgroup=[256,1,1]`),
   so `threadIdx.x % warpSize` is the physical lane and the quotient is 0 or 1 on
   wave64. Shift 0 or 32 of the `unsigned long long` `__ballot` result, narrowed
   after the shift, so logical warp 1 gets bits 32..63 as its bits 0..31. On
   wave32 the quotient is always 0, the shift is 0, and the emitted code is
   unchanged. The `ITS` branch is `__ballot_sync( 0xFFFFFFFF, predicate )`, byte
   for byte what both original call sites expanded to; `ITS` is defined only for
   `CUDART_VERSION >= 9000` (`RadixSortKernels.h:26-28`), so the CUDA path is
   untouched. Both sites converted (`:480`, `:485`); no third ballot in either
   staged subtree. The collapsed blocks are semantically unchanged, including the
   second site's `( 0xFFFFFFFF * bit ) ^ ...`, which previously XORed in 64 bits
   and truncated on assignment to a `u32` and now XORs in 32.
   Consistency with the rest of the loop holds: `lowerMask` (`:489`), `__popc`,
   `__ffs` and `if( lane == leaderIdx )` are all logical-lane relative, and there
   is no `__shfl` broadcast that would need a physical lane. Phases 2 and 3
   (`:634-680`) use `warp`/`lane` only to address shared memory, and
   `scanExclusive` (`:309-345`) and `ldsScanExclusive` are LDS plus
   `__syncthreads` with no warp collective at all.
2. **The three-pattern sweep holds, including under spellings it did not name.**
   Pattern 2 run verbatim over `hiprt/` and `contrib/Orochi/ParallelPrimitives`
   returns exactly nine hits, all `__popcll` counts or `__ffsll` indices
   (`PlocBuilderKernels.h:195,200`; `hiprt_device_impl.h:147,186`;
   `BvhBuilderKernels.h:402,726,837,869,1141`) -- a count or an index over 64
   lanes fits in 32 bits, so all nine are correct. Every mask-shaped destination
   in `hiprt/` is `uint64_t` (`BvhBuilderUtil.h:167,292`,
   `PlocBuilderKernels.h:282`, `BvhBuilderKernels.h:144,380,709,1104,1192,1239,1982`).
   Pattern 1 leaves `RadixSortKernels.h:485` benign as described, and the
   `1 << j` sites in `BvhBuilderKernels.h:612-625,1549-1571,1638-1695` are vertex
   masks with `j < 4`, not lanes. Pattern 3: `:448` is the only carve in either
   subtree, and `BvhNode.h:713-789` is word/bit-offset packing into `m_data`
   (`m_data[loWord]`), not lanes. I also swept the spellings the three patterns
   do not cover -- `>> 5`, `& 31`, `& 0x1f`, `% 32u`, `/ 32u` -- and the only
   hits are `BvhNode.h:1087`, `Triangle.h:93` and `Obb.h:79`, none lane-derived.
   `hiprt/` and `contrib/Orochi/ParallelPrimitives` are indeed the whole staged
   set (`setup.py:92`, `:150-161`; the installed `hiprt_root` contains nothing
   else).
3. **50/50 reproduced, and the negative control re-run rather than trusted.**
   Full harness at `f35600e` on this host, cache cleared first: 50/50, and every
   number identical to the record (0.241, [1.820, 4.574], 62/64 missing [5, 25],
   cosine 0.9825, fd 1.0000/1.0021, 0.9999/0.7428, 0.9993/0.9387, 0.9923/0.8856,
   bounce 3.627e-01; P=4096 0.19876/0.386/1343, P=16384 0.25417/0.457/3309).
   My own negative control: staged `RadixSortKernels.h` replaced with the blob
   from the pinned tag, cache cleared, scale cases only ->
   `Memory Fault Error [... kernel: OnesweepReorderKeyPair64]`, exit by core
   dump; patched file restored, cache cleared, scale cases pass again. So the
   check is demonstrably sensitive to this hunk, which is the property that makes
   "the harness survived" a real assertion at this size rather than a tautology.
   `check_scale` deliberately not using completeness is right and the reasoning
   is sound -- at P=2048 a correct sort traces 1047/2048 into the same slab.
4. **The patch reproduces the build tree.** Fresh clone of
   `3.1.0.cb09c56` (`8602b8c475255fb922c2792654aae0a6bcdeb0af`), `git apply` of
   the committed patch: applies clean, 12 files, 133 insertions, 49 deletions.
   `diff -r` against the build clone at `/var/lib/jenkins/HIPRT` is empty except
   HIP RT's own gitignored generated `hiprt/hiprt.h` and `hiprt/hiprtew.h`, and
   the installed `diff_surfel_tracing/hiprt_root` matches it too, so the runtime
   source that produced the 50/50 is the source the patch describes.
5. **The Orochi attribution is exact.** The blob at the pinned tag is
   `3fe37293fb4255b190ec21099bd63b0351c71f8b`, and the GitHub contents API
   returns the same SHA for `Orochi:ParallelPrimitives/RadixSortKernels.h` and
   for `HIPRT:contrib/Orochi/ParallelPrimitives/RadixSortKernels.h` at their
   current HEADs (Orochi `78fb3df`, 2026-08-13; HIP RT `e3c01fc`). Byte-identical
   at all three points, so the defect is live upstream in Orochi and the report
   belongs there, exactly as the deferral states.
6. **The lesson is accurate and portable.** `fault-classes.md:86-114`. The three
   patterns are as described, the "vendored `contrib/` subtrees are staged and
   runtime-compiled like the dependency's own headers" point generalizes, and the
   reachability trap ("if a dependency picks kernels by problem size, one test
   case has to cross every threshold") is the reusable half. The cross-reference
   to the cuSZ entry at `:130-133` is correct and the two fix forms agree.
   The round-4 false all-clear is corrected in place at `notes.md:834-843`, as an
   inserted correction rather than a rewrite, which preserves the record.
7. **Hygiene.** `jargon.py --port` clean; `prose.py` clean on the `f35600e` body;
   the fork tree is clean at `f35600e` and matches `origin/moat-port`; all eight
   titles are `[ROCm]`-prefixed and 47-61 characters; no `Co-Authored-By`, no
   agent or vendor account references, no host names, ASCII throughout, AI
   assistance disclosed. The CUDA path over the whole branch is still only
   `optix_tracer/auxiliary.h` (+19/-1) and `optix_tracer/params.h` (+21), with
   `CMake/`, `CMakeLists.txt` and `optix_tracer/*.cu` untouched.

## Port round 6 (2026-08-14, linux-gfx90a, MI250X, ROCm 7.14) -- REVIEW FINDINGS FIXED

All three round-5 findings are addressed. No device or build code changed this
round: the port's behaviour at `f35600e` is the port's behaviour at
`1f59cef`, re-measured rather than carried (50/50 below).

New fork history. The eight existing commits are message-only rewrites, so
every sha moved; two new commits carry the content changes:

| was | now | title |
|---|---|---|
| `1afb40b` | `72c6f82` | `[ROCm] Fix undefined behavior in quat_to_rotmat_transpose` |
| `6847672` | `02ac712` | `[ROCm] Add a HIP RT trace back end for AMD GPUs` |
| `db2f6ba` | `33f63f3` | `[ROCm] Guard the CUDA-only vector operators in auxiliary.h` |
| `fa7df21` | `f038d98` | `[ROCm] Fix a 64-lane wavefront hang in the HIP RT build` |
| `0eab27b` | `e796a26` | `[ROCm] Fix two more 32-bit lane masks in the HIP RT patch` |
| `c086cde` | `e63c5ee` | `[ROCm] Stage only the HIP RT files the runtime compiler reads` |
| `5254aa6` | `062f13f` | `[ROCm] Drop stale pointers and author lines from the back end` |
| `f35600e` | `b1d9838` | `[ROCm] Fix a truncated wavefront ballot in the radix sort` |
| -- | `f908248` | `[ROCm] Add a validation script for the AMD back end` |
| -- | `1f59cef` | `[ROCm] Correct the README note on the HIP RT patch` (`head_sha`) |

The rewrite is message-only and confirmed by tree hash, not by inspection:
all ten commit trees are pairwise identical to their pre-rewrite counterparts
and `git diff` between the two tips is empty. No platform held a
`validated_sha` (all three are `null` in `status.json`), so nothing was
orphaned. The pre-rewrite tip is preserved locally on this host as
`refs/moat/pre-round6-rewrite` = `ff2c14b`, which is also where
`git filter-branch` left `refs/original/refs/heads/moat-port`.

### Finding 1: the harness ships in the fork, and the Test Plans name it

`projects/diff-surfel-tracing/validation/validate_tracer_rocm.py` is now
`example/validate_rocm.py` in the fork. **Moved, not copied** -- the MOAT copy
and the `validation/` directory are deleted, so there is one file and it is the
one the gate runs. `plan.md`, `surface.json` and the run command in the test
plan point at the fork path.

Placement and name follow the ruling and the repository's own shape: there is
no `tests/` directory and no test of any kind, and the only standalone runnable
script upstream ships is `example/render.py`, so the second standalone script
goes beside it. `validate_rocm.py` rather than `validate_tracer_rocm.py`
because it asserts `torch.version.hip` -- it checks the AMD back end, not the
tracer in general. Only the docstring's `Run:` line changed inside the file.
The README's AMD section now ends with a paragraph pointing at it and the
command to run it, in the section's own descriptive style.

Seven commit bodies promised `python3 validate_tracer_rocm.py`; all seven now
say `python3 example/validate_rocm.py`, which is a real path in the branch a
maintainer clones. `f038d98` never carried the command and is unchanged.

Two upstream-visible claims of a report that has not been filed went with the
same rewrite, since finding 2's reasoning is not specific to the README:
`02ac712` said the HIP RT fixes "are being contributed to HIP RT separately"
(now "are tracked for reporting to the projects that own them"), and `b1d9838`
said the Orochi defect "is reported to a different project" (now "belongs to a
different project").

### Finding 2: the README says what is true about the four patch entries

`README.md:57-58`, in the AMD install block, was "they are HIP RT bugs and are
being contributed upstream". Both halves were wrong after round 5's
reclassification. It now reads that three are bugs in HIP RT itself and one is
in the copy of Orochi it vendors, that none is fixed at either project's HEAD,
and that they are tracked for upstream reporting -- the same distinction the
patch header already draws. That is `1f59cef`, docs only.

### Finding 3: the pattern-3 sweep records the positive property

Corrected in place at round 5's "The sweep, this time in three patterns", item
3, as an inserted correction rather than a rewrite, the way round 5 corrected
round 4. The absence claim ("never carve a wave into logical halves") is
replaced by the four places HIP RT does carve one --
`BvhBuilderKernels.h:141-145`, `:706-710`, `:1100-1106`, `:1789-1790` -- and by
what makes them correct: a `WarpSize`-relative carve, 64-bit subgroup masks,
a wave-absolute ballot index reduced with `% BranchingFactor` at `:169` before
it is compared against a subgroup index, and `PackLeavesWarp:1844` already
shifting a `uint64_t` ballot by `LanesPerLeafPacketTask * subwarpIndex`, which
is the same shape `logicalWarpBallot` now uses. Every line cited was re-checked
against the pinned tag on this host rather than transcribed from the review
(`WarpSize` is at `hiprt_common.h:204`/`:206`, not `:202`).

### Full suite at the new location: 50/50

Run as shipped, from the fork checkout, with the path a maintainer would use.
No reinstall: no source the extension compiles changed this round, and the
harness's own cold-cache check recompiles every kernel at the end regardless.

```
PYTORCH_ROCM_ARCH=gfx90a, ROCm 7.14.60850, torch 2.14.0a0+git7d05abc
cd projects/diff-surfel-tracing/src
HIP_VISIBLE_DEVICES=0 python3 example/validate_rocm.py
50/50 checks passed
```

Every number identical to the round-5 record: covered fraction 0.241, depth
range [1.820, 4.574], 62/64 traced missing [5, 25], median `grads3D` cosine
0.9825, fd colors 1.0000/1.0021, opacities 0.9999/0.7428, means3D
0.9993/0.9387, scales 0.9923/0.8856, bounce 3.627e-01, P=4096 0.19876/0.386/1343,
P=16384 0.25417/0.457/3309.

`1f59cef`'s own Test Plan was run too: a fresh clone of `3.1.0.cb09c56`
(`8602b8c`) takes `git apply --check --stat` of the committed patch cleanly,
12 files, 133 insertions, 49 deletions.

### Gotchas

- **`git mv` into `projects/<name>/src/` stages the destination even though
  `.gitignore` has `projects/*/src/`.** `git mv` adds the target path
  explicitly, which overrides the ignore rule, so a move from MOAT into the
  fork checkout silently starts tracking the fork file in MOAT as well --
  exactly the two-copy outcome the ruling was avoiding. `git rm --cached` the
  destination, then commit the deletion on the MOAT side and the addition
  inside the fork clone as two separate repositories' commits.
- **A message-only rewrite is provable, so prove it.** `git filter-branch -f
  --msg-filter <script> <base>..moat-port` plus a pairwise comparison of
  `<commit>^{tree}` before and after is a two-line check that turns "I only
  edited messages" into evidence a reviewer can re-run. Take a named ref at the
  old tip first (`git update-ref refs/moat/pre-<round>`), because
  `refs/original/` is expendable and `--force-with-lease` wants the old sha
  anyway.

## Review 2026-08-14 (round 6, linux-gfx90a, `moat-port` 1f59cef vs f35600e) -- CHANGES REQUESTED

Problems only. All three round-5 findings are genuinely closed: the harness is a
single file in the fork, the rewrite is provably message-only, and the pattern-3
correction is accurate (list at the end). No device or build code moved this
round and none of the round-5 verification is disturbed.

The findings below are all in text that round 6 itself wrote, and all three are
the same failure the round-5 review was about: an upstream-visible sentence that
states more than what is true. Two of them replaced one inaccuracy with another,
which is why they come back rather than being let go. The third is a wave64
caveat that no round has recorded and that this project's own gate structurally
cannot see.

### 1. The README now says the patch has four entries. It has eleven.

`README.md:57-60` in the fork, upstream-visible, introduced by `1f59cef`:

```
# Build HIP RT, applying the fixes it still needs (see the patch header for what
# each one is for; three are bugs in HIP RT itself and one is in the copy of
# Orochi it vendors, none of them is fixed at either project's HEAD, and they
# are tracked for upstream reporting)
```

"see the patch header for what each one is for" scopes the sentence to every
entry in the patch, and "three ... and one ..." then counts four of them.
`third_party/hiprt-rocm-fixes.patch` has eleven entries in its header table over
twelve files (`grep -n '^--- a/'`), and its own text says so two paragraphs
above: "The first four entries are one defect in four shapes". The seven the
README's count leaves out are `Compiler.cpp getCacheFilename`, `addCommonOpts`
and `buildProgram`, the `Compiler.{h,cpp}`/`Context.cpp` init split, the
`hiprt.cpp`/`hiprt_libpath.h`/`Orochi hipew.cpp` library-path fix, `Orochi.cpp
oroSetRawDevice`, and the `MemoryArena.h`/`BvhNode.h`/`hiprt_device_impl.h`
warning fixes. `notes.md:185` records the same thing from the other side: "There
are ten patched HIP RT files, not four."

The Orochi half is wrong by the same count. Three of the twelve patched files
are under `contrib/Orochi/` -- `ParallelPrimitives/RadixSortKernels.h`,
`Orochi/Orochi.cpp` and `contrib/hipew/src/hipew.cpp` -- not one.

`1f59cef`'s own body carries the error into the commit log: "Three of the four
are in HIP RT's own sources and the fourth is in the copy of Orochi that HIP RT
vendors", introduced by the sentence before it, which scopes itself to "the
entries in third_party/hiprt-rocm-fixes.patch".

This is round 5's finding 2 in a new shape, and the round-5 write-up seeded it by
saying "all four patch entries" -- that phrasing was wrong too and should not
have been taken as the specification. The fix is to drop the count and say what
the patch header already says, which is true of all eleven: they are bugs in HIP
RT and in the copy of Orochi it vendors, none is fixed at either project's HEAD,
and they are tracked for upstream reporting. Verified unfixed at HEAD for the
four lane-mask ones (below); if the reworded sentence is to cover all eleven at
HEAD, either check the other seven or scope the HEAD clause to the ones that
were checked.

### 2. "all eleven backward gradients" -- the harness checks seven of them

Three upstream-visible copies of the same overclaim, all shipped this round:

- `example/validate_rocm.py:5` (the module docstring, now in the fork)
- `README.md:80` (the new AMD paragraph, `f908248`)
- `f908248`'s commit body, second paragraph

`_C.trace_surfels_backward` returns eleven tensors -- `grad_ray_o`, `grad_ray_d`,
`grad_means3D`, `grad_grads3D`, `grad_shs`, `grad_colors_precomp`,
`grad_others_precomp`, `grad_opacities`, `grad_scales`, `grad_rotations`,
`grad_cov3Ds_precomp` (`diff_surfel_tracing/__init__.py:197`). `check_backward`
iterates the `inputs` dict built in `trace()` (`validate_rocm.py:158-160`), which
holds seven: `means3D`, `grads3D`, `scales`, `rotations`, `opacities`, `colors`,
`others`. The other four are not examined and cannot be, in this configuration:
`ray_o`/`ray_d` are built without `requires_grad` (`make_rays`, `:113-121`), and
`shs`/`cov3D_precomp` are the mutually exclusive alternatives to
`colors_precomp` and the scale/rotation pair, so `SurfelTracer.forward`
(`:291-305`) substitutes empty dummy tensors and raises if you pass both. The
run's own output is the plainest refutation: seven `backward grad_* finite`
lines, not eleven.

Two acceptable fixes. Reword to what is checked -- "every gradient this scene
exercises (seven of the tracer's eleven; `shs` and `cov3D_precomp` are the
alternative parameterization and `ray_o`/`ray_d` are not differentiated here)"
-- or raise the count honestly by setting `requires_grad` on `ray_o`/`ray_d`,
which makes it nine and costs three lines. `shs` and `cov3D_precomp` would need
a second scene; that is a real gap but not one this finding asks you to close.
`surface.json:139` and `notes.md:173` carry the same "all eleven" wording and
should move with it, so the record and the shipped text agree.

### 3. HIP RT's `WarpSize` is an architecture allowlist, and nothing records that

`hiprt/hiprt_common.h:202-206` at the pinned tag:

```
#if __gfx900__ || __gfx902__ || __gfx904__ || __gfx906__ || __gfx908__ || __gfx909__ || __gfx90a__ || __gfx90c__ || \
	__gfx940__ || __gfx941__ || __gfx942__
constexpr uint32_t WarpSize = 64;
#else
constexpr uint32_t WarpSize		   = 32;
#endif
```

Every wave64 argument this port makes rests on `WarpSize` being the real
wavefront width -- the three `subwarpMask` fixes, the `PairTriangles` and
`PackLeavesWarp` fixes, `logicalWarpBallot`, and round 5's item 3 ("the carve is
relative to the real `WarpSize`"). It is the real width only for the eleven
architectures named on those two lines. `gfx950` is not among them, so on an
MI350 the runtime compiler (`--offload-arch=<gcnArch>`, from this port's own
patch) builds HIP RT's kernels with `WarpSize = 32` against 64-lane hardware:
`laneIndex = threadIdx.x % WarpSize` stops being the lane, and every subgroup
carve and ballot reduction downstream is wrong. I checked HIP RT HEAD by API --
`hiprt/hiprt_common.h` at HEAD has the identical list -- so this is live
upstream, not staleness in the pinned tag. It is textbook hardcoded-wavefront:
the durable fix is `__AMDGCN_WAVEFRONT_SIZE__` rather than an enumerated list.

The gate cannot see this: gfx90a and gfx942 are both on the allowlist, so wave64
coverage passes while the exposure is invisible, and the README tells a user to
set `PYTORCH_ROCM_ARCH` to whatever their GPU is.

What is required is the record, not a patch hunk: do NOT add `__gfx950__` to a
twelve-file patch you cannot validate on any host here. Register it as a
`rocm-bug-report` deferral (`utils/deferred.py add`) so it joins the GPUOpen
report the other HIP RT defects are already queued for, and put one paragraph in
notes stating that the port's wave64 correctness is conditional on that
allowlist. Whether the README's AMD section should carry an architecture caveat
is your call; the deferral and the note are not.

### Verified independently

Re-checked against the code and this host, not against the round-6 write-up.

1. **One harness, and it is the one that ran.** `git show 7ffc729^:.../validation/validate_tracer_rocm.py` diffed against
   `example/validate_rocm.py` on the fork branch differs in exactly the docstring
   `Run:` lines (2 lines). MOAT tracks no copy: `git ls-files
   projects/diff-surfel-tracing` returns the six record files only, the
   `validation/` directory is gone from the working tree, and no
   `validate_tracer_rocm` string survives anywhere in the fork. `plan.md:286`,
   `:325`, `:353`, `surface.json:116`, `:139` and `notes.md:14-19` all point at
   the fork path; the pre-round-6 mentions further down the notes are the record
   of what those rounds did and are correctly left alone.
2. **50/50 reproduced from a fork checkout, as shipped.** `cd
   projects/diff-surfel-tracing/src && HIP_VISIBLE_DEVICES=0 python3
   example/validate_rocm.py`, no reinstall, no cache priming: `50/50 checks
   passed`, exit 0, on MI250X / ROCm 7.14.60850 / torch 2.14.0a0+git7d05abc.
   Every number identical to the round-6 record (0.241, [1.820, 4.574], 62/64
   missing [5, 25], cosine 0.9825, 1.0000/1.0021, 0.9999/0.7428, 0.9993/0.9387,
   0.9923/0.8856, 3.627e-01, P=4096 0.19876/0.386/1343, P=16384
   0.25417/0.457/3309). Running it as `example/validate_rocm.py` puts `example/`
   on `sys.path[0]`, not the repository root, so `import diff_surfel_tracing`
   still resolves to the installed package and not to the source directory next
   to it -- the relocation did not change what gets imported.
3. **The rewrite is message-only, provably.** All ten commits pairwise
   tree-identical between `refs/moat/pre-round6-rewrite` (`ff2c14b`) and
   `1f59cef` (`ef6f24b..` in reverse order, `<sha>^{tree}` compared), `git diff
   ff2c14b 1f59cef` empty. Message diffs are exactly the seven
   `validate_tracer_rocm.py` -> `example/validate_rocm.py` substitutions plus the
   two claim fixes in `02ac712` and `b1d9838`, and nothing else; `f038d98` and
   the two new commits are untouched. All three platforms hold
   `validated_sha: null`, so nothing was orphaned. The push flag itself is not
   recoverable from the repository (a reflog records "update by push", not the
   options), but the outcome is right: `origin/moat-port == moat-port ==
   1f59cef`, the old tip is preserved at two refs, and no upstream PR exists for
   the fork's pre-push hook to defend.
4. **The two commit-body claim fixes are correct.** `02ac712` now says the HIP RT
   fixes "are tracked for reporting to the projects that own them" and `b1d9838`
   that the Orochi defect "belongs to a different project" -- neither asserts a
   report that has not been filed. `deferred.json` backs "tracked": four entries,
   `hiprt-jit-and-cache-fixes` and `hiprt-collapse-reference-count-hang` ruled
   `now`, the two lane-mask ones open, all four `upstream_issue: null`.
5. **The four lane-mask defects really are unfixed at HEAD.** Fetched
   `hiprt/impl/BvhBuilderKernels.h` at HIP RT HEAD (blob
   `64cc5d6c690323cb215716e35f84540b2b461165`): `( 1 << BranchingFactor )` at
   `:144`, `:709`, `:1104`, `~( 1u << firstPairedLane )` at `:405`, `1 <<
   broadcastLane` at `:1989`/`:1993` -- all present. With round 5's byte-identical
   check on the Orochi blob at both HEADs, the README's HEAD clause holds for the
   four it means, whatever happens to its count.
6. **The pattern-3 correction is accurate.** `WarpSize` is at
   `hiprt/hiprt_common.h:204` (64) and `:206` (32), so round 6's correction of
   round 5's `:202` is right, and it is `hiprt/hiprt_common.h`, not
   `hiprt/impl/`. The four carve sites and the positive property they turn on are
   as recorded.
7. **The promoted lesson is accurate and portable.**
   `validation.md:110-114`. "Ship the harness with the port when it is one
   standalone file importing only the standard library, the framework the project
   already depends on, and the project itself" generalizes cleanly, the
   one-copy-not-two rule and the `agent_space/` warning are the durable half, and
   the `git mv` gotcha it leans on is recorded at `notes.md:1639-1645`.
8. **Hygiene.** `jargon.py --port`, `--commits ef6f24b..moat-port` and `--diff
   ef6f24b..moat-port` all clean; `prose.py` clean on both new bodies; `check.py`
   clean. The fork tree is clean and matches `origin/moat-port`. All ten titles
   are `[ROCm]`-prefixed and 47-61 characters, all ten bodies disclose AI
   assistance and carry a `Test Plan:`, none carries `Co-Authored-By`,
   `Signed-off-by` or a ghstack marker, and there are no agent or vendor account
   references. `example/validate_rocm.py` is pure ASCII and free of MOAT
   vocabulary, which it has to be now that it ships.

Not findings, recorded so the next round does not relitigate them: the Test Plans
in commits 1-8 name `example/validate_rocm.py`, which only exists from `f908248`
onward, so an intermediate checkout cannot run them. A PR series is tested at its
tip and reordering would move every sha again for no reader benefit. And
`f908248`'s "the two surfels it tolerates as untraced" describes the two that are
untraced, not the margin of four (`validate_rocm.py:223`); the code comment
explains the margin correctly.
