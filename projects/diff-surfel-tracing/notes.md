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
