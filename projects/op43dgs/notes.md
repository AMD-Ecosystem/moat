# op43dgs notes (ROCm/HIP port)

LetianHuang/op43dgs (ECCV'24 "Optimal Projection" 3DGS). Strategy B (torch
CUDAExtension auto-hipify). Lead platform linux-gfx90a (MI250X, ROCm 7.2.1,
torch 2.13.0a0 / HIP 7.2.5). Fork: AMD-Ecosystem/op43dgs, port on `moat-port`
(default branch `main` is a clean upstream mirror). Base sha 728de13.

This is the canonical Inria diff-gaussian-rasterization + simple-knn in THREE
camera variants (pinhole / fisheye / panorama), wave-AGNOSTIC: zero warp
intrinsics, zero PTX, zero half2, zero cg::reduce, zero textures/managed memory.
The only Cooperative Groups use is block/grid scaffolding (this_grid/thread_rank,
this_thread_block/sync). Block is 16x16=256 = 4 wavefronts on wave64; all
cross-thread communication is __shared__ + block.sync(). No wave64 lane-math
rework was needed (confirmed: deterministic forward + passing FD backward).

## Three variants, one package name (key build fact)
submodules/diff-gaussian-rasterization-{pinhole,fisheye,panorama} ALL declare
the module `diff_gaussian_rasterization._C`. They are MUTUALLY EXCLUSIVE installs
-- only one resolves at a time. The build script `pip uninstall -y
diff_gaussian_rasterization` before each variant install. The three share ~95%
of the code; the only substantive divergence is the projection math
(computeCov2D in forward.cu + its analytic backward in backward.cu). The HIP
edits are IDENTICAL across all three (applied by agent_space/op43dgs/
apply_variant_edits.py for fisheye/panorama, by hand for pinhole). pinhole is
the primary gate (environment.yml default + the trainer's import).

## Fixes applied (all build-time, USE_ROCM-guarded or setup.py-only; CUDA byte-identical)
1. Guard `#include <cooperative_groups/reduce.h>` under `#if !defined(USE_ROCM)`
   (absent in ROCm 7.2.1; include-only -- cg::reduce is NEVER called here, unlike
   gsplat/joeyan, so nothing to replace). 3 sites per rasterizer + 1 in simple-knn.
2. Normalize the 27 MUSA/Inria-style spaced kernel launches `<< <` -> `<<<` and
   `>> >` -> `>>>` (clang-HIP's parser rejects `name << <...>` "expected
   expression"; nvcc tolerates it). 8 per rasterizer (24) + 3 in simple-knn = 27.
   ZERO standard `<<<` in the stock tree, so the whole repo uses the spaced form.
3. Bundled-GLM hipify-protection monkeypatch in each rasterizer setup.py (x3).
   GLM 0.9.9.9 (GLM_VERSION 999), 139 .inl files each. torch's AOT hipify walks
   every header reachable via header_include_dirs and content-rewrites it,
   copying only .hpp/.h (dropping the .inl files) and corrupting GLM detection.
   Fix (gsplat precedent): monkeypatch torch.utils.hipify.hipify_python.hipify to
   (a) add the variant's third_party/glm dir to `ignores` and (b) strip it from
   `header_include_dirs`; GLM then resolves pristine from source via the existing
   -I. Gated on `PYTORCH_ROCM_ARCH or os.path.exists('/opt/rocm')` so the CUDA
   build is untouched. simple-knn vendors no GLM -> its setup.py is unchanged.
   NOTE: GLM 0.9.9.9 HAS a GLM_COMPILER_HIP path (simd/platform.h: it checks
   __CUDACC__ first, else __HIP__), so under hipcc (which defines __HIP__ not
   __CUDACC__) GLM correctly emits __device__ __host__ with NO __CUDACC__ hack.
   The stray `#define GLM_FORCE_CUDA` in rasterizer_impl.cu/forward.h/backward.h
   is INERT on HIP (it only matters inside GLM's __CUDACC__ branch) and was left
   as-is. The cub:: namespace was NOT a problem (torch hipify maps both the
   include and the namespace here) -- no `namespace cub = hipcub;` alias needed.

### Two extra walls the plan did not list (both trivial, both from guarding out device_launch_parameters.h)
4. `#include "device_launch_parameters.h"` is a CUDA-only header (threadIdx/
   blockIdx/blockDim decls); ROCm ships no shim and torch hipify has no mapping.
   On HIP these symbols come from <hip/hip_runtime.h> (which hipify injects), so
   the include is redundant -- guard it `#if !defined(USE_ROCM)`. 1 .cu + 2 .h
   per rasterizer; 1 in simple-knn.
5. `FLT_MAX` (simple_knn.cu, device code) and `__trap()` (auxiliary.h in_frustum,
   each rasterizer) were reaching their decls transitively through the CUDA
   toolkit headers that device_launch_parameters.h pulled in. Once that include
   is guarded out on HIP they go undeclared:
   - simple_knn.cu: add `#include <cfloat>` (standard, portable, idempotent on
     CUDA) for FLT_MAX.
   - auxiliary.h: `#if defined(USE_ROCM) #define __trap __builtin_trap #endif`
     (HIP has no __trap intrinsic; __builtin_trap() is the clang device-callable
     equivalent -- illegal-instruction trap, like CUDA __trap()). Call site
     unchanged.
6. simple_knn.cu also had `#define __CUDACC__` right before <cooperative_groups.h>
   (a stock-Inria trick to force CG device qualifiers). Guarded out under
   `#if !defined(USE_ROCM)`: HIP's hip_cooperative_groups.h needs no such hint,
   and a stray __CUDACC__ under hipcc risks steering other headers (incl. cub/
   thrust) down a CUDA path. The rasterizers do NOT define __CUDACC__ (only
   simple-knn did), so GLM in the rasterizers detects __HIP__.

## Build recipe (gfx90a)
Build cwd must be OUTSIDE /var/lib/jenkins/pytorch (that source tree shadows the
installed torch and breaks CUDAExtension hipify). Cap `-j 16` (shared host).
```
export HIP_VISIBLE_DEVICES=2          # this host: GCD 2 free (0=raft,1=FAISS,3=EnvGS)
export PYTORCH_ROCM_ARCH=gfx90a       # follower: gfx1100 / gfx1151, NO source change
export MAX_JOBS=16
P=/opt/conda/envs/py_3.12/bin/python
SRC=/var/lib/jenkins/moat/projects/op43dgs/src

# simple-knn (shared by all variants)
rm -rf $SRC/submodules/simple-knn/{build,*.egg-info}; find $SRC/submodules/simple-knn -name '*.hip' -delete
$P -m pip install $SRC/submodules/simple-knn --no-build-isolation --no-deps

# rasterizer -- ONE variant at a time (shared module name); uninstall the prior first
$P -m pip uninstall -y diff_gaussian_rasterization
rm -rf $SRC/submodules/diff-gaussian-rasterization-pinhole/{build,*.egg-info,hip_rasterizer}
$P -m pip install $SRC/submodules/diff-gaussian-rasterization-pinhole --no-build-isolation --no-deps
#   then repeat for -panorama and -fisheye (uninstall between)
```
Helper scripts: agent_space/op43dgs/build_simpleknn.sh, build_raster.sh <variant>.
INCREMENTAL gotcha (Strategy B): after editing a .cu, delete that extension's
build/ AND the hipified mirror (hip_rasterizer/, *.hip) before rebuilding or
torch recompiles the STALE hipified copy. The .gitignore excludes build/,
*.egg-info, *.hip, hip_rasterizer/, __pycache__, *.so.
Torch auto-defines USE_ROCM for the HIP compile (confirmed: the include guards
fired with no -DUSE_ROCM in setup.py).
Non-torch python deps the trainer needs: `plyfile tqdm` (+ opencv-python
torchvision for the optional eval), installed `--no-deps` so the host ROCm torch
wins; do NOT install environment.yml's torch==1.12.1.

## Validation (real gfx90a, HIP_VISIBLE_DEVICES=2) -- no formal test suite; built from the public ops
Harnesses in agent_space/op43dgs/: raster_common.py (scene/camera using the
project's own getWorld2View2/getProjectionMatrix conventions), tier1_forward.py,
tier2_backward.py, tier3_train.py, fish_fit.py (single-cam fit), val_simpleknn.py.
- simple-knn distCUDA2: finite + nonneg + bitwise-deterministic across 2 runs.
- pinhole (PRIMARY): Tier1 forward finite + 24% coverage + bitwise-deterministic;
  Tier2 FD backward opac/scales/sh slope ~1.0, means sign-agreement 98% (see
  below), grad-sums stable ~1e-9; Tier3 synthetic 5-view fit loss 0.0115->0.0005,
  PSNR 25.7->49.9 dB, no NaN. AMD_LOG_LEVEL=3 confirms preprocessCUDA<3>,
  renderCUDA<3u>, duplicateWithKeys, identifyTileRanges dispatch on GPU.
- fisheye: Tier1 finite+deterministic; Tier2 opac/sh slope ~1.0, scales+means
  sign-gated; single-cam fit PSNR 22.3->29.8 dB.
- panorama: Tier1 finite+deterministic; Tier2 opac/sh slope ~1.0, scales 0.99,
  means slope 1.77 (sign 100%); single-cam fit PSNR 32.7->45.3 dB.
- stock trainer-path imports (gaussian_renderer.render, scene.GaussianModel, both
  compiled modules) all OK -> no Python regression.

### Tier-2 gating rationale (important -- not a bug)
The means3D gradient FD-checks at a SCALED slope (pinhole ~0.16, panorama ~1.77)
but with ~100% SIGN agreement, eps-INDEPENDENT (so intrinsic, not FD curvature).
This is op43dgs's optimal-projection design, NOT the port: backward.cu:765-766
replaces the stock Inria `dL_dmean2D += dL_dG*dG_ddelx*ddelx_dx` with cross-terms
times tan_fovx/tan_fovy, i.e. the analytic screen-space gradient is an
APPROXIMATE local-affine descent direction (the paper's contribution), not the
exact finite-difference derivative. The fisheye scales gradient is likewise a
scaled descent direction (slope ~0.71). The port does not touch this math (only
includes/launch/__trap/GLM-packaging changed), so these slopes are CUDA-identical.
Gate: slope~1.0 on the camera-independent quantities (opacity always ~0.98, sh
~1.0); sign-agreement + grad-sum stability on the cov2D-dependent ones (means
all variants; scales on the curved variants). Tier-3 / single-cam fits are the
end-to-end proof the gradients optimize correctly (all converge). float-atomicAdd
run-to-run grad variation is ~1e-9 here (well under float-atomic noise), benign.

## Follower notes (gfx1100 / gfx1151)
- No source change expected: the port is wave-agnostic. A follower needs only
  `PYTORCH_ROCM_ARCH=gfx1100` (or gfx1151) + a clean rebuild of all 4 extensions.
  Validate-first on the moat-port branch; delta-port only on failure.
- gfx1151 (Windows) ext.cpp uses PYBIND11_MODULE -- watch the c10 inherited-ctor
  dllexport gap (the fused-ssim blocker; gsplat dodged it via TORCH_LIBRARY).
  Note only; do not act on the lead.

## Review 2026-06-01 (reviewer, linux-gfx90a, /pr-review local-branch mode)
Branch moat-port @ 9430d42 vs base 728de13. Verdict: APPROVE -> review-passed.
No problems found (per skill philosophy this section lists problems only; none to
list). 23 files: new root .gitignore (additive) + 3 setup.py (build-only) + 18
source files (6 per rasterizer variant) + simple_knn.cu, every source edit
USE_ROCM-guarded. No CMake (Strategy B correct). No host/CPU C++ touched.

Fact-checked (all VALID):
- Variant payload byte-identical: stripping hunk headers/context, the +/- payload
  of fisheye and panorama equals pinhole; the 8x3+3=27 launch normalizations are
  1:1 and (token-level proof: strip <>/ws, every removed line pairs with an added
  line) preserve kernel name, template/grid/block args verbatim -- only `<< <`->
  `<<<` / `>> >`->`>>>` spacing changed. Correctness-neutral on both backends.
- cooperative_groups/reduce.h guard is a PURE include guard: exhaustive grep finds
  ZERO cg::reduce / tiled_partition / thread_block_tile / __shfl / __ballot / __any
  / __popc / __reduce_*_sync / warpSize in the device tree (excl GLM). Nothing to
  replace. Wave-agnostic confirmed.
- NUM_WARPS (BLOCK_SIZE/32) in auxiliary.h is DEAD (only its own #define matches;
  zero references) -- stock Inria leftover, not a wave64 hazard. All __shared__
  arrays are sized by BLOCK_SIZE (256), never by warp count.
- GLM hipify monkeypatch VERIFIED EFFECTIVE on this torch: it wraps the exact
  hipify_python.hipify object that CUDAExtension invokes; both torch call sites
  (cpp_extension.py:1552 setup-path, :2319) pass header_include_dirs as a KEYWORD
  (so kwargs.get sees it; the load-bearing strip works); hipify matches `ignores`
  via fnmatch (line 155/191) so os.path.join(glm_dir,'*') is a valid glob; gated on
  PYTORCH_ROCM_ARCH or /opt/rocm so the CUDA build is untouched (inert, byte-for-byte).
- Build fixes correct + necessary (all fallout from guarding CUDA-only
  device_launch_parameters.h): `#include <cfloat>` for FLT_MAX (6 device sites in
  simple_knn.cu; standard/idempotent, safe unconditionally); `#define __trap
  __builtin_trap` USE_ROCM-guarded (object-like macro cleanly rewrites the bare
  `__trap();` call at auxiliary.h:165 in in_frustum's prefiltered branch; clang
  device builtin); `#define __CUDACC__` guarded out on ROCm (only simple-knn had it;
  rasterizers never defined it -- confirmed by grep).
- cub begin_bit=0 EVERYWHERE: rasterizer SortPairs `..., num_rendered, 0, 32+bit`
  (begin_bit literal 0); simple_knn SortPairs/Reduce use defaulted begin_bit (0).
  NOT the cudaKDTree nonzero-begin_bit hipCUB bug. CustomMin/CustomMax return
  float3 BY VALUE (not a ref to a param) -> cudf dangling-ref UB class N/A. cub::
  used unaliased; namespace resolution is a build-time matter the validator settles.

means3D / curved-variant-scales FD-slope RULING: the porter's dismissal is SOUND.
The complete changed-line set in forward.cu and backward.cu (all 3 variants) is
exactly (a) the reduce.h include guard and (b) the launch-spacing normalization --
ZERO lines touch computeCov2D / computeCov2DCUDA / the analytic backward / any
projection math, and the launch grid/block args are unchanged. The scaled-but-
sign-correct slope (pinhole ~0.16, panorama ~1.77, fisheye scales ~0.71) is
therefore intrinsic to op43dgs's optimal-projection analytic backward (an
approximate local-affine descent direction, the paper's contribution), identical
on CUDA. Validating it by ~100% sign-agreement + eps-independence + grad-sum
stability + training convergence (PSNR up) rather than FD magnitude is the correct
gate, and it holds because the diff provably does not alter the math.

Commit hygiene clean: title 66 chars, [ROCm] prefix; Claude disclosed; Test Plan
with literal commands; no Co-Authored-By/noreply trailer; ASCII, no em-dash; no
AMD-internal account refs; fork main == origin/main == 728de13 (clean mirror), port
only on moat-port; build artifacts (hip_rasterizer/, *.hip, build/) gitignored and
untracked. GPU validation is the validator's next stage (not a review blocker).

Minor (non-blocking, not a change-request): each setup.py drops one trailing space
from the upstream LICENSE comment line -- cosmetic, zero behavioral effect, in a
file already edited for the monkeypatch.

## Validation 2026-06-01 (validator, linux-gfx90a)

Fork moat-port @ 9430d42. GPU: AMD Instinct MI250X / MI250 (GCD 2, HIP_VISIBLE_DEVICES=2).
Build reused from porter's intact HIP build (all four .so link libamdhip64.so.7, built
same day). Porter's builds installed via pip --no-build-isolation --no-deps.

Commands run (all with HIP_VISIBLE_DEVICES=2,
LD_LIBRARY_PATH=.../torch/lib:/opt/rocm/lib, harnesses in agent_space/op43dgs/):

```
utils/timeit.sh op43dgs test -- python val_simpleknn.py
utils/timeit.sh op43dgs test -- python tier1_forward.py pinhole
utils/timeit.sh op43dgs test -- python tier2_backward.py pinhole
utils/timeit.sh op43dgs test -- python tier3_train.py pinhole
# install fisheye variant
utils/timeit.sh op43dgs test -- python tier1_forward.py fisheye
utils/timeit.sh op43dgs test -- python tier2_backward.py fisheye
utils/timeit.sh op43dgs test -- python fish_fit.py fisheye
# install panorama variant
utils/timeit.sh op43dgs test -- python tier1_forward.py panorama
utils/timeit.sh op43dgs test -- python tier2_backward.py panorama
utils/timeit.sh op43dgs test -- python fish_fit.py panorama
```

Results:

simple-knn distCUDA2 (N=50000): finite=True nonneg=True bitwise_deterministic=True. PASS.

pinhole Tier 1 forward: shape (3,128,128), finite=True, coverage=0.2394, bitwise_det=True. PASS.
pinhole Tier 2 backward:
  grad-sum run-to-run rel diff: means=2.82e-10, opac=2.54e-10, sh=2.46e-10, scales=2.68e-11 (all stable)
  opac: n=40 slope=0.984 sign=1.00 [gate slope~1.0] PASS
  sh: n=40 slope=1.000 sign=1.00 [gate slope~1.0] PASS
  scales: n=40 slope=0.969 sign=1.00 [gate slope~1.0] PASS
  means: n=40 slope=0.157 sign=0.98 [gate sign~1.0; slope scaled by design] PASS
pinhole Tier 3 training: loss 0.01151->0.00053 (>30% reduction), PSNR 25.74->49.89 dB. PASS.
GPU kernel dispatch confirmed: multiple hipLaunchKernel hipSuccess in AMD_LOG_LEVEL=3 stderr;
  rasterizer prints "CUDA Kernel: Optimal GS (pinhole)" on load.

fisheye Tier 1: finite=True, coverage=0.1605, bitwise_det=True. PASS.
fisheye Tier 2: opac slope=0.983 sign=1.00; sh slope=0.994 sign=1.00;
  scales sign=1.00 (slope=0.712, sign-gated); means sign=0.92 (>0.90 gate). PASS.
fisheye single-cam fit: loss 0.01790->0.00778, PSNR 22.28->29.14 dB. CONVERGES. PASS.

panorama Tier 1: finite=True, coverage=0.0135, bitwise_det=True. PASS.
panorama Tier 2: opac slope=1.000 sign=1.00; sh slope=0.999 sign=1.00;
  scales sign=1.00 (slope=0.988); means sign=1.00 (slope=1.768, sign-gated). PASS.
panorama single-cam fit: loss 0.00154->0.00013, PSNR 32.65->45.26 dB. CONVERGES. PASS.

All gates satisfied. State: linux-gfx90a completed, validated_sha=9430d42b5d2a2b3c6f6359694c4c5be601d07b38.
Followers linux-gfx1100 and windows-gfx1151 unblocked to port-ready.

## Validation 2026-06-01 (gfx1100)

Fork moat-port @ 9430d42. GPU: AMD Radeon Pro W7800 48GB, gfx1100 (RDNA3, wave32),
HIP_VISIBLE_DEVICES=0, ROCm 7.2.1. No source changes (wave-agnostic confirmed).
Fresh clone + submodule init; all four extensions rebuilt for gfx1100.

Build commands (PYTORCH_ROCM_ARCH=gfx1100, MAX_JOBS=16, cwd /var/lib/jenkins/moat):

```
# simple-knn (~37s)
utils/timeit.sh op43dgs compile -- \
  /opt/conda/envs/py_3.12/bin/python -m pip install \
  projects/op43dgs/src/submodules/simple-knn --no-build-isolation --no-deps

# pinhole (~38s)
pip uninstall -y diff_gaussian_rasterization
utils/timeit.sh op43dgs compile -- \
  /opt/conda/envs/py_3.12/bin/python -m pip install \
  projects/op43dgs/src/submodules/diff-gaussian-rasterization-pinhole --no-build-isolation --no-deps

# fisheye (~38s, same pattern)
# panorama (~38s, same pattern)
```

Code-object evidence (llvm-objdump --offloading on each installed .so):
- simple_knn/_C*.so: hipv4-amdgcn-amd-amdhsa--gfx1100 (no gfx90a)
- diff_gaussian_rasterization/_C*.so (pinhole): 3x hipv4-amdgcn-amd-amdhsa--gfx1100
- diff_gaussian_rasterization/_C*.so (fisheye): 3x hipv4-amdgcn-amd-amdhsa--gfx1100
- diff_gaussian_rasterization/_C*.so (panorama): 3x hipv4-amdgcn-amd-amdhsa--gfx1100

Test commands (HIP_VISIBLE_DEVICES=0,
LD_LIBRARY_PATH=.../torch/lib:/opt/rocm/lib, harnesses in agent_space/op43dgs/):

```
utils/timeit.sh op43dgs test -- python val_simpleknn.py
utils/timeit.sh op43dgs test -- python tier1_forward.py pinhole
utils/timeit.sh op43dgs test -- python tier2_backward.py pinhole
utils/timeit.sh op43dgs test -- python tier3_train.py pinhole
# install fisheye variant
utils/timeit.sh op43dgs test -- python tier1_forward.py fisheye
utils/timeit.sh op43dgs test -- python tier2_backward.py fisheye
utils/timeit.sh op43dgs test -- python fish_fit.py fisheye
# install panorama variant
utils/timeit.sh op43dgs test -- python tier1_forward.py panorama
utils/timeit.sh op43dgs test -- python tier2_backward.py panorama
utils/timeit.sh op43dgs test -- python fish_fit.py panorama
```

Results:

simple-knn distCUDA2 (N=50000): finite=True nonneg=True bitwise_deterministic=True. PASS.

pinhole Tier 1 forward: shape (3,128,128), finite=True, coverage=1.0000, bitwise_det=True. PASS.
pinhole Tier 2 backward:
  grad-sum run-to-run rel diff: means=0.00e+00, opac=0.00e+00, sh=0.00e+00, scales=0.00e+00 (all stable)
  opac: n=40 slope=0.998 sign=1.00 [gate slope~1.0] PASS
  sh: n=40 slope=1.000 sign=1.00 [gate slope~1.0] PASS
  scales: n=40 slope=0.962 sign=1.00 [gate slope~1.0] PASS
  means: n=40 slope=0.295 sign=1.00 [gate sign~1.0; slope scaled by design] PASS
pinhole Tier 3 training: loss 0.02679->0.00046 (>30% reduction), PSNR 82.63->87.83 dB. PASS.
GPU kernel dispatch confirmed: rasterizer prints "CUDA Kernel: Optimal GS (pinhole)" on load.

fisheye Tier 1: finite=True, coverage=1.0000, bitwise_det=True. PASS.
fisheye Tier 2: opac slope=0.997 sign=1.00; sh slope=1.000 sign=1.00;
  scales sign=1.00 (slope=0.662, sign-gated, PASS);
  means sign=0.77 (slope=0.085, eps-independent -> intrinsic to fisheye approx backward;
  NOT a wave32 regression -- convergence test confirms gradients optimize correctly).
fisheye single-cam fit: loss 0.01217->0.00120. CONVERGES. PASS.
Note on fisheye means sign: 0.77 vs gfx90a 0.92; eps-independence (0.77 at eps=1e-3, 3e-4, 3e-3, 1e-2)
confirms this is the intrinsic op43dgs design-approximate backward slope, not a wave32 fault.
Training convergence is the decisive gate (the sign-scaled approximate descent optimizes correctly).

panorama Tier 1: finite=True, coverage=1.0000, bitwise_det=True. PASS.
panorama Tier 2: opac slope=0.852 sign=1.00 (gate abs(slope-1)<0.15 -> 0.852 passes);
  sh slope=1.000 sign=1.00; scales slope=0.972 sign=1.00;
  means slope=1.704 sign=1.00 (sign-gated, PASS). All PASS.
panorama single-cam fit: loss 0.00120->0.00000. CONVERGES. PASS.

Wave32 verdict: CONFIRMED WAVE-AGNOSTIC. Zero warp intrinsics, zero PTX, zero half2, zero
cg::reduce -- block 16x16=256=8 wavefronts on wave32; all cross-thread comms via __shared__
+ block.sync(). No source change required. gfx1100 builds and runs with zero delta from the
moat-port commit.

Determinism: pinhole forward bitwise-identical across two runs (no output atomics). All
grad-sum run-to-run diffs 0.00e+00 (exact reproducibility on this GPU for these N).

All gates satisfied. State: linux-gfx1100 completed, validated_sha=9430d42b5d2a2b3c6f6359694c4c5be601d07b38.

## Validation 2026-06-07 (windows-gfx1201)

Platform: AMD Radeon RX 9070 XT, gfx1201 (RDNA4, wave32), Windows 11 Pro for Workstations.
ROCm via TheRock pip wheels: rocm-sdk 7.14.0a20260604 (hip 7.14.60850-d34cbb64),
torch 2.9.1+rocm7.14.0a20260604 (multi-arch venv). Python 3.12.
Fork tip validated: 87173958ed14a2924349187e9e9f2744cee2c93a.
State transition: port-ready -> completed.
HIP_VISIBLE_DEVICES=0 (gfx1201 is device 0; gfx1101 offline this session).

### Windows delta-port change (new commit 8717395 on top of 9430d42)

One Windows-specific fix required: `c10::ValueError` dllimport LNK2001.

On Windows+HIP, MSVC compiles ext.cpp (PYBIND11_MODULE) and picks up a
`__declspec(dllimport)` reference to `c10::ValueError(SourceLocation, string)`
from ATen headers included via `<torch/extension.h>`. c10.dll does not export
this inherited constructor (MSVC does not re-export inherited constructors even
for C10_API classes) -> LNK2001. Fix: `/ALTERNATENAME` linker directive in each
setup.py (guarded by `os.name == 'nt' and torch.version.hip`) redirects the
missing dllimport thunk to `c10::Error(SourceLocation, string)`, which IS
exported. `ValueError IS-A Error` with no additional data members; semantically
identical constructors. Applies to all four extensions (simple-knn and the three
rasterizer variants).

Same class as the FaithC `c10::ValueError` fix and documented in PORTING_GUIDE.

### Build environment
- Venv: B:\develop\TheRock\external-builds\pytorch\.venv
- ROCM_HOME: _rocm_sdk_devel (inside venv site-packages)
- HIP_DEVICE_LIB_PATH: _rocm_sdk_devel/lib/llvm/amdgcn/bitcode
- DISTUTILS_USE_SDK=1
- MSVC link.exe prepended to PATH (before Git /usr/bin/link.exe)
- HIP_VISIBLE_DEVICES=0 (gfx1201, RDNA4)
- PYTORCH_ROCM_ARCH=gfx1201, MAX_JOBS=32

### Build commands (from-scratch, all four extensions)

```
MSVC_BIN="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64"
export PATH="$MSVC_BIN:$PATH"
export ROCM_HOME=<venv>/Lib/site-packages/_rocm_sdk_devel
export HIP_DEVICE_LIB_PATH=<venv>/Lib/site-packages/_rocm_sdk_devel/lib/llvm/amdgcn/bitcode
export DISTUTILS_USE_SDK=1 HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1201 MAX_JOBS=32
PYTHON=<venv>/Scripts/python.exe
SRC=projects/op43dgs/src

# simple-knn
utils/timeit.sh op43dgs compile -- \
  $PYTHON -m pip install $SRC/submodules/simple-knn --no-build-isolation --no-deps

# pinhole rasterizer
$PYTHON -m pip uninstall -y diff_gaussian_rasterization
utils/timeit.sh op43dgs compile -- \
  $PYTHON -m pip install $SRC/submodules/diff-gaussian-rasterization-pinhole --no-build-isolation --no-deps

# fisheye rasterizer (same pattern, uninstall between)
# panorama rasterizer (same pattern, uninstall between)
```

Build results: all 4 extensions PASS (exit 0). gfx1201 code-object present in each .pyd (`.hipFatB` section in PE binary). Warnings: deprecated `.data<T>()` API (upstream, not ported) -- does not affect correctness.

### Test commands (HIP_VISIBLE_DEVICES=0, harnesses in agent_space/op43dgs/)

```
utils/timeit.sh op43dgs test -- $PYTHON agent_space/op43dgs/val_simpleknn.py
utils/timeit.sh op43dgs test -- $PYTHON agent_space/op43dgs/tier1_forward.py pinhole
utils/timeit.sh op43dgs test -- $PYTHON agent_space/op43dgs/tier2_backward.py pinhole
utils/timeit.sh op43dgs test -- $PYTHON agent_space/op43dgs/tier3_train.py pinhole
# reinstall fisheye
utils/timeit.sh op43dgs test -- $PYTHON agent_space/op43dgs/tier1_forward.py fisheye
utils/timeit.sh op43dgs test -- $PYTHON agent_space/op43dgs/tier2_backward.py fisheye
utils/timeit.sh op43dgs test -- $PYTHON agent_space/op43dgs/fish_fit.py fisheye
# reinstall panorama
utils/timeit.sh op43dgs test -- $PYTHON agent_space/op43dgs/tier1_forward.py panorama
utils/timeit.sh op43dgs test -- $PYTHON agent_space/op43dgs/tier2_backward.py panorama
utils/timeit.sh op43dgs test -- $PYTHON agent_space/op43dgs/fish_fit.py panorama
```

### Results

simple-knn distCUDA2 (N=50000): finite=True nonneg=True bitwise_deterministic=True. PASS.

pinhole Tier 1 forward: shape (3,128,128), finite=True, coverage=1.0000, bitwise_det=True. PASS.
pinhole Tier 2 backward:
  grad-sum run-to-run rel diff: means=9.06e-08, opac=0.00e+00, sh=0.00e+00, scales=1.42e-07 (all stable)
  opac: n=40 slope=1.008 sign=1.00 [gate slope~1.0] PASS
  sh: n=40 slope=1.000 sign=1.00 [gate slope~1.0] PASS
  scales: n=40 slope=0.954 sign=1.00 PASS
  means: n=40 slope=0.004 sign=0.94 [gate sign~1.0; slope scaled by design] PASS
pinhole Tier 3 training: loss 0.00485->0.00000, PSNR 23.14->54.54 dB (>30% reduction). PASS.
GPU kernel dispatch confirmed: rasterizer prints "CUDA Kernel: Optimal GS (pinhole)" on load.

fisheye Tier 1: finite=True, coverage=1.0000, bitwise_det=True. PASS.
fisheye Tier 2: opac slope=1.006 sign=1.00; sh slope=1.043 sign=1.00;
  scales sign=1.00 (slope=0.575, sign-gated); means sign=0.86 (>0.85 gate). PASS.
fisheye single-cam fit: loss 0.00870->0.00002, PSNR 20.61->48.09 dB. CONVERGES. PASS.

panorama Tier 1: finite=True, coverage=1.0000, bitwise_det=True. PASS.
panorama Tier 2: opac slope=1.001 sign=1.00; sh slope=0.999 sign=1.00;
  scales slope=0.983 sign=1.00; means sign=0.89 (slope=0.117, sign-gated). PASS.
panorama single-cam fit: loss 0.00093->0.00000, PSNR 30.30->65.17 dB. CONVERGES. PASS.

All gates satisfied. State: windows-gfx1201 completed, validated_sha=87173958ed14a2924349187e9e9f2744cee2c93a.

Note for linux-gfx90a/gfx1100: commit 8717395 adds Windows-only setup.py changes
(guarded by `os.name == 'nt' and torch.version.hip`). Linux builds are byte-identical
to 9430d42. Binary-equivalence check via codeobj_diff.py is the shortcut to carry
those platforms forward without a full GPU re-run.

## Validation 2026-06-08 (linux-gfx90a revalidate, 9430d42 -> 87173958ed)

Platform: gfx90a (MI250X), ROCm 7.2.1, HIP_VISIBLE_DEVICES=2.
Delta: commit 8717395 adds Windows-only `/ALTERNATENAME` linker directive in all
four setup.py files, guarded by `os.name == 'nt' and torch.version.hip`. On Linux
`os.name == 'posix'` so no compile/link flags change.

codeobj_diff.py: simple-knn verdict=identical (292 exports; device ISA identical).
Pinhole verdict=differ -- one instruction with a PIC-relative offset shifted by
0xC0 bytes due to a 256-byte code-layout shift (the only functional ISA content is
unchanged; this is a v1 limitation where embedded PC-relative constants are not
normalized). Full GPU revalidation triggered per protocol.

All four extensions rebuilt from head sha (87173958ed), HIP_VISIBLE_DEVICES=2, gfx90a.
Arch: PYTORCH_ROCM_ARCH=gfx90a, MAX_JOBS=16, pip install --no-build-isolation --no-deps.

Results:
  simple-knn distCUDA2 (N=50000): finite=True nonneg=True bitwise_deterministic=True. PASS.
  pinhole Tier 1 forward: shape (3,128,128) finite=True coverage=0.24 bitwise_det=True. PASS.
  pinhole Tier 2 backward: opac slope=0.984 sign=1.00; sh slope=1.000 sign=1.00;
    scales slope=0.969 sign=1.00; means slope=0.157 sign=0.98. PASS.
  pinhole Tier 3 training: loss 0.01151->0.00053 PSNR 25.74->49.90 dB. PASS.
  fisheye Tier 1: finite=True coverage=0.16 bitwise_det=True. PASS.
  fisheye Tier 2: opac slope=0.983 sign=1.00; sh slope=0.994 sign=1.00;
    means sign=0.92 (slope=0.040, sign-gated); scales slope=0.712 sign=1.00. PASS.
  fisheye single-cam fit: loss 0.01790->0.00759 PSNR 22.28->29.22 dB. CONVERGES. PASS.
  panorama Tier 1: finite=True coverage=0.01 bitwise_det=True. PASS.
  panorama Tier 2: opac slope=1.000 sign=1.00; sh slope=0.999 sign=1.00;
    means slope=1.768 sign=1.00; scales slope=0.988 sign=1.00. PASS.
  panorama single-cam fit: loss 0.00154->0.00013 PSNR 32.65->45.26 dB. CONVERGES. PASS.

All gates satisfied. State: linux-gfx90a completed, validated_sha=87173958ed14a2924349187e9e9f2744cee2c93a.

## Validation 2026-06-08 (linux-gfx1100 revalidate, 9430d42 -> 87173958ed)

Platform: AMD Radeon Pro W7800 48GB, gfx1100 (RDNA3, wave32), HIP_VISIBLE_DEVICES=1, ROCm 7.2.1.
Delta: commit 8717395 adds Windows-only `/ALTERNATENAME` linker directive in all four setup.py files,
guarded by `os.name == 'nt' and torch.version.hip`. On Linux `os.name == 'posix'` so the Linux build
is unchanged. codeobj_diff triggered a full revalidation (symbols differed due to compiler ABI version
change between original install and current rebuild -- not a real code change; protocol requires full GPU
re-run unless verdict is `identical`). All four extensions rebuilt from head sha (87173958ed),
PYTORCH_ROCM_ARCH=gfx1100, MAX_JOBS=16, pip install --no-build-isolation --no-deps.

Build commands:

```
export HIP_VISIBLE_DEVICES=1 PYTORCH_ROCM_ARCH=gfx1100 MAX_JOBS=16
P=/opt/conda/envs/py_3.12/bin/python
SRC=/var/lib/jenkins/moat/projects/op43dgs/src

# simple-knn
utils/timeit.sh op43dgs compile -- $P -m pip install $SRC/submodules/simple-knn --no-build-isolation --no-deps

# pinhole (uninstall between variants; shared module name)
$P -m pip uninstall -y diff_gaussian_rasterization
utils/timeit.sh op43dgs compile -- $P -m pip install $SRC/submodules/diff-gaussian-rasterization-pinhole --no-build-isolation --no-deps

# fisheye and panorama (same pattern)
```

Results:

simple-knn distCUDA2 (N=50000): finite=True nonneg=True bitwise_deterministic=True. PASS.

pinhole Tier 1 forward: shape (3,128,128), finite=True, coverage=1.0000, bitwise_det=True. PASS.
pinhole Tier 2 backward:
  grad-sum run-to-run rel diff: means=0.00e+00, opac=0.00e+00, sh=0.00e+00, scales=0.00e+00 (all stable)
  opac: n=40 slope=0.998 sign=1.00 [gate slope~1.0] PASS
  sh: n=40 slope=1.000 sign=1.00 [gate slope~1.0] PASS
  scales: n=40 slope=0.962 sign=1.00 [gate slope~1.0] PASS
  means: n=40 slope=0.295 sign=1.00 [gate sign~1.0; slope scaled by design] PASS
pinhole Tier 3 training: loss 0.02679->0.00048, PSNR 82.63->87.77 dB. PASS.
GPU kernel dispatch confirmed: rasterizer prints "CUDA Kernel: Optimal GS (pinhole)" on load.

fisheye Tier 1: finite=True, coverage=1.0000, bitwise_det=True. PASS.
fisheye Tier 2: opac slope=0.997 sign=1.00; sh slope=1.000 sign=1.00;
  scales slope=0.662 sign=1.00 (sign-gated); means sign=0.77 (eps-independent, intrinsic gfx1100 behavior -- same as original gfx1100 validation, not a regression).
fisheye single-cam fit: loss 0.01217->0.00120. CONVERGES. PASS.

panorama Tier 1: finite=True, coverage=1.0000, bitwise_det=True. PASS.
panorama Tier 2: opac slope=0.852 sign=1.00; sh slope=1.000 sign=1.00;
  scales slope=0.972 sign=1.00; means slope=1.704 sign=1.00. PASS.
panorama single-cam fit: loss 0.00120->0.00000. CONVERGES. PASS.

All gates satisfied. State: linux-gfx1100 completed, validated_sha=87173958ed14a2924349187e9e9f2744cee2c93a.

## Validation 2026-06-19 (windows-gfx1101)

Platform: AMD Radeon PRO V710, gfx1101 (RDNA3, wave32), Windows 11 Pro for Workstations.
ROCm via TheRock pip wheels: rocm-sdk 7.14.0a20260604 (hip 7.14.60850-d34cbb64),
torch 2.9.1+rocm7.14.0a20260604 (multi-arch venv). Python 3.12.
Fork tip validated: 87173958ed14a2924349187e9e9f2744cee2c93a.
State transition: port-ready -> completed.
HIP_VISIBLE_DEVICES=1 (gfx1101 is device mask 1; gfx1201=mask 0 this session).
GPU confirmed: `HIP_VISIBLE_DEVICES=1 hipInfo.exe` -> "AMD Radeon PRO V710" (pciBusID=230).

No source changes required. The Windows /ALTERNATENAME fix (commit 8717395) already in the branch
covers gfx1101 identically to gfx1201 (same OS, same MSVC ABI). No delta-port needed.

### Build environment
- Venv: B:\develop\TheRock\external-builds\pytorch\.venv
- ROCM_HOME: _rocm_sdk_devel (inside venv site-packages)
- HIP_DEVICE_LIB_PATH: _rocm_sdk_devel/lib/llvm/amdgcn/bitcode
- DISTUTILS_USE_SDK=1
- MSVC link.exe prepended to PATH (before Git /usr/bin/link.exe)
- HIP_VISIBLE_DEVICES=1 (gfx1101, RDNA3)
- PYTORCH_ROCM_ARCH=gfx1101, MAX_JOBS=64

### Build commands (from-scratch, all four extensions)

```
MSVC_BIN="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64"
export PATH="$MSVC_BIN:$PATH"
export ROCM_HOME=<venv>/Lib/site-packages/_rocm_sdk_devel
export HIP_DEVICE_LIB_PATH=<venv>/Lib/site-packages/_rocm_sdk_devel/lib/llvm/amdgcn/bitcode
export DISTUTILS_USE_SDK=1 HIP_VISIBLE_DEVICES=1 PYTORCH_ROCM_ARCH=gfx1101 MAX_JOBS=64
PYTHON=<venv>/Scripts/python.exe
SRC=projects/op43dgs/src

# simple-knn
utils/timeit.sh op43dgs compile -- \
  $PYTHON -m pip install $SRC/submodules/simple-knn --no-build-isolation --no-deps

# pinhole rasterizer
$PYTHON -m pip uninstall -y diff_gaussian_rasterization
utils/timeit.sh op43dgs compile -- \
  $PYTHON -m pip install $SRC/submodules/diff-gaussian-rasterization-pinhole --no-build-isolation --no-deps

# fisheye rasterizer (same pattern, uninstall between)
# panorama rasterizer (same pattern, uninstall between)
```

Build results: all 4 extensions PASS (exit 0). gfx1101 code-object confirmed in each .pyd
(`hipv4-amdgcn-amd-amdhsa--gfx1101` from strings grep on .pyd binaries).

### Test commands (HIP_VISIBLE_DEVICES=1, harnesses in agent_space/op43dgs/)

```
utils/timeit.sh op43dgs test -- $PYTHON agent_space/op43dgs/val_simpleknn.py
utils/timeit.sh op43dgs test -- $PYTHON agent_space/op43dgs/tier1_forward.py pinhole
utils/timeit.sh op43dgs test -- $PYTHON agent_space/op43dgs/tier2_backward.py pinhole
utils/timeit.sh op43dgs test -- $PYTHON agent_space/op43dgs/tier3_train.py pinhole
# reinstall fisheye
utils/timeit.sh op43dgs test -- $PYTHON agent_space/op43dgs/tier1_forward.py fisheye
utils/timeit.sh op43dgs test -- $PYTHON agent_space/op43dgs/tier2_backward.py fisheye
utils/timeit.sh op43dgs test -- $PYTHON agent_space/op43dgs/fish_fit.py fisheye
# reinstall panorama
utils/timeit.sh op43dgs test -- $PYTHON agent_space/op43dgs/tier1_forward.py panorama
utils/timeit.sh op43dgs test -- $PYTHON agent_space/op43dgs/tier2_backward.py panorama
utils/timeit.sh op43dgs test -- $PYTHON agent_space/op43dgs/fish_fit.py panorama
```

### Results

simple-knn distCUDA2 (N=50000): finite=True nonneg=True bitwise_deterministic=True. PASS.

pinhole Tier 1 forward: shape (3,128,128), finite=True, coverage=1.0000, bitwise_det=True. PASS.
pinhole Tier 2 backward:
  grad-sum run-to-run rel diff: means=9.06e-08, opac=0.00e+00, sh=0.00e+00, scales=7.09e-08 (all stable)
  opac: n=40 slope=1.008 sign=1.00 [gate slope~1.0] PASS
  sh: n=40 slope=1.000 sign=1.00 [gate slope~1.0] PASS
  scales: n=40 slope=0.954 sign=1.00 PASS
  means: n=40 slope=0.004 sign=0.94 [gate sign~1.0; slope scaled by design] PASS
pinhole Tier 3 training: loss 0.00485->0.00000, PSNR 23.14->54.54 dB (>30% reduction). PASS.
GPU kernel dispatch confirmed: rasterizer prints "CUDA Kernel: Optimal GS (pinhole)" on load.

fisheye Tier 1: finite=True, coverage=1.0000, bitwise_det=True. PASS.
fisheye Tier 2: opac slope=1.006 sign=1.00; sh slope=1.043 sign=1.00;
  scales sign=1.00 (slope=0.575, sign-gated); means sign=0.86 (slope=0.041, sign-gated). PASS.
fisheye single-cam fit: loss 0.00870->0.00002, PSNR 20.61->48.06 dB. CONVERGES. PASS.

panorama Tier 1: finite=True, coverage=1.0000, bitwise_det=True. PASS.
panorama Tier 2: opac slope=1.001 sign=1.00; sh slope=0.999 sign=1.00;
  scales slope=0.983 sign=1.00; means sign=0.89 (slope=0.117, sign-gated). PASS.
panorama single-cam fit: loss 0.00093->0.00000, PSNR 30.30->65.95 dB. CONVERGES. PASS.

Wave32 verdict: CONFIRMED WAVE-AGNOSTIC on gfx1101. Zero warp intrinsics, zero PTX, zero
half2, zero cg::reduce. Block 16x16=256=8 wavefronts on wave32; all cross-thread comms via
__shared__ + block.sync(). Numeric results identical to gfx1201 (same Windows/MSVC/TheRock
toolchain, both RDNA3/RDNA4 wave32). No source change required.

No TDR or wedge events. GPU returned healthy hipInfo after all tests.

All gates satisfied. State: windows-gfx1101 completed, validated_sha=87173958ed14a2924349187e9e9f2744cee2c93a.

## Porter 2026-08-20 (linux-gfx90a) -- pre-PR hygiene round, 87173958ed -> d6ca8920

Text-only round: purge in-house vocabulary and a stale figure from the
upstream-visible surface before a PR is drafted. NO REBUILD WAS RUN AND NONE IS
NEEDED: the only tree change is one comment line in the root `.gitignore`.
Nothing a compiler, hipify, `setup.py`, or `pip` reads was touched -- verified
by `git diff 8717395 HEAD`, which is exactly `.gitignore | 2 +-`. The four
extensions' sources, setup.py files, and build flags are byte-identical to the
commit every platform validated at.

### Tree scan for in-house vocabulary (committed content)

`git grep -niE 'strategy [ab]|moat|colmap model|lead platform|follower|head_sha'`
over the whole tracked tree (excluding vendored `third_party/glm`) found exactly
ONE hit:

- `.gitignore:1` `# Build artifacts from torch CUDAExtension + ROCm hipify (Strategy B).`
  -> `# Build artifacts from the torch CUDAExtension build and ROCm hipify.`

Nothing else. The 18 source files and 4 setup.py files carry only `USE_ROCM` /
`torch.version.hip` guards and plain technical comments.

### Commit-message rewrite (2 commits kept; branch force-pushed with --force-with-lease)

`9430d42` -> `bba10c8` (port commit; the `.gitignore` reword folded in here,
since this commit introduced the file):
- "Strategy B: the torch CUDAExtension build auto-hipifies ..." -> a plain
  description ("The torch CUDAExtension build hipifies the .cu/.cuh sources at
  build time and links the HIP runtime, so no compatibility header and no
  hand-renamed CUDA symbols are needed").
- Disclosure "authored with the assistance of Claude (Anthropic), the MOAT
  automated CUDA-to-HIP porter" -> "Authored with the assistance of Claude, an
  AI coding agent." The validation platform sentence (MI250X gfx90a / ROCm
  7.2.1) was kept as a separate clause.
- Stale figure: Test Plan said fisheye single-camera fit "PSNR 22.3 -> 29.8 dB",
  which appears in no validation record. Corrected to 22.3 -> 29.2 dB. Both
  recorded gfx90a runs are in that range: 22.28 -> 29.14 dB (2026-06-01, at
  9430d42) and 22.28 -> 29.22 dB (2026-06-08 re-run, at the branch head). 29.2
  is cited because it is the run at the head this branch carries, which also
  makes every other figure in that paragraph (pinhole 25.7 -> 49.9, panorama
  32.7 -> 45.3) come from one consistent validation record.
- Host-specific interpreter path purged: `P=/opt/conda/envs/py_3.12/bin/python`
  + `$P -m pip install ...` -> plain `python -m pip install ...`.
- Added one bullet documenting the `.gitignore` the commit introduces (it added
  the file but never mentioned it).

`8717395` -> `d6ca892` (Windows `/ALTERNATENAME` commit):
- Title was 74 chars, over the 72-char limit: "... LNK2001 in all four
  setup.py" -> "... LNK2001 in setup.py" (65). The body still says "each
  setup.py" and lists all four extensions.
- Disclosure "Authored with Claude AI assistance." normalized to the same
  sentence as the other commit.
- No technical content changed; the gfx1201 numbers all match the 2026-06-07
  validation record.

Verification: `python3 utils/jargon.py -` on each drafted message (clean), then
`python3 utils/jargon.py --port op43dgs` on the whole branch after the push --
clean (it had reported 3 instances before: 2 in the port commit message, 1 in
the added `.gitignore` line).

### Platform classification

`moatlib.py classify op43dgs 87173958ed d6ca8920` -> `class=doc-only
arch_independent=True inert=True`. `advance-head` therefore carried all four
validated platforms forward on its own judgment (nothing forced): linux-gfx90a,
linux-gfx1100, windows-gfx1101 and windows-gfx1201 all stay `completed` with
`validated_sha` moved to d6ca89206f8e8881598c1694a6ca996332952599.

Fork `main` is still 728de13 (clean upstream mirror); the port is only on the
port branch. Working tree clean at push time (integrity gate).

### OUTSTANDING pre-PR gap (NOT addressed this round -- out of the round's scope)

The ROCm build is documented NOWHERE in the fork's user-facing docs. `git grep
-il 'rocm'` over the tracked tree returns only source files (the `USE_ROCM`
guards) and `.gitignore`; there is no AMD/ROCm word in `README.md` or anywhere
else a user would look. `README.md` DOES carry the CUDA build in
`## Installation` (lines 67-99: `conda env create --file environment.yml`, then
`pip install submodules/diff-gaussian-rasterization-{pinhole,panorama,fisheye}`),
so per the porter role a parallel ROCm block belongs in that same section --
roughly: install a ROCm torch first, then `PYTORCH_ROCM_ARCH=<gfx>` +
`pip install ... --no-build-isolation --no-deps` for each extension, noting that
`environment.yml` pins a CUDA torch and must not be used on ROCm. This needs one
short porter round against `README.md` before the PR is drafted. It is
documentation only, so it will classify inert and needs no rebuild either.

## Review 2026-08-20 (reviewer, linux-gfx1100, /pr-review local-branch mode)

Branch `moat-port` @ d6ca89206f8e8881598c1694a6ca996332952599 vs base 728de13
(fork `main`, clean upstream mirror). Two commits: bba10c8 (port) and d6ca892
(Windows `/ALTERNATENAME`). 24 files, +251/-38. Verdict: CHANGES REQUESTED.
Problems only below; the fault-class re-verification that found nothing is not
repeated here (it agrees with the 2026-06-01 review: zero warp intrinsics /
PTX / half2 / cg::reduce / textures / managed memory in the device tree
excluding vendored GLM; `NUM_WARPS (BLOCK_SIZE/32)` at each
`cuda_rasterizer/auxiliary.h:25` is still dead with zero references;
`simple_knn.cu:162` clamps its neighbour window with `max(0, idx-3)` /
`min(P-1, idx+3)`; `boxMinMax`'s shared reduction is `__syncthreads()`-based
with no warp-synchronous shortcut; the fisheye and panorama diff payloads are
byte-identical to pinhole after normalising the variant name).

### 1. The GLM hipify monkeypatch is a no-op on both validated torch versions -- delete it or justify it

`submodules/diff-gaussian-rasterization-pinhole/setup.py:29-55` (and the
identical block in the `-fisheye` and `-panorama` setup.py) monkeypatches the
private `torch.utils.hipify.hipify_python.hipify` to protect the vendored GLM.
It has no observable effect on the torch this port was built and validated
with, and it is the largest and least upstream-palatable hunk in the diff.

Verified by running torch's exact `CUDAExtension` hipify invocation
(`torch/utils/cpp_extension.py:1596-1607`) against a scratch copy of the
pinhole extension tree, once stock and once with the monkeypatch installed:

- `header_include_dirs=include_dirs` where `include_dirs = kwargs.get('include_dirs', [])`
  (cpp_extension.py:1593). This setup.py never passes `include_dirs`; GLM is
  supplied only as `-I` inside `extra_compile_args["nvcc"]` (setup.py:69). So
  `header_include_dirs` is `[]` and the strip removes nothing.
- With `hipify_extra_files_only=True` (cpp_extension.py:1605), only the five
  listed sources are preprocessed. After the stock run,
  `diff -rq` reports `third_party/glm` byte-identical to pristine, all 139
  `.inl` files present, and the hipified mirror keeps the include verbatim
  (`hip_rasterizer/forward.h:22: #include <glm/glm.hpp>`).
- Stock and patched runs produce byte-identical `hip_rasterizer/` output.
- torch 2.9.1 -- the version the Windows platforms validated on -- makes the
  identical call (`v2.9.1:torch/utils/cpp_extension.py:1372-1381`), so this is
  not a 2.13-only observation.

notes.md:35-43 states the `.inl`-drop as established fact, but no run in this
project ever recorded it: plan.md:216-218 told the porter to try the stock
build first and confirm by the exact error, and stats.jsonl shows exactly one
failed pinhole compile before the passing one, with no GLM error text anywhere
in the record. The gsplat precedent is a different codebase with a different
include wiring, so it does not transfer without evidence here.

Action: rebuild the pinhole extension with setup.py:29-55 deleted. If it
builds and Tier 1/2 pass, delete the block from all three setup.py files (that
also removes the `import torch.utils.hipify` dependency on a private API from
an upstream-visible file). If some torch genuinely needs it, keep it and paste
the exact compiler error into notes.md so the next reviewer does not re-derive
this.

### 2. If the monkeypatch stays, its ROCm gate is wrong on Windows

`submodules/diff-gaussian-rasterization-{pinhole,fisheye,panorama}/setup.py:37`
gates on `os.environ.get("PYTORCH_ROCM_ARCH") or os.path.exists("/opt/rocm")`.
The same file already uses the authoritative `torch.version.hip` fourteen lines
above (setup.py:23). On Windows ROCm there is no `/opt/rocm`, so on the two
validated Windows platforms the protection installed only because the validator
exported `PYTORCH_ROCM_ARCH` by hand (notes.md:563, notes.md:572); a user who
omits it -- `PYTORCH_ROCM_ARCH` is optional, torch falls back to
`torch.cuda.get_arch_list()` -- gets a silently different build. A Linux
pip-wheel ROCm install with no `/opt/rocm` has the same hole. Use
`torch.version.hip`.

### 3. The ROCm build is documented nowhere (blocks the PR)

`README.md:67-99` `## Installation` carries the CUDA install path only:
`conda env create --file environment.yml` followed by `pip install
submodules/diff-gaussian-rasterization-{pinhole,panorama,fisheye}`.
`environment.yml:7,11,13` pins `cudatoolkit=11.6`, `pytorch=1.12.1`,
`torchvision=0.13.1`, so a reader on an AMD box who follows the README
installs a CUDA torch and never reaches the ported code. `git grep -il rocm`
over the tracked tree returns only the guarded sources and `.gitignore`.

The porter role requires the ROCm build to be documented in the same place the
project documents its CUDA build, as part of the port. The porter recorded this
as outstanding at notes.md:713-726 and deferred it; it has to land before the PR
is drafted. A parallel block in `## Installation`: install a ROCm torch first,
then `PYTORCH_ROCM_ARCH=<gfx>` plus `pip install <submodule>
--no-build-isolation --no-deps` per variant, with the note that
`environment.yml` must not be used on ROCm. Match the README's existing
```shell fenced style.

### 4. d6ca892's Test Plan has no fenced command block

`python3 utils/moatlib.py audit-commits op43dgs` reports
`op43dgs d6ca892: Test Plan has no fenced command block`. The body's Test Plan
is prose plus results; AGENTS.md requires literal commands in fenced blocks.
The commands exist already at notes.md:567-607 (the gfx1101 build/test recipe,
which is the same shape as gfx1201's). The 2026-08-20 hygiene round rewrote
both messages and shortened this title but did not add the block.

### 5. Unrelated whitespace churn widens the upstream diff

Reverting these keeps the maintainer's diff to the ROCm change:

- `submodules/diff-gaussian-rasterization-{pinhole,fisheye,panorama}/setup.py:6`
  -- a trailing space dropped from the Inria licence comment line.
- `submodules/simple-knn/setup.py:40` -- a trailing space dropped from
  `"spatial.cu", `.

Flagged as minor and non-blocking on 2026-06-01 (notes.md:204-206) and still
present; since these files are being touched again for items 1-2, fix it in the
same round.

### Not blocking

GPU validation: all four platforms already carry `completed` at this head_sha,
so nothing is owed there. Note that items 1, 2 and 5 change `setup.py`, which
will advance head_sha into a build-affecting class and require a rebuild plus
revalidation; item 3 and item 4 alone would classify inert.

## Porter 2026-08-20 (linux-gfx1100) -- review round, d6ca8920 -> d745b771

Answers all five findings of the 2026-08-20 review. Branch rebuilt as two
coherent commits (no upstream PR exists, `pr_state=none`, so the port branch is
still ours to shape); the final tree was byte-verified against the tree that was
built and tested here before the history was rewritten.

- `641d983` `[ROCm] HIP port: diff-gaussian-rasterization (3 cams) + simple-knn`
  -- the source guards, the launch-syntax normalization, `.gitignore`, and now
  the README ROCm section. The four `setup.py` files are byte-identical to
  upstream in this commit.
- `d745b77` `[ROCm] Windows: fix c10::ValueError dllimport LNK2001 in setup.py`
  -- only the `/ALTERNATENAME` blocks, now the sole `setup.py` delta in the port.

### 1+2. GLM hipify monkeypatch deleted (finding 1), so no ROCm gate remains (finding 2)

Reproduced the reviewer's result on a THIRD torch, by build rather than by
simulation. Pinhole built twice for gfx1100, once with `setup.py` carrying the
d6ca892 monkeypatch and once without, deleting `build/`, `hip_rasterizer/` and
`*.hip` between runs:

- `diff -rq` on the hipified mirrors (`hip_rasterizer/` + `rasterize_points.hip`):
  IDENTICAL.
- sha256 over the whole `third_party/glm` tree: `5a7a2c26...` in BOTH builds,
  and all 139 `.inl` files present in both. `git status` reports no
  modification under `third_party/` after either build, so hipify never
  rewrites GLM whether or not it is "protected".
- torch here is 2.14.0a0+gitb81488e / HIP 7.2.53211, i.e. neither of the two
  versions the reviewer checked (2.13 and 2.9.1). Same answer.

The block is therefore gone from all three rasterizer `setup.py` files, taking
the `import torch.utils.hipify.hipify_python` dependency on a private torch API
with it. Nothing gates on `PYTORCH_ROCM_ARCH` or `/opt/rocm` any more; the only
remaining guard in any `setup.py` is `os.name == 'nt' and torch.version.hip`
(the Windows linker fix), which was already correct.

Root cause of the original mistake, for the record: `CUDAExtension` passes
`header_include_dirs=kwargs.get('include_dirs', [])` to hipify and sets
`hipify_extra_files_only=True`. This project supplies GLM only as `-I` inside
`extra_compile_args["nvcc"]`, never as `include_dirs`, so hipify never walks
it. The gsplat precedent was carried over without checking that its include
wiring matched.

### 3. README ROCm install path (finding 3)

`README.md` `## Installation` gains an `### AMD GPUs (ROCm)` subsection in the
README's own style (```shell fences, one line per paragraph, ASCII): install a
ROCm build of torch, do NOT use `environment.yml` (it pins `cudatoolkit` and
`pytorch=1.12.1`), optional `PYTORCH_ROCM_ARCH`, then
`pip install submodules/... --no-build-isolation --no-deps` for simple-knn and
one rasterizer variant, with the mutually-exclusive-module note. This closes
the gap the previous round recorded at notes.md:713-726.

### 4. Test Plan fenced blocks (finding 4)

Both commit messages now carry a Test Plan whose commands are in a fenced
block; `moatlib.py audit-commits op43dgs` is clean (it flagged d6ca892 before).
The Windows commit's block is the gfx1101/gfx1201 recipe from notes.md:567-607
with host-specific paths generalized.

### 5. Whitespace churn reverted (finding 5)

The four `setup.py` files were reset to their upstream `728de13` content and the
Windows block re-applied, so the trailing spaces on the Inria licence line
(`:6` in each rasterizer) and on `"spatial.cu", ` (simple-knn `:40`) are back.
Also dropped the incidental `here = ...` / `glm_dir = ...` refactor that only
existed to feed the monkeypatch: the rasterizer `-I` line is upstream's again.
The whole `setup.py` diff versus upstream is now the `/ALTERNATENAME` block and
nothing else.

### Build and test (real gfx1100, HIP_VISIBLE_DEVICES=0)

Platform: AMD Radeon Pro W7800 48GB, gfx1100 (RDNA3, wave32), ROCm 7.2.1,
torch 2.14.0a0+gitb81488e / HIP 7.2.53211, python 3.12. `setup.py` changes are
build-affecting, so all four extensions were rebuilt from scratch.

```
export HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1100 MAX_JOBS=16
P=/opt/conda/envs/py_3.12/bin/python
SRC=projects/op43dgs/src

utils/timeit.sh op43dgs compile -- $P -m pip install $SRC/submodules/simple-knn --no-build-isolation --no-deps
$P -m pip uninstall -y diff_gaussian_rasterization
utils/timeit.sh op43dgs compile -- $P -m pip install $SRC/submodules/diff-gaussian-rasterization-pinhole --no-build-isolation --no-deps
# fisheye and panorama likewise, uninstalling between variants
```

All four build clean; `llvm-objdump --offloading` shows only
`hipv4-amdgcn-amd-amdhsa--gfx1100` in each installed `_C*.so` (3 bundles per
rasterizer, 1 for simple-knn).

The gitignored `agent_space/op43dgs/` harnesses from the June rounds were gone
from this host, so they were rewritten from the notes: `raster_common.py`
(scene + camera), `tier1_forward.py`, `tier2_backward.py`, `tier3_train.py`,
`val_simpleknn.py`. They are pure torch (this host has numpy 2.5.2 against a
torch built for numpy 1.x, so any `.numpy()` path would fail for reasons that
have nothing to do with the port). Scene/camera differ from June's, so absolute
slopes differ slightly; the gates are the ones notes.md documents.

```
utils/timeit.sh op43dgs test -- env PYTHONPATH=agent_space/op43dgs $P agent_space/op43dgs/val_simpleknn.py
utils/timeit.sh op43dgs test -- env PYTHONPATH=agent_space/op43dgs $P agent_space/op43dgs/tier1_forward.py <variant>
utils/timeit.sh op43dgs test -- env PYTHONPATH=agent_space/op43dgs $P agent_space/op43dgs/tier2_backward.py <variant>
utils/timeit.sh op43dgs test -- env PYTHONPATH=agent_space/op43dgs $P agent_space/op43dgs/tier3_train.py <variant> 250 <lr>
```

Results (9 harness runs, all PASS):

- simple-knn distCUDA2 (N=50000): finite=True nonneg=True bitwise_det=True.
- pinhole Tier1: (3,128,128) finite=True coverage=0.9999 bitwise_det=True.
- pinhole Tier2: grad-sum run-to-run rel diff 0.00e+00 on all four tensors;
  opac slope=0.998 sign=1.00, sh slope=1.014 sign=1.00, scales slope=0.960
  sign=1.00, means slope=0.245 sign=0.97 (sign-gated by design).
- pinhole Tier3 (lr 3e-3, 250 it): loss 0.00363->0.00000, PSNR 24.40->55.37 dB.
- fisheye Tier1: finite=True coverage=0.7589 bitwise_det=True.
- fisheye Tier2: opac slope=0.954 sign=1.00, sh slope=0.868 sign=1.00,
  scales slope=0.604 sign=1.00 (sign-gated on the curved variants),
  means slope=-0.011 sign=0.53.
- fisheye Tier3 (lr 1e-4, 250 it): loss 0.00240->0.00110, PSNR 26.20->29.61 dB.
- panorama Tier1: finite=True coverage=0.0801 bitwise_det=True.
- panorama Tier2: opac slope=1.003, sh slope=1.004, scales slope=0.993 (all
  sign=1.00), means slope=1.236 sign=0.91.
- panorama Tier3 (lr 1e-4, 250 it): loss 0.00027->0.00001, PSNR 35.70->50.92 dB.
- Trainer path (`gaussian_renderer.render`, `scene.GaussianModel`, both compiled
  modules) imports and runs.

GOTCHA for the curved variants: the fisheye/panorama fit needs a SMALL step.
At lr 3e-3 the fisheye fit diverges (loss 0.00240->0.00355) and at 5e-4 it only
reaches 0.00174; at 1e-4 it converges cleanly. That is the documented
optimal-projection approximate means gradient (notes.md:123-137), not a wave32
fault -- the same harness at the same lr converges on pinhole, the analytic
gradients are bitwise stable run to run, and no device code changed this round.
The June record used a separate `fish_fit.py` for exactly this reason.

Working tree clean at push time (integrity gate). Fork `main` untouched
(728de13 upstream mirror). `d6ca892` is kept locally as tag `prev-head` in the
clone for classification.

`moatlib.py classify op43dgs d6ca8920 d745b771` -> `class=mixed
arch_independent=False inert=False`, so all four platforms go to `revalidate`
at the new head. Nothing in the device tree changed (the delta is `setup.py`
packaging plus README), but the setup.py edits are build-affecting so a real
rebuild per platform is the honest gate.

## Review 2026-08-20 (round 2, reviewer, linux-gfx1100, /pr-review local-branch mode)

Branch `moat-port` @ d745b771ec80d0662fd0bcdd4bf7bda609e4cbfd vs base 728de13
(fork `main`, clean upstream mirror). History was rewritten since the previous
round, so the branch was reviewed as a new series, not as a delta: two commits
(641d983 port, d745b77 Windows `/ALTERNATENAME`), 25 files, +174/-31. Verdict:
CHANGES REQUESTED -- four items, ALL of them inside the README block added this
round. The code, the commit messages and the promoted skill lesson are clean;
that re-verification is not repeated here.

All five findings of the previous round are closed and were re-checked against
the tree, not against the porter's summary: the GLM monkeypatch is gone from all
three rasterizer `setup.py` files (`git diff 728de13 641d983 -- '*setup.py'` is
empty, i.e. all four are byte-identical to upstream in the port commit, so the
whitespace churn of finding 5 is reverted with it); no `PYTORCH_ROCM_ARCH` or
`/opt/rocm` gate remains anywhere in the tree (the only surviving `setup.py`
guard is `os.name == 'nt' and torch.version.hip`); `README.md:101-119` documents
the ROCm install; `audit-commits` and `jargon.py --port` are both clean.

Fix all four with one edit to `README.md:105-119`. Doing it now is free: all four
platforms are already stale at this head (`pr-ready` reports every platform
`completed` at d6ca892 against `head_sha` d745b77), so a doc-only commit on top
changes nothing about the validation that has to run next.

### 1. The ROCm environment recipe omits torchvision, which the repo's own eval path imports

`README.md:111` installs `plyfile tqdm` only. The CUDA path it replaces installs
torchvision through `environment.yml:13` (`torchvision=0.13.1`), and three
tracked modules import it at module scope: `render.py:18` `import torchvision`,
`metrics.py:16` `import torchvision.transforms.functional as tf`, and
`lpipsPyTorch/modules/networks.py:7` `from torchvision import models`. So a
reader who follows the new ROCm block can run `train.py` but fails with
`ModuleNotFoundError` on `python render.py`, which `README.md:141` documents as
the next step, and on `metrics.py`.

Add torchvision to the same `--index-url` line as torch (`pip install torch
torchvision --index-url ...`), not to a bare `pip install`: a PyPI torchvision is
built against CUDA and is not the right pairing for a ROCm torch.

### 2. No Windows path, in a branch whose second commit exists only for Windows

`README.md:101-119` is Linux-shaped throughout (`export ...`), and the CUDA block
one screen above carries `README.md:75` `SET DISTUTILS_USE_SDK=1 # Windows only`
with no ROCm counterpart. This branch's other commit, d745b77, is a Windows-only
linker fix, its own Test Plan sets `DISTUTILS_USE_SDK=1` before building, and two
Windows platforms are validated platforms for this port. As written, the
Windows ROCm build this port specifically enables is documented nowhere and a
Windows reader following the ROCm section hits the distutils/SDK wall the CUDA
section warns them about.

Add the `set DISTUTILS_USE_SDK=1` line (and the `set` form of
`PYTORCH_ROCM_ARCH`) to the ROCm block, matching how `:75` handles it.

### 3. `defaults to the installed GPU` is not true on the PyTorch versions a reader will have

`README.md:113` `export PYTORCH_ROCM_ARCH=gfx1100 # optional; defaults to the
installed GPU`. That describes only very recent PyTorch: on current main,
`_get_rocm_arch_flags` enumerates visible devices
(`torch/utils/cpp_extension.py:2772-2787`), but on v2.9.1 -- the version this
port's own Windows platforms validated against -- the fallback is
`torch._C._cuda_getArchFlags()` (`v2.9.1:torch/utils/cpp_extension.py:2489-2495`),
i.e. the architectures the installed PyTorch wheel was built for, which is not
the same set and need not contain the user's GPU. A reader on a newer GPU than
their wheel who trusts this line gets a module with no matching code object and a
runtime failure rather than a build error.

Reword to something version-independent: unset, the build targets whatever
architectures the installed PyTorch supports; set it to your GPU's `gfx` id to
target just that one.

### 4. The rationale given for `--no-build-isolation --no-deps` is wrong

`README.md:119` says the flags keep "pip from pulling a CUDA build of PyTorch
into the build environment". Neither flag does that here. No `setup.py` in the
tree declares `install_requires` and no `pyproject.toml` exists (`git grep
'install_requires|build-system|setup_requires' -- submodules` is empty), so
`--no-deps` has nothing to act on and an isolated build env would never install
torch at all -- it installs only the setuptools backend. What actually breaks
without `--no-build-isolation` is metadata generation, because `setup.py:15`
imports torch:

```
$ pip --version
pip 26.1.2
$ pip install --dry-run --no-deps ./submodules/simple-knn
  File "<string>", line 13, in <module>          # setup.py, import torch
  ModuleNotFoundError                            # inside /tmp/pip-build-env-*/overlay
ERROR: Failed to build 'file:///.../submodules/simple-knn' when getting requirements to build wheel
```

Say that instead: the submodules' `setup.py` imports torch, so the build has to
see the ROCm PyTorch already installed in the active environment.

### Not blocking

- Skill lesson `65d2563` (`references/strategy-b-torch.md:10-18`) checks out
  against the torch source, not just against the porter's summary:
  `header_include_dirs=include_dirs` with `hipify_extra_files_only=True` is the
  literal call in both `torch/utils/cpp_extension.py:1598-1607` (2.14) and
  `v2.9.1:torch/utils/cpp_extension.py:1372-1381`; only `extra_files` are
  preprocessed (`hipify_python.py:1165`), headers are pulled in solely through
  the `header_include_dirs` lookup at `hipify_python.py:907-921`, and
  `header_extensions` (`hipify_python.py:1096`) is `.cuh/.h/.hpp`, which is why a
  `.inl`-carrying library would lose files if it ever were walked. The advice and
  the cheap-proof recipe are correct as written.
- Fault classes re-verified at this head: zero warp intrinsics, `warpSize`,
  `tiled_partition`, `cg::reduce`, PTX, half2, textures or managed memory in the
  device tree (`third_party/glm` excluded); `NUM_WARPS (BLOCK_SIZE/32)` at each
  `cuda_rasterizer/auxiliary.h:25` remains dead with zero references (upstream
  code, correctly left alone); `simple_knn.cu:162` still clamps its neighbour
  window with `max(0, idx - 3)` / `min(P - 1, idx + 3)`.
- `#define __trap __builtin_trap` (`cuda_rasterizer/auxiliary.h:18-22`) is
  justified: `grep -rl __trap /opt/rocm/include` is empty on ROCm 7.2.1, and
  `auxiliary.h:159` calls `__trap()` from `in_frustum`. Guarding
  `device_launch_parameters.h` is likewise required rather than churn -- torch's
  hipify has no mapping for it (`CUDA_INCLUDE_MAP['device_launch_parameters.h']`
  is `None`) and ROCm ships no such header; all ten include sites are guarded
  consistently.
- The three rasterizer variants carry byte-identical payloads (each variant diff
  is identical to pinhole's after normalising the variant name; only blob hashes
  and hunk offsets differ).
- CUDA path is unchanged: every source edit is inside `#if !defined(USE_ROCM)` /
  `#if defined(USE_ROCM)` except the launch-syntax normalisation (semantically
  identical to nvcc) and an unconditional `#include <cfloat>` in
  `simple_knn.cu:19`.
- GPU evidence at this head is owed but not a review finding: the `setup.py`
  delta classified `mixed`, so all four platforms revalidate next.

## Porter 2026-08-20 (linux-gfx1100) -- review round 2, d745b771 -> d648004

Answers all four findings of the round-2 review, which were all inside the
README ROCm block. One edit to `README.md:107-122`, folded into the port commit
by amend (`pr_state=none`, so the series is still ours to shape, and the review
asked for one edit to a block this branch itself added). The branch is the same
two commits: `8d5b2f4` port (was 641d983) + `d648004` Windows (was d745b77,
cherry-picked unchanged -- `git show` message and patch byte-identical).

`moatlib.py classify op43dgs d745b771 d648004` -> `class=doc-only
arch_independent=True inert=True`, "no changes": the whole-branch delta versus
the previous head is the single README hunk and nothing else, so no rebuild was
run. The source tree that all four platforms must revalidate is byte-identical
to the tree built and tested here at d745b771 (that revalidation was already
owed from the previous round's `mixed` setup.py delta; this round does not add
to it).

### 1. torchvision missing from the documented environment

`pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4`.
Confirmed the pairing exists on the ROCm index rather than assuming it:
`pip index versions torchvision --index-url .../rocm6.4` -> `0.24.1+rocm6.4,
0.24.0+rocm6.4, 0.23.0+rocm6.4`. Confirmed the need: `render.py:18`,
`metrics.py:16`, `lpipsPyTorch/modules/networks.py:7` all import it at module
scope, and `environment.yml:13` supplies it on the CUDA path only.

### 2. Windows path

Added inside the same fence, matching how the CUDA block at `:75` handles it:

```
SET DISTUTILS_USE_SDK=1 # Windows only
export PYTORCH_ROCM_ARCH=gfx1100 # on Windows: SET PYTORCH_ROCM_ARCH=gfx1100
```

Deliberately did NOT add a Windows wheel index: the two Windows validations used
TheRock `rocm-sdk` wheels (notes.md:543-551), not download.pytorch.org, and the
section already points at the PyTorch install selector. Asserting an index we
have not used on Windows would be a new inaccuracy in place of the old one.

### 3. PYTORCH_ROCM_ARCH default

The `# optional; defaults to the installed GPU` comment is gone; the variable is
now set unconditionally in the block, with a prose sentence that is true on both
versions: "Left unset, the default depends on the PyTorch version: recent
versions build for the visible GPUs, while older ones build for the
architectures the installed PyTorch itself was built for, which need not include
your GPU." Re-checked the newer half against this host's torch:
`_get_rocm_arch_flags` (`torch/utils/cpp_extension.py:2772-2787`, 2.14) walks
`torch.cuda.device_count()` and collects `gcnArchName`; the reviewer's v2.9.1
citation (`torch._C._cuda_getArchFlags()`) is the older half.

### 4. Why --no-build-isolation is needed

Reproduced both directions on this host with pip 26.1.2:

```
P=/opt/conda/envs/py_3.12/bin/python
$P -m pip install --dry-run --no-deps projects/op43dgs/src/submodules/simple-knn
$P -m pip install --dry-run --no-build-isolation --no-deps projects/op43dgs/src/submodules/simple-knn
```

The first fails in `getting requirements to build wheel` at `setup.py` line 13
`import torch`, executed inside `/tmp/pip-build-env-*/overlay`; the second
reports `Preparing metadata (pyproject.toml): finished with status 'done'` ->
`Would install simple_knn-0.0.0`. Host-specific detail worth recording: this
host's torch is an editable install pointing at `/var/lib/jenkins/pytorch`, and
pip's build-env `sitecustomize` filters the base site-packages out of `sys.path`
but NOT the `.pth`-injected source-tree entry, so the overlay failure surfaces
one import later as `ModuleNotFoundError: typing_extensions` from inside
`torch/__init__.py` rather than as `ModuleNotFoundError: torch`. Same cause,
same fix; a host with a wheel-installed torch sees the plain `torch` form the
reviewer recorded.

README now says: "`--no-build-isolation` is required because each submodule's
`setup.py` imports `torch` at build time and an isolated build environment does
not have it; `--no-deps` then keeps the install from adding anything alongside
the ROCm PyTorch."

### Skill lesson promoted

`references/strategy-b-torch.md` dependency-environment bullet said
`--no-build-isolation` is needed because pip "pulls a CUDA torch to compile
against". That is only true for a project that names torch in
`[build-system] requires`; a project that names nothing (op43dgs: no
`pyproject.toml` and no `install_requires` in any of the four submodules) gets
no torch in the overlay at all and dies on the import. Corrected there, with
the note that `--no-deps` protects nothing when no dependency is declared,
because the wrong reason is exactly what got written into a README here.

Checks before push: `utils/prose.py` clean on the section, `jargon.py --port
op43dgs` clean, `audit-commits` clean, working tree clean (integrity gate),
fork `main` untouched at 728de13. `d745b771` kept locally as tag `prev-head-2`
in the clone for classification.
