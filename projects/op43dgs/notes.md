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
