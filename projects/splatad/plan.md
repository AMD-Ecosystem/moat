# splatad -- ROCm/HIP port plan (lead: linux-gfx90a / MI250X, ROCm 7.2.1)

## Project
- Name: splatad
- Upstream: https://github.com/carlinds/splatad (carlinds/splatad)
- Default branch: main
- Paper: "SplatAD: Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for Autonomous Driving" (CVPR 2025).
- What it is: a fork/extension of nerfstudio-project/gsplat v1.0.0 that adds a LIDAR
  rendering path (spherical-coordinate rasterization) and rolling-shutter
  compensation on top of gsplat's standard camera 3DGS rasterizer. The SplatAD model
  itself (dataloading, decoders) is NOT in this repo; it ships separately via
  neurad-studio. This repo is purely the differentiable camera + lidar rendering
  kernels. README: "Our code introduce no additional dependencies" (only gsplat's).

## Existing AMD support -> DECISION: PORT (improve an incomplete partial path)
- splatad inherits gsplat v1.0.0's partial-ROCm setup.py: `if torch.version.hip:`
  defines `USE_ROCM`, undefs `__HIP_NO_HALF_CONVERSIONS__`, and filters `"hip" not in
  path` sources. This is build-config-only and INCOMPLETE -- the kernels using
  cooperative-groups `cg::reduce`/`cg::labeled_partition`, `cub::`, and the bundled
  GLM submodule will not build under ROCm as-is (exactly the state the MOAT gsplat
  port found and fixed). No HIP path is functional today.
- No OpenCL/Vulkan/SYCL/HIP alternative renderer exists. So a ROCm/HIP port of the
  CUDA code adds real value.
- DECISIVE LEVERAGE: MOAT has ALREADY completed the gsplat port on all three targets
  (projects/gsplat: linux-gfx90a, linux-gfx1100, windows-gfx1151 all `completed` @
  sha 5cdaa15). splatad IS gsplat (v1.0.0) + lidar. Every camera-path fault class is
  already solved and documented in projects/gsplat/notes.md. This port re-applies
  that known playbook to splatad's OLDER FLAT LAYOUT and extends it over the lidar
  kernels. NOT a from-scratch port.
- IMPORTANT layout caveat: the ported gsplat is v1.5.3 (refactored: gsplat/cuda/
  include/*, Projection*.cu, build.py, Config.h module system, 3DGUT, cuda::std ->
  libhipcxx). splatad is v1.0.0 (FLAT: gsplat/cuda/csrc/{projection,rasterization,
  sh,relocation}.cu + helpers.cuh + utils.cuh + bindings.h; pybind11 m.def; NO
  build.py, NO Config.h, NO 3DGUT, NO cuda::std). So we CANNOT copy the gsplat fork's
  files; we re-derive the same fixes against the flat v1.0.0 tree. One simplification:
  splatad uses NO cuda::std anywhere, so libhipcxx is NOT needed (unlike gsplat 3DGUT).

## Build classification: torch-extension (Strategy B). EVIDENCE:
- setup.py imports `from torch.utils.cpp_extension import BuildExtension, CUDAExtension`
  and builds a single `CUDAExtension("gsplat.csrc", sources=glob("gsplat/cuda/csrc/
  *.cu")+*.cpp, ...)` gated on `BUILD_CUDA=1` (setup.py:24-93). cmdclass uses
  `BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)`.
- A second JIT path: gsplat/cuda/_backend.py calls `torch.utils.cpp_extension.load(...)`
  over the same `csrc/*.cu` if the AOT module is not importable (lazy build).
- setup.py already branches on `torch.version.hip` (defines USE_ROCM). Confirms the
  build machinery expects a ROCm torch. No find_package(Torch)/CMake. Pure Strategy B.

## Port strategy: B (torch hipify; build against ROCm torch). RATIONALE:
- Host torch is ROCm and verified on this host: torch 2.13.0a0+gitb5e90ff,
  torch.version.hip 7.2.53211, torch.cuda.is_available()=True, device "AMD Instinct
  MI250X / MI250" (gfx90a). Same env as the gsplat port.
- Torch's CUDAExtension/load() AUTO-hipifies the .cu/.cuh at build time and links the
  HIP runtime. Keep sources in CUDA spelling; fix ONLY what hipify cannot, guarded by
  USE_ROCM. No compat header, no hand-renaming (per PORTING_GUIDE Strategy B).
- This matches the four prior splatting ports (gsplat / fused-ssim / gaussian_splatting
  / LiteGS) and op43dgs -- all Strategy B.

## Camera-rasterizer WAVE CLASSIFICATION: EASY (wave-agnostic, gsplat/op43dgs-like).
- The rasterizer is the gsplat TILE-OF-(tile_size x tile_size) model: each tile is ONE
  thread block `dim3 threads={tile_size,tile_size,1}; dim3 blocks={C,H,W}`
  (rasterization.cu:1061-1062, 1328-1329, 818-819), all cross-thread exchange via
  `extern __shared__` + `block.sync()`. N_THREADS=256 (bindings.h:7).
- It is NOT the LiteGS / vanilla-3DGS `dim3(32,N)` two-32-thread-tiles-per-wavefront
  class. There is NO `__any_sync(0xffffffff,...)` per-warp early-exit in the blend
  loop, NO `blockDim.x==32` packing, NO `lane=tid&31; if(lane==0)` election. So the
  LiteGS wave64 fault cluster (guide lines 268-270: divergent __any_sync runtime fault,
  __reduce_*_sync per-tile, half2 operators/intrinsics) does NOT apply.
- Zero raw warp intrinsics: grep for `__shfl|__ballot|__any_sync|warpSize|hardcoded-32`
  across rasterization.cu + projection.cu is EMPTY. No half/half2 anywhere (all counts
  0) -- pure fp32, so no half2 enablement needed.
- The only cross-lane ops are CUDA Cooperative Groups over `cg::tiled_partition<32>`
  tiles: `cg::reduce(warp,v,cg::plus/greater)` (helpers.cuh warpSum/warpMax),
  `warp.any(valid)` (rasterization.cu:3093,4020 inside the rasterize fwd/bwd blend
  loops), and `cg::labeled_partition(warp,gid/cid)` (projection.cu, 8 sites in the
  packed/fused projection backward). HIP's tiled_partition<32> is correctly
  TILE-RELATIVE on wave64 (width=32 shuffles stay in the 32-lane group), so these are
  wave64-correct by construction -- exactly the gsplat verdict (gsplat notes "Wave64
  verdict"; guide 217/258). The shims below are HIP-CG API GAPS, not wave-size math.

## LIDAR rendering BACKEND: SPHERICAL RASTERIZATION (custom kernels) -> PORTABLE.
## NOT OptiX. NOT B7-gated. NO HIP-RT needed. The lidar half ports with the camera half.
- DECISIVE: grep of the ENTIRE repo for `optix|optixTrace|\.bvh|hiprt|gas_|raygen|
  closesthit|anyhit|__miss__|accel.*build|owl|embree` returns ZERO matches. There is
  no ray-tracing API, no BVH build/traversal, no SBT, no PTX-loaded program model.
- The only submodule is glm (header-only math, g-truc/glm). No OptiX SDK, no tcnn, no
  nerfstudio import in gsplat/ or examples/ (grep empty). README confirms no extra deps.
- The lidar path is option (c)/(b) from the brief, concretely a RASTERIZATION to
  SPHERICAL coordinates -- the same algorithm class as the camera rasterizer, with a
  pinhole projection swapped for an azimuth/elevation spherical projection:
  - `lidar_proj` (utils.cuh:281) projects gaussian means+cov to spherical coords and
    builds a 2D covariance via a spherical Jacobian (cov2d = J cov3d J^T), plus depth
    compensation. `lidar_proj_vjp` is its backward.
  - `fully_fused_lidar_projection_{fwd,bwd}`, `isect_lidar_tiles`,
    `map_points_to_lidar_tiles`, `points_mapping_offset_encode`,
    `populate_image_from_points_kernel`, `rasterize_to_indices_in_range_lidar_kernel`,
    `rasterize_to_points` (the lidar analog of rasterize_to_pixels). All follow the
    identical block-per-tile + shared-mem + cub-sort structure as the camera kernels.
  - The "rays" in test_lidar_rasterization are just a static azimuth x elevation
    raster grid of sample points (image_width = (max_azimuth-min_azimuth)/azimuth_res;
    n_elevation_channels rows on a non-linear elevation grid), each rasterized via the
    tile-blend kernel. No per-ray BVH traversal.
- Velocity/rolling-shutter: `compute_lidar_velocity` / `compute_pix_velocity` +
  their vjps (helpers.cuh) are pure glm/fp32 math. Portable.
- CONCLUSION: a STAGED EnvGS-style split is NOT required here. Both the camera path
  and the lidar path are mechanical Strategy-B ports. Land them TOGETHER. (Contrast
  EnvGS, whose reflection path was genuinely OptiX-bound and deferred; splatad has no
  such path.)

## CUDA surface inventory
Files: gsplat/cuda/csrc/{projection.cu (115KB), rasterization.cu (245KB), sh.cu,
relocation.cu, ext.cpp, bindings.h, helpers.cuh, utils.cuh} + third_party/glm.
- Kernels: SH fwd/bwd (sh.cu); quat_scale_to_covar_preci, persp_proj, world_to_cam,
  lidar_proj, compute_{pix,lidar}_velocity and vjps; fully_fused_{,lidar_}projection
  fwd/bwd (packed + non-packed); isect_tiles / isect_lidar_tiles /
  isect_offset_encode / map_points_to_lidar_tiles / points_mapping_offset_encode;
  rasterize_to_pixels fwd/bwd (camera), rasterize_to_indices_in_range{,_lidar},
  populate_image_from_points, rasterize_to_points (lidar). 33 pybind ops (ext.cpp),
  13 lidar-named.
- Cooperative Groups: `<cooperative_groups.h>` + `<cooperative_groups/reduce.h>`;
  `cg::tiled_partition<32>`, `cg::reduce` with `cg::plus`/`cg::greater` (helpers.cuh
  warpSum/warpMax, ~15 sites), `warp.any()` (2 sites), `cg::labeled_partition`
  (projection.cu, 8 sites). cg::this_grid().thread_rank() (grid-stride, portable).
- CUB (`<cub/cub.cuh>`): host-side `cub::DeviceRadixSort::SortPairs` via the CUB_WRAPPER
  macro for the per-tile isect sort (rasterization.cu: 6 call sites, begin_bit=0 so the
  cudaKDTree nonzero-begin_bit hipCUB bug does NOT apply), `cub::DoubleBuffer`, and
  `cub::BlockReduce`/`cub::BlockScan<int32_t,N_THREADS>` in the packed projection
  (projection.cu:1305,1317).
- Atomics: float `atomicAdd` for gradient scatter in the backward kernels (standard).
- `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` on
  the rasterizer kernels to raise the dynamic-shared-mem cap (rasterization.cu: ~12
  sites incl. rasterize_to_indices_in_range_kernel, _lidar_kernel,
  rasterize_to_pixels_fwd_kernel<C> templated over channel counts).
- GLM: bundled submodule (gsplat/cuda/csrc/third_party/glm) used pervasively for
  vec/mat math in device code. Currently UNINITIALIZED in the clone (0 .inl files).
- Streams: at::cuda::getCurrentCUDAStream() (torch-managed, portable).
- NO: textures/surfaces, cuBLAS/cuFFT/cuRAND/cuSPARSE/Thrust, pinned/managed memory,
  half/half2, cuda::std/libcu++, OptiX/BVH, inline PTX. (relocation.cu is pure fp32
  binomial math with a host-precomputed binom table -- trivial.)

## Risk list
- [LOW-MED, the one to watch on HW] `warp.any(valid)` in the rasterize fwd/bwd blend
  loops (rasterization.cu:3093, 4020). This is a CG TILE op (`cg::tiled_partition<32>`),
  NOT a full-wavefront `__any_sync(0xffffffff,...)`, so it is confined to the tile's 32
  lanes and is wave64-correct (gsplat validated the identical construct on gfx90a). It
  is, however, the closest analog to the LiteGS divergent-`__any_sync` HARD FAULT
  (guide 268), so it is the single highest-risk runtime item: if any rasterizer kernel
  faults at runtime (HSA_STATUS_ERROR_EXCEPTION / queue abort), diagnose with
  `AMD_SERIALIZE_KERNEL=3 AMD_LOG_LEVEL=3` and read the last ShaderName. Expectation:
  no fault (it is a tile op), but verify on hardware -- the lidar rasterize_to_points
  path is NEW vs gsplat and uses the same construct.
- [MED] HIP cooperative_groups (ROCm 7.2.1) has NO `cg::reduce`, NO `cg::plus`/
  `cg::greater`, and NO `<cooperative_groups/reduce.h>`. helpers.cuh warpSum/warpMax
  will not compile. Fix per gsplat fault #4 / guide 217+258: replace with a width-32
  butterfly `shfl_xor` all-reduce (USE_ROCM-guarded; keep cg::reduce on CUDA).
  wave64-correct (width=32 stays in the tile).
- [MED] HIP cooperative_groups has NO `cg::labeled_partition` (projection.cu packed/
  fused backward, 8 sites). Fix per gsplat fault #7 / guide 217: rebuild the labeled
  group from `warp.match_any(label)` (HIP CG DOES provide match_any) -- size()=
  __popcll(mask), rank=__popcll(mask & below-lane), masked all-reduce over same-label
  lanes, ONE atomicAdd by the lowest same-label lane. CRITICAL: do NOT use a trivial
  per-lane "self group" (every lane atomicAdds): float-atomic is associative so the
  VALUE is right, but the larger atomic FAN-OUT changes float accumulation ORDER and
  flapped gsplat's v_viewmats gradient tolerance run-to-run. Match CUDA's
  one-atomic-per-distinct-label granularity.
- [LOW] hipify maps `<cub/cub.cuh>` -> `<hipcub/hipcub.hpp>` but leaves the `cub::`
  namespace unrenamed -> cub::DeviceRadixSort/DoubleBuffer/BlockReduce/BlockScan
  undeclared. Fix per gsplat fault #6: `namespace cub = hipcub;` under USE_ROCM after
  the include (projection.cu, rasterization.cu). Verify the CUB_WRAPPER macro
  (bindings.h/helpers) resolves under the alias.
- [LOW] `cudaFuncSetAttribute(kernel, ...)` -> HIP `hipFuncSetAttribute` takes the
  kernel as `const void*` only (CUDA has a templated T* overload). Fix per gsplat
  fault #9: cast `(const void*)kernel<...>` at all ~12 sites -- portable, CUDA also
  accepts const void* (NOT USE_ROCM-guarded). Note the templated
  rasterize_to_pixels_fwd_kernel<C> sites need the cast per instantiation.
- [LOW] hipify rewrites `std::min`/`std::max` -> global `::min`/`::max`; on the HIP
  HOST pass (g++) global ::min/::max(float,float) do not exist. splatad's .cu/.cuh
  show no direct std::min/max, but GLM and torch headers can surface them on the host
  pass. If a host TU fails with "::min has not been declared", add the gsplat fault #2
  host-only (`USE_ROCM && !__HIP_DEVICE_COMPILE__`) global min/max -> std forwarding.
  Keep ready; apply only if it fires.
- [MED, numeric] `--use_fast_math` (set unconditionally in setup.py nvcc_flags). ROCm
  clang's fast-math is MORE aggressive than CUDA's and perturbs ill-conditioned
  projection/covariance gradients past the upstream test tolerances (gsplat fault #13:
  ~1 element of v_covars at a near-degenerate gaussian). splatad's setup.py hardcodes
  `--use_fast_math` for both backends. Fix: on ROCm, drop `--use_fast_math` (or map to
  a tamer `-ffp-contract=on`); rebuild and confirm the projection-gradient tests pass.
  The _backend.py JIT path already honors NO_FAST_MATH=1 -- use it there.
- [BUILD] Bundled GLM mangled by torch hipify (the biggest gsplat fault, #3). torch's
  CUDAExtension AOT hipify walks every .hpp under cwd + include_dirs and content-
  rewrites GLM headers: it DROPS GLM's .inl files (hipify copies only .hpp/.h) and
  corrupts GLM's __CUDACC__/CUDA_VERSION detection ("GLM requires CUDA 7.0",
  "scalar_constants.inl: No such file", "host fn from device fn"). GLM 1.0.x already
  detects __HIP__ and compiles verbatim under -x hip. Fix: in setup.py monkeypatch
  `hipify_python.hipify` to add the glm dir to `ignores` AND strip it from
  `header_include_dirs`; GLM then resolves pristine from source via -I. (The JIT
  load() path passes ignores only for ROCM/TORCH, so the AOT setup.py monkeypatch is
  required for `pip install`.) ALSO: the GLM submodule is currently UNINITIALIZED --
  must `git submodule update --init` first or it silently falls back to system GLM
  0.9.9 (no HIP detection, wrong errors).
- [LOW] PORTING_GUIDE 219: `cudaFuncSetAttribute` const-void* (covered above);
  std::array CTAD / .at() are __host__-only under clang/HIP -- splatad shows no
  std::array in csrc, so likely N/A, but watch the host pass.
- [LOW] Constexpr-member-through-instance not a constant expression under clang
  (gsplat fault #8). splatad v1.0.0 has no Lidars.cuh class; likely N/A. Watch for any
  `constexpr` reading a struct field; change to `const` if it fires.
- Follower (gfx1100/gfx1151, wave32): tiled_partition<32> is exactly ONE wavefront on
  RDNA, so the butterfly shfl_xor reductions and match_any LabeledGroup are correct
  there too (gsplat validated this on both). Expect no wave32 source change.

## File-by-file change list (all HIP-only behind USE_ROCM unless noted)
- setup.py: monkeypatch hipify_python.hipify to ignore + de-include the bundled GLM
  dir (gsplat fault #3); on ROCm drop `--use_fast_math` from nvcc_flags (or gate it
  off by default, opt-in via env) (fault #13). [The torch.version.hip USE_ROCM /
  __HIP_NO_HALF_CONVERSIONS__ block already exists -- keep it.]
- gsplat/cuda/csrc/helpers.cuh: replace `cg::reduce(warp, v, cg::plus/greater)` in
  warpSum/warpMax with a width-32 shfl_xor butterfly all-reduce under USE_ROCM
  (cg::reduce on CUDA). Drop/guard `<cooperative_groups/reduce.h>` include on ROCm.
- gsplat/cuda/csrc/projection.cu: `namespace cub = hipcub;` under USE_ROCM after the
  cub include; replace the 8 `cg::labeled_partition` sites with a LABELED_PARTITION
  shim (match_any-based group + one-atomic-per-label) under USE_ROCM; drop
  `<cooperative_groups/reduce.h>` on ROCm.
- gsplat/cuda/csrc/rasterization.cu: `namespace cub = hipcub;` under USE_ROCM; cast all
  `cudaFuncSetAttribute(kernel,...)` -> `(const void*)kernel` (not guarded; both the
  camera rasterize_to_pixels_fwd_kernel<C> instantiations and the lidar
  rasterize_to_indices_in_range_lidar_kernel). `warp.any()` left as-is (tile-correct).
- gsplat/cuda/csrc/sh.cu, relocation.cu: expected no change (pure math; relocation uses
  no cg/cub). Verify they compile.
- gsplat/cuda/csrc/bindings.h / ext.cpp: no change expected (pybind m.def). Verify the
  CUB_WRAPPER macro and DEVICE_GUARD/CHECK_INPUT macros compile under HIP.
- gsplat/cuda/csrc/utils.cuh: lidar_proj/vjp are glm math; expected no change. Verify.
- (host min/max shim in a shared header) ONLY if the host pass fails with ::min/::max
  (fault #2); apply minimally then.
- GLM submodule initialized (delivered by the fork checkout, not a source edit).
- NO libhipcxx needed (splatad uses no cuda::std).

## Build commands (lead, gfx90a)
Two equivalent paths; the AOT `pip install` is the validation build.

    cd projects/splatad/src
    git submodule update --init --recursive gsplat/cuda/csrc/third_party/glm

    # AOT (the one the tests import as gsplat.csrc):
    HIP_VISIBLE_DEVICES=<gcd> PYTORCH_ROCM_ARCH=gfx90a BUILD_CUDA=1 \
      NO_FAST_MATH=1 MAX_JOBS=16 \
        pip install -e . --no-build-isolation -v

Notes:
- PYTORCH_ROCM_ARCH=gfx90a scopes to the lead arch (the ROCm torch wheel otherwise
  builds gfx90a/942/950/1100). Followers pass PYTORCH_ROCM_ARCH=gfx1100 / gfx1151 with
  NO source change (the curated commit is arch-agnostic).
- NO_FAST_MATH=1 / dropping --use_fast_math on ROCm is the fault-#13 default. If the
  setup.py edit makes ROCm fast-math-off the default, the env var is redundant.
- build/run with cwd OUTSIDE /var/lib/jenkins/pytorch (that source tree shadows the
  installed torch) -- use projects/splatad/src. Pin one GCD via HIP_VISIBLE_DEVICES
  for all build + validation (host has 4 GCDs; cap at 4 concurrent GPU agents).
- If editing .cu after a build, re-run the build (torch re-hipifies); a stale hipified
  mirror is a known incremental gotcha.
- Optional CPU-only compile smoketest: docker image rocm/dev-ubuntu-24.04:7.2.4-complete
  (manual only; NOT a validation gate, NOT wired into Actions).

## Test plan
Upstream tests compare the HIP extension against a PURE-PYTORCH reference in
gsplat/cuda/_torch_impl.py (the gold oracle), so correctness is verifiable on AMD with
NO NVIDIA baseline. Tests are GPU-gated (`@skipif(not torch.cuda.is_available())`).
Run from a dir outside /var/lib/jenkins/pytorch, HIP_VISIBLE_DEVICES pinned.

GPU CORRECTNESS GATE (must pass on gfx90a):
1. CAMERA path -- tests/test_basic.py non-lidar:
     pytest tests/test_basic.py -k "(test_quat_scale_to_covar_preci or test_proj or \
       test_projection or test_fully_fused_projection_packed or test_isect or \
       test_sh or test_world_to_cam or test_persp_proj or test_compute_pix_velocity) \
       and not lidar" -v
   These exercise projection fwd+bwd (fused+packed, the LabeledGroup path), covariance,
   SH, tile intersection, velocity -- the full camera math vs the torch reference at
   the upstream tolerances. Run TWICE for determinism.
2. CAMERA rasterization -- tests/test_rasterization.py::test_rasterization (RGB / RGB+D
   / D, packed True/False, sh_degree, per_view_color) and
   tests/test_basic.py::test_rasterize_to_pixels. If any reference path imports nerfacc
   (CUDA-only wheel, _C is None on ROCm) it will ModuleNotFoundError -- that is the
   documented gsplat non-defect (the HIP kernel itself is fine). In that case
   substitute an independent compositor check (see gsplat notes [B]/[C]/[D]): forward
   render finite + range-sane, analytic d(loss)/d(colors) vs central finite-difference
   (rel err < ~2%), and 2-seed determinism (forward bit-exact; backward grads match
   within float-atomicAdd noise).
3. LIDAR path -- the KEY new surface (must pass):
   - tests/test_basic.py::test_lidar_proj, test_compute_lidar_velocity,
     test_lidar_projection, test_isect_lidar, test_map_points_to_lidar_tiles,
     test_populate_image_from_points, test_rasterize_to_points,
     test_accumulate_until_points (fwd + autograd backward vs _torch_impl reference).
   - tests/test_rasterization.py::test_lidar_rasterization (channels 3/32/128): full
     end-to-end lidar render on a synthetic azimuth/elevation scene, fwd + backward.
   - (some lidar tests need scipy: `pip install scipy`.)
4. SHORT TRAINING / gradcheck smoke: run a few backward steps through
   gsplat.rendering.lidar_rasterization and rasterization on a small scene, assert
   gradients FINITE (no NaN/Inf) and the loss decreases over ~50-100 steps. This
   guards the EnvGS-style "fwd finite + gradcheck + short training" gate from the brief.

DETERMINISM/GRADIENT POLICY (guide 273): do NOT require bitwise-identical gradients
across runs -- the backward scatters per-primitive grads via atomicAdd (nondeterministic
float order on every GPU incl. CUDA). Correct bar: (a) forward bit-deterministic
(forward nondeterminism WOULD flag a real wave64 race), (b) backward grads FINITE,
(c) grad SUM per tensor stable ~1e-9, (d) grad VALUES correct vs finite-difference.

NON-GPU REGRESSION SET (must not regress): tests/test_compression.py
(test_png_compression -- uses the PNG/sort utilities; note it still needs a CUDA
device per its skipif, but the kernel surface is the relocation/sort utils) and
tests/test_strategy.py (densification strategy ops over rasterization). The
distributed tests (tests/_test_distributed.py) are multi-GPU and out of scope for a
single-GCD gate.

VALIDATION COMPLETE when: camera (1,2) + lidar (3) tests pass on gfx90a vs the torch
reference with no NaN/Inf, the short-training gradcheck (4) is finite + decreasing, and
the non-GPU set (compression/strategy) does not regress.

## Open questions
- Does any lidar test pull nerfacc into its reference path (like the camera
  test_rasterization)? If so, substitute the independent compositor check (gsplat
  pattern) for that specific case; the lidar HIP kernel correctness is still proven by
  the _torch_impl-referenced lidar tests (test_rasterize_to_points etc.).
- splatad's gsplat v1.0.0 base predates the v1.5.3 fixes; confirm there is no
  additional v1.0.0-only construct (older cg spelling, an older CUB_WRAPPER) beyond
  the seven fault classes above. Low risk -- the surface grepped clean.
- Confirm the fast-math gradient sensitivity reproduces on splatad's projection tests
  (it did on gsplat); if a specific projection test flaps 1 element at the tight
  tolerance even with fast-math off, treat as the known float-atomic order noise (passes
  in isolation / on rerun), not a port defect -- do not loosen upstream tolerances.

## Handoff
After this plan: `python3 utils/moatlib.py set-state splatad linux-gfx90a planned
--agent planner`; commit plan.md + status.json + upstream.json; push to MOAT.
