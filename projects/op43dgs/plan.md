# op43dgs ROCm/HIP port plan (lead: linux-gfx90a)

## Project
- Upstream: `LetianHuang/op43dgs` (branch `main`); ECCV'24 "On the Error Analysis of
  3D Gaussian Splatting and an Optimal Projection Strategy" (Huang et al.). Fork: TBD
  (`AMD-Ecosystem/op43dgs`, port on a `moat-port` branch; the fork default stays a clean
  upstream mirror). Clone present at `projects/op43dgs/src/` (depth-1, read-only).
- A 3DGS variant that swaps the EWA local-affine projection for an "optimal projection"
  (project the Gaussian mean to the tangent plane, polar coords, an invertible Q*J
  Jacobian -- eq 15/18 of the paper). Otherwise it is the **canonical Inria graphdeco
  `diff-gaussian-rasterization` + `simple-knn`** codebase, in THREE camera variants.
- ext_type: **torch-extension** -> **Strategy B** (torch hipifies the `.cu`/`.cuh` at
  build time; do NOT add a compat header, do NOT hand-rename CUDA symbols).

## Existing AMD support (assessment) -> PROCEED (fresh CUDA->HIP, no prior AMD path)
Virgin CUDA, zero ROCm awareness: no `USE_ROCM` / `__HIP_PLATFORM_AMD__` / `torch.version.hip`
anywhere; `environment.yml` pins `cudatoolkit=11.6` + `pytorch=1.12.1`; README install is
CUDA-only. No OpenCL/Vulkan/SYCL path either. A ROCm/HIP port adds value.
Decision: a Strategy-B CUDA->HIP bring-up, the **wave-AGNOSTIC (gsplat / gaussian_splatting)
family**, NOT the LiteGS wave32-hardcoded family. Correctness-first MECHANICAL port guarded
by `USE_ROCM`. The kernels are bespoke EWA/optimal-projection splatting math (glm mat3, SH,
alpha compositing), NOT CUTLASS/CuTe/wgmma, so the perf-rewrite caveat does NOT apply and no
MFMA/CK/rocWMMA rewrite is warranted.

## Build classification (evidence)
torch-extension. Decisive lines (identical shape in all three variants):
- `submodules/diff-gaussian-rasterization-pinhole/setup.py`:
  `from torch.utils.cpp_extension import CUDAExtension, BuildExtension`; one
  `CUDAExtension(name="diff_gaussian_rasterization._C", sources=[rasterizer_impl.cu,
  forward.cu, backward.cu, rasterize_points.cu, ext.cpp], extra_compile_args={"nvcc":
  ["-I.../third_party/glm/"]})`; `cmdclass={'build_ext': BuildExtension}`.
- `submodules/simple-knn/setup.py`: same shape, `CUDAExtension(name="simple_knn._C",
  sources=[spatial.cu, simple_knn.cu, ext.cpp])`.
- `gaussian_renderer/__init__.py:14` `from diff_gaussian_rasterization import
  GaussianRasterizationSettings, GaussianRasterizer` -- the Python trainer calls the
  compiled op through a `torch.autograd.Function` (`_RasterizeGaussians.apply`).
Per PORTING_GUIDE "Build classification": `torch.utils.cpp_extension` + `CUDAExtension`/
`BuildExtension` => pytorch extension => Strategy B.

### Packaging fact that shapes the whole plan: THREE variants, ONE package name
`submodules/diff-gaussian-rasterization-{pinhole,fisheye,panorama}/setup.py` ALL declare
`name="diff_gaussian_rasterization"` / `"diff_gaussian_rasterization._C"`. They are
MUTUALLY EXCLUSIVE installs (only one `diff_gaussian_rasterization` resolvable at a time).
`environment.yml` installs **pinhole** + simple-knn by default; the README installs
pinhole/panorama/fisheye separately for different camera models. The three share ~95% of
the code; the ONLY substantive divergence is the projection math (`computeCov2D` /
`preprocess` in forward.cu + its analytic backward in backward.cu):
- forward.cu 622 / 688 / 699 lines (pinhole / fisheye / panorama)
- backward.cu 876 / 867 / 961
- rasterizer_impl.cu 435 / 437 / 445 (near-identical; binning/sort/range scaffolding)
So a fix to one rasterizer almost always applies verbatim to the other two. Lead-platform
scope: build + GPU-validate **all three** rasterizer variants + simple-knn, with **pinhole
as the primary/default gate** (the variant the trainer and environment.yml use).

## ROCm torch env (VERIFIED PRESENT on this host)
- conda python `/opt/conda/envs/py_3.12/bin/python`, torch ~2.13.0a0, `torch.version.hip`
  ~7.2.5, device "AMD Instinct MI250X / MI250" (gfx90a, CDNA2, wave64). Host ROCm 7.2.1,
  hipcc `/opt/rocm/bin/hipcc`. `CUDA_HOME=None`, `ROCM_HOME=/opt/rocm` -> a `CUDAExtension`
  build auto-hipifies `.cu`/`.cuh` and links `amdhip64`/`c10_hip`/`torch_hip` (Strategy B).
  4x MI250X GCD; cap 4 concurrent GPU agents (`HIP_VISIBLE_DEVICES=<free GCD>`).
- MUST build/run with cwd OUTSIDE `/var/lib/jenkins/pytorch` (that source tree shadows the
  installed torch and breaks CUDAExtension hipify). Run from the project src dir or `/tmp`.
- CONFIRMED on this host: `/opt/rocm/include/cooperative_groups/reduce.h` does NOT exist;
  `/opt/rocm/include/hip/hip_cooperative_groups.h` DOES (hipify maps `<cooperative_groups.h>`
  to it). So the `<cooperative_groups/reduce.h>` include must be guarded out (Risk 1), but
  there is nothing to replace because `cg::reduce` is never called (see below).
- NumPy 2.x is installed in this env; some prebuilt deps warn "compiled with NumPy 1.x".
  Benign for the build; if a test import crashes on it, pin `numpy<2` in a throwaway venv or
  use the deps already present. Not a porting concern.

## CUDA surface inventory (whole device tree across all 4 extensions; greps are exhaustive)

### THE KEY QUESTION -- wave-agnostic (gsplat-like, EASY) or wave32-hardcoded (LiteGS-like, HARD)?
**WAVE-AGNOSTIC. EASY. The easiest of the splatting precedents so far.** Whole-tree greps
over `submodules/**/*.{cu,cuh,h}` (excluding `third_party/glm`):
- raw warp intrinsics `__shfl* / __ballot / __any / __all / __activemask / __syncwarp /
  __reduce_*_sync / __popc / __match* / warpSize`: **ZERO hits.**
- inline PTX (`asm(` / `asm volatile`): **ZERO.** (Contrast LiteGS raster.cu ex2.approx.f16x2.)
- half2 SIMD / compare-masks (`half2`, `__hle2/__hge2/__hgt2/__hmin2/__hmax2`, `h2exp`,
  `h2rcp`, `__vcmp*`): **ZERO.** (Contrast LiteGS half2 alpha-blend.)
- `cg::reduce` / `cg::tiled_partition` / `cg::thread_block_tile<32>` / `cg::labeled_partition`:
  **ZERO.** (This is EVEN SIMPLER than joeyan/gaussian_splatting, which DID use
  `cg::reduce` over a `tiled_partition<32>`; op43dgs does NOT.)
- The ONLY Cooperative Groups use is grid/block scaffolding (the stock Inria pattern):
  `cg::this_grid().thread_rank()` (per-element global index), `cg::this_thread_block()` +
  `block.sync()` + `block.thread_rank()` (the per-tile render loop). HIP CG provides all of
  these. The render/backward tile loop is a `__shared__` chunk cache (`collected_id /
  collected_xy / collected_conic_opacity / collected_colors`) blended with `block.sync()`
  (== `__syncthreads()`) ONLY -- NO warp shuffle, NO warp reduction, NO lane math anywhere.
Conclusion: the rasterizer is wave-size-AGNOSTIC by construction (block 16x16 = 256 threads
= 4 wavefronts on wave64; all cross-thread communication is block-wide `__shared__` +
`block.sync()`). NO wave64 lane-math rework is required. None of the LiteGS/popsift wave64
traps (positional 2-rows-per-block packing, `lane=tid&31` leader election, divergent
`__any_sync`, whole-wavefront `__reduce_*_sync`, half2 masks, PTX) are present.

### Per-kernel rundown (the same set in each of the 3 rasterizer variants + simple-knn)
- `forward.cu`:
  - `computeColorFromSH` (`__device__`, glm vec3, SH eval, `glm::max`). Per-Gaussian.
  - `computeNaiveCov2D` -- the ORIGINAL EWA projection, kept "only to determine the tiles
    involved" (comment forward.cu:74). Per-Gaussian glm mat3.
  - `computeCov2D` -- **op43dgs's OPTIMAL PROJECTION** (the paper's contribution): tangent-
    plane projection, `atan2/sin/cos/sqrt`, the invertible Q*J Jacobian as a `glm::mat3`.
    Pure per-Gaussian math; NO warp ops, NO atomics, NO shared mem. (panorama adds
    `KERNEL_SIZE 0.3` in config.h; fisheye/panorama vary this kernel's body.)
  - `preprocessCUDA<C>` (`__global__`) -- per-Gaussian projection + frustum cull + tile-
    touch count. Launch `(P+255)/256, 256`. No warp ops.
  - `renderCUDA<C>` (`__global__`) -- the tile rasterizer, block (BLOCK_X=16, BLOCK_Y=16).
    `__shared__` chunk cache + `block.sync()` alpha compositing. The `__expf`/`expf`
    Gaussian falloff. No warp ops, no atomics (forward accumulates in registers, writes
    once per pixel).
- `backward.cu`:
  - `computeCov2DCUDA` / `preprocessCUDA` (`__global__`) -- analytic gradient of the optimal
    projection + SH backward (glm mat3 chain rule). Per-Gaussian.
  - `renderCUDA<C>` (`__global__`) -- backward rasterizer, block 16x16, `__shared__` cache +
    `block.sync()`. Gradients scattered to global via **`atomicAdd`** (see Atomics). No warp
    reduction -- each thread atomicAdds its per-pixel contribution directly (the stock Inria
    backward; NOT the joeyan per-tile warp-reduce-then-leader-atomic pattern).
- `rasterizer_impl.cu`:
  - `getHigherMsb` (host helper -> the radix-sort end_bit).
  - `duplicateWithKeys` / `identifyTileRanges` / `checkFrustum` / `markVisible` (`__global__`,
    `cg::this_grid().thread_rank()` indexing). No warp ops.
  - cub `DeviceScan::InclusiveSum` (tile-touch prefix) + cub `DeviceRadixSort::SortPairs`
    (sort the (tile|depth) keys). See Libraries.
- `simple-knn` (`simple_knn.cu` / `spatial.cu`): Morton-order kNN. `coord2Morton` /
  `boxMinMax` / `boxMeanDist` (`__global__`), `cg::this_grid().thread_rank()`. cub
  `DeviceReduce::Reduce` (CustomMin/CustomMax) + cub `DeviceRadixSort::SortPairs` + thrust
  `device_vector` / `sequence`. `distCUDA2` is the public op. No warp ops.

### Atomics
Only `atomicAdd`, all float, in each variant's `backward.cu` renderCUDA: `dL_dmean2D.{x,y}`,
`dL_dconic2D.{x,y,w}`, `dL_dopacity`, `dL_dcolors[..]` (per-channel). All on **plain torch
device tensors** (`hipMalloc` device memory, NOT `hipMallocManaged`). The cudaKDTree managed-
memory `atomicMin/Max`-silently-dropped class is **N/A** (no atomicMin/Max, no managed
memory). float `atomicAdd` is well-supported on gfx90a. Accumulation ORDER across threads is
non-deterministic on ANY GPU (CUDA included) -> the backward grads are deterministic only to
float-atomic noise (same bar as gsplat/LiteGS; this is the stock Inria backward, so the AMD
granularity already matches CUDA exactly -- no fan-out, so no new tolerance flap expected).

### Libraries / sort
- cub (each rasterizer `rasterizer_impl.cu`): `DeviceScan::InclusiveSum`,
  `DeviceRadixSort::SortPairs(..., binningState.point_list_keys_unsorted, ...,
  binningState.point_list_unsorted, ..., num_rendered, 0, 32 + bit)`.
  **begin_bit = 0** (full-width-from-bit-0 sort; end_bit = `32 + getHigherMsb(tile_grid)`).
  This is NOT the cudaKDTree nonzero-begin_bit hipCUB bug (that bug needs a NONZERO begin_bit;
  here begin_bit is 0). hipCUB is a drop-in for these.
- cub (`simple_knn.cu`): `DeviceReduce::Reduce` with `CustomMin`/`CustomMax` functors +
  `DeviceRadixSort::SortPairs(..., P)` (begin_bit/end_bit defaulted -> full key). hipCUB
  drop-in. Watch the cudf "functor returning a reference to a by-value param" UB class:
  `CustomMin`/`CustomMax` here return BY VALUE (stock Inria), so that class is N/A -- but the
  porter should eyeball them once.
- thrust (`simple_knn.cu`): `device_vector` / `sequence`. rocThrust is a true drop-in (same
  `thrust::` API + header paths under /opt/rocm/include; cudaKDTree lesson).
- hipify maps `<cub/cub.cuh>` -> `<hipcub/hipcub.hpp>` but leaves the `cub::` NAMESPACE
  unrenamed -> `cub::DeviceRadixSort` undeclared on the ROCm path (gsplat fault #6 / cudaKDTree
  alias lesson). Fix if it surfaces: `namespace cub = hipcub;` under `USE_ROCM` after the
  include, in `rasterizer_impl.cu` and `simple_knn.cu`. (May already be handled by torch's
  hipify mapping table; try the stock build first and add the alias only if the namespace is
  reported undeclared.)
- NO cuBLAS/cuFFT/cuRAND/cuSPARSE/cuDNN.

### Textures / surfaces / managed / pinned / streams / events
**NONE** (whole-tree grep empty). So no texture rule-of-five, no 256B pitch, no layered
array, no linear-filter, no cudaArray, no stream/event fault classes apply.

### GLM (vendored, the gsplat hazard)
Each rasterizer vendors `third_party/glm/` = **GLM 0.9.9.9** (GLM_VERSION 999; setup.hpp).
`setup.py` adds `-I.../third_party/glm/`. forward.cu/backward.cu use `glm::vec3/vec4/mat3`,
`glm::transpose`, `glm::length`, `glm::max` heavily in `__device__` code. Two facts:
- This 0.9.9.9 DOES have a `GLM_COMPILER_HIP` path (setup.hpp:147 etc.), so it CAN emit
  device qualifiers under hipcc WITHOUT defining `__CUDACC__` (unlike LichtFeld's 0.9.9.8).
  So NO `__CUDACC__`/`GLM_FORCE_CUDA` hack is needed (and it would break rocThrust anyway).
- BUT the gsplat hazard is version-independent: torch's AOT hipify walks every `.hpp` under
  cwd + the extension include dirs into its file set and content-rewrites the bundled GLM
  headers -- it copies only `.hpp/.h` (DROPS GLM's **139** `.inl` files here) and corrupts
  GLM's compiler detection -> "scalar_constants.inl: No such file", "GLM requires CUDA ...",
  "no matching function for length/...". This is Risk 3.

### __CUDA_ARCH__ / fast-math / std::min-max
- NO `__CUDA_ARCH__` device/host `#ifdef` branching in the device tree (the cudaKDTree class
  is N/A). `min`/`max` in computeCov2D are on plain floats (`min(limx, max(-limx, txtz))`);
  the CV-CUDA NaN-selection class is about random-byte buffers and bit-exact gold -- N/A for a
  render/training correctness gate.
- setup.py passes only the GLM `-I` (NO `--use_fast_math`, NO nvcc-only flags). So the gsplat
  fast-math gradient-edge issue is N/A. clang-HIP defaults `-ffp-contract=fast` vs nvcc
  expression-only (CV-CUDA class): irrelevant to a finite/PSNR gate; only matters for a
  bit-exact compare, which this repo has none of. Try stock; pin `-ffp-contract=on` only if a
  numeric comparison ever needs it (not expected).

## Risk list (ranked; each tied to the precedent that solved it)
1. **`<cooperative_groups/reduce.h>` include fails on ROCm** -- HIGH (certain compile wall,
   trivial fix). CONFIRMED absent in ROCm 7.2.1. Present in all 4 extensions
   (rasterizer forward.cu/backward.cu/rasterizer_impl.cu x3 + simple_knn.cu). hipify leaves
   the include verbatim. Fix: guard it `#if !defined(USE_ROCM)`. UNLIKE gsplat/joeyan, there
   is **nothing to replace** -- `cg::reduce` is never called -- so this is a pure include
   guard, no butterfly needed. Keep `<cooperative_groups.h>` (hipify -> HIP CG, which provides
   `this_grid`/`this_thread_block`/`thread_rank`/`sync`). (Precedent: gsplat fault #4 / LiteGS
   include guard; here it's the include-only half.)
2. **MUSA-style spaced launch syntax `<< <` / `>> >`** -- HIGH (certain compile wall across
   ALL 4 extensions). Confirmed: **27** kernel launches use the spaced form (`renderCUDA<C>
   << <grid, block>> >`, `coord2Morton << <...>> >`, `identifyTileRanges << <...>> >`, ...) and
   **ZERO** use the standard `<<<`. clang-HIP's parser rejects `name << <...>> >` with
   "expected expression"; nvcc tolerates it. (Note: this is upstream Inria style here, not the
   MooreThreads reformat, but the clang-HIP failure is identical.) Fix: normalize every
   `<< <` -> `<<<` and `>> >` -> `>>>` (correctness-neutral on both backends). (Precedent:
   LiteGS "Launch-syntax normalization", 35 sites.) These are extension `.cu` sources, so
   hipify runs on them but does NOT fix the spacing -- it must be a source edit.
3. **Bundled GLM mangled by torch AOT hipify** -- HIGH (the gsplat "biggest one"). hipify
   drops GLM's 139 `.inl` files and corrupts GLM's compiler detection. Fix (gsplat precedent,
   applied PER rasterizer setup.py, x3): in each `setup.py`, monkeypatch
   `torch.utils.hipify.hipify_python.hipify` to (a) add the extension's `third_party/glm` dir
   to `ignores` and (b) strip it from `header_include_dirs`, leaving GLM pristine and resolved
   from source via the existing `-I`. (torch's AOT hipify call forwards no `ignores`, so it
   must be injected at the setup.py level.) simple-knn vendors no GLM, so its setup.py is
   untouched. This is a build-only (setup.py) change, still within Strategy B.
   - First, try the stock build: it is POSSIBLE 0.9.9.9's HIP path + this ROCm's hipify behave
     better than gsplat's 1.0.2 case did; but the `.inl`-drop is mechanical and version-
     independent, so expect to need the monkeypatch. Confirm by the exact error
     ("scalar_constants.inl: No such file" / "no matching function for glm::length").
4. **`cub::` namespace undeclared after hipify maps the include** -- MEDIUM. hipify may map
   `<cub/cub.cuh>` -> `<hipcub/hipcub.hpp>` but leave `cub::DeviceRadixSort` / `cub::DeviceScan`
   / `cub::DeviceReduce` unrenamed. Fix if reported undeclared: `namespace cub = hipcub;`
   under `USE_ROCM` in `rasterizer_impl.cu` (x3) and `simple_knn.cu`. (Precedent: gsplat #6 /
   cudaKDTree alias.) Try stock first.
5. **gradient atomicAdd order nondeterminism** -- LOW/MEDIUM. backward grads are deterministic
   only to float-atomic noise (universal to every GPU, CUDA included). This is the STOCK Inria
   backward (direct per-thread atomicAdd, NO warp-reduce-then-leader), so the AMD atomic
   granularity already equals CUDA -- the gsplat "match the CUDA atomic count, don't fan out"
   flake does NOT apply (there is no labeled_partition/fan-out here). Validate via FD-slope +
   sign-agreement on the largest grads + grad-SUM stability (the LiteGS Tier-2 bar), not a
   bitwise rerun. If a check is noisy, it's float-atomic order, not a wave64 bug.
6. **PYBIND11_MODULE on the gfx1151 follower** -- LOW (follower-only, NOT lead). `ext.cpp` uses
   `PYBIND11_MODULE`. On Windows/gfx1151 the pybind `at::Tensor` caster can hit the c10
   inherited-ctor dllexport gap (the fused-ssim gfx1151 blocker; gsplat dodged it via
   TORCH_LIBRARY). Note for the gfx1151 delta-plan, do not act now.
7. **`expf`/`__expf` accuracy vs CUDA** -- LOW. renderCUDA uses the exp falloff. There is no
   bit-exact gold in this repo, only finite/PSNR/FD gates, which absorb a ULP. N/A unless a
   numeric compare is added.

## File-by-file change list (all source edits guarded `#if defined(USE_ROCM)`; CUDA byte-identical)
Applied to EACH of the three rasterizer variants (pinhole primary; fisheye + panorama get the
SAME edits -- the files are ~95% identical), plus simple-knn:
- `submodules/diff-gaussian-rasterization-<v>/cuda_rasterizer/forward.cu`,
  `.../backward.cu`, `.../rasterizer_impl.cu`:
  - guard `#include <cooperative_groups/reduce.h>` under `#if !defined(USE_ROCM)` (Risk 1).
  - normalize all `<< <`/`>> >` -> `<<<`/`>>>` (Risk 2).
  - IF cub namespace is reported undeclared: `namespace cub = hipcub;` under `USE_ROCM` after
    the cub include in `rasterizer_impl.cu` (Risk 4).
- `submodules/simple-knn/simple_knn.cu`:
  - guard `#include <cooperative_groups/reduce.h>` under `USE_ROCM` (Risk 1).
  - normalize `<< <`/`>> >` -> `<<<`/`>>>` (3 sites; Risk 2).
  - IF needed: `namespace cub = hipcub;` (Risk 4).
- `submodules/diff-gaussian-rasterization-<v>/setup.py` (x3): add the hipify `ignores` +
  `header_include_dirs`-strip monkeypatch protecting `third_party/glm` (Risk 3). Build-only.
  IF torch does not auto-define `USE_ROCM` for the HIP compile (it normally does -- confirmed
  for gaussian_splatting/LiteGS), add `-DUSE_ROCM` to the HIP branch here; verify at build.
- NO compat header, NO CUDA-symbol renames (Strategy B). NO CMake (none exists). NO GitHub
  Actions; after `gh repo fork`, disable Actions on the fork
  (`gh api -X PUT repos/AMD-Ecosystem/op43dgs/actions/permissions -F enabled=false`).
- The optimal-projection math (`computeCov2D` fwd + its backward) needs NO HIP-specific edit:
  it is plain glm/`atan2`/`sin`/`cos`/`sqrt` per-Gaussian arithmetic. Do NOT touch the math.

## Build commands (gfx90a; from the project src dir or /tmp, NOT /var/lib/jenkins/pytorch)
```
export HIP_VISIBLE_DEVICES=<free GCD>      # cap 4 concurrent GPU agents (rocm-smi --showuse)
export PYTORCH_ROCM_ARCH=gfx90a            # follower: gfx1100 / gfx1151, no source change
export MAX_JOBS=16
P=/opt/conda/envs/py_3.12/bin/python
cd /var/lib/jenkins/moat/projects/op43dgs/src

# simple-knn (shared by all camera variants)
rm -rf submodules/simple-knn/build submodules/simple-knn/*.egg-info submodules/simple-knn/**/*.hip
$P -m pip install ./submodules/simple-knn --no-build-isolation --no-deps -v

# rasterizer -- ONE variant at a time (they share the package name diff_gaussian_rasterization)
# PRIMARY gate = pinhole (environment.yml default + the trainer's import):
rm -rf submodules/diff-gaussian-rasterization-pinhole/build .../*.egg-info .../**/*.hip
$P -m pip install ./submodules/diff-gaussian-rasterization-pinhole --no-build-isolation --no-deps -v
# then, separately (uninstall the previous first; same module name), repeat for:
#   ./submodules/diff-gaussian-rasterization-panorama
#   ./submodules/diff-gaussian-rasterization-fisheye
```
Torch auto-hipifies each `.cu`/`.cuh` (gitignored `.hip` mirror under build/) and links
`amdhip64`/`c10_hip`/`torch_hip`. The module is `diff_gaussian_rasterization._C` (and
`simple_knn._C`). INCREMENTAL gotcha (Strategy B): after editing a `.cu`, delete that
extension's `build/` and any `*.hip` mirror before rebuilding, or torch recompiles the STALE
hipified mirror. The arch comes from `PYTORCH_ROCM_ARCH`; a follower needs only
`PYTORCH_ROCM_ARCH=gfx1100` (or gfx1151) + clean rebuild, no source edit (no head_sha churn).
Add `*.hip`, `*_hip.cuh`, `build/`, `*.egg-info` to `.gitignore` so hipify artifacts are not
committed (LiteGS/gsplat precedent). Install Python deps the trainer needs (NON-torch only,
`--no-deps` so the host ROCm torch wins; do NOT install environment.yml's torch==1.12.1):
`plyfile tqdm` + (for the optional training/eval) `opencv-python torchvision`.

## Test plan (real GPU, gfx90a) -- there is NO formal test suite; mirror gsplat/LiteGS gates
This is a research training repo: NO `test_*.py`, no pytest. Validation is built from the
public ops, exactly as LiteGS/gsplat were closed (forward finite + autograd/FD backward +
short training loss-down/PSNR-up). Write the harnesses under `agent_space/op43dgs_*.py`.

### Tier 1 -- forward render correctness (per variant; pinhole primary)
- Build a small synthetic scene (or a handful of random Gaussians) and call the public path
  `GaussianRasterizer` (gaussian_renderer-style) / `_RasterizeGaussians.apply`. Assert the
  rendered image is FINITE (no NaN/Inf), non-trivial (nonzero coverage), correct shape, and
  **bitwise-deterministic across two runs** (forward has NO output atomics -> must be
  bit-identical; a non-deterministic forward would fingerprint a wave64 race -- not expected
  here since there are no warp ops). simple-knn: `distCUDA2` on a random cloud is finite and
  bitwise-deterministic (the LiteGS simple-knn gate).
- This is the decisive gate that the projection kernels + the `__shared__`/`block.sync()`
  tile loop are correct on wave64.

### Tier 2 -- backward gradient correctness (the strongest single gate)
- Finite-difference check on the rasterizer backward: perturb inputs (means3D / opacities /
  sh / scales) and compare the analytic HIP grad against central differences of the HIP
  forward -- slope ~1.0, ~100% sign agreement on the largest-magnitude grads, low median rel
  err; grad SUMs (order-independent) stable across runs to ~1e-6. (LiteGS Tier-2 bar.) The
  optimal-projection backward (`computeCov2DCUDA`) is the op43dgs-specific code path this
  exercises. A `torch.autograd.gradcheck` in float64 is the ideal oracle IF the kernels carry
  a double instantiation; the Inria rasterizer is float-only, so prefer the FD check at fp32
  with a coarse eps (gradcheck's default float64 path will not apply to a float-only kernel).
- Run-to-run grad variation at ~float-atomic-noise level is benign (Risk 5), universal to CUDA.

### Tier 3 -- end-to-end short training (loss-down / PSNR-up)
- `train.py` loss is `(1-lambda_dssim)*L1 + lambda_dssim*(1-SSIM)` (train.py:91-92), metric
  `psnr` (utils/image_utils.py). Run a few hundred iters on a small scene and assert loss
  DECREASES and PSNR INCREASES monotone-ish, no NaN. (gsplat/LiteGS Tier-3 bar.)
  - If a real COLMAP / Mip-NeRF360 scene is available on the host, use it via the README
    entry. If not, do a self-contained SYNTHETIC multi-view fit (the LiteGS Tier-3 approach: a
    controlled ground-truth scene is a STRONGER correctness oracle than a stock dataset run).
    `convert.py` (COLMAP) is not needed for the synthetic path.

### "Validated on gfx90a" (concrete bar)
(a) **pinhole** (primary): Tier 1 forward finite + bitwise-deterministic across 2 runs; Tier 2
FD backward slope ~1.0 + sign agreement + grad-SUM stability; Tier 3 short training loss-down
& PSNR-up, no NaN. AND simple-knn `distCUDA2` finite + deterministic. THEN
(b) **panorama** and **fisheye**: at least the Tier-1 forward-finite + bitwise-deterministic
gate (their only divergence from pinhole is the projection math, which Tier 1 exercises), and
a Tier-2 FD spot-check on each. Builds must succeed for all three.
NO non-GPU regression set exists (no CPU unit tests in the repo); "no regression" = the stock
Python trainer/eval still import and run.

## Staged strategy + likely walls (keyed to the EASY gsplat/gaussian_splatting precedent)
1. **simple-knn first** (shared, smallest). Stock build expects exactly the two compile walls:
   the `<cooperative_groups/reduce.h>` include (Risk 1) and the spaced launches (Risk 2);
   maybe the `cub::` alias (Risk 4). Fix, rebuild (delete build/ for re-hipify), validate
   `distCUDA2` finite + deterministic.
2. **pinhole rasterizer.** Stock build expects: the same include + spaced-launch walls in
   forward/backward/rasterizer_impl, PLUS the bundled-GLM hipify mangling (Risk 3 -- apply the
   gsplat setup.py monkeypatch). Get a clean compile of `diff_gaussian_rasterization._C`.
3. **Run Tier 1 -> Tier 2 -> Tier 3 on pinhole.** Wave-agnostic expectation: they pass with
   NO wave64 source rework (gsplat/gaussian_splatting/fused-ssim all needed zero lane-math
   work once the include/GLM/launch walls were cleared). If a forward render is NOT bitwise-
   deterministic, that would be the surprise -- re-examine for any hidden warp dependence (none
   found in the greps). If a backward FD check is noisy, it's float-atomic order (Risk 5),
   confirm via grad-SUM stability before suspecting a bug.
4. **panorama + fisheye:** apply the identical edits (the diffs are mechanical), build each
   (uninstall the prior same-named module first), run the Tier-1 forward gate + a Tier-2 spot
   check on each.
Likely-wall summary: the ENTIRE port is expected to be (a) one include guard, (b) the spaced-
launch normalization, (c) the gsplat bundled-GLM setup.py monkeypatch, (d) maybe the `cub=`
alias -- all build-time, all USE_ROCM-guarded or build-only, CUDA path byte-identical. NO
wave64 lane-math, NO PTX, NO half2, NO cg::reduce replacement. This is the EASIEST of the four
splatting precedents on the device-code axis (joeyan needed a cg::reduce butterfly; op43dgs
needs none), with the gsplat GLM-bundle wrinkle and the LiteGS spaced-launch wrinkle added by
the vendored-GLM + Inria-MUSA-style source. No AMD-native (MFMA/CK/rocWMMA) rewrite warranted.

## Open questions
- Does torch's AOT hipify mangle the vendored GLM **0.9.9.9** the same way it mangled gsplat's
  1.0.2 (drop .inl + corrupt detection)? Expected yes (the .inl-drop is version-independent);
  resolve by trying the stock build and reading the exact error, then apply the monkeypatch.
- Does torch define `USE_ROCM` automatically for these CUDAExtension HIP compiles (so the
  `#if defined(USE_ROCM)` include guards fire without help)? gaussian_splatting/LiteGS
  confirmed yes; verify in the op43dgs build log, else add `-DUSE_ROCM` in the setup.py HIP
  branch.
- Does hipify rename the `cub::` namespace here or only the include (deciding whether the
  `namespace cub = hipcub;` alias is needed)? Resolve at compile.
- Is a real COLMAP/Mip-NeRF360 scene present on the host for Tier 3, or do we use a synthetic
  multi-view fit (LiteGS approach)? Either satisfies the loss-down/PSNR-up gate.
- panorama's `KERNEL_SIZE 0.3` and the fisheye projection: confirm their forward/backward FD
  checks pass (same wave-agnostic structure; the math differs but the porting surface does not).

## Disposition
Strategy B mechanical bring-up + GPU validation. The wave-AGNOSTIC (gsplat / gaussian_splatting
/ fused-ssim) family -- the EASY class -- here the canonical Inria diff-gaussian-rasterization
(3 camera variants sharing the package name `diff_gaussian_rasterization`) + simple-knn, with
op43dgs's optimal-projection `computeCov2D` swapped into forward/backward. The whole device
surface is `__shared__` + `block.sync()` tile compositing + per-Gaussian glm/SH math +
float `atomicAdd` gradients + cub/thrust sort (begin_bit 0) -- ZERO raw warp intrinsics, ZERO
PTX, ZERO half2, ZERO cg::reduce, ZERO textures/managed memory. None of the LiteGS/popsift
wave64 traps apply. The fixes are all build-time / include-guard: (1) guard out
`<cooperative_groups/reduce.h>`; (2) normalize the MUSA/Inria spaced launches `<< <`->`<<<`;
(3) the gsplat bundled-GLM hipify-protection monkeypatch in each rasterizer setup.py; (4)
maybe `namespace cub = hipcub;`. Correctness-first; no MFMA/CK rewrite. Record quirks in
notes.md; likely no NEW PORTING_GUIDE entry (re-confirms the existing gsplat GLM-bundle, LiteGS
spaced-launch, and cg-reduce-include items) unless the GLM-0.9.9.9-under-this-hipify behavior
differs from the recorded gsplat-1.0.2 case, which would be worth a one-line changelog note.
