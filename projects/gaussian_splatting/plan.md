# gaussian_splatting ROCm/HIP port plan (lead: linux-gfx90a)

## Project
- Upstream: `joeyan/gaussian_splatting` @ `8ddaa799846ab75810305f457daf36668b68bddf` (branch `main`).
- "A from-scratch re-implementation of 3D Gaussian Splatting for Real-Time Radiance
  Field Rendering" (Kerbl/Kopanas) -- MIT license. A PyTorch CUDA extension
  (`splat_cuda`) plus a pure-Python training/dataloader/optimizer layer (`splat_py/`).
- NOTE: this is NOT the canonical Inria/graphdeco `diff-gaussian-rasterization` +
  `simple-knn` submodule repo. It is a single self-contained `CUDAExtension` named
  `splat_cuda` built from `src/*.cu` + `src/bindings.cpp` via `setup.py` +
  `BuildExtension`. There are NO git submodules, NO diff-gaussian-rasterization, NO
  simple-knn, and the per-tile depth sort is done in PyTorch ATen (`torch::sort`),
  not cub. So the LiteGS three-extensions/simple-knn shape does NOT apply here.
- ext_type: **torch-extension** -> **Strategy B** (torch hipifies the `.cu`/`.cuh` at
  build time; do NOT add a compat header or hand-rename symbols).

## Existing AMD support (assessment) -> PROCEED (fresh CUDA->HIP, no prior AMD path)
Virgin CUDA codebase, NO ROCm awareness:
- `setup.py` builds one `CUDAExtension(name="splat_cuda", sources=[bindings.cpp,
  depth.cu, precompute_sh.cu, projection.cu, projection_backward.cu, render.cu,
  render_backward.cu, tile_culling.cu])` with `cmdclass={"build_ext": BuildExtension}`
  and PLAIN flags `c_flags=nvcc_flags=["-O3","-std=c++17"]`. There is NO
  `torch.version.hip` branch (unlike fused-ssim/gsplat which already had one).
- No `#if defined(USE_ROCM)` / `__HIP_PLATFORM_AMD__` guards in any `.cu`/`.cuh`.
- README "Installation" says "requires CUDA ... from developer.nvidia.com"; no
  AMD/OpenCL/Vulkan/HIP mention anywhere.
Decision: a full Strategy-B CUDA->HIP bring-up. This is the direct sibling of the
gsplat and fused-ssim ports (the wave-AGNOSTIC playbook), NOT the LiteGS one (no raw
warp intrinsics / no inline PTX / no half2 SIMD here). Disposition is a
correctness-first MECHANICAL port guarded by `USE_ROCM`; the kernels are bespoke
EWA-splatting math, not CUTLASS/CuTe/wgmma, so the perf-rewrite caveat does not apply
and no MFMA/CK rewrite is warranted.

## Build classification (evidence)
torch-extension. Decisive lines:
- `setup.py:2` `from torch.utils.cpp_extension import CUDAExtension, BuildExtension`;
  `:9-21` the single `CUDAExtension(name="splat_cuda", sources=[...8 files...])`;
  `:27` `cmdclass={"build_ext": BuildExtension}`.
- `splat_py/cuda_autograd_functions.py:3` `from splat_cuda import (...)` -- the Python
  layer calls the compiled ops through `torch.autograd.Function` wrappers.
Per PORTING_GUIDE "Build classification": presence of `torch.utils.cpp_extension` +
`CUDAExtension`/`BuildExtension` => pytorch extension => Strategy B.

## ROCm torch env (VERIFIED PRESENT on this host)
- conda env `py_3.12`, python `/opt/conda/envs/py_3.12/bin/python`. torch
  **2.13.0a0+gitb5e90ff**, `torch.version.hip` **7.2.53211**,
  `torch.cuda.is_available()=True`, device "AMD Instinct MI250X / MI250" (gfx90a).
- Host ROCm **7.2.1**, hipcc `/opt/rocm/bin/hipcc`. `CUDA_HOME=None`,
  `ROCM_HOME=/opt/rocm` -> a `CUDAExtension` build auto-hipifies the `.cu`/`.cuh` and
  links `amdhip64`/`c10_hip`/`torch_hip` (Strategy B). 4x MI250X GCD (wave64), ids 0-3.
- MUST build/run with cwd OUTSIDE `/var/lib/jenkins/pytorch` (that source tree shadows
  the installed `torch`); run from the project src dir or `/tmp`. Pin a free GCD via
  `HIP_VISIBLE_DEVICES` (cap 4 concurrent GPU agents; check `rocm-smi --showuse`).
- CONFIRMED: `find /opt/rocm/include -name reduce.h -path '*cooperative*'` is EMPTY
  -- ROCm 7.2.1 ships `<hip/hip_cooperative_groups.h>` but NO
  `<cooperative_groups/reduce.h>` and NO `cg::reduce`. This is the one real fix
  (see Risk 1), identical to gsplat fault #4.

## CUDA surface inventory (8 source files; the whole device surface)

### THE KEY QUESTION -- wave-agnostic (gsplat-like) or wave32-hardcoded (LiteGS-like)?
**WAVE-AGNOSTIC, gsplat-like. EASY.** Whole-tree greps over `src/*.cu`+`*.cuh`:
- `__shfl* / __ballot / __any / __all / __activemask / __syncwarp / __reduce_*_sync /
  __popc / warpSize / __match*`: **ZERO hits.** No raw warp intrinsics anywhere.
- inline PTX (`asm(`/`asm volatile`): **ZERO.** (Contrast LiteGS raster.cu ex2.approx.)
- half2 SIMD / compare-masks (`half2`, `__hle2`, `__hge2`, `__hgt2`, `__hmin2`,
  `__hmax2`, `h2exp`, `h2rcp`, `__vcmp*`): **ZERO.** (Contrast LiteGS half2 alpha-blend.)
- The ONLY warp-level construct is Cooperative Groups with an explicit 32-thread tile,
  in `render_backward.cu` ONLY:
  - `:1-2` `#include <cooperative_groups.h>` + `#include <cooperative_groups/reduce.h>`,
    `:10` `namespace cg = cooperative_groups;`
  - `:33-34` `auto block = cg::this_thread_block(); cg::thread_block_tile<32> warp_cg =
    cg::tiled_partition<32>(block);`
  - `:244-246,253,259` `cg::reduce(warp_cg, <grad>, cg::plus<T>())` over grad_opa /
    grad_u / grad_v / grad_sh[..] (per channel x N_SH) / grad_conic_splat[0..2].
  - `:263` leader election `if (warp_cg.thread_rank() == 0) atomicAdd(grad_* , ...)` --
    one rank-0 per 32-tile combines the per-tile partial into the global gradient.
This is EXACTLY the gsplat pattern (tiled_partition<32> + cg::reduce + rank-0
atomicAdd). It is wave-size-AGNOSTIC by construction: `tiled_partition<32>` requests a
32-thread subgroup regardless of hardware wave width. The backward launches
`block_size(16,16,1)` = 256 threads (`render_backward.cu:386`), a multiple of 32, so on
gfx90a (wave64) the 256-thread block splits into 4 wavefronts each holding 2 tiles of
32; each tile<32> `cg::reduce` and `thread_rank()==0` operate per 32-lane group, so each
tile behaves like a 32-lane NVIDIA warp and each tile's rank-0 does its own atomicAdd.
NO positional 2-rows-per-block packing (popsift trap) and NO `lane=tid&31; if(lane==0)`
hand-rolled election (popsift ballot trap) are present. So NO wave64 lane-math rework is
required; the single change is supplying a `cg::reduce` equivalent on HIP (Risk 1).

### Per-kernel rundown (21 host wrappers / `__global__` kernels across 8 files)
- `render.cu` -- forward rasterizer `render_tiles_kernel<T, CHUNK_SIZE, N_SH>`. Block
  (16,16)=256. Per-tile alpha-blend over a `__shared__` chunk cache (`_uvs/_opacity/
  _rgb/_conic/_image`) with `__syncthreads()` only (NO warp ops). `__expf` fast-exp
  path. Instantiated float {N_SH=1,4,9,16} and double {1,4,9,16}. Pure shared-mem +
  block-sync reduction; wave-agnostic.
- `render_backward.cu` -- backward rasterizer (above). The ONLY `cg::reduce` site. float
  AND double instantiations {N_SH=1,4,9,16} -- so the float64 gradcheck tests genuinely
  exercise the reduce in double precision (the replacement MUST template over T;
  `__shfl_xor` supports double on HIP). `_gaussian_idx_by_splat_idx` (`:71`) is a
  `__shared__` cache; the `[i]` read at `:264` is correct, not a bug.
- `tile_culling.cu` -- `compute_num_splats_kernel` (per-gaussian tile overlap count,
  `:171` `atomicAdd(num_gaussians_per_tile+tile_idx, 1)` int counter), `compute_tiles_
  kernel` (writes gaussian idx + a **double** sort key `z + tile_idx_multiplier*tile_idx`
  with an explicit OOB guard `(output_start_idx+n_tiles) < output_end_idx` at `:233`),
  then `get_sorted_gaussian_list` sorts via **`torch::sort(sort_keys)`** (ATen, `:327`)
  + `index_select`. NO cub/thrust. Block (1024,1,1). No warp ops.
- `projection.cu` / `projection_backward.cu` -- camera projection, sigma-world, jacobian,
  conic fwd/bwd. Per-element matrix math (uses `matrix.cuh` transpose/multiply helpers).
  No warp ops, no atomics, no shared mem.
- `precompute_sh.cu` -- SH-to-RGB fwd/bwd (`spherical_harmonics.cuh` `__constant__` SH
  coefficient tables). Per-gaussian math. No warp ops.
- `depth.cu` -- depth render; `__shared__` chunk cache + `__syncthreads()` + `__expf`.
  No warp ops, no atomics.

### Atomics
Only `atomicAdd`: float/double gradient accumulation in `render_backward.cu`
(`:272,278,279,280,283`, leader-only) and one int tile counter in `tile_culling.cu:171`.
All on plain device memory (torch tensors are `hipMalloc` device memory, NOT managed).
The cudaKDTree managed-memory `atomicMin/Max`-silently-dropped class is **N/A** (no
atomicMin/Max, no managed memory). atomicAdd is well-supported on gfx90a.

### Libraries / sort
NO cub, NO thrust, NO cuBLAS/cuFFT/cuRAND/cuSPARSE. The per-tile depth sort is
`torch::sort` (PyTorch ATen, already hipified inside torch_hip). So the cudaKDTree
nonzero-begin_bit hipCUB bug is N/A, and there is no hipCUB/rocThrust swap to do.

### Textures / surfaces / managed / pinned / streams / events
**NONE** (whole-tree grep empty). So no texture rule-of-five, no 256B pitch, no layered
array, no linear-filter, no stream/event fault classes apply.

## Risk list (ranked; with the precedent that solved each)
1. **`cg::reduce` + `<cooperative_groups/reduce.h>` absent on ROCm (render_backward.cu)**
   -- HIGH, the ONE certain fix. CONFIRMED: ROCm 7.2.1 has no
   `<cooperative_groups/reduce.h>` and no `cg::reduce`. The 6 `cg::reduce(warp_cg, v,
   cg::plus<T>())` sites (`:244-246,253,259`) will not compile under hipcc. Fix exactly
   as gsplat fault #4: a `USE_ROCM` butterfly all-reduce `warpReduceSum<T>(warp_cg, v)`
   = fold via `__shfl_xor(v, offset)` for offset = tile_size/2 .. 1 over the 32-lane
   tile (`cg::thread_block_tile<32>` provides a `width=32` `shfl_xor`), giving the
   tile-wide sum in every lane; on CUDA keep `cg::reduce`. MUST be templated over T
   (float AND double; HIP `__shfl_xor` supports both). A width-32 tile shfl_xor is
   wave64-correct because the tile only spans 32 lanes of the 64-wide wavefront. Route
   all 6 sites through a `WARP_REDUCE_SUM(warp_cg, v)` macro (`warpReduceSum` on ROCm,
   `cg::reduce(.,.,cg::plus<T>())` on CUDA). (Precedent: gsplat Utils.cuh
   warpReduceSum/Max shfl_xor butterfly; this project is simpler -- only `plus`, no
   `greater`/max.)
2. **`<cooperative_groups/reduce.h>` include itself failing** -- MEDIUM, falls out of
   Risk 1. Once `cg::reduce` is gone on the ROCm path, also guard the
   `#include <cooperative_groups/reduce.h>` line under `#if !defined(USE_ROCM)` (the
   header does not exist on ROCm). Keep `<cooperative_groups.h>` (HIP ships
   `<hip/hip_cooperative_groups.h>`, which hipify maps to; `this_thread_block`,
   `tiled_partition<32>`, `thread_rank`, `.sync()`, and tile `shfl_xor` are all provided
   by HIP CG). Verify `thread_block_tile<32>` shfl works on HIP (gsplat confirmed it).
3. **gradient atomicAdd accumulation-order nondeterminism / gradcheck tolerance** --
   MEDIUM. The backward combines per-tile partials via `atomicAdd` (float and double).
   atomicAdd ordering is non-deterministic on ANY GPU (CUDA included). For the float64
   `gradcheck` tests this is the highest-precision path and the reduce replacement must
   not change the math beyond float reassociation. gsplat saw a tight-tolerance flap
   (2e-3) ONLY in the labeled_partition fan-out case; here the granularity already
   matches CUDA (one rank-0 atomicAdd per 32-tile), so expect no flap. If a gradcheck
   edge flaps, it is float-atomic order, not a wave64 bug -- confirm by rerun + a
   determinism check, do NOT loosen the upstream tolerance without cause. (Precedent:
   gsplat fault #7 "match the CUDA atomic granularity, do not fan out" -- already
   satisfied here.)
4. **`-ffast-math` / FMA contraction drift** -- LOW. setup.py passes only
   `-O3 -std=c++17` (NO `--use_fast_math`), so the gsplat fast-math gradient-edge issue
   (fault #13) does NOT apply. The `use_fast_exp` flag is a runtime kernel arg (a coarse
   `__expf` early-out), independent of compiler fast-math; the forward exact-pixel tests
   (places=5) were generated WITH `use_fast_exp=true`, so HIP `__expf` accuracy vs CUDA
   `__expf` is the thing to watch at the 5-decimal forward gate. clang-HIP defaults
   `-ffp-contract=fast` vs nvcc expression-only; if a forward exact-pixel test misses at
   the 5th decimal, pin `-ffp-contract=on` in the HIP nvcc flags (CV-CUDA class). Try the
   stock build first; only add the flag if a pixel test fails by a tight margin.
5. **PYBIND11_MODULE on the gfx1151 follower** -- LOW (follower-only, NOT lead).
   `bindings.cpp:118` uses `PYBIND11_MODULE`. On Windows/gfx1151 the pybind at::Tensor
   caster can hit the c10 inherited-ctor dllexport gap (the fused-ssim gfx1151 blocker);
   gsplat dodged it via TORCH_LIBRARY. This is a documented external torch-wheel/clang
   issue, out of lead scope; note it for the gfx1151 delta-plan, do not act now.
6. **`__CUDA_ARCH__` / std::min/max / GLM / cuda::std** -- N/A. None present in any
   source (greps empty). matrix.cuh/spherical_harmonics.cuh are plain templated device
   math + `__constant__` tables; `#include <cuda.h>` hipifies fine. So the cudaKDTree
   `__CUDA_ARCH__` host/device-branch class and the gsplat std::min host-shim / GLM
   hipify-mangling classes do NOT apply.

## File-by-file change list (all edits guarded `#if defined(USE_ROCM)`; CUDA path byte-identical)
- `src/render_backward.cu` -- the ONLY file with a required source edit:
  - `:2` guard `#include <cooperative_groups/reduce.h>` out under `USE_ROCM`.
  - add a `USE_ROCM` templated `warpReduceSum<T>(cg::thread_block_tile<32>&, T)` =
    `__shfl_xor` butterfly (offset 16,8,4,2,1), and a `WARP_REDUCE_SUM(warp, v)` macro
    (= `warpReduceSum(warp,v)` on ROCm, `cg::reduce(warp, v, cg::plus<T>())` on CUDA).
  - route the 6 `cg::reduce(...)` sites (`:244-246,253,259`) through `WARP_REDUCE_SUM`.
- NO other `.cu`/`.cuh` edits expected (render.cu/projection*/precompute_sh/depth/
  tile_culling are plain math + shared-mem + atomicAdd + torch::sort, all hipify-clean).
- `setup.py` -- OPTIONAL. Prefer NO edit if the stock `-O3 -std=c++17` + torch
  auto-hipify builds clean (it should; no nvcc-only flags are present). If a forward
  pixel test needs it, add a `torch.version.hip` branch to append `-ffp-contract=on`
  (and that branch is the natural place for `-DUSE_ROCM`, though torch already defines
  `USE_ROCM` for HIP extension compiles -- verify, else add it there). A setup.py
  change is build-only, still acceptable under Strategy B.
- NO compat header, NO symbol renames (Strategy B). NO CMake. NO GitHub Actions; after
  forking, disable Actions on the fork (`gh api -X PUT
  repos/AMD-Ecosystem/<fork>/actions/permissions -F enabled=false`).

## Build commands (gfx90a; from the project src dir or /tmp, NOT /var/lib/jenkins/pytorch)
```
export HIP_VISIBLE_DEVICES=<free GCD>      # cap 4 concurrent GPU agents
export PYTORCH_ROCM_ARCH=gfx90a            # also pass gfx1100 at follower time, no source change
cd /var/lib/jenkins/moat/projects/gaussian_splatting/src
/opt/conda/envs/py_3.12/bin/python -m pip install -e . --no-build-isolation -v
# (equivalently: python setup.py build_ext --inplace)
```
Torch auto-hipifies each `.cu`/`.cuh` (gitignored `.hip` mirror under build/) and links
`amdhip64`/`c10_hip`/`torch_hip`. The module is `splat_cuda`. INCREMENTAL gotcha
(Strategy B): after editing a `.cu`, torch may recompile the STALE hipified mirror --
delete `build/` (and any `*.hip`) before rebuilding so the edit is re-hipified. The arch
comes from `PYTORCH_ROCM_ARCH`; a follower needs only `PYTORCH_ROCM_ARCH=gfx1100` +
clean rebuild, no source edit (no churn to head_sha).
Also install the Python deps the tests import (subset of requirements.txt that the test
suite needs): numpy, opencv-python (cv2), scipy, torchmetrics (and torchvision for the
trainer). Do NOT pin the requirements.txt torch==2.1.2 (the host ROCm torch 2.13 must
win); install only the non-torch test deps.

## Test plan (real GPU, gfx90a) -- mirrors how gsplat/fused-ssim were closed
The suite is Python `unittest` (`python -m unittest discover test`, run from the repo
root with the built `splat_cuda` importable and cwd off /var/lib/jenkins/pytorch).

### Tier 1 -- gradient correctness (the strongest gate for the cg::reduce fix)
- `test/test_rasterize_autograd.py` (8 tests): `torch.autograd.gradcheck` on
  `RenderImage.apply` for SH degrees {0,4,9,16} x {background, no-background}, in
  **float64** (gradcheck default; the backward kernel HAS a double instantiation, so
  this exercises the `cg::reduce` replacement in double). gradcheck compares analytic
  grads (the HIP backward) against central finite differences of the HIP forward. SH_16
  uses `atol=3e-5`. A wave64/reduce bug => gradcheck FAILS. This is the decisive test for
  Risk 1/3.
- `test/test_cuda_autograd_functions.py` (8 tests): gradcheck on the projection / sigma-
  world / jacobian / conic / SH-precompute autograd functions (the non-rasterizer
  kernels). Validates projection_backward.cu, precompute_sh.cu fwd+bwd.

### Tier 2 -- forward render exactness
- `test/test_rasterize.py` (3 tests): renders a fixed synthetic scene
  (`gaussian_test_data.get_test_data`) and asserts specific pixel RGB values to
  `places=5` (5 decimals) for no-SH, full-SH-precompute, and per-pixel-viewdir paths.
  This is a tight forward-correctness gate on render.cu (the `__expf` fast-exp + alpha
  compositing). If a pixel misses only at the 5th decimal, suspect HIP `__expf` /
  FMA-contraction (Risk 4) -> try `-ffp-contract=on`, do not loosen the assert.
- `test/test_depth.py` (1), `test/test_projection.py` (4), `test/test_tile_culling.py`
  (1): forward checks of depth.cu, projection.cu, tile_culling.cu (the torch::sort tile
  binning). test_tile_culling validates the double sort-key ordering survives on ROCm.

### Tier 3 -- determinism + (optional) short training
- Re-run Tier 1+2 twice; the forward (no output atomics) must be BITWISE-deterministic
  across runs; the backward grads must match within float-atomicAdd noise (the gsplat
  determinism bar). A stable forward + grads-within-atomic-noise rules out a wave64 race.
- OPTIONAL end-to-end: `colmap_splat.py 7k --dataset_path <scene> --downsample_factor 4`
  for a few hundred iters on a small Mip-NeRF360 scene, asserting loss-down / PSNR-up /
  no-NaN. Only if a small COLMAP scene is available on the host; the unit suite (Tiers
  1-2) is the primary gate (gsplat/fused-ssim were closed on the unit gradchecks +
  forward compares without requiring a full training run).

### "Validated on gfx90a" (concrete bar)
(a) `python -m unittest discover test` GPU tests PASS: all 8 rasterize gradchecks + 8
autograd-function gradchecks + 3 forward-pixel + projection/depth/tile_culling, AND
(b) forward render bitwise-deterministic across two runs with backward grads within
float-atomic noise, with NO regression in the CPU-only tests (below).

### Non-GPU regression set (must not regress)
CPU-only tests (no `splat_cuda` device work): `test/test_dataloader.py` (3,
`device="cpu"`), `test/test_structs.py` (1, cpu), `test/test_utils.py` (4, cuda-or-cpu).
We touch only render_backward.cu, so these must keep passing unchanged. There is no
separate C++/host unit suite.

## Staged strategy + likely walls
1. **Stock build first** (`pip install -e .`), expecting exactly ONE compile wall: the
   `cg::reduce` + `<cooperative_groups/reduce.h>` on ROCm (Risk 1/2). This is the same
   wall gsplat hit and the fix is known (shfl_xor butterfly). Get a clean COMPILE.
2. **Apply the `cg::reduce` -> warpReduceSum shfl_xor fix** (templated over T, width-32
   tile, USE_ROCM-guarded). Rebuild (delete build/ for re-hipify).
3. **Run Tier 1 gradchecks** -- the decisive correctness gate. Wave-agnostic expectation:
   they pass with no further source edits (gsplat/fused-ssim both needed zero wave-lane
   rework once the CG-reduce gap was filled). If a gradcheck fails:
   - all-zero / wrong grads => the shfl_xor reduce is mis-sized (check width=32, offset
     16..1, and that it sums over exactly the tile lanes) -- NOT a 64-lane rework (the
     tile is 32 by construction).
   - tight-tolerance flap on SH_16 => float-atomic order (Risk 3); rerun + determinism
     check before touching tolerance.
4. **Run Tier 2 forward-pixel tests.** If a 5th-decimal miss: `-ffp-contract=on` (Risk
   4), then HIP `__expf` accuracy. Likely passes stock.
5. **Tier 3 determinism**, optional short training if a scene is present.
Likely-wall summary: the ENTIRE port is expected to be the single gsplat-style
`cg::reduce` shfl_xor swap in render_backward.cu plus the guarded include -- the smallest
of the three splatting precedents (fused-ssim = 0 edits, gsplat = ~14 faults, LiteGS =
PTX+sm_80-redux+half2-masks+tile-remap; this one = 1 fault). No AMD-native rewrite.

## Open questions
- Does the width-32 `thread_block_tile<32>` `__shfl_xor` butterfly reproduce
  `cg::reduce(plus)` bit-for-bit-enough for the float64 gradchecks? (Expected yes;
  gsplat confirmed width-32 tile shuffles are wave64-correct. Verify via Tier 1.)
- Does HIP define `USE_ROCM` automatically for CUDAExtension device compiles, or must
  setup.py add `-DUSE_ROCM`? (fused-ssim/gsplat relied on torch defining it; confirm at
  build time, add to a setup.py HIP branch if the guard does not fire.)
- Forward exact-pixel tests at places=5: does stock HIP `__expf` + default
  `-ffp-contract=fast` hold 5 decimals, or is `-ffp-contract=on` needed? (Resolve at
  Tier 2.)
- Small COLMAP scene availability for the optional Tier-3 training run (fetch a
  Mip-NeRF360 scene if wanted; not required for the gate).

## Disposition
Strategy B mechanical bring-up + GPU validation. This is the 4th gaussian-splatting
port and the wave-AGNOSTIC (gsplat/fused-ssim) variety: a single `cg::reduce`->shfl_xor
fix in render_backward.cu (ROCm has no `<cooperative_groups/reduce.h>`), guarded by
USE_ROCM, CUDA path byte-identical. No raw warp intrinsics, no PTX, no half2 SIMD, no
cub/thrust, no textures, no managed memory -- none of the LiteGS/popsift wave64 traps
apply. Correctness-first; no MFMA/CK rewrite warranted. Record quirks in notes.md; the
only generalizable lesson is a re-confirmation of the gsplat "ROCm lacks cg::reduce ->
shfl_xor butterfly" item (already in the changelog), so likely no new PORTING_GUIDE
entry unless a surprise appears.
