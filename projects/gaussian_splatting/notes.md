# gaussian_splatting (joeyan) -- ROCm/HIP port notes

Upstream: joeyan/gaussian_splatting @ 8ddaa799846ab75810305f457daf36668b68bddf (branch main).
Fork: AMD-Ecosystem/gaussian_splatting, port on `moat-port` (master/main is a clean upstream mirror).
A from-scratch MIT 3DGS reimpl: ONE PyTorch CUDAExtension `splat_cuda` built from
`src/*.cu` + `src/bindings.cpp`. No submodules, no simple-knn, no cub (per-tile depth sort
is `torch::sort`). Strategy B (torch auto-hipify). Wave-agnostic (gsplat/fused-ssim family).

## The one required fix (gfx90a lead)
`src/render_backward.cu` only. ROCm 7.2.1 ships `<hip/hip_cooperative_groups.h>` (which
hipify maps `<cooperative_groups.h>` to) but has NO `<cooperative_groups/reduce.h>`, no
`cg::reduce`, no `cg::plus`. The backward kernel uses `cg::tiled_partition<32>` plus six
`cg::reduce(warp, grad, cg::plus<T>())` then a rank-0 `atomicAdd` per 32-tile.

Fix (all `#if defined(USE_ROCM)`-guarded; CUDA path byte-identical):
1. Guard out `#include <cooperative_groups/reduce.h>` under `#if !defined(USE_ROCM)`.
   hipify does NOT rewrite this include (it leaves it verbatim), so it must be guarded.
2. Add a templated butterfly and route the six sites through a macro:
   ```
   template <typename T>
   __device__ __forceinline__ T warpReduceSum(const cg::thread_block_tile<32>& warp, T val) {
       #pragma unroll
       for (int offset = warp.size() / 2; offset > 0; offset >>= 1)
           val += warp.shfl_xor(val, offset);
       return val;
   }
   #define WARP_REDUCE_SUM(warp, val) warpReduceSum(warp, val)             // ROCm
   #define WARP_REDUCE_SUM(warp, val) cg::reduce(warp, val, cg::plus<T>()) // CUDA
   ```
   MUST template over T: the backward is instantiated for BOTH float AND double (the
   float64 gradchecks exercise the double path; HIP tile shfl_xor supports double).

Wave64 correctness: a `thread_block_tile<32>` `shfl_xor` uses the tile width (32) as the
shuffle width, so it stays inside the tile's 32 lanes of a 64-wide wavefront. The block is
16x16 = 256 threads -> 4 wavefronts, each holding 2 independent 32-lane tiles; each tile's
rank-0 does its own atomicAdd, matching the NVIDIA warp granularity. No lane-math rework.
This is the direct gsplat fault #4 template (see PORTING_GUIDE changelog 2026-05-30).

NO other source edits. NO compat header, NO symbol renames, NO setup.py change: torch's
HIP CUDAExtension compile already passes `-DUSE_ROCM=1` (confirmed in the build log), so the
guards fire without help. NO `-ffp-contract` pin was needed (see validation below).

## Build recipe (gfx90a; cwd must be OUTSIDE /var/lib/jenkins/pytorch)
```
export HIP_VISIBLE_DEVICES=3            # free GCD; cap concurrent GPU agents at 4
export PYTORCH_ROCM_ARCH=gfx90a         # follower: gfx1100, clean rebuild, NO source edit
export MAX_JOBS=16
cd projects/gaussian_splatting/src
rm -rf build *.egg-info src/*.hip       # Strategy B: delete build/ so edited .cu re-hipifies
/opt/conda/envs/py_3.12/bin/python -m pip install -e . --no-build-isolation --no-deps -v
```
Reusable script: `agent_space/gsplat_build.sh`. torch 2.13.0a0 (ROCm 7.2.5), device MI250X.
hipify writes `src/*.hip` + `src/matrix_hip.cuh` mirrors next to the sources; these are
gitignored (`*.hip`, `*_hip.cuh`) and must NOT be committed.

Test deps (install with --no-deps so the host ROCm torch is not clobbered; do NOT install
requirements.txt torch==2.1.2): numpy, opencv-python (cv2), scipy, yaml already present;
added `torchmetrics lightning_utilities` (test_utils SSIM) and `tyro docstring_parser
typeguard shtab` (test_dataloader import). torchvision (trainer/colmap_splat only) not installed.

## Test invocation gotcha
`python -m unittest discover test` MUST run with cwd = the repo src dir (it puts both the
repo root, for `splat_py`, and `test/`, for `gaussian_test_data`, on sys.path). Do NOT run
it through `utils/timeit.sh` directly: timeit.sh cd's to the MOAT root, so `discover test`
then resolves Python's stdlib `test` package instead. Use `agent_space/gsplat_test.sh`
(cd's to src first) and wrap THAT with timeit. Single modules: `cd test && PYTHONPATH=<src>
python -m unittest <module>`.

## Validation on gfx90a (HIP_VISIBLE_DEVICES=3) -- DECISIVE GATE PASSES
Tier 1 (decisive for the reduce fix): all 16 float64 gradchecks PASS, deterministically
across two independent runs:
- `test_rasterize_autograd` 8/8 (gradcheck on RenderImage, SH {0,4,9,16} x {bg,no-bg}).
- `test_cuda_autograd_functions` 8/8 (projection/sigma/jacobian/conic/SH precompute).
These prove the warpReduceSum butterfly is numerically exact in float (and double).

Tier 2/3: `test_rasterize.test_rasterize_no_sh` PASSES (forward bitwise to golden);
the full forward render is bitwise-deterministic across runs (verified).

### Pre-existing upstream test failures -- NOT caused by the port (all reproduce off-GPU)
Proven platform-independent, do not block the gate:
- `test_rasterize` SH paths (use_precompute, use_per_pixel_viewdir): the recorded goldens
  are inconsistent with the CURRENT source's SH math. At pixel [340,348] exactly ONE splat
  contributes; a pure float64 CPU recompute of that single-splat composite (replicating
  render.cu arithmetic) reproduces the HIP value [0.6331,0.1562,0.1562] EXACTLY, not the
  golden [0.5363,0.0593,0.0593]. The `splat_cuda` precompute kernel itself matches a
  hand-computed numpy SH reference to 7 decimals in both f32 and f64. Setting the higher-order
  SH coeffs to 0 makes the SH path reproduce the no-SH golden exactly (DC path is exact).
- `test_projection` (uv[0,1], jacobian[0,1,2]) at places=4 and `test_utils`
  test_compute_rays_world_frame at the default places=7: goldens recorded tighter than
  float32 epsilon. The EXACT float64 math gives uv[0,1]=573.98641 (rounds to .9864) while the
  golden recorded .9863; HIP's float32 573.98645 is actually CLOSER to the true value than
  the golden. `-ffp-contract=on` cannot fix a wrong golden, so it was NOT applied.
- `test_dataloader` (3 tests): hardcodes the upstream author's local path
  `/home/joe/Downloads/garden/sparse/0/points3D.bin`. CPU-only; cannot pass without that
  dataset. Unrelated to the port.

Upstream runs NO CI (no workflows), so these stale/fragile goldens were never gated.

## Followers (delta-plan, not lead scope)
- gfx1100: expect a clean rebuild with `PYTORCH_ROCM_ARCH=gfx1100`, NO source edit (the fix
  is wave-agnostic; tile<32> shfl_xor is correct on wave32 too).
- gfx1151 (Windows): `bindings.cpp` uses PYBIND11_MODULE which can hit the c10 inherited-ctor
  dllexport blocker (the fused-ssim gfx1151 wall). Note for the delta-plan; gsplat dodged it
  via TORCH_LIBRARY.

## Review 2026-06-01 (reviewer, linux-gfx90a)
Verdict: APPROVE -> review-passed. Reviewed `git diff 8ddaa79...HEAD` on AMD-Ecosystem/gaussian_splatting@moat-port (HEAD a4d8b9a) with the /pr-review skill (ROCm-fault-class aware). The delta is exactly 2 files (.gitignore +4 hipify-artifact ignores; src/render_backward.cu +25/-5), additive, all `#if defined(USE_ROCM)`-guarded, CUDA path byte-identical. No problems found. The validator runs the real-GPU gates next (a missing GPU run is expected at review time and is not a blocker).

warpReduceSum equivalence + wave64-tile correctness (verified against the ROCm 7.2.1 HIP CG header /opt/rocm-7.2.1/include/hip/amd_detail/amd_hip_cooperative_groups.h):
- `cg::thread_block_tile<32>::size()` returns the tile size 32 (numThreads=tileSize, header :811/:824), so warpReduceSum (render_backward.cu:19-26) folds offsets 16,8,4,2,1 = a complete XOR butterfly over 2^5=32 lanes; every lane ends with the tile-wide sum, the same all-reduce cg::reduce(plus) provides.
- `thread_block_tile<32>::shfl_xor` forwards to `__shfl_xor(var, laneMask, numThreads)` with numThreads=32 as the WIDTH (header :863-864). Width-32 partitions the 64-lane wavefront into two independent 32-lane segments, so each tile's butterfly stays inside its own 32 lanes -- no cross-tile contamination on wave64. is_valid_tile_size<32> passes (pow2 and <= wavefront 64). No lane-math rework needed; the claim is correct.
- T=double: shfl_xor<T> is templated, double is supported on HIP (two 32-bit shuffles); the float64 gradchecks exercise the double instantiations.
- Leader election: `thread_rank()` is tile-relative (`workgroup::thread_rank() & (numThreads-1)`, header :816), so rank-0 fires once per 32-tile; one atomicAdd per tile == NVIDIA warp granularity, no fan-out/double-count.
- All 6 upstream cg::reduce sites (orig :244-246,253,259) route through WARP_REDUCE_SUM; zero cg::reduce survives on the ROCm path. CUDA-path macro expands to the original `cg::reduce(warp,val,cg::plus<T>())` (T from the enclosing kernel template) -> byte-identical preprocessed CUDA.

Stale-golden vs port-bug ruling (anchored on the gradchecks): the 8 non-passing tests are PORT-INDEPENDENT, not ROCm regressions. Decisive: the edited file (render_backward.cu) is backward-only; the forward rasterizer (render.cu) is shared-mem + __syncthreads with NO cg::reduce, NO warp ops, NO output atomics, and does not include the edited file -- so the failing FORWARD-pixel tests (test_rasterize SH x2, test_projection, test_utils ray) physically cannot be affected by the warpReduceSum change. The 16/16 float64 gradchecks (test_rasterize_autograd 8/8, test_cuda_autograd_functions 8/8) are the real correctness gate: torch.autograd.gradcheck compares the analytic HIP backward against central finite differences of the HIP forward and never reads any stored golden, so a pass proves the reduce (incl. double) is self-consistent regardless of golden staleness; SH_16 at atol=3e-5 exercises the full 16-coeff SH backward in double. The 2 SH-golden failures are explained by the source's SH math (sh_to_rgb DC term = SH_0 * (0.5/SH_0) = 0.5 exact, matching the passing DC-only no-SH golden; the higher-order goldens were recorded against older SH math). test_projection (places=4 on ~574 magnitude) and test_utils ray (default places=7 on ~0.5-0.7) are recorded tighter than float32 epsilon -- fragile on any float32 backend including CUDA. test_dataloader (3) ERRORs on a hardcoded missing path TEST_DATASET_PATH="/home/joe/Downloads/garden" (test_dataloader.py:7), CPU-only, unrelated.

Fault-class sweep (whole src/ tree, all clean): no raw warp intrinsics / warpSize / lane masks / PTX / half2 / textures / surfaces / cudaArray / managed memory / atomicMin/Max / cub / thrust / cuBLAS / __CUDA_ARCH__ / std::min-max / GLM / math_constants.h. Only atomicAdd (float/double leader-only in render_backward.cu; one int counter tile_culling.cu:171) on plain torch device tensors (not managed) -> atomicMin/Max-silent-drop class N/A. setup.py is a torch CUDAExtension with only `-O3 -std=c++17` (no --use_fast_math) and is unedited -> Strategy B correct; the `__expf` (depth.cu:94, render.cu:138, render_backward.cu:181) is a runtime fast-exp kernel arg, not a compiler flag. render_backward.cu is genuinely the only file needing a source fix.

Commit hygiene: clean. Title 54 chars, `[ROCm]` prefix; body discloses Claude ("Authored with assistance from the Claude agent (MOAT porter)"); no Co-Authored-By/noreply trailer; ASCII-only (-> arrows are ASCII); Test Plan with fenced commands. `.hip`/`_hip.cuh` artifacts gitignored and untracked (git check-ignore confirms; `git ls-files` shows none committed). Fork main == upstream 8ddaa79 (clean mirror); moat-port = 1 commit on top of base. No AMD-internal codenames/hosts/customers in the diff or message; author is the user's own public identity; fork is the public AMD-Ecosystem/gaussian_splatting.

## Validation 2026-06-01

Platform: linux-gfx90a. Device: AMD Instinct MI250X / MI250 (HIP_VISIBLE_DEVICES=3). Fork HEAD: a4d8b9aa308b23ba46a10d9734817237fd2474fd.

Build: fresh (`rm -rf build *.egg-info src/*.hip` then `PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=16 pip install -e . --no-build-isolation --no-deps`). Strategy B (torch auto-hipify). Completed successfully; `.so` at `src/splat_cuda.cpython-312-x86_64-linux-gnu.so`.

Commands:
```
# Build
bash utils/timeit.sh gaussian_splatting compile -- bash agent_space/gsplat_build.sh

# Decisive gate (run twice)
cd projects/gaussian_splatting/src
HIP_VISIBLE_DEVICES=3 python -m unittest test.test_rasterize_autograd test.test_cuda_autograd_functions -v

# Full suite
HIP_VISIBLE_DEVICES=3 python -m unittest discover test -v
```

Decisive gate (16 float64 gradchecks):
- Run 1: test_rasterize_autograd 8/8 PASS, test_cuda_autograd_functions 8/8 PASS. Ran 16 tests in 78.4s -- OK
- Run 2: test_rasterize_autograd 8/8 PASS, test_cuda_autograd_functions 8/8 PASS. Ran 16 tests in 80.3s -- OK
- Deterministic across both runs.

Full suite (33 tests): FAILED (failures=5, errors=3). Non-passing tests verified to be exactly the pre-documented stale-golden/missing-dataset set -- no new regressions:
- FAIL test_rasterize_full_sh_use_precompute, test_rasterize_full_sh_use_per_pixel_viewdir: stale SH goldens
- FAIL test_project_points, test_compute_projection_jacobian: places=4 tighter than float32 epsilon
- FAIL test_compute_rays_world_frame: places=7 tighter than float32 epsilon
- ERROR test_dataloader (3): hardcoded missing path /home/joe/Downloads/garden (CPU-only, dataset absent)
- All other GPU tests PASS: test_rasterize_no_sh, test_depth, test_tile_culling, test_structs, test_cuda_autograd_functions, test_rasterize_autograd, test_projection (other), test_utils (other)

Decision: PASS -> completed. validated_sha = a4d8b9aa308b23ba46a10d9734817237fd2474fd. Followers linux-gfx1100 and windows-gfx1151 auto-unblocked to port-ready.

## Validation 2026-06-01 (gfx1100)

Platform: linux-gfx1100. Device: AMD Radeon Pro W7800 48GB (HIP_VISIBLE_DEVICES=0). Fork HEAD: a4d8b9aa308b23ba46a10d9734817237fd2474fd. Fork cloned fresh from AMD-Ecosystem/gaussian_splatting@moat-port; no submodules.

Build: fresh (`rm -rf build *.egg-info src/*.hip` then `PYTORCH_ROCM_ARCH=gfx1100 MAX_JOBS=16 pip install -e . --no-build-isolation --no-deps`). Strategy B (torch auto-hipify). Completed in 39.8s. `.so` at `src/splat_cuda.cpython-312-x86_64-linux-gnu.so`.

gfx1100 code-object evidence: `llvm-objdump --offloading splat_cuda.cpython-312-x86_64-linux-gnu.so` shows 7 bundles all `amdgcn-amd-amdhsa--gfx1100`; no gfx90a.

Commands:
```
# Build
bash utils/timeit.sh gaussian_splatting compile -- bash agent_space/gauss_splatting_build_gfx1100.sh

# Decisive gate (run twice)
bash utils/timeit.sh gaussian_splatting test -- bash agent_space/gauss_splatting_test_gfx1100.sh

# Full suite
bash utils/timeit.sh gaussian_splatting test -- bash agent_space/gauss_splatting_test_full_gfx1100.sh
```

Decisive gate (16 float64 gradchecks):
- Run 1: test_rasterize_autograd 8/8 PASS, test_cuda_autograd_functions 8/8 PASS. Ran 16 tests in 63.0s -- OK
- Run 2: test_rasterize_autograd 8/8 PASS, test_cuda_autograd_functions 8/8 PASS. Ran 16 tests in 61.1s -- OK
- Deterministic across both runs.

Full suite (33 tests): FAILED (failures=5, errors=3). Non-passing tests are exactly the pre-documented gfx90a set -- no new regressions on gfx1100:
- FAIL test_rasterize_full_sh_use_precompute, test_rasterize_full_sh_use_per_pixel_viewdir: stale SH goldens
- FAIL test_project_points, test_compute_projection_jacobian: places=4 tighter than float32 epsilon
- FAIL test_compute_rays_world_frame: places=7 tighter than float32 epsilon
- ERROR test_dataloader (3): hardcoded missing path /home/joe/Downloads/garden (CPU-only, dataset absent)
- All other GPU tests PASS: test_rasterize_no_sh, test_depth, test_tile_culling, test_structs, test_cuda_autograd_functions (8/8), test_rasterize_autograd (8/8), test_projection (other 2/4), test_utils (other 2/3)

wave32 verdict: The fix uses `cg::thread_block_tile<32>` with `shfl_xor` butterfly (warpReduceSum template). On wave32 (gfx1100), a `thread_block_tile<32>` maps exactly one-to-one onto the hardware wavefront, so each tile's 32-lane XOR butterfly is a complete all-reduce of the entire wavefront. No cross-tile contamination, no lane-math needed. The 16/16 float64 gradchecks (forward+backward consistency via torch.autograd.gradcheck) confirm the warpReduceSum is numerically exact on gfx1100. No HSA 0x1016, no HIP errors, no NaN/Inf in any output.

No source edits needed for gfx1100. The fix is wave-agnostic as predicted: `shfl_xor` with `tile.size()=32` is correct on both wave64 (gfx90a, 2 independent 32-lane segments per wavefront) and wave32 (gfx1100, 1 exact 32-lane wavefront). No fork amendment.

Decision: PASS -> completed. validated_sha = a4d8b9aa308b23ba46a10d9734817237fd2474fd.

## Validation 2026-06-07 (windows-gfx1201)

Platform: windows-gfx1201. Device: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), HIP_VISIBLE_DEVICES=0 (gfx1101 V710 offline this session). Fork HEAD: d7f2227cb285966c290c36efaf7f8240f10379cb.

Python env: B:\develop\TheRock\external-builds\pytorch\.venv (torch 2.9.1+rocm7.14.0a20260604, HIP 7.14.60850-d34cbb64).

### Windows-specific fix committed (d7f2227, on top of a4d8b9a)

On Windows with HIP, MSVC-compiled bindings.cpp cannot link c10::ValueError (and other pybind11/c10 exception types) from the clang-built c10.dll due to the inherited-constructor ABI mismatch. Fix: at setup.py execution time, copy bindings.cpp to src/bindings_winhip.cu so BuildExtension routes it through hipcc/amdclang++ (same ABI as c10.dll). Guarded by `os.name == 'nt' and torch.version.hip is not None`; Linux and CUDA paths unchanged. src/bindings_winhip.cu added to .gitignore.

### Build command

```
set HIP_VISIBLE_DEVICES=0
set PYTORCH_ROCM_ARCH=gfx1201
set MAX_JOBS=32
python -m pip install -e . --no-build-isolation --no-deps -v
```

Build script: agent_space/gauss_build_gfx1201.py. Strategy B (torch auto-hipify). Completed successfully; .pyd at splat_cuda.cp312-win_amd64.pyd (in-place editable install). Confirmed gfx1201 device code in .pyd (.hipFatB section present, dumpbin /DEPENDENTS shows amdhip64_7.dll).

### Decisive gate (16 float64 gradchecks -- run 1)

```
cd projects/gaussian_splatting/src
HIP_VISIBLE_DEVICES=0 python -m unittest test.test_rasterize_autograd test.test_cuda_autograd_functions -v
```

test_rasterize_autograd 8/8 PASS, test_cuda_autograd_functions 8/8 PASS. Ran 16 tests in 67.3s -- OK.

### Full suite

```
cd projects/gaussian_splatting/src
HIP_VISIBLE_DEVICES=0 python -m unittest discover test -v
```

Ran 31 tests in 65.8s. FAILED (failures=5, errors=1). Non-passing tests are exactly the pre-documented gfx90a/gfx1100 set -- no new regressions on gfx1201:
- FAIL test_rasterize_full_sh_use_precompute, test_rasterize_full_sh_use_per_pixel_viewdir: stale SH goldens
- FAIL test_project_points, test_compute_projection_jacobian: places=4 tighter than float32 epsilon
- FAIL test_compute_rays_world_frame: places=7 tighter than float32 epsilon (diff 5.96e-8)
- ERROR test_dataloader: missing tyro module + hardcoded path /home/joe/Downloads/garden (CPU-only, dataset absent)
- All other GPU tests PASS: test_rasterize_no_sh, test_depth (test_rasterize_no_sh), test_tile_culling, test_structs, test_cuda_autograd_functions (8/8), test_rasterize_autograd (8/8), test_projection (compute_conic, compute_sigma_world), test_utils (compute_rays_camera_frame, quaternion_to_rotation_torch, transform_points_torch)

wave32 verdict (gfx1201 RDNA4): same as gfx1100 wave32 analysis. The warpReduceSum template with `cg::thread_block_tile<32>::shfl_xor` butterfly maps exactly one-to-one onto the hardware wavefront; each tile's 32-lane XOR butterfly is a complete all-reduce. No cross-tile contamination, no lane-math needed. 16/16 float64 gradchecks pass (forward+backward consistency via torch.autograd.gradcheck) confirming numerical correctness.

Decision: PASS -> completed. validated_sha = d7f2227cb285966c290c36efaf7f8240f10379cb.

Note for linux-gfx90a and linux-gfx1100 revalidators: the Windows fix (setup.py + .gitignore delta d7f2227) is entirely `os.name == 'nt'`-guarded. A binary-equivalence check (`python3 utils/codeobj_diff.py`) between builds at a4d8b9a and d7f2227 on Linux will show identical device code objects and exported symbols, allowing carry-forward without a GPU re-run.

## Validation 2026-06-08 (linux-gfx90a revalidate carry-forward)

Platform: linux-gfx90a. validated_sha a4d8b9aa30 -> head_sha d7f2227cb2.

Delta: d7f2227 adds a Windows-only setup.py guard (`if os.name == 'nt' and torch.version.hip is not None`) that copies bindings.cpp to bindings_winhip.cu on Windows HIP builds only. Dead code on Linux (os.name == 'posix'). Also adds two lines to .gitignore for the generated bindings_winhip.cu.

`python3 utils/moatlib.py classify gaussian_splatting a4d8b9aa30 d7f2227cb2` -> `class=mixed arch_independent=False` (classifier flagged setup.py token change -- not inert by classifier alone, so binary check required).

Build at d7f2227 (Strategy B, gfx90a, MAX_JOBS=16) via `bash utils/timeit.sh gaussian_splatting compile -- bash agent_space/gsplat_build.sh`. Build succeeded; .so size unchanged at 811360 bytes.

`python3 utils/codeobj_diff.py agent_space/gsplat_diff/old agent_space/gsplat_diff/new`:
```
verdict=identical
  splat_cuda.cpython-312-x86_64-linux-gnu.so: identical (exported symbols + device ISA identical (230 exports))
```

Carry-forward: `python3 utils/moatlib.py carry-forward gaussian_splatting linux-gfx90a d7f2227cb2 binary-equiv "Windows-only setup.py guard (os.name=='nt') is dead on Linux; codeobj_diff: identical (230 exports, device ISA)"`.

Decision: CARRY-FORWARD -> completed. validated_sha = d7f2227cb285966c290c36efaf7f8240f10379cb. No GPU re-run needed.

## Validation 2026-06-08 (linux-gfx1100 revalidate carry-forward)

Platform: linux-gfx1100 (GPU 3, HIP_VISIBLE_DEVICES=3). validated_sha a4d8b9aa30 -> head_sha d7f2227cb2.

Delta: d7f2227 adds a Windows-only setup.py guard (`if os.name == 'nt' and torch.version.hip is not None`) that copies bindings.cpp to bindings_winhip.cu on Windows HIP builds only. Dead code on Linux (os.name == 'posix'). Also adds two lines to .gitignore for the generated bindings_winhip.cu.

`python3 utils/moatlib.py classify gaussian_splatting a4d8b9aa30 d7f2227cb2` -> `class=mixed arch_independent=False` (classifier flagged setup.py token change -- not inert by classifier alone, so binary check required).

Build at d7f2227 (Strategy B, gfx1100, MAX_JOBS=16, HIP_VISIBLE_DEVICES=3) via `bash utils/timeit.sh gaussian_splatting compile -- bash agent_space/gauss_splatting_build_gfx1100_gpu3.sh`. Build succeeded; .so size 799072 bytes (same as old).

```
python3 utils/codeobj_diff.py agent_space/gaussian_splatting-gfx1100-gpu3/old agent_space/gaussian_splatting-gfx1100-gpu3/new
verdict=identical
  splat_cuda.cpython-312-x86_64-linux-gnu.so: identical (exported symbols + device ISA identical (230 exports))
```

Carry-forward: `python3 utils/moatlib.py carry-forward gaussian_splatting linux-gfx1100 d7f2227cb2 binary-equiv "Windows-only setup.py guard (os.name=='nt') is dead on Linux; codeobj_diff gfx1100: identical (230 exports, device ISA)"`.

Decision: CARRY-FORWARD -> completed. validated_sha = d7f2227cb285966c290c36efaf7f8240f10379cb. No GPU re-run needed.

## Validation 2026-06-19 (windows-gfx1101)

Platform: windows-gfx1101. Device: AMD Radeon PRO V710 (gfx1101, RDNA3, wave32), HIP_VISIBLE_DEVICES=1. Fork HEAD: dfb1dd1cbcc508e08c7395b27953620121dad1fc (squashed moat-port).

Python env: B:\develop\TheRock\external-builds\pytorch\.venv (torch 2.9.1+rocm7.14, HIP 7.14).

### Build command

```
set HIP_VISIBLE_DEVICES=1
set PYTORCH_ROCM_ARCH=gfx1101
set MAX_JOBS=64
python agent_space/gauss_build_gfx1101.py
```
(wrapped: `utils/timeit.sh gaussian_splatting compile -- python agent_space/gauss_build_gfx1101.py`)

Strategy B (torch auto-hipify). Completed successfully; .pyd at splat_cuda.cp312-win_amd64.pyd. Confirmed gfx1101 device code in .pyd (.hipFatB + .hip_fat sections present, dumpbin /DEPENDENTS shows amdhip64_7.dll).

No source edits needed for gfx1101 -- the fix is wave-agnostic and the Windows HIP bindings fix (setup.py, bindings_winhip.cu) is already in the squashed commit.

### Decisive gate (16 float64 gradchecks)

```
cd projects/gaussian_splatting/src
HIP_VISIBLE_DEVICES=1 python -m unittest test.test_rasterize_autograd test.test_cuda_autograd_functions -v
```

test_rasterize_autograd 8/8 PASS, test_cuda_autograd_functions 8/8 PASS. Ran 16 tests in 55.8s -- OK.

### Full suite

```
cd projects/gaussian_splatting/src
HIP_VISIBLE_DEVICES=1 python -m unittest discover test -v
```

Ran 31 tests in 55.2s. FAILED (failures=5, errors=1). Non-passing tests are exactly the pre-documented gfx90a/gfx1100/gfx1201 set -- no new regressions on gfx1101:
- FAIL test_rasterize_full_sh_use_precompute, test_rasterize_full_sh_use_per_pixel_viewdir: stale SH goldens
- FAIL test_project_points, test_compute_projection_jacobian: places=4 tighter than float32 epsilon
- FAIL test_compute_rays_world_frame: places=7 tighter than float32 epsilon (diff 5.96e-8)
- ERROR test_dataloader: missing tyro module + hardcoded path /home/joe/Downloads/garden (CPU-only, dataset absent)
- All other GPU tests PASS: test_rasterize_no_sh, test_depth (test_rasterize_no_sh), test_tile_culling, test_structs, test_cuda_autograd_functions (8/8), test_rasterize_autograd (8/8), test_projection (compute_conic, compute_sigma_world), test_utils (compute_rays_camera_frame, quaternion_to_rotation_torch, transform_points_torch)

wave32 verdict (gfx1101 RDNA3): identical to gfx1100/gfx1201 wave32 analysis. The warpReduceSum template with `cg::thread_block_tile<32>::shfl_xor` butterfly maps exactly one-to-one onto the hardware wavefront on wave32; each tile's 32-lane XOR butterfly is a complete all-reduce. 16/16 float64 gradchecks pass (forward+backward consistency via torch.autograd.gradcheck) confirming numerical correctness. No TDR events, no hipErrorLaunchFailure, GPU healthy post-test.

Decision: PASS -> completed. validated_sha = dfb1dd1cbcc508e08c7395b27953620121dad1fc.
