# LiteGS ROCm/HIP port plan (lead: linux-gfx90a)

## Project
- Upstream: MooreThreads/LiteGS @ `004b95215c90c36cdaf4b354301132b700ac287b` (branch `master`).
- "Training 3DGS in 50 seconds" -- a refactored modular 3D Gaussian-splatting trainer. Pure-Python pipeline plus a fused CUDA backend.
- ext_type: **torch-extension** -> **Strategy B** (torch hipifies the `.cu`/`.cuh` at build time; do NOT add a compat header or hand-rename symbols).
- THREE PyTorch CUDA extensions to build (no top-level setup.py; each submodule has its own):
  1. `litegs/submodules/gaussian_raster` -> module `litegs_fused` (the rasterizer + transforms + binning + SH + compaction + sparse Adam). The hard one.
  2. `litegs/submodules/simple-knn` -> module `simple_knn._C` (Morton kNN for point-cloud init spacing). Easy.
  3. `litegs/submodules/fused_ssim` -> git submodule, url `github.com/kemchenj/fused-ssim` branch `fused-l1-ssim-loss` (a FORK of rahul-goel/fused-ssim with an added L1+SSIM loss). MOAT already ported the upstream fused-ssim with ZERO source changes; this fork is the same kernels plus a loss head.

## Existing AMD support (assessment) -> PROCEED (fresh CUDA->HIP, no prior AMD path)
LiteGS is a virgin CUDA codebase with NO ROCm awareness anywhere:
- No `torch.version.hip` branch in any of the three setup.py (cf. fused-ssim/gsplat which already had one). `gaussian_raster/setup.py` and `simple-knn/setup.py` pass plain `nvcc` flags (`-O3 --use_fast_math`).
- No `#if defined(USE_ROCM)`/`__HIP_PLATFORM_AMD__` guards in any `.cu`/`.cuh`.
- `raster.cu:1-4` force-defines `__CUDACC__`/`__NVCC__` if unset (an editor-intellisense guard); harmless under hipcc but signals a CUDA-only mindset.
- README advertises "Pure Python or CUDA"; no AMD/OpenCL/Vulkan/HIP mention.
Decision: a full Strategy-B CUDA->HIP bring-up. This is the THIRD member of the gaussian-splatting playbook (gsplat, fused-ssim, LichtFeld-Studio) but the rasterizer is the MOST NVIDIA-coupled of the four -- see CUDA surface. Disposition is a correctness-first MECHANICAL port (guarded by `USE_ROCM`), NOT an AMD-native rewrite: LiteGS's kernels are bespoke half2-SIMD alpha-blending, not CUTLASS/CuTe/wgmma, so the perf-rewrite caveat (PORTING_GUIDE) does not apply. A later MFMA/CK pass is not warranted.

## ROCm torch env (verified present on this host, identical to the 3 precedents)
- conda env `py_3.12`, torch **2.13.0a0+gitb5e90ff**, `torch.version.hip 7.2.53211`, `torch.cuda.is_available()=True`, device "AMD Instinct MI250X / MI250" (gfx90a). Host ROCm **7.2.1**, hipcc `/opt/rocm/bin/hipcc`, ninja present. 4x MI250X GCD (wave64), ids 0-3.
- `CUDA_HOME=None`, `ROCM_HOME=/opt/rocm` -> a `CUDAExtension` build auto-hipifies and links `amdhip64`/`c10_hip`/`torch_hip` (Strategy B).
- MUST build/run with cwd OUTSIDE `/var/lib/jenkins/pytorch` (that source tree shadows the installed `torch`); run from the LiteGS src dir or `/tmp`. Pin a free GCD via `HIP_VISIBLE_DEVICES` (cap 4 concurrent GPU agents; check `rocm-smi --showuse`).

## Build classification (evidence)
torch-extension. Decisive lines:
- `gaussian_raster/setup.py:2` `from torch.utils.cpp_extension import CUDAExtension, BuildExtension, COMMON_NVCC_FLAGS`; `:24` `CUDAExtension(name="litegs_fused", sources=[binning.cu, compact.cu, cuda_errchk.cpp, ext_cuda.cpp, raster.cu, transform.cu], ...)`; `:39` `cmdclass={'build_ext': BuildExtension}`.
- `simple-knn/setup.py:13,24` `CUDAExtension(name="simple_knn._C", sources=[spatial.cu, simple_knn.cu, ext.cpp])` + `BuildExtension`.
- `gaussian_raster/CMakeLists.txt` EXISTS but `README.MD:71-80` documents it only as an OPTIONAL "CUDA Debug in Visual Studio" path (`find_package(Torch)` + `CMAKE_PREFIX_PATH` from `torch.utils.cmake_prefix_path`). The shipped/Linux build path is `pip install` of each setup.py (README:52-68). We use the setup.py path (Strategy B); the CMake path is not needed and is not ported.

## CUDA surface inventory (per extension)

### A) litegs_fused (gaussian_raster) -- 21 `__global__` kernels across 4 .cu. THE WAVE64 RISK.
Forward/backward rasterizer (`raster.cu`): `raster_forward_kernel`, `raster_backward_kernel`, `pack_forward_params`, `unpack_gradient`.
Transforms (`transform.cu`): jacobian rayspace, transform-matrix fwd/bwd, MVP fwd/bwd, world2ndc fwd/bwd, create_cov2d fwd/bwd, sh2rgb fwd/bwd, eigh+inv 2x2 fwd/bwd.
Binning (`binning.cu`): duplicate_with_keys, tile_range, get_allocate_size + the cub depth-tile SortPairs.
Compaction/optimizer (`compact.cu`): viewproj fwd/bwd, sparse_chunk_adam, sparse_primitive_adam, frustum_culling_aabb, activate fwd/bwd, sparse_scatter.

Wave/warp primitives (the meat -- ALL hardcoded to 32-lane NVIDIA warps):
- `raster.cu:84-90` `warp_reduce_sum<T,broadcast>`: `__shfl_down_sync(0xffffffff, data, 16/8/4/2/1)` + `__shfl_sync(0xffffffff, data, 0)` broadcast. Hardcoded offset-16 start (32-lane assumption).
- `raster.cu:96,103,110,120,128,131,141,149,152,155,694` `__reduce_add_sync`/`__reduce_max_sync(0xffffffff, ...)`: **sm_80+ HARDWARE warp-redux intrinsics** (11 sites). These are the float/float2/float3/uint specializations of `warp_reduce_sum` PLUS the backward `index_in_tile` max. NOT a plain shuffle.
- `raster.cu:226,753,796` `__any_sync(0xffffffff, ...)` early-exit / valid-mask; `raster.cu:795` `__ballot_sync(0xffffffff, ...)` (commented out but present).
- `raster.cu:178,616` `PIXELS_PER_THREAD = (tile_size_x*tile_size_y)/(32*VECTOR_SIZE)` with `//assert blockDim.x==32`; `raster.cu:620-622` shared arrays `[...][4*32]`; pixel addressing `threadIdx.x % tile_size_x`, `threadIdx.x / tile_size_x * PIXELS_PER_THREAD * VECTOR_SIZE`. The whole tile<->thread<->register-buffer mapping is built on a 32-lane warp == one tile-row group.
- `compact.cu:478-499` `frustum_culling_aabb_kernel`: `__ballot_sync(0xffffffff, global_visible)` into an `unsigned` (32-bit) mask, `lane_id = threadIdx.x & 0x1f`, `__popc(warp_mask & ((1<<lane_id)-1))` exclusive prefix, lane-0 `atomicAdd(&visible_num_in_block, __popc(warp_mask))`, `__shfl_sync(0xffffffff, warp_offset, 0)` broadcast. Block of 256 -> the popsift wave64 leader-election + 32-vs-64 ballot trap (see Risks).

half2-SIMD intrinsics (alpha-blend math, `raster.cu`):
- `raster.cu:76` inline PTX `asm("ex2.approx.f16x2 %0, %1;" : "=r"(...) : "r"(...))` inside `fast_exp_approx(half2)`. **PTX inline asm DOES NOT COMPILE under hipcc** -- hard wall, must rewrite.
- half2 compare-to-mask: `__hle2_mask`, `__hge2_mask`, `__hgt2_mask` (`raster.cu:260,264,266,749,750`) and the SIMD video op `__vcmpleu2` (`raster.cu:751`). HIP coverage uncertain (see Risks).
- `__hmin2`, `h2rcp`, `__hmul2`/operators, `half2` ctors -- standard HIP half2, expected OK.

Libraries / sort:
- `binning.cu:23-24,205,215` `cub::DeviceRadixSort::SortPairs<int,int>(..., 0, bit)` with **begin_bit=0** -> hipCUB drop-in; the cudaKDTree nonzero-begin_bit hipCUB bug does NOT apply (gsplat/LichtFeld precedent). Per-view loop reuses one temp buffer.
- `cooperative_groups.h` + `cooperative_groups/reduce.h` included in all 4 .cu and `namespace cg = cooperative_groups`, BUT grep shows NO `cg::` USES in raster/transform/binning (only `compact.cu` includes; no cg:: call). The reductions are hand-rolled shuffles, NOT `cg::reduce`/`tiled_partition` (this is the key difference from gsplat/LichtFeld, which were wave-agnostic via `tiled_partition<32>`). Include hipifies fine; no cg shim needed unless an unused include errors.
- `<cuda/atomic>` included (`raster.cu:8`, `transform.cu:8`) -- libcudacxx atomic header; hipify maps to the rocm libhipcxx `<hip/atomic>` equivalent. Verify it resolves (no `cuda::atomic` USES found, so a stray include may just need the shim or removal).
- `cuda_errchk.cpp` `cudaDeviceSynchronize`/`cudaGetLastError`/`cudaGetErrorString` -- hipified 1:1.

Atomics: all `atomicAdd` (float gradients `raster.cu:804-847`, int/uint counters `compact.cu:485,493`, statistics `raster.cu:297-298,817`). All on plain device/shared memory (torch tensors are `hipMalloc` device memory, not managed). The cudaKDTree managed-memory atomicMin/Max-silently-dropped class is **N/A** (no atomicMin/Max, no managed memory).

NO textures/surfaces, NO managed memory, NO streams/events, NO pinned/host-registered memory anywhere in the tree (whole-tree grep empty). So: no texture rule-of-five, no 256B pitch, no layered-array, no linear-filter fault classes apply.

### B) simple_knn._C (simple-knn) -- 3 kernels, EASY.
- `simple_knn.cu`: `coord2Morton`, `boxMinMax`, `boxMeanDist`. Uses `cub::DeviceReduce::Reduce` (CustomMin/CustomMax), `cub::DeviceRadixSort::SortPairs` (default begin/end bits = full width), `thrust::device_vector`/`thrust::sequence`. All hipCUB/rocThrust drop-ins. Includes `cooperative_groups` but the kernels are plain index math (no shfl/ballot grep hit). Expect ZERO source edits (mirrors fused-ssim).
- This is the original Inria simple-knn shipped with vanilla 3DGS; widely known to hipify clean.

### C) fused_ssim submodule -- ALREADY PORTED BY MOAT, ZERO source changes.
- Fork `kemchenj/fused-ssim@fused-l1-ssim-loss`. MOAT's `projects/fused-ssim` ported `rahul-goel/fused-ssim` (the parent) with zero kernel edits: the 2D SSIM is a separable 11x11 Gaussian conv with shared-memory tiling, block (16,16)=256, NO warp primitives, the only literal-32 is a benign loop-tiling stride (confirmed wave64- AND wave32-correct). This fork adds an L1+SSIM loss head on the same kernels. Expect the same: build against ROCm torch, zero source changes; re-verify the literal-32 stride is still benign (it is, per fused-ssim notes.md).

## Risk list (ranked; with the precedent that solved each)
1. **sm_80 `__reduce_{add,max}_sync` (11 sites)** -- HIGH. ROCm/clang-HIP does not provide the sm_80 hardware warp-redux as a cross-lane reduce on gfx90a. The CUDA `__reduce_*_sync(mask,v)` returns the warp-wide reduction in every lane. Replacement on HIP: a shuffle butterfly all-reduce (`__shfl_xor` over the wavefront) or fold via `__shfl_down`/shared then broadcast, sized to the wavefront. CRITICAL nuance: the rasterizer's `warp_reduce_sum<float>` uses a clever exponent-rescale-to-int trick (`raster.cu:100-113`) so that `__reduce_add_sync` over INTs is exact/deterministic; the HIP replacement must preserve the int-domain add (a float shuffle-add changes the rounding the original deliberately avoided). Safest: implement a portable `warp_reduce_sum` that does the SAME int/float ops over `kWarpSize` lanes with a 64-bit mask, matching the rescale path. (Precedent: LichtFeld warp_reduce.cuh offset-16->kWarpSize/2 + 64-bit mask; MPPI "drop the warp-synchronous tail, run the syncthreads tree" for plain reductions -- but here the reduce is per-warp leader-atomicAdd-combined, the AutoDock class, so a native-64 butterfly is correct.)
2. **Inline PTX `asm("ex2.approx.f16x2")` (raster.cu:76)** -- HIGH, hard wall. hipcc cannot assemble NVIDIA PTX. Rewrite `fast_exp_approx(half2)` on HIP to use the HIP intrinsic `h2exp2(scaled_input)` (half2 base-2 exp; the code already scales by log2(e) then wants ex2). Guard `#if defined(USE_ROCM)` -> `h2exp2`, else the PTX. Verify `h2exp2` exists in ROCm `<hip/hip_fp16.h>`; if not, fall back to per-element `exp2f` on `__low2half`/`__high2half` then repack, or `__float2half2_rn(exp2f(...))`. Numerically `ex2.approx` is an APPROXIMATE op; `h2exp2` accuracy may differ slightly -- the wrapper tolerances (1e-5 abs / 1e-3 rel) and the training-PSNR gate absorb this. This is the single most likely build-blocker; resolve it first.
3. **Wave64 tile/thread/register mapping in the rasterizer** -- HIGH. `PIXELS_PER_THREAD = tile_area/(32*VECTOR_SIZE)` and `blockDim.x==32` assume a 32-lane warp owns a tile-row group; the forward/backward share `[...][4*32]` shared buffers and pixel addressing keyed on 32. Two valid strategies: (a) KEEP the launch at 32 threads/row-group and force `kWarpSize`-agnostic intrinsics so a 32-thread block runs as a 32-lane SUBSET of a 64-wide wavefront (HIP supports launching <warpSize blocks; the upper 32 lanes are inactive) -- minimal source churn, matches the `0xffffffff` 32-bit-mask shuffles if they operate within the low 32 lanes; OR (b) re-derive the mapping for 64 lanes (PIXELS_PER_THREAD halves, shared `[...][4*64]`), more invasive. Lean toward (a) FIRST (least churn, the shuffles' width-32 semantics then match a 32-lane active group), and verify empirically via the wrapper script-vs-fused comparison + a determinism check. If (a) underperforms or the half2-per-2-pixels packing fights it, fall back to (b). (Precedent: popsift "treat the wavefront as two 32-lane groups / width-32 shuffles"; but LiteGS uses ONE 32-row-group per block, so a single low-32 active group is the natural analog.)
4. **frustum_culling_aabb 32-vs-64 ballot leader-election (compact.cu:478-499)** -- HIGH (the popsift wave64 leader-election trap, exact fingerprint). On wave64: `__ballot_sync` returns a 64-bit mask but it is stored in `unsigned` (truncated to low 32) -> wrong popcount for the upper 32 lanes; `lane_id = threadIdx.x & 0x1f` makes lane 0 appear TWICE per 64-wavefront so the `if(lane_id==0) atomicAdd(__popc(warp_mask))` FIRES TWICE and double-counts; `__popc` (32-bit) on a 64-bit ballot drops the upper lanes. This produces a wrong/overflowing `visible_chunk_id` compaction index -> OOB scatter or dropped chunks. Fix on HIP: 64-bit ballot (`unsigned long long`), `__popcll`, full-wavefront lane = `threadIdx.x & (kWarpSize-1)`, single leader per 64-lane wavefront, broadcast width = kWarpSize. (Precedent: popsift extrema_count "one full-wavefront 64-bit ballot, __popcll, exclusive prefix __popcll(ballot & ((1<<wflane)-1))"; fingerprint = stable primary count but wrong/nondeterministic compaction.)
5. **half2 compare-mask intrinsics `__hle2_mask`/`__hge2_mask`/`__hgt2_mask`/`__vcmpleu2`** -- MEDIUM. `__h*2_mask` return a 32-bit mask (0xFFFF per lane) on CUDA; HIP `<hip/hip_fp16.h>` provides `__hge2`/`__hle2` returning a `half2` of 1.0/0.0, not the `*_mask` 0xFFFF form, and may NOT provide the `_mask` variants. `__vcmpleu2` is a SIMD-in-a-word video intrinsic (`<sm_..._intrinsics>`); HIP has a `__vcmpleu2`-family in some versions but coverage is uncertain. Plan: provide `USE_ROCM` helpers that compute the same 0xFFFF/0x0000-per-half mask from `__hge2`/`__hle2` (compare -> select bits), and reimplement `__vcmpleu2` (per-16-bit unsigned `<=` -> 0xFFFF mask) by hand. These feed `valid_mask &=` and `reinterpret_cast<unsigned*>(&alpha)[0] &= valid_mask` bit-masking, so the mask MUST be the 0xFFFF-per-element form, not a half2 0/1. Verify each intrinsic's HIP presence with a tiny probe before writing shims.
6. **`<cuda/atomic>` libcudacxx include (raster.cu:8, transform.cu:8)** -- LOW/MEDIUM. No `cuda::atomic` USES found, so the include may be vestigial. If hipify/libhipcxx does not resolve it, either drop the include under `USE_ROCM` or map to `<hip/atomic>`. (Precedent: LichtFeld "ProjectionUT uses optional only in comments -- no cuda::std needed".)
7. **`--use_fast_math` -> hipcc** -- LOW. Accepted by hipcc (fused-ssim/gsplat passed it). Watch the clang-HIP default `-ffp-contract=fast` vs nvcc expression-only (CV-CUDA class): the alpha-blend is half2 and the gate tolerances are loose (1e-5/1e-3), so a ~1 ULP FMA drift is absorbed; do NOT pin `-ffp-contract=on` unless a gate test fails by a tight margin.
8. **`__CUDACC__`/`__NVCC__` force-define (raster.cu:1-4)** -- LOW. `#ifndef __CUDACC__ #define __CUDACC__ ...`. Under hipcc `__CUDACC__` is undefined, so this fires and defines it -- which can make included headers (e.g. ATen TensorAccessor, or glm if pulled) take a CUDA path. ATen accessors are fine. If it causes a rocThrust/cub header to take the broken CUDA-system path, guard the force-define out under `__HIPCC__`. (Precedent: LichtFeld/MPPI __CUDACC__-defining-breaks-rocThrust; here it is pre-existing in source, so only touch if it breaks the build.)
9. **fused_ssim fork drift** -- LOW. The kemchenj fork adds an L1+SSIM loss to the already-MOAT-ported kernels; re-confirm no new warp primitive was added in the loss head (grep), otherwise zero edits.

## File-by-file change list (all source edits guarded `#if defined(USE_ROCM)` / `__HIP_PLATFORM_AMD__`; CUDA path byte-identical)
- `gaussian_raster/raster.cu`:
  - `fast_exp_approx` (line ~72-78): USE_ROCM branch using `h2exp2` (or per-element exp2 repack); keep PTX on CUDA.
  - `warp_reduce_sum` specializations + the `__reduce_*_sync` sites (~84-157, 694): a portable `kWarpSize`-aware reduce (64-bit mask, offset start = kWarpSize/2, preserving the int-rescale path) replacing the sm_80 redux; OR keep 32-wide low-lane semantics if strategy 3(a) holds. Decide with the empirical gate.
  - half2 mask helpers `__h*2_mask`/`__vcmpleu2` (~260-267, 749-751): USE_ROCM shim functions returning the 0xFFFF-per-half mask form.
  - `__any_sync`/`__ballot_sync` masks (~226,753,795-796): 64-bit mask on HIP if the reduce goes wave64; else width-32.
  - tile/thread/register mapping constants (~178, 616, 620-622) ONLY if strategy 3(b) is needed.
  - `<cuda/atomic>` include guard if unresolved.
  - `__CUDACC__` force-define guard (~1-4) only if it breaks rocThrust/cub.
- `gaussian_raster/compact.cu`: `frustum_culling_aabb_kernel` (478-499) wave64 ballot/popcll/leader fix (the only correctness-bearing edit outside raster.cu likely).
- `gaussian_raster/binning.cu`, `transform.cu`: expected NO edits (cub begin_bit=0 is fine; transform kernels are plain math). Touch only if a stray intrinsic/include breaks.
- `gaussian_raster/setup.py`, `simple-knn/setup.py`: optionally add a `torch.version.hip` branch to drop NVIDIA-only flags / add `-DUSE_ROCM` and skip `--use_fast_math` quirks IF needed (fused-ssim model). Prefer NO setup.py edit if torch's auto-hipify + default flags build clean; a setup.py change still counts as a source edit but is build-only.
- `simple-knn/*.cu`: expected ZERO edits.
- `fused_ssim` submodule: ZERO edits (reuse MOAT fused-ssim result).
- NO compat header, NO symbol renames (Strategy B). NO GitHub Actions. NO CMake edits.

## Build commands (gfx90a; from the LiteGS src dir or /tmp, NOT /var/lib/jenkins/pytorch)
Initialize the submodule first (fused_ssim): `git submodule update --init --recursive` in the fork clone.
```
export HIP_VISIBLE_DEVICES=<free GCD>      # cap 4 concurrent GPU agents
export PYTORCH_ROCM_ARCH=gfx90a            # also pass gfx1100 at follower time, no source change
cd projects/LiteGS/src
/opt/conda/envs/py_3.12/bin/python -m pip install -e litegs/submodules/simple-knn      --no-build-isolation -v
/opt/conda/envs/py_3.12/bin/python -m pip install -e litegs/submodules/fused_ssim       --no-build-isolation -v
/opt/conda/envs/py_3.12/bin/python -m pip install -e litegs/submodules/gaussian_raster  --no-build-isolation -v
```
Torch auto-hipifies each `.cu`/`.cuh` (gitignored `.hip` mirror) and links `amdhip64`/`c10_hip`/`torch_hip`. INCREMENTAL gotcha (Strategy B): after editing a `.cu`, torch may recompile the STALE hipified mirror -- delete the extension's `build/` (and any `*.hip`) before rebuilding so the edit is re-hipified. The arch is taken from `PYTORCH_ROCM_ARCH`; a follower needs only `PYTORCH_ROCM_ARCH=gfx1100` + clean rebuild, no source edit (no churn to head_sha).

## Test plan (real GPU, gfx90a)

### Tier 1 -- kernel-vs-reference (the built-in gate, mirrors gsplat/fused-ssim)
`litegs/utils/wrapper.py` defines `BaseWrapper` subclasses that each run the FUSED CUDA kernel and a pure-PyTorch `script` reference over `torch.randn` inputs and assert agreement at `_absolute_error_threshold=1e-5` / `_relative_error_threshold=1e-3` (some relax to 1e-2/5e-2), forward AND backward (autograd). `wrapper.check()` runs all subclasses. Covered fused kernels: `CreateTransformMatrix`, `CreateRaySpaceTransformMatrix`, `CreateCov2dDirectly`, `SphericalHarmonicToRGB` (sh2rgb fwd+bwd), `EighAndInverse2x2Matrix`, `Binning` (duplicate_with_keys + tileRange + the cub sort). This is the primary correctness gate for transforms/SH/binning. A failing wrapper = wave64/intrinsic bug. Run twice for bitwise determinism (the atomicAdd-combined reductions + the ballot compaction are the nondeterminism risk; popsift fingerprint = stable primary, nondeterministic secondary).
- Entry: import `litegs.utils.wrapper` and call `wrapper.check()` (or instantiate each subclass `.validate()`), `HIP_VISIBLE_DEVICES` pinned. (Confirm/locate a `check_wrapper.py` driver the README:139 mentions; if absent, a 3-line harness calling `wrapper.check()` suffices -- put it in agent_space.)

### Tier 2 -- rasterizer forward+backward correctness (NOT covered by Tier 1)
`GaussiansRasterFunc` (the alpha-blend fwd/bwd with the half2 SIMD + sm_80 reduces + the tile/warp mapping) is the highest-risk kernel and has NO `BaseWrapper` script twin. Validate by:
- A direct render: build a small synthetic scene (a few hundred to few thousand Gaussians), call `litegs.render` forward, check the image is finite (no NaN/Inf), non-trivial, and BITWISE-DETERMINISTIC across two runs (rules out the wave64 reduction race + ballot nondeterminism).
- Gradient check: backward through a sum-of-output loss, assert grads finite and deterministic; cross-check a coarse finite-difference or compare the fused path against the pure-Python `call_script` rasterization path if one exists in `litegs/render` (the modular design exposes a Python rendering path -- use it as the oracle, the gsplat reference-vs-CUDA model).

### Tier 3 -- end-to-end short training convergence (the integration gate)
A real 3DGS training run on a small COLMAP scene for a few hundred iterations, asserting the LOSS DECREASES monotonically-ish and PSNR rises into a sane range (no NaN, no divergence), via `example_train.py --sh_degree 3 -s <scene> -i <images> -m <out>` (or `3dgs_challenge_train.py`). "Validated on gfx90a" = (a) all Tier-1 wrappers PASS at their tolerances, (b) the rasterizer render+grad are finite and bitwise-deterministic across runs, AND (c) a short training run converges (loss down, PSNR up, no NaN) on a small scene. This mirrors how the splatting precedents were closed (gsplat: render+grad vs reference at 1e-4; fused-ssim: fwd+grad vs reference + determinism). Need a small dataset on the host -- a few-hundred-image COLMAP scene (e.g. a Mip-NeRF360 scene or the 3dgs-challenge sample); if none is present, fetch a small standard scene or synthesize a tiny COLMAP set. Record the exact scene + iteration count in notes.md.

### Non-GPU regression set
LiteGS has no CPU unit-test suite (the pipeline is GPU-only). The "non-GPU must-not-regress" set is effectively the pure-Python `script` reference paths inside wrapper.py / `litegs.render` -- they must keep working unchanged (we touch only `.cu`). No CPU test to regress.

## Staged strategy + likely walls
1. **Build simple-knn + fused_ssim first** (expected zero-edit, like fused-ssim) -- proves the ROCm-torch CUDAExtension path on this tree and gives quick wins. Wall: fused_ssim submodule must be `git submodule update --init` (its url is the kemchenj fork).
2. **Build litegs_fused; clear the inline-PTX wall FIRST** (`fast_exp_approx` -> `h2exp2`) -- it is a hard compile blocker. Then the sm_80 `__reduce_*_sync` (won't compile/link on HIP) and the half2 `*_mask`/`__vcmpleu2` shims. Get a clean COMPILE before worrying about numerics.
3. **Wave64 correctness pass**: the frustum-cull ballot leader-election (compact.cu) and the rasterizer reduce/tile-mapping. Drive entirely by Tier-1 wrappers + Tier-2 determinism. Likely walls, with templates:
   - The rasterizer's exponent-rescale int-reduce: a naive float shuffle-add reorders the adds the original rescale avoided -> small drift or nondeterminism; preserve the int-domain reduce (AutoDock atomicAdd-combine class -- native-64 butterfly is correct since each warp's partial is leader-atomicAdd'd).
   - The 32-vs-64 tile/register mapping: try strategy 3(a) (32-thread block = low-32-active wavefront, width-32 shuffles) first; the `0xffffffff` masks and offset-16 reduce then match a 32-lane group with no math change. If half2-per-2-pixel packing or shared-buffer sizing fights it, go to 3(b) (re-derive for 64 lanes). The empirical tell is the wrapper compare + bitwise determinism.
   - The frustum-cull double-fire (popsift fingerprint): stable visible-COUNT but wrong/nondeterministic `visible_chunk_id` order or OOB. Fix = 64-bit ballot + __popcll + single 64-lane leader.
4. **Tier-3 short training** once kernels are correct; absorb the `h2exp2`-vs-`ex2.approx` and FMA-contract ~1 ULP drift in the PSNR/loss gate.

## Open questions
- Does `h2exp2` exist and match `ex2.approx.f16x2` closely enough on gfx90a, or is per-element `exp2f` repack needed? (Probe at build time; numerics absorbed by gate tolerances.)
- Are `__hge2`/`__hle2` (to synthesize the `*_mask` form) and a `__vcmpleu2` equivalent available in ROCm `<hip/hip_fp16.h>`? (Tiny probe before writing shims.)
- Strategy 3(a) vs 3(b) for the tile mapping: resolve empirically; 3(a) preferred for minimal churn. The reviewer/porter should read each shuffle site to confirm width-32 low-lane semantics before committing to 3(a).
- Is there a shipped `check_wrapper.py` driver, or do we add a 3-line `wrapper.check()` harness in agent_space? (README:139 references it; locate or create.)
- Small COLMAP scene availability on the host for Tier-3 (fetch a Mip-NeRF360 scene or the 3dgs-challenge sample if absent).
- `<cuda/atomic>` include: vestigial (droppable) or does libhipcxx resolve it? (Build will tell.)

## Disposition
Strategy B mechanical bring-up + GPU validation. This is the 3rd gaussian-splatting port; reuse the gsplat/fused-ssim torch-hipify model and the LichtFeld-Studio wave64-reduction findings, but expect MORE source surgery than either because LiteGS's rasterizer uses raw `__shfl`/sm_80 `__reduce_*_sync`/inline PTX/half2-SIMD-masks instead of wave-agnostic cooperative-groups tiles. Correctness-first; no AMD-native rewrite warranted (no CUTLASS/wgmma). Append the LiteGS-specific lessons (sm_80 redux replacement, ex2.approx->h2exp2, half2 mask shims, the 32-thread-block-on-wave64 tile-mapping resolution) to the PORTING_GUIDE changelog as they are confirmed.
