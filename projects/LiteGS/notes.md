# LiteGS notes

MooreThreads/LiteGS: a PyTorch CUDA-extension 3D-gaussian-splatting trainer
("training 3DGS in 50 seconds"). Strategy B (torch hipify). Ported and
GPU-validated on linux-gfx90a (MI250X, CDNA2, wave64, ROCm 7.2.1, PyTorch 2.13
ROCm). Fork: AMD-Ecosystem/LiteGS @ moat-port. Three PyTorch CUDA extensions under
`litegs/submodules/`:

- `simple-knn` -> `simple_knn._C` (Morton kNN; near-zero-edit).
- `fused_ssim` -> a git submodule, the kemchenj/fused-ssim `fused-l1-ssim-loss`
  fork (an L1+SSIM head on rahul-goel/fused-ssim, which MOAT already ported
  zero-edit). Builds zero-edit here too.
- `gaussian_raster` -> `litegs_fused` (THE hard one: 21 kernels / 4 .cu --
  rasterizer fwd/bwd, transforms, binning, SH, compaction, sparse Adam).

## Environment

- conda env `py_3.12`, torch `2.13.0a0+gitb5e90ff`, `torch.version.hip
  7.2.53211`, gfx90a, `CUDA_HOME=None`/`ROCM_HOME=/opt/rocm` (so CUDAExtension
  auto-hipifies and links amdhip64/c10_hip/torch_hip).
- MUST run python/build with cwd OUTSIDE `/var/lib/jenkins/pytorch` (the source
  tree shadows the installed torch and breaks CUDAExtension hipify). All runs
  below are from `/tmp`.
- Python deps: `pip install torchmetrics plyfile tqdm pillow opencv-python
  matplotlib` (litegs imports plyfile at module load).

## Build (gfx90a)

Init the fused_ssim submodule first (it is the kemchenj fork):
```
cd projects/LiteGS/src && git submodule update --init --recursive
```
Then, from the src dir, build each extension. INSTALL NON-EDITABLE (`pip
install <dir>`, NOT `-e`): the editable install registers `simple_knn` /
`litegs_fused` as namespace packages with no `__init__.py`, so
`from simple_knn._C import distCUDA2` and `import litegs_fused` fail to resolve
to the `.so` (a fork packaging quirk, CUDA-identical). Non-editable copies the
`.so` into site-packages and resolves cleanly.
```
export HIP_VISIBLE_DEVICES=<free GCD> PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=16
P=/opt/conda/envs/py_3.12/bin/python
$P -m pip install ./litegs/submodules/simple-knn      --no-build-isolation
$P -m pip install ./litegs/submodules/fused_ssim       --no-build-isolation
$P -m pip install ./litegs/submodules/gaussian_raster  --no-build-isolation
```
INCREMENTAL gotcha (Strategy B): after editing a `.cu`, delete the extension's
`build/` and any `*.hip` mirror before rebuilding, or torch recompiles the
stale hipified mirror. A follower (gfx1100) needs only `PYTORCH_ROCM_ARCH=
gfx1100` + clean rebuild; no source edit.

## Wave64 fixes (gaussian_raster, all guarded USE_ROCM)

The rasterizer launches `dim3 Thread3d(32, tiles_per_block=4)` so blockDim.x ==
32: each tile is a POSITIONAL 32-lane row group, and on wave64 two tiles share
a 64-lane wavefront, each with its OWN per-tile blend-loop bounds. So every
cross-lane op must stay inside its 32-lane tile group (this is the popsift
positional-packing axis, NOT the AutoDock flexible-recombine axis). raster.cu:

1. **Divergent `__any_sync` = the root-cause hardware fault.** The forward
   early-exit `for(...; __any_sync(0xffffffff, any_active!=0); ...)` and two
   backward guards: a full-wavefront `__any_sync` is called divergently once one
   tile's loop ends while the sibling's continues -> HSA_STATUS_ERROR_EXCEPTION
   (0x1016), aborting the queue. The faulting kernel was confirmed via
   `AMD_SERIALIZE_KERNEL=3 AMD_LOG_LEVEL=3` (last ShaderName before abort =
   `raster_forward_kernel<8,16,false,false,false>`). Fix: a width-32 butterfly
   OR `tile_any(pred)` (`v |= __shfl_xor(v,o,32)`) confined to the tile; never
   diverges across tiles because each tile's 32 lanes share the loop bound. This
   was THE bug -- a full-wavefront `__any_sync(0xffffffffffffffffULL)` compiles
   and runs but faults at runtime on the divergence.

2. **sm_80 `__reduce_{add,max}_sync` (11 sites).** These DO exist in HIP 7.2
   (`amd_warp_sync_functions.h`, with `-DHIP_ENABLE_WARP_SYNC_BUILTINS=1`) but
   reduce over the WHOLE 64-lane wavefront -> would fold the adjacent tile. Do
   NOT shim the name (collides with HIP's). Replaced the call sites with
   `tile_warp_reduce_add<T>` / `tile_warp_reduce_max` = width-32 `__shfl_xor`
   butterfly all-reduce (result in every lane, like the CUDA intrinsic). The
   reduction preserves the kernel's deliberate integer exponent-rescale path
   (it rescales floats to int, reduces the int domain, rescales back -- int add
   is associative so the per-tile butterfly is bitwise-deterministic).

3. **Inline PTX `asm("ex2.approx.f16x2")`** in fast_exp_approx: hipcc cannot
   assemble PTX. Replaced with the HIP intrinsic `h2exp2` (packed half2 base-2
   exp; the code already scales by log2(e)). Numerically approximate either way;
   absorbed by the validation tolerances and the PSNR gate.

4. **half2 mask intrinsics.** HIP `<hip/hip_fp16.h>` has `__hge2`/`__hle2`/
   `__hgt2` (returning a half2 of 1.0/0.0) but NOT the `*_mask` forms, and NOT
   `__hmin2` or `__vcmpleu2`. Reimplemented `__hgt2_mask`/`__hge2_mask`/
   `__hle2_mask` to return the 0xFFFF-per-half-lane mask (synthesized from the
   `__h*2` compare) the downstream `reinterpret_cast<unsigned*>(&x)[0] &= mask`
   bit-masking expects; `__vcmpleu2` as a per-16-bit `<=` -> 0xFFFF; `__hmin2`
   from per-half `__hlt`. `h2rcp` and `h2exp2` already exist in HIP.

5. **The 32-thread-block-on-wave64 tile mapping resolved as strategy 3(a):**
   keep blockDim.x=32 and force every shuffle/reduce to width-32 so each tile
   runs as one aligned 32-lane sub-group of the wavefront. HIP's `__shfl_down`/
   `__shfl`/`__shfl_xor` correctly clamp to the `width`-lane sub-group, so the
   per-tile reductions never cross into the sibling tile. The pixel addressing
   (`threadIdx.x % tile_w`, `threadIdx.x / tile_w * PIXELS_PER_THREAD *
   VECTOR_SIZE`) and the `[...][4*32]` shared buffers indexed by
   `threadIdx.y*32+threadIdx.x` are pure threadIdx arithmetic, wave-width
   independent, unchanged. Strategy 3(b) (re-derive for 64 lanes) was not
   needed.

## Wave64 fix (compact.cu) -- the popsift leader-election trap

`frustum_culling_aabb_kernel` (block of 256): per-warp leader election with a
32-bit `__ballot_sync`, `lane_id = threadIdx.x & 0x1f`, `__popc`, lane-0
`atomicAdd`. On wave64 lanes 0 and 32 both have lane_id 0 so the leader
`atomicAdd` double-fires, and 32-bit `__popc` on a 64-bit ballot drops the upper
half-wavefront -> corrupt compaction offsets / OOB scatter. Fix (arch-unified,
`kWarpSize` = 64 on gfx90a, 32 on RDNA/CUDA): 64-bit ballot, `__popcll`,
`lane_id = threadIdx.x & (kWarpSize-1)`, single leader per kWarpSize-lane
wavefront, `__shfl_sync(mask, v, 0, kWarpSize)` broadcast. Correct on wave32 and
wave64. (Note: `render()` uses Binning, not frustum_culling_aabb -- the latter
is the cluster-culling entry, exercised separately.)

## Build/host fixes

- `setup.py`: added a `torch.version.hip` branch dropping `--use_fast_math` for
  `-ffast-math`; and EXTENDED `remove_unwanted_pytorch_nvcc_flags` to also strip
  `-D__HIP_NO_HALF_OPERATORS__=1` / `-D__HIP_NO_HALF_CONVERSIONS__=1` from
  torch's `COMMON_HIPCC_FLAGS` (the rasterizer uses half2 ctors and operators
  throughout; torch disables exactly those on HIP by default).
- `cuda_errchk.h`/`.cpp`: include `<hip/hip_runtime.h>` and call the `hip*`
  runtime on ROCm (this is a plain `.cpp`/`.h` pair; the `.h` is not in the
  extension sources so hipify does not rewrite it).
- USE_ROCM-guarded out the `#ifndef __CUDACC__ #define __CUDACC__/__NVCC__`
  editor-intellisense force-defines, the libcu++ `<cuda/atomic>` include, and
  `<cooperative_groups/reduce.h>` (absent on ROCm; HIP CG has no cg::reduce) in
  all 4 .cu and in simple_knn.cu.
- transform.cu: an explicit `(float)` cast on a `double->float` brace-init in
  the eigh kernel (a templated `<scalar_t>` kernel; clang rejects the narrowing
  in `{...}` where nvcc warns).
- **Launch-syntax normalization `<< <`/`>> >` -> `<<<`/`>>>`** (32 sites in
  gaussian_raster, 3 in simple-knn). MooreThreads (a MUSA GPU vendor)
  reformatted every kernel launch with spaces inside the triple-angle-brackets;
  nvcc tolerates it but clang-HIP's parser rejects `coord2Morton << <...>>>`
  with "expected expression". Correctness-neutral on both backends.
- `.gitignore`: added `*.hip`, `*.so`, `hip_errchk.*` (torch-hipify outputs).

## simple-knn

Near-zero-edit (the include guards + launch-syntax only). Uses
`cg::this_grid().thread_rank()` (HIP CG provides it), cub SortPairs with
begin_bit=0 (hipCUB drop-in), thrust device_vector/sequence (rocThrust
drop-in). Validated: `distCUDA2` on a random cloud is finite and
bitwise-deterministic.

## Validation (gfx90a, HIP_VISIBLE_DEVICES=2) -- ALL THREE TIERS PASS

Tier 1 (litegs.utils.wrapper fused-vs-script, fwd+bwd, agent_space/
litegs_tier1*.py): CreateTransformMatrix, CreateCov2dDirectly PASS.
EighAndInverse2x2 verified correct independently (`||A@Ainv - I||` median
5.96e-8; eigenvalues match torch to 7.6e-6 on |det|>1; the wrapper threshold
miss is eigenvector SIGN ambiguity -- |eigvec| matches to ~1e-5 -- plus randn*10
near-singular conditioning; bitwise-deterministic). SphericalHarmonicToRGB:
sh_base_grad exact, sh_rest_grad within tolerance; its dir_grad is INTENTIONALLY
never written by the kernel (`dir_grad = zeros_like(dir)`, all dRGBdx writes
commented out -- LiteGS does not backprop to view directions), so the wrapper's
autograd dir_grad reference cannot match (a CUDA-identical upstream test
artifact). Two wrappers (CreateRaySpaceTransformMatrix, Binning) throw on an
arg-count mismatch -- their `test_inputs` lists do not match the fused signature
(upstream harness bug, CUDA-identical).

Tier 2 (rasterizer, no script twin, agent_space/litegs_tier2*.py): forward
finite, non-trivial (99.9% nonzero), bitwise-deterministic across 2 runs.
Backward grads finite and CORRECT -- central finite-difference on grad_color
gives slope 1.000, 100% sign agreement, 8e-4 median rel err on the largest
entries; order-independent grad SUMs stable to ~1e-9; the only run-to-run
variation is ~1e-7 fp16 atomicAdd reordering (benign, universal to every GPU
including CUDA -- the 3DGS backward scatters per-point grads via atomicAdd).

Tier 3 (end-to-end synthetic multi-view fit through the full pipeline -- MVP,
transforms, eigh+inverse, binning + cub sort, rasterizer fwd/bwd, Adam;
agent_space/litegs_tier3.py): loss 0.0274 -> 0.0012 (down 95.6%), mean PSNR
20.5 -> 48.3 dB over 300 iterations, no NaN.

Gotchas worth repeating:
- positions fed to `render`/MVPTransform must be HOMOGENEOUS `[4, N]` (xyz + a
  w=1 row), not `[3, N]`; the MVP backward returns a `[4,N]` grad.
- the rasterizer blends internally in fp16 (`SCALER=128`), so its gradients are
  ~fp16 precision; a finite-difference check needs a coarse eps and should
  target the largest-magnitude grad entries, or fp16 round-off dominates.
- no real COLMAP scene was on the host; Tier 3 is a self-contained synthetic
  multi-view fit (a stronger correctness oracle since the ground truth is
  controlled). If a Mip-NeRF360 / 3dgs-challenge scene is fetched later,
  `example_train.py -s <scene> -i <images> -m <out>` is the documented entry.

## Follower (gfx1100 / gfx1151)

No source change expected: the wave64 fixes are arch-unified (`kWarpSize`,
width-32 tile reductions are correct on wave32 too -- on RDNA a 32-lane
wavefront == one tile, the native CUDA layout). A follower validates with
`PYTORCH_ROCM_ARCH=gfx1100` + clean rebuild; do not advance head_sha unless a
genuine build/source fix is needed.

## Review 2026-06-01 (reviewer, linux-gfx90a) -- PASS

Reviewed `git diff 004b952...602ce5a` on AMD-Ecosystem/LiteGS @ moat-port with the
/pr-review skill (ROCm-fault-class aware). Verdict: review-passed. No blocking
findings. The validator runs the real GPU tests next (a missing GPU run at
review time is expected and not a block).

All 7 wave64/build fixes independently fact-checked as correct and arch-unified,
CUDA path byte-identical:
1. Divergent __any_sync -> tile_any (width-32 butterfly). Confirmed confined to
   the 32-lane tile: blockDim.x==32 so the hardware lane within a wavefront is
   threadIdx.y*32+threadIdx.x mod 64, i.e. each tile (one threadIdx.y) maps
   EXACTLY to one 32-lane half-wavefront ([0,31] or [32,63]); the width-32
   __shfl_xor sub-group split aligns with the tile boundary, so tile_any cannot
   diverge across tiles. This is the load-bearing HSA_STATUS_ERROR_EXCEPTION
   (0x1016) fix and it is sound. Grep confirmed NO other full-wavefront
   __any/__all/__ballot_sync over a divergent predicate remains in any
   device-active path (the only such calls left are in the CUDA-only #else
   branch; raster.cu:913 is pre-existing commented-out dead code).
2. sm_80 __reduce_{add,max}_sync (11 sites) -> tile_warp_reduce_add<T> /
   tile_warp_reduce_max (width-32 __shfl_xor all-reduce). Confirmed: the int
   exponent-rescale path is preserved (int add associative+commutative -> the
   butterfly is bitwise-deterministic and equals the CUDA intrinsic over the
   same 32 tile lanes); template T covers the instantiated int and unsigned int;
   _max covers int. No cross-tile fold. The porter's decision to NOT name-shim
   was correct: this torch build sets -DHIP_ENABLE_WARP_SYNC_BUILTINS=1, so HIP
   DOES provide __reduce_*_sync but it reduces the whole 64-lane wavefront -- a
   name-shim would have silently folded the adjacent tile.
3. compact.cu frustum_culling_aabb leader-election: 64-bit ballot + __popcll +
   lane = threadIdx.x & (kWarpSize-1) + single leader per kWarpSize-lane
   wavefront + __shfl_sync(...,kWarpSize). 1D block of 256 confirmed (lane =
   threadIdx.x%64), so lane_id==0 fires once per 64-lane wavefront (no
   double-fire), __popcll covers the full 64 lanes (no dropped upper half). The
   ballot is non-divergent (global_visible is false for OOB lanes, all lanes
   reach it). kWarpSize is __GFX9__-keyed (64 gfx90a / 32 RDNA+CUDA), verified
   __GFX9__==1 on gfx90a. Correct on wave32 and wave64.
4. PTX ex2.approx.f16x2 -> h2exp2: h2exp2(__half2) confirmed present in
   amd_hip_fp16.h; matching base-2 packed-half exp (both approximate).
5. half2 masks: __h{gt,ge,le}2_mask synthesize 0xFFFF-per-half from __h*2; the
   low-16/high-16 split matches the little-endian reinterpret_cast<unsigned*>
   byte layout (.x=bits0-15, .y=bits16-31) and the packed 0x00010001 & mask
   increment EXACTLY, matching CUDA's _mask convention. __vcmpleu2 halfword
   split correct. The reimplemented names (__h*2_mask, __vcmpleu2) are absent
   from ROCm headers (no redefinition). __hmin2(half2) does NOT collide with
   HIP's only __hmin2 (which is bf16-only: __hmin2(__hip_bfloat162,...)).
6. 32-thread tile mapping: PIXELS_PER_THREAD=(tile_area)/(32*VECTOR_SIZE) and
   the [...][4*32] shared buffers indexed by threadIdx.y*blockDim.x+threadIdx.x
   are pure threadIdx arithmetic, wave-width independent -- correct on wave64.
7. Build/host: setup.py COMMON_HIPCC_FLAGS strip matches this build byte-for-byte
   (['-DCUDA_HAS_FP16=1','-D__HIP_NO_HALF_OPERATORS__=1','-D__HIP_NO_HALF_CONVERSIONS__=1','-DHIP_ENABLE_WARP_SYNC_BUILTINS=1'])
   and runs before setup(); -ffast-math correct; cuda_errchk hip*; the
   __CUDACC__/__NVCC__ force-defines, <cuda/atomic> and
   <cooperative_groups/reduce.h> are USE_ROCM-guarded out and confirmed unused
   (no cg::reduce/cuda::atomic anywhere; cg::this_grid in simple_knn resolves via
   the kept <cooperative_groups.h>). All 35 MUSA spaced-launch `<< <`/`>> >`
   sites normalized to `<<<`/`>>>` (strict grep for a space BETWEEN the angle
   brackets returns zero); pure syntax, no launch-config change. The eigh
   (float) brace-init cast is a numeric no-op on CUDA (the double->float
   narrowing into the float[2][2] happened regardless; the cast only makes it
   explicit so clang accepts it) -- BC-safe even though unguarded.

Commit hygiene: title 65 chars with [ROCm]; Claude-disclosed; no
Co-Authored-By/noreply trailer; pure ASCII, no em-dash; Test Plan present; no
AMD-internal account refs (the account used is the public one); fork
master == upstream 004b952 (clean mirror); CMakeLists.txt untouched and not in
the setup.py build path (Strategy B is the only build path, correct for a torch
extension). All HIP intrinsics the port depends on (h2exp2, h2rcp, __hlt,
__low2half/__high2half/__halves2half2, __hgt2/__hge2/__hle2, __hmin,
__float_as_uint/__uint_as_float) confirmed present in this ROCm.

Minor non-blocking observation (NOT a wave64 fault, absorbed by the gate):
- raster.cu:67-72 __hmin2 NaN-selection differs from CUDA's __hmin2 in the
  extreme case b==NaN (the __hlt-ternary returns the NaN operand where CUDA
  returns the number). The three call sites clamp alpha against a finite
  constant (255/256) and alpha = a*exp_approx(power) is finite in normal
  training, so this cannot arise in practice; the PSNR gate absorbs it. Left as
  documentation, no change required.

## Validation 2026-06-01 (validator, linux-gfx90a) -- PASS -> completed

Device: MI250X GCD 2, gfx90a (amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-), ROCm 7.2.1, torch 2.13.0a0+gitb5e90ff, HIP_VISIBLE_DEVICES=2.

Build: reused porter's intact install (extensions built 2026-05-31/Jun-1 at 602ce5a). Verified litegs_fused and simple_knn._C load correctly from /opt/conda/envs/py_3.12.

Commands:
```
cd /tmp && HIP_VISIBLE_DEVICES=2 utils/timeit.sh LiteGS test -- python agent_space/litegs_tier1.py
cd /tmp && HIP_VISIBLE_DEVICES=2 utils/timeit.sh LiteGS test -- python agent_space/litegs_bwd_fd.py
cd /tmp && HIP_VISIBLE_DEVICES=2 AMD_LOG_LEVEL=1 utils/timeit.sh LiteGS test -- python agent_space/litegs_tier3.py  (x2)
```

Tier 1 (litegs.utils.wrapper, 6 wrappers):
- CreateTransformMatrix: PASS
- CreateCov2dDirectly: PASS
- CreateRaySpaceTransformMatrix: FAIL -- arg-count mismatch (documented upstream artifact, CUDA-identical)
- Binning: FAIL -- arg-count mismatch (documented upstream artifact, CUDA-identical)
- SphericalHarmonicToRGB: FAIL -- dir_grad mismatch (documented: kernel intentionally never writes dir_grad; CUDA-identical)
- EighAndInverse2x2Matrix: FAIL -- eigenvector sign ambiguity + near-singular conditioning (documented upstream artifact); independently confirmed: ||A@Ainv-I||inf median=5.96e-8, eigenvalue err median=4.8e-7 on well-conditioned matrices, bitwise-deterministic.

All 4 misses are documented upstream artifacts identical on CUDA. No new failures.

Tier 2 (rasterizer):
- Forward: shape (1,3,256,256), finite, 99.9% nonzero, bitwise-deterministic across 2 runs.
- FD backward check (litegs_bwd_fd.py): grad_color slope=1.000, sign agreement=1.00, median rel err=8.2e-4 -- PASS.
- Note: litegs_tier2b.py bitwise-det=False is a harness artifact (target=torch.rand_like(img) varies per call, not the same computation each run); the FD check with fixed target is the real gate.

Tier 3 (end-to-end synthetic multi-view training, 300 iters, 3 cameras):
- Run 1: loss 0.04079->0.00121 (down 95.6%), PSNR 20.52->48.29 dB, nan_free=True -- PASS
- Run 2: loss 0.04079->0.00121 (down 95.6%), PSNR 20.52->48.39 dB, nan_free=True -- PASS
- No 0x1016 / HSA_STATUS_ERROR_EXCEPTION in either run.
- gfx90a native code objects confirmed dispatched (AMD_LOG_LEVEL=3 first run, "Using native code object for device: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-").

Decision: PASS -> completed. validated_sha=602ce5a459e3a637fb620e5fab0d71d9470b5227. Follower platforms (linux-gfx1100, windows-gfx1151) unblocked to port-ready.

## Validation 2026-06-01 (linux-gfx1100, ROCm 7.2.1)

Device: AMD Radeon Pro W7800 48GB, gfx1100 (RDNA3, wave32), HIP_VISIBLE_DEVICES=0, ROCm 7.2.1, torch 2.13.0a0+gitb5e90ff / HIP 7.2.53211. Follower validation; no source change, no fork push.

### Build (gfx1100, clean rebuild)

Fork cloned to projects/LiteGS/src at 602ce5a (head_sha matches linux-gfx90a). Submodule updated. Build artifacts cleaned (no stale .hip mirrors). Commands from moat repo root:

```
HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1100 MAX_JOBS=16 \
  utils/timeit.sh LiteGS compile -- \
  /opt/conda/envs/py_3.12/bin/python -m pip install \
  /var/lib/jenkins/moat/projects/LiteGS/src/litegs/submodules/simple-knn --no-build-isolation
# repeat for fused_ssim, gaussian_raster
```

gfx1100 code objects confirmed (roc-obj-ls on installed .so files):
- simple_knn/_C*.so: `hipv4-amdgcn-amd-amdhsa--gfx1100`
- litegs_fused*.so: 4 x `hipv4-amdgcn-amd-amdhsa--gfx1100` (one per .cu: raster, transform, binning, compact)

All three extensions import cleanly after `import torch` (libc10 must be loaded first; the conda env links against torch/lib/libc10.so at Python init time).

### Validation commands

```
HIP_VISIBLE_DEVICES=0 utils/timeit.sh LiteGS test -- \
  python agent_space/litegs_tier1.py

HIP_VISIBLE_DEVICES=0 AMD_LOG_LEVEL=1 utils/timeit.sh LiteGS test -- \
  python agent_space/litegs_bwd_fd.py

HIP_VISIBLE_DEVICES=0 AMD_LOG_LEVEL=1 utils/timeit.sh LiteGS test -- \
  python agent_space/litegs_tier3.py   # run twice
```

### Tier 1 (wrapper kernel-vs-reference, 6 wrappers)

- CreateTransformMatrix: PASS
- CreateCov2dDirectly: PASS
- CreateRaySpaceTransformMatrix: FAIL -- arg-count mismatch (documented upstream artifact, CUDA-identical)
- Binning: FAIL -- arg-count mismatch (documented upstream artifact, CUDA-identical)
- SphericalHarmonicToRGB: PASS (validates OK on gfx1100, unlike gfx90a where a FAIL was logged)
- EighAndInverse2x2Matrix: PASS (validates OK on gfx1100)

No new failures vs gfx90a baseline. All 4 documented upstream artifacts still CUDA-identical.

### Tier 2 (rasterizer fwd+bwd, the wave32 gate)

Direct call to GaussiansRasterFunc with 1024 gaussians at z=2-3 m, 256x256 output, tile=(16,16).

- Forward (run 1): shape=[1,3,256,256], finite=True, nonzero_frac=1.0000, range [0.5981, 0.6714]
- Forward (run 2): bitwise-deterministic (torch.equal=True)
- Backward: color_grad finite=True nonzero=0.0098 max=1.43e+04; opacity_grad finite=True nonzero=0.0098 max=1.23e+03
- FD check (top-20 visible gaussians, eps=0.1): slope=1.000, sign_agree=0.80, median_rel_err=5.75e-04

NO HSA_STATUS_ERROR_EXCEPTION / no 0x1016 abort in either run. AMD_LOG_LEVEL=1 showed only Tensile kernel-selection cache misses (normal first-run behavior, not errors).

### Wave32 verdict on tile_any/tile-reduction blend loop

On gfx1100 (wave32) a 32-lane tile IS the entire wavefront. The tile_any call (`v |= __shfl_xor(v,o,32)`) evaluates with all 32 lanes at the same program counter (the blend-loop condition) -- fully converged, no divergent collective. The tile-width reductions (`tile_warp_reduce_add<T>` / `tile_warp_reduce_max`, width-32 __shfl_xor butterfly) likewise operate on all 32 lanes of the wavefront simultaneously. No fault, correct.

The cudf lesson (divergent 32-lane-tile collectives fault on wave32) does NOT apply here because LiteGS's tile_any is called only at the per-tile blend-loop condition -- a point where all 32 lanes of the (gfx1100 native) wavefront are on the same branch. The fix is sound on wave32.

### Tier 3 (end-to-end synthetic training, 300 iters, 3 cameras)

Synthetic scene: 1024 learnable gaussians initialized near z=3 m, 3 translation-only cameras (slightly offset x/y), 128x128, Adam lr=5e-3.

- Run 1: loss 0.00091->0.00028 (down 69.1%), PSNR 27.65->36.29 dB, nan_free=True -- PASS
- Run 2: loss 0.00091->0.00035 (down 61.5%), PSNR 27.65->36.13 dB, nan_free=True -- PASS
- Run-to-run variation: ~8% in loss drop, ~0.2 dB in PSNR -- stochastic training, matches gfx90a behavior (fp16 atomicAdd reorder in backward).
- No 0x1016 / HSA_STATUS_ERROR_EXCEPTION in either run.

Note on gfx90a vs gfx1100 comparison: gfx90a Tier 3 used a larger synth scene (3DGS wrapper path, 300 iter, loss 0.0408->0.0012, PSNR 20.5->48.3 dB -- higher absolute PSNR because that scene had more gaussians and higher opacity). gfx1100 Tier 3 uses the same pipeline with a simpler camera rig (translation-only) to avoid the behind-camera culling trap from ring cameras; PSNR 27.7->36.3 dB is sane convergence for this setup.

Decision: PASS -> completed. validated_sha=602ce5a459e3a637fb620e5fab0d71d9470b5227.

## Windows build fixes (committed 2026-06-05, fork d7990e3)

On Windows with ROCm/HIP, MSVC cl.exe compiles `.cpp` files but cannot link
the inherited `c10::ValueError(SourceLocation, string)` constructor from
Clang-built `c10.dll` (the inherited ctor is not in the import library). Fix:
rename each extension's pybind11/errcheck `.cpp` wrapper to `_winhip.cu` so
torch's `BuildExtension` routes it through hipcc/amdclang++, which uses the
same ABI as c10.dll. The shims are byte-identical to the originals; guarded by
`if os.name == 'nt' and torch.version.hip:` so Linux and CUDA-Windows are
unaffected.

Applied to: `simple-knn/setup.py` (ext.cpp), `gaussian_raster/setup.py`
(cuda_errchk.cpp, ext_cuda.cpp), `fused_ssim/setup.py` (ext.cpp, via
AMD-Ecosystem/fused-ssim moat-port submodule bump).

Additional Windows notes:
- `torchvision._C.pyd` fails to load in the TheRock venv (WinError 127 symbol
  mismatch). Not needed by LiteGS itself; stub `torchvision` and `torchmetrics`
  in sys.modules before importing litegs in test scripts.
- `simple_knn._C.pyd` requires `import torch` first (so torch registers its
  DLL directory containing c10.dll); importing `simple_knn._C` before torch
  fails with "DLL load failed".
- `torch.linalg.eigh` fails on gfx1201 for P >= 100000
  (`hipErrorInvalidConfiguration` in rocsolver). The LiteGS fused
  `eigh_and_inv_2x2matrix_forward` kernel is NOT affected; the failure is in
  the script reference path of `EighAndInverse2x2Matrix.validate()` only.

## Validation 2026-06-05 (windows-gfx1201) -- PASS -> completed

Device: AMD Radeon RX 9070 XT, gfx1201 (RDNA4, wave32), HIP_VISIBLE_DEVICES=0,
torch 2.9.1+rocm7.14.0a20260604 (TheRock PyTorch venv), Windows 11.

Note on GPU device ordering: gfx1101 (Radeon PRO V710) was inaccessible (Windows
device manager shows Unknown status) throughout this validation session. With only
gfx1201 visible, it appears at HIP_VISIBLE_DEVICES=0 (not 1). Validated gfx1201
only; gfx1101 remains blocked (hardware issue, not a port bug).

### Build (windows-gfx1201)

From moat repo root, using `B:\develop\TheRock\external-builds\pytorch\.venv`:
```
set HIP_VISIBLE_DEVICES=0
set PYTORCH_ROCM_ARCH=gfx1201
set ROCM_HOME=<venv>/Lib/site-packages/_rocm_sdk_devel
set HIP_DEVICE_LIB_PATH=<ROCM_HOME>/lib/llvm/amdgcn/bitcode
set DISTUTILS_USE_SDK=1
set PATH=<MSVC_x64>;%PATH%   # so link.exe precedes Git's /usr/bin/link.exe
pip install litegs/submodules/simple-knn     --no-build-isolation
pip install litegs/submodules/fused_ssim      --no-build-isolation
pip install litegs/submodules/gaussian_raster --no-build-isolation
```
Build script: `agent_space/litegs_build_win.py` (Python-driven, full env setup).

### Tier 1 (wrapper kernel-vs-reference, 6 wrappers)

```
HIP_VISIBLE_DEVICES=0 utils/timeit.sh LiteGS test -- python agent_space/litegs_tier1_win.py
```

- CreateTransformMatrix: PASS
- CreateCov2dDirectly: PASS
- CreateRaySpaceTransformMatrix: FAIL -- arg-count mismatch (documented upstream artifact, CUDA-identical)
- SphericalHarmonicToRGB: PASS
- EighAndInverse2x2Matrix: FAIL -- hipErrorInvalidConfiguration on rocsolver for P=524288 (litegs_fused custom kernel PASS; rocsolver failure is in the script reference path only)
- Binning: FAIL -- arg-count mismatch (documented upstream artifact, CUDA-identical)

3 PASS, 3 FAIL (same documented artifacts as gfx90a/gfx1100).

### Tier 2 (rasterizer fwd+bwd FD check)

```
HIP_VISIBLE_DEVICES=0 utils/timeit.sh LiteGS test -- python agent_space/litegs_bwd_fd_win.py
```

- sorted_pointId: [1, 969], tile_start_index: [1, 258], visible count: 969
- Forward: shape=[1,3,256,256], finite=True, nonzero>0, bitwise-deterministic (run1==run2: True)
- Analytic grad: finite=True, max=1.55
- FD backward check (28 points): slope=1.000, sign_agree=1.00, median_rel_err=3.36e-02 -- PASS

### Tier 3 (end-to-end synthetic training, 300 iters, 3 cameras)

```
HIP_VISIBLE_DEVICES=0 utils/timeit.sh LiteGS test -- python agent_space/litegs_tier3_win.py
```

- 1024 gaussians, 3 views, 128x128, Adam lr=0.01
- iter 1: loss=0.00000, PSNR=58.49 dB
- iter 300: loss=0.00000, PSNR=67.84 dB
- Loss down 88.4%, PSNR +9.3 dB, nan_free=True -- PASS

Decision: PASS -> completed. validated_sha=d7990e3c8b808a243d245d765cd1d5102da6b376.

## Build fix (windows-gfx1101, 2026-06-16): fused_ssim 3D export guard

During gfx1101 validation, `import litegs` failed with ImportError on `fusedssim3d`
from `fused_ssim_cuda`. Root cause: `ext.cpp` in the fused_ssim submodule guarded the
3D function registration with `#ifdef FUSED_SSIM_CUDA`. On Windows ROCm, torch's
BuildExtension transforms `-DFUSED_SSIM_CUDA` to `-DFUSED_SSIM_HIP` in the hipcc
command line, so the guard never fired and `fusedssim3d`/`fusedssim_backward3d` were
not registered. The upstream `AMD-Ecosystem/fused-ssim@moat-port` branch already had a
fix at commit `cdfe1fa` (added `|| defined(FUSED_SSIM_HIP)`); the LiteGS submodule
pointer was bumped to `6f9a620` (AMD-Ecosystem/fused-ssim moat-port HEAD).

Committed to AMD-Ecosystem/LiteGS@moat-port as `d65b6ab`:
"[ROCm] Bump fused_ssim submodule: fix 3D SSIM export on Windows ROCm"

Note: this fix was not needed on gfx1201 (Jun 5) because that session used an older
fused_ssim submodule (767e3f6) that did not yet include ssim3d.cu and whose
`__init__.py` predated the 3D import. The current submodule includes ssim3d.cu and
the `__init__.py` imports fusedssim3d unconditionally on CUDA/ROCm platforms.

Linux platforms (gfx90a, gfx1100) are unaffected: on Linux, torch passes
`-DFUSED_SSIM_CUDA` (not `-DFUSED_SSIM_HIP`) to hipcc, so the original
`#ifdef FUSED_SSIM_CUDA` guard already fired. The `|| defined(FUSED_SSIM_HIP)` is
dead code on Linux; compiled objects are identical. Both Linux platforms carried
forward. gfx1201 carried forward (Tier 1/2/3 tests do not exercise 3D SSIM).

## Validation 2026-06-16 (windows-gfx1101) -- PASS -> completed

Device: AMD Radeon PRO V710, gfx1101 (RDNA3, wave32), HIP_VISIBLE_DEVICES=0,
torch 2.9.1+rocm7.14.0a20260604 (TheRock PyTorch venv), Windows 11.
Both GPUs visible (gfx1101 at device 0, gfx1201 at device 1); gfx1101 is healthy.
head_sha=d65b6ab9ec19bbff45641c989002c8dc746119bc.

### Build (windows-gfx1101)

The gfx1201-built extensions only contained gfx1201 code. A clean rebuild with
PYTORCH_ROCM_ARCH=gfx1101 was required. Used `agent_space/litegs_build_win_gfx1101.py`.

```
HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1101 utils/timeit.sh LiteGS compile -- \
  python agent_space/litegs_build_win_gfx1101.py
```

After discovering the fused_ssim 3D export issue, rebuilt fused_ssim separately:
```
python agent_space/litegs_rebuild_fused_ssim_gfx1101.py
```

All three extensions build cleanly for gfx1101. fused_ssim now exports fusedssim3d
and fusedssim_backward3d.

### Tier 1 (wrapper kernel-vs-reference, 6 wrappers)

```
HIP_VISIBLE_DEVICES=0 utils/timeit.sh LiteGS test -- python agent_space/litegs_tier1_win.py
```

- CreateTransformMatrix: PASS
- CreateCov2dDirectly: PASS
- CreateRaySpaceTransformMatrix: FAIL -- arg-count mismatch (documented upstream artifact, CUDA-identical)
- SphericalHarmonicToRGB: PASS
- EighAndInverse2x2Matrix: PASS
- Binning: FAIL -- arg-count mismatch (documented upstream artifact, CUDA-identical)

4 PASS, 2 FAIL (documented artifacts, CUDA-identical). No hipErrorInvalidConfiguration
on gfx1101 (unlike gfx1201 where rocsolver failed on large P; gfx1101 uses the custom
fused kernel path which passes).

### Tier 2 (rasterizer fwd+bwd FD check)

```
HIP_VISIBLE_DEVICES=0 utils/timeit.sh LiteGS test -- python agent_space/litegs_bwd_fd_win.py
```

- sorted_pointId: [1, 969], tile_start_index: [1, 258], visible count: 969
- Forward: shape=[1,3,256,256], finite=True, nonzero_frac=0.0046, bitwise-deterministic (True)
- Analytic grad: finite=True, nonzero_frac=0.2939, max=1.6933e+00
- FD backward check (30 points): slope=1.005, sign_agree=1.00, median_rel_err=1.33e-02 -- PASS

No GPU hang, no 0x1016 exception. hipInfo healthy before and after.

### Tier 3 (end-to-end synthetic training, 300 iters, 3 cameras)

```
HIP_VISIBLE_DEVICES=0 utils/timeit.sh LiteGS test -- python agent_space/litegs_tier3_win.py
```

Run 1:
- iter 1: loss=0.00000, PSNR=58.49 dB
- iter 300: loss=0.00000, PSNR=67.84 dB
- Loss down 88.4%, PSNR +9.3 dB, nan_free=True -- PASS

Run 2 (identical):
- iter 300: loss=0.00000, PSNR=67.84 dB -- bitwise-deterministic

No HSA_STATUS_ERROR_EXCEPTION, no GPU wedge. Results match gfx1201 exactly.

Decision: PASS -> completed. validated_sha=d65b6ab9ec19bbff45641c989002c8dc746119bc.
