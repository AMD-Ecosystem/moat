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

## Correction 2026-11-19: the Linux validations carried to d65b6ab did not hold

The `completed` records for linux-gfx90a and linux-gfx1100 at d65b6ab were carry-forwards (`source-class`) from runs at 602ce5a, and none of them ever ran the real trainer. They were wrong for two independent reasons:

1. **Broken submodule.** d7990e3 (not only d65b6ab) repointed `.gitmodules` from upstream's `kemchenj/fused-ssim` (branch `fused-l1-ssim-loss`, 409da0b) to `https://github.com/jeffdaily/fused-ssim.git` branch `moat-port`. That URL now answers "Repository not found", so a fresh `git submodule update --init` fails. The commits it named (767e3f6, then 6f9a620 in d65b6ab; both still reachable on AMD-Ecosystem/fused-ssim `moat-port`) are rahul-goel/fused-ssim plus the 3D SSIM and Windows changes, and neither one defines `fused_l1_ssim_loss`, which `litegs/training/trainer.py:145` calls on every step. So training fails for every user at the first iteration. Both carry-forward details said the submodule change was "dead code on Linux". That covered only the `ext.cpp` guard, not the Python API the trainer imports. (The older note above that 767e3f6 "did not yet include ssim3d.cu" is also wrong: 767e3f6's `__init__.py` already imports `fusedssim3d`.)
2. **A real-trainer fault that the synthetic tiers could not reach** (the next section). Even with the right submodule, `example_train.py` aborted on gfx90a after a few epochs.

The Tier 1/2/3 scripts tested kernels and a synthetic fit. They never imported `fused_ssim` through `trainer.py`, so neither failure was visible to them. The real entry point is now part of validation (below).

## Porter round 2026-11-19 (linux-gfx90a): upstream submodule restored, depth-sort fault fixed

Fork commits on `moat-port` (pushed d65b6ab..dcf5784, new commits only, nothing rewritten):
- `a8d1e6f` Restore the upstream fused_ssim submodule: `.gitmodules` and the gitlink are byte-identical to `master` again (`git diff master -- .gitmodules litegs/submodules/fused_ssim` is empty).
- `633f0b7` Zero-initialize view positions before the depth sort: in `transform.cu` `mvp_transform_forward`, `torch::empty` becomes `torch::zeros` for `view_position`. A one-line change, unguarded and arch-independent.
- `dcf5784` [ROCm] Document building the extensions on AMD GPUs: a short paragraph in README "Getting Started". This documentation was missing from the port until now.

### Upstream fused_ssim needs no change on Linux

kemchenj/fused-ssim 409da0b already carries rahul-goel's "amd gpu installation fix (#29)": `setup.py` has a `torch.version.hip` branch that uses `-ffast-math` in place of the nvcc-only flags, and `fused_l1_ssim_loss` / `fusedl1ssim_loss[_backward]` are present. It built and ran unchanged here. The submodule now points only at public, merged upstream sources.

### Windows: open question for jeff

The Windows ROCm build needs, in fused_ssim's `setup.py`, the `_winhip.cu` copy that routes `ext.cpp` through hipcc (MSVC cannot link the inherited `c10::ValueError` constructor from the clang-built `c10.dll`). That change exists only on AMD-Ecosystem/fused-ssim `moat-port` (checked 2026-11-19: rahul-goel/fused-ssim `main` `setup.py` has no `os.name == 'nt'` branch), and that branch is based on rahul-goel `main`, not on kemchenj's `fused-l1-ssim-loss`. With upstream's submodule restored, the Windows build of fused_ssim is expected to fail at link time again. LiteGS's own two setup.py Windows shims (d7990e3) are still in place. The 3D-export guard fix (6f9a620) does not apply, because 409da0b has no ssim3d. Options for jeff: (a) offer the `_winhip.cu` change to rahul-goel/fused-ssim and ask kemchenj to merge it into `fused-l1-ssim-loss`; (b) document Windows as needing a locally patched fused_ssim; (c) ship this port as Linux-only for now. Windows was out of scope for this round, and nothing was changed for it.

### The fault: torch.sort on ROCm corrupts keys with signaling NaNs

Symptom (before 633f0b7, Tanks and Temples `truck`, 3000 iterations): `HSA_STATUS_ERROR_EXCEPTION ... code: 0x1016`, kernel `at::native::index_elementwise_kernel<128,4,...OpaqueType<4>...>`, then `hipErrorLaunchFailure`. It failed in 4 of 4 plain runs, at epochs 2 to 6 (random frame order). Under `AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1` the Python stack lands on `litegs/utils/wrapper.py:741` `allocate_size[i]=allocate_size[i,depth_sorted_index[i]]`. The 0x1016 is torch's device-side index bounds assert (a deliberate OOB index reproduces the same message), not a memory-aperture fault.

Root cause, isolated step by step:
- `view_depth.sort()` occasionally returned indices >= N (stale float bits as int64). Checking right after every sort and dumping on the first bad one showed about 350 indices duplicated and as many missing, sorted values out of order, and unwritten output slots.
- The input is `view_pos[:, 2, :]`. `mvp_transform_forward` allocates `view_position` with `torch::empty`, and the kernel writes only indices below `valid_length`. So the tail of the depth row (the culled-buffer capacity, `1.2 x` the previous epoch's visible chunk count, minus the valid points) is uninitialized memory. In the dump it held 210 NaNs, 6 of them signaling (`0x7f9d652f`, `0x7f82858f`, `0xffb891ea`, ...).
- Replaying the dumped keys corrupts the sort every time (the out-of-range index is the intermittent part, depending on what stale memory sits in the unwritten slots). Quieting the sNaNs, canonicalizing all NaNs, or replacing NaN with 0 gives a correct permutation. torch takes `at::cuda::cub::radix_sort_pairs` -> `hipcub::DeviceRadixSort::SortPairs` for a single segment > 4096.
- A standalone hipCUB program reproduces it with no torch and no data file (rocPRIM 4.6.0, hipCUB 4.7.0 from the ROCm 10.1 pip SDK, gfx90a): n=136064 randn keys + 2000 random-payload quiet NaNs + 50 signaling NaNs -> 33368 duplicated / 41509 missing; 200 qNaN + 6 sNaN -> 403/404; 2000 qNaN alone or 50 sNaN alone -> correct. Registered as `rocprim-radix-sort-float-snan-corrupt` (deferred, rocm-bug-report, not filed; filing on ROCm/rocm-libraries with `bug` + `project: rocprim` is a person's call). CUDA's cub radix sort treats float keys as bit patterns, so upstream never sees this.

```
// hipcc --offload-arch=<gfx> -O2 radix_snan_synth.hip; ./a.out [nqnan=2000] [nsnan=50]
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s\n", hipGetErrorString(e)); return 2; } } while (0)
int main(int argc, char** argv) {
  const int n = 136064, nq = argc > 1 ? atoi(argv[1]) : 2000, ns = argc > 2 ? atoi(argv[2]) : 50;
  std::mt19937 g(0); std::normal_distribution<float> nd; std::uniform_int_distribution<uint32_t> u;
  std::vector<float> k(n); for (auto& x : k) x = nd(g);
  for (int i = 0; i < nq + ns; i++) {
    uint32_t b = (u(g) & 0x80000000u) | 0x7f800000u | (u(g) % 0x3fffffu + 1) | (i < nq ? 0x400000u : 0u);
    memcpy(&k[u(g) % n], &b, 4);
  }
  std::vector<int64_t> v(n); for (int i = 0; i < n; i++) v[i] = i;
  float *dk, *dko; int64_t *dv, *dvo; void* tmp = nullptr; size_t tb = 0;
  CK(hipMalloc(&dk, n * 4)); CK(hipMalloc(&dko, n * 4)); CK(hipMalloc(&dv, n * 8)); CK(hipMalloc(&dvo, n * 8));
  CK(hipMemcpy(dk, k.data(), n * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dv, v.data(), n * 8, hipMemcpyHostToDevice));
  CK(hipcub::DeviceRadixSort::SortPairs(tmp, tb, dk, dko, dv, dvo, n));
  CK(hipMalloc(&tmp, tb));
  CK(hipcub::DeviceRadixSort::SortPairs(tmp, tb, dk, dko, dv, dvo, n));
  CK(hipMemcpy(v.data(), dvo, n * 8, hipMemcpyDeviceToHost));
  std::vector<int> cnt(n, 0); long oob = 0, dup = 0, miss = 0;
  for (int i = 0; i < n; i++) { if (v[i] < 0 || v[i] >= n) oob++; else cnt[v[i]]++; }
  for (int i = 0; i < n; i++) { dup += cnt[i] > 1; miss += cnt[i] == 0; }
  printf("n=%d qNaN=%d sNaN=%d: duplicated=%ld missing=%ld out_of_range=%ld -> %s\n", n, nq, ns, dup, miss, oob, (dup || miss || oob) ? "CORRUPT" : "ok");
  return (dup || miss || oob) ? 1 : 0;
}
```

Why the fix belongs in LiteGS and is safe for upstream to take: sorting uninitialized memory is a latent defect on any backend. With `torch::zeros`, the unused tail has depth 0, `get_allocate_size` already skips those points (it bounds by `valid_length`, and z <= 0.2 is invisible anyway), so their allocate size stays 0 and the prefix sum and binning are unchanged. The cost is one memset of `[B,4,N]` floats. `ndc_position` stays `torch::empty` (it is never sorted). No other sort in the trainer reads uninitialized data (`.sort(` sites: depth here, morton codes in `scene/point.py` over real points, tile blend counts that are fully written; `wrapper.py:692` is the pure-Python binning path, not used by the trainer).

Also seen and left alone (upstream, benign, not ROCm-specific): `frustum_culling_aabb_kernel` writes `visibility[m]` with no `m < M` check, up to 255 bools past the end of an `M`-byte tensor. The caching allocator's 512-byte rounding absorbs it on both backends. The eval branch in `trainer.py` (`--test_epochs`) calls `render_preprocess`/`render` with the old signatures and would raise a TypeError. Use `example_metrics.py` for evaluation instead.

### Environment (this host, this round)

Image Python `/opt/python-3.11-venv` (read-only): torch `2.16.0a0+git6c3af02`, `torch.version.hip` `7.16.26385`, ROCm SDK from pip wheels `rocm-sdk-* 10.1.0` (`ROCM_HOME=.../site-packages/_rocm_sdk_devel`, no `/opt/rocm`), gfx90a MI250X, `HIP_VISIBLE_DEVICES=2`. This torch is a source build (`/var/lib/jenkins/pytorch` @ 6c3af02), not a released wheel. Extensions and Python deps go into the job venv, which sees the image site-packages through a `.pth` file. Deps: `plyfile<1.1 opencv-python-headless<4.11 matplotlib torchmetrics numpy==1.26.4` (plyfile 1.1.5 and opencv 4.11+ pull numpy 2, which this torch was not built against). torchvision is absent, so `example_metrics.py`'s LPIPS (VGG) cannot run and is stubbed to nan. SSIM and PSNR come from the script itself.

Scene: the public 3DGS evaluation bundle `tandt_db.zip` (Tanks and Temples truck/train + Deep Blending playroom/drjohnson, COLMAP `sparse/0` + `images`), 682,628,995 bytes, sha256 `816e62f22a161abbfe841d2a6b10cdf036e297c9fa289b3bfeee9c6ec526d7e1`, from `https://huggingface.co/camenduru/gaussian-splatting/resolve/main/tandt_db.zip` (a mirror of repo-sam.inria.fr `.../datasets/input/tandt_db.zip`, same size). Used `tandt/truck` (251 images, 979x546), kept under the job directory.

Build (from `/tmp`, non-editable as before):
```
export PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=32
git -C <src> submodule update --init --recursive
for d in simple-knn fused_ssim gaussian_raster; do
  python -m pip install --no-build-isolation --no-deps <src>/litegs/submodules/$d
done
```
All three built cleanly. fused_ssim built with zero edits.

Real-trainer runs. `example_train.py` itself is unchanged. A wrapper (`train_probe.py`, job scratch) runs it with `runpy` and wraps `fused_ssim.fused_l1_ssim_loss` to log per-step loss and the rendered-vs-GT PSNR of the training view:
```
python train_probe.py <src> -s tandt/truck -i images -m <out> --iterations 3000
python train_probe.py <src> -s tandt/truck -i images -m <out_eval> --eval --iterations 3000
python metrics_nolpips.py <src> -s tandt/truck -i images -m <out_eval> --eval   # example_metrics.py, LPIPS stubbed
```
- Before 633f0b7: 4/4 plain runs aborted (0x1016, epochs 2-6).
- After 633f0b7: 3/3 runs complete 2761 iterations (11 epochs), all finite, about 29.5 s of training (37 s wall):
  - run 1: iter 1 loss 0.28337 / PSNR 12.09 dB -> iter 2761 loss 0.07198 / 23.97 dB (first-50 mean 0.23253 / 13.52, last-50 mean 0.07702 / 22.80)
  - run 2: 0.31778 / 10.54 -> 0.07016 / 22.43 (0.23046 / 13.75 -> 0.07950 / 22.53)
  - run 3: 0.29844 / 11.25 -> 0.06918 / 22.75 (0.22543 / 13.88 -> 0.07631 / 22.89)
- `--eval` run (220 train / 31 test frames): 2847 iterations, then `example_metrics.py`: Trainingset SSIM 0.7551 / PSNR 22.20 dB; **Testset SSIM 0.7488 / PSNR 21.80 dB** (LPIPS not measured). For scale, upstream's README reports truck at 25.78 dB after the full 30k iterations with 1M primitives.

CUDA no-regression compile: not run (cuda-not-validated: no nvcc on this host, and a headless worker cannot dispatch to the nvcc fleet host). The CUDA-side delta of this round is `torch::empty` -> `torch::zeros` on one host-side tensor allocation plus README text, so the CUDA path changes no device code. It still owes an nvcc compile before the PR is presented.

### PR preparation: commit messages need a squash or rewrite

The three earlier fork commits carry in-house terms that `jargon.py --port LiteGS` reports (602ce5a: 3 hits, "MOAT" and `agent_space` paths; d7990e3: 7 hits, "moat-port" and the personal fork URL, `agent_space` test-plan lines; d65b6ab: 2 hits, plus `audit-commits` "body has no Test Plan"). They were not rewritten, because validated commits are not amended. The three new commits scan clean. Before any upstream PR, the history needs a person-made squash into clean messages (a PR-prep squash to an identical tree; `squash-carry-forward` exists for that).

## Lesson for cuda-to-rocm: rocPRIM radix sort corrupts float keys that mix signaling and quiet NaNs (libraries.md)

Promoted to cuda-to-rocm/references/libraries.md in bb61ae96.

hipCUB/rocPRIM `DeviceRadixSort::SortPairs` on float keys that contain both signaling and quiet NaNs returns a broken permutation: values duplicated and missing, and output slots never written. CUDA's cub sorts the bit patterns and is unaffected. Either NaN kind alone sorts correctly. Uninitialized device memory (a `torch::empty` buffer only partly written by a kernel) is where such keys come from, and `torch.sort` on ROCm inherits the defect. The symptom is usually far from the sort: a later gather on the sorted indices trips torch's device-side index assert (`HSA_STATUS_ERROR_EXCEPTION 0x1016` naming `index_elementwise_kernel`) only when the stale slot happens to hold an out-of-range value, so the failure looks intermittent. Diagnose by checking the sort output right after the sort for duplicates (`bincount(idx).max() > 1`), not only for range. Fix on the caller's side: never sort uninitialized data (zero-fill the buffer or sort only the valid prefix). Observed on ROCm 10.1 pip SDK (rocPRIM 4.6.0, hipCUB 4.7.0) on gfx90a (wave64) and gfx1201 (wave32); other ROCm versions not checked. Exact duplicate/missing counts vary run to run. (LiteGS)

## Review 2026-11-19 (reviewer, linux-gfx90a) -- changes requested

Reviewed `git diff master...moat-port` (004b952...dcf5784, all six commits) with the pr-review checklist and the cuda-to-rocm index. Problems only; each was re-read against the cited code before being recorded.

1. **README.MD:89 claims the ROCm build works on every AMD platform, but the Windows build of fused_ssim is known to fail.** The sentence starts "On AMD GPUs, ..." with no OS qualifier. Upstream's README supports Windows (README.MD:77, the PowerShell line), and the submodule pinned at the head, kemchenj/fused-ssim 409da0b, has no Windows ROCm branch in `setup.py` (checked: no `os.name`/`_winhip` handling). This round's notes say its Windows link step is expected to fail ("Windows: open question for jeff"). Fix: limit the sentence to Linux for now (for example "On Linux with AMD GPUs, ..."), and widen it again only if jeff's Windows decision restores a working build. (Verified by reading README.MD and fused_ssim/setup.py at 409da0b.)
2. **litegs/submodules/simple-knn/setup.py:6 strips a trailing space from the upstream licence header.** This is an unrelated whitespace edit to a file the port changes for other reasons, and it shows up as noise in the upstream diff. Fix: restore the line byte-for-byte (`use ` with the trailing space). (Verified with `git diff master...HEAD -- litegs/submodules/simple-knn/setup.py | cat -A`.)
3. **The 633f0b7 commit body gives the wrong reason the zero tail gets no tiles.** It says "Those points have zero depth, so get_allocate_size already gives them no tiles". That is not why. `get_allocate_size_kernel` returns early for `index >= valid_length[0]` (binning.cu:311) and `allocated_size` is `torch::zeros` (binning.cu:418), so tail points get 0 whatever their depth. The depth only matters to the sort. The notes say this correctly ("it bounds by valid_length, and z <= 0.2 is invisible anyway"). Commits are not amended, so fix the wording in the PR-prep squash message that the notes already require. (Verified.)
4. **Lesson for cuda-to-rocm (rocPRIM radix sort, sNaN plus qNaN) does not say where it was seen.** The skill's convention is that an entry names the ROCm version it was observed on. Add "seen on ROCm 10.1 (rocPRIM 4.6.0, hipCUB 4.7.0), gfx90a; not checked on other arches" before promotion. The claim itself holds: I rebuilt the reproducer from this file with this host's ROCm 10.1 pip SDK hipcc (`hipcc --offload-arch=gfx90a -O2`, scratch only) and ran it on gfx90a. Results: 2000 qNaN + 50 sNaN gives 30010 duplicated / 35641 missing, CORRUPT; 200 + 6 gives 230 / 230, CORRUPT. Each kind alone is ok: 2000 / 50000 qNaN, and 6 / 50 / 500 / 2000 / 50000 sNaN. The exact counts differ from run to run, as the notes describe.

Pipeline caution, not a code finding: a8d1e6f removes the fused_ssim Windows changes that the windows-gfx1201 (d7990e3) and windows-gfx1101 (d65b6ab) passes depended on. Neither Windows record may be carried forward across it.

Checked and holding (recorded so the next round need not redo them): `.gitmodules` and the gitlink equal master's (kemchenj/fused-ssim, branch fused-l1-ssim-loss, 409da0b). `git ls-remote` shows `refs/heads/fused-l1-ssim-loss` at 409da0b, and the pinned `fused_ssim/__init__.py:83` defines `fused_l1_ssim_loss`, which trainer.py:145 calls. Nothing else in the diff names a fork or a private source. For 633f0b7, every reader of `view_pos` past `valid_length` either guards on `valid_length` (transform.cu:37, :397, :491; binning.cu:311) or writes only when `allocated_size > 0` (duplicate_with_keys_kernel). The visible points keep their relative depth order, so the prefix sum and the table are unchanged. The d7990e3 Windows shims are no-ops on Linux and on CUDA, and the source order is the same as upstream's. The wave64 fixes in raster.cu and compact.cu are unchanged since 602ce5a (`git diff 602ce5a HEAD --stat` touches neither). The three new commit messages scan clean with `jargon.py --port LiteGS`, carry no Co-Authored-By trailer, and use `[ROCm]` only on dcf5784.

## Porter round 2026-11-19 (2): review findings addressed

- Finding 1 (README scope): fork `38aa67c` "[ROCm] Scope the AMD GPU build note to Linux".
- Finding 2 (licence-header whitespace from d7990e3): fork `867da59` restores the line; `git diff master -- litegs/submodules/simple-knn/setup.py` no longer touches the header. simple-knn rebuilt (gfx90a) and `distCUDA2` on 1000 random points returns finite values.
- Finding 3 (633f0b7 body gives the wrong reason the zero tail gets no tiles): not amendable. **For the PR-prep squash message:** the tail gets no tiles because `get_allocate_size_kernel` returns early for `index >= valid_length[0]` on an `allocated_size` that starts as `torch::zeros`, not because of its zero depth.
- Finding 4 (lesson lacks provenance): added ROCm 10.1 / rocPRIM 4.6.0 / hipCUB 4.7.0 / gfx90a and "other arches not checked" to the lesson.
- Reviewer's pipeline caution, kept for the Windows records: a8d1e6f removes the fused_ssim Windows changes that the windows-gfx1201 and windows-gfx1101 passes depended on, so neither may be carried forward across it.

## Re-review 2026-11-19 (reviewer, linux-gfx90a) -- PASS

Reviewed the delta dcf5784...38aa67c (867da59, 38aa67c) and re-judged the whole port, `git diff master...moat-port` at 38aa67c. All four findings from the review above are resolved:
- `git diff master...HEAD -- litegs/submodules/simple-knn/setup.py` no longer touches the licence header.
- README.MD:89 now reads "On Linux with AMD GPUs".
- The 633f0b7 wording fix is recorded for the squash message the PR needs.
- The lesson now carries its provenance line.

Nothing else in the port changed. The submodule still matches master exactly, and `jargon.py --port LiteGS` hits only the three old commits already listed for the PR squash. No new problems. Still owed outside review: the gfx90a validation at this head, and an nvcc compile of the CUDA path before the port is presented.

## Validation 2026-11-19 (validator, linux-gfx90a) -- GPU PASS at 38aa67c; recorded validation-failed --gate jargon

Device: AMD Instinct MI250X GCD (gfx90a:sramecc+:xnack-), `HIP_VISIBLE_DEVICES=2`. PyTorch `2.16.0a0+git6c3af02` (source build), `torch.version.hip` `7.16.26385`, ROCm 10.1.0 pip SDK (`rocm-sdk-*` wheels; rocPRIM 4.6.0, hipCUB 4.7.0). Python 3.11.17. head_sha `38aa67ca155b8d2db59a3b2895d4348045377203`, fused_ssim submodule at 409da0b (kemchenj, upstream's pin).

Build: clean (`git clean -fdx` in the fork and in the submodule, every `*.hip` mirror and `build/` removed, the three packages uninstalled first). Then from `/tmp` with `PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=32`, `python -m pip install --no-build-isolation --no-deps <src>/litegs/submodules/{simple-knn,fused_ssim,gaussian_raster}`. All three built. Each installed `.so` carries only `amdgcn-amd-amdhsa--gfx90a` code objects, and the hipified `transform.hip` contains the 633f0b7 change.

Self-check (the project's own `litegs.utils.wrapper.check()`, which compares each fused op with its PyTorch reference on forward and backward). The wrapper calls `validate()` per class, because `check()` stops at the first exception: 2 PASS / 2 mismatch / 2 harness errors of 6, identical to the 2026-06-01 gfx90a baseline. CreateTransformMatrix PASS, CreateCov2dDirectly PASS. SphericalHarmonicToRGB backward tensor #3 (dir_grad, deliberately never written by the kernel). EighAndInverse2x2Matrix forward tensor #1 (eigenvector sign). CreateRaySpaceTransformMatrix and Binning raise TypeError (the `test_inputs` lists do not match the fused signatures). All four are upstream harness artifacts, the same on CUDA, as recorded earlier. No new failures.

Real trainer (`example_train.py` unchanged, run through `train_probe.py`, which logs per-step loss and training-view PSNR; scene `tandt_db.zip` as in the porter section):
```
python train_probe.py <src> -s tandt/truck -i images -m <out> --iterations 3000          # x3
python train_probe.py <src> -s tandt/truck -i images -m <out_eval> --eval --iterations 3000
python metrics_nolpips.py <src> -s tandt/truck -i images -m <out_eval> --eval              # example_metrics.py, LPIPS stubbed (no torchvision)
python train_probe.py <src> -s tandt/train -i images -m <out_train> --iterations 7000
```
| run | iterations | first iter loss / PSNR | last iter loss / PSNR | first-50 -> last-50 mean loss / PSNR | finite |
|---|---|---|---|---|---|
| truck 1 | 2761 | 0.33724 / 9.58 dB | 0.08266 / 22.14 dB | 0.21858 / 14.13 -> 0.07860 / 22.62 | yes |
| truck 2 | 2761 | 0.30036 / 11.34 dB | 0.08163 / 22.33 dB | 0.22443 / 14.09 -> 0.07663 / 22.87 | yes |
| truck 3 | 2761 | 0.24175 / 13.24 dB | 0.07747 / 22.98 dB | 0.23031 / 13.78 -> 0.07601 / 22.94 | yes |
| truck --eval | 2847 | 0.26546 / 12.48 dB | 0.08630 / 22.45 dB | 0.22824 / 13.81 -> 0.09202 / 21.38 | yes |
| train (301 img) | 6923 | 0.32278 / 10.93 dB | 0.09414 / 21.15 dB | 0.28191 / 11.98 -> 0.09248 / 22.09 | yes |

`example_metrics.py --eval` on the truck --eval model: Trainingset SSIM 0.7547 / PSNR 22.14 dB; Testset (31 held-out views) SSIM 0.7501 / PSNR 21.77 dB. Training times: truck about 29.8 s per 3000-iteration run, train 124 s for 7000 iterations. No 0x1016, no hipErrorLaunchFailure, in 5 runs and about 18,900 iterations in total (before 633f0b7, every plain run aborted within 6 epochs).

Released ROCm PyTorch wheel: not tested here; this host has only a source-built torch. Nothing in the build needs one, though: the extensions use the stock `torch.utils.cpp_extension.CUDAExtension` and `BuildExtension`, no private ATen API, and only the ROCm headers and hipcc that a released ROCm wheel plus a ROCm install (or the pip ROCm SDK) provide. So a released wheel should suffice. That is an inference, not a measurement. Separately, whether a released ROCm's rocPRIM has the NaN sort defect is unknown; 633f0b7 removes LiteGS's exposure to it either way.

CUDA no-regression compile: `cuda-not-validated: no nvcc on this host (headless fleet worker, cannot dispatch to the nvcc host)`. The port's CUDA-side delta since the last review is `torch::empty` -> `torch::zeros` in host code plus README/comment text. An nvcc compile is still owed before the port is presented.

Gates before completion: documentation is present (README "Getting Started", Linux scope). **Jargon is not clean**: `python3 utils/jargon.py --port LiteGS` exits 1 with 12 hits, all in commit messages of the three earlier commits (602ce5a: 3, d7990e3: 7, d65b6ab: 2). The six newer commits are clean. jeff's instruction for this round was not to rewrite those commits, so the porter cannot fix this in a new commit. The only fix is a person-made PR-prep squash to an identical tree with clean messages (fold in the corrected 633f0b7 reason from "Porter round 2026-11-19 (2)"). So this platform is recorded `validation-failed --gate jargon`, which keeps the GPU pass (`gpu_passed_sha` = 38aa67c). After the squash, a validator checks that jargon is clean and completes the platform with `carry-forward` (`squash-carry-forward` does not apply, because this platform is not `completed`).

Integrity: `git status --porcelain` is empty in the fork and in the submodule at 38aa67c.

Other platforms: linux-gfx1100 reads `revalidate` at this head (its d65b6ab record carried the broken submodule). windows-gfx1101 and windows-gfx1201 must not be carried forward across a8d1e6f, and the Windows fused_ssim build is the open question above.

## Validation 2026-11-19 (validator, linux-gfx1201) -- GPU PASS at 38aa67c; recorded validation-failed --gate jargon

Device: AMD Radeon AI PRO R9700, gfx1201 (RDNA4, wave32; `torch.cuda.get_device_properties(0).warp_size` = 32), one GPU pinned by `HIP_VISIBLE_DEVICES`. PyTorch `2.16.0a0+git6c3af02` (source build, the same torch as the gfx90a run), `torch.version.hip` `7.16.26385`, ROCm 10.1.0 pip SDK (`rocm-sdk-*` 10.1.0 wheels, `ROCM_HOME=.../site-packages/_rocm_sdk_devel`, no `/opt/rocm`; rocPRIM 4.6.0 (`ROCPRIM_VERSION 400600`), hipCUB 4.7.0 (`HIPCUB_VERSION 400700`)). Python 3.11.17. head_sha `38aa67ca155b8d2db59a3b2895d4348045377203`, fused_ssim submodule at 409da0b (kemchenj, upstream's pin).

Environment: the job venv sees the image site-packages (torch) through a `.pth` file. Deps were installed into it: `plyfile<1.1 opencv-python-headless<4.11 matplotlib torchmetrics numpy==1.26.4`. torchvision is absent, so LPIPS is stubbed to nan as before.

Build: a fresh clone of the fork at 38aa67c, `git submodule update --init --recursive`. Then from `/tmp` with `PYTORCH_ROCM_ARCH=gfx1201 MAX_JOBS=6`:
```
for d in simple-knn fused_ssim gaussian_raster; do
  utils/timeit.sh LiteGS compile -- python -m pip install --no-build-isolation --no-deps <src>/litegs/submodules/$d
done
```
All three built with no source change. Each installed `.so` carries only `amdgcn-amd-amdhsa--gfx1201` code objects (litegs_fused 4, simple_knn 1, fused_ssim_cuda 1). The hipified `transform.hip:453` has `view_position = torch::zeros(...)` (633f0b7).

Self-check (`litegs.utils.wrapper`, `validate()` per subclass, since `check()` stops at the first exception): CreateTransformMatrix PASS, CreateCov2dDirectly PASS. SphericalHarmonicToRGB prints backward tensor #3 mismatch (dir_grad, never written by the kernel). CreateRaySpaceTransformMatrix and Binning raise TypeError (the `test_inputs` do not match the fused signatures). Those four are the known upstream harness artifacts, identical to gfx90a. **EighAndInverse2x2Matrix differs from gfx90a, but not because of the port.** Its default input (a batch of 524,288 2x2 matrices) makes the pure-PyTorch *reference* `torch.linalg.eigh` raise `hipErrorInvalidConfiguration` ("invalid configuration argument") on gfx1201. The fused LiteGS kernel ran fine on that same input. Plain `torch.linalg.eigh(torch.randn(n,2,2))` reproduces it with no LiteGS code: n=1024 and n=65536 OK, n=131072 and n=524288 fail. This is a torch/rocSOLVER batched-eigh limit on this arch and ROCm build, outside the port and not registered as a deferral. With the same generator at n=65536, `validate()` reports forward tensor #1 (eigenvectors) mismatch, as on gfx90a. Checked directly: eigenvalue max rel err 3.7e-3 (threshold 1e-2), inverse 1.7e-4, eigenvectors differ only in sign (sign-aligned max abs diff 7.7e-6). Net: no new failure caused by the port.

Real trainer (`example_train.py` unchanged, run through `train_probe.py`, which wraps `fused_ssim.fused_l1_ssim_loss` to log per-step loss and training-view PSNR; scene `tandt_db.zip`, 682,628,995 bytes, sha256 `816e62f2...26d7e1` verified, same mirror as the porter section):
```
utils/timeit.sh LiteGS test -- python train_probe.py <src> -s tandt/truck -i images -m <out> --iterations 3000          # x3
utils/timeit.sh LiteGS test -- python train_probe.py <src> -s tandt/truck -i images -m <out_eval> --eval --iterations 3000
utils/timeit.sh LiteGS test -- python metrics_nolpips.py <src> -s tandt/truck -i images -m <out_eval> --eval              # example_metrics.py, LPIPS stubbed
utils/timeit.sh LiteGS test -- python train_probe.py <src> -s tandt/train -i images -m <out_train> --iterations 7000
```
| run | iterations | first iter loss / PSNR | last iter loss / PSNR | first-50 -> last-50 mean loss / PSNR | finite | wall |
|---|---|---|---|---|---|---|
| truck 1 | 2761 | 0.31424 / 10.72 dB | 0.07062 / 23.26 dB | 0.22532 / 13.88 -> 0.07795 / 22.77 | yes | 25.7 s |
| truck 2 | 2761 | 0.28549 / 11.61 dB | 0.07231 / 23.52 dB | 0.22559 / 13.89 -> 0.07736 / 22.92 | yes | 25.7 s |
| truck 3 | 2761 | 0.30003 / 11.36 dB | 0.08177 / 22.31 dB | 0.21949 / 14.06 -> 0.07918 / 22.57 | yes | 25.7 s |
| truck --eval | 2847 | 0.27394 / 11.94 dB | 0.08217 / 22.45 dB | 0.22444 / 14.00 -> 0.09233 / 21.30 | yes | 30.3 s |
| train (301 img) | 6923 | 0.32219 / 10.64 dB | 0.13705 / 19.13 dB | 0.27136 / 12.48 -> 0.09668 / 21.71 | yes | 75.3 s |

(train's last single iteration is one noisy view; its last-50 mean, 21.71 dB, is in line with gfx90a's 22.09.) `example_metrics.py --eval` on the truck --eval model: Trainingset SSIM 0.7574 / PSNR 22.24 dB; **Testset (31 held-out views) SSIM 0.7514 / PSNR 21.86 dB** (gfx90a: 0.7501 / 21.77). Wall times include data load. No 0x1016, no `hipError`, no `HSA_STATUS` in any log: 5 runs, about 18,900 iterations, every one completed.

The rocPRIM mixed-NaN sort defect also occurs on this arch. The reproducer from the porter section, built with `hipcc --offload-arch=gfx1201 -O2` from the same SDK: 2000 qNaN + 50 sNaN -> 29433 duplicated / 34590 missing, CORRUPT. 200 + 6 -> 163 / 163, CORRUPT. 2000 qNaN alone and 50 sNaN alone are ok. So it is not gfx90a- or wave64-specific (the lesson's "other arches not checked" can now name gfx1201). The trainer completing here confirms that 633f0b7 keeps LiteGS clear of it on wave32 too.

CUDA no-regression compile: `cuda-not-validated: no nvcc on this host` (no `/opt/conda`; headless worker). Still owed before the port is presented, as recorded for gfx90a.

Gates: documentation present (README.MD:89, Linux scope). **Jargon is not clean**: `python3 utils/jargon.py --port LiteGS` exits 1 with the same 12 hits as on gfx90a, all in commit messages of the three earlier commits (602ce5a: 3, d7990e3: 7, d65b6ab: 2). Fixing them takes the person-made PR-prep squash described above, so this platform is recorded `validation-failed --gate jargon` with the GPU pass kept (`gpu_passed_sha` = 38aa67c). After the squash, a validator checks jargon and completes with `carry-forward`. (To run jargon in a fresh clone, create local `master` and `moat-port` branches first; it resolves `master..moat-port` by local name.)

Integrity: `git status --porcelain` is empty in the fork and in the submodule; the clone is on `moat-port` at 38aa67c. No source edit, no fork push.

## CUDA compile check 2026-11-19 (nvcc, compile and link only) -- PASS at 38aa67c, no port regression

Host: Linux x86_64, no NVIDIA GPU (compile, link and import only; nothing ran on a device). CUDA toolkit 12.8 (`nvcc` release 12.8, V12.8.93, `/usr/local/cuda`), gcc 13.3.0, Python 3.11.17, ninja via `torch.utils.cpp_extension.BuildExtension`. PyTorch `2.11.0+cu128` from the official wheel index (`https://download.pytorch.org/whl/cu128`, matching the 12.8 toolkit), installed into the job's own venv.

Trees, each cloned fresh with `git submodule update --init --recursive`:
- port: AMD-Ecosystem/LiteGS at head_sha `38aa67ca155b8d2db59a3b2895d4348045377203`.
- base: MooreThreads/LiteGS at the merge base `004b95215c90c36cdaf4b354301132b700ac287b` ("remove: delete obsolete/unused scripts"), which is upstream `master`'s tip today.
- fused_ssim is kemchenj/fused-ssim `409da0b` in both trees (same gitlink).

Recipe: upstream has no CI; its README builds each extension with `pip install litegs/submodules/<ext>`. The same command, run against the CUDA torch with verbose output, from `/tmp`, with the build dir removed first:
```
export PATH=/usr/local/cuda/bin:$VENV/bin:$PATH CUDA_HOME=/usr/local/cuda MAX_JOBS=64
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;10.0;12.0"
for d in simple-knn fused_ssim gaussian_raster; do
  python -m pip install -v --no-build-isolation --no-deps --force-reinstall <tree>/litegs/submodules/$d
done
python -c "import torch, simple_knn._C, fused_ssim, litegs_fused"
```
Two settings are needed only because this host has no NVIDIA GPU, and they apply identically to both trees:
- `TORCH_CUDA_ARCH_LIST` must be set: with no device, torch 2.11 derives an empty arch list and `_get_cuda_arch_flags` raises `IndexError` before compiling. The list must start at sm_80: with 7.5 included, `raster.cu` fails on both trees with `identifier "__reduce_add_sync" is undefined` / `"__reduce_max_sync"` (those intrinsics need sm_80+). That is upstream's minimum arch, not a port issue.
- fused_ssim's `setup.py` chooses its backend from `torch.cuda.is_available()`; with no device it falls through to `hasattr(torch, 'xpu')` and compiles with `icpx` (exit 127). For fused_ssim only, a scratch `sitecustomize.py` on `PYTHONPATH` (`import torch; torch.cuda.is_available = lambda: True`) sends it down its CUDA branch, where `current_device()` fails and it uses its own fallback gencodes (sm_75/80/89, `--maxrregcount=32 --use_fast_math`). No source was edited.

| extension | base 004b952 | port 38aa67c |
|---|---|---|
| simple-knn | built, ninja 3/3, 8 warnings | built, ninja 3/3, 8 warnings |
| fused_ssim | built, ninja 2/2, 0 warnings | built, ninja 2/2, 0 warnings |
| gaussian_raster (litegs_fused) | built, ninja 6/6, 40 warning lines | built, ninja 6/6, 20 warning lines |
| import of all three | OK | OK |

Configure (setup.py) succeeded for all six builds; there is no separate configure step. No errors on either tree. `litegs_fused` from the port tree carries 4 cubins for each of sm_80, sm_86, sm_90, sm_100 and sm_120 (`cuobjdump --list-elf`). `litegs_fused` exports 32 names in both trees; `fused_ssim` exposes `fused_ssim` and `fused_l1_ssim_loss` in both.

Warnings, compared after normalizing paths:
- simple-knn: identical. 6x `"__CUDACC__" redefined` (simple_knn.cu:24 on base, :27 on port, the same line moved by the port's added lines) and 2x deprecated `Tensor.data<T>()` in spatial.cu.
- fused_ssim: none on either tree.
- gaussian_raster: the only difference is a warning the port removes. Base has `warning #2361-D: invalid narrowing conversion from "double" to "float"` at transform.cu:38 (15 lines, 3 per arch) and at transform.cu:1379 (20 lines, 4 per arch). The port keeps the first one (now transform.cu:42, the same `float3 t{...}` line) and loses the second, because its `(float)` casts on the `input_matrix` initializer silence it. No new warning anywhere: the 35 respaced launches, the 64-bit ballot path in compact.cu, and `torch::zeros` in transform.cu compile clean under nvcc.

Verdict: the CUDA path compiles and links with real nvcc at head 38aa67c, with no new error or warning against upstream's base. This closes the "nvcc compile still owed" item recorded by the gfx90a and gfx1201 validations. Stage and platform records are unchanged by this check. `git status --porcelain` is empty in the port clone and its submodule.

## Ruling 2026-11-19 (jeffdaily): fused_ssim submodule moves to AMD-Ecosystem/fused-ssim moat-port

jeffdaily ruled on 2026-11-19: LiteGS's pin `kemchenj/fused-ssim` 409da0b (branch `fused-l1-ssim-loss`) is one commit by Kem Chen, "Add fused L1 SSIM loss and backward functions", which is rahul-goel/fused-ssim PR 33, open since December 2025 with the maintainer undecided. Its base is 14 commits behind rahul-goel `main`. The original repository has had no commit since February 2026 and no maintainer reply since March 2026, and five later PRs (one an AMD build fix) are unanswered. So our fork `AMD-Ecosystem/fused-ssim` carries a `moat-port` branch with everything LiteGS needs, with Kem Chen's change brought in as a cherry-pick that keeps his authorship, and LiteGS's submodule points at that branch. Nothing is sent to the fused-ssim maintainers or to Kem Chen. This settles the "Windows: open question for jeff" above.

## fused-ssim fork commits and tests (AMD-Ecosystem/fused-ssim, no project record of its own)

fused-ssim has no MOAT project folder (recorded as already supported upstream), so its round is recorded here.

`moat-port` before: 6f9a620 = rahul-goel `main` (a7c48d6) + 666987f (validation, README), 767e3f6 (Windows `_winhip.cu` shim), cdfe1fa (Windows `FUSED_SSIM_HIP` guard), 6f9a620 (comment wording). Pushed 6f9a620..**d3213b4** (fast-forward, one commit):

- **d3213b4** "Add fused L1 SSIM loss and backward functions": `git cherry-pick -x 409da0b`, author Kem Chen <chenkem@gmail.com> (2025-11-11), committer Jeff Daily. Two textual conflicts, resolved and described in the commit message:
  - `ext.cpp`: `fusedl1ssim_loss` / `fusedl1ssim_loss_backward` are registered inside the existing `#if defined(FUSED_SSIM_CUDA) || defined(FUSED_SSIM_HIP)` block next to `fusedssim3d`, not unconditionally as in 409da0b, because the Metal (`ssim.mm`) and SYCL (`ssim_sycl.cpp`) backends compile the same `ext.cpp` and define neither function (the same reason upstream 0ac916a guarded the 3D ones). The block comment now reads "3D SSIM and fused L1 + SSIM loss (CUDA/HIP only)".
  - `fused_ssim/__init__.py`: the CUDA branch imports the 2D, 3D and fused L1 functions and keeps `is_3D_supported = True`; `FusedL1SSIMLossMap` and `fused_l1_ssim_loss` are appended after `fused_ssim` unchanged. The MPS and XPU branches import exactly what they did before.
  - `ssim.cu`, `ssim.h`, `tests/test_fused_l1_ssim_loss.py` applied cleanly and are byte-identical to 409da0b's additions.
- No HIP commit was needed: the new kernels hipify and build zero-edit. Their only literal 32 (`warp_id = tid / 32`, `col += 32` in `fusedl1ssim_loss_backwardCUDA`) is the same plain tiling loop as the existing `fusedssim_backwardCUDA`, with no warp-synchronous operation, so it is correct on wave64 and wave32.
- No Windows commit was needed (reasoned, not built here): on Windows ROCm the setup.py shim (767e3f6) compiles `ext.cpp` as a copy `ext_winhip.cu` through hipcc, where `-DFUSED_SSIM_CUDA` arrives as `-DFUSED_SSIM_HIP`; the new registrations sit in the same guarded block as the 3D ones that cdfe1fa made reachable on Windows, and the kernels are in `ssim.cu`, which Windows already compiles.
- Commit message checked: `python3 utils/jargon.py -C <clone> --commits 6f9a620..d3213b4` and `--diff` both clean; read by hand against the upstream-text rules.

Build (gfx90a, this host): image torch `2.16.0a0+git6c3af02`, `torch.version.hip` 7.16.26385, ROCm 10.1 pip SDK, Python 3.11, from `/tmp` with `PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=32`: `python -m pip install -v --no-build-isolation --no-deps <fused-ssim clone>` -> built; only torch-header `-Wnan-infinity-disabled` warnings from `-ffast-math`.

Tests, the whole `tests/` directory, run from a scratch copy of `tests/` + `images/` (several scripts write into `../images`), `HIP_VISIBLE_DEVICES=0` (MI250X GCD):
| script | result |
|---|---|
| `test.py` (2D vs reference and pytorch_msssim, 100 iterations of asserts) | PASS |
| `test_3D.py` (3D, 10 iterations of asserts) | PASS |
| `test_fused_l1_ssim_loss.py` as a script (100 iterations of `test_fused_l1_ssim_loss` + benchmark) | PASS; fused 1.16 ms vs l1 + fused_ssim 1.45 ms fwd+bwd at 3x1080x1920 |
| `pytest test_fused_l1_ssim_loss.py` | 1 passed |
| `train_image.py`, `train_image_3D.py` | PASS (final SSIM 0.9999 / 0.99990) |
| `genplot.py` | PASS |
| `genplot_3D.py` | PASS after one scratch-copy fix of an upstream harness bug: line 2 `from fused_ssim3d import fused_ssim3d` names a module that has never existed; changed to `from fused_ssim import fused_ssim3d` in the copy only |
`train_image*.py` and `genplot.py` first failed only at the final image save: the device name "AMD Instinct MI250X / MI250" contains a slash, so the save path names a missing directory. Creating that directory in the scratch copy let them finish. Upstream harness artifacts, not port faults; nothing changed in the fork for them.

Margin probe, fused loss vs `l1 + fused_ssim` (`torch.isclose` defaults, as the test uses), same padding and valid, plus two-run bitwise determinism:
- random 3x1080x1920, random B2 C3 545x979, near-copy (gt + 1e-3 noise), piecewise-flat (rounded) images: loss |d| <= 1.2e-7, gradient max |d| <= 5e-12, deterministic, finite.
- **near-constant images (0.7 + 1e-4 noise, 256x256): disagree beyond the test's tolerance.** Loss 9.426e-5 (fused) vs 9.964e-5 (l1 + fused_ssim); gradient max |d| 5.2e-7 of 5.3e-6.

Finding (investigated, not loosened, kernels not changed): the cause is upstream 0f89827 ("Fix numerical stability in SSIM calculation", in fused-ssim `main` after Kem Chen's base). It clamps `sigma^2 >= 0` and `SSIM` to `[-1, 1]` in `fusedssimCUDA`. Kem's `fusedl1ssim_lossCUDA` duplicates the older SSIM arithmetic without the clamps. In near-constant windows float32 `E[x^2] - mu^2` cancels catastrophically (76,082 of 196,608 float32 pixels have SSIM > 1 before the clamp, max 1.0005). A pure-PyTorch reference gives float64 exact 9.4389e-5 (the same with or without clamps), float32 unclamped 9.379e-5, and float32 clamped 1.129e-4. So the fused kernel (9.426e-5) is the closer of the two to the exact value, the clamp biases fused_ssim there, and the split is pure float32 arithmetic, the same on CUDA. It is not a HIP fault. Random and natural images do not reach it (the truck runs below match the earlier pin). **For jeff:** whether the fused loss should adopt 0f89827's clamps (a commit of ours on fused-ssim `moat-port`, which would make it agree with `fused_ssim` but move it away from the exact value in such windows) is a ruling, not done here.

## Porter round 2026-11-19 (3) (linux-gfx90a): submodule repointed to AMD-Ecosystem/fused-ssim

Fork commit on `moat-port`, pushed e24abb2..**1664e6d** (one new commit, nothing rewritten): "[ROCm] Move the fused_ssim submodule to a branch on current fused-ssim". `.gitmodules` url `https://github.com/kemchenj/fused-ssim.git` -> `https://github.com/AMD-Ecosystem/fused-ssim`, `branch = fused-l1-ssim-loss` -> `branch = moat-port`; gitlink 409da0b -> d3213b4. Nothing else (`git show --stat`: 2 files, 3+/3-).

jargon: `python3 utils/jargon.py --port LiteGS` exits 1 with 2 hits, both the `.gitmodules` line `branch = moat-port` (the term "moat-port"/"moat"). That line names the actual branch, as jeffdaily's instruction for this round required ("keep whatever `branch =` convention the file uses, naming `moat-port`"). The commit message itself is clean. Clearing it would need either dropping the `branch =` line or a fused-ssim branch under a plain name (for example `rocm` or `fused-l1-ssim-loss`) and a new `.gitmodules` commit; that choice is jeff's. All older-commit hits are gone (the e24abb2 squash).

Environment this round: the job venv (`$FLEET_VENV`) is isolated from the image torch, so a `.pth` in it adds the image site-packages (read-only); deps installed into the job venv: `plyfile<1.1 opencv-python-headless<4.11 torchmetrics numpy==1.26.4 tqdm pytorch_msssim tifffile matplotlib`. Scene `tandt_db.zip` sha256 `816e62f2...26d7e1` verified, `tandt/truck` (251 images).

Build (from `/tmp`, fresh clone of the fork at the new commit, `git submodule sync && git submodule update --init --recursive` fetched d3213b4 from the new URL), `PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=32`, `python -m pip install -v --no-build-isolation --no-deps <src>/litegs/submodules/{simple-knn,fused_ssim,gaussian_raster}`: all three built (only the known `nodiscard hipError_t` warnings). `import torch, simple_knn._C, fused_ssim, litegs_fused` OK; `fused_ssim` exposes `fused_ssim`, `fused_ssim3d`, `fused_l1_ssim_loss`.

Real trainer (`example_train.py` unchanged, through `train_probe.py`; same commands as the 38aa67c validation):
| run | iterations | first iter loss / PSNR | last iter loss / PSNR | first-50 -> last-50 mean loss / PSNR | finite | wall |
|---|---|---|---|---|---|---|
| truck 1 | 2761 | 0.28567 / 11.18 dB | 0.08003 / 22.19 dB | 0.22978 / 13.65 -> 0.07711 / 22.84 | yes | 36.8 s |
| truck 2 | 2761 | 0.26825 / 12.44 dB | 0.07387 / 23.50 dB | 0.22118 / 14.20 -> 0.07709 / 22.78 | yes | 36.0 s |
| truck 3 | 2761 | 0.27948 / 11.87 dB | 0.07561 / 23.45 dB | 0.22401 / 13.93 -> 0.07609 / 22.91 | yes | 37.1 s |
| truck --eval | 2847 | 0.25989 / 11.94 dB | 0.09337 / 19.20 dB | 0.22319 / 14.04 -> 0.09933 / 20.80 | yes | 47.3 s |

`example_metrics.py --eval` (LPIPS stubbed): Trainingset SSIM 0.7575 / PSNR 22.21 dB; **Testset SSIM 0.7520 / PSNR 21.83 dB**.

Compared with 38aa67c (gfx90a, kemchenj 409da0b): plain runs last-50 loss 0.0761-0.0771 vs 0.0760-0.0786, last-50 PSNR 22.78-22.91 vs 22.62-22.94 dB; testset 0.7520 / 21.83 vs 0.7501 / 21.77 (trainset 0.7575 / 22.21 vs 0.7547 / 22.14). All within run-to-run spread. The --eval run's last-50 training-view window (0.0993 / 20.80) is below 38aa67c's single --eval run (0.0920 / 21.38). That window covers 50 random training views of one run, and the held-out metrics, which measure the model, are slightly better, so I read it as spread. No 0x1016, `hipError` or `HSA_STATUS` in any run log.

Integrity: `git status --porcelain` empty in the fork clone and the submodule at 1664e6d.

## For the Windows validation (next, on a Windows host)

- Fork head 1664e6d; submodule `litegs/submodules/fused_ssim` = AMD-Ecosystem/fused-ssim **d3213b4** (`moat-port`). Run `git submodule sync && git submodule update --init --recursive` in an existing clone, since the URL changed.
- fused_ssim's Windows path: `setup.py` copies `ext.cpp` to `ext_winhip.cu` when `os.name == 'nt' and torch.version.hip` (767e3f6), so hipcc builds the pybind module and `-DFUSED_SSIM_CUDA` arrives as `-DFUSED_SSIM_HIP`. `ext.cpp` registers under `#if defined(FUSED_SSIM_CUDA) || defined(FUSED_SSIM_HIP)`: `fusedssim3d`, `fusedssim_backward3d`, `fusedl1ssim_loss`, `fusedl1ssim_loss_backward`; `fusedssim` / `fusedssim_backward` are unconditional. No new build flag: the HIP flags are unchanged (`-O3 -DFUSED_SSIM_CUDA -ffast-math`). Check `from fused_ssim_cuda import fusedl1ssim_loss, fusedl1ssim_loss_backward, fusedssim3d` after the build, because `fused_ssim/__init__.py` imports all six at module load and LiteGS's trainer calls `fused_l1_ssim_loss` every step.
- The Windows fused-ssim build with the cherry-pick has not been run anywhere; this round reasoned it from the code only. LiteGS's own two setup.py Windows shims (in e24abb2) are unchanged.
- README.MD:89 still scopes the AMD build note to Linux ("On Linux with AMD GPUs"). If Windows passes, widening it is a separate commit.
- Neither Windows record may be carried forward; both need a fresh run with the real trainer, as on Linux.

## Review 2026-11-19 (3) (reviewer, linux-gfx90a) -- PASS

Scope: `git diff e24abb2...1664e6d` in the LiteGS fork (`.gitmodules` url/branch and the `litegs/submodules/fused_ssim` gitlink 409da0b -> d3213b4), plus the submodule content delta: AMD-Ecosystem/fused-ssim d3213b4 against 6f9a620 (previous fused-ssim fork tip) and against kemchenj 409da0b (previous LiteGS pin). No blocking problems. Checked and holding: `ssim.cu`, `ssim.h` and `tests/test_fused_l1_ssim_loss.py` hunks in 6f9a620..d3213b4 are line-identical to 409da0b's; `FusedL1SSIMLossMap` / `fused_l1_ssim_loss` are byte-identical to 409da0b; the MPS/XPU import branches are untouched; the two new `m.def`s sit inside the `FUSED_SSIM_CUDA || FUSED_SSIM_HIP` block that the Windows `ext_winhip.cu` copy (setup.py:130) reaches; the literal 32 in `fusedl1ssim_loss_backwardCUDA` (ssim.cu:762-768) is a row/column tiling of a 16x16 block followed by `block.sync()`, with no warp-synchronous operation, so it is wave-width independent; the commit message's numbers match the porter round's table and the 14-commit / Metal-SYCL / stability claims match `git log` (0ac916a, 0f89827, ed33721). LiteGS calls only `fused_l1_ssim_loss` (trainer.py:145).

Notes for jeff (not blocking, recorded per instruction):
- `jargon.py --port LiteGS` reports 2 hits, both `.gitmodules:4` `branch = moat-port`. jeffdaily's instruction for this round required that line verbatim; not requested as a change. Before the upstream PR, this remains the one jargon hit and the PR gate will see it.
- The near-constant-image disagreement is reproduced on this host (MI250X, `$FLEET_VENV` build of d3213b4): `fused_l1_ssim_loss` 9.4263e-5 vs `l1 + fused_ssim` 9.9640e-5; float64 reference 9.4389e-5 with or without clamps; float32 unclamped 9.3787e-5, clamped 1.1291e-4; 76,082 of 196,608 float32 pixels exceed SSIM 1 before the clamp. `fusedssimCUDA` carries 0f89827's clamps (ssim.cu:247-272) and `fusedl1ssim_lossCUDA` does not (ssim.cu:680-715). Random, near-copy and piecewise-flat cases agree under `isclose`, deterministic. It does not block this delta: the kernel LiteGS calls is unchanged from the previous pin, so LiteGS's loss is the same as before; only `fused_ssim` moved (it gained the clamp upstream). Whether to add the clamps to the fused kernel is correctly left as jeff's ruling. The porter's "the same on CUDA" is reasoned from the code (the clamp is backend independent), not measured.
- fused-ssim d3213b4's body (Kem Chen's message plus an agent-written conflict-resolution paragraph) has no AI-assistance line and no Test Plan, unlike the four earlier fork commits (666987f, 767e3f6, cdfe1fa, 6f9a620). It is already pushed on `moat-port` and LiteGS pins it, so changing it means rewriting a published branch, which the no-force-push rule forbids; recorded for jeff rather than requested.
- Owed before presenting (not a review gate): an nvcc compile of 1664e6d with d3213b4 (the new kernels and the moved `m.def`s have not been compiled with nvcc in this round), and the Windows fused-ssim build with the cherry-pick, which notes say has only been reasoned.

## Validation 2026-11-19 (3) (validator, linux-gfx90a) -- GPU PASS at 1664e6d; recorded validation-failed --gate jargon

Device: AMD Instinct MI250X GCD (gfx90a:sramecc+:xnack-), `HIP_VISIBLE_DEVICES=0`. PyTorch `2.16.0a0+git6c3af02` (source build), `torch.version.hip` 7.16.26385, ROCm 10.1.0 pip SDK, Python 3.11. head_sha `1664e6da19e9fd89b437bcd6e38db1e654edb6b1`, fused_ssim submodule AMD-Ecosystem/fused-ssim d3213b4.

Build: clean (`git clean -fdx` in the fork and the submodule, the three packages uninstalled from the job venv). From `/tmp`, `PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=32`, each wrapped in `utils/timeit.sh LiteGS compile --`: `python -m pip install -v --no-build-isolation --no-deps <src>/litegs/submodules/{simple-knn,fused_ssim,gaussian_raster}`. All three built. Code objects: litegs_fused 4, simple_knn._C 1, fused_ssim_cuda 2 (ssim + ssim3d), all `amdgcn-amd-amdhsa--gfx90a`. `transform.hip:453` has `torch::zeros` for `view_position`.

Self-check (`selfcheck.py`, `validate()` per class): 2 PASS / 2 mismatch / 2 TypeError of 6, identical to the 38aa67c and 2026-06-01 baselines (known upstream harness artifacts).

fused_ssim's own tests at d3213b4: see "fused-ssim fork commits and tests" above (all pass on this GPU).

Real trainer (`example_train.py` unchanged, through `train_probe.py`; absolute scene paths, since `timeit.sh` changes cwd; each wrapped in `utils/timeit.sh LiteGS test --`):
| run | iterations | first iter loss / PSNR | last iter loss / PSNR | first-50 -> last-50 mean loss / PSNR | finite | wall |
|---|---|---|---|---|---|---|
| truck 1 | 2761 | 0.26825 / 12.44 dB | 0.07751 / 24.19 dB | 0.22832 / 13.78 -> 0.07866 / 22.74 | yes | 37.0 s |
| truck 2 | 2761 | 0.28639 / 11.39 dB | 0.09336 / 20.97 dB | 0.23220 / 13.61 -> 0.08030 / 22.53 | yes | 35.9 s |
| truck 3 | 2761 | 0.26131 / 12.65 dB | 0.07323 / 23.53 dB | 0.22774 / 13.75 -> 0.08049 / 22.58 | yes | 35.5 s |
| truck --eval | 2847 | 0.29844 / 11.25 dB | 0.10008 / 20.41 dB | 0.23669 / 13.50 -> 0.09452 / 21.09 | yes | 47.3 s |
| train (301 img) | 6923 | 0.36493 / 9.13 dB | 0.09926 / 19.19 dB | 0.28864 / 11.89 -> 0.09719 / 21.54 | yes | 134.4 s |

`example_metrics.py --eval` (LPIPS stubbed): Trainingset SSIM 0.7534 / PSNR 22.19 dB; **Testset SSIM 0.7468 / PSNR 21.79 dB**.

Against 38aa67c (409da0b pin): truck last-50 loss 0.0787-0.0805 here, 0.0761-0.0771 in the porter round at this head, 0.0760-0.0786 at 38aa67c gfx90a, 0.0774-0.0792 at 38aa67c gfx1201; testset 21.79 / 0.7468 vs 21.77 / 0.7501 (gfx90a) and 21.86 / 0.7514 (gfx1201); train last-50 0.0972 / 21.54 vs 0.0925 / 22.09 (gfx90a) and 0.0967 / 21.71 (gfx1201). Within run-to-run spread (random frame order; the fused loss kernel LiteGS calls is the same code as at 409da0b). No 0x1016, `hipError` or `HSA_STATUS` in any run log: 5 runs, about 18,900 iterations. (A first attempt of the runs failed at once with `FileNotFoundError` on a relative scene path because `timeit.sh` changes cwd; an environment error, not a port result.)

CUDA no-regression compile: `cuda-not-validated: no nvcc on this host (no /opt/conda, headless worker)`. The CUDA-relevant delta since the nvcc-checked 38aa67c is the fused_ssim submodule (409da0b -> d3213b4: rahul-goel main + 3D + Windows setup shim + the same fused-loss code); an nvcc compile of 1664e6d with its submodule is owed before the port is presented.

Gates: documentation present (README.MD:89, Linux scope). **Jargon is not clean**: `python3 utils/jargon.py --port LiteGS` exits 1 with 2 hits, both the `.gitmodules` line `branch = moat-port`, which jeffdaily's instruction for this round required verbatim. A validator may not waive it, so this platform is recorded `validation-failed --gate jargon` with the GPU pass kept (`gpu_passed_sha` = 1664e6d). **For jeff:** either rule the line acceptable for the PR, or give fused-ssim's port branch a plain name (or drop the `branch =` line) in a new `.gitmodules` commit; such a commit changes no compiled output, so a validator then completes this platform with `carry-forward ... source-class` from this pass.

Integrity: `git status --porcelain` empty in the fork clone and the submodule at 1664e6d.

## Porter round 2026-11-19 (4) (linux-gfx1201): `branch =` line dropped from `.gitmodules`

Instruction from jeffdaily: the 1664e6d `.gitmodules` line `branch = moat-port` is in-house vocabulary in a maintainer-visible diff, and a submodule is pinned by its commit, so remove the line and change nothing else.

Fork commit on `moat-port`, pushed 1664e6d..**a995109** (one new commit, nothing rewritten, plain push): "Drop the branch line from the fused_ssim submodule entry" (no `[ROCm]` prefix: the added lines name nothing ROCm-specific). `git show --stat`: `.gitmodules | 1 -`. Gitlink unchanged at AMD-Ecosystem/fused-ssim d3213b4. The entry against upstream `master` now differs in the `url` line (kemchenj `.git` URL -> `https://github.com/AMD-Ecosystem/fused-ssim`) and in the absence of upstream's `branch = fused-l1-ssim-loss` line; no other line.

Side effect, for the reviewer and jeff: `git submodule update --remote` (which upstream's README does not use) now follows the remote's default branch (fused-ssim's `main` mirror) instead of a named branch; `--init` and plain `update` check out the gitlink exactly as before.

Checks:
- Fresh clone (`git clone` of the fork, `git checkout moat-port`, `git submodule update --init --recursive`): `Submodule path 'litegs/submodules/fused_ssim': checked out 'd3213b4956811f0250f9a4968b9ee673a312e4b5'`, remote URL `https://github.com/AMD-Ecosystem/fused-ssim`. d3213b4 is reachable on the fork's `moat-port` branch, so the default fetch finds it without a `branch` hint.
- `python3 utils/jargon.py --port LiteGS`: **clean** (exit 0) over the whole branch `master..moat-port`, code and commit messages. No old commit message hit remains (the e24abb2 squash already removed them; the only hits at 1664e6d were the `.gitmodules` line).
- `python3 utils/moatlib.py audit-commits LiteGS`: OK.
- Build (gfx1201, `$FLEET_VENV` with a `.pth` to the image site-packages, torch `2.16.0a0+git6c3af02`, HIP 7.16.26385), from `/tmp`, `PYTORCH_ROCM_ARCH=gfx1201`, `utils/timeit.sh LiteGS compile -- python -m pip install -v --no-build-isolation --no-deps <src>/litegs/submodules/{simple-knn,fused_ssim,gaussian_raster}`: all three built; `fused_ssim` exposes `fused_ssim`, `fused_ssim3d`, `fused_l1_ssim_loss`.

## Review 2026-11-19 (4) (reviewer, linux-gfx1201) -- PASS

Scope: `git diff 1664e6d..a995109` in the LiteGS fork (reviewed_sha was 1664e6d). One deleted line, `.gitmodules` `branch = moat-port`; the gitlink is unchanged (d3213b4). No problems found. Checked: `moatlib.py classify LiteGS 1664e6d a995109` -> `doc-only`, inert; the README never documents `git submodule update --remote` or any clone recipe that would read the `branch` key (README.MD only runs `pip install litegs/submodules/<ext>`), and nothing else in the tree (no `.github`, no scripts) reads it, so dropping it changes no documented workflow; a fresh clone plus `git submodule update --init --recursive` checks out d3213b4 (the porter's check, and `git submodule status` here). The commit message's claims (`--init` checks out the recorded commit whatever `branch` says; only `.gitmodules` changes) are accurate; `utils/prose.py` on the body: clean; `jargon.py --port LiteGS`: clean over the whole branch; `audit-commits`: OK. The open items for jeff in Review (3) (fused-loss clamp ruling, the d3213b4 message without an AI line or Test Plan, nvcc compile of the current pin, the Windows fused-ssim build) are unchanged by this delta and still stand.

## Validation 2026-11-19 (4) (validator, linux-gfx1201) -- PASS at a995109 -> completed

Device: AMD Radeon AI PRO R9700, gfx1201 (RDNA4, `warp_size` 32), one GPU pinned by `HIP_VISIBLE_DEVICES`. PyTorch `2.16.0a0+git6c3af02` (image source build), `torch.version.hip` 7.16.26385, ROCm 10.1 pip SDK, Python 3.11.17. head_sha `a995109ec81235aeb9dab69bfc6209ceeae3e053`, fused_ssim submodule AMD-Ecosystem/fused-ssim d3213b4 (checked out by `git submodule update --init --recursive` from the new `.gitmodules`, no `branch` line). All GPU work under `ulimit -c 0`.

Environment: `$FLEET_VENV` is isolated from the image torch; a `.pth` in it adds the image site-packages (read-only). Deps installed into the job venv: `plyfile<1.1 opencv-python-headless<4.11 torchmetrics numpy==1.26.4 tqdm pytorch_msssim tifffile matplotlib pytest`. Scene: `tandt/truck` and `tandt/train` extracted from `tandt_db.zip` (sha256 `816e62f2...26d7e1` verified, the same file as the earlier gfx1201 round). Probe scripts (`train_probe.py`, `metrics_nolpips.py`, `selfcheck.py`) are the ones the 38aa67c gfx1201 validation used, unchanged.

Build: clean (`git clean -fdx` in the fork and the submodule, the three packages uninstalled first). From `/tmp`, `PYTORCH_ROCM_ARCH=gfx1201`, each under `utils/timeit.sh LiteGS compile --`, the README commands: `python -m pip install -v --no-build-isolation --no-deps <src>/litegs/submodules/{simple-knn,fused_ssim,gaussian_raster}`. All three built, no source change. Code objects (all `amdgcn-amd-amdhsa--gfx1201`): simple_knn._C 1, fused_ssim_cuda 2, litegs_fused 4. `transform.hip:453` has `torch::zeros` for `view_position`. `fused_ssim` exposes `fused_ssim`, `fused_ssim3d`, `fused_l1_ssim_loss`.

Self-check (`selfcheck.py`, `validate()` per class): CreateTransformMatrix PASS, CreateCov2dDirectly PASS, **SphericalHarmonicToRGB PASS** (it printed a dir_grad mismatch at 38aa67c; that tensor is never written by the kernel, so the comparison reads uninitialized memory and can go either way), CreateRaySpaceTransformMatrix and Binning TypeError (harness signature mismatch), EighAndInverse2x2Matrix `hipErrorInvalidConfiguration` from the torch `linalg.eigh` reference at the default 524,288 batch (the known torch/rocSOLVER limit on this arch recorded at 38aa67c). Same known harness artifacts; nothing new from the port.

Real trainer (`example_train.py` unchanged, through `train_probe.py`, absolute scene paths, each under `utils/timeit.sh LiteGS test --`; commands as in the 38aa67c gfx1201 section):
| run | iterations | first iter loss / PSNR | last iter loss / PSNR | first-50 -> last-50 mean loss / PSNR | finite | wall |
|---|---|---|---|---|---|---|
| truck 1 | 2761 | 0.28946 / 11.51 dB | 0.06917 / 23.03 dB | 0.22039 / 14.11 -> 0.08013 / 22.48 | yes | 26.2 s |
| truck 2 | 2761 | 0.25804 / 13.64 dB | 0.09393 / 21.11 dB | 0.23098 / 13.68 -> 0.07789 / 22.77 | yes | 26.0 s |
| truck 3 | 2761 | 0.28833 / 11.60 dB | 0.07598 / 22.92 dB | 0.22463 / 14.03 -> 0.07869 / 22.68 | yes | 25.6 s |
| truck --eval | 2847 | 0.26994 / 11.90 dB | 0.08325 / 21.82 dB | 0.23055 / 13.73 -> 0.09232 / 21.24 | yes | 30.4 s |
| train (301 img) | 6923 | 0.35985 / 9.76 dB | 0.07131 / 24.67 dB | 0.27748 / 12.09 -> 0.09387 / 22.01 | yes | 75.6 s |

`example_metrics.py --eval` (LPIPS stubbed): Trainingset SSIM 0.7614 / PSNR 22.35 dB; **Testset SSIM 0.7557 / PSNR 21.96 dB**.

Against the 38aa67c gfx1201 run (kemchenj 409da0b pin): truck last-50 loss 0.0779-0.0801 vs 0.0774-0.0792, last-50 PSNR 22.48-22.77 vs 22.57-22.92 dB; --eval last-50 0.0923 / 21.24 vs 0.0923 / 21.30; testset 0.7557 / 21.96 vs 0.7514 / 21.86 (trainset 0.7614 / 22.35 vs 0.7574 / 22.24); train last-50 0.0939 / 22.01 vs 0.0967 / 21.71. All within run-to-run spread (random frame order); the held-out and train figures are slightly better, truck plain slightly lower, overlapping ranges. Same speed (truck 25.6-26.2 s vs 25.7 s). No 0x1016, `hipError` or `HSA_STATUS` in any trainer log: 5 runs, about 18,900 iterations.

fused-ssim repository tests at d3213b4 on this GPU (scratch copy of `tests/` + `images/`, `genplot_3D.py` with the upstream import fix in the copy only, as on gfx90a; the device name has no slash, so no save-path workaround was needed), each under `utils/timeit.sh LiteGS test --`:
| script | result |
|---|---|
| `test.py` (2D vs reference and pytorch_msssim, asserts) | PASS |
| `test_3D.py` (3D, asserts) | PASS |
| `test_fused_l1_ssim_loss.py` as a script | PASS; fused 1.12 ms vs l1 + fused_ssim 1.55 ms fwd+bwd at 3x1080x1920 (27.9% faster) |
| `pytest test_fused_l1_ssim_loss.py` | 1 passed |
| `train_image.py` | PASS (final SSIM 0.9999) |
| `train_image_3D.py` | PASS (final SSIM 0.99990) |
| `genplot.py`, `genplot_3D.py` | PASS |
7 of 7 scripts pass, pytest 1 of 1.

CUDA no-regression compile: `cuda-not-validated: no nvcc on this host` (no `/usr/local/cuda`, no `nvcc` on PATH). The owed nvcc compile of the current pin (d3213b4) recorded in Review (3) still stands; this round's delta is `.gitmodules` only.

Gates: documentation present (README.MD:89, Linux scope); `python3 utils/jargon.py --port LiteGS`: **clean**. Integrity: `git status --porcelain` empty in the fork clone and the submodule, clone at a995109.

linux-gfx90a: its kept GPU pass is at 1664e6d (`--gate jargon`), and a995109 is `doc-only` over it, so a gfx90a validator can close it with `carry-forward LiteGS linux-gfx90a a995109... source-class` after checking jargon; not done from this host, since each host writes its own platform's record.

## Validation 2026-11-19 (5) (validator, windows-gfx1151) -- PASS at a995109 -> completed

First Windows build of fused-ssim d3213b4 (the fused L1+SSIM cherry-pick) anywhere, and the first LiteGS validation on this platform. Device: AMD Radeon 8060S (gfx1151, RDNA3.5, wave32, APU), Windows 11, `HIP_VISIBLE_DEVICES=0`. Python 3.13.13 venv with torch `2.12.0+rocm7.14.0a20260519` (`torch.version.hip` 7.13.26190) and `rocm-sdk-core/devel/libraries/device-gfx1151 7.14.0a20260519` wheels; hipcc `HIP version: 7.13.26190`, AMD clang 23.0.0git; MSVC 14.44.35207 `link.exe`. head_sha `a995109ec81235aeb9dab69bfc6209ceeae3e053`; clone reset to `origin/moat-port`, `git submodule sync && git submodule update --init --recursive` checked out fused_ssim d3213b4 from `https://github.com/AMD-Ecosystem/fused-ssim`. No source change in either repository.

Build environment (the same variables as the 2026-06-05 Windows validation, plus clang-cl as the host compiler, which torch-extension builds on this host use): `ROCM_PATH`/`HIP_PATH` = `<venv>/Lib/site-packages/_rocm_sdk_devel`, `HIP_DEVICE_LIB_PATH=<ROCM_PATH>/lib/llvm/amdgcn/bitcode`, `DISTUTILS_USE_SDK=1`, `PYTORCH_ROCM_ARCH=gfx1151`, `CC=CXX=<ROCM_PATH>/lib/llvm/bin/clang-cl.exe`, the MSVC `HostX64/x64` directory first on `PATH`, `MAX_JOBS=4`. The three README commands, one at a time, each under `utils/timeit.sh LiteGS compile --`, with `--target <scratch site dir>` added so the shared venv is not modified (the scratch dir is then on `PYTHONPATH`):
```
python -m pip install -v --no-build-isolation --no-deps --target <site> <src>/litegs/submodules/simple-knn
python -m pip install -v --no-build-isolation --no-deps --target <site> <src>/litegs/submodules/fused_ssim
python -m pip install -v --no-build-isolation --no-deps --target <site> <src>/litegs/submodules/gaussian_raster
```
All three built on the first attempt. fused_ssim: three hipcc translation units (`ssim.hip`, `ssim3d.hip`, `ext_winhip.cu`); the `ext_winhip.cu` command line carries `-DFUSED_SSIM_HIP -O3 -ffast-math --offload-arch=gfx1151`, as the porter round reasoned. No DLL staging was needed: every module is a `.pyd` loaded after `import torch`, which loads the HIP runtime from the wheel first.

Imports (`import torch` first): `simple_knn._C` exports `distCUDA2`. `fused_ssim_cuda` exports exactly `fusedssim`, `fusedssim_backward`, `fusedssim3d`, `fusedssim_backward3d`, `fusedl1ssim_loss`, `fusedl1ssim_loss_backward` (2D, 3D and fused L1+SSIM all registered). The `fused_ssim` package exposes `fused_ssim`, `fused_ssim3d`, `fused_l1_ssim_loss`, `FusedSSIMMap`, `FusedL1SSIMLossMap`, `is_3D_supported = True`. `litegs_fused` exports 26 functions (`rasterize_forward`, `rasterize_backward`, `mvp_transform_forward`, `cull_compact_activate`, `adamUpdate`, ...).

fused-ssim repository tests at d3213b4, run from a scratch copy of `tests/` + `images/`, each under `utils/timeit.sh LiteGS test --`. This APU laptop shares system RAM and must not run large or long loops, so the copies have SMALLER sizes and fewer repeats than upstream; the assertions and tolerances are unchanged:
| script | change in the scratch copy | result |
|---|---|---|
| `test.py` (2D vs reference and pytorch_msssim, value and gradient asserts) | `B,CH,H,W` 5,5,1080,1920 -> 2,3,256,256; 100 -> 10 iterations | PASS |
| `test_3D.py` (3D, value and gradient asserts, 10 iterations) | `B,CH,D,H,W` 2,1,96,96,96 -> 1,1,48,48,48 | PASS |
| `test_fused_l1_ssim_loss.py` as a script | image 3x1080x1920 -> 3x270x480; 100 -> 10 repeats of `test_fused_l1_ssim_loss`, benchmark 100 -> 10 | PASS (at this size fused 0.40 ms vs l1 + fused_ssim 0.39 ms; too small to show the speedup) |
| `pytest test_fused_l1_ssim_loss.py` | same small image | 1 passed |
| `train_image.py` | `albert.jpg` downscaled 3250x4333 -> 406x541 | PASS (reaches SSIM 0.9999, 1518 steps) |
| `train_image_3D.py` | none (98x120x96 volume) | PASS (reaches SSIM 0.9999, 1209 steps) |

Not run here: `genplot.py`, `genplot_3D.py` (multi-size benchmark plots), and the upstream full-size versions of the tests above.

Self-check (`litegs.utils.wrapper`, `validate()` per class): CreateTransformMatrix PASS, CreateCov2dDirectly PASS, SphericalHarmonicToRGB backward tensor #3 mismatch (dir_grad, never written by the kernel), EighAndInverse2x2Matrix forward tensor #1 mismatch (eigenvector sign), CreateRaySpaceTransformMatrix and Binning TypeError (harness signature mismatch). 2 PASS / 2 mismatch / 2 TypeError, identical to the gfx90a baseline at 38aa67c and 1664e6d: known upstream harness artifacts, nothing new.

Real trainer (`example_train.py` unchanged, run through a `train_probe.py` that wraps `fused_ssim.fused_l1_ssim_loss` to log per-step loss and training-view PSNR). No public scene is on this machine and the 683 MB `tandt_db.zip` was not downloaded (laptop limits), so the scene is a small synthetic COLMAP text model made by a scratch script: a textured unit sphere over a textured square, 24 ring cameras, 320x240 PINHOLE images, 8192 noisy initial points. It goes through the real loader, clustering, densification, the sparse-gradient optimizer, the rasterizer and the fused loss:
```
utils/timeit.sh LiteGS test -- python train_probe.py <src> <log> -s <scene> -i images -m <out> --iterations 1000
```
| run | iterations | first iter loss / PSNR | last iter loss / PSNR | first-50 -> last-50 mean loss / PSNR | finite | peak GPU memory | wall |
|---|---|---|---|---|---|---|---|
| synthetic, 24 views | 984 (41 epochs) | 0.11543 / 15.83 dB | 0.00599 / 32.51 dB | 0.04309 / 23.36 -> 0.00616 / 31.89 | yes | 651 MiB | 20.2 s (15.9 s training) |

Loss at every 100th step: 0.1154, 0.0124, 0.0100, 0.0307, 0.0253, 0.0071, 0.0074, 0.0075, 0.0097, 0.0064 (the rise at steps 300-400 was not investigated; it recovers, and I take it for the densification phase). One run; no 0x1016, `hipError` or `HSA_STATUS` in the log. Not measured on Windows: a public scene (truck), `--eval`, and `example_metrics.py`; the Linux figures for those are in the sections above.

Environment notes (host, not port): the `torchmetrics` import chain in this venv reaches a `transformers`/`tokenizers` pair that refuses to import (`tokenizers>=0.14,<0.19 is required ... found 0.20.3`), so the probe and the self-check stub `torchmetrics.image.psnr`; the trainer uses it only in the `--test_epochs` branch, which was not run. Pure-Python test dependencies (`pytorch_msssim tifffile matplotlib` and their pure deps) were installed into the scratch site dir with `--no-deps`.

CUDA no-regression compile: `cuda-not-validated: Windows host, ROCm-only PyTorch` (the nvcc compile of the d3213b4 pin recorded as owed in Review (3) still stands).

Gates: `python utils/jargon.py --port LiteGS`: **clean**. Documentation: README.MD:89 is still scoped "On Linux with AMD GPUs". That is true as written, and the same three commands with `--no-build-isolation` did build here, but only with the Windows host setup listed above (ROCm SDK variables, `HIP_DEVICE_LIB_PATH`, `DISTUTILS_USE_SDK=1`, clang-cl as `CC`/`CXX`, MSVC `link.exe` first on `PATH`). **For jeff:** whether the README (and the pull request text) should now say Windows works, and how much of that setup to name, is a porter documentation commit; it changes no compiled output. Integrity: `git status --porcelain` empty in the fork clone and the submodule at a995109 (only ignored build output: `*.hip`, `*_winhip.cu`, `build/`, `*.egg-info`).
