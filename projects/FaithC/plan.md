# FaithC -- ROCm/HIP port plan (lead: linux-gfx90a, MI250X, ROCm 7.2.1)

## Project
- Name: FaithC ("Faithful Contouring", `faithcontour`), CVPR 2026 Oral.
- Upstream: https://github.com/Luo-Yihao/FaithC  (default branch `main`, base sha 1580e2e "License update").
- What it is: a near-lossless 3D voxel mesh representation (Faithful Contour Tokens). The encoder voxelizes a mesh against an octree, does SAT polygon clipping + QEF solve per active voxel, and computes per-edge flux signs by segment-triangle intersection; the decoder reconstructs a mesh from the tokens. GPU work is in custom torch CUDA kernels plus the `atom3d` and `torch_scatter` dependencies.

## Existing AMD support
- None. README requires "NVIDIA GPU with CUDA support"; `__init__.py` asserts `torch.cuda.is_available()`; the shipped wheel `dist/faithcontour-0.1.0-cp310-cp310-linux_x86_64.whl` contains a CUDA-only prebuilt `_C.cpython-310-x86_64-linux-gnu.so`. No HIP path, no OpenCL/Vulkan/SYCL fallback, no pure-CPU fallback (ops.py hard-imports `_C`).
- Decision: PROCEED with a fresh CUDA->HIP port of FaithC's own `_C` extension. The kernels are portable hand-written CUDA (no CUTLASS/CuTe, no Hopper PTX, no warp intrinsics), so this is a mechanical Strategy-B port, NOT a reimplement.

## Build classification: torch-extension (Strategy B)
Evidence:
- `src/faithcontour/_C/{kernels.cu, bindings.cpp, kernels.h}` are `#include <torch/extension.h>` sources with a `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` (bindings.cpp:50) and `AT_DISPATCH_FLOATING_TYPES` dispatch (kernels.cu:324,713,729,744,795). Classic torch C++/CUDA extension.
- `ops.py:4` does `from . import _C`; `ops.py:7-10` binds the four kernels. `_C` is a hard import with no fallback.
- PACKAGING GAP (must fix to build at all): the repo's `pyproject.toml` is a PURE-PYTHON `setuptools.build_meta` build with NO `setup.py` and NO `CUDAExtension`; it never compiles `_C`. The v0.1.0 wheel was built out-of-band and ships a CUDA-only `_C.so`. So a normal `pip install -e .` from the v1.5 source produces a package whose `import _C` fails. To build for ROCm we must add a `setup.py` (or wire `pyproject` to) a `torch.utils.cpp_extension.CUDAExtension` over the three `_C` sources. On a ROCm torch, building a `CUDAExtension` AUTO-hipifies the `.cu/.cuh` and links amdhip64/c10_hip/torch_hip (PORTING_GUIDE Strategy B). README's "no C++ compilation required" (v1.5 marketing) is contradicted by the hard `_C` import -- treat the C extension as required.

## Port strategy: B (torch-hipify), minimal source change
1. Add a `setup.py` (kept CUDA-native: plain `CUDAExtension("faithcontour._C", [bindings.cpp, kernels.cu])` + `BuildExtension`). On a ROCm torch this auto-hipifies; on a CUDA torch it builds the original. Do NOT add a compat header and do NOT hand-rename symbols (Strategy B rule).
2. Keep `.cu` in CUDA spelling; hipify translates it. The four kernels use only `atomicAdd`, `__syncthreads`, dynamic/`extern __shared__`, `fminf/fmaxf/rsqrtf/sqrt/fabs`, `data_ptr<float|long|int|...>()`, and `AT_DISPATCH_FLOATING_TYPES` -- all 1:1 under hipify with no fault-class edits expected. There is NOTHING to guard with `USE_ROCM` in the device code (no warp size, no shfl/ballot, no cub/thrust/cublas/curand, no textures/surfaces, no cooperative groups, no managed/pinned memory, no streams/events).
3. If hipify leaves a stale mirror after an edit, re-run the project's hipify/build before rebuilding (Strategy B incremental gotcha).

Rationale: the kernel surface is small (4 kernels, ~810 lines) and entirely portable; the only real work is restoring a buildable extension and validating numerics on wave64.

## CUDA surface inventory (FaithC `_C` only)
Kernels (all in `src/faithcontour/_C/kernels.cu`):
- `k_segment_tri_intersection_fused_float` (L37) -- one block per segment, 256-thread tiled broadphase (AABB) + Moller-Trumbore narrowphase; writes hits via `atomicAdd(counter,1)` into `out_seg_indices/out_tri_indices/out_dots`. Dynamic `extern __shared__ float` tile (`smem_fused_kernel_float`, sized host-side `threads*3*2*sizeof(float)` at L165). `__syncthreads()` at L80,136.
- `k_preprocess_tris<scalar_t>` (L242) -- elementwise triangle preprocessing (bound but currently commented out in ops.py).
- `k_gen_candidates_overlap<T>` (L284) -- 2D grid (32x32 block, L321) broadphase AABB-overlap candidate gen; `atomicAdd(counter,1)` (L306) with a `cap`/`overflow` guard.
- `sat_hit_kernel` / `sat_centroid_kernel<T,MAXV>` / `sat_clip_kernel<T,MAXV>` (L433/447/564) -- narrowphase tri-AABB SAT + Sutherland-Hodgman 6-plane clip + barycentric centroid; MAXV templated 7/8; 256-thread 1D grid. No atomics.
- `k_voxelize_mark<T,USE_SAT>` (L766) -- 2D grid (32x32, L794) voxel-AABB mark with optional SAT; writes `active_mask` bytes (no atomic; idempotent set-to-1).
Host wrappers + bindings: `gen_candidates_overlap`, `aabb_tri_sat_clip_select` (mode 0/1/2), `voxelize_mark`, `segment_tri_intersection_fused` (bindings.cpp).

Warp intrinsics: NONE. Shuffle/ballot/activemask: NONE. `warpSize`/hardcoded 32 as a WARP width: NONE (the `dim3 blk(32,32)` at L321/L794 are 2D BLOCK tile dims, not warp assumptions -- arch-agnostic). cooperative groups: NONE. Textures/surfaces: NONE. cuBLAS/cuFFT/cuRAND/cuSPARSE/Thrust/CUB: NONE. Pinned/managed memory, streams, events: NONE (default stream, plain tensor allocs).

Dependency GPU surface (NOT in this repo, gates full-pipeline validation):
- `atom3d` (https://github.com/Luo-Yihao/Atom3d, ~25% CUDA): provides `MeshBVH` and `atom3d.grid.{OctreeIndexer,CubeGrid}` -- the BVH build + octree traversal that feed the encoder/decoder. Its own CUDAExtension build is CUDA-only with no HIP path.
- `torch_scatter` (`scatter_sum/mean/max/min/softmax`): used by qef_solver/segment_ops/decoder/api. Upstream torch_scatter builds on ROCm torch (auto-hipify) but must be compiled for this ROCm.

## Risk list
- Atomic-ordering nondeterminism (PRIMARY validation concern, NOT a bug): both `atomicAdd(counter,1)` paths assign OUTPUT SLOTS in nondeterministic order, so `out_seg_indices/out_tri_indices/out_dots` (and `cand_a/cand_t`) come back in a run-dependent permutation on every GPU (CUDA included). Validation MUST be order-independent: sort the (seg_idx, tri_idx) pairs (and the candidate pairs) before comparing to a reference; never assert positional equality across runs.
- wave64: no warp-width exposure at all (no shfl/ballot/warpReduce/cub, no per-warp shared producer->consumer, no leader-election-by-ballot). The `extern __shared__` tile is sized purely by `blockDim.x=256` and fully `__syncthreads`-guarded, so it is wave-size-agnostic. Expectation: clean on both gfx90a (wave64) and gfx1100 (wave32) with no `kWarpSize` work. Still BUILD the multi-arch fat binary as the warp-size correctness test even though no warp code is present (cheap insurance).
- Float-contraction / fast-math drift: clang(HIP) defaults to `-ffp-contract=fast` vs nvcc expression-only; the Moller-Trumbore det/u/v/t chain and the SAT projections are multi-statement float math, so HIP results can drift ~1 ULP from a CUDA/CPU reference. This is geometry with epsilon thresholds (`eps`), not bit-exact KATs, so a tolerance compare absorbs it. If a hit/no-hit boundary case flaps, pin `-ffp-contract=on` in the extension's HIP flags (extra_compile_args for the hip cxx). Not expected to need it given the eps guards.
- `data_ptr<long>()` over int64 tensors (kernels.cu:171,329, etc.): `long` is 64-bit on Linux x86-64 and on the ROCm device ABI, matching `torch.kInt64`; no change. (Would only matter on Windows LLP64 -- a gfx1151 follower note, not lead.)
- `rsqrtf`/`rsqrt`/`sqrt`: device sqrt on gfx90a can be 1-ULP off correctly-rounded (PORTING_GUIDE), but FaithC uses them for normalization with eps floors (`fmax(nn,1e-38)`, `norm_sq>eps`), not bit-exact compares -- no `__dsqrt_rn` routing needed.
- Buffer-overflow guards are already in the kernels (`counter <= max_hits_guess` TORCH_CHECK at L175; `cand` `cap`/`overflow` at L307). The `max_hits_guess = num_segs*8+4096` heuristic (L152) is arch-independent; no OOB neighbor reads (no stencil / +-1 gather pattern).
- DEPENDENCY GATE (biggest practical risk to FULL validation): the end-to-end demo/encoder/decoder need `atom3d` (separate CUDA repo) AND `torch_scatter` on ROCm. atom3d is not yet a MOAT project. FaithC's OWN kernels are validatable standalone (see Test plan), but the demo is not until atom3d is ported. Recommend scaffolding `Luo-Yihao/Atom3d` as a MOAT project and recording it as a FaithC dependency; torch_scatter is an external pip dep to build against ROCm torch.

## File-by-file change list
- ADD `setup.py` (new): `CUDAExtension("faithcontour._C", ["src/faithcontour/_C/bindings.cpp","src/faithcontour/_C/kernels.cu"])` + `cmdclass={"build_ext": BuildExtension}`, `package_dir={"":"src"}`, `packages=find_packages("src")`. CUDA-native spelling; ROCm torch auto-hipifies. (Restores the buildable extension the v1.5 pyproject dropped.)
- EDIT `pyproject.toml`: keep metadata; ensure the build does not shadow setup.py (either remove the pure `setuptools.build_meta` redirect so setup.py drives build_ext, or add the ext via the build backend). Minimal: let setup.py own build_ext.
- `src/faithcontour/_C/kernels.cu`, `bindings.cpp`, `kernels.h`: EXPECTED no source edits (hipify handles it). Only touch if a fault-class issue surfaces in validation (e.g. add `-ffp-contract=on` via setup.py extra_compile_args, not a source change).
- No compat header, no `USE_ROCM` guards anticipated (Strategy B).

## Build commands (gfx90a)
Build the extension against the host ROCm torch (torch 2.13.0a0+, hip 7.2.53211; ROCm 7.2.1, hipcc present):
```
cd projects/FaithC/src
PYTORCH_ROCM_ARCH=gfx90a python setup.py build_ext --inplace   # or: pip install -e . --no-build-isolation
```
Multi-arch warp-size build-test (one fat binary, then confirm both code objects):
```
PYTORCH_ROCM_ARCH="gfx90a;gfx1100" python setup.py build_ext --inplace
python -c "import faithcontour._C as c; print(c.__file__)"
llvm-objdump --offloading src/faithcontour/_C.cpython-*-x86_64-linux-gnu.so | grep -E "gfx90a|gfx1100"
```
torch_scatter for ROCm (needed only for the full pipeline, not the kernel slice):
```
PYTORCH_ROCM_ARCH=gfx90a pip install --no-build-isolation git+https://github.com/rusty1s/pytorch_scatter.git
```

## Test plan
The project ships NO automated test suite (no tests/, no pytest, no ctest) -- only `demo.py`. The validatable GPU slice is the four `_C` kernels, exercised directly; the demo is the full-pipeline check once deps are ported.

GPU-validatable slice (independent of atom3d, the lead-platform validation gate):
- Build `_C` for gfx90a, then a Python harness (in agent_space/ during dev) that drives each binding on synthetic CUDA tensors and compares to a pure-torch CPU reference:
  1. `segment_tri_intersection_fused`: random segments + triangles; reference = vectorized torch Moller-Trumbore. Compare the SET of (seg_idx,tri_idx) hit pairs (sorted) and the per-pair `dots` within tol (~1e-4 rel); order-independent due to atomic slotting.
  2. `gen_candidates_overlap`: random AABBs + tri AABBs; reference = broadcasted torch overlap test. Compare the sorted set of candidate pairs; exercise the `overflow` path with a small `cap`.
  3. `voxelize_mark` (use_sat False AND True): reference = torch AABB-overlap (+ a CPU tri-AABB SAT for the SAT path). Compare `active_mask` exactly (idempotent set, no atomics -> deterministic).
  4. `aabb_tri_sat_clip_select` mode 0/1/2: reference = CPU SAT + Sutherland-Hodgman clip; compare hit_mask exactly and centroids/areas within tol (per-row aligned by candidate index, which is deterministic for these non-atomic kernels).
- Determinism check: run each twice; the non-atomic kernels (voxelize_mark, sat_*) must be bit-identical; the atomic kernels must give the same SORTED pair set both runs.
- Multi-arch: confirm gfx90a + gfx1100 code objects in the fat binary (above).

Full-pipeline validation (requires atom3d + torch_scatter on ROCm -- gated, likely a follow-up once atom3d is a MOAT project):
- `python demo.py` (default icosphere r=128) and `python demo.py -p assets/examples/pirateship.glb -r 512 -o output/pirateship.glb`; success = a non-degenerate reconstructed mesh exported (vertex/face counts in the README ballpark for the resolution) and no CUDA/HIP fault. Geometry tolerance compare against a CUDA reference run if one is available.

Non-GPU regression set: none in-repo (no CPU tests). Do not regress the CUDA build (setup.py must still produce a working CUDA `_C` on an NVIDIA torch -- keep sources CUDA-native).

## Open questions
- atom3d dependency: should MOAT scaffold `Luo-Yihao/Atom3d` and mark FaithC `depends_on` it so the FULL demo can be validated? The FaithC `_C` kernel slice is independently validatable on GPU now (the lead gate), but the README demo is not end-to-end runnable on AMD until atom3d is ported. Recommendation: validate the kernel slice for the FaithC lead port; track atom3d as a separate MOAT project for the end-to-end story.
- Confirm with the porter that re-introducing a `setup.py`/`CUDAExtension` is acceptable as the minimal change (it restores what the v1.5 pyproject silently dropped; the v0.1.0 wheel proves the extension is the intended build artifact). No upstream-visible action without approval.

## Delta plan: linux-gfx1100 (RDNA3, wave32) -- on demand
No anticipated delta: the kernels have zero warp-width dependence (no shfl/ballot/warpReduce/cub, fully `__syncthreads`-fenced shared mem). Validate by REBUILDING the same fork branch with `PYTORCH_ROCM_ARCH=gfx1100` and rerunning the kernel-slice harness; expect PASS with no source change. Only `-ffp-contract` could differ at eps boundaries (same flag fix applies to both arches). gfx1151 (Windows): additionally watch `long` width (LLP64) in `data_ptr<long>()` paths and torch_scatter/atom3d availability under the Windows HIP SDK.
