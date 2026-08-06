# cubvh -- ROCm/HIP port plan (lead: linux-gfx90a)

## Project
- Name: cubvh
- Upstream: https://github.com/ashawkey/cubvh (canonical, maintained; ashawkey merges contributor PRs, last default-branch activity 2026-01-27)
- Default branch: main
- Cloned at HEAD 7855c000f95e43742081060d869702b2b2b33d1f
- License: MIT (plus LICENSE_NVIDIA for the tiny-cuda-nn-derived gpu_memory.h / pcg32.h). Permissive; upstreamable.
- PR target: ashawkey/cubvh (NOT JeffreyXiang/cubvh@trellis.2). The existing AMD-Ecosystem/cubvh fork is in ashawkey's network, so a branch off ashawkey/main on that fork PRs directly to ashawkey.

## Existing AMD support -> decision: from-scratch port our way
- README/docs grep (`amd|rocm|hip|gfx`): zero hits. No upstream AMD path of any kind (no OpenCL/Vulkan either).
- Web search ("cubvh ROCm/AMD/HIP"): nothing project-specific.
- Upstream PR queue (all authors, state=all, ROCm/HIP/AMD): no AMD-support PR has ever been opened.
- Fork scan (`gh api repos/ashawkey/cubvh/forks`): one separately-named ROCm fork exists -- `manjunaths/cubvh-rocm` (https://github.com/manjunaths/cubvh-rocm), a fork-of ashawkey/cubvh, 0 stars, single author, last push 2026-02-27.
  - Authoritativeness: NON-authoritative community fork. Its entire delta vs upstream HEAD is `readme.md +5/-4` and `setup.py +4/-4`; the setup.py change merely COMMENTS OUT the NVCC-only flags (`--extended-lambda`, `--expt-relaxed-constexpr`, `-Xcompiler=...`). It carries NONE of the source fixes ROCm actually needs: no `thrust::cuda::par` -> `thrust::hip::par` (so the sparse-marching-cubes path would not compile on rocThrust), no torch-ROCm stream include, no arch handling, no validation. Treat as a reference-only HINT (confirms "the NVCC flags must come off the HIP path"); do NOT adopt as a base.
- Decision: a clean from-scratch HIP port targeting ROCm, contributed upstream to ashawkey/cubvh. This is genuinely additive (first AMD support).

## Build classification -> torch extension (Strategy B)
Evidence:
- `setup.py` line 3: `from torch.utils.cpp_extension import BuildExtension, CUDAExtension`; lines 109-130 build a single `CUDAExtension(name="_cubvh", sources=[bvh.cu, api_gpu.cu, bindings.cpp])` with `cmdclass={"build_ext": BuildExtension}`.
- `pyproject.toml`: `dependencies = [... "torch" ...]`, build-backend setuptools.
- No CMake anywhere. ext_type already set to `torch-extension` in upstream.json/status.json (correct).

On a ROCm torch, `CUDAExtension` auto-runs `torch.utils.hipify` over the `.cu`/`.cuh` sources and links amdhip64/c10_hip/torch_hip. Host toolchain confirmed: ROCm 7.2.1, torch 2.13.0a0 (hip 7.2), `IS_HIP_EXTENSION=True`, hipify `__version__=2.0.0`.

## Port strategy: Strategy B (torch hipify) + 3 targeted source/build edits
Keep all sources in CUDA spelling; let torch hipify translate the bulk (cudaMalloc/cudaMemcpy/cudaStream_t/atomics are 1:1). Hand-fix only what hipify cannot:

1. `include/gpu/spcumc.cuh` -- rocThrust has no `thrust::cuda::par`. Add a guarded policy macro near the includes:
   ```cpp
   #ifdef USE_ROCM
   #include <thrust/system/hip/execution_policy.h>
   #define THRUST_CUDA_PAR thrust::hip::par
   #else
   #define THRUST_CUDA_PAR thrust::cuda::par
   #endif
   ```
   Replace all 14 `thrust::cuda::par.on(stream)` call sites (lines 649,676,695,702,722,761,763,789,793,802,812,821,825,847) with `THRUST_CUDA_PAR.on(stream)`. (notes.md said ~16; the real count is 14.) Re-applied logically against ashawkey/main's spcumc.cuh, NOT cherry-picked from the trellis.2-based AMD-Ecosystem/cubvh@moat-windows commit.

2. `src/api_gpu.cu` -- under `USE_ROCM`, include the torch-ROCm masquerading stream header so `at::cuda::getCurrentCUDAStream()` resolves cleanly:
   ```cpp
   #ifdef USE_ROCM
   #include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
   #endif
   ```
   Note: on this host's hipify v2 the `at::cuda::getCurrentCUDAStream().stream()` spelling already survives (v2 keeps the CUDA-masquerading API public), so this include is belt-and-suspenders for older hipify v1 (TheRock/Windows). Harmless on Linux.

3. `setup.py` -- ROCm build path. Detect HIP and branch the flags:
   - `from torch.utils.cpp_extension import IS_HIP_EXTENSION` (True on a ROCm torch).
   - On HIP: do NOT pass the NVCC-only flags (`--extended-lambda`, `--expt-relaxed-constexpr`, `-Xcompiler=...`, `-allow-unsupported-compiler`, the `-U__CUDA_NO_HALF*` defines are unnecessary under hipcc). Pass a clean hipcc set (`-O3 -std=c++17`) plus `-DUSE_ROCM`. Provide offload-arch handling: default to the GPU's native arch (let PYTORCH_ROCM_ARCH / torch's default drive it) and allow an env override; never hardcode gfx90a. Keep the existing NVCC branch byte-identical for the CUDA build.
   - The `-DUSE_ROCM` define is what activates edits 1 and 2 (torch also defines USE_ROCM for hipified TUs, but defining it in the extension args guarantees the header `.cuh` sees it regardless of include order).

DO NOT port `src/bindings_winhip.cu` from the CuMesh dep commit: it `#include <eigen_hip_compat.h>` (CuMesh's private header) and exists only to compile the pybind binding via hipcc on Windows/MSVC. It is CuMesh-coupled and not upstreamable as-is. The Linux gfx90a port needs only edits 1-3. Windows/MSVC pybind compilation is scoped OUT of the first PR (Windows follower handled separately, see Open questions).

## CUDA surface inventory
- Kernels: 112 `__global__`/`__device__`/`__host__` occurrences across bvh.cu, api_gpu.cu, floodfill.cuh, hashtable.cuh, spcumc.cuh, triangle.cuh, bounding_box.cuh, bvh.cuh, gpu_memory.h. All translate 1:1 under hipify.
- Runtime API: cudaMalloc/Free/Memset/Memcpy/MemcpyAsync/Memcpy{H2D,D2H,D2D}/StreamSynchronize/DeviceSynchronize/GetLastError/GetErrorString/cudaStream_t. All 1:1 hipify, no risk.
- Thrust: heavy use in spcumc.cuh (device_vector, sort, scan, reduce_by_key, copy_if, scatter, gather, transform, sequence, sort_by_key, for_each, zip/counting/transform iterators) and api_gpu.cu (device_vector, raw_pointer_cast). -> rocThrust on ROCm. The ONLY non-1:1 piece is the execution policy `thrust::cuda::par` (edit 1).
- Atomics: atomicCAS (hashtable.cuh:124, floodfill.cuh:97), atomicExch (floodfill.cuh:101). 1:1 on ROCm.
- Streams: `at::cuda::getCurrentCUDAStream()` (api_gpu.cu, 4 sites) -> torch-ROCm stream (edit 2).
- NO cub/hipCUB, NO warp intrinsics (`__shfl*`/`__ballot`/`__activemask`), NO `__syncthreads`/`__syncwarp`, NO textures/surfaces/cudaArray, NO cuBLAS/cuFFT/cuRAND/cuSPARSE/cuDNN/NCCL, NO pinned/managed memory, NO events. (`__popc` in spcumc on bit-mask cube indices is arch-agnostic.)
- pcg32.h is a host/device PRNG header (no CUDA API). gpu_memory.h is a tiny-cuda-nn-derived RAII allocator using only cudaMalloc/Free/Memset/Memcpy -- 1:1.

## Risk list
- LOW overall: this is one of the cleaner Strategy-B surfaces (no warp ops, no textures, no math-library swaps, no cub).
- Warp size: NO hardcoded-32 warp geometry. The `32`/`& 31` hits are PRNG rotation bits (pcg32) and 8-bit marching-cubes cube indices (`ci |= 32`), not wavefront logic. `bvh.cuh:24 MAX_SIZE=32` is a traversal-stack depth, not a lane count. No wave64/wave32 hazard -> the gfx1100/gfx1151 followers should be a clean rebuild with no delta.
- rocThrust execution policy (edit 1) is the one real translation; well-understood.
- `at::cuda::getCurrentCUDAStream().stream()` -> the masquerading type must expose `.stream()`; covered by edit 2. Watch the hipify v1 vs v2 axis on the Windows follower (TheRock torch may be v1; this Linux host is v2). Guard on `torch.utils.hipify.__version__` only if a real divergence appears -- do not pre-emptively add it.
- Fresh device memory not zeroed on ROCm: floodfill.cuh and gpu_memory.h `cudaMalloc` then `cudaMemset` before use (checked: floodfill memsets, allocator memsets) -- no reliance on implicit zeroing observed. Re-confirm during bringup if floodfill output looks wrong.
- Eigen: build needs `third_party/eigen` (a submodule, NOT in the shallow clone). Clone `--recursive` (or apt `libeigen3-dev`) for the build. Eigen is host-only here (CPU mesh setup in api_gpu.cu); no device Eigen risk.
- `-ffp-contract`: distance queries compare against trimesh CPU ground truth at atol=1e-5 (signed/unsigned_distance.py). hipcc defaults to `-ffp-contract=fast`; if the 1e-5 tolerance is exceeded, pin `-ffp-contract=on` in the HIP cflags. Likely fine at 1e-5 but flagged.
- No rule-of-five / texture / OOB-neighbor / 256B-pitch / smid-pool classes apply (none of those constructs are present).

## File-by-file change list
- `include/gpu/spcumc.cuh`: add USE_ROCM THRUST_CUDA_PAR macro + include; replace 14 `thrust::cuda::par.on(stream)` -> `THRUST_CUDA_PAR.on(stream)`. (Copyright header: add AMD parallel line + author, substantive edit.)
- `src/api_gpu.cu`: add USE_ROCM-guarded `#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>`. (Copyright header line.)
- `setup.py`: IS_HIP_EXTENSION branch -> hipcc flag set + `-DUSE_ROCM` + env-overridable native offload-arch; keep NVCC branch unchanged.
- `readme.md`: add a brief ROCm/AMD install note in the project's house style (parallel to the existing "Make sure torch and CUDA are installed" block) -- PR-prep, house-style, no imposed build-command block beyond what upstream already documents.

## Build commands (gfx90a)
```bash
# fetch eigen submodule (shallow clone omitted it)
git -C projects/cubvh/src submodule update --init --recursive third_party/eigen
# or: apt-get install -y libeigen3-dev  (and point include at it)

# build against the ROCm torch, in-place
cd <fork>/  # branch off ashawkey/main on AMD-Ecosystem/cubvh
PYTORCH_ROCM_ARCH=gfx90a pip install -e . --no-build-isolation -v
# (followers: PYTORCH_ROCM_ARCH=gfx1100 / gfx1201, same source, no edit)
```
Host: ROCm 7.2.1, torch 2.13 (hip 7.2), IS_HIP_EXTENSION=True, hipify 2.0.0.

## Test plan
Real-GPU correctness gates (self-contained, no external mesh, reference-checked):
- `python test/signed_distance.py` -- builds BVH on a dodecahedron, queries signed_distance(mode=raystab), asserts allclose vs trimesh.proximity at atol=1e-5. Exercises BVH build + ray-stab + distance kernels.
- `python test/unsigned_distance.py` -- same shape, unsigned_distance vs reference. Exercises UDF + closest-face kernels.
- `python test/cuhashtable.py` -- hard assertions on cuHashTable build/search (3D, 4D, 200k-key random). Exercises hashtable.cuh atomicCAS path. Run via `pytest test/cuhashtable.py` (it has test_* functions).
- `python test/state_dict.py` -- BVH serialize/deserialize round-trip on GPU.
- `python test/sparse_voxel.py` -- the spcumc (sparse CUDA marching cubes) path -- THIS is the code edited in spcumc.cuh (thrust::hip::par). Runs UDF on a res=1024 grid + floodfill + sparse_marching_cubes. Confirms edit 1 works end-to-end. (Heavier; uses generated geometry / a small mesh.)
- floodfill is exercised transitively by sparse_voxel / watertight_remesh.

Non-GPU regression set (must not break): CPU paths -- `test/hashtable.py` (CPU HashTable), `test/merge_vertices.py`, `test/repair_holes.py`, `test/decimation.py`, `sparse_marching_cubes_cpu`. These do not touch the HIP edits; confirm they still import and run (some need a mesh-path arg -- smoke a tiny mesh).

Validation bar: signed/unsigned_distance + cuhashtable + sparse_voxel all pass on gfx90a (ROCm 7.2.1), CPU tests unregressed. A docker CPU build is not sufficient.

## PR plan (against ashawkey)
1. Branch `moat-port` off ashawkey/main on AMD-Ecosystem/cubvh; disable Actions on the fork (`gh api -X PUT repos/AMD-Ecosystem/cubvh/actions/permissions -F enabled=false`).
2. Land edits 1-3 + readme note as the port; validate on gfx90a; then gfx1100 (clean rebuild expected) and a Windows tier arch (gfx1201) -- Windows needs the pybind/hipcc question resolved (see below), so the first upstream PR may scope Windows out and claim Linux only, with Windows as a follow-up.
3. Single upstream PR `moat-port` -> ashawkey/cubvh:main, `[ROCm]`-prefixed, no MOAT vocabulary. Show full draft to jeff before opening (upstream-visible gate).
4. Follow-up (not in this PR): repoint CuMesh's `third_party/cubvh` submodule from AMD-Ecosystem/cubvh@moat-windows to the merged ashawkey commit once ROCm support lands (deferred coordination, per notes.md).

## Open questions
- Windows/MSVC pybind: upstream compiles bindings.cpp with MSVC + nvcc on Windows. Under ROCm the binding TU goes through hipcc, which MSVC rejects for HIP attributes (the problem the CuMesh bindings_winhip.cu worked around via a CuMesh-private header). For a standalone cubvh this must be solved inside cubvh (e.g. keep bindings.cpp host-only and not hipified, or a small in-repo compat shim) -- scope it as a Windows follower task, not the lead. The Windows-tier PR-readiness can be satisfied by gfx1201 once this is solved; the lead PR need not block on it.
- `-ffp-contract=on` only if the 1e-5 distance tolerance fails on gfx90a (likely not needed).
- Eigen: prefer the recursive submodule for a reproducible build; document apt libeigen3-dev as the fallback in the readme note if upstream prefers it.
