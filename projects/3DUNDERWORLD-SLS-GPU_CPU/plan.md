# 3DUNDERWORLD-SLS-GPU_CPU -- ROCm/HIP port plan

Upstream: https://github.com/theICTlab/3DUNDERWORLD-SLS-GPU_CPU (branch `dev`)
Lead platform: linux-gfx90a (CDNA2, wave64). GPU ordinal 2 on this host.

## What the project is

Structured-light 3D scanner. A projector throws gray-code stripe patterns at an
object; two calibrated cameras photograph the lit scene. Each pixel's temporal
on/off sequence across the pattern images decodes to a projector (x,y) via gray
code; pixels in both cameras that decode to the same projector cell are
correspondences, triangulated (ray-ray midpoint) into a colored point cloud.

Two reconstruction backends produce comparable point clouds:
- CPU: `src/lib/core/` (always built). The reference path.
- CUDA/GPU: `src/lib/ReconstructorCUDA/` (built only with `-DENABLE_CUDA=ON`).

The GPU path is what we port to HIP.

## Existing AMD support assessment

None. No HIP path, no OpenCL/Vulkan/SYCL alternative backend, no stale ROCm
branch (branches: dev, master, develop, DBG, refactor_gpu, add-license-1,
fix/github-actions-apt -- none AMD-related). The GPU code is plain CUDA C++
(`.cu`/`.cuh`) gated on `find_package(CUDA)`. This is a genuine CUDA->HIP port,
high value. Not already-supported, not ported-elsewhere.

## Build classification + strategy

Pure CMake project, NOT a pytorch extension (no Torch anywhere). The CUDA build
uses the legacy `find_package(CUDA)` module with `cuda_add_library` /
`cuda_add_executable` (CMakeLists at repo root line 42; `src/lib/ReconstructorCUDA/CMakeLists.txt`;
`src/app/CMakeLists.txt` line 6). That module predates first-class HIP and cannot
target AMD.

-> **Strategy A** (minimal footprint):
1. Add one compat header `src/lib/ReconstructorCUDA/cuda_to_hip.h` aliasing the
   small CUDA surface to HIP under `USE_HIP`, plain CUDA otherwise.
2. Replace the legacy `cuda_add_*` CMake with a `USE_HIP` (AMD) / `ENABLE_CUDA`
   (NVIDIA) switch: on HIP `enable_language(HIP)`, mark the existing `.cu` files
   `LANGUAGE HIP`, arch from `${CMAKE_HIP_ARCHITECTURES}` (default gfx90a only
   when unset -- never a literal, so gfx1100/gfx1151 need no source edit).
3. CUDA path stays byte-for-byte: `ENABLE_CUDA=ON` still goes through the old
   FindCUDA flow unchanged.

The NVIDIA build is untouched; only `.cu` translation units see the HIP
toolchain on AMD; host C++ (core/, app/SLS, calibration, graycode) is unchanged.

## CUDA surface (small + clean)

From a grep over `src/lib/ReconstructorCUDA` + `src/app/App_CUDA.cu`:
- Runtime/memory: cudaMalloc, cudaFree, cudaMemcpy, cudaMemcpyHostToDevice,
  cudaMemcpyDeviceToHost, cudaMemset, cudaPeekAtLastError, cudaGetErrorString,
  cudaError_t, cudaSuccess.
- Events (profiling only): cudaEvent_t, cudaEventCreate/Record/Synchronize/ElapsedTime.
- Atomics: atomicInc (bucket insert, uint, ReconstructorCUDA.cuh:35);
  atomicOr/atomicAnd are present only in commented-out lines (DynamicBits.cuh:43,53).
- Misc device: `__threadfence()` + inline `asm("trap;")` overflow guard
  (DynamicBits.cuh to_uint, ~line 87).
- Headers: `<device_launch_parameters.h>`, `<cuda_runtime.h>` (CUDA_Error.cuh:5).
- NO `__shfl*`/`__ballot`/`__activemask`/`__popc`/`warpSize`, NO textures/surfaces,
  NO curand/cublas/cufft/thrust/cub. All kernels are grid-stride, per-thread
  independent work.

Kernels: genPatternArray, buildBuckets, getPointCloud2Cam (ReconstructorCUDA.cu);
computeMask_kernel (FileReaderCUDA.cu); toUintArray_kernel, toNormalizedUintArray
(DynamicBits.cu).

## Fault classes that apply

- **wave64 / warp size: NONE.** No warp-level primitives or reductions anywhere;
  no hardcoded 32; no warp-sized shared arrays. gfx90a's 64-lane wavefront is
  irrelevant to these grid-stride kernels. (This also means gfx1100/gfx1151
  RDNA wave32 should pass with no delta.)
- **NVPTX inline asm:** `asm("trap;")` is the only NV-specific intrinsic. It is
  PTX, illegal for amdgcn. Replace the `__threadfence(); asm("trap;")` overflow
  guard with `__trap()` (HIP+CUDA both provide it) -- guarded in the compat
  header or via `__trap()` directly. This is a never-taken guard (bitsPerElem <=
  uint width here), so it has no effect on output, but it must compile.
- **OOB reads (AMD strict):** reads are safe; one upstream WRITE is not (see
  below).
  - `add2Bucket` guards `bktIdx > NUM_BKTS_-1` (ReconstructorCUDA.cuh:41).
  - `buildBuckets` writes only when decoded (x,y) < projector resolution AND
    mask set (ReconstructorCUDA.cu:212).
  - `getPointCloud2Cam` color index `minIdx0/1` are camera pixel indices pulled
    from bucket contents (in-bounds by construction); arrays sized resX*resY*3.
  - Read side of the buckets is in-bounds: the loops run `i < count_[idx]` with
    `count_ <= MAX_CNT_PER_BKT_`, so `idx*MAX_CNT_PER_BKT_+i` stays in the row.
  No stencil/neighbor +/-1 gathers exist, so no clamp fix is needed.
- **Upstream bucket-insert overrun (write side, NOT introduced by this port and
  NOT fixed by it):** `ReconstructorCUDA.cuh:45` passes `MAX_CNT_PER_BKT_` where
  the idiom needs `MAX_CNT_PER_BKT_-1`. `atomicInc(p, val)` stores
  `old >= val ? 0 : old+1` and returns `old`, so with `MAX_CNT_PER_BKT_ == 110`
  the 111th insert into a bucket sees `old == 110`, resets the counter and
  returns 110; the write `data_[110 + bktIdx*110]` lands in slot 0 of bucket
  `bktIdx+1`, and for the last bucket one uint past the
  `MAX_CNT_PER_BKT_*NUM_BKTS_` allocation. Present verbatim in the pre-port
  merge-base (c87fe37), identical semantics on CUDA and HIP, so it is upstream
  behaviour rather than a porting fault, and fixing it is outside "smallest
  complete port". Registered as deferred work
  (`sls-gpu-bucket-atomicinc-overrun`) to raise with the maintainers. Note that
  a clean run is NOT evidence against it: it needs >110 camera pixels decoding
  to one projector cell, and a one-uint overrun of a ~346 MB allocation stays
  inside the same page, so it would corrupt silently rather than fault.
- **atomicInc on managed memory:** N/A. Buckets use plain `cudaMalloc` device
  memory, and atomicInc is not in the dropped-RMW class (only int/uint
  atomicMin/atomicMax are, per PORTING_GUIDE) -- no emulation needed.
- Textures/pitch/rule-of-five/library swaps: none apply (no textures, no libs,
  no raw texture/stream/event RAII beyond two cudaEvents created+used in one
  scope).

## Dependencies to install

- OpenCV 4: already present (4.6.0). `find_package(OpenCV 4.0 REQUIRED)`.
- GLM: NOT installed; header-only. Install `libglm-dev` (apt) so FindGLM.cmake
  resolves `glm/glm.hpp`. Used by both CPU and GPU.
- ROCm/hipcc: present (/opt/rocm, gfx90a visible on GPU 2).

## Validation strategy (real GPU, GPU 2, HIP_VISIBLE_DEVICES=2)

The shipped test harness `test/RunTest.cpp` (gtest, `-DGTEST=ON`) reconstructs
the `alexander` and `arch` datasets on the **CPU** path and compares the output
PLY/OBJ against in-tree reference clouds `test/data/{alexander,arch}.{ply,obj}`
with per-coordinate tolerance MAX_DIFF=10.0. That validates the CPU reference but
does NOT exercise the GPU (runCPUTest links `core` only).

So for GPU validation:
1. Build CPU `SLS` and GPU `SLS_GPU` (the latter via the HIP path) against the
   `alexander` dataset (the README demo dataset; 1024x768, jpg).
2. Run `SLS_GPU` on GPU 2 -> output_gpu.ply. Run CPU `SLS` on the same data ->
   output_cpu.ply. Both decode the same gray code and triangulate the same way.
3. Correctness check (a small Python comparator in agent_space, not committed):
   - The GPU writes a fixed NUM_BKTS slots (zeros for empty buckets); export
     drops zero rows, so both PLYs are sparse non-zero points. Compare on the
     SET of points: for each CPU point, nearest GPU point within a small
     Euclidean tol; assert high coverage both directions and matching point
     count to within a small fraction. (CPU `intersectionOfBucket_` averages all
     ray-pair midpoints; GPU `getPointCloud2Cam` averages too -- same estimator,
     so points should agree closely, not just topologically.)
   - Cross-check both against the gtest reference cloud `test/data/alexander.ply`
     (MAX_DIFF=10 world units) as ground truth, via the project's own comparator
     semantics.
   - Sanity: decoded correspondence count (non-zero buckets) is in the expected
     thousands, not ~0 (which would signal a decode/atomic fault).
4. Determinism: run `SLS_GPU` twice on GPU 2; assert identical (or
   tol-identical) point clouds. Bucket insertion uses atomicInc so intra-bucket
   order can vary, but the per-bucket estimator is an average over all pairs and
   the min-dist color pick is order-stable enough; if run-to-run shows only
   ULP-level coordinate jitter (from float reduction order) that is acceptable
   and noted, but the non-zero point SET must be stable.
5. Also run the CPU gtest (`-DGTEST=ON && make && ctest`) to prove no regression
   in the existing suite.

A CPU-only docker compile is NOT a validation gate (cannot see GPU faults); the
real run on GPU 2 is the gate.

## Steps

1. apt install libglm-dev.
2. Add `src/lib/ReconstructorCUDA/cuda_to_hip.h`; include it from CUDA_Error.cuh
   (the common header) instead of bare `<cuda_runtime.h>`.
3. Replace `asm("trap;")` with `__trap()` (portable) in DynamicBits.cuh.
4. Rewrite the three CMake spots for the USE_HIP/ENABLE_CUDA switch; keep
   `cuda_add_*` for the CUDA branch, plain `add_library`/`add_executable` +
   `LANGUAGE HIP` for the HIP branch. Arch from `${CMAKE_HIP_ARCHITECTURES}`.
5. Configure+build with `-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a` (+
   `-DGTEST=ON`). Record the exact build command in notes.md.
6. Run GPU 2 reconstruction + CPU reconstruction + comparator + determinism +
   CPU gtest. Record numbers in notes.md.
7. On success: linux-gfx90a -> ported; commit status.json, plan.md, notes.md,
   PORTING_GUIDE changelog line. No fork/push, no README, no GHA.
