# cupoch -> ROCm/HIP port plan (lead: linux-gfx90a)

Upstream: https://github.com/neka-nat/cupoch @ bd6cec71d788060d06e4ce8fae903c8bc217d434
Target: gfx90a (MI250X, CDNA2, wave64), ROCm 7.2.1. Followers gfx1100/gfx1151 reuse this branch.

## What cupoch is

GPU 3D point-cloud / geometry-processing library (an Open3D analog) with pybind11
Python bindings (NOT torch -> Strategy A). Modules: geometry, knn, registration,
odometry, integration, io, collision, planning, kinematics, kinfu, imageproc,
visualization, utility. Heavy Thrust user; small custom-kernel surface.

## Existing AMD support assessment

NONE. `grep -riE 'hip|rocm|gfx|amdgpu'` over src/cmake/CMakeLists = 0 hits. Pure
CUDA. So this is a fresh CUDA->HIP port (not finishing a stale one). Strategy A.

## CUDA surface (decisive for strategy)

- Thrust everywhere (94 files): transform / sort_by_key / reduce_by_key / scan /
  gather / copy. rocThrust is a header drop-in (same `thrust::` API, same
  `<thrust/...>` paths under /opt/rocm/include) -> NO source swap for these.
- NO cuBLAS / cuSOLVER / cuSPARSE / cuRAND / CUB anywhere (grep = 0). So no math
  library compat header is needed at all. (Eigen does the small dense solves on
  host/device via its own headers.)
- Raw CUDA runtime surface is SMALL and lives mostly in utility/platform.cu and a
  `cudaSafeCall` macro: cudaStreamCreate, cudaGetDevice/SetDevice/GetDeviceCount,
  cudaGetDeviceProperties, cudaDeviceSynchronize, cudaMemcpy(+kinds), cudaMalloc/
  FreeHost, cudaGetLastError/GetErrorString. All 1:1 hip* spellings.
- Custom `__global__` kernels in only 4 files (knn/bruteforce_nn, knn/lbvh_knn,
  geometry/distancetransform, integration/scalable_tsdfvolume, registration/
  permutohedral). NO `__shfl`/`__ballot`/`__activemask`/`warpSize`/`atomicMin`/
  `atomicMax` in core src -> wave64 risk is LOW (no warp-collective or
  warp-width-dependent code on the core path). Will still grep each ported kernel.
- CUDA-OpenGL interop (cudaGraphicsGLRegisterBuffer etc.) only in visualization
  -> GUI module, DEFERRED.

## Third-party dependencies (the real difficulty)

- rmm (RAPIDS Memory Manager): CUDA-only, libcudacxx-coupled. OPTIONAL via
  `USE_RMM`. When OFF, `utility::device_vector<T>` falls back to
  `thrust::device_vector<T>` (device_vector.h) and the RMM allocator setup in
  device_vector.cu is `#ifdef USE_RMM` dead code. -> BUILD WITH `USE_RMM=OFF`;
  this removes the single hardest dependency cleanly with no source edit.
- stdgpu (GPU STL): HAS a first-class HIP backend (`STDGPU_BACKEND_HIP`,
  src/stdgpu/hip/impl/*). Header `<stdgpu/functional.h>` is on the geometry
  compile path via utility/helper.h, so stdgpu must build/configure for HIP.
- flann (header CUDA kdtree, third_party/flann/algorithms/kdtree_cuda_3d_index.cu,
  657 lines, Thrust-based, 1 cudaMemset, uses textures): on the CORE path -- used
  by knn/kdtree_flann + geometry estimate_normals/down_sample(outlier removal)/
  iss_keypoints/cluster + registration. Needed for EstimateNormals; NOT needed for
  VoxelDownSample.
- libSGM (stereo): imageproc only -> DEFERRED.
- Eigen, spdlog, dlpack, tinyobjloader, imgui, rply/liblzf/jsoncpp: host-only or
  header-only, backend-agnostic.

## Build-system obstacle (central challenge)

cupoch uses the LEGACY FindCUDA module: top-level `find_package(CUDA REQUIRED)` +
`enable_language(CUDA)`, and every GPU lib is built with `cuda_add_library(...)`
(FindCUDA), driving `CUDA_NVCC_FLAGS`. Strategy A's normal recipe
(`enable_language(HIP)` + `set_source_files_properties(.. LANGUAGE HIP)`) assumes
modern target-based CUDA, which this project does not use. Plan:

- Add `option(USE_HIP)`. When ON: do NOT `find_package(CUDA)`/`enable_language(CUDA)`;
  instead `enable_language(HIP)` and set `CMAKE_HIP_ARCHITECTURES` only if unset
  (NEVER hardcode gfx90a literally -- followers must build with just
  `-DCMAKE_HIP_ARCHITECTURES=<arch>`).
- Provide a `cuda_add_library()` shim macro (USE_HIP only) that calls
  `add_library()` + marks the `.cu` sources `LANGUAGE HIP`. Keeps every module
  CMakeLists (`cuda_add_library(cupoch_geometry ...)`) UNCHANGED -> minimal diff,
  CUDA path byte-identical.
- Map `CUDA_LIBRARIES`/include dirs to the HIP runtime + rocThrust/hipCUB/rocprim
  include paths under USE_HIP.
- Build stdgpu sub-project with `-DSTDGPU_BACKEND=STDGPU_BACKEND_HIP`.

All changes guarded by `if(USE_HIP)` / `#if defined(USE_HIP)`; the NVIDIA build is
untouched.

## Compat header

Add `src/cupoch/utility/cuda_to_hip.h` (the only file that knows HIP): on
USE_HIP, `#include <hip/hip_runtime.h>` and alias the ~20 `cuda*` runtime symbols
the project actually uses (enumerated above) to their hip* spellings; everywhere
else a no-op `#include <cuda_runtime.h>`. Include it from utility/platform.h and
device_vector.h in place of the bare `<cuda_runtime.h>`. Per cudaKDTree lesson,
include `<cstring>`/`<cstdlib>` before the HIP runtime so libc host decls win.
`cuda_gl_interop.h` is visualization-only and stays out of the core path.

## Scope (pragmatic, per dispatch)

CORE TARGET (must build + GPU-validate):
- cupoch_utility + cupoch_geometry (+ its link deps cupoch_knn, cupoch_camera).
- If flann builds under HIP: include knn/kdtree_flann so EstimateNormals works.

DEFERRED (documented in notes.md, not a failure):
- visualization (CUDA-GL interop), imageproc/libSGM stereo, kinfu, kinematics,
  planning, integration, io (beyond what geometry needs), full pybind11 Python
  module, RMM allocator path. These are peripheral to proving a real end-to-end
  GPU geometry op and/or carry their own heavy CUDA-coupled deps.

Stretch (only if core is clean and time remains): registration ICP, then the
Python module.

## Validation (real GPU, not compile)

Primary: VoxelDownSample (PURE Thrust: transform -> sort_by_key -> reduce_by_key;
no flann, no custom kernel). C++ harness on gfx90a (HIP_VISIBLE_DEVICES=1):
- Build a known point cloud (deterministic generator), run
  `PointCloud::VoxelDownSample(voxel)` on GPU.
- CPU reference: replicate the exact grid-bin formula (floor((p-min)/voxel)),
  average points per occupied voxel; compare centroids set-wise within ~1e-4.
- Determinism: run >=3x, assert identical output (bitwise / within 0 tol on the
  sorted centroid set).

Secondary (if flann/kdtree builds): EstimateNormals with KNN search. CPU
reference: brute-force kNN + 3x3 covariance + smallest-eigenvector normal; compare
|dot(n_gpu, n_cpu)| ~ 1 (sign-free) within ~1e-3 over all points; determinism over
repeated runs.

Compile-only or lint is explicitly NOT acceptance.

## Fault classes to watch (from PORTING_GUIDE)

- wave64: low risk (no warp-collective/atomicMin-Max/shfl on core path); re-grep
  each ported kernel anyway.
- rocThrust/hipCUB/rocPRIM require C++17 (project already builds C++17 -> fine).
- Two-phase lookup / `this->` on dependent bases; `__CUDA_ARCH__` vs
  `__HIP_DEVICE_COMPILE__` in any `__host__ __device__` macro -- grep utility
  headers for these (helper.h, eigen.inl) and fix if hit.
- HIP `__shfl*`/ballot need 64-bit masks -- only relevant if a ported kernel uses
  them (none on core path yet).
- pybind11 + HIP LTO drops PyInit_ (NO_EXTRAS) -- only relevant if we attempt the
  Python module (deferred); noted for later.

## Deliverables

Port lives in projects/cupoch/src (gitignored; parent delivers the fork). Commit to
MOAT only: status.json, plan.md, notes.md, and any PORTING_GUIDE changelog line.
On validated core -> set linux-gfx90a `ported`; record covered/deferred in notes.md.

## PR-review reopen (Jeff, 2026-06-24): full-port scope + build-system fixes

The core-only scope above was a planner convenience for a headless validation
host, not a porting limit. PR review rejected it. Expanded requirements:

1. SCOPE: USE_HIP must build the SAME module set as the CUDA build. Of the 14
   modules, only `visualization` is genuinely GL/CUDA-GL-interop coupled (11
   gui/codec CMake refs); the other 9 currently skipped (collision, imageproc,
   integration, io, kinematics, kinfu, odometry, planning, registration) are
   GPU-compute modules with ~1 incidental ref each and should port like the
   core did. Port them all and GPU-validate. Drop CUPOCH_CORE_ONLY as the
   deliverable: USE_HIP should build everything portable; guard only the truly
   non-viable bits (visualization GL interop, IF it proves unportable on ROCm)
   automatically -- not behind a user-facing partial-build switch. Attempt the
   pybind11 Python module too (it links visualization today; may need the viz
   question resolved first). `io` needs png/jpeg codecs (host libs, apt-
   installable) -- not a GPU blocker.

2. BUILD FIX (done in working tree): -DNOMINMAX / -D_USE_MATH_DEFINES were in
   the unconditional `if(WIN32)` block, so they altered the NVIDIA Windows
   (MSVC+nvcc) build too -- moved under the USE_HIP Windows-clang branch.

3. BUILD FIX (done in working tree): USE_RMM was a plain option forcing the user
   to pass -DUSE_RMM=OFF. Now `cmake_dependent_option(... "NOT USE_HIP" OFF)` so
   it is forced OFF and hidden under USE_HIP.

4. NON-ISSUE confirmed: cuda_add_library in the CUDA path comes from FindCUDA
   (find_package(CUDA REQUIRED)); the shim only shadows it under USE_HIP. Leave
   as-is (FindCUDA deprecation is pre-existing upstream usage).

DECISION (Jeff, 2026-06-24): "Compute modules + best-effort viz". Port all 9
compute modules + the Python bindings; ATTEMPT visualization (CUDA-GL interop);
if ROCm HIP-GL interop is non-viable after a real attempt, guard visualization
out under USE_HIP automatically and document it as the single boundary. No
user-facing partial-build flag in the final shape.

Re-validate on real GPU after expansion (functional change -> all platforms
revalidate). Only then redo PR prep (docs already cover the build; revisit the
scope paragraph in README once the module set is final) and re-squash.
