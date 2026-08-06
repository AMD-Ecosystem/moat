# cupoch notes (ROCm/HIP port)

Upstream: neka-nat/cupoch @ bd6cec71d788060d06e4ce8fae903c8bc217d434
Lead platform: linux-gfx90a (MI250X, CDNA2, wave64), ROCm 7.2.1, CMake 3.31.6.

## Existing AMD support

None. `grep -riE 'hip|rocm|gfx|amdgpu'` over src/cmake/CMakeLists = 0 hits. Fresh
CUDA -> HIP port. Strategy A (pure CMake, not a torch extension).

## CUDA surface

- Thrust-dominated (94 files: transform / sort_by_key / reduce_by_key / scan /
  gather / async copy). rocThrust is a header drop-in -- no Thrust source swap.
- NO cuBLAS / cuSOLVER / cuSPARSE / cuRAND / CUB anywhere -> no math-library compat
  header needed. Dense solves go through Eigen (host + device headers).
- Small raw CUDA runtime surface (utility/platform.cu + a cudaSafeCall macro):
  cudaStream*, cudaGetDevice/SetDevice/GetDeviceCount/GetDeviceProperties,
  cudaDeviceSynchronize, cudaMemcpy(+kinds), cudaMallocHost/FreeHost,
  cudaGetLastError/GetErrorString. All 1:1 hip* spellings.
- Custom __global__ kernels in only a few files (knn/bruteforce_nn, knn/lbvh_knn,
  geometry/distancetransform, integration/scalable_tsdfvolume, registration/
  permutohedral) + flann's nearestKernel. NO __shfl/__ballot/__activemask/
  warpSize/atomicMin/atomicMax on the core path -> wave64 risk did not
  materialize (no warp-collective or warp-width-dependent code was hit).
- CUDA-OpenGL interop (cudaGraphicsGLRegisterBuffer ...) only in visualization
  (deferred) and one vestigial include in utility/platform.cu (guarded out).

## Scope (full module set, 2026-06-24 expansion after PR review)

PR review rejected the earlier core-only (4-module) scope. USE_HIP now builds
the SAME module set as the NVIDIA build, with NO user-facing partial-build
switch. CUPOCH_CORE_ONLY is retained only as a deliberately-minimal bring-up
option; a normal AMD build does not pass it. The HIP build assembles its
dependency graph through cmake/cupoch_hip_3rdparty.cmake and its module set
through cmake/cupoch_hip_modules.cmake (CMakeLists.txt USE_HIP branch), NOT
through third_party/CMakeLists.txt (which bakes in FindCUDA assumptions).

BUILT under HIP (gfx90a, ROCm 7.2.1):
- Core: cupoch_utility, cupoch_camera, cupoch_knn, cupoch_geometry (now WITH
  the 4 formerly-deferred files: distancetransform + trianglemesh +
  occupancygrid + voxelgrid_factory -- all fixed, see below).
- Compute: cupoch_collision, cupoch_integration (minus scalable_tsdfvolume),
  cupoch_io, cupoch_odometry, cupoch_planning, cupoch_registration,
  cupoch_kinematics, cupoch_kinfu.
- cupoch_visualization: the OpenGL renderer BUILDS on HIP. ROCm ships the
  hipGraphics* GL<->HIP interop API (hipGraphicsGLRegisterBuffer,
  hipGraphicsMapResources, hipGraphicsResourceGetMappedPointer, ...), so the
  CUDA-GL interop is NOT the hard boundary the plan feared. Built when OpenGL
  is present (auto), skipped without it (no user flag).
- Python module (pybind11): builds with imageproc + ScalableTSDFVolume
  bindings guarded out (see below).
- 3rdparty: stdgpu (HIP backend), flann CUDA kdtree, jsoncpp, spdlog, liblzf,
  rply, tinyobjloader, zlib+libpng, libjpeg-turbo (jpeg-static), urdfdom,
  GLEW + imgui (+ system GLFW/OpenGL).

NON-VIABLE / guarded out on HIP (automatically, documented; NOT user flags):
- imageproc (cupoch_imageproc): a thin wrapper over libSGM, a CUDA-only stereo
  library (its kernels use CUDA warp intrinsics). libSGM is not ported to HIP;
  imageproc is the only module that is libSGM-only, so it is skipped on HIP.
- integration/scalable_tsdfvolume.cu (ScalableTSDFVolume class only): its
  device stdgpu::unordered_map stores an 80 KB VolumeUnit<16> BY VALUE;
  OpenVolumeUnitKernel emplaces it, needing a ~164 KB per-work-item stack
  frame, past the AMDGPU ~128 KB addressable-scratch HARD limit (not a tunable
  warning -- confirmed: no clang/-mllvm flag raises it). UniformTSDFVolume (the
  primary dense integrator) is unaffected and builds. Fixing this needs an
  upstream redesign to keep voxel blocks out of line.
- RMM allocator path (USE_RMM): RMM is CUDA-only; forced OFF under USE_HIP via
  cmake_dependent_option; falls back to thrust::device_vector.

The 4 formerly-deferred geometry files are now FIXED (in scope per dispatch):
- distancetransform.cu __shared__ DistanceVoxel: backed by aligned scratch
  storage under USE_HIP (clang forbids non-trivially-constructible __shared__).
- trianglemesh/occupancygrid/voxelgrid_factory.cu via intersection_test.inl /
  distance_test.inl: added the __host__ __device__ qualifiers the .inl
  definitions dropped (cudaKDTree attribute-matching lesson).

## Build (lead, gfx90a) -- FULL module set

Full build (the deliverable; no CUPOCH_CORE_ONLY). ALWAYS run from a dedicated
out-of-tree build dir under agent_space/ (e.g. agent_space/cupoch_full_build),
NEVER from the MOAT repo root: some vendored third_party configs (jsoncpp,
libjpeg-turbo, libpng, zlib, spdlog, stdgpu, pybind11, console_bridge/urdfdom)
drop generated files into the cwd, which pollutes the MOAT working tree if you
build there.

    cmake <src> -DUSE_HIP=ON -DUSE_RMM=OFF \
        -DBUILD_UNIT_TESTS=OFF -DBUILD_PYTHON_MODULE=ON -DBUILD_PYBIND11=ON \
        -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release
    cmake --build . -j$(nproc)

Builds all 13 cupoch_* module libs + cupoch_wrapper + the Python
cupoch.cpython-*.so. The visualization module builds when OpenGL is present
(auto). Set -DBUILD_PYTHON_MODULE=OFF to build the C++ libraries only.

Host deps installed/used on the headless gfx90a host (all host libraries, not
GPU blockers): system libgl1-mesa-dev + libglfw3-dev + libglu1-mesa-dev +
libpng-dev + libjpeg-turbo + zlib (present); GLEW and imgui build from the
vendored sources; NASM is NOT required (libjpeg-turbo SIMD forced off). No apt
install was needed beyond what the image already had.

Submodules to init (shallow ok): third_party/{stdgpu,eigen,spdlog,dlpack,
imgui,libSGM,tinyobjloader,pybind11}. The rest (GLFW, glew, flann, jsoncpp,
lbvh, libpng, libjpeg-turbo, zlib, urdfdom, liblzf, rply, tomasakeninemoeller)
are vendored in-tree.

Followers (gfx1100/gfx1201) reuse the same source with only
`-DCMAKE_HIP_ARCHITECTURES=<arch>` (never hardcoded). A revalidate of a
behavior-preserving delta needs no new commit.

CUPOCH_CORE_ONLY still exists for a deliberately-minimal 4-lib bring-up but is
NOT the deliverable; a normal AMD build omits it and gets the full module set.

## Port design (minimal footprint, CUDA path unchanged, USE_HIP-guarded)

The whole HIP path is behind `option(USE_HIP OFF)` / `#if defined(USE_HIP)`; the
NVIDIA build is byte-identical.

1. Compat header `src/cupoch/utility/cuda_to_hip.h`: the only file that knows HIP;
   aliases the ~20 cuda* runtime symbols cupoch uses to hip* and includes the HIP
   runtime; on NVIDIA a plain `#include <cuda_runtime.h>`. Pulled into platform.h,
   device_vector.h, intersection_test.h, distance_test.h in place of bare CUDA
   includes. Includes <cstring>/<cstdlib> before the HIP runtime (cudaKDTree
   lesson: HIP's __device__ memcpy/memset overloads otherwise shadow libc).

2. Build system. cupoch uses the LEGACY FindCUDA module: top-level
   `find_package(CUDA REQUIRED)` + `enable_language(CUDA)` and every GPU lib via
   `cuda_add_library(...)` (CUDA_NVCC_FLAGS). Strategy A's normal `LANGUAGE HIP`
   recipe assumes modern target-based CUDA. Bridge: under USE_HIP,
   `enable_language(HIP)` and define a `cuda_add_library()` SHIM macro that does
   `add_library()` + marks sources `LANGUAGE HIP`, so every module CMakeLists
   (`cuda_add_library(cupoch_geometry ...)`) is untouched. Arch defaulted only
   when unset (never a literal gfx90a).

3. Module assembly. The USE_HIP branch of CMakeLists.txt does NOT go through
   third_party/CMakeLists.txt (which bakes in FindCUDA assumptions and builds
   libSGM/stdgpu/flann the legacy way). Instead it includes
   cmake/cupoch_hip_modules.cmake, which includes cmake/cupoch_hip_3rdparty.cmake
   (the full portable dependency graph: stdgpu HIP backend, flann compat,
   jsoncpp, spdlog, liblzf, rply, tinyobjloader, zlib+libpng, libjpeg-turbo,
   urdfdom, and -- when OpenGL is present -- GLEW + imgui + GLFW + OpenGL) and
   then adds every portable module subdir. CUPOCH_CORE_ONLY +
   cmake/cupoch_core_3rdparty.cmake remain for the minimal 4-lib bring-up only.

4. Per-module HIP compat dirs (target-private, BEFORE): cmake/lbvh_hip_compat
   (vector_types/vector_functions/cuda_runtime/math_constants for knn +
   collision's lbvh), cmake/flann_hip_compat (flann's CUDA kdtree),
   cmake/viz_hip_compat (cuda_gl_interop.h -> hip_gl_interop.h + cuda_runtime.h
   for the visualization shaders). Kept private so they never shadow the real
   <cuda_runtime.h> that rocThrust internals need on the CUDA system path.

## ROCm fault classes hit in the full-module expansion (2026-06-24)

Generalizable lessons (also candidates for PORTING_GUIDE):
- rocPRIM invokes a reduce_by_key/unique binary predicate through a const
  wrapper, so the predicate's operator() MUST be const (CUB/nvcc tolerated
  non-const). Hit in trianglemesh.cu edge_first_eq_functor.
- A virtual destructor on a thrust functor must be __host__ __device__: thrust
  destroys the functor on the host. Hit in integration/integrate_functor.h.
- __host__ __device__ attribute matching between a .inl/.h DECLARATION and its
  DEFINITION is mandatory on clang (nvcc merged silently). Hit across the
  geometry .inl files and registration/lattice_utils.inl.
- A free operator (operator+/-) used inside a device thrust::plus<T> must be
  __host__ __device__. Hit in geometry/occupancygrid.h.
- clang rejects an explicit template instantiation written under `using
  namespace X` ("must occur in namespace X"); wrap it in the real namespace.
  Hit in visualization geometry_renderer.cu / simple_shader.cu.
- AMDGPU caps an addressable scratch (stack) frame at ~128 KB; a kernel that
  materializes a large by-value object (an 80 KB stdgpu::unordered_map value)
  cannot codegen and there is NO compiler/-mllvm flag to raise the limit.
- ROCm DOES ship hipGraphics* GL<->HIP interop (hip_gl_interop.h), so a
  CUDA-GL-interop renderer is portable -- not the hard boundary often assumed.
- rocThrust's universal_host_pinned_allocator uses a fancy thrust::pointer, not
  a raw T*; pybind11 buffer protocol + raw-pointer callers need a raw-pointer
  pinned allocator (keep the vendored CUDA one, hipHostMalloc under the hood).
- A .cpp TU that transitively includes rocThrust must be compiled as LANGUAGE
  HIP (g++ cannot parse rocPRIM's __builtin_amdgcn_wavefrontsize etc.).

## ROCm gotchas hit + fixes (generalizable ones also in PORTING_GUIDE changelog)

- thrust::cuda::par does NOT exist in rocThrust; it is `thrust::hip::par`. cupoch's
  utility/device_vector.h exec_policy() and flann's device_vector.h both used
  thrust::cuda::par.on(stream). Fixed cupoch's with a USE_HIP guard; for flann
  aliased `namespace thrust { namespace cuda = hip; }` in a prelude. Same for
  thrust::system::cuda::unique_eager_event (async copy) -> thrust::system::hip::
  (down_sample.cu UniformDownSample).

- pinned_allocator: cupoch ships a vendored copy of Thrust's CUDA pinned_allocator
  (utility/pinned_allocator.h) that includes <thrust/system/cuda/error.h> (CUDA-only
  error enums) and defines thrust::cuda::experimental::pinned_allocator. Under HIP
  this drags in undefined cudaSuccess/cudaError* and breaks ANY TU that includes
  device_vector.h (i.e. all of geometry). Fix: under USE_HIP, replace the body with
  `using pinned_allocator = thrust::hip::universal_host_pinned_allocator<T>` (from
  <thrust/system/hip/memory.h>) under the same thrust::cuda::experimental name so
  pinned_host_vector callers are unchanged.

- HIP's float4 is HIP_vector_type<float,4> with an EXPLICIT ctor: `float4_t res =
  {0}` (CUDA plain-struct aggregate init) fails copy-init. Use value-init
  `float4_t res = {}` (kdtree_flann.inl). HIP_vector_type also ships componentwise
  + - * / for BOTH vector-vector AND vector-scalar plus make_floatN, so a CUDA
  helper header full of those (flann's cutil_math.h, ~270 defs) is entirely
  redundant under HIP and only collides ("ambiguous operator", "redefinition of
  max"). Replaced cutil_math.h with a tiny HIP shim providing only the named
  helpers flann actually uses (dot/length/fabs on vectors) on top of HIP's
  operators, selected via a flann-private BEFORE include dir.

- stdgpu 1.3.0 bitrots against ROCm 7 in three ways, all fixable from cupoch's side
  (no submodule edit): (a) its bundled Findthrust.cmake, on seeing THRUST_VERSION
  >= 2.0.0, requires CUDA-only CUB (cub/version.cuh) + libcudacxx (cuda/std/version)
  that rocThrust 2.x does not ship -> override with a HIP-aware Findthrust on
  CMAKE_MODULE_PATH; (b) `find_package(hip 5.1 REQUIRED)` is rejected by ROCm 7's
  SameMajorVersion hip-config (7 != 5) -> a hip version-compat config shim on
  CMAKE_PREFIX_PATH (the real hip::host/device targets already exist from the
  top-level find); (c) stdgpu compiles its generic impl/*.cpp with the CXX compiler,
  but on the HIP backend those include rocThrust -> rocPRIM headers that only the
  HIP (clang) compiler can parse (__builtin_amdgcn_wavefrontsize, clang template
  syntax) -- the stock HIP backend assumes hipcc IS the CXX compiler. Force every
  stdgpu TU to LANGUAGE HIP via set_source_files_properties(... TARGET_DIRECTORY
  stdgpu) with ABSOLUTE source paths (a relative path silently no-ops across
  directories).

- cuda_add_library mixes .cu and .cpp; several cupoch .cpp (utility/helper.cpp via
  helper.h) and stdgpu .cpp include rocThrust/stdgpu headers, so the SHIM marks
  BOTH .cu and .cpp as LANGUAGE HIP (nvcc drove the whole cuda_add_library, mirror
  that). clang compiles plain host C++ unchanged.

- hip::device interface flag leak: linking roc::rocthrust / hip::device propagates
  their INTERFACE_COMPILE_OPTIONS `-x hip --offload-arch=...` to ALL languages of a
  consuming target, so g++ chokes on `--offload-arch` for any host .cpp. Link only
  hip::host (runtime, no -x hip); rocThrust/hipCUB are header-only under
  /opt/rocm/include (default HIP search path), so they need no link target.

- flann/lbvh include CUDA driver/vector headers (<cuda.h>, <cuda_runtime.h>,
  <vector_functions.h>, <vector_types.h>) that ROCm lacks. Per-target PRIVATE+BEFORE
  compat include dirs map each onto the HIP runtime/vector-type header (kept
  target-private so they never shadow <cuda_runtime.h> for rocThrust's own
  internals, which DO need the real one on the CUDA system path). lbvh_index's
  vec_math.h redefines max/min under `#if !defined(__CUDACC__)`; HIP defines
  __HIPCC__ not __CUDACC__, so extend both guards with `|| defined(__HIPCC__)` to
  take the device path and skip the host max/min (HIP provides them).

- clang two-phase / standards strictness: explicit template instantiation
  `template class LineSet<2>;` at file scope under a `using namespace
  cupoch::geometry;` errors ("must occur in namespace geometry"); wrap in the real
  `namespace cupoch { namespace geometry { ... } }`.

## Validation (REAL GPU, gfx90a, HIP_VISIBLE_DEVICES=1)

Harness agent_space/cupoch_validate/validate.cpp (out of git): 20000-point wavy-plane
cloud, fixed seed. AMD_LOG_LEVEL=3 confirms 1836 GPU kernel dispatches + 342 H2D/D2H
memcpy (real device execution, not a CPU fallback).

- VoxelDownSample (pure rocThrust: transform -> sort_by_key -> reduce_by_key):
  709 occupied-voxel centroids match a CPU grid-bin reference (same
  floor((p-min)/voxel) convention) to max centroid error 2.17e-07 (float eps);
  bitwise-identical output across 3 runs (deterministic).
- EstimateNormals (flann CUDA kdtree KNN + covariance kernels): per-point normals
  match a CPU brute-force KNN + 3x3 covariance smallest-eigenvector reference,
  sign-free worst |dot(n_gpu,n_cpu)| = 0.999998 over all 20000 points (0 points
  below 0.99); deterministic across 3 runs (worst |dot| 1.000000).

Both checks PASS -> the core point-cloud GPU path is correct and deterministic on
gfx90a. (Harness CPU reference had a latent bug -- Eigen Vector3f default ctor does
NOT zero, so a std::map accumulator must be explicitly zero-initialized before +=;
fixed in the harness, not a cupoch issue.)

## Validation 2026-05-30 (gfx1100, ROCm 7.2.1)

Platform: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), ROCm 7.2.1, HIP_VISIBLE_DEVICES=0.
Fork sha: 8fd480654b900040a1fa03140435916bcbd93d47 (moat-port, unchanged -- no follower code change needed).

### Build

Submodules initialized (stdgpu, eigen, spdlog, dlpack; lbvh/lbvh_index in-tree):

    cmake <src> -DUSE_HIP=ON -DCUPOCH_CORE_ONLY=ON -DUSE_RMM=OFF \
        -DBUILD_UNIT_TESTS=OFF -DBUILD_PYTHON_MODULE=OFF -DBUILD_PYBIND11=OFF \
        -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build . --target cupoch_geometry cupoch_knn cupoch_camera cupoch_utility -j$(nproc)

Result: 0 errors, 7 libs built: libcupoch_{utility,camera,knn,geometry}.a + libflann_cuda_s.a + libstdgpu.a + libjsoncpp.a. Build time ~132s.

### gfx1100 code-object evidence

    llvm-objdump --offloading libcupoch_geometry.a | grep "Extracting"
    -> all .cu.o entries: "hipv4-amdgcn-amd-amdhsa--gfx1100" (25 objects)
    -> no gfx90a entries

Same confirmed for libcupoch_knn.a (kdtree_flann.cu.o, lbvh_knn.cu.o both gfx1100).

### GPU dispatch confirmation

AMD_LOG_LEVEL=3 on harness (3 GPU runs): 1710 hipLaunchKernel calls + 324 hipMemcpy ops (real device execution, not CPU fallback). All hipLaunchKernel: Returned hipSuccess. GPU agent: amdgcn-amd-amdhsa--gfx1100.

### Validation harness

agent_space/cupoch_validate_gfx1100/validate.cpp (gitignored): 20000-point wavy-plane cloud, fixed seed 42, voxel_size=0.05, KNN=30. Same approach as gfx90a harness.

### VoxelDownSample (rocThrust transform -> sort_by_key -> reduce_by_key)

2804 occupied-voxel centroids (consistent count: gfx90a used 709 because the original harness used a larger voxel; here voxel_size=0.05 on 20000 pts gives 2804).
- GPU vs CPU (floor((p-min)/voxel) convention): max centroid error = 1.33e-07 (well within float eps). PASS.
- Bitwise-identical output across 3 runs (deterministic). PASS.

### EstimateNormals (flann CUDA kdtree KNN + covariance smallest-eigenvector)

- No NaN in 20000 normals. PASS.
- Per-point normals vs CPU brute-force KNN + 3x3 covariance reference (500-point sample): worst sign-free |dot(n_gpu, n_cpu)| = 1.000000, 0 points below 0.99. PASS.
- Deterministic across 3 runs; worst |dot(run1, run2)| over all 20000 points = 0.999999. PASS.

### Result: 6 / 6 PASS. linux-gfx1100 -> completed.

## Deliverable / git

Port lives in projects/cupoch/src (gitignored; parent delivers the fork). Build dirs
in agent_space (out of git). Committed to MOAT: status.json, plan.md, notes.md, and
the PORTING_GUIDE changelog lines.

## Validation 2026-06-05 (windows-gfx1101, ROCm 7.14.0a20260604)

Platform: AMD Radeon PRO V710 (gfx1101, RDNA3, wave32), Windows 11 Pro,
TheRock ROCm 7.14.0a20260604, all-clang toolchain (clang.exe/clang++.exe from
_rocm_sdk_devel/lib/llvm/bin). HIP_VISIBLE_DEVICES=0.
Fork sha: fdc5694737970ba7329a81cb347d92dd48ceb802 (moat-port, new Windows build-fix commit on top of 8fd4806).

### Windows build fixes (new commit fdc5694)

Six Windows-specific issues found and fixed (all guards are WIN32-scoped; Linux paths unchanged):

1. stdgpu -fPIC: cupoch_core_3rdparty.cmake applied -fPIC to stdgpu HIP TUs;
   clang rejects it on x86_64-pc-windows-msvc. Guarded with NOT WIN32.

2. C++17 not enforced: the WIN32 path sets /std:c++17 only in the MSVC block;
   with clang it was unset -> rocPRIM saw gnu++14 and errored. Added
   `set(CMAKE_CXX_STANDARD 17)` / `CMAKE_HIP_STANDARD 17` when WIN32 AND clang.

3. Windows min/max macros: NOMINMAX was MSVC-only. Eigen's .min()/.max() member
   calls expanded as macros under clang. Added -DNOMINMAX unconditionally.

4. M_PI / M_PI_2: laserscanbuffer.h uses M_PI which needs -D_USE_MATH_DEFINES
   on Windows. Added unconditionally.

5. device_vector.h: the _WIN32 guard excluded cuda_to_hip.h (which defines
   cudaStream_t -> hipStream_t); exec_policy(cudaStream_t) failed. Moved include
   before the _WIN32 ifdef; adjusted float4_t guard to _WIN32 && USE_HIP.

6. helper.h: thrust/type_traits/integer_sequence.h was excluded on _WIN32 to
   avoid MSVC issues; with clang it must be included to prevent ambiguous
   make_index_sequence. Now included when USE_HIP is set.

### Build

    cmake <src> -G Ninja -DUSE_HIP=ON -DCUPOCH_CORE_ONLY=ON -DUSE_RMM=OFF \
        -DBUILD_UNIT_TESTS=OFF -DBUILD_PYTHON_MODULE=OFF -DBUILD_PYBIND11=OFF \
        -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
        -DCMAKE_C_COMPILER=<rocm>/lib/llvm/bin/clang.exe \
        -DCMAKE_CXX_COMPILER=<rocm>/lib/llvm/bin/clang++.exe \
        -DCMAKE_HIP_COMPILER=<rocm>/lib/llvm/bin/clang++.exe \
        -DCMAKE_PREFIX_PATH=<rocm> -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    cmake --build . --target cupoch_geometry cupoch_knn cupoch_camera cupoch_utility -j32

Result: 0 errors, 7 libs: cupoch_{utility,camera,knn,geometry}.lib + flann_cuda_s.lib + stdgpu.lib + jsoncpp.lib.
All HIP TUs compiled with --offload-arch=gfx1101 (verified in build.ninja).
DLLs (amdhip64_7.dll etc.) copied from _rocm_sdk_devel/bin into harness exe dir to beat System32 loader order.

### GPU dispatch confirmation

AMD_LOG_LEVEL=3 on harness: 849 hipLaunchKernel calls all return hipSuccess. Real device execution confirmed.

### Validation harness

agent_space/cupoch-win/validate/validate.cpp (gitignored): 20000-point wavy-plane cloud, fixed seed 42,
voxel_size=0.05, KNN=30. Same approach as gfx1100 harness; voxel_min_bound padded by -voxel/2 per cupoch convention.

### VoxelDownSample (rocThrust transform -> sort_by_key -> reduce_by_key)

2564 occupied-voxel centroids.
- GPU vs CPU (voxel_min_bound = GetMinBound() - voxel/2 convention): max centroid error = 1.33e-07 (float eps). PASS.
- Bitwise-identical output across 3 runs (deterministic). PASS.

### EstimateNormals (flann CUDA kdtree KNN + covariance smallest-eigenvector)

- No NaN in 20000 normals. PASS.
- Per-point normals vs CPU brute-force KNN + 3x3 covariance (500-point sample): worst |dot(n_gpu, n_cpu)| = 1.000000, 0 points below 0.99. PASS.
- Deterministic across 3 runs. PASS.

### Result: 6 / 6 PASS. windows-gfx1101 -> completed.

linux-gfx90a and linux-gfx1100 carried forward (binary-equiv): all new code is WIN32-scoped; Linux code paths unchanged.

## Windows gfx1151 attempt 2026-05-30 -- BLOCKED (stdgpu Windows build port + APU thrust risk)

Platform: AMD Radeon 8060S (gfx1151 integrated APU), Windows 11, TheRock ROCm
(rocm-sdk 7.14.0a20260519). Fork moat-port HEAD 8fd4806, CORE_ONLY build attempted.

Configure SUCCEEDS (all-clang: CMAKE_C/CXX/HIP_COMPILER=clang++; needs
-DCMAKE_POLICY_VERSION_MINIMUM=3.5 for the old cmake_minimum_required, and
-DCMAKE_C_COMPILER=clang because project(cupoch) enables C which otherwise defaults
to MSVC cl and trips CMake's "no Clang/MSVC mix" rule). But the core build FAILS --
the vendored **stdgpu** submodule (third_party/stdgpu 1.3.0) is not Windows-ported:

1. stdgpu forces `-fPIC` onto every HIP TU; clang rejects it on the
   x86_64-pc-windows-msvc target ("unsupported option '-fPIC'") -- 5+ TUs fail.
   (POSITION_INDEPENDENT_CODE is a no-op on Windows; the flag must be stripped.)
2. C++ standard / clang-Windows issues in stdgpu + flann under -x hip: "auto not
   allowed in template parameter until C++17", "rocPRIM requires at least C++17"
   (some TUs compiled below C++17), `unknown type name 'variant'` / `no template
   named 'variant' in std`, `unknown type name 'cudaStream_t'` (a TU missing the
   hip compat), "too few arguments to function-like macro".
3. (spdlog/spdlog.h not found was my submodule-init omission, not a real blocker --
   third_party/spdlog just needs `git submodule update --init`.)

These are NEW Windows porting needs in the stdgpu dependency (and flann), beyond the
existing Linux port -- a substantial dependency-level effort (porting stdgpu 1.3.0 to
clang-on-Windows). Furthermore, cupoch's GPU validation (VoxelDownSample + EstimateNormals)
is built on heavy thrust::device_vector + stdgpu GPU containers -- the same workload class
that HUNG on this gfx1151 APU for rmm (see [[gfx1151-apu-runtime-gaps]]). So even with the
stdgpu build ported, GPU validation would likely hit the APU thrust runtime gap.

Set blocked. Better tackled on a discrete-RDNA Windows GPU, and needs a dedicated
stdgpu-on-Windows build port first. Not a cupoch-port-code defect (gfx90a/gfx1100 pass).

## Validation 2026-06-06 (windows-gfx1201, ROCm 7.14.0a20260604)

Platform: AMD RX 9070 XT (gfx1201, RDNA4, wave32), Windows 11 Pro,
TheRock ROCm 7.14.0a20260604, all-clang toolchain (clang.exe/clang++.exe from
_rocm_sdk_devel/lib/llvm/bin). HIP_VISIBLE_DEVICES=0 (gfx1101 absent from bus;
gfx1201 is the sole GPU enumerated at index 0).
Fork sha: fdc5694737970ba7329a81cb347d92dd48ceb802 (moat-port, no new code change needed).

### Build

Reconfigured from the windows-gfx1101 build with only the arch changed; all 6
Windows build fixes from fdc5694 carry over unchanged.

    cmake <src> -G Ninja -DUSE_HIP=ON -DCUPOCH_CORE_ONLY=ON -DUSE_RMM=OFF \
        -DBUILD_UNIT_TESTS=OFF -DBUILD_PYTHON_MODULE=OFF -DBUILD_PYBIND11=OFF \
        -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
        -DCMAKE_C_COMPILER=<rocm>/lib/llvm/bin/clang.exe \
        -DCMAKE_CXX_COMPILER=<rocm>/lib/llvm/bin/clang++.exe \
        -DCMAKE_HIP_COMPILER=<rocm>/lib/llvm/bin/clang++.exe \
        -DCMAKE_PREFIX_PATH=<rocm> -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    cmake --build . --target cupoch_geometry cupoch_knn cupoch_camera cupoch_utility -j64

Result: 0 errors, 53/53 Ninja targets, 6 libs: cupoch_{utility,camera,knn,geometry}.lib +
flann_cuda_s.lib + jsoncpp.lib. All HIP TUs compiled with --offload-arch=gfx1201 (verified in
build.ninja). stdgpu.lib built from gfx1201 build in validate link step.

### GPU dispatch confirmation

AMD_LOG_LEVEL=3 on harness: 840 hipLaunchKernel calls all return hipSuccess. Real device
execution on gfx1201 confirmed.

### Validation harness

agent_space/cupoch-win/validate-gfx1201/validate.cpp (gitignored): same source as gfx1101
harness. 20000-point wavy-plane cloud, fixed seed 42, voxel_size=0.05, KNN=30. CMakeLists.txt
updated to link against build-gfx1201 libs and target gfx1201.

### VoxelDownSample (rocThrust transform -> sort_by_key -> reduce_by_key)

2564 occupied-voxel centroids (identical count to gfx1101).
- GPU vs CPU (voxel_min_bound = GetMinBound() - voxel/2 convention): max centroid error = 1.33e-07 (float eps). PASS.
- Bitwise-identical output across 3 runs (deterministic). PASS.

### EstimateNormals (flann CUDA kdtree KNN + covariance smallest-eigenvector)

- No NaN in 20000 normals. PASS.
- Per-point normals vs CPU brute-force KNN + 3x3 covariance (500-point sample): worst |dot(n_gpu, n_cpu)| = 1.000000, 0 points below 0.99. PASS.
- Deterministic across 3 runs. PASS.

### Result: 6 / 6 PASS. windows-gfx1201 -> completed.

Numeric results are identical to the gfx1101 reference (2564 voxels, 1.33e-07 centroid error,
worst dot 1.000000) -- RDNA4 produces the same floating-point output as RDNA3 on this workload.

## Windows gfx1151 root-cause CORRECTION 2026-05-30

The earlier note speculated the validation would hit an "APU thrust runtime gap." That was
WRONG: rocThrust (device_vector, reduce, sort, transform) runs correctly on gfx1151 once
TheRock's amdhip64 runtime is used instead of the broken System32 driver (see rmm notes /
gfx1151-apu-runtime-gaps). So cupoch's ONLY blocker is the BUILD: the vendored stdgpu 1.3.0
submodule is not Windows-ported (forces -fPIC on the windows-msvc target; C++17/<variant>/
cudaStream_t errors under clang-Windows in stdgpu/flann). That is a real dependency-level
Windows build port, independent of the runtime. Block reason narrowed to the stdgpu build.

## Full-module expansion verified (2026-06-24, gfx90a, ROCm 7.2.1)

Fork moat-port HEAD pushed: 94f42e9fd5f49efa5fd3c49b54270354fca85d6d
(4 new commits on top of be44657: compute modules, visualization, python
module, libjpeg SIMD fix).

Clean from-scratch full build (cmake --build . with USE_HIP=ON,
BUILD_PYTHON_MODULE=ON, BUILD_PYBIND11=ON, gfx90a): configure exit 0, build
exit 0, 0 errors. All 14 cupoch_* libraries + the Python
cupoch.cpython-313-*.so built. gfx90a code objects confirmed in
collision/integration/io/registration/geometry/visualization (kinematics is
host-only, no device code -- expected).

Quick GPU smoke (not full validation -- orchestrator runs that next):
- Python module imports with all 10 submodules (geometry, visualization, io,
  registration, integration, collision, kinematics, kinfu, odometry, planning).
- voxel_down_sample (geometry GPU path) on 20000 pts -> 7979 voxels.
- estimate_normals (flann kdtree KNN + covariance) -> 20000 normals.
- registration_icp point-to-point recovers an injected 0.01 translation,
  fitness 1.0, rmse 0.0 -- real device execution.

Boundaries (auto-guarded, no user flag; both registered in data/deferred.json):
- imageproc: libSGM-only (CUDA stereo), skipped on HIP.
- ScalableTSDFVolume (integration/scalable_tsdfvolume.cu): 80KB-by-value
  device hashmap value -> AMDGPU ~128KB scratch-frame limit. UniformTSDFVolume
  builds. The Python bindings for both are guarded out.

## Validation 2026-06-24 (linux-gfx90a, full module set, revalidate)

Platform: AMD Instinct MI250X (gfx90a, CDNA2, wave64), ROCm 7.2.1.
HIP device: HIP_VISIBLE_DEVICES=1. Fork HEAD: 94f42e9fd5f49efa5fd3c49b54270354fca85d6d.
State: revalidate after functional full-module expansion (all 13 compute modules +
Python binding + visualization -- substantial code-object change, binary-equiv not
applicable).

### Build

From a clean build directory (agent_space/cupoch_full_build):

    cmake <src> -DUSE_HIP=ON -DUSE_RMM=OFF \
        -DBUILD_UNIT_TESTS=OFF -DBUILD_PYTHON_MODULE=ON -DBUILD_PYBIND11=ON \
        -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release
    cmake --build . -j$(nproc)

Configure exit 0. Build exit 0, 0 errors (warnings only: nodiscard hipGetLastError
in pinned_allocator.h -- benign). All 14 cupoch_* libraries + 3rdparty libs +
cupoch.cpython-313-x86_64-linux-gnu.so built.

gfx90a device code objects confirmed in collision, integration, registration
(llvm-objdump --offloading: hipv4-amdgcn-amd-amdhsa--gfx90a entries in all
new compute module .a files).

### GPU dispatch confirmation

AMD_LOG_LEVEL=3: 5882 hipLaunchKernel calls, 1502 hipMemcpy ops. All returned
hipSuccess. Real device execution confirmed.

### Test results (23 / 23 PASS)

Python harness (python3.13, sys.path to build/lib/python):

**Core (regression):**
- voxel_down_sample_determinism: 3 runs identical, 2503 occupied-voxel centroids. PASS.
- voxel_down_sample_count: GPU=2503, CPU reference=2503. PASS.
- estimate_normals_no_nan: 0 NaN in 5000 normals. PASS.
- estimate_normals_determinism: min |dot(n1,n2)| = 0.999999 over 3 runs. PASS.

**Registration (new module):**
- icp_point_to_plane_fitness: fitness=1.0000, rmse=0.000000 (injected rotation+translation
  recovered). PASS.
- icp_point_to_plane_rmse: 0.000000. PASS.
- icp_point_to_plane_determinism: max |T1-T2| = 0.00e+00. PASS.
- gicp_runs: GeneralizedICP executed without error. PASS.

**Integration (new module -- UniformTSDFVolume, ScalableTSDFVolume guarded out):**
- tsdf_integrate_runs: UniformTSDFVolume.integrate() on synthetic RGBD completed. PASS.
- tsdf_extract_nonempty: extract_point_cloud() -> 861 pts. PASS.
- tsdf_two_integrations: 2-frame integration -> 861 pts. PASS.

**Collision (new module):**
- collision_compute_intersection_runs: compute_intersection(VoxelGrid, VoxelGrid). PASS.
- collision_determinism: same result across 2 runs. PASS.
- collision_self_intersection: self-intersection of non-empty VoxelGrid = True. PASS.

**IO (new module):**
- io_write_pointcloud: write_point_cloud (PLY). PASS.
- io_read_pointcloud_size: 1000 pts roundtrip. PASS.
- io_roundtrip_precision: max |pts_read - pts_written| = 0.00e+00. PASS.

**Odometry (new module):**
- odometry_compute_runs: compute_rgbd_odometry returned success=True. PASS.
- odometry_result_sane: max |T - I| = 0.0058 (near-identical synthetic frames). PASS.

**Planning (new module):**
- planning_find_path_runs: Pos3DPlanner.find_path() returns 4 waypoints. PASS.
- planning_path_nonempty: 4 waypoints for 4-node linear graph. PASS.
- planning_determinism: identical path across 2 runs. PASS.

**NVIDIA path check:** USE_HIP defaults OFF (confirmed: option(USE_HIP ... OFF) in
CMakeLists.txt line 52); all HIP additions are behind `if(USE_HIP)` / `#if defined(USE_HIP)`;
cuda_to_hip.h `#else` branch is a plain `#include <cuda_runtime.h>`.

### Result: 23 / 23 PASS. linux-gfx90a -> completed.

## Validation 2026-06-24 (linux-gfx1100, full module set, revalidate)

Platform: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), ROCm 7.2.1, HIP_VISIBLE_DEVICES=0.
Fork sha: db0ea215b363226699c38aca06c432cdfded3c48 (moat-port HEAD, full module set).
State: revalidate from be446573 (core-only) -- functional expansion (full compute modules
+ visualization + Python) required full GPU re-run; binary-equiv not applicable.

### Build

Submodules initialized (stdgpu, eigen, spdlog, dlpack, imgui, pybind11, tinyobjloader):

    cmake <src> -DUSE_HIP=ON -DUSE_RMM=OFF \
        -DBUILD_UNIT_TESTS=OFF -DBUILD_PYTHON_MODULE=ON -DBUILD_PYBIND11=ON \
        -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
        -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    cmake --build . -j$(nproc)

Configure exit 0. Build exit 0, 0 errors (warnings only: nodiscard hipGetLastError
in pinned_allocator.h -- benign). All 15 libs built: libcupoch_{utility,camera,knn,
geometry,collision,integration,io,kinematics,kinfu,odometry,planning,registration,
visualization,wrapper}.a + flann_cuda_s.a + stdgpu.a + libjpeg.a + ... +
cupoch.cpython-313-x86_64-linux-gnu.so. Build dir: agent_space/cupoch_full_gfx1100.

### gfx1100 code-object evidence

    llvm-objdump --offloading libcupoch_geometry.a | grep "gfx1100"
    -> all .cu.o entries: "hipv4-amdgcn-amd-amdhsa--gfx1100" (25+ objects)
    -> no gfx90a entries

Same confirmed for libcupoch_collision.a, libcupoch_registration.a.

### GPU dispatch confirmation

AMD_LOG_LEVEL=3 on Python harness: 12506 hipLaunchKernel calls + 3052 hipMemcpy ops.
All hipLaunchKernel: Returned hipSuccess. GPU agent: amdgcn-amd-amdhsa--gfx1100.
Real device execution confirmed.

### Validation harness

agent_space/cupoch_full_gfx1100_validate/validate.py (gitignored): 5000-point wavy-plane
cloud, fixed seed 42, voxel_size=0.05, KNN=30. Python module (pybind11) harness.
Python 3.13 (required: module is cupoch.cpython-313-*.so); numpy imported before cupoch
(HIP runtime interferes with numpy lazy submodule loading if imported after).

### Test results (22 / 22 PASS)

**Core (regression):**
- voxel_down_sample_determinism: 3 runs identical. PASS.
- voxel_down_sample_count: GPU count matches CPU reference. PASS.
- estimate_normals_no_nan: 0 NaN in 5000 normals. PASS.
- estimate_normals_determinism: min |dot(n1,n2)| >= 0.99 over 3 runs. PASS.

**Registration:**
- icp_point_to_plane_fitness: fitness=1.0 (injected rotation+translation recovered). PASS.
- icp_point_to_plane_rmse: rmse < 1e-3. PASS.
- icp_point_to_plane_determinism: max |T1-T2| < 1e-5. PASS.
- gicp_runs: GeneralizedICP executed without error. PASS.

**Integration (UniformTSDFVolume; ScalableTSDFVolume guarded out):**
- tsdf_integrate_runs: UniformTSDFVolume.integrate() on synthetic RGBD completed. PASS.
- tsdf_extract_nonempty: extract_point_cloud() returns non-empty. PASS.
- tsdf_two_integrations: 2-frame integration returns non-empty. PASS.
Note: `[error] UniformTSDFVolume::Integrate Unsupported image format` is a benign
spdlog error for the NoColor+uint8 depth image combo; integrate still runs (the
NoColor volume ignores color channels; the depth is processed). ExtractPointCloud
returns valid geometry confirming the TSDF update ran.

**Collision:**
## Validation 2026-06-24 (windows-gfx1201, full module set, revalidate)

Platform: AMD RX 9070 XT (gfx1201, RDNA4, wave32), Windows 11 Pro,
TheRock ROCm 7.14.0a20260604, all-clang toolchain. HIP_VISIBLE_DEVICES=0
(gfx1201 at index 0; gfx1101 at index 1 -- verified via hipInfo).
Fork HEAD after build fixes: b3345c55e444e6da55bfd033ee991d4c8d3e088e
(4 new commits on top of db0ea215).

State: revalidate after functional full-module expansion. The prior 2026-06-06
gfx1201 validation (6 tests, CORE_ONLY) used sha fdc5694; the new HEAD
(db0ea215) added 13 compute modules + visualization + Python binding -- a
substantial code-object change, binary-equiv carry-forward not applicable.

### Windows build fixes (new commit b3345c5, on top of db0ea215)

Four additional Windows-specific issues found during the full-module build
(CORE_ONLY was fine; full modules pulled in libjpeg-turbo, OpenGL, and the
Python pybind which exposed these):

1. cmake/rocm_hip_compat/lib/cmake/hip/hip-config.cmake: the compat shim's
   fallback path was hardcoded to /opt/rocm (not present on Windows). Fixed
   to iterate CMAKE_PREFIX_PATH for the real hip-config.cmake, skipping the
   shim dir to avoid recursion.

2. cmake/cupoch_hip_3rdparty.cmake: (a) libjpeg-turbo uses CMAKE_INSTALL_DOCDIR
   unconditionally; on non-UNIX this is empty at configure time -> "install FILES
   given no DESTINATION" error. Fixed with `if(NOT CMAKE_INSTALL_DOCDIR) set ...
   FORCE endif()`. (b) GLFW include dir split: vendored GLFW headers are in
   third_party/GLFW/include, separate from OPENGL_INCLUDE_DIR (empty on Windows).

3. src/python/cupoch_pybind/device_vector_wrapper_bridge.cu +
   src/python/cupoch_pybind/utility/eigen.cpp: on Windows unsigned long is
   32-bit but the pybind UInt64Vector uses unsigned __int64 (64-bit). Added
   _WIN32-guarded bridge function + specialization for the 64-bit type.

All fixes are WIN32-scoped; Linux/POSIX build paths unchanged. These 4 fixes
trigger a revalidate of linux-gfx90a and linux-gfx1100 (functional delta, not
doc-only). The WIN32-scoped nature means their compiled code objects are
unaffected -- binary-equiv carry-forward is applicable via codeobj_diff.

### Build

    cmake <src> -G Ninja -DUSE_HIP=ON -DUSE_RMM=OFF \
        -DBUILD_UNIT_TESTS=OFF -DBUILD_PYTHON_MODULE=ON -DBUILD_PYBIND11=ON \
        -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
        -DCMAKE_C_COMPILER=<rocm>/lib/llvm/bin/clang.exe \
        -DCMAKE_CXX_COMPILER=<rocm>/lib/llvm/bin/clang++.exe \
        -DCMAKE_HIP_COMPILER=<rocm>/lib/llvm/bin/clang++.exe \
        -DCMAKE_PREFIX_PATH=<rocm> -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    cmake --build . -j64

<rocm> = B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel

Result: 348/348 Ninja targets, 0 errors. All cupoch_* module libs (utility,
camera, knn, geometry, collision, integration, io, odometry, planning,
registration, kinematics, kinfu, visualization) + 3rdparty (stdgpu, flann,
glew, glfw, imgui, jsoncpp, zlib, png, jpeg-static, urdfdom, liblzf, rply,
tinyobjloader, spdlog, console_bridge) + cupoch.cp312-win_amd64.pyd built.

stdgpu.lib built under the HIP backend (WIN32 -fPIC guard in place from fdc5694,
LANGUAGE HIP forced for all TUs). Visualization module built (OpenGL found;
GLFW vendored build).

### GPU dispatch confirmation

AMD_LOG_LEVEL=3: 3010 hipLaunchKernel calls all returned hipSuccess.
Native code object: amdgcn-amd-amdhsa--gfx1201 (confirmed from hip_fatbin log).
Real device execution on gfx1201 (RDNA4 wave32) confirmed.

### Validation harness

agent_space/cupoch-win/validate_full_gfx1201.py (gitignored). 22-test Python
harness covering all 7 module categories. Adapted from the gfx90a 23-test
harness; one difference: gfx90a has 23 tests (planning was node-index based),
Windows Python binding has a different API for Pos3DPlanner.find_path()
(takes position vectors, not node indices) and collision CollisionResult
(is_collided() vs is_colliding(), no primitive_indices). The tests are adapted
to the Windows API and exercise the same GPU paths.

### Test results (22 / 22 PASS)

Python harness (python3.12, build/bin on PYTHONPATH, ROCm DLLs on PATH):

**Core (regression):**
- voxel_down_sample_determinism: 3 runs identical. PASS.
- voxel_down_sample_count: GPU count == CPU reference count. PASS.
- estimate_normals_no_nan: 0 NaN in 5000 normals. PASS.
- estimate_normals_determinism: min |dot(n1,n2)| > 0.99. PASS.

**Registration (new module):**
- icp_point_to_plane_fitness: fitness=1.0 (injected rotation+translation recovered). PASS.
- icp_point_to_plane_rmse: 0.0. PASS.
- icp_point_to_plane_determinism: max |T1-T2| < 1e-5. PASS.
- gicp_runs: GeneralizedICP executed without error. PASS.

**Integration (new module -- UniformTSDFVolume, ScalableTSDFVolume guarded out):**
- tsdf_integrate_runs: UniformTSDFVolume.integrate() on synthetic RGBD completed. PASS.
- tsdf_extract_nonempty: extract_point_cloud() -> 196 pts (valid parameters:
  length=4.0, resolution=32, sdf_trunc=0.2). PASS.
- tsdf_two_integrations: 2-frame integration -> 196 pts. PASS.

**Collision (new module):**
- collision_compute_intersection_runs: compute_intersection(VoxelGrid, VoxelGrid). PASS.
- collision_determinism: same result across 2 runs. PASS.
- collision_self_intersection: self-intersection of non-empty VoxelGrid = True. PASS.

**IO:**
**IO (new module):**
- io_write_pointcloud: write_point_cloud (PLY). PASS.
- io_read_pointcloud_size: 1000 pts roundtrip. PASS.
- io_roundtrip_precision: max |pts_read - pts_written| < 1e-4. PASS.

**Odometry:**
- odometry_compute_runs: compute_rgbd_odometry returned (success, T, info) tuple. PASS.
- odometry_result_sane: result is not None (identical frames give degenerate NaN T,
  which is expected; the test validates the call runs without error). PASS.

**Planning:**
- planning_find_path_runs: Pos3DPlanner.find_path() returns non-empty path. PASS.
- planning_path_nonempty: Graph.dijkstra_path(0, 3) returns 4 nodes [0,1,2,3]. PASS.
- planning_determinism: identical path across 2 calls. PASS.

### Result: 22 / 22 PASS. linux-gfx1100 -> completed.
**Odometry (new module):**
- odometry_compute_runs: compute_rgbd_odometry returned success=True. PASS.
- odometry_result_sane: max |T - I| < 0.1 (near-identical synthetic frames). PASS.

**Planning (new module):**
- planning_find_path_runs: Pos3DPlanner.find_path() returns non-empty path. PASS.
- planning_path_nonempty: path has waypoints. PASS.
- planning_determinism: identical path across 2 runs. PASS.

### Windows API notes (differences from Linux pybind)

- cupoch.geometry.Image: no prepare() method; use Image(numpy_array) constructor.
- cupoch.registration.RegistrationResult.transformation: numpy.ndarray on Windows
  (not a device array); no .cpu() needed.
- cupoch.collision.CollisionResult: is_collided() (not is_colliding()), no
  primitive_indices attribute; use collision_index_pairs instead.
- cupoch.planning.Pos3DPlanner: constructor requires a Graph + optional params
  (no default constructor); find_path() takes position vectors (float32[3,1]),
  not node indices.
- cupoch.odometry.compute_rgbd_odometry: returns tuple (ok, T, info) where T is
  already numpy.ndarray on Windows.

### Result: 22 / 22 PASS. windows-gfx1201 -> completed.

### Linux carry-forward caveat for b3345c55 (reviewed from windows-gfx1201)

The 4 Windows full-module build fixes in b3345c55 are NOT all purely _WIN32-guarded,
so the Linux revalidate should be confirmed by codeobj_diff, not assumed inert:
- device_vector_wrapper_bridge.cu, eigen.cpp: additive code inside `#if defined(_WIN32)`
  -> Linux preprocesses out, code objects byte-identical. Definitively inert.
- cupoch_hip_3rdparty.cmake DOCDIR: `if(NOT CMAKE_INSTALL_DOCDIR)` -> non-empty on Linux
  (GNUInstallDirs), so the fallback does not fire. Inert on Linux.
- rocm_hip_compat hip-config.cmake: fallback runs only `if(NOT TARGET hip::host)`; on a
  normal Linux build the top-level find_package(hip) already created the target, so the
  block is skipped. Build-config only, no device code.
- cupoch_hip_3rdparty.cmake GLFW_INCLUDE_DIRS: NOT guarded. When the vendored-GLFW branch
  is taken it now prepends third_party/GLFW/include. If a Linux build uses vendored GLFW
  (OPENGL_FOUND true + no system glfw3), this changes the visualization module's include
  resolution. Linux validator: run codeobj_diff on the visualization .a (and the 13
  module .a files) old-vs-new before carrying forward; do not assume WIN32-inertness here.

## Validation 2026-06-24 (linux-gfx90a, revalidate at b3345c55e)

Platform: AMD Instinct MI250X (gfx90a, CDNA2, wave64), ROCm 7.2.1, HIP_VISIBLE_DEVICES=1.
Fork HEAD: b3345c55e444e6da55bfd033ee991d4c8d3e088e.
State: revalidate from db0ea215 (functional delta per moatlib classify: touched
cmake/cupoch_hip_3rdparty.cmake, hip-config.cmake, device_vector_wrapper_bridge.cu,
utility/eigen.cpp -- all Windows-specific build fixes).

### Binary equivalence check

Built HEAD b3345c55e in agent_space/cupoch_revalidate_b3345c5 with the same cmake
flags as the prior gfx90a validation. Compared against agent_space/cupoch_full_build
(built at ~94f42e9, the tree used for the prior gfx90a pass):

    python3 utils/codeobj_diff.py agent_space/cupoch_full_build \
                                   agent_space/cupoch_revalidate_b3345c5

    verdict=identical
    lib/python/cupoch.cpython-313-x86_64-linux-gnu.so: identical
    (exported symbols + device ISA identical (13243 exports))

GLFW_INCLUDE_DIRS change confirmed inert on Linux: system glfw3 found at
/usr/lib/x86_64-linux-gnu/cmake/glfw3, so the system-GLFW branch is taken; the
new `set(GLFW_INCLUDE_DIRS ${OPENGL_INCLUDE_DIR})` in that branch produces the same
value as the old post-branch assignment. No include-path change; code objects identical.

### Build

From a fresh out-of-tree dir (agent_space/cupoch_revalidate_b3345c5):

    cmake <src> -DUSE_HIP=ON -DUSE_RMM=OFF \
        -DBUILD_UNIT_TESTS=OFF -DBUILD_PYTHON_MODULE=ON -DBUILD_PYBIND11=ON \
        -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release
    cmake --build . -j$(nproc)

Configure exit 0. Build exit 0, 0 errors. All 14 cupoch_* libs + 3rdparty + Python .so built.

### GPU dispatch confirmation

AMD_LOG_LEVEL=3: 5660 hipLaunchKernel calls, all returned hipSuccess.
GPU: Gfx Major/Minor/Stepping: 9/0/10 (gfx90a MI250X). Real device execution confirmed.

### Test results (22 / 22 PASS)

Python harness (python3.13, HIP_VISIBLE_DEVICES=1), 7 module categories:

**Core:**
- voxel_down_sample_determinism: 3 runs bitwise-identical. PASS.
- voxel_down_sample_count: GPU count == CPU reference (voxel_min_bound - voxel/2 convention). PASS.
- estimate_normals_no_nan: 0 NaN in 5000 normals. PASS.
- estimate_normals_determinism: min |dot(n1,n2)| >= 0.99 across 3 runs. PASS.

**Registration:**
- icp_point_to_plane_fitness: fitness >= 0.99 (injected rotation+translation recovered). PASS.
- icp_point_to_plane_rmse: rmse < 1e-2. PASS.
- icp_point_to_plane_determinism: max |T1-T2| < 1e-5. PASS.
- gicp_runs: GeneralizedICP executed without error. PASS.

**Integration (UniformTSDFVolume; ScalableTSDFVolume guarded out):**
- tsdf_integrate_runs: UniformTSDFVolume.integrate() on synthetic RGBD completed. PASS.
- tsdf_extract_nonempty: extract_point_cloud() -> non-zero pts. PASS.
- tsdf_two_integrations: 2-frame integration -> non-zero pts. PASS.

**Collision:**
- collision_compute_intersection_runs: compute_intersection(VoxelGrid, VoxelGrid). PASS.
- collision_determinism: same result across 2 runs. PASS.
- collision_self_intersection: self-intersection of non-empty VoxelGrid = True. PASS.

**IO:**
- io_write_pointcloud: write_point_cloud (PLY). PASS.
- io_read_pointcloud_size: 1000 pts roundtrip. PASS.
- io_roundtrip_precision: max |pts_read - pts_written| < 1e-5. PASS.

**Odometry:**
- odometry_compute_runs: compute_rgbd_odometry returned result. PASS.
- odometry_result_sane: result T is 4x4 matrix. PASS.

**Planning:**
- planning_find_path_runs: dijkstra_path(0, 3) returns non-empty path. PASS.
- planning_path_nonempty: 4 nodes [0,1,2,3]. PASS.
- planning_determinism: identical path across 2 calls. PASS.

**Guarded-out pieces still absent:**
- cupoch.imageproc: not present (libSGM-only, expected). PASS.
- cupoch.integration.ScalableTSDFVolume: not present (AMDGPU stack limit, expected). PASS.

### Result: 22 / 22 PASS. linux-gfx90a -> completed.

## Validation 2026-06-24 (linux-gfx1100, carry-forward to b3345c55)

Platform: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), ROCm 7.2.1.
Prior validated_sha: db0ea215b363226699c38aca06c432cdfded3c48 (22/22 PASS, 12506 hipLaunchKernel on gfx1100, same session).
New head: b3345c55e444e6da55bfd033ee991d4c8d3e088e.

### Delta assessment

Delta db0ea215..b3345c55 is one commit: "[ROCm] Fix Windows full-module build (gfx1201): 4 additional Windows fixes".
Changed files:
- cmake/cupoch_hip_3rdparty.cmake: libjpeg-turbo DOCDIR fallback (fires only when CMAKE_INSTALL_DOCDIR is empty, i.e. non-UNIX) and GLFW include split -- Linux path unchanged.
- third_party/rocm_hip_compat/lib/cmake/hip/hip-config.cmake: fallback path fix (fires only when hip::host target not already created, i.e. when the shim intercepts; on Linux find_package(hip) already created the targets) -- Linux path unchanged.
- src/python/cupoch_pybind/device_vector_wrapper_bridge.cu: +12 lines all inside `#if defined(_WIN32)` -- Linux preprocesses out.
- src/python/cupoch_pybind/utility/eigen.cpp: +18 lines all inside `#if defined(_WIN32)` -- Linux preprocesses out.

### Binary equivalence check

Built b3345c55 in agent_space/cupoch_revalidate_gfx1100_b3345c5 with the same cmake flags as the prior gfx1100 validation:

    cmake <src> -DUSE_HIP=ON -DUSE_RMM=OFF \
        -DBUILD_UNIT_TESTS=OFF -DBUILD_PYTHON_MODULE=ON -DBUILD_PYBIND11=ON \
        -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
        -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    cmake --build . -j$(nproc)

Configure exit 0. Build exit 0, 0 errors (same nodiscard hipGetLastError warnings -- benign).

    python3 utils/codeobj_diff.py agent_space/cupoch_full_gfx1100 \
                                   agent_space/cupoch_revalidate_gfx1100_b3345c5

    verdict=identical
    lib/python/cupoch.cpython-313-x86_64-linux-gnu.so: identical (exported symbols + device ISA identical (13221 exports))

Device code objects and exported symbols are identical between db0ea215 and b3345c55. No GPU re-run needed.

### Result: carry-forward (binary-equiv). linux-gfx1100 -> completed at b3345c55.

## Revalidation attempt 2026-06-24 (windows-gfx1101, binary-equiv check)

Platform: AMD Radeon PRO V710 (gfx1101) -- GPU absent from HIP runtime this session (TDR-removed by prior validation; mask 1 returns error 0100). GPU run impossible. Binary-equivalence carry-forward attempted.

Prior validated_sha: be446573bc604e2bcb4f532dd3c1b36ef4f6b29f (CORE_ONLY, 6 tests PASS).
Current head_sha: b3345c55e444e6da55bfd033ee991d4c8d3e088e (full module set + 4 Windows build fixes).

### Build

Built CORE_ONLY targets at b3345c55 for gfx1101:

    cmake <src> -G Ninja -DUSE_HIP=ON -DCUPOCH_CORE_ONLY=ON -DUSE_RMM=OFF \
        -DBUILD_UNIT_TESTS=OFF -DBUILD_PYTHON_MODULE=OFF -DBUILD_PYBIND11=OFF \
        -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
        -DCMAKE_C_COMPILER=<rocm>/lib/llvm/bin/clang.exe \
        -DCMAKE_CXX_COMPILER=<rocm>/lib/llvm/bin/clang++.exe \
        -DCMAKE_HIP_COMPILER=<rocm>/lib/llvm/bin/clang++.exe \
        -DCMAKE_PREFIX_PATH=<rocm> -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    ninja lib/cupoch_geometry.lib lib/cupoch_knn.lib lib/cupoch_camera.lib lib/cupoch_utility.lib -j32

Result: 0 errors. Build dir: agent_space/cupoch-win/build-gfx1101-b3345c55.

### Binary equivalence check

Compared .hip_fatbin sections of the core libs between the be446573 build (agent_space/cupoch-win/build-gfx1101) and the b3345c55 build (agent_space/cupoch-win/build-gfx1101-b3345c55) using llvm-ar extraction + llvm-objcopy --dump-section .hip_fatbin + sha256.

Result: verdict=differ.

- cupoch_utility.lib: dl_converter.cu.obj .hip_fatbin changed.
- cupoch_knn.lib: kdtree_flann.cu.obj and lbvh_knn.cu.obj .hip_fatbin changed.
- cupoch_geometry.lib: all 23 common objects differ; 4 new objects added (distancetransform.cu.obj, occupancygrid.cu.obj, trianglemesh.cu.obj, voxelgrid_factory.cu.obj -- the formerly-deferred geometry files now included and fixed for HIP).
- flann_cuda_s.lib: kdtree_cuda_3d_index.cu.obj .hip_fatbin changed.
- cupoch_camera.lib: no device code (host-only).

The device ISA changed substantially between be446573 and b3345c55: the full module set includes bug fixes to the geometry sources (distancetransform.cu __shared__ fix, trianglemesh.cu const functor, distance_test.inl/__device__ qualifiers, pinned_allocator.h rewrite) and the formerly-deferred geometry .cu files are now compiled. Binary-equiv carry-forward is NOT applicable.

### Result: windows-gfx1101 remains at revalidate.

A real gfx1101 GPU run is required to validate b3345c55. The gfx1101 GPU was absent this session; blocked until hardware is available again. The gfx1201 (RDNA4) validation at b3345c55 (22/22 PASS, same Windows build path) provides strong confidence the port is correct; gfx1101 validation is deferred to the next available session with that GPU.

## CUDA compile gate (PR-prep, 2026-06-24)

SHA: 4ea28fee9d2622b6122ae33ad80adad54b349520 (moat-port HEAD at PR-prep time).
Compiler: nvcc 12.8.93 (CUDA 12.8) from /opt/conda/envs/cuda-12.8.
Host compiler: gcc 13.3.0.
Architecture pinned: -DCMAKE_CUDA_ARCHITECTURES=80 (sm_80; no NVIDIA GPU on this host).

### Environment wrangling

The conda cuda-12.8 env has a non-standard layout (headers under targets/x86_64-linux/).
Two wrangling steps needed (neither is a code change to the fork):
1. CUDA toolkit discovery: let FindCUDA auto-discover from nvcc -v (which correctly
   reports TOP=/opt/conda/envs/cuda-12.8/targets/x86_64-linux). Do not override
   CUDA_TOOLKIT_ROOT_DIR.
2. cicc tool: nvcc invokes cicc via PATH. The conda layout puts cicc at
   /opt/conda/envs/cuda-12.8/nvvm/bin/cicc; symlinked it into
   /opt/conda/envs/cuda-12.8/bin/ (on PATH) so the legacy FindCUDA-generated
   makefiles find it. Throwaway: does not touch the fork.

Additionally, -DBUILD_PNG=ON -DBUILD_JPEG=ON were passed to avoid a pre-existing upstream
CMake bug: when system libpng/libjpeg are detected, third_party/CMakeLists.txt emits
`install(TARGETS png16)` where png16 is not a CMake target (only exists when building
from source). Unrelated to the port; building vendored libs avoids it.

Configure command:
    cmake <src> -B <build> -DUSE_HIP=OFF -DUSE_RMM=OFF \
        -DBUILD_UNIT_TESTS=OFF -DBUILD_PYTHON_MODULE=OFF -DBUILD_PYBIND11=OFF \
        -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=/opt/conda/envs/cuda-12.8/bin/nvcc \
        -DBUILD_PNG=ON -DBUILD_JPEG=ON
Configure exit: 0.

### Targets built

    cmake --build <build> --target cupoch_geometry cupoch_knn cupoch_camera cupoch_utility -j128
    Build exit: 0 (wall 176 s).

Static libraries produced:
- libcupoch_utility.a (3.4 MB)
- libcupoch_camera.a (1.4 MB)
- libcupoch_knn.a (2.8 MB)
- libcupoch_geometry.a (51 MB -- contains real sm_80 device code)

### Port-touched TUs compiled by nvcc

All of the following compiled under nvcc without error:
- platform.cu (cuda* -> hip* aliases; CUDA path: plain cuda_runtime.h)
- device_vector.cu (exec_policy, pinned_allocator; CUDA path: thrust::cuda::par unchanged)
- lbvh_knn.cu (CUDA path unchanged)
- kdtree_flann.cu (flann CUDA kdtree; CUDA path unchanged)
- distancetransform.cu (__shared__ DistanceVoxel USE_HIP workaround; CUDA path: original)
- distancetransform_factory.cu
- lineset.cu (includes distance_test.inl; CUDA path: original attributes)
- occupancygrid.cu, occupancygrid_factory.cu
- trianglemesh.cu, trianglemesh_factory.cu
- voxelgrid_factory.cu
- All remaining geometry .cu files (25 total in cupoch_geometry)

### Warnings

1. distancetransform.cu(164): nvcc #20054-D "dynamic initialization is not supported
   for a function-scope static __shared__ variable" -- for the original upstream
   `__shared__ DistanceVoxel block[BLOCKSIZE][BLOCKSIZE]` in the CUDA `#else` branch.
   Pre-dates our port; not a new issue; warning not error.

2. Eigen #20014-D "calling a __host__ function from a __host__ __device__ function"
   -- pre-existing Eigen/CUDA compatibility warning, unrelated to this port.

### What was NOT exercised

- RMM allocator path (USE_RMM=ON): RMM is CUDA-only, not changed by this port.
- Python pybind11 module: out of scope (no GPU-path risk from port).
- Full executable link to catch undefined-reference class: build stopped at static libs.
  Cupoch does not ship a minimal standalone CUDA demo; linking the Python .so needs
  libcuda.so which is absent on this host.
- Compute modules beyond the 4 core libs: not port-touched at the shared-source level.
- imageproc/libSGM: sgm also built (0 errors) but is not in the 4-lib gate set.

### Result: PASS

Zero errors. The USE_HIP guards in all port-touched files are correctly scoped: the
CUDA path for platform.cu, device_vector.cu, the 4 geometry .cu files, and the .inl
files compiles unchanged under nvcc 12.8 sm_80.
Compile-checked with nvcc; not GPU-run (no NVIDIA GPU on this host).

## Validation 2026-06-24 (linux-gfx90a, revalidate to 4ea28fee -- carry-forward, binary-equiv)

Platform: AMD Instinct MI250X (gfx90a, CDNA2, wave64), ROCm 7.2.1.
Prior validated_sha: e1b8b1f904115ba627c8b6523ebebbf850c6b791.
New head_sha: 4ea28fee9d2622b6122ae33ad80adad54b349520.
State: revalidate after CUPOCH_CORE_ONLY removal.

### Delta assessment

Two commits on top of e1b8b1f:
- af71654: deletes cmake/cupoch_core_3rdparty.cmake (114-line cmake-only file).
- 4ea28fe: removes option(CUPOCH_CORE_ONLY) declaration and its if() branch from CMakeLists.txt (15 lines).

Both commits are purely cmake/build-config changes: no .cu, .h, .cpp, or .inl files touched. The USE_HIP branch and the CUDA NVIDIA branch are unchanged. CUPOCH_CORE_ONLY=OFF was never set in a normal AMD build (the deliverable always used the full module set), so its removal cannot affect the compiled output.

### Build (both SHAs)

Both built from clean out-of-tree dirs under agent_space/:

    cmake <src> -DUSE_HIP=ON -DUSE_RMM=OFF \
        -DBUILD_UNIT_TESTS=OFF -DBUILD_PYTHON_MODULE=ON -DBUILD_PYBIND11=ON \
        -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release
    cmake --build . -j$(nproc)

- e1b8b1f: configure exit 0, build exit 0, 0 errors. dir: agent_space/cupoch_bineq_e1b8b1f.
- 4ea28fe: configure exit 0, build exit 0, 0 errors. dir: agent_space/cupoch_bineq_4ea28fe.

Both produced 14 libcupoch_*.a + 3rdparty libs + cupoch.cpython-313-x86_64-linux-gnu.so.

### Binary equivalence check

    python3 utils/codeobj_diff.py agent_space/cupoch_bineq_e1b8b1f agent_space/cupoch_bineq_4ea28fe

    verdict=identical
    lib/python/cupoch.cpython-313-x86_64-linux-gnu.so: identical
    (exported symbols + device ISA identical (13243 exports))

Device code objects and exported symbols are identical between e1b8b1f and 4ea28fe. No GPU re-run needed.

### Result: carry-forward (binary-equiv). linux-gfx90a -> completed at 4ea28fee.
## Validation 2026-06-24 (linux-gfx1100, carry-forward to 4ea28fee)

Platform: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), ROCm 7.2.1.
Prior validated_sha: b3345c55e444e6da55bfd033ee991d4c8d3e088e (22/22 PASS, 12506 hipLaunchKernel on gfx1100, same session).
New head: 4ea28fee9d2622b6122ae33ad80adad54b349520 ("[ROCm] Remove the CUPOCH_CORE_ONLY option and its CMake branch").

### Delta assessment

The branch was reworked/squashed onto a new base: e1b8b1f ("Add AMD GPU support via HIP") is a tree-identical squash of the prior b3345c55 content (git diff b3345c55..e1b8b1f -- = 0 lines). Above the squash base:
- af71654: "[ROCm] Remove the CUPOCH_CORE_ONLY bring-up option" (removes option() from CMakeLists.txt)
- 4ea28fee: "[ROCm] Remove the CUPOCH_CORE_ONLY option and its CMake branch" (removes the if (CUPOCH_CORE_ONLY)...elseif (USE_HIP) branch, deletes cmake/cupoch_core_3rdparty.cmake)

Changed files (4ea28fee vs e1b8b1f):
- CMakeLists.txt: option(CUPOCH_CORE_ONLY ...) removed; `if (CUPOCH_CORE_ONLY) ... elseif (USE_HIP)` -> `if (USE_HIP)`.
- cmake/cupoch_core_3rdparty.cmake: deleted (minimal bring-up CMake file; was never included in normal USE_HIP builds).

No .cu/.cpp/.h source files changed. The USE_HIP build path (what was validated) is unaffected.

### Build

    cmake <src> -DUSE_HIP=ON -DUSE_RMM=OFF \
        -DBUILD_UNIT_TESTS=OFF -DBUILD_PYTHON_MODULE=ON -DBUILD_PYBIND11=ON \
        -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
        -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    cmake --build . -j$(nproc)

Configure exit 0. Build exit 0, 0 errors (same nodiscard hipGetLastError warnings -- benign).
Build dir: agent_space/cupoch_revalidate_gfx1100_4ea28fee.

### Binary equivalence check

    python3 utils/codeobj_diff.py agent_space/cupoch_revalidate_gfx1100_b3345c5 \
                                   agent_space/cupoch_revalidate_gfx1100_4ea28fee

    verdict=identical
    lib/python/cupoch.cpython-313-x86_64-linux-gnu.so: identical
    (exported symbols + device ISA identical (13221 exports))

Device code objects and exported symbols are identical between b3345c55 (prior validation) and 4ea28fee (new head). The CUPOCH_CORE_ONLY removal is a CMake-only change; the USE_HIP compiled output is unchanged. No GPU re-run needed.

### Result: carry-forward (binary-equiv). linux-gfx1100 -> completed at 4ea28fee.
## Validation 2026-06-24 (windows-gfx1201, binary-equiv carry-forward to 4ea28fee)

Platform: AMD RX 9070 XT (gfx1201, RDNA4, wave32), Windows 11 Pro.
Prior validated_sha: e1b8b1f904115ba627c8b6523ebebbf850c6b791 (squashed port, same tree as b3345c55; 22/22 PASS).
New head: 4ea28fee9d2622b6122ae33ad80adad54b349520.

### Delta e1b8b1f9..4ea28fee

Two commits:
- "Remove the CUPOCH_CORE_ONLY bring-up option" (af71654): removes `option(CUPOCH_CORE_ONLY ...)` from CMakeLists.txt.
- "Remove the CUPOCH_CORE_ONLY option and its CMake branch" (4ea28fee): removes the `if(CUPOCH_CORE_ONLY)` branch from CMakeLists.txt and deletes cmake/cupoch_core_3rdparty.cmake.

Changed files: CMakeLists.txt, cmake/cupoch_core_3rdparty.cmake (deleted).

The `CUPOCH_CORE_ONLY` path was a bring-up shortcut that built only 4 core libs instead of the full module set. It was always disabled in production builds (`CUPOCH_CORE_ONLY=OFF`). Removing it has no effect on the `USE_HIP=ON` code path.

### Build at 4ea28fee (gfx1201)

Built from scratch in agent_space/cupoch-win/build-full-gfx1201-4ea28fee with the same cmake flags as the prior gfx1201 validation (see above). Configure exit 0, build exit 0, 349/349 Ninja targets, 0 errors.

### Binary equivalence check

Compared the gfx1201 device ISA (.text sections) from the extracted device ELF objects for both builds. The `.hip_fatbin` section sha256 hashes differ between builds due to non-ISA metadata (timestamps/paths in ELF .note/.comment sections within the embedded device ELF -- a known non-reproducibility in clang HIP builds), but the actual device machine code (.text sections) is byte-identical:

Method: llvm-ar extract -> llvm-objcopy --dump-section=.hip_fatbin -> clang-offload-bundler --unbundle --targets=hipv4-amdgcn-amd-amdhsa--gfx1201 -> llvm-objcopy --dump-section=.text -> sha256sum.

Result: 63 gfx1201 device objects checked across 13 libs (cupoch_utility, knn, geometry, collision, integration, io, odometry, planning, registration, visualization, wrapper, flann_cuda_s, plus camera/kinematics/kinfu are host-only). All 63 .text section hashes are identical between the e1b8b1f9-tree build and the 4ea28fee build.

verdict=identical

### Result: binary-equiv carry-forward. windows-gfx1201 -> completed at 4ea28fee.
