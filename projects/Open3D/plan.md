# Open3D -- ROCm/HIP Port Plan (lead platform: linux-gfx90a)

## Project
- Name: Open3D
- Upstream: https://github.com/isl-org/Open3D
- Default branch: main
- Base SHA analyzed: 0333798fcff5a2fe95470e69291eca6a9efbae6c
- Domain: modern 3D data-processing library (C++ + Python). Two GPU subsystems:
  the "tensor" stack (`cpp/open3d/core`, `cpp/open3d/t`) and ML/contrib ops
  (`cpp/open3d/ml`). GPU acceleration is CUDA-only today.

## Existing AMD support: NONE (genuine CUDA->HIP target -> PROCEED)
- No HIP/ROCm build path, no hipcc, no `<hip/hip_runtime.h>`, no AMD arch
  anywhere. The only "ROCm" strings are DLPack enum constants
  (`kDLROCM`/`kDLROCMHost` in `cpp/open3d/core/DLPack.h`, mirrored in
  `cpp/pybind/core/tensor.cpp`) -- Open3D never produces or consumes that
  device type; they are inert.
- GPU acceleration is CUDA-only (`BUILD_CUDA_MODULE`, `enable_language(CUDA)`).
  There is a SYCL path (`BUILD_SYCL_MODULE`, Intel oneAPI, mutually exclusive
  with CUDA) and a Vulkan compute path (gaussian-splat rendering only) -- neither
  is a HIP path, so per PORTING_GUIDE "AMD via OpenCL/Vulkan/SYCL with no HIP
  path" a ROCm/HIP port of the CUDA code still adds value.
- Decision: PROCEED with a correctness-first mechanical Strategy-A port of the
  core CUDA tensor stack, scoping out the NVIDIA-only / perf-tuned pieces named
  below.

## Build classification: pure CMake -> STRATEGY A
- Evidence: top-level `CMakeLists.txt:42` `option(BUILD_CUDA_MODULE ... OFF)`;
  `CMakeLists.txt:408-463` does `find_package(CUDAToolkit)` + `enable_language(CUDA)`
  + `set(CMAKE_CUDA_STANDARD 17)`, with clean `CMAKE_CUDA_ARCHITECTURES` handling
  (user value respected at line 425; otherwise a per-toolkit default list or
  `native`). No `find_package(Torch)` / `torch.utils.cpp_extension` in the core
  build -- PyTorch/TensorFlow appear ONLY in the optional ML-ops modules
  (`BUILD_PYTORCH_OPS`/`BUILD_TENSORFLOW_OPS`, both OFF by default), and even
  those are CMake-driven custom-op libs, not the core build path.
- This is the compat-header approach: `.cu`/`.cuh` translation units take the HIP
  toolchain; host C++ is untouched. Set `ext_type = cmake`.

## Port strategy: A (compat header + USE_HIP CMake gate), correctness-first
Rationale and the precise minimal footprint:
1. Add a `USE_HIP` option that, on AMD, swaps `enable_language(CUDA)` ->
   `enable_language(HIP)`, marks the compiled `.cu` (listed below) `LANGUAGE HIP`,
   and sets `HIP_ARCHITECTURES` from `${CMAKE_HIP_ARCHITECTURES}` (default the
   lead arch only when unset -- never a literal `gfx90a`, per the CudaSift/Gpufit
   churn lesson). The NVIDIA path stays byte-for-byte.
2. One compat header (e.g. `cpp/open3d/core/CUDAToHIP.h`, force-included on HIP
   TUs via `CMAKE_HIP_FLAGS -include`) aliasing the `cuda*` runtime symbols the
   project uses to `hip*`, and providing the full-warp 64-bit mask + any
   `__ffs(ull)` overloads (see Risk list). Everywhere else keep CUDA spelling.
3. Swap the math/library link target (`3rdparty_cublas` INTERFACE target,
   `3rdparty/find_dependencies.cmake:1808-1851`) to hipBLAS/hipSOLVER/hipSPARSE
   under `USE_HIP`; route rocThrust/hipCUB via include paths (header-only on
   `/opt/rocm/include`), NOT by linking `hip::device`/`roc::rocthrust` (the
   cupoch/STRUMPACK `-x hip` INTERFACE_COMPILE_OPTIONS propagation trap -- link
   `hip::host` only).

### Scope for the LEAD bringup (what to build + validate first)
Configure: `BUILD_CUDA_MODULE=ON` (as `USE_HIP`), `BUILD_PYTORCH_OPS=OFF`,
`BUILD_TENSORFLOW_OPS=OFF`, `BUILD_GUI=OFF`, `BUILD_WEBRTC=OFF`,
`BUILD_EXAMPLES=OFF`, `BUILD_UNIT_TESTS=ON`. This compiles ONLY the core tensor
GPU surface and excludes every NVIDIA-locked piece.

DEFERRED / scoped-OUT of the lead correctness target (documented, not failures):
- **CUTLASS-based ML conv kernels** (`cpp/open3d/ml/impl/{continuous_conv,sparse_conv}/*.cuh`):
  use the ancient CUTLASS 1.3.3 GEMM API (`cutlass::gemm::SgemmTraits`,
  `3rdparty/cutlass/cutlass.cmake`). Per PORTING_GUIDE (changelog 2026-05-30,
  "CUTLASS does NOT port to ROCm"), do NOT shim CUTLASS. These are reached ONLY
  via `BUILD_PYTORCH_OPS`/`BUILD_TENSORFLOW_OPS` (OFF), so keeping ML ops off
  removes them cleanly. A later AMD-native pass could reimplement the conv GEMM
  on Composable Kernel; out of scope for correctness-first. NOTE: the CUTLASS
  ExternalProject is fetched whenever `BUILD_CUDA_MODULE` is ON
  (`find_dependencies.cmake:557`) but it is a header-only INTERFACE include dir
  that NO compiled TU includes when ML ops are off -- a dead download, harmless
  to the HIP build (leave it, or guard the fetch to ML-ops-only as a tidy-up).
- **NPP image ops** (`cpp/open3d/t/geometry/kernel/NPPImage.{cpp,h}`, includes
  `<npp.h>`; `nppiDilateBorder`, `nppiFilterBilateralGaussBorder`, Resize,
  RGBToGray, Filter): NVIDIA Performance Primitives has NO ROCm equivalent.
  `cpp/open3d/t/geometry/Image.cpp` dispatches GPU image filters to NPP and CPU
  to IPP. For the lead port, `USE_HIP`-guard `NPPImage.*` out of the build and
  make the CUDA branch in `Image.cpp` `LogError`/throw "unsupported on ROCm" so
  the GPU `Image` filter ops are an explicitly-unsupported path; the CPU (IPP)
  paths and all non-Image GPU code are unaffected. (A later pass can hand-write
  the ~6 image ops as HIP kernels.) Image-filter GPU tests are scoped out.
- **SlabHash hashmap backend** (`cpp/open3d/core/hashmap/CUDA/SlabHash*`,
  `SlabNodeManager.*`): a 32-lane-warp-cooperative concurrent hashtable
  (`__ballot_sync(kSyncLanesMask,...)`, `__shfl_sync(...,kWarpSize)`, `tid>>5`
  warp-id, `threadIdx.x&0x1f` lane, `atomicCAS` lane election). This is a dense
  wave64 fault-class body. It is the NON-default backend (`HashBackendType::Slab`);
  the DEFAULT (`CreateCUDAHashBackend.cu:44`) is the stdgpu backend. Make SlabHash
  COMPILE (64-bit masks so it does not block the build) but validate the stdgpu
  backend (what the default `HashMap`/`HashSet` tests exercise). Guard/skip the
  Slab-specific test cases; a correct wave64 SlabHash rewrite (per the RXMesh
  32-bit-word-ballot and cuCollections lane-election lessons) is a deferred
  follow-up.

### Core GPU surface that MUST build + validate (the lead deliverable)
Compiled `.cu`/`.cpp(CUDA)` in the default build (from the per-dir CMakeLists):
- `core/`: `hashmap/CUDA/{CreateCUDAHashBackend,CUDAHashBackendBuffer,SlabNodeManager}.cu`,
  `kernel/{ArangeCUDA,BinaryEWCUDA,IndexGetSetCUDA,IndexReductionCUDA,NonZeroCUDA,ReductionCUDA,UnaryEWCUDA}.cu`,
  `linalg/{AddMMCUDA,InverseCUDA,LeastSquaresCUDA,LUCUDA,MatmulCUDA,SolveCUDA,SVDCUDA}.cpp` + `linalg/TriCUDA.cu`,
  `nns/{FixedRadiusSearchOps,KnnSearchOps}.cu`, `nns/kernel/BlockSelectFloat{32,64}.cu`,
  plus the stdgpu-backend header `hashmap/CUDA/StdGPUHashBackend.h`.
- `t/geometry/kernel/`: `ImageCUDA.cu`(non-NPP geometry kernels), `PointCloudCUDA.cu`,
  `TriangleMeshCUDA.cu`, `TransformCUDA.cu`, `VoxelBlockGridCUDA.cu`.
- `t/pipelines/kernel/`: `RegistrationCUDA.cu`, `FillInLinearSystemCUDA.cu`,
  `RGBDOdometryCUDA.cu`, `TransformationConverter.cu`, `FeatureCUDA.cu`.
- `ml/contrib/IoU.cu` (the only always-built ML `.cu`).

## CUDA surface inventory
- Kernels / `__global__`/`__device__`: dense across core/kernel, t/geometry,
  t/pipelines, nns. Elementwise kernels (Unary/Binary/IndexGetSet/Arange) have
  NO cross-lane ops -- wave64-correct unchanged (grep for warp primitives empty).
- Warp intrinsics (`__shfl*_sync`, `__ballot_sync`, `__ffs`, `__syncwarp`,
  `warpSize`): concentrated in (a) `core/nns/kernel/` (FAISS-derived warp-select:
  `WarpShuffle.cuh`, `Select.cuh`, `MergeNetwork.cuh`, `BlockSelect*`,
  `DeviceDefs.cuh` with `kWarpSize=32`), (b) `core/hashmap/CUDA/SlabHash*`
  (scoped out), (c) `core/kernel/ReductionCUDA.cu`, (d) `ml/impl/*` (scoped out)
  and `ml/contrib`/`ml/impl/sparse_conv` conv kernels (scoped out).
- Block reductions: `t/pipelines/kernel/RegistrationCUDA.cu` uses
  `cub::BlockReduce` + `atomicAdd`-combine (NOT manual warp tails) -> hipCUB
  drop-in, wave-size-agnostic, wave64-correct by construction.
- Math libraries: cuBLAS (`gemm`,`trsm`: Sgemm/Dgemm/Strsm/Dtrsm) and cuSOLVER
  (`cusolverDn{S,D}{getrf,getrs,gesvd,geqrf,ormqr}`, Create/Destroy) in
  `core/linalg/{BlasWrapper.h,LinalgHeadersCUDA.h,*CUDA.cpp}`. All typed (Dn{S,D})
  APIs -> hipBLAS/hipSOLVER direct, no X-path/64-bit-generic/batched-QR gaps
  (per the raft hipSOLVER lesson). Static link list also names cusparse/cublasLt
  (`find_dependencies.cmake:1808-1851`) -> hipsparse/hipblaslt.
- Thrust/CUB: `thrust::` in hashmap, `core/kernel/{NonZeroCUDA,ReductionCUDA}.cu`,
  `nns/FixedRadiusSearchImpl.cuh`, `t/pipelines/kernel/{RGBDOdometry,Registration}CUDA.cu`;
  `cub::` in `RegistrationCUDA.cu`. rocThrust/hipCUB are header drop-ins
  (cudaKDTree lesson): rocThrust under same `thrust::` API/headers, hipCUB needs
  the `#define cub hipcub` (or `<hipcub/...>` include) alias in the compat header.
- Textures/surfaces: NONE in the core (the "texture" hits are mesh-material
  prose or the `texture_alignment` int memory-alloc parameter in FixedRadiusSearch).
  Texture fault classes (256B pitch, linear filtering, layered arrays) are N/A.
- Managed/pinned memory: no `cudaMallocManaged` in GPU paths. Memory manager
  (`core/MemoryManagerCUDA.cpp`) uses `cudaMalloc`/`cudaFree`/`cudaMemcpyAsync`
  and `cudaMallocAsync`/`cudaFreeAsync` (stream-ordered) when
  `ENABLE_CACHED_CUDA_MANAGER` -> all map 1:1 to `hip*`. There is a separate
  `MemoryManagerCached.cpp` (Open3D's own caching allocator).
- Streams/events: Open3D wraps a current stream (`core::cuda::GetStream()`,
  `cudaStream_t`) -> `hipStream_t`; standard 1:1.
- Device props / warp size: Open3D ALREADY has a runtime warp-size query
  (`GetCUDACurrentWarpSize()` -> `cudaDeviceGetAttribute(cudaDevAttrWarpSize)`,
  `CUDAUtils.cpp:283`), which returns 64 on gfx90a -- so HOST-side warp sizing is
  already correct. The compile-time `kWarpSize=32` constant is the problem (next).
- 3rdparty GPU deps managed BY Open3D (ExternalProject, not MOAT projects):
  stdgpu (commit d7c07d0, the default hashmap backend) and CUTLASS 1.3.3 (ML only).

## Risk list (fault classes; cited to PORTING_GUIDE)
1. **Warp size 32-vs-64 -- FAISS nns/kernel** (PORTING_GUIDE "Warp size";
   popsift/AutoDock/raft wave64 lessons). `core/nns/kernel/DeviceDefs.cuh` hardcodes
   `constexpr int kWarpSize = 32`; the warp-select/block-select/merge-network code
   (`WarpShuffle.cuh`, `Select.cuh`, `MergeNetwork.cuh`, `BlockSelect*`) is
   templated on it and assumes 32-lane warp-bitonic semantics. Two sub-risks:
   (a) COMPILE: `__shfl_sync(0xffffffff, ...)` / `SHFL_SYNC` use a 32-bit mask;
   HIP `__shfl_*_sync` static_assert `sizeof(mask)==8`, so the 0xffffffff literal
   fails to compile (raft/AutoDock lesson) -- centralize a 64-bit full-warp mask
   in the compat header. (b) CORRECTNESS: the bitonic warp-select sort assumes a
   32-lane warp. Decision: this is the highest-risk correctness body. Validate
   the KNN/FixedRadius results against a CPU brute-force reference (the cupoch
   KNN-validation pattern) and run-to-run determinism; if wave64 mis-sorts,
   either (i) pin these kernels to a 32-lane sub-group model (popsift two-halves
   /cuCollections tiled_partition<32> lesson) or (ii) set `kWarpSize=64` and
   verify the merge-network step counts. KnnSearch/FixedRadiusSearch tests gate
   this.
2. **Warp-mask literal in ReductionCUDA** (same COMPILE class). `ReductionCUDA.cu`
   `shfl_down` default `mask=0xffffffff` -> 64-bit on HIP. The reduction tree
   already uses the runtime `warpSize` (lines 615-626), so it is wave-size-aware;
   still confirm the `__shfl_down_sync` warp tail is the synced form and does not
   race on wave64 (MPPI warp-synchronous-tail lesson) via a fixed-seed
   determinism check on `Reduce*Sum*`.
3. **SlabHash wave64 (scoped out of validation, must still compile)**
   (RXMesh 32-bit-word-ballot lesson + cuCollections lane-election). `kSyncLanesMask`
   32-bit -> 64-bit to compile; the lane-election/`__ballot`/`atomicCAS` logic is
   wave64-incorrect but the backend is non-default. Guard Slab tests.
4. **Library swaps** (PORTING_GUIDE "Library swaps"; raft hipSOLVER lesson).
   cuBLAS->hipBLAS, cuSOLVER->hipSOLVER, cuSPARSE->hipSPARSE, cublasLt->hipblasLt.
   Watch: `cusolverStatus_t` fmt::formatter switch (`LinalgHeadersCUDA.h`) lists
   the full CUSOLVER_STATUS_* enum -- map only the codes hipSOLVER defines and
   `USE_HIP`-guard orphan `case` labels (MPPI hipFFT/hipSOLVER enum non-1:1
   lesson). `hipsolverDn{S,D}*` typed APIs all exist; no batched-QR/X-path used.
5. **rocThrust/hipCUB drop-in details** (cudaKDTree/cupoch). rocThrust is a true
   `thrust::` drop-in; hipCUB needs the `cub`->`hipcub` alias. Watch the cupoch
   `thrust::cuda::par`->`thrust::hip::par` namespace delta IF any exec-policy is
   used (grep showed plain algorithm calls; verify during build). Do NOT link
   `hip::device`/`roc::rocthrust` (the `-x hip` propagation trap that breaks host
   `.cpp` in the same target -- cupoch/STRUMPACK); link `hip::host` + include dir.
6. **stdgpu HIP backend bringup** (cupoch stdgpu-1.3.0 lesson, changelog 193).
   stdgpu at the pinned commit (d7c07d0) HAS an experimental HIP backend
   (`STDGPU_BACKEND_HIP`). It is fetched via ExternalProject_Add
   (`3rdparty/stdgpu/stdgpu.cmake`) with CUDA-specific args
   (`CUDAToolkit_ROOT`, `THRUST_INCLUDE_DIR`, `CMAKE_CUDA_ARCHITECTURES`). Under
   USE_HIP, pass `-DSTDGPU_BACKEND=STDGPU_BACKEND_HIP` and ROCm flags instead;
   expect the cupoch-class bitrot (Findthrust seeing rocThrust THRUST_VERSION,
   hip-config SameMajorVersion rejection, forcing TUs LANGUAGE HIP). This is a
   sub-build inside the port and the single biggest integration risk for the
   DEFAULT hashmap; if the stdgpu HIP backend is too rotten, fall back to
   exercising the Slab backend after its wave64 fix instead -- but try stdgpu
   first since it is the default the tests use.
7. **atomicAdd float-accumulation determinism** (AutoDock/MPPI/STRUMPACK).
   Registration (`cub::BlockReduce` + `atomicAdd`-combine into the 6x6/29-element
   global sum) is order-nondeterministic at ULP level on ANY GPU. Validate ICP
   registration by transform-accuracy tolerance + physics, NOT bitwise file
   equality; run-to-run differences at ULP are expected, not a bug.
8. **__CUDA_ARCH__ / __CUDACC__ guard traps** (cudaKDTree/MPPI/raft/cudf family).
   Grep core for `#ifdef __CUDA_ARCH__` device-vs-host data-pointer selection and
   `#if defined(__CUDACC__)` device-decoration gates; HIP defines neither
   (`__HIP_DEVICE_COMPILE__`/`__HIPCC__` instead). Rewrite each to
   `... || defined(__HIP_DEVICE_COMPILE__)` / `... || defined(__HIPCC__)`. Most
   of Open3D's host/device split is via its own `OPEN3D_HOST_DEVICE`-style macros
   (`Macro.h`, `CUDAUtils.h`) -- audit those macros for `__CUDACC__` gating first.
9. **Rule-of-five on resource handles** (PORTING_GUIDE; colmap CuTexObj). The
   linalg `CuSolverContext`/`CuBLASContext` RAII handle wrappers
   (`LinalgUtils.{h,cpp}`) -- confirm explicit default-init + guarded destroy so a
   double-destroy/default-construct does not fault on HIP.
10. **OOB neighbor reads** (PORTING_GUIDE; colmap ComputeDOG). The nns radius/knn
    kernels gather neighbors; check edge index clamping. Lower risk than 1-3 but
    on the same nns path; the KNN-vs-CPU validation catches it.

## File-by-file change list (lead port; all additive / USE_HIP-guarded)
- `CMakeLists.txt`: add `option(USE_HIP ...)`; under USE_HIP do
  `enable_language(HIP)` + set `HIP_ARCHITECTURES` from `${CMAKE_HIP_ARCHITECTURES}`
  (default lead arch only when unset) instead of the `enable_language(CUDA)` block
  (lines 408-463). Keep CUDA path intact (`else()`).
- NEW `cpp/open3d/core/CUDAToHIP.h` (compat header): `#include <hip/hip_runtime.h>`
  under HIP (after `<cstring>`/`<cstdlib>` per gpuRIR host-memcpy lesson); 64-bit
  full-warp mask macro; `cub`->`hipcub` alias; `__ffs(unsigned long long)` overload
  if a tile.ballot()/__ffs path needs it; any `cuda*`->`hip*` runtime aliases not
  auto-handled. Force-include on HIP TUs via `CMAKE_HIP_FLAGS -include`.
- `cmake/Open3DSetGlobalProperties.cmake`: the `--expt-extended-lambda` flag
  (line 169, `$<$<COMPILE_LANGUAGE:CUDA>:...>`) -> add a HIP-language branch
  (hipcc accepts `-fgpu-rdc`/relocatable + extended lambda differently; HIP
  enables extended lambda by default). Add `BUILD_CUDA_MODULE` define handling
  for the HIP build so `#if defined(BUILD_CUDA_MODULE)` host blocks (e.g.
  `DeviceHashBackend.cpp:30`, `Image.cpp` CUDA branches) compile (AutoDock
  "define USE_CUDA for the HIP build" minimal-footprint lesson -- here Open3D
  keys host GPU blocks on `BUILD_CUDA_MODULE`, which is already set for the HIP
  build, so this mostly works as-is).
- `3rdparty/find_dependencies.cmake`: under USE_HIP, build the `3rdparty_cublas`
  INTERFACE target from `hip::hipblas`/`roc::hipsolver`/`roc::hipsparse`/
  `roc::hipblaslt` instead of `CUDA::*` (lines 1808-1851); guard the NPP block
  (1855+) out; pass HIP args to the stdgpu ExternalProject (`stdgpu.cmake`);
  the CUTLASS fetch can be left or guarded to ML-ops-only.
- `cpp/open3d/core/nns/kernel/DeviceDefs.cuh`: `kWarpSize` per-arch
  (64 on `__GFX9__`, else 32) per the PORTING_GUIDE warp-size template; +
  `WarpShuffle.cuh`/`ReductionCUDA.cu` 64-bit masks.
- `cpp/open3d/core/hashmap/CUDA/SlabHash*`,`SlabNodeManager.*`: 64-bit masks to
  COMPILE; wave64-correctness deferred (backend non-default).
- `cpp/open3d/core/linalg/LinalgHeadersCUDA.h`: USE_HIP-guard orphan
  CUSOLVER_STATUS_* `case` labels; swap headers to hipblas/hipsolver under HIP.
- `cpp/open3d/t/geometry/kernel/NPPImage.{cpp,h}` + `Image.cpp`: USE_HIP-guard the
  NPP path out; GPU image-filter ops LogError on HIP.
- Per-dir `CMakeLists.txt` (core, t/geometry/kernel, t/pipelines/kernel,
  ml/contrib): mark the listed `.cu` `LANGUAGE HIP` under USE_HIP (a top-level
  add_library/add_executable override per the MPPI lesson can do this with zero
  per-dir churn if preferred). `core/linalg/*CUDA.cpp` are plain host .cpp that
  call cublas/cusolver -- they do NOT need LANGUAGE HIP (host C++), only the link
  swap; `TriCUDA.cu` does.

## Build commands (gfx90a)
System deps (headless, no GUI): `bash util/install_deps_ubuntu.sh assume-yes`
(or a subset: the core build needs only build-essential + libssl-dev + a C++
compiler; GUI/Filament/OSMesa deps are unneeded with BUILD_GUI=OFF). ROCm 7.2.1
provides hipBLAS/hipSOLVER/hipSPARSE/hipCUB/rocThrust at /opt/rocm.

Configure (lead bringup):
```
cmake -S projects/Open3D/src -B projects/Open3D/src/build \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DBUILD_CUDA_MODULE=ON \
  -DBUILD_PYTORCH_OPS=OFF -DBUILD_TENSORFLOW_OPS=OFF \
  -DBUILD_GUI=OFF -DBUILD_WEBRTC=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF -DBUILD_PYTHON_MODULE=OFF \
  -DBUILD_UNIT_TESTS=ON -DBUILD_ISPC_MODULE=OFF \
  -DCMAKE_BUILD_TYPE=Release -DDEVELOPER_BUILD=ON
```
(`BUILD_CUDA_MODULE` is the existing switch; the porter wires `USE_HIP` to drive
the HIP language inside it. Note: how `USE_HIP` and `BUILD_CUDA_MODULE` compose
is a porter decision -- either reuse `BUILD_CUDA_MODULE=ON` and add a
`USE_HIP`/`OPEN3D_USE_HIP` sub-flag, or have `USE_HIP` imply the CUDA-module code
paths. The `#if defined(BUILD_CUDA_MODULE)` host guards must stay active on HIP.)
For followers: same command, only `-DCMAKE_HIP_ARCHITECTURES=gfx1100` (or
gfx1151) changes -- no source edit, by design.

Build: `cmake --build projects/Open3D/src/build -j$(nproc) --target tests`
(also `-DBUILD_COMMON_CUDA_ARCHS=OFF` keeps it single-arch/fast).

Optional CPU-only compile smoke (manual, never a gate): docker image
`rocm/dev-ubuntu-24.04:7.2.4-complete`.

## Test plan
- GPU tests: the single gtest binary `build/bin/tests` (built by
  `BUILD_UNIT_TESTS=ON`). Open3D's `PermuteDevices`/`PermuteDevicePairs` gtest
  fixtures (`cpp/tests/core/CoreTest.cpp:36`) parameterize over
  `{CPU, CUDA}` and auto-run every parameterized test on the gfx90a device when
  the CUDA(HIP) module is built and a GPU is visible. Run SERIALLY on the single
  assigned GPU (`./bin/tests`, NOT `-jN` -- MPPI single-GPU contention lesson):
  `cd build && ./bin/tests --gtest_shuffle`.
- Targeted GPU suites that gate the core port (filter with `--gtest_filter`):
  `*Tensor*`, `*MemoryManager*`, `*Device*`, `*HashMap*` (default=stdgpu backend),
  `*NearestNeighborSearch*` / `*KnnIndex*` / `*FixedRadiusIndex*` (the FAISS
  wave64 risk), `*Reduction*`/`*Reduce*Sum*` (warp reduction), and the t-geometry
  GPU tests `*PointCloud*`, `*TriangleMesh*`, `*VoxelBlockGrid*`, `*LineSet*`,
  plus registration/odometry under `cpp/tests/t/pipelines` (ICP). These run on
  both CPU and CUDA params, so CPU is the built-in reference oracle.
- Correctness oracles (per PORTING_GUIDE): the CPU param of each PermuteDevices
  test IS the reference -- assert CUDA==CPU within tolerance. For KNN/FixedRadius,
  additionally a CPU brute-force + run-to-run determinism check (cupoch pattern).
  For ICP registration, transform-accuracy tolerance + run-to-run stability (ULP
  atomicAdd nondeterminism expected, AutoDock/MPPI lesson) -- not bitwise.
- Scoped-OUT test cases (documented, not regressions): SlabHash-backend-specific
  HashMap cases (`HashBackendType::Slab`), GPU `Image` filter ops (NPP path), and
  anything requiring `BUILD_PYTORCH_OPS`/`BUILD_TENSORFLOW_OPS` (the CUTLASS conv
  ops). Python tests (`run_python_tests`) require the python module / ML ops and
  are out of the lead C++ GPU scope.
- Non-GPU regression set that MUST NOT regress: the CPU param of every test
  above; the build is the same binary, so a full `./bin/tests` run with no GPU
  visible (or the CPU-only params) is the non-GPU baseline. The validator should
  confirm the CPU-side counts are unchanged from a stock CUDA-off build.

## Inter-project MOAT dependencies
NONE. Open3D vendors its GPU 3rdparty deps (stdgpu, CUTLASS, CUB) itself via
ExternalProject_Add inside its own build -- they are not jeffdaily forks and not
MOAT projects. (stdgpu's HIP backend is upstream in stdgpu itself.) No
`set-deps` needed. Math libs (hipBLAS/hipSOLVER/hipSPARSE/hipCUB/rocThrust) come
from the system ROCm install, not MOAT.

## Open questions (for porter/validator, not blockers)
1. How cleanly does the pinned stdgpu commit's experimental HIP backend build
   under ROCm 7.2.1 (Risk 6)? If it is too rotten and cannot be made to build in
   reasonable effort, validate the SlabHash backend after its wave64 fix instead
   (HashMap tests can select the backend), and document stdgpu-HIP as deferred.
2. Does the FAISS warp-select (nns/kernel) produce correct KNN ordering on wave64
   with `kWarpSize=64`, or must it be pinned to 32-lane sub-groups (Risk 1)? The
   KNN-vs-CPU + determinism validation decides; budget for the sub-group rewrite.
3. Exact composition of `USE_HIP` vs the existing `BUILD_CUDA_MODULE` switch and
   the `#if defined(BUILD_CUDA_MODULE)` host guards -- a porter design choice
   (reuse the flag + sub-toggle vs. a parallel option); both keep the NVIDIA path
   intact.
