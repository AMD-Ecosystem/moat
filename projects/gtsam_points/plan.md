# gtsam_points -- ROCm/HIP port plan (lead: linux-gfx90a, MI250X, ROCm 7.2.1)

## Project
- Name: gtsam_points
- Upstream: https://github.com/koide3/gtsam_points
- Default branch: master (cloned read-only at projects/gtsam_points/src; HEAD shallow @ v1.2.1, project version 1.2.1)
- What it is: a collection of GTSAM factors and optimizers for range-based / point-cloud SLAM. CPU library plus an optional GPU library (`gtsam_points_cuda`) whose centerpiece is the GPU VGICP registration factor (`IntegratedVGICPFactorGPU`) and the GPU Gaussian voxelmap (`GaussianVoxelMapGPU`).

## Existing AMD support -> decision: fresh CUDA->HIP port (clear value)
- No HIP/ROCm/AMD path anywhere upstream: only the `master` branch exists, no stale ROCm branch/PR/fork, README and Dockerfiles mention CUDA only (CUDA 12.2/12.5/13.1), GPU lib links `CUDA::cudart` directly.
- Not even an OpenCL/Vulkan alternative -- the GPU path is CUDA-only. So a ROCm/HIP port of the CUDA registration + voxelmap is net-new GPU support for AMD. Proceed.
- Not performance-critical-rewrite territory: there is NO CUTLASS/CuTe/wgmma/warp-specialized code. The GPU work is entirely rocThrust/hipCUB device primitives over Eigen-typed functors. A correctness-first mechanical Strategy-A port is the right and sufficient approach; no AMD-native (CK/rocWMMA/MFMA) rewrite is warranted.

## Build classification: pure CMake -> Strategy A
Evidence (CMakeLists.txt):
- `project(gtsam_points VERSION 1.2.1 LANGUAGES CXX)`, `cmake_minimum_required 3.22`, `set(CMAKE_CXX_STANDARD 17)`.
- No `find_package(Torch)`, no `torch.utils.cpp_extension`, no setup.py/pyproject. This is a standalone C++/CMake library.
- GPU is gated by `option(BUILD_WITH_CUDA "Build with GPU support" OFF)` (line 19). When ON it: sets `GTSAM_POINTS_USE_CUDA`, `add_definitions(-DBUILD_GTSAM_POINTS_GPU)` (line 64), `find_package(CUDAToolkit REQUIRED)`, `enable_language(CUDA)`, and builds a SEPARATE shared lib `gtsam_points_cuda` from the `.cu`/`.cpp` GPU sources (lines 236-289), which the main `gtsam_points` lib links.
- ext_type = `cmake`. Set in upstream.json and status.json.

So the HIP build mirrors `BUILD_WITH_CUDA` with a parallel `BUILD_WITH_HIP` (USE_HIP) option: same `gtsam_points_cuda` target and source list, `.cu` marked `LANGUAGE HIP`, gated `enable_language(HIP)`.

## Port strategy: A (compat header + LANGUAGE HIP), colmap/cupoch model
Rationale: the entire GPU surface is HIP-mechanical. There are NO `__global__` kernels and NO warp intrinsics (see inventory); all device work is rocThrust/hipCUB primitives plus `__host__ __device__` Eigen functors, which compile under clang/hipcc unchanged (cudaKDTree precedent: rocThrust/hipCUB are header drop-ins under /opt/rocm/include). The diff stays: one compat header, a CMake HIP option, a thin set of toolkit-include shims, and a handful of guarded fixes. The NVIDIA build path stays byte-for-byte unchanged behind `if(USE_HIP)` / `#if defined(USE_HIP)`.

Mechanics (per PORTING_GUIDE Strategy A + the MPPI-Generic/cupoch refinements):
1. Add `include/gtsam_points/cuda/cuda_to_hip.h` (the only file that knows HIP). On `USE_HIP`/`__HIP_PLATFORM_AMD__` include `<cstring>`/`<cstdlib>` BEFORE `<hip/hip_runtime.h>` (gpuRIR lesson: host memcpy/memset must resolve to libc, not HIP `__device__` overloads), then alias the cuda* symbols the project actually uses to hip* (full list in inventory). On NVIDIA it is a no-op `#include <cuda_runtime.h>`. Use torch hipify `cuda_to_hip_mappings.py` as the authoritative name source.
2. Provide CUDA-toolkit-named forwarding shim headers in a `hip_compat/` dir for the angle-bracket includes ROCm has no shim for: `<cuda_runtime.h>`, `<cuda_runtime_api.h>`, `<cuda.h>`, `<device_atomic_functions.h>`. Each just `#include "cuda_to_hip.h"`. Add this dir to the HIP include path ONLY (target_include_directories under `if(USE_HIP)`, PRIVATE+BEFORE on the gtsam_points_cuda target). On CUDA the dir is absent so the real toolkit headers win (MPPI-Generic lesson). This avoids editing the ~10 source `#include <cuda_runtime.h>` lines and avoids globally shadowing the real header (which would break rocThrust internals).
3. CMake: `option(BUILD_WITH_HIP ...)` -> `enable_language(HIP)`; default `CMAKE_HIP_ARCHITECTURES` to `gfx90a` only when unset (NEVER a literal -- CudaSift/Gpufit lesson: a hardcoded arch overrides `-DCMAKE_HIP_ARCHITECTURES` and forces every follower to edit CMake, churning head_sha). Mark the GPU lib's `.cu` sources `LANGUAGE HIP` and set its `HIP_ARCHITECTURES` from `${CMAKE_HIP_ARCHITECTURES}`. Keep `GTSAM_POINTS_USE_CUDA` and `BUILD_GTSAM_POINTS_GPU` defined for the HIP build too (AutoDock-GPU lesson: define USE_CUDA for the HIP build so all the existing `#ifdef GTSAM_POINTS_USE_CUDA` test/host guards compile unchanged with zero churn). Link `hip::host` (runtime) NOT `hip::device` (cupoch lesson: hip::device puts `-x hip --offload-arch` in INTERFACE_COMPILE_OPTIONS which would break any host `.cpp` in the same target -- the GPU lib mixes `.cu` and `.cpp`). rocThrust/hipCUB/rocPRIM are header-only on the default clang search path, so a LANGUAGE-HIP `.cu` needs no extra link target.
4. Guard genuinely divergent code with `#if defined(USE_HIP)`; keep rare.

## CUDA surface inventory
GPU library = `gtsam_points_cuda` (CMakeLists 236-289). All device code lives in:
- `src/gtsam_points/cuda/*.cu|.cpp` (runtime infra): check_error{,_cusolver,_curand}, cuda_memory, cuda_buffer, cuda_stream, cuda_device_sync, cuda_device_prop, cuda_graph, cuda_graph_exec, stream_roundrobin, stream_temp_buffer_roundrobin, nonlinear_factor_set_gpu{,_create}. (`gl_buffer_map.cu` is COMMENTED OUT in CMake line 249 -- out of scope.)
- `src/gtsam_points/types/*.cu`: point_cloud.cu, point_cloud_gpu.cu, gaussian_voxelmap_gpu.cu, gaussian_voxelmap_gpu_funcs.cu.
- `src/gtsam_points/factors/integrated_vgicp_derivatives*.cu` (+ `_inliers/_compute/_linearize`) and `integrated_vgicp_factor_gpu.cpp`.
- `src/gtsam_points/util/easy_profiler_cuda.cu`.
- Device kernel headers (header-only functors): `include/gtsam_points/cuda/kernels/{vgicp_derivatives,lookup_voxels,vector3_hash,linearized_system,pose,untie}.cuh`.

Kernels / device entry points:
- ZERO `__global__` kernels. All device execution is via rocThrust (`thrust::for_each / transform / sequence`) and hipCUB (`cub::DeviceReduce::Reduce`, `cub::DeviceSelect::If`) over `__host__ __device__` functors. This is the cupoch/cudaKDTree model.
- The VGICP linearize/error/inlier path: a chain of `thrust::make_transform_iterator` (lookup_voxels_kernel -> vgicp_derivatives_kernel) reduced by `cub::DeviceReduce::Reduce` with `thrust::plus<LinearizedSystem6>` and identity `LinearizedSystem6::zero()`. `LinearizedSystem6` is a struct of Eigen 6x6/6x1 matrices with `__host__ __device__ operator+`.

Warp intrinsics / hardcoded 32:
- NONE. grep for `__shfl* / __ballot / __activemask / __any / __all / warpSize / __syncwarp / __popc / cooperative_groups / tiled_partition` is EMPTY across src+include. The only literal `32` is `case 31:`/`case 32?` in a cuSOLVER error-code switch (not a warp width). => the wave64 reduction wall that this class of project usually hits is ABSENT here (see Risk list for why).

Atomics (all in gaussian_voxelmap_gpu.cu):
- `atomicCAS(int*, -1, idx)` + `atomicAdd(int*)` for open-addressing linear-probe hash bucket insertion (voxel_bucket_assignment_kernel).
- `atomicAdd(int*)`, `atomicAdd(float*)` x12 for per-voxel point/mean/cov accumulation (accumulate_points_kernel).
- ONE `atomicMax(unsigned int*, __float_as_uint(intensity))` for max-intensity per voxel.

Textures / surfaces: NONE. (The "surface" hits are `enable_surface_validation`, a geometric normal-vs-ray dot-product gate, unrelated to texture hardware.)

Libraries:
- Thrust/CUB -> rocThrust/hipCUB (present in /opt/rocm/include; header drop-ins). Need `#define cub hipcub` in the compat header for the `cub::` spellings (cudaKDTree).
- cuRAND, cuSOLVER, cuSPARSE: NO actual compute calls. `check_error_curand.cu` includes `<curand.h>` only to name `CURAND_STATUS_*` enums; `check_error_cusolver.cu` includes `<cusparse.h>` only and uses integer literals. -> map to hipRAND/hipSPARSE headers (present) for the enum values, USE_HIP-guard any orphan enum case that has no hip equivalent (MPPI-Generic: hipRAND/hipFFT enum coverage is not 1:1). No cuBLAS/cuFFT/cuDNN.
- CUDA Graphs: cuda_graph.cu / cuda_graph_exec.cu / cuda_graph.cuh use `cudaGraphCreate/Destroy/AddDependencies/AddChildGraphNode/Instantiate/ExecDestroy/Launch`, `cudaStreamBeginCapture/EndCapture`, `cudaStreamCaptureModeGlobal`. HIP has hipGraph* 1:1. Watch the `#if CUDA_VERSION >= 13000` arity branch on `cudaGraphAddDependencies` (needs a HIP-correct path).

Streams / events:
- Streams everywhere (`cudaStreamCreateWithFlags(cudaStreamNonBlocking)`, Destroy, Synchronize); HIP 1:1.
- `easy_profiler_cuda.cu`: `cudaEvent*` Create/Record/Synchronize/ElapsedTime/Destroy; HIP 1:1 (amgcl note: the hip timing calls are nodiscard, but here they are consumed by `check_error <<`, so fine).

Memory:
- Stream-ordered async malloc pervasive: `cudaMallocAsync`/`cudaFreeAsync` (105 sites), `cudaMemcpyAsync`, `cudaMemsetAsync`, `cudaMemGetInfo`. gfx90a supports hipMallocAsync/stream-ordered mempools; map 1:1.
- Pinned host memory: `cudaMallocHost`/`cudaFreeHost` (cuda_buffer.cu), `cudaHostRegister`/`Unregister` (cuda_stream.cu RegisteredMemory). HIP 1:1.
- NO managed memory (`cudaMallocManaged`) and no `__managed__`.

API type bridge (pervasive):
- The project forward-declares `struct CUstream_st;` / `CUgraph_st` / `CUgraphNode_st` and uses `CUstream_st*` as the public stream type in 27 files incl. many installed headers (e.g. `GaussianVoxelMapGPU::insert(..., CUstream_st* stream)`). On HIP `hipStream_t` is `ihipStream_t*`, NOT `CUstream_st*`. The compat header must make `CUstream_st`/`CUgraph_st`/`CUgraphNode_st` resolve to the HIP opaque structs (e.g. `#define CUstream_st ihipStream_t` or a typedef bridge) so every `CUstream_st*` signature equals `hipStream_t`. This is the single most cross-cutting type fix; the porter must verify the exact HIP struct spellings and that the alias keeps the CUDA build untouched.

`cuda_malloc_async.hpp` gotcha: it does `#if (CUDA_VERSION < 11000) #define cudaMallocAsync ... cudaMalloc`. With `CUDA_VERSION` undefined on HIP this evaluates to `0 < 11000` = TRUE and would silently downgrade every async alloc to sync hipMalloc (loses stream-ordering, and the `(ptr,size,stream)` macro arity differs). Fix: in the HIP build either define `CUDA_VERSION>=11000` so the file is inert and `cudaMallocAsync`->`hipMallocAsync` via the compat header, or USE_HIP-guard the file body. Decide for hipMallocAsync (semantics match, gfx90a supports it).

## Risk list
- wave64 reduction (the usual point-cloud wall) -- LOW/ABSENT. The registration cost/Jacobian accumulation is NOT a hand-rolled warp reduction; it is `cub::DeviceReduce::Reduce` over `LinearizedSystem6 operator+`, fully library-managed and wave-size agnostic (contrast popsift/MPPI hand-unrolled warp tails, which is exactly what is NOT present here). Watch only: float-sum reduction ORDER differs from the CUDA build, so the converged transform will differ at ULP level -- fine against the test's coarse tolerances (rot < 0.015 rad, trans < 0.15 m), and any run-to-run nondeterminism in hipCUB segmented reduce is a determinism note, not a correctness fail.
- hipCUB reduce over a non-trivial value type (`LinearizedSystem6`, ~165 floats of Eigen matrices) with a custom `thrust::plus`. Two cudf-derived watch items: (1) rocPRIM/hipCUB DeviceReduce calls the binary functor through a CONST wrapper -- `thrust::plus` is const-correct so OK, but if any custom op is added it must have a const `operator()`. (2) the cudf dangling-reference class: none of these functors return a reference to a by-value/forwarded param (they return by value), so not exposed -- but the reviewer should confirm no `operator()` returns `T&`/`T&&` of a local.
- atomicMax on unsigned int (intensity) -- the cudaKDTree fault class: int/unsigned atomicMin/atomicMax are SILENTLY DROPPED on COARSE-GRAINED (hipMallocManaged) memory on gfx90a. Here the target is `cudaMallocAsync` DEVICE memory (fine-grained), where atomicMax works -- so expected OK, but it is a real watch item; if voxel intensities come back wrong/zero, emulate with an atomicCAS loop (unsigned compare). Validation `VoxelMapGPU`/`test_types` intensity checks will catch it.
- `CUstream_st`/`CUgraph_st`/`CUgraphNode_st` opaque-type bridge (see inventory) -- HIGH cross-cutting surface (27 files, public headers). Get the alias right once in the compat header or the GPU lib will not even compile.
- rocThrust namespace deltas (cupoch): `thrust::cuda::par` / `thrust::cuda::par_nosync` do NOT exist on the HIP device system; they are `thrust::hip::par` / `thrust::hip::par_nosync`. 12 call sites across gaussian_voxelmap_gpu*.cu and the inliers .cu. USE_HIP-guard or alias.
- `device_atomic_functions.h` include (gaussian_voxelmap_gpu.cu:14) -- CUDA-only toolkit header; route through the hip_compat shim dir.
- CUDA Graph `cudaGraphAddDependencies` arity (`#if CUDA_VERSION >= 13000`) -- ensure the HIP path picks the correct hipGraphAddDependencies signature; confirm hipStreamBeginCapture/EndCapture/hipGraphAddChildGraphNode exist in ROCm 7.2.1 (they do). NOTE: the Graph path may be a latent/unused code path at test time (the VGICP factor uses streams + hipCUB directly, not graph capture) -- it must COMPILE but may not be exercised by the GPU gate; flag for the validator.
- `cuda_malloc_async.hpp` CUDA_VERSION-undefined downgrade trap (see inventory).
- Eigen in device code -- LOW. Eigen 3.4.0 is installed and is used heavily in `__host__ __device__` functors. Eigen has a native HIP device path keyed on `__HIP_DEVICE_COMPILE__` and compiles under hipcc. The project does NOT define `__CUDA_ARCH__` for HIP, so the MPPI-Generic "Eigen takes the CUDA path -> needs EIGEN_NO_CUDA" trap should not trigger -- but if the compat header ends up defining `__CUDA_ARCH__`/`__CUDACC__` for any reason, add `EIGEN_NO_CUDA` to force Eigen's HIP path (MPPI-Generic). Verify a trivial Eigen `.inverse()` in a functor compiles before assuming clean.
- `--expt-relaxed-constexpr` / `-Xcudafe` nvcc flags (CMakeLists 88-106) are nvcc-only and live under the CUDA branch; the HIP branch must not inherit them (clang/hipcc rejects `-Xcudafe`). Keep them in the `enable_language(CUDA)` arm only.
- OOB neighbor reads / texture pitch / rule-of-five on texture handles -- N/A (no textures, no stencils; hash probing is bounded by `max_bucket_scan_count` and `% num_buckets`).
- GTSAM dependency build (see Open questions / Test plan) -- the real schedule risk, not a code risk.

## File-by-file change list (additive; NVIDIA path unchanged)
- NEW `include/gtsam_points/cuda/cuda_to_hip.h`: the compat header. Includes `<cstring>`,`<cstdlib>` then `<hip/hip_runtime.h>` under USE_HIP; aliases the cuda* runtime symbols used (cudaMalloc/Free[Async], cudaMemcpy[2D]Async, cudaMemsetAsync, cudaMemGetInfo, cudaMallocHost/cudaFreeHost, cudaHostRegister/Unregister, cudaStreamCreateWithFlags/Destroy/Synchronize/BeginCapture/EndCapture, cudaStreamNonBlocking, cudaStreamCaptureModeGlobal, cudaEvent* family, cudaGraph* family, cudaGetErrorName/String, cudaError_t/cudaSuccess/cudaStream_t, cudaMemcpyKind values, cudaGetDeviceCount/cudaGetDeviceProperties/cudaDeviceProp/cudaDeviceSynchronize, __float_as_uint, atomicCAS/Add/Max passthrough); `#define cub hipcub`; bridge `CUstream_st`/`CUgraph_st`/`CUgraphNode_st` to HIP opaque structs; alias `thrust::cuda::par[_nosync]` usage if not done at call sites; settle `CUDA_VERSION` for the cuda_malloc_async + cuda_graph guards.
- NEW `hip_compat/{cuda_runtime.h,cuda_runtime_api.h,cuda.h,device_atomic_functions.h}`: one-line forwarders to cuda_to_hip.h; on HIP include path only.
- EDIT `CMakeLists.txt`: add `option(BUILD_WITH_HIP ...)`; under it `enable_language(HIP)`, default-when-unset CMAKE_HIP_ARCHITECTURES, set `GTSAM_POINTS_USE_CUDA`/`-DBUILD_GTSAM_POINTS_GPU`, build the SAME gtsam_points_cuda source list with those `.cu` marked `LANGUAGE HIP` and `HIP_ARCHITECTURES` from the cache var, link `hip::host`, add the hip_compat include dir PRIVATE+BEFORE, force-include cuda_to_hip.h on the HIP target (`-include`), keep nvcc-only flags in the CUDA arm. Mirror the GPU-version `config.hpp.in` defines (GTSAM_POINTS_USE_CUDA stays the gate; consider also emitting a GTSAM_POINTS_USE_HIP for clarity but the existing macro name is what the tests use). `find_package(CUDAToolkit)` only in the CUDA arm; HIP arm uses `find_package(hip)`.
- EDIT (small, USE_HIP-guarded) likely needed: `src/gtsam_points/types/gaussian_voxelmap_gpu.cu` and `_funcs.cu`, `integrated_vgicp_derivatives_inliers.cu` -- `thrust::cuda::par[_nosync]` -> `thrust::hip::par[_nosync]` (or via a compat alias). `include/gtsam_points/cuda/cuda_malloc_async.hpp` -- guard the CUDA_VERSION<11000 body so HIP keeps hipMallocAsync. `src/gtsam_points/cuda/cuda_graph.cu` -- HIP-correct hipGraphAddDependencies arity.
- Possible: `check_error_curand.cu` / `check_error_cusolver.cu` -- include hiprand/hipsparse headers under USE_HIP and guard orphan enum cases.
The change set is intentionally small; the bulk is the compat header + CMake, mirroring colmap/cupoch.

## Build commands (gfx90a)
Prereq: GTSAM (see Test plan). Then, in projects/gtsam_points/src:
```
cmake -S . -B build_hip \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_WITH_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DBUILD_WITH_OPENMP=ON \
  -DBUILD_WITH_TBB=OFF \
  -DBUILD_TESTS=ON \
  -DBUILD_DEMO=OFF -DBUILD_EXAMPLE=OFF -DBUILD_TOOLS=OFF
cmake --build build_hip -j$(nproc)
```
(Followers reuse the same commit with only `-DCMAKE_HIP_ARCHITECTURES=gfx1100`/`gfx1151`; no source change because the target reads the cache var.)
A CPU-only compile smoketest in `rocm/dev-ubuntu-24.04:7.2.4-complete` is acceptable as a manual check only -- never the gate, and never wired into the fork's Actions (disable Actions on the fork).

## Test plan
Real GPU correctness gate (this is "validated on GPU"); run ctest SERIALLY on the assigned GPU (MPPI lesson: -j on one GPU induces false flaps), with WORKING_DIRECTORY at the repo root so `./data/...` resolves:
1. `test_matching_cost_factors` -- the headline gate. The `VGICP_CUDA_*` parametrized cases build `IntegratedVGICPFactorGPU` on 5 KITTI submaps (fixed-seed mt19937(8191) pose noise +/-0.1), run LevenbergMarquardtOptimizerExt (30 iters), and ASSERT the optimized relative pose converges to ground truth: rotation error < 0.015 rad AND translation error < 0.15 m (test_matching_cost_factors.cpp:227-228), across FORWARD/BACKWARD/UNARY/MULTI_FRAME and NONE/OMP parallelism. This exercises BOTH high-risk GPU paths end to end: GaussianVoxelMapGPU::insert (atomic hash insertion + atomic accumulation) and IntegratedVGICPDerivatives::linearize (hipCUB reduce over LinearizedSystem6). Passing these is the core proof.
2. `test_voxelmap` -- the `VoxelMapGPU` case (gated `GTSAM_POINTS_USE_CUDA`): `overlap_gpu` (hash lookup + hipCUB reduce) matches CPU overlap within 0.01 (line 238); all voxel means/covs finite (catches atomic-accumulation corruption); `merge_frames_gpu` path. Direct voxelmap correctness gate.
3. `test_types` -- the `GTSAM_POINTS_USE_CUDA` section (PointCloudGPU round-trips, GPU voxel data download/finite checks).
Define "validated on GPU" = (1) all `VGICP_CUDA_*` cases of test_matching_cost_factors PASS, (2) test_voxelmap `VoxelMapGPU` PASSES, (3) test_types GPU section PASSES, (4) determinism sanity: two runs of the VGICP_CUDA alignment agree to the test tolerance (ULP-level reduction-order drift is acceptable; gross nondeterminism is not), all under AMD_LOG_LEVEL>=2 confirming device dispatch.

Non-GPU regression set (MUST NOT regress): the CPU parametrizations of test_matching_cost_factors (ICP/GICP/VGICP), and the full non-GPU suite -- test_alignment, test_bundle_adjustment, test_colored_gicp, test_compact_mahalanobis, test_continuous_time, test_continuous_trajectory, test_global_registration, test_headers, test_kdtree, test_loam_factors, test_voxel_raycaster, plus the CPU portions of test_types/test_voxelmap. (test/pcl/* requires `-DBUILD_TESTS_PCL=ON` + PCL; out of scope unless PCL is trivially available.) Baseline these on the CPU-only build first so any regression is attributable to the HIP changes, then confirm they still pass in the HIP build (HIP changes are additive/guarded, so they should be identical).

## Staged strategy + hardest walls
1. Stand up GTSAM (dependency wall #1, schedule risk). Try `apt install libgtsam-dev` (Ubuntu 24.04 universe ships 4.2.0+dfsg, which satisfies `find_package(GTSAM 4.2 REQUIRED)` and provides gtsam + gtsam_unstable + GTSAMConfig.cmake; the `GTSAM_VERSION < 4.3` branch then selects the `gtsam4.2/` ISAM2 sources, which exist). If the apt 4.2.0 fails the unstable find or an API mismatch appears, build GTSAM 4.3a0 from source (borglab/gtsam @ 4.3a0, `-DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_WITH_TBB=OFF -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF`); 128 cores + ~1 TB RAM make this fast. Either way, install into `_deps/`-style prefix or system and point CMake at it. This is the porter's first task and the most likely time sink -- NOT a blocker.
2. Bring up the build: compat header + hip_compat shims + CMake HIP option; get gtsam_points_cuda to compile with `.cu` as LANGUAGE HIP. Expected friction (front-loaded): the `CUstream_st`/graph opaque-type bridge, `thrust::cuda::par`->`thrust::hip::par`, the `device_atomic_functions.h` shim, the cuda_malloc_async CUDA_VERSION trap, and confirming Eigen-in-device compiles under hipcc. Largely mechanical given no `__global__`/warp code.
3. GPU validation wall #2 (most likely real-correctness surface): the GPU voxelmap. Atomic hash insertion (atomicCAS/Add -- fine on device memory) + the atomicMax-unsigned intensity (watch the coarse-grained drop class, but memory is device-fine-grained so expected OK). The `VoxelMapGPU` finite/overlap checks will surface any accumulation bug.
4. GPU validation wall #3: VGICP convergence. Because the reduction is hipCUB-managed (no wave64 hand-reduction), the main risk is only ULP-level reduction-order drift vs the coarse convergence tolerance -- expected to pass. If it fails, localize with the cudf-style multi-path probe (library reduce vs raw hipCUB vs thrust::reduce over the same iterator) and check the functor const-/lifetime-correctness, not a warp bug.
Net: the classic point-cloud wave64 wall (Open3D slab-hashmap, registration warp reduction) does NOT apply here because the project never hand-codes warp cooperation -- it delegates to rocThrust/hipCUB. The schedule risk is GTSAM, and the correctness risk is the atomic voxelmap accumulation; both are bounded and covered by the shipped gtests.

## Open questions
- GTSAM: does apt `libgtsam-dev` 4.2.0 cleanly satisfy `find_package(GTSAM_UNSTABLE 4.2 REQUIRED)` and export the targets the CMake links (`gtsam`, `gtsam_unstable`)? If not, source-build 4.3a0. (Resolvable by the porter without asking; record the choice in notes.md.)
- Does the CUDA Graph capture path get exercised by any GPU test, or is it compile-only at validation time? (Affects how hard the validator leans on the graph code; the VGICP gate itself does not appear to use graph capture.)
- TBB: build with `-DBUILD_WITH_TBB=OFF` for the lead (the TBB test parametrizations self-skip when GTSAM_POINTS_USE_TBB is undefined). Revisit only if a TBB-specific path matters.
