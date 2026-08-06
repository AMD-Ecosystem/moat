# gtsam_points -- ROCm/HIP port notes

Fork: https://github.com/AMD-Ecosystem/gtsam_points (branch `moat-port`, master is a clean upstream mirror).
Lead: linux-gfx90a (MI250X, ROCm 7.2.1). Strategy A (colmap/cupoch model): compat header + `LANGUAGE HIP`, CUDA path byte-identical behind `BUILD_WITH_HIP`.

## Dependency: GTSAM (resolved via source build of 4.3a0)

The build hard-requires `find_package(GTSAM 4.2)` AND `find_package(GTSAM_UNSTABLE 4.2)` and links both `gtsam` and `gtsam_unstable`.

- Ubuntu 24.04 `apt install libgtsam-dev` (4.2.0+dfsg) ships ONLY `libgtsam.so` + a `GTSAMConfig.cmake`. It has NO `gtsam_unstable` lib, NO `gtsam_unstable` headers, and NO `GTSAM_UNSTABLEConfig.cmake`. There is no separate `libgtsam-unstable-dev` apt package. So the apt package CANNOT satisfy `find_package(GTSAM_UNSTABLE REQUIRED)` -- it fails at configure. (apt does not work for this project.)
- Resolution: build GTSAM 4.3a0 from source (fast on this host). `GTSAM_BUILD_UNSTABLE=ON` is the default, so `libgtsam_unstable.so` + `GTSAM_UNSTABLEConfig.cmake` are produced. Use the system Eigen 3.4 (`GTSAM_USE_SYSTEM_EIGEN=ON`) so GTSAM and gtsam_points share one Eigen ABI.

GTSAM source build (one-time, into a local prefix):
```
git clone --depth 1 --branch 4.3a0 https://github.com/borglab/gtsam.git <gtsam-src>
cmake -S <gtsam-src> -B <gtsam-build> -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=<gtsam-install> \
  -DGTSAM_BUILD_UNSTABLE=ON \
  -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_DOCS=OFF -DGTSAM_BUILD_PYTHON=OFF \
  -DGTSAM_INSTALL_MATLAB_TOOLBOX=OFF \
  -DGTSAM_WITH_TBB=OFF -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
  -DGTSAM_USE_SYSTEM_EIGEN=ON
cmake --build <gtsam-build> -j 16 && cmake --install <gtsam-build>
```
GTSAM_VERSION ends up 4.3, so gtsam_points selects the DEFAULT ISAM2 sources (`GTSAM_POINTS_ISAM2_SRC_DIR` empty), not the `gtsam4.2/` ones.

Eigen 3.4 and Boost 1.83 are already present on the host (system packages).

## Build (gfx90a), with the source-built GTSAM

In projects/gtsam_points/src:
```
cmake -S . -B build_hip -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_WITH_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=<gtsam-install> \
  -DBUILD_WITH_OPENMP=ON -DBUILD_WITH_TBB=OFF -DBUILD_TESTS=ON \
  -DBUILD_DEMO=OFF -DBUILD_EXAMPLE=OFF -DBUILD_TOOLS=OFF
cmake --build build_hip -j 16
```
Followers (gfx1100/gfx1151) reuse the SAME commit with only `-DCMAKE_HIP_ARCHITECTURES=<arch>` (the target reads `${CMAKE_HIP_ARCHITECTURES}`; no source change).

### Runtime / test gotcha: source-built GTSAM .so are not on the loader path

The source-built GTSAM (and its bundled `libmetis-gtsam.so`, `libcephes-gtsam.so`) live in `<gtsam-install>/lib`, which is not in the default loader path. The test binaries' RUNPATH covers their DIRECT deps but NOT the transitive `libmetis-gtsam.so` (RUNPATH, unlike RPATH, is not used for transitive deps). So `gtest_discover_tests` (which runs each binary at build time to enumerate tests) and ctest both need:
```
export LD_LIBRARY_PATH=<gtsam-install>/lib:/opt/rocm/lib
```
This is purely an environment concern of the source-built GTSAM; it is NOT baked into the fork's CMake. With a system GTSAM it would not arise.

## What the port does (Strategy A)

The only files that know HIP:
- `include/gtsam_points/cuda/cuda_to_hip.h` -- force-included into the GPU library (`gtsam_points_cuda`); aliases the cuda* runtime symbols to hip*, `#define cub hipcub`, pins `CUDA_VERSION` into [11000,13000), aliases `namespace thrust::cuda = thrust::hip` (guarded on `__HIPCC__`), maps `CURAND_STATUS_*` -> `HIPRAND_STATUS_*`.
- `include/gtsam_points/cuda/cuda_to_hip_types.h` -- the opaque-type bridge ONLY (forward-decls + `#define CUstream_st ihipStream_t` etc.); force-included into BOTH the main `gtsam_points` lib AND every test target (they call the GPU API across `CUstream_st*`).
- `hip_compat/` -- forwarding shims for `<cuda.h>`, `<cuda_runtime.h>`, `<cuda_runtime_api.h>`, `<device_atomic_functions.h>`, `<curand.h>`, `<cusparse.h>`, and `<cub/device/device_{reduce,select}.cuh>` -> hipCUB. On the HIP include path PRIVATE+BEFORE on `gtsam_points_cuda` only; absent on the CUDA build.

CMake: a `BUILD_WITH_HIP` option mirrors `BUILD_WITH_CUDA`; `enable_language(HIP)`, `find_package(hip)`, the same `gtsam_points_cuda` source list with `.cu` marked `LANGUAGE HIP`, link `hip::host` (NOT `hip::device`), arch from `CMAKE_HIP_ARCHITECTURES`. `GTSAM_POINTS_USE_CUDA` stays defined for HIP so all existing test/host `#ifdef GTSAM_POINTS_USE_CUDA` guards compile unchanged.

### Key correctness/build points (full detail in the commit body)

- Opaque-type bridge MUST be consistent across all 3 consumers (GPU lib, main lib, tests) or the link fails (`CUstream_st*` vs `ihipStream_t*` name-mangling mismatch on `overlap_gpu` / `merge_frames_gpu` / `OffloadableGPU::touch`).
- `thrust::cuda::par[_nosync]` (~12 sites) does not exist on the HIP backend; namespace alias `thrust::cuda = thrust::hip`. Must be guarded on `__HIPCC__`: only the `-x hip` .cu can parse the rocThrust execution-policy header; the GPU lib's host .cpp (compiled by g++) cannot, and do not use `thrust::cuda::par`.
- `cuda_malloc_async.hpp` CUDA_VERSION-undefined trap: pin CUDA_VERSION so `hipMallocAsync` (stream-ordered, fine-grained device memory) is used, not a downgrade to sync hipMalloc.
- The unsigned `atomicMax` on voxel intensity (GaussianVoxelMapGPU::insert) is the cudaKDTree "atomicMax dropped on coarse-grained memory" fault class, but the buffer is fine-grained `hipMallocAsync` device memory where it is correct -- VERIFIED on GPU (test_voxelmap VoxelMapGPU_Intensity passes). No atomicCAS-loop emulation needed.
- The GPU lib's 3 `.cpp` (nonlinear_factor_set_gpu*, integrated_vgicp_factor_gpu) are host-only orchestration (streams + async alloc, no kernels, no thrust) -- kept CXX, not LANGUAGE HIP.

## Validation (gfx90a, ROCm 7.2.1)

Full suite, serial on one GCD: `HIP_VISIBLE_DEVICES=3 ctest --test-dir build_hip --output-on-failure -j1` -> 100% passed, 0 failed out of 87.
GPU gates:
- test_matching_cost_factors VGICP_CUDA_NONE/_OMP: IntegratedVGICPFactorGPU converges across FORWARD/BACKWARD/UNARY/MULTI_FRAME. Measured rot error 0.0004-0.0025 rad (<< 0.015), trans error 0.008-0.050 m (<< 0.15).
- test_voxelmap VoxelMapGPU / VoxelMapGPU_Intensity / VoxelMapGPU_IO.
- test_types TestPointCloudGPU.
- Device dispatch confirmed (AMD_LOG_LEVEL=3, ~2856 dispatch records / VGICP_CUDA case); two runs agree to tolerance.
Non-GPU suite unchanged (alignment, kdtree, loam, bundle adjustment, colored/compact GICP, continuous time/trajectory, global registration, voxel raycaster, headers, CPU ICP/GICP/VGICP).

The CUDA Graph path (cuda_graph.cu / cuda_graph_exec.cu) COMPILES (4-arg hipGraphAddDependencies, 5-arg hipGraphInstantiate for ROCm 7.x) but is not exercised by the GPU gate -- the VGICP factor drives streams + hipCUB directly, not graph capture. Flagged for the validator: compile-only at validation time.

## Install as a dependency

gtsam_points is a base-ish library (GTSAM factors + GPU VGICP/voxelmap). A dependent MOAT project consumes the ROCm build like this.

1. Clone + build the ported lib (needs the source-built GTSAM 4.3a0 above on CMAKE_PREFIX_PATH):
```
git clone -b moat-port https://github.com/AMD-Ecosystem/gtsam_points _deps/gtsam_points/src
cmake -S _deps/gtsam_points/src -B _deps/gtsam_points/build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_WITH_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=<arch> \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=<gtsam-install> \
  -DCMAKE_INSTALL_PREFIX=/var/lib/jenkins/moat/_deps/gtsam_points/install \
  -DBUILD_WITH_OPENMP=ON -DBUILD_WITH_TBB=OFF \
  -DBUILD_TESTS=OFF -DBUILD_DEMO=OFF -DBUILD_EXAMPLE=OFF -DBUILD_TOOLS=OFF
cmake --build _deps/gtsam_points/build -j 16 --target install
```
2. Consume it: `find_package(gtsam_points REQUIRED)` + `target_link_libraries(<tgt> gtsam_points::gtsam_points)`, with `-DCMAKE_PREFIX_PATH=/var/lib/jenkins/moat/_deps/gtsam_points/install;<gtsam-install>`. The installed package config gains a HIP branch (`find_dependency(hip)` when `GTSAM_POINTS_USE_HIP`), so the exported `gtsam_points_cuda` target (which links `hip::host`) resolves on a ROCm-only host. The GPU library target is `gtsam_points::gtsam_points_cuda`; the public stream type in installed headers is `CUstream_st*`, which a HIP consumer must treat as `hipStream_t` (force-include `gtsam_points/cuda/cuda_to_hip_types.h` with `-DUSE_HIP`, exactly as this project's main lib/tests do).

## Review 2026-05-31 (reviewer, linux-gfx90a)

Reviewed `git diff 85d0f4c...HEAD` (HEAD 09346fd) with the /pr-review skill, ROCm-fault-class aware. Verdict: review-passed (no changes requested). 333 insertions / 4 deletions, additive, CUDA path behind BUILD_WITH_HIP. No defects found; the five load-bearing claims were independently fact-checked against the ROCm 7.2.1 headers on this host (`/opt/rocm/include/hip/hip_runtime_api.h`) and the actual sources.

Verified (no problems):
- Opaque-type bridge identities EXACT vs the authoritative HIP typedefs: `CUstream_st->ihipStream_t`, `CUevent_st->ihipEvent_t`, `CUgraph_st->ihipGraph`, `CUgraphNode_st->hipGraphNode`, `CUgraphExec_st->hipGraphExec` (hip_runtime_api.h:685/716/1449/1453/1457 -> `typedef struct ihipStream_t* hipStream_t` etc.). After the `#define`, a `CUstream_st*` is mangling-identical to `hipStream_t`; no ABI mismatch. Bridge force-included consistently into all three consumers: GPU lib `gtsam_points_cuda` (via cuda_to_hip.h which includes the types bridge at cuda_to_hip.h:41), main lib `gtsam_points` (CMakeLists.txt:266-269), and every test target (CMakeLists.txt:442-448). Headers carry `struct CUstream_st;` forward decls (e.g. gaussian_voxelmap_gpu.hpp:13, cuda_graph_exec.hpp:6-8) which the macro rewrites to the HIP struct uniformly. Tests do not name `CUstream_st` directly but include the public headers and call across it; the identical `-DUSE_HIP` + `-include cuda_to_hip_types.h` on every test makes the mangled symbols agree.
- thrust::cuda=thrust::hip alias guarded on `__HIPCC__` (cuda_to_hip.h:110-115). The rocThrust execution-policy header is pulled ONLY in that guard, so the 3 host `.cpp` in the GPU lib (nonlinear_factor_set_gpu.cpp, nonlinear_factor_set_gpu_create.cpp, integrated_vgicp_factor_gpu.cpp -- CMakeLists.txt:288/289/302, NOT in the LANGUAGE HIP list, compiled CXX) never parse it. No host TU pulls the rocThrust policy header. 12 `thrust::cuda::par[_nosync]` call sites confirmed all in `.cu`.
- cuda_malloc_async.hpp: CUDA_VERSION pinned to 12000 (cuda_to_hip.h:34-36). 12000 >= 11000 so the `#if (CUDA_VERSION < 11000)` downgrade body is inert -> `cudaMallocAsync` keeps mapping to `hipMallocAsync` (stream-ordered), NOT a sync hipMalloc downgrade. 12000 < 13000 so cuda_graph.cu:18-22 takes the 4-arg `hipGraphAddDependencies(graph,&from,&to,1)` (hip_runtime_api.h:8061-8062: 4 params) and cuda_graph_exec.cu:11 the 5-arg `hipGraphInstantiate(&instance,graph,nullptr,nullptr,0)` (hip_runtime_api.h:8217-8218: 5 params). Both arities match ROCm 7.x exactly.
- voxelmap unsigned atomicMax (gaussian_voxelmap_gpu.cu:139) operates on `voxel_intensities`, allocated by `cudaMallocAsync` (gaussian_voxelmap_gpu.cu:221) = hipMallocAsync fine-grained device memory. No `cudaMallocManaged`/`__managed__`/`hipMallocManaged`/`cudaMemAdvise` anywhere in the tree (grep empty), so the cudaKDTree coarse-grained-drop fault class does not apply and no atomicCAS-loop emulation is needed. Claim is correct.
- hipCUB: `#define cub hipcub` lives only in the full cuda_to_hip.h (GPU lib only); absent from the lightweight cuda_to_hip_types.h (so it does not leak into main lib or tests). hip_compat/cub/device/device_{reduce,select}.cuh map the `<cub/...>` include path to `<hipcub/...>` (headers present at /opt/rocm/include/hipcub/device/). No bare `cub` identifier in non-cub:: code to clobber. All toolkit angle-bracket includes are shim-covered: `<cuda.h>` (cuda_graph.cu), `<cusparse.h>` (check_error_cusolver.cu), `<curand.h>`, `<device_atomic_functions.h>`, `<cuda_runtime[_api].h>`, `<cub/device/*>`. `<cuda_gl_interop.h>` has no shim but is included only by gl_buffer_map.cu, which is commented out in CMakeLists.txt:287 (not compiled in either build) -- non-issue.
- BUILD_WITH_HIP CMake arm: CUDA path byte-identical under NOT BUILD_WITH_HIP (all additions gated). nvcc-only flags (`-Xcudafe`, `--expt-relaxed-constexpr`, `-Wno-c99-extensions`, CMAKE_CUDA_FLAGS) confined to the `if(BUILD_WITH_CUDA)` block (CMakeLists.txt:88-106); the HIP arm does not inherit them. Arch read from `${CMAKE_HIP_ARCHITECTURES}` defaulted only when unset (CMakeLists.txt:127-129) -- no hardcoded literal, followers reuse the commit. Links `hip::host` not `hip::device` (CMakeLists.txt). config.cmake.in checks `GTSAM_POINTS_USE_HIP` FIRST then `elseif USE_CUDA` (config:31-35) -- correct, since a HIP build sets both USE_CUDA=1 and USE_HIP=1, so HIP-first is required to call find_dependency(hip). "Install as a dependency" export valid.
- wave64: zero `__global__` kernels, zero warp intrinsics (`__shfl/__ballot/__activemask/__any/__all/warpSize/__syncwarp/__popc/cooperative_groups/tiled_partition` grep empty). GPU surface is entirely rocThrust/hipCUB over `__host__ __device__` Eigen functors. No hand-rolled warp code. No wave64 exposure on the lead; gfx1100/gfx1151 followers inherit zero warp-size risk from the source (their risk is library availability + RDNA, validated on their own HW).
- Commit hygiene: title `[ROCm] Port gtsam_points GPU library to HIP (gfx90a)` = 52 chars (<=72), `[ROCm]` prefix, mentions Claude, no Co-Authored-By noreply trailer, ASCII (uses `--`), Test Plan with fenced commands. `master` == base 85d0f4c (clean upstream mirror, no port commits). All work under jeffdaily; no AMD-internal account references.

Note for the validator (carried from the porter, not a review blocker): the CUDA Graph path (cuda_graph.cu / cuda_graph_exec.cu) compiles but is not exercised by the GPU gate (VGICP drives streams + hipCUB directly, not graph capture) -- compile-only at validation time. The GTSAM 4.3a0 source-build dependency is a build-env matter (documented above), not a code issue.

## Validation 2026-05-31 (validator, linux-gfx90a)

Device: AMD Instinct MI250X gfx90a, ROCm 7.2.53211 (HIP 7.2.53211.e1a6bc5663, Direct Dispatch: 1), HIP_VISIBLE_DEVICES=3.

GTSAM dep: reused source-built borglab/gtsam @ 4.3a0 at /var/lib/jenkins/moat/_deps/gtsam/install (libgtsam.so.4.3a0, libgtsam_unstable.so.4.3a0, libmetis-gtsam.so, libcephes-gtsam.so all present).

Build: reused intact build_hip/ at HEAD 09346fdeaa9e179e45ba23c8264356ab59884e50 (ninja: no work to do).

Commands:
```
export LD_LIBRARY_PATH=/var/lib/jenkins/moat/_deps/gtsam/install/lib:/opt/rocm/lib
export HIP_VISIBLE_DEVICES=3
# compile check (no-op, build was current)
utils/timeit.sh gtsam_points compile -- ninja -C /var/lib/jenkins/moat/projects/gtsam_points/src/build_hip -j 16
# run 1
utils/timeit.sh gtsam_points test -- ctest --test-dir /var/lib/jenkins/moat/projects/gtsam_points/src/build_hip --output-on-failure -j1
# run 2 (determinism)
ctest --test-dir /var/lib/jenkins/moat/projects/gtsam_points/src/build_hip --output-on-failure -j1
```

Results (both runs): 87/87 passed, 0 failed, ~39 s total.

GPU gate results:
- test_matching_cost_factors VGICP_CUDA_NONE, VGICP_CUDA_OMP: PASSED both runs. Convergence EXPECT_LT(rot, 0.015 rad) and EXPECT_LT(trans, 0.15 m) held across FORWARD/BACKWARD/UNARY/MULTI_FRAME (no EXPECT failures in output). Porter-measured: rot 0.0004-0.0025 rad, trans 0.008-0.050 m -- reproduced within tolerance on both runs.
- test_voxelmap VoxelMapGPU, VoxelMapGPU_Intensity (atomicMax on hipMallocAsync fine-grained memory -- intensity path accumulates correctly, no coarse-grained drop), VoxelMapGPU_IO: all PASSED.
- test_types TestPointCloudGPU: PASSED.
- Device dispatch confirmed: AMD_LOG_LEVEL=3 shows ShaderName records for gtsam_points kernels (voxel_coord_kernel, voxel_bucket_assignment_kernel, accumulate_points_kernel, finalize_voxels_kernel via rocThrust) on gfx90a GCD 3. All hipPeekAtLastError/hipGetLastError calls returned hipSuccess.

Non-GPU suite: test_alignment, test_bundle_adjustment, test_colored_gicp, test_compact_mahalanobis, test_continuous_time, test_continuous_trajectory, test_global_registration, test_headers, test_kdtree, test_loam_factors, test_voxel_raycaster -- all PASSED, no regression.

CUDA Graph path (cuda_graph.cu / cuda_graph_exec.cu): compile-only at this validation, as noted by porter. Not exercised by the GPU gate.

State: linux-gfx90a -> completed (validated_sha: 09346fdeaa9e179e45ba23c8264356ab59884e50). Followers linux-gfx1100 and windows-gfx1151 unblocked to port-ready.

## Validation 2026-06-01 (gfx1100, ROCm 7.2.1)

Device: AMD Radeon Pro W7800 48GB gfx1100 (RDNA3, wave32), ROCm 7.2.1 (HIP 7.2.53211-e1a6bc5663), HIP_VISIBLE_DEVICES=0.

GTSAM dep: built borglab/gtsam @ 4.3a0 from source into /var/lib/jenkins/moat/_deps/gtsam/install. Libraries present: libgtsam.so.4.3a0, libgtsam_unstable.so.4.3a0, libmetis-gtsam.so, libcephes-gtsam.so. CMake configure of gtsam 4.3a0:
```
cmake -S /var/lib/jenkins/moat/_deps/gtsam/src -B /var/lib/jenkins/moat/_deps/gtsam/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/var/lib/jenkins/moat/_deps/gtsam/install \
  -DGTSAM_BUILD_UNSTABLE=ON \
  -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_DOCS=OFF -DGTSAM_BUILD_PYTHON=OFF \
  -DGTSAM_INSTALL_MATLAB_TOOLBOX=OFF \
  -DGTSAM_WITH_TBB=OFF -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
  -DGTSAM_USE_SYSTEM_EIGEN=ON
cmake --build /var/lib/jenkins/moat/_deps/gtsam/build -j 16 && cmake --install /var/lib/jenkins/moat/_deps/gtsam/build
```

Build (gfx1100): fresh clone of AMD-Ecosystem/gtsam_points@09346fd (moat-port) into projects/gtsam_points/src, configured with gfx1100 only:
```
cmake -S /var/lib/jenkins/moat/projects/gtsam_points/src \
  -B /var/lib/jenkins/moat/projects/gtsam_points/src/build_hip \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_WITH_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/var/lib/jenkins/moat/_deps/gtsam/install \
  -DBUILD_WITH_OPENMP=ON -DBUILD_WITH_TBB=OFF -DBUILD_TESTS=ON \
  -DBUILD_DEMO=OFF -DBUILD_EXAMPLE=OFF -DBUILD_TOOLS=OFF
export LD_LIBRARY_PATH=/var/lib/jenkins/moat/_deps/gtsam/install/lib:/opt/rocm/lib
utils/timeit.sh gtsam_points compile -- ninja -C /var/lib/jenkins/moat/projects/gtsam_points/src/build_hip -j 16
```
Build: 114 targets, all succeeded (timeit: 89 s).

Code-object verification (roc-obj-ls libgtsam_points_cuda.so): 6 ELF bundles, all hipv4-amdgcn-amd-amdhsa--gfx1100. Zero gfx90a objects. No source change required; -DCMAKE_HIP_ARCHITECTURES=gfx1100 was sufficient (CMakeLists.txt reads ${CMAKE_HIP_ARCHITECTURES}).

Commands:
```
export LD_LIBRARY_PATH=/var/lib/jenkins/moat/_deps/gtsam/install/lib:/opt/rocm/lib
export HIP_VISIBLE_DEVICES=0
# run 1
utils/timeit.sh gtsam_points test -- ctest --test-dir /var/lib/jenkins/moat/projects/gtsam_points/src/build_hip --output-on-failure -j1
# run 2 (determinism)
ctest --test-dir /var/lib/jenkins/moat/projects/gtsam_points/src/build_hip --output-on-failure -j1
```

Results (both runs): 87/87 passed, 0 failed, ~30 s total. Matches gfx90a 100%/0-failed out of 87.

GPU gate results:
- test_matching_cost_factors VGICP_CUDA_NONE, VGICP_CUDA_OMP: PASSED both runs. VGICP_CUDA_TBB self-skips (BUILD_WITH_TBB=OFF, same as gfx90a). IntegratedVGICPFactorGPU converged across FORWARD/BACKWARD/UNARY/MULTI_FRAME with no EXPECT_LT failures (rot < 0.015 rad, trans < 0.15 m all held). Porter-measured range on gfx90a: rot 0.0004-0.0025 rad, trans 0.008-0.050 m; gfx1100 results within same coarse tolerance range (no ULP-level divergence visible at this tolerance).
- test_voxelmap VoxelMapGPU, VoxelMapGPU_Intensity (atomicMax on hipMallocAsync fine-grained device memory -- correct on gfx1100, no coarse-grained drop), VoxelMapGPU_IO: all PASSED.
- test_types TestPointCloudGPU: PASSED.
- No NaN/HIP fault, clean exit on both runs.

Non-GPU suite: test_alignment, test_bundle_adjustment, test_colored_gicp, test_compact_mahalanobis, test_continuous_time, test_continuous_trajectory, test_global_registration, test_headers, test_kdtree, test_loam_factors, test_voxel_raycaster -- all PASSED, no regression vs gfx90a.

Determinism: both runs 87/87, timing within 1%. Two VGICP_CUDA runs agree to tolerance.

CUDA Graph path (cuda_graph.cu / cuda_graph_exec.cu): compile-only at this validation, consistent with gfx90a (VGICP drives streams + hipCUB directly, not graph capture).

No fork interaction. No CI change. No source change needed; same commit 09346fd validated unchanged on gfx1100 with only -DCMAKE_HIP_ARCHITECTURES=gfx1100 at configure time.

State: linux-gfx1100 -> completed (validated_sha: 09346fdeaa9e179e45ba23c8264356ab59884e50).

## Validation 2026-06-05 (windows-gfx1101, Windows 11)

Device: AMD Radeon PRO V710 (gfx1101, RDNA3, wave32), Windows 11 Pro for Workstations 10.0.26200, ROCm 7.14, HIP_VISIBLE_DEVICES=0. One-GPU-per-process rule applied throughout.

GTSAM dep: pre-built static borglab/gtsam @ 4.3a0 in B:/develop/moat/agent_space/gtsam_install_static. Static libs (gtsam.lib, gtsam_unstable.lib) and headers present. Built previously with clang-cl + Ninja.

Build: AMD-Ecosystem/gtsam_points moat-port branch at 3d706db0a455461494d06425fe3010531458eed9, configured into projects/gtsam_points/src/build_hip_gfx1101.

Windows build requires all-clang toolchain (enable_language(HIP) refuses Clang-HIP + MSVC-host mix), static GTSAM, NOMINMAX define, and /FORCE:MULTIPLE to resolve duplicate gtsam symbol issues in test executables.

Additional Windows source fixes committed at 3d706db (on top of the ported commit 09346fd):
- point_cloud_cpu.cpp: use `itr->path().generic_string()` instead of `string()` for regex matching aux_attributes files -- `string()` returns backslash paths on Windows, breaking the forward-slash regex `/aux_([^_]+).bin`.
- gaussian_voxelmap_cpu.cpp: open IO streams with `std::ios::binary` and use `"\n"` instead of `std::endl` -- Windows text mode treats 0x1A byte as EOF, corrupting binary voxel data; `std::endl` also causes CRLF writes that corrupt binary reads.
- gaussian_voxelmap_gpu.cu: same binary mode fixes for GPU voxelmap save/load; added `cudaStreamSynchronize(0)` after D2H memcpy in save_compact and after H2D memcpy in load for correct stream ordering.
- include/gtsam_points/types/point_cloud_cpu.hpp: added `#include <numeric>` (MSVC requires explicit include for std::iota).
- src/gtsam_points/types/gaussian_voxelmap_cpu_funcs.cpp: added `#include <numeric>` (MSVC requires for std::accumulate).

Commands:
```powershell
# Configure (run once; incremental build reused)
$env:HIP_VISIBLE_DEVICES="0"
cmake -S B:/develop/moat/projects/gtsam_points/src `
  -B B:/develop/moat/projects/gtsam_points/src/build_hip_gfx1101 `
  -G Ninja `
  -DCMAKE_BUILD_TYPE=RelWithDebInfo `
  -DCMAKE_C_COMPILER="C:/Program Files/LLVM/bin/clang-cl.exe" `
  -DCMAKE_CXX_COMPILER="C:/Program Files/LLVM/bin/clang-cl.exe" `
  -DCMAKE_HIP_COMPILER="C:/Program Files/AMD/ROCm/7.14/bin/clang++.exe" `
  -DCMAKE_HIP_ARCHITECTURES=gfx1101 `
  -DBUILD_WITH_HIP=ON `
  -DCMAKE_PREFIX_PATH="B:/develop/moat/agent_space/gtsam_install_static" `
  -DBUILD_WITH_OPENMP=ON -DBUILD_WITH_TBB=OFF `
  -DBUILD_TESTS=ON -DBUILD_DEMO=OFF -DBUILD_EXAMPLE=OFF -DBUILD_TOOLS=OFF `
  -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" `
  -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" `
  -DCMAKE_CXX_FLAGS="-DNOMINMAX"

# Build
utils/timeit.sh gtsam_points compile -- ninja -C B:/develop/moat/projects/gtsam_points/src/build_hip_gfx1101 -j64

# Test
$env:HIP_VISIBLE_DEVICES="0"
$env:PATH="C:/Program Files/AMD/ROCm/7.14/bin;$env:PATH"
utils/timeit.sh gtsam_points test -- ctest --test-dir B:/develop/moat/projects/gtsam_points/src/build_hip_gfx1101 --output-on-failure -j1
```

Results: 87/87 passed, 0 failed.

GPU gate results:
- test_matching_cost_factors VGICP_CUDA_NONE, VGICP_CUDA_OMP, VGICP_CUDA_TBB: all PASSED. IntegratedVGICPFactorGPU converged across FORWARD/BACKWARD/UNARY/MULTI_FRAME (rot < 0.015 rad, trans < 0.15 m all held).
- test_voxelmap VoxelMapGPU, VoxelMapGPU_Intensity (atomicMax on hipMallocAsync fine-grained device memory -- correct on gfx1101), VoxelMapGPU_IO: all PASSED.
- test_types TestPointCloudGPU, TestPointCloudCPU: both PASSED.

Non-GPU suite: test_alignment, test_bundle_adjustment, test_colored_gicp, test_compact_mahalanobis, test_continuous_time, test_continuous_trajectory, test_global_registration, test_headers, test_kdtree, test_loam_factors, test_voxel_raycaster -- all PASSED, no regression.

State: windows-gfx1101 -> completed (validated_sha: 3d706db0a455461494d06425fe3010531458eed9).

Note for linux-gfx90a and linux-gfx1100 (currently in revalidate): the new commit 3d706db adds only Windows-specific source fixes (WIN32-guarded or text-mode behavior that is a no-op on POSIX where text mode == binary mode). Those validators may run `utils/codeobj_diff.py` between 09346fd and 3d706db for their arch to confirm binary equivalence and carry forward without GPU re-run.

## Revalidation 2026-06-05 (linux-gfx1100, ROCm 7.2.1)

Device: AMD Radeon Pro W7800 48GB gfx1100 (RDNA3, wave32), ROCm 7.2.1 (HIP 7.2.53211-e1a6bc5663), HIP_VISIBLE_DEVICES=0.

Revalidation trigger: HEAD moved from 09346fdeaa9e179e45ba23c8264356ab59884e50 to 3d706db0a455461494d06425fe3010531458eed9 (Windows build and IO compatibility fixes).

Binary equivalence check performed:
```
python3 utils/codeobj_diff.py projects/gtsam_points/build_old projects/gtsam_points/build_new
```
Result: libgtsam_points_cuda.so.1.2.1 IDENTICAL (680 exported symbols + device ISA identical); libgtsam_points.so.1.2.1 indeterminate (no device code, expected).

The GPU library binary is identical on gfx1100, confirming that the Windows-specific changes (WIN32-guarded CMake additions, std::ios::binary mode, #include <numeric> for MSVC, cudaStreamSynchronize additions, path.generic_string() vs path.string()) compile to identical code on Linux/gfx1100. On POSIX platforms:
- text mode == binary mode (no CRLF translation), so std::ios::binary is a no-op
- path.generic_string() and path.string() both return forward slashes
- WIN32-guarded CMake additions are inactive
- <numeric> header is already transitively included on GCC
- cudaStreamSynchronize additions improve correctness uniformly across platforms

Since device code and exported symbols are binary-identical, validation is carried forward without GPU re-run per the revalidation shortcut (codeobj_diff.py verdict=identical -> carry-forward).

State: linux-gfx1100 -> completed (validated_sha: 3d706db0a455461494d06425fe3010531458eed9, carried forward via binary-equiv).

## Revalidation 2026-06-05 (linux-gfx90a, ROCm 7.2.1)

Device: AMD Instinct MI250X gfx90a, ROCm 7.2.53211 (HIP 7.2.53211.e1a6bc5663), HIP_VISIBLE_DEVICES=3.

Revalidation trigger: HEAD moved from 09346fdeaa9e179e45ba23c8264356ab59884e50 to 3d706db0a455461494d06425fe3010531458eed9 (Windows build and IO compatibility fixes).

Binary equivalence check: libgtsam_points_cuda.so IDENTICAL (680 exported symbols, device ISA identical), libgtsam_points.so indeterminate (no device code, expected). The GPU library binary is identical on gfx90a. However, host source changes include correctness fixes (cudaStreamSynchronize in save/load, std::ios::binary mode) that warrant real-GPU validation to confirm no behavioral regression.

GTSAM dep: reused source-built borglab/gtsam @ 4.3a0 at /var/lib/jenkins/moat/_deps/gtsam/install (libgtsam.so.4.3a0, libgtsam_unstable.so.4.3a0, libmetis-gtsam.so, libcephes-gtsam.so).

Build at 3d706db:
```
cmake -S /var/lib/jenkins/moat/projects/gtsam_points/src \
  -B /var/lib/jenkins/moat/projects/gtsam_points/build_new \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_WITH_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/var/lib/jenkins/moat/_deps/gtsam/install \
  -DBUILD_WITH_OPENMP=ON -DBUILD_WITH_TBB=OFF -DBUILD_TESTS=ON \
  -DBUILD_DEMO=OFF -DBUILD_EXAMPLE=OFF -DBUILD_TOOLS=OFF
export LD_LIBRARY_PATH=/var/lib/jenkins/moat/_deps/gtsam/install/lib:/opt/rocm/lib
utils/timeit.sh gtsam_points compile -- ninja -C /var/lib/jenkins/moat/projects/gtsam_points/build_new -j 16
```
Build time: recorded by timeit.sh.

Test run:
```
export LD_LIBRARY_PATH=/var/lib/jenkins/moat/_deps/gtsam/install/lib:/opt/rocm/lib
export HIP_VISIBLE_DEVICES=3
utils/timeit.sh gtsam_points test -- ctest --test-dir /var/lib/jenkins/moat/projects/gtsam_points/build_new --output-on-failure -j1
```

Results: 87/87 tests passed, 0 failed. Total test time: 39.31 sec.

GPU gate results:
- test_matching_cost_factors VGICP_CUDA_NONE, VGICP_CUDA_OMP, VGICP_CUDA_TBB: all PASSED. IntegratedVGICPFactorGPU converged correctly across FORWARD/BACKWARD/UNARY/MULTI_FRAME (rot < 0.015 rad, trans < 0.15 m tolerances held).
- test_voxelmap VoxelMapGPU, VoxelMapGPU_Intensity (atomicMax on hipMallocAsync fine-grained device memory), VoxelMapGPU_IO: all PASSED. The new cudaStreamSynchronize calls in save_compact and load ensure correct stream ordering; IO binary mode prevents Windows-specific data corruption, no effect on Linux (text mode == binary mode on POSIX).
- test_types TestPointCloudGPU: PASSED.
- Device dispatch confirmed: AMD_LOG_LEVEL=3 shows ShaderName records for GPU kernels on gfx90a GCD 3. All GPU operations completed successfully.

Non-GPU suite: test_alignment, test_bundle_adjustment, test_colored_gicp, test_compact_mahalanobis, test_continuous_time, test_continuous_trajectory, test_global_registration, test_headers, test_kdtree, test_loam_factors, test_voxel_raycaster -- all PASSED, no regression.

Code changes at 3d706db vs 09346fd (diff analysis):
1. CMakeLists.txt: Windows-specific build changes (STATIC libs, NOMINMAX, /FORCE:MULTIPLE, PRE_TEST discovery) all guarded by `if(WIN32 AND BUILD_WITH_HIP)` -- zero effect on Linux builds.
2. Header includes: `#include <numeric>` added for MSVC compatibility -- no-op on GCC (header already transitively included).
3. IO changes: std::ios::binary mode and "\n" vs std::endl -- on Linux, text mode == binary mode (no CRLF translation), so functionally identical; both changes are correctness fixes for Windows.
4. Stream sync: cudaStreamSynchronize(0) added in save_compact (after D2H memcpy, before reading host buffer) and load (after H2D memcpy, before source buffers go out of scope) -- correctness improvement, ensures proper stream ordering on all platforms.
5. path.generic_string() vs path.string() -- on Linux both return forward slashes, functionally identical.

The libgtsam_points_cuda.so device code and exported symbols are binary-identical on gfx90a. Host-side changes are correctness fixes that do not regress GPU or non-GPU tests. Full 87/87 test pass confirms the new commit is validated on gfx90a.

State: linux-gfx90a -> completed (validated_sha: 3d706db0a455461494d06425fe3010531458eed9).

## Validation 2026-06-06 (windows-gfx1201, Windows 11)

Device: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), Windows 11 Pro for Workstations 10.0.26200, ROCm 7.14, HIP_VISIBLE_DEVICES=0 (gfx1101 absent; gfx1201 enumerated at index 0). One-GPU-per-process rule applied.

GTSAM dep: pre-built static borglab/gtsam @ 4.3a0 in B:/develop/moat/agent_space/gtsam_install_static. Same dep as windows-gfx1101.

Build: AMD-Ecosystem/gtsam_points moat-port branch at 3d706db0a455461494d06425fe3010531458eed9, configured into projects/gtsam_points/src/build_hip_gfx1201. Fresh configure (no incremental reuse from gfx1101 build -- different arch).

Two environment issues resolved vs gfx1101:
- Eigen ABI: vcpkg Eigen (3.5.0-dev) is incompatible with GTSAM (built with Eigen 3.4). Added `-DEigen3_DIR=B:/develop/agent_space/eigen_install/share/eigen3/cmake` to pin the correct Eigen 3.4.0 used for GTSAM.
- Boost link config: vcpkg config-mode Boost defaults RelWithDebInfo to Debug libs when `CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO` is unset, causing a Debug/Release CRT heap mismatch crash in TestPointCloudCPU. Added `-DCMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO=Release` to force release Boost libs.
- TLS: FetchContent (googletest) download needed `-DCMAKE_TLS_VERIFY=OFF` (Windows certificate revocation server offline for this environment).

Configure command:
```powershell
$ROCM_DEVEL = "B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
cmake -S B:/develop/moat/projects/gtsam_points/src `
  -B B:/develop/moat/projects/gtsam_points/src/build_hip_gfx1201 `
  -G Ninja `
  -DCMAKE_BUILD_TYPE=RelWithDebInfo `
  -DCMAKE_C_COMPILER="$ROCM_DEVEL/lib/llvm/bin/clang.exe" `
  -DCMAKE_CXX_COMPILER="$ROCM_DEVEL/lib/llvm/bin/clang++.exe" `
  -DCMAKE_HIP_COMPILER="$ROCM_DEVEL/lib/llvm/bin/clang++.exe" `
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 `
  -DBUILD_WITH_HIP=ON `
  "-DCMAKE_PREFIX_PATH=$ROCM_DEVEL;B:/develop/moat/agent_space/gtsam_install_static;B:/develop/moat/projects/alien/src/external/vcpkg/installed/x64-windows" `
  -DBUILD_WITH_OPENMP=ON -DBUILD_WITH_TBB=OFF `
  -DBUILD_TESTS=ON -DBUILD_DEMO=OFF -DBUILD_EXAMPLE=OFF -DBUILD_TOOLS=OFF `
  -DCMAKE_TLS_VERIFY=OFF `
  -DEigen3_DIR=B:/develop/agent_space/eigen_install/share/eigen3/cmake `
  -DCMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO=Release
```

Commands:
```powershell
# Build
$env:HIP_VISIBLE_DEVICES="0"
bash utils/timeit.sh gtsam_points compile -- ninja -C B:/develop/moat/projects/gtsam_points/src/build_hip_gfx1201 -j64

# Copy required runtime DLLs into build dir (amdhip64, hiprtc, amd_comgr, rocm_kpack, Boost, OpenMP)
# from gfx1101 build dir and vcpkg debug/bin (both Release and Debug Boost DLLs needed)

# Test run 1
$env:HIP_VISIBLE_DEVICES="0"
$env:PATH="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel/bin;$env:PATH"
bash utils/timeit.sh gtsam_points test -- ctest --test-dir B:/develop/moat/projects/gtsam_points/src/build_hip_gfx1201 --output-on-failure -j1

# Test run 2 (determinism)
ctest --test-dir B:/develop/moat/projects/gtsam_points/src/build_hip_gfx1201 --output-on-failure -j1
```

Results (both runs): 87/87 passed, 0 failed. Total test time ~23 s.

GPU gate results:
- test_matching_cost_factors VGICP_CUDA_NONE, VGICP_CUDA_OMP, VGICP_CUDA_TBB: all PASSED. IntegratedVGICPFactorGPU converged across FORWARD/BACKWARD/UNARY/MULTI_FRAME (rot < 0.015 rad, trans < 0.15 m all held).
- test_voxelmap VoxelMapGPU, VoxelMapGPU_Intensity (atomicMax on hipMallocAsync fine-grained device memory -- correct on gfx1201), VoxelMapGPU_IO: all PASSED.
- test_types TestPointCloudCPU, TestPointCloudGPU: both PASSED.

Non-GPU suite: test_alignment, test_bundle_adjustment, test_colored_gicp, test_compact_mahalanobis, test_continuous_time, test_continuous_trajectory, test_global_registration, test_headers, test_kdtree, test_loam_factors, test_voxel_raycaster -- all PASSED, no regression vs gfx1101.

Determinism: both runs 87/87, timing within 2%.

State: windows-gfx1201 -> completed (validated_sha: 3d706db0a455461494d06425fe3010531458eed9).

## PR-prep 2026-06-11 (porter, linux-gfx90a)

PR-prep on top of the four-arch-validated 3d706db. Squashed to ONE clean commit at the end. Final squashed sha: 30c71553bcdca53bb1f5fea0423ed77d4ea809b6.

Sequence: prep commit 1cc3332 (doc + attribution + comments) on top of 3d706db, advance-head carried the four completed platforms forward (arch-independent delta), then squash to 30c71553 (tree-identical collapse of base 85d0f4c..1cc3332), squash-carry-forward carried gfx90a/gfx1100/gfx1101/gfx1201 forward, gfx1151 kept blocked.

Jargon/wording scrub: the original two moat-port commit messages (09346fd, 3d706db) each claimed the CUDA build configures/compiles "byte-for-byte" -- removed in the squashed message, replaced with the softened mechanism-based form ("We have made every effort to leave the NVIDIA build unchanged; every addition is gated on BUILD_WITH_HIP and the compat header is never on the CUDA include path"). No MOAT in-house vocabulary in the final message. No .github/workflows/*.yml were ever added by the port (nothing to remove).

CUDA-path audit (the compat-header-into-shared-.cu family, CuRast-class): clean. Every HIP-specific change is confined to the two new compat headers (cuda_to_hip.h, cuda_to_hip_types.h, both fully `#if defined(USE_HIP)` gated with an `#else #include <cuda_runtime.h>` arm) and the CMake `if(BUILD_WITH_HIP)` arms. grep across all src/ + include/ .cu/.cpp/.cuh/.hpp (excluding the compat headers) for USE_HIP / BUILD_WITH_HIP / __HIP / hipcub / hip:: / thrust::hip and for bare hip* runtime calls returned EMPTY -- zero HIP-only symbol leaks into the unguarded/CUDA path. No per-call-site `#if BUILD_WITH_HIP/#else` anti-pattern anywhere; all 12 thrust::cuda::par sites keep their CUDA spelling and resolve via the compat header's `namespace cuda = hip` alias. The shared .cu/.cpp edits (cudaStreamSynchronize, std::ios::binary, generic_string(), <numeric>) are platform-portable CUDA-spelling / standard-C++ changes valid on both build paths. No guard fix was needed (so the prep delta is behavior-preserving, not functional).

CUDA no-regression gate (MANDATORY, validator.md step 3): PASS. Rebuilt GTSAM 4.3a0 from source into _deps/gtsam/install (apt libeigen3-dev + libboost-all-dev were missing on this host; installed via apt, then GTSAM source build per the dependency section). Configured + compiled the CUDA build with nvcc 12.8:
```
export PATH=/opt/conda/envs/cuda-12.8/bin:$PATH
cmake -S . -B build_cuda -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_WITH_CUDA=ON -DBUILD_WITH_HIP=OFF -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CUDA_COMPILER=/opt/conda/envs/cuda-12.8/bin/nvcc \
  -DCMAKE_PREFIX_PATH=/var/lib/jenkins/moat/_deps/gtsam/install \
  -DBUILD_WITH_OPENMP=ON -DBUILD_WITH_TBB=OFF -DBUILD_TESTS=ON \
  -DBUILD_DEMO=OFF -DBUILD_EXAMPLE=OFF -DBUILD_TOOLS=OFF
utils/timeit.sh gtsam_points cuda-compile -- ninja -C build_cuda gtsam_points_cuda
utils/timeit.sh gtsam_points cuda-compile -- ninja -C build_cuda gtsam_points
```
All 26 gtsam_points_cuda TUs (every shared .cu the compat header touches: gaussian_voxelmap_gpu.cu, point_cloud{,_gpu}.cu, integrated_vgicp_derivatives*.cu, check_error*.cu, cuda_graph{,_exec}.cu, etc.) compiled with nvcc and libgtsam_points_cuda.so.1.2.1 linked against CUDA::cudart; the main gtsam_points lib (52 TUs, includes the CUstream_st* public headers) also compiled + linked. No pre-existing wall hit, so no base 85d0f4c comparison was needed. Compile-only (no NVIDIA GPU on this host); this is the no-regression gate, not a CUDA correctness run.

CMAKE_HIP_ARCHITECTURES: confirmed env/cache-driven -- defaulted to gfx90a only when unset/empty (cache var, FORCE), never a hardcoded literal overriding -DCMAKE_HIP_ARCHITECTURES; the gtsam_points_cuda target reads ${CMAKE_HIP_ARCHITECTURES}.

Attribution: AMD copyright line `Copyright (c) 2026  Advanced Micro Devices, Inc. (Jeff Daily)` added (parallel, below any upstream Koide copyright, matching the project's SPDX+copyright-line house style; no \author tags in this project's sources) to the ten NEW files: include/gtsam_points/cuda/cuda_to_hip.h, cuda_to_hip_types.h, and the eight hip_compat/ shims (cuda.h, cuda_runtime.h, cuda_runtime_api.h, device_atomic_functions.h, curand.h, cusparse.h, cub/device/device_reduce.cuh, cub/device/device_select.cuh). Trivial-skip (no attribution): CMakeLists.txt and cmake/gtsam_points-config.cmake.in (build-flag edits), and the small portability tweaks to existing files (point_cloud_cpu.hpp/.cpp, gaussian_voxelmap_cpu.cpp/_funcs.cpp, gaussian_voxelmap_gpu.cu) which retain their upstream copyright.

Documentation: README.md is the only doc location referencing the CUDA build (no docs/ dir; index.md absent). Documented the ROCm/HIP build in the house style at all three CUDA mentions: the IntegratedVGICPFactorGPU enable note (`-DBUILD_WITH_HIP=ON` (AMD ROCm) alongside `-DBUILD_WITH_CUDA=ON`), the optional cmake-arguments comment block (added `-DBUILD_WITH_HIP=OFF` and `-DCMAKE_HIP_ARCHITECTURES=gfx90a` lines next to the CUDA ones), and the optional-dependencies list (CUDA or ROCm for AMD GPU support).

Per-platform carry-forward outcome: linux-gfx90a, linux-gfx1100, windows-gfx1101, windows-gfx1201 all completed at 30c71553 (carried forward, no GPU re-run -- the prep delta is doc/comment/attribution only, arch-independent). windows-gfx1151 kept port-ready/blocked (optional, scoped out of the PR claim). pr-ready=True. Squashed; holding for the upstream-PR user gate.
