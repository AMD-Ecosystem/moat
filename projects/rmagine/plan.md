# rmagine -- ROCm/HIP porting plan (first platform: linux-gfx90a, MI250X, ROCm 7.2.1)

## Project
- Name: rmagine -- a 3D range-sensor simulation library for robots (Embree CPU / OptiX NVIDIA / Vulkan ray-casting backends plus a CUDA compute backend).
- Upstream: https://github.com/uos/rmagine  (default branch `main`, version 2.4.2, BSD-3-Clause)
- Fork: https://github.com/AMD-Ecosystem/rmagine, branch `moat-port`, off upstream main 6b93e86.
- Clone: projects/rmagine/src/
- Port target: `rmagine_core` + `rmagine_cuda` (Stage 1, the CUDA compute backend), then a HIPRT ray-tracing backend `rmagine_hiprt` (Stage 2).

### How this plan came to be written about rmcl
This project was scaffolded as **rmcl** (https://github.com/uos/rmcl), a ROS 2 mobile-robot-localization stack that is a thin layer over rmagine. The analysis below started from rmcl and concluded that rmcl's portable GPU compute IS rmagine's CUDA backend, so the real porting target is rmagine; the record was renamed accordingly (open question 1, now answered). rmcl is retained here as the downstream consumer that motivated the port and as the source of the scope decisions: everything in a section marked "downstream (rmcl)" is OUT of scope for this project and would be a separate one. Where the analysis is about rmagine it has been left as written and verified against the fork.

## TL;DR decision
Read through rmcl, the GPU ray-mesh intersection has TWO distinct GPU paths and they decide port-vs-defer differently. Both of them live in rmagine:

1. The RMCL / Monte-Carlo (global localization) GPU path is NVIDIA-OptiX-gated -> B7 cluster, DEFER. The `rmcl_localization` node only builds `if(RMCL_OPTIX AND RMCL_EMBREE)` (rmcl_ros/CMakeLists.txt:709) and hard-includes `PCDSensorUpdaterOptix.hpp`; its per-beam likelihood evaluation is an OptiX program (`__raygen__`/`__closesthit__`/`__miss__` + `optixTrace`, src/rmcl/optix/BeamEvaluateProgram.cu) compiled to PTX. No ROCm equivalent without a HIPRT or hand-written HIP-BVH reimplementation of rmagine's OptiX backend. HIPRT is NOT installed on this host and NOT in ROCm 7.2.1 (confirmed: `import hiprt` fails; only hiprtc = runtime compilation is present). Defer pending jeff's HIPRT decision.

2. The MICP-L (pose tracking) GPU path plus rmcl's own custom CUDA compute is PORTABLE (and CONFIRMED OptiX-optional: every OptiX reference in micp_localization is `#ifdef RMCL_OPTIX`-guarded with a graceful 'backend optix not compiled' runtime error, and the backend is chosen at runtime via the `corr_backend` param -- so MICP-L builds and runs without OptiX) (Strategy A hipify + a wave64 reduction fix), BUT it is gated on rmagine providing a non-OptiX GPU ray-casting backend. The ray casting itself lives entirely in rmagine, not rmcl; rmcl only ships the correspondence/statistics reductions, the particle motion/resample kernels, and the ROS orchestration.

Therefore the REAL porting target is rmagine, and the highest-value, immediately-validatable deliverable is **rmagine_core + rmagine_cuda** (the CUDA compute backend), which carries a genuine wave64 correctness bug and has NO ROCm path today. rmcl's own `.cu` (particle_motion/resampling) ports trivially on top. Recommendation: PORT the rmagine_cuda compute path (correctness-first, GPU-validated via rmagine's own CUDA gtests on gfx90a), and DEFER both the OptiX RT backend (B7) and the Vulkan RT backend (no Vulkan SDK on this host; CUDA-external-memory interop). See "Disposition".

## Existing AMD support
- NONE via HIP/ROCm. rmagine README "Backends" table lists exactly Embree (CPU), OptiX (NVIDIA only), Vulkan (cross-vendor RT). There is no `rmagine_hip` / ROCm component and no `USE_HIP` anywhere in rmcl or rmagine. This is NOT the "already-supported" skip class.
- AMD is reachable ONLY through rmagine's Vulkan RT backend today (cross-vendor), and that path on rmcl additionally requires `rmagine::vulkan-cuda-interop` (CUDA external-memory import) + `rmagine::cuda`, i.e. it is wired through CUDA, not a standalone AMD path. Per the guide ("AMD via OpenCL/Vulkan/SYCL with no HIP path -> a ROCm/HIP port still adds value"), a HIP port of the CUDA compute remains valuable.
- Decision: improvable/greenfield HIP port of the rmagine CUDA compute backend (rmagine_core + rmagine_cuda), with rmcl's `.cu` on top. Abandoned-port: none. Mature-ROCm: none.

## Build classification: pure CMake (cmake), NOT a pytorch extension
- Evidence: no `find_package(Torch)`, no `torch.utils.cpp_extension`/`CUDAExtension`, no setup.py/pyproject anywhere. rmcl is ament_cmake (rmcl_ros/CMakeLists.txt:39 `find_package(ament_cmake REQUIRED)`); rmcl core is plain CMake (rmcl/CMakeLists.txt). rmagine is plain CMake (rmagine/CMakeLists.txt, `add_subdirectory(src/rmagine_*)`). CUDA enters via `enable_language(CUDA)` + legacy `find_package(CUDA)` fallback (rmcl/CMakeLists.txt:217-243, rmagine src/rmagine_cuda/CMakeLists.txt:42-65).
- ext_type set to `cmake` in upstream.json and status.json. -> Strategy A.

## Port strategy: A (compat-header + LANGUAGE HIP), applied to rmagine_cuda first
Rationale: a standalone CMake project with `.cu` sources + CUDA runtime + cuRAND. The colmap/cupoch model fits: gate language on a `USE_HIP` option, mark the existing `.cu` `LANGUAGE HIP`, and add a single cuda->hip compat header for the symbols actually used (small surface; see inventory). Keep the NVIDIA path byte-identical. Do NOT rename `.cu` files; do NOT touch host C++.

Port ORDER (dependency-first; each step independently GPU-validatable):
1. rmagine_core (header-heavy math; no CUDA) -- builds with plain g++; prerequisite for everything. Check it compiles under the ROCm toolchain unchanged (it is CPU C++ + Eigen/Boost/assimp/TBB).
2. rmagine_cuda (THE deliverable) -- Strategy A: compat header + LANGUAGE HIP + cuRAND->hipRAND swap + the wave64 warp-reduction fix. Validate with rmagine's own ctest CUDA suite on gfx90a.
3. rmcl core `rmcl-cuda` (CorrespondencesCUDA.cpp -> rm::statistics_p2l) -- builds once rmagine_cuda is a HIP lib; it is pure host C++ calling rmagine. No `.cu` of its own.
4. rmcl_ros `.cu` (particle_motion.cu, resampling.cu) -- trivial hipify (curand swap; the reduction here is already wave-safe). NOTE this needs ROS 2, which is NOT on this host; treat steps 3-4 as a secondary (rmcl-layer) milestone behind the rmagine_cuda milestone, build them only if a ROS 2 / ament environment is provisioned (see Open questions).

A correctness-first MECHANICAL port is the right first step (no AMD-native rewrite needed): these are simple reductions and elementwise kernels, not CUTLASS/CuTe/Hopper-tuned GEMM/attention. No port-vs-rewrite tension here.

## CUDA surface inventory

### rmagine_cuda (the port target) -- src/rmagine_cuda/
- 49 `__global__` kernels across: math/statistics.cu (sum/p2p/p2l/objectwise-p2l cross-statistics reductions), math/math_batched.cu (chunk_sums / chunk_sums_masked / cov reductions), math/memory_math.cu, math/linalg.cu (SVD/cholesky helpers used by registration), map/mesh_preprocessing.cu (face-normal precompute), noise/*.cu (Gaussian/uniform sensor noise), types/MemoryCuda.cu, util/cuda/random.cu.
- Runtime API (small, plain): cudaMalloc/Free, cudaMallocHost/FreeHost, cudaMallocManaged, cudaMemcpy{,Async}, cudaMemset, cudaStreamCreate{,WithFlags}/Destroy/Synchronize, cudaGetDeviceProperties, cudaGetDeviceCount, cudaRuntimeGetVersion/DriverGetVersion, cudaGetLastError, cudaGetErrorString/Name, cudaError_t/cudaSuccess, cudaStream_t, cudaDeviceProp. All 1:1 hip* (compat header).
- cuRAND: curandState, curand_init, curand_uniform, curand_normal (device, util/cuda/random.cuh + noise kernels). -> hipRAND (hiprand_kernel.h; the device API is the same spelling under hipify mappings: curandState->hiprandState etc., or alias in the compat header).
- cuBLAS/cuSOLVER: referenced ONLY as link targets in CMake (CUDA::cublas, CUDA::cusolver) -- the SVD/cholesky in linalg.cu are HAND-WRITTEN device kernels, not library calls (confirm by grep: no cusolverDn*/cublas* call sites in the .cu). So no hipBLAS/hipSOLVER functional dependency expected; drop the link refs on HIP unless a call site surfaces. (Verify during port.)
- NO textures, NO surfaces, NO CUB/Thrust, NO cooperative groups, NO managed-memory atomics in the hot path. No `__shfl`/`__ballot`/`__activemask`.
- Context util (util/cuda/CudaContext.cpp) is HEAVILY on the CUDA DRIVER API (CONFIRMED call sites): cuInit, cuCtxCreate, cuCtxDestroy, cuCtxGetCurrent/SetCurrent/PushCurrent, cuCtxGetDevice -- all have hipCtx* equivalents. BUT cuCtxSetSharedMemConfig(CU_SHARED_MEM_CONFIG_{FOUR,EIGHT}_BYTE_BANK_SIZE) (CudaContext.cpp:119-123) has NO HIP equivalent (shared-mem bank-size config is NVIDIA-specific; CDNA has no configurable LDS bank width). USE_HIP-guard out just those two calls; map the rest cuCtx*->hipCtx*. The explicit primary-context dance is likely unnecessary on HIP (runtime makes one lazily), so the fallback is to relax it under USE_HIP. This is now a REQUIRED edit, not just possible.

### rmagine_cuda warp-size fault (DECISIVE correctness item -- MPPI-Generic class, PORTING_GUIDE line 188)
statistics.cu and math_batched.cu repeat the classic CUDA warp-synchronous unrolled reduction tail in ~6 kernels:
```
for(unsigned int s = blockSize/2; s > 32; s >>= 1) { if(tid<s) sdata[tid]+=sdata[tid+s]; __syncthreads(); }  // tree stops AT 32
if(tid < blockSize/2 && tid < 32) warpReduce<blockSize>(sdata, tid);                                          // 32-lane unsynced tail
// warpReduce: volatile T* sdata; sdata[tid]+=sdata[tid+32]; +16; +8; +4; +2; +1;  -- NO __syncwarp
```
On gfx90a (wave64) the low 32 lanes of a 64-lane wavefront are NOT guaranteed lockstep across those unsynced `+32..+1` steps, so the reduction is WRONG and run-to-run non-deterministic (manifests as NaN / garbage covariance -- exactly what rmagine's `cuda_math_statistics` test guards). Fix per the guide: on HIP drop the warp-synchronous tail and let the `__syncthreads()`-synchronized tree run to size 1 (change loop bound `s > 32` -> `s > 0` and remove the `warpReduce` call, USE_HIP-guarded; CUDA path unchanged). Same add order, block-wide barrier, correct on any wave size. Verify with a fixed-input determinism check (two runs bit-identical) plus the no-NaN assertion.
- Secondary: `sdata[tid] *= 0.0` / `sdata[tid].sum = 0.0` zero-init of a `CrossStatistics` struct -- confirm the struct's `operator*=`/field init is defined and produces a true zero (not NaN) on HIP; a NaN seed would poison the new full-tree reduction just as it does the warp tail.

### downstream (rmcl) -- own CUDA (src/rmcl_ros/src/rmcl/) -- OUT of scope for this project
- particle_motion.cu: 1 trivial elementwise kernel (`particle_move_and_forget_kernel`), no warp ops, no shared mem. Pure hipify.
- resampling.cu: 3 kernels -- `init_curand_kernel` (curand_init), `simple_stats_kernel<blockSize>` (a block reduction that runs the `__syncthreads()` tree ALL THE WAY to s>0 -- ALREADY wave-safe, no warp tail), `gladiator_resample_kernel` (curand/curand_normal per-particle MCL resample). cuRAND -> hipRAND swap; otherwise clean. wave64-safe as written.
- rmcl-cuda (CorrespondencesCUDA.cpp): host C++ calling `rm::statistics_p2l` from rmagine; no kernels.
- MICP*SensorCUDA.cpp (5 files): ROS orchestration (subscriptions, TF, buffer resize) calling rmagine simulators + rmcl-cuda correspondences. No kernels. Builds only with ROS 2.

### OptiX surface (DEFER -- B7) -- do NOT port
- rmcl: src/rmcl/registration/RCCOptix.cpp (+ RCCOptix.hpp), rmcl_ros/src/rmcl/optix/{BeamEvaluateProgram.cu, eval_modules, eval_program_groups, eval_pipelines}.cpp, PCDSensorUpdaterOptix.cpp, optix/* headers. ~43 OptiX symbols in rmagine_optix (optixAccelBuild/optixTrace/optixModuleCreate/SBT/pipelines; 60 files). BeamEvaluateProgram.cu is a raygen/closesthit/miss PTX program. This is the OptiX->HIPRT wall; no mechanical port.

### Vulkan surface (DEFER on this host) -- not portable here
- rmcl: RCCVulkan.cpp/.hpp (Spherical/Pinhole/O1Dn/OnDn) using rmagine SphereSimulatorVulkan etc. + `MemoryView<..., VULKAN_AS_CUDA>` interop views. rmagine_vulkan (GLSL/SPIR-V RT shaders) + rmagine_vulkan_cuda_interop (cudaImportExternalMemory / cudaExternalMemoryGetMappedBuffer over OpaqueFd). HIP HAS hipImportExternalMemory/hipExternalMemoryGetMappedBuffer (the interop could in principle become Vulkan<->HIP), but: (a) NO Vulkan SDK on this host (vulkaninfo/glslangValidator/glslc all absent), (b) it is a separate RT backend with its own shaders. The Vulkan path is the cross-vendor route to AMD ray casting, but it is out of scope for a CUDA-to-HIP compute port and unvalidatable here. Note it in plan; do not attempt now.

## Risk list
- WAVE64 warp-synchronous reduction in rmagine statistics.cu / math_batched.cu (HIGH, the decisive correctness fix). See inventory; fix = full `__syncthreads()` tree on HIP. Determinism + no-NaN check is the proof.
- rmagine_cuda uses the LEGACY FindCUDA path (`find_package(CUDA)` fallback) but ALSO the modern `enable_language(CUDA)` + CUDAToolkit (rmagine_cuda/CMakeLists.txt:42-65). Since it reaches `enable_language(CUDA)` and lists explicit `RMAGINE_CUDA_SRCS`, the colmap recipe applies: under USE_HIP `enable_language(HIP)` + `set_source_files_properties(${RMAGINE_CUDA_SRCS} PROPERTIES LANGUAGE HIP)`. Watch the `CUDA_NVCC_FLAGS_*` cache blocks (no-ops on HIP). If a `cuda_add_library`/`cuda_compile_ptx` is used anywhere in the CUDA component, bridge with the cupoch shim -- but grep shows rmagine_cuda uses plain add_library, only the OptiX PTX path uses cuda_compile_ptx (deferred).
- CUDA driver-API context management (CudaContext.cpp): CONFIRMED uses cuInit/cuCtxCreate/Destroy/Get/Set/Push-Current/GetDevice (all map to hipCtx*) AND cuCtxSetSharedMemConfig/CU_SHARED_MEM_CONFIG_* which has NO HIP analogue -- USE_HIP-guard those two out (CDNA LDS bank width is not configurable). Map the rest in the compat header; relax the explicit-context logic to runtime-lazy under USE_HIP if init-order bugs appear. (MEDIUM.)
- cuRAND device header (`curand_kernel.h`) -> hiprand_kernel.h. hipRAND's device generator state/byte-layout differs from cuRAND; results will NOT be bit-identical to a CUDA run (expected). Validate by STATISTICS (the MCL/noise distributions), not bitwise vs CUDA. curandState size/alignment in `MemoryView<curandState,...>` must follow the hiprandState type (compat alias so the buffer sizing is right).
- `*= 0.0` / field zero-init of CrossStatistics structs as a reduction seed (see inventory) -- NaN-seed hazard under the new full-tree reduction.
- Rule-of-five on CUDA stream/context handles (CudaStream.cpp): default-constructed/double-destroyed stream handles fault on AMD where CUDA tolerates them (colmap class). Give explicit `=0` init + guarded destroy if the wrappers lack it.
- cuBLAS/cuSOLVER: CONFIRMED link-refs only -- grep of all rmagine_cuda src+include found ZERO cusolver*/cublas* call sites; linalg.cu's SVD/cholesky are self-contained device kernels. Safe to drop the CUDA::cublas/CUDA::cusolver link entries on HIP; no hipBLAS/hipSOLVER needed. (LOW.)
- Out-of-bounds neighbor reads: low risk (no stencils); the reductions are bounds-checked (`if(globId+... < N)`). mesh_preprocessing.cu face-normal gather indexes face vertex ids -- check it clamps/validates (AMD faults a 1-past read).
- NO project test suite in rmcl itself (BUILD_TESTS option exists but there are zero gtests / ament_add_test). Validation must come from rmagine's tests + a localization run -> see Test plan.
- ENVIRONMENT (HIGH for the rmcl LAYER, not for rmagine_cuda): no ROS 2 (`/opt/ros` absent, no colcon), no Embree, no Vulkan SDK on this host. The rmagine_cuda milestone is fully buildable/validatable WITHOUT any of these (rmagine builds standalone with -DRMAGINE_*_DISABLE=ON for embree/optix/vulkan). The rmcl ROS nodes + MICP CUDA sensors need a ROS 2 + Embree provisioning step (apt: ros-jazzy-desktop or a docker, libembree-dev). Flag, do not block: deliver and validate rmagine_cuda first.

## File-by-file change list (rmagine_cuda milestone -- the validated deliverable)
Worked on the rmagine fork (AMD-Ecosystem/rmagine @ moat-port), since rmcl's portable GPU compute IS rmagine's CUDA backend. (rmcl carries no `.cu` other than the trivial particle_motion/resampling, which are part of the secondary rmcl-layer milestone.)
- NEW src/rmagine_cuda/include/rmagine/util/cuda/cuda_to_hip.h -- the single compat header: `#if defined(USE_HIP)||defined(__HIP_PLATFORM_AMD__)` include <hip/hip_runtime.h> (+ <hiprand/hiprand_kernel.h> for the device cuRAND) and `#define` the ~30 cuda* runtime symbols + curand* device symbols + cuCtx* driver symbols used (table above) to their hip*/hiprand* spellings; `#else` include the real CUDA headers. Force-include on every HIP TU via CMAKE_HIP_FLAGS `-include` so it precedes any use.
- EDIT src/rmagine_cuda/CMakeLists.txt -- add `option(USE_HIP ...)`; under USE_HIP `enable_language(HIP)`, default `CMAKE_HIP_ARCHITECTURES` to gfx90a ONLY when unset (never a literal -- followers pass -DCMAKE_HIP_ARCHITECTURES), `set_source_files_properties(${RMAGINE_CUDA_SRCS} PROPERTIES LANGUAGE HIP)`, set target HIP_ARCHITECTURES from the cache var, link hip::host (+ hiprand) instead of CUDA::cudart/curand, force-include the compat header. NVIDIA branch untouched.
- EDIT src/rmagine_cuda/src/math/statistics.cu -- USE_HIP-guard the warp-reduction tail: full `__syncthreads()` tree (s>0), drop warpReduce on HIP, in `sum_kernel`, `statistics_p2p_kernel`, `statistics_p2l_kernel`, `statistics_objectwise_p2l_kernel` (every kernel using the warpReduce pattern). Verify CrossStatistics zero-seed.
- EDIT src/rmagine_cuda/src/math/math_batched.cu -- same warp-tail fix in chunk_sums_kernel / chunk_sums_masked_kernel / the cov reductions (4+ sites).
- POSSIBLE EDIT src/rmagine_cuda/src/util/cuda/CudaContext.cpp -- guard/relax driver-API context creation under USE_HIP if hipCtx* mapping misbehaves.
- POSSIBLE EDIT src/rmagine_cuda/src/util/cuda/random.cu + noise/*.cu -- only if hipRAND device spellings need explicit aliasing beyond the compat header.
- EDIT top-level rmagine CMakeLists.txt -- thread the USE_HIP option down; build with embree/optix/vulkan/ouster DISABLEd for the headless ROCm validation (`-DRMAGINE_EMBREE_DISABLE=ON -DRMAGINE_OPTIX_DISABLE=ON -DRMAGINE_VULKAN_DISABLE=ON -DRMAGINE_VULKAN_CUDA_INTEROP_DISABLE=ON`), keeping core+cuda.
- NEW projects/rmagine/notes.md: the build recipe + the wave64 fix, and a "## Install as a dependency" section, which IS required here -- rmagine is a dependency provider (rmcl consumes it via find_package), and the exported CMake package config is part of the port. That section must be verified by actually consuming an install from a separate project, with ROCm off `PATH`, and by including the public headers from a plain C++ TU. Both of those were found broken after the first four platform validations passed; see notes.md.

downstream (rmcl) milestone (secondary, a separate project; needs ROS 2 + Embree): compat header is unnecessary for rmcl's two `.cu` if built through rmagine's HIP toolchain; mark particle_motion.cu/resampling.cu LANGUAGE HIP and alias curand* -> hiprand*. CorrespondencesCUDA.cpp + MICP*CUDA.cpp are host C++, no change beyond linking the HIP rmagine. The OptiX targets (rmcl-optix, rmcl_ros_optix, rmcl_localization) stay OFF on AMD (rmagine built without optix => those targets are not created; gates already exist).

## Build commands (gfx90a)
rmagine_cuda milestone (standalone, no ROS/Embree/Vulkan needed):
```
# deps present: ROCm 7.2.1, Eigen3, TBB. Need: Boost, assimp (apt: libboost-all-dev libassimp-dev) for rmagine_core.
cd <rmagine fork>
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DRMAGINE_EMBREE_DISABLE=ON -DRMAGINE_OPTIX_DISABLE=ON \
  -DRMAGINE_VULKAN_DISABLE=ON -DRMAGINE_VULKAN_CUDA_INTEROP_DISABLE=ON \
  -DRMAGINE_OUSTER_DISABLE=ON \
  -DRMAGINE_BUILD_TESTS=ON
cmake --build build -j
```
Follower arch: same command, only `-DCMAKE_HIP_ARCHITECTURES=gfx1100` (or gfx1151) -- no source change (the wave64 fix is wave-agnostic: a full __syncthreads tree is correct on wave32 too).

rmcl-layer milestone (only if ROS 2 jazzy + Embree provisioned):
```
# in a colcon workspace with the HIP-ported rmagine installed/overlaid
colcon build --packages-select rmcl_msgs rmcl rmcl_ros \
  --cmake-args -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DDISABLE_OPTIX=ON -DDISABLE_VULKAN=ON
```

## Test plan (real GPU; the gate)
PRIMARY GATE (self-contained, real gfx90a, no ROS/mesh/Vulkan/OptiX) -- rmagine's own CUDA ctest suite, link-only against rmagine::cuda:
```
cd <rmagine fork>/build && ctest --output-on-failure -R '^cuda_'
# cuda_math, cuda_memory, cuda_memory_slicing, cuda_math_svd, cuda_math_statistics, cuda_math_reduction
```
- cuda_math_statistics (tests/cuda/math_statistics.cpp) and cuda_math_reduction directly exercise the wave64-sensitive `statistics_p2l`/`statistics_p2p`/`sum_kernel`/`warpReduce` kernels rmcl depends on; they assert NO NaN in the cross-statistics means/covariance. This is exactly where the warp-reduction bug surfaces. cuda_math_svd covers linalg.cu.
- Determinism proof for the reduction fix: run cuda_math_statistics (or a small fixed-input harness over statistics_p2l) twice and assert bit-identical CrossStatistics across runs (the fix removes the nondeterminism). Confirm device execution with AMD_LOG_LEVEL=3 (named kernel dispatch).
- Numeric correctness: compare GPU CrossStatistics (mean/covariance/n_meas) against a CPU reference computed from the same point set to ~float eps; SVD/cholesky outputs against a CPU/Eigen reference.

SECONDARY GATE (rmcl layer; ONLY if ROS 2 + Embree are provisioned) -- a localization run converging to ground truth:
- rmcl ships no unit tests, so use the maintainer's hands-on examples: https://github.com/amock/rmcl_examples (sample meshes + recorded sensor trajectories + ground truth). Run `micp_localization_node` (MICP-L pose tracking) on a sample mesh + a recorded sensor bag with a known initial pose, and assert the published `rmcl/pose` (TF map->base) converges to and tracks the ground-truth trajectory within tolerance (translation ~cm, rotation ~deg, no divergence over the run). rmagine also ships sample meshes (dat/sphere.ply, dat/triangle.ply, dat/*.dae) for a minimal SphereSimulator+correction smoke run.
- Backend for that GPU run on AMD: requires a working non-OptiX rmagine GPU ray-casting backend. Since OptiX is deferred and Vulkan needs an SDK, the realistic near-term cross-check is MICP-L on the EMBREE (CPU) backend vs the rmcl-cuda correspondence statistics (GPU) -- i.e. verify the GPU statistics path produces the same correspondences/pose as the CPU path on the same scan. A full GPU-ray-cast localization on AMD is blocked until either rmagine-vulkan (with an SDK) or a HIPRT rmagine backend exists.

NON-GPU REGRESSION SET (must not regress): rmagine's core ctests -- `ctest -R '^(core_|math)'` (tests/core/: math, math_reduction, math_svd, math_cholesky, math_cov_transform, quaternion, memory_slicing) build with plain g++ and are unaffected by the HIP change; run them before/after to prove the host path is untouched. (Embree tests need libembree; skip on this host.)

## Disposition
- Recommend: PORT the rmagine CUDA compute backend (rmagine_core + rmagine_cuda) -- correctness-first Strategy A with the wave64 reduction fix -- and validate it on real gfx90a via rmagine's `cuda_*` ctests. This is the tractable, high-value, in-scope deliverable that gives AMD users the GPU correspondence/statistics/noise compute with NO ROCm path today, and fixes a genuine wave64 bug. rmcl's own two `.cu` ride on top.
- DEFER (B7): the OptiX ray-tracing backend (rmagine_optix + rmcl's RCCOptix / BeamEvaluateProgram / rmcl_localization). No ROCm equivalent; needs a HIPRT or hand-written HIP-BVH reimplementation. Pending jeff's HIPRT decision. (Mirror EnvGS: a portable compute path now, the hardware-RT path as a deferred stage.)
- DEFER (environment): the Vulkan RT backend (cross-vendor AMD route) -- no Vulkan SDK on this host and it is a separate shader-based RT engine, out of scope for a CUDA->HIP compute port; revisit if a Vulkan-capable env + a Vulkan<->HIP interop (hipImportExternalMemory) is wanted.
- Do NOT mark `triage.py skip` -- there IS a portable, valuable port (rmagine_cuda). The decisive question's answer is: the GPU RAY-CASTING is OptiX/Vulkan (defer the RT), but the GPU COMPUTE around it (statistics/reductions/noise/resample) is custom CUDA and IS portable.

## Stage 2 -- HIPRT ray-tracing backend (scope added after the plan above)
rmagine's OptiX backend (`rmagine_optix`: optixAccelBuild / optixTrace / module+SBT+pipeline, ~43 OptiX symbols across ~60 files) is the NVIDIA-only ray-casting path and has no mechanical port. Stage 2 reimplements it against AMD HIPRT as a new `rmagine_hiprt` component: HIPRT builds the BVH, one HIP trace kernel replaces the raygen/closesthit/miss programs, and the existing (Stage 1, HIP-validated) compute consumes the result. All four sensor types (Pinhole, Spherical, O1Dn, OnDn) are implemented. HIPRT is not part of ROCm 7.2 and needs the SDK via `HIPRT_PATH`; `rmagine_hiprt` is therefore USE_HIP-gated, opt-in, and has NO install rules, so it is not part of the installed package. The OptiX->HIPRT symbol mapping, the layout-sensitivity of the device structs, and the validation numbers are recorded in notes.md ("Stage 2"), which is the authority for what was actually built.

## Status
Delivered and validated: Stage 1 (rmagine_core + rmagine_cuda under `USE_HIP`, with the wave64 reduction fix) and Stage 2 (`rmagine_hiprt`, all four simulators). Validated on linux-gfx90a, linux-gfx1100, windows-gfx1101 and windows-gfx1201. Two later rounds fixed the install path that no in-tree gate covered: the exported package config resolving CUDA at consumer time, and the compatibility header making the public headers non-includable from a plain C++ TU. Not done: the README/wiki wording for `USE_HIP` (the README defers build detail to an external wiki repository; to be offered in the upstream PR body rather than imposed), multi-mesh HIPRT scenes, and the downstream rmcl layer.

## Open questions
1. ~~Forks: this port's deliverable lives in rmagine, not rmcl.~~ ANSWERED: rmagine is the ported project and AMD-Ecosystem/rmagine is the fork; the project record was renamed from rmcl to rmagine. rmcl, if it is ever taken on, is a separate project that `depends_on` rmagine and pins the validated commit.
2. ROS 2 + Embree provisioning for the rmcl-layer milestone and the localization gate: install ros-jazzy + libembree-dev (apt) on this host, or run that gate in a ROS 2 docker? The rmagine_cuda gate does NOT need it; the end-to-end localization gate does.
3. rmagine version pin: rmcl wants rmagine 2.4-2.5.0 (RMAGINE_MAX_VERSION 2.5.0); source_dependencies pins rmagine `main`. Pin the validated rmagine commit so rmcl + rmagine move together.
4. CudaContext driver-API: confirm whether the explicit primary-context management is needed on HIP or can be relaxed to runtime-lazy (affects the compat-header cuCtx* mapping vs a USE_HIP stub).
5. cuRAND determinism: hipRAND will not match cuRAND bitwise; confirm the MCL/noise validation is statistical (distribution moments), which is the correct bar anyway.
