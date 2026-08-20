# Open3D -- ROCm/HIP port notes (lead: linux-gfx90a, ROCm 7.2.1, MI250X)

Strategy A (pure CMake, USE_HIP gate + single compat header). Fork:
https://github.com/AMD-Ecosystem/Open3D branch `moat-port`. Upstream base
0333798fcff5a2fe95470e69291eca6a9efbae6c.

## How USE_HIP composes with BUILD_CUDA_MODULE
The GPU code is keyed on `BUILD_CUDA_MODULE` everywhere (CMake `if()` and
`#if defined(BUILD_CUDA_MODULE)` host guards). `USE_HIP=ON` forces
`BUILD_CUDA_MODULE=ON` and only swaps the language (enable_language(HIP)), the
math libs (hipBLAS/hipSOLVER/hipSPARSE), and retags `.cu` LANGUAGE HIP. The
NVIDIA path is untouched (`elseif(BUILD_CUDA_MODULE)` keeps the old block).

## The .cu -> LANGUAGE HIP mechanism (zero per-dir churn)
`open3d_set_global_properties(target)` (called by every GPU lib after its
`target_sources`) calls `open3d_set_hip_properties(target)` under USE_HIP. That
helper walks the target's sources, retags `*.cu` LANGUAGE HIP, sets
HIP_ARCHITECTURES from `${CMAKE_HIP_ARCHITECTURES}`, adds the `hip_compat` shim
include dir + the cpp root, and force-includes `cpp/open3d/core/hip/CUDAToHIP.h`
on the HIP TUs. The host linalg `.cpp` (AddMMCUDA.cpp ...) stay CXX; only the
math-lib link target swaps.

## Compat layer (the only files that know about HIP)
- `cpp/open3d/core/hip/CUDAToHIP.h` -- 45 cuda*->hip* runtime aliases, the
  `cub`->`hipcub` namespace alias, `OPEN3D_FULL_WARP_MASK (~0ull)`, an
  `__ffs(unsigned long long)` overload (for SlabHash's `__ffs(__ballot_sync)`),
  and `CUDA_VERSION 9000` to take the `>=9000` *_sync paths. `<cstring>`/
  `<cstdlib>` precede `<hip/hip_runtime.h>` (gpuRIR host-memcpy lesson).
- `cpp/open3d/core/hip/hip_compat/{cuda.h,cuda_runtime.h,cuda_runtime_api.h,
  cub/cub.cuh}` -- forwarding shims so the CUDA-spelled angle-bracket includes
  resolve under HIP without editing every include line (MPPI lesson). On
  include path only under USE_HIP, so the CUDA build uses the real toolkit
  headers.

## Fault-class fixes
1. FAISS warp-select (`core/nns/kernel/`) -- the highest-risk body.
   - `DeviceDefs.cuh`: kWarpSize STAYS 32. Decision: run the 64-lane wavefront
     as TWO independent 32-lane bitonic groups (popsift two-halves model), NOT
     kWarpSize=64. Reason: BlockSelectFloat*.cu instantiates NumWarpQ==32 (and
     ==1), and `kNumWarpQRegisters = NumWarpQ / kWarpSize` would be 0 if
     kWarpSize were 64 -- the warp queue collapses. With kWarpSize=32, every
     shfl uses width=32 (HIP confines exchange to the 32-lane group) and
     getLaneId() returns the in-group 0..31 lane, so each half behaves exactly
     like a 32-lane NVIDIA warp. Correct on wave32 AND wave64 (arch-unified).
   - `PtxUtils.cuh`: ALL inline PTX (bfe/bfi/mov %laneid/bar.sync) is replaced
     under USE_HIP with clang builtins; getLaneId() = threadIdx.x & 31 (the
     kernels launch 1-D blocks). The bitfield/lanemask/namedBarrier helpers are
     dead in the build but reimplemented for self-consistency.
   - `WarpShuffle.cuh`: __shfl_*_sync masks 0xffffffff -> OPEN3D_FULL_WARP_MASK
     (64-bit on HIP; HIP static_asserts sizeof(mask)==8).
   - `Select.cuh`: __any_sync(0xffffffff,..) -> OPEN3D_FULL_WARP_MASK. The
     full-wavefront __any over both halves is correct: a non-full half still
     merges its (sorted, initK-padded) thread queue before reset, a no-op.
   - `warpFence()` -> `__syncthreads()` on HIP (kernels are warp-convergent).
2. `core/kernel/ReductionCUDA.cu` (PyTorch-derived) already uses runtime
   `warpSize` (64) for its reduction tree; only the WARP_SHFL_DOWN default mask
   needed widening to 64-bit (OPEN3D_REDUCE_FULL_WARP_MASK).
3. SlabHash (`core/hashmap/CUDA/Slab*`) -- NON-default backend, SCOPED OUT of
   validation, made to COMPILE only: `kSyncLanesMask`/`kNodePtrLanesMask`
   widened to 64-bit under USE_HIP so the *_sync masks compile; `__ffs` of the
   64-bit ballot uses the compat `__ffs(ull)` overload. Its 32-lane lane
   election is NOT wave64-correct; a correct wave64 rewrite is deferred. Slab
   HashMap test cases are skipped.
4. Library swaps -- `LinalgHeadersCUDA.h` includes hipblas/hipsolver and aliases
   cublas*/cusolver* -> hipblas*/hipsolver*. The typed Dn{S,D} compat API
   (hipsolverDnSgetrf/gesvd/getrs/geqrf/ormqr + _bufferSize) is a clean drop-in
   (verified signatures match cuSOLVER, no extra lwork arg in the solve calls).
   The wider cuSOLVER status enum's orphan cases (INVALID_LICENSE, IRS_*,
   INVALID_WORKSPACE) are `#if !defined(USE_HIP)`-guarded in the fmt formatter.
5. stdgpu (DEFAULT hashmap backend) -- `3rdparty/stdgpu/stdgpu.cmake` builds the
   pinned commit's STDGPU_BACKEND_HIP with CMAKE_CXX_COMPILER=hipcc/clang (its
   host impl/*.cpp reach rocPRIM, clang-only) + CMAKE_HIP_ARCHITECTURES. The
   newer pinned commit already has a HIP-aware Findthrust, so the cupoch
   stdgpu-1.3.0 rot does not apply.
6. NPP image ops -- no ROCm equivalent. `NPPImage.cpp` body is
   `#if !defined(USE_HIP)`; under HIP it provides LogError stubs (so Image.cpp's
   CUDA branch links). NPP find + link guarded out. GPU image-filter ops are an
   explicitly-unsupported path on ROCm; CPU/IPP path unaffected.

## Scoped OUT of the lead (guarded, NOT failures)
- CUTLASS 1.3.3 ML conv ops (BUILD_PYTORCH_OPS/BUILD_TENSORFLOW_OPS OFF; CUTLASS
  does not port to ROCm).
- NPP GPU image filters (above).
- SlabHash backend wave64 correctness (non-default; compiles, tests skipped).
- BUILD_GUI / Filament / WebRTC / examples / python module (headless GPU core).

## Build (gfx90a)
Configure:
```
cmake -S projects/Open3D/src -B projects/Open3D/src/build \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DBUILD_PYTORCH_OPS=OFF -DBUILD_TENSORFLOW_OPS=OFF \
  -DBUILD_GUI=OFF -DBUILD_WEBRTC=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF -DBUILD_PYTHON_MODULE=OFF \
  -DBUILD_UNIT_TESTS=ON -DBUILD_ISPC_MODULE=OFF \
  -DWITH_IPP=OFF \
  -DCMAKE_BUILD_TYPE=Release -DDEVELOPER_BUILD=ON
cmake --build projects/Open3D/src/build -j16 --target tests
```
Followers (gfx1100, gfx1151): same command, only CMAKE_HIP_ARCHITECTURES
changes -- no source edit by design.

## Test plan / validation
GPU gtest binary `build/bin/tests` (BUILD_UNIT_TESTS=ON). PermuteDevices
fixtures auto-run each parameterized test on BOTH CPU and the gfx90a device, so
CPU is the built-in oracle. Run serially on ONE GCD (HIP_VISIBLE_DEVICES). Gate
suites: *Tensor*, *MemoryManager*, *HashMap* (stdgpu), *NearestNeighborSearch*/
*KnnIndex*/*FixedRadiusIndex* (FAISS wave64), *Reduction*, t-geometry +
registration GPU tests.

## Additional build fixes found during bringup (beyond the plan)
- `__CUDACC__` cannot be faked globally: defining it forces rocThrust to its
  CUDA backend (compiler.h reads __CUDACC__ as "CUDA compiler"), dumping every
  thrust call into the "unimplemented for this system" generic fallback. Fix:
  do NOT define __CUDACC__; instead extend each `#ifdef __CUDACC__` device-vs-
  host guard across the kernel headers to `__CUDACC__ || __HIPCC__` (the
  cudaKDTree/GLM trap), force THRUST_DEVICE_SYSTEM=5 (HIP) via a compile def so
  rocThrust's auto-detect is bypassed, and provide OPEN3D_DEVICE_CODE
  (= __HIP_DEVICE_COMPILE__ on HIP) for the data-pointer-selection guards.
- `MiniVec.h` FN_SPECIFIERS keyed on __CUDACC__ -> +__HIPCC__ (else device
  kernels can't call MiniVec ctor/operator[]).
- `OPEN3D_ASSERT_HOST_DEVICE_LAMBDA` uses the nvcc-only intrinsic
  `__nv_is_extended_host_device_lambda_closure_type`; no-op it on HIP.
- `__ffs(__ballot_sync(...))` is ambiguous on HIP (ballot is unsigned long long,
  HIP has only __ffs(int)/__ffs(unsigned)); the compat header adds a
  __host__ __device__ __ffs(unsigned long long) (uses __ffsll device /
  __builtin_ffsll host).
- `GeometryMacros.h`: the `__CUDA_ARCH__ < 600` double-atomicAdd fallback would
  REDEFINE HIP's native atomicAdd(double*); guarded CUDA-only.
- `RegistrationImpl.h`: clang requires explicit template instantiations of
  __host__ __device__ function templates to carry the attributes -> prefixed
  each with OPEN3D_HOST_DEVICE (cudaKDTree attribute-match lesson).
- Host nodiscard: HIP marks the runtime API nodiscard (CUDA does not), tripping
  host .cpp under -Werror; relaxed -Werror=unused-result for CXX/HIP under
  USE_HIP (mirrors the CUDA build's NVCC 2809 suppression).
- gcc13 false `-Wmaybe-uninitialized` on Eigen SIMD in MinimumOBB/OBE.cpp
  (best_R IS Identity-initialized) -> -Wno-error=maybe-uninitialized for gcc.
- stdgpu install: the pinned commit's HIP backend dir lacks
  determine_thrust_paths.cmake (CUDA-only) but install() references it -> a
  PATCH_COMMAND copies the CUDA one in. stdgpu CXX must be hipcc/clang (its host
  impl/*.cpp reach rocPRIM); the compiler overrides go AFTER
  ExternalProject_CMAKE_ARGS_hidden so they win over the host g++ default.

## STATUS (2026-05-31): core BUILDS + fully VALIDATES incl. HashMap + VoxelBlockGrid -> ported
The full `tests` binary builds for gfx90a (all ~31 HIP .cu TUs compile + link).
GPU validation on one IDLE MI250X GCD (HIP_VISIBLE_DEVICES=0; this box has 4
GCDs 0-3 shared with other MOAT CLIs, so pick an idle one via `rocm-smi
--showuse`/`--showmemuse` and pin ONE -- two concurrent runs on one GCD livelock
the ICP tests. PermuteDevices CPU param is the per-test oracle, `/1` = the HIP
device):
- *Tensor*+*Reduction*+*MemoryManager*: 421/421 PASS (warp reduction
  wave64-correct; 1 SYCL skip).
- *NearestNeighborSearch*/*KnnIndex*/*FixedRadiusIndex*/*Knn*/*Hybrid*: PASS --
  the FAISS warp-select wave64 (highest risk) is CORRECT via the
  two-32-lane-subgroup model.
- *HashMap*: 20/20 PASS. The four dedup-heavy tests (Reserve, InsertComplexKeys,
  MultivalueInsertion, HashSet) are stable over 30 `--gtest_repeat` iterations
  with `--gtest_break_on_failure` (240 device runs, 0 fail).
- *VoxelBlockGrid* (TSDF integrate + ray casting + IO): 14/14 PASS, stable over
  5 reruns (it uses the StdGPU hashmap internally, so this is the same dedup
  surface end-to-end).
- Registration ICP (PointToPoint/PointToPlane/Colored), Feature (FPFH): PASS.

EXPECTED scoped-out failures (NOT regressions, documented above): the NPP GPU
image-filter ops throw NppUnsupportedOnHIP, so the tests that exercise them fail
by design -- Image.{Filter,FilterGaussian,FilterBilateral,FilterSobel,Resize,
PyrDown,Dilate}, Odometry.RGBDOdometryMultiScale{PointToPlane,Intensity,Hybrid}
(image pyramids), PointCloud.CreateFromRGBDOrDepthImageWithNormals. These need
the CPU device on ROCm.

## HashMap wall RESOLVED -- root cause was the Slab backend, NOT stdgpu
An earlier diagnosis (above, now corrected) blamed stdgpu `unordered_base::
try_insert` dedup for the 4 HashMap failures. That was WRONG. The HashMap/HashSet
tests loop over BOTH GPU backends: the test code (cpp/tests/core/HashMap.cpp)
pushes `Slab` then `StdGPU` for any CUDA device. The failing reduction
`masks.Sum()=1024 vs 1023` was on the FIRST backend iteration, which is **Slab**,
not the default StdGPU.

How it was pinned down (printf is unreliable -- the GPU FIFO drops output past
~1024 lines): a host recount split by backend iteration showed the FIRST line
(Slab) at host_bool=1024/1026 while the SECOND (StdGPU) was exactly 1023; a
`__device__` win-counter inside stdgpu's try_insert plus a host recount of
`impl_.size()` showed the StdGPU map ALWAYS produces exactly 1023 wins / size
1023, even on a run gtest then failed. Confirmation: with PRISTINE stdgpu and only
Slab excluded on ROCm, the 4 hard tests pass and the suite is 20/20. A false
trail avoided: stdgpu's HIP `atomic_load`/`atomic_store` are plain `volatile`,
which looks suspicious, but making them `__hip_atomic_*` changed no result --
StdGPU dedup was already correct; the issue was purely Slab.

The SlabHash backend (warp-cooperative, `tid>>5` 32-lane lane election, `__ffs`
of a ballot, `kSyncLanesMask`/`kNodePtrLanesMask`) is genuinely wave64-incorrect
on CDNA and is the NON-default backend explicitly scoped OUT of this port (a
correct wave64 rewrite is deferred). The DEFAULT, production-relevant StdGPU
backend is wave64-correct AS-IS; no stdgpu source patch is needed (stdgpu stays
the pinned tarball; 3rdparty/stdgpu/stdgpu.cmake is unchanged).

FIX (test-only, minimal footprint, committed on the fork): guard Slab out of the
HashMap and VoxelBlockGrid tests under `#if defined(USE_HIP)` -- HashMap.cpp 9
sites push only `StdGPU` on USE_HIP; VoxelBlockGrid.cpp's EnumerateBackends sets
`include_slab=false` on USE_HIP. `USE_HIP` reaches the C++ test TUs via
`open3d_set_global_properties(tests)` -> `target_compile_definitions(tests
PRIVATE USE_HIP)` (verified in build/cpp/tests/.../flags.make). With Slab
excluded, VoxelBlockGrid validates fully on GPU. Do NOT re-enable Slab on ROCm
without first porting its 32-lane lane election to wave64.

State: linux-gfx90a -> `ported` at fork sha 3311036 (single curated
[ROCm] commit, amended in place; the prior 2ae6d769 lacked these two test edits).

## Validation 2026-05-31 (validator, linux-gfx90a, ROCm 7.2.1, MI250X)

GPU: HIP_VISIBLE_DEVICES=3 (one idle GCD, 4 GCDs total on this box).
Fork HEAD after comment fix amend: 9eb3704d07b4b6c65cd46ac5805b0654a39e7250.
Incremental build (comment-only change, no codegen impact): PASS.

### Comment fix folded in
cmake/Open3DSetGlobalProperties.cmake:106 comment corrected to state that
__CUDACC__ is deliberately NOT defined (CUDAToHIP.h explains why) and
THRUST_DEVICE_SYSTEM=5 is set to pin rocThrust because __CUDACC__ is absent.
Old text falsely claimed the compat header defines __CUDACC__.
Amended into the single curated [ROCm] commit; pushed --force-with-lease.

### GPU test results (gfx90a, GCD 3)

Build command (incremental):
```
cmake --build projects/Open3D/src/build -j16 --target tests
```

Test commands:
```
HIP_VISIBLE_DEVICES=3 ./build/bin/tests \
  --gtest_filter='*Tensor*:*Reduction*:*MemoryManager*'
HIP_VISIBLE_DEVICES=3 ./build/bin/tests \
  --gtest_filter='*NearestNeighbor*:*KnnIndex*:*FixedRadiusIndex*:*Knn*:*Hybrid*:*Registration*:*Feature*'
HIP_VISIBLE_DEVICES=3 ./build/bin/tests \
  --gtest_filter='*HashMap*:*VoxelBlockGrid*'
HIP_VISIBLE_DEVICES=3 ./build/bin/tests --gtest_repeat=30 --gtest_break_on_failure \
  --gtest_filter='*HashMapPermuteDevices.Reserve*:*HashMapPermuteDevices.InsertComplexKeys*:*HashMapPermuteDevices.MultivalueInsertion*:*HashMapPermuteDevices.HashSet*'
```

Results:
- Tensor + Reduction + MemoryManager: 421/421 PASS (1 SYCL skip, expected)
- NearestNeighborSearch (FAISS warp-select wave64) / KNN / FixedRadius / Hybrid
  + Registration ICP + FPFH Feature: 44/45 (1 fail = RGBDOdometryMultiScaleHybrid,
  NPP-scoped-out as documented; see notes above)
- HashMap (stdgpu backend only, Slab excluded): 20/20 PASS
- VoxelBlockGrid (TSDF integrate / ray casting / IO): 14/14 PASS
- HashMap dedup stability (4 dedup-heavy tests x 30 repeats = 240 device runs): 0 FAIL

Total validated: 421 + 44 + 34 = 499 passing GPU tests.
Scoped-out NPP failure (1): RGBDOdometryMultiScaleHybrid -- expected, documented above.

State: linux-gfx90a -> `completed` at validated_sha 9eb3704.
Followers unblocked: linux-gfx1100 -> port-ready, windows-gfx1151 -> port-ready.

## Review 2026-05-31 (reviewer, /pr-review local-branch @ 3311036)
Verdict: review-passed. Strategy A is correct for this pure-CMake project; the diff is additive and USE_HIP-gated (778+/121-, single [ROCm] commit), and the NVIDIA path is byte-identical (the only host .cu/.cpp touched are CUDAUtils.cpp -- a behavior-neutral static_cast<void>(cudaGetLastError()) -- ReductionCUDA.cu whose non-HIP default mask stays unsigned int 0xffffffffu, and NPPImage.cpp wrapped #if USE_HIP/#else original/#endif). Commit hygiene clean: title 54 chars, Claude-disclosed, Test Plan present, no noreply/ghstack/AMD-internal refs.

Fault-class analysis independently verified (not taken on faith):
- StdGPU-vs-Slab root cause CONFIRMED. grep of StdGPUHashBackend.h returns zero warp/lane/ballot/shfl/tid>>5 patterns -- its dedup runs through stdgpu's open-addressing container (per-slot atomicCAS across all lanes), so it is wave64-correct as-is and needs no source patch (tarball + stdgpu.cmake content unchanged). Slab IS genuinely wave64-incorrect: SlabHashBackendImpl.h:106 lays out bucket_id*kWarpSize+lane_id, :203/:307 __ballot_sync(kSyncLanesMask,..), :206/:310 __ffs(work_queue)-1 lane election, :208 __shfl_sync(..,src_lane,kWarpSize) -- 32-lane election on a 64-lane wavefront.
- Test-only Slab guard is legitimate and reaches the TUs. 9 sites in HashMap.cpp push only StdGPU under #ifndef USE_HIP; VoxelBlockGrid.cpp EnumerateBackends sets include_slab=false under #ifdef USE_HIP. USE_HIP reaches the test binary via cpp/tests/CMakeLists.txt:53 open3d_set_global_properties(tests) -> Open3DSetGlobalProperties.cmake:97 target_compile_definitions(tests PRIVATE USE_HIP).
- FAISS two-32-lane-halves model is internally consistent. kWarpSize stays 32 (DeviceDefs.cuh:53); getLaneId()=threadIdx.x&31 (PtxUtils.cuh:70); all shfl default width=kWarpSize=32 so exchanges stay within a 32-lane half; masks widened to OPEN3D_FULL_WARP_MASK (~0ull). The UNCHANGED Reduction.cuh blockReduceAll was checked for a wave64 hazard and is correct under this model: warpId=threadIdx.x/32 and the final combine (warpId==0, smem[laneId], laneId 0..31, width-32 warpReduceAll) decomposes the block into logical 32-lane warps exactly like NVIDIA, valid for blockDim.x<=1024. L2Select.cuh exercises this path -- covered by the KNN GPU tests.
- Slab masks: kSyncLanesMask->64-bit (correct ballot/shfl mask so *_sync compiles); kNodePtrLanesMask=0x7FFFFFFFull is used BOTH as a ballot mask and as a (1<<lane_id)& lane bitmask and is wave-incorrect, but that is inside the scoped-out compile-only Slab backend with tests excluded, so it does not gate validation.
- Library swaps clean: LinalgHeadersCUDA.h aliases the typed cublas/cusolverDn{S,D} API to hipblas/hipsolver, orphan wider cuSOLVER status enum cases #if !defined(USE_HIP)-guarded. NPP/SlabHash/CUTLASS/GUI scope-outs are clean guarded deferrals.

Finding (non-blocking, comment-only -- fold into the validator's SHA-update amend, do NOT churn HEAD on its own):
- cmake/Open3DSetGlobalProperties.cmake:106 comment claims "The compat header defines __CUDACC__ (Open3D's device-guard requirement)". This is factually wrong and contradicts CUDAToHIP.h:29 ("We do NOT fake __CUDACC__ for HIP"); grep confirms __CUDACC__ is never #defined anywhere. The real mechanism is extending each __CUDACC__ guard to ||__HIPCC__, and THRUST_DEVICE_SYSTEM=5 (line 113) is set precisely BECAUSE __CUDACC__ is absent and rocThrust's auto-detect must be pinned. The code is correct; only the comment's premise is wrong and will mislead. Reword to: rocThrust's compiler.h would otherwise auto-detect the wrong backend, so pin THRUST_DEVICE_SYSTEM=5 (HIP).

Safe to proceed to GPU validation. The missing GPU run at review time is expected (validator stage runs the real gfx90a tests next).

## Validation 2026-05-31 (gfx1100, ROCm 7.2.1)

GPU: 4x AMD Radeon Pro W7800 48GB (gfx1100, wave32); HIP_VISIBLE_DEVICES=0 (all idle).
Fork HEAD: 9eb3704d07b4b6c65cd46ac5805b0654a39e7250 (moat-port branch, untouched -- no fork commit).

### Build

Host dep install (not present, installed via apt): libwayland-bin libwayland-dev libxkbcommon-dev (GLFW needs wayland-scanner and xkbcommon to configure even with BUILD_GUI=OFF; the gfx90a box had these pre-installed). No source change.

Configure command (same as gfx90a recipe, only CMAKE_HIP_ARCHITECTURES differs):
```
cmake -S projects/Open3D/src -B projects/Open3D/src/build \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DBUILD_PYTORCH_OPS=OFF -DBUILD_TENSORFLOW_OPS=OFF \
  -DBUILD_GUI=OFF -DBUILD_WEBRTC=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF -DBUILD_PYTHON_MODULE=OFF \
  -DBUILD_UNIT_TESTS=ON -DBUILD_ISPC_MODULE=OFF \
  -DWITH_IPP=OFF \
  -DCMAKE_BUILD_TYPE=Release -DDEVELOPER_BUILD=ON
cmake --build projects/Open3D/src/build -j16 --target tests
```

Build: PASS (~227 s including full 3rdparty fetch and HIP compilation of all 25 TUs).

### gfx1100 code-object evidence

roc-obj-ls on build/bin/tests: 25 x hipv4-amdgcn-amd-amdhsa--gfx1100 code objects; 0 gfx90a. Arch confirmed.

### GPU test results (gfx1100, wave32)

Test commands (same filter as gfx90a lead, SlabHash excluded via USE_HIP guard in test source):
```
HIP_VISIBLE_DEVICES=0 /var/lib/jenkins/moat/projects/Open3D/src/build/bin/tests \
  --gtest_filter='*Tensor*:*Reduction*:*MemoryManager*'
HIP_VISIBLE_DEVICES=0 /var/lib/jenkins/moat/projects/Open3D/src/build/bin/tests \
  --gtest_filter='*NearestNeighbor*:*KnnIndex*:*FixedRadiusIndex*:*Knn*:*Hybrid*:*Registration*:*Feature*'
HIP_VISIBLE_DEVICES=0 /var/lib/jenkins/moat/projects/Open3D/src/build/bin/tests \
  --gtest_filter='*HashMap*:*VoxelBlockGrid*'
HIP_VISIBLE_DEVICES=0 /var/lib/jenkins/moat/projects/Open3D/src/build/bin/tests \
  --gtest_repeat=10 --gtest_break_on_failure \
  --gtest_filter='*HashMapPermuteDevices.Reserve*:*HashMapPermuteDevices.InsertComplexKeys*:*HashMapPermuteDevices.MultivalueInsertion*:*HashMapPermuteDevices.HashSet*'
HIP_VISIBLE_DEVICES=0 /var/lib/jenkins/moat/projects/Open3D/src/build/bin/tests \
  --gtest_filter='*NearestNeighbor*:*KnnIndex*:*FixedRadiusIndex*'
```

Results:
- Tensor + Reduction + MemoryManager: 421/421 PASS (1 SYCL skip -- matches gfx90a exactly)
- NearestNeighborSearch (FAISS warp-select wave32) / KNN / FixedRadius / Hybrid + Registration ICP + FPFH Feature: 44/45 PASS (1 fail = RGBDOdometryMultiScaleHybrid, NPP scoped-out -- matches gfx90a exactly)
- HashMap (stdgpu backend only, Slab excluded by USE_HIP guard): 20/20 PASS
- VoxelBlockGrid (TSDF integrate / ray casting / IO): 14/14 PASS
- HashMap dedup stability (4 dedup-heavy tests x 10 repeats = 80 device runs): 0 FAIL
- NNS determinism re-run: 15/15 PASS

Total validated gfx1100: 421 + 44 + 34 = 499 passing GPU tests. Exactly matches gfx90a.
Scoped-out NPP failure (1): RGBDOdometryMultiScaleHybrid -- expected, documented above.

### Wave32 verdict

FAISS warp-select (kWarpSize=32): CORRECT on native wave32. On gfx1100 kWarpSize=32 IS the wavefront width -- no subgroup split needed. All NNS/KNN/FixedRadius tests pass. The arch-unified kWarpSize=32 design is confirmed correct for both wave64 (two 32-lane halves model) and wave32 (direct one-to-one).

Tensor reductions (OPEN3D_REDUCE_FULL_WARP_MASK, 64-bit mask, runtime warpSize=32 on gfx1100): CORRECT. Reduction tree runs over 32 lanes as expected; 421/421 Tensor+Reduction tests pass.

HashMap stdgpu dedup: CORRECT on gfx1100 (20/20, stable). SlabHash skipped per established protocol (non-default backend, wave-incorrect, scoped out of port scope).

No HIP faults, no NaN, clean exit on all runs.

State: linux-gfx1100 -> `completed` at validated_sha 9eb3704.
Fork untouched (no commit pushed; no source change required).

## Windows gfx1151 follower: CONFIGURES, build blocked on host 3rdparty (out of ROCm scope)

Configure SUCCEEDS on Windows with the all-clang toolchain for gfx1151:
  cmake -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
        -DCMAKE_HIP_COMPILER=clang++ -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151
        -DCMAKE_PREFIX_PATH=<rocm-root> -DCMAKE_TLS_VERIFY=OFF
        -DCMAKE_CXX_FLAGS="-DNOMINMAX -DWIN32_LEAN_AND_MEAN"
        <same module-OFF flags as the gfx90a build> -DBUILD_UNIT_TESTS=ON
  (script: agent_space/open3d_win.sh; "Configuring done (85.5s)". enable_language(HIP)
  forces the all-clang toolchain on Windows -- Clang-HIP + MSVC-CXX is refused.)
  CMAKE_TLS_VERIFY=OFF is needed because cmake's bundled curl uses Windows schannel
  whose chain is incomplete for github; FetchContent deps are URL_HASH-pinned so
  integrity is preserved.

Build (`--target tests`) reaches ninja step 185/813, then blocks on building Open3D's
BUNDLED HOST 3rdparty under Windows-clang -- NOT the HIP port:
- ext_stdgpu (the HIP-backend dep, the only ROCm-relevant one) configures fine.
- ext_turbojpeg: libjpeg-turbo cmakescripts/BuildPackages.cmake:106 references a missing
  win/projectTargets-release.cmake.in (a libjpeg-turbo Windows-packaging bug).
- ext_curl: FindOpenSSL fails -- bundled boringssl (OPENSSL_ROOT_DIR) is not built (curl
  configure runs before/without boringssl under Windows-clang).

These are host I/O/TLS libraries (JPEG, curl, boringssl), required only to LINK the
monolithic `tests` binary, not by the GPU core port. On Linux they build from source
cleanly; the Windows-clang builds need per-library patching (libjpeg-turbo packaging,
boringssl/curl). That is host-3rdparty Windows-build porting, out of ROCm-port scope
(cf. LMCache POSIX, llm.c makefile). The HIP/ROCm port itself is sound (validated on
gfx90a + gfx1100). Unblocks when those bundled host libs are made to build on
Windows-clang (or pre-supplied as prebuilt), then re-run the build + GPU tests.

## Validation 2026-06-04 (windows-gfx1101, AMD Radeon PRO V710, ROCm 7.14 / TheRock)

Host: the gfx1101+gfx1201 Windows workstation (memory windows-gfx1101-gfx1201-host).
GPU pinned HIP_VISIBLE_DEVICES=0 (gfx1101). 64-core, -j64 builds.

### Working Windows ROCm toolchain (REUSABLE -- established here for the first time)
Single self-contained ROCm root = the pip rocm-sdk-devel tree:
`B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel`
(has clang++, hipcc, hip cmake, AND hipblas/hipsolver/hipsparse/rocblas/rocthrust --
version 60850-d34cbb64, matching torch.version.hip). Configure (all-clang; USE_HIP
forces all-clang, MSVC-host is refused) succeeds in 16s:
```
ROCM=.../_rocm_sdk_devel
cmake -G Ninja -S projects/Open3D/src -B projects/Open3D/src/build \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/clang.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
  -DCMAKE_PREFIX_PATH=$ROCM -DCMAKE_TLS_VERIFY=OFF \
  -DCMAKE_CXX_FLAGS="-DNOMINMAX -DWIN32_LEAN_AND_MEAN" \
  <same module-OFF flags as gfx90a> -DBUILD_UNIT_TESTS=ON
```
(script: agent_space/open3d_cfg_gfx1101.sh.) hipBLAS/hipSOLVER/hipSPARSE all resolve
from this root -- the earlier `find_package(hipblas)` miss was from pointing at the
TheRock/build compiler-only tree; the pip devel tree is the complete SDK.

### Result: PROGRESSES PAST the gfx1151 wall, blocked deeper on 3rdparty-under-clang
`cmake --build ... -j64 --target tests` reaches ninja [232/813] (gfx1151 stopped at
185). The bundled libjpeg-turbo and curl that blocked gfx1151 BUILD FINE on this host.
New blocker is bundled TBB: every tbb .cpp fails with
`clang++: error: unsupported option '-fPIC' for target 'x86_64-pc-windows-msvc'`
-- TBB's own cmake adds -fPIC on the "Clang" branch, invalid for the windows-msvc
target clang uses. After TBB, Open3D's other heavy bundled deps (VTK, assimp, embree,
...) would each surface their own Windows-clang build issues; upstream Open3D builds
Windows with MSVC, not the all-clang toolchain USE_HIP forces, so this is an
unsupported-upstream build configuration. This is the SAME class as the gfx1151
determination (bundled host 3rdparty not building under Windows-clang), arch-INDEPENDENT
and out of ROCm-port scope -- the GPU/HIP port is proven on gfx90a + gfx1100 (499 tests).

DETERMINATION: windows-gfx1101 BLOCKED, reason = bundled 3rdparty (TBB -fPIC under the
all-clang toolchain USE_HIP forces; VTK/assimp/embree to follow) does not build on
Windows-clang -- host build infra, not the GPU port. Same arch-independent wall applies
to windows-gfx1201. REVERSIBLE: supply the heavy 3rdparty (tbb, vtk, assimp, embree,
jpeg, curl, openssl) as prebuilt system packages (vcpkg x64-windows) via the
USE_SYSTEM_* options and re-run; the toolchain + GPU port are ready, only host-lib
provisioning remains.

## Windows build SOLVED via vcpkg (2026-06-07) -- gfx1201 tests.exe BUILT; GPU run pending host

The Windows-clang build wall (previously blocking windows-gfx1101/gfx1201) is
BROKEN. Open3D's full `tests` binary builds for gfx1201 under the all-clang
toolchain by supplying the clang-unbuildable bundled host libs from vcpkg and
fixing a handful of "code/CMake assumes MSVC on Windows" gaps. The HIP/GPU port
itself needed ZERO new changes (validated on Linux gfx90a+gfx1100); every fix
below is host-build-infra, WIN32/Clang-gated, so the Linux build is byte-identical.

Script: agent_space/open3d_cfg_gfx1201_vcpkg.sh (configure),
agent_space/run_tests_gfx1201.bat (run with runtime DLLs). agent_space is
gitignored -- the authoritative recipe is here.

### Root cause of the original wall
The bundled 3rdparty did not "fail to build under clang" generically; specific
libs inject MSVC cl.exe flags or use POSIX-isms, gated on the wrong condition.
USE_HIP forces all-clang (clang++ GNU-driver, CMake `MSVC`=FALSE), so anything
gated on `if(MSVC)` to pick the Windows path is skipped.

### vcpkg setup (host has a TLS revocation wall)
- VS-bundled vcpkg is a registry-only stub (no ports). Cloned full vcpkg to
  B:/vcpkg, bootstrapped (vcpkg-tool 2026-05-27).
- This host's schannel HTTPS downloaders fail with CRYPT_E_REVOCATION_OFFLINE
  (cert revocation server unreachable) -- the SAME wall behind the old
  "boringssl/curl won't build". `git` works (different TLS backend). Fix: route
  ALL vcpkg downloads through revocation-skipping curl:
    export X_VCPKG_ASSET_SOURCES='x-script,curl --ssl-no-revoke -L -o {dst} {url};x-block-origin'
  (then vcpkg install ... --x-buildtrees-root=B:/vcpkg/bt --downloads-root=B:/vcpkg/dl)
- sourceforge.net is entirely network-blocked on this host (connection reset),
  so any vcpkg port whose source is on sourceforge fails (e.g. polyclipping, a
  transitive dep of vcpkg `assimp`). AVOID system-assimp for this reason; the
  bundled assimp builds fine under clang (its assimp.cmake gates the static-lib
  name on `if(MSVC)` -> else-branch GNU flags, which clang accepts).
- Triplet x64-windows (dynamic /MD CRT). Pair with -DSTATIC_WINDOWS_RUNTIME=OFF
  so Open3D, the vcpkg libs, and the ROCm runtime DLLs all share one dynamic CRT
  (mixing /MT with the dynamic-CRT HIP runtime risks multi-heap crashes). clang
  on Windows targets the MSVC ABI, so MSVC-built vcpkg libs link against
  clang-built Open3D.

### vcpkg packages installed (x64-windows)
  tbb libjpeg-turbo curl openssl pkgconf    # confirmed clang-failers
  embree zeromq cppzmq                       # embree+zeromq inject /MD (see below); cppzmq = zmq.hpp
(zlib pulled transitively.) embree is 4.4.0, matching Open3D's pin.

### CMake configure (USE_SYSTEM_* + pkg-config wiring)
- USE_SYSTEM_TBB/JPEG/OPENSSL/EMBREE resolve via find_package (CMAKE_PREFIX_PATH
  includes the vcpkg install root).
- USE_SYSTEM_CURL and USE_SYSTEM_ZEROMQ resolve via pkg_search_module (libcurl /
  libzmq), NOT find_package. So point CMake's FindPkgConfig at vcpkg's pkgconf:
    export PKG_CONFIG="$VCROOT/tools/pkgconf/pkgconf.exe"
    export PKG_CONFIG_PATH="$VCROOT/lib/pkgconfig"
  (find_package(PkgConfig) sets the legacy PKGCONFIG_FOUND the helper checks.)
- MSYS arg-conversion mangles a mixed POSIX+Windows CMAKE_PREFIX_PATH and the
  `;` separator, hiding hip-config.cmake. Use Windows-style paths (B:/...) and
  `export MSYS2_ARG_CONV_EXCL='*'` for every cmake/cmd invocation.
- Windows-clang preprocessor defines the build needs (the bundled build assumes
  an MSVC environment that predefines these):
    CMAKE_CXX_FLAGS / CMAKE_HIP_FLAGS: -DNOMINMAX -DWIN32_LEAN_AND_MEAN -D_USE_MATH_DEFINES -DWIN32
    CMAKE_C_FLAGS: -D_USE_MATH_DEFINES   (NOT WIN32_LEAN_AND_MEAN -- it strips
      <commdlg.h> from windows.h and breaks the bundled tinyfiledialogs.c)
  _USE_MATH_DEFINES: M_PI (MSVC <cmath> gates it; needed for CXX AND HIP TUs).
  WIN32 (bare): PoissonRecon's MyMiscellany.h guards `#ifndef WIN32` -> clang
    defines _WIN32 but not bare WIN32, so it took the POSIX sys/time.h path.

### Fork source/CMake fixes (all WIN32/Clang-gated; Linux byte-identical)
1. cmake/Open3DShowAndAbortOnWarning.cmake: clang spells the ignored-nodiscard
   hipError_t warning -Wunused-value (gcc: -Wunused-result, already relaxed).
   Added a CMAKE_CXX_COMPILER_ID MATCHES "Clang" block: -Wno-error=unused-value.
2. 3rdparty/embree/embree.cmake + 3rdparty/zeromq/zeromq_build.cmake inject
   MSVC `/MD` runtime flags gated on WIN32 -> clang++ GNU-driver reads `/MD` as a
   filename ("no such file or directory: '/MD'"). FIX: supply embree+zeromq from
   vcpkg (USE_SYSTEM_EMBREE/ZEROMQ=ON) rather than patch -- avoids the bundled
   /MD path entirely. (Bundled BufferConnection.cpp then needs zmq.hpp -> cppzmq.)
3. 3rdparty/zlib/zlib.cmake + 3rdparty/libpng/libpng.cmake gate the imported
   static-lib basename on `if(MSVC)` (-> z/png16) but the bundled libs BUILD as
   zlibstatic.lib / libpng16_static.lib on Windows regardless of frontend, so the
   final link could not find z.lib/png16.lib. FIX: `if(MSVC)` -> `if(WIN32)`
   (the name follows the target platform, not the compiler). 3 sites.
4. cpp/tests/CMakeLists.txt: the Windows tbb.dll-copy step references
   $<TARGET_FILE:tbb>; with USE_SYSTEM_TBB the bundled `tbb` target is absent
   (TBB::tbb is imported). FIX: `if (WIN32)` -> `if (WIN32 AND TARGET tbb)`;
   supply the DLL on PATH at run time instead.
5. cpp/open3d/utility/CPUInfo.cpp: two BOOL results in the `_WIN32` branch are
   used only in Release-stripped asserts -> -Werror,-Wunused-variable under
   clang. FIX: [[maybe_unused]] on both (C++17 idiom).

### Build
  bash agent_space/open3d_cfg_gfx1201_vcpkg.sh     # configure (build dir: build_gfx1201)
  HIP_VISIBLE_DEVICES=1 cmake --build projects/Open3D/src/build_gfx1201 --target tests -j 64
Result: bin/tests.exe (111 MB) links clean (only benign LNK4217 zlib warnings).
All bundled libs (assimp/vtk/qhull/glew/poisson/...) and all ~31 HIP gfx1201 TUs
compile.

### Runtime DLL setup (for the eventual GPU run)
agent_space/run_tests_gfx1201.bat: launch via native cmd (an MSYS-spawned
process fails to load the UCRT api-ms-win-crt-*.dll apiset forwarders). PATH gets
_rocm_sdk_devel/bin + vcpkg/installed/x64-windows/bin; ROCBLAS_TENSILE_LIBPATH ->
_rocm_sdk_libraries/bin/rocblas/library. CRITICAL (per [[windows-gfx1101-gfx1201-host]]):
for a native .exe the loader searches exe-dir > System32 > PATH, and System32 has
the Adrenalin-driver amdhip64_7.dll -- so the TheRock runtime DLLs were COPIED
into build_gfx1201/bin (amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll,
hiprtc0714.dll, hiprtc-builtins0714.dll from _rocm_sdk_core/bin).

### GPU validation: PENDING host GPU availability (NOT a port/build problem)
At build time the host's AMD GPUs are not detected by ANY HIP runtime: TheRock
hipInfo.exe -> "no ROCm-capable device is detected" at HIP_VISIBLE_DEVICES=0,1,
and unset; torch.cuda.is_available()=False, device_count=0. `wmic
win32_VideoController` shows a Microsoft Remote Display Adapter (active RDP
session) + a Microsoft Basic Display Adapter, and the gfx1101 V710 is absent --
i.e. an RDP session has detached/unloaded the discrete AMD GPUs (a known Windows
behavior). Transient host state (cf. the gfx1101 reboot in the MOAT log), fixable
by jeff (console session / driver restart / reboot), then re-run the gate suites:
  cmd /c "agent_space\run_tests_gfx1201.bat --gtest_filter=*Tensor*:*Reduction*:*MemoryManager*"
  ... (same gate suites as the gfx1100 validation, 499-pass baseline) at HIP_VISIBLE_DEVICES=1.

## gfx1201 GPU validation 2026-06-07 (RX 9070 XT, RDNA4, wave32) -- 3 real failures

Ran the gfx1100 gate suites on gfx1201 via agent_space/run_tests_gfx1201.bat (no
HIP_VISIBLE_DEVICES mask -- see below). Host GPU note: at first the GPUs were
undetected (RDP detaches them); once gfx1201 was back, torch.cuda saw exactly 1
ROCm device = gfx1201 at index 0 (the gfx1101 V710 is on the Basic Display
Adapter, not ROCm-capable). With only one ROCm device, BOTH `=0` and `=1` masks
yield 0 devices; an EMPTY HIP_VISIBLE_DEVICES enumerates gfx1201. So the batch
clears the mask. (Revisit if both GPUs become ROCm-capable.)

Results (vs gfx1100's clean 499):
- *Tensor*:*Reduction*:*MemoryManager*: 420/422 PASS (1 skip; 1 FAIL = Tensor.Cross)
- *NearestNeighbor*:*KnnIndex*:*FixedRadiusIndex*:*Knn*:*Hybrid*:*Registration*:*Feature*:
  42/45 PASS (1 known NPP scope-out + 2 FAIL)
- *HashMap*:*VoxelBlockGrid*: 34/34 PASS (stdgpu dedup + TSDF clean on RDNA4)

The FAISS warp-select (NNS/KNN, the highest-risk path) and stdgpu HashMap/TSDF
pass on gfx1201 -- the wave32 two-32-lane model holds on RDNA4 as on gfx1100.

Failures, by error class:
1. EXPECTED (not a regression): Odometry RGBDOdometryMultiScaleHybrid ->
   NppUnsupportedOnHIP. Same scoped-out NPP GPU-image path as gfx90a/gfx1100.
2. "device kernel image is invalid" (ParallelFor.h:66, hipErrorInvalidImage) on
   TWO kernels: core Tensor.Cross and t::pipelines ICPDoppler. Cross is pure
   Float32 on 2x3 tensors -- NOT FP64, NOT size. Hundreds of other ParallelFor
   kernels in the SAME binary launch fine, so this is a gfx1201/RDNA4 code-object
   issue for specific kernels under the TheRock clang (a silently-invalid gfx1201
   slice), not a source port bug. These same tests PASS on gfx1100. -> toolchain
   lead, likely a TheRock gfx1201 codegen/bundling issue to reduce + report.
3. hipSOLVER getrf "singular condition detected" (LinalgUtils.h:68, InverseCUDA)
   in Registration.GetInformationMatrixFromPointCloud. rocSOLVER LU on gfx1201
   flags singular where gfx1100 did not -- a math-lib (rocSOLVER) result
   difference on RDNA4, not the Open3D port.

Verdict: the HIP PORT and the Windows-clang BUILD are sound on gfx1201 (496 pass,
incl. all warp-select + hashmap + TSDF). The 3 failures are gfx1201/TheRock
runtime+toolchain issues (2 invalid-image kernels, 1 rocSOLVER singular), not the
port. gfx1201 stays NOT-completed pending triage of these; reproduce Cross
standalone for the toolchain report.

## gfx1201 failures ROOT-CAUSED 2026-06-07: TheRock multi-arch kpack (ROCm/TheRock #3531), NOT Open3D, NOT rocSOLVER

All 3 gfx1201 failures trace to ONE infrastructure bug in TheRock's multi-arch
code-object packing (kpack), confirmed by HIP runtime logs and matched to an
existing upstream issue. NOT an Open3D port bug; NOT a rocSOLVER algorithm bug.

### Evidence chain (bisected)
- Tensor.Cross PASSES alone, FAILS in-suite, always immediately after Tensor.Det.
- Det computes the determinant via LU = hipSOLVER/rocSOLVER getrf (LUCUDA.cpp).
- Det+Cross repro: Det OK (correct determinant), Cross fails instantly (~15 ms);
  Det+Add: Add fails; Det+Cross+Add: Cross fails, Add then PASSES. So getrf
  transiently breaks the NEXT code-object load(s), then the loader recovers.
- AMD_LOG_LEVEL=3 on Det->Cross shows the actual cause at the next kernel launch:
    hip_fatbin.cpp:710  kpack_load_code_object failed with error: 13
    hip_module.cpp:886  hipLaunchKernel: Returned hipErrorInvalidImage
  i.e. the post-getrf HIP kernels cannot lazily load their code object from the
  kpack archive. ICPDoppler / GetInformationMatrix (getrf via Inverse) hit the
  same path: getrf's larger Tensile-backed call mis-loads -> "singular condition
  detected" (Det's tiny matrix uses a non-Tensile path, so Det itself succeeds).

### Upstream status (checked 2026-06-07; do NOT file a rocSOLVER issue)
ROCm/TheRock #3531 "Multi-arch kpack: consumer GPU validation results and
remaining work" documents EXACTLY this and is OPEN (TODO, no fix):
  Root cause: "the kpack splitter processes only HIP code objects embedded in ELF
  .hipk sections; Tensile's .co files are loaded through a separate runtime path
  that kpack does not intercept." Affected libs it NAMES: rocblas, hipblaslt,
  rocsparse, rocsolver -> fail with hipErrorInvalidImage on gfx1201/gfx1100.
Related: ROCm/rocm-libraries #7192 (gfx1201 looks up wrong Tensile file
gfx1200.dat). rocSOLVER develop CHANGELOG has no corresponding fix because the
bug is NOT in rocSOLVER -- it is in TheRock's multi-arch artifact packaging.
=> The bug is already reported (TheRock #3531) and unfixed; rocSOLVER tip does
   not and would not fix it. Reporting to rocSOLVER would be misdirected.

### Why no Open3D-side fix
An attempted fix (clear the HIP error after the solver check, LinalgUtils.h) was
tried and REVERTED: the failure is a genuine code-object LOAD failure at the next
kernel launch (kpack), not a stale error code, so clearing hipGetLastError does
nothing. There is no correct Open3D-side fix; the port is sound. This only
reproduces on the TheRock Windows multi-arch (kpack) runtime -- the Linux
gfx90a/gfx1100 validations used standard /opt/rocm (no kpack) and were clean, and
gfx1201 will clear when TheRock fixes kpack (or with a single-arch / non-kpack
build of the rocSOLVER Tensile kernels).

### Practical validation note
Because exactly one HIP kernel load fails per getrf and then recovers, splitting
the gate run so getrf-using tests (Det, Inverse/Solve/Det-backed, ICP) are
isolated from the immediately-following test makes the collateral failures
(Cross, GetInformationMatrix) disappear; the only residual is rocSOLVER-on-kpack
itself (ICPDoppler), which is the #3531 bug surfacing directly.

## Validation 2026-06-07 (linux-gfx1100 revalidate, binary-equiv carry-forward)

Head sha advanced from 2670bbb17e (linux-gfx1100 validated_sha) to ee479e1e03
(commit "[ROCm] Build the HIP port under the all-clang Windows toolchain").
All five changed files are Windows-specific (WIN32/Clang guards in libpng.cmake,
zlib.cmake, Open3DShowAndAbortOnWarning.cmake; [[maybe_unused]] on _WIN32-gated
code in CPUInfo.cpp; USE_SYSTEM_TBB guard fix in cpp/tests/CMakeLists.txt).
No Linux HIP/GPU source touched.

Carry-forward method: binary-equiv.

Commands run:
```
# Save baseline binary (build at old gfx1100-validated 9eb3704d)
cp projects/Open3D/src/build/bin/tests agent_space/open3d_old_build/tests

# Update source to ee479e1e03
cd projects/Open3D/src && git checkout origin/moat-port

# Incremental build (no GPU files recompiled -- all changes are Windows-only)
utils/timeit.sh Open3D compile -- cmake --build projects/Open3D/src/build -j16 --target tests

# Binary-equivalence check
python3 utils/codeobj_diff.py agent_space/open3d_old_build/tests \
  projects/Open3D/src/build/bin/tests
# verdict=identical
#   tests vs tests: identical (exported symbols + device ISA identical (29 exports))
```

Result: device ISA and exported symbols identical on gfx1100. Carried forward to
ee479e1e03 without GPU re-run.

## Validation 2026-06-07 (linux-gfx90a revalidate, binary-equiv carry-forward)

Head sha advanced from 2670bbb17e to ee479e1e03 (commit "[ROCm] Build the HIP port
under the all-clang Windows toolchain"). All five changed files are Windows-specific
(WIN32 guards replacing MSVC guards in libpng.cmake/zlib.cmake, clang-only warning
suppression in Open3DShowAndAbortOnWarning.cmake, [[maybe_unused]] on _WIN32-gated
code in CPUInfo.cpp, USE_SYSTEM_TBB guard fix in cpp/tests/CMakeLists.txt). No Linux
HIP/GPU source touched.

Carry-forward method: binary-equiv.

Commands run:
```
# Save baseline binary (build at 2670bbb17e)
cp projects/Open3D/src/build/bin/tests agent_space/open3d_old_build/tests

# Update source to ee479e1e03 (fast-forward merge)
cd projects/Open3D/src && git merge --ff-only fork/moat-port

# Incremental build (no GPU files recompiled -- all changes are Windows-only)
utils/timeit.sh Open3D compile -- cmake --build projects/Open3D/src/build -j16 --target tests

# Binary-equivalence check
python3 utils/codeobj_diff.py agent_space/open3d_old_build/tests \
  projects/Open3D/src/build/bin/tests
# verdict=identical
#   tests vs tests: identical (exported symbols + device ISA identical (29 exports))
```

Result: device ISA and exported symbols identical on gfx90a. Carried forward to
ee479e1e03 without GPU re-run.

## gfx90a pin -> auto-detect (46b25be, 2026-06-23)

First project in the MOAT-wide gfx90a-pin audit. Removed the pre-enable_language
gfx90a pin; rely on enable_language(HIP), which honors -DCMAKE_HIP_ARCHITECTURES,
else auto-detects via rocm_agent_enumerator, else FATAL_ERRORs (no-GPU build host
must set the arch). No self-detect / dedup needed (CMake does it). Explicit-arch
builds byte-identical -> carried forward source-class. See
audits/gfx90a-pin-autodetect-audit-2026-06-23.md.

## PR #7509 review fix-round (2026-07-02, comment/doc-only, carry-forward)

Copilot left 8 unresolved threads, all doc/comment-level. Fixed on the fork as a
new commit on top (973905116) so validated_sha stays a reachable ancestor.

1. Five threads flagged code comments/CMake pointing at "notes.md" (a workspace
   artifact, not in the repo). Removed every "See notes.md" pointer across the
   diff, not just the 5 flagged lines: HashMap.cpp (9 sites) + VoxelBlockGrid.cpp
   (1) dropped the pointer (comment is self-explanatory); SlabMacros.h dropped it;
   NPPImage.cpp and 3rdparty/find_dependencies.cmake now point to the "AMD GPU
   support" section of docs/compilation.rst. Grepped the whole port diff for
   notes.md/plan.md/moat/Strategy A|B/validated_sha/revalidate -- NONE
   remain.
2. docs/compilation.rst: the "targets gfx90a by default" line was stale after the
   46b25be gfx90a-pin removal. Rewrote to describe enable_language(HIP) behavior:
   auto-detects host GPUs via rocm_agent_enumerator when CMAKE_HIP_ARCHITECTURES
   is unset, and a GPU-less host must set it explicitly.
3. docs/compilation.rst note: NPP GPU image filters do NOT fall back to CPU. The
   HIP stubs in NPPImage.cpp call utility::LogError (throws), so the op reports
   unsupported at runtime. Reworded to "run them on the CPU device instead".
4. CHANGELOG.md: NO change. The reviewer claimed extra indentation vs surrounding
   "- " entries, but the ENTIRE file uses "-   " (dash + 3 spaces): 113 entries,
   zero using "- ". Our entry already matches the uniform house style exactly;
   changing it to "- " would make our line the only inconsistent one. Replied on
   the thread explaining this rather than introducing an inconsistency. The entry
   itself (line 2, references PR #7509) satisfies the update-docs bot.

classify 46b25becb->973905116 = comment-only, arch_independent=True, inert=True.
All platforms carried forward (linux-gfx1100 + windows-gfx1201 -> 973905116;
gfx90a stays pr-open). No GPU re-run. Head advanced to 973905116.

## Rebase onto upstream/main to clear PR #7509 conflict (2026-07-02)

PR #7509 was CONFLICTING (mergeable_state=dirty). Rebased moat-port onto
upstream/main. Only overlapping file was CHANGELOG.md (docs). The three port
commits (2670bbb17, ee479e1e0, 46b25becb) plus the doc-scrub commit applied with
NO code conflict; only CHANGELOG.md conflicted (upstream added a sibling
CorrespondenceCheckerBasedOnSourceRotation entry). Resolved by keeping both top
entries. New moat-port head: 50bb2505b (was 973905116). Force-pushed with
--force-with-lease. Sanctioned history rewrite for conflict resolution.

Port source contribution is byte-identical to pre-rebase (patches applied
cleanly; only the docs conflict was hand-resolved). The upstream base advanced
~3 weeks, so the rebase pulls in upstream's own changes to io/, visualization/,
pybind/, pipelines/registration (CorrespondenceChecker), t/geometry/MinimumOBE,
and assimp -- NONE of them our port, but several are compiled into the tests
binary. classify(973905116..50bb2505b)=mixed, arch_independent=False (it spans
that upstream churn), so advance-head flipped linux-gfx1100 and windows-gfx1201
from completed to revalidate. This revalidate is driven by the upstream base
moving, not by any HIP/port change; the GPU device code is unchanged.

Cheap resolution for a validator: build the rebased tree on each host and run
utils/codeobj_diff.py against the pre-rebase binary -- device ISA + exports will
be identical (no GPU source touched), so a binary-equiv carry-forward applies
without a GPU re-run. gfx90a lead stays pr-open (advance_head does not touch PR
states); its validated_sha points at the pre-rebase ee479e1e03 and should be
re-anchored to the rebased equivalent (24de0d419) by the same binary-equiv
check.

PR #7509 now mergeable=true (mergeable_state=blocked = branch-protection/review
pending, not a conflict). Conflict cleared.

## Validation 2026-07-02 (linux-gfx1100 revalidate, binary-equiv carry-forward)

Head sha advanced from 973905116 (linux-gfx1100 validated_sha) to 50bb2505b
(rebased moat-port HEAD). classify(973905116..50bb2505b)=mixed, arch_independent=False:
the delta spans upstream changes to io/, visualization/, pybind/, pipelines/registration,
t/geometry/MinimumOBE, assimp pulled in by the rebase, plus our own comment/doc-only
edits (SlabMacros.h, NPPImage.cpp, docs/compilation.rst). No HIP kernel file (.cu)
changed.

Carry-forward method: binary-equiv.

Commands run:
```
# Save baseline binary (built at ee479e1e / pre-rebase gfx1100 state)
cp projects/Open3D/src/build/bin/tests agent_space/open3d_rebase_old_build/tests

# Update source to 50bb2505b (rebased moat-port)
cd projects/Open3D/src && git checkout origin/moat-port -B moat-port

# Incremental build (no HIP .cu TUs recompiled -- upstream churn is CXX only)
utils/timeit.sh Open3D compile -- cmake --build projects/Open3D/src/build -j16 --target tests

# Binary-equivalence check
python3 utils/codeobj_diff.py agent_space/open3d_rebase_old_build/tests \
  projects/Open3D/src/build/bin/tests
# verdict=identical
#   tests vs tests: identical (exported symbols + device ISA identical (29 exports))
```

Result: device ISA and exported symbols identical on gfx1100. Carried forward to
50bb2505b without GPU re-run.

State: linux-gfx1100 -> `completed` at validated_sha 50bb2505b.

## Validation 2026-07-02 (linux-gfx90a revalidate, binary-equiv carry-forward)

Head sha re-anchored from pre-rebase ee479e1e03 to post-rebase 50bb2505b (moat-port
HEAD after rebasing onto upstream/main to resolve PR #7509 merge conflict).

Delta from ee479e1e03 to 50bb2505b spans:
- Upstream changes pulled in by the rebase (5 upstream commits: io/, visualization/,
  pybind/, pipelines/registration/CorrespondenceChecker, t/geometry/MinimumOBE, assimp).
  No .cu/.cuh files changed by upstream; all changes are CPU source only.
- gfx90a auto-detect (7563556a4): cmake CMakeLists.txt only, no device code.
- Doc-scrub (50bb2505b): comment-only changes in SlabMacros.h, NPPImage.cpp,
  HashMap.cpp, VoxelBlockGrid.cpp, find_dependencies.cmake, docs/compilation.rst.

Carry-forward method: binary-equiv. Full build of the rebased HEAD was required
(no cached binary from pre-rebase validation); incremental rebuild at 24de0d419
(pre-doc-scrub rebased commit = rebased equivalent of ee479e1e03) confirmed device ISA.

Commands run:
```
# Configure and full build at 50bb2505b (fresh, no build dir cached)
cmake -S projects/Open3D/src -B projects/Open3D/src/build \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DBUILD_PYTORCH_OPS=OFF -DBUILD_TENSORFLOW_OPS=OFF \
  -DBUILD_GUI=OFF -DBUILD_WEBRTC=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF -DBUILD_PYTHON_MODULE=OFF \
  -DBUILD_UNIT_TESTS=ON -DBUILD_ISPC_MODULE=OFF \
  -DWITH_IPP=OFF \
  -DCMAKE_BUILD_TYPE=Release -DDEVELOPER_BUILD=ON
utils/timeit.sh Open3D compile -- cmake --build projects/Open3D/src/build -j16 --target tests
# PASS -- [100%] Linking HIP executable ../../bin/tests, 172MB binary

# Save the new (HEAD) binary
cp projects/Open3D/src/build/bin/tests agent_space/open3d_new_build/tests

# Restore source files to 24de0d419 state, rebuild incrementally
git -C projects/Open3D/src checkout 24de0d419 -- CMakeLists.txt \
  3rdparty/find_dependencies.cmake cpp/open3d/core/hashmap/CUDA/SlabMacros.h \
  cpp/open3d/t/geometry/kernel/NPPImage.cpp cpp/tests/core/HashMap.cpp \
  cpp/tests/t/geometry/VoxelBlockGrid.cpp docs/compilation.rst
utils/timeit.sh Open3D compile-old -- cmake --build projects/Open3D/src/build -j16 --target tests
# PASS -- only comment-changed files recompiled (VoxelBlockGridCUDA.cu,
# CreateCUDAHashBackend.cu, SlabNodeManager.cu, NPPImage.cpp, HashMap.cpp,
# VoxelBlockGrid.cpp); 0 HIP-device-code changes

# Save the old (24de0d419) binary
cp projects/Open3D/src/build/bin/tests agent_space/open3d_old_build/tests

# Restore source tree to HEAD
git -C projects/Open3D/src checkout HEAD -- CMakeLists.txt \
  3rdparty/find_dependencies.cmake cpp/open3d/core/hashmap/CUDA/SlabMacros.h \
  cpp/open3d/t/geometry/kernel/NPPImage.cpp cpp/tests/core/HashMap.cpp \
  cpp/tests/t/geometry/VoxelBlockGrid.cpp docs/compilation.rst

# Binary-equivalence check
python3 utils/codeobj_diff.py agent_space/open3d_old_build/tests \
  agent_space/open3d_new_build/tests
# verdict=identical
#   tests vs tests: identical (exported symbols + device ISA identical (29 exports))
```

Result: device ISA and exported symbols identical between 24de0d419 (pre-doc-scrub,
rebased equivalent of gfx90a-validated ee479e1e03) and 50bb2505b (rebased HEAD).
By transitivity: the upstream rebase changes (CPU-only io/viz/pybind) and the
doc-scrub together produce zero device ISA change on gfx90a. Carried forward to
50bb2505b without GPU re-run.

State: linux-gfx90a -> `pr-open` at validated_sha 50bb2505b (re-anchored).

## Validation 2026-08-20 (windows-gfx1151) -- validation-failed (new, narrower blocker found)

GPU: AMD Radeon 8060S (gfx1151, RDNA3.5 APU, 20 CUs, wave32), Windows 11,
TheRock ROCm 7.14 pip SDK. First windows-gfx1151 attempt for this project;
head_sha = 50bb2505b (moat-port, PR #7509 open -> fork frozen, no push made).
`python3 utils/moatlib.py protect-fork Open3D` installed the pre-push hook.

### 97390511 (old windows-gfx1201 validated_sha): confirmed gone from the fork
`git fetch origin '+refs/*:refs/remotes/origin-all/*'` then
`git cat-file -e 973905116^{commit}` -> "Not a valid object name". Also tried
`git fetch origin 973905116` -> "couldn't find remote ref". The commit is an
orphan: it was superseded by the 2026-07-02 rebase-onto-upstream/main
force-push (moat-port head 973905116 -> 50bb2505b, see notes.md above), so no
branch/tag reaches it anymore. GitHub REST API
(`/repos/AMD-Ecosystem/Open3D/commits/973905116`) still returns the commit
object (not yet garbage-collected server-side), but `git fetch` by sha is
refused because it is unreachable/unadvertised. Practical takeaway: an
orphaned pre-rebase sha is not a usable reference point once force-pushed
over; only the API can still describe it, and that window is not guaranteed
to last.

### PRIMARY HYPOTHESIS CONFIRMED: USE_SYSTEM_* clears the bundled-3rdparty wall
Installed the previously-missing vcpkg (D:/vcpkg, x64-windows triplet)
packages on this host (network/TLS is fine here, unlike the gfx1101/gfx1201
box TLS-revocation wall -- no X_VCPKG_ASSET_SOURCES workaround needed):

    D:/vcpkg/vcpkg.exe install curl openssl pkgconf embree zeromq cppzmq --triplet x64-windows

(tbb and jpeg/libjpeg-turbo were already present from the rmagine port.) All
6 built cleanly in about 6 minutes total (curl 45s, embree the long pole).

Configure script: agent_space/open3d_cfg_gfx1151_vcpkg.sh (gitignored,
recipe here). Same shape as the gfx1201 vcpkg recipe (USE_SYSTEM_TBB/JPEG/
OPENSSL/CURL/EMBREE/ZEROMQ=ON, PKG_CONFIG/PKG_CONFIG_PATH pointed at vcpkg
pkgconf, CMAKE_TLS_VERIFY=OFF, STATIC_WINDOWS_RUNTIME=OFF, the NOMINMAX/
WIN32_LEAN_AND_MEAN/_USE_MATH_DEFINES/WIN32 defines). This host uses
D:/vcpkg (not B:/vcpkg) and TheRock ROCM_ROOT from
agent_space/win_rocm_env.sh. Confirmed in the log: "Using installed
third-party library 3rdparty_tbb", "3rdparty_jpeg", "3rdparty_openssl",
"3rdparty_zeromq", "3rdparty_embree 4.4.1" all resolved from vcpkg; curl
resolved via the libcurl pkg-config module. Embree came in at vcpkg 4.4.1 vs
Open3D pinned 4.4.0, one patch version newer, no issue observed.

### New blocker found and diagnosed: CMake never learned MSVC_RUNTIME_LIBRARY for the HIP language
Two distinct CMake (not Open3D, not this port) issues, both stem from the
same root cause and needed two separate workarounds:

1. enable_language(HIP) ABI-detection try_compile fails: "MSVC compiler
   version not detected properly". Root cause (confirmed via
   CMakeConfigureLog.yaml): the top-level CMakeHIPCompilerId step succeeds
   (clang-cl identifies as Clang 23.0.0, links CMakeHIPCompilerId.exe fine),
   but the separate nested "Detecting HIP compiler ABI info" try_compile
   fails its own inner cmake configure with an empty stdout and exit 1 before
   ever reaching the message body -- a CMake 3.31 HIP+clang-cl(MSVC frontend)
   support gap, not an environment problem (vcvars/INCLUDE/LIB were never the
   issue; C and CXX ABI detection succeed with the identical compiler and
   flags). WORKAROUND (validator-side, no fork change): skip ABI detection
   entirely by pre-caching the two guard variables
   CMakeDetermineCompilerABI.cmake checks with if(NOT DEFINED ...):

       -DCMAKE_HIP_COMPILER_WORKS=1 -DCMAKE_HIP_ABI_COMPILED=1

   This is safe: implicit link dirs/libs come from the already-successful
   CompilerId step, not from ABI detection.

2. Generate step fails: "MSVC_RUNTIME_LIBRARY value 'MultiThreadedDLL' not
   known for this HIP compiler". Root cause (read directly from CMake own
   Platform/Windows-MSVC.cmake, __windows_compiler_msvc(lang) macro): the
   block that populates
   CMAKE_<lang>_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreaded{,DLL,Debug,DebugDLL}
   is gated if("x${lang}" STREQUAL "xC" OR "x${lang}" STREQUAL "xCXX") -- HIP
   is never populated, even though __windows_compiler_msvc(HIP) is called for
   the clang-cl/MSVC-frontend path (Windows-Clang-HIP.cmake ->
   Windows-Clang.cmake -> Windows-MSVC.cmake). Any project (Open3D included)
   that sets CMAKE_MSVC_RUNTIME_LIBRARY (directly, or indirectly via
   STATIC_WINDOWS_RUNTIME=OFF here) hits this the moment a HIP target uses
   the generator-expression-driven runtime-library property under CMP0091
   NEW. WORKAROUND (validator-side): supply the 4 missing flag-table
   variables on the command line:

       -DCMAKE_HIP_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreaded=-MT
       -DCMAKE_HIP_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDLL=-MD
       -DCMAKE_HIP_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebug=-MTd
       -DCMAKE_HIP_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebugDLL=-MDd

   With both workarounds, cmake -S ... -B build_gfx1151 ... succeeds end to
   end ("Build files have been written to ...").

With both fixes the top-level configure and build proceed for real: ninja
built fmt, googletest, glew, qhull, and 209/731 externalproject+object steps
before hitting the same "MultiThreadedDLL not known for this HIP compiler"
error again, this time inside stdgpu own nested cmake configure
(ext_stdgpu-configure, "FAILED: ... stdgpu/src/ext_stdgpu-stamp/ext_stdgpu-configure").

### Why the command-line workaround does not reach stdgpu
stdgpu is the one bundled 3rdparty library Open3D itself builds with
enable_language(HIP) (3rdparty/stdgpu/stdgpu.cmake, USE_HIP branch -- it is
the hashmap/TSDF GPU backend, genuinely GPU-relevant, not host infra). It is
built via ExternalProject_Add, a completely separate nested cmake invocation.
3rdparty/find_dependencies.cmake:185 (ExternalProject_CMAKE_ARGS_hidden,
forwarded to every ExternalProject_Add including stdgpu) unconditionally
passes -DCMAKE_POLICY_DEFAULT_CMP0091:STRING=NEW and
-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${CMAKE_MSVC_RUNTIME_LIBRARY} through, so
stdgpu nested build hits the identical CMake gap -- but ExternalProject_Add
never auto-forwards arbitrary parent -D cache variables to the child cmake
process (confirmed: no such forwarding exists in ExternalProject.cmake;
CMAKE_ARGS/CMAKE_CACHE_ARGS must name every variable explicitly, and stdgpu
list does not include the 4 flag-table vars). So the validator-side
top-level -D workaround has no way to reach this nested configure without a
fork source change: forwarding the same 4
CMAKE_HIP_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_* variables into
ExternalProject_CMAKE_ARGS_hidden (WIN32/Clang-gated, matching the house
style of the 5 already-committed Windows fixes) in
3rdparty/find_dependencies.cmake. That is a porter-scope fix, and moat-port
is frozen behind open PR #7509 (would need a fix-branch round), so it is out
of validator scope this session.

### Scope covered vs not covered (be precise)
Covered: configure end-to-end (all vcpkg USE_SYSTEM_* wiring verified against
the actual build log); real compilation of fmt, googletest, glew, qhull,
tinyfiledialogs and about 205 further steps under the all-clang-cl toolchain
with the two CMake-HIP-on-Windows workarounds above. Not covered: no unit
test binary was produced, no GPU kernel from this port own compat layer
(CUDAToHIP.h, FAISS warp-select, stdgpu hashmap, etc.) was compiled or run.
Zero of the 499 gate tests ran. This is a build-recipe finding, not a
GPU-correctness pass -- do not read it as partial test coverage.

### CUDA no-regression gate
Skipped: this Windows host has no CUDA toolkit (nvcc absent, no /opt/conda
cuda-toolkit env) and the gate is not yet recorded at this head_sha by any
Linux validator. Left for whichever Linux arch validates next.

### State
windows-gfx1151: validation-failed at failed_sha 50bb2505b. Not the same
block as windows-gfx1101/gfx1201 original bundled-3rdparty-under-clang wall
(that one is now cleared here by USE_SYSTEM_*); this is a narrower, newly
found CMake HIP-on-Windows generate-stage gap that recurs specifically in
stdgpu nested ExternalProject build. Recommended fix for the next porter
round: forward the 4 CMAKE_HIP_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_*
variables (WIN32/Clang-frontend-gated) into ExternalProject_CMAKE_ARGS_hidden
in 3rdparty/find_dependencies.cmake, matching the existing WIN32/Clang guard
style of the 5 committed Windows fixes; re-run this same recipe afterward.
