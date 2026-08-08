# GooFit notes

## Port attempt 1 (2026-06-05)

### Summary
GooFit is a massively-parallel fitting framework using Thrust. The port requires Strategy A (CMake with cuda_to_hip.h compat header), compiling all sources with hipcc since rocThrust headers require the HIP compiler.

### Completed
1. Created `include/goofit/detail/cuda_to_hip.h` CUDA-to-HIP compat header
2. Modified CMakeLists.txt:
   - Added `USE_HIP` option
   - Enabled HIP language with C++17 (required for rocThrust)
   - Set `THRUST_DEVICE_SYSTEM_HIP` for rocThrust compatibility
   - Mark all `.cpp` and `.cu` files as HIP language (rocThrust headers need hipcc)
   - Disabled CUDA-specific `-Xcompiler` flags for HIP
   - Disabled IPO for HIP builds
3. Modified GlobalCudaDefines.h:
   - Added HIP support to THRUST_DEVICE_SYSTEM checks
   - Updated compiler detection for HIPCC
4. Renamed Application.cpp to Application.cu and added HIP device info output

### Build result
Compilation succeeds but linking fails with:
```
lld: error: undefined hidden symbol: GooFit::MetricTaker::operator()(thrust::tuple<...>) const
```

### Root cause analysis
The linker error indicates a device code visibility issue specific to HIP/ROCm:

1. `MetricTaker` is a functor with a `__device__ operator()` defined in a header
2. GooFit uses this functor in `thrust::transform_reduce` calls
3. On CUDA, separable compilation allows device code to be linked across TUs
4. On HIP/ROCm 7.2.1, the rocPRIM device code templates instantiate with `hidden` visibility by default, making cross-TU device symbol resolution fail

This is NOT a simple porting fix -- it requires either:
- Making device code visible across TUs (e.g., `-fgpu-rdc` + device link, or explicit visibility attributes)
- Restructuring GooFit to keep all device code instantiation in a single TU
- Using a different approach for the Thrust functors

### Blocking reason
HIP device code visibility/linking differs from CUDA separable compilation. GooFit's architecture (device functors in headers, used across multiple TUs via Thrust algorithms) exposes this difference. The fix requires understanding GooFit's device code structure and may need upstream changes.

### Files changed (uncommitted in jeffdaily fork)
- CMakeLists.txt
- include/goofit/GlobalCudaDefines.h
- include/goofit/detail/cuda_to_hip.h (new)
- src/goofit/CMakeLists.txt
- src/goofit/Application.cu (renamed from .cpp)

## Port attempt 2 (2026-06-11, linux-gfx90a) -- link blocker RESOLVED

Started from scratch (attempt 1 was never committed/pushed). Fork cloned at
projects/GooFit/src, branch moat-port. Pushed: AMD-Ecosystem/GooFit @ moat-port
d95236e57.

### Original blocker (MetricTaker / cross-TU device visibility): RESOLVED
The "undefined hidden symbol: MetricTaker::operator()" link failure is the
RXMesh fault class (PORTING_GUIDE 2026-05-30): __device__ members declared in
headers, defined in separate .cu TUs, used across the library need relocatable
device code on HIP. The fix has three parts that ALL must be present:
1. -fgpu-rdc on every GooFit HIP TU + HIP_SEPARABLE_COMPILATION ON.
2. Mark BOTH .cu and the .cpp files that include rocThrust as LANGUAGE HIP.
3. CRITICAL and non-obvious: the HIP device link only sees device objects
   passed DIRECTLY on the link line, never objects inside a .a archive. GooFit
   is many small static libs, so the device link found no device objects and
   left __hip_fatbin_*/__hip_gpubin_handle_* undefined. Solution: build the
   GooFit libraries as OBJECT libraries and gather them into ONE shared library
   (goofit_lib) that does a single device link spanning all TUs (-fgpu-rdc +
   --hip-link on that link). This resolves every cross-TU __device__/__constant__
   global and the device function-pointer tables in one shot.

### Design
- GOOFIT_DEVICE=HIP backend parallel to CUDA. THRUST_DEVICE_SYSTEM stays HIP
  (rocThrust auto-selects it under hipcc; do NOT force CUDA).
- include/goofit/detail/cuda_to_hip.h: CUDA runtime -> hip runtime symbol map,
  force-included (-include) on HIP TUs via the GooFit target functions.
- include/goofit/detail/CudaCompat.h: GOOFIT_DEVICE_IS_GPU (true for CUDA OR HIP
  Thrust system). All GooFit GPU-path guards switched from
  "THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA" to this macro.
- C++17 (rocThrust requires it; GooFit default was 11). IPO disabled on HIP.
- Skip extern/thrust on HIP (rocThrust via roc::rocthrust).

### Other genuine fixes
- MetricTaker.cu binned operator(): device new[]/delete[] -> fixed per-thread
  array fptype[MAX_NUM_OBSERVABLES] (HIP device malloc heap is small/unreliable;
  arch-unified). This alone fixed the alpha parameter in GaussianTest.
- RO_CACHE(x) -> plain (x) on HIP (HIP __ldg only takes scalar types).
- StepPdf.cu: removed unused "device_function_ptr hptr_to_Step = device_Step"
  host global (taking a __device__ function address in host code; undefined on
  HIP, dead on CUDA).
- Log.h / ParameterContainer.cu __CUDACC__ / __CUDA_ARCH__ guards made
  HIP-aware (__HIPCC__ / __HIP_DEVICE_COMPILE__).

### Build status (gfx90a, ROCm 7.2.1, -DGOOFIT_PHYSICS=OFF)
Library + basic/combine PDFs + 24 test/example executables BUILD and LINK.
Configure/build script: projects/GooFit/build_hip.sh.

### NEW blocker: unbinned NLL fits read garbage normalization on device
Compilation/linking work, but unbinned maximum-likelihood fits diverge to a
parameter bound on HIP. Reproduced minimally (ExpPdf, generate exp(rate=1.5),
fit alpha): integrate()=0.5 and normalize()=0.5 are CORRECT (host analytic
path), and evaluateAtPoints per-event device eval is CORRECT (verified
v[i] = normalized Gaussian to 4 digits), but the fit gives alpha=+10 (upper
bound) instead of -1.5.
Root cause narrowed: instrumenting calculateNLL's "norm" argument
(pc.getNormalization(0), which reads the __device__ fptype* d_normalizations
published via hipMemcpyToSymbol from SmartVectorGPU::sync) shows early fit
iterations read valid norms (~0.67) but LATER iterations read 6.25e-310 -- the
classic uninitialized-double bit pattern. So the device reads stale/garbage
normalization values mid-fit. normRanges (a raw gooMalloc pointer passed
directly to thrust) works; the symbol-published d_normalizations does not stay
valid across iterations.
Tried and did NOT fix: hipDeviceSynchronize after the H2D copy in sync()
(so it is not a simple thrust-stream race). Per-iteration parameter updates use
SmartVector::smart_sync (writes changed device_copy[i] without re-publishing the
pointer); the normalization uses full sync (re-publishes). Suspect a rocThrust
device_vector storage/lifetime interaction with the pointer stored in the
__device__ symbol, or smart_sync's per-element device_reference writes not
landing. Next attempt: dump d_normalizations contents on device right before the
reduce vs host_normalizations; check whether device_copy realloctes; consider
replacing the SmartVector symbol-pointer scheme on HIP with a stable gooMalloc'd
buffer (like normRanges) that is hipMemcpy'd each sync.

### Deferred: physics PDFs + MCBooster (GOOFIT_PHYSICS=OFF)
The amplitude-analysis PDFs (Amp3Body/Amp4Body/kMatrix) need MCBooster ported
and a device-side Eigen complex 5x5 matrix inverse. MCBooster WIP is saved in
projects/GooFit/mcbooster-hip-wip.patch (Vector3R/4R __host__ __device__ on
out-of-line defs, Config.h HIP backend, thrust::cuda::par -> thrust::hip::par,
GContainers HIP device_vector). MCBooster is a submodule (GooFit/MCBooster) so a
real port needs an MCBooster fork + submodule pointer bump; left
uncommitted to avoid dangling the pointer. Remaining MCBooster/physics errors:
Eigen Array<thrust::complex>/Array<double> operator() not EIGEN_DEVICE_FUNC
(compute_inverse5.h, FOCUS.cu); omp_get_thread_num in HIP path of
Generate.h/EvaluateArray.h; thrust hip_rocprim vs cpp tag mismatch in copy_if;
__thrust_forceinline__ unknown in GSpline.cu.

## Resuming (2026-08-07)

The port continues: this was judged a port worth finishing rather than an unportable
codebase, so linux-gfx90a is no longer marked blocked. The blocker as last recorded, which
is where to pick it up:

Build/link blocker RESOLVED (cross-TU device visibility fixed via -fgpu-rdc + OBJECT-libs-to-single-shared-lib device link; core+basic+combine PDFs and 24 test/example exes build and link on gfx90a). NEW blocker: unbinned NLL fits diverge to a parameter bound because the device reads garbage normalization values (6.25e-310 uninitialized-double pattern) from the hipMemcpyToSymbol-published __device__ d_normalizations during the per-event reduction; analytic normalize() and per-event device eval are correct in isolation. See notes.md attempt 2 for the minimal repro and next steps. Physics PDFs scoped out (GOOFIT_PHYSICS=OFF; MCBooster+device Eigen complex inverse deferred, WIP patch saved).
