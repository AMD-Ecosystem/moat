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

## Port attempt 3 (2026-08-08, linux-gfx1100) -- builds and PASSES; blocked only on fork write access

Resumed from AMD-Ecosystem/GooFit @ moat-port d95236e57 (the gfx90a work). No
source change was needed to make it build or pass on gfx1100.

### Build (gfx1100, ROCm 7.2.53211-c2d9476115, CMake 3.31.6, Ninja)
```
cmake -S . -B build-hip -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGOOFIT_DEVICE=HIP -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DGOOFIT_TESTS=ON -DGOOFIT_EXAMPLES=ON -DGOOFIT_CERNROOT=OFF \
  -DGOOFIT_PYTHON=OFF -DGOOFIT_PHYSICS=OFF
cmake --build build-hip -j32
```
185/185 targets built and linked clean, first try, no edits.

### Tests: 24/24 PASS on GPU 3 (gfx1100), 9.62 s total
`ctest --test-dir build-hip --output-on-failure`. Repeated 3x, 24/24 every
time. NormalizeTest, SimpleTest, BinningTest, BlindTest, MonteCarloTest,
GenArgusTest, GenGaussianTest, the 15 tests/convert PDF tests (including
ConvolutionTest, 3.43 s), exponential_Example, exponential2_Example.

### The gfx90a "garbage normalization" blocker does NOT reproduce on gfx1100
Attempt 2 recorded that unbinned NLL fits diverge to a parameter bound because
the device reads 6.25e-310 from the hipMemcpyToSymbol-published
`d_normalizations`. On gfx1100 with the same committed tree the unbinned
maximum-likelihood fit converges correctly:

    examples/exponential/exponential   ->  alpha = -1.001102381 +/- 0.003165763921

against a generated alpha of -1, which is inside the [-1.01, -0.99] window the
example itself asserts (it returns 1 otherwise). Run 10 times: bit-identical
every time, 35 function calls, Edm 2.7e-07. So there is no race visible here.
The remaining hypotheses for gfx90a, in the order worth testing there: (a) it
was fixed between the ROCm the gfx90a session used (7.2.1) and 7.2.5; (b) it is
genuinely gfx90a-specific; (c) the gfx90a repro was built from the uncommitted
tree rather than d95236e57. Whoever next has a gfx90a should re-run the
committed tree BEFORE re-debugging SmartVector -- the note in attempt 2 may be
describing a state that no longer exists.

### BLOCKER (not technical): no push access to the fork
`git push` to AMD-Ecosystem/GooFit is refused:

    remote: Permission to AMD-Ecosystem/GooFit.git denied to jeffdaily.
    fatal: ... The requested URL returned error: 403

`gh api repos/AMD-Ecosystem/GooFit --jq .permissions` reports
`{"admin":false,"maintain":false,"pull":true,"push":false,"triage":false}`,
while sibling forks (visionaray, cuBQL) report `push:true,triage:true`. So this
one repository is missing the collaborator/team grant the others have; it is an
org access-configuration fix, not a porting problem. The gfx90a session pushed
d95236e57 to it, so the grant existed and was lost.

Pending work saved as `projects/GooFit/rocm-build-docs.patch` (one commit,
df9aaa298 locally): documents the ROCm build in README.md (requirements
collapsible + backend-selection section) and docs/SYSTEM_INSTALL.md (an Ubuntu
ROCm build recipe next to the existing per-OS recipes). Apply with
`git am` on moat-port once push access is restored, then `advance-head`.

### The CPP/OMP no-regression build cannot be configured on this host
Unrelated upstream submodule rot: `.gitmodules` gives `extern/thrust` the
relative url `../../thrust/thrust.git`, which now redirects to NVIDIA/cccl, and
cccl does not contain the recorded commit 8551c9787. `git submodule update`
fails with "not our ref", leaving extern/thrust at CCCL v3.3.3, whose layout
FindThrust.cmake cannot parse ("CMake Error at
extern/cmake_utils/FindThrust.cmake:44 (file)"). This affects upstream GooFit
identically and is not caused by the port; the HIP build is unaffected because
it skips extern/thrust and uses rocThrust. No nvcc on this host, so the CUDA
no-regression compile was not run either.

### Reviewed, not changed
- `src/PDFs/utilities/DebugTools.cu` was not deleted by the port, it moved from
  PDFCore to `src/PDFs/physics/CMakeLists.txt`; it is only referenced by the
  physics PDFs, so the CUDA build is unaffected.
- Every CMakeLists.txt change is inside `GOOFIT_DEVICE STREQUAL HIP` guards or
  is the additive `HIP` entry in `DEVICE_LISTING`; the CUDA path is untouched.
- No `warpSize`, `__shfl*`, `__ballot` or `__activemask` anywhere in src/ or
  include/, so nothing is wavefront-width dependent. The one `__shared__` use
  (ConvolutionPdf.cu) is guarded so that every thread that enters the functor
  reaches both `THREAD_SYNCH` points. Threads that thrust masks off never enter
  the functor at all, which is the usual latent wave64 barrier-divergence risk;
  ConvolutionTest passes here and reportedly built on gfx90a, but it is the
  first place to look if gfx90a hangs.

### MOAT tooling gap found while starting
`moatlib.set_state(name, arch, "porting")` short-circuits at
`if new_state == cur and not revalidated: return obj` when the PROJECT stage is
already `porting` (left there by an earlier host), so it prints the transition,
writes nothing, and never reaches the `obj["porting"] = {...}` acquisition ten
lines below. The lock had to be taken with `moatlib.py port-lock GooFit --take
linux-gfx1100`. The acquisition needs to happen before that short-circuit, the
same way the exclusivity refusal already does.

## Resuming (2026-08-07)

The port continues: this was judged a port worth finishing rather than an unportable
codebase, so linux-gfx90a is no longer marked blocked. The blocker as last recorded, which
is where to pick it up:

Build/link blocker RESOLVED (cross-TU device visibility fixed via -fgpu-rdc + OBJECT-libs-to-single-shared-lib device link; core+basic+combine PDFs and 24 test/example exes build and link on gfx90a). NEW blocker: unbinned NLL fits diverge to a parameter bound because the device reads garbage normalization values (6.25e-310 uninitialized-double pattern) from the hipMemcpyToSymbol-published __device__ d_normalizations during the per-event reduction; analytic normalize() and per-event device eval are correct in isolation. See notes.md attempt 2 for the minimal repro and next steps. Physics PDFs scoped out (GOOFIT_PHYSICS=OFF; MCBooster+device Eigen complex inverse deferred, WIP patch saved).
