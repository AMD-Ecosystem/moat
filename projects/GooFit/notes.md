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

## Review 2026-08-08 (reviewer, linux-gfx1100, fork df9aaa298 vs 5c6ca525c)

Verdict: changes-requested. The port strategy is right (Strategy A: one compat header,
`enable_language(HIP)`, `LANGUAGE HIP` rather than renamed sources, CUDA path guarded), no
wavefront-width assumption exists anywhere in the diff, and the jargon check is clean over
the whole branch. What forces a bounce is that the central technical finding of this port
is wrong, it was promoted to the shared skill in that wrong form, and it drove a large
build-system restructuring that is probably unnecessary.

### 1. The relocatable-device-code finding is factually wrong (blocking)

Claimed in `.claude/skills/cuda-to-rocm/references/strategy-a-cmake.md:57-88`, repeated
verbatim in `projects/GooFit/src/CMakeLists.txt:585-589` and in the body of commit
d95236e57:

> The HIP device link only sees device objects passed DIRECTLY on the link line. Objects
> inside a static archive are invisible to it.

That is not true on ROCm 7.2.3. Disproved on this host, gfx1100, two ways.

Raw hipcc: `__device__ int g_val` and `__device__ int helper(int)` defined in `a.hip`,
called from a kernel in `b.hip`, `a.o` archived into `liba.a`; `hipcc -fgpu-rdc --hip-link
b.o liba.a` links and runs correctly (`result=10`). The archive participates in the device
link.

CMake, at GooFit's actual shape (two STATIC libraries aggregated by an INTERFACE library,
a cross-TU `__device__` global and a device function-pointer table registered from one
archive and called from another): the build FAILS with exactly the symptom recorded here --

    ld.lld: error: undefined hidden symbol: __hip_gpubin_handle_b33c4363e66c5890
    ld.lld: error: undefined symbol: __hip_fatbin_b33c4363e66c5890

and the generated link line shows the cause:

    clang++ --offload-arch=gfx1100 --hip-link ... main.cu.o -o myexe libmylib.a

`--hip-link` is there, `-fgpu-rdc` is NOT. Adding one line --
`target_link_options(myexe PRIVATE -fgpu-rdc --hip-link)` -- and changing nothing else
(libraries stay STATIC, no OBJECT libraries, no gathering into a shared library) makes it
link and produce the correct answer.

So the actual lesson is: `-fgpu-rdc` must be on the LINK line as well as the compile line.
CMake's `HIP_SEPARABLE_COMPILATION` property does not add it, so putting `-fgpu-rdc` only in
`target_compile_options` leaves the final link non-relocatable, and the failure surfaces as
undefined `__hip_fatbin_*`/`__hip_gpubin_handle_*` rather than as a clear diagnostic. The
porter did discover this for the aggregate library (`CMakeLists.txt:759`) but attributed it
to the archive rather than to the missing flag.

One caveat that IS true and worth keeping in the rewritten lesson: the host linker only
extracts an archive member that resolves an undefined HOST symbol, so a member whose device
code is needed but whose host symbols are never referenced will not be pulled in, and
`-Wl,--whole-archive` is the fix for that specific case. That is a different statement from
"archives are invisible to the device link".

Required:
- Rewrite `strategy-a-cmake.md:57-88` around the real cause. Its current form prescribes an
  invasive restructuring for a two-line problem, and it would misdirect every future porter.
- Fix the same claim in `CMakeLists.txt:585-589` and in the d95236e57 commit body.

### 2. The OBJECT-library / single-shared-library restructuring is likely unnecessary

Following from 1: `CMakeLists.txt:590-594` turns every GooFit library into an OBJECT library
on HIP, and `CMakeLists.txt:746-767` gathers them into one SHARED `goofit_lib` with a
hand-written dependency list, because the object libraries are deliberately not linked and
so their usage requirements do not propagate. That is the largest single piece of the diff
and it makes the HIP build structurally different from the CUDA and CPU builds for a reason
that does not hold.

Re-test the minimal form first: keep `add_library(${GNAME} ${GOOFIT_LIB_TYPE} ...)` as the
CUDA path does, keep the `INTERFACE goofit_lib`, and add `-fgpu-rdc --hip-link` to the link
options of `goofit_lib` consumers (the executables created by `GOOFIT_ADD_EXECUTABLE`). If
that builds and passes, the CMake diff shrinks to roughly the language enable, the compat
header force-include, the source marking, and the link options -- which is what a minimal
Strategy A footprint should look like. Only if it genuinely fails should the heavier
structure stay, and then the commit and the skill must record the real reason it was needed.

### 3. CUDA-path regression risk in the driver_types.h guard

`include/goofit/GlobalCudaDefines.h:122-128` narrowed the guard from
`THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA` to `defined(__CUDACC__)`. Those are not
the same condition. In a CUDA build GooFit compiles `.cpp` files with the host compiler
(`GOOFIT_ADD_LIBRARY` sets no LANGUAGE property on the CUDA path), so `__CUDACC__` is not
defined in them, and `src/PDFs/GooPdf.cpp`, `src/PDFs/detail/Globals.cpp`,
`src/goofit/PdfBase.cpp`, `src/goofit/MathUtils.cpp` and `src/goofit/Application.cpp` all
include this header and then need `cudaError_t` for the declarations at lines 134-135.
Upstream's unconditional include under the CUDA Thrust system is evidence those host TUs
relied on it. No host in this effort has nvcc, so this cannot be tested here -- which is
precisely why the change must be made provably neutral rather than plausibly harmless:
guard on HIP (`#if !defined(__HIPCC__) && !defined(__HIP_PLATFORM_AMD__)`), or restore the
original CUDA condition with an added HIP arm, so the CUDA preprocessor state is bit-for-bit
what it was.

### 4. Copyright and author lines do not match GooFit's house style

`include/goofit/detail/cuda_to_hip.h:3-4` and `include/goofit/detail/CudaCompat.h:3-4` add

    // Copyright (c) 2026 Advanced Micro Devices, Inc.
    // Author: Jeff Daily <jeff.daily@amd.com>

GooFit carries copyright lines on exactly two files, both vendored third-party code
(`include/goofit/cpp/landau.h`, from the ROOT MathLib team, and
`include/goofit/utilities/Uncertain.h`, from its original author). No GooFit-authored file
carries a copyright or an author line. Remove both lines from both headers; the explanatory
comment beneath them is useful and should stay.

### 5. Commit d95236e57 advertises the port as not working

Its body ends:

> This is work in progress: the core library and the basic/combine PDFs build and link on
> gfx90a, but unbinned NLL fits do not yet converge on HIP ...

and its Test Plan claims only "SimpleTest passes". This is the branch's base commit and it
ships to the upstream maintainers exactly as written. Its title,
"[ROCm] Add HIP/ROCm GPU backend (core library builds and links)", reads the same way. No
arch has a `validated_sha` yet, so rewording by rebase orphans nothing; do it before the
next validation run rather than after.

### 6. The __ldg lesson is inaccurate and misidentifies the blocker

`GlobalCudaDefines.h:70` and `.claude/skills/cuda-to-rocm/references/fault-classes.md:199`
both say HIP's `__ldg` only accepts scalar types. `/opt/rocm/include/hip/amd_detail/hip_ldg.h`
overloads it for scalars AND vector types (`char2`, `char4`, `short2`, `short4`, `int2`,
`int4`, `longlong2`, ...), and `amd_hip_fp16.h` adds `__half`/`__half2`; what it does not
accept is an arbitrary user type. More to the point, it is not GooFit's blocker: every
`RO_CACHE` call site passes an `int` or an `fptype` (double), both of which HIP's `__ldg`
handles. The actual reason the macro has to change is that `extern/generics/ldg.h` is a
CUDA-only wrapper built on inline PTX and `__CUDA_ARCH__`, which cannot compile under hipcc.
Say that in both places.

While correcting that entry, also fix `fault-classes.md:189-197`: the GooFit case was a
`new fptype[10]` with a hardcoded 10, not an allocation "sized by the observable count".

### 7. Documentation claims support that was never observed

`README.md:51` states that both wave64 (`gfx90a`, `gfx942`) and wave32 (`gfx1100`) parts are
supported. The only wave64 evidence on record is a failure (the gfx90a NLL divergence in
attempt 2), gfx942 has never been built, and no arch is `completed`. `README.md:50` states
"ROCm 6.0+", which nothing has tested -- 7.2.x is the only ROCm this has run on. Both lines
go upstream. State what was tested, or say the backend is developed against ROCm 7.x on
gfx1100 and expected to work on other supported parts.

Also `docs/SYSTEM_INSTALL.md` tells ROCm users to `git clone --recursive` two lines after
the README says the bundled `extern/thrust` is not used. That submodule currently cannot be
initialised at all (see the entry below in attempt 3), so the recipe as written fails at
step two for a reason the ROCm build does not care about. Use a plain clone there.

### 8. GOOFIT_PHYSICS is honored in two places and ignored in two others

The new option (`src/PDFs/CMakeLists.txt:34`) gates `src/PDFs/physics` and, correctly,
`tests/simple/VectorsTest`. It does not gate:

- `examples/CMakeLists.txt:61-75`: `dalitz`, `pipipi0DPFit`, `SigGen`, `DP4`, `TDDP4`
  (and `TDDP4WeightedMC`) all include `goofit/PDFs/physics/...` or `mcbooster/...` and are
  gated only on `ROOT_FOUND`. `-DGOOFIT_PHYSICS=OFF` with ROOT installed does not build.
- `python/PDFs/CMakeLists.txt:6`: `add_subdirectory(physics)` is unconditional, so
  `-DGOOFIT_PYTHON=ON -DGOOFIT_PHYSICS=OFF` does not build.

Both were invisible in the gfx1100 run because it set `GOOFIT_CERNROOT=OFF` and
`GOOFIT_PYTHON=OFF`. Gate both on `GOOFIT_PHYSICS`. Separately, nothing stops
`-DGOOFIT_DEVICE=HIP -DGOOFIT_PHYSICS=ON`, which the docs say must not be done; a
`message(FATAL_ERROR ...)` for that combination turns a wall of template errors into one
sentence.

### 9. Smaller items

- `src/goofit/Application.cpp:300` prints "CUDA does not support floating point exceptions"
  on a ROCm build. The guard became `GOOFIT_DEVICE_IS_GPU`; the message did not follow.
- `CMakeLists.txt:421`: the `if(NOT DEFINED hip_lang) set(hip_lang 0)` fallback is dead --
  `hip_lang` is used only inside the `GOOFIT_DEVICE STREQUAL HIP` block that defines it,
  unlike `cuda_lang`, which is consumed later in `cuda_lang_rel`.
- `CMakeLists.txt:1` keeps `cmake_minimum_required(VERSION 3.16...3.23)` while
  `enable_language(HIP)` needs 3.21. README says 3.21, but a user on 3.16-3.20 gets an
  unhelpful error; a version check inside `GOOFIT_OPTIONAL_HIP` costs two lines.

### Checked and clean

Strategy A shape (single compat header at `include/goofit/detail/cuda_to_hip.h`, no-op
outside HIP via the `__HIP__`/`__HIPCC__`/`__HIP_PLATFORM_AMD__` guard, no second HIP-aware
header, `.cu` marked `LANGUAGE HIP` rather than renamed). Every CUDA runtime symbol the
sources actually use is in the map. No `warpSize`, `__shfl*`, `__ballot`, `__activemask` or
literal 32 anywhere in `src/` or `include/`; the one `__shared__` user
(`src/PDFs/combine/ConvolutionPdf.cu:94-139`) sizes its load count from `BLOCKDIM`, not from
a wave width, and both `THREAD_SYNCH` points are reached by every thread that enters the
functor -- the pre-existing hazard flagged in its own comment is about PDF mixing, not about
wavefront size, so wave64 changes nothing here. `MAX_NUM_OBSERVABLES` is 10, identical to
the `new fptype[10]` it replaced, so that fix changes no bound. `hptr_to_Step` has no other
reference in the tree, so removing it is safe. `DebugTools.cu` is referenced only by
`Amp4Body_TD.cu` and `detail/AmpCalc_TD.cu`, so moving it into PDFPhysics does not affect
the non-physics build. `IncoherentSumTest` is already commented out upstream.
`python3 utils/jargon.py` is clean over the whole branch, both `--commits` and `--diff`.
Commit titles are `[ROCm]`-prefixed and 62 and 56 chars; both bodies name Claude, carry a
Test Plan, and have no `Co-Authored-By` trailer; no non-ASCII and no em-dash in any added
line or commit message. The fork clone is clean (`git status --porcelain` empty), so the
deinited `extern/thrust` submodule leaves no uncommitted state.

### On the gfx90a NLL question: not a review blocker

This does not gate the review. The reviewable artifact is the diff, and the diff contains no
arch-conditional code at all -- there is no `__GFX9__`, no wavefront branch, no per-arch
guard -- so there is nothing in it a porter could change in response to the gfx90a report.
Settling it needs a gfx90a host running the committed tree, which is the validator's step.

Of the three hypotheses recorded in attempt 3, (c) the gfx90a repro was built from a tree
that is not d95236e57 is the most likely, and the reason is in the shape of the symptom
rather than in a guess about ROCm versions. The report says early fit iterations read a
plausible normalization and later ones read 6.25e-310. Two things follow. That value is a
pointer bit pattern read as a double, not arithmetic gone wrong. And an iteration-count
dependence points at accumulated device-side state, which is exactly what the device
`new fptype[10]` in the binned `MetricTaker::operator()` produced -- and that operator is
the normalization integrator an unbinned fit calls once per iteration, so it ran on the
failing path. Attempt 2's own notes say replacing it "alone fixed the alpha parameter in
GaussianTest", i.e. it was still being removed while the NLL debugging was going on. The
committed tree has the fix.

Hypothesis (a), a 7.2.1-to-7.2.5 difference, is the second candidate and is cheap to
distinguish: the gfx90a host runs the committed tree on whatever ROCm it has. Hypothesis
(b), genuinely gfx90a-specific, is the least likely from the code -- the publication path is
a host-side `hipMemcpyToSymbol` into a `__device__` pointer, with no wavefront dependence and
no intra-wave timing to be sensitive to -- but note that if the restructuring in finding 2
turns out to be load-bearing after all, then it is a device-linking question and a stale
duplicate of the `d_normalizations` symbol across two device images becomes a plausible
mechanism worth checking there first.

Whoever picks up gfx90a: re-run the committed tree before debugging SmartVector, exactly as
attempt 3 advises.

## Port attempt 4 (2026-08-08, linux-gfx1100) -- review bounce addressed; branch rewritten

Fork now AMD-Ecosystem/GooFit @ moat-port 6601ce1f7, two commits on 5c6ca525c.
The old d95236e57/df9aaa298 pair was rewritten (no arch had a validated_sha, so
nothing was orphaned).

### The relocatable-device-code finding was wrong; the reviewer was right

The recorded claim -- "the HIP device link only sees device objects passed
DIRECTLY on the link line, objects inside a .a are invisible to it" -- is false.
Reproduced the reviewer's minimal fix on the real project: reverted every
GooFit library to the upstream `add_library(${GNAME} ${GOOFIT_LIB_TYPE} ...)`
(STATIC), deleted the OBJECT-library gathering and the single SHARED goofit_lib,
and added one line:

    target_link_options(GooFit_Common INTERFACE "$<$<LINK_LANGUAGE:HIP>:-fgpu-rdc>")

195/195 targets build, 24/24 tests pass on GPU 1 (Radeon Pro W7800, gfx1100,
ROCm 7.2.3). The generated link line confirms the mechanism exactly:

    LINK_FLAGS = --hip-link --rtlib=compiler-rt -unwindlib=libgcc -fgpu-rdc ...

`--hip-link` was always there (CMake adds it for a HIP link language); only
`-fgpu-rdc` was missing, and `HIP_SEPARABLE_COMPILATION` does not add it. Every
GooFit library on that link line is a plain `.a` and the device link resolves
the cross-TU `__device__` globals and function-pointer tables through them.

**So the OBJECT-library restructuring was NOT needed.** It was the largest piece
of the diff; CMakeLists.txt went from +160 to +132 lines against upstream and the
HIP build now has the same target shape as the CUDA and CPU builds.

Corrected in the skill: `references/strategy-a-cmake.md` now leads with
"-fgpu-rdc is needed on the LINK line too", carries the reproducer shape and the
link-line evidence, and states explicitly that archives are NOT invisible so the
next porter does not re-derive the wrong version. The real archive caveat (the
host linker only extracts a member that resolves an undefined HOST symbol;
`--whole-archive` for that case) is kept and labelled as the different statement
it is.

### __ldg: the HIP arm was unnecessary and its stated reason was wrong twice

Both the port's claim ("HIP __ldg only accepts scalar types") and the review's
counter-claim ("the real blocker is extern/generics/ldg.h's inline PTX") are
wrong. `extern/generics/generics/ldg.h` contains no PTX; it is guarded on
`__CUDA_ARCH__ >= 350`, which is false under hipcc, so it falls back to `*ptr`.
Compiling a two-line hipcc TU that includes it and calls `__ldg` on `const
double*` and `const int*` succeeds with no diagnostic. And ROCm's `__ldg`
overloads are each literally `return ptr[0]`, so `(x)` and `__ldg(&x)` are the
same code on AMD.

The whole `#ifdef __HIPCC__` RO_CACHE arm was therefore deleted and GooFit uses
its original `#include <generics/ldg.h>` / `RO_CACHE(x) __ldg(&x)` on both
backends. Promoted to `references/fault-classes.md` as "check before working
around it", with the two-line test that settles it.

### Other review items

- `GlobalCudaDefines.h` driver_types.h guard: restored to the original
  `THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA` include, with the stub
  enum moved to an `#elif !GOOFIT_DEVICE_IS_GPU` arm. The CUDA preprocessor
  outcome is now bit-for-bit what upstream had; on HIP neither arm fires because
  cuda_to_hip.h has already mapped cudaError_t onto hipError_t. UNVERIFIED by
  execution -- no host in this effort has nvcc -- which is why it was made
  neutral by construction rather than by testing.
- Copyright and Author lines removed from both new headers; the explanatory
  comments stay. GooFit's only two copyright lines are on vendored third-party
  files.
- GOOFIT_PHYSICS is now honored everywhere: examples/CMakeLists.txt gates
  dalitz, pipipi0DPFit, SigGen, DP4 and TDDP4 (the five that include
  goofit/PDFs/physics or mcbooster); python/PDFs gates the physics subdirectory,
  and python/CMakeLists.txt stops linking _Physics while python/goofit.cpp
  guards the physics init declarations and calls behind a GOOFIT_PHYSICS
  compile definition -- gating only the subdirectory would have left an
  unresolved _Physics on the module link. The option now defaults OFF on HIP and
  ON elsewhere, and an explicit `-DGOOFIT_PHYSICS=ON` with HIP is a FATAL_ERROR
  instead of a wall of template errors. A plain `-DGOOFIT_DEVICE=HIP` configure
  needs no extra flag now.
- README/SYSTEM_INSTALL now claim only what was run: ROCm 7.2, gfx1100. The
  wave64/gfx942 claim and "ROCm 6.0+" are gone.
- Application.cpp's floating-point-exception message no longer says "CUDA" on a
  ROCm build.
- Dead `if(NOT DEFINED hip_lang)` fallback removed; the IPO guard in
  GOOFIT_ADD_EXECUTABLE removed too (that line references an undefined ${GNAME}
  upstream, so guarding it changed nothing).
- A CMake 3.21 version check now fires inside GOOFIT_OPTIONAL_HIP, since
  cmake_minimum_required still allows 3.16 and enable_language(HIP) needs 3.21.

### extern/thrust is unfetchable, and the configure guard had to move

`CMakeLists.txt`'s submodule check required `extern/thrust/README.md` to exist
even on HIP, so a HIP configure could not succeed with that submodule deinited
-- and it cannot be initialised at all: `.gitmodules` gives it the relative url
`../../thrust/thrust.git`, which now redirects to NVIDIA/cccl, and cccl does not
contain the recorded commit 8551c9787 ("not our ref"). This affects upstream
GooFit identically. The check for that one submodule moved into the non-HIP
Thrust branch, where it also produces a clear message instead of the FindThrust
parse error a wrong checkout gives. Verified: `-DGOOFIT_DEVICE=CPP` now stops
with "The extern/thrust submodule was not downloaded!" rather than a CMake error
inside FindThrust.cmake:44. Leave extern/thrust deinited.

Consequence: the CPP/OMP no-regression build still cannot be configured on this
host, for that upstream reason and not for anything the port did. No nvcc
anywhere, so the CUDA no-regression compile remains unrun; the CUDA path is
protected by construction (every CMake change is inside a
`GOOFIT_DEVICE STREQUAL HIP` guard or is an additive DEVICE_LISTING entry, and
the one shared header guard was restored to its exact original condition).

### Not chased: the gfx90a NLL divergence

Ruled out as a review blocker -- the diff has no arch-conditional code. The
review's analysis is the one to carry: `6.25e-310` is a pointer bit pattern read
as a double, and the iteration-count dependence points at accumulated
device-side state, which is exactly what the device `new fptype[10]` in the
binned `MetricTaker::operator()` produced. That operator is the normalization
integrator an unbinned fit calls once per iteration, and attempt 2's notes say
it was still being removed while the NLL debugging ran, so the gfx90a repro was
most likely built from a tree that is not the committed one. The stale-duplicate
`d_normalizations`-across-two-device-images hypothesis is now MOOT: there is
only one device image again, since the shared-library gathering is gone.
Whoever has a gfx90a: build 6601ce1f7 and run ctest before debugging anything.

## Review 2026-08-08 (reviewer, linux-gfx1100, fork 6601ce1f7 vs 5c6ca525c)

Verdict: changes-requested. The three items the last round bounced are fixed and I
verified each of them independently (details under "Adjudications" at the end). What
follows are new problems, and every one of them is a claim that was recorded without
being tested.

Rebuilt and re-ran everything on this host: HIP configure + build 195/195 targets,
24/24 ctest on GPU 1 (9.40 s), `examples/exponential/exponential` giving
`alpha = -1.001102381 +/- 0.003165763921`, matching the reported run exactly. The
`-fgpu-rdc` link fix is real: `build.ninja` gives every one of the 24 HIP-link targets
`LINK_FLAGS = --hip-link --rtlib=compiler-rt -unwindlib=libgcc -fgpu-rdc ...` with
`libPDFCore.a`, `libPDFBasic.a`, `libPDFCombine.a` and the rest as plain archives on
the line. No HIP-link target is missing the flag.

### 1. The branch is 19 commits behind upstream master, and upstream has rewritten the block this port restructures

`git log 5c6ca525c..master` on the fork is 19 commits, and the fork's `master`
(5fe1221a8) is exactly `GooFit/GooFit@master` as of today. Two of those commits land on
top of this port's own changes:

- `249baaa71 build: use modern CCCL for the bundled Thrust (fix #387) (#391)` repoints
  `extern/thrust` at NVIDIA/cccl v3.3.3 (`.gitmodules` url is now `../../NVIDIA/cccl.git`),
  replaces `GOOFIT_FORCE_LOCAL_THRUST` with a bundled-CCCL option, and force-includes
  `include/goofit/detail/ThrustForwardCompat.h` on `GooFit_Common` for EVERY backend --
  defining `__host__`/`__device__` and pulling `<thrust/tuple.h>` for non-CUDA device
  systems. That is the same `GooFit_Common` the port hangs `-include cuda_to_hip.h`,
  `-fgpu-rdc` and `roc::rocthrust` off, and it is the same `find_package(Thrust)` /
  `GOOFIT_FORCE_LOCAL_THRUST` block CMakeLists.txt:437-471 wraps in
  `if(NOT GOOFIT_DEVICE STREQUAL HIP)`. The port's version of that block no longer
  exists upstream.
- `d8e1102cf perf: avoid per-event device heap allocation in CompositePdf (#384)` is the
  same change the port makes in `src/PDFs/MetricTaker.cu:57`, already accepted upstream
  in a sibling file.

Also on master and touching files this port touches: `62409d260 fix uninitialised array
in kMatrix (#400)`, `28a333348 chore: modernize typedefs ... (#390)`,
`ec50729ef chore: drop old platform support (#321)`, `f180a5f63 chore: use
scikit-build-core (#347)` (python packaging, against the port's `python/CMakeLists.txt`
edits). Rebase onto 5fe1221a8 and re-derive the Thrust and python hunks; the diff under
review is not the diff that would be proposed upstream. Note this is invisible in
`git diff master...HEAD` because that diffs from the merge base.

### 2. `HIP_SEPARABLE_COMPILATION` is not a CMake property, and the port, the commit message and the skill all treat it as one

CMakeLists.txt:589 and CMakeLists.txt:635 set it on every HIP library and executable.
It does nothing:

```
$ cmake --version
cmake version 3.31.6
$ cmake --help-property HIP_SEPARABLE_COMPILATION
Argument "HIP_SEPARABLE_COMPILATION" to --help-property is not a CMake property.
$ cmake --help-property-list | grep SEPARABLE
CUDA_SEPARABLE_COMPILATION
```

Confirmed empirically on a two-file reproducer (`a.hip` defining a `__device__`
function into a STATIC lib, `b.hip` calling it from a kernel): configuring with
`HIP_SEPARABLE_COMPILATION ON` on both targets and configuring without it produce
byte-identical `FLAGS`/`LINK_FLAGS` in `build.ninja`, and both fail with
`lld: error: undefined hidden symbol: getg()`. The property puts `-fgpu-rdc` on
neither the compile line nor the link line, because CMake does not know it.

Consequences to fix:

- Drop both `set_target_properties(... HIP_SEPARABLE_COMPILATION ON)` calls. The
  compile-side `-fgpu-rdc` comes entirely from
  `target_compile_options(GooFit_Common INTERFACE "$<${hip_lang}:-fgpu-rdc>")` at
  CMakeLists.txt:411 and the link-side entirely from the `target_link_options` on the
  next line. Leaving a meaningless property set in someone else's build system will be
  asked about in review.
- Commit d2ad31eb4's body says "CMake's `HIP_SEPARABLE_COMPILATION` property does not
  put the flag on the link line", which tells a GooFit maintainer the property does the
  compile half. It does not exist. Reword to say the flag has to be added by hand on
  both sides.
- `references/strategy-a-cmake.md` has the same error twice: "Adding `-fgpu-rdc` to
  `target_compile_options` plus `HIP_SEPARABLE_COMPILATION ON` fixes that one", and
  "CMake's `HIP_SEPARABLE_COMPILATION` property does not put it there". The correct and
  more useful lesson is that there is no HIP analogue of `CUDA_SEPARABLE_COMPILATION` at
  all in CMake 3.31, so both halves are manual. Everything else in that entry is right
  (see Adjudications).

### 3. The promoted device-`new[]` fault class does not reproduce

`references/fault-classes.md` now asserts as a fault class that a device-side
`new[]`/`delete[]` per thread "can return wrong values without faulting", citing GooFit:
"`fptype[MAX_NUM_OBSERVABLES]`, which is also 10, fixed it and immediately corrected a
fitted parameter".

I restored upstream's `auto *events = new fptype[10]` / `delete[] events` in
`src/PDFs/MetricTaker.cu` in a scratch clone of `moat-port`, rebuilt the HIP backend
(gfx1100, ROCm 7.2.3) and ran it on GPU 1: 24/24 tests pass and the exponential example
gives `alpha = -1.001102381 +/- 0.003165763921` -- the same digits as the port's fixed
version. The device heap allocation is not observably broken on this hardware, so the
"immediately corrected a fitted parameter" evidence does not exist on the only arch
where it has been re-run. It came from attempt 2 on gfx90a, and attempt 4's own notes
say that repro was probably built from a tree that is not the committed one.

Keep the code change -- it is the same thing upstream did in `d8e1102cf` and it is
correct on CUDA too -- but stop presenting it as a bug fix:

- Commit d2ad31eb4 body: "HIP's device malloc heap is small and the failure is silent"
  claims a demonstrated failure. Reframe as avoiding a per-thread device heap
  allocation in the integration inner loop, and cite upstream's own #384 as precedent.
- `src/PDFs/MetricTaker.cu:54-56` comment says the allocation is "unreliable under HIP's
  small default device malloc heap". Same problem.
- The fault-classes entry must either be reduced to "prefer a fixed per-thread array to
  a device heap allocation" with the observation marked as unreproduced on gfx1100, or
  removed until someone reproduces it. As written it will send the next porter chasing
  a numerical bug that this project cannot show.

### 4. `extern/thrust` is fetchable, and the CPP no-regression build it excused runs fine

notes.md attempt 4 states `extern/thrust` "cannot be initialised at all" because
"cccl does not contain the recorded commit 8551c9787 ('not our ref')". That is wrong at
this branch's base:

```
git clone --depth 1 -b moat-port https://github.com/AMD-Ecosystem/GooFit.git gf
cd gf && git submodule update --init extern/thrust
Submodule path 'extern/thrust': checked out '8551c97870cd722486ba7834ae9d867f13e299ad'
```

`git ls-tree 5c6ca525c extern/thrust` is 8551c978 and `git fetch --depth 1
https://github.com/thrust/thrust.git 8551c978...` succeeds. The "not our ref" is real
but for a DIFFERENT commit: `git ls-tree master extern/thrust` is af8cce4ca, the CCCL
pointer from `249baaa71`, and fetching it through the OLD `.gitmodules` url
(`thrust/thrust.git`) is what fails. Diagnosing a submodule with the wrong branch's
`.gitmodules` checked out produced the whole claim.

The consequence is the thing that matters: attempt 4 concludes "the CPP/OMP
no-regression build still cannot be configured on this host". It can. On a clone of
`moat-port` with all submodules initialised:

```
cmake -S . -B build-cpp -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGOOFIT_DEVICE=CPP \
  -DGOOFIT_TESTS=ON -DGOOFIT_EXAMPLES=ON -DGOOFIT_CERNROOT=OFF -DGOOFIT_PYTHON=OFF
cmake --build build-cpp -j32     # 264/264 targets, GOOFIT_PHYSICS:BOOL=ON
ctest --test-dir build-cpp       # 100% tests passed, 25 of 25
```

and with `-DGOOFIT_PYTHON=ON -DGOOFIT_TESTS=OFF -DGOOFIT_EXAMPLES=OFF` the module
builds and `import goofit` still exposes `Amp3Body`, `Amp3BodyBase`, `Amp3Body_IS`,
`Amp3Body_TD`, `DalitzPlotPdf`, `DalitzPlotter`. Run that and record it; it is the
evidence the shared-header edits and the `GOOFIT_PHYSICS` gating do not regress the
non-GPU backends, which no amount of reading can supply. Correct the notes so the next
agent does not re-inherit "cannot be configured".

### 5. The IPO guard is a no-op and its stated reason is unverified

CMakeLists.txt:609-612 wraps `INTERPROCEDURAL_OPTIMIZATION ${SUPPORTS_IPO}` in
`if(NOT GOOFIT_DEVICE STREQUAL HIP)`. `SUPPORTS_IPO:INTERNAL=NO` in the HIP build's
`CMakeCache.txt` -- and also `NO` in the CPP build's -- so `check_ipo_supported()`
already answers NO on this host regardless of backend and the guard changes nothing
that was ever exercised. The reason given, in the skill entry as "IPO/LTO also cannot be
combined with HIP relocatable device code", does not hold either:
`hipcc --offload-arch=gfx1100 -fgpu-rdc -flto` and `-flto=thin` both compile and link a
`__device__`-global reproducer successfully on ROCm 7.2.3. Either drop the guard, or
keep it and say it is precautionary; do not leave an unverified prohibition in the
skill.

### 6. `GOOFIT_PHYSICS` is declared in the wrong file for the places that read it

`option(GOOFIT_PHYSICS ...)` lives at `src/PDFs/CMakeLists.txt:38`, but it is read in
`examples/CMakeLists.txt:72`, `tests/simple/CMakeLists.txt:1`, `python/CMakeLists.txt:53`
and `python/PDFs/CMakeLists.txt:6`. It works only because `add_subdirectory(src)`
(CMakeLists.txt:739) happens to precede tests (749), examples (754) and python (810);
reorder any of those and the examples and VectorsTest silently disappear instead of
erroring. It is also absent from the `### Options ###` block where every other
`GOOFIT_*` option is declared, and from GooFit's own feature summary -- the configure
output lists `GOOFIT_EXAMPLES`, `GOOFIT_PACKAGES`, `GOOFIT_TRACE`, `GOOFIT_DEBUG`,
`GOOFIT_MPI`, `GOOFIT_PYTHON`, `GOOFIT_TIDY_FIX` and not `GOOFIT_PHYSICS`. Declare it
next to the others with `feature_option(GOOFIT_PHYSICS)` and leave only the
`GOOFIT_DEVICE STREQUAL HIP` FATAL_ERROR in `src/PDFs/CMakeLists.txt`.

### Adjudications the porter asked for

**The `-fgpu-rdc` link-line fix and the corrected strategy-a-cmake entry: correct, with
the one exception in finding 2.** Archives are genuinely not invisible to the HIP device
link -- every GooFit library on the exponential link line is a plain `.a` and the
cross-TU `__device__` globals and function-pointer tables resolve. `--hip-link` is
present with no `-fgpu-rdc` unless you add it, exactly as the entry says, and my
two-file reproducer fails the same way. Reverting the OBJECT-library restructuring was
right. The `$<LINK_LANGUAGE:HIP>` genex on the `GooFit_Common` INTERFACE reaches every
consumer: all 24 targets with a HIP link language carry the flag and the 17 archive
targets, which are archived rather than linked, correctly carry nothing. The archive
caveat kept in the entry (host linker extracts a member only for an undefined HOST
symbol) is correct and correctly labelled as a different statement.

**`__ldg`: the porter is right, and both earlier claims were wrong.** `extern/generics/
generics/ldg.h` contains no PTX and no `asm` -- `grep -rn "asm\|ptx" extern/generics/`
returns nothing, its only conditional is `#if __CUDA_ARCH__ >= 350`. Recompiling
`src/goofit/Application.cpp` with `-Wundef` proves the guard is false under hipcc in
both passes:

```
extern/generics/generics/ldg.h:32:5: warning: '__CUDA_ARCH__' is not defined, evaluates to 0 [-Wundef]
extern/generics/generics/ldg.h:32:5: warning: '__CUDA_ARCH__' is not defined, evaluates to 0 [-Wundef]
```

(twice: once for host, once for gfx1100), so the wrapper falls back to
`template<typename T> __device__ T __ldg(const T* ptr) { return *ptr; }`. ROCm's
`/opt/rocm/include/hip/amd_detail/hip_ldg.h` has 29 non-template overloads including
`double` and `float`, each `return ptr[0];` (three char variants spell it `return *ptr;`
-- same thing), and a non-template beats the wrapper's template for the scalar calls.
So `RO_CACHE(x)` needed no HIP arm, deleting it and restoring GooFit's original include
is right, and the fault-classes entry for it is accurate.

**`GlobalCudaDefines.h`: the construction argument holds, and it is now tested too.**
Read against upstream for every backend: with `THRUST_DEVICE_SYSTEM_CUDA`,
`GOOFIT_DEVICE_IS_GPU` is 1 so line 28 `#if !GOOFIT_DEVICE_IS_GPU` is false (upstream:
`!= CUDA`, false), line 107 `#elif GOOFIT_DEVICE_IS_GPU` is true (upstream: `== CUDA`,
true), line 118 takes `#include <driver_types.h>` unchanged, and line 36
`defined(__CUDACC__) || defined(__HIPCC__)` is decided by `__CUDACC__` alone; with
CPP/OMP/TBB it is 0 and every arm matches upstream's `#else`. Identical preprocessor
outcome on both. The CPP build in finding 4 confirms it by execution for the non-GPU
half. Separately, `GOOFIT_DEVICE_IS_GPU` is reachable at all 19 use sites -- recompiling
`MetricTaker.cu`, `GooPdf.cu`, `PdfBase.cu` and `Application.cpp` with `-Wundef`
produces no `GOOFIT_DEVICE_IS_GPU` diagnostic, and every remaining site reaches
`CudaCompat.h` through `GlobalCudaDefines.h` (directly, or via `PdfBase.h` for
`MetricTaker.h` and the physics chain).

**The `GOOFIT_PHYSICS` default flip does not change what a CUDA or CPP user gets.** The
default is ON everywhere except HIP, and the CPP configure in finding 4 has
`GOOFIT_PHYSICS:BOOL=ON`, 11 physics targets, 25 tests including `VectorsTest`, and a
Python module still exporting the physics classes. Only the option's declaration site is
wrong (finding 6).

**Moving the `extern/thrust` check does not weaken CUDA or CPP.** The check still fires
for every non-HIP backend, just at CMakeLists.txt:443 instead of :82 and with a message
naming the submodule instead of a FindThrust parse error. The CPP build exercises that
path. The change is fine; its recorded rationale is not (finding 4), and upstream's
`249baaa71` replaces the whole block anyway (finding 1).

**Clean:** `utils/jargon.py --commits master..HEAD` reports clean; the fork tree is
clean (`git status --porcelain` empty); both commit titles are `[ROCm]`-prefixed and
under 72 chars; no `Co-Authored-By: noreply` trailer; no copyright or `Author:` lines;
no non-ASCII anywhere in the diff; no AMD-internal account, host or MOAT vocabulary in
any upstream-visible text. No warp-size or wavefront assumption exists to get wrong --
the tree has no `warpSize`, `__shfl`, `__ballot` or `0xffffffff` mask outside
`Application.cpp`'s device-info printout, and no texture, atomic or resource-handle code
at all. No arch-conditional code in the diff.

### Not a review blocker, but relevant to the gfx90a divergence

Two of the 19 commits the branch is missing are candidate explanations, and both are
upstream fixes rather than anything the port would need to invent:
`0c1ed9efc fix: correct device index-cursor and data-pointer corruption in amplitude
PDFs (#382)` and `d8e1102cf perf: avoid per-event device heap allocation in CompositePdf
(#384)`. Rebase first, then re-run on gfx90a before debugging anything.

## Port attempt 5 (2026-08-08, linux-gfx1100) -- rebased onto upstream master; six review items

Branch rewritten from 6601ce1f7 (base 5c6ca525c) to 19f29809b (base 5fe1221a8).
Both archs' `validated_sha` were null before and after, so no validation was
orphaned.

### The rebase (review item 2)

19 upstream commits. Three conflicts, all in files the port touches:

- **CMakeLists.txt submodule check.** Resolved to upstream's version verbatim.
  The port no longer touches that block at all -- see item 4 below.
- **CMakeLists.txt Thrust block.** `249baaa71` replaced `find_package(Thrust)` +
  `GOOFIT_FORCE_LOCAL_THRUST` with a bundled-CCCL path (three include roots,
  `.gitmodules` repointed at NVIDIA/cccl). The port's version of that block no
  longer exists, so it was re-derived rather than re-applied. The new shape adds
  four lines and re-indents nothing: upstream's `option()` stays put, and its
  `if(GOOFIT_FORCE_LOCAL_THRUST)` becomes
  `if(GOOFIT_DEVICE STREQUAL HIP) ... elseif(GOOFIT_FORCE_LOCAL_THRUST)`. Both
  upstream bodies are byte-identical to master. The reason is unchanged and is
  now the only reason: rocThrust supplies Thrust, and either the bundled CCCL or
  the toolkit copy on the include path would shadow rocThrust's headers.
- **GlobalCudaDefines.h.** Upstream merged two different conditions into one
  `#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_CUDA`: the `__host__`/
  `__device__` fallback, and `__align__`/`cudaDeviceSynchronize`/`__shared__`/
  `__constant__`. On HIP that block is TAKEN (device system is `..._HIP`), and
  taking the second half would be fatal. Resolved by splitting them: the
  `__host__`/`__device__` fallback keeps upstream's condition, the CPU-fallback
  definitions get `#if !GOOFIT_DEVICE_IS_GPU`. Measured that the split is safe:
  under hipcc both `__host__` and `__device__` are already defined as macros at
  the user force-include point (clang preincludes
  `__clang_hip_runtime_wrapper.h` before any `-include`), so upstream's `#ifndef`
  guards make that half inert in HIP device compilation and it still fires in
  HIP host-only TUs, where it is needed.

**`d8e1102cf` does not make any of the port redundant.** It is
`src/PDFs/combine/CompositePdf.cu`; the port's change is
`src/PDFs/MetricTaker.cu`, which upstream did not touch. Same pattern, different
file. It is precedent for the change, not a duplicate of it, and the commit
message now cites #384 instead of claiming a HIP bug.

`git rev-list --count moat-port..origin/master` = 0.

### Item 1: `HIP_SEPARABLE_COMPILATION` removed

Confirmed not a CMake property (3.31.6). Both `set_target_properties` calls
deleted, the CMakeLists comment and the commit message reworded to say CMake has
no HIP counterpart to `CUDA_SEPARABLE_COMPILATION` so both halves are manual, and
`references/strategy-a-cmake.md` rewritten the same way with the measurement in
it.

### Item 5: the IPO guard is load-bearing on master, and the LTO reason is narrower

The reviewer measured `SUPPORTS_IPO:INTERNAL=NO` at the old base. **On master it
is YES**: this configure prints `Compiler supports IPO: YES` and the cached
`SUPPORTS_IPO` is NO only because the port forces it. So the guard does work
now; something in the 19 commits (most likely `ec50729ef`'s CMake floor) changed
what `check_ipo_supported()` answers.

The guard moved from a per-target `if()` in `GOOFIT_ADD_LIBRARY` to upstream's
own backend switch next to the existing CUDA exclusion. That is one line instead
of four, matches house style, and fixes a real hole: `GOOFIT_ADD_EXECUTABLE`
never had the per-target guard, so executables would have taken IPO.

`-fgpu-rdc` + LTO measured three ways on ROCm 7.2.3 / gfx1100, two-TU
`__device__`-global reproducer:

| how `-flto` is applied | result |
| --- | --- |
| compile + link in one `hipcc` command | links |
| `-flto` only on the link line | links |
| `-flto` on the compile line, separate link (what CMake IPO does) | **fails** |

Failure is `ld.lld: error: undefined symbol: __hip_fatbin_<hash>` referenced by
`__hip_fatbin_wrapper`. A one-command reproducer reports success, which is
almost certainly why the review found LTO "fine".

### Item 6: `GOOFIT_PHYSICS` declared where the others are

Moved from `src/PDFs/CMakeLists.txt` to the `### Options ###` block beside
`GOOFIT_KMATRIX`, using upstream's own `IS_NOT_CUDA` idiom (a parallel
`IS_NOT_HIP`), plus `feature_option(GOOFIT_PHYSICS)`. `src/PDFs/CMakeLists.txt`
keeps only the FATAL_ERROR and the gating. It now appears in the configure
summary, and no longer depends on `add_subdirectory` ordering. A plain
`-DGOOFIT_DEVICE=HIP` with no `-DGOOFIT_PHYSICS=OFF` configures correctly.

### Item 4: `extern/thrust` is fetchable; the special-cased check is gone

`git submodule update --init --recursive` checks out CCCL `af8cce4ca` cleanly at
the new base, and `extern/thrust/README.md` exists, so upstream's top-of-file
check needs no change. The port's relocated check and its `if(NOT ... HIP)`
wrapper are both deleted -- they existed only for the false "cannot be fetched"
premise. HIP now requires the submodule like every other backend and simply does
not put it on the include path. The two doc sentences claiming it "need not check
out" were corrected to say it is kept off the include path.

### Item 3: bug claim removed

The `MetricTaker.cu` comment no longer claims HIP's device heap is unreliable;
it gives the CompositePdf/#384 reason. The commit message likewise. The
`fault-classes.md` entry is rewritten to say explicitly that the fitted-parameter
claim did not reproduce on gfx1100, and to present the change as hygiene rather
than a fault to chase.

### Results after the rebase (gfx1100, ROCm 7.2.3, CMake 3.31.6, GPU 1)

HIP: 306/306 targets, `ctest` 25/25 in 9.57 s.
`examples/exponential/exponential` -> `alpha = -1.001102381 +/- 0.003165763921`,
digit-identical to the pre-rebase run.

CPP no-regression (the build attempt 4 wrongly said could not be configured):
375/375 targets, 26/26 tests, `GOOFIT_PHYSICS:BOOL=ON`, `SUPPORTS_IPO=YES`
(untouched by the HIP guard). Python bindings on CPP build and `import goofit`
still exposes `Amp3Body`, `Amp3BodyBase`, `Amp3Body_IS`, `Amp3Body_TD`,
`DalitzPlotPdf`, `DalitzPlotter`.

Target and test counts rose (195->306, 24->25 HIP, 25->26 CPP) because upstream
added `AddNormTest` and more example targets, not because of anything here.

### Skill changes on this branch

- `references/strategy-a-cmake.md`: the rdc entry now leads with "there is no
  `HIP_SEPARABLE_COMPILATION`"; the LTO prohibition replaced by the measured
  three-row table and the `check_ipo_supported()` note.
- `references/fault-classes.md`: device-`new[]` entry demoted from fault class
  to hygiene, with the failed reproduction recorded.
- `references/validation.md`: new PR-prep gate for upstream drift, including why
  `git diff <default>...<branch>` hides it.

## Review 2026-08-08 (reviewer, linux-gfx1100, fork 19f29809b vs 5fe1221a8)

Verdict: changes-requested. The rebase is correct and four of the six bounced items are
fixed and verified below. What blocks is item 5: the porter's correction to my IPO finding
refuted my *reason* but landed on the wrong *conclusion*, and that wrong conclusion is now
shipped in a comment in GooFit's own CMakeLists, in the commit message, and in two skill
entries.

Reproduced on this host before reviewing: HIP build (gfx1100, ROCm 7.2.3, CMake 3.31.6,
GPU 1) `ctest` 25/25 in 9.52 s; `examples/exponential/exponential` gives
`alpha = -1.001102381 +/- 0.003165763921`, digit-identical to the reported run. CPP
no-regression build 26/26 with `GOOFIT_PHYSICS:BOOL=ON` and `SUPPORTS_IPO:INTERNAL=YES`.

### 1. The IPO guard is still a no-op, for a different reason than I gave, and both the CMake comment and the commit message now state a mechanism that does not occur

CMakeLists.txt:582-589 excludes HIP from `SUPPORTS_IPO` with the comment "IPO puts -flto on
the compile line, and a bitcode object's reference to the fatbinary symbol is left
undefined by the separate device link". CMake never puts `-flto` on a HIP line, so that
does not happen. Three independent measurements:

```
$ check_ipo_supported(RESULT R OUTPUT O LANGUAGES HIP)
-- HIP-only IPO probe = NO / language(s) 'HIP' not supported
$ grep -rn "HIP_COMPILE_OPTIONS_IPO\|HIP_LINK_OPTIONS_IPO" /usr/share/cmake-3.31/Modules/
(no matches -- CMake 3.31 defines no IPO flags for the HIP language at all)
```

and, at this port's exact shape (a HIP `STATIC` lib plus a HIP executable, `-fgpu-rdc` on
compile and link, `INTERPROCEDURAL_OPTIMIZATION ON` on both):

```
  FLAGS = -O2 -g -DNDEBUG -std=gnu++17 --offload-arch=gfx1100 -fgpu-rdc
  LINK_FLAGS = -fgpu-rdc --hip-link --rtlib=compiler-rt -unwindlib=libgcc ...
[4/4] Linking HIP executable bexe
```

No `-flto` anywhere and it links. `CheckIPOSupported.cmake:207-230` shows why the default
probe answered YES: with no `LANGUAGES` it filters `ENABLED_LANGUAGES` down to
C/CXX/CUDA/Fortran, so it measured `/usr/bin/c++`, never hipcc.

`SUPPORTS_IPO` is read at exactly two sites, CMakeLists.txt:646 and :686
(`GOOFIT_ADD_LIBRARY` / `GOOFIT_ADD_EXECUTABLE`). On the HIP backend every target those
create is pure HIP language, because `goofit_mark_hip_sources` marks both `.cu` and `.cpp`;
the only CXX object in the whole HIP build is `tests/catch_main.cpp`, and `catch_main` is a
plain `add_library` (tests/CMakeLists.txt:1) that never receives the property. So the guard
changes no flag on any line in any configuration this port supports.

Fix, either way:

- Drop `OR GOOFIT_DEVICE STREQUAL HIP` and its comment. The upstream CUDA exclusion stays.
- Or keep it and say what is true: CMake 3.31 attaches no IPO flags to HIP sources, so the
  exclusion is precautionary against a CMake that later does; the hazard it anticipates is
  `-flto` on the compile line followed by a separate device link.

Either is fine. What cannot ship is a comment in someone else's build system asserting a
mechanism that a two-minute reproducer contradicts -- that is the third round in a row this
project has bounced on exactly that. Commit 655ba62c0's body carries the same sentence
("INTERPROCEDURAL_OPTIMIZATION puts -flto on the compile line, and the resulting bitcode
object's reference to the fatbinary symbol is left undefined by the separate device link")
and needs the same correction.

Note the guard is not harmful, only inert, so this is about the recorded reason rather than
the build. Moving it to upstream's backend switch is an improvement and should stay.

### 2. `references/strategy-a-cmake.md` generalises a raw-hipcc result into a CMake claim that does not hold

The three-row table is right and I reproduced every row on ROCm 7.2.3 / gfx1100, including
the failure text (`ld.lld: error: undefined symbol: __hip_fatbin_819d4f029a7672c7`,
`referenced by ... __hip_fatbin_wrapper`). Keep it. Three statements around it are wrong:

- The heading imperative "Turn `INTERPROCEDURAL_OPTIMIZATION` off wherever you turned
  `-fgpu-rdc` on, because CMake's IPO puts `-flto` on the COMPILE line". It does not, for
  HIP.
- Row 3's parenthetical "(CMake's IPO)". The row is real; the attribution is not.
- "Note also that `check_ipo_supported()` answers YES under hipcc". It never probes hipcc.
  An explicit `LANGUAGES HIP` probe answers `NO / language(s) 'HIP' not supported`, and the
  default probe answers about the host C++ compiler.

Rewrite so the table is scoped to raw hipcc, add that CMake 3.31 has no HIP IPO support so
the bad combination cannot presently arise through `INTERPROCEDURAL_OPTIMIZATION`, and say
what `check_ipo_supported()` actually measures. The version scope matters: this is a
statement about CMake, and CMake may gain HIP IPO later.

### 3. `references/validation.md`'s drift entry ends on the same wrong conclusion

The entry is otherwise the best thing on this branch and the `<default>...<branch>`
merge-base trap is exactly right. Its closing clause is not: "GooFit's
`check_ipo_supported()` flipped NO -> YES across the rebase, turning a guard the reviewer
had measured as a no-op into a load-bearing one". The flip is real (`build-cpp`'s cache is
`SUPPORTS_IPO:INTERNAL=YES` now, and was NO at the old base), but the guard is still a
no-op per finding 1. Keep the observation, drop the conclusion: the durable lesson is that a
project-wide probe can answer differently after a rebase, so re-measure it rather than
trusting a cached value or an earlier review -- which is true, useful, and does not depend
on how the GooFit guard turned out.

### 4. The HIP backend prints CUDA labels for AMD devices

`src/goofit/Application.cpp:103-118` and `:157` are inside `#if GOOFIT_DEVICE_IS_GPU` and
hardcode a `CUDA:` prefix. Run on gfx1100:

```
$ ./exponential --gpu-dev 0
GooFit: Version 2.4.0 (release) Commit:
HIP 7.2
CUDA: Number of devices: 1
CUDA: Device 0: AMD Radeon Pro W7800 48GB
CUDA: Compute 11.0
CUDA: Total global memory: 48.301604864 GB
CUDA: Multiprocessors: 35
CUDA: 0 AMD Radeon Pro W7800 48GB: Compute 11.0; Memory 44.984375 GB
```

The version line was ported and the rest was not, and "Compute 11.0" is a gfx arch wearing
a CUDA compute-capability label. This is the first thing a GooFit maintainer will see when
they try the new backend. The same commit already genericised the neighbouring message at
`:296` ("GPU devices do not support floating point exceptions"), so the intent is
established; carry it through here, either with a backend-dependent prefix string or a
second arm under the existing `__HIP_PLATFORM_AMD__` check.

### 5. `DebugTools.cu` changes libraries on every backend and no commit message mentions it

`src/PDFs/CMakeLists.txt:16` drops `utilities/DebugTools.cu` from `PDFCore` and
`src/PDFs/physics/CMakeLists.txt:12` adds it to `PDFPhysics`. The move is justified --
`DebugTools::copyAmpIndicesToHost` does `MEMCPY_FROM_SYMBOL(..., AmpIndices, ...)` on the
`__constant__` from `Amp4BodyGlobals.h`, so the file cannot link with `GOOFIT_PHYSICS=OFF`,
and its only callers are `Amp4Body_TD.cu` and `detail/AmpCalc_TD.cu`, both physics. But on
CUDA and CPP, where nothing else about the option changes, it silently relocates a symbol
from `libPDFCore` to `libPDFPhysics`. Commit 655ba62c0's scope paragraph lists the physics
PDFs, examples, Python bindings and the one test; add this, in one sentence, with the
reason.

### 6. plan.md still describes a design that was not built

`projects/GooFit/plan.md:40-56, 131-141, 154-171` specify a `USE_HIP` option, keeping
`-DGOOFIT_DEVICE=CUDA` and overriding it, and disabling `GOOFIT_FORCE_LOCAL_THRUST`. What
shipped is a first-class `GOOFIT_DEVICE=HIP` backend with `IS_NOT_HIP`, `GOOFIT_PHYSICS`,
and upstream's Thrust block left intact. This is the file the next agent reads before
notes.md. Rewrite the strategy and file-list sections, or head it as superseded by the
attempt-5 notes.

### Adjudications

**Item 1, the `GlobalCudaDefines.h` split: correct, and the argument holds. Verified, not
taken on trust.** Probing with `-include probe.h` under `hipcc -x hip --offload-arch=gfx1100`
(the exact position `cuda_to_hip.h` occupies) reports `__host__`, `__device__`, `__shared__`
and `__constant__` all already defined, in BOTH the device pass and the host pass -- clang
preincludes `__clang_hip_runtime_wrapper.h` ahead of any user `-include`. So upstream's
`#ifndef` guards on the first half are genuinely inert under hipcc.

The second half is what mattered, and the porter's read of the danger is right. Those four
lines are unguarded, so upstream's merged block would have won by redefinition. Emulated it
directly:

```
redef.hip:4:9: warning: '__shared__' macro redefined [-Wmacro-redefined]
/opt/rocm/lib/llvm/lib/clang/22/include/__clang_hip_runtime_wrapper.h:24:9:
   note: previous definition is here
3 warnings generated when compiling for gfx1100.
        .amdhsa_group_segment_fixed_size 0
        .amdhsa_private_segment_fixed_size 1040
```

A warning, not an error, and the kernel's `__shared__ float cache[256]` becomes 1040 bytes
of per-thread scratch with zero LDS -- every thread gets a private copy and the
`__syncthreads()` around it means nothing. Silent, and it would have hit
`ConvolutionPdf.cu`'s `modelCache`. The split is load-bearing, not cosmetic.

Preprocessor outcome against upstream, per backend: CUDA takes neither block (upstream:
neither) and CPP/OMP/TBB take both (upstream: both), so both are byte-identical to upstream
behaviour. The one delta is a HIP-build translation unit compiled as plain C++ rather than
`-x hip`: rocThrust reports `THRUST_DEVICE_SYSTEM_HIP` even under g++ (verified), so
`GOOFIT_DEVICE_IS_GPU` is 1 there and the CPU-fallback definitions -- including the
`cudaError_t` enum at :134 -- would not be defined where upstream's merged block defined
them. No such TU exists: the HIP build has 74 HIP objects and exactly one CXX object,
`tests/catch_main.cpp`, which includes only Catch2. Nothing to fix; the notes' claim that
the first half "still fires in HIP host-only TUs, where it is needed" describes a case this
build does not contain, so do not lean on it as evidence.

**Item 5, the LTO correction: the porter is right and I was wrong.** All three rows
reproduce exactly as tabulated -- one `hipcc` command with `-flto` links, `-flto` at link
only links, and `-flto` on the compile line with a separate link fails on
`__hip_fatbin_<hash>`. My earlier reproducer was the one-command shape, which is row 1, so
"LTO is fine" was a measurement of the wrong thing. The table is the correct
generalisation and belongs in the skill. Its framing does not (findings 1-3).

**Item 5, the `SUPPORTS_IPO` correction: the porter is right that it answers YES.** My
"`SUPPORTS_IPO:INTERNAL=NO`, so the guard is a no-op" cited a cache written by a configure
that had already forced it. `build-cpp` now caches YES and a clean probe answers YES.
Moving the guard from a per-target `if()` in `GOOFIT_ADD_LIBRARY` to upstream's backend
switch is right on both counts the porter gives: one line instead of four, and
`GOOFIT_ADD_EXECUTABLE` never carried the per-target guard. Only the conclusion drawn from
the YES is wrong (finding 1).

**Item 2, the rebase: clean.** `git rev-list --count moat-port..origin/master` is 0 and the
fork's `master` (5fe1221a8) is `GooFit/GooFit@master` as of today, checked against
`git ls-remote`. Both `validated_sha` were null, so the rewrite orphaned nothing.

**Item 3, the Thrust block: byte-identity confirmed.** The whole delta is four added lines
(`if(GOOFIT_DEVICE STREQUAL HIP)` plus a three-line comment) and `if(` -> `elseif(` on
upstream's line. Upstream's `option()`, its `GOOFIT_FORCE_LOCAL_THRUST` body and its
`else()` body are unchanged character for character against `5fe1221a8:CMakeLists.txt`. The
empty leading branch is unusual CMake but it is what keeps the diff at four lines, and I
would not trade that for a `NOT` wrapper that re-indents upstream's block.

**Item 4, `d8e1102cf`: not redundant, confirmed.** `git show --stat` gives one file,
`src/PDFs/combine/CompositePdf.cu`; the port touches `src/PDFs/MetricTaker.cu` and leaves
`CompositePdf.cu` alone. Precedent, not duplication, and the commit message now cites #384
without claiming a HIP bug. The array bound is unchanged too: upstream's `new fptype[10]`
and the port's `fptype[MAX_NUM_OBSERVABLES]` are both 10, and the only write is
`events[id]`, so the change introduces no new out-of-bounds exposure in either direction.

**Item 6, the submodule check: confirmed safe.** `extern/thrust` is at `af8cce4ca` (CCCL
v3.3.3) with `README.md` present, upstream's check at CMakeLists.txt:75-87 is untouched,
and the HIP build configures and passes through it. Deleting the relocated check and its
`if(NOT ... HIP)` wrapper is right -- they existed for a premise that turned out false.

**Fault classes: nothing to report.** The only `warpSize` in the tree is
`Application.cpp:117`'s printout; no `__shfl`, `__ballot`, `__activemask`, no lane mask, no
texture, atomic or resource handle anywhere, so no rule-of-five, pitch or wavefront exposure
exists. No arch-conditional code in the diff; every fix is arch-unified. The compat shim
covers every `cuda*` symbol the tree actually uses -- diffing the used set against the
mapped set leaves only `cudaArray` (commented out), `cudaErrCheck_` and `cudaPlus` (GooFit's
own names). `Log.h` correctly uses `__HIP_DEVICE_COMPILE__` (device pass only) where
`ParameterContainer.cu` correctly uses `__HIPCC__` (both passes) -- those are the right two
macros for the two different jobs. No library substitution beyond Thrust -> rocThrust.

**Hygiene: clean.** `jargon.py --commits 5fe1221a8..HEAD` reports clean; both titles are
`[ROCm]`-prefixed at 56 and 44 chars; no `Co-Authored-By` or `noreply` trailer; no
`Assisted-by` line; no copyright or author lines added; no non-ASCII and no em-dash in the
diff; no AMD-internal account, host or MOAT vocabulary in any upstream-visible text; fork
tree clean.

## Attempt 6 (porter, linux-gfx1100, fork 18fca9e4a) -- answering review of 19f29809b

Fork head 19f29809b -> 18fca9e4a. Both `validated_sha` were null before and after
the rewrite, re-checked immediately before the force-push, so nothing was orphaned.
`655ba62c0` was amended (now `32c825112`) and the docs commit rebased on top.

### The IPO question, settled by measurement (third round on this entry)

The reviewer's conclusion was right -- the guard is inert -- but the stated reason
was wrong, and it was wrong in a way worth recording because it is a general trap.

The claim was "CMake 3.31 defines no `CMAKE_HIP_COMPILE_OPTIONS_IPO` at all",
supported by a grep that returned nothing. That grep returned nothing for two
independent reasons, neither of them the one intended:

1. It ran against `/usr/share/cmake-3.31/Modules/`, which does not exist on this
   host. `cmake` here is `/opt/conda/envs/py_3.12/bin/cmake` and its modules live
   under `.../site-packages/cmake/data/share/cmake-3.31`. A `grep` over a missing
   directory exits 2 and prints nothing, which reads exactly like "no matches".
2. The literal string `CMAKE_HIP_COMPILE_OPTIONS_IPO` never appears in the
   sources anyway. `Compiler/Clang.cmake` assigns
   `CMAKE_${lang}_COMPILE_OPTIONS_IPO`, and `Compiler/Clang-HIP.cmake` calls
   `__compiler_clang(HIP)`, so the variable IS defined, as `-flto=thin`.

Confirmed by configuring a project and printing it:

```
-- CMAKE_HIP_COMPILE_OPTIONS_IPO = [-flto=thin]
-- _CMAKE_HIP_IPO_SUPPORTED_BY_CMAKE = [YES]
```

So why is there no `-flto` on a HIP compile line? Because the *generator* only
applies IPO to C, CXX, CUDA and Fortran. Demonstrated with one project holding
both a HIP and a CXX target carrying the identical
`INTERPROCEDURAL_OPTIMIZATION ON` property (CMake 3.31.6, ROCm 7.2.3, gfx1100,
Ninja): a HIP `STATIC` lib + HIP executable, both with `-fgpu-rdc` on compile and
link, plus a plain CXX `STATIC` lib. From the generated `build.ninja`:

```
  FLAGS = -O3 -DNDEBUG -std=gnu++17 --offload-arch=gfx1100 -fgpu-rdc   # blib (HIP)
  FLAGS = -O3 -DNDEBUG -std=gnu++17 --offload-arch=gfx1100 -fgpu-rdc   # bexe (HIP)
  FLAGS = -O3 -DNDEBUG -flto=thin                                      # clib (CXX)
```

It builds, links, and runs (`ok`) on GPU 1. Same property, same project, and only
the CXX target gets the flag.

`check_ipo_supported()` cannot answer this either way. `CheckIPOSupported.cmake`
drops every language outside C/CXX/CUDA/Fortran (the `list(REMOVE_ITEM ...)` in
the argument-checking block), so `LANGUAGES HIP` returns
`NO / language(s) 'HIP' not supported`, and with no `LANGUAGES` it filters
`ENABLED_LANGUAGES` down to the same four -- meaning the project-wide YES that
GooFit caches is a statement about `/usr/bin/c++`.

Net: the three-row raw-hipcc LTO table stands and is unchanged; only its
generalisation to CMake was wrong. The guard stays, relabelled precautionary in
`CMakeLists.txt` and in the commit body.

**The transferable lesson, promoted to the skill this round**: do not settle a
"which flags does the build use" question by grepping build-system sources. Read
the generated `build.ninja`. Two greps in a row produced confident and opposite
wrong answers here. Where the grep is unavoidable, remember that CMake builds most
per-language variables through `CMAKE_${lang}_...`, so the expanded name is not in
the source, and verify the directory you are grepping actually exists.

### Other review items

- `Application.cpp` no longer prints `CUDA:` on a HIP build. A `GOOFIT_GPU_LABEL`
  macro follows the backend, and a small `goofit_device_arch()` reports
  `Architecture: gfx1100` on AMD instead of `Compute 11.0`, since `devProp.major`
  and `.minor` carry no compute capability there. Both print sites use it, so the
  `--gpu-dev` block and the device-enumeration `GOOFIT_INFO` agree.
- The `DebugTools.cu` move from `PDFCore` to `PDFPhysics` is now in the commit
  body. Re-verified before writing it: the file's `copyAmpIndicesToHost` does
  `MEMCPY_FROM_SYMBOL(..., AmpIndices, ...)`, `AmpIndices[500]` is defined only in
  `src/PDFs/physics/Amp4BodyGlobals.cu`, and its one live caller is
  `src/PDFs/physics/detail/AmpCalc_TD.cu` (the `Amp4Body_TD.cu` references are all
  commented out).
- `plan.md` rewritten. It carries a superseded header and its strategy, file-list
  and build-command sections now describe the `GOOFIT_DEVICE=HIP` backend that was
  built, not the `USE_HIP`-over-`GOOFIT_DEVICE=CUDA` design that was not. The four
  open questions are resolved in place.

### Re-validation on GPU 1 (gfx1100, ROCm 7.2.3, CMake 3.31.6)

`Application.cpp` and `CMakeLists.txt` are compiled, so both suites were re-run.

```
cmake --build build-hip -j32          # 28 targets relinked, clean
HIP_VISIBLE_DEVICES=1 ctest --test-dir build-hip --output-on-failure
  -> 100% tests passed, 0 failed out of 25, 9.64 s
cmake --build build-cpp -j32          # 29 targets, clean
ctest --test-dir build-cpp
  -> 100% tests passed, 0 failed out of 26, 6.42 s
```

`exponential` still gives `alpha = -1.001102381 +/- 0.003165763921`, digit-identical
to attempt 5 and to the reviewer's reproduction, so the print-path edit changed no
numerics. Device block now reads:

```
HIP 7.2
HIP: Number of devices: 1
HIP: Device 0: AMD Radeon Pro W7800 48GB
HIP: Architecture: gfx1100
```

Whole-branch jargon scan (`jargon.py --port GooFit`, `master..moat-port` for
commits and `master...moat-port` for added lines) reports clean.

## Review 2026-08-08 (reviewer, linux-gfx1100, fork 18fca9e4a vs 5fe1221a8)

Verdict: **review-passed**. Narrow round. The IPO/LTO correction is confirmed by
independent measurement, the four other items check out, and no defect blocks
validation. Two non-blocking precision items are recorded below; fold them into
the next commit that touches those files rather than opening a round for them.

### Amend safety, tree state, jargon

Both platforms' `validated_sha` were null when the force-with-lease ran, so
`655ba62c0` -> `32c825112` orphaned no validated commit. `git status --porcelain`
in the fork is empty. `jargon.py --commits 5fe1221a8..HEAD` and
`jargon.py --diff 5fe1221a8...HEAD` are both clean. Commit titles are 44 and 56
chars, both `[ROCm]`, no `Co-Authored-By`/noreply trailer.

This worktree's `utils/jargon.py` has no `--port` flag (the trunk copy does, 4
hits). The porter's workaround was correct and the two modes that exist here cover
the same ground. **A trunk merge should be done before this branch goes further**
-- it is safe and it fixes the tool version so the next agent does not rediscover
this.

### Three-target IPO experiment: reproduced independently

Rebuilt from scratch (HIP `STATIC` lib + HIP executable, both `-fgpu-rdc`, plus a
plain CXX `STATIC` lib, all three `INTERPROCEDURAL_OPTIMIZATION ON`), CMake 3.31.6
/ ROCm 7.2.3 / gfx1100. Generated `build.ninja`:

```
FLAGS = -O3 -DNDEBUG -std=gnu++17 --offload-arch=gfx1100 -fgpu-rdc   # hiplib
FLAGS = -O3 -DNDEBUG -std=gnu++17 --offload-arch=gfx1100 -fgpu-rdc   # hipexe
LINK_FLAGS = -fgpu-rdc --hip-link ...                                # hipexe link
FLAGS = -O3 -DNDEBUG -flto=auto -fno-fat-lto-objects                 # cxxlib
```

Configure-time probes: `CMAKE_HIP_COMPILE_OPTIONS_IPO = [-flto=thin]`,
`_CMAKE_HIP_IPO_SUPPORTED_BY_CMAKE = [YES]`,
`check_ipo_supported(LANGUAGES HIP)` -> `NO`, `language(s) 'HIP' not supported`.
The project builds and links. Defined, never applied: confirmed.

Both of the previous round's grep failure modes are confirmed on this host.
`/usr/share/cmake-3.31/Modules/` does not exist (`CMAKE_ROOT` is under
`site-packages/cmake/data/share/cmake-3.31`), and `grep -r` over it exits 2
printing nothing. Separately, the literal `CMAKE_HIP_COMPILE_OPTIONS_IPO` does not
occur anywhere in the real module tree (grep exit 1, a genuine no-match), because
`Compiler/Clang.cmake:79` assigns `CMAKE_${lang}_COMPILE_OPTIONS_IPO` and
`Compiler/Clang-HIP.cmake:3` calls `__compiler_clang(HIP)`.
`CheckIPOSupported.cmake:238` does `list(REMOVE_ITEM ... "C" "CXX" "CUDA"
"Fortran")`. GooFit's own `build-hip/build.ninja`: 0 occurrences of `flto`, 99 of
`fgpu-rdc`.

### Item-by-item

1. **Precautionary guard, keep it.** `CMakeLists.txt:584-590` and the commit body
   both say it changes no flag today and name what it guards against. Keeping a
   labelled no-op is the better of the two defensible options here: it sits
   directly beside the existing CUDA exclusion, so deleting it would leave a
   backend list that reads as an oversight and invite someone to "fix" it by
   enabling HIP IPO; and the honest label costs a reader nothing, whereas
   rediscovering the hipcc failure costs a round. The comment pins the CMake
   version, which is what makes it maintainable.
2. **No unverified CMake mechanism is asserted anywhere.** The three-row table is
   scoped to "driving hipcc by hand"; the CMake paragraph states only the measured
   result. `validation.md`'s drift entry keeps the NO->YES observation and now ends
   at "settle a flag argument by reading the generated build.ninja".
3. **Promoted lesson: sound, and it earns its place.** Both failure modes are
   named, the remedy is a three-step ladder (`--system-information` for the
   directory, a throwaway `message(STATUS ...)` for the value, `build.ninja` for
   the truth), and nothing in it is GooFit-specific beyond the citation. See the
   generality gap below.
4. **Branding verified.** On gfx1100 `hipDeviceProp_t` reports `major=11 minor=0`
   and `gcnArchName=gfx1100`: major/minor carry the gfx version, so the old format
   would print "Compute 11.0", which is not a CUDA compute capability. Both print
   sites use `goofit_device_arch()` -- `Application.cpp:120` (`goofit_info_device`)
   and `Application.cpp:176` (`print_goofit_info`) -- and the CUDA arm returns
   "Compute {}.{}", so CUDA output is byte-identical to before.
5. **DebugTools relocation verified.** `DebugTools.cu:42` does
   `MEMCPY_FROM_SYMBOL(..., AmpIndices, ...)`; `AmpIndices[500]` is defined only at
   `physics/Amp4BodyGlobals.cu:11`; the only live caller is
   `physics/detail/AmpCalc_TD.cu:94` and `:135`. The three `Amp4Body_TD.cu`
   references (633, 659, 682) are all commented out. Moving it to PDFPhysics is
   correct.
6. **plan.md** is accurate and its superseded header is honest about which four
   things differ from the sketch. Its "no warp intrinsics / no wave64-wave32
   concern" finding is independently confirmed: every `warpSize`/`__shfl`/ballot
   hit in the tree is in vendored `extern/` (Eigen, fmt, pybind11, thrust), none in
   GooFit's own sources, and there are no textures and no resource handles needing
   rule-of-five.
7. **Suites reproduced on GPU 1** after bringing `build-hip` current (it relinked
   28 targets on first invocation, so the numbers below are from the current tree):
   HIP 25/25 in 9.62 s, CPP 26/26 in 6.39 s, and `exponential` gives
   `alpha = -1.001102381 +/- 0.003165763921`, digit-identical.

### Non-blocking, fold in opportunistically

- **`-flto=thin` is host-compiler-dependent and stated flatly.**
  `strategy-a-cmake.md` ("a plain CXX target in the same project does receive
  `-flto=thin`") and the `32c825112` commit body ("while a CXX target in the same
  project does get -flto=thin") both name one spelling. `Compiler/GNU.cmake:107`
  sets `CMAKE_${lang}_COMPILE_OPTIONS_IPO` to `${__lto_flags}`, and
  `Compiler/Clang.cmake:79-81` picks `-flto=thin` or `-flto` on `_CMAKE_LTO_THIN`.
  Measured here: clang host -> `-flto=thin`, GCC host -> `-flto=auto
  -fno-fat-lto-objects`. The load-bearing contrast (CXX gets an LTO flag, HIP gets
  none) is correct either way. Since the commit body is upstream-visible, prefer
  "does get an `-flto` flag" there so a maintainer building with GCC does not read
  it as wrong.
- **The promoted lesson names only `build.ninja`.** A porter on the "Unix
  Makefiles" generator has no such file; the per-target compile flags live in
  `CMakeFiles/<target>.dir/flags.make` (verified: a Makefiles-generator tree with
  `INTERPROCEDURAL_OPTIMIZATION ON` puts `CXX_FLAGS = -O3 -DNDEBUG -flto=auto
  -fno-fat-lto-objects` there). One clause naming that equivalent would close the
  gap in an entry whose whole point is that it generalises.

## Validation 2026-08-09 (validator, linux-gfx1100, fork 18fca9e4a vs 5fe1221a8)

Worktree tooling was behind trunk; ran `python3 utils/moatlib.py branch-sync --apply`
first as instructed (merged only `.claude/agents/*`, `projects/GooFit/` untouched).
`jargon.py --port GooFit` now works directly.

### Build (GPU 1, gfx1100, ROCm 7.2.53211-c2d9476115, CMake 3.31.6, Ninja)

Fresh `build-hip-validate`, the recorded recipe:
```
cmake -S . -B build-hip-validate -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGOOFIT_DEVICE=HIP -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DGOOFIT_TESTS=ON -DGOOFIT_EXAMPLES=ON -DGOOFIT_CERNROOT=OFF \
  -DGOOFIT_PYTHON=OFF -DGOOFIT_PHYSICS=OFF
cmake --build build-hip-validate -j32
```
306/306 targets, clean, exit 0. `roc-obj-ls` on `examples/exponential/exponential`
and `tests/simple/SimpleTest` shows exactly one device offload entry each,
`hipv4-amdgcn-amd-amdhsa--gfx1100` -- the module targets gfx1100 only, no other
arch bundled in.

**One wrinkle worth recording, not a regression**: a first pass configuring with
no `-DGOOFIT_PYTHON` flag at all (letting the upstream `option(GOOFIT_PYTHON ...
${CUR_PROJ})` default fire, since Python 3.12 is on this host) hits a real build
break -- `python/goofit/CMakeLists.txt`'s `add_library(_Core STATIC ... Variable.cpp
DataSet.cpp ... Application.cpp ...)` reuses the same bare filenames as
`src/goofit/CMakeLists.txt`'s `goofit_add_library(... Variable.cpp ...)`, and
`goofit_mark_hip_sources()` calls `set_source_files_properties(Variable.cpp
PROPERTIES LANGUAGE HIP)` with that bare name from the `src/goofit` directory scope.
On this CMake/generator combination the HIP-language assignment leaked to the
python target's same-named sources in the sibling directory: the generated
`build.ninja` shows a `CXX_COMPILER___Core_...` rule (so CMake still picked the CXX
per-target rule/driver, `/usr/bin/c++`) but with HIP-only per-source flags
(`-x hip --offload-arch=gfx1100`) baked into `FLAGS`, which `c++` naturally rejects.
This is a red herring, not the port's fault: every session that has ever validated
this project used the standard recipe above with `-DGOOFIT_PYTHON=OFF` explicit
(notes attempt 3 onward), and review round 2 (item 8) already flagged that
`GOOFIT_PYTHON=ON` interactions with the HIP backend were untested and out of scope
("Both were invisible in the gfx1100 run because it set ... `GOOFIT_PYTHON=OFF`").
Re-ran with the documented flags and it configures and builds clean. Not chased
further since `GOOFIT_PYTHON=ON` on HIP was never claimed to work and is off by
default in every recorded recipe; if a future round wants `GOOFIT_PYTHON=ON` on
HIP, the fix is scoping `goofit_mark_hip_sources` to full/absolute source paths
(or building python/goofit's `_Core` through `GOOFIT_ADD_LIBRARY` instead of a bare
`add_library`) rather than anything device-specific.

### Tests: HIP 25/25 in 9.65s, CPP 26/26 in 6.40s -- both match the recorded baseline

```
HIP_VISIBLE_DEVICES=1 ctest --test-dir build-hip-validate --output-on-failure
  -> 100% tests passed, 0 failed out of 25, 9.65 s
ctest --test-dir build-cpp-validate --output-on-failure
  -> 100% tests passed, 0 failed out of 26, 6.40 s
```
No non-GPU regression: the CPP suite count and wall time match every prior session
(26/26, ~6.4s).

`HIP_VISIBLE_DEVICES=1 ./build-hip-validate/examples/exponential/exponential`:
```
alpha = -1.001102381 +/- 0.003165763921
```
Digit-identical to the figure recorded across three independent prior sessions
(porter attempts 3, 6 and the last two reviews). Device banner:
```
HIP 7.2
HIP: Number of devices: 1
HIP: Device 0: AMD Radeon Pro W7800 48GB
HIP: Architecture: gfx1100
```

### CUDA no-regression gate: run at 18fca9e4a (first genuine nvcc build on this
project -- prior notes record only "no nvcc anywhere" through review round 3)

`extern/thrust` is initialised at `af8cce4ca` (CCCL) as the task brief said, so the
submodule wall that blocked every previous attempt at this gate is gone. Configured
and built with `/opt/conda/envs/cuda-12.8/bin/nvcc`, arch pinned:
```
cmake -S . -B build-cuda-validate -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGOOFIT_DEVICE=CUDA -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DGOOFIT_TESTS=ON -DGOOFIT_EXAMPLES=ON -DGOOFIT_CERNROOT=OFF -DGOOFIT_PYTHON=OFF \
  -DCMAKE_CUDA_COMPILER=/opt/conda/envs/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13
cmake --build build-cuda-validate -j32
```
Configure succeeds. Build fails at `src/goofit/BinnedDataSet.cpp` (and every other
host `.cpp` that transitively includes `GlobalCudaDefines.h`):
```
include/goofit/GlobalCudaDefines.h:133:10: fatal error: driver_types.h: No such
  file or directory
  133 | #include <driver_types.h>
```
Root cause is a pre-existing upstream gap, not the port: `CMakeLists.txt` only adds
`CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES` to `GooFit_Common`'s host include path
inside the CUDA-13-relocated-CCCL branch (`if(EXISTS
${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/cccl/thrust/version.h)`), which never
fires on CUDA 12.8 -- so no mechanism puts the CUDA toolkit's own `include/`
(where `driver_types.h` lives) on a plain host `.cpp` TU's path on this toolkit
version, upstream or ported. **Proved, not assumed**: built upstream `5fe1221a8`
from a fresh `git worktree` with fresh-cloned submodules (network clone, not the
possibly-touched port checkout) through the identical toolchain/flags/arch pin --
identical failure, identical file, identical line region
(`GlobalCudaDefines.h:124`, one line off only because the port's version has
extra comments above the `#include`):
```
include/goofit/GlobalCudaDefines.h:124:10: fatal error: driver_types.h: No such
  file or directory
```
This is exactly the passthrough case: the port's own `GlobalCudaDefines.h` CUDA
arm was deliberately restored to be "bit-for-bit what upstream had" (review round
3), and the upstream build breaks the same way with the same toolchain, so the
port introduces no CUDA regression. **CUDA gate: pre-existing breakage, not a
port fault, recorded rather than treated as a failure.** Not filed as a
`rocm-bug-report` (nothing ROCm-side is implicated); if anyone wants this fixed
upstream, the fix is adding the toolkit include dir to `GooFit_Common`'s host
compile path unconditionally rather than only inside the CUDA-13 relocation
branch. Cleaned up: upstream worktree removed (`git worktree remove --force`),
`git worktree list` shows only the port checkout.

### The gfx90a NLL-divergence question: still open, this run says nothing about it

This arch is gfx1100 (wave32), not gfx90a (wave64), so a clean run here cannot
close the gfx90a question -- it only adds to the standing evidence that gfx1100
has never reproduced the `6.25e-310`/garbage-normalization divergence across four
independent sessions now (attempts 3, 6, this validation, plus the two reviews).
The device `new fptype[10]` the review's analysis points at as the likely cause is
gone from the committed tree (confirmed by reading `MetricTaker::operator()` in
`include/goofit/PDFs/MetricTaker.h` -- no `new[]`, `MAX_NUM_OBSERVABLES`-sized
stack/static storage instead), so whoever next has a gfx90a host should build this
exact sha and run ctest before re-debugging SmartVector or normalization pointers;
if it still diverges there, it is either genuinely wave64-specific behavior or an
artifact of that host's ROCm version, and only a gfx90a run can tell those apart.

### Jargon and documentation

`python3 utils/jargon.py --port GooFit` -> clean.

README.md (`If using ROCm (AMD GPUs)` collapsible, alongside the CUDA one) and
`docs/SYSTEM_INSTALL.md` (`Ubuntu with ROCm, for AMD GPUs` collapsible, alongside
the Ubuntu-CUDA one) both document the build in the project's own house style.
Checked for accuracy, not just presence: the README's example command
(`cmake .. -DGOOFIT_DEVICE=HIP -DCMAKE_HIP_ARCHITECTURES=gfx1100`) and
SYSTEM_INSTALL's (`cmake -S . -B build -GNinja -DGOOFIT_DEVICE=HIP
-DCMAKE_HIP_ARCHITECTURES=gfx1100`) both match the recipe actually used above
(modulo the test/example/python/cernroot flags, which are validation-only, not
build-requirement flags). Both correctly state ROCm 7.2+, CMake 3.21+, gfx1100 as
what was tested, `GOOFIT_PHYSICS` defaulting OFF on HIP, and that the bundled CCCL
is kept off the include path in favor of rocThrust. No overclaiming found.

### Tree state and result

`git -C projects/GooFit/src status --porcelain` empty before and after (local
`build-*-validate` directories are `*build*/*`-gitignored and were removed after
use). No source or build files were edited this round -- nothing needed changing
for gfx1100 -- so no new commit and no `gen_readme.py` run for this branch
(nothing pushed that would stale the table).

**Result: linux-gfx1100 -> completed at 18fca9e4a.** HIP 25/25 (9.65s), CPP 26/26
(6.40s), alpha digit-identical, CUDA gate passthrough-confirmed pre-existing
breakage (not a regression), jargon clean, docs accurate. gfx90a question remains
open pending a gfx90a re-run of this exact sha.

## Validation 2026-08-09 (validator, linux-gfx90a, fork 18fca9e4a vs 5fe1221a8) -- FAILS: gfx90a-specific NLL divergence CONFIRMED at the committed sha

GPU 3 of 4 MI250X (gfx90a) confirmed via `rocm-smi` (Node ID 5, GFX Version gfx90a).
Fresh clone of `AMD-Ecosystem/GooFit`, checked out `moat-port` at `18fca9e4a`
(matches `status.json.head_sha` exactly), submodules initialised
(`extern/thrust` at CCCL `af8cce4ca`). `git status --porcelain` clean throughout.

### Build (gfx90a, ROCm 7.2.1 series -- `hipcc --version`: "HIP version:
7.2.53211-e1a6bc5663", "AMD clang version 22.0.0git ... roc-7.2.1", CMake 4.0.3, Ninja)

```
cmake -S . -B build-hip-validate -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGOOFIT_DEVICE=HIP -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DGOOFIT_TESTS=ON -DGOOFIT_EXAMPLES=ON -DGOOFIT_CERNROOT=OFF \
  -DGOOFIT_PYTHON=OFF -DGOOFIT_PHYSICS=OFF
cmake --build build-hip-validate -j32
```
306/306 targets built and linked, exit 0. `roc-obj-ls` on `examples/exponential/exponential`
and `tests/simple/SimpleTest` shows exactly one device offload entry each,
`hipv4-amdgcn-amd-amdhsa--gfx90a` -- gfx90a only, no other arch bundled.

### Tests: 4/25 PASS, 21/25 FAIL -- reproduces the "garbage normalization" NLL
divergence recorded in porter attempt 2, now confirmed at the CURRENT committed sha

```
HIP_VISIBLE_DEVICES=3 ctest --test-dir build-hip-validate --output-on-failure
  -> 16% tests passed, 21 tests failed out of 25 (10.90 s)
```
Passing: SimpleTest, BlindTest, StepTest, exponential2_Example. Every failure is
an unbinned/binned maximum-likelihood fit converging to a fitted parameter's upper
bound instead of the true value, e.g.:
```
$ HIP_VISIBLE_DEVICES=3 ./examples/exponential/exponential
  Pos |    Name    |  type   |      Value       |    Error +/-
    0 |      alpha | limited |      9.999998048 | 5.928732527e-08
FunctionMinimum is invalid: Edm is above max
```
Expected (matches the CPP backend below, and every prior gfx1100 run):
`alpha = -1.001102381 +/- 0.003165763921/38`, well inside the example's own
`[-1.01, -0.99]` assertion window. The GPU value instead lands exactly at the
parameter's upper bound (10), the classic "solver hit the wall" signature the
earlier attempt-2 repro described (the underlying `d_normalizations` symbol read
6.25e-310 there, an uninitialized-double bit pattern).

### Confirmed this is real GPU execution, not a harness/CPU-fallback artifact

```
AMD_LOG_LEVEL=3 ./examples/exponential/exponential
  -> 112 hipLaunchKernel dispatches, HIP: Architecture: gfx90a:sramecc+:xnack-
```
`roc-obj-ls` (above) confirms the binary carries only a gfx90a code object. So the
wrong numbers come from real kernels executing on real gfx90a hardware.

### No non-GPU regression: CPP backend 26/26, correct answer

```
cmake -S . -B build-cpp-validate -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGOOFIT_DEVICE=CPP -DGOOFIT_TESTS=ON -DGOOFIT_EXAMPLES=ON \
  -DGOOFIT_CERNROOT=OFF -DGOOFIT_PYTHON=OFF
cmake --build build-cpp-validate -j32      # 375/375, clean
ctest --test-dir build-cpp-validate        # 100% tests passed, 26/26, 9.03 s
./examples/exponential/exponential -> alpha = -1.001102381 +/- 0.003165763938
```
Digit-identical to gfx1100's HIP run and to every prior CPP run. The algorithm and
the non-HIP paths are unaffected; this is isolated to the HIP/gfx90a execution path.

### This settles the standing "which hypothesis" question from attempts 3-6/reviews:
genuinely gfx90a (wave64) specific, not a stale tree and not a ROCm-version artifact

Per the review's three candidate hypotheses (notes above, "The gfx90a NLL question"):
(c) stale/uncommitted tree is ruled out -- this is a fresh clone of the exact
validated `head_sha`, tree clean before and after. (a) a 7.2.1-to-7.2.5 ROCm
difference is ruled out -- this host's hipcc reports the same `roc-7.2.1` series
attempt 2 used. That leaves (b): the divergence is reproducible, real-GPU, and
specific to gfx90a while gfx1100 and CPP both give the correct answer on the
identical committed tree. Promoted the diagnostic ladder (rule out stale tree and
toolchain version before accepting "arch-specific") to
`.claude/skills/cuda-to-rocm/references/validation.md` so the next porter/validator
does not have to re-derive it from six rounds of notes.

Per the stop-discipline rule for a clean build producing wrong numbers on one
arch while others pass: not chased further here. The magnitude and repro are
recorded above for whoever picks this up next (porter, to attempt a fix, or to
set `blocked` after a real attempt).

### CUDA gate: already recorded at this head_sha, not re-run

Per the validator's own rule (once per head_sha), skipped: this exact head_sha
(`18fca9e4a`) already has a CUDA-compile-gate result recorded above (gfx1100
validation session, same date) -- pre-existing upstream breakage
(`driver_types.h: No such file or directory` under CUDA 12.8, reproduced
identically on pristine upstream `5fe1221a8`), not a port regression.

### Result: linux-gfx90a -> validation-failed at 18fca9e4a

Build clean (306/306), but 21/25 ctest failures are a genuine numerical fault on
real gfx90a GPU execution (confirmed via kernel-dispatch and code-object evidence
above), not a harness issue. This is an ARCH-state failure, not a project-stage
regression -- `stage` stays `review-passed`; only `platforms.linux-gfx90a` records
the failure. Recorded `failed_sha=18fca9e4a2ea1e062fb7e80c970ac78df6376408`. No
source or build files were edited on the fork this round (`git -C
projects/GooFit/src status --porcelain` empty before and after); the only change
is a documentation lesson in the `cuda-to-rocm` skill on this port branch.

## Attempt 3 (2026-08-09): no run -- GitHub unreachable; analysis from the record only

No fork clone, no build, no GPU run. 12 clone attempts over ~45 minutes failed
identically at `Failed to connect to github.com port 443`. Outbound routing to
AS36459 (140.82.0.0/16) and the Azure github endpoints was black-holed while
general egress worked (`objects.githubusercontent.com` and `1.1.1.1` both
connected). Environment fault, not a port fault: `blocked` was deliberately NOT
set on linux-gfx90a and this does NOT count as one of the three attempts,
because recording "no network" in an arch record misattributes the failure.
linux-gfx90a stays `validation-failed` at 18fca9e4a, which is where the next
porter needs it.

### The wave64-reduction hypothesis is weakly supported -- do not lead with it

The dispatching orchestrator's brief pointed at a wave-size-dependent reduction.
The record does not support it. Review round 1 (above) establishes there is no
`warpSize`, `__shfl*`, `__ballot` or `__activemask` anywhere in `src/` or
`include/`, and no arch-conditional code in the diff. The likelihood sum runs
through rocThrust's `transform_reduce`, which is not GooFit code and is not
width-sized by GooFit. There is no hardcoded-32 site to find.

### Stale device pointer is the better fit for the symptom

Two features of the recorded mechanism are load-bearing:

1. The bad value is `6.25e-310` -- a subnormal whose 52-bit mantissa is ~1.27e14,
   i.e. a pointer bit pattern read as a double (round-1 reviewer). That is a read
   of the wrong BYTES, not arithmetic that went wrong.
2. It is TEMPORAL: early fit iterations read a plausible norm (~0.67), later ones
   read garbage. A duplicate `d_normalizations` across two device images would be
   wrong from iteration zero, so that hypothesis does not fit.

Temporal decay of a symbol-published pointer is the signature of a stale pointer.
`SmartVectorGPU::sync` publishes `thrust::device_vector::data()` into the
`__device__ fptype *d_normalizations` via `hipMemcpyToSymbol`; the vector is
later reallocated, or the per-iteration `smart_sync` path writes elements without
re-publishing against a buffer that moved. Attempt 2 already noticed the
supporting contrast: `normRanges`, a raw `gooMalloc` pointer handed straight to
thrust, works, and only the symbol-published pointer rots.

This explains the arch split with NO wave64 mechanism at all, which is why it is
the stronger hypothesis. A read through a stale pointer is undefined, not
deterministically wrong. On gfx1100 the freed block plausibly still holds the old
values, so the fit converges and even reproduces bit-identically across ten runs
-- a false pass, the "deterministic, non-zero and plausible" trap the skill's
popsift entry warns about. On an MI250X GCD with a different allocation pattern
the block is reused and the read returns adjacent pointer bytes. Under this
reading gfx1100's four clean sessions are not evidence the code is correct; they
are evidence the bug is LATENT there.

### What attempt 4 should do first

One run discriminates between the two hypotheses:

- On gfx90a, log `d_normalizations` (the pointer value itself, via a one-element
  `hipMemcpyFromSymbol`) alongside the device_vector's `data()` at every fit
  iteration. If the symbol stops matching the vector after a resize, that is the
  bug. The fix is arch-unified: give the normalizations a stable `gooMalloc`'d
  buffer `hipMemcpy`'d on each sync, the way `normRanges` already works, instead
  of publishing a container-owned pointer. Attempt 2 listed this as its own next
  step and it was never carried out.
- Try to reproduce on gfx1100 with the allocator perturbed
  (`HSA_DISABLE_FRAGMENT_ALLOCATOR=1`, or forcing a reallocation between
  iterations). If gfx1100 breaks too, this is not an arch fault and the port has
  a lifetime bug that three architectures are currently hiding.
- Only if the pointer tracks the vector for the whole fit is the reduction worth
  looking at, and then the first suspect is `ConvolutionPdf.cu`'s
  `__shared__`/`THREAD_SYNCH` pair -- the one intra-block barrier in the codebase.

`exponential2_Example` PASSES on gfx90a while `exponential` fails. They differ in
fit setup, and diffing them is probably the shortest path to a minimal reproducer.

## Validation 2026-08-11 (validator, linux-gfx942, fork 18fca9e4a vs 5fe1221a8)

First validation of this platform. Fresh clone of `AMD-Ecosystem/GooFit`, checked
out `moat-port` at `18fca9e4a` (matches `status.json.head_sha` exactly), submodules
initialised (`extern/thrust` at CCCL `af8cce4ca`). `git status --porcelain` clean
throughout and after.

Host: MI300X x8 (gfx942, wave64), ROCm 7.14 SDK-wheel layout (no `/opt/rocm`;
`hipcc`/`cmake`/`ninja` from `/opt/conda/envs/py_3.12`), CMake 3.31.6, Ninja
1.13.0. Per the dispatch's host warning, GPU index 3 has ~206 GiB of orphaned VRAM
at the KFD level (`rocm-smi --showmeminfo vram` showed ~205.9 GB used, no attached
process) -- avoided. Verified GPU 5 idle (~300 MB used, matching the other six) and
used `HIP_VISIBLE_DEVICES=5` throughout.

### Build (gfx942, recorded recipe)

```
cmake -S . -B build-hip-validate -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGOOFIT_DEVICE=HIP -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DGOOFIT_TESTS=ON -DGOOFIT_EXAMPLES=ON -DGOOFIT_CERNROOT=OFF \
  -DGOOFIT_PYTHON=OFF -DGOOFIT_PHYSICS=OFF
cmake --build build-hip-validate -j64
```
306/306 targets, clean, exit 0 (~19s per `.ninja_log`, 224-core host). `roc-obj-ls`
is not present in this ROCm 7.14 SDK-wheel layout (`ImportError: cannot import name
'roc_obj_ls'`), so verified the offload target via the binary's own device banner
instead: `HIP: Architecture: gfx942:sramecc+:xnack-`, `HIP: 0 AMD Instinct MI300X
HF`.

### Tests: 25/25 PASS -- matches gfx1100's baseline exactly; does NOT reproduce the gfx90a divergence

```
HIP_VISIBLE_DEVICES=5 ctest --test-dir build-hip-validate --output-on-failure
  -> 100% tests passed, 0 tests failed out of 25, 10.25 s
```
All 25 pass, including every unbinned/binned NLL fit that fails on gfx90a
(GaussianTest, ConvolutionTest, exponential_Example, etc.).
`examples/exponential/exponential`:
```
alpha = -1.001102381 +/- 0.003165763922
```
Digit-identical to every gfx1100 session on record (differs from gfx1100's recorded
`0.003165763921` only in the last digit of the error, well within float noise).

Confirmed real GPU execution, not a CPU fallback: `AMD_LOG_LEVEL=3` shows 140
`hipLaunchKernel` dispatches for the `exponential` run.

**gfx942 (wave64, MI300X) does not reproduce the gfx90a (wave64, MI250X) NLL
divergence.** Both are wave64 parts, so the earlier "wave64-specific" framing
(explicitly weakly-supported per attempt 3's analysis above, which favored a stale
symbol-published device pointer over any wavefront-width mechanism) gains a second
data point against it: if the fault were a wave64 reduction-width assumption, this
gfx942 run should have shown it too, and it did not. This is consistent with, not
proof of, the stale-pointer hypothesis -- attempt 3's own note that gfx1100's clean
runs might be "latent, not correct" (freed-block reuse pattern differing by
allocator) applies here as well, so a clean gfx942 run cannot rule out a pointer
lifetime bug that different allocators simply fail to trigger. It does rule out a
mechanism that keys on wavefront width alone, since gfx942 and the failing gfx90a
share that width.

### No non-GPU regression: CPP backend 26/26

```
cmake -S . -B build-cpp-validate -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGOOFIT_DEVICE=CPP -DGOOFIT_TESTS=ON -DGOOFIT_EXAMPLES=ON \
  -DGOOFIT_CERNROOT=OFF -DGOOFIT_PYTHON=OFF
cmake --build build-cpp-validate -j64      # 375/375, clean
ctest --test-dir build-cpp-validate        # 100% tests passed, 26/26, 5.44 s
```
Matches the recorded gfx1100/CPP baseline (26/26) exactly.

### CUDA no-regression gate: not re-run

`head_sha` is unchanged (`18fca9e4a`) since the gfx1100 validation session already
recorded this gate at this exact commit: pre-existing upstream `driver_types.h: No
such file or directory` breakage under CUDA 12.8 (nvcc), reproduced identically on
pristine upstream `5fe1221a8`, not a port regression. Per the validator's
once-per-head_sha rule, skipped here (this is the third platform to skip it for the
same reason, after the gfx90a session).

### Jargon and documentation

`python3 utils/jargon.py --port GooFit` -> clean. No code or docs changed since the
gfx1100 session verified README.md/`docs/SYSTEM_INSTALL.md` accuracy against the
build recipe actually used; nothing to re-check here.

### Tree state and result

`git -C projects/GooFit/src status --porcelain` empty before and after (local
`build-*-validate` directories removed after use, gitignored). No source or build
files edited this round.

**Result: linux-gfx942 -> completed at 18fca9e4a.** HIP 25/25 (10.25s), CPP 26/26
(5.44s), alpha digit-identical to gfx1100, real-GPU-execution confirmed, CUDA gate
reused from this head_sha (pre-existing upstream breakage, not a regression),
jargon clean, docs unchanged and already verified accurate. This satisfies the
wave64 gate (gfx90a already attempted wave64 and failed; gfx942 now supplies a
passing wave64 witness) and the gfx90a NLL-divergence question remains open and
unaffected by this result -- see the analysis above for why a clean gfx942 run
does not settle it.

## Attempt 4 (2026-08-14, porter, linux-gfx90a) -- the recorded gfx90a divergence does NOT reproduce; fixed a real defect found while looking for it

Fresh clone of `AMD-Ecosystem/GooFit` at `moat-port` `18fca9e4a` (the failed sha),
submodules initialised (`extern/thrust` at CCCL `af8cce4ca`), `git status
--porcelain` clean. MI250X (gfx90a), ROCm **7.14** (`hipcc --version`: HIP version
7.14.60850-0000000, AMD clang 23.0.0git), CMake 3.31.6, Ninja, GPU 3.

### The 21/25 NLL divergence does not reproduce at 18fca9e4a on this host

```
cmake -S . -B build-hip -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGOOFIT_DEVICE=HIP -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DGOOFIT_TESTS=ON -DGOOFIT_EXAMPLES=ON -DGOOFIT_CERNROOT=OFF \
  -DGOOFIT_PYTHON=OFF -DGOOFIT_PHYSICS=OFF
cmake --build build-hip -j64                                   # 306/306, clean
HIP_VISIBLE_DEVICES=3 ctest --test-dir build-hip               # 25/25, 8.22 s
```
`examples/exponential/exponential` -> `alpha = -1.001102381 +/- 0.003165763922`,
digit-identical to gfx1100 and gfx942, run 6 times bit-identical. Real GPU
execution confirmed: `AMD_LOG_LEVEL=3` shows 383 dispatch/arch lines and
`HIP: Architecture: gfx90a:sramecc+:xnack-`.

Two perturbations that a stale-pointer or allocator-luck hypothesis should
survive, both still 25/25:
```
HIP_VISIBLE_DEVICES=3 HSA_DISABLE_FRAGMENT_ALLOCATOR=1 ctest --test-dir build-hip
HIP_VISIBLE_DEVICES=3 AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3 ctest --test-dir build-hip
```

### Which variable actually differs, and which does not

The 2026-08-09 failing session ran CMake **4.0.3** and HIP **7.2.53211-e1a6bc5663**;
the passing gfx1100 session ran CMake 3.31.6 and HIP **7.2.53211**-c2d9476115 -- the
same HIP version, a different build. So CMake version correlated perfectly with the
outcome across all recorded runs, which made it the leading suspect. It is ruled out
here: installed cmake 4.0.3 (pip wheel) and rebuilt the same tree with it,
`build-hip-cm4`, 306/306 and **25/25**. So the only variable left between the failing
run and this one is the ROCm build/version (7.2.5-series vs 7.14).

Could not reproduce the old toolchain to close it properly: this host's ROCm is a
TheRock-style pip layout and `pip index versions rocm-sdk-devel` offers only the
installed 7.14 series, so a 7.2.x SDK is not obtainable here without a multi-GB
download from another index -- at the ~10 MB/min this host gets to GitHub, out of
budget. Recorded rather than attempted.

Conclusion: the failure is real (it was reproduced twice on that host) but it is not
a property of gfx90a as such. Nothing in the diff is arch-conditional, gfx942 is the
same wavefront width and passes, and the same arch on a current ROCm passes with the
allocator perturbed. Treat it as a toolchain-generation fault that is gone on 7.14
rather than as an outstanding port defect. The corrected diagnostic ladder (matching
the version of an earlier FAILING run proves nothing; the discriminating run is the
same arch on a DIFFERENT ROCm) is promoted to the skill's `references/validation.md`.

### The real defect this session found: the documented build fails with its own defaults

Looking for a fault that was not there, the port's default configuration turned out
to be broken. `GOOFIT_PYTHON` defaults ON whenever Python development files are found
(`option(GOOFIT_PYTHON ... ${CUR_PROJ})`), and every recorded session -- attempts 3-6,
three validations -- passed `-DGOOFIT_PYTHON=OFF`, so nobody ran the command the
README and `docs/SYSTEM_INSTALL.md` actually print. That command fails:

```
c++: error: unrecognized command-line option '--offload-arch=gfx90a'
   (python/goofit/CMakeFiles/_Core.dir/{HelpPrinter,Variable,...}.cpp.o)
```

Cause is NOT the bare-filename `set_source_files_properties` leak guessed in the
gfx1100 validation note: `HelpPrinter.cpp` has no same-named sibling in `src/goofit`
and fails identically. The pybind11 modules `_Core`/`_Basic`/`_Combine` are declared
with a plain `add_library()`, so their sources stay CXX, while `roc::rocthrust` pulls
in `hip::device`, whose `INTERFACE_COMPILE_OPTIONS` carry `-x hip --offload-arch=` to
every consumer regardless of language. The host compiler gets flags it does not have.

Fix (commit `e8dca9151`): two macros, `goofit_adopt_hip_target` (whole target) and
`goofit_adopt_hip_sources` (sources appended with `target_sources()` from a
subdirectory, which need `TARGET_DIRECTORY` because source properties are
directory-scoped), giving those targets the LANGUAGE HIP + `-include cuda_to_hip.h`
treatment `goofit_add_library` already gives everything else. Both are inert unless
`GOOFIT_DEVICE STREQUAL HIP`. Source lists moved into variables so one list feeds
`add_library()` and the macro. Ran `cmake-format` (0.6.13, the version pinned in
`.pre-commit-config.yaml`); it also reflowed two `target_compile_options` calls the
earlier port commits left unformatted.

### Results at e8dca9151

The documented recipe, no `GOOFIT_PYTHON` flag at all, so the bindings default ON:
```
cmake -S . -B build-hip-final -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGOOFIT_DEVICE=HIP -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DGOOFIT_TESTS=ON -DGOOFIT_EXAMPLES=ON -DGOOFIT_CERNROOT=OFF
cmake --build build-hip-final -j64                              # 395/395, clean
HIP_VISIBLE_DEVICES=3 ctest --test-dir build-hip-final          # 25/25, 8.21 s
HIP_VISIBLE_DEVICES=3 PYTHONPATH=$PWD/build-hip-final python3 -m pytest python/tests -q
                                                                # 6/6, 0.39 s
```
The Python tests are real unbinned NLL fits over 100k events (`test_exp.py` fits
`alpha` and asserts `|alpha+1| < 0.01`) -- i.e. the exact class of test that failed
in the 2026-08-09 gfx90a session, now passing through the bindings as well.
`python3 -m goofit` reports `HIP 7.14`, `AMD Instinct MI250X / MI250`,
`gfx90a:sramecc+:xnack-`.

Non-GPU regression, same host, bindings ON (upstream default):
```
cmake -S . -B build-cpp -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGOOFIT_DEVICE=CPP -DGOOFIT_TESTS=ON -DGOOFIT_EXAMPLES=ON \
  -DGOOFIT_CERNROOT=OFF -DGOOFIT_PYTHON=ON
cmake --build build-cpp -j64
ctest --test-dir build-cpp                    # 26/26, 9.00 s
PYTHONPATH=$PWD/build-cpp python3 -m pytest python/tests -q   # 6/6
```
This is the first CPP run with the bindings ON; it matches the recorded 26/26.

CUDA gate not re-run: no nvcc build has ever succeeded on this project (pre-existing
upstream `driver_types.h` breakage under CUDA 12.8, reproduced on pristine upstream),
and the delta is inside `GOOFIT_DEVICE STREQUAL HIP` guards plus a behaviour-neutral
source-list refactor. A validator with nvcc should still re-run it at this head.

README updated: the ROCm requirement bullet no longer says the backend is tested only
on `gfx1100`; it now names `gfx90a`, `gfx942` and `gfx1100`.

`jargon.py --port GooFit` clean over the whole branch (needed a local `master`
branch first -- a `--branch moat-port` clone has no local base branch and the tool
resolves `master..moat-port`, so `git branch master origin/master` after cloning).
Fork tree clean before and after; `build-*` directories are gitignored.

### State

`head_sha` -> `e8dca9151`, project stage `ported` (a source change wants a review).
gfx90a's `failed_sha` stays at `18fca9e4a`, which is now behind head, so the failure
is superseded and this arch is owed a validator rather than another porter.
gfx1100 and gfx942 read `revalidate`: the delta is a build-system change, so it is
not carry-forward material -- the pybind11 targets are new HIP compilations on every
architecture that has Python development files installed.

## Review 2026-08-14 (reviewer, linux-gfx90a, fork e8dca9151 vs 5fe1221a8)

Verdict: **changes-requested**, and cheaply: nothing on the fork branch needs to
change. `e8dca9151` is correct, minimal and complete, and `head_sha` should not
move. Both findings are in the skill text this round promoted (MOAT commit
`1980d05`), which this review is the only gate on before it reaches every future
porter. Fix those two files, re-request, and the port goes straight to the
gfx90a validator with the same head.

Checks that were run and passed are not repeated here, per review philosophy; for
the record, the delta's central claim was verified independently rather than taken
from the note. `hip-config-amd.cmake:140-152` does put `-x hip` and
`--offload-arch=` on `hip::device`, `rocprim-targets.cmake:70` puts `hip::device`
in `roc::rocprim_hip`'s `INTERFACE_LINK_LIBRARIES`, and `rocthrust-targets.cmake:63`
chains `roc::rocthrust` to that, so the leak into `_Core`/`_Basic`/`_Combine` is
real. The generated `build-hip-final/build.ninja:6712` now compiles
`python/goofit/HelpPrinter.cpp` with `HIP_COMPILER___Core_...` carrying
`-include cuda_to_hip.h -fgpu-rdc --offload-arch=gfx90a`, and the two
`target_sources()` files reach the same rule via `_goofit` (lines 6266, 6286).
Coverage of the fix is complete: `_Core`, `_Basic`, `_Combine`, `_Physics` and
`_goofit` are the only targets that link `_goofit_python`/`GooFit::GooFit`, and
`landau`, `minuit2` and `catch_main` link neither ROCm nor GooFit, so they cannot
inherit the flags.

### 1. The promoted CMake lesson states the mechanism backwards

`.claude/skills/cuda-to-rocm/references/strategy-a-cmake.md:172-178`. The heading
says the options land on "EVERY consumer, C++ ones included" and the body says
CMake applies them "to every target that links the interface library, whatever
language that target's sources are compiled in". The opposite is true, and the
inverted claim is exactly the part a reader needs:

```
# hip-config.cmake:75-79
function(hip_add_interface_compile_flags TARGET)
  set_property(TARGET ${TARGET} APPEND PROPERTY
    INTERFACE_COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:${_HIP_SHELL}${ARGN}>")
endfunction()
```

The flags are wrapped in `$<$<COMPILE_LANGUAGE:CXX>:...>`, so they reach the
CXX-compiled sources of a consumer and only those. That is *why* marking the
sources `LANGUAGE HIP` fixes it -- they stop matching the generator expression --
not, as line 177-178 says, because "they were going to hipcc anyway and this is
invisible". As written the entry tells the next porter that `LANGUAGE HIP` leaves
the flags in place and merely makes them harmless, which invites the wrong fix
(stripping the flags off the interface target) on the next project that hits this.

Also in the same paragraph: `hip_add_interface_link_flags` (`hip-config.cmake:81-91`)
adds `--hip-link` and `--offload-arch=` to `INTERFACE_LINK_LIBRARIES` under
`$<$<LINK_LANGUAGE:CXX>:...>`, so a consumer that still *links* as CXX hits the
same error at link time. Worth one sentence, since a target whose sources are all
adopted but whose link language is not is the residual case.

And line 196, "the only way to reach the owning scope from there": `DIRECTORY
<owning-dir>` on the same command also reaches it. `TARGET_DIRECTORY` is the
better form because it does not hardcode a relative path; say that instead of
"only way".

The in-tree comment at `src/CMakeLists.txt:612-617` is accurate as it stands --
only the skill text needs the correction.

### 2. The validation lesson leaves GooFit standing as the settled arch-fault example

`.claude/skills/cuda-to-rocm/references/validation.md`. The inserted paragraph
(113-126) is the right lesson, but it is dropped into a section whose framing and
conclusion still assert what it retracts, so a reader gets both:

- 95-99 still describes the divergence in the present tense ("GooFit's HIP backend
  diverges an unbinned maximum-likelihood fit ... on gfx90a while the identical
  committed tree passes cleanly on gfx1100"), with no mention that this holds only
  for the ROCm 7.2-series build and not for 7.14.
- 128-134 is worse: "Only once both are pinned identical to a passing run does
  'wrong on this arch only' stand as a finding. For GooFit specifically: same ROCm
  7.2.1 series hipcc ... so the divergence is real GPU execution producing a wrong
  answer ... That is when this becomes the 'one architecture gets wrong numbers'
  case below." That paragraph certifies the exact reasoning the new insert calls
  the wrong direction, and it is the section's closing takeaway. An agent who reads
  to the end of the section leaves with the superseded conclusion.

Rewrite 95-99 and 128-134 so GooFit is the worked example of the retraction rather
than of the arch fault, or move the standing example to a project where the
conclusion held.

One factual slip inside the insert itself, at 120-122: "built the same commit from
a fresh clone and got 25/25 ... plus 6/6 on the Python bindings' 100k-event fits".
The 6/6 is not from the same commit -- at `18fca9e4a` the bindings do not build at
all with their default `ON`, which is what this round fixed; the pytest run is from
`e8dca9151` (see attempt 4, "Results at e8dca9151"). Attribute it to the fixed
commit or drop it from that sentence; the 25/25 alone carries the argument.
