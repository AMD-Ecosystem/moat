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
