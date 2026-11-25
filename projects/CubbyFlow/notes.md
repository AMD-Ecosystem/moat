# CubbyFlow notes

## Port complete 2026-06-11 (linux-gfx90a)

Strategy A (compat header + modern CMake HIP language). ROCm 7.2.1, gfx90a.
The earlier "blocked: __CUDA_ARCH__ return-type selection" determination was
wrong; the real fix is below.

### The __CUDA_ARCH__ host/device return-type blocker (resolved)

CUDAArrayBase / CUDAStdVector select between a device accessor (returns T&) and
a host accessor (returns a copy-back wrapper / value) via `#ifdef __CUDA_ARCH__`.
This is NOT impossible on HIP; the prior attempt mis-diagnosed it.

Root cause of the failure: nvcc does NOT parse `__host__` function BODIES during
its device pass, so a host function calling `arr[i]` is never type-checked in the
device pass. clang (HIP) parses ALL bodies in BOTH passes and defers the
cross-space-call diagnostic. With the original `#ifdef __CUDA_ARCH__` the host
overloads are entirely ABSENT in the device pass, so when clang parses a `__host__`
function body during the device pass, `operator[]` resolves to the only visible
(device) overload and errors "call to __device__ function from __host__ function".
Confirmed: the error fires in the DEVICE-ONLY compile, not the host pass.

Fix (CUDAArrayBase.hpp/-Impl, CUDAStdVector.hpp/-Impl): under `__HIP__`, declare
and define BOTH the device and host accessor overloads at once, distinguished by
`__host__`/`__device__` attributes, and let clang resolve by call context in each
pass. CUDA path kept byte-identical with `#elif defined(__CUDA_ARCH__) / #else`.
Pattern used everywhere: `#if defined(__HIP__) || defined(__CUDA_ARCH__)` (device)
paired with `#if defined(__HIP__) || !defined(__CUDA_ARCH__)` (host). The shim
also defines `__CUDA_ARCH__` only in the HIP device pass (guarded by
`__HIP_DEVICE_COMPILE__`) so the remaining intra-body `#ifdef __CUDA_ARCH__` device
selections resolve correctly per pass.

### Other faults fixed

- __CUDACC__ vs __HIPCC__: project gates kernels/attribute macros on `__CUDACC__`
  (hipcc does not define it). Do NOT `#define __CUDACC__` -- rocThrust keys its
  backend on it and would pick the CUDA backend (missing CUB header). Instead
  extend each guard to `defined(__CUDACC__) || defined(__HIPCC__)` (Macros.hpp,
  CUDAArray-Impl.hpp, CUDAAlgorithms.hpp). Also force `-DTHRUST_DEVICE_SYSTEM=5`
  (HIP) belt-and-suspenders.
- -Impl.hpp definitions omitted the `__host__ __device__` their declarations
  carry; nvcc merges, clang requires the match. Added CUBBYFLOW_CUDA_HOST_DEVICE
  to every shared definition in CUDAStdArray-Impl, CUDAArrayBase-Impl,
  CUDAArrayView-Impl, CUDASPHKernels2/3-Impl.
- HIP nodiscard: hipError_t and hipDeviceReset are nodiscard (cudaError_t /
  cudaDeviceReset are not). Bound the result to a typed local in
  _CUBBYFLOW_CUDA_CHECK and `static_cast<void>` the reset; one test discarded
  cudaDeviceSynchronize -> cast to void.
- HIP vector types provide arithmetic/compound/equality operators for floatN, so
  CUDAUtils.hpp's CUDA-side floatN operators are ambiguous -> guard them out with
  `#if !defined(__HIP__)`; kept the named helpers (Dot/Length/To*).
- clang -Werror flags nvcc tolerates: -Wno-class-memaccess is GNU-only (scope to
  C/CXX language); -Wno-reorder-ctor / -Wno-unused-private-field / -Wno-unused-
  variable must come AFTER -Werror (clang honours a later -Wno-* over -Werror),
  so appended as `$<$<COMPILE_LANGUAGE:HIP>:...>` at the end of
  DEFAULT_COMPILE_OPTIONS, not in CMAKE_HIP_FLAGS.
- cuda_runtime.h includes -> `#if defined(__HIP__) #include <Core/CUDA/cuda_to_hip.h>
  #else #include <cuda_runtime.h> #endif` (9 sites).
- CMake: USE_HIP option, enable_language(HIP), CMAKE_HIP_ARCHITECTURES default
  gfx90a, force-include the shim via CMAKE_HIP_FLAGS `-include`, mark the CUDA
  directory's .cu AND its .cpp (they touch device types/thrust) as LANGUAGE HIP.
  Tests/CUDATests and Examples/CUDASPHSim: same .cu->HIP marking; the example's
  source glob was CUDA-only -> now `USE_CUDA OR USE_HIP`. Python bindings stay
  excluded on GPU builds (USE_GPU).

### Build

```
cd projects/CubbyFlow/src
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build -j$(nproc)
```

`git submodule update --init --recursive` first. CMake wants clang directly for
HIP, NOT the hipcc wrapper (CMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++).

### Validation (gfx90a, real GPU)

- CUDATests: 35/35 cases, 3168/3168 assertions PASS (CUDA array/vector/stdarray,
  particle system data, particle system solver, point hash grid searcher; the
  hash-grid test cross-checks Keys/Start/End/SortedIndices and nearby-point
  callbacks against the CPU searcher).
- UnitTests (CPU regression): 722/722 PASS.
- CUDASPHSim example: runs the full GPU pipeline end-to-end (512 particles), all
  kernels execute, densities verified sane (~350 vs target 1000).
- WCSPH solver with a container set: 5 frames on GPU, all positions finite,
  particles settle to the floor (y-range [0, 0.205]) -- confirms neighbor search,
  density, EOS pressure, viscosity, integration, and boundary collision are all
  numerically correct on ROCm/gfx90a.

### Known non-issue: CUDASPHSim x,y = FLT_MAX in output

The shipped CUDASPHSim example never calls SetContainer(), so the solver's
m_container is the default-empty BoundingBox3F (lowerCorner=+FLT_MAX,
upperCorner=-FLT_MAX, from BoundingBox::Reset). The integration kernels clamp
x.x/x.y to [lower, upper]; with the empty box every finite coordinate is
`< lowerCorner` so it gets set to +FLT_MAX. This is pure CPU-side float math in
unmodified upstream code, identical on CUDA and HIP (the solver .cu files and
Geometry/BoundingBox are untouched by the port) -- NOT a port defect, an upstream
example-config detail. PCISPH's integration clamps only x,y (no z), so z shows
correct free-fall while x,y saturate; setting a container makes all axes finite
and physical (verified above). Not deferred/blocking.

### Files changed

Compat header: Includes/Core/CUDA/cuda_to_hip.h (new).
Host/device overload restructure: CUDAArrayBase.hpp, CUDAArrayBase-Impl.hpp,
CUDAStdVector.hpp, CUDAStdVector-Impl.hpp.
Attribute matching: CUDAStdArray-Impl.hpp, CUDAArrayView-Impl.hpp,
CUDASPHKernels2-Impl.hpp, CUDASPHKernels3-Impl.hpp.
Macros/guards: Includes/Core/Utils/Macros.hpp, CUDAAlgorithms.hpp,
CUDAArray-Impl.hpp, CUDAUtils.hpp, and the 9 cuda_runtime.h include sites.
CMake: CMakeLists.txt, Sources/Core/CMakeLists.txt, Tests/CUDATests/CMakeLists.txt,
Examples/CUDASPHSim/CMakeLists.txt, Builds/CMake/CompileOptions.cmake.
Test cast: Tests/CUDATests/CUDAArray2Tests.cu.

## Validation 2026-06-12 (linux-gfx90a, validator)

Platform: linux-gfx90a (AMD Instinct MI250X, gfx90a). HIP_VISIBLE_DEVICES=0,1.
Fork HEAD: 83ee5063a0 ([ROCm] Add AMD ROCm/HIP support for the CUDA SPH solvers).
Source tree clean (no uncommitted tracked files).

### Build

```
cd projects/CubbyFlow/src
git submodule update --init --recursive
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build -j$(nproc)
```

Build: PASS (all targets built cleanly, 100%).

### GPU tests

```
./build/bin/CUDATests
```

Result: 35/35 test cases, 3168/3168 assertions PASS.

```
./build/bin/CUDASPHSim -f 5
```

Result: 5 frames written, 13824 particles, no crash (GPU end-to-end PASS).

### CPU regression tests

```
./build/bin/UnitTests
```

Result: 722/722 PASS (no non-GPU regression).

### CUDA no-regression gate

Compiled the CUDA path (`USE_CUDA=ON`) with nvcc 12.8 from
/opt/conda/envs/cuda-12.8 and host gcc 13, `-DCMAKE_CUDA_ARCHITECTURES=80`.
Required adding `/opt/conda/envs/cuda-12.8/nvvm/bin` to PATH for `cicc`.
Build: PASS (warnings only, no errors). No type aliases or deleted code caused
CUDA regressions; the port's `#if defined(__HIP__)` guards are cleanly
CUDA-inert.

### Verdict: PASS -> completed

## Review 2026-06-12 (linux-gfx90a, reviewer)

review-passed. Read-only review of the moat-port branch (HEAD 83ee5063 vs base
5b786fee61) with the /pr-review skill, ROCm-fault-class aware. No blocking
problems found; no changes requested.

Non-blocking observation (not a defect, recorded for completeness): under
`#if defined(__HIP__) || defined(__CUDA_ARCH__)` the CUDAStdVector device
`operator[]` declaration (CUDAStdVector.hpp) and definition
(CUDAStdVector-Impl.hpp) gain an explicit `__device__` that the base lacked
(base was unannotated, body calls `__device__ At`). This is a strict
generalization: the block is only visible in nvcc's device pass, so it makes
the existing device-context-only function explicit rather than changing CUDA
numerics or runtime behavior. Acceptable under the additive-and-guarded
"strict generalization" clause; flagged only so a future CUDA-path diff review
knows it is intentional.

Verified: Strategy A correct for a pure-CMake project; single compat header
(cuda_to_hip.h) is a no-op on NVIDIA (guarded by CUBBYFLOW_USE_CUDA && __HIP__,
and CUBBYFLOW_USE_CUDA is defined on both backends). CUDA path preserved
byte-identical via `#elif defined(__CUDA_ARCH__)` / `#else` in CUDAArrayBase.hpp
and the StdVector headers. No fault-class hazards: no warp intrinsics, no
hardcoded 32/warpSize, no atomics, no __shared__, no textures/surfaces, no
streams/events -- kernels are warp-size-agnostic 1D decomposition, so wave64 is
safe and a future wave32 follower needs no per-arch delta. No kernel body logic
changed (only attribute/include edits plus one test `static_cast<void>`).
`.cu` marked LANGUAGE HIP (not renamed); CUDATests main.cpp stays host C++
(touches no device types). Build gates HIP behind USE_HIP (default OFF),
enable_language(HIP), arch default gfx90a not hardcoded over
-DCMAKE_HIP_ARCHITECTURES. THRUST_DEVICE_SYSTEM=5 pins rocThrust HIP backend;
__CUDACC__ correctly NOT defined. __align__ is provided by HIP runtime header.
Commit hygiene clean: `[ROCm]` title 56 chars, names Claude, Test Plan present,
no noreply trailer, the author's own public account, no MOAT jargon in the diff, no
AMD-internal account references. GPU validation already recorded (CUDATests
35/35, UnitTests 722/722, WCSPH end-to-end) -- the validator stage re-runs it.

## Validation 2026-06-12 (linux-gfx1100, validator)

Platform: linux-gfx1100 (AMD Radeon Pro W7800, gfx1100, RDNA3). HIP_VISIBLE_DEVICES=0.
Fork HEAD: 83ee5063a0 ([ROCm] Add AMD ROCm/HIP support for the CUDA SPH solvers).
Source tree clean (no uncommitted tracked files).

### Build

```
cd projects/CubbyFlow/src
git submodule update --init --recursive
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build -j$(nproc)
```

Build: PASS (all targets built cleanly).

### GPU tests

```
HIP_VISIBLE_DEVICES=0 ./build/bin/CUDATests
```

Result: 35/35 test cases, 3168/3168 assertions PASS.

```
HIP_VISIBLE_DEVICES=0 ./build/bin/CUDASPHSim -f 5
```

Result: 5 frames written, 13824 particles, no crash (GPU end-to-end PASS).

### CPU regression tests

```
HIP_VISIBLE_DEVICES=0 ./build/bin/UnitTests
```

Result: 722/722 PASS (no non-GPU regression).

### Verdict: PASS -> completed

## Validation 2026-06-12 (windows-gfx1201, RX 9070 XT, RDNA4)

Platform: windows-gfx1201 (AMD Radeon RX 9070 XT, gfx1201, RDNA4).
HIP_VISIBLE_DEVICES=1 (gfx1101 HARD-WEDGED at device 0; always pin =1 for gfx1201).
Fork HEAD: 3700598eff ([ROCm] Suppress -Wnontrivial-memcall on Windows+Clang).
Source tree clean (no uncommitted tracked files).
Toolchain: TheRock clang 23.0.0 (all-clang, MSVC-ABI target), Ninja.

### Windows build fix

Clang on Windows fires -Werror,-Wnontrivial-memcall on memset/memcpy calls in
the bundled Flatbuffers-generated header (pre-existing upstream code, not the HIP
port). Added a Clang+WIN32-guarded -Wno-nontrivial-memcall in CompileOptions.cmake.
This change is unreachable on Linux (WIN32 is false there), so Linux device code
is byte-identical to the gfx90a/gfx1100 validated state.
Committed as a new commit on top of the HIP port commit (3700598eff).

### Build

```
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
cd projects/CubbyFlow/src
git submodule update --init --recursive
cmake -B build -S . -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_HIP_COMPILER=${ROCM}/lib/llvm/bin/clang++.exe \
  -DCMAKE_CXX_COMPILER=${ROCM}/lib/llvm/bin/clang++.exe \
  -DCMAKE_C_COMPILER=${ROCM}/lib/llvm/bin/clang.exe \
  -DCMAKE_PREFIX_PATH=${ROCM} \
  -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build --target CubbyFlow CUDATests UnitTests CUDASPHSim -j64
```

Copy TheRock runtime DLLs into build/bin/ to override System32's Adrenalin amdhip64:
  amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll, hiprtc-builtins0714.dll
from _rocm_sdk_core/bin.

Build: PASS (334 targets, all built cleanly).

### GPU tests

```
HIP_VISIBLE_DEVICES=1 ./build/bin/CUDATests.exe
```

Result: 35/35 test cases, 3168/3168 assertions PASS.

```
HIP_VISIBLE_DEVICES=1 ./build/bin/CUDASPHSim.exe -f 5
```

Result: 5 frames written, 13824 particles, no crash (GPU end-to-end PASS).

### CPU regression tests

```
./build/bin/UnitTests.exe
```

Result: 722/722 PASS (no non-GPU regression).

### Verdict: PASS -> completed

### Linux revalidation note

The Windows-only CompileOptions.cmake fix advances fork HEAD from 83ee5063 to
3700598eff, flipping linux-gfx90a and linux-gfx1100 to revalidate. The new code
is guarded by WIN32, making the Linux device code byte-identical. Linux validators
should binary-equiv carry-forward (codeobj_diff IDENTICAL -> carry-forward).

## Validation 2026-06-19 (windows-gfx1101, Radeon PRO V710, RDNA3)

Platform: windows-gfx1101 (AMD Radeon PRO V710, gfx1101, RDNA3).
HIP_VISIBLE_DEVICES=1 (verified: hipInfo reports AMD Radeon PRO V710 at mask 1).
Fork HEAD: 9c901552588ecba6bf28d609a3e07aee1689f90e ([ROCm] Sync after the second resize-with-fill in CUDAArray2 test).
Source tree clean (no uncommitted tracked files).
Toolchain: TheRock clang 23.0.0 (all-clang, MSVC-ABI target), Ninja.

### Build

Used a fresh build_gfx1101/ directory (separate from the existing gfx1201 build/).

```
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
cmake -B build_gfx1101 -S . -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
  -DCMAKE_HIP_COMPILER=${ROCM}/lib/llvm/bin/clang++.exe \
  -DCMAKE_CXX_COMPILER=${ROCM}/lib/llvm/bin/clang++.exe \
  -DCMAKE_C_COMPILER=${ROCM}/lib/llvm/bin/clang.exe \
  -DCMAKE_PREFIX_PATH=${ROCM} \
  -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build_gfx1101 --target CubbyFlow CUDATests UnitTests CUDASPHSim -j64
```

Copy TheRock runtime DLLs into build_gfx1101/bin/ to override System32's Adrenalin amdhip64:
  amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll, hiprtc-builtins0714.dll
from _rocm_sdk_core/bin.

Build: PASS (334/334 targets, all built cleanly).

### GPU tests

```
HIP_VISIBLE_DEVICES=1 ./build_gfx1101/bin/CUDATests.exe
```

Result: 35/35 test cases, 3170/3170 assertions PASS.
(3170 vs 3168 on gfx1201: the head commit 9c90155 adds a cudaDeviceSynchronize assertion, consistent with carry-forward note.)

```
HIP_VISIBLE_DEVICES=1 ./build_gfx1101/bin/CUDASPHSim.exe -f 5
```

Result: 5 frames written, 13824 particles, no crash (GPU end-to-end PASS).

### CPU regression tests

```
./build_gfx1101/bin/UnitTests.exe
```

Result: 722/722 PASS (no non-GPU regression).

### Verdict: PASS -> completed

## Fix round 2026-08-20 (linux-gfx1100, porter): merge upstream main

PR #145 is OPEN and was `CONFLICTING` / `mergeStateStatus: DIRTY`. Upstream moved
far ahead of the PR base (~422 files): C++17 -> C++23 project-wide, vendored git
submodules under `Libraries/` replaced by a vcpkg manifest (`vcpkg.json` +
`vcpkg-configuration.json` with a `Libraries` overlay port for cnpy), a rewritten
README/ARCHITECTURE.md, and a new `CUBBYFLOW_REQUIRES` concepts macro in
`Includes/Core/Utils/Macros.hpp`.

Thread traffic on #145 is bots only -- no maintainer comment. coderabbitai posted
a review-stack comment and eventually approved; codacy-production reports "Not up
to standards" with 9 medium ErrorProne issues (Codacy's own static analysis on the
merged diff, not a maintainer request). Not chased in this round.

### Staged on moat-fix-145 (branched from published tip 62f4604c)

- `e506244969` `[ROCm] Merge upstream main into the AMD/HIP branch`
  (merge of `upstream/main` 8f774f6a4f)
- `2c51f93ecd` `[ROCm] Correct the AMD GPU docs and a stale build comment`

### The one conflict

Root `CMakeLists.txt`. Upstream deleted `add_subdirectory(Libraries/pybind11)`
(pybind11 now comes from `find_package(pybind11 CONFIG REQUIRED)` via vcpkg) right
next to the block the port edits. Resolution = upstream's new structure with the
port delta reapplied verbatim: the `USE_HIP` option + `enable_language(HIP)` block,
`USE_CUDA OR USE_HIP` on Tests/CUDATests and Examples/CUDASPHSim, and
`NOT (USE_CUDA OR USE_HIP)` on the Python bindings. Verified by diffing the
resolution against `upstream/main:CMakeLists.txt` -- the only delta is the port's.

### Auto-merged seams, verified by hand

- `Builds/CMake/CompileOptions.cmake`: upstream's `CXX_STANDARD 23` /
  `CXX_STANDARD_REQUIRED ON` and its removal of `-std=c++1z` sit next to the port's
  HIP-only warning suppressions; those stay `$<$<COMPILE_LANGUAGE:HIP>:...>`-scoped.
- `Sources/Core/CMakeLists.txt`: upstream's tinyobjloader/flatbuffers wiring merged
  with the port's `USE_HIP` glob that marks `CUDA/*.cu` and `CUDA/*.cpp` `LANGUAGE HIP`.
- `Tests/CUDATests/CMakeLists.txt`: upstream's doctest linkage plus the port's
  `LANGUAGE HIP` marking, both present.
- `Includes/Core/Utils/Macros.hpp`: upstream's `CUBBYFLOW_REQUIRES` and the port's
  `__HIPCC__` additions to the host/device and alignment guards, both present.
- `Includes/Core/CUDA/CUDAPointHashGridSearcher{2,3}.hpp`: upstream's member-init
  cleanups plus the port's include swap to the compatibility header (the
  copy/move members there are upstream's, not this branch's -- corrected per
  the 2026-08-20 review).

### C++23 and CUBBYFLOW_REQUIRES under HIP: no change needed

Anticipated hazard that did not materialize. `CUBBYFLOW_REQUIRES` is guarded
`#if !defined(__CUDACC__) && defined(__cpp_concepts) && __cpp_concepts >= 201907L`,
and `cuda_to_hip.h` deliberately does NOT define `__CUDACC__` (that would flip
rocThrust to its CUDA backend). So the `requires(...)` clauses DO reach hipcc.
They compile clean: CMake propagates `CXX_STANDARD 23` to the HIP language, HIP
TUs get `-std=c++23`, and ROCm 7.2 clang 22.0.0 accepts constrained templates in
device compilation. Verified in `build/compile_commands.json`:

```
/opt/rocm/llvm/bin/clang++ ... -std=c++23 --offload-arch=gfx1100 -Werror ...
```

The macro's guard was left exactly as upstream wrote it. Do not add `__HIPCC__` to
it -- that would silently disable concepts on the AMD path for no reason.

### Build procedure (UPDATED -- supersedes the pre-vcpkg recipe above)

`git submodule update --init --recursive` is obsolete; `.gitmodules` is gone.
A bootstrapped vcpkg checkout and `VCPKG_ROOT` are now required, and CMake
installs `vcpkg.json`'s dependencies during configure.

```
git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh -disableMetrics
sudo apt-get install -y autoconf autoconf-archive automake libtool pkg-config \
    python3-dev zip unzip curl
export VCPKG_ROOT=~/vcpkg

cd projects/CubbyFlow/src
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build -j$(nproc)
```

`autoconf-archive` is REQUIRED and is not in the project's documented apt list:
vcpkg pulls python3 (for pybind11), whose libb2 dependency fails `autoreconf` with
`BUILD_FAILED` without it. First configure ~137 s (vcpkg builds python3, pybind11,
gtest, benchmark, doctest, flatbuffers 1.7.1, pystring, tinyobjloader[double],
cnpy from the overlay port); afterwards it is cached.

TBB is not installed on this host, so CMake falls back to
`-DCUBBYFLOW_TASKING_OPENMP` (OpenMP 4.5). Not a port concern; the same fallback
applies to the CUDA path.

CMake still wants clang directly for HIP, NOT the hipcc wrapper
(`-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++`).

### Build result (linux-gfx1100, Radeon Pro W7800, ROCm 7.2.3, clang 22.0.0)

PASS. All 15 binaries built. Zero errors AND zero warnings under `-Werror` --
none of the port's HIP warning suppressions became stale, and the C++23 move
introduced no new hipcc diagnostics.

### GPU tests at the staging tip 2c51f93ecd

```
HIP_VISIBLE_DEVICES=0 ./build/bin/CUDATests
```

35/35 test cases, 3170/3170 assertions PASS.

```
cd <scratch>; HIP_VISIBLE_DEVICES=0 <src>/build/bin/CUDASPHSim -f 5
```

5 frames written, 13824 particles, exit 0 (GPU end-to-end PASS). Matches the
2026-06-12 gfx1100 validation exactly.

### CPU regression tests

```
HIP_VISIBLE_DEVICES=0 ./build/bin/UnitTests
```

814/814 PASS in 168 suites. (Was 722 before the merge; upstream added tests --
MPM system data / SnowMPMSolver among them. No failures.)

### CUDA no-regression gate

nvcc 12.8 from /opt/conda/envs/cuda-12.8, host gcc 13, `CMAKE_CUDA_ARCHITECTURES=80`.
The conda toolkit does not have the legacy-FindCUDA layout, so
`CUDA_TOOLKIT_ROOT_DIR` must point at the `targets/x86_64-linux` subtree or
`find_package(CUDA)` reports `missing: CUDA_INCLUDE_DIRS` and silently turns
`USE_CUDA` back OFF (configure still succeeds -- check the `Using CUDA:` line):

```
export VCPKG_ROOT=~/vcpkg
export PATH=/opt/conda/envs/cuda-12.8/bin:/opt/conda/envs/cuda-12.8/nvvm/bin:$PATH
cmake -B build-cuda -S . -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCUDA_TOOLKIT_ROOT_DIR=/opt/conda/envs/cuda-12.8/targets/x86_64-linux \
  -DCUDA_TOOLKIT_INCLUDE=/opt/conda/envs/cuda-12.8/targets/x86_64-linux/include \
  -DCUDA_CUDART_LIBRARY=/opt/conda/envs/cuda-12.8/lib/libcudart.so \
  -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build-cuda -j32 --target CubbyFlow
cmake --build build-cuda -j32 --target CUDATests CUDASPHSim
```

Build: PASS, no errors. The port's `#if defined(__HIP__)` guards remain
CUDA-inert across upstream's C++23 move. No GPU run (no NVIDIA GPU on this host).

### Follow-up commit 2c51f93ecd: three text-only corrections

1. `Documents/Install.md` still claimed `CMAKE_HIP_ARCHITECTURES` "defaults to
   gfx90a". Commit 62f4604c removed that default in favor of CMake's host-GPU
   detection and did not update the doc. Fixed.
2. Upstream's new README carries a Quick Start whose prerequisite list names a
   CUDA toolkit for the optional GPU backend. Added the parallel one-line ROCm
   prerequisite there. Deliberately NOT a build block: the README defers the
   step-by-step GPU build to `Documents/Install.md`, which already has the
   `USE_HIP` section.
3. The comment above the Thrust backend pin in `CMakeLists.txt` claimed the compat
   header defines `__CUDACC__`. It does not (and `cuda_to_hip.h`'s own comment
   explains why it must not). Reworded without the false premise; the pin stays.

### jargon.py

`--port CubbyFlow` clean. `--commits moat-port..moat-fix-145` clean.
`--diff moat-port..moat-fix-145` reports 2 hits, BOTH false positives: the strings
`MOaT` and `OsrB` occur inside the base64 payload of `Medias/Logos/Logo.svg`,
added by upstream commit a1b680f3de ("docs: update project logo", Chris Ohk). Not
our text. A diff-range jargon scan over a merge sees all of upstream's delta;
check `--commits` and `--port` for our own text.

### NOT DONE -- push is blocked on a token scope

The merge carries upstream's `.github/workflows/*` changes and this host's gh
token lacks the `workflow` scope, so pushing `moat-fix-145` is known-blocked.
The branch is committed LOCALLY at `2c51f93ecd` and is unpushed; `head_sha`
therefore still reads 62f4604c and `advance-head` was deliberately NOT run.

Remaining steps, in order, after `gh auth refresh -s workflow`:

1. `git -C projects/CubbyFlow/src push origin moat-fix-145`
2. `python3 utils/moatlib.py advance-head CubbyFlow 2c51f93ecd...` (full sha
   `2c51f93ecdda50b320c2190f723ef26dd02d008e`) -- flips the four completed
   platforms to `revalidate`
3. delta review of the round
4. revalidation on the required gates (wave64, wave32, windows)
5. fix review PR (`upstream.py --fix-review`), person approves, then
   `upstream.py --merge-fix --apply`

## Review 2026-08-20 (linux-gfx1100, reviewer): fix round delta

changes-requested. Scope: `moat-port..moat-fix-145` (e506244969 merge of
upstream/main 8f774f6a4f, 2c51f93ecd text corrections), reviewed locally with
the /pr-review skill. Both problems are in upstream-visible TEXT, both are
cheap to fix now because the branch is still unpushed, and both become frozen
once `moat-port` is fast-forwarded into PR #145.

### 1. Merge commit body claims a change this branch never made

`e506244969` body, last "seams worth calling out" bullet:

> Includes/Core/CUDA/CUDAPointHashGridSearcher{2,3}.hpp: main's
> member-initializer cleanups landed alongside the explicit copy/move members
> this branch added.

This branch added no copy/move members to those files. Its entire delta there,
then and now, is the `cuda_runtime.h` -> `cuda_to_hip.h` include swap
(`Includes/Core/CUDA/CUDAPointHashGridSearcher2.hpp:19-23`,
`CUDAPointHashGridSearcher3.hpp:19-23`). The copy/move declarations at
`CUDAPointHashGridSearcher2.hpp:115-129` are upstream's and predate the port
(present at 5b786fee61). Checks: `git diff 5b786fee61...moat-port` and
`git diff upstream/main...moat-fix-145` on both files show only the include
hunk, and `git diff upstream/main...moat-fix-145 | grep '^+.*= \(default\|delete\)'`
is empty for the whole branch. Upstream's delta on those files is default
member initializers only (`m_gridSpacing = 1.0f`, `make_uint2(1, 1)`,
`uint1 m_dummy{}`).

Fix: drop or correct that bullet (the honest version is "main's member-
initializer cleanups landed next to this branch's include swap"), and fix the
matching sentence in this file's "Auto-merged seams, verified by hand" list
(notes.md, `## Fix round 2026-08-20`).

### 2. The replacement Thrust-pin comment names the wrong trigger

`CMakeLists.txt:88-90` (the comment 2c51f93ecd rewrote):

```
# rocThrust auto-detects its backend, and a build that defines
# CUBBYFLOW_USE_CUDA can end up on its CUDA backend, which then includes a
# CUDA-only CUB header. Pin Thrust to its HIP backend explicitly.
```

rocThrust never inspects `CUBBYFLOW_USE_CUDA`. It picks its backend in
`thrust/detail/config/compiler.h:111` (`#if defined(__CUDACC__) ... NVCC`) and
`thrust/detail/config/device_system.h:29-36` (HIP when the device compiler is
HIP, CUDA otherwise, i.e. also for a plain g++ TU). The old comment was wrong
about the shim defining `__CUDACC__`; the replacement swaps that for a
different non-mechanism, in a commit whose whole point was comment accuracy.

For the record, in this tree the pin is purely defensive: every TU that reaches
a thrust header (`Sources/Core/CUDA/*.cu`, `CUDAPointHashGridSearcher{2,3}-Impl.hpp`
via `Sources/Core/CUDA/*.cpp`) is compiled clang + `__HIP__`, and the three
non-HIP TUs under the CUDA directories (`Tests/CUDATests/main.cpp`,
`Examples/CUDASPHSim/{main,SPHSimExample}.cpp`) include no thrust.

Fix: state the real trigger, e.g. "rocThrust selects its backend from the
compiler it sees (`__CUDACC__`, else the host compiler); pin it to HIP so any
translation unit that reaches a thrust header cannot land on the CUDA backend
and its CUDA-only CUB header."

### Verified clean (no action)

Merge fidelity: `git merge-base upstream/main moat-fix-145` == `upstream/main`
(8f774f6a4f) and `git diff upstream/main...moat-fix-145 --name-status` is 28 M
plus `A Includes/Core/CUDA/cuda_to_hip.h` -- no upstream file lost, nothing
extra added, no conflict markers anywhere in tracked files. Diffing that delta
against the pre-merge `git diff 5b786fee61...moat-port` leaves exactly four
content differences: the reworded Thrust comment, the Install.md gfx90a
sentence, the new README line, and upstream's removal of
`add_subdirectory(Libraries/pybind11)` from the block the port conditions. The
root CMakeLists resolution is upstream's structure with only the port's
`USE_HIP` / `USE_CUDA OR USE_HIP` / `NOT (USE_CUDA OR USE_HIP)` deltas.

`CUBBYFLOW_REQUIRES` left untouched is the right call, and the reasoning holds:
`cuda_to_hip.h` defines `__CUDA_ARCH__` in the device pass but never
`__CUDACC__` (cuda_to_hip.h:18-33), so `Macros.hpp:85` is live under hipcc;
HIP TUs really do get `-std=c++23` (`build/compile_commands.json`, all 35 HIP
entries carry `-std=c++23 --offload-arch=gfx1100 -Werror`); and the only
`CUBBYFLOW_REQUIRES` users are the CPU searchers
(`Includes/Core/Searcher/Point*.hpp`), where the clauses disambiguate
`Serialize`/`Deserialize` overloads that already differ in parameter type -- no
header depends on the macro being empty for device correctness. Same shape at
`Includes/Core/Utils/Parallel.hpp:226,249` and `Parallel-Impl.hpp:522,536`,
which upstream added in this merge: `#ifdef __CUDACC__` selects an
unconstrained template, so the HIP build takes the `std::random_access_iterator`
branch, consistently in every TU of a HIP build (only the nvcc build is
internally split, which is upstream's own property).

Install.md correction is accurate: with `CMAKE_HIP_ARCHITECTURES` unset, CMake
runs `rocm_agent_enumerator` and fatal-errors if it finds nothing
(`CMakeDetermineHIPCompiler.cmake:296-334`); confirmed with a throwaway
`enable_language(HIP)` project, which reported `HIP arch: gfx1100;...` and
picked `/opt/rocm/lib/llvm/bin/clang++` unaided.

Evidence reproduced on this host at the staging tip: `./build/bin/CUDATests`
35/35 cases, 3170/3170 assertions; `./build/bin/UnitTests` 814/814 in 168
suites. The nvcc gate did not fail open -- `build-cuda/CMakeCache.txt` has
`USE_CUDA:BOOL=ON`, `CUDA_VERSION:STRING=12.8`,
`CUDA_NVCC_EXECUTABLE=/opt/conda/envs/cuda-12.8/targets/x86_64-linux/bin/nvcc`,
and `*_generated_*.cu.o` objects exist for CubbyFlow, CUDATests and CUDASPHSim.

Fault classes: no warp intrinsics, `warpSize`, hardcoded 32, `__shared__`,
atomics or `__syncthreads` anywhere under `Sources/Core/CUDA`,
`Includes/Core/CUDA`, `Tests/CUDATests`, `Examples/CUDASPHSim`; kernels remain
1D `blockIdx.x * blockDim.x + threadIdx.x`, so wave32/wave64 are equivalent.
Upstream's delta to the CUDA-adjacent sources is `std::make_shared` cleanups
and doctest include/link changes, and it added no new `.cu` files, so the
unchanged 35-case count is expected rather than a silently excluded test file.
Both source globs still enumerate the same set on the HIP and CUDA paths
(`Sources/Core/CMakeLists.txt:16-33`, `Tests/CUDATests/CMakeLists.txt:5-18`).

jargon: `--commits moat-port..moat-fix-145` clean; `--diff` reports `MOaT` and
`OsrB`, both confirmed inside the base64 payload of `Medias/Logos/Logo.svg`
(`git grep -l` on the branch finds them in that file only, and the file is
byte-identical to upstream). `--port CubbyFlow` cannot run from a worktree
without the clone; run it from /var/lib/jenkins/moat.

Commit hygiene: both titles `[ROCm]`-prefixed and under 72 chars (52 and 49),
both bodies disclose AI assistance and carry a Test Plan in fenced blocks, no
`Co-Authored-By`, no AMD-internal account references.

Skill lessons on this branch are accurate and generalizable: the
`!defined(__CUDACC__)` inversion entry in `fault-classes.md` matches
rocThrust's real backend keying and this build's `-std=c++23` HIP lines, and
both `validation.md` entries (legacy FindCUDA failing open on a conda toolkit,
`autoconf-archive` for a vcpkg manifest) match what the cache and build here
show.

### Bookkeeping, not a defect

`head_sha` still reads 62f4604c because the push is blocked, so `fix-ready`
correctly answers `no-delta ... nothing is staged` and no fix review PR can be
opened prematurely. After the push, `advance-head 2c51f93ecdda50b320c2190f723ef26dd02d008e`
is still required before revalidation; this review is recorded against the
staging tip, not against what status.json currently names.

## Porter response to Review 2026-08-20

Both findings fixed by recreating the two local-only commits (branch had
never been pushed; trees otherwise identical):
- Merge commit is now 40927ee55b: the seam bullet for
  CUDAPointHashGridSearcher{2,3}.hpp says "this branch's include swap to the
  compatibility header" -- the copy/move members are upstream's. The same
  correction is applied to the seam list above.
- Tip is now 29bac0b9b1: the Thrust-pin comment in CMakeLists.txt names the
  real mechanism (rocThrust keys on compiler detection, never project macros;
  every Thrust-reaching TU compiles as HIP, so the pin is defensive), and the
  commit body describes the comment it actually installs. Only the comment
  changed vs the previous tip (git diff 2c51f93e..29bac0b9b1 = 1 file,
  comment-only), so build/test evidence carries: comment text cannot affect
  compilation.
After gh auth refresh: push moat-fix-145, advance-head CubbyFlow
29bac0b9b1a..., then revalidation and the fix-review PR.

## Review 2026-08-20 (linux-gfx1100, reviewer): re-review of the fix round

review-passed at `29bac0b9b1e8981d9921446829f9c0df1b97f2fe`. Both findings from
the review above are fixed and nothing else moved. No new problems.

Scope check first: `git rev-parse 40927ee55b^{tree}` equals
`git rev-parse e5062449^{tree}` (7514c67ec5) and the recreated merge carries the
same two parents (62f4604c, 8f774f6a4f), so the merge commit changed by message
only. `git diff 2c51f93e..29bac0b9b1` is one file, four comment lines in
`CMakeLists.txt:88-91`. `git ls-remote origin moat-fix-145` is empty, so nothing
published was rewritten. The earlier verification of merge fidelity, the
`CUBBYFLOW_REQUIRES` judgment, the fault-class sweep, and the reproduced test
evidence (CUDATests 35/35 3170 assertions, UnitTests 814/814, nvcc gate with
real `*_generated_*.cu.o` objects) therefore carry unchanged.

Finding 1: the merge body bullet now reads "main's member-initializer cleanups
landed alongside this branch's include swap to the compatibility header", which
is what the code shows -- the branch's whole delta in those files is
`CUDAPointHashGridSearcher2.hpp:19-23` / `CUDAPointHashGridSearcher3.hpp:19-23`,
and the copy/move declarations at `CUDAPointHashGridSearcher2.hpp:115-129` are
upstream's. The seam list in this file is corrected to match.

Finding 2: the installed comment at `CMakeLists.txt:88-91` now reads "rocThrust
picks its backend from compiler detection (`__CUDACC__` under nvcc, the host
compiler otherwise), never from this project's macros. Every Thrust-reaching
source here compiles as HIP, so this pin is defensive: it keeps the backend
explicit rather than inferred." Both claims check out against
`thrust/detail/config/compiler.h:111` (NVCC when `__CUDACC__`/`_NVHPC_CUDA`,
else the host-compiler chain, with clang splitting on `__CUDA__`/`__HIP__`) and
`device_system.h:29-36` (HIP device compiler -> HIP, clang -> CPP, else CUDA),
and against this tree, where every thrust includer (`Sources/Core/CUDA/*.cu`,
`CUDAPointHashGridSearcher{2,3}-Impl.hpp` reached from `Sources/Core/CUDA/*.cpp`)
is marked LANGUAGE HIP and the three non-HIP TUs in the CUDA directories include
no thrust. The tip commit body describes the comment it installs.

Hygiene rechecked at the new tip: titles `[ROCm]`-prefixed at 57 and 50 chars,
both bodies disclose AI assistance and carry a Test Plan, no `Co-Authored-By`;
`jargon.py --commits moat-port..moat-fix-145` clean and `--diff` still reports
only the two `Medias/Logos/Logo.svg` base64 strings adjudicated above; the fork
clone has no modified tracked files (untracked `build-cuda/` and two test logs
only).

Bookkeeping unchanged: `head_sha` still names 62f4604c while the push waits on
the token scope, so `fix-ready` answers `no-delta` and no fix review PR can open
early. After the push, `advance-head 29bac0b9b1e8981d9921446829f9c0df1b97f2fe`
is required before revalidation on wave64/wave32/windows.

## Revalidation linux-gfx1100 2026-08-20 (fix round, tip 29bac0b9b1)

Full real-GPU revalidation at `29bac0b9b1e8981d9921446829f9c0df1b97f2fe`
(`moat-fix-145`, Radeon Pro W7800, ROCm 7.2.3, clang 22.0.0). Fork clone
verified at that exact commit, no modified tracked files (untracked
`build-cuda/`, `UnitTests.log`, `cuda_tests.log` only). Ran the real tests at
the tip rather than carrying forward, since they are fast; the earlier
reviewer note that `git diff 2c51f93e..29bac0b9b1` is comment-only in
`CMakeLists.txt` is corroborated by the rebuild doing zero recompilation
(every target reported "Built target X" with no compile lines).

Build (`export VCPKG_ROOT=/var/lib/jenkins/vcpkg`):

```
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build -j$(nproc)
```

PASS, all 15 targets built, no compile invocations needed (incremental,
object files unaffected by the comment-only change), consistent with zero
errors/warnings recorded for this tree in the fix-round build above.

GPU tests:

```
HIP_VISIBLE_DEVICES=0 ./build/bin/CUDATests
```

35/35 test cases, 3170/3170 assertions PASS.

```
HIP_VISIBLE_DEVICES=0 ./build/bin/UnitTests
```

814/814 PASS in 168 suites.

```
HIP_VISIBLE_DEVICES=0 <src>/build/bin/CUDASPHSim -f 5
```

5 frames written, 13824 particles, exit 0. All three counts match the
fix-round evidence recorded above exactly.

CUDA no-regression gate: re-ran at this head_sha too (previously recorded
only at `2c51f93ecd`), using the existing `build-cuda/` tree and the same
recipe as above (`CMAKE_CUDA_ARCHITECTURES=80`, nvcc 12.8 from
`/opt/conda/envs/cuda-12.8`). PASS, `CubbyFlow`, `CUDATests`, `CUDASPHSim`
all build cleanly. Note: this project's own CMake picks
`-gencode;arch=compute_75,code=sm_75` for the actual nvcc invocation
regardless of the `-D` pin (legacy `FindCUDA` path, not
`CMAKE_CUDA_ARCHITECTURES`-driven); sm_75 is a real modern architecture
(Turing) with native `atomicAdd(double*)`, so this is not the "arch pin did
not take, silently degraded to an ancient arch" fingerprint -- just a
different fixed target than requested, and irrelevant to a compile-only
passthrough check. No NVIDIA GPU on this host, no runtime claim made.

`python3 utils/jargon.py --port CubbyFlow` (run from `/var/lib/jenkins/moat`,
not the worktree) is clean. Documentation: `Documents/Install.md` already
carries the corrected HIP/ROCm build section (`USE_HIP=ON`,
`CMAKE_HIP_ARCHITECTURES`) reviewed in the fix round above; unchanged by this
revalidation.

Commands: `bash utils/timeit.sh CubbyFlow compile -- bash agent_space/build_hip.sh`,
`bash utils/timeit.sh CubbyFlow test -- bash agent_space/run_cudatests.sh`,
`bash utils/timeit.sh CubbyFlow test -- bash agent_space/run_unittests.sh`,
`bash utils/timeit.sh CubbyFlow test -- bash agent_space/run_sphsim.sh`,
`bash utils/timeit.sh CubbyFlow cuda-compile -- bash agent_space/build_cuda.sh`.
All exit 0. Result: PASS, `completed` at `29bac0b9b1e8981d9921446829f9c0df1b97f2fe`.

## Revalidation linux-gfx90a 2026-08-20 (fix round, tip 29bac0b9b1)

Platform: linux-gfx90a (AMD Instinct MI250X, gfx90a, ROCm 7.14 TheRock SDK via
conda env `py_3.12`, clang 23.0.0 -- `/opt/conda/envs/py_3.12/lib/python3.12/
site-packages/_rocm_sdk_devel`; this host has no `/opt/rocm`). HIP_VISIBLE_DEVICES=0
(4 GCDs present, all idle). `classify` returned `class=unknown
arch_independent=False (classification failed -> revalidate)`, so a full
rebuild + real-GPU run was required (not a carry-forward candidate).

No local fork clone existed on this host; cloned fresh
(`git clone --branch moat-fix-145 --single-branch
https://github.com/AMD-Ecosystem/CubbyFlow.git projects/CubbyFlow/src`,
plus `git fetch origin main:main` afterward for jargon.py's diff range).
`git rev-parse HEAD` = `29bac0b9b1e8981d9921446829f9c0df1b97f2fe`, matching
`head_sha` exactly. Installed the protect-fork pre-push hook
(`python3 utils/moatlib.py protect-fork CubbyFlow`). Tree clean throughout
(`git status --porcelain` empty; only untracked `build/`).

Bootstrapped a fresh vcpkg (this host had none): a `--depth 1` shallow clone
of vcpkg FAILS `cmake --build` at configure time (`vcpkg install failed`,
"vcpkg was cloned as a shallow repository... failed to git show
versions/baseline.json" for every manifest port) because vcpkg's versioned
baseline resolution needs full history, not just the tip tree. Full clone
(`git clone https://github.com/microsoft/vcpkg.git`, no `--depth`) plus
`./bootstrap-vcpkg.sh -disableMetrics` fixed it. Also needed `apt-get install
autoconf-archive` (matches the gfx1100 note; libtool/zip/unzip/curl were
already present on this host).

### Build

```
export ROCM=/opt/conda/envs/py_3.12/lib/python3.12/site-packages/_rocm_sdk_devel
export VCPKG_ROOT=/var/lib/jenkins/vcpkg
cd projects/CubbyFlow/src
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=${ROCM}/lib/llvm/bin/clang++ \
  -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build -j$(nproc)
```

PASS, 100%, all targets built cleanly (`bash utils/timeit.sh CubbyFlow
compile -- bash agent_space/cubbyflow_gfx90a_build.sh`).

### GPU tests

```
HIP_VISIBLE_DEVICES=0 ./build/bin/CUDATests
```

35/35 test cases, 3170/3170 assertions PASS -- exact match to the gfx1100
fix-round evidence.

```
HIP_VISIBLE_DEVICES=0 ./build/bin/CUDASPHSim -f 5
```

5 frames written, 13824 particles, exit 0 -- exact match.

### CPU regression tests

```
HIP_VISIBLE_DEVICES=0 ./build/bin/UnitTests
```

814/814 PASS in 168 suites -- exact match, no non-GPU regression.

### CUDA no-regression gate

Not re-run: already recorded PASS at this exact head_sha
(`29bac0b9b1e8981d9921446829f9c0df1b97f2fe`) in the linux-gfx1100
revalidation above (nvcc 12.8, `CMAKE_CUDA_ARCHITECTURES=80`, CubbyFlow/
CUDATests/CUDASPHSim all build cleanly). The gate is per-head_sha, not
per-platform.

### jargon / documentation

`python3 utils/jargon.py --port CubbyFlow` (from `/var/lib/jenkins/moat`,
after `git fetch origin main:main` in the fork clone so the diff range
resolves): clean. `Documents/Install.md:104-113` documents the ROCm/HIP
build (`USE_HIP=ON`, `CMAKE_HIP_ARCHITECTURES`) in the same section as the
CUDA build; unchanged by this revalidation.

### Verdict: PASS -> completed at `29bac0b9b1e8981d9921446829f9c0df1b97f2fe`

## Validation 2026-08-20 (windows-gfx1151, validator) -- FAILED at build

Platform: windows-gfx1151 (AMD Radeon 8060S, RDNA3.5 APU, 20 CUs, wave32),
Windows 11. Fix round tip 29bac0b9b1e8981d9921446829f9c0df1b97f2fe
(moat-fix-145). Fresh clone (no prior local checkout on this host):
`git clone --branch moat-fix-145 --single-branch
https://github.com/AMD-Ecosystem/CubbyFlow.git projects/CubbyFlow/src`.
`git rev-parse HEAD` matched head_sha exactly. protect-fork installed
(the upstream PR #145 is open; moat-port stays untouched throughout).
62f4604c834e0c25ec8f5a829fc1670db23d3cb0 (the sha windows-gfx1101/gfx1201
validated) still exists as the tip of moat-port on the fork (git ls-remote
confirms). Tree stayed clean (git status --porcelain only ever showed the
untracked build_gfx1151/); no fork source or build file was edited during
this session.

### Toolchain prerequisites fixed locally (validator-side only, no fork edits)

- D:/vcpkg was a shallow clone (git rev-list --count HEAD == 1); a manifest
  install fails at configure with "vcpkg was cloned as a shallow
  repository ... failed to git show versions/baseline.json" for every port
  (same fault class the linux-gfx90a validator hit and fixed the same way).
  Fixed with `git fetch --unshallow origin` (30161 commits after).
- CubbyFlow's CMakeLists.txt line 2 requires cmake_minimum_required(VERSION
  3.31.6); the system CMake (3.31.0, both the PATH one and vcpkg's own
  bundled tool) is below that floor. Used vcpkg's cached
  cmake-3.31.10-windows tool
  (C:/Users/jdaily/AppData/Local/vcpkg/downloads/tools/cmake-3.31.10-windows/...)
  explicitly for both configure and build.

### Attempt 1: all-clang-cl (CMAKE_C/CXX/HIP_COMPILER = clang-cl.exe)

Per the general Windows-ROCm house style (clang-cl as host compiler).
vcpkg install succeeded (22 min cold, this host's first CubbyFlow vcpkg
build). CMake configure failed at HIP language detection:

CMake Error at .../Windows-MSVC.cmake:69 (message):
  MSVC compiler version not detected properly:
Call Stack: .../Windows-Clang-HIP.cmake:1, CMakeHIPInformation.cmake:35,
  CMakeLists.txt:81 (enable_language)
CMake Error at .../CMakeDetermineCompilerABI.cmake:74 (try_compile):
  Failed to configure test project build system.
-- Configuring incomplete, errors occurred!

This is the recorded CMake-3.31-vs-clang-cl-as-HIP-compiler ABI-detection
wall (this host has no newer CMake). Environmental, not project-specific.

### Attempt 2: all-clang++ (GNU-driver, matching the project's own prior
Windows recipe from the 2026-06-12/2026-06-19 windows-gfx1201/gfx1101
validations: CMAKE_HIP_COMPILER=CMAKE_CXX_COMPILER=clang++.exe,
CMAKE_C_COMPILER=clang.exe)

vcpkg restored the whole dependency set from the now-warm binary cache in
under 5s (the compiler-hash cache key differs per compiler, but package
contents matched, so vcpkg served every port from
C:\Users\jdaily\AppData\Local\vcpkg\archives). CMake configure succeeded,
including HIP language ABI detection (-- Using HIP: arch gfx1151). Build
failed compiling the first HIP translation unit:

[149/326] Building HIP object Tests/CUDATests/.../CUDAArray1Tests.cu.obj
D:\...\amdclang++.exe ... -std=c++23 --offload-arch=gfx1151 ... -x hip -c
  .../Tests/CUDATests/CUDAArray1Tests.cu
.../__clang_cuda_math_forward_declares.h:93:17: error: __device__ function
  'isfinite' cannot overload __host__ __device__ function 'isfinite'
C:\...\VC\Tools\MSVC\14.44.35207\include\cmath:672:17: note: previous
  declaration is here
  672 | _CLANG_BUILTIN1(isfinite)

19 such errors (isfinite/isinf/isnan/isnormal for float/double/long double
plus related overloads), all in the first HIP TU compiled -- every HIP TU
in the project would fail identically, since they all pull the same
__clang_hip_runtime_wrapper.h chain.

### Attempt 3 (diagnostic, not a fix): amdclang.exe/amdclang++.exe

Same GNU-driver family as attempt 2 (CuRast's Windows recipe uses
amdclang++.exe specifically). Ran to confirm the failure is not an
artifact of using plain clang++.exe instead of the amdclang wrapper.
vcpkg restored again from cache (408 ms). Configure succeeded identically.
Build hit the exact same 19 errors on the same file (CUDAArray1Tests.cu.obj,
isfinite/isinf/isnan/isnormal), confirming the fault is independent of
which GNU-driver clang binary is used.

### Root cause and the fix, precisely (not applied -- validator role, no fork edits)

CubbyFlow's CMake sets CXX_STANDARD 23 project-wide (upstream's C++23
move, Builds/CMake/CompileOptions.cmake), and per the 2026-08-20 fix-round
review, this propagates to the HIP language too -- there is no HIP-scoped
override. On Linux with ROCm's own libc++/libstdc++, -std=c++23 on a HIP
TU is fine (confirmed on linux-gfx1100/gfx90a: -std=c++23
--offload-arch=gfx1100 compiles clean). On Windows, an all-GNU-driver HIP
TU still targets the MSVC CRT (-D_DLL -D_MT -Xclang
--dependent-lib=msvcrt), so it still includes the MSVC C++23 STL <cmath>,
which marks isfinite/isinf/isnan/isnormal __host__ __device__ via
_CLANG_BUILTIN1(...). HIP's own __clang_cuda_math_forward_declares.h
declares the same names __device__-only. A __device__-only redeclaration
of an existing __host__ __device__ overload is an error, not a merge --
hence every HIP TU fails at TU #1.

This is the same fault class CuRast's Windows port hit and fixed (its fix
#2, projects/CuRast/notes.md): pin HIP translation units to -std=c++20
(via target_compile_options($<$<COMPILE_LANGUAGE:HIP>:-std=c++20>),
WIN32-scoped so Linux/macOS keep C++23 unchanged) so the HIP TU never
reaches the C++23 STL branch of <cmath> that carries the conflicting
builtin markers. CubbyFlow's CMake has no such HIP-scoped standard pin
today -- this is a genuine, describable, porter-actionable build gap, not
a toolchain/environment wall and not a candidate for a Windows waiver (the
give-away: a fix a porter could ship, already proven on a sibling project
on this exact host today).

Attempt 1's failure is a separate, environmental issue (CMake 3.31 cannot
ABI-probe clang-cl as a HIP-language compiler on this host; no newer CMake
installed) -- unrelated to attempt 2/3's cmath conflict. Not verified
whether a newer CMake would let clang-cl reach the same cmath conflict
(clang-cl also targets the MSVC CRT, so it looks at least as exposed);
attempt 1 never got past configure so this was not directly confirmed.

### No GPU tests run

The build never produced CUDATests.exe, UnitTests.exe, or CUDASPHSim.exe
for this arch/toolchain combination, so no GPU or CPU-regression evidence
was collected on windows-gfx1151 this round. USE_HIP=ON marks only
Sources/Core/CUDA, Tests/CUDATests, and Examples/CUDASPHSim sources as HIP
language, so the CPU-only UnitTests target compiles no HIP TUs and would
likely build independently of this fault -- not verified this round
(stopped after confirming the root cause, per the revised time budget).

### CUDA no-regression gate

Not applicable/not re-run on this host (no CUDA toolkit; already recorded
PASS at this exact head_sha by linux-gfx1100, the gate is per-head_sha,
not per-platform).

### Host stability

Kernel_141 GPU-engine-timeout report count: 18 before this session, 18
after (unchanged) -- no new GPU timeouts or crashes from this session's
failed builds.

### Verdict: FAIL -> validation-failed at 29bac0b9b1e8981d9921446829f9c0df1b97f2fe

Sent back to the porter. Needed fix: pin HIP-language translation units to
-std=c++20 on Windows (WIN32-scoped, mirroring CuRast's fix in
Sources/Core/CMakeLists.txt or the shared HIP compile-options block),
leaving the CXX-language CXX_STANDARD 23 and the Linux/macOS HIP build
unchanged. After that fix, this platform needs a full re-run; vcpkg deps
are already warm in this host's binary cache from this session, so the
next attempt should reconfigure/build in well under 5 minutes before
reaching GPU tests.

## Porter round 2026-11-20 (windows-gfx1151): HIP sources at C++20 on Windows

Answers the 2026-08-20 windows-gfx1151 build failure. Fresh clone of
moat-fix-145 at 29bac0b9b1 into projects/CubbyFlow/src (protect-fork run;
PR #145 open, moat-port untouched at 62f4604c). New commit on the staging
branch: `cc41a8056aae1f334bf68628b137e1a3c7dd7297`
`[ROCm] Compile HIP sources as C++20 on Windows` (pushed; moat-fix-145
is a strict descendant of moat-port).

### The fix

`Builds/CMake/CompileOptions.cmake`, after `DEFAULT_PROJECT_OPTIONS`:

```
if (USE_HIP AND WIN32)
    list(APPEND DEFAULT_PROJECT_OPTIONS
        HIP_STANDARD              20
        HIP_STANDARD_REQUIRED     ON
    )
endif ()
```

The target property `HIP_STANDARD` is the clean override: CMake falls back to
`CXX_STANDARD` for HIP only when `HIP_STANDARD` is unset, which is how HIP TUs
got `-std=c++23`. A `$<$<COMPILE_LANGUAGE:HIP>:-std=c++20>` compile option would
instead leave two `-std=` flags on the line. Every HIP-bearing target
(CubbyFlow, CUDATests, CUDASPHSim) already applies `DEFAULT_PROJECT_OPTIONS`.
Verified in build.ninja: 35 HIP objects at `-std=c++20`, 734 CXX objects at
`-std=c++23` (no other `-std=` value).

Hypothesis check (C++23-only constructs in HIP code): grep over Includes/,
Sources/, Tests/CUDATests, Examples/CUDASPHSim for std::expected/print/mdspan/
ranges::to/to_underlying/unreachable/byteswap, `if consteval`, deducing this,
`uz` literals, multi-arg operator[], [[assume]] -- none (the only hits were
pybind11 `kwargs.contains` in the Python API, which is not built under HIP).
`CUBBYFLOW_REQUIRES` concepts are C++20, so they still apply under HIP. The
build compiling every HIP TU confirms it.

CUDA path: the block is inside `USE_HIP AND WIN32`; a CUDA build has USE_HIP
OFF, so the CUDA (and Linux HIP) build sees an identical property list. No
Windows CUDA configure was run (it would need a fresh MSVC/nvcc vcpkg
configure; not cheap). The change cannot reach nvcc.

### Build (windows-gfx1151, TheRock ROCm SDK 7.14 clang 23, CMake 3.31.10)

Script `agent_space/build_cf.sh` (worktree), the GNU-driver recipe:

```
unset INCLUDE LIB LIBPATH EXTERNAL_INCLUDE VCINSTALLDIR VSINSTALLDIR
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -vi 'visual studio' | paste -sd:)
export VCToolsInstallDir='C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\'
export ROCM_ROOT=D:/Develop/TheRock/.venv/Lib/site-packages/_rocm_sdk_devel
export HIP_DEVICE_LIB_PATH=$ROCM_ROOT/lib/llvm/amdgcn/bitcode
export VCPKG_ROOT=D:/Develop/vcpkg
cmake-3.31.10 -B build_gfx1151_b -S . -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER=$ROCM_ROOT/lib/llvm/bin/clang++.exe \
  -DCMAKE_CXX_COMPILER=$ROCM_ROOT/lib/llvm/bin/clang++.exe \
  -DCMAKE_C_COMPILER=$ROCM_ROOT/lib/llvm/bin/clang.exe \
  -DCMAKE_PREFIX_PATH=$ROCM_ROOT -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake-3.31.10 --build build_gfx1151_b --target CubbyFlow CUDATests UnitTests CUDASPHSim -j 12
```

Result: PASS, 633/633 steps, no errors or warnings under -Werror.

Host-environment trap (not a port issue): the first attempt ran in a shell
with the VS 2022 BuildTools developer environment loaded (INCLUDE/LIB/
EXTERNAL_INCLUDE at MSVC 14.44). With that, every HIP TU compiled (the fix
works against the 14.44 STL, the exact headers the validator failed on), but
linking UnitTests.exe failed: `lld-link: error: undefined symbol:
__std_find_last_not_of_trivial_pos_1`, referenced from vcpkg's
tinyobjloader_double.lib, which the binary cache had built with MSVC 14.50
(VS 18). Clang's MSVC detection follows INCLUDE/LIB/EXTERNAL_INCLUDE and
VCToolsInstallDir, so scrubbing the VS 2022 environment and pointing
VCToolsInstallDir at 14.50 matched the toolchain to the cached libs. A
validator on this host must do the same (or build vcpkg deps with 14.44).
The final build is entirely 14.50 headers + libs, so the fix is proven
against both the 14.44 and 14.50 C++23 <cmath>.

clang-cl (validator attempt 1): not retried. Its failure is inside CMake's own
Windows-Clang-HIP / Windows-MSVC modules ("MSVC compiler version not detected
properly" during enable_language(HIP)), before any project code runs, so no
project-side build-file workaround belongs here; the GNU-driver clang++ path
the Install.md command uses works.

### Smoke tests (porter evidence, single runs, ulimit -c 0)

TheRock amdhip64_7.dll, amd_comgr.dll, hiprtc*.dll, rocm_kpack.dll copied
next to the exes in build_gfx1151_b/bin.

- `CUDATests.exe`: 35/35 test cases, 3170/3170 assertions PASS.
- `CUDASPHSim.exe -f 5`: 5 frames, 13824 particles, exit 0, no NaN; x,y =
  FLT_MAX (the known empty-container non-issue above), mean z 0.205.
- `UnitTests.exe` (CPU): 814/814 PASS (count grew from 722 with upstream's
  merge).

Docs: Documents/Install.md already carries the USE_HIP block; this
build-file fix needs no doc change. jargon.py --port CubbyFlow: clean.

## Review 2026-11-20 (windows-gfx1151, reviewer): fix round tip cc41a805

changes-requested at `cc41a8056aae1f334bf68628b137e1a3c7dd7297` (moat-fix-145).

Scope. `reviewed_sha` was unset when the lock was taken (the 2026-08-20 pass was
recorded while `head_sha` still named 62f4604c, and the later `advance-head`
cleared it), so a pass here stamps the whole staged delta 62f4604c..cc41a805.
The history is linear: `git log 62f4604c..HEAD` ends 40927ee55b (merge),
29bac0b9b1 (docs/comment), cc41a8056a, and 29bac0b9b1 is the exact commit the
2026-08-20 re-review passed, so that review's verification of the merge, docs and
fault-class sweep carries; this review read the new commit
(`git diff 29bac0b9..cc41a805`, one file, `Builds/CMake/CompileOptions.cmake:36-45`)
in full and re-read the delta's readability as a whole.

What checks out (not findings):
- Guard: `USE_HIP` is an option set at `CMakeLists.txt:74`, before
  `include(Builds/CMake/CompileOptions.cmake)` at `CMakeLists.txt:163`. A CUDA
  build has `USE_HIP` OFF and a Linux HIP build has `WIN32` false, so both see the
  identical property list; even if the block were entered, `HIP_STANDARD` only
  affects HIP-language sources and cannot reach nvcc (CUDA keeps
  `CMAKE_CUDA_STANDARD 17`, `CMakeLists.txt:102`). The Windows CUDA configure the
  porter skipped therefore does not matter for this change, and the nvcc gate at
  29bac0b9 is untouched by it.
- Coverage: only `Sources/Core/CMakeLists.txt:29`, `Tests/CUDATests/CMakeLists.txt:12`
  and `Examples/CUDASPHSim/CMakeLists.txt:24` mark sources `LANGUAGE HIP`, and all
  three targets apply `${DEFAULT_PROJECT_OPTIONS}` (`:48`, `:23`, `:33`).
  `build_gfx1151_b/build.ninja`: 35 HIP objects (26 CubbyFlow, 8 CUDATests,
  1 CUDASPHSim) all at `-std=c++20`, 734 at `-std=c++23`, no other value; C++20
  keeps `__cpp_concepts >= 201907L`, so `CUBBYFLOW_REQUIRES` stays live on the HIP
  path as on Linux.
- Commit: title 46 chars with `[ROCm]`, ASCII, prose AI disclosure, fenced Test
  Plan, no `Co-Authored-By`; `python utils/jargon.py --port CubbyFlow`: clean.

### Finding 1: the comment names the wrong cause, so nobody will know when to drop the pin

`Builds/CMake/CompileOptions.cmake:36-39` says the MSVC STL's C++23 `<cmath>`
"declares isfinite/isinf/isnan/isnormal `__host__ __device__`". It does not. In
14.50 `cmath:602-604` the `_CLANG_BUILTIN1_ARG` overloads are `_CONSTEXPR23`,
which `yvals_core.h:963` makes `constexpr` under C++23 (P0533R9) and `inline`
otherwise; clang in HIP mode treats a `constexpr` function as implicitly
`__host__ __device__`, and `__clang_hip_runtime_wrapper.h` includes
`<__clang_cuda_math_forward_declares.h>` AFTER `<cmath>` (lines 147 vs 114 in
the TheRock 7.14 clang 23 used for this build), so its `__device__`-only
declarations collide. That is a clang/ROCm wrapper ordering bug, fixed in
llvm/llvm-project#201563 and present from ROCm 10.1 nightlies
(`cuda-to-rocm` `references/build-headers-macros.md`, "HIP + C++23 on an older
toolchain"); the MSVC STL is conforming. As written, an upstream maintainer
reading the comment would blame their STL and keep a C++20 pin forever. Reword
the comment (a new commit; cc41a805's body cannot be amended and carries the same
misstatement, so the follow-up commit body should state the correction) to say:
C++23 makes these `constexpr`, which HIP treats as host+device, and older ROCm
clang declares device-only overloads after `<cmath>`; drop this once the
minimum supported ROCm includes the wrapper fix.

### Finding 2: the fix deviates from the documented remedy without the analysis that would justify it

The same skill section measures the alternatives on this exact host/toolchain and
prefers `assets/HipCxx23Shim.cmake` (generated wrapper with the includes
reordered, language standard untouched), ranking the `-std=c++20` pin last
because HIP TUs and host TUs then build against different standard-library
configurations -- an ODR hazard MSVC cannot detect (no `detect_mismatch` for
`_HAS_CXX*`). It also says not to guard on `WIN32` (libstdc++ adopting P0533R9
breaks Linux identically) and to probe instead. CubbyFlow is exactly the mixed
case: the `CubbyFlow` static library links 26 C++20 HIP objects with C++23 CXX
objects that include the same `Includes/Core` headers. The porter notes
(2026-11-20 round, "Hypothesis check") checked only that HIP code uses no C++23
language features; they do not address the STL-configuration split. Either
switch to the shim, or record in notes.md why the pin is safe here (which STL
types cross between the HIP and CXX objects of the `CubbyFlow` library and that
none of them changes layout or inline definition under `_HAS_CXX23`), and why a
`WIN32` guard is acceptable over the probe. The pin's smaller footprint is a
legitimate reason to prefer it for an upstream PR, but that choice has to be
argued, not defaulted to from the CuRast precedent the validator cited, which
predates the skill's measurement.

Readability of the staged delta as a whole: an upstream maintainer sees their own
merged history plus three of our commits (merge resolution, docs, this one); with
Finding 1 fixed the new commit is self-explanatory. No other problems.

## Porter round 2026-11-20 (windows-gfx1151): answer to review of cc41a805

New commit on `moat-fix-145` (strict descendant of moat-port 62f4604c, which
is untouched): `aeb9540d6be87d4fcde76a00ded5f972b4615c35`
`[ROCm] Build HIP sources at C++23 with a probed header workaround` (pushed).
Remedy shipped: the skill's generated-wrapper shim, adapted; the C++20 pin is
gone, so no ODR analysis is needed.

### Cause, verified on this host's headers

- MSVC 14.50 `include/cmath:602-605`: `_CLANG_BUILTIN1_ARG` defines
  `_NODISCARD _Check_return_ _CONSTEXPR23 bool NAME(ARG) noexcept`;
  `yvals_core.h:962-966` makes `_CONSTEXPR23` `constexpr` under `_HAS_CXX23`,
  `inline` otherwise. Nothing is declared `__host__ __device__` by the STL.
- TheRock 7.14 clang 23 (`AMD clang version 23.0.0git ... 52226beb24`):
  `lib/clang/23/include/__clang_hip_runtime_wrapper.h` has `#include <cmath>`
  at line 114 and `#include <__clang_cuda_math_forward_declares.h>` at 147,
  i.e. the pre-llvm/llvm-project#201563 order.
- Negative control (free, from the first rebuild of this round, when the shim
  re-probe was still broken): with the pin removed and no shim, every HIP TU
  fails `__clang_cuda_math_forward_declares.h:93:17: error: __device__ function
  'isfinite' cannot overload __host__ __device__ function 'isfinite'`.

### The change

- `Builds/CMake/CompileOptions.cmake`: the `USE_HIP AND WIN32` HIP_STANDARD 20
  block is removed (file back to its 29bac0b9 content, i.e. upstream's).
- New `Builds/CMake/HipCxx23Shim.cmake` (59 lines, CubbyFlow CMake style),
  `cubbyflow_hip_cxx23_shim(OUT_DIR)`: `try_compile` (SOURCE_FROM_CONTENT,
  NO_CACHE, HIP_STANDARD 23, STATIC_LIBRARY) of
  `#include <hip/hip_runtime.h> / #include <cmath> / __global__ ...`; returns
  empty if it builds or fails with any other diagnostic. Only on
  "cannot overload __host__ __device__" it reads the installed wrapper from
  `-print-resource-dir`, refuses if already in fixed order or unexpected
  shape, writes the reordered copy to `<build>/HipCxx23Shim/`, re-probes with
  `-I` that dir, and discards it with a warning (log attached) if it does not
  help.
- `CMakeLists.txt`, inside `if (USE_HIP)` after the `-include cuda_to_hip.h`
  flag: two-line comment (real cause, drop once ROCm carries #201563), then the
  call, appending `-I"<dir>"` to `CMAKE_HIP_FLAGS` when non-empty. Using
  CMAKE_HIP_FLAGS keeps the directory HIP-only (CXX TUs of the mixed CubbyFlow
  target never see it) and ahead of the resource dir.
- Deviation from the asset: try_compile instead of execute_process, so the
  probe sees the project's own HIP toolchain settings; the first attempt
  dropped `<hip/hip_runtime.h>` from the probe and the shim re-probe failed on
  the host pass with `Can't find declaration for hipLaunchKernel` (a bare
  `__global__` needs the runtime header), so the include was restored as in
  the asset. Verified by hand with
  `clang++ -std=c++23 --offload-arch=gfx1151 -x hip [-I shim] -c probe2.hip`:
  rc=74 without, rc=0 with.

CUDA and non-HIP builds: the whole change sits inside `if (USE_HIP)`; a CUDA
build has USE_HIP OFF and `CompileOptions.cmake` is back to upstream's, so the
CUDA/CPU configure is identical to 29bac0b9 (the commit whose nvcc gate passed).
No nvcc run this round for that reason.

Linux HIP: the 2026-08-20 fix round already built Linux HIP at `-std=c++23`
with libstdc++ (ROCm 7.2.3 clang 22, gfx1100; ROCm 7.14 TheRock, gfx90a;
see "C++23 and CUBBYFLOW_REQUIRES under HIP" above). So on Linux the first
try_compile succeeds and the function returns empty: no generated header, no
extra flag, HIP flags identical to 29bac0b9. The cost is one HIP probe compile
per configure. If libstdc++ later adopts constexpr <cmath>, the probe catches
it there too (that is why it is not guarded on WIN32). On a ROCm carrying
#201563 the probe passes and nothing is added on Windows either.

### Build and tests (windows-gfx1151, same recipe as "Porter round 2026-11-20")

`agent_space/build_cf.sh` in the worktree (MSVC 14.50 VCToolsInstallDir, VS
2022 env scrubbed, TheRock 7.14 clang++ GNU driver, CMake 3.31.10, Ninja),
now `-j 6`. Reconfigure of the existing `build_gfx1151_b`:
`-- Using reordered HIP runtime wrapper for C++23: .../build_gfx1151_b/HipCxx23Shim`.
build.ninja: 35 HIP objects (all `--offload-arch` FLAGS lines) at
`-std=c++23`, each with `-I".../HipCxx23Shim"`; 734 CXX objects at
`-std=c++23`; no other `-std=`. Rebuild: 43/43 steps (35 HIP objects + links),
0 errors, 0 warnings under -Werror. Generated header differs from the
installed one only by the moved forward-declares line.

Single runs, `ulimit -c 0`, TheRock amdhip64_7.dll / amd_comgr.dll next to
the exes:
- `CUDATests.exe`: 35/35 test cases, 3170/3170 assertions PASS.
- `CUDASPHSim.exe -f 5`: 5 frames, 13824 particles, exit 0, 0 NaN/inf,
  mean z 0.20536 (x,y FLT_MAX as before).
- `UnitTests.exe`: 814/814 PASS.

`python utils/jargon.py --port CubbyFlow`: clean.

### Replies to the review

- Finding 1 (wrong cause): fixed. The CMakeLists.txt comment and the module
  header state the real cause (C++23 constexpr <cmath>, HIP implicit
  host+device, wrapper declares device-only overloads after <cmath>) and the
  drop condition (ROCm carrying llvm/llvm-project#201563). aeb9540d's body
  opens with the explicit correction of cc41a805's explanation.
- Finding 2 (deviation from the documented remedy): resolved by switching to
  the shim with a configure-time probe, not a WIN32 guard; HIP and CXX objects
  now share one standard-library configuration, so the ODR question is moot.
  Decision (shim over pin) was made by the orchestrating session following the
  skill's preference and is open for the maintainer to overrule; the pin's
  advantage was only footprint (11 lines vs 67).

## Lesson for cuda-to-rocm: HipCxx23Shim as try_compile, scoped through CMAKE_HIP_FLAGS

Promoted to cuda-to-rocm/references/build-headers-macros.md in 0bd85094.

For `references/build-headers-macros.md` ("HIP + C++23 on an older
toolchain"). The asset's execute_process probe can also be written as two
`try_compile(... SOURCE_FROM_CONTENT ... NO_CACHE HIP_STANDARD 23)` calls,
which reuse the project's HIP compiler settings; keep `<hip/hip_runtime.h>` in
the probe, since a bare `__global__` fails the host pass ("Can't find
declaration for hipLaunchKernel") and makes the shim look ineffective. In a
target mixing HIP and C++ sources, appending the shim's `-I` to
`CMAKE_HIP_FLAGS` scopes it to HIP translation units only. (CubbyFlow)

## Review 2026-11-20 (windows-gfx1151, reviewer): re-review at aeb9540d

review-passed at `aeb9540d6be87d4fcde76a00ded5f972b4615c35` (moat-fix-145). Scope:
the whole staged delta 62f4604c..aeb9540d is stamped; 62f4604c..29bac0b9 carries
from the 2026-08-20 re-review, cc41a805 from the review above, and this review
read `git diff 29bac0b9..aeb9540d` (net: new `Builds/CMake/HipCxx23Shim.cmake`,
`CMakeLists.txt:87-94`; `CompileOptions.cmake` byte-identical to 29bac0b9) and both
commit bodies in full. No findings against the fork.

Earlier findings: both resolved. Finding 1: `CMakeLists.txt:87-88` and
`HipCxx23Shim.cmake:4-9` state the real cause (C++23 constexpr `<cmath>`, implicit
host+device, wrapper declares device-only overloads after `<cmath>`) and the drop
condition (ROCm carrying llvm/llvm-project#201563); aeb9540d's body opens by
retracting cc41a805's explanation. Finding 2: the pin is gone; HIP and CXX TUs are
both at C++23, so the ODR question no longer arises.

Shim vs `assets/HipCxx23Shim.cmake`, checked against the installed TheRock 7.14
wrapper (`#include <cmath>` line 114, forward-declares line 147, one occurrence of
each, both inside `!__HIPCC_RTC__` branches) and the generated copy in
`build_gfx1151_b/HipCxx23Shim/` (differs only by the moved line; licence header
intact): faithful where it matters -- acts only on the exact diagnostic, refuses a
wrapper already in fixed order or missing either include, re-probes and discards on
failure. Probe evidence in `build_gfx1151_b/CMakeFiles/CMakeConfigureLog.yaml`: the
first probe fails at `-std=gnu++23` with the diagnostic, the second passes with the
`-I`. Robustness: `NO_CACHE` re-probes every configure, and `CMAKE_HIP_FLAGS` is a
normal variable recomputed from the cache value, so a re-configure probes the bare
toolchain (a stale generated header is never on the path after a ROCm upgrade that
fixes the bug, and is overwritten otherwise); `STATIC_LIBRARY` avoids linking
(cross-compile safe); Ninja Multi-Config is irrelevant to a header path; a compiler
with different diagnostic text gets no shim and fails as before, which is the
correct outcome. `CMAKE_TRY_COMPILE_TARGET_TYPE` is function-scoped; nothing runs
outside `if (USE_HIP)`; nothing generated is installed (`Sources/Core/CMakeLists.txt:82-84`
installs only the library and source headers). Licence: the LLVM header is copied
into the build tree at configure time, never committed or installed, with its
Apache-2.0 WITH LLVM-exception header kept; no maintainer concern I can see.
Deviations from the asset that are not defects: no RESULT check on
`-print-resource-dir` and `file(READ)` would be fatal if the path were wrong, but
that branch is reached only after clang itself emitted the diagnostic; the probe
inherits try_compile's Debug config and `gnu++23` rather than the project's
Release/`c++23`, which does not change `_HAS_CXX23` and was shown not to matter.
Linux HIP: not configured at this head; the validators will see the first probe
pass (Linux HIP built at C++23 on 2026-08-20).

Commits: titles 65 and 46 chars with `[ROCm]`, ASCII, prose AI disclosure, fenced
Test Plans with placeholders, no trailer; `python utils/jargon.py --port CubbyFlow`:
clean. Fork worktree has no modified tracked files.

For the person approving the fix round (not a defect): the staged history adds the
C++20 pin in cc41a805 and removes it in aeb9540d. Fix branches are never rebased,
and aeb9540d's body explicitly corrects cc41a805, so the history is honest and
acceptable as is; the net diff a maintainer sees for this round is only the shim
(67 lines). If the maintainer squash-merges, the wrong explanation in cc41a805 is
not carried.

For the session promoting the lesson below: "reuse the project's HIP compiler
settings" is slightly overstated. try_compile reuses the compiler, `CMAKE_HIP_FLAGS`
and architectures, but not the build configuration (it compiled Debug, `-O0`,
msvcrtd here) or `CXX_EXTENSIONS OFF` (probe at `gnu++23`). Say which settings
carry, and note that the inherited `CMAKE_HIP_FLAGS` includes any project
force-includes.

## Validation 2026-11-20 (linux-gfx1100, validator): fix-round head aeb9540d

PASS at `aeb9540d6be87d4fcde76a00ded5f972b4615c35` (`moat-fix-145`), Radeon Pro W7800 (gfx1100), `HIP_VISIBLE_DEVICES=0`, `ulimit -c 0`. Full rebuild and real-GPU run, not a carry-forward (`classify 29bac0b9..aeb9540d`: `class=mixed arch_independent=False`).

Toolchain: TheRock ROCm SDK `10.2.0a20261022` (AMD clang 24.0.0git, ROCm/llvm-project 2792f8ceeb), installed into the job's own venv with `pip install --index-url https://nightly.repo.amd.com/rocm/whl-next/ 'rocm[libraries,devel,device-gfx1100]==10.2.0a20261022'`. No 10.2 release wheel is published (repo.amd.com stops at 7.14.1); this is the newest 10.2 nightly on whl-next (10.3 nightlies follow). The host image itself carries 10.1.0; no system `/opt/rocm`. CMake 3.31.6, Ninja, g++ 13.3 for CXX. Fresh full vcpkg clone + `bootstrap-vcpkg.sh -disableMetrics` in the job directory.

Fork clone: `git clone --branch moat-fix-145 --single-branch https://github.com/AMD-Ecosystem/CubbyFlow.git projects/CubbyFlow/src`, `HEAD` = head_sha, protect-fork hook installed, `git status --porcelain` empty after build and tests.
## Validation 2026-11-20 (linux-gfx950, validator): fix round tip aeb9540d

Full real-GPU validation at `aeb9540d6be87d4fcde76a00ded5f972b4615c35`
(`moat-fix-145`), AMD Instinct MI355X (gfx950), HIP_VISIBLE_DEVICES=5 (the
dispatcher's pin). First validation of this platform; `classify` not needed.

Toolchain: TheRock ROCm 10.2.0 wheels (`rocm[devel,device-gfx950]==10.2.0`
from `https://stable.repo.amd.com/rocm/core/whl-next/`) installed into the job
venv, not a system ROCm; `rocm-sdk version` = 10.2.0, `AMD clang version
24.0.0git (ROCm/llvm-project bdc35841c97f)`. CMake 3.31.6, Ninja, host gcc 13.3.
Fresh full vcpkg clone + `bootstrap-vcpkg.sh -disableMetrics` in the job
directory (configure incl. vcpkg deps: 707 s; ftp.gnu.org timed out for
automake-1.17 once, vcpkg's retry got it). Fork clone fresh from
`moat-fix-145`, `git rev-parse HEAD` = head_sha, protect-fork installed; no
modified tracked files at completion (untracked `build/`, `build-cuda/`,
`UnitTests.log`, `cuda_tests.log` only).

### Build

```
export ROCM=$(rocm-sdk path --root)        # job venv, 10.2.0a20261022
export VCPKG_ROOT=<job>/agent_space/vcpkg
cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=${ROCM}/lib/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=${ROCM} -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build -j16
```

PASS, 417/417 steps, no errors under `-Werror`.

### HipCxx23Shim is inert on Linux (the point of this delta)

- Configure output: no `Using reordered HIP runtime wrapper` line, no shim warning.
- `build/CMakeFiles/CMakeConfigureLog.yaml`: exactly one try_compile from `HipCxx23Shim.cmake:24` (the first probe), compiled at `-std=gnu++23 --offload-arch=gfx1100` with the project's `-include cuda_to_hip.h`, exit code 0. No try_compile from the second (re-probe) call.
- `build/HipCxx23Shim/` does not exist: no reordered wrapper generated.
- build.ninja: 35 HIP objects (FLAGS lines with `--offload-arch`), all `-std=c++23`, none with an `-I...HipCxx23Shim` flag (0 matches); 367 CXX FLAGS lines at `-std=c++23`, no other `-std=`. The only two mentions of `HipCxx23Shim` in build.ninja are the module listed as a CMake re-run dependency. `CMAKE_HIP_FLAGS` cache entry empty. HIP FLAGS line: `-include ".../cuda_to_hip.h" -O3 -DNDEBUG -std=c++23 --offload-arch=gfx1100 -fPIC -Wall ... -Werror ...`, i.e. the 29bac0b9 flags.
- Two independent reasons the probe passes here: libstdc++ 13 `<cmath>` is not constexpr, and the 10.2 SDK's `__clang_hip_runtime_wrapper.h` already has the fixed order (`#include <__clang_cuda_math_forward_declares.h>` at line 102, `#include <cmath>` at 103), so this SDK carries llvm/llvm-project#201563.

### Tests (`utils/timeit.sh CubbyFlow test -- bash agent_space/run_<name>.sh`, each `ulimit -c 0; HIP_VISIBLE_DEVICES=0`, from `build/`)

- `./bin/CUDATests`: 35/35 test cases, 3170/3170 assertions PASS.
- `./bin/CUDASPHSim -f 5`: 5 frames, 13824 particles, exit 0; frame 4 has 0 NaN/inf, mean z 0.20536 (x,y FLT_MAX as before), same as the windows-gfx1151 porter run.
- `./bin/UnitTests` (CPU): 814/814 PASS in 168 suites.

All counts match the 29bac0b9 evidence exactly.

### CUDA no-regression gate

`cuda-not-validated` on this host: no CUDA toolkit (no nvcc, no conda). Not needed for this delta: everything aeb9540d adds is inside `if (USE_HIP)` and `CompileOptions.cmake` is byte-identical to 29bac0b9, so a CUDA configure is identical to 29bac0b9, whose nvcc gate passed (linux-gfx1100 revalidation 2026-08-20).

### Jargon / documentation

`python3 utils/jargon.py --port CubbyFlow`: clean. `Documents/Install.md` HIP/ROCm build section unchanged by this delta and still matches the build command.
ROCM=$(<venv>/bin/rocm-sdk path --root); export VCPKG_ROOT=<job>/vcpkg
cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx950 \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=$ROCM -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build -j32
```

PASS: 417/417 steps, 0 warnings, 0 errors (under -Werror).

### HipCxx23Shim on Linux (the delta since 29bac0b9): inert, as intended

- `CMakeConfigureLog.yaml`: exactly one try_compile from
  `Builds/CMake/HipCxx23Shim.cmake:24` (variable `PROBE_OK`, `exitCode: 0`;
  probe compiled `-std=gnu++23 --offload-arch=gfx950` with the inherited
  `-include cuda_to_hip.h`). No second (shim) try_compile.
- No "Using reordered HIP runtime wrapper" message and no warning in the
  configure output; `build/HipCxx23Shim/` does not exist.
- `build.ninja`: 35 HIP objects (every `--offload-arch` FLAGS line), all at
  `-std=c++23`; 367 other FLAGS lines at `-std=c++23`; no other `-std=`.
  No `-I...HipCxx23Shim` anywhere; the only `HipCxx23Shim` mentions are the
  module file in the RERUN_CMAKE dependency list. `CMAKE_HIP_FLAGS` cache
  value empty.
- Aside: ROCm 10.2's clang 24 already ships the fixed wrapper order
  (`__clang_hip_runtime_wrapper.h:102` forward-declares, `:103` `<cmath>`), so
  the probe would also pass on Windows with this ROCm.

### GPU tests (`ulimit -c 0`)

- `./build/bin/CUDATests`: 35/35 test cases, 3170/3170 assertions PASS.
- `CUDASPHSim -f 5` (run from a scratch dir): 13824 particles, 5 frames
  written, exit 0; 0 non-finite values in all frames; mean z per frame
  0.25000, 0.23641, 0.21858, 0.20843, 0.20536 (final value matches the
  windows-gfx1151 porter run); x,y = FLT_MAX as on every other platform.

### CPU regression tests

`./build/bin/UnitTests`: 814 tests from 168 suites, all PASS (matches the
gfx90a/gfx1100/gfx1151 counts).

### CUDA compile check at aeb9540d (nvcc, compile only, no NVIDIA GPU)

```
export VCPKG_ROOT=<job>/vcpkg; export PATH=/usr/local/cuda-12.8/bin:$PATH
cmake -B build-cuda -S . -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build-cuda -j32 --target CubbyFlow CUDATests CUDASPHSim
```

nvcc 12.8 (V12.8.93), host gcc 13.3. PASS: 184/184 steps, 21 NVCC device
objects, CubbyFlow, CUDATests and CUDASPHSim linked, 0 errors. As in the
2026-08-20 runs, the project's FindCUDA path uses
`-gencode;arch=compute_75,code=sm_75` regardless of the `-D` pin (a real
modern arch, not the degraded-arch fingerprint). The HIP shim is not
reached (no HipCxx23Shim try_compile in the CUDA configure log, no generated
dir). Two gcc `-Wattributes` warnings ("declared with greater visibility than
the type of its field") from the host pass of
`CUDAParticleSystemData3Tests.cu` / `CUDAPointHashGridSearcher3Tests.cu`,
instantiating `CUDAPointHashGridSearcher3::ForEachNearbyPointFunc` with a
test-local anonymous-namespace callback; neither file nor that header is in
the 29bac0b9..aeb9540d delta, so not a regression. Nothing was run.

### jargon / documentation

`python3 utils/jargon.py --port CubbyFlow`: clean. `Documents/Install.md`
"Building with GPU Acceleration" documents `USE_HIP=ON`; the delta needs no
doc change.

Commands: `bash utils/timeit.sh CubbyFlow compile -- bash agent_space/build_hip.sh`,
`bash utils/timeit.sh CubbyFlow test -- bash agent_space/run_tests.sh cuda|sph|unit`,
`bash utils/timeit.sh CubbyFlow cuda-compile -- bash agent_space/build_cuda.sh`.
All exit 0.

### Verdict: PASS -> completed at `aeb9540d6be87d4fcde76a00ded5f972b4615c35`

## Validation 2026-11-20 (windows-gfx1151, validator): fix-round head aeb9540d

PASS at `aeb9540d6be87d4fcde76a00ded5f972b4615c35` (`moat-fix-145`), AMD Radeon 8060S (gfx1151, wave32, 20 CUs), Windows 11, TheRock ROCm SDK 7.14.0a20260612 (AMD clang 23.0.0git), CMake 3.31.10, Ninja, MSVC 14.50.35717 (VS 2022 developer env scrubbed, per the porter's recipe), vcpkg at D:/Develop/vcpkg. Full rebuild in a FRESH directory `build_val/` (not the porter's `build_gfx1151_b/`), `-j 6`, `ulimit -c 0`, single runs. `git rev-parse HEAD` = head_sha; `git status --porcelain` shows only untracked `build_gfx1151_b/` and `build_val/`.

Commands (scripts in the worktree's agent_space/: `val_cfg.sh` is the porter's `build_cf.sh` with `B=build_val` and no build step; `val_build.sh` runs the build in the same scrubbed environment):

```
utils/timeit.sh CubbyFlow compile -- bash agent_space/val_cfg.sh
utils/timeit.sh CubbyFlow compile -- bash agent_space/val_build.sh   # --target CubbyFlow CUDATests UnitTests CUDASPHSim -j 6
```

Evidence:
- Configure printed `-- Using reordered HIP runtime wrapper for C++23: .../build_val/HipCxx23Shim` (the probe hit the "cannot overload __host__ __device__" diagnostic, the shim re-probe passed).
- build.ninja: 35 HIP objects (`--offload-arch=gfx1151`), all 35 with `-std=c++23` and `-I".../build_val/HipCxx23Shim"` after the `-include cuda_to_hip.h`; 769 `-std=` occurrences, all `c++23`. Build 633/633, 0 warnings, 0 errors (-Werror).
- TheRock amdhip64_7.dll, amd_comgr.dll, hiprtc*.dll, rocm_kpack.dll copied into `build_val/bin` (System32 copies are broken on this host).
- `hipInfo.exe`: AMD Radeon(TM) 8060S Graphics, gcnArchName gfx1151.
- `CUDATests.exe`: 35/35 test cases, 3170/3170 assertions PASS. A second run with `AMD_LOG_LEVEL=3` logged 2376 `hipLaunchKernel` calls returning hipSuccess (log: agent_space/val_gpuproof.log), so the tests ran on the GPU.
- `CUDASPHSim.exe -f 5`: 5 frames, 13824 particles, exit 0, 0 non-finite values; mean z 0.20536 at frame 4 (x,y = FLT_MAX for particle 0, the known empty-container non-issue noted earlier).
- `UnitTests.exe` (CPU): 814/814 PASS (168 suites).
- `python utils/jargon.py --port CubbyFlow`: clean. Docs: Documents/Install.md already carries the USE_HIP build block.
- CUDA no-regression gate: not rerun here; Windows host has no nvcc build of this and the nvcc gate already passed at this head on Linux.

Differences from the porter's run: none in counts (35/35, 3170, 814, mean z 0.20536). Configure took 24 s, build was short (vcpkg cache warm, ~633 steps).
## Validation 2026-11-20 (linux-gfx90a, validator): fix-round head aeb9540d

PASS at `aeb9540d6be87d4fcde76a00ded5f972b4615c35` (`moat-fix-145`), AMD Instinct MI250X / MI250 (gfx90a), `HIP_VISIBLE_DEVICES=1` (the dispatcher's pin), `ulimit -c 0`. Full rebuild and real-GPU run: `classify 29bac0b9..aeb9540d` gives `class=mixed arch_independent=False inert=False`, so no carry-forward.

Toolchain: TheRock ROCm 10.2.0 wheels installed into the job venv, not the image's 10.1.0 and not a system ROCm (the host has no `/opt/rocm`): `$FLEET_VENV/bin/pip install --index-url https://stable.repo.amd.com/rocm/core/whl-next/ 'rocm[libraries,devel,device-gfx90a]==10.2.0'`; `rocm-sdk version` = 10.2.0, `AMD clang version 24.0.0git (ROCm/llvm-project bdc35841c97f)`. CMake 3.31.6, Ninja, g++ 13.3. Fresh full vcpkg clone + `bootstrap-vcpkg.sh -disableMetrics` in the job's `agent_space/` (configure including vcpkg deps: 389 s). Fork clone fresh from `moat-fix-145` (`HEAD` = head_sha), protect-fork installed; `git status --porcelain` empty after build and tests (untracked `build/` only).

### Build

```
ROCM=$($FLEET_VENV/bin/rocm-sdk path --root); export VCPKG_ROOT=<job>/agent_space/vcpkg
cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=$ROCM -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build -j$MAX_JOBS     # 42
```

PASS: 417/417 steps, 0 warnings, 0 errors under `-Werror`.

### HipCxx23Shim on Linux: inert

- Configure output: no `Using reordered HIP runtime wrapper` line, no shim warning.
- `CMakeConfigureLog.yaml`: exactly one try_compile from `Builds/CMake/HipCxx23Shim.cmake:24` (variable `PROBE_OK`, `exitCode: 0`), compiled `-std=gnu++23 --offload-arch=gfx90a` with the inherited `-include cuda_to_hip.h`. No second (re-probe) try_compile.
- `build/HipCxx23Shim/` does not exist: no reordered wrapper generated.
- build.ninja: 35 HIP objects (`--offload-arch` FLAGS lines), all at `-std=c++23`; 367 other FLAGS lines at `-std=c++23`; 402 `-std=` flags in total, all `c++23`. No `-I...HipCxx23Shim` flag; the only two `HipCxx23Shim` mentions are the module in the RERUN_CMAKE dependency lists. `CMAKE_HIP_FLAGS` cache value empty. HIP FLAGS: `-include ".../cuda_to_hip.h" -O3 -DNDEBUG -std=c++23 --offload-arch=gfx90a -fPIC -Wall ... -Werror ...`, the 29bac0b9 flags.
- As on gfx950, this SDK's `__clang_hip_runtime_wrapper.h` already has the fixed order (forward-declares at line 102, `<cmath>` at 103).

### Tests (`bash utils/timeit.sh CubbyFlow test -- bash agent_space/run_tests.sh cuda|sph|unit`, each under `ulimit -c 0`, `HIP_VISIBLE_DEVICES=1`)

- `build/bin/CUDATests`: 35/35 test cases, 3170/3170 assertions PASS.
- `build/bin/CUDASPHSim -f 5` (run from a scratch dir): 13824 particles, 5 frames, exit 0; 0 non-finite values; mean z per frame 0.25000, 0.23641, 0.21858, 0.20843, 0.20536, identical to gfx950 and the windows-gfx1151 porter run.
- `build/bin/UnitTests` (CPU): 814/814 PASS in 168 suites.

All counts match the 29bac0b9 gfx90a evidence and the aeb9540d gfx1100/gfx950 runs.

### CUDA no-regression gate

Not re-run: no CUDA toolkit on this host, and the nvcc 12.8 compile gate already passed at this head_sha (linux-gfx950 validation above).

### Jargon / documentation

`python3 utils/jargon.py --port CubbyFlow`: clean. `Documents/Install.md:104-109` documents `USE_HIP=ON`; unchanged by this delta.

### Verdict: PASS -> completed at `aeb9540d6be87d4fcde76a00ded5f972b4615c35`
