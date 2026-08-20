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
