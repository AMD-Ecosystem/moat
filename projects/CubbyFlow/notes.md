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
