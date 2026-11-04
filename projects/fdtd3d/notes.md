# fdtd3d notes

## Build

### HIP/ROCm build (gfx90a)
```bash
cd /var/lib/jenkins/moat/projects/fdtd3d/src
mkdir build && cd build
cmake .. \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSOLVER_DIM_MODES=DIM3 \
  -DVALUE_TYPE=d \
  -DCOMPLEX_FIELD_VALUES=ON \
  -DPRINT_MESSAGE=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build . -j$(nproc)
```

### Test commands
```bash
# Unit test (requires device ID argument)
./Source/UnitTests/unit-test-cuda-grid 0

# 3D simulation
./Source/fdtd3d --3d --size x:20,y:20,z:20 --use-cuda --cuda-gpus 0 --time-steps 100
```

## Port notes

### HIP-specific changes
1. **Compat header**: `Source/Helpers/cuda_to_hip.h` aliases CUDA runtime APIs to HIP
2. **PAssert.h host/device handling**: HIP validates `__device__` function bodies during host compilation, unlike NVCC. We provide a unified `__host__ __device__` wrapper for `program_fail` and simplified DPRINTF that skips SOLVER_SETTINGS checks to avoid host/device symbol visibility issues.
3. **Template keyword**: Added `template` keyword before dependent template member function calls in `InternalScheme.inc.h` -- required by HIP/clang but NVCC was lenient.
4. **Separable compilation**: All HIP libraries and executables use `HIP_SEPARABLE_COMPILATION ON` for cross-TU device code linking.
5. **Test exclusions**: Some CPU unit tests require all DIM modes or MPI; they are excluded from HIP builds when those conditions are not met.

### Verified on
- AMD Instinct MI250X (gfx90a)
- ROCm 7.2.1

## Install as a dependency

N/A - fdtd3d is an end-user application, not a library.

## Review 2026-06-05

### Summary

This port adds HIP support for AMD GPUs using Strategy A (compat header + LANGUAGE HIP). It is a clean, minimal-footprint port of a pure CMake project with no warp-level primitives, no textures, and no external CUDA libraries. The port follows MOAT best practices.

### Port Correctness

No issues. The port correctly:
- Uses a single `cuda_to_hip.h` compat header with CUDA->HIP symbol aliases
- Marks `.cu` files as `LANGUAGE HIP` instead of renaming
- Adds `template` keyword for dependent template member calls (required by clang, valid C++ that nvcc also accepts)
- Handles HIP's stricter host/device code separation in PAssert.h via a three-way `#if __HIPCC__ / #elif __CUDACC__ / #else` pattern that preserves the original CUDA path
- Sets `__CUDA_ARCH__` to 600 on HIP so native `atomicAdd(double*)` is used

### Fault Classes

No issues. This project has no fault class exposure:
- No warp-level primitives (`__shfl*`, `__ballot`, etc.) -- no wave64/wave32 hazard
- No hardcoded 32 for warp size (the `32` in unit-test-cuda-grid.cu is a grid dimension)
- No textures or surfaces -- no rule-of-five or pitch alignment concerns
- No external CUDA libraries -- no library swap concerns
- No OOB neighbor reads -- the FDTD stencils are within grid bounds

### Minimal Footprint

No issues. Host C++ is untouched. Changes are confined to:
- CMakeLists.txt files (CMake USE_HIP gating)
- cuda_to_hip.h (new compat header)
- PAssert.h (HIP-specific host/device handling)
- InternalScheme.inc.h (template keyword -- valid C++, identical behavior on CUDA)
- CudaInclude.h (one line to include compat header)

### Build System

No issues. Uses `enable_language(HIP)` gated by `USE_HIP` option (default OFF). `CMAKE_HIP_ARCHITECTURES` defaults to gfx90a when unset but accepts any arch via cache variable.

### Commit Hygiene

No issues:
- Title `[ROCm] Add HIP support for AMD GPUs` is 35 chars with correct prefix
- Body explains the port, mentions Claude, has a Test Plan section
- No `Co-Authored-By: noreply` trailer
- No MOAT jargon in upstream-visible text
- No AMD-internal account references

### Backward Compatibility

No issues. The CUDA path is preserved via `elseif ("${CUDA_ENABLED}")` in CMake and `#elif defined(__CUDACC__)` in PAssert.h.

### Recommendation

**Approve** -- The port is correct, minimal, and ready for validation.

## Validation 2026-06-05

### Platform: linux-gfx90a (AMD Instinct MI250X)

Built from scratch at commit baae8b3c91db20e5a45c332500c8da2200941fe0.

### Build
```bash
cd /var/lib/jenkins/moat/projects/fdtd3d/src/build
HIP_VISIBLE_DEVICES=3 cmake .. \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSOLVER_DIM_MODES=DIM3 \
  -DVALUE_TYPE=d \
  -DCOMPLEX_FIELD_VALUES=ON \
  -DPRINT_MESSAGE=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build . -j$(nproc)
```
Build time: 312.8s (5.2 minutes)

### Test Results
All tests passed on real GPU hardware.

1. **GPU unit test**: `unit-test-cuda-grid 0`
   - Status: PASS
   - Duration: 0.255s
   - Tests grid operations on device

2. **3D electromagnetic simulation**: 
   ```
   ./Source/fdtd3d --3d --size x:20,y:20,z:20 --use-cuda --cuda-gpus 0 --time-steps 100
   ```
   - Status: PASS
   - Duration: 0.587s (simulation runtime: 0.277s)
   - Completed all 100 time steps
   - Grid size: 20x20x20
   - Device: AMD Instinct MI250X
   - No numerical errors

### Summary
Port validated successfully on gfx90a. Both GPU tests pass with correct numerical output.

## Validation 2026-06-05 (gfx1100)

### Platform: linux-gfx1100 (AMD Radeon Pro W7800 48GB)

Built from scratch at commit baae8b3c91db20e5a45c332500c8da2200941fe0.

### Build
```bash
HIP_VISIBLE_DEVICES=0 cmake /var/lib/jenkins/moat/projects/fdtd3d/src \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSOLVER_DIM_MODES=DIM3 \
  -DVALUE_TYPE=d \
  -DCOMPLEX_FIELD_VALUES=ON \
  -DPRINT_MESSAGE=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -B/var/lib/jenkins/moat/projects/fdtd3d/src/build
cmake --build /var/lib/jenkins/moat/projects/fdtd3d/src/build -j$(nproc)
```
Build completed successfully.

### Test Results
All tests passed on real GPU hardware.

1. **GPU unit test**: `unit-test-cuda-grid 0`
   - Status: PASS
   - Grid operations on device verified

2. **3D electromagnetic simulation**: 
   ```
   ./Source/fdtd3d --3d --size x:20,y:20,z:20 --use-cuda --cuda-gpus 0 --time-steps 100
   ```
   - Status: PASS
   - Duration: 0.216s simulation runtime
   - Completed all 100 time steps
   - Grid size: 20x20x20
   - Device: AMD Radeon Pro W7800 48GB (gfx1100)
   - No numerical errors

### Summary
Port validated successfully on gfx1100. Both GPU tests pass with correct numerical output.

## Validation 2026-06-08 (windows-gfx1201)

### Platform: windows-gfx1201 (AMD Radeon RX 9070 XT, RDNA4)

Built from scratch at commit 049a6237251dc27e9b5273fb1e18aa722ddd9f5f (adds Windows
POSIX-header fixes on top of the original port at baae8b3c).

### Windows-specific fixes required (new commit on fork)

Three POSIX-only system headers are absent on Windows MSVC/clang ABI:

1. `Source/Settings/Settings.cpp`: `<alloca.h>` -> `<malloc.h>` on Windows via `#ifdef _WIN32`
2. `Source/Grid/Clock.h`: `clock_gettime(CLOCK_MONOTONIC, ...)` not available; shimmed via `timespec_get` (C11/UCRT)
3. `Source/main.cpp`: `<sys/time.h>` / `gettimeofday` not available; shimmed via `timespec_get`

Additionally, two CMake configuration adjustments needed:
- `-G Ninja` (HIP language is not supported by Visual Studio generator)
- `-DCMAKE_CXX_FLAGS="-D_USE_MATH_DEFINES -DWIN32"` and `-DCMAKE_HIP_FLAGS=...` (M_PI requires _USE_MATH_DEFINES)
- `-DCMAKE_HIP_USING_LINKER_DEFAULT="-fuse-ld=lld"` (note: overridden by platform module; must patch build.ninja directly)

The lld-link issue: CMake's `Platform/Windows-Clang.cmake` sets `-fuse-ld=lld-link` as the HIP linker
default, but `-fuse-ld=lld-link` is rejected when clang invokes `--hip-link` (offload bundler) mode.
Replace with `-fuse-ld=lld` in the generated `build.ninja` after cmake configuration.

### Build
```bash
ROCM_SDK="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
CLANGPP="$ROCM_SDK/lib/llvm/bin/clang++.exe"

cmake B:/develop/moat/projects/fdtd3d/src \
  -G Ninja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSOLVER_DIM_MODES=DIM3 \
  -DVALUE_TYPE=d \
  -DCOMPLEX_FIELD_VALUES=ON \
  -DPRINT_MESSAGE=ON \
  -DCMAKE_HIP_COMPILER="$CLANGPP" \
  -DCMAKE_C_COMPILER="$CLANGPP" \
  -DCMAKE_CXX_COMPILER="$CLANGPP" \
  -DCMAKE_PREFIX_PATH="$ROCM_SDK" \
  -DCMAKE_CXX_FLAGS="-D_USE_MATH_DEFINES -DWIN32" \
  -DCMAKE_HIP_FLAGS="-D_USE_MATH_DEFINES -DWIN32" \
  -B build-gfx1201

# Patch the generated build files: -fuse-ld=lld-link -> -fuse-ld=lld
sed -i 's/-fuse-ld=lld-link/-fuse-ld=lld/g' build-gfx1201/build.ninja

HIP_VISIBLE_DEVICES=0 cmake --build build-gfx1201 -j24
```
Note: EasyBMP zip must be downloaded and extracted first (the CMake wget/unzip does not work on Windows):
```bash
cd src/Third-party/EasyBMP && curl -k -L "https://github.com/zer011b/EasyBMP/archive/v1.6.zip" -o EasyBMP.zip && unzip -q EasyBMP.zip && mv EasyBMP-1.6/source . && rm -rf EasyBMP-1.6 EasyBMP.zip
```

Build completed successfully (65/65 targets).

### Test Results
All tests passed on real GPU hardware (HIP_VISIBLE_DEVICES=0, AMD Radeon RX 9070 XT gfx1201).

1. **GPU unit test**: `unit-test-cuda-grid.exe 0`
   - Status: PASS (exit 0)
   - Creates and operates on CudaGrid objects on device
   - Device: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32)

2. **3D electromagnetic simulation**:
   ```
   ./Source/fdtd3d.exe --3d --size x:20,y:20,z:20 --use-cuda --cuda-gpus 0 --time-steps 100
   ```
   - Status: PASS (exit 0)
   - All 100 time steps completed
   - Total time = 0.054948 seconds
   - Grid size: 20x20x20
   - Device: AMD Radeon RX 9070 XT (gfx1201)
   - No numerical errors

### Summary
Port validated successfully on gfx1201 (RDNA4). Both GPU tests pass with correct numerical output.
Three Windows-specific POSIX compatibility patches were required (committed as a follow-on commit);
the lld-link->lld linker substitution is a build.ninja post-process step (CMake platform module
always overrides the cache variable).

Note: linux-gfx90a and linux-gfx1100 were flipped to `revalidate` by advance_head since the
Windows fixes classify as `mixed`. The `#ifdef _WIN32` guards are zero-effect on Linux (the `#else`
branches replicate the original code exactly), so those platforms should carry forward via
codeobj_diff binary equivalence check.

## Validation 2026-06-08 (linux-gfx90a revalidate)

### Platform: linux-gfx90a (AMD Instinct MI250X, GCD 1)

Revalidation at head 049a6237251dc27e9b5273fb1e18aa722ddd9f5f (Windows POSIX fixes commit on top of baae8b3c).

### Change classification

Delta is one commit: "[ROCm] Fix Windows build: POSIX headers missing on MSVC/clang target". All changes are `#ifdef _WIN32 ... #else ... #endif` guards where the `#else` branch replicates the original Linux code exactly (`alloca.h`, `sys/time.h`, `clock_gettime` usage). Zero effect on Linux/gfx90a.

`moatlib.py classify` returned `unknown` (mixed delta -- tool could not auto-classify). `codeobj_diff.py` returned `indeterminate` (the project produces executables, not shared libs -- no `.so` entry points found). Per carry-forward rules, an `indeterminate` verdict requires full GPU revalidation.

### Build

Built new SHA (049a6237) from scratch at the same config as prior validation:

```bash
cmake /var/lib/jenkins/moat/projects/fdtd3d/src \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSOLVER_DIM_MODES=DIM3 \
  -DVALUE_TYPE=d \
  -DCOMPLEX_FIELD_VALUES=ON \
  -DPRINT_MESSAGE=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -B /var/lib/jenkins/moat/projects/fdtd3d/build-new
cmake --build /var/lib/jenkins/moat/projects/fdtd3d/build-new -j$(nproc)
```
Build time: ~321s

### Test Results

All tests passed on real GPU hardware (HIP_VISIBLE_DEVICES=1, AMD Instinct MI250X / MI250, gfx90a).

1. **GPU unit test**: `unit-test-cuda-grid 0`
   - Status: PASS
   - Grid operations on device verified

2. **3D electromagnetic simulation**:
   ```
   HIP_VISIBLE_DEVICES=1 ./Source/fdtd3d --3d --size x:20,y:20,z:20 --use-cuda --cuda-gpus 0 --time-steps 100
   ```
   - Status: PASS
   - Total time = 0.276s (100 time steps, 20x20x20 grid)
   - Device: AMD Instinct MI250X / MI250

3. **CPU unit tests** (non-GPU regression check):
   - unit-test-settings: PASS
   - unit-test-clock: PASS
   - unit-test-complex: PASS
   - unit-test-coordinate: PASS
   - unit-test-approximation: PASS
   - unit-test-grid: PASS
   - unit-test-layout: PASS

Test suite wall time: 148.6s

### Summary

All GPU and CPU tests pass at head 049a6237. No regressions. The Windows POSIX compatibility fixes have zero effect on Linux, confirming expected arch-independence.

## Validation 2026-06-19 (windows-gfx1101)

### Platform: windows-gfx1101 (AMD Radeon PRO V710, RDNA3/gfx1101)

Validated at commit 049a6237251dc27e9b5273fb1e18aa722ddd9f5f. Device confirmed at `HIP_VISIBLE_DEVICES=1` (mapping verified with hipInfo at session start).

### Build

```bash
ROCM_SDK="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
CLANGPP="$ROCM_SDK/lib/llvm/bin/clang++.exe"

# EasyBMP already extracted from gfx1201 validation
cmake B:/develop/moat/projects/fdtd3d/src \
  -G Ninja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSOLVER_DIM_MODES=DIM3 \
  -DVALUE_TYPE=d \
  -DCOMPLEX_FIELD_VALUES=ON \
  -DPRINT_MESSAGE=ON \
  -DCMAKE_HIP_COMPILER="$CLANGPP" \
  -DCMAKE_C_COMPILER="$CLANGPP" \
  -DCMAKE_CXX_COMPILER="$CLANGPP" \
  -DCMAKE_PREFIX_PATH="$ROCM_SDK" \
  -DCMAKE_CXX_FLAGS="-D_USE_MATH_DEFINES -DWIN32" \
  -DCMAKE_HIP_FLAGS="-D_USE_MATH_DEFINES -DWIN32" \
  -B build-gfx1101

# Patch the generated build files: -fuse-ld=lld-link -> -fuse-ld=lld
sed -i 's/-fuse-ld=lld-link/-fuse-ld=lld/g' build-gfx1101/build.ninja

HIP_VISIBLE_DEVICES=1 cmake --build build-gfx1101 -j64
```

Build completed successfully (65/65 targets). Same warnings as gfx1201 (memcpy on non-trivially copyable type), no errors.

TheRock runtime DLLs (amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll, hiprtc-builtins0714.dll) copied from `_rocm_sdk_core/bin` into Source/ and Source/UnitTests/ to beat System32.

### Test Results

All tests passed on real GPU hardware (HIP_VISIBLE_DEVICES=1, AMD Radeon PRO V710, gfx1101).

1. **GPU unit test**: `unit-test-cuda-grid.exe 0`
   - Status: PASS (exit 0)
   - Creates and operates on CudaGrid objects on device
   - Device: AMD Radeon PRO V710 (gfx1101, RDNA3, wave32)

2. **3D electromagnetic simulation**:
   ```
   HIP_VISIBLE_DEVICES=1 fdtd3d.exe --3d --size x:20,y:20,z:20 --use-cuda --cuda-gpus 0 --time-steps 100
   ```
   - Status: PASS (exit 0)
   - All 100 time steps completed
   - Total time = 0.072957 seconds
   - Grid size: 20x20x20
   - Device: AMD Radeon PRO V710 (gfx1101)
   - No numerical errors
   - Phase velocity match: 1.00000000000000 (identical to other platforms)

3. **CPU unit tests** (non-GPU regression check):
   - unit-test-settings: PASS
   - unit-test-clock: PASS
   - unit-test-complex: PASS
   - unit-test-coordinate: PASS
   - unit-test-approximation: PASS
   - unit-test-grid: PASS
   - unit-test-layout: PASS

### Summary

Port validated successfully on gfx1101 (RDNA3). Both GPU tests and all 7 CPU unit tests pass. No TDR events. GPU remained healthy throughout. No gfx1101-specific issues found -- identical build recipe to gfx1201 except `--offload-arch=gfx1101`.

## Validation 2026-06-08 (linux-gfx1100 revalidate)

### Platform: linux-gfx1100 (AMD Radeon Pro W7800 48GB, gfx1100)

Revalidation at head 049a6237251dc27e9b5273fb1e18aa722ddd9f5f (Windows POSIX fixes commit on top of baae8b3c). Validated_sha was baae8b3c.

### Change classification

Delta is one commit: "[ROCm] Fix Windows build: POSIX headers missing on MSVC/clang target". All changes are `#ifdef _WIN32 ... #else ... #endif` guards where the `#else` branch replicates the original Linux code exactly. Zero effect on Linux/gfx1100. moatlib classify returned `mixed` (not auto-inert); codeobj_diff returns `indeterminate` for executables (no shared-lib exports), so full GPU revalidation was performed per protocol.

### Build

```bash
export HIP_VISIBLE_DEVICES=1
cmake /var/lib/jenkins/moat/projects/fdtd3d/src \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSOLVER_DIM_MODES=DIM3 \
  -DVALUE_TYPE=d \
  -DCOMPLEX_FIELD_VALUES=ON \
  -DPRINT_MESSAGE=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -B /var/lib/jenkins/moat/agent_space/fdtd3d-gfx1100-gpu1/build-head
cmake --build /var/lib/jenkins/moat/agent_space/fdtd3d-gfx1100-gpu1/build-head -j$(nproc)
```
Build completed successfully (warnings only, no errors).

### Test Results

All tests passed on real GPU hardware (HIP_VISIBLE_DEVICES=1, AMD Radeon Pro W7800 48GB, gfx1100).

1. **GPU unit test**: `unit-test-cuda-grid 0`
   - Status: PASS
   - Grid operations on device verified

2. **3D electromagnetic simulation**:
   ```
   HIP_VISIBLE_DEVICES=1 ./Source/fdtd3d --3d --size x:20,y:20,z:20 --use-cuda --cuda-gpus 0 --time-steps 100
   ```
   - Status: PASS
   - Total time = 0.198460 seconds (100 time steps, 20x20x20 grid)
   - Device: AMD Radeon Pro W7800 48GB (gfx1100)
   - No numerical errors

3. **CPU unit tests** (non-GPU regression check):
   - unit-test-settings: PASS
   - unit-test-clock: PASS
   - unit-test-complex: PASS
   - unit-test-coordinate: PASS
   - unit-test-approximation: PASS
   - unit-test-grid: PASS
   - unit-test-layout: PASS

### Summary

All GPU and CPU tests pass at head 049a6237 on gfx1100. No regressions. The Windows POSIX compatibility fixes have zero effect on Linux/gfx1100.
