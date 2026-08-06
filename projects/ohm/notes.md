# ohm notes

## Port summary

HIP backend added to ohm for AMD GPU support. The project has an OpenCL-style kernel abstraction: kernels are written in .cl files using OpenCL semantics, compiled via a compat header that maps OpenCL primitives to the backend (CUDA or HIP).

### Changes
- New `cmake/OhmHip.cmake` for HIP feature detection
- New `gputil/hip/` directory with HIP implementation (mirrors `gputil/cuda/`)
- New `.hip` kernel wrappers in `ohmgpu/gpu/` that include `.cl` files via HIP compat header
- Updated CMake files to support `OHM_FEATURE_HIP` option
- Fixed two upstream bugs: missing `#include <utility>` in `PlyPointStream.cpp` and `VoxelBuffer.cpp`

### Build (gfx90a)

```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DOHM_FEATURE_HIP=ON \
  -DOHM_FEATURE_CUDA=OFF \
  -DOHM_FEATURE_OPENCL=OFF \
  -DOHM_FEATURE_TEST=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a

cmake --build build -j$(nproc)
```

### Test

```bash
HIP_VISIBLE_DEVICES=3 ./build/bin/gputiltesthip  # 8/8 tests PASS
HIP_VISIBLE_DEVICES=3 ./build/bin/ohmtesthip     # 58/58 tests PASS
```

## Key implementation notes

1. The project already has excellent GPU abstraction -- CUDA and OpenCL share the same .cl kernel files. HIP follows the same pattern.

2. HIP provides vector operators for its vector types (float3, int3, etc.), so hutil_math.h disables its operator definitions to avoid ambiguity. Only the math functions (clamp, dot, length, normalize) are kept.

3. The `__local` macro in HIP conflicts with our definition; this causes harmless warnings.

4. HIP sources are identified by `.hip` extension and compiled with `LANGUAGE HIP`.

## Review 2026-06-05

**Reviewer**: MOAT reviewer agent

**Verdict**: APPROVE

The port correctly adds a HIP backend that mirrors the existing CUDA backend architecture. Key observations:

1. **Strategy**: Appropriate adaptation of Strategy A -- instead of a single cuda_to_hip.h compat header, this project uses its existing OpenCL-style abstraction layer with a parallel `gputil/hip/` directory containing `hutil_importcl.h` (analogous to `cutil_importcl.h` for CUDA).

2. **Fault classes**: No warp intrinsics, no hardcoded warp size, no textures/surfaces. Atomics correctly mapped. The float CAS (`gputilAtomicCasF32`) uses `__float_as_int`/`__int_as_float` which are available in HIP.

3. **hutil_math.h**: Correctly disables vector operators (`#if 0` block) that would conflict with HIP's built-in operators, while keeping math functions (dot, length, normalize, clamp).

4. **CMake**: OhmHip.cmake properly uses `check_language(HIP)`, defaults `CMAKE_HIP_ARCHITECTURES` only when unset (allowing follower platforms to override), and gates all HIP dependencies.

5. **Upstream fixes**: The two `#include <utility>` additions are legitimate portability fixes.

6. **Commit hygiene**: Title 43 chars, properly prefixed `[ROCm]`, mentions Claude, no noreply trailer, no MOAT jargon.

No issues found. Ready for validation.

## Validation 2026-06-05 (linux-gfx90a)

**GPU arch**: gfx90a (MI250X)  
**ROCm version**: 7.2.1  
**Build command**:
```bash
cmake -B build_gfx90a -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DOHM_FEATURE_HIP=ON \
  -DOHM_FEATURE_CUDA=OFF \
  -DOHM_FEATURE_OPENCL=OFF \
  -DOHM_FEATURE_TEST=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a

cmake --build build_gfx90a -j$(nproc)
```

**Test results**:
- `gputiltesthip`: 8/8 tests PASSED
- `ohmtesthip`: 58/58 tests PASSED

**Total**: 66/66 tests PASSED

All GPU tests executed successfully on real hardware with no failures.

## Validation 2026-06-05 (linux-gfx1100)

**GPU arch**: gfx1100 (Navi31)  
**ROCm version**: 7.2.1  
**Build command**:
```bash
/usr/bin/cmake -B build_gfx1100 -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DOHM_FEATURE_HIP=ON \
  -DOHM_FEATURE_CUDA=OFF \
  -DOHM_FEATURE_OPENCL=OFF \
  -DOHM_FEATURE_TEST=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100

/usr/bin/cmake --build build_gfx1100 -j$(nproc)
```

**Test results**:
- `gputiltesthip`: 8/8 tests PASSED
- `ohmtesthip`: 58/58 tests PASSED

**Total**: 66/66 tests PASSED

All GPU tests executed successfully on real hardware with no failures. Note: Used system cmake (/usr/bin/cmake 3.28.3) instead of conda cmake to avoid googletest compatibility issues.

## Validation 2026-06-08 (windows-gfx1201)

**GPU**: AMD Radeon RX 9070 XT, gfx1201 (RDNA4, wave32), HIP_VISIBLE_DEVICES=0
**ROCm**: 7.14.0a20260604 (TheRock nightly)
**Fork**: AMD-Ecosystem/ohm moat-port HEAD 80ea8a8 (09782e7 + Windows iomanip fix)

### Windows-specific source fix (committed on top of port)

Six test files used `std::setw`/`std::setfill` without `#include <iomanip>`. On Linux these were available transitively; MSVC-ABI clang++ on Windows requires the explicit include. Added `#include <iomanip>` to:
- tests/gputiltest/KernelTest.cpp
- tests/ohmtest/RayPatternTests.cpp
- tests/ohmtest/VoxelMeanTests.cpp
- tests/ohmtestgpu/GpuVoxelMeanTests.cpp
- tests/ohmtestgpu/GpuLineKeysTests.cpp
- tests/ohmtestgpu/GpuMapTest.cpp

### Build

```bash
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
ROCM_CORE=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_core
VCPKG=B:/develop/moat/projects/alien/src/build/vcpkg_installed/x64-windows

cmake -B agent_space/ohm_gfx1201_build -S projects/ohm/src \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$ROCM/lib/llvm/bin/clang.exe" \
  -DCMAKE_CXX_COMPILER="$ROCM/lib/llvm/bin/clang++.exe" \
  -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/clang++.exe" \
  -DCMAKE_PREFIX_PATH="$ROCM;$ROCM_CORE;$VCPKG" \
  -DOHM_FEATURE_HIP=ON -DOHM_FEATURE_CUDA=OFF -DOHM_FEATURE_OPENCL=OFF \
  -DOHM_FEATURE_TEST=ON -DOHM_FEATURE_EIGEN=ON -DOHM_FEATURE_THREADS=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  "-DGLM_INCLUDE_DIRS=B:/develop/moat/agent_space/glm/glm" \
  -Dglm_FOUND=TRUE \
  -DOHM_SYSTEM_GTEST=ON \
  "-DGTest_DIR=$VCPKG/share/gtest" \
  "-DEigen3_DIR=B:/develop/agent_space/eigen_install/share/eigen3/cmake" \
  -DCMAKE_CXX_FLAGS="-D_USE_MATH_DEFINES -DNOMINMAX -DWIN32" \
  -DCMAKE_C_FLAGS="-D_USE_MATH_DEFINES"

# Strip -fuse-ld=lld-link (CMake 4.3 injects it; amdclang++ device-link rejects it)
sed -i 's/-fuse-ld=lld-link//g' agent_space/ohm_gfx1201_build/build.ninja

cmake --build agent_space/ohm_gfx1201_build -j24
```

217/217 targets built cleanly. HIP compiler: amdclang++ 23.0.0 (gfx1201).

Runtime DLLs copied to bin/: amdhip64_7.dll, amd_comgr.dll, hiprtc0714.dll, hiprtc-builtins0714.dll, rocm_kpack.dll, gtest.dll, gtest_main.dll, zlib1.dll.

### Test results

```bash
HIP_VISIBLE_DEVICES=0 ctest --test-dir agent_space/ohm_gfx1201_build -j1
```

```
1/5: gputiltesthip ..... Passed  4.28 sec   (8/8 GPU tests)
2/5: ohmtest ........... Passed 11.14 sec   (53/53 CPU tests)
3/5: ohmtesthip ........ Passed 57.77 sec   (58/58 GPU tests)
4/5: ohmtestheightmap .. Passed  3.65 sec
5/5: slamiotest ........ Passed  0.09 sec
100% tests passed, 0 tests failed out of 5
```

**GPU tests: 66/66 PASSED (8 gputiltesthip + 58 ohmtesthip)**
**All tests: 5/5 executables PASSED**

All occupancy-map population, ray integration, line query, NDT, traversal, and TSDF GPU tests pass on gfx1201 (RDNA4 wave32). Results match prior linux-gfx90a and linux-gfx1100 validations.

## Validation 2026-06-08 (linux-gfx90a revalidate at 80ea8a81)

**GPU arch**: gfx90a (MI250X, GCD 3)
**ROCm version**: 7.2.1
**Fork HEAD**: 80ea8a81 (09782e7f + `#include <iomanip>` Windows portability fix in 6 test files)

**Delta class**: Source classifier returned `unknown` (forced full GPU revalidation). The delta is 6 x `#include <iomanip>` additions in test .cpp files only -- no kernel or GPU-abstraction source changes.

**Build command**:
```bash
/usr/bin/cmake -B agent_space/ohm_build_gfx90a -S projects/ohm/src \
  -DCMAKE_BUILD_TYPE=Release \
  -DOHM_FEATURE_HIP=ON \
  -DOHM_FEATURE_CUDA=OFF \
  -DOHM_FEATURE_OPENCL=OFF \
  -DOHM_FEATURE_TEST=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a

/usr/bin/cmake --build agent_space/ohm_build_gfx90a -j$(nproc)
```

**Test results**:
- `gputiltesthip`: 8/8 PASSED
- `ohmtesthip`: 58/58 PASSED
- `ohmtest` (CPU): 53/53 PASSED
- `slamiotest` (CPU): 4/4 PASSED

**Total**: 66/66 GPU tests PASSED, 57/57 CPU tests PASSED

All GPU and CPU tests pass. No regressions versus prior validation at 09782e7f.

## Validation 2026-06-08 (linux-gfx1100 revalidate at 80ea8a81 -- carry-forward)

**GPU arch**: gfx1100 (Navi31, GPU index 3)
**ROCm version**: 7.2.1
**Fork HEAD**: 80ea8a81 (09782e7f + `#include <iomanip>` Windows portability fix in 6 test files)

**Delta**: 6 x `#include <iomanip>` additions in test .cpp host files only; no GPU kernel or device-abstraction source changes.

**Method**: Binary equivalence via `utils/codeobj_diff.py`. Built both shas at gfx1100 into separate directories and compared GPU binaries:

```bash
export HIP_VISIBLE_DEVICES=3

# Build validated sha (09782e7)
/usr/bin/cmake -B agent_space/ohm-gfx1100-gpu3/build-old -S projects/ohm/src \
  -DCMAKE_BUILD_TYPE=Release -DOHM_FEATURE_HIP=ON -DOHM_FEATURE_CUDA=OFF \
  -DOHM_FEATURE_OPENCL=OFF -DOHM_FEATURE_TEST=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
utils/timeit.sh ohm compile -- /usr/bin/cmake --build agent_space/ohm-gfx1100-gpu3/build-old -j$(nproc)

# Build head sha (80ea8a81)
/usr/bin/cmake -B agent_space/ohm-gfx1100-gpu3/build-new -S projects/ohm/src \
  -DCMAKE_BUILD_TYPE=Release -DOHM_FEATURE_HIP=ON -DOHM_FEATURE_CUDA=OFF \
  -DOHM_FEATURE_OPENCL=OFF -DOHM_FEATURE_TEST=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
utils/timeit.sh ohm compile -- /usr/bin/cmake --build agent_space/ohm-gfx1100-gpu3/build-new -j$(nproc)

# Compare
python3 utils/codeobj_diff.py \
  agent_space/ohm-gfx1100-gpu3/build-old/bin/gputiltesthip \
  agent_space/ohm-gfx1100-gpu3/build-new/bin/gputiltesthip
# verdict=identical (exported symbols + device ISA identical, 19 exports)

python3 utils/codeobj_diff.py \
  agent_space/ohm-gfx1100-gpu3/build-old/bin/ohmtesthip \
  agent_space/ohm-gfx1100-gpu3/build-new/bin/ohmtesthip
# verdict=identical (exported symbols + device ISA identical, 21 exports)

python3 utils/codeobj_diff.py \
  agent_space/ohm-gfx1100-gpu3/build-old/lib/libohmhip.a \
  agent_space/ohm-gfx1100-gpu3/build-new/lib/libohmhip.a
# verdict=identical (exported symbols + device ISA identical)
```

All GPU binaries identical on gfx1100 -- carry-forward applied (no GPU re-run needed).

## Validation 2026-06-19 (windows-gfx1101)

**GPU**: AMD Radeon PRO V710, gfx1101 (RDNA3, wave32), HIP_VISIBLE_DEVICES=1
**ROCm**: 7.14.0a20260604 (TheRock nightly)
**Fork**: AMD-Ecosystem/ohm moat-port HEAD 80ea8a81

### Build

Same recipe as windows-gfx1201, with `CMAKE_HIP_ARCHITECTURES=gfx1101` and a separate build dir:

```bash
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
ROCM_CORE=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_core
VCPKG=B:/develop/moat/projects/alien/src/build/vcpkg_installed/x64-windows

cmake -B agent_space/ohm_gfx1101_build -S projects/ohm/src \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$ROCM/lib/llvm/bin/clang.exe" \
  -DCMAKE_CXX_COMPILER="$ROCM/lib/llvm/bin/clang++.exe" \
  -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/clang++.exe" \
  -DCMAKE_PREFIX_PATH="$ROCM;$ROCM_CORE;$VCPKG" \
  -DOHM_FEATURE_HIP=ON -DOHM_FEATURE_CUDA=OFF -DOHM_FEATURE_OPENCL=OFF \
  -DOHM_FEATURE_TEST=ON -DOHM_FEATURE_EIGEN=ON -DOHM_FEATURE_THREADS=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
  "-DGLM_INCLUDE_DIRS=B:/develop/moat/agent_space/glm/glm" \
  -Dglm_FOUND=TRUE \
  -DOHM_SYSTEM_GTEST=ON \
  "-DGTest_DIR=$VCPKG/share/gtest" \
  "-DEigen3_DIR=B:/develop/agent_space/eigen_install/share/eigen3/cmake" \
  -DCMAKE_CXX_FLAGS="-D_USE_MATH_DEFINES -DNOMINMAX -DWIN32" \
  -DCMAKE_C_FLAGS="-D_USE_MATH_DEFINES"

# Strip -fuse-ld=lld-link (CMake 4.3 injects it; amdclang++ device-link rejects it)
sed -i 's/-fuse-ld=lld-link//g' agent_space/ohm_gfx1101_build/build.ninja

cmake --build agent_space/ohm_gfx1101_build -j64
```

217/217 targets built cleanly. HIP compiler: amdclang++ 23.0.0 (gfx1101).

Runtime DLLs copied to bin/: amdhip64_7.dll, amd_comgr.dll, hiprtc0714.dll, hiprtc-builtins0714.dll, rocm_kpack.dll, gtest.dll, gtest_main.dll, zlib1.dll.

### Test results

```bash
HIP_VISIBLE_DEVICES=1 ctest --test-dir agent_space/ohm_gfx1101_build -j1
```

```
1/5: gputiltesthip ..... Passed  (8/8 GPU tests)
2/5: ohmtest ........... Passed  (53/53 CPU tests)
3/5: ohmtesthip ........ Passed  (58/58 GPU tests)
4/5: ohmtestheightmap .. Passed  (19/19 heightmap tests)
5/5: slamiotest ........ Passed  (4/4 IO tests)
100% tests passed, 0 tests failed out of 5
Total Test time: 62.52 sec
```

**GPU tests: 66/66 PASSED (8 gputiltesthip + 58 ohmtesthip)**
**All tests: 5/5 executables PASSED**

All occupancy-map population, ray integration, line query, NDT, traversal, and TSDF GPU tests pass on gfx1101 (RDNA3 wave32). Results match prior validations on linux-gfx90a, linux-gfx1100, and windows-gfx1201.
