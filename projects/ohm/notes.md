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

## Masquerade port 2026-11-04 (linux-gfx90a porter)

Restart ruled by jeffdaily on 2026-11-03 (see "Strategy revision" in plan.md). `moat-port` was reset to upstream 4e2e769; the separate-directory port stays on `moat-port-separate-hip-backend` (141e356) for reference.

### Commits (local fork clone, on top of 4e2e769)

```
4ad6bfc [ROCm] Add missing <utility> includes
da4f062 [ROCm] Add missing <iomanip> includes in tests
19dc171 [ROCm] Recognise the HIP compiler in the CUDA device code checks
ea8bbea [ROCm] Build the CUDA backend for AMD GPUs with HIP
```

### Size compared with the first port

```
git diff --shortstat 4e2e769 ea8bbea   # 24 files changed, 157 insertions(+), 16 deletions(-)
git diff --shortstat 4e2e769 141e356   # 57 files changed, 4003 insertions(+), 27 deletions(-)
```

No copied source, no `gputil/hip/`, no `.hip` files. New files: `cmake/OhmHip.cmake` (29 lines) and `gputil/cuda/hip_compat/{cuda_runtime.h,cuda.h,math_constants.h}` (the compat header and two one-line forwards). The README gains a "Building for AMD GPUs" section.

### Design decisions

- **Reuses the cuda target and type names** (`gputilcuda`, `ohmcuda`, `gputiltestcuda`, `ohmtestcuda`, `GPUTIL_TYPE == GPUTIL_CUDA`, `GPUTIL_CUDA_DECLARE_KERNEL`), with no hip aliases. `OHM_FEATURE_HIP=ON` makes `OhmCuda.cmake` include `OhmHip.cmake` and `return()` before any CUDA detection. OhmHip sets `OHM_FEATURE_CUDA` (a normal variable), so every existing `if(OHM_FEATURE_CUDA)` block configures the same targets, and it redefines `cuda_setup()` to enable HIP and set `OHM_CUDA_LIBRARY=hip::host`. Reason: aliases would need edits to every `GPUTIL_TYPE == GPUTIL_CUDA` check (the first port touched 9 host files for that alone) and to every CMake block, with no difference in behaviour. The binaries keep their cuda names, as the README says.
- **Shim headers rather than an included compat header**: the CUDA sources include `<cuda_runtime.h>`, `<cuda.h>` and `<math_constants.h>` by name, so `gputil/cuda/hip_compat/` is added `SYSTEM BEFORE PUBLIC` to `gputilcuda` on HIP builds only (and installed with them). No CUDA source line changes for the runtime mapping.
- **Deviation from the plan text: `gputil/cuda/*.cpp` stay host C++, not `LANGUAGE HIP`.** The first build attempt marked them HIP and failed: `gpuKernel.cpp` gets a device pass in which `cutil_decl.h`'s host-only helpers are guarded out (`error: unknown type name 'OptimalGroupSizeCalculation'`). Under CUDA those files never see nvcc; the host compiler builds them. Keeping them `CXX` matches the CUDA build exactly. Only `.cu` files are marked HIP (`ohm_hip_sources()` filters on `\.cu$`, because the lists also hold `.cl` files).
- **Implicit `cuda_runtime.h` include**: nvcc includes it in every `.cu`, and `cutil_importcl.h` uses `threadIdx`/`blockDim` before any include, so the HIP build failed with `use of undeclared identifier 'gridDim'`. `cuda_setup()` adds `$<$<COMPILE_LANGUAGE:HIP>:SHELL:-include cuda_runtime.h>`, which resolves to the shim.
- **Guards for the real differences** (commit 19dc171): the `__CUDACC__` checks in `gpu_ext.h`, `cutil_atomic.h` and `MapCoord.h` also accept `__HIPCC__` (clang in HIP mode does not define `__CUDACC__`, checked with `clang++ -x hip -dM -E`). `cutil_decl.h`'s `__CUDA_ARCH__` device-pass check also tests `__HIP_DEVICE_COMPILE__`. The `cutil_math.h` `_VECTOR_OPS`/`_VECTOR_UOPS` instantiations are skipped under `__HIP_PLATFORM_AMD__`: HIP_vector_type already defines those operators, and its `operator==` returns a bool all-equal, which is the same meaning. In the shim, `cudaDeviceProp` maps to `hipDeviceProp_t` and `cudaHostAlloc` to `hipHostMalloc` (hipify's mapping). The `__local` warning the first port saw did not appear.
- **Arch auto-detect**: no arch is pinned anywhere. The build below omitted `CMAKE_HIP_ARCHITECTURES`, and `enable_language(HIP)` detected `gfx90a` (one entry per visible GPU). This history never had a pin to remove, so this fix has no commit of its own; ea8bbea's body states it.
- **The `<utility>` fix is needed on Linux too**: upstream 4e2e769 does not build with GCC 13.3 (`'exchange' is not a member of 'std'` in `PlyPointStream.cpp` and `VoxelBuffer.cpp`). Before anything was published, the two include commits were recreated once in this local clone to correct that commit's body, which had wrongly blamed the Windows toolchain.

### CUDA path unchanged (no nvcc on this host)

All 9 `.cu` wrappers were preprocessed with g++ from 4e2e769 and from ea8bbea, with empty stub `cuda_runtime.h`/`cuda.h`/`math_constants.h`:

```
g++ -std=c++14 -E -P -x c++ -D__CUDACC__ <includes> <file.cu>                       # host pass
g++ -std=c++14 -E -P -x c++ -D__CUDACC__ -D__CUDA_ARCH__=700 <includes> <file.cu>   # device pass
```

18/18 outputs are byte-identical. In CMake, every addition sits inside `if(OHM_FEATURE_HIP)` or behind the `OhmCuda.cmake` early return. With HIP off, `if(OHM_FEATURE_CUDA AND NOT OHM_FEATURE_HIP)` and `elseif(NOT OHM_USE_DEPRECATED_CMAKE_CUDA)` evaluate as before.

### Build (gfx90a, MI250X, ROCm 7.16 TheRock SDK, CMake 3.31.6)

```bash
ROCM=<TheRock _rocm_sdk_devel dir>   # /opt/rocm on a packaged install
cmake -B agent_space/build-hip -S projects/ohm/src -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DOHM_FEATURE_HIP=ON -DOHM_FEATURE_OPENCL=OFF -DOHM_FEATURE_TEST=ON \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$ROCM \
  -DGLM_INCLUDE_DIRS=<glm 1.0.1 source dir>
utils/timeit.sh ohm compile -- cmake --build agent_space/build-hip -j 64
```

glm is not installed on this image, so glm 1.0.1 headers were downloaded to agent_space. The build is clean apart from pre-existing `-Wunused-*`/`-Wsign-compare` warnings in the upstream `.cl` kernels and a pre-existing FindOpenGL CMP0072 dev warning.

An environment quirk, not a port issue: TheRock's pip layout names the library `libamdhip64.so.7.16.26385-0000000`, which CMake does not recognise as a shared library. Executables linked by the C++ linker (`ohmtestcuda`, `ohmtest`, ...) therefore get no build RPATH and fail with exit 127, `libamdhip64.so.7: cannot open shared object file`. `gputiltestcuda` links through the HIP linker and has the RUNPATH. Run with `LD_LIBRARY_PATH=$ROCM/lib`. A packaged `/opt/rocm` is on the loader path and is unaffected.

### Tests at ea8bbea (HIP_VISIBLE_DEVICES pinned to one GCD, ulimit -c 0)

```bash
export LD_LIBRARY_PATH=$ROCM/lib
cd agent_space/build-hip/bin
./gputiltestcuda     # 8 tests from 3 suites: PASSED 8
./ohmtestcuda        # 58 tests from 16 suites: PASSED 58 (70.4 s)
./ohmtest            # 53 tests from 18 suites: PASSED 53
./ohmtestheightmap   # 19 tests: PASSED 19
./slamiotest         # 4 tests: PASSED 4
```

GPU 66/66 and CPU 76/76, matching the first port's counts (gputiltesthip 8/8, ohmtesthip 58/58, ohmtest 53, ohmtestheightmap 19, slamiotest 4). The suites ran twice, once after the first successful build and again after a full rebuild from the committed ea8bbea tree, with the same counts both times.

CPU-only build of ea8bbea (`-DOHM_FEATURE_OPENCL=OFF -DOHM_FEATURE_TEST=ON`, HIP off, g++ 13.3): builds; ohmtest 53/53, ohmtestheightmap 19/19, slamiotest 4/4. The same CPU-only build of upstream 4e2e769 fails to compile on g++ 13.3 (the `<utility>` errors above), so this host has no baseline counts for it.

### Fork update blocked: stage left at `porting`

The fast-forward of `moat-port` to ea8bbea (remote `origin` = AMD-Ecosystem/ohm) was refused by the fleet job's PreToolUse guard: "(a MOAT job pushes only to repositories in the AMD-Ecosystem organization)". The guard appears to match the remote name, not the URL it resolves to. As the guard instructs, no other route was tried. So `advance-head` and `ported` were NOT run: a head the fork does not have cannot be reviewed or validated. linux-gfx90a still holds the porting lock.

To finish:
1. A person, or a job whose guard allows it, applies the series below to `moat-port` at 4e2e769 with `git am`.
2. Check whether the result is ea8bbea28270871f3c32ac61c49789b13b60c3f6. `git am` keeps author and dates, but a different committer changes the sha; in that case use the resulting sha.
3. Publish the branch as a fast-forward, then run:

```bash
python3 utils/moatlib.py advance-head ohm <published sha>
python3 utils/moatlib.py set-state ohm linux-gfx90a ported
python3 utils/moatlib.py commit-project ohm "ohm: masquerade port published, ported"
```

<details><summary>git format-patch 4e2e769..ea8bbea</summary>

~~~~
From 4ad6bfce1ce4c31919eca6ee3be3866f3852c065 Mon Sep 17 00:00:00 2001
From: Jeff Daily <jeff.daily@amd.com>
Date: Wed, 4 Nov 2026 00:23:39 +0000
Subject: [PATCH 1/4] [ROCm] Add missing <utility> includes

PlyPointStream.cpp and VoxelBuffer.cpp use std::exchange, std::move and std::swap without including <utility>. Older standard libraries pulled <utility> in transitively; GCC 13's libstdc++ does not, and both files fail with "'exchange' is not a member of 'std'". Include the header where it is used.

Authored with Claude.

Test Plan:
```
cmake -B build -S . -DOHM_FEATURE_OPENCL=OFF -DOHM_FEATURE_TEST=ON
cmake --build build -j
./build/bin/ohmtest
```
---
 ohm/VoxelBuffer.cpp        | 2 ++
 ohmutil/PlyPointStream.cpp | 1 +
 2 files changed, 3 insertions(+)

diff --git a/ohm/VoxelBuffer.cpp b/ohm/VoxelBuffer.cpp
index f717cca..801d1d9 100644
--- a/ohm/VoxelBuffer.cpp
+++ b/ohm/VoxelBuffer.cpp
@@ -5,6 +5,8 @@
 // Author: Kazys Stepanas
 #include "VoxelBuffer.h"
 
+#include <utility>
+
 namespace ohm
 {
 template <typename VoxelBlock>
diff --git a/ohmutil/PlyPointStream.cpp b/ohmutil/PlyPointStream.cpp
index 0f6b9a5..b9d9479 100644
--- a/ohmutil/PlyPointStream.cpp
+++ b/ohmutil/PlyPointStream.cpp
@@ -12,6 +12,7 @@
 #include <iostream>
 #include <limits>
 #include <stdexcept>
+#include <utility>
 
 namespace ohm
 {
-- 
2.56.0

From da4f062761d70685b8557f5a259757f770d1c4e9 Mon Sep 17 00:00:00 2001
From: Jeff Daily <jeff.daily@amd.com>
Date: Wed, 4 Nov 2026 00:23:39 +0000
Subject: [PATCH 2/4] [ROCm] Add missing <iomanip> includes in tests

Six test files use std::setw or std::setfill without including <iomanip>. On Linux with libstdc++ the header arrives transitively; the clang-based ROCm toolchain on Windows does not provide it that way and the tests fail to compile. Include it explicitly.

Authored with Claude.

Test Plan:
```
cmake -B build -S . -DOHM_FEATURE_OPENCL=OFF -DOHM_FEATURE_TEST=ON
cmake --build build -j
./build/bin/ohmtest
```
---
 tests/gputiltest/KernelTest.cpp        | 1 +
 tests/ohmtest/RayPatternTests.cpp      | 1 +
 tests/ohmtest/VoxelMeanTests.cpp       | 1 +
 tests/ohmtestgpu/GpuLineKeysTests.cpp  | 1 +
 tests/ohmtestgpu/GpuMapTest.cpp        | 1 +
 tests/ohmtestgpu/GpuVoxelMeanTests.cpp | 1 +
 6 files changed, 6 insertions(+)

diff --git a/tests/gputiltest/KernelTest.cpp b/tests/gputiltest/KernelTest.cpp
index f6c69f3..f376434 100644
--- a/tests/gputiltest/KernelTest.cpp
+++ b/tests/gputiltest/KernelTest.cpp
@@ -13,6 +13,7 @@
 #include <gputil/gpuQueue.h>
 
 #include <chrono>
+#include <iomanip>
 
 #if GPUTIL_TYPE == GPUTIL_OPENCL
 #include "matrixResource.h"
diff --git a/tests/ohmtest/RayPatternTests.cpp b/tests/ohmtest/RayPatternTests.cpp
index 57f9d04..70ead10 100644
--- a/tests/ohmtest/RayPatternTests.cpp
+++ b/tests/ohmtest/RayPatternTests.cpp
@@ -17,6 +17,7 @@
 
 #include <array>
 #include <cstdio>
+#include <iomanip>
 #include <vector>
 
 #include <glm/ext.hpp>
diff --git a/tests/ohmtest/VoxelMeanTests.cpp b/tests/ohmtest/VoxelMeanTests.cpp
index 5688b16..c23c03b 100644
--- a/tests/ohmtest/VoxelMeanTests.cpp
+++ b/tests/ohmtest/VoxelMeanTests.cpp
@@ -25,6 +25,7 @@
 
 #include <chrono>
 #include <fstream>
+#include <iomanip>
 #include <iostream>
 #include <random>
 
diff --git a/tests/ohmtestgpu/GpuLineKeysTests.cpp b/tests/ohmtestgpu/GpuLineKeysTests.cpp
index f7e7f88..d8350a3 100644
--- a/tests/ohmtestgpu/GpuLineKeysTests.cpp
+++ b/tests/ohmtestgpu/GpuLineKeysTests.cpp
@@ -18,6 +18,7 @@
 #include <ohmutil/GlmStream.h>
 
 #include <chrono>
+#include <iomanip>
 #include <random>
 
 #include <gtest/gtest.h>
diff --git a/tests/ohmtestgpu/GpuMapTest.cpp b/tests/ohmtestgpu/GpuMapTest.cpp
index 53a6ccd..7f3166d 100644
--- a/tests/ohmtestgpu/GpuMapTest.cpp
+++ b/tests/ohmtestgpu/GpuMapTest.cpp
@@ -26,6 +26,7 @@
 #include <chrono>
 #include <fstream>
 #include <future>
+#include <iomanip>
 #include <iostream>
 #include <random>
 #include <thread>
diff --git a/tests/ohmtestgpu/GpuVoxelMeanTests.cpp b/tests/ohmtestgpu/GpuVoxelMeanTests.cpp
index e9df632..19d7374 100644
--- a/tests/ohmtestgpu/GpuVoxelMeanTests.cpp
+++ b/tests/ohmtestgpu/GpuVoxelMeanTests.cpp
@@ -23,6 +23,7 @@
 
 #include <chrono>
 #include <fstream>
+#include <iomanip>
 #include <iostream>
 #include <random>
 
-- 
2.56.0

From 19dc171d2b96f60beb9405596d8f1a6d49b86cbf Mon Sep 17 00:00:00 2001
From: Jeff Daily <jeff.daily@amd.com>
Date: Wed, 4 Nov 2026 00:24:41 +0000
Subject: [PATCH 3/4] [ROCm] Recognise the HIP compiler in the CUDA device code
 checks

Prepare the CUDA code to compile with HIP for AMD GPUs. The checks below test macros that only nvcc defines; clang in HIP mode defines __HIPCC__ for every pass and __HIP_DEVICE_COMPILE__ for the device pass instead. Each check gains the HIP equivalent, so the CUDA build sees exactly the same code as before.

- gpu_ext.h, cutil_atomic.h, MapCoord.h: __CUDACC__ also accepts __HIPCC__.
- cutil_decl.h: the host-only kernel helpers are excluded from the device pass by __HIP_DEVICE_COMPILE__ as well as __CUDA_ARCH__.
- cutil_math.h: HIP's vector types on AMD GPUs already define the arithmetic and comparison operators, with the same meaning, so the CUDA definitions would be ambiguous there. They are skipped when __HIP_PLATFORM_AMD__ is defined; the other helpers (dot, length, normalize, clamp, the logic and conversion functions) are kept.

Authored with Claude.

Test Plan:
The CUDA translation units preprocess identically before and after this change, for both the host pass and the device pass:
```
for f in ohmgpu/gpu/*.cu tests/gputiltest/cuda/matrix_kernel.cu; do
  g++ -std=c++14 -E -P -x c++ -D__CUDACC__ [-D__CUDA_ARCH__=700] <include dirs> $f
done   # 18 outputs, all byte-identical to the previous commit
```
---
 gputil/cuda/cutil_atomic.h |  4 ++--
 gputil/cuda/cutil_decl.h   |  6 +++---
 gputil/cuda/cutil_math.h   |  3 +++
 gputil/gpu_ext.h           |  2 +-
 ohm/MapCoord.h             | 12 ++++++------
 5 files changed, 15 insertions(+), 12 deletions(-)

diff --git a/gputil/cuda/cutil_atomic.h b/gputil/cuda/cutil_atomic.h
index 3d6ed93..ab2f2a7 100644
--- a/gputil/cuda/cutil_atomic.h
+++ b/gputil/cuda/cutil_atomic.h
@@ -11,7 +11,7 @@
 //----------------------------------------------------------------------------------------------------------------------
 // Alias CUDA atomics to match C/OpenCL 2.0 atomics
 //----------------------------------------------------------------------------------------------------------------------
-#ifdef __CUDACC__
+#if defined(__CUDACC__) || defined(__HIPCC__)
 using atomic_int = int;
 using atomic_uint = uint;
 using atomic_long = long long;
@@ -209,6 +209,6 @@ inline __device__ uint gputilAtomicXor(atomic_uint *p, uint val)
   return atomicXor(p, val);
 }
 
-#endif  // __CUDACC__
+#endif  // defined(__CUDACC__) || defined(__HIPCC__)
 
 #endif  // CUTIL_ATOMIC_H
diff --git a/gputil/cuda/cutil_decl.h b/gputil/cuda/cutil_decl.h
index 9dd12c9..a0eb03f 100644
--- a/gputil/cuda/cutil_decl.h
+++ b/gputil/cuda/cutil_decl.h
@@ -9,7 +9,7 @@
 // Whilst the functions below are __host__, the declaration of std::function apparent is __host__ __device__.
 // This can yield a situation where a __host__ function is calling a __host__ __device__ function which is not
 // allowed. We force the issue by excluding this code when building device code.
-#ifndef __CUDA_ARCH__
+#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
 
 #include <functional>
 
@@ -47,10 +47,10 @@ using OptimalGroupSizeCalculation = std::function<int(size_t *, const SharedMemC
     };                                                                                                         \
   }
 
-#else  // !__CUDA_ARCH__
+#else  // !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
 
 #define GPUTIL_CUDA_DEFINE_KERNEL(kernel_name)
 
-#endif  // !__CUDA_ARCH__
+#endif  // !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
 
 #endif  // CUTIL_DECL_H
diff --git a/gputil/cuda/cutil_math.h b/gputil/cuda/cutil_math.h
index b732366..a081f7e 100644
--- a/gputil/cuda/cutil_math.h
+++ b/gputil/cuda/cutil_math.h
@@ -381,6 +381,8 @@ using ulonglong = unsigned long long int;
   inline __host__ __device__ type##4 operator-(const type##4 & a) { return make_##type##4(-a.x, -a.y, -a.z, -a.w); }
 
 
+// HIP vector types on AMD GPUs already define these operators.
+#ifndef __HIP_PLATFORM_AMD__
 _VECTOR_OPS(char);
 _VECTOR_UOPS(uchar);
 _VECTOR_OPS(short);
@@ -391,6 +393,7 @@ _VECTOR_OPS(long);
 _VECTOR_UOPS(ulong);
 _VECTOR_OPS(float);
 _VECTOR_OPS(double);
+#endif  // __HIP_PLATFORM_AMD__
 
 //----------------------------------------------------------------------------------------------------------------------
 // OpenCL equivalent extensions
diff --git a/gputil/gpu_ext.h b/gputil/gpu_ext.h
index 1d652f6..698bc90 100644
--- a/gputil/gpu_ext.h
+++ b/gputil/gpu_ext.h
@@ -29,7 +29,7 @@
 // Setup GPUTIL_DEVICE. This has a non-zero value only when building device code.
 #if defined(__OPENCL_C_VERSION__)
 #define GPUTIL_DEVICE GPUTIL_OPENCL
-#elif defined(__CUDACC__)
+#elif defined(__CUDACC__) || defined(__HIPCC__)
 #define GPUTIL_DEVICE GPUTIL_CUDA
 #else
 #define GPUTIL_DEVICE GPUTIL_NULL
diff --git a/ohm/MapCoord.h b/ohm/MapCoord.h
index 33d17b6..28a34b2 100644
--- a/ohm/MapCoord.h
+++ b/ohm/MapCoord.h
@@ -71,12 +71,12 @@ inline __device__ __host__ int pointToRegionVoxel(coord_real coord, coord_real v
   {
     coord -= epsilon;
   }
-#if defined(__CUDACC__)
+#if defined(__CUDACC__) || defined(__HIPCC__)
   return (int)floorf((coord / voxel_resolution));
-#else   // defined(__CUDACC__)
+#else   // defined(__CUDACC__) || defined(__HIPCC__)
   // NOLINTNEXTLINE
   return (int)floor((coord / voxel_resolution));
-#endif  // defined(__CUDACC__)
+#endif  // defined(__CUDACC__) || defined(__HIPCC__)
 }
 
 #if GPUTIL_DEVICE != 1
@@ -84,12 +84,12 @@ template <typename coord_real>  // NOLINT(readability-identifier-naming)
 #endif                          // GPUTIL_DEVICE != 1
 inline __device__ __host__ int pointToRegionCoord(coord_real coord, coord_real resolution)
 {
-#if defined(__CUDACC__)
+#if defined(__CUDACC__) || defined(__HIPCC__)
   return (int)floorf(coord / resolution + (coord_real)0.5);
-#else   // defined(__CUDACC__)
+#else   // defined(__CUDACC__) || defined(__HIPCC__)
   // NOLINTNEXTLINE
   return (int)floor(coord / resolution + (coord_real)0.5f);
-#endif  // defined(__CUDACC__)
+#endif  // defined(__CUDACC__) || defined(__HIPCC__)
 }
 #if !GPUTIL_DEVICE
 }  // namespace ohm
-- 
2.56.0

From ea8bbea28270871f3c32ac61c49789b13b60c3f6 Mon Sep 17 00:00:00 2001
From: Jeff Daily <jeff.daily@amd.com>
Date: Wed, 4 Nov 2026 00:24:41 +0000
Subject: [PATCH 4/4] [ROCm] Build the CUDA backend for AMD GPUs with HIP

Add an OHM_FEATURE_HIP option that compiles the existing CUDA backend with HIP from ROCm, so ohm's GPU map population and queries run on AMD GPUs. No CUDA source is copied: the CUDA targets, source lists and options are reused, and the libraries, utilities and tests keep their cuda names (gputilcuda, ohmcuda, gputiltestcuda, ohmtestcuda).

- cmake/OhmHip.cmake: when OHM_FEATURE_HIP is on, OhmCuda.cmake hands over to it before any CUDA detection. It sets OHM_FEATURE_CUDA so the CUDA targets are configured, enables the HIP language and links hip::host where the CUDA build links cudart. The GPU architecture comes from CMAKE_HIP_ARCHITECTURES, and when that is unset enable_language(HIP) detects the GPUs present instead of assuming one.
- The .cu files compile as HIP. The gputil CUDA .cpp files stay host C++, as in the CUDA build. As nvcc does, every .cu file implicitly includes cuda_runtime.h.
- gputil/cuda/hip_compat: cuda_runtime.h maps the CUDA runtime names gputil uses to HIP; cuda.h and math_constants.h forward to it. This directory is on the include path of HIP builds only and is installed with them.
- The root CMakeLists.txt skips the CUDA toolkit lookup for HIP builds, and the package config finds hip for consumers of a HIP build.
- readme.md documents the build.

The CUDA build compiles the same sources with the same options as before; all additions are behind OHM_FEATURE_HIP.

Authored with Claude.

Test Plan:
On an AMD Instinct MI250X (gfx90a), ROCm 7.16, architecture auto-detected:
```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DOHM_FEATURE_HIP=ON -DOHM_FEATURE_OPENCL=OFF -DOHM_FEATURE_TEST=ON -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build -j
./build/bin/gputiltestcuda    # 8/8 passed
./build/bin/ohmtestcuda       # 58/58 passed
./build/bin/ohmtest           # 53/53 passed
./build/bin/ohmtestheightmap  # 19/19 passed
./build/bin/slamiotest        # 4/4 passed
```
CPU-only build (no GPU feature) still builds and passes ohmtest 53/53, ohmtestheightmap 19/19, slamiotest 4/4.
---
 CMakeLists.txt                          |  4 +--
 cmake/OhmCuda.cmake                     |  7 ++++
 cmake/OhmHip.cmake                      | 29 +++++++++++++++
 cmake/ohm-config.cmake                  |  8 +++--
 gputil/CMakeLists.txt                   | 15 ++++++++
 gputil/cuda/hip_compat/cuda.h           |  2 ++
 gputil/cuda/hip_compat/cuda_runtime.h   | 48 +++++++++++++++++++++++++
 gputil/cuda/hip_compat/math_constants.h |  2 ++
 ohmgpu/CMakeLists.txt                   |  3 ++
 readme.md                               | 16 +++++++++
 tests/gputiltest/CMakeLists.txt         |  3 ++
 11 files changed, 133 insertions(+), 4 deletions(-)
 create mode 100644 cmake/OhmHip.cmake
 create mode 100644 gputil/cuda/hip_compat/cuda.h
 create mode 100644 gputil/cuda/hip_compat/cuda_runtime.h
 create mode 100644 gputil/cuda/hip_compat/math_constants.h

diff --git a/CMakeLists.txt b/CMakeLists.txt
index e0041b6..1500c3b 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -122,14 +122,14 @@ endif(OpenCL_FOUND)
 set(OHM_OPENCL_2_FEATURES "" CACHE STRING "Features required for OpenCL 2.x. Fallback to 1.2 otherwise.")
 
 # Look for CUDA to setup CUDA build options and work out which CMake CUDA paradigm we'll use.
-if(OHM_FEATURE_CUDA)
+if(OHM_FEATURE_CUDA AND NOT OHM_FEATURE_HIP)
   if(NOT OHM_USE_DEPRECATED_CMAKE_CUDA)
     find_package(CUDAToolkit REQUIRED)
   else(NOT OHM_USE_DEPRECATED_CMAKE_CUDA)
     # Need to find CUDA for deprecated project configuration
     find_package(CUDA REQUIRED)
   endif(NOT OHM_USE_DEPRECATED_CMAKE_CUDA)
-endif(OHM_FEATURE_CUDA)
+endif(OHM_FEATURE_CUDA AND NOT OHM_FEATURE_HIP)
 
 if(OHM_FEATURE_EIGEN)
   find_package(Eigen3 CONFIG REQUIRED)
diff --git a/cmake/OhmCuda.cmake b/cmake/OhmCuda.cmake
index 32cad73..bf76ed9 100644
--- a/cmake/OhmCuda.cmake
+++ b/cmake/OhmCuda.cmake
@@ -1,3 +1,10 @@
+# OHM_FEATURE_HIP builds the CUDA code for AMD GPUs instead, and replaces the CUDA setup below.
+option(OHM_FEATURE_HIP "Build ohm library and utilities for AMD GPUs by compiling the CUDA code with HIP?" OFF)
+if(OHM_FEATURE_HIP)
+  include(OhmHip)
+  return()
+endif(OHM_FEATURE_HIP)
+
 # OHM_FEATURE_CUDA_DEFAULT is used to initialise the OHM_FEATURE_CUDA option. We have to work out if CUDA is present first.
 # How we do so depends on the CMake version.
 set(OHM_FEATURE_CUDA_DEFAULT OFF) # Initialisation value for OHM_FEATURE_CUDA
diff --git a/cmake/OhmHip.cmake b/cmake/OhmHip.cmake
new file mode 100644
index 0000000..1d0abb4
--- /dev/null
+++ b/cmake/OhmHip.cmake
@@ -0,0 +1,29 @@
+# Build the CUDA backend with HIP for AMD GPUs. The CUDA sources, targets and options are reused: OHM_FEATURE_CUDA is
+# set so the CUDA targets are configured, the .cu files compile as HIP, and gputil/cuda/hip_compat supplies the CUDA
+# headers they include.
+if(CMAKE_VERSION VERSION_LESS 3.21)
+  message(FATAL_ERROR "OHM_FEATURE_HIP requires CMake 3.21 or later.")
+endif(CMAKE_VERSION VERSION_LESS 3.21)
+
+set(OHM_FEATURE_CUDA ON)
+set(OHM_USE_DEPRECATED_CMAKE_CUDA OFF)
+
+# Compile the .cu files in the given list as HIP. Other files in the list are left alone.
+function(ohm_hip_sources)
+  foreach(_source ${ARGN})
+    if(_source MATCHES "\\.cu$")
+      set_source_files_properties(${_source} PROPERTIES LANGUAGE HIP)
+    endif()
+  endforeach(_source)
+endfunction(ohm_hip_sources)
+
+macro(cuda_setup)
+  # Use CMAKE_HIP_ARCHITECTURES to select the GPU targets. When unset, enable_language(HIP) detects the GPUs present.
+  enable_language(HIP)
+  set(CMAKE_HIP_STANDARD ${CMAKE_CXX_STANDARD})
+  set(CMAKE_HIP_STANDARD_REQUIRED ${CMAKE_CXX_STANDARD_REQUIRED})
+  # nvcc implicitly includes cuda_runtime.h in every .cu file. Do the same, resolving to gputil/cuda/hip_compat.
+  add_compile_options("$<$<COMPILE_LANGUAGE:HIP>:SHELL:-include cuda_runtime.h>")
+  find_package(hip REQUIRED)
+  set(OHM_CUDA_LIBRARY hip::host)
+endmacro(cuda_setup)
diff --git a/cmake/ohm-config.cmake b/cmake/ohm-config.cmake
index 218ace1..b83f06b 100644
--- a/cmake/ohm-config.cmake
+++ b/cmake/ohm-config.cmake
@@ -17,6 +17,7 @@
 set(OHM_VERSION @ohm_VERSION@)
 
 set(OHM_FEATURE_CUDA @OHM_FEATURE_CUDA@)
+set(OHM_FEATURE_HIP @OHM_FEATURE_HIP@)
 set(OHM_FEATURE_HEIGHTMAP_IMAGE @OHM_FEATURE_HEIGHTMAP_IMAGE@)
 set(OHM_FEATURE_OPENCL @OHM_FEATURE_OPENCL@)
 set(OHM_FEATURE_PDAL @OHM_FEATURE_PDAL@)
@@ -115,10 +116,13 @@ if(OHM_FEATURE_CUDA)
       set(OHM_GPUTIL_LIBRARY ohm::gputilcuda)
     set(OHM_GPU_LIBRARY ohm::ohmcuda)
   endif(NOT DEFINED OHM_GPU_LIBRARY)
-  if(NOT OHM_USE_DEPRECATED_CMAKE_CUDA)
+  if(OHM_FEATURE_HIP)
+    # CUDA code built with HIP. Need to have hip::host defined.
+    find_package(hip)
+  elseif(NOT OHM_USE_DEPRECATED_CMAKE_CUDA)
     # Using new CUDA configuration. Need to have CUDA::cudart[_static] defined by finding the toolkit.
     find_package(CUDAToolkit)
-  endif(NOT OHM_USE_DEPRECATED_CMAKE_CUDA)
+  endif()
 endif(OHM_FEATURE_CUDA)
 
 if(OHM_FEATURE_OPENCL)
diff --git a/gputil/CMakeLists.txt b/gputil/CMakeLists.txt
index 2294af5..ef37310 100644
--- a/gputil/CMakeLists.txt
+++ b/gputil/CMakeLists.txt
@@ -184,6 +184,18 @@ if(OHM_FEATURE_CUDA)
     PRIVATE
       $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/cuda>
   )
+  if(OHM_FEATURE_HIP)
+    target_include_directories(gputilcuda SYSTEM BEFORE
+      PUBLIC
+        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/cuda/hip_compat>
+        $<INSTALL_INTERFACE:${OHM_PREFIX_INCLUDE}/gputil/cuda/hip_compat>
+    )
+    list(APPEND PUBLIC_HEADERS_HIP_COMPAT
+      cuda/hip_compat/cuda.h
+      cuda/hip_compat/cuda_runtime.h
+      cuda/hip_compat/math_constants.h
+    )
+  endif(OHM_FEATURE_HIP)
   _gputil_setup_library(gputilcuda ${GPUTIL_TYPE_CUDA})
 endif(OHM_FEATURE_CUDA)
 
@@ -191,6 +203,9 @@ install(FILES ${PUBLIC_HEADERS} DESTINATION ${OHM_PREFIX_INCLUDE}/gputil)
 if(PUBLIC_HEADERS_CUDA)
   install(FILES ${PUBLIC_HEADERS_CUDA} DESTINATION ${OHM_PREFIX_INCLUDE}/gputil/cuda)
 endif(PUBLIC_HEADERS_CUDA)
+if(PUBLIC_HEADERS_HIP_COMPAT)
+  install(FILES ${PUBLIC_HEADERS_HIP_COMPAT} DESTINATION ${OHM_PREFIX_INCLUDE}/gputil/cuda/hip_compat)
+endif(PUBLIC_HEADERS_HIP_COMPAT)
 if(PUBLIC_HEADERS_OPENCL)
   install(FILES ${PUBLIC_HEADERS_OPENCL} DESTINATION ${OHM_PREFIX_INCLUDE}/gputil/cl)
 endif(PUBLIC_HEADERS_OPENCL)
diff --git a/gputil/cuda/hip_compat/cuda.h b/gputil/cuda/hip_compat/cuda.h
new file mode 100644
index 0000000..9dd94e7
--- /dev/null
+++ b/gputil/cuda/hip_compat/cuda.h
@@ -0,0 +1,2 @@
+// HIP build stand-in for the CUDA driver header. See cuda_runtime.h.
+#include "cuda_runtime.h"
diff --git a/gputil/cuda/hip_compat/cuda_runtime.h b/gputil/cuda/hip_compat/cuda_runtime.h
new file mode 100644
index 0000000..582391e
--- /dev/null
+++ b/gputil/cuda/hip_compat/cuda_runtime.h
@@ -0,0 +1,48 @@
+// Maps the CUDA runtime API used by gputil onto HIP. This directory is on the include path of HIP builds only
+// (OHM_FEATURE_HIP), so the CUDA sources compile unchanged for AMD GPUs.
+#ifndef GPUTIL_HIP_COMPAT_CUDA_RUNTIME_H
+#define GPUTIL_HIP_COMPAT_CUDA_RUNTIME_H
+
+#include <hip/hip_runtime.h>
+
+#define cudaDeviceProp hipDeviceProp_t
+#define cudaError_t hipError_t
+#define cudaErrorLaunchFailure hipErrorLaunchFailure
+#define cudaErrorMemoryAllocation hipErrorMemoryAllocation
+#define cudaErrorNotReady hipErrorNotReady
+#define cudaEventBlockingSync hipEventBlockingSync
+#define cudaEventCreateWithFlags hipEventCreateWithFlags
+#define cudaEventDestroy hipEventDestroy
+#define cudaEventQuery hipEventQuery
+#define cudaEventRecord hipEventRecord
+#define cudaEventSynchronize hipEventSynchronize
+#define cudaEvent_t hipEvent_t
+#define cudaFree hipFree
+#define cudaFreeHost hipHostFree
+#define cudaGetDevice hipGetDevice
+#define cudaGetDeviceCount hipGetDeviceCount
+#define cudaGetDeviceProperties hipGetDeviceProperties
+#define cudaGetErrorString hipGetErrorString
+#define cudaHostAlloc hipHostMalloc
+#define cudaLaunchKernel hipLaunchKernel
+#define cudaMalloc hipMalloc
+#define cudaMemcpy hipMemcpy
+#define cudaMemcpyAsync hipMemcpyAsync
+#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
+#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
+#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
+#define cudaMemcpyHostToHost hipMemcpyHostToHost
+#define cudaMemcpyKind hipMemcpyKind
+#define cudaMemset hipMemset
+#define cudaMemsetAsync hipMemsetAsync
+#define cudaOccupancyMaxPotentialBlockSizeVariableSMem hipOccupancyMaxPotentialBlockSizeVariableSMem
+#define cudaSetDevice hipSetDevice
+#define cudaStreamAddCallback hipStreamAddCallback
+#define cudaStreamCreate hipStreamCreate
+#define cudaStreamDestroy hipStreamDestroy
+#define cudaStreamSynchronize hipStreamSynchronize
+#define cudaStreamWaitEvent hipStreamWaitEvent
+#define cudaStream_t hipStream_t
+#define cudaSuccess hipSuccess
+
+#endif  // GPUTIL_HIP_COMPAT_CUDA_RUNTIME_H
diff --git a/gputil/cuda/hip_compat/math_constants.h b/gputil/cuda/hip_compat/math_constants.h
new file mode 100644
index 0000000..190e021
--- /dev/null
+++ b/gputil/cuda/hip_compat/math_constants.h
@@ -0,0 +1,2 @@
+// HIP build stand-in for the CUDA math constants header. See cuda_runtime.h.
+#include <hip/hip_math_constants.h>
diff --git a/ohmgpu/CMakeLists.txt b/ohmgpu/CMakeLists.txt
index 6196abe..e8a240c 100644
--- a/ohmgpu/CMakeLists.txt
+++ b/ohmgpu/CMakeLists.txt
@@ -246,6 +246,9 @@ if(OHM_FEATURE_CUDA)
     gpu/TransformSamples.cu
     gpu/TsdfUpdate.cu
   )
+  if(OHM_FEATURE_HIP)
+    ohm_hip_sources(${GPU_SOURCES})
+  endif(OHM_FEATURE_HIP)
 
   if(OHM_USE_DEPRECATED_CMAKE_CUDA)
     cuda_add_library(ohmcuda ${SOURCES} ${GPU_SOURCES} ${GPU_HEADERS})
diff --git a/readme.md b/readme.md
index b690c37..e5dbde4 100644
--- a/readme.md
+++ b/readme.md
@@ -19,6 +19,9 @@ Building ohm requires:
   - OpenCL 1.2 runtime. OpenCL 2.x also supported (command line selectable)
 - For CUDA
   - CUDA 10
+- For AMD GPUs using ROCm
+  - ROCm 7
+  - CMake 3.21
 
 The following 3rd-party libraries are required to build ohm:
 
@@ -56,6 +59,8 @@ Setup of OpenCL requires mode detailed [instructions (link)](./OpenCL.md).
 
 For CUDA setup instructions, visit [NVIDIA CUDA Zone](https://developer.nvidia.com/cuda-zone).
 
+For ROCm setup instructions, see the [ROCm installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/).
+
 Additional, recommended packages can be installed using:
 
 ```bash
@@ -80,6 +85,17 @@ sudo apt install libglew-dev libglfw3-dev libpng-dev libeigen3-dev
         - For Visual Studio, open the solution file and build.
         - For make based platforms, run `make -j`
 
+### Building for AMD GPUs
+
+The CUDA implementation also builds for AMD GPUs using HIP from ROCm. Enable `OHM_FEATURE_HIP` in place of `OHM_FEATURE_CUDA`; the libraries, utilities and tests keep their `cuda` names (for example `ohmcuda` and `ohmtestcuda`).
+
+```bash
+cmake -B build -DOHM_FEATURE_HIP=ON -DOHM_FEATURE_OPENCL=OFF -DCMAKE_PREFIX_PATH=/opt/rocm
+cmake --build build -j
+```
+
+The GPU architecture is detected from the GPUs present. Set `-DCMAKE_HIP_ARCHITECTURES` to build for other GPUs, for example `-DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100"`. If CMake does not find the HIP compiler, add `-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++` or put `/opt/rocm/bin` on `PATH`.
+
 ### Building with vcpkg
 
 Ohm supports building with in [vcpkg manifest mode](https://vcpkg.readthedocs.io/en/latest/users/manifests/). To build using vcpkg, first follow the [vcpkg install instructions](https://vcpkg.io/en/getting-started.html) then build with the vcpkg toolchain enabled.
diff --git a/tests/gputiltest/CMakeLists.txt b/tests/gputiltest/CMakeLists.txt
index b6a7fd3..ac8d2be 100644
--- a/tests/gputiltest/CMakeLists.txt
+++ b/tests/gputiltest/CMakeLists.txt
@@ -66,6 +66,9 @@ if(OHM_FEATURE_CUDA)
   list(APPEND GPU_SOURCES
     cuda/matrix_kernel.cu
   )
+  if(OHM_FEATURE_HIP)
+    ohm_hip_sources(${GPU_SOURCES})
+  endif(OHM_FEATURE_HIP)
   if(OHM_USE_DEPRECATED_CMAKE_CUDA)
     cuda_add_executable(gputiltestcuda ${SOURCES} ${GPU_SOURCES})
   else(OHM_USE_DEPRECATED_CMAKE_CUDA)
-- 
2.56.0

~~~~

</details>

### Fork updated 2026-11-04: moat-port at ea8bbea, stage ported

jeffdaily confirmed that the guard resolves the remote with `git -C <dir> remote get-url --push origin`, which printed `https://github.com/AMD-Ecosystem/ohm`. On the remote, `moat-port` was still 4e2e769 (`git ls-remote`). The local branch was a fast-forward from 4e2e769, and the clone's working tree was clean. A plain update ran with the literal clone path and `origin moat-port` and gave `4e2e769..ea8bbea  moat-port -> moat-port`. `git ls-remote` then showed ea8bbea28270871f3c32ac61c49789b13b60c3f6. After that, `advance-head ohm ea8bbea28270871f3c32ac61c49789b13b60c3f6` and `set-state ohm linux-gfx90a ported` ran, releasing the lock. The "Fork update blocked" section above is now resolved; its `git am` series stays as the record of the content.

## Review 2026-11-04 (linux-gfx942 reviewer, moat-port ea8bbea vs master 4e2e769)

### Summary
The port builds ohm's existing CUDA backend through HIP (an `OHM_FEATURE_HIP` option, a three-file shim include directory, macro guards in five headers, two include fixes) with no copied source. Two upstream-visible commit messages are wrong or not literal; the code itself has no defects found. **Request Changes** (commit messages only).

### Commit Hygiene
1. **ea8bbea body misstates the shim.** It says "cuda.h and math_constants.h forward to it" (to `cuda_runtime.h`). `gputil/cuda/hip_compat/math_constants.h:2` is `#include <hip/hip_math_constants.h>`; only `cuda.h` forwards to `cuda_runtime.h`. Reword to: "cuda_runtime.h maps the CUDA runtime names gputil uses to HIP, cuda.h forwards to it, and math_constants.h forwards to <hip/hip_math_constants.h>."
2. **19dc171 Test Plan is not a literal command.** It shows `g++ ... [-D__CUDA_ARCH__=700] <include dirs> $f`, with an optional-argument bracket and a placeholder (AGENTS.md: a Test Plan has literal commands). Replace with this, which the reviewer ran from the repository root at 4e2e769 and at ea8bbea (18 files each, `diff -r` identical):

```
mkdir -p /tmp/stub /tmp/pp && touch /tmp/stub/cuda_runtime.h /tmp/stub/cuda.h /tmp/stub/math_constants.h
for f in ohmgpu/gpu/*.cu tests/gputiltest/cuda/matrix_kernel.cu; do
  g++ -std=c++14 -E -P -x c++ -D__CUDACC__ -I/tmp/stub -I. -Igputil -Igputil/cuda -Iohm -Iohmgpu -Iohmgpu/gpu -Itests/gputiltest $f > /tmp/pp/$(basename $f).host.i
  g++ -std=c++14 -E -P -x c++ -D__CUDACC__ -D__CUDA_ARCH__=700 -I/tmp/stub -I. -Igputil -Igputil/cuda -Iohm -Iohmgpu -Iohmgpu/gpu -Itests/gputiltest $f > /tmp/pp/$(basename $f).device.i
done
# run before and after the change into two directories, then: diff -r <before> <after>
```

Fixing both rewrites two commit messages on `moat-port` (no upstream PR is open, so a `--force-with-lease` push is allowed), then `advance-head`. No code changes. No platform has validated ea8bbea yet, so no evidence is lost.

### Reviewer checks (evidence for the record; no finding)
- CUDA build is upstream's code: the 9 `.cu` files, host and device pass, preprocess byte-identically at 4e2e769 and ea8bbea (18/18, command above); a control with `-D__HIPCC__ -D__HIP_PLATFORM_AMD__` does differ, so the check detects guard changes. Host C++ in a CUDA build defines neither `__CUDACC__` nor `__HIPCC__`, so the changed headers are unchanged there. All CMake additions sit behind `OHM_FEATURE_HIP` or the `OhmCuda.cmake` early return.
- No copied source: `git diff --diff-filter=A` adds only `cmake/OhmHip.cmake` and `gputil/cuda/hip_compat/{cuda.h,cuda_runtime.h,math_constants.h}` (81 lines).
- Macro claims: `clang++ -x hip -dM -E` (ROCm 10.1 TheRock) defines `__HIPCC__` on both passes, `__HIP_DEVICE_COMPILE__` on the device pass, and never `__CUDACC__`/`__CUDA_ARCH__`.
- `cutil_math.h` operator skip: HIP's mixed vector/scalar operators convert the scalar to the element type first, while the skipped CUDA templates compute `a.x * scalar` in the promoted type. An AST dump of every `.cu` device pass lists the HIP operator instantiations used: every scalar operand matches the element type except `unsigned int * float3` in CovarianceHitNdt.cu, which converts to float either way. No result changes.
- Fault classes: no warp-size, lane-mask, shuffle or ballot use in `gputil/cuda`, `ohmgpu/gpu` or the test kernel; `cudaHostAlloc(..., 0)` maps to `hipHostMalloc(..., 0)` (default flags), with no deprecation warnings in the build.
- `jargon.py --port ohm`: clean. Commits: `[ROCm]` titles of 72 characters or fewer, "Authored with Claude.", no trailer.
- Skill lesson f54724a2 (`strategy-a-cmake.md`, "A per-backend directory..."): matches the code; the `-include` snippet is the form that builds.

### Reviewer corroboration on gfx942 (not a validation record)
Built ea8bbea on an AMD Instinct MI300X (gfx942, ROCm 10.1 TheRock pip SDK, CMake 3.31.6, glm 1.0.1), arch auto-detected (CMake listed gfx942 once per visible GPU), one GPU pinned, `ulimit -c 0`, `LD_LIBRARY_PATH=<sdk>/_rocm_sdk_core/lib`:

```
cmake -B agent_space/ohm-hip -S projects/ohm/src -G Ninja -DCMAKE_BUILD_TYPE=Release -DOHM_FEATURE_HIP=ON -DOHM_FEATURE_OPENCL=OFF -DOHM_FEATURE_TEST=ON -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$ROCM -DGLM_INCLUDE_DIRS=<glm-1.0.1>
cmake --build agent_space/ohm-hip -j 22     # 227/227
gputiltestcuda 8/8, ohmtestcuda 58/58 (71.9 s), ohmtest 53/53, ohmtestheightmap 19/19, slamiotest 4/4
```

A first attempt with a newer glm from another job's vcpkg tree failed in `CovarianceVoxel.h` (`glm::dquat` not declared): an environment issue (glm version), unrelated to the port.

### Proposed review PR title and body (for jeff)

Title: `[ROCm] Build the CUDA backend for AMD GPUs with HIP`

Before opening: the Testing table lists only the GPUs run at ea8bbea (MI250X by the porter, MI300X by this review). Add the wave32 and Windows rows (with GPU names and counts) once those validations pass at the final head, and update the sha-specific numbers if the head moves. The `<iomanip>` Windows claim comes from the first port's Windows validation and must still hold on the Windows run of this head. `jargon.py -` and `prose.py` are clean on the body below.

~~~~
This adds an `OHM_FEATURE_HIP` CMake option that builds ohm's existing CUDA backend for AMD GPUs with HIP from ROCm, so GPU map population and queries run on AMD GPUs. No CUDA source is copied or renamed: the CUDA targets, source lists and options are reused, and the libraries, utilities and tests keep their `cuda` names (`gputilcuda`, `ohmcuda`, `gputiltestcuda`, `ohmtestcuda`). With the option off, the CUDA and OpenCL builds compile the same code with the same options as before.

## Changes

- `cmake/OhmHip.cmake` (new): when `OHM_FEATURE_HIP` is on, `OhmCuda.cmake` hands over to it before any CUDA detection. It sets `OHM_FEATURE_CUDA` so the CUDA targets are configured, enables the HIP language, and links `hip::host` where the CUDA build links cudart. The `.cu` files in the existing source lists compile as HIP; the `gputil/cuda/*.cpp` files stay host C++, as in the CUDA build. As nvcc does, every `.cu` file implicitly includes `cuda_runtime.h`. The GPU architecture comes from `CMAKE_HIP_ARCHITECTURES`, or is detected from the GPUs present when that is unset.
- `gputil/cuda/hip_compat/` (new, HIP builds only): `cuda_runtime.h` maps the CUDA runtime names gputil uses to HIP, `cuda.h` forwards to it, and `math_constants.h` forwards to `<hip/hip_math_constants.h>`. The directory is on the include path of HIP builds only and is installed with them.
- Device code checks: `__CUDACC__` checks in `gpu_ext.h`, `cutil_atomic.h` and `MapCoord.h` also accept `__HIPCC__`, and the `__CUDA_ARCH__` check in `cutil_decl.h` also tests `__HIP_DEVICE_COMPILE__`. HIP's vector types on AMD GPUs already define the arithmetic and comparison operators, so the `cutil_math.h` operator definitions are skipped there; the other helpers are kept.
- The root `CMakeLists.txt` skips the CUDA toolkit lookup for HIP builds, and `ohm-config.cmake` finds hip for consumers of a HIP build.
- Missing standard includes: `<utility>` in `PlyPointStream.cpp` and `VoxelBuffer.cpp` (GCC 13 fails with "'exchange' is not a member of 'std'", with or without a GPU feature), and `<iomanip>` in six test files (needed by the Windows ROCm toolchain).
- `readme.md` documents the build.

## Testing

Built with `-DOHM_FEATURE_HIP=ON -DOHM_FEATURE_OPENCL=OFF -DOHM_FEATURE_TEST=ON`, GPU architecture auto-detected:

| GPU | Architecture | gputiltestcuda | ohmtestcuda | ohmtest | ohmtestheightmap | slamiotest |
|-----|--------------|----------------|-------------|---------|------------------|------------|
| AMD Instinct MI250X (Linux) | gfx90a | 8/8 | 58/58 | 53/53 | 19/19 | 4/4 |
| AMD Instinct MI300X (Linux) | gfx942 | 8/8 | 58/58 | 53/53 | 19/19 | 4/4 |

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DOHM_FEATURE_HIP=ON -DOHM_FEATURE_OPENCL=OFF -DOHM_FEATURE_TEST=ON -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build -j
./build/bin/gputiltestcuda
./build/bin/ohmtestcuda
./build/bin/ohmtest
./build/bin/ohmtestheightmap
./build/bin/slamiotest
```

A CPU-only build (`-DOHM_FEATURE_OPENCL=OFF -DOHM_FEATURE_TEST=ON`) builds with GCC 13.3 and passes ohmtest 53/53, ohmtestheightmap 19/19 and slamiotest 4/4.

No CUDA toolchain was available, so the CUDA build was checked by preprocessing: each of the nine `.cu` files, for the host pass (`-D__CUDACC__`) and the device pass (`-D__CUDACC__ -D__CUDA_ARCH__=700`), produces byte-identical output before and after this change (18 of 18).

This change was prepared with the help of an AI coding agent.
~~~~

## 2026-11-04 (windows-gfx1151, porter by orchestrator) -- commit-message rewrite, ea8bbea -> 7c09435

jeff approved rewriting the two commit messages the 2026-11-04 review flagged (no upstream PR open, nothing validated at ea8bbea), and ran the `--force-with-lease=moat-port:ea8bbea` push himself. 19dc171 and ea8bbea were re-created on da4f062 with `git cherry-pick -n` + `git commit -F`, same author and author date:
- 2e21f04 (was 19dc171): Test Plan replaced with a literal script (stub headers in /tmp/stub, `pp()` preprocessing host and device passes, `git checkout HEAD~1` / `git checkout -`, `diff -r /tmp/pp-before /tmp/pp-after`). Run on this host with Strawberry g++ at 19dc171: 18 outputs, `diff -r` identical.
- 7c09435 (was ea8bbea): "cuda_runtime.h maps the CUDA runtime names gputil uses to HIP, cuda.h forwards to it, and math_constants.h forwards to <hip/hip_math_constants.h>."

`git rev-parse 7c09435^{tree}` == `ea8bbea^{tree}` == c13dfc817c494b5fe5c76c031b692b2fdfbddf29: no content change. `jargon.py -` on both messages: clean.

## Review 2026-11-04 (linux-gfx90a reviewer, moat-port 7c09435 vs master 4e2e769)

### Summary
Re-review of the commit-message rewrite (ea8bbea -> 7c09435). Both findings of the earlier 2026-11-04 review are fixed, and nothing else changed. **Approve** (`review-passed`).

### Checks (evidence; no finding)
- Content unchanged: `git diff ea8bbea 7c09435` is empty; `7c09435^{tree}` == `ea8bbea^{tree}` == c13dfc8. `19dc171^{tree}` == `2e21f04^{tree}` == e4ce740. Both rewritten commits have parent chains on the unchanged da4f062 / 4ad6bfc. Author, author date and committer are the same as the old commits.
- Messages: `diff` of `git log -1 --format='%an <%ae>|%ad|%cn%n%B'` old vs new shows only the two intended edits. In 19dc171 -> 2e21f04, the placeholder Test Plan was replaced and "(18 outputs)" added. In ea8bbea -> 7c09435, the hip_compat bullet now reads "cuda.h forwards to it, and math_constants.h forwards to <hip/hip_math_constants.h>", which matches `gputil/cuda/hip_compat/math_constants.h:2`. 7c09435's raw message ends with one extra blank line, which git and GitHub do not display.
- 2e21f04 Test Plan is literal and runs: the fenced block was extracted verbatim from `git log -1 --format=%B 2e21f04` and run with `bash -e` (GNU g++, Linux) on a detached checkout of 2e21f04. Exit 0, 18 files in each of `/tmp/pp-before` and `/tmp/pp-after`, none empty, `diff -r` printed nothing, and HEAD ended back at 2e21f04. Host and device outputs differ (e.g. RaysQuery.cu), so the device pass does apply `__CUDA_ARCH__`.
- `jargon.py -` is clean on all four commit messages, and `jargon.py --port ohm` is clean. Titles are 38/47/65/52 characters, each starts with `[ROCm]`, each body has "Authored with Claude.", and none has a trailer or non-ASCII text.
- The proposed review PR body (2026-11-04 review section) was checked against the code at 7c09435. Its claims hold: the OhmCuda.cmake early return into OhmHip.cmake, `OHM_FEATURE_CUDA` set, `enable_language(HIP)`, `hip::host` in `OHM_CUDA_LIBRARY` (which gputilcuda and ohmcuda link where the CUDA build has `CUDA::cudart`), `ohm_hip_sources` on the ohmgpu and gputiltest `.cu` lists, `-include cuda_runtime.h`, hip_compat on HIP-only include paths and install, guards in gpu_ext.h/cutil_atomic.h/MapCoord.h/cutil_decl.h/cutil_math.h, the root CMakeLists and ohm-config changes, `<utility>` in two files and `<iomanip>` in six, and nine `.cu` files. MI250X matches this host's gfx90a market name. `jargon.py -` and `prose.py` are clean on it, and it is ASCII. No corrected body is needed. Its "Before opening" note should be read as at 7c09435: the tree is identical, so the MI250X and MI300X rows still describe the code, but the validated-at sha for every gate must be 7c09435 (or a later head) before the PR opens.

## Validation 2026-11-04 (linux-gfx1100, moat-port 7c09435)

Full real-GPU run (the earlier gfx1100 record at 141e356 was the separate-directory port, so nothing carries forward). AMD Radeon PRO W7900 (gfx1100, wave32), ROCm 10.1 TheRock pip SDK, CMake 3.31.6, Ninja, g++ 13.3, glm 1.0.1 headers downloaded to agent_space. One GPU pinned by the dispatcher (`HIP_VISIBLE_DEVICES` kept), `ulimit -c 0`. Fork clone checked out detached at 7c09435fcf06ec3351ae6757c25d1b3b201a2134; `git status --porcelain` empty afterwards.

```bash
ROCM=/opt/python-3.11-venv/lib/python3.11/site-packages/_rocm_sdk_devel
GLM=agent_space/glm-dl/glm-1.0.1
cmake -B agent_space/build-hip -S projects/ohm/src -G Ninja -DCMAKE_BUILD_TYPE=Release -DOHM_FEATURE_HIP=ON -DOHM_FEATURE_OPENCL=OFF -DOHM_FEATURE_TEST=ON -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$ROCM -DGLM_INCLUDE_DIRS=$GLM
utils/timeit.sh ohm compile -- cmake --build agent_space/build-hip -j 32     # 231/231, no errors
export LD_LIBRARY_PATH=/opt/python-3.11-venv/lib/python3.11/site-packages/_rocm_sdk_core/lib
for t in gputiltestcuda ohmtestcuda ohmtest ohmtestheightmap slamiotest; do utils/timeit.sh ohm test -- agent_space/build-hip/bin/$t; done
```

Arch was auto-detected: `CMAKE_HIP_ARCHITECTURES=gfx1100;gfx1100;gfx1100;gfx1100` (one entry per GPU on the host). The `LD_LIBRARY_PATH` is the TheRock pip-layout RPATH quirk the porter recorded; without it the binaries exit 127.

| Suite | gfx1100 at 7c09435 | porter, gfx90a at ea8bbea (same tree) |
|-------|--------------------|----------------------------------------|
| gputiltestcuda | 8/8 (4.1 s) | 8/8 |
| ohmtestcuda | 58/58 (43.5 s) | 58/58 (70.4 s) |
| ohmtest | 53/53 | 53/53 |
| ohmtestheightmap | 19/19 | 19/19 |
| slamiotest | 4/4 | 4/4 |

GPU 66/66, CPU 76/76. Matches the porter, the gfx942 reviewer corroboration, and this platform's 2026-06-05 counts for the first port (gputiltesthip 8/8, ohmtesthip 58/58).

CPU-only build (no GPU feature; cache shows `OHM_FEATURE_CUDA=OFF`, `OHM_FEATURE_HIP=OFF`, no cuda targets built):

```bash
cmake -B agent_space/build-cpu -S projects/ohm/src -G Ninja -DCMAKE_BUILD_TYPE=Release -DOHM_FEATURE_OPENCL=OFF -DOHM_FEATURE_TEST=ON -DGLM_INCLUDE_DIRS=$GLM
utils/timeit.sh ohm compile -- cmake --build agent_space/build-cpu -j 16   # 161/161
```

ohmtest 53/53, ohmtestheightmap 19/19, slamiotest 4/4.

Checks: `python3 utils/jargon.py --port ohm` clean (after creating local `master`/`moat-port` refs from origin in the fresh clone). readme.md has the "Building for AMD GPUs" section next to the CUDA build instructions.

CUDA gate: cuda-not-validated on this host. There is no CUDA toolkit, and the fleet guard refused installing pip nvcc wheels into the job venv. The CUDA path evidence at this tree remains the 18/18 byte-identical preprocessing check (porter and gfx90a reviewer; the tree of 7c09435 equals ea8bbea). An nvcc compile with `-DCMAKE_CUDA_ARCHITECTURES=80` is still owed by a host that has the toolkit.
## Validation 2026-11-04 (linux-gfx90a, moat-port 7c09435)

**GPU**: AMD Instinct MI250X (gfx90a), one GCD pinned by the dispatcher (`HIP_VISIBLE_DEVICES` kept), `ulimit -c 0`. **Toolchain**: ROCm 10.1 TheRock pip SDK (`_rocm_sdk_devel`, HIP clang 24.0.0), CMake 3.31.6, Ninja, g++ 13.3, glm 1.0.1 headers downloaded to agent_space. Fork clone checked out detached at 7c09435fcf06ec3351ae6757c25d1b3b201a2134 (== `origin/moat-port` per `git ls-remote`); `git status --porcelain` empty after both builds.

HIP build (arch auto-detected: `CMAKE_HIP_ARCHITECTURES=gfx90a;gfx90a;gfx90a;gfx90a`):

```bash
ROCM=/opt/python-3.11-venv/lib/python3.11/site-packages/_rocm_sdk_devel
cmake -B agent_space/build-hip -S projects/ohm/src -G Ninja -DCMAKE_BUILD_TYPE=Release -DOHM_FEATURE_HIP=ON -DOHM_FEATURE_OPENCL=OFF -DOHM_FEATURE_TEST=ON -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$ROCM -DGLM_INCLUDE_DIRS=$PWD/agent_space/dl/glm-1.0.1
utils/timeit.sh ohm compile -- cmake --build agent_space/build-hip -j 64     # 231/231
export LD_LIBRARY_PATH=$ROCM/lib   # TheRock soname quirk, see the masquerade port section
utils/timeit.sh ohm test -- $PWD/agent_space/build-hip/bin/<suite>
```

| Suite | Result |
|-------|--------|
| gputiltestcuda | 8/8 PASSED (3 suites, 3.5 s) |
| ohmtestcuda | 58/58 PASSED (16 suites, 70.1 s) |
| ohmtest | 53/53 PASSED (18 suites) |
| ohmtestheightmap | 19/19 PASSED |
| slamiotest | 4/4 PASSED |

Matches the porter's counts at the same tree (ea8bbea, tree c13dfc8) and this platform's earlier record (66/66 GPU at 141e356).

CPU-only build (no GPU feature; cache shows `OHM_FEATURE_CUDA/HIP/OPENCL=OFF`):

```bash
cmake -B agent_space/build-cpu -S projects/ohm/src -G Ninja -DCMAKE_BUILD_TYPE=Release -DOHM_FEATURE_OPENCL=OFF -DOHM_FEATURE_TEST=ON -DGLM_INCLUDE_DIRS=$PWD/agent_space/dl/glm-1.0.1
utils/timeit.sh ohm compile -- cmake --build agent_space/build-cpu -j 48     # 161/161
```

ohmtest 53/53, ohmtestheightmap 19/19, slamiotest 4/4 PASSED.

Checks: `python3 utils/jargon.py --port ohm` clean (needed local `master`/`moat-port` branches in a fresh clone). readme.md has the "Building for AMD GPUs" section. CUDA gate: cuda-not-validated on this host (no CUDA toolkit and no conda); the reviewer's 18/18 byte-identical preprocessing check at 7c09435 is the CUDA evidence for this head.
