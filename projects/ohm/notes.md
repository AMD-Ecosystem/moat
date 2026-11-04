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
