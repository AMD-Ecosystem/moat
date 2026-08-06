# CUDA-ScanMatcher-ICP notes

## Build (HIP/ROCm)

```bash
cd src
mkdir build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

For other architectures, override CMAKE_HIP_ARCHITECTURES:
```bash
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100  # RDNA3
```

## Test (non-visual mode)

The project is a visualization demo with no automated tests. For headless validation,
set `VISUALIZE 0` in main.cpp (already set for this port):

```bash
HIP_VISIBLE_DEVICES=2 ./bin/cis565_ScanMatching
# Expected output: 10 ICP iterations with ~480us NN timing per step
```

## Port details

- Strategy A: single cuda_to_hip.h compat header
- All sources compiled as HIP (required because hip::host target pollutes CXX with HIP flags)
- GLM device support via glm_device.h (defines __CUDACC__ + CUDA_VERSION for GLM's qualifier detection)
- Include order matters: Thrust before GLM (rocThrust backend selection checks __CUDACC__ before __HIP__)
- GL-HIP interop updated to modern graphics resource API (hipGraphicsGLRegisterBuffer)
- Headless testing works with VISUALIZE=0 (GL interop fails without a real GPU-backed display)

## Gotchas

1. rocThrust/rocPRIM require C++17; upstream used C++11
2. svd3.h uses rsqrt() which is device-only on HIP; replaced with 1/sqrtf()
3. Kernel launch syntax `<< <` (with space) rejected by hipcc; fixed to `<<<`
4. Old cudaGL* interop functions (cudaGLSetGLDevice, cudaGLRegisterBufferObject) do not exist
   on HIP; ported to modern hipGraphicsGLRegisterBuffer/MapResources API
5. hip::host CMake target adds USE_PROF_API=1 which breaks hip_prof_str.h under plain gcc;
   compile all sources as HIP to avoid

## Review 2026-06-05

### Port Correctness

1. **main.cpp:80** -- Direct use of `hipSetDevice(gpuDevice)` instead of going through the compat header. This breaks the CUDA build because `hipSetDevice` is not defined without HIP headers. Should use `cudaSetDevice` and add the mapping to cuda_to_hip.h.

2. **main.cpp:133-139** -- Mixed HIP/CUDA symbols in the same code block: uses `cudaGraphicsGLRegisterBuffer` (compat-mapped) but directly uses `hipError_t`, `hipSuccess`, and `hipGetErrorString` without mappings. These HIP-specific symbols will cause CUDA build failures. The error handling code should use the CUDA spellings (`cudaError_t`, `cudaSuccess`, `cudaGetErrorString`) which the compat header already maps.

### Minimal Footprint

3. **CMakeLists.txt:112** -- All sources including host `.cpp` files are marked as `LANGUAGE HIP`. Per PORTING_GUIDE Strategy A, only `.cu` files should be marked HIP; host C++ should remain untouched by the HIP toolchain. The current approach compiles main.cpp, glslUtility.cpp, and utilityCore.cpp with hipcc, which is unnecessary and deviates from the minimal-footprint principle.

### Backward Compatibility (upstream)

4. **The CUDA build is broken** -- Due to issues #1 and #2 above, building with `USE_HIP=OFF` will fail. The compat header should map:
   - `cudaSetDevice` -> `hipSetDevice`
   - And code should use CUDA spellings consistently, letting the compat header do the translation.

### Recommendation

**Request Changes**

The port breaks the CUDA build path due to direct use of HIP-specific symbols in main.cpp. All four issues must be fixed:
- Add `cudaSetDevice` mapping to cuda_to_hip.h
- Replace `hipSetDevice` with `cudaSetDevice` in main.cpp:80
- Replace `hipError_t`/`hipSuccess`/`hipGetErrorString` with `cudaError_t`/`cudaSuccess`/`cudaGetErrorString` in main.cpp:133-139
- Consider marking only `.cu` files as LANGUAGE HIP (minor, but per Strategy A guidelines)

## Fix 2026-06-05

Resolved review findings:
1. Added `cudaSetDevice` mapping to cuda_to_hip.h
2. Changed `hipSetDevice(gpuDevice)` to `cudaSetDevice(gpuDevice)` in main.cpp:80
3. Changed `hipError_t`/`hipSuccess`/`hipGetErrorString` to `cudaError_t`/`cudaSuccess`/`cudaGetErrorString` in main.cpp:133-139

The port now uses CUDA spellings consistently; cuda_to_hip.h performs translation on HIP builds. CUDA build path (USE_HIP=OFF) should work unchanged.

Kept all sources as LANGUAGE HIP because gotcha #5: hip::host CMake target adds USE_PROF_API=1 which breaks hip_prof_str.h under plain gcc. Compiling .cpp files with hipcc avoids this.

## Review 2026-06-05 (re-review after fixes)

Previous review findings resolved:
1. cudaSetDevice mapping added to cuda_to_hip.h
2. main.cpp:80 now uses cudaSetDevice (mapped via compat header)
3. main.cpp:133-139 now uses cudaError_t/cudaSuccess/cudaGetErrorString (mapped via compat header)

Fault classes verified:
- No warp intrinsics or hardcoded warpSize=32
- No textures, no pitch alignment concerns
- No resource handle rule-of-five issues
- No OOB neighbor reads requiring clamps

LANGUAGE HIP on all sources (including host .cpp) justified by gotcha #5.

Ready for GPU validation.

## Validation 2026-06-05

### Platform: linux-gfx90a

**Build**: gfx90a
```bash
cd src
rm -rf build && mkdir build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

Build succeeded with warnings (nodiscard return values ignored, not critical).
llvm-objdump confirms gfx90a code objects embedded.

**Test**: HIP_VISIBLE_DEVICES=2 (MI250X gfx90a)
```bash
HIP_VISIBLE_DEVICES=2 ./bin/cis565_ScanMatching
```

**Results**: PASS
- 10 ICP iterations completed successfully
- NN timing: 14320us (first), ~447-546us (warmup), then stable ~481-484us per iteration
- No crashes, no errors, no NaN values
- GPU computation works correctly on gfx90a

### Platform: linux-gfx1100

**Build**: gfx1100
```bash
cd src
rm -rf build && mkdir build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_BUILD_TYPE=Release -Dglfw3_DIR=/usr/lib/x86_64-linux-gnu/cmake/glfw3
cmake --build . --parallel
```

Build succeeded with warnings (nodiscard return values ignored, same as gfx90a).
Binary contains gfx1100 code objects.

**Test**: HIP_VISIBLE_DEVICES=0 (Radeon Pro W7800 gfx1100)
```bash
HIP_VISIBLE_DEVICES=0 ./bin/cis565_ScanMatching
```

**Results**: PASS
- 10 ICP iterations completed successfully
- NN timing: 11375us (first), then stable ~252-254us per iteration
- No crashes, no errors, no NaN values
- GPU computation works correctly on gfx1100

## Validation 2026-06-07

### Platform: windows-gfx1201

**GPU**: AMD Radeon RX 9070 XT (gfx1201, RDNA4) at HIP_VISIBLE_DEVICES=0 (V710 offline this session)

**Build**: gfx1201, all-clang (clang.exe/clang++.exe from _rocm_sdk_devel), Ninja
```cmd
cd src
mkdir build_gfx1201 && cd build_gfx1201
cmake .. -G Ninja \
  -DCMAKE_C_COMPILER=<rocm_devel>/lib/llvm/bin/clang.exe \
  -DCMAKE_CXX_COMPILER=<rocm_devel>/lib/llvm/bin/clang++.exe \
  -DCMAKE_HIP_COMPILER=<rocm_devel>/lib/llvm/bin/clang++.exe \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_PREFIX_PATH=<rocm_devel> -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel 32
```

Copied TheRock DLLs (amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll,
hiprtc-builtins0714.dll) into bin/ so the exe loads them instead of System32's amdhip64.

Build succeeded with warnings (nodiscard return values ignored, same as Linux platforms).

**Test**: HIP_VISIBLE_DEVICES=0 (RX 9070 XT gfx1201)
```cmd
set HIP_VISIBLE_DEVICES=0
bin\cis565_ScanMatching.exe
```

**Results**: PASS
- 10 ICP iterations completed successfully
- NN timing: 3688us (first, JIT compile overhead), then stable ~578-588us per iteration
- No crashes, no errors, no NaN values
- GPU computation works correctly on gfx1201

## PR-prep 2026-06-11

Prep edits committed on top of validated head 52e8006, then squashed.

Jargon scrub (whole base..HEAD diff): only one leak found and fixed --
CMakeLists.txt comment said "followers override via -DCMAKE_HIP_ARCHITECTURES";
reworded to spell out the per-architecture override (gfx90a default, gfx1100
RDNA3, gfx1201 RDNA4) with no MOAT vocabulary. Source-file diffs were clean.
W.cu has spaced `<< <` launches but is upstream-untouched and not in the build
(not in cuda_sources), so it was left alone.

Attribution: added `Copyright (c) 2026 Advanced Micro Devices, Inc.` and
`@author Jeff Daily` (Doxygen house style) to the two
substantially-new headers src/cuda_to_hip.h and src/glm_device.h.

Docs: added an "AMD GPUs (ROCm/HIP)" subsection under README "Build
Instructions", parallel to the existing Windows and Linux CUDA subsections,
in the project's descriptive course-README style (USE_HIP, CMAKE_HIP_ARCHITECTURES).

CMake: left the existing arch handling (default gfx90a + override) -- it is
already reasonable house style; only reworded its comment.

Validation smoke (gfx90a, GCD 0, MI250X): rebuilt clean with USE_HIP=ON
gfx90a; headless run completed 10 ICP iterations, NN timing ~440us per step,
no errors. Behavior-preserving prep confirmed.

advance-head 2822242 classified inert (doc/comment only): carried
linux-gfx90a, linux-gfx1100, windows-gfx1201 forward. Squashed to one
tree-identical commit ce20c461b35aaa5781724cc0fd2742f8604c4b89, force-pushed
to moat-port; squash-carry-forward carried all three required platforms.
pr-ready=True. Upstream default branch = master. PR draft at pr-draft.md
(not opened -- awaiting approval).

## Validation 2026-06-19

### Platform: windows-gfx1101

**GPU**: AMD Radeon PRO V710 (gfx1101, RDNA3) at HIP_VISIBLE_DEVICES=1 (RX 9070 XT at mask 0)

**Build**: gfx1101, all-clang (clang.exe/clang++.exe from _rocm_sdk_devel), Ninja

Build dir ended up at MOAT repo root (timeit.sh resets cwd; cmake was invoked from within timeit.sh). CMakeCache.txt confirms gfx1101 arch and USE_HIP=ON.

```cmd
cd B:\develop\moat
cmake B:\develop\moat\projects\CUDA-ScanMatcher-ICP\src -G Ninja \
  -DCMAKE_C_COMPILER=<rocm_devel>/lib/llvm/bin/clang.exe \
  -DCMAKE_CXX_COMPILER=<rocm_devel>/lib/llvm/bin/clang++.exe \
  -DCMAKE_HIP_COMPILER=<rocm_devel>/lib/llvm/bin/clang++.exe \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
  -DCMAKE_PREFIX_PATH=<rocm_devel> -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel 64
```

Copied TheRock DLLs (amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll,
hiprtc-builtins0714.dll) from _rocm_sdk_core/bin into bin/ so the exe loads them
instead of System32's amdhip64.

Build succeeded with warnings (nodiscard return values ignored, same as other platforms).

**Test**: HIP_VISIBLE_DEVICES=1 (Radeon PRO V710 gfx1101), headless (VISUALIZE=0)
```cmd
set HIP_VISIBLE_DEVICES=1
bin\cis565_ScanMatching.exe
```

**Results**: PASS
- 10 ICP iterations completed successfully
- NN timing: 2863us (first, JIT compile overhead), then stable ~267-277us per iteration
- No crashes, no errors, no NaN values
- GPU computation works correctly on gfx1101
- Post-test hipInfo health check: GPU still healthy (no TDR event)
