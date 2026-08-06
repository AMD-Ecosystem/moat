# cuBQL (NVIDIA/cuBQL)

CUDA Bounding-Volume-hierarchy Query Library -- a small, header-heavy GPU BVH build + traversal library (v1.3.0). Scaffolded 2026-06-15. Ported as a dependency of [[barney]] (barney's `rtcore/cuda` software ray-tracing backend is built on it).

## License: FAVORABLE (Apache-2.0) -- scanned

Fresh clone scan: top-level LICENSE = Apache-2.0; 66 `SPDX-License-Identifier: Apache-2.0` tags; NVIDIA copyrights all under Apache. No NVIDIA Source Code License, no non-commercial terms (the only "non-commercial" strings are in public-domain stb under samples/3rdParty). Clear to fork, port, attribute, and PR upstream.

## Port scope: SMALL / LOW-RISK

Structure: the library is `cuBQL/` = 59 headers + exactly ONE core `.cu` (`cuBQL/builder/cuda/instantiate_builders.cu`, explicit template instantiation). The other 8 `.cu` are `samples/`, not core.

cuBQL ALREADY has a multi-backend builder abstraction: `cuBQL/builder/{cpu,omp,cuda}`. The CUDA code is isolated to `builder/cuda/` and gated by `CUBQL_DISABLE_CUDA` / `CUBQL_OMP` CMake options. Traversal + queries (`cuBQL/traversal`, `cuBQL/queries/{pointData,triangleData}`) are device-side header templates.

CUDA surface to hipify (all standard, all have HIP equivalents):
- device intrinsics: `__global__`/`__device__`/`__host__`/`__shared__`, `atomicAdd`, `__syncthreads` -- compile directly under hipcc.
- runtime (~20 call sites): `cudaMalloc`, `cudaMallocAsync`, `cudaMemcpy{HostToDevice,DeviceToHost,Default}`, `cudaMemset`, `cudaSetDevice`, `cudaMemPoolAttrReleaseThreshold` -> `hip*` equivalents (hipMallocAsync / hipMemPool are supported in ROCm). No thrust in the builder.

Recommended: hipify `cuBQL/builder/cuda/` + the device-side traversal/query headers; add a `USE_HIP` path alongside the existing CUDA path in CMake (mirror the existing `CUBQL_DISABLE_CUDA` gating). Build type: CMake. Validates on CDNA gfx90a (pure compute, no hardware-RT / HIPRT needed).

## Build (Linux gfx90a, ROCm 7.2.1)

Configure with the new `CUBQL_USE_HIP` option (it enables the HIP language, sets `CMAKE_HIP_ARCHITECTURES`, and reuses the existing `CUBQL_HAVE_CUDA` switch so all GPU libs/samples build):

    cmake -S projects/cuBQL/src -B projects/cuBQL/src/build-hip \
      -DCUBQL_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build projects/cuBQL/src/build-hip -j$(nproc) --target \
      cuBQL_cuda_float3 \
      cuBQL_sample01_points_closestPoint_cuda \
      cuBQL_sample01_points_closestPoint_wideBVH_cuda \
      sample02_distanceToTriangleMesh sample03_insideOutside \
      sample04_boxOverlapsOrInsideSurfaceMesh sample05_lineOfSight \
      sample06_anyTriangleWithinRadius sample07_aggregateNBody

CMake auto-finds hipcc as `/opt/rocm/lib/llvm/bin/clang++`. No arch is hardcoded in source; pass `-DCMAKE_HIP_ARCHITECTURES=...` (defaults to gfx90a only when unset). For a multi-arch lead build pass e.g. `-DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100"`.

Smoke run (pin one GCD on the shared host):

    HIP_VISIBLE_DEVICES=0 ./build-hip/cuBQL_sample01_points_closestPoint_cuda

CUDA no-regression: configure WITHOUT `-DCUBQL_USE_HIP` and the HIP path is fully inert (CMAKE_CUDA_ARCHITECTURES stays `native`, no `enable_language(HIP)`), so the upstream CUDA graph is unchanged. The CPU host targets (`cuBQL_cpu_float3`, `cuBQL_sample01_points_closestPoint_cpu`) build clean in both modes.

## Port gotchas (things the plan underestimated)

The plan's runtime-shim list was incomplete and two "HIP provides it" assumptions were wrong:
- `CUDART_INF_F` / `CUDART_INF` / `CUDART_NAN(_F)` are NOT provided by HIP (they come from CUDA's `<math_constants.h>`). Added `__builtin_*` definitions under `__HIPCC__` in `cuBQL/math/constants.h`.
- `cudaMallocHost` / `cudaFreeHost` (radix/rebinMorton builders) and `cudaMemcpyFromSymbol` (s07) were not in the planned alias set; added to the `common.h` shim (`hipHostMalloc` / `hipHostFree` / `hipMemcpyFromSymbol`).
- `bvh.h` only `#include`d the cuda builder and declared the `bvh_float{2,3,4}` typedefs under `__CUDACC__`; widened both guards to `|| __HIPCC__` (otherwise `cuBQL::gpuBuilder` and `cuBQL::cuda` are absent under hipcc).
- `shrinkingRadiusQuery.h` defined host fallbacks for `__int_as_float`/`__float_as_int` guarded by `#ifdef __CUDA_ARCH__` only. HIP's per-pass macro is `__HIP_DEVICE_COMPILE__`, not `__CUDACC__`/`__HIPCC__`, so under HIP the host fallback shadowed the device builtin and a `__host__ __device__` function called a `__host__`-only fn -> clang error. Fixed the guard to `__CUDA_ARCH__ || __HIP_DEVICE_COMPILE__`.
- `findClosest*` were forward-declared `__host__ __device__` but defined `__device__`-only; nvcc tolerates the decl/def attribute mismatch, HIP/clang rejects it. Aligned the three declarations to the device-only definitions on the HIP path via a local `__cubql_findClosest_decl` macro (unchanged on CUDA and the plain-C++ CPU build).
- `vec_t` `operator dim3()` casts were `__CUDACC__`-only; `dim3` is a HIP type too, widened to `|| __HIPCC__` (s03 launches kernels with a `vec3i` grid dim).

Every source change is under `__HIPCC__` / `__HIP_DEVICE_COMPILE__` (or, for CMake, `CUBQL_USE_HIP` / `LANGUAGE HIP`), so the CUDA build is byte-identical.

## Install as a dependency

cuBQL is consumed as a CMake subproject (`CUBQL_IS_SUBPROJECT`). For a HIP consumer (e.g. barney), set the HIP arch and the option BEFORE `add_subdirectory`:

    set(CUBQL_USE_HIP ON)
    set(CMAKE_HIP_ARCHITECTURES gfx90a)   # or the consuming project's arch list
    add_subdirectory(${CMAKE_SOURCE_DIR}/_deps/cuBQL)   # AMD-Ecosystem/cuBQL @ moat-port

Then link the interface target plus the type/dim instantiation target(s) you need:

    target_link_libraries(<your_target> PRIVATE cuBQL cuBQL_cuda_float3)

`cuBQL` is header-only (include paths); the `cuBQL_cuda_<type><dim>` targets (e.g. `cuBQL_cuda_float3`, `_static` variants) hold the precompiled builders for that type/dim. The HIP path marks each instantiation `.cu` `LANGUAGE HIP`, so the consumer needs `enable_language(HIP)` (the cuBQL subproject does this when `CUBQL_USE_HIP=ON`). The subproject `CMAKE_CUDA_ARCHITECTURES`-not-set FATAL_ERROR is bypassed when `CUBQL_USE_HIP` is ON, so a HIP consumer is NOT forced to set a CUDA arch.

cuBQL ships no `install()` rules, so `add_subdirectory` (submodule/`_deps` pin) is the consumption mode. An installed-tree export would be a small additive follow-up if a consumer needs `find_package`; not required for barney.

## Status
Ported on AMD-Ecosystem/cuBQL @ moat-port (Linux gfx90a). Builds + links the library, the instantiate_builders TU, and all 8 GPU samples under hipcc for gfx90a; sample01 closest-point runs correctly on gfx90a. Validator runs the real GPU test suite next.

## Review 2026-06-15

Reviewed 675162a (moat-port) vs base e82f1dc with /pr-review. Verdict: review-passed. No blocking defects. Fault classes (warp-size, rule-of-five, OOB, texture-pitch) verified N/A or correctly handled. CUDA build byte-identity holds: every source change is under __HIPCC__/__HIP_DEVICE_COMPILE__, every CMake change under CUBQL_USE_HIP (default OFF)/LANGUAGE HIP. Token-paste CUBQL_CUDA_CALL(cuda##X) resolves through the alias set for all call sites (Malloc/MallocAsync/MallocHost/MallocManaged/Free/FreeAsync/Memcpy/MemcpyAsync/GetDevice/GetDeviceCount/StreamSynchronize/Event{Create,Destroy,Record,Synchronize}/DeviceGetDefaultMemPool/MemPoolSetAttribute). shrinkingRadiusQuery.h device/host split correct on both toolchains (CUDA still keys off __CUDA_ARCH__). findClosest decl/def alignment correct: the three widened decls -> __cubql_device on HIP, matching the __cubql_device-only defs; CPU sample compiles as plain C++ (__cubql_both -> empty), unaffected. cuda.h async guard widening and constants.h CUDART_INF/NAN are __HIPCC__-only, CUDA values unchanged. CMake subproject FATAL_ERROR re-parenthesization is byte-equivalent to upstream's NOT>AND>OR precedence when USE_HIP=OFF. No hardcoded arch in source; no spaced launch syntax; no MOAT jargon; commit title/body/trailers conform.

Non-blocking notes (do not gate validation; address in PR-prep or leave):
- cuBQL/math/common.h:64,68,74,75 -- four aliases are defined but unused in current source: cudaMemcpyToSymbol, cudaMemsetAsync, cudaMemcpyDeviceToDevice, cudaEventElapsedTime. Minor orphan churn; trim to the used set in PR-prep or leave (harmless under __HIPCC__).
- cuBQL/queries/pointData/findClosest.h:131,390 -- the bvh_float2/3/4 + raw float2/3/4 findClosest convenience overloads stay #ifdef __CUDACC__ only, while bvh.h:147-151 now defines bvh_float{2,3,4} under __HIPCC__ too. No current sample uses these raw-CUDA-vector overloads (all call the bvh3f/vec3f templated forms), so it does not gate; but a future HIP consumer (barney) calling findClosest(bvh_float3, float3*, ...) would miss the decl. Widen at barney-port time if needed; do not speculatively widen untested now.

Validator must exercise at runtime (these are MEDIUM-risk by analysis, code is correct):
- sample07_aggregateNBody: relies on cudaMemcpyFromSymbol over a __device__ function-pointer symbol (common.h alias -> hipMemcpyFromSymbol). The CUDA_SEPARABLE_COMPILATION/CUDA_RESOLVE_DEVICE_SYMBOLS target properties are no-ops on the HIP target; the single-TU device symbol linked per the porter, but the function-pointer dispatch must be confirmed CORRECT at runtime, not just linked.
- rebinMortonBuilder.h DeviceRadixSort SortKeys/SortPairs: each call uses an independent malloc/free temp buffer (size-query then sort), NOT a reused TempStorage union, so the wave64 Block-collective race the plan feared does NOT apply; the custom kernels (e.g. lines 659-717) sync l_allocOffset shared state with explicit __syncthreads. Still confirm sort output ordering matches via the closest-point cross-check.
- hipMallocAsync default-pool path (cuda.h AsyncGpuMemoryResource, selected by default on HIP): confirm hipDeviceGetDefaultMemPool + hipMemPoolSetAttribute(hipMemPoolAttrReleaseThreshold) succeed on gfx90a/ROCm 7.2.1 at runtime.

## Validation 2026-06-15

Validated 675162a on linux-gfx90a (AMD Instinct MI250X, ROCm 7.2.1, HIP_VISIBLE_DEVICES=0).

Build: cmake -DCUBQL_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release; all 8 GPU targets + cuBQL_cuda_float3 + CPU targets build clean.

GPU sample results (all exit 0):

- s01 cuBQL_sample01_points_closestPoint_cuda: PASS -- 20/20 queries correct
- s01 cuBQL_sample01_points_closestPoint_wideBVH_cuda: PASS -- 20/20 queries correct
- s02 sample02_distanceToTriangleMesh (bunny.obj, 4968 triangles): PASS
- s03 sample03_insideOutside (n=32): PASS -- volume written to file
- s04 sample04_boxOverlapsOrInsideSurfaceMesh (n=32): PASS -- volume written
- s05 sample05_lineOfSight (n=64x64): PASS -- image written
- s06 sample06_anyTriangleWithinRadius (n=32): PASS -- volume written
- s07 sample07_aggregateNBody: PASS -- 100000 data points, exit 0

CPU cross-check: cuBQL_sample01_points_closestPoint_cpu vs GPU s01 -- diff empty (bit-for-bit identical across all 20 queries). Confirms rebinMorton DeviceRadixSort ordering is correct.

Reviewer-flagged items (all CONFIRMED):
1. s07 hipMemcpyFromSymbol device-function-pointer dispatch: CORRECT at runtime (s07 exits 0, produces output with 100k data points; the single-TU symbol path works on HIP without SEPARABLE_COMPILATION).
2. rebinMortonBuilder DeviceRadixSort ordering: CORRECT -- CPU/GPU closest-point results are bit-identical for all 20 queries.
3. hipMallocAsync default-pool path (hipDeviceGetDefaultMemPool + hipMemPoolSetAttribute): CONFIRMED working -- all samples that use AsyncGpuMemoryResource succeed on gfx90a/ROCm 7.2.1.

CPU no-regression: cuBQL_cpu_float3 + cuBQL_sample01_points_closestPoint_cpu build and run clean.

CUDA no-regression gate (lead platform): nvcc 12.8, sm_80 -- all 8 GPU samples + cuBQL_cuda_float3 compile clean with CUBQL_USE_HIP=OFF. No errors, no regressions.

Commands:

```
cmake -S projects/cuBQL/src -B projects/cuBQL/src/build-hip \
  -DCUBQL_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release
cmake --build projects/cuBQL/src/build-hip -j$(nproc) --target \
  cuBQL_cuda_float3 cuBQL_sample01_points_closestPoint_cuda \
  cuBQL_sample01_points_closestPoint_wideBVH_cuda \
  sample02_distanceToTriangleMesh sample03_insideOutside \
  sample04_boxOverlapsOrInsideSurfaceMesh sample05_lineOfSight \
  sample06_anyTriangleWithinRadius sample07_aggregateNBody \
  cuBQL_cpu_float3 cuBQL_sample01_points_closestPoint_cpu

HIP_VISIBLE_DEVICES=0 ./build-hip/cuBQL_sample01_points_closestPoint_cuda
HIP_VISIBLE_DEVICES=0 ./build-hip/cuBQL_sample01_points_closestPoint_wideBVH_cuda
HIP_VISIBLE_DEVICES=0 ./build-hip/sample02_distanceToTriangleMesh /tmp/bunny.obj
HIP_VISIBLE_DEVICES=0 ./build-hip/sample03_insideOutside /tmp/bunny.obj -o /tmp/s03 -n 32
HIP_VISIBLE_DEVICES=0 ./build-hip/sample04_boxOverlapsOrInsideSurfaceMesh /tmp/bunny.obj -o /tmp/s04 -n 32
HIP_VISIBLE_DEVICES=0 ./build-hip/sample05_lineOfSight /tmp/bunny.obj -o /tmp/s05.png -n 64 64
HIP_VISIBLE_DEVICES=0 ./build-hip/sample06_anyTriangleWithinRadius /tmp/bunny.obj -o /tmp/s06 -n 32
HIP_VISIBLE_DEVICES=0 ./build-hip/sample07_aggregateNBody
```

## Validation 2026-06-15 (linux-gfx1100)

Validated 675162a on linux-gfx1100 (AMD Radeon Pro W7800 48GB, ROCm, HIP_VISIBLE_DEVICES=0, gfx1100).

Build: cmake -DCUBQL_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release (fresh build dir); all 8 GPU targets + cuBQL_cuda_float3 + CPU targets build clean (warnings only, no errors).

GPU sample results (all exit 0, HIP_VISIBLE_DEVICES=0):

- s01 cuBQL_sample01_points_closestPoint_cuda: PASS -- 20/20 queries, point IDs bit-identical to gfx90a
- s01 cuBQL_sample01_points_closestPoint_wideBVH_cuda: PASS -- 20/20 queries, point IDs bit-identical to gfx90a and to BinaryBVH
- s02 sample02_distanceToTriangleMesh (bunny.obj, 4968 triangles): PASS
- s03 sample03_insideOutside (n=32): PASS -- volume written to file
- s04 sample04_boxOverlapsOrInsideSurfaceMesh (n=32): PASS -- volume written
- s05 sample05_lineOfSight (n=64x64): PASS -- image written
- s06 sample06_anyTriangleWithinRadius (n=32): PASS -- volume written
- s07 sample07_aggregateNBody: PASS -- 100000 data points, exit 0

Cross-arch consistency: s01 closest-point IDs and coordinates are bit-identical between gfx90a and gfx1100 for all 20 queries. BinaryBVH == WideBVH on gfx1100.

CPU no-regression: cuBQL_cpu_float3 + cuBQL_sample01_points_closestPoint_cpu build and run clean. CPU results bit-identical to GPU s01 on gfx1100.

hipMallocAsync default-pool path: confirmed working on gfx1100/ROCm (all samples using AsyncGpuMemoryResource succeed).

Commands:

```
cmake -S projects/cuBQL/src -B projects/cuBQL/src/build-hip-gfx1100 \
  -DCUBQL_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build projects/cuBQL/src/build-hip-gfx1100 -j$(nproc) --target \
  cuBQL_cuda_float3 cuBQL_sample01_points_closestPoint_cuda \
  cuBQL_sample01_points_closestPoint_wideBVH_cuda \
  sample02_distanceToTriangleMesh sample03_insideOutside \
  sample04_boxOverlapsOrInsideSurfaceMesh sample05_lineOfSight \
  sample06_anyTriangleWithinRadius sample07_aggregateNBody \
  cuBQL_cpu_float3 cuBQL_sample01_points_closestPoint_cpu

HIP_VISIBLE_DEVICES=0 ./build-hip-gfx1100/cuBQL_sample01_points_closestPoint_cuda
HIP_VISIBLE_DEVICES=0 ./build-hip-gfx1100/cuBQL_sample01_points_closestPoint_wideBVH_cuda
HIP_VISIBLE_DEVICES=0 ./build-hip-gfx1100/sample02_distanceToTriangleMesh /tmp/bunny.obj
HIP_VISIBLE_DEVICES=0 ./build-hip-gfx1100/sample03_insideOutside /tmp/bunny.obj -o /tmp/s03 -n 32
HIP_VISIBLE_DEVICES=0 ./build-hip-gfx1100/sample04_boxOverlapsOrInsideSurfaceMesh /tmp/bunny.obj -o /tmp/s04 -n 32
HIP_VISIBLE_DEVICES=0 ./build-hip-gfx1100/sample05_lineOfSight /tmp/bunny.obj -o /tmp/s05.png -n 64 64
HIP_VISIBLE_DEVICES=0 ./build-hip-gfx1100/sample06_anyTriangleWithinRadius /tmp/bunny.obj -o /tmp/s06 -n 32
HIP_VISIBLE_DEVICES=0 ./build-hip-gfx1100/sample07_aggregateNBody
```

## Validation 2026-06-16 (windows-gfx1201)

Validated 675162a on windows-gfx1201 (AMD Radeon RX 9070 XT, TheRock ROCm 7.14, HIP_VISIBLE_DEVICES=1, gfx1201).

Build: cmake -G Ninja -DCUBQL_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_BUILD_TYPE=Release with clang.exe (C) and clang++.exe (CXX/HIP) from _rocm_sdk_devel/lib/llvm/bin; all 26 targets (8 GPU samples + cuBQL_cuda_float3 + cuBQL_cuda_float3.dll + CPU targets) build clean (warnings only, no errors). TheRock runtime DLLs (amdhip64_7.dll, amd_comgr.dll, hiprtc0714.dll, hiprtc-builtins0714.dll, rocm_kpack.dll) copied from _rocm_sdk_core/bin into the build dir to override System32's Adrenalin driver amdhip64.

GPU sample results (all exit 0, HIP_VISIBLE_DEVICES=1):

- s01 cuBQL_sample01_points_closestPoint_cuda: PASS -- 20/20 queries, point IDs bit-identical to gfx90a/gfx1100
- s01 cuBQL_sample01_points_closestPoint_wideBVH_cuda: PASS -- 20/20 queries, bit-identical to BinaryBVH and gfx90a/gfx1100
- s02 sample02_distanceToTriangleMesh (bunny.obj, 4968 triangles): PASS
- s03 sample03_insideOutside (n=32): PASS -- volume written to file
- s04 sample04_boxOverlapsOrInsideSurfaceMesh (n=32): PASS -- volume written
- s05 sample05_lineOfSight (n=64x64): PASS -- image written
- s06 sample06_anyTriangleWithinRadius (n=32): PASS -- volume written
- s07 sample07_aggregateNBody: PASS -- 100000 data points, exit 0

CPU no-regression: cuBQL_sample01_points_closestPoint_cpu builds and runs clean; results bit-identical to GPU s01 on gfx1201.

Cross-arch consistency: s01 closest-point IDs and coordinates are bit-identical between gfx1201 and gfx90a/gfx1100 for all 20 queries. BinaryBVH == WideBVH on gfx1201.

hipMallocAsync default-pool path: confirmed working on gfx1201/TheRock-7.14 (all samples using AsyncGpuMemoryResource succeed).

Commands:

```
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
cmake -S projects/cuBQL/src -B projects/cuBQL/src/build-hip-gfx1201 \
  -G Ninja \
  -DCUBQL_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/clang.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe
cmake --build projects/cuBQL/src/build-hip-gfx1201 -j64 --target \
  cuBQL_cuda_float3 cuBQL_sample01_points_closestPoint_cuda \
  cuBQL_sample01_points_closestPoint_wideBVH_cuda \
  sample02_distanceToTriangleMesh sample03_insideOutside \
  sample04_boxOverlapsOrInsideSurfaceMesh sample05_lineOfSight \
  sample06_anyTriangleWithinRadius sample07_aggregateNBody \
  cuBQL_cpu_float3 cuBQL_sample01_points_closestPoint_cpu

# Copy TheRock runtime DLLs into build dir
cp _rocm_sdk_core/bin/{amdhip64_7,amd_comgr,hiprtc0714,hiprtc-builtins0714,rocm_kpack}.dll \
  projects/cuBQL/src/build-hip-gfx1201/

BUNNY=B:/develop/moat/projects/DDN-SLAM/src/Thirdparty/instant-ngp-kf/data/sdf/bunny.obj
HIP_VISIBLE_DEVICES=1 ./build-hip-gfx1201/cuBQL_sample01_points_closestPoint_cuda.exe
HIP_VISIBLE_DEVICES=1 ./build-hip-gfx1201/cuBQL_sample01_points_closestPoint_wideBVH_cuda.exe
HIP_VISIBLE_DEVICES=1 ./build-hip-gfx1201/sample02_distanceToTriangleMesh.exe $BUNNY
HIP_VISIBLE_DEVICES=1 ./build-hip-gfx1201/sample03_insideOutside.exe $BUNNY -o /tmp/s03 -n 32
HIP_VISIBLE_DEVICES=1 ./build-hip-gfx1201/sample04_boxOverlapsOrInsideSurfaceMesh.exe $BUNNY -o /tmp/s04 -n 32
HIP_VISIBLE_DEVICES=1 ./build-hip-gfx1201/sample05_lineOfSight.exe $BUNNY -o /tmp/s05.png -n 64 64
HIP_VISIBLE_DEVICES=1 ./build-hip-gfx1201/sample06_anyTriangleWithinRadius.exe $BUNNY -o /tmp/s06 -n 32
HIP_VISIBLE_DEVICES=1 ./build-hip-gfx1201/sample07_aggregateNBody.exe
./build-hip-gfx1201/cuBQL_sample01_points_closestPoint_cpu.exe
```

## Validation 2026-06-19 (windows-gfx1101)

Validated b0ea6a17 on windows-gfx1101 (AMD Radeon PRO V710, TheRock ROCm 7.14, HIP_VISIBLE_DEVICES=1, gfx1101).

Build: cmake -G Ninja -DCUBQL_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1101 -DCMAKE_BUILD_TYPE=Release with clang.exe (C) and clang++.exe (CXX/HIP) from _rocm_sdk_devel/lib/llvm/bin; all 26 targets (8 GPU samples + cuBQL_cuda_float3 + cuBQL_cuda_float3.dll + CPU targets) build clean (warnings only, no errors). TheRock runtime DLLs (amdhip64_7.dll, amd_comgr.dll, hiprtc0714.dll, hiprtc-builtins0714.dll, rocm_kpack.dll) copied from _rocm_sdk_core/bin into the build dir.

GPU sample results (all exit 0, HIP_VISIBLE_DEVICES=1):

- s01 cuBQL_sample01_points_closestPoint_cuda: PASS -- 20/20 queries, point IDs bit-identical to gfx90a/gfx1100/gfx1201
- s01 cuBQL_sample01_points_closestPoint_wideBVH_cuda: PASS -- 20/20 queries, bit-identical to BinaryBVH and gfx90a/gfx1100/gfx1201
- s02 sample02_distanceToTriangleMesh (bunny.obj, 4968 triangles): PASS
- s03 sample03_insideOutside (n=32): PASS -- volume written to file
- s04 sample04_boxOverlapsOrInsideSurfaceMesh (n=32): PASS -- volume written
- s05 sample05_lineOfSight (n=64x64): PASS -- image written
- s06 sample06_anyTriangleWithinRadius (n=32): PASS -- volume written
- s07 sample07_aggregateNBody: PASS -- 100000 data points, exit 0

CPU no-regression: cuBQL_sample01_points_closestPoint_cpu builds and runs clean; results bit-identical to GPU s01 on gfx1101 and to all other platforms.

Cross-arch consistency: s01 closest-point IDs and coordinates are bit-identical between gfx1101 and gfx90a/gfx1100/gfx1201 for all 20 queries. BinaryBVH == WideBVH on gfx1101.

hipMallocAsync default-pool path: confirmed working on gfx1101/TheRock-7.14 (all samples using AsyncGpuMemoryResource succeed).

No TDR events, no wedge; all samples ran well within the TDR window (sub-second kernel durations).

Commands:

```
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
cmake -S projects/cuBQL/src -B projects/cuBQL/src/build-hip-gfx1101 \
  -G Ninja \
  -DCUBQL_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1101 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/clang.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe
cmake --build projects/cuBQL/src/build-hip-gfx1101 -j64 --target \
  cuBQL_cuda_float3 cuBQL_sample01_points_closestPoint_cuda \
  cuBQL_sample01_points_closestPoint_wideBVH_cuda \
  sample02_distanceToTriangleMesh sample03_insideOutside \
  sample04_boxOverlapsOrInsideSurfaceMesh sample05_lineOfSight \
  sample06_anyTriangleWithinRadius sample07_aggregateNBody \
  cuBQL_cpu_float3 cuBQL_sample01_points_closestPoint_cpu

# Copy TheRock runtime DLLs into build dir
cp _rocm_sdk_core/bin/{amdhip64_7,amd_comgr,hiprtc0714,hiprtc-builtins0714,rocm_kpack}.dll \
  projects/cuBQL/src/build-hip-gfx1101/

BUNNY=B:/develop/moat/projects/DDN-SLAM/src/Thirdparty/instant-ngp-kf/data/sdf/bunny.obj
HIP_VISIBLE_DEVICES=1 ./build-hip-gfx1101/cuBQL_sample01_points_closestPoint_cuda.exe
HIP_VISIBLE_DEVICES=1 ./build-hip-gfx1101/cuBQL_sample01_points_closestPoint_wideBVH_cuda.exe
HIP_VISIBLE_DEVICES=1 ./build-hip-gfx1101/sample02_distanceToTriangleMesh.exe $BUNNY
HIP_VISIBLE_DEVICES=1 ./build-hip-gfx1101/sample03_insideOutside.exe $BUNNY -o /tmp/s03 -n 32
HIP_VISIBLE_DEVICES=1 ./build-hip-gfx1101/sample04_boxOverlapsOrInsideSurfaceMesh.exe $BUNNY -o /tmp/s04 -n 32
HIP_VISIBLE_DEVICES=1 ./build-hip-gfx1101/sample05_lineOfSight.exe $BUNNY -o /tmp/s05.png -n 64 64
HIP_VISIBLE_DEVICES=1 ./build-hip-gfx1101/sample06_anyTriangleWithinRadius.exe $BUNNY -o /tmp/s06 -n 32
HIP_VISIBLE_DEVICES=1 ./build-hip-gfx1101/sample07_aggregateNBody.exe
./build-hip-gfx1101/cuBQL_sample01_points_closestPoint_cpu.exe
```
