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

## Follow-up round 2026-11-19 (porter, linux-gfx950): rocPRIM bit-range sort workaround

Requested by jeff: prepare and validate the cuBQL side of the rocPRIM radix sort defect (ROCm/rocm-libraries#16299) on the fork. He has NOT approved opening an upstream PR; the draft below waits for his decision. Background, root cause and the original patch: the claude-shared plan `rocm-defects-from-warp-port-reproducers-mi355.md` ("Defect A", "cuBQL impact").

### Record path taken

- PR #35 was merged by a real merge commit (`d1bfc3c`, parents `e82f1dc` + `b0ea6a1`), not squashed: `published_sha` `b0ea6a1` is an ancestor of upstream main, so the mirror could be synced past it.
- The folder was trunk-resident at `review-passed` with no `port/cuBQL`. No CLI command re-homes a merged (not maintained, not open-PR) project before the lock is taken: `fix-branch` needs an open PR, `sync-branch` a maintained fork, and `advance-head` re-homes but porter.md says to run it from inside `porting`, while `set-state ... porting` on `main` would commit the lock to the trunk. Ran the tooling's own `moatlib.ensure_port_branch('cuBQL')` (the step `advance-head`/`fix-branch` run first; it only pushes `port/cuBQL` cut from origin/main, no state change), checked out `port/cuBQL`, ran orient (it offered nothing for cuBQL, as expected), then `set-state cuBQL linux-gfx950 porting` (legal `review-passed -> porting` edge).
- `moatlib.py mirror-branch cuBQL`: fork `main` fast-forwarded `e82f1dc -> 6bd4a51` (upstream main, 2026-11-14).
- `moat-port`: `git merge --ff-only upstream/main` (b0ea6a1 is an ancestor, so the merge is a fast-forward; no rewrite), then one commit `e26657f` "[ROCm] Sort all 64 bits where builders key on the upper half" (builder fix + new test). Pushed plainly `b0ea6a1..e26657f`; `advance-head cuBQL e26657f...` from inside `porting`. `jargon.py --port cuBQL`, `audit-commits cuBQL`, `prose.py` on the commit body: clean.

### The change

`builder_common.h` defines `CUBQL_UPPER_HALF_SORT_BEGIN_BIT` (0 under `__HIPCC__`, 32 otherwise); the four `[32,64)` sorts (`sm_builder.h`, `sah_builder.h`, `elh_builder.h` SortKeys of PrimState words; `rebinMortonBuilder.h` `keyValueSort` of RebinRange words) use it as begin bit with end bit 64. Same as the plan's patch, plus two comment sentences (the exact key range, and why not [32,63)).

### Open question settled: [32,63) is NOT valid; kept [0,64)

[32,63) is exact and stable only if bit 63 of every sorted word is 0. Value ranges from the code (upstream main `6bd4a51`):

- PrimState (`sm_builder.h:12-24`, used by SAH and ELH too): `{primID:31, done:1, nodeID:32}`, sorted word bits 32..63 = nodeID. `initPrims` sets `nodeID = (uint32_t)-1` for every invalid prim (box lower > upper) (`sm_builder.h:87`, `sah_builder.h:143`, `elh_builder.h:163`); `updatePrims` skips `done` prims, so the sentinel reaches the sort, and the struct comment says the layout is chosen so nodeID -1 sorts last. `writePrimsAndLeafOffsets` relies on that (`(int)ps.nodeID < 0` skip). So bit 63 is 1 for every invalid prim. Also, nodes are allocated `2*numPrims` and primID has 31 bits, so for more than 2^30 prims real node IDs reach 2^31 and have bit 63 set.
- RebinRange (`rebinMortonBuilder.h:890-898`): upper half = `beginOfRebinPrimRange`, from `atomicAdd` on the `int` counter `numBroadPhaseRebinPrims` (`:1143`), bounded by numPrims, distinct per range, no sentinel. Bit 63 is always 0 here, so [32,63) would be valid for this one sort only.

Since the prim builders fail the criterion, [32,63) is not valid for every affected sort; one macro with [0,64) everywhere.

Evidence (agent_space/sortbench.hip, hipcub SortKeys on PrimState-shaped words, ~1% sentinel prims, median of 50, MI355X gfx950, ROCm 10.2.0, rocPRIM 4.8.0):

    collision [0,64):  out = node 7fffffff, node ffffffff -> invalid prim last
    collision [32,63): out = node ffffffff, node 7fffffff -> invalid prim NOT last
    n=20000    [0,64) 0.0419 ms   [32,63) 0.0374 ms   [32,64) 0.0387 ms NOT GROUPED
    n=200000   [0,64) 0.0696 ms   [32,63) 0.0655 ms   [32,64) 0.0633 ms NOT GROUPED
    n=1048576  [0,64) 0.1772 ms   [32,63) 0.1686 ms   [32,64) 0.1564 ms NOT GROUPED
    n=4194304  [0,64) 0.3641 ms   [32,63) 0.2024 ms   [32,64) 0.1920 ms (onesweep, correct)

So [0,64) costs 5-12% over [32,63) up to 2^20 keys and 0.16 ms more at 4M keys (8 digit passes instead of 4); small next to a build. The tree checker passes with both forms at the tested sizes (135/135 each; agent_space/cubql-63 is the [32,63) variant), because a node ID of 0x7FFFFFFF needs about 2^30 prims: the checker cannot tell the forms apart, the code reading and the collision check do.

### Test added to cuBQL (step 3)

`tests/builder_leaves.cu` + 9 lines in `tests/CMakeLists.txt`, beside upstream's `morton_sort_range.cu` and in its style (CUDA spelling, `CUBQL_BUILD_TESTS=ON`, `add_test`). It links `cuBQL_cuda_float3` (the precompiled builders), is built under `CUBQL_HAVE_CUDA OR CUBQL_HAVE_HIP` and marked `LANGUAGE HIP` under `CUBQL_USE_HIP` (as the samples are). It is `cubql_check.hip` from the plan, minus the useless tree hash, plus a third scene: uniform with every 97th box inverted, run through the three prim builders only (the radix builders require nonempty boxes), so the `(uint32_t)-1` sentinel goes through the sort and the check asserts no invalid prim is referenced. 3 scenes x 3 sizes (1000, 20000, 200000) x leaf 1/4/8 x builders = 135 checks, 2.0 s on gfx950.

### Build and results (linux-gfx950, MI355X, ROCm 10.2.0 TheRock wheels, HIP_VISIBLE_DEVICES=2, ulimit -c 0)

ROCm 10.2 installed into the job directory (the job venv links the image's 10.1; installing over it would touch the shared env):

    $FLEET_VENV/bin/python -m pip install --target <job>/rocm-10.2 --index-url https://stable.repo.amd.com/rocm/whl-next/ 'rocm[libraries,devel]==10.2.0' rocm-sdk-device-gfx950==10.2.0
    PYTHONPATH=<job>/rocm-10.2 $FLEET_VENV/bin/python -m rocm_sdk init
    R=<job>/rocm-10.2/_rocm_sdk_devel
    cmake -S projects/cuBQL/src -B projects/cuBQL/src/build-hip -G Ninja -DCUBQL_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx950 \
      -DCMAKE_BUILD_TYPE=Release -DCUBQL_BUILD_TESTS=ON -DCMAKE_PREFIX_PATH=$R \
      -DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$R -DCMAKE_BUILD_RPATH=$R/lib
    cmake --build projects/cuBQL/src/build-hip -j 64 --target cuBQL_cuda_float3 cubql_builder_leaves <the 8 GPU samples> cuBQL_cpu_float3 cuBQL_sample01_points_closestPoint_cpu
    ctest --test-dir projects/cuBQL/src/build-hip -R builder_leaves --output-on-failure

- Unpatched builders (upstream main + the new test, build-base): 65 ok, 17 FAILED (uniform 20000/200000 leaf 4/8 and clustered 20000 for spatial median/SAH/ELH, e.g. `uniform n=20000 leaf=4 spatial median FAILED (missing 9111, duplicated 5896)`), then clustered 200000 rebinRadix: illegal memory access in `rebinRadixBuilder_impl::broadPhaseCreateNodes`, `cudaEventSynchronize` code 700. Matches the plan's table.
- Patched (`e26657f`): `cubql_builder_leaves` 135/135 ok, ctest 1/1 passed.
- Samples on the patched build: s01 closest point (binary and wide BVH) exit 0, both identical to the CPU sample's 20 answers; s07 aggregateNBody exit 0; CPU sample exit 0. s02-s06 not run: they need `bunny.obj`, absent on this host (not in the repo; earlier validations brought their own).
- CUDA (nvcc 12.8.93, /usr/local/cuda): `cmake -S projects/cuBQL/src -B .../build-cuda -G Ninja -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release -DCUBQL_BUILD_TESTS=ON`, then `cuBQL_cuda_float3 cubql_builder_leaves cubql_morton_sort_range` build clean. Token stream: `nvcc -std=c++17 -x cu -E -Xcompiler -P -DCUBQL_HAVE_CUDA=1 -DCUBQL_INSTANTIATE_T=float -DCUBQL_INSTANTIATE_D=3 -arch=sm_80 -I. cuBQL/builder/cuda/instantiate_builders.cu` (blank lines dropped) on `6bd4a51` and on `e26657f`: 135484 lines each, identical (the `numPrims,32,64,s` sorts appear 6 times in both).

### State left

Stage `delta-ported` at head `e26657f` (reviewer reviews the delta `b0ea6a1..e26657f`, which is upstream's 19 commits since the merge plus the one fix commit; the port's own delta past upstream main is `6bd4a51..e26657f`). linux-gfx90a, linux-gfx1100, windows-gfx1201 and windows-gfx1101 stay `completed` at `b0ea6a1` and read as revalidate once the review passes. Not done here and needed before a follow-up PR can be offered: review, revalidation on wave64/wave32/windows, and jeff's decision to open it (the checkup's `--review` flow; `published_sha` is recorded, the mirror is synced).

### Draft follow-up PR for NVIDIA/cuBQL (needs jeff's decision; nothing opened)

====
**Title:** HIP: sort all 64 bits where builders key on the upper half

On ROCm, the spatial-median, SAH and ELH GPU builders return trees whose leaves reference the wrong primitives (some primitives in no leaf, others in two) once a leaf holds more than one primitive and there are more than 1024 primitives, and `rebinRadixBuilder` can fault with an illegal memory access on clustered input. The radix builders are not affected.

Cause: these builders sort packed 64-bit words by bits [32,64), the node ID of a `PrimState` or the range begin of a `RebinRange`. rocPRIM's radix sort mis-orders 64-bit keys when `end_bit` is 64 and `begin_bit` is not 0, for 1025 to 2^20 keys (ROCm/rocm-libraries#16299), so the node IDs are not grouped and the leaf offsets point into other leaves.

Change: on HIP only, the four affected sorts use bits [0,64). That keeps the order of the upper halves and only orders equal upper halves by their lower half, which none of the callers depend on. Bits [32,63) would also avoid the defect and sort slightly faster, but they are not correct: invalid prims carry node ID `(uint32_t)-1`, which under [32,63) shares a key with node `0x7FFFFFFF`. The CUDA build is unchanged: the new macro expands to 32 there.

Test: the new `tests/builder_leaves.cu` (built with `-DCUBQL_BUILD_TESTS=ON`, for CUDA or HIP) builds trees with all six binary builders over 1000 to 200000 uniform and clustered boxes with leaf sizes 1, 4 and 8, and with the spatial-median, SAH and ELH builders also over partly empty boxes (the radix builders require nonempty boxes). It checks that every valid primitive is in exactly one reachable leaf, no empty box is referenced, and every box and child lies inside its node. On MI355X (gfx950) with ROCm 10.2.0 it passes 135/135 with the change; without it, 17 trees are wrong before `rebinRadixBuilder` faults. The existing `cubql_morton_sort_range` test now also builds and runs on HIP (four more CUDA runtime names are mapped in `cuBQL/math/common.h`) and passes there, including its `--benchmark` validation. With nvcc 12.8 the library and both tests compile, and the preprocessed `instantiate_builders.cu` and `morton_sort_range.cu` are identical before and after.

This change was prepared with the help of an AI coding agent.
====

`jargon.py -` and `prose.py -` on the body: clean.

## Lesson for cuda-to-rocm: picking a workaround for rocPRIM's bit-range radix sort defect

Belongs in `libraries.md`, beside the warp-1.14 lesson on the defect itself (64-bit keys, end_bit 64, begin_bit > 0, 1025 to 2^20 keys). Two workarounds avoid the broken comparator: sort the full width (`begin_bit = 0`), or stop one bit short (`end_bit = width - 1`). The second is exact and stable but only when the top key bit is 0 for every element, and packed keys often break that: a sentinel such as `(uint32_t)-1` in the upper half ("invalid, sort last") sets it, and under the narrowed range the sentinel then ties with the largest valid value. A unit test at ordinary sizes passes either way, because the colliding valid value is near 2^31. Read where every packed field is written, sentinels included, before narrowing; otherwise sort the full width, which is correct whenever no caller depends on the order of keys equal in the selected bits (a 5-12% cost at up to 2^20 64-bit keys on MI355X). (cuBQL)

## Lesson for moat-porting: a follow-up round on a merged, trunk-resident project has no CLI re-home before the lock

When a port's upstream PR merged and its folder is on `main` with no `port/<name>`, the porter cannot take the lock: `set-state <name> <platform> porting` on `main` commits the lock to the trunk, `fix-branch` refuses without an open PR, `sync-branch` without a maintained ruling, and `advance-head` (which does re-home) belongs inside `porting`. This round called `moatlib.ensure_port_branch(<name>)` from Python, then checked out the branch and took the lock. A `moatlib.py` subcommand for it (or `set-state` re-homing on entering `porting` from `review-passed` with a finished PR) would close the gap. (cuBQL)

## Review 2026-11-19 (reviewer, linux-gfx950): follow-up round, delta b0ea6a1..e26657f

Scope: the port's own commit `e26657f` (`6bd4a51..e26657f`), plus the HIP impact of upstream's 19 commits `b0ea6a1..6bd4a51` that this round brought into `moat-port` by fast-forward. Fact-checked: no other `DeviceRadixSort` call in the tree has a nonzero begin bit (`radix.h:691/707` and `rebinMortonBuilder.h:1100/1113/1229/1244` all begin at 0, so the rocPRIM defect cannot reach them). The `(uint32_t)-1` nodeID sentinel (`sm_builder.h:87`, `sah_builder.h:143`, `elh_builder.h:163`) and its `(int)ps.nodeID < 0` readers are as the porter says, so [0,64) over [32,63) is correct. Reran independently on gfx950 (ROCm 10.2.0, HIP_VISIBLE_DEVICES=2, `ulimit -c 0`): `cubql_builder_leaves` 135/135, PASSED.

### Testing

1. `tests/CMakeLists.txt:4` -- upstream's new `cubql_morton_sort_range` test (it covers upstream's `radix.h` rework and the new `radixBuilder<uint32_t|uint64_t>` entry points that this round merged into the HIP port) is still built only `if (CUBQL_HAVE_CUDA)`, so no HIP build or validator ever runs it. It builds and passes on HIP once four aliases are added to the `__HIPCC__` block of `cuBQL/math/common.h` (next to `using cudaMemPool_t = hipMemPool_t;`, line 39): `using cudaDeviceProp = hipDeviceProp_t;`, `#define cudaGetDeviceProperties hipGetDeviceProperties`, `#define cudaRuntimeGetVersion hipRuntimeGetVersion`, `#define cudaDriverGetVersion hipDriverGetVersion` (without them: `morton_sort_range.cu:514/518` errors on `cudaDeviceProp`, `cudaRuntimeGetVersion`, `cudaDriverGetVersion`). Then gate it `if (CUBQL_HAVE_CUDA OR CUBQL_HAVE_HIP)` with `set_source_files_properties(morton_sort_range.cu PROPERTIES LANGUAGE HIP)` under `CUBQL_USE_HIP`, as `cubql_builder_leaves` is. Checked in a scratch copy (`agent_space/cubql-copy`, those two edits only): default run exit 0, every case `status=PASS`, `query_oracle=PASS`, 0.25 s; `--benchmark` exit 0, `validation=PASS`. Put it in its own `[ROCm]` commit (the added lines name HIP), and add `cubql_morton_sort_range` to the HIP Test Plan and the draft PR body.

### Draft follow-up PR body

2. The Test paragraph says the test "builds trees with all six binary builders over 1000 to 200000 uniform, clustered and partly empty boxes". The partly-empty scene runs only the spatial-median, SAH and ELH builders (`builder_leaves.cu:118`, the radix builders require nonempty boxes). Say so, e.g. "...uniform and clustered boxes, and the spatial-median, SAH and ELH builders also over partly empty boxes...". The commit body of `e26657f` has the same wording; leave the commit as it is (no amend) and fix it in the PR body.

### Recommendation

**Request Changes** (item 1; item 2 is a wording fix to the draft). The builder change itself, the macro, the new test and the commit hygiene need nothing.

## Changes-requested round 2026-11-19 (porter, linux-gfx950): head 1a2b9bb

Both findings of the 2026-11-19 review addressed; `e26657f` left as it is.

1. New commit `1a2b9bb` "[ROCm] Build and run the Morton sort range test on HIP": `cuBQL/math/common.h` HIP block gains `using cudaDeviceProp = hipDeviceProp_t;` and `cudaGetDeviceProperties`/`cudaRuntimeGetVersion`/`cudaDriverGetVersion` -> `hip*` defines (exactly the reviewer's four); `tests/CMakeLists.txt` now builds `cubql_morton_sort_range` under `CUBQL_HAVE_CUDA OR CUBQL_HAVE_HIP` with `LANGUAGE HIP` under `CUBQL_USE_HIP`, and both tests share that one block (the separate `builder_leaves` block from `e26657f` was folded into it).
2. Draft PR body (the `====` block in the follow-up round section above) corrected in place: the partly empty scene is named as running the spatial-median, SAH and ELH builders only, and the Test paragraph now also names `cubql_morton_sort_range` on HIP and the identical preprocessed `morton_sort_range.cu`. `jargon.py -` and `prose.py -` on the body: clean. The commit body of `e26657f` keeps the old wording (no amend).

Results, gfx950 MI355X, ROCm 10.2.0 wheels, HIP_VISIBLE_DEVICES=2, `ulimit -c 0`, build recipe as in the follow-up round section:

    cmake --build projects/cuBQL/src/build-hip -j 64 --target cubql_morton_sort_range cubql_builder_leaves cuBQL_cuda_float3
    ctest --test-dir projects/cuBQL/src/build-hip --output-on-failure      -> 2/2 passed (builder_leaves 2.07 s)
    ./cubql_morton_sort_range              -> exit 0, 27 x status=PASS, 0 FAIL, 2 x query_oracle=PASS
    ./cubql_morton_sort_range --benchmark  -> exit 0, validation=PASS (benchmark_device=AMD Instinct MI355X)

CUDA, nvcc 12.8.93: `cuBQL_cuda_float3 cubql_builder_leaves cubql_morton_sort_range` build clean in build-cuda (sm_80). Preprocessed with `nvcc -std=c++17 -x cu -E -Xcompiler -P -DCUBQL_HAVE_CUDA=1 -DCUBQL_INSTANTIATE_T=float -DCUBQL_INSTANTIATE_D=3 -arch=sm_80 -I. -Itests`, blank lines dropped, upstream `6bd4a51` vs `1a2b9bb`: `instantiate_builders.cu` 135484 lines identical, `tests/morton_sort_range.cu` 139239 lines identical.

`jargon.py --port cuBQL`, `audit-commits cuBQL`, `prose.py` on the commit body: clean. Pushed `e26657f..1a2b9bb`, `advance-head cuBQL 1a2b9bb...`, handed back as `delta-ported` (review scope for this round: `e26657f..1a2b9bb` plus the notes draft).

## Review 2026-11-19 (reviewer, linux-gfx950): changes-requested round, head 1a2b9bb

Scope: the delta past what PR #35 shipped (`b0ea6a1..1a2b9bb`); the port's own part is `6bd4a51..1a2b9bb` (`e26657f` + `1a2b9bb`), with this round's new commit `1a2b9bb` and the corrected draft PR body read in full. Upstream's 19 commits `b0ea6a1..6bd4a51` were covered by the previous review.

No findings. Checked:

- Both earlier findings are addressed: `1a2b9bb` adds exactly the four aliases inside the `__HIPCC__` block of `cuBQL/math/common.h` (CUDA preprocessing unaffected) and builds `cubql_morton_sort_range` under `CUBQL_HAVE_CUDA OR CUBQL_HAVE_HIP` with `LANGUAGE HIP`; the `CUBQL_HAVE_CUDA=1` define under HIP matches how `cuBQL/CMakeLists.txt` defines it for the HIP-compiled instantiation targets. The draft body now scopes the partly empty scene to the three prim builders and names the morton test on HIP.
- Sort coverage: every `DeviceRadixSort` call in `cuBQL/builder/cuda/` either uses `CUBQL_UPPER_HALF_SORT_BEGIN_BIT` (sm/sah/elh `SortKeys`, `rebinMortonBuilder.h:1072/1083`) or begin bit 0 (`radix.h:697/713` `0,SortEndBit`; `rebinMortonBuilder.h:1106/1119` `0,64`; `:1235/1250` `0,32`), so no remaining call hits the rocPRIM defect's begin>0/end=64 shape.
- `tests/morton_sort_range.cu` uses no warp intrinsics or warp-width constants; its only HIP-relevant names are the four now mapped (`:514-518`).
- Commit hygiene: both titles `[ROCm]`, under 72 chars, AI disclosure in prose, no trailer; `audit-commits cuBQL` OK, `jargon.py --port cuBQL` clean.
- Lessons: the cuda-to-rocm lesson's 5-12% figure matches the recorded bench (0.0419/0.0374, 0.0696/0.0655, 0.1772/0.1686) and its narrowing criterion is correct; the moat-porting lesson still holds on trunk (`ensure_port_branch` exists in `moatlib.py` but no subcommand exposes it).

Independent run, gfx950 MI355X, ROCm 10.1.0 (image wheels, not the porter's 10.2.0), HIP_VISIBLE_DEVICES=2, `ulimit -c 0`:

    cmake -S projects/cuBQL/src -B agent_space/cubql-build -G Ninja -DCUBQL_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx950 -DCMAKE_BUILD_TYPE=Release -DCUBQL_BUILD_TESTS=ON -DCMAKE_PREFIX_PATH=$R -DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$R -DCMAKE_BUILD_RPATH=$R/lib
    cmake --build agent_space/cubql-build -j 32 --target cubql_morton_sort_range cubql_builder_leaves
    ctest --test-dir agent_space/cubql-build --output-on-failure   -> 2/2 passed (builder_leaves 2.51 s)
    cubql_morton_sort_range               -> exit 0, 27 status=PASS, 0 FAIL, 2 query_oracle=PASS
    cubql_morton_sort_range --benchmark   -> exit 0, validation=PASS

CUDA, nvcc at /usr/local/cuda, `-DCMAKE_CUDA_ARCHITECTURES=80 -DCUBQL_BUILD_TESTS=ON`: `cuBQL_cuda_float3 cubql_builder_leaves cubql_morton_sort_range` build clean at `1a2b9bb`.

Recommendation: **Approve**. Next: revalidation on wave64, wave32 and windows at `1a2b9bb` (s02-s06 need a `bunny.obj`), then jeff's decision on opening the follow-up PR from the draft above.

## Validation 2026-11-19 (linux-gfx950)

Validated `1a2b9bb` (moat-port; port delta past upstream main `6bd4a51` is `e26657f` + `1a2b9bb`) on linux-gfx950, AMD Instinct MI355X, one GPU pinned with HIP_VISIBLE_DEVICES, `ulimit -c 0`. Toolchain: TheRock ROCm 10.2.0 wheels installed into the job directory (not the system or image ROCm), AMD clang 24.0.0git, rocPRIM 4.8.0 (`ROCPRIM_VERSION 400800`); `ldd` confirms the binaries load `libamdhip64.so.7` from the 10.2 tree.

    python -m pip install --target <job>/rocm-10.2 --index-url https://stable.repo.amd.com/rocm/whl-next/ 'rocm[libraries,devel]==10.2.0' rocm-sdk-device-gfx950==10.2.0
    PYTHONPATH=<job>/rocm-10.2 python -m rocm_sdk init
    R=<job>/rocm-10.2/_rocm_sdk_devel
    cmake -S projects/cuBQL/src -B projects/cuBQL/src/build-hip -G Ninja -DCUBQL_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx950 \
      -DCMAKE_BUILD_TYPE=Release -DCUBQL_BUILD_TESTS=ON -DCMAKE_PREFIX_PATH=$R \
      -DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$R -DCMAKE_BUILD_RPATH=$R/lib
    cmake --build projects/cuBQL/src/build-hip -j 64 --target cuBQL_cuda_float3 cubql_builder_leaves cubql_morton_sort_range \
      cuBQL_sample01_points_closestPoint_cuda cuBQL_sample01_points_closestPoint_wideBVH_cuda sample02_distanceToTriangleMesh \
      sample03_insideOutside sample04_boxOverlapsOrInsideSurfaceMesh sample05_lineOfSight sample06_anyTriangleWithinRadius \
      sample07_aggregateNBody cuBQL_cpu_float3 cuBQL_sample01_points_closestPoint_cpu
    ctest --test-dir projects/cuBQL/src/build-hip --output-on-failure

Build: 30/30 steps, clean (host-side warnings only).

Tests:

- ctest: 2/2 passed (`cubql_morton_sort_range` 0.52 s, `cubql_builder_leaves` 2.93 s).
- `cubql_builder_leaves` run directly: 135 ok, 0 FAILED, `PASSED` (missing/duplicated/invalid/containment all 0, including the with-empty scene).
- `cubql_morton_sort_range`: exit 0, 27 `status=PASS`, 0 FAIL, 2 `query_oracle=PASS`; `--benchmark`: exit 0, `validation=PASS`, `benchmark_device=AMD Instinct MI355X`.

Samples (all exit 0), with the 4968-triangle Stanford bunny (`https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj`, the same mesh the 2026-06 validations used):

- s01 closest point, binary and wide BVH: output identical to the CPU sample (all 20 queries).
- s02 distanceToTriangleMesh: 4968 triangles, closest-triangle answers printed.
- s03 insideOutside -n 32: 32x32x25 volume, values {0,1}, 6715 inside.
- s04 boxOverlapsOrInsideSurfaceMesh -n 32: values {0,1,2}, 8466 nonzero.
- s05 lineOfSight -n 64 64: image written.
- s06 anyTriangleWithinRadius -n 32: values {0,1}, 10885 nonzero.
- s07 aggregateNBody: 100000 data points.
- CPU sample and `cuBQL_cpu_float3`: build and run clean.

CUDA gate: not rerun; it is already recorded at `1a2b9bb` (nvcc 12.8.93, sm_80, `cuBQL_cuda_float3 cubql_builder_leaves cubql_morton_sort_range` clean, preprocessed `instantiate_builders.cu` and `morton_sort_range.cu` identical to upstream `6bd4a51`; changes-requested round and its review above).

Content checks: `python3 utils/jargon.py --port cuBQL` clean (the fresh fork clone needs local `main` and `moat-port` branches for it to resolve the range). README documents the ROCm build next to the CUDA build (`README.md:319`). Fork tree clean apart from untracked `build-hip/`.
## Validation 2026-11-19 (linux-gfx1100): follow-up round, head 1a2b9bb

Revalidated `1a2b9bb` (from `b0ea6a1`) on linux-gfx1100, AMD Radeon Pro W7800 48GB (gfx1100, wave32), ROCm 10.2.0 TheRock wheels installed into the job directory (not a system ROCm; `libamdhip64.so.7` resolves to the wheel tree, HIP 7.17.26434, clang 24.0.0), HIP_VISIBLE_DEVICES=3, `ulimit -c 0`. `classify b0ea6a1 1a2b9bb` = mixed, not inert (builder sort change, new tests, upstream's 19 commits), so a full GPU run, no carry-forward.

    pip install --target <job>/rocm-10.2 --index-url https://stable.repo.amd.com/rocm/whl-next/ 'rocm[libraries,devel]==10.2.0' rocm-sdk-device-gfx1100==10.2.0
    PYTHONPATH=<job>/rocm-10.2 python -m rocm_sdk init
    R=<job>/rocm-10.2/_rocm_sdk_devel
    cmake -S projects/cuBQL/src -B projects/cuBQL/src/build-hip -G Ninja -DCUBQL_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      -DCMAKE_BUILD_TYPE=Release -DCUBQL_BUILD_TESTS=ON -DCMAKE_PREFIX_PATH=$R \
      -DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$R -DCMAKE_BUILD_RPATH=$R/lib
    cmake --build projects/cuBQL/src/build-hip -j 32 --target cuBQL_cuda_float3 cubql_builder_leaves cubql_morton_sort_range <the 8 GPU samples> cuBQL_cpu_float3 cuBQL_sample01_points_closestPoint_cpu
    ctest --test-dir projects/cuBQL/src/build-hip --output-on-failure

- Build: 30/30 steps, 0 errors (warnings only).
- ctest: 2/2 passed -- `cubql_morton_sort_range` 0.45 s, `cubql_builder_leaves` 1.58 s.
- `cubql_builder_leaves` run directly: 135/135 ok, 0 FAILED, `PASSED` (includes the `with-empty` sentinel scenes and the 20000/200000 sizes that failed unpatched on gfx950).
- `cubql_morton_sort_range`: exit 0, 27 status=PASS, 0 FAIL, 2 query_oracle=PASS; `--benchmark` exit 0, validation=PASS, benchmark_device=AMD Radeon Pro W7800 48GB.
- Samples: s01 binary BVH, s01 wide BVH and the CPU s01 exit 0 with identical 20 answers (md5 of the result lines equal); s07 aggregateNBody exit 0. No `bunny.obj` on this host, so s02-s06 ran on a generated UV sphere (radius 1, 5184 triangles, `agent_space/sphere.py`, not committed): all exit 0; s03 inside fraction 0.5247 vs pi/6 = 0.5236; s02 distance from (0.84,0.84,0.84) 0.454957 vs 0.84*sqrt(3)-1 = 0.4549.
- CUDA gate: not rerun; recorded at `1a2b9bb` by the porter (nvcc 12.8.93) and the reviewer.
- `jargon.py --port cuBQL` clean; README documents the HIP build (`README.md:319`); fork tree clean apart from untracked `build-hip/`.
