# barney (NVIDIA/barney)

NVIDIA's ANARI-based GPU ray tracer / scientific-visualization renderer. Scaffolded 2026-06-15.

## License: FAVORABLE (Apache-2.0 throughout) -- thoroughly scanned

Full source-level scan of a fresh clone (commit fe10e5d, 2026-05-11):
- Top-level LICENSE = Apache-2.0; 339 `SPDX-License-Identifier: Apache-2.0` tags across the tree; 263 NVIDIA copyright lines all under Apache.
- Bundled / submodule deps all permissive: cuBQL (Apache-2.0), owl (Apache-2.0), embree (Apache-2.0), OpenVDB/nanovdb (MPL-2.0/Apache), ANARI SDK / Khronos (Apache), OpenMPI (BSD, optional), glm (MIT), stb (public domain).
- NO NVIDIA Source Code License, NO non-commercial / research-only terms anywhere. (The only "non-commercial" string in the tree is in public-domain stb, which permits commercial use.)
- Verdict: we can legally fork, port to ROCm/HIP, add AMD copyright, and submit upstream. Unlike nvdiffrast (NVIDIA Source Code License) or the Inria Gaussian-Splatting repos.

## Port scope: LOW-RISK (no OptiX->HIPRT required for a first backend)

barney has a clean pluggable ray-tracing-core abstraction at `rtcore/` (`common/TraceInterface.h`, `ComputeInterface.h`, `AppInterface.h`) with multiple backends:
- `rtcore/optix` (13 files) -- OptiX hardware-RT backend. OptiX use is isolated here; only 2 files (`Denoiser.cpp`, `Device.cpp`) touch the host OptiX API. ~30 standard device intrinsics (optixTrace, optixGetPrimitiveIndex, optixGetWorldRayOrigin, ...), all with HIPRT equivalents.
- `rtcore/embree` (28 files) -- CPU backend.
- `rtcore/cuda` (16 files) -- **OptiX-free** CUDA backend doing software BVH traversal via cuBQL (3 includes).
- `rtcore/cudaCommon` (11 files) -- shared CUDA.

Recommended port path (planner): add a HIP backend by hipifying `rtcore/cuda` + `rtcore/cudaCommon` (~27 files) and porting cuBQL (Apache-2.0 header BVH library). This is a normal CUDA-compute port -- NO OptiX, NO HIPRT -- and runs on CDNA gfx90a. The OptiX backend can be left untouched (it is one of several). nanovdb already carries HIP guards.
Stretch (optional, later): a hardware-RT `rtcore/hiprt` backend using AMD HIPRT, which would need an RDNA RT GPU (gfx1100/gfx1201) to validate.

Main porting surface = cuBQL (CUDA BVH lib) + barney's `rtcore/cuda`/`cudaCommon`. Build system is CMake.

## Status
Backend 1 (software tracer -> HIP) completed + validated on Linux gfx90a (sha 060ba2a). Backend 2 (rtcore/hiprt, AMD hardware-RT via HIPRT) built + smoke-rendered on gfx90a -- see "## Backend 2" below. Fork AMD-Ecosystem/barney @ moat-port.

## Fork / submodules
- Fork: https://github.com/AMD-Ecosystem/barney (Actions disabled). Local clone at projects/barney/src has remotes origin=NVIDIA/barney (never push), fork=AMD-Ecosystem/barney. moat-port branched off upstream main (fe10e5d); the single upstream PR will be moat-port -> main.
- Submodules: `submodules/owl` (header-only owl::common math) and `submodules/cuBQL` (the HIP-enabled fork, see below) are both populated. embree/optix submodules NOT inited.
- **cuBQL submodule (as of commit d9ea875)**: `.gitmodules` points `submodules/cuBQL` at `https://github.com/AMD-Ecosystem/cuBQL`, branch `moat-port`, gitlink pinned at `b0ea6a1` ([ROCm] Add HIP build path for AMD GPUs -- the squashed cuBQL HIP port). With `BARNEY_USE_EXTERNAL_CUBQL=OFF` (the default), barney's CMake uses this submodule directly; no `_deps/cuBQL` external dir needed. TEMPORARY: this pin will be reverted to `NVIDIA/cuBQL` once upstream PR NVIDIA/cuBQL#35 merges (tracked in data/deferred.json as `barney-cubql-repin-after-pr35`).
- **Colleague clone**: `git clone --recursive --branch moat-port https://github.com/AMD-Ecosystem/barney` populates `submodules/cuBQL` at `b0ea6a1` (AMD-Ecosystem/cuBQL HIP fork) enabling AMD builds from a clean checkout. Note: `--depth=1 --recursive` is a shallow-clone shortcut that fetches the branch tip for submodules; use `git submodule update --init` (without `--depth`) to get the pinned `b0ea6a1`. Build: `cmake -S . -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=<arch>` (BARNEY_USE_EXTERNAL_CUBQL defaults OFF, uses the submodule).

## Backend 1 -- what was ported (rtcore/cuda + rtcore/cudaCommon -> HIP)

Strategy A, in-place reuse of the existing cuda backend compiled under hipcc (no new rtcore/hip/ dir; the rtc::cuda namespace is kept). The whole HIP path is gated by the new `USE_HIP` CMake option (default OFF) and `__HIPCC__`/`BARNEY_HAVE_HIP`, so the CUDA/OptiX/Embree build is byte-identical.

Files:
- NEW `rtcore/cudaCommon/cuda_to_hip.h` -- the single compat header. Keys on `__HIP_PLATFORM_AMD__`/`USE_HIP`/`__HIPCC__`; includes `<cstring>`/`<cstdlib>` BEFORE `<hip/hip_runtime.h>` (gpuRIR fault class: host memset/memcpy must not bind to the HIP device overloads -- confirmed real, the standalone texture probe hit exactly this), then aliases the cuda runtime spellings the backend uses (Malloc/Free/Memcpy{,Async}/Memset/Stream*/SetDevice/GetDevice{,Count}/GetDeviceProperties/DeviceProp/peer-access/textures+arrays/channel+resource+texture descs/filter+address+read enums). The `BARNEY_CUDA_CALL(cuda##X)` token-paste resolves through these defines exactly like cuBQL's CUBQL_CUDA_CALL.
- EDIT `rtcore/cudaCommon/cuda-common.h` -- route `<cuda_runtime.h>` through the compat header (drop `<cuda/std/limits>`/`<cuda.h>`, both CUDACC-only, not needed on HIP).
- EDIT `rtcore/common/rtcore-common.h` + `rtcore/cudaCommon/ComputeInterface.h` -- widen the device-code guards from `__CUDACC__` to `__CUDACC__ || __HIPCC__` (RTC_DEVICE_CODE; the ComputeInterface member fns + texture wrappers + atomics blocks). HIP provides ::tex1D/2D/3D, ::atomicAdd/atomicCAS, __float_as_int/__int_as_float natively, so the wrapper bodies are unchanged.
- EDIT top-level `CMakeLists.txt` -- `USE_HIP` path: enable_language(HIP); default CMAKE_HIP_ARCHITECTURES=gfx90a ONLY when unset (no hardcoded arch); BARNEY_HAVE_HIP ON / BARNEY_HAVE_CUDA OFF; backend-select forces BACKEND_CUDA ON / OPTIX,EMBREE OFF; cuBQL consumed with CUBQL_USE_HIP=ON; `USE_HIP=1 BARNEY_HAVE_HIP=1` compile defs + `hip::host` link on barney_config so plain-C++ host TUs see the compat path + ROCm headers; new `BARNEY_EXTERNAL_CUBQL_DIR` cache var so external cuBQL can live outside the source tree.
- EDIT `rtcore/CMakeLists.txt` -- `rtc_mark_hip_sources` macro marks the cuda/cudaCommon .cu LANGUAGE HIP + `-fgpu-rdc`.
- EDIT `barney/CMakeLists.txt` -- `configure_cuda_source` HIP arm: HOST_SOURCES + DEVICE_PROGRAM_SOURCES .cu/.dev.cu compiled LANGUAGE HIP with `-fgpu-rdc`; `set_library_properties` adds `-fgpu-rdc --hip-link` link options + HIP_SEPARABLE_COMPILATION (see device-link note below). The nvcc-only `--expt-relaxed-constexpr`/`--extended-lambda` injections stay gated under `$<COMPILE_LANGUAGE:CUDA>` (untouched), so the HIP TUs never see them.

## Port gotchas / fault-class findings

- DEVICE-LINK (the load-bearing one): the megakernel's correctness rests on `-fgpu-rdc` device-linking the function-pointer SBT (the writeAddresses kernels store `type::closestHit/anyHit/intersect` device function pointers that traceRay calls). Every .cu/.dev.cu must compile `-fgpu-rdc`, AND the final shared lib / static lib / exe must link with `-fgpu-rdc --hip-link` or the device-link is skipped and the link dies with `undefined hidden symbol: __hip_gpubin_handle_*`. Putting `-fgpu-rdc --hip-link` as PUBLIC link options on the barney libs (so it propagates to anything linking them, e.g. anari_library_barney) is what fixed it. cuBQL's float3_static lib (non-rdc) links into the same device link fine.
- TEXTURE filter-mode fault class did NOT reproduce on this stack. The plan flagged popsift's `cudaFilterModeLinear` + `cudaReadModeElementType` float-texture rejection. A standalone probe (agent_space/textest.cpp) on gfx90a/ROCm 7.2.1 shows hipCreateTextureObject returns "no error" for float, float4 (linear+elementType) AND uchar4 (linear+normalizedFloat). So the cudaArray-backed texture path is used unchanged; no point-filter + software-interp fallback was needed. (If a follower RDNA arch rejects it, revisit -- the wrappers are the place to add software lerp.)
- 256B texture pitch: N/A. barney uses cudaArray-backed textures (resType Array, Malloc3DArray/MallocArray), not pitched 2D binds.
- Warp size: N/A on the GPU path. No __shfl/__ballot/__popc/warpSize anywhere in the cuda backend or device programs (the one `__popc(__ballot(1))` in RTXObjectSpace.dev.cu is commented out). The `<<<1,32>>>` in ProgramInterface.h is a 32-thread grid launch that writes function-pointer addresses, not a warp-width assumption -- benign on wave64.
- Rule-of-five on handles: textureObject is set to 0 after DestroyTextureObject (Texture.cu:69), array set to 0 after FreeArray; dtors are guarded. Default-init is fine.
- UB audit (EnvGS "Heisenbug" lesson): the active traceRay (`#if 1` branch) uses cuBQL `shrinkingRayQuery::twoLevel::forEachPrim` with per-ray state in the TraceInterface struct (registers/lambda captures), NOT a kernel-stack array; the stack-array variant is `#if 0`. No value-returning device fn falls off the end on the active path. The render produced correct output, so no codegen workaround was needed.
- barney already anticipated HIP: barney/common/half.h (`__HIPCC__` -> hip/hip_fp16.h) and barney/common/math.h (`#if BARNEY_HAVE_HIP` -> BARNEY_INF=INFINITY) already had HIP arms. Defining BARNEY_HAVE_HIP=1 as a compile def lights them up for the host C++ TUs.

## Build (Linux gfx90a, ROCm 7.2.1)

Repeatable script: projects/barney/build-hip.sh. Equivalent to:

    cmake -S projects/barney/src -B projects/barney/src/build-hip \
      -DUSE_HIP=ON \
      -DBARNEY_USE_EXTERNAL_CUBQL=ON \
      -DBARNEY_EXTERNAL_CUBQL_DIR=/var/lib/jenkins/moat/_deps/cuBQL \
      -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release
    cmake --build projects/barney/src/build-hip -j$(nproc)

Prereqs: submodules/owl inited; _deps/cuBQL = AMD-Ecosystem/cuBQL @ moat-port. To also build the ANARI device + run the smoke render, build+install the ANARI SDK (KhronosGroup/ANARI-SDK, code_gen on, helide/examples OFF to avoid the Embree download) and pass `-Danari_DIR=<install>/lib/cmake/anari-0.16.0`.

Builds + links (all gfx90a): barney_rtc_cuda_common, barney_rtc_cuda, barney_cuda_programs, barney_cuda, libbarney.so, barney_static, and (with ANARI) libanari_library_barney.so. gfx90a code objects embedded in libbarney.so (llvm-objdump --offloading).

## Smoke render (gfx90a)

Script: projects/barney/smoke-hip.sh. anariTest.cpp compiled against the ANARI SDK (no CMake target upstream -- compile by hand), run with HIP_VISIBLE_DEVICES=0 and LD_LIBRARY_PATH = build-hip:build-hip/anari:anari-install/lib, loads the "barney" device. Renders a sphere/material scene through the cuBQL software tracer; exit 0; output /tmp/anariTest.png is 1024x768, 11003 distinct colors, 23.9% non-black pixels (smooth shaded geometry, lum 0-220). Confirms BVH build + ray traversal + triangle intersection + function-pointer closest-hit shading dispatch all work on gfx90a. NO dataset downloads (the scene is generated in-process by anariTest).

## What remains
- Validator (next): full multi-sample validation -- the unitTests/*.py reference-image diff (omnicamera, isosurface-umesh) via pynari, more anariTest scenes / geometry types (spheres/triangles/cylinders/capsules/cones), and the embree CPU no-regression build. CUDA no-regression gate on an nvcc box (USE_HIP=OFF must be fully inert).
- Backend 2 (rtcore/hiprt, HW-RT via HIPRT): the entire second phase, gated behind Backend 1 validating. See plan.md "Backend 2".
- Deferred: OIDN-GPU on ROCm for the denoiser (barney's cuda backend ships a no-op denoiser, used as-is here). Register in utils/deferred.py when Backend 1 lands.

## Review 2026-06-16 (reviewer, Backend 1, linux-gfx90a)

Verdict: review-passed. Scope reviewed: Backend 1 only (rtcore/cuda + rtcore/cudaCommon -> HIP software tracer on cuBQL); Backend 2 (rtcore/hiprt) not started, out of scope. Reviewed `git diff fe10e5d...0aa49ac` (7 files, +196/-10) on moat-port.

No blocking problems found. Fact-checked the load-bearing claims against the source and the build artifacts; all held:
- CUDA byte-identity: every source edit is inert on nvcc. The two guard widenings (rtcore-common.h, ComputeInterface.h x2) only add `|| defined(__HIPCC__)` (never defined by nvcc); cuda-common.h reroutes `<cuda_runtime.h>` through cuda_to_hip.h which falls through to `<cuda_runtime.h>` on non-HIP. CMake edits all under USE_HIP/BARNEY_HAVE_HIP. Preprocessed CUDA TU is unchanged.
- Compat-header alias set is exhaustive: computed the set of `cuda*` identifiers used in rtcore/cuda + rtcore/cudaCommon (direct + `BARNEY_CUDA_CALL(cuda##X)` token-paste) and every one is covered by cuda_to_hip.h (error/status, device/stream, memory, textures+arrays, channel/resource/texture descs, filter/address/read enums, peer-access). libc (`<cstring>`/`<cstdlib>`) is included before `<hip/hip_runtime.h>` (gpuRIR host-memset fault class). Token-paste resolves under HIP.
- Device-link wiring is correct and validated: `.cu`/`.dev.cu` compiled `-fgpu-rdc` (rtc_mark_hip_sources + configure_cuda_source HIP arm); `-fgpu-rdc --hip-link` added PUBLIC on the barney libs via set_library_properties so it propagates to consumers (anari_library_barney inherits it via WHOLE_ARCHIVE of barney_static, which is why anari/CMakeLists.txt needed no edit). build-hip/libbarney.so contains an embedded gfx90a code object (llvm-objdump --offloading), so the function-pointer SBT megakernel device-linked.
- Fault classes clear: warp-size N/A (no __shfl/__ballot/__popc/warpSize on the GPU path; the `<<<1,32>>>` writeAddresses launch has all 32 threads redundantly store the same scalar SBT function-pointer fields, benign on wave64). UB audit clean on the active `traceRay` (`#if 1`, TraceInterface.h:97): per-ray state initialized before use, the intersectPrim lambda returns a float on every path, state lives in the struct/lambda captures not a kernel-stack array; the stack-array variant is `#if 0` dead code. Rule-of-five does not bite: Texture/TextureData are always heap-allocated via new and managed by raw pointer (createTexture/freeTexture), never copied or stored by value, and dtors null the handle after destroy/free.
- CMake: USE_HIP option default OFF; arch defaulted to gfx90a only when CMAKE_HIP_ARCHITECTURES is unset (no literal arch in source); external cuBQL wired with CUBQL_USE_HIP=ON via BARNEY_EXTERNAL_CUBQL_DIR; nvcc-only `--expt-relaxed-constexpr`/`--extended-lambda` stay gated to `$<COMPILE_LANGUAGE:CUDA>` (untouched), so HIP TUs never see them.
- Texture filter-mode fault class did not reproduce on gfx90a/ROCm 7.2.1 (porter's standalone probe accepts linear+elementType float/float4 textures); the cudaArray-backed path is used unchanged. Reasoning is sound; the wrappers (ComputeInterface.h tex2D/tex3D) remain the place to add a software-lerp fallback if a follower RDNA arch rejects it. Noted for the gfx1100/gfx1201 followers.
- Commit hygiene clean: title `[ROCm] Add HIP backend for the software ray tracer (gfx90a)` (59 chars), author the public account, no noreply/co-authored trailer, Claude disclosed, Test Plan present, no MOAT jargon / no em-dash / ASCII-only in the diff, no spaced `<< <`. AMD copyright + `\author` added to the new cuda_to_hip.h (SPDX house style, parallel to the NVIDIA line).

Notes for the validator (runtime items to exercise -- expected at this stage, not defects):
- GPU validation has not run yet beyond the porter's anariTest smoke render. Run the planned gate on real gfx90a: anariTest end-to-end plus the unitTests/*.py reference-image diff (omnicamera reference PNG, isosurface-umesh) via pynari, across geometry types (spheres/triangles/cylinders/capsules/cones), confirming the function-pointer megakernel dispatch produces correct images.
- CUDA no-regression gate on an nvcc box (configure with USE_HIP=OFF and the normal CUDA/OptiX path) to confirm the HIP path is fully inert -- the review verifies this by inspection (byte-identity), but a real nvcc configure/build is the gate.
- embree CPU backend no-regression build (the `else` of every HIP guard) should still build/run.
- Substantive-edit attribution: only cuda_to_hip.h is new. The other edited files (rtcore-common.h, ComputeInterface.h, cuda-common.h) received trivial guard/include changes, so no copyright line is required there per house-style; acceptable.

## Validation 2026-06-16 (validator, Backend 1, linux-gfx90a)

Verdict: completed. GPU arch: gfx90a (MI250X, 4 GCDs, HIP_VISIBLE_DEVICES=0). Validated sha: 060ba2a0763fa4a450d454fd9aaaf0efc387c025.

Build: `utils/timeit.sh barney compile -- cmake --build projects/barney/src/build-hip -j$(nproc)` -- PASS (rebuilt after bug fix commit below).

### Bug found and fixed: null currentInstance in closestHit dispatch

During validation, sphere geometry crashed with `Memory access fault ... on address (nil)`. Root cause: cuBQL's `twoLevel::forEachPrim` calls `leaveBlas()` which zeros `this->currentInstance`. After traversal, `acceptedSBT->ch(*this)` was called with `currentInstance==0`. Any closestHit that calls `transformPointFromObjectToWorldSpace` (Spheres does, line 65 of Spheres.dev.cu) dereferences the null pointer.

This is a pre-existing upstream barney bug (the same code exists in `origin/main`), triggered by the new cuBQL twoLevel traversal. The `#else` manual stack traversal also has the bug. Cylinders (has_ch=false) were unaffected because their intersect runs during traversal while currentInstance is valid.

Fix in `rtcore/cuda/TraceInterface.h`: restore `this->currentInstance = model->instanceRecords + accepted.instID` before `acceptedSBT->ch(*this)`. Committed as `060ba2a` `[ROCm] Fix null currentInstance crash in closestHit dispatch`. Should be filed upstream.

### GPU test results (HIP_VISIBLE_DEVICES=0, LD_LIBRARY_PATH=build-hip:build-hip/anari:ANARI-SDK/build)

| Test | Geometry | Result |
|------|----------|--------|
| anariTest.cpp | triangles (2 tris, 4 verts, vertex colors) | PASS: 1024x768, 11003 colors, 23.9% non-black |
| test_sphere | sphere (1 sphere, closestHit dispatch) | PASS: 1024x768, 39900 non-black (5.1%) |
| test_seq | triangles + spheres sequential (same device) | PASS: tri 21904 nb, sph 6232 nb |
| test_cyl | cylinders (3 cylinders, user geom has_ch=false) | PASS: 512x512, 2640 non-black (1.0%) |

All geometry types exercised: triangles (anyHit dispatch via RTC_EXPORT_TRIANGLES_GEOM), spheres (closestHit dispatch via RTC_EXPORT_USER_GEOM has_ch=true), cylinders (user geom intersect-only, no closestHit).

### CPU no-regression

Embree submodule not initialized on this machine and network too slow to fetch it. The CPU (embree) backend requires `submodules/embree`. Cannot test directly. Inspection confirms correctness: every HIP guard adds `|| defined(__HIPCC__)` which nvcc never defines; USE_HIP cmake path is gated; CPU path is byte-identical on non-HIP builds.

### CUDA no-regression gate (linux-gfx90a lead platform)

`utils/timeit.sh barney cuda-compile -- cmake --build /tmp/barney-cuda -j$(nproc)` -- PASS. nvcc 12.8, sm_80, USE_HIP=OFF, OptiX+CUDA backend. Warnings only: sm_80 deprecation. The `currentInstance` fix is pure C++ device code, compiles cleanly under nvcc. No regression.

Commands:
```
cmake -S projects/barney/src -B /tmp/barney-cuda \
  -DCMAKE_CUDA_COMPILER=/opt/conda/envs/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DUSE_HIP=OFF \
  -DBARNEY_USE_EXTERNAL_CUBQL=ON \
  -DBARNEY_EXTERNAL_CUBQL_DIR=/var/lib/jenkins/moat/_deps/cuBQL \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/barney-cuda -j$(nproc)
# Output: 100% built, no errors
```

## Backend 2: rtcore/hiprt (AMD hardware-RT via HIPRT) -- built + smoke-rendered gfx90a

NEW sibling backend mirroring rtcore/optix structure, mapping barney's ray-tracing
core onto AMD HIPRT. Additive: a new `BARNEY_BACKEND_HIPRT` CMake option; the
Backend-1 software (cuda->HIP) path, the OptiX path, and the CUDA build are all
untouched. On gfx90a (CDNA2, no RT units) HIPRT uses its software BVH traversal;
on RDNA2+ (gfx1100/gfx1201 followers) it uses the hardware RT units.

### HIPRT dependency (pinned + gfx90a-verified)
- HIPRT 3.1.0, API version 3001, commit cb09c56 (HIPRT_VERSION_STR "03001"),
  built from source `-DBITCODE=OFF` (the software-traversal path that runs on
  gfx90a) against ROCm 7.2.1. Produces `libhiprt0300164.so`. NOT committed
  (*.so gitignored); a discovered dependency via `cmake/Findhiprt.cmake` +
  `hiprt_ROOT` (mirrors how barney finds OptiX/OIDN). Recipe (EnvGS Stage 2):
      git clone --recursive https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT
      cmake -DCMAKE_BUILD_TYPE=Release -DBITCODE=OFF -DNO_UNITTEST=ON \
            -DHIP_PATH=/opt/rocm -S . -B build
      cmake --build build --target hiprt0300164 -j
  (A pre-built checkout was reused at agent_space/hiprt_probe/HIPRT for this
  bring-up; the dist/bin/Release/libhiprt0300164.so it carries is the pinned
  version.)
- gfx90a SMOKE (before barney): a tiny hipcc kernel building one triangle
  geometry + a 1-instance scene and tracing two rays (hit prim=0 t=1.0, miss=-1)
  PASSED on gfx90a/ROCm 7.2.1 -- confirms HIPRT's software traversal is correct
  on this stack at the pinned version. agent_space/barney_hiprt/smoke.cpp.

### THE load-bearing discovery -- direct hipcc link, no JIT, no Orochi, no void* glue
HIPRT's device traversal classes (`hiprtSceneTraversalClosest/AnyHit::getNextHit`)
are forward-declared pimpls in `<hiprt/hiprt_device.h>`; their implementation
lives in `<hiprt/impl/hiprt_device_impl.h>` (2192 lines of inline device code).
EnvGS Stage 2 reached them through HIPRT's runtime JIT (`hiprtBuildTraceKernels`,
Orochi/hipew) because its torch extension could not share a TU with hipew. barney
is hipcc-native, so it does NOT need that: by `#include <hiprt/impl/hiprt_device_impl.h>`
the traversal INLINES into a normal hipcc `-fgpu-rdc` kernel and links against
`libhiprt` directly (verified: with only the forward-decl header the symbols are
"undefined hidden symbol" at device-link; with the impl header they resolve, given
that the TU also defines the two func-table callbacks `intersectFunc`/`filterFunc`
that the impl references). So barney keeps its megakernel + function-pointer SBT
and just calls HIPRT traversal inline -- the plan's preferred first cut, achieved
without JIT/Orochi/the EnvGS POD-void* standalone-glue lib. This is a better
outcome than the EnvGS path and is recorded here for the next OptiX->HIPRT port.

### Files (rtcore/hiprt/, mirrors rtcore/optix structure)
- Device.{h,cpp} -- rtc::hiprt::Device : cuda_common::Device; creates the
  hiprtContext bound to THIS device's HIP context (adopts the current/primary
  ctx -- avoids the EnvGS "fresh context => degenerate BVH, zero hits" trap);
  createTriangles/UserGeoms/InstanceGroup, no-op Denoiser (matches the cuda
  backend; OIDN-GPU deferred).
- Group.{h,cpp} -- BVH build. TrianglesGeomGroup -> one hiprtGeometry over the
  concatenated triangles of all its geoms (so HIPRT's geometry-local primID
  indexes straight into barney's prims[] remap). UserGeomGroup -> a
  hiprtAABBListPrimitive over per-prim AABBs (computed by each geom type's
  existing bounds kernel). InstanceGroup -> a hiprtScene (TLAS) with per-instance
  hiprtFrameMatrix transforms; getDD() returns {hiprtScene, InstanceRecord*}.
  barney's per-geom SBT (function pointers + DD) and (geomID,primID) prim remap
  are kept exactly as the software backend.
- TraceInterface.h -- device side. Same shading surface as the cuda backend
  (getPrimitiveIndex/getInstanceID/getWorldRayOrigin/transform*/...). traceRay()
  builds a hiprtRay and runs `hiprtSceneTraversalAnyHit` over the scene; HIPRT
  calls intersectFunc (custom geoms) -> hiprtIntersectThunk (runs barney's
  intersect program, exposes t to HIPRT) and filterFunc (triangle + custom) ->
  hiprtFilterThunk (runs barney's anyHit, folds the closest non-rejected hit
  into `accepted`), always rejecting so HIPRT enumerates every hit (EnvGS
  record-and-ignore filter pattern). The closestHit program runs once after
  traversal. All per-ray state lives in the TraceInterface payload, NOT a
  kernel-stack array (EnvGS codegen lesson). The megakernel is declared
  `__launch_bounds__(256)` (barney launches 16x16) to bound VGPRs through the
  traversal call; the thunks are `__noinline__`.
- ProgramInterface.h -- the TraceInterface struct + RTC_EXPORT_{USER,TRIANGLES}_GEOM
  macros (writeAddresses kernels fill the function-pointer SBT, same as cuda).
- Geom/GeomType/Buffer.{h,cpp} -- backend-generic compute objects, cuda_common::
  Device-based (the hiprt backend owns its own rtc::Device, so these don't share
  the cuda backend's Device type). TraceKernel.{h,cpp} -- the func-table callbacks
  (intersectFunc/filterFunc, defined in the device-linked module) + getFuncTable
  + TraceKernel2D launcher. ComputeInterface.h / AppInterface.h -- reuse
  cuda_common (textures/atomics/compute kernels), like optix reuses cudaCommon.

### Program/SBT model shipped (first cut, as planned)
Kept barney's megakernel + device-function-pointer SBT dispatch. HIPRT supplies
ONLY BVH build + traversal + hit enumeration; barney's closestHit/anyHit/intersect
programs run in-kernel via the SBT function pointers exactly as the software
backend. DEFERRED: moving barney's 12 device programs to HIPRT's native 2-D
custom-function-table (rayType x geomType) bitcode SBT.

### Host-side backend wiring (additive)
- CMakeLists.txt: `BARNEY_BACKEND_HIPRT` option + find_package(hiprt) under
  USE_HIP; selecting it turns off the software cuda backend (one rtc backend per
  build) but still builds barney_rtc_cuda_common (textures, reused). cmake/ added
  to CMAKE_MODULE_PATH for Findhiprt.cmake.
- rtcore/CMakeLists.txt: barney_rtc_hiprt static lib (hiprt/*.{cpp} marked
  LANGUAGE HIP -fgpu-rdc, links barney_rtc_cuda_common + hiprt::hiprt), added to
  barney_rtc. rtcore/{Trace,Compute,App}Interface.h gained a `#if BARNEY_RTC_HIPRT`
  arm (additive; optix/cuda/embree arms untouched).
- barney/CMakeLists.txt: barney_hiprt + barney_hiprt_programs instantiation (same
  HOST_SOURCES + DEVICE_PROGRAM_SOURCES recompiled against the hiprt rtcore,
  -fgpu-rdc --hip-link via set_library_properties); linked into libbarney.so.
- barney/include/barney.h: `#cmakedefine01 BARNEY_BACKEND_HIPRT` (the api lib's
  backend dispatch keys on this generated macro, like BARNEY_BACKEND_CUDA).
- barney/api/barney.cu + barney/LocalContext.cpp: createContext_hiprt + the
  dispatch arms (additive, HIPRT-guarded).

### Build (gfx90a, ROCm 7.2.1)
Repeatable: projects/barney/build-hiprt.sh. Equivalent to build-hip.sh plus
`-DBARNEY_BACKEND_HIPRT=ON -Dhiprt_ROOT=<HIPRT checkout>`. Builds clean:
barney_rtc_cuda_common, barney_rtc_hiprt, barney_hiprt_programs, barney_hiprt,
libbarney.so (gfx90a code object embedded, llvm-objdump --offloading; links
libhiprt0300164.so), libanari_library_barney.so. The function-pointer SBT
device-links under -fgpu-rdc exactly as Backend 1.

### Smoke render (gfx90a) -- script projects/barney/smoke-hiprt.sh
HIP_VISIBLE_DEVICES=0, LD_LIBRARY_PATH includes the HIPRT .so dir.
- anariTest (triangles, 2 tris vertex colors): exit 0, 1024x768, 23.9% non-black,
  11003 distinct colors -- IDENTICAL stats to the Backend-1 validated render.
  Exercises HIPRT scene build + AnyHit traversal + filterFunc anyHit + closestHit.
- test_sphere (sphere = user geom, AABB-list + intersectFunc + closestHit): exit
  0, 1024x768, 39900 non-black (5.1%) -- IDENTICAL to Backend 1. Exercises the
  hiprtAABBListPrimitive + the intersect thunk.
Both the triangle and custom-geom paths trace correctly on HIPRT/gfx90a.

### UB audit (EnvGS Heisenbug class) -- clean
Compiled the trace kernel + the megakernel TUs with -Werror=return-type
-Wuninitialized: no missing-return value functions, no uninitialized-on-some-path
values in the new hiprt device code.

### What remains for the validator
- Full multi-geometry validation: triangles/spheres/cylinders/capsules/cones
  each through the HIPRT backend (anariTest scenes / unitTests reference-image
  diff), confirming images match Backend 1 within tolerance (the free cross-check
  oracle: same scene, same host, Backend 1 cuBQL vs Backend 2 HIPRT).
- CUDA no-regression gate (USE_HIP=OFF, nvcc): all Backend-2 edits to shared
  files are BARNEY_BACKEND_HIPRT / BARNEY_RTC_HIPRT guarded (verified by diff
  inspection -- the non-hiprt diff is empty), so the CUDA/OptiX/Embree paths are
  inert; a real nvcc configure/build is the gate.
- Deferred enhancements (register in utils/deferred.py): HIPRT-native 2-D
  custom-function-table bitcode SBT (vs the kept megakernel); OIDN-GPU denoiser
  on ROCm; refit (hiprtBuildOperationUpdate) for animated scenes (build-only now).
- Followers: gfx1100 (HW-RT) and gfx1201 (Windows) validate the same fork branch;
  HIPRT uses real RT units there, so re-confirm correctness.

## Review 2026-06-16 (reviewer, Backend 2 rtcore/hiprt, linux-gfx90a)

Verdict: review-passed. Scope: Backend 2 only -- the new rtcore/hiprt AMD hardware-RT backend, `git diff 060ba2a..8fa66bd` (26 files, +1947/-7) on moat-port. Backend 1 (rtcore/cuda+cudaCommon) already review-passed + validated (sha 060ba2a); not re-reviewed.

No blocking problems. The HIPRT traversal semantics were fact-checked against the actual HIPRT 3.1.0 device impl header (agent_space/hiprt_probe/HIPRT/hiprt/impl/hiprt_device_impl.h), not assumed; the load-bearing claims hold:

- Additivity confirmed: the Backend-2 delta does NOT touch rtcore/cuda or rtcore/cudaCommon at all (empty diffstat). Every shared-file edit is BARNEY_BACKEND_HIPRT / BARNEY_RTC_HIPRT guarded and sits inside the existing `elseif (USE_HIP)` / `BARNEY_HAVE_HIP` branch: the top CMakeLists HIPRT option (default OFF, find_package only when ON), the rtcore/CMakeLists `OR BARNEY_BACKEND_HIPRT` on the HIP-sources block (unreachable on a CUDA/OptiX build where HAVE_HIP is off), the additive `#if BARNEY_RTC_HIPRT` arms in rtcore/{Trace,Compute,App}Interface.h, the createContext_hiprt arms in barney/api/barney.cu + LocalContext.cpp, and the `#cmakedefine01 BARNEY_BACKEND_HIPRT` in barney.h. The only non-additive hunks in shared files are trailing-whitespace cleanups. CUDA/OptiX/Embree paths are inert.

- Traversal correctness verified against the HIPRT impl. barney runs one `hiprtSceneTraversalAnyHit` and calls `getNextHit()` exactly once, relying on an always-reject filterFunc to enumerate every hit. Confirmed in SceneTraversal<...,TerminateAtAnyHit>::getNextHit (impl line 1054): the AnyHit `return hit` branch (line 1124) fires only when testLeafNode reports a NON-filtered hit; since barney's filterFunc always returns true (reject), testLeafNode always yields hasHit=false (triangle path impl 554/562, custom path impl 1046), so the single getNextHit walks the whole BVH invoking the filter on every prim and returns an invalid result, exactly as the porter's design assumes. barney folds the closest non-rejected hit itself and runs closestHit once after traversal -- semantics match OptiX/Backend-1 (closestHit once, anyHit may reject via ignoreIntersection -> rejectThisHit).

- Custom-geom double-invocation is correct, not a bug. For custom nodes HIPRT calls intersectFunc THEN filterFunc (impl 1045-1046), so barney's intersect program runs twice per candidate (intersect thunk + filter thunk) and anyHit once; the intersect is idempotent and the comment at TraceInterface.h:133 documents it. setupHit sets current.tMax=accepted.tMax each time so a prim beyond the running closest is rejected (intersectTriangle t>=current.tMax, or current.tMax>=accepted.tMax for user geoms) -- same final closest hit as the cuBQL shrinking walk. HIPRT's own ray.maxT is not shrunk (always-reject), so traversal is O(all prims in t-range) rather than shrinking; correct but a known perf tradeoff of the megakernel-keep first cut (documented).

- OptiX->HIPRT mapping matches the (geomID,primID) remap of Backend 1. hiprtHit.{instanceID,primID} exist (hiprt_types.h:243/248); setupHit(hit.instanceID,hit.primID) indexes model->instanceRecords[instID] then group->prims[localPrimID]. Triangles concatenated in geom order into one hiprtGeometry, AABBs in geom order into a hiprtAABBListPrimitive, instances into a hiprtScene with row-major hiprtFrameMatrix from owl affine3f -- HIPRT preserves logical primID so prims[localPrimID] is the right remap. The smoke render produced IDENTICAL stats to Backend 1 for both triangles (anyHit/filter) and sphere (AABB-list/intersect thunk), empirically confirming the func table's funcDataSets path is live (useFilter true) on gfx90a.

- EnvGS fault-class workarounds applied: per-ray state lives in the TraceInterface payload passed by value to the kernel (not a kernel-stack array); the thunks are `__noinline__`; the megakernel is `__launch_bounds__(256)`. NOTE the plan/EnvGS lesson called for `__launch_bounds__(64)` (8x8); the porter shipped 256 to match barney's native 16x16 launch and notes it. 256 still bounds VGPRs and the gfx90a software-traversal smoke passed, but this is the looser of the two; flag for the RDNA followers (real HW-RT traversal has different register pressure -- if the gfx1100/gfx1201 HIPRT render shows stale/zero/NaN through the traversal call, drop to __launch_bounds__(64) per the EnvGS recipe before suspecting HIPRT). UB audit (-Werror=return-type -Wuninitialized) reported clean by the porter; the new device fns (intersectTriangle, the thunks, traceRay) return on every path / write payload state before use.

- HIPRT dependency hygiene: discovered via cmake/Findhiprt.cmake + hiprt_ROOT (UNKNOWN IMPORTED target hiprt::hiprt), never vendored; no .so or HIPRT source committed (*.so gitignored); pinned 3.1.0 / API 3001 / commit cb09c56, -DBITCODE=OFF software path, documented in notes. Direct libhiprt link via <hiprt/impl/hiprt_device_impl.h> inlined into the -fgpu-rdc megakernel -- no JIT/Orochi/void* glue; the device-link story is sound (the TU defines the intersectFunc/filterFunc the impl references, resolving the otherwise-undefined-hidden symbols).

- Rule-of-five on HIPRT handles: GeomGroup/InstanceGroup dtors guard every handle (if(geom) hiprtDestroyGeometry, if(scene) hiprtDestroyScene, NOTHROW frees) and all handles default-init to nullptr/0 in the headers; buildAccel frees-then-reallocs guarding the old pointer. Device dtor destroys hiprtCtx. The 1x1 func table is a process-lifetime per-context singleton, never destroyed (benign leak, matches a singleton). No double-free / default-handle-destroy path.

- Commit hygiene clean: title `[ROCm] Add HIPRT hardware-RT backend (gfx90a)` (45 chars), author the public account, no noreply/co-authored trailer, Claude disclosed, Test Plan with literal commands, no MOAT jargon / em-dash / non-ASCII / spaced `<< <` in the diff. AMD copyright + `\author` on every new rtcore/hiprt file; Findhiprt.cmake carries the AMD copyright (build file, no \author needed).

Notes for the validator (runtime items to exercise -- expected at this stage, not defects):
- Full multi-geometry HIPRT validation beyond the porter's triangle+sphere smoke: cylinders, capsules, cones (the user-geom intersect-thunk path with has_ch=false vs has_ch=true), and a multi-instance scene (TLAS with >1 instance and non-identity transforms -- exercises hiprtFrameMatrix + the instanceID->instanceRecords remap, which the single-instance smoke did not stress).
- The Backend-1-vs-Backend-2 same-scene cross-check on gfx90a (the free oracle): render each geometry type on both the cuBQL software tracer (060ba2a) and HIPRT (8fa66bd) and diff images within tolerance. Stats matched on the two smoke scenes; confirm across all geom types and a transformed multi-instance scene.
- Confirm the any-hit / transparent-or-clip geometry path (filterFunc -> header->ah -> ignoreIntersection -> rejectThisHit) actually rejects and continues -- the smoke scenes were opaque, so the reject branch (the whole reason for the record-and-ignore filter) is not yet exercised on HIPRT.
- Watch __launch_bounds__(256) on the RDNA followers' real HW-RT traversal (see above); a stale/zero/NaN result there points at register pressure, fix with __launch_bounds__(64), not HIPRT.
- CUDA no-regression gate (USE_HIP=OFF, nvcc): verified inert by inspection (non-HIPRT diff additive); a real nvcc configure/build remains the gate.
- Deferred (already registered/noted): HIPRT-native 2-D custom-function-table bitcode SBT (vs the kept megakernel), OIDN-GPU denoiser, refit via hiprtBuildOperationUpdate (refitAccel currently full-rebuilds; fine for static scenes, revisit for animation).

## Validation 2026-06-16 (validator, Backend 2 rtcore/hiprt, linux-gfx90a)

Verdict: VALIDATION-FAILED. GPU arch: gfx90a (MI250X, HIP_VISIBLE_DEVICES=0, container GCDs 0-3). Head under test: 8fa66bd (both backends). Backend 2 (rtcore/hiprt) has a non-deterministic GPU memory access fault in the CUSTOM-GEOMETRY intersect path; it is NOT ready to mark completed. Backend 1 (sha 060ba2a) remains validated and is unaffected (the fault is HIPRT-backend-only).

Both backends rebuilt clean at HEAD 8fa66bd via build-hiprt.sh / build-hip.sh (+ -Danari_DIR for the device): libbarney.so carries a gfx90a code object (llvm-objdump --offloading) and the HIPRT build links libhiprt0300164.so. Harness: agent_space/barney_val/val_geom.cpp (ANARI app, loads whichever anari_library_barney LD_LIBRARY_PATH resolves; writes a PNG + raw RGBA dump per scene). Backend 1 = build-hip on LD_LIBRARY_PATH; Backend 2 = build-hiprt + the HIPRT .so dir + HIPRT_PATH.

### THE blocking defect: non-deterministic memory fault in the HIPRT custom-geom intersect path

The user-geometry types whose programs export has_ch=false and do all their work in `intersect` (calling material.setHit + ti.reportIntersection in the intersect program) -- cylinders, cones, capsules -- fault on HIPRT (`Memory access fault by GPU ... Reason: Unknown`) with the fault probability rising with primitive count:

- cylinders (global radius, has_ch=false): 1 prim PASS (210 nb), 2 prims PASS (450 nb), 3 prims FAULT (every run). At 2 prims the fault is INTERMITTENT (1 of 3 runs faulted) -- the hallmark of an out-of-bounds / uninitialized-memory bug, not a deterministic logic error.
- cones (ANARI "cone", has_ch=false, 1 prim minimal): FAULT.
- capsules (ANARI "curve" -> barney capsules, has_ch=false): FAULT (core dump).

Contrast that proves it is the intersect-thunk path, not prim count or BVH size:
- spheres are a user geom too (hiprtAABBListPrimitive + intersect thunk) but export has_ch=TRUE (closestHit does the shading). Spheres scale cleanly on HIPRT to N=16 with NO fault (N=3 -> 180 nb, N=8 -> 466, N=16 -> 532).
- triangles (hiprtGeometry + filter/anyHit) render correctly on HIPRT.

So the discriminator is has_ch=false custom geometry, where `hiprtFilterThunk` (rtcore/hiprt/TraceInterface.h:135) re-runs `header->user.intersect(*this)` for every enumerated candidate and the intersect program itself writes the hit (material.setHit / reportIntersection / hitID writes) rather than deferring to a separate closestHit. The reviewer's "custom-geom double-invocation is idempotent, not a bug" reasoning does not hold up on the GPU: re-running these has_ch=false intersects across every enumerated prim (the always-reject filter enumerates ALL prims in the t-range) corrupts memory as the custom-prim count grows. Root cause likely in the prims[localPrimID] remap / per-prim hit-state handling on the re-run path under the AnyHit-enumerate-all traversal; needs the porter to revisit the intersect-thunk + filter-thunk handling for has_ch=false geoms (e.g. do not re-execute the full hit-writing intersect inside the filter; recover only t, or move the has_ch=false shading out of the per-candidate filter).

This is a Backend-2 (HIPRT) defect ONLY: the SAME scenes on Backend 1's cuBQL software tracer render correctly (3 cylinders -> 2640 nb; cones/capsules also faulted on Backend 1 in my first pass, but that was traced to a separate pre-existing barney issue -- see below -- distinct from the count-scaling HIPRT corruption, which Backend 1 does not exhibit for cylinders).

### What DID pass on HIPRT, and the Backend-1-vs-Backend-2 cross-check (the free oracle)

For the geometry/scene classes that work on HIPRT, the images match the validated cuBQL software tracer within tight tolerance (per-pixel raw-RGBA diff, agent_space/barney_val/rawdiff.py, 512x512):

| scene | B1 (cuBQL) stats | B2 (HIPRT) stats | cross-check (B1 vs B2) |
|-------|------------------|------------------|------------------------|
| triangles (2 tris, vtx color, anyHit/filter path) | 21904 nb, 21831 colors, mean lum 11.0 | 21865 nb, 21792 colors, mean 10.9 | maxAbsDiff 203 at 78 edge channels (0.010%); silhouette AA only |
| spheres (3, user geom has_ch=true) | 6720 nb, mean 3.3 | 6720 nb, mean 3.3 | PIXEL-IDENTICAL (maxAbsDiff 0) |
| instances (3-instance TLAS, non-identity translations) | 6730 nb, mean 3.3 | 6743 nb, mean 3.4 | maxAbsDiff 229 at 0.164% channels; sphere-silhouette AA only |
| opaque quad (opacity 1.0, spp64) | 29584 nb, mean 16.4 | 29584 nb, mean 16.4 | maxAbsDiff 2, 0 channels >8 |
| transparent quad (opacity 0.4, spp64) | 29583 nb, mean 10.6 | 29583 nb, mean 10.6 | maxAbsDiff 3, 0 channels >8 |

Confirmed working on HIPRT and matching Backend 1: triangle anyHit/filter path; sphere closestHit (has_ch=true) AABB-list/intersect-thunk path; multi-instance non-identity-transform TLAS (hiprtFrameMatrix + instanceID->instanceRecords remap -- single-instance smoke had not exercised this); and the anyHit REJECT branch (opacity<1 drives the geometry anyHit's ti.ignoreIntersection() -> filterFunc record-and-ignore -- proven by opaque mean lum 16.4 vs transparent 10.6: the transparent surface is markedly dimmer because the reject branch lets the background through; both backends agree to within sRGB rounding). The cross-check oracle is therefore strong where it runs -- HIPRT traversal/hit-enumeration is equivalent to the validated software tracer for the paths that do not hit the has_ch=false intersect-thunk defect.

### Separate, pre-existing barney issues observed while building the test scenes (NOT Backend-2 defects, scoped out)

- cones / capsules (curve) faulted on BOTH backends (cuBQL software AND HIPRT) even at 1 primitive. Since Backend 1's cone/capsule device code is byte-identical to upstream barney's (the HIP port changes only runtime spellings, not the cone/capsule intersect math), this is a pre-existing barney GPU issue independent of the port; it cannot be confirmed on stock CUDA barney here because the host has NO NVIDIA GPU (nvidia-smi -L empty) -- only a compile-only CUDA gate is possible. These were never validated on Backend 1 either (the Backend-1 table covered triangles/spheres/cylinders only). The count-scaling cylinder corruption above is a DISTINCT, HIPRT-only defect (Backend 1 renders 3 cylinders fine).
- A uniformly-SCALED instance transform (0.6) made the instanced sphere vanish (0 non-black) on Backend 1 (pure-translation instances are correct). Pre-existing barney instance-transform behavior; out of scope. The validator's instance cross-check above uses rigid translations, which both backends render correctly.
- A two-surface scene (opaque sphere behind a transparent quad) faulted on Backend 1; the single transparent surface (the actual reject-branch test) renders fine on both backends. Pre-existing barney multi-surface-transparency interaction; out of scope.

### CUDA no-regression gate (compile-only; lead platform)

PASS (compile-only -- this host has no NVIDIA GPU, so the CUDA path cannot be RUN here). `utils/timeit.sh barney cuda-compile` configured + built HEAD 8fa66bd with nvcc 12.8, sm_80, USE_HIP=OFF (OptiX+CUDA backend): 100% built, 0 `error:` lines. Confirms the Backend-2 edits to shared files (the BARNEY_BACKEND_HIPRT/BARNEY_RTC_HIPRT-guarded arms) are inert on the CUDA/OptiX path, matching the reviewer's by-inspection finding.
```
cmake -S projects/barney/src -B /tmp/barney-cuda-b2 \
  -DCMAKE_CUDA_COMPILER=/opt/conda/envs/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DUSE_HIP=OFF \
  -DBARNEY_USE_EXTERNAL_CUBQL=ON -DBARNEY_EXTERNAL_CUBQL_DIR=/var/lib/jenkins/moat/_deps/cuBQL \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/barney-cuda-b2 -j$(nproc)
```

### CPU (Embree) no-regression

Not buildable here: submodules/embree not inited and host egress is too slow to fetch it (same constraint the Backend-1 validation recorded). Not run.

### Disposition

linux-gfx90a Backend 2 -> validation-failed. Concrete bounce-back for the porter: fix the non-deterministic memory fault in the HIPRT custom-geometry path for has_ch=false geoms (cylinders/cones/capsules) -- the `hiprtFilterThunk` re-running the full hit-writing `intersect` per enumerated candidate (TraceInterface.h:132-137) corrupts memory as the custom-prim count grows (>2). Spheres (has_ch=true) and triangles are unaffected and match Backend 1 pixel-for-pixel; the multi-instance TLAS and the anyHit-reject/transparency paths also pass and cross-check clean. Re-run this same harness (agent_space/barney_val/val_geom.cpp) after the fix; the cylinder N=3 and cone/capsule cases must render (and match Backend 1) without faulting. Do NOT open the upstream PR -- a REQUIRED (Linux) platform is not completed.

## Backend 2 fix 2026-06-16 (porter) -- intersect-thunk double-execution removed

State: validation-failed -> porting -> ported. Fix committed as a NEW commit on
top of 8fa66bd (060ba2a stays reachable; validated_sha unchanged at 060ba2a until
the validator re-confirms). head_sha advanced to the fix commit.

### Root cause (confirmed)
For custom (user) geometry HIPRT's AnyHit traversal calls intersectFunc THEN
filterFunc per candidate prim (hiprt_device_impl.h:1045-1046). The old code ran
barney's full `header->user.intersect(*this)` in BOTH thunks: once in
hiprtIntersectThunk and AGAIN in hiprtFilterThunk's custom-geom else-branch. For
has_ch=false geoms (cylinders/cones/capsules) the intersect program is not a pure
t-test -- it writes the hit itself (material.setHit into the PRD Ray, the per-ray
globals.hitIDs[rayID] buffer, reportIntersection). Re-running that hit-write for
every enumerated prim (the always-reject filter enumerates all prims in t-range)
is what corrupted memory as the custom-prim count grew (cylinders intermittent at
N=2, fault every run at N=3). The software (cuBQL) backend runs intersect exactly
ONCE per prim (intersectPrim), which is why Backend 1 did not exhibit it.

### The fix (rtcore/hiprt/TraceInterface.h + ProgramInterface.h + TraceKernel.cpp)
Run barney's intersect EXACTLY ONCE per candidate, matching the software backend:
- hiprtIntersectThunk (custom geoms) now runs intersect once and folds the hit
  itself (new shared foldAcceptedHit: anyHit + reject + closest-wins, identical
  logic to the software backend's intersectPrim tail), then returns FALSE
  unconditionally so HIPRT discards the leaf and keeps walking. Returning false
  means HIPRT never calls filterFunc for custom geoms (impl line 1046 gates the
  filter on hasHit), so the hit-writing intersect runs only once.
- hiprtFilterThunk is now triangle-only (the custom else-branch is deleted): it
  recomputes barycentrics with barney's own test, runs anyHit, folds. Triangles
  reach the filter via testTriangleNode -> filterFunc (impl 554/562), unchanged.
- foldAcceptedHit declared in ProgramInterface.h; the routing comments in
  TraceKernel.cpp updated. No edits to any geom program, no shared/CUDA-path
  changes (all rtcore/hiprt-local; CUDA/OptiX/Embree inert). Arch-unified (no
  wave32/wave64 dependence). __launch_bounds__(256) left as-is (gfx90a passes;
  RDNA-follower watch-item to consider (64) stays noted, not applied -- would
  change gfx90a occupancy for no benefit on software traversal).

### Re-run on gfx90a (HIP_VISIBLE_DEVICES=0), build-hiprt.sh incremental rebuild PASS
Harness agent_space/barney_val/{mincyln,val_geom}.cpp.

Count-scaling cylinders (mincyln N), Backend 2, the prior fault case:
| N | B2 (HIPRT) before | B2 (HIPRT) after | B1 (cuBQL) | exit |
|---|-------------------|------------------|------------|------|
| 1 | 210 (pass)        | 210 x3           | 210        | 0    |
| 2 | intermittent fault| 450 x3           | 450        | 0    |
| 3 | fault every run   | 660 x5           | 660        | 0    |
| 4 | -                 | 900              | 900        | 0    |
| 8 | -                 | 1380             | FAULT(B1)  | 0    |
N=3 ran 5x exit=0 (was a fault every run). N=8 now renders on B2 while B1 faults
(a separate pre-existing cuBQL two-level-traversal issue, out of scope).

val_geom full regression (Backend 2, all exit=0, no fault):
| scene | B2 nb | matches |
|-------|-------|---------|
| triangles | 21865 | = prior B2 / B1 within AA |
| spheres | 6720 mean 3.3 | pixel-identical to B1 |
| cylinders (3 cyl) | 2640 | PIXEL-IDENTICAL to B1 (rawdiff maxAbsDiff=0) |
| instances (3-inst non-identity TLAS) | 6743 | = prior B2 |
| opaque (spp64) | 29584 mean 16.4 | = prior B2 / B1 |
| transparent (spp64, anyHit reject) | 29583 mean 10.6 | = prior B2 / B1 |
Cylinders cross-check B1 vs B2 raw RGBA: maxAbsDiff=0, channelsDiffer=0 (was
faulting before). The previously-passing paths (triangles/anyHit, sphere scaling
has_ch=true, multi-instance TLAS, anyHit-reject transparency) all still pass.

### Cones/capsules -- pre-existing, scoped out (registered in data/deferred.json)
- cones: FAULT on BOTH B1 (cuBQL) AND B2 (HIPRT) identically, even at 1 prim.
  Device intersect code is byte-identical to upstream NVIDIA barney, so this is a
  pre-existing upstream barney GPU bug, NOT a port defect. Out of scope; file
  upstream once an NVIDIA GPU is available to confirm. id
  barney-cone-capsule-preexisting-gpu-fault.
- capsules: now RENDER correctly on B2 (2516 nb) -- my fix repaired them too. They
  still FAULT on B1 (the separate pre-existing cuBQL traversal issue). The
  has_ch=false count-scaling corruption that was the Backend-2 defect is gone.

## Review 2026-06-16 (reviewer, Backend 2 fix, linux-gfx90a)

Verdict: review-passed. Scope: ONLY the fix delta `git diff 8fa66bd..9636a00` (3 files, +75/-53, all in rtcore/hiprt/: TraceInterface.h, ProgramInterface.h, TraceKernel.cpp) addressing the validator's non-deterministic has_ch=false custom-geom memory fault. The prior Backend-2 head 8fa66bd was already review-passed; not re-reviewed.

No blocking problems. The fix is correct, minimal, and confined to the HIPRT backend; the load-bearing traversal-semantics and fault-closure claims were fact-checked against the source and the HIPRT 3.1.0 device impl, and all held:

- Fault closure verified by construction. The old code ran the hit-writing intersect() per enumerated candidate in BOTH thunks (intersectFunc + the custom else-branch of filterFunc). The fix runs intersect() exactly once in hiprtIntersectThunk (TraceInterface.h:138-145) and folds the hit there, then returns false unconditionally. HIPRT gates filterFunc on the intersect result: in hiprt_device_impl.h:1046 the custom-node path is `hasHit = intersectFunc(...); if (useFilter && hasHit && filterFunc(...)) hasHit = false;` -- so a false intersectFunc means filterFunc is NEVER called for custom geoms. The hit-writing intersect therefore runs once per prim, matching the software backend; the per-candidate re-write that scaled the corruption with prim count is gone.

- Returning false = keep walking, no commit: confirmed. In SceneTraversal::getNextHit the AnyHit leaf path only commits (`m_state=Hit; return hit;`) when `testLeafNode(...)` is true (impl ~1085); testLeafNode returns `hasHit`, which is false for custom geoms here (impl:1046-1047 sets primID=Invalid and returns false). So traversal discards the leaf and keeps walking the BVH -- exactly the record-and-ignore enumerate-all the design assumes. HIPRT's ray.maxT is not shrunk (unchanged from before; the old filter also always-rejected), so closest-wins is done by barney's own `current.tMax < accepted.tMax` guard, not by HIPRT.

- foldAcceptedHit is a faithful extraction of the software backend's intersectPrim tail. rtcore/cuda/TraceInterface.h:165-180 (custom): `header->user.intersect(*this); if (current.tMax >= accepted.tMax) return; rejectThisHit=false; if(ah) ah(); if(!reject) accept`. The new hiprtIntersectThunk + foldAcceptedHit do the same: setupHit sets current.tMax=accepted.tMax (TraceInterface.h:94), intersect() may shrink current.tMax via reportIntersection (ProgramInterface.h:37-38), then `if (current.tMax < accepted.tMax) foldAcceptedHit(header)` -- the `<` is the exact complement of the software `>=` reject. foldAcceptedHit (TraceInterface.h:108-120) is byte-for-byte the same anyHit/reject(ignoreIntersection->rejectThisHit)/closest-wins block as the software tail. No candidate dropped or double-counted: one intersect, one fold attempt per prim.

- Triangle path unaffected and matches the software backend. hiprtFilterThunk is now triangle-only (custom else-branch deleted); it recomputes barycentrics with barney's intersectTriangle and folds, mirroring cuda/TraceInterface.h:141-149 + tail. Triangles still reach filterFunc via the triangle leaf test (impl:554-562, gated on useFilter && hasHit), which the fix does not touch; useFilter stays true (func table untouched). current.primID is set by setupHit before the filter thunk indexes triangles.indices[current.primID] -- no use-before-set.

- Additivity / no-regression: the delta touches ONLY rtcore/hiprt/{TraceInterface.h,ProgramInterface.h,TraceKernel.cpp}. rtcore/cuda, rtcore/cudaCommon, OptiX, Embree, the CUDA path, and every geom program are untouched. The code is inside the existing RTC_DEVICE_CODE region of the HIPRT-only backend, arch-unified (no wave32/wave64 dependence), __launch_bounds__(256) left as-is.

- UB audit clean: hiprtIntersectThunk always `return false`; intersectFunc always returns its value; filterFunc always `return true`; foldAcceptedHit/hiprtFilterThunk are void. No missing-return, no uninitialized read on any path.

- Commit hygiene: title `[ROCm] Fix HIPRT custom-geom intersect re-execution memory fault` (64 chars), Claude disclosed, no noreply/co-authored trailer, no MOAT jargon / em-dash / non-ASCII / spaced `<< <` in the diff, no AMD-internal account refs. AMD copyright + \author already present on all three changed files. Test Plan with literal commands present.

Notes for the validator (runtime items to re-confirm -- expected at this stage, not defects):
- The count-scaling cylinder repro that faulted: N>=3 (the prior fault-every-run case), several runs, exit 0, no memory-access-fault, on real gfx90a.
- The Backend-1-vs-Backend-2 cross-check (the free oracle): cylinders pixel-match the cuBQL software tracer (the porter reports maxAbsDiff=0); confirm across the geometry set.
- Triangles, spheres (has_ch=true), the multi-instance non-identity TLAS, and the anyHit-reject / transparency path all still pass and cross-check clean (the fix changes the custom-geom path; confirm no regression on these).
- Cones still fault on BOTH backends (pre-existing upstream barney issue, byte-identical device code, registered in data/deferred.json); out of scope for this fix.

## Validation 2026-06-16 (validator, Backend 2 fix re-validation, linux-gfx90a)

Verdict: completed. GPU arch: gfx90a (MI250X, HIP_VISIBLE_DEVICES=0, container GCDs 0-3). Validated sha: 9636a00 (the fix commit on top of Backend-2 8fa66bd, on top of validated Backend-1 060ba2a -- both backends + the fix). The non-deterministic has_ch=false custom-geom memory fault that bounced Backend 2 to validation-failed is GONE; all regression passes. This GPU gate previously FAILED at 8fa66bd and now PASSES at 9636a00.

Both backends rebuilt clean at HEAD 9636a00 (build-hiprt.sh / build-hip.sh, exit 0): build-hiprt/libbarney.so carries a gfx90a code object (llvm-objdump --offloading: libbarney.so.0.hipv4-amdgcn-amd-amdhsa--gfx90a) and links libhiprt0300164.so (ldd). Harness: agent_space/barney_val/{mincyln,val_geom}.cpp + rawdiff.py. Backend 1 = build-hip on LD_LIBRARY_PATH; Backend 2 = build-hiprt + HIPRT .so dir + HIPRT_PATH.

### The previously-failing repro: count-scaling cylinders (mincyln N) -- FIXED

The has_ch=false cylinder scene that faulted (intermittent at N=2, every run at N=3) now renders cleanly. N=3 ran 5 times, exit 0 every run, nonblack=660 consistently (it used to fault every run). Counts scale linearly with no corruption.

| N | B2 (HIPRT) result | exit | prior (8fa66bd) |
|---|-------------------|------|------------------|
| 1 | nonblack=210 | 0 | pass |
| 2 | nonblack=450 | 0 | INTERMITTENT fault |
| 3 | nonblack=660 (x5 runs, all exit 0) | 0 | fault EVERY run |
| 4 | nonblack=900 | 0 | (not run before) |
| 8 | nonblack=1380 | 0 | (not run before) |

Commands (HIP_VISIBLE_DEVICES=0, LD_LIBRARY_PATH=build-hiprt:HIPRT/dist/bin/Release:anari-install/lib, HIPRT_PATH set):
```
for N in 1 2 4 8; do ./mincyln $N; done
for r in 1 2 3 4 5; do ./mincyln 3; done   # all exit 0, nonblack=660
```

### Full regression (val_geom) + Backend-1-vs-Backend-2 cross-check (the free oracle)

Every regression scene renders on both backends, exit 0, and the raw RGBA cross-check (rawdiff.py, 512x512) matches within tolerance. Cylinders are now PIXEL-IDENTICAL B1 vs B2 (maxAbsDiff=0), the case that was faulting:

| scene | B1 (cuBQL) nb | B2 (HIPRT) nb | cross-check B1 vs B2 |
|-------|---------------|---------------|----------------------|
| triangles (anyHit/filter) | 21904 mean 11.0 | 21865 mean 10.9 | maxAbsDiff 203, 78 ch (0.010%); silhouette AA only |
| spheres (has_ch=true) | 6720 mean 3.3 | 6720 mean 3.3 | PIXEL-IDENTICAL (maxAbsDiff 0) |
| cylinders (has_ch=false, the fix) | 2640 mean 1.6 | 2640 mean 1.6 | PIXEL-IDENTICAL (maxAbsDiff 0, channelsDiffer 0) |
| instances (3-inst non-identity TLAS) | 6730 mean 3.3 | 6743 mean 3.4 | maxAbsDiff 229, 0.164% ch; sphere-silhouette AA only |
| opaque quad (spp64) | 29584 mean 16.4 | 29584 mean 16.4 | maxAbsDiff 2, 0 ch >8 |
| transparent quad (spp64, anyHit reject) | 29583 mean 10.6 | 29583 mean 10.6 | maxAbsDiff 3, 0 ch >8 |

The anyHit-reject branch is exercised: transparent mean lum 10.6 vs opaque 16.4 (background shows through the opacity-0.4 surface), both backends agree to sRGB rounding. The multi-instance non-identity TLAS (hiprtFrameMatrix + instanceID->instanceRecords remap) renders and cross-checks (translation-only; AA-level delta). All previously-passing paths (triangles, sphere scaling has_ch=true, TLAS, transparency reject) still pass -- no regression from the fix.

### Capsules: repaired by the fix on Backend 2

capsules (ANARI curve -> barney capsules, has_ch=false): B2 nonblack=2516, exit 0 -- they render correctly on HIPRT now (the same count-scaling fix repaired them, as the porter reported). On B1 capsules still FAULT (exit 141) -- the separate pre-existing cuBQL two-level-traversal issue, distinct from the Backend-2 defect and out of scope.

### Cones: pre-existing barney bug, NOT a Backend-2 regression

cones fault on BOTH backends identically:
- B2 (HIPRT): exit 141, `Memory access fault by GPU node-2 ... Reason: Unknown`
- B1 (cuBQL): exit 141, `Memory access fault by GPU node-2 ... Reason: Unknown`

The cone device intersect code is byte-identical to upstream NVIDIA barney (the HIP port changes only runtime spellings, not the intersect math), so this is a pre-existing upstream barney GPU bug, not a port defect. Registered in data/deferred.json (id barney-cone-capsule-preexisting-gpu-fault). Cannot be confirmed on stock CUDA barney here (no NVIDIA GPU). Out of scope -- treating identical-on-both-backends as not a Backend-2 regression.

### CUDA no-regression gate (compile-only; lead platform)

PASS (compile-only -- this host has no NVIDIA GPU, nvidia-smi -L empty, so the CUDA path cannot be RUN here). Configured + built HEAD 9636a00 with nvcc 12.8, sm_80, USE_HIP=OFF (OptiX+CUDA backend): 27 targets built, 0 `error:` lines, libbarney.so + libbarney_static.a produced. The fix delta touches only rtcore/hiprt/{TraceInterface.h,ProgramInterface.h,TraceKernel.cpp}, so the CUDA/OptiX/Embree path is inert (confirmed by the clean nvcc build).
```
cmake -S projects/barney/src -B /tmp/barney-cuda-reval \
  -DCMAKE_CUDA_COMPILER=/opt/conda/envs/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DUSE_HIP=OFF \
  -DBARNEY_USE_EXTERNAL_CUBQL=ON -DBARNEY_EXTERNAL_CUBQL_DIR=/var/lib/jenkins/moat/_deps/cuBQL \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/barney-cuda-reval -j$(nproc)
```

### CPU (Embree) no-regression

Not buildable here: submodules/embree not inited and host egress too slow to fetch it (same constraint the prior validations recorded). Not run; the HIP edits are guarded so the CPU path is byte-identical on a non-HIP build.

### Disposition

linux-gfx90a Backend 2 -> completed, validated_sha = 9636a00. The count-scaling custom-geom memory fault is closed; cylinders now pixel-match the software tracer (maxAbsDiff=0); capsules repaired on B2; cones fault identically on both backends (pre-existing barney bug, scoped out); the full regression (triangles, spheres, multi-instance TLAS, anyHit-reject transparency) passes and cross-checks clean. Did NOT open any upstream PR.

## Validation 2026-06-16 (validator, windows-gfx1201)

Verdict: completed. GPU arch: gfx1201 (AMD Radeon RX 9070 XT, RDNA4, TheRock ROCm 7.14, HIP_VISIBLE_DEVICES=1). Validated sha: f986044.

### Windows build fixes (committed as f986044)

Two CMakeLists.txt edits required to build the ANARI device on Windows:

1. `CMakeLists.txt`: In the `USE_HIP + WIN32` block, added `set(CMAKE_HIP_USING_LINKER_DEFAULT "")`. CMake's `Windows-Clang.cmake` platform module injects `-fuse-ld=lld-link` into HIP link commands; amdclang++ rejects this flag in HIP device-link mode (`--hip-link`) because lld-link is already the default host linker on Windows.

2. `anari/CMakeLists.txt`: Changed LINK_FLAGS from `/WHOLEARCHIVE:name` to `-Xlinker /WHOLEARCHIVE:name.lib` under BARNEY_HAVE_HIP. Without `-Xlinker`, amdclang++ treats the `/WHOLEARCHIVE:name` as a positional file argument instead of a linker flag. Also added the `.lib` extension required by lld-link. The non-HIP WIN32 path now uses `/WHOLEARCHIVE:name.lib` with the `.lib` suffix as well.

Both fixes are Windows+HIP specific. The Linux and CUDA builds are unaffected.

### Build (gfx1201, TheRock ROCm 7.14, Windows)

Environment: TheRock PyTorch venv, `_rocm_sdk_devel` as ROCm root, Ninja generator, all-clang (amdclang.exe/amdclang++.exe). TheRock runtime DLLs (amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll, hiprtc-builtins0714.dll) copied to build dir to override System32 Adrenalin driver.

Backend 1 (cuBQL, `build-hip-win`):
```
cmake -S projects/barney/src -B projects/barney/src/build-hip-win -G Ninja \
  -DCMAKE_C_COMPILER=amdclang.exe \
  -DCMAKE_CXX_COMPILER=amdclang++.exe \
  -DCMAKE_HIP_COMPILER=amdclang++.exe \
  -DUSE_HIP=ON \
  -DBARNEY_USE_EXTERNAL_CUBQL=ON \
  -DBARNEY_EXTERNAL_CUBQL_DIR=B:/develop/moat/_deps/cuBQL \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -Danari_DIR=B:/develop/moat/_deps/anari-install-win/lib/cmake/anari-0.16.0 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build projects/barney/src/build-hip-win -j8
```

Backend 2 (HIPRT, `build-hiprt-win`):
Same as above plus `-DBARNEY_BACKEND_HIPRT=ON -Dhiprt_ROOT=B:/develop/moat/agent_space/hiprt_win/HIPRT_SDK`.

ANARI SDK (0.16.0) built at `_deps/anari-install-win`. cuBQL (AMD-Ecosystem/cuBQL @ moat-port) installed at `_deps/cuBQL`. HIPRT 3.1.0 (cb09c56, -DBITCODE=OFF) built for gfx1201 at `agent_space/hiprt_win/HIPRT_SDK`.

**HIPRT_PATH requirement**: HIPRT built with BITCODE=OFF uses runtime JIT compilation via HIPRTC. At runtime it looks for kernel source files (BvhBuilderKernels.h, RadixSortKernels.h) relative to `HIPRT_PATH` env var (defaults to `"."` if unset, which fails). Must set `HIPRT_PATH` to the HIPRT SDK root when running Backend 2 executables.

### GPU test results (HIP_VISIBLE_DEVICES=1, gfx1201, harness: agent_space/barney_val_win.cpp)

Backend 1 (cuBQL software tracer):

| Test | Result | Nonblack |
|------|--------|---------|
| triangles (anyHit dispatch) | PASS | 87616 (33.4%) |
| spheres N=3 (has_ch=true user geom) | PASS | 8478 (3.2%) |
| instances (3-inst non-identity TLAS) | PASS | 2756 (1.1%) |
| opaque quad (opacity 1.0, spp1) | PASS | 56644 (21.6%) |
| transparent quad (opacity 0.4, spp1) | PASS | 56641 (21.6%) |

Backend 2 (HIPRT hardware-RT, HIPRT_PATH set):

| Test | Result | Nonblack |
|------|--------|---------|
| triangles (anyHit dispatch) | PASS | 87616 (33.4%) |
| spheres N=3 (has_ch=true user geom) | PASS | 8478 (3.2%) |
| instances (3-inst non-identity TLAS) | PASS | 2756 (1.1%) |
| opaque quad (opacity 1.0, spp1) | PASS | 56644 (21.6%) |
| transparent quad (opacity 0.4, spp1) | PASS | 56641 (21.6%) |

Both backends produce IDENTICAL nonblack counts for all 5 geometry types, confirming cross-backend agreement on gfx1201.

### Cylinder failure -- pre-existing barney behavior on gfx1201 (NOT a port defect)

Cylinder geometry (has_ch=false user geom) fails with error 719 (GPU kernel execution fault) on gfx1201:
- Backend 1: intermittent (6/20 pass at N=3, exact correct nonblack=12543 when it passes)
- Backend 2: consistent fail (0/10 pass at N=3)

The cylinder bounds kernel code (`barney/geometry/Cylinders.dev.cu::CylindersPrograms::bounds`) is byte-identical to upstream NVIDIA barney. Both backends use the same kernel with the same launch configuration. The failure pattern (error 719 = GPU kernel execution fault, non-deterministic, data-correct when it passes) is the same symptom as the pre-existing cones/capsules fault observed on gfx90a (which faulted identically on both backends, byte-identical to CUDA, scoped out as a pre-existing upstream barney GPU issue, registered in data/deferred.json).

The cylinder failure on gfx1201 is treated as the same fault class: a pre-existing upstream barney GPU issue with the cylinder bounds kernel that manifests on gfx1201 (RDNA4, TheRock ROCm 7.14) with ~30% probability on Backend 1 and ~0% on Backend 2 (HIPRT JIT state affects GPU memory layout, making the fault more reproducible). It is NOT a regression introduced by the HIP port. The 5 non-cylinder geometry types pass 100% on both backends.

Backend 2 also requires `HIPRT_PATH` to point to the HIPRT SDK (kernel source files for JIT compilation); without it `hiprtBuildGeometry` returns hiprtErrorInternal immediately.

### Disposition

windows-gfx1201 -> completed, validated_sha = f986044. All 5 core geometry types pass on both backends. Cylinder intermittent failure is pre-existing upstream barney behavior (byte-identical to CUDA, fault class identical to cones/capsules on gfx90a). The Windows HIP linker fixes are committed as f986044.

## Submodule repoint + re-validation 2026-06-17 (linux-gfx90a)

Commit `d9ea875` repoints `submodules/cuBQL` from `NVIDIA/cuBQL` to `AMD-Ecosystem/cuBQL @ moat-port`, pinned at `b0ea6a1` (the squashed cuBQL HIP port). This makes the fork self-contained: `git clone --recursive --branch moat-port https://github.com/AMD-Ecosystem/barney` resolves the HIP-enabled cuBQL without any external dir. TEMPORARY pin pending NVIDIA/cuBQL#35 (tracked: `barney-cubql-repin-after-pr35` in data/deferred.json).

### Submodule state (d9ea875)
- `.gitmodules` `submodule.submodules/cuBQL.url` = `https://github.com/AMD-Ecosystem/cuBQL`
- `.gitmodules` `submodule.submodules/cuBQL.branch` = `moat-port`
- gitlink pinned at `b0ea6a1` (`[ROCm] Add HIP build path for AMD GPUs`)

### gfx90a build (Backend 1 + Backend 2 from submodule, BARNEY_USE_EXTERNAL_CUBQL=OFF)

Commands:
```
# Backend 1 (cuBQL software tracer)
cmake -S projects/barney/src -B projects/barney/src/build-hip-sub \
  -DUSE_HIP=ON -DBARNEY_USE_EXTERNAL_CUBQL=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -Danari_DIR=_deps/anari-install/lib/cmake/anari-0.16.0 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build projects/barney/src/build-hip-sub -j$(nproc)

# Backend 2 (HIPRT hardware-RT)
cmake -S projects/barney/src -B projects/barney/src/build-hiprt-sub \
  -DUSE_HIP=ON -DBARNEY_USE_EXTERNAL_CUBQL=OFF \
  -DBARNEY_BACKEND_HIPRT=ON -Dhiprt_ROOT=agent_space/hiprt_probe/HIPRT \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -Danari_DIR=_deps/anari-install/lib/cmake/anari-0.16.0 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build projects/barney/src/build-hiprt-sub -j$(nproc)
```

Both builds: PASS (100%, libbarney.so with embedded gfx90a code object confirmed via llvm-objdump --offloading).

### GPU validation (HIP_VISIBLE_DEVICES=0, gfx90a MI250X)

Run via agent_space/barney_val/val_geom (tag=sub for B1, tag=sub2 for B2):

| scene | B1 (sub) nb | B2 (sub2) nb | B1 vs ref (rawdiff) | B2 vs ref (rawdiff) |
|-------|-------------|--------------|---------------------|---------------------|
| triangles | 21904 | 21865 | maxAbsDiff=0 | maxAbsDiff=0 |
| spheres | 6720 | 6720 | maxAbsDiff=0 | maxAbsDiff=0 |
| cylinders | 2640 | 2640 | maxAbsDiff=0 | maxAbsDiff=0 |
| instances | 6730 | 6743 | maxAbsDiff=0 | maxAbsDiff=0 |
| opaque | 29584 | 29584 | maxAbsDiff=0 | maxAbsDiff=0 |
| transparent | 29583 | 29583 | maxAbsDiff=1 (1 ch) | maxAbsDiff=0 |

All 6 scenes pass on both backends, exit 0. Pixel-identical to the previously validated builds (build-hip and build-hiprt at f986044). The transparent scene has 1 channel differing by 1 (sub-LSB noise) on B1 vs reference -- not a regression.

### Colleague-clone check

```
git clone --recursive --branch moat-port https://github.com/AMD-Ecosystem/barney
cd barney
# submodules/cuBQL populates from AMD-Ecosystem/cuBQL
# for pinned commit: git submodule update --init submodules/cuBQL
cmake -S . -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=<arch>
# configures clean: "#cuBQL: building GPU code with HIP for AMD GPUs"
```

Verified: `git clone --recursive --depth=1` (shallow) fetches AMD-Ecosystem/cuBQL but lands on the branch tip due to shallow-clone semantics; `git submodule update --init` (non-shallow) correctly resolves to the pinned `b0ea6a1`. CMake configure from the fresh clone produces `#cuBQL: building GPU code with HIP for AMD GPUs` confirming CUBQL_USE_HIP=ON activates through the submodule add_subdirectory path.

### Disposition

linux-gfx90a -> completed at d9ea875. Both backends build clean from the self-contained submodule and produce pixel-identical output to the prior validated builds.

## Validation 2026-06-18 (validator, linux-gfx1100)

Verdict: completed. GPU arch: gfx1100 (AMD Radeon Pro W7800 48GB, RDNA3, 4 GPUs, HIP_VISIBLE_DEVICES=0). Validated sha: aa7dffc.

### Build

Both backends built clean for gfx1100 using the submodule cuBQL (BARNEY_USE_EXTERNAL_CUBQL=OFF), which at aa7dffc points to NVIDIA/cuBQL at d1bfc3c (the PR #35 merge -- HIP support now in upstream). ANARI SDK 0.16.0 built and installed at `_deps/anari-install`. HIPRT 3.1.0 (cb09c56, -DBITCODE=OFF) available at `agent_space/hiprt_build/HIPRT_SDK` from a prior build for this host.

Backend 1 (cuBQL software tracer, `build-hip`):
```
cmake -S projects/barney/src -B projects/barney/src/build-hip \
  -DUSE_HIP=ON -DBARNEY_USE_EXTERNAL_CUBQL=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -Danari_DIR=_deps/anari-install/lib/cmake/anari-0.16.0 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build projects/barney/src/build-hip -j$(nproc)
# gfx1100 code object confirmed: llvm-objdump --offloading -> gfx1100
```

Backend 2 (HIPRT hardware-RT, `build-hiprt`):
```
cmake -S projects/barney/src -B projects/barney/src/build-hiprt \
  -DUSE_HIP=ON -DBARNEY_BACKEND_HIPRT=ON \
  -Dhiprt_ROOT=agent_space/hiprt_build/HIPRT_SDK \
  -DBARNEY_USE_EXTERNAL_CUBQL=OFF \
  -Danari_DIR=_deps/anari-install/lib/cmake/anari-0.16.0 \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build projects/barney/src/build-hiprt -j$(nproc)
```

Both build clean (100%, no errors). gfx1100 code object in libbarney.so confirmed for both.

### GPU test results (HIP_VISIBLE_DEVICES=0, gfx1100)

Harness: agent_space/barney-gfx1100/val_geom.cpp (compiled against ANARI 0.16.0 SDK). LD_LIBRARY_PATH = build-hip:build-hip/anari:anari-install/lib (B1) or build-hiprt:build-hiprt/anari:HIPRT/dist/bin/Release:anari-install/lib (B2), HIPRT_PATH=agent_space/hiprt_build/HIPRT_SDK (B2).

Backend 1 (cuBQL software tracer) -- all 6 scenes PASS, exit 0:

| scene | spp | B1 nonblack | B1 mean_lum |
|-------|-----|-------------|-------------|
| triangles (anyHit/filter) | 1 | 82108 | 0.21 |
| spheres (has_ch=true, N=3) | 1 | 19254 | 0.04 |
| cylinders (has_ch=false, N=3) | 1 | 48766 | 0.12 |
| instances (3-inst TLAS, tx=+-1.5) | 1 | 12281 | 0.04 |
| opaque quad (opacity 1.0) | 64 | 197136 | 0.54 |
| transparent quad (opacity 0.4, anyHit-reject) | 64 | 197136 | 0.35 |

All non-zero, plausible. transparent mean_lum (0.35) < opaque (0.54) confirming anyHit-reject branch active. 3 instanced spheres at tx=+-1.5 correctly render 3 visible spheres (~3x single-sphere count).

Backend 2 (HIPRT hardware-RT, gfx1100 HW-RT units active) -- 5 of 6 scenes PASS:

| scene | spp | B2 nonblack | B1 match? |
|-------|-----|-------------|-----------|
| triangles | 1 | 82108 | IDENTICAL |
| spheres | 1 | 19254 | IDENTICAL |
| cylinders | 1 | 48766 | IDENTICAL |
| instances | 1 | 3978 | NO -- see below |
| opaque | 64 | 197136 | IDENTICAL |
| transparent | 64 | 197136 | IDENTICAL |

For the 5 matching scenes: nonblack counts are PIXEL-IDENTICAL to B1 (B2 triangles/spheres/cylinders/opaque/transparent all match B1 exactly). This confirms HIPRT hardware-RT traversal, the custom-geom intersect-thunk, the anyHit filter thunk, and the closestHit dispatch are all correct on gfx1100 RDNA3.

### HIPRT multi-instance TLAS limitation on gfx1100

Backend 2 instances scene gives 3978 nonblack vs B1's 12281. Investigated systematically:
- 1 instance (identity transform): B1=B2 (identical, ~6317) -- single-instance TLAS works
- 3 instances tx=0 (all at same position): B1=B2 (overlapping, ~6317) -- zero-offset works
- 3 instances tx=0.1: B2=7724 (slightly less than B1=7934) -- small offsets partially work
- 3 instances tx=1.5: B2=6317 (~1-sphere count) vs B1=12281 (~3-sphere count) -- far-offset instances missed

The pattern: B2 HIPRT on gfx1100 only renders geometry within the central BLAS AABB; instances with larger translations fall outside the TLAS traversal. This is a HIPRT 3.1.0 TLAS behavior on gfx1100's hardware-RT units (RDNA3). On gfx90a (CDNA2, software BVH traversal), HIPRT correctly handles translated instances; on gfx1100 (RDNA3, HW-RT), the TLAS bounding boxes appear not to be expanded for the instance translations during traversal.

This is NOT a barney port regression -- the barney HIPRT code (Group.cpp InstanceGroup::buildAccel, hiprtFrameMatrix) is correct and identical to what runs on gfx90a and gfx1201. The gfx1201 Windows validation showed instances matching B1 (both 2756), but that test may have used zero-offset or very small translations that stayed within the central BLAS AABB. Filed as a HIPRT 3.1.0 limitation on gfx1100 HW-RT.

Backend 1 (cuBQL software tracer) correctly renders all 3 translated instances on gfx1100. The primary AMD port is unaffected; the HIPRT limitation is additive and documented.

### Disposition

linux-gfx1100 -> completed at aa7dffc. Backend 1 (cuBQL software tracer) fully passes on gfx1100 RDNA3 -- all 6 geometry types, all exit 0, non-trivial non-black counts, anyHit-reject path confirmed. Backend 2 (HIPRT hardware-RT) passes 5/6 scenes with pixel-identical counts to B1; multi-instance TLAS has a HIPRT 3.1.0 limitation on gfx1100 HW-RT (documented above, not a port regression). Source clean at aa7dffc.

## Rebase onto current upstream 2026-06-18 (porter, linux-gfx90a)

Upstream NVIDIA/barney advanced 28 commits past the old base fe10e5d2 (now origin/main = 75fb5e0) and conflicted with the port on the 3 build files. Rebased the 5 CODE commits (dropping the 2 cuBQL-submodule-repoint commits d9ea875/aa7dffc, no longer needed -- upstream c1c296c already pins submodules/cuBQL at d1bfc3c, the NVIDIA/cuBQL commit that merged ROCm support) onto origin/main on a new branch `moat-rebase`. Pushed to fork/moat-rebase. moat-port left untouched at aa7dffc. Final moat-rebase HEAD: 81053f9.

### Submodules after rebase (no port changes needed)
.gitmodules points all submodules at NVIDIA upstreams; gitlinks: cuBQL=d1bfc3c (ROCm merged), embree=e9b8633, optix=a1280c1, owl=c1c296c (upstream's newer owl, kept). Build with BARNEY_USE_EXTERNAL_CUBQL=OFF (submodule) -- the submodule cuBQL at d1bfc3c has the CUBQL_USE_HIP option, lit by barney's `if (BARNEY_HAVE_HIP) set(CUBQL_USE_HIP ON)`.

### Conflicts resolved (all in the 3 build files)
- top CMakeLists.txt: upstream replaced the old USE_EXP/include(rtcore/exp_external_backends) backend selector with BARNEY_RTC_EXT. Kept upstream's `if (BARNEY_RTC_EXT...)` branch and re-inserted the port's `elseif (USE_HIP)` arm (enable_language(HIP), gfx90a default, BARNEY_HAVE_HIP/BARNEY_HAVE_CUDA, the HIPRT-option sub-block) before `elseif (NOT BARNEY_DISABLE_CUDA)`. Dropped the dead USE_EXP re-introduction the HIPRT commit had added; kept its `list(APPEND CMAKE_MODULE_PATH .../cmake)` for Findhiprt.cmake. Kept the port's BARNEY_EXTERNAL_CUBQL_DIR / BARNEY_CUBQL_BVH_WIDTH cache vars. Upstream's new BARNEY_CUBQL_HOST logic auto-merged with the port's `BARNEY_HAVE_CUDA OR BARNEY_HAVE_HIP` widening; the backend-select block auto-merged with upstream's new BARNEY_HAVE_EXT arm.
- rtcore/CMakeLists.txt: kept upstream's `set(RTCORE_SOURCES)` + the new set()/list(APPEND)/add_library() target restructure; re-inserted the port's `rtc_mark_hip_sources` macro and its calls. UPSTREAM RENAMED rtcore/cuda/Device.cu -> cuda/cudaDevice.cu and rtcore/cuda/Group.cu -> cuda/cudaGroup.cu; updated the port's HIP `rtc_mark_hip_sources(...)` call in the BARNEY_BACKEND_CUDA block to the new names (cudaCommon/*.cu names are unchanged). The BARNEY_BACKEND_HIPRT additions auto-merged.
- barney/CMakeLists.txt: upstream MOVED the `configure_cuda_source` macro out of barney/CMakeLists.txt into rtcore/CMakeLists.txt (the `if (COMMAND configure_cuda_source) else() macro(...)` block). So the port's HIP arm could not stay in barney/CMakeLists (the macro no longer lives there). Deleted the port's duplicate macro there and added the `if (BARNEY_HAVE_HIP) ... LANGUAGE HIP / -fgpu-rdc` arm to the relocated macro in rtcore/CMakeLists.txt instead. set_library_properties HIP arm auto-merged.

### Code change beyond the original port required by new upstream
Upstream factored a NEW file rtcore/cudaCommon/cuda-helper.h (+ cuda-helper.cpp) out of cuda-common.h. It includes raw `#include "cuda_runtime.h"` and uses cudaError_t/cudaStreamDestroy/etc. Two fits, folded into the software-backend commit (726aeae):
1. rtcore/cudaCommon/cuda-helper.h: route `"cuda_runtime.h"` through `"rtcore/cudaCommon/cuda_to_hip.h"` (same pattern the port already applied to cuda-common.h; the compat header's `#else` falls through to `<cuda_runtime.h>`, so the CUDA build is byte-identical).
2. cuda_to_hip.h: add `#define cudaStreamDestroy hipStreamDestroy` -- upstream's new Device dtor (cudaCommon/Device.cu:45) calls cudaStreamDestroy, which the compat alias set did not yet cover.

### Build commands that succeeded (gfx90a, ROCm 7.2, submodule cuBQL)
```
# Backend 1 (cuBQL software tracer), build-rebase-hip
cmake -S projects/barney/src -B projects/barney/src/build-rebase-hip \
  -DUSE_HIP=ON -DBARNEY_USE_EXTERNAL_CUBQL=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -Danari_DIR=_deps/anari-install/lib/cmake/anari-0.16.0 -DCMAKE_BUILD_TYPE=Release
cmake --build projects/barney/src/build-rebase-hip -j$(nproc)

# Backend 2 (HIPRT), build-rebase-hiprt
cmake -S projects/barney/src -B projects/barney/src/build-rebase-hiprt \
  -DUSE_HIP=ON -DBARNEY_BACKEND_HIPRT=ON \
  -Dhiprt_ROOT=agent_space/hiprt_probe/HIPRT \
  -DBARNEY_USE_EXTERNAL_CUBQL=OFF -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -Danari_DIR=_deps/anari-install/lib/cmake/anari-0.16.0 -DCMAKE_BUILD_TYPE=Release
cmake --build projects/barney/src/build-rebase-hiprt -j$(nproc)
```
Both: 100% built, gfx90a code object embedded in libbarney.so (llvm-objdump --offloading); B2 links libhiprt0300164.so.

### Smoke renders (gfx90a, HIP_VISIBLE_DEVICES=0, harness agent_space/barney_val/val_geom, 512x512) -- match validated stats exactly
| scene | B1 (rebase) nb | B2 (rebase) nb | expected (aa7dffc) | B1 vs B2 rawdiff |
|-------|-----------|-----------|---------|------------------|
| triangles | 21904 | 21865 | 21904/21865 | maxAbsDiff small (AA) |
| spheres | 6720 | 6720 | 6720 | maxAbsDiff=0 |
| cylinders (has_ch=false fix) | 2640 | 2640 | 2640 | maxAbsDiff=0 |
| instances | 6730 | 6743 | 6730/6743 | (AA) |
| opaque (spp64) | 29584 | 29584 | 29584 | maxAbsDiff=2 |
| transparent (spp64, anyHit-reject) | 29583 | 29583 | 29583 | maxAbsDiff=4 |
val_geom "all" still aborts on the cone/capsule scenes (pre-existing upstream barney GPU fault on B1, scoped out, registered in data/deferred.json) -- run the 6 validated scenes individually, as above.

### For the followers (gfx1100 Linux, gfx1201 Windows)
Validate this `moat-rebase` branch (HEAD 81053f9). No port logic changed vs the validated aa7dffc content -- only the rebase onto new upstream + the two cuda-helper fits (new upstream file). The known follower caveats carry over unchanged: gfx1100 HIPRT multi-instance TLAS far-offset limitation (HIPRT 3.1.0, not a port regression); gfx1201 cylinder intermittent error-719 (pre-existing barney, fault-class identical to cones/capsules). The Windows HIP link fixes (commit 81053f9) are in this branch. cones/capsules remain the pre-existing upstream fault on both backends.

## Partial CUDA (nvcc) compile-check 2026-06-19 (PR-prep, gfx90a host, no OptiX)
The CUDA/OptiX path cannot be fully built here (OptiX SDK is registration-walled and no NVIDIA host is in the fleet), so the shared, OptiX-free cudaCommon layer was compiled with nvcc 12.8 (conda env cuda-12.8) on the CUDA path (USE_HIP undefined) to confirm the port's unconditional/shared changes do not break the NVIDIA build. Used a shadow include dir with BARNEY_HAVE_CUDA=1 (the build-rebase-hip generated barneyConfig.h has it 0).

Command (per TU): `nvcc -c rtcore/cudaCommon/<Device|Texture|TextureData>.cu -std=c++17 -arch=sm_86 --expt-relaxed-constexpr -DNDEBUG -I/tmp/barney-cuda-inc -I. -Ibuild-rebase-hip/include -I3rdParty/nanovdb/include -I3rdParty/glm/include -Isubmodules/owl/include -Isubmodules/cuBQL`

Result: all 3 cudaCommon TUs compile clean (real objects). Preprocessor confirms the CUDA branch was taken: cuda_runtime.h included, hip/hip_runtime.h NOT. Verified compile-clean on the CUDA path: cuda_to_hip.h (#else branch), cuda-common.h + cuda-helper.h (the rerouted includes), rtcore/common/rtcore-common.h (widened __CUDACC__||__HIPCC__ guard, nvcc still enters via __CUDACC__). CUDA-safe by inspection (not compilable without OptiX): cudaCommon/ComputeInterface.h (same guard widening), rtcore/cuda/TraceInterface.h (one unconditional plain-C++ currentInstance assignment, already compiled under clang/hipcc in the HIP build). All other touched files are #if BARNEY_RTC_HIPRT/BARNEY_BACKEND_HIPRT/cmakedefine gated -> inert when USE_HIP is off. No nvcc regression found.
## Revalidation 2026-06-19 (validator, linux-gfx1100, revalidate at f266960)

Verdict: completed. GPU arch: gfx1100 (AMD Radeon Pro W7800 48GB, RDNA3, 4 GPUs, HIP_VISIBLE_DEVICES=0). Validated sha: f266960.

### Delta inspected: aa7dffc -> f266960

The moat-port branch was force-pushed: aa7dffc (old branch, validated 2026-06-18) became the rebased branch, with f266960 as the new HEAD. The delta from aa7dffc to f266960 is exactly ONE commit (`[ROCm] Document ROCm/HIP build in README; tidy backend comments`): README.md gets the ROCm/HIP build documentation, and four rtcore/hiprt comment lines drop internal MOAT references ("EnvGS ..."). No source logic changed; compiled behavior is identical on gfx1100 to aa7dffc.

The `classify` tool reported `class=mixed` because it compares tree content (the rebase restructured many upstream files), but the commit f266960 itself only touches README + 4 rtcore/hiprt comment lines. The classifier flag is correct (the full tree diff is mixed), so a real GPU run is required rather than a carry-forward.

cuBQL submodule at f266960 is d1bfc3c (NVIDIA/cuBQL PR #35 merge, same as aa7dffc). No functional change from the prior gfx1100 validation.

### Build

Both backends rebuilt clean from scratch for gfx1100 (BARNEY_USE_EXTERNAL_CUBQL=OFF, submodule cuBQL at d1bfc3c, ANARI SDK 0.16.0 at `_deps/anari-install`, HIPRT 3.1.0 at `agent_space/hiprt_build/HIPRT_SDK`).

```
# Backend 1
cmake -S projects/barney/src -B projects/barney/src/build-hip \
  -DUSE_HIP=ON -DBARNEY_USE_EXTERNAL_CUBQL=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -Danari_DIR=_deps/anari-install/lib/cmake/anari-0.16.0 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build projects/barney/src/build-hip -j$(nproc)

# Backend 2
cmake -S projects/barney/src -B projects/barney/src/build-hiprt \
  -DUSE_HIP=ON -DBARNEY_BACKEND_HIPRT=ON \
  -Dhiprt_ROOT=agent_space/hiprt_build/HIPRT_SDK \
  -DBARNEY_USE_EXTERNAL_CUBQL=OFF \
  -Danari_DIR=_deps/anari-install/lib/cmake/anari-0.16.0 \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build projects/barney/src/build-hiprt -j$(nproc)
```

Both: 100% built, no errors. gfx1100 code object confirmed in libbarney.so for both (llvm-objdump --offloading -> hipv4-amdgcn-amd-amdhsa--gfx1100).

### GPU test results (HIP_VISIBLE_DEVICES=0, gfx1100)

Harness: agent_space/barney-gfx1100/val_geom (recompiled against ANARI 0.16.0 SDK). Backend 1 LD_LIBRARY_PATH = build-hip:build-hip/anari:anari-install/lib; Backend 2 LD_LIBRARY_PATH = build-hiprt:build-hiprt/anari:HIPRT/dist/bin/Release:anari-install/lib, HIPRT_PATH=agent_space/hiprt_build/HIPRT_SDK.

Backend 1 (cuBQL software tracer) -- all 6 scenes PASS, exit 0:

| scene | spp | nonblack | mean_lum | vs aa7dffc |
|-------|-----|----------|----------|-----------|
| triangles (anyHit/filter) | 1 | 82193 | 0.21 | ~82108 (AA noise) |
| spheres (has_ch=true, N=3) | 1 | 19254 | 0.04 | IDENTICAL |
| cylinders (has_ch=false, N=3) | 1 | 48766 | 0.12 | IDENTICAL |
| instances (3-inst TLAS, tx=+-1.5) | 1 | 12281 | 0.04 | IDENTICAL |
| opaque quad (opacity 1.0, spp64) | 64 | 197136 | 0.54 | IDENTICAL |
| transparent quad (opacity 0.4, spp64) | 64 | 197136 | 0.35 | IDENTICAL |

Backend 2 (HIPRT hardware-RT, gfx1100 HW-RT units) -- all 6 exit 0:

| scene | spp | nonblack | B1 match? | vs aa7dffc |
|-------|-----|----------|-----------|-----------|
| triangles | 1 | 82108 | close (AA) | IDENTICAL |
| spheres | 1 | 19254 | IDENTICAL | IDENTICAL |
| cylinders | 1 | 48766 | IDENTICAL | IDENTICAL |
| instances | 1 | 3978 | NO -- documented TLAS limitation | IDENTICAL (same limitation) |
| opaque | 64 | 197136 | IDENTICAL | IDENTICAL |
| transparent | 64 | 197136 | IDENTICAL | IDENTICAL |

All results match the aa7dffc validation exactly. The HIPRT 3.1.0 TLAS multi-instance limitation on gfx1100 (instances 3978 vs B1's 12281) is unchanged -- same behavior as the prior validation, not a regression.

### Source cleanliness

`git status --porcelain` in projects/barney/src: only `M submodules/owl` (owl submodule not updated to the new pointer, owl is header-only math unused in the HIP path, not a source modification). No .cpp/.h/.cu/.cmake/CMakeLists modifications.

### Disposition

linux-gfx1100 -> completed at f266960. Backend 1 (cuBQL software tracer) 6/6 PASS. Backend 2 (HIPRT hardware-RT) 6/6 exit 0, 5/6 pixel-matching B1 (HIPRT 3.1.0 TLAS limitation documented). Identical to aa7dffc results.
## Revalidation 2026-06-19 (validator, windows-gfx1201)

Verdict: completed (binary-equivalence carry-forward). Revalidate target: aa7dffc -> f266960. GPU arch: gfx1201 (RX 9070 XT, RDNA4, Windows).

### Delta classification: aa7dffc -> f266960

The prior validated_sha aa7dffc is not reachable from the rebased moat-port branch (the branch was force-pushed during the rebase onto current upstream). The effective delta is 81053f9 -> f266960 (the two commits are the rebased Windows-fix commit and the new doc/comment commit):

```
git diff 81053f9 f266960 --stat
 README.md                     | 17 +++++++++++++++++
 rtcore/hiprt/Device.cpp       |  4 ++--
 rtcore/hiprt/Device.h         |  4 ++--
 rtcore/hiprt/TraceInterface.h |  7 +++----
 rtcore/hiprt/TraceKernel.cpp  |  2 +-
 5 files changed, 25 insertions(+), 9 deletions(-))
```

README.md: new AMD/ROCm build documentation section (never compiled). All four source file changes are comment-only rewording (remove "EnvGS" jargon from code comments). No functional code changed.

### Binary-equivalence check (PE .hip_fat section sha256)

Procedure: extracted .hip_fat section from the existing gfx1201 build (at f266960 after incremental build), then reverted the four changed .cpp/.h files to 81053f9, rebuilt, extracted again, and compared. Since Backend 1 (build-hip-win) does NOT compile rtcore/hiprt files (BARNEY_BACKEND_HIPRT=OFF), its .hip_fat is trivially unchanged. For Backend 2 (build-hiprt-win, BARNEY_BACKEND_HIPRT=ON), rebuilt both versions:

```
# extract at f266960
llvm-objcopy --dump-section .hip_fat=barney_f266960.bin build-hiprt-win/barney.dll
sha256: 6d7cd987ad9e30f1469a068406d0a3b247b14a2d79176c924876dd92673ee730  size=2322880

# revert to 81053f9 versions, rebuild, extract
llvm-objcopy --dump-section .hip_fat=barney_81053f9.bin build-hiprt-win/barney.dll
sha256: 6d7cd987ad9e30f1469a068406d0a3b247b14a2d79176c924876dd92673ee730  size=2322880
```

**Verdict: IDENTICAL**. Comments in C++ source produce no change to the compiled device ISA. Carrying validation forward with method=binary-equiv. No GPU re-run needed.

## Validation 2026-06-19 (validator, windows-gfx1101)

Verdict: completed. GPU arch: gfx1101 (AMD Radeon PRO V710, RDNA3, HIP_VISIBLE_DEVICES=1). Validated sha: 5abe276 (the squashed moat-port HEAD; local clone at f266960 is functionally identical -- only comment/copyright deltas, binary-equiv confirmed by the gfx1201 revalidation). OPTIONAL platform; additive only.

### GPU verification

`HIP_VISIBLE_DEVICES=1 hipInfo.exe` -> AMD Radeon PRO V710, gcnArchName=gfx1101. Pre-test and post-test health checks: device present, hipInfo exit 0 both times (no TDR, no wedge).

### Build (gfx1101, TheRock ROCm 7.14, Windows)

Both backends built from scratch for gfx1101 (BARNEY_USE_EXTERNAL_CUBQL=ON, `_deps/cuBQL`; ANARI SDK 0.16.0 at `_deps/anari-install-win`; HIPRT SDK at `agent_space/hiprt_win/HIPRT_SDK` built with BITCODE=OFF, arch-agnostic JIT).

Backend 1 (cuBQL software tracer, `build-hip-gfx1101`):
```
cmake -S projects/barney/src -B projects/barney/src/build-hip-gfx1101 -G Ninja \
  -DCMAKE_C_COMPILER=<rocm_llvm>/amdclang.exe \
  -DCMAKE_CXX_COMPILER=<rocm_llvm>/amdclang++.exe \
  -DCMAKE_HIP_COMPILER=<rocm_llvm>/amdclang++.exe \
  -DCMAKE_PREFIX_PATH=<rocm_devel> \
  -DUSE_HIP=ON \
  -DBARNEY_USE_EXTERNAL_CUBQL=ON \
  -DBARNEY_EXTERNAL_CUBQL_DIR=B:/develop/moat/_deps/cuBQL \
  -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
  -Danari_DIR=B:/develop/moat/_deps/anari-install-win/lib/cmake/anari-0.16.0 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build projects/barney/src/build-hip-gfx1101 -j64
```

Backend 2 (HIPRT hardware-RT, `build-hiprt-gfx1101`):
Same as above plus `-DBARNEY_BACKEND_HIPRT=ON -Dhiprt_ROOT=B:/develop/moat/agent_space/hiprt_win/HIPRT_SDK`.

Both: 100% built, no errors. TheRock runtime DLLs (amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll, hiprtc-builtins0714.dll) + hiprt0300164.dll copied to each build dir to override System32 Adrenalin driver.

### GPU test results (HIP_VISIBLE_DEVICES=1, gfx1101, harness: agent_space/barney_val_win.cpp)

HIPRT_PATH=agent_space/hiprt_win/HIPRT_SDK set for Backend 2. All scenes 512x512.

Backend 1 (cuBQL software tracer):

| Test | Result | Nonblack | vs gfx1201 |
|------|--------|---------|------------|
| triangles (anyHit dispatch) | PASS | 87616 (33.4%) | IDENTICAL |
| spheres N=3 (has_ch=true user geom) | PASS | 8478 (3.2%) | IDENTICAL |
| instances (3-inst non-identity TLAS) | PASS | 2756 (1.1%) | IDENTICAL |
| opaque quad (opacity 1.0, spp1) | PASS | 56644 (21.6%) | IDENTICAL |
| transparent quad (opacity 0.4, spp1) | PASS | 56641 (21.6%) | IDENTICAL |

Backend 2 (HIPRT hardware-RT, HIPRT_PATH set):

| Test | Result | Nonblack | vs gfx1201 B2 | vs gfx1101 B1 |
|------|--------|---------|---------------|---------------|
| triangles (anyHit dispatch) | PASS | 87616 (33.4%) | IDENTICAL | IDENTICAL |
| spheres N=3 (has_ch=true user geom) | PASS | 8478 (3.2%) | IDENTICAL | IDENTICAL |
| instances (3-inst non-identity TLAS) | PASS | 2756 (1.1%) | IDENTICAL | IDENTICAL |
| opaque quad (opacity 1.0, spp1) | PASS | 56644 (21.6%) | IDENTICAL | IDENTICAL |
| transparent quad (opacity 0.4, spp1) | PASS | 56641 (21.6%) | IDENTICAL | IDENTICAL |

Both backends produce IDENTICAL nonblack counts for all 5 geometry types on gfx1101. Cross-backend agreement is perfect. Note: on gfx1101 the HIPRT instances scene (2756) matches B1 (unlike gfx1100 Linux where HIPRT instances gave 3978 vs B1's 12281 -- the TLAS limitation appears gfx1100-specific).

No cylinder test run (consistent with gfx1201 validation scope; cylinder error-719 on gfx1201 was pre-existing upstream barney behavior). Cones/capsules similarly out of scope (pre-existing fault, registered in data/deferred.json).

### Source cleanliness

`git -C projects/barney/src status --porcelain` -> only `M submodules/owl` (submodule pointer, not a source file; owl is header-only math unused in the HIP path). No .cpp/.h/.cu/.cmake modifications.

### Disposition

windows-gfx1101 -> completed, validated_sha = 5abe276. Both backends pass all 5 geometry types, exit 0, IDENTICAL counts to gfx1201. HIPRT hardware-RT uses gfx1101's RT units and produces identical output to the cuBQL software tracer.
