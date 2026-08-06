# HIPRT feasibility probe -- gfx90a / ROCm 7.2.1

Scoping report. Not a port. No `projects/` fork, status.json, or upstream artifact was
touched. Purpose: decide later whether to reimplement our OptiX-gated ports on AMD's
HIPRT (HIP Ray Tracing), or keep deferring them.

Host: AMD Instinct MI250X / MI250, gfx90a:sramecc+:xnack-, ROCm 7.2.1, HIP 7.2.53211,
hipcc (AMD clang 22). GPU pinned to `HIP_VISIBLE_DEVICES=0`.

Naming trap (resolved): HIPRT (ray tracing SDK, NOT in ROCm, fetched from GPUOpen) is
distinct from hiprtc (runtime compilation, `/opt/rocm/lib/libhiprtc.so`, already
installed). HIPRT *uses* hiprtc under the hood to JIT its traversal kernels.

---

## (a) HIPRT on gfx90a / ROCm 7.2.1: BUILDS = YES, RUNS = YES

### Fetch
- Source: `github.com/GPUOpen-LibrariesAndSDKs/HIPRT`, latest release tag
  **3.1.0.cb09c56** (Jan 2026; this checkout HEAD e3c01fc, version.txt 3.1.0.cb09c56).
- The GPUOpen prebuilt-binary download (gpuopen.com/hiprt) is gated behind a clickwrap
  license and returns HTTP 400 to a direct fetch; it is also Windows-Adrenalin / older
  (web page advertises v3.0). Building from source was simpler and current, so that is
  what was validated.
- **Orochi is vendored** in-tree at `contrib/Orochi` (NOT a git submodule -- `.gitmodules`
  is empty in the release). So "needs Orochi submodule" is a non-issue: a plain
  `git clone` (no `--recursive`) has everything. embree/zstd/gtest are also vendored.
- The SDK demos/tutorials live in a *separate* repo, `GPUOpen-LibrariesAndSDKs/HIPRTSDK`
  (older, last touched 2025-09); they require the GPUOpen prebuilt package. Not needed.

### gfx90a support is explicit in HIPRT source
- `scripts/bitcodes/common_tools.py:56` lists `gfx90a` (and `gfx942`) in the precompiled
  arch set; `hiprt/hiprt_common.h:202` includes `__gfx90a__` in the CDNA/Vega macro arm.
- gfx90a (CDNA2) has **no hardware RT units**, so HIPRT selects its **software BVH
  traversal** path: `HIPRT_RTIP = 0` (vs 11/20/31 for RDNA2/3/4), `WarpSize = 64`,
  `BranchingFactor = 4`. This is a supported configuration, not a hack -- GPUOpen lists
  even RDNA1 "without hardware acceleration" as supported; CDNA follows the same software
  path. Traversal correctness does not depend on RT hardware.

### Build (5 s)
```
git clone --depth 1 https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT.git
cd HIPRT
export HIP_PATH=/opt/rocm
cmake -DCMAKE_BUILD_TYPE=Release -DBITCODE=OFF -DNO_UNITTEST=ON -DHIP_PATH=/opt/rocm -S . -B build
cmake --build build --target hiprt03001 -j 16     # -> dist/bin/Release/libhiprt0300164.so
```
The `64` suffix = wave64 build. CUDA auto-disables cleanly (no CUDA on the host -- just a
warning). The host library is arch-agnostic; nothing gfx-specific is baked. With
`BITCODE=OFF` (the default, HIPRT-team-tested path) the BVH-builder and traversal *device*
kernels are **JIT-compiled at runtime via Orochi -> hiprtc** from kernel source shipped in
the tree, then cached. (`BITCODE=ON` ships precompiled bitcode instead; not needed here.)

### Runtime requirements discovered
1. `HIPRT_PATH` env var must point at the HIPRT source root so the runtime can find its
   device-kernel `.h` sources to JIT (`Utility::getRootDir()` reads `HIPRT_PATH`, else
   falls back to `..`). A real port would either set this or build with
   `-DBITCODE=ON`/baked kernels to embed them.
2. **A real HIPRT bug surfaced on MI250X** (worth recording for any future port):
   `Compiler::getCacheFilename` (hiprt/impl/Compiler.cpp) builds the JIT cache filename
   from the raw device name, which on this card is `"AMD Instinct MI250X / MI250"`. The
   embedded `/` is interpreted by `std::filesystem::path` as a directory separator, so the
   cache file lands in a non-existent subdir and `cacheBinaryToFile` (which does not
   `create_directories`) throws "Unable to open ...bin". One-line fix (sanitize `/` and
   `\\` to `_` in the device name before using it as a path component) was applied locally
   to validate; it is a clean upstream candidate. Without it, HIPRT cannot cache or run
   on any "/"-containing AMD device name.

### Minimal trace test: PASS
A self-contained test (`agent_space/hiprt_probe/HIPRT/minimal/`) builds a one-triangle
geometry BVH (`hiprtCreateGeometry` + `hiprtBuildGeometry`, `hiprtTriangleMeshPrimitive`),
JIT-compiles a raygen kernel calling `hiprtGeomTraversalClosest::getNextHit()`
(`hiprtBuildTraceKernels`), launches it on gfx90a, and checks two rays:

```
Device: AMD Instinct MI250X / MI250  (gcnArch=gfx90a:sramecc+:xnack-)
Geometry BVH built (1 triangle).
Raygen kernel JIT-compiled for gfx90a:sramecc+:xnack-.
Ray0 (aimed at triangle): primID=0  t=1.0000  (expect primID=0, t~1.0)
Ray1 (aimed at empty):    primID=4294967295  (expect primID=invalid)
VERDICT: PASS -- HIPRT trace correct on this GPU
```
Cold-cache and warm-cache (0.63 s) reruns both PASS. The hit reports the correct primId
and exact t=1.0; the miss correctly reports the invalid sentinel. **HIPRT's software
ray-triangle traversal is functional and correct on gfx90a/ROCm 7.2.1.**

Two test-author gotchas found (not HIPRT bugs, but a porter will hit them):
- `hiprtHit` is `alignas(16)`, sizeof 48: `instanceID` is a 16-byte union (instanceIDs),
  so `primID` is at offset 16, `t` at offset 44 (NOT 4 / 28). Use the real struct, not a
  guessed mirror.
- For a kernel with **no** custom intersection/filter functors, call
  `hiprtBuildTraceKernels` with `numGeomTypes=0, numRayTypes=0`. Passing `1,1` with a null
  `funcNameSets` makes hiprt.cpp build an empty vector that `getCacheFilename` then indexes
  out of bounds -> SIGSEGV. (Pass real funcNameSets only when you actually have functors.)
- Synchronize with `oroStreamSynchronize(0)`, not `oroCtxSynchronize()` -- the latter maps
  to the deprecated `hipCtxSynchronize` which returns hipErrorNotSupported (801) here.

---

## (b) HIPRT vs OptiX API-model mapping

OptiX and HIPRT are both "build an accel structure, trace rays against it on the GPU", but
the programming model differs in one decisive way: **OptiX is a multi-shader pipeline with
a Shader Binding Table; HIPRT is a single ordinary HIP kernel that calls a traversal object
and supplies behavior via device functors.** There are no separate compiled shader programs
and no SBT in HIPRT.

| OptiX concept | HIPRT equivalent | Maps cleanly? |
|---|---|---|
| `optixDeviceContextCreate` | `hiprtCreateContext` (over an Orochi/HIP ctx) | Yes |
| `optixAccelBuild` (TRIANGLES GAS) | `hiprtCreateGeometry`+`hiprtBuildGeometry`, `hiprtTriangleMeshPrimitive` | Yes -- validated above |
| Instance acceleration (IAS) | `hiprtScene` (`hiprtCreateScene`/`hiprtBuildScene`, `hiprtInstance`) | Yes |
| `optixModuleCreate` + PTX | `hiprtBuildTraceKernels` (JIT from HIP source via hiprtc) or `...FromBitcode` | Yes, but it is JIT, not an offline PTX blob |
| `optixLaunch` + launch params | ordinary `hipModuleLaunchKernel`/`oroModuleLaunchKernel` of your own `__global__`; params are plain kernel args | Yes |
| `__raygen__` program | the body of your trace kernel (1 HIP kernel) | Yes -- merges into one kernel |
| `optixTrace(...)` | construct a `hiprtGeom/SceneTraversal{Closest,AnyHit}[CustomStack]` and loop `getNextHit()` | Yes |
| `__closesthit__` program | code after `getNextHit()` returns the closest `hiprtHit` (no separate shader) | Yes -- inlined |
| `__miss__` program | the `!hit.hasHit()` branch in your kernel | Yes -- inlined |
| `__anyhit__` program | a device **filter functor** `bool filter(ray,data,payload,hit)` registered in a `hiprtFuncTable`; return true = ignore/continue | Yes -- this is the natural anyhit analog |
| `__intersection__` (custom prim) | a device **intersection functor** `bool intersect(ray,data,payload,hit&)` + `hiprtAABBList` geometry | Yes (custom primitives) |
| Shader Binding Table (SBT) | `hiprtFuncTable` (geomType x rayType grid of {intersect,filter} func names) + your own per-prim data buffers passed as kernel args | **Mostly** -- this is the main restructure (below) |
| `optixGet{Ray,World}*`, `optixGetPrimitiveIndex` | fields of `hiprtRay` / `hiprtHit` (`hit.primID`, `hit.uv`, `hit.normal`, `ray.maxT`...) | Yes |
| ray payload via `optixSetPayload`/registers | a `void* payload` you pass to the traversal ctor; HIPRT hands it to your functors | Yes -- simpler (real pointer, no register packing/`packPointer`) |
| `optixIgnoreIntersection()` | `return true;` from the filter functor | Yes |
| `optixTerminateRay()` | use the AnyHit traversal (terminates at first accepted hit) or stop the getNextHit loop | Yes |

What maps cleanly:
- **Triangle GAS build** is essentially 1:1 and is validated. Where OptiX takes a vertex
  buffer + uint3 index buffer, HIPRT takes the same in `hiprtTriangleMeshPrimitive`.
- **raygen / closesthit / miss collapse into one ordinary HIP kernel.** This is *less*
  code than OptiX, not more: no program-group plumbing, no module/pipeline objects, no
  stack-size computation, no SBT record packing for these.
- **anyhit -> filter functor** is a direct, idiomatic translation. HIPRT's documented
  pattern (filter functor returning bool, registered in a func table) is exactly the
  anyhit "record/accumulate then ignore to continue traversal" idiom our tracers use.
- **payload** is *simpler*: HIPRT passes a real `void*` you control, eliminating OptiX's
  2x32-bit register packing of a pointer (`packPointer`/`getPayload`).

What is hard / needs real work:
- **The SBT -> functor-table restructure.** OptiX dispatches per-hit to a shader chosen by
  SBT offset/stride/instance; HIPRT has no per-hit dispatch beyond the (geomType x rayType)
  functor table and otherwise expects you to branch in-kernel. Ports that lean on multiple
  SBT records / per-instance shaders must flatten that into a func-table + in-kernel logic.
  (Our tracers use a single SBT record, so this is light for them -- see per-port below.)
- **Per-2DGS-disk geometry layout (EnvGS).** Not actually a HIPRT problem: EnvGS already
  pre-tessellates each surfel disk into triangles on the host and builds ONE triangle GAS
  (`OPTIX_BUILD_INPUT_TYPE_TRIANGLES`, trace_surfels.cpp:83). That host tessellation is
  reused verbatim; the GPU side is the plain triangle BVH that is validated here. The disk
  alpha/Gaussian math runs in the anyhit functor off `hit.primID` + barycentrics, identical
  to OptiX.
- **anyhit-for-alpha-blending semantics.** The semantics port (record K-closest into a
  payload buffer, `return true` to continue), but two correctness watch-items: (1) HIPRT
  may invoke the filter functor in a different order / multiplicity than OptiX anyhit, so
  the K-closest **sorted-insertion buffer must stay order-independent** (the EnvGS anyhit
  already sorts by t into a fixed buffer, so it is robust; verify no reliance on call
  order). (2) re-trace-with-advanced-min-t chunking must be re-expressed with HIPRT's
  traversal-state loop / custom stack -- mechanically similar but must be re-validated.
- **Differentiable BACKWARD traversal (EnvGS, and any learned tracer).** OptiX/HIPRT only
  give you the forward hit stream; the gradient w.r.t. surfel parameters is hand-written in
  `backward.cu` and simply re-runs the same traversal accumulating grads. This is HIPRT-
  agnostic device math and ports as ordinary HIP -- but it is the bulk of the line count
  and must be re-validated (finite-difference) on the new backend.
- **JIT vs offline.** OptiX ships PTX; HIPRT JITs HIP at runtime (hiprtc) and caches to
  disk. A port must ship the kernel sources (or baked bitcode), set `HIPRT_PATH`/cache dir,
  and carry the cache-name fix. First-run latency is one-time per (kernel, arch, driver).
- **No hardware RT on CDNA.** Functionally fine (software traversal is correct), but
  perf is software-BVH, not RT-core. For research correctness/training this is acceptable;
  for real-time inference it is a throughput consideration, not a correctness one.

---

## (c) Per-port reimplementation effort

### EnvGS Stage 2 (diff-surfel-tracing reflection path) -- effort: MEDIUM
- OptiX surface is small and favorable: only `__raygen__ot` + `__anyhit__ot` (no
  closesthit/miss/intersection programs), a **single** SBT record, one triangle GAS, 6
  `optixTrace` sites, payload-pointer pattern. Files: forward.cu (660), backward.cu (1209),
  optix_wrapper.cpp (233), trace_surfels.cpp (471).
- Clean mappings dominate: triangle GAS (validated), raygen->kernel, anyhit->filter functor
  (the existing anyhit is already a t-sorted K-closest insert + ignore = textbook HIPRT
  filter), single SBT record (minimal func-table), real-pointer payload (drop packPointer).
- The medium (not small) rating is driven by: (1) the **differentiable backward** (~1200
  LOC) must be re-expressed on HIPRT traversal and re-validated by finite-difference, the
  same bar Stage 1 met; (2) the chunked re-trace / transmittance-threshold loop must move to
  HIPRT's traversal-state / custom-stack model; (3) keep `diff_surfel_tracing/__init__.py`
  autograd API + `ext.cpp` bindings byte-stable so the already-ported Stage-1 rasterizer and
  the EnvGS sampler import unchanged; (4) wire HIPRT into the torch extension build (link
  libhiprt + Orochi, JIT cache path, the cache-name fix) -- new build surface for a project
  whose Stage 1 was a pure torch hipify.
- The host disk-tessellation and per-ray surfel-intersection + alpha-compositing math are
  reused as-is. No CUTLASS/CuTe, no warp primitives (per EnvGS notes: zero `__shfl`/ballot;
  serial per-ray compositing + atomicAdd), so wave64 is moot. This is the strongest HIPRT
  candidate of the three: a contained, single-pipeline, single-GAS tracer with an anyhit
  that already matches HIPRT's functor model.

### rmcl (via rmagine OptiX backend) -- effort: LARGE
- The OptiX dependency is NOT in rmcl proper; it lives in the external **rmagine** library
  (`rmagine_optix`, ~6700 LOC). rmcl's own deliverable -- the rmagine_cuda compute backend +
  rmcl's particle/correspondence `.cu` -- is a portable mechanical hipify (Strategy A) with a
  known wave64 reduction fix, and does NOT need HIPRT. That is the right first milestone and
  is independent of this probe.
- The HIPRT reimplementation target is a **new `rmagine_hiprt` backend** mirroring
  `rmagine_optix`: full raygen+closesthit+miss across FIVE sensor models (Pinhole, Spherical,
  O1Dn, OnDn + range/gen/hit program variants), 12 `optixTrace` sites, 13 program groups,
  instance-level acceleration (`optixGetInstanceId`/`optixGetSbtGASIndex` -> HIPRT scene +
  `hiprtInstance`), and multiple SBT records -> a real func-table/in-kernel-dispatch
  restructure (the hard column above applies in full here, unlike EnvGS). rmcl also pulls in
  ROS 2 / ament (not on this host) for the node layer.
- Large because it is a general multi-sensor RT backend for a third-party library, not a
  single contained tracer, and because the value is gated behind first shipping rmagine_cuda
  and standing up rmagine's backend abstraction for a new RT provider. Recommendation stands
  (per rmcl plan): port rmagine_cuda first (immediately validatable), treat rmagine_hiprt as
  a separate, larger follow-on.

### splatad -- effort: NOT APPLICABLE (no OptiX; this port is not HIPRT-gated)
- **splatad does not use OptiX or any ray tracing.** A grep of the repo for
  `optix|optixTrace|raygen|closesthit|anyhit|__miss__|accel.*build|hiprt|bvh|owl|embree`
  returns ZERO matches (confirmed against `projects/splatad/src/`). The premise that splatad
  is "OptiX-gated (lidar)" is incorrect.
- SplatAD's lidar path is **spherical-coordinate RASTERIZATION** (custom rasterizer kernels:
  `lidar_proj`, `fully_fused_lidar_projection_{fwd,bwd}`, `isect_lidar_tiles`,
  `rasterize_to_points`, ...), the same algorithm class as gsplat's camera 3DGS rasterizer
  with a pinhole->azimuth/elevation projection swap. The "rays" are a static az x el sample
  grid, not traced rays.
- Therefore splatad is a **mechanical Strategy-B hipify** (it IS gsplat v1.0.0 + lidar; the
  gsplat camera fault classes are already solved in projects/gsplat). Camera and lidar paths
  land together. **No HIPRT involvement; do not gate splatad on this decision.**

---

## Bottom line for the port-vs-defer call

- HIPRT **builds and traces correctly on gfx90a/ROCm 7.2.1** (validated, with one trivial
  upstream-worthy cache-name fix on MI250X). It is a viable OptiX replacement on CDNA via
  software traversal -- correctness yes; RT-core perf no (acceptable for research/training).
- **EnvGS Stage 2 = MEDIUM** and is the best first HIPRT port: small OptiX surface, anyhit
  already matches HIPRT's functor model, triangle GAS validated; the real work is the
  differentiable backward + torch-extension build wiring, not the ray-tracing model.
- **rmcl = LARGE** and is really an rmagine-backend project; its high-value, HIPRT-free
  milestone (rmagine_cuda) should go first regardless.
- **splatad = mislabeled**; it has no OptiX and needs no HIPRT -- it is a straight gsplat
  hipify and can proceed independently of any HIPRT decision.

Artifacts (all under `agent_space/hiprt_probe/`, gitignored): the HIPRT clone + Release
build, the minimal trace test (`HIPRT/minimal/{min_trace.cpp,trace_kernel.h}`), and the
one-line cache-name patch in `HIPRT/hiprt/impl/Compiler.cpp`.
