# Port Plan: anari-visionaray

## Project

- Name: anari-visionaray
- Upstream: https://github.com/szellmann/anari-visionaray
- Default branch: main (cloned @ 7edfca5 "Fix bounding box computation of nanovdbs")
- Description: An ANARI device (Khronos ANARI 1.0) implemented over the visionaray ray tracer. Apache-2.0, C++17, CMake >= 3.23. Builds into separate device shared libraries: CPU (`anari_library_visionaray`, always), CUDA (`anari_library_visionaray_cuda`, opt-in `ANARI_VISIONARAY_ENABLE_CUDA`), and HIP (`anari_library_visionaray_hip`, opt-in `ANARI_VISIONARAY_ENABLE_HIP`).
- depends_on: visionaray (AMD-Ecosystem/visionaray @ moat-port; HIP support landed upstream via szellmann/visionaray#51). Consumed via `find_package(visionaray 0.6.1)`.

## Existing AMD support

**Finding: AUTHORITATIVE but INCOMPLETE upstream HIP path. Decision: validate-and-complete (NOT from-scratch, NOT skip).**

- The upstream author (szellmann) wrote the only HIP support that exists. README line 39 calls it "experimental but seldom tested." It is the maintainer's own work, so it is authoritative, not a community hack -- adopt and complete it.
- No competing/standalone AMD port exists: `gh pr list --repo szellmann/anari-visionaray --state all --search "ROCm OR HIP OR AMD"` and the issue search return nothing; no fork under a ROCm/AMD/GPUOpen org (the five forks are personal/unrelated: jar091, faouziH21, disini, HellmannM, jeffamstutz). Web search surfaced only generic ROCm/ANARI pages, no anari-visionaray AMD project. So MOAT is not duplicating AMD's own work.
- This is the actively-developed downstream of our already-landed base `visionaray` HIP port (see projects/anari-visionaray/notes.md provenance), so it is the higher-value home for AMD support.

Authoritative-vs-community judgment: AUTHORITATIVE (upstream-maintainer-authored), so the move is to point MOAT at the existing `.hip`/option scaffolding, fill the gaps below, validate on real gfx90a, and contribute the completion upstream.

## Build classification

**cmake -> Strategy A.** Evidence:
- Root `CMakeLists.txt`: `cmake_minimum_required(VERSION 3.23)`, `project(anari_library_visionaray LANGUAGES C CXX)`. No `find_package(Torch)`, no setup.py/pyproject, no torch.
- HIP is wired as a first-class CMake language already: `option(ANARI_VISIONARAY_ENABLE_HIP ...)` -> `enable_language(HIP)` + `find_package(hip)` (CMakeLists.txt lines 33-38), and a `${PROJECT_NAME}_hip` SHARED target (lines 61-69, 220-251).

This is a Strategy A variant: the project already keeps native parallel `WITH_CUDA`/`WITH_HIP` source paths (the `.cu`/`.hip` translation units are thin `#include "X.cpp"` wrappers that recompile the host `.cpp` under the GPU compiler), so there is no compat-header to add and no symbol hand-renaming. The work is completing the existing HIP path, not introducing one.

## Port strategy

**Strategy A variant -- complete the maintainer's existing HIP device path; do NOT re-architect.**

Architecture (verified): each GPU translation unit is a thin wrapper, e.g. `frame/Frame.hip` is just `#include "Frame.cpp"`; the device-vs-host divergence lives inside the shared `.cpp`/`.h` guarded by `#ifdef WITH_CUDA / #elif defined(WITH_HIP) / #else`. The device libraries are built by compiling the SAME `.cpp` bodies under nvcc/hipcc via these wrappers. The codebase already carries thorough `WITH_HIP` branches across DeviceArray.h, DeviceBVH.h, DeviceCopyableObjects.h, VisionarayScene.cpp, common.h, for_each.h (a complete `hip::for_each` already exists), and the four shipped `.hip` wrappers (Frame, DirectLight_impl, Raycast_impl, VisionarayScene).

The gaps (below) are: (1) the volume spatial-field device units have NO HIP path -- no `.hip` wrappers and their device kernels are `#ifdef WITH_CUDA`-only, so under HIP they silently fall to a CPU `#else` that reads device-resident `DeviceArray` data on the host (incorrect / likely crash); (2) the HIP library is built but not installed (CMake bug); (3) NanoVDB is excluded from HIP; (4) a latent HIP-path compile typo in DeviceBVH.h; (5) no device LBVH builder is invoked for HIP (a deliberate `flags=0` fallback to the CPU builder -- correctness-preserving, kept as-is for this port).

Mechanical-vs-AMD-native: this is a ray tracer with NO warp intrinsics and NO CUTLASS/wgmma/tensor-core kernels (confirmed below). A correctness-first mechanical completion of the existing HIP path is the right and sufficient deliverable; no AMD-native rewrite is warranted.

## CUDA surface inventory

### Device kernels / GPU dispatch (all already `WITH_HIP`-aware unless noted)
| Location | GPU surface | HIP status |
|---|---|---|
| frame/Frame.cpp (.hip wrapper exists) | frame fill / accumulation `for_each` | HIP path present |
| renderer/DirectLight_impl.cpp, Raycast_impl.cpp (.hip wrappers exist) | render kernels via `hip::for_each` | HIP path present |
| scene/VisionarayScene.cpp (.hip wrapper exists) | `getPrimBoundsGPU<<<1,1>>>`, instance/light hipMemcpy | HIP path present (line 14-49, 85-134) |
| scene/volume/spatial_field/GridAccel.cpp | `computeMaxOpacities` `cuda::for_each` | **GAP**: `#ifdef WITH_CUDA cuda::for_each #else CPU`; no `WITH_HIP` arm -> falls to CPU path over device data |
| scene/volume/spatial_field/UnstructuredField.cpp | `UnstructuredField_buildGridGPU<<<>>>` + lbvh/bounds (5 `#ifdef WITH_CUDA`, 0 `WITH_HIP`) | **GAP**: device kernel + cuda_index_bvh build + cudaMemcpy bounds are CUDA-only |
| scene/volume/spatial_field/BlockStructuredField.cpp | `BlockStructuredField_buildGridGPU<<<>>>` (3 `#ifdef WITH_CUDA`, 0 `WITH_HIP`) | **GAP**: same shape as Unstructured |
| scene/volume/spatial_field/NanoVDBField.cpp | nanovdb device upload, `<cuda_runtime.h>`, nanovdb CUDA DeviceBuffer | **GAP**: CUDA-only; HIP explicitly excluded (see scope decision) |

### Warp intrinsics
**None.** `grep -rniE '__shfl|__ballot|__activemask|warpSize|warp'` over the device sources returns nothing. The base visionaray port reached the same conclusion. No wave64-vs-wave32 risk in this repo. (Followers gfx1100/gfx1201 therefore expected to be rebuild-and-revalidate, no arch delta.)

### CUDA libraries
- Thrust/CUB: not used directly in anari-visionaray; the GPU LBVH builder (CUB->hipCUB) lives in the base visionaray library we already ported and validated. anari-visionaray does NOT invoke the device LBVH builder under HIP today (VisionarayScene.cpp line 85-88 passes `flags=0`, and DeviceBVH.h line 290-295 routes the non-fast-build case to the CPU builder), so no rocThrust/hipCUB link is required by this repo for the planned scope.
- cuBLAS/cuFFT/cuRAND/cuSPARSE: none.
- NanoVDB: vendored under external/nanovdb (header-only); its CUDA DeviceBuffer (`external/nanovdb/util/cuda/...`) is the only piece coupled to the CUDA runtime. Used only by NanoVDBField (deferred -- see scope).

### Textures / surfaces
- Texture objects come from the base visionaray `hip_texture.h` (already ported); anari-visionaray selects it via common.h line 39-40 (`#elif defined(WITH_HIP) #include "visionaray/texture/hip_texture.h"`). The transfer-function sampler (`tex1D` in GridAccel.cpp computeMaxOpacities) and Image1D/2D/3D samplers ride that path. No layered arrays, no surfaces in this repo.

### Memory / streams
- hipMalloc/hipFree/hipMemcpy and hipStreamCreate/Destroy already used in the present HIP branches (GridAccel.cpp 16-17/26, VisionarayScene.cpp, DeviceArray.h, DeviceBVH.h). The missing branches must add the parallel hip* calls.

## Risk list

1. **Volume spatial-field CPU `#else` fallback over device memory (correctness, primary risk).** GridAccel/Unstructured/BlockStructured wrap their GPU work in `#ifdef WITH_CUDA ... #else (host loop) #endif`. Compiled under HIP the `#else` runs, but `m_elements`/`m_vertices`/`vaccel.*` are `DeviceArray`/`DevicePointer` device allocations (DeviceArray.h: HIP uses hipMalloc; DevicePointer's `operator*`/`operator[]` are `__device__`-only). A host loop dereferencing them reads device pointers on the host -> garbage or fault. Fix is to add real `WITH_HIP` GPU arms mirroring `WITH_CUDA` (launch the `__global__` build-grid kernels via hipcc's `<<<>>>`, use `hip::for_each`, `hip_index_bvh`, `HIP_SAFE_CALL(hipMemcpy(...DeviceToHost))` for bounds). The `__global__` kernels themselves are arch-neutral and compile under hipcc unchanged; only their guard and launch site need the HIP arm.

2. **DeviceBVH.h HIP-path typo (compile blocker).** Line 182 `hitPointerAttribute_t attributes = {};` should be `hipPointerAttribute_t` (line 265 spells it correctly). This is dead under CUDA but breaks the HIP build of `rebuildHostIndexBVH2`. Also audit the adjacent free-guard asymmetry: line 211 frees when `attributes.memoryType == hipMemoryTypeDevice` while line 286 frees when `== hipMemoryTypeHost`; reconcile to free only the locally-allocated staging buffer (the host-staging case), matching the CUDA `if (!attributes.devicePointer)` semantics.

3. **install() omits the HIP library (ship blocker).** CMakeLists.txt lines 272-275 append `${PROJECT_NAME}_cuda` to `DEVICE_LIBS` but never `${PROJECT_NAME}_hip`, so the built HIP device is not installed and ANARI cannot loadLibrary("visionaray_hip") from an install tree. Add the parallel `if (hip) set(DEVICE_LIBS ${DEVICE_LIBS} ${PROJECT_NAME}_hip) endif()`.

4. **No clean-environment package discovery.** `find_package(hip)` (CMakeLists.txt line 37) derives `/opt/rocm` from PATH on our hosts but fails on a clean ROCm container (PORTING_GUIDE: hip_DIR-NOTFOUND). Always pass `-DCMAKE_PREFIX_PATH=/opt/rocm` (or append `;/opt/rocm` to the existing prefix that points at the ANARI-SDK+visionaray install) in the recipe and in any documented build block.

5. **Texture linear-filter over float on gfx90a (CDNA2).** The transfer-function `tex1D` sampler and Image samplers come from base visionaray's hip_texture path. If any are created `filterMode=Linear` + `readMode=ElementType` over float, gfx90a rejects creation at runtime (PORTING_GUIDE fault class). The base visionaray port validated its texture path on gfx90a, so this is low risk here, but the validator must watch for a "hipCreateTextureObject: operation not supported" at first volume render and, if it appears, treat it as a base-visionaray follow-up (not an anari-visionaray source change).

6. **HIP `<<<>>>` triple-chevron launch under hipcc.** The CUDA arms use `kernel<<<grid,block>>>(...)`; hipcc accepts the chevron syntax, so the HIP build-grid arm can use the same spelling (for_each.h's hip branch instead uses `hipLaunchKernelGGL` -- either is fine; match the file's local style).

7. **NanoVDB device buffer is CUDA-coupled.** NanoVDBField uses `<cuda_runtime.h>` and nanovdb's CUDA DeviceBuffer; there is no hip DeviceBuffer in the vendored nanovdb. Porting it means providing a hip-backed nanovdb device buffer -- larger scope. Deferred (below); keep the existing `message(WARNING "No VDB support with HIP")` so the HIP device simply lacks the VDB spatial-field subtype.

8. **rule-of-five on GridAccel stream handle.** GridAccel holds a `stream` created in the ctor and destroyed in the dtor (GridAccel.cpp 14-28). It is not move-aware; if a GridAccel is ever moved/copied the stream could double-destroy (PORTING_GUIDE: AMD faults where CUDA tolerates). Pre-existing under CUDA too; verify GridAccel is only ever held by-value/never moved before treating as in-scope. Likely note-only.

## Scope decision: NanoVDB / volume-field device units

- **In scope (this port):** GridAccel, UnstructuredField, BlockStructuredField device paths under HIP -- these are the structured/AMR/unstructured volume accelerators and are reachable without external runtime coupling (they need only hipMalloc/hipMemcpy/`<<<>>>`/`hip::for_each`, all already available). Completing them is what makes the HIP device render volumes, which is the differentiator this device is known for.
- **Deferred (register in data/deferred.json):** NanoVDB-on-HIP. Rationale: the vendored `external/nanovdb` provides only a CUDA DeviceBuffer (`util/cuda/CudaDeviceBuffer.h`); a HIP device-buffer shim for nanovdb is a separable, larger piece, and upstream itself excludes it (`message(WARNING "No VDB support with HIP")`). The HIP device will build and run with every non-VDB spatial-field type; VDB volumes simply remain CUDA/CPU-only. Register as kind `feature-port`, resume point: provide a hip-backed nanovdb DeviceBuffer + a `WITH_HIP` arm in NanoVDBField.cpp, then add NanoVDBField.hip to the `_hip` target and drop the warning.

## File-by-file change list

### CMake (CMakeLists.txt)
- HIP target source list (lines 220-227): add the three volume-field `.hip` wrappers ->
  `scene/volume/spatial_field/BlockStructuredField.hip`, `GridAccel.hip`, `UnstructuredField.hip`.
- Install (lines 272-275): add `if (hip) set(DEVICE_LIBS ${DEVICE_LIBS} ${PROJECT_NAME}_hip) endif()`.
- Leave the NanoVDB `message(WARNING "No VDB support with HIP")` (line 233) as-is (deferred).
- Do NOT hardcode an arch literal; the project relies on `-DCMAKE_HIP_ARCHITECTURES`. (No `set(CMAKE_HIP_ARCHITECTURES gfx90a)` literal exists today; keep it that way so followers need no source change.)

### New thin `.hip` wrappers (mirror the existing `.cu` wrappers; each is `#include "X.cpp"`)
- scene/volume/spatial_field/GridAccel.hip
- scene/volume/spatial_field/UnstructuredField.hip
- scene/volume/spatial_field/BlockStructuredField.hip

### Add `WITH_HIP` GPU arms (mirror the `WITH_CUDA` arms)
- scene/volume/spatial_field/GridAccel.cpp: add `#elif defined(WITH_HIP)` calling `hip::for_each(stream, ...)` parallel to the `WITH_CUDA cuda::for_each` (line 74-78).
- scene/volume/spatial_field/UnstructuredField.cpp: add `WITH_HIP` arms at the five `#ifdef WITH_CUDA` sites (lines 232, 246, 301, 334 (guard the `__global__` for HIP too), 366) -- hip_index_bvh build, `hipMemcpy` DeviceToHost bounds, and the `UnstructuredField_buildGridGPU<<<>>>` launch.
- scene/volume/spatial_field/BlockStructuredField.cpp: add `WITH_HIP` arms at the three `#ifdef WITH_CUDA` sites (lines 104, 144, 176) mirroring Unstructured.
- Where a `__global__` kernel is currently inside `#ifdef WITH_CUDA`, widen the guard to `#if defined(WITH_CUDA) || defined(WITH_HIP)` (the kernel body is arch-neutral; VisionarayScene.cpp line 14 already uses exactly this idiom -- follow it).

### Bug fix
- DeviceBVH.h line 182: `hitPointerAttribute_t` -> `hipPointerAttribute_t`; reconcile the free-guard memoryType check (line 211/286) to free only the locally hipMalloc'd staging buffer.

### Attribution
- The three new `.hip` wrappers are trivial includes (no copyright header beyond matching the `.cu` siblings' SPDX). The substantive `WITH_HIP` source additions to the three spatial-field `.cpp` files get a parallel `Copyright (c) 2026 Advanced Micro Devices, Inc.` line under the existing upstream header, in the project's house style, authored `Jeff Daily` (per copyright-attribution-rule).

## Dependency build/install commands (do these first)

Build into a shared prefix `_deps/anari-visionaray/install` (gitignored). Mirror the CI yml for ANARI-SDK + visionaray flags; add ROCm/HIP flags for visionaray and point CMAKE_PREFIX_PATH at both the prefix and /opt/rocm.

```bash
ROOT=/var/lib/jenkins/moat
PREFIX=$ROOT/_deps/anari-visionaray/install
mkdir -p "$PREFIX"

# 1) ANARI-SDK (KhronosGroup/ANARI-SDK @ next_release).
#    BUILD_EXAMPLES=ON so we get anariTutorial for headless validation; helide OFF.
git clone --depth=1 -b next_release https://github.com/KhronosGroup/ANARI-SDK \
  $ROOT/_deps/anari-visionaray/ANARI-SDK
cmake -S $ROOT/_deps/anari-visionaray/ANARI-SDK -B $ROOT/_deps/anari-visionaray/ANARI-SDK/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DBUILD_EXAMPLES=ON -DBUILD_VIEWER=OFF -DBUILD_HELIDE_DEVICE=OFF \
  -DINSTALL_VIEWER_LIBRARY=OFF -DBUILD_TESTING=OFF
cmake --build $ROOT/_deps/anari-visionaray/ANARI-SDK/build -j$(nproc) --target install

# 2) visionaray (our ported AMD-Ecosystem/visionaray @ moat-port), header-only install
#    with features off (per CI) BUT HIP enabled so the installed config exposes the HIP path.
git clone --depth=1 -b moat-port https://github.com/AMD-Ecosystem/visionaray \
  $ROOT/_deps/anari-visionaray/visionaray
cd $ROOT/_deps/anari-visionaray/visionaray && git submodule update --init --recursive
cmake -S $ROOT/_deps/anari-visionaray/visionaray -B $ROOT/_deps/anari-visionaray/visionaray/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DVSNRAY_ENABLE_EXAMPLES=OFF -DVSNRAY_ENABLE_VIEWER=OFF -DVSNRAY_ENABLE_COMMON=ON \
  -DVSNRAY_ENABLE_CUDA=OFF -DVSNRAY_ENABLE_HIP=ON -DVSNRAY_ENABLE_TBB=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build $ROOT/_deps/anari-visionaray/visionaray/build -j$(nproc) --target install
```

Note: VSNRAY_ENABLE_COMMON=ON is needed only if the sample viewer (visionaray_common) is built for validation; the CI installs visionaray header-only with COMMON=OFF. If a `visionaray_common` link is required by the viewer path and Boost is unavailable, prefer the anariTutorial-based validation (below) and install visionaray with COMMON=OFF.

## Build commands (anari-visionaray, gfx90a)

```bash
PREFIX=/var/lib/jenkins/moat/_deps/anari-visionaray/install
cmake -S /var/lib/jenkins/moat/projects/anari-visionaray/src -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DANARI_VISIONARAY_ENABLE_HIP=ON \
  -DANARI_VISIONARAY_ENABLE_CUDA=OFF \
  -DANARI_VISIONARAY_ENABLE_NANOVDB=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_INSTALL_PREFIX=build/install \
  -DCMAKE_PREFIX_PATH="$PREFIX;/opt/rocm"
cmake --build build -j$(nproc) --target install
```

Followers (gfx1100/gfx1201): identical command with `-DCMAKE_HIP_ARCHITECTURES=<arch>`; no source change expected (no warp intrinsics). Set `ANARI_VISIONARAY_ENABLE_NANOVDB=OFF` for HIP since VDB is deferred (with NANOVDB=ON the HIP build still works -- the warning fires and VDB is simply absent from the HIP device -- but OFF keeps the CPU/HIP device feature sets aligned during validation).

## Test plan

There is NO in-tree unit-test target. Validation drives the built `visionaray_hip` ANARI device on a real GPU and renders frames.

### GPU validation (gfx90a, the lead gate)
1. **Headless render via ANARI-SDK anariTutorial path.** anariTutorial.c hardcodes "helide"; the robust substitute is the ANARI "environment" loader which honors `ANARI_LIBRARY` (the in-repo viewer uses `loadLibrary("environment")`, viewer.cpp line 1003). Plan to add a tiny headless harness under `apps/` (no Boost, no GUI) that:
   - `anari::loadLibrary("visionaray_hip")`, `anariNewDevice(lib,"default")`,
   - builds (a) a triangle/sphere surface scene and (b) a **structured-regular + unstructured volume** scene (to exercise the newly-completed GridAccel/UnstructuredField/BlockStructuredField HIP device paths -- the whole point of this port),
   - renders one frame, reads back the framebuffer, asserts non-empty / finite / non-constant pixels, and writes a PPM for eyeball check.
   This harness doubles as the GPU proof artifact (analogous to base visionaray's hip_test). Build it only under `ANARI_VISIONARAY_ENABLE_HIP`.
   Run: `HIP_VISIBLE_DEVICES=<n> ANARI_LIBRARY=visionaray_hip <harness> volume_scene out.ppm` on a gfx90a GPU (this host has 4: HIP devices 0-3).
2. **Confirm the device ran on the GPU, not a silent CPU fallback** (PORTING_GUIDE validation trap): the loaded library must be `visionaray_hip` (its entrypoint is the only one that compiles the `WITH_HIP` paths), and the volume scene must hit the spatial-field HIP grid build (add a one-time stderr log in the new HIP arm during bring-up, or verify via rocprof that device kernels launched).
3. **Cross-check output against the CPU device** for the same scene (`ANARI_LIBRARY=visionaray`): the HIP-rendered image should match the CPU image within tolerance (ray tracer is deterministic for a fixed sample count / fixed seed). A divergence localizes a HIP device-path bug. This also serves as the deterministic cross-arch consistency gate the followers reuse.

### Follower platforms (after lead passes)
- gfx1100 (Linux), gfx1201 (Windows): rebuild with the matching arch and re-run the headless harness + CPU cross-check. No warp-size delta expected (no intrinsics). If the deterministic image diff vs the lead/CPU diverges, that is the bug to root-cause (loose "non-empty" gate would false-pass per PORTING_GUIDE).

### Non-GPU regression set (must not regress)
- CPU device build: `-DANARI_VISIONARAY_ENABLE_HIP=OFF -DANARI_VISIONARAY_ENABLE_CUDA=OFF` must still configure/build/install and render the same scenes (the `#else` host paths in the spatial-field files must be untouched by our edits).
- CUDA device build path: our changes are purely additive `#elif defined(WITH_HIP)` arms + new `.hip` files + the install/typo fixes; the `WITH_CUDA` arms are byte-unchanged. State this in the PR (do not claim "byte-for-byte"; per cuda-unchanged-phrasing, say the CUDA path is preserved because every change is HIP-guarded or build-only). We cannot run the CUDA build here (no NVIDIA GPU); rely on the guard structure + the unchanged CUDA arms.

## Open questions

1. Headless harness vs sample viewer: the plan adds a small headless `apps/` harness rather than depending on the Boost-needing `visionaraySampleViewer`. Confirm with the porter that adding a new validation app is acceptable to upstream, or whether to instead ship it under a test/ dir guarded by the HIP option (preferred: keep it minimal and HIP-gated so it does not perturb the default build).
2. Does the unstructured/AMR volume device path need the device LBVH builder, or is the CPU-built-then-uploaded BVH (current HIP fallback, flags=0) sufficient for correct rendering? Plan assumes the CPU-built BVH path is correctness-equivalent (only slower); if a volume scene needs the device builder, that is a base-visionaray follow-up, not anari-visionaray.
3. NanoVDB-on-HIP is deferred. Confirm registering it in data/deferred.json (kind feature-port) and scoping it out of the upstream PR body is the right call (it is, given upstream itself excludes VDB from HIP).
4. visionaray_common / Boost: whether any chosen validation path forces VSNRAY_ENABLE_COMMON=ON (and thus Boost). The anariTutorial/headless path should avoid it; confirm during bring-up.
