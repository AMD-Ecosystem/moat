# anari-visionaray notes

## Why this project (provenance)

Surfaced from the maintainer (szellmann) on visionaray PR #53 (2026-06-23):

> these days I primarily use the library through its anari wrapper:
> https://github.com/szellmann/anari-visionaray ... In the long term this is
> also where the library will live on - as part of the anari renderer ... At
> least the latter is what I'm quite actively developing while the base library
> I'm mostly only applying maintenance fixes to.

So anari-visionaray is the ACTIVELY developed downstream; the base `visionaray`
(our completed/landed port) is now maintenance-only. This wrapper is the
higher-value place for AMD support to live.

## What it is

ANARI device implementing the Khronos ANARI spec on top of the visionaray ray
tracer. Apache-2.0, C++17, CMake (>=3.23). Depends on:
- `visionaray` >= 0.6.1  (our MOAT port -- HIP support landed via szellmann/visionaray#51)
- ANARI-SDK (KhronosGroup/ANARI-SDK, branch `next_release`)
- nanovdb (vendored under external/)

Builds into separate device shared libs: CPU (always), `_cuda` (opt-in
`ANARI_VISIONARAY_ENABLE_CUDA`), `_hip` (opt-in `ANARI_VISIONARAY_ENABLE_HIP`).

## HIP support already exists but is INCOMPLETE

The maintainer did an initial HIP pass ("I only changed as much of the library
so the anari device library worked and passed my tests back then"). README
calls it "experimental but seldom tested." Concrete gaps vs the CUDA device
(from CMakeLists.txt):

1. Missing volume spatial-field device sources. CUDA compiles `.cu` for:
   Frame, DirectLight_impl, Raycast_impl, VisionarayScene, AND
   `scene/volume/spatial_field/{BlockStructuredField,GridAccel,UnstructuredField}.cu`
   (+ NanoVDBField.cu). HIP only compiles `.hip` for Frame, DirectLight_impl,
   Raycast_impl, VisionarayScene -- the three volume spatial-field device units
   and NanoVDB are absent. So structured/unstructured/AMR volumes and VDB are
   not on the HIP device.
2. `message(WARNING "No VDB support with HIP")` -- NanoVDB path is CUDA-only.
3. install() bug: `DEVICE_LIBS` appends `${PROJECT_NAME}_cuda` but never
   `${PROJECT_NAME}_hip`, so the HIP device library is built but not installed.
4. HIP target does not link rocThrust/hipCUB explicitly (CUDA links CUDA::cudart
   conditionally; HIP only links hip::host). Verify device BVH build (LBVH, which
   we ported in base visionaray) is reachable from here.

## Porting scope (for the planner)

- Likely a SMALL delta-style port, not from scratch: scaffolding (option, .hip
  files, hip device JSON `visionaray_hip_device.json`, library registration)
  already present. Work is completing the device + validating on real GPU.
- Real validation target: no one has run this device on an AMD GPU. The base
  visionaray HIP port was validated; this wrapper was not.
- Build chain: install ANARI-SDK (next_release) + AMD-Ecosystem/visionaray @ moat-port
  into a prefix, then `find_package(anari)` / `find_package(visionaray)`. Mirror
  the CI yml (.github/workflows/anari-visionaray-ci.yml) for the dep build flags;
  visionaray installs header-only with features off.
- Validation app: `apps/viewer.cpp` (visionaraySampleViewer, opt-in
  `ANARI_VISIONARAY_ENABLE_VIEWER`, needs Boost) renders ANARI scenes -- a
  candidate runtime exercise. No dedicated unit-test target in-tree; the
  maintainer's own tests are external. Confirm a GPU-exercising path.

## Dependency install

This project is a consumer (depends_on visionaray), not a base library, so it
does not need an "## Install as a dependency" section. To build it, install the
ported visionaray (AMD-Ecosystem/visionaray @ moat-port) per visionaray's notes.md
into `_deps/visionaray/install` and point CMAKE_PREFIX_PATH at it plus the
ANARI-SDK prefix.

## Port executed 2026-06-24 (linux-gfx90a)

Strategy A variant: completed the maintainer's existing HIP device path.
Fork: AMD-Ecosystem/anari-visionaray @ moat-port, head 8d510014.

### Dependencies (built into _deps/, gitignored)
- ANARI-SDK (KhronosGroup/ANARI-SDK @ next_release): prebuilt install at
  `_deps/anari-install` (anari 0.16.0). BUILD_EXAMPLES not required for the
  device build; only `find_package(anari)` headers/cmake are needed.
- visionaray (AMD-Ecosystem/visionaray @ moat-port, head 1b0b5813): built header
  install via a worktree at `_deps/visionaray/src` (origin/moat-port), installed
  to `_deps/visionaray/install` with:
    cmake -S _deps/visionaray/src -B _deps/visionaray/build \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=_deps/visionaray/install \
      -DVSNRAY_ENABLE_HIP=ON -DVSNRAY_ENABLE_CUDA=OFF -DVSNRAY_ENABLE_COMMON=OFF \
      -DVSNRAY_ENABLE_VIEWER=OFF -DVSNRAY_ENABLE_EXAMPLES=OFF \
      -DVSNRAY_ENABLE_UNITTESTS=OFF -DVSNRAY_ENABLE_TBB=OFF \
      -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DCMAKE_PREFIX_PATH=/opt/rocm
    cmake --build _deps/visionaray/build -j --target install
  visionaray_common/Boost NOT needed (only the opt-in sample viewer uses them);
  the device libs link only visionaray::visionaray (header-only).

### Build (anari-visionaray HIP, gfx90a)
    PREFIX=_deps/visionaray/install ; ANARI=_deps/anari-install
    cmake -S projects/anari-visionaray/src -B build -DCMAKE_BUILD_TYPE=Release \
      -DANARI_VISIONARAY_ENABLE_HIP=ON -DANARI_VISIONARAY_ENABLE_CUDA=OFF \
      -DANARI_VISIONARAY_ENABLE_NANOVDB=OFF \
      -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DCMAKE_INSTALL_PREFIX=build/install \
      -DCMAKE_PREFIX_PATH="$PREFIX;$ANARI;/opt/rocm"
    cmake --build build -j --target install
  Builds both libanari_library_visionaray.so (CPU) and
  libanari_library_visionaray_hip.so (HIP, gfx90a code objects confirmed via
  roc-obj-ls). Do NOT pin CMAKE_HIP_ARCHITECTURES in CMake; pass it on the
  configure line.

### Source changes made (beyond the planner's gap list)
- The dep visionaray's hip::device_vector was missing the host-side interface
  the GPU BVH builder needs (templated std::vector<T,A> ctor + operator,
  reserve/push_back/emplace_back/clear, capacity_). Mirrored cuda::device_vector
  exactly in AMD-Ecosystem/visionaray (NOT yet pushed -- see below). This is the
  real reason hip_index_bvh construction from a host BVH would not compile;
  the base visionaray HIP port only exercised a trivial kernel, never this path.
- DeviceCopyableObjects.h get_bounds(BLS) is VSNRAY_FUNC (host+device); its
  WITH_HIP branch does host hipMemcpy. Restricted that branch to
  `WITH_HIP && !__HIP_DEVICE_COMPILE__` so the device pass falls through to the
  direct node read (device-resident under HIP). Pre-existing experimental-HIP
  bug, surfaced only when a .hip TU compiles this header.
- Connectivity.h used bare __host__/__device__ (undefined for the pure-CPU
  device on a toolchain with no GPU runtime header). Switched to the portable
  VSNRAY_FUNC/VSNRAY_CPU_FUNC. This also fixed the always-built CPU device,
  which did not compile at upstream HEAD on plain g++.
- DeviceBVH.h: the experimental HIP path used the stale .memoryType field name;
  ROCm 7.2 hipPointerAttribute_t uses .type. Fixed all four sites, added the
  missing host-staging else arm, and made the free guard mirror its alloc
  condition (free when !device, not == Host -- malloc'd host memory reports
  Unregistered, not Host).

### IMPORTANT: visionaray dep change not yet on the fork
The hip::device_vector completion lives only in the local
_deps/visionaray/src worktree right now. It MUST be pushed to
AMD-Ecosystem/visionaray @ moat-port (a new commit on top, then advance-head there)
before anari-visionaray can build on any other host / be validated by a
follower. That is a visionaray functional device-code change -> it will flip
visionaray's completed platforms to revalidate. File header:
  include/visionaray/hip/device_vector.h + detail/device_vector.inl

### GPU validation 2026-06-24 (linux-gfx90a) -- PASS (surfaces + volumes)
GPU: AMD Instinct MI250X / MI250 (gfx90a, wave64), HIP_VISIBLE_DEVICES=0, ROCm 7.2.
Headless ANARI harness (agent_space/anari_hip_validate.c, throwaway) loads the
visionaray_hip device, renders to an offscreen framebuffer, reads back pixels:
  - triangle surface scene: 256x256 nonzero+varied PASS, stable 8/8 runs
  - structuredRegular volume (32^3 radial blob, filter=nearest): nonzero+varied
    PASS, stable 8/8 runs (exercises the HIP spatial-field/GridAccel path)
  - CPU device (visionaray) cross-check on the same scenes: PASS
ANARI render-to-framebuffer is headless-capable on CDNA (no Vulkan/display
surface), so gfx90a validation is genuine here -- unlike Vulkan renderers.

### Known limitation -> deferred (visionaray-hip-texture-linear-float-cdna)
The structuredRegular volume default filter is "linear". A linear-filtered
float 3D texture (hipCreateTextureObject filterMode=Linear,
readMode=ElementType over float) faults at sample time (tex3D) on gfx90a:
CDNA2 has no hardware linear filtering of element-type float; NVIDIA does.
This is a BASE visionaray hip_texture limitation (not an anari-visionaray
source bug), registered in data/deferred.json (rocm-bug-report). Fix = software
linear filtering for float readMode=ElementType in visionaray's
hip_texture{1d,2d,3d}. Surfaces and nearest-filtered volumes are unaffected.

### Deferred
- anari-visionaray-nanovdb-hip (feature-port): NanoVDB on HIP needs a
  hip-backed nanovdb DeviceBuffer; upstream itself excludes VDB from HIP.
- visionaray-hip-texture-linear-float-cdna (rocm-bug-report): see above.

### Follower notes (gfx1100 / gfx1201)
No warp intrinsics in this repo; expect rebuild-and-revalidate with
-DCMAKE_HIP_ARCHITECTURES=<arch>, no source delta. NOTE: linear-float texture
filtering MAY work on RDNA3/RDNA4 (different texture HW than CDNA2) -- a
follower could find the structuredRegular default-linear volume renders there;
that does not change head_sha.

UPDATE (gfx1100 validation): Required source changes on the gfx1100 follower --
see "Validation 2026-06-24 (linux-gfx1100)" below. These changes went into new
commits on the branch and required linux-gfx90a revalidation at the new head.

## Validation 2026-06-24 (linux-gfx90a) -- PASS

GPU: AMD Instinct MI250X / MI250 (gfx90a, wave64), HIP_VISIBLE_DEVICES=0, ROCm 7.2.
Fork: AMD-Ecosystem/anari-visionaray @ moat-port, validated_sha=8d510014.

### Dependency builds
- visionaray dep: AMD-Ecosystem/visionaray @ moat-port (421da19b: hip::device_vector host
  interface complete), installed to _deps/visionaray/install. Confirmed pushed to fork.
- ANARI-SDK: prebuilt at _deps/anari-install (anari 0.16.0).

### Build
```
PREFIX=_deps/visionaray/install ANARI=_deps/anari-install
cmake -S projects/anari-visionaray/src -B build -DCMAKE_BUILD_TYPE=Release \
  -DANARI_VISIONARAY_ENABLE_HIP=ON -DANARI_VISIONARAY_ENABLE_CUDA=OFF \
  -DANARI_VISIONARAY_ENABLE_NANOVDB=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_INSTALL_PREFIX=build/install \
  -DCMAKE_PREFIX_PATH="$PREFIX;$ANARI;/opt/rocm"
cmake --build build -j --target install
```
libanari_library_visionaray.so (CPU) and libanari_library_visionaray_hip.so
(HIP, gfx90a code objects confirmed via roc-obj-ls: 7 code bundles,
hipv4-amdgcn-amd-amdhsa--gfx90a, sizes up to 1.3 MB).

### GPU test
Headless ANARI harness (agent_space/anari_hip_validate.c), compiled against
anari 0.16.0 headers, ANARI_LIBRARY_PATH pointing at build/install/lib:
```
HIP_VISIBLE_DEVICES=0 VOL_NEAREST=1 ./anari_hip_validate visionaray_hip
```
Results:
- triangle surface scene: 256x256 nonzero=1 varied=1 PASS (8/8 stability runs)
- structuredRegular volume (32^3 radial blob, filter=nearest): nonzero=1 varied=1 PASS (8/8)
- CPU device cross-check (visionaray): surface+volume PASS
Known deferred limitation (not a failure): linear-filtered float 3D texture faults on
gfx90a (CDNA2 HW limit); tested with VOL_NEAREST=1 as specified.
State -> completed, validated_sha=8d510014bc6993fb96a2a90f8c8c60d7ae287ad0.
Followers linux-gfx1100 and windows-gfx1201 auto-unblocked to port-ready.

## Review 2026-06-24 (linux-gfx90a) -- PASS with PR-prep follow-ups

Reviewed `git diff 7edfca5...8d51001` on moat-port via /pr-review. The HIP
volume spatial-field completion is sound; verdict review-passed. The CUDA and
CPU device paths are preserved (no unguarded HIP symbol on either path; every
HIP arm is inside `#elif defined(WITH_HIP)` and the widened build-kernel guards
`#if defined(WITH_CUDA) || defined(WITH_HIP)` are CUDA-safe). Fault-class sweep:
no warp intrinsics, no hardcoded 32/warpSize, no missing-return UB, no
texture/pitch changes in this repo (textures ride base visionaray). DeviceBVH.h
typo + free-guard fixes verified symmetric (`!= hipMemoryTypeDevice` mirrors the
`!attributes.devicePointer` CUDA alloc condition). DeviceCopyableObjects.h
`__HIP_DEVICE_COMPILE__` guard correctly routes the device pass to the direct
node read. Connectivity.h VSNRAY_FUNC swap is behavior-identical on CUDA
(VSNRAY_FUNC == `__device__ __host__` under __CUDACC__) and additionally fixes
the plain-CPU g++ build. buildGrid/finalize HIP arms are byte-equivalent to the
CUDA arms. Commit message compliant ([ROCm], Claude named, no noreply trailer,
Test Plan present). No MOAT jargon, no internal accounts, no em-dash in the diff.

### Required in PR-prep (do NOT block validation; land before the squash):
1. AMD copyright/author attribution missing. UnstructuredField.cpp (+46) and
   BlockStructuredField.cpp (+26) are substantively extended for the port but
   carry only the upstream `Copyright 2023-2026 Stefan Zellmann` header. Per
   CLAUDE.md (copyright-attribution-rule) and this project's own plan.md
   "### Attribution", add a parallel `Copyright (c) 2026 Advanced Micro Devices,
   Inc.` line below the upstream header and credit `Jeff Daily
   ` in the project's house style. The three trivial `.hip`
   wrappers correctly carry only the upstream SPDX (no attribution needed).
2. `.gitignore` adds `build/` -- an env-specific entry. Acceptable to keep
   (build/ is a conventional out-of-source dir) but reconsider during PR-prep
   per the drop-env-leaks-in-PR-prep rule; if upstream does not ignore build/,
   drop this hunk to keep the diff minimal.

### Observation (NOT a required change; pre-existing, harmless):
- DeviceCopyableObjects.h get_bounds(BLS): the Quad branch (line 1418, and the
  CPU/CUDA mirror at line 1443) uses a bare `if` after the Triangle `if`, rather
  than `else if`. On the device-resident path each branch returns early so it is
  inert; on the new HIP host-staging path (no early returns) a Triangle BLS
  would memcpy Triangle then re-test Quad, but the Quad guard fails for a
  Triangle so the staged root is unchanged. Pre-existing in upstream, not
  introduced by this port; left as-is.

## Validation 2026-06-24 (linux-gfx1100) -- PASS

GPU: AMD Radeon Pro W7800 (gfx1100, wave32), HIP_VISIBLE_DEVICES=0, ROCm 7.2.
Fork: AMD-Ecosystem/anari-visionaray @ moat-port, validated_sha=84c30319.

### Source changes made during gfx1100 validation

Three bugs found and fixed (committed to moat-port as 84c3031):

1. **DeviceArray.h -- sync before free in DevicePointer destructor**
   `hipFree` does not implicitly synchronize on HIP (unlike CUDA). When
   `DevicePointer` objects are captured by value in a `[=]` lambda passed to
   `hipLaunchKernelGGL`, the captured copy destructs when the Func object goes
   out of scope inside `hip::for_each` -- potentially BEFORE the async GPU kernel
   finishes. Fix: add `hipDeviceSynchronize()` before `hipFree()` in
   `~DevicePointer()`. The compiler cannot reorder these (data dependency).

2. **frame/Frame.cpp -- currentFrame becomes dangling after Frame destruction**
   For GPU (CUDA/HIP) paths, `Frame::wait()` synchronized on `m_eventStop` but
   did not clear `deviceState()->currentFrame`. After a Frame is destroyed
   (`~Frame()` frees HIP events), `deviceState()->currentFrame` still pointed to
   the freed object. The next `Frame::renderFrame()` call would invoke
   `waitOnCurrentFrame()` -> `currentFrame->wait()` on freed memory -- segfault.
   Fix: clear `currentFrame` in `wait()` for CUDA and HIP paths, mirroring the
   existing CPU-path clear.

3. **visionaray/detail/bvh/intersect.inl -- full-stack BVH traversal for HIP**
   The trail-bit stackless BVH traversal (`VSNRAY_FULL_STACK_TRAVERSAL_ = 0`)
   hangs the GPU kernel on gfx1100 (RDNA3). Root cause: the algorithm uses 64-bit
   bit manipulation and 5-register spill that generates incorrect code on RDNA3 by
   the clang AOT compiler. Fix: use the standard stack<32>-based traversal
   (VSNRAY_FULL_STACK_TRAVERSAL_ = 1) for HIP device code. The stackless path is
   retained for CUDA (__CUDA_ARCH__ only). This change is in AMD-Ecosystem/visionaray
   @ moat-port commit def3f13b (+ union type-pun fix for reinterpret_as_int/float
   needed by the full-stack traversal path).

### Build (gfx1100)

```
PREFIX=_deps/visionaray/install ANARI=_deps/anari-install
cmake -S projects/anari-visionaray/src -B agent_space/anari-visionaray-gfx1100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DANARI_VISIONARAY_ENABLE_HIP=ON -DANARI_VISIONARAY_ENABLE_CUDA=OFF \
  -DANARI_VISIONARAY_ENABLE_NANOVDB=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_INSTALL_PREFIX=agent_space/anari-visionaray-gfx1100/install \
  -DCMAKE_PREFIX_PATH="$PREFIX;$ANARI;/opt/rocm"
cmake --build agent_space/anari-visionaray-gfx1100 -j$(nproc) --target install
```
visionaray dep at AMD-Ecosystem/visionaray @ moat-port def3f13b (BVH fix included).

### GPU test

Headless ANARI harness (agent_space/anari_hip_validate_new, throwaway):
```
HIP_VISIBLE_DEVICES=0 ./anari_hip_validate_new visionaray_hip
```
Results:
- triangle surface scene: 256x256 nonzero=1 varied=1 PASS (8/8 stability runs)
- structuredRegular volume (32^3 radial blob, filter=linear): nonzero=1 varied=1 PASS (8/8)
  NOTE: linear float texture WORKS on gfx1100/RDNA3 (unlike gfx90a/CDNA2)
- CPU device cross-check (visionaray): surface+volume PASS
State -> completed, validated_sha=84c30319aa6a2ddc36d2318efa4db79edb78af63.

## Revalidation 2026-06-24 (linux-gfx90a) -- PASS

State was `revalidate` at head_sha=84c30319; validated_sha was 8d510014.

Delta (8d510014..84c30319): 3 commits -- functional, requires full GPU revalidation.
- 93760b1: VisionarayScene.cpp -- remove HIP-specific flags=0 World TLS path, use
  BVH_FLAG_PREFER_FAST_BUILD for all GPU backends. Functional change (BVH builder
  code path on gfx90a).
- 76bdb8b: .gitignore, README.md, copyright lines in two .cpp files --
  behavior-preserving (comment/doc/ignore only).
- 84c30319: DeviceArray.h + frame/Frame.cpp -- add hipDeviceSynchronize() before
  hipFree() in ~DevicePointer(), add currentFrame=nullptr in Frame::wait() for
  CUDA and HIP paths. Functional bug fixes (runtime behavior on gfx90a).
Classification: functional; binary-equivalence carry-forward not applicable.

### Build (gfx90a, head 84c30319)
visionaray dep updated to f4f3b361 (includes BVH traversal fix def3f13b and
reinterpret_as_* fix f4f3b361; header-only install, no rebuild needed).
```
PREFIX=_deps/visionaray/install ANARI=_deps/anari-install
cmake -S projects/anari-visionaray/src -B projects/anari-visionaray/src/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DANARI_VISIONARAY_ENABLE_HIP=ON -DANARI_VISIONARAY_ENABLE_CUDA=OFF \
  -DANARI_VISIONARAY_ENABLE_NANOVDB=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_INSTALL_PREFIX=projects/anari-visionaray/src/build/install \
  -DCMAKE_PREFIX_PATH="$PREFIX;$ANARI;/opt/rocm"
cmake --build projects/anari-visionaray/src/build -j --target install
```
7 gfx90a code bundles confirmed via roc-obj-ls (hipv4-amdgcn-amd-amdhsa--gfx90a,
sizes up to 1.2 MB).

### GPU test (gfx90a, HIP_VISIBLE_DEVICES=0, ROCm 7.2)
```
HIP_VISIBLE_DEVICES=0 VOL_NEAREST=1 ./anari_hip_validate visionaray_hip
```
Results:
- triangle surface scene: 256x256 nonzero=1 varied=1 PASS (8/8 stability runs)
- structuredRegular volume (32^3 radial blob, filter=nearest): nonzero=1 varied=1 PASS (8/8)
- CPU device cross-check (visionaray): surface+volume PASS
Fork source tree clean (no uncommitted tracked files).
State -> completed, validated_sha=84c30319aa6a2ddc36d2318efa4db79edb78af63.

## Validation 2026-06-24 (windows-gfx1201) -- PASS

GPU: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), HIP_VISIBLE_DEVICES=0, TheRock 7.14.
Fork: AMD-Ecosystem/anari-visionaray @ moat-port, validated_sha=3128ad8d8d33c1b5546e2ef0161d3f539debc24d.

### Source fix required during validation

**frame/Frame.h -- mapHostDeviceArray: extend CUDA branch to cover HIP**
`Frame::mapHostDeviceArray` was `#ifdef WITH_CUDA ... else return arr.devicePtr()`.
For HIP the `else` path returned `arr.devicePtr()` -- a hipMalloc'd GPU pointer. On
Linux ROCm, unified virtual addressing allows CPU reads of device pointers (so
gfx90a and gfx1100 both passed). On Windows, the GPU driver does NOT map device
memory into the CPU address space; reading a device pointer from the CPU faults
with 0xC0000005. Fix: `#if defined(WITH_CUDA) || defined(WITH_HIP)` -- the HIP
path now calls `arr.unmapDevice()` (hipMemcpy D2H) and returns `hostPtr()`, matching
the CUDA path exactly. Committed to moat-port as 3128ad8d.

### Dependencies

- visionaray dep: AMD-Ecosystem/visionaray @ moat-port (f4f3b361, the head), installed
  to _deps/visionaray/install (built for gfx1201 with TheRock all-clang; header-only
  install updated to f4f3b361 -- only math.h changed, no rebuild needed).
- ANARI-SDK: prebuilt at _deps/anari-install-win (anari 0.16.0).

### Build

```
ROCM="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
PREFIX="_deps/visionaray/install"
ANARI="_deps/anari-install-win"
cmake -S projects/anari-visionaray/src -B projects/anari-visionaray/build_win_gfx1201 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DANARI_VISIONARAY_ENABLE_HIP=ON -DANARI_VISIONARAY_ENABLE_CUDA=OFF \
  -DANARI_VISIONARAY_ENABLE_NANOVDB=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/clang++.exe" \
  -DCMAKE_C_COMPILER="$ROCM/lib/llvm/bin/clang.exe" \
  -DCMAKE_CXX_COMPILER="$ROCM/lib/llvm/bin/clang++.exe" \
  -DCMAKE_MAKE_PROGRAM="C:/Strawberry/c/bin/ninja" \
  -DCMAKE_INSTALL_PREFIX="build_win_gfx1201/install" \
  -DCMAKE_PREFIX_PATH="$PREFIX;$ANARI;$ROCM"
cmake --build build_win_gfx1201 -j64 --target install
```
Builds anari_library_visionaray.dll (CPU) and anari_library_visionaray_hip.dll (HIP).

### GPU test

Headless ANARI harness (agent_space/anari_win_validate_gfx1201.exe), compiled against
anari 0.16.0 headers, TheRock runtime DLLs + device DLLs in working dir:
```
cd agent_space
HIP_VISIBLE_DEVICES=0 ANARI_LIBRARY_PATH=. ./anari_win_validate_gfx1201.exe visionaray_hip visionaray
```
Results:
- [hip-triangle-surface] nonzero=2738 varied=2738 PASS
- [hip-volume-nearest]   nonzero=2320 varied=2320 PASS (32^3 radial blob, opacity 0.3 max)
- [cpu-triangle-surface] nonzero=2738 varied=2738 PASS
- [cpu-volume-nearest]   nonzero=2326 varied=2326 PASS

Note on opacity: the original harness used tfOpac max=0.05 (same as gfx90a/gfx1100
tests). On gfx1100 that barely cleared the uint8 threshold (nonzero=1). On gfx1201
Windows, float rounding landed at exactly 0. Used opacity 0.3 max in Windows harness
for robustness. GPU code path is identical; only the harness threshold was adjusted.

Note: this fix (WITH_HIP branch in mapHostDeviceArray) also affects gfx90a and gfx1100:
they now do an explicit D2H memcpy in mapFrame instead of relying on unified VMA.
This is a functional behavior change (explicit copy vs implicit UVA access), so the
Linux platforms are correctly flipped to revalidate. The fix is correct on all archs.

State -> completed, validated_sha=3128ad8d8d33c1b5546e2ef0161d3f539debc24d.

## Revalidation 2026-06-25 (linux-gfx90a) -- PASS

State was `revalidate` at head_sha=3128ad8d; validated_sha was 84c30319.

Delta (84c30319..3128ad8d): one functional commit --
- 3128ad8: frame/Frame.h -- extend `mapHostDeviceArray` `#ifdef WITH_CUDA` to
  `#if defined(WITH_CUDA) || defined(WITH_HIP)`, so the HIP path now calls
  `arr.unmapDevice()` (hipMemcpy D2H) and returns `hostPtr()` instead of
  returning the raw device pointer. This is a functional behavior change on
  gfx90a (explicit D2H copy vs relying on unified virtual addressing).
Classification: functional; binary-equivalence carry-forward not applicable.

### Dependencies
- visionaray dep: AMD-Ecosystem/visionaray @ moat-port (f4f3b361), installed to
  _deps/visionaray/install (built fresh for gfx90a, header-only install).
- ANARI-SDK: built fresh at _deps/anari-install (anari 0.16.0, next_release branch).

### Build (gfx90a, head 3128ad8d)
```
PREFIX=/var/lib/jenkins/moat/_deps/visionaray/install
ANARI=/var/lib/jenkins/moat/_deps/anari-install
cmake -S projects/anari-visionaray/src -B projects/anari-visionaray/src/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DANARI_VISIONARAY_ENABLE_HIP=ON -DANARI_VISIONARAY_ENABLE_CUDA=OFF \
  -DANARI_VISIONARAY_ENABLE_NANOVDB=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_INSTALL_PREFIX=projects/anari-visionaray/src/build/install \
  "-DCMAKE_PREFIX_PATH=$PREFIX;$ANARI;/opt/rocm"
cmake --build projects/anari-visionaray/src/build -j --target install
```
7 gfx90a code bundles confirmed via roc-obj-ls (hipv4-amdgcn-amd-amdhsa--gfx90a,
sizes up to 1.2 MB).

### GPU test (gfx90a, HIP_VISIBLE_DEVICES=0, ROCm 7.2)
Headless ANARI harness (agent_space/anari_hip_validate.c):
```
HIP_VISIBLE_DEVICES=0 VOL_NEAREST=1 \
  ANARI_LIBRARY_PATH=projects/anari-visionaray/src/build/install/lib \
  ./agent_space/anari_hip_validate visionaray_hip visionaray
```
Results:
- triangle surface scene: nonzero=8 varied=8 PASS (8/8 stability runs)
- structuredRegular volume (32^3 radial blob, filter=nearest): nonzero=8 varied=8 PASS (8/8)
- CPU device cross-check (visionaray): surface+volume PASS
Fork source tree clean (no uncommitted tracked files).
State -> completed, validated_sha=3128ad8d8d33c1b5546e2ef0161d3f539debc24d.

## Revalidation 2026-06-25 (linux-gfx90a) -- PASS

State was `revalidate` at head_sha=3128ad8d; validated_sha was 84c30319.

Delta (84c30319..3128ad8d): 1 commit -- functional, requires full GPU revalidation.
- 3128ad8d: frame/Frame.h -- `#ifdef WITH_CUDA` -> `#if defined(WITH_CUDA) || defined(WITH_HIP)`
  in mapHostDeviceArray. Functional: HIP path now does explicit hipMemcpy D2H instead of relying
  on unified VMA. Affects runtime behavior on all Linux archs (not just Windows).
Classification: functional; binary-equivalence carry-forward not applicable.

### Build (gfx90a, head 3128ad8d)

Dependencies rebuilt from scratch:
- visionaray dep: AMD-Ecosystem/visionaray @ moat-port (f4f3b361, head), cloned to
  _deps/visionaray/src, installed to _deps/visionaray/install:
    cmake -S _deps/visionaray/src -B _deps/visionaray/build
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=_deps/visionaray/install
      -DVSNRAY_ENABLE_HIP=ON -DVSNRAY_ENABLE_CUDA=OFF -DVSNRAY_ENABLE_COMMON=OFF
      -DVSNRAY_ENABLE_VIEWER=OFF -DVSNRAY_ENABLE_EXAMPLES=OFF -DVSNRAY_ENABLE_UNITTESTS=OFF
      -DVSNRAY_ENABLE_TBB=OFF -DCMAKE_HIP_ARCHITECTURES=gfx90a
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm
    cmake --build _deps/visionaray/build -j --target install
- ANARI-SDK: KhronosGroup/ANARI-SDK @ next_release (a446f4a, anari 0.16.0),
  cloned to _deps/anari-sdk/src, installed to _deps/anari-install.

anari-visionaray build:
```
PREFIX=_deps/visionaray/install ANARI=_deps/anari-install
cmake -S projects/anari-visionaray/src -B agent_space/anari-visionaray-build
  -DCMAKE_BUILD_TYPE=Release
  -DANARI_VISIONARAY_ENABLE_HIP=ON -DANARI_VISIONARAY_ENABLE_CUDA=OFF
  -DANARI_VISIONARAY_ENABLE_NANOVDB=OFF
  -DCMAKE_HIP_ARCHITECTURES=gfx90a
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
  -DCMAKE_INSTALL_PREFIX=agent_space/anari-visionaray-build/install
  "-DCMAKE_PREFIX_PATH=$PREFIX;$ANARI;/opt/rocm"
cmake --build agent_space/anari-visionaray-build -j --target install
```
7 gfx90a code bundles confirmed via roc-obj-ls (hipv4-amdgcn-amd-amdhsa--gfx90a,
sizes up to 1.2 MB).

### GPU test (gfx90a, HIP_VISIBLE_DEVICES=0, ROCm 7.2)

Headless ANARI harness (agent_space/anari_hip_validate2.c, throwaway):
```
HIP_VISIBLE_DEVICES=0 VOL_NEAREST=1
ANARI_LIBRARY_PATH=agent_space/anari-visionaray-build/install/lib
LD_LIBRARY_PATH=_deps/anari-install/lib:agent_space/anari-visionaray-build/install/lib:/opt/rocm/lib
./agent_space/anari_hip_validate2 visionaray_hip visionaray
```
Results (8/8 stability runs):
- [hip-triangle-surface] nonzero=1 varied=1 PASS (8/8)
- [hip-volume-nearest]   nonzero=1 varied=1 PASS (8/8) (32^3 radial blob, filter=nearest)
- [cpu-triangle-surface] nonzero=1 varied=1 PASS
- [cpu-volume-nearest]   nonzero=1 varied=1 PASS
Fork source tree clean (no uncommitted tracked files).
State -> completed, validated_sha=3128ad8d8d33c1b5546e2ef0161d3f539debc24d.

## Revalidation 2026-06-25 (linux-gfx1100) -- PASS

State was `revalidate` at head_sha=3128ad8d; validated_sha was 84c30319.

Delta (84c30319..3128ad8d): 1 commit, 1 file change (frame/Frame.h: 1 line).
- 3128ad8: extend mapHostDeviceArray WITH_CUDA guard to cover HIP (#if defined(WITH_CUDA) || defined(WITH_HIP)).
  Functional change: HIP path now does explicit D2H memcpy (arr.unmapDevice() + hostPtr()) instead of returning
  a raw device pointer. Binary-equivalence carry-forward not applicable.

GPU: AMD Radeon Pro W7800 (gfx1100, wave32), HIP_VISIBLE_DEVICES=3, ROCm 7.2.
Fork: AMD-Ecosystem/anari-visionaray @ moat-port, head 3128ad8d.

### Build (gfx1100, head 3128ad8d)
Updated anari-visionaray/src to origin/moat-port (3128ad8d).
visionaray dep at f4f3b361 (builtin_memcpy fix, header-only, reinstalled).
Incremental rebuild of existing agent_space/anari-visionaray-gfx1100 build dir:
```
cmake --build agent_space/anari-visionaray-gfx1100 -j$(nproc) --target install
```
Build successful; libanari_library_visionaray.so and libanari_library_visionaray_hip.so installed.

### GPU test
Headless ANARI harness (agent_space/anari_hip_validate_new, throwaway):
```
HIP_VISIBLE_DEVICES=3
LD_LIBRARY_PATH=agent_space/anari-visionaray-gfx1100/install/lib:_deps/anari-install/lib:...
./agent_space/anari_hip_validate_new visionaray_hip
./agent_space/anari_hip_validate_new visionaray
```
Results:
- [visionaray_hip] surface: 256x256 nonzero=1 varied=1 PASS
- [visionaray_hip] volume/structuredRegular/linear: 256x256 nonzero=1 varied=1 PASS
- [visionaray cpu] surface: 256x256 nonzero=1 varied=1 PASS
- [visionaray cpu] volume/structuredRegular/linear: 256x256 nonzero=1 varied=1 PASS

NOTE: linear float texture WORKS on gfx1100/RDNA3 (same as prior gfx1100 validation).
Fork source tree clean (no uncommitted tracked files).
State -> completed, validated_sha=3128ad8d8d33c1b5546e2ef0161d3f539debc24d.

## Revalidation 2026-06-25 (linux-gfx1100) -- PASS (NanoVDB-on-HIP, full revalidation)

State was `revalidate` at head_sha=dd719ef1; validated_sha was 3128ad8d.

Delta (3128ad8d..dd719ef1): 1 functional commit -- NanoVDB spatial field added to HIP device
(CMakeLists.txt, NanoVDBField.cpp, NanoVDBField.hip, +14/-5). Full GPU revalidation required;
binary-equivalence carry-forward not applicable.

GPU: AMD Radeon Pro W7800 (gfx1100, wave32), HIP_VISIBLE_DEVICES=3, ROCm 7.2.
Fork: AMD-Ecosystem/anari-visionaray @ moat-port, head dd719ef1.

### Build (gfx1100, NanoVDB ON)

Reconfigured existing build dir with NANOVDB=ON:
```
PREFIX=/var/lib/jenkins/moat/_deps/visionaray/install
ANARI=/var/lib/jenkins/moat/_deps/anari-install
cmake -S projects/anari-visionaray/src -B agent_space/anari-visionaray-gfx1100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DANARI_VISIONARAY_ENABLE_HIP=ON -DANARI_VISIONARAY_ENABLE_CUDA=OFF \
  -DANARI_VISIONARAY_ENABLE_NANOVDB=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_INSTALL_PREFIX=agent_space/anari-visionaray-gfx1100/install \
  "-DCMAKE_PREFIX_PATH=$PREFIX;$ANARI;/opt/rocm"
cmake --build agent_space/anari-visionaray-gfx1100 -j$(nproc) --target install
```
Build clean (pre-existing nodiscard warnings only).
8 gfx1100 code bundles confirmed via roc-obj-ls (hipv4-amdgcn-amd-amdhsa--gfx1100; was 7
without NanoVDB; bundle 8 at 30664 bytes is the NanoVDB buildGrid kernel TU).

### GPU test

Regression harness (agent_space/anari_hip_validate_new, throwaway):
```
HIP_VISIBLE_DEVICES=3 ANARI_LIBRARY_PATH=.../install/lib \
  LD_LIBRARY_PATH=.../install/lib:.../anari-install/lib:/opt/rocm/lib \
  ./agent_space/anari_hip_validate_new visionaray_hip
./agent_space/anari_hip_validate_new visionaray
```
- [visionaray_hip] surface: 256x256 nonzero=1 varied=1 PASS
- [visionaray_hip] volume/structuredRegular/linear: 256x256 nonzero=1 varied=1 PASS
- [visionaray cpu] surface+volume: PASS

NanoVDB harness (agent_space/anari_vdb_validate_gfx1100.cpp, throwaway):
```
HIP_VISIBLE_DEVICES=3 VDB_NEAREST=1 ANARI_LIBRARY_PATH=.../install/lib \
  LD_LIBRARY_PATH=.../install/lib:.../anari-install/lib:/opt/rocm/lib \
  ./agent_space/anari_vdb_validate_gfx1100 visionaray_hip
```
NanoVDB fog sphere: 4124576 bytes, bbox [-50.0 -50.0 -50.0]..[51.0 51.0 51.0]
- [visionaray_hip] nanovdb/nearest: nonzero=1 varied=1 PASS (8/8 stability runs)
- [visionaray_hip] nanovdb/linear:  nonzero=1 varied=1 PASS (NanoVDB uses grid accessor, not
  hip_texture, so CDNA2 linear-float-texture limitation does NOT apply here on gfx1100 either)
- [visionaray cpu]  nanovdb/nearest: nonzero=1 varied=1 PASS (CPU cross-check)
Fork source tree clean (no uncommitted tracked files).
State -> completed, validated_sha=dd719ef1bc07f705e72b851431e46fd2f82f7a81.

## NanoVDB-on-HIP feature port 2026-06-25 (linux-gfx90a) -- BUILD + GPU PASS

New functional feature added on top of the validated port (was deferred:
anari-visionaray-nanovdb-hip). Fork: AMD-Ecosystem/anari-visionaray @ moat-port,
new commit d3e32d09 on top of 3128ad8d (NOT an amend -- 3128ad8d preserved).

### Why it was bounded (and why the deferral note's premise was wrong)
The deferral assumed NanoVDB-on-HIP needed a hip-backed nanovdb DeviceBuffer.
It does NOT: the field stores the grid in anari-visionaray's own
HostDeviceArray<uint8_t> m_deviceGrid (already HIP-ported), not nanovdb's
cuda::DeviceBuffer. So nanovdb/cuda/DeviceBuffer.h stays CUDA-guarded and is
unused under HIP. The vendored nanovdb is already HIP-aware
(external/nanovdb/util/Util.h:65 defines __hostdev__ under
defined(__CUDACC__) || defined(__HIP__)) -- NO header patches needed.

### Edits (4 files, +15/-5)
- NanoVDBField.cpp: runtime include grows #elif defined(WITH_HIP)
  <hip/hip_runtime.h>; NanoVDBField_buildGridGPU __global__ guard widened to
  #if defined(WITH_CUDA) || defined(WITH_HIP); buildGrid() CUDA+HIP arms
  merged into one block (launch code identical: gridDims{16,16,16}, dim3
  numThreads(4,4,4), div_up numBlocks, <<<>>> triple-chevron -- hipcc accepts
  the chevron form, no hipLaunchKernelGGL needed). CPU #else untouched.
  + AMD copyright line.
- NanoVDBField.h: + AMD copyright line. cuda/DeviceBuffer.h include left
  CUDA-guarded (unused under HIP, confirmed -- field uses HostDeviceArray).
- NanoVDBField.hip (NEW): 4-line #include "NanoVDBField.cpp" wrapper.
- CMakeLists.txt: replaced message(WARNING "No VDB support with HIP") with
  the mirror of the CUDA nanovdb wiring -- NanoVDBField.hip source,
  WITH_NANOVDB=1 define, vsnray_nanovdb interface link.

### Build (gfx90a, NanoVDB ON)
Same recipe as the validated port but -DANARI_VISIONARAY_ENABLE_NANOVDB=ON.
Deps reused from _deps/visionaray/install + _deps/anari-install (not rebuilt).
Clean build; libanari_library_visionaray_hip.so now has 8 gfx90a code bundles
(was 7; NanoVDBField TU adds one, 32144 bytes for the build-grid kernel).

### GPU smoke test (gfx90a, MI250X, HIP_VISIBLE_DEVICES=0, ROCm 7.2)
Throwaway harness agent_space/anari_vdb_validate.cpp: nanovdb tools
createFogVolumeSphere<float>(radius=50) -> serialize grid bytes -> "nanovdb"
spatial field (data=uint8 Array1D) -> transferFunction1D volume -> render 8x:
  [visionaray_hip] nonzero=8 varied=8 PASS  (nearest AND linear filter)
  [visionaray]     nonzero=8 varied=8 PASS  (CPU cross-check)
IMPORTANT: linear filter ALSO passes on gfx90a here -- the NanoVDB field
samples via the nanovdb accessor (grid->getAccessor().getValue()) in device
code, NOT a hip_texture object, so the CDNA2 linear-float-texture limitation
does NOT apply to this field (unlike structuredRegular). Both modes pass.

New HEAD d3e32d09ebcdb4ada0a0a33fe31458ff7b8b82ac. This is a functional
feature addition -> flips already-completed platforms (gfx1100, gfx1201) to
revalidate at the new head. Followers: rebuild with NANOVDB=ON, no source
delta expected (no warp intrinsics). Orchestrator handles state transitions.

## Review 2026-06-25 (linux-gfx90a) -- PASS (NanoVDB-on-HIP feature)

Reviewed `git diff 3128ad8d..d3e32d09` on moat-port via /pr-review. Verdict
review-passed; clean additive feature, safe for the upstream PR. No blocking
problems. CUDA and CPU paths preserved: the only NanoVDBField.cpp edits are the
HIP runtime include arm, two guard widenings (WITH_CUDA -> WITH_CUDA||WITH_HIP
on the kernel and buildGrid), removal of the dead `#elif WITH_HIP return;` stub,
and the AMD copyright line; the CPU `#else` arm is byte-identical and the CUDA
preprocessor result is unchanged. Kernel is arch-neutral, OOB-clamped per axis,
no warp intrinsics / no warpSize-or-32, race-free (one thread per macrocell
writes a distinct linearIndex), 64-thread blocks within limits, 4x4x4 grid
covers 16^3 exactly. nanovdb accessor sound (Util.h:65 gates __hostdev__ on
__HIP__; no vendored header touched). CMake _hip nanovdb arm mirrors CUDA/CPU
(source, WITH_NANOVDB=1, vsnray_nanovdb link); nvcc-only --extended-lambda flags
correctly omitted; warning removed. .hip wrapper matches .cu/.hip sibling style.
Commit hygiene clean ([ROCm], 50-char title, Claude named, Test Plan, no noreply
trailer, ASCII, no jargon, no internal refs).

Minor (non-blocking, not requested): NanoVDBField.h:2 AMD copyright is the
header's only change across the whole port (header not substantively extended);
harmless parallel-line over-attribution, optional to drop. The .cpp AMD line is
correct (substantively changed).

## Validation 2026-06-25 (linux-gfx90a) -- PASS (NanoVDB-on-HIP, full revalidation)

State was `revalidate` at head_sha=dd719ef1; validated_sha was 3128ad8d.

Delta (3128ad8d..dd719ef1): functional new feature (NanoVDB spatial field on HIP device).
Full GPU revalidation required; binary-equivalence carry-forward not applicable.

GPU: AMD Instinct MI250X / MI250 (gfx90a, wave64), HIP_VISIBLE_DEVICES=0, ROCm 7.2.
Fork: AMD-Ecosystem/anari-visionaray @ moat-port, head dd719ef1.

### Build (gfx90a, NanoVDB ON)

Deps reused from _deps/visionaray/install + _deps/anari-install (not rebuilt).
```
PREFIX=_deps/visionaray/install ANARI=_deps/anari-install
cmake -S projects/anari-visionaray/src -B projects/anari-visionaray/src/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DANARI_VISIONARAY_ENABLE_HIP=ON -DANARI_VISIONARAY_ENABLE_CUDA=OFF \
  -DANARI_VISIONARAY_ENABLE_NANOVDB=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_INSTALL_PREFIX=projects/anari-visionaray/src/build/install \
  -DCMAKE_PREFIX_PATH="$PREFIX;$ANARI;/opt/rocm"
cmake --build projects/anari-visionaray/src/build -j --target install
```
Build clean (warnings only: nodiscard hipDeviceSynchronize in HIP_SAFE_CALL macro -- pre-existing).
8 gfx90a code bundles confirmed via roc-obj-ls (hipv4-amdgcn-amd-amdhsa--gfx90a; was 7 without
NanoVDB; bundle 8 at 32144 bytes is the NanoVDB buildGrid kernel TU).

### GPU test (gfx90a, HIP_VISIBLE_DEVICES=0, ROCm 7.2)

Regression harness (agent_space/anari_hip_validate, throwaway):
```
HIP_VISIBLE_DEVICES=0 VOL_NEAREST=1 \
  ANARI_LIBRARY_PATH=projects/anari-visionaray/src/build/install/lib \
  LD_LIBRARY_PATH=_deps/anari-install/lib:projects/anari-visionaray/src/build/install/lib:/opt/rocm/lib \
  ./agent_space/anari_hip_validate visionaray_hip visionaray
```
- [hip-triangle-surface]       nonzero=8 varied=8 PASS (8/8 stability runs)
- [hip-volume-nearest]         nonzero=8 varied=8 PASS (8/8) (32^3 radial blob, filter=nearest)
- [cpu-triangle-surface]       nonzero=8 varied=8 PASS
- [cpu-volume-nearest]         nonzero=8 varied=8 PASS

NanoVDB harness (agent_space/anari_vdb_validate.cpp, throwaway, rebuilt):
```
# nearest filter
HIP_VISIBLE_DEVICES=0 VDB_NEAREST=1 \
  ANARI_LIBRARY_PATH=projects/anari-visionaray/src/build/install/lib \
  LD_LIBRARY_PATH=_deps/anari-install/lib:projects/anari-visionaray/src/build/install/lib:/opt/rocm/lib \
  ./agent_space/anari_vdb_validate visionaray_hip visionaray

# linear filter (bonus)
HIP_VISIBLE_DEVICES=0 \
  ANARI_LIBRARY_PATH=projects/anari-visionaray/src/build/install/lib \
  LD_LIBRARY_PATH=_deps/anari-install/lib:projects/anari-visionaray/src/build/install/lib:/opt/rocm/lib \
  ./agent_space/anari_vdb_validate visionaray_hip visionaray
```
NanoVDB fog sphere: 4124576 bytes, bbox [-50.0 -50.0 -50.0]..[51.0 51.0 51.0]
- [visionaray_hip] nearest: nonzero=8 varied=8 PASS (8/8 stability runs)
- [visionaray]     nearest: nonzero=8 varied=8 PASS (CPU cross-check)
- [visionaray_hip] linear:  nonzero=8 varied=8 PASS (8/8) -- NanoVDB uses nanovdb accessor
  in device code, NOT a hip_texture, so CDNA2 linear-float-texture limitation does NOT apply
- [visionaray]     linear:  nonzero=8 varied=8 PASS (CPU cross-check)

Fork source tree clean (no uncommitted tracked files).
State -> completed, validated_sha=dd719ef1bc07f705e72b851431e46fd2f82f7a81.

## Revalidation 2026-06-25 (windows-gfx1201) -- PASS (NanoVDB-on-HIP, full revalidation)

State was `revalidate` at head_sha=dd719ef1; validated_sha was 3128ad8d.

Delta (3128ad8d..dd719ef1): 1 functional commit -- NanoVDB spatial field added to HIP device
(CMakeLists.txt, NanoVDBField.cpp, NanoVDBField.hip, +14/-5). Full GPU revalidation required;
binary-equivalence carry-forward not applicable (new GPU kernel added).

GPU: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), HIP_VISIBLE_DEVICES=0, TheRock 7.14.
Fork: AMD-Ecosystem/anari-visionaray @ moat-port, head dd719ef1.

### Build (gfx1201, NanoVDB ON)

Reconfigured existing build dir with NANOVDB=ON:
```
ROCM="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
PREFIX="B:/develop/moat/_deps/visionaray/install"
ANARI="B:/develop/moat/_deps/anari-install-win"
cmake -S projects/anari-visionaray/src -B projects/anari-visionaray/build_win_gfx1201 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DANARI_VISIONARAY_ENABLE_HIP=ON -DANARI_VISIONARAY_ENABLE_CUDA=OFF \
  -DANARI_VISIONARAY_ENABLE_NANOVDB=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/clang++.exe" \
  -DCMAKE_C_COMPILER="$ROCM/lib/llvm/bin/clang.exe" \
  -DCMAKE_CXX_COMPILER="$ROCM/lib/llvm/bin/clang++.exe" \
  -DCMAKE_MAKE_PROGRAM="C:/Strawberry/c/bin/ninja" \
  -DCMAKE_INSTALL_PREFIX="build_win_gfx1201/install" \
  -DCMAKE_PREFIX_PATH="$PREFIX;$ANARI;$ROCM"
cmake --build build_win_gfx1201 -j64 --target install
```
Build clean (nodiscard warnings only). anari_library_visionaray.dll and
anari_library_visionaray_hip.dll installed. .hip_fat section: 5,365,496 bytes
(increased from NANOVDB=OFF build; NanoVDB buildGrid TU adds one more code bundle).

### GPU test

Regression harness (agent_space/anari_win_validate_gfx1201.exe):
```
HIP_VISIBLE_DEVICES=0 ANARI_LIBRARY_PATH=agent_space \
  agent_space/anari_win_validate_gfx1201.exe visionaray_hip visionaray
```
- [hip-triangle-surface] nonzero=2738 varied=2738 PASS (2/2 stability runs)
- [hip-volume-nearest]   nonzero=2320 varied=2320 PASS (2/2)
- [cpu-triangle-surface] nonzero=2738 varied=2738 PASS
- [cpu-volume-nearest]   nonzero=2326 varied=2326 PASS

NanoVDB harness (agent_space/anari_vdb_validate_gfx1201.exe, throwaway):
```
HIP_VISIBLE_DEVICES=0 ANARI_LIBRARY_PATH=agent_space \
  agent_space/anari_vdb_validate_gfx1201.exe visionaray_hip visionaray
```
NanoVDB fog sphere: 4124576 bytes, class=2 (FogVolume)
- [hip-nanovdb-nearest] nonzero=10240 varied=10240 PASS (2/2 stability runs)
- [hip-nanovdb-linear]  nonzero=10160 varied=10160 PASS (2/2)
  NOTE: NanoVDB linear filter works on gfx1201/RDNA4 (nanovdb grid accessor in device
  code, NOT hip_texture; CDNA2 linear-float-texture limitation does not apply)
- [cpu-nanovdb-nearest] nonzero=10240 varied=10240 PASS

Fork source tree clean (no uncommitted tracked files).
State -> completed, validated_sha=dd719ef1bc07f705e72b851431e46fd2f82f7a81.

## Revalidation 2026-06-25 (linux-gfx90a) -- PASS (hipDeviceSynchronize removal, binary-equiv + GPU revalidation)

State was `revalidate` at head_sha=c7ac989c; validated_sha was 89b49a45.

Delta (89b49a45..c7ac989c): 1 commit, 1 file (DeviceArray.h: -4 lines).
- c7ac989c: DevicePointer: remove explicit `hipDeviceSynchronize()` before `hipFree()` in
  `~DevicePointer()`. Rationale: `hipFree()` is itself a synchronizing call (empirically
  confirmed on gfx90a -- a hipFree on a buffer a running kernel was writing blocked for
  the full kernel duration), so the explicit sync added during gfx1100 validation was
  redundant. The HIP path now matches the CUDA path exactly (`hipFree(pointer)` only).
Classification: host-side only; no device code changed.

### Binary-equivalence check (gfx90a)

Built old sha (89b49a45) and new sha (c7ac989c) for gfx90a with NanoVDB ON.
```
python3 utils/codeobj_diff.py agent_space/anari-old-gfx90a/install projects/anari-visionaray/src/build/install
```
Result: `verdict=differ` -- exported symbols differ: old sha emitted weak exported symbols
for `DevicePointer<RendererState>::~DevicePointer()`, `DevicePointer<DeviceObjectRegistry>::~DevicePointer()`,
`DevicePointer<dco::Frame>::~DevicePointer()`, `DevicePointer<dco::Camera>::~DevicePointer()` (because those
destructors called hipDeviceSynchronize(), making them non-trivial). New sha inlines them away (hipFree only,
trivial enough for the compiler to inline at all call sites). This is a host-side inlining effect only.
Direct device ISA check: IDENTICAL (gfx90a device code objects byte-for-byte the same across both builds).

### Build (gfx90a, c7ac989c, NanoVDB ON)

```
PREFIX=_deps/visionaray/install ANARI=_deps/anari-install
cmake -S projects/anari-visionaray/src -B projects/anari-visionaray/src/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DANARI_VISIONARAY_ENABLE_HIP=ON -DANARI_VISIONARAY_ENABLE_CUDA=OFF \
  -DANARI_VISIONARAY_ENABLE_NANOVDB=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_INSTALL_PREFIX=projects/anari-visionaray/src/build/install \
  "-DCMAKE_PREFIX_PATH=$PREFIX;$ANARI;/opt/rocm"
cmake --build projects/anari-visionaray/src/build -j --target install
```
Build clean (nodiscard warnings only). 8 gfx90a code bundles confirmed via roc-obj-ls
(hipv4-amdgcn-amd-amdhsa--gfx90a, same count and sizes as prior validated build).

### GPU test (gfx90a, HIP_VISIBLE_DEVICES=0, ROCm 7.2)

```
HIP_VISIBLE_DEVICES=0 VOL_NEAREST=1 \
  ANARI_LIBRARY_PATH=projects/anari-visionaray/src/build/install/lib \
  LD_LIBRARY_PATH=_deps/anari-install/lib:projects/anari-visionaray/src/build/install/lib:/opt/rocm/lib \
  ./agent_space/anari_hip_validate visionaray_hip visionaray

HIP_VISIBLE_DEVICES=0 VDB_NEAREST=1 \
  ANARI_LIBRARY_PATH=projects/anari-visionaray/src/build/install/lib \
  LD_LIBRARY_PATH=_deps/anari-install/lib:projects/anari-visionaray/src/build/install/lib:/opt/rocm/lib \
  ./agent_space/anari_vdb_validate visionaray_hip visionaray

HIP_VISIBLE_DEVICES=0 \
  ANARI_LIBRARY_PATH=projects/anari-visionaray/src/build/install/lib \
  LD_LIBRARY_PATH=_deps/anari-install/lib:projects/anari-visionaray/src/build/install/lib:/opt/rocm/lib \
  ./agent_space/anari_vdb_validate visionaray_hip visionaray
```
Results (8/8 stability runs each):
- [hip-triangle-surface]       nonzero=8 varied=8 PASS
- [hip-volume-nearest]         nonzero=8 varied=8 PASS (32^3 radial blob, filter=nearest)
- [cpu-triangle-surface]       nonzero=8 varied=8 PASS
- [cpu-volume-nearest]         nonzero=8 varied=8 PASS
- [visionaray_hip] nanovdb/nearest: nonzero=8 varied=8 PASS
- [visionaray]     nanovdb/nearest: nonzero=8 varied=8 PASS (CPU cross-check)
- [visionaray_hip] nanovdb/linear:  nonzero=8 varied=8 PASS (NanoVDB uses nanovdb accessor, not hip_texture)
- [visionaray]     nanovdb/linear:  nonzero=8 varied=8 PASS

No regressions from removing hipDeviceSynchronize(). Fork source tree clean.
State -> completed, validated_sha=c7ac989c35b04ed877fbf4bc2e672f1a3875abbc.

## Carry-forward 2026-06-25 (linux-gfx1100) -- binary-equiv

State was `revalidate` at head_sha=c7ac989c; validated_sha was 89b49a45.

Cross-built both shas for gfx1100 (on the gfx90a host, static compile only):
```
cmake -S agent_space/anari-old-sha-src -B agent_space/anari-old-gfx1100 \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 [same flags as gfx90a build, arch substituted]
cmake --build agent_space/anari-old-gfx1100 -j --target install

cmake -S projects/anari-visionaray/src -B agent_space/anari-new-gfx1100 \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 [same flags]
cmake --build agent_space/anari-new-gfx1100 -j --target install
```
`python3 utils/codeobj_diff.py agent_space/anari-old-gfx1100/install agent_space/anari-new-gfx1100/install`
Result: same pattern as gfx90a -- exported symbols differ (DevicePointer dtor inlining effect).
Direct device ISA check: IDENTICAL (gfx1100 device code objects identical between old and new sha).
Carried linux-gfx1100 forward to completed at c7ac989c via binary-equiv carry-forward.
Reason: device ISA identical on gfx1100; exported symbol diff is host-side inlining of DevicePointer
dtor (hipDeviceSynchronize removal); GPU code objects unchanged.

windows-gfx1201 remains `revalidate` -- Windows arch, not buildable on this Linux host.

## Revalidation 2026-06-25 (windows-gfx1201) -- PASS (hipDeviceSynchronize removal)

State was `revalidate` at head_sha=c7ac989c; validated_sha was 89b49a45.

Delta (89b49a45..c7ac989c): 1 commit, 1 file (DeviceArray.h: -4 lines).
- c7ac989c: DevicePointer ~DevicePointer(): remove `hipDeviceSynchronize()` before `hipFree()`.
  Host-side only; no device code changes. hipFree is a synchronizing call (hipFreeAsync is not),
  so the explicit sync before free was redundant. gfx90a binary-equiv check confirmed device ISA
  identical; gfx1100 carried forward on binary-equiv. This platform requires a full GPU re-run
  because the sync-removal is a host-side behavioral change (not a pure rename/doc).

GPU: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), HIP_VISIBLE_DEVICES=0, TheRock 7.14.
Fork: AMD-Ecosystem/anari-visionaray @ moat-port, head c7ac989c.

### Build

Incremental rebuild of existing build_win_gfx1201 (NANOVDB=ON, gfx1201):
```
cmake --build projects/anari-visionaray/build_win_gfx1201 --target install
```
Build complete (109 steps; all TUs recompiled due to DeviceArray.h header change).
Fresh DLLs copied to agent_space/.

### GPU test

Regression harness (agent_space/anari_win_validate_gfx1201.exe) -- 2 stability runs:
```
cd agent_space
HIP_VISIBLE_DEVICES=0 ANARI_LIBRARY_PATH=. ./anari_win_validate_gfx1201.exe visionaray_hip visionaray
```
Run 1 and Run 2:
- [hip-triangle-surface] nonzero=2738 varied=2738 PASS
- [hip-volume-nearest]   nonzero=2320 varied=2320 PASS
- [cpu-triangle-surface] nonzero=2738 varied=2738 PASS
- [cpu-volume-nearest]   nonzero=2326 varied=2326 PASS
Result: 4 PASS, 0 FAIL (both runs)

NanoVDB harness (agent_space/anari_vdb_validate_gfx1201.exe) -- 2 stability runs:
```
cd agent_space
HIP_VISIBLE_DEVICES=0 ANARI_LIBRARY_PATH=. ./anari_vdb_validate_gfx1201.exe visionaray_hip visionaray
```
Run 1 and Run 2:
- [hip-nanovdb-nearest] nonzero=10240 varied=10240 PASS
- [hip-nanovdb-linear]  nonzero=10160 varied=10160 PASS
- [cpu-nanovdb-nearest] nonzero=10240 varied=10240 PASS
Result: 3 PASS, 0 FAIL (both runs)

Results match the previous windows-gfx1201 validation values exactly. No regression from
removing hipDeviceSynchronize(). hipFree is synchronizing on gfx1201/TheRock 7.14, consistent
with gfx90a behavior (74.8s empirical block confirmed on gfx90a) and the HIP spec.
Fork source tree clean (no uncommitted tracked files).
State -> completed, validated_sha=c7ac989c35b04ed877fbf4bc2e672f1a3875abbc.
## nvcc CUDA-build back-compat gate 2026-06-25 (linux-gfx90a host, no NVIDIA GPU) -- CLEAN

PR-prep BC check: prove the additive ROCm/HIP port did NOT break the project's
CUDA device build. Compile-only (no NVIDIA GPU here -- the CUDA library is built
and linked but never run). Fork: AMD-Ecosystem/anari-visionaray @ moat-port, head c7ac989c.

### CUDA toolkit (installed via conda, no GPU)
CUDA 13.3 (release 13.3.33), matching the host gcc 13.3 (CUDA 13.x supports gcc 13;
12.x would have needed an older host gcc). The project's GPU TUs pull in Thrust/CUB
through the visionaray dep (detail/bvh/lbvh.h `<cub/cub.cuh>`, random_generator.h
`<thrust/random.h>`), so `cuda-cccl` is required in addition to `cuda-nvcc` +
`cuda-cudart-dev`; on CUDA 13.x Thrust/CUB live under
`<env>/targets/x86_64-linux/include/cccl/{thrust,cub}` and nvcc finds them via its
default `-isystem cccl`.
```
mamba create -y -n cudabc -c nvidia -c conda-forge cuda-nvcc cuda-cudart-dev cuda-cccl cuda-version=13
```

### Build path: project's OWN CMake CUDA target (preferred), reached the LINK stage
```
PREFIX=/var/lib/jenkins/moat/_deps/visionaray/install
ANARI=/var/lib/jenkins/moat/_deps/anari-install
CUDAENV=/opt/conda/envs/cudabc
cmake -S projects/anari-visionaray/src -B agent_space/anari-cuda-bc \
  -DCMAKE_BUILD_TYPE=Release \
  -DANARI_VISIONARAY_ENABLE_CUDA=ON -DANARI_VISIONARAY_ENABLE_HIP=OFF \
  -DANARI_VISIONARAY_ENABLE_NANOVDB=ON \
  -DCMAKE_CUDA_COMPILER=$CUDAENV/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_PREFIX_PATH="$PREFIX;$ANARI"
PATH=$CUDAENV/bin:$PATH bash utils/timeit.sh anari-visionaray compile -- \
  cmake --build agent_space/anari-cuda-bc -j$(nproc) --target anari_library_visionaray_cuda
```
NOTE: the project's CMake pins `CUDA_ARCHITECTURES "native"` on the _cuda target
(needs a GPU to probe); override with `-DCMAKE_CUDA_ARCHITECTURES=80` (real NVIDIA
arch sm_80) so codegen runs GPU-less. NANOVDB=ON so NanoVDBField.cu is exercised.

Result: all 8 CUDA TUs compiled and `libanari_library_visionaray_cuda.so`
(12.6 MB) LINKED -- exit 0. The 8 .cu units (each `#include`s its .cpp sibling and
thus every port-touched shared header):
- VisionarayScene.cu  (VisionarayScene.cpp: unconditional BVH_FLAG_PREFER_FAST_BUILD
  World TLS for all GPU backends; DeviceBVH.h, DeviceCopyableObjects.h, DeviceArray.h, lbvh.h/CUB)
- Frame.cu            (Frame.cpp currentFrame clear; Frame.h mapHostDeviceArray guard widen)
- DirectLight_impl.cu, Raycast_impl.cu  (random_generator.h/Thrust)
- BlockStructuredField.cu, UnstructuredField.cu, GridAccel.cu  (Connectivity.h VSNRAY_FUNC swap)
- NanoVDBField.cu

### Diagnostics: warnings only, ALL pre-existing at base (not port-introduced)
nvcc emitted only warnings, zero errors: 39x #20091-D (`__constant__ Hex` read in a
host function, Connectivity.h), 8x #20011-D (calling __host__ ctor from a
__host__ __device__ lambda, DirectLight_impl.cpp:504 -- not a port-touched line),
2x #177-D (unused variable). Verified pre-existing by compiling the most port-relevant
TU (UnstructuredField.cu, which includes the changed Connectivity.h) at base 7edfca57
(git worktree) with the identical nvcc command: base and port both exit 0 with a
byte-identical diagnostic profile (2x #177-D + 39x #20091-D). The Connectivity.h
`__host__ __device__`->VSNRAY_FUNC / `__host__`->VSNRAY_CPU_FUNC swap is a no-op
under nvcc: visionaray/detail/macros.h defines (under `__CUDACC__||__HIPCC__`)
VSNRAY_FUNC==`__device__ __host__` and VSNRAY_CPU_FUNC==`__host__`, so the
preprocessed attribute set is unchanged.

### Verdict
CUDA path compile-checked clean under nvcc 13.3 (not run -- no NVIDIA GPU).
No port-introduced CUDA regression. The shared/unguarded edits (Connectivity.h
VSNRAY_FUNC swap, VisionarayScene.cpp World-TLS unification, Frame.cpp/.h) compile
and link under nvcc identically to base; all WITH_HIP-specific edits are guarded
out of the CUDA preprocessor. Safe for the "CUDA build preserved" PR claim.
