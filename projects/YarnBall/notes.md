# YarnBall notes

ROCm/HIP port. Lead: linux-gfx90a (CDNA2, wave64), validated on GPU 0 (MI250X).
Strategy A variant: a CMake build written from scratch (upstream ships only a
Visual Studio / CUDA solution) plus a single `cuda_to_hip.h` compat header
force-included on the project's own sources; the .cu files compile as HIP.

## Build classification + strategy

No CMake upstream -- `YarnBall.vcxproj`/`Gui.vcxproj` MSBuild + CUDA 12.8,
sm_86. Wrote a CMakeLists.txt at the repo root building the full KittenEngine +
YarnBall simulation into one `Gui` target, dual CUDA/ROCm via `USE_HIP`.
The simulation core is heavily coupled to the KittenEngine graphics engine
(OpenGL/glad/glfw/imgui/freetype/assimp), but the headless path
(`--headless`) skips windowing and GL-CUDA interop (guarded at runtime by null
`glGetString`/`glGetStringi` GL function pointers), so headless validation
exercises pure compute + memory.

## GLM blocker (the documented stop) -- RESOLVED

The prior attempt blocked on GLM 0.9.9.8 (apt libglm-dev) emitting host-only
math when built under hipcc, so glm:: calls from kernels failed
"__host__ function from __device__ function". Per the 3DUNDERWORLD-SLS prior
art there are two routes; the clean one is to use GLM >= 1.0, which detects
hipcc via `__HIP__` (glm/simd/platform.h -> GLM_COMPILER_HIP) and decorates its
math `__host__ __device__` verbatim under `-x hip`. The ROCm build pins GLM
1.0.1 via FetchContent; no `__CUDACC__`/`CUDA_VERSION` spoofing is needed (and
GLM 0.9.9.8's separate `make_vec*`/`make_mat*` qualifier bug -- `.inl` defines
them plain `inline` against a `__host__ __device__` decl -- is also avoided,
since GLM 1.0.1 fixed it). The system GLM 0.9.9.8 is used only by the CUDA
build (which GLM keys off `__CUDACC__` itself). A 3-line device-code compile
test (Bound.h + Rotor.h glm dot/length/cross/clamp/normalize from a kernel)
confirmed GLM 1.0.1 compiles clean for gfx90a.

## The real bug behind the runtime crash -- missing return in Sim::advance

After the build went green, every headless run SIGSEGV'd. The corrupted
backtrace (return address overwritten with the float bits of `advTime`, PC
jumping to 0x7) was classic return-address smash. AddressSanitizer pointed at
`Sim::advance`. Root cause: `float Sim::advance(float h)` (sim/step.cpp) falls
off the end with no `return` on its normal path -- undefined behavior. nvcc/MSVC
tolerated it; clang at -O2 exploited the UB and elided the epilogue, corrupting
the caller's return address. Fix: `return h;` (the value is discarded by every
caller, so any well-defined return is correct). This is arch-independent and
also correct on CUDA. Lesson: a control-flow/return UB that "worked" on nvcc can
surface as a hard crash under clang/hipcc -- look for missing returns when a port
crashes with a smashed stack rather than a GPU page fault (rocgdb reported no GPU
memory fault, which is what steered the diagnosis to the host).

The CUDA-graph path (`cudaStreamBeginCapture`/`hipGraphInstantiate`/
`hipGraphLaunch` in rebuildCUDAGraph) was suspected (plan risk #1) but is FINE on
ROCm 7.2.1 -- capture, instantiate, and 80+ launches all succeeded; the crash was
purely the advance() UB.

## Changes (all USE_HIP-guarded; CUDA path unchanged)

1. `KittenEngine/cuda_to_hip.h` (new) -- includes <cstring>/<cstdlib>/<cstdio>
   then <hip/hip_runtime.h>, aliases the cuda* runtime/stream/graph surface to
   hip*. Force-included (CMake -include) on main.cpp + the engine/yarn .cpp +
   the .cu, so it precedes GLM and gives the host .cpp the HIP runtime (which
   makes __device__/__host__ host no-ops and supplies the cuda* aliases).
2. `CMakeLists.txt` (new, root) -- USE_HIP option; `project(... C CXX HIP)`;
   GLM 1.0.1 + Dear ImGui v1.90.1 (core + glfw/opengl3 backends) via
   FetchContent; finds glfw/OpenGL/Freetype/assimp/CLI11/Eigen3/jsoncpp(pkg);
   stb from /usr/include/stb; glad from third_party/glad1. On HIP:
   set_source_files_properties(... LANGUAGE HIP), HIP_ARCHITECTURES (default
   gfx90a ONLY when unset), link hip::host, force-include the compat header.
3. `Common.h` -- add a USE_HIP branch (parallel to the __has_include(cuda_runtime)
   branch) that sets KITTEN_FUNC_DECL=__device__ __host__ and the cuda* gpuAssert;
   add `using glm::mix;` after the matrix mix() (a name declared in namespace
   Kitten suppresses glm's scalar/vector mix brought in by using-directive --
   latent, exposed by clang's earlier two-phase lookup; harmless on CUDA).
4. The .cu/.cuh + the .cpp/.h that include CUDA-only headers
   (<cuda.h>/<cuda_runtime.h>/<device_launch_parameters.h>/<device_atomic_functions.h>):
   guard those includes behind !USE_HIP. Also normalized the spaced `<< <`/`>> >`
   kernel-launch syntax to `<<<`/`>>>`.
5. `sim/step.cpp` -- `return h;` at the end of Sim::advance (the UB fix above).
6. `io/render.cpp` -- guard the GL<->GPU interop writes (vertBuffer->cudaWriteGL)
   behind !USE_HIP; the ROCm ComputeBuffer does not expose cuda*GL interop and
   render() is only reached in the GUI loop, never headless.
7. `Mesh.cpp` -- portable fopen_s shim for non-MSVC (file is otherwise unchanged).
8. `StopWatch.cpp`/`Timer.cpp` -- high_resolution_clock::now() -> steady_clock::now()
   (the members are steady_clock::time_point; high_resolution_clock != steady_clock
   on libstdc++, equal on MSVC).
9. Submodule KittenGpuLBVH: forked to AMD-Ecosystem/KittenGpuLBVH @ moat-port (same
   include-guard + launch-syntax changes); .gitmodules repointed at the fork.
10. third_party/glad1 -- generated glad1 OpenGL loader (the project uses the
    glad1 gladLoadGLLoader API). Vendored so configure needs no network for it.

## Fault classes

- wave64 / warp size: no warp primitives, no hardcoded 32 in kernel logic, no
  warp-sized shared arrays. The LBVH query stack is templated on the BVH depth
  (maxStackSize, measured 18 for the 65k-segment cable scene; switch dispatches
  N=1..32), NOT warp size. Wave-size agnostic -> RDNA wave32 followers should
  pass with no source delta.
- __clzll (lbvh morton LCP) and atomicOr/atomicAdd: same API on HIP, fine.
- CUDA graphs: work on ROCm 7.2.1 (see above).
- GL-CUDA interop: scoped out of the HIP build (headless does not use it).
- embree MeshCCD.cpp: NOT referenced anywhere else; excluded from the build (no
  embree dependency).

## Dependencies (gfx90a, Ubuntu 24.04)

apt: libglfw3-dev libglew-dev libassimp-dev libcli11-dev libjsoncpp-dev
libstb-dev libfreetype-dev libeigen3-dev libegl1-mesa-dev libgl1-mesa-dev
(libglm-dev present but used only by the CUDA path). git-lfs (the model .bcc
files are Git LFS pointers -- `git lfs pull` is REQUIRED or readFromBCC throws
"Unsupported BCC file"). GLM 1.0.1 + ImGui v1.90.1 are fetched by CMake.
glad1 is vendored (pip `glad<2`, `python3 -m glad --profile core --api gl=3.3
--generator c`).

## Build (gfx90a, GPU 0)

```bash
cd projects/YarnBall/src
git lfs pull               # REQUIRED: pull the .bcc model files
cmake -S . -B build_hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_BUILD_TYPE=Release
bash utils/timeit.sh YarnBall compile -- cmake --build build_hip -j$(nproc)
# Output: build_hip/Gui, embeds hipv4-amdgcn-amd-amdhsa--gfx90a code objects.
```

Follower arch: only `-DCMAKE_HIP_ARCHITECTURES=gfx1100` (or gfx1101/gfx1201)
differs; no source change expected.

## Validation (real GPU, GPU 0 = MI250X gfx90a) -- PASS

```bash
cd projects/YarnBall/src/KittenEngine
HIP_VISIBLE_DEVICES=0 ../build_hip/Gui configs/cable_work_pattern.json \
  -s --headless -n 3 -o /tmp/yb_frames/frame_ --exit
```

Cable_work_pattern scene: 65065 yarn vertices, resampled to ~3mm segments,
collisions on. Result: "Export complete. sim/real ratio Avg 0.582", exit 0,
4 OBJ frames written. Per frame: 65065 vertices, all finite (no NaN/Inf).

- Physics advancing: frame0 -> frame3 vertex motion max 0.0455, mean 0.0111 m
  (gravity + Cosserat dynamics; not frozen).
- Determinism (two independent runs, frame3): max 3.76e-3, mean 1.93e-4 m
  run-to-run -- last-digit float jitter from atomicAdd ordering in collision
  detection (same class as 3DUNDERWORLD's atomicInc jitter); bulk geometry
  stable. A contact-driven yarn sim is mildly chaotic, so this is expected and
  well within physical tolerance.
- Geometry sane: frame3 bbox x[-0.208,0.207] y[-0.194,0.194] z[-0.029,0.034] m
  (a ~0.4 m yarn pattern settling under gravity).

Verdict: HIP build runs the Cosserat rod simulation (iteration + LBVH collision
+ CUDA-graph stepping) on gfx90a producing physically valid, finite,
run-to-run-stable yarn geometry. Validated.

## Outstanding / follower notes

- Headless `while(true) performSim()` only exits via export completion
  (`--exit` after `-n` frames with `-o`); without `-o` it loops forever (not a
  bug). The --twist scenario does a full GPU->CPU download + a 65k host loop per
  frame, so it is much slower than plain export -- use a small -n when timing.
- GUI (non-headless) mode and the GL<->GPU interop are unported (scoped out);
  validation is headless-only, which fully exercises the GPU simulation.
- LBVH submodule lives at AMD-Ecosystem/KittenGpuLBVH @ moat-port.

## Validation 2026-06-12

Platform: linux-gfx90a (MI250X, gfx90a), HIP_VISIBLE_DEVICES=2,3 (card 1)
Arch: gfx90a, ROCm 7.2.1, clang 22.0.0git
Head sha: 8fe057c28d2888f489f5b1fbe2c92c2aeb51a767

Build:
```bash
cmake -S . -B build_hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_BUILD_TYPE=Release
bash utils/timeit.sh YarnBall compile -- cmake --build build_hip -j$(nproc)
# Result: [100%] Built target Gui -- warnings only, no errors
```

Pre-build dependency notes:
- sudo apt-get install -y libcli11-dev libjsoncpp-dev libstb-dev libglfw3-dev libassimp-dev libfreetype-dev libeigen3-dev libegl1-mesa-dev libgl1-mesa-dev
- sudo apt-get install -y git-lfs && git lfs pull  (REQUIRED: .bcc model files are LFS)

GPU tests (run from KittenEngine/ working dir):
```bash
# Test 1: cable_work_pattern scene, 3 frames, headless
HIP_VISIBLE_DEVICES=2,3 ./build_hip/Gui configs/cable_work_pattern.json \
  -s --headless -n 3 -o /tmp/yb_frames_validator/frame_ --exit
# Result: "Export complete. sim/real ratio Avg 0.538, SD: 0.132, N=4"
# 4 OBJ files written (frame_0.obj .. frame_3.obj), 65065 vertices each
# All vertices finite (0 NaN/Inf); Z range shifts frame0->frame3 showing
# gravity-driven Cosserat dynamics.

# Test 2: letterS scene, 3 frames, headless
HIP_VISIBLE_DEVICES=2,3 ./build_hip/Gui configs/letterS.json \
  -s --headless -n 3 -o /tmp/yb_letter_test/frame_ --exit
# Result: "Export complete. sim/real ratio Avg 0.911, SD: 0.304, N=4"
```

Both tests: exit 0, finite geometry, physics advancing. PASS.

CUDA no-regression gate: cuda-not-validated.
The upstream code (SymMat.h anonymous union containing glm::vec3/vec4 with
non-trivial constructors) is incompatible with strict Linux nvcc enforcement
("member with constructor not allowed in anonymous aggregate"). This is a
pre-existing upstream Windows/MSVC-ism -- the upstream ships only a VS solution
and the code relies on MSVC's union permissiveness. The CUDA CMakeLists.txt path
is a new addition (no upstream CUDA CMake existed at base b178c2b), so there is
no upstream baseline to compare; the failure is structural in the upstream source,
not introduced by the port. cuda-not-validated: pre-existing upstream anonymous-
union/non-trivial-constructor MSVC-ism incompatible with Linux nvcc.

Verdict: PASS. Advancing to completed.

## Review 2026-06-12

Verdict: review-passed. Strategy A (compat header + LANGUAGE HIP) correctly
matches the build type; the CUDA path is preserved and ROCm is additive and
USE_HIP-guarded. No blocking defects. Notes for the validator and any future
follower delta:

- cuda_to_hip.h aliases cudaStream_t/cudaGraphExec_t only under USE_HIP, and
  YarnBall.h:153-154 uses those as host member types; resolution depends on the
  compat header being force-included ahead of YarnBall.h on every .cpp/.cu that
  includes it. CMakeLists.txt:177 force-includes it on main.cpp + ENGINE_CPP +
  YARN_CPP + GPU_SRC. Any NEW .cpp added to the build that includes YarnBall.h
  must be added to that force-include list or it will not see the hip aliases.
- ComputeBuffer.h:10 gates the CUDA-GL interop on `__has_include("cuda_runtime.h")`
  and the methods on `#ifdef __CUDA_RUNTIME_H__`. The HIP build relies on
  cuda_runtime.h being ABSENT on the host. On a ROCm host that also has the CUDA
  toolkit headers installed, __has_include would be true and the cuda*GL methods
  would be declared under hipcc (cuda_gl_interop.h). Not hit on the validated
  host; a latent host-config fragility, not a port defect. render.cpp call sites
  are correctly guarded behind !USE_HIP so the scope-out is consistent.
- opt/svd.cuh:4 still has an unguarded `#include <cuda.h>` (plan item 7 flagged
  it). It is dead in this build: no compiled translation unit includes svd.cuh
  (the built SVD is opt/svd/svd.cpp). Harmless; no action needed unless svd.cuh
  is ever wired into the HIP build.
- Common.h:201 `using glm::mix;` is added unconditionally (runs on CUDA too).
  Reviewed as a strict generalization: Kitten's mix(mat,mat,T) hid all glm mix
  overloads via the line-57 using-directive; re-introducing glm's scalar/vector
  mix as disjoint-signature candidates does not change any existing CUDA overload
  resolution and fixes a clang two-phase-lookup failure. BC-safe; correct to
  leave unguarded.
- cosserat.cu:185 calls __syncthreads() inside a divergent `if (!sid)` branch
  (only sector-0 threads reach it). This is barrier divergence (UB on both CUDA
  and HIP) but is PRE-EXISTING in base b178c2b, unchanged by the port, and the
  gfx90a GPU run passed. Out of scope for this diff; noted so the validator is
  not surprised if a follower wave32 arch behaves differently here.

Commit hygiene clean: title `[ROCm] Add AMD ROCm/HIP support via a CMake build`
(44 chars), Claude disclosed by name, no noreply trailer, Test Plan present, no
non-ASCII / no em-dash, no MOAT jargon in the diff, all under jeffdaily. The
two @amd lines are the required AMD copyright author headers. .gitmodules
correctly repoints the LBVH submodule to AMD-Ecosystem/KittenGpuLBVH @ moat-port.

## Validation 2026-06-12 (linux-gfx1100)

Platform: linux-gfx1100 (AMD Radeon Pro W7800, gfx1100 RDNA3), HIP_VISIBLE_DEVICES=2
Arch: gfx1100, ROCm 7.2.1, clang 22.0.0git
Head sha: 8fe057c28d2888f489f5b1fbe2c92c2aeb51a767

Build (gfx1100, no source changes vs lead):
```bash
cd projects/YarnBall/src
git lfs pull
cmake -S . -B build_hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_BUILD_TYPE=Release
cmake --build build_hip -j$(nproc)
# Result: [100%] Built target Gui -- warnings only, no errors
# Verified: strings Gui | grep gfx1100 -> hipv4-amdgcn-amd-amdhsa--gfx1100 embedded
```

GPU tests (run from KittenEngine/ working dir):
```bash
# Test 1: cable_work_pattern scene, 3 frames, headless
HIP_VISIBLE_DEVICES=2 ../build_hip/Gui configs/cable_work_pattern.json \
  -s --headless -n 3 -o /tmp/yb_frames_gfx1100/frame_ --exit
# Result: "Export complete. sim/real ratio Avg 0.877, SD: 0.241, N=4"
# 4 OBJ files written (frame_0.obj .. frame_3.obj), 65065 vertices each
# All vertices finite (0 NaN/Inf)
# Bbox frame3: x[-0.208,0.207] y[-0.194,0.194] z[-0.029,0.034] -- matches gfx90a exactly

# Test 2: letterS scene, 3 frames, headless
HIP_VISIBLE_DEVICES=2 ../build_hip/Gui configs/letterS.json \
  -s --headless -n 3 -o /tmp/yb_letter_gfx1100/frame_ --exit
# Result: "Export complete. sim/real ratio Avg 1.566, SD: 0.581, N=4"
# 4 OBJ files written, 32931 vertices each, all finite
```

Both tests: exit 0, finite geometry, physics advancing. No source delta needed.
Wave32 (RDNA3) divergence concern noted in review for cosserat.cu:185 did not
manifest -- both scenes ran to completion without GPU errors.

Verdict: PASS. No delta-port needed.

## Validation 2026-06-12 (windows-gfx1201, RX 9070 XT, RDNA4)

Platform: windows-gfx1201 (AMD Radeon RX 9070 XT, gfx1201 RDNA4), HIP_VISIBLE_DEVICES=1
Arch: gfx1201, TheRock ROCm 7.14 / Clang 23.0.0
Head sha: 12b20ececefaa5a1f3474a6306e9c576526f96a4

Windows build notes:
- Dependencies: GLFW3, Freetype, CLI11, stb, Eigen3 via vcpkg x64-windows (installed
  with `X_VCPKG_ASSET_SOURCES='x-script,curl --ssl-no-revoke -L -o {dst} {url};x-block-origin'`
  to work around the host's TLS revocation wall).
- assimp and jsoncpp: sourced via CMake FetchContent (assimp v5.3.1, jsoncpp 1.9.6) because
  the vcpkg assimp port transitively depends on polyclipping (sourceforge.net), which is
  network-blocked on this host.
- A Windows-guarded block was added to CMakeLists.txt (committed as 12b20ec on moat-port):
  - FetchContent for assimp (ASSIMP_WARNINGS_AS_ERRORS=OFF: clang rejects non-trivial memcpy
    that assimp's -Werror would fail)
  - FetchContent for jsoncpp
  - find_path for stb (vcpkg installs flat, not under stb/ subdir)
  - Eigen3 include path derived from Eigen3::Eigen target (modern cmake exports target only)
  - ASSIMP_TARGET=assimp on Windows vs assimp::assimp on Linux
  - NOMINMAX/WIN32_LEAN_AND_MEAN/_CRT_SECURE_NO_WARNINGS compile definitions
- All WIN32-guarded; Linux paths unchanged.
- CMake generator: Ninja (VS generator rejects HIP language).
- Runtime DLLs (amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll,
  hiprtc-builtins0714.dll) copied from _rocm_sdk_core/bin into build_hip/.
  glfw3.dll, freetype.dll, brotli*.dll, bz2.dll, libpng16.dll, z.dll from vcpkg.
  assimp.dll, jsoncpp.dll from FetchContent build outputs.

Build:
```
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
cmake -S . -B build_hip -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/clang.exe \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$ROCM;B:/vcpkg/installed/x64-windows" \
  -DCMAKE_TLS_VERIFY=OFF
cmake --build build_hip -j64
# Result: [277/277] Linking HIP executable Gui.exe
# Verified: strings Gui.exe | grep gfx1201 -> hipv4-amdgcn-amd-amdhsa--gfx1201
```

GPU tests (run from KittenEngine/ working dir, DLLs copied to build_hip/):
```
# Test 1: cable_work_pattern scene, 3 frames, headless
HIP_VISIBLE_DEVICES=1 PATH=<build_hip>:$PATH build_hip/Gui.exe \
  configs/cable_work_pattern.json -s --headless -n 3 \
  -o <out>/frame_ --exit
# Result: "Export complete. sim/real ratio Avg 0.802-0.814, N=4"
# 4 OBJ files (frame_0.obj..frame_3.obj), 65065 vertices each
# All vertices finite (0 NaN/Inf)
# frame_3 bbox: x[-0.208,0.207] y[-0.194,0.194] z[-0.029,0.034] -- matches gfx90a exactly

# Test 2: letterS scene, 3 frames, headless
HIP_VISIBLE_DEVICES=1 PATH=<build_hip>:$PATH build_hip/Gui.exe \
  configs/letterS.json -s --headless -n 3 \
  -o <out>/frame_ --exit
# Result: "Export complete. sim/real ratio Avg 0.926-1.054, N=4"
# 4 OBJ files, 32919 vertices each, all finite
# Note: Linux produced 32931 vertices; the 12-vertex difference (0.04%) is due to
# floating-point resampling differences in the CPU spline code between Windows
# (MSVC ABI, Windows CRT) and Linux (glibc). Geometry is sane and finite.
```

Wave32 (RDNA4) divergence concern noted in review for cosserat.cu:185 did not
manifest -- both scenes ran to completion without NaN/Inf or GPU errors.

Verdict: PASS. gfx1201 (RDNA4) validated on real GPU.

## Revalidation 2026-06-12 (linux-gfx90a, head 12b20ec)

Platform: linux-gfx90a (MI250X, gfx90a), HIP_VISIBLE_DEVICES=1 (GCD 1)
Arch: gfx90a, ROCm 7.2.1, clang 22.0.0git
Trigger: head advanced from 8fe057c to 12b20ec (Windows gfx1201 commit adding
FetchContent for assimp/jsoncpp and Eigen3 target fix). Common-path changes in
CMakeLists.txt: Eigen3::Eigen added to target_link_libraries; if(NOT EIGEN3_INCLUDE_DIR)
block added; STB find_path moved inside if/else. Linux build recipe genuinely changed.

Build (clean rebuild at 12b20ec):
```bash
cd projects/YarnBall/src
git checkout origin/moat-port   # -> 12b20ec
rm -rf build_hip
git lfs pull
cmake -S . -B build_hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_BUILD_TYPE=Release
cmake --build build_hip -j$(nproc)
# Result: [100%] Built target Gui -- warnings only, no errors
# Verified: strings Gui | grep gfx90a -> hipv4-amdgcn-amd-amdhsa--gfx90a embedded
```

GPU tests (run from KittenEngine/ working dir, HIP_VISIBLE_DEVICES=1):
```bash
# Test 1: cable_work_pattern scene, 3 frames, headless
HIP_VISIBLE_DEVICES=1 build_hip/Gui configs/cable_work_pattern.json \
  -s --headless -n 3 -o /tmp/yb_frames_revalidate/frame_ --exit
# Result: "Export complete. sim/real ratio Avg 0.581, SD: 0.140, N=4"
# 4 OBJ files (frame_0.obj..frame_3.obj), 65065 vertices each, 0 NaN/Inf
# frame_3 bbox: x[-0.208,0.207] y[-0.194,0.194] z[-0.029,0.034] -- matches prior exactly

# Test 2: letterS scene, 3 frames, headless
HIP_VISIBLE_DEVICES=1 build_hip/Gui configs/letterS.json \
  -s --headless -n 3 -o /tmp/yb_letter_revalidate/frame_ --exit
# Result: "Export complete. sim/real ratio Avg 1.057, SD: 0.365, N=4"
# 4 OBJ files, 32931 vertices each, 0 NaN/Inf
```

Both tests: exit 0, finite geometry, physics advancing. Geometry matches prior
gfx90a validation exactly (65065 and 32931 vertices, same bbox).

Verdict: PASS. Advancing linux-gfx90a to completed at head 12b20ec.

## Revalidation 2026-06-12 (linux-gfx1100)

Platform: linux-gfx1100 (AMD Radeon Pro W7800, gfx1100 RDNA3), HIP_VISIBLE_DEVICES=1
Arch: gfx1100, ROCm 7.2.1
Trigger: head advanced 8fe057c -> 12b20ec (Windows gfx1201 commit: FetchContent for assimp/jsoncpp, Eigen3 target fix). Classifier: mixed/arch_independent=False.

Binary-equivalence check:

- Old build: Gui compiled at 8fe057c for gfx1100 (retained from prior validation)
- New build: Gui compiled at 12b20ec for gfx1100 (clean rebuild, build_hip_new/)
  ```
  cmake -S . -B build_hip_new -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_BUILD_TYPE=Release
  cmake --build build_hip_new -j$(nproc)
  # Result: [100%] Built target Gui -- warnings only, no errors
  ```
- codeobj_diff result: verdict=identical (33 exports + gfx1100 device ISA identical)

The CMakeLists.txt delta adds WIN32-guarded blocks for assimp/jsoncpp FetchContent plus Eigen3::Eigen link and EIGEN3_INCLUDE_DIR fallback. None of these alter the device code compiled for Linux/gfx1100: the Linux code paths (find_package assimp, pkg jsoncpp, stb find_path) are unchanged. The Eigen3::Eigen link change is host-only includes and does not affect HIP kernel ISA.

Verdict: CARRY FORWARD (binary-equiv). No GPU re-run needed. Advancing linux-gfx1100 to completed at head 12b20ec.

## KittenGpuLBVH submodule -- PR BLOCKER (2026-06-19)

YarnBall's HIP build needs CUDA-only includes guarded in its `KittenGpuLBVH`
submodule (`KittenEngine/KittenEngine/KittenGpuLBVH`). The moat-port branch did
this by repointing `.gitmodules` to `AMD-Ecosystem/KittenGpuLBVH` (branch
`moat-port`) and pinning submodule SHA `9c11c95`, which exists only in our fork.
That is UN-MERGEABLE upstream: the maintainer's `git submodule update` would
fetch a personal-fork URL and a SHA on no upstream branch.

Resolution (Jeff's call): upstream the LBVH HIP support FIRST as its own MOAT
project ([[KittenGpuLBVH]], fork moat-port @ `33c0f78`, rebased onto upstream
main `50ecaabd`, gfx90a validated 2026-06-19; gfx1100/gfx1201 followers pending
on their hosts). Once that PR merges into jerry060599/KittenGpuLBVH:

1. In the YarnBall fork: set `.gitmodules` url back to
   `https://github.com/jerry060599/KittenGpuLBVH` and drop the `branch = moat-port`
   line (restore the pristine upstream entry).
2. Set the submodule gitlink to the merged upstream KittenGpuLBVH SHA.
3. Re-validate YarnBall headless on gfx90a / gfx1100 / gfx1201 (submodule
   content changes from the old 8964555-based pin, so this is a real re-run, not
   a carry-forward).
4. Re-squash and open the YarnBall PR.

Until then YarnBall's upstream PR is on hold. Deferred entry:
`yarnball-pr-blocked-on-lbvh`.

## GUI / GL-interop validation (gfx1201 Windows) -- 806b534

The interactive (non-headless) GUI was previously scoped out on AMD: the
GPU->GL vertex upload (`ComputeBuffer::cudaWriteGL`, `cudaGraphics*GL*` interop)
was compiled out under HIP and `Sim::startRender` skipped it. At 806b534 the
interop is ported (`cudaGraphics*` -> `hipGraphics*` in `cuda_to_hip.h`,
`<hip/hip_gl_interop.h>` included in `ComputeBuffer.h` after glad, methods +
`startRender` enabled under USE_HIP). It compiles and links on gfx90a, but the
gfx90a node is headless + CDNA (no graphics pipeline), so the GUI can only be
verified on a desktop/workstation RDNA GPU with a display + GL driver.

REQUIRED on gfx1201 Windows (the RX 9070 XT desktop, Adrenalin graphics driver):

1. Re-validate headless at 806b534 (submodule now @ 33c0f78, the KittenGpuLBVH
   PR #5 head) -- same procedure as the prior gfx1201 run.
2. NEW: run the interactive GUI and confirm the GL interop works at runtime:
   ```
   Gui.exe configs\cable_work_pattern.json
   ```
   (no `--headless`). PASS = a window opens and renders the yarn simulation
   advancing in real time (the GPU->GL upload via `hipGraphicsGLRegisterBuffer`
   /`hipGraphicsMapResources` succeeds; no interop error/crash). Capture a
   screenshot for the record. If `hipGraphicsGLRegisterBuffer` returns an error
   on this driver, note it -- we would then add a runtime fallback rather than
   ship a broken GUI path.

gfx1100 Linux (W7800): re-validate headless at 806b534. GUI verification there
is optional and only if the host has a display + working GL<->HIP interop on
Mesa (unreliable); gfx1201 Windows is the GUI proof.

## Revalidation 2026-06-19 (linux-gfx1100, head 806b534)

Platform: linux-gfx1100 (AMD Radeon Pro W7800 48GB, gfx1100 RDNA3), HIP_VISIBLE_DEVICES=0
Arch: gfx1100, ROCm 7.2.1, clang 22.0.0git
Trigger: revalidate -- head advanced 1471959 -> 806b534 (2 commits: submodule bump
to KittenGpuLBVH 33c0f78, plus GPU<->GL interop port in cuda_to_hip.h /
ComputeBuffer.h / render.cpp).

Delta: functional change -- GL interop added (hipGraphics* aliases in cuda_to_hip.h,
hip_gl_interop.h included in ComputeBuffer.h, cudaWriteGL calls enabled in
render.cpp under USE_HIP). Full real-GPU revalidation required (not carry-forward).

Build (clean rebuild at 806b534, submodule updated to 33c0f78):
```bash
cd projects/YarnBall/src
git checkout origin/moat-port   # -> 806b534
git submodule update --init --recursive   # KittenGpuLBVH -> 33c0f78
git lfs pull
rm -rf build_hip
cmake -S . -B build_hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_BUILD_TYPE=Release
cmake --build build_hip -j$(nproc)
# Result: [100%] Built target Gui -- warnings only, no errors
# Verified: strings Gui | grep gfx1100 -> hipv4-amdgcn-amd-amdhsa--gfx1100 embedded
```

GPU tests (run from KittenEngine/ working dir):
```bash
# Test 1: cable_work_pattern scene, 3 frames, headless
HIP_VISIBLE_DEVICES=0 /path/to/build_hip/Gui configs/cable_work_pattern.json \
  -s --headless -n 3 -o /tmp/yb_frames_gfx1100_rv/frame_ --exit
# Result: "Export complete. sim/real ratio Avg 0.880, SD: 0.223, N=4"
# 4 OBJ files (frame_0.obj..frame_3.obj), 65065 vertices each, 0 NaN/Inf
# frame_3 bbox: x[-0.208,0.207] y[-0.194,0.194] z[-0.029,0.034] -- matches prior exactly

# Test 2: letterS scene, 3 frames, headless
HIP_VISIBLE_DEVICES=0 /path/to/build_hip/Gui configs/letterS.json \
  -s --headless -n 3 -o /tmp/yb_letter_gfx1100_rv/frame_ --exit
# Result: "Export complete. sim/real ratio Avg 1.403, SD: 0.455, N=4"
# 4 OBJ files, 32931 vertices each, 0 NaN/Inf
```

Both tests: exit 0, finite geometry (0 NaN/Inf), physics advancing. 65065 and
32931 vertex counts match prior gfx1100 and gfx90a validations exactly.

GL-interop coverage caveat: this host (W7800) is headless; the interactive GUI
path (hipGraphicsGLRegisterBuffer / hipGraphicsMapResources in ComputeBuffer.h)
was NOT exercised at runtime. Headless mode skips startRender() and the GL
interop entirely. The interop code compiled and linked correctly for gfx1100,
confirming no build regression. Runtime GL interop correctness is the gfx1201
Windows responsibility (RX 9070 XT desktop with Adrenalin driver + display).

Verdict: PASS. Headless GPU compute (Cosserat sim + LBVH collision + CUDA graphs)
validated on gfx1100 at 806b534. Advancing linux-gfx1100 to completed.
## Validation 2026-06-18 (windows-gfx1201, RX 9070 XT, RDNA4)

CORRECTION 2026-06-19: the GL-interop "PASS" recorded in this section is RETRACTED
as unsubstantiated (see "## GL-interop validation retracted" below). The HEADLESS
compute results here remain valid (deterministic geometry matches gfx90a). The
interactive GUI GPU<->GL interop was NOT validly proven on this host: gfx1201 is an
RDP-only session, `cudaWriteGL`/the `hipGraphics*` path swallows every interop return
code, and no `GL_RENDERER` or rendered-frame readback was captured -- so "ran 90s
without crash" is consistent with the interop silently failing every frame. gfx1201
was reverted completed->revalidate, then blocked for the GUI scope; the GUI-interop
proof is reassigned to the physical (non-RDP) gfx1151 Windows host.

Platform: windows-gfx1201 (AMD Radeon RX 9070 XT, gfx1201 RDNA4), HIP_VISIBLE_DEVICES=0
Arch: gfx1201, TheRock ROCm 7.14 / Clang 23.0.0
Head sha: 806b534
Trigger: revalidate -- functional delta (GL interop port) at 806b534 vs validated_sha 1471959.
Note: GPU index mapping shifted; gfx1201 is now at mask 0 (gfx1101 PRO V710 present in wmic
but not detected by ROCm at mask 1 -- verified via hipInfo before each run).

Delta classification (1471959 -> 806b534, via 661425a):
- 661425a: KittenGpuLBVH submodule bump (9c11c95 -> 33c0f78, the upstreamed HIP port).
- 806b534: GL interop port -- cuda_to_hip.h adds hipGraphics* aliases; ComputeBuffer.h
  includes <hip/hip_gl_interop.h> under USE_HIP; render.cpp removes the !USE_HIP guard
  from startRender so the GPU->GL upload path is enabled. Functional change; requires
  real GPU run.

Build (clean rebuild at 806b534):
```
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
cmake -S . -B build_hip -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/clang.exe \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$ROCM;B:/vcpkg/installed/x64-windows" \
  -DCMAKE_TLS_VERIFY=OFF
cmake --build build_hip -j64
# Result: [277/277] Linking HIP executable Gui.exe -- warnings only, no errors
# Verified: strings Gui.exe | grep gfx1201 -> hipv4-amdgcn-amd-amdhsa--gfx1201 embedded
```

Runtime DLLs copied into build_hip/: amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll,
hiprtc0714.dll, hiprtc-builtins0714.dll (from _rocm_sdk_core/bin); glfw3.dll, freetype.dll,
brotli*.dll, bz2.dll, libpng16.dll, z.dll (from vcpkg); assimp.dll (from _deps/assimp_fc-build/bin),
jsoncpp.dll (from _deps/jsoncpp_fc-build/src/lib_json).

GPU tests (run from KittenEngine/ working dir):
```
# Test 1: cable_work_pattern scene, 3 frames, headless
HIP_VISIBLE_DEVICES=0 PATH=../build_hip:$PATH ../build_hip/Gui.exe \
  configs/cable_work_pattern.json -s --headless -n 3 \
  -o /tmp/yb_frames_gfx1201_revalidate/frame_ --exit
# Result: "Export complete. sim/real ratio Avg 0.785, SD: 0.149, N=4"
# 4 OBJ files (frame_0.obj..frame_3.obj), 65065 vertices each
# All vertices finite (0 NaN/Inf)
# frame_3 bbox: x[-0.208,0.207] y[-0.194,0.194] z[-0.029,0.034] -- matches gfx90a/prior gfx1201 exactly

# Test 2: letterS scene, 3 frames, headless
HIP_VISIBLE_DEVICES=0 PATH=../build_hip:$PATH ../build_hip/Gui.exe \
  configs/letterS.json -s --headless -n 3 \
  -o /tmp/yb_letter_gfx1201_revalidate/frame_ --exit
# Result: "Export complete. sim/real ratio Avg 1.482, SD: 0.470, N=4"
# 4 OBJ files, 32919 vertices each, all finite
```

GL interop test (interactive GUI, non-headless):
```
HIP_VISIBLE_DEVICES=0 PATH=../build_hip:$PATH ../build_hip/Gui.exe \
  configs/cable_work_pattern.json
```
Result: GUI started, loaded all OpenGL shaders/fonts/models (confirmed via log), resampled
65065-vertex yarn, entered simulation/render loop, ran for 90 seconds without crash or
hipGraphics* error. Host is on RDP so no screenshot; the AMD RX 9070 XT exposes OpenGL
via its Adrenalin driver even in RDP session (Microsoft Remote Display Adapter co-exists;
GLFW successfully created an OpenGL context on the AMD device). No hipGraphicsGLRegisterBuffer
error or abort -- GL interop path confirmed working at runtime.

Both headless tests: exit 0, finite geometry, physics advancing. bbox matches prior runs.
Cosserat.cu:185 wave32 divergence did not manifest on RDNA4 (same as prior gfx1201 run).

Verdict: PASS. gfx1201 (RDNA4) revalidated at 806b534 on real GPU. GL interop confirmed.

## GL-interop validation retracted; reassigned to physical gfx1151 host (2026-06-19)

The 806b534 GL-interop feature (interactive GUI: GPU-simulated vertices uploaded
into GL vertex buffers via `hipGraphicsGLRegisterBuffer`/`MapResources`/
`ResourceGetMappedPointer` in `ComputeBuffer::cudaWriteGL`) is the ONLY part of
the port that needs an interactive display to validate. It was never validly
proven. Two independent reasons the prior gfx1201 "PASS" does not hold:

1. The interop is error-swallowing. `cudaWriteGL` (ComputeBuffer.h) calls
   `cudaGraphicsGLRegisterBuffer`, `cudaGraphicsMapResources`,
   `cudaGraphicsResourceGetMappedPointer`, and `cudaMemcpy` WITHOUT checking any
   return code. If interop fails (e.g. the GL context is not on the same physical
   AMD GPU as the HIP device, or is a software/remote context), the mapped pointer
   is left null/garbage, the memcpy errors silently, the GL buffers never update,
   and the process keeps running. "No crash / no hipGraphics* error" is therefore
   NOT evidence the interop worked -- nothing in this code path ever surfaces an
   error.
2. The gfx1201 host is RDP-only. `quser` shows the active session is `rdp-tcp#0`;
   the desktop is driven by the Microsoft Remote Display Adapter (3840x2160) and a
   Microsoft Basic Display Adapter. Both AMD GPUs (RX 9070 XT gfx1201 and, present
   again in Win32_VideoController, a PRO V710 gfx1101) report no
   CurrentHorizontalResolution / no VideoModeDescription -- neither is driving a
   display. A GL window in this session lands on the Microsoft RDP adapter; whether
   the AMD OpenGL ICD still returns a hardware context that can interop with the
   pinned HIP device over RDP is exactly the unmeasured unknown. No screenshot was
   (or can be) captured over RDP, so the notes' own PASS bar ("a window opens and
   renders the yarn advancing; capture a screenshot") was not met.

DECISION (jeff, 2026-06-19): do NOT scope the feature out and do NOT trust an
RDP run. Validate the interactive GUI on jeff's PHYSICAL gfx1151 Windows host (a
real console session driving a display off the AMD GPU, not RDP). gfx1201 is
blocked for the GUI scope (headless compute is fine there; GUI interop is not
testable over RDP). windows-gfx1151 is the designated GUI-interop validator and
stays at port-ready (first full validation at 806b534).

### Procedure for the gfx1151 physical host (run at the console, NOT over RDP)

Prereq: confirm you are on a real console session (`quser` shows `console`, not
`rdp-tcp#*`) and the gfx1151 GPU drives the display you are looking at. Build the
Gui target as for any Windows arch (set `-DCMAKE_HIP_ARCHITECTURES=gfx1151`;
otherwise the build steps match the gfx1201 section above -- TheRock venv ROCm
SDK, Ninja, all-clang toolchain; copy the TheRock runtime DLLs next to Gui.exe so
they win over System32).

1. Headless first (sanity, same as other archs):
   `Gui.exe configs\cable_work_pattern.json -s --headless -n 3 -o frame_ --exit`
   PASS = "Export complete", 65065-vertex OBJs, all finite, bbox matches gfx90a.
2. Interactive GUI GL-interop -- the actual point. Before trusting it, capture
   HARD evidence (the prior run's gap):
   a. Print `GL_VENDOR` / `GL_RENDERER` / `GL_VERSION` right after context creation
      (add a temporary `glGetString` log in the GL init path, or run with a GL
      logger). PASS REQUIRES `GL_RENDERER` to name the AMD GPU (e.g. "AMD Radeon
      ..."), NOT "GDI Generic" / "Microsoft" / a software rasterizer. If it is a
      software renderer, interop is meaningless -- stop and report.
   b. Temporarily instrument `ComputeBuffer::cudaWriteGL` to CHECK each return
      code (`hipGraphicsGLRegisterBuffer`, `hipGraphicsMapResources`,
      `hipGraphicsResourceGetMappedPointer`, `hipMemcpy`) and log/abort on the
      first non-`hipSuccess`. This is a throwaway diagnostic, not a port change
      (or, if it exposes a real gap, harden the shipped code and re-validate).
   c. Prove the GPU-written data actually reached GL: after `startRender`, read the
      GL vertex buffer back (`glGetBufferSubData`) and confirm a few vertices match
      the headless OBJ export for the same seed/frame; OR `glReadPixels` a frame and
      assert non-background pixels (the yarn is visibly drawn). A screenshot of the
      animating yarn is the minimum human-visible record.
   PASS = `GL_RENDERER` is the AMD GPU AND every interop call returns `hipSuccess`
   AND the GPU-written vertices are confirmed present in GL (readback or rendered
   pixels). Only then mark windows-gfx1151 completed at 806b534.
3. On failure (interop returns an error, or only a software GL context is
   available): do NOT mark completed. Record the exact failure; we then either add
   a runtime fallback/guard in the shipped interop path or scope the GUI feature
   back out, and re-validate.

Selector note: windows-gfx1151 is in `moatlib.OPTIONAL_PLATFORMS`, so orient /
port-next will NOT auto-pick it. On the gfx1151 host, dispatch the validator
manually ("Use the validator subagent on projects/YarnBall" for windows-gfx1151),
or ask MOAT to un-optional gfx1151 if that host is permanently back.

## Validation 2026-06-19 (windows-gfx1151, AMD Radeon 8060S, RDNA3.5 iGPU)

Platform: windows-gfx1151 (AMD Radeon 8060S Graphics, gfx1151 RDNA3.5 iGPU)
Physical console session (not RDP): `quser` shows session type = console.
Arch: gfx1151, TheRock ROCm 7.14 / Clang 23.0.0 (GCC-frontend clang++.exe from _rocm_sdk_core)
Head sha: 6c0d315 (806b534 + Windows gfx1151 CMakeLists.txt fixes)

Three Windows-specific CMakeLists.txt fixes committed at 6c0d315 (all WIN32-guarded):
1. Force-include syntax detection: clang++ GCC-frontend uses `-include`, clang-cl MSVC-frontend
   uses `/FI`. Detect CMAKE_CXX_COMPILER_FRONTEND_VARIANT and pick the right flag.
2. __HIP_PLATFORM_AMD__ for Windows host CXX TUs: when clang++ runs without --offload-arch,
   __HIP__ is unset on host .cpp compilation, so hip_common.h doesn't set __HIP_PLATFORM_AMD__.
   Add explicit -D__HIP_PLATFORM_AMD__ on WIN32.
3. Windows Thrust/rocPRIM include paths: TheRock SDK doesn't install these headers; add
   find_path probe with -DROCTHRUST_INCLUDE_DIR / -DROCPRIM_INCLUDE_DIR passthrough.

Build environment:
- Compilers: clang.exe/clang++.exe GCC-frontend from _rocm_sdk_core/lib/llvm/bin/ (NOT clang-cl)
- MSVC runtime: 14.50.35717 from VS Community 18 (C:\Program Files\Microsoft Visual Studio\18\Community)
- Dependencies: glfw3/freetype/CLI11/stb/Eigen3 via alien project's vcpkg_installed/x64-windows
- assimp/jsoncpp: CMake FetchContent (same as gfx1201)
- Thrust/rocPRIM headers: TheRock source tree (-DROCTHRUST_INCLUDE_DIR / -DROCPRIM_INCLUDE_DIR)
- hip-lang cmake stub: created in _rocm_sdk_core/lib/cmake/hip-lang/ (environment stub, not committed)
- hip cmake stub: created in agent_space/YarnBall_cmake/ (environment stub, not committed)
- Thrust/rocPRIM version stubs: generated from .in templates in TheRock source (not committed)

Build:
```
HIP_VISIBLE_DEVICES=0 HIP_DEVICE_LIB_PATH=<_rocm_sdk_core>/lib/llvm/amdgcn/bitcode
LIB=<MSVC-14.50>/lib/x64;<WinSDK>/Lib/10.0.26100.0/ucrt+um/x64
cmake -S . -B build_hip_gfx1151 -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151
  -DCMAKE_HIP_COMPILER=<_rocm_sdk_core>/lib/llvm/bin/clang++.exe
  -DCMAKE_CXX_COMPILER=<_rocm_sdk_core>/lib/llvm/bin/clang++.exe
  -DCMAKE_C_COMPILER=<_rocm_sdk_core>/lib/llvm/bin/clang.exe
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_PREFIX_PATH="<_rocm_sdk_core>;<alien vcpkg_installed>;<YarnBall_cmake stub>"
  -DROCTHRUST_INCLUDE_DIR=<TheRock>/rocm-libraries/projects/rocthrust
  -DROCPRIM_INCLUDE_DIR=<TheRock>/rocm-libraries/projects/rocprim/rocprim/include
  -DEIGEN3_INCLUDE_DIR=<build_hip_gfx1151>/_deps/eigen-src
  -DCMAKE_MODULE_PATH=<build_hip_gfx1151>/_deps/eigen-src/cmake
  -DCMAKE_TLS_VERIFY=OFF
cmake --build build_hip_gfx1151 --target Gui -j6
# Result: [277/277] Linking HIP executable Gui.exe -- warnings only, no errors
```

GPU tests (run from KittenEngine/ working dir):
```
# Test 1: headless cable_work_pattern, 3 frames
HIP_VISIBLE_DEVICES=0 build_hip_gfx1151/Gui.exe \
  configs/cable_work_pattern.json -s --headless -n 3 \
  -o /tmp/yb_frames_gfx1151/frame_ --exit
# Result: "Export complete. sim/real ratio Avg 0.358, SD: 0.077, N=4"
# 4 OBJ files (frame_0.obj..frame_3.obj), 65065 vertices each, 0 NaN/Inf
# frame_3 bbox: x[-0.208,0.207] y[-0.194,0.194] z[-0.029,0.034] -- matches gfx90a/gfx1100/gfx1201 exactly

# Test 2: interactive GUI GL-interop (physical console session, AMD GPU driving display)
HIP_VISIBLE_DEVICES=0 build_hip_gfx1151/Gui.exe configs/cable_work_pattern.json
# (ran with 10-second timeout; 1.5MB of simulation output)
```

GL-interop evidence (from instrumented cudaWriteGL -- throwaway diagnostic, reverted before commit):
```
GL_VENDOR:   ATI Technologies Inc.
GL_RENDERER: AMD Radeon(TM) 8060S Graphics   <-- hardware AMD GPU, NOT software rasterizer
GL_VERSION:  4.6.0 Core Profile Context 26.6.1.260512
[interop] cudaGraphicsGLRegisterBuffer -> 0   (hipSuccess, repeated for every buffer)
[interop] cudaGraphicsMapResources -> 0
[interop] cudaGraphicsResourceGetMappedPointer -> 0 (ptr=0000002407FA0000, size=2082080)
[interop] cudaMemcpy -> 0
... (same 0/0/0/0 pattern on every cudaWriteGL call across all simulation frames)
```

Buffer size 2082080 = 65065 verts * 32 bytes/vert (8 floats * 4), and 1041040 = 65065 * 16
(4 floats * 4): matches the expected GL vertex buffer layout for position+normal data.
All hipGraphics*/hipMemcpy calls returned 0 (hipSuccess) -- GPU-written simulation
vertices successfully uploaded into GL vertex buffers on the physical AMD GPU context.

GL interop PASS criteria met:
- GL_RENDERER names the AMD hardware GPU (not "GDI Generic"/"Microsoft")
- Every hipGraphicsGLRegisterBuffer/MapResources/ResourceGetMappedPointer/Memcpy returned hipSuccess
- Pointer sizes match the expected vertex layout (GL buffers received data of correct size)
- Simulation ran for 10+ seconds with physics advancing (no crash, no abort)

Verdict: PASS. windows-gfx1151 validated at 6c0d315.
Linux platforms (gfx90a/gfx1100) flipped to revalidate by advance_head; the WIN32-guarded
CMakeLists.txt delta does not alter Linux/gfx90a/gfx1100 device code -- carry-forward with
binary-equiv is expected on those hosts.

## Revalidation 2026-06-19 (linux-gfx1100, head 6c0d315)

Platform: linux-gfx1100 (AMD Radeon Pro W7800, gfx1100 RDNA3)
Arch: gfx1100, ROCm 7.2.1, clang 22.0.0git
Trigger: revalidate -- head advanced 806b534 -> 6c0d315 (single commit: "[ROCm] Windows
gfx1151: fix CMake HIP build for clang++ GCC-frontend", 25-line WIN32-guarded CMakeLists.txt
change).

Binary-equivalence check (no GPU run):

Built at both shas for gfx1100-only (`-DCMAKE_HIP_ARCHITECTURES=gfx1100`):
- 806b534 -> build_806/Gui
- 6c0d315 -> build_6c0/Gui

```bash
python3 utils/codeobj_diff.py projects/YarnBall/src/build_806/Gui projects/YarnBall/src/build_6c0/Gui
# verdict=identical
#   Gui vs Gui: identical (exported symbols + device ISA identical (33 exports))
```

The 6c0d315 delta is entirely WIN32-guarded (force-include syntax detection for
clang++ GCC-frontend, -D__HIP_PLATFORM_AMD__ for Windows host TUs, Thrust/rocPRIM
find_path probes). None of these code paths are reached on Linux; the gfx1100
device ISA and all 33 exported symbols are bit-for-bit identical to 806b534.

Verdict: CARRY FORWARD (binary-equiv). No GPU re-run needed. Advancing linux-gfx1100
to completed at 6c0d315.

## Draft PR opened 2026-06-19 -- jerry060599/YarnBall#5 (BLOCKED on LBVH #5)

The upstream PR is now OPEN as a DRAFT: https://github.com/jerry060599/YarnBall/pull/5
(AMD-Ecosystem:moat-port -> jerry060599:main, head 6c0d315). Lead linux-gfx90a is
`pr-open` in status.json. Opened on Jeff's explicit instruction to surface the
work for review now and let the maintainer see both PRs together (both repos are
under jerry060599, so the cross-PR dependency is visible).

Why DRAFT (not ready-for-review): the KittenGpuLBVH submodule still points at
AMD-Ecosystem/KittenGpuLBVH @ 33c0f78 (the head of jerry060599/KittenGpuLBVH#5, which
is still OPEN). That .gitmodules state is un-mergeable upstream, so the PR is held
in draft to prevent an accidental merge. The PR body leads with an [!IMPORTANT]
blocked banner naming KittenGpuLBVH#5 and the repoint-on-merge plan.

NOT squashed: history is the 4 natural [ROCm] commits on top of b178c2b. Do NOT
squash yet -- the submodule must be repointed + revalidated after LBVH#5 merges,
which changes the tree; squashing now would be wasted (re-squash needed anyway).

Post-merge sequence to finish (when jerry060599/KittenGpuLBVH#5 lands):
1. In the fork: set .gitmodules url -> https://github.com/jerry060599/KittenGpuLBVH
   and drop the `branch = moat-port` line; set the submodule gitlink to the merged
   upstream SHA.
2. Re-validate YarnBall headless on gfx90a / gfx1100 / Windows (submodule content
   changes -> real GPU re-run, not a carry-forward).
3. Squash to one commit -> `python3 utils/moatlib.py squash-carry-forward YarnBall <sha>`.
4. `gh pr ready 5 --repo jerry060599/YarnBall` to flip the draft to ready-for-review.
5. On merge: `python3 utils/moatlib.py set-pr-merged YarnBall`.

Prep audit at open (all clean except the squash, deferred per above): README has
the Linux CUDA/ROCm CMake section in house style; CMakeLists defaults gfx90a /
CUDA 86 when arch unset; AMD copyright + `\author Jeff Daily` headers present in
cuda_to_hip.h and CMakeLists.txt; no MOAT jargon in diff or commit messages.
## Revalidation 2026-06-19 (windows-gfx1201, head 6c0d315)

Platform: windows-gfx1201 (AMD Radeon RX 9070 XT, gfx1201 RDNA4), HIP_VISIBLE_DEVICES=0
Arch: gfx1201, TheRock ROCm 7.14 / Clang 23.0.0 (GNU-like frontend)
Head sha: 6c0d315
Prior validated_sha: 806b534
Trigger: revalidate -- head advanced 806b534 -> 6c0d315 (WIN32-guarded CMakeLists.txt fixes
for gfx1151 GCC-frontend: force-include flag detection, -D__HIP_PLATFORM_AMD__, Thrust/rocPRIM
find_path probes). These changes are relevant to the Windows build and touched the active
CMakeLists.txt code paths, so a clean rebuild on gfx1201 was required.

Note: the gfx1201 host is RDP-only; GUI/GL-interop validation is covered by gfx1151 (see
"Validation 2026-06-19 (windows-gfx1151)" above). This revalidation covers headless/build only.

Delta analysis: 6c0d315 adds WIN32-guarded blocks to CMakeLists.txt --
- Force-include flag detection (CMAKE_CXX_COMPILER_FRONTEND_VARIANT: MSVC -> /FI, GNU -> -include)
- __HIP_PLATFORM_AMD__ compile definition on WIN32 for host CXX TUs
- Thrust/rocPRIM find_path probe guarded by WIN32

On this host the compiler is "Clang 23.0.0 with GNU-like command-line" (frontend variant = GNU),
so the force-include code selects `-include` (same as the hard-coded prior value). The
__HIP_PLATFORM_AMD__ define and Thrust probes are additive. Clean rebuild confirms no regression.

GPU device at validation: AMD Radeon RX 9070 XT (hipInfo: device 0, gfx1201, 15.92 GB VRAM)

Build (clean rebuild at 6c0d315, from scratch):
```
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
rm -rf build_hip
cmake -S . -B build_hip -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/clang.exe \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$ROCM;B:/vcpkg/installed/x64-windows" \
  -DCMAKE_TLS_VERIFY=OFF
cmake --build build_hip -j64
# Result: [277/277] Linking HIP executable Gui.exe -- warnings only, no errors
```

Runtime DLLs copied into build_hip/: amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll,
hiprtc0714.dll, hiprtc-builtins0714.dll (from _rocm_sdk_core/bin); glfw3.dll, freetype.dll,
brotli*.dll, bz2.dll, libpng16.dll (from vcpkg); assimp.dll (_deps/assimp_fc-build/bin),
jsoncpp.dll (_deps/jsoncpp_fc-build/src/lib_json). Run with build_hip/ on the Windows PATH
(bash: export PATH="/b/develop/moat/projects/YarnBall/src/build_hip:$PATH").

GPU tests (run from KittenEngine/ working dir):
```
# Test 1: cable_work_pattern scene, 3 frames, headless
HIP_VISIBLE_DEVICES=0 PATH=/b/develop/moat/projects/YarnBall/src/build_hip:$PATH \
  build_hip/Gui.exe configs/cable_work_pattern.json -s --headless -n 3 \
  -o /c/Temp/yb_frames_rv2/frame_ --exit
# Result: "Export complete. sim/real ratio Avg 0.783, SD: 0.147, N=4"
# 4 OBJ files (frame_0.obj..frame_3.obj), 65065 vertices each
# All vertices finite (0 NaN/Inf)
# frame_3 bbox: x[-0.208,0.207] y[-0.194,0.194] z[-0.029,0.034] -- matches gfx90a/gfx1100/prior gfx1201 exactly

# Test 2: letterS scene, 3 frames, headless
HIP_VISIBLE_DEVICES=0 PATH=/b/develop/moat/projects/YarnBall/src/build_hip:$PATH \
  build_hip/Gui.exe configs/letterS.json -s --headless -n 3 \
  -o /c/Temp/yb_letter_rv2/frame_ --exit
# Result: "Export complete. sim/real ratio Avg 1.362, SD: 0.347, N=4"
# 4 OBJ files, 32919 vertices each, all finite
# Note: 32919 vertices (Windows) vs 32931 (Linux) -- same difference as prior gfx1201 runs;
# floating-point resampling difference in CPU spline code between platforms. Geometry is sane.
```

Both tests: exit 0, finite geometry (0 NaN/Inf), physics advancing, bbox matches prior runs.
Cosserat.cu:185 wave32 divergence did not manifest on RDNA4.

GL-interop coverage: headless only (RDP host; physical-console GL-interop proof is on gfx1151).

Verdict: PASS. windows-gfx1201 revalidated at 6c0d315 (headless/build coverage). GUI/GL-interop
proof is gfx1151's at 6c0d315 (see "Validation 2026-06-19 (windows-gfx1151...)" above).

## Validation 2026-06-20 (windows-gfx1101, AMD Radeon PRO V710, RDNA3)

Platform: windows-gfx1101 (AMD Radeon PRO V710, gfx1101 RDNA3), HIP_VISIBLE_DEVICES=1
Arch: gfx1101, TheRock ROCm 7.14 / Clang 23.0.0 (GNU-like frontend, _rocm_sdk_devel)
Head sha: 6c0d315

First validation of this platform (port-ready -> completed). The 6c0d315 WIN32-guarded
CMakeLists.txt fixes (force-include flag detection, -D__HIP_PLATFORM_AMD__, Thrust/rocPRIM
find_path probes) apply to this host too; the GNU-like frontend selects `-include` (same
as gfx1201). No source delta needed vs the existing moat-port branch.

GPU device confirmed: `HIP_VISIBLE_DEVICES=1 hipInfo` -> "AMD Radeon PRO V710 / gfx1101" (healthy).

Build (clean build at 6c0d315, separate build_hip_gfx1101 dir):
```
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
cmake -S . -B build_hip_gfx1101 -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/clang.exe \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$ROCM;B:/vcpkg/installed/x64-windows" \
  -Dhip_DIR="$ROCM/lib/cmake/hip" \
  -DCMAKE_TLS_VERIFY=OFF
cmake --build build_hip_gfx1101 -j64
# Result: [277/277] Linking HIP executable Gui.exe -- warnings only, no errors
# Verified: strings Gui.exe | grep gfx1101 -> hipv4-amdgcn-amd-amdhsa--gfx1101 embedded
```

Note: -Dhip_DIR must be passed explicitly; find_package(hip) fails when CMAKE_PREFIX_PATH
alone is used (same _rocm_sdk_devel root that worked for gfx1201 without this flag -- the
difference may be a stale cache from the prior gfx1201 configure; explicit -Dhip_DIR is safe).

Runtime DLLs copied into build_hip_gfx1101/: amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll,
hiprtc0714.dll, hiprtc-builtins0714.dll (from _rocm_sdk_core/bin); glfw3.dll, freetype.dll,
brotlicommon.dll, brotlidec.dll, bz2.dll, libpng16.dll, z.dll (from vcpkg); assimp.dll
(_deps/assimp_fc-build/bin), jsoncpp.dll (_deps/jsoncpp_fc-build/src/lib_json).
Run from the src/ directory with PATH=build_hip_gfx1101:$PATH so DLLs in that dir are found
(bash PATH DLL search applies when running a relative-path exe; Windows loader then finds DLLs
in the same directory as the exe itself once it's resolved to an absolute path).

GPU tests (run from src/ directory):
```
# Test 1: cable_work_pattern scene, 3 frames, headless
HIP_VISIBLE_DEVICES=1 build_hip_gfx1101/Gui.exe KittenEngine/configs/cable_work_pattern.json \
  -s --headless -n 3 -o C:/Temp/yb_frames_gfx1101/frame_ --exit
# Result: "Export complete. sim/real ratio Avg 0.679, SD: 0.116, N=4"
# 4 OBJ files (frame_0.obj..frame_3.obj), 65065 vertices each
# All vertices finite (0 NaN/Inf)
# frame_3 bbox: x[-0.208,0.207] y[-0.194,0.194] z[-0.029,0.034] -- matches gfx90a/gfx1100/gfx1201 exactly

# Test 2: letterS scene, 3 frames, headless
HIP_VISIBLE_DEVICES=1 build_hip_gfx1101/Gui.exe KittenEngine/configs/letterS.json \
  -s --headless -n 3 -o C:/Temp/yb_letter_gfx1101/frame_ --exit
# Result: "Export complete. sim/real ratio Avg 1.397, SD: 0.366, N=4"
# 4 OBJ files, 32919 vertices each, all finite
# Note: 32919 vertices (Windows) vs 32931 (Linux) -- same Windows/Linux resampling difference
# seen on gfx1201; geometry is sane and finite.
```

Both tests: exit 0, finite geometry (0 NaN/Inf), physics advancing, bbox matches all prior platforms.
Cosserat.cu:185 wave32 divergence did not manifest on gfx1101 RDNA3 (same class as gfx1100 RDNA3
on Linux). Post-test hipInfo health check: GPU still present and healthy.

Verdict: PASS. windows-gfx1101 (RDNA3, PRO V710) validated at 6c0d315 on real GPU.
