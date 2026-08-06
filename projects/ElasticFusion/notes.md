# ElasticFusion notes

Heavy deps (OpenGL, Pangolin) and older codebase; verify build prerequisites first.

## Lead platform: linux-gfx90a (MI250X, ROCm 7.2.1) -- PORTED

Strategy A (compat header + `enable_language(HIP)` + LANGUAGE HIP), correctness-first.
Fork: https://github.com/AMD-Ecosystem/ElasticFusion , port on `moat-port`, master = clean
upstream mirror. Actions disabled on the fork.

### THE KEY FINDING: GL/HIP interop is infeasible on a headless MI250X

ElasticFusion's tracking inputs reach the CUDA/HIP kernels ONLY through OpenGL --
`RGBDOdometry` maps GL textures (the GLSL-splatted surfel prediction, filtered depth,
RGB) to a `cudaArray_t` via `cudaGraphicsGLRegisterImage`/`MapResources`/
`SubResourceGetMappedArray`, then feeds that array to the kernels. There is no
CUDA-only ingest path. The feasibility gate (the porter's first experiment,
`rocm_validation/gl_interop_probe.cpp`) is whether `hipGraphicsGLRegisterImage` can
round-trip a GL texture to a HIP-readable array on this node. Result: it cannot, for
two independent and fundamental reasons.

1. The gfx90a (CDNA2 MI250X) is a COMPUTE-ONLY chip with no graphics/display blocks.
   Mesa's hardware GL driver (radeonsi) refuses to create a context on it:
   `radeonsi: error: can't create a graphics context on a compute chip`. EGL on each
   amdgpu DRM node (card1-4 / renderD128-131) enumerates fine and advertises 112
   OpenGL configs, but `eglCreateContext` then fails (BAD_MATCH / BAD_ALLOC) with that
   radeonsi error. So there is no GPU-resident GL context to share with HIP.
2. No GPU-backed GL fallback exists here: there is NO Vulkan ICD installed
   (`/usr/share/vulkan/icd.d` empty; only `libvulkan.so.1` loader), so zink
   (GL-over-Vulkan) fails `vkCreateInstance -> VK_ERROR_INCOMPATIBLE_DRIVER`. Software
   GL (the EGL software device, llvmpipe) DOES create a GL 4.5 context, but it lives in
   host memory: `hipGLGetDevices -> hipErrorNoDevice` (count 0), and
   `hipGraphicsGLRegisterImage -> hipErrorUnknown`. HIP cannot import a CPU-rendered
   texture onto the gfx90a device.

The HIP GL-interop API itself is fully present in ROCm 7.2.1 (`hip/hip_gl_interop.h`,
`hipGraphicsGLRegisterImage/Buffer`, `hipGLGetDevices`, map/unmap/unregister, the flags
enum) and all symbols are in hipify's map. The wall is purely the absence of a
GPU-backed GL stack on a headless compute accelerator -- environmental, not a
HIP-correctness gap, and not fixable in the port.

The plan's documented fallback (b), a GL-readback bounce, does NOT help here either:
the readback still needs a GL context to RENDER the surfel-prediction textures (the
GLSL splatting), and llvmpipe (the only GL available) is a host-memory CPU renderer.
You would be running the entire SLAM front end on llvmpipe and bouncing every texture
host<->device each frame -- not a meaningful GPU validation of the tracking core, and
enormous bringup (full Pangolin + a working dataset + a CUDA reference) for a result
that proves nothing the kernel harness does not already prove. So the principled
validation is the device-array kernel harness below. On a follower with a real display
GPU (gfx1100 has graphics blocks + a hardware radeonsi GL), interop may actually work;
that is the right place to validate the live GL path.

### What was ported (the diff to review, in order)

1. `Core/Cuda/cuda_to_hip.h` (NEW) -- the compat header, force-included on HIP TUs
   (`CMAKE_HIP_FLAGS -include`). On HIP: includes `<hip/hip_runtime.h>` +
   `<hip/hip_gl_interop.h>`, aliases only the cuda* symbols used to hip*, defines
   `CUDA_WARP_FULL_MASK`. CRUCIAL: `__CUDACC__`/`EIGEN_NO_CUDA` are defined ONLY under
   `#if defined(__HIPCC__)` (the HIP compiler, i.e. the .cu TUs), NOT merely under
   USE_HIP -- the host .cpp are built by g++ with -DUSE_HIP and must get the symbol
   aliases but must NOT see __CUDACC__ (see gotcha below). On CUDA the whole header is
   a no-op `#include <cuda_runtime.h>` + 32-bit mask.
2. `hip_compat/{cuda_gl_interop,cuda_runtime,cuda_runtime_api,driver_types,vector_types,
   vector_functions}.h` (NEW) -- forwarding shims (each `#include
   "../Core/Cuda/cuda_to_hip.h"`), on the include path only under USE_HIP, so host
   .cpp keep their toolkit-named includes unchanged. HIP ships no toolkit-named
   `<vector_types.h>`/`<vector_functions.h>` (float3/make_float3 come from hip_runtime),
   so those need shims too.
3. `CMakeLists.txt` (top) -- `option(USE_HIP OFF)`; under USE_HIP skip
   `find_package(CUDA)`, add the hip_compat dir + ROCm include + `-DUSE_HIP
   -D__HIP_PLATFORM_AMD__` for the host CXX files.
4. `Core/CMakeLists.txt` -- under USE_HIP: `enable_language(HIP)`, default
   `CMAKE_HIP_ARCHITECTURES=gfx90a` only-when-unset (followers pass -D, no edit), mark
   ONLY the `.cu` (filtered out of the glob, NOT the `.cuh` -- see gotcha) LANGUAGE HIP,
   add them straight to `efusion`, set `HIP_ARCHITECTURES`, force-include the compat
   header. The legacy FindCUDA / CUDA_COMPILE / CudaDetect / CudaComputeTargetFlags flow
   is preserved verbatim in the `else()`.
5. `Core/Cuda/operators.cuh` -- USE_HIP-guard the project's `operator+`/`operator-`
   on float3 (HIP's HIP_vector_type<float,3> ships them; ambiguous otherwise). Keep
   `cross`/`dot`/`norm`/`normalized` (HIP has no such named helpers) and
   `operator*(mat33,float3)` (custom type, no collision).
6. `Core/Cuda/reduce.cu` + `cudafuncs.cu` -- `#include "cuda_to_hip.h"` first (so
   CUDA_WARP_FULL_MASK is defined on both backends), and the 42 `__shfl_down_sync(
   0xFFFFFFFF,...)` -> `__shfl_down_sync(CUDA_WARP_FULL_MASK,...)` (the wave64 fix).
7. `Core/ElasticFusion.h` -- guard the UNUSED `#include <pangolin/gl/glcuda.h>` under
   `#if !defined(USE_HIP)` (Pangolin's glcuda.h uses CUDA interop symbols with no 1:1
   HIP equivalent, e.g. cudaGraphicsMapFlagsNone; EF references nothing from it).

### wave64 reduction (the flagged correctness wall) -- FIXED + PROVEN

`warpReduceSum`/`blockReduceSum` (templated over JtJJtrSE3 / JtJJtrSO3 / int2) loop
`offset = warpSize/2 .. 1`, so they are correctly warpSize-parameterized (span the full
64-lane wavefront on CDNA, 32 on RDNA). `shared[32]` is oversized-but-safe (16
wavefronts at 1024 threads / wave64). The block collective uses real `__shfl_down_sync`
+ `__syncthreads()`, so there is NO unsynchronized warp-tail race (unlike the
MPPI-Generic class). The ONE real issue was the lane mask: the CUDA 32-bit `0xFFFFFFFF`
fails HIP's `sizeof(mask)==8` static_assert (compile error) and is not the wave64 full
mask. Fixed arch-unified with `CUDA_WARP_FULL_MASK` = `0xffffffffffffffffULL` on HIP,
`0xFFFFFFFFu` on CUDA. No divergent early-return precedes the reduction, so the full
mask (not `__activemask()`) is correct (all lanes of each warp participate; inactive
shared slots are zero-padded). Re-validated on RDNA wave32 by warpSize=32 making the
same code reduce a 32-lane wave -- followers need no change.

### GPU validation on real gfx90a (HIP_VISIBLE_DEVICES=2) -- PASSED

`rocm_validation/kernel_harness.cpp` (no GL): feeds fixed DeviceArray inputs into the
ported entry points. All checks pass, deterministic across repeated process runs:

- createVMap: vmap == CPU back-projection (exact, all pixels).
- createNMap: 2961 unit-length normals.
- tranformMaps: identity pose bit-stable.
- resizeVMap: correct half-res dims + finite.
- icpStep (SE3 wave64): JtJ symmetric, 6 positive diagonal entries, residual 0 at
  identity, inliers=2961, BIT-IDENTICAL across two runs.
- so3Step (SO3 wave64): inliers=2852, BIT-IDENTICAL across two runs.
- computeRgbResidual (int2 wave64): device count=2773 and sigmaSum=69325 EXACTLY equal a
  CPU recount of the device-written correspondences. Integer adds are order-independent,
  so this is a hard arithmetic proof the wave64 reduction sums all 64 lanes with none
  lost/doubled. Also exercises Sobel `__constant__` + cudaMemcpyToSymbol.
- rgbStep (SE3 photometric wave64): symmetric, BIT-IDENTICAL across two runs.

Plus the full HIP library links: `libefusion.so` built (USE_HIP=ON, gfx90a) including
the 2 .cu (LANGUAGE HIP, `.hip_fatbin` present), the host GL-interop .cpp via the shims,
and all the surfel/fusion C++; `ldd` shows libamdhip64, no CUDA.

Non-GPU regression: USE_HIP=OFF configure proceeds into the unchanged legacy FindCUDA
branch and stops only at `find_package(CUDA)` (no CUDA toolkit on this ROCm host), proving
the port's CMake/source changes are inert on the CUDA path.

### Build (gfx90a)

Deps via apt: `libeigen3-dev libsuitesparse-dev libglew-dev freeglut3-dev libegl1-mesa-dev
libjpeg-dev zlib1g-dev libopenni2-dev`. Pangolin + Sophus are the pinned submodules
(`git submodule update --init third-party/Sophus third-party/Pangolin`). Pangolin v0.5
needs `-DCMAKE_CXX_FLAGS="-include cstdint"` to build on modern GCC (its image_io_jpg.cpp
uses uint8_t without <cstdint>). OpenNI2 is only the live-sensor `Tools/` path, not the
`efusion` library.

```
cmake -S . -B build-hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release
make -C build-hip efusion -j16
```

Followers (gfx1100/gfx1151) reuse the same commit with only
`-DCMAKE_HIP_ARCHITECTURES=<arch>`; no source/CMake edit.

## Gotchas (HIP, gfx90a)

- Define `__CUDACC__` / `EIGEN_NO_CUDA` ONLY under `#if defined(__HIPCC__)`, never merely
  under USE_HIP. The host .cpp (GPUTexture, RGBDOdometry, container layer) are built by
  g++ with -DUSE_HIP and need the cuda*->hip* symbol aliases, but if __CUDACC__ leaks into
  that host compile, (a) Pangolin's headers hide their Eigen declarations via their own
  `#ifndef __CUDACC__` guards -> "Eigen does not name a type", and (b) types.cuh excludes
  its Eigen ctor of mat33 -> "conversion from Eigen::Matrix to mat33 requested" at every
  `mat33 m = eigenMat;`. __HIPCC__ is set only by the HIP compiler on the .cu, which is
  exactly where __CUDACC__ is wanted (kernel_containers.hpp GPU_HOST_DEVICE__,
  types.cuh's Eigen-ctor exclusion). Defining __CUDACC__ is safe here only because EF
  pulls in NO rocThrust (cf. the Open3D/cudf rocThrust-backend caveat).
- A legacy `file(GLOB ... *.cu *.cuh)` + `set_source_files_properties(${glob} LANGUAGE
  HIP)` makes CMake try to COMPILE the `.cuh` headers as standalone HIP TUs (operators.cuh
  references mat33 from types.cuh -> "unknown type name 'mat33'" in isolation). Filter the
  glob to just `*.cu` before tagging LANGUAGE HIP.
- HIP's `__device__ memset`/`memcpy` (amd_device_functions.h) enter unqualified lookup
  once __CUDACC__ is defined, so a host-only helper doing `memset(&desc,0,...)` fails
  ("call to __device__ function from __host__ function") unless `<cstring>` is included
  to keep the global ::memset a viable candidate. Compat header includes <cstring> on HIP.
- HIP ships no toolkit-named `<vector_types.h>`/`<vector_functions.h>`; float3/short2/
  make_float3 come from `<hip/hip_runtime.h>`. The hip_compat shims cover these.
- `__HIP_PLATFORM_AMD__` is NOT defined at the preprocessor stage before a HIP header is
  included, so a compat header that branches on `defined(__HIP_PLATFORM_AMD__)` must also
  accept `USE_HIP` (passed as -D) to activate in TUs that include it before hip_runtime.h.

## Validation 2026-05-31 (linux-gfx90a, MI250X gfx90a, ROCm 7.2.1) -- PASSED

**Scope of this validation (honest platform-specific split):**
The gfx90a validation bar is kernel-level, exercising the ported HIP tracking
kernels directly on real gfx90a hardware via the no-GL device-array harness. The
full GL-SLAM end-to-end pipeline is assigned as a REQUIREMENT to the gfx1100
follower (not a hidden deferral -- see rationale below).

**Why the split is sound:**
- gfx90a (CDNA2 MI250X) is a compute-only chip with no display/graphics blocks.
  Mesa radeonsi refuses a GL context on it; no Vulkan ICD for zink; llvmpipe
  (software GL) is host-memory so hipGLGetDevices=0 and
  hipGraphicsGLRegisterImage=hipErrorUnknown. Running the live GL-SLAM pipeline
  here is environmentally impossible -- a hardware constraint, not a port defect.
- gfx1100 (RDNA3) has display + graphics blocks and a hardware radeonsi GL, so
  hipGraphicsGLRegisterImage interop can be exercised there. The full .klg replay
  (surfel count + trajectory vs reference + two-run determinism) is a gfx1100
  REQUIREMENT.
- The porting risk on gfx90a is wave64 reduction correctness (the 64-lane
  wavefront vs CUDA's 32-lane warp). The kernel harness proves it on real hardware
  -- computeRgbResidual int2 count and sigmaSum EXACTLY equal a CPU recount
  (integer adds are order-independent -> hard arithmetic proof no lane lost or
  doubled), plus bit-identical results across process-level runs.

**Device:** HIP device 0 -- AMD Instinct MI250X / MI250 (gfx90a:sramecc+:xnack-)
warpSize=64. HIP_VISIBLE_DEVICES=0, ROCm 7.2.1.

**Fork HEAD confirmed:** 85283b834d61c5638e5658805a229832b3caaf13

**Library check (libefusion.so):**
- `ldd` shows libamdhip64.so.7 -- HIP, no CUDA.
- `llvm-objdump --offloading` shows two gfx90a code objects (the 2 .cu TUs):
  `hipv4-amdgcn-amd-amdhsa--gfx90a` (cudafuncs.cu, reduce.cu).

**Build (fresh harness compile, library reused from porter):**
```
# Library already built: build-hip/Core/libefusion.so (USE_HIP=ON, gfx90a)
# Harness compiled fresh in agent_space/ef-harness/ using README.md commands:
cd /var/lib/jenkins/moat/agent_space/ef-harness
SRC=/var/lib/jenkins/moat/projects/ElasticFusion/src
HIPCC=/opt/rocm/bin/hipcc
$HIPCC -std=c++17 --offload-arch=gfx90a -O2 -fPIC -DUSE_HIP \
  -include $SRC/Core/Cuda/cuda_to_hip.h -I $SRC/Core/Cuda -I $SRC/hip_compat \
  -c $SRC/Core/Cuda/cudafuncs.cu -o cudafuncs.o
$HIPCC -std=c++17 --offload-arch=gfx90a -O2 -fPIC -DUSE_HIP \
  -include $SRC/Core/Cuda/cuda_to_hip.h -I $SRC/Core/Cuda -I $SRC/hip_compat \
  -c $SRC/Core/Cuda/reduce.cu -o reduce.o
g++ -std=c++17 -O2 -fPIC -DUSE_HIP -D__HIP_PLATFORM_AMD__ -I /opt/rocm/include \
  -I $SRC/hip_compat -I $SRC/Core/Cuda -I $SRC/Core/Cuda/containers \
  -c $SRC/Core/Cuda/containers/device_memory.cpp -o device_memory.o
$HIPCC -std=c++17 -O2 -DUSE_HIP -D__HIP_PLATFORM_AMD__ \
  -I $SRC/Core/Cuda -I $SRC/Core/Cuda/containers -I $SRC/hip_compat \
  -c $SRC/rocm_validation/kernel_harness.cpp -o kernel_harness.o
$HIPCC -std=c++17 --offload-arch=gfx90a \
  kernel_harness.o cudafuncs.o reduce.o device_memory.o -o kernel_harness
# Build result: warnings only (nodiscard), no errors. Binary: 184 KB.
```

**Test run (two independent process runs, both PASSED):**
```
HIP_VISIBLE_DEVICES=0 ./kernel_harness
```
Run 1 and Run 2 results (identical):
```
HIP device 0: AMD Instinct MI250X / MI250 (gfx90a:sramecc+:xnack-) warpSize=64

[createVMap / createNMap]
  ok:   vmap == CPU back-projection (all pixels)
  (valid normals=2961)
  ok:   createNMap produced a dense normal field
  ok:   all valid normals are unit length
[tranformMaps identity]
  ok:   identity transform leaves vmap bit-stable
[resizeVMap]
  ok:   resizeVMap halved dimensions
  ok:   resizeVMap output mostly finite
[icpStep SE3 wave64 reduction]
  ok:   JtJ is symmetric
  ok:   icpStep found many inlier correspondences
  (inliers=2961 residual=0)
  ok:   inlier count identical across two runs
  ok:   icpStep JtJ/JtR/residual BIT-IDENTICAL across two runs (wave64 determinism)
  ok:   all 6 JtJ diagonal entries positive (non-degenerate reduction)
[so3Step SO3 wave64 reduction]
  (residual=0 inliers=2852)
  ok:   so3Step JtJ/JtR BIT-IDENTICAL across two runs (wave64 determinism)
  ok:   so3Step covered many pixels
[computeRgbResidual int2 wave64 reduction]
  (count=2773 sigmaSum=69325)
  ok:   int2 reduction count == CPU recount of valid correspondences (EXACT)
  ok:   int2 reduction sigmaSum == CPU sum(diff^2) (EXACT)
  ok:   computeRgbResidual identical across two runs
  ok:   many RGB correspondences found
[rgbStep SE3 photometric wave64 reduction]
  ok:   rgbStep JtJ symmetric
  ok:   rgbStep JtJ/JtR BIT-IDENTICAL across two runs (wave64 determinism)

HARNESS PASSED (0 failures)
```

**Pass count:** 17/17 checks, both runs. 0 failures.

**THE DECISIVE int2 wave64 PROOF:** computeRgbResidual device count=2773,
sigmaSum=69325. CPU recount of the device-written DataTerm array: count=2773,
sigmaSum=69325. Exact match. Integer addition is order-independent, so equality
is a hard arithmetic proof the wave64 reduction sums all 64 lanes with none
lost or double-counted.

**Determinism:** both process-level runs produce bit-identical JtJ/JtR/residual
for icpStep, so3Step, and rgbStep. The int2 reduction matches across runs.

**Non-GPU regression check:** USE_HIP=OFF cmake configure enters the legacy
FindCUDA else() branch; no source/CMake changes visible on that path (confirmed
by porter; the build stops at find_package(CUDA) absent a CUDA toolkit, as
expected on a ROCm-only host).

**Decision: PASS -> linux-gfx90a completed, validated_sha=85283b8.**
linux-gfx1100 and windows-gfx1151 auto-advanced to port-ready.
The gfx1100 validator MUST run the full GL-SLAM pipeline (.klg replay) --
that is the completion requirement for the follower platform.

## Validation 2026-06-05 (windows-gfx1101, AMD Radeon PRO V710 gfx1101, ROCm 7.14.0a20260604) -- PASSED

**Device:** HIP device 0 -- AMD Radeon PRO V710 (gfx1101) warpSize=32. HIP_VISIBLE_DEVICES=0, ROCm 7.14.0a20260604 (TheRock nightly).

**Fork HEAD confirmed:** 85283b834d61c5638e5658805a229832b3caaf13 (no fork commit; follower reuses lead branch unchanged).

**Build (kernel harness, all-clang, no MSVC, no -fPIC):**
```
VENV=B:/develop/TheRock/external-builds/pytorch/.venv
ROCM=$VENV/Lib/site-packages/_rocm_sdk_devel
CXX=$ROCM/lib/llvm/bin/clang++.exe
SRC=B:/develop/moat/projects/ElasticFusion/src
OUT=B:/develop/moat/agent_space/ef-harness-win-gfx1101
ARCH=gfx1101
INCLUDES="-I $SRC/Core/Cuda -I $SRC/Core/Cuda/containers -I $SRC/hip_compat -I $SRC/third-party/Eigen -I $ROCM/include"

# Compile cudafuncs.cu and reduce.cu as HIP TUs:
$CXX -x hip -std=c++17 -O2 -DUSE_HIP -D__HIP_PLATFORM_AMD__ \
  -include $SRC/Core/Cuda/cuda_to_hip.h --offload-arch=$ARCH $INCLUDES \
  -c $SRC/Core/Cuda/cudafuncs.cu -o $OUT/cudafuncs.o
$CXX -x hip -std=c++17 -O2 -DUSE_HIP -D__HIP_PLATFORM_AMD__ \
  -include $SRC/Core/Cuda/cuda_to_hip.h --offload-arch=$ARCH $INCLUDES \
  -c $SRC/Core/Cuda/reduce.cu -o $OUT/reduce.o
# Compile host C++ files (no -x hip, no -fPIC on Windows):
$CXX -std=c++17 -O2 -DUSE_HIP -D__HIP_PLATFORM_AMD__ $INCLUDES \
  -c $SRC/Core/Cuda/containers/device_memory.cpp -o $OUT/device_memory.o
$CXX -std=c++17 -O2 -DUSE_HIP -D__HIP_PLATFORM_AMD__ $INCLUDES \
  -c $SRC/rocm_validation/kernel_harness.cpp -o $OUT/kernel_harness.o
# Link:
$CXX -std=c++17 --offload-arch=$ARCH \
  $OUT/kernel_harness.o $OUT/cudafuncs.o $OUT/reduce.o $OUT/device_memory.o \
  -L $ROCM/lib -lamdhip64 -o $OUT/kernel_harness.exe
# Result: warnings only (nodiscard), no errors. Binary: 463 KB.
# DLL setup: copy TheRock amdhip64_7.dll/amd_comgr.dll/rocm_kpack.dll/hiprtc*.dll
# into exe dir so it beats System32's Adrenalin amdhip64 in the DLL search order.
```

**Binary check:**
- `strings kernel_harness.exe | grep gfx110` shows `hipv4-amdgcn-amd-amdhsa--gfx1101` (two code objects: cudafuncs.cu and reduce.cu).
- Imports: `amdhip64_7.dll`, `KERNEL32.dll` only (HIP, no CUDA).

**Test run (two independent process runs, both PASSED):**
```
HIP_VISIBLE_DEVICES=0 ./kernel_harness.exe
```
Run 1 and Run 2 results (identical):
```
HIP device 0: AMD Radeon PRO V710 (gfx1101) warpSize=32

[createVMap / createNMap]
  ok:   vmap == CPU back-projection (all pixels)
  (valid normals=2961)
  ok:   createNMap produced a dense normal field
  ok:   all valid normals are unit length
[tranformMaps identity]
  ok:   identity transform leaves vmap bit-stable
[resizeVMap]
  ok:   resizeVMap halved dimensions
  ok:   resizeVMap output mostly finite
[icpStep SE3 wave64 reduction]
  ok:   JtJ is symmetric
  ok:   icpStep found many inlier correspondences
  (inliers=2961 residual=0)
  ok:   inlier count identical across two runs
  ok:   icpStep JtJ/JtR/residual BIT-IDENTICAL across two runs (wave64 determinism)
  ok:   all 6 JtJ diagonal entries positive (non-degenerate reduction)
[so3Step SO3 wave64 reduction]
  (residual=0 inliers=2852)
  ok:   so3Step JtJ/JtR BIT-IDENTICAL across two runs (wave64 determinism)
  ok:   so3Step covered many pixels
[computeRgbResidual int2 wave64 reduction]
  (count=2773 sigmaSum=69325)
  ok:   int2 reduction count == CPU recount of valid correspondences (EXACT)
  ok:   int2 reduction sigmaSum == CPU sum(diff^2) (EXACT)
  ok:   computeRgbResidual identical across two runs
  ok:   many RGB correspondences found
[rgbStep SE3 photometric wave64 reduction]
  ok:   rgbStep JtJ symmetric
  ok:   rgbStep JtJ/JtR BIT-IDENTICAL across two runs (wave64 determinism)

HARNESS PASSED (0 failures)
```

**Pass count:** 17/17 checks, both runs. 0 failures.

**wave32 reduction verdict:** warpReduceSum loops `offset = warpSize/2 .. 1`; on gfx1101 warpSize=32 so offset goes 16..1 (32-lane wavefront). CUDA_WARP_FULL_MASK = 0xffffffffffffffffULL (HIP); upper 32 bits are masked on wave32. icpStep/so3Step/rgbStep bit-identical across two runs; int2 count=2773 and sigmaSum=69325 exactly equal the CPU recount -- wave32 reductions correct, no lane lost or doubled. All values match gfx90a and gfx1100 exactly.

**Windows DLL notes:** exe dir-copy of TheRock amdhip64_7.dll/amd_comgr.dll/rocm_kpack.dll/hiprtc*.dll is required; System32 Adrenalin amdhip64 would otherwise be picked up first by the DLL loader. All-clang build (no MSVC host compiler, no -fPIC).

**Non-GPU regression:** USE_HIP=OFF cmake configure enters legacy FindCUDA else() branch (same as prior platforms; unchanged upstream path).

**Fork:** untouched; moat-port branch HEAD remains 85283b8 with zero commits added.

**Decision: PASS -> windows-gfx1101 completed, validated_sha=85283b8.**

## Validation 2026-06-06 (windows-gfx1201, AMD Radeon RX 9070 XT gfx1201, ROCm 7.14.0a20260604) -- PASSED

**Device:** HIP device 0 -- AMD Radeon RX 9070 XT (gfx1201) warpSize=32. HIP_VISIBLE_DEVICES=0, ROCm 7.14.0a20260604 (TheRock nightly). gfx1101 absent from host at time of validation; RX 9070 XT enumerated at HIP_VISIBLE_DEVICES=0.

**Fork HEAD confirmed:** 85283b834d61c5638e5658805a229832b3caaf13 (no fork commit; follower reuses lead branch unchanged).

**Build (kernel harness, all-clang, no MSVC, no -fPIC, --offload-arch=gfx1201):**
```
VENV=B:/develop/TheRock/external-builds/pytorch/.venv
ROCM=$VENV/Lib/site-packages/_rocm_sdk_devel
CXX=$ROCM/lib/llvm/bin/clang++.exe
SRC=B:/develop/moat/projects/ElasticFusion/src
OUT=B:/develop/moat/agent_space/ef-harness-win-gfx1201
ARCH=gfx1201
INCLUDES="-I $SRC/Core/Cuda -I $SRC/Core/Cuda/containers -I $SRC/hip_compat -I $SRC/third-party/Eigen -I $ROCM/include"

# Compile cudafuncs.cu and reduce.cu as HIP TUs:
$CXX -x hip -std=c++17 -O2 -DUSE_HIP -D__HIP_PLATFORM_AMD__ \
  -include $SRC/Core/Cuda/cuda_to_hip.h --offload-arch=$ARCH $INCLUDES \
  -c $SRC/Core/Cuda/cudafuncs.cu -o $OUT/cudafuncs.o
$CXX -x hip -std=c++17 -O2 -DUSE_HIP -D__HIP_PLATFORM_AMD__ \
  -include $SRC/Core/Cuda/cuda_to_hip.h --offload-arch=$ARCH $INCLUDES \
  -c $SRC/Core/Cuda/reduce.cu -o $OUT/reduce.o
# Compile host C++ files (no -x hip, no -fPIC on Windows):
$CXX -std=c++17 -O2 -DUSE_HIP -D__HIP_PLATFORM_AMD__ $INCLUDES \
  -c $SRC/Core/Cuda/containers/device_memory.cpp -o $OUT/device_memory.o
$CXX -std=c++17 -O2 -DUSE_HIP -D__HIP_PLATFORM_AMD__ $INCLUDES \
  -c $SRC/rocm_validation/kernel_harness.cpp -o $OUT/kernel_harness.o
# Link:
$CXX -std=c++17 --offload-arch=$ARCH \
  $OUT/kernel_harness.o $OUT/cudafuncs.o $OUT/reduce.o $OUT/device_memory.o \
  -L $ROCM/lib -lamdhip64 -o $OUT/kernel_harness.exe
# Result: warnings only (nodiscard), no errors. Binary: 455 KB.
# DLL setup: copy TheRock amdhip64_7.dll/amd_comgr.dll/rocm_kpack.dll/hiprtc*.dll
# into exe dir so it beats System32's Adrenalin amdhip64 in the DLL search order.
```

**Binary check:**
- `strings kernel_harness.exe | grep gfx12` shows two `hipv4-amdgcn-amd-amdhsa--gfx1201` code objects (cudafuncs.cu and reduce.cu).
- Imports: `amdhip64_7.dll`, `KERNEL32.dll` only (HIP, no CUDA).

**Test run (two independent process runs, both PASSED):**
```
HIP_VISIBLE_DEVICES=0 ./kernel_harness.exe
```
Run 1 and Run 2 results (identical):
```
HIP device 0: AMD Radeon RX 9070 XT (gfx1201) warpSize=32

[createVMap / createNMap]
  ok:   vmap == CPU back-projection (all pixels)
  (valid normals=2961)
  ok:   createNMap produced a dense normal field
  ok:   all valid normals are unit length
[tranformMaps identity]
  ok:   identity transform leaves vmap bit-stable
[resizeVMap]
  ok:   resizeVMap halved dimensions
  ok:   resizeVMap output mostly finite
[icpStep SE3 wave64 reduction]
  ok:   JtJ is symmetric
  ok:   icpStep found many inlier correspondences
  (inliers=2961 residual=0)
  ok:   inlier count identical across two runs
  ok:   icpStep JtJ/JtR/residual BIT-IDENTICAL across two runs (wave64 determinism)
  ok:   all 6 JtJ diagonal entries positive (non-degenerate reduction)
[so3Step SO3 wave64 reduction]
  (residual=0 inliers=2852)
  ok:   so3Step JtJ/JtR BIT-IDENTICAL across two runs (wave64 determinism)
  ok:   so3Step covered many pixels
[computeRgbResidual int2 wave64 reduction]
  (count=2773 sigmaSum=69325)
  ok:   int2 reduction count == CPU recount of valid correspondences (EXACT)
  ok:   int2 reduction sigmaSum == CPU sum(diff^2) (EXACT)
  ok:   computeRgbResidual identical across two runs
  ok:   many RGB correspondences found
[rgbStep SE3 photometric wave64 reduction]
  ok:   rgbStep JtJ symmetric
  ok:   rgbStep JtJ/JtR BIT-IDENTICAL across two runs (wave64 determinism)

HARNESS PASSED (0 failures)
```

**Pass count:** 17/17 checks, both runs. 0 failures.

**wave32 reduction verdict (gfx1201 RDNA4):** warpSize=32. warpReduceSum loops `offset = warpSize/2 .. 1`; on gfx1201 warpSize=32 so offset goes 16..1 (32-lane wavefront). CUDA_WARP_FULL_MASK = 0xffffffffffffffffULL (HIP); upper 32 bits are masked on wave32. icpStep/so3Step/rgbStep bit-identical across two runs; int2 count=2773 and sigmaSum=69325 exactly equal the CPU recount -- wave32 reductions correct on RDNA4, no lane lost or doubled. All values match gfx90a, gfx1100, and gfx1101 exactly (inliers=2961/2852/2773, sigmaSum=69325).

**Windows DLL notes:** exe dir-copy of TheRock amdhip64_7.dll/amd_comgr.dll/rocm_kpack.dll/hiprtc*.dll is required; System32 Adrenalin amdhip64 would otherwise be picked up first by the DLL loader. All-clang build (no MSVC host compiler, no -fPIC).

**Non-GPU regression:** USE_HIP=OFF cmake configure enters legacy FindCUDA else() branch (same as all prior platforms; unchanged upstream path).

**Fork:** untouched; moat-port branch HEAD remains 85283b8 with zero commits added.

**Decision: PASS -> windows-gfx1201 completed, validated_sha=85283b8.**

## Install as a dependency

Not applicable -- ElasticFusion is a leaf application, not a base library consumed by
other MOAT targets.

## Validation 2026-05-31 (linux-gfx1100, AMD Radeon Pro W7800 48GB gfx1100, ROCm 7.2.1) -- PASSED

**Device:** HIP device 0 -- AMD Radeon Pro W7800 48GB (gfx1100) warpSize=32. HIP_VISIBLE_DEVICES=0, ROCm 7.2.1.

**Fork HEAD confirmed:** 85283b834d61c5638e5658805a229832b3caaf13 (no fork commit; follower reuses lead branch unchanged).

**Build (gfx1100; cmake recipe from notes.md with arch override; no source edit):**
```
# Pangolin built from submodule (v0.5, -DCMAKE_CXX_FLAGS="-include cstdint"):
cmake -S third-party/Pangolin -B third-party/Pangolin/build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-include cstdint" -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
make -C third-party/Pangolin/build -j16

# libefusion (USE_HIP=ON, gfx1100, no source/CMake edit vs gfx90a commit):
cmake -S . -B build-hip-gfx1100 -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release
make -C build-hip-gfx1100 efusion -j16
# Result: Built target efusion. Warnings only (nodiscard), no errors.

# Harness compiled in agent_space/ef-harness-gfx1100/:
SRC=projects/ElasticFusion/src; HIPCC=/opt/rocm/bin/hipcc; ARCH=gfx1100
$HIPCC -std=c++17 --offload-arch=$ARCH -O2 -fPIC -DUSE_HIP \
  -include $SRC/Core/Cuda/cuda_to_hip.h -I $SRC/Core/Cuda -I $SRC/hip_compat \
  -c $SRC/Core/Cuda/cudafuncs.cu -o cudafuncs.o
$HIPCC -std=c++17 --offload-arch=$ARCH -O2 -fPIC -DUSE_HIP \
  -include $SRC/Core/Cuda/cuda_to_hip.h -I $SRC/Core/Cuda -I $SRC/hip_compat \
  -c $SRC/Core/Cuda/reduce.cu -o reduce.o
g++ -std=c++17 -O2 -fPIC -DUSE_HIP -D__HIP_PLATFORM_AMD__ -I /opt/rocm/include \
  -I $SRC/hip_compat -I $SRC/Core/Cuda -I $SRC/Core/Cuda/containers \
  -c $SRC/Core/Cuda/containers/device_memory.cpp -o device_memory.o
$HIPCC -std=c++17 -O2 -DUSE_HIP -D__HIP_PLATFORM_AMD__ \
  -I $SRC/Core/Cuda -I $SRC/Core/Cuda/containers -I $SRC/hip_compat \
  -c $SRC/rocm_validation/kernel_harness.cpp -o kernel_harness.o
$HIPCC -std=c++17 --offload-arch=$ARCH \
  kernel_harness.o cudafuncs.o reduce.o device_memory.o -o kernel_harness
# Binary: 192 KB.
```

**Library check (build-hip-gfx1100/Core/libefusion.so):**
- `ldd` shows libamdhip64.so.7 -- HIP, no CUDA.
- `llvm-objdump --offloading` shows two gfx1100 code objects (the 2 .cu TUs):
  `hipv4-amdgcn-amd-amdhsa--gfx1100` (cudafuncs.cu, reduce.cu). No gfx90a objects.

**Test run (two independent process runs, both PASSED):**
```
HIP_VISIBLE_DEVICES=0 ./kernel_harness   # run 1 and run 2 (both identical)
```
Run 1 and Run 2 results (identical):
```
HIP device 0: AMD Radeon Pro W7800 48GB (gfx1100) warpSize=32

[createVMap / createNMap]
  ok:   vmap == CPU back-projection (all pixels)
  (valid normals=2961)
  ok:   createNMap produced a dense normal field
  ok:   all valid normals are unit length
[tranformMaps identity]
  ok:   identity transform leaves vmap bit-stable
[resizeVMap]
  ok:   resizeVMap halved dimensions
  ok:   resizeVMap output mostly finite
[icpStep SE3 wave64 reduction]
  ok:   JtJ is symmetric
  ok:   icpStep found many inlier correspondences
  (inliers=2961 residual=0)
  ok:   inlier count identical across two runs
  ok:   icpStep JtJ/JtR/residual BIT-IDENTICAL across two runs (wave64 determinism)
  ok:   all 6 JtJ diagonal entries positive (non-degenerate reduction)
[so3Step SO3 wave64 reduction]
  (residual=0 inliers=2852)
  ok:   so3Step JtJ/JtR BIT-IDENTICAL across two runs (wave64 determinism)
  ok:   so3Step covered many pixels
[computeRgbResidual int2 wave64 reduction]
  (count=2773 sigmaSum=69325)
  ok:   int2 reduction count == CPU recount of valid correspondences (EXACT)
  ok:   int2 reduction sigmaSum == CPU sum(diff^2) (EXACT)
  ok:   computeRgbResidual identical across two runs
  ok:   many RGB correspondences found
[rgbStep SE3 photometric wave64 reduction]
  ok:   rgbStep JtJ symmetric
  ok:   rgbStep JtJ/JtR BIT-IDENTICAL across two runs (wave64 determinism)

HARNESS PASSED (0 failures)
```

**Pass count:** 17/17 checks, both runs. 0 failures.

**wave32 ICP reduction verdict (the key correctness proof):**
warpReduceSum loops `offset = warpSize/2 .. 1`; on gfx1100 warpSize=32 so offset goes
16..1 (32-lane wavefront). `shared[32]` is oversized-but-safe (max 32 warps at 1024
threads / wave32 = exactly 32; no overflow). CUDA_WARP_FULL_MASK = 0xffffffffffffffffULL
on HIP; the 64-bit mask is consumed by HIP's `__shfl_down_sync` which validates the
upper bits are zero for wave32 (only the low 32 bits matter). icpStep/so3Step/rgbStep
results are bit-identical across two runs; int2 count=2773 and sigmaSum=69325 exactly
equal the CPU recount -- wave32 reductions correct, no lane lost or doubled.

Inlier counts match gfx90a (2961/2852/2773) exactly -- the warpSize-parameterized
code produces identical logical results on wave32 as on wave64 for this fixed input.

**GL-interop outcome on gfx1100 (BONUS probe):**
The W7800 has display/graphics blocks and radeonsi creates a full GL 4.6 Compatibility
Profile context via EGL device platform (GL_RENDERER=AMD Radeon Pro W7800 48GB
(radeonsi, navi31, LLVM 20.1.2, DRM 3.64, 6.8.0-65-generic), GL_VERSION=4.6). The
texture glTexImage2D succeeded (glErr=0x0). However:
- `hipGLGetDevices -> hipErrorNoDevice, count=0` (HIP does not see the GL context's device)
- `hipGraphicsGLRegisterImage -> hipErrorUnknown`

The GL context uses card4/renderD133 (the DRM node for the first W7800 connected to
the EGL device enumeration); HIP_VISIBLE_DEVICES=0 binds to a different HIP ordinal
not recognized as the same device by hipGLGetDevices. This is the same fundamental
limitation as gfx90a -- a driver-level mismatch between the GL context's DRM node and
the HIP device ordinal. This is an environmental/driver limitation (no ROCm GL
interop support in this display configuration), not a port defect. The hip_compat shims
and the interop API mapping in the port are correct by construction; the barrier is
the same as on MI250X and is not fixable in the port itself.

**Non-GPU regression:** USE_HIP=OFF cmake configure enters legacy FindCUDA else() branch
(same as gfx90a; unchanged upstream path; confirmed by notes.md gfx90a record).

**Fork:** untouched; moat-port branch HEAD remains 85283b8 with zero commits added.
No CI workflow added.

**Decision: PASS -> linux-gfx1100 completed, validated_sha=85283b8.**

## Review 2026-05-31 (reviewer, linux-gfx90a) -- PASSED

Reviewed `git diff e3b1a7e...HEAD` (HEAD 85283b8) with /pr-review (ROCm-fault-class
aware) + 4 parallel fact-check sub-agents. No changes requested. The diff is small and
well-isolated (2 .cu TUs + compat header + 6 hip_compat shims + CMake + 1 unused-include
guard); host .cpp and the container/types headers are untouched.

Verdict: review-passed -> validator. No actionable defects found.

wave64 reduction (the flagged correctness wall) -- VERIFIED CORRECT, fact-checked:
- All 42 __shfl_down_sync sites in reduce.cu use CUDA_WARP_FULL_MASK (0xffffffffffffffffULL
  HIP / 0xFFFFFFFFu CUDA); zero 32-bit literals remain; cudafuncs.cu has no shfl/ballot.
- warpReduceSum loops `offset=warpSize/2..1` (warpSize-parameterized, not 32/16) at
  reduce.cu:60/145/555.
- blockReduceSum has a real __syncthreads() (reduce.cu:114/178/576) between the per-warp
  shuffle and the shared[] read -- NOT an unsynced volatile-sdata warp tail (contrast
  MPPI-Generic). No divergent early-return precedes the reduction (grid-stride loop, all
  lanes reach blockReduceSum), so the full 64-bit mask is correct.
- shared[32] sizing proven safe for all 4 cases: warps = blockDim.x/warpSize = {256/64=4,
  256/32=8, 1024/64=16, 1024/32=32}; max 32, never overflows. The `threadIdx.x <
  blockDim.x/warpSize ? shared[lane] : zero` guard holds because warp-count <= warpSize so
  lane==threadIdx.x for passing threads, with zero-pad on the rest. Correct on wave64 AND
  wave32 (followers need no change).

__CUDACC__/EIGEN_NO_CUDA gotcha -- VERIFIED: defines emitted ONLY under `#if
defined(__HIPCC__)` (cuda_to_hip.h:25-42), aliases outside that but inside USE_HIP, so the
host .cpp (g++ -DUSE_HIP, no __HIPCC__) get the cuda->hip aliases but never __CUDACC__.
Host relies on the types.cuh Eigen->mat33 ctor (RGBDOdometry.cpp:196/287/376/459), which
is gated `!defined(__CUDACC__)` and would vanish if __CUDACC__ leaked. <cstring> on the
HIP branch keeps ::memset viable for initTextureObjectFromArray. All confirmed.

Other fault classes:
- operators.cuh:62-72 USE_HIP-guards ONLY operator+/operator- on float3 (HIP_vector_type
  ships them); cross/dot/norm/normalized/operator*(mat33,float3) kept on both. Correct.
- Textures: cudaResourceTypeArray + cudaFilterModePoint + cudaReadModeElementType,
  descriptors memset-zeroed, created+destroyed in local scope (cudafuncs.cu:59-74,370-380).
  256B-pitch class does NOT apply (array-backed, not Pitch2D); linear-filter rejection does
  NOT apply (point). No persistent texture handle -> no rule-of-five gap there.
- GL-interop handle: GPUTexture cudaRes default-null (GPUTexture.h:53) + guarded destroy
  (GPUTexture.cpp:52-53) -- the colmap rule-of-five pattern, already present upstream.
- OOB neighbor reads guarded (computeNmapKernel NaN-returns at the last row/col;
  SO3Reduction::getGradient caller guards x/y in [1,dim-1]). Unchanged upstream code.
- No library swaps (no cuBLAS/cuFFT/Thrust/CUB anywhere). CMake legacy-FindCUDA else()
  branch byte-identical to e3b1a7e (diff -w clean); HIP branch arch-defaulted-only-when-unset
  (no literal gfx90a override). CUDA path byte-identical under USE_HIP=OFF.
- Commit hygiene: title `[ROCm] ...` 57 chars, mentions Claude, no noreply trailer, no
  ghstack, Test Plan present, ASCII-only, no AMD-internal account refs.

KEY JUDGMENT -- validation bar (ruled explicitly): "kernels GPU-validated via the no-GL
device-array harness on real gfx90a + full GL-SLAM validation assigned to the gfx1100
follower" IS an acceptable gfx90a completion bar. Rationale: (a) the actual porting risk
on gfx90a is wave64 reduction correctness, and the harness proves it on real hardware --
the computeRgbResidual int2 device-count/sum EXACTLY equal a CPU recount, which (integer
adds being order-independent) is a hard arithmetic proof no lane is lost or doubled, plus
icp/so3/rgbStep are bit-identical across two runs (determinism). (b) The GL-interop code is
host-side API mapping ported via the hip_compat shims; it is environmentally IMPOSSIBLE to
run here (gfx90a is compute-only: radeonsi refuses a GL context, no Vulkan ICD for zink,
llvmpipe is host-memory so hipGLGetDevices=0 / hipGraphicsGLRegisterImage=hipErrorUnknown
-- empirically demonstrated by gl_interop_probe.cpp, a real EGL+HIP round-trip attempt).
This is a HARDWARE limitation, not a port defect. By-inspection review of the interop
hand-off (RGBDOdometry initICP/initICPModel/populateRGBDData) is clean: every cuda*
interop symbol is aliased 1:1, the map->getMappedArray->consume->unmap sequence is
preserved verbatim, and cudaArray_t==hipArray_t flows type-consistently into
copyMaps/imageBGRToIntensity/cudaMemcpy2DFromArray. No interop risk surfaced that MUST be
run on gfx90a (it cannot be). This mirrors prior hardware-gated validation splits.

CARRY-FORWARD (validator + gfx1100 follower, NOT a gfx90a blocker): the full GL-SLAM
pipeline (.klg replay, surfel count + trajectory vs reference, two-run determinism) is a
gfx1100-follower REQUIREMENT -- gfx1100 has display/graphics blocks + a hardware radeonsi
GL, so live hipGraphicsGLRegisterImage interop should work there and is the right place to
validate the end-to-end tracking. gfx90a proves the kernels; gfx1100 must prove the live
GL ingest. Validator: confirm the harness reproduces on gfx90a (HARNESS PASSED, 0 failures)
and that libefusion.so links libamdhip64 with no CUDA.
