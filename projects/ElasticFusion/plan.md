# ElasticFusion -- ROCm/HIP port plan (lead platform: linux-gfx90a)

## Project
- Name: ElasticFusion -- real-time dense visual SLAM, surfel-based, RGB-D.
- Upstream: https://github.com/mp3guy/ElasticFusion
- Default branch: master (only branch; HEAD e3b1a7e "Merge branch 'RMonica-new_cuda_texture'").
- License: Imperial College non-commercial. (Fork/port allowed for non-commercial; PR is to a non-commercial-licensed upstream -- note for the user gate.)
- Cloned read-only into projects/ElasticFusion/src.

## Existing AMD support -- finding and decision
- No HIP/ROCm path anywhere: zero `hip`/`rocm`/`amdgpu` references in the tree, only a `master` branch upstream, no AMD/HIP fork or PR found via `gh search`. The README FAQ explicitly says non-NVIDIA users must "rewrite the tracking code."
- Not even an OpenCL/Vulkan compute path: the *non-tracking* pipeline is OpenGL GLSL (portable), but the *tracking* (ICP/RGB/SO3) is CUDA-only. So a ROCm/HIP port of the CUDA tracking code is the only GPU-compute path on AMD.
- Decision: PROCEED with a fresh CUDA->HIP port (Strategy A). Not a skip, not a finish-abandoned-port, not an improve-existing. This is a correctness-first mechanical port; no AMD-native rewrite is warranted (the kernels are memory-bound per-pixel maps and small fixed-size reductions, not CUTLASS/GEMM/attention).

## Build classification -- cmake (NOT a torch extension)
- Evidence: top-level `CMakeLists.txt` uses `find_package(CUDA REQUIRED)` (legacy FindCUDA module) and `find_package(Pangolin/OpenNI2/SuiteSparse/BLAS/LAPACK)`. No `find_package(Torch)`, no `torch.utils.cpp_extension`, no setup.py/pyproject. `Core/CMakeLists.txt` compiles `.cu` via the legacy `CUDA_COMPILE(cuda_objs ${cuda})` + `CUDA_NVCC_FLAGS` + `CMakeModules/CudaDetect.cmake`/`CudaComputeTargetFlags.cmake` (nvcc `-gencode`).
- => Strategy A (compat-header + enable_language(HIP) + LANGUAGE HIP).
- Wrinkle vs the colmap exemplar: ElasticFusion uses the *legacy* `FindCUDA`/`CUDA_COMPILE` flow, not modern first-class `enable_language(CUDA)`. So the port cannot simply retag existing first-class CUDA sources; it must replace the `CUDA_COMPILE` path with a HIP path. Cleanest: under `if(USE_HIP)`, `enable_language(HIP)`, drop the `CudaDetect`/`CudaComputeTargetFlags`/`CUDA_COMPILE` block, mark the 2 `.cu` files `LANGUAGE HIP`, and add them straight into the `efusion` target's sources (no separate `cuda_objs`). The legacy CUDA path stays intact in the `else()`.

## Port strategy -- A (compat header + LANGUAGE HIP), rationale
- Tiny, well-isolated CUDA surface: exactly two `.cu` TUs (`Core/Cuda/cudafuncs.cu`, `Core/Cuda/reduce.cu`) plus headers under `Core/Cuda/`. No CUDA libraries (no cuBLAS/cuFFT/cuRAND/cuSPARSE/Thrust/CUB -- confirmed by grep). Host C++ stays on the plain CUDA spelling; only the `.cu`/`.cuh` see HIP.
- Add one compat header `Core/Cuda/cuda_to_hip.h` aliasing only the symbols used: `cudaMalloc/Free/MallocPitch/Memcpy/Memcpy2D/Memcpy2DFromArray/MemcpyToSymbol/DeviceSynchronize/GetLastError/GetErrorString`, `cudaError_t/cudaSuccess`, the texture-object API (`cudaTextureObject_t/cudaResourceDesc/cudaTextureDesc/cudaCreateTextureObject/cudaDestroyTextureObject/cudaResourceTypeArray/cudaFilterModePoint/cudaAddressModeClamp/cudaReadModeElementType/tex2D`), `cudaArray_t`, and the GL-interop calls (`cudaGraphicsGLRegisterImage/MapResources/SubResourceGetMappedArray/UnmapResources/UnregisterResource/cudaGraphicsResource/cudaGraphicsRegisterFlagsReadOnly`, header `cuda_gl_interop.h` -> `hip/hip_gl_interop.h`). All have 1:1 hip* spellings (verified in `/opt/rocm/include/hip` and in pytorch hipify `cuda_to_hip_mappings.py`).
- The GL-interop calls live in HOST `.cpp` files (`Core/GPUTexture.cpp`, `Core/Utils/RGBDOdometry.cpp`), not in `.cu`. Per the compat-header approach, host files keep the CUDA spelling and the compat header is only for HIP TUs -- but these host files must alias the interop symbols too under HIP. Use the MPPI-Generic forwarding-shim trick: a `hip_compat/` dir holding `cuda_gl_interop.h`/`cuda_runtime_api.h`/`driver_types.h` shims (each `#include "../Core/Cuda/cuda_to_hip.h"`), added to the include path only under `if(USE_HIP)`, so `GPUTexture.h`'s `#include <cuda_gl_interop.h>` resolves to the shim on AMD and the real toolkit header on NVIDIA -- zero edits to the include lines.
- CMake: `option(USE_HIP OFF)`; if set, `enable_language(HIP)`, default `CMAKE_HIP_ARCHITECTURES` to `gfx90a` ONLY when unset (never a literal -- so gfx1100/gfx1151 followers need no CMake edit, per the CudaSift/Gpufit lesson), `set_source_files_properties(${cuda} PROPERTIES LANGUAGE HIP)`, `set_target_properties(efusion PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")`, and `-include Core/Cuda/cuda_to_hip.h` in `CMAKE_HIP_FLAGS` so the compat defines precede every include.

## CUDA surface inventory (complete; ex third-party)
Two `.cu` TUs, five `.cuh` headers, one container layer. No streams, no events, no pinned/managed memory, no async copies, no atomics, no surfaces, no CUDA libraries.

Kernels in `Core/Cuda/cudafuncs.cu` (all embarrassingly parallel, per-pixel, block (32,8)):
- pyrDownGaussKernel, computeVmapKernel, computeNmapKernel, tranformMapsKernel, copyMapsKernelTex, pyrDownKernelGaussF, resizeMapKernel<bool>, pyrDownKernelIntensityGauss, verticesToDepthKernel, bgr2IntensityKernel, applyKernel (Sobel), projectPointsKernel.
- `__constant__ float gsobel_x3x3[9]`, `gsobel_y3x3[9]` set via `cudaMemcpyToSymbol` (1:1 HIP).
- Texture objects: `initTextureObjectFromArray` -> `cudaResourceTypeArray` (cudaArray-backed), `cudaFilterModePoint`, `cudaReadModeElementType`; read with `tex2D<float4>`/`tex2D<uchar4>`. Created and destroyed in local scope per call.

Kernels/reductions in `Core/Cuda/reduce.cu`:
- warpReduceSum / blockReduceSum / reduceSum overloaded for `JtJJtrSE3` (27+2 floats), `JtJJtrSO3` (9+2), `int2`. Functor structs ICPReduction, RGBReduction, RGBResidual, SO3Reduction; kernels icpKernel/rgbKernel/residualKernel/so3Kernel. Host entry points: icpStep, rgbStep, computeRgbResidual, so3Step.
- Reduction uses `__shfl_down_sync(0xFFFFFFFF, x, offset)` with `offset = warpSize/2 .. 1`, and `static __shared__ T shared[32]`. Launch configs `reduceThreads=256`, `reduceBlocks=64`, `MAX_THREADS=1024`.

Container layer `Core/Cuda/containers/` (PCL-derived): `cudaMalloc`/`cudaMallocPitch`/`cudaMemcpy`/`cudaMemcpy2D` only; `PtrStep/PtrStepSz`. `GPU_HOST_DEVICE__` and `types.cuh` gate on `__CUDACC__`.

CUDA-OpenGL interop (host `.cpp`, on the SLAM critical path -- see Risk #1):
- `Core/GPUTexture.cpp`: `cudaGraphicsGLRegisterImage(&cudaRes, glTexId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly)`; `cudaGraphicsUnregisterResource` in dtor.
- `Core/Utils/RGBDOdometry.cpp` (x7): `cudaGraphicsMapResources` + `cudaGraphicsSubResourceGetMappedArray` -> `cudaArray_t`, then `cudaMemcpy2DFromArray` (depth) or fed to `copyMaps`/`imageBGRToIntensity` (texture-read), then `cudaGraphicsUnmapResources`.

ROCm/HIP mapping: every symbol above has a 1:1 hip* equivalent present in ROCm 7.2.1 (`hipMalloc`, `hipMemcpy2DFromArray`, `hipCreateTextureObject`+`tex2D`, `hipGraphicsGLRegisterImage`, ...). No library swaps required.

## Risk list (ranked; keyed to PORTING_GUIDE fault classes + precedents)

1. CUDA-OpenGL interop on a headless ROCm node -- HIGHEST RISK, likely the hardest wall.
   - Why it is core, not GUI: tracking inputs reach CUDA *only* through GL textures. `processFrame` builds the surfel prediction (vertex/normal/image) with GLSL splatting shaders into FBO textures (`indexMap.vertexTex()`, `fillIn.vertexTexture`), and `RGBDOdometry::initICPModel`/`initICP`/`populateRGBDData` map those GL textures (and the filtered-depth / RGB GL textures) to `cudaArray_t` via interop to feed ICP/RGB. There is NO CUDA-only ingest path. So a "headless build with no Pangolin / GL" that still exercises the fusion+tracking core is NOT feasible -- GL is load-bearing for the algorithm, not decoration. (Contrast Open3D/cupoch/RXMesh, whose compute cores run without GL; ElasticFusion does not.)
   - API is fine: `hip_gl_interop.h` and `hipGraphicsGLRegisterImage/MapResources/SubResourceGetMappedArray/UnregisterResource` exist in ROCm 7.2.1 and in hipify's map. The risk is RUNTIME: `hipGraphicsGLRegisterImage` requires the GL context to live on the SAME amdgpu device as the HIP device. On NVIDIA, GL+CUDA share one driver; on AMD, GL is Mesa (radeonsi/zink) and HIP is ROCm -- they share buffers only when GL runs on the amdgpu DRM render node (radeonsi/zink), NOT under llvmpipe/swrast software GL.
   - Host probe already done: ROCm 7.2.1, gfx90a MI250X; `radeonsi_dri.so` + `zink_dri.so` present; `/dev/dri` exposes card0-4 + renderD128-131; `Xvfb`, `libEGL`, `libGL` present; `mesa-utils`/`glxinfo` NOT installed. So a hardware-accelerated AMD GL context via EGL-surfaceless on a renderD node is plausible but UNPROVEN here.
   - Porter's FIRST experiment (gate the whole port on it): a standalone repro in agent_space that (a) creates an EGL/GL context on the amdgpu render node (radeonsi or zink, NOT llvmpipe), (b) makes a GL_TEXTURE_2D, (c) `hipGraphicsGLRegisterImage` + map + `hipGraphicsSubResourceGetMappedArray` + read back, and checks the data round-trips. Install `mesa-utils` + any radeonsi/zink runtime via apt as needed.
   - Fallbacks if AMD GL+HIP interop does NOT share on this node, in order of preference:
     a) Force Mesa GL onto amdgpu: `__GLX_VENDOR_LIBRARY_NAME=mesa`, `MESA_LOADER_DRIVER_OVERRIDE=radeonsi` or `zink`, `EGL_PLATFORM=surfaceless`, point at `/dev/dri/renderD12x`; run under `xvfb-run` only if a GLX window is still required.
     b) If interop cannot share, REPLACE the interop hand-off with an explicit GL-readback bounce: `glGetTexImage`/PBO -> host (or GL->host->`hipMemcpy`) -> device buffer, behind the same `RGBDOdometry`/`copyMaps`/`imageBGRToIntensity` entry points (they already accept a `cudaArray_t`; swap to a device-pointer path under USE_HIP). Correctness-preserving, slower, but unblocks GPU validation of the tracking kernels. This is the principled degrade and keeps the diff inside the compat boundary.
   - This risk decides project feasibility; if neither (a) nor (b) yields a working core on this hardware after a real attempt, set linux-gfx90a `blocked` with the concrete reason (AMD GL/HIP interop unavailable headless) rather than guessing.

2. wave64 in the ICP/RGB/SO3 reductions -- the likely correctness wall the task flags.
   - `warpReduceSum` loops `offset = warpSize/2 .. 1`: `warpSize` is the HIP builtin (64 on gfx90a), so the loop spans the full 64-lane wavefront -- structurally correct on wave64. `blockReduceSum`'s `shared[32]` is sized for 32 warps; with blockDim 256/1024 and wave64 there are 4/16 wavefronts, so `shared[32]` is oversized-but-safe and the zero-padding (`val = (tid < blockDim.x/warpSize) ? shared[lane] : zero`) covers the unused lanes. So unlike MPPI-Generic, there is NO unsynchronized `volatile sdata` warp-tail to race -- this uses real `__shfl_down_sync` data exchange. Net: probably correct as-is on wave64.
   - The ONE real concern: the lane mask is a 32-bit `0xFFFFFFFF` while the wavefront is 64 lanes. On HIP `__shfl_down_sync`'s mask is effectively ignored (all active lanes participate), so the 64-wide reduction should still be complete. Must be PROVEN, not assumed: validate with a fixed-input determinism + value check on the reduction (two runs bit-identical, and the 27/9-float JtJ sum matches a CPU recomputation of the same products). If a divergence appears, the clean fix is to keep `__shfl_down` over `warpSize` (drop the spurious mask, or use the maskless `__shfl_down`) -- do NOT hardcode 32 (PORTING_GUIDE warp-size rule; also re-validated on gfx1100/gfx1151 wave32 where warpSize=32 makes the same code reduce a 32-lane wave correctly).
   - Also: `__shfl_down_sync` is the *_sync variant; ensure the compat/HIP path provides it (ROCm does). The block-collective TempStorage race (CV-CUDA) does NOT apply (no cub/hipCUB here).

3. `__CUDACC__` not defined by hipcc -- compile-time wall (MPPI-Generic precedent).
   - `Core/Cuda/containers/kernel_containers.hpp:43` gates `GPU_HOST_DEVICE__` (= `__host__ __device__ __forceinline__`) on `#if defined(__CUDACC__)`; `Core/Cuda/types.cuh:58,70` gate the Eigen include / Eigen ctor on `#if !defined(__CUDACC__)`. hipcc defines `__HIPCC__`, not `__CUDACC__`. If left as-is, on HIP `GPU_HOST_DEVICE__` collapses to empty (PtrStep::ptr becomes host-only -> "reference to __host__ function in __device__ function"), and `types.cuh` would pull `<Eigen/Core>` into the device pass.
   - Fix: in the compat header, `#define __CUDACC__ 1` when `__HIPCC__` is defined (MPPI-Generic). This makes `kernel_containers.hpp` decorate correctly and `types.cuh` correctly EXCLUDE the host-only Eigen ctor from the device TU. Verify Eigen is not otherwise dragged into device code (it is only included in `types.cuh` under `!__CUDACC__`, and host `.cpp` build mat33 from Eigen -- so define `EIGEN_NO_CUDA` defensively if any Eigen header reaches the device pass; likely unnecessary here since no Eigen in `.cu`).

4. float3/float4 operator overloads (`Core/Cuda/operators.cuh`) -- compile-time ambiguity (MPPI-Generic).
   - The project hand-rolls `operator-`/`operator+`/`operator*(mat33,float3)` plus non-operator `cross`/`dot`/`norm`/`normalized` for `float3`. HIP's `HIP_vector_type` provides the full set of float3/4 arithmetic operators, so the hand-rolled `operator+`/`operator-` (and any vec-scalar) will hit "use of overloaded operator '+' is ambiguous" under hipcc. Note: ElasticFusion uses raw `float3`/`float4` from `<vector_types.h>` (NOT a custom vec type), and HIP's `float3` IS `HIP_vector_type<float,3>` with operators -- so the collision is real.
   - Fix: USE_HIP-guard the project's `operator+`/`operator-` (and `operator*` if it collides) in `operators.cuh`; KEEP `cross`/`dot`/`norm`/`normalized` (HIP does not provide those). Verify with a one-line probe before deleting (MPPI-Generic method).

5. Texture fault classes -- mostly DO NOT apply, but verify the resource type holds post-interop.
   - The texture objects are `cudaResourceTypeArray` (cudaArray-backed), `cudaFilterModePoint`, `cudaReadModeElementType`. So: linear-filter rejection (popsift/gpuRIR) does NOT apply (point filtering); 256B pitch (colmap/CudaSift) does NOT apply (array-backed, not `cudaResourceTypePitch2D` -- and per the CudaSift addendum I confirmed the resource type is genuinely Array, not assumed); layered-array collapse (popsift) does NOT apply (no `cudaArrayLayered`); rule-of-five on texture handles (colmap) does NOT apply (objects are local-scope, created+destroyed per call). The arrays come from GL interop, so the only texture risk is subsumed by Risk #1 (does the interop'd `cudaArray_t` read back correctly on AMD).

6. Out-of-bounds neighbor reads -- low risk (colmap ComputeDOG class).
   - `computeNmapKernel` reads `[u+1]`/`[v+1]` but guards `if (u == cols-1 || v == rows-1) return;` first. `SO3Reduction::getGradient` reads `[x-1]/[x+1]/[y-1]/[y+1]` but the caller guards `x>=1 && x<cols-1 && y>=1 && y<rows-1`. `resizeMapKernel` reads `[xs+1]/[ys+1]` over a half-res grid (in-bounds). pyr/Sobel kernels clamp with min/max. So edges are guarded; CUDA-lenient-OOB is unlikely. Spot-check during bringup but no change expected.

7. Legacy FindCUDA -> HIP CMake conversion -- build-system risk (no prior MOAT precedent for legacy CUDA_COMPILE).
   - Must replace `CUDA_COMPILE`/`CudaDetect`/`CudaComputeTargetFlags`/`CUDA_NVCC_FLAGS` with `enable_language(HIP)` + `LANGUAGE HIP` under `if(USE_HIP)`, keeping the legacy CUDA branch untouched in `else()`. Also drop `--expt-relaxed-constexpr;--disable-warnings` (nvcc-only) from the HIP flags. Keep the diff localized to `CMakeLists.txt` + `Core/CMakeLists.txt`.

8. Dependency build (Pangolin/OpenNI2/Eigen/Sophus/SuiteSparse) on Ubuntu 24.04 / ROCm 7.2.1 -- environmental, the notes.md "verify build prerequisites first" warning.
   - Submodules: Pangolin (stevenlovegrove), OpenNI2 (occipital), Eigen (eigenteam mirror), Sophus (strasdat). These are old pins; Pangolin in particular may not build clean on a modern toolchain. OpenNI2 is only needed for LIVE sensors -- for `.klg` log replay (the validation path) it can likely be stubbed/disabled. SuiteSparse/BLAS/LAPACK via apt. Porter installs deps via apt/conda (allowed without asking); if a submodule is unbuildable, prefer the distro Pangolin/Eigen/Sophus packages. This is bringup friction, not a fault class.

## File-by-file change list
- `Core/Cuda/cuda_to_hip.h` (NEW): the compat header -- HIP runtime + GL-interop include, `#define __CUDACC__ 1` (when `__HIPCC__`), and 1:1 cuda*->hip* aliases for only the symbols inventoried above. On non-HIP, includes `<cuda_runtime.h>`/`<cuda_gl_interop.h>` and is otherwise a no-op.
- `hip_compat/{cuda_gl_interop.h,cuda_runtime_api.h,driver_types.h}` (NEW): forwarding shims (each includes `../Core/Cuda/cuda_to_hip.h`), added to the include path ONLY under `if(USE_HIP)`, so host `.cpp` includes of those toolkit headers resolve on AMD without editing include lines.
- `CMakeLists.txt` (top): add `option(USE_HIP OFF)`; under USE_HIP add the `hip_compat/` include dir and skip `find_package(CUDA)` (or make it conditional), keep CUDA path in else.
- `Core/CMakeLists.txt`: under `if(USE_HIP)`: `enable_language(HIP)`, default `CMAKE_HIP_ARCHITECTURES=gfx90a` only-when-unset, mark `${cuda}` `LANGUAGE HIP`, add `.cu` directly to `efusion` (drop `CUDA_COMPILE`/`cuda_objs`/`CudaDetect`/`CudaComputeTargetFlags`), set `HIP_ARCHITECTURES` on the target, add `-include .../cuda_to_hip.h` to `CMAKE_HIP_FLAGS`. Keep the legacy CUDA branch in `else()`.
- `Core/Cuda/operators.cuh`: USE_HIP-guard the colliding `operator+`/`operator-`/`operator*` (keep dot/cross/norm/normalized).
- `Core/Cuda/reduce.cu`: only if the wave64 value/determinism check (Risk #2) fails -- adjust the `__shfl_down_sync` mask handling (never hardcode 32). Expected: no change.
- `Core/Utils/RGBDOdometry.cpp` + `Core/Cuda/cudafuncs.cu` (`copyMaps`/`imageBGRToIntensity`): only if interop cannot share (Risk #1 fallback b) -- add a USE_HIP GL-readback device-pointer path behind the existing entry points.
- No changes to the GLSL shaders, the surfel/fusion C++, or the container layer (beyond the `__CUDACC__` define landing via the forced-include).

## Build commands (gfx90a)
Dependencies first (porter, via apt; exact set finalized at bringup):
```
sudo apt-get update && sudo apt-get install -y \
  libeigen3-dev libsuitesparse-dev libglew-dev freeglut3-dev \
  zlib1g-dev libjpeg-dev mesa-utils libegl1-mesa-dev   # + Pangolin (submodule or distro)
```
Configure + build:
```
cd projects/ElasticFusion/src
cmake -S . -B build-hip \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j
```
(Followers reuse the same commit with only `-DCMAKE_HIP_ARCHITECTURES=gfx1100` / `gfx1151`; no source or CMake edit.)
Optional CPU-only compile smoketest: `rocm/dev-ubuntu-24.04:7.2.4-complete` (compile-only, never a validation gate; do not wire into Actions).

## Test plan
ElasticFusion ships NO gtest/ctest suite (confirmed: no `enable_testing`/`add_test`/gtest). It is an application validated by replaying a dataset and checking the SLAM outputs.

GPU validation gate (what "validated on gfx90a" means here):
1. Fetch a known RGB-D sequence as `.klg`: the upstream sample `dyson_lab.klg`, and/or an ICL-NUIM living-room log (`-icl` flag for the negative-focal-length normal flip). These give a fixed, deterministic input.
2. Run headless replay to completion and dump outputs:
   `./ElasticFusion -l <seq>.klg -q [-icl]`  (`-q` quits + triggers `savePly()` at end; trajectory `.freiburg` written too). Run under the GL context established for Risk #1 (radeonsi/zink on renderD via EGL-surfaceless, `xvfb-run` if a window is needed).
3. Compare against a reference run (NVIDIA, or the pre-port CUDA build, or a published trajectory) on the SAME sequence:
   - Final surfel count: `getGlobalModel().lastCount()` (MainController prints it; also the `.ply` vertex count). Expect a stable count per sequence; accept a small tolerance (surfel fuse/cull has float-order sensitivity, so define e.g. within a few % of reference rather than bit-exact).
   - Estimated trajectory: the `.freiburg` file (TUM `timestamp tx ty tz qx qy qz qw`). Compare against the reference trajectory (ATE/RPE, or a tight per-pose epsilon) -- this exercises ICP+RGB+SO3 tracking end to end, which is exactly the ported CUDA path.
   - Determinism: two AMD runs of the same sequence must agree (fingerprints the wave64 reduction non-determinism class from popsift/MPPI; a stable-but-varying secondary count is the warning sign).
4. Kernel-level confidence (decouples interop from kernels): a small standalone harness that feeds fixed host buffers straight into `icpStep`/`rgbStep`/`so3Step`/`createVMap`/`createNMap` (device-array path, no GL) and checks the JtJ/JtR sums + vmap/nmap against a CPU recomputation. This validates the wave64 reduction and the map kernels even if interop bring-up lags, and isolates a tracking-math bug from an interop bug.

Non-GPU regression set: none in-repo (no CPU test suite). "Do not regress" reduces to: the NVIDIA/CUDA build must still configure and build unchanged with `USE_HIP=OFF` (the compat header and CMake branch are no-ops on CUDA) -- verify by a CUDA-path configure (compile-only is acceptable as it is the unchanged upstream path).

## Open questions
- Does AMD GL (radeonsi or zink) + HIP interop actually share buffers via `hipGraphicsGLRegisterImage` on THIS headless MI250X node? (Risk #1 -- the porter's first experiment; gates feasibility. If no, fall back to GL-readback bounce; if that also fails, block with a concrete reason.)
- Reference data for the comparison: is there a published trajectory/surfel count for `dyson_lab.klg` or an ICL-NUIM sequence, or must the porter capture a reference from the upstream CUDA build on an NVIDIA box? (A pre-port CUDA build on this repo, run once on NVIDIA if available, is the cleanest reference; otherwise ICL-NUIM has ground-truth trajectories for ATE.)
- Will the pinned submodule Pangolin build on Ubuntu 24.04 / modern GCC, or should the port depend on the distro Pangolin? (Bringup; porter decides, may patch or swap to distro package.)
- Surfel-fusion determinism tolerance: how tight can the surfel-count / trajectory match be before float-order differences (FMA contraction -- pin `-ffp-contract=on` per CV-CUDA if bit-closeness is wanted) make it flaky? Define the acceptance band during bringup against the reference.

## Follower platforms
gfx1100/gfx1151 (RDNA, wave32): reuse this gfx90a fork branch, validate first, delta-port only on failure. The wave64 reduction code is warpSize-parameterized, so on wave32 `warpSize=64`->32 and the same `__shfl_down`/`shared[32]` logic reduces a 32-lane wave correctly -- expected to pass without change. The dominant follower risk is the SAME GL/HIP interop question on a different GL stack (gfx1100 is a display GPU with a real radeonsi GL, so interop may actually be EASIER there; gfx1151 is an APU on the Windows HIP SDK where GL interop and Pangolin may be the gating issue). No re-plan; append a `## Delta plan: <platform>` here if a real divergence appears.
