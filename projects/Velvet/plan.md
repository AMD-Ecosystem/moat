# Plan: Velvet

## Project

- **Name**: Velvet
- **Upstream**: https://github.com/vitalight/Velvet
- **Default branch**: main
- **Description**: CUDA-accelerated cloth simulation engine based on Extended Position Based Dynamics (XPBD)

## Existing AMD support

**Assessment**: No existing AMD/ROCm/HIP support found.

Searches performed:
- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/` -- no matches
- `gh api repos/vitalight/Velvet/forks` -- 20+ forks examined, none with rocm/hip/amd in name
- WebSearch for "Velvet vitalight ROCm AMD GPU HIP" -- no relevant results
- WebSearch for "Velvet XPBD cloth simulation AMD ROCm" -- no AMD port found

**Decision**: Proceed with from-scratch HIP port. No existing AMD work to validate or improve.

## Build classification

**Classification**: Visual Studio project (vcxproj + sln), NOT pure CMake, NOT pytorch extension.

**Evidence**:
- `Velvet.sln` and `Velvet/Velvet.vcxproj` (VS2019, C++17, CUDA 11.1)
- `$(VCTargetsPath)\BuildCustomizations\CUDA 11.1.props` / `.targets` in vcxproj
- No CMakeLists.txt, no setup.py, no pyproject.toml, no torch dependency
- Dependencies via vcpkg: glfw3, glad, fmt, glm, assimp, imgui

**Build system note**: The upstream is Windows-only (Visual Studio + vcpkg). For Linux/ROCm, a CMake build system must be created as part of the port. This is a non-trivial addition but straightforward given the clear project structure.

## Port strategy

**Strategy**: Create CMake build system + Strategy A (cuda_to_hip.h compat header)

**Rationale**:
1. No existing CMakeLists.txt means we must create one anyway
2. Pure CUDA simulation code (not pytorch extension)
3. Small CUDA surface (2 .cu files, 3 .cuh files) makes compat header tractable
4. OpenGL interop for CUDA-GL buffer sharing must be ported to HIP-GL interop (hipGraphics*)

The project is a standalone XPBD cloth simulator with OpenGL visualization. The CUDA code handles physics simulation while OpenGL handles rendering. The CUDA-OpenGL interop via `cudaGraphicsGLRegisterBuffer` maps to `hipGraphicsGLRegisterBuffer`.

## CUDA surface inventory

### Source files

| File | Type | Description |
|------|------|-------------|
| VtClothSolverGPU.cu | device | XPBD constraint solvers, collision, normals |
| SpatialHashGPU.cu | device | Spatial hashing for neighbor finding |
| Common.cuh | header | CUDA helpers, macros, memory allocation |
| VtClothSolverGPU.cuh | header | Solver function declarations, SDFCollider struct |
| SpatialHashGPU.cuh | header | Hash params struct, HashObjects declaration |

### Kernels (`__global__`)

| Kernel | Location | Notes |
|--------|----------|-------|
| InitializePositions_Kernel | VtClothSolverGPU.cu:30 | Simple position transform |
| PredictPositions_Kernel | VtClothSolverGPU.cu:42 | Velocity integration |
| SolveStretch_Kernel | VtClothSolverGPU.cu:65 | Distance constraint solver |
| SolveBending_Kernel | VtClothSolverGPU.cu:117 | Dihedral angle constraint |
| SolveAttachment_Kernel | VtClothSolverGPU.cu:205 | Attachment/pin constraint |
| ApplyDeltas_Kernel | VtClothSolverGPU.cu:253 | Jacobi averaging |
| CollideSDF_Kernel | VtClothSolverGPU.cu:289 | SDF collision response |
| CollideParticles_Kernel | VtClothSolverGPU.cu:329 | Particle-particle collision |
| Finalize_Kernel | VtClothSolverGPU.cu:388 | Velocity/position update |
| ComputeTriangleNormals | VtClothSolverGPU.cu:419 | Normal computation |
| ComputeVertexNormals | VtClothSolverGPU.cu:443 | Normal normalization |
| ComputeParticleHash_Kernel | SpatialHashGPU.cu:34 | Spatial hash computation |
| FindCellStart_Kernel | SpatialHashGPU.cu:44 | Cell boundary finding |
| CacheNeighbors_Kernel | SpatialHashGPU.cu:79 | Neighbor caching |

### Device functions (`__device__`)

| Function | Location | Notes |
|----------|----------|-------|
| AtomicAdd (vec3) | VtClothSolverGPU.cu:13 | Reordered atomic for vec3 |
| ComputeFriction | VtClothSolverGPU.cu:272 | Friction computation |
| SDFCollider::sgn | VtClothSolverGPU.cuh:20 | Sign helper |
| SDFCollider::ComputeSDF | VtClothSolverGPU.cuh:22 | SDF evaluation |
| SDFCollider::VelocityAt | VtClothSolverGPU.cuh:91 | Collider velocity |
| length2 | Common.cuh:48 | Squared length |
| ComputeIntCoord | SpatialHashGPU.cu:13 | Grid coordinate |
| HashCoords | SpatialHashGPU.cu:18 | Hash function |
| HashPosition | SpatialHashGPU.cu:24 | Position to hash |

### CUDA Runtime API usage

| API | Location | HIP Equivalent |
|-----|----------|----------------|
| cudaMallocManaged | Common.cuh:70 | hipMallocManaged |
| cudaFree | Common.cuh:77 | hipFree |
| cudaMemcpyToSymbolAsync | VtClothSolverGPU.cu:26, SpatialHashGPU.cu:173 | hipMemcpyToSymbolAsync |
| cudaMemsetAsync | VtClothSolverGPU.cu:462, SpatialHashGPU.cu:185 | hipMemsetAsync |
| cudaDeviceSynchronize | VtClothSolverGPU.hpp:107,144, SpatialHashGPU.hpp:98 | hipDeviceSynchronize |
| cudaEventCreate | Timer.hpp:106,107 | hipEventCreate |
| cudaEventRecord | Timer.hpp:111,118 | hipEventRecord |
| cudaEventSynchronize | Timer.hpp:128 | hipEventSynchronize |
| cudaEventElapsedTime | Timer.hpp:134 | hipEventElapsedTime |
| cudaEventDestroy | Timer.hpp:37,136,137 | hipEventDestroy |
| cudaEvent_t | Timer.hpp:105,228 | hipEvent_t |
| cudaMemcpy | VtBuffer.hpp:214,227 | hipMemcpy |
| cudaMemcpyDefault | VtBuffer.hpp:214,227 | hipMemcpyDefault |

### CUDA-OpenGL Interop

| API | Location | HIP Equivalent |
|-----|----------|----------------|
| cudaGraphicsGLRegisterBuffer | VtBuffer.hpp:169 | hipGraphicsGLRegisterBuffer |
| cudaGraphicsMapResources | VtBuffer.hpp:172 | hipGraphicsMapResources |
| cudaGraphicsResourceGetMappedPointer | VtBuffer.hpp:173 | hipGraphicsResourceGetMappedPointer |
| cudaGraphicsUnmapResources | VtBuffer.hpp:178 | hipGraphicsUnmapResources |
| cudaGraphicsUnregisterResource | VtBuffer.hpp:154 | hipGraphicsUnregisterResource |
| cudaGraphicsRegisterFlagsNone | VtBuffer.hpp:169 | hipGraphicsRegisterFlagsNone |
| struct cudaGraphicsResource | VtBuffer.hpp:185 | struct hipGraphicsResource |

### Library usage

| Library | Files | HIP Equivalent |
|---------|-------|----------------|
| CUB (cub::DeviceRadixSort) | SpatialHashGPU.cu | hipCUB (hipcub::DeviceRadixSort) |
| Thrust (device_ptr, transform, sort) | Common.cuh (included but not heavily used) | rocThrust |

### Constant memory

| Symbol | Location | Notes |
|--------|----------|-------|
| d_params (VtSimParams) | VtClothSolverGPU.cu:10 | Simulation parameters |
| d_params (HashParams) | SpatialHashGPU.cu:10 | Hash grid parameters |

### NVIDIA CUDA SDK helpers

| Header | Location | Notes |
|--------|----------|-------|
| helper_cuda.h | External/cuda/ | Error checking macros (checkCudaErrors) |
| helper_math.h | External/cuda/ | Vector math operators |
| helper_string.h | External/cuda/ | String utilities |

## Risk list

1. **Visual Studio -> CMake conversion**: The upstream has no CMake. A CMakeLists.txt must be created to build on Linux. This is non-trivial but the project structure is clean.

2. **CUDA-OpenGL interop**: The project uses `cudaGraphicsGLRegisterBuffer` for CUDA-GL buffer sharing. HIP has equivalent `hipGraphicsGLRegisterBuffer` APIs, but these require proper OpenGL context setup with HIP. Verify HIP-GL interop works on gfx90a.

3. **cudaMallocManaged usage**: The project allocates all simulation buffers via `cudaMallocManaged` (Common.cuh:70). On gfx90a, managed memory works but atomicMin/atomicMax have issues. This project only uses atomicAdd which is fine.

4. **NVIDIA SDK helper headers**: The External/cuda/ headers (helper_cuda.h, helper_math.h, helper_string.h) are NVIDIA SDK files. These will need:
   - helper_cuda.h: Rewrite checkCudaErrors to use HIP error handling
   - helper_math.h: Vector operators already provided by HIP
   - helper_string.h: String utilities (host-only, can be kept)

5. **checkCudaErrors macro**: Used throughout for CUDA error checking. Must be ported to HIP.

6. **__device__ __constant__ memory**: Both .cu files use `__device__ __constant__` for parameters. HIP supports this with same syntax but `hipMemcpyToSymbolAsync` replaces `cudaMemcpyToSymbolAsync`.

7. **CUB -> hipCUB**: cub::DeviceRadixSort::SortPairs must become hipcub::DeviceRadixSort::SortPairs. The API is nearly identical.

8. **Windows-focused project**: The upstream targets Windows (vcpkg + VS). Linux build will need apt/conda packages for glfw3, glad, fmt, glm, assimp, imgui.

9. **No warp intrinsics**: No `__shfl*`, `__ballot`, `__activemask`, or hardcoded warpSize=32 found. The kernels use simple atomicAdd for parallel reductions. No wave32/wave64 concerns.

## File-by-file change list

### New files to create

| File | Description |
|------|-------------|
| CMakeLists.txt | Top-level CMake build file |
| Velvet/cuda_to_hip.h | CUDA-to-HIP compat header |
| cmake/FindHIP.cmake | (optional) HIP detection module |

### Files to modify

| File | Changes |
|------|---------|
| Velvet/Common.cuh | Include cuda_to_hip.h; modify helper includes |
| Velvet/VtClothSolverGPU.cu | Include cuda_to_hip.h |
| Velvet/SpatialHashGPU.cu | Include cuda_to_hip.h; CUB -> hipCUB |
| Velvet/Timer.hpp | Include cuda_to_hip.h |
| Velvet/VtBuffer.hpp | Include cuda_to_hip.h |
| Velvet/VtClothSolverGPU.hpp | Include cuda_to_hip.h |
| Velvet/External/cuda/helper_cuda.h | Add HIP path or create HIP-compatible wrapper |

### cuda_to_hip.h content (planned)

```cpp
#pragma once

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#include <hip/hip_gl_interop.h>

// Runtime API
#define cudaMallocManaged          hipMallocManaged
#define cudaFree                   hipFree
#define cudaMemcpy                 hipMemcpy
#define cudaMemcpyDefault          hipMemcpyDefault
#define cudaMemcpyToSymbolAsync    hipMemcpyToSymbolAsync
#define cudaMemsetAsync            hipMemsetAsync
#define cudaDeviceSynchronize      hipDeviceSynchronize

// Events
#define cudaEvent_t                hipEvent_t
#define cudaEventCreate            hipEventCreate
#define cudaEventRecord            hipEventRecord
#define cudaEventSynchronize       hipEventSynchronize
#define cudaEventElapsedTime       hipEventElapsedTime
#define cudaEventDestroy           hipEventDestroy

// Error handling
#define cudaError_t                hipError_t
#define cudaSuccess                hipSuccess
#define cudaGetErrorName           hipGetErrorName
#define cudaGetLastError           hipGetLastError

// OpenGL interop
#define cudaGraphicsResource       hipGraphicsResource
#define cudaGraphicsGLRegisterBuffer       hipGraphicsGLRegisterBuffer
#define cudaGraphicsMapResources           hipGraphicsMapResources
#define cudaGraphicsResourceGetMappedPointer hipGraphicsResourceGetMappedPointer
#define cudaGraphicsUnmapResources         hipGraphicsUnmapResources
#define cudaGraphicsUnregisterResource     hipGraphicsUnregisterResource
#define cudaGraphicsRegisterFlagsNone      hipGraphicsRegisterFlagsNone

#else
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#endif
```

### CUB -> hipCUB migration (SpatialHashGPU.cu)

```cpp
#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/device/device_radix_sort.cuh>
#endif
```

## Build commands

### Linux gfx90a (ROCm)

```bash
# Install dependencies
sudo apt-get install -y libglfw3-dev libglm-dev libfmt-dev libassimp-dev

# Build glad (header-only, may need generation)
# Build imgui (typically built as part of the project)

# Configure
cmake -B build -S . \
    -DUSE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx90a \
    -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(nproc)
```

### Follower platforms

```bash
# gfx1100
cmake -B build -S . -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100

# Windows gfx1101/gfx1201: Use HIP SDK + Visual Studio or CMake with MSVC
cmake -B build -S . -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1101
```

## Test plan

### No automated test suite

The upstream has no formal test suite. Validation is visual/interactive:

1. **Build validation**: Project compiles without errors
2. **Runtime validation**: Application launches and runs without crashes
3. **Visual validation**: Cloth simulation behaves physically plausibly
   - Cloth falls under gravity
   - Cloth drapes over spheres/cubes
   - Self-collision prevents interpenetration
   - Attachment constraints hold

### Validation procedure

```bash
# Run the application
./build/Velvet

# Interactive scenes to test:
# - "Cloth / Attach" - basic draping
# - "Cloth / SDF Collision" - collision with animated sphere
# - "Cloth / Self Collision" - cloth folds on itself
# - "Cloth / Friction" - cloth sliding
# - "Cloth / High Resolution" - 200x200 grid stress test

# Expected behavior:
# - 60+ FPS on MI250X for standard scenes
# - Cloth physics look smooth and stable
# - No visual artifacts, NaN positions, or crashes
```

### Non-GPU tests

None. This is a GPU-accelerated graphics application with no CPU fallback or unit tests.

### Performance baseline (NVIDIA RTX2060, from README)

- 50k particles can approach 60 FPS limit
- High resolution (200x200 = 40k particles) scene runs smoothly

### Potential validation metrics

Since the project is visual, we could add optional validation:
1. Frame timing consistency (no frame spikes indicating GPU errors)
2. Particle position bounds check (no NaN/Inf in positions)
3. Energy conservation check (kinetic + potential should be bounded)

## Open questions

1. **imgui packaging**: The upstream uses vcpkg imgui with OpenGL3+GLFW bindings. On Linux, imgui may need to be built as a subproject or fetched via CMake FetchContent.

2. **HIP-GL interop on gfx90a**: Confirm that `hipGraphicsGLRegisterBuffer` and related APIs work correctly with the ROCm stack and a standard OpenGL context (not EGL). The project creates a standard GLFW window.

3. **glad on Linux**: glad is a loader generator, not a library. May need to generate glad sources or use a glad CMake integration.

4. **Upstream PR receptiveness**: The README mentions Windows/VS only. An upstream PR adding Linux/CMake/ROCm support may need framing as a significant feature addition, not just "AMD support."

5. **Performance tuning**: The CUDA code uses simple launch configs (BLOCK_SIZE=256). May need tuning for AMD occupancy, but correctness first.
