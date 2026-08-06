# YarnBall ROCm Port Plan

## Project
- Name: YarnBall
- Upstream: https://github.com/jerry060599/YarnBall
- Default branch: main
- Description: High-performance GPU Cosserat Rods simulator for yarn physics (SIGGRAPH 2025)

## Existing AMD support

**grep docs**: No matches in README or docs for AMD/ROCm/HIP/gfx references.

**Web search + GitHub**: No ROCm/HIP-specific results. Lehong-Wang/YarnBall-Linux fork exists but only addresses Linux build compatibility for NVIDIA; no AMD GPU support.

**Upstream branches/PRs/issues**: No ROCm/HIP branches. No AMD-related PRs or issues.

**Forks**: Scanned 20+ forks; none contain AMD/ROCm work. No ROCm org forks.

**Merge policy**: Standard -- upstream accepts platform improvements directly (no linked-fork model).

**Decision**: Port from scratch. No existing AMD support to reuse or improve.

## Build classification

**Classification**: Visual Studio MSBuild project with CUDA 12.8 (NOT CMake, NOT torch-extension)

**Evidence**:
- `YarnBall.vcxproj` (line 58-59): `<Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 12.8.props" />`
- `YarnBall.vcxproj` (line 176): `<CodeGeneration>compute_86,sm_86</CodeGeneration>`
- No CMakeLists.txt anywhere in the project
- No setup.py or torch dependency

**Build model**: The project uses the Visual Studio CUDA build customization (nvcc integration via MSBuild). For Linux ROCm, a CMakeLists.txt must be created from scratch.

## Port strategy

**Strategy**: Custom CMake build (Strategy A variant)

**Rationale**: The upstream is Windows/VS-only with no CMake. For Linux ROCm, create a CMakeLists.txt that:
1. Uses `enable_language(HIP)` with `set_source_files_properties(<files> PROPERTIES LANGUAGE HIP)` to compile existing `.cu` files as HIP
2. Adds a single `cuda_to_hip.h` compat header for symbol aliasing
3. Keeps CUDA spelling in all sources; the header translates on ROCm, is a no-op on CUDA

This mirrors the compat-header approach but requires writing CMake from scratch (not modifying existing CMake).

## CUDA surface inventory

### Source files
| File | Type |
|------|------|
| `KittenEngine/YarnBall/YarnBall.cu` | Core sim (allocations, streams, memcpy, kernels) |
| `KittenEngine/YarnBall/sim/iteration.cu` | Time-stepping kernels |
| `KittenEngine/YarnBall/sim/cosserat.cu` | Cosserat rod physics kernels, shared memory |
| `KittenEngine/YarnBall/sim/collision.cu` | Collision detection kernels |
| `KittenEngine/KittenEngine/KittenGpuLBVH/lbvh.cu` | BVH construction/query (submodule), Thrust usage |
| `KittenEngine/KittenEngine/opt/svd.cuh` | Device SVD implementation |

### Kernels (`__global__`)
- `copyTempData`, `zeroVels` (YarnBall.cu)
- `initItr`, `endItr` (iteration.cu)
- `cosseratItr<LIMIT>`, `quaternionLambdaItr` (cosserat.cu)
- `buildAABBs`, `buildCollisionList`, `recomputeStepLimitKernel` (collision.cu)
- `mortonKernel`, `lbvhBuildInternalKernel`, `mergeUpKernel`, `lbvhQueryKernel` (lbvh.cu)

### Runtime API usage
| API | Files | HIP equivalent |
|-----|-------|----------------|
| `cudaMalloc` | YarnBall.cu | `hipMalloc` |
| `cudaFree` | YarnBall.cu | `hipFree` |
| `cudaMemcpy*` | YarnBall.cu | `hipMemcpy*` |
| `cudaMemset*` | YarnBall.cu, collision.cu | `hipMemset*` |
| `cudaStreamCreate/Destroy/Synchronize` | YarnBall.cu | `hipStream*` |
| `cudaDeviceSynchronize` | YarnBall.cu, lbvh.cu | `hipDeviceSynchronize` |
| `cudaGraphExecDestroy` | YarnBall.cu | `hipGraphExecDestroy` |
| `cudaGetLastError` | YarnBall.cu | `hipGetLastError` |
| `cudaGetErrorString` | Common.h | `hipGetErrorString` |
| `cudaSuccess` | Common.h | `hipSuccess` |
| `cudaError_t` | Common.h | `hipError_t` |
| `cudaGraphicsGLRegisterBuffer` | ComputeBuffer.h | `hipGraphicsGLRegisterBuffer` |
| `cudaGraphicsMapResources` | ComputeBuffer.h | `hipGraphicsMapResources` |
| `cudaGraphicsResourceGetMappedPointer` | ComputeBuffer.h | `hipGraphicsResourceGetMappedPointer` |

### Library usage
| Library | Files | HIP equivalent |
|---------|-------|----------------|
| Thrust (`device_vector`, `sort_by_key`, `reduce`, `fill`, `swap`) | lbvh.cu | rocThrust (same API, same headers under `/opt/rocm/include`) |

### Device intrinsics
| Intrinsic | Files | HIP equivalent |
|-----------|-------|----------------|
| `__clzll` | lbvh.cu | `__clzll` (same) |
| `__threadfence` | lbvh.cu | `__threadfence` (same) |
| `atomicAdd`, `atomicOr` | lbvh.cu, collision.cu | Same API |
| `__syncthreads` | cosserat.cu, lbvh.cu | Same API |
| `__fadd_rn`, `__fsub_rn`, `__frsqrt_rn` | svd.cuh | Same API (HIP provides these) |

### Warp intrinsics
**None found.** No `__shfl*`, `__ballot`, `__activemask`, `__syncwarp`, `warpSize`, or hardcoded 32 in kernel logic.

### Textures/surfaces
**None.** The project uses raw memory (device pointers), no texture objects or surface load/store.

### Streams/events
- Uses `cudaStream_t` for async operations
- Uses `cudaGraphExec_t` for CUDA graphs

### Shared memory
- `cosserat.cu` line 149: `__shared__ float sharedData[18 * VERTEX_PER_BLOCK]` where `VERTEX_PER_BLOCK = 64`
- `lbvh.cu` lines 210-212: `__shared__ ivec2 sharedRes[MAX_RES_PER_BLOCK]`, `__shared__ int sharedCounter`, `__shared__ int sharedGlobalIdx`

## Risk list

1. **CUDA Graphs** (`cudaGraphExec_t`): Used in YarnBall.h/cu for step execution optimization. HIP supports `hipGraph*` API with mostly 1:1 mapping, but graph capture semantics may differ. Verify graph capture/exec works or fall back to direct kernel launches.

2. **OpenGL-CUDA interop** (`cudaGraphicsGLRegisterBuffer`): Used in ComputeBuffer.h for GPU-GL buffer sharing. HIP provides `hipGraphicsGLRegisterBuffer` but requires ROCm OpenGL interop support. May need runtime conditional or may be non-essential for headless simulation.

3. **CUDA 12.8 features**: Project targets sm_86 (Ampere). No CUDA 12-specific features observed in kernels, but check for any compute_86-specific codegen.

4. **Thrust usage in lbvh.cu**: rocThrust is a drop-in but needs `#include <thrust/...>` paths verified. The submodule (KittenGpuLBVH) uses Thrust extensively.

5. **CUDA math intrinsics in svd.cuh**: `__fadd_rn`, `__fsub_rn`, `__frsqrt_rn` etc. are supported in HIP but verify correct semantics (round-to-nearest).

6. **Submodule dependency**: KittenGpuLBVH is a git submodule that must be ported alongside the main repo.

7. **No CMake exists**: Must create a complete CMakeLists.txt from scratch, handling:
   - All .cpp sources (KittenEngine, YarnBall, optimization libs)
   - All .cu sources (must work as HIP on ROCm, CUDA on NVIDIA)
   - External dependencies: GLM, Eigen3, assimp, glad, glfw, imgui, jsoncpp, CLI11, freetype

8. **Windows-only upstream**: Build tested only on Windows. Linux compatibility needs verification for non-GPU code paths (filesystem, OpenGL context).

## File-by-file change list

### New files
1. `CMakeLists.txt` (root): Main build file with HIP/CUDA dual support
2. `src/cuda_to_hip.h`: Compat header (CUDA-to-HIP runtime aliases)

### Modified files
1. `KittenEngine/KittenEngine/includes/modules/Common.h`:
   - Add HIP detection alongside CUDA
   - Include `cuda_to_hip.h` for runtime aliases
   - Adjust `checkCudaErrors` macro for HIP

2. `KittenEngine/YarnBall/YarnBall.cu`:
   - Add `#include "cuda_to_hip.h"` at top
   - No other source changes needed (symbols aliased by header)

3. `KittenEngine/YarnBall/YarnBall.h`:
   - Types `cudaStream_t`, `cudaGraphExec_t` need aliasing in compat header

4. `KittenEngine/KittenEngine/includes/modules/ComputeBuffer.h`:
   - GL-CUDA interop calls need HIP equivalents or conditional compile

5. `KittenEngine/YarnBall/sim/*.cu`:
   - Add compat header include

6. `KittenEngine/KittenEngine/KittenGpuLBVH/lbvh.cu` and `lbvh.cuh`:
   - Add compat header include
   - Thrust includes work unchanged

7. `KittenEngine/KittenEngine/opt/svd.cuh`:
   - Add `#include <cuda.h>` -> include compat header instead

### Dependencies to install (Linux)
- GLM (libglm-dev or vcpkg)
- Eigen3 (libeigen3-dev or vcpkg)
- assimp (libassimp-dev)
- GLFW (libglfw3-dev)
- glad (bundled or vcpkg)
- imgui (bundled or vcpkg)
- jsoncpp (libjsoncpp-dev)
- CLI11 (header-only, vcpkg)
- freetype (libfreetype-dev)
- ROCm 7.2+ with HIP, rocThrust

## Build commands

### Configure (ROCm/HIP)
```bash
cmake -B build -S . \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_BUILD_TYPE=Release
```

### Build
```bash
cmake --build build -j$(nproc)
```

### Configure (CUDA, for verification)
```bash
cmake -B build-cuda -S . \
  -DUSE_HIP=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_BUILD_TYPE=Release
```

## Test plan

### GPU validation tests
The project has no formal test suite. Validation uses built-in simulation scenarios:

1. **Twist animation test** (configs/cable_work_pattern.json):
   ```bash
   ./build/Gui configs/cable_work_pattern.json --twist -s --headless -n 100 --exit
   ```
   - Success: Runs 100 frames without crash, exits cleanly
   - Verify: Check simulation time advances, no CUDA/HIP errors

2. **Letter simulation test** (configs/letterS.json):
   ```bash
   ./build/Gui configs/letterS.json -s --headless -n 100 --exit
   ```
   - Success: Completes without error

3. **Export test** (correctness check):
   ```bash
   ./build/Gui configs/cable_work_pattern.json --twist -s --headless -n 50 -o /tmp/frame_ --exit
   ls /tmp/frame_*.obj
   ```
   - Verify: OBJ files are non-empty and contain valid mesh data

4. **LBVH self-test** (built-in BVH validation):
   - The lbvh.cu contains `testLBVH()` function that runs 100k object collision detection
   - Compares GPU results against CPU brute force
   - Add a simple test binary that calls `Kitten::testLBVH()`

### Non-GPU regression
- None identified (all functionality is GPU-based)

### Cross-platform note
- GUI mode requires OpenGL; headless mode (`--headless`) bypasses it
- Use headless mode for GPU validation on compute nodes

## Open questions

1. **OpenGL-GPU interop priority**: Is ComputeBuffer GL interop required for validation, or can we test headless-only? The headless path bypasses GL entirely.

2. **CUDA Graphs usage**: The `stepGraph` member suggests CUDA graph optimization. Is this critical path or optional optimization? May need conditional enablement.

3. **Submodule handling**: KittenGpuLBVH is a separate repo. Should the port modify it in-place (submodule commit) or fork it separately?

4. **Upstream PR scope**: Should the CMakeLists.txt be proposed upstream (adds Linux support), or is this a MOAT-only addition?
