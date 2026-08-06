# GPU_IPC ROCm Port Plan

## Project

- **Name**: GPU_IPC
- **Upstream**: https://github.com/KemengHuang/GPU_IPC
- **Default branch**: main
- **Description**: GPU-optimized Incremental Potential Contact (IPC) simulation framework for physics-based contact mechanics. Implements a Gauss-Newton solver with a multilevel additive Schwarz (MAS) preconditioner for PCG convergence.

## Existing AMD support

**None found.** Searched:
- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/` -- no matches
- WebSearch for "GPU_IPC ROCm", "GPU_IPC AMD", "Incremental Potential Contact HIP" -- no relevant results
- `gh api repos/KemengHuang/GPU_IPC/forks` -- 18 forks, none in ROCm/AMD/GPUOpen orgs, none with rocm/hip in name
- No upstream rocm/hip branches or related PRs

**Decision**: Proceed with a from-scratch HIP port. No existing AMD effort to adopt or improve.

## Build classification

**Pure CMake project (Strategy A)**

Evidence (CMakeLists.txt):
- Line 16: `project(gpu_ipc LANGUAGES CXX CUDA)`
- Line 37: `find_package(CUDAToolkit REQUIRED)` -- uses CUDA Toolkit components
- Lines 109-115: Links `CUDA::cudart`, `CUDA::cublas`, `CUDA::cusolver`, `CUDA::cusparse`, `CUDA::cufft`
- No `find_package(Torch)`, no `torch.utils.cpp_extension`, no setup.py

Build type: Standalone CMake with direct CUDA language support and CUDA library linking.

## Port strategy

**Strategy A: pure CMake, compat-header approach**

Rationale:
- Pure CMake project with no PyTorch dependency
- All CUDA code in `.cu`/`.cuh` files under GPU_IPC/
- Can add a single cuda_to_hip.h compat header and mark .cu files as LANGUAGE HIP
- The CUDA library usage (cuBLAS, cuSOLVER, cuSPARSE, cuFFT) is linked but appears to be DEAD CODE (grep shows cusolverDn calls are all commented out in mlbvh.cu lines 1613-1721); only cudart runtime API is actively used

## CUDA surface inventory

### Files requiring port

| File | Lines | Content |
|------|-------|---------|
| GIPC.cu | 10324 | Main IPC solver, collision detection, BVH traversal |
| GIPC_PDerivative.cuh | 5500+ | Barrier Hessian derivative computations |
| MASPreconditioner.cu | 1800 | Multilevel Additive Schwarz preconditioner with warp-level ops |
| PCG_SOLVER.cu | 1580 | Preconditioned Conjugate Gradient solver with warp reductions |
| femEnergy.cu | 1700 | FEM energy computation kernels |
| mlbvh.cu | 1600 | Linear BVH construction with Morton codes, Thrust sorting |
| gpu_eigen_libs.cu | 1050 | Custom GPU linear algebra utilities |
| ACCD.cu | 600 | Additive Continuous Collision Detection |
| device_fem_data.cu | 100 | Device memory allocation/deallocation |
| gl_main.cpp | 950 | Main entry point, OpenGL visualization, CUDA device init |

### Runtime API usage

| API | ROCm Equivalent | Location |
|-----|-----------------|----------|
| cudaMalloc | hipMalloc | device_fem_data.cu, PCG_SOLVER.cu, GIPC.cu, MASPreconditioner.cu |
| cudaFree | hipFree | device_fem_data.cu, mlbvh.cu |
| cudaMemcpy | hipMemcpy | PCG_SOLVER.cu, load_mesh.cpp |
| cudaMemset | hipMemset | device_fem_data.cu, PCG_SOLVER.cu, mlbvh.cu, MASPreconditioner.cu |
| cudaSetDevice | hipSetDevice | gl_main.cpp |
| cudaEventCreate | hipEventCreate | mlbvh.cu (performance timing) |
| CUDA_SAFE_CALL macro | Convert to HIP | cuda_tools.h (error checking) |

### Warp intrinsics (via gipc namespace)

| Intrinsic | Definition | Notes |
|-----------|------------|-------|
| gipc::WARP_SHFL_DOWN | __shfl_down_sync | PCG_SOLVER.cu, mlbvh.cu -- used with hardcoded width 32 |
| gipc::WARP_SHFL | __shfl_sync | MASPreconditioner.cu |
| gipc::WARP_BALLOT | __ballot_sync | MASPreconditioner.cu -- returns uint32, uses 0xffffffff mask |
| gipc::SYNC_THREADS | __syncthreads | Throughout |
| gipc::THREAD_FENCE | __threadfence | PCG_SOLVER.cu, mlbvh.cu |
| __popc | __popc | MASPreconditioner.cu -- operates on 32-bit masks |
| __ffs | __ffs | MASPreconditioner.cu |
| __clz | __clz | MASPreconditioner.cu |
| __clzll | __clzll | mlbvh.cu |

### BANKSIZE hardcoding (CRITICAL)

MASPreconditioner.cu line 40: `#define BANKSIZE 32`

This is a **logical warp size constant** used throughout the MAS preconditioner for:
- Grouping vertices into 32-element clusters for block-diagonal Hessian factorization
- Lane masks (`_LanemaskLt` returns `(1U << laneIdx) - 1`)
- Matrix block dimensions (96x96 = 32 vertices x 3 DOF)
- Warp-level prefix sums with `__popc(mask & _LanemaskLt(laneId))`

**Analysis**: BANKSIZE=32 is an ALGORITHMIC choice (cluster size for MAS preconditioner), not a warp-size assumption. The algorithm groups 32 vertices per cluster for efficient 96x96 block Cholesky. The warp intrinsics operate within this 32-thread logical group. On wave64, these 32-thread groups still work correctly because:
1. The ballot masks are 32-bit `unsigned int` and only track connectivity within the 32-vertex cluster
2. Width-32 shuffles operate on the first 32 lanes of the wavefront (width parameter implicit = 32)
3. The `__popc` counts are on 32-bit masks

The PCG_SOLVER.cu reductions (lines 39-70) use explicit width-32 shuffles: `gipc::WARP_SHFL_DOWN(temp, i)` with `i` iterating 1,2,4,8,16 -- this is a standard 32-lane reduction pattern that works on both wave32 and wave64 (operates on the first 32 lanes).

**Risk**: LOW. The width-32 operations are used for logical 32-element groups, not physical warp size. On wave64, the operations run with width=32 (default) which is safe. No 64-bit ballot masks, no lane-count assumptions beyond width-32 operations.

### Thrust usage

| Call | Location | ROCm Equivalent |
|------|----------|-----------------|
| thrust::sequence | mlbvh.cu, GIPC.cu | rocThrust (drop-in) |
| thrust::sort_by_key | mlbvh.cu, GIPC.cu | rocThrust (drop-in) |
| thrust::exclusive_scan | GIPC.cu, MASPreconditioner.cu | rocThrust (drop-in) |
| thrust::device_ptr | Throughout | rocThrust (drop-in) |

rocThrust is a true drop-in: same `thrust::` namespace, same `<thrust/...>` headers.

### CUDA library linking

CMakeLists.txt links:
- CUDA::cudart -> hiprt (via hipcc)
- CUDA::cublas -> NOT ACTUALLY USED (grep shows no cublasCreate/cublas API calls)
- CUDA::cusolver -> NOT ACTUALLY USED (all cusolverDn calls commented out in mlbvh.cu)
- CUDA::cusparse -> NOT ACTUALLY USED (no cusparse API calls)
- CUDA::cufft -> NOT ACTUALLY USED (no cufft API calls)

**These library links are dead code.** Only the CUDA runtime is actually used.

### Cooperative groups

MASPreconditioner.cu includes `<cooperative_groups.h>` and uses `using namespace cooperative_groups`, but grep shows no actual `cg::` calls -- the include appears unused.

### Vector types

Heavy use of CUDA vector types: `double3`, `uint3`, `uint4`, `float3`, `int2`. These have direct HIP equivalents with same names in `<hip/hip_runtime.h>`.

## Risk list

1. **BANKSIZE=32 warp operations** (LOW): The 32-element cluster size is an algorithmic choice for the MAS preconditioner. Width-32 warp shuffles and 32-bit ballot masks work correctly on wave64 (operate on logical 32-lane groups). No changes needed.

2. **No automated test suite** (MEDIUM): The project is a research demo with visual output (OpenGL). Validation requires running simulation scenes and checking for:
   - No GPU errors/crashes
   - Simulation stability (no NaN/explosion)
   - Visual correctness of collision resolution
   - PCG convergence metrics (residual prints)

3. **Thrust to rocThrust** (LOW): rocThrust is a drop-in replacement with same namespace and headers. No source changes needed.

4. **Dead CUDA library links** (LOW): cuBLAS/cuSOLVER/cuSPARSE/cuFFT are linked but not called. Remove from HIP build.

5. **OpenGL/GLUT dependency**: The project renders via OpenGL. On a headless gfx90a node, may need to run with `DISPLAY=:0` or modify to support headless output. Consider adding a `--headless` mode that runs simulation without rendering and outputs mesh files.

6. **Separable compilation**: CMakeLists.txt line 55 enables `CUDA_SEPARABLE_COMPILATION ON`. Need equivalent HIP handling with `-fgpu-rdc` if cross-TU device calls exist, or disable if not needed.

## File-by-file change list

### New files

1. **GPU_IPC/cuda_to_hip.h** - CUDA-to-HIP compatibility header
   - Include `<hip/hip_runtime.h>` under USE_HIP
   - Alias cudaMalloc->hipMalloc, cudaFree->hipFree, cudaMemcpy->hipMemcpy, cudaMemset->hipMemset
   - Alias cudaSetDevice->hipSetDevice, cudaError_t->hipError_t, cudaSuccess->hipSuccess
   - Alias cudaGetErrorString->hipGetErrorString
   - Alias cudaEventCreate->hipEventCreate, cudaEventRecord->hipEventRecord, cudaEventElapsedTime->hipEventElapsedTime
   - Alias cooperative_groups header path

### Modified files

1. **CMakeLists.txt**
   - Add `option(USE_HIP "Build with HIP for AMD GPUs" OFF)`
   - Under USE_HIP: `enable_language(HIP)`, mark .cu files `LANGUAGE HIP`
   - Set HIP_ARCHITECTURES from CMAKE_HIP_ARCHITECTURES with gfx90a default
   - Remove unused CUDA library links (cublas, cusolver, cusparse, cufft) for HIP build
   - Add hip_runtime link for HIP build
   - Handle separable compilation for HIP if needed

2. **GPU_IPC/cuda_tools.h**
   - Add `#include "cuda_to_hip.h"` at top (replaces direct cuda_runtime.h include)
   - CUDA_SAFE_CALL macro works unchanged (uses cudaError_t/cudaSuccess aliases)

3. **GPU_IPC/device_utils.h**
   - Add `#include "cuda_to_hip.h"` (for cuda_runtime.h -> hip_runtime.h)

4. **GPU_IPC/gl_main.cpp**
   - Add `#include "cuda_to_hip.h"` before cuda_runtime.h
   - Error message update: "CUDA-capable" -> "CUDA/HIP-capable"

5. **GPU_IPC/*.cuh headers** (PCG_SOLVER.cuh, mlbvh.cuh, femEnergy.cuh, ACCD.cuh, GIPC_PDerivative.cuh, gpu_eigen_libs.cuh, FrictionUtils.cuh, load_mesh.h)
   - Replace `#include <cuda_runtime.h>` with `#include "cuda_to_hip.h"`

6. **GPU_IPC/MASPreconditioner.cu**
   - Replace `#include <cooperative_groups.h>` with HIP path: `#include <hip/hip_cooperative_groups.h>` under USE_HIP guard

## Build commands

### Configuration (gfx90a)

```bash
mkdir build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_BUILD_TYPE=Release
```

### Build

```bash
cmake --build . -j$(nproc)
```

### Dependencies (Ubuntu)

```bash
# ROCm (already installed on gfx90a host)
# Graphics libraries
sudo apt install libglew-dev freeglut3-dev libeigen3-dev
```

## Test plan

### GPU tests

1. **Smoke test**: Run the gipc executable with default scene
   ```bash
   ./gipc
   ```
   - Expect: OpenGL window opens, simulation runs without crash
   - Check: No "hipError" or crash messages, PCG converges (residual decreases)

2. **Simulation stability**: Run for 100+ timesteps
   - Check: No NaN values, simulation doesn't explode
   - Check: Contact resolution works (objects don't interpenetrate)

3. **Output verification**: Enable mesh output and compare
   - The project can output .obj mesh files per frame
   - Compare mesh vertex positions frame-by-frame between CUDA and HIP builds for identical simulation

### Non-GPU tests

None. This is a pure GPU simulation project with no CPU-only test suite.

### Validation criteria

- Simulation runs without runtime errors
- PCG solver converges (printed residual metrics)
- Visual contact resolution is correct (no interpenetration)
- Mesh output is deterministic between runs
- Cross-platform: identical results between CUDA and HIP for same input scene

## Open questions

1. **Headless validation**: The gfx90a node may not have a display. Need to either:
   - Use VirtualGL/Xvfb for offscreen rendering
   - Modify gl_main.cpp to support headless mode (run simulation, skip rendering, output meshes)
   - Run with `DISPLAY=:0` if X server available

2. **Separable compilation**: Need to verify if cross-TU device calls require `-fgpu-rdc` for HIP. Initial analysis suggests the kernels are defined and called within single TUs, so separable compilation may be disable-able.

3. **Performance regression testing**: The paper claims high performance; should validate that HIP port maintains reasonable performance vs CUDA (within expected HIP vs CUDA variance).
