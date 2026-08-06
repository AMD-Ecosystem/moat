# CUDA-ScanMatcher-ICP ROCm Port Plan

## Project

- **Name**: CUDA-ScanMatcher-ICP
- **Upstream**: https://github.com/botforge/CUDA-ScanMatcher-ICP
- **Default branch**: main
- **Description**: GPU-accelerated Iterative Closest Point (ICP) point cloud scan matching with Octree optimization

## Existing AMD support

**None found.** Searched:
- Upstream README and docs: no AMD/ROCm/HIP references
- GitHub forks: ~20 forks checked, all CUDA-only personal copies
- WebSearch for "CUDA-ScanMatcher-ICP ROCm/AMD/HIP": no results
- Upstream PRs/issues: no ROCm-related

**Decision**: Proceed with a from-scratch HIP port. This is an educational/reference ICP implementation; no authoritative AMD port exists.

## Build classification

**Pure CMake project** (Strategy A). Evidence:
- `CMakeLists.txt` line 25: `find_package(CUDA 10 REQUIRED)`
- `CMakeLists.txt` line 89: `cuda_add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers})`
- No `find_package(Torch)`, no `torch.utils.cpp_extension`, no `CUDAExtension`
- No `setup.py` or `pyproject.toml`

Uses the legacy `FindCUDA` / `cuda_add_executable()` pattern (pre-CMake 3.8 native CUDA support).

## Port strategy

**Strategy A (colmap-style compat header)**: Add a single `cuda_to_hip.h` header that aliases CUDA symbols to HIP on ROCm; mark `.cu` files as `LANGUAGE HIP`; gate on `USE_HIP` option. Keep the CUDA build unchanged.

This project is a straightforward CMake+CUDA setup with no complex dependencies beyond OpenGL (GLFW/GLEW/GLM) and Thrust. The kernels are simple element-parallel ICP operations.

## CUDA surface inventory

### Include headers
| File | CUDA headers |
|------|--------------|
| `src/main.hpp` | `<cuda_runtime.h>`, `<cuda_gl_interop.h>` |
| `src/utilityCore.hpp` | `<cuda.h>` |
| `src/pointcloud.h` | `<cuda.h>`, Thrust headers |
| `src/pointcloud.cu` | `<cuda.h>` |
| `src/W.cu` | `<cuda.h>` |
| `src/scanmatch.h` | `<cuda.h>`, Thrust headers |
| `src/octree.h` | `<cuda.h>`, Thrust headers |

### CUDA runtime API symbols used
```
cudaMalloc
cudaFree
cudaMemcpy (cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost)
cudaMemset
cudaDeviceSynchronize
cudaGetLastError
cudaGetErrorString
cudaSuccess
cudaError_t
cudaDeviceProp
cudaGetDeviceCount
cudaGetDeviceProperties
cudaThreadSynchronize (deprecated, same as cudaDeviceSynchronize)
```

### CUDA-OpenGL interop
```
cudaGLSetGLDevice
cudaGLRegisterBufferObject
cudaGLMapBufferObject
cudaGLUnmapBufferObject
```

### Kernels (18 `__global__` functions)
| File | Kernel(s) |
|------|-----------|
| `src/scanmatch.cu` | `kernUpdatePositions`, `kernNNGPU_NAIVE`, `kernNNGPU_OCTREE`, `kernReshuffleGPU`, `kernComputeNorms`, `kernComputeHarray` |
| `src/pointcloud.cu` | `kernRotTrans`, `kernSetRGB`, `kernBuildTargetSinusoid`, `kernBuildSrcSinusoid`, `kernCopyPositionsToVBO`, `kernCopyRGBToVBO` |
| `src/W.cu` | Same as pointcloud.cu (appears to be a duplicate/variant file) |

### Device helpers
- `__host__ __device__ hash()` (pointcloud.cu, W.cu)
- `__host__ __device__ generateRandomVec3()` (pointcloud.cu, W.cu)
- `__device__ findLeafOctant()` (scanmatch.cu)

### Thrust usage
- `thrust::device_ptr<>`, `thrust::device_pointer_cast`
- `thrust::reduce` with custom glm::vec3/mat3 plus functor
- `thrust::default_random_engine`, `thrust::uniform_real_distribution`
- `thrust::sort`, `thrust::device_vector` (headers included but may not be actively used)

### NOT used (good -- simplifies port)
- No warp intrinsics (`__shfl*`, `__ballot`, `__activemask`)
- No hardcoded warp size 32
- No textures or surfaces
- No cuBLAS, cuFFT, cuRAND, cuSPARSE, cuDNN
- No atomics
- No shared memory
- No pinned/managed memory
- No streams/events (beyond implicit default stream)

## Risk list

1. **CUDA-OpenGL interop -> HIP-GL interop**: The project uses `cudaGLRegisterBufferObject`/`cudaGLMapBufferObject` for VBO sharing. HIP provides `hipGLRegisterBufferObject`/`hipGLMapBufferObject` (1:1 mapping). This should work but must be tested with an X11/GLX display or headless.

2. **`cudaThreadSynchronize` (deprecated)**: Used in one commented-out line (scanmatch.cu:610). Map to `hipDeviceSynchronize` (same as `cudaDeviceSynchronize`).

3. **Thrust -> rocThrust**: rocThrust is a drop-in for Thrust; same `thrust::` namespace, same headers under `/opt/rocm/include/thrust/`. The `thrust::reduce` with custom `glm::vec3`/`glm::mat3` plus functors should work unchanged.

4. **Legacy FindCUDA CMake**: The build uses `find_package(CUDA)` and `cuda_add_executable()`. We will modernize to native CMake `enable_language(HIP)` with a `USE_HIP` option, set `.cu` files to `LANGUAGE HIP`, and drop the legacy CUDA module when building for HIP.

5. **CMake compute detection**: `cmake/CUDAComputesList.cmake` detects CUDA compute capabilities. Not applicable to HIP builds; bypass on `USE_HIP=ON`.

6. **OpenGL dependencies (GLFW, GLEW, GLM)**: These are display/visualization dependencies. On headless validation, we may need to stub out VISUALIZE or use Xvfb. GLM is math-only (no display). GLFW/GLEW require an X display or EGL.

7. **W.cu appears to be a duplicate**: `src/W.cu` has nearly identical content to `src/pointcloud.cu` (same kernels, same functions). It is not listed in CMakeLists.txt sources, so it may be dead code. Confirm and exclude if unused.

## File-by-file change list

### New files
- `src/cuda_to_hip.h` -- compat header with CUDA->HIP symbol aliases

### Modified files
- `CMakeLists.txt` -- add `USE_HIP` option, `enable_language(HIP)`, set `.cu` to `LANGUAGE HIP`, link HIP runtime and rocThrust
- `src/main.hpp` -- include `cuda_to_hip.h` instead of `<cuda_runtime.h>` and `<cuda_gl_interop.h>`
- `src/utilityCore.hpp` -- include `cuda_to_hip.h`
- `src/pointcloud.h` -- include `cuda_to_hip.h`
- `src/pointcloud.cu` -- include `cuda_to_hip.h`
- `src/scanmatch.h` -- include `cuda_to_hip.h`
- `src/scanmatch.cu` -- include `cuda_to_hip.h`
- `src/octree.h` -- include `cuda_to_hip.h`
- `src/octree.cu` -- include `cuda_to_hip.h` (if it has CUDA symbols)

### Potentially exclude
- `src/W.cu` -- confirm unused (not in CMakeLists.txt sources list)

## Build commands

### Configure (gfx90a)
```bash
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm
```

### Build
```bash
cmake --build . --parallel
```

### Multi-arch fat binary test (validates warp-size correctness)
```bash
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100"
cmake --build . --parallel
llvm-objdump --offloading bin/cis565_ScanMatching  # should show gfx90a AND gfx1100
```

## Test plan

### Manual validation (this is a visualization-only project, no automated test suite)

The project has no automated tests (no gtest, no pytest, no CTest). Validation is visual:
1. Build and run the executable
2. Observe point cloud alignment in the OpenGL window
3. Check console output for NN timing (should show sensible microsecond values, not NaN or crash)

### GPU test
```bash
# Ensure HIP_VISIBLE_DEVICES is set (default device 0)
./bin/cis565_ScanMatching
# Press Escape to exit
```

Expected behavior:
- Window opens showing two point clouds (orange source, green target)
- Over iterations, the orange cloud aligns to the green cloud
- Console prints timing per NN step (e.g., "193" microseconds)

### Headless validation (if no display)
- Use Xvfb: `xvfb-run --auto-servernum ./bin/cis565_ScanMatching`
- Or modify `main.cpp` to set `#define VISUALIZE 0` for non-visual execution (runs the ICP loop without rendering)

### Non-GPU tests
None. This project is GPU-only.

## Open questions

1. **Headless GPU validation**: The project requires OpenGL for visualization. On a headless MI250 node, we need either Xvfb or to disable visualization. Confirm the headless path works.

2. **W.cu**: Confirm this file is unused. It defines the same functions as `pointcloud.cu` (ODR violation if both are compiled). Since it's not in CMakeLists.txt, it should be fine to ignore.

3. **Convergence test**: Without an automated test suite, how do we validate correctness beyond "it doesn't crash"? We could add a simple convergence check: after N iterations, the mean distance between matched points should be below a threshold. This would be a porter enhancement, not a blocker.
