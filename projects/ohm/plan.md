# Plan: ohm linux-gfx90a

## Project

- **Name**: ohm
- **Upstream**: https://github.com/csiro-robotics/ohm
- **Default branch**: main
- **Description**: GPU-based probabilistic voxel occupancy map library supporting CUDA and OpenCL backends with an OpenCL-style abstraction layer

## Existing AMD support

**Finding**: No existing ROCm/HIP support. The project has two GPU backends:
- CUDA (primary, better performance per docs)
- OpenCL (secondary, supports versions 1.2/2.0/3.0)

Searches performed:
- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/` -- no matches
- WebSearch for "ohm csiro-robotics ROCm/AMD/HIP" -- no relevant results
- `gh api repos/csiro-robotics/ohm/forks` -- no AMD-related forks found
- No rocm/hip branches or PRs in upstream

**Decision**: Proceed with HIP port. While OpenCL provides theoretical AMD GPU support, a native HIP port offers better integration with the ROCm ecosystem and matches the project's preferred CUDA path.

**Merge policy**: Standard GitHub -- an upstream PR is appropriate.

## Build classification

**Classification**: Pure CMake project (Strategy A)

**Evidence**:
- `/CMakeLists.txt` lines 1-3: `cmake_minimum_required(VERSION 3.10)` / `project(ohm)` -- standalone CMake
- No `find_package(Torch)`, `torch.utils.cpp_extension`, `CUDAExtension`, or PyTorch dependencies
- CUDA enabled via `OHM_FEATURE_CUDA` option, uses modern CMake `enable_language(CUDA)` approach (lines 104 in `cmake/OhmCuda.cmake`)
- `.cu` files compiled via `cuda_add_library` (deprecated path) or direct `add_library` (modern path)

## Port strategy

**Strategy**: A (pure CMake, compat-header model with OpenCL compatibility layer)

**Rationale**: The project has a unique architecture that SIMPLIFIES the port:

1. **OpenCL-style kernel abstraction**: The project already maintains an OpenCL-to-CUDA compatibility layer (`gputil/cuda/cutil_importcl.h`). Kernels are written in `.cl` files using OpenCL semantics, then compiled as CUDA via `#include` into `.cu` wrapper files with the compat header.

2. **Existing abstraction layer**: The `gputil/` directory provides GPU backend abstraction with parallel implementations for OpenCL (`gputil/cl/`) and CUDA (`gputil/cuda/`). Adding a HIP backend follows the same pattern.

3. **Minimal CUDA surface**: CUDA APIs are isolated to `gputil/cuda/*.cpp` files. The kernels themselves use abstracted OpenCL-style primitives mapped by `cutil_importcl.h`.

**Implementation approach**:
- Add `gputil/hip/` directory mirroring `gputil/cuda/`
- Create `gputil/hip/hutil_importcl.h` as HIP version of `cutil_importcl.h`
- Add HIP-specific `gpu*.cpp` implementations
- Add new `.cu` -> `.hip` compilation path or mark `.cu` as `LANGUAGE HIP`
- Add `OHM_FEATURE_HIP` option alongside `OHM_FEATURE_CUDA`

## CUDA surface inventory

### Kernel files (`.cu`)

All in `ohmgpu/gpu/`:
- `RegionUpdate.cu` -- includes `.cl` via `cutil_importcl.h`
- `RegionUpdateNdt.cu`
- `TsdfUpdate.cu`
- `RoiRangeFill.cu`
- `TransformSamples.cu`
- `LineKeys.cu`
- `CovarianceHitNdt.cu`
- `RaysQuery.cu`

Test kernel in `tests/gputiltest/cuda/`:
- `matrix_kernel.cu` -- includes `matrix.cl` via `cutil_importcl.h`

### GPU abstraction layer (`gputil/cuda/`)

| File | CUDA APIs | HIP equivalent |
|------|-----------|----------------|
| `gpuDevice.cpp` | `cudaGetDevice`, `cudaGetDeviceCount`, `cudaGetDeviceProperties`, `cudaSetDevice`, `cudaStreamCreate` | `hipGetDevice`, `hipGetDeviceCount`, `hipGetDeviceProperties`, `hipSetDevice`, `hipStreamCreate` |
| `gpuBuffer.cpp` | `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemcpyAsync`, `cudaFreeHost` | `hipMalloc`, `hipFree`, `hipMemcpy`, `hipMemcpyAsync`, `hipHostFree` |
| `gpuKernel.cpp` | `cudaLaunchKernel`, `cudaOccupancyMaxPotentialBlockSizeVariableSMem`, `cudaSetDevice`, `cudaStreamWaitEvent`, `cudaEventRecord` | `hipLaunchKernel`, `hipOccupancyMaxPotentialBlockSizeVariableSMem`, `hipSetDevice`, `hipStreamWaitEvent`, `hipEventRecord` |
| `gpuEvent.cpp` | `cudaEventCreate`, `cudaEventDestroy`, `cudaEventSynchronize`, `cudaEventQuery`, `cudaEventElapsedTime` | `hipEventCreate`, `hipEventDestroy`, `hipEventSynchronize`, `hipEventQuery`, `hipEventElapsedTime` |
| `gpuQueue.cpp` | `cudaStreamSynchronize`, `cudaStreamDestroy` | `hipStreamSynchronize`, `hipStreamDestroy` |
| `gpuPinnedBuffer.cpp` | `cudaMemcpy` (host-to-host via pinned) | `hipMemcpy` |

### Compat headers (`gputil/cuda/`)

| Header | Purpose |
|--------|---------|
| `cutil_importcl.h` | OpenCL->CUDA mapping: `__kernel`->`__global__`, `barrier`->`__syncthreads`, `get_local_id`->`threadIdx`, etc. |
| `cutil_atomic.h` | OpenCL atomics -> CUDA atomics: `atomic_int`, `gputilAtomicAdd/Sub/Min/Max/Cas`, etc. |
| `cutil_math.h` | Math operations (52KB, comprehensive float/int vector ops) |
| `cutil_decl.h` | Kernel pointer exposure macros: `GPUTIL_CUDA_DEFINE_KERNEL` |

### Special APIs

- **No warp intrinsics**: No `__shfl*`, `__ballot`, or `warpSize` usage found
- **No textures/surfaces**: No texture or surface API usage
- **No cuBLAS/cuFFT/cuRAND/cuSPARSE/Thrust/CUB**: Pure CUDA runtime only
- **Pinned memory**: Uses `cudaMallocHost`/`cudaFreeHost` via `gputil::PinnedBuffer` -- maps to `hipHostMalloc`/`hipHostFree`
- **Streams/Events**: Standard usage, 1:1 HIP mapping

## Risk list

1. **LOW**: `cudaOccupancyMaxPotentialBlockSizeVariableSMem` -- maps to `hipOccupancyMaxPotentialBlockSizeVariableSMem` but signature/behavior should be verified

2. **LOW**: `<math_constants.h>` include in `cutil_importcl.h` -- HIP provides `<hip/hip_runtime.h>` which includes math constants, or use `<cmath>` directly

3. **LOW**: Pinned memory semantics -- `cudaMallocHost` -> `hipHostMalloc` with appropriate flags

4. **NONE**: No warp-size concerns (no warp intrinsics used)

5. **NONE**: No texture/surface concerns

6. **MEDIUM**: OpenCL 2.0 atomics semantics vs HIP -- the `cutil_atomic.h` layer maps OpenCL atomics to CUDA; need to verify these mappings work for HIP (atomicCAS, atomicExch particularly)

## File-by-file change list

### New files (Strategy A: add HIP backend alongside CUDA)

1. **`cmake/OhmHip.cmake`** -- HIP feature detection and setup (mirrors `OhmCuda.cmake`)
2. **`gputil/hip/`** directory -- HIP backend implementation:
   - `hutil_importcl.h` -- OpenCL->HIP mapping (adapt from `cutil_importcl.h`)
   - `hutil_atomic.h` -- Atomic operations (adapt from `cutil_atomic.h`)
   - `hutil_math.h` -- Math operations (adapt from `cutil_math.h`, may require minimal changes)
   - `hutil_decl.h` -- Kernel exposure macros for HIP
   - `gpuApiExceptionCode.cpp`
   - `gpuBuffer.cpp`
   - `gpuBufferDetail.h`
   - `gpuDevice.cpp`
   - `gpuDeviceDetail.h`
   - `gpuEvent.cpp`
   - `gpuEventDetail.h`
   - `gpuKernel.cpp`
   - `gpuKernel2.h`
   - `gpuKernelDetail.h`
   - `gpuMemRegion.cpp`
   - `gpuMemRegion.h`
   - `gpuPinnedBuffer.cpp`
   - `gpuPlatform2.h`
   - `gpuProgram.cpp`
   - `gpuProgramDetail.h`
   - `gpuQueue.cpp`
   - `gpuQueueDetail.h`
   - `ref.h`

3. **`ohmgpu/gpu/*.hip`** -- HIP kernel wrappers (or mark `.cu` as `LANGUAGE HIP`):
   - One wrapper per existing `.cu` file, including the `.cl` via HIP compat header

### Modified files

1. **`CMakeLists.txt`**:
   - Add `include(OhmHip)` after `include(OhmCuda)`
   - Add `OHM_FEATURE_HIP` option

2. **`gputil/CMakeLists.txt`**:
   - Add `GPUTIL_TYPE_HIP` constant
   - Add `if(OHM_FEATURE_HIP)` block creating `gputilhip` target

3. **`ohmgpu/CMakeLists.txt`**:
   - Add `if(OHM_FEATURE_HIP)` block creating `ohmhip` target

4. **`tests/gputiltest/CMakeLists.txt`**:
   - Add `if(OHM_FEATURE_HIP)` block for `gputiltesthip`

5. **`tests/ohmtestgpu/CMakeLists.txt`**:
   - Add `if(OHM_FEATURE_HIP)` block for `ohmtesthip`

## Build commands

### Configure (gfx90a)

```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DOHM_FEATURE_HIP=ON \
  -DOHM_FEATURE_CUDA=OFF \
  -DOHM_FEATURE_OPENCL=OFF \
  -DOHM_FEATURE_TEST=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a
```

### Build

```bash
cmake --build build -j$(nproc)
```

### Alternative: multi-arch build (gfx90a + gfx1100 fat binary)

```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DOHM_FEATURE_HIP=ON \
  -DOHM_FEATURE_CUDA=OFF \
  -DOHM_FEATURE_OPENCL=OFF \
  -DOHM_FEATURE_TEST=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100"
```

## Test plan

### GPU tests (primary validation)

```bash
HIP_VISIBLE_DEVICES=3 ./build/bin/gputiltesthip --gtest_output=xml:test-reports/
HIP_VISIBLE_DEVICES=3 ./build/bin/ohmtesthip --gtest_output=xml:test-reports/
```

Test binaries:
- `gputiltesthip` -- GPU utility layer tests (buffer, device, kernel invocation)
- `ohmtesthip` -- Full GPU map tests (16 test files covering copy, incidents, line keys, line query, mapper, map, ranges, ray pattern, rays query, serialisation, tests, touch time, traversal, tsdf, voxel mean, NDT)

### Non-GPU tests (must not regress)

```bash
./build/bin/ohmtest --gtest_output=xml:test-reports/
./build/bin/slamiotest --gtest_output=xml:test-reports/
```

These are CPU-only tests that validate core map functionality independent of GPU backend.

## Open questions

1. **Architecture decision**: Should HIP backend be a new `gputil/hip/` directory copying from `gputil/cuda/`, or should we unify CUDA/HIP in `gputil/cuda/` with `#ifdef USE_HIP` guards? The separate directory approach is cleaner and matches existing OpenCL separation, but has more code duplication.

2. **Kernel file approach**: Should we create `.hip` wrapper files, or mark existing `.cu` files as `LANGUAGE HIP`? The `LANGUAGE HIP` approach is minimal footprint but requires CMake guards. Given the `.cu` files are thin wrappers around `.cl` includes, marking as `LANGUAGE HIP` is likely cleanest.

3. **Existing OpenCL path for AMD**: The project supports OpenCL which theoretically works on AMD via ROCm OpenCL. Should we validate this path as an alternative, or focus purely on native HIP? HIP is preferred for performance and ecosystem alignment.
