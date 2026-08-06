# CubbyFlow Porting Plan

## Project

- Name: CubbyFlow
- Upstream: https://github.com/utilForever/CubbyFlow
- Default branch: main
- Description: Voxel-based fluid simulation engine for games (SPH solvers)

## Existing AMD support

**No existing AMD support found.**

- Upstream docs grep (`README*`, `docs/`): no matches for amd/rocm/hip/gfx
- Web search ("<project> ROCm", "<project> AMD GPU"): no results
- GitHub forks (gh api): none under ROCm/AMD/GPUOpen orgs or with rocm/hip in name
- Upstream branches: no rocm/hip branches
- Upstream PRs/issues: no AMD-related PRs or issues

Decision: Proceed with a fresh HIP port. No duplicate work.

## Build classification

**Pure CMake project (Strategy A)**

Evidence:
- Root `CMakeLists.txt` line 55: `find_package(CUDA)` + legacy FindCUDA
- `Sources/Core/CMakeLists.txt` line 28: `cuda_add_library(${target} ${sources} STATIC ...)`
- `setup.py` uses pybind11 via CMake (not `torch.utils.cpp_extension`)
- No `find_package(Torch)`, no `CUDAExtension`, no pytorch dependency

## Port strategy

**Strategy A: compat-header approach**

Rationale: This is a standalone CMake project with `.cu` sources that does not depend on PyTorch. The CUDA surface is minimal (memory management, kernel launches, Thrust), making a compat-header port straightforward.

The project currently uses the legacy `FindCUDA` module with `cuda_add_library()`. The port will modernize to `enable_language(HIP)` with `set_source_files_properties(... LANGUAGE HIP)` for ROCm builds while keeping the CUDA path functional.

## CUDA surface inventory

### Runtime API (minimal)

| CUDA symbol | HIP equivalent | Count |
|-------------|---------------|-------|
| `cudaMalloc` | `hipMalloc` | 1 |
| `cudaFree` | `hipFree` | 1 |
| `cudaMemcpy` | `hipMemcpy` | 1 |
| `cudaMemcpyKind` | `hipMemcpyKind` | 1 |
| `cudaMemcpyDeviceToDevice` | `hipMemcpyDeviceToDevice` | 1 |
| `cudaMemcpyDeviceToHost` | `hipMemcpyDeviceToHost` | 1 |
| `cudaMemcpyHostToDevice` | `hipMemcpyHostToDevice` | 1 |
| `cudaGetLastError` | `hipGetLastError` | macro |
| `cudaGetErrorString` | `hipGetErrorString` | macro |
| `cudaDeviceReset` | `hipDeviceReset` | macro |
| `cudaSuccess` | `hipSuccess` | macro |

### Kernel annotations

- `__global__`: ~20 kernel functions
- `__device__`: ~5 device functions
- `__host__`: used in `CUBBYFLOW_CUDA_HOST_DEVICE` macro

### Device-side intrinsics

- `blockIdx`, `threadIdx`, `blockDim`: standard launch indices (1:1 HIP)
- `make_float2/3/4`, `make_int2/3`, `make_uint2/3`: vector constructors (1:1 HIP)
- No warp intrinsics (`__shfl*`, `__ballot`, `__any`, `__all`)
- No shared memory (`__shared__`)
- No atomics
- No syncthreads

### Libraries

| CUDA | ROCm/HIP | Status |
|------|----------|--------|
| Thrust | rocThrust | Required (device_ptr, counting_iterator, for_each, inclusive_scan, transform, extrema, copy, scan) |

Thrust is used in:
- `CUDASPHSystemData3.cu`
- `CUDAParticleSystemData3.cu`
- `CUDASPHSystemData2.cu`
- `CUDAParticleSystemData2.cu`

rocThrust is a drop-in replacement with the same `thrust::` namespace and `<thrust/...>` headers on ROCm.

### Textures/Surfaces

None used.

### Streams/Events

None used.

### Pinned/Managed memory

None used (only standard `cudaMalloc`/`cudaFree`).

## Risk list

1. **Low: Thrust compatibility** -- rocThrust is a drop-in for the Thrust features used (device_ptr, for_each, scan, transform). No CUB usage, so no hipCUB needed.

2. **None: Warp size** -- No warp intrinsics or warpSize references. Kernels use simple 1D grid/block decomposition (`blockIdx.x * blockDim.x + threadIdx.x`), which is warp-size agnostic.

3. **Low: Legacy FindCUDA migration** -- The project uses `cuda_add_library()` from the legacy FindCUDA module. The port will use modern CMake `enable_language(HIP)` + `set_source_files_properties(... LANGUAGE HIP)` pattern.

4. **None: Texture/Surface** -- Not used.

5. **None: Atomics** -- Not used.

6. **Low: Float vector operator overloads** -- The project defines `operator+/-/*//` for `float2/3/4` in `CUDAUtils.hpp`. HIP provides the same types and `make_floatN` constructors, so these should work unchanged.

## File-by-file change list

### New files

1. `Includes/Core/CUDA/cuda_to_hip.h` -- Compat header with CUDA-to-HIP aliases

### Modified files

1. `CMakeLists.txt` -- Add `USE_HIP` option, `enable_language(HIP)` path, arch configuration
2. `Sources/Core/CMakeLists.txt` -- Replace `cuda_add_library()` with modern CMake library + HIP language
3. `Examples/CUDASPHSim/CMakeLists.txt` -- Replace `cuda_add_executable()` with modern CMake
4. `Tests/CUDATests/CMakeLists.txt` -- Replace `cuda_add_executable()` with modern CMake
5. `Includes/Core/Utils/Macros.hpp` -- Add HIP detection for `CUBBYFLOW_CUDA_HOST_DEVICE`, `__CUDACC__` check, and error macros
6. `Includes/Core/CUDA/CUDAUtils.hpp` -- Include compat header
7. `Includes/Core/CUDA/CUDAAlgorithms.hpp` -- Include compat header
8. `Includes/Core/CUDA/ThrustUtils.hpp` -- Include compat header (Thrust headers unchanged)
9. All other CUDA headers that include `<cuda_runtime.h>` -- Include compat header instead

## Build commands

### Configure (gfx90a)

```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DUSE_CUDA=OFF \
  -DBUILD_TESTS=ON \
  -DBUILD_EXAMPLES=ON
```

### Build

```bash
cmake --build build -j$(nproc)
```

### Configure (multi-arch validation)

```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" \
  -DUSE_CUDA=OFF \
  -DBUILD_TESTS=ON
```

## Test plan

### GPU tests (primary validation)

1. **CUDATests** -- Unit tests for CUDA array, vector, particle system, hash grid, solver components
   ```bash
   ./build/bin/CUDATests
   ```

2. **CUDASPHSim** -- Example SPH simulation
   ```bash
   ./build/bin/CUDASPHSim -s wcsph -f 100
   ```

### Non-GPU tests (must not regress)

1. **UnitTests** -- CPU-only unit tests
   ```bash
   ./build/bin/UnitTests
   ```

2. **ManualTests** -- Manual tests (if applicable)

3. **MemPerfTests / TimePerfTests** -- Performance tests (optional validation)

### Validation criteria

- All CUDATests pass on gfx90a
- UnitTests pass (non-GPU regression check)
- CUDASPHSim example runs without crash and produces sensible particle output

## Open questions

1. **Python bindings** -- The CMakeLists.txt excludes Python bindings when CUDA is enabled (`if (NOT USE_CUDA ...)`). Should the HIP build enable Python bindings? (Answer: likely no, follow the existing pattern and exclude when USE_HIP is also set.)

2. **doctest vs gtest** -- CUDATests uses doctest (via `doctest_proxy.hpp`), while UnitTests uses gtest. Both should work unchanged on ROCm.
