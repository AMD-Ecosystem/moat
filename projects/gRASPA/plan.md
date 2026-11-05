# gRASPA Porting Plan

## Project

- **Name**: gRASPA
- **Upstream**: https://github.com/snurr-group/gRASPA
- **Default branch**: main
- **Description**: GPU-accelerated Monte Carlo simulation software for molecular adsorption in nanoporous materials (zeolites, MOFs)

## Existing AMD Support

**Assessment**: No existing ROCm/HIP support.

- Upstream docs (`README.md`, no `docs/` directory) contain no AMD/ROCm/HIP references (`grep -rniE 'amd|rocm|hip|gfx[0-9]'` returned no matches)
- Fork scan (`gh api repos/snurr-group/gRASPA/forks`): `peterspackman/gRASPA-HIP` exists but is 0 commits ahead of upstream -- empty placeholder, no actual HIP work
- No HIP/ROCm branches or PRs in upstream
- Upstream does ship an experimental **SYCL version** (release `v-sycl`) which could run on AMD via oneAPI, but per PORTING_GUIDE: "AMD supported only via OpenCL/Vulkan/SYCL with no HIP path -> a ROCm/HIP port is still valuable"
- Web search found no third-party ROCm/HIP ports

**Decision**: Proceed with a from-scratch HIP port (Strategy A). A native HIP port provides lower overhead and better integration with the ROCm ecosystem than SYCL.

## Build Classification

**Type**: Pure CUDA project (not tied to PyTorch)

**Evidence**: The build uses the NVIDIA HPC SDK compiler (`nvc++`) with GPU offloading via a shell script (`NVC_COMPILE`):
- `src_clean/NVC_COMPILE` line 5: `CXX="/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/bin/nvc++"`
- Line 9: `NVCFLAG="-O3 -std=c++20 -target=gpu -Minline -fopenmp -cuda -stdpar=multicore"`
- Compiles `.cu` and `.cpp` files directly, no CMake or setup.py
- No `find_package(Torch)`, `torch.utils.cpp_extension`, or `CUDAExtension` present
- Uses standard CUDA runtime API, not PyTorch extensions

**ext_type**: `cmake` (will add CMakeLists.txt as the build currently has none; a Makefile-based script is not suitable for cross-platform HIP support)

## Port Strategy

**Strategy A** (pure CMake, minimal footprint):

1. Add a `src_clean/cuda_to_hip.h` compat header that aliases CUDA symbols to HIP equivalents
2. Add a `CMakeLists.txt` to replace the `nvc++`-specific shell script
3. Mark existing `.cu` files as `LANGUAGE HIP` under the `USE_HIP` option
4. Gate HIP vs CUDA paths via CMake options

The existing CUDA code is clean and straightforward:
- No warp intrinsics (`__shfl*`, `__ballot`, `warpSize`) -- all reductions use `__syncthreads()`-based tree reduction
- No textures/surfaces
- No cuBLAS/cuFFT/cuRAND/cuSPARSE dependencies
- No CUTLASS/CuTe
- Thrust includes exist but are unused (no actual `thrust::` calls)

## CUDA Surface Inventory

### Kernels (`__global__`)
| File | Kernel | Purpose |
|------|--------|---------|
| VDW_Coulomb.cu | `one_thread_GPU_test` | Debug single-thread test |
| VDW_Coulomb.cu | `Calculate_Single_Body_Energy_VDWReal` | Single body VDW + Coulomb energy |
| VDW_Coulomb.cu | `Calculate_Single_Body_Energy_VDWReal_LambdaChange` | Lambda-scaled single body energy |
| VDW_Coulomb.cu | `Energy_difference_LambdaChange` | Lambda energy difference |
| VDW_Coulomb.cu | `Calculate_Multiple_Trial_Energy_VDWReal` | Multiple trial energies (CBMC) |
| VDW_Coulomb.cu | `TotalVDWRealCoulomb` | Total system VDW + Coulomb |
| VDW_Coulomb.cu | `REZERO_VALS` | Zero array |
| Ewald_Energy_Functions.h | `Setup_Wave_Vector_Ewald` | Ewald k-space setup |
| Ewald_Energy_Functions.h | `TotalFourierEwald` (multiple variants) | Fourier-space Ewald summation |
| mc_widom.h | (inlined kernel code) | Widom insertion |

### Device Functions (`__device__`)
| File | Function | Purpose |
|------|----------|---------|
| VDW_Coulomb.cu | `VDW` | Lennard-Jones energy |
| VDW_Coulomb.cu | `CoulombReal` | Real-space Coulomb energy |
| VDW_Coulomb.cu | `VDWCoulEnergy_Total` | Combined VDW + Coulomb |
| VDW_Coulomb.cu | `determine_comp_and_Atomindex_from_thread` | Thread-to-atom mapping |
| VDW_Coulomb.cu | `determine_comp_and_Molindex_from_thread` | Thread-to-molecule mapping |
| maths.cuh | `PBC` | Periodic boundary conditions |
| maths.cuh | `VDW` (inline) | VDW energy |
| maths.cuh | `CoulombReal` (inline) | Coulomb energy |
| maths.cuh | Operator overloads for double3 | Vector math |

### Runtime API Calls
| Symbol | Count | Files |
|--------|-------|-------|
| `cudaMalloc` | ~30 | fxn_main.h, cppflow_LCLin.h, others |
| `cudaFree` | ~15 | Various |
| `cudaMemcpy` | ~80 | Everywhere |
| `cudaMemcpyAsync` | 1 | cppflow_LCLin.h |
| `cudaMemset` | ~10 | Various |
| `cudaMallocHost` | 2 | fxn_main.h |
| `cudaMallocManaged` | 1 | fxn_main.h |
| `cudaDeviceSynchronize` | ~5 | mc_swap_utilities.h |
| `cudaGetLastError` | ~5 | VDW_Coulomb.cu |
| `cudaGetErrorString` | 1 | VDW_Coulomb.cu |

### Includes
- `<cuda_runtime.h>` -- standard runtime
- `<cuda_fp16.h>` -- included but **unused** (no `__half` / `half2` / conversion calls)
- `<thrust/device_ptr.h>`, `<thrust/reduce.h>` -- included but **unused** (no `thrust::` calls)

### Kernel Launch
- Uses `<<<Nblock, Nthread, shared_mem_size>>>` syntax
- Shared memory via `extern __shared__ double sdata[]`
- Block sizes: `DEFAULTTHREAD = 128` (data_struct.h line 16)

### Thread/Block Indices
- `threadIdx.x`, `blockIdx.x`, `blockDim.x` -- standard 1D indexing

### Intrinsics
- `__syncthreads()` -- block synchronization (wave-size safe)
- `__float_as_int`, `__hiloint` -- bit manipulation
- `__forceinline__` -- inlining hint

### NOT Used (No Risk)
- No `__shfl*`, `__ballot`, `__any`, `__all`, `warpSize`, `__activemask` (no warp primitives)
- No `__syncwarp`
- No textures, surfaces, texture objects
- No cuBLAS, cuFFT, cuRAND, cuSPARSE, cuDNN
- No streams or events (aside from one `cudaMemcpyAsync` which uses default stream)
- No cooperative groups
- No CUTLASS, CuTe, or any NVIDIA-specific template libraries

## Risk List

| Risk | Severity | Mitigation |
|------|----------|------------|
| Build system overhaul | Medium | Must write CMakeLists.txt from scratch (the existing `nvc++` script is NVIDIA-specific) |
| `cudaMallocManaged` behavior | Low | Single usage in fxn_main.h; HIP supports managed memory but may have different coherence semantics on gfx90a -- test and verify |
| `cuda_fp16.h` include | Low | Unused; map to `<hip/hip_fp16.h>` in compat header for completeness |
| Thrust includes | Low | Unused; rocThrust provides same `<thrust/...>` headers under `/opt/rocm/include` |
| No existing test harness | Medium | The project uses output file energy drift checks rather than unit tests; must design validation based on physics (energy drift < 1e-3) |
| NVIDIA HPC SDK C++20 features | Low | The code uses C++20 (`-std=c++20`); clang/HIP supports C++20 |
| OpenMP (`-fopenmp`) | Low | OpenMP works with hipcc/clang; keep for host-side parallelism |

## File-by-File Change List

### New Files

| File | Purpose |
|------|---------|
| `src_clean/cuda_to_hip.h` | CUDA-to-HIP compat header with symbol aliases |
| `CMakeLists.txt` (root) | Modern CMake build replacing `NVC_COMPILE` |
| `src_clean/CMakeLists.txt` | Source directory CMake |

### Modified Files

| File | Changes |
|------|---------|
| `src_clean/VDW_Coulomb.cu` | Add `#include "cuda_to_hip.h"` at top |
| `src_clean/VDW_Coulomb.cuh` | Add `#include "cuda_to_hip.h"` at top |
| `src_clean/axpy.cu` | Add `#include "cuda_to_hip.h"` at top |
| `src_clean/maths.cuh` | Add `#include "cuda_to_hip.h"` at top |
| `src_clean/fxn_main.h` | Add `#include "cuda_to_hip.h"` at top |
| `src_clean/Ewald_Energy_Functions.h` | Add `#include "cuda_to_hip.h"` at top |
| `src_clean/data_struct.h` | Add `#include "cuda_to_hip.h"` at top (for DEFAULTTHREAD; may need device query) |
| `src_clean/mc_swap_utilities.h` | Add `#include "cuda_to_hip.h"` at top |
| `src_clean/mc_widom.h` | Add `#include "cuda_to_hip.h"` at top |
| `src_clean/cppflow_LCLin.h` | Add `#include "cuda_to_hip.h"` at top |

### cuda_to_hip.h Content

```cpp
#pragma once
#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Runtime API
#define cudaMalloc              hipMalloc
#define cudaFree                hipFree
#define cudaMemcpy              hipMemcpy
#define cudaMemcpyAsync         hipMemcpyAsync
#define cudaMemset              hipMemset
#define cudaMallocHost          hipHostMalloc
#define cudaMallocManaged       hipMallocManaged
#define cudaDeviceSynchronize   hipDeviceSynchronize
#define cudaGetLastError        hipGetLastError
#define cudaGetErrorString      hipGetErrorString

// Enums
#define cudaMemcpyDeviceToHost  hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice  hipMemcpyHostToDevice
#define cudaSuccess             hipSuccess

// Types
#define cudaError_t             hipError_t
#define cudaStream_t            hipStream_t

// Thrust (rocThrust provides same headers)
// No changes needed; <thrust/...> paths exist under /opt/rocm/include

#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif
```

### CMakeLists.txt (Root)

```cmake
cmake_minimum_required(VERSION 3.21)
project(gRASPA LANGUAGES CXX)

option(USE_HIP "Build with HIP for AMD GPUs" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

if(USE_HIP)
  enable_language(HIP)
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
    set(CMAKE_HIP_ARCHITECTURES "gfx90a")
  endif()
  add_compile_definitions(USE_HIP)
else()
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)  # Default to Ampere
  endif()
endif()

add_subdirectory(src_clean)
```

## Build Commands

### Configure (gfx90a, ROCm)
```bash
cd /var/lib/jenkins/moat/projects/gRASPA/src
mkdir -p build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
         -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
```

### Build
```bash
cmake --build . -j$(nproc)
```

### Environment
```bash
export HIP_VISIBLE_DEVICES=1
```

## Test Plan

### Validation Approach

The project uses energy drift checks rather than unit tests. The validation script (`Examples/test_examples.py`) parses output files for:
1. `ENERGY DRIFT` section with `Total Energy:` < 1e-3 (internal units)
2. `GPU DRIFT` final energy
3. `Work took N seconds` completion marker

### GPU Tests (Must Pass)

Run the example simulations and verify energy conservation:

```bash
# Navigate to build directory
cd /var/lib/jenkins/moat/projects/gRASPA/src/build

# Run a representative example
cd ../Examples/CO2-MFI
../../build/nvc_main.x  # Or whatever the executable is named
# Check output.txt for ENERGY DRIFT < 1e-3
```

**Example set for validation** (from test_examples.py):
1. `CO2-MFI` -- basic GCMC
2. `Methane-TMMC` -- Transition Matrix Monte Carlo
3. `Bae-Mixture` -- Mixture adsorption
4. `NU2000-pX-LinkerRotations` -- Framework flexibility
5. `Tail-Correction` -- Analytical tail correction

### Validation Criteria

For each example:
1. **Energy drift** < 1e-3 (same as CUDA)
2. **GPU drift** should be small (same order as CUDA)
3. **Completion** (Work took N seconds message present)
4. **Determinism**: Re-run with same seed should produce identical output

### Non-GPU Tests (Must Not Regress)

The project has no CPU-only tests. All tests exercise GPU code.

## Open Questions

1. **CMake complexity**: The existing build is a simple shell script; CMake migration may need careful handling of the DNN/LibTorch/cppflow optional paths (these are disabled by default via patch markers in source)

2. **Reference outputs**: The test validation relies on energy drift, which is inherently stochastic. Need to establish if Monte Carlo acceptance/rejection rates should also match within tolerance.

3. **pybind11 extension**: The project has optional pybind11 bindings (`pybind.h`). These are not required for core validation but may need attention for completeness.

4. **Machine learning potentials**: The DNN paths (Allegro, LCLin via cppflow/LibTorch) are optional and likely not needed for initial HIP validation. They may require additional library ports.
