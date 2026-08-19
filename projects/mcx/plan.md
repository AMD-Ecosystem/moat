# Plan: mcx (Monte Carlo eXtreme)

## Project

- Name: mcx
- Upstream: https://github.com/fangq/mcx
- Default branch: main
- Description: GPU-accelerated Monte Carlo photon transport simulator for 3D heterogeneous media

## Existing AMD support

**Status: Incomplete / Under Development**

The README explicitly states: "MCX is written in C/CUDA and requires NVIDIA GPUs (support for AMD/Intel CPUs/GPUs via ROCm is still under development)."

**Related projects:**
- **MCX-CL (mcxcl)**: A separate OpenCL implementation exists at https://github.com/fangq/mcxcl that supports AMD GPUs via OpenCL. This is a full rewrite in OpenCL, not a HIP port of the CUDA code.

**Assessment:**
- No upstream HIP/ROCm branches found
- No HIP/ROCm PRs or issues found
- No community forks with ROCm/HIP work identified
- The source code contains zero HIP/ROCm references

**Decision:** Proceed with the HIP port. The OpenCL path (mcxcl) is a separate codebase entirely; a native HIP port of the CUDA mcx adds value by providing direct ROCm support with the primary CUDA codebase (better feature parity as mcx evolves, single codebase maintenance for CUDA and HIP).

**Merge policy:** The upstream actively develops both mcx (CUDA) and mcxcl (OpenCL) and states ROCm support is "under development" -- an upstream PR adding HIP support would likely be welcome.

## Build classification

**Classification: Pure CMake (Strategy A)**

**Evidence:**
- `/var/lib/jenkins/moat/projects/mcx/src/src/CMakeLists.txt` line 18: `find_package(CUDA QUIET REQUIRED)`
- Uses `cuda_add_library` and `cuda_add_executable` from legacy FindCUDA
- No `find_package(Torch)`, no `torch.utils.cpp_extension`, no `CUDAExtension`
- The pmcx Python binding uses CMake + pybind11, not torch extensions

The Makefile build uses nvcc directly. The CMake build uses the legacy FindCUDA module.

## Port strategy

**Strategy A: Pure CMake with CUDA-to-HIP compat header**

Rationale:
1. Single `.cu` file (`mcx_core.cu`) contains all GPU code (~4600 lines)
2. No warp intrinsics (`__shfl*`, `__ballot`, `warpSize`) -- ZERO warp-size fault class exposure
3. No texture objects in active use (commented out `texture<uchar, 1>` at line 211)
4. No cuBLAS/cuFFT/cuRAND/cuSPARSE dependencies
5. No Thrust/CUB usage
6. Standard CUDA runtime API (`cudaMalloc`, `cudaMemcpy`, `cudaDeviceSynchronize`, etc.)
7. Uses fp16 via `cuda_fp16.h` -- HIP has `hip/hip_fp16.h` equivalent

The port should:
1. Add `src/cuda_to_hip.h` compat header aliasing all used CUDA symbols
2. Modernize CMakeLists.txt to use CMake 3.24+ native HIP support (`enable_language(HIP)`)
3. Mark `mcx_core.cu` as `LANGUAGE HIP` under `USE_HIP`
4. Handle the fp16 include (`cuda_fp16.h` -> `hip/hip_fp16.h`)
5. Handle float3/float4 operators (HIP provides these, CUDA does not -- guard MCX's custom ones)

## CUDA surface inventory

### mcx_core.cu (~4656 lines)

**Kernels (__global__):**
- `mcx_test_rng` (line 1745) -- RNG test kernel
- `mcx_adjoint_dcoeff_kernel` (line 1845) -- adjoint simulation
- `mcx_adjoint_kernel` (line 1912) -- adjoint simulation  
- `mcx_main_loop` (line 1982) -- main Monte Carlo photon transport kernel

**Device functions (__device__):** ~50 device functions for:
- Vector math (float3 operators: +, -, *, at lines 105-164)
- atomicadd (line 217) -- custom double atomicAdd via atomicCAS, float via native atomicAdd
- RNG (xorshift128+ via included .cu file)
- Photon transport physics (scatter, reflect, refract, etc.)

**Shared memory:**
- `extern __shared__ char sharedmem[]` (line 201) -- dynamic shared memory

**Constant memory:**
- `__constant__ MCXParam gcfg` (referenced via cudaMemcpyToSymbol)
- `__constant__ Medium gproperty[...]`

**Memory operations (~153 CUDA API calls):**
- `cudaMalloc` (~30 calls)
- `cudaMemcpy` (~40 calls, H2D and D2H)
- `cudaMemcpyToSymbol` (~7 calls for constant memory)
- `cudaMemcpyFromSymbol` (1 call)
- `cudaFree` (~20 calls)
- `cudaDeviceSynchronize` (2 calls)

**Device management:**
- `cudaGetDeviceCount`
- `cudaGetDeviceProperties`
- `cudaSetDevice`

**Events:**
- `cudaEventCreate`, `cudaEventRecord`, `cudaEventQuery` -- for progress bar

**Half precision (fp16):**
- `#include "cuda_fp16.h"` (line 53)
- `__half_raw`, `__half2float` usage in property unpacking (lines 602-648)

### RNG files (mcx_rand_*.cu)
- Pure `__device__` functions
- No CUDA API calls, no warp intrinsics

### Other .cu files: None (mcx_core.cu is the only .cu)

## Risk list

1. **Float3/Float4 operators (LOW):** MCX defines custom `__device__` operators for float3 (lines 105-164). HIP's `HIP_vector_type` provides these -- need to guard MCX's versions under `#ifndef USE_HIP` or similar to avoid ambiguous overloads.

2. **FP16 path (LOW):** `cuda_fp16.h` -> `hip/hip_fp16.h`. The `__half_raw` union and `__half2float` have HIP equivalents. May need minor tweaks.

3. **Constant memory (LOW):** `cudaMemcpyToSymbol` -> `hipMemcpyToSymbol` is 1:1.

4. **Custom atomicAdd for double (LOW):** Line 217-238 implements double atomicAdd via atomicCAS. HIP natively supports `atomicAdd(double*, double)` on gfx90a (CDNA2). Can simplify or keep the CAS fallback for broader compatibility.

5. **Legacy CMake (MEDIUM):** Uses deprecated FindCUDA (`cuda_add_library`). Port to modern CMake 3.24+ with native HIP language support. This is mechanical but touches the entire build.

6. **No warp intrinsics (NONE):** Zero `__shfl*`, `__ballot`, `warpSize` usage. No wave64 vs wave32 concerns.

7. **No texture objects (NONE):** Texture code is commented out at line 211.

8. **No library dependencies (NONE):** No cuBLAS/cuFFT/cuRAND/cuSPARSE/Thrust/CUB.

## File-by-file change list

### New files:
- `src/cuda_to_hip.h` -- CUDA-to-HIP compatibility header

### Modified files:
- `src/CMakeLists.txt` -- Add USE_HIP option, enable_language(HIP), set_source_files_properties LANGUAGE HIP, handle HIP architectures
- `src/mcx_core.cu` -- Include cuda_to_hip.h, guard float3/float4 operators under `#if !defined(USE_HIP)`
- `src/Makefile` -- Add hip target (optional, for Makefile users)

### No changes needed:
- `src/mcx_core.h` -- Pure C header
- `src/mcx_utils.c/h` -- Pure C
- `src/mcx_shapes.c/h` -- Pure C
- `src/mcx_bench.c/h` -- Pure C
- `src/mcx_tictoc.c/h` -- Pure C
- `src/mcx_lang.c/h` -- Pure C
- `src/mcx_mie.cpp/h` -- Pure C++
- `src/mcx.c` -- Pure C main
- `src/mcxlab.cpp` -- MATLAB interface (host code only)
- `src/pmcx.cpp` -- Python interface (host code only)
- `src/mcx_rand_*.cu` -- Device-only code, will compile as HIP via LANGUAGE property

## Build commands

### Configure (gfx90a):
```bash
cd projects/mcx/src
mkdir build && cd build
cmake ../src \
    -DUSE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx90a \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DBUILD_MEX=OFF \
    -DBUILD_PYTHON=OFF \
    -DCMAKE_BUILD_TYPE=Release
```

### Build:
```bash
cmake --build . -j$(nproc)
```

### With Python bindings:
```bash
cmake ../src \
    -DUSE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx90a \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DBUILD_MEX=OFF \
    -DBUILD_PYTHON=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Makefile build (alternative):
```bash
cd projects/mcx/src/src
make hip HIPCC=hipcc HIP_ARCHITECTURES=gfx90a
```

## Test plan

### Primary test suite (GPU tests):
```bash
cd projects/mcx/src/test
./testmcx.sh
```

The `testmcx.sh` script runs ~40+ tests including:
- GPU detection (`mcx -L`)
- Built-in benchmarks (cube60, cube60b, cube60planar, spherebox, skinvessel, colin27)
- Various source types (isotropic, cone, fourier, pencil array, disk, gaussian, zgaussian, line, slit)
- Boundary conditions (reflection, cyclic)
- Photon detection
- Multiple output formats
- Debug modes
- Memory checks (valgrind if available)

Test validation: Each test greps for expected absorption percentages (e.g., "absorbed:.*17.[0-9]+%") which are physics-based results.

### Python tests:
```bash
cd projects/mcx/src/pmcx
pip install -e .
cd test
python -m pytest test_all.py test_utils.py -v
```

### Non-GPU tests (must not regress):
- `mcx --version` -- version string
- `mcx --help` -- help text  
- `mcx --bench` -- benchmark listing
- JSON input/output parsing tests

### Physics validation:
MCX is a Monte Carlo photon transport simulator. Key physics checks:
- Absorption percentages for known geometries (cube60: ~17%, cube60b: ~27%, etc.)
- Detected photon counts
- Energy conservation

### Determinism:
Monte Carlo methods are stochastic. For validation, use fixed seeds and compare:
- Same seed should give same results on same hardware
- Cross-platform results may differ due to RNG and floating-point ordering

## Open questions

1. **pmcx Python module:** The CMake build for pmcx (pybind11) may need additional work to properly link against HIP instead of CUDA. The setup.py passes `-DCUDA_NVCC_FLAGS=...` which won't apply to HIP. Need to add HIP-specific handling.

2. **Multi-GPU support:** MCX supports multi-GPU via OpenMP. The `cudaSetDevice` calls need to become `hipSetDevice`. Should work 1:1 but needs validation.

3. **Progress bar events:** The Windows WHQL driver workaround using cudaEvent (lines 3732-3869) should port directly to hipEvent but may need testing on Windows gfx1101.

4. **Half precision on RDNA:** FP16 support varies between CDNA (gfx90a) and RDNA (gfx1100/gfx1151). The `__half2float` path should work but packed fp16 operations may have perf differences.
