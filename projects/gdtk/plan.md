# Port Plan: gdtk (linux-gfx90a)

## Project

- **Name**: gdtk (Gas Dynamics Toolkit)
- **Upstream**: https://github.com/gdtk-uq/gdtk
- **Default branch**: main

## Existing AMD support

**None found.** No AMD/ROCm/HIP references in README or docs (grep -rniE returned nothing). No AMD-related forks under gdtk-uq. Web search for "gdtk-uq ROCm AMD GPU HIP" found no existing port. The project only has CUDA support via the "chicken" CFD solver.

**Decision**: Proceed with a fresh HIP port.

## Build classification

**Makefile-based CUDA build** (not CMake, not pytorch extension).

Evidence: `/var/lib/jenkins/moat/projects/gdtk/src/src/chicken/makefile` lines 51-71 show nvcc invocations:
```makefile
chicken_for_cuda : $(CUDA_SOURCE_FILES)
    nvcc -DCUDA -std c++17 -O2 --expt-relaxed-constexpr -lineinfo -o chkn-run -D$(GAS_MODEL) \
        main_with_rev_string.cu -lz -Xcompiler -fopenmp
```

The tests under `src/chicken/test/makefile` also use direct nvcc calls.

## Port strategy

**Strategy A variant: Makefile + compat header**.

This is not a CMake project, so we adapt Strategy A to the makefile-based build. The approach:

1. Add a `cuda_to_hip.h` compat header with CUDA-to-HIP symbol aliases for the runtime API symbols used.
2. Add a `USE_HIP` build path in the makefile that invokes `hipcc` instead of `nvcc`, defines `USE_HIP`, and force-includes the compat header.
3. Map the `-DCUDA` define to a generic `USE_GPU` or keep separate `CUDA`/`HIP` defines.
4. The existing `#ifdef CUDA` blocks in main.cu and simulate.cu will need parallel `#ifdef HIP` (or combined `#if defined(CUDA) || defined(HIP)`).

## CUDA surface inventory

### Source files (5081 lines total in src/chicken/)

| File | Lines | Purpose |
|------|-------|---------|
| main.cu | 89 | Entry point, CUDA device check, dispatch |
| simulate.cu | 794 | Time-stepping, 9 `__global__` kernels |
| block.cu | 1188 | Fluid block structure, GPU memory alloc |
| bcs.cu | 808 | Boundary conditions |
| config.cu | 527 | Configuration |
| face.cu | 390 | Face calculations |
| cell.cu | 229 | Cell data structures |
| vector3.cu | 435 | Vector3 math |
| rsla.cu | 231 | Linear algebra |
| gas.cu | 122 | Gas model |
| flow.cu | 105 | Flow state |
| spline.cu | 111 | Spline interpolation |
| number.cu | 52 | Float/double typedef |

### Kernels (all in simulate.cu)

9 `__global__` kernels:
1. `estimate_allowed_dt_on_gpu` - CFL timestep estimation
2. `encodeConserved_on_gpu` - Encode flow states
3. `copy_conserved_data_on_gpu` - Copy conserved quantities
4. `setup_LSQ_arrays_on_gpu` - LSQ gradient setup
5. `calculate_fluxes_on_gpu` - Flux calculation
6. `update_stage_1_on_gpu` - RK stage 1
7. `update_stage_2_on_gpu` - RK stage 2
8. `update_stage_3_on_gpu` - RK stage 3
9. (one more for multi-stage integration)

### `__device__`/`__host__` functions

Many `__host__ __device__` dual-annotated helper functions across all files for cell operations, vector math, gas properties. These will work unchanged under HIP.

### Runtime API usage (block.cu, simulate.cu)

- `cudaMalloc` - 8 calls
- `cudaMemcpy` - 16 calls (HostToDevice and DeviceToHost)
- `cudaFree` - 5 calls
- `cudaGetLastError` - Multiple calls
- `cudaGetErrorString` - Error reporting
- `cudaGetDeviceCount` - Device query (main.cu)

### Atomics

- `atomicMin` - 1 call (simulate.cu:361) on `long long int*` allocated via `cudaMalloc` (device memory, NOT managed -- safe on gfx90a)
- `atomicAdd` - 4 calls (simulate.cu:393,431,452,476) on `int*` allocated via `cudaMalloc`

### NOT used (good for simplicity)

- No warp intrinsics (`__shfl*`, `__ballot`, `warpSize`) -- no wave64 concerns
- No textures/surfaces
- No cuBLAS/cuFFT/cuRAND/cuSPARSE
- No Thrust/CUB
- No shared memory coordination requiring sync
- No cooperative groups

### Additional CUDA code

- `src/eilmer/utils/alpha_qss_kernel_cuda_tmplt.cu` - Template for thermochemistry kernel (1 `__global__`, 2 `__device__`). Used via D language bindings, separate build path. Out of scope for initial port.
- `src/extern/cuda.d/` - D language CUDA bindings. Not directly relevant to chicken port.

## Risk list

| Risk | Severity | Mitigation |
|------|----------|------------|
| `atomicMin` on `long long int` | Low | Uses `cudaMalloc` (device memory), not managed memory. Should work on gfx90a. |
| `/proc/driver/nvidia` check | Medium | main.cu:72 checks for NVIDIA driver presence. Must be gated on `CUDA` only, add separate HIP device check. |
| `--expt-relaxed-constexpr` nvcc flag | Low | hipcc uses standard C++17 constexpr semantics by default; flag not needed. |
| `-Xcompiler -fopenmp` | Low | hipcc equivalent is just `-fopenmp` (no wrapper). |
| Build system is Makefile, not CMake | Low | Makefile modifications are straightforward. Add HIP=1 option parallel to GPU=1. |

## File-by-file change list

### New files

1. **`src/chicken/cuda_to_hip.h`** - Compat header with:
   - `#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)`
   - `#include <hip/hip_runtime.h>`
   - cudaMalloc -> hipMalloc, cudaMemcpy -> hipMemcpy, cudaFree -> hipFree
   - cudaMemcpyHostToDevice -> hipMemcpyHostToDevice, etc.
   - cudaError_t -> hipError_t, cudaSuccess -> hipSuccess
   - cudaGetLastError -> hipGetLastError, cudaGetErrorString -> hipGetErrorString
   - cudaGetDeviceCount -> hipGetDeviceCount

### Modified files

1. **`src/chicken/makefile`**
   - Add `HIP=1` option parallel to `GPU=1`
   - Add `chicken_for_hip` target using hipcc
   - Use `-std=c++17 -O2 -include cuda_to_hip.h -DHIP`
   - Remove `--expt-relaxed-constexpr` (not needed for hipcc)
   - Change `-Xcompiler -fopenmp` to `-fopenmp`
   - Target arch: `-offload-arch=gfx90a` (configurable via HIP_ARCH var)

2. **`src/chicken/main.cu`**
   - Line 29-33: Add `#elif defined(HIP)` branch for "HIP flavour of program"
   - Line 72-76: Guard `/proc/driver/nvidia` check with `#ifdef CUDA`, add HIP device check via `hipGetDeviceCount`

3. **`src/chicken/simulate.cu`** (and any file using CUDA runtime)
   - Add `#include "cuda_to_hip.h"` at top (or force-include via compile flags)
   - Change `#ifdef CUDA` to `#if defined(CUDA) || defined(HIP)` where appropriate

4. **`src/chicken/block.cu`**
   - Add include for compat header

5. **`src/chicken/test/makefile`**
   - Add HIP=1 option for test builds using hipcc

## Build commands

### Configure + Build for gfx90a (HIP)

```bash
cd /var/lib/jenkins/moat/projects/gdtk/src/src/chicken

# Build HIP version with ideal air gas model (default)
make HIP=1 HIP_ARCH=gfx90a install

# Build HIP version with reacting gas model
make HIP=1 HIP_ARCH=gfx90a AB_GAS=1 install

# Build HIP FP32 version
make HIP=1 HIP_ARCH=gfx90a FP32=1 install
```

Expected hipcc invocation:
```bash
hipcc -std=c++17 -O2 --offload-arch=gfx90a -DHIP -DIDEAL_AIR \
  -include cuda_to_hip.h -fopenmp -o chkn-run main_with_rev_string.cu -lz
```

### Build tests

```bash
cd /var/lib/jenkins/moat/projects/gdtk/src/src/chicken/test
make HIP=1
make test
```

## Test plan

### Unit tests (CPU-only, must not regress)

Located in `src/chicken/test/`:
- `gas_test` - Gas model unit tests
- `rsla_test` - Linear algebra unit tests
- `vector3_test` - Vector3 math tests
- `spline_test` - Spline interpolation tests

These compile with nvcc but do not use GPU; they test `__host__ __device__` functions on host. They should pass unchanged with hipcc.

```bash
cd src/chicken/test
make HIP=1 test
```

### Integration tests (GPU validation)

The main validation is `examples/chicken/test_chicken.py` which runs:
1. Forward-facing step example
2. Checks shock position matches expected value

GPU test procedure:
```bash
# Set up environment
export HIP_VISIBLE_DEVICES=3
export DGD=$HOME/gdtkinst
export PATH=$PATH:$DGD/bin

# Build and install
cd src/chicken
make HIP=1 HIP_ARCH=gfx90a install

# Run shock tube example
cd examples/chicken/shock-tube
chkn-prep -f sod
chkn-run --job=sod
chkn-post --job=sod

# Verify output (check for correct shock position, no NaN/inf)

# Run forward-facing step integration test
cd examples/chicken
python3 test_chicken.py
```

### Determinism check

Run the same simulation twice and diff outputs:
```bash
chkn-run --job=sod
cp sod/flow/t0001/* /tmp/run1/
chkn-run --job=sod
diff -r sod/flow/t0001/ /tmp/run1/
# Should be identical (deterministic)
```

### Non-GPU tests that must not regress

The chicken unit tests (`gas_test`, `rsla_test`, `vector3_test`, `spline_test`) must pass. These test the `__host__ __device__` math functions on the host side.

## Open questions

1. **Eilmer thermochemistry CUDA kernel**: The `alpha_qss_kernel_cuda_tmplt.cu` is used by Eilmer (the main CFD solver) via D language bindings. It has a separate build path and is out of scope for the initial chicken port. If Eilmer GPU chemistry is desired, that would be a separate effort.

2. **Windows support**: The chicken makefile is Linux-oriented. Windows HIP SDK support would require additional work (different compiler invocation, potentially CMake migration).

3. **Multi-GPU**: The current code checks device count but appears to use only one GPU. Multi-GPU support is not tested.
