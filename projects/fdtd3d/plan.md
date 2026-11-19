# fdtd3d ROCm/HIP Port Plan

## Project

- **Name**: fdtd3d
- **Upstream**: https://github.com/zer011b/fdtd3d
- **Default branch**: master
- **Description**: Open source 1D, 2D, 3D FDTD electromagnetic solver with MPI, OpenMP and CUDA support for x64, ARM, ARM64, RISC-V, PowerPC, Wasm architectures

## Existing AMD support

**Status**: None found.

- Grepped upstream docs for AMD/ROCm/HIP: no references found in README.md or Docs/
- Searched web for "fdtd3d ROCm", "fdtd3d AMD GPU", "fdtd3d HIP": no results
- Checked forks via `gh api repos/zer011b/fdtd3d/forks`: 30+ forks, none with ROCm/AMD/HIP in name or from AMD-related orgs
- Checked issues/PRs: no AMD/ROCm mentions

**Decision**: Proceed with from-scratch HIP port. No prior AMD work exists.

## Build classification

**Classification**: Pure CMake project (Strategy A)

**Evidence** (CMakeLists.txt lines 28, 93-97):
```cmake
option(CUDA_ENABLED "Cuda support enabled" OFF)
...
if ("${CUDA_ENABLED}")
  enable_language (CUDA)
  set(CUDA_SEPARABLE_COMPILATION ON)
  set(CUDA_PROPAGATE_HOST_FLAGS ON)
endif ()
```

The project uses native CMake CUDA support with `enable_language(CUDA)`. No PyTorch, no `torch.utils.cpp_extension`, no `CUDAExtension`. Uses CUDA separable compilation for device-link.

## Port strategy

**Strategy A**: Pure CMake with cuda_to_hip.h compat header

**Rationale**:
1. Clean CMake project with `enable_language(CUDA)` pattern
2. All CUDA sources are `.cu` files that can be marked `LANGUAGE HIP`
3. Uses CUDA separable compilation -- HIP supports this via `-fgpu-rdc`
4. No external CUDA libraries (cuBLAS, cuFFT, etc.) -- pure runtime API
5. Minimal CUDA API surface -- basic memory management and kernel launches

## CUDA surface inventory

### Source files (13 .cu files, ~2313 lines total)
- Source/main.cu (25 lines) -- includes main.cpp
- Source/Grid/CudaGrid.cu (1169 lines) -- core grid operations
- Source/Grid/Grid.cu (23 lines) -- includes Grid.cpp
- Source/Layout/Approximation.cu (23 lines) -- includes Approximation.cpp
- Source/Layout/YeeGridLayout.cu (23 lines) -- includes YeeGridLayout.cpp
- Source/Settings/Settings.cu (57 lines) -- device settings management
- Source/Kernels/FieldValue.cu (48 lines) -- atomicAdd for double (pre-sm_60)
- Source/Scheme/InternalScheme.cu (271 lines) -- GPU scheme implementation
- Source/Scheme/Scheme.cu (25 lines) -- includes Scheme.cpp
- Source/Scheme/SchemeHelper.cu (25 lines) -- includes SchemeHelper.cpp
- Source/Scheme/CallBack.cu (97 lines) -- callback functions
- Source/UnitTests/unit-test-cuda-grid.cu (500 lines) -- GPU grid unit tests
- Source/UnitTests/unit-test-internalscheme.cu (27 lines) -- scheme tests

### Runtime API usage (compat header required)
| CUDA API | HIP equivalent | Count |
|----------|----------------|-------|
| cudaMalloc | hipMalloc | 16 |
| cudaFree | hipFree | 10 |
| cudaMemcpy | hipMemcpy | 20 |
| cudaMemcpyHostToDevice | hipMemcpyHostToDevice | 15 |
| cudaMemcpyDeviceToHost | hipMemcpyDeviceToHost | 7 |
| cudaMemcpyToSymbol | hipMemcpyToSymbol | 1 |
| cudaMemcpyFromSymbol | hipMemcpyFromSymbol | 71 |
| cudaMemset | hipMemset | 1 |
| cudaGetLastError | hipGetLastError | 1 |
| cudaGetErrorString | hipGetErrorString | 1 |
| cudaSuccess | hipSuccess | 1 |
| cudaError_t | hipError_t | 1 |
| cudaSetDevice | hipSetDevice | 1 |
| cudaGetDevice | hipGetDevice | 1 |
| cudaGetDeviceProperties | hipGetDeviceProperties | 1 |
| cudaDeviceProp | hipDeviceProp_t | 1 |

### Kernel decorators (no changes needed -- HIP recognizes CUDA spellings)
- `__global__` -- 14 occurrences in InternalScheme.inc.h, unit-test-cuda-grid.cu
- `__device__` -- used in FieldValue.h/cu, PAssert.h
- `__host__` -- PAssert.h defines CUDA_HOST/CUDA_DEVICE macros
- `__constant__` -- Settings.cu line 28 (cudaSolverSettings pointer)

### Kernel launch syntax
- `<<<blocks, threads>>>` -- 18 occurrences (InternalScheme.inc.h, unit-test-cuda-grid.cu)
- HIP supports identical syntax

### Warp intrinsics
- **None found**. No `__shfl*`, `__ballot`, `__activemask`, or warpSize usage.
- No hardcoded warp size 32.

### Textures/surfaces
- **None found**. No texture or surface references.

### External CUDA libraries
- **None**. No cuBLAS, cuFFT, cuRAND, cuSPARSE, Thrust, or CUB usage.

### Streams/events
- **None**. Default stream only.

### atomicAdd for double
- Custom implementation in FieldValue.cu for pre-sm_60 devices
- HIP provides native atomicAdd(double*) for all AMD targets -- the CUDA fallback guarded by `__CUDA_ARCH__ < 600` will be skipped

## Risk list

### Low risk
1. **__constant__ memory**: `cudaSolverSettings` in Settings.cu uses `cudaMemcpyToSymbol`/`cudaMemcpyFromSymbol`. HIP has direct equivalents `hipMemcpyToSymbol`/`hipMemcpyFromSymbol`.

2. **CUDA separable compilation**: Project uses `CUDA_SEPARABLE_COMPILATION ON`. HIP equivalent is `-fgpu-rdc` (relocatable device code). CMake's HIP language support handles this via `set_target_properties(<tgt> PROPERTIES HIP_SEPARABLE_COMPILATION ON)` or just enabling via `CMAKE_HIP_FLAGS` with `-fgpu-rdc`.

3. **__CUDACC__ macro**: PAssert.h uses `#ifdef __CUDACC__` to define CUDA_DEVICE/CUDA_HOST. HIP defines `__HIPCC__` instead. The compat header should define `__CUDACC__` when `__HIPCC__` is set.

4. **__CUDA_ARCH__ macro**: PAssert.h and FieldValue.cu use `#ifdef __CUDA_ARCH__` for device-vs-host code paths. HIP uses `__HIP_DEVICE_COMPILE__`. The compat header should define `__CUDA_ARCH__` when in HIP device compilation.

5. **atomicCAS intrinsics**: FieldValue.cu uses `atomicCAS`, `__double_as_longlong`, `__longlong_as_double`. HIP provides these identically.

### Medium risk
None identified -- this is a straightforward port.

### Project-specific considerations
1. **CUDA_ARCH_SM_TYPE build flag**: Currently `-arch sm_XX`. For HIP, this becomes `CMAKE_HIP_ARCHITECTURES` (e.g., `gfx90a`).

2. **Test scripts hardcode sm_35**: Tests/suite/*/build.sh sets `-DCUDA_ARCH_SM_TYPE=sm_35`. Need to add HIP arch handling or ignore this for HIP builds.

## File-by-file change list

### New files
1. **Source/Helpers/cuda_to_hip.h** -- Compat header with CUDA->HIP aliases

### CMakeLists.txt modifications
1. **CMakeLists.txt** (root):
   - Add `option(USE_HIP "Build with HIP for AMD GPUs" OFF)`
   - Add HIP language enable block with arch handling
   - Set HIP separable compilation equivalent

2. **Source/CMakeLists.txt**:
   - Add `if(USE_HIP)` block to set LANGUAGE HIP on .cu files
   - fdtd3d executable links with HIP runtime

3. **Source/Scheme/CMakeLists.txt**:
   - Mark .cu sources as LANGUAGE HIP under USE_HIP
   - Set HIP_SEPARABLE_COMPILATION ON

4. **Source/UnitTests/CMakeLists.txt**:
   - Add USE_HIP handling for unit-test-cuda-grid

5. **Source/Grid/CMakeLists.txt** (if exists):
   - Mark CudaGrid.cu as LANGUAGE HIP

### Source file modifications
1. **Source/Helpers/CudaInclude.h**:
   - Include cuda_to_hip.h at top under `#ifdef CUDA_SOURCES`

2. **Source/Helpers/PAssert.h**:
   - Add `__HIPCC__` to `__CUDACC__` check for CUDA_DEVICE/CUDA_HOST macros

3. **Source/Kernels/FieldValue.cu**:
   - The `#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600` guard should work since HIP atomicAdd(double*) is native; may need `__HIP_DEVICE_COMPILE__` guard adjustment

## Build commands

### Configure (gfx90a)
```bash
mkdir build && cd build
cmake .. \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSOLVER_DIM_MODES=DIM3 \
  -DVALUE_TYPE=d \
  -DCOMPLEX_FIELD_VALUES=ON \
  -DPRINT_MESSAGE=ON
```

### Build
```bash
make -j$(nproc) fdtd3d
```

### Build unit tests
```bash
make -j$(nproc) unit-test-cuda-grid unit-test-internalscheme
```

## Test plan

### GPU tests (require real GPU)
1. **Unit tests**:
   ```bash
   ./Source/UnitTests/unit-test-cuda-grid
   ./Source/UnitTests/unit-test-internalscheme
   ```

2. **Integration test suite** (GPU mode):
   ```bash
   cd /path/to/fdtd3d
   ./Tests/run-testsuite.sh 0 1  # Sequential GPU tests (no MPI)
   ./Tests/run-testsuite.sh 1 1  # MPI + GPU tests (if MPI available)
   ```
   
   Individual tests (30 tests: t1.1-t1.2, t2.1-t2.3, t3, t4.1-t4.3, t5, t6.1-t6.13, t7.1-t7.6, t8):
   ```bash
   ./Tests/run-test.sh t1.1 0 1 $(pwd)/Tests $(pwd)
   ```

### Non-GPU tests (must not regress)
1. **CPU unit tests** (always run):
   ```bash
   ./Source/UnitTests/unit-test-settings
   ./Source/UnitTests/unit-test-clock
   ./Source/UnitTests/unit-test-grid
   ./Source/UnitTests/unit-test-dumpers-loaders
   ./Source/UnitTests/unit-test-coordinate
   ./Source/UnitTests/unit-test-layout
   ./Source/UnitTests/unit-test-approximation
   ./Source/UnitTests/unit-test-complex
   ```

2. **CPU integration tests**:
   ```bash
   ./Tests/run-testsuite.sh 0 0  # Sequential CPU tests
   ./Tests/run-testsuite.sh 1 0  # MPI CPU tests
   ```

### Test output validation
Tests compare computed electromagnetic field values against analytical solutions (exact.cpp compiled per test). Norm comparison via cmp_norm.sh validates numerical accuracy.

## Open questions

1. **CUDA_ARCH_SM_TYPE in test scripts**: The test build scripts hardcode `-DCUDA_ARCH_SM_TYPE=sm_35`. Should the HIP port:
   - Add parallel HIP_ARCH_TYPE option?
   - Ignore SM type when USE_HIP is enabled?
   - Modify test scripts to detect GPU backend?

   **Recommendation**: Ignore CUDA_ARCH_SM_TYPE when USE_HIP=ON; use CMAKE_HIP_ARCHITECTURES exclusively.

2. **MPI+HIP testing**: The test suite supports MPI+CUDA (mode 3). MPI+HIP should work identically since HIP-aware MPI follows the same patterns. Validate on MI250 with multiple GCDs.

3. **Benchmark executable**: `fdtd3dbench` is CPU-only (built without CUDA). No HIP changes needed for it.
