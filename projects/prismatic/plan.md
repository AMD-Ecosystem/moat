# Prismatic Port Plan (linux-gfx90a)

## Project

- **Name**: prismatic
- **Upstream**: https://github.com/prism-em/prismatic
- **Default branch**: main
- **Description**: C++/CUDA package for parallelized simulation of image formation in Scanning Transmission Electron Microscopy (STEM) using the PRISM and multislice algorithms

## Existing AMD support

**None found.** Grepping the upstream repo docs shows no references to AMD/ROCm/HIP. Web searches for "prismatic ROCm", "prismatic AMD GPU", "prism-em HIP" return no results indicating AMD support. The GitHub API shows no forks with ROCm/HIP/AMD in the name, no upstream ROCm/HIP branches, and no related issues/PRs.

**Project status**: The README notes "As of January 2026, Prismatic is no longer actively maintained." This means an upstream PR is unlikely to be merged but is still worth creating for community benefit. The port should be self-contained and complete.

**Decision**: Proceed with a from-scratch HIP port.

## Build classification

**Pure CMake project using LEGACY FindCUDA** (Strategy A, adapted for FindCUDA shim)

Evidence (CMakeLists.txt):
- Line 143: `find_package(CUDA REQUIRED)` -- uses legacy FindCUDA, not modern CMake CUDA language
- Lines 178, 232, 287, 321: `cuda_add_executable`, `cuda_add_library` -- legacy FindCUDA macros
- Lines 182, 228, 243, 288, 325: `cuda_add_cufft_to_target` -- legacy FindCUDA cuFFT helper
- No `find_package(Torch)` or `torch.utils.cpp_extension` -- not a PyTorch extension
- setup.py drives CMake for pyprismatic but is not a CUDAExtension

**ext_type**: cmake

## Port strategy

**Strategy A (pure CMake) with FindCUDA shim**

This project uses the legacy FindCUDA module (`cuda_add_executable`, etc.) rather than modern CMake's `enable_language(CUDA)`. Per the PORTING_GUIDE cupoch lesson, bridge this with a `cuda_add_library`/`cuda_add_executable` SHIM macro that forwards to `add_library`/`add_executable` + marks sources `LANGUAGE HIP`. This keeps every existing `cuda_add_*` call untouched and the NVIDIA path byte-identical.

Approach:
1. Add `include/cuda_to_hip.h` -- the CUDA->HIP symbol compat header
2. Add `cmake/FindHIP.cmake` -- a USE_HIP option that defines shim macros for cuda_add_*
3. In CMakeLists.txt, under `if(USE_HIP)`, include the HIP shim before any cuda_add_* call
4. Mark BOTH .cu AND .cpp sources LANGUAGE HIP (the legacy cuda_add_* mixed them, and some .cpp include headers that only clang parses)
5. Link `hip::host` + `hipfft`; map CUDA_LIBRARIES to hip::host

## CUDA surface inventory

### Files (5 .cu, ~4851 lines total)

| File | Lines | Content |
|------|-------|---------|
| src/utility.cu | 516 | Element-wise kernels (multiply, divide, abs_squared, setAll, initializePsi, integrateDetector, DPC_numerator/denominator_reduce), atomicAdd |
| src/fileIO.cu | 327 | File I/O with GPU memory (copySMatrixCoeffs_d2H, copyPsiToStack, etc.) |
| src/PRISM02_calcSMatrix.cu | 1124 | S-matrix calculation, cuFFT plan creation/execution, memory alloc/copy, stream management |
| src/PRISM03_calcOutput.cu | 1960 | Output calculation, warp-synchronous reductions (warpReduce_cx), scaleReduceS kernels |
| src/Multislice_calcOutput.cu | 924 | Multislice algorithm GPU kernels, cuFFT |

### Kernels (58 __global__/__device__ functions)

Dominated by simple element-wise operations and reductions. Key kernel categories:
- Element-wise complex arithmetic (multiply_inplace, multiply_cx, divide_inplace, abs_squared)
- Array initialization (setAll, initializePsi, initializePsi_oneNonzero)
- Reductions with warp-synchronous tail (warpReduce_cx, scaleReduceS)
- Index manipulation (shiftIndices, zeroIndices, resetIndices)
- Detector integration (integrateDetector, DPC_numerator_reduce, DPC_denominator_reduce)
- Phase coefficient computation (computePhaseCoeffs)
- S-matrix copy operations (copySMatrixCoeffs_d2H, copyPsiToStack)

### CUDA Runtime API usage

| Symbol | Count | HIP equivalent |
|--------|-------|----------------|
| cudaMalloc | ~30 | hipMalloc |
| cudaMallocHost | ~15 | hipHostMalloc |
| cudaFree | ~20 | hipFree |
| cudaFreeHost | ~10 | hipHostFree |
| cudaMemcpyAsync | ~30 | hipMemcpyAsync |
| cudaMemcpy2DAsync | ~5 | hipMemcpy2DAsync |
| cudaMemset | ~5 | hipMemset |
| cudaStreamCreate | ~5 | hipStreamCreate |
| cudaStreamDestroy | ~5 | hipStreamDestroy |
| cudaStreamSynchronize | ~10 | hipStreamSynchronize |
| cudaDeviceSynchronize | ~5 | hipDeviceSynchronize |
| cudaSetDevice | ~5 | hipSetDevice |
| cudaMemGetInfo | ~2 | hipMemGetInfo |
| cudaDeviceReset | ~2 | hipDeviceReset |
| cudaStream_t | many | hipStream_t |
| cudaError_t | ~5 | hipError_t |
| cudaSuccess | ~5 | hipSuccess |
| cudaGetErrorString | ~2 | hipGetErrorString |

### cuFFT usage

| Symbol | HIP equivalent |
|--------|----------------|
| cufftHandle | hipfftHandle |
| cufftPlanMany | hipfftPlanMany |
| cufftPlan2d | hipfftPlan2d |
| cufftSetStream | hipfftSetStream |
| cufftExecC2C | hipfftExecC2C |
| cufftExecZ2Z | hipfftExecZ2Z |
| cufftDestroy | hipfftDestroy |
| CUFFT_C2C | HIPFFT_C2C |
| CUFFT_Z2Z | HIPFFT_Z2Z |
| CUFFT_FORWARD | HIPFFT_FORWARD |
| CUFFT_INVERSE | HIPFFT_BACKWARD |
| CUFFT_SUCCESS | HIPFFT_SUCCESS |

### cuComplex usage

| Symbol | HIP equivalent |
|--------|----------------|
| cuFloatComplex | hipFloatComplex |
| cuDoubleComplex | hipDoubleComplex |
| cuCrealf | hipCrealf |
| cuCimagf | hipCimagf |
| cuCreal | hipCreal |
| cuCimag | hipCimag |
| cuCaddf | hipCaddf |
| cuCadd | hipCadd |
| cuCmulf | hipCmulf |
| cuCmul | hipCmul |
| make_cuFloatComplex | make_hipFloatComplex |
| make_cuDoubleComplex | make_hipDoubleComplex |

### Warp intrinsics

**None directly used.** The warp-synchronous reductions use volatile shared memory (`volatile cuFloatComplex* sdata`) without explicit warp intrinsics like __shfl.

### Atomics

- `atomicAdd` on float/double in utility.cu (integrateDetector, DPC_numerator_reduce, DPC_denominator_reduce) -- works identically on HIP

### Textures/Surfaces

**None.** No texture or surface usage found.

### Thrust/CUB

**None.** No Thrust or CUB usage found.

### Pinned/Managed memory

- `cudaMallocHost` -- pinned host memory, maps to `hipHostMalloc`
- `cudaMallocManaged` -- unified/managed memory (1 instance in utility.cu for debug), maps to `hipMallocManaged`

### Streams/Events

- `cudaStream_t` and stream management throughout
- No explicit cudaEvent_t usage found

## Risk list

### Medium risk: Classic unrolled warp-synchronous reduction (PRISM03_calcOutput.cu)

The `warpReduce_cx` template functions (lines 727-947) implement the classic CUDA unrolled warp-synchronous reduction using `volatile` shared memory without __syncwarp. This pattern RACES on wave64 because the low 32 lanes of a 64-lane wavefront are not lockstep with the high 32.

**Fix per PORTING_GUIDE (GPUMD lesson)**: On HIP, drop the warp-synchronous tail entirely and let the __syncthreads()-synchronized tree run to size 1. Keep the CUDA path unchanged. The template BlockSize_numBeams values are: 2, 4, 8, 16, 32, 64, 128, 256, 512.

Example fix pattern:
```cpp
#if defined(USE_HIP)
// On HIP, do full __syncthreads() tree to size 1 -- no warp-synchronous tail
if (BlockSize_numBeams >= 64 && idx < 32) {
    sdata[idx].x += sdata[idx + 32].x;
    sdata[idx].y += sdata[idx + 32].y;
}
__syncthreads();
if (BlockSize_numBeams >= 32 && idx < 16) {
    sdata[idx].x += sdata[idx + 16].x;
    sdata[idx].y += sdata[idx + 16].y;
}
__syncthreads();
// ... continue to size 1
#else
// Original CUDA warp-synchronous reduction
#endif
```

### Low risk: Legacy FindCUDA shim

Requires implementing `cuda_add_executable`/`cuda_add_library`/`cuda_add_cufft_to_target` shim macros. Per cupoch lesson, this is mechanical: the shim forwards to add_executable/add_library, marks sources LANGUAGE HIP, and links hip::host.

### Low risk: cuFFT -> hipFFT

Straightforward 1:1 symbol swap. Note CUFFT_INVERSE -> HIPFFT_BACKWARD naming difference.

### Low risk: cuComplex -> hipComplex

1:1 symbol swap via compat header.

### Low risk: Compile-time arch default

CMakeLists.txt line 28 has `-arch=sm_60`. For HIP, set CMAKE_HIP_ARCHITECTURES from cache variable (default gfx90a when unset) to allow followers to configure their arch without source edits.

### Low risk: volatile shared memory pattern

The warp reductions use `volatile cuFloatComplex*` which is C++ standard but slightly discouraged. Works on HIP unchanged.

## File-by-file change list

### New files

| File | Description |
|------|-------------|
| `include/cuda_to_hip.h` | CUDA->HIP compat header with all symbol mappings |
| `cmake/HIPShim.cmake` | USE_HIP option, HIP language enable, cuda_add_* shim macros |

### Modified files

| File | Changes |
|------|---------|
| `CMakeLists.txt` | Add USE_HIP option at top; include HIPShim.cmake under USE_HIP before CUDA checks; skip find_package(CUDA) when USE_HIP; link hipfft under USE_HIP |
| `include/defines.h` | Include cuda_to_hip.h at top of PRISMATIC_ENABLE_GPU block; add hipFFT equivalents for PRISMATIC_CUFFT_* macros |
| `src/PRISM03_calcOutput.cu` | Fix warpReduce_cx templates: USE_HIP-guarded __syncthreads() tree instead of unsynced warp tail |
| `src/PRISM02_calcSMatrix.cu` | Include cuda_to_hip.h (if not via defines.h) |
| `src/Multislice_calcOutput.cu` | Include cuda_to_hip.h (if not via defines.h) |
| `src/utility.cu` | Include cuda_to_hip.h (if not via defines.h) |
| `src/fileIO.cu` | Include cuda_to_hip.h (if not via defines.h) |

## Build commands

### Configure (gfx90a)
```bash
mkdir build && cd build
cmake .. \
    -DUSE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx90a \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DPRISMATIC_ENABLE_GPU=ON \
    -DPRISMATIC_ENABLE_CLI=ON \
    -DPRISMATIC_TESTS=ON \
    -DCMAKE_PREFIX_PATH="/opt/rocm;$CONDA_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release
```

### Build
```bash
cmake --build . -j$(nproc)
```

### Follower configure (gfx1100)
```bash
cmake .. \
    -DUSE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
    # ... rest same
```

## Test plan

### Unit tests (Boost.Test)

The project has a comprehensive Boost.Test suite with ~25 test cases in 8 test files:

```bash
# Run all tests
./prismatic-tests --log_level=all

# Run specific test suites
./prismatic-tests --run_test=potentialTests
./prismatic-tests --run_test=probeTests
./prismatic-tests --run_test=processingTests
./prismatic-tests --run_test=ioTests
./prismatic-tests --run_test=hrtemTests
./prismatic-tests --run_test=aberrationsTests
./prismatic-tests --run_test=seriesTests
./prismatic-tests --run_test=refocusTests
```

Test categories:
- potentialTests: Projected potential calculation (CPU/GPU)
- probeTests: Probe generation and grid calculations
- processingTests: Array binning, noise, subindexing
- ioTests: HDF5 file I/O, data reorganization
- hrtemTests: HRTEM simulation
- aberrationsTests: Aberration handling
- seriesTests: CC series simulation
- refocusTests: Image refocus calculations

### CLI validation

```bash
# Basic PRISM simulation
./prismatic -i ../SI100.XYZ -o output.h5 -a p -g 1 -j 4

# Basic Multislice simulation
./prismatic -i ../SI100.XYZ -o output_ms.h5 -a m -g 1 -j 4

# Verify output exists and contains expected datasets
h5ls output.h5
```

### Determinism check

```bash
# Run twice with same parameters, compare outputs
./prismatic -i ../SI100.XYZ -o run1.h5 -a p -g 1 -j 1
./prismatic -i ../SI100.XYZ -o run2.h5 -a p -g 1 -j 1
h5diff run1.h5 run2.h5
```

### GPU validation

Confirm GPU execution via:
```bash
AMD_LOG_LEVEL=3 ./prismatic -i ../SI100.XYZ -o test.h5 -a p -g 1 2>&1 | grep -i 'hipLaunchKernel\|kernel'
```

### Non-GPU regression tests

The tests that do not require GPU (potentialTests, processingTests, some ioTests) must not regress. Build and test in CPU-only mode:
```bash
cmake .. -DPRISMATIC_ENABLE_GPU=OFF -DPRISMATIC_TESTS=ON
cmake --build . -j$(nproc)
./prismatic-tests --log_level=all
```

### Physics validation

Compare GPU output to reference:
- Load SI100.XYZ with known parameters
- Run both PRISM and Multislice algorithms
- Output should match expected diffraction patterns (visual inspection or diff against known-good HDF5)

## Open questions

1. **Upstream PR acceptance**: The project is no longer actively maintained (as of Jan 2026). An upstream PR may not be merged but is still worth creating for community visibility. Consider whether to fork permanently under jeffdaily or attempt the PR.

2. **Multi-GPU**: The project supports multi-GPU via `-g <num_gpus>`. Need to verify `cudaSetDevice`/`hipSetDevice` works correctly across multiple AMD GPUs. Low risk -- the pattern is standard.

3. **Python bindings (pyprismatic)**: The setup.py drives CMake to build pyprismatic. After the core CLI port validates, extend the HIP support to the Python bindings. Not blocking for initial validation.

4. **GUI (prismatic-gui)**: Requires Qt5. May be scoped out of initial validation but the GPU kernels are shared with CLI, so once CLI validates, GUI should work with identical kernels.
