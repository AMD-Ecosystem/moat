# SCAMP ROCm Port Plan

## Project

- **Name**: SCAMP
- **Upstream**: https://github.com/zpzim/SCAMP
- **Default branch**: main
- **Description**: Fast GPU implementation for computing matrix profiles on time series data

## Existing AMD support

**Assessment**: No AMD/ROCm/HIP support exists.

- Searched README.md, docs/ -- no mention of AMD, ROCm, HIP, or gfx architectures
- Web search for "SCAMP ROCm", "SCAMP AMD GPU", "SCAMP HIP" found no results
- `gh api repos/zpzim/SCAMP/forks` shows ~30 forks, none with rocm/hip/amd in name or description
- Project documentation explicitly mentions "NVIDIA GPU" and "cuda" throughout
- No upstream rocm/hip branches or related PRs/issues

**Decision**: Proceed with fresh ROCm/HIP port. No existing work to reuse or improve.

## Build classification

**Classification**: Pure CMake project (Strategy A)

**Evidence**:
- Root CMakeLists.txt uses `check_language(CUDA)` + `enable_language(CUDA)` (line 86-98)
- No `find_package(Torch)`, no `torch.utils.cpp_extension`, no `CUDAExtension`
- No setup.py torch dependency (setup.py exists but is for pyscamp without CUDAExtension)
- Uses `CUDA::cufft` from CUDAToolkit (line 17 of src/core/CMakeLists.txt)
- CUB usage via `#include <cub/device/device_merge_sort.cuh>` (kernels.cu:8)

## Port strategy

**Strategy A: Pure CMake, compat-header approach (preferred)**

**Rationale**:
- Pure CMake CUDA project with `.cu` files that should compile with HIP's `enable_language(HIP)` + `LANGUAGE HIP`
- Single compat header (`cuda_to_hip.h`) can alias all CUDA runtime + cuFFT + CUB symbols
- Minimal diff vs upstream; NVIDIA path remains untouched

## CUDA surface inventory

### Kernels and device functions

| File | Type | Description |
|------|------|-------------|
| `src/core/gpu_kernel/kernels.cu` | Entry | Top-level dispatcher, CUB DeviceMergeSort |
| `src/core/gpu_kernel/kernel_1nn.cu` | Kernel | 1NN profile kernel |
| `src/core/gpu_kernel/kernel_1nn_index.cu` | Kernel | 1NN with index kernel |
| `src/core/gpu_kernel/kernel_sum_thresh.cu` | Kernel | Sum threshold kernel |
| `src/core/gpu_kernel/kernel_matrix_summary.cu` | Kernel | Matrix summary kernel |
| `src/core/gpu_kernel/kernel_approx_all_neighbors.cu` | Kernel | Approximate all neighbors kernel |
| `src/core/gpu_kernel/kernel_gpu_utils.cu` | Utils | GPU utilities |
| `src/core/gpu_kernel/qt_kernels.cu` | FFT helpers | cuFFT-based sliding dot product kernels |
| `src/core/gpu_kernel/kernel_variant*.cu.in` | Generated | Per-variant TUs generated at configure time |

### Warp intrinsics (RISK: hardcoded 32-lane assumptions)

| File | Line(s) | Usage | Wave64 Risk |
|------|---------|-------|-------------|
| `kernels_compute.h` | 157 | `__shfl_down_sync(0xffffffff, sum, i)` for i in {16,8,4,2,1} | Yes: 32-bit mask |
| `kernels_compute.h` | 159 | `(threadIdx.x & 0x1f) == 0` lane-0 check | Yes: assumes 32-lane warp |
| `kernels_compute_shfl.h` | 388-402 | `__shfl_down_sync(0xffffffffu, ...)` warp reduction | Yes: 32-bit mask |
| `kernels_compute_shfl.h` | 556 | `__shfl_sync(0xffffffffu, state.cov[DPT-1], ...)` | Yes: 32-bit mask |
| `kernels_impl_shfl.h` | 63 | `warps_per_block = BLOCKSZ / 32` | Yes: hardcoded warp size |
| `kernels_impl_shfl.h` | 64 | `static_assert(BLOCKSZ % 32 == 0, ...)` | Yes: hardcoded warp size |
| `kernels_impl_shfl.h` | 65-69 | `tile_height <= 32 * DPT` static_assert | Yes: hardcoded warp size |
| `kernels_impl_shfl.h` | 117 | `state.warpln = threadIdx.x & 31u` | Yes: hardcoded |
| `kernels_impl_shfl.h` | 119 | `state.srcln = (state.warpln - 1u) & 31u` | Yes: hardcoded |
| `kernel_gpu_utils.cu` | 68 | `warps_per_block = blocksz / 32` | Yes: hardcoded |

### Library usage

| Library | Files | HIP Equivalent |
|---------|-------|----------------|
| cuFFT | `qt_helper.cpp`, `qt_helper.h`, `qt_kernels.h` | hipFFT |
| CUB DeviceMergeSort | `kernels.cu` | hipCUB DeviceMergeSort |
| cuda_runtime | All GPU files | HIP runtime |

### cuFFT symbols used (qt_helper.cpp/h)

- `cufftPlan1d`, `cufftDestroy`, `cufftSetStream`
- `cufftExecD2Z`, `cufftExecZ2D`, `cufftExecR2C`, `cufftExecC2R`
- `cufftHandle`, `cufftResult`, `cufftResult_t`
- `CUFFT_D2Z`, `CUFFT_Z2D`, `CUFFT_R2C`, `CUFFT_C2R`
- `cuDoubleComplex`

### CUDA runtime symbols used

- Memory: `cudaMalloc`, `cudaFree`, `cudaMallocAsync`, `cudaFreeAsync`, `cudaMemcpy`, `cudaMemset`
- Device: `cudaGetDevice`, `cudaSetDevice`, `cudaDeviceGetAttribute`, `cudaGetDeviceProperties`
- Stream: `cudaStream_t`, `cudaStreamCreate`, `cudaStreamDestroy`, `cudaStreamSynchronize`
- Error: `cudaError_t`, `cudaSuccess`, `cudaGetLastError`, `cudaGetErrorString`
- Event: `cudaEvent_t`, `cudaEventCreate`, `cudaEventRecord`, etc.

### Textures/surfaces

None used.

### Pinned/managed memory

None observed in GPU kernel code.

### Streams/events

Used extensively for async operations.

## Risk list

1. **Warp size 32 vs 64 (HIGH)**: The cov-shuffle kernel family (`kernels_compute_shfl.h`, `kernels_impl_shfl.h`) has extensive hardcoded 32-lane assumptions:
   - `__shfl_*_sync(0xffffffff, ...)` uses 32-bit masks (HIP requires 64-bit masks, see PORTING_GUIDE)
   - `threadIdx.x & 31`, `threadIdx.x & 0x1f` for lane calculation
   - `warps_per_block = BLOCKSZ / 32`
   - Tile geometry constraints `tile_height <= 32 * DPT`
   
   **Fix approach**: 
   - Device code: Use per-arch kWarpSize constant (`__GFX9__ ? 64 : 32`)
   - Host code: Query `hipGetDeviceProperties().warpSize` at runtime
   - Shuffle masks: Widen to 64-bit (`0xffffffffffffffffULL` on HIP)
   - Lane calculations: Use `threadIdx.x % kWarpSize` or `threadIdx.x & (kWarpSize-1)`
   
   **Note**: The SUM_THRESH profile uses warp-shuffle reduction with `for (i = 16; i >= 1; i /= 2)` which only covers 32 lanes. On wave64 this would need adjustment to cover 64 lanes.

2. **CUB -> hipCUB (LOW)**: Single use of `cub::DeviceMergeSort::SortKeys` in `kernels.cu`. hipCUB provides this API 1:1.

3. **cuFFT -> hipFFT (LOW)**: cuFFT symbols are isolated to `qt_helper.cpp/h` and `qt_kernels.h`. hipFFT is API-compatible. Symbol mapping is straightforward.

4. **cuda::barrier in shfl kernel (MEDIUM)**: `kernels_impl_shfl.h` uses `cuda::barrier<thread_scope_block>` for per-row synchronization. HIP may not have direct equivalent in all cases. The code already has a fallback to `__syncthreads()` for sm_60 via `ShflRowBarrier` which should work on HIP.

5. **Eigen in device code (LOW)**: The project uses Eigen arrays in device code with `--expt-relaxed-constexpr`. HIP/clang should handle this, but verify constexpr functions are callable from device.

6. **HOST_DEVICE_FUNCTION macro (LOW)**: Defined as `__host__ __device__` when `_HAS_CUDA_` is set (common/common.h:21). Needs corresponding HIP handling.

## File-by-file change list

### New files to add

1. **`src/core/gpu_kernel/cuda_to_hip.h`**: Compat header with:
   - CUDA->HIP runtime symbol aliases
   - cuFFT->hipFFT symbol aliases
   - CUB->hipCUB namespace alias
   - Warp size abstraction macros/constants
   - 64-bit shuffle mask definitions for HIP

### CMake modifications

1. **`CMakeLists.txt`** (root):
   - Add `option(USE_HIP "Build with HIP for AMD GPUs" OFF)`
   - Under USE_HIP: `enable_language(HIP)`, set `CMAKE_HIP_ARCHITECTURES` with default `gfx90a`
   - Gate CUDAToolkit find on `NOT USE_HIP`
   - Add hipFFT linking for HIP builds

2. **`src/core/CMakeLists.txt`**:
   - Link `hip::hipfft` instead of `CUDA::cufft` when USE_HIP
   - Mark sources `LANGUAGE HIP` when USE_HIP

3. **`src/core/gpu_kernel/CMakeLists.txt`**:
   - Mark GPU kernel sources `LANGUAGE HIP` when USE_HIP
   - Link hipCUB when USE_HIP

4. **`cmake/SCAMPMacros.cmake`**:
   - Add HIP arch handling in a new `set_hip_architectures()` macro
   - Modify `mark_cuda_if_available()` to also set `_HAS_HIP_` or use `_HAS_CUDA_` for unified GPU path

### Source modifications

1. **`src/common/common.h`**:
   - Extend `HOST_DEVICE_FUNCTION` macro for HIP (`defined(__HIP_PLATFORM_AMD__)`)

2. **`src/core/qt_helper.cpp`**, **`src/core/qt_helper.h`**, **`src/core/gpu_kernel/qt_kernels.h`**:
   - Include `cuda_to_hip.h` and use aliased cuFFT symbols (unchanged CUDA spelling)

3. **`src/core/gpu_kernel/kernels_compute.h`**:
   - Line 157: Widen shuffle mask, extend reduction for wave64
   - Line 159: Replace `threadIdx.x & 0x1f` with warp-size-aware lane check

4. **`src/core/gpu_kernel/kernels_compute_shfl.h`**:
   - Lines 388, 401-402, 556: Widen shuffle masks to 64-bit on HIP
   - Add wave64 reduction steps (stride 32 for wave64)

5. **`src/core/gpu_kernel/kernels_impl_shfl.h`**:
   - Line 63: Make `warps_per_block` warp-size-aware
   - Lines 64-69: Adjust static_asserts for wave64 compatibility
   - Lines 117, 119: Use kWarpSize instead of hardcoded 31/32

6. **`src/core/gpu_kernel/kernel_gpu_utils.cu`**:
   - Line 68: Use device warpSize or compile-time constant

7. **`src/core/gpu_kernel/kernels.cu`**:
   - Include `cuda_to_hip.h`
   - CUB usage should map to hipCUB via namespace alias

## Build commands

### Configure (Linux gfx90a)
```bash
cmake -B build \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DBUILD_SCAMP_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
```

### Build
```bash
cmake --build build -j$(nproc)
```

### Multi-arch verification
```bash
cmake -B build-multi \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" \
  -DBUILD_SCAMP_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-multi -j$(nproc)
llvm-objdump --offloading build-multi/SCAMP  # Verify both archs present
```

## Test plan

### GPU tests (must pass on real hardware)

1. **Integration test suite** (`test/run_tests.sh`):
   ```bash
   cd test && ./run_tests.sh ../build/SCAMP out.log
   ```
   - Validates matrix profile correctness against reference outputs
   - Tests self-join and AB-join with various tile sizes
   - Uses multiple input files (randomwalk8K through 64K_nan)
   - Compares MP values (difference.py) and MP index (diff)

2. **Python tests** (`test/test_pyscamp.py`):
   ```bash
   pip install ./build  # Install pyscamp
   pytest test/test_pyscamp.py -v
   ```

3. **C++ unit tests** (if enabled):
   ```bash
   ctest --test-dir build --output-on-failure
   ```

### Non-GPU regression tests

1. **CPU-only build** (must still work):
   ```bash
   cmake -B build-cpu -DFORCE_NO_CUDA=ON -DBUILD_SCAMP_TESTS=ON
   cmake --build build-cpu -j$(nproc)
   cd test && ./run_tests.sh ../build-cpu/SCAMP out-cpu.log
   ```

### Validation criteria

- All test/run_tests.sh tests pass with error < 1% on MP index
- difference.py reports acceptable MP value differences
- Determinism: Two runs with identical inputs produce identical outputs
- Wave64/wave32 correctness: Test on both gfx90a (wave64) and gfx1100 (wave32) when available

## Open questions

1. **Shfl kernel complexity**: The cov-shuffle kernel has deep 32-lane assumptions. Is a correctness-first port with degraded performance (disabling shfl variants on wave64) acceptable as a first step, with wave64-optimized shfl variants as a follow-up?

2. **cuda::barrier availability**: The shfl kernel uses `cuda::barrier<thread_scope_block>` on sm_70+. ROCm/HIP support for `cuda::barrier` via libhipcxx needs verification. The fallback to `__syncthreads()` exists but may impact performance.

3. **Autotune cache**: The project has an autotune system that profiles kernel configs per GPU. The autotune cache format may need HIP-specific entries. Verify autotune works correctly on AMD GPUs.

4. **Python bindings**: The pyscamp Python module uses the same GPU code. Verify the HIP build integrates correctly with pybind11 bindings.
