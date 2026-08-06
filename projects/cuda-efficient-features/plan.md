# cuda-efficient-features Port Plan

## Project
- **Name**: cuda-efficient-features
- **Upstream**: https://github.com/fixstars/cuda-efficient-features
- **Default branch**: main
- **Description**: CUDA implementation of keypoint detection (multi-scale FAST) and descriptor extraction (BAD, HashSIFT)

## Existing AMD support
**None found.**

Searches performed:
- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/` -- no results
- WebSearch for "cuda-efficient-features ROCm AMD HIP" -- no relevant results
- `gh api repos/fixstars/cuda-efficient-features/forks` -- no AMD/ROCm forks (11 forks, all personal)
- No upstream rocm/hip branches
- No relevant PRs or issues

**Decision**: Proceed with fresh port.

## Build classification: Pure CMake (Strategy A)
**Evidence**:
- `/modules/cuda_efficient_features/CMakeLists.txt` line 8: `project(${PROJECT_NAME} LANGUAGES CXX CUDA)`
- No `find_package(Torch)`, no `torch.utils.cpp_extension`, no pytorch dependency
- Uses `find_package(CUDAToolkit REQUIRED)` and `find_package(OpenCV REQUIRED)`

## Port strategy: Strategy A (compat-header approach)

**Rationale**: This is a pure CMake project. Add a `cuda_to_hip.h` compat header, mark `.cu` files as `LANGUAGE HIP` when `USE_HIP` is enabled, and swap cuBLAS for hipBLAS.

## CUDA surface inventory

### Runtime API symbols (need compat-header aliases)
| Symbol | Count | Files |
|--------|-------|-------|
| `cudaStream_t` | ~10 | cuda_efficient_features.cu, cuda_bad.cu, cuda_hash_sift.cu, cuda_fast.cu |
| `cudaMemcpyAsync` | 2 | cuda_efficient_features.cu |
| `cudaMemsetAsync` | 1 | cuda_efficient_features.cu |
| `cudaMemcpyDeviceToDevice` | 1 | cuda_efficient_features.cu |
| `cudaMemcpyDeviceToHost` | 1 | cuda_efficient_features.cu |
| `cudaStreamSynchronize` | 1 | cuda_efficient_features.cu |
| `cudaGetLastError` | ~10 | all .cu files |
| `cudaMalloc` | 1 | device_buffer.cpp |
| `cudaFree` | 1 | device_buffer.cpp |
| `cudaMemcpyToSymbol` | 2 | cuda_bad.cu |
| `cudaSuccess` | implicit | via CUDA_CHECK macro |

### Kernels (`__global__`)
| Kernel | File | Notes |
|--------|------|-------|
| `computeBADKernel` | cuda_bad.cu | Uses `__shfl_xor_sync` with width 4,2,1 (OK for wave64) |
| `computePatchSIFTKernel` | cuda_hash_sift.cu | Uses `__shfl_xor_sync` in `normalizeDescriptors` |
| `binarizeDescriptorsKernel` | cuda_hash_sift.cu | Simple pixel ops |
| `calcKeypointsKernel` | cuda_fast.cu | FAST corner detection |
| `nptPerBlockKernel` | cuda_efficient_features.cu | atomicAdd for counting |
| `assignIndexKernel` | cuda_efficient_features.cu | atomicAdd for indexing |
| `radiusSuppressionKernel` | cuda_efficient_features.cu | Local maxima suppression |
| `calcResponsesKernel` | cuda_efficient_features.cu | Harris corner response |
| `calcAnglesKernel` | cuda_efficient_features.cu | Orientation calculation |
| `scalePointsKernel` | cuda_efficient_features.cu | Scale keypoints |
| `convertKeypointsKernel` | cuda_efficient_features.cu | Format conversion |

### Warp intrinsics
| Intrinsic | File | Line | Analysis |
|-----------|------|------|----------|
| `__shfl_xor_sync(0xffffffff, byte, 4)` | cuda_bad.cu | 306 | Width 4 shuffle -- OK on wave64/wave32 |
| `__shfl_xor_sync(0xffffffff, byte, 2)` | cuda_bad.cu | 307 | Width 2 shuffle -- OK |
| `__shfl_xor_sync(0xffffffff, byte, 1)` | cuda_bad.cu | 308 | Width 1 shuffle -- OK |
| `__shfl_xor_sync(0xffffffff, sum, mask)` | cuda_hash_sift.cu | 191 | Warp reduction with masks 16,8,4,2,1 -- OK |

**Warp-size analysis**: The shuffles use small widths (1-16) which work correctly on both wave64 and wave32. The `WARP_SIZE` constant at cuda_hash_sift.cu:39 is used only for loop stride (`i += WARP_SIZE`) where threadIdx.x iterates over DESCRIPTOR_SIZE (128), so the iteration works regardless of physical warp size.

### Constant memory
| Symbol | File |
|--------|------|
| `__constant__ BoxPairParams box_pair_params[512]` | cuda_bad.cu |
| `__constant__ float thresholds_[512]` | cuda_bad.cu |
| `__constant__ uchar c_table[]` | cuda_fast.cu |

### Shared memory
| Symbol | File |
|--------|------|
| `__shared__ uchar patch[32][32]` | cuda_hash_sift.cu |
| `__shared__ float histbuf[6][6][10]` | cuda_hash_sift.cu |
| `__shared__ float descriptors[129]` | cuda_hash_sift.cu |

### Library usage
| Library | Usage | HIP equivalent |
|---------|-------|----------------|
| cuBLAS | `cublasSgemm_v2` for HashSIFT GEMM | hipBLAS |
| Thrust | `thrust::exclusive_scan`, `thrust::sort_by_key` | rocThrust (drop-in) |

### OpenCV CUDA modules dependency (CRITICAL BLOCKER)
| Module | Usage | ROCm support |
|--------|-------|--------------|
| `opencv2/cudafilters.hpp` | `cuda::createGaussianFilter` | **NO native support** |
| `opencv2/cudaimgproc.hpp` | Included but usage unclear | **NO native support** |
| `opencv2/cudawarping.hpp` | `resize()` for pyramid | **NO native support** |
| `opencv2/cudev/grid/detail/integral.hpp` | `integral_detail::integral` | **NO native support** |
| `opencv2/core/cuda.hpp` | `GpuMat`, `Stream` | **NO native support** |

**OpenCV CUDA modules do NOT support ROCm/HIP.** The project deeply integrates with OpenCV's CUDA backend for:
1. Image pyramid construction (`resize`)
2. Gaussian blur (`createGaussianFilter`)
3. Integral image computation (`cudev::integral_detail::integral`)
4. GPU memory management (`GpuMat`, `Stream`)

## Risk list

### BLOCKER: OpenCV CUDA dependency
**Severity**: CRITICAL -- blocks the port entirely

OpenCV's CUDA modules (`cudafilters`, `cudaimgproc`, `cudawarping`, `cudev`) are NVIDIA-specific and have no ROCm/HIP support. The project uses these for:
- `cuda::createGaussianFilter()` -- 7x7 Gaussian blur for descriptor computation
- `resize()` from cudawarping -- image pyramid
- `cudev::integral_detail::integral()` -- integral image for BAD descriptor
- `GpuMat` and `Stream` -- GPU memory and stream management

**Options**:
1. **Rewrite OpenCV CUDA calls to HIP**: Replace `GpuMat` with raw `hipMalloc`/`hipMemcpy`, implement Gaussian filter kernel, implement resize kernel, implement integral image kernel. This is substantial work (~500-800 lines) but feasible.
2. **Use AMD alternatives**: rocCV (preview only in ROCm 7.0), RPP (ROCm Performance Primitives), or MIVisionX. These have different APIs and would require significant refactoring.
3. **Skip the port**: If the effort outweighs the benefit.

**Recommendation**: Option 1 (rewrite OpenCV CUDA calls). The OpenCV usage is limited to:
- Gaussian blur: straightforward separable filter kernel
- Image resize: bilinear interpolation kernel
- Integral image: prefix sum kernel
- GpuMat: replace with raw device pointers + hipMemcpy

This is achievable but requires writing ~4 custom kernels to replace OpenCV CUDA functionality.

### Moderate: Warp-size constant
`static constexpr int WARP_SIZE = 32` in cuda_hash_sift.cu is used only as loop stride, which is arch-agnostic. No fix needed.

### Low: cuBLAS -> hipBLAS
Straightforward 1:1 API mapping. The v2 suffix drops in hipBLAS.

### Low: Thrust -> rocThrust
Drop-in replacement, same API.

## File-by-file change list

### New files
1. `modules/cuda_efficient_features/src/cuda_to_hip.h` -- compat header with CUDA->HIP symbol aliases
2. `modules/cuda_efficient_features/src/hip_kernels.cu` -- replacement kernels for OpenCV CUDA functions (resize, Gaussian blur, integral)

### Modified files

**CMakeLists.txt (root)**
- Add `USE_HIP` option
- Default HIP arch when unset

**modules/cuda_efficient_features/CMakeLists.txt**
- Gate language on `USE_HIP`: `enable_language(HIP)` vs `enable_language(CUDA)`
- Mark `.cu` sources as `LANGUAGE HIP` when `USE_HIP=ON`
- Replace `CUDA::cublas` with `hip::hipblas` when `USE_HIP=ON`
- Add `--offload-arch` for HIP architectures

**modules/cuda_efficient_features/src/cuda_macro.h**
- Include `cuda_to_hip.h` header
- Alias `cudaError_t`, `cudaSuccess`, `cudaGetErrorString` for HIP

**modules/cuda_efficient_features/src/device_buffer.cpp**
- Include `cuda_to_hip.h`

**modules/cuda_efficient_features/src/cuda_bad.cu**
- Include `cuda_to_hip.h`
- Replace `cudev::integral_detail::integral` with custom HIP integral kernel
- Handle `cudaMemcpyToSymbol` -> `hipMemcpyToSymbol`

**modules/cuda_efficient_features/src/cuda_hash_sift.cu**
- Include `cuda_to_hip.h`

**modules/cuda_efficient_features/src/cuda_hash_sift.cpp**
- Replace `cublas_v2.h` with `hipblas/hipblas.h` under `USE_HIP` guard
- Replace `cublasSgemm_v2` with `hipblasSgemm`
- Replace `cublasHandle_t` with `hipblasHandle_t`
- Replace CUBLAS enums with HIPBLAS equivalents

**modules/cuda_efficient_features/src/cuda_efficient_features.cu**
- Include `cuda_to_hip.h`

**modules/cuda_efficient_features/src/cuda_efficient_features.cpp**
- Replace OpenCV CUDA includes with HIP kernels under `USE_HIP` guard:
  - `cuda::createGaussianFilter` -> custom HIP kernel
  - `resize` -> custom HIP resize kernel
- Replace `GpuMat` operations with raw HIP memory ops under `USE_HIP` guard
- This is the most substantial change file

**modules/cuda_efficient_features/src/cuda_fast.cu**
- Include `cuda_to_hip.h`

### Tests
**tests/CMakeLists.txt**
- Gate `cuda_add_executable` on CUDA vs HIP
- For HIP, use standard `add_executable` with HIP language

## Build commands

### Configure (gfx90a)
```bash
cmake -B build \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DBUILD_TESTS=ON \
  -DBUILD_SAMPLES=ON \
  -DOpenCV_DIR=/path/to/opencv/build \
  ..
```

### Build
```bash
cmake --build build -j$(nproc)
```

**Note**: OpenCV must be built WITHOUT CUDA modules for the HIP build, as we replace all OpenCV CUDA functionality with custom HIP kernels.

## Test plan

### GPU tests
```bash
# Initialize test data submodules
cd tests && git submodule update --init

# Run tests
./build/tests/tests
```

The test suite (`descriptor_test.cpp`) runs:
- BAD descriptor test: 22 test cases (2 bit-depths x 11 images)
- HashSIFT descriptor test: 22 test cases (2 bit-depths x 11 images)

Tests compare GPU descriptor output against CPU reference with error tolerance.

### Sample validation
```bash
# Feature extraction
./build/samples/sample_feature_extraction tests/data/images/100_7100.JPG

# Feature matching
./build/samples/sample_feature_matching tests/data/images/100_7100.JPG tests/data/images/100_7101.JPG

# Benchmark
./build/samples/sample_benchmark tests/data/images/100_7100.JPG
```

### Non-GPU tests
The `efficient_features` module (CPU implementation) has no tests but should not regress -- verify samples work in CPU mode.

## Open questions

1. **OpenCV CUDA replacement scope**: The plan assumes rewriting OpenCV CUDA calls with custom HIP kernels. An alternative is to require a HIP-compatible OpenCV build, but such a build does not exist upstream. Confirm this approach is acceptable.

2. **GpuMat abstraction**: Replacing `GpuMat` with raw pointers changes the API surface. Should we keep a thin `HipMat` wrapper for source compatibility, or refactor more extensively?

3. **Test data**: Tests require the SceauxCastle dataset via git submodule. Confirm this is accessible on the test system.

4. **OpenCV build for tests**: The CPU reference in tests uses OpenCV's `cv::BAD` and `cv::HashSIFT` from the `efficient_features` module (CPU-only). This should work without modification.
