# PLVS Port Plan

## Project

- **Name**: plvs
- **Upstream**: https://github.com/luigifreda/plvs
- **Default branch**: main
- **Description**: Real-time SLAM system combining sparse SLAM, volumetric mapping, and 3D unsupervised incremental segmentation. Supports Points, Lines, Volumetric mapping, and Segmentation (PLVS).

## Existing AMD support

**None found.** Searches performed:
- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/` -- no hits
- Web search for "plvs SLAM ROCm AMD GPU HIP" -- no relevant results
- `gh api repos/luigifreda/plvs/forks` -- no AMD/ROCm/HIP-named forks
- Upstream branches -- no rocm/hip branches
- No upstream PRs or issues mentioning AMD/ROCm/HIP

**Decision**: Proceed with a from-scratch ROCm/HIP port of the CUDA code.

## Build classification

**Pure CMake project** (Strategy A)

**Evidence** (CMakeLists.txt lines 17, 128-190):
- `set(WITH_CUDA OFF CACHE BOOL "Add CUDA support")` -- optional CUDA flag
- `find_package(CUDA REQUIRED)` when WITH_CUDA=ON
- Uses `cuda_compile(CUDA_OBJS ${CUDA_SOURCES})` to compile `.cu` files
- No `find_package(Torch)`, no `torch.utils.cpp_extension`, no setup.py
- Standard CMake with CUDA language, not a PyTorch extension

**CUDA sources** (CMakeLists.txt lines 137-142):
```cmake
set(CUDA_SOURCES
    src/cuda/Allocator_gpu.cu
    src/cuda/Fast_gpu.cu
    src/cuda/Orb_gpu.cu
    src/cuda/Cuda.cu
)
```

## Port strategy

**Strategy A (compat header + LANGUAGE HIP)**

Rationale:
- Pure CMake build with small, focused CUDA kernel files
- Main CUDA code is isolated in `src/cuda/` directory (4 files, ~67KB total)
- No warp intrinsics (`__shfl*`, `__ballot`, `warpSize`) in the main PLVS CUDA code
- No cuBLAS/cuFFT/cuRAND/cuSPARSE usage in main code
- Uses OpenCV CUDA headers (may need compatibility work)

**Third-party CUDA libraries**: libsgm and libelas-gpu are bundled third-party libraries with their own CUDA code. These require separate HIP porting:
- **libsgm**: Has hardcoded `WARP_SIZE = 32` (utility.hpp:24) and uses `__shfl_*` intrinsics -- requires wave64-aware fixes for CDNA
- **libelas-gpu**: No warp intrinsics, straightforward port

## CUDA surface inventory

### Main PLVS CUDA code (src/cuda/)

| File | Description | CUDA symbols |
|------|-------------|--------------|
| Allocator_gpu.cu | Custom GPU memory allocator | cudaMallocManaged, cudaFree |
| Cuda.cu | Device sync wrapper | cudaDeviceSynchronize |
| Orb_gpu.cu | ORB descriptor computation | cudaMemcpyToSymbol, cudaMalloc, cudaFree, cudaStreamCreate/Destroy, cudaMemcpyAsync, __constant__, __global__ |
| Fast_gpu.cu | FAST keypoint detection (~58KB) | cudaMallocManaged, cudaStreamAttachMemAsync, cudaMemsetAsync, __syncthreads, __global__, __device__, OpenCV CUDA reduce<32>, thrust::tie |

**Kernels**:
- `calcOrb_kernel` -- ORB descriptor computation
- `tileCalcKeypoints_kernel` -- FAST keypoint detection
- `IC_Angle_kernel` -- Keypoint orientation
- `addBorder_kernel` -- Border addition

**No warp primitives** (`__shfl*`, `__ballot`, `warpSize`) in main PLVS code.

**OpenCV CUDA dependency**: Uses `opencv2/core/cuda/{common,utility,reduce,functional}.hpp`. These are OpenCV's internal CUDA headers that include warp-level operations. OpenCV 4.x has HIP support via cmake option `-DWITH_HIP=ON`.

### Third-party: libsgm (Thirdparty/libsgm/)

| File | Warp usage |
|------|------------|
| utility.hpp:24 | `static constexpr unsigned int WARP_SIZE = 32u;` -- **HARDCODED** |
| winner_takes_all.cu | `__shfl_xor_sync`, `__shfl_sync` with GROUP_SIZE/WARP_SIZE |
| horizontal_path_aggregation.cu | `__shfl_up_sync`, `__shfl_down_sync` with SUBGROUP_SIZE |
| vertical_path_aggregation.cu | WARP_SIZE-derived SUBGROUP_SIZE |
| oblique_path_aggregation.cu | WARP_SIZE-derived SUBGROUP_SIZE |

**Risk**: libsgm assumes 32-lane warps. On wave64 (gfx90a), this will produce incorrect results unless WARP_SIZE is made architecture-aware.

### Third-party: libelas-gpu (Thirdparty/libelas-gpu/)

| File | CUDA usage |
|------|------------|
| GPU/elas_gpu.cu | __global__, __device__, __constant__, __shared__, __sad intrinsic |

**No warp primitives** -- straightforward port.

### External CUDA libraries

| Library | Usage | HIP equivalent |
|---------|-------|----------------|
| cudaMallocManaged | Unified memory | hipMallocManaged |
| cudaStreamAttachMemAsync | Stream memory attachment | hipStreamAttachMemAsync |
| NVTX (optional) | Profiling | roctx (optional) |

## Risk list

1. **libsgm WARP_SIZE=32 hardcode** (HIGH): libsgm hardcodes `WARP_SIZE = 32u` in utility.hpp. On wave64 (gfx90a), this breaks the shuffle-based reductions. Fix: make WARP_SIZE architecture-aware using `__GFX9__` guards (64 for CDNA, 32 for RDNA), or use logical 32-lane operations with tiled_partition.

2. **OpenCV CUDA headers** (MEDIUM): Fast_gpu.cu uses OpenCV CUDA internal headers (`opencv2/core/cuda/reduce.hpp`). OpenCV 4.x supports HIP via `-DWITH_HIP=ON`. Either:
   - Build OpenCV with HIP support and link against it, OR
   - Rewrite the reduce<32> calls to use a manual block reduction

3. **cudaMallocManaged / Unified memory** (LOW): HIP supports `hipMallocManaged` but with different coherence semantics. The PLVS usage (simple allocator, stream-attached memory) should work unchanged. Note the PORTING_GUIDE warning about atomicMin/atomicMax on coarse-grained managed memory on gfx90a (but PLVS does not use these atomics in managed memory).

4. **`__sad` intrinsic** (LOW): libelas-gpu uses `__sad()` (sum of absolute differences). HIP provides `__sad()` so this is 1:1.

5. **helper_cuda.h error strings** (LOW): The project bundles NVIDIA's helper_cuda.h which has error enum tables for cuBLAS/cuFFT/cuSPARSE/cuRAND. These are not used at runtime (just error-to-string helpers) and can be ifdef'd out for HIP.

6. **OpenCV cv::cuda::GpuMat** (LOW): The code uses OpenCV's GpuMat with custom allocator. OpenCV with HIP support provides HIP-compatible GpuMat.

## File-by-file change list

### CMakeLists.txt
- Add `option(USE_HIP "Build with HIP for AMD GPUs" OFF)`
- When USE_HIP: `enable_language(HIP)`, set CMAKE_HIP_ARCHITECTURES with default to gfx90a
- Mark CUDA_SOURCES as `LANGUAGE HIP` under USE_HIP
- Add include path for cuda_to_hip.h compat header

### src/cuda/cuda_to_hip.h (NEW)
- Compat header with runtime API aliases (cudaMalloc->hipMalloc, etc.)
- Include at top of all .cu files

### src/cuda/Allocator_gpu.cu
- Include cuda_to_hip.h
- No other changes expected

### src/cuda/Cuda.cu
- Include cuda_to_hip.h
- No other changes expected

### src/cuda/Orb_gpu.cu
- Include cuda_to_hip.h
- `cv::cuda::StreamAccessor::wrapStream` -- verify OpenCV HIP compatibility
- No other changes expected

### src/cuda/Fast_gpu.cu
- Include cuda_to_hip.h
- OpenCV CUDA headers dependency -- either:
  - Build with HIP-enabled OpenCV, OR
  - Replace reduce<32> with a manual reduction (the reduce<32> operates on width-32 logical groups, which is arch-agnostic)
- No warp-size changes needed for main code

### include/cuda/helper_cuda.h
- Guard the cuBLAS/cuFFT/cuSPARSE/cuRAND error enum tables with `#ifndef USE_HIP`

### include/cuda/Utils.hpp
- NVTX ifdef already exists; no changes needed unless roctx wanted

### Thirdparty/libsgm/src/utility.hpp
- Change `static constexpr unsigned int WARP_SIZE = 32u;` to architecture-aware:
```cpp
#if defined(__HIP_PLATFORM_AMD__)
  #if defined(__GFX9__)
    static constexpr unsigned int WARP_SIZE = 64u;
  #else
    static constexpr unsigned int WARP_SIZE = 32u;
  #endif
#else
  static constexpr unsigned int WARP_SIZE = 32u;
#endif
```
- OR: use 32-lane logical operations within 64-lane wavefronts (SUBGROUP_SIZE and GROUP_SIZE are already < 32 in many cases)

### Thirdparty/libsgm/CMakeLists.txt
- Add HIP build option similar to main CMakeLists.txt

### Thirdparty/libelas-gpu/CMakeLists.txt
- Add HIP build option
- Mark .cu files as LANGUAGE HIP

### Thirdparty/libelas-gpu/GPU/elas_gpu.cu
- Include a compat header or use HIP-compatible symbols
- `__sad` intrinsic should work unchanged on HIP

## Build commands

### Configure (gfx90a)
```bash
# Build third-party libraries first
cd Thirdparty/libsgm
mkdir -p build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
make -j$(nproc)
cd ../../..

cd Thirdparty/libelas-gpu
mkdir -p build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
make -j$(nproc)
cd ../../..

# Main build
mkdir -p build && cd build
cmake .. \
  -DWITH_CUDA=OFF \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DWITH_LIBSGM=ON \
  -DWITH_LIBELAS=ON
```

### Build
```bash
make -j$(nproc)
```

### Dependencies
- ROCm 7.x (tested with 7.2.1)
- OpenCV 4.x (either with `-DWITH_HIP=ON` for full GPU support, or standard build for CPU-only validation)
- Other dependencies per upstream: Eigen3, Boost, Pangolin, PCL, GLOG, octomap, Protobuf, GLFW3

## Test plan

### GPU tests
PLVS does not have a traditional unit test suite. Validation is through running the SLAM examples on standard datasets:

1. **TUM RGB-D datasets**: `./Scripts/run_tum_rgbd.sh`
   - Configure DATASET_BASE_FOLDER and DATASET in script
   - Verifies full pipeline: feature extraction (CUDA), tracking, mapping

2. **EuRoC stereo datasets**: `./Scripts/run_euroc_stereo.sh`
   - Tests stereo disparity (libsgm) if enabled

3. **KITTI stereo datasets**: `./Scripts/run_kitti_stereo.sh`
   - Tests stereo pipeline

**Validation criteria**:
- No runtime errors or GPU crashes
- Trajectory output matches expected format
- Determinism: two runs on same dataset produce identical trajectory output
- Visual inspection of point cloud and map outputs in viewer

### Non-GPU regression tests
- CPU-only build (`-DWITH_CUDA=OFF -DUSE_HIP=OFF`) must continue to work
- Mono SLAM examples (no stereo disparity) work without GPU
- Vocabulary generation tool compiles and runs

### Benchmarking (optional)
- `Benchmarking/benchmark_tum.sh` for timing comparisons

## Open questions

1. **OpenCV CUDA/HIP compatibility**: Does the installed OpenCV have HIP support? If not, need to either:
   - Build OpenCV from source with `-DWITH_HIP=ON`, OR
   - Rewrite Fast_gpu.cu's reduce<32> calls manually

2. **libsgm wave64 correctness**: The libsgm fix may require more than just changing WARP_SIZE constant. The shuffle operations use SUBGROUP_SIZE which is `MAX_DISPARITY / DP_BLOCK_SIZE` (often 8 or 16, well under 32). Need to verify if wave64 affects the shuffle semantics at these smaller widths.

3. **libsgm upstream**: libsgm is bundled from fixstars/libSGM. Should we port the bundled copy or reference an existing AMD port? Quick search shows no existing AMD port of libSGM.

4. **Runtime warp size query**: For host-side code that needs warp size, use `hipGetDeviceProperties(&prop, dev); prop.warpSize` as per PORTING_GUIDE.

5. **Managed memory coherence**: Verify that hipMallocManaged with hipStreamAttachMemAsync works correctly for the GpuMat allocator pattern.
