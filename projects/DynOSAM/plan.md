# DynOSAM Port Plan

## Project
- Name: DynOSAM (Dynamic Object Smoothing And Mapping)
- Upstream: https://github.com/ACFR-RPG/DynOSAM
- Default branch: main
- Description: Visual SLAM framework for dynamic environments with ROS2 integration, using GPU-accelerated YOLOv8 for object detection

## Existing AMD support
**None found.** Exhaustive search performed:
- Upstream docs grep (`grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/`): "amd64" references are x86-64 architecture, not AMD GPU
- Web search ("DynOSAM ROCm", "DynOSAM AMD GPU", "DynOSAM HIP"): No results
- GitHub forks scan (`gh api repos/ACFR-RPG/DynOSAM/forks`): Zero forks exist
- Upstream merge policy: Not applicable (no platform forks)

**Decision**: Fresh port required, but BLOCKED by proprietary dependencies.

## Build classification
**Pure CMake project** (Strategy A candidate, but blocked)

Evidence:
- `dynosam_nn/CMakeLists.txt` line 3: `project(dynosam_nn LANGUAGES C CXX CUDA)`
- Uses `find_package(CUDAToolkit)` and `find_package(TensorRT)`
- No pytorch/torch dependency for extension building
- Single `.cu` file: `dynosam_nn/src/YoloV8CudaUtils.cu`

## Port strategy
**BLOCKED -- cannot port**

The project has hard dependencies on NVIDIA-proprietary libraries with no ROCm equivalents:

### 1. TensorRT (CRITICAL BLOCKER)
TensorRT is NVIDIA's proprietary deep learning inference optimizer and runtime. DynOSAM uses it as the core inference engine for:
- YOLOv8 object detection (instance segmentation)
- ONNX model loading and optimization
- GPU-accelerated inference execution

Files using TensorRT:
- `dynosam_nn/src/TrtUtilities.cc`: Engine building, runtime creation
- `dynosam_nn/src/YoloV8ObjectDetector.cc`: Inference execution context
- `dynosam_nn/include/dynosam_nn/TrtUtilities.hpp`: `nvinfer1::ICudaEngine`, `IExecutionContext`, `IRuntime`

**There is NO ROCm/HIP equivalent to TensorRT.** AMD's inference stack (MIGraphX, rocAL) does not provide a drop-in replacement, and rewriting the inference pipeline would require significant architectural changes beyond the scope of a CUDA-to-HIP port.

### 2. OpenCV CUDA modules (SECONDARY BLOCKER)
The project extensively uses OpenCV's CUDA-specific modules:
- `cv::cuda::GpuMat` - GPU matrix operations
- `opencv2/cudawarping.hpp` - `cv::cuda::resize`
- `opencv2/cudaimgproc.hpp` - `cv::cuda::threshold`
- `opencv2/cudaarithm.hpp` - Arithmetic operations
- `opencv2/cudaoptflow.hpp` - Optical flow (sparse LK tracker)

These modules (`opencv_cuda*`) are compiled against NVIDIA CUDA and are NOT available in ROCm-enabled OpenCV builds. OpenCV's ROCm support is limited to OpenCL backends via T-API, which does not expose the same `cv::cuda::*` API.

### 3. Portable CUDA surface (would be trivial IF blockers were resolved)
The actual CUDA kernel code is minimal and straightforward:
- 2 `__global__` kernels (YOLO post-processing and mask combination)
- 1 `__device__` helper (`sigmoidf`)
- Standard runtime API: `cudaMalloc`, `cudaMemcpy*`, `cudaStream*`, `cudaEvent*`

No warp intrinsics, no textures, no cuBLAS/cuFFT/Thrust/CUB usage.

## CUDA surface inventory

### Kernels (2)
| File | Kernel | Description |
|------|--------|-------------|
| YoloV8CudaUtils.cu:23 | `YOLO_PostProcess_Kernel` | Find max confidence, atomic allocation, write detections |
| YoloV8CudaUtils.cu:80 | `YOLO_Mask_Combination_Kernel` | Linear combination + sigmoid for mask generation |

### Device functions (1)
| File | Function | Description |
|------|----------|-------------|
| YoloV8CudaUtils.cu:74 | `sigmoidf` | `__device__ __forceinline__` sigmoid using `__expf` |

### Runtime API usage
| Symbol | HIP Equivalent | Files |
|--------|----------------|-------|
| `cudaMalloc` | `hipMalloc` | TrtUtilities.cc |
| `cudaMallocHost` | `hipHostMalloc` | TrtUtilities.cc |
| `cudaFree` | `hipFree` | (via deleters) |
| `cudaMemcpyAsync` | `hipMemcpyAsync` | TrtUtilities.hpp |
| `cudaStreamCreate` | `hipStreamCreate` | TrtUtilities.cc |
| `cudaStreamDestroy` | `hipStreamDestroy` | TrtUtilities.cc |
| `cudaStreamSynchronize` | `hipStreamSynchronize` | YoloV8CudaUtils.cu, YoloV8ObjectDetector.cc |
| `cudaEventCreate` | `hipEventCreate` | CudaUtils.cc |
| `cudaGetLastError` | `hipGetLastError` | YoloV8CudaUtils.cu |

### Library dependencies (BLOCKERS)
| Library | ROCm Equivalent | Status |
|---------|-----------------|--------|
| TensorRT (`NvInfer.h`, `nvinfer1::*`) | **NONE** | BLOCKING |
| OpenCV CUDA (`cv::cuda::*`) | **NONE** | BLOCKING |

## Risk list
1. **TensorRT (CRITICAL)**: Proprietary NVIDIA library with no AMD equivalent. Would require complete inference backend rewrite (MIGraphX, ONNX Runtime, or Python fallback).
2. **OpenCV CUDA modules**: No ROCm build of `opencv_cuda*` modules. Would require rewrite to use OpenCL T-API or CPU fallback.
3. **Python inference fallback**: Project has Python YOLO bindings (`PyObjectDetector.cc`) that could theoretically work on AMD via PyTorch/ultralytics, but this is a significant performance regression and architectural change, not a port.

## File-by-file change list
**N/A -- port blocked**

If blockers were hypothetically resolved, changes would be limited to:
- Add `cuda_to_hip.h` compat header in `dynosam_nn/include/`
- Modify `dynosam_nn/CMakeLists.txt`: Add USE_HIP option, enable_language(HIP), set_source_files_properties(YoloV8CudaUtils.cu LANGUAGE HIP)
- Update `CudaUtils.hpp`: Include compat header
- Update `TrtUtilities.hpp`: Include compat header (plus complete TensorRT replacement)

## Build commands
**N/A -- port blocked**

## Test plan
**N/A -- port blocked**

The project has extensive CPU-side unit tests (~30 test files in `dynosam/test/`) that test SLAM algorithms, factors, camera models, etc. These do not require GPU. The GPU-dependent tests in `dynosam_cv/test/test_cuda_cache.cc` test OpenCV CUDA GpuMat reference counting.

## Open questions
None -- the blocking status is clear.

## Recommendation
**Skip this project** with disposition `cant-port`.

The project's GPU acceleration is fundamentally built on NVIDIA-proprietary technology (TensorRT) that has no ROCm equivalent. Porting would require:
1. Complete rewrite of the inference backend to use MIGraphX, ONNX Runtime with ROCm EP, or similar
2. Rewrite of OpenCV CUDA module usage to use OpenCL T-API or CPU fallback
3. Significant performance testing and validation

This is architectural redesign, not a CUDA-to-HIP port, and is outside MOAT scope.
