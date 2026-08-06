## Title

[ROCm] Add AMD GPU support for cv::cuda via HIP (core)

## Body

This adds the OpenCV core build-system and infrastructure changes needed to compile the CUDA-accelerated modules for AMD GPUs through HIP, AMD's CUDA-compatible kernel language and runtime. It is the core half of a two-repository change; the cv::cuda modules themselves live in opencv_contrib and are ported in a companion PR (opencv/opencv_contrib), which depends on this one.

Motivation: the cv::cuda modules are NVIDIA-only today. ROCm provides a CUDA-compatible programming model (HIP) and the roc*/hip* analogues of the CUDA libraries (hipBLAS for cuBLAS, hipFFT for cuFFT, rocDecode for NVDEC), so the existing CUDA sources can target AMD GPUs with the same cv::cuda public API and no NVIDIA hardware. This change keeps the CUDA build byte-for-byte unchanged: every edit is additive and guarded (`__HIP_PLATFORM_AMD__` / `HAVE_HIP` / a CMake `NOT HAVE_HIP`), and the NVIDIA code path is preserved verbatim in the `#else` branch of each guard.

### Build

A new `WITH_HIP` CMake option (default OFF) enables `enable_language(HIP)` and `cmake/OpenCVDetectHIP.cmake`, which finds the ROCm toolkit, sets `HAVE_HIP`, and auto-detects `CMAKE_HIP_ARCHITECTURES` from the GPUs present (defaulting to a common set when none are detected). The CUDA module sources are compiled with the ROCm HIP compiler via per-source `LANGUAGE HIP`; modules link `hip::host` (not `hip::device`, which would inject device flags into plain host `.cpp`). `WITH_HIP` and `WITH_CUDA` are mutually exclusive.

```
cmake -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules <opencv_source>
cmake --build . -j
```

A single force-included compatibility header (`modules/core/include/opencv2/core/cuda/cuda_to_hip.h`) maps the CUDA runtime, texture, constant-memory, and function-attribute spellings used by the modules onto their HIP equivalents, so the bulk of the device code compiles unchanged. The warp layer (the one wave-size-sensitive surface) keeps a logical warp width of 32 and pins shuffle widths to 32 lanes, which is correct on both wave32 and wave64 hardware; `cv::cuda::saturate_cast`, FP16 conversion, and the SIMD-in-a-word helpers gain HIP implementations that match the NVIDIA PTX semantics (saturating, round-to-nearest) bit-for-bit. The cuBLAS/cuFFT call sites in cudaarithm are aliased to hipBLAS/hipFFT through a host shim, and `WITH_ROCDECODE` wires in the rocDecode video-decode backend.

### Validation

Built with ROCm 7.2.1 and exercised on an AMD Instinct MI250X (gfx90a, wave64), together with the companion opencv_contrib modules. The core GpuMat/warp/saturate infrastructure is covered transitively by every cv::cuda module test suite; with the companion PR applied the per-module suites pass as follows on gfx90a: cudev 402/402, cudaarithm 11417/11417, cudafilters 16028/16028, cudaimgproc 3788/3788, cudawarping 4535/4535, cudastereo 128/128, cudafeatures2d 256/256, cudaobjdetect 18/18, cudalegacy 14/14, cudacodec (YUV converter) 240/240. The same suites were validated on two RDNA (wave32) architectures as well, gfx1100 (Radeon Pro W7800, RDNA3) and gfx1201 (Radeon RX 9070 XT, RDNA4), which exercises the warp layer on both wave64 and wave32; on those parts (which carry a video engine) the rocDecode hardware-decode path is additionally exercised.

### Scope and limitations

The following NVIDIA-proprietary facilities have no ROCm analog; on the HIP build they report `cv::Error::StsNotImplemented` rather than silently misbehaving, and the corresponding software algorithms remain available:

- NVIDIA hardware optical flow (NvidiaOpticalFlow_1_0/2_0): no AMD hardware optical-flow engine. The PyrLK, Brox, Farneback, and TV-L1 software flows are unaffected.
- cudalegacy graph-cut stereo: no ROCm maxflow library (NPP dropped graphcut at CUDA 8.0, so this entry point is already a no-op on a current CUDA build).
- Hardware video decode/encode: rocDecode covers H.264/H.265/AV1 decode but not MPEG-1/2, VC-1, VP8, or MJPEG; there is no ROCm-native hardware encoder, so encode is delegated to the FFmpeg AMF encoders via the videoio backend. On compute-only parts without a video engine (such as gfx90a/MI250X) hardware decode is reported as not implemented and the portable YUV-to-RGB surface converter is still available.

Two cudaarithm-dependent cases (OpticalFlowDual_TVL1.Async) compare a default-stream result against multi-stream results at exact (0.0) tolerance. The TV-L1 convergence loop exits on a sum computed by a floating-point atomic reduction, whose accumulation order depends on block-completion order and therefore differs between the default-stream and explicit-stream schedulers; the resulting last-ULP difference shifts the iteration count by one and produces a small flow delta at a single pixel. The flow is correct (the TV-L1 accuracy test passes at its normal tolerance), and each stream context is internally deterministic; the exact-tolerance cross-stream comparison is unsound for a non-associative atomic reduction independently of this change.

This work was authored with the assistance of Claude, an AI assistant by Anthropic.

### Test Plan

```
cmake -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++ \
  -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules \
  -DBUILD_LIST="core,cudev,cudaarithm,cudafilters,cudaimgproc,cudawarping,cudastereo,cudafeatures2d,cudaobjdetect,cudalegacy,cudaoptflow,cudacodec,imgproc,video,optflow,objdetect,calib3d,ts" \
  -DBUILD_TESTS=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF <opencv_source>
cmake --build . -j

# real AMD GPU (gfx90a, ROCm 7.2.1):
OPENCV_TEST_DATA_PATH=<opencv_extra>/testdata ./bin/opencv_test_cudev          # 402/402
OPENCV_TEST_DATA_PATH=<opencv_extra>/testdata ./bin/opencv_test_cudaarithm     # 11417/11417
OPENCV_TEST_DATA_PATH=<opencv_extra>/testdata ./bin/opencv_test_cudafilters    # 16028/16028
OPENCV_TEST_DATA_PATH=<opencv_extra>/testdata ./bin/opencv_test_cudaimgproc    # 3788/3788
OPENCV_TEST_DATA_PATH=<opencv_extra>/testdata ./bin/opencv_test_cudawarping    # 4535/4535
OPENCV_TEST_DATA_PATH=<opencv_extra>/testdata ./bin/opencv_test_cudastereo     # 128/128
OPENCV_TEST_DATA_PATH=<opencv_extra>/testdata ./bin/opencv_test_cudafeatures2d # 256/256
OPENCV_TEST_DATA_PATH=<opencv_extra>/testdata ./bin/opencv_test_cudaobjdetect  # 18/18
OPENCV_TEST_DATA_PATH=<opencv_extra>/testdata ./bin/opencv_test_cudalegacy     # 14/14
OPENCV_TEST_DATA_PATH=<opencv_extra>/testdata ./bin/opencv_test_cudacodec      # 240/240 (YUV converter)
```
