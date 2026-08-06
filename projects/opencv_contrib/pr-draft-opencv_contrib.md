## Title

[ROCm] Port the cv::cuda modules to AMD GPUs via HIP

## Body

This ports all of the CUDA-accelerated cv::cuda modules in opencv_contrib to AMD GPUs through HIP, AMD's CUDA-compatible kernel language and runtime, so the same cv::cuda public API runs on AMD hardware. It is the modules half of a two-repository change and requires the companion core PR (opencv/opencv, "Add AMD GPU support for cv::cuda via HIP (core)"), which adds the `WITH_HIP` build system, the CUDA-to-HIP compatibility header, and the warp/saturate infrastructure. Build the core PR first, then point its `OPENCV_EXTRA_MODULES_PATH` at these modules.

Motivation: the cv::cuda modules are NVIDIA-only today. ROCm provides a CUDA-compatible programming model (HIP) and the roc*/hip* analogues of the CUDA libraries (hipBLAS for cuBLAS, hipFFT for cuFFT, rocDecode for NVDEC), so the existing CUDA module sources can target AMD GPUs unchanged in spelling. Every change here is additive and guarded (`__HIP_PLATFORM_AMD__` / `__HIP_DEVICE_COMPILE__` / `HAVE_HIP`); the NVIDIA code path is preserved verbatim in the `#else` branch of each guard, so the CUDA build is byte-for-byte unchanged.

### Build

Configure the core build with `-DWITH_HIP=ON` (instead of `-DWITH_CUDA=ON`) and, optionally, `-DCMAKE_HIP_ARCHITECTURES=<gfx targets>`; the ROCm toolkit must be installed.

```
cmake -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules <opencv_source>
cmake --build . -j
```

The modules' device code largely compiles unchanged through the core compatibility header. The module-specific work falls into a few classes: the few NPP entry points each route to an existing OpenCV kernel where one exists or to a faithful direct HIP kernel matching the NPP semantics (flip/mirror, per-channel bitwise/shift constants, mean/std-dev, rectStdDev, morphology and rank filters, box and sliding-window sums); the cuBLAS/cuFFT calls in cudaarithm map to hipBLAS/hipFFT; Thrust execution policies switch to rocThrust; and a 32-lane-warp wavelet-matrix median is replaced on the HIP build by a wave-size-agnostic counting-rank median (the wavelet path stays CUDA-only). The warp-cooperative reductions in the matchers, ORB, stereo, and HOG already pass an explicit width of 32 and are correct on both wave32 and wave64.

### Validation

Built with ROCm 7.2.1 and exercised on an AMD Instinct MI250X (gfx90a, wave64). Per-module GPU test results on gfx90a:

| Module | Result |
|---|---|
| cudev | 402/402 |
| cudaarithm | 11417/11417 |
| cudafilters | 16028/16028 |
| cudaimgproc | 3788/3788 |
| cudawarping | 4535/4535 |
| cudastereo | 128/128 |
| cudafeatures2d | 256/256 |
| cudaobjdetect | 18/18 |
| cudalegacy | 14/14 |
| cudacodec (YUV converter) | 240/240 |
| cudabgsegm | builds (MOG/MOG2 accuracy tests are gated behind HAVE_VIDEO_INPUT, undefined in this tree on CUDA as well) |
| cudaoptflow | 41/47 (the 6 remaining are the scoped-out cases below) |

The same suites were also validated on two RDNA (wave32) architectures, exercising the warp layer on a different wave size than gfx90a: gfx1100 (Radeon Pro W7800, RDNA3) and gfx1201 (Radeon RX 9070 XT, RDNA4). Those parts carry a video engine, so on them the rocDecode VideoReader hardware-decode path (H.264/H.265/AV1) is exercised by opencv_test_cudacodec; the MPEG-1/2, VC-1, VP8, and MJPEG streams in that suite are outside rocDecode's codec coverage and report not-implemented (see Scope and limitations).

### Scope and limitations

The following NVIDIA-proprietary facilities have no ROCm analog; on the HIP build they report `cv::Error::StsNotImplemented` rather than silently misbehaving, and the corresponding software algorithms remain available:

- NVIDIA hardware optical flow (NvidiaOpticalFlow_1_0/2_0): no AMD hardware optical-flow engine. The PyrLK, Brox, Farneback, and TV-L1 software flows are unaffected and pass. The 4 NvidiaOpticalFlow test cases instantiate the unavailable hardware class and therefore raise the not-implemented error, the same outcome as a CUDA build configured without the NVIDIA Optical Flow SDK.
- cudalegacy graph-cut stereo: no ROCm maxflow library (NPP dropped graphcut at CUDA 8.0, so this entry point is already a no-op on a current CUDA build).
- Hardware video decode/encode in cudacodec: rocDecode covers H.264/H.265/AV1 decode but not MPEG-1/2, VC-1, VP8, or MJPEG; there is no ROCm-native hardware encoder, so encode is delegated to the FFmpeg AMF encoders via the videoio backend. On compute-only parts without a video engine (such as gfx90a/MI250X) hardware decode is reported as not implemented; the portable YUV-to-RGB surface converter is still available and validated.

Two cudaoptflow cases (OpticalFlowDual_TVL1.Async) compare a default-stream result against multi-stream results at exact (0.0) tolerance. The TV-L1 convergence loop exits on a sum computed by a floating-point atomic reduction, whose accumulation order depends on block-completion order and therefore differs between the default-stream and explicit-stream schedulers; the resulting last-ULP difference shifts the iteration count by one and produces a small flow delta at a single pixel. The flow is correct (the TV-L1 accuracy test passes at its normal tolerance), and each stream context is internally deterministic; the exact-tolerance cross-stream comparison is unsound for a non-associative atomic reduction independently of this change.

A few NPP-replacement kernels (cudawarping rotate, cudaarithm rectStdDev, cudafilters box max/min and row/column sums, cudaimgproc gamma and alpha-premultiply) have no accuracy test in their module upstream, so they build and dispatch but are not exercised by the test suites; their math is given in the source.

This work was authored with the assistance of Claude, an AI assistant by Anthropic.

### Test Plan

```
# build core with WITH_HIP and these modules (see the companion opencv core PR), then on a real AMD GPU (gfx90a, ROCm 7.2.1):
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
OPENCV_TEST_DATA_PATH=<opencv_extra>/testdata ./bin/opencv_test_cudaoptflow    # 41/47 (6 scoped-out per the PR description)
```
