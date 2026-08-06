# OpenCV CUDA modules -> ROCm/HIP port plan

Synthesis of three deep-research passes (module inventory; build/port strategy; proprietary-dependency replacement) plus prior investigation. Written so the port is 100% understood -- every module, every feature, every proprietary dependency -- before any code is touched. No feature is silently scoped out: the feature-coverage ledger below accounts for all of them.

## Scope (TWO upstream repos)

`cv::cuda` spans two repos and the port must touch both:
- **opencv/opencv (core)** -- the GpuMat infrastructure (`modules/core/src/cuda/{gpu_mat,gpu_mat_nd}.cu`), the public CUDA API (`core/include/opencv2/core/cuda.hpp`, `cuda_types.hpp`, `cuda_stream_accessor.hpp`), the device-side **warp-layer headers** (`core/include/opencv2/core/cuda/{warp,warp_shuffle,warp_reduce,emulation,reduce,scan,vec_math,saturate_cast,functional}.hpp`), and ALL the CUDA build logic (`cmake/OpenCVDetectCUDA*.cmake`, `OpenCVModule.cmake`, `OpenCVUtils.cmake`).
- **opencv/opencv_contrib** -- the 12 CUDA modules + the `cudev` device-template module (built via `OPENCV_EXTRA_MODULES_PATH`).

The contrib modules do NOT compile without the core CUDA infra, so Phase 0 = core. Fork BOTH (AMD-Ecosystem/opencv + AMD-Ecosystem/opencv_contrib); expect TWO upstream PRs (core changes to opencv, module changes to opencv_contrib). This MOAT project (`opencv_contrib`) tracks the whole effort; the core work is Phase 0 here. (base_sha/fork for the core repo to be recorded when Phase 0 starts.)

## STRATEGIC CONTEXT -- read before committing effort

- There is **no existing OpenCV cv::cuda ROCm/HIP port** anywhere (verified: no GitHub fork, no OpenCV issue/PR, no AMD project).
- AMD's CV-on-ROCm strategy is **not** an OpenCV port. It is (a) OpenCL/T-API (`cv::UMat`) for OpenCV itself -- AMD started OpenCV's OpenCL module in 2011 and sponsored the Transparent API -- and (b) native libraries: **RPP, rocCV, rocAL, MIVisionX**. So this is an **AMD-owned domain**, analogous to RAPIDS being owned by ROCm-DS [[moat-rapids-owned-by-rocm-ds]] [[moat-opencv-cv-cuda-rocm-gap]].
- Therefore a full cv::cuda->ROCm port is **large AND orthogonal to AMD's chosen direction**. It is a multi-month effort, not a typical one-shot MOAT port. Registered as a candidate at jeff's request; this plan exists so the decision (proceed as a showcase vs defer like RAPIDS) is made with the full scope understood. **Open decision #1 below.**

## Architecture / strategy

Hipify the hand-written kernels + the core/cudev device templates behind a new `WITH_HIP` first-class-language CMake path, with a compat shim for the warp layer, and replace each proprietary dependency per-module. Keep the upstream CUDA build byte-for-byte untouched.

- **Build path:** add `WITH_HIP` + `cmake/OpenCVDetectHIP.cmake` mirroring the existing `OpenCVDetectCUDALanguage.cmake` (`find_package(hip)`, `check_language(HIP)`, `enable_language(HIP)`), and set `HAVE_CUDA=1` so every existing `HAVE_CUDA`-gated path (the `src/cuda/*.cu` glob in `ocv_glob_module_sources`, `ocv_define_module`, link lines) lights up unchanged. In `ocv_add_library`, set `LANGUAGE HIP` on the globbed sources; feed `CMAKE_HIP_ARCHITECTURES` from a `GPU_TARGETS`-style var with native auto-detect (`amdgpu-arch`/`rocm_agent_enumerator`), defaulting to gfx90a;gfx1100;gfx1101;gfx1201. hipcc compiles `.cu` directly (no rename).
- **Regression-guard-friendly:** wrap every CUDA-specific clause (the per-module `CUDA::npp*`/`CUDA::cublas`/`CUDA::cufft` link lines, arch flags) in `if(WITH_CUDA) ... elseif(WITH_HIP) ...` rather than editing the CUDA branches in place. The default CUDA build stays identical.
- **Compat shim** (`cuda_to_hip`-style, force-included): warp-size macro, `laneId`, 64-bit shuffle masks, 64-bit ballot handling, `__CUDA_ARCH__`->`__HIP_DEVICE_COMPILE__` guard mapping.
- **Do NOT use RPP/rocCV/MIVisionX as a cv::cuda backend.** RPP is batch/tensor-oriented (`RpptDescPtr`, N/H/W/C); cv::cuda is single-image GpuMat with bit-exact accuracy-test gates. Routing through RPP would fail OpenCV's own tests and is a rewrite, not a hipify. Precedent: Huawei's Ascend backend added a *separate* `AscendMat` type, not a `cv::cuda` backend swap. A clean hipify that preserves the hand-written kernels is the landable path. (RPP only as a per-primitive NPP substitute where semantics happen to match -- case by case.)

## #1 RISK: the warp layer (wave64 correctness)

The highest-correctness-risk surface, and a MOAT wave64 trap. Concentrated in the core + cudev warp headers; fix once, all 12 modules inherit it. Validate on REAL gfx90a -- do not assume.

- `WARP_SIZE = 1 << 5 = 32` is **hardcoded** in both `core/.../cuda/warp.hpp` and `cudev/.../warp/warp.hpp`. Correct on RDNA followers (wave32) but **WRONG on the gfx90a lead (wave64)** -> warp-cooperative kernels (reductions, scans, ballot) silently produce wrong results on gfx90a. Derive warp size from the target (`#if defined(__GFX9__) 64 #else 32`).
- `laneId()` uses NVIDIA **inline PTX** (`mov.u32 %0,%%laneid`) -- does NOT hipify; replace with `__lane_id()`/thread-index compute.
- Shuffle: the `__shfl* -> __shfl_*_sync(0xFFFFFFFF,...)` redefs use a 32-bit mask; HIP `_sync` requires a **64-bit** mask (`0xFFFFFFFFFFFFFFFFULL`) -- wrong on wave64.
- `__ballot`/`__activemask` return **64-bit** on HIP (32-bit on CUDA); `emulation.hpp`'s ballot tag math (`<< (32-5)`) is wave32-baked.
- `__CUDA_ARCH__ >= 300` (and `CV_CUDEV_ARCH`) guards: under HIP `__CUDA_ARCH__` is undefined -> these collapse to the `else` no-op fallback, so shuffle-based reductions **compile but do nothing** unless taught about `__HIP_DEVICE_COMPILE__`. Correctness trap (cf. MOAT's `__CUDA_ARCH__` body-less-kernel and missing-return-UB lessons).
- ROCm 7.0 made `warpSize` **non-constexpr** -- any enum/array-bound/template-arg use must move to a macro/arch-conditional constant.
- **Validation rule:** wave64 (gfx90a) vs wave32 (RDNA) -- warp-layer kernels must be validated on EACH arch independently. NO carry-forward / binary-equivalence across the wave-size boundary for warp-layer-touching modules; treat them as full revalidation per arch.

Textures are a near-non-issue: cudev uses the modern texture-object API (`cudaTextureObject_t` -> `hipTextureObject_t`, maps 1:1). `PtrStepSz`/ptr2d/grid templates hipify cleanly. ~90%+ of kernel/template lines hipify mechanically; the warp layer is the localized manual+risk surface.

## Module inventory (all 12; features abridged -- full API per module header)

| Module | Feature area (complete API in the module header) | External deps | Class |
|---|---|---|---|
| cudev | header-only device templates (warp/block/grid primitives, ptr2d accessors, expr templates) -- the foundation | none | Foundation |
| cudaarithm | element ops, compare/logic, threshold/inRange, magnitude/phase/polar, merge/split/transpose/flip/copyMakeBorder, LUT, norms/sum/minMax/reduce/meanStdDev/normalize/integral, gemm, dft/DFT/Convolution, mulSpectrums | NPP + cuBLAS(gemm) + cuFFT(dft) | NPP + BLAS/FFT |
| cudafilters | box/linear/laplacian/sep-linear/deriv/sobel/scharr/gaussian/morphology/boxMax/boxMin/rowSum/colSum/median filters | NPP (filterbox/dilate/erode/max/min/sumwindow) + CUB(median) | NPP-heavy |
| cudaimgproc | cvtColor/demosaicing/gamma/alphaComp/swapChannels, calcHist/equalizeHist/CLAHE/histEven/histRange, Canny, Hough lines/segments/circles + generalized Hough, Harris/minEigen/goodFeatures, meanShift*, templateMatching, bilateral, blendLinear, connectedComponents, moments | NPP (color/alpha/gamma/histogram) + Thrust | NPP + custom |
| cudawarping | remap, resize, warpAffine, warpPerspective, buildWarp*Maps, rotate, pyrDown, pyrUp | NPP (warpAffine/Perspective/rotate) | NPP-light |
| cudabgsegm | BackgroundSubtractorMOG, MOG2 | none (custom) | PURE custom |
| cudastereo | StereoBM, StereoBeliefPropagation, StereoConstantSpaceBP, StereoSGM, DisparityBilateralFilter, reprojectImageTo3D, drawColorDisp | none (custom) | PURE custom |
| cudafeatures2d | DescriptorMatcher(BF), Feature2DAsync, FastFeatureDetector, ORB | Thrust | PURE custom |
| cudaobjdetect | HOG, CascadeClassifier (Haar via cudalegacy NCV) | none direct (NCV transitive) | PURE custom |
| cudalegacy | ImagePyramid, BG sub GMG/FGD, optical flow BM + interpolateFrames + needleMap + NCV Brox, **graphcut**, connectivityMask/labelComponents, transformPoints/projectPoints/solvePnPRansac, NCV framework + NPP_staging(OpenCV's own kernels) | NPP (only graphcut is real vendor NPP) + Thrust(fgd) | mostly custom; graphcut is the NPP blocker |
| cudaoptflow | Brox/SparsePyrLK/DensePyrLK/Farneback/TVL1 (portable) + NvidiaOpticalFlow_1_0/2_0 (HW) | NVIDIA Optical Flow SDK (only the 2 HW classes) | custom + OF SDK |
| cudacodec | VideoReader (decode), VideoWriter (encode), NVSurfaceToColorConverter, format/codec enums | NVIDIA Video Codec SDK (NVCUVID + NVENC) + cuda driver | SDK-blocked (only 2 custom kernels) |

## Proprietary-dependency replacement plans

(Standard deps already solved: cuBLAS->hipBLAS, cuFFT->hipFFT, cuRAND/cuSOLVER/cuSPARSE->hip*, Thrust/CUB->rocThrust/hipCUB. cuRAND/cuSOLVER/cuSPARSE/NVML/NVTX/NVRTC/NvJPEG confirmed NOT used by any cuda* module.)

### NPP -> RPP + custom HIP kernels  (HARDEST by surface; modules: cudaarithm, cudafilters, cudaimgproc, cudawarping, cudalegacy, +cudaobjdetect transitive)
~13 functional families (~330 typed symbols). IMPORTANT: the `nppiSt*` "NPP staging" in cudalegacy is **OpenCV's OWN CUDA kernels** (NCV framework, returns NCVStatus) -- just hipify, NOT a vendor dep. Real vendor `nppi*` coverage vs RPP (`rppt_*`):
- Covered by RPP (with a GpuMat->RpptDesc adapter): warpAffine/Perspective/rotate, flip(mirror), transpose, filterBox, dilate/erode (square SE only), magnitude, bitwise and/or/xor, gamma, swapChannels, mean/stddev, threshold.
- **MISSING from RPP -> write direct HIP kernels:** FilterMax/Min (rank), SumWindowRow/Column, HistogramEven/Range, AlphaComp/AlphaPremul (Porter-Duff), LShiftC/RShiftC, RectStdDev.
- **graphcut/graphcut8 (maxflow): NO ROCm analog at all** (NVIDIA removed graphcut from NPP at 8.0). Port a GPU push-relabel/maxflow kernel OR scope out (it is one stereo path).
- PLAN: do NOT build a generic NPP->RPP shim (data-model mismatch: per-image vs batched-tensor; bit-exact accuracy tests). Recommended pass 1: **reimplement the NPP entry points as direct HIP kernels** (OpenCV often already has a parallel hand-written CUDA path; the missing families are modest self-contained kernels). Adopt RPP later only if profiling demands. Effort High; risk Medium-High (numeric/border/rounding parity is the main correctness risk).

### NVIDIA Video Codec SDK -> rocDecode (decode) + AMF/FFmpeg (encode)  (module: cudacodec)
- **Decode (VideoReader) -> rocDecode**: near 1:1 API mirror (`cuvidCreateDecoder`->`rocDecCreateDecoder`, `CUVIDDECODECREATEINFO`->`RocDecoderCreateInfo`, parser flow identical). Mechanical shim. Codec loss: rocDecode lacks MPEG-1/2, VC-1, VP8, MJPEG (use rocJPEG for JPEG). Prefer the existing `ffmpeg_video_source` demuxer; only the HW-decode call swaps. Effort Medium, risk Low-Medium.
- **Encode (VideoWriter) -> hard**: no ROCm-native encoder. AMF is a Vulkan/DX stack outside ROCm, needs HIP<->Vulkan external-memory interop. RECOMMENDED pass 1: gate HW-encode off on ROCm and route encode via OpenCV's FFmpeg videoio backend with FFmpeg's `h264_amf`/`hevc_amf`/`av1_amf` encoders (preserves the GPU-encode feature without an in-tree interop bridge). Full native AMF = high effort, new dep. Add `WITH_ROCDECODE`/`HAVE_ROCDECODE` gates mirroring `WITH_NVCUVID`.

### NVIDIA Optical Flow SDK -> NONE (unavoidable HW feature loss)  (module: cudaoptflow)
AMD has no dedicated hardware optical-flow engine and AMF exposes no optical-flow API. Only the `NvidiaOpticalFlow_1_0/2_0` classes are affected; Brox/Farneback/PyrLK/TVL1 are pure CUDA and port normally. PLAN: `#if`-gate `nvidiaOpticalFlow.cpp` + its factory; keep the public symbols but have the factory throw `cv::Error::StsNotImplemented` ("hardware optical flow not available on ROCm") -- explicit error, NOT a silent drop. Register the deferral in `data/deferred.json`. The software flows ARE the substitute and stay working. Effort Low; this is a genuine, documented feature loss that must be scoped out of any PR claim.

### cuDNN -> MIOpen (OUT OF SCOPE for cuda* contrib)
Verified: zero cuDNN use in any cuda* module. cuDNN is the `opencv/opencv` **dnn module's CUDA backend** (`modules/dnn/src/cuda4dnn/csl/cudnn/`). A separate, sizable future project (add a `cuda4dnn/csl/miopen/` backend; MIOpen is "AMD's cuDNN", with Nd-tensor-API gaps to bridge). NOTED so it is not forgotten; not part of this scope unless jeff expands it.

## FEATURE-COVERAGE LEDGER (nothing silently skipped)

- **Ported by hipify (no proprietary dep):** cudev, cudabgsegm, cudastereo (except graph-cut path), cudafeatures2d, cudaobjdetect (HOG + Haar/NCV), cudalegacy (NCV/staging/BM/interp/Brox/labeling/PnP), the algorithmic optical flows, and all non-NPP kernels in cudaarithm/cudafilters/cudaimgproc/cudawarping.
- **Replaced (AMD lib or custom HIP kernel):** NPP families -> RPP-or-custom-kernel; cuBLAS/cuFFT -> hipBLAS/hipFFT; video DECODE -> rocDecode; video ENCODE -> FFmpeg-AMF (or native AMF).
- **Genuine feature losses -- must be explicit in any PR claim, never passed off as complete:**
  1. `cudalegacy::graphcut` / graph-cut stereo -- no ROCm maxflow analog (NPP-removed). Port a kernel or scope out.
  2. `cudaoptflow::NvidiaOpticalFlow_1_0/2_0` -- no AMD HW optical-flow engine. Scope out (StsNotImplemented); software flows remain.
  3. `cudacodec` codec deltas -- rocDecode lacks MPEG-1/2, VC-1, VP8, MJPEG vs NVDEC; native HW encode pending AMF interop.

## Phased port order

0. **opencv core** -- GpuMat infra + the warp-layer headers (fix WARP_SIZE/laneId/shuffle-mask/ballot/`__CUDA_ARCH__`). Validate on gfx90a (wave64 bites first here). Gates everything.
1. **cudev** -- header-only templates; pure hipify + the Phase-0 shim.
2. **Pure-custom modules** (no replacement work): cudabgsegm, cudafeatures2d, cudaobjdetect, cudastereo, cudalegacy. Cleanest early wins + warp-layer stress (stereo/features2d exercise reductions/ballot heavily on gfx90a).
3. **NPP-light:** cudawarping (5 kernels, 2 NPP libs), cudafilters (38 kernels, 2 NPP libs).
4. **NPP/cuBLAS/cuFFT-heavy:** cudaimgproc, then cudaarithm (the heaviest dep surface).
5. **SDK-blocked (last, may stay `blocked`):** cudaoptflow (gate the HW class), cudacodec (rocDecode decode; encode via FFmpeg-AMF or blocked). A documented `blocked` here does not block the upstream PR if the claim is scoped.

## Validation

OpenCV's own GPU accuracy tests (bit-exact vs CPU reference) are the bar -- not just compile. Per-arch wave64/wave32 validation for any warp-cooperative kernel. Real-GPU runs required.

## OPEN DECISIONS for jeff

1. **Proceed vs defer?** This is AMD-owned domain (like RAPIDS) + multi-month. Showcase port, or defer? (Registered + fully scoped either way.)
2. **NPP:** hand-written HIP kernels (recommended pass 1) vs RPP backend.
3. **Video encode:** FFmpeg-AMF delegation (recommended) vs native AMF+Vulkan interop vs gate-off.
4. **graphcut:** port a maxflow kernel vs scope out.
5. **Scope:** cuda* contrib only, or also the opencv dnn CUDA backend (cuDNN->MIOpen)?
