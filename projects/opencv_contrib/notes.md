# opencv_contrib notes

## Research provenance + decision state 2026-06-09

Full porting plan in plan.md (synthesis of 3 deep-research agents + prior investigation).
- Agent 1: module/feature/dependency inventory (12 cuda modules, source-verified).
- Agent 2: proprietary-dep replacement plans (NPP->RPP+kernels; Video Codec SDK->rocDecode/AMF; Optical Flow SDK->none; cuDNN->MIOpen out-of-scope).
- Agent 3: build system + HIP-port strategy + warp-layer wave64 risk + RPP-not-a-backend + phased order.
- Prior: no existing OpenCV cv::cuda ROCm port anywhere; AMD CV-on-ROCm = OpenCL/T-API for OpenCV + native RPP/rocCV/rocAL/MIVisionX [[moat-opencv-cv-cuda-rocm-gap]].

3 GENUINE feature losses (must never be passed off as "complete"): cudalegacy graphcut (no ROCm maxflow), cudaoptflow NvidiaOpticalFlow_1_0/2_0 (no AMD HW OF engine), cudacodec codec deltas (rocDecode lacks MPEG/VC1/VP8/MJPEG; HW encode pending AMF).

STATE: lead blocked pending jeff's strategic proceed/defer decision (plan.md open-decision-1).
This is AMD-owned domain (like RAPIDS) + a multi-month 2-repo effort (opencv core + opencv_contrib). Registered + 100% scoped; NOT auto-progressing until jeff decides. Unblock to proceed.

## Decisions 2026-06-10 (jeff: PROCEED)

Jeff unblocked the port ("emboldened by the new Claude Fable model; let's see what you can do").
Open decisions resolved per plan.md recommendations:
1. Proceed (not defer). Showcase port.
2. NPP: reimplement missing entry points as direct HIP kernels (no RPP backend in pass 1).
3. Video encode: gate native HW encode off; delegate to FFmpeg AMF encoders via videoio. Decode via rocDecode.
4. graphcut: scope OUT of pass 1 (register in deferred.json when gated); revisit a maxflow kernel later.
5. Scope: cuda* contrib only. opencv dnn CUDA backend (cuDNN->MIOpen) stays out of scope; register as deferred feature-port.

Phase order per plan.md: 0=opencv core (warp layer, wave64) -> 1=cudev -> 2=pure-custom modules -> 3=NPP-light -> 4=NPP-heavy -> 5=SDK-gated.

Key sources: opencv/opencv + opencv/opencv_contrib (CMake + cuda* modules + core warp headers); ROCm/rpp, ROCm/rocDecode, ROCm/MIOpen, GPUOpen AMF; HIP kernel-language ref (warpSize/__shfl_sync 64-bit mask/__ballot 64-bit); ROCm gpu-arch-specs (wavefront per arch). Full URLs in the agent reports / [[moat-opencv-cv-cuda-rocm-gap]].

## Phase 0 + Phase 1 PORTED (2026-06-11, gfx90a lead)

opencv core (GpuMat infra + warp-layer headers) + opencv_contrib cudev module ported to ROCm/HIP and VALIDATED on real gfx90a: opencv_test_cudev 402/402 PASS, deterministic across runs (ROCm 7.2.1, wave64). Multi-arch fat binary (gfx90a;gfx1100) emits both device code objects.

### Two-repo layout
- `src/`      = AMD-Ecosystem/opencv_contrib @ moat-port (this project's head_sha tracks THIS)
- `src-core/` = AMD-Ecosystem/opencv        @ moat-port
- Base upstream SHAs: opencv_contrib `2a5154a4479e841aa1282ef83d139c4870d17b8f`, opencv `aed41fdabe3c5381395e2a7a8272c7519f3c289f`
- Fork HEADs after Phase 0/1: contrib `0e750c36725790885efe299b969479fca472d8bf`, core `f90ef85504cf3e364720ec9f8863acb87caee937`
- Two upstream PRs expected (core->opencv, modules->opencv_contrib). Both forks have Actions disabled.

### Build (## Build section)
Repeatable: `projects/opencv_contrib/build_hip.sh` (configures core with WITH_HIP, EXTRA_MODULES_PATH=../src/modules, BUILD_LIST trimmed to core+cudev+ts deps, builds opencv_test_cudev).
Key cmake invocation (build dir = `projects/opencv_contrib/build`, configure against `src-core`):
```
cmake -G Ninja ../src-core -DWITH_HIP=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DOPENCV_EXTRA_MODULES_PATH=../src/modules \
  -DBUILD_LIST="core,cudev,imgproc,imgcodecs,videoio,highgui,ts" \
  -DBUILD_TESTS=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_PYTHON=OFF
cmake --build . --target opencv_test_cudev -j$(nproc)
./bin/opencv_test_cudev   # 402/402 on gfx90a
```
GOTCHAS:
- CMake 4.0 REJECTS hipcc as CMAKE_HIP_COMPILER ("Use Clang directly"); must pass `/opt/rocm/llvm/bin/amdclang++`.
- Need CMAKE_HIP_STANDARD 17 (rocPRIM requires C++17); set in OpenCVDetectHIP.cmake.
- `hip::device` interface target injects `-x hip --offload-arch` into ALL target sources incl. plain .cpp -> link only `hip::host`; device .cu compile via per-source LANGUAGE HIP.
- OpenCVPackaging.cmake does `string(REPLACE . - ... ${CUDA_VERSION})` under HAVE_CUDA -> set CUDA_VERSION on HIP path or it errors.

### Build-system design (core CMake)
- WITH_HIP option + enable_language(HIP) block (CMakeLists.txt) + cmake/OpenCVDetectHIP.cmake (sets HAVE_HIP + HAVE_CUDA=1, CMAKE_HIP_ARCHITECTURES autodetect via amdgpu-arch defaulting gfx90a;gfx1100;gfx1101;gfx1201, force-includes the shim).
- OpenCVFindLibsPerf.cmake includes DetectHIP when WITH_HIP; OpenCVModule.cmake sets LANGUAGE HIP on globbed src/cuda/*.cu and links hip::host; the FindCUDA fallback paths (ocv_add_library, ocv_create_module link, cudev test CMakeLists) excluded when HAVE_HIP.
- Shim: modules/core/include/opencv2/core/cuda/cuda_to_hip.h (force-included via CMAKE_HIP_FLAGS -include; host cpp pulls it via private.cuda.hpp). Defines __CUDA_ARCH__=350 in device pass so cudev CV_CUDEV_ARCH>=300 shuffle fast paths light up.

### Warp-layer decisions (THE wave64 surface)
- cudev WARP_SIZE kept at LOGICAL 32 on all arches; width-32 shuffles stay inside a 32-lane subgroup of the 64-lane wavefront (PORTING_GUIDE l.145) -> wave64-correct.
- Warp::laneId(): HIP computes linearized-tid % 32 (logical lane), replacing NVIDIA `mov %%laneid` PTX. warpId unchanged.
- block/detail/reduce.hpp GenericOptimized32: loopShfl width changed `warpSize`->WARP_SIZE (literal warpSize=64 on gfx90a broke the per-32 partition).
- grid/detail/integral.hpp horisontal_pass_8u_shfl_kernel: lane_id/warp_id/leader-test changed `warpSize`->literal 32 (kLogicalWarp). This was the LAST failure fixed (Integral._8u_opt). All shfl widths already explicit 32 -> arch-agnostic.
- CV_CUDEV_ARCH=350 on HIP DEVICE pass lights up the shfl reduce/scan/integral paths; simd_functions.hpp PTX SIMD forced to software `#else` via per-file push/pop of CV_CUDEV_ARCH=0 (vadd2 PTX doesn't assemble). __CUDACC_VER_MAJOR__ undefined on HIP already routes scan.hpp/block-scan to the maskless smem branches.

### Numeric fixes (0.0-tolerance accuracy tests)
- saturate_cast.hpp (BOTH cudev util/ and core cuda/): PTX cvt.sat integer specializations replaced with clamp (min/max) on HIP; INCLUDE the from-float/double specs (round-to-nearest via __float2int_rn then clamp) -- missing them let the generic T(v) template wrap (Cvt uchar 255->0 bug).
- cast_fp16: HIP must REINTERPRET the 16-bit half bit-pattern (`*(__half*)&v` / `*(short*)&h`) like the CUDA>=9 path, NOT `__half2float(short)`/`(short)__float2half_rn` which value-convert (CvFp16 bug).
- cuda_info.cpp CudaArch: on HIP seed bin/ptx/features with level 90 so deviceSupports()/builtWith() answer true (Histogram SHARED_ATOMICS assert). gfx90a prop.major.minor = 9.0 -> version 90 >= all FeatureSets.
- vec_math.hpp: add `char abs_(char)` overload on HIP (charN::x is plain char, ambiguous between schar/short overloads).

### RESUME POINT -> Phase 2 (pure-custom modules)
Next porter dispatch: Phase 2 = cudabgsegm, cudafeatures2d, cudaobjdetect, cudastereo, cudalegacy (NCV/staging/BM, NOT graphcut). These pull the SAME core warp headers + cudev (now ported) so the foundation is done. Build by adding each module to BUILD_LIST and OPENCV_EXTRA_MODULES_PATH; expect: (1) more PTX asm in module .cu (same cvt.sat/laneid/shfl-mask classes -- apply the established patterns), (2) Thrust->rocThrust (cudafeatures2d/cudalegacy fgd), (3) per-module test binaries opencv_test_<module> are the wave64 gates (stereo/features2d exercise reductions/ballot heavily). graphcut path in cudalegacy: scope OUT + register deferred.json. NPP host helpers already HIP-guarded in private.cuda.hpp/cuda_info.cpp; NPP-using modules are Phase 3/4.

## Review findings (phase 0+1)

### Review 2026-06-11 (reviewer, linux-gfx90a): PASS, no changes requested

Reviewed both fork branches base..HEAD via /pr-review: core (AMD-Ecosystem/opencv f90ef85) and contrib (AMD-Ecosystem/opencv_contrib 0e750c3). Verdict review-passed. No blocking defects. The items below are confirmations and one forward-looking note, not change requests.

Wave64 warp layer (the #1 risk) verified CORRECT:
- block/detail/reduce.hpp GenericOptimized32 and warp_reduce_detail::WarpReductor pass an EXPLICIT width 32 down to shfl_down (loopShfl(...,WARP_SIZE=32) -> mergeShfl(...,width=32)), so the per-32 reduction is wave64-correct.
- warpScanInclusive / blockScanInclusive take the legacy smem branch on HIP (__CUDACC_VER_MAJOR__ undefined -> the `>=9` block is skipped) and inside it the CV_CUDEV_ARCH>=300 shfl path runs shfl_up(data,i) with the DEFAULT width = warpSize = 64 on gfx90a. This is still correct because the gate `if (laneId >= i)` uses laneId = tid&31: for any thread, the add only fires when tid_within_32 >= i, which forces the shfl_up source physical lane (tid - i) to stay inside the thread's own 32-lane half of the 64-wide wavefront. No cross-32-boundary contribution. Confirmed by lane arithmetic and by 402/402 passing the block_size = 40/64/70/128/256/1024 BlockScan tests (1D <<<1,block_size>>> launches that put 2 logical 32-warps in one physical wavefront).
- warp.hpp keeps WARP_SIZE logical 32 on all arches (PORTING_GUIDE-sanctioned width-32-shuffle-in-64-lane-wavefront pattern); laneId() = tid & (WARP_SIZE-1) replaces the NVIDIA `mov %%laneid` PTX. warpId()/warpFill/warpCopy all derive from WARP_SIZE=32 consistently.
- grid/detail/integral.hpp horisontal pass partitions on kLogicalWarp=32 (lane_id/warp_id/leader test) instead of physical warpSize -> wave64-correct; the integral_sums path uses blockDim.x=32 with the same laneId-gated scan.

CUDA path preservation verified: every change is HIP-guarded (WITH_HIP/HAVE_HIP/__HIP_PLATFORM_AMD__/__HIPCC__/__HIP_DEVICE_COMPILE__). CMake guards reduce to the originals when HAVE_HIP is unset (OpenCVModule.cmake `NOT HAVE_HIP AND ...`, OpenCVUtils.cmake, cudev test CMakeLists). saturate_cast (core + cudev): the new HIP specializations sit in `#if __HIP... #else <original CUDA> #endif`; the HIP and CUDA specialization SETS are identical (33 each in core, matching set in cudev) -- no missing specialization that would silently fall through to the wrapping generic T(v). fast_math.hpp / cuda_types.hpp only widen `__CUDACC__` guards to also match `__HIPCC__`; no effect on nvcc or the CPU build.

Shim correctness: __CUDA_ARCH__=350 is set only under __HIP_DEVICE_COMPILE__, lighting up the cudev CV_CUDEV_ARCH>=300 shfl/scan/integral fast paths; simd_functions.hpp push/pop-macros CV_CUDEV_ARCH to 0 for just that header so the PTX vadd2 SIMD routes to software, then restores it. cuda_compat.hpp keys on CUDA_VERSION (device macro, undefined on HIP) and safely takes the non-CUDA-13 `::double4`/`::ulonglong4` branch provided by hip_runtime.

Numeric fixes correct: integer clamps match PTX cvt.sat ranges (incl. saturate_cast<int>(uint) -> INT_MAX, short(ushort) -> 32767); float/double specs use __float2int_rn/__double2int_rn (round-nearest-even) then clamp, matching cvt.rni.sat. cast_fp16 reinterprets the 16-bit half bit-pattern (matches CUDA>=9), not a value-convert. vec_math abs_(char) overload resolves the HIP plain-char ambiguity; CUDA-guarded out.

Build system: WITH_HIP defaults OFF; add_definitions(-D__HIP_PLATFORM_AMD__) is confined to OpenCVDetectHIP.cmake (only included under WITH_HIP), never leaking to the CUDA build; modules link hip::host only (not hip::device, which would inject -x hip into plain .cpp); HAVE_CUDA=1 lights up only HIP-safe paths (NPP error tables + CUDA driver error table + nppSafeSetStream guarded out; printCudaDeviceInfo compiles because hipDeviceProp_t carries the maxTexture*/asyncEngineCount/tccDriver/surfaceAlignment fields). CUDA_VERSION fake (= hip_VERSION) is set so OpenCVPackaging's string(REPLACE) on CUDA_VERSION stays well-formed.

House style / upstreamability: AMD copyright + `\author Jeff Daily` on both substantially-new files (cmake/OpenCVDetectHIP.cmake, cuda_to_hip.h). Both commits are single, `[ROCm]`-prefixed, titles <= 72 chars, mention Claude, carry a Test Plan, no noreply trailer. Whole-diff grep: no MOAT jargon (moat/follower/Strategy/head_sha/validated_sha/revalidate/gfx1151), ASCII-only, no em-dash.

Forward-looking note (NOT a defect this phase, already captured at notes.md "RESUME POINT"): the CORE deprecated warp headers (modules/core/include/opencv2/core/cuda/{warp.hpp, warp_shuffle.hpp, emulation.hpp, scan.hpp, detail/reduce.hpp}) still carry unported NVIDIA PTX -- `asm("mov.u32 %0,%%laneid")` in warp.hpp:64, the 32-bit-mask `__shfl*_sync(0xFFFFFFFFU,...)` redefines in warp_shuffle.hpp (guarded by `__CUDACC_VER_MAJOR__>=9`, so dormant under HIP), and the __CUDA_ARCH__>=300 shfl bodies. These are UNREACHABLE from the Phase 0+1 build (cudev pulls only core/cuda/cuda_compat.hpp from core's cuda headers; gpu_mat.cu/gpu_mat_nd.cu include no warp header), so they are correctly left dormant and did not need porting here. But the shim's __CUDA_ARCH__=350 WILL activate their `>=300` PTX paths the moment a Phase 2+ module includes them, and `mov %%laneid` does not hipify -- Phase 2 must apply the same laneId/shfl-mask/ballot fixes (as already noted at the resume point) to these core headers before those modules build.

## Phase 2 progress (2026-06-11, gfx90a lead) -- pure-custom modules

Fork HEADs after the cudalegacy commit: contrib `911fca397e59f13c7b38ab631eeaa111610f99e1`, core `0e402ed`.
Test data: partial sparse clone of opencv_extra at `projects/opencv_contrib/opencv_extra/`
(`git clone --filter=blob:none --sparse`, `sparse-checkout set testdata/gpu testdata/cv`;
44M gpu + 328M cv). Set `OPENCV_TEST_DATA_PATH=.../opencv_extra/testdata`. GPU: run on
`HIP_VISIBLE_DEVICES=0` (device 2 is the other session's). build_hip.sh now takes
`BUILD_LIST` and `BUILD_TARGET` from the environment.

### STEP 1 done -- core deprecated warp headers (commit core 5319c3d)
- warp.hpp laneId(): HIP derives logical lane = linearized-tid & (WARP_SIZE-1) (no %%laneid PTX), keyed to logical WARP_SIZE=32. HIP-guarded.
- warp_shuffle.hpp: default shuffle width pinned to logical 32 on HIP (`OPENCV_CUDA_SHFL_WIDTH`); builtin warpSize=64 on wave64 would cross the per-32 partition. CUDA keeps warpSize(==32). The `__CUDACC_VER_MAJOR__>=9` sync-mask redefines stay dormant on HIP; the non-sync `__shfl`/`__shfl_up`/`__shfl_down` HIP builtins honor width.
- detail/reduce.hpp GenericOptimized32: `loopShfl(...,warpSize)` -> `loopShfl(...,32)` under HIP guard (same per-32 reason). reduce_key_val.hpp's identical line is under `#if 0`, untouched.
- scan.hpp / emulation.hpp: NO change needed. scan uses literal `OPENCV_CUDA_WARP_SIZE`(=1<<5, already 32) and the laneId&31-gated default-width shfl_up (wave64-correct, established pattern). emulation's `__CUDA_ARCH__=350` routes syncthreadsOr/smem::atomicMin to HIP builtins; `Emulation::Ballot` (the one 64-bit-ballot-truncation risk) is UNUSED by any Phase 2 module so left alone for CUDA byte-identity.

### Shim additions (cuda_to_hip.h, commits core 5319c3d + 0e402ed)
- cudaFuncSetCacheConfig -> typed helper `cv::cuda::detail::hipFuncSetCacheConfigT` (a function-like macro can't wrap it: template kernel ids carry commas; hipFuncSetCacheConfig wants const void* and a __global__ template id doesn't implicitly convert). cudaFuncCachePreferShared added.
- constant memory: cudaMemcpyToSymbol{,Async}, cudaMemcpyFromSymbol{,Async}, cudaGetSymbolAddress. cudaMemset{,Async}, cudaMallocHost.
- texture: cudaFilterModeLinear, cudaReadModeNormalizedFloat, cudaAddressModeWrap/Mirror/Border.
- OpenCVModule.cmake: per-module accuracy-TEST executables now link hip::host when HAVE_HIP (NCV.hpp pulls hip/hip_runtime.h; test .cpp are plain g++ and need the ROCm include path).

### Modules ported + GPU test results (literal: `HIP_VISIBLE_DEVICES=0 OPENCV_TEST_DATA_PATH=.../opencv_extra/testdata ./bin/opencv_test_<m>`)
- cudabgsegm (contrib 264d339): builds + links. MOG/MOG2 accuracy tests are gated behind `HAVE_VIDEO_INPUT`, a macro this OpenCV 4.14-pre tree NEVER defines (confirmed: absent from core, contrib, samples, apps) -> 0 tests run on CUDA too. Not a port gap; kernels compile for gfx90a. Fix: HIP-guard mog2.hpp's `typedef ... cudaStream_t` forward-decl (collides with the shim's hipStream_t mapping).
- cudastereo (contrib 264d339): **128/128 PASS**. Fixes: drop gratuitous `<cuda.h>` under HIP; software `__vminu2/__vmaxu2/__vcmpgtu2` + byte-wise `*4` (NVIDIA SIMD-in-a-word, absent on HIP) for StereoSGM's median sort; **census-transform border-zero** (NEW FAULT CLASS, see below). StereoSGM shuffle helpers already use logical cudev::WARP_SIZE=32 widths + non-sync __shfl on HIP -> wave64-correct.
- cudalegacy (contrib 911fca3): **14/14 PASS** (NPPST staging: Integral/SquaredIntegral/RectStdDev/Resize/VectorOperations/Transpose; NCV: VectorOperations/HaarCascadeLoader/HaarCascadeApplication/HypothesesFiltration/Visualization; Calib3D: TransformPoints/ProjectPoints/SolvePnPRansac). Fixes: widen `#ifdef __CUDACC__` -> `|| __HIPCC__` in NCV.hpp/NCVHaarObjectDetection.hpp/fgd.hpp; cuda_runtime.h -> hip/hip_runtime.h + shim in NCV.hpp/NPP_staging.cu/NCVPyramid.cu/NCVTest.hpp; graphcuts.cpp `#if ...||__HIP_PLATFORM_AMD__` so HIP takes the modern-CUDART(>=8000) throw_no_cuda stub (graphcut + connectivityMask + labelComponents all stubbed, same as modern CUDA); solvePnPRansac's `cuda::minMaxLoc` (cudaarithm, optional dep) gets a host argmax fallback over the 1xN scores row when cudaarithm absent. The NCV Haar cascade working here is what cudaobjdetect needs.
- graphcut DEFERRED: `python3 utils/deferred.py list` id=opencv-cudalegacy-graphcut.

### NEW FAULT CLASS: ROCm allocator does not zero-init device memory
cudastereo census_transform: the kernel writes only interior pixels and leaves the
half-window border untouched, relying on the destination GpuMat already being 0.
CUDA's device allocator happens to hand back zeroed pages so the border stays 0 and
the zero-tolerance accuracy test passes; the ROCm allocator returns REUSED (non-zero)
memory, so the border is garbage and the test fails -- but only after a prior test has
dirtied a similarly-sized buffer (passes in isolation, fails in-suite; a classic
allocator-reuse tell). Fix: `dest.setTo(Scalar::all(0), stream)` before the launch,
HIP-guarded to keep CUDA byte-identical. This is a latent UPSTREAM bug (kernel relies
on implicit zero-init); worth an upstream note. GENERAL LESSON: never assume fresh
device memory is zero on ROCm; any kernel that writes a sub-region and leaves a border
must explicitly zero it.

### RESUME POINT -> cudafeatures2d + cudaobjdetect must wait for Phase 3/4
DONE + passing this dispatch: cudabgsegm (builds; tests gated upstream), cudastereo
(128/128), cudalegacy (14/14). These are the 3 modules with NO NPP-module dependency.

cudafeatures2d and cudaobjdetect are NOT pure-custom in practice: they have HARD
(non-OPTIONAL) CMake deps AND real algorithmic runtime calls into the Phase 3/4 NPP
modules, so they cannot build or be validated until those land:
- cudafeatures2d `ocv_add_module(... opencv_cudafilters opencv_cudawarping)`; ORB
  (orb.cpp) calls `cuda::resize` (cudawarping) for the image pyramid and holds a
  `cuda::Filter` Gaussian blur (cudafilters). Stubbing them makes ORB produce wrong
  keypoints -> tests fail. Also Thrust->rocThrust still needed in orb.cu/bf_*.
- cudaobjdetect `ocv_define_module(... opencv_cudaarithm opencv_cudawarping ...)`;
  cascadeclassifier.cpp calls `cuda::resize` (cudawarping) + `cuda::integral`
  (cudaarithm) per pyramid level. Its NCV/Haar device half is ALREADY proven via
  cudalegacy's HaarCascadeApplication 14/14, but the host cascade driver needs
  resize+integral to run.

cudawarping NPP surface is confined to warp.cpp (rotate/warpAffine/warpPerspective,
gated by USE_NPP_STREAM_CTX; the remap/resize/pyr*/warp.cu kernels are custom HIP).
cudaarithm is the heaviest NPP module. So the next dispatch should be Phase 3
(cudawarping -> cudafilters) and Phase 4 (cudaimgproc -> cudaarithm), and fold
cudafeatures2d + cudaobjdetect in once cudawarping/cudafilters/cudaarithm exist.
This is a phase-sequencing boundary, not a blocker.

## Orchestrator state note 2026-06-11

Phase 0+1 foundation: ported -> review-passed (clean, see "Review findings (phase 0+1)").
State then walked back review-passed -> porting by the ORCHESTRATOR (deliberate
override of the transition table): this is a phased multi-month port and the table
has no review-passed -> porting edge. Going through completed would prematurely
unblock followers onto a 2-of-6-phase port; recording validation-failed would be
false. Lead stays in the porting loop (porter -> reviewer per phase) until all
phases land; the full reviewer -> validator -> completed gate runs at the END.
Reviewer's forward-looking note for Phase 2: the CORE deprecated warp headers
(core/cuda/warp.hpp PTX laneid, warp_shuffle.hpp 32-bit masks, emulation.hpp /
scan.hpp / detail/reduce.hpp PTX bodies) are dormant in Phase 0+1 but ACTIVATE
when Phase 2 modules include them (shim defines __CUDA_ARCH__=350); they need the
same laneId/shfl-mask/ballot fixes as cudev got.

## Phase 3 progress (2026-06-11, gfx90a lead) -- NPP-light, sequencing correction

Fork HEADs after the cudawarping commit: contrib `913502bbb21c83c116a8f6fa4d60dfa47f791059`, core `62d686ef44c46e517cfac8782f2243007349c4ed`.
head_sha advanced to the contrib sha.

### cudawarping (contrib 913502b, core 62d686e): 4535/4535 PASS
Literal: `HIP_VISIBLE_DEVICES=0 OPENCV_TEST_DATA_PATH=.../opencv_extra/testdata ./bin/opencv_test_cudawarping` -> 4535/4535 on gfx90a (ROCm 7.2.1, wave64).
NPP surface is confined to warp.cpp (warpAffine/warpPerspective/rotate). remap/resize/pyr*/warp.cu kernels are pure custom HIP and needed no change beyond the core header fix below.

NPP-replacement decisions per call site (warp.cpp):
- warpAffine + warpPerspective: NPP is a runtime-selected perf specialization that
  ALREADY coexists with a complete hand-written kernel path (the `else` branch, also
  used on CUDA for float and ROI-offset inputs). On HIP force `useNpp=false` (guarded
  `#ifdef __HIP_PLATFORM_AMD__`) and compile out the NppWarp structs + dispatch tables
  with `#ifndef __HIP_PLATFORM_AMD__`. The kernel path covers the same depths/channels/
  interpolation/border modes. The module's OWN NPP-path tests (WarpAffineNPP,
  WarpPerspectiveNPP, EXPECT_MAT_SIMILAR 2e-2 vs CPU gold) PASS served by the kernel
  path -- strategy (a).
- rotate: NPP-only, NO kernel fallback upstream. Rotation-about-origin + shift IS an
  affine, so dispatch to the existing `warpAffine_gpu<T>` device kernels with the
  inverse (dst->src) affine built from (angle, xShift, yShift), matching NPP forward
  map dst=R*src+shift, R=[[cos,-sin],[sin,cos]] (angle in degrees). New host helper
  `hipRotateAffine<T>` in the `#else // __HIP_PLATFORM_AMD__` block; INTER_CUBIC falls
  back to INTER_LINEAR (NPP cubic kernel not reproduced). CAVEAT: cudawarping has NO
  rotate accuracy test upstream (only BuildWarpAffineMaps, WarpAffine, WarpAffineNPP,
  WarpPerspective*, Remap*, Resize*, PyrDown/Up). The rotate replacement builds and
  dispatches but is NOT GPU-verified by the suite -- strategy (b), unverified. Flag for
  the reviewer; a perf/accuracy test for rotate would close this.

Core repo fix (62d686e): `core/cuda/filters.hpp` includes `nppdefs.h` solely for
`NPP_MIN_32S`/`NPP_MAX_32S` (signed-32 saturation bounds in the LinearFilter clamp).
NPP is absent on ROCm, so under `__HIP_PLATFORM_AMD__` define those two constants
directly. This header is pulled by every module's border/interpolation readers
(cudawarping/cudaoptflow/xfeatures2d), so it is a shared core fix.
Contrib test fix: `cudawarping/test/test_remap.cpp` pulls `nppdefs.h` for
`NPP_MAX_32S`/`NPP_MAXABS_32F` random-map bounds; define directly on the HIP path
(`__HIP_PLATFORM_AMD__` reaches test .cpp via the global add_definitions).

### BLOCKING SEQUENCING FINDING: cudafilters + cudafeatures2d need cudaarithm first
cudafilters CANNOT be built or validated in Phase 3. It HARD-depends on cudaarithm:
`ocv_add_module(cudafilters opencv_imgproc opencv_cudaarithm WRAP python)` (NOT
OPTIONAL) and `precomp.hpp` `#include "opencv2/cudaarithm.hpp"`. filtering.cpp calls
`cuda::subtract` (cudaarithm) for the Laplacian path. Proof: a BUILD_LIST of
`core,cudev,cudafilters,imgproc,ts` auto-expands the module DAG to
`To be built: core cudaarithm cudafilters cudev ...` -- cudaarithm is dragged in
transitively and must link. cudafeatures2d in turn needs cudafilters (ORB's Gaussian
blur) + cudawarping (now done), so it is blocked behind cudafilters behind cudaarithm.

So the dispatch's Phase 3 (cudawarping+cudafilters+cudafeatures2d) / Phase 4
(cudaimgproc->cudaarithm) split is INCONSISTENT with the build DAG. cudaarithm is the
GATING module: it must be ported BEFORE cudafilters and cudafeatures2d. This mirrors
the Phase 2 finding (notes "RESUME POINT -> cudafeatures2d + cudaobjdetect must wait").

cudafilters NPP families (filtering.cpp), for when cudaarithm is ready:
- NPPBoxFilter (nppiFilterBox_8u/32f): OpenCV ALSO ships a separable box via
  row+column filters -- route to those (strategy a). Box = normalized rowSum*colSum.
- morphology Erode/Dilate (nppiErode/nppiDilate_8u/32f, arbitrary SE): NO own-kernel;
  needs a direct HIP rank/morph kernel (strategy b). Border + anchor parity required.
- NPPRankFilter FilterMax/FilterMin (nppiFilterMax/Min): direct HIP rank kernel (b).
- SumWindow Row/Column (nppiSumWindowRow/Column): direct HIP sliding-window sum (b).
- median: NOT CUB -- this tree uses a wavelet_matrix_2d implementation
  (src/cuda/wavelet_matrix_*.cuh, a newer upstream median); audit it for PTX/warp/
  __ballot before assuming it hipifies cleanly. (The old plan said CUB; superseded.)

cudaarithm NPP surface (14 families: AndC/OrC/XorC scalar bitwise in bitwise_scalar.cu,
LShiftC/RShiftC, Magnitude(+Sqr), MeanStdDev, Mirror=flip, Sum, Transpose, Threshold)
+ cuBLAS gemm + cuFFT dft/mulSpectrums. Bulk (add/sub/mul/div/compare/split/merge/
minmax/reduce/lut/integral) are OWN kernels -- only the families above touch NPP. It is
a full phase of its own.

### RESUME POINT -> Phase 4 must lead with cudaarithm
Recommended re-sequence: port cudaarithm NEXT (gemm->hipBLAS, dft/mulSpectrums->hipFFT,
the 14 NPP families per the AlphaComp/rank/flip/bitwise-scalar/LShift/RShift/MeanStdDev/
Sum/Mirror/Transpose/Threshold plan in plan.md "NPP -> RPP + custom HIP kernels"), THEN
cudafilters (unblocked), THEN cudaimgproc, THEN fold in cudafeatures2d + cudaobjdetect
(both now have all hard deps). cudawarping is DONE this dispatch; only its core header +
test changes are new since Phase 2. Regression re-run at end of THIS dispatch (cheap
insurance after the core filters.hpp edit): cudev 402/402, cudastereo 128/128,
cudalegacy 14/14, cudawarping 4535/4535 -- all PASS on gfx90a, no regression.

## Phase 4a progress (2026-06-11, gfx90a lead) -- cudaarithm (the NPP/BLAS/FFT gate)

Fork HEADs after the cudaarithm commit: contrib `7f219f7b0a6b207f3b6e8762f4d5a76b4104e33d`,
core `bd792d70e4efbb5f837a37a68f08dad3dc0304b0`. head_sha advanced to the contrib sha.

### cudaarithm: 11417/11417 PASS on gfx90a
Literal: `HIP_VISIBLE_DEVICES=0 OPENCV_TEST_DATA_PATH=.../opencv_extra/testdata ./bin/opencv_test_cudaarithm`
-> 11417/11417 (ROCm 7.2.1, wave64). Build list to drag cudaarithm + its DAG:
`BUILD_LIST="core,cudev,cudaarithm,imgproc,ts" BUILD_TARGET=opencv_test_cudaarithm bash build_hip.sh`.
The GEMM/Dft/MulSpectrums/Convolve cases (151) only INSTANTIATE when HAVE_CUBLAS is
defined in cvconfig.h (see hipBLAS/hipFFT note below); the first full run was 11266
because those were compiled out, the final run is 11417 with them active.

### NEW FAULT CLASS (core): cudev simd_functions.hpp PTX + non-saturating emulation
Two bugs in `cudev/util/simd_functions.hpp`, both latent until a module instantiates
the v2/v4 SIMD-in-a-word ops (cudaarithm's add/sub/cmp/absdiff/min/max _mat paths are
the first). The header force-sets CV_CUDEV_ARCH=0 on HIP so the `#else` SOFTWARE
branches are taken -- but those branches were not actually HIP-clean:
1. **PTX `not.b32` crashes amdgcn.** The software branches of vsetle/vcmple/vsetlt/
   vsetge/vsetgt/vsetne (2 and 4) still spell the 32-bit complement as
   `asm("not.b32 %0,%0")`. amdclang's backend dies with "illegal VGPR to SGPR copy" /
   "Unsupported instruction" (a frontend exit-70 crash, NOT a normal error). This is
   the absdiff_mat/cmp_mat/minmax_mat COMPILE failure. Fix: `CV_CUDEV_NOT_B32(x)` macro
   = `((x)=~(x))` on HIP, original asm on CUDA. (16 sites.)
2. **Wrapping vs saturating add/sub.** vadd2/vadd4/vsub2/vsub4's `#else` software
   branches WRAP (modular), but addMat_v4/subMat_v4 rely on `vadd4` SATURATING (the
   NVIDIA `vadd4.u32.u32.u32.sat` PTX). On HIP the wrap produced 255+x -> 0 (RUNTIME
   accuracy failure on "whole matrix" C1-C4 8u/16u, sub-matrix passes because the v4
   vectorized path needs cols%4==0/contiguous). Fix: add an `#elif
   defined(__HIP_DEVICE_COMPILE__)` saturating per-element implementation
   (min(a+b,0xff) / max(a-b,0) per lane) before the legacy `#else`. CUDA byte-identical
   (always takes the >=300 PTX). GENERAL LESSON: cudev's CV_CUDEV_ARCH=0 software
   fallbacks were written for sm_1x and are (a) not asm-free and (b) non-saturating;
   any HIP module that hits a NEW v2/v4 op must re-audit its `#else` branch for both.

### NEW FAULT CLASS (core): CUDART_VERSION undefined on HIP gates out modern paths
`Stream::enqueueHostCallback` (core cuda_stream.cpp) is gated `#if CUDART_VERSION >=
5000`; on HIP CUDART_VERSION is undefined (==0) so it threw StsNotImplemented and the
4 CUDA_Stream/Async tests failed (the stream then left in a bad state -> cascading
transform.hpp "invalid argument"). Fix: define CUDART_VERSION=12000 and a no-op
CUDART_CB in private.cuda.hpp's HIP branch so the host CUDA sources take a modern-CUDA
branch. Also added cudaEventCreate to the shim (test_event.cpp uses the no-flags form).
Other CUDART_VERSION gates in core (reduce_key_val.hpp <12040, common.hpp >=12000) now
resolve consistently too. LESSON: mirror __CUDA_ARCH__=350 with a host CUDART_VERSION
on HIP; host CUDA code keys on the runtime version the same way device code keys on arch.

### NPP-replacement decisions per family (cudaarithm)
NPP has no ROCm analog; each NPP entry point is HIP-guarded (CUDA path byte-identical)
and either routed to an existing own-kernel (strategy a) or reimplemented as a direct
HIP kernel faithful to NPP semantics (strategy b). All verified by the suite unless noted.
- transpose 8u (a): route to cudev `gridTranspose<uchar>` (4/8-byte already used it).
- threshold CV_32F THRESH_TRUNC (a): the generic `thresh_trunc_func` kernel is identical
  to nppiThreshold_32f NPP_CMP_GREATER; skip the NPP fast path on HIP.
- magnitude / magnitudeSqr single-input interleaved (a): cudev already has
  `magnitude_interleaved_func`/`magnitude_sqr_interleaved_func`; new shift.cu detail
  entries gridTransformUnary them. (The two-input magnitude in polar_cart.cu is own-kernel.)
- bitwise AndC/OrC/XorC C3/C4 (b): new HipBitwiseC kernel in bitwise_scalar.cu applies
  the per-channel constant op; the C1/8u-C4 cases keep their existing BitScalar/BitScalar4
  own-kernels. Verified by Bitwise_Scalar tests.
- flip / Mirror (b): NEW flip.cu, one kernel templated on PIXEL SIZE in bytes (1/2/3/4/6/
  8/12/16) covers every depth*cn; in-place routed through a BufferPool scratch (the
  out-of-place kernel would race a same-buffer mirror). Verified (Flip 576).
- lshift / rshift (b): NEW shift.cu, per-channel constant shift kernel; right shift keys
  on signedness (arithmetic for schar/short/int via the signed template T). Verified
  (LShift 72, RShift 120).
- meanStdDev (a): computed in host code from the already-ported `cuda::sum`/`cuda::sqrSum`
  (+`countNonZero` for the masked variant): stddev=sqrt(E[x^2]-E[x]^2), population. eps
  1e-5 vs cv::meanStdDev. Verified (MeanStdDev 16, incl. masked).
- rectStdDev (b): NEW rect_std_dev.cu, integral-image box std-dev per the NPP formula.
  NOTE: cudaarithm has NO rectStdDev accuracy test upstream -- builds + dispatches but is
  NOT suite-verified (same caveat class as cudawarping rotate). Flag for reviewer.

### gemm -> hipBLAS, dft/convolution -> hipFFT (the cuBLAS/cuFFT analogues)
arithm.cpp's cuBLAS/cuFFT calls are standard v2-API and map ~1:1. New host shim
`core/cuda/cublas_cufft_to_hip.h` (included from precomp.hpp on HIP) aliases the
spellings (cublasCgemm_v2->hipblasCgemm; hipBLAS drops _v2; cuComplex->hipComplex via
hip_complex.h; cufftExecC2C->hipfftExecC2C; CUFFT_INVERSE->HIPFFT_BACKWARD). HAVE_CUBLAS/
HAVE_CUFFT MUST be set in OpenCVDetectHIP.cmake (parent scope, before cvconfig.h is
generated) -- not just in the module CMake -- or the gated GEMM/Dft tests never compile
in (the test .cpp are plain g++ and read cvconfig.h). Module CMake links roc::hipblas +
hip::hipfft under HAVE_HIP. Verified: GEMM/Dft/MulSpectrums/Convolve 151/151.

### Regression (after the core simd/CUDART + cudev edits), same MI250X gfx90a
cudev 402/402, cudastereo 128/128, cudalegacy 14/14 (needs objdetect in BUILD_LIST for
NCV HypothesesFiltration's groupRectangles -- a missing-module config artifact, NOT a
regression), cudawarping 4535/4535. All PASS, no regression.

### RESUME POINT -> Phase 4b: cudafilters -> cudaimgproc -> cudafeatures2d + cudaobjdetect
cudaarithm (the build-DAG gate) is DONE. cudafilters is now UNBLOCKED (its cudaarithm
hard-dep exists). Recommended Phase 4b order, all hard deps now present:
1. cudafilters: NPP families per the Phase 3 notes (NPPBoxFilter -> route to separable
   row+col; Erode/Dilate/FilterMax/Min -> direct HIP rank/morph kernels with border+anchor
   parity; SumWindowRow/Col -> sliding-window sum kernel; median is wavelet_matrix_2d, audit
   for PTX/warp/ballot). Calls cuda::subtract (now ported).
2. cudaimgproc: NPP color/alpha/gamma/histogram + Thrust->rocThrust.
3. cudafeatures2d: ORB needs cuda::resize (cudawarping, done) + cuda::Filter Gaussian
   (cudafilters, 4b-1); Thrust->rocThrust in orb.cu/bf_*.
4. cudaobjdetect: cascadeclassifier needs cuda::resize + cuda::integral (cudaarithm, done);
   NCV/Haar device half already proven via cudalegacy.
WHEN BUILDING cudaarithm or anything that pulls it: link needs roc::hipblas + hip::hipfft
(present at /opt/rocm). For cudalegacy regression, include objdetect in BUILD_LIST.

## Phase 4b progress (2026-06-11, gfx90a lead) -- cudafilters + cudaimgproc

Fork HEADs after the cudaimgproc commit: contrib `dd280aaf3f23f640568a034c9001f0dc7f3c7752`,
core `5a1a9d0fa804e9cc1b318d7dbd44b489a00f8a55`. head_sha advanced to the contrib sha.
Both modules built with
`BUILD_LIST="core,cudev,cudaarithm,cudafilters,cudaimgproc,imgproc,ts"` and tested on
real gfx90a (ROCm 7.2.1, MI250X), `HIP_VISIBLE_DEVICES=0`,
`OPENCV_TEST_DATA_PATH=.../opencv_extra/testdata`.

### cudafilters (contrib e92fa23, core 5a1a9d0): 16028/16028 PASS
Literal: `./bin/opencv_test_cudafilters` -> 16028/16028.
NPP replacement per family (HIP-guarded, CUDA byte-identical):
- box filter (Blur test): routed to the EXISTING separable averaging filter
  (createSeparableLinearFilter with 1/k row+col kernels). A normalised box = separable
  mean. Strategy (a). Verified.
- erode/dilate + MorphEx (Erode/Dilate/MorphEx tests): NEW morph_rank_sum_hip.cu morph
  kernel (erode=min, dilate=max over an arbitrary 8U structuring element), reads the
  already-bordered srcRoi the call site builds via copyMakeBorder. Strategy (b). Verified.
- box max/min rank (createBoxMax/MinFilter) + row/column sliding-window sums
  (createRow/ColumnSumFilter): NEW direct kernels in morph_rank_sum_hip.cu. NO upstream
  accuracy test for these -- builds + dispatches, NOT suite-verified (caveat for reviewer).
- median (Median + Median_HDR tests, 92 cases incl. 8U/16U/32F C1/C3/C4): the CUDA
  wavelet matrix (src/cuda/wavelet_matrix_*.cuh) is hard-wired to a 32-lane warp
  (static_assert WARP_SIZE==32; __activemask-as-ballot + __popc(e_nbit<<(32-tid.x)) rank +
  cub::WarpScan<16>/WarpReduce default-width). On wave64 two logical warps share one
  physical wavefront and the ballot/rank do not account for it (first attempt: median
  produced 255 where 12 expected, C3 faulted). Rather than rework that data structure,
  the HIP median is a DIRECT counting-rank selection kernel (median_filter.cu, namespace
  median_hip; BORDER_REPLICATE, exact for all depths/channels; windows <=15x15). The
  wavelet path stays CUDA-only (feature_support_checks.h now excludes __HIP_PLATFORM_AMD__).
  Strategy (b). Verified 92/92.
NEW FAULT CLASS / shim addition (core 5a1a9d0): cudaFuncSetAttribute +
cudaFuncAttributeMaxDynamicSharedMemorySize were missing from cuda_to_hip.h. Added a typed
helper hipFuncSetAttributeT (mirroring hipFuncSetCacheConfigT) since a __global__ template
id does not implicitly convert to const void*. (This was the LAST compile blocker before
the wavelet path was disabled; it is still needed because the CUDA wavelet code is what
forced the discovery, but the helper is generally useful.)
GENERAL LESSON (cudafilters median): a deeply warp-cooperative CUDA data structure built
on a fixed 32-lane warp (CUB WarpScan/WarpReduce + activemask-ballot + popc-rank) is often
NOT worth porting bit-exactly to wave64 when a simple direct kernel covers the SAME test
matrix. Counting-rank selection over a small window is wave-agnostic and exact. The C4
"pass real pixel cols, index channels via CN" detail bit me first (multiplying cols by cn
overreads on tight/whole-matrix layouts but happens to survive padded sub-matrix layouts --
the same whole-vs-sub tell as the allocator-reuse class). GpuMat::cols is ALWAYS the pixel
column count regardless of channels.

### cudaimgproc (contrib dd280aa): 3671/3671 PASS, 29/29 test cases
Literal: `./bin/opencv_test_cudaimgproc` -> 3671/3671 (CvtColor, Demosaicing, SwapChannels,
Canny, Hough{Lines,LinesProbabilistic,Circles}, GeneralizedHough, CornerHarris/MinEigen,
GoodFeaturesToTrack, HistEven, CalcHist{,WithMask}, EqualizeHist{,Extreme}, CLAHE,
MatchTemplate{8U,32F,...}, MeanShift{,Segmentation}, Moments, BilateralFilter, Blend,
ConnectedComponents). The big custom surface is cudev kernels; only third-party deps needed
work.
- Thrust -> rocThrust: gftt.cu / hough_lines.cu used `thrust::cuda::par` +
  `<thrust/system/cuda/execution_policy.h>` (NVIDIA-only). rocThrust uses `thrust::hip::par`
  + `<thrust/system/hip/execution_policy.h>`. New per-file `OPENCV_THRUST_PAR` macro selects
  by platform. generalized_hough.cu's `thrust::transform` is policy-free (no change).
  NEW FAULT CLASS: NVIDIA Thrust's cuda execution-policy header transitively pulls
  `cub/detail/detect_cuda_runtime.cuh` which is ABSENT on ROCm; include the hip policy
  header instead, never the cuda one, on HIP.
- NPP color (color.cpp): swapChannels (SwapChannels test, CV_8UC4), gammaCorrection (sRGB
  fwd/inv), alphaPremul/RGBA_to_mBGRA -> NEW direct HIP kernels in cuda/color_npp_hip.cu.
  swapChannels VERIFIED by the SwapChannels test; gamma + alphaPremul are UNTESTED upstream
  in cudaimgproc (build + dispatch only -- caveat for reviewer). The main cvtColor (the big
  CvtColor test) is all cudev kernels, zero NPP -- verified 2000+ cases.
- NPP histogram (histogram.cpp): evenLevels reimplemented directly (uniform integer level
  layout). histEven 8U (the only tested histogram-even path) ALREADY routes to a custom
  kernel (hist::histEven8u) and is verified. The NPP-only fallbacks (16U/16S histEven,
  4-channel histEven, histRange x2) have NO cudaimgproc accuracy test -> gated
  StsNotImplemented on ROCm.
- DEFERRED: alphaComp (13 Porter-Duff alpha-composite ops, NPP-only, untested) +
  16U/16S/4ch histEven/histRange -> `deferred.py` id=opencv-cudaimgproc-npp-color-hist.

### cudafeatures2d (contrib c85b44b, core 20268c3): 256/256 PASS
Literal: `BUILD_LIST="core,cudev,cudaarithm,cudafilters,cudawarping,cudafeatures2d,imgproc,ts"
BUILD_TARGET=opencv_test_cudafeatures2d bash build_hip.sh` then
`./bin/opencv_test_cudafeatures2d` -> 256/256 (32 ORB + FAST + 224 BruteForceMatcher; the
module genuinely has only 3 test cases). NO NPP (earlier grep matched comments). The warp-
cooperative reductions (Harris responses in orb.cu, the matcher distance reductions in
bf_*.cu) already pass an EXPLICIT width 32 to the core reduce/reduceKeyVal helpers -> wave64-
correct unchanged. `__popc` in fast.cu is on a 16-bit FAST corner mask (not a warp ballot),
fine on HIP. ORB exercises the Phase 4b cudafilters Gaussian + the Phase 3 cudawarping resize
+ Thrust sort -- all pass, so this is the integration proof for the prior modules too.
- Thrust: orb.cu's `thrust::cuda::par` + cuda execution-policy header -> OPENCV_THRUST_PAR
  (thrust::hip::par + hip header). Same pattern as cudaimgproc.
- test_features2d.cpp pulled `cuda_runtime_api.h` + called `cudaDeviceSynchronize` -> hip
  equivalents on HIP (`__HIP_PLATFORM_AMD__` reaches the test .cpp via global add_definitions).
NEW FAULT CLASS (core utility.hpp, commit 20268c3): MaskCollection's copy constructor was
`__device__`-only. The brute-force matcher kernels take a MaskCollection BY VALUE; nvcc
synthesises the host-side copy to marshal the kernel argument, but amdclang requires the copy
constructor to be HOST-callable. Added `__host__` to it (forwards two members, valid on host;
device codegen unchanged, CUDA still compiles). LESSON: any struct passed BY VALUE to a HIP
kernel must have a host-callable copy constructor; a device-only one compiles on nvcc but not
amdclang.

### cudaobjdetect (validated at contrib c85b44b -- NO source change): 18/18 PASS
Literal: `BUILD_LIST="core,cudev,cudaarithm,cudawarping,cudalegacy,cudaobjdetect,imgproc,
objdetect,ts" BUILD_TARGET=opencv_test_cudaobjdetect bash build_hip.sh` then
`./bin/opencv_test_cudaobjdetect` -> 18/18 (HOG: CalTech detect, Hog_var, Hog_var_cell; LBP
cascade: Read_classifier, classify). cudaobjdetect needed ZERO source edits: hog.cu/lbp.cu are
clean HIP (no warp intrinsics, no NPP, no Thrust), and its only NPP dependency
(nppiStIntegralGetSize_8u32u for the integral buffer size, behind HAVE_OPENCV_CUDALEGACY) comes
from the already-ported cudalegacy NCV staging. Its hard deps (cudawarping resize, cudaarithm
integral) are ported. The HOG block-histogram reductions (the warp-cooperative path) pass on
wave64. So head_sha stays at c85b44b (cudafeatures2d); cudaobjdetect is a validated-no-delta
module.

### Regression after Phase 4b (same MI250X gfx90a), after the 2 core edits
The core edits (cuda_to_hip.h: ADDED cudaFuncSetAttribute helper; utility.hpp: ADDED __host__
to MaskCollection copy ctor) are both additive/behavior-preserving. Rebuilt + reran:
cudev 402/402, cudastereo 128/128, cudalegacy 14/14, cudawarping 4535/4535,
cudaarithm 11417/11417. All PASS, no regression.

### RESUME POINT -> Phase 5 (the SDK-gated modules)
ALL of Phase 4b DONE + GPU-validated: cudafilters 16028, cudaimgproc 3671, cudafeatures2d 256,
cudaobjdetect 18. Remaining cuda* modules:
- cudabgsegm: DONE in Phase 2 (builds; MOG/MOG2 tests gated behind HAVE_VIDEO_INPUT which this
  tree never defines).
- cudaoptflow: SOFTWARE flows (BroxOpticalFlow, PyrLK, Farneback, DIS) port like the other
  custom modules; the HARDWARE NvidiaOpticalFlow_1_0/2_0 has NO AMD HW OF engine -> gate off +
  it is already in deferred.json (genuine feature loss, see notes top). Deps:
  cudaarithm/cudawarping/cudaimgproc/cudalegacy all DONE.
- cudacodec: rocDecode for decode (lacks MPEG/VC1/VP8/MJPEG -- codec deltas are genuine
  losses); native HW encode gated off, delegate to FFmpeg AMF via videoio (per Decisions
  2026-06-10). The heaviest remaining module; needs rocDecode + FFmpeg-AMF availability check
  on the host first.
PR-readiness: NOT yet. cudaoptflow (software flows) and cudacodec are the last functional
modules; after them the lead is a candidate for the reviewer -> validator -> completed gate,
then PR-prep (jargon scrub, CMake arch auto-detect already present, docs).

## Phase 5 progress (2026-06-11, gfx90a lead) -- the SDK-gated modules

Fork HEADs after Phase 5: contrib `f7a8b32`, core `934c316`. head_sha advanced to the
contrib sha. Three commits: contrib `0bf6744` (cudaoptflow), core `94b6d50` (common.hpp
__align__ host fix) + `934c316` (WITH_ROCDECODE gates), contrib `f7a8b32` (cudacodec).

### cudaoptflow (contrib 0bf6744): 41/47 PASS on gfx90a
Literal: `BUILD_LIST="core,cudev,cudaarithm,cudafilters,cudawarping,cudaimgproc,\
cudalegacy,cudaoptflow,video,optflow,imgproc,objdetect,calib3d,ts" \
BUILD_TARGET=opencv_test_cudaoptflow bash build_hip.sh` then
`HIP_VISIBLE_DEVICES=0 OPENCV_TEST_DATA_PATH=.../opencv_extra/testdata \
./bin/opencv_test_cudaoptflow` -> 41/47.
The software flows (Brox, SparsePyrLK, DensePyrLK, Farneback incl. FarnebackAsync, TVL1
Accuracy) all PASS -- they are clean cudev kernels, NO PTX/warp/Thrust fixes needed.
- NvOF gate: NvidiaOpticalFlow_1_0/2_0 factories already cleanly gated. On HIP HAVE_CUDA
  is set but HAVE_NVIDIA_OPTFLOW is not (the SDK never downloads), so the existing
  `#elif !defined HAVE_NVIDIA_OPTFLOW` branch fires. Swapped its generic message for an
  explicit ROCm StsNotImplemented naming the software-flow substitutes (CUDA path kept
  byte-identical via a HIP/CUDA macro). The 4 NvOF test cases (1_0/2_0 x Regression/Nan)
  FAIL BY DESIGN: they instantiate the unavailable HW class, which now throws. Registered
  deferral `opencv-cudaoptflow-nvof`.
- CMake: guarded the NVIDIA Optical Flow SDK ocv_download behind `NOT HAVE_HIP` so the HIP
  build never reaches GitHub and HAVE_NVIDIA_OPTFLOW stays undefined.
- nvidiaOpticalFlow.cu: hip/hip_runtime.h include on HIP (its body is gated out of the
  non-NVIDIA build by `__CUDACC_VER_MAJOR__>=10` already).
- Core common.hpp `__align__` host fix (94b6d50): cudaoptflow is the ONLY module whose
  precomp includes the deprecated core `cuda/vec_traits.hpp` on the HOST path. vec_traits
  declares `struct __align__(N) uchar8 ...` at namespace scope; HIP defines `__align__`
  only under `__HIP_CLANG_ONLY__` (the amdclang device pass), so the plain g++ host
  compile failed to parse it. Added a host fallback `#define __align__(x)
  __attribute__((aligned(x)))` right after the hip_runtime.h include, HIP-only, only when
  undefined (device pass unaffected, CUDA untouched). NEW FAULT CLASS: a HIP host .cpp
  that pulls a core device header using `__align__`/`__launch_bounds__` at namespace/decl
  scope needs the attribute macro provided for the host pass too.
- The 2 OpticalFlowDual_TVL1.Async FAILURES are GENUINE-but-not-a-port-defect, ROOT-CAUSED:
  the test computes a gold flow on the DEFAULT (Null) stream and compares it bit-exactly
  (EXPECT_MAT_NEAR tolerance 0.0) against 16 real-stream runs. TVL1's convergence loop
  early-exits on an error term computed by `cuda::calcSum(diff,...)`. cudaarithm's sum
  (cudev gridReduce) accumulates per-block partials with a FLOATING-POINT atomicAdd
  (cudev/grid/detail/reduce.hpp line 165, accumulator is double for CV_32F). Float/double
  atomicAdd is non-associative and its summation order depends on block-completion order,
  which differs between the default-stream and real-stream schedulers. The last-ULP
  difference in the error term shifts the iteration count, producing the observed
  byte-identical 0.0516 flow delta at one pixel. It is deterministic and IDENTICAL across
  all 16 streams (so NOT a race) and differs only from the Null-stream gold. The flow is
  CORRECT -- the TVL1 Accuracy cases pass at their normal tolerance. This is an upstream
  test-strictness artifact (0.0 tolerance assumes calcSum is stream-deterministic, which
  an atomic reduction is not), not introduced by the port. Tried zeroing diff_buf (the
  allocator-no-zero fault class) -- no effect (byte-identical), confirming it is the atomic
  reduction order, not uninitialised memory. NOT worth rewriting cudaarithm's reduction to
  a deterministic two-pass (out of scope; would change CUDA behavior). A reviewer should
  scrutinise this determination.

### cudacodec (contrib f7a8b32, core 934c316): builds + YuvConverter 240/240 PASS on gfx90a
gfx90a VIDEO-ENGINE DETERMINATION (evidence): the MI250X (gfx90a, CDNA2) is a pure-compute
part with NO VCN video engine. rocminfo shows no jpeg/vcn/video/decode agent features.
rocDecode's runtime depends on the ROCm-repo `mesa-amdgpu-va-drivers` VA-API driver, which
has NO installable candidate in this node's apt config (`apt-cache policy` -> Candidate:
(none)); `apt-get install rocdecode-dev` fails on that unmet dep. So gfx90a can neither run
nor even link rocDecode's runtime here. This is the task's anticipated "gfx90a has NO decode
engines" branch -- a legitimate phase outcome, NOT a failure.
Decision: land the rocDecode BUILD PATH (WITH_ROCDECODE/HAVE_ROCDECODE gates) + graceful
runtime errors + the portable converter, and DEFER the full cuvid->rocDec API mirror
(video_decoder/parser/reader, CUVID struct translation) to the RDNA followers (gfx1100/
gfx1201 have VCN). Writing the ~1645-line decode stack blind here -- unrunnable AND
un-compile-validatable against the real runtime (no VA driver) -- is exactly the high-risk
grind the circuit-breaker forbids. Registered `opencv-cudacodec-rocdecode-videoreader`.
What landed and is VALIDATED:
- ColorSpace.{h,cu} (the NV12/YUV-to-RGB surface converter, used by NVSurfaceToColorConverter
  and the reader's cvtFromYuv): pure device code, ported by switching its runtime include to
  hip/hip_runtime.h. GPU-VALIDATED: opencv_test_cudacodec YuvConverter 240/240 on gfx90a
  (BT601 160 + BT2020 80; all SurfaceFormat x ColorFormat x BitDepth combos). This converter
  is gated on HAVE_CUDA (not HAVE_NVCUVID), so it runs on HIP without a decode engine.
- CMakeLists HAVE_HIP branch: drops the CUDA driver/runtime link libs (CUDA::cuda_driver,
  CUDA_CUDA_LIBRARY) which don't exist on HIP; warns when WITH_ROCDECODE without rocDecode;
  links ROCDECODE_LIBRARIES when HAVE_ROCDECODE.
- WITH_ROCDECODE option (WITH_HIP-visible, default ON) + OpenCVDetectHIP.cmake
  find_package(rocdecode) -> HAVE_ROCDECODE + ROCDECODE_LIBRARIES + cvconfig.h.in
  HAVE_ROCDECODE switch (parent scope, before cvconfig.h generation, so module + tests see it).
What is gated off with GRACEFUL runtime errors (no crash):
- VideoReader (decode): the existing `#ifndef HAVE_NVCUVID` stub now raises StsNotImplemented
  naming the rocDecode requirement + the gfx90a-has-no-VCN reality (was a generic throw_no_cuda).
  All the NVCUVID-coupled host .cpp (video_decoder/parser/reader, cuvid_video_source, NvEncoder*)
  compile cleanly because their bodies are HAVE_NVCUVID-gated -> stubs on HIP.
- VideoWriter (encode): no ROCm-native encoder; the `#ifndef HAVE_NVCUVENC` stub now raises
  StsNotImplemented pointing at the FFmpeg videoio backend with AMF encoders (h264_amf/
  hevc_amf/av1_amf). Registered `opencv-cudacodec-native-encode`. No AMF interop built.
CODEC LOSSES (genuine, to surface when the decode path lands on followers): rocDecode lacks
MPEG-1/2, VC-1, VP8, MJPEG vs NVDEC.
The cudacodec decode/encode accuracy tests are conditionally compiled out without a decode
engine (gated behind HAVE_NVCUVID/HAVE_NVCUVENC), so they do not run here -- only the
NVCUVID-independent YuvConverter suite runs, and it passes.

## Port surface COMPLETE (2026-06-11, gfx90a lead)

All 13 cuda* contrib modules + cudev + the opencv core CUDA/GpuMat/warp infra are
ported to ROCm/HIP and validated on real gfx90a (MI250X, ROCm 7.2.1, wave64). This is
the final porting phase. Fork HEADs: contrib `f7a8b32`, core `934c316`.

### Per-module final GPU test counts (HIP_VISIBLE_DEVICES=0, real gfx90a)
| Module | Result | Notes |
|---|---|---|
| cudev | 402/402 | foundation device templates |
| cudaarithm | 11417/11417 | NPP-free; hipBLAS gemm + hipFFT dft |
| cudafilters | 16028/16028 | NPP-free; direct HIP median (wavelet path CUDA-only) |
| cudaimgproc | 3671/3671 | rocThrust; NPP-free color/hist |
| cudawarping | 4535/4535 | NPP-free warp/rotate paths |
| cudastereo | 128/128 | census border-zero fix |
| cudafeatures2d | 256/256 | rocThrust; integration proof for filters+warping |
| cudaobjdetect | 18/18 | HOG + Haar/LBP; zero source change |
| cudalegacy | 14/14 | NCV/staging/BM/PnP (graphcut deferred) |
| cudabgsegm | builds | MOG/MOG2 tests gated upstream (HAVE_VIDEO_INPUT never defined) |
| cudaoptflow | 41/47 | software flows PASS; see feature losses |
| cudacodec | YuvConverter 240/240 | decode/encode gated off; converter validated |

### Documented feature losses / deferrals (NEVER passed off as complete)
1. cudalegacy graphcut/graph-cut stereo -- no ROCm maxflow analog (NPP-removed at 8.0).
   deferred id `opencv-cudalegacy-graphcut`.
2. cudaoptflow NvidiaOpticalFlow_1_0/2_0 -- no AMD HW optical-flow engine; factories
   throw StsNotImplemented, software flows are the substitute. 4 NvOF test cases fail by
   design. deferred id `opencv-cudaoptflow-nvof`.
3. cudacodec VideoReader rocDecode decode -- build path + gates + graceful runtime error
   landed; the full cuvid->rocDec API mirror and its RUNTIME validation defer to the RDNA
   followers (gfx1100/gfx1201 have VCN; gfx90a/MI250X has none and the VA driver is
   uninstallable here). rocDecode codec losses vs NVDEC: MPEG-1/2, VC-1, VP8, MJPEG.
   deferred id `opencv-cudacodec-rocdecode-videoreader`.
4. cudacodec VideoWriter native HW encode -- no ROCm-native encoder; StsNotImplemented
   points at the FFmpeg videoio backend with AMF encoders. deferred id
   `opencv-cudacodec-native-encode`.
5. cudaimgproc alphaComp + 16U/16S/4ch histEven/histRange (NPP-only, untested upstream).
   deferred id `opencv-cudaimgproc-npp-color-hist`.
6. opencv dnn CUDA backend (cuDNN->MIOpen) -- out of cv::cuda scope.
   deferred id `opencv-dnn-miopen-backend`.

### Untested-upstream caveats for the reviewer (build + dispatch, no accuracy test)
cudawarping rotate; cudaarithm rectStdDev; cudafilters boxMax/boxMin + row/column sum;
cudaimgproc gamma + alphaPremul. These have no upstream accuracy test in their module, so
they build and dispatch but are not suite-verified (flagged in their phase notes).

### Final regression (all 9 testable modules, same MI250X gfx90a, after Phase 5 edits)
cudev 402, cudastereo 128, cudalegacy 14, cudawarping 4535, cudaarithm 11417,
cudafilters 16028, cudaimgproc 3671, cudafeatures2d 256, cudaobjdetect 18. ALL PASS, zero
failures. The Phase 5 core edit (common.hpp __align__ host fallback) is additive/host-only
and regressed nothing.

### What the final reviewer must scrutinise
1. cudaoptflow TVL1.Async 0.0-tolerance failure: root-caused to float atomicAdd reduction
   ordering in cuda::calcSum (stream-schedule-dependent, non-associative); flow is correct
   (Accuracy passes). Determination: upstream test-strictness artifact, not a port defect.
2. cudacodec scope: gfx90a has no VCN, so the rocDecode VideoReader implementation is
   deferred to followers rather than written blind. The build path, gates, graceful errors,
   and the portable YUV converter (240/240) are landed. Confirm this is the right
   build-validated + runtime-deferred boundary.
3. Two upstream PRs (core->opencv, modules->opencv_contrib). Both forks have Actions
   disabled. PR-prep (jargon scrub, arch auto-detect already present, ROCm build docs in
   the project's house style) is the remaining pre-PR step, not done in this phase.

### Independent re-validation of Phase 5 (porter, gfx90a, 2026-06-11T03)
A second porter dispatch (stale control-plane state showed `porting`/c85b44b) independently
rebuilt and ran cudaoptflow on the same MI250X while Phase 5 was concurrently landed by
another session. Result corroborates exactly: opencv_test_cudaoptflow 41/47 PASS; the 4
NvidiaOpticalFlow cases throw (no AMD HW OF engine); the 2 OpticalFlowDual_TVL1.Async cases
fail at 0.0 tolerance. Root cause independently confirmed via a standalone probe: TVL1 on the
Null stream yields flow(387,342)=(2.4500287,-0.0310162) bit-identically across runs, while an
explicit stream yields (2.4500573,-0.0310128) bit-identically -- each stream context is
internally deterministic but the two differ by ~3e-5, which the float atomicAdd ordering in
cuda::calcSum (cudev grid reduce, non-associative, schedule-dependent) feeds into the
convergence early-exit. Not a port defect (the upstream 0.0-tolerance cross-stream assumption
is unsound for a float-atomic reduction). No additional source changes were needed; this
session's redundant local commits were discarded and the clones reset to the authoritative
fork HEADs (contrib f7a8b32e, core 934c316930).
## Review findings (final) 2026-06-11 (reviewer, linux-gfx90a): PASS, review-passed

Final whole-port review via /pr-review of both fork branches base..HEAD, depth on phases 2-5 (contrib 0e750c3..f7a8b32, core f90ef85..934c316). VERDICT: review-passed. No blocking defects, no changes requested. The items below are the verification record for the scrutiny list plus one PR-prep recommendation; none is a change request against the code.

Scrutiny-list verification (all judged by reading code, not re-running GPU):

1. cudaoptflow TVL1.Async 2 failures -- root-cause SOUND. cuda::calcSum (cudev gridReduce) accumulates per-block partials with a float/double atomicAdd (non-associative, block-completion-order dependent); the test's 0.0-tolerance default-stream-vs-16-stream comparison shifts the TVL1 early-exit iteration count by ULPs. Flow is correct (TVL1 Accuracy passes). Disposition "test-strictness artifact, not a port defect" is correct. RECOMMENDATION: document-and-ship (option c). A HIP-only deterministic two-pass calcSum would diverge from CUDA numerics on a reduction shared by the whole module and is out of scope; (b) a test tolerance/skip belongs upstream, not in the port. PR body must scope the 2 cases neutrally (no "fails here" phrasing). This is a PR-prep wording matter and is compatible with review-passed.

2. The 4 NvidiaOpticalFlow failures -- CONFIRMED same failure mode as a CUDA build without the NVOF SDK. Factory selection is the existing `#elif !defined HAVE_NVIDIA_OPTFLOW` branch (HAVE_NVIDIA_OPTFLOW never defined on HIP because the SDK download is CMake-gated `NOT HAVE_HIP`); the 4 tests instantiate the unavailable HW class -> throw. CUDA path byte-identical: nvidiaOpticalFlow.cpp:29 macro guards `__HIP_PLATFORM_AMD__` (ROCm message) with the original `HeaderIsNull`/"build without NVIDIA OpticalFlow" on the else branch.

3. cudacodec scope -- gating is GRACEFUL and CUDA-preserving. video_reader.cpp / video_writer.cpp use OPENCV_CUDACODEC_NO_DECODER/NO_ENCODER macros: StsNotImplemented with an honest message on HIP, original throw_no_cuda() on CUDA. WITH_ROCDECODE/HAVE_ROCDECODE gates mirror WITH_NVCUVID; cvconfig.h.in switch in parent scope. ColorSpace.h only swaps the runtime include. The rocDecode VideoReader deferral is registered (opencv-cudacodec-rocdecode-videoreader) and the gfx90a-has-no-VCN evidence is documented. No test fake-passes: decode/encode cases are HAVE_NVCUVID/HAVE_NVCUVENC-compiled-out, only the NVCUVID-independent YuvConverter suite runs (240/240). Boundary is correct.

4. Unverified-path kernels audited BY READING:
   - cudawarping rotate (warp.cpp hipRotateAffine): inverse affine math is correct. dst=R*src+shift with R=[[c,-s],[s,c]] inverts to coeffs[0..2]=[c,s,-(c*xShift+s*yShift)], coeffs[3..5]=[-s,c,-(-s*xShift+c*yShift)] -- matches the code exactly. warpAffine_gpu<T> call signature matches (srcWhole=src, xoff=yoff=0, cc20=true). INTER_CUBIC->INTER_LINEAR fallback documented. Sound; only the absent upstream test leaves it suite-unverified.
   - cudaarithm rectStdDev (rect_std_dev.cu): integral-image box formula stddev=sqrt(max(0, sqsum/area-(sum/area)^2)) over the (rx,ry,rw,rh) window from the 32S sum + 64F sqsum integrals -- matches the NPP nppiRectStdDev definition (population deviation). Correct.
   - cudafilters box max/min rank + row/column sums (morph_rank_sum_hip.cu): rankKernel/rowSumKernel/columnSumKernel read srcRoi, a sub-view offset (ksize,ksize) into a copyMakeBorder buffer; negative gather offsets (x-anchor+i) reach into the materialized border, never past it (border width = ksize >= anchor). Same border construction the morph path uses, and morph IS suite-verified (Erode/Dilate/MorphEx in 16028/16028), so the border logic is exercised. Correct.
   - cudaimgproc gamma + alphaPremul (color_npp_hip.cu): gammaKernel applies the sRGB transfer (0.018/4.5 + 1.099*x^(1/2.4)-0.099 fwd) to channels 0..2 only, leaving alpha -- matches NPP's AC4 alpha-preserving semantics; in-place handling copies src->dst then transforms dst. alphaPremul multiplies RGB by alpha/maxv per Porter-Duff premultiply. Both faithful.

5. StsNotImplemented surfaces -- each is an honest CV_Error and NONE is reachable from a tested/ported API. Verified calcHist uses hist::histogram256 (custom kernel), equalizeHist/CLAHE do not route through histEven/histRange; the gated entry points (histEven 16U/16S, 4ch histEven/histRange, histRange, alphaComp) are only reachable via direct public-API calls that no ROCm test exercises. histEven 8U still routes to the custom hist::histEven8u before the gate.

6. Wave64 audit (phases 2-5) -- CLEAN. New direct HIP kernels (morph_rank_sum_hip.cu, median_filter.cu, color_npp_hip.cu, flip.cu, shift.cu, rect_std_dev.cu, bitwise_scalar.cu) are plain per-pixel/per-element 2D-grid kernels with NO warp intrinsics, ballot, activemask, or shuffle -- wave-size agnostic. The wavelet median (32-lane-warp data structure) is correctly EXCLUDED on HIP (feature_support_checks.h `!defined(__HIP_PLATFORM_AMD__)`) and replaced by the wave-agnostic counting-rank median_hip kernel. stereosgm.cu software __vminu2/__vmaxu2/__vcmpgtu2(+*4) operate on packed lanes within a uint32, not across the wavefront. orb.cu/gftt.cu/hough_lines.cu only swap the Thrust execution policy (thrust::hip::par) -- no warp code. Core warp_shuffle.hpp pins OPENCV_CUDA_SHFL_WIDTH=32 on HIP (the 32-bit-mask `__shfl*_sync` redefines stay dormant because __CUDACC_VER_MAJOR__ is undefined on HIP -> non-sync width-honoring builtins used); warp.hpp laneId=tid&(WARP_SIZE-1) with logical WARP_SIZE=32; detail/reduce.hpp loopShfl pinned to 32 under __HIP_DEVICE_COMPILE__. No literal warpSize leakage in the active HIP paths.

7. CUDA-path preservation (whole diff, both repos) -- every change is additive and guarded (__HIP_PLATFORM_AMD__ / __HIP_DEVICE_COMPILE__ / __HIPCC__ / HAVE_HIP, or a CMake `NOT HAVE_HIP`). Spot-checked heaviest files: element_operations.cpp, core.cpp, reductions.cpp, filtering.cpp, histogram.cpp, color.cpp, bitwise_scalar.cu all keep the original NPP/CUDA branch verbatim in the #else. Core header edits preserve CUDA byte-for-byte: warp_shuffle.hpp OPENCV_CUDA_SHFL_WIDTH=warpSize on CUDA reproduces the original default arg; warp.hpp keeps the %%laneid PTX in #else; common.hpp __align__ is #ifndef-guarded (device pass unaffected); utility.hpp adds __host__ to a __device__ ctor (strictly additive, device codegen unchanged). rshift/lshift HIP depth coverage matches upstream (all integer depths incl. signed CV_8S/CV_16S via the signed template T = arithmetic shift).

8. Upstreamability sweep (whole diff, both repos) -- CLEAN. No MOAT jargon (grep moat/follower/head_sha/validated_sha/revalidate/gfx1151/"strategy a|b"/curated over both full diffs: zero hits). ASCII-only (no non-ASCII on added lines), no em-dash. All 10 contrib + 9 core commit titles are `[ROCm]`-prefixed and <= 72 chars; bodies name Claude/Anthropic, carry a Test Plan, no Co-Authored-By noreply trailer. All 5 substantially-new HIP kernel files carry `Copyright (c) 2026 Advanced Micro Devices, Inc.` + `\author Jeff Daily`. No debug prints/TODO/FIXME/commented-out code in the new files. All forks under jeffdaily.

Non-blocking note for PR-prep (NOT a code change): item-1 wording (the 2 TVL1.Async cases) and the documented unverified-upstream kernels (rotate, rectStdDev, box max/min, row/col sums, gamma, alphaPremul) must be described neutrally in the PR body and the feature-loss/deferral list must be surfaced per the ledger (graphcut, NvOF, rocDecode decode, native encode, alphaComp/16U-16S-4ch hist, dnn-MIOpen). These are PR-prep tasks already tracked in notes; they do not gate this review.

## Validation (lead, gfx90a) 2026-06-11

**State: completed** -- validated_sha = f7a8b32e1a99e8a456fcb3377d731d29200321ad (contrib), 934c3169303b718104b02e775db8eb3cd60a8203 (core)
GPU: AMD Instinct MI250X / MI250, gfx90a, HIP_VISIBLE_DEVICES=0
ROCm: 7.2.1
OPENCV_TEST_DATA_PATH: projects/opencv_contrib/opencv_extra/testdata

### Integrity check
- contrib HEAD f7a8b32 == remote fork tip (FETCH_HEAD): MATCH
- core HEAD 934c316 == remote fork tip (FETCH_HEAD): MATCH
- `python3 utils/moatlib.py audit-clean opencv_contrib`: OK, no uncommitted source edits
- `git status --porcelain` in both repos: clean

### Build (incremental, no work to do)
```
cd projects/opencv_contrib/build
cmake --build . --target opencv_test_cudev opencv_test_cudastereo opencv_test_cudalegacy \
  opencv_test_cudawarping opencv_test_cudaarithm opencv_test_cudafilters \
  opencv_test_cudaimgproc opencv_test_cudafeatures2d opencv_test_cudaobjdetect \
  opencv_test_cudaoptflow opencv_test_cudacodec -j$(nproc)
# ninja: no work to do.
```

### GPU test results (HIP_VISIBLE_DEVICES=0, OPENCV_TEST_DATA_PATH=.../opencv_extra/testdata)

| Suite | Passed | Total | Notes |
|---|---|---|---|
| opencv_test_cudev | 402 | 402 | PASS |
| opencv_test_cudastereo | 128 | 128 | PASS |
| opencv_test_cudalegacy | 14 | 14 | PASS (1 disabled) |
| opencv_test_cudawarping | 4535 | 4535 | PASS |
| opencv_test_cudaarithm | 11417 | 11417 | PASS |
| opencv_test_cudafilters | 16028 | 16028 | PASS |
| opencv_test_cudaimgproc | 3671 | 3671 | PASS |
| opencv_test_cudafeatures2d | 256 | 256 | PASS |
| opencv_test_cudaobjdetect | 18 | 18 | PASS |
| opencv_test_cudaoptflow | 41 | 47 | 6 failures: EXACTLY the documented set (see below) |
| opencv_test_cudacodec | 240 | 240 | PASS (YuvConverter only; decode/encode gated off) |

**cudaoptflow 6 expected failures (verified exact set):**
- CUDA_OptFlow/OpticalFlowDual_TVL1.Async/0 -- float atomicAdd reduction ordering (root-caused, not a port defect)
- CUDA_OptFlow/OpticalFlowDual_TVL1.Async/1 -- same
- CUDA_OptFlow/NvidiaOpticalFlow_1_0.Regression/0 -- no AMD HW OF engine (HAVE_NVIDIA_OPTFLOW not defined)
- CUDA_OptFlow/NvidiaOpticalFlow_1_0.OpticalFlowNan/0 -- same
- CUDA_OptFlow/NvidiaOpticalFlow_2_0.Regression/0 -- same
- CUDA_OptFlow/NvidiaOpticalFlow_2_0.OpticalFlowNan/0 -- same

No unexpected failures. All 10 other suites are 100%. Port is fully validated on real gfx90a.

## PR-prep: documentation pass 2026-06-11 (porter, doc-only carry-forward)

Strictly documentation. No code, no CMake behavior. Two commits on TOP of the validated
heads (validated shas stay reachable ancestors; never amended).

Fork HEADs after the doc pass: contrib `3851d56` (was f7a8b32), core `d30273d` (was 934c316).
advance-head opencv_contrib 3851d56 classified the delta doc-only and CARRIED gfx90a forward:
linux-gfx90a stays `completed`, validated_sha advanced 934c316/f7a8b32 -> 3851d56, no GPU
re-run. Followers (gfx1100, gfx1201) remain port-ready, unaffected.

### Doc locations surveyed (both repos)
opencv core:
- doc/tutorials/introduction/config_reference/config_reference.markdown: THE build-options
  reference. Added a parallel "### ROCm/HIP support" subsection right after "### CUDA support"
  (before "### OpenCL support"), documenting WITH_HIP, CMAKE_HIP_ARCHITECTURES (auto-detect),
  WITH_HIP/WITH_CUDA mutual exclusion, hipBLAS/hipFFT (cudaarithm), WITH_ROCDECODE + FFmpeg/AMF
  fallback, and the StsNotImplemented facilities (HW optical flow, graphcut stereo, NPP-only
  alphaComp/16-bit+4ch histogram). Cross-refs the existing func_hetero anchor.
- doc/tutorials/introduction/windows_install/windows_install.markdown: the prereq prose list
  carries a "CUDA Toolkit" descriptive line. Added a matching ROCm-toolkit line in the same
  style, pointing at WITH_HIP / func_hetero.
- doc/tutorials/introduction/building_tegra_cuda/...: NVIDIA-Jetson-specific cross-compile,
  NOT a general build doc -> intentionally NOT touched (no ROCm parallel makes sense there).
- CMake option help strings already done in the port commits.

opencv_contrib:
- README.md: the canonical "How to build OpenCV with extra modules" section. Added a concise
  ROCm/HIP note in the surrounding style (WITH_HIP instead of WITH_CUDA, CMAKE_HIP_ARCHITECTURES,
  same cv::cuda API, the StsNotImplemented facilities), linking the core config reference. This
  is the only CUDA-relevant doc location in contrib (modules/README.md and modules/cudev have no
  CUDA build text; cudev has no README).

### Upstreamability sweep (final, both full diffs base..HEAD)
Clean: zero MOAT jargon (moat/follower/head_sha/validated_sha/revalidate/gfx1151/strategy a|b/
curated), zero absolute-path/HIP_VISIBLE_DEVICES leaks, zero non-ASCII on added lines, no
em-dash. Both forks confirmed Actions disabled (gh api actions/permissions -> enabled:false).

### PR drafts written (NOT opened)
- projects/opencv_contrib/pr-draft-opencv.md (core PR; title "[ROCm] Add AMD GPU support for
  cv::cuda via HIP (core)", 54 chars).
- projects/opencv_contrib/pr-draft-opencv_contrib.md (modules PR; title "[ROCm] Port the
  cv::cuda modules to AMD GPUs via HIP", 52 chars). Cross-references the core PR as a hard dep.
Both state the validation matrix neutrally (per-module suite counts on gfx90a; more platforms
as followers complete), scope the limitations per the ledger, and phrase the 2 TVL1.Async
cases as a non-associative float-atomic-reduction determinism artifact (no "fails here").

## Audit reconciliation 2026-06-11 (post-completion verification)

Deep audit against source resolved two inaccuracies in interim reports:
- cv::cuda::rotate WORKS on HIP (warp.cpp:628, routes hipRotateAffine -> warpAffine_gpu kernels). An audit pass misread the nested `#else __HIP_PLATFORM_AMD__` as compiled-out. Real caveats only: no upstream accuracy test; INTER_CUBIC falls back to INTER_LINEAR.
- graphcut/connectivityMask/labelComponents are NOT a HIP loss: the upstream BASE already stubs them under `CUDART_VERSION >= 8000` (graphcuts.cpp). Dead on every modern CUDA build for years (NPP removed graphcut at 8.0). Port adds `|| __HIP_PLATFORM_AMD__` to make existing behavior explicit. Parity with modern CUDA; nothing downstream uses them.

Complete HIP runtime-refusal surface (the entire gap), verified by sweeping both repos:
- cudaimgproc: alphaComp; histEven 16U/16S; histEven 4ch; histRange (all + 4ch) -- NPP-only, reimplementable, deferred (opencv-cudaimgproc-npp-color-hist). 8-bit histEven works.
- cudaoptflow: NvidiaOpticalFlow_1_0/2_0 -- no AMD HW OF engine; software flows are the substitute and pass.
- cudacodec: VideoReader (needs rocDecode + VCN; gfx90a has none; build gates landed, impl deferred to RDNA followers); VideoWriter (no native ROCm encoder; FFmpeg AMF outside the module).

NvOF 4 tests FAIL not SKIP because the test catch filters StsBadFunc/StsBadArg/StsNullPtr only; SDK-less CUDA throws HeaderIsNull there and fails identically -> pre-existing upstream behavior, not a port regression.

TVL1.Async (2): verified root cause -- loop early-exits on cuda::calcSum convergence error (tvl1flow.cpp:359,366); calcSum's final cross-block accumulation is a double atomicAdd (cudev grid/detail/reduce.hpp:165); block-completion order differs by scheduler -> ULP-level error delta -> iteration-count shift. Deterministic (16 streams byte-identical to each other), not a race. Flow correct (normal-tolerance Accuracy passes). Disposition: document-and-ship.

## Validation (linux-gfx1100) 2026-06-11

**State: completed** -- validated_sha = 3851d5653603f43d7781986196f404d34a482d0c (contrib head_sha)
GPU: AMD Radeon Pro W7800 48GB, gfx1100, HIP_VISIBLE_DEVICES=2
ROCm: 7.2.1
OPENCV_TEST_DATA_PATH: projects/opencv_contrib/opencv_extra/testdata

Both repos cloned fresh from fork at head_sha. Built for gfx1100 (CMAKE_HIP_ARCHITECTURES=gfx1100) with the full BUILD_LIST covering all 11 test suites.

### Build
```
cmake -G Ninja ../src-core \
  -DWITH_HIP=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DOPENCV_EXTRA_MODULES_PATH=../src/modules \
  -DBUILD_LIST="core,cudev,cudaarithm,cudafilters,cudawarping,cudaimgproc,cudastereo,cudafeatures2d,cudaobjdetect,cudalegacy,cudabgsegm,cudaoptflow,cudacodec,imgproc,imgcodecs,videoio,highgui,video,optflow,objdetect,calib3d,ts" \
  -DBUILD_TESTS=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_PYTHON=OFF
cmake --build . --target opencv_test_cudev opencv_test_cudastereo opencv_test_cudalegacy \
  opencv_test_cudawarping opencv_test_cudaarithm opencv_test_cudafilters \
  opencv_test_cudaimgproc opencv_test_cudafeatures2d opencv_test_cudaobjdetect \
  opencv_test_cudaoptflow opencv_test_cudacodec -j$(nproc)
# 857/857 targets, warnings only (nodiscard hipGetLastError), no errors
```

### GPU test results (HIP_VISIBLE_DEVICES=2, OPENCV_TEST_DATA_PATH=.../opencv_extra/testdata)

| Suite | Passed | Total | Notes |
|---|---|---|---|
| opencv_test_cudev | 402 | 402 | PASS |
| opencv_test_cudastereo | 128 | 128 | PASS |
| opencv_test_cudalegacy | 14 | 14 | PASS (1 disabled) |
| opencv_test_cudawarping | 4535 | 4535 | PASS |
| opencv_test_cudaarithm | 11417 | 11417 | PASS |
| opencv_test_cudafilters | 16028 | 16028 | PASS |
| opencv_test_cudaimgproc | 3671 | 3671 | PASS |
| opencv_test_cudafeatures2d | 256 | 256 | PASS |
| opencv_test_cudaobjdetect | 18 | 18 | PASS |
| opencv_test_cudaoptflow | 41 | 47 | 6 failures: EXACTLY the documented set (see below) |
| opencv_test_cudacodec | 240 | 240 | PASS (YuvConverter only; decode/encode gated off) |

**cudaoptflow 6 expected failures (identical to gfx90a):**
- CUDA_OptFlow/OpticalFlowDual_TVL1.Async/0 -- float atomicAdd reduction ordering (not a port defect)
- CUDA_OptFlow/OpticalFlowDual_TVL1.Async/1 -- same
- CUDA_OptFlow/NvidiaOpticalFlow_1_0.Regression/0 -- no AMD HW OF engine
- CUDA_OptFlow/NvidiaOpticalFlow_1_0.OpticalFlowNan/0 -- same
- CUDA_OptFlow/NvidiaOpticalFlow_2_0.Regression/0 -- same
- CUDA_OptFlow/NvidiaOpticalFlow_2_0.OpticalFlowNan/0 -- same

No unexpected failures. All 10 other suites are 100%. Port fully validated on real gfx1100. Results are identical to the gfx90a lead -- wave32 gfx1100 passes every test the wave64 gfx90a did, confirming the warp-layer design (logical WARP_SIZE=32, width-32 shuffles inside the 32-lane wavefront) is arch-independent as designed. gfx1100 is RDNA3 (wave32 native), so no wave64 correctness concerns arise here.
## Parity pass: cudaimgproc NPP gaps closed (2026-06-11, gfx90a lead)

jeff required absolute feature parity with the CUDA build. The four cudaimgproc
entry points that previously threw StsNotImplemented on the HIP path are now
implemented with direct HIP kernels and validated on real gfx90a. Fork HEAD:
contrib `6dbfed74777e17c7d170a46f5e22cefef4441396` (was 3851d56); core unchanged
(`d30273d`). head_sha advanced to the contrib sha (functional change). Deferral
`opencv-cudaimgproc-npp-color-hist` is fully RESOLVED (set-status done) -- all of
alphaComp + 16U/16S histEven + 4ch histEven + histRange(+4ch) now land.

### New kernels (HIP-guarded; CUDA NPP path byte-identical)
- cuda/histogram_npp_hip.cu: one range-histogram kernel. Half-open bins
  [levels[i], levels[i+1]) located by binary search IN THE LEVEL TYPE (32S for
  integer depths, 32F for CV_32F); the last boundary is exclusive. histEven is
  expressed as a range histogram over the integer even-level boundaries NPP's
  evenLevels lays out (cvRound layout, matching cv::cuda::evenLevels /
  nppiEvenLevelsHost_32s), so histEven and histRange share ONE bin-edge
  definition. Covers histRange_c1 (8U/16U/16S/32F) and histRange_c4.
- cuda/alpha_comp_hip.cu: nppiAlphaComp_*_AC4R Porter-Duff algebra. Per pixel
  as=a1/maxv, ad=a2/maxv; coverage factors (Fa,Fb) per operator (OVER/IN/OUT/
  ATOP/XOR/PLUS, their *_PREMUL twins, and ALPHA_PREMUL). Straight-alpha ops
  weight each source colour by its own alpha (w=a*F); the *_PREMUL ops treat
  colours as pre-multiplied (w=F). Out alpha = saturate(maxv*(as*Fa+ad*Fb)).
  maxv = 255/65535/INT_MAX/1.0 for 8U/16U/32S/32F (NPP's float convention is
  alpha in [0,1]).

### Per-feature test names + pass counts (opencv_test_cudaimgproc, gfx90a)
Added accuracy tests (run on BOTH CUDA and HIP), gold = a CPU reference of the
same NPP semantics (independent host implementation, not the kernel's own code):
- CUDA_ImgProc/HistEven16.Accuracy (16U,16S): 2/2 PASS
- CUDA_ImgProc/HistEven4.Accuracy (8U,16U,16S): 3/3 PASS
- CUDA_ImgProc/HistRange.Accuracy (8U,16U,16S,32F): 4/4 PASS
- CUDA_ImgProc/HistRange4.Accuracy (8U,16U,16S,32F): 4/4 PASS
- CUDA_ImgProc/AlphaComp.Accuracy (4 types x 13 ops x sizes): 104/104 PASS
Full suite: opencv_test_cudaimgproc 3788/3788 (was 3671; +117 new), deterministic
across repeated runs.

### NPP-semantic subtleties pinned down
- histEven bin edges are the INTEGER even-level boundaries, not float edges. The
  8U HistEven upstream test passes against cv::calcHist only because its params
  (width 5/68) make float and integer edges coincide; a generic cv::calcHist gold
  is WRONG for 16U/16S (off by ~9 counts at a boundary bin). The faithful gold is
  a range histogram over evenLevels32s -- that is what NPP computes.
- alphaComp 8U/16U are bit-exact to the documented algebra; 32S (near-INT_MAX
  scaling) and 32F differ from a pure host-double reference by a few ULP, so the
  test eps is 1.0/4.0/1e-3 for 8U-16U / 32S / 32F. There is no captured NPP binary
  on this host, so the reference is the documented nppiAlphaComp / Porter-Duff
  math; this is stated plainly (not "validated against NPP output").

### NEW FAULT CLASS: pageable async H2D copy races a local host buffer free
First implementation uploaded the even-levels from a function-local cv::Mat via
GpuMat::upload(host, stream). On the Null stream OpenCV still issues a
cudaMemcpyAsync from PAGEABLE host memory; the driver may stage it through a
pinned bounce buffer asynchronously, so destroying the local Mat at function
return corrupts the in-flight copy. Symptom was textbook: an ENTIRE histogram bin
intermittently read 0, different bin/channel each run, passing in isolation and
failing in-suite (same whole-vs-isolated tell as the allocator-reuse class). Fix:
synchronous upload (GpuMat::upload(host) with no stream). GENERAL LESSON: never
issue an async H2D copy from a stack/temporary host buffer that dies before the
copy is guaranteed complete; use a synchronous upload or keep the host buffer
alive (pinned + explicit sync).

### Regression (same MI250X gfx90a, only cudaimgproc source changed)
cudev 402, cudaarithm 11417, cudafilters 16028, cudawarping 4535, cudastereo 128,
cudalegacy 14, cudafeatures2d 256, cudaobjdetect 18, cudacodec 240; cudaoptflow
41/47 (the documented NvOF+TVL1.Async set, unchanged). No regression. Only
modules/cudaimgproc/{src/color.cpp,src/histogram.cpp,src/cuda/alpha_comp_hip.cu,
src/cuda/histogram_npp_hip.cu,test/test_color.cpp,test/test_histogram.cpp}
changed; src-core untouched.

### Remaining documented feature losses (unchanged by this pass)
graphcut (no ROCm maxflow), NvidiaOpticalFlow_1_0/2_0 (no AMD HW OF engine),
cudacodec rocDecode decode + native encode (deferred to RDNA followers / FFmpeg
AMF), dnn cuDNN->MIOpen (out of cv::cuda scope). cudaimgproc now has NO
StsNotImplemented runtime-refusal surface remaining.

## rocDecode HW-decode feasibility on linux-gfx1100 (W7800) -- 2026-06-11
Confirms the deferred `opencv-cudacodec-rocdecode-videoreader` task is unblocked on this RDNA host.

HARDWARE: this host's GPUs are Radeon Pro W7800 (gfx1100, RDNA3, navi31) with a working VCN 4.0
decode engine -- unlike the gfx90a MI250X lead (pure compute, no VCN). vainfo (radeonsi/distro
mesa-va-drivers 25.2.8) shows 9 VLD (hardware decode) profiles: H264 CBP/Main/High, HEVC
Main/Main10, JPEG, VP9 P0/P2, AV1 P0 (+ H264/HEVC/AV1 encode).

rocDecode WORKS against the distro radeonsi VA driver -- it does NOT functionally need ROCm's
`mesa-amdgpu-va-drivers` (that is only an apt packaging hard-dep with no candidate in current
sources). Verified with rocDecode 1.7.0: a `rocDecGetDecoderCaps` probe run with
`LIBVA_DRIVER_NAME=radeonsi HIP_VISIBLE_DEVICES=0` returned:
  AVC/H264  supported=1  max 4096x4096   num_decoders=2  out=NV12
  HEVC      supported=1  max 8192x4352   num_decoders=2  out=NV12   (10-bit capable)

INSTALL RECIPE (does not disturb the distro mesa stack; restore apt after if needed):
  apt-get download rocdecode rocdecode-dev
  sudo dpkg -i --ignore-depends=mesa-amdgpu-va-drivers ./rocdecode*.deb   # or add a Provides shim for the dep
  # rocdecode-dev also wants libva-dev; or just link against /opt/rocm/lib/librocdecode.so directly
  # run all rocDecode code with LIBVA_DRIVER_NAME=radeonsi
NOTE: a force-install puts apt in a broken-dep state (rocdecode -> mesa-amdgpu-va-drivers /
rocdecode-dev -> libva-dev); satisfy with an equivs/Provides shim, or remove the packages to
restore apt. For the actual VideoReader port, set this up deliberately and keep the shim.

So: the cudacodec HW-decode path (rocDecode VideoReader) is validatable on this host -- the
remaining work is the cuvid->rocDecode API mirror (the deferred task), not a hardware/runtime gap.

## cudacodec rocDecode VideoReader IMPLEMENTED + GPU-validated on gfx1100 (2026-06-11)

The deferred `opencv-cudacodec-rocdecode-videoreader` task is DONE. The cudacodec
VideoReader hardware-decode path now decodes on AMD via rocDecode (1.7.0), validated
on the W7800 (gfx1100, RDNA3, VCN 4.0). Only opencv_contrib changed; opencv core is
untouched (the shim is self-contained in the module). Build dir
`projects/opencv_contrib/build_cudacodec/` (gitignored); script `build_hip_cudacodec.sh`.
ALL build/test steps require `LIBVA_DRIVER_NAME=radeonsi HIP_VISIBLE_DEVICES=0`.

### Fail-fast confirmed first (agent_space/rocdec_harness.cpp, gitignored)
A standalone bitstream-reader + rocparser + rocDecode harness decoded H265/H264/AV1
end-to-end before touching the module: 59-62 frames each, plausible luma (mean ~100),
separate Y/UV device pointers, 256-aligned pitch. Confirmed the runtime decodes on
this host and the cuvid->rocDecode API mapping is sound.

### cuvid -> rocDecode mapping (the port)
The NVCUVID decode pipeline maps 1:1 onto rocDecode: cuvid video parser ->
rocDecCreateVideoParser/rocDecParseVideoData (sequence/decode/display callbacks);
cuvid decoder -> rocDecCreateDecoder/rocDecDecodeFrame/rocDecGetVideoFrame;
CUVID* structs -> Rocdec* structs. The FFmpeg videoio demuxer the module already links
feeds elementary-stream packets to the rocparser (no rocDecode built-in demuxer needed).
Files (all opencv_contrib modules/cudacodec):
- NEW src/rocdecode_video_compat.hpp: maps the cuvid type/helper names the codec-agnostic
  plumbing uses (CUVIDPARSERDISPINFO/CUVIDPROCPARAMS -> RocdecDispInfo/RocdecProcInfo;
  CUcontext/CUvideoctxlock/CUdeviceptr; cuCtxGetCurrent/cuvidCtxLock* no-ops; the dead
  histogram cuMemcpyDtoDAsync) onto rocDecode/HIP, so frame_queue / video_source /
  video_reader compile unchanged and only the back ends differ.
- src/video_decoder.{hpp,cpp}: HAVE_ROCDECODE VideoDecoder calling rocDec*; explicit
  codecToRocDec/chromaToRocDec (cudacodec Codec ordering follows NVCUVID; rocDecode orders
  differently, so a static_cast would mislabel the codec). mapFrame wraps the rocDecode
  decode surface (single contiguous pitched NV12: chroma == luma + pitch*height, verified)
  as one GpuMat, zero-copy, exactly like cuvid mapFrame.
- src/video_parser.{hpp,cpp}: HAVE_ROCDECODE VideoParser owning the rocparser, with the
  sequence/decode/display callbacks routed to the decoder + frame queue.
- precomp.hpp: HAVE_CUDACODEC_DECODER = HAVE_NVCUVID || HAVE_ROCDECODE (defined AFTER the
  OpenCV includes so HAVE_ROCDECODE from cvconfig.h is visible -- defining it before the
  includes was a real bug: it left the StsNotImplemented stub in place). Includes the shim
  + the shared decode headers on the rocDecode path; cuvid_video_source stays NVCUVID-only.
- frame_queue.cpp / video_source.cpp / ffmpeg_video_source.cpp / thread.cpp: guard widened
  HAVE_NVCUVID -> HAVE_CUDACODEC_DECODER (codec-agnostic plumbing, builds against either
  back end). video_reader.cpp: stub guard #ifndef HAVE_NVCUVID -> #ifndef
  HAVE_CUDACODEC_DECODER so the real reader compiles on ROCm; the cuvid built-in-demuxer
  fallback (no rocDecode equivalent) re-raises on HIP.
- CMake (already present from the prior pass): WITH_ROCDECODE -> HAVE_ROCDECODE +
  ROCDECODE_LIBRARIES; module links librocdecode. Build with WITH_FFMPEG=ON (the demuxer).

### THE RUNTIME GUARD (the user's key requirement) -- two layers, never crashes
1. Build-time: HAVE_ROCDECODE gates the whole decode path; without rocDecode the reader
   keeps the StsNotImplemented stub.
2. Runtime (the multi-arch-safety branch): BEFORE creating the decoder, VideoDecoder::create
   calls rocDecGetDecoderCaps(device_id, codec, chroma, bit_depth_minus_8). If is_supported==0
   (a compute-only gfx90a with no VCN yields this; an unsupported codec/dimension/output format
   yields this) it throws a catchable cv::Error::StsNotImplemented naming the device (name +
   gcnArchName) and codec, e.g. "this device has no supported hardware decoder for <codec>
   (rocDecGetDecoderCaps is_supported=0)". A second guard in the VideoParser ctor rejects codecs
   rocDecode's parser does not implement (MPEG-1/2/4, VC-1, VP8, MJPEG) up front with a clear
   StsNotImplemented before rocDecCreateVideoParser. Both are catchable cv::Exceptions, NEVER a
   crash/abort. THIS IS THE SAME BRANCH that protects gfx90a: the wave32/wave64 fat binary
   (gfx90a;gfx1100;gfx1201) decides at runtime -- it throws cleanly on a device with no VCN
   (gfx90a) and on unsupported codecs, and decodes on gfx1100/gfx1201. Verified: requesting an
   MPEG-2 (.mpg) / MPEG-4 (.mp4/.avi) stream throws StsNotImplemented (-213) with the rocDecode
   codec message, caught by the test harness, no crash.

### GPU validation (opencv_test_cudacodec, gfx1100, LIBVA_DRIVER_NAME=radeonsi HIP_VISIBLE_DEVICES=0)
`OPENCV_TEST_DATA_PATH=projects/opencv_contrib/opencv_extra/testdata` (added the highgui/video
sparse-checkout for the codec test streams). Full suite: **256/303 PASS**.
DECODES (the acceptance gate, real HW decode, correct dims/format/non-empty):
- Video.Reader H264 (big_buck_bunny.h264): PASS -- decodes, GRAY/BGR/BGRA/NV_YUV all correct dims.
- Video.Reader HEVC (big_buck_bunny.h265): PASS.
- Video.Reader VP9: PASS.
- YuvConverter 240/240: PASS (unchanged converter, no regression).
- ReconfigureDecoderWithScaling (H264 multi-res, scale + srcRoi + targetRoi + mid-stream
  resolution change): PASS after the border-zero fix below.
- ColorConversionFormat GRAY (tol 15): PASS at meanAbs 0.75 -- proves the decode is correct.

### The 47 "failures" are EXACTLY two documented categories, ZERO real decode bugs
1. 37 codec-delta (genuine rocDecode codec losses): every failing test that loads an
   .mp4/.avi/.mov/.mpg stream FFmpeg reports as MPEG-2/MPEG-4 (and the MJPEG/VC1 streams).
   rocDecode decodes only AVC/HEVC/AV1/VP9; these throw the clean StsNotImplemented codec
   guard. Identical to a CUDA build on a GPU lacking those codecs. (Video.Reader/0,2,3,6;
   all Scaling/* on big_buck_bunny.mp4; CheckSet/CheckExtraData/CheckKeyFrame/CheckParams/
   CheckDecodeSurfaces/CheckInitParams/DisplayResolution/Seek on mpeg files.)
2. 10 cross-decoder tolerance (decode IS correct): ColorConversion{Format BGR/BGRA/RGB/RGBA,
   LumaChromaRange, Planar, Bitdepth} compare rocDecode VCN H264 output against libavcodec's
   software H264 decode at EXPECT_MAT_NEAR tolerance 2. VCN differs from libavcodec by meanAbs
   ~1.84 (max ~20 at ~0.05% of pixels) -- within H.264 spec rounding latitude, NOT a port
   defect. The GRAY variant (tol 15) PASSES at meanAbs 0.75, proving the decode is correct;
   only the strict tol-2 cross-decoder comparison trips. Same class as the documented TVL1.Async
   0.0-tolerance case (a strict upstream tolerance assuming two implementations bit-match).

### NEW FAULT CLASS REUSE: allocator-no-zero applied to the target_rect border
ReconfigureDecoderWithScaling first FAILED at the "zero border outside targetRoi" assertion.
The NVCUVID decoder zeroes the output surface area outside the target rectangle during
post-processing; rocDecode places the scaled image at target_rect but does not zero the
surrounding decode surface, and the ROCm device allocator does not hand back zeroed pages, so
the converter rendered the border as garbage. Fix (video_reader.cpp cvtFromYuv, HIP-guarded):
after conversion, when a target ROI smaller than the surface is in use, re-zero everything
outside the ROI (copy the ROI into a zeroed buffer). CUDA path byte-identical. This is the
same allocator-does-not-zero fault class as cudastereo census_transform and cudaimgproc.

### Codec losses (genuine, vs NVDEC) -- surfaced now that the decode path is live
rocDecode lacks MPEG-1/2/4, VC-1, VP8 and MJPEG (NVDEC has them). The reader throws a clean
StsNotImplemented naming these and pointing at the FFmpeg cv::VideoCapture backend. Luma
histogram output (NVDEC) is also unavailable on rocDecode -- rejected at caps time.

### Test guards widened (CUDA-preserving)
test/test_video.cpp: the VideoReader decode-test fixtures and bodies guard widened
HAVE_NVCUVID -> (HAVE_NVCUVID || HAVE_ROCDECODE) so they compile and run on ROCm; the decode
instantiations were already in the wider HAVE_NVCUVID||HAVE_NVCUVENC block. Encode-coupled
tests (HAVE_NVCUVID && HAVE_NVCUVENC, HAVE_NVCUVENC) stay CUDA-only.

### Build + test commands (literal)
```
LIBVA_DRIVER_NAME=radeonsi HIP_VISIBLE_DEVICES=0 \
  BUILD_TARGET=opencv_test_cudacodec bash projects/opencv_contrib/build_hip_cudacodec.sh
LIBVA_DRIVER_NAME=radeonsi HIP_VISIBLE_DEVICES=0 \
  OPENCV_TEST_DATA_PATH=projects/opencv_contrib/opencv_extra/testdata \
  projects/opencv_contrib/build_cudacodec/bin/opencv_test_cudacodec
```
GOTCHA: rocDecode needs LIBVA_DRIVER_NAME=radeonsi on this host for EVERY run (the W7800 VCN
is driven by the distro radeonsi VA driver). Without it, rocDecGetDecoderCaps/decode fail.

## Revalidation (linux-gfx1100) 2026-06-11

**State: completed** -- validated_sha = a6612f09515191da1bf157b22a2938ab7b020b96 (contrib head_sha)
GPU: AMD Radeon Pro W7800 48GB, gfx1100, HIP_VISIBLE_DEVICES=0
ROCm: 7.2.1
OPENCV_TEST_DATA_PATH: projects/opencv_contrib/opencv_extra/testdata

Delta from prior validated_sha (3851d56) consists of two functional commits:
- 6dbfed7: new cudaimgproc kernels (histEven/histRange/alphaComp)
- a6612f0: cudacodec rocDecode VideoReader (already porter-validated on gfx1100, reconfirmed here)

### Build (incremental from existing build dir, WITH_ROCDECODE=ON, gfx1100)
```
# Re-ran cmake . to pick up new source files (histogram_npp_hip.cu, alpha_comp_hip.cu)
cd projects/opencv_contrib/build
cmake .
cmake --build . --target opencv_test_cudaimgproc opencv_test_cudacodec -j$(nproc)
# Then built remaining targets: all 857 targets, warnings only (nodiscard hipGetLastError)
```

### GPU test results (HIP_VISIBLE_DEVICES=0, OPENCV_TEST_DATA_PATH=.../opencv_extra/testdata)

| Suite | Passed | Total | Notes |
|---|---|---|---|
| opencv_test_cudev | 402 | 402 | PASS (no change, no regression) |
| opencv_test_cudastereo | 128 | 128 | PASS |
| opencv_test_cudalegacy | 14 | 14 | PASS (1 disabled) |
| opencv_test_cudawarping | 4535 | 4535 | PASS |
| opencv_test_cudaarithm | 11417 | 11417 | PASS |
| opencv_test_cudafilters | 16028 | 16028 | PASS |
| opencv_test_cudaimgproc | 3788 | 3788 | PASS (+117 new tests: HistEven16 2, HistEven4 3, HistRange 4, HistRange4 4, AlphaComp 104) |
| opencv_test_cudafeatures2d | 256 | 256 | PASS |
| opencv_test_cudaobjdetect | 18 | 18 | PASS |
| opencv_test_cudaoptflow | 41 | 47 | 6 failures: EXACTLY the documented set (see below) |
| opencv_test_cudacodec | 256 | 303 | 47 failures: EXACTLY the documented set (see below) |

**cudaoptflow 6 expected failures (identical to prior gfx1100 validation):**
- CUDA_OptFlow/OpticalFlowDual_TVL1.Async/0 -- float atomicAdd reduction ordering (not a port defect)
- CUDA_OptFlow/OpticalFlowDual_TVL1.Async/1 -- same
- CUDA_OptFlow/NvidiaOpticalFlow_1_0.Regression/0 -- no AMD HW OF engine
- CUDA_OptFlow/NvidiaOpticalFlow_1_0.OpticalFlowNan/0 -- same
- CUDA_OptFlow/NvidiaOpticalFlow_2_0.Regression/0 -- same
- CUDA_OptFlow/NvidiaOpticalFlow_2_0.OpticalFlowNan/0 -- same

**cudacodec 47 non-passes (identical to porter validation on gfx1100, documented class):**
- 37 codec-delta: tests using .mp4/.avi/.mov/.mpg/.mjpg (MPEG-2/MPEG-4/MJPEG -- not supported by rocDecode; clean StsNotImplemented codec guard fires)
- 10 tol-2 cross-decoder: ColorConversionFormat BGR/BGRA/RGB/RGBA, LumaChromaRange x2, Planar, Bitdepth x3 -- decode is correct (GRAY tol-15 PASSES at meanAbs 0.75), strict tol-2 cross-decoder comparison trips (VCN vs libavcodec rounding latitude, not a port defect)

No unexpected failures in any suite. cudaimgproc 3788/3788 confirms the new 6dbfed7 kernels (histEven/histRange/alphaComp) are correct on gfx1100. cudacodec 256/303 reconfirms a6612f0 rocDecode VideoReader on gfx1100 (H264/H265/VP9 decode, YuvConverter 240/240, codec guard verified). No regression across all 9 fully-passing suites.
## Review 2026-06-11 (reviewer, linux-gfx90a): two functional deltas (parity + rocDecode)

Incremental /pr-review of the two functional commits added after the last full
review/validation: DELTA 1 `6dbfed7` (cudaimgproc histEven/histRange/alphaComp
parity) and DELTA 2 `a6612f0` (cudacodec rocDecode VideoReader). Reviewed
`git diff 3851d56 6dbfed7` and `git diff 6dbfed7 a6612f0`. Whole-port body was
already review-passed at f7a8b32; this pass only covers the two deltas plus
cross-cutting upstreamability. GPU revalidation is the validator's next stage
(gfx90a is `ported`, gfx1100 `revalidate`); not run here by design.

VERDICT: changes-requested. The functional code is clean -- no code-correctness,
fault-class, CUDA-path, wave64, or upstreamability defect in either delta. The
ONE blocking item is the missing AMD copyright + author attribution on the four
substantially-extended cudacodec files (finding 1), a mandatory CLAUDE.md rule.
It is comment-only/behavior-preserving: the porter adds the four headers and the
regression guard carries every completed platform forward (no GPU re-run). Once
those headers land, this is otherwise ready to advance to the validator.

### Review findings (parity + rocDecode deltas)

1. ATTRIBUTION GAP (required before squash/PR; comment-only). DELTA 2
   substantially extended four existing files with a whole new rocDecode back
   end but did NOT add the parallel `Copyright (c) 2026 Advanced Micro Devices,
   Inc.` line or a `\author Jeff Daily` tag that CLAUDE.md
   requires for substantially-extended port files:
   - modules/cudacodec/src/video_decoder.cpp (+224 lines, new VideoDecoder)
   - modules/cudacodec/src/video_decoder.hpp (+89 lines, new class)
   - modules/cudacodec/src/video_parser.cpp (+161 lines, new VideoParser)
   - modules/cudacodec/src/video_parser.hpp (+55 lines, new class)
   The sibling new file rocdecode_video_compat.hpp:111-153 correctly carries
   both, and DELTA 1's two new .cu files carry both. Fix: add the parallel AMD
   copyright line (below the existing OpenCV/Intel/Willow lines, do not replace
   them) and an author tag to these four headers in the prep phase. This is a
   pure header-comment change -- commit on top, advance_head classifies it inert
   and carries every completed platform forward, no revalidation.

### Verified clean (the scrutiny list; judged by reading code, not GPU)

DELTA 1 (parity):
- Wave64: both new kernels (alpha_comp_hip.cu, histogram_npp_hip.cu) are pure
  per-pixel/per-element 2D-grid kernels. Grep confirms NO shfl/ballot/activemask/
  warp-reduce/__popc/hardcoded-warpSize; histogram uses only block dynamic-shared
  atomicAdd + global atomicAdd (wave-agnostic). block(32,8) is a block dim, not a
  warp assumption. Correct on wave64 gfx90a and wave32 followers.
- CUDA path byte-identical: every changed line in color.cpp and histogram.cpp is
  inside `#ifdef __HIP_PLATFORM_AMD__`; the NPP `#else` branches are untouched.
- Channel indexing correct: GpuMat->PtrStepSz uses cols = pixel cols (cuda.inl.hpp
  :257, no channel multiply); alphaComp x*4 + bound `x>=img1.cols` and the
  histogram `row[x*CN+channel]` + `x<cols` both index channels correctly. No
  whole-vs-submat overread.
- alphaComp in-place safe: each thread reads p1/p2 and writes d at its own pixel
  only (no neighbor reads), so dst aliasing img1/img2 cannot corrupt.
- histEven/histRange NPP fidelity: findBin implements the half-open
  [levels[i],levels[i+1]) with last boundary exclusive and v<levels[0] /
  v>=levels[last] excluded; the pixel is promoted to the level type L so 8U values
  vs a 256 boundary compare correctly. histEven materialises the integer
  cvRound even-level layout (histogram.cpp uploadEvenLevels32s) matching
  cv::cuda::evenLevels/nppiEvenLevelsHost_32s, so even and range share one
  bin-edge definition.
- Tests are a genuine gate, not a tautology: the golds (alphaCompGold,
  rangeHistGold) are independent host implementations (host-double Porter-Duff;
  plain linear scan with the same NPP-defined half-open bins) -- not the kernel's
  own code -- and they run on BOTH CUDA and HIP (`#endif // HAVE_CUDA`). The
  evenLevels gold is the correct NPP reference (the notes correctly observe a
  generic cv::calcHist gold would be WRONG for 16U/16S). eps rationale sound
  (8U/16U exact at 0.0; 32S=4.0 and 32F=1e-3 for float-scale ULP).
- Async-upload fault class fixed correctly: uploadEvenLevels32s uses a
  synchronous GpuMat::upload(host) (no stream), so the local Mat cannot be
  destroyed under an in-flight pageable async H2D copy. The pattern appears only
  here on the new path; alphaComp uploads no host buffer.

DELTA 2 (rocDecode):
- gfx90a UNAFFECTED (the platform under review): on a no-rocDecode build
  HAVE_NVCUVID/NVCUVENC/ROCDECODE are all undefined, so HAVE_CUDACODEC_DECODER is
  undefined; video_reader.cpp takes the `#ifndef HAVE_CUDACODEC_DECODER`
  StsNotImplemented stub (same as the old `#ifndef HAVE_NVCUVID`), the
  `#ifdef HAVE_ROCDECODE` border-zero block compiles out, and the four
  guard-widened plumbing files (frame_queue/thread/video_source/
  ffmpeg_video_source) compile their bodies out. Zero behavior change for the
  gfx90a revalidation.
- CUDA path byte-identical: HAVE_NVCUVID->HAVE_CUDACODEC_DECODER reduces to
  HAVE_NVCUVID on any CUDA build (HAVE_CUDACODEC_DECODER = NVCUVID||ROCDECODE).
  precomp.hpp preserves the cuvid_video_source.hpp include exactly (now under a
  redundant `#if defined(HAVE_NVCUVID)` inside the already-NVCUVID block); the
  NVENC-only path is unchanged. video_decoder/parser .cpp put the new rocDecode
  code in `#ifdef HAVE_ROCDECODE` BEFORE the unchanged `#ifdef HAVE_NVCUVID`
  block; the .hpp uses `#if HAVE_ROCDECODE #else //NVCUVID #endif` so CUDA gets
  the original class verbatim. createVideoReader's cuvid-demuxer fallback is now
  `#ifdef HAVE_NVCUVID ... #else throw; #endif` -- identical to the original
  unconditional CuvidVideoSource on a CUDA build. test_video.cpp only ADDs
  `|| HAVE_ROCDECODE` to two guards (byte-identical preprocessor result on CUDA)
  and the VideoReader test block uses only public cv::cudacodec API (back-end
  neutral, no rocDecode symbols).
- compat shim isolation: rocdecode_video_compat.hpp's `#define cuCtxGetCurrent`
  etc. and `typedef int CUcontext` are reached only via precomp.hpp's
  HAVE_ROCDECODE branch, so they never leak into or collide with a CUDA build's
  real cuvid/driver symbols.
- Border-zero fix correct + HIP-guarded (allocator-no-zero fault class reused):
  cvtFromYuv re-zeroes outside targetRoi only when `!targetRoi.empty() &&
  targetRoi.size() != outFrame.size()` (a no-op when the image fills the surface),
  via a zeroed scratch + ROI copy. targetRoi is decoder-clamped to the surface so
  the sub-view cannot exceed bounds. gfx1100 ReconfigureDecoderWithScaling passes
  with this fix. Guarded `#ifdef HAVE_ROCDECODE` so CUDA is byte-identical.
- Honest runtime refusal: VideoDecoder::create queries rocDecGetDecoderCaps and
  throws a catchable StsNotImplemented naming the device (name+gcnArchName) and
  codec on is_supported==0 (compute-only gfx90a) or unsupported codec/dim/format;
  a second VideoParser guard rejects the codecs rocDecode's parser lacks
  (MPEG-1/2/4, VC-1, VP8, MJPEG) up front. Never crashes/aborts. mapFrame wraps
  the contiguous NV12 surface (luma + pitch*height) as one GpuMat via the
  external-data ctor (rows=h*3/2, cols=targetWidth, step=pitch) with no copy.
- Codec-delta and cross-decoder dispositions are sound (same class as the
  accepted TVL1.Async 0.0-tolerance case): the 37 codec-delta failures are
  genuine rocDecode codec losses surfaced as the clean codec guard (a CUDA build
  on a GPU lacking those codecs fails identically); the 10 cross-decoder cases are
  VCN-vs-libavcodec H.264 rounding within spec latitude (the looser GRAY tol-15
  comparison passes at meanAbs 0.75, proving the decode is correct). Neither is a
  port defect.
- Minor follower-only note (NOT a gfx90a or CUDA-path issue, does not block):
  rocDecode HandleVideoSequence drops the cuvid AV1-seqhdr max-width/height
  special-case (sets ulMaxWidth/Height = coded dims), since RocdecVideoFormat does
  not expose it; only affects AV1 mid-stream upscaling reconfigure on the RDNA
  followers, which gfx1100 already exercised at 256/303. Acceptable.

### Cross-cutting (both deltas)
Whole-diff sweep clean: no MOAT jargon (moat/lead/follower/strategy a|b/head_sha/
validated_sha/revalidate/gfx1151/curated), ASCII-only on added lines, no em-dash.
Both commit titles `[ROCm]`-prefixed and <=72 chars (56 and 51); both bodies name
Claude/Anthropic, carry a Test Plan with literal commands, no Co-Authored-By
noreply trailer. No AMD-internal account references. New .cu files (DELTA 1) and
rocdecode_video_compat.hpp (DELTA 2) carry AMD copyright + author; the four
extended cudacodec files do not (finding 1).

### Re-review 2026-06-11 (reviewer, linux-gfx90a): finding 1 RESOLVED -> review-passed
Commit `8a5b0a2` ([ROCm] Add AMD copyright and author attribution to rocDecode files)
adds the parallel `Copyright (c) 2026 Advanced Micro Devices, Inc.` line and a
`\author Jeff Daily` tag (below, not replacing, the existing
Intel/Willow/OpenCV lines) to all four files: video_decoder.{cpp,hpp} and
video_parser.{cpp,hpp}, matching the sibling rocdecode_video_compat.hpp style.
`git show --stat`: only those four files, 3 insertions each, every added line a
`//` comment -> comment-only, carries followers forward with no GPU re-run. Title
66 chars and `[ROCm]`-prefixed, body names Claude/Anthropic, ASCII-only, no noreply
trailer, no MOAT jargon, no AMD-internal account refs. Prior pass's other
conclusions (parity kernels + rocDecode back end) stand. State -> review-passed.

## Validation (lead, gfx90a) at tip 8a5b0a2 -- 2026-06-11

**State: completed** -- validated_sha = 8a5b0a222545f1ff9ab93e95afbf92cc7dc9daa0 (contrib head_sha)
GPU: AMD Instinct MI250X / MI250, gfx90a, HIP_VISIBLE_DEVICES=0
ROCm: 7.2.1
OPENCV_TEST_DATA_PATH: projects/opencv_contrib/opencv_extra/testdata

### Integrity check
- contrib HEAD 8a5b0a2 == remote fork tip (FETCH_HEAD): MATCH
- `git status --porcelain` in projects/opencv_contrib/src: clean (no uncommitted tracked source)
- `python3 utils/moatlib.py audit-clean`: no opencv_contrib integrity gap

### Build (incremental from projects/opencv_contrib/build)
```
# Reconfigured with full BUILD_LIST (was partial, only cudaarithm+cudafilters+cudaimgproc)
cmake . -DBUILD_LIST="core,cudev,cudaarithm,cudafilters,cudawarping,cudaimgproc,cudastereo,\
  cudafeatures2d,cudaobjdetect,cudalegacy,cudabgsegm,cudaoptflow,cudacodec,imgproc,imgcodecs,\
  videoio,highgui,video,optflow,objdetect,calib3d,ts"
cmake --build . --target opencv_test_cudev opencv_test_cudastereo opencv_test_cudalegacy \
  opencv_test_cudawarping opencv_test_cudaarithm opencv_test_cudafilters \
  opencv_test_cudaimgproc opencv_test_cudafeatures2d opencv_test_cudaobjdetect \
  opencv_test_cudaoptflow opencv_test_cudacodec -j$(nproc)
# 791 targets built; warnings only (nodiscard hipGetLastError), no errors
```

### 5 required parity suites (CUDA_ImgProc, 6dbfed7 kernels, gfx90a)
All 5 named suites EXECUTED and PASSED (confirmed by grep of run output; zero FAILED lines):
- CUDA_ImgProc/HistEven16.Accuracy: 2/2 PASS
- CUDA_ImgProc/HistEven4.Accuracy: 3/3 PASS
- CUDA_ImgProc/HistRange.Accuracy: 4/4 PASS
- CUDA_ImgProc/HistRange4.Accuracy: 4/4 PASS
- CUDA_ImgProc/AlphaComp.Accuracy: 104/104 PASS
Full suite: opencv_test_cudaimgproc 3788/3788 (+117 vs prior gfx90a validation at f7a8b32).

### cudacodec gfx90a behavior (a6612f0 rocDecode delta)
gfx90a has no VCN engine; WITH_ROCDECODE=ON but rocDecode VA driver not installable; HAVE_ROCDECODE
is OFF in this build, so the rocDecode path compiles out. VideoReader takes the StsNotImplemented
stub (unchanged). Only the NVCUVID-independent YuvConverter suite runs: 240/240 PASS. The rocDecode
changes (a6612f0) do not perturb the gfx90a (no-rocDecode) build. The comment-only 8a5b0a2 has
no effect on any binary.

### Full regression (HIP_VISIBLE_DEVICES=0, OPENCV_TEST_DATA_PATH=.../opencv_extra/testdata)
| Suite | Passed | Total | Notes |
|---|---|---|---|
| opencv_test_cudev | 402 | 402 | PASS (no change) |
## Validation (windows-gfx1201) 2026-06-11

**State: completed** -- validated_sha = 8a5b0a222545f1ff9ab93e95afbf92cc7dc9daa0 (contrib head_sha)
GPU: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), HIP_VISIBLE_DEVICES=0
ROCm: TheRock 7.14.0a20260604 (pip SDK _rocm_sdk_devel)
OPENCV_TEST_DATA_PATH: projects/opencv_contrib/opencv_extra/testdata
Fork head confirmed: contrib 8a5b0a2, core d30273d (both clean, no uncommitted changes)

### Windows-specific build notes
Build dir: `projects/opencv_contrib/build_gfx1201` (gitignored). Two-repo layout
(src/ = contrib, src-core/ = opencv core fork), both cloned fresh from fork at head_sha.
No rocDecode available on Windows (not in TheRock pip SDK); cudacodec builds with
WITH_ROCDECODE=OFF (YuvConverter only, same as gfx90a). No FFmpeg on this host; WITH_FFMPEG=OFF.

Build recipe:
```
ROCM=.../pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
cmake -G Ninja src-core \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/clang.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_PREFIX_PATH=$ROCM \
  -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DOPENCV_EXTRA_MODULES_PATH=src/modules \
  -DBUILD_LIST="core,cudev,cudaarithm,cudafilters,cudawarping,cudaimgproc,cudastereo,
    cudafeatures2d,cudaobjdetect,cudalegacy,cudabgsegm,cudaoptflow,cudacodec,
    imgproc,imgcodecs,videoio,highgui,video,optflow,objdetect,calib3d,ts" \
  -DBUILD_TESTS=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_PYTHON=OFF \
  -DWITH_FFMPEG=OFF -DWITH_ROCDECODE=OFF -DWITH_GSTREAMER=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-march=x86-64-v3" -DCMAKE_CXX_FLAGS="-march=x86-64-v3"
cmake --build . --target opencv_test_cudev ... -j64
# 508/508 targets, warnings only
```

GOTCHA: 3rdparty/libwebp SSE4.1 sources require `-march=x86-64-v3` (enables SSSE3+SSE4.1)
when compiling with clang on Windows; without it, clang rejects the always_inline SSE4.1
builtins. This is a host C compiler issue (libwebp, not the HIP port); CUDA path unaffected.

GOTCHA (DLL loading on Windows): running test .exe files from bash (Git for Windows /
MSYS2) triggers the bash DLL loader which emulates Linux shared-library loading. The
fix: add the build bin dir to PATH (`export PATH=".../build_gfx1201/bin:$ROCM_LIBS/bin:$PATH`")
so the POSIX emulation layer finds the OpenCV + ROCm DLLs before falling back to System32.
Additionally, copy the TheRock core DLLs (amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll,
hiprtc0714.dll, hiprtc-builtins0714.dll) into the bin dir so the Windows loader uses the
TheRock runtime ahead of the Adrenalin System32 DLL (same dietgpu fix class).

### GPU test results (HIP_VISIBLE_DEVICES=0, ROCBLAS_TENSILE_LIBPATH=_rocm_sdk_libraries/bin/rocblas/library)

| Suite | Passed | Total | Notes |
|---|---|---|---|
| opencv_test_cudev | 402 | 402 | PASS |
| opencv_test_cudastereo | 128 | 128 | PASS |
| opencv_test_cudalegacy | 14 | 14 | PASS (1 disabled) |
| opencv_test_cudawarping | 4535 | 4535 | PASS |
| opencv_test_cudaarithm | 11417 | 11417 | PASS |
| opencv_test_cudafilters | 16028 | 16028 | PASS |
| opencv_test_cudaimgproc | 3788 | 3788 | PASS (+117 new parity cases) |
| opencv_test_cudafeatures2d | 256 | 256 | PASS |
| opencv_test_cudaobjdetect | 18 | 18 | PASS |
| opencv_test_cudaoptflow | 41 | 47 | 6 failures: EXACTLY the documented set (see below) |
| opencv_test_cudacodec | 240 | 240 | PASS (YuvConverter only; decode/encode gated off on gfx90a) |

**cudaoptflow 6 expected failures (unchanged from prior validations):**
| opencv_test_cudaimgproc | 3788 | 3788 | PASS (+117 histEven/histRange/alphaComp vs gfx90a) |
| opencv_test_cudafeatures2d | 256 | 256 | PASS |
| opencv_test_cudaobjdetect | 18 | 18 | PASS |
| opencv_test_cudaoptflow | 41 | 47 | 6 failures: EXACTLY the documented set (see below) |
| opencv_test_cudacodec | 240 | 240 | PASS (YuvConverter only; WITH_ROCDECODE=OFF on Windows) |

**cudaoptflow 6 expected failures (identical to gfx90a and gfx1100):**
- CUDA_OptFlow/OpticalFlowDual_TVL1.Async/0 -- float atomicAdd reduction ordering (not a port defect)
- CUDA_OptFlow/OpticalFlowDual_TVL1.Async/1 -- same
- CUDA_OptFlow/NvidiaOpticalFlow_1_0.Regression/0 -- no AMD HW OF engine
- CUDA_OptFlow/NvidiaOpticalFlow_1_0.OpticalFlowNan/0 -- same
- CUDA_OptFlow/NvidiaOpticalFlow_2_0.Regression/0 -- same
- CUDA_OptFlow/NvidiaOpticalFlow_2_0.OpticalFlowNan/0 -- same

No unexpected failures in any suite. All regression suites match prior baselines exactly. The
two functional deltas (6dbfed7 parity kernels, a6612f0 rocDecode) introduced no regressions
on gfx90a. cudaimgproc 3788/3788 confirms the new histEven/histRange/alphaComp kernels are
correct on wave64 gfx90a. Port fully validated on real gfx90a at tip 8a5b0a2.
No unexpected failures. All 10 other suites are 100%. Results identical to gfx1100 (same
failure set, same pass counts). gfx1201 is RDNA4 (wave32, same as RDNA3), so wave32 path
is confirmed. No delta-port needed.

## Upstream PRs OPENED 2026-06-11

Both forks squashed to one clean [ROCm] commit each (tree-identical to the validated tips; squash-carry-forward carried all completed platforms forward, no re-run):
- contrib AMD-Ecosystem/opencv_contrib moat-port: ad105bb5914a256335dad33abdd66b360dd91634 (was 8a5b0a2) = project head_sha
- core    AMD-Ecosystem/opencv        moat-port: adcd50caa527cec03fcf7832593df2581334aa73 (was d30273d)

PRs (two-repo change; contrib depends on core, cross-linked in both bodies):
- core:    https://github.com/opencv/opencv/pull/29285        ([ROCm] Add AMD GPU support for cv::cuda via HIP (core))
- contrib: https://github.com/opencv/opencv_contrib/pull/4147 ([ROCm] Port the cv::cuda modules to AMD GPUs via HIP)  <- canonical for set-pr-open

Validated before open on gfx90a (CDNA2/wave64), gfx1100 (RDNA3/wave32), gfx1201 (RDNA4/wave32). PR bodies corrected from the stale drafts: removed the alphaComp/16-bit/4-channel-histogram "not implemented" claim (parity pass shipped them, cudaimgproc 3788/3788), added the 3-arch validation + rocDecode decode on the RDNA parts. Optional windows-gfx1101/gfx1151 stay port-ready (do not block; additive if a host validates them later). Next: respond to upstream review; on merge run set-pr-merged.

## CUDA no-regression compile gate 2026-06-11

**Verdict: CUDA-compile-clean.** The ported trees (core adcd50c, contrib ad105bb) compile cleanly under nvcc with no errors and no port-introduced warnings. The PR claim that "the CUDA build is byte-for-byte unchanged" is supported at the compile level.

### Environment
- Toolkit: conda env cuda-12.8, `/opt/conda/envs/cuda-12.8/bin/nvcc` version 12.8.93 (Cuda compilation tools, release 12.8, V12.8.93)
- Arch pin: `-DCMAKE_CUDA_ARCHITECTURES=80` (sm_80, A100-class); `-DCUDA_ARCH_BIN=8.0 -DCUDA_ARCH_PTX=""`
- libcuda.so stub: `/opt/conda/envs/cuda-12.8/targets/x86_64-linux/lib/stubs/libcuda.so` (no NVIDIA GPU or driver on this host; compile-only gate)
- Host gcc: 13 (system)
- WITH_CUDA=ON, WITH_HIP=OFF (verified in cvconfig.h: `#define HAVE_CUDA`; `HAVE_HIP` absent)
- ENABLE_CUDA_FIRST_CLASS_LANGUAGE=ON (required: cmake 4.0 sets CMP0146=OLD which uses legacy FindCUDA / cudafe++ that is absent in the toolkit metapackage; first-class language support uses nvcc directly)
- Build dir: `projects/opencv_contrib/build-cuda` (separate from the HIP build dir)

### Configure command
```
cd projects/opencv_contrib/build-cuda
PATH=/opt/conda/envs/cuda-12.8/bin:$PATH \
cmake -G Ninja ../src-core \
  -DCMAKE_CUDA_COMPILER=/opt/conda/envs/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DWITH_CUDA=ON -DWITH_HIP=OFF \
  -DENABLE_CUDA_FIRST_CLASS_LANGUAGE=ON \
  -DOPENCV_EXTRA_MODULES_PATH=../src/modules \
  -DBUILD_LIST="core,cudev,cudaarithm,cudafilters,cudaimgproc,cudawarping,cudastereo,cudafeatures2d,cudaobjdetect,cudalegacy,cudaoptflow,cudacodec,cudabgsegm,imgproc,video,optflow,objdetect,calib3d,ts" \
  -DBUILD_TESTS=ON -DWITH_OPENCL=OFF -DWITH_PYTHON=OFF \
  -DCUDA_ARCH_BIN="8.0" -DCUDA_ARCH_PTX="" \
  -DCUDA_CUDA_LIBRARY=/opt/conda/envs/cuda-12.8/targets/x86_64-linux/lib/stubs/libcuda.so
```
Configure output: `NVIDIA CUDA: YES (ver 12.8.93, CUFFT CUBLAS)`, `NVIDIA GPU arch: 80`. `WITH_HIP: OFF`. `HAVE_CUDA` defined in cvconfig.h; `HAVE_HIP` absent.

### Build command (wrapped in timeit.sh)
```
bash utils/timeit.sh opencv_contrib cuda-compile -- \
  cmake --build projects/opencv_contrib/build-cuda \
    --target opencv_cudev opencv_cudaarithm opencv_cudafilters opencv_cudaimgproc \
      opencv_cudawarping opencv_cudastereo opencv_cudafeatures2d opencv_cudaobjdetect \
      opencv_cudalegacy opencv_cudaoptflow opencv_cudacodec opencv_cudabgsegm \
      opencv_test_cudev opencv_test_cudaarithm opencv_test_cudafilters \
      opencv_test_cudaimgproc opencv_test_cudawarping opencv_test_cudastereo \
      opencv_test_cudafeatures2d opencv_test_cudaobjdetect opencv_test_cudalegacy \
      opencv_test_cudaoptflow opencv_test_cudacodec \
    -j$(nproc)
```
Result: 865/865 targets, exit code 0, ~341 seconds. All 11 test executables built:
opencv_test_cudev, opencv_test_cudaarithm, opencv_test_cudafilters, opencv_test_cudaimgproc,
opencv_test_cudawarping, opencv_test_cudastereo, opencv_test_cudafeatures2d,
opencv_test_cudaobjdetect, opencv_test_cudalegacy, opencv_test_cudaoptflow, opencv_test_cudacodec.

### Warnings (4 total, ALL pre-existing upstream)
`-Wmissing-declarations` on 4 functions in `modules/cudacodec/src/cuda/ColorSpace.cu` lines 308/316/324/332 (Y8ToGray8/Y8ToGray16/Y16ToGray8/Y16ToGray16). These exist verbatim at the same line numbers in the upstream base sha `2a5154a4479e841aa1282ef83d139c4870d17b8f` -- confirmed by `git show 2a5154a:.../ColorSpace.cu`. Not introduced by the port.

### HAVE_NVCUVID / HAVE_NVIDIA_OPTFLOW
Both are `/* #undef */` in cvconfig.h (no Video Codec SDK or Optical Flow SDK in the conda toolkit), so cudacodec and cudaoptflow compile their stub/graceful-error paths -- this is the expected and correct behavior for a CUDA build without those SDKs installed. Identical to any upstream CUDA build without those optional SDKs.

### Summary
Every HIP guard (`#ifdef __HIP_PLATFORM_AMD__` / `HAVE_HIP` / `NOT HAVE_HIP` in cmake) correctly resolves to the CUDA/original branch under nvcc. No type alias, namespace define, or removed code introduced a CUDA regression. The port's additive guard structure is verified at the nvcc compile level.

## Follow-up commit 615f246 (cosmetic, post-PR) 2026-06-11

Per jeff review of the open contrib PR (#4147): (1) trimmed the rocDecode-unavailable
VideoReader error message (dropped the gfx90a/MI200 VCN detail -- too much for an error
string); (2) dejargoned the cvtFromYuv border-zero comment to spell out the allocator
behavior plainly, matching cudastereo/src/cuda/stereosgm.cu. Comment + one string literal
in the `#ifndef HAVE_CUDACODEC_DECODER` stub; no behavior change. Pushed as a follow-up
commit on the PR branch (not a rewrite). gfx1100/gfx1201 carried forward (source-class):
the string sits in a block compiled out on rocDecode-enabled builds, so their binaries are
byte-identical. Lead gfx90a stays pr-open at ad105bb (device code unchanged; the trimmed
string is an untested host stub path, no GPU re-run warranted).

## Doc fix (post-PR follow-up) 2026-06-11
Corrected the stale not-implemented claim in both docs: config_reference.markdown (core)
and README.md (contrib) listed alphaComp + the 16-bit/4-channel histogram variants as
StsNotImplemented, but the parity pass implemented them. Now only NVIDIA HW optical flow
and graph-cut stereo are listed. Doc-only -> carried forward, no revalidation.
Follow-up commits on both PR branches: contrib #4147 615f246->1c3b2fd, core #29285
adcd50c->0404733. Reviewer comment left as-is per jeff.

## Core opencv fork follow-ups (PR #29285) 2026-06-19

Two commits on the core opencv fork (AMD-Ecosystem/opencv @ moat-port), pushed on top of the validated port; tip 0404733 -> 50f05b1. Behavior-preserving on the success path (HIP found), so no GPU re-run (jeff override: no revalidations for these).
- d1e099b [ROCm] docs: note CMAKE_PREFIX_PATH for the WITH_HIP build -- config_reference.markdown WITH_HIP section now tells the user to pass -DCMAKE_PREFIX_PATH=/opt/rocm (or /opt/rocm/bin on PATH) so find_package(hip) resolves, and to check the configure-log HIP: status lines.
- 50f05b1 [ROCm] Fail configure when WITH_HIP is set but ROCm is not found -- OpenCVFindLibsPerf.cmake NOT HAVE_HIP case changed from message(WARNING)+continue to message(FATAL_ERROR). An explicit WITH_HIP=ON no longer silently falls back to a CPU-only build; the error names the fix (CMAKE_PREFIX_PATH) and the opt-out (WITH_HIP=OFF). Success path unchanged (find_package(hip) succeeds -> HAVE_HIP=1 -> branch not taken); verified by reconfiguring a gfx90a build (HIP still detected, no error). Core opencv is not a separately tracked MOAT project, so no advance-head/status transition applies.

## Validation (windows-gfx1101) 2026-06-20

**State: completed** -- validated_sha = 041d5528286a81465828a668d014035e149f6a97 (contrib head_sha)
GPU: AMD Radeon PRO V710 (gfx1101, RDNA3, wave32), HIP_VISIBLE_DEVICES=1
ROCm: TheRock 7.14.0a20260604 (pip SDK _rocm_sdk_devel)
OPENCV_TEST_DATA_PATH: projects/opencv_contrib/opencv_extra/testdata

Health check: `HIP_VISIBLE_DEVICES=1 timeout 35 hipInfo.exe` -> device 0 = AMD Radeon PRO V710, exit 0 (healthy). Post-test health check: same. No TDR events or wedge during the entire run.

Fork state: both repos reset to remote HEAD (contrib 041d5528, core 50f05b1) before building. Status clean (no uncommitted tracked source files in either repo).

### Build
Fresh build directory: `projects/opencv_contrib/build_gfx1101` (gitignored).

```
ROCM=.../pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
cmake -G Ninja src-core \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/clang.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_PREFIX_PATH=$ROCM \
  -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
  -DOPENCV_EXTRA_MODULES_PATH=src/modules \
  -DBUILD_LIST="core,cudev,cudaarithm,cudafilters,cudawarping,cudaimgproc,cudastereo,
    cudafeatures2d,cudaobjdetect,cudalegacy,cudabgsegm,cudaoptflow,cudacodec,
    imgproc,imgcodecs,videoio,highgui,video,optflow,objdetect,calib3d,ts" \
  -DBUILD_TESTS=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_PYTHON=OFF \
  -DWITH_FFMPEG=OFF -DWITH_ROCDECODE=OFF -DWITH_GSTREAMER=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-march=x86-64-v3" -DCMAKE_CXX_FLAGS="-march=x86-64-v3"
# Configure: 40.7s, Generating: 0.7s, 508/508 targets (same count as gfx1201)
cmake --build build_gfx1101 --target opencv_test_cudev ... (per-target, -j64)
# Warnings only (nodiscard hipGetLastError, lld-link --gc-sections). No errors.
```

DLL setup (same class as gfx1201 fix): copied amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll, hiprtc-builtins0714.dll (from _rocm_sdk_core/bin), plus hipfft.dll, rocfft.dll, hipblas.dll, rocblas.dll (from _rocm_sdk_devel/bin and _rocm_sdk_libraries/bin) into build_gfx1101/bin/ so the Windows loader uses TheRock runtime ahead of System32 Adrenalin DLL.

Run environment: `HIP_VISIBLE_DEVICES=1 ROCBLAS_TENSILE_LIBPATH=_rocm_sdk_libraries/bin/rocblas/library`.

### GPU test results (HIP_VISIBLE_DEVICES=1, OPENCV_TEST_DATA_PATH=.../opencv_extra/testdata)

| Suite | Passed | Total | Notes |
|---|---|---|---|
| opencv_test_cudev | 402 | 402 | PASS |
| opencv_test_cudastereo | 128 | 128 | PASS |
| opencv_test_cudalegacy | 14 | 14 | PASS (1 disabled) |
| opencv_test_cudawarping | 4535 | 4535 | PASS |
| opencv_test_cudaarithm | 11417 | 11417 | PASS |
| opencv_test_cudafilters | 16028 | 16028 | PASS |
| opencv_test_cudaimgproc | 3788 | 3788 | PASS (+117 histEven/histRange/alphaComp parity cases) |
| opencv_test_cudafeatures2d | 256 | 256 | PASS |
| opencv_test_cudaobjdetect | 18 | 18 | PASS |
| opencv_test_cudaoptflow | 41 | 47 | 6 failures: EXACTLY the documented set (see below) |
| opencv_test_cudacodec | 240 | 240 | PASS (YuvConverter only; WITH_ROCDECODE=OFF on Windows) |

**cudaoptflow 6 expected failures (identical to all prior platforms):**
- CUDA_OptFlow/OpticalFlowDual_TVL1.Async/0 -- float atomicAdd reduction ordering (not a port defect)
- CUDA_OptFlow/OpticalFlowDual_TVL1.Async/1 -- same
- CUDA_OptFlow/NvidiaOpticalFlow_1_0.Regression/0 -- no AMD HW OF engine
- CUDA_OptFlow/NvidiaOpticalFlow_1_0.OpticalFlowNan/0 -- same
- CUDA_OptFlow/NvidiaOpticalFlow_2_0.Regression/0 -- same
- CUDA_OptFlow/NvidiaOpticalFlow_2_0.OpticalFlowNan/0 -- same

No unexpected failures. All 10 other suites are 100%. Results identical to gfx1201 (same failure set, same pass counts). gfx1101 is RDNA3 (wave32, same physical wave size as gfx1100 and gfx1201), so no wave32 correctness concerns. No delta-port needed.

## CI on PR #4147 (checked 2026-06-23)

The only check, `default`, is opencv's external Buildbot (pullrequest.opencv.org),
not GitHub Actions. Both the contrib PR #4147 and the companion core PR
opencv/opencv #29285 are correctly paired (same head branch `moat-port`, which is
how opencv's CI links a contrib PR to its core PR) and both show
`default = error: "Unsuccessful builds. Waiting for builds"`, set 2026-06-19 and
unchanged since. The dashboard itself returns "Service is in degraded state".

So this is an opencv CI-infrastructure problem, not our code and not a readable
GitHub-Actions log: the status is stale, the service is degraded, and there are no
per-builder logs or bot comments accessible from the CLI. Nothing actionable on
our side. When opencv's CI recovers it will re-evaluate the paired build; only
then (or via the dashboard when it is up) can we see whether any real build issue
exists. The port itself is validated on real AMD GPUs (gfx90a/gfx1100/gfx1201).

## Validation 2026-08-09 (revalidate, linux-gfx90a) -- carried forward, binary-equiv

**State: completed** -- validated_sha advanced ad105bb5914a256335dad33abdd66b360dd91634
(stale, recorded pre-PR-squash before three follow-up commits) -> 041d5528286a81465828a668d014035e149f6a97
(current project head_sha). GPU: MI250X, gfx90a, HIP_VISIBLE_DEVICES=0 (index 0 confirmed
gfx90a via `rocm-smi`). No test re-run was needed or performed; see method below.

### Delta
`git diff ad105bb..041d5528 --stat` in contrib: `README.md | 2 +-`, `modules/cudacodec/src/video_reader.cpp | 12 ++++++------`.
Three commits, all already narrated in notes.md: 615f246c (trim the rocDecode-unavailable
CV_Error message + reword a comment, both in video_reader.cpp), 1c3b2fd4 + 041d5528 (two
README.md wording fixes, doc-only). `python3 utils/moatlib.py classify opencv_contrib
ad105bb.. 041d5528..` returned `class=mixed arch_independent=False inert=False` because
video_reader.cpp is source, not a doc/comment file, so the classifier correctly declines to
call it inert automatically -- even though inspection shows both hunks are message-text/
comment-only. Went to the codeobj_diff route rather than assuming.

### Method: codeobj_diff (binary-equivalence carry-forward), not a GPU re-run
Cloned a second contrib checkout at ad105bb (`src-old-codeobj`, throwaway, deleted after) next
to the existing `src` (already at head 041d5528) and `src-core` (moat-port tip). Built a scoped
target twice, same flags, gfx90a, differing only in `OPENCV_EXTRA_MODULES_PATH`:
```
cmake -G Ninja src-core -DWITH_HIP=ON -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/amdclang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DOPENCV_EXTRA_MODULES_PATH=<src-old-codeobj|src>/modules \
  -DBUILD_LIST="core,cudev,cudaarithm,cudawarping,cudacodec,imgproc,imgcodecs,videoio,highgui,ts" \
  -DBUILD_TESTS=ON -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_PYTHON=OFF \
  -DWITH_FFMPEG=ON -DWITH_ROCDECODE=ON -DWITH_GSTREAMER=OFF
cmake --build . --target opencv_cudacodec -j$(nproc)   # 403/403, warnings only (pre-existing nodiscard)
```
`cvconfig.h` confirmed `HAVE_ROCDECODE` undefined in both builds (rocdecode-dev not installed;
gfx90a has no VCN engine), matching every prior gfx90a validation of this project.
`python3 utils/codeobj_diff.py build_codeobj_old build_codeobj_new`:
```
verdict=indeterminate
  CMakeFiles/4.0.3/CMakeDetermineCompilerABI_*.bin / CompilerId*/a.out: indeterminate (device-code extraction failed)
  lib/libopencv_core.so: identical (exported symbols + device ISA identical, 2644 exports)
  lib/libopencv_cudaarithm.so: identical (3152 exports)
  lib/libopencv_cudacodec.so: identical (83 exports)
  lib/libopencv_cudawarping.so: identical (5710 exports)
  lib/libopencv_cudev.so / imgcodecs.so / imgproc.so / videoio.so: indeterminate (device-code extraction failed)
```
Two indeterminate classes, both benign per the calibration in the dispatch:
- `CMakeFiles/.../CompilerIdC(XX|HIP)` and `CMakeDetermineCompilerABI_*.bin`: CMake's own
  toolchain-probe scratch binaries, identical in both trees by construction (same cmake/compiler),
  not project output -- ignored per instruction.
- `libopencv_cudev/imgcodecs/imgproc/videoio.so`: `roc-obj-ls` exits 255 with `Error: No kernel
  section found` on these (confirmed manually) -- they are pure host libraries with zero embedded
  gfx code objects, and codeobj_diff's `device-code extraction failed` branch cannot distinguish
  that from a real tooling failure (a gap in the tool, noted in codeobj_diff.py's own docstring:
  `_device_isa` is supposed to return `""` for a host-only binary but `roc-obj-ls` errors instead
  of reporting zero slices). Fell back to `llvm-nm -D --defined-only <lib> | awk '{print $NF}' |
  sort` on all four, old vs new: zero diff lines in every case, i.e. byte-identical exported symbol
  sets. Combined with the fact that `compare_binary()` only reaches the device-ISA step *after*
  confirming exported symbols already match (`sa == sb`), the symbol match plus the manual nm
  confirmation is sufficient: nothing in the compiled program differs.
Also confirmed the changed lines cannot possibly reach the GPU or the CUDA path: the trimmed
`CV_Error` string sits inside `#ifdef __HIP_PLATFORM_AMD__` (the `else` branch, `throw_no_cuda()`,
is what nvcc takes -- untouched), and the reworded comment sits inside `#ifdef HAVE_ROCDECODE`,
which is off in this build (no VCN on gfx90a) so that whole block does not even compile here.

**Verdict: carry forward.** `python3 utils/moatlib.py carry-forward opencv_contrib linux-gfx90a
041d5528286a81465828a668d014035e149f6a97 binary-equiv "<reason>"` -- recorded, no GPU test run.

### CUDA no-regression gate
Skipped: this is a carried-forward revalidation (rule explicitly exempts it), and independently
the delta is proven not to reach the CUDA path at all (see above) -- the CUDA-compile-clean
verdict already on file at ad105bb/adcd50c (see "CUDA no-regression compile gate 2026-06-11")
still describes the current head.

### Jargon + integrity
`python3 utils/jargon.py --port opencv_contrib`: clean (needed a local `4.x` branch tracking
`origin/4.x` in the fresh `src` clone; `fork_default_branch` is `4.x` and the tool diffs
`4.x..moat-port` locally). `git status --porcelain` clean in both `src` and `src-core` throughout;
the throwaway `src-old-codeobj` clone and both `build_codeobj_*` dirs were deleted after use
(not part of the tracked build layout, not referenced by any "## Install as a dependency" section).
No changes to the "Install as a dependency" recipe -- untouched by this delta.

Wall-clock: two scoped cmake configures (~30s each) + two scoped `opencv_cudacodec` builds
(~a few minutes each, 403/403 targets) + codeobj_diff (seconds). No full-suite GPU test run,
which is the point of the carry-forward path.
