# DynOSAM notes

## Why this is back

Re-opened 2026-08-07 after two premises changed.

The prior screen declined this on TensorRT and OpenCV CUDA. Both have answers now. Its TensorRT use is ONNX load plus inference in a single file (`dynosam_nn/src/YoloV8ObjectDetector.cc`), which is the shape MIGraphX is built for -- rewritten integration, not a rewritten project. And `cv::cuda::*` is covered by MOAT's own opencv_contrib port, completed on four architectures and upstream at opencv/opencv_contrib#4147, so this wants `depends_on: [opencv_contrib]` rather than a decline.

Check before assuming either: whether it reaches TensorRT through ONNX Runtime, in which case the MIGraphX execution provider may need no application change at all; and which `cv::cuda` calls it actually makes against what opencv_contrib ported.

## The prior analysis

Do not redo it. The earlier screen's full write-up is in history:

    git show 27f7646^:projects/DynOSAM/plan.md

(The sha quoted in the earlier version of this file, `b40576d53399`, is not reachable
from this branch. `27f7646` is the commit that removed the folder; its parent still
carries `plan.md` and `status.json`.)

Read it first and test only what has changed.

## Second screen 2026-08-12 (intake, linux-gfx1100)

Upstream screened at `ACFR-RPG/DynOSAM` main, shallow clone into `agent_space/` (no fork
exists; nothing was written upstream). Recommendation recorded as `fork`. Every fact below
was checked against the source, not carried over from the first screen.

### Licence: BSD-3-Clause, tier 1

`python3 utils/licenses.py check ACFR-RPG/DynOSAM` -> `license=BSD-3-Clause tier=1`,
and the top-level `LICENSE` file is a verbatim BSD 3-Clause text (c) 2024 ACFR-RPG,
The University of Sydney -- read directly, not taken from the GitHub field.
Recorded in `status.json.license_spdx`.

Per-file checks, both clean:

- `python3 utils/licenses.py scan-nvidia agent_space/DynOSAM-screen` -> no NVIDIA
  proprietary licence text anywhere in the tree.
- No submodules (`.gitmodules` absent) and no `third_party/`, `vendor/`, `external/`
  or `3rd*` directory. The one vendored component is
  `dynosam/include/dynosam/frontend/anms/`, carrying its own MIT LICENSE
  (c) 2018 Oleksandr Bailo. MIT under BSD-3 is an ordinary compatible mix with a
  licence file of its own, not the ambiguous per-part case that would need a person.

Scope note: this is a contribution assessment. Nothing here says anything about
shipping or depending on DynOSAM in our own software.

### Duplicate effort: none

- No `AMD-Ecosystem/DynOSAM` and no `ROCm/DynOSAM` (both 404).
- `gh search repositories dynosam` returns the upstream only.
- All 33 upstream forks enumerated: every one is a vanilla `main` (plus feature
  branches on `abersier` and `itshaihong`, none ROCm-related). No platform fork.
- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/` on the clone: the only hits are
  "amd64" (x86-64, e.g. `docker/Dockerfile.amd64`) and "ship" inside ordinary words.
  No AMD GPU support claimed or linked.
- Web search for a DynOSAM ROCm/HIP port: nothing.

Upstream is alive, not archived: 327 stars, `pushed_at` 2026-07-30, `updated_at`
2026-08-10, 13 open issues, T-RO paper (arXiv 2501.11893). A pull request has a
destination.

### What the GPU surface actually is

Two distinct halves, and the distinction is the whole screen.

**Half 1 -- `cv::cuda`, and it is the SLAM-critical path.** `dynosam` (the core
package), `dynosam_cv`, `dynosam_common` and `dynosam_nn` all use it. The frontend's
`FeatureTracker` constructor builds a GPU KLT tracker *unconditionally*:

    lk_cuda_tracker_ = cv::cuda::SparsePyrLKOpticalFlow::create(
        klt_window_size, klt_max_level, 30);          // FeatureTracker.cc:60

Full inventory across the repo: `GpuMat` (45), `Stream` (13), `SparsePyrLKOpticalFlow`
(4), `PtrStepSz` (3), `HostMem` (3), `getCudaEnabledDeviceCount` (2),
`createGoodFeaturesToTrackDetector` (2), `StreamAccessor` (2), `OpticalFlowPyrLK` (2),
`threshold`, `resize`, `FastFeatureDetector`, `CornersDetector` (1 each). Headers:
`core/cuda.hpp`, `core/cuda_stream_accessor.hpp`, `cudaarithm.hpp`, `cudaimgproc.hpp`,
`cudaoptflow.hpp`, `cudawarping.hpp`.

Every one of those modules is ported and GPU-validated by MOAT's own opencv_contrib
port -- on this exact platform. From `origin/port/opencv_contrib` notes, gfx1100
(RDNA3, wave32): cudaarithm 11417/11417, cudawarping 4535/4535, cudaimgproc 3788/3788,
cudafeatures2d 256/256, cudaoptflow 41/47. The six cudaoptflow non-passes are the
documented set -- 4 NvidiaOpticalFlow_1_0/2_0 (no AMD HW optical-flow engine) and 2
TVL1.Async (float atomicAdd ordering, ruled a test-strictness artifact). **PyrLK is not
among them**, so the one cv::cuda call DynOSAM cannot run without is a passing case.

So for half 1 the DynOSAM source needs approximately no change: it is a build-and-
validate exercise against our ported OpenCV. That also makes DynOSAM the first
downstream consumer to exercise that port end to end, which is worth something on its
own.

**Half 2 -- its own CUDA, which is small.** One `.cu` file,
`dynosam_nn/src/YoloV8CudaUtils.cu` (327 lines): two `__global__` kernels
(`YOLO_PostProcess_Kernel`, `YOLO_Mask_Combination_Kernel`) and one `__device__`
`sigmoidf`. No warp intrinsics, no textures, no Thrust/CUB, no cuBLAS/cuFFT. Runtime
API across the whole repo is ~30 distinct symbols (`cudaMalloc`, `cudaMallocHost`,
`cudaHostAlloc`/`cudaHostGetDevicePointer`, `cudaMemcpyAsync`, stream and event
create/destroy/sync, `cudaGetLastError`), all with direct HIP equivalents. Pure CMake
(`project(dynosam_nn LANGUAGES C CXX CUDA)`, `find_package(CUDAToolkit)`, no Torch
extension) -- Strategy A shape. The kernels take `cv::cuda::PtrStepSz` arguments, so
they compile against the same ported OpenCV headers half 1 needs.

### The one real gap: TensorRT, and it is NOT ONNX Runtime

The premise to test was whether it reaches TensorRT through ONNX Runtime. It does not.
It uses the raw TensorRT C++ API directly: `nvinfer1::IBuilder` +
`nvonnxparser::createParser` in `buildEngineFromOnnx`, `nvinfer1::createInferRuntime` +
`deserializeEngine` for the cached `.engine`, and `IExecutionContext` for inference.
`nvinfer1::` symbol counts: DataType 15, Dims 10, TensorIOMode 9, IRuntime 6,
OptProfileSelector 4, ICudaEngine 4, ILogger 3, and one each of IBuilder,
IBuilderConfig, INetworkDefinition, IHostMemory, IExecutionContext. Confined to
`dynosam_nn/{src/TrtUtilities.cc (463 lines), include/dynosam_nn/TrtUtilities.hpp
(348)}` plus the inference call sites in `src/YoloV8ObjectDetector.cc (718)`.

So there is no "install the MIGraphX execution provider and change nothing" route. But
the shape does map cleanly: `parse_onnx -> compile(target("gpu")) -> eval` with `.mxr`
save/load is the MIGraphX analogue of parse-ONNX / serialize-engine / execution-
context. MIGraphX 2.15 is packaged for this ROCm (`apt-cache policy migraphx` ->
`2.15.0.70203-90~24.04` from repo.radeon.com/rocm/apt/7.2.3; not currently installed on
this host, ROCm 7.2.3).

Two further facts a planner needs, both verified:

- **The TensorRT dependency is hard, not optional, despite appearances.**
  `dynosam_nn/CMakeLists.txt` has `option(DYNOSAM_NN_USE_TRT "Build with TensorRT" ON)`
  which sets `ENABLE_TENSORRT_CXX_VALUE` to 0 when TensorRT is missing -- and that
  variable is then **never read anywhere in the repository**. Dead code. The target
  links `TensorRT::TensorRT` unconditionally, `project(... CUDA)` is unconditional, and
  `dynosam/CMakeLists.txt` has `find_package(dynosam_nn REQUIRED)` + links
  `dynosam_nn::dynosam_nn`, with `dynosam/src/frontend/vision/FeatureTracker.cc`
  including `dynosam_nn/YoloV8ObjectDetector.hpp` directly. There is no build
  configuration today that produces a working DynOSAM without TensorRT present.
- **But the detector is runtime-optional, and there is a second implementation.** The
  TRT detector is constructed only under `if (!params_.prefer_provided_object_detection)`
  (FeatureTracker.cc:63); the offline dataset loaders default to provided masks, and
  the nn README documents `prefer_provided_object_detection=false` as the opt-in.
  Separately `dynosam_nn/src/PyObjectDetector.cc` + `dynosam_nn_py/object_detection.py`
  run YOLOv8 through ultralytics/PyTorch (`self.device = "cuda" if
  torch.cuda.is_available()`), which is exactly what ROCm PyTorch satisfies unchanged.

That gives the planner a genuinely smallest-complete first port: make the TensorRT
dependency honour the project's own `DYNOSAM_NN_USE_TRT` option (a change upstream
plausibly wants regardless -- their option is broken today), port the one `.cu` to HIP,
build the stack against ROCm OpenCV, and validate the dynamic-SLAM pipeline on
precomputed masks plus the ultralytics/ROCm-PyTorch detector. The MIGraphX
reimplementation of `TrtUtilities` is then a well-bounded follow-up with a named
target, registrable as deferred work rather than a blocker. It is the planner's call
whether to fold it in; it should not be assumed away.

### Dependencies

`depends_on = [opencv_contrib, opencv]`, both hard, both recorded.

- `opencv_contrib` supplies cudaarithm/cudawarping/cudaimgproc/cudaoptflow/
  cudafeatures2d. `dep_status` -> `ok, completed` (stage review-passed, upstream PR
  opencv/opencv_contrib#4147 open).
- `opencv` (core) supplies `cv::cuda::GpuMat`/`Stream`/`StreamAccessor` and the entire
  `WITH_HIP` build machinery; opencv_contrib's own build configures against it as
  `src-core`. `dep_status` -> `waiting, adopted, no arch has started`, so **recording
  it truthfully will show DynOSAM in `moatlib.py dep-blocked`**. That is accurate but
  it will not clear by itself: `projects/opencv/notes.md` says the fork's `moat-port`
  branch already carries the port and that its validation lives with opencv_contrib
  (it has no test suite of its own), so its empty `platforms` map is bookkeeping, not
  missing work. A person should resolve the `opencv` record -- or say plainly that
  DynOSAM may proceed against it -- before a planner is dispatched here. Flagging it
  rather than omitting the dependency, because an omitted hard dependency is the
  failure that is invisible later.
- `orient.sh` already reports `dep-doc MISSING opencv_contrib (needed by
  DynOSAM,plvs)`. Confirmed: `projects/opencv_contrib/notes.md` on
  `origin/port/opencv_contrib` twice *refers* to "the 'Install as a dependency'
  recipe" but contains no `## Install as a dependency` heading. DEPENDENCIES.md makes
  that section a MUST for a provider, and it is the first thing DynOSAM's porter would
  go looking for. Not fixed here -- it belongs to opencv_contrib's branch and its
  owner, not to this screen.

The build DAG is otherwise CPU: GTSAM, OpenGV, MPI, Boost.Python, pybind11, glog/gflags,
SuiteSparse, TBB.

### Environment cost, which is the honest counterweight

This is a ROS 2 workspace: seven ament packages (`dynosam`, `dynosam_common`,
`dynosam_cv`, `dynosam_nn`, `dynosam_opt`, `dynosam_ros`, `dynosam_utils`), depending
on rclcpp, cv_bridge, image_transport, message_filters, tf2_ros and
`dynamic_slam_interfaces`. Upstream's own `docker/Dockerfile.amd64` builds on a
ROS 2 kilted image and compiles opencv + opencv_contrib 4.10.0 from source with
`WITH_CUDA=ON` -- which is convenient for us, because a from-source OpenCV is the
documented path either way and our ported forks slot into exactly that step (note the
version gap: upstream pins 4.10.0, our forks track `4.x`).

This host has no ROS 2 installed (`/opt/ros` absent; Ubuntu 24.04 -> Jazzy is the
matching distro) and no MIGraphX installed. The dominant cost of this port is
environment and integration, not kernel translation. That is the thing to weigh, and
it is why the recommendation is a case rather than a formality.

### Tests available for the validation bar

33 `test_*.cc` files. `dynosam_cv/test/test_cuda_cache.cc` is the explicit GPU test
(GpuMat caching/reference counting, includes `cuda_runtime.h`); the `dynosam` suite
(~30 files: factors, backend, triangulation, camera, pipelines) is CPU and must not
regress. The frontend feature-tracking path is where a real GPU run has to land, since
that is where `SparsePyrLKOpticalFlow` and `createGoodFeaturesToTrackDetector` live.

### Recommendation

`fork`, viable yes. Recorded with `set-intake`. Nobody has decided anything: the fork
appearing is the decision, and if the answer is no, the reason to reach for is
`cant-port` on the ground that the TensorRT reimplementation plus a ROS 2 + GTSAM +
from-source-OpenCV stack is more integration than the port is worth -- not on the
ground that the CUDA cannot be translated, which is no longer true.
