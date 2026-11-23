# autoware_universe port plan

## Project

- Name: autoware_universe (autowarefoundation/autoware_universe, the Autoware Foundation's ROS 2 monorepo of about 350 ament packages)
- Upstream: https://github.com/autowarefoundation/autoware_universe, default branch `main`, planned at `6cea1ac7b729` (2026-11-23, "test(autoware_pointcloud_preprocessor): compare concatenated points without row order (#13619)"). The fork's `main` mirror is at `5f8fcca`, behind upstream; the porter starts `moat-port` from the current upstream tip and re-checks the package list below against it.
- Fork: https://github.com/AMD-Ecosystem/autoware_universe, port branch `moat-port` (not created yet).
- Hard dependencies (MOAT projects): `autoware_cuda` (round 1 `autoware_cuda_utils` review-passed and validated on linux-gfx950 at fork head `992d703`; round 2, the MIGraphX engine layer inside `autoware_tensorrt_common`, in progress under Jeff's rulings Q1 = A and Q2 = a) and `cuda_blackboard` (review-passed, validated on linux-gfx942 and linux-gfx1100 at `9e0fb86`).
- Scope ruling (Jeff Daily, 2026-11-11): full scope, staged: the pure-CUDA packages first, then a MIGraphX backend for the TensorRT nodes (plugin-free models first, then the custom plugins as MIGraphX custom ops, then bevfusion, ptv3, bevformer and VAD). The CUDA/TensorRT path is preserved. Windows is waived.
- Planned on linux-gfx942 (mi300 fleet host). No GPU used and nothing built. There is no lead platform; the plan is arch-independent except where a delta is named.

**Decisions Jeff must make are at the top of the `## Planning` section of notes.md (Q1-Q5), together with the interface list that autoware_cuda's round 2 has to provide.** The plan is written for the recommended answer to each and says where another answer changes it.

## Existing AMD support

None. Intake found no `hip|rocm|amd|migraphx` PRs, issues or branches upstream, no AMD-Ecosystem or ROCm repository, and no MIGraphX backend for Autoware anywhere. The AMD Silo AI / Autoware Foundation collaboration (2025-12-16) has published no code touching autoware_universe; coordination with it is Jeff's call and was not made a precondition. Re-run the search when the first PR is prepared.

Upstream takes PRs (CONTRIBUTING, semantic PR titles, DCO); it does not link platform forks instead of merging, so upstream PRs are the vehicle. Upstream's norm is one package per PR (`feat(autoware_xxx): ...`); see Q4 for the delivery shape.

Port versus rewrite: the pre/post-processing kernels are plain SIMT CUDA (voxelisation, NMS, resize, scatter) with CUB/Thrust; a mechanical HIP port is the final form. The exceptions are (1) the sparse convolution in `autoware_tensorrt_plugins` and the bevfusion voxeliser, which call spconv/cumm generated kernels (tensor-core PTX, SM-keyed tuner) and have no HIP counterpart until the spconv/cumm ports exist (Q1), and (2) every TensorRT engine and plugin, which is reimplemented against MIGraphX (the engine layer lives in autoware_cuda; the custom ops live here).

## Build classification

**cmake** (ament_cmake / ament_cmake_auto in a colcon workspace), recorded with `set-ext-type autoware_universe cmake`. Evidence, the same pattern in all 21 GPU packages:

- Legacy FindCUDA: `find_package(CUDA)` and `cuda_add_library(...)` with hand-written `CUDA_NVCC_FLAGS` gencode lists (sm_75..sm_120), e.g. `sensing/autoware_cuda_pointcloud_preprocessor/CMakeLists.txt:5,53-71,82-92`, `perception/autoware_lidar_centerpoint/CMakeLists.txt:13-33,72-93`.
- TensorRT through `find_library(NVINFER nvinfer)` / `NVONNXPARSER` or `tensorrt_cmake_module`'s `find_package(TENSORRT)`; a `TRT_AVAIL AND CUDA_AVAIL` gate with an `else()` that builds nothing (for example `perception/autoware_lidar_centerpoint/CMakeLists.txt:36-52,203-210`).
- No `find_package(Torch)`, no setup.py extension. One optional ONNX Runtime path (diffusion_planner).

## Port strategy

**Strategy A, per package**: a HIP branch next to the unchanged CUDA branch in each package's CMakeLists.txt, `enable_language(HIP)` with `set_source_files_properties(<.cu> PROPERTIES LANGUAGE HIP)` (or `hip_add_library`), sources kept in CUDA spelling and mapped through compat headers on the HIP include path only. A CUDA machine sees no new behaviour.

Backend selection, identical in every package and matching autoware_cuda: the CUDA branch is tried first; the HIP branch is taken when CUDA is not found or when `-DAUTOWARE_CUDA_USE_HIP=ON` is set (needed on hosts that have both toolkits, because FindCUDA finds `/usr/local/cuda`). Each package must also honour the backend recorded by the installed `autoware_cuda_utils` (`autoware_cuda_utils_USE_HIP`, and its configure-time `FATAL_ERROR` when a consumer finds CUDA against a HIP build; see autoware_cuda notes "Install as a dependency"). A package with the option ON must therefore skip its own `find_package(CUDA)`/`CUDAToolkit`, or the deferred check fails its configure.

### Compat headers (Q3)

Sources include `<cuda_runtime.h>`, `<cub/cub.cuh>`, `<thrust/...>`, `<cuda/std/optional>`, `<cuda/functional>`, `<cuda_fp16.h>`, `NvInfer.h`. On ROCm:

- Runtime API names: `autoware_cuda_utils` already installs shim `cuda_runtime.h` / `cuda_runtime_api.h` mapping 34 names. The union over autoware_universe is about 70 (memcpy kinds, `cudaMemcpyToSymbolAsync`, `cudaMallocAsync`, `cudaDeviceGetAttribute`, events, `cudaStreamBeginCapture` and graphs in diffusion_planner, `cudaGetLastError`, `cudaErrorLaunchFailure`, ...). Recommended (Q3 option A): autoware_cuda's round 2 extends its shim to the full set autoware_universe needs, plus `cub/cub.cuh` -> hipCUB (`namespace cub = hipcub`), `cuda_fp16.h` -> `hip/hip_fp16.h`, and `cuda/std/*` / `cuda/functional` (below), so a consumer that links `autoware_cuda_utils` needs no per-package runtime header. Option B: each package carries its own HIP-only `<pkg>_hip_compat.h` force-included on the HIP branch (`-include`, HIP sources and host .cpp of that package), so sources stay byte-identical; about 20 near-duplicate headers. Packages that use CUDA but do not link `autoware_cuda_utils` today (transfusion; occupancy grid uses only its include dirs) add the dependency on the HIP branch under either option.
- libcu++ (`cuda/std/optional`, `cuda::std::less`, `cuda::std::not_equal_to`, `cuda/functional`) in the pointcloud preprocessor: the pip ROCm SDK carries no libhipcxx (checked on this host, ROCm 10.1 SDK). Map the HIP-only shim headers `cuda/std/optional` and `cuda/functional` onto `<optional>`/`<functional>` with `namespace cuda::std = ::std` (hip-clang treats constexpr functions as host-device, and `std::optional` and `std::less` are constexpr in C++17). libhipcxx is the fallback if a device-side use does not compile; it would become a build dependency to document.
- Thrust policies: `thrust::cuda::par` / `thrust::cuda::par_nosync` do not exist in rocThrust's HIP backend; `thrust::hip::par` and `thrust::hip::par_nosync` do (checked in `thrust/system/hip/detail/par.h`). autoware_cuda round 1 guarded its one site with `THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP`; do the same at each site (pointcloud preprocessor downsample filter 4 sites, frnet 6, ptv3 1) or with one shared macro in the compat header. Do not `#define cuda hip`.
- `__constant__` symbols with `cudaMemcpyToSymbol[Async]` (pointcloud preprocessor downsample and undistort, frnet): HIP needs `HIP_SYMBOL(sym)`; a compat macro cannot add it, so these are small guarded source edits (or `#define cudaMemcpyToSymbolAsync(s, ...) hipMemcpyToSymbolAsync(HIP_SYMBOL(s), ...)` in the compat header, which is the less invasive choice).

### TensorRT consumers on MIGraphX

Per Jeff's autoware_cuda Q1 = A, `autoware_tensorrt_common` keeps its name and `TrtCommon` API and gains a MIGraphX backend with HIP-only stand-in `NvInfer.h` value types. Consumers here therefore mostly compile unchanged. What each consumer still changes:

- CMake: the HIP branch replaces `find_library(nvinfer)`/`TENSORRT` with `find_package(autoware_tensorrt_common)` and its exported backend variable (requested from round 2: `autoware_tensorrt_common_BACKEND` = `tensorrt` | `migraphx`), and drops the unused `CUBLAS_LIBRARIES` / `CUDA_curand_LIBRARY` links (streampetr, image_projection_based_fusion, calibration_status_classifier, centerpoint, transfusion, bevfusion, frnet, ptv3, VAD, diffusion_planner link them; none calls them). `autoware_shape_estimation` keys its GPU path off a `USE_CUDA` macro; the HIP branch defines the same macro (CMake-side only).
- Direct TensorRT use outside `TrtCommon` (the only consumer source edits that depend on the engine layer):
  - `autoware_camera_streampetr` `lib/network/network.cpp:105-117,153`: `set_sigmoid_and_softmax_layers_to_fp32(pts_head_->getNetwork())` walks `INetworkDefinition` with no null check; on MIGraphX `getNetwork()` returns `nullptr` and fp16 (the default) would crash. HIP branch: skip the layer walk and ask the engine layer to keep `sigmoid`/`softmax` out of `quantize_fp16` (round 2 offered an optional HIP-only `TrtCommonConfig` field for exactly this; it is on the interface list).
  - `autoware_diffusion_planner` `src/inference/utils.cpp:124-127`: `getBuilderConfig()->setMaxAuxStreams(0)` is already null-checked; no change. Its CUDA-graph capture around `enqueueV3` (`utils.cpp:52-84,141-155`; `use_cuda_graph` false in `config/diffusion_planner.param.yaml:8`, true as the node default) must be validated with hipGraph capture of MIGraphX `run_async`, or refused with a clear error on the MIGraphX backend. The optional ONNX Runtime path (`src/inference/onnxruntime_inference.cpp:63-79`, CUDA and TensorRT EPs) maps to the MIGraphX EP when an ONNX Runtime ROCm build is found; otherwise it stays off as upstream does without onnxruntime.
  - `autoware_tensorrt_bevformer` builds and runs its engine itself (`src/bevformer_inference_engine.cpp`, about 400 lines of builder/runtime calls, dlopen of `libtensorrt_ops.so` + `getPluginRegistry`). HIP branch: move it onto `TrtCommon` (preferred, if its fp16/precision-constraint needs are expressible) or a local MIGraphX path; decided by the stage-3 porter and recorded.
  - INT8 calibrators in yolox and classifier subclass `nvinfer1::IInt8LegacyCalibrator`, `IInt8EntropyCalibrator2` and `IInt8MinMaxCalibrator`; the stand-ins must declare all three (interface list). int8 is approximate on MIGraphX (Q2 = a): `yolox_s_plus_opt` defaults to int8 and is validated for "runs, detections plausible" plus an fp16-vs-int8 agreement number, not exact parity.
- Plugins. MIGraphX cannot load TensorRT plugin libraries. The engine layer keeps the `plugin_paths` dlopen (round 2 plan) and a MIGraphX plugin library registers its ops with `migraphx::register_experimental_custom_op` from a static initialiser. Here that means a HIP branch in `autoware_tensorrt_plugins` that builds a library of MIGraphX custom ops installed under the same name and path (`share/autoware_tensorrt_plugins/plugins/libautoware_tensorrt_plugins.so`), so consumer parameter files stay unchanged, wrapping the same hipified kernels. Op names must equal the ONNX `op_type` the models use (the TensorRT registered names: `Argsort`, `QuickCumsumCuda`, `MultiScaleDeformableAttentionPlugin`, `RotatePlugin`, `SegmentCSR`, `SelectAndPadPlugin`, `CustomUnique`, `GetIndicePairsImplicitGemm`, `GetIndicePairs`, `ImplicitGemm`, `IndiceConv`). Ops with data-dependent output size (`CustomUnique`, both `GetIndicePairs*`) need a padded-to-max shape plus a count output, since MIGraphX custom ops declare output shapes from input shapes; the consumer then reads the count. Standard ONNX ops in the graphs (ScatterND/ScatterElements with reduction in transfusion and frnet, NonMaxSuppression) must be checked against MIGraphX's parser on the real models.
  - `EfficientNMS_TRT` (TensorRT's built-in plugin) is attached to the default `yolox-tiny.onnx` (`perception/autoware_tensorrt_yolox/README.md:95-99`); the plain-YOLOX decode path in `tensorrt_yolox.cpp:130-148` already handles models without it. Implement `EfficientNMS_TRT` as one more MIGraphX custom op in the plugin library (HIP kernel; attribute semantics from TensorRT's public plugin documentation, not NVIDIA's source) so the shipped artifact works unchanged. This is the only built-in TensorRT plugin found in the default models (traffic_light_fine_detector's `tlr_car_ped_yolox_s_batch_6.onnx` must be checked).

## Stages

The stages follow the dependency gates. Each stage is one porting round on the same `moat-port` branch (the autoware_cuda Q3 = i pattern, recommended again in Q4): work for later stages is recorded under `owed` in surface.json, which the surface gate accepts and `pr_ready` refuses, so nothing partial reaches upstream unnoticed. A later stage is started deliberately (dispatch names the stage), not by the selector, because the selector cannot see the autoware_cuda round-2 gate.

**Stage 1: pure CUDA (needs `autoware_cuda_utils` round 1 and cuda_blackboard; both are available now).**
1. `sensing/autoware_cuda_pointcloud_preprocessor` (148 KB of .cu, 36 kernels; 5 components)
2. `perception/autoware_ground_segmentation_cuda` (32 KB, 11 kernels; depends on 1)
3. `perception/autoware_probabilistic_occupancy_grid_map` (59 KB, 13 kernels; CUDA path optional, `USE_CUDA`)

**Stage 2: plugin-free TensorRT consumers (needs autoware_cuda round 2, the MIGraphX `TrtCommon`, validated on the platform).**
`autoware_tensorrt_yolox` (+ the `EfficientNMS_TRT` custom op, which needs the plugin-library skeleton of stage 3; it may land in stage 2), `autoware_tensorrt_classifier`, `autoware_traffic_light_classifier`, `autoware_traffic_light_fine_detector`, `autoware_traffic_light_pipeline`, `autoware_lidar_centerpoint`, `autoware_lidar_transfusion`, `autoware_lidar_frnet`, `autoware_image_projection_based_fusion` (pointpainting, builds on centerpoint), `autoware_calibration_status_classifier`, `autoware_camera_streampetr`, `autoware_simpl_prediction`, `autoware_shape_estimation`, `autoware_lidar_apollo_instance_segmentation`, `autoware_diffusion_planner` (if its ONNX needs no custom op; it passes `plugins_path` generically and no specific op was found in code; inspect the graph). Order inside the stage: classifier -> yolox -> traffic light packages; centerpoint -> pointpainting; the rest independent.

**Stage 3: custom plugins as MIGraphX custom ops (needs stage 2's engine path working).**
`autoware_tensorrt_plugins` without the spconv ops (Argsort, CustomUnique, SegmentCSR, QuickCumsumCuda, MultiScaleDeformableAttentionPlugin, RotatePlugin, SelectAndPadPlugin, plus EfficientNMS_TRT if not done in stage 2) and its kernel tests; `e2e/autoware_tensorrt_vad` (MSDA, Rotate, SelectAndPad); `autoware_tensorrt_bevformer` (own engine code, three IPluginV2DynamicExt plugins ModulatedDeformableConv2d / MultiScaleDeformableAttn / Rotate in TRT and TRT2 variants, cuBLAS -> hipBLAS `gemm_ex`).

**Stage 4: sparse convolution (needs a ROCm spconv_cpp; Q1).**
The four spconv plugins (`GetIndicePairsImplicitGemm`, `GetIndicePairs`, `ImplicitGemm`, `IndiceConv`), `autoware_bevfusion` (spconv `Point2Voxel` voxeliser in `lib/preprocess/preprocess_kernel.cu:36,114,146`, sparse-conv backbone, QuickCumsumCuda) and `autoware_ptv3` (SegmentCSR plus sparse conv; its five synthetic GPU kernel tests can run from stage 2 onward since they need no model).

**Not scheduled pending a ruling:** `autoware_tensorrt_bevdet` (all GPU code is in the external `autowarefoundation/bevdet_vendor`, which has no licence; Q2).

**No port needed:** `autoware_bytetrack` lists `autoware_cuda_utils`, `autoware_tensorrt_common`, `autoware_cuda_dependency_meta` and `tensorrt_cmake_module` in package.xml but no source uses them; it builds on ROCm as is once those packages are in the workspace (confirm in stage 2). `sensing/autoware_pointcloud_preprocessor` mentions cuda_blackboard only in comments.

## CUDA surface inventory

Regex census (`surface.py`): 89 files, 177 `__global__`, 185 launches, 18 warp-intrinsic hits, CUB 65, Thrust 220, cuBLAS 150 (bevformer only), cuDNN 12 (bevformer `attachToContext` signatures only), cuRAND 1 (link only). No textures, no driver API, no NVRTC/PTX files. The census sees neither TensorRT usage nor CUDA runtime calls in .hpp/.cpp; those are added to surface.json by hand.

### Stage 1

`autoware_cuda_pointcloud_preprocessor`: 8 .cu, 36 kernels, no warp intrinsics. CUB `DeviceRadixSort::SortPairs`, `DeviceScan::InclusiveSum`, `DeviceReduce::{Sum,Min,Max}` over `thrust::make_transform_iterator`, `DeviceRunLengthEncode::Encode`, `DeviceMergeSort::SortPairs` with `cuda::std::less<cuda::std::optional<int>>` (all in hipCUB; checked `device_merge_sort.hpp`, `device_run_length_encode.hpp`), behind a generic `CubExecutor` lambda wrapper. Thrust `minmax_element`, `fill`, `adjacent_difference`, `inclusive_scan` with `thrust::cuda::par_nosync(alloc).on(stream)`; a `thrust::device_malloc_allocator` subclass; 16 `thrust::device_vector` members in a header compiled by the host compiler (`cuda_pointcloud_preprocessor.hpp:111-136`, included from `cuda_pointcloud_preprocessor_node.cpp`), so that .cpp must see rocThrust with `__HIP_PLATFORM_AMD__` and `hip::host` (it already compiles device_vector headers with g++ on CUDA). libcu++ as above. Float `atomicAdd` (downsample centroids, noise intensity), `atomicMax` u32, `atomicExch`. `__constant__` + `cudaMemcpyToSymbolAsync` (downsample `.cu:66,395`; undistort `.cu:30`). A `__device__` lambda (outlier `.cu:381`; `--extended-lambda` is nvcc-only, hip-clang needs nothing). `__device__ [[nodiscard]] inline` ordering (outlier `.cu:135,141,147`) to compile-check under hip-clang. Stream-ordered pools (`cudaMemPoolCreate`, `cuda_utils::create_memory_pool`, `cudaMallocFromPoolAsync`), `cudaDeviceGetAttribute(MultiProcessorCount)` for grid sizing, `cudaSetDeviceFlags(BlockingSync)`. Block sizes 256/512. Fresh memory: the outlier and noise filters accumulate with atomics into buffers from the pooled `cuda_utils::make_unique(n, stream, pool)`, which memsets (fork `cuda_unique_ptr.hpp:76`, checked); downsample and the thrust allocator memset themselves.

`autoware_ground_segmentation_cuda`: 11 kernels, **warp-synchronous by design**: `WARP_SIZE (32)` and `FULL_MASK (0xFFFFFFFF)` (`include/autoware/cuda_scan_ground_segmentation/cuda_common.hpp:30-35`), one 32-thread group per sector in `sectorProcessingKernel` (`src/cuda_scan_ground_segmentation/cuda_scan_ground_segmentation_filter.cu:420-436`), `__shfl_down_sync` reductions (`:337-342`, `:395-398`), `__shfl_sync(..., 0)` broadcasts (`:347-349`, `:417`, `:525-527`), `__ballot_sync` into `uint32_t` and `32 - __ffs(...)` (`:375-384`), `__syncwarp`. On wave64 two sectors share a wavefront: a width-less `__shfl_sync(x, 0)` broadcasts the lower sector's lane 0 into the upper sector, the ballot loses the upper half, and the 32-bit mask does not compile under HIP (64-bit mask required). Fix that is correct on both widths and leaves CUDA unchanged: keep 32-lane logical groups; pass `width = WARP_SIZE` to every `__shfl*` (CUDA semantics are identical with width 32); take the ballot as 64-bit on HIP and shift by `(lane_in_wave / 32) * 32` to extract this group's 32 bits; a 64-bit mask type on HIP. `__syncwarp` maps to a wave barrier; per-group shared slices are indexed by `threadIdx.x / WARP_SIZE`, unchanged. CUB `DeviceScan::ExclusiveSum`. Integer atomics. `computeCellId` (`.cu:94-104`) does not bound `point_cell_id` (upstream, not port-specific; AMD faults on the OOB write where CUDA may not: note it for the reviewer and guard only if it reproduces).

`autoware_probabilistic_occupancy_grid_map`: 13 kernels, no warp code, no CUB/Thrust. 64-bit `atomicMin` on `unsigned long long` packed range|bits, `__float_as_uint`. Eigen `Matrix3f`/`Vector3f` in device code (hip-clang supports Eigen in device code via `EIGEN_` HIP support; compile-check). `CUDA_SEPARABLE_COMPILATION ON` (`CMakeLists.txt:49`): check whether any cross-TU device call needs it; if none, compile without `-fgpu-rdc` on HIP, else use `-fgpu-rdc` on the HIP target only. Compile-time `USE_CUDA` split plus a runtime `use_cuda` constructor flag hard-coded per node (pointcloud node true, laserscan and fusion false). Its own `CudaPointCloud2` helper (`utils/cuda_pointcloud.hpp`), not cuda_blackboard's.

### Stages 2-4

Per-package details (kernels, TensorRT methods, models, tests) are in notes.md "Planning", "Package census". Summary of the port-relevant points:

- Kernels hipify mechanically everywhere except: `segment_csr.cu` (`autoware_tensorrt_plugins`, `FULL_MASK 0xffffffff` with `__shfl_down_sync`, `:31,71-72`; the loop never runs because TB = 1, so delete-or-64-bit-mask, no algorithm change), `__CUDA_ARCH__ >= 530/610/800` fast paths in MSDA and bevformer kernels (HIP silently takes the fallback; add `|| defined(__HIP_DEVICE_COMPILE__)` where the HIP half intrinsics exist), inline PTX `dp4a` in bevformer int8 paths (fallback compiles; int8 kCHW4 formats are TensorRT-only and unused on MIGraphX), `__half2` member access `.x/.y` in bevformer MSDA (check against `hip_fp16.h`), bf16 `__nv_bfloat16` in scatter atomics (map to `__hip_bfloat16`).
- `select_and_pad_kernel.cu:84-87` (`autoware_tensorrt_plugins`): `buf + sizeof(int) * Q` on an `int *` overruns its workspace 4x, and the CUB temp size is a hard-coded guess (`767 + 4Q`) rather than a query. Out-of-bounds writes fault on AMD. Fix in the port (query `DeviceSelect::Flagged` temp size, byte offsets); it is a shared-code fix with no `[ROCm]` prefix.
- `frnet` `preprocess_kernel.cu:408-409` builds a `thrust::device_vector` from raw device pointers (treated as host iterators); use `thrust::device_pointer_cast`. Centerpoint `generateFeatures_kernel` returns before `__syncthreads` (`preprocess_kernel.cu:225` vs `:252`), as does pointpainting (`preprocess_kernel.cu:168` vs `:188`): a whole-wave exit is harmless on AMD, a partial one is UB on both vendors; reviewer to confirm the exit is wave-uniform. Centerpoint/transfusion run Thrust on the default stream after a kernel on a created blocking stream: same legacy-stream semantics on HIP.
- `cudaMalloc`-based `make_unique` (centerpoint/transfusion local `cuda_utils.hpp`, non-pooled `autoware_cuda_utils::make_unique`) does not zero; every accumulator found is cleared per frame, but the porter re-checks each new allocation's first access.
- camera_streampetr `memory.cu:71,82` launches `<<<n, 1>>>` (one thread per block): correct, slow; leave.

## Risk list

- **Wavefront width**: ground segmentation (above) is the only warp-synchronous code in stage 1 and the riskiest change; validate on wave64 and wave32 with a scene where adjacent sectors differ. segment_csr's dead shuffle loop must still compile.
- **MIGraphX availability** on validators: not in the pip ROCm SDK. autoware_cuda round 2 establishes the recipe; stages 2-4 reuse it. MIGraphX on wave32 (gfx1100/gfx1201): check the target list of the build used before claiming the wave32 gate for stages 2-4.
- **ROS 2 and Autoware dependency closure**: stage 1 alone needs `autoware_pointcloud_preprocessor` (PCL, pcl_ros), `autoware_utils`, `autoware_point_types`, `autoware_agnocast_wrapper` and messages from autoware_core / autoware_msgs / tier4_autoware_msgs, and the occupancy grid needs `grid_map`, `nav2_costmap_2d`, `pointcloud_to_laserscan`, `laser_geometry`. No fleet image has ROS 2; earlier jobs built Jazzy from source in the job directory (cuda_blackboard notes, 152 packages). Expect a larger closure here: use `rosdep keys --from-paths` on the stage's packages to list it, build from source in the job directory, and record the recipe.
- **Model artifacts** for stages 2-4: public on Hugging Face (`AutowareFoundation/lidar_centerpoint` v4.1, `lidar_transfusion` v2.1, `bevfusion` v2.0, `camera_streampetr` v1.0, `tensorrt_vad` v0.1) or the Autoware artifacts S3 bucket (yolox, classifiers, traffic light, apollo, diffusion planner, simpl, shape estimation, frnet, ptv3, calibration status). bevformer's ONNX is only behind a personal SharePoint link (`autoware_tensorrt_bevformer/README.md:50`): it may have to be exported from the BEVFormer_tensorrt tooling, or bevformer validation stays owed.
- **Custom-op semantics**: MIGraphX custom ops declare static output shapes; data-dependent ops need padding plus counts, and the consumer code that reads sizes from TensorRT's size tensors changes accordingly.
- **Dynamic shapes**: transfusion, frnet, ptv3, bevfusion and calibration_status_classifier use wide min/opt/max profiles (bevfusion with min 0); depends on how round 2 implements profiles (native dynamic dims or per-shape compile cache) and its compile latency on a new shape.
- **int8** (yolox_s_plus_opt default, traffic light option): approximate on MIGraphX (Q2 = a).
- **rocPRIM radix sort**: argsort (int64 keys, bits 0-64), unique, the pointcloud preprocessor and ptv3 sort with full bit ranges, so the known `end_bit = 64, begin_bit > 0` defect does not apply; sorting uninitialised float keys with mixed NaNs would (check that sorted buffers are fully written).
- **Upstream reception**: about 25 packages; see Q4.
- **Agnocast** variant: optional (`ENABLE_AGNOCAST`), same open question as cuda_blackboard (Q5).
- Textures, 256-byte pitch, rule-of-five on texture handles: not applicable (none used).

## File-by-file change list

Per package, unless a stage says otherwise:

- `CMakeLists.txt`: HIP branch (backend option and `autoware_cuda_utils_USE_HIP` agreement, `enable_language(HIP)`, `.cu` as `LANGUAGE HIP`, `CMAKE_HIP_ARCHITECTURES` from the user or detection instead of gencode, `hip::host`/`roc::rocthrust`/`hip::hipcub` without exporting `hip::device` to g++ consumers (cuda_blackboard lesson), MIGraphX/`autoware_tensorrt_common` backend instead of `nvinfer` for TensorRT packages, tests wired to the same sources). The CUDA branch is untouched.
- Compat headers: per Q3.
- Source edits, guarded and minimal: Thrust policies; `HIP_SYMBOL` sites (unless macro'd); ground segmentation sub-wave shuffles/ballot; segment_csr mask; `__CUDA_ARCH__` conditions in MSDA/bevformer; streampetr fp32-layer path; frnet `device_pointer_cast`; select_and_pad workspace fix (shared fix, no prefix).
- `autoware_tensorrt_plugins`: a HIP/MIGraphX branch building `cuda_ops` from the hipified kernels and a custom-op library (new `src/migraphx/*.cpp`, one file per op family) installed at the TensorRT library's path; the TensorRT IPluginV3 sources are not built on that branch.
- `autoware_tensorrt_bevformer`: MIGraphX engine path and custom ops for its three plugins; cuBLAS calls through hipBLAS.
- Package READMEs: a short ROCm build note where the package documents its build.
- Not touched: `.github/` workflows, `build_depends_*.repos` (a ROCm workspace swaps `autoware_cuda` and `cuda_blackboard` for the AMD-Ecosystem forks at `moat-port`; documented in notes and the PR, not by editing the repos files), launch and parameter files (plugin library path kept).

## Build commands

In a ROS 2 Jazzy (or Humble) workspace with autoware_core, autoware_msgs, tier4_autoware_msgs and the stage's other dependencies, plus the forks of `autoware_cuda` and `cuda_blackboard` at `moat-port` and `autowarefoundation/autoware_cmake`:

```bash
source <ros2>/install/setup.bash
export ROCM_PATH=<rocm root>            # pip SDK: python -m rocm_sdk path --root
colcon build --packages-up-to autoware_cuda_pointcloud_preprocessor autoware_ground_segmentation_cuda autoware_probabilistic_occupancy_grid_map \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DAUTOWARE_CUDA_USE_HIP=ON \
               -DCMAKE_HIP_ARCHITECTURES=<gfx> -DCMAKE_PREFIX_PATH="$ROCM_PATH;<deps prefix>" \
               -DPython3_EXECUTABLE=<venv python>
```

Stages 2-4 add the stage's packages and the MIGraphX prefix to `CMAKE_PREFIX_PATH`. Confirm in each configure log that the package took the HIP branch: an Autoware GPU package without its toolkit prints a warning and installs only launch files, which looks like success.

CUDA no-regression (mi355 fleet host, CUDA 12.8 at `/usr/local/cuda`): build the same packages with nvcc on the CUDA branch, and for stages 2-4 with TensorRT 10 headers and libraries (NVIDIA/TensorRT `include/` and the `tensorrt-cu12` wheels, as autoware_cuda's plan describes), compile and link only; compare install trees with the unmodified upstream build.

## Test plan

GPU gate on each validated platform (Linux; repeat each GPU test 20 times, `--ctest-args --repeat until-fail:20`; a skip means no device was seen and fails the gate):

Stage 1
- `test_cuda_pointcloud_preprocessor`: 12 GPU `TEST_F` on `autoware::cuda_utils::CudaTest` plus 8 CPU `TEST`; `test_cuda_combine_cloud_handler`: 8 GPU `TEST_F`. All pass, none skipped.
- `autoware_ground_segmentation_cuda` has no tests upstream. Validator harness (job directory, not committed): a synthetic scene (ground plane with slope, curb, boxes, sparse far points; at least two adjacent sectors with different terrain) through `CudaScanGroundSegmentationFilter`; assert ground/non-ground labels against the construction, identical output across 20 runs, and identical labels on wave64 and wave32 platforms for the same input (store the input and the label hash in notes). A sector-boundary case is mandatory, because that is where the wave64 fault would show.
- `autoware_probabilistic_occupancy_grid_map`: its gtests and launch tests are commented out upstream (issue #7724, "disabled due to lack of GPU capable machines"; `CMakeLists.txt:157-184`). The validator builds them from the fork's own test sources in a scratch CMake project against the installed package (no fork edit) and runs `test_pointcloud_based_method.py` (GPU path) plus the CPU tests; plus a probe comparing the pointcloud-based OGM on the GPU with the same update on the CPU code path where one exists (`use_cuda` false), within a stated cell-agreement tolerance.
- Example launches run for each node with a recorded or synthetic cloud and produce output (cuda_blackboard topic delivery included).

Stage 2 (per package): the upstream GPU tests that exist (`test_tensorrt_yolox_node`, `test_tensorrt_yolox_detector`, `test_cnn_classifier`, `test_cnn_lamp_recognizer`, `test_traffic_light_recognition_node`, `test_model_inference`, `test_calibration_status_classifier`, `test_preprocessing`, the disabled centerpoint/transfusion kernel tests built from the fork's sources as above) with the real models installed, none skipped; for each detector without a test, a probe that runs the node on a fixed input and compares detections with a CPU ONNX Runtime run of the same model (fp32: box IoU and score within tolerance; fp16: stated looser tolerance). The CPU tests of every package (`test_label`, `test_simpl`, `test_shape_estimation`, traffic light CPU tests, ...) pass on the HIP build. ptv3's five synthetic GPU kernel tests run as soon as the package builds.

Stage 3: `reference_kernels_test` (argsort, segment_csr, unique) passes; new HIP-side reference tests for each custom op against a CPU reference (MSDA, Rotate, SelectAndPad, QuickCumsum, EfficientNMS, the bevformer ops), added in the HIP-only test file of the plugin package; VAD and bevformer end-to-end on their models with output compared to a CPU ONNX Runtime run where the model is runnable there (custom ops excluded: compare at op level).

Stage 4: per Q1.

Non-GPU regression: every package's CPU tests pass on the HIP build, and the CUDA branch configures, compiles and links unchanged with nvcc (+ TensorRT 10 for stages 2-4).

Gates: `wave64` and `wave32` on Linux; `windows` waived (Jeff Daily, 2026-11-11).

## Open questions

Q1-Q5 (spconv path, bevdet licence, compat-header placement, delivery shape, Agnocast) and the interface list for autoware_cuda round 2 are at the top of notes.md's `## Planning` section. Technical points the porter resolves and records without a ruling: libcu++ shim versus libhipcxx, `-fgpu-rdc` for the occupancy grid, bevformer on `TrtCommon` versus a local MIGraphX path, which standard ONNX ops in each model MIGraphX parses, the diffusion planner CUDA-graph path.
