# colmap -- porting plan

## Project

- Name: colmap
- Upstream: https://github.com/colmap/colmap (default branch `main`)
- Fork: https://github.com/AMD-Ecosystem/colmap, port branch `moat-port`
- Planned against upstream `d6d2bc8e0d3b4ba37139982d0b80f11af93deaf3` (2026-08-07)
- Licence: BSD-3-Clause, tier 1. Vendored `src/thirdparty` is separately licensed and
  COPYING.txt says so; SiftGPU (the subject of this port) is UNC Chapel Hill
  non-commercial. `approval_scope` stays `contribute-only`: that licence bars USE, not
  contribution, and we contribute through COLMAP's own process. Settled at intake, not
  re-opened here.

## Existing AMD support

**Classification: the existing AMD support is OURS and is already merged upstream. This is
"finish a partially-landed port", not a fresh port and not `already-supported`.**

- `colmap/colmap#4420` "Add ROCm/HIP support for patch_match_stereo (AMD GPU)" merged
  2026-08-05 as squash `b09267a21`. Opened by iShengnan; several commits ours.
- Upstream now documents the ROCm build itself (`doc/install.rst:122-151`) and states the
  covered scope in one line at `doc/install.rst:141`: *"The HIP backend currently
  accelerates dense reconstruction (`patch_match_stereo`)"*. Per
  `references/assess-existing-support.md`, that sentence is simultaneously the gap list and
  the line a follow-up pull request has to update.
- `README.md:31` says "AMD GPUs are supported via HIP/ROCm when building from source".
- No independent AMD or ROCm-org colmap effort exists. `AMD-Ecosystem/colmap` carries only
  inherited upstream branches.

Delivery vehicle: an ordinary upstream pull request. COLMAP merges platform support rather
than linking platform forks, and #4420 is the proof that the route works and that ahojnnes
reviews it.

Port versus rewrite: mechanical port. Nothing here is an NVIDIA-tuned performance kernel.
SiftGPU is 2007-vintage plain CUDA C with no CUTLASS, no CuTe, no wgmma, no warp
specialization and no warp intrinsics at all. There is no AMD-native rewrite to prefer.

### Do not rebase the parked branch

`jeffdaily/colmap:rocm-sift-gpu` @ `e41e06e0b` cannot be rebased onto current upstream: its
first four commits are inside the squash `b09267a21`, and #4420 continued for five more
commits the branch never saw. Re-derive. The branch is our own prior work and is a
legitimate reference for WHAT to change and WHY, but the diff must be written fresh against
current upstream. The three commits worth re-deriving:

- `bf064e920` "Enable GPU SIFT (SiftGPU) under ROCm/HIP" -- build wiring plus the dispatch
  sites.
- `e95eb3806` "fix double-destroy and DoG edge OOB" -- the two runtime faults.
- `3345a9819` "route tex2D through linear binding on HIP" -- the pitch-alignment fix.

`690348f33` (version banner) and `566e4df7e` / `e41e06e0b` (docs) are superseded: upstream
already documents the ROCm build. Only the coverage SENTENCE needs editing now.

## Build classification: cmake -> Strategy A

Evidence:

- Root `CMakeLists.txt:127` `project(COLMAP LANGUAGES C CXX)`; no `find_package(Torch)`,
  no `torch.utils.cpp_extension`, no `CUDAExtension`, no torch dependency anywhere.
- `CMakeLists.txt:40-51` already declares `option(CUDA_ENABLED ...)` and
  `option(HIP_ENABLED ...)`, mutually exclusive, with the CMake floor raised to 3.21 only
  when HIP is requested.
- `cmake/FindDependencies.cmake:129-187` already does `find_package(hip/hiprand/rocrand)`,
  `enable_language(HIP)`, ROCm-root and architecture detection, and defines
  `COLMAP_HIP_ENABLED`.
- `src/colmap/mvs/CMakeLists.txt:292-320` marks the MVS `.cu` files `LANGUAGE HIP`.
- `src/colmap/util/cuda_to_hip.h` (141 lines) is the single compat header.

So Strategy A is not merely the right choice, it is the shape already in the tree. The port
extends an established pattern rather than introducing one. `ext_type` = `cmake`.

## What is actually left (the intake lead, verified)

Intake recorded "the remaining gap is GPU SIFT". Verified against the source, and it is
correct but INCOMPLETE. Three further residual gaps were found that a SIFT-only reading
would miss, and all three are consequences of #4420 having been scoped to
`patch_match_stereo`.

### 1. GPU SIFT is not built at all on a ROCm build (the main gap)

`src/thirdparty/SiftGPU/CMakeLists.txt:12` gates the CUDA sources strictly
`if(CUDA_ENABLED)`. `grep -ri hip src/thirdparty/SiftGPU/` returns zero matches. On a ROCm
build COLMAP silently falls back to CPU SIFT (or OpenGL/GLSL SIFT if a GUI build supplies a
context), which for a typical structure-from-motion run is the single most-used GPU path in
the program.

### 2. `GPU_ENABLED` does not know about HIP

`cmake/FindDependencies.cmake:643-648`:

    set(GPU_ENABLED OFF)
    if(OPENGL_ENABLED OR CUDA_ENABLED)
        list(APPEND COLMAP_COMPILE_DEFINITIONS COLMAP_GPU_ENABLED)
        set(GPU_ENABLED ON)
    endif()

`HIP_ENABLED` is absent. `GPU_ENABLED` is what gates `add_subdirectory(SiftGPU)`
(`src/thirdparty/CMakeLists.txt:29`) and linking `colmap_sift_gpu` into `colmap_feature`
(`src/colmap/feature/CMakeLists.txt:73`), and `COLMAP_GPU_ENABLED` is what compiles the
whole `SiftGPUFeatureExtractor` / `SiftGPUFeatureMatcher` region of
`src/colmap/feature/sift.cc`. So on a headless ROCm build today COLMAP has no GPU feature
path compiled in at all.

### 3. Nine dispatch sites still test `COLMAP_CUDA_ENABLED` alone

These decide whether the GPU path is taken, not whether it exists, so they are as
load-bearing as the kernels:

| site | what it decides on HIP today |
|---|---|
| `feature/sift.cc:581` | never passes `-cuda <idx>` to SiftGPU, so the extractor never selects the compute backend |
| `feature/sift.cc:1374` | matcher calls `SetLanguage(SIFTMATCH_GLSL)` instead of `SIFTMATCH_CUDA_DEVICE0 + idx` |
| `feature/sift.cc:1400` | emits the misleading "OpenGL version only supports N matches" warning |
| `feature/extractor.cc:103` | `RequiresOpenGL()` returns `use_gpu`, wrongly demanding a GL context |
| `feature/matcher.cc:86` | same, for matching |
| `controllers/feature_extraction.cc:416` | `gpu_index = -1` never expands to all devices |
| `controllers/feature_matching_utils.cc:68,267` | per-worker device selection and the same expansion |
| `controllers/automatic_reconstruction.cc:421` | **skips patch match stereo entirely on ROCm** and logs "CUDA is not available", although the HIP patch match is merged and works |
| `ui/dense_reconstruction_widget.cc:395` | GUI dense tab disabled on ROCm |
| `pycolmap/pipeline/mvs.cc:9`, `pycolmap/util/cuda.cc:8`, `pycolmap/util/bindings.cc:10,20`, `pycolmap/utils.h:9` | pycolmap exposes no `PatchMatchOptions` / `patch_match_stereo`, no `get_num_cuda_devices`, and `Device.AUTO` resolves to CPU |

`controllers/automatic_reconstruction.cc:421` and the pycolmap group are genuine unfinished
business from the merged pull request, independent of SIFT. `exe/mvs.cc:260` was updated in
#4420 and is the model to follow.

### 4. Symforce-Caspar: a CUDA surface that GREW after our merge

`src/thirdparty/Symforce-Caspar` (482 generated `.cu`, Apache-2.0) did not exist when #4420
was written. `CASPAR_ENABLED` defaults OFF (`CMakeLists.txt:77`) and
`CASPAR_ENABLED AND NOT CUDA_ENABLED` is a hard `FATAL_ERROR`.

**Scoped out of this port.** Reasons, in order of weight:

1. Cooperative groups, 2392 references, including `cg::labeled_partition`,
   `cg::binary_partition`, `cg::coalesced_threads`, `cg::memcpy_async`, `cg::reduce` and
   `cg::this_grid`. HIP's cooperative-groups support does not cover that surface. This is
   not a rename problem; several of those have no HIP analogue.
2. `generated/f32/CMakeLists.txt:9` pins `CMAKE_CUDA_ARCHITECTURES 75 80 86 89` with the
   comment "sm_75 minimum required by cooperative_groups::reduce".
3. The code is GENERATED by `caspar_generate.py`. A correct fix belongs in the generator,
   which is a different change in a different place, and hand-editing 482 generated files
   would be undone by the next regeneration.
4. `cub::DeviceReduce::Sum`, `cub::DeviceRadixSort::SortPairs`/`SortKeys` map to hipCUB
   cleanly and are the easy part; they are not the blocker.

Registered in `data/deferred.json` as a feature-port deferral rather than dropped.

### 5. Deliberately untouched

- `estimators/bundle_adjustment_ceres.cc:141` and `estimators/global_positioning.cc:361`
  select Ceres' own CUDA linear solvers. That is Ceres' backend, not COLMAP's, and Ceres has
  no ROCm support upstream. Not a COLMAP source port.
- `feature/onnx_utils.cc:145` selects the ONNX Runtime CUDA execution provider. Switching to
  the ROCm execution provider is a dependency and packaging change, not a kernel port.
- The OpenGL/GLSL SiftGPU backend stays exactly as it is. It is the fallback and it works.

## CUDA surface inventory

### `src/thirdparty/SiftGPU/ProgramCU.cu` (1978 lines, the only `.cu` in the directory)

- **28 `__global__` kernels**: `FilterH`, `FilterV`, `UpsampleKernel`, `DownsampleKernel`
  (x2), `ChannelReduce_Kernel`, `ChannelReduce_Convert_Kernel`, `ConvertByteToFloat_Kernel`,
  `ComputeDOG_Kernel` (x2), `ComputeKEY_Kernel`, `InitHist_Kernel`, `ReduceHist_Kernel`,
  `ListGen_Kernel`, `ComputeOrientation_Kernel`, `ComputeDescriptor_Kernel`,
  `ComputeDescriptorRECT_Kernel`, `NormalizeDescriptor_Kernel`, `ConvertDOG_Kernel`,
  `ConvertGRD_Kernel`, `ConvertKEY_Kernel`, `DisplayKeyPoint_Kernel`, `DisplayKeyBox_Kernel`,
  `MultiplyDescriptor_Kernel`, `MultiplyDescriptorG_Kernel`,
  `MultiplyDescriptorGRay_Kernel`, `RowMatch_Kernel`, `ColMatch_Kernel`.
- **`__constant__`**: `d_kernel[33]` at line 100, written with `cudaMemcpyToSymbol`.
  Already aliased in `cuda_to_hip.h`.
- **Warp intrinsics: none.** No `__shfl*`, no `__ballot`, no `__activemask`, no `warpSize`.
  This is a genuinely low-risk kernel set for the wave32-versus-wave64 class.
- **Device math**: `__mul24` (as `IMUL`), `__fdividef` (as `FDIV`), `__saturatef`,
  `__int_as_float`. All present in HIP.
- **Textures**: 44 `cudaTextureObject_t`, 61 `tex1Dfetch`, 3 `tex2D`, one `cudaArray` path
  (`CopyToTexture2D`, dead: `SIFTGPU_ENABLE_LINEAR_TEX2D` is never defined so it compiles,
  but `InitTexture2D`/`CopyToTexture2D` are only reachable from the unused array binding).
  `cudaCreateTextureObject` / `cudaDestroyTextureObject` / `cudaCreateChannelDesc` (39) /
  `cudaResourceTypeLinear` / `cudaResourceTypePitch2D`. Two static `cudaTextureDesc`
  singletons at lines 102 and 114, both `cudaFilterModePoint`, `cudaAddressModeClamp`,
  non-normalized coords; the second uses `cudaReadModeNormalizedFloat`.
- **Runtime**: `cudaMalloc`, `cudaFree`, `cudaFreeArray`, `cudaMallocArray`, `cudaMemcpy`,
  `cudaMemcpyAsync`, `cudaMemcpy2DToArray`, `cudaGetDevice`, `cudaSetDevice`,
  `cudaGetDeviceCount`, `cudaGetDeviceProperties`, `cudaDeviceSynchronize`,
  `cudaGetLastError`, `cudaGetErrorString`.
- **No cuBLAS, no cuFFT, no cuRAND, no cuSPARSE, no Thrust, no CUB**, no streams beyond the
  default, no events, no pinned or managed memory, no cooperative groups. There is no
  library substitution to make anywhere in SiftGPU.
- **Absent by design**: no driver API, no NVRTC, no runtime-compiled PTX, no non-C++ build
  path.

Mapping: every symbol above has a direct HIP equivalent, and roughly half are already
aliased in `cuda_to_hip.h` from the MVS work. The additions needed are
`cudaResourceTypeLinear`, `cudaResourceTypePitch2D`, `cudaChannelFormatKindFloat` /
`Signed` / `Unsigned` / `None`, `cudaMallocArray`, `cudaMemcpy2DToArray`, and (already
present) `cudaMemcpyToSymbol`.

### `src/thirdparty/SiftGPU/CuTexImage.{h,cpp}` (78 + 294 lines, host C++)

Device-memory wrapper. `InitTexture` (`cudaMalloc`), `CopyFromHost` / `CopyToHost`,
`BindTexture` (linear), `BindTexture2D` (pitch2D), and the `CuTexObj` RAII handle. Also the
**legacy CUDA-OpenGL PBO interop**: `cudaGLRegisterBufferObject`, `cudaGLMapBufferObject`,
`cudaGLUnmapBufferObject`, `cudaGLUnregisterBufferObject` at lines 112-113, 146-147,
258-264, 282-286, plus `#include <cuda_gl_interop.h>`.

**That legacy API has no HIP equivalent.** ROCm 7.2.1's `hip/amd_detail/amd_hip_gl_interop.h`
exports only `hipGraphicsGLRegisterBuffer`, `hipGraphicsGLRegisterImage`, `hipGLGetDevices`
and the `hipGLDeviceList*` enum. There is no `hipGLMapBufferObject`.

It is safe to compile those call sites out on ROCm, and the evidence is that COLMAP never
reaches them:

- `PyramidCU::ConvertInputToCU` (`PyramidCU.cpp:925-951`) takes the
  `if(input->_pixel_data)` branch, which is `InitTexture` + `CopyFromHost`. COLMAP always
  supplies host pixel data: `sift.cc:694` calls `RunSIFT(pitch, height, data, GL_LUMINANCE,
  GL_UNSIGNED_BYTE)`.
- The PBO branches live in `PyramidCU::ConvertTexCU2GL`, reachable only via
  `GetLevelTexture`, which is called only from `SiftGPU.cpp:491,544,588` inside SiftGPU's own
  `SiftGPUEX` viewer. `grep -rn 'GetLevelTexture|DisplayKeyPoint|DisplayKeyBox|SiftGPUEX'
  src/colmap/ src/pycolmap/` returns nothing.

So on ROCm this is a scope-out with a proof, not a hole. The CUDA path must remain
byte-identical.

### `src/thirdparty/SiftGPU/PyramidCU.cpp` (1196) and `SiftMatchCU.cpp` (248)

Host drivers. They call `ProgramCU::*`, hold `CuTexImage` members, and include
`<cuda_runtime.h>` / `GL/glew.h`. `SiftMatchCU.cpp:240` calls `cudaGetLastError`.
`ProgramCU::CheckCudaDevice` (`ProgramCU.cu:1366`) reads `deviceProp.major`/`minor` and
`totalGlobalMem`; `hipDeviceProp_t` provides all three.

### Already ported and merged (regression surface, not new work)

`src/colmap/mvs/{gpu_mat_prng.cu, gpu_mat_ref_image.cu, patch_match_cuda.cu,
gpu_mat_test.cu}` and `src/colmap/util/{cuda.cc, cudacc.cc, cuda_to_hip.h}`. This port must
not disturb them. Note `patch_match_cuda.cu:1631` carries a runtime gfx9 check that falls
back to point filtering for layered textures; leave it alone.

## Risk list

1. **`CuTexObj` rule-of-five double-destroy -- confirmed present, will fault on AMD.**
   `CuTexImage.h:43-47` declares `struct CuTexObj { cudaTextureObject_t handle; ~CuTexObj(); }`
   with no default member initializer and no copy or move members. The destructor
   (`CuTexImage.cpp:43`) calls `cudaDestroyTextureObject(handle)` unconditionally.
   `ProgramCU.cu:959-971` default-constructs `texObjF4` and `texObjList` (handle
   uninitialized), then copy-ASSIGNS one of them from `BindTexture(...)`; the temporary is
   destroyed immediately, so the surviving object holds a destroyed handle and destroys it
   again at scope exit. In the `existing_keypoint` branch `texObjList` is never bound at all,
   so a garbage handle is both passed to the kernel and destroyed. CUDA tolerates this; AMD
   does not. Fix: `handle = 0` default init, move-only with move ctor and move assignment
   that null the source, and a destructor guarded on `handle != 0`. This is the standing
   fault class in the skill, and colmap is the entry's source project.

2. **`ComputeDOG_Kernel` out-of-bounds neighbour reads -- confirmed present.**
   `ProgramCU.cu:467-486`: inside `if(col < width && row < height)` it fetches
   `index - 1`, `index + 1`, `index - width`, `index + width`. At the first and last pixel
   those are outside the bound buffer. The texture object is `cudaResourceTypeLinear`, where
   `addressMode` does not apply, so it is a real out-of-range fetch and not a clamp. Fix:
   clamp the neighbour indices to image bounds. Clamp UNCONDITIONALLY (decided; see Open
   questions 1) rather than behind an `#ifdef COLMAP_HIP_ENABLED` guard, because a
   correctness fix that both backends want is a better upstream contribution than a
   platform-conditional; measure before deciding.

3. **256-byte texture pitch on `BindTexture2D` -- confirmed present, blocks real images.**
   `CuTexImage.cpp:66-83` binds `cudaResourceTypePitch2D` with
   `pitchInBytes = _imgWidth * _numChannel * sizeof(float)` from a tightly packed
   `cudaMalloc`. AMD requires 256-byte row pitch (32 on NVIDIA), so
   `hipCreateTextureObject` returns `hipErrorInvalidValue` for any pyramid level whose row is
   not a multiple of 256 bytes. Measured previously: a 640x480 input downsamples to an
   80-wide float2 level, a 640-byte row, which fails.
   The cheap fix is the right one: only 3 `tex2D` fetches exist (`ProgramCU.cu:851, 1033,
   1106`), all `cudaFilterModePoint`, and all three kernels already clamp x and y to image
   bounds before fetching, so hardware address clamping is not relied on either. Point
   sampling over a pitch2D bind is exactly `tex1Dfetch(tex, int(y) * width + int(x))` over a
   linear bind, and linear bindings have no pitch requirement. Replace the two
   `BindTexture2D` call sites (`ProgramCU.cu:973, 1187`) and the three fetches.
   Do NOT take the `cudaMallocPitch` refactor: it would touch the row indexing of every
   kernel in the file for no gain here.

4. **A headless build makes every GPU SIFT test silently no-op and PASS.** This is the
   highest-value item in the whole plan and it has already produced one false green.
   `sift_test.cc:116-137` wraps every GPU test in `RunGpuTest`, which builds a `QApplication`
   and calls `RunThreadWithOpenGLContext(&thread)`. With `GUI_ENABLED=OFF`,
   `util/opengl_utils.h:89-96` makes `OpenGLContextManager::MakeCurrent()` return false and
   `RunThreadWithOpenGLContext()` an empty inline function -- **the test body is never
   executed**. A headless run therefore reports `ExtractSiftFeaturesGPU.Nominal` and every
   `MatchSiftFeaturesGPU` test as passing without touching the GPU. The earlier "100% tests
   passed, 145 tests" on this project was exactly that. Validation MUST use
   `-DGUI_ENABLED=ON` under `xvfb-run`, and must check the per-test wall time: these tests
   take hundreds of milliseconds when they really run and ~0 when they do not.
   Note this affects CUDA identically, so it is upstream behaviour and not something the
   port introduced. Changing it is out of scope; detecting it is mandatory.

5. **`_IsNvidia == 0` forces `_UseCUDA = 0`.** `GlobalUtil.cpp:370`, inside
   `InitGLParam(NotTargetGL=0)`, reads the GL vendor string and disables the compute backend
   on any non-NVIDIA GPU. The compute paths call `InitGLParam(1)`
   (`PyramidCU.cpp:82`, and `SiftMatch.cpp:686-688` short-circuits before the `InitGLParam(0)`
   branch when the language is already CUDA), so the ordinary flow avoids it. But
   `sift_test` runs GLSL and compute tests in one process, `PyramidGL.cpp:178` and
   `SiftMatch.cpp:136` both call `InitGLParam(0)`, and `GlobalUtil` is process-global static
   state. If a GLSL test initializes GL first, a later compute test on an AMD GL context
   sees `_UseCUDA` cleared and silently falls back to GLSL. Watch for a GPU test that passes
   alone and behaves differently in-suite, and check the "[SiftGPU Language]" line
   (`SiftGPU.cpp:163`). Prefer widening the vendor test over adding a new global.

6. **Wave32 versus wave64.** Low risk, stated explicitly because the gate demands both.
   There are no warp intrinsics and no `warpSize`. The three hardcoded 32s are block
   dimensions, not warp assumptions: `ROWMATCH_BLOCK_WIDTH 32`, `COLMATCH_BLOCK_WIDTH 32`,
   and `FILTERV_BLOCK_HEIGHT 32`. `RowMatch_Kernel`'s tree reduction
   (`ProgramCU.cu:1911-1923`) has an explicit `__syncthreads()` inside the loop, so it does
   not rely on warp-synchronous execution and is correct at either width; a 32-thread block
   is simply a partial wave on wave64. Every early `return` in the file occurs AFTER the
   kernel's last `__syncthreads()`, so the intra-wave barrier-divergence class does not
   apply. Confirm rather than assume on the wave32 host.

7. **Compute-only CDNA has no graphics pipeline.** gfx90a cannot provide a hardware GL
   context, so the GLSL SiftGPU backend and the `RunGpuTest` context both run on software GL
   (Mesa llvmpipe under Xvfb). That is fine and is how the GPU tests get their context, but
   it means a GLSL-versus-compute comparison on that host compares llvmpipe against the AMD
   GPU. Do not read a GLSL test's timing or results as GPU evidence.

8. **`colmap_estimators` and `colmap_controllers` link `colmap_util_cuda` only under
   `CUDA_ENABLED`.** `controllers/CMakeLists.txt:72` has no `HIP_ENABLED` arm, so enabling
   the patch-match call site in `automatic_reconstruction.cc` will fail to LINK until that
   arm is added. `exe/CMakeLists.txt:44-49` already has the pattern to copy.
   `estimators/CMakeLists.txt:83` needs no arm: its only CUDA use is the Ceres solver, which
   is out of scope.

9. **Link-time OpenGL dependency on a headless ROCm build.** `colmap_sift_gpu` links
   `OpenGL::GL` and `GLEW::GLEW` unconditionally (`SiftGPU/CMakeLists.txt`), and
   `ProgramCU.cu:24` includes `GL/glew.h`, so a ROCm build needs libGL and GLEW development
   packages even with `OPENGL_ENABLED=OFF`. Install them; do not try to strip the dependency.

10. **CUDA no-regression.** Every change must leave the CUDA build a pure passthrough. This
    matters more than usual here because the port touches vendored third-party code that
    upstream did not write. Compile-check the CUDA path with nvcc before the pull request
    (skill `references/validation.md`, PR-prep gate).

## File-by-file change list

### Build

| file | change |
|---|---|
| `cmake/FindDependencies.cmake:643-648` | include `HIP_ENABLED` in the `GPU_ENABLED` condition |
| `src/thirdparty/SiftGPU/CMakeLists.txt` | add a parallel `if(HIP_ENABLED)` arm beside the existing `if(CUDA_ENABLED)` arm: define `SIFTGPU_CUDA_ENABLED`, list the same sources, link `colmap_util_cuda` + `hip::host` instead of `CUDA::cudart`/`CUDA::curand`, and `set_source_files_properties(ProgramCU.cu PROPERTIES LANGUAGE HIP)`. Keep `OPTIONAL_CUDA_*` untouched and add `OPTIONAL_HIP_*` alongside so the CUDA build is unchanged |
| `src/colmap/controllers/CMakeLists.txt:72-77` | add the `if(HIP_ENABLED)` arm linking `colmap_util_cuda` and, under `MVS_ENABLED`, `colmap_mvs_cuda` |

### Compat header

| file | change |
|---|---|
| `src/colmap/util/cuda_to_hip.h` | add `cudaResourceTypeLinear`, `cudaResourceTypePitch2D`, `cudaChannelFormatKindFloat`/`Signed`/`Unsigned`/`None`, `cudaMallocArray`, `cudaMemcpy2DToArray`. Keep the file the only place that names a `hipXxx` symbol |

### SiftGPU sources

| file | change |
|---|---|
| `ProgramCU.cu` | include `colmap/util/cuda_to_hip.h` under `COLMAP_HIP_ENABLED` instead of the CUDA headers; clamp the four `ComputeDOG_Kernel` neighbour indices UNCONDITIONALLY (not `#ifdef`-guarded -- Open questions 1); replace the 3 `tex2D<float2>` fetches with `tex1Dfetch<float2>` and the 2 `BindTexture2D` call sites with `BindTexture`, also unconditionally |
| `CuTexImage.h` | give `CuTexObj` `handle = 0`, an explicit move constructor and move assignment that null the source, deleted copies, and a destructor guarded on `handle != 0` |
| `CuTexImage.cpp` | same include switch; compile the four `cudaGL*` PBO regions out under `COLMAP_HIP_ENABLED` with a comment naming `hipGraphicsGLRegisterBuffer` as the modern API and stating that COLMAP does not reach the PBO path |
| `PyramidCU.cpp`, `SiftMatchCU.cpp` | include switch only |

### Test infrastructure (headless GPU tests -- Open questions 2)

| file | change |
|---|---|
| `src/colmap/util/opengl_utils.h` | `RunThreadWithOpenGLContext` is an empty inline when `GUI_ENABLED=OFF`, which silently turns every GPU test body into a no-op that reports PASS. Give it a headless path that actually runs the thread; a GPU test needs a GPU, not a window |
| `src/colmap/feature/sift_test.cc` | `RunGpuTest` (line 116) must execute its body on a headless host. Keep the windowed path for a GUI build |
| test | `opengl_utils_test` moves back INTO scope -- it covers the function being changed, and it was scoped out only while this was not our work |

Both backends get this. It is shared test infrastructure and the pull request body must say so
plainly, naming the no-op it removes: this project has already reported "145 tests passed" with
every GPU body skipped, and that is the failure being fixed.

### Dispatch sites (widen `COLMAP_CUDA_ENABLED` to `COLMAP_CUDA_ENABLED || COLMAP_HIP_ENABLED`)

`feature/sift.cc:581,1374,1400`; `feature/extractor.cc:103`; `feature/matcher.cc:86`;
`controllers/feature_extraction.cc:416`; `controllers/feature_matching_utils.cc:68,267`;
`controllers/automatic_reconstruction.cc:421`; `ui/dense_reconstruction_widget.cc:395`;
`pycolmap/pipeline/mvs.cc:9`; `pycolmap/util/cuda.cc:8`; `pycolmap/util/bindings.cc:10,20`;
`pycolmap/utils.h:9`. Also `controllers/feature_matching_utils.cc:39` includes `<cuda_runtime.h>` directly and
`feature/sift_test.cc:248` gates `CreateSiftGPUMatcherCUDA.Nominal` on CUDA alone; route the
first through `cuda_to_hip.h` and widen the second so the test runs on ROCm. Follow the shape already used at `exe/mvs.cc:260`.

### Docs

| file | change |
|---|---|
| `doc/install.rst:141` | replace "currently accelerates dense reconstruction (`patch_match_stereo`)" with the new coverage: dense reconstruction plus GPU SIFT feature extraction and matching. This is the sentence that states the scope, so it is the sentence the pull request updates |
| `README.md:31` | leave as is; it is already correct and unqualified |

## Build commands (gfx90a)

Dependencies this host lacks and the builder must install: `libglew-dev`, `libgl1-mesa-dev`,
`xvfb`, Qt6 (for `GUI_ENABLED=ON`), plus COLMAP's usual Ceres/glog/gflags/FreeImage/Boost
stack.

Configuration A, headless. Builds everything and runs the non-GPU suite. Fast, and enough
for the compile and regression check. It CANNOT validate GPU SIFT (risk 4).

    cmake -S . -B build-hip -GNinja \
      -DCUDA_ENABLED=OFF -DHIP_ENABLED=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_PREFIX_PATH=/opt/rocm \
      -DCMAKE_BUILD_TYPE=Release \
      -DTESTS_ENABLED=ON -DGUI_ENABLED=OFF \
      -DCGAL_ENABLED=OFF -DDOWNLOAD_ENABLED=OFF -DONNX_ENABLED=OFF
    cmake --build build-hip -j"$(nproc)"

Configuration B, GUI plus Xvfb. **This is the validation build.**

    cmake -S . -B build-hip-gui -GNinja \
      -DCUDA_ENABLED=OFF -DHIP_ENABLED=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_PREFIX_PATH=/opt/rocm \
      -DCMAKE_BUILD_TYPE=Release \
      -DTESTS_ENABLED=ON -DGUI_ENABLED=ON \
      -DCGAL_ENABLED=OFF -DDOWNLOAD_ENABLED=OFF -DONNX_ENABLED=OFF
    cmake --build build-hip-gui -j"$(nproc)"

Notes. `-DCMAKE_PREFIX_PATH=/opt/rocm` is required on a clean ROCm container even though it
is redundant when `/opt/rocm/bin` is on `PATH`; the upstream install document should say so
too. Do not pin `CMAKE_HIP_ARCHITECTURES` in committed CMake: `enable_language(HIP)` plus
the existing `rocm-sdk` detection already resolves it, and another architecture reuses the
recipe with only `-DCMAKE_HIP_ARCHITECTURES=<arch>`. Add
`-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++` only if CMake fails to find the HIP
compiler.

Wrap every phase in `utils/timeit.sh colmap <phase> -- <cmd>`.

## Test plan

### GPU tests that must really run (Configuration B, under Xvfb)

    xvfb-run -a ctest --test-dir build-hip-gui -R 'feature/sift_test' --output-on-failure -V

`sift_test` is the gate. Its GPU tests are `ExtractSiftFeaturesGPU.Nominal`,
`CreateSiftGPUMatcherOpenGL.Nominal`, `CreateSiftGPUMatcherCUDA.Nominal`,
`MatchSiftFeaturesGPU.{Nominal,TypeMismatch}`,
`MatchGuidedSiftFeaturesGPU.{Nominal,TypeMismatch,EssentialMatrix,Spherical,SphericalMixedHemispheres,UnprojectableKeypoints,SharedFocal,SharedFocalPerPairFocal}`,
and the two that matter most:

- **`MatchSiftFeaturesCPUvsGPU.Nominal`** (`sift_test.cc:1251`)
- **`MatchGuidedSiftFeaturesCPUvsGPUGuided.EssentialMatrix`** (`sift_test.cc:1497`)

Those two compare the GPU result against the CPU result inside the suite. They are the
answer to the equivalence question ahojnnes asked on #4420 and the reason this port has a
better evidence story than that one did.

**Mandatory anti-no-op check.** Before believing a green `sift_test`, confirm the GPU tests
actually executed. Any of:

- `ExtractSiftFeaturesGPU.Nominal` takes hundreds of milliseconds, not ~0
- the suite log contains `[SiftGPU Language]: CUDA` (SiftGPU's own name for the compute
  backend; it is HIP here)
- `rocm-smi` shows utilization during the run

A `sift_test` that finishes in ~0.15s has proven nothing.

### Full suite, no regressions

    xvfb-run -a ctest --test-dir build-hip-gui -j"$(nproc)" --output-on-failure

~148 registered tests. `ctest` names are `<folder>/<name>`
(`cmake/CMakeHelper.cmake:164`). The non-GPU regression set is everything outside
`feature/sift_test` and `mvs/gpu_mat_test`: `math/*`, `geometry/*`, `scene/*`,
`estimators/*`, `sfm/*`, `controllers/*`, `image/*`, `util/*`. None of them should change,
and any that does is a real regression.

### The merged MVS path must not regress

    ctest --test-dir build-hip -R 'mvs/' --output-on-failure

`mvs/gpu_mat_test` is a real HIP GPU test and it was already passing. Also exercise
`automatic_reconstruction` once the dispatch site is widened, since that call path is new on
ROCm.

### End-to-end

A small image set (3 images is enough) through
`colmap feature_extractor --FeatureExtraction.use_gpu 1` and
`colmap exhaustive_matcher --FeatureMatching.use_gpu 1`, then `automatic_reconstructor` for
the dense path. Do this at two resolutions, one of which produces a pyramid level whose row
is not 256-byte aligned (640x480 is the known case, per risk 3), because that is exactly the
input that used to fail and a 1024x768-only run will not catch a regression of it.

### CUDA no-regression

nvcc compile-check the `CUDA_ENABLED=ON` configuration on this GPU-less host before the pull
request, reaching the link stage for at least `colmap_sift_gpu` plus one executable. Report
it honestly as compile-checked and not run.

### Coverage gates

wave64 from gfx90a. wave32 from any RDNA host; risk 6 says the exposure is small but it is
not zero and it must be run, not reasoned about. windows is not expected to be reachable
here (COLMAP's Windows build plus Qt plus Xvfb-equivalent), so a waiver request is likely;
an agent may only suggest one.

## Open questions

1. **DECIDED (jeffdaily, 2026-08-08): unconditional, not `#ifdef`.** The `ComputeDOG`
   neighbour clamp and the `tex2D`-to-linear rebind both go in for every backend. They are
   correctness fixes rather than AMD workarounds -- the out-of-bounds read is out of bounds
   on NVIDIA too -- and guarding a correctness fix behind the platform that happened to
   surface it leaves the bug in place for everyone else and makes the diff harder to review.
   The CUDA no-regression run (see Test plan) is still required and is now the evidence for
   the change rather than the trigger for a decision: run it, report the numbers, and if
   CUDA results MOVE, stop and say so rather than reaching for the guard.

2. **DECIDED (jeffdaily, 2026-08-08): make `RunGpuTest` work headless, in this change.**
   The point is to test as much as possible on headless platforms, which is what MOAT's
   hosts are. Leaving it out preserves the exact trap that already produced one false green
   on this project -- 145 tests "passing" while every GPU body was skipped -- and every
   headless run after it, ours and upstream's, would keep reporting coverage it does not
   have. That it also helps NVIDIA is an argument for it, not against.

   Say so plainly in the pull request body: it is shared test infrastructure, it is
   deliberate, and here is the no-op it removes. A reviewer weighing two things is a smaller
   cost than a reviewer trusting a green run that proved nothing.

3. **Symforce-Caspar.** Scoped out above with reasons. If it is ever wanted, the work starts
   at `caspar_generate.py` and at HIP's cooperative-groups coverage, not at the 482 generated
   files. Registered in `data/deferred.json`.

4. **Windows gate.** Almost certainly a waiver request. A person decides.
