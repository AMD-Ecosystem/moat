# openMVS port plan

Planner run 2026-11-20 on linux-gfx942. Analysis against a shallow clone of upstream `develop` at `14ad2571aac650bfabf54296d31aabed7a19fdde` (2026-11-06). Intake read `efa0b619` three days earlier; the facts it recorded still hold at this head except where noted below.

**Two scope questions are open for Jeff and must be ruled before a porter starts.** They are stated in full under "Open questions" and at the top of the planner section of `notes.md`:

1. gfx942/gfx950 (MI300/MI355) have no HIP texture/surface API, and every openMVS kernel reads its images through hardware-filtered textures. Either support those GPUs with a buffer-backed fallback in this port, or block them and get wave64 coverage from gfx90a.
2. Port the vendored SiftGPU's CUDA backend (`ports/siftgpu`, used by the SFM library) too, or leave SiftGPU on its GLSL path in a ROCm build.

The plan below is written so either answer slots in: the core work (sections 1 to 6 of the change list) is the same either way, and each ruling adds or removes one marked work item.

## Project

- Name: openMVS
- Upstream: https://github.com/cdcseacave/openMVS (AGPL-3.0, tier 3: a person clears the licence before anything goes upstream)
- Default branch: `develop`
- Fork: https://github.com/AMD-Ecosystem/openMVS, port branch `moat-port`
- `depends_on`: none (HIP runtime and rocRAND/hipRAND only)

## Existing AMD support

None. Intake checked upstream PRs, issues, docs, the ROCm and AMD-Ecosystem orgs and the web. The maintainer said in #1085 that GPU acceleration is NVIDIA-only. Since then they have merged a Metal backend (`libs/MVS/PatchMatchMetal.{h,mm,metal}`), so they accept non-CUDA GPU backends. Decision: **new port**. Deliver it as an upstream PR, because upstream merges platform backends rather than linking forks.

Mechanical port or AMD-native rewrite: **mechanical port**. The kernels are plain SIMT code (PatchMatch, confidence, mesh refinement). They use no tensor cores, CUTLASS/CuTe, wgmma or warp specialization. Nothing here needs rocWMMA, CK or MFMA.

## Build classification

**cmake** (recorded: `set-ext-type openMVS cmake`). CMake first-class CUDA language, no torch:

- `CMakeLists.txt:35`: `OPTION(OpenMVS_USE_CUDA ... ON)`. Lines 213-285: `CHECK_LANGUAGE(CUDA)` / `ENABLE_LANGUAGE(CUDA)`, `find_exported_dependency(CUDAToolkit)`, the `CMAKE_CUDA_ARCHITECTURES` parser that `FATAL_ERROR`s on non-numeric archs (248-250), `CMAKE_CUDA_SEPARABLE_COMPILATION ON` + `RESOLVE_DEVICE_SYMBOLS ON`, `--expt-relaxed-constexpr`, links `CUDA::cuda_driver CUDA::cudart CUDA::curand`, sets `_USE_CUDA`.
- `CMakeLists.txt:76-77`: CUDA adds vcpkg manifest feature `cuda`, which pulls the vcpkg `cuda` port plus `ceres[cuda]` and `siftgpu[cuda]` (`vcpkg.json:83-100`).
- `libs/MVS/CMakeLists.txt:25-32`: globs `*.cu` into the MVS library when `_USE_CUDA`.
- `libs/Common/CMakeLists.txt:33-35`: passes `_CUDA_ARCHS_SASS`, `_CUDA_ARCHS_SASS_EXACT` and `_CUDA_ARCH_PTX` to `UtilCUDA.cpp` for the runtime device check.
- Dependencies come from vcpkg (manifest mode, overlay ports in `ports/`, `VCPKG_ROOT` auto-loads the toolchain at `CMakeLists.txt:52-58`). Upstream CI builds Windows, Ubuntu and macOS with `-DOpenMVS_USE_CUDA=OFF`, so upstream never builds the GPU path.

## Port strategy

**Strategy A (compat header + CMake HIP language)**, keeping the CUDA spelling in the sources.

- New option `OpenMVS_USE_HIP` (default OFF, mutually exclusive with CUDA; when both are ON, CUDA wins and a message says so). When ON: `ENABLE_LANGUAGE(HIP)`, `find_package(hip)`, `find_package(hiprand)` / `rocrand`, `set_source_files_properties(<the three .cu> PROPERTIES LANGUAGE HIP)`, link `hip::host` (host TUs) and `hip::hiprand`, define `_USE_CUDA` **and** `_USE_HIP`. `_USE_CUDA` keeps meaning "the GPU path is compiled", so the 40-odd `#ifdef _USE_CUDA` sites in apps, tests and libs need no edits. `_USE_HIP` exists only for the few places that must differ (see SiftGPU and Ceres below).
- **Do not request the vcpkg `cuda` feature in a HIP build.** It needs an NVIDIA toolkit, and `ceres[cuda]`/`siftgpu[cuda]` cannot build against ROCm.
- One compat header, `libs/Common/CudaToHip.h` (name to taste), included by `UtilCUDADevice.h`, `UtilCUDA.h` and `CUDA/Maths.h` in place of `<cuda.h>`, `<cuda_runtime.h>`, `<curand_kernel.h>`, `<math_constants.h>` and `<vector_types.h>` when `_USE_HIP`. It maps the runtime API, the driver-API subset (below), `curand*` to the `hiprand*`/rocRAND device API, and the warp intrinsics (with 64-bit masks; see the risk list). On NVIDIA it includes the original CUDA headers and nothing else.
- No `-fgpu-rdc`. No `.cu` references a `__device__` or `__constant__` symbol from another TU (`grep 'extern __device__|extern __constant__'` is empty, `__constant__` lives in `PatchMatchCUDA.cu` alone). So separable compilation is not needed for HIP, and leaving it off avoids the Windows `-fgpu-rdc` link-wrapper class. The CUDA path keeps its RDC settings untouched.
- Architectures: honour `CMAKE_HIP_ARCHITECTURES`. If unset, use CMake's default (detected GPUs) and print the list, as the CUDA branch does with `native`. The numeric CUDA-arch parser stays CUDA-only.

## CUDA surface inventory

Generated floor: `surface.json` (15 files; 49 kernel markers, 157 device-code, 55 launches, 22 warp-intrinsic, 104 texture, 2 driver-API markers, curand 15). Per file:

| file | lines | content | HIP mapping / risk |
|---|---|---|---|
| `libs/MVS/PatchMatchCUDA.cu` | 770 | 4 kernels (InitializeScore, Black/RedPixelProcess checkerboard, FilterPlanes); `tex2D<float>` on image and depth textures; `curandState` + `curand_init`/`curand_uniform`; `__constant__` cameras and params uploaded with `cudaMemcpyToSymbolAsync`; `__ldg`; `__launch_bounds__(256,2)` / `(512,1)`; device Eigen incl. 3x3 `.inverse()` | textures: risk T1/T2; curand -> hiprand device API (risk C1); `hipMemcpyToSymbolAsync` with `HIP_SYMBOL`; `__ldg` is fine on HIP; launch bounds: risk L1; Eigen 3x3 inverse uses the closed-form cofactor path, expected device-safe (risk E1) |
| `libs/MVS/PatchMatchCUDA.cpp` | 554 | host: streams, a global `cudaEvent_t` chaining workers' constant-memory writes, `cudaMallocArray` + `cudaCreateTextureObject` (LINEAR filter, WRAP address, unnormalized coords), pinned staging (`cudaHostAlloc`), `cudaMemcpy2DToArrayAsync` from pageable and pinned sources, `cudaMallocAsync`/`cudaFreeAsync` | all have HIP equivalents; textures T1/T2; pageable 2D async source lifetime: risk M1 |
| `libs/MVS/ConfidenceCUDA.cu` | 351 | 2 kernels; optional `tex2D` on PatchMatch's resident depth texture; own stream; `dim3 block(32,8)` | textures T1; block geometry only, not a warp assumption |
| `libs/MVS/SceneRefineCUDA.cu` | 1179 | 16 kernels (mesh rasterizer with 64-bit `atomicMin` keys, owner-bit ballot, block prefix scan, tile sort, photometric accumulation, gradient); `__ballot_sync(0xffffffffu, ...)` (:199), `__shfl_up_sync(0xffffffffu, ...)` (:273, :282), `tex2D` LINEAR/CLAMP, `surf2Dread`/`surf2Dwrite` on 32F surfaces, `atomicCAS` on 64-bit, `__popc`, `__ffs`, 1024-thread block with dynamic shared memory | wavefront: risk W1/W2 (the main code fault in this port); surfaces T1/T3; 64-bit atomics fine on HIP |
| `libs/MVS/SceneRefineCUDA.cpp` | 1186 | host: `ArrayRT32F` arrays with `CUDA_ARRAY3D_SURFACE_LDST`, texture + surface objects over driver-API arrays (`(cudaArray_t)(CUarray)` casts), `cuMemsetD32/D16`(+Async), `cuMemAllocHost`, `cuCtxSynchronize`, stream capture into a graph (`cudaStreamBeginCapture(..ThreadLocal)`, `cudaGraphInstantiate/Launch`) | HIP: `hipArray_t` serves as both runtime and driver array type, so the cast holds; `hipMemsetD32/D16(Async)`, `hipMemAllocHost`, `hipCtxSynchronize` (deprecated: map to `hipDeviceSynchronize`); graph capture supported, and no library call is captured (risk G1) |
| `libs/Common/UtilCUDA.{h,cpp}` | 475 | driver-API memory layer (`cuMemAlloc/Free`, `cuMemcpyHtoD/DtoH`, `cuMemcpy2D` host->array, `cuArray3DCreate/Destroy`, `CUarray_format`, `cuGetErrorName/String`) and device selection by compute capability | `hipMalloc`/`hipFree`, `hipMemcpyHtoD/DtoH`, `hipMemcpyParam2D`, `hipArray3DCreate`/`hipArrayDestroy`, `HIP_ARRAY3D_DESCRIPTOR`, `hipDrvGetErrorName/String`; the CC check needs a HIP branch (risk D1) |
| `libs/Common/UtilCUDADevice.h` | 51 | `CUDA_CHECK`, `cudaError_t` abort helper | compat header |
| `libs/MVS/CUDA/Maths.h`, `CUDA/Camera.h` | 454 | Eigen types in device code; `#ifdef __CUDACC__` guards (lines 53, 60, 72, 219, 262) | `__CUDACC__` is not defined by HIP clang: extend each guard with `\|\| defined(__HIPCC__)` (risk H1) |
| `libs/Common/Config.h` | -- | `HOST_DEVICE` on `__CUDACC__` (:217); Debug device `ASSERT` on `__CUDA_ARCH__` (:276) | extend both guards for HIP (`__HIPCC__`, `__HIP_DEVICE_COMPILE__`); risk H1 |
| `libs/MVS/SceneDensify.cpp`, apps, `apps/Tests` | -- | `#ifdef _USE_CUDA` host branches, `initDevices`, `isEnabled` | unchanged (they key on `_USE_CUDA`) |
| `libs/SFM/FeaturesExtractor.cpp:292,333`, `PairsMatcher.cpp:1641`, `GlobalPositioning.cpp:405` | -- | SiftGPU `-cuda` backend selection and the SiftGPU CUDA matcher keyed on `_USE_CUDA`; Ceres `ceres::CUDA` dense/sparse solver keyed on `_USE_CUDA` | must key on the dependency's real capability in a HIP build (risk S1); see Open question 2 |
| `ports/siftgpu/source/src/ProgramCU.cu`, `CuTexImage.{h,cpp}`, `SiftMatchCU.cpp` | ~2.3k | vendored UNC SiftGPU CUDA backend (vcpkg overlay port; licence non-commercial, see intake) | Open question 2 |
| `libs/SFM/ImportROMA2.cpp` | -- | scan false positive (no CUDA use; a matched token) | nothing |

Not present: NVRTC, PTX/module loading (`cuModule*`), cuBLAS/cuFFT/cuSPARSE/cuSOLVER/cuDNN, Thrust/CUB, NCCL, cooperative groups, managed memory, `__CUDA_ARCH__` numeric gates in kernels, CUDA-GL interop (the Viewer uses no CUDA beyond the `--gpu-device` string).

## Risk list

- **T1, textures and surfaces on gfx942/gfx950 (blocking for that arch).** ROCm defines `__HIP_NO_IMAGE_SUPPORT` for gfx94x+, so `tex2D<float>`, `surf2Dread`/`surf2Dwrite` are compile errors when built for those targets. At runtime `hipMallocArray`/`hipArray3DCreate`/`hipCreateTextureObject` return `hipErrorNotSupported` (measured by the colmap port on MI300X). openMVS relies on hardware LINEAR filtering with WRAP (PatchMatch) and CLAMP (SceneRefine) address modes and on surface load/store, so this is not a pointer swap. **Open question 1.** gfx90a (CDNA2, wave64), gfx11xx and gfx12xx support all of it.
- **T2, hardware bilinear filtering precision.** AMD and NVIDIA samplers both use low-precision (about 8-bit) interpolation weights, but the rounding differs, so GPU results will not match CUDA bit for bit. The tests compare against the CPU or ground truth with wide windows (see Test plan), which is the right bar; do not tighten them.
- **T3, texture pitch.** Arrays are allocated through `cudaMallocArray`/`cuArray3DCreate` (the driver picks the pitch) and copied with `Memcpy2D` using the host row stride, so the 256-byte pitch rule does not apply to any bind here. Check that no pitched linear memory is bound to a texture (none found).
- **W1, owner-bit ballot (`SceneRefineCUDA.cu:197-201`).** The code writes one 32-bit word per 32 faces from lane `tid & 31 == 0`. On wave64 `__ballot` returns 64 bits covering two 32-face groups. Fix: take the half for this thread's group, `(uint32_t)(ballot >> (threadIdx.x & 32))`, which is valid because the launch is 1D with 256 threads (a multiple of 64; `SceneRefineCUDA.cu:1029`). The literal `0xffffffffu` mask must become a 64-bit full mask under HIP, or the HIP `*_sync` static assert fails to compile. Correct on both widths: on wave32 the shift is 0.
- **W2, block prefix scan (`blockScanExclusive`, :264-292).** It hard-codes 32-lane warps (`lane = threadIdx.x & 31`, `warpSums[32]`, `numWarps = blockDim.x >> 5`). That stays correct on wave64 if every `__shfl_up_sync` uses logical width 32 (pass `32` explicitly; HIP honours the width argument), so lanes never read across a 32-lane group, and the masks are 64-bit. The `warp == 0` branch then runs in half a wave64, which is fine: the shuffles read only lanes 0-31. Do not switch this code to `warpSize`: keeping 32-lane logical groups preserves the shared-memory layout and the CUDA path byte for byte. Sweep the rest of the three `.cu` files for `& 31`, `>> 5`, `31`, `32` used as a lane width (the scan above found only these). `MAX_VIEWS 32`, `NUM_SAMPLES 32`, `BLOCK_W 32`, `dim3(32,8)` and the 32-direction bitmask in `SceneRefineCUDA.cpp:781-814` are data sizes, not warp widths.
- **C1, curand.** `curandState` (XORWOW) + `curand_init(seed, subsequence, offset)` + `curand_uniform`. Map them to `hiprandState`/`hiprand_init`/`hiprand_uniform` (rocRAND XORWOW). The sequences differ from cuRAND's, which is fine: PatchMatch random init and perturbation are stochastic and the tests do not compare against a CUDA golden. The state is allocated per pixel (`sizeof(curandState) * size`); use `sizeof` of the mapped type, never a hard-coded size. rocRAND device headers need C++14 or newer from ROCm 10.1 (the project is C++17).
- **L1, `__launch_bounds__(256, 2)`.** HIP accepts the second argument (min blocks per multiprocessor) and converts it to waves per EU. Keep it. On AMD the second form `(512,1)` is selected only by a build toggle. No register-count heuristics on the host.
- **M1, pageable 2D async upload (`PatchMatchCUDA.cpp:136-139`).** For images under 1.5 MP, `cudaMemcpy2DToArrayAsync` reads straight from the caller's `cv::Mat`. On HIP a pageable 2D/3D async source must stay alive and unchanged until the stream syncs. The callers at `PatchMatchCUDA.cpp:366` and `:372` disagree. The image upload (`:366`) is safe, because `image` is moved into the member `images[i]` (`:375`) and a `cv::Mat` move keeps its buffer. **The depth-prior upload (`:369-372`) is a real use-after-free on HIP.** `DepthMap depthMap(view.depthMap)` is a loop-body local. When `ResizeUniform` rescales it, it owns a fresh buffer that is freed at the end of the iteration, long before the `cudaStreamSynchronize` at `:440`. (Without the resize it shares `view.depthMap`'s refcounted buffer and is safe.) This is geometric-consistency mode with neighbour views at a different scale and an area under 1.5 MP, which is the synthetic test scale. Fix (shared code, correct on CUDA too, where the same lifetime is undocumented): keep the resized prior alive until the sync, e.g. a per-slot member `cv::Mat` holding it the way `images[i]` holds the image, or always route a resized prior through the pinned staging path. Do not add a sync after the 1D uploads. A validator can confirm any residual copy-lifetime suspicion with `AMD_SERIALIZE_COPY=3`.
- **M2, fresh allocations are not zero on ROCm.** Before each `atomicMin`/`atomicAdd`/`atomicCAS` target (`projKey`, `hist`, `vertexStamp`, `photoCount`), check that a memset or init kernel precedes its first use on the same stream. The code issues `cuMemsetD32` in eight places, which looks deliberate. The porter maps each target to its memset and records the result.
- **D1, device selection (`UtilCUDA.cpp:73-108`).** `_isComputeCapabilitySupported` compares `props.major/minor` against CUDA SASS/PTX lists. AMD compute-capability values collide with NVIDIA's (gfx942 reports 9.4, gfx90a 9.0), so the HIP build must instead compare `props.gcnArchName` (base name before `:`) against the compiled `CMAKE_HIP_ARCHITECTURES`, passed as a string define from `libs/Common/CMakeLists.txt`. The log message suggests `-DCMAKE_HIP_ARCHITECTURES=<gfx>`. The `-1` "best device" ranking by `major/minor` then perf is meaningless across AMD archs, so rank by perf only under HIP. `cudaDevAttrComputeMode` maps to `hipDeviceAttributeComputeMode`.
- **H1, macros HIP does not define.** `__CUDACC__` (Maths.h, Config.h) and `__CUDA_ARCH__` (Config.h Debug assert). Extend each guard, never re-key it. Without this, Maths.h declares host `min`/`max` in the HIP device pass and Config.h's Debug `ASSERT` calls host functions from device code.
- **G1, graph capture (`SceneRefineCUDA.cpp:837-850`).** `hipStreamBeginCapture` with `hipStreamCaptureModeThreadLocal` and graph instantiate/launch are supported. The captured region contains only kernel launches and memsets. Confirm that no allocation (`hipMallocAsync`) or blocking call is captured, and keep the existing fallback when capture fails.
- **E1, Eigen in device code.** Only fixed-size arithmetic and a 3x3 `inverse()`, which Eigen implements in closed form, so it should compile under hipcc. No decompositions are called from device code (grep for `ldlt|llt|Svd|EigenSolver|solve(` in kernels is empty). If hipcc rejects it, the fix is an explicit 3x3 adjugate inverse in `CUDA/Camera.h`, guarded so CUDA keeps Eigen.
- **S1, SiftGPU and Ceres keyed on `_USE_CUDA`.** In a HIP build `_USE_CUDA` is defined but SiftGPU and Ceres have no CUDA backend. Without a change the SFM code would pass `-cuda` to SiftGPU and take the SiftGPU CUDA matcher branch (`PairsMatcher.cpp:1641`), which then fails at runtime. Guard these three sites with `!defined(_USE_HIP)` (or key them on a define the SiftGPU package exports). Ceres already checks `IsDenseLinearAlgebraLibraryTypeAvailable(ceres::CUDA)` at runtime and falls back, but gate it the same way so the HIP build does not print a misleading warning.
- **B1, IPO.** `OpenMVS_ENABLE_IPO` (default ON) sets `INTERPROCEDURAL_OPTIMIZATION`. If CMake passes `-flto` to the HIP compile of the three `.cu` files and the device link breaks, turn IPO off for those sources only (`set_source_files_properties` or a separate object library), not project-wide.
- **B2, vcpkg on the build host.** The documented build fetches about 40 ports (OpenCV, CGAL, Boost, ...) through vcpkg and needs `autoconf-archive`. The first build is long (an hour or more). Use a job-local `VCPKG_ROOT` and binary cache, never a shared image path. A configure with `-DOpenMVS_USE_SIFTGPU=OFF -DOpenMVS_USE_PYTHON=OFF -DOpenMVS_BUILD_VIEWER=OFF` cuts the dependency set for iteration. The final validation build uses the defaults (see Test plan), so the Python module and Viewer are built at least once.
- **B3, Windows.** clang-cl/HIP through TheRock, all-clang toolchain. With no `-fgpu-rdc`, `hip_link_win.py` is not needed. Expect the usual CRT/DLL staging issues (`windows.md`). The Python package wrapper `build/python/__init__.py` (`_resolve_cuda_bin`, from `CUDA_PATH` or the NVIDIA install dir) adds the CUDA `bin/x64` directory to the DLL search path on Windows. A ROCm build needs the ROCm/TheRock `bin` directory instead (e.g. from `HIP_PATH`/`ROCM_PATH`). Keep the CUDA lookup unchanged.

## File-by-file change list

Core work (needed whatever the rulings):

1. `CMakeLists.txt`: `OPTION(OpenMVS_USE_HIP "Enable HIP (ROCm) library" OFF)`. A HIP branch next to the CUDA one (213-285): `ENABLE_LANGUAGE(HIP)`, `find_exported_dependency(hip)` and `hiprand`, `SET(_USE_CUDA TRUE)` + `ADD_DEFINITIONS(-D_USE_HIP)`, the arch list string for `UtilCUDA.cpp`, no RDC, `OpenMVS_EXTRA_LIBS` += `hip::host hip::hiprand`. Do not append the `cuda` vcpkg feature. macOS keeps both off.
2. `libs/MVS/CMakeLists.txt`: under `_USE_HIP`, mark `${LIBRARY_FILES_CUDA}` `LANGUAGE HIP`; install the compat header with the other headers. `libs/Common/CMakeLists.txt`: under `_USE_HIP`, pass the HIP arch list instead of the SASS/PTX defines. Also update the exported-config template so a downstream `find_package(OpenMVS)` re-finds `hip`/`hiprand` (`build/Templates`, where `find_exported_dependency` records them; check that it does so automatically).
3. New `libs/Common/CudaToHip.h` (compat header): the runtime, driver and curand aliases used above, plus `__ballot_sync`/`__shfl_up_sync` wrappers that take a 64-bit mask and pass logical width 32 explicitly where the call site asks for it. Use real function overloads or inline functions, not macros named after ROCm's own warp functions (`build-headers-macros.md`). It must be includable from plain host TUs.
4. `libs/Common/UtilCUDADevice.h`, `UtilCUDA.h`, `libs/MVS/CUDA/Maths.h`: include the compat header under `_USE_HIP` instead of the CUDA headers. Maths.h and `Config.h`: the `__CUDACC__`/`__CUDA_ARCH__` guard extensions (H1).
5. `libs/Common/UtilCUDA.cpp`: the HIP device check by `gcnArchName` and perf-only ranking (D1); `cuGetErrorName/String` -> `hipDrvGetErrorName/String`.
6. `libs/MVS/SceneRefineCUDA.cu`: W1 and W2 (ballot half-word extraction, 64-bit masks, explicit width 32 on both shuffles). `PatchMatchCUDA.cu`, `ConfidenceCUDA.cu`: no source change expected beyond what the compat header supplies. The porter confirms by compiling for gfx90a and gfx1100 (or the host arch) and recording any diagnostics.
7. `libs/SFM/FeaturesExtractor.cpp`, `PairsMatcher.cpp`, `GlobalPositioning.cpp`: the S1 guards (if Open question 2 is answered "leave SiftGPU on GLSL"; if answered "port it", only the Ceres guard).
8. `docs/wiki/Building.md` (and the `--gpu-device` help strings, which say "GPU" already): a short "ROCm (AMD GPUs)" paragraph with the configure line, `CMAKE_HIP_ARCHITECTURES`, and which features are GPU-accelerated on AMD. `README.md` dependency list: "CUDA or ROCm (optional)".
9. `build/python/__init__.py` (B3): the ROCm DLL directory on Windows.
9b. `libs/MVS/PatchMatchCUDA.cpp:369-372`: keep the resized depth prior alive until the stream sync (M1).

Ruling-dependent:

10. **If Open question 1 is ruled "support gfx94x+":** a buffer-backed texture/surface path for image-less GPUs, dual-path as described in `faults-textures-interop.md` ("A project that reads EVERYTHING through texture objects"). In the device pass under `#if __HIP_NO_IMAGE_SUPPORT`, `tex2D<float>` becomes a software bilinear fetch over a linear buffer that honours WRAP/CLAMP and the +0.5 texel-centre convention, and `surf2Dread/write` become indexed loads/stores. On the host, `ArrayRT`/`cudaMallocArray` allocate linear buffers when `hipDeviceAttributeImageSupport` is 0, cross-checked against a `__device__` flag read with `hipMemcpyFromSymbol`. Free/destroy dispatch goes through a registry. The texture-capable arches keep the hardware path byte for byte. Estimated size: comparable to the rest of the port together (four texture families: PatchMatch images/depths, Confidence depth, SceneRefine image/gradients, SceneRefine surfaces).
11. **If ruled "block gfx94x+":** under `_USE_HIP`, `CMakeLists.txt` refuses (`FATAL_ERROR`, with the reason) any `CMAKE_HIP_ARCHITECTURES` entry matching `gfx94*`/`gfx95*`, and `initDevices` skips a device whose `hipDeviceAttributeImageSupport` is 0 with a log line ("no texture support; using the CPU"). The CPU path then runs, as upstream already does for an unsupported GPU. Record `set-blocked openMVS linux-gfx942` (and gfx950) with the reason. wave64 is validated on gfx90a.
12. **If Open question 2 is ruled "port SiftGPU's CUDA backend":** `ports/siftgpu/source` gets a HIP build (its own CMake `CUDA_ENABLED` option gains a HIP twin, and `ports/siftgpu/portfile.cmake` and `vcpkg.json` gain a `hip` feature). Reuse the three SiftGPU faults already diagnosed in the colmap work (CuTexObj rule-of-five double destroy, ComputeDOG OOB read, `tex1Dfetch` linear binding in place of the pitched 2D bind). SiftGPU's 44 texture objects hit T1 on gfx94x+ as well. Validation adds the SFM test with SiftGPU forced to its CUDA backend.

## Build commands

Linux (any ROCm host; `<gfx>` = the host arch, e.g. gfx90a, gfx1100):

```
export VCPKG_ROOT=<job dir>/vcpkg     # git clone https://github.com/microsoft/vcpkg && ./bootstrap-vcpkg.sh
sudo apt install autoconf-archive     # plus the Building.md apt list if missing
cmake -S projects/openMVS/src -B build-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenMVS_USE_CUDA=OFF -DOpenMVS_USE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=<gfx> \
  -DOpenMVS_HEADLESS_DEBUG=OFF
cmake --build build-hip -j"$MAX_JOBS"
```

Use `utils/timeit.sh openMVS compile -- ...` around both. For iteration, add `-DOpenMVS_USE_SIFTGPU=OFF -DOpenMVS_USE_PYTHON=OFF -DOpenMVS_BUILD_VIEWER=OFF`. For the validated build, use the defaults: a headless host may need `-DOpenMVS_BUILD_VIEWER=OFF` if GLFW/GL dev packages are unavailable, and must say so in notes. Also build once with `-DCMAKE_BUILD_TYPE=Debug` (or RelWithDebInfo plus `_DEBUG`) to compile the Debug device-assert path (H1). It need not be run in full.

Windows: the same configure from a TheRock environment with the all-clang toolchain, `-DVCPKG_TARGET_TRIPLET=x64-windows`. See `windows.md`.

CUDA no-regression: configure with `-DOpenMVS_USE_CUDA=ON` (default) and `-DCMAKE_CUDA_ARCHITECTURES=80;90` on a host with nvcc (the mi355 fleet host's toolkit, or the conda `cuda-12.8` env), and build at least the MVS and Common libraries plus `Tests`. Record the command and result.

## Test plan

The project's own suite, registered in CTest (`CMakeLists.txt:434-438`), all driven by one `Tests` binary:

| ctest name | command | GPU? |
|---|---|---|
| `CommonUnitTests` | `Tests 0` | no (data structures, math, PLY round-trip) |
| `SFMPipelineTest` | `Tests 1` | only via SiftGPU (GLSL/EGL, or HIP if Open question 2 says so) and Ceres; the core port does not touch it, so it is the non-GPU regression set |
| `MVSPipelineTest` | `Tests 2` | **yes**: `MVS::PipelineTest` runs DensifyPointCloud (GPU PatchMatch + Confidence when a device is enabled), ReconstructMesh, RefineMesh and TextureMesh on `apps/Tests/data/scene.mvs` and checks plausibility windows. `MVS::MeshRefineSyntheticTest` runs CPU and GPU `RefineMeshCUDA` on the same synthetic fixture and requires GPU RMS within +-10% of CPU RMS and under 0.5 px footprint. |

Validator commands:

```
cd build-hip
ctest --output-on-failure                      # all three
./bin/Tests 2 2                                # MVS suite, verbose: writes Tests-*.log
./bin/Tests 2 2 1                              # same, forced CPU: the control run
```

Evidence that the GPU path actually ran (a green CTest does not show it, because the code falls back to the CPU silently when no device initializes):

- the verbose log must contain the `CUDA device <n> initialized: <AMD device name>` line from `initDevices` and the `MeshRefineSyntheticTest CUDA: RMS -> ... (CPU ratio ...)` line;
- and kernel dispatches observed, e.g. `rocprofv3 --kernel-trace -- ./bin/Tests 2 2` listing `BlackPixelProcess`, `RedPixelProcess`, `kernelProjectMesh` and the confidence kernels.

Record both RMS values and the ratio, and the PipelineTest point/face counts for the GPU and forced-CPU runs.

Real-data check (beyond the suite): run `DensifyPointCloud` on a small public dataset, e.g. the openMVS sample scene from `openMVS_sample` (https://github.com/cdcseacave/openMVS_sample), GPU vs `--gpu-device -2`. Compare point counts and timings, and record both. It is not a gate, but it is the user-visible feature.

Coverage: wave64 (gfx90a; gfx942 only if Open question 1 says support), wave32 (gfx1100/gfx1201), Windows (gfx1151 laptop, last, after both Linux gates pass at the same head). W1/W2 are the code faults that differ by width, so both widths must run `MVSPipelineTest`, which exercises SceneRefine's ballot and scan through `MeshRefineSyntheticTest`.

Non-GPU regressions that must not move: `CommonUnitTests` and `SFMPipelineTest` pass in the HIP build exactly as in a `-DOpenMVS_USE_CUDA=OFF` build on the same host. Run both builds if time allows, or at least compare against upstream CI's expectation of all-pass.

## Open questions

1. **gfx942/gfx950 (MI300/MI355): support or block?** (Jeff rules.) Every openMVS kernel samples images through hardware-filtered texture objects (LINEAR + WRAP/CLAMP) and SceneRefine writes surfaces. These GPUs have no HIP image API, at compile time or at run time.
   - (a) Block them in this port: refuse those archs at configure time, fall back to the CPU at runtime, and record the gfx942/gfx950 platforms blocked. Validate wave64 on gfx90a, as colmap did. This is the smallest upstream diff. MI300/MI355 users get the CPU path. A buffer-backed follow-up can be registered as deferred work if you want it tracked.
   - (b) Support them in this port: a dual-path buffer-backed texture/surface emulation under `__HIP_NO_IMAGE_SUPPORT` (change-list item 10). This roughly doubles the port and the upstream diff, and adds a software-bilinear code path the maintainer must carry and that only AMD Instinct exercises. In exchange, openMVS GPU densify and mesh refinement run on MI300/MI355.
   - (c) Buffer-only for all AMD GPUs: one path, no hardware textures under HIP. Simpler than (b), but it is slower on RDNA/CDNA2, where textures work (LEAP measured 2-3x on bilinear-heavy kernels), and results there drift further from the CUDA build.
   - Planner's lean: (a) now, with (b) as a separately ruled follow-up. Smallest complete port first, gfx90a satisfies the wave64 gate, and the colmap precedent is the same call. The lean is not a ruling.
2. **SiftGPU's CUDA backend: port it or leave it on GLSL?** (Jeff rules.) It lives in this repository (`ports/siftgpu/source`, a vcpkg overlay; `ProgramCU.cu` is 69 KB). The SFM library uses it for GPU feature extraction and matching when `_USE_CUDA`. Its licence is UNC's non-commercial research licence (mislabelled BSD-3 in `ports/siftgpu/vcpkg.json`), the same component as colmap's.
   - (a) Leave it: a ROCm build uses SiftGPU's GLSL/EGL backend (or the CPU SIFT), and the SFM code stops selecting the CUDA backend under HIP (risk S1). This is a partial port in the sense of the standing rule: the SFM layer's GPU SIFT backend stays NVIDIA-only.
   - (b) Port it in this port: change-list item 12, reusing the colmap SiftGPU fixes. It hits T1 again on gfx94x+. It adds a second large vendored-code diff upstream, and the licence question for AMD shipping or depending on it.
   - Planner's lean: (a). The openMVS GPU value is the MVS kernels, and SiftGPU is optional, vendored, non-commercially licensed and already being worked on the colmap side. The lean is not a ruling.
3. Not a question, recorded for completeness: Ceres' CUDA solvers (`ceres[cuda]`) are an external dependency with no HIP backend, so a ROCm build uses CPU Ceres. Nothing in this repository could change that.
