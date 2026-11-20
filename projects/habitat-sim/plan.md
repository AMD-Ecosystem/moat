# habitat-sim port plan

Planner run 2026-11-20 on linux-gfx1100 (fleet job, no GPU used, nothing built). Analysis is at upstream `main` 57ee4941dc47 (2026-05-07), which is also the fork's `main` tip.

**Rulings needed from Jeff before a porter starts: see "Open questions" (Q1 to Q4) at the end. They are repeated at the top of the planner section of notes.md.**

## Project

- Name: habitat-sim
- Upstream: https://github.com/facebookresearch/habitat-sim (archived on GitHub; README carries Meta's sunset notice)
- Fork: https://github.com/AMD-Ecosystem/habitat-sim, default branch `main` (pristine mirror), port branch `moat-port`
- Licence: MIT (intake), with two NVIDIA-EULA CUDA-samples headers vendored (`src/esp/gfx_batch/cuda_helpers/HelperCuda.h`, `HelperString.h`); see Q1.
- Delivery vehicle: no upstream PR is possible (archived repository refuses PRs). The deliverable is `moat-port` on the fork, which is the `set-no-upstream` case in AGENTS.md; recording that ruling is Jeff's (Q1b).

## Existing AMD support

None. Intake found no upstream, fork, PR or branch effort (`amd` and `AMD-AGI` orgs unchecked: SAML). Upstream's last stance (issue #2402, 2024, maintainer aclegg3): no motivation to support AMD, others welcome to. No authoritative or community AMD port to inherit. Decision: port.

Performance-kernel question (mechanical vs AMD-native): not applicable. The only kernel is a per-pixel depth-noise kernel (a few hundred FLOPs per pixel, cuRAND normals); a mechanical HIP port is the right and final form.

## Build classification

Pure CMake (Strategy A), packaged through scikit-build-core:

- `pyproject.toml` `[tool.scikit-build.cmake.define]` maps env vars to CMake options (`BUILD_WITH_CUDA = {env = "HABITAT_WITH_CUDA"}`, `BUILD_TEST = {env = "HABITAT_BUILD_TEST"}`, ...). `build.sh` and `setup.py` are thin wrappers around `pip install -e . --no-build-isolation`.
- `src/CMakeLists.txt:15` `option(BUILD_WITH_CUDA ... OFF)`; `:25-30` `project(esp LANGUAGES C CXX CUDA)` plus `find_library(CUDART_LIBRARY cudart ...)` when on.
- `src/esp/CMakeLists.txt:37-38` sets `ESP_BUILD_WITH_CUDA` (configure.h); `:477-481` adds `sensor/RedwoodNoiseModel.{cpp,h}` and the kernel `.cu/.cuh`; `:537-552` builds `noise_model_kernels` STATIC and links cudart into `habitat_sim`.
- `src/esp/gfx_batch/CMakeLists.txt:38-45` adds the CUDA include dirs and `cuda_helpers/` PUBLIC, links cudart. `src/tests/CMakeLists.txt:127-131` adds CUDA include dirs to `GfxBatchRendererTest`.
- No `find_package(Torch)`, no `CUDAExtension`. PyTorch is only a Python runtime consumer (tensor data pointers passed into the bindings). Recorded: `set-ext-type habitat-sim cmake`.

CUDA is entirely optional upstream; the default build has none.

## Port strategy

Strategy A with shim headers, keeping CUDA-spelled sources byte-identical where possible:

1. New CMake option `BUILD_WITH_HIP` (env `HABITAT_WITH_HIP` in `pyproject.toml`, `--with-hip` in `build.sh`), mutually exclusive with `BUILD_WITH_CUDA` (FATAL_ERROR if both). It does `enable_language(HIP)` (with `CMAKE_HIP_STANDARD 17`; `CMAKE_CXX_STANDARD` does not reach the HIP language and ROCm 10.1's rocRAND device headers need C++14), `find_package(hip REQUIRED)`, `find_package(hiprand REQUIRED)`, marks `sensor/RedwoodNoiseModel.cu` `LANGUAGE HIP`, links `hip::host` (and `hip::hiprand` for the kernel library).
2. Under `BUILD_WITH_HIP`, CMake also sets `ESP_BUILD_WITH_CUDA` (the existing feature macro meaning "GPU compute and GL interop compiled in"), so every `#ifdef ESP_BUILD_WITH_CUDA` block, the `cuda_enabled` Python attribute, the GPU noise model, `gpu2gpu_transfer` and the test gates all light up without touching their sources. It additionally defines `ESP_BUILD_WITH_HIP` for the few places that must differ. Rationale: smallest diff, and PyTorch on ROCm also spells its device `cuda`, so `cuda_enabled` stays truthful from the Python user's point of view. Add a `hip_enabled` attribute next to `cuda_enabled` in `src/esp/bindings/Bindings.cpp` so users can tell the backends apart.
3. A HIP-only shim include directory (proposed `src/esp/hip_compat/`, on the include path only when `BUILD_WITH_HIP`) with project-owned shadow headers `cuda_runtime.h`, `cuda_gl_interop.h`, `curand_kernel.h` that include the HIP headers and `#define` the CUDA names used here to their HIP names (list in the inventory below), plus a project-owned `HelperCuda.h` providing only `checkCudaErrors(...)` written from scratch (print `hipGetErrorName`/`hipGetErrorString`, file, line; abort). Under HIP, `cuda_helpers/` is NOT added to the include path, so the NVIDIA-EULA files are neither compiled nor derived from.
4. The real AMD blocker is not CUDA (see "EGL device selection" below); it gets a small project-owned helper used by both EGL context creation sites, and applies to every build, HIP or not.

The CUDA build must stay a pure passthrough: no edits inside CUDA-only code paths, the NVIDIA EGL selection behaviour unchanged when the NVIDIA EGL driver is present.

## CUDA surface inventory

`utils/surface.py` census: 6 files. Its library counts for cuBLAS/cuFFT/cuSPARSE/cuSOLVER (1 each) and most of the "warp_intrinsic"/"texture" markers come from `HelperCuda.h`'s error-name tables (NVIDIA-EULA file, not ported, not compiled under HIP). Real surface:

| site | CUDA use | HIP mapping | risk |
|---|---|---|---|
| `src/esp/sensor/RedwoodNoiseModel.cu` | 2 `__global__` (`redwoodNoiseModelKernel`, `curandStatesSetupKernel`), 1 `__device__`, `<<<>>>` x2, `curandState_t` (XORWOW), `curand_init(seed, subseq, 0, &s)`, `curand_normal`, `cudaMalloc/Free/Memcpy` | `hiprandState_t` (XORWOW in hipRAND), `hiprand_init`, `hiprand_normal`; HIP runtime 1:1 | low. Random stream differs from cuRAND, which the test tolerates (it compares a 20-run GPU mean against the CPU model with tolerance 1e-2; at noise 0 with 1e-5). `warpSize` is queried at runtime on the host (`RedwoodNoiseModel.cpp:51-55`) and only used to round the block size: wave64-safe. Every state is initialised by the setup kernel before use; output fully written. No unchecked-zero hazard. |
| `src/esp/sensor/RedwoodNoiseModel.cpp` | `cudaGetDevice/SetDevice`, `cudaDeviceGetAttribute(cudaDevAttrMaxThreadsPerBlock / cudaDevAttrWarpSize)`, `cudaMalloc/Memcpy/Free` | 1:1 (`hipDeviceAttributeMaxThreadsPerBlock`, `hipDeviceAttributeWarpSize`). Code spells `cudaDeviceAttr::cudaDevAttrWarpSize` (scoped enum spelling): the shim must make `cudaDeviceAttr` name `hipDeviceAttribute_t` so the qualified names resolve. | low |
| `src/esp/gfx/RenderTarget.cpp:277-365` | `cudaGraphicsGLRegisterImage(GL_RENDERBUFFER, ReadOnly)`, `cudaGraphicsMapResources`, `cudaGraphicsSubResourceGetMappedArray`, `cudaMemcpy2DFromArray(D2D)`, `cudaGraphicsUnmapResources`, `cudaGraphicsUnregisterResource`; `cudaArray`, `cudaGraphicsResource_t` | `hipGraphicsGLRegisterImage`, `hipGraphicsMapResources`, `hipGraphicsSubResourceGetMappedArray`, `hipMemcpy2DFromArray`, ... all declared in ROCm 10.1 (`hip/hip_gl_interop.h`, `hip_runtime_api.h:6063,10220`). Map/copy/unmap is scoped per call: no "pointer dies at unmap" hazard. | **HIGH, feasibility.** Registering a GL *renderbuffer* (not a texture) and copying out of its mapped array is the least-trodden HIP-GL path. Porter probes it first. |
| `src/esp/gfx_batch/RendererStandalone.cpp:31-115,204-262` | `cudaGraphicsGLRegisterBuffer` on a GL pixel-pack buffer, `cudaGraphicsResourceGetMappedPointer`, map kept until the next call or destruction | 1:1 HIP | medium. The returned pointer is valid only while mapped; upstream unmaps exactly at the next read or in `~State()`, so the contract holds. |
| `src/esp/sim/BatchReplayRenderer.cpp:182-210`, bindings (`GfxBindings.cpp:160`, `SensorBindings.cpp:24,292`, `SimBindings.cpp:531-541`, `Bindings.cpp:18`) | no CUDA calls, only `ESP_BUILD_WITH_CUDA` gates | unchanged | none |
| `src/tests/GfxBatchRendererTest.cpp:36-37,2248-2316` | `cuda_gl_interop.h`, `cudaGetDeviceCount`, `cudaMemcpy`, `.setCudaDevice(0)` | via shim; `setCudaDevice(0)` goes through the new device selection | low |
| Python `src_python/habitat_sim/sensors/sensor_wrapper.py:98-101,202`, `noise_models/redwood_depth_noise_model.py` | `torch.device("cuda", id)`, `tensor.data_ptr()` handed to C++ | unchanged; ROCm PyTorch uses the same spelling and HIP pointers | low |

What the census cannot see, added to surface.json:

- **EGL device selection through `EGL_NV_device_cuda`** (`src/esp/gfx/WindowlessContext.cpp:41-45` calls `config.setCudaDevice(device)` whenever `gpu_device_id != -1`, default 0; `src/esp/gfx_batch/RendererStandalone.cpp:81-82` passes its `cudaDevice` through). Magnum (pinned 70b0d76f) resolves this only through `eglQueryDeviceAttribEXT(EGL_CUDA_DEVICE_NV)` (`WindowlessEglApplication.cpp:192-238`), so on Mesa every headless habitat-sim fails with "unable to find CUDA device 0 among N EGL devices", CUDA build or not. This is issue #2402 and is the highest-value part of the port.
- `ESP_BUILD_WITH_CUDA`-gated Python paths (`cuda_enabled`, `gpu2gpu_transfer`, GPU noise model) and their pytest coverage.
- The pip/scikit-build option plumbing (`pyproject.toml`, `build.sh`, `setup.py` help text) and build docs.

## EGL device selection (applies to every build)

New project-owned helper (proposed `src/esp/gfx/EglDeviceSelection.{h,cpp}`, compiled only when `MAGNUM_TARGET_EGL`), called from `WindowlessContext::Impl` and `RendererStandalone::State` in place of the bare `setCudaDevice`:

1. Enumerate EGL devices (`eglQueryDevicesEXT`). If any device exposes `EGL_NV_device_cuda`, keep today's behaviour exactly: `setCudaDevice(id)`. NVIDIA path unchanged.
2. Otherwise map the habitat-sim GPU ordinal to an EGL device index and call Magnum's existing `Configuration::setDevice(index)`:
   - HIP build: `hipDeviceGetPCIBusId(id)` gives the PCI address of the HIP device the same process will compute on; match it against each EGL device's DRM node (`EGL_EXT_device_drm` `EGL_DRM_DEVICE_FILE_EXT`, or `EGL_EXT_device_drm_render_node` `EGL_DRM_RENDER_NODE_FILE_EXT`) resolved to a PCI address through `/sys/dev/char/<major>:<minor>/device`. This keeps GL and HIP on the same physical GPU, which GL interop requires (cuda-to-rocm linux-host-setup.md: mismatched numbering fails the first register call with "invalid argument").
   - Non-HIP build: take the `id`-th hardware EGL device (skip `EGL_MESA_device_software`), ordered by PCI address, and log which device was chosen.
3. If no hardware device matches (or none exists), fail with a message naming what was searched, unless the fallback in Q2 is ruled in.
4. `gpu_device_id = -1` keeps meaning "default EGL display" as upstream.

No Magnum change is needed (Magnum already has `setDevice`). A Magnum-side helper would be an upstream conversation with mosra/magnum and is not part of this port.

## Risk list

1. **HIP-GL interop against a Mesa EGL context (feasibility).** ROCm 10.1's `libamdhip64` references `eglGLInteropExportObjectMESA` / `eglGLInteropQueryDeviceInfoMESA` (checked by `strings` on this host's `_rocm_sdk_core`), i.e. CLR implements GL interop on top of Mesa's `MESA_GL_interop` for EGL as well as GLX, so it is plausible. Unverified: the renderbuffer image registration, `hipMemcpy2DFromArray` from the mapped array, and whether radeonsi exports depth (`GL_DEPTH_COMPONENT32F`) and the R32UI object-id renderbuffer. Porter probes this first, before the rest of the port. If renderbuffer registration fails but buffer registration works, the fallback inside `RenderTarget` is to `glReadPixels` into a GL pixel-pack buffer and register that buffer (the pattern `RendererStandalone` already uses), under `ESP_BUILD_WITH_HIP` only. If no interop works at all, `gpu2gpu_transfer` cannot be offered on HIP; that would be a scope ruling for Jeff, not a porter decision.
2. **CDNA has no graphics pipeline** (cuda-to-rocm faults-textures-interop.md). On gfx90a/gfx942/gfx950 radeonsi refuses a context; only llvmpipe, and a software-GL object cannot be registered with HIP. Rendering (the core of habitat-sim) and GL interop cannot run on hardware there. See Q2.
3. **Device numbering**: `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES` renumber HIP devices but not EGL/DRM devices; the PCI-address match handles it in a HIP build. In a non-HIP build the PCI-order choice can disagree with the user's mental numbering; document it.
4. **Wavefront width**: no warp intrinsics, shared memory or lane masks in the kernel; the only `warpSize` use is a runtime host query for block rounding. Clean on wave32 and wave64.
5. **Rule-of-five on interop handles**: `RenderTarget::Impl` holds `cudaGraphicsResource_t` members and unregisters in its destructor; `RendererStandalone::State` likewise. Both are held by `std::unique_ptr`/`Cr::Pointer` pimpls and not copied; reviewer confirms no copy path.
6. **Unchecked HIP calls** in `RedwoodNoiseModel.{cu,cpp}` (as upstream on CUDA). Not a port defect; leave as is unless a failure needs diagnosis.
7. **Library swap**: cuRAND -> hipRAND only. hipRAND/rocRAND must be found by CMake from the pip ROCm SDK on fleet hosts (`ROCM_PATH`/`CMAKE_PREFIX_PATH`).
8. **Build plumbing traps**: `CMAKE_HIP_STANDARD` (above); `CMAKE_HIP_ARCHITECTURES` must be set for the host gfx; RUNPATH to the ROCm lib dir for the Python extension (cuda-to-rocm build-cmake-link.md, "A HIP-linked shared library ... gets no RUNPATH"); host `.cpp` files must NOT be marked `LANGUAGE HIP` (they use HIP host API via `hip::host` with the host compiler and `__HIP_PLATFORM_AMD__` defined by the hip target).
9. **Upstream CI test flag typo**: `.github/workflows/install_and_test.yml:129` sets `HABITAT_BUILD_TESTS=ON`, but `pyproject.toml` reads `HABITAT_BUILD_TEST`. Upstream CI may never have built the C++ tests; expect C++ test failures unrelated to the port and establish a non-GPU baseline (see Test plan). Do not fix upstream's CI (fork Actions stay disabled).
10. **Audio** (`BUILD_WITH_AUDIO`, rlr-audio-propagation prebuilt binary, licence unchecked by intake): CPU only, off by default, untouched by the port.

## File-by-file change list (porter)

| file | change |
|---|---|
| `src/CMakeLists.txt` | `option(BUILD_WITH_HIP ...)`; mutual exclusion with `BUILD_WITH_CUDA`; `enable_language(HIP)`, `CMAKE_HIP_STANDARD 17`, `find_package(hip)`, `find_package(hiprand)` |
| `src/esp/CMakeLists.txt` | set `ESP_BUILD_WITH_CUDA` and `ESP_BUILD_WITH_HIP` when HIP; add the Redwood sources; `noise_model_kernels` with `.cu` as `LANGUAGE HIP`, link `hip::hiprand hip::host`; shim include dir PRIVATE/PUBLIC as needed; add `gfx/EglDeviceSelection.{h,cpp}` to `gfx_SOURCES` under `MAGNUM_TARGET_EGL` |
| `src/esp/core/configure.h.cmake` | `#cmakedefine ESP_BUILD_WITH_HIP` |
| `src/esp/gfx_batch/CMakeLists.txt` | HIP branch: shim dir instead of `cuda_helpers/`, link `hip::host`; link the EGL selection helper |
| `src/tests/CMakeLists.txt` | HIP branch for `GfxBatchRendererTest` include dirs |
| `src/esp/hip_compat/{cuda_runtime.h,cuda_gl_interop.h,curand_kernel.h,HelperCuda.h}` (new) | shadow headers and the project-owned `checkCudaErrors` |
| `src/esp/gfx/EglDeviceSelection.{h,cpp}` (new) | device mapping described above |
| `src/esp/gfx/WindowlessContext.cpp` | call the helper instead of `config.setCudaDevice(device)` |
| `src/esp/gfx_batch/RendererStandalone.cpp` | same for `.setCudaDevice(configuration.state->cudaDevice)` when it is not `~0u` |
| `src/esp/bindings/Bindings.cpp` | `hip_enabled` attribute |
| `src_python/habitat_sim/__init__.py`, `bindings/__init__.py`, `tests/conftest.py` | export and report `hip_enabled` (report header only) |
| `pyproject.toml`, `build.sh`, `setup.py` | `HABITAT_WITH_HIP` / `--with-hip` plumbing and help text |
| `BUILD_FROM_SOURCE.md` (or a new ROCm section in it) | ROCm build and headless Mesa EGL run instructions; GPU selection; which features need an RDNA (graphics-capable) GPU |
| NOT touched | `src/esp/gfx_batch/cuda_helpers/*` (NVIDIA EULA), `RedwoodNoiseModel.cu/.cpp`, `RenderTarget.cpp` (unless risk 1's fallback is needed), `conda-build/`, `.github/` |

## Build commands (Linux, ROCm)

Prerequisites: ROCm SDK (pip `rocm[devel]` or `/opt/rocm`) with hipRAND; Mesa EGL with radeonsi (`libegl1`, `libgl1-mesa-dri`, `libegl-dev`), `git-lfs`; Python deps from `requirements.txt`; ROCm PyTorch for the gpu2gpu and GPU noise tests. Submodules: `git submodule update --init --recursive`.

```
export ROCM_PATH=<rocm root>   # pip SDK: python -c "import rocm_sdk; print(rocm_sdk.find_rocm_root())" or equivalent
HABITAT_BUILD_GUI_VIEWERS=OFF HABITAT_WITH_BULLET=ON HABITAT_WITH_HIP=ON HABITAT_BUILD_TEST=ON \
  pip install -e . --no-build-isolation \
  --config-settings=cmake.define.CMAKE_HIP_ARCHITECTURES=<gfx> \
  --config-settings=cmake.define.CMAKE_PREFIX_PATH=$ROCM_PATH
```

(`./build.sh --headless --with-hip --run-tests` once `build.sh` is plumbed.) Cap `MAX_JOBS`/`CMAKE_BUILD_PARALLEL_LEVEL` to the host's physical cores. Non-GPU baseline: the same command with `HABITAT_WITH_HIP=OFF` in a separate build directory.

## Test plan

Test data (needed by most C++ and Python tests):

```
python src_python/habitat_sim/utils/datasets_download.py --uids ci_test_assets ycb rearrange_dataset_v2 --replace --data-path data/ --no-prune
```

GPU tests that exercise the port (HIP build, RDNA GPU, hardware Mesa EGL):

1. EGL selection: `python -c "import habitat_sim; ..."` creating a `Simulator` with `gpu_device_id=0` headless must succeed without `EGL_NV_device_cuda`; log must name the radeonsi device (not llvmpipe). With two visible GPUs, `gpu_device_id=1` must pick the other PCI device.
2. `ctest --test-dir <build> --output-on-failure`, specifically `GfxBatchRendererTest` (its `cudaInterop` case must RUN, not skip), `BatchReplayRendererTest` (device-pointer branch), `GfxBatchHbaoTest`, `DepthUnprojectionTest`, `SensorTest`, `SimTest`, `GfxReplayTest`.
3. `pytest tests/test_noise_models.py` (`test_compare_gpu_cpu_redwood_depth`, both parametrizations must run, not skip).
4. `pytest tests/test_sensors.py` including every `gpu2gpu=True` parametrization (must run, not skip; it compares GPU-transferred observations with ground truth images).
5. Full `pytest -n 4 tests/` for the rest; `pytest -m sim_benchmarks` optional.

Evidence must show the GPU paths ran: the pytest report header must print `--cuda_enabled: True` (and `hip_enabled: True`), and skips in items 2-4 counted and explained.

Non-GPU regression set: the same ctest and pytest runs on a `HABITAT_WITH_HIP=OFF` build on the same host, before and after the port; the HIP build must not fail anything the non-GPU build passes. Upstream's second CI configuration (no Bullet, no GPU: `pytest tests/test_physics.py tests/test_sensors.py`) is part of this set.

CUDA no-regression: configure and build `HABITAT_WITH_CUDA=ON` with real nvcc on the mi355 fleet host (or the `cuda-12.8` conda env) at least through `noise_model_kernels`, `gfx_batch` and `habitat_sim`, and confirm the EGL helper takes the `setCudaDevice` path when `EGL_NV_device_cuda` is present (code reading plus a unit-level check if a test can mock the extension list).

Platforms: wave32 Linux RDNA (gfx1100 / gfx1201) is the primary and only fully meaningful platform. wave64 and windows: see Q2 and Q3.

## Open questions (Jeff must rule before porting)

Q1. Licence and delivery (carried from intake, still unruled):
 (a) Is leaving the two NVIDIA-EULA files (`src/esp/gfx_batch/cuda_helpers/HelperCuda.h`, `HelperString.h`) untouched in the fork, never compiled or derived from on the HIP path, sufficient? Options: (i) yes, proceed; (ii) record a licence ruling first; (iii) additionally delete them on `moat-port` and switch the CUDA path to the same project-owned macro (larger diff, changes the CUDA build).
 (b) Record `set-no-upstream habitat-sim --evidence https://github.com/facebookresearch/habitat-sim --by <who>` (upstream is archived; the API reports `archived: true` today). Planner recommends yes.

Q2. wave64 gate on CDNA. habitat-sim is a GL renderer; CDNA has no graphics pipeline, so on any wave64 fleet GPU rendering runs only on llvmpipe and GL interop is impossible. Options:
 (A) Keep wave64 with a reduced claim: the CDNA validator runs the HIP build with the EGL helper falling back to Mesa's software device (with a warning), validates the GPU Redwood noise kernel and the CPU-readback rendering tests, and the GL-interop tests (`GfxBatchRendererTest::cudaInterop`, `gpu2gpu=True`) are recorded as not runnable on CDNA. Requires the software fallback in the EGL helper (a behaviour change on hosts with no usable hardware EGL device).
 (B) Waive wave64 (configured as waivable) and validate on RDNA Linux only; the EGL helper fails loudly when no hardware device matches. Simplest and honest; the MI250X users of issue #2402 would still get software rendering only if they pass `gpu_device_id=-1`.
 (C) Validate wave64 on a graphics-capable wave64 GPU (GCN gfx906-class with radeonsi), if one exists anywhere in the fleet. Planner knows of none.
 Planner recommends (A) if the issue #2402 audience (MI250X clusters) matters, otherwise (B).

Q3. windows gate. habitat-sim has no Windows support upstream (docs list Linux and macOS; headless and EGL are Linux-only; no Windows CI). Planner suggests waiving `windows`; only Jeff can approve a waiver.

Q4. Scope confirmation (nothing is deferred by this plan, but confirm): in scope are the EGL device selection, the HIP build option, the GPU noise kernel, both GL-interop paths (`RenderTarget` readFrame*GPU and `RendererStandalone`/`BatchReplayRenderer` device pointers), Python `gpu2gpu_transfer`, tests and docs. Proposed out of scope, each needing your yes: `conda-build/` packaging (Meta-internal CUDA conda recipes with hard-coded `/public/apps/cuda` paths; no ROCm conda channel to publish to), the GUI `viewer`/`replayer` executables beyond building them (they need a display; the port does not touch them), and the audio sensor (CPU-only prebuilt binary, off by default). If risk 1's probe shows HIP-GL interop is infeasible on Mesa EGL, the port would ship without `gpu2gpu_transfer` on ROCm; that cut would come back to you before the porter proceeds.
