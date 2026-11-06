# habitat-sim notes

## Intake screen (2026-11-05, intake agent, windows-gfx1151 host)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict unknown, AMD path none). Screened at upstream `main` (shallow clone in `agent_space/habitat-sim-screen`).

### Recommendation: fork (take up), with no upstream destination

Upstream is ARCHIVED and has a sunset notice; the port's deliverable would be the AMD-Ecosystem fork itself, never an upstream PR. The AMD gap is real, specific and small, and users have asked for it (upstream issue #2402, MI250X cluster). If adopted, the adopter should also record `moatlib.py set-no-upstream habitat-sim --evidence https://github.com/facebookresearch/habitat-sim --by <who>`.

If a person judges that an archived, sunset upstream is not worth a fork-only deliverable, the decline reason would be `other` (no upstream destination, project sunset), NOT `already-supported`: there is no AMD support anywhere.

### Licence

- Top level: MIT (`LICENSE`), `licenses.py check` -> tier 1. Recorded with `set-license-spdx`.
- `licenses.py scan-nvidia` hits 2 files carrying the NVIDIA CUDA-samples EULA text ("use ... outside the terms of the EULA is strictly prohibited"):
  - `src/esp/gfx_batch/cuda_helpers/HelperCuda.h`
  - `src/esp/gfx_batch/cuda_helpers/HelperString.h`
  These are the old CUDA SDK `helper_cuda.h`/`helper_string.h` (only for `checkCudaErrors`), included by `src/esp/gfx/RenderTarget.cpp` and `src/esp/gfx_batch/RendererStandalone.cpp` under `ESP_BUILD_WITH_CUDA`. A port must NOT hipify or derive from them; replace the use with a small project-owned error-check macro and leave the two files untouched. See question 1 below.
- Submodules (16, all in `src/deps`): rapidjson, magnum, corrade, magnum-plugins, magnum-integration, magnum-bindings, m.css, pybind11, assimp, glfw, recastnavigation, tinyobjloader, bullet3, openexr, rlr-audio-propagation, zstd. All are well-known permissively licensed projects except `facebookresearch/rlr-audio-propagation` (optional audio, ships a prebuilt binary; not checked here -- the planner should confirm its licence before any build pulls it in; it is off by default).

### Maintenance status (candid)

- `archived: true` on GitHub; last push 2026-07-21. README opens with: "Beyond v0.3.4 this project is no longer receiving official active development or maintenance by Meta internal teams. Please feel free to continue forking and developing the software independently."
- Last release tag v0.3.3 (2026-02-12); last commits 2026-05-07 (sunset banner; a CUDA 13 CI fix). Activity in Feb 2026 was substantial (numpy 2, API refactors), so the code base is current, just frozen.
- Meta's stance on AMD before the sunset (issue #2402, maintainer aclegg3, 2024): "I also don't see much motivation for our team to support AMD GPUs in the foreseeable future", while inviting someone else to solve it.
- Still the base of habitat-lab and much embodied-AI research (3.8k stars), so an AMD-usable fork has an audience (LUMI-class MI250X clusters are the asker in #2402).

### Duplicate effort

None found.
- Upstream README/docs: no AMD/ROCm/HIP mention.
- Upstream PRs (all states, "hip OR rocm OR amd OR EGL device"): none relevant. No hip/rocm/amd branches.
- Upstream forks list (newest 100): none AMD-named. `AMD-Ecosystem/habitat-sim` and `ROCm/habitat-sim`: 404. GitHub repo search "habitat-sim rocm": nothing.
- Not checked: the `amd` and `AMD-AGI` orgs (SAML SSO blocks this token), so an internal effort there cannot be ruled out.

### What the CUDA surface is (it is not what the matrix lead suggests)

About 8 KB of `.cu` is ONE kernel file, and it is NOT the GL interop:
1. `src/esp/sensor/RedwoodNoiseModel.cu/.cuh` -- Redwood depth-sensor noise model kernel, uses `curand_kernel.h` (`curandState_t`, `curand_normal`). Maps to hipRAND (`hiprand_kernel.h`). Built as `noise_model_kernels` only under `BUILD_WITH_CUDA`; the CPU path exists without it.
2. CUDA-GL interop in host C++ (`src/esp/gfx/RenderTarget.cpp`, `src/esp/gfx_batch/RendererStandalone.cpp`, `src/esp/sim/BatchReplayRenderer.cpp`, bindings): `cudaGraphicsGLRegisterImage` / `MapResources` / `SubResourceGetMappedArray` / `cudaMemcpy2DFromArray` to copy color/depth/objectId render targets straight into a torch CUDA tensor (`readFrameRgbaGPU` etc.). HIP has the same API (`hip/hip_gl_interop.h`, `hipGraphicsGLRegisterImage`...). Whether ROCm's GL interop works against a Mesa radeonsi EGL context on Linux is UNVERIFIED -- the planner must test it early (it is the one piece that could be infeasible; fallback is the existing CPU readback path).
3. CMake: `option(BUILD_WITH_CUDA)` in `src/CMakeLists.txt`, `HABITAT_WITH_CUDA=ON` env in `setup.py`; `project(esp LANGUAGES C CXX CUDA)`, links cudart. CUDA is fully OPTIONAL; default builds have none.

### The real AMD blocker is EGL device selection, not CUDA

`src/esp/gfx/WindowlessContext.cpp` calls Magnum's `WindowlessGLContext::Configuration::setCudaDevice(device)` whenever `gpu_device_id != -1` (default is 0) on EGL/headless builds. Magnum resolves that through `EGL_NV_device_cuda` (`EGL_CUDA_DEVICE_NV`), which only the NVIDIA EGL driver exposes. On AMD this fails even in a non-CUDA build -- exactly the error in #2402 with the conda headless package on MI250X:
`Platform::WindowlessEglApplication::tryCreateContext(): unable to find CUDA device 0 among 9 EGL devices in total`.
`gfx_batch/RendererStandalone` has the same `setCudaDevice` path.

So headless EGL rendering on AMD should work with Mesa (radeonsi, EGL_EXT_device_enumeration) if device selection is done another way: likely `gpu_device_id=-1` already works as a workaround (untested here), and a proper fix maps a HIP device ordinal to an EGL device by PCI bus id / DRM render node (`hipDeviceGetPCIBusId` vs `EGL_EXT_device_drm(_render_node)`), or uses Magnum's plain `setDevice()` index. Fixing this in habitat-sim alone is enough; a Magnum-side helper would be a separate upstream conversation with mosra/magnum (active) and is a person's call.

### Realistic deliverable

1. EGL device selection that works on AMD (the high-value fix; benefits every AMD user, CUDA build or not).
2. HIP build option: Redwood noise kernel via hipRAND; HIP-GL interop for GPU-to-torch frame transfer if ROCm GL interop works with Mesa EGL, else documented CPU readback.
3. Replace the NVIDIA-EULA `checkCudaErrors` helpers with a project-owned macro for the HIP path.
4. Build/run docs for ROCm (headless EGL on Mesa), run the C++ tests (`src/tests`, GfxBatchRendererTest, BatchReplayRendererTest) and Python tests on a real AMD GPU.
Linux only in practice: habitat-sim supports Linux and macOS, not Windows, so the `windows` gate will need a waiver suggestion (a person approves). Dependencies are vendored submodules; no MOAT project dependency (`set-deps` not needed). Optional runtime dep: PyTorch (ROCm build) for the GPU-tensor path.

### Questions for jeff

1. Two upstream files (`src/esp/gfx_batch/cuda_helpers/HelperCuda.h`, `HelperString.h`) carry the NVIDIA CUDA-samples EULA. The plan is to leave them untouched and use our own error-check macro on the HIP path, so the port derives nothing from them. Since there is no upstream PR (archived), is that sufficient for the fork to be published, or do you want a licence ruling recorded first?
2. Upstream is archived with a Meta sunset notice, so the fork is the only deliverable. Do you want a fork-only port of a sunset project (recommended: the EGL device-selection gap is real, small, and requested), or decline with reason `other`?
