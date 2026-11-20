# Falcor notes

## Intake screen (2026-11-05, gap-matrix seed; matrix verdict: unknown)

### Licence
- Top level `LICENSE.md` is BSD-3-Clause (NVIDIA copyright). GitHub reports NOASSERTION. Recorded `license_spdx` BSD-3-Clause.
- `python3 utils/licenses.py scan-nvidia agent_space/falcor-screen` (depth-1 clone of `master`): no NVIDIA proprietary licence text in the tree. Submodules are third-party permissive (pybind11 via an NVIDIA employee's fork, glfw, args, fmt, imgui, Vulkan-Headers).
- Not in the tree, but linked: `LICENSE.md` and README state Falcor "links to" DLSS, RTXGI, RTXDI and NRD under their own NVIDIA licences; DLSS, RTXDI and NRD are fetched by `setup.bat`/packman from `dependencies.xml` (packages `rtxdi`, `nrd`, plus `nv-usd`, `nvtt`, `slang`, `nanovdb`, `pix`), and OptiX is a separately downloaded SDK. A port would not edit those, but building and validating Falcor as shipped pulls them in. The tree itself is clean file by file.

### Viability (the deciding question)
- Falcor is a DX12/Vulkan real-time rendering research framework built on Slang (slang-gfx). Its GPU work is in Slang shaders, not CUDA.
- CUDA surface: one `.cu` file (`Source/Samples/CudaInterop/CopySurface.cu`, 2.8 KB, a sample), the interop helpers `Source/Falcor/Utils/CudaUtils.{h,cpp}` and `CudaRuntime.h` (shared-resource interop between the graphics API and CUDA), and the `OptixDenoiser` render pass (OptiX, no ROCm equivalent; would map to an AMD denoiser rather than a HIP port). All of it is optional: `FALCOR_HAS_CUDA` is OFF unless the CUDA toolkit is found.
- Platform coverage, not CUDA, is what decides whether Falcor runs on AMD: it targets Windows DX12 (with DXR) primarily, and Vulkan; it is not a HIP/ROCm workload. Making the CUDA interop sample use HIP would be a few-KB change that no Falcor user needs to run Falcor on an AMD GPU.
- Upstream: active (pushed 2026-09-21, 3.3k stars), not archived. No AMD/ROCm/HIP PRs; three closed old issues about AMD GPUs (#16 RX Vega support, #31 AMDGPU Pro GLSL, #252 GPU selection).

### Duplicate effort
- None found on ROCm, AMD-Ecosystem, or by search. The `amd` and `AMD-AGI` orgs are behind SAML SSO for this token (not decisive here, given the recommendation).

### Recommendation
Decline, `not-a-target`: a DX12/Vulkan renderer whose CUDA is an optional interop sample plus an OptiX denoiser pass; there is no meaningful HIP port, and AMD support for it is a graphics-API question outside ROCm. The licence is clean in-tree, so this decline rests on scope, not licence.
