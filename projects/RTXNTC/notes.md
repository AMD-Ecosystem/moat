# RTXNTC notes

## Intake screen (2026-11-05, gap-matrix seed; matrix verdict: partial, upstream)

### Licence (the deciding question)
- Top level `LICENSE.txt` is the "NVIDIA RTX SDKs LICENSE", a proprietary SDK licence, not an open-source one. GitHub reports NOASSERTION. Recorded `license_spdx` LicenseRef-NvidiaProprietary, the SPDX id the source files themselves declare.
- Grant (section 1): install and use the SDK; "modify and create derivative works of sample source code delivered in the SDK"; distribute SDK materials only "as incorporated in object code format into a software application". Limitations (section 4b): "you may not copy, sell, rent, sublicense, transfer, distribute, modify, or create derivative works of any portion of the SDK" except as expressly provided; 4e forbids using it in any way that would subject it to an open-source licence.
- `python3 utils/licenses.py scan-nvidia agent_space/rtxntc-screen` (depth-1 clone of `main`, submodules not fetched): 86 files carry NVIDIA proprietary text (`SPDX-License-Identifier: LicenseRef-NvidiaProprietary` plus the "licensors retain all intellectual property" clause), including the top-level `CMakeLists.txt`, `libraries/ntc-utils/*`, `libraries/ntc.py`, and the `samples/renderer` C++/HLSL/Slang sources.
- The CUDA lives in the submodule `libraries/RTXNTC-Library` (NVIDIA-RTX/RTXNTC-Library). Cloned it shallow to `agent_space/rtxntc-screen-lib` and scanned: 103 files flagged, same `LicenseRef-NvidiaProprietary` header on the 8 `.cu` files (the CUDA compression/training path), under the same RTX SDKs LICENSE. Other submodules: Donut (NVIDIA-RTX), RTXTF-Library, RTXTS-TTM, optix-dev, lodepng, implot (not scanned; the decision does not depend on them).
- Every hit read is real licence text on code the port would have to modify (the CUDA library), and the licence permits derivative works only of sample code, which the core library is not, and forbids distributing modified SDK source. A HIP port of the library could not be published as source at all.

### Viability
- What uses CUDA: the NTC compression (encoder training) path in RTXNTC-Library, 8 `.cu`; Windows build requires the CUDA SDK. Decompression (on-load and on-sample) runs in DX12/Vulkan shaders and already works on AMD: README says it is validated on AMD Radeon RX 6000 series, with known AMD issues (DX12 LinAlg incompatible with the current AMD developer preview driver; Feedback-mode inference broken on AMD GPUs). Those are driver/graphics-API items, not ROCm work.
- Upstream: active (pushed 2026-11-04, ~700 stars), not archived. No AMD/ROCm/HIP PRs or issues; no community ROCm port found by search; nothing on ROCm or AMD-Ecosystem. The `amd` and `AMD-AGI` orgs are behind SAML SSO for this token (not decisive given the licence).

### Recommendation
Decline, `license-blocked`: proprietary NVIDIA RTX SDK licence on every source file, including the CUDA library a port would modify; derivative works are allowed only of sample code and modified SDK source may not be distributed. Fails jeff's file-by-file bar outright; there is no clean subset worth porting.
