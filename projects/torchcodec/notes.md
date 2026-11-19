# torchcodec notes

## Intake screen (2026-11-05, windows-gfx1151 host, intake role)

Source: third-party "AMD AI Software Gap Matrix" seed, entry "TorchCodec GPU decoding (PyTorch's media I/O, NVDEC)", verdict not-yet, AMD path none. That lead is out of date for JPEG and accurate for video.

Upstream: https://github.com/meta-pytorch/torchcodec (1197 stars, active, last push 2026-11-06, not archived). Screened at 403a7b29531ba66606610a539f52870074df69bd.

### Licence

BSD-3-Clause (tier 1, `licenses.py check`), confirmed in LICENSE ("Copyright 2024 Meta"). `licenses.py scan-nvidia agent_space/torchcodec-screen`: no NVIDIA proprietary licence text. The vendored `nvcuvid_include/` headers (cuviddec.h, nvcuvid.h) are NVIDIA copyright under the MIT-style permission notice; the vendored `giflib/` carries its own permissive headers. No submodules. Clean.

### CUDA surface

- One kernel file: `src/torchcodec/_core/color_conversion.cu` (12844 bytes, NV12 to RGB).
- Video decode: `CudaDeviceInterface` (FFmpeg CUDA hwaccel) and `BetaCudaDeviceInterface` (direct NVDEC via `NVCUVIDRuntimeLoader`, `NVDECCache`), gated by `ENABLE_CUDA` / `USE_CUDA` in `_core/CMakeLists.txt`.
- JPEG: nvJPEG decode/encode (`DecodeJpegCuda.cpp`, `EncodeJpegCuda.cpp`), gated by `TORCHCODEC_BUILD_NVJPEG`.
- Device abstraction: `_core/DeviceInterface.h` is a pluggable virtual interface (`initialize_video_decoding`, `convert_av_frame_to_frame_output`, `send_packet`/`receive_frame`, encoder hooks) with `register_device_interface(...)` exported as `TORCHCODEC_THIRD_PARTY_API`. `test/third-party-interface/` is a worked example. PR #1151 (dvrogozh, merged 2026-06) added entry-point autoloading of out-of-tree plugins (`torchcodec.backends` group). A rocDecode backend could be written entirely against this API without touching the tree.

### Existing AMD effort (duplicate check)

All found with `gh pr list --repo meta-pytorch/torchcodec --state all --search "rocm OR rocjpeg OR rocdecode OR hip"` and follow-up `gh pr view`.

1. rocJPEG, in-tree, active, maintainer-driven with the AMD team:
   - #1554 "Add ROCJPEG decoder" https://github.com/meta-pytorch/torchcodec/pull/1554 -- opened 2026-07-27 by NicolasHug (torchcodec maintainer, "ported from TV", i.e. the torchvision rocJPEG path). OPEN, not draft, mergeStateStatus DIRTY / CONFLICTING, last update 2026-08-13. Adds `DecodeJpegRocm.cpp`, `packaging/install_rocjpeg.sh`, `.github/workflows/linux_rocm.yaml` (a ROCm CI job), wheel repair. Open discussion: whether the ROCm wheel bundles librocjpeg (as CUDA wheels bundle libnvjpeg) or treats it as a runtime dependency of the ROCm install.
   - #1642 "[ROCm] Test ROCm failures on PR#1554" https://github.com/meta-pytorch/torchcodec/pull/1642 -- akashveramd (AMD), base `rocm_decoder` (the #1554 head branch, in the upstream repo). OPEN, CLEAN/MERGEABLE, updated 2026-11-06, Dr. CI green, moves CI to ROCm 10.1; jithunnair-amd (AMD) has a changes-requested review on it.
   - #1689 and #1769 (AryanSalmanpour, AMD), both MERGED into `rocm_decoder_analysis`: rocJPEG batch-decode corruption fix and a tolerance revert.
   - Upstream branches `rocm_decoder` and `rocm_decoder_analysis` exist in meta-pytorch/torchcodec.
   Verdict: authoritative and live. The maintainer and three AMD engineers are landing it with a ROCm CI job. Nothing for MOAT to add except possibly gfx1151 / Windows evidence, and torchcodec has no ROCm-Windows story at all yet.

2. rocDecode video decode, out-of-tree, AMD, parked:
   - #1247 "[WIP] Rocm GPU enablement" https://github.com/meta-pytorch/torchcodec/pull/1247 -- Cemberk (public member of the ROCm GitHub org), draft, +4823 lines, adds `RocDecodeDeviceInterface`, `RocDecodeRuntimeLoader`, `HIPColorspaceKernels.cpp`, vendored rocdecode headers, Dockerfiles. CONFLICTING, last activity 2026-03-14. Motivation stated: HF Transformers unit tests on ROCm fall back to CPU decode.
   - NicolasHug's response on #1247: TorchCodec wants new devices as OUT-OF-CORE extensions in a separate repository implementing `DeviceInterface` (pointing at dvrogozh/torchcodec-xpu as the model), not in-tree. Cemberk started https://github.com/Cemberk/torchcodec-rocm accordingly (2 commits, last push 2026-02-26, README is a title only).
   - AndreasKaratzas (vLLM ROCm) commented that vLLM wants upstream ROCm packages; today vLLM builds torchcodec v0.10.0 from source for ROCm torch (`vllm/tools/install_torchcodec_rocm.sh`), FFmpeg CPU decode only.
   Verdict: authoritative intent (an AMD ROCm-org member), but parked for eight months and not yet functional as a published plugin.

3. Other: #148 "[AMD] hipify torchcodec" (NicolasHug, 2024, closed); #558 / #832 Intel XPU and ffmpeg-vaapi (dvrogozh, Intel) show the out-of-tree plugin route in practice. No ROCm or AMD-Ecosystem repo named torchcodec (`gh search repos torchcodec --owner ROCm`, `--owner AMD-Ecosystem`: empty). The `amd` and `AMD-AGI` orgs are behind SAML SSO for this token, so an internal effort there could not be checked.

### Viability and fit

- JPEG half: already being done upstream by the maintainer and AMD. A MOAT fork of torchcodec would compete with it.
- Video half: the real gap. But the maintainer has said where it goes -- a separate plugin repository, not a PR to torchcodec. That does not fit MOAT's fork-upstream / `moat-port` / upstream-PR shape: the deliverable would be a NEW repository (rocDecode `DeviceInterface` plugin, porting `color_conversion.cu` and the NVDEC cache/parser flow to rocDecode + HIP), with no upstream to fork. And an AMD engineer already owns the start of exactly that repository.
- Context: TorchCodec is in the PyTorch org (meta-pytorch) where jeff is a maintainer, so any in-tree contribution (e.g. Windows/RDNA coverage for the rocJPEG path, or a doc pointing at a ROCm plugin) would land through the normal PyTorch review path with Meta CLA; jeff could also nudge #1554 forward directly.

### Recommendation

Decline, reason `duplicate`. JPEG is being landed in-tree by the maintainer with AMD engineers (#1554/#1642), and the video path the maintainer wants out-of-tree, where an AMD ROCm-org member already started the plugin (Cemberk/torchcodec-rocm, #1247). The useful action is coordination, not a MOAT port: ask the AMD owners (Cemberk, jithunnair-amd) whether the rocDecode plugin is still theirs before anyone builds a second one.

### Question for jeff

The rocDecode video-decode plugin is the real remaining gap, and its only start (Cemberk/torchcodec-rocm, PR #1247) has been idle since March 2026. Should MOAT offer to take it over or contribute to it (as a new out-of-tree plugin repository, which is outside MOAT's usual fork-and-upstream-PR shape), after checking with Cemberk / jithunnair-amd that it is not still owned internally? If not, the decline stands as recorded.
