# lmdeploy notes

## Intake screen (2026-11-05, intake agent, from the AMD AI Software Gap Matrix seed)

Upstream: https://github.com/InternLM/lmdeploy, screened at bc2c8da35a4e70133f4b55acba04e2da69e5cedc (2026-11-05). Active (pushed 2026-11-05), not archived, ~8.1k stars. Matrix lead: "LMDeploy (TurboMind)", verdict partial, AMD path upstream.

### Licence
- `utils/licenses.py check InternLM/lmdeploy`: Apache-2.0, tier 1, cleared to contribute.
- `utils/licenses.py scan-nvidia agent_space/lmdeploy-screen`: no NVIDIA proprietary licence text. No git submodules. TurboMind's CMake FetchContents NVIDIA/cutlass (BSD-3), fmt, yaml-cpp, Catch2, xgrammar (all permissive); only relevant if TurboMind were ever in scope.

### Two engines
1. **PyTorch engine** (`lmdeploy/pytorch/`): Python plus 41 Triton kernel modules under `lmdeploy/pytorch/kernels/cuda/`. No compiled extension of its own. Optional third-party accelerators imported when present: deep_gemm, flash_mla, deep_ep, flashinfer, flash_attn_interface (FA3), fla (flash-linear-attention), tilelang.
2. **TurboMind** (`src/turbomind/`, ~1.43 MB of `.cu/.cuh`, 128 `.cu` files): C++/CUDA, `find_package(CUDAToolkit REQUIRED)`, built only when `LMDEPLOY_TARGET_DEVICE=cuda`. Heaviness: 165 inline `asm` sites in 31 files; `mma.sync`/`ldmatrix`/`cp.async` in 16 files; CuTe in ~33 files; 114 files reference SM90/wgmma/TMA (`gemm_universal_sm90_*.h`, `gmma_*_sm90.h`, `linear_attn/kernel/sm_90`, `sm_120`); 94 `__shfl` sites and ~91 32-lane constants/masks. Its GEMM is a hand-written SM70/SM80/SM90 framework, so hipify has nothing to translate on the hot path: a TurboMind HIP backend would be an AMD-native reimplementation (CK/rocWMMA/MFMA), not a port. No upstream branch or PR for a TurboMind HIP path exists.

### What upstream supports on ROCm today
- `docs/en/get_started/installation.md` documents ROCm install: `rocm/pytorch:latest` docker plus `LMDEPLOY_TARGET_DEVICE=rocm pip install`. That target skips TurboMind and installs `requirements/runtime_rocm.txt`; users then run the PyTorch engine with `device_type='cuda'` (torch's HIP alias). `'rocm'` is not a valid `device_type` (`messages.py:656` allows cuda/ascend/maca/camb).
- Added by PR #3925 (merged 2025-09-09) by Vivicai1005 (GitHub profile company: AMD): docs, a ROCm tiling branch `_kernel_meta_rocm` in `pytorch/kernels/cuda/flashattention.py` (LDS out-of-range fix), `runtime_rocm.txt`, `requirements_rocm.txt`, and a `'rocm'` branch in `utils.is_bf16_supported`.
- No ROCm CI. Maintainer (windreamer, on issue #4022): "Sadly we do not have a usable RoCM environment to test and solve it now."
- DLInfer (`pytorch/backends/dlinfer`) serves Ascend, MACA (MetaX) and Cambricon only; it is not an AMD path.

### Concrete gaps found in the screen (cheap reads, not yet run on a GPU)
- **bf16 misdetected on ROCm.** `lmdeploy/utils.py` `is_bf16_supported('cuda')` requires `torch.version.cuda` (None on ROCm) and compute capability major >= 8, so it returns False on every AMD GPU; the `'rocm'` branch is unreachable because no caller passes `'rocm'`. The engine logs "Device does not support bfloat16" and falls back to fp16. Open issue #4022 (7900 XTX / gfx1100, ROCm 6.4.4, Qwen3-0.6B) shows exactly that warning followed by `!!!!...` garbage output, consistent with fp16 overflow. Open since 2025-10-03, unfixed.
- **Packaging drift.** `runtime_rocm.txt` is `common.txt` + accelerate + peft only; `runtime_cuda.txt` additionally pins torch/torchvision/triton ranges and lists flash-linear-attention, tilelang, apache-tvm-ffi, aiohttp, prometheus_client, opencv-python-headless. Some of those are needed by the serve path and by linear-attention models (Qwen3.5 etc.) regardless of vendor.
- Untested on RDNA/wave32 and unclear on gfx90a/gfx942 beyond the PR author's run; `tests/pytorch/kernel/` has 60 Triton kernel tests that can be run on ROCm as the GPU gate.

### Duplicate effort
- No ROCm/, AMD-Ecosystem/ or AMD-named fork of lmdeploy found (`gh api search/repositories` for org:ROCm and org:AMD-Ecosystem; fork list filtered for amd/rocm owners). The `amd` and `AMD-AGI` orgs are behind SAML SSO for this token, so a fork there could not be checked; that part of the duplicate check stays open.
- Upstream PR/branch search (`gh pr list --state all --search "rocm OR hip OR amd"`, branches grep): only #3925 is ROCm work; no open ROCm PR, no hip/rocm/amd branch.
- AMD participation exists (the #3925 author) but appears to be a one-off contribution: the follow-up bug #4022 went to them in October 2025 and is still open. Coordination, not a race: if the fork is made, the upstream PR should credit/cc that contributor.

### Recommendation: fork, narrow scope (PyTorch engine on ROCm), TurboMind out of scope
Work: validate the PyTorch engine on ROCm on wave64 and wave32 (run `tests/pytorch/kernel` plus an end-to-end pipeline smoke on a small model in bf16), fix `is_bf16_supported` for HIP (closes #4022's root cause), bring `runtime_rocm.txt` to parity with the vendor-neutral parts of `runtime_cuda.txt`, fix any Triton kernel tiling that fails on wave32/RDNA LDS limits as #3925 did for its target, and update the ROCm install docs with the validated scope. Upstream is active, already accepts ROCm changes, and has no AMD hardware to test on, so validated fixes have a clear landing path.

TurboMind: not recommended. It is a CUDA-native inference engine whose hot path is hand-written SM70/80/90 GEMM and attention using PTX, mma.sync, CuTe and wgmma; a HIP backend is a multi-month AMD-native rewrite, upstream has shown no interest in one, and on ROCm the PyTorch engine is upstream's documented route. If someone wants TurboMind on AMD that is a separate decision and should be recorded as a deferral.

### Dependencies
No MOAT hard dependencies. Optional GPU dependencies of the PyTorch engine, relevant on ROCm: triton (ROCm build from the torch wheel), flash-linear-attention (pure Triton), tilelang (EmbeddedLLM/tilelang AMD fork exists), flash_attn (ROCm/flash-attention exists). deep_gemm, flash_mla, deep_ep, flashinfer are CUDA-only and are optional; the ROCm path should keep them off.

## Scope ruling (2026-11-11, Jeff Daily)

Jeff Daily ruled TurboMind in scope now, not deferred: "I don't see why not to port TurboMind now. Sure it's a huge effort, but we've got the time and the agents and the tokens." The port covers the whole project: the PyTorch engine on ROCm (validation on all gates, the HIP `is_bf16_supported` fix behind #4022, `runtime_rocm.txt` parity) AND a HIP port of TurboMind (`src/turbomind/`, built for `LMDEPLOY_TARGET_DEVICE=rocm` alongside the CUDA build, CUDA path preserved). The planner should stage it (build system and non-GEMM kernels first, then attention, then the hand-written SM70/80/90 GEMM paths with their inline asm / mma.sync / ldmatrix / cp.async / CuTe / wgmma, mapped to MFMA on wave64 and WMMA on wave32, or to hipBLASLt / Composable Kernel where a library covers the shape), and record per-stage evidence. Any further cut inside TurboMind is again Jeff's decision.
