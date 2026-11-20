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

## Planner (2026-11-20, linux-gfx942, read-only)

### Decisions needed from Jeff before a porter starts

The project is recorded `planned` so the plan is visible, but Q1 and Q2 change what the first porter round does, and Q3/Q4 are cuts inside the ruled scope. Nobody could answer during this job. Until Jeff rules, a porter should not start (a `set-hold` is the person's tool for that if wanted).

**Q1. Two open upstream ROCm PRs now overlap stage P.** Since intake, menglcai opened https://github.com/InternLM/lmdeploy/pull/5135 (`is_bf16_supported` on ROCm, the #4022 root cause; 2026-11-15) and https://github.com/InternLM/lmdeploy/pull/5138 (W8A8 int8 on ROCm: portable `libdevice.round` + 2-stage autotune configs for 64 KB LDS; 2026-11-16). Both open, unreviewed by maintainers, small (+3 and +10/-3), with careful evidence on gfx90a, gfx1201, gfx1150, gfx1151 and an H100 CUDA check. The author is an active ROCm-ecosystem contributor (open PRs on ROCm/aiter, ROCm/AMDMIGraphX, ROCm/torch_migraphx, vllm-omni ROCm fixes; no affiliation on the profile). Options:
- A. Do not reimplement those two fixes. The porter merges the two PR branches (authorship preserved) into `moat-port` so our validation runs with them, and our own upstream PR excludes them (it is based after they merge, or states it depends on them).
- B. Reimplement independently in our port. Duplicates open work by someone already engaged with upstream; not recommended.
- C. Coordinate with menglcai on the rest of the PyTorch-engine ROCm work (their own baseline in #5135: 170 failed / 16 crashing files on gfx90a, about 345 failed on RDNA, at upstream main), so the two efforts do not race. Needs Jeff (the person may be a colleague; any upstream comment is a person's decision).
- Planner recommendation: A, plus C if Jeff knows the contributor.

**Q2. One MOAT project or two.** Stage P (PyTorch engine) is weeks of Python/Triton work with no build; TurboMind (T0-T5) is months, including native MFMA/WMMA attention and GEMM. Under one project, P's fixes cannot reach upstream until TurboMind is done, and the single upstream PR would be enormous for a maintainer with no AMD hardware. Options:
- A. Keep one project: one `moat-port`, one upstream PR at the end, stages recorded in notes.
- B. Split: `lmdeploy` keeps stage P (plus packaging/docs); a new project (for example `lmdeploy-turbomind`, scaffolded by a person against the same fork with its own port branch via `set-fork-branch`) takes T0-T5 and can depend on `lmdeploy`. Under B the PyTorch-engine project has no Windows story upstream (its Triton requirement is pinned to Linux in `runtime_cuda.txt`), so it would need a `windows` gate waiver, which only a maintainer can approve; the agent suggestion is recorded here only. TurboMind on Windows (tp=1, as upstream supports on CUDA) carries the windows gate in the TurboMind project.
- C. One project, staged upstream PRs (P first, TurboMind later). MOAT tooling assumes one upstream PR per project, so this would mean a tooling change or a manual sequence.
- Planner recommendation: B. plan.md is written so stage P stands alone either way.

**Q3. TurboMind DeepEP (multi-node MoE all-to-all) on ROCm.** `comm/nccl/deepep/` is built whenever NCCL is found, needs the NCCL >= 2.30.4 device API (`nccl_device.h`, `ncclDevComm`, LSA windows) and JIT-compiles DeepEP kernels with nvcc at run time. RCCL has no device API. It is used only for `moe_a2a_backend='deepep'`, which `lmdeploy/turbomind/turbomind.py:92-93` auto-selects when `nnodes > 1`. Options:
- A. Build TurboMind without DeepEP on ROCm (new CMake guard; Python raises a clear error for `deepep` on ROCm and `auto` picks `default`); register a deferral for multi-node EP on ROCm.
- B. Port DeepEP's elastic dispatch/combine onto rocSHMEM or MORI. A separate multi-month effort on a third-party library.
- Planner recommendation: A with a deferral. (NVLS multicast in `comm/cuda_ipc` has no AMD hardware equivalent and is compiled out; that is not a choice.)

**Q4. PyTorch-engine features that hard-require CUDA-only third-party libraries.** Speculative decoding requires FA3 (`backends/cuda/attention/__init__.py:28` raises without it); DeepSeek-V3.2, GLM-5 and DeepSeek-V4 configurations require flash_mla and/or deep_gemm; EP uses deep_ep; the optional all-reduce fusion uses flashinfer. None has a ROCm build. Options:
- A. Leave them off on ROCm; features and models that hard-require them stay unsupported on ROCm, stated in the docs. Everything with an upstream Triton fallback is in scope.
- B. Add aiter-backed (or ROCm/flash-attention) providers for MLA decode and the speculative-decoding attention: a new optional ROCm dependency and a larger upstream diff.
- C. A now, B registered as deferrals (one per provider) for a later round.
- Planner recommendation: C.

**Decisions the plan took itself (overrule if wanted):** AMD-native MFMA (gfx9, wave64) and WMMA (gfx11/gfx12, wave32) kernels inside TurboMind's own attention and GEMM frameworks, with no Composable Kernel or aiter dependency (TurboMind's own weight packing and int4/int8 paged-KV layout would need re-packing adapters); hipBLASLt for dense fp16/bf16 GEMM through TurboMind's existing cuBLAS family; TurboMind's SIMT code keeps `WARP_SIZE = 32` as a logical warp with every shuffle pinned to width 32; the sm90/sm120 instruction-stack libraries (`gemm2_sm90`, `linear_attn_kernels_sm90`, `linear_attn_kernels_sm120`) are `scoped_out` in `surface.json` as NVIDIA-only specializations, while the formats only they serve today (fp8 w8a8 block-scaled, nvfp4, mxfp4 x fp8) are separate in-scope `feature:` entries for the AMD GEMM family; the PyTorch engine stays on `device_type='cuda'`.

### What was done

- Upstream cloned at 63549b282790f77ebf4bf2a0d6e44a70de14d85b (2026-11-20) into `projects/lmdeploy/src` (read-only, depth 1). Inventory done with four read-only sub-analyses (TurboMind GEMM; attention/linear-attn/core; TurboMind host/comm/build; PyTorch engine). Spot-checked against source: `kernels/core/common.h:5-38,68` (arch macros keyed on `__CUDA_ARCH__`, `WARP_SIZE = 32`), `comm/nccl/CMakeLists.txt:12` and `deepep/CMakeLists.txt:27-49` (DeepEP unconditional, NCCL >= 2.30.4), `python/bind.cpp:83-93` (no `kDLROCM`), `GatedDeltaNetLayer.cc:69` (`getSMVersion()*10`), `backends/cuda/blockedf8_modules.py:238` (`_has_gluon` on major == 9), `backends/cuda/awq_modules.py:14-16` (TurboMind AWQ provider on `(9,0)`), `kernels/cuda/awq_kernels.py:35-50` (PTX `lop3`/`fma.f16x2` in Triton), `backends/cuda/attention/__init__.py:15-37` (FA3 required for speculative decoding).
- Since intake, upstream moved: `runtime_cuda.txt` now also lists tilelang 0.1.11 and apache-tvm-ffi; the sm90 GEMM stack gained nvfp4 and mxfp4 x fp8 kernels; `linear_attn` gained sm_90/sm_120 CuTe families. `runtime_rocm.txt` is unchanged.
- `ext_type` set to `cmake` (TurboMind; the PyTorch engine has no compiled code).
- `surface.json`: generated floor (253 CUDA files, 71 components) plus 17 planner entries the census cannot see (PyTorch engine and Triton kernels, Python test suites, end-to-end validations, setup.py rocm target, docs, DLPack binding, driver API, DeepEP runtime nvcc JIT, sm90-only GEMM formats, attention prefill, Windows tp=1). `scoped_out`: the three sm90/sm120 libraries and `option:SPARSITY_SUPPORT` (cuSPARSELt include only, no calls). DeepEP's `library:token_dispatcher` and `library:turbomind_nccl_stub` are left unaccounted on purpose until Q3 is ruled. `utils/surface.py check lmdeploy` passes (it judges only once the port claims success).
- Plan: `plan.md` (stages P, T0-T5 with per-stage gates, file-by-file list, risk list, build and test commands).

### Useful facts for the porter

- Upstream's only HIP branch is `hip_mode = getattr(torch.version, 'hip', None) is not None` in `flashattention.py:555`; follow that idiom.
- `menglcai`'s PR #5135 notes Triton <= 3.8 truncates bf16 multiplies on gfx11/gfx12 (triton-lang/triton#11217, fixed in 3.9): bf16 rotary tests fail on RDNA with older Triton. Record the Triton version with every RDNA result.
- `gemm/test/test_utils.cu` is compiled into production `gemm2_core`, so production TurboMind links cuBLAS and cuRAND; keep that in mind when mapping libraries.
- `models/llama/CMakeLists.txt` is never added (names a nonexistent `LlamaV2.cc`); the generated surface lists `library:Llama` from it; it is `scoped_out` as dead upstream code (the llama `.cu` files build through `library:models`).
- `test_ipc_comm` (gloo) is built even without `BUILD_TEST`.
