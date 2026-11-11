# slime notes

## Intake screen (2026-11-05, intake agent, gap-matrix seed)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: partial / AMD path: upstream). Screened at upstream main, shallow clone in `agent_space/slime-screen`. THUDM/slime: ~8.8k stars, very active (pushed 2026-11-05), RL post-training on Megatron-LM + SGLang.

### Licence
- `licenses.py check THUDM/slime`: Apache-2.0, tier 1. No `.gitmodules`.
- `licenses.py scan-nvidia agent_space/slime-screen`: not needed for the decline below; the CUDA files are slime's own.

### CUDA surface (~18 KB)
- `slime/backends/megatron_utils/kernels/int4_qat/fake_int4_quant_cuda.cu` (12.8 KB): INT4 QAT fake-quant. Already ROCm-enabled upstream: #2274 "[ROCm] Support the INT4 QAT kernel on ROCm" (merged 2026-08-15); its `setup.py` branches on `torch.version.hip` and sets `PYTORCH_ROCM_ARCH`.
- `slime/backends/sglang_utils/jit_kernels/csrc/gemm/glm5_router_gemm.cuh` (5.9 KB): an SGLang JIT kernel used only by the GLM-5 training/inference alignment path (`megatron_utils/alignment/deepgemm_forward.py`, built around DeepGEMM, which is NVIDIA-only). Optional, model-specific; not on the AMD recipe's path.

### Existing AMD support (decides this)
- Upstream docs: `docs/en/platform_support/amd_tutorial.md` (MI300/MI325 recipe, contact point at AMD), `scripts/run-qwen3-4B-amd.sh`, `docker/Dockerfile.rocm`, `Dockerfile.rocm_MI350-5` (gfx950 on `rocm/sgl-dev`, AITER), `slime/utils/rocm_checkpoint_writer.py`, accelerator abstraction branching on `torch.version.hip`.
- Upstream PRs: AMD-authored ROCm enablement since 2025-07 (#19, #20, #119, #124, #444 by yushengsu-thu, whose GitHub profile lists AMD; #279, #510 by gramesh-amd), #2274 merged INT4 QAT on ROCm, and two OPEN AMD ROCm refresh PRs: #2369 "[ROCm] Update and unify ROCm support for gfx942 and gfx950" (+6135/-2028) and #2506 "[ROCm] Support ROCm 7.15 on MI300/MI325" by jun-amd (2026-10-29).
- So the ROCm path is upstream and AMD engineers are actively maintaining it. The SGLang and Megatron deps are AMD-supported (SGLang disposition already-supported; Megatron through ROCm/Primus).
- `amd`/`AMD-AGI` orgs not searchable with this token (SAML SSO); immaterial given the upstream evidence.

### Residual gap (not MOAT-sized)
- `glm5_router_gemm.cuh` and the DeepGEMM alignment path have no ROCm path; they belong to the AMD team already working upstream, not a parallel MOAT port. RDNA/Windows are out of scope for a Megatron + SGLang multi-GPU RL stack.

### Recommendation: decline, SKIP_REASON `already-supported`
Mature upstream ROCm support (docs, Dockerfiles for gfx942/gfx950, ROCm-enabled QAT kernel) maintained by AMD engineers with live ROCm PRs open; MOAT would duplicate AMD's own work.

## Take-up ruling (2026-11-09, recorded 2026-11-11)

Jeff Daily ruled that slime gets an AMD-Ecosystem home: a new fork created by a person, not a transfer of jun-amd/slime. Intake had recommended decline (already-supported: ROCm support upstream, AMD engineers' PRs THUDM/slime#2369 and #2506 open). Recorded `set-intake slime fork` and `awaiting-fork`; scope (validation of the upstream ROCm path on all gates, plus the GLM-5 DeepGEMM-alignment router kernel that lacks a path) is for the planner, with any cut put to Jeff.
