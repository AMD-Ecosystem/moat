# axolotl notes

## Intake screen (2026-11-05, intake agent, gap-matrix seed)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: partial / AMD path: upstream). Screened at upstream main, shallow clone in `agent_space/axolotl-screen`. axolotl-ai-cloud/axolotl: ~12.7k stars, very active (pushed 2026-11-05). LLM fine-tuning framework on transformers/PEFT/TRL.

### Licence
- `licenses.py check axolotl-ai-cloud/axolotl`: Apache-2.0, tier 1.
- `licenses.py scan-nvidia agent_space/axolotl-screen`: no NVIDIA proprietary licence text. The vendored Marlin sources carry Apache-2.0 (vLLM) and the original Marlin headers (Elias Frantar / Neural Magic), per `marlin_w4a16/_csrc/PROVENANCE.md`.

### CUDA surface (~45 KB, all optional)
- Only `src/axolotl/integrations/kernels/libs/scattermoe_lora/marlin_w4a16/_csrc/` (ops_standalone.cu, repack_standalone.cu, two sm80 kernel instantiations, marlin.cuh, marlin_dtypes.cuh): NVFP4 weight-only Marlin MoE GEMM vendored verbatim from vLLM, JIT-built with `torch.utils.cpp_extension.load` for an optional sm80+/sm120 ScatterMoE-LoRA backend. Falls back to CUTLASS/DeepGEMM/torch when `marlin_w4a16_available()` is False.
- Marlin is `mma.sync`/`ldmatrix` PTX tensor-core code; there is no mechanical HIP path (vLLM itself does not run Marlin MoE on ROCm). Out of scope for a port; at most defer.
- Small real gap found: `marlin_w4a16_available()` checks `torch.cuda.is_available()`, capability major >= 8 and `CUDA_HOME is not None`, with no `torch.version.hip` check. On ROCm torch `get_device_capability()` reports e.g. (9, 4) for gfx942, so on a host where `CUDA_HOME` resolves it would try to JIT-hipify the Marlin PTX and fail. Candidate one-line fix.
- `quantize_moe_experts` is explicitly CUDA-only (`utils/schemas/config.py:1021,2150`, `cli/config_options.py:2729`, `docs/expert_quantization.qmd:142` "not tested with ROCm"). Worth checking on ROCm: it is bitsandbytes-based, and bitsandbytes now supports ROCm.

### Existing AMD support (partial, docs-level, community)
- Docs say "NVIDIA ... or AMD GPU" (`README.md:113`, `docs/installation.qmd:16`), `docs/attention.qmd:188` points to flash-attention's AMD support, `docs/lora_optims.qmd:98` says the Triton LoRA kernels run on AMD, and `docs/amd_hpc.qmd` is an HPC install guide pinned to `module load rocm/5.7.1` (stale).
- No ROCm Dockerfile (`docker/` has only CUDA/Blackwell uv images), no ROCm CI, no ROCm extras in `pyproject.toml`.
- Upstream PRs (`--search "rocm OR amd OR hip OR mi300 OR mi355"`): #1550 "Add experimental install guide for ROCm" (xzuyn, OPEN since 2024-04, stale); #3260 MoE Grouped MM (lhl, open, not ROCm-specific); #2343 merged doc fix for the AOTriton flag. Old AMD issues #960, #964, #1519 (closed). No AMD-org maintainer effort found; no ROCm/ or AMD-Ecosystem fork. `amd`/`AMD-AGI` orgs not searchable with this token (SAML SSO), so that one check is open.

### GPU dependencies (record only; none is a MOAT porting project)
- Core (non-darwin): bitsandbytes 0.50.2 (ROCm upstream; MOAT screen bitsandbytes = decline already-supported), liger-kernel 0.8.5 (ROCm upstream; MOAT screen Liger-Kernel = decline already-supported), triton, xformers >= 0.0.33 (ROCm wheels exist, coverage partial), torchao 0.18.0 (ROCm wheels), flash-linear-attention 0.6.0 (Triton), HF `kernels` hub loader (needs checking: hub kernels are mostly CUDA builds).
- Extras: deepspeed (ROCm/DeepSpeed + upstream ROCm), mamba-ssm 1.2.0.post1 + causal_conv1d (causal-conv1d disposition already-supported; mamba-ssm has a HIP path upstream), vllm (ROCm upstream), fbgemm-gpu-genai (ROCm/FBGEMM), flash-attn-4[cu13] and quack-kernels[cu13] (CUDA-only CuTe-DSL; ROCm users use flash-attn 2 via ROCm/flash-attention or SDPA/AOTriton).

### Recommendation: take up (fork), validate lane
Work: install axolotl on ROCm torch and fix packaging (ROCm-safe extras/pins, ROCm Dockerfile alongside the uv images); run the GPU test suite and representative e2e fine-tunes (LoRA, QLoRA via bitsandbytes, FSDP/DeepSpeed multi-GPU, Liger and Triton LoRA kernels, flash-attn 2 / SDPA) on wave64 (MI300/MI355) and wave32; add the `torch.version.hip` guard to `marlin_w4a16_available()`; test whether `quantize_moe_experts` can drop its CUDA-only restriction; refresh `docs/amd_hpc.qmd` (ROCm 5.7.1) and installation docs. Marlin `.cu` itself is not ported. Upstream is active and has no competing AMD effort; stale #1550 should be superseded or credited, not duplicated blindly.
- Windows gate: axolotl is Linux-first (multi-GPU training stack, deepspeed/xformers); whether a Windows run is meaningful or the gate should be waived is a planner finding and a person's decision.
