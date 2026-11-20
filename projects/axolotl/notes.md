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

## Planning (2026-11-20, planner, linux-gfx942)

### Decisions needed from jeff before a porter starts

Nobody could answer these during the planning job, so nothing below is assumed. Each lists the options; the plan is written against the recommended one and marks what changes otherwise.

1. **Q1 Scope cut (partial port).** Recommended (a): validate and fix the portable surface (core training, Triton kernels, Liger, cut-cross-entropy, bitsandbytes QLoRA, flash attention via the ROCm/flash-attention package, SDPA, DeepSpeed/FSDP2, ScatterMoE-LoRA bf16) and scope out the CUDA-only optional backends that already have fallbacks: the vendored Marlin `.cu` (HIP guard only), FlashAttention-4 / quack / SonicMoE / `cutlass_fp4` (CuTe DSL), DeepGEMM, the NVFP4/MXFP4 Blackwell paths, SageAttention, FA3, and the `blackwell` and `fla` (tilelang) extras. (b) Also write an AMD-native NVFP4/W4A16 MoE GEMM to replace Marlin (large, separate effort; no AMD Marlin MoE exists upstream or in vLLM). (c) Docs and packaging only, no code changes.
2. **Q2 Windows gate.** Recommended: ask for a waiver of the `windows` gate (Linux-first stack: DeepSpeed, multi-GPU FSDP, xformers, vLLM; upstream documents no Windows support). Alternative: attempt a single-GPU LoRA run on the Windows platform with whatever installs there. A waiver is a maintainer's approval; this is only the suggestion.
3. **Q3 Delivery.** Recommended (a): one upstream "ROCm support" PR with the guards and gating fixes, test adjustments, docs, and a ROCm Dockerfile (the tested install recipe), crediting stale community PR #1550. (b) Code and docs upstream, Dockerfile kept on the fork. (c) Docs only upstream. Whatever the choice, upstream's AI-contribution policy (`AGENTS.md`, `.github/CONTRIBUTING.md` "AI-Assisted Contributions") requires the PR body to state the real use case (AMD users fine-tuning on ROCm), what was actually run, and the AI tools used, and requires a human to have reviewed the full diff before the PR (even a draft) is opened; the fork review PR approval must be that review.
4. **Q4 Multi-GPU bar.** Recommended: FSDP2/DeepSpeed multi-GPU lanes on one wave64 Instinct platform only; wave32 and Windows platforms run the single-GPU lanes. Alternative: a 2-GPU run on every Linux platform that has two GPUs.

### Findings (detail in plan.md)

- Upstream `main` at `de8549a4406327ab5311f7187a8afaa232645487` (2026-11-20). `ext_type` set to `python-triton`: pure-Python setuptools package, 42 modules with Triton kernels, the only compiled code is the optional runtime-JIT Marlin extension (11 files).
- Hugging Face hub kernels have no ROCm builds: `kernels-community/{flash-attn2,flash-attn3,causal-conv1d,mamba-ssm,sage-attention}` `build/` trees list CUDA, XPU, CPU and Windows-CUDA variants only (queried `https://huggingface.co/api/models/kernels-community/<name>/tree/main/build`, 2026-11-20); flash-attn2's kernel record says `supportedDriverFamilies: ["cuda","xpu","cpu"]`. On ROCm, flash attention must come from the `flash_attn` package (ROCm/flash-attention), which transformers and `tests/e2e/utils.py:48` already accept.
- Compute-capability collisions: `cli/config.py:361` treats cc >= (9,0) as FP8-capable (wrong for gfx90a, gfx1100) and builds an `sm_<major><minor>` string that the schema validators read as an NVIDIA arch (gfx942 becomes `sm_94`, "Hopper or newer"). The `quantize_moe_experts` "requires CUDA, not ROCm" check (`utils/schemas/config.py:2158`) therefore never fires on ROCm: the restriction exists only in docs and messages. Whether the feature works on ROCm (bitsandbytes-based) is a porter measurement.
- `scattermoe_lora/swiglu_epilogue.py:14` imports `triton.language.extra.cuda.libdevice` unconditionally (NVIDIA extern); other kernels already use the backend-neutral import.
- `marlin_w4a16_available()` has no `torch.version.hip` check (intake's finding, confirmed).
- Environment for the porter: axolotl requires Python >= 3.12 and torch >= 2.13, < 2.16, and pins `triton` and `xformers` unconditionally; install ROCm torch first and hold it with a constraints file (plan.md R1 and Build commands).
- Upstream test lanes: `cicd/cicd.sh` (single GPU), `cicd/multigpu.sh` (2 GPUs), `cicd/cicd_blackwell.sh` (`gpu_blackwell`, B200), `cicd/nvfp4.sh` (SM100+). CPU suite command from upstream `CLAUDE.md`. The plan's Test plan adapts them.
- Generated `surface.json`: the census found only the Marlin files and no build-system components (no CMake/setup.py), so the planner added 31 components (library, Triton kernel groups, plugins, gating, dependency paths, test lanes, packaging, Docker, docs, examples) and 8 scoped-out entries, each marked pending the Q1 ruling. `python3 utils/surface.py check axolotl` passes.
- No CUDA-path compile check is needed for the planned changes (Python only, no `.cu` edits); CUDA safety rests on every change being under a `torch.version.hip` branch and on the CPU suite.
