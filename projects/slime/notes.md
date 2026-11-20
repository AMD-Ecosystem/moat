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

## Planner (2026-11-20, linux-gfx942 fleet job, no GPU used)

### DECISIONS NEEDED FROM JEFF BEFORE A PORTER STARTS

1. **GLM-5 alignment scope (partial-port cut).** `glm5_router_gemm.cuh` is reachable only through slime's GLM-5 deterministic train/rollout alignment (`slime/backends/megatron_utils/alignment/`, 4.9k lines). That path also needs DeepGEMM (no ROCm build), DeepEP low-latency and slime's patched batch-invariant SGLang (`docker/patch/*/sglang-deterministic.patch`), and its test needs 8 GPUs and GLM-5 weights. Options:
   - A: HIP-port the router kernel and add a kernel-level test; defer the rest.
   - B (recommended): A, and register the alignment path (DeepGEMM to AITER/hipBLASLt, DeepEP-LL to ROCm/DeepEP) as a deferral for the AMD team.
   - C: port the whole alignment path (a multi-week GEMM reimplementation that overlaps the AMD team).
   - D: scope out the router kernel too, and only validate the existing ROCm path.
2. **Gates.** `wave32` is not waivable (`config/arches.toml`), and the RL stack (sgl-kernel ROCm, Megatron/TE ROCm) is CDNA-only. Recommended: satisfy wave32 with the two kernel-level tests only and say so. The alternative is a config/policy change. `windows`: recommend a waiver (no Ray/Megatron/SGLang on Windows); it needs a maintainer's approval.
3. **Delivery vehicle.** AMD's own THUDM/slime#2506 (jun-amd, OPEN/BLOCKED, updated 2026-11-12) and #2369 (LZ-QWQ, OPEN/DIRTY) cover the Dockerfile, tutorials and AMD script. Options:
   - a (recommended, after #2506 merges): one small upstream PR with only the setup.py arch fix, the router-kernel HIP branch and the kernel tests (disjoint from #2506).
   - b: no MOAT PR; hand the findings to jun-amd.
   - c: a, plus validation feedback posted on #2506 (a GitHub write; the text needs approval).

### Findings
- Fork `main` at `79aa41a` is a pristine mirror; there is no `moat-port` branch yet.
- Upstream ROCm on `main` is stale: `Dockerfile.rocm` is ROCm 6.3.4 / SGLang 0.4.9-0.5.0rc0, and `amd_patch/latest` is identical to `amd_patch/sglv0.5.0rc0` (untouched since 2026-08-28). CUDA tracks SGLang v0.5.21 and Megatron core 0.20.
- **Defect**: `kernels/int4_qat/setup.py` does `os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx950")` on ROCm. A plain build on gfx942 or RDNA gives gfx950-only code objects and overrides torch's own arch default.
- `fake_int4_quant_cuda.cu` is already HIP-clean (#2274): its width-32 maskless `__shfl_xor` makes it correct on both wavefront widths.
- `glm5_router_gemm.cuh` will not compile under hipcc. SGLang v0.5.21 `load_jit` uses hipcc on ROCm (`kernels/jit/utils/compile/toolchain.py`) without hipify, and SGLang's ROCm JIT kernels are native HIP sources. The blockers are `<cuda_bf16.h>`/`__nv_bfloat16`, `__shfl_xor_sync(0xffffffff, ...)`, and probably `kDLCUDA` device checks against ROCm tensors (`kDLROCM`).
- Build classification: torch-extension (`CUDAExtension` in the int4_qat `setup.py`), plus one runtime-JIT kernel. `surface.py` found 0 components (it does not parse this setup.py), so I added 11 components by hand.
- Validation environment is the biggest risk: it needs the #2506 stack (Primus v26.5 base, SGLang v0.5.21 + patches, sgl-kernel `setup_rocm.py`, Megatron core_v0.19.2, ROCm TE, torch_memory_saver HIP) built into a job venv or provided as an image, and E2E tests need a reserved 4-8 GPU host.

Full design: `plan.md`. Accounting: `surface.json` (nothing covered or scoped out yet; that waits for the rulings above).
