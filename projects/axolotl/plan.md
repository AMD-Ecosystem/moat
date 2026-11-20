# axolotl port plan

Planned 2026-11-20 on linux-gfx942 against upstream `main` at `de8549a4406327ab5311f7187a8afaa232645487`. Four decisions are open for jeff before a porter starts; they are listed at the top of the 2026-11-20 planner section in `notes.md` and repeated under "Open questions" below. This plan states the recommended answer for each and marks every place that depends on it.

## Project

- Name: axolotl (LLM fine-tuning framework on transformers / PEFT / TRL / accelerate)
- Upstream: https://github.com/axolotl-ai-cloud/axolotl, default branch `main`, Apache-2.0
- Fork: https://github.com/AMD-Ecosystem/axolotl, port branch `moat-port`
- Size: 825 Python files under `src/`, 718 test files (123 under `tests/e2e`), 42 modules with `@triton.jit` kernels, one vendored optional CUDA extension (Marlin, 11 files).

## Existing AMD support

Classification: **improvable** (docs-level, community, stale).

- `README.md:113` and `docs/installation.qmd:16` say "NVIDIA GPU ... or AMD GPU"; `docs/amd_hpc.qmd` is an HPC install guide pinned to `module load rocm/5.7.1`, the `rocm5.7` torch index, and a hipify patch for ROCmSoftwarePlatform/flash-attention on gfx90a. Every version in it is obsolete (axolotl now requires Python >= 3.12 and torch >= 2.13, < 2.16).
- No ROCm image (`docker/` has only CUDA uv images), no ROCm CI lane, no ROCm extras, no ROCm-aware device gates beyond two `torch.version.hip` early-outs in autotune helpers (`src/axolotl/kernels/gemma4_rope_autotune.py:61`, `src/axolotl/integrations/kernels/libs/dsv4/autotune_profiles.py:73`).
- https://github.com/axolotl-ai-cloud/axolotl/pull/1550 ("Add experimental install guide for ROCm", community, open and stale since 2024-04): community, not authoritative. A docs refresh supersedes it; credit it in the PR body rather than ignore it.
- No AMD-official axolotl effort was found (intake could not search the `amd`/`AMD-AGI` orgs because of SAML SSO; still open).

Decision: take it up as a **validation-lane port**. There is almost nothing to translate; the value is a working, documented ROCm install, ROCm-correct device gating, and recorded GPU evidence for the training paths AMD users rely on.

Delivery-vehicle facts (they feed open question Q3):

- Upstream merges platform support in-tree (it already carries XPU support in `tests/accelerator.py` and AMD wording in the docs); it does not link platform forks. An upstream PR is the normal vehicle.
- Upstream has an AI-assisted contribution policy (`AGENTS.md`, `CLAUDE.md`, `.github/CONTRIBUTING.md` "AI-Assisted Contributions"): a PR must state the concrete use case or maintainer request behind it, report what was actually run, disclose the AI tools used, and a human must have reviewed the complete diff before the PR is opened (draft PRs included). MOAT's fork review PR plus jeff's `/moat approve` is that human review; the upstream PR body must carry the motivation (AMD Instinct/Radeon users fine-tuning with axolotl on ROCm) and the real test results. A bare batch of unrelated fixes would read as the "contribution farming" the policy refuses, so keep the PR one coherent "ROCm support" change.

Port versus AMD-native rewrite: the only NVIDIA-tuned compiled kernels are the vendored Marlin W4A16 MoE GEMM (PTX `mma.sync`/`ldmatrix`) and the CuTe-DSL/CUTLASS/DeepGEMM backends (FlashAttention-4, quack, SonicMoE, `cutlass_fp4`). None has a mechanical HIP path and all are optional with a selector that falls back to Triton or torch. This plan does neither a port nor a rewrite of them (recommended answer to Q1); an AMD-native NVFP4/MXFP4 MoE GEMM would be a separate project.

## Build classification

`ext_type`: **python-triton** -- a pure-Python setuptools package (`pyproject.toml:1-3`, `setuptools` + `setuptools_scm`, no `ext_modules`, no `setup.py`) whose GPU code is Triton JIT kernels plus PyTorch ops. Nothing is compiled at install time. The one compiled path is `torch.utils.cpp_extension.load` at runtime for the vendored Marlin sources (`src/axolotl/integrations/kernels/libs/scattermoe_lora/marlin_w4a16/__init__.py:67-105`), which is optional.

GPU dependencies are pinned in `pyproject.toml:17-99`: `torch>=2.13.0,<2.16.0`, `triton>=3.4.0`, `xformers>=0.0.33.post2,<0.0.36`, `bitsandbytes==0.50.2`, `liger-kernel==0.8.5`, `torchao==0.18.0`, `fla-core`/`flash-linear-attention==0.6.0`, `kernels>=0.17.2,<0.18.0`, `nvidia-ml-py==12.560.30`; extras `deepspeed`, `mamba-ssm`, `vllm`, `fbgemm-gpu`, `blackwell` (`flash-attn-4[cu13]`, `quack-kernels[cu13]`), `fla` (tilelang).

## Port strategy

Neither A nor B in the hipify sense. It is Strategy B's environment (ROCm PyTorch, torch's own HIP masquerade as `torch.cuda`) with no extension to hipify:

1. Build a ROCm environment that satisfies `pyproject.toml` without pip replacing ROCm torch or ROCm Triton with CUDA wheels (see Risks R1).
2. Fix the small set of places where axolotl decides by CUDA compute capability or imports a CUDA-only Triton extern, so ROCm takes the portable path (R2-R5).
3. Validate the real GPU test lanes and representative end-to-end fine-tunes on each required platform.
4. Deliver the install recipe as a ROCm Dockerfile beside the CUDA uv images and refreshed docs (scope per Q3).

Every code change must be a no-op on CUDA: gate on `torch.version.hip` (or a helper that reads it), never edit a CUDA branch's behavior.

## CUDA surface inventory

### Compiled CUDA (surface.json `cuda.files`, 11 files)

`src/axolotl/integrations/kernels/libs/scattermoe_lora/marlin_w4a16/_csrc/**`: `ops_standalone.cu`, `repack_standalone.cu`, `sm80_kernel_bfloat16_fe2m1f_bfloat16.cu`, `sm80_kernel_bfloat16_fe4m3fn_bfloat16.cu`, `marlin.cuh`, `marlin_dtypes.cuh`, `marlin_mma.h`, `dequant.h`, `kernel.h`, `marlin_template.h`, `launcher_body.inc` (36 kernels, 3 launches, 5 warp intrinsics). Vendored from vLLM (`_csrc/PROVENANCE.md`); PTX tensor-core code with no HIP equivalent, and vLLM itself does not run Marlin MoE on ROCm. Disposition: **scoped out**; the port adds a HIP guard so the selector never tries to JIT-hipify it (change C1).

### Triton kernels (42 modules)

| area | modules | ROCm expectation |
|---|---|---|
| core LoRA/MLP kernels | `src/axolotl/kernels/{lora,swiglu,geglu,dora,relu2,quantize,rms_norm_gated,gemma4_fused_rope,glm4_moe_lite}.py` | portable; upstream docs (`docs/lora_optims.qmd:98`) already claim AMD. `rms_norm_gated.py`/`gemma4_fused_rope.py` already prefer the backend-neutral `triton.language.extra.libdevice`. In scope. |
| ScatterMoE-LoRA (bf16 path) | `integrations/kernels/libs/scattermoe_lora/{kernels/*,*.py}` | portable except `swiglu_epilogue.py:14` (R4). In scope for bf16 / bnb experts. |
| ScatterMoE FP4/FP8 quant paths | `scattermoe_lora/cutlass_fp4/*`, `marlin_w4a16/*`, `dequant_grouped.py` (DeepGEMM), `nvfp4_*` | NVIDIA hardware formats (NVFP4, block-fp8 via DeepGEMM/CUTLASS). Scoped out (Q1). |
| SonicMoE | `integrations/kernels/libs/sonicmoe/*` | needs `sonic-moe`/quack (CuTe DSL). Scoped out (Q1). |
| DSv4 / GLM-DSA / Mistral-Large4 attention kernels | `integrations/kernels/libs/{dsv4,glm_dsa,mistral_large4}/*` | Triton, but tuned with Hopper/Blackwell smem and tmem assumptions (`dsv4/attention.py:63`); model-specific, large models. Run their unit tests where they exist; not an e2e target. |
| attention monkeypatches | `monkeypatch/attention/*` (2 Triton modules) | FA4 path scoped out; the rest in scope. |
| misc | `utils/optimizers/*`, `core/trainers/ebft/*` | in scope via their tests. |

### PyTorch-level device usage (no CUDA symbols, but device-dependent)

- Capability-keyed decisions (R2): `cli/config.py:361-391` (`compute_supports_fp8` = cc >= (9,0); `compute_capability` string `"sm_<major><minor>"`), consumed by `utils/schemas/config.py:2071` (fp8), `:2092` (sdpa+packing warning), `:2111` (SageAttention allow-list), `:2130` (fp8 attention preflight), `:2158` (`quantize_moe_experts` "requires CUDA" check, which never fires on ROCm because the string still starts with `sm_`). Elsewhere: `monkeypatch/attention/flash_attn_4.py:137,213`, `core/trainers/async_grpo/trl_compat.py:247`, `integrations/kernels/plugin.py:101`, `libs/sonicmoe/{nvfp4_lora.py:190,nongated.py:22,fp8_bwd.py:63,fp4_cute.py:57}`, `libs/scattermoe_lora/{experts.py:842,887,1263,dequant_grouped.py:54,73}`, `libs/dsv4/attention.py:63`, `libs/glm_dsa/attention_mla_absorb.py:205`, `kernels/grouped_mode.py:389`.
- `torch._grouped_mm` / `_scaled_grouped_mm`: 46 call sites, mostly behind cc >= 9 gates (MoE paths).
- `utils/bench.py:12` uses `pynvml` with a try/except fallback (memory reporting only; `nvidia-ml-py` is a core dependency). Reports nothing useful on ROCm; cosmetic.
- `utils/cuda13.py`: CUDA 13 uv image helper, no-op when `nvidia.cu13` is absent.
- Hugging Face hub kernels loaded with `kernels.get_kernel`: `kernels-community/{flash-attn2,flash-attn3,flash-attn4,causal-conv1d,mamba-ssm,sage-attention,sonic-moe,deep-gemm,fla}`. Checked 2026-11-20: the `build/` trees of flash-attn2, flash-attn3, causal-conv1d, mamba-ssm and sage-attention carry CUDA, XPU, CPU and Windows-CUDA variants and **no ROCm variant**; flash-attn2 declares `supportedDriverFamilies: ["cuda","xpu","cpu"]`. On ROCm the hub fallback cannot supply these, so flash attention needs the `flash_attn` package from ROCm/flash-attention (transformers' `is_flash_attn_2_available` accepts it, as does `tests/e2e/utils.py:48`), and Mamba needs `mamba-ssm`/`causal-conv1d` pip builds with their HIP paths.
- Distributed: NCCL through torch.distributed (RCCL on ROCm), DeepSpeed, FSDP2, Ray; no direct NCCL calls.

No textures, no driver API, no NVRTC, no CUDA graphs authored by axolotl, no inline PTX outside Marlin. `triton.language.extra.cuda.libdevice` is the one NVIDIA-only Triton extern (R4).

## Risks

- **R1 Packaging replaces ROCm torch/Triton.** `pip install axolotl` resolves `torch` from PyPI (CUDA), and the unconditional `triton>=3.4.0` and `xformers` pins can swap ROCm `pytorch-triton-rocm`/ROCm xformers for CUDA wheels; Python markers cannot distinguish ROCm. Install ROCm torch first from the `download.pytorch.org/whl/rocm*` index that carries a torch in `[2.13, 2.16)`, then axolotl with a constraints file pinning `torch`, `triton`/`pytorch-triton-rocm`, `xformers`, `torchvision`, and verify `torch.version.hip` after every install (cuda-to-rocm strategy-b-torch.md, "The same trap fires on ANY pip install"). Requires Python >= 3.12; build a job venv for it. `xformers` ROCm wheels exist on the same index; if none matches the torch, decide (with evidence) between building it and an install that skips it, and record which.
- **R2 Compute-capability collision.** ROCm reports gfx90a as (9,0) like Hopper, gfx942 (9,4), gfx950 (9,5), gfx1100 (11,0) and gfx1151 (11,5) inside the Blackwell `in (10, 11)` sets, and gfx1201 (12,0) like sm120 (cuda-to-rocm platform-limits.md, "AMD compute-capability values COLLIDE"). Classify every gate: (a) gates followed by an import of a CUDA-only package (`flash_attn.cute`, `deep_gemm`, `sonic-moe`, `quack`, `cutlass`) fail closed on ROCm and need nothing beyond confirming the import failure is caught; (b) gates that choose a code path from capability alone must get a ROCm-aware predicate. Known (b) cases: `compute_supports_fp8` (true on gfx90a/gfx1100, which lack FP8), the `sm_XY` string fed to the schema validators, `dsv4/attention.py:63` (tile cap, perf only), `experts.py:887,1263` (NVFP4, out of scope but must not mis-raise for bf16 experts), and the `torch._grouped_mm` gates (`nvfp4_lora.py:190`, `nongated.py:22`) where torch's ROCm grouped-mm support differs by arch. Fix only those a test or e2e run on ROCm actually reaches, plus the schema inputs; select by `gcnArchName` (stripped of feature suffixes), not by major/minor.
- **R3 Import-time CUDA-only imports.** A top-level `import cutlass` / `cuda.bindings` / `nvidia.*` in a module that axolotl imports eagerly would break `import axolotl` on ROCm. First porter step: `python -c "import axolotl; from axolotl.cli.main import main"` and import every integration module the e2e tests load.
- **R4 NVIDIA libdevice in Triton.** `scattermoe_lora/swiglu_epilogue.py:14` imports `triton.language.extra.cuda.libdevice` unconditionally and calls `libdevice.tanh`; on the AMD backend that extern does not resolve. Switch to the backend-neutral `triton.language.extra.libdevice` with the same fallback ladder `rms_norm_gated.py:27-31` already uses.
- **R5 Marlin JIT on ROCm.** `marlin_w4a16_available()` (`marlin_w4a16/__init__.py:43-55`) checks only `torch.cuda.is_available()`, cc major >= 8 and `CUDA_HOME is not None`; on ROCm torch `CUDA_HOME` can resolve, and gfx942 passes the cc check, so the selector would JIT-hipify PTX and fail mid-forward. Return False under `torch.version.hip`.
- **R6 Hub kernels absent for ROCm** (see inventory). Flash attention, Mamba, SageAttention and FA3 must come from pip packages or be reported unavailable. The validator must confirm `attn_implementation: flash_attention_2` resolves to the ROCm `flash_attn` package, and that with it absent the validator's error is clear rather than a hub download failure. SageAttention and FA3/FA4 are scoped out on ROCm.
- **R7 Wave32 and Triton tuning.** Triton handles wavefront width, but autotune configs written for NVIDIA (num_warps 8, large smem tiles in `grouped_gram.py:37`) can exceed LDS on RDNA (64 KiB) and slow first-step compiles badly (cuda-to-rocm strategy-b-torch.md, Triton cold compile). Set a persistent `TRITON_CACHE_DIR`; treat `OutOfResources` on RDNA as a config-prune fix in the kernel's prune hook, not a skip.
- **R8 Numerics.** bf16/fp8 GEMM and attention differ numerically on MI300 (fnuz fp8 on gfx942, OCP fp8 on gfx950/gfx12). Loss-curve and allclose tests that fail by small margins get ROCm-conditional tolerances with the measured magnitude recorded, never a looser CUDA tolerance.
- **R9 Multi-GPU.** The FSDP2/DeepSpeed/TP/context-parallel e2e lanes need >= 2 GPUs (most use 2). A shared host must pin an idle pair. RCCL replaces NCCL transparently; `deepspeed-kernels` is CUDA-only and is skipped.
- **R10 Tests that query CUDA by name.** `tests/conftest.py:126` auto-marks any skip whose reason mentions "cuda"/"gpu" as `gpu`, and `tests/cuda_gates.py` keys skips on capability, so on gfx942 tests meant for Hopper (`gpu_blackwell`) are not skipped by capability. Run the `gpu_blackwell` lane once to triage; ROCm skips go in only per test with a reason naming the missing CUDA-only dependency.

## File-by-file change list

All changes are guarded so the CUDA behavior is byte-for-byte the same. Paths are relative to the fork root.

- C1 `src/axolotl/integrations/kernels/libs/scattermoe_lora/marlin_w4a16/__init__.py`: `marlin_w4a16_available()` returns False when `torch.version.hip` is set. (R5)
- C2 `src/axolotl/cli/config.py`: `compute_supports_fp8()` and `gpu_capabilities()` answer correctly on ROCm: FP8 only on FP8-capable gfx targets (gfx942, gfx950, gfx12xx), and a `compute_capability` value that the validators can tell is ROCm (for example the stripped `gcnArchName`), so no `sm_XY` look-alike reaches them. Then make each consumer in `src/axolotl/utils/schemas/config.py` (`:2071`, `:2092`, `:2111`, `:2130`, `:2158`) say the right thing for ROCm. For `quantize_moe_experts`, measure first (it is bitsandbytes-based and bitsandbytes supports ROCm): if it trains on ROCm, drop the "not compatible with ROCm" wording in `config.py:1023`, `config_options.py:2887` and `docs/expert_quantization.qmd:142`; if it fails, make the check actually reject ROCm with the observed reason. (R2)
- C3 `src/axolotl/integrations/kernels/libs/scattermoe_lora/swiglu_epilogue.py`: backend-neutral libdevice import. (R4)
- C4 capability gates reached on ROCm (R2 class b): only those a ROCm test or e2e run reaches, each with a `torch.version.hip` branch or a small shared helper (one place, for example next to `compute_supports_fp8`), not scattered checks. Record each gate's classification in notes.
- C5 tests: ROCm-conditional tolerances (R8) and per-test skips naming a missing CUDA-only dependency (R10), no wholesale skips; `tests/cuda_gates.py` may gain a `rocm` predicate if several tests need it.
- C6 `docker/Dockerfile-rocm-uv` (or the name the uv family suggests; scope per Q3): ROCm base image or ROCm torch wheel index, Python 3.12, the constraints from R1, ROCm `flash_attn`, bitsandbytes, and the axolotl install, mirroring `docker/Dockerfile-uv-base` + `Dockerfile-uv`. Built and used by the validator to run the tests, so it is proven, not decorative. No CI workflow and no fork GitHub Actions.
- C7 docs: rewrite `docs/amd_hpc.qmd` for the current ROCm and torch (no `module load rocm/5.7.1`, no hipify patch), add a ROCm section to `docs/installation.qmd`, note in `docs/attention.qmd` that on ROCm flash attention comes from ROCm/flash-attention (the hub kernel has no ROCm build), and list ROCm in `docs/support-matrix.qmd` with what was validated and what is CUDA-only. README "Running" examples carry no device-selection variables.
- C8 `pyproject.toml`: only if a marker-expressible change helps (it usually cannot distinguish ROCm); otherwise leave the dependency list alone and carry the ROCm pins in the Dockerfile and docs as a constraints file.

Not changed: the vendored Marlin `.cu` sources, the CuTe-DSL/CUTLASS/DeepGEMM/SonicMoE/FA4 modules, CUDA Dockerfiles, CI workflows, `packages/tree-recurrence` (no GPU code of its own; it loads hub kernels and is exercised through axolotl's tests).

## Build commands

No compile step. On the executing host (Linux; pin GPUs with `HIP_VISIBLE_DEVICES` when the host is shared):

```bash
# Python >= 3.12 job venv
uv venv --python 3.12 .venv && . .venv/bin/activate
# ROCm torch first; pick the newest rocm index that has a torch in [2.13, 2.16)
uv pip install --index-url https://download.pytorch.org/whl/rocm<X.Y> "torch>=2.13,<2.16" torchvision
python -c "import torch; print(torch.__version__, torch.version.hip)"
pip freeze | grep -Ei '^(torch|torchvision|pytorch-triton-rocm|triton)==' > constraints-rocm.txt
# axolotl from the fork checkout with the test group, holding ROCm torch/Triton
uv pip install -c constraints-rocm.txt -e '.[deepspeed]' --group test
python -c "import torch; print(torch.__version__, torch.version.hip)"   # must still be the ROCm build
# flash attention for ROCm (CK backend on CDNA, Triton backend where CK lacks the arch)
pip install --no-build-isolation git+https://github.com/ROCm/flash-attention.git   # record the tag/commit used
python -c "import axolotl, axolotl.cli.main"
```

The porter records the exact index, versions and any package that had to be built from source in `notes.md`, then turns the same steps into C6 so the validator runs the tests inside it.

## Test plan

CPU regression set (must not regress; run on every platform before GPU lanes):

```bash
pytest -m 'not gpu and not slow and not distributed_cpu and not nf4_distributed' --ignore=tests/e2e --ignore=tests/integrations/kernels tests/
```

GPU lanes, adapted from `cicd/cicd.sh` and `cicd/multigpu.sh` (drop `--cov`, keep the markers):

```bash
export TRITON_CACHE_DIR=$PWD/.triton-cache
# 1. single-GPU unit tests
pytest -v -n4 --maxfail=50 -m "gpu and not slow and not nf4_distributed and not gpu_blackwell" \
  --ignore=tests/e2e/ --ignore=tests/integrations/ --ignore=tests/patched/ --ignore=tests/cli tests/
# 2. GPU kernel tests (Triton LoRA/MLP/MoE kernels)
pytest -v -m gpu tests/kernels tests/integrations/kernels tests/e2e/kernels
# 3. e2e single GPU (patched, solo, integrations, then the rest serially)
pytest -v tests/e2e/patched
pytest -v -n1 tests/e2e/solo
pytest -v tests/e2e/integrations
pytest -v -m "gpu and not gpu_blackwell" --ignore=tests/e2e/kernels/ --ignore=tests/integrations/kernels/ tests/integrations/
pytest -v --ignore=tests/e2e/multigpu --ignore=tests/e2e/patched --ignore=tests/e2e/solo --ignore=tests/e2e/integrations --ignore=tests/e2e/kernels tests/e2e
# 4. multi-GPU (>= 2 GPUs; wave64 Instinct host)
pytest -v -n1 -m gpu tests/monkeypatch/test_fsdp2_bnb_checkpoint.py tests/integrations/test_expert_parallel_bnb.py
pytest -v -n1 --ignore=tests/e2e/multigpu/solo/ --ignore=tests/e2e/multigpu/patched/ tests/e2e/multigpu/
pytest -v -n1 -m "not nvfp4 and not slow" tests/e2e/multigpu/solo
pytest -v -n1 tests/e2e/multigpu/patched/
# 5. triage only: what the Hopper/Blackwell lane does on ROCm
pytest -v -m gpu_blackwell tests/ --ignore=tests/e2e/multigpu
```

The bar: lanes 1-3 on every required platform; lane 4 on a wave64 platform with >= 2 GPUs (recommended answer to Q4); lane 5 is classification evidence, not a gate. Every failure is either fixed, given a ROCm-conditional tolerance with the measured magnitude, or skipped per test with a reason that names a CUDA-only dependency; record pass/fail/skip counts per lane per platform, against the CUDA lane's counts where upstream publishes them.

Representative end-to-end fine-tunes, with the configs recorded in notes (small models so they fit any GPU; e2e tests use SmolLM2-135M and similar):

- full fine-tune and LoRA SFT with `flash_attention: true` (ROCm flash_attn) and with `sdpa`, sample packing on;
- QLoRA via bitsandbytes 4-bit;
- Liger kernels plugin and the Triton LoRA kernels (`lora_mlp_kernel`, `lora_qkv_kernel`, `lora_o_kernel`);
- cut-cross-entropy plugin;
- DPO (TRL) single GPU;
- a ScatterMoE-LoRA bf16 MoE run (small Qwen3-MoE/OLMoE) to cover the MoE Triton path;
- `merge-lora` and `inference` on the result;
- multi-GPU: FSDP2 LoRA and DeepSpeed ZeRO-2/3 on 2 GPUs.

Loss must decrease and match a CPU or reference run's first steps to within recorded tolerance where the test asserts it.

## Open questions

These need jeff's ruling before a porter starts; they are repeated at the top of the 2026-11-20 planner section in `notes.md`.

- **Q1 Scope cut.** Recommended: validate and fix the portable surface (core training, Triton kernels, Liger, CCE, bitsandbytes, flash attention via ROCm/flash-attention, SDPA, DeepSpeed/FSDP2, ScatterMoE bf16) and scope out, as CUDA-only optional backends with existing fallbacks: the vendored Marlin `.cu` (guard only), FlashAttention-4/quack/SonicMoE/`cutlass_fp4` (CuTe DSL), DeepGEMM, the NVFP4/MXFP4 Blackwell paths, SageAttention, FA3, and the `blackwell`/`fla` (tilelang) extras. Alternatives: (b) also port the Marlin W4A16 MoE GEMM to an AMD-native kernel (a large separate effort; no upstream AMD Marlin MoE exists); (c) a narrower docs-and-packaging-only port with no code changes.
- **Q2 Windows gate.** Recommended: request a waiver; axolotl's stack (DeepSpeed, multi-GPU FSDP, xformers, bitsandbytes ROCm, vLLM) is Linux-first and upstream documents no Windows support. Alternative: attempt a single-GPU LoRA run on the Windows platform with the subset that installs there. Only a maintainer may approve the waiver.
- **Q3 Delivery.** Recommended: one upstream PR "ROCm support" containing C1-C5 + C7, with C6 (ROCm Dockerfile) included because it is the tested install recipe; the PR body states the AMD-user motivation and real results as upstream's AI policy requires, and credits #1550. Alternatives: (b) code + docs only upstream, the Dockerfile kept on the fork; (c) docs only upstream.
- **Q4 Validation bar for multi-GPU.** Recommended: multi-GPU lanes (FSDP2, DeepSpeed) on one wave64 Instinct platform only; wave32 and Windows platforms run lanes 1-3. Alternative: require a 2-GPU run on every Linux platform that has two GPUs.
