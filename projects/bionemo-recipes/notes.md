# bionemo-recipes notes

## Intake screen (2026-11-12)

Validation project (lane `validate`, seed `amd-gap-matrix`, NVIDIA-owned upstream), taken up per jeff's 2026-11-12 ruling on the third-party "AMD AI software gap" list.

- Upstream: https://github.com/NVIDIA-BioNeMo/bionemo-recipes (the old `NVIDIA/bionemo-framework` name redirects here), default branch `main`, very active (pushed 2026-11-12), not archived.
- Licence: GitHub reports NONE because the licence lives at `LICENSE/license.txt`. Read it: byte-identical to the Apache License 2.0 text from apache.org (same md5). Recorded as Apache-2.0 (tier 1). Per-model files: `models/amplify` MIT (chandar-lab + NVIDIA), `models/esm2`, `models/geneformer`, recipe `LICENSE` files Apache-2.0. `LICENSE/third_party.txt` lists third-party notices. No submodules.
- `utils/licenses.py scan-nvidia` on a depth-1 clone: one hit, `docs/docs/models/evo2.md`. Read it: a model-card paragraph stating the Evo2 NIM container is governed by the NVIDIA Software License Agreement and the model by the NVIDIA Open Model License. Documentation prose about a separate product, not a licence on any source file; needs a person's sign-off all the same.
- CUDA surface: none of its own (no `.cu`). Everything goes through Transformer Engine (`transformer_engine[pytorch]`), megatron-FSDP, PyTorch FSDP2, and for the Megatron recipes Megatron-Core / Megatron-Bridge / NeMo / NeMo-RL. Every recipe is a self-contained directory with its own Dockerfile (`nvcr.io/nvidia/pytorch:26.09-py3`), requirements and pytest suite (275 test files across 25 `tests/` directories).
- Recipe-by-recipe dependency picture:
  - TE + PyTorch only (most likely to work): `models/{esm2,amplify,llama3,mixtral,qwen,codonfm,geneformer}`, `recipes/{esm2_native_te,esm2_accelerate_te,esm2_peft_te,llama3_native_te,opengenome2_llama_native_te,mixtral_native_te,codonfm_native_te,codonfm_ptl_te,geneformer_native_te_mfsdp_fp8,vit}`. Also need `nvdlfw_inspect` (pure Python, from NVIDIA/nvidia-dlfw-inspect) and `megatron-fsdp` (pure Python).
  - Megatron/NeMo stack (likely gaps): `recipes/{evo2_megatron,eden_megatron,evo2_phage_gen}` need `subquadratic-ops-torch-cu13` (NVIDIA binary-only Hyena kernels), `flash-attn-4[cu13]` (CuTe-DSL, Hopper/Blackwell only), `causal-conv1d`, `mamba-ssm`, `nvidia-modelopt`, Megatron-Bridge, NeMo. `recipes/nemotron-stitch` needs NeMo-RL + vLLM. `recipes/vllm_inference/esm2` needs vLLM.
  - Precision features: FP8 (TE ROCm: gfx942 FNUZ, gfx950 OCP), MXFP8 (gfx950 only), NVFP4 (no AMD equivalent; tests will need skipping by capability).

## Duplicate effort

- No upstream PR or issue about ROCm/AMD (searched 2026-11-12; the hits are dependabot and `amd64` noise). No fork under ROCm or AMD-Ecosystem.
- Back, do not compete with: AMD's `ROCm/TransformerEngine` (dev branch, daily activity) and `ROCm/Megatron-LM` (rocm_dev). Both are MOAT projects whose intake recommends decline `ported-elsewhere`; recorded as dependencies so this waits for them to be dispositioned.

## Dependencies

- Hard: `TransformerEngine` (every recipe), `Megatron-LM` (megatron-fsdp, and Megatron-Core for the Megatron recipes). Recorded with `set-deps`.
- Optional / per-recipe: `flash-attention` (FA4 cu13 in the Evo2 recipes has no ROCm build), vLLM (ROCm builds exist), mamba-ssm / causal-conv1d (ROCm forks exist in the community; not verified), subquadratic-ops (binary-only, no ROCm).

## Validation scope

- **Delivery route:** fork-only by default. NVIDIA-owned upstream (`nvidia-org`: no upstream PR expected). The fork's `moat-port` carries ROCm Dockerfiles (`rocm/pytorch` base with ROCm TE) and capability-based skips; jeff decides whether anything goes upstream.
- **Scope question for jeff (not assumed):** expect a recipe-by-recipe result. The natural cut is to validate the TE + PyTorch recipes and models listed above and record the Megatron/NeMo/Evo2 recipes and NVFP4 as gaps with reasons. Per the standing rule a partial scope needs jeff's explicit answer before the planner/porter starts; the planner should present the cut, not take it.
- **GPU gate:** each in-scope directory's own pytest suite on GPU, run inside a ROCm container with ROCm TE: at minimum `models/esm2/tests` (`test_modeling_esm_te.py`, `test_hub_checkpoints.py`, `test_distributed_fp8.py`), `models/llama3/tests`, `recipes/esm2_native_te/tests`, `recipes/llama3_native_te/tests` (`test_train.py`, `test_quantization.py`, `test_train_two_gpu.py`), `recipes/geneformer_native_te_mfsdp_fp8/tests`, `recipes/codonfm_native_te/tests`; plus a short convergence run of `recipes/esm2_native_te/train_fsdp2.py` against the upstream benchmark curve. Multi-GPU and context-parallel tests need an Instinct node.
- **CUDA-only dependencies:** Transformer Engine (substitute: ROCm/TransformerEngine; check THD/context-parallel and FP8 recipe coverage there), megatron-FSDP (pure Python, expected fine), subquadratic-ops-torch-cu13 (gap, binary-only), flash-attn-4 cu13 (gap; FA2 ROCm or TE attention as substitute requires recipe edits), NVFP4 (gap), `nvcr.io` base images (substitute `rocm/pytorch`).
- **Windows:** no upstream Windows story (Docker/Linux, multi-GPU training); ROCm Transformer Engine has no Windows build. Suggest a `windows` gate waiver; only jeff approves it.
- **Risks:** largest scope of the batch; TE-ROCm feature parity (THD packing, CP, MXFP8) decides most recipes; wave32 Radeon has limited TE support and small memory, so the wave32 gate may come down to single-GPU ESM2 / Llama3 tests.
