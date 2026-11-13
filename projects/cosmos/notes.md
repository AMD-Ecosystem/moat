# cosmos notes

## Intake screen (2026-11-12)

Origin: Jeff Daily's 2026-11-12 ruling on a third-party "AMD AI software gap" list -- validation project for NVIDIA Cosmos 3.

- Repo check: `NVIDIA/Cosmos` exists (canonical name `NVIDIA/cosmos`, default branch `main`, pushed 2026-11-11, not archived) and is the current home. nvidia-cosmos/cosmos-predict2.5's README says Cosmos 3 supersedes it, the repo gets only limited maintenance, and points to https://github.com/NVIDIA/Cosmos. So this screen covers NVIDIA/cosmos, not predict2.5.
- **What the repo is:** a landing repo with no code -- README, COOKBOOK.md, AGENTS.md, docs/reference (models, inference benchmarks), assets. The quickstart runs Cosmos 3 through Hugging Face diffusers (`Cosmos3OmniPipeline`, diffusers from git main) and serving through vLLM / vLLM-Omni / SGLang / TensorRT-LLM / NIM. Training code and recipes live in **NVIDIA/cosmos-framework** (OpenMDW-1.1, 4922 files, 10 `.cu` files in `packages/ludus-renderer` for the simulation renderer, pytest `tests/`). Whether the MOAT project should instead (or also) target NVIDIA/cosmos-framework is Jeff's call; a fork of NVIDIA/cosmos can only hold docs.
- Licence: **OpenMDW-1.1** (LICENSE, NVIDIA/cosmos; GitHub reports NOASSERTION). OpenMDW is the Linux Foundation's permissive model-and-data licence, but it is not in `config/licenses.toml`, so `utils/licenses.py tier OpenMDW-1.1` returns **tier 4** -- contributing needs Jeff's licence clearance (`record-license-clearance`), or a licences.toml addition. cosmos-framework carries the same OpenMDW-1.1 text under an NVIDIA "All rights reserved" copyright line. `scan-nvidia` on a depth-1 clone of NVIDIA/cosmos: clean. No submodules.
- CUDA surface (this repo): none. Runtime surface: PyTorch, diffusers, transformers, vLLM.
- Duplicate effort: no ROCm/AMD/HIP issues, PRs or branches in NVIDIA/cosmos or NVIDIA/cosmos-framework; no AMD-Ecosystem or ROCm fork found.

## Validation scope

- Delivery route: validation on ROCm of the README quickstart and cookbook recipes; fixes belong in the runtimes (diffusers, vLLM -- both already have ROCm support, so report or fix there) and, if Jeff retargets, in cosmos-framework. The only upstream deliverable to NVIDIA/cosmos itself is a docs PR adding a ROCm environment section, and that waits on the tier-4 licence clearance. Otherwise fork-only.
- GPU gate: README "Generate your first video" diffusers example (`Cosmos3OmniPipeline.from_pretrained("nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16)`) plus the Reasoner via transformers/vLLM; if cosmos-framework is taken up, its `tests/` pytest suite. Model weights are large (Nano is the smallest generator; Cosmos3-Edge is 4B).
- CUDA-only dependencies: TensorRT-LLM and NIM (no ROCm path -- out of scope); FP8 checkpoints (`*-fp8-*` benchmarks) need fp8 GEMM support on the target arch; FlashAttention 4 in cosmos-framework (SM100/SM120 only; ROCm substitute is CK/aiter flash-attn or SDPA). vLLM ROCm and diffusers on ROCm torch are the substitutes for the serving stack.
- Windows: no upstream Windows story (Linux, NGC containers, H100/GB300 recipes). Suggest a windows waiver; only Jeff approves one.
- Existing AMD work to back: none in Cosmos repos; vLLM and diffusers ROCm support upstream is what this validation leans on.

## Declined (2026-11-12)

Jeff Daily declined Cosmos on 2026-11-12: NVIDIA/Cosmos holds no code, its runtimes (diffusers, vLLM) already support ROCm, and NVIDIA/cosmos-framework is not taken up, to keep MOAT in scope. OpenMDW-1.1 is not added to the licence table (licence decisions are not his to make). Intake is re-recorded as decline and the project is held (`set-hold cosmos on`), since awaiting-fork has no edge back to unclaimed. Do not request a fork.
