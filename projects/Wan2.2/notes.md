# Wan2.2 notes

## Intake screen (2026-11-12)

Origin: Jeff Daily's 2026-11-12 ruling on a third-party "AMD AI software gap" list -- validation project (PyTorch + flash-attn; ROCm builds of both exist, never tested by MOAT on Instinct, Radeon or Windows).

- Upstream: https://github.com/Wan-Video/Wan2.2, default branch `main`, pushed 2026-09-21, not archived.
- Licence: Apache-2.0 (LICENSE.txt), tier 1. `scan-nvidia` on a depth-1 clone: clean. No submodules. (Model weights carry their own terms; not in scope for contribution.)
- CUDA surface: no `.cu`/`.cuh`. Pure PyTorch: `generate.py`, `wan/` (T2V-A14B, I2V-A14B MoE, TI2V-5B, S2V, Animate), FSDP + Ulysses sequence parallel via torchrun.
- Duplicate effort: AMD-authored upstream PR #30 (zhangnju, AMD; open since 2025-07-29, mergeable, no maintainer response) "[ROCM][AMD CDNA GPU] Advise wan2.2-14B users to tune frame-num" -- reports all Wan2.2 models run on CDNA3 and adds an AMD frame-count hint in `generate.py`. Back it with validation evidence rather than compete. Community: dikullu/wan2gp-rocm, yangxiarui4-dotcom/wan22-strix-halo-windows-guide (Windows on gfx1151 Strix Halo, via ComfyUI/Wan2GP) -- non-authoritative hints only. No AMD-Ecosystem or ROCm fork.

## Validation scope

- Delivery route: upstream PR of any fixes plus a ROCm install note (requirements make `flash_attn` mandatory), coordinated with AMD's open PR #30; fork-only if upstream stays silent (it has not answered #30 in 15 months).
- GPU gate: `tests/test.sh <model dir> <gpus>` (upstream's own end-to-end run of `generate.py` across t2v-A14B, i2v-A14B, ti2v-5B, s2v, animate, single and multi-GPU with FSDP/Ulysses). On a single Radeon card the 5B TI2V model is the practical gate; A14B MoE needs Instinct-class memory.
- CUDA-only dependencies: `flash_attn` (FA2) and `flash_attn_interface` (FA3, Hopper only). ROCm substitute: ROCm/flash-attention (CK backend on MI200/MI300/MI350; Triton AMD backend on RDNA). `wan/modules/attention.py` falls back to `torch.nn.functional.scaled_dot_product_attention` when neither is importable, but `requirements.txt` lists `flash_attn` unconditionally, so a plain install fails where no wheel exists (also upstream issue #108). FA3 path is irrelevant on AMD.
- Windows: no upstream Windows story (Linux, torchrun, flash_attn). Community runs Wan2.2 on Windows Strix Halo through ComfyUI, so it is not impossible, but suggest a windows waiver for this repo; only Jeff approves one.
- Existing AMD work to back: AMD PR #30 upstream (validation evidence and a comment there would be the natural channel, via Jeff).

## Planning (2026-11-20, linux-gfx942, planner; no GPU used)

**Jeff must rule before a porter starts.** Three questions; details and the recommended default for each are in `plan.md` under "Open questions".

- **Q1, scope.** A (default): all five tasks (t2v-A14B, i2v-A14B, ti2v-5B, s2v-14B, animate-14B), Animate preprocessing on the GPU (pose2d.py gets an onnxruntime MIGraphX provider choice instead of the silent CPU fallback it gets today), CosyVoice TTS attempted, with a deferral against CosyVoice if it will not install. B: all five generation tasks, but Animate preprocessing stays on the CPU and TTS is scoped out (a partial port). C: only the three tasks `tests/test.sh` runs; S2V and Animate deferred (a partial port).
- **Q2, wave32 gate.** A (default): TI2V-5B (with the offload options) plus whatever 14B tasks the Radeon card's memory allows; the 14B tasks are proven on wave64. B: every in-scope task must pass on wave32, which needs a 48 GB+ RDNA card.
- **Q3, make flash_attn optional with an SDPA fallback?** A (default): no. ROCm flash-attention exists, and the change collides with third-party PR #390. B: yes, as an extra portability feature.
- Q4 (after validation): comment on AMD PR #30 with evidence, and coordinate the install text with PR #390. Not blocking.

Findings that correct the intake screen:
- `flash_attn` is mandatory at runtime, not only in requirements. Every pipeline calls `flash_attention()`, which asserts `FLASH_ATTN_2_AVAILABLE`; the SDPA fallback is in `attention()`, which nothing calls (grep; PR #390's body says the same). S2V and Animate also call `flash_attn_func` directly.
- `tests/test.sh` covers t2v-A14B, i2v-A14B and ti2v-5B only, all multi-GPU through torchrun with FSDP and Ulysses (its 1-GPU lines are commented out). S2V and Animate have no upstream test, so the README commands are their gate.
- `import wan` needs a GPU: `t5.py:478` has `device=torch.cuda.current_device()` as a default argument. There is no CPU regression set.
- Upstream PRs that touch the same area: #30 (AMD, a frame_num hint; do not compete with it) and #390 (third party, open, mergeable: moves flash_attn out of requirements.txt, makes s2v/animate imports lazy, turns the assert into a RuntimeError). Keep our install note out of the lines #390 changes.

Classification: pure-Python PyTorch app, `ext_type` `python-pytorch`. No .cu, C++, CMake or extension, and the generated surface floor is empty, so I wrote all 14 surface.json components by hand. The only NVIDIA-specific code path in the repo is onnxruntime's `CUDAExecutionProvider` in `animate/preprocess/pose2d.py`. Everything else (autocast('cuda'), torch.cuda.*, the nccl backend) runs unchanged on ROCm PyTorch.

Dependencies: `depends_on: [flash-attention]` (a hard runtime dependency). Dao-AILab/flash-attention is dispositioned `already-supported`, so this blocks nothing. Matrix-Game records the same dependency.
