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
