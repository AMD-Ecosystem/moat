# LTX-2 notes

## Intake screen (2026-11-12)

Origin: Jeff Daily's 2026-11-12 ruling on a third-party "AMD AI software gap" list named Lightricks/LTX-Video. **LTX-Video's README now says LTX-2.5 (released 2026-08-11) supersedes both LTX-Video (v0.9.x) and LTX-2, with code and weights in https://github.com/Lightricks/LTX-2**, so this screen covers the current repo, Lightricks/LTX-2 (pushed 2026-11-12), not LTX-Video (last pushed 2026-11-02, legacy). If Jeff wants the legacy repo validated too, it is a separate project.

- Upstream: https://github.com/Lightricks/LTX-2, default branch `main`, not archived. uv workspace: `packages/ltx-core`, `ltx-pipelines`, `ltx-trainer`, plus opt-in `ltx-kernels`.
- Licence: Apache-2.0 (LICENSE), tier 1. `scan-nvidia` on a depth-1 clone: clean. No submodules. (Weights carry the LTX model licence; not in scope for contribution.)
- CUDA surface: 23 `.cu`/`.cuh` files, all in `packages/ltx-kernels` (all2all / allgather multi-GPU kernels, a vendored DeepGEMM blockwise FP8 GEMM using CuTe). `ltx-kernels` is excluded from the uv workspace and is an opt-in `kernels` group ("keep it out of the workspace so a plain `uv sync` never forces a CUDA toolchain"), so the default inference path is pure PyTorch. Porting ltx-kernels (DeepGEMM/CuTe is Hopper-specific) is a separate, much larger question; out of scope for validation.
- Duplicate effort: **maintainer closed #141 "Is AMD / ROCm support planned?" with "AMD/ROCm is not planned. Closing."** and closed #325 (gfx1102 Windows ComfyUI corruption/freeze) as not actionable here. No upstream ROCm PRs or branches; no AMD-Ecosystem or ROCm fork; community repo The-Deflection-Engine/ltx-2.5-rocm-generator (non-authoritative hint).

## Validation scope

- Delivery route: **fork-only** is the realistic default given the maintainer's "not planned" ruling; at most offer small packaging fixes (e.g. the Linux-only `nvidia-cudnn-cu13` override and CUDA torch index) upstream if Jeff wants to try.
- GPU gate: no upstream test directory was found. Gate is the README pipelines: `DistilledPipeline` text/image-to-video (8+4 steps, fastest) end to end, then the DFR production pipeline; VAE decode with Triton and eager neighbourhood-attention fallbacks.
- CUDA-only dependencies: `natten` extra (Linux+CUDA only; README says it falls back to Triton or eager automatically -- Triton on ROCm is the path to check), FlashAttention 3/4 (Hopper/Blackwell; SDPA used otherwise, which is the AMD path), `nvidia-cudnn-cu13` override (Linux, CUDA), `ltx-kernels` (out of scope, see above). FP8/int8 checkpoint variants need checking per arch.
- Windows: upstream has a Windows story (README: natten skipped on Windows/macOS, Triton or eager fallback, "the same command works everywhere"), so Windows is in scope -- no waiver. Note #325's gfx1102 freeze on a 16 GB card with the 22B model: memory-heavy runs on the gfx1151 laptop need care (light runs only).
- Existing AMD work to back: none official; community generator repo only.
