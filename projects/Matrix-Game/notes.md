# Matrix-Game notes

## Intake screen (2026-11-12)

Validation project (lane `validate`, seed `amd-gap-matrix`), taken up per jeff's 2026-11-12 ruling on the third-party "AMD AI software gap" list: probably runs on ROCm through PyTorch, untested on Instinct, Radeon and Windows.

- Upstream: https://github.com/SkyworkAI/Matrix-Game, default branch `main`, active (pushed 2026-10-29), not archived.
- Licence: MIT at the top level (tier 1). Sub-trees: `Matrix-Game-3/LICENSE.txt` and `Matrix-Game-4/LICENSE` are Apache-2.0 (tier 1); `Matrix-Game-1/GameWorldScore/GameWorld/third_party/DROID-SLAM` is BSD-3-Clause. `utils/licenses.py scan-nvidia` on a depth-1 clone: no NVIDIA proprietary licence text. No submodules.
- Layout: four independent model generations in one repo. `Matrix-Game-1` (diffusers, flash_attn 2.7.4 pinned, torch 2.5.1), `Matrix-Game-2` (Wan-based, flash_attn, optional TensorRT/pycuda/onnxruntime for the VAE), `Matrix-Game-3` (Wan2.2-based, torch 2.10, flash_attn FA3 -> FA2 with a hard `assert`), `Matrix-Game-4` (current, `pyproject.toml` package `matrix-game-4.0`, torch 2.12, `matrix-game` CLI, CPU pytest suite under `tests/`).
- CUDA surface: no `.cu` in the model code. The only `.cu` files (12) are the vendored DROID-SLAM extension under `Matrix-Game-1/GameWorldScore`, an evaluation benchmark, not inference. GPU code is PyTorch plus flash-attn plus Triton: `Matrix-Game-4/matrix_game/inference/{kernels.py,rope_kernel.py}` and per-GPU preset kernels under `inference/rtx5070ti`, `rtx5080`, `rtx5090`, `rtx5090_sage` (SageAttention, FP8 / int8 / MXFP8 linears; `tl.float8e4nv`, `torch.float8_e8m0fnu`).
- Hardware stated upstream: inference on Linux + NVIDIA (A100 80 GB `realtime` preset; RTX 5070 Ti / 5080 / 5090 presets). Standalone game client builds on macOS/Linux; "the legacy Windows entry is retained but has not been validated".

## Duplicate effort

- Upstream PR #67 (open, mergeable, last updated 2026-09-29, no review): https://github.com/SkyworkAI/Matrix-Game/pull/67 by ZJLi2013, "add AMD ROCm support via AITER CK flash attention backend". Touches only `Matrix-Game-3/wan/modules/attention.py` (+32/-4, dispatch FA3 -> AITER CK -> FA2) and the MG-3 README; tested on MI300X / ROCm 7.2 / `amd-aiter 0.1.13`. This is AMD's effort to back, not compete with.
- No fork or mirror under ROCm or AMD-Ecosystem (GitHub search 2026-11-12). No AMD/ROCm mention in upstream docs.
- No other open or closed upstream PR matching rocm/amd/hip.

## Dependencies

- Hard: `flash-attention` (MOAT project, intake recommends decline `already-supported`: upstream ships ROCm CK and Triton backends). Recorded with `set-deps`.
- Optional: AITER (`ROCm/aiter`) for the PR #67 CK path; SageAttention for the `rtx5090_sage` preset (not a MOAT project; Triton-based, so likely runs, unverified); TensorRT / pycuda (MG-2 only).

## Validation scope

- **Delivery route:** upstream PR of fixes. MIT/Apache, active, non-NVIDIA upstream. First action is to get PR #67 reviewed and landed (test it on wave64 and wave32, comment with results via jeff); then extend the same dispatch to `Matrix-Game-4/matrix_game/inference/wan/modules/attention.py` and `inference/runtime.py` (FA3 -> FA2 -> `assert`, no SDPA fallback), plus a README ROCm install note. Fork holds staged fixes.
- **GPU gate:** upstream has no GPU tests (`Matrix-Game-4/tests` is CPU-only; run it anyway, it must not regress). The gate is an end-to-end inference run: Matrix-Game-4 `matrix-game` CLI with the downloaded weights and the `realtime` preset on Instinct (80 GB class), and an RTX-class preset on Radeon (memory permitting), checking the output video against a CUDA reference frame-wise; plus Matrix-Game-3 `generate.py` as PR #67 tests it. Matrix-Game-1/2 are legacy and optional.
- **CUDA-only dependencies:** flash-attn (ROCm substitute: upstream flash-attn CK/Triton backend, or AITER as in #67). TensorRT and pycuda in MG-2's optional VAE acceleration (gap; use the PyTorch path, or MIGraphX/ORT-ROCm if anyone wants it). Triton FP8 preset kernels use `tl.float8e4nv` (OCP e4m3): fine on gfx950 / gfx12, but gfx942's native FP8 is FNUZ, so the FP8 / MXFP8 RTX presets may need a dtype selection or stay NVIDIA-only there (unverified); the bf16 `realtime` path has no such issue. The vendored DROID-SLAM `.cu` (GameWorldScore metric) is out of scope unless someone needs the benchmark.
- **Windows:** upstream inference is Linux-only and the Windows client entry is explicitly unvalidated. Suggest a `windows` gate waiver (jeff approves waivers, not agents). A gfx1151 run is also impractical for the 80 GB-class realtime preset.
- **Risks:** model size vs Radeon memory; FP8 dtype on gfx942; FA3-only paths (`is_hopper_gpu`) silently not taken on AMD, which is fine; SageAttention Triton kernels unverified on wave64.

## Planning (2026-11-20, planner on linux-gfx950)

**DECISIONS NEEDED FROM JEFF BEFORE A PORTER STARTS.** The plan would port only part of the repository, so it needs his ruling; nothing below has been assumed. Full costing in plan.md, "Open questions".

- **Q1. Which model generations are in scope?**
  - A (recommended): Matrix-Game-4 + Matrix-Game-3; MG-1 and MG-2 scoped out as superseded legacy generations.
  - B: A + Matrix-Game-2 (one optional-import fix, README note that its TensorRT/pycuda VAE is NVIDIA-only, one wave64 end-to-end run).
  - C: B + Matrix-Game-1 (inference is cheap; its GameWorldScore benchmark vendors DROID-SLAM/lietorch/RAFT CUDA extensions, a real torch-extension HIP port; planner recommends scoping that benchmark out even under C).
- **Q2. Matrix-Game-4's six NVIDIA-tuned RTX presets (`rtx5090`, `rtx5080`, `rtx5070ti`, and `-sage` variants)?**
  - A (recommended): scope them all out; AMD users run `realtime`/`quality`/`baseline`; the port adds a clear refusal on ROCm. Caveat: if the wave32 card cannot hold bf16 `realtime`, wave32 then has no fitting fast preset and needs B or a waiver.
  - B: make the three non-Sage RTX presets run where the datatypes exist (FP8 on gfx950/gfx12, FNUZ on gfx942; MXFP8 on gfx950 only with ROCm's scale layout, since PyTorch's ROCm MX path rejects `SWIZZLE_32_4_4`; INT8 Triton everywhere). 50-100 lines, validated on gfx950 and a gfx12 card.
  - C: B + the `-sage` presets, which need SageAttention 2.2 on ROCm (CUDA C++ with sm89 PTX mma, no ROCm build): a separate intake, not this port.

surface.json leaves the Q1/Q2 components unaccounted (notes say "PENDING Jeff ruling"); the porter records them covered or scoped_out per the ruling.

### Findings

- Upstream `main` at `51913b2bfa78` (2026-10-29). Census: 0 CUDA files in scope; the only `.cu` are vendored third-party extensions under `Matrix-Game-1/GameWorldScore/GameWorld/third_party/`. `ext_type` set to `python-triton` (pure Python PyTorch + Triton; MG-4 is a setuptools package with no ext_modules).
- PR #67 (https://github.com/SkyworkAI/Matrix-Game/pull/67) rechecked 2026-11-20: OPEN, MERGEABLE/CLEAN, head `9ad32b1d90ef`, 0 comments, last update 2026-09-29. MG-3 only. Its AITER call drops `window_size`/`dropout_p`/`deterministic` (benign: MG-3 inference uses the defaults).
- MG-4 already falls back to SDPA when flash-attn is absent (`wan/modules/attention.py`; `runtime.py` uses FA2 varlen only when importable), so MG-4 needs no flash-attn on ROCm. The SDPA fallback ignores padding masks; that matters for the `quality` path's padded text cross-attention (plan risk 2).
- MG-4 FA3 is gated on the device name ("h100"/"hopper"), so it stays off on AMD. MG-3 turns FA3 on whenever `flash_attn_interface` imports (`WAN_FA_VERSION` default "3"); the validator must check that module is absent from the ROCm environment.
- MG-3's README model download (`huggingface-cli download Matrix-Game-3.0`) names no HF organisation; the repository id is unverified.
- PyTorch ROCm MX GEMM check read in a torch 2.16 dev source (`aten/src/ATen/native/cuda/ScaledBlas.cpp`): MX archs are gfx950 (and gfx1250); swizzle must be `NO_SWIZZLE` or `SWIZZLE_32_8`. Re-check in the torch 2.12 ROCm wheel before acting on it under Q2-B.
- Delivery: one upstream PR from `moat-port`, Matrix-Game-4 only (ROCm install docs, plus code only where a real ROCm run needs it). MG-3 goes through PR #67: validate its head on wave64 (and wave32 if memory allows), draft a results comment here for Jeff's approval. No competing PR.
- GPU gate: upstream ships no GPU tests. Use upstream's own release gate (docs/testing.md): the same clip of more than 21 latent chunks through `--dit-runtime native --no-vae-graph` and through `realtime`, compared frame-wise, plus `quality`, each with and without ROCm flash-attn; MG-3 single-GPU `generate.py` via ROCm FA2, PR #67 + AITER, and `--fa_version 0` as reference. Non-GPU regression: `python -m unittest discover -s tests -p 'test_*.py'` in Matrix-Game-4.
- Windows gate waived (Jeff, 2026-11-12); no Windows plan.
- No GPU was used by this planning run; nothing was built.

## Rulings on the planner's questions (Jeff Daily, 2026-11-23)

- Q1 (generations): **A**. Matrix-Game-4 and Matrix-Game-3; Matrix-Game-1 and Matrix-Game-2 scoped out as superseded legacy generations.
- Q2 (RTX presets): **B**. Make the three non-Sage RTX presets run where the datatypes exist; the three `-sage` variants are scoped out with a registered deferral.
- Q2 addition: the porter spends a bounded check (about 30 minutes, no porting) on SageAttention: does the 2024 community fork github.com/EmbeddedLLM/SageAttention-rocm (last pushed 2024-11-21, SageAttention 2.0.0 beta; the reason thu-ml/SageAttention was dispositioned `ported-elsewhere` on 2026-05-30) build on current ROCm, and what did SageAttention 2.2 add over what that fork covers. The answer goes in the notes for Jeff; it decides nothing by itself.
- Standing ruling of the same day: Windows laptop GPU runs go last across all ports. The planner's suggested windows waiver is not approved or declined yet.
