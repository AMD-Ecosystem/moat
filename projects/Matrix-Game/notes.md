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
