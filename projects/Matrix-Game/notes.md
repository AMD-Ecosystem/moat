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

## Porting (linux-gfx950, 2026-11-23, porter job 144930)

### Questions for Jeff (nothing below assumes an answer)

1. **Comment on upstream PR #67.** A draft is below under "Draft comment for PR #67". Options: (a) post it as drafted, (b) post a shorter results-only version, (c) post nothing yet. Evidence: the MG-3 runs in this section. Nothing was posted.
2. **MG-3 `--fa_version 0` has no SDPA path (upstream bug on every platform, not ROCm-specific).** `wan/modules/model.py:475` and `:503` call `flash_attention()` directly, which asserts that flash-attn is installed. `--fa_version 0` only affects the `attention()` wrapper. On upstream `main`, `python generate.py ... --fa_version 0` without flash-attn fails with `AssertionError` at `wan/modules/attention.py:114` (data `mg3out/sdpa.log`). Options: (a) mention it in the PR #67 comment (included in the draft as one sentence), (b) a separate upstream issue, (c) leave it. This is upstream-visible, so it is your call.
3. **Windows waiver record conflict.** `status.json` carries an approved `windows` waiver ("approved_by Jeff Daily, 2026-11-12"), but the rulings section above says the planner's suggested waiver "is not approved or declined yet". The tooling treats it as approved. Please confirm, or rule otherwise with the waiver tooling.

### What this round did

An earlier job of this platform (130701) pushed `moat-port` at `6d94b2f` (three commits) but never recorded the head, and its evidence was in its own job directory. This job took the lock over, reviewed those three commits, and re-ran every build and GPU check from scratch in its own job directory. The review found nothing to change, so no new fork commit was made. Head: `6d94b2fb5ccdd1351b3ce5b5242cadda5f367f70`.

Fork commits on `moat-port` (base upstream `main` `51913b2`):
- `a75895e` rtx5080: use the portable libdevice namespace for tanh (shared fix, no prefix)
- `f7bf6bf` [ROCm] Run the FP8 and MXFP8 RTX presets on AMD GPUs
- `6d94b2f` [ROCm] Document installing on AMD GPUs (README "AMD ROCm" subsection plus a pointer in docs/installation.md)

### Environment (all inside the job directory)

- `$FLEET_VENV` (Python 3.11): `torch 2.12.0+rocm10.2.0a20261015` (git `9ccf2fd0`), torchvision 0.27.0, `triton 3.8.0+git669b31ac.rocm10.2.0a20261015`, `rocm[devel]==10.2.0a20261015`, from `--index-url https://nightly.repo.amd.com/rocm/whl-next/` with `torch[device-gfx950]`. HIP 7.17.26412. `torch.cuda.device_count()` reports 0 while compute works (amdsmi count quirk in this image); harmless.
- MG-4: `pip install -e '.[inference,server,download,test]'`. MG-3: the same venv plus `easydict decord opencv-python==4.10.0.84 trimesh pandas==2.2.3` (pandas 2.0.3 breaks AITER's JIT).
- **Deviation:** MG-3 pins torch 2.10.0 and transformers 4.57.3. TheRock ROCm 10.2 publishes only torch 2.12 to 2.15, so MG-3 ran on torch 2.12.0 and transformers 5.17.0. download.pytorch.org has `torch-2.10.0+rocm7.1`, which this job did not test (the fleet guard allows only the one venv).
- flash-attention `v2.8.3.post2` (`49774b31`), two builds:
  - CK: `BUILD_TARGET=rocm GPU_ARCHS=gfx950 OPT_DIM=128 MAX_JOBS=32 pip wheel --no-build-isolation --no-deps .`, 2155 objects, about 75 min on a loaded host. The all-head-dims build (2782 objects) was too slow and was replaced; both models use head dim 128.
  - Triton AMD backend: `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, a pure-Python wheel, no compile.
- AITER `v0.1.30` (`77ba72b70`), `GPU_ARCHS=gfx950 PREBUILD_KERNELS=0 pip wheel --no-build-isolation --no-deps .`. The wheel reports version 0.0.0 (setuptools-scm); kernels JIT on first call (about 18 s, `fmha_v3_varlen_fwd`). Smoke test of `aiter.ops.mha.flash_attn_varlen_func` against SDPA (varlen 1000+777, H 24, D 128, bf16): max abs err 0.
- Weights (downloaded in the job directory; 936 GB free at the start): MG-4 `Skywork/Matrix-Game-4.0` revision `2c53100457da7abcf0e966f75637c644d3423a63`, scene ancient-egypt, game-assets and models manifests (every file SHA256-verified by `matrix-game download`), 23 GB plus a 23 GB HF cache. MG-3 `Skywork/Matrix-Game-3.0` (the repository id the planner guessed is correct) at `d4c84ef2b8e9028a00a5d58b6be2df04f1cb6042`, 53 GB.
- GPU runs: `HIP_VISIBLE_DEVICES=7` (the dispatcher's pin), `ulimit -c 0`, `timeout` on each run.

### PyTorch ROCm MX GEMM constraint, re-checked in the installed wheel

`torch.nn.functional.scaled_mm` with `ScalingType.BlockWise1x32` in `torch 2.12.0+rocm10.2.0a20261015` on gfx950:
- `SwizzleType` has only `NO_SWIZZLE` and `SWIZZLE_32_4_4`. `SWIZZLE_32_4_4` gives `ValueError: For ROCM MX gemm, swizzle_a and swizzle_b must both be NO_SWIZZLE`.
- `NO_SWIZZLE` with row-major `[M, K/32]` e8m0 scales works (rel err 0.037). Transposed scales give garbage (rel err 2.45).
- M=405 fails: `Matrix dimensions must be multiples of 32 for block-wise scaling`. The activation rows (not only the scales) must be padded to a multiple of 32.
- With bias: `HIPBLAS_STATUS_INVALID_VALUE` from `hipblasLtMatmulAlgoGetHeuristic` ("rocRoller does not support bias").
- FP32 output: `Block-wise scaling only supports BFloat16 or Half output types`.
- Per-tensor FP8 `_scaled_mm` with `float8_e4m3fnuz` on gfx950: `could not find valid hipblaslt solution`. gfx950 multiplies only OCP e4m3fn, which is why the format is chosen per arch.

These match what the ROCm branch in `rtx5070ti/linear.py` does: row-major scales, `NO_SWIZZLE`, rows padded to 32, bias added after the GEMM. Cost: one extra BF16 rounding on the `cross_q` role compared with a fused epilogue.

### Results on gfx950 (MI355X)

Non-GPU: `python -m unittest discover -s tests -p 'test_*.py'` in Matrix-Game-4: 81 tests OK on upstream `main` and on `moat-port`. `node tests/browser/test_audio.cjs` (AUDIO_SCHEDULING_PASS) and `node tests/browser/test_hud.cjs` (HUD_TEST_PASSED) pass. The other four browser tests need Playwright and a live server and were not run.

Per-layer GPU check (`check_linears.py`, M=405, D=3072, FFN 14336, outlier columns, rel err against the BF16 layer): FP8Linear 0.037, PaddedFP8Linear 0.037, TransposedFFN 0.054, Int8Linear qkv 0.051, INT8 GELU 0.052, INT8 K>4096 0.015, MXFP8 at M=405/1/37/4096: 0.038/0.041/0.037/0.038, all finite. The MXFP8 layer replays bit-identically from a CUDA graph.

MG-4 end to end: `matrix-game render --scene ancient-egypt --resources <res> --input egypt-pan-120.mp4 --prompt-mode fixed --seed 42 --preset <p>`. The input clip is a 120-frame lossless pan over the scene's `sink.png` (sha256 `25c542ae1956c01e3ec35f29279d4b7cce9d77d768f3caae8d1e1ae9a901ca51`), 30 latent chunks, so the 21-chunk window evicts. Every run below rendered 120/120 frames at 864x480, with no NaN and non-constant frames (min per-frame std 44 or more). FPS is the CLI's "processing FPS" at frame 101, measured while a 32-job compile shared the CPU, so it is approximate.

| run | attention | FPS | peak alloc GiB | PSNR vs native (SDPA) mean/min dB |
|---|---|---|---|---|
| realtime | SDPA | 28.9 | 20.2 | 22.2 / 19.6 |
| native (`--dit-runtime native --no-vae-graph`) | SDPA | 17.6 | 20.2 | ref |
| rtx5090 | SDPA | 54.5 | 15.7 | 19.3 / 17.2 |
| rtx5080 | SDPA | 60.9 | 10.9 | 20.0 / 17.8 |
| rtx5070ti (MXFP8 `cross_q`) | SDPA | 58.7 | 10.9 | 18.9 / 16.8 |
| rtx5070ti, MXFP8 forced off (the BF16 fallback non-gfx950 GPUs take) | SDPA | 61.7 | 10.9 | 20.0 / 17.9 |
| quality | SDPA | 17.4 | 42.1 | n/a (other codec) |
| baseline | SDPA | 17.2 | 42.1 | n/a |
| realtime | FA2 Triton | 10.1 | 20.2 | 20.8 vs SDPA realtime |
| native | FA2 Triton | 24.8 | 20.2 | 21.8 |
| quality | FA2 Triton | 15.4 | 42.1 | 19.1 vs SDPA quality |
| rtx5090 (with FA2 Triton installed) | -- | 56.0 | 15.7 | -- |
| realtime | FA2 CK | 33.9 | 20.2 | 20.1 vs CK native |
| native | FA2 CK | 28.6 | 20.2 | 21.2 |
| quality | FA2 CK | 16.5 | 42.1 | 19.7 vs SDPA quality |

How to read the PSNR: the model is autoregressive, so any numerical difference (bf16 realtime against native, SDPA against FA2) grows over the clip, and correct paths land at 19 to 22 dB from one another. Upstream says realtime and native are not bitwise and publishes no number. The quantized presets sit 2 to 3 dB below bf16 realtime, as quantization would predict. Contact sheets (`data/montage-rtx.png`, `montage-quality.png`) show coherent scenes that follow the input pan in every run, with no corruption. CUDA-graph capture works with SDPA, FA2 CK and FA2 Triton (plan risk 1 does not occur).

Kernel trace (`rocprofv3 --kernel-trace --stats`, 24-frame clip `ed1eba18...`, CK flash-attn and AITER installed): rtx5070ti ran the rocRoller MX GEMM `RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_...` (2574 calls), `_mxfp8_quantize_kernel` (2604), `_int8_gemm_kernel` (12876) and `_quantize_rows_kernel`. rtx5080 ran hipBLASLt F8 GEMMs `Cijk_..._F8SS_...` (about 20700 calls), `_scale_quantize_rows_padded` (15570) and `_bias_transpose_kernel` (5148). Both also ran `aiter::fmha_fwd_hd128_bf16` for attention and the Triton `_ln_modulate_kernel` and `_qk_rms_rope_kernel`.

MG-3 end to end (`generate.py --size 704*1280 --use_int8 --num_iterations 4 --num_inference_steps 3 --image demo_images/001/image.png --seed 42 --vae_type mg_lightvae --lightvae_pruning_rate 0.5`, README prompt), 177 frames at 1280x704 each:

| code | backend | result | wall |
|---|---|---|---|
| upstream `main`, `--fa_version 0`, no flash-attn | (none) | `AssertionError` (question 2) | -- |
| upstream `main`, `--fa_version 2` | FA2 CK | OK | 71 s |
| upstream `main`, `--fa_version 2 --compile_vae` | FA2 CK | OK; 59.2 dB mean against no compile | 76 s |
| upstream `main`, `--fa_version 2` | FA2 Triton | OK; 24.8 dB against CK | 115 s |
| PR #67 head `9ad32b1d` | AITER v0.1.30 | OK; 24.8 dB against CK, 25.6 dB against FA2 Triton | 272 s (first call JITs) |
| PR #67, `--compile_vae` | AITER | OK | 132 s |

`flash_attn_interface` is not importable in this environment, so plan risk 3 does not occur here. The FA2 and AITER outputs are visually indistinguishable (`montage-mg3-quality.png`). The Triton INT8 autotune configs compiled on gfx950 (plan risk 5 did not occur). Multi-GPU torchrun was not run: this job had one GPU.

### CUDA path

No CUDA C++ is involved. Every Python change is `torch.version.hip`-gated or keeps the CUDA values (`float8_e4m3fn`, 448, `SWIZZLE_32_4_4`, bias in the GEMM). The four changed Triton kernels were compiled for sm_120 and sm_90 with Triton's NVIDIA backend and ptxas, from upstream `51913b2` and from `moat-port`. The PTX is identical once the DWARF debug sections (which embed the source path) are stripped. Script: `cuda_ptx_check.py`. This is Triton's compiler on this host, not an nvcc build; nothing else in the change reaches nvcc.

### What remains unproven

**gfx942 (MI300, FNUZ FP8, no MX):**
- `e4m3_format()` returns `float8_e4m3fnuz`/240 there. Never executed: gfx950 has no FNUZ GEMM, so it cannot stand in. The Triton cast to FNUZ via `y_ptr.dtype.element_ty` is bit-exact against torch's cast on gfx950, but whether hipBLASLt on gfx942 picks an FNUZ `_scaled_mm` solution for each shape (BF16 out with bias for rtx5090; FP32 out without bias for rtx5080) is untested.
- `rtx5070ti` there takes the BF16 fallback for MX roles (exercised on gfx950 by forcing it, above) plus FNUZ for its fp8 roles.
- The `-sage` deferral is unaffected.
- Accuracy with max 240 (less headroom than 448) is unmeasured.

**gfx12 (RDNA4, OCP FP8, no MX):**
- FP8 `_scaled_mm` on gfx1200/gfx1201 is untested in this torch.
- The Triton INT8 GEMM's NVIDIA tile configs (for example 64x256x64, 3 stages) may exceed RDNA LDS or perform poorly.
- The FP64 RoPE kernel is slow on RDNA.
- Memory: rtx5080/rtx5070ti peak at 10.9 GiB allocated, so a 16 GB card should fit; rtx5090 needs 15.7 GiB allocated (18.1 reserved), so it does not fit 16 GB.
- Wave32 occupancy of the Triton kernels is unmeasured.
- A gfx12 run needs the wave32 validator.

**Other gaps:**
- gfx1250 (an MX-capable arch in newer torch) takes the BF16 fallback because `mxfp8_supported()` matches only gfx950. Safe, but untested.
- The documented install line (`download.pytorch.org/whl/rocm7.2`, `torch==2.12.0`, cp312) exists on the index (checked) but was not installed or run: the guard allows one venv, and it holds the TheRock build. Validators should run the documented wheel where they can.

### SageAttention check (bounded, about 10 minutes, no porting)

- **The fork is a mirror.** github.com/EmbeddedLLM/SageAttention-rocm is a mirror of thu-ml/SageAttention as of 2024-11-21, with no ROCm work: no commit by EmbeddedLLM, and no `__HIP_PLATFORM`/hip/rocm/gfx/amd hits in `csrc/`, `setup.py` or `sageattention/` on any branch.
  - `main` (`4171d501`) is exactly upstream's "2.0.0 beta" plus a README commit.
  - `2.0.0` (`726941a2`) and `sageattention-1` (`f54ccfde`) are upstream's pre-force-push branches. Their only extra commits are README/image changes by a thu-ml author.
- **Build against the job ROCm** (`ROCM_HOME` set, `python -m pip wheel --no-build-isolation --no-deps .`):
  - `main` and `2.0.0` fail during metadata with `RuntimeError: Cannot find CUDA_HOME. CUDA must be available to build the package.` (setup.py lines 47-49; line 110 then requires nvcc 12.0 or later).
  - `sageattention-1` builds `sageattention-1.0.6-py3-none-any.whl`, which is pure Triton.
- **The fork's Triton INT8 attention runs on gfx950** against SDPA (bf16, B1 H8 S1024): max abs err 5.9e-3 non-causal and 3.9e-2 to 4.3e-2 causal for D 64 and 128, cosine 0.9999 or better. It is not what Matrix-Game calls.
- **What the `-sage` presets call** (all three, through `loader.py`):
  - `sageattention._fused.quant_per_block_int8_fuse_sub_mean_cuda`, `transpose_pad_permute_cuda`, `scale_fuse_quant_cuda` and `quant_per_warp_int8_cuda`. These are in the fork, with the same signatures as v2.2.0.
  - `sageattention.sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf` (NHD, non-causal, per-warp, returns LSE, head dim 128). It is **not** in the fork: neither the `sm89_compile` module nor the FP8-PV kernel with fp16 accumulation exists there.
  - Matrix-Game's `sageattention.patch` only threads `at::cuda::getCurrentCUDAStream()` through each launch, for CUDA graphs. It changes no kernel math.
- **What v2.2.0 (`eb615cf6`) adds over the fork:**
  - Extensions `_qattn_sm80`, `_qattn_sm89` (seven split FP8 kernels: accum f32/f16, inst_buf, fuse_v_scale, fuse_v_mean; built for sm89/90/120) and `_qattn_sm90` (wgmma plus TMA).
  - The `sm80/sm89/sm90_compile.py` torch.library wrappers, `fa3_wrapper.py`, and a separate sageattention3_blackwell FP4 tree.
  - New PTX: `mma.sync m16n8k32 f16.e4m3.e4m3.f16` (4 sites), wgmma (59), `cp.async.bulk.tensor` (2), mbarrier (5). Shared with the fork: s8/s4/f16/e4m3 `mma.sync`, ldmatrix, cp.async, `cvt e4m3x2`, shfl.
- **Assessment (decides nothing):** a ROCm build of what the presets need is four simple quantize kernels (a wave64 audit of their shuffle reductions) plus one attention kernel that is a rewrite, not a hipify.
  - INT8 `mma.sync` maps to i8 MFMA.
  - The FP8 PV MMA with fp16 accumulation has no CDNA counterpart (MFMA FP8 accumulates in fp32; gfx942 is FNUZ; RDNA3 has no FP8 WMMA).
  - `ldmatrix` becomes LDS loads in MFMA operand layouts, which also changes the 16-token V permutation that `transpose_pad_permute` and Matrix-Game's `_scatter_v_kernel` encode.
  - A cheaper route is a Triton backend for `SageWindow`: the INT8-QK Triton kernel already runs on gfx950, and an FP8-PV variant would be a modest extension.
  - The EmbeddedLLM fork contributes nothing to either route.
  - The registered deferral `matrix-game-sageattention-rocm` stands as written.

### Draft comment for PR #67 (for Jeff's approval; NOT posted)

Per the posting rules this carries agent analysis, so it would go up with the preface line and quoted. Shown as it would be posted:

====
My agent came up with this analysis, but I've read and approved it.

> Tested this PR's head (9ad32b1d) on an AMD Instinct MI355X (gfx950) with a ROCm build of PyTorch 2.12 and AITER built from source (v0.1.30). `python generate.py --size 704*1280 --use_int8 --num_iterations 4 --num_inference_steps 3 --vae_type mg_lightvae --lightvae_pruning_rate 0.5 --image demo_images/001/image.png --seed 42` completes with and without `--compile_vae` (177 frames), and the output agrees with the FA2 path (flash-attention 2.8.3 CK build on the same GPU) to the run-to-run level we see between any two attention backends, with no visible difference. AITER's varlen call also matches SDPA exactly on a standalone check.
>
> Three small things:
> 1. `pip install aiter` in the README and in the new RuntimeError installs an unrelated PyPI package (async iterator helpers). The AMD package is `amd-aiter`, or the source build from ROCm/aiter that the README also lists.
> 2. The "Flash Attention Configuration" banner in `pipeline/inference_pipeline.py` and `pipeline/inference_interactive_pipeline.py` does not know about AITER, so on ROCm it prints `Actual: None (SDPA)` while AITER is in use.
> 3. Unrelated to this PR: `--fa_version 0` does not give an SDPA run, because `wan/modules/model.py` calls `flash_attention()` directly, so without flash-attn or AITER it still fails at the assert.
====

Validator notes: the head is already validated by this porter's runs on gfx950, but the validator role records its own evidence. Re-run at least `realtime`, `native`, `rtx5090`, `rtx5080`, `rtx5070ti` and `quality`, the unit suite, and MG-3 with FA2 and with PR #67 + AITER. Helper scripts used here: `compare.py`, `montage.py`, `make_clip.py`, `check_linears.py`, `mg4_render.sh`, `mg3_run.sh` (job directory `scripts/`; not durable).

## Lesson for cuda-to-rocm: PyTorch's ROCm MX GEMM wants unswizzled scales, 32-aligned M, no bias

For `libraries.md` (BLAS, hipBLASLt section). `torch.nn.functional.scaled_mm(..., ScalingType.BlockWise1x32, ...)` on ROCm (gfx950, hipBLASLt rocRoller, checked in torch 2.12) rejects cuBLAS's `SWIZZLE_32_4_4` scale layout. It needs `NO_SWIZZLE` with plain row-major `[M, K/32]` e8m0 scales (transposed scales run but return garbage). M, N and K must all be multiples of 32: pad the activation rows with zeros, not only the scales. A `bias=` argument fails at the heuristic query (`HIPBLAS_STATUS_INVALID_VALUE`), and the output must be BF16 or FP16. So a CUDA MXFP8 linear ports as: quantize to row-major scales, pad rows to 32, GEMM without bias, slice, add the bias. Separately, per-tensor FP8 `_scaled_mm` takes `float8_e4m3fnuz` (max 240) on gfx942 and only `float8_e4m3fn` (max 448) on gfx950 and gfx12, so pick the e4m3 type per arch (as PyTorch's tests do with `e4m3_type`) instead of hard-coding either.

## Lesson for cuda-to-rocm: Triton's CUDA-namespaced libdevice and hard-coded FP8 types

For `strategy-b-torch.md` (or wherever Triton kernels are covered). Triton's AMD backend refuses `tl.extra.cuda.libdevice.*` ("Implicit conversion of CUDA __nv_tanhf device function has been dropped"). `triton.language.extra.libdevice` dispatches per backend and compiles to the same PTX on NVIDIA. Likewise, a Triton FP8 quantizer that casts to the literal `tl.float8e4nv` with 448 clamps cannot emit gfx942's FNUZ format. Cast to `y_ptr.dtype.element_ty` and pass the format's max as a constexpr, so the host-side tensor dtype decides; the NVIDIA build then compiles to identical PTX (verified by stripping debug sections and comparing).

## Review 2026-11-23 (reviewer, linux-gfx950, job 153422)

Port Review: Matrix-Game (`moat-port` at `6d94b2f` vs upstream `main` `51913b2`, three commits). Summary: the port makes the rtx5090/rtx5080/rtx5070ti presets run on ROCm (per-arch e4m3 format, ROCm MX GEMM layout with BF16 fallback off gfx950, portable libdevice `tanh`) and documents the ROCm install. Verdict: **Approve**, no findings.

How each finding candidate was settled (fact-checked against the code and on the GPU, not taken from the porter's notes):

- **NVIDIA path unchanged (Python).** Every new branch is behind `torch.version.hip`: `e4m3_format()` returns `(float8_e4m3fn, 448.0)` without querying the device on CUDA; `quantize_mxfp8` keeps `padded = rows`, the 128-row scale buffer and `ROW_MAJOR_SCALES=False` (the branch is constexpr-eliminated); `MXFP8Linear.project` keeps `SWIZZLE_32_4_4` and the bias in the GEMM; `mxfp8_supported()` is `True` on CUDA.
- **NVIDIA path unchanged (Triton, the `tanh` spelling and the FP8 `MAX`/`element_ty` changes).** PyPI `triton==3.7.0` (the version torch 2.12.0's CUDA wheel pins; NVIDIA backend and its bundled ptxas, no GPU needed) compiled the five changed kernel instances (`_scale_quantize_rows` at K 3072 and 14336, `_scale_quantize_rows_padded`, `_bias_transpose_kernel` with GELU, `_mxfp8_quantize_kernel`) from both trees for sm_90, sm_100 and sm_120. With `TRITON_DISABLE_LINE_INFO=1` the PTX, the cubin and the `cuobjdump -sass` text are byte-identical between upstream and the port for all 15 pairs. With line info on, the SASS is still identical and only the PTX/cubin debug sections differ (source paths and line numbers). Script `scripts/cuda_ptx_check.py`, log `data/cuda_ptx_check.log` (job directory, not durable).
- **Why the `tanh` change is needed.** On gfx950 with the job's Triton 3.8 ROCm build, a kernel calling `tl.extra.cuda.libdevice.tanh` fails: `Implicit conversion of CUDA __nv_tanhf device function has been dropped`. `triton.language.extra.libdevice.tanh` runs, max error 1.2e-7 against `torch.tanh`.
- **What each AMD target gets.** gfx950: OCP e4m3fn/448 for FP8 (checked `e4m3_format()` returns that), MXFP8 through the ROCm MX GEMM. gfx94x: `float8_e4m3fnuz`/240 (substring `gfx94`), MXFP8 roles in BF16. gfx12, gfx1250 and everything else: e4m3fn/448, MXFP8 roles in BF16 (`startswith("gfx950")`). GPUs with no FP8 GEMM (gfx90a, RDNA3) fail rtx5090/rtx5080 inside `torch._scaled_mm`, as the README says ("need a GPU with FP8 matrix multiplication"); rtx5070ti there runs INT8 plus BF16. Only gfx950 is executed; the gfx942 and gfx12 cells stay unproven as the porter recorded.
- **The ROCm MX GEMM constraints, re-measured** (torch 2.12.0+rocm10.2.0a20261015, `data/probe.log`): `SWIZZLE_32_4_4` refused; FP32 output refused; `bias=` gives `HIPBLAS_STATUS_INVALID_VALUE` ("rocRoller does not support bias"); M=405 refused even with scales padded to 416 rows ("Matrix dimensions must be multiples of 32"), so the activation rows do have to be padded. Port's quantizer and `MXFP8Linear` at M = 1, 37, 405, 4096: rel err 0.038 to 0.039 against BF16, finite, right output shape; padded weight rows (N=3080) also work. FNUZ `_scaled_mm` on gfx950 has no hipBLASLt solution, so gfx950 must not take the FNUZ branch, and it does not.
- **BF16 fallback.** `build_linear("mxfp8")` returns `_Fused(linears, fp8=False)`, the same plain-`nn.Linear` path the upstream `bf16` format uses, so it needs no new code to be right; the validator exercises it by forcing `mxfp8_supported()` off.
- **`-sage` refusal path.** No refusal was added. That follows the Q2-B ruling and plan item 5. On ROCm the three `-sage` presets reach upstream's own `ImportError("this runtime requires SageAttention 2.2 built with ...sageattention.patch")` from `rtx5090_sage/sage_attention.py`. For rtx5090-sage and rtx5080-sage that happens after the DiT is loaded and quantized. For rtx5070ti-sage it happens at the first `init_kv_cache` (the `from .sage import SageWindow` there). CUDA without SageAttention behaves the same way, and the README's AMD section says the `-sage` presets need SageAttention, which has no ROCm build. The validator records the observed exit for each of the three.
- **Not findings, recorded for the record.** On ROCm, an `mxfp8` role whose out-features are not a multiple of 32 would keep the padded weight rows as extra output columns, and the `.view` would fail. No role of this model hits that (3072, 14336), and `RuntimeConfig` only reaches it through a hand-edited config. The MXFP8 bias is added after the GEMM in BF16 (one extra rounding against the fused epilogue), as the commit says.
- **Commit hygiene.** Titles are 38/52/54 characters. `[ROCm]` is on the two ROCm commits and absent from the shared `tanh` fix. AI disclosure is present, there is no `Co-Authored-By`, and the Test Plans are fenced. `python3 utils/jargon.py --port Matrix-Game`: clean.
- **Lessons.** Both lessons check out against the measurements above: the MX layout, M padding, bias and output dtype, and FNUZ on gfx942 versus e4m3fn elsewhere for the first; the libdevice refusal and identical NVIDIA output (PTX/SASS) for the second. One wording point for the promoting session: in the MX lesson, "M, N and K must all be multiples of 32" was measured for M only. N was padded by the quantizer, and K is multiple-of-128 by construction. The torch error names "matrix dimensions" in general.

## Correction and rulings (Jeff Daily, 2026-11-23, later the same day)

- The `windows` waiver recorded in status.json (approved by Jeff Daily, 2026-11-12) is his and stands. The sentence in the rulings section above saying the waiver was "not approved or declined yet" was wrong (written by the dispatching session).
- The SageAttention check result was reported to Jeff; the `-sage` presets stay deferred.
- Porter questions 1 and 2 (the PR #67 comment and the `--fa_version 0` upstream bug): the draft was shown to Jeff; not ruled yet. Nothing is posted without his approval.
## Validation 2026-11-24 (validator, linux-gfx950, job 153422)

Head `6d94b2fb5ccdd1351b3ce5b5242cadda5f367f70`, AMD Instinct MI355X (gfx950), one GPU (`HIP_VISIBLE_DEVICES=7`, the dispatcher's pin), `ulimit -c 0`, `timeout` on every run. Everything was rebuilt and re-run in this job's own directory. Nothing was reused from the porter's job.

### Environment

- `$FLEET_VENV` (Python 3.11): `torch 2.12.0+rocm10.2.0a20261015`, torchvision 0.27.0, `triton 3.8.0+git669b31ac.rocm10.2.0a20261015`, `rocm[devel]==10.2.0a20261015`. Index `https://nightly.repo.amd.com/rocm/whl-next/`, installed with `torch[device-gfx950]`. HIP 7.17.26412. MG-4: `pip install -e '.[inference,server,download,test]'` (transformers 5.17.0). MG-3 extras: `easydict decord opencv-python==4.10.0.84 trimesh pandas==2.2.3`. MG-3 again ran on torch 2.12 rather than its pinned 2.10 (same deviation as the porter: TheRock 10.2 has no torch 2.10).
- flash-attention `v2.8.3.post2` (`49774b31`), two builds:
  - CK: `BUILD_TARGET=rocm GPU_ARCHS=gfx950 OPT_DIM=128 MAX_JOBS=42 pip wheel --no-build-isolation --no-deps .` with `ROCM_HOME` = `rocm-sdk path --root`. 704 objects, about 35 minutes.
  - Triton AMD backend: `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, a pure-Python wheel; the same variable is needed at run time.
  - Standalone varlen check against SDPA (lens 1000+777, H 24, D 128, bf16): CK max abs err 2.0e-3.
- AITER `v0.1.30` (`77ba72b7`): `GPU_ARCHS=gfx950 PREBUILD_KERNELS=0 pip wheel --no-build-isolation --no-deps .`. Same check against SDPA: max abs err 0.
- **Triton swap during the AITER build (caught and redone).** AITER's `setup.py` uninstalled the venv's TheRock triton while its wheel built (`Successfully uninstalled triton-3.8.0+git669b31ac.rocm10.2.0a20261015`) and installed `triton-kernels` from pypi.amd.com. Python then fell back to the base image's `triton 3.8.0+amd.rocm7.2.0.git7cb7b059` through the fleet venv's base-env links. The base env itself was not modified. With that triton, flash-attention's Triton backend failed at import (`ValueError: key contains names that are not kernel arguments` in `bwd_prefill_onekernel.py`).
  - Fix: `pip install --no-deps triton==3.8.0+git669b31ac.rocm10.2.0a20261015` from the nightly index. Every GPU run reported below was redone afterwards, and each phase's log prints the triton path in use.
  - The first runs under the wrong triton are kept as `out-basetriton`. Realtime under the wrong triton is bit-identical to realtime under the right one.
- `flash_attn_interface` is not importable (MG-3's FA3 path cannot trigger; plan risk 3 does not occur).
- Weights in the job directory (680 GB free before download):
  - MG-4 `Skywork/Matrix-Game-4.0` revision `2c53100457da7abcf0e966f75637c644d3423a63`, scene ancient-egypt, models plus game-assets manifests, every file SHA256-verified by `matrix-game download` (23 GB plus 23 GB HF cache).
  - MG-3 `Skywork/Matrix-Game-3.0` at `d4c84ef2b8e9028a00a5d58b6be2df04f1cb6042` (53 GB).

### Non-GPU tests

`python -m unittest discover -s tests -p 'test_*.py'` in Matrix-Game-4: **81 tests OK** on `moat-port` and 81 OK on upstream `51913b2`. No regression.

### CUDA gate

There is no CUDA C++ in the change; the shared code it touches is Triton. Done in the review above, at this head: PyPI `triton==3.7.0` (the CUDA torch 2.12.0 pin) compiled the five changed kernel instances for sm_90, sm_100 and sm_120 from upstream and from the port. With `TRITON_DISABLE_LINE_INFO=1` the PTX, cubin and SASS are byte-identical (15/15). nvcc itself has nothing to compile here.

### MG-4 end to end (gfx950)

- Command: `matrix-game render --scene ancient-egypt --resources <res> --input egypt-pan-120.mp4 --prompt-mode fixed --seed 42 --preset <p> [--dit-runtime native --no-vae-graph]`.
- Input: a 120-frame lossless pan/zoom over `models/ancient-egypt/sink.png` (sha256 `94e03c1a...45f6`); the clip is sha256 `a9f744dd...d9ce`. That is 30 latent chunks, so the 21-chunk window evicts.
- Every run below: rc 0, 120/120 frames at 864x480, no NaN in the logs, minimum per-frame std 44 or more.
- FPS is the CLI's "processing FPS" at frame 101; peak memory is `torch.cuda.max_memory_allocated`.

**What the frame-wise comparison is against:** no CUDA reference video exists on this host, so nothing below is compared to NVIDIA output. The reference is upstream's own release gate (docs/testing.md): the bf16 `native` runtime (`--dit-runtime native --no-vae-graph`, SDPA) on the same MI355X, same clip, same seed. The model is autoregressive, so any numerical difference grows along the clip. Each row gives mean and minimum PSNR, plus the first-10 and last-10 frame means, which show that growth. Run-to-run noise is zero: realtime repeated (`realtime-rep`) is bit-identical, and rtx5070ti with and without flash-attn installed is bit-identical. So every number below is a real numerical difference between paths, not noise.

| run (SDPA unless noted) | FPS | peak alloc GiB | PSNR vs native mean / min dB | first10 / last10 dB |
|---|---|---|---|---|
| native (reference) | 31.5 | 20.2 | -- | -- |
| realtime | 40.0 | 20.2 | 22.21 / 19.60 | 33.1 / 21.0 |
| realtime, repeat | 38.8 | 20.2 | identical to realtime (bit-exact) | |
| rtx5090 (FP8) | 52.8 | 15.65 (18.1 reserved) | 19.27 / 17.18 | 25.0 / 18.4 |
| rtx5080 (FP8, padded/transposed) | 61.4 | 10.90 | 19.95 / 17.80 | 25.7 / 19.3 |
| rtx5070ti (INT8 + MXFP8 `cross_q`) | 62.4 | 10.90 | 18.93 / 16.79 | 26.4 / 17.8 |
| rtx5070ti, MXFP8 forced to the BF16 fallback (what non-gfx950 AMD GPUs run) | 57.6 | 10.90 | 19.95 / 17.94 | 28.0 / 19.0 |
| native, FA2 CK | 25.1 | 20.2 | 21.22 / 18.23 | 31.1 / 19.9 |
| native, FA2 Triton | 28.6 | 20.2 | 21.78 / 18.63 | 31.6 / 20.4 |
| realtime, FA2 CK | 31.1 | 20.2 | 20.09 / 17.83 vs native-CK | 30.0 / 19.0 |
| realtime, FA2 Triton | 9.7 | 20.2 | 21.48 / 18.89 vs native-Triton | 31.4 / 19.9 |

Against bf16 realtime instead of native: rtx5090 19.19, rtx5080 19.73, rtx5070ti 18.67, rtx5070ti-BF16-fallback 19.77 dB mean. rtx5070ti against its own BF16 fallback: 19.47 dB.

| quality path (Wan VAE, native DiT, flex_attention) | FPS | peak alloc GiB | PSNR vs quality-SDPA mean / min dB |
|---|---|---|---|
| quality, SDPA | 18.0 | 42.05 | -- |
| baseline, SDPA | 17.6 | 42.05 | 21.81 / 19.09 |
| quality, FA2 CK | 15.1 | 42.05 | 19.51 / 17.21 |
| quality, FA2 Triton | 14.6 | 42.05 | 19.66 / 17.21 |

Reading:
- The three RTX presets sit 2 to 3 dB below bf16 realtime against the same reference, and start 25 to 28 dB from native instead of 33. That is the cost of quantization, not of a defect.
- The MXFP8 role costs about 1 dB against its BF16 fallback. A layer-level check shows the ROCm MX GEMM computes the intended MXFP8 function: `MXFP8Linear` output against an exact software dequantize (e4m3 x 2^(e8m0-127), FP32 matmul, + bias) has rel 4.0e-3 and max error at most one BF16 ulp, at M = 1, 37, 405 and 4096 (`mx_emul.py`). So the drop is quantization error that CUDA's MXFP8 path would share, not a layout error.
- Contact sheet `data/montage-mg4.png`: every preset gives a coherent scene that follows the input pan, with no corruption, smearing or colour faults.
- CUDA-graph capture works with SDPA, FA2 CK and FA2 Triton.
- FA2 Triton is slow under the realtime runtime (9.7 FPS) but correct. That is a performance observation about the flash-attention Triton backend, not a port issue.

**`-sage` refusal path.** `--preset rtx5090-sage`, `rtx5080-sage` and `rtx5070ti-sage` each exit 1 and write no output. The error is `ImportError: this runtime requires SageAttention 2.2 built with matrix_game/inference/rtx5090_sage/sageattention.patch` (cause `ModuleNotFoundError: No module named 'sageattention'`), after 20 s, 56 s and 52 s (after the DiT load). This matches the review's reading and the README's AMD note.

**Kernel trace** (`rocprofv3 --kernel-trace --stats`, 24-frame clip, CK flash-attn and AITER installed):
- rtx5070ti: rocRoller `RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2` (2574 calls), `_mxfp8_quantize_kernel` (2604), `_int8_gemm_kernel` (12876), `_quantize_rows_kernel` (12876).
- rtx5080: hipBLASLt `Cijk_..._F8SS_...` FP8 GEMMs (15582 + 2574 + 2574), `_scale_quantize_rows_padded` (15570), `_bias_transpose_kernel` (5148).
- Both: `aiter::fmha_fwd_hd128_bf16` (5148) for attention.

### MG-3 end to end (gfx950)

`generate.py --size 704*1280 --ckpt_dir Matrix-Game-3.0 --use_int8 --num_iterations 4 --num_inference_steps 3 --image demo_images/001/image.png --prompt "<README prompt>" --seed 42 --vae_type mg_lightvae --lightvae_pruning_rate 0.5`. Each run produces 177 frames at 1280x704. No CUDA reference here either: the comparison is FA2 CK against the other backends on the same GPU.

| code | attention installed | result | wall | PSNR vs FA2 CK mean / min dB |
|---|---|---|---|---|
| upstream `main`, `--fa_version 0` | none | `AssertionError` at `wan/modules/attention.py:114` (porter question 2 reproduced) | 58 s | -- |
| upstream `main`, `--fa_version 2` | FA2 CK | OK | 130 s | reference |
| upstream `main`, `--fa_version 2 --compile_vae` | FA2 CK | OK | 86 s | 59.2 / 33.3 |
| upstream `main`, `--fa_version 2` | FA2 Triton | OK | 90 s | 24.8 / 15.4 |
| PR #67 head `9ad32b1d` | AITER only | OK; banner prints `Actual: None (SDPA)` | 94 s | 24.8 / 13.7 |
| PR #67, `--compile_vae` | AITER only | OK; same banner | 89 s | 24.8 / 13.7 |
| PR #67 | AITER + FA2 CK | OK; AITER is used for self/cross attention, banner prints `Actual: Flash Attention 2` | 93 s | 25.8 / 15.4 |

The last-10-frame PSNR between any two backends is 14 to 16 dB. That is autoregressive divergence over 4 iterations, and it hits FA2 Triton against CK exactly as hard as AITER against CK. PR #67 with FA2 also installed differs from AITER-only (24.8 dB) because `wan/modules/action_module.py` takes flash-attn's `flash_attn_func` whenever flash-attn imports. Contact sheet `data/montage-mg3.png`: all four backends give the same scene and camera path, with the late frames differing only in detail (building texture). Multi-GPU torchrun was not run: one GPU.

### Not run, and why

- The documented install line (`download.pytorch.org/whl/rocm7.2`, `torch==2.12.0+rocm7.2`, cp312; the wheel exists on the index, checked) was not installed: this job was told to use the TheRock 10.2 wheels.
- gfx942 (FNUZ branch) and gfx12 remain unexecuted, as the porter recorded. The wave32 gate needs its own validator. gfx942 evidence would need an MI300 host.
- Matrix-Game-4's racing scene with `--prompt-mode dynamic` (plan step 7) was not run.

### Questions for Jeff (nothing assumed)

The porter's questions 1 (comment on PR #67) and 2 (the `--fa_version 0` upstream bug) are still open; question 3 (the windows waiver) was answered in commit `3c6aa9eb`. One correction to the PR #67 draft follows from the runs above. Point 2 says the banner prints `Actual: None (SDPA)` while AITER is in use. That is true only when flash-attn is absent; with flash-attn also installed it prints `Actual: Flash Attention 2` while AITER runs. Revised draft (options as before: post as drafted, post a shorter results-only version, or post nothing yet; NOT posted):

====
My agent came up with this analysis, but I've read and approved it.

> Tested this PR's head (9ad32b1d) on an AMD Instinct MI355X (gfx950) with a ROCm build of PyTorch 2.12 and AITER built from source (v0.1.30). `python generate.py --size 704*1280 --use_int8 --num_iterations 4 --num_inference_steps 3 --vae_type mg_lightvae --lightvae_pruning_rate 0.5 --image demo_images/001/image.png --seed 42` completes with and without `--compile_vae` (177 frames), with AITER alone and with flash-attention also installed. The output agrees with the flash-attention 2.8.3 paths on the same GPU (CK and Triton builds) to the level any two of those backends agree with each other, with no visible difference. AITER's varlen call also matches SDPA exactly on a standalone check.
>
> Three small things:
> 1. `pip install aiter` in the README and in the new RuntimeError installs an unrelated PyPI package (async iterator helpers). The AMD package is `amd-aiter`, or the source build from ROCm/aiter that the README also lists.
> 2. The "Flash Attention Configuration" banner in `pipeline/inference_pipeline.py` and `pipeline/inference_interactive_pipeline.py` does not know about AITER: with AITER alone it prints `Actual: None (SDPA)`, and with flash-attn also installed it prints `Actual: Flash Attention 2`, while AITER is the backend in use in both cases.
> 3. Unrelated to this PR: `--fa_version 0` does not give an SDPA run, because `wan/modules/model.py` calls `flash_attention()` directly, so without flash-attn or AITER it still fails at the assert.
====

Verdict for linux-gfx950: **pass**. Unit tests 81/81. All six generic and RTX presets plus `baseline` render end to end with SDPA, FA2 CK and FA2 Triton. The three `-sage` presets refuse with upstream's message. MG-3 runs through FA2 (CK and Triton) and through PR #67 + AITER. The tree is clean at `6d94b2f`, and `jargon.py --port Matrix-Game` is clean.

## Lesson for cuda-to-rocm: building AITER from source replaces the environment's Triton

For `libraries.md` (or wherever AITER is covered). ROCm/aiter's `setup.py` (seen at v0.1.30, torch 2.9.1 or newer) uninstalls whatever Triton is installed (`triton`, `pytorch-triton-rocm`, `triton-rocm`, ...) and installs its own pin from pypi.amd.com, even under `pip wheel --no-deps`. A ROCm environment whose PyTorch was built against a specific Triton then silently runs on a different one. Here, flash-attention's Triton AMD backend broke at import (`ValueError: key contains names that are not kernel arguments`). Set `AITER_USE_SYSTEM_TRITON=1` when building AITER to keep the installed Triton. Then check `python -c "import triton; print(triton.__version__, triton.__file__)"` before trusting any later run.

## Validation 2026-11-24 (validator, linux-gfx1100, job rx7900-20261123-164859)

Head `6d94b2fb5ccdd1351b3ce5b5242cadda5f367f70`, AMD Radeon Pro W7800 48GB (gfx1100, wave32; 44.98 GiB visible to torch), one GPU (`HIP_VISIBLE_DEVICES=1`, the dispatcher's pin), `ulimit -c 0`, `timeout` on every GPU run. Recipe of the gfx950 validation above, rebuilt from scratch in this job's directory.

### Environment

- `$FLEET_VENV` (Python 3.11): `torch 2.12.0+rocm10.2.0a20261015` (the same build as gfx950), torchvision 0.27.0, `triton 3.8.0+git669b31ac.rocm10.2.0a20261015`, `rocm[devel]==10.2.0a20261015`, installed with `pip install --pre --index-url https://nightly.repo.amd.com/rocm/whl-next/ 'torch[device-gfx1100]==2.12.0+rocm10.2.0a20261015' torchvision==0.27.0 'rocm[devel]==10.2.0a20261015'`. HIP 7.17.26412. These shadow the image's ROCm 10.1 base links; `torch`, `triton` and `rocm_sdk` all resolve inside the venv (checked).
- MG-4: `pip install -e '.[inference,server,download,test]'` (transformers 5.17.0). MG-3 extras: `easydict decord opencv-python==4.10.0.84 trimesh pandas==2.2.3`. MG-3 again on torch 2.12 instead of its pinned 2.10 (same deviation as gfx950).
- flash-attention `v2.8.3.post2` (`49774b31`), Triton AMD backend only (the RDNA route; the CK build's `setup.py` allows only gfx90a, gfx942 and gfx950): `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE pip wheel --no-build-isolation --no-deps .`, and the same variable at run time. Standalone check against SDPA (bf16, H 24, D 128): varlen 1000+777 max abs err 2.0e-3, causal S 1560 3.9e-3.
- AITER and PR #67 were not built or run here (see "Not run").
- Weights in the job directory (2.3 TB free before download): MG-4 `Skywork/Matrix-Game-4.0` revision `2c53100457da7abcf0e966f75637c644d3423a63`, ancient-egypt, models plus game-assets manifests, every file SHA256-verified by `matrix-game download` (23 GB plus 23 GB HF cache); MG-3 `Skywork/Matrix-Game-3.0` at `d4c84ef2b8e9028a00a5d58b6be2df04f1cb6042` (53 GB).

### What gfx1100 can multiply

Probe in the installed torch: per-tensor `torch._scaled_mm` with `float8_e4m3fn` and with `float8_e4m3fnuz`, and `F.scaled_mm(..., ScalingType.BlockWise1x32, NO_SWIZZLE)`, all refuse with `torch._scaled_mm is only supported on CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+` (RuntimeError for per-tensor, NotImplementedError for block-wise). The port's selectors answer `e4m3_format() == (float8_e4m3fn, 448.0)` and `mxfp8_supported() == False` here, as the review predicted.

Per-layer check of the rtx5070ti linears (`check_linears.py`, D 3072, FFN 14336, inputs with 20x outlier columns, rel err against the BF16 `nn.Linear`): `Int8Linear` fused qkv 0.046 to 0.048, INT8 tanh-GELU `ffn_in` 0.047 to 0.050, INT8 K=14336 0.058 to 0.061, at M = 1, 37, 405, 4096, all finite; the `mxfp8` role builds `_Fused(fp8=False)` (plain BF16), rel err 0. The INT8 Triton GEMM (upstream's NVIDIA tile configs, up to 64x256x64 with 3 stages) compiles and runs on gfx1100 and matches an exact emulation of the same function (quantized operands, float64 integer accumulation, scales, bias, GELU) to BF16 output rounding: rel 1.6e-3 to 2.5e-3, max one BF16 ulp per element (`int8_emul.py`); the row quantizer differs from a torch reference by at most one code (rounding ties). So the 0.05 above is INT8 quantization error, not a kernel or wave32 fault. The plan's LDS risk for these tiles does not occur on gfx1100.

### Non-GPU tests

`python -m unittest discover -s tests -p 'test_*.py'` in Matrix-Game-4: **81 tests OK** on `moat-port` and 81 OK on upstream `51913b2` (worktree, `PYTHONPATH` pointing at it). No regression.

### CUDA gate

Not rerun: recorded at this head in the review (Triton NVIDIA backend, PTX/cubin/SASS byte-identical 15/15). No CUDA C++ in the change.

### MG-4 end to end (gfx1100)

- Command: `matrix-game render --scene ancient-egypt --resources <res> --input egypt-pan-120.mp4 --prompt-mode fixed --seed 42 --preset <p> [--dit-runtime native --no-vae-graph]`, through a wrapper that prints `torch.cuda.max_memory_allocated`/`reserved` at exit.
- Input: 120-frame lossless (libx264rgb crf 0) pan/zoom over `models/ancient-egypt/sink.png` (sha256 `94e03c1a...45f6`, the same image as gfx950), 864x480 at 30 fps; clip sha256 `869278df...121b`. The generator differs from gfx950's, so PSNR values compare runs on this host only, not across hosts.
- FPS is the CLI's "processing FPS" at frame 101. "Render s" is `render.json` `seconds` for the 120 frames (4 s of video).
- Reference as on gfx950: upstream's release gate, bf16 `native` with SDPA on the same GPU. Run-to-run noise is zero: native twice and realtime twice are each bit-identical. So every PSNR below is a real numerical difference between paths.

| preset (SDPA unless noted) | result | FPS | render s | peak alloc GiB (reserved) | PSNR vs native mean / min dB | first10 / last10 dB |
|---|---|---|---|---|---|---|
| native (reference) | 120/120 | 10.1 | 12.8 | 20.21 (20.26) | -- | -- |
| realtime (ring-fused, VAE graph) | 120/120 | 10.1 (repeat 10.6) | 12.7 | 20.21 (20.26) | 34.40 / 26.90 | 37.8 / 31.1 |
| quality (Wan VAE, two steps) | 120/120 | 0.84 | 129.3 | 28.41 (29.84) | 19.28 / 16.71 | 21.6 / 16.9 |
| baseline | 120/120 | 2.78 | 44.6 | 20.21 (21.21) | bit-identical to quality | |
| rtx5070ti (INT8 plus BF16) | 120/120 | 15.1 | 8.5 | 10.90 (10.93) | 31.20 / 21.97 | 34.5 / 28.7 |
| rtx5090 | exit 1, no output, after 78 s | -- | -- | 15.65 (18.09) at failure | -- | -- |
| rtx5080 | exit 1, no output, after 69 s | -- | -- | 10.90 (10.93) at failure | -- | -- |
| rtx5090-sage / rtx5080-sage / rtx5070ti-sage | exit 1, no output, after 26 / 67 / 64 s | -- | -- | -- | -- | -- |
| native, FA2 Triton | 120/120 | 2.83 | -- | 20.21 | 33.69 / 27.49 | 37.4 / 30.9 |
| realtime, FA2 Triton | 120/120 | 11.4 | -- | 20.21 | 33.36 / 29.01 (33.65 vs native-FA2) | 37.5 / 29.5 |
| quality, FA2 Triton | 120/120 | 2.50 | -- | 20.21 (21.21) | 33.05 / 26.77 vs quality-SDPA | 38.4 / 27.3 |
| rtx5070ti, FA2 Triton installed | 120/120 | 13.7 | -- | 10.90 | bit-identical to rtx5070ti SDPA | |

Every completed run: rc 0, `render.json` status complete, 120 frames at 864x480, no NaN in the logs, per-frame std 45.7 or more (input 47.3). rtx5070ti against bf16 realtime: 30.96 / 21.92 dB.

What each preset does on this GPU:
- **Generic presets.** `realtime`, `native`, `quality` and `baseline` all fit in 48 GB and render all 120 frames. Peak use is 20.2 GiB for realtime and native, and up to 28.4 GiB for quality.
- **`quality` vs `baseline`.** In the render CLI the two resolve to identical settings (`scenes.resolve_model_arguments`: Wan VAE, native DiT, steps 1000,500), so the bit-identical output is expected. gfx950 measured 21.8 dB between them, which these runs do not explain. `quality`'s 129 s against `baseline`'s 45 s, and its higher reserved peak, come from MIOpen's first-use tuning of the Wan VAE convolutions: its find-db `gfx110046.HIP.3_6_2_3dc098eb.ufdb.txt` was written at 01:09:02, inside the quality run (01:07:28 to 01:09:47).
- **`rtx5070ti` runs its INT8 plus BF16 path** (INT8 Triton GEMMs, `cross_q` in BF16 because `mxfp8_supported()` is False). It is the fastest preset on this card: 15.1 FPS against 10.1 for realtime/native, at 10.9 GiB. So wave32 does have a fitting fast preset. It sits 3.2 dB below bf16 realtime against native, the cost of INT8 (gfx950 saw 2 to 3 dB for the quantized presets). None of the presets reaches the clip's 30 fps on this card.
- **`rtx5090` and `rtx5080` fail the way the README says** ("need a GPU with FP8 matrix multiplication"). Each loads and quantizes the DiT, then exits 1 with no output at the first FP8 GEMM while refreshing the text K/V: `RuntimeError: torch._scaled_mm is only supported on CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+`. For rtx5090 the call is `rtx5090/fp8_linear.py:150 _gemm`; for rtx5080 it is `rtx5080/linears.py:90 _padded_gemms`. The message is PyTorch's own and names no preset. A user gets a traceback rather than a Matrix-Game message, which matches what the README promises but is not friendly.
- **The three `-sage` presets exit 1 with upstream's message** `ImportError: this runtime requires SageAttention 2.2 built with matrix_game/inference/rtx5090_sage/sageattention.patch` (cause `ModuleNotFoundError: No module named 'sageattention'`). rtx5090-sage fails at transformer load (26 s); rtx5080-sage and rtx5070ti-sage fail after the DiT load (67 s, 64 s; rtx5070ti-sage at `init_kv_cache`). This matches gfx950 and the review.
- **FA2 Triton.** Correct under every runtime, with the same 33 to 34 dB from native as SDPA paths show among themselves. It is slow under `native` (2.8 FPS against 10.1 with SDPA) and slightly faster under `realtime` (11.4). The rtx5070ti runtime does not call flash-attn: its output is bit-identical with or without it installed.
- **Contact sheets** (`montage-mg4-sdpa.png`, `montage-mg4-fa.png`, job directory): every completed run gives a coherent scene that follows the input pan and zoom. There is no corruption, smearing or colour fault, and runs differ only in transient projectile effects.
- **CUDA graphs.** Capture works (realtime and rtx5070ti run with VAE graph and DiT graphs).

### MG-3 end to end (gfx1100): does not run on this card

`generate.py --size 704*1280 --ckpt_dir Matrix-Game-3.0 --use_int8 --num_iterations 4 --num_inference_steps 3 --image demo_images/001/image.png --prompt "A colorful, animated cityscape with a gas station and various buildings." --seed 42 --vae_type mg_lightvae --lightvae_pruning_rate 0.5`, upstream `51913b2` (the port does not touch Matrix-Game-3). `704*1280` is the only size `generate.py` accepts (`wan/configs MAX_AREA_CONFIGS`).

| run | result | wall | peak alloc GiB |
|---|---|---|---|
| `--fa_version 0`, flash-attn not importable | `AssertionError` at `wan/modules/attention.py:114` (porter question 2 reproduced on gfx1100) | 78 s | 27.17 |
| `--fa_version 2`, FA2 Triton | DiT denoising of iteration 1 completes (3/3 steps, INT8 GEMMs JIT fine); VAE decode fails: `Error in stream_decode: CUDA error: profiler is not initialized`, then `AttributeError: 'NoneType' object has no attribute 'cpu'` at `pipeline/inference_pipeline.py:652` | 157 s | 27.72 |
| same, `--compile_vae` | same failure | 126 s | 27.72 |
| FA2 Triton, VAE attention forced to the MATH SDPA backend (harness wrapper, no source edit) | decode OOM: `CUDA out of memory. Tried to allocate 18.85 GiB ... total capacity of 44.98 GiB of which 16.45 GiB is free` | 156 s | 27.72 |
| same, attention also chunked over frames; then every Conv3d input and 5-D pad output made `channels_last_3d` | same 18.85 GiB OOM each time | 95 to 150 s | 28.22 |

Two separate obstacles, neither in the port and neither specific to MG-3's attention backend:

1. **PyTorch ROCm SDPA fault on gfx1100.** The decode error comes from the plain `F.scaled_dot_product_attention` in the Wan2.2 VAE `AttentionBlock` (`wan/modules/vae2_2.py:420`). That block is single-head, with head dim 512 in the MG-LightVAE decoder and 640 in the teacher encoder, over 44x80 = 3520 tokens. Standalone (`sdpa_hd.py`, `sdpa_one.py`, fresh process per case), both fused backends, `FLASH_ATTENTION` and `EFFICIENT_ATTENTION`, accept head dims 288, 320, 384 and 512 and then fail at launch with that misleading error. It happens at any length (64, 1024, 3520) and in bf16, fp16 and fp32, and it is sticky, so the process is lost. Head dims 64 to 256 work (max err against MATH up to 9.8e-4). At 1024 both fused backends correctly say "No available kernel" and the default falls back to MATH. MATH works at every size. Registered as deferral `pytorch-rocm-sdpa-gfx1100-headdim-257-512`. Not checked on a newer torch: the job is limited to its one venv.
2. **Memory.** With the attention routed to MATH, the decoder's upsample `CausalConv3d` (input `(1, 512, 13, 176, 320)` bf16, weight `(256, 512, 3, 3, 3)`, `vae2_2.py:207`) asks for 18.85 GiB. That is exactly an im2col buffer (512x27 x 13x176x320 x 2 bytes), on top of 26.7 GiB already allocated. Standalone (`conv3d_ws.py`), the same conv through MIOpen with NCDHW input takes 19.20 GiB beyond its inputs at T=13 (5.91 GiB at T=4); PyTorch's native conv with cudnn off takes 19.28 GiB. With `channels_last_3d` input it takes only 2.32 GiB (0.87 at T=4). Forcing channels-last inside the model did not change the request, so the in-model call still takes the large path, for a reason not found here. Stopped after these attempts (stop discipline). MG-3's decode at its only resolution does not fit this 48 GB card with the stock conv path. gfx950 (288 GB) never sees this.

So MG-3 on gfx1100: denoising runs, decode does not, and no video is produced.

### Not run, and why

- PR #67 / AITER: not built. The request named flash-attention's Triton backend as the RDNA route, and AITER's source build replaces the environment's Triton (lesson above). MG-3 cannot decode here regardless of the attention backend, so a PR #67 run could not produce a video.
- The documented install line (`download.pytorch.org/whl/rocm7.2`) was not installed: this job was told to use the TheRock 10.2 wheels.
- No kernel trace (rocprofv3) on this host. That rtx5070ti takes the BF16 path for `cross_q` is shown by `build_linear` returning `_Fused` and by `mxfp8_supported() == False`, not by a trace.

### Questions for Jeff (nothing assumed)

1. **MG-3 on RDNA3.** Upstream Matrix-Game-3 does not produce a video on a 48 GB gfx1100. The VAE decode first hits the PyTorch SDPA fault (head dim 512), then needs 18.85 GiB more than the card has. The port does not touch MG-3, so this validation records it and passes the platform on the MG-4 evidence. Options:
   - (a) leave MG-3 as an RDNA3 gap, documented here;
   - (b) a porter round adding a ROCm workaround in MG-3's VAE (route head dims over 256 to MATH, and a lower-memory decode), which is a scope addition you would rule;
   - (c) take the SDPA defect upstream to PyTorch (deferral registered; filing is your call).
2. Porter questions 1 (PR #67 comment) and 2 (`--fa_version 0` upstream bug) are still open. Nothing from this run changes the drafts; the `--fa_version 0` assert reproduces on gfx1100 too.

Verdict for linux-gfx1100: **pass** (wave32 gate).
- Unit tests 81/81, with no regression against upstream.
- `realtime`, `native`, `quality` and `baseline` fit and render all frames with SDPA and FA2 Triton.
- `rtx5070ti` runs INT8 plus BF16 and is the fastest preset on the card.
- `rtx5090` and `rtx5080` refuse with PyTorch's FP8 `_scaled_mm` message, as the README says.
- The three `-sage` presets exit with upstream's SageAttention message.
- MG-3 does not run on this card (above).
- The tree is clean at `6d94b2f`, and `jargon.py --port Matrix-Game` is clean.

## Lesson for cuda-to-rocm: ROCm fused SDPA on gfx1100 fails at head dims 257..512 with a bogus HIP error

For `diagnosis.md` (or wherever attention backends are covered). On gfx1100 with torch 2.12 (TheRock ROCm 10.2), `F.scaled_dot_product_attention` takes the AOTriton flash or memory-efficient kernel for head dims 257 to 512. The launch then fails with `CUDA error: profiler is not initialized` (`hipErrorProfilerNotInitialized`), a code that says nothing about attention, and the context is unusable afterwards. Head dims up to 256 and above 512 are fine; above 512 the dispatcher reports no kernel and falls back to MATH. VAEs with single-head spatial attention over the full channel count (Wan2.x VAE, head dim 512 or 640) hit it. To tell it from a real fault, rerun the call under `with sdpa_kernel([SDPBackend.MATH])`. Work around by routing such calls to MATH, chunked if the [L, L] matrix is large.

## Server session (validator, linux-gfx950, 2026-11-24)

Purpose: the owed surface entry `server:matrix_game.server`. Head `6d94b2fb5ccdd1351b3ce5b5242cadda5f367f70`, AMD Instinct MI355X (gfx950), one GPU (`HIP_VISIBLE_DEVICES=7`, the dispatcher's pin), `ulimit -c 0`, `timeout` on every run. Fork clone clean afterwards (`git status --porcelain` empty).

### Environment (all inside the job directory)

- `$FLEET_VENV` (Python 3.11) with `$FLEET_VENV/bin` first on PATH: `torch[device-gfx950]==2.12.0+rocm10.2.0a20261015`, `torchvision==0.27.0+rocm10.2.0a20261015`, `triton==3.8.0+git669b31ac.rocm10.2.0a20261015` from `https://nightly.repo.amd.com/rocm/whl-next/`; Matrix-Game-4 installed editable with `.[inference,server,download,test]` (aiohttp 3.14.3, transformers 5.17.0). SDPA attention (no flash-attn installed).
- Weights: the review/validation job's MG-4 resources (`Skywork/Matrix-Game-4.0` at `2c531004`, scene ancient-egypt), used read-only rather than downloaded again, because the host disk had 59 GB free at the start and later 0. `matrix-game verify --manifest releases/models.json` and `--manifest releases/game-assets.json` (`--scene ancient-egypt`) re-checked every file's SHA256 first.
- Native packages, no root and nothing installed into the image: `apt-get download` with a private `Dir::State::Lists` (the image's lists were stale and gave 404s), 125 Ubuntu 24.04 debs (`libsdl2-dev`, `libglew-dev`, `ffmpeg`, `python3-numpy`, `gir1.2-gstreamer-1.0`, `gir1.2-gst-plugins-base-1.0`, `gstreamer1.0-{tools,plugins-base,plugins-good,plugins-bad,plugins-ugly,rtsp,libav}` plus their closure, and `libgstreamer1.0-0`/`libgstreamer-plugins-base1.0-0` at the same version as the plugins), unpacked with `dpkg-deb -x` into a job sysroot. GStreamer 1.24.2.
- Game: upstream's `bash scripts/build_game.sh` with `CMAKE_PREFIX_PATH=<sysroot>/usr`, `CXXFLAGS=-I<sysroot>/usr/include/x86_64-linux-gnu` (SDL2's multiarch `_real_SDL_config.h`), `LDFLAGS=-Wl,-rpath-link,<sysroot>/usr/lib/x86_64-linux-gnu/pulseaudio` and `BUILD_JOBS=$MAX_JOBS`. It built `CastleReferenceServer` and `MedievalCastleBBoxMap`; **ctest 6/6 passed** (combat_cues, scene_config, road_surface, reference_pacing, camera_input, demo_media). No source change.
- Media sidecar: `STUDIO_WEBRTC_PYTHON` pointed at a wrapper that runs `/usr/bin/python3` (3.12, system `python3-gi`) with the sysroot's GI typelibs, GStreamer plugins and NumPy. `appsrc`, `videoconvert`, `x264enc`, `h264parse` and `rtspclientsink` are all present.
- MediaMTX v1.21.4 `linux_amd64` from its GitHub release, checked against the release's `checksums.sha256`, run with upstream's `configs/mediamtx.yml` unchanged.
- **OpenGL: llvmpipe, by necessity.** gfx950 has no graphics pipeline, so the game ran on Mesa's software device with upstream's test-only `--allow-software-rendering` (the server logs `Software OpenGL enabled for testing: llvmpipe (LLVM 20.1.2, 256 bits)`). The game's `HeadlessGL` takes the first EGL device that initializes, from at most 16. On this host Mesa lists 16 DRM devices first (GPU and XCP partition nodes) and the software device after them, so unmodified it would try other teams' GPUs and never reach llvmpipe. The game therefore ran through a wrapper `--game-root` whose `bin/CastleReferenceServer` sets `LD_PRELOAD` to a 30-line shim. The shim returns zero devices from `drmGetDevices2` and refuses any `/dev/dri` open, and it applies to the game process only, not to the model server. With it, Mesa offers one device (`EGL_MESA_device_software`) and the process holds no `/dev/dri` fd. Disclosure: before the shim existed, one probe program called `eglInitialize` once on EGL device 0 (`/dev/dri/card192`, an `amdgpu_xcp_55` partition node). radeonsi refused it, no work was submitted and no fd stayed open.

### What drove the sessions

`scripts/serve.sh --scene ancient-egypt --resources <res> --game-root <wrapper> --runtime <dir> --allow-software-rendering --profile-chunks --preset realtime --fps 24` (`STUDIO_WEBRTC_PYTHON=<wrapper>`, everything else as serve.sh sets it). Runtime config reported: TAEHV, 1 step, `ring-fused`, SDPA, VAE graph, T5 graph, early publish. Server ready (model loaded on the GPU, CUDA graphs captured) 45 to 49 s after launch.

The client is a script, not a browser. It does what the page does through the server's own API: `GET /api/status`, then `POST /api/start` (`{"mode": "reference"}`, `{"mode": "model", "monitor": true}` or `{"mode": "model"}`). It holds the controller WebSocket `/api/<id>/ws` and sends a fixed control loop at 20 Hz (8 s forward, 6 s strafe with a sinusoidal `look_x` pan, 6 s pan back while firing with one jump), all validated by `app.controls()`. It reads every frame-metadata and stats message, records the published H.264 path `native-<id>` from MediaMTX over RTSP (GStreamer `rtspsrc ! rtph264depay ! h264parse ! matroskamux`), then calls `POST /api/<id>/stop`. Session reports come from the server's own `runtime/session-<id>.json` and `profile-<id>.json`.

### Results

| run | session | frames | chunks | model FPS | online FPS | encoded / dropped | error |
|---|---|---|---|---|---|---|---|
| 1 | reference, 20 s | 349 | 87 | -- | 18.5 | 349 / 0 | none |
| 1 | model, dual view, 60 s | 993 | 248 | 62.6 | 16.9 | 991 / 0 | none |
| 1 | model, 60 s | 913 | 228 | 65.6 | 15.6 | 911 / 0 | none |
| 2 | reference, 15 s | 245 | 61 | -- | 17.9 | 244 / 0 | none |
| 2 | model, 60 s | 1233 | 308 | 58.4 | 21.2 | 1232 / 0 | none |
| 3 | model, dual view, 40 s | 765 | 191 | 56.3 | 19.7 | 763 / 0 | none |
| 4 | model, 40 s, under rocprofv3 | 785 | 196 | 61.0 | 20.6 | 786 / 0 | none |

- "Model FPS" is the server's generated frames over inference time. "Online FPS" is bound by llvmpipe capture: median `native_capture_ms` is 148 to 186 ms per 4-frame chunk, against 61 to 64 ms model processing per chunk. Neither number is a deployment FPS measurement; upstream says so for software GL.
- Per-chunk GPU event timings from `--profile-chunks` (run 1, every model chunk): `cuda_denoise_ms` median 24.0, `cuda_clean_kv_ms` 23.4 to 27.3, `cuda_encode_ms` 2.6, `cuda_decode_ms` 4.6, `cuda_pack_copy_ms` 0.18.
- Kernel trace (run 4, `rocprofv3 --kernel-trace --stats` around serve.sh; covers load, warmup and the 40 s session): 126 kernels, 732k dispatches, 9.6 s GPU time. Top: `aiter::fmha_fwd_hd128_bf16` (torch's ROCm SDPA, 23940 calls, 20%), hipBLASLt bf16 GEMMs `Cijk_Alik_Bljk_BBS_BH_Bias_...` (about 29% together), torch cat/reduce/elementwise, MIOpen `naive_conv_ab_nonpacked_fwd_nchw_hip_bfloat16` (TAEHV), and the port's Triton `_causal_rope_kernel` (34620 calls).
- Decoded streams: run 2 model, 1220 frames at 864x480 with no decoder warnings; run 3 dual view, 751 frames at 1728x480. In the dual view the left half is the llvmpipe low-poly game render and the right half is the model's render of the same camera. That render is a photoreal ancient-Egypt city with the player character, following the walk, strafe and pan, and showing the weapon and muzzle effects while firing. Over 76 sampled frames: PSNR between the halves 8.4 dB mean (7.4 to 10.6), model-half per-frame std 24.8 or more, no black frames, mean absolute change between samples 32.6. So the published frames are generated, not the reference passed through. The run 2 reference stream decodes to the low-poly render only.
- Harness faults found and fixed along the way (none in the server):
  - Run 1's recordings: the first RTSP reader died on GIO's proxy lookup; the publisher sets `GIO_USE_PROXY_RESOLVER=dummy` for itself and the reader now does too. The reader then ran with `rtspsrc latency=0`, whose jitterbuffer dropped late fragments: 372 of 1085 decoded frames were corrupt, with more frames than were encoded. `latency=500` gave clean decodes. The server's own counters showed 0 dropped throughout.
  - Run 2's dual-view session could not start: the sidecar died with SIGBUS (-7) because the GStreamer registry file sat on the then-full overlay disk. Run 3 put the registry on tmpfs and the same session ran.

### Not exercised

- The browser's WHEP/WebRTC media leg. An aiortc 1.16 WHEP client got `201` with an SDP answer from MediaMTX for the live session, but ICE failed: upstream's MediaMTX config offers only ICE-TCP candidates on 127.0.0.1:8190 (UDP disabled), and aioice has no ICE-TCP. The RTSP read of the same MediaMTX path stands in for it. The browser page, HUD, audio and the Playwright tests (`tests/browser/test_browser.cjs`, `test_stability.cjs`) were not run.
- Hardware OpenGL. A host with a graphics-capable GPU next to the model GPU would be needed.
- Scenes castle and racing (`--prompt-mode dynamic`), `--preset quality` in the server, and `--reference-only` start-up. The reference-mode sessions above cover the no-model transport with the model loaded.

### Questions for Jeff (nothing assumed)

1. Coverage. `server:matrix_game.server` is moved from `owed` to `covered` in `surface.json`, because real sessions ran end to end through the real server, game binary, sidecar and MediaMTX, with frames generated on the GPU. The caveats are llvmpipe instead of hardware GL, a scripted client instead of a browser, and RTSP instead of WHEP for the read-back. If you want hardware GL or a browser WHEP session before counting it, move it back to `owed`, or record it as `scoped_out` with that reason.
2. Disk on the mi355 host. `/` (3.5 TB overlay) hit 100% during this job and broke one run plus a profiler attempt. The four finished Matrix-Game job directories (planner/porter `122610`, `130701`, `144930`, review/validator `153422`) hold about 460 GB between them, mostly duplicate MG-3/MG-4 weights. Removing them is your call. This job's directory holds about 10 GB (mostly the venv) and reads the `153422` MG-4 weights.

## Lesson for cuda-to-rocm: headless EGL on an Instinct host lists every GPU before the software device

For `linux-host-setup.md` (or wherever headless rendering next to GPU compute is covered). A CUDA project that renders offscreen through `EGL_EXT_platform_device` (enumerate with `eglQueryDevicesEXT`, take the first device that `eglInitialize` accepts) assumes the NVIDIA EGL device is the GL device. On a Linux host with AMD Instinct GPUs (gfx942, gfx950), Mesa lists one EGL device per DRM node, including the per-partition XCP nodes, so a host with 8 GPUs can expose dozens, and the software (llvmpipe) device comes last. Instinct parts have no graphics pipeline, so radeonsi rejects each of them, and a loop capped at 16 devices never reaches llvmpipe and fails with "no EGL device initialized". The same loop also opens the DRM nodes of GPUs that belong to other jobs. To get llvmpipe without editing the project, run only the rendering process with an `LD_PRELOAD` shim that returns 0 from `drmGetDevices2` (and, as a guard, refuses opens under `/dev/dri`). Mesa then offers exactly one device, `EGL_MESA_device_software`. Neither `LIBGL_ALWAYS_SOFTWARE=1` nor `EGL_PLATFORM=surfaceless` changes the device list (Mesa 25.2). Do not apply the shim to the compute process: HIP needs its render node.

## Validation 2026-11-24 (validator, linux-gfx942, job mi300b-20261123-191444)

Head `6d94b2fb5ccdd1351b3ce5b5242cadda5f367f70`, AMD Instinct MI300X (gfx942, `gfx942:sramecc+:xnack-`), four GPUs (`HIP_VISIBLE_DEVICES=4,5,6,7`, the dispatcher's pin; single-GPU runs on one of them), `ulimit -c 0`, `timeout` on every GPU run, `OMP_NUM_THREADS` 8 (MG-3) or 16 (MG-4). First run of the port on gfx942. Recipe of the gfx950 section, rebuilt in this job's directory.

### Environment

- `$FLEET_VENV` (Python 3.11) with `$FLEET_VENV/bin` first on PATH: `pip install --index-url https://nightly.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple 'torch[device-gfx942]==2.12.0+rocm10.2.0a20261015' 'torchvision==0.27.0+rocm10.2.0a20261015' 'triton==3.8.0+git669b31ac.rocm10.2.0a20261015' 'rocm[devel]==10.2.0a20261015'`. HIP 7.17.26412, RCCL 2.32.3 (`torch.cuda.nccl.version()`). `torch`, `triton` and `rocm_sdk` resolve inside the venv, shadowing the image's ROCm 10.1 base links (checked).
- MG-4 `pip install -e '.[inference,server,download,test]'` (transformers 5.17.0); MG-3 extras `easydict decord opencv-python==4.10.0.84 trimesh pandas==2.2.3`. MG-3 on torch 2.12, not its pinned 2.10 (same deviation as gfx950).
- flash-attention `v2.8.3.post2` (`49774b31`):
  - CK: `BUILD_TARGET=rocm GPU_ARCHS=gfx942 OPT_DIM=128 ROCM_HOME=$(rocm-sdk path --root) pip wheel --no-build-isolation --no-deps .` with MAX_JOBS=64, 704/704 objects, 305 s.
  - Triton backend: `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, pure-Python wheel.
  - Each was installed with `pip install --target` into its own directory and put on `PYTHONPATH` only for the runs that use it, so MG-4's SDPA runs could not pick it up. MG-4 imports flash-attn whenever it is importable.
  - Check against SDPA (bf16, H 24, D 128): CK varlen 1000+777 max abs err 2.0e-3, causal S 1560 7.8e-3; Triton 2.0e-3 and 3.9e-3.
- Weights in the job directory (2.7 TB free before download): MG-4 `Skywork/Matrix-Game-4.0` at `2c53100457da7abcf0e966f75637c644d3423a63`, ancient-egypt, models plus game-assets manifests, every file SHA256-verified by `matrix-game download` (23 GB plus 23 GB HF cache); MG-3 `Skywork/Matrix-Game-3.0` at `d4c84ef2b8e9028a00a5d58b6be2df04f1cb6042` (53 GB).

### Non-GPU tests

`python -m unittest discover -s tests -p 'test_*.py'` in Matrix-Game-4: **81 tests OK** on `moat-port`, 81 OK on upstream `51913b2` (worktree, `PYTHONPATH` pointing at it). No regression.

### CUDA gate

Not rerun: recorded at this head in the review (Triton NVIDIA backend, PTX/cubin/SASS byte-identical 15/15). No CUDA C++ in the change.

### gfx942 FP8 path, layer level

- Selectors: `e4m3_format() == (torch.float8_e4m3fnuz, 240.0)`. `mxfp8_supported() == False`, so `build_linear("mxfp8")` returns `_Fused(fp8=False)`: the MXFP8 role runs in BF16.
- `FP8Linear` (rtx5090) and `PaddedFP8Linear` (rtx5080): K/N in {3072/3072, 3072/14336, 14336/3072}, M in {1, 37, 405, 4096}, inputs with 20x outlier columns (`fp8_probe.py`).
  - Activations quantize to FNUZ with max |q| = 240, and every output is finite.
  - Rel err against the BF16 `nn.Linear`: 0.034 to 0.038.
  - Against an exact emulation of the intended function (dequantized FNUZ operands times scales, FP32 matmul, plus bias): rel 1.6e-3 to 1.7e-3, max abs 0.03 (0.06 once). That is BF16 output rounding.
  - The padded variant matches the plain one to 3e-5.
- FNUZ/240 against OCP e4m3fn/448 with the same per-tensor scaling, in software on the same tensors (`fnuz_vs_fn.py`): activation and weight qdq rel 0.0265 / 0.0264, GEMM rel against FP32 0.0375 / 0.0374 for both formats. So the FNUZ branch loses nothing to the OCP format the NVIDIA and gfx950 paths use.

### MG-4 end to end (gfx942, one GPU)

- Command: `matrix-game render --scene ancient-egypt --resources <res> --input egypt-pan-120.mp4 --prompt-mode fixed --seed 42 --preset <p> [--dit-runtime native --no-vae-graph]`.
  - Run through a wrapper that prints `torch.cuda.max_memory_allocated`/`reserved` at exit.
  - With `MG_DET=1`, the wrapper also sets `torch.backends.cudnn.deterministic = True`. This is a harness setting; the source is not edited.
- Input: 120-frame lossless (libx264rgb crf 0) 864x480 pan/zoom over `models/ancient-egypt/sink.png` (sha256 `94e03c1a...45f6`); clip sha256 `e44b5daf...bc7c`. It is a different clip from the other hosts', so PSNR compares runs on this host only.
- Reference: upstream's release gate, bf16 `native` (`--preset realtime --dit-runtime native --no-vae-graph`: TAEHV, one step, SDPA) on the same GPU.
  - With no `--preset`, the render CLI resolves to the Wan-VAE two-step path. My first "native" run was that path and is listed below as `quality`.

**Run-to-run nondeterminism on gfx942 (found here, absent on gfx950/gfx1100).** With PyTorch defaults, two identical `native` runs differ: 34.2 dB mean, from frame 0. Two `realtime` runs differ the same way (34.5 dB).
- The Wan-VAE paths (`quality`, and realtime with `--vae-backend wan`) are bit-identical run to run, and so are TAEHV encode and one-shot decode standalone.
- The source is TAEHV's streaming decode. A forward-hook diff of two decodes of the same latents puts the first divergence at `core.decoder.4.conv.0`: bf16 `Conv2d` 512->256, 3x3, input `(1, 512, 30, 54)`, max diff 0.0039.
- Standalone, that conv differs on 19 of 19 repeated calls at batch 1, and never at batch 12, with a 1x1 kernel, or with `torch.backends.cudnn.deterministic = True`.
- MIOpen logging shows the default pick is `ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC`; deterministic mode picks `GemmFwdRest`.
- With deterministic mode, `native` and `realtime` each repeat bit-exactly, at the same speed (native 25.7 against 24.6 FPS).
- So this is MIOpen's default non-deterministic solver choice, not the port and not upstream code. The table below uses deterministic mode, so its noise floor is zero.

| run (SDPA unless noted; cudnn deterministic) | rc | render wall s | FPS at frame 101 | peak alloc GiB (reserved) | PSNR vs native mean / min dB | first10 / last10 dB |
|---|---|---|---|---|---|---|
| native (reference) | 0 | 80 | 25.7 | 20.21 (20.26) | -- (repeat bit-identical) | -- |
| realtime (ring-fused, VAE graph) | 0 | 84 | 26.7 | 20.21 (20.26) | 33.47 / 25.49 (repeat bit-identical) | 38.3 / 27.9 |
| rtx5090 (FP8 FNUZ) | 0 | 100 | 44.2 | 15.65 (18.09) | 28.86 / 23.62 | 34.3 / 24.9 |
| rtx5080 (FP8 FNUZ, padded/transposed) | 0 | 88 | 50.2 | 10.90 (10.93) | 29.04 / 23.45 | 32.7 / 24.8 |
| rtx5070ti (INT8, `cross_q` MXFP8 role in BF16) | 0 | 80 | 45.4 | 10.90 (10.93) | 31.79 / 25.54 | 36.0 / 27.7 |
| native, FA2 CK | 0 | 80 | 21.7 | 20.21 | 33.44 / 25.69 | 37.2 / 28.2 |
| realtime, FA2 CK | 0 | 81 | 25.9 | 20.21 | 33.69 / 26.07 (32.97 vs native-CK) | 38.0 / 28.8 |

Against bf16 realtime instead: rtx5090 29.14, rtx5080 29.25, rtx5070ti 32.07 dB mean.

The same presets with PyTorch defaults (non-deterministic conv) all completed too: native, realtime, rtx5090, rtx5080 and rtx5070ti each rc 0, 120/120 frames.
- FPS: 24.6 / 26.5 / 45.3 / 49.2 / 46.7.
- PSNR against that run's native: 33.7 / 28.9 / 28.5 / 30.0 dB.
- `realtime` with FA2 CK, run twice: 36.7 dB from each other (that is the conv noise).
- `quality` (Wan VAE, native DiT, two steps): rc 0, 42.05 GiB peak (55 reserved), bit-identical across two runs. 3.3 FPS on its first run (MIOpen first-use tuning, as gfx1100 saw), 12.3 FPS afterwards.

Reading:
- Every run: rc 0, 120/120 frames at 864x480, no NaN in the logs, per-frame std 46 or more (input 47).
- The FP8 presets run the FNUZ branch on gfx942 and are the fastest presets here, up to 1.9x realtime's FPS at about half its memory.
- They sit about 4.5 dB below bf16 realtime against native. rtx5070ti, which here runs INT8 plus BF16 like gfx1100, sits 1.7 dB below.
- The layer checks above show the FP8 GEMM computes the intended FNUZ function to BF16 rounding, with the same quantization error as e4m3fn/448. So the gap is quantization cost on this clip, not an FNUZ defect.
- The dB gaps between quantized and bf16 presets are not comparable across hosts: the clips differ, and gfx950's realtime itself sat at 22 dB from its native.
- Contact sheet `out/montage-mg4.png` (job directory, not durable): input, native, realtime, rtx5090, rtx5080 and rtx5070ti at frames 0/40/80/119. All coherent and following the pan, with no corruption or colour faults; runs differ only in transient effects.
- The `-sage` presets were not run here (not in this job's request).

### MG-3 distributed (gfx942, 2 and 4 GPUs)

Common arguments, as in the gfx950 runs: `generate.py --size 704*1280 --ckpt_dir Matrix-Game-3.0 --use_int8 --num_iterations 4 --num_inference_steps 3 --image demo_images/001/image.png --prompt "A colorful, animated cityscape with a gas station and various buildings." --seed 42 --vae_type mg_lightvae --lightvae_pruning_rate 0.5 --fa_version 2`, with FA2 CK.
- Multi-GPU runs use `torchrun --nnodes 1 --nproc_per_node N --rdzv-backend static --master-addr 127.0.0.1 --master-port <p> --local-addr 127.0.0.1`. The static rendezvous on 127.0.0.1 was used as asked; the container hostname did resolve here (to 127.0.1.1).
- The port does not touch Matrix-Game-3, so upstream `51913b2` is the code under test.
- Every run produces 177 frames at 1280x704. RCCL smoke test first (`all_to_all_single`, `all_gather`, `all_reduce`, as Ulysses and FSDP use them): correct on 2 and 4 ranks.

**Upstream as shipped cannot run multi-GPU, on any GPU.** There are two upstream bugs, both already reported upstream by others, neither in the port or the environment:

1. `--ulysses_size > 1` (test.sh's form, and the only way FSDP is ever enabled, because `_validate_args` turns `--dit_fsdp/--t5_fsdp` off when `ulysses_size <= 1`). Every rank fails in the first DiT block:
   - Error: `TypeError: attention() got an unexpected keyword argument 'fa_version'` at `wan/distributed/ulysses.py:63`. `distributed_attention` passes `fa_version=` and `attention()` takes `version=`.
   - Seen on 2 GPUs (`--dit_fsdp --t5_fsdp --ulysses_size 2`, rc 1, 94 s) and 4 GPUs (rc 1, 95 s).
   - Upstream issue SkyworkAI/Matrix-Game#80 (open, no comments); the one-word fix is open PR SkyworkAI/Matrix-Game#61 (mergeable, not merged).
2. The README's own command, `torchrun --nproc_per_node=$NUM_GPUS generate.py ... --dit_fsdp --t5_fsdp` with no `--ulysses_size`:
   - FSDP is silently disabled and every rank runs the whole job.
   - The DiT then keys its distributed behaviour on `world_size > 1`, not on whether sequence parallelism is on. It returns a Python list (`wan/modules/model.py` end of `forward`), and the scheduler fails with `TypeError: only integer tensors of a single element can be converted to an index` at `wan/utils/fm_solvers_unipc.py:323`.
   - Every rank fails this way, on 2 GPUs (rc 1, 103 s) and 4 GPUs (rc 1, 102 s).
   - Upstream issue SkyworkAI/Matrix-Game#68 reports exactly this from an NVIDIA run (6 GPUs, FA3).

**With PR #61's one-word fix, the distributed layer works on ROCm.**
- Setup: a throwaway worktree at `51913b2` (outside the fork checkout, which stays clean) with PR #61's diff applied, which changes `fa_version=` to `version=` in `ulysses.py`.
- Single-GPU from that tree is bit-identical to single-GPU from the port, so the fix does not touch the 1-GPU path.

| run | GPUs | rc | wall s | DiT steps it/s | PSNR vs single-GPU FA2 CK mean / min dB | first10 / last10 dB |
|---|---|---|---|---|---|---|
| single, FA2 CK (reference) | 1 | 0 | 290 cold, 110 repeat | 1.01 | -- (repeat bit-identical) | -- |
| single, FA2 Triton (yardstick: two correct backends) | 1 | 0 | 118 | 0.67 | 26.68 / 17.58 | 33.8 / 17.8 |
| FSDP + Ulysses 2 (`--dit_fsdp --t5_fsdp --ulysses_size 2`) | 2 | 0 | 115 | 1.30 | 27.88 / 19.54 (repeat bit-identical) | 35.7 / 19.6 |
| FSDP + Ulysses 4 | 4 | 0 | 116 | 2.25 | 28.30 / 18.50 (repeat bit-identical) | 35.5 / 19.1 |
| FSDP + Ulysses 3 + `--use_async_vae --async_vae_warmup_iters 1`, VAE worker on the 4th GPU (test.sh's pattern; 40 heads padded to 42) | 3+1 | 0 | 120 | 1.47 | 27.26 / 18.63 | 35.7 / 18.7 |

Reading:
- Each distributed run is deterministic: its repeat is bit-identical.
- Each one differs from single-GPU by a deterministic numerical difference. Frame 0 is at 39.6 dB in every case, and the gap grows through the 4 autoregressive iterations.
- That gap is no larger than the one between two correct single-GPU attention backends (FA2 CK against FA2 Triton: 26.7 dB mean), and SP2 against SP4 is 28.0 dB. So the distributed output matches single-GPU to the level independent correct paths agree.
- DiT denoising scales from 1.01 it/s (1 GPU) to 1.30 (2) and 2.25 (4). End-to-end wall stays about 115 s at this length, because model load and rank 0's serial VAE decode dominate 4 iterations.
- Contact sheet `out/montage-mg3.png` (single CK, single Triton, SP2, SP4, SP3+async VAE): same scene and camera path in all, with late frames differing only in detail.

Surface entry `model:mg3-distributed`: covered. Real 2- and 4-GPU FSDP+Ulysses runs over RCCL completed and matched single-GPU at the same seed, with the caveat that they needed upstream's open one-word PR #61. Without it, upstream's multi-GPU paths fail on every platform with the two upstream bugs above.

### Not run, and why

- The `-sage` presets, MG-4's racing scene, PR #67/AITER and MG-3 `--compile_vae` were not in this job's request.
- The documented `download.pytorch.org/whl/rocm7.2` install line was not used: the job was told to use the TheRock 10.2 wheels.

### Questions for Jeff (nothing assumed)

1. **MG-3 multi-GPU in the port.** Upstream's multi-GPU MG-3 does not run as shipped on any GPU (issues #80 and #68 above). The port does not touch MG-3. Options:
   - (a) leave it to upstream: PR #61 is open and fixes #80; #68 has no fix;
   - (b) a porter round carrying PR #61's one-word fix in the port, a scope addition for you to rule;
   - (c) a results-only comment on PR #61 or issue #80 saying the fix was exercised on MI300X with 2 and 4 GPUs. That is a GitHub post, so it needs your approval of a draft first; none is drafted here.
2. **MIOpen non-deterministic conv.** On gfx942, TAEHV's streaming decode is not bit-reproducible run to run under PyTorch defaults. Measured effect: about 34 dB between two identical 120-frame renders.
   - Cause: MIOpen's default solver `ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC` for a batch-1 bf16 3x3 conv. `torch.backends.cudnn.deterministic = True` removes it at no measured cost.
   - It is permitted behaviour (cuDNN makes the same no-determinism promise), so I registered nothing. Whether it deserves an MIOpen report is your call.

Verdict for linux-gfx942: **pass**.
- Unit tests 81/81, no regression.
- realtime, rtx5090 and rtx5080 (FP8 FNUZ branch) and rtx5070ti (INT8 plus BF16 MXFP8 fallback) render end to end. Layer-level FP8 matches an exact FNUZ emulation to BF16 rounding.
- MG-3 runs single-GPU, and 2- and 4-GPU FSDP+Ulysses with PR #61's fix.
- The tree is clean at `6d94b2f`, and `jargon.py --port Matrix-Game` is clean.

## Lesson for cuda-to-rocm: tell MIOpen's non-deterministic conv solvers from a real fault before comparing runs

For `diagnosis.md` (or wherever run-to-run reproducibility checks are covered).
- What happened: a video model that repeats bit-exactly on gfx950 and gfx1100 gave a different output on every run on gfx942 (about 34 dB PSNR between identical runs). Without a known noise floor, every A/B comparison on that host was ambiguous.
- Cause: MIOpen (PyTorch's `cudnn` backend on ROCm) picks `ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC` for some bf16 convolutions on gfx942. It is not bit-reproducible: a batch-1 3x3 512->256 conv gave a different result on every call, up to one bf16 ulp. The same conv at batch 12, or as 1x1, was exact.
- How to find it: register forward hooks on every leaf module, run the same input twice in one process, and report the first module whose output differs. That names the layer in one run. Then repeat that layer alone at the observed shape. `MIOPEN_ENABLE_LOGGING=1 MIOPEN_LOG_LEVEL=6` prints the chosen solver.
- Fix for comparisons: `torch.backends.cudnn.deterministic = True` makes MIOpen choose a deterministic solver (`GemmFwdRest` here), with no measurable cost in this model.
- Set it for A/B validation, and establish the noise floor (run the reference twice) before reading any difference between two code paths as real.

## Verification of the PR #67 comment claims (linux-gfx942, 2026-11-24)

Request from Jeff Daily: measure every claim in his draft comment on SkyworkAI/Matrix-Game#67 before he posts it. This is not a pipeline role. No project state was changed, nothing was pushed to a fork, and nothing was posted. Job `mi300c-20261124-021312-re-verify-pr67-prompt`. It re-ran the killed job `mi300c-20261123-155958-verify-pr67-prompt` from scratch: weights re-downloaded, venvs rebuilt, AITER and flash-attention rebuilt. Only the two source clones were copied from it and checked (`git describe`: AITER `v0.1.30` = `77ba72b7`, flash-attention `v2.8.3.post2` = `49774b31`, both `git clean -xffd` including submodules before building).

- PR state at check time: OPEN, head `9ad32b1d90ef35c6780e2023aabc115e2f33496e`, `mergeable` MERGEABLE, `mergeStateStatus` CLEAN. Upstream main `51913b2bfa78a6d54de912b8348b1d318f4d36e2`.

### Environment, and how it differs from the PR's claim

- **The PR claims** "MI300X, gfx942, ROCm 7.2", run in `rocm/pytorch:rocm7.2.1_ubuntu22.04_py3.10_pytorch_release_2.9.1` with `amd-aiter 0.1.13`. Its README installs torch from `https://download.pytorch.org/whl/rocm7.2`.
  - PyPI has no `amd-aiter` 0.1.13 (the only release is 0.1.7.post2.dev18), so 0.1.13 must have come from a source build or an internal index.
- **This environment:**
  - GPU: an AMD Instinct MI300X (gfx942, `gfx942:sramecc+:xnack-`) on a host in **CPX partition mode**. The dispatcher's pin `HIP_VISIBLE_DEVICES=4` is one partition: 38 CUs (`torch.cuda.get_device_properties(0).multi_processor_count`) and 24.0 GiB, against 304 CUs and 192 GB for a whole MI300X. MIOpen names it `gfx942_38`.
  - Software, all job-local:
    - Python 3.11.17.
    - TheRock wheels from `https://repo.amd.com/rocm/whl/gfx94X-dcgpu/`: `torch 2.10.0+rocm7.13.0` (MG-3's own `requirements.txt` pins `torch==2.10.0`), `torchvision 0.25.0+rocm7.13.0`, `triton 3.6.0+rocm7.13.0`, `rocm[libraries,devel]==7.13.0`. HIP 7.13.99004.
    - Ubuntu 24.04 host userspace, amdgpu 6.16.13.
- So the GPU is the same model in a 1/8 partition, the ROCm is 7.13 not 7.2, and torch is 2.10 not 2.9.1. Every result below is on this partition with ROCm 7.13 and torch 2.10.
- **Toolchain check (the addendum's).** `rocm-sdk path --root` = `$FLEET_VENV/lib/python3.11/site-packages/_rocm_sdk_devel`. The host has no `/opt/rocm`; the image's base links (ROCm 10.1, triton 3.9) were disabled in the job venv (`zz-base-links.pth` renamed).
  - Builds ran with `ROCM_HOME=ROCM_PATH=HIP_PATH=<that root>`; `hipcc --version` printed `HIP version: 7.13.99004-3309c6114a`, AMD clang 23.0.0git; `torch.utils.cpp_extension.ROCM_HOME` = the same root.
  - flash-attention CK build log: all 705 compiler invocations are `<root>/bin/hipcc`, 704 objects with `--offload-arch=gfx942`.
  - AITER's runtime JIT (`aiter/jit/build/*/build.ninja`) also uses `<root>/bin/hipcc`.
- **Builds.**
  - flash-attention CK: `BUILD_TARGET=rocm GPU_ARCHS=gfx942 OPT_DIM=128 MAX_JOBS=96 pip wheel --no-build-isolation --no-deps .`, about 13 min.
    - Installed with `pip install --target fa-site` and put on `PYTHONPATH` only for runs that should see it, so AITER-only runs could not import it.
  - flash-attention Triton backend: `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE pip wheel ...` (pure Python), installed into its own `fa-tri-site`.
  - AITER: `GPU_ARCHS=gfx942 PREBUILD_KERNELS=0 MAX_JOBS=16 pip wheel --no-build-isolation --no-deps .`, wheel `amd_aiter-0.0.0-cp311`.
    - Its `setup.py` again uninstalled the venv's `triton 3.6.0+rocm7.13.0` and installed `triton 3.8.0+amd.rocm7.2.0` from pypi.amd.com (the gfx950 lesson above). That was reverted with `pip install --no-deps triton==3.6.0+rocm7.13.0` before any GPU run.
    - Its dependency `flydsl==0.5.1` was then installed with a constraints file pinning torch, triton and torchvision.
- **Venvs.**
  - Main: `$FLEET_VENV` holds AITER v0.1.30 from source.
  - `extra/plain`: torch and MG-3's requirements only, with no flash-attn and no AITER.
  - `extra/pypi-aiter` and `extra/amd-aiter`: the fresh venvs for claims 4 and 5.
  - All pip commands ran with `PYTHONPATH` unset. The fleet's `pylib` on `PYTHONPATH` otherwise makes pip skip `typing_extensions` in a fresh venv, which then cannot import torch; the first claim-5 attempt hit exactly that and was redone.
  - `pandas` in the main venv was raised from the image's 2.0.3 to 2.2.3 (2.0.3 fails to import against numpy 2.2.6, which made `import aiter` fail).
- `ulimit -c 0` and `timeout` on every GPU run; only GPU 4.

### What had to change to run MG-3 at all on a 24 GiB partition (harness side; generate.py and the repo unchanged)

Runs go through `scripts/run_mg3.py`, which calls `generate.py` with `runpy` after some instrumentation and patches:
- **Instrumentation.** It counts calls to `aiter.ops.mha.flash_attn_varlen_func`, `flash_attn.flash_attn_varlen_func`, `flash_attn.flash_attn_func` and `F.scaled_dot_product_attention`, and saves the uint8 frames handed to `process_video` (`raw.npy`, used for the metrics).
- **Patches, each forced by the partition:**
  1. **T5 on CPU.** The README command OOMs during the Wan VAE encode with the 11 GB T5 on the GPU: `torch.OutOfMemoryError ... Of the allocated memory 22.56 GiB is allocated by PyTorch`, rc 1, 60 s. `--t5_cpu` does not help on one GPU. T5 is kept on the CPU and only its output moves to the GPU.
  2. **DiT parked on CPU during VAE decode.** Even then the first VAE `stream_decode` OOMs: `Tried to allocate 3.44 GiB ... 18.52 GiB is allocated`, and with `PYTORCH_ALLOC_CONF=expandable_segments:True` `Tried to allocate 1.95 GiB ... 21.93 GiB`. The pipeline then crashes with `AttributeError: 'NoneType' object has no attribute 'cpu'` at `pipeline/inference_pipeline.py:652`. The DiT's weights are moved to the CPU for each `stream_decode` call and back afterwards (values unchanged; `expandable_segments` kept on).
  3. **`--compile_vae` only: VAE feature cache moved off the GPU between decodes.** After the first compiled decode, 3.4 GiB more stays allocated (7.67 against 4.03 GiB). The DiT then OOMs in the next iteration (`Tried to allocate 482.00 MiB ... 21.90 GiB allocated`; rc 1 in 3 of 3 attempts, AITER and FA2 alike). With the `feat_cache` tensors moved to the CPU after each decode and back before the next, it completes.
  4. **MIOpen's naive conv solver disabled for the first run.** `MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD=0`. On the first, cold run, MIOpen's Find (`DYNAMIC_HYBRID`) benchmarked `ConvDirectNaiveConvFwd` for each Wan-VAE conv3d shape. It spent more than 5 minutes on one shape (`320-3-178-322-3x3x3-320-...-NDHWC-BF16`) on 38 CUs, and the previous job's run was killed while still stuck there.
- **None of these patches changes the numerics:**
  - The patch-2 run with the cache offload but without `--compile_vae` is bit-identical to the run without the offload (177/177 frames).
  - With the naive solver re-enabled (`=1`, a warm find-db), the AITER run is bit-identical to the one with it disabled (rc 0, 144 s).
- The T5-on-CPU embedding cannot be compared to a GPU one here (that run OOMs), so every MG-3 number below is "T5 on CPU", identically for all backends.
- Two-backend PSNR from the full-GPU hosts (gfx942 26.7 dB, gfx950 24.8 dB) is not directly comparable to these numbers.

Common command (each run adds what its row says):

```
python generate.py --size 704*1280 --use_int8 --num_iterations 4 --num_inference_steps 3 \
  --vae_type mg_lightvae --lightvae_pruning_rate 0.5 --image demo_images/001/image.png --seed 42 \
  --ckpt_dir <job>/weights/Matrix-Game-3.0 \
  --prompt "A colorful, animated cityscape with a gas station and various buildings." \
  --output_dir <out> --save_name video
```

Weights: `Skywork/Matrix-Game-3.0` at `d4c84ef2b8e9028a00a5d58b6be2df04f1cb6042`, downloaded fresh by `hf download` (53 GB, exit 0). Wall times are the wrapper's whole-process wall, model load included.

### Claim 1: the command completes with and without `--compile_vae` and writes 177 frames

- **The command exactly as written in the comment fails at argument validation** (PR head, AITER venv), whatever the GPU:
  - As written: `AssertionError: Please specify the checkpoint directory.` (`generate.py:23`), rc 1, 8 s.
  - With `--ckpt_dir` added: `AssertionError: please specify the prompt.` (`generate.py:24`), rc 1, 7 s.
  - `generate.py` has no default for either, so the comment needs `--ckpt_dir Matrix-Game-3.0 --prompt "..."` (as in the README) or must say the command is abbreviated.
- With those two added, AITER v0.1.30 only (no flash-attn importable), PR head:

| run | rc | wall | frames in `video.mp4` (decoded / container) | notes |
|---|---|---|---|---|
| no harness patches | 1 | 60 s | none written | OOM in VAE encode (22.56 GiB) |
| T5 on CPU | 1 | 88 s | none written | OOM in VAE `stream_decode` (18.52 GiB), then `AttributeError` |
| T5 on CPU + DiT offload during decode | **0** | 156 s | 177 / 177, 1280x704, 17 fps | `pr-aiter-1` |
| same, repeat | 0 | 146 s | 177 / 177 | `pr-aiter-2`, bit-identical to `pr-aiter-1` |
| `--compile_vae`, T5 on CPU + DiT offload | 1 | 185 s, 109 s, 110 s | none written | 3 attempts; OOM in the DiT after the compiled decode |
| `--compile_vae`, + VAE cache offload | **0** | 155 s | 177 / 177 | `pr-aiter-cvo-1` |
| `--compile_vae`, + VAE cache offload, repeat | 0 | 156 s | 177 / 177 | `pr-aiter-cvo-2`, bit-identical to `-cvo-1` |

- Frame counts were read by decoding every frame of `video.mp4` with OpenCV (`scripts/frames.py`), and match the `(177, 704, 1280, 3)` array handed to the writer.
- `--compile_vae` against no compile, AITER: frames 0 to 56 identical. From frame 57 on, mean PSNR 54.7 dB (min 53.0) and max abs diff 18 levels. FA2 shows the same 54.7 dB.

Verdict: **confirmed with qualifications.**
- Confirmed: with `--ckpt_dir` and `--prompt` added, the PR head with source-built AITER v0.1.30 completes with and without `--compile_vae` on an MI300X-class GPU and writes 177 frames.
- Not confirmed as literally written: the command as quoted fails argument validation.
- Not confirmed on this hardware as shipped: on a 24 GiB CPX partition it completes only with the harness memory patches above. A whole MI300X (192 GB) was not available to this job, so completion without patches was not observed here; the earlier gfx950 job saw it complete on a whole MI355X.

### Claim 2: AITER output agrees with FA2 CK "to the run-to-run level between any two attention backends"

Metrics are per frame over the 177 uint8 frames (`scripts/compare.py`): mean and max abs difference in 0..255 levels, and PSNR. The mean PSNR is over frames 1..176, because frame 0 (the input image) is identical in every pair.

| pair (same seed, same command) | bit-identical | mean abs diff, mean over frames (first10 / last10) | max abs diff, max over frames | PSNR mean / min dB | PSNR last10 dB |
|---|---|---|---|---|---|
| AITER run 1 vs AITER run 2 | **yes**, 177/177 | 0 | 0 | inf | inf |
| FA2 CK run 1 vs run 2 | **yes** | 0 | 0 | inf | inf |
| masked SDPA run 1 vs run 2 | **yes** | 0 | 0 | inf | inf |
| repo SDPA fallback run 1 vs run 2 | **yes** | 0 | 0 | inf | inf |
| AITER vs FA2 CK | no | 8.83 (0.74 / 24.04) | 255 | 25.51 / 14.70 | 14.83 |
| AITER vs FA2 CK, both `--compile_vae` | no | 8.83 (0.74 / 24.02) | 255 | 25.51 / 14.71 | 14.84 |
| FA2 Triton vs FA2 CK | no | 10.71 (0.71 / 27.61) | 255 | 24.76 / 13.71 | 13.84 |
| AITER vs FA2 Triton | no | 12.78 (0.59 / 32.78) | 255 | 24.99 / 12.91 | 13.06 |
| masked SDPA vs FA2 CK | no | 7.76 (0.74 / 23.22) | 255 | 25.20 / 14.21 | 14.41 |
| masked SDPA vs FA2 Triton | no | 8.72 (0.70 / 20.18) | 255 | 25.85 / 15.96 | 16.39 |
| masked SDPA vs AITER | no | 10.81 (0.72 / 28.64) | 255 | 24.41 / 13.57 | 13.66 |
| AITER (+FA2 action module) vs AITER only | no | 9.07 | 255 | 26.03 / 14.21 | 14.31 |
| repo SDPA fallback vs FA2 CK | no | 34.70 (18.25 / 47.79) | 255 | 13.07 / 11.46 | 11.58 |
| repo SDPA fallback vs AITER | no | 35.92 | 255 | 12.88 / 11.00 | 11.10 |

How the SDPA runs were made:
- **Repo SDPA fallback.** SDPA cannot run through the CLI (claim 7). This run is the repo's own `attention(version='0')` path, reached by routing the DiT's direct `flash_attention()` calls to it in the harness: 1094 SDPA calls, no flash calls.
  - That path drops the key-padding mask (it warns "Padding mask is disabled when using scaled_dot_product_attention"). Cross-attention therefore attends to the zero-padded text positions up to 512, which is a different function, not a numerical variant. That explains its 13 dB.
- **Masked SDPA.** The fair SDPA baseline: the same routing, but with per-sample slices to `q_lens`/`k_lens`, as the varlen kernels do.

Which kernels each run used (call counts over the whole run):
- AITER-only runs: 720 `aiter...flash_attn_varlen_func`, 374 SDPA. The action module uses SDPA when flash-attn is absent.
- FA2 runs: 720 `flash_attn_varlen_func`, 360 `flash_attn_func` (action module), 14 SDPA.

Reading:
- **Run to run, every backend is bit-exact on this host.** The "run-to-run level" is therefore zero difference.
- AITER against FA2 CK is not zero: it diverges from frame 1 (mean abs 0.74 levels over the first 10 frames) and grows over the 4 autoregressive iterations to 14.8 dB over the last 10 frames.
- That difference sits inside the band between any two correct backends here: 24.4 to 26.0 dB mean and 12.9 to 16.4 dB over the last 10 frames, across FA2 CK, FA2 Triton, masked SDPA and AITER. AITER-vs-FA2-CK (25.5 / 14.8) is in the middle of it.

Verdict: **not confirmed as worded.** "To the run-to-run level" is false: run-to-run is exactly zero and AITER-vs-FA2 is 25.5 dB. "To the level between any two attention backends" is confirmed.

Wording these numbers support (proposed):

> With the same seed, the AITER output differs from the flash-attention 2.8.3 CK build on the same GPU about as much as any two correct attention backends differ from each other (FA2 CK, FA2 Triton, and SDPA with the padding mask): 25.5 dB mean PSNR against 24.4 to 26.0 dB for the other pairs. Each backend is bit-identical run to run, so this is the usual autoregressive divergence between kernels, not noise.

A side note from the PR body itself: "Output: identical 1.8MB MP4 video across all backends" does not hold here; no two backends gave identical output.

### Claim 3: AITER's varlen call "matches SDPA exactly" on a standalone check

`scripts/varlen_vs_sdpa.py <backend>` runs on GPU 4 against per-sequence `F.scaled_dot_product_attention` (same dtype) and against FP32 SDPA. `rel` is max abs error / max |FP32 reference|. AITER v0.1.30 from source; the call used `module_fmha_v3_varlen_fwd` (asm v3) and the CK `mha_varlen_fwd_*` modules.

| dtype | causal | H | D | q / k lens | AITER vs SDPA max abs | rel | frac of elements != | AITER vs FP32 max abs | SDPA vs FP32 max abs |
|---|---|---|---|---|---|---|---|---|---|
| bf16 | no | 24 | 128 | 1000+777 | 9.766e-04 | 1.68e-03 | 0.0017 | 1.366e-03 | 1.366e-03 |
| bf16 | no | 40 | 128 | 3520 | 4.883e-04 | 2.20e-03 | 0.0041 | 5.695e-04 | 5.695e-04 |
| bf16 | no | 40 | 128 | 14080 | 4.883e-04 | 4.59e-03 | 0.1247 | 2.879e-04 | 3.166e-04 |
| bf16 | no | 40 | 128 | 3520 / 512 | 1.953e-03 | 2.31e-03 | 0.0012 | 2.127e-03 | 2.127e-03 |
| bf16 | no | 40 | 128 | 2048+1024 | 9.766e-04 | 2.17e-03 | 0.0025 | 1.128e-03 | 1.128e-03 |
| bf16 | yes | 24 | 128 | 1000+777 | 7.812e-03 | 1.92e-03 | 0.2896 | 9.202e-03 | 9.202e-03 |
| bf16 | no | 16 | 64 | 1024 | 1.953e-03 | 3.86e-03 | 0.4492 | 1.634e-03 | 1.285e-03 |
| fp16 | no | 24 | 128 | 1000+777 | 4.883e-04 | 7.35e-04 | 0.6391 | 4.377e-04 | 1.473e-04 |
| fp16 | no | 40 | 128 | 3520 | 2.441e-04 | 9.65e-04 | 0.6413 | 1.589e-04 | 1.103e-04 |
| fp16 | no | 40 | 128 | 14080 | 6.104e-05 | 4.41e-04 | 0.6411 | 8.925e-05 | 4.947e-05 |
| fp16 | no | 40 | 128 | 3520 / 512 | 4.883e-04 | 5.09e-04 | 0.6380 | 5.404e-04 | 2.437e-04 |
| fp16 | no | 40 | 128 | 2048+1024 | 4.883e-04 | 8.31e-04 | 0.6404 | 3.452e-04 | 2.313e-04 |
| fp16 | yes | 24 | 128 | 1000+777 | 1.953e-03 | 4.78e-04 | 0.6294 | 1.953e-03 | 1.226e-03 |
| fp16 | no | 16 | 64 | 1024 | 2.441e-04 | 3.79e-04 | 0.6385 | 3.270e-04 | 2.056e-04 |

The same check for the flash-attention builds:
- FA2 CK against SDPA, bf16: max abs 4.9e-4 to 1.56e-2, rel 1.9e-3 to 4.6e-3, with 1.7 to 25 % of elements differing.
- FA2 Triton: bf16 4.9e-4 to 3.9e-3; fp16 6.1e-5 to 9.8e-4.

Verdict: **not confirmed.** The difference is not zero for any of the 14 cases. bf16 differs from SDPA in 0.1 % to 45 % of elements, up to 7.8e-3 abs (rel 1e-3 to 5e-3); fp16 in about 64 %.
- AITER's error against an FP32 reference is the same size as SDPA's own (e.g. bf16 1000+777: 1.366e-3 for both), and no larger than FA2 CK's.
- The right word is "agrees with SDPA to within bf16/fp16 rounding" (max relative error about 5e-3 bf16, 1e-3 fp16), not "exactly".
- The gfx950 job's single case (bf16 1000+777, H 24, D 128) gave max abs 0 on MI355X. That case gives 9.8e-4 here on gfx942, so "exactly" was an arch-specific observation from one shape.

### Claim 4: `pip install aiter`

Fresh venv `extra/pypi-aiter` (`/opt/python-3.11-venv/bin/python -m venv --clear`, `PYTHONPATH` unset), `python -m pip install aiter`:
- pip output: `Downloading aiter-0.13.20191203.tar.gz (9.3 kB)` ... `Successfully installed aiter-0.13.20191203`.
- `pip show aiter`: Summary "Useful patterns building upon asynchronous iterators.", Home-page `https://github.com/richardkiss/aiter`, Author Richard Kiss, no dependencies.
- The package contains `active_aiter.py`, `azip.py`, `map_aiter.py` and similar files, and no `ops/`.
- `python -c "import aiter.ops.mha"`: `ModuleNotFoundError: No module named 'aiter.ops'`, rc 1.
- `pip index versions aiter`: 0.13.20191203, 0.1.2, 0.1.

Effect at the PR head: `attention.py` catches every exception on that import, so AITER is silently unavailable. Without flash-attn the run then ends in the PR's own `RuntimeError` telling the user to `pip install aiter` again.

Verdict: **confirmed** (the comment's point holds). `pip install aiter` installs Richard Kiss's async-iterator helper 0.13.20191203, and `import aiter.ops.mha` fails afterwards.

### Claim 5: "The AMD package is `amd-aiter`"

Fresh venv `extra/amd-aiter` with the same TheRock torch 2.10.0+rocm7.13.0 stack, `python -m pip install -c constraints.txt amd-aiter`:
- pip output: `Downloading amd_aiter-0.1.7.post2.dev18-py3-none-any.whl (40.2 MB)`; installed with einops, pandas 3.1.1, psutil, pybind11, python-dateutil and tzdata. `pip check`: no broken requirements.
- `import aiter.ops.mha` right after install: `ModuleNotFoundError: No module named 'packaging'` (`aiter/jit/core.py:19`). The wheel's metadata does not declare `packaging`, while the source build's does.
- After a manual `pip install packaging`, the import works: `aiter/ops/mha.py`, `flash_attn_varlen_func` present. It JIT-builds `module_aiter_enum` with the warning `clang (LLVM option parsing): Unknown command line argument '-amdgpu-coerce-illegal-types=1'`.
- **Running it on this GPU fails.** `varlen_vs_sdpa.py aiter` in that venv JIT-builds `module_fmha_v3_varlen_fwd` (41 s) and then prints, at the first bf16 call:
  `[AITER] .../aiter/jit/build/module_fmha_v3_varlen_fwd/blob/asm_fmha_fwd_v3_gfx942.cpp:170 fail to call hipModuleLaunchKernel(kernel_func, ...) ---> [HIP error](invalid resource handle)`
  - The process then **exits with status 0** and no further output: its `HIP_CALL` macro calls `exit(0)`.
  - Cause, from the generated source: the kernel's constructor loads its code object only when `get_num_cu_func()` is 304 (`MI300/`) or 80/64 (`MI308/`), and otherwise returns without loading. On a 38-CU CPX partition the launch therefore uses a null function. The source-built v0.1.30 has no such restriction and runs on the same partition (claim 3).
- **generate.py with it** (PR head, `extra/amd-aiter` plus MG-3's requirements, same harness): it loads the models, prints the banner, and at the first DiT attention call prints the same `invalid resource handle` line. The process then ends with **rc 0 after 60 s, and no video is written**. A user would see a "successful" exit with no output.
- Not tested: a whole MI300X (304 CUs, SPX mode), where that wheel would load its `MI300/` kernels. Whether it works there is unknown from this job.

Verdict: **not confirmed as something to recommend.**
- PyPI `amd-aiter` is indeed AMD's package name (its wheel ships `aiter/ops/mha.py`, and ROCm/aiter's `setup.py` has `PACKAGE_NAME = "amd-aiter"`).
- But its only release, 0.1.7.post2.dev18, does not import without a manual `pip install packaging`.
- On this MI300X partition its varlen attention fails, and generate.py exits 0 with no output.
- The comment should not tell users to `pip install amd-aiter`. Say instead that `pip install aiter` is the wrong package and point to the source build from ROCm/aiter, which the README already lists and which is what was tested (v0.1.30).

### Claim 6: what the "Flash Attention Configuration" banner prints with AITER

Both pipelines run the same `_log_flash_attention_config`, which imports only `FLASH_ATTN_2_AVAILABLE` and `FLASH_ATTN_3_AVAILABLE`; there is no AITER branch. The banner is quoted from the logs. The interactive runs used `--interactive` with stdin `u`/`w` for every prompt, completed with rc 0, and wrote 177 frames.

| pipeline | environment | flag | printed | backend actually called (call counts) |
|---|---|---|---|---|
| `pipeline/inference_pipeline.py` | AITER only | default | `Requested: Default (3)` / `Actual:    None (SDPA)` | AITER 720, SDPA 374 (action module) |
| `pipeline/inference_interactive_pipeline.py` | AITER only | default | `Requested: Default (3)` / `Actual:    None (SDPA)` | AITER 720, SDPA 374 |
| `inference_pipeline.py` | AITER + FA2 CK | default | `Requested: Default (3)` / `Actual:    Flash Attention 2` | AITER 720, `flash_attn_func` 360, SDPA 14 |
| `inference_interactive_pipeline.py` | AITER + FA2 CK | default | `Requested: Default (3)` / `Actual:    Flash Attention 2` | AITER 720, `flash_attn_func` 360, SDPA 14 |
| `inference_pipeline.py` | AITER only | `--fa_version 0` | `Requested: 0` / `Actual:    Disabled (SDPA)` | AITER 720, SDPA 374; output bit-identical to the default AITER run |
| `inference_interactive_pipeline.py` | AITER only | `--fa_version 0` | `Requested: 0` / `Actual:    Disabled (SDPA)` | AITER 720, SDPA 374 |

Each banner is preceded by the line `🚀 Flash Attention Configuration:`.

Verdict: **confirmed.**
- With AITER alone the banner says `None (SDPA)`. With flash-attn also installed it says `Flash Attention 2`. Under `--fa_version 0` it says `Disabled (SDPA)`.
- In every case AITER does the DiT self- and cross-attention.
- With flash-attn present, flash-attn's `flash_attn_func` also does the action-module attention, which is why AITER+FA2 output differs from AITER-only (26.0 dB).

### Claim 7: `--fa_version 0`

Without flash-attn and without AITER (`extra/plain` venv, `pip list` shows neither), same command plus `--fa_version 0`:
- **Upstream main `51913b2`:** rc 1 after 66 s. Banner `Requested: 0` / `Actual:    Disabled (SDPA)`. Traceback ends:
  ```
    File ".../wan/modules/model.py", line 475, in forward
      x = flash_attention(
    File ".../wan/modules/attention.py", line 114, in flash_attention
      assert FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE
  AssertionError
  ```
- **PR head `9ad32b1d`:** rc 1 after 58 s, same banner. Traceback ends:
  ```
    File ".../wan/modules/model.py", line 475, in forward
      x = flash_attention(
    File ".../wan/modules/attention.py", line 150, in flash_attention
      raise RuntimeError(
  RuntimeError: No flash attention backend available. Install one of: flash-attn (NVIDIA), or aiter (AMD ROCm 7.x: pip install aiter)
  ```
- In both, the only SDPA call before the failure is one in the VAE encoder; the DiT never reaches SDPA.

With flash-attn installed (FA2 CK on `PYTHONPATH`), upstream main, `--fa_version 0`:
- rc 0, 144 s, 177 frames. Banner `Actual:    Disabled (SDPA)`.
- Call counts: `flash_attn_varlen_func` 720, `flash_attn_func` 360, SDPA 14, exactly the counts of the default FA2 run without the flag.
- The output is bit-identical to that run (177/177 frames).
- Cause: `wan/modules/model.py` lines 475 and 503 call `flash_attention()` directly, which ignores `version='0'`. Only `attention()` honours it, and the DiT does not use `attention()`.
- `wan/modules/action_module.py` picks `flash_attn_func` whenever flash-attn imports, regardless of the flag.
- At the PR head with AITER, `--fa_version 0` likewise still runs AITER (bit-identical to the default AITER run).

Verdict: **confirmed.**
- `--fa_version 0` does not give an SDPA run.
- Without a backend it fails at main (bare `AssertionError`) and at the PR head (the new `RuntimeError`).
- With flash-attn or AITER present it silently uses them while the banner says `Disabled (SDPA)`. Determined by call counting and by bit-identical output.

### Summary for the comment

| claim | verdict |
|---|---|
| 1 completes, 177 frames, with/without `--compile_vae` | confirmed once `--ckpt_dir` and `--prompt` are added (the quoted command fails validation); on this 24 GiB CPX partition only with harness memory patches |
| 2 agreement with FA2 CK | "run-to-run level" not confirmed (run-to-run is bit-exact); "level between any two backends" confirmed (25.5 dB inside 24.4 to 26.0 dB) |
| 3 "matches SDPA exactly" | not confirmed; agrees to bf16/fp16 rounding (max rel ~5e-3 bf16) |
| 4 `pip install aiter` is the wrong package | confirmed |
| 5 "the AMD package is `amd-aiter`" | name correct, recommendation not confirmed: needs `packaging`, fails on this partition, and generate.py exits 0 with no video |
| 6 banner ignores AITER | confirmed (quotes above) |
| 7 `--fa_version 0` is not SDPA | confirmed |

### Not done, and why

- A whole MI300X (SPX, 192 GB, 304 CUs) was not available. The pin was one CPX partition. So no unpatched end-to-end run, and no `amd-aiter` run where its MI300 kernels would load.
- The PR's exact environment (ROCm 7.2.1 container, torch 2.9.1, Python 3.10, `amd-aiter 0.1.13`) was not reproduced; 0.1.13 is not on PyPI.

### Question for Jeff (nothing posted)

The comment needs these corrections before posting:
- the command needs `--ckpt_dir`/`--prompt`;
- "run-to-run level" should become the wording proposed under claim 2;
- "exactly" should become "to within bf16 rounding";
- point 1 should drop the `amd-aiter` recommendation in favour of the source build.

Whether to also mention the `amd-aiter` silent exit-0 failure to AITER (a separate ROCm/aiter issue, not this PR) is your call; nothing was filed.
