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
