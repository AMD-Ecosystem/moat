# Matrix-Game port plan

Planner: linux-gfx950, 2026-11-20, against upstream `main` at `51913b2bfa78` (2026-10-29).

**Decisions needed from Jeff before a porter starts** (details in "Open questions" at the end and in notes.md, "Planning (2026-11-20)"):

1. **Q1, which model generations are in scope.** Recommended: Matrix-Game-4 and Matrix-Game-3; Matrix-Game-1 and Matrix-Game-2 recorded as scoped out (legacy).
2. **Q2, Matrix-Game-4's NVIDIA-tuned RTX presets** (`rtx5090`, `rtx5080`, `rtx5070ti` and their `-sage` variants). Recommended: scope out; the generic `realtime` / `quality` / `baseline` presets are the port.

This plan is written for the recommended answers. Each alternative is costed below so the porter can run whichever Jeff picks. `surface.json` leaves the components these questions cover unaccounted for (neither covered nor scoped out) until he rules.

## Project

- Name: Matrix-Game (Skywork AI interactive world models)
- Upstream: https://github.com/SkyworkAI/Matrix-Game, default branch `main`
- Fork: https://github.com/AMD-Ecosystem/Matrix-Game, port branch `moat-port`
- Licence: MIT at the top level; Matrix-Game-3 and Matrix-Game-4 are Apache-2.0 (intake).
- Layout: four independent model generations, one directory each, with no shared code:
  - `Matrix-Game-4/`: the current release. Package `matrix-game-4.0` (`pyproject.toml`), `matrix-game` CLI, torch 2.12, a CPU unittest suite, a C++/SDL2/OpenGL game client, and a browser/WebRTC server.
  - `Matrix-Game-3/`: Wan2.2-based 5B model, `generate.py` (single GPU or torchrun/FSDP), torch 2.10, flash-attn FA3/FA2, Triton INT8 GEMMs.
  - `Matrix-Game-2/`: Wan-based, unconditional `flash_attn` import, optional TensorRT/pycuda VAE.
  - `Matrix-Game-1/`: diffusers, torch 2.5.1, flash_attn 2.7.4 pinned; the `GameWorldScore` benchmark vendors DROID-SLAM, lietorch and RAFT CUDA extensions.

## Existing AMD support

**Improvable; one authoritative AMD effort, upstream and unreviewed.**

- https://github.com/SkyworkAI/Matrix-Game/pull/67 (ZJLi2013, AMD), "add AMD ROCm support via AITER CK flash attention backend". Open, `MERGEABLE`/`CLEAN` at head `9ad32b1d`, no comments or reviews, last updated 2026-09-29 (checked 2026-11-20). Touches only `Matrix-Game-3/wan/modules/attention.py` (+AITER `flash_attn_varlen_func` between FA3 and FA2, a clear error instead of the bare `assert`) and the MG-3 README. Tested by its author on MI300X / ROCm 7.2.
  - Judgement: authoritative (an AMD engineer, upstream PR), narrow (MG-3 attention only), stalled for lack of review. Value is in validating it and backing it, not in competing with it.
- No AMD/ROCm mention anywhere in upstream docs; no ROCm or AMD-Ecosystem fork other than ours (intake, rechecked 2026-11-20).
- Upstream does not link platform forks; it takes PRs (#67 is pending, not refused). An upstream PR is the right vehicle.
- Port vs rewrite: there are no NVIDIA-tuned CUDA kernels to rewrite. The GPU code is PyTorch, Triton, and third-party attention libraries. The Triton kernels are portable as written; a performance pass for AMD (tile/num_warps tuning) is out of scope for a first correctness port and is not planned.

## Build classification

**Pure Python (PyTorch + Triton), no compiled extension in the in-scope code.** `ext_type` recorded as `python-triton`.

Evidence:
- `Matrix-Game-4/pyproject.toml`: `setuptools.build_meta`, `packages.find include = ["matrix_game*"]`, no `ext_modules`, torch only as an optional `inference` dependency.
- `Matrix-Game-3/`: `requirements.txt` only, run in place (`python generate.py` / `torchrun`).
- `surface.py generate`: 0 CUDA files in the census. The only `.cu` files in the repository are vendored third-party extensions under `Matrix-Game-1/GameWorldScore/GameWorld/third_party/` (DROID-SLAM, lietorch, RAFT `alt_cuda_corr`, plus Eigen's own GPU benchmarks).
- `Matrix-Game-4/game/**/CMakeLists.txt`: a C++20 SDL2/OpenGL/FFmpeg game and reference server, no CUDA language, no CUDA library.

## Port strategy

Neither A nor B: there is nothing to hipify. This is a **runtime-portability port**: make the Python inference paths select working ROCm backends, fix whatever a real ROCm run shows breaking, and document the ROCm install. Torch's ROCm build already maps `torch.cuda.*`, CUDA graphs, SDPA, `torch.compile`/flex_attention and Triton onto HIP.

Dependency map for the in-scope paths:

| CUDA-side dependency | Where | ROCm route |
|---|---|---|
| PyTorch CUDA wheel | MG-4, MG-3 | PyTorch ROCm wheel of the pinned version (`torch==2.12.0` from `download.pytorch.org/whl/rocm<x>` for MG-4; `torch==2.10.0` for MG-3) |
| `flash_attn` (FA2) | MG-4 optional (SDPA fallback built in), MG-3 required unless `--fa_version 0` | Dao-AILab flash-attention ROCm build: CK backend on gfx90a/gfx942/gfx950, Triton backend (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`) on RDNA3/RDNA4. MOAT records flash-attention as `already-supported` (dispositioned by Jeff 2026-11-10). |
| `flash_attn_interface` (FA3, Hopper) | MG-4, MG-3 | Not used on AMD. MG-4 gates it on the device name containing "h100"/"hopper", so it stays off. MG-3 enables FA3 whenever the module imports (`WAN_FA_VERSION` default "3"); it must not be installed on ROCm. |
| AITER `aiter.ops.mha.flash_attn_varlen_func` | MG-3 via PR #67 | `pip install amd-aiter` or a source build of ROCm/aiter (CK). |
| SDPA, flex_attention, CUDA graphs | MG-4 | Native in PyTorch ROCm (AOTriton/CK SDPA; Inductor Triton for flex_attention; hipGraph). |
| Triton kernels (`kernels.py`, `rope_kernel.py`, MG-3 `triton_kernels.py`) | MG-4, MG-3 | Triton ROCm backend shipped with the torch ROCm wheel. |
| SageAttention 2.2 (`sm89_compile`, `_fused`: CUDA C++ with sm89 PTX `mma`) | MG-4 `-sage` presets and `rtx5080`'s Sage import path | No ROCm build exists. Out of scope under the Q2 recommendation. |
| `torch._scaled_mm` FP8 (e4m3fn), `scaled_mm` MXFP8 `SWIZZLE_32_4_4` | MG-4 `rtx5090`/`rtx5080`/`rtx5070ti` | FP8 e4m3fn: gfx950, gfx12 (gfx942 is FNUZ). MXFP8: gfx950 only, and PyTorch's ROCm MX path rejects `SWIZZLE_32_4_4` (it takes `NO_SWIZZLE` or `SWIZZLE_32_8`, checked in a torch 2.16 dev `aten/src/ATen/native/cuda/ScaledBlas.cpp`). Out of scope under the Q2 recommendation. |
| TensorRT, pycuda, onnxruntime-gpu | MG-2 optional VAE | No equivalent wanted (MIGraphX would be a rewrite). Out of scope under the Q1 recommendation. |

## CUDA surface inventory

No `__global__`/`__device__`, warp intrinsics, textures, or CUDA libraries in the in-scope code (census: 0 files). The device-specific surface is Python-level:

**Matrix-Game-4 (`Matrix-Game-4/matrix_game/inference/`)**
- `wan/modules/attention.py`: FA3 (name-gated) -> FA2 varlen -> SDPA fallback. The SDPA fallback ignores padding masks (warns). On ROCm without flash-attn the SDPA path is taken; with ROCm flash-attn, FA2.
- `runtime.py` (`CastleRuntime`, used by `realtime` = `ring-fused`): calls `flash_attn.flash_attn_varlen_func` directly when FA2 is importable, otherwise routes through `attention()` (SDPA); raises if FA3 is "available". Captures the 30-block stack into one CUDA graph on a side stream (`capture_error_mode="thread_local"`). Flash-attn's CK and Triton paths must be capture-safe after warmup; SDPA on ROCm is.
- `kernels.py`: Triton `ln_modulate`, `gate_ln_affine`, `gate_residual` (BF16 in, FP32 math, explicit RNE rounding via int32 bitcast). `num_warps` from a block-size table; on wave64 each "warp" is 64 lanes, which changes occupancy but not correctness.
- `rope_kernel.py`: Triton RoPE that rotates in FP64 (`tl.float64`). FP64 is native on CDNA; on RDNA it is slow but correct. Device check is `x.device.type != "cuda"`, true on ROCm.
- `vae_graph.py`, `text.py`: CUDA graph capture of TAEHV decode and T5 text encode.
- `wan/modules/causal_model.py`: `torch.compile(flex_attention, mode="default")` (quality/native path), FP64 complex RoPE in eager.
- `loader.py`: `torch.backends.cudnn.allow_tf32 = False` (harmless on ROCm).
- RTX presets (pending Q2): `rtx5090/` (FP8 `torch._scaled_mm`, Triton FP8 quantize with `tl.float8e4nv`, SDPA attention, block-stack graphs), `rtx5080/` (FP8 GEMMs, imports `rtx5090_sage.SageWindow` for its Sage path), `rtx5070ti/` (Triton INT8 GEMM with fused GELU, MXFP8 `scaled_mm` with `SWIZZLE_32_4_4` and `float8_e8m0fnu` scales), `rtx5090_sage/` (SageAttention CUDA kernels).

**Matrix-Game-3 (`Matrix-Game-3/`)**
- `wan/modules/attention.py`: FA3 if `flash_attn_interface` imports and `WAN_FA_VERSION` is "3" -> FA2 (bare `assert`) ; `--fa_version 0` selects SDPA. PR #67 inserts AITER before FA2.
- `wan/modules/action_module.py`: `flash_attn_interface` or `flash_attn` `flash_attn_func`, else a fallback; plus `flex_attention`.
- `wan/triton_kernels.py`: Triton INT8 quantize and autotuned INT8 GEMM with bias/GELU (`--use_int8`), FP8 quantize helper (`torch.float8_e4m3fn`, unused by the default int8 path).
- `wan/modules/vae2_2.py`: `torch.compile` of the VAE decoder (`--compile_vae`).
- `generate.py`, `wan/distributed/`: `dist.init_process_group(backend="nccl")` for torchrun/FSDP/Ulysses (RCCL on ROCm, transparent).

## Risk list

1. **CUDA graph capture of flash-attn on ROCm** (MG-4 `realtime` with flash-attn installed). CK varlen and the Triton backend allocate or autotune on first call; capture follows an eager warmup, which should cover it. If capture fails, the fix is in our fork's choice of backend under capture, not a change to flash-attn. Test both with and without flash-attn installed.
2. **SDPA fallback ignores padding masks** (MG-4 `attention()`). Harmless for the realtime runtime (frame-aligned, unpadded by construction; `CastleRuntime` raises otherwise) but the `quality`/native path passes `k_lens=context_lens` for cross-attention over padded text. Without flash-attn the text padding is attended to: a quality regression versus CUDA, not a crash. Decide from the frame-wise comparison whether the ROCm install note must require flash-attn for `quality`, or whether SDPA gets a mask (an upstream-visible change to shared code; keep the CUDA path identical).
3. **MG-3 enables FA3 whenever `flash_attn_interface` imports.** Some ROCm flash-attention or AITER packagings may ship a module of that name; if so MG-3 would call a Hopper API. The validator records whether the module exists in the ROCm environment.
4. **PR #67's AITER path drops `window_size`, `dropout_p` and `deterministic`.** MG-3 uses the defaults ((-1, -1), 0.0, False) everywhere in inference, so this is benign today; note it in the review comment.
5. **Triton autotune configs written for NVIDIA** (MG-3 `int8_gemm_bias_kernel`: `num_stages`, `BLOCK_*`, `SPLIT_K`). Large `num_stages` can exceed the 64 KB LDS on AMD and fail to compile or be pruned. A failure is fixable with an AMD-guarded config list.
6. **Memory on the wave32 host.** MG-4 bf16 `realtime` is the A100-80 GB preset: ~10 GB DiT checkpoint, umT5-XXL encoder (~11 GB bf16, released after the fixed prompt is encoded), TAEHV, a 21-frame KV window. Upstream reaches 16 GB cards only through the INT8/FP8 RTX presets. If the wave32 card cannot hold `realtime`, the wave32 gate needs Q2-B (an AMD-runnable reduced-memory preset) or a waiver Jeff approves. The validator records peak `torch.cuda.max_memory_allocated()` on wave64 first, so this is known before the wave32 run.
7. **FP64 RoPE on RDNA**: correct but slow; throughput on wave32 is reported, not gated.
8. **Numerical comparison has no shipped reference.** Upstream's GPU release gate (docs/testing.md) is a self-consistency check: the same clip through `--dit-runtime native --no-vae-graph` and through the fast runtime, compared numerically. A CUDA reference video from us is not available on this fleet; use upstream's own native-vs-fast comparison on each AMD arch, plus visual sanity (non-degenerate frames, no NaN), plus native-runtime agreement between wave64 and wave32.
9. **Torch version pins.** MG-4 pins `torch==2.12.0`, MG-3 `torch==2.10.0`. Validate against the matching PyTorch ROCm wheels where they exist; record the exact wheel and ROCm version. A newer torch is acceptable evidence only if the matching one is unavailable, and must be stated.
10. No wavefront, texture, OOB, rule-of-five, pitch or library-swap risks: there is no device C++.

## File-by-file change list

Fork branch `moat-port`, smallest complete change. Everything below is additive on ROCm and leaves the CUDA path byte-identical in behaviour.

1. `Matrix-Game-4/README.md` and `Matrix-Game-4/docs/installation.md`: an "AMD ROCm" install subsection: torch 2.12.0 ROCm wheel, optional ROCm flash-attention (CK on Instinct, `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` on Radeon), tested GPUs, which presets run on AMD, that `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` both select the device. Requirements line gains "or AMD Instinct/Radeon with ROCm". No device-selection variables in front of example commands.
2. `Matrix-Game-4/matrix_game/inference/wan/modules/attention.py` and `runtime.py`: **only if a ROCm run shows a need** (risk 1 or 2). Expected: no change. If MG-4 should also offer AITER (mirroring #67), that is a separate, optional commit and needs evidence it beats SDPA/FA2.
3. `Matrix-Game-4/matrix_game/scenes.py` / `loader.py`: if an RTX preset is selected on a ROCm device, fail early with a clear message rather than deep in a Triton or `scaled_mm` call (only under Q2-A; a few lines, `torch.version.hip` gated).
4. `Matrix-Game-3/`: **no code change in our fork** under the recommended route. PR #67 is the MG-3 change; we validate its head and, with Jeff's approval, comment with results. If #67 needs fixes (risk 3 or 5), propose them as a review suggestion on #67, never as a competing PR. MG-3 also runs on ROCm with zero code change through the ROCm flash-attention FA2 build; the validator records both routes.
5. Q2-B only: `rtx5070ti/linear.py` (MXFP8 scales in `NO_SWIZZLE` layout and `SwizzleType.NO_SWIZZLE` on ROCm, or fall back to INT8/BF16 for `cross_q` where MX is unsupported), `rtx5090/fp8_linear.py` and `rtx5080/linears.py` (select `float8_e4m3fnuz` on gfx942 the way PyTorch's own tests pick `e4m3_type`), a ROCm-guarded refusal of the Sage paths. Estimated 50-100 lines plus validation on gfx950 and one gfx12 card.
6. Q1-B/C only: MG-2: make `wan/modules/action_module.py`'s top-level `from flash_attn import flash_attn_func` optional like its siblings; document that the TensorRT VAE is NVIDIA-only. MG-1: `GameWorldScore` CUDA extensions (DROID-SLAM, lietorch, RAFT `alt_cuda_corr`) are torch extensions needing a Strategy B port, a separate and much larger piece of work.

Commit titles: `[ROCm] ...` for the docs and any ROCm-gated code. Run `python3 utils/jargon.py --port Matrix-Game` before every push.

## Build commands

No compile step for the in-scope code. Environment per arch (validator; not run by the planner):

```bash
# Matrix-Game-4 (from Matrix-Game-4/)
python3.12 -m venv .venv && . .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/rocm<ver>
python -m pip install -e '.[inference,server,download,test]'
# optional, second pass: ROCm flash-attention (Instinct: CK backend)
#   git clone https://github.com/Dao-AILab/flash-attention && cd flash-attention && GPU_ARCHS=<gfx> MAX_JOBS=<physical cores> pip install --no-build-isolation .
# optional, Radeon: FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE pip install --no-build-isolation .
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"

# Matrix-Game-3 (from Matrix-Game-3/), separate venv
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/rocm<ver>
grep -v -E '^torch|flash.attn' requirements.txt | python -m pip install -r /dev/stdin
# route 1: ROCm flash-attention FA2 (as above); route 2: PR #67 head + pip install amd-aiter
```

Dependency builds go into the job directory, never the image's shared paths. Wrap installs and runs with `utils/timeit.sh Matrix-Game <phase> -- ...`.

Weights (pinned revision from the README, verified by SHA256):

```bash
matrix-game download --manifest releases/models.json --repo-id Skywork/Matrix-Game-4.0 --repo-type model \
  --revision 2c53100457da7abcf0e966f75637c644d3423a63 --scene ancient-egypt --resources resources
matrix-game download --manifest releases/game-assets.json --repo-id Skywork/Matrix-Game-4.0 --repo-type model \
  --revision 2c53100457da7abcf0e966f75637c644d3423a63 --scene ancient-egypt --resources resources
huggingface-cli download Skywork/Matrix-Game-3.0 --local-dir Matrix-Game-3/Matrix-Game-3.0
```

The MG-3 README writes `huggingface-cli download Matrix-Game-3.0` with no organisation; `Skywork/Matrix-Game-3.0` is the planner's guess, unverified. Confirm the repository id before downloading.

## Test plan

Upstream has no GPU tests. The gate is end-to-end inference plus upstream's own native-vs-fast consistency check (docs/testing.md, "GPU Release Gate").

**Non-GPU regression set (must not regress, every arch):**
- `cd Matrix-Game-4 && python -m unittest discover -s tests -p 'test_*.py'` (CLI, preprocessing pixel-exactness, sink/cache contracts, resources; needs torch installed, no GPU, no weights). Record the pass/fail/skip counts against the same command with the CPU torch wheel or a CUDA reference count if one is available.
- Optional: `node tests/browser/test_audio.cjs`, `node tests/browser/test_hud.cjs` (pure JS, unaffected by the port).

**GPU gate, Matrix-Game-4 (wave64 and wave32):**
1. An input clip of more than 21 latent chunks (>84 frames at 4 frames per chunk) so cache eviction runs: record one locally with `bash scripts/build_game.sh` + `scripts/play.sh` if a GL context is available, or use any low-poly RGB mp4 from the asset bundle (`resources/game`), fixed for all runs; record its sha256.
2. Fast path: `matrix-game render --scene ancient-egypt --input <clip> --resources resources --preset realtime --prompt-mode fixed --output out/realtime` (TAEHV, 1 step, ring-fused, graphs).
3. Native reference with the same codec and timesteps: `... --preset realtime --dit-runtime native --no-vae-graph --output out/native`.
4. Quality path: `... --preset quality --output out/quality` (Wan VAE, 2 steps, flex_attention, native DiT).
5. Each of 2-4 twice: without flash-attn (SDPA), then with ROCm flash-attn installed.
6. Checks: every output decodes, frame count matches input, no NaN/Inf, frames non-constant; per-frame PSNR and max abs difference of `realtime` vs `native` (upstream says not bitwise; record the numbers and compare with what upstream reports, if anything); wave64 vs wave32 native-runtime PSNR; throughput and peak memory from the run report (`--profile-chunks` where available). A kernel-dispatch count (`rocprofv3 --kernel-trace` on a short run) proves the Triton and attention kernels actually ran on the GPU.
7. Racing scene once with `--prompt-mode dynamic` on wave64 (exercises the T5 graph and caption refresh).

**GPU gate, Matrix-Game-3 (wave64 required; wave32 if memory allows):**
1. Single GPU: `python generate.py --size 704*1280 --ckpt_dir Matrix-Game-3.0 --fa_version 2 --use_int8 --num_iterations 4 --num_inference_steps 3 --image demo_images/001/image.png --prompt "<README prompt>" --seed 42 --vae_type mg_lightvae --lightvae_pruning_rate 0.5 --output_dir out --save_name rocm_fa2` on: (a) upstream `main` + ROCm flash-attention; (b) PR #67 head `9ad32b1d` + AITER; (c) `--fa_version 0` (SDPA) as the in-tree reference. Compare (a)/(b) against (c) frame-wise.
2. Once with `--compile_vae` and once without.
3. Multi-GPU torchrun (`--dit_fsdp --t5_fsdp --ulysses_size N`) only if the validating host can give the job several GPUs without disturbing other jobs; otherwise record it as not run.

**CUDA no-regression:** there is no CUDA compile. The CUDA path's guarantee is code review: every change is docs or `torch.version.hip`-gated. If any shared Python changes (risk 2), the reviewer confirms the CUDA branch is unchanged line by line.

**Windows:** the `windows` gate is waived (Jeff, 2026-11-12). No Windows run.

## Open questions

### Q1. Which model generations are in scope? (Jeff rules; porter waits)

- **A (recommended): Matrix-Game-4 + Matrix-Game-3.** MG-4 is the current release and MG-3 is what PR #67 targets. MG-1 and MG-2 are recorded `scoped_out` as superseded legacy generations. Cost: as planned above.
- **B: A + Matrix-Game-2.** Adds one optional-import fix in `wan/modules/action_module.py`, a README note that its TensorRT/pycuda VAE acceleration is NVIDIA-only, and an end-to-end `inference.py` run on wave64. Roughly one extra validation day; weights are a separate download.
- **C: B + Matrix-Game-1.** MG-1's inference is plain diffusers + flash_attn (cheap), but its `GameWorldScore` benchmark vendors DROID-SLAM, lietorch and RAFT CUDA extensions (12 `.cu` files plus Eigen), which is a genuine torch-extension HIP port of third-party code pinned to torch 2.5.1. Recommend scoping out `GameWorldScore` even under C.

### Q2. Matrix-Game-4's RTX presets? (Jeff rules; porter waits)

- **A (recommended): scope out all six RTX presets.** They are hand-tuned for specific NVIDIA consumer GPUs (names, SM counts and sm_120 numbers are in their configs). AMD users get `realtime`/`quality`/`baseline`. The port adds an early, clear refusal on ROCm. Risk: if the wave32 card cannot hold bf16 `realtime` (risk 6), wave32 has no runnable fast preset and needs B or a waiver.
- **B: make the three non-Sage RTX presets run where the hardware has the datatypes.** FP8 on gfx950/gfx12 (FNUZ selection on gfx942), MXFP8 only on gfx950 with ROCm's scale layout, INT8 Triton GEMM everywhere. 50-100 lines; validated on gfx950 and a gfx12 card. Gives Radeon 16 GB cards a fitting preset. Presets keep NVIDIA names, which is odd for AMD users; an AMD-named preset alias would be a further upstream-visible design choice.
- **C: B + the `-sage` presets.** Needs SageAttention 2.2 on ROCm (CUDA C++ with sm89 PTX `mma`, plus the patch Matrix-Game ships). No ROCm build exists; this is a new project for intake, not part of this port.

### Not questions, recorded for the porter

- Delivery: one upstream PR from `moat-port` touching Matrix-Game-4 only. MG-3 goes through PR #67; any comment on #67 is drafted into notes.md for Jeff's approval, never posted by an agent.
- `flash-attention` (hard dependency) is dispositioned `already-supported`; `moatlib.py deps` reports READY.
