# Wan2.2 port plan

Planned 2026-11-20 on linux-gfx942 (planner role, no GPU used). Upstream read at `1ea34ff48f87` (main, 2026-09-21).

**Decisions from Jeff are needed before a porter starts.** They are listed under "Open questions" at the end of this plan and at the top of the planning section in `notes.md`. Where the plan says "default", that is the recommended answer, not a ruling.

## Project

- Name: Wan2.2
- Upstream: https://github.com/Wan-Video/Wan2.2 (Apache-2.0), default branch `main`
- Fork: https://github.com/AMD-Ecosystem/Wan2.2, port branch `moat-port`
- Kind: an inference application, pure Python on PyTorch. Five video-generation pipelines (`generate.py --task t2v-A14B | i2v-A14B | ti2v-5B | s2v-14B | animate-14B`), single GPU or multi-GPU (torchrun, FSDP plus DeepSpeed-style Ulysses sequence parallelism over NCCL), plus an Animate preprocessing tool.
- Windows gate: waived by Jeff on 2026-11-12 (recorded in status.json). Required gates left: wave64 and wave32.

## Existing AMD support

Category: **improvable**. Nothing to inherit; the value is validation, a ROCm install note, and whatever fixes validation finds.

- https://github.com/Wan-Video/Wan2.2/pull/30 (zhangnju, AMD; open since 2025-07-29, mergeable, no maintainer response). Authoritative (AMD-authored) but tiny: a log hint in `generate.py` telling ROCm users on CDNA3 that `--frame_num` can be raised to use HBM. It adds no ROCm support code; its body reports that all Wan2.2 models run on CDNA3. Our port must not compete with it: do not touch `--frame_num` help text or add a similar hint. Validation evidence from this port is the natural way to back #30 (a comment there is a GitHub write, so it is Jeff's call after validation).
- https://github.com/Wan-Video/Wan2.2/pull/390 (third party, open, mergeable, 2026-09-18): moves `flash_attn` out of `requirements.txt` into a README step, makes s2v/animate imports lazy, replaces `assert FLASH_ATTN_2_AVAILABLE` with a `RuntimeError`. It overlaps the files our install note touches (`README.md`, `requirements.txt`, `wan/modules/attention.py`). The porter must keep our diff away from the lines #390 changes so both can merge; the ROCm note goes in its own README/INSTALL.md subsection.
- Community (hints only, not code): dikullu/wan2gp-rocm, yangxiarui4-dotcom/wan22-strix-halo-windows-guide.
- Delivery vehicle: an ordinary upstream PR is right (Alibaba upstream, Apache-2.0, does not link platform forks). Upstream has answered neither #30 nor most install PRs, so the fork's `moat-port` may end up the practical deliverable; that is a later ruling, not a planning one.
- Port vs AMD-native rewrite: not applicable. There is no CUDA kernel code. The performance-critical piece is attention, provided by the flash-attention package, whose ROCm backends (Composable Kernel; Triton via aiter) are upstream and AMD-maintained (MOAT disposition `already-supported` for Dao-AILab/flash-attention).

## Build classification

**No native build at all.** `ext_type`: `python-pytorch`.

Evidence:
- `find . -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.c' -o -name 'CMakeLists.txt' -o -name setup.py` -> nothing. `utils/surface.py generate` reports 0 CUDA files and 0 components.
- `pyproject.toml`: setuptools, `packages = ["wan"]`, pure-Python package data (`"wan" = ["**/*.py"]`). No `CUDAExtension`.
- Install is `pip install -r requirements.txt` (+ `requirements_s2v.txt`, `requirements_animate.txt`). The only native code is in dependencies: torch/torchvision/torchaudio (ROCm wheels exist), `flash_attn` (builds from source; ROCm backends upstream), `decord` (CPU video decode), `onnxruntime` (Animate preprocessing), SAM-2 (Animate preprocessing; optional CUDA extension), CosyVoice (optional S2V TTS, external package).

## Port strategy

Neither Strategy A nor B: there is nothing to hipify. The port is:

1. **Environment**: ROCm PyTorch wheel, flash-attention built from upstream source with its ROCm backend (CK by default on CDNA; `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` on RDNA where CK has no instances for the card, or to compare), the rest from `requirements*.txt`. `torch.cuda.*`, `torch.amp.autocast('cuda')`, `device.type == 'cuda'` and `backend="nccl"` all work unchanged on ROCm PyTorch (HIP masquerades as `cuda`; `nccl` maps to RCCL).
2. **Documentation**: a ROCm install subsection (README "Installation" and INSTALL.md) giving the ROCm torch index URL and the flash-attention ROCm build command. This is the core upstream-visible deliverable, since `pip install -r requirements.txt` builds `flash_attn` against whatever torch is present and fails or picks the wrong backend without guidance.
3. **Code changes only where an NVIDIA-specific path exists or validation finds a fault**. One is known in advance (Animate preprocessing, below, depending on the scope ruling). Everything else is "validate, fix what breaks".

## CUDA surface inventory

No kernels, no `__global__`, no warp intrinsics, no CUDA libraries called directly. The GPU-specific surface is at the Python level:

| Surface | Where | ROCm mapping | Risk |
|---|---|---|---|
| `flash_attn.flash_attn_varlen_func` (FA2), via `flash_attention()` | `wan/modules/attention.py:113`; called from `model.py:145,175`, `distributed/ulysses.py:37`, `s2v/motioner.py`, `s2v/model_s2v.py`, `animate/model_animate.py`, `animate/clip.py` | flash-attention ROCm backend (CK on gfx90a/gfx942/gfx950; CK or Triton on RDNA3/4) | **Mandatory.** Every pipeline calls `flash_attention()` directly and it does `assert FLASH_ATTN_2_AVAILABLE`. The SDPA fallback lives only in `attention()`, which nothing calls (confirmed by grep; also noted in PR #390). The intake note's "falls back to SDPA" is therefore wrong for real runs. Arguments used: varlen, `dropout_p=0`, `causal` (CLIP only), `window_size=(-1,-1)`, `deterministic=False`; head dim 128 (DiT), 80 (CLIP ViT-H), all bf16. All are inside CK FMHA fwd coverage; confirm on each arch. |
| `flash_attn_interface.flash_attn_varlen_func` (FA3) | `attention.py:96`, taken whenever `flash_attn_interface` imports | FA3 Hopper kernels: no AMD target. flash-attention's `hopper/` setup can install an `flash_attn_interface` module on ROCm (Triton backend) | If a ROCm env has that module installed, Wan takes the FA3 path, which passes `seqused_q/k` and indexes `[0]` of the result (signature drift, cf. upstream #312). Validator records `FLASH_ATTN_3_AVAILABLE` and runs with it False by default. |
| `flash_attn.flash_attn_func` (non-varlen) | `animate/face_blocks.py:73`, `s2v/auxi_blocks.py:76` (mode `"flash"` default; `flash_attn_func=None` on ImportError, then a crash) | same backend | Mandatory for S2V and Animate. |
| transformers `attn_implementation="flash_attention_2"` | `wan/utils/prompt_extend.py:323,332` (local Qwen prompt extension) | transformers detects ROCm flash-attn | Pin `transformers<=4.51.3` (requirements) must accept the ROCm flash_attn version; check. |
| `torch.amp.autocast('cuda', ...)`, `torch.cuda.amp.autocast`, `torch.cuda.synchronize/empty_cache/current_device` | throughout `wan/` | unchanged on ROCm PyTorch | `animate.py:482` passes `device_type=str(self.device)`; an upstream issue if it fails, not ROCm-specific. |
| float64 / complex128 RoPE (`torch.polar`, `view_as_complex` in fp64) | `model.py:28-66`, `s2v/motioner.py:512` | supported on ROCm | Slow on RDNA (weak fp64), correctness fine. Note timings. |
| `dist.init_process_group(backend="nccl")`, FSDP, all-to-all for Ulysses | `generate.py:327-333`, `wan/distributed/*` | RCCL | Needs several GPUs of one host at once (see Test plan). |
| onnxruntime `CUDAExecutionProvider` | `animate/preprocess/pose2d.py:26,61` | onnxruntime with MIGraphX EP (`MIGraphXExecutionProvider`; the ROCm EP was removed from recent onnxruntime) | **Known NVIDIA-only path.** On ROCm the CUDA EP is absent and onnxruntime silently runs pose detection on the `CPUExecutionProvider` that the list also names. Per the ROCm-library-over-CPU-fallback rule this needs a provider selection that prefers MIGraphX when available. Scope ruling Q1. |
| SAM-2 (`facebookresearch/sam2` @ `0e78a118`, editable install) | `animate/preprocess/sam_utils.py`, `process_pipepline.py` (replace mode) | SAM-2's optional `_C` CUDA extension (connected components); its setup tolerates a failed extension build and falls back | Build it with ROCm torch (torch hipify) or accept the documented fallback; record which. |
| FLUX Kontext via diffusers `.to("cuda")` | `process_pipepline.py:36` (optional relighting reference) | unchanged | Large extra weights; optional. |
| CosyVoice TTS | `speech2video.py:682-692` (`--enable_tts`) | external package (FunAudioLLM/CosyVoice) with its own onnxruntime/TensorRT options | Not part of this repo; scope ruling Q1. |

## Risk list

- **Wavefront width**: none in this repo (no device code). Wave-size risk lives entirely in flash-attention's backends, which is why both wave64 and wave32 runs are still required.
- **flash-attention on RDNA**: CK FMHA instances for gfx11/gfx12 are newer and narrower than CDNA's; the Triton backend is the fallback. Record which backend each platform ran and the flash-attention commit.
- **Memory**: A14B MoE (two ~14B experts), S2V-14B and Animate-14B need ~80 GB on one GPU (README) or FSDP over several. MI300-class (192 GB) runs them single-GPU; typical Radeon cards (16-48 GB) can run only TI2V-5B with `--offload_model True --convert_model_dtype --t5_cpu`. Scope ruling Q2.
- **Weights**: T2V-A14B and I2V-A14B ~126 GB each, TI2V-5B ~34 GB, S2V-14B and Animate-14B tens of GB each, plus Qwen2.5-3B(-VL) for prompt extension. Download time and disk are the validator's dominant cost; use a shared HF cache under the job directory.
- **transformers pin** `<=4.51.3` vs a current ROCm torch: may need a newer transformers; any pin change is a shared change and goes to Jeff as a finding, not a silent edit.
- **numpy<2 and decord**: decord has no recent wheels for new Pythons; may force Python 3.10/3.11 venv. Environment issue, record it.
- **FA3 path** taken accidentally (above).
- **Output correctness**: generation is stochastic, but `--base_seed` fixes it. There is no CUDA reference on this fleet's AMD hosts; the validator checks outputs are non-degenerate (no NaN/black frames; decode succeeds; visual sanity frame dumps) and, where a CUDA host is available, compares frame PSNR against an identical-seed CUDA run. Bit-exactness is not expected across vendors.
- **Overlap with PR #30 / #390** (Existing AMD support).

## File-by-file change list

Expected minimal, contingent on validation:

| File | Change | Condition |
|---|---|---|
| `README.md` (Installation) and `INSTALL.md` | New "ROCm (AMD GPUs)" subsection: ROCm torch install, flash-attention ROCm build (`pip install flash-attn --no-build-isolation` from upstream source with ROCm torch present, CK default; `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` for the Triton backend), the onnxruntime MIGraphX package for Animate preprocessing. No device-selection environment variables in front of example commands. | always |
| `wan/modules/animate/preprocess/pose2d.py` | Provider selection: when `torch.version.hip` is set and `MIGraphXExecutionProvider` is in `onnxruntime.get_available_providers()`, use it (device id from the torch device), else keep the current CUDA EP list; CPU EP stays last. Both `__init__` (line 26) and `set_providers` path (line 61). CUDA behaviour byte-identical. | Q1 option A |
| `requirements_animate.txt` | Nothing by default; the ROCm onnxruntime package is named in the docs rather than replacing `onnxruntime` for everyone. | -- |
| anything else | Only for a fault found in validation, with the evidence in notes.md. | -- |

Explicitly not changed: `--frame_num` and any HBM hint (PR #30's territory); `requirements.txt` `flash_attn` line and the `assert` in `attention.py` (PR #390's territory); `attention()` / SDPA fallback (Q3).

## Build commands

No build of the project. Environment, in a job-local venv (Linux, any arch):

```
python3 -m venv $VENV && . $VENV/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm<ver>   # or the host's ROCm torch
git clone https://github.com/Dao-AILab/flash-attention && cd flash-attention
pip install --no-build-isolation .          # CK backend (CDNA default)
# RDNA, or to cross-check: FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE pip install --no-build-isolation .
cd <fork> && pip install -r requirements.txt  # flash_attn already satisfied
pip install -r requirements_s2v.txt           # Q1
pip install -r requirements_animate.txt       # Q1 (plus an onnxruntime build with the MIGraphX EP for option A; confirm the package name and source)
python -c "import flash_attn, wan.modules.attention as a; print(flash_attn.__version__, a.FLASH_ATTN_2_AVAILABLE, a.FLASH_ATTN_3_AVAILABLE)"
```

Record torch, ROCm, flash-attention commit and backend, transformers, onnxruntime versions in notes.md. Weights: `huggingface-cli download Wan-AI/Wan2.2-<model> --local-dir <models>/Wan2.2-<model>`.

## Test plan

Upstream has no unit tests. The real suite is `tests/test.sh <model dir> <gpus>`: torchrun runs of `generate.py` for **t2v-A14B, i2v-A14B and ti2v-5B only** (the intake note listed s2v and animate as part of it; they are not), all multi-GPU with `--dit_fsdp --t5_fsdp --ulysses_size $GPUS`, including local-Qwen prompt extension; the DashScope case runs only with `DASH_API_KEY`. Its 1-GPU cases are commented out.

GPU gate, per platform (wave64 = MI300-class; wave32 = an RDNA card):

1. Import check above; `FLASH_ATTN_2_AVAILABLE=True`, `FLASH_ATTN_3_AVAILABLE=False`.
2. Single-GPU README commands for every in-scope task, fixed `--base_seed 42`, default steps:
   - `python generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ... --offload_model True --convert_model_dtype`
   - `--task i2v-A14B --size 1280*720 --image examples/i2v_input.JPG ...`
   - `--task ti2v-5B --size 1280*704` (t2v) and with `--image examples/i2v_input.JPG` (i2v), with `--offload_model True --convert_model_dtype --t5_cpu` on a Radeon card
   - `--task s2v-14B --size 1024*704 --image examples/i2v_input.JPG --audio "examples/Five Hundred Miles.MP3"` (Q1)
   - `--task animate-14B` in animation and `--replace_flag` modes after running the preprocessing tool on its example inputs (Q1)
3. Multi-GPU: `bash tests/test.sh <models> <N>` with N >= 2 (wave64 host; the upstream suite as written). Needs N whole GPUs of one host for the run; coordinate with other jobs (RCCL collectives need them at once).
4. Pass criteria per run: exit 0, an `.mp4` written that decodes with the expected frame count and size, no NaN/inf in the latent (add a one-off check in agent_space, not the fork), frames not constant; save a few frames as evidence. Record wall time and peak memory (`torch.cuda.max_memory_allocated`) for comparison with README's efficiency table.
5. Animate preprocessing (Q1 A): confirm onnxruntime reports `MIGraphXExecutionProvider` as the session's active provider (`session.get_providers()`), not CPU.

Non-GPU regression set: none upstream (no CPU tests; `import wan` itself needs a GPU because of `t5.py:478`). The CUDA-path check for this port is that every change is either docs or an additive branch that leaves the CUDA provider list identical; there is nothing to compile with nvcc. If `pose2d.py` changes, run it once on a CUDA host if one is available, otherwise record `cuda-not-validated` with the reason (no NVIDIA GPU; change is an additive `torch.version.hip` branch).

## Open questions (Jeff must rule before a porter starts)

**Q1. Which pipelines are in scope?** Five tasks exist; upstream's own suite covers only three.
- A (default, full): all five generation tasks, Animate preprocessing on the GPU (pose2d gets MIGraphX provider selection; SAM-2 built with ROCm torch or its documented fallback recorded), S2V TTS via CosyVoice attempted and, if CosyVoice itself does not install on ROCm, recorded as a deferral against CosyVoice rather than cut silently.
- B: all five generation tasks; Animate preprocessing left on onnxruntime's CPU fallback (scoped out for GPU) and CosyVoice TTS scoped out. A partial port.
- C: only t2v-A14B, i2v-A14B, ti2v-5B (what `tests/test.sh` runs); S2V and Animate deferred. A partial port.

**Q2. What does the wave32 gate require?** The 14B tasks need ~80 GB, beyond most Radeon cards.
- A (default): wave32 is satisfied by TI2V-5B (t2v and i2v modes, single GPU with offload options) plus every 14B task the card's memory allows, each attempted and its outcome recorded; the 14B tasks are proven on wave64.
- B: every in-scope task must pass on wave32, which needs a 48 GB+ RDNA card and may still OOM; if unmet, the gate stays open.

**Q3. Should the port make flash-attention optional (SDPA fallback in `flash_attention()`)?**
- A (default): no. ROCm has a working flash-attention, so the port does not need it; it changes shared code paths and collides with PR #390. Optionally register a deferral for it.
- B: yes, route `flash_attention()` to `torch.nn.functional.scaled_dot_product_attention` (with a varlen mask) when neither FA2 nor FA3 imports; useful where no flash-attention build exists (Windows, new RDNA), but it is a feature beyond a ROCm port.

**Q4 (after validation, not blocking the porter).** Whether to comment on AMD's PR #30 with the validation evidence, and whether to coordinate with PR #390's author on the install section.
