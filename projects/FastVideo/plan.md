# FastVideo: plan

Planned 2026-11-20 on linux-gfx1100 against the fork's `main` at `4deb3a50fff8` (upstream mirror of 2026-11-10). Upstream `main` was at `5af9897` (2026-11-19) by then, 23 commits ahead; that delta adds only another CUDA sm_121a library, which is off on ROCm (see "Base"). Two of the open questions at the end need Jeff's ruling before a porter starts: Q1 (Windows) and Q2 (confirming the CUDA-arch-only kernels stay out).

## Project

- Upstream: https://github.com/hao-ai-lab/FastVideo (Apache-2.0), default branch `main`.
- Fork: https://github.com/AMD-Ecosystem/FastVideo. `main` mirrors upstream. No `moat-port` branch exists yet.
- Kind of project: a **validation project** (Jeff's ruling of 2026-11-10 in notes.md). The job is to validate upstream's ROCm platform (#669) and its Triton video sparse attention route (#1851) on every required gate (wave64, wave32, windows), fix what breaks on `moat-port`, and send those fixes upstream. Nothing CUDA gets translated to HIP.

## Existing AMD support

Upstream already supports ROCm and the support is improvable. It is maintained in upstream itself, so it counts as authoritative. Nobody runs it in CI: `.github/workflows` and `.buildkite` have no AMD runner.

| effort | URL | judgement |
|---|---|---|
| ROCm platform, single-GPU inference | https://github.com/hao-ai-lab/FastVideo/pull/669 (sopiko99, AMD, merged 2025-07-30) | authoritative, upstream |
| VSA backends routed to Triton on ROCm | https://github.com/hao-ai-lab/FastVideo/pull/1851 (merged 2026-10-07) | authoritative, upstream |
| detection without amdsmi | https://github.com/hao-ai-lab/FastVideo/pull/1849 (merged) | authoritative, upstream |
| Python + Triton kernel package when HIP cannot be configured | https://github.com/hao-ai-lab/FastVideo/pull/1871 (merged) | authoritative, upstream |
| FP8 `_scaled_mm` gated on gfx950 | https://github.com/hao-ai-lab/FastVideo/pull/1859 (merged) | authoritative, upstream |
| gfx950 VSA Triton autotune configs | https://github.com/hao-ai-lab/FastVideo/pull/1860 (stayinalive181, open, updated 2026-11-10) | active third-party work; do not duplicate |
| **Native Windows: ROCm detection + gloo without NCCL** | https://github.com/hao-ai-lab/FastVideo/pull/2117, issue #2116 (klin2024, opened 2026-11-19, open, under bot review, mergeStateStatus BLOCKED on review/CI) | active third-party work that covers exactly the Windows gaps this plan found; do not duplicate (Q1) |
| `amd-wangfan/FastVideo` "add vsa triton opt on gfx942" | https://github.com/amd-wangfan/FastVideo | stale personal fork; compare its one commit with #1851 during porting (task P6), never inherit it |

Decision: **validate and improve; do not re-port.** No kernel is rewritten for AMD hardware. The NVIDIA-tuned kernels (ThunderKittens Hopper wgmma/TMA, Blackwell tcgen05 PTX, CUTLASS/CuTe FP4) already have upstream's Triton fallbacks, and upstream is tuning those fallbacks for AMD (#1860). A HipKittens or Composable Kernel VSA would compete with the direction upstream chose rather than fill a gap. The vehicle is an ordinary upstream PR: upstream merges AMD work (five PRs from one contributor since September) and does not point users at platform forks.

## Build classification

Two builds, one per package:

- `fastvideo` is pure Python (setuptools, `pyproject.toml`). It has no compiled code.
- `fastvideo-kernel` is a scikit-build-core CMake **torch extension**: `fastvideo-kernel/CMakeLists.txt` calls `find_package(Torch REQUIRED)` (line ~170) and `Python_add_library(fastvideo_kernel_ops ...)` (line ~504), and `pyproject.toml` declares `build-backend = "scikit_build_core.build"` with `torch==2.12.0` in its build requirements. Recorded with `set-ext-type FastVideo torch-extension`.

On ROCm the CMake already branches (`GPU_BACKEND=ROCM` from `build.sh --rocm`):
- lines 34-52: `enable_language(HIP)` when a HIP toolchain is found. Otherwise, under AUTO, `BUILD_CXX_KERNELS OFF`, which builds the Python + Triton package alone.
- lines 86, 178, 323, 559: the CUDA arch probe, the NCCL-device Ulysses kernel, attn_qat_infer FP4 and the CUDA runtime link are all skipped on ROCm.
- lines 427-505: when `BUILD_CXX_KERNELS` is ON, `EXTENSION_SOURCES` is still `common_extension.cpp` plus four turbodiffusion `.cu` files. With CUDA not enabled, CMake gives those `.cu` files no compiler, so the module is likely built from `common_extension.cpp` alone. That file calls `register_quant` / `register_rms_norm` / `register_layer_norm` / `register_gemm` unconditionally (lines 89-92), so the result would fail to import with undefined symbols. Every Python import of `fastvideo_kernel._C` is wrapped in try/except and falls back to Triton, which would make this a silent dead artifact rather than a crash. **Not verified by a build** (R1, task P1).

## Port strategy

Neither A nor B in the usual sense. Nothing is hipified, because the ROCm path has no CUDA C++ on it. The strategy has four parts:

1. Validate the existing ROCm paths on each gate, following the test plan.
2. Fix what breaks, in Python or in the build, keeping every change inside a ROCm or HIP guard (`torch.version.hip`, `current_platform.is_rocm()`, `GPU_BACKEND STREQUAL "ROCM"`) so the CUDA path stays byte-identical.
3. Make the ROCm build and install path honest: the build should not produce a dead `.so`, and the install should not pull in a CUDA wheel or a CUDA torch.
4. Document a reproducible ROCm walkthrough in `docs/getting_started/installation/amd.md`, filling the gap that page names itself ("Documentation still needed").

## CUDA surface inventory

`surface.py` census: 52 CUDA files, with markers kernel 36, device_code 172, launch 43, warp_intrinsic 27, and libraries cutlass 163, nccl 48, cudnn 2. Every one of them is NVIDIA-arch-specific and already excluded from the ROCm build:

| source | what | arch | ROCm today | plan |
|---|---|---|---|---|
| `csrc/attention/st_attn_h100.cu`, `block_sparse_h100.cu` + `include/tk` | ThunderKittens STA / VSA fwd+bwd | sm_90a only (`-DKITTENS_HOPPER`) | off (TK AUTO needs 9.0a in the CUDA arch list) | out; Triton `block_sparse_attn_triton.py` is the ROCm path |
| `csrc/attention/block_sparse_*_blackwell.cu[h]` | VSA fwd/bwd, tcgen05/setmaxnreg PTX | sm_100a/103a | off (arch list gate) | out; `is_supported()` falls back to Triton |
| `attn_qat_infer/blackwell/*`, `quantization/fp4_quantization_4d.cu` | FP4 attention / quant, CUTLASS/CuTe | sm_120a/121a | off (`GPU_BACKEND ROCM`, line 323) | out (Q2) |
| `csrc/quantization/minimax_h3_swiglu_gemm.cu` (upstream after 4deb3a5) | H3 SwiGLU NVFP4 GEMM, CUTLASS | sm_121a | off (needs `ENABLE_ATTN_QAT_INFER`) | out (Q2) |
| `csrc/comm/ulysses_all_to_all.cu` | NCCL 2.29 device-API all-to-all | CUDA + NCCL | off (line 178) | out; plain collectives are used |
| `csrc/turbodiffusion/{gemm,norm,quant}` | CUTLASS int8 GEMM, RMSNorm, LayerNorm, quant | generic CUDA | not compiled (R1) | out (Q2): no caller in `fastvideo/`; CUTLASS has no HIP target; the H3 int8 VAE has an `F.linear` dequant fallback |
| `wan22_vae_decode/solution/*.cu` | JIT-built Wan2.2 VAE decoder | sm_100a/103a | gated by capability (10,0)/(10,3) in `fastvideo/models/wan/vae.py:1280` | out (Q2); the fallback must hold (R3) |

The Python/Triton surface that is the real ROCm deliverable (`surface.json` `package:fastvideo_kernel-triton`, `feature:vsa-backends-rocm`, `feature:rocm-platform`):
- `fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py`: VSA 64-token fwd, bwd preprocess, dkdv, dq. Autotunes over `num_stages` 2-7 and `num_warps` 4/8 at BLOCK 64x64, with no HIP-specific config (#1860 adds gfx950 configs). It has an fp8e5 V path (`tl.float8e5`, line 114).
- `block_sparse_attn_256.py` / `block_sparse_attn_varlen.py` / `vsa_utils.py` / `vsa_tile_layout.py` / `fused_compress_topk.py` / `vmoba.py` / `index.py`.
- `st_attn_triton.py` has a HIP branch (`is_cdna3_cdna4()`, which actually lists RDNA archs). Upstream removed STA from the pipeline (#1124), so this is dead code: no test beyond `test_sta.py`, not in scope.
- `fastvideo/platforms/rocm.py` and the detection in `fastvideo/platforms/__init__.py:93-141`.

## Risk list

- **R1: the ROCm compiled extension is probably dead.** See Build classification. On a host with `hip-lang` (every fleet host with `/opt/rocm`), `build.sh --rocm` takes the `enable_language(HIP)` branch. Expected outcome: the configure or the import fails quietly and the Triton fallback hides it. Fix, guarded to ROCm: since no source has a HIP implementation, the ROCm configure skips the extension (`BUILD_CXX_KERNELS OFF` with a status line) instead of building a module that cannot import. The alternative is to keep a module but guard the `register_*` calls; that has no value with nothing to register. Confirm with a build first.
- **R2: the install path pulls CUDA artifacts.**
  - (a) `pyproject.toml` depends on `fastvideo-kernel==0.3.5; sys_platform == 'linux'`. PyPI ships only cp312 manylinux wheels, built for CUDA. On Python 3.12 a ROCm user gets the CUDA wheel. On 3.10/3.11 pip builds the sdist with `GPU_BACKEND` defaulting to CUDA, and that build fails without nvcc.
  - (b) `pyproject_other.toml`, which `docker/rocm7_1.Dockerfile` installs with `.[rocm]`, pins `torch==2.11.0` on a `rocm/pytorch ... pytorch_release_2.9.1` base. uv would likely replace the ROCm torch with a CUDA one from PyPI.
  - (c) The main `pyproject.toml` pins `torch==2.12.0`. Check that a ROCm 2.12.0 wheel exists for the host's ROCm.
  - Validate the documented route (`amd.md` plus `build.sh --rocm`) as a ROCm user would run it. Fix what breaks in the Dockerfile or `pyproject_other.toml`, or document the working order (install ROCm torch, `pip install -e . --no-deps` or a constraints file, build the kernel package from source) in `amd.md`. Prefer a documentation fix over changing upstream's dependency markers. If changing a marker is the only clean fix, keep the change small and say so in the PR.
- **R3: capability collisions** (cuda-to-rocm, platform-limits "AMD compute-capability values COLLIDE"). On ROCm, `torch.cuda.get_device_capability()` returns the gfx generation: gfx90a=(9,0), gfx942=(9,4), gfx950=(9,5), gfx1100=(11,0), gfx1151=(11,5), gfx1201=(12,0), gfx1030=(10,3). Call sites reached on a default path that need checking (the porter confirms each; none was proven wrong by a run):
  - `fastvideo/pipelines/basic/pan2/pan2_pipeline.py:40`: `[0] < 10` lets every RDNA3/4 GPU through to try FA4. It then falls back on `ImportError` of `flash_attn.cute`, so it is safe unless a ROCm flash-attn ships `cute`.
  - `fastvideo/models/encoders/minimax_h3_checkpoint_nvfp4.py:203`: `>= 10` is true on RDNA. Check what it selects.
  - `fastvideo/layers/quantization/mxfp8_config.py:132`, `nvfp4_config.py:517`: `[0] == 10` / `< 10` (gfx1030, gfx101x collide; not a gate platform).
  - `fastvideo/models/wan/vae.py:1280`: (10,3) matches gfx1030, which would try to JIT-build the CUDA decoder; it logs `_kernel_unavailable` and falls back.
  - `fastvideo_kernel/block_sparse_attn.py:26` `_is_sm90()`: (9,0) matches gfx90a, but it also requires the TK ops to exist, so it is safe.
  - `fastvideo_kernel/triton_kernels/attn_qat_train.py:20-36`: guarded by Triton `backend == "cuda"`, so safe.
  - `fastvideo/attention/backends/sla.py:375`: SLA is rejected on ROCm.
  - The fix pattern is upstream's own (#1859): branch on `torch.version.hip` / `gcnArchName`, never on the tuple.
- **R4: Triton on wave32 and Windows.** The VSA autotune space (12 configs, `num_stages` up to 7) may pick or fail differently on RDNA's 64 KB LDS. Triton's autotuner prunes `OutOfResources`, so a fault there should show up as slowness, not as a failure. The fp8e5 V path in `_attn_fwd_sparse` needs OCP fp8 dot support: gfx942 has only FNUZ fp8 and gfx11 has no fp8 dot at all. Find out whether any default path feeds fp8 V. If only an opt-in does, record it as unsupported on those archs rather than fixing it. On Windows Triton comes from `triton-windows` (cuda-to-rocm windows.md), and `fastvideo-kernel` declares `triton` only for `sys_platform == 'linux'`.
- **R5: the kernel tests assume CUDA.** Several call `assert torch.cuda.is_available(), "VSA kernels require CUDA"` (true on ROCm too) or import CUDA-only ops. Tests that target CUDA-arch-only kernels must skip on ROCm, not error. The fix is a skip marker in the test, guarded on HIP.
- **R6: the SSIM harness has no AMD device.** `fastvideo/tests/ssim/inference_similarity_utils.py:23-57` raises `ValueError: Unsupported device for ssim tests` at collection for any device name outside A40/L40S/H100/H200/GB200/B200. SSIM references are NVIDIA-generated per GPU. Do not add AMD reference folders upstream: uploading references to their HF repo is their call. The validator compares through the same helpers against the A40 or L40S reference (T3) and records the SSIM numbers.
- **R7: memory.** Wan2.1-T2V-1.3B with its fp32 UMT5 text encoder needs roughly 20+ GB. gfx1100 has 24 GB, so use CPU offload flags (`--text-encoder-cpu-offload`, `--vae-cpu-offload`, or `dit_cpu_offload`) if needed. The Windows gfx1151 laptop must not run large-memory jobs that starve system RAM (standing rule), so its end-to-end run waits on Q1 and uses the smallest configuration.
- **R8: upstream moves fast.** 23 commits landed between 2026-11-10 and 2026-11-19. Rebase nothing; merge upstream in sync rounds. Cut `moat-port` from the current upstream tip, not from the stale mirror.
- Wavefront, texture, memory-ordering and library-swap fault classes do not apply: there is no HIP C++ on the ROCm path.

## File-by-file change list

The expected changes; each is conditional on the validation finding it needed.

| task | file | change | condition |
|---|---|---|---|
| P1 | `fastvideo-kernel/CMakeLists.txt` | on `GPU_BACKEND ROCM`, do not build `fastvideo_kernel_ops` (no HIP sources); say so in the configure summary | R1 confirmed by a `build.sh --rocm` build with hip-lang present |
| P1 | `fastvideo-kernel/README.md` "Rocm Build" | state that the ROCm build is the Python + Triton package | with P1 |
| P2 | `docker/rocm7_1.Dockerfile`, `pyproject_other.toml` | keep the base image's ROCm torch (matching pin, or install with constraints) | R2(b) confirmed |
| P2 | `docs/getting_started/installation/amd.md` | a validated walkthrough: arch, ROCm/torch versions, install order, kernel build, model, command, memory, output (the page asks for exactly this) | always; the validated commands come from T1-T3 |
| P3 | capability call sites from R3 | ROCm guard where a collision selects a CUDA-only path | only those shown wrong |
| P4 | `fastvideo-kernel/tests/*` | HIP skip markers on CUDA-arch-only tests that error instead of skipping | R5 confirmed per test |
| P5 | Windows (Q1) | none until Jeff rules; #2117 already carries the detection and gloo changes | Q1 |
| P6 | none (read-only) | compare `amd-wangfan/FastVideo`'s commit with #1851 and note anything #1851 lacks | always; record in notes.md |

Explicitly not to change: anything in #1860's lane (HIP autotune configs), routing of the other sparse backends (SLA, BSA, VMoBA, NABLA, Sage; rejected by `rocm.py`), training stacks, and the CUDA kernels.

## Base

Before the first commit, fetch upstream `main` and cut `moat-port` from its tip (`5af9897` or newer). Fast-forwarding the fork's `main` mirror is not the porter's job. Record the base SHA in notes.md.

## Build commands

Linux, any gate host, in a job-owned venv (never the image's Python):

```
python -m venv $VENV && . $VENV/bin/activate
pip install torch==2.12.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm<X.Y>   # match the host ROCm; record the exact version
pip install -e . --no-deps && pip install <fastvideo deps minus torch and fastvideo-kernel>          # or the order P2 settles
pip uninstall -y fastvideo-kernel                                                                     # if the PyPI CUDA wheel came in
cd fastvideo-kernel && ./build.sh --rocm                         # with hip-lang present: exercises R1
cd fastvideo-kernel && ./build.sh --rocm --python-only           # the path upstream documents for pip-SDK images
python -c "import fastvideo_kernel, fastvideo; from fastvideo.platforms import current_platform as p; print(p.__class__.__name__)"
```

Wrap build and test steps in `utils/timeit.sh FastVideo <phase> -- ...`. Windows commands wait on Q1.

CUDA no-regression: every change is Python or CMake inside a ROCm guard. The nvcc gate (mi355 host) is a CUDA configure of `fastvideo-kernel` (`TORCH_CUDA_ARCH_LIST=9.0a ./build.sh` or a configure-only `cmake -DGPU_BACKEND=CUDA`) when P1 touches CMake. Where the host's ROCm torch blocks it, record `cuda-not-validated: <error>`. Python-only changes need a CPU-platform import and unit-test pass instead.

## Test plan

- **T1 (kernel suite, every gate):** `cd fastvideo-kernel && pytest tests -v -rs`. Must pass: `test_vsa.py`, `test_vsa_forward.py`, `test_vsa_triton_backward_scale.py`, `test_vsa128_backward.py`, `test_vsa256_*.py`, `test_vsa_varlen.py`, `test_vsa_combine.py`, `test_vsa_tile_layout.py`, `test_vsa_utils.py`, `test_fused_compress_topk.py`, `test_vmoba_correctness.py`, `test_device_specs.py`, `test_build_arch.py`. Must skip cleanly, not error: `test_attn_qat_*`, `test_block_sparse_*blackwell*`, `test_turbodiffusion.py`, `test_wan22_vae_decode.py`, `test_nvfp4_benchmark_imports.py`, `test_sta.py` (STA is removed from the pipeline; record its result without fixing). Also run top-level `tests/test_bsa.py` and `tests/test_block_sparse_blackwell.py` and record their result. Record pass/skip/fail counts and confirm the Triton kernels actually dispatch (cuda-to-rocm validation.md, "A green result does not prove a GPU test ran").
- **T2 (package unit tests, every gate):** `pytest fastvideo/tests/platforms fastvideo/tests/attention fastvideo/tests/ops fastvideo/tests/distributed -v -rs`, then the CPU-safe remainder of `fastvideo/tests` the porter can run in budget. Required: `test_rocm_detection_without_amdsmi.py`, `test_rocm_vsa_backend_routing.py`, `test_fp8_compute_gate.py`. Compare against the same subset on the CPU platform (`HIP_VISIBLE_DEVICES=` empty) so ROCm-only failures are separated from pre-existing ones.
- **T3 (end to end, every gate, one GPU):** text-to-video with `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` at the SSIM test's parameters (480x832, 45 frames, 4 steps, seed 1024) with `num_gpus=1, sp_size=1`. Run it under `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA`, under `FLASH_ATTN` if a ROCm flash-attn is installed, and under `VIDEO_SPARSE_ATTN` with a VSA checkpoint (`FastVideo/FastWan2.1-T2V-1.3B-Diffusers`, sparsity per its model card). Compare SDPA output against the A40/L40S reference video through `fastvideo/tests/ssim` helpers, and VSA against SDPA on the same GPU. Record SSIM, wall time and peak memory, and save the videos. A cross-vendor SSIM below the test's threshold is evidence to classify (numerics, not automatically a fault), never something to hide.
- **T4 (multi-GPU, wave64 only, when a host with 2 free GPUs exists):** the SSIM test's own `num_gpus=2, sp_size=2` on SDPA, which exercises RCCL through `CudaCommunicator` on ROCm. Optional evidence, not a gate.
- **T5 (benchmarks, once per gate):** `fastvideo-kernel/benchmarks/bench_vsa.py` and `bench_vsa_combine.py`, numbers only.
- **Regression set that must not move:** `fastvideo/tests/platforms/test_cpu_sdpa.py`, `test_mps_vsa_error.py` and the CPU-platform run of T2.

## Open questions (Jeff must rule before a porter starts)

**Q1 -- Windows gate.** Upstream's own docs say NVIDIA runs on "Linux or Windows WSL" (`docs/getting_started/installation/gpu.md:7`). The README claims "Linux, Windows, MacOS". Native Windows is not supported for anyone today. ROCm on native Windows fails at detection (no `/dev/kfd`, no amdsmi, so it becomes `CpuPlatform`) and at distributed init (NCCL forced). The kernel package and Triton are linux-only dependency markers. Third-party PR #2117 (klin2024, opened 2026-11-19, tested on a Radeon 8060S, i.e. gfx1151) fixes detection and the gloo fallback. Options:
- (a) Wait for #2117 to land, then validate native Windows on gfx1151 on top of it. MOAT's own Windows work is then limited to whatever is still missing (the `triton-windows` and kernel-package install route, docs). Windows goes in `owed` until then.
- (b) Carry #2117's changes on `moat-port` now so Windows can be validated immediately. This duplicates an open third-party PR, the gsplat failure mode.
- (c) Validate under WSL instead of native, matching upstream's documented NVIDIA Windows support. Needs ROCm-on-WSL on the laptop.
- (d) Ask for a windows gate waiver: FastVideo does not support native Windows for CUDA either.
- Recommendation: (a). Linux gates go first in any case.

**Q2 -- confirm the scope cut of CUDA-arch-only kernels.** The plan validates and fixes the ROCm paths upstream already chose, and covers no HIP translation of: ThunderKittens Hopper VSA/STA, Blackwell PTX VSA fwd/bwd, CUTLASS/CuTe FP4 attention and quantization (sm_120a/121a), the H3 SwiGLU NVFP4 GEMM (sm_121a), the NCCL-device Ulysses all-to-all, the CUTLASS turbodiffusion int8 GEMM/norm/quant (no caller in `fastvideo/`), and the Wan2.2 JIT VAE decoder (sm_100a/103a). Each has a working fallback on ROCm (Triton VSA, SDPA/FA, plain collectives, `F.linear` dequant, `AutoencoderKLWan`). Options:
- (a) Confirm they are out. They are recorded as `scoped_out` with that reason, and the port is a validation-plus-fixes PR. Recommended; it matches the 2026-11-10 ruling.
- (b) Name any to bring in as an AMD-native rewrite (Composable Kernel / FlyDSL). That would be a separate, large project per kernel, and would compete with #1860's Triton direction for VSA.

**Q3 (informational, no ruling needed unless Jeff wants otherwise).** The other sparse backends (SLA, BSA, VMoBA, NABLA, Sage) stay rejected on ROCm. Routing them is new feature work in the active contributor's lane, so it is not in this plan.
