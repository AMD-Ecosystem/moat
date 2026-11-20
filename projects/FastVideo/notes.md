# FastVideo notes

## Intake screen (2026-11-05, gap-matrix seed)

Upstream: https://github.com/hao-ai-lab/FastVideo, screened at a shallow clone of `main` taken 2026-11-05 (pushed 2026-11-05, not archived, ~4.8k stars). Lead from the third-party gap matrix: "FastVideo (sliding-tile and video sparse attention)", verdict partial, AMD path upstream.

### Licence

Apache-2.0 at the top level (`LICENSE`) and in `fastvideo-kernel/LICENSE`; recorded with `set-license-spdx` (tier 1). `utils/licenses.py scan-nvidia agent_space/FastVideo-screen` (submodules included) reports 109 hits, all inside the vendored `fastvideo-kernel/include/cutlass` submodule (NVIDIA/cutlass: `EULA.txt`, `LICENSE.txt`, and the CuTeDSL Python tree plus its examples). Zero hits in FastVideo's own source. FastVideo's kernels use only CUTLASS/CuTe C++ headers, not the CuTeDSL tree. That would only matter for a port, and the recommendation below is not to port.

### Existing AMD support (upstream, maintained)

- `docs/getting_started/installation/amd.md` and `docs/getting_started/hardware.md` document a ROCm platform. `fastvideo/platforms/rocm.py` came from AMD (PR #669, author `sopiko99`, @AMD, merged 2025-07-30). It routes to SDPA, or to FlashAttention when it is installed and compatible, and falls back to SDPA otherwise.
- The video sparse attention backends (`VIDEO_SPARSE_ATTN`, `VIDEO_SPARSE_ATTN_H3`) route to Triton kernels on ROCm (PR #1851, merged 2026-09-15). The kernel package installs as Python + Triton when the HIP toolchain is unavailable (#1871, #2064). Platform detection works from a HIP torch build without amdsmi (#1849). FP8 `_scaled_mm` is gated on CDNA4 gfx950 (#1859). Fused FP8 activation quantization in Triton (#1861).
- Sliding-tile attention: a Triton kernel with ROCm support (PR #916), fixed for RDNA (PR #969). STA was later removed from the pipeline (PR #1124, "Remove STA", 2026-02-23); the `st_attn_h100.cu` / `st_attn_triton.py` kernels remain in the kernel package, but there is no STA attention backend left to port.
- There is a ROCm Dockerfile (`docker/rocm7_1.Dockerfile`, PR #952) and the ROCm build is documented (`fastvideo-kernel/README.md` "Rocm Build", `build.sh --rocm`).
- Active ongoing AMD work: open PR #1860 (stayinalive181, updated 2026-11-05) adds HIP-only Triton autotune configs for VSA on gfx950 (1.22-1.55x on MI355X). The same contributor landed #1849/#1851/#1859/#1861/#1871 in Sep 2026.
- Fork `amd-wangfan/FastVideo` ("add vsa triton opt on gfx942", 2026-02-27), 7 ahead / 487 behind, last merge from upstream 2026-05-14. A personal fork from an AMD-named account, now stale.
- No AMD runner in upstream CI (`.github/workflows`: CUDA and macOS only). ROCm/ and AMD-Ecosystem have no FastVideo repository. The `amd` / `AMD-AGI` orgs were not searchable with this token (SAML SSO), so an internal AMD effort there is not ruled out.

### CUDA surface (fastvideo-kernel, ~610 KB own source)

Every compiled kernel is an NVIDIA-architecture-specific acceleration, and each has a working fallback on ROCm:
- `csrc/attention/st_attn_h100.cu`, `block_sparse_h100.cu`: ThunderKittens, `-DKITTENS_HOPPER`, sm_90a only (wgmma/TMA). Triton fallbacks ship in `python/fastvideo_kernel/triton_kernels` and are what ROCm uses.
- `csrc/attention/block_sparse_*_blackwell.cu[h]`: sm_100a/sm_103a tcgen05/setmaxnreg PTX VSA fwd/bwd. Blackwell only; the Python `is_supported()` check falls back to Triton.
- `attn_qat_infer/blackwell/*`, `quantization/fp4_quantization_4d.cu`: CUTLASS/CuTe FP4 (sm_120a/121a). CMake disables it explicitly on ROCm.
- `csrc/comm/ulysses_all_to_all.cu`: NCCL 2.29 device API, optional, disabled on ROCm (plain collectives are used instead).
- `csrc/turbodiffusion/{gemm,norm,quant}`: CUTLASS int8 GEMM/RMSNorm/LayerNorm/quant. These are the only generic-arch kernels, but nothing in `fastvideo/` calls them (they are only re-exported from `fastvideo_kernel/__init__.py`). On a ROCm build CMake enables HIP and not CUDA, and the CUDA flags are scoped to `COMPILE_LANGUAGE:CUDA`, so these `.cu` sources are not built for HIP. Not verified by a build.
- A HipKittens path (the ThunderKittens authors' AMD port, dispositioned in MOAT as ported-elsewhere) is plausible in principle for the TK VSA kernel, but it would be an AMD-native rewrite of an optional fast path. Upstream has chosen the Triton route on ROCm and is tuning it for MI355X, so a HipKittens kernel would compete with that direction rather than fill a gap.

### Residual gaps (small, not a port)

- `rocm.py` rejects the other Triton/Python sparse backends (SLA, BSA, VMoBA, NABLA, Sage) as "Invalid attention backend". Routing some of them could be a small Python change, but it falls in the active contributor's lane.
- `amd.md` says that a maintained ROCm walkthrough (arch, versions, model, command, memory) is still missing. RDNA wave32 and Windows are not covered at all upstream. That is a possible validate-and-document differentiator if a person wants one, but there is no CUDA porting work in it.

### Recommendation

Decline, `already-supported`. Upstream maintains a ROCm platform with SDPA/FlashAttention backends and Triton routes for its video sparse attention, and an external contributor is actively landing AMD work, including MI355X tuning in open PR #1860. The CUDA-only kernels are Hopper TK, Blackwell PTX/FP4 and NCCL-device accelerations, and each has a working fallback on ROCm. The matrix's "sliding-tile attention" lead is moot: STA has been removed from the pipeline (#1124).

Question for jeff (only if a different call is wanted): should MOAT offer a ROCm validation walkthrough for consumer RDNA (wave32) and Windows, plus routing for the SLA/BSA Triton backends on ROCm, as a docs/packaging contribution coordinated with the active contributor? The screen recommends not to, by default.

## Ruling: validation project (Jeff Daily, 2026-11-10)

Jeff Daily overrode the intake decline: FastVideo becomes a validation project. Validate upstream's ROCm platform support (#669) and the Triton video sparse attention path (#1851). Validate on every required gate (wave64, wave32, Windows) and fix what breaks; the fixes go on `moat-port` of a new AMD-Ecosystem fork. A person creates that fork, never an agent; the project waits in `awaiting-fork` until it exists, then `release-forks` moves it to `screened` for the planner.
Jeff asked whether to transfer amd-wangfan/FastVideo instead. It holds one commit ("add vsa triton opt on gfx942", 2026-02-27) plus upstream merges, is 495 commits behind, and no upstream PR came from it; #1851 likely supersedes the commit. A new fork from current upstream is preferred; the commit can be compared against #1851 during validation.

## Planning (2026-11-20, linux-gfx1100, planner)

**Decisions needed from Jeff before a porter starts** (full text in plan.md, "Open questions"):

- **Q1, Windows gate.** Native Windows on ROCm fails today at platform detection (no `/dev/kfd`, no amdsmi, so it becomes `CpuPlatform`) and at distributed init (NCCL is forced). The kernel package and Triton are declared linux-only. Upstream supports NVIDIA only on "Linux or Windows WSL". Third-party PR https://github.com/hao-ai-lab/FastVideo/pull/2117 (klin2024, opened 2026-11-19, open, tested on a Radeon 8060S) already fixes detection and the gloo fallback. Options: (a) wait for #2117, then validate native Windows on gfx1151 on top of it, with Windows `owed` meanwhile (recommended); (b) carry #2117's changes on `moat-port` now, duplicating an open PR; (c) validate under WSL instead; (d) ask for a windows waiver, since native Windows is unsupported for CUDA too.
- **Q2, scope cut.** The plan validates and fixes only the ROCm paths upstream chose. It translates none of the CUDA-arch-only kernels: TK Hopper, Blackwell PTX VSA, CUTLASS/CuTe FP4 and the H3 SwiGLU GEMM (sm_120a/121a), NCCL-device Ulysses, CUTLASS turbodiffusion int8 (no caller), and the Wan2.2 JIT VAE decoder (sm_100a/103a). Each has a working ROCm fallback. Options: (a) confirm they are out and record them `scoped_out` (recommended; matches the 2026-11-10 ruling); (b) name any to bring in as an AMD-native rewrite, which would be a separate large project each. `surface.json` marks them "proposed out, pending jeff". Nothing is recorded as `scoped_out` yet.

What was done:
- Took the planning lock on linux-gfx1100. Cloned the fork's `main` (`4deb3a50fff8`) read-only into `projects/FastVideo/src`. Set `ext_type` to `torch-extension`. Generated and extended `surface.json` (52 CUDA files, 3 CMake libraries plus 11 added components) and wrote `plan.md`.
- Upstream `main` was at `5af9897` (2026-11-19), 23 commits past the mirror. The only build delta is a new sm_121a CUTLASS library (`fastvideo_h3_swiglu_gemm`), off on ROCm. The porter cuts `moat-port` from the current upstream tip.

Findings for the porter (detail and line numbers in plan.md):
- R1: with hip-lang present, `build.sh --rocm` enables HIP but compiles only `common_extension.cpp`. The turbodiffusion `.cu` files are CUDA-language with CUDA not enabled, and `register_quant` and friends are called unconditionally, so the module probably fails to import and the Triton fallback hides it. Unverified; build it first.
- R2: install path. PyPI `fastvideo-kernel==0.3.5` has only cp312 manylinux wheels, built for CUDA, and the sdist defaults to `GPU_BACKEND=CUDA`. The ROCm Dockerfile installs `pyproject_other.toml`, which pins `torch==2.11.0` over a ROCm 2.9.1 base image.
- R3: capability tuples collide (gfx90a=9.0, gfx1201=12.0, gfx11xx>=10). The call sites to check are listed in plan.md, and none was shown wrong by a run. Upstream's own fix pattern is #1859.
- R6: the SSIM harness raises `Unsupported device` for any non-NVIDIA device name. The validator compares against the A40/L40S reference through the same helpers and does not add AMD references upstream.
- Active third-party AMD work not to duplicate: #1860 (gfx950 VSA autotune, open) and #2117 (Windows, open).
