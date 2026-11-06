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
