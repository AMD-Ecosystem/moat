# nunchaku notes

## Intake screen (2026-11-05, intake agent, windows-gfx1151 host, for jeff)

Seed: third-party "AMD AI Software Gap Matrix" (verdict not-yet, AMD path: none). Treated as a lead; findings below are this screen's own.

Canonical repo: `nunchux-ai/nunchaku` (formerly `mit-han-lab/nunchaku`; the API returns `nunchux-ai/nunchaku` as `full_name`). 3965 stars, not archived, not a fork. Screened at upstream `main` 302e0e97024ebd68688fe890e5df83731edf7b54 (2026-09-06). Activity has slowed: last commits 2026-09-06 (docs), 2026-03-05, 2026-02-13.

### Licence

- `python3 utils/licenses.py check nunchux-ai/nunchaku` -> Apache-2.0, tier 1, cleared to contribute. `LICENCE.txt` read: Apache License 2.0 text.
- `python3 utils/licenses.py scan-nvidia agent_space/nunchaku-screen` (shallow clone with `--recurse-submodules --shallow-submodules`) -> no NVIDIA proprietary licence text, submodules included.
- Submodules (all read): `third_party/cutlass` BSD-3-Clause (NVIDIA, permissive); `third_party/Block-Sparse-Attention` BSD-3-Clause (MIT HAN Lab, `nunchaku` branch); `third_party/json` MIT; `third_party/mio` MIT; `third_party/spdlog` MIT. No unlicensed vendored component.
- Recorded: `license_spdx` = Apache-2.0.

### Duplicate effort

- No upstream PR or branch: `gh pr list --state all --search "hip OR rocm OR amd"` -> none; no branch matching hip/rocm/amd.
- Upstream README/docs grep for amd/rocm/hip/gfx: no AMD mention.
- Upstream issue #75 "Support for RDNA3 and other AMD archs" (closed): collaborator lmxyy: "Supporting AMD would require rewriting all the kernels. We do not plan to support this for now." Not a refusal of contributions, but no maintainer interest either, so an upstream PR is uncertain.
- GitHub search (nunchaku rocm / nunchaku amd / svdquant; ROCm org): nothing. `ultism/svdquant-kernels` targets NVIDIA Blackwell and Huawei Ascend, not AMD. AMD-AGI org search blocked by SAML (not checked).
- Web: one anecdotal Japanese blog post (gadgetzamurai.jp) about running Nunchaku in ComfyUI on an RX 7900 XT; it says the 4-bit FP4/INT4 paths may be unsupported or unstable on AMD. No code and not authoritative. Not a port.
- Related MOAT project: `llm-awq` (same lab; adopted, fork AMD-Ecosystem/llm-awq). nunchaku vendors its own copy of AWQ kernels in `src/kernels/awq/` (gemm_awq.cu, gemv_awq.cu, dequantize.cuh), so the llm-awq port's fixes are a direct reference for that part. This is not a build dependency: no `set-deps`.
- Verdict: no existing AMD effort.

### Viability (cheap checks; the deep analysis is the planner's)

- Genuine CUDA torch extension (`setup.py`, CUDAExtension, nvcc gencode for sm_75/80/86/89/120a/121a only). 18 `.cu` files plus headers under `src/`, about 470 KB. Block-Sparse-Attention (a flash-attention-2 derived CuTe/CUTLASS fork, 8 `flash_fwd_*_sm80.cu` instantiations) is always compiled in non-debug builds.
- Inline PTX is the core, not an edge: the hot W4A4 GEMM (`src/kernels/zgemm/gemm_w4a4.cuh`, `mma.cuh`, `mma_earlycuda.cuh`, `gemm_utils.cuh`) is hand-written `mma.sync.aligned` (INT4 `s4.s4`/`u4.s4` m8n8k32 and m16n8k64; fp16/bf16 m16n8k16; s8 m16n8k32; NVFP4 `kind::mxf4nvf4.block_scale.scale_vec::4X ... e2m1 ... ue4m3` m16n8k64 for sm_120a), `ldmatrix`, `cp.async`, `cvt.pack.sat.s4/u4`, `cvt.rn.satfinite.e2m1x2`. About 95 `asm` sites across 10 files. Everything assumes NVIDIA 32-lane mma fragment layouts. hipify translates none of it.
- CUTLASS 2.x device GEMMs in `gemm_f16.cu`, `gemm_w8a8.cu`, `gemm_batched.cu`, `dwconv.cu` (via `dispatch_cutlass.h`). These need hipBLASLt / Composable Kernel / rocWMMA replacements.
- Matrix-core mapping on AMD (see `references/no-hip-equivalent.md`):
  - INT4 x INT4 (the main SVDQuant path): native on RDNA WMMA (gfx11 `v_wmma_i32_16x16x16_iu4`, gfx12 `iu4` variants), so wave32 gates have a direct target. CDNA gfx90a/gfx942/gfx950 MFMA have no int4 input; the wave64 route is to unpack s4 to s8 in registers and use the i8 MFMA (functional, memory stays 4-bit).
  - NVFP4 (e2m1 with ue4m3 block-16 scales): gfx950 scaled MFMA (f8f6f4, MX e8m0 scales, block 32) and gfx1250 WMMA f4 support FP4 operands, but the NVFP4 scale format and block size do not match MX directly; scale handling would need software. FP4 is the lower-priority path (Blackwell-only upstream too).
  - fp16/bf16/int8 MMA: native MFMA/WMMA on every gate.
- Shape of the work: an AMD-native rewrite of the GEMM/attention kernels (the maintainer's own estimate is "rewriting all the kernels"), plus the Python/packaging side (setup.py is nvcc-only). Not a mechanical hipify port. Large and performance-sensitive, but feasible on every required gate.
- Tests: `tests/flux`, `tests/sana`, `tests/v1/{flux,qwenimage,sdxl,z_image}` are end-to-end diffusion-model runs (download quantized FLUX/SANA/Qwen-Image checkpoints, compare to reference images by LPIPS). They need a large-VRAM GPU and network. `src/kernels/zgemm/gemm_w4a4_test.cu` exists as a kernel test entry. The gfx1151 laptop is likely too light for FLUX-scale runs (memory note: light ports only).

### Recommendation

Fork, with viable=yes but flagged heavy. Demand exists (issue #75, ComfyUI-nunchaku has about 2.9k stars, the gap matrix lists AMD path none), the licence is clean, and there is no other AMD effort. The case against: it is a kernel rewrite rather than a port, and the maintainers stated no plan to support AMD, so the realistic deliverable may be the AMD-Ecosystem fork itself rather than a merged upstream PR. If declined, the SKIP_REASON would be `cant-port` (cost: every hot kernel is inline NVIDIA PTX with no HIP equivalent), not `already-supported`.

### Question for jeff

Nunchaku's hot W4A4 kernels are hand-written NVIDIA mma/ldmatrix PTX (INT4 and NVFP4), so this is an AMD-native kernel rewrite (RDNA WMMA iu4 on wave32; int4 unpacked onto i8 MFMA on CDNA), not a hipify port, and the maintainers said in issue #75 they do not plan AMD support. Do you want MOAT to take on a rewrite of this size, and if so, is the fork-as-deliverable outcome acceptable if upstream will not merge it? Also: should the NVFP4 (sm_120a) path be in scope, or INT4 only?
