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

## Planner (2026-11-20, planner agent, linux-gfx90a fleet job, for jeff)

### Decisions needed from jeff before a porter starts

The project was adopted (AMD-Ecosystem/nunchaku exists), which answers intake's "take on a rewrite of this size" question. Three questions remain. The plan is written for the recommended answer of each; a porter must not start until jeff has ruled.

**Q1. Is the NVFP4 path in scope, or INT4 only?** Upstream has two 4-bit formats: INT4 (`svdq-int4_*` checkpoints, every pre-Blackwell NVIDIA GPU) and NVFP4 (`svdq-fp4_*`, sm_120a/121a only, `mma ... kind::mxf4nvf4.block_scale ... e2m1 ... ue4m3`). Options:
- (a) INT4 only (recommended). FP4 kernels trap on ROCm exactly as they do upstream on any GPU below sm_120; `get_precision()` returns `int4` on every AMD GPU, so every model and test runs with the INT4 checkpoints. The FP4 component is then `scoped_out` with this ruling as its reason. Follow-up recorded as a deferral.
- (b) INT4 plus a software NVFP4 path on every gate (decode e2m1 values and ue4m3 block-16 scales to f16 in registers, then the f16 primitive). Lets AMD users load the FP4 checkpoints; slower than INT4; roughly +20% of the kernel work and a second set of LPIPS runs.
- (c) INT4 plus native FP4 only where hardware has it (gfx950 scaled MFMA f8f6f4, gfx1250 WMMA f4). The NVFP4 scale format (ue4m3, block 16) does not match MX (e8m0, block 32), so scales still need software handling; covers one or two arches only.

**Q2. The three unreachable components: port or leave out?** None of these is reachable from any Python entry point or model:
- `src/kernels/gemm_w8a8.cu` `gemm_w8a8_fp16` (CUTLASS): no caller anywhere.
- `src/kernels/gemm_batched.cu` and Block-Sparse-Attention `mha_fwd_block`: reached only through `Attention::forward(qkv, pool_qkv, sparsityRatio)` with `sparsityRatio > 0`, and every caller passes the literal `0.0f` (`FluxModel.cpp:1289,1392`).
Options:
- (a) Recommended: leave `gemm_w8a8_fp16` out of the ROCm source list; give `gemm_batched_fp16` a hipBLAS strided-batched implementation (small, keeps the sparse path compiling and callable); make `mha_fwd_block` throw a clear "not supported on ROCm" error. All three recorded in `surface.json` with this ruling.
- (b) Port all three (block-sparse attention would need a CK/FlyDSL rewrite of a flash-attention-2 kernel that nothing calls: large cost, no user value today).
- (c) Leave all three out (smallest; the sparse path then fails at link or load time instead of with a message, which is worse).

**Q3. Is a correctness-first first PR acceptable when it may be well below NVIDIA speed?** The plan keeps NVIDIA's `mma.sync` fragment layout (the published checkpoints and the Python LoRA/rotary packers depend on it) and reimplements the handful of PTX helpers on AMD: stage 1 is a lane-gather reference that is exact on every arch; stage 2 maps the INT4 path to RDNA WMMA iu4 and to CDNA i8 MFMA (int4 unpacked to int8), each proven bit-identical to stage 1. Options:
- (a) Recommended: the port is complete when stage 1 passes everywhere and stage 2 exists wherever stage 1 is too slow to run the end-to-end tests in practical time; speed is recorded, not gated, and an AMD-native tiling pass (CK/FlyDSL, AMD-chosen tiles with a load-time repack) is a separate later project.
- (b) Require stage 2 matrix-core paths on every gate before review, regardless of stage-1 speed.
- (c) Skip the layout-preserving approach and go straight to an AMD-native layout with a checkpoint repacker and a second Python packer (much larger, forks the model format).

### What the planner found (summary; plan.md has the detail)

- Build: torch extension (`setup.py` `CUDAExtension` `nunchaku._C`), Strategy B. `ext_type` recorded as `torch-extension`.
- The NVIDIA fragment layout is the project's data contract: published checkpoints are pre-packed for it (`gemm_base.cuh` "weights need to be repacked when the tiling size changes"), and Python packs LoRA weights (`nunchaku/lora/flux/packer.py`) and rotary embeddings (`nunchaku/models/embeddings.py:130`, `transformers/transformer_flux.py:84`, `caching/utils.py:311`) in it at run time. This is why the plan keeps it and reimplements the primitives instead of rewriting kernels.
- All tensor-core PTX goes through a few helpers (`zgemm/mma.cuh`, `mma_earlycuda.cuh`, `gemm_utils.cuh`, `GEMM_W4A4::mma`, `GEMM_W8A8::mma`, plus AWQ's own `ldmatrix`/`mma` in `awq/gemm_awq.cu`); the zgemm framework above them is plain CUDA C++ on a logical 32-lane warp (`WARP_SIZE = 32` constant, 256-thread blocks).
- Wavefront hazards: default-width `__shfl_sync` in `zgemm/epilogues.cuh` (~99, 125, 137) and `awq/gemv_awq.cu` (55-57) would read across the two logical warps of a wave64 wavefront; 32-bit masks in `reduction_utils.cuh` do not compile under HIP's 64-bit-mask `*_sync`.
- `gemm_base.cuh:893` `invoke_kernel` compiles every kernel body on HIP (no `__CUDA_ARCH__`), including the FP4 kernels, so FP4 needs an explicit ROCm guard whatever Q1 says.
- Dense attention for the C++ Flux model (`mha_fwd`, Block-Sparse-Attention/flash-attn-2) maps to torch SDPA on ROCm through the existing torch interop; the Python v1 models already use SDPA.
- CUTLASS reached by live code: `gemm_f16` (Sana `kv_linear`, LoRA on AWQ norm linears) -> hipBLAS(Lt); `dwconv_f16` (Sana) -> small HIP kernel.
- Python: `nunchaku/utils.py` precision/arch checks key on `get_device_capability()`, which collides on ROCm (gfx90a reports (9,0)); needs a `torch.version.hip` arm.
- No kernel-level tests upstream; every test is an end-to-end diffusion run judged by LPIPS against a 16-bit reference (network, HF downloads, FLUX.1-dev gated, about 24 GB GPU memory). The plan adds `tests/ops/` golden tests (exact for integer paths) as the first gate. Windows validation uses the small models (SDXL, SANA) only.
- `surface.json`: 40 CUDA files in the census plus 35 hand-added components (build, primitives, kernels, CUTLASS/flash-attention replacements, Python arch detection, every test directory, examples, app). Nothing covered yet; the FP4 and Q2 components are left undispositioned until jeff rules.
- llm-awq (MOAT, same lab) ported the same `dequantize.cuh` lop3 decoder and logical-32 shuffle fixes; a reference for the AWQ part, not a dependency.

Analysed at upstream 302e0e97024ebd68688fe890e5df83731edf7b54 (with submodules). No GPU used, nothing built.

## Rulings on the planner's questions (Jeff Daily, 2026-11-23)

- Q1 (NVFP4): **c**. INT4 everywhere, plus native FP4 only where the hardware has it (gfx950 scaled MFMA, gfx1250 WMMA f4), with the NVFP4 scale format (ue4m3, block 16) handled in software there. No software FP4 path on other targets; there FP4 traps as it does upstream below sm_120.
- Q2 (the three unreachable components): **a**. `gemm_w8a8_fp16` left out of the ROCm source list; `gemm_batched_fp16` gets a hipBLAS strided-batched implementation; `mha_fwd_block` raises a clear "not supported on ROCm" error.
- Q3 (speed): **b**. Stage 2 matrix-instruction paths (RDNA WMMA, CDNA MFMA) are required on every gate before review, whatever stage 1's speed. Stage 1 stays as the exact reference each stage 2 path is proven bit-identical against.
- Standing ruling of the same day: Windows laptop GPU runs go last across all ports; they are queued, not started.
