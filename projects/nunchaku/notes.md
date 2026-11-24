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

## Rulings on the porter's questions (Jeff Daily, 2026-11-24)

Answers to the three questions of the gfx942 porter round below (relayed in the gfx950 porter dispatch of 2026-11-24).

- Q1 (FP4 scaling): **(b) for now**: decode e2m1 x ue4m3 to f16 in registers (exact) and run the f16 matrix instructions. On gfx950 use (b), because its scaled MFMA takes only power-of-two scales per 32 elements. jeff noted that gfx1250 also has a 16-element block scale: upstream LLVM has `v_wmma_scale16_f32_16x16x128_f8f6f4` and `v_wmma_scale16_f32_32x16x128_f4` (64-bit scale operands, "wmma-n16-insts") with scale formats E8, E5M3 and E4M3 (`MATRIX_SCALE_FMT_E4M3`). If the scale16 form gives one E4M3 scale per 16 elements of K, that is NVFP4's own block and scale format and gfx1250 can run NVFP4 natively. So on gfx1250: first establish from the LLVM sources and MC tests (and the ISA documentation if available) the exact scale16 operand layout (scales per lane, the K range each covers, and that E4M3 is the unsigned ue4m3 NVFP4 uses); if it matches NVFP4, write the gfx1250 path on the native scale16 WMMA, otherwise use (b) there too; record what was found either way.
- Q2: **(b)**: add the f16/bf16 MFMA path for the m16n8k16 primitive on CDNA (gfx90a, gfx942, gfx950), with the same design as the RDNA WMMA path (paired n16 products, transposed operands, C kept outside the matrix instruction and added in fp32). Bit identity with stage 1 is not expected; use the RDNA round's tolerance and its justification, and show determinism run to run.
- Q3: keep `get_precision("auto")` at `int4` on gfx950 and gfx1250 until an FP4 run on those cards shows equal or better LPIPS and speed than INT4; then switch. Record the measurement that can be made on gfx950.
- Order of work for the gfx950 round: CDNA f16/bf16 MFMA first, then FP4 on gfx950 via (b) (remove the trap only on targets with a working path), then the gfx1250 path compiled and checked in the code objects (unproven until a gfx1250 run). gfx11/gfx12 behaviour unchanged, CUDA path untouched, new commits on top of `moat-port`; stage stays `porting` while a required proof is missing.

## Porter round 2026-11-24 (porter, linux-gfx942 fleet job, for jeff)

### Questions for jeff (nothing below waits on them except where stated)

1. **FP4 on gfx950/gfx1250 (Q1 = c): how should "scales in software" be done?** The NVFP4 format scales every 16 elements with a ue4m3 value; the gfx950 scaled MFMA (`v_mfma_scale_f32_*_f8f6f4`) and gfx1250 WMMA scale every 32 elements with a power of two (e8m0). A ue4m3 scale is not a power of two and its block is half the hardware block, so it cannot be folded into the hardware scale. Options: (a) FP4 matrix instructions with unit hardware scales on K slices that hold one 16-element block each, ue4m3 products applied to the fp32 partial sums in software (the other slices zero: 2x to 4x the instruction count of a native NVFP4 MMA); (b) decode e2m1 x ue4m3 to f16 in registers (exact: at most 5 significant bits) and use the f16 MFMA/WMMA, which is native matrix hardware but not the FP4 instructions; (c) a person's other design. Evidence: `gemm_w4a4.cuh` `mma_fp4` and `quantize_w4a4_fp4_from_fpsum_warp` (ue4m3 per 16, `scale_vec::4X`), packer `pack_micro_scale`. Nothing for FP4 was written in this round; the FP4 kernels trap on every AMD target, as on NVIDIA below sm_120. Waits on this answer: surface `kernel:gemm_w4a4-nvfp4-gfx950`, `kernel:gemm_w4a4-nvfp4-gfx1250`.
2. **Does Q3 = b ("stage 2 on every gate before review") include the f16/bf16 m16n8k16 primitive on CDNA?** The plan's stage 2 list names INT4 for CDNA (done here) and INT4 plus f16/bf16 WMMA for RDNA. The f16/bf16 primitive feeds the LoRA up/down epilogues, the nunchaku-fp16 attention, the LiteLA epilogue and the AWQ GEMM; it runs on the stage 1 lane-gather everywhere today. Options: (a) INT4 only on CDNA, f16 stays stage 1 there (current state); (b) add f16/bf16 MFMA on CDNA too (the same transposed-operand and shared-memory staging design applies; f16 results are float, so "bit-identical to stage 1" would become a tolerance unless the f16 MFMA reproduces stage 1's summation order, which it does not). Surface `kernel:stage2-rdna-wmma-f16` is recorded owed for RDNA either way; nothing is recorded for CDNA f16 until this is answered.
3. **`get_precision("auto")` on gfx950/gfx1250 once FP4 works there:** keep `int4` (current code) or pick `fp4` as upstream does on Blackwell? (plan.md, Python section.)

### What was done

Fork: https://github.com/AMD-Ecosystem/nunchaku branch `moat-port` (cut from upstream 302e0e97). Commits, oldest first:

- `a4abb15` Add the template keyword to dependent member template names (shared C++ fix: clang rejects `typename GEMM::EpilogueBias<...>` without `template`; nvcc accepts it).
- `6e23cc9` Initialize std::array with braces in mma_f16xf16_f32 (shared C++ fix: `std::array<T,2>(a, b)` is not valid C++).
- `9875a78` [ROCm] Build the extension for AMD GPUs with HIP (setup.py ROCm branch; `zgemm/mma_rocm.cuh` stage 1 primitives; CDNA MFMA INT4 main loop; HIP arms of every PTX helper; 32-lane shuffle wrappers; hipBLAS `gemm_f16` and `gemm_batched_fp16`; direct depthwise kernel; `src/interop/flash_api_rocm.{h,cpp}`; Python arch detection).
- `67db6fa` [ROCm] Run the INT4 GEMM main loop on RDNA WMMA (**unproven**: compiled for gfx1100/gfx1201 only).
- `8eafc6d` Add kernel-level tests for the W4A4 GEMM, the AWQ kernels and SVDQuant linear (`tests/ops/`).
- `a7f58c5` [ROCm] Document building for AMD GPUs (`docs/source/installation/installation.rst`, after "Build from Source"; the README defers installation to the doc site).
- `5c32be5` Add a kernel-level test for the nunchaku-fp16 attention kernel.

Design as built (plan.md has the rationale):

- The mma.sync fragment layout stays the data layout. `mma_rocm.cuh` implements `mma_m16n8k16_f16f16f16f16`, `mma_m16n8k16_f32f16f16f32<bf16>` and `mma_m16n8kx_s32common<s4|u4|s8, s4|s8>` by a lane gather over the 32-lane warp (width-32 `__shfl`) plus scalar dot products, k in increasing order (f16: fp32 accumulation, one rounding).
- Stage 2 CDNA (`gemm_w4a4.cuh` `gemm_w4a4_block_mfma`, macro `NUNCHAKU_ROCM_MFMA_INT4` for `__gfx90a__`/`__gfx942__`/`__gfx950__`): the weights are shared by all warps of a block, so one `v_mfma_i32_32x32x16_i8` (gfx90a: `32x32x8i8`) per wavefront takes weights as A (32 output columns = two n-tiles) and activations as B (16 rows from each of the wavefront's two warps). Operands are loaded from global memory directly in the MFMA layout, s4/u4 widened to s8 with `v_perm` + a sign-extension multiply. Each 32x32 result is staged through the dynamic shared memory (4 waves x 32 x 36 int, 18 KB; the launcher always allocates it on ROCm; a `__syncthreads` precedes the epilogue, which may reuse it) into the mma.sync accumulator layout, and `apply_scales_tile` applies the scales with apply_scales' arithmetic. MFMA layout verified first with a standalone program (`agent_space/nunchaku/mfma/layout.hip`: 0 mismatches).
- Stage 2 RDNA (`gemm_w4a4_block_wmma`, macro `NUNCHAKU_ROCM_WMMA_INT4` for `__GFX11__`/`__GFX12__`): `v_wmma_i32_16x16x16_iu4` per warp tile, weights as A, activations as B; gfx11 operands are 2 dwords (all 16 k, half-waves repeated), gfx12 1 dword (k split across half-waves); D layouts written as gfx11 row = 2v + half, gfx12 row = v + 8 half. Not run anywhere yet.
- `NUNCHAKU_ROCM_MMA_REFERENCE=1` at build time (setup.py adds `-DNUNCHAKU_ROCM_MMA_REFERENCE`) compiles stage 1 in place of both stage 2 paths.
- FP4: traps on every AMD target (question 1). `get_precision()` returns int4 on ROCm; `check_hardware_compatibility` accepts int4 on any AMD GPU and fp4 only on gfx950/gfx1250 (`ROCM_FP4_ARCHS`), which today would then trap at the kernel.
- Q2 = a as ruled: `gemm_w8a8_fp16` not built; `gemm_batched_fp16` = `cublasGemmStridedBatchedEx` (hipBLAS after hipify); `mha_fwd_block` throws "Block-sparse attention is not supported on ROCm". Planner miss fixed: `mha_varlen_fwd` (Sana cross attention) is per-sample torch SDPA.
- The AWQ GEMM config CTA_M=32/CTA_K=128 needs 72 KB of LDS with 4 stages; ROCm uses 3 (52.5 KB). The AWQ split-K semaphores are zeroed on ROCm before launch (upstream allocates them with `Tensor::empty` and never zeroes them; the same latent issue exists on CUDA, left untouched there).
- The QuantizedGEMM88 W8A8 path (`zgemm/gemm_w8a8.cu`) compiles but sets 92160 bytes of dynamic shared memory, above 64 KB on gfx90a/gfx942: it would fail at launch. No caller in `nunchaku/` (plan section E); not exercised.

### Environment and build

- Host gfx942 (MI300X, partitioned; `HIP_VISIBLE_DEVICES=1` as pinned), `ulimit -c 0`, every GPU run under `timeout`.
- Job venv `$FLEET_VENV`: TheRock wheels from `https://stable.repo.amd.com/rocm/whl-next/`: `rocm[libraries,devel,device-gfx942]==10.2.0` (HIP 7.17.26434, AMD clang 24.0.0git), `torch==2.14.0+rocm10.2.0`, `torchvision==0.29.0+rocm10.2.0`, `amd-torch-device-gfx942==2.14.0+rocm10.2.0`. torch 2.15.0+rocm10.2.0 exists but has no torchvision wheel on that index (LPIPS needs torchvision), hence 2.14. Test deps: the pyproject `ci` pins (accelerate 1.9, diffusers 0.36, transformers 4.55.2, peft 0.17, datasets 3.6, torchmetrics 1.8, controlnet-aux 0.0.10, image-gen-aux e08ae3d, opencv 4.11.0.86) plus numpy 2.4.6 with pandas 3.1.1, scikit-learn and scikit-image upgraded to numpy-2 builds.
- `torch.cuda.device_count()` returns 0 before CUDA init on this host (amdsmi sees no device in the container); setup.py calls `torch.cuda.init()` before counting devices.
- The fleet environment exports `PYTORCH_ROCM_ARCH="gfx90a;gfx942;gfx950;gfx1100;gfx1201"`; set it explicitly. Build (documented command): `PATH=$FLEET_VENV/bin:$PATH PYTORCH_ROCM_ARCH=gfx942 MAX_JOBS=64 python -m pip install -e . --no-build-isolation -v`. Iteration builds used `python setup.py build_ext --inplace` (same flags). gfx942-only build: about 8 minutes with stage 2; the stage 1 reference build takes much longer (the bf16 INT4 launch file spent over 45 minutes in device code generation), because the fully inlined lane-gather INT4 code is huge.
- CUDA compile check: not run (no nvcc on this host); owed on the mi355 host before this goes to jeff (standing rule).

- All-target compile of 5c32be5 (separate copy of the tree): `PYTORCH_ROCM_ARCH="gfx90a;gfx942;gfx950;gfx1100;gfx1201" python setup.py build_ext --inplace` -> exit 0, 0 errors. In the code objects of `gemm_w4a4_launch_fp16_int4.o` (`clang-offload-bundler --unbundle`, `llvm-objdump -d --mcpu=<gfx>`): gfx90a 21504 `v_mfma_i32_32x32x8_i8`, gfx942 and gfx950 10752 `v_mfma_i32_32x32x16_i8` each, gfx1100 and gfx1201 21504 `v_wmma_i32_16x16x16_iu4` each, and no matrix instruction of the other family. So each target compiles to its stage 2 path; only gfx942 ran.

### Test results (gfx942, fork 5c32be5, built with the documented editable install)

- `tests/ops`: **38 passed** (12 W4A4 GEMM vs float64 reference, 16 AWQ GEMM/GEMV, 4 SVDQW4A4Linear end to end with simulated activation quantization: rel. error 0.0077/0.0079 fp16, 0.022/0.023 bf16, 6 attention_fp16 vs SDPA incl. NaN-padded keys).
- **Stage 2 vs stage 1, bit for bit** (ruling Q3 = b): `NUNCHAKU_ROCM_MMA_REFERENCE=1` build vs default build, both gfx942, same GPU, script below (34 cases: GEMM with LoRA up and bias for fp16/bf16, signed/unsigned activations, 5 shapes up to 4352x256x3072 and 1024x3072x768; fused GEMM + quantize-for-next-layer + LoRA down; the quantize kernel). Every GEMM output, packed next-layer activation and scale is identical. The only differing outputs are the fp32 `lora_act_out` tensors (10 of them), which the kernels accumulate with float atomics across warps: the reference build differs from itself between two runs in exactly the same 10 outputs, by at most 2.2e-7 relative (the N=256 cases, one block column, match exactly). Result: **bit-identical on all deterministic outputs**.
- Final run, one command on the installed build: `HF_HOME=<job>/hf NUNCHAKU_TEST_CACHE_ROOT=<job>/test_cache python -m pytest -v -s tests/ops tests/v1/sdxl/test_sdxl.py::test_sdxl_lpips tests/v1/sdxl/test_sdxl_turbo.py::test_sdxl_turbo_lpips tests/sana tests/v1/z_image` -> **48 passed** in 564 s (cached INT4 images were deleted first so they were regenerated by this build; the 16-bit reference images were generated on this host by the same tests):
  - `test_sdxl_lpips`: LPIPS 0.148 (upstream expected 0.25, limit x1.15).
  - `test_sdxl_turbo_lpips`: LPIPS 0.212 (expected 0.25).
  - `test_zimage_turbo` rank 32/128/256: LPIPS 0.371/0.374/0.375 (expected int4-bf16 0.40/0.38/0.37, limit x1.15; rank 256 is above the bare expected value and within the limit). The reference images are upstream's, downloaded by the test. The three `_turing` variants return early for bf16 (no check).
  - Earlier runs with intermediate builds (same GEMM outputs bit for bit) gave SDXL 0.154, SDXL-Turbo 0.220, Z-Image 0.384/0.376/0.351.
  - `tests/sana/test_examples.py`: 2 PASSED (sana1.6b.py, sana1.6b_pag.py; C++ QuantizedSanaModel: hipBLAS gemm_f16, depthwise conv, varlen attention, LiteLA epilogue). Image inspected: correct ink-drawing panda. A first run failed with `hipFuncSetAttribute ... invalid argument`: the LiteLA epilogue's 42 KB dynamic shared memory plus a static 18 KB MFMA staging buffer plus 17 KB of epilogue statics exceeded 64 KB; fixed by staging in the dynamic region (now in 9875a78).
  - FLUX C++ model (QuantizedFluxModel) without the gated pipeline: public INT4 FLUX.1-schnell transformer on random inputs (4096 image + 512 text tokens), `flashattn2` (SDPA replacement) vs `nunchaku-fp16`: both finite, output mean |x| 0.885/0.881, cosine 0.980 between them across 57 blocks (`agent_space/nunchaku/flux_attn_check.py`, unit-noise latents, text states x0.01). With unit-variance random 4096-wide text states the nunchaku-fp16 path gives NaN (fp16 overflow in its fp16 Q.K accumulation; the kernel alone is correct, so this is not attributed to the port, but it was not compared with NVIDIA).
- Not run, and why: every FLUX.1 and Qwen-Image test (`black-forest-labs/FLUX.1-*` are gated; this host has no Hugging Face token); `tests/flux/test_flux_qencoder.py` (needs FLUX); `test_flux_speed.py`, `test_turing.py` (skip by design); `test_device_id.py` (one GPU pinned); speed tests (`*_time_cost`, deselected; speed is recorded, never gated). The AWQ T5 kernels are covered by `tests/ops/test_awq.py` instead.

### What remains before `ported` (this round stops in `porting`)

Required by Q3 = b before review, by platform:

- **wave32 RDNA (gfx1100 W7800 / gfx1201 / gfx1151 hosts): prove 67db6fa.** On the RDNA host: build twice, `PYTORCH_ROCM_ARCH=<gfx> pip install -e . --no-build-isolation` and the same with `NUNCHAKU_ROCM_MMA_REFERENCE=1` (separate trees or save `nunchaku/_C*.so` between them); run `python mma_dump.py dump <file>` from the repository root with each build and `python mma_dump.py compare ref.pt wmma.pt`; every non-`lora_act_out` output must be identical. Then `pytest tests/ops` and the end-to-end set above. If the D-layout assumption is wrong the GEMM tests fail outright (values land in the wrong columns), so a failure there points at the `result[col][...]` write in `gemm_w4a4_block_wmma`.
- **RDNA f16/bf16 WMMA for the m16n8k16 primitive** (plan's RDNA stage 2 bullet; surface `kernel:stage2-rdna-wmma-f16`): not written.
- **gfx950 and gfx1250 native FP4** (Q1 = c): not written; waits on question 1.
- **CDNA f16/bf16 MFMA**: only if question 2 is answered (b).
- gfx90a: the MFMA path uses `v_mfma_i32_32x32x8i8` there (compiled in the all-target build, not run); the gfx90a validator runs the bit-identity script.
- CUDA compile of the fork on the mi355 host (nvcc), before anything goes to jeff.

### Bit-identity script (`mma_dump.py`; run from the fork root with the extension built in place, `PYTHONPATH=.`)

```python
"""Run the W4A4 GEMM (and the ops built on it) on fixed random inputs and save every output.

    python mma_dump.py dump OUT.pt        # run from the repository root, with one build of the extension
    python mma_dump.py compare A.pt B.pt  # bitwise comparison of two dumps

Used to prove the matrix-instruction path bit-identical to the reference build
(NUNCHAKU_ROCM_MMA_REFERENCE=1) on the same GPU.
"""

import sys

import torch


def cases():
    from nunchaku.lora.flux.packer import NunchakuWeightPacker
    from tests.ops.test_gemm_w4a4 import pack_act, pack_ascales

    from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda
    from nunchaku.ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda

    G = 64
    packer = NunchakuWeightPacker(bits=4)
    for dtype in (torch.float16, torch.bfloat16):
        for act_unsigned in (False, True):
            for m, n, k in [(256, 128, 128), (512, 384, 1024), (256, 3072, 3072), (1024, 3072, 768), (4352, 256, 3072)]:
                g = torch.Generator().manual_seed(m + 3 * n + 7 * k + int(act_unsigned) + (dtype == torch.bfloat16))
                lo, hi = (0, 16) if act_unsigned else (-8, 8)
                qact = torch.randint(lo, hi, (m, k), dtype=torch.int32, generator=g)
                qwgt = torch.randint(-8, 8, (n, k), dtype=torch.int32, generator=g)
                # include extreme values in a few rows and columns
                qact[:3] = hi - 1
                qact[3:5] = lo
                qwgt[:2] = -8
                qwgt[2:4] = 7
                ascales = (torch.rand(m, k // G, generator=g) * 0.02 + 0.001).to(dtype)
                wscales = (torch.rand(n, k // G, generator=g) * 0.02 + 0.001).to(dtype)
                rank = 32
                lora_up = (torch.randn(n, rank, generator=g) * 0.1).to(dtype)
                lora_act = torch.randn(m, rank, generator=g) * 0.1
                bias = (torch.randn(n, generator=g) * 0.1).to(dtype)
                out = torch.empty(m, n, dtype=dtype, device="cuda")
                svdq_gemm_w4a4_cuda(
                    act=pack_act(qact).cuda(),
                    wgt=packer.pack_weight(qwgt).cuda(),
                    out=out,
                    ascales=pack_ascales(ascales).cuda(),
                    wscales=packer.pack_scale(wscales, group_size=G).cuda(),
                    lora_act_in=lora_act.cuda(),
                    lora_up=packer.pack_lowrank_weight(lora_up, down=False).cuda(),
                    bias=packer.pack_scale(bias, group_size=-1).cuda(),
                    act_unsigned=act_unsigned,
                )
                yield f"gemm_{dtype}_{act_unsigned}_{m}x{n}x{k}", out

                # fused path: gemm output quantized again for the next layer (EpilogueQuantize) plus LoRA down
                if n % 256 == 0:
                    smooth = (torch.rand(n, generator=g) + 0.5).to(dtype)
                    lora_down = (torch.randn(rank, n, generator=g) * 0.1).to(dtype)
                    qout = torch.empty(m, n // 2, dtype=torch.uint8, device="cuda")
                    oscales = torch.empty(n // G, m, dtype=dtype, device="cuda")
                    lora_act_out = torch.zeros(m, rank, dtype=torch.float32, device="cuda")
                    out2 = torch.empty(m, n, dtype=dtype, device="cuda")
                    svdq_gemm_w4a4_cuda(
                        act=pack_act(qact).cuda(),
                        wgt=packer.pack_weight(qwgt).cuda(),
                        out=out2,
                        qout=qout,
                        ascales=pack_ascales(ascales).cuda(),
                        wscales=packer.pack_scale(wscales, group_size=G).cuda(),
                        oscales=oscales,
                        lora_act_in=lora_act.cuda(),
                        lora_up=packer.pack_lowrank_weight(lora_up, down=False).cuda(),
                        lora_down=packer.pack_lowrank_weight(lora_down, down=True).cuda(),
                        lora_act_out=lora_act_out,
                        bias=packer.pack_scale(bias, group_size=-1).cuda(),
                        smooth_factor=packer.pack_scale(smooth, group_size=-1).cuda(),
                        act_unsigned=act_unsigned,
                    )
                    yield f"fused_{dtype}_{act_unsigned}_{m}x{n}x{k}", (out2, qout, oscales, lora_act_out)

        # quantize kernel (no matrix instructions; must match too)
        x = (torch.randn(512, 3072, generator=torch.Generator().manual_seed(5)) * 2).to(dtype).cuda()
        lora_down = (torch.randn(32, 3072, generator=torch.Generator().manual_seed(6)) * 0.1).to(dtype)
        smooth = (torch.rand(3072, generator=torch.Generator().manual_seed(7)) + 0.5).to(dtype)
        q, s, la = svdq_quantize_w4a4_act_fuse_lora_cuda(
            x,
            lora_down=packer.pack_lowrank_weight(lora_down, down=True).cuda(),
            smooth=packer.pack_scale(smooth, group_size=-1).cuda(),
        )
        yield f"quantize_{dtype}", (q, s, la)


def dump(path):
    results = {}
    for name, value in cases():
        torch.cuda.synchronize()
        if isinstance(value, tuple):
            results[name] = tuple(v.cpu() for v in value)
        else:
            results[name] = value.cpu()
    torch.save(results, path)
    print(f"saved {len(results)} results to {path}")


def as_bytes(t):
    return t.contiguous().view(torch.uint8)


def compare(a_path, b_path):
    a, b = torch.load(a_path), torch.load(b_path)
    assert a.keys() == b.keys(), "different case lists"
    bad = 0
    for name in a:
        va = a[name] if isinstance(a[name], tuple) else (a[name],)
        vb = b[name] if isinstance(b[name], tuple) else (b[name],)
        for i, (x, y) in enumerate(zip(va, vb)):
            same = x.shape == y.shape and torch.equal(as_bytes(x), as_bytes(y))
            nbytes = as_bytes(x).numel()
            ndiff = (as_bytes(x) != as_bytes(y)).sum().item() if x.shape == y.shape else nbytes
            print(f"{'OK  ' if same else 'DIFF'} {name}[{i}] {tuple(x.shape)} {x.dtype} differing bytes {ndiff}/{nbytes}")
            bad += not same
    print(f"{len(a)} cases, {bad} outputs differ")
    return bad


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2])
    else:
        sys.exit(1 if compare(sys.argv[2], sys.argv[3]) else 0)
```

## CUDA compile check 2026-11-24 (mi355 fleet job, nvcc, no GPU used, for jeff)

Result: **PASS, no CUDA regression at fork `5c32be5` against upstream base `302e0e97`.** Same 36 objects, 0 errors in both, the identical warning set, a byte-identical `build.ninja`, and identical SASS for every device object except one `__assertfail` line-number literal. Nothing in the fork was changed or pushed; stage and lock untouched.

### Environment

- nvcc `/usr/local/cuda-12.8/bin/nvcc`: Cuda compilation tools, release 12.8, V12.8.93. Host compiler `c++` = g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0. No NVIDIA GPU on the host (`torch.cuda.device_count()` 0); `CUDA_VISIBLE_DEVICES=` set anyway; no AMD GPU touched.
- Job venv (Python 3.11.17): `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128` (`--index-url https://download.pytorch.org/whl/cu128`; pin both: an unpinned `torchvision` pulled torch 2.15.0+cu132 from PyPI), `diffusers==0.36.0`, `transformers==4.55.2`, `accelerate==1.9`, `peft==0.17`, `huggingface-hub==0.34`, einops, protobuf, sentencepiece.
- Trees: `git clone https://github.com/AMD-Ecosystem/nunchaku`, `git checkout 5c32be5`, `git worktree add ../base 302e0e97`, `git submodule update --init --recursive` in each. Submodule pointers are the same at both commits (cutlass v3.3.0 a75b4ac4, Block-Sparse-Attention 99511c34, json 63258397, mio 8b6b7d87, spdlog 27cb4c76).

### Build (identical for both trees, run in parallel)

```
export PATH=$FLEET_VENV/bin:/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8 CUDA_VISIBLE_DEVICES= NUNCHAKU_INSTALL_MODE=ALL MAX_JOBS=16
python setup.py build_ext --inplace -v
```

`NUNCHAKU_INSTALL_MODE=ALL` because the default FAST mode reads the visible GPUs and there are none; ALL is the wheel target set: `Detected SM targets: ['75', '80', '86', '89', '120a']` (sm_120a since nvcc >= 12.8), so the sm_75-only and the FP4 (sm_120a) kernel bodies were compiled too. Base: exit 0 in 285 s; 5c32be5: exit 0 in 288 s. `setup.py --version` gives `1.3.0.dev20261124+cu12.8torch2.11` in both.

### Comparison

- **setup.py takes the CUDA branch unchanged on a CUDA torch**: the generated `build/temp*/build.ninja` is byte-identical between the trees once the tree path is substituted: same 36 sources (Block-Sparse-Attention's 10 and `gemm_w8a8.cu` included, `flash_api_rocm.cpp` absent), same `cflags`/`post_cflags` (g++ `-std=c++20 -Og ...`) and `cuda_post_cflags` (nvcc `-g -std=c++20 -UNDEBUG -Xcudafe --diag_suppress=20208 -U__CUDA_NO_*` x8 `--threads=5 --expt-relaxed-constexpr --expt-extended-lambda --ptxas-options=--allow-expensive-optimizations=true --generate-line-info` plus `-gencode arch=compute_X,code=sm_X` for 75/80/86/89/120a). No HIP flag, no `PYTORCH_ROCM_ARCH`, no `-DUSE_ROCM` reaches the CUDA build. Linked `NEEDED` libraries of `_C*.so` identical (libc10, libc10_cuda, libcudart.so.12, libtorch, libtorch_cpu, libtorch_python, libstdc++, ...).
- **Objects**: the same 36 `.o` in both trees; per-file compile result PASS for every one in both: `nunchaku/csrc/pybind`, `src/{FluxModel,Linear,Module,SanaModel,Serialization,activation,layernorm,interop/torch}`, `src/kernels/{activation_kernels,dwconv,gemm_batched,gemm_f16,layernorm_kernels,misc_kernels}`, `src/kernels/awq/{gemm_awq,gemv_awq}`, `src/kernels/zgemm/{attention,gemm_w4a4,gemm_w4a4_launch_{bf16_fp4,bf16_int4,fp16_fp4,fp16_int4,fp16_int4_fasteri2f},gemm_w4a4_test,gemm_w8a8}`, the 2 Block-Sparse-Attention `.cpp` and its 8 `flash_fwd*_sm80.cu`.
- **Errors**: 0 in both logs.
- **Warnings**: identical multiset (paths and line numbers normalized): `gemm_awq.cu` #177-D unused `blockIdx_x`/`blockIdx_z`/`kSmemCol`/`kSmemSizeZeros`/`total_global_iters` (40 instances, same counts in both; `gemm_awq.cu` is a changed file and gains none), `FluxModel.cpp` unused `device` (pre-existing), Block-Sparse-Attention `flash_api.cpp` unused `counter_offset`/`device_guard`. No new warning in any changed file.
- **SASS** (`cuobjdump -sass` per object, all five cubins, `nvdisasm` 12.8 from the triton wheel since the host toolkit lacks it; source-path identifier line dropped): identical function lists everywhere, and identical instructions and encodings for all 26 device-code objects except `layernorm_kernels.o`, which differs in 4 instructions (8 diff lines): `IMAD.MOV.U32 R8, RZ, RZ, 0x1da` -> `0x24f` and `0x1cc` -> `0x241` before the `__assertfail` call in `vllm::generalLayerNorm_fuse_sum`. Those are the `__LINE__` arguments of the two `assert(false)` in `src/kernels/utils.cuh`, which moved from lines 474/460 to 591/577 because the port added lines above them. So the CUDA machine code is unchanged.
- **Shared C++ fixes** `a4abb15` (the `template` disambiguator in `attention.cu`, `epilogues.cuh`, `gemm_base.cuh`, `gemm_w4a4.cuh`, `gemm_w4a4_launch_impl.cuh`, `gemm_w4a4_test.cu`) and `6e23cc9` (`std::array<half2_t, 2>{...}` in `gemm_base.cuh`): compile cleanly under nvcc 12.8 and its g++ 13.3 host pass (these headers are reached only from `.cu` files), with no new warning, and the objects that include them (`attention.o`, the five `gemm_w4a4_launch_*.o`, `gemm_w4a4_test.o`, `gemm_w8a8.o`) have identical SASS to the base, confirming "generated code is unchanged". Both commit bodies say "the nvcc build was not run here"; that sentence is now out of date and the porter may want to replace it with the nvcc result when those commits are next touched (no change made here).

### Python (what a CUDA install takes)

- `import nunchaku` (full package, with the fork's CUDA-built `_C`), `nunchaku._C`, `nunchaku.ops.gemm`, `nunchaku.ops.quantize`, `nunchaku.models`, `nunchaku.lora.flux.packer`: import OK on torch 2.11.0+cu128 in both trees. The pybind surface (`dir()` of `_C.ops`, `_C.utils`, `_C.QuantizedFluxModel`) is identical. `get_rocm_arch()` returns None on CUDA. `py_compile` of `tests/ops/*.py`: OK.
- Behaviour, `nunchaku/utils.py` loaded from each tree with `torch.cuda.get_device_capability` faked to sm 7.5, 8.0, 8.6, 8.9, 9.0, 10.0, 12.0, 12.1 and `torch.cuda.get_device_properties` replaced by a function that raises (so the ROCm arm cannot be entered silently): `get_precision("auto")` for devices `cuda`, `cuda:1`, `torch.device("cuda", 0)` and with int4/fp4 model names (incl. the warnings), `get_precision("int4"|"fp4")`, `is_turing()` on two devices, `check_hardware_compatibility` for int4 and `fp4_e2m1_all` configs: JSON output of base and 5c32be5 is identical (e.g. sm75 int4 + is_turing True; sm80/86/89 int4, fp4 config rejected; sm120/121 fp4, int4 config rejected; sm90/sm100 "Unsupported GPU architecture").

### CUDA fixes proposed

None needed.
## Porter round 2026-11-24, RDNA (porter, linux-gfx1100 fleet job, for jeff)

Scope from jeff's dispatch (Q3 = b, RDNA part): prove 67db6fa on an RDNA GPU, write and prove the RDNA f16/bf16 WMMA path for the m16n8k16 primitive, write neither FP4 nor CDNA f16/bf16 MFMA (both wait on questions 1 and 2 above). Stage left at `porting`. Fork head 5c32be5 -> **8aa04e7**.

### Environment

- Host gfx1100 (Radeon Pro W7800 48 GB), `HIP_VISIBLE_DEVICES` pinned to one GPU by the dispatcher, `ulimit -c 0`, every GPU run under `timeout`, MAX_JOBS=8 (one short overlap at 9 while the cross-arch compile checks ran beside the last reference-build file).
- Job venv `$FLEET_VENV` (the image's venv has rocm 10.1 linked in): `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel,device-gfx1100]==10.2.0" "torch==2.14.0+rocm10.2.0" "torchvision==0.29.0+rocm10.2.0" "amd-torch-device-gfx1100==2.14.0+rocm10.2.0"`, then the same test dependencies as the gfx942 round (accelerate 1.9, diffusers 0.36, transformers 4.55.2, peft 0.17, datasets 3.6, torchmetrics 1.8, huggingface-hub 0.34, controlnet-aux 0.0.10, image-gen-aux e08ae3d, opencv 4.11.0.86, pytest 8.4.2, pandas 3.1.1 and current scikit-learn/scikit-image for numpy 2.4.6) plus matplotlib (the SDXL tests import it). The fleet guard refuses `python -m pip` unless the interpreter is spelled `$FLEET_VENV/bin/python`.
- **torch SDPA is broken on gfx1100 in this wheel set**: `torch.nn.functional.scaled_dot_product_attention` with the flash or memory-efficient backend fails with `hipErrorInvalidImage` (fp16, bf16, fp32; math backend works). The wheels ship no `aotriton.images` directory, and the index has no separate AOTriton package. This is outside the port, but it hits the port's `flash_api_rocm.cpp` (SDPA) and every diffusers text encoder. For the end-to-end tests a `sitecustomize.py` on `PYTHONPATH` called `torch.backends.cuda.enable_flash_sdp(False)` and `enable_mem_efficient_sdp(False)` (subprocesses inherit it). No source change.
- Builds, `PYTORCH_ROCM_ARCH=gfx1100`, `python setup.py build_ext --inplace` in three separate trees (git worktrees of the fork): default 5c32be5 315 s (7 jobs); `NUNCHAKU_ROCM_MMA_REFERENCE=1` 5c32be5 **8064 s** (its last file, `gemm_w4a4_launch_bf16_int4.hip`, compiled alone on one core for about 1 h 50 min at 1.3 GB RSS); default working tree (8aa04e7 content) 398 s. Documented command afterwards on the working tree: `PYTORCH_ROCM_ARCH=gfx1100 $FLEET_VENV/bin/python -m pip install -e . --no-build-isolation -v` -> exit 0, 375 s, `nunchaku-1.3.0.dev20261124+rocm10.2torch2.14`. Builds were run detached (a background shell is stopped at two hours) and not wrapped with `timeit.sh`, so no compile telemetry was recorded for them.

### 1. 67db6fa (INT4 main loop on WMMA): proven, no fix needed

- `mma_dump.py` (the script above, unchanged) with the reference build and with the default build of 5c32be5, same GPU: `python mma_dump.py compare dump_ref.pt dump_wmma.pt` -> **34 cases, 10 outputs differ, and the 10 are exactly the fp32 `lora_act_out` tensors** (float atomics across warps). A second run of the reference build against the first differs in the same 10 tensors (`differing bytes` of the same order, e.g. 23547/131072 vs 23600/131072), so they are run-to-run nondeterminism, not the WMMA path. Every GEMM output, packed next-layer activation and output scale is bit-identical. The gfx11 D layout assumption in `gemm_w4a4_block_wmma` (`row = 2v + half`) is therefore right.
- `pytest tests/ops` on the 5c32be5 default build: 38 passed.

### 2. RDNA f16/bf16 WMMA for the m16n8k16 primitive: fork 8aa04e7

Design: every caller issues the m16n8k16 product in pairs that share the A fragment (`GEMMBase::mma_f16xf16_f32` for the LoRA up/down epilogues and LiteLA, `attention.cuh` `mma_f16xf16_f16`, both AWQ main loops). On ROCm the pair now goes through new helpers in `mma_rocm.cuh`, `mma_m16n16k16_f32f16f16f32<bf16>(a, b, c0, c1)` and `mma_m16n16k16_f16f16f16f16(a, b, c)`. With `NUNCHAKU_ROCM_WMMA_F16` (gfx11/gfx12, not with `NUNCHAKU_ROCM_MMA_REFERENCE`) they run one `v_wmma_f32_16x16x16_{f16,bf16}`; elsewhere they call the stage 1 m16n8k16 functions twice, exactly as the callers did before.
- The WMMA computes the transposed product (B as the WMMA A operand, A as the B operand). Moving fragments into the WMMA layout and the result back takes eight width-32 shuffles each, with a rotation schedule: in step s, lanes taking index < 8 read word s and lanes taking index >= 8 word s + 2, so every source lane is read for one register per step. The result goes back with source i = (q + t) % 4. Selects pick the register at source and destination; no shared memory is used, so callers' LDS budgets are untouched.
- C is not passed to the WMMA. It computes from a zero accumulator and C is added in fp32 afterwards (f16-accumulator variant: f16 C widened, added, rounded once, as stage 1 does). Reason: gfx11 WMMA does not round its float accumulator to nearest (cuda-to-rocm `faults-numerics.md`), and keeping C outside keeps that bias out of the callers' long k chains. Measured below: the isolated 16x16x16 WMMA returned 1-ulp-off results for exact small-integer data (16 of 256 outputs).
- Code objects (`clang-offload-bundler` on `.hip_fatbin`): gfx1100 and gfx1201 both have `v_wmma_f32_16x16x16_f16` (attention 64, AWQ 72, fp16 INT4 launch 3888) and `..._bf16` (AWQ 72, bf16 INT4 launch 4400 / 4320 on gfx1201) beside the 21504 `v_wmma_i32_16x16x16_iu4`; gfx942 has none, only its 10752 `v_mfma_i32_32x32x16_i8` per INT4 launch file.

Proof on gfx1100:
- Primitive test (`notes-rdna-scripts.md`, `test.hip`), 2000 random 16x16x16 problems per mode at three scales, against both the reference functions and a float64 host reference: layout right (no output off by more than fp32 rounding). Max |error| / sum |a*b| (including |c|): f32-accumulator f16 inputs: WMMA 2.4e-7, reference 3.0e-7; bf16 inputs: WMMA 1.6e-7, reference 1.3e-7; both under 17 x 2 x 2^-24. f16 accumulator: 99.95% of outputs bitwise equal to the reference, the rest one f16 ulp apart. Exact-integer problem: reference exact, WMMA 16/256 outputs one fp32 ulp off (the gfx11 accumulator rounding above), so no exactness test is used for WMMA on gfx11.
- Op outputs, reference build (5c32be5, stage 1 everywhere) vs default 8aa04e7, `f16_dump.py` (the 34 GEMM cases of `mma_dump.py` plus 8 attention_fp16 and 8 AWQ GEMM cases up to 4608 tokens and 1024x1024x4096): **floating-point bit identity is not expected**, because the WMMA sums each 16-term product in its own order and rounding. Tolerance applied: max |new - ref| <= 2 units of roundoff of the output type times max |output| (2 x 2^-11 = 9.8e-4 for float16, 2 x 2^-8 = 7.8e-3 for bfloat16). Reason: per product the two differ by about 1e-7 relative (primitive test), far below the 16-bit output rounding, so differences can only appear as output-rounding flips, which downstream steps (softmax normalisation, LoRA, re-quantisation) carry into neighbouring values by at most a unit or two. Measured: worst max |diff| / max |out| 5.2e-4 for float16 outputs (AWQ 1024x1024x4096) and 4.0e-3 for bfloat16 (attention 512 tokens with 500 valid), all within the bound; GEMM outputs max 4 ulp element-wise (8 in one fused case at a small value); packed 4-bit next-layer activations: at most 2 nibbles differ per case (of 0.8 to 3.1 million), by one step; output scales max 2 ulp. Each build's own error against float64 (attention vs SDPA in float64, AWQ vs float64 matmul) is the same to three significant digits (e.g. attention fp16 4608 tokens 0.00239 both, AWQ bf16 512x4096x4096 0.00304 both). The quantize kernel, which uses no 16-bit product for its quantized output and scales, is bit-identical there; the fp32 `lora_act_out` of the fused cases, whose LoRA-down now goes through the WMMA product, differs by up to 1.4e-4 of its largest value (run-to-run noise alone is 3e-7).
- Two runs of the 8aa04e7 build against each other: only the 10 `lora_act_out` tensors differ (max rel 3.1e-7), so the WMMA path is deterministic.
- `pytest tests/ops` on the 8aa04e7 build: **38 passed**.
- End to end on the installed 8aa04e7 build (with the SDPA workaround above): `HF_HOME=<job>/hf NUNCHAKU_TEST_CACHE_ROOT=<job>/test_cache python -m pytest -v -s tests/v1/sdxl/test_sdxl.py::test_sdxl_lpips tests/v1/sdxl/test_sdxl_turbo.py::test_sdxl_turbo_lpips tests/sana tests/v1/z_image` -> **10 passed** in 1975 s: SDXL LPIPS 0.157 (expected 0.25), SDXL-Turbo 0.222 (0.25), Z-Image Turbo rank 32/128/256 0.393/0.387/0.371 (expected int4-bf16 0.40/0.38/0.37, limit x1.15), both Sana example scripts. The 16-bit SDXL references were generated on this host by the tests; the Z-Image references are upstream's. A first run without the workaround failed all six image tests with `hipErrorInvalidImage` inside torch SDPA.
- Speed (recorded, not gated), W7800, ms, 5c32be5 default vs 8aa04e7: AWQ GEMM 512x4096x4096 fp16 5.81 -> 3.57, bf16 7.79 -> 3.68; attention_fp16 24 heads x 4608 tokens 119.5 -> 56.7; W4A4 GEMM 4096x3072x3072 with LoRA rank 32 bf16 6.32 -> 5.20, rank 128 6.85 -> 5.73, but **fp16 6.80 -> 7.73 (rank 32) and 7.07 -> 8.10 (rank 128), 14% slower**. The INT4 GEMM kernels spill heavily in both builds (192 VGPRs, 1.0 to 2.7 KB scratch per lane depending on the epilogue, moving both ways between the builds); the fp16 slowdown is a register-allocation effect of the larger LoRA epilogue code in kernels that already spill, a performance-pass item.

### Shared code the CDNA path uses (gfx942 must rerun)

8aa04e7 changes code that gfx90a/gfx942/gfx950 compile: `gemm_base.cuh` `mma_f16xf16_f32`, `attention.cuh` `mma_f16xf16_f16` and the AWQ loops now call the `mma_m16n16k16_*` helpers on all ROCm targets (the AWQ ROCm `mma_m16n8k16` wrapper became `mma_m16n16k16`). On CDNA the helpers call the same stage 1 functions twice with the same arguments in the same order, so results should be bit-identical to what gfx942 proved, and the gfx942 code objects contain no WMMA (compiled here with the build's command line and `--offload-arch=gfx942`: attention, AWQ, both INT4 launch files, exit 0). It has **not been run on CDNA**: the gfx942 (and gfx90a) validator should rerun `mma_dump.py compare` against the reference build, and `f16_dump.py compare` between a 5c32be5 and an 8aa04e7 default build, where every output except `lora_act_out` should be identical.

### What remains before `ported` (stage stays `porting`)

- **gfx12 proof** (surface `kernel:stage2-rdna-wmma-gfx12`): the gfx12 operand and result layouts of both RDNA paths (iu4 in `gemm_w4a4_block_wmma`, f16/bf16 in `mma_rocm::wmma_m16n16k16_f32`) compile for gfx1201 (code objects checked above) but cannot run on gfx1100. A gfx1201 host must: build `test.hip` for gfx1201 and run it; build the extension twice (default and `NUNCHAKU_ROCM_MMA_REFERENCE=1`; plan about 2.5 hours for the reference build with 8 jobs, its bf16 INT4 file alone takes about 2 hours); `mma_dump.py compare` (only `lora_act_out` may differ) and `f16_dump.py compare` / `reldiff.py` (within the tolerance above); `pytest tests/ops` and the end-to-end set. If the gfx12 D layout (`row = v + 8 * half`) is wrong, the GEMM outputs fail outright.
- **gfx942 rerun** of the shared-code change (above) and **gfx90a run** of the MFMA path (`v_mfma_i32_32x32x8i8`, compiled, never run).
- **FP4** on gfx950/gfx1250 (question 1) and **CDNA f16/bf16 MFMA** (question 2): not written, waiting on jeff.
- **nvcc compile** of the fork on the mi355 host before anything goes to jeff (8aa04e7 touches only `USE_ROCM` branches and `mma_rocm.cuh`, but the standing rule applies).
- Windows gfx1151 last, per the standing ruling.

## CUDA compile check 2026-11-24, fork 8aa04e7 (mi355 fleet job, nvcc, no GPU used, for jeff)

Result: **PASS, no CUDA regression at fork `8aa04e7` against upstream base `302e0e97`.** Same 36 objects, 0 errors in both, the identical warning multiset, a byte-identical `build.ninja`, identical SASS for every device object except the two `__assertfail` line-number literals already seen at 5c32be5, and host code that differs only where 9875a78 calls `cudaFuncSetAttribute` with a `const void *` (the same C runtime call the template overload makes; details below). The fork was checked at 8aa04e7 exactly; nothing in it was changed or pushed; stage (`porting`) and the linux-gfx1201 lock were not touched. No GPU was used (none was assigned; `CUDA_VISIBLE_DEVICES=` set for every command).

### Environment

- nvcc `/usr/local/cuda-12.8/bin/nvcc`: Cuda compilation tools, release 12.8, V12.8.93 (Build cuda_12.8.r12.8/compiler.35583870_0). Host compiler `c++` = g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0.
- Job venv `$FLEET_VENV` (Python 3.11): `$FLEET_VENV/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 torchvision==0.26.0+cu128`, then `diffusers==0.36.0 transformers==4.55.2 accelerate==1.9 peft==0.17 huggingface-hub==0.34 einops protobuf sentencepiece` (torch stayed 2.11.0+cu128).
- Trees, in the job directory: `git clone https://github.com/AMD-Ecosystem/nunchaku port; git -C port checkout 8aa04e7; git -C port worktree add ../base 302e0e97; git submodule update --init --recursive` in each. Submodule pointers identical at both commits (Block-Sparse-Attention 99511c34, cutlass a75b4ac4, json 63258397, mio 8b6b7d87, spdlog 27cb4c76).

### Build (identical for both trees, run in parallel)

```
export PATH=$FLEET_VENV/bin:/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8 CUDA_VISIBLE_DEVICES= NUNCHAKU_INSTALL_MODE=ALL MAX_JOBS=16
python setup.py build_ext --inplace -v
```

`NUNCHAKU_INSTALL_MODE=ALL` because FAST reads the visible GPUs and there are none; `Detected SM targets: ['75', '80', '86', '89', '120a']`, so the sm_75-only and FP4 (sm_120a) kernel bodies were compiled too. Base: exit 0 in 297 s; 8aa04e7: exit 0 in 305 s. `python setup.py --version` -> `1.3.0.dev20261124+cu12.8torch2.11` in both. (Builds were not wrapped with `timeit.sh`; no compile telemetry recorded.)

### Comparison

- **setup.py takes the CUDA branch unchanged on a CUDA torch**: `build/temp*/build.ninja` is byte-identical between the trees after substituting the tree path: 25 `cuda_compile` + 11 `compile` edges (Block-Sparse-Attention's 10 sources and `gemm_w8a8.cu` included, `flash_api_rocm.cpp` absent), same g++ and nvcc flags, `-gencode` for compute_75/80/86/89/120a, no `hip`/`rocm` string anywhere in it. The ROCm code in setup.py is behind `IS_ROCM = torch.version.hip is not None`. `readelf -d` `NEEDED` of `_C*.so` identical (libc10, libtorch, libtorch_cpu, libtorch_python, libcudart.so.12, libc10_cuda, libstdc++, libm, libgcc_s, libc, ld-linux).
- **Objects, per file: PASS for all 36 in both trees**: `nunchaku/csrc/pybind`, `src/{FluxModel,Linear,Module,SanaModel,Serialization,activation,layernorm,interop/torch}`, `src/kernels/{activation_kernels,dwconv,gemm_batched,gemm_f16,layernorm_kernels,misc_kernels}`, `src/kernels/awq/{gemm_awq,gemv_awq}`, `src/kernels/zgemm/{attention,gemm_w4a4,gemm_w4a4_launch_{bf16_fp4,bf16_int4,fp16_fp4,fp16_int4,fp16_int4_fasteri2f},gemm_w4a4_test,gemm_w8a8}`, Block-Sparse-Attention `flash_api`, `flash_api_adapter` and its 8 `flash_fwd*_sm80.cu`.
- **Errors**: 0 lines matching `error` in either log.
- **Warnings**: identical multiset after normalizing paths and line numbers: `gemm_awq.cu` #177-D unused `blockIdx_x` (10), `blockIdx_z` (5), `kSmemCol` (5), `kSmemSizeZeros` (10), `total_global_iters` (10); `FluxModel.cpp` unused `device` (1); Block-Sparse-Attention `flash_api.cpp` unused `counter_offset`/`device_guard` (3 each). All pre-existing; no changed file gains a warning.
- **SASS** (`cuobjdump -sass` per object, all five cubins; `nvdisasm` 12.8 from the venv's triton wheel because the host toolkit has none; source-path `identifier` line dropped): 24 of the 25 device objects identical, including `attention.o`, all five `gemm_w4a4_launch_*.o`, `gemm_w4a4_test.o`, `gemm_w8a8.o`, `gemm_awq.o`, `gemv_awq.o`. `layernorm_kernels.o` differs in 4 instructions: `IMAD.MOV.U32 R8, RZ, RZ, 0x1da` -> `0x24f` (x3) and `0x1cc` -> `0x241`, the `__LINE__` arguments (474 -> 591, 460 -> 577) of the two `assert(false)` in `src/kernels/utils.cuh`, which moved because the port added lines above them. The CUDA device code is unchanged.
- **Host code** (new in this check; `objdump -d -C` of every object, compared per function so emission order and addresses do not count, nvcc's random `tmpxft_*` names normalized): 27 of 36 objects have identical function sets and bodies apart from `mov $imm,%edx|%esi` immediates (the `__LINE__` argument of `checkCUDA`/error macros, shifted by inserted lines). The other 9 (`gemm_awq.o`, `attention.o`, the five `gemm_w4a4_launch_*.o`, `gemm_w4a4_test.o`, `gemm_w8a8.o`) differ for one source reason: 9875a78 changes 11 shared, unguarded `cudaFuncSetAttribute(func, ...)` calls (`gemm_awq.cu` 3, `attention.cu` 1, `gemm_w4a4_launch_impl.cuh` 3, `gemm_w4a4_test.cu` 2, `gemm_w8a8.cu` 2) to `cudaFuncSetAttribute(reinterpret_cast<const void *>(func), ...)`. On CUDA the template overload is `return ::cudaFuncSetAttribute((const void*)func, attr, value);` (`cuda_runtime.h:2402-2410`), so the base instantiates that wrapper per kernel type (86 in each launch file, 4 in `gemm_awq.o`; e.g. `gemm_awq.o` has 4 `R_X86_64_PLT32 cudaFuncSetAttribute` relocations, all inside the wrappers) and the port calls the same C function directly from the call sites (10 relocations in `gemm_awq.o`, 0 wrappers). The remaining instruction differences in those functions are a dropped alignment `nop` in device stubs and TLS-init call offsets. Same runtime call, same arguments: no behaviour change. For the reviewer: this shared-code cast sits in the `[ROCm]` commit 9875a78; it is harmless on CUDA, but it is not ROCm-specific code.
- **Shared C++ fixes** `a4abb15` (the `template` disambiguator in `attention.cu`, `epilogues.cuh`, `gemm_base.cuh`, `gemm_w4a4.cuh`, `gemm_w4a4_launch_impl.cuh`, `gemm_w4a4_test.cu`) and `6e23cc9` (`std::array<half2_t, 2>{...}` in `gemm_base.cuh`): compile under nvcc 12.8 and its g++ 13.3 host pass with no error or new warning, and every object that includes them has SASS identical to the base. The commit bodies' "the nvcc build was not run here" remains out of date (noted at 5c32be5; unchanged here).
- **8aa04e7 itself** (`gemm_awq.cu`, `attention.cuh`, `gemm_base.cuh`, `mma_rocm.cuh`): every change is inside `#ifdef USE_ROCM`, and `mma_rocm.cuh` is included only under `USE_ROCM` (`gemm_base.cuh:8-12`, `gemm_awq.cu:12-13`); the SASS and host-code results above confirm the CUDA build does not see it.

### Python (what a CUDA install takes)

- From `/tmp` with `PYTHONPATH=<tree>`: `import nunchaku, nunchaku._C, nunchaku.ops.gemm, nunchaku.ops.quantize, nunchaku.models, nunchaku.lora.flux.packer, nunchaku.utils` OK in both trees on torch 2.11.0+cu128 with each tree's CUDA-built `_C`. Pybind surface (`dir()` of `_C.ops`, `_C.utils`, `_C.QuantizedFluxModel`, `_C.QuantizedSanaModel`): 101 names, identical hash in both. `get_rocm_arch()` returns None on CUDA. `py_compile` of `tests/ops/*.py`: OK.
- Behaviour: `nunchaku/utils.py` of each tree loaded by path (`python -I`), `torch.cuda.get_device_capability` faked to sm 7.5, 8.0, 8.6, 8.9, 9.0, 10.0, 12.0, 12.1 and `torch.cuda.get_device_properties` replaced by a function that raises (so a ROCm arm cannot be entered silently); per capability: `get_precision("auto")` for devices `"cuda"`, `"cuda:1"`, `torch.device("cuda", 0)` with no model and with int4/fp4 model names (return value and warnings), `get_precision("int4"|"fp4")` with each model name, `is_turing()`, and `check_hardware_compatibility` for int4 and `fp4_e2m1_all` configs: **192 results, base and 8aa04e7 identical** (sm75 int4 + `is_turing` True; sm80/86/89 int4, fp4 config rejected; sm120/121 fp4, int4 config rejected; sm90/sm100 "Unsupported GPU architecture"); the port's extra `get_rocm_arch` returned None for all three devices.

### CUDA fixes proposed

None needed.

## Lesson for cuda-to-rocm: hipify turns `std::max`/`std::min` into HIP's non-constexpr device functions

(For `strategy-b-torch.md`, hipify section.) torch's build-time hipify rewrites `std::max(` and `std::min(` to `::max(` and `::min(`, HIP's `__device__` overloads. Runtime uses still compile, but a compile-time use breaks with "constexpr variable must be initialized by a constant expression" or "reference to __device__ function 'min' in global initializer" (nunchaku: `static constexpr int X = std::min(WARP_SIZE, WARP_N / PACK)` in a class template, and `constexpr int unroll = 16 / std::max(sizeof(a), sizeof(b))`). Fix without touching behaviour: give the template argument explicitly, `std::min<int>(...)`, which hipify's pattern does not match. Seen with ROCm 10.2 / torch 2.14 (nunchaku).

## Lesson for cuda-to-rocm: a shared-memory staging buffer for matrix-instruction results counts against the 64 KB with the kernel's other shared memory

(For `platform-limits.md`, shared memory.) When a ported GEMM stages MFMA/WMMA results through LDS to convert them back to an mma.sync accumulator layout, a static `__shared__` buffer adds to every kernel instantiation, including those whose epilogue already takes large dynamic shared memory; on 64 KB parts the launch then fails with `hipFuncSetAttribute(MaxDynamicSharedMemorySize) ... invalid argument` only for those epilogues (nunchaku: 18 KB staging + 42 KB LiteLA epilogue + 17 KB other statics). Put the staging buffer in the dynamic region, have the launcher always request at least its size, and add a block barrier before the epilogue reuses it.

## Lesson for cuda-to-rocm: moving mma.sync fragments into WMMA without shared memory, and keeping C out of the gfx11 accumulator

(For `no-hip-equivalent.md`, the mma.sync reimplementation bullets.) When m16n8k16 products come in pairs sharing A (they usually do: an n16 tile), one gfx11/gfx12 `v_wmma_f32_16x16x16_{f16,bf16}` replaces both. Compute the transposed product (B as the WMMA A operand, A as its B operand): then each fragment converts into the WMMA operand layout with the same eight width-32 shuffles, and the result returns to the accumulator layout with eight more, if the steps are rotated so that each source lane is read for only one register per step (operands: lanes taking index < 8 read word s, lanes taking index >= 8 word s + 2; result: lane 4g + t reads source (q + t) % 4 of its four source lanes), with selects at source and destination instead of shuffling every candidate register. No LDS is needed, so callers' shared-memory budgets do not change. Do not pass the running accumulator to the WMMA: on gfx11 its float accumulator does not round to nearest, so start it from zero and add C in fp32 afterwards; that keeps the bias inside one 16-term product instead of growing along the caller's k loop. Measured on gfx1100 (nunchaku): per-product error 2.4e-7 of sum |a*b| (lane-gather FMA reference 3.0e-7), 16-bit outputs within two units of roundoff of the reference, attention 2.1x and AWQ GEMM 1.6-2.1x faster than the lane-gather version.

## Lesson for cuda-to-rocm: torch 2.14.0+rocm10.2.0 gfx1100 wheels fail flash and memory-efficient SDPA with hipErrorInvalidImage

(For `diagnosis.md`, or the Linux host setup reference.) With the TheRock wheels `torch==2.14.0+rocm10.2.0` plus `amd-torch-device-gfx1100` from `stable.repo.amd.com/rocm/whl-next`, `torch.nn.functional.scaled_dot_product_attention` on a gfx1100 GPU fails with `CUDA error: device kernel image is invalid` (`hipErrorInvalidImage`) for the flash and memory-efficient backends in fp16, bf16 and fp32; the math backend works. The install has `libaotriton_v2.so` but no `aotriton.images` directory, and the index has no separate AOTriton package. Any HF text encoder using SDPA fails the same way, so it looks like a port failure in end-to-end tests. Check with a three-line SDPA call before blaming the port; to run tests anyway, put a `sitecustomize.py` on `PYTHONPATH` that calls `torch.backends.cuda.enable_flash_sdp(False)` and `enable_mem_efficient_sdp(False)` (subprocesses inherit it). Seen 2026-11-24 (nunchaku).

## Evidence run 2026-11-24, CDNA rerun of 8aa04e7 (linux-gfx942 fleet job, for jeff)

Not a porter round: no work lock taken, stage untouched, nothing changed in or pushed to the fork. Scope: the gfx942 item of "What remains before `ported`" in the RDNA round above (rerun of the shared-code change 8aa04e7 on CDNA). **Result: everything passes; 8aa04e7 changes no output on gfx942.** The gfx90a run of `v_mfma_i32_32x32x8i8` is still owed (this host is gfx942, whose path is `v_mfma_i32_32x32x16_i8`).

### Environment

- Host gfx942 (MI300X, `gfx942:sramecc+:xnack-`), the one GPU pinned by the dispatcher in `HIP_VISIBLE_DEVICES`, `ulimit -c 0`, every GPU run under `timeout`.
- Job venv `$FLEET_VENV` (the image's venv links rocm 10.1): `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel,device-gfx942]==10.2.0" "torch==2.14.0+rocm10.2.0" "torchvision==0.29.0+rocm10.2.0" "amd-torch-device-gfx942==2.14.0+rocm10.2.0"` (HIP 7.17.26434), then the test dependencies of the first round (accelerate 1.9.0, diffusers 0.36.0, transformers 4.55.2, peft 0.17.1, datasets 3.6.0, torchmetrics 1.8.2, huggingface-hub 0.34.6, controlnet-aux 0.0.10, image-gen-aux e08ae3d, pytest 8.4.2, numpy 2.4.6, pandas 3.1.1, scikit-learn 1.9.1, scikit-image 0.26.0, matplotlib). controlnet-aux pulls `opencv-python-headless` 5.1, which shadows the pinned cv2: it was uninstalled and `opencv-python==4.11.0.86` force-reinstalled. scikit-image had to be upgraded after numpy 2.4.6 (`numpy.dtype size changed` otherwise).
- Fork trees: three clones of https://github.com/AMD-Ecosystem/nunchaku at 8aa04e7 (reference), 8aa04e7 (default) and 5c32be5 (default), submodules at the recorded commits, `git status --porcelain --ignore-submodules=all` clean in each after every run (the dump scripts were copied in and removed again; the Sana example writes untracked PNGs).
- Builds, each in its own tree, `PATH=$FLEET_VENV/bin:$PATH PYTORCH_ROCM_ARCH=gfx942 MAX_JOBS=48 $FLEET_VENV/bin/python setup.py build_ext --inplace` (48 = half the exported MAX_JOBS=96, because two builds ran at once), reference with `NUNCHAKU_ROCM_MMA_REFERENCE=1`; run detached with `setsid nohup` from a script (a background shell is stopped at two hours), not wrapped in `timeit.sh`, so no compile telemetry. Default 8aa04e7: exit 0, 424 s. Default 5c32be5: exit 0, 425 s. Reference 8aa04e7: exit 0, **3442 s**, almost all of it the device link (lld code generation) of `gemm_w4a4_launch_bf16_int4`, on one core.
- Code objects (`llvm-objcopy --dump-section .hip_fatbin`, `clang-offload-bundler --unbundle --targets=hip-amdgcn-amd-amdhsa--gfx942`, `llvm-objdump -d --mcpu=gfx942`): both default builds have 10752 `v_mfma_i32_32x32x16_i8` in each INT4 launch file (fp16, bf16) and no other MFMA and no WMMA; the reference build has no MFMA and no WMMA in either (2.8M and 7.7M lines of disassembly against 2.6M and 3.9M). 5c32be5 vs 8aa04e7: the attention code object is byte-identical in its disassembly; the AWQ one is rescheduled (57050 vs 56998 lines) with identical outputs below; the INT4 launch files differ in 44 and 60 lines of 2.6M/3.9M.

### Results (all on the one gfx942 GPU)

- **`mma_dump.py compare` reference 8aa04e7 vs default 8aa04e7** (script in this file, unchanged): `34 cases, 10 outputs differ`, 64 outputs identical. The 10 are exactly the fp32 `lora_act_out` tensors (8 `fused_*[3]`, 2 `quantize_*[2]`), max relative difference 2.4e-7 of the largest value. Two runs of the reference build against each other differ in the same 10 tensors by up to 2.8e-7 (e.g. `fused_torch.float16_True_1024x3072x768[3]`: 24192/131072 bytes ref vs ref, 24508/131072 ref vs default), so this is the float-atomic nondeterminism, not the MFMA path. Every GEMM output, packed next-layer activation and scale is **bit-identical**: the gfx942 MFMA INT4 path at 8aa04e7 reproduces stage 1 exactly, as it did at 5c32be5.
- **`f16_dump.py compare` default 5c32be5 vs default 8aa04e7** (script in `notes-rdna-scripts.md`, unchanged): 90 outputs, **80 identical**, 10 differ and they are exactly the `lora_act_out` tensors (max rel 2.9e-7). All 8 attention_fp16 cases and all 8 AWQ GEMM cases are bit-identical, as are every GEMM output, packed activation and scale. Two runs of the 8aa04e7 build against each other: the same 10 tensors differ (max rel 3.1e-7), nothing else. The reference build vs default 8aa04e7 under `f16_dump.py` gives the same picture (80 identical, same 10 at max rel 2.9e-7). So on CDNA the paired `mma_m16n16k16_*` helpers are output-equivalent to the two m16n8k16 calls they replace.
- **`pytest tests/ops`**: 38 passed (8aa04e7 default, 9.3 s); 38 passed (5c32be5 default).
- **End to end on the 8aa04e7 default build** (in-place build on `PYTHONPATH`, not the editable install; the INT4 cached images were deleted first so this build regenerated them; the 16-bit SDXL references are the ones the first gfx942 round generated on this host with the same torch, copied from that job; the Z-Image references are upstream's; models from that job's Hugging Face cache, hard-linked, `HF_HUB_OFFLINE=1`): `HF_HOME=<job>/hf NUNCHAKU_TEST_CACHE_ROOT=<job>/test_cache timeout 7000 python -m pytest -v -s tests/ops tests/v1/sdxl/test_sdxl.py::test_sdxl_lpips tests/v1/sdxl/test_sdxl_turbo.py::test_sdxl_turbo_lpips tests/sana tests/v1/z_image` -> **48 passed** in 575 s. SDXL LPIPS 0.153 (expected 0.25), SDXL-Turbo 0.217 (0.25), Z-Image Turbo rank 32/128/256 0.380/0.371/0.370 (expected int4-bf16 0.40/0.38/0.37, limit x1.15), both Sana example scripts (image inspected: the ink-drawing panda). First round at 5c32be5 for comparison: 0.148, 0.212, 0.371/0.374/0.375.
- Speed (recorded, not gated; `speed.py` from `notes-rdna-scripts.md`, two alternating runs per build), ms, 5c32be5 -> 8aa04e7: W4A4 GEMM 4096x3072x3072 fp16 rank 32 3.79 -> 3.80, rank 128 6.21 -> 6.21; bf16 rank 32 3.58 -> 3.57, rank 128 6.15-6.28 -> 6.15; AWQ 512x4096x4096 fp16 6.39 -> 6.34, bf16 6.29 -> 6.23; attention_fp16 24 heads x 4608 tokens 59.0/64.4 -> 61.9/63.7 (identical code; the spread is run-to-run). No change on gfx942.

### What this leaves

The gfx942 rerun item of the RDNA round is done. Still owed from that list: gfx90a run of the MFMA path (`v_mfma_i32_32x32x8i8`, never run), the gfx12 proof, FP4 and CDNA f16/bf16 MFMA (questions 1 and 2), Windows gfx1151 last.

## Evidence run 2026-11-24, gfx90a MFMA INT4 path at 8aa04e7 (linux-gfx90a fleet job, for jeff)

Not a porter round: no work lock taken, stage untouched, nothing changed in or pushed to the fork. Scope: the gfx90a item of "What remains before `ported`" in the RDNA round above (first run of the gfx90a MFMA INT4 path, `v_mfma_i32_32x32x8i8`, plus the CDNA rerun of the shared-code change 8aa04e7). **Result: everything passes. The gfx90a MFMA INT4 path is bit-identical to stage 1 on every deterministic output, and 8aa04e7 changes no output on gfx90a.**

### Environment

- Host gfx90a (MI250X/MI250 GCD, `gfx90a:sramecc+:xnack-`), the one GPU pinned by the dispatcher in `HIP_VISIBLE_DEVICES`, `ulimit -c 0`, every GPU run under `timeout`.
- Job venv `$FLEET_VENV` (the image's venv links rocm 10.1): `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel,device-gfx90a]==10.2.0" "torch==2.14.0+rocm10.2.0" "torchvision==0.29.0+rocm10.2.0" "amd-torch-device-gfx90a==2.14.0+rocm10.2.0"` (HIP 7.17.26434); `torch` and `rocm_sdk` import from the venv, not the 10.1 links. Test dependencies: accelerate 1.9.0, diffusers 0.36.0, transformers 4.55.2, peft 0.17.0, datasets 3.6.0, torchmetrics 1.8.0, huggingface-hub 0.34.0, controlnet-aux 0.0.10, image-gen-aux e08ae3d, opencv-python 4.11.0.86 (`cv2.__version__` 4.11.0 imports, although controlnet-aux also pulled opencv-python-headless 5.1), pytest 8.4.2, numpy 2.4.6, pandas 3.1.1, scikit-learn 1.9.1, matplotlib 3.11.2, lpips 0.1.4. The first end-to-end attempt failed at collection with `ValueError: numpy.dtype size changed` from scikit-image 0.22 (via controlnet_aux); `pip install --upgrade scikit-image` (0.26.0) fixed it, as on gfx942.
- torch SDPA (flash, memory-efficient, math) works on gfx90a in this wheel set, so the gfx1100 workaround was not used.
- Fork trees: one clone of https://github.com/AMD-Ecosystem/nunchaku and three detached worktrees, 8aa04e7 (reference), 8aa04e7 (default), 5c32be5 (default), submodules at the recorded commits; `git status --porcelain --untracked-files=no` empty in each after every run (the dump scripts were copied in untracked; the Sana example writes untracked PNGs).
- Builds, each in its own tree, `PATH=$FLEET_VENV/bin:$PATH PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=<n> $FLEET_VENV/bin/python setup.py build_ext --inplace`, all three started together (n = 64 for the reference, 32 for each default, within the exported MAX_JOBS=128), reference with `NUNCHAKU_ROCM_MMA_REFERENCE=1` (the macro appears on its compile lines, on neither default build's); run detached with `setsid nohup` from a script (a background shell is stopped at two hours), not wrapped in `timeit.sh`, so no compile telemetry. Default 8aa04e7: exit 0, 643 s. Default 5c32be5: exit 0, 660 s. Reference 8aa04e7: exit 0, **5078 s**; its last job was `gemm_w4a4_launch_fp16_int4`, whose device link (lld code generation) ran alone on one core for over an hour at 1.6 GB RSS.
- Code objects (`llvm-objcopy --dump-section=.hip_fatbin=...`, `clang-offload-bundler --type=o --unbundle --targets=hipv4-amdgcn-amd-amdhsa--gfx90a`, `llvm-objdump -d --mcpu=gfx90a`): both default builds (5c32be5 and 8aa04e7) have **21504 `v_mfma_i32_32x32x8_i8`** in each INT4 launch file (fp16, bf16) and no other MFMA, no WMMA; attention and AWQ have no matrix instruction (stage 1 on CDNA, question 2); the reference build has no MFMA in either INT4 launch file.

### Results (all on the one gfx90a GPU)

- **`mma_dump.py compare` reference 8aa04e7 vs default 8aa04e7** (script in this file, unchanged, `PYTHONPATH=. python mma_dump.py dump <file>` from each tree root): `34 cases, 10 outputs differ`, 64 outputs identical. The 10 are exactly the fp32 `lora_act_out` tensors (8 `fused_*[3]`, 2 `quantize_*[2]`), max |diff| / max |value| 3.1e-7. Two runs of the default 8aa04e7 build against each other differ in the same 10 tensors by the same 3.1e-7 (e.g. `fused_torch.bfloat16_True_256x3072x3072[3]` 5851/32768 bytes ref vs default, 5900/32768 default vs default), so this is the float-atomic nondeterminism, not the MFMA path. Every GEMM output, packed next-layer activation and scale is **bit-identical**: the gfx90a MFMA INT4 path (`v_mfma_i32_32x32x8i8`, first run anywhere) reproduces stage 1 exactly.
- **`f16_dump.py compare` default 5c32be5 vs default 8aa04e7** (script in `notes-rdna-scripts.md`, unchanged): 90 outputs, **80 identical**, 10 differ and they are exactly the `lora_act_out` tensors (max rel 2.9e-7). All 8 attention_fp16 and all 8 AWQ GEMM cases are bit-identical, as are every GEMM output, packed activation and scale. The reference build vs default 8aa04e7 under `f16_dump.py` gives the same picture (80 identical, the same 10 at max rel 2.9e-7). So on gfx90a, as on gfx942, the paired `mma_m16n16k16_*` helpers are output-equivalent to the two m16n8k16 calls they replace.
- **`pytest tests/ops`** on the 8aa04e7 default build (`PYTHONPATH=. python -m pytest -q tests/ops`): **38 passed** in 22.8 s.
- **End to end on the 8aa04e7 default build** (in-place build on `PYTHONPATH`, not the editable install; fresh caches, so models were downloaded and the 16-bit SDXL references generated on this host by the tests; the Z-Image references are upstream's): `HF_HOME=<job>/hf NUNCHAKU_TEST_CACHE_ROOT=<job>/test_cache timeout 10800 python -m pytest -v -s tests/v1/sdxl/test_sdxl.py::test_sdxl_lpips tests/v1/sdxl/test_sdxl_turbo.py::test_sdxl_turbo_lpips tests/sana tests/v1/z_image` -> **10 passed** in 3246 s (most of it downloads and reference generation). SDXL LPIPS 0.154 (expected 0.25), SDXL-Turbo 0.218 (0.25), Z-Image Turbo rank 32/128/256 0.385/0.372/0.374 (expected int4-bf16 0.40/0.38/0.37, limit x1.15; rank 256 above the bare value, within the limit, as on gfx942 at 5c32be5), both Sana example scripts (image inspected: the ink-drawing panda). With the 38 of `tests/ops` that is the 48 of the first round's set.
- Not run, as in the earlier rounds: FLUX.1 and Qwen-Image tests (gated models, no Hugging Face token on this host), `test_device_id.py` (one GPU pinned), speed tests.

### What this leaves

The gfx90a item and the CDNA rerun are done on this arch. Still owed from the RDNA round's list: the gfx12 proof, FP4 and CDNA f16/bf16 MFMA (questions 1 and 2), Windows gfx1151 last.
## Porter round 2026-11-24, gfx12 proof (porter, linux-gfx1201 fleet job, for jeff)

Scope from jeff's dispatch: the "gfx12 proof" item of the RDNA round's list only (gfx12 operand and result layouts of `gemm_w4a4_block_wmma` iu4 and `mma_rocm::wmma_m16n16k16_f32` f16/bf16). No FP4, no CDNA f16/bf16 MFMA. Stage left at `porting`.

**Result: both gfx12 layouts are right. No fork change; head stays 8aa04e7.** No line that gfx11 or CDNA compiles was touched.

### Environment

- Host gfx1201 (AMD Radeon AI PRO R9700, 32 GB), `HIP_VISIBLE_DEVICES=0` as pinned, `ulimit -c 0`, every GPU run under `timeout`, MAX_JOBS 24 in total across concurrent builds.
- Job venv: `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel,device-gfx1201]==10.2.0" "torch==2.14.0+rocm10.2.0" "torchvision==0.29.0+rocm10.2.0" "amd-torch-device-gfx1201==2.14.0+rocm10.2.0"`, then **`amd-torch-device-gfx12-0==2.14.0+rocm10.2.0`** (the gfx12 family wheel, same index), then the test dependencies of the earlier rounds (accelerate 1.9, diffusers 0.36.0, transformers 4.55.2, peft 0.17, datasets 3.6, torchmetrics 1.8, huggingface-hub 0.34, controlnet-aux 0.0.10, image-gen-aux e08ae3d, opencv-python 4.11.0.86, pytest 8.4.2, numpy 2.4.6, pandas 3.1.1, matplotlib) and `--upgrade scikit-image scikit-learn scipy matplotlib PyWavelets` (the venv's copied scikit-image 0.2x was a numpy-1 build: `numpy.dtype size changed` at `controlnet_aux` import; now scikit-image 0.26.0, scikit-learn 1.9.1, scipy 1.17.1).
- **torch SDPA: the family wheel fixes it, no workaround needed.** With only the chip wheel `amd-torch-device-gfx1201`, `scaled_dot_product_attention` fails exactly as on gfx1100 (flash and memory-efficient backends, fp16/bf16/fp32: `CUDA error: device kernel image is invalid`; math works; `torch/lib` has `libaotriton_v2.so` but no `aotriton.images`). Installing `amd-torch-device-gfx12-0` adds `torch/lib/aotriton.images`, after which flash fp16/bf16 and memory-efficient fp16/bf16/fp32 all work (max error vs float64 0.0016 / 0.014 / 3.5e-6; flash has no fp32 kernel at all, "No available kernel", which is normal and the dispatcher falls through). The end-to-end runs below used no `sitecustomize`. The index lists `amd-torch-device-{gfx110x,gfx115x,gfx12-0}` family wheels; the gfx1100 round's workaround should be replaced by `amd-torch-device-gfx110x` there.
- **Trap: the first three extension builds used the image's ROCm 10.1 compiler, not 10.2.** The fleet venv reaches the image's `_rocm_sdk_core`, `_rocm_sdk_devel` and `_rocm_sdk_libraries` through `venv-base-links` on `sys.path`. pip installed the 10.2 `rocm-sdk-core`/`-libraries` payloads into the venv, but the 10.2 `rocm-sdk-devel` wheel ships its payload as `rocm_sdk_devel/_devel.tar` and expands it only when something asks for the devel root; until then `import _rocm_sdk_devel` finds the image's 10.1 copy, so `torch.utils.cpp_extension.ROCM_HOME` was `/opt/python-3.11-venv/.../_rocm_sdk_devel` and every compile ran its `bin/hipcc` (HIP 7.16.26385, AMD clang d6f6cb69). `$FLEET_VENV/bin/hipcc --version` printed 10.2 (HIP 7.17.26434, clang bdc35841) throughout, because that launcher comes from `rocm_sdk_core`, so it does not reveal the problem. Running `rocm-sdk path --root` expanded the tarball into the venv's `_rocm_sdk_devel` and `ROCM_HOME` moved there. Check with `python -c "import torch.utils.cpp_extension as c; print(c.ROCM_HOME)"` before building. The earlier gfx942 and gfx1100 rounds used the same venv recipe and may have built with 10.1 too (not checked from here). Everything below is reported for both compilers; they produce identical results.

### Builds (`PYTORCH_ROCM_ARCH=gfx1201`, `python setup.py build_ext --inplace` unless stated, separate trees)

| tree | compiler | flags | result |
|---|---|---|---|
| default 8aa04e7 | 10.1 | | exit 0, 277 s (8 jobs) |
| default 8aa04e7, documented `pip install -e . --no-build-isolation -v` | 10.1 | | exit 0, 273 s |
| reference 8aa04e7 | 10.1 | `NUNCHAKU_ROCM_MMA_REFERENCE=1` | exit 0, **4334 s** (16 jobs; the bf16 INT4 launch file alone took most of it in `ld.lld` LTO device code generation, 1.3 GB RSS) |
| default 8aa04e7, documented `pip install -e . --no-build-isolation -v` (installed build for every test below) | 10.2 | | exit 0, 286 s, `nunchaku-1.3.0.dev20261124+rocm10.2torch2.14`, 17 compiles through the venv's 10.2 `hipcc`, none through the image's |
| default 5c32be5 (iu4 WMMA, f16 still lane-gather) | 10.2 | | exit 0, 181 s |
| reference 8aa04e7 | 10.2 | `NUNCHAKU_ROCM_MMA_REFERENCE=1` | exit 0, **5903 s** (14 jobs; same long last file), no matrix instruction in the code objects |

The reference build was run detached (`setsid nohup <script> &`, a script that runs the build under `utils/timeit.sh` and appends `EXIT <code> <seconds>` to its log), because a harness background command is stopped at two hours. Code objects (`llvm-objcopy --dump-section=.hip_fatbin`, `clang-offload-bundler --unbundle`, `llvm-objdump -d --mcpu=gfx1201`): default 8aa04e7 has 21504 `v_wmma_i32_16x16x16_iu4` in each INT4 launch file plus `v_wmma_f32_16x16x16_f16` (fp16 INT4 launch 3888, attention 64, fasteri2f 64, AWQ 72) and `..._bf16` (bf16 INT4 launch 4064, AWQ 72); 5c32be5 has the 21504 iu4 and no f16/bf16 WMMA; the reference build has no matrix instruction at all.

### 1. f16/bf16 primitive test (`test.hip` from `notes-rdna-scripts.md`, unchanged)

Compiled with the venv's 10.2 `hipcc` exactly as `build.sh` does (`--offload-arch=gfx1201`, includes from the hipified `common_hip.h`/`utils_hip.cuh` of a built tree); the binary has 2 `v_wmma_f32_16x16x16_f16` and 1 `..._bf16`. `timeout 300 ./test_gfx1201` -> **PASS**: no output over the bound, against both the lane-gather reference and float64, in all 9 runs (3 modes x 3 scales, 2000 problems each). Max |err| / sum |terms|: f32-accumulator f16 WMMA 1.3e-7 to 1.7e-7 (reference 2.3e-7 to 3.0e-7); bf16 WMMA 1.1e-7 to 1.5e-7 (reference 1.2e-7 to 1.3e-7); f16 accumulator 99.97% of outputs bitwise equal to the reference, max error identical (4.6e-4, the f16 output rounding). **Exact small-integer problem: WMMA exact (0/256 inexact in every run)**, where gfx1100 gave 16/256 one fp32 ulp off. So the gfx12 float accumulator behaves like round-to-nearest on this case (the C-outside design is kept; it costs nothing on gfx12).

### 2. INT4 iu4 path (`gemm_w4a4_block_wmma`, gfx12 operand split and `row = v + 8 * half`): bit-identical

- `mma_dump.py compare` reference (10.1) vs default **5c32be5** (10.2): **34 cases, 10 outputs differ, exactly the fp32 `lora_act_out` tensors** (the 8 fused `[3]` and the 2 quantize `[2]`). Two runs of the reference build against each other differ in the same 10 (bytes of the same order, e.g. 5430/32768 vs 6381/32768). Every GEMM output, packed next-layer activation and output scale is bit-identical.
- Same with every build on ROCm 10.2: reference (10.2) vs default 5c32be5 (10.2): 34 cases, only the same 10 `lora_act_out` tensors differ. The 10.1 and 10.2 reference builds agree on everything except those 10, as do the 10.1 and 10.2 default builds of 8aa04e7.
- Note for whoever reruns this on another RDNA chip: `mma_dump.py compare` reference vs default **8aa04e7** is no longer bit-identical in the 16-bit GEMM outputs (here: 53 outputs differ, from 2 to 1043 bytes of up to 6.3 M, plus 1 to 2 packed nibbles in some fused cases), because 8aa04e7 also routes the LoRA-up and LoRA-down epilogues of these same cases through the f16/bf16 WMMA. Those are rounding flips covered by section 3, not iu4 layout errors (a wrong D layout puts whole columns in the wrong place). Bit identity of the iu4 path needs a build with the f16 WMMA off, which today means building 5c32be5. On CDNA, where there is no f16 WMMA, reference vs default 8aa04e7 should still be bit-identical. The RDNA round's instruction "only `lora_act_out` may differ" therefore applies to 5c32be5, not to 8aa04e7.

### 3. f16/bf16 path in the ops (`f16_dump.py` / `reldiff.py`, unchanged): within the recorded tolerance

Reference vs default 8aa04e7, for the pairs reference 10.1 / default 10.2, reference 10.1 / default 10.1 and reference 10.2 / default 10.2: the same numbers to every printed digit (only the fp32 `lora_act_out` maxima move, in the 7th digit, by run-to-run noise). Tolerance from the RDNA round: max |new - ref| / max |output| <= 2 units of roundoff of the output type (9.8e-4 float16, 7.8e-3 bfloat16). Worst per output kind:

| output | float16 | bfloat16 |
|---|---|---|
| W4A4 GEMM out (`gemm[0]`) | 3.05e-4 | 2.44e-3 |
| fused GEMM out (`fused[0]`) | 1.77e-4 | 3.26e-4 |
| fused output scales (`fused[2]`) | 1.15e-4 | identical |
| attention_fp16 | 5.98e-4 (4608 tokens) | 4.03e-3 (512 tokens, 500 valid) |
| AWQ GEMM | 5.18e-4 (1024x1024x4096) | 4.00e-3 (512x4096x4096) |

All within the bound. Packed next-layer activations: at most 2 nibbles differ per case, by one step. Quantize kernel outputs and scales: identical. fp32 `lora_act_out` of the fused cases: max rel 1.4e-4 (gfx1100: 1.4e-4), quantize `lora_act_out` 2.6e-7 (run-to-run noise 2.9e-7). Each build's own error against float64 is identical to three significant digits in every attention and AWQ case (e.g. attention fp16 4608 tokens 0.00239 both, AWQ bf16 512x4096x4096 0.00304 both). Determinism: two runs of the default 8aa04e7 build differ only in the 10 `lora_act_out` tensors (max rel 2.9e-7), with either compiler; the 10.1 and 10.2 default builds give identical outputs apart from those 10.

### 4. Tests on the installed 8aa04e7 build (10.2)

- `PYTHONPATH=. python -m pytest -q tests/ops` -> **38 passed** (also 38 with the 10.1 build).
- `HF_HOME=<job>/hf NUNCHAKU_TEST_CACHE_ROOT=<job>/test_cache python -m pytest -v -s tests/v1/sdxl/test_sdxl.py::test_sdxl_lpips tests/v1/sdxl/test_sdxl_turbo.py::test_sdxl_turbo_lpips tests/sana tests/v1/z_image` -> **10 passed** in 581 s, no SDPA workaround: SDXL LPIPS 0.156 (expected 0.25), SDXL-Turbo 0.231 (0.25), Z-Image Turbo rank 32/128/256 0.386/0.387/0.364 (expected int4-bf16 0.40/0.38/0.37, limit x1.15), both Sana example scripts (`sana1.6b-int4.png` inspected: correct ink-drawing panda). The cached INT4 SDXL images of the 10.1 run were moved aside first, so this build generated its own; the 16-bit SDXL and Z-Image reference images were generated on this host by the tests (`test_cache/fp16`, `test_cache/z-image-turbo/bf16`). The same set on the 10.1 build: 10 passed in 1227 s (including downloads), SDXL 0.151, SDXL-Turbo 0.224, Z-Image 0.386/0.363/0.358.
- Speed (recorded, not gated), R9700, ms, installed 10.2 build (`speed.py`): W4A4 GEMM 4096x3072x3072 with LoRA bf16 3.53 (rank 32) / 3.92 (rank 128), fp16 6.60 / 7.07; AWQ GEMM 512x4096x4096 fp16 2.66, bf16 2.89; attention_fp16 24 heads x 4608 tokens 53.7. The fp16 W4A4 GEMM is 1.9x slower than bf16 here (gfx1100: 1.5x), the register-pressure item the RDNA round recorded.

### What remains before `ported` (stage stays `porting`)

- gfx12 proof: **done** (surface `kernel:stage2-rdna-wmma-gfx12` moved from owed to covered).
- Done by the parallel jobs recorded above, merged while this round ran: the gfx942 rerun of 8aa04e7, the gfx90a MFMA run, and the nvcc compile of 8aa04e7 (all pass).
- **FP4** on gfx950/gfx1250 (question 1) and **CDNA f16/bf16 MFMA** (question 2): not written, waiting on jeff. These two answers, and whatever work they add, are the only porter items left from the RDNA round's list.
- The gfx942 and gfx90a evidence runs report HIP 7.17.26434 from `hipcc --version`, which does not show which devel tree torch compiled with (the venv trap above); if their `ROCM_HOME` was the image's, their results are still evidence for the same source, since here the 10.1 and 10.2 builds gave identical outputs.
- Windows gfx1151 last, per the standing ruling.
- Optional, for future RDNA validators: a build switch that turns off only the f16/bf16 WMMA would let the iu4 bit-identity check run on the current head instead of 5c32be5. Not added (the round's scope was proof only).

## Lesson for cuda-to-rocm: TheRock torch needs the GPU family wheel for flash and memory-efficient SDPA

(For `diagnosis.md` or the Linux host setup reference; corrects the gfx1100 lesson above, which stops at a workaround.) With `torch==2.14.0+rocm10.2.0` from `stable.repo.amd.com/rocm/whl-next` and only the per-chip wheel (`amd-torch-device-gfx1100`, `amd-torch-device-gfx1201`), `scaled_dot_product_attention` fails on the flash and memory-efficient backends with `hipErrorInvalidImage` ("device kernel image is invalid") because `torch/lib/aotriton.images` is missing. The AOTriton kernel images come in the GPU family wheel of the same version: `amd-torch-device-gfx110x` (RDNA3), `amd-torch-device-gfx115x`, `amd-torch-device-gfx12-0` (RDNA4). Installing it adds `aotriton.images` and both backends work (checked on gfx1201: flash fp16/bf16, memory-efficient fp16/bf16/fp32). Install the family wheel next to the chip wheel instead of disabling SDPA backends. Seen 2026-11-24 (nunchaku).

## Lesson for cuda-to-rocm: gfx12 WMMA f32 accumulation was exact where gfx11 was one ulp off

(For `faults-numerics.md`, next to the gfx11 WMMA accumulator note.) The same 16x16x16 `v_wmma_f32_16x16x16_{f16,bf16}` test with exactly representable small-integer data and a nonzero C gave 16/256 outputs one fp32 ulp off on gfx1100 and 0/256 on gfx1201 (nine runs, f16 and bf16), and on random data the gfx1201 WMMA error was at or below a lane-by-lane fp32 FMA chain (1.1e-7 to 1.7e-7 of sum |a*b|). Code that keeps C outside the gfx11 WMMA to avoid its accumulator bias can keep doing so on gfx12 at no accuracy cost, but gfx12 does not need it. Seen 2026-11-24 (nunchaku), ROCm 10.2.

## Lesson for moat-porting: in a fleet job venv, torch builds with the image's ROCm until the devel wheel is expanded

(For the fleet build recipes.) The job venv reaches the image's `_rocm_sdk_core`, `_rocm_sdk_devel` and `_rocm_sdk_libraries` through `venv-base-links`. Installing `rocm[devel]==<newer>` puts the newer core and libraries in the venv, but `rocm-sdk-devel` ships its payload as `rocm_sdk_devel/_devel.tar` and expands it only on first request, so until then `torch.utils.cpp_extension.ROCM_HOME` is the image's `_rocm_sdk_devel` and every extension compile uses the image's hipcc and headers, while `$FLEET_VENV/bin/hipcc --version` (a `rocm_sdk_core` launcher) still prints the newer version. Run `rocm-sdk path --root` once after installing (it expands the tarball into the venv), then check `python -c "import torch.utils.cpp_extension as c; print(c.ROCM_HOME)"` points into `$FLEET_VENV` before building. Seen 2026-11-24 (nunchaku gfx1201 job: three builds with the image's ROCm 10.1 before noticing).

## Porter round 2026-11-24, gfx950: CDNA f16/bf16 MFMA, NVFP4, gfx1250 (porter, linux-gfx950 fleet job, for jeff)

Scope: jeff's rulings of 2026-11-24 (heading above). Fork head 8aa04e7 -> **75e2528**, four new commits, pushed to `moat-port`: `5db74f4` CDNA f16/bf16 MFMA (Q2), `9ce05de` NVFP4 on gfx950 (Q1 b), `01c0f59` docs correction, `75e2528` gfx1250 native NVFP4 plus the gfx1250 build fix. Stage left at `porting` (what remains is at the end). Scripts new in this round: `notes-gfx950-scripts.md`.

### Environment

- Host gfx950 (MI355X, `gfx950:sramecc+:xnack-`), `HIP_VISIBLE_DEVICES=7` as pinned, `ulimit -c 0`, every GPU run under `timeout`, at most 128 build jobs in total across concurrent builds; disk 2.6 TB free at the start.
- Job venv: `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel,device-gfx950]==10.2.0" "torch==2.14.0+rocm10.2.0" "torchvision==0.29.0+rocm10.2.0" "amd-torch-device-gfx950==2.14.0+rocm10.2.0"`, then `rocm-sdk path --root` once. **Compiler actually used**: `torch.utils.cpp_extension.ROCM_HOME` = `$FLEET_VENV/lib/python3.11/site-packages/_rocm_sdk_devel`, and every device compile line in every build log of this round invokes `$FLEET_VENV/lib/python3.11/site-packages/_rocm_sdk_devel/bin/hipcc` (HIP 7.17.26434, ROCm 10.2), never the image's 10.1. The gfx950 chip wheel already ships `torch/lib/aotriton.images`; SDPA needed nothing. Test dependencies as in the earlier rounds (accelerate 1.9.0, diffusers 0.36.0, transformers 4.55.2, peft 0.17.1, datasets 3.6.0, torchmetrics 1.8.2, huggingface-hub 0.34.6, controlnet-aux 0.0.10, image-gen-aux e08ae3d, pytest 8.4.2, numpy 2.4.6, pandas 3.1.1, matplotlib, lpips; scikit-image/scikit-learn/scipy/PyWavelets upgraded; opencv-python-headless removed and opencv-python 4.11.0.86 force-reinstalled).
- Builds, `PYTORCH_ROCM_ARCH=<archs> MAX_JOBS=<n> python setup.py build_ext --inplace`, each in its own worktree, detached (`setsid -f`), wrapped in `timeit.sh` (records in `stats.jsonl`): 8aa04e7 default gfx950 380 s; 8aa04e7 `NUNCHAKU_ROCM_MMA_REFERENCE=1` gfx950 **1984 s** (much faster than on the other hosts); f16 MFMA tree (= 5db74f4) 366 s; FP4 tree (= 9ce05de) 403 s; 75e2528 for `gfx90a;gfx942;gfx950;gfx1100;gfx1201` 1391 s; 75e2528 for gfx1250 3091 s (INT4 on the reference path there); 8aa04e7 for `gfx1100;gfx1201` 461 s (for the RDNA comparison). All exit 0, no errors.

### 1. CDNA f16/bf16 MFMA (ruling Q2 b): fork 5db74f4

Design. The ruling asked for the RDNA design (paired n16 products, C outside the instruction, added in fp32). On CDNA one 64-lane wavefront holds two 32-lane warps with different operands, and the first design tried (16x16x16/16x16x32 MFMA with each warp feeding its own K lanes and zero rows for the other warp's results, 50% of the instruction used) passed the primitive test but was dropped: the other warp's B values multiply the zero rows, so a NaN or infinity in one warp (the attention kernel pads K with NaN) would turn the other warp's outputs into NaN. The committed path uses the **four-block 16x16x4 MFMA** (`v_mfma_f32_16x16x4_4b_{f16,bf16}` on gfx942/gfx950, `v_mfma_f32_16x16x4f16` / `v_mfma_f32_16x16x4bf16_1k` on gfx90a): block b reads its operands only from lanes 16b..16b+15, so blocks 0-1 are the lower warp's and 2-3 the upper warp's, nothing mixes. Each warp's two blocks take different halves of K (two instructions per 16x16x16 product, all lanes useful) and are added; operands move with 4 width-32 shuffles each (the rotation of the RDNA round), the result returns with 8 wavefront-wide shuffles (it is spread over all 64 lanes). C is added in fp32 afterwards. Precondition, met by every caller: all 64 lanes execute the helper together (the kernels branch only on block-uniform values; the 32-thread quantize kernels never reach it). Code: `mma_rocm.cuh` (`operand_words`, `mfma_16x16x16`, `mfma_result_fragments`, `mfma_m16n16k16_f32`), selected by `NUNCHAKU_ROCM_MFMA_F16` (gfx90a/gfx942/gfx950, not with `NUNCHAKU_ROCM_MMA_REFERENCE`).

Proof on gfx950:
- Primitive test (`test_cdna.hip`, 2000 random problems per mode and scale, two warps per wavefront with different problems): **PASS**; max |err| / sum |a*b|: f32-accumulator f16 1.3e-7 to 1.5e-7 (lane-gather reference 2.3e-7 to 3.0e-7), bf16 1.1e-7 to 1.2e-7 (reference 1.2e-7 to 1.3e-7); f16 accumulator 99.96% bitwise equal to the reference, same max error; exact small-integer problem **exact** (0/256). With `NANTEST=1` (NaN/inf in the upper warp only) the lower warp's results are all within the bound: PASS. Compiled for gfx942 and gfx90a too (code objects: 4 f16 and 2 bf16 MFMA each, the forms named above).
- `f16_dump.py compare` 8aa04e7 default (and the 8aa04e7 reference build, identical for these outputs on CDNA) vs 5db74f4: bit identity not expected; tolerance and its justification from the RDNA round (max |new - ref| / max |output| <= 2 units of roundoff of the output type: 9.8e-4 float16, 7.8e-3 bfloat16). Worst per output kind, float16 / bfloat16: W4A4 GEMM out 3.05e-4 / 2.44e-3, fused GEMM out 1.77e-4 / 3.26e-4, fused output scales 1.15e-4 / identical, attention_fp16 5.98e-4 (4608 tokens) / 4.03e-3 (512 tokens, 500 valid), AWQ GEMM 5.18e-4 / 4.00e-3: **all within the bound** (the same numbers as gfx1201's WMMA to three digits). Packed next-layer activations: at most 1 step on single nibbles; quantize kernel outputs identical; fp32 `lora_act_out` of the fused cases max rel 1.4e-4 (LoRA-down now through the MFMA). Each build's error against float64 is identical to three digits in every attention and AWQ case (e.g. attention fp16 4608 tokens 0.00239 both, AWQ bf16 512x4096x4096 0.00304 both).
- Determinism: two runs of 5db74f4 differ only in the 10 fp32 `lora_act_out` tensors (max rel 2.9e-7, the float-atomic noise every build has). Everything else bit-identical.
- `mma_dump.py`: reference 8aa04e7 vs default 8aa04e7 on gfx950: 34 cases, only the 10 `lora_act_out` differ (max rel ~3e-7, same as two reference runs): **the gfx950 INT4 MFMA path (`v_mfma_i32_32x32x16_i8`) is bit-identical to stage 1** (its first run on gfx950). 8aa04e7 vs 5db74f4: 53 outputs differ, because the LoRA-up/down epilogues of the same GEMMs now run on the MFMA: GEMM outputs at most 1 ulp (fused cases up to 8 ulp at a small value), scales 2 ulp, 1 nibble; the INT4 main loop itself is unchanged and stays proven. Those differences are the rounding covered by the f16 tolerance above.
- `pytest tests/ops`: 38 passed (5db74f4).
- End to end on 5db74f4 (in-place build; references generated on this host; Z-Image references upstream's): `pytest -v -s tests/v1/sdxl/test_sdxl.py::test_sdxl_lpips tests/v1/sdxl/test_sdxl_turbo.py::test_sdxl_turbo_lpips tests/sana tests/v1/z_image` -> **10 passed** in 671 s: SDXL LPIPS 0.153 (expected 0.25), SDXL-Turbo 0.216 (0.25), Z-Image Turbo rank 32/128/256 0.383/0.348/0.379 (expected int4-bf16 0.40/0.38/0.37, limit x1.15), both Sana example scripts.
- Speed, ms, MI355X, 8aa04e7 -> 5db74f4 (`speed.py`, two alternating runs, ranges): W4A4 GEMM 4096x3072x3072 fp16 rank 32 2.20-2.27 -> 2.15-2.31, rank 128 3.63-3.71 -> 2.37-2.53; bf16 rank 32 2.84-3.05 -> 2.53-2.72, rank 128 4.22-4.43 -> 3.05-3.25; AWQ 512x4096x4096 fp16 4.85-4.91 -> 3.84, bf16 4.65-4.73 -> 3.17; attention_fp16 24 heads x 4608 tokens 37.8 -> 22.5-22.8 (1.67x).

### 2. NVFP4 on gfx950 (ruling Q1 b): fork 9ce05de (+ docs 01c0f59)

Design: `compute_fp4` on ROCm decodes each e2m1 value times its ue4m3 block scale (one per 16 values of K) into f16 in registers and runs the f16 MFMA of section 1. The decoded value has at most 6 significant bits and lies between 2^-10 and 2688, so it is exact in f16 and the products are exact in fp32: only the fp32 summation order differs from the block-scaled mma.sync. No data movement is needed for the fragments: product c (0..3) of an m16n8k64 takes elements 2c and 2c+1 of every 8-element register of A and B, which hold the same k in A and B, so the four m16n8k16 products sum to the FP4 product; each row's and column's scale word is fetched from the lane that holds it (A: lane 4g + 2*ida + (row >= 8); B: lane 4g + 2*idb + n8-tile), byte b = K block b. The A operands are converted to the MFMA layout once per k step, the four products of each output tile summed in the MFMA accumulator and added to psum once. Decode: the three magnitude bits go to f16 bits 9-11 and the sign to bit 15 (value x 2^-14), times the scale (e4m3 bits << 7 = scale x 2^-8, times 256), times 2^14, each multiply exact; scale 0x7f gives NaN. The FP4 activation quantizer needed float -> e2m1 and float -> e4m3 with round-to-nearest-even and saturation (`cvt.rn.satfinite` on NVIDIA): written out in C++ (`gemm_utils.cuh`), not through the gfx950 conversion instructions, because their saturation/NaN behaviour would have needed its own proof; NaN input to e2m1 gives a signed zero (NVIDIA's result for NaN is not documented; the group's scale is NaN then anyway). `FP4_AVAILABLE` is true on gfx950 (`NUNCHAKU_ROCM_FP4`), so the FP4 GEMM kernels and the FP4 quantize epilogue no longer trap there; every other AMD target still traps (Q1 = c). In reference builds gfx950 FP4 uses the same decode with the lane-gather f16 product. The CUDA `mma_fp4` is now compiled only on CUDA (its ROCm trap arm became dead).

Tests (new, `tests/ops/test_gemm_w4a4_fp4.py`, skip on GPUs without NVFP4 kernels): FP4 GEMM vs float64 (3 shapes x fp16/bf16, e2m1 extremes, random e4m3 scales 2^-6..4 plus 448, 2^-9 and 0, alpha, per-channel scales; activation and activation-scale layouts packed in Python from the layout the quantize kernel writes, weights by the checkpoint packer); FP4 activation quantization (2 shapes x fp16/bf16, rows from 2^-12 to 2^6 so scales cover e4m3 subnormals and zero): scales within one e4m3 step of max|v|/6 and every dequantized value within the e2m1-plus-scale rounding bound; NVFP4 `SVDQW4A4Linear` end to end against float64 with simulated activation quantization. On 9ce05de: **`pytest tests/ops` 52 passed** (38 + 14). GEMM max error at the fp16 output rounding (3.2e-4 relative). Two test-design errors were fixed before the commit (my first version let the extreme scales overflow fp16 output, and its quantize bound ignored e4m3 subnormal and saturating scales); the kernels were not changed for them.

End to end (Q3 measurement, public non-gated NVFP4 checkpoints the tests use: `nunchaku-tech/nunchaku-z-image-turbo` `svdq-fp4_r32` and `svdq-fp4_r128`; FLUX.1-schnell also has `svdq-fp4_r32` but its base pipeline was not run here): the unchanged Z-Image test with `-p force_fp4` (`notes-gfx950-scripts.md`), 9ce05de build: `pytest -v -s -p force_fp4 tests/v1/z_image` -> **6 passed**, FP4 rank 32 LPIPS **0.349** (expected fp4-bf16 0.33, limit x1.15), rank 128 **0.293** (expected 0.30); rank 256 has no FP4 checkpoint (the test returns early).

### Q3 on gfx950: FP4 against INT4 (Z-Image Turbo, 1024x1024, 9 steps, bf16)

| | LPIPS r32 | LPIPS r128 | s/image r32 | s/image r128 |
|---|---|---|---|---|
| INT4 (5db74f4 / 9ce05de builds) | 0.383 | 0.348 | 14.6 (first run), 15.5 | 17.1, 17.8 |
| NVFP4 (9ce05de) | 0.349 | 0.293 | 21.7-23.6 | 23.9-24.4 |

(`zimage_speed.py`, 3 timed images after a warm-up each; the transformer is 99% of the time; the two INT4 numbers are two separate runs.) Kernel level (`fp4_speed.py`, bf16, LoRA rank 32): W4A4 GEMM 4096x3072x3072 INT4 2.63 ms, FP4 4.65 ms (1.77x); 4096x12288x3072 10.1 / 15.6 (1.54x); 4096x3072x12288 9.9 / 18.2 (1.83x). **FP4 is better in LPIPS and 1.35-1.5x slower per image, so by the ruling `get_precision("auto")` stays `int4` on gfx950.** The speed gap is the decode plus the half-rate 16x16x4 MFMA; a faster FP4 path is a performance item.

Finding for jeff (no change made): the v1 model loaders (`NunchakuZImageTransformer2DModel.from_pretrained`, `transformer_zimage.py:403`, and the other v1 models the same way) take the precision from `get_precision()` with no argument and ignore the checkpoint's `quantization_config`. With `auto` = `int4` on AMD, an NVFP4 checkpoint therefore cannot be loaded through the public API on gfx950 (the load fails with shape mismatches); the tests above needed the plugin. 01c0f59 corrects the installation guide, which 9ce05de had made say "load them explicitly". Whether to offer a way to choose FP4 on AMD (an environment variable, or honouring the checkpoint) is jeff's call; nothing was added.

### 3. gfx1250: fork 75e2528 (compiled, **unproven**)

What the sources say about the scale16 WMMA:
- LLVM (llvm-project main 098ffdbd, 2026-11-24): `__builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(fmt_a, v16i32 a, fmt_b, v16i32 b, c_mod, v8f32 c, scale_sel_a, scale_fmt_a, i64 scale_a, scale_sel_b, scale_fmt_b, i64 scale_b, reuse_a, reuse_b)` (`BuiltinsAMDGPU.td`, feature `wmma-n16-insts`, wave32) and `..._32x16x128_f4`; `SIDefines.h` `MATRIX_SCALE_ROW0/ROW1`, `MATRIX_SCALE_FMT_E8/E5M3/E4M3 = 0/1/2`; clang shortens the v16i32 operand to the format's size (8 dwords for FP4, `builtins-amdgcn-gfx1250-wmma-w32.cl`). The LLVM docs say only "64-bit (16-bit per element) scale exponents"; they do not give the lane layout.
- MLIR `amdgpu.scaled_wmma` (`AMDGPUOps.td`): block size 16 or 32, scale element types `f8E8M0FNU` or `f8E4M3FN`; the scale16 form carries 8 scales per lane (64 bits) and `first_scale_lane` 0/16 picks the lane half (16 lanes per scale VGPR for a 16-row operand).
- FlyDSL (ROCm/FlyDSL 47d04a12, `lib/Dialect/FlyROCDL/GFX1250/MmaAtomScale.cpp`, `tests/unit/test_gfx1250_wmma_scale_layout.py`, which cite the CDNA5 ISA section 7.12.6.1 and say the layouts were corrected on hardware): FP4 operand lane l holds row l % 16 with K = 32 * (l / 16) + v % 32 + 64 * (v / 32); byte j of the scale word is the scale of K block j of that lane's row (8 blocks of 16 for scale16); result register v of lane l is row 8 * (l / 16) + v, column l % 16.
- aiter PR #8289 (ROCm/aiter, "gfx1250 pa_decode: NVFP4 K/V on a wave-owned scaled-WMMA tile"): runs NVFP4 data ("packed e2m1 with one E4M3 scale per 16 values", scale tensors `float8_e4m3fn`) directly on `v_wmma_scale16_f32_16x16x128_f8f6f4` on an MI455X, tests passing there; it also reports a hardware erratum: the instruction reads its 64-bit scale operands correctly only from VGPRs, not from SGPRs or inline constants.
- Conclusion: one E4M3 scale per 16 consecutive K per row and per column is NVFP4's block and scale format. NVFP4 scales are positive, so E4M3 (signed) and ue4m3 agree on every value used. **So gfx1250 runs NVFP4 natively.** Not established from a primary document: which lanes hold the scales (taken as lanes 0-15 = rows 0-15 with selector ROW0, from the MLIR description and FlyDSL's use of the selector), and nibble order within a word (low nibble first, as everywhere else). No ISA document was available to me.

Code (75e2528): `compute_fp4_wmma` in `gemm_w4a4.cuh` (`NUNCHAKU_ROCM_WMMA_FP4`, gfx1250, not in reference builds): the instruction takes 128 values of K, both pipeline stages, so the FP4 main loop issues it once per pair of stages (at stage 0, right after the stage-1 load). Operands come from the mma.sync FP4 fragments with the same four-step shuffle as the CDNA path (now shared as `mma_rocm::operand_words`); the scale words are fetched from their mma.sync lanes into a 64-bit pair (step 0 in the low word = K blocks 0-3, step 1 in the high word = blocks 4-7); the result returns to the accumulator layout with eight shuffles. A reference build on gfx1250 uses the decode path of section 2 with the lane-gather f16 product.

Build fix in the same commit: **8aa04e7 did not compile for gfx1250 at all.** gfx1250 defines `__GFX12__`, but it has no `wmma-128b-insts` (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12 needs target feature wmma-128b-insts`). The RDNA macros now exclude `__gfx1250__`, so gfx1250 takes the lane-gather reference for INT4 and the f16/bf16 products. `setup.py`'s `NUNCHAKU_INSTALL_MODE=ALL` list does not include gfx1250 (left so: it would add about 50 minutes of reference-path INT4 compile to every ALL build); a gfx1250 machine gets it from the visible GPU or `PYTORCH_ROCM_ARCH`.

Code objects (75e2528, `llvm-objdump --mcpu=<gfx>`):
- gfx1250 build (3091 s, exit 0): each FP4 launch file has **2688 `v_wmma_scale16_f32_16x16x128_f8f6f4`**, encoded `matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4 matrix_a_scale_fmt:MATRIX_SCALE_FMT_E4M3 matrix_b_scale_fmt:MATRIX_SCALE_FMT_E4M3`, 8-VGPR A and B operands, and both scale operands in VGPR pairs (none from `s[...]`, so not hit by the erratum above). No other matrix instruction anywhere in the gfx1250 objects (INT4, attention and AWQ on the reference path). The gfx1250 code spills heavily (537,908 scratch instructions in the bf16 FP4 launch object's disassembly, from the reference-path kernels).
- The other targets: per object, fp16 INT4 launch / bf16 INT4 launch / attention / AWQ / bf16 FP4 launch: gfx90a 4672 `v_mfma_f32_16x16x4f16` + 21504 `v_mfma_i32_32x32x8_i8` / 7104 `..._16x16x4bf16_1k` + 21504 / 128 / 144 + 144 / 128 (trap kernels' LoRA only); gfx942 4672 `v_mfma_f32_16x16x4_4b_f16` + 10752 `v_mfma_i32_32x32x16_i8` / 7104 `_4b_bf16` + 10752 / 128 / 144 + 144 / 128; gfx950 the same as gfx942 except the bf16 FP4 launch: 21504 `_4b_f16` + 7104 `_4b_bf16`; gfx1100 3888 `v_wmma_f32_16x16x16_f16` + 21504 `v_wmma_i32_16x16x16_iu4` / 4400 `..._bf16` + 21504 / 64 / 72 + 72 / 64; gfx1201 the same as gfx1100 except 4320 bf16 in the bf16 INT4 launch: the RDNA counts are those of the earlier rounds.
- gfx950 on 75e2528 (all-target build): `pytest tests/ops` 52 passed; `f16_dump`/`mma_dump` against 5db74f4: only the 10 `lora_act_out` tensors differ (float-atomic noise), everything else bit-identical.

### Shared code that gfx11/gfx12 compile (RDNA should rerun)

Source changes RDNA compiles: `mma_rocm.cuh` (`select4` moved out of the WMMA block into a shared one, unchanged; the RDNA macro gained `!defined(__gfx1250__)`), `gemm_w4a4.cuh` (ROCm `compute_fp4` and the gfx1250 code are not instantiated on RDNA, where FP4 still traps; lines moved), `gemm_utils.cuh` (the FP4 conversions; not reached on RDNA). Result: gfx1100 and gfx1201 code objects of 75e2528 against 8aa04e7 (`rdnabase` build, same compiler and flags): every object identical in disassembly except the two INT4 launch files and `gemm_w4a4_test`, and there everything is identical except PC-relative string offsets (assert/printf strings moved with the line numbers) and **10 INT4 GEMM kernels whose instruction schedule differs** (2 of 219 functions in the gfx1100 fp16 INT4 launch file, 8 of 199 in the gfx1201 bf16 one; up to 621 changed lines in a 22,000-line function; same instruction count within a few, branch offsets moved). No operation was added or removed by the source change; this looks like scheduling and register allocation reacting to changed constants. The outputs should be unchanged, but **this has not been run on RDNA**: a gfx1100 or gfx1201 host should rerun `mma_dump.py` against its reference build (identity expected in the INT4 outputs only with 5c32be5-style f16-off builds, see the gfx12 round's note) and `f16_dump.py` between 8aa04e7 and 75e2528 default builds (expected: only `lora_act_out` differs), `pytest tests/ops` (52 expected: the 14 FP4 tests skip there, so 38 pass and 14 skip) and the end-to-end set.

### What remains before `ported` (stage stays `porting`)

- **gfx942 and gfx90a runs of the CDNA f16 MFMA path** at 75e2528 (surface `kernel:stage2-cdna-mfma-f16-runs`): primitive test `test_cdna.hip` with `--offload-arch=gfx942` / `gfx90a`, `f16_dump.py`/`reldiff.py` reference vs default (tolerance above), `mma_dump.py` reference vs default (only the GEMM outputs touched by the LoRA epilogue may differ, within the f16 tolerance; on 8aa04e7 the INT4 path was already proven there), determinism (two runs), `pytest tests/ops` (38 pass, 14 FP4 skip), end-to-end set, speed. gfx942 uses the same `_4b` instructions as gfx950; gfx90a its own encodings.
- **gfx1250 run** (surface `kernel:gemm_w4a4-nvfp4-gfx1250`): `PYTORCH_ROCM_ARCH=gfx1250 pip install -e . --no-build-isolation` (about 50 minutes here), `pytest tests/ops` (52 expected, the FP4 tests on the native WMMA), then the Z-Image FP4 test with `-p force_fp4` and the INT4 set. If the scale lane or nibble assumptions above are wrong, `test_gemm_w4a4_fp4` fails outright. A gfx1250 matrix-instruction path for INT4 and f16 (surface `kernel:stage2-gfx1250`) is not written; gfx1250 is not a required gate, so whether it is owed is jeff's call.
- **wave32 reruns** at 75e2528 on gfx1100 and gfx1201 (previous section).
- **nvcc recheck** of 75e2528 on the mi355 host: the CUDA path should be untouched (every change is under `USE_ROCM` or the `NUNCHAKU_ROCM_*` macros, `mma_fp4` moved under `#ifndef USE_ROCM` with its body unchanged, the `FP4_AVAILABLE` condition gained `|| defined(NUNCHAKU_ROCM_FP4)`, false on CUDA); the new `tests/ops/test_gemm_w4a4_fp4.py` runs on sm_120/121 and was not run on NVIDIA.
- Commit-gate note: `utils/check.py` flags `01c0f59` ("Test Plan has no fenced command block"; a docs-only commit, already pushed, so not amended); fix at the PR-prep squash.
- Questions for jeff (nothing above waits on them): (1) a way to load NVFP4 checkpoints on AMD while `auto` stays `int4` (finding above); (2) whether a gfx1250 matrix-instruction path for INT4/f16 is owed.
- Windows gfx1151 last, per the standing ruling.

## Lesson for cuda-to-rocm: per-warp matrix products on a 64-lane wavefront without cross-warp contamination

(For `no-hip-equivalent.md`, the mma.sync reimplementation bullets, CDNA.) Code written for 32-lane warps keeps two independent warps in each CDNA wavefront, but a 16x16 or 32x32 MFMA reads all 64 lanes. Feeding each warp's K into its own lanes and zeroing the A rows whose results land in the other warp's lanes computes per-warp products at 50% use, but the other warp's B values multiply those zero rows, so a NaN or infinity in one warp (attention kernels often pad keys with NaN) makes the other warp's outputs NaN. The multi-block MFMAs avoid that: `v_mfma_f32_16x16x4_4b_{f16,bf16}` (gfx942/gfx950; `v_mfma_f32_16x16x4f16` and `..._16x16x4bf16_1k` on gfx90a) is four independent 16x16x4 products, block b reading its operands only from lanes 16b..16b+15, so blocks 0-1 belong to the lower warp and 2-3 to the upper one. Give each warp's two blocks different halves of K and add them; all lanes are useful. Each block's result is spread over the whole wavefront (register 4b + v of lane l = block b, row 4 * (l / 16) + v, column l % 16), so the move back to the per-warp layout uses wavefront-wide shuffles. Measured on gfx950 (nunchaku): per-product error 1.5e-7 of sum |a*b| (lane-gather FMA 3.0e-7), exact on exact data, NaNs in one warp never reach the other; attention 1.67x and AWQ GEMM 1.3-1.5x faster than the lane-gather version.

## Lesson for cuda-to-rocm: NVFP4 (e2m1 with e4m3 block-16 scales) on AMD GPUs

(For `no-hip-equivalent.md` or the library-gap reference, FP4.) NVIDIA's `mma ... kind::mxf4nvf4.block_scale.scale_vec::4X ... ue4m3` scales every 16 values of K with an e4m3 number. gfx950's scaled MFMA (`v_mfma_scale_f32_*_f8f6f4`) takes only power-of-two (e8m0) scales per 32 values, so it cannot apply them. An e2m1 value times an e4m3 scale is exact in f16 (at most 6 significant bits, 2^-10 to 2688), so decoding into f16 in registers and running the f16 matrix instruction gives the NVFP4 product with exact products and fp32 sums. The m16n8k64 FP4 fragment then needs no shuffles: elements 2c and 2c + 1 of every 8-element register of A and B form the c-th m16n8k16 f16 product, because A and B hold the same k at the same place. gfx1250 has `v_wmma_scale16_f32_16x16x128_f8f6f4` (and `..._32x16x128_f4`) with a 64-bit scale operand: eight E4M3 scales per row, one per 16 values of K. That is NVFP4's format, so gfx1250 runs NVFP4 natively; read the 64-bit scales from VGPRs (an MI455X erratum reported in ROCm/aiter #8289 says SGPR or constant scales are read wrongly). The `cvt.rn.satfinite` e2m1/e4m3 conversions of the activation quantizer are short C++ functions (comparisons for e2m1; mantissa rounding with carry plus a subnormal branch for e4m3). Seen 2026-11-24 (nunchaku): gfx950 measured; gfx1250 compiled only.

## Lesson for cuda-to-rocm: gfx1250 defines __GFX12__ but has none of the gfx12 16x16x16 WMMA instructions

(For `build-headers-macros.md`, target macros.) Code that selects the gfx12 WMMA forms with `#if defined(__GFX12__)` (`__builtin_amdgcn_wmma_f32_16x16x16_{f16,bf16}_w32_gfx12`, `..._i32_16x16x16_iu4_w32_gfx12`) fails to compile for gfx1250 with "needs target feature wmma-128b-insts": gfx1250 defines `__GFX12__` but has a different WMMA set (16x16x32 f16/bf16, 16x16x64 iu8, 16x16x128 fp8/f8f6f4). Exclude `__gfx1250__` from such guards and give gfx1250 its own path or a fallback. Seen with ROCm 10.2 (nunchaku).

## Lesson for moat-porting: timeit.sh runs the command from the MOAT checkout

(For the fleet build recipes / telemetry section.) `utils/timeit.sh` changes into the MOAT repository root before running the wrapped command, so `timeit.sh <name> test -- python -m pytest tests/...` looks for the tests in the MOAT checkout ("file or directory not found"), and `setup.py` runs in the wrong place. Wrap the command as `timeit.sh <name> <phase> -- bash -c 'cd "$0" && exec ...' <tree> <args>`. In a fleet venv, timeit may also not find a usable `python3` on PATH ("tried python3 and python on PATH"); `MOAT_PYTHON=/usr/bin/python3` fixes it. Seen 2026-11-24 (nunchaku gfx950 job).
