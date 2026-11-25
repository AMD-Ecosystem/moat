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

## CUDA compile check 2026-11-24, fork 75e2528 (mi355 fleet job, nvcc, no GPU used, for jeff)

Result: **PASS, no CUDA regression at fork `75e2528` against upstream base `302e0e97`.** Same 36 objects, 0 errors in both, the identical warning multiset, a byte-identical `build.ninja`, identical SASS for every device object except the two `__assertfail` line-number literals already seen at 5c32be5 and 8aa04e7, and host code that differs only through the `cudaFuncSetAttribute` cast of 9875a78 (already analyzed at 8aa04e7). The fork was checked at 75e2528 exactly; nothing in it was changed or pushed (`git status --porcelain` shows only untracked build output); stage (`porting`) and the lock (none held) were not touched. No GPU was used: none was assigned, the host has no NVIDIA GPU, `CUDA_VISIBLE_DEVICES=` was set for every command, and no AMD GPU was touched.

### Environment

- nvcc `/usr/local/cuda-12.8/bin/nvcc`: Cuda compilation tools, release 12.8, V12.8.93 (Build cuda_12.8.r12.8/compiler.35583870_0). Host compiler `c++` = g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0.
- Job venv `$FLEET_VENV` (Python 3.11): `$FLEET_VENV/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 torchvision==0.26.0+cu128` (`torch.version.cuda` 12.8, `torch.version.hip` None), then `diffusers==0.36.0 transformers==4.55.2 accelerate==1.9 peft==0.17 huggingface-hub==0.34 einops protobuf sentencepiece ninja setuptools wheel pytest==8.4.2`.
- Trees in the job directory: `git clone https://github.com/AMD-Ecosystem/nunchaku port; git -C port checkout 75e2528; git -C port worktree add ../base 302e0e97; git submodule update --init --recursive` in each. Submodule pointers identical at both commits (Block-Sparse-Attention 99511c34, cutlass a75b4ac4, json 63258397, mio 8b6b7d87, spdlog 27cb4c76).

### Build (identical for both trees, run in parallel, wrapped in `timeit.sh`, recorded in `stats.jsonl`)

```
export PATH=$FLEET_VENV/bin:/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8 CUDA_VISIBLE_DEVICES= NUNCHAKU_INSTALL_MODE=ALL MAX_JOBS=10
python setup.py build_ext --inplace -v
```

`NUNCHAKU_INSTALL_MODE=ALL` because FAST reads the visible GPUs and there are none; `Detected SM targets: ['75', '80', '86', '89', '120a']` in both, so the sm_75-only and the FP4 (sm_120a) kernel bodies were compiled too (every device object carries the five cubins, e.g. 865 functions in `gemm_w4a4_launch_bf16_fp4.o`). Base: exit 0 in 298 s; 75e2528: exit 0 in 295 s. `python setup.py --version` -> `1.3.0.dev20261124+cu12.8torch2.11` in both.

### Comparison

- **setup.py takes the CUDA branch unchanged on a CUDA torch**: `setup.py` has not changed since 8aa04e7, and `build/temp*/build.ninja` is byte-identical between the trees after substituting the tree path: 25 `cuda_compile` + 11 `compile` edges (Block-Sparse-Attention's 10 sources and `gemm_w8a8.cu` included, `flash_api_rocm.cpp` absent), same g++ and nvcc flags, `-gencode` for compute_75/80/86/89/120a, no `hip`/`rocm` string anywhere in it. `readelf -d` `NEEDED` of `_C*.so` identical (libc10, libtorch, libtorch_cpu, libtorch_python, libcudart.so.12, libc10_cuda, libstdc++, libm, libgcc_s, libc, ld-linux).
- **Objects, per file: PASS for all 36 in both trees**: `nunchaku/csrc/pybind`, `src/{FluxModel,Linear,Module,SanaModel,Serialization,activation,layernorm,interop/torch}`, `src/kernels/{activation_kernels,dwconv,gemm_batched,gemm_f16,layernorm_kernels,misc_kernels}`, `src/kernels/awq/{gemm_awq,gemv_awq}`, `src/kernels/zgemm/{attention,gemm_w4a4,gemm_w4a4_launch_{bf16_fp4,bf16_int4,fp16_fp4,fp16_int4,fp16_int4_fasteri2f},gemm_w4a4_test,gemm_w8a8}`, Block-Sparse-Attention `flash_api`, `flash_api_adapter` and its 8 `flash_fwd*_sm80.cu`.
- **Errors**: 0 lines matching `error` in either log.
- **Warnings**: identical multiset after normalizing paths and line numbers: `gemm_awq.cu` #177-D unused `blockIdx_x` (10), `blockIdx_z` (5), `kSmemCol` (5), `kSmemSizeZeros` (10), `total_global_iters` (10); `FluxModel.cpp` unused `device` (1); Block-Sparse-Attention `flash_api.cpp` unused `counter_offset`/`device_guard` (3 each). All pre-existing; no changed file gains a warning (`gemm_utils.cuh` and `gemm_w4a4.cuh`, the files 75e2528 changed since 8aa04e7, produce none).
- **SASS** (`cuobjdump -sass` per object, all five cubins, `nvdisasm` 12.8 from the venv's triton wheel; source-path `identifier` line dropped): 24 of the 25 device objects identical, including the five `gemm_w4a4_launch_*.o` (the two FP4 ones included), `gemm_w4a4.o`, `gemm_w4a4_test.o`, `attention.o`, `gemm_w8a8.o`, `gemm_awq.o`, `gemv_awq.o`. `layernorm_kernels.o` differs in the same 4 instructions as at 5c32be5 and 8aa04e7: `IMAD.MOV.U32 R8, RZ, RZ, 0x1da` -> `0x24f` (x3) and `0x1cc` -> `0x241`, the `__LINE__` arguments (474 -> 591, 460 -> 577) of the two `assert(false)` in `src/kernels/utils.cuh`. The CUDA device code is unchanged, so the sm_120a FP4 code (`mma_fp4`, `quantize_float2_fp4`, `quantize_float4_fp8`) is identical to upstream after `mma_fp4` moved under `#ifndef USE_ROCM`.
- **Host code** (`objdump -d -C` per object, compared per function with `tmpxft_*` names and call targets normalized; script `hostcmp.py`, below): 27 of 36 objects have identical function sets, and their only differing functions differ in `$imm` operands (`__LINE__` arguments of the error-check macros). The other 9 (`gemm_awq.o`, `attention.o`, the five `gemm_w4a4_launch_*.o`, `gemm_w4a4_test.o`, `gemm_w8a8.o`) lack the base's `cudaFuncSetAttribute<...>` template wrapper instantiations (86 per launch file, 4/2/2/1 in the others), the 8aa04e7 finding (9875a78's `reinterpret_cast<const void *>` makes the call sites call the C runtime function directly; `cuda_runtime.h`'s template wrapper does exactly that cast). The remaining instruction differences in those objects are RIP-relative `lea` offsets shifted by the removed wrappers (e.g. `0x2318f(%rip)` -> `0x2315d(%rip)`, a constant 0x32), dropped alignment `nop`s in the device stubs, and the call sites themselves. No other host change.
- **Shared C++ fixes** `a4abb15` (the `template` disambiguator in `attention.cu`, `epilogues.cuh`, `gemm_base.cuh`, `gemm_w4a4.cuh`, `gemm_w4a4_launch_impl.cuh`, `gemm_w4a4_test.cu`) and `6e23cc9` (`std::array<half2_t, 2>{...}` in `gemm_base.cuh`): both are in 75e2528 and compile under nvcc 12.8 and its g++ 13.3 host pass with no error or new warning; every object that includes them has SASS identical to the base.
- **What 75e2528 changed since 8aa04e7 in shared files** (read, then confirmed by the results above): `gemm_utils.cuh` adds `float_to_e2m1_satfinite`/`float_to_e4m3_satfinite` inside `#ifdef USE_ROCM` and changes only the ROCm arms of `quantize_float2_fp4`/`quantize_float4_fp8`; `gemm_w4a4.cuh` moves `mma_fp4` under `#ifndef USE_ROCM` (body unchanged), adds the ROCm `compute_fp4`/`compute_fp4_wmma` under `#ifdef USE_ROCM` with upstream's `compute_fp4` in the `#else`, selects `compute_fp4_wmma` in the main loop only under `NUNCHAKU_ROCM_WMMA_FP4`, and widens `FP4_AVAILABLE` with `|| defined(NUNCHAKU_ROCM_FP4)`. `NUNCHAKU_ROCM_FP4`/`_WMMA_FP4`/`_MFMA_*` are defined only in `mma_rocm.cuh` from `__gfx*__` macros, and that header is included only under `USE_ROCM` (`gemm_base.cuh:9`, `gemm_awq.cu:13`), so all are undefined on CUDA.

### Python (what a CUDA install takes)

- From `/tmp` with `PYTHONPATH=<tree>`: `import nunchaku, nunchaku._C, nunchaku.ops.gemm, nunchaku.ops.quantize, nunchaku.models, nunchaku.lora.flux.packer, nunchaku.utils` OK in both trees on torch 2.11.0+cu128, each with its own CUDA-built `_C`. Pybind surface (`dir()` of `_C.ops`, `_C.utils`, `_C.QuantizedFluxModel`, `_C.QuantizedSanaModel`): 101 names, identical hash in both. `get_rocm_arch()` returns None on CUDA. `py_compile` of `tests/ops/*.py` (including the new `test_gemm_w4a4_fp4.py`): OK.
- Behavior (`pycheck.py`, below): `nunchaku/utils.py` of each tree loaded by path (`python -I`), `torch.cuda.get_device_capability` faked to sm 7.5, 8.0, 8.6, 8.9, 9.0, 10.0, 12.0, 12.1 and `torch.cuda.get_device_properties` replaced by a function that raises (so a ROCm arm cannot be entered silently); per capability: `get_precision("auto")` for devices `"cuda"`, `"cuda:1"`, `torch.device("cuda", 0)` with no model and with int4/fp4 model names, `get_precision("int4"|"fp4")` with each model name, `is_turing()` on two devices, `check_hardware_compatibility` for int4 and `fp4_e2m1_all` configs on two devices, return values, warnings and log output captured: **168 results, base and 75e2528 byte-identical JSON** (sm75 int4 + `is_turing` True; sm80/86/89 int4, fp4 config rejected; sm120/121 `auto` = fp4, int4 config rejected; sm90/sm100 "Unsupported GPU architecture"). `get_rocm_arch` returned None for all three devices without calling `get_device_properties`.
- New test `tests/ops/test_gemm_w4a4_fp4.py` on a CUDA torch: `fp4_supported()` with faked capabilities is True for sm 12.0 and 12.1, False for 7.5, 8.9 and 9.0 (it checks `torch.cuda.is_available()` first, so a GPU-less CUDA install skips); `pytest -q tests/ops/test_gemm_w4a4_fp4.py` with a plugin faking `is_available` and sm 8.9: collects 14, **14 skipped** ("the GPU has no NVFP4 kernels"). It has not been run on an sm_120 GPU (none available).

### Scripts (in the job directory, not committed)

`build.sh` (the build above, per tree), `hostcmp.py` (per-function `objdump` comparison; `$imm`-only differences counted separately), `pycheck.py` (the faked-capability `utils.py` matrix), `plug/fakecap.py` (pytest plugin setting `torch.cuda.is_available` and `get_device_capability` from `FAKE_CAP`).

### CUDA fixes proposed

None needed.

## Evidence run 2026-11-24, gfx942 CDNA f16/bf16 MFMA path at 75e2528 (linux-gfx942 fleet job, for jeff)

Not a porter round: no work lock taken, stage untouched (`porting`), nothing changed in or pushed to the fork. Scope: the gfx942 half of the gfx950 round's first "What remains" item (surface `kernel:stage2-cdna-mfma-f16-runs`), plus the FP4 refusal on a non-FP4 arch. **Result: everything passes. The gfx942 f16/bf16 MFMA path (`v_mfma_f32_16x16x4_4b_{f16,bf16}`) is within the recorded tolerance of stage 1 and deterministic. The gfx942 numbers equal the gfx950 round's to three significant digits in every worst case. The FP4 kernels refuse cleanly. One speed regression (not gated): W4A4 bf16 rank 32, +17%.** gfx90a is still owed (its own encodings, `v_mfma_f32_16x16x4f16` / `..._16x16x4bf16_1k`).

### Environment

- Host gfx942 (MI300X, `gfx942:sramecc+:xnack-`), `HIP_VISIBLE_DEVICES=4` as pinned by the dispatcher (one GPU, kept), `ulimit -c 0`, every GPU run under `timeout`.
- Job venv: `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel,device-gfx942]==10.2.0" "torch==2.14.0+rocm10.2.0" "torchvision==0.29.0+rocm10.2.0" "amd-torch-device-gfx942==2.14.0+rocm10.2.0"`. Then `rocm-sdk path --root` once (it printed `$FLEET_VENV/lib/python3.11/site-packages/_rocm_sdk_devel`). `torch.utils.cpp_extension.ROCM_HOME` is that same directory.
- **Compiler actually invoked**: every device compile line in all three build logs (17 per build) runs `$FLEET_VENV/lib/python3.11/site-packages/_rocm_sdk_devel/bin/hipcc` (HIP 7.17.26434, AMD clang 24.0.0git bdc35841c97f), and none of them names the image's `/opt/python-3.11-venv` ROCm 10.1.
- Test dependencies as in the earlier rounds: accelerate 1.9.0, diffusers 0.36.0, transformers 4.55.2, peft 0.17.1, datasets 3.6.0, torchmetrics 1.8.2, huggingface-hub 0.34.6, controlnet-aux 0.0.10, image-gen-aux e08ae3d, pytest 8.4.2, numpy 2.4.6, pandas 3.1.1, matplotlib, lpips. scikit-image 0.26.0, scikit-learn, scipy and PyWavelets were upgraded, opencv-python-headless was uninstalled, and opencv-python 4.11.0.86 was force-reinstalled. Models: the first gfx942 round's Hugging Face cache, hard-linked (`HF_HUB_OFFLINE=1`). The 16-bit SDXL references are copied from that round (same host, same torch). The Z-Image references are upstream's. No INT4 images were cached, so this build generated its own.
- Fork trees: `git clone https://github.com/AMD-Ecosystem/nunchaku`, `git checkout 75e2528` (75e25288499066d5f635aac8500ff955d5da63a5) plus detached worktrees at 75e2528 (reference) and 8aa04e7 (default, the "before" build), with submodules at the recorded commits. `git status --porcelain --untracked-files=no --ignore-submodules=all` is empty in each after every run; scripts ran from outside the trees, and the Sana example and Z-Image test write untracked images.
- Builds: `PATH=$FLEET_VENV/bin:$PATH PYTORCH_ROCM_ARCH=gfx942 MAX_JOBS=<n> $FLEET_VENV/bin/python setup.py build_ext --inplace -v`, one tree each, all three started together. n = 48 for the reference build (`NUNCHAKU_ROCM_MMA_REFERENCE=1`, the macro on its compile lines only) and 24 for each default build, for 96 in total, the exported MAX_JOBS. They ran detached (`setsid nohup`) under `timeit.sh` (in `stats.jsonl`). Default 75e2528: exit 0, 471 s. Default 8aa04e7: exit 0, 419 s. Reference 75e2528: exit 0, **3479 s**, almost all of it the single-core device link of the bf16 INT4 launch file (ld.lld, 1.5 GB RSS). 0 errors in any log.
- Code objects (`llvm-objcopy --dump-section=.hip_fatbin`, `clang-offload-bundler --unbundle`, `llvm-objdump -d --mcpu=gfx942`). Default 75e2528, per object:
  - fp16 INT4 launch: 4672 `v_mfma_f32_16x16x4_4b_f16` + 10752 `v_mfma_i32_32x32x16_i8`.
  - bf16 INT4 launch: 7104 `..._4b_bf16` + 10752 `..._i8`.
  - attention: 128 `_4b_f16`.
  - AWQ: 144 + 144.
  - bf16 FP4 launch: 128 `_4b_bf16`; fp16 FP4 launch: 128 `_4b_f16` (only the LoRA in the trap kernels).

  These are the gfx942 counts the gfx950 round recorded from its all-target build. Default 8aa04e7: only the 10752 `..._i8` per INT4 launch file. Reference 75e2528: no MFMA or WMMA in any object (2.8M and 7.7M lines of disassembly for the fp16 and bf16 INT4 launch files).

### Results (all on the one gfx942 GPU)

- **Primitive test** (`test_cdna.hip` from `notes-gfx950-scripts.md`, unchanged except that it includes the hipified `mma_rocm_hip.cuh` of the built 75e2528 tree). It was compiled with the venv's 10.2 `hipcc --offload-arch=gfx942` and the build.sh flags, giving 4 `v_mfma_f32_16x16x4_4b_f16` + 2 `..._4b_bf16` in the binary. `timeout 300 ./test_gfx942` -> **PASS**: no output over the bound, against either the lane-gather reference or float64, in all 9 runs (2000 problems each). Max |err| / sum |terms|:
  - f32-accumulator f16: MFMA 1.31e-7 to 1.48e-7, reference 2.28e-7 to 3.04e-7.
  - bf16: MFMA 1.12e-7 to 1.20e-7, reference 1.24e-7 to 1.34e-7.
  - f16 accumulator: 99.96% bitwise equal to the reference (511815/512000), the same max error 4.6e-4.
  - Exact small-integer problem: **exact, 0/256 inexact** in every run.

  `NANTEST=1` (NaN and infinity in the upper warp's problems only) -> **PASS**: the lower warp's outputs are all within the bound, so there is no cross-warp contamination. These are the gfx950 numbers.
- **`f16_dump.py compare` / `reldiff.py`, reference 75e2528 vs default 75e2528** (scripts from `notes-rdna-scripts.md`, unchanged). Bit identity is not expected. The tolerance and its justification are the RDNA round's: max |new - ref| / max |output| <= 2 units of roundoff of the output type, 9.8e-4 for float16 and 7.8e-3 for bfloat16. 90 outputs: 21 identical, 69 differ, **all within the bound**. Worst per output kind:

| output | float16 | bfloat16 |
|---|---|---|
| W4A4 GEMM out (`gemm[0]`) | 3.05e-4 (4352x256x3072 unsigned) | 2.44e-3 (4352x256x3072 unsigned) |
| fused GEMM out (`fused[0]`) | 1.77e-4 | 3.26e-4 |
| fused output scales (`fused[2]`) | 1.15e-4 | identical |
| attention_fp16 | 5.98e-4 (4608 tokens) | 4.03e-3 (512 tokens, 500 valid) |
| AWQ GEMM | 5.18e-4 (1024x1024x4096) | 4.00e-3 (512x4096x4096) |

  More detail:
  - GEMM outputs: max 1 ulp element-wise (3 of 20 identical); fused outputs: up to 8 ulp at a small value.
  - Packed next-layer activations: at most 2 nibbles differ per case (of 0.8 to 3.1 million), by one step; 7 of 12 identical.
  - Output scales: max 2 ulp.
  - Quantize kernel outputs and scales: identical.
  - fp32 `lora_act_out`: fused cases max rel 1.4e-4 (the LoRA-down goes through the MFMA); quantize cases 2.75e-7 (noise).
  - Each build's own error against float64 is identical to three digits in all 16 attention and AWQ cases (e.g. attention fp16 4608 tokens 0.00239 both, AWQ bf16 512x4096x4096 0.00304 both).

  As a cross-check, the reference 75e2528 against the default **8aa04e7** build differs only in the 10 fp32 `lora_act_out` tensors (max rel 2.9e-7). So on CDNA the reference build is the 8aa04e7 behavior, and `reldiff` of default 8aa04e7 vs default 75e2528 gives the same table to every printed digit.
- **Determinism**: two runs of the default 75e2528 build. Under `f16_dump.py`, 80 of 90 outputs are identical and 10 differ, exactly the fp32 `lora_act_out` tensors (8 `fused[3]` + 2 `quantize[2]`, max rel 2.87e-7). `mma_dump.py` gives the same 10 (`34 cases, 10 outputs differ`). Two runs of the reference build also differ in the same 10 (the float-atomic noise every build has). So the MFMA f16 path is deterministic.
- **`mma_dump.py`, the INT4 GEMM outputs** (script in this file, unchanged):
  - **Reference 75e2528 vs default 75e2528: `34 cases, 53 outputs differ`** (21 identical). Which ones differ:
    - 17 of 20 `gemm[0]`, from 2 to 1031 bytes of up to 6.3M; the 3 identical are bf16 256x128x128 (both signs) and bf16 512x384x1024 signed.
    - all 12 fused GEMM outputs `fused[0]`.
    - 5 of 12 packed next-layer activations `fused[1]` (1 to 2 nibbles).
    - 5 of 12 output scales `fused[2]` (1 to 2 ulp).
    - all 12 fused `lora_act_out` `fused[3]`, including the 4 N=256 cases that are bit-stable run to run.
    - the 2 quantize `lora_act_out` `quantize[2]` (noise).

    Identical: all `quantize[0]` and `quantize[1]` outputs (no 16-bit product), and the remaining `fused[1]`/`fused[2]`. **This matches the gfx950 round's explanation exactly.** That round also had 53 differing outputs: its 8aa04e7 vs 5db74f4 comparison found GEMM outputs at most 1 ulp, fused up to 8 ulp at a small value, scales 2 ulp and single nibbles. The INT4 main loop is unchanged, but 75e2528 routes the LoRA-up and LoRA-down epilogues of these same GEMMs through the f16/bf16 MFMA. The differences are that path's rounding flips (within the f16 tolerance above), not INT4 errors: a wrong layout would move whole columns.
  - **Reference 75e2528 vs default 8aa04e7: only the 10 `lora_act_out` tensors differ.** The 8aa04e7 build has the same INT4 MFMA main loop (`v_mfma_i32_32x32x16_i8`, 10752 per INT4 launch file in both builds) with the LoRA epilogues still on stage 1, so this is the INT4 path's bit identity against the 75e2528 reference build. Every GEMM output, packed activation and scale matches.
- **`pytest tests/ops`**: default 75e2528 **38 passed, 14 skipped** in 10.3 s. The 14 skips are `test_gemm_w4a4_fp4.py`, "the GPU has no NVFP4 kernels". Reference 75e2528: 38 passed, 14 skipped.
- **FP4 refuses cleanly on gfx942** (`fp4_refuse.py`, below):
  - `get_precision("auto")` -> `int4`. `check_hardware_compatibility` accepts an int4 config and rejects `fp4_e2m1_all` with `ValueError: "fp4" quantization needs an AMD GPU with FP4 matrix instructions (gfx950, gfx1250); please use "int4" quantization on gfx942.`
  - A direct FP4 GEMM launch (`svdq_gemm_w4a4_cuda(..., fp4=True)`, 256x256x256 bf16) and a direct FP4 activation quantize (`svdq_quantize_w4a4_act_fuse_lora_cuda(..., fp4=True)`) each print `FP4 is not available on this device`. The queue aborts with `HSA_STATUS_ERROR_EXCEPTION ... code: 0x1016` naming `gemm_w4a4_fp4_kernel` / `quantize_w4a4_fuse_lora_kernel<false, true>`. The next `torch.cuda.synchronize()` raises `torch.AcceleratorError: CUDA error: unspecified launch failure` (0.39 s / 0.25 s after the launch), which Python catches; the process exits in about 6.4 s.
  - A fresh process uses the GPU normally afterwards. This is the designed trap (`trap_no_fp4`, as upstream on NVIDIA below sm_120); the FP4 launch objects contain only the trap kernels' LoRA MFMA.
- **End to end on the default 75e2528 build** (in-place build on `PYTHONPATH`): `HF_HOME=<job>/hf NUNCHAKU_TEST_CACHE_ROOT=<job>/test_cache HF_HUB_OFFLINE=1 timeout 7000 python -m pytest -v -s tests/v1/sdxl/test_sdxl.py::test_sdxl_lpips tests/v1/sdxl/test_sdxl_turbo.py::test_sdxl_turbo_lpips tests/sana tests/v1/z_image` -> **10 passed** in 569 s.
  - SDXL LPIPS 0.150 (expected 0.25).
  - SDXL-Turbo 0.226 (0.25).
  - Z-Image Turbo rank 32/128/256: 0.375/0.378/0.373 (expected int4-bf16 0.40/0.38/0.37, limit x1.15). Rank 256 is above the bare value and within the limit, as in earlier gfx942 rounds; the `_turing` variants return early for bf16.
  - Both Sana example scripts (`sana1.6b-int4.png` inspected: the ink-drawing panda).

  With the 38 of `tests/ops` this is the earlier rounds' 48-test set. Earlier gfx942 numbers: 0.148/0.212/0.371/0.374/0.375 at 5c32be5, 0.153/0.217/0.380/0.371/0.370 at 8aa04e7.
- Not run, as in the earlier rounds: FLUX.1 and Qwen-Image tests (gated models, no Hugging Face token), `test_device_id.py` (one GPU), speed tests.

### Speed (recorded, not gated)

MI300X, ms, `speed.py` (`notes-rdna-scripts.md`), two alternating runs per build, 8aa04e7 default -> 75e2528 default:

| kernel | 8aa04e7 | 75e2528 |
|---|---|---|
| W4A4 GEMM 4096x3072x3072 fp16 LoRA rank 32 | 3.80-3.82 | 3.84 |
| same, fp16 rank 128 | 6.21-6.22 | 4.21 |
| same, bf16 rank 32 | 3.56-3.57 | **4.18-4.20 (+17%)** |
| same, bf16 rank 128 | 6.11-6.17 | 5.11-5.12 |
| AWQ GEMM 512x4096x4096 fp16 | 6.31-6.32 | 4.97-4.98 |
| AWQ GEMM 512x4096x4096 bf16 | 6.23-6.24 | 3.81 |
| attention_fp16 24 heads x 4608 tokens | 60.8-61.2 | 39.9 (1.53x) |

The bf16 rank 32 W4A4 GEMM is slower on gfx942 in both runs, while gfx950 got faster there (2.84-3.05 -> 2.53-2.72). A possibly related observation, not shown to be the cause: across the 112 LoRA-up GEMM kernels in the bf16 INT4 launch object, the per-lane scratch (`.private_segment_fixed_size`) minimum rose from 1256 to 1568 bytes and the total from 182632 to 217704 (+19%). This is the same register-pressure picture the RDNA round recorded for its fp16 slowdown; it is a performance-pass item, and nothing was changed.

### Scripts

`mma_dump.py` is in this file, `f16_dump.py`, `reldiff.py` and `speed.py` are in `notes-rdna-scripts.md`, and `test_cdna.hip` is in `notes-gfx950-scripts.md`, all unchanged. New in this run, `fp4_refuse.py` (from the fork root, `PYTHONPATH=.`; no argument for the Python checks, `gemm` or `quant` for one FP4 launch in its own process):

```python
"""FP4 on a GPU without NVFP4 kernels: python-level checks, then (argv[1] = gemm|quant) one FP4 kernel launch."""
import sys, time
import torch
from nunchaku.utils import get_precision, check_hardware_compatibility, get_rocm_arch, ROCM_FP4_ARCHS

if len(sys.argv) == 1:
    print("arch", get_rocm_arch(), "ROCM_FP4_ARCHS", ROCM_FP4_ARCHS)
    print("get_precision('auto') ->", get_precision("auto"))
    for dt in ("int4", "fp4_e2m1_all"):
        try:
            check_hardware_compatibility({"weight": {"dtype": dt}})
            print(f"check_hardware_compatibility({dt}) -> accepted")
        except ValueError as e:
            print(f"check_hardware_compatibility({dt}) -> ValueError: {e}")
    sys.exit(0)

from nunchaku.lora.flux.packer import NunchakuWeightPacker
from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda
from nunchaku.ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda
from tests.ops.test_gemm_w4a4_fp4 import pack_act_fp4, pack_ascales_fp4, random_scales

dtype, m, n, k = torch.bfloat16, 256, 256, 256
g = torch.Generator().manual_seed(0)
packer = NunchakuWeightPacker(bits=4)
t = time.time()
if sys.argv[1] == "gemm":
    out = torch.empty(m, n, dtype=dtype, device="cuda")
    svdq_gemm_w4a4_cuda(
        act=pack_act_fp4(torch.randint(0, 16, (m, k), generator=g)).cuda(),
        wgt=packer.pack_weight(torch.randint(0, 16, (n, k), dtype=torch.int32, generator=g)).cuda(),
        out=out,
        ascales=pack_ascales_fp4(random_scales((m, k // 16), g)).cuda(),
        wscales=packer.pack_scale(random_scales((n, k // 16), g).to(dtype), group_size=16).cuda(),
        fp4=True, alpha=0.5,
        wcscales=packer.pack_scale(torch.ones(n, dtype=dtype), group_size=-1).cuda(),
        lora_act_in=torch.randn(m, 32, device="cuda") * 0.1,
        lora_up=packer.pack_lowrank_weight((torch.randn(n, 32, generator=g) * 0.1).to(dtype), down=False).cuda(),
    )
else:
    x = torch.randn(m, k, generator=g).to(dtype).cuda()
    svdq_quantize_w4a4_act_fuse_lora_cuda(
        x, fp4=True,
        lora_down=packer.pack_lowrank_weight((torch.randn(32, k, generator=g) * 0.1).to(dtype), down=True).cuda(),
        smooth=packer.pack_scale((torch.rand(k, generator=g) + 0.5).to(dtype), group_size=-1).cuda(),
    )
try:
    torch.cuda.synchronize()
    print("RETURNED without error", time.time() - t)
except Exception as e:
    print("EXCEPTION", type(e).__name__, str(e).splitlines()[0], f"after {time.time() - t:.2f} s")
```

### What this leaves

The gfx942 part of `kernel:stage2-cdna-mfma-f16-runs` is done. Still owed from the gfx950 round's list:
- the gfx90a run of the CDNA f16 MFMA path (its own encodings);
- the gfx1250 run;
- the wave32 reruns at 75e2528;
- Windows gfx1151 last.

The nvcc recheck of 75e2528 was merged while this ran (pass). Surface accounting was not edited here (evidence run, notes only).

## Evidence run 2026-11-24, gfx90a CDNA f16/bf16 MFMA path at 75e2528 (linux-gfx90a fleet job, for jeff)

Not a porter round: no work lock taken, stage untouched (`porting`), nothing changed in or pushed to the fork. Scope: the gfx90a half of the gfx950 round's first "What remains" item (surface `kernel:stage2-cdna-mfma-f16-runs`; gfx90a uses its own encodings `v_mfma_f32_16x16x4f16` / `v_mfma_f32_16x16x4bf16_1k`), plus the FP4 refusal on a non-FP4 arch. **Result: every op output is within the recorded tolerance of stage 1, the path is deterministic, `tests/ops` and the end-to-end set pass, and FP4 refuses cleanly. One failure: the standalone primitive test `test_cdna.hip` FAILS on gfx90a for f16 inputs, because gfx90a's f16 MFMA reads subnormal f16 inputs as zero (hardware behavior, smallest reproducer below). bf16 passes, and the failure disappears entirely when subnormal f16 inputs are flushed in the test data. In the ops this shows only as ulp gaps at near-zero outputs (up to 178 ulp at |x| < 1e-4); every worst-case tolerance number equals gfx942/gfx950's to three digits.** Nothing was fixed; whether gfx90a's f16 path should keep the MFMA, or whether the primitive test's bound should exempt subnormal inputs on gfx90a, is jeff's call.

### Environment

- Host gfx90a (MI250X/MI250 GCD, `gfx90a:sramecc+:xnack-`), `HIP_VISIBLE_DEVICES=0` as pinned by the dispatcher (one GPU, kept), `ulimit -c 0`, every GPU run under `timeout`.
- Job venv: `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel,device-gfx90a]==10.2.0" "torch==2.14.0+rocm10.2.0" "torchvision==0.29.0+rocm10.2.0" "amd-torch-device-gfx90a==2.14.0+rocm10.2.0"`, then `rocm-sdk path --root` once (it printed `$FLEET_VENV/lib/python3.11/site-packages/_rocm_sdk_devel`); `torch.utils.cpp_extension.ROCM_HOME` is that directory. **Compiler actually invoked**: the `build.ninja` of all three builds names `$FLEET_VENV/lib/python3.11/site-packages/_rocm_sdk_devel/bin/hipcc` (HIP 7.17.26434, AMD clang 24.0.0git bdc35841c97f) and nothing from the image's ROCm 10.1; the standalone HIP tests were compiled with the same `hipcc`. SDPA needed nothing on gfx90a. Test dependencies as in the earlier rounds: accelerate 1.9.0, diffusers 0.36.0, transformers 4.55.2, peft 0.17.1, datasets 3.6.0, torchmetrics 1.8.2, huggingface-hub 0.34.6, controlnet-aux 0.0.10, image-gen-aux e08ae3d, pytest 8.4.2, numpy 2.4.6, pandas 3.1.1, matplotlib, lpips 0.1.4; scikit-image (0.26.0), scikit-learn, scipy, PyWavelets upgraded; opencv-python-headless uninstalled, opencv-python 4.11.0.86 force-reinstalled.
- Fork trees: `git clone https://github.com/AMD-Ecosystem/nunchaku`, `git checkout 75e2528` (75e25288499066d5f635aac8500ff955d5da63a5) plus detached worktrees at 75e2528 (reference), 75e2528 (default) and 8aa04e7 (default, the "before" build), submodules at the recorded commits. `git status --porcelain --untracked-files=no` empty in every tree after every run (scripts ran from outside the trees). The first 75e2528 default build was stopped after a minute and restarted: an interrupted `git submodule update` had left `third_party/mio` half checked out (staged deletions); `git -C third_party/mio reset --hard 8b6b7d8`, `build/` removed, rebuilt.
- Builds: `PATH=$FLEET_VENV/bin:$PATH PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=<n> $FLEET_VENV/bin/python setup.py build_ext --inplace`, one tree each, detached (`setsid nohup`) under `timeit.sh` (in `stats.jsonl`); n = 64 for the reference (`NUNCHAKU_ROCM_MMA_REFERENCE=1`, on its compile lines only) and 32 for each default, 128 in total, the exported MAX_JOBS. Default 75e2528: exit 0, 706 s. Default 8aa04e7: exit 0, 645 s. Reference 75e2528: exit 0, **5047 s**, almost all of it the single-core device link of the last INT4 launch file (ld.lld, 1.5 GB RSS).
- Code objects (`llvm-objcopy --dump-section=.hip_fatbin`, `clang-offload-bundler --unbundle --targets=hipv4-amdgcn-amd-amdhsa--gfx90a`, `llvm-objdump -d --mcpu=gfx90a`). Default 75e2528: fp16 INT4 launch 4672 `v_mfma_f32_16x16x4f16` + 21504 `v_mfma_i32_32x32x8i8`; bf16 INT4 launch 7104 `v_mfma_f32_16x16x4bf16_1k` + 21504 `..._i8`; attention 128 f16; AWQ 144 + 144; fasteri2f 128 f16; fp16 / bf16 FP4 launch 128 f16 / 128 bf16 (the trap kernels' LoRA only). These are the gfx90a counts the gfx950 round recorded from its all-target build. Default 8aa04e7: only the 21504 `..._i8` per INT4 launch file. Reference 75e2528: no MFMA or WMMA in any object.
- Models: the earlier gfx90a evidence job's Hugging Face cache, hard-linked; the 16-bit SDXL references copied from that job (generated on gfx90a by the same tests with the same torch); the Z-Image references are upstream's (from that job's cache). No INT4 images were cached, so this build generated its own.

### Results (all on the one gfx90a GPU)

- **Primitive test `test_cdna.hip`: FAIL for f16 inputs, PASS for bf16** (unchanged from `notes-gfx950-scripts.md`, compiled with the venv's `hipcc --offload-arch=gfx90a` and the build.sh flags against the hipified headers of the built 75e2528 tree; the binary has 4 `v_mfma_f32_16x16x4f16` + 2 `v_mfma_f32_16x16x4bf16_1k`). `timeout 300 ./test_gfx90a` -> exit 1. Outputs over the bound (of 512000 per run) and max |err| / sum |terms| of the MFMA:
  - f32-accumulator f16 (mode 0): scale 1: **404** over, 2.2e-5; scale 0.01: **72280** over, 4.38e-3; scale 30: 0 over, 3.08e-7 (lane-gather reference 2.28e-7 to 3.04e-7, never over).
  - f16 accumulator (mode 2): scale 1: **6** over; scale 0.01: **38559** over, 3.86e-3 (reference 4.5e-4); scale 4: 0 over.
  - bf16 (mode 1): 0 over in all three, MFMA 1.12e-7 to 1.20e-7 (reference 1.24e-7 to 1.34e-7).
  - Exact small-integer problem: exact (0/256) in every run.

  The error grows as the data shrinks and vanishes at scale 30, the signature of subnormal inputs (f16 subnormals are below 6.1e-5). **Cause, shown with the smallest reproducer** (`denorm.hip` in `notes-gfx90a-scripts.md`, no nunchaku code): one `__builtin_amdgcn_mfma_f32_16x16x4f16` with every A = 2^-20 or 2^-24 (f16 subnormals), B = 1, C = 0 returns **0 in 1024/1024 outputs** instead of 4A; with A = 2^-14 (smallest normal) all 1024 are exact. `v_mfma_f32_16x16x4bf16_1k` likewise returns 0 for bf16 A = 2^-130 and is exact at 2^-126 (irrelevant at these magnitudes). This is documented MI200 behavior: PyTorch's numerical-accuracy notes ("Reduced Precision FP16 and BF16 GEMMs and Convolutions on AMD Instinct MI200 devices") say the FP16 and BF16 V_DOT2 and MFMA instructions flush input and output denormals to zero on MI200 and on no other supported AMD GPU; gfx942/gfx950 pass the same test. Here the MFMA outputs are fp32, so only the input flush matters. **With the subnormal f16 inputs zeroed in the test data** (`FTZ_INPUTS=1`, one-line change in `notes-gfx90a-scripts.md`), the same binary **PASSES** all 9 runs: f32-accumulator f16 MFMA 1.43e-7 to 1.49e-7 (reference 2.28e-7 to 3.04e-7), f16 accumulator 99.96% to 99.99% bitwise equal to the reference, exact problem exact; and `FTZ_INPUTS=1 NANTEST=1` (NaN/inf in the upper warp only) **PASSES**: no cross-warp contamination. So the gfx90a encoding, operand layout and result layout are right; the only deviation is the hardware flush.
- **`f16_dump.py compare` / `reldiff.py`, reference 75e2528 vs default 75e2528** (unchanged scripts). Tolerance and justification from the RDNA round: max |new - ref| / max |output| <= 9.8e-4 float16, 7.8e-3 bfloat16. 90 outputs: 20 identical, 70 differ, **all within the bound**. Worst per output kind, float16 / bfloat16: W4A4 GEMM out 3.05e-4 (4352x256x3072 signed) / 2.44e-3 (4352x256x3072 signed); fused GEMM out 1.77e-4 / 3.26e-4; fused output scales 1.23e-4 (gfx942/gfx950: 1.15e-4) / identical; attention_fp16 5.98e-4 (4608 tokens) / 4.03e-3 (512 tokens, 500 valid); AWQ GEMM 5.18e-4 (1024x1024x4096) / 4.00e-3 (512x4096x4096). Every number except the fp16 scales equals the gfx942 and gfx950 runs' to three digits. Each build's own error against float64 is identical to three digits in all 16 attention and AWQ cases (attention fp16 4608 tokens 0.00239 both, AWQ bf16 512x4096x4096 0.00304 both, AWQ fp16 1024x1024x4096 0.000405 both).
  - Where gfx90a differs from gfx942: float16 outputs have **larger ulp gaps at near-zero values**: GEMM out up to 178 ulp (gfx942: 1 ulp), fused out up to 89, attention up to 1888 and AWQ up to 2586 ulp, always at |x| below about 1e-3 of max |x| (`ulp_where.py`); where |x| >= 1% of max |x| the gaps are at most 1 ulp (GEMM), 2 (fused), 3 (scales), 19-68 (fp16 attention, 4608 tokens worst) and 6-8 (fp16 AWQ). bfloat16 GEMM outputs are at most 1 ulp (3 ulp fused, 8 ulp at a small value). Packed next-layer activations: at most 11 nibbles per case (of 0.8 to 3.1 million), one step. Quantize kernel outputs and scales: identical. fp32 `lora_act_out`: fused max rel 1.4e-4 (LoRA-down on the MFMA), quantize 1.2e-5 fp16 / 2.4e-7 bf16. The fp16 quantize value is the flush too (`quant_ftz.py`): the quantize kernel's LoRA-down also runs on the f16 MFMA, its `lora_down` input has 44 subnormal f16 values, and with those (and any in x) zeroed in both builds the difference drops to 3.1e-7, the run-to-run noise (gfx942: 2.75e-7 without any change).
  - **The near-zero gaps are the subnormal flush, shown directly** (`gemm_ftz.py`): the 10 float16 GEMM cases of `mma_dump.py` with the same inputs give, reference vs default 75e2528, max 4 to 178 ulp (180 to 8673 elements differ per case); with the LoRA-up inputs whose float16 value is subnormal zeroed beforehand (both builds), **max 1 ulp in all 10 cases** (13 to 1033 elements), the gfx942 picture, and the same max |diff| / max |out| where the worst case sits.
  - Cross-check: reference 75e2528 vs default **8aa04e7**: only the 10 fp32 `lora_act_out` tensors differ (max rel 2.2e-7) under both scripts, so on gfx90a too the reference build is the 8aa04e7 behavior. The 8aa04e7 default build of this job against the earlier gfx90a job's 8aa04e7 **reference** dumps (same card type, possibly the 10.1 compiler there): the same 10 only.
- **Determinism**: two runs of the default 75e2528 build: `f16_dump.py` 80 of 90 identical, the 10 that differ are exactly the fp32 `lora_act_out` tensors (8 `fused[3]` + 2 `quantize[2]`, max rel 2.36e-7); `mma_dump.py` the same 10 (`34 cases, 10 outputs differ`, max rel 3.1e-7). Two runs of the reference build: the same 10 (2.6e-7). Deterministic.
- **`mma_dump.py`, the INT4 GEMM outputs: reference 75e2528 vs default 75e2528: `34 cases, 54 outputs differ`** (20 identical). Which ones differ:
  - 17 of 20 `gemm[0]` (all 10 fp16; 7 of 10 bf16, the identical three are bf16 256x128x128 both signs and bf16 512x384x1024 unsigned);
  - all 12 fused GEMM outputs `fused[0]`;
  - 5 of 12 packed next-layer activations `fused[1]` (fp16 4 of 6, bf16 1 of 6; 1 to 11 nibbles, one step);
  - 6 of 12 output scales `fused[2]` (all 6 fp16, 1 to 3 ulp; no bf16);
  - all 12 fused `lora_act_out` `fused[3]` (including the 4 N=256 cases that are bit-stable run to run);
  - the 2 quantize `lora_act_out` `quantize[2]` (noise).

  Identical: `quantize[0]`, `quantize[1]` (no 16-bit product) and the remaining `fused[1]`/`fused[2]`. **This matches the gfx950 round's explanation** (53 outputs there and on gfx942, 54 here: one more fp16 output scale flips): the INT4 main loop is unchanged and 75e2528 routes the LoRA-up and LoRA-down epilogues of the same GEMMs through the f16/bf16 MFMA, whose rounding flips (plus, on gfx90a, the subnormal flush above) are what differ; no column-shaped error anywhere. The sizes differ from gfx950's "GEMM outputs at most 1 ulp, scales 2 ulp" only by the flush: up to 178 ulp at near-zero fp16 GEMM values and 3 ulp in one fp16 scale, all within the f16 tolerance. The default build lists the same 54 against the 8aa04e7 default build, and **reference 75e2528 vs default 8aa04e7 differs only in the 10 `lora_act_out` tensors**: the INT4 path (`v_mfma_i32_32x32x8i8`, 21504 per INT4 launch file in both default builds) stays bit-identical to stage 1 on gfx90a.
- **`pytest tests/ops`**: default 75e2528 **38 passed, 14 skipped** in 26 s (the 14 are `test_gemm_w4a4_fp4.py`, "the GPU has no NVFP4 kernels"); reference 75e2528 38 passed, 14 skipped; default 8aa04e7 38 passed.
- **FP4 refuses cleanly on gfx90a** (`fp4_refuse.py`): `get_precision("auto")` -> `int4`; `check_hardware_compatibility` accepts int4 and rejects `fp4_e2m1_all` with `ValueError: "fp4" quantization needs an AMD GPU with FP4 matrix instructions (gfx950, gfx1250); please use "int4" quantization on gfx90a.`; `fp4_supported()` of the FP4 test file -> False. Calling `test_gemm_w4a4_fp4(256, 256, 256, bf16)` and `test_quantize_w4a4_act_fp4(256, 256, bf16)` directly (skip bypassed): each prints `FP4 is not available on this device`, the queue aborts with `HSA_STATUS_ERROR_EXCEPTION ... code: 0x1016`, the next `torch.cuda.synchronize()` raises `torch.AcceleratorError: CUDA error: unspecified launch failure`, and the process exits 1 after 9 s / 7 s; no hang, and the next process (the dumps) used the GPU normally. This is the designed trap (`trap_no_fp4`).
- **End to end on the default 75e2528 build** (in-place build on `PYTHONPATH`): `HF_HOME=<job>/hf NUNCHAKU_TEST_CACHE_ROOT=<job>/test_cache timeout 10800 python -m pytest -v -s tests/v1/sdxl/test_sdxl.py::test_sdxl_lpips tests/v1/sdxl/test_sdxl_turbo.py::test_sdxl_turbo_lpips tests/sana tests/v1/z_image` -> **10 passed** in 2617 s. SDXL LPIPS 0.152 (expected 0.25), SDXL-Turbo 0.226 (0.25), Z-Image Turbo rank 32/128/256 0.390/0.363/0.376 (expected int4-bf16 0.40/0.38/0.37, limit x1.15; rank 256 above the bare value, within the limit, as before), both Sana example scripts (`sana1.6b-int4.png` inspected: the ink-drawing panda). With the 38 of `tests/ops` this is the earlier rounds' 48-test set. Earlier gfx90a at 8aa04e7: 0.154/0.218/0.385/0.372/0.374.
- Not run, as in the earlier rounds: FLUX.1 and Qwen-Image tests (gated models, no Hugging Face token), `test_device_id.py` (one GPU), speed tests.

### Speed (recorded, not gated)

MI250X GCD, ms, `speed.py`, two alternating runs per build (the reference build's single-core link was running on the CPU meanwhile; the GPU was not shared), 8aa04e7 default -> 75e2528 default:

| kernel | 8aa04e7 | 75e2528 |
|---|---|---|
| W4A4 GEMM 4096x3072x3072 fp16 LoRA rank 32 | 15.36-15.41 | **16.74 (+9%)** |
| same, fp16 rank 128 | 23.75 | 18.04-18.06 |
| same, bf16 rank 32 | 21.76-21.78 | **23.74-23.75 (+9%)** |
| same, bf16 rank 128 | 30.06-30.09 | 28.00 |
| AWQ GEMM 512x4096x4096 fp16 | 23.04-23.11 | 17.18 |
| AWQ GEMM 512x4096x4096 bf16 | 22.00-22.01 | 14.75 |
| attention_fp16 24 heads x 4608 tokens | 380.2-380.4 | 353.8-353.9 (1.07x) |

Rank 32 W4A4 is 9% slower in both dtypes (gfx942: bf16 +17%, fp16 flat), rank 128 and AWQ faster; attention gains only 7% here against 1.53x on gfx942 and 1.67x on gfx950. Absolute times are 4 to 6 times gfx942's. Performance-pass items; nothing was changed.

### Scripts

`notes-gfx90a-scripts.md`: `denorm.hip` (the reproducer), the `FTZ_INPUTS` line for `test_cdna.hip`, `gemm_ftz.py`, `quant_ftz.py`, `ulp_where.py`, `fp4_refuse.py`.

### What this leaves

The gfx90a part of `kernel:stage2-cdna-mfma-f16-runs` has its evidence, with the primitive-test failure above for jeff to rule on: (a) accept the gfx90a f16 MFMA with its hardware subnormal flush (ops within tolerance, end to end passes), (b) keep f16 (not bf16) on the lane-gather path on gfx90a, or (c) something else. With gfx942 also done, still owed from the gfx950 round's list: the gfx1250 run, the wave32 reruns at 75e2528, Windows gfx1151 last. Surface accounting was not edited here (evidence run, notes only).

## Lesson for cuda-to-rocm: gfx90a f16 and bf16 MFMA read subnormal inputs as zero

(For `faults-numerics.md`, next to the matrix-instruction accumulator notes.) On gfx90a (MI200), `v_mfma_f32_16x16x4f16` returns 0 for a product whose f16 input is subnormal (|x| < 2^-14): one instruction with A = 2^-20, B = 1 gives 0 in all 1024 outputs, A = 2^-14 is exact. `v_mfma_f32_16x16x4bf16_1k` does the same for bf16 subnormals (below 2^-126, rarely reached). gfx942 and gfx950 do not flush. PyTorch documents the same for MI200 (FP16/BF16 V_DOT2 and MFMA flush input and output denormals), where it only reaches rocBLAS and MIOpen; hand-written MFMA code meets it directly. A port that replaces a CUDA mma.sync f16 product (which keeps f16 subnormals) with the gfx90a MFMA therefore fails a per-product error bound on small-magnitude data (nunchaku's test: 14% of outputs over the bound at data scale 0.01, none at scale 30), while whole-op outputs move only by ulps at near-zero values. Test for it by flushing subnormal f16 inputs in the test data: if the failures vanish, the layout is right and the difference is the hardware. Where it matters, keep f16 on another path on gfx90a or pre-scale the operands; bf16 is safe in practice. Seen 2026-11-24 (nunchaku), ROCm 10.2.

## Evidence run 2026-11-25, gfx1201 RDNA rerun at 75e2528 (linux-gfx1201 fleet job, for jeff)

Not a porter round: no work lock taken, stage untouched (`porting`), nothing changed in or pushed to the fork. Scope: the "wave32 reruns at 75e2528" item of the gfx950 round, for gfx1201 (gfx1100 is still owed), plus the FP4 refusal. **Result: 75e2528 changes nothing observable on gfx1201. Every output of the 75e2528 default build equals the 8aa04e7 default build bit for bit, except the 10 fp32 `lora_act_out` tensors, which differ run to run in every build (float atomics). The 75e2528 reference build equals the 8aa04e7 reference build in the same way. The f16 tolerance table is the gfx12 round's to every printed digit. `tests/ops` gives 38 passed and 14 skipped, the end-to-end set 10 passed, and the FP4 kernels refuse cleanly. Speed is unchanged: the timed kernels are instruction-identical between the two builds.** gfx1201 has no FP4 path at 75e2528, so jeff's ruling leaves `auto` at int4 here; there is nothing to measure.

### Was 75e2528 meant to change RDNA? (read from `git diff 8aa04e7 75e2528`)

No. The four commits (5db74f4, 9ce05de, 01c0f59, 75e2528) touch `mma_rocm.cuh`, `gemm_w4a4.cuh`, `gemm_utils.cuh`, a new FP4 test file and the docs. The RDNA part of the diff:
- `mma_rocm.cuh`: the new macros `NUNCHAKU_ROCM_MFMA_F16` (gfx90a/gfx942/gfx950), `NUNCHAKU_ROCM_FP4` (gfx950/gfx1250) and `NUNCHAKU_ROCM_WMMA_FP4` (gfx1250) are all undefined on gfx11/gfx12. The `NUNCHAKU_ROCM_WMMA_F16` condition gained `&& !defined(__gfx1250__)`, which is still true on gfx1201. `select4` moved into a shared block. The `mma_m16n16k16_*` helpers keep their WMMA arm under `#ifdef NUNCHAKU_ROCM_WMMA_F16` inside the new `WMMA_F16 || MFMA_F16` condition.
- `gemm_w4a4.cuh`: `FP4_AVAILABLE` is still false on gfx1201, so `trap_no_fp4()` stays in both FP4 kernels. The ROCm `compute_fp4` and the gfx1250 code are not instantiated on RDNA.
- `gemm_utils.cuh`: the e2m1/e4m3 conversions replace `trap_unsupported_arch()` in the FP4 quantize helpers on every ROCm target. On gfx1201 they are reached only behind the quantize kernel's `FP4_AVAILABLE` trap.

Code objects confirm this. Each gfx1201 object of the 8aa04e7 and 75e2528 default builds was compared per function: `llvm-objdump -d --no-show-raw-insn --mcpu=gfx1201`, with comments, hex literals, branch targets and the PC-relative `s_getpc`/`s_add*` operands normalized. Result: 14 of 15 objects are identical in every function, including both FP4 launch files. The bf16 INT4 launch file has 3 of 199 functions that differ: the `EpilogueRMSNormRope` + `EpiloguePackQKV` / `EpilogueLoraDown` kernels, with a few instructions of scheduling (22021 vs 22029/22025 and 28074 vs 28061 lines). `mma_dump.py`, `f16_dump.py` and `speed.py` do not run those kernels. The gfx950 round counted 8 differing functions in this file without that normalization. Per-kernel scratch (`.private_segment_fixed_size`) in that file is the same in both builds: 173 kernels, 155324 bytes in total, 1696 at most, 192 VGPRs at most. Matrix instructions in the 75e2528 default build: 21504 `v_wmma_i32_16x16x16_iu4` in each INT4 launch file; `v_wmma_f32_16x16x16_f16` fp16 INT4 launch 3888, attention 64, fasteri2f 64, AWQ 72, fp16 FP4 launch 64 (the trap kernels' LoRA); `..._bf16` bf16 INT4 launch 4320, AWQ 72, bf16 FP4 launch 64. The 8aa04e7 build has the same counts. The gfx12 round's "bf16 INT4 launch 4064" is not what this 8aa04e7 build contains (4320, as the RDNA and gfx950 rounds recorded for gfx1201).

### Environment

- Host gfx1201 (AMD Radeon AI PRO R9700), `HIP_VISIBLE_DEVICES=2` as pinned by the dispatcher (one GPU, kept), `ulimit -c 0`, every GPU run under `timeout`, MAX_JOBS=8 in total (reference 5 + default 3, the default builds one after another).
- Job venv: `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel,device-gfx1201]==10.2.0" "torch==2.14.0+rocm10.2.0" "torchvision==0.29.0+rocm10.2.0" "amd-torch-device-gfx1201==2.14.0+rocm10.2.0" "amd-torch-device-gfx12-0==2.14.0+rocm10.2.0"`. The family wheel supplies `torch/lib/aotriton.images`, so SDPA needed no workaround. Then `rocm-sdk path --root` once; it printed `$FLEET_VENV/lib/python3.11/site-packages/_rocm_sdk_devel`, and `torch.utils.cpp_extension.ROCM_HOME` is that directory.
- **Compiler actually invoked**: every build log has 17 device compile lines running `$FLEET_VENV/lib/python3.11/site-packages/_rocm_sdk_devel/bin/hipcc` (HIP 7.17.26434, AMD clang 24.0.0git bdc35841c97f, ROCm 10.2). None names the image's `/opt/python-3.11-venv`. `test.hip` was compiled with the same `hipcc`.
- Test dependencies as in the earlier rounds: accelerate 1.9.0, diffusers 0.36.0, transformers 4.55.2, peft 0.17.1, datasets 3.6.0, torchmetrics 1.8.2, huggingface-hub 0.34.6, controlnet-aux 0.0.10, image-gen-aux e08ae3d, pytest 8.4.2, numpy 2.4.6, pandas 3.1.1, matplotlib, lpips 0.1.4, scikit-image 0.26.0, scikit-learn 1.9.1, scipy 1.17.1, PyWavelets 1.9.0. opencv-python-headless was uninstalled and opencv-python 4.11.0.86 force-reinstalled.
- Fork: `git clone https://github.com/AMD-Ecosystem/nunchaku`, `git checkout 75e2528` (75e25288499066d5f635aac8500ff955d5da63a5), plus detached worktrees at 75e2528 (reference) and 8aa04e7 (default, the "before" build), with submodules. `git status --porcelain --untracked-files=no --ignore-submodules=all` is empty in all three after every run; scripts ran from outside the trees.
- Builds: `PATH=$FLEET_VENV/bin:$PATH PYTORCH_ROCM_ARCH=gfx1201 MAX_JOBS=<n> $FLEET_VENV/bin/python setup.py build_ext --inplace -v`, one tree each. They ran detached (`setsid nohup`) under `timeit.sh` (in `stats.jsonl`). Default 75e2528: exit 0, 362 s. Default 8aa04e7: exit 0, 366 s. Reference 75e2528 (`NUNCHAKU_ROCM_MMA_REFERENCE=1`; the macro is in its `build.ninja`): exit 0, **7299 s**, almost all of it the single-core device link of the bf16 INT4 launch file (ld.lld, 1.3 GB RSS). 0 errors in any log.
- Earlier evidence used from the gfx12 round's job directory on this host (same card model, the same wheel set, its 10.2 builds of 8aa04e7): its `mma_*`/`f16_*` dumps (reference 8aa04e7 = `ref102`, default 8aa04e7 = `def102`, two runs each), its 16-bit SDXL and Z-Image reference images, its INT4 SDXL images, and its Hugging Face cache (hard-linked, `HF_HUB_OFFLINE=1`).

### Results (all on the one gfx1201 GPU)

- **Primitive test** (`test.hip` from `notes-rdna-scripts.md`, unchanged). Built as `build.sh` there does: the 75e2528 `mma_rocm.cuh` with `__nv_bfloat162` -> `__hip_bfloat162`, the hipified `common_hip.h`/`utils_hip.cuh` of the built tree, `hipcc -O3 -std=c++20 --offload-arch=gfx1201 ...`. The binary has 2 `v_wmma_f32_16x16x16_f16` + 1 `..._bf16`. `timeout 300 ./test_gfx1201` -> **PASS** in all 9 runs: f32-accumulator f16 WMMA 1.34e-7 to 1.67e-7 (reference 2.28e-7 to 3.04e-7); bf16 WMMA 1.10e-7 to 1.46e-7; f16 accumulator 99.96% to 99.99% bitwise equal to the reference; exact small-integer problem exact (0/256). The printed output is byte-identical to that of the gfx12 round's 8aa04e7 binary run here on the same GPU, and to a second run of this binary.
- **`mma_dump.py`** (74 outputs over 34 cases):
  - Default 75e2528 vs the gfx12 round's default 8aa04e7 dumps (`def102`, and its second run `def102b`): **10 differ, exactly the fp32 `lora_act_out` tensors** (8 `fused[3]` + 2 `quantize[2]`, max rel 1.1e-7 to 3.7e-7). Every GEMM output, packed next-layer activation and output scale is bit-identical.
  - Reference 75e2528 vs the gfx12 round's reference 8aa04e7 (`ref102`): the same 10 only.
  - Reference 75e2528 vs default 75e2528: **53 differ**, the same set as reference vs default 8aa04e7 in the gfx12 round:
    - 17 of 20 `gemm[0]` (all 10 fp16, 7 of 10 bf16);
    - all 12 `fused[0]`;
    - 5 `fused[1]` (4 fp16, 1 bf16);
    - 5 `fused[2]` (fp16);
    - the 12 `fused[3]` and 2 `quantize[2]`.

    These are the f16/bf16 WMMA rounding flips of the LoRA epilogues (8aa04e7 behavior), checked against the tolerance below. As the gfx12 round recorded, a bit-identity check of the iu4 path needs a 5c32be5-style build; it is not affected by 75e2528, whose INT4 output equals 8aa04e7's bit for bit.
- **`f16_dump.py` / `reldiff.py`** (90 outputs plus 16 float64-error entries), tolerance from the RDNA round (max |new - ref| / max |output| <= 9.8e-4 float16, 7.8e-3 bfloat16):
  - Default 75e2528 vs default 8aa04e7 (`def102`), and reference 75e2528 vs reference 8aa04e7 (`ref102`): only the 10 `lora_act_out` tensors differ in each pair.
  - Reference 75e2528 vs default 75e2528: 69 differ, **all within the bound**. The `reldiff` output is the same as the gfx12 round's reference vs default 8aa04e7 on every line, apart from one `lora_act_out` noise digit (2.75e-6 vs 2.77e-6). Worst per kind, float16 / bfloat16:
    - W4A4 GEMM out: 3.05e-4 / 2.44e-3.
    - fused GEMM out: 1.77e-4 / 3.26e-4.
    - fused output scales: 1.15e-4 / identical.
    - attention_fp16: 5.98e-4 (4608 tokens) / 4.03e-3 (512 tokens, 500 valid).
    - AWQ GEMM: 5.18e-4 (1024x1024x4096) / 4.00e-3 (512x4096x4096).
    - quantize kernel outputs and scales: identical.
- **Determinism**: two runs of the default 75e2528 build differ only in the 10 `lora_act_out` tensors, under both scripts (max rel 4.2e-7 `mma_dump`, 2.9e-7 `f16_dump`). Two runs of the reference 75e2528 build give the same 10.
- **`pytest tests/ops`**: default 75e2528 **38 passed, 14 skipped** in 33 s; reference 75e2528 38 passed, 14 skipped. The skips are `test_gemm_w4a4_fp4.py` (6 + 4 + 4): "the GPU has no NVFP4 kernels".
- **FP4: gfx1201 has no FP4 path at 75e2528**: `FP4_AVAILABLE` is false and `ROCM_FP4_ARCHS = ("gfx950", "gfx1250")`. It refuses cleanly (`fp4_refuse.py`, gfx942 run's script, unchanged):
  - `get_precision("auto")` -> `int4`; `check_hardware_compatibility` accepts int4 and rejects `fp4_e2m1_all` with `ValueError: "fp4" quantization needs an AMD GPU with FP4 matrix instructions (gfx950, gfx1250); please use "int4" quantization on gfx1201.`; `fp4_supported()` -> False.
  - A direct FP4 GEMM launch and a direct FP4 activation quantize each print `FP4 is not available on this device`. The queue aborts with `HSA_STATUS_ERROR_EXCEPTION ... code: 0x1016` naming `gemm_w4a4_fp4_kernel` / `quantize_w4a4_fuse_lora_kernel<false, true>`. The next `torch.cuda.synchronize()` raises `torch.AcceleratorError: CUDA error: unspecified launch failure` (0.18 s / 0.12 s after the launch), which Python catches. The process exits 0 in 5.5 s / 5.3 s, and a fresh process uses the GPU normally.
- **End to end on the default 75e2528 build** (in-place, on `PYTHONPATH`): `HF_HOME=<job>/hf NUNCHAKU_TEST_CACHE_ROOT=<job>/test_cache HF_HUB_OFFLINE=1 timeout 7000 python -m pytest -v -s tests/v1/sdxl/test_sdxl.py::test_sdxl_lpips tests/v1/sdxl/test_sdxl_turbo.py::test_sdxl_turbo_lpips tests/sana tests/v1/z_image` -> **10 passed** in 703 s.
  - SDXL LPIPS 0.148 (expected 0.25).
  - SDXL-Turbo 0.231 (0.25).
  - Z-Image Turbo rank 32/128/256: 0.391/0.381/0.357 (expected int4-bf16 0.40/0.38/0.37, limit x1.15).
  - Both Sana example scripts (`sana1.6b-int4.png` inspected: the ink-drawing panda).

  The gfx12 round at 8aa04e7: 0.156/0.231/0.386/0.387/0.364 (10.2 build) and 0.151/0.224/0.386/0.363/0.358 (10.1 build). The INT4 images are not pixel-identical to that round's (nearly every pixel differs, by up to 241 levels) even though every GEMM, attention and AWQ output above is bit-identical. Its two builds also moved LPIPS by this much. So the end-to-end images are not reproducible bit for bit across runs; the obvious suspects are the float-atomic `lora_act_out` and torch-side nondeterminism. Not investigated: it is outside the 75e2528 delta.

### Speed (recorded, not gated)

R9700, ms, `speed.py`, alternating 8aa04e7 / 75e2528 default builds: 4 runs each for the bf16 W4A4 rows, 2 for the rest.

| kernel | 8aa04e7 | 75e2528 |
|---|---|---|
| W4A4 GEMM 4096x3072x3072 fp16 LoRA rank 32 | 6.61-6.68 | 6.60-6.61 |
| same, fp16 rank 128 | 7.07-7.15 | 7.07-7.08 |
| same, bf16 rank 32 | 3.47-3.87 | 3.68-3.90 |
| same, bf16 rank 128 | 3.86-4.23 | 4.06-4.28 |
| AWQ GEMM 512x4096x4096 fp16 | 2.66-2.68 | 2.66-2.67 |
| AWQ GEMM 512x4096x4096 bf16 | 2.91-2.93 | 2.91-2.92 |
| attention_fp16 24 heads x 4608 tokens | 53.74-53.76 | 53.81-53.97 |

The bf16 W4A4 ranges overlap; their medians are about 5% apart in 75e2528's disfavor. The kernels timed are instruction-identical in both builds (above), so this is run-to-run variation of this card, not a code change. The gfx12 round measured 3.53 / 3.92 for these two rows. fp16 W4A4 is still 1.7-1.9x slower than bf16, the register-pressure item already recorded.

### Scripts

`mma_dump.py` is in this file, `test.hip`, `f16_dump.py`, `reldiff.py` and `speed.py` in `notes-rdna-scripts.md`, `fp4_refuse.py` in the gfx942 75e2528 run above, all unchanged. New, `diffsum.py` (which outputs of two dumps differ, with max |diff| / max |a|):

```python
"""Summarize two dumps (mma_dump or f16_dump): which outputs differ, and max |diff| / max |a| for those."""
import sys, torch
a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
same = diff = 0
for name in a:
    va, vb = a[name], b[name]
    if torch.is_tensor(va): va, vb = (va,), (vb,)
    for i, (x, y) in enumerate(zip(va, vb)):
        if not torch.is_tensor(x):
            continue
        if torch.equal(x.view(torch.uint8) if x.dtype != torch.uint8 else x, y.view(torch.uint8) if y.dtype != torch.uint8 else y) if x.is_contiguous() and y.is_contiguous() else torch.equal(x, y):
            same += 1
            continue
        diff += 1
        if x.is_floating_point():
            d = (x.double() - y.double()).abs()
            fin = torch.isfinite(d)
            print(f"DIFF {name}[{i}] {x.dtype} n={int((x != y).sum())}/{x.numel()} maxrel {(d[fin].max() / x.double()[fin].abs().max()).item():.3g}")
        else:
            print(f"DIFF {name}[{i}] {x.dtype} n={int((x != y).sum())}/{x.numel()}")
print(f"identical {same}, differ {diff}")
```

### What this leaves

The gfx1201 half of the wave32 rerun at 75e2528 is done, with no difference from 8aa04e7. Still owed from the gfx950 round's list:
- the gfx1100 rerun (that card has the gfx11 code paths, and its INT4 launch files had the scheduling differences the gfx950 round found);
- the gfx1250 run;
- jeff's ruling on the gfx90a f16 MFMA subnormal flush;
- Windows gfx1151 last.

Surface accounting was not edited here (evidence run, notes only).

## Evidence run 2026-11-25, gfx1100 RDNA rerun at 75e2528 (linux-gfx1100 fleet job, for jeff)

Not a porter round: no work lock taken, stage untouched (`porting`), nothing changed in or pushed to the fork. Scope: the gfx1100 half of the gfx950 round's "wave32 reruns at 75e2528" item (the gfx1201 half is the run above), plus the FP4 refusal. **Result: 75e2528 changes nothing observable on gfx1100. Every output of the 75e2528 default build equals the previous RDNA round's 8aa04e7 outputs bit for bit, except the 10 fp32 `lora_act_out` tensors, which differ run to run in every build (float atomics); the 75e2528 reference build equals the previous round's 5c32be5 reference build in the same way. The f16 tolerance table is the previous gfx1100 round's, `tests/ops` 38 passed + 14 FP4 skipped, the end-to-end set 10 passed, speed unchanged, FP4 refuses cleanly.** Nothing failed, so there is no reproducer to record.

### Was 75e2528 meant to change RDNA? (read from `git diff 8aa04e7 75e2528`)

No, and the gfx1201 run above reads the diff the same way. On gfx1100: `NUNCHAKU_ROCM_MFMA_F16`, `NUNCHAKU_ROCM_FP4` and `NUNCHAKU_ROCM_WMMA_FP4` are undefined; the `NUNCHAKU_ROCM_WMMA_INT4`/`_F16` condition gained `&& !defined(__gfx1250__)`, still true; `select4` moved into a shared block unchanged; the `mma_m16n16k16_*` helpers keep their WMMA arm; `FP4_AVAILABLE` stays false, so both FP4 kernels and the FP4 quantize epilogue still reach `trap_no_fp4()`; the new e2m1/e4m3 conversions in `gemm_utils.cuh` replace `trap_unsupported_arch()` but are reached only behind that trap. Code objects (`llvm-objcopy --dump-section=.hip_fatbin`, `clang-offload-bundler --unbundle --targets=hipv4-amdgcn-amd-amdhsa--gfx1100`, `llvm-objdump -d --mcpu=gfx1100`), 8aa04e7 -> 75e2528 default: fp16 INT4 launch 21504 `v_wmma_i32_16x16x16_iu4` + 3888 `v_wmma_f32_16x16x16_f16` in both (2,119,039 vs 2,119,047 lines); bf16 INT4 launch 21504 iu4 + 4400 `..._bf16` in both (3,693,755 lines each); bf16 FP4 launch 64 bf16 (the trap kernels' LoRA) in both; attention 64 f16, AWQ 72 + 72, identical line counts; no MFMA and no `v_wmma_scale16` anywhere. These are the gfx1100 counts the gfx950 round recorded. Reference 75e2528: no WMMA in any of these objects (1.79M and 6.86M lines for the two INT4 launch files).

### Environment

- Host gfx1100 (AMD Radeon Pro W7800 48GB), `HIP_VISIBLE_DEVICES=1` as pinned by the dispatcher (one GPU, kept), `ulimit -c 0`, every GPU run under `timeout`, MAX_JOBS=16 (the exported value) in total: reference build 10 jobs, the two default builds 6, one after the other.
- Job venv: `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel,device-gfx1100]==10.2.0" "torch==2.14.0+rocm10.2.0" "torchvision==0.29.0+rocm10.2.0" "amd-torch-device-gfx1100==2.14.0+rocm10.2.0" "amd-torch-device-gfx110x==2.14.0+rocm10.2.0"`. Then `rocm-sdk path --root` once; it printed `$FLEET_VENV/lib/python3.11/site-packages/_rocm_sdk_devel`, and `torch.utils.cpp_extension.ROCM_HOME` is that directory.
- **The gfx110x family wheel fixes SDPA on gfx1100 too** (the previous gfx1100 round needed a `sitecustomize` that disabled flash and memory-efficient SDPA): `torch/lib/aotriton.images` is present, and flash fp16/bf16 and memory-efficient fp16/bf16/fp32 run, max error vs float64 9.8e-4 / 7.8e-3 / 3.7e-6 (flash has no fp32 kernel, "No available kernel", as on gfx1201). No workaround was used below. This confirms the gfx12 round's lesson (`amd-torch-device-gfx110x`) on gfx1100.
- **Compiler actually invoked**: all three build logs have 17 device compile lines each running `$FLEET_VENV/lib/python3.11/site-packages/_rocm_sdk_devel/bin/hipcc` (HIP 7.17.26434, AMD clang 24.0.0git bdc35841c97f, ROCm 10.2); none names the image's `/opt/python-3.11-venv`. `test.hip` was compiled with the same `hipcc`. The previous gfx1100 round's build logs, still in its job directory on this host, also invoke that job venv's `_rocm_sdk_devel/bin/hipcc` (expanded there), so that round built with 10.2 too; the gfx12 round's open question about it is answered for gfx1100.
- Test dependencies as in the earlier rounds: accelerate 1.9.0, diffusers 0.36.0, transformers 4.55.2, peft 0.17.1, datasets 3.6.0, torchmetrics 1.8.2, huggingface-hub 0.34.6, controlnet-aux 0.0.10, image-gen-aux e08ae3d, pytest 8.4.2, numpy 2.4.6, pandas 3.1.1, matplotlib, lpips 0.1.4, scikit-image 0.26.0, scikit-learn 1.9.1, scipy, PyWavelets upgraded; opencv-python-headless uninstalled and opencv-python 4.11.0.86 force-reinstalled.
- Fork: `git clone https://github.com/AMD-Ecosystem/nunchaku`, `git checkout 75e2528` (75e25288499066d5f635aac8500ff955d5da63a5), plus detached worktrees at 75e2528 (reference) and 8aa04e7 (default, the "before" build), submodules at the recorded commits. `git status --porcelain --untracked-files=no --ignore-submodules=all` is empty in all three at the end; scripts ran from outside the trees (the Sana example writes untracked PNGs).
- Builds: `PATH=$FLEET_VENV/bin:$PATH PYTORCH_ROCM_ARCH=gfx1100 MAX_JOBS=<n> $FLEET_VENV/bin/python setup.py build_ext --inplace -v`, one tree each, detached (`setsid nohup`, a script appending `EXIT <code> <seconds>`). Not wrapped in `timeit.sh`, because this dispatch allowed committing notes only, so no compile telemetry. Default 75e2528: exit 0, 380 s. Default 8aa04e7: exit 0, 390 s. Reference 75e2528 (`NUNCHAKU_ROCM_MMA_REFERENCE=1`, on all 17 compile lines): exit 0, **7779 s**, almost all of it the single-core device link of the bf16 INT4 launch file (ld.lld). 0 errors in any log.
- Earlier evidence used from the previous gfx1100 round's job directory on this host (same card model, same wheel set, 10.2 builds): its dumps `dump_ref.pt` (mma_dump, reference 5c32be5), `f16_ref.pt` (f16_dump, reference 5c32be5) and `f16_new.pt` (f16_dump, default 8aa04e7), its 16-bit SDXL and Z-Image reference images, and its Hugging Face cache (hard-linked, `HF_HUB_OFFLINE=1`). No INT4 images were copied, so this build generated its own.

### Results (all on the one gfx1100 GPU)

- **Primitive test** (`test.hip` from `notes-rdna-scripts.md`, unchanged except that it includes the hipified `mma_rocm_hip.cuh` of the built 75e2528 tree, compiled with `-I <tree>/src` and the `build.sh` flags, `--offload-arch=gfx1100`): `timeout 300 ./test_gfx1100` -> **PASS**, no output over the bound in all 9 runs. Max |err| / sum |terms|: f32-accumulator f16 WMMA 2.0e-7 to 2.4e-7 (reference 2.3e-7 to 3.0e-7); bf16 WMMA 1.5e-7 to 1.6e-7 (reference 1.2e-7 to 1.3e-7); f16 accumulator 99.95% to 99.99% bitwise equal to the reference, same max error 4.6e-4; exact small-integer problem: reference exact, WMMA 16/256 one fp32 ulp off in f32 modes (0/256 with the f16 accumulator). The same numbers as the previous gfx1100 round (2.4e-7, 1.6e-7, 16/256): the gfx11 accumulator rounding, which is why C stays outside the WMMA.
- **`f16_dump.py` / `reldiff.py`** (90 outputs plus 16 float64-error entries; scripts in `notes-rdna-scripts.md`, unchanged):
  - **Default 75e2528 vs the previous round's default 8aa04e7 (`f16_new.pt`)**: 80 identical, **10 differ, exactly the fp32 `lora_act_out` tensors** (8 `fused[3]` + 2 `quantize[2]`, max rel 2.9e-7). All 8 attention_fp16 and 8 AWQ cases, every GEMM output, packed next-layer activation and scale are bit-identical.
  - **Reference 75e2528 vs the previous round's reference 5c32be5 (`f16_ref.pt`)**: 80 identical, the same 10 differ (max rel 2.5e-7).
  - Reference 75e2528 vs default 75e2528: 18 identical, 72 differ, **all within the bound** (RDNA round's tolerance: max |new - ref| / max |output| <= 9.8e-4 float16, 7.8e-3 bfloat16). The `reldiff` output equals reference 5c32be5 vs default 75e2528 on every line apart from four `lora_act_out` noise digits, and `f16_dump compare` of the previous round's own pair (reference 5c32be5 vs default 8aa04e7) gives the same worst-per-kind line. Worst per kind, float16 / bfloat16:
    - W4A4 GEMM out: 3.05e-4 / 2.44e-3 (4352x256x3072 unsigned); max 4 ulp element-wise.
    - fused GEMM out: 1.77e-4 / 3.26e-4; max 8 ulp at a small value.
    - fused output scales: 1.15e-4 / identical; max 2 ulp.
    - attention_fp16: 4.02e-4 (1024 tokens) / 4.03e-3 (512 tokens, 500 valid).
    - AWQ GEMM: 5.18e-4 (1024x1024x4096) / 4.00e-3 (512x4096x4096).
    - packed next-layer activations: at most 2 nibbles per case, one step; quantize kernel outputs and scales identical; fused `lora_act_out` max rel 1.4e-4 (LoRA-down through the WMMA).
    - each build's own error against float64 is identical to three digits in all 16 attention and AWQ cases (e.g. attention fp16 4608 tokens 0.00239 both, AWQ bf16 512x4096x4096 0.00304 both).
- **`mma_dump.py`** (74 outputs over 34 cases, script in this file, unchanged):
  - Reference 75e2528 vs the previous round's reference 5c32be5 (`dump_ref.pt`): `34 cases, 10 outputs differ`, the 10 `lora_act_out` tensors.
  - Reference 75e2528 vs default 75e2528, and the previous round's reference 5c32be5 vs default 75e2528: **`34 cases, 56 outputs differ`**: all 20 `gemm[0]` (1 to 1185 bytes of up to 6.3M), all 12 `fused[0]`, 5 `fused[1]` (1 to 2 nibbles), 5 `fused[2]` (2 to 5 bytes), the 12 `fused[3]` and 2 `quantize[2]`. These are the f16/bf16 WMMA rounding flips of the LoRA-up/down epilogues that 8aa04e7 introduced, covered by the tolerance above, not INT4 layout errors (a wrong layout moves whole columns). gfx1201 counted 53 because 3 bf16 `gemm[0]` cases came out identical there; that the gfx11 accumulator rounding (16/256 inexact in the primitive test, 0/256 on gfx1201) accounts for the 3 is a guess, not checked. The INT4 main loop's bit identity stays as proven at 5c32be5: 75e2528's INT4 outputs equal 8aa04e7's bit for bit (previous bullet), and 8aa04e7's iu4 path is 5c32be5's.
- **Determinism**: two runs of the default 75e2528 build differ only in the 10 `lora_act_out` tensors, under both scripts (`mma_dump`: `34 cases, 10 outputs differ`; `f16_dump`: 80 identical, max rel 2.9e-7). Two runs of the reference build: the same 10.
- **`pytest tests/ops`**: default 75e2528 **38 passed, 14 skipped** in 17.6 s; reference 75e2528 38 passed, 14 skipped. The 14 skips are `test_gemm_w4a4_fp4.py` (6 + 4 + 4), "the GPU has no NVFP4 kernels".
- **FP4: gfx1100 has no FP4 path at 75e2528** (`FP4_AVAILABLE` false, `ROCM_FP4_ARCHS = ('gfx950', 'gfx1250')`), and it refuses cleanly (`fp4_refuse.py`, the gfx942 75e2528 run's script, unchanged):
  - `get_precision('auto')` -> `int4`; `check_hardware_compatibility` accepts int4 and rejects `fp4_e2m1_all` with `ValueError: "fp4" quantization needs an AMD GPU with FP4 matrix instructions (gfx950, gfx1250); please use "int4" quantization on gfx1100.`
  - A direct FP4 GEMM launch and a direct FP4 activation quantize each print `FP4 is not available on this device`; the queue aborts with `HSA_STATUS_ERROR_EXCEPTION ... code: 0x1016` naming `gemm_w4a4_fp4_kernel` / `quantize_w4a4_fuse_lora_kernel<false, true>`; the next `torch.cuda.synchronize()` raises `torch.AcceleratorError: CUDA error: unspecified launch failure` (0.18 s / 0.11 s after the launch), which Python catches; each process exits 0 in 5.6 s / 5.5 s, and a fresh process uses the GPU normally. Under Jeff's ruling, auto stays int4; there is no FP4 run to make on this card.
- **End to end on the default 75e2528 build** (in-place, on `PYTHONPATH`, no SDPA workaround): `HF_HOME=<job>/hf NUNCHAKU_TEST_CACHE_ROOT=<job>/test_cache HF_HUB_OFFLINE=1 timeout 9000 python -m pytest -v -s tests/v1/sdxl/test_sdxl.py::test_sdxl_lpips tests/v1/sdxl/test_sdxl_turbo.py::test_sdxl_turbo_lpips tests/sana tests/v1/z_image` -> **10 passed** in 773 s.
  - SDXL LPIPS 0.149 (expected 0.25).
  - SDXL-Turbo 0.216 (0.25).
  - Z-Image Turbo rank 32/128/256: 0.410/0.394/0.354 (expected int4-bf16 0.40/0.38/0.37, limit x1.15 = 0.46/0.437/0.426). Ranks 32 and 128 are above the bare expected value and within the limit.
  - Both Sana example scripts (`sana1.6b-int4.png` inspected: the ink-drawing panda).

  The previous gfx1100 round at 8aa04e7 (with the math-SDPA workaround): 0.157/0.222/0.393/0.387/0.371. **Z-Image run-to-run control**, because the kernel outputs above are bit-identical between 8aa04e7 and 75e2528 while Z-Image moved: `pytest -v -s tests/v1/z_image` twice per build on this GPU (Z-Image regenerates its INT4 images every run; 6 passed each time). LPIPS rank 32/128/256:

  | build | run 1 | run 2 |
  |---|---|---|
  | 8aa04e7 default | 0.375 / 0.371 / 0.376 | 0.404 / 0.376 / 0.362 |
  | 75e2528 default | 0.410 / 0.394 / 0.354 (full set above) | 0.403 / 0.379 / 0.351 |

  The same build moves by up to 0.029 (8aa04e7 rank 32) between runs, and the two builds' ranges overlap at every rank, so the shift is the end-to-end nondeterminism the gfx1201 run also saw (float-atomic `lora_act_out` and torch-side kernels are the obvious suspects; not investigated, it is outside the 75e2528 delta), not 75e2528.
- Not run, as in the earlier rounds: FLUX.1 and Qwen-Image tests (gated models, no Hugging Face token), `test_device_id.py` (one GPU), speed tests.

### Speed (recorded, not gated)

W7800, ms, `speed.py` (`notes-rdna-scripts.md`, unchanged), two alternating runs per build, 8aa04e7 default -> 75e2528 default; the previous gfx1100 round's 8aa04e7 numbers on the same card model in the last column:

| kernel | 8aa04e7 | 75e2528 | previous round, 8aa04e7 |
|---|---|---|---|
| W4A4 GEMM 4096x3072x3072 fp16 LoRA rank 32 | 7.84-7.90 | 7.79-7.85 | 7.73 |
| same, fp16 rank 128 | 8.22-8.41 | 8.28-8.38 | 8.10 |
| same, bf16 rank 32 | 5.21-5.27 | 5.16-5.26 | 5.20 |
| same, bf16 rank 128 | 5.76-5.80 | 5.75-5.76 | 5.73 |
| AWQ GEMM 512x4096x4096 fp16 | 3.66 | 3.62-3.66 | 3.57 |
| AWQ GEMM 512x4096x4096 bf16 | 3.82 | 3.75-3.80 | 3.68 |
| attention_fp16 24 heads x 4608 tokens | 58.2-59.1 | 58.1-58.4 | 56.7 |

No change from 75e2528 (every pair within 1%, 75e2528 not slower anywhere). The reference build's long compile ran on the CPU during these timings, which may explain the 1-3% gap to the previous round. fp16 W4A4 is still 1.5x slower than bf16 on gfx1100, the register-pressure item already recorded.

### Scripts

`mma_dump.py` is in this file; `test.hip` (one include changed as stated), `f16_dump.py`, `reldiff.py` and `speed.py` are in `notes-rdna-scripts.md`; `fp4_refuse.py` is in the gfx942 75e2528 run above. All were checked byte-identical to the recorded copies before use. No new script.

### What this leaves

Both wave32 reruns at 75e2528 are now done (gfx1201 above, gfx1100 here), so the "wave32 reruns" item of the gfx950 round is closed. Still owed from that list: the gfx1250 run, jeff's ruling on the gfx90a f16 MFMA subnormal flush, Windows gfx1151 last. Surface accounting was not edited (evidence run, notes only).

## Rulings on the gfx950 round's questions (Jeff Daily, 2026-11-25)

Answers to the two questions of "Porter round 2026-11-24, gfx950" (relayed in the porter dispatch of 2026-11-25).

- Q1 (loading NVFP4 checkpoints on AMD while `auto` stays `int4`): **(a)**. The v1 model loaders take the precision from the checkpoint's `quantization_config` when the device supports that precision (`check_hardware_compatibility`), and fall back to `get_precision()` as today otherwise. So an NVFP4 checkpoint loads through the public API on gfx950 while `auto` stays `int4`.
- Q2 (a gfx1250 matrix-instruction path for INT4 and f16): **(b)**. Write a proper gfx1250 matrix-instruction path for INT4 and the f16/bf16 products (gfx1250 took the lane-gather reference path, which spills heavily). Jeff wants real MI450 support.
- The gfx1250 hardware run follows in a separate job on the MI450 host; this round compiles for gfx1250 only.

## Porter round 2026-11-25, gfx950: checkpoint precision (Q1 a), gfx1250 WMMA (Q2 b) (porter, linux-gfx950 fleet job, for jeff)

Fork head 75e2528 -> **555105f**, two new commits on `moat-port`: `662be3a` (Q1: loaders take the checkpoint's precision when the GPU supports it, test, installation guide) and `555105f` (Q2: gfx1250 WMMA for the INT4 GEMM and the 16-bit products, plus a 256-thread launch bound on gfx1250). Stage left at `porting` (what remains is at the end). Scripts new in this round: `notes-gfx1250-scripts.md`.

### Environment

- Host gfx950 (MI355X), `HIP_VISIBLE_DEVICES=7` as pinned, `ulimit -c 0`, every GPU run under `timeout`, at most 64 build jobs in total (the exported `MAX_JOBS`) across concurrent builds.
- Job venv: `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel,device-gfx950]==10.2.0" "torch==2.14.0+rocm10.2.0" "torchvision==0.29.0+rocm10.2.0" "amd-torch-device-gfx950==2.14.0+rocm10.2.0"`, then `rocm-sdk path --root` once (prints `$FLEET_VENV/lib/python3.11/site-packages/_rocm_sdk_devel`). `torch.utils.cpp_extension.ROCM_HOME` is that directory. **Compiler actually used**: every device compile line of every build log of this round invokes `$FLEET_VENV/lib/python3.11/site-packages/_rocm_sdk_devel/bin/hipcc` (HIP 7.17.26434, AMD clang version 24.0.0git, ROCm/llvm-project bdc35841c97f), never the image's 10.1. Test dependencies as in the earlier rounds (accelerate 1.9.0, diffusers 0.36.0, transformers 4.55.2, peft 0.17.1, datasets 3.6.0, torchmetrics 1.8.2, huggingface-hub 0.34.6, controlnet-aux 0.0.10, image-gen-aux e08ae3d, pytest 8.4.2, numpy 2.4.6, pandas 3.1.1, matplotlib, lpips; scikit-image/scikit-learn/scipy/PyWavelets upgraded; opencv-python-headless removed, opencv-python 4.11.0.86 reinstalled), plus ruff 0.14.5, black 25.11.0 and clang-format 21.1.5 (the pre-commit pins) for formatting checks.
- Hugging Face downloads with `hf-xet` stalled after 2 GB on one file; `HF_HUB_DISABLE_XET=1` downloads ran at about 50 MB/s each.
- Builds (`PYTORCH_ROCM_ARCH=<archs> python setup.py build_ext --inplace`, detached, wrapped in `timeit.sh`, recorded in `stats.jsonl`), all exit 0, no errors: 75e2528 for `gfx90a;gfx942;gfx950;gfx1100;gfx1201` 1397 s; 75e2528 for gfx1250 **3089 s**; new code for gfx1250 **312 s** (without the launch bound) and 288 s (with it); new code for the five targets 1404 s; the same-path builds for the identity check (below) 1394 s (75e2528) and 1390 s (final tree) and 260 s (final tree, gfx1250); documented install `PYTORCH_ROCM_ARCH=gfx950 MAX_JOBS=16 python -m pip install -e . --no-build-isolation -v` 391 s (`nunchaku-1.3.0.dev20261125+rocm10.2torch2.14`).
- ROCm build trap met again: after editing a header, hipify skips it ("skipped, already hipified") and the extension's ninja rules for `.hip` files carry no header dependencies, so an in-place rebuild compiles nothing. Delete the build directory (and the converted header) after a header edit.

### 1. Q1: the loaders take the checkpoint's precision when the GPU supports it (fork 662be3a)

Loaders found (every v1 loader calling `get_precision()` with no argument): `NunchakuZImageTransformer2DModel` (`transformer_zimage.py:403`), `NunchakuQwenImageTransformer2DModel` (`transformer_qwenimage.py:397`), `NunchakuFluxTransformer2DModelV2` (`transformer_flux_v2.py:416`), `NunchakuSDXLUNet2DConditionModel` (`unet_sdxl.py:513`). `NunchakuFluxTransformer2dModel` and `NunchakuSanaTransformer2DModel` (the C++ models) pass a precision argument and were not changed (question below).

Change: a new `nunchaku.utils.get_checkpoint_precision(quantization_config)` returns `"int4"` or `"fp4"` from the checkpoint's `weight.dtype` (`int4`, `fp4_e2m1_all`) when `check_hardware_compatibility(quantization_config)` accepts it, and `get_precision()` otherwise (also for an empty config or another dtype). The four loaders call it in place of `get_precision()`; the unused `get_precision` imports went. `get_precision("auto")` itself is unchanged (`int4` on every AMD GPU). Both use the default device `"cuda"` as the loaders did.

Selection table, from the code (`check_hardware_compatibility` on CUDA accepts int4 only on sm_75/80/86/89, fp4 only on sm_120/121, nothing elsewhere; on ROCm int4 everywhere and fp4 on `ROCM_FP4_ARCHS` = gfx950, gfx1250). "fails" means the state dict does not fit the model built for that precision.

| device support | checkpoint | before (precision, load) | after |
|---|---|---|---|
| CUDA int4-only (sm_75, 80, 86, 89) | int4 | int4, loads | int4, loads |
| | NVFP4 | int4, fails | int4, fails (check rejects; fallback) |
| | no config | int4 | int4 (fallback) |
| CUDA fp4-only (sm_120, 121) | int4 | fp4, fails | fp4, fails (check rejects; fallback) |
| | NVFP4 | fp4, loads | fp4, loads |
| | no config | fp4 | fp4 (fallback) |
| CUDA other (sm_90, sm_100) | int4 | int4, loads | int4, loads (check rejects everything; fallback) |
| | NVFP4 | int4, fails | int4, fails |
| ROCm without NVFP4 kernels (gfx90a, gfx942, gfx11, gfx12.0) | int4 | int4, loads | int4, loads |
| | NVFP4 | int4, fails | int4, fails (check rejects; fallback) |
| ROCm with NVFP4 kernels (gfx950, gfx1250) | int4 | int4, loads | int4, loads |
| | NVFP4 | int4, **fails** | **fp4 (nvfp4), loads** |

On CUDA, whenever the check accepts a checkpoint its precision equals what `get_precision()` returns, so every CUDA row is unchanged by construction; the only changed row is the last.

Measured (`loader_matrix.py`, `notes-gfx1250-scripts.md`): the public checkpoints the tests use, Z-Image Turbo `svdq-{int4,fp4}_r32` and `_r128`, SDXL and SDXL-Turbo `svdq-int4_r32`, FLUX.1-schnell `svdq-{int4,fp4}_r32` (v2 loader), Qwen-Image `svdq-{int4,fp4}_r32`, each loaded through its loader onto the CPU with the GPU faked as sm 7.5, 8.0, 8.6, 8.9, 9.0, 10.0, 12.0, 12.1 and as gfx90a, gfx942, gfx950, gfx1100, gfx1201, gfx1250: 140 loads with the 75e2528 tree and 140 with the new tree. **8 results differ, all NVFP4 checkpoints on gfx950 and gfx1250: "int4, state dict fails" before, "nvfp4, loads" after. No CUDA result differs** (same precision and the same load error text in every case). Unfaked on this gfx950, model moved to the GPU: 75e2528 loads every int4 checkpoint as int4 and fails every NVFP4 one; the new code loads all ten, the NVFP4 ones as nvfp4. (Passing `device="cuda"` to the Z-Image loader fails before and after with `SVDQW4A4Linear() got multiple values for keyword argument 'device'`: it forwards its keyword arguments to the layers; the tests do not pass it. Upstream behavior, unchanged.)

Unit test (new, `tests/ops/test_checkpoint_precision.py`, 23 tests): fakes `torch.version.hip`, `get_device_capability` and `get_device_properties` (a CUDA device that asks for its ROCm architecture fails the test), checks the table above for int4, NVFP4 and empty configs on 8 CUDA and 6 ROCm devices, that every CUDA device gets exactly `get_precision()`, and that an unknown dtype falls back. Runs on any backend and needs no GPU.

Installation guide: "INT4 models run on every supported AMD GPU, and NVFP4 models also run on ``gfx950``: the model loaders take the precision from the checkpoint when the GPU supports it. ``get_precision()`` returns ``int4`` on AMD GPUs, so code that chooses a checkpoint by its result, as the examples do, uses the INT4 checkpoints." gfx1250 is not named there until the MI450 run.

### 2. Q2: gfx1250 WMMA for INT4 and the 16-bit products (fork 555105f)

Which instructions, and why. At the compiler used (AMD clang 24.0.0git bdc35841c97f), the gfx1250 WMMA builtins are `__builtin_amdgcn_wmma_{f32,f16,bf16,bf16f32}_16x16x32_{f16,bf16}`, `..._16x16x{64,128}_{fp8,bf8}_{fp8,bf8}`, `..._f32_16x16x128_f8f6f4`, `..._f32_16x16x4_f32`, `..._f64_16x16x4_f64`, `..._i32_16x16x64_iu8`, the `wmma_scale{,16}` and `32x16x128_f4` forms, and the gfx1250 `swmmac` forms (sparse). **There is no 4-bit integer WMMA for gfx1250**: the iu4 builtins (`wmma_i32_16x16x{16,32}_iu4_w32_gfx12`, `swmmac_i32_16x16x{32,64}_iu4`) need the target feature `wmma-128b-insts`, which gfx1250 lacks (`error: '__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12' needs target feature wmma-128b-insts,wavefrontsize32`); `llvm-mc -mcpu=gfx1250 -mattr=help` lists the features. So INT4 uses **`v_wmma_i32_16x16x64_iu8` with the 4-bit values widened to bytes** (exact, and one instruction covers the whole k = 64 of the m16n8k64 primitive), and the 16-bit products use **`v_wmma_f32_16x16x32_{f16,bf16}`**. The FP4 path keeps `v_wmma_scale16_f32_16x16x128_f8f6f4` (75e2528).

Layouts. Two independent sources agree: FlyDSL (ROCm/FlyDSL 0768aa5, `lib/Dialect/FlyROCDL/GFX1250/MmaAtom.cpp` and `WmmaLayout.h`, with the hardware test `tests/unit/test_gfx1250_mma_atom_precision_check.py` building bf16 16x16x32 fragments that way) and composable_kernel (ROCm/rocm-libraries 9e6660e, `ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_base_traits.hpp`, gfx12 base traits that the gfx1250 iu8 and 16-bit traits use: kK1PerLane 8, kKLane 2, kK0PerLane = K / 16). Operand: lane l holds row (A) or column (B) l % 16; element v (byte for iu8, half for f16/bf16) is at k = 16 * (v / 8) + 8 * (l / 16) + v % 8. Result (f32/i32, 8 registers): register v of lane l is row 8 * (l / 16) + v, column l % 16.

Code (`mma_rocm.cuh`, `gemm_w4a4.cuh`, `gemm_base.cuh`, following the RDNA path's structure; selection unchanged otherwise):
- The RDNA macros `NUNCHAKU_ROCM_WMMA_INT4` / `NUNCHAKU_ROCM_WMMA_F16` now also cover gfx1250 (the `!defined(__gfx1250__)` exclusion of 75e2528 is gone); `NUNCHAKU_ROCM_MMA_REFERENCE` still selects the lane-gather reference everywhere.
- INT4 (`mma_rocm::wmma_i32_16x16x64_int4<a_unsigned, b_unsigned>`, gfx1250 only, called from `gemm_w4a4_block_wmma`): the GEMM loads its operands exactly as on gfx12 (step s takes chunk 2s + l / 16 of the mma.sync fragment, i.e. k 16s + 8 * (l / 16) .. + 7), and the helper widens each word with the existing `widen_int4x8` (sign-extended for signed operands) into words 2s and 2s + 1, which is where the iu8 layout wants those k, then issues one `v_wmma_i32_16x16x64_iu8` (signedness flags: weights signed, activations signed unless `ACT_UNSIGNED`). The result goes through the shared-memory staging of the gfx12 path (gfx1250 has the gfx12 result layout) and the scales are applied as before, so it is exact.
- 16-bit (`wmma_m16n16k16_f32`, gfx12 branch): words 0-3 of the gfx1250 operand are the gfx12 16x16x16 operand (k 8 * (l / 16) .. + 7), words 4-7 (k 16 + ...) are zero; the result layout is gfx12's, so the gfx12 data movement in and out is reused unchanged. C stays outside the instruction (added in fp32), as on CDNA and RDNA. The upper half of k is wasted: every caller issues one k = 16 product at a time (attention, LoRA, AWQ loops), and filling k = 32 would need two consecutive products fused in each caller, which is a performance item for after the MI450 measurement.
- Launch bound (`gemm_base.cuh`, `invoke_kernel`, `#if defined(__gfx1250__)` only): `__launch_bounds__(256)`. Every kernel launched through `invoke_kernel` uses at most 256 threads (GEMM, attention and W8A8 blocks of 8 warps, the quantize kernels 32, `vk_mul_q` 128 or 256). Without a bound the compiler assumes 1024 threads and caps every gfx1250 kernel at 128 VGPRs (all heavy kernels sat at exactly 128 and spilled, in 75e2528 and in the WMMA code alike). With it the compiler uses up to 511 VGPRs per wave in the INT4/FP4 GEMM kernels (404 in attention), addressing registers above 255 with `s_set_vgpr_msb` (3686 of them in `attention.o`), and reports `.vgpr_spill_count: 0` for attention. So **gfx1250 waves can address more than 256 VGPRs** (the "max 256 addressable per wave" fact I was given is the plain encoding without the MSB mode); whether the MI450 runs these blocks at that size is the first thing its run shows (a launch failure would say so; see the recipe).

gfx1250 code objects, 75e2528 against this round (`cocmp.py`; "scratch" = scratch load/store instructions in the disassembly, "priv" = largest private segment of a kernel, "VGPR" = largest `.vgpr_count`):

| object | matrix instructions 75e2528 | matrix instructions now | scratch 75e2528 | scratch WMMA only | scratch now | priv bytes 75e2528 / now | VGPR 75e2528 / now |
|---|---|---|---|---|---|---|---|
| gemm_w4a4_launch_fp16_int4 | none | 5376 iu8 + 4656 f16 | 339159 | 252481 | 6109 | 3104 / 1024 | 128 / 496 |
| gemm_w4a4_launch_bf16_int4 | none | 5376 iu8 + 4864 bf16 | 1856866 | 227575 | 8157 | 5760 / 760 | 128 / 488 |
| gemm_w4a4_launch_fp16_fp4 | 2688 scale16 | 2688 scale16 + 4656 f16 | 492942 | 329521 | 7013 | 3472 / 1040 | 128 / 511 |
| gemm_w4a4_launch_bf16_fp4 | 2688 scale16 | 2688 scale16 + 4864 bf16 | 537652 | 334378 | 9243 | 3216 / 800 | 128 / 511 |
| gemm_w4a4_launch_fp16_int4_fasteri2f | none | 64 f16 | 446 | 1555 | 0 | 232 / 0 | 128 / 439 |
| attention | none | 64 f16 | 2550 | 1171 | 0 | 2956 / 0 | 128 / 404 |
| gemm_awq | none | 72 f16 + 72 bf16 | 6375 | 3071 | 3071 | 2052 / 1300 | 128 / 128 |
| gemm_w8a8 (no caller) | none | none | 1654 | 1654 | 310 | 1344 / 672 | 128 / 224 |
| gemm_w4a4_test | none | none | 191 | 191 | 1 | 704 / 16 | 128 / 284 |
| misc_kernels | none | none | 15352 | 15352 | 15352 | 880 / 880 | 128 / 128 |

(`gemm_awq` and `misc_kernels` are launched directly, not through `invoke_kernel`, so the bound does not reach them; adding one there is a follow-up. The other objects have no scratch in either build.) Matrix-instruction counts: the INT4 files have a quarter of gfx12's 21504 iu4 instructions, since one iu8 instruction covers the k = 64 of four iu4 ones. The gfx1250 build takes 288 s instead of 3089 s.

Other targets unchanged (proof): both versions built one after the other at the same path (`cmpbuild.sh`; building them in different directories puts different paths into assert strings and `__hip_cuid_*`, which moves string offsets), every object and every target compared with `cocmp.py`: **the preprocessed device code of all 17 `.hip` sources is identical for all five targets (85 of 85; `ppall.sh`: device-only `-E -P`, build flags, tree path mapped)**, so the compiler input for gfx90a, gfx942, gfx950, gfx1100 and gfx1201 did not change. The disassembly is identical for 73 of the 75 object/target pairs; the other two, `gemm_w4a4_launch_bf16_int4.o` for gfx1100 and gfx1201, differ in instruction order and register choice inside a few GEMM kernels, as the gfx950 round's RDNA check also saw. That is the compiler, not the source: **compiling one and the same source twice with the same command gives different gfx11 code** (`cc1.sh`, 75e2528's `gemm_w4a4_launch_fp16_int4.hip` for gfx1100: 224 differing lines between the two compiles, 48 lines in the sorted instruction multiset, e.g. a `v_mul_f32` / `v_fma_mix*` group in another order with other `s_delay_alu` hints; adding 25 blank lines to the header gives a third variant). The same for the file that differed here: 75e2528's `gemm_w4a4_launch_bf16_int4.hip` compiled twice for gfx1201 gives two objects whose disassembly differs in 1316 lines (62 in the sorted multiset). So for gfx11/gfx12 the preprocessed source is the proof, and identical disassembly cannot be expected from any two builds. The first comparison of this round, between builds in different directories, differed in 17 pairs; those builds also differ in the tree paths inside assert strings and in `__hip_cuid_*`. The gfx12 branch of `wmma_m16n16k16_f32` first had its `using operand_t` line moved below the operand loop, which showed up as the only gfx1201 token difference; the committed version keeps that line in its original place for gfx11/gfx12, which is what the 85/85 result above is for.

### 3. gfx1250 primitive test (`test_gfx1250.hip`, `notes-gfx1250-scripts.md`), to run first on the MI450

Wave32, one wavefront, in the style of `test_cdna.hip`: the f16/bf16 modes of `test_cdna.hip` unchanged (2000 random 16x16x16 problems per mode at three scales, plus an exact small-integer problem, against the lane-gather reference and float64), which on gfx1250 exercise `v_wmma_f32_16x16x32_{f16,bf16}`; and new INT4 modes (s4 x s4 and u4 x s4, 2000 random 16x16x64 problems each plus an extreme-value problem and a one-nonzero-per-row sign/nibble-order problem) that load the fragments exactly as `gemm_w4a4_block_wmma` does and call `mma_rocm::wmma_i32_16x16x64_int4`, checked bit for bit against `mma_m16n8kx_s32common` and an exact host product. Compiled for gfx1250 here: the binary has `v_wmma_f32_16x16x32_f16` (2), `v_wmma_f32_16x16x32_bf16` (1), `v_wmma_i32_16x16x64_iu8` (2) and no scratch. Run on this gfx950 (`-DNTHREADS=64`, CDNA paths): PASS, INT4 reference 0/512000 wrong in both modes (so the host packing matches the fragment layout), INT4 WMMA "not compiled for this target" as designed.

### 4. Tests on gfx950

- `pytest tests/ops` on the five-target in-place build of this round: **75 passed** (the 52 of 75e2528 plus the 23 new precision tests).
- `mma_dump.py` and `f16_dump.py`, 75e2528 against this round, both five-target builds, gfx950: every GEMM output, packed next-layer activation, output scale, attention and AWQ output **bit-identical**; only the 10 fp32 `lora_act_out` tensors differ (max rel 2.2e-7 / 2.4e-7, the float-atomic noise; two runs of the new build differ in the same tensors, max rel 3.6e-7).
- Documented install (above), then one command from that tree: `HF_HOME=<job>/hf NUNCHAKU_TEST_CACHE_ROOT=<job>/test_cache python -m pytest -v -s tests/ops tests/v1/sdxl/test_sdxl.py::test_sdxl_lpips tests/v1/sdxl/test_sdxl_turbo.py::test_sdxl_turbo_lpips tests/sana tests/v1/z_image` -> **85 passed** in 508 s: SDXL LPIPS 0.151 (expected 0.25), SDXL-Turbo 0.229 (0.25), Z-Image Turbo INT4 rank 32/128/256 0.390/0.366/0.374 (expected int4-bf16 0.40/0.38/0.37, limit x1.15), both Sana example scripts, the three `_turing` variants returning early (bf16). The 16-bit SDXL references were generated on this host by the tests; the Z-Image references are upstream's.
- **NVFP4 through the public API, no plugin**: `python -m pytest -v -s test_zimage_fp4_public.py` (same tree): upstream's `test_zimage_turbo` unchanged, with only the test module's own `precision` variable (which chooses the checkpoint file name and the threshold) set to `fp4`; nunchaku is not patched, `NunchakuZImageTransformer2DModel.from_pretrained` chose `nvfp4` itself (recorded and asserted). **3 passed** in 183 s: rank 32 LPIPS **0.362** (expected fp4-bf16 0.33, limit x1.15), rank 128 **0.285** (expected 0.30); rank 256 has no NVFP4 checkpoint and returns early. (gfx950 round with the plugin: 0.349 / 0.293.)
- The two runs above used a build of the tree before its last two edits (the gfx1250-only launch bound and the restored `using` position in the gfx12 branch; neither reaches gfx950). **Rerun on the final commit 555105f**: documented install (389 s), then the same two commands: **85 passed** in 282 s (SDXL 0.151, SDXL-Turbo 0.229, Z-Image Turbo INT4 rank 32/128/256 0.371/0.358/0.381, both Sana scripts) and the NVFP4 test **3 passed** in 192 s (loader chose `nvfp4`; rank 32 LPIPS **0.347**, rank 128 **0.297**). Z-Image LPIPS moves by a few hundredths between runs of the same build (the LoRA-down sums use float atomics), inside the limits each time. Also on the final tree's five-target in-place build: `pytest tests/ops` 75 passed, and `mma_dump.py`/`f16_dump.py` against 75e2528 bit-identical apart from the 10 `lora_act_out` tensors (max rel 2.6e-7).

### 5. Recipe for the MI450 job (gfx1250)

In order; every GPU command under `timeout` with `ulimit -c 0`; record exact numbers in notes.md.

1. Venv: `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel,device-gfx1250]==10.2.0" "torch==2.14.0+rocm10.2.0" "torchvision==0.29.0+rocm10.2.0" "amd-torch-device-gfx1250==2.14.0+rocm10.2.0"` (both exist on that index), `rocm-sdk path --root` once, check `torch.utils.cpp_extension.ROCM_HOME` points into the venv, then the test dependencies of section "Environment". Check SDPA (`torch/lib/aotriton.images` present; a flash/memory-efficient SDPA call in fp16 and bf16 works); if it fails with `hipErrorInvalidImage`, use the gfx1100 round's `sitecustomize.py` that disables those two backends, and say so.
2. Clone the fork at 555105f with submodules into three trees (the build trap above: never rebuild in place after an edit).
3. **Primitive test first**: build the extension once in tree A (step 4) for `common_hip.h`, then `t1250/build.sh <tree A> gfx1250` and `REQUIRE_INT4=1 timeout 300 ./test_gfx1250`. Expected `PASS` with both INT4 lines `exact` and all f16/bf16 modes within the bound (report the exact-integer problem's inexact count: gfx11 WMMA was 16/256 off, gfx12 and CDNA 0). If INT4 is wrong, the operand layout assumption is (k = 16 * (v / 8) + 8 * (l / 16) + v % 8 in `wmma_i32_16x16x64_int4`); if f16 is wrong, the zero-padded upper half or the result layout is.
4. Builds, `MAX_JOBS` as exported, `python setup.py build_ext --inplace` each: tree A default `PYTORCH_ROCM_ARCH=gfx1250` (about 5 minutes here); tree B `NUNCHAKU_ROCM_MMA_REFERENCE=1` (about 50 minutes here); tree C INT4-only: `sed -i 's/^#define NUNCHAKU_ROCM_WMMA_F16 1$//' src/kernels/zgemm/mma_rocm.cuh` before building (16-bit products on the reference path, INT4 on WMMA; checked to compile here), never committed.
5. Bit identity of the INT4 WMMA: `python mma_dump.py dump` with B and with C, `python mma_dump.py compare B.pt C.pt`: expected only the 10 `lora_act_out` outputs differ (float atomics; confirm with two runs of B).
6. 16-bit products: `f16_dump.py dump` with B and A, `f16_dump.py compare B.pt A.pt` and `reldiff.py`: within the RDNA round's tolerance (max |new - ref| / max |output| <= 9.8e-4 float16, 7.8e-3 bfloat16), each build's error against float64 the same to about three digits; determinism (two runs of A: only `lora_act_out` differs). Also `mma_dump.py compare B.pt A.pt` (expected: GEMM outputs within that tolerance, as on CDNA/RDNA).
7. `pytest tests/ops` with A: 75 expected (the 14 FP4 tests on the native scale16 WMMA, not run anywhere yet).
8. If any kernel fails to launch (out of resources, invalid configuration), the 256-thread launch bound is the suspect (up to 511 VGPRs per wave): rebuild A with the `__launch_bounds__(256)` block in `gemm_base.cuh` removed, rerun, and report both.
9. End to end with A, installed by the documented command `PYTORCH_ROCM_ARCH=gfx1250 pip install -e . --no-build-isolation`: `python -m pytest -v -s tests/ops tests/v1/sdxl/test_sdxl.py::test_sdxl_lpips tests/v1/sdxl/test_sdxl_turbo.py::test_sdxl_turbo_lpips tests/sana tests/v1/z_image` (INT4 set; thresholds as above), then the FP4 test `python -m pytest -v -s test_zimage_fp4_public.py` (no plugin; expects the loader to choose nvfp4).
10. Speed against the reference path: `speed.py` (W4A4 GEMM 4096x3072x3072 rank 32/128 fp16/bf16, AWQ 512x4096x4096, attention 24 x 4608) with A and B, `fp4_speed.py` with A (INT4 vs NVFP4 kernels), `zimage_speed.py 32 int4 fp4` and `128 int4 fp4` with A: per-image latency and LPIPS decide Q3 for gfx1250 (`auto` stays int4 unless FP4 is at least as good in both).
11. If everything passes: gfx1250 can be named in the installation guide (NVFP4 and the WMMA list) and added to `setup.py`'s `NUNCHAKU_INSTALL_MODE=ALL` list (its build is now 5 minutes, the reason for leaving it out is gone); both are fork changes for the porter.

### Files of this round that a CUDA build reads (for the nvcc compile check)

- `src/kernels/zgemm/gemm_base.cuh`: the only change is inside `#if defined(__gfx1250__)` (never defined by nvcc).
- `src/kernels/zgemm/gemm_w4a4.cuh`: changes only inside `gemm_w4a4_block_wmma`, which is inside `#ifdef NUNCHAKU_ROCM_WMMA_INT4` (defined only in `mma_rocm.cuh`).
- Python a CUDA install runs: `nunchaku/utils.py` (new `get_checkpoint_precision`), `nunchaku/models/transformers/transformer_zimage.py`, `transformer_qwenimage.py`, `transformer_flux_v2.py`, `nunchaku/models/unets/unet_sdxl.py` (loader precision); new `tests/ops/test_checkpoint_precision.py` (runs on CUDA too, no GPU needed).
- Not read by a CUDA build: `src/kernels/zgemm/mma_rocm.cuh` (included only under `USE_ROCM`), `docs/source/installation/installation.rst`.
- Expected from the nvcc build: SASS identical to 75e2528 apart from `__LINE__` literals of any assert that sits below the inserted lines (5 lines at line 901 of `gemm_base.cuh`, the `#if defined(__gfx1250__)` lines in `gemm_w4a4_block_wmma`), plus the two `utils.cuh` asserts already known; the Python behavior matrix of the earlier checks (`pycheck.py`) can add `get_checkpoint_precision` (expected: equal to `get_precision()` for every faked capability and config) and the four loaders (`loader_matrix.py` faked CUDA rows: unchanged).

### What remains before `ported` (stage stays `porting`)

- **gfx1250 run** on the MI450 (recipe above): proves the INT4 iu8 and 16-bit WMMA paths of 555105f, the launch bound, and the FP4 scale16 path of 75e2528 (surfaces `kernel:stage2-gfx1250`, `kernel:gemm_w4a4-nvfp4-gfx1250`).
- **nvcc recheck** of 555105f on the mi355 host (files above).
- **jeff's ruling on the gfx90a f16 MFMA subnormal flush** (gfx90a evidence run, 2026-11-24): still open.
- **Revalidation**: 555105f changes no gfx90a/gfx942/gfx950/gfx1100/gfx1201 code object (section 2), and Q1 changes only the NVFP4 load on gfx950/gfx1250, so the CDNA and RDNA evidence at 75e2528 carries over by the identity proof; whether that is recorded as carry-forward is for the reviewer and validators.
- Windows gfx1151 last, per the standing ruling.

### Questions for jeff (nothing above waits on them)

1. **The C++ FLUX loader has the same problem as the v1 loaders had, worse**: `NunchakuFluxTransformer2dModel.from_pretrained` (`transformer_flux.py:573`) overwrites the precision with `get_precision(device=device)` for single-file checkpoints, even when the caller passes `precision="fp4"`, then calls `check_hardware_compatibility`, which accepts an NVFP4 checkpoint on gfx950, and builds the model with `use_fp4=False`; so an NVFP4 FLUX checkpoint cannot be loaded through the C++ model on gfx950 at all. On CUDA, whenever that check passes the checkpoint's precision equals `get_precision()`, so changing line 573 to `precision = get_checkpoint_precision(quantization_config)` (after the config is read) would leave CUDA unchanged. Options: (a) make that one-line change too; (b) leave the C++ loader as it is (the ruling named the v1 loaders). The Sana loader honors an explicit `precision=` argument and has no NVFP4 checkpoint, so it needs nothing.
2. The upper half of k in the gfx1250 16-bit WMMA is zero-padded (section 2). Whether to fuse pairs of k steps in the callers (LoRA, attention, AWQ) to fill it is for after the MI450 speed numbers; no action now.

## Lesson for cuda-to-rocm: AMD clang can emit different code for the same source on large gfx11/gfx12 kernels

(For `validation.md`, where it recommends comparing code objects to show a change does not touch other targets.) With ROCm 10.2's AMD clang 24.0.0git (bdc35841c97f), compiling one unchanged `.hip` source twice with the same command for gfx1100 or gfx1201 gave code objects whose disassembly differed: a few dozen instructions in other order, other registers and `s_delay_alu` hints, inside very large WMMA GEMM kernels (2 to 3.6 million disassembly lines per object). CDNA targets and smaller objects came out identical every time. A "disassembly identical before and after" check therefore reports false differences there. Compare the compiler input instead: the device-only preprocessed source per target (`hipcc -E -P --cuda-device-only --offload-arch=<gfx>` with the build's flags, `-fmacro-prefix-map=<tree>=<fixed>` so `__FILE__` matches), and build both versions at the same path so path strings and `__hip_cuid_*` do not differ. Seen 2026-11-25 (nunchaku).

## Lesson for cuda-to-rocm: gfx1250 kernels without launch bounds are limited to 128 VGPRs; with them they can use up to 512

(For `platform-limits.md`, registers, and `no-hip-equivalent.md`, gfx1250 WMMA.) Without `__launch_bounds__` the compiler must assume 1024-thread blocks and gives every gfx1250 kernel at most 128 VGPRs per wave; nunchaku's GEMM, attention and AWQ kernels all sat at exactly 128 and spilled (1.86 million scratch instructions in one object). Declaring the real block size (`__launch_bounds__(256)`, guarded by `#if defined(__gfx1250__)` to leave other targets alone) let the compiler use up to 511 VGPRs per wave, addressing registers above 255 with `s_set_vgpr_msb`, and the spill code fell to 8157 instructions (attention: none). So a gfx1250 wave can address more than 256 VGPRs. Also: gfx1250 has no 4-bit integer WMMA (the iu4 builtins need `wmma-128b-insts`), so INT4 products go through `v_wmma_i32_16x16x64_iu8` with the values widened to bytes, which is exact; its operand layout is lane l = row l % 16, byte v at k = 16 * (v / 8) + 8 * (l / 16) + v % 8, the same as the 16-bit 16x16x32 forms per element. Seen 2026-11-25 (nunchaku), compiled only; not yet run on an MI450.

## Ruling 2026-11-25 (Jeff Daily): gfx90a f16 MFMA subnormal flush

(a) Accept the hardware flush on gfx90a; no alternate mode. nunchaku is inference only, and PyTorch's own default for MI200 forward passes is the original (flushing) FP16 instruction, using rocBLAS/MIOpen's alternate implementation only in the backward pass. Owed in the next porter round: the primitive test (`test_cdna.hip`) zeroes subnormal f16 inputs on gfx90a only, citing the documented MI200 behavior, and keeps the strict check on every other target; the docs say that on MI200 the f16 products flush subnormal inputs to zero, as PyTorch's FP16 GEMMs there do in inference.

## Evidence run 2026-11-25, gfx1250 (MI450) at 555105f (linux-gfx1250 fleet job, for jeff)

Not a porter round: no work lock taken, stage untouched, nothing changed in or pushed to the fork. Scope: the "Recipe for the MI450 job" of the gfx950 round above, run in its order. **Result: everything in the recipe passes on gfx1250 once torch's own SDPA is worked around (the recipe's step-1 fallback).** The INT4 iu8 WMMA path is bit-identical to the reference, the 16-bit WMMA path is within tolerance, the native FP4 `v_wmma_scale16` path passes all 14 FP4 op tests (so the scale-lane and nibble-order assumptions hold), the 256-thread launch bound launches everywhere, and the NVFP4 checkpoint loads through the public loader.

### Environment

- Host: one MI450, `gfx1250`, wave32, `HIP_VISIBLE_DEVICES=0` as pinned by the dispatcher, `ulimit -c 0`, every GPU command under `timeout`. **The GPU is A0 silicon** (`rocminfo` ASIC Revision 0, `amd-smi` REV_ID 0x00, product name "AMD Eng Sample: 100-000001046-04"); the ISA the runtime reports is plain `amdgcn-amd-amdhsa--gfx1250`.
- GPU-free check before every GPU run: `sudo -n fuser -v /dev/kfd` and `sudo -n lsof /dev/kfd` (psmisc and lsof installed with apt in this container; they were missing). Every check found no holder. Limitation: the container has its own PID namespace (8 processes visible), so those tools only see this container's processes, not other teams' processes on the host; no host-level check was possible from inside the job.
- Job venv `$FLEET_VENV` (Python 3.11.17), `$FLEET_VENV/bin` first on PATH: `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel,device-gfx1250]==10.2.0" "torch==2.14.0+rocm10.2.0" "torchvision==0.29.0+rocm10.2.0" "amd-torch-device-gfx1250==2.14.0+rocm10.2.0"`. `rocm-sdk path --root` (run once) printed `$FLEET_VENV/lib/python3.11/site-packages/_rocm_sdk_devel`; `torch.utils.cpp_extension.ROCM_HOME` is the same directory. **Compiler actually used**: all 17 device compile lines of each of the four builds (A, B, C, and the documented install of A) invoke `$FLEET_VENV/lib/python3.11/site-packages/_rocm_sdk_devel/bin/hipcc` (HIP 7.17.26434, AMD clang 24.0.0git, ROCm/llvm-project bdc35841c97f); no `/opt/` path in any log. Test dependencies as in the gfx950 round (accelerate 1.9.0, diffusers 0.36.0, transformers 4.55.2, peft 0.17.1, datasets 3.6.0, torchmetrics 1.8.2, huggingface-hub 0.34.6, controlnet-aux 0.0.10, image-gen-aux e08ae3d, pytest 8.4.2, numpy 2.4.6, pandas 3.1.1, matplotlib, lpips; scikit-image/scikit-learn/scipy/PyWavelets upgraded; opencv-python-headless removed, opencv-python 4.11.0.86 reinstalled). Model downloads with `HF_HUB_DISABLE_XET=1`.
- **torch environment problems on this A0 part, both outside the port:**
  - hipBLASLt in the 10.2 wheels has no A0 GEMM library: every torch fp16/bf16 matmul through hipBLASLt fails (`rocblaslt error: gfx1250 device with asicRevision 0 (A0) needs the gfx1250v0 GEMM library, but ".../hipblaslt/library/gfx1250v0" holds no Tensile library ... Not falling back to ".../hipblaslt/library/gfx1250": its kernels give wrong results on A0`, then `HIPBLAS_STATUS_INVALID_VALUE when calling hipblasLtMatmulAlgoGetHeuristic`; `gfx1250v0/` holds only a few B8 libraries). rocBLAS works (fp16/bf16/fp32 512x384x256 matmul against float64: max rel 3.4e-4 / 2.5e-3 / 1.0e-6). **Every torch process of this run set `TORCH_BLAS_PREFER_HIPBLASLT=0`** (rocBLAS for torch's own GEMMs: the 16-bit reference pipelines, text encoders, VAE). The nunchaku kernels do not use either library. The hipBLASLt error lines are still printed at start-up.
  - torch SDPA flash and memory-efficient backends fail on gfx1250 for head_dim 512 with `torch.AcceleratorError: CUDA error: profiler is not initialized` (`hipErrorProfilerNotInitialized`), fp16 and bf16, sequence 256, 4096 and 16384; the math backend works; head_dim 64, 128 and 256 work (the head_dim 64 check against float32: flash and memory-efficient max error 1.8e-3 fp16, 1.4e-2 bf16). `torch/lib/aotriton.images` is present. Smallest reproducer (`sdpa_vae.py` below): `q = torch.randn(1, 1, 256, 512, device="cuda", dtype=torch.bfloat16); F.scaled_dot_product_attention(q, q, q)`. The diffusers VAE decoder's mid-block attention (1 head, 512 channels) hits it, so the first end-to-end run failed 5 tests (below). For the end-to-end and Z-Image speed runs the recipe's fallback was used: a `sitecustomize.py` on `PYTHONPATH` calling `torch.backends.cuda.enable_flash_sdp(False)` and `enable_mem_efficient_sdp(False)` (checked: both report False in a fresh interpreter). No source change.
- Fork trees: three clones of https://github.com/AMD-Ecosystem/nunchaku at **555105f** with submodules at the recorded commits: A default, B `NUNCHAKU_ROCM_MMA_REFERENCE=1`, C INT4-only (`sed -i 's/^#define NUNCHAKU_ROCM_WMMA_F16 1$//' src/kernels/zgemm/mma_rocm.cuh`, never committed). A and B `git status --porcelain --ignore-submodules=all` clean after every run.
- Builds, `PYTORCH_ROCM_ARCH=gfx1250 python setup.py build_ext --inplace`, all three at once, detached, wrapped in `timeit.sh` (`stats.jsonl`), 32 + 64 + 32 jobs (= the exported MAX_JOBS=128): **A 287 s, B 1649 s, C 394 s**, all exit 0. Documented install of A, `PYTORCH_ROCM_ARCH=gfx1250 MAX_JOBS=64 python -m pip install -e . --no-build-isolation -v`: exit 0, 287 s, `nunchaku-1.3.0.dev20261125+rocm10.2torch2.14`. The dumps of A below used A's in-place build; ops, end to end and speed used the installed build (same source and flags).
- Code objects of A (`llvm-objcopy --dump-section .hip_fatbin`, `clang-offload-bundler --unbundle --targets=hip-amdgcn-amd-amdhsa--gfx1250`, `llvm-objdump -d --mcpu=gfx1250`): `gemm_w4a4_launch_bf16_fp4` 2688 `v_wmma_scale16_f32_16x16x128_f8f6f4` + 4864 `v_wmma_f32_16x16x32_bf16`; `gemm_w4a4_launch_fp16_int4` 5376 `v_wmma_i32_16x16x64_iu8` + 4656 `v_wmma_f32_16x16x32_f16`, as the porter round counted. (Trap: `llvm-objcopy --dump-section` with no output file rewrites its input in place; it touched two `.o` files in A's build directory after the link, which changed nothing that ran.)

### Results (recipe steps in order)

1. **Primitive test** (`t1250/build.sh <tree A> gfx1250`, `REQUIRE_INT4=1 timeout 300 ./test_gfx1250`): **PASS**, exit 0.
   - `int4 s4 x s4, 2000 problems: reference wrong 0/512000, iu8 wmma exact`; `int4 u4 x s4`: the same. The operand layout assumption of `wmma_i32_16x16x64_int4` (k = 16 * (v / 8) + 8 * (l / 16) + v % 8) holds.
   - f16/bf16 modes, all three scales: exact-integer case **inexact 0/256** for the matrix path in every mode (as on gfx12 and CDNA; gfx11 was 16/256), reference wrong 0, over bound 0 for both. Max |err|/sum|terms| of the matrix path: mode 0 (f32 acc, f16) 1.27e-7 to 1.33e-7; mode 1 (f32 acc, bf16) 9.5e-8 to 1.05e-7; mode 2 (f16 acc) 4.5e-4 to 4.59e-4 (reference 4.5e-4 to 4.59e-4). Bitwise identical to the reference: mode 0 about 39%, mode 1 about 68%, mode 2 99.96% of 512000 outputs (the rest are fp32 rounding-order differences within the bound).
2. **Bit identity of the INT4 WMMA** (`mma_dump.py`, B against C): **34 cases, 10 outputs differ, exactly the 10 fp32 `lora_act_out` tensors** (`fused[3]`, `quantize[2]`; 5464 to 26128 differing bytes). Two runs of B differ in the same 10 tensors (5557 to 25050 bytes), and two runs of A likewise (max rel 2.9e-7). Every GEMM output, packed next-layer activation and output scale is bit-identical.
3. **16-bit products** (`f16_dump.py` / `reldiff.py`, B against A): within the RDNA tolerance. Worst max |new - ref| / max |output|: **float16 5.98e-4** (attention 4608x4608; bound 9.8e-4), **bfloat16 4.03e-3** (attention 512x500; AWQ 512x4096x4096 4.0e-3; bound 7.8e-3). Per kind: GEMM outputs at most 1 ulp (largest relative 2.44e-3, case gemm_torch.bfloat16_True_4352x256x3072), fused outputs at most 8 ulp, packed activations at most 2 of 3.1M nibbles one step apart, attention up to 1888 ulp f16 / 28251 ulp bf16 on near-zero values, AWQ up to 161 ulp. Error against float64 is identical to three digits between the builds in every attention and AWQ case (e.g. attention f16 4608: 0.00239 both; AWQ bf16 1024x1024x4096: 0.00318 both). Determinism: two runs of A differ only in `lora_act_out` (max rel 2.9e-7 / 1.6e-7); `mma_dump.py compare B A` shows the same picture (GEMM outputs differ by at most a few ulp in under 0.03% of elements, from the 16-bit LoRA epilogue).
4. **`pytest tests/ops`** with A: **75 passed** in 34 s, 0 skipped (the 52 of 75e2528 plus the 23 precision tests of 662be3a; the dispatch said 52, which predates 662be3a). **The 14 FP4 tests, `test_gemm_w4a4_fp4` (6), `test_quantize_w4a4_act_fp4` (4), `test_svdq_w4a4_linear_nvfp4` (4), all pass on the native `v_wmma_scale16` instruction**: the scale-lane and nibble-order assumptions of the native FP4 path are right on hardware.
5. **Launch bound** (recipe step 8): no kernel failed to launch in any run, so the rebuild without `__launch_bounds__(256)` was not needed. The up-to-511-VGPR kernels run on the MI450.
6. **End to end, INT4 set** (`python -m pytest -v -s tests/ops tests/v1/sdxl/test_sdxl.py::test_sdxl_lpips tests/v1/sdxl/test_sdxl_turbo.py::test_sdxl_turbo_lpips tests/sana tests/v1/z_image`, installed A):
   - First run without the SDPA workaround: **80 passed, 5 failed** in 270 s. The failures were `test_sdxl_lpips`, `test_sdxl_turbo_lpips` and `test_zimage_turbo[32/128/256]`, all raising `hipErrorProfilerNotInitialized` inside torch's `F.scaled_dot_product_attention` in the diffusers VAE decoder of the 16-bit reference pipeline (`autoencoder_kl.py:208 _decode` -> `unet_2d_blocks.py:745` -> `attention_processor.py`). That is the torch SDPA problem above, not a nunchaku kernel.
   - Rerun with the `sitecustomize.py`: **85 passed** in 510 s. LPIPS: SDXL **0.144** (threshold 0.25), SDXL-Turbo **0.214** (0.25), Z-Image Turbo INT4 rank 32/128/256 **0.379 / 0.347 / 0.355** (expected int4-bf16 0.40 / 0.38 / 0.37, limit x1.15). Both Sana example scripts returned 0; the three `_turing` variants return early (bf16). The 16-bit SDXL references were generated on this host by the tests, with rocBLAS and math SDPA; the Z-Image references are upstream's.
7. **Z-Image NVFP4 through the public loader** (`python -m pytest -v -s test_zimage_fp4_public.py`, same tree, same workaround, no plugin): **3 passed** in 119 s. The loader chose `nvfp4` itself (recorded `['nvfp4']` for ranks 32 and 128; `[]` for rank 256, which has no NVFP4 checkpoint and returns early). LPIPS rank 32 **0.342** (expected fp4-bf16 0.33, limit 0.3795), rank 128 **0.287** (expected 0.30).
8. **Speed** (A = default WMMA build, B = reference lane-gather build; median timings as the scripts print them):

| kernel | A (WMMA) | B (reference) | B / A |
|---|---|---|---|
| W4A4 GEMM fp16 4096x3072x3072 rank 32 | 0.534 ms | 2.037 ms | 3.8x |
| W4A4 GEMM fp16 rank 128 | 0.567 ms | 2.156 ms | 3.8x |
| W4A4 GEMM bf16 rank 32 | 0.490 ms | 9.586 ms | 19.6x |
| W4A4 GEMM bf16 rank 128 | 0.562 ms | 9.818 ms | 17.5x |
| AWQ fp16 512x4096x4096 | 1.523 ms | 2.881 ms | 1.9x |
| AWQ bf16 512x4096x4096 | 1.429 ms | 2.919 ms | 2.0x |
| attention fp16 24 heads x 4608 | 4.973 ms | 22.516 ms | 4.5x |

   `fp4_speed.py` (A, bf16, rank 32): INT4 against NVFP4 W4A4 GEMM 4096x3072x3072 0.490 / 0.944 ms (FP4 1.93x slower), 4096x12288x3072 1.421 / 2.713 ms (1.91x), 4096x3072x12288 1.721 / 3.424 ms (1.99x). `zimage_speed.py` (A, 1024x1024, 9 steps, bf16, three images each, SDPA workaround on): rank 32 INT4 **5.00 s** per image (transformer 4.77 s) against NVFP4 **6.93-6.96 s** (6.70-6.73 s), 1.39x; rank 128 INT4 **5.37 s** (5.14 s) against NVFP4 **7.65-8.64 s** (7.40-8.40 s), 1.43-1.61x.
   **Q3 for gfx1250**: NVFP4 has the better LPIPS at rank 32 (0.342 vs INT4 0.379) and rank 128 (0.287 vs 0.347), but is 1.4-1.6x slower per image and about 1.9x slower per GEMM. By the recipe's rule (`auto` stays int4 unless FP4 is at least as good in both), `auto` stays `int4` on gfx1250.
9. **Recipe step 11 (not done here, porter work):** everything passed, so gfx1250 can be named in the installation guide (NVFP4 and the WMMA list) and added to `setup.py`'s `NUNCHAKU_INSTALL_MODE=ALL` list (its build took 287 s here). Both are fork changes for the next porter round. Whether the docs should mention the torch SDPA and hipBLASLt problems on A0 parts is a question for jeff; neither touches nunchaku code.

### Scripts of this run (`agent_space`, not committed; the recipe's scripts unchanged apart from job paths)

- `t.sh` additionally exports `TORCH_BLAS_PREFER_HIPBLASLT=0 HF_HUB_DISABLE_XET=1` and runs a `/dev/kfd` holder check (`fuser`, `lsof`) before each command, refusing to start if one is found.
- `sitecustomize.py` (on `PYTHONPATH` for the end-to-end and Z-Image speed runs): `import torch`, `torch.backends.cuda.enable_flash_sdp(False)`, `torch.backends.cuda.enable_mem_efficient_sdp(False)`. (`timeit.sh`'s system Python prints "Error in sitecustomize ... No module named 'torch'" at the end of each such run; harmless.)
- `sdpa_vae.py <dtype> <seq> <head_dim> <any|flash|eff|math>`: one `F.scaled_dot_product_attention(q, q, q)` on `torch.randn(1, 1, seq, head_dim)` under the chosen backend. Results above (fails for head_dim 512 with any non-math backend).

### What this changes in "What remains before `ported`"

- gfx1250 run: **done**, all recipe checks pass at 555105f (surfaces `kernel:stage2-gfx1250`, `kernel:gemm_w4a4-nvfp4-gfx1250` proven on hardware). Owed to the porter: the docs and `setup.py` items of step 9.
- Still open: the nvcc recheck of 555105f on the mi355 host, the gfx90a test/doc change from the subnormal ruling, revalidation/carry-forward, Windows last.

## Lesson for cuda-to-rocm: TheRock torch 2.14 on gfx1250 A0 parts: hipBLASLt has no A0 library and SDPA fails at head_dim 512

(For `diagnosis.md`, where it tells you to rule out the framework before blaming the port.) On an MI450 engineering sample (`rocminfo` ASIC Revision 0) with `torch==2.14.0+rocm10.2.0` and `amd-torch-device-gfx1250` from `stable.repo.amd.com/rocm/whl-next`, two torch features fail on their own, with no project code involved: every fp16/bf16 matmul through hipBLASLt (`needs the gfx1250v0 GEMM library ... holds no Tensile library`, then `HIPBLAS_STATUS_INVALID_VALUE`; set `TORCH_BLAS_PREFER_HIPBLASLT=0` to use rocBLAS, which works), and `scaled_dot_product_attention` with the flash or memory-efficient backend at head_dim 512 (`hipErrorProfilerNotInitialized`, "profiler is not initialized", fp16 and bf16; head_dim up to 256 works, the math backend works). Diffusers VAE decoders use one 512-channel attention head, so every image pipeline fails at decode. Check both with a three-line script before reading a test failure as a port failure; to run tests anyway, disable the two SDPA backends from a `sitecustomize.py` on `PYTHONPATH`. Seen 2026-11-25 (nunchaku).
