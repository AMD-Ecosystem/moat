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

## Lesson for cuda-to-rocm: hipify turns `std::max`/`std::min` into HIP's non-constexpr device functions

(For `strategy-b-torch.md`, hipify section.) torch's build-time hipify rewrites `std::max(` and `std::min(` to `::max(` and `::min(`, HIP's `__device__` overloads. Runtime uses still compile, but a compile-time use breaks with "constexpr variable must be initialized by a constant expression" or "reference to __device__ function 'min' in global initializer" (nunchaku: `static constexpr int X = std::min(WARP_SIZE, WARP_N / PACK)` in a class template, and `constexpr int unroll = 16 / std::max(sizeof(a), sizeof(b))`). Fix without touching behaviour: give the template argument explicitly, `std::min<int>(...)`, which hipify's pattern does not match. Seen with ROCm 10.2 / torch 2.14 (nunchaku).

## Lesson for cuda-to-rocm: a shared-memory staging buffer for matrix-instruction results counts against the 64 KB with the kernel's other shared memory

(For `platform-limits.md`, shared memory.) When a ported GEMM stages MFMA/WMMA results through LDS to convert them back to an mma.sync accumulator layout, a static `__shared__` buffer adds to every kernel instantiation, including those whose epilogue already takes large dynamic shared memory; on 64 KB parts the launch then fails with `hipFuncSetAttribute(MaxDynamicSharedMemorySize) ... invalid argument` only for those epilogues (nunchaku: 18 KB staging + 42 KB LiteLA epilogue + 17 KB other statics). Put the staging buffer in the dynamic region, have the launcher always request at least its size, and add a block barrier before the epilogue reuses it.
