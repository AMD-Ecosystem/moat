# nunchaku port plan

Planner run 2026-11-20 on linux-gfx90a (no GPU used, nothing built), for jeff. Analysed at upstream `main` 302e0e97024ebd68688fe890e5df83731edf7b54 with submodules.

**Rulings (Jeff Daily, 2026-11-23, recorded in `notes.md`): Q1 = c, Q2 = a, Q3 = b.** The planner wrote this plan for Q1 = a and Q3 = a; the porter updated it on 2026-11-24 (linux-gfx942) so that every section below states the ruled scope:

- Q1 = c: INT4 on every platform. Native NVFP4 only where the hardware has FP4 matrix instructions (gfx950 scaled MFMA `f8f6f4`, gfx1250 WMMA f4), with the NVFP4 scales (ue4m3, block 16) handled in software there. Every other AMD target traps on FP4 exactly as upstream does below sm_120, and `get_precision()` returns `fp4` only on gfx950 and gfx1250.
- Q2 = a: `gemm_w8a8_fp16` (CUTLASS) is left out of the ROCm source list; `gemm_batched_fp16` gets a hipBLAS strided-batched implementation; `mha_fwd_block` raises "Block-sparse attention is not supported on ROCm".
- Q3 = b: the stage 2 matrix-instruction paths (RDNA WMMA, CDNA MFMA) are required on every gate before review, whatever stage 1's speed. Stage 1 stays in the tree as the exact reference each stage 2 path is proven bit-identical against.

## Project

- Name: nunchaku (SVDQuant 4-bit diffusion inference engine)
- Upstream: https://github.com/nunchux-ai/nunchaku (formerly mit-han-lab/nunchaku), default branch `main`
- Fork: https://github.com/AMD-Ecosystem/nunchaku, port branch `moat-port`
- Licence Apache-2.0 (intake). Submodules: cutlass (BSD-3), Block-Sparse-Attention (BSD-3, MIT HAN Lab), json, mio, spdlog (MIT).

## Existing AMD support

None (intake: no upstream PR, branch, issue work or third-party port; issue #75 closed with "Supporting AMD would require rewriting all the kernels. We do not plan to support this for now."). No authoritative or community port to inherit. Decision: port from scratch.

Delivery vehicle: upstream does not link platform forks and has not refused contributions, so an upstream PR stays the default target; the maintainers' stated lack of interest makes the fork the likely long-lived deliverable (a `set-maintained` ruling later, if they decline). Nothing in this plan depends on that outcome.

Port versus rewrite: **layout-preserving primitive port first, AMD-native performance pass later.** Reasons, in order of weight:

1. The NVIDIA `mma.sync` fragment layout is the data contract of the whole project, not an implementation detail of the kernels. The published quantized checkpoints (`nunchaku-tech/*/svdq-int4_r32-*.safetensors`) store weights, weight scales and LoRA factors pre-packed in that layout (`gemm_base.cuh`: "weights need to be repacked when the tiling size changes"). The Python side also packs tensors in it at run time: `nunchaku/lora/flux/packer.py` (`MmaWeightPackerBase`, LoRA conversion), `nunchaku/models/embeddings.py:130` and `nunchaku/models/transformers/transformer_flux.py:84` (rotary embeddings, "mma-16816-c" layout), `nunchaku/caching/utils.py:311`. An AMD-native layout would mean a load-time repacker for every checkpoint format plus a second Python packer, and a fork of the model format.
2. The PTX is concentrated. Every tensor-core instruction is reached through a handful of device helpers (`zgemm/mma.cuh`, `zgemm/mma_earlycuda.cuh`, `zgemm/gemm_utils.cuh`, `GEMM_W4A4::mma`, `GEMM_W8A8::mma`), and the 3000+ lines of tiling, epilogue, LoRA and attention code above them are plain CUDA C++ indexed by a logical 32-lane warp. Reimplementing the helpers on AMD keeps all of that shared with the CUDA path.
3. Integer accumulation is exact, so every INT4/INT8 helper can be proven bit-identical against a scalar reference on random data, which makes a later matrix-core pass safe to land.

A later AMD-native pass (MFMA/WMMA tiles chosen for AMD, CK or FlyDSL) is a legitimate follow-up for performance; it is not part of this port and is not owed by it.

## Build classification

**torch-extension** (recorded with `set-ext-type`). Evidence: `setup.py` imports `torch.utils.cpp_extension` and builds one `CUDAExtension` named `nunchaku._C` from 22 `.cu` and 10 `.cpp` sources (plus 10 Block-Sparse-Attention sources in non-debug builds); `pyproject.toml` build requirement `torch>=2.7`. No CMake.

## Port strategy

**Strategy B (torch build-time hipify)**, plus hand-written HIP implementations of the PTX helpers behind `USE_ROCM` guards. No compat header, no hand renaming.

- `setup.py` branches on `torch.version.hip`: skip `get_sm_targets()` and all nvcc-only flags (`-gencode`, `-Xcudafe`, `--expt-*`, `--ptxas-options`, `--threads`, `--generate-line-info`, `-U__CUDA_NO_*`), pass `-U__HIP_NO_HALF_OPERATORS__ -U__HIP_NO_HALF_CONVERSIONS__ -U__HIP_NO_BFLOAT16_CONVERSIONS__` (and the half2 forms), `-std=c++20`, `-O3` for device code, `-DUSE_ROCM`; architectures from `PYTORCH_ROCM_ARCH` (torch's own mechanism; default to the arch of the visible devices when unset, mirroring `NUNCHAKU_INSTALL_MODE=FAST`). Version tag `+rocm<hip major.minor>torch<x.y>` instead of `+cu<ver>`. Drop the Block-Sparse-Attention sources on ROCm (see below).
- Hipified mirror files go in `.gitignore` (strategy-b-torch.md, first section).
- Check whether hipify reaches `third_party/Block-Sparse-Attention/csrc/block_sparse_attn` headers through `include_dirs`; if the HIP build drops those sources, `flash_api.h` must still not be included on ROCm (guard in `FluxModel.cpp`, `SanaModel.cpp`).

## CUDA surface inventory

Census (`surface.py`, regex): 40 CUDA-bearing files under `src/`; 22 kernels, 45 launches, 28 warp-intrinsic sites, cuBLAS 18 sites, CUTLASS 195 sites. Inline `asm` about 95 sites in 10 files. `third_party/` is outside the census and is added by hand below.

### A. PTX helpers (the core of the port)

| helper | where | HIP implementation |
|---|---|---|
| `mma_m16n8kx_s32common<s4,s4>`, `<u4,s4>` (m16n8k64 INT4, s32 acc) | `zgemm/mma.cuh`, `GEMM_W4A4::mma` | layout-preserving reimplementation (see "Primitive layer") |
| `mma_m16n8kx_s32common` m8n8k32 (sm_75 form) | `zgemm/mma_earlycuda.cuh` | not used on ROCm: the HIP path always takes the `mma.cuh` signatures |
| s8 m16n8k32 | `zgemm/gemm_w8a8.cuh` `GEMM_W8A8::mma` | same primitive layer, s8 operands |
| `mma_m16n8k16_f16f16f16f16`, `mma_m16n8k16_f32f16f16f32<bf16>` | `zgemm/mma.cuh`, used by LoRA (`lora.cuh`), epilogues and `attention.cuh` | same primitive layer, f16/bf16 operands |
| `mma_fp4` (`kind::mxf4nvf4.block_scale.scale_vec::4X ... e2m1 ... ue4m3`), `quantize_float2_fp4`, `quantize_float4_fp8` | `zgemm/gemm_w4a4.cuh`, `gemm_utils.cuh` | Q1 = c: native on gfx950 (`v_mfma_scale_f32_*_f8f6f4` with FP4 operands) and gfx1250 (WMMA f4), NVFP4 ue4m3 block-16 scales applied in software (the MX hardware scale is e8m0 per 32); FP4 quantize through the gfx950/gfx1250 FP4/FP8 conversion instructions. Every other AMD target traps (`trap_no_fp4`) as upstream does below sm_120 |
| `ldmatrix.sync.aligned.{x4,x4.trans}.m8n8.shared.b16` | `gemm_utils.cuh`, `awq/gemm_awq.cu` (both forms) | plain LDS loads into the same fragment registers (pure data movement, bit-identical) |
| `movmatrix.sync.aligned.m8n8.trans.b16` | `gemm_utils.cuh` | width-32 `__shfl` transpose |
| `ld.shared.v2/v4`, `st.shared.v2/v4`, predicated `ld.global`/`st.global` | `gemm_utils.cuh` `load`/`store`/`load_pred`/`store_pred` | plain C++ vector loads, `if (pred)` |
| `cvt.rni.s32.f32` + `cvt.pack.sat.{s4,u4,s8}.s32.b32` | `gemm_utils.cuh` `quantize_float2<4|8,…>` | `__float2int_rn`, clamp, shift-and-or packing in the same nibble order |
| `cvt.rni.sat.s8.{f32,f16}` | `utils.cuh` | `__float2int_rn` + clamp |
| `tanh/rcp/rsqrt/sin/cos/ex2 .approx` | `gemm_utils.cuh` | `tanhf`, `__frcp_rn`/`__builtin_amdgcn_rcpf`, `rsqrtf`, `__sinf`, `__cosf`, `exp2f` |
| `red.relaxed.gpu.global.add.f32` | `gemm_utils.cuh` `reduce_add(_pred)` | `atomicAdd` (relaxed, agent scope; `unsafeAtomicAdd` is acceptable on coarse-grained memory) |
| `lop3.b32`, `prmt.b32` int-to-half tricks | `gemm_utils.cuh` `int2float_fast`, `int2half2_fast_{8192,4096_rn,512}`; `awq/dequantize.cuh` | equivalent bitwise expressions and `__byte_perm`; llm-awq's port already translated the same `dequantize.cuh` lop3 immLuts (reference only, not a dependency) |
| `cp.async.cg.shared.global` with L2 cache hint, `cp.async.wait_group` | `awq/gemm_awq.cu` | synchronous global-to-LDS copy |
| `ld.global.acquire`, `red.release` semaphore | `awq/semaphore.h` | `__hip_atomic_load/fetch_add` with acquire/release, agent scope |
| `__nanosleep` | `utils.cuh` `trap_unsupported_arch`, `gemm_w4a4.cuh` `trap_no_fp4` | `__builtin_amdgcn_s_sleep` loop or drop before `__trap()` |

### B. Primitive layer (new HIP-only header, e.g. `src/kernels/zgemm/mma_hip.cuh`)

Same function names and signatures as `mma.cuh` (`uint4 a, uint2 b, uint4 c` in the PTX per-lane fragment layout), selected where `mma.cuh` / `mma_earlycuda.cuh` are included (`gemm_base.cuh`) under `#if defined(USE_ROCM)`. The rest of the zgemm code calls it unchanged.

Stage 1, correctness reference (required, every arch): each lane gathers the A-row and B-column words its outputs need from the other lanes of its **logical 32-lane warp** with width-32 `__shfl`, and computes its PTX-layout outputs with scalar or packed dot products (`__builtin_amdgcn_sdot4` / `sdot8` / `udot` where the target has them, checked with `__has_builtin` and the target feature list; plain C otherwise). This is wavefront-width agnostic: on wave64 the two logical warps of a wavefront never exchange data. Bit-exact for INT4/INT8 by construction.

Stage 2, matrix cores, per arch family, behind `__GFX9__`/`__gfx90a__`/`__gfx942__`/`__gfx950__`/`__GFX11__`/`__GFX12__`, each proven **bit-identical to stage 1** on random inputs before it is enabled:

- RDNA wave32 (gfx11, gfx1151, gfx12): one logical warp is one wavefront. INT4 has a native target: `v_wmma_i32_16x16x16_iu4` (gfx11), the gfx12 `iu4` forms. f16/bf16: `v_wmma_f32_16x16x16_{f16,bf16}`. Shuffle PTX fragments into the WMMA operand layout and the result back (layouts in `no-hip-equivalent.md`).
- CDNA wave64 (gfx90a, gfx942, gfx950): no INT4 MFMA input. Unpack s4/u4 to s8 in registers (memory stays 4-bit) and use the i8 MFMA. Because each wavefront holds two logical warps with different operands, the 16x16 MFMA cannot be used per warp directly; the per-warp-safe form is the 4x4 block MFMA (`__builtin_amdgcn_mfma_i32_4x4x4i8` on gfx90a, the gfx94x 4x4 i8 form), sixteen independent 4-lane blocks. Whether a 16x16 form with one warp's operands broadcast per issue is faster is a performance-pass question, not this port's.
- Ruling Q3 = b: stage 2 is required on every gate before review (wave64 CDNA MFMA, wave32 RDNA WMMA), each proven bit-identical to stage 1 on that hardware with the build switch that selects stage 1 (the porter's notes name it). Speed is recorded, never gated.
- Porter correction (2026-11-24) to the CDNA bullet above: the W4A4 GEMM loads the same weight tile in every warp of a block (`load_wgt` has no warp index), so one 32x32 MFMA per wavefront can take the A rows of both 32-lane warps (rows 0-15 and 16-31) against two 16-column weight tiles with no wasted lanes. The CDNA stage 2 therefore uses the i8 32x32 MFMA (`v_mfma_i32_32x32x16_i8` on gfx942/gfx950, `v_mfma_i32_32x32x8i8` on gfx90a) over the whole GEMM block, not the per-warp 4x4 block form.

### C. Wavefront-width sites (fix on both widths; logical warp stays 32)

All zgemm and AWQ code indexes a logical 32-lane warp (`GEMMConfig_*::WARP_SIZE = 32`, `threadIdx.x % WARP_SIZE`, launches of `WARP_SIZE * NUM_WARPS` = 256 threads), which is wavefront-safe as long as every cross-lane op is width-32. Sites to fix:

- `__shfl_sync(~0, v, laneId % LANES_PER_HEAD)` and `__shfl_xor_sync(~0, …)` with **no width argument** in `zgemm/epilogues.cuh` (lines ~99, 125, 137) and `awq/gemv_awq.cu` (55-57): default width is `warpSize` = 64 on wave64; pass 32.
- 32-bit masks (`0xffffffff`, `~0`, `FINAL_MASK`) in `reduction_utils.cuh` (28, 38, 115), `epilogues.cuh`, `gemv_awq.cu`: HIP's `*_sync` requires a 64-bit mask and fails to compile; use the non-sync `__shfl_xor(v, mask, 32)` under `USE_ROCM` (faults-wavefront.md, "A compat layer for CUDA's `*_sync` warp functions").
- Every `__shfl*` inside `gemm_utils.cuh`, `gemm_base.cuh` (`broadcast_wscale`, `broadcast_ascale`), `gemm_w4a4.cuh` (quantize), `lora.cuh`, `attention.cuh`: sweep all, width 32 everywhere.
- `quantize_w4a4_act_kernel` / `quantize_w4a4_wgt_kernel` launch with `GEMM::WARP_SIZE` = 32 threads per block: half a wavefront on CDNA. Safe for the shuffle-only code; must not reach a stage-2 MFMA (which reads all 64 lanes).
- `gemv_awq.cu` `#define WARP_SIZE 32`, `out_smem[BlockSize / WARP_SIZE * 2]`: logical 32, fine once shuffles are width-32.
- `layernorm_kernels_impl.cuh`, `misc_kernels_impl.cuh`, `activation_kernels_impl.cuh`: block reductions through `reduction_utils.cuh` (`blockReduceSum`: `shared[32]`, `lane = tid & 31`, `wid = tid >> 5`), already logical-32 and correct on wave64 once the shuffles are width-32 with a valid mask.

### D. Arch macros and dispatch (`build-headers-macros.md`)

- `gemm_base.cuh:893` `invoke_kernel`: on HIP `__CUDA_ARCH__` is undefined, so it takes the `#else` branch and compiles every kernel body for every target, including the FP4 kernels (`MIN_ARCH = 1200`). FP4 and sm_75-only kernels need an explicit HIP guard that traps, or they emit the PTX `asm` into HIP and fail to compile.
- `__CUDA_ARCH__ < 800` / `>= 800` / `== 800` / `== 750` gates in `utils.cuh`, `mma.cuh`, `attention.cuh`, `gemm_w4a4.cuh:724` (i2f selection), `gemv_awq.cu:109`, `gemm_awq.cu:381,947`, `semaphore.h`: all read as "not defined" on HIP. Each needs a `USE_ROCM` arm chosen deliberately (e.g. i2f: `i2f_normal`; bf16 conversions: the sm_80 path).
- `__CUDACC_VER_MAJOR__` in `gemm_awq.cu:25` (cache-hint macro): undefined on HIP; the HIP path does not use cache hints.
- `#pragma nv_diag_suppress 177`: harmless warning under clang; leave.

### E. CUTLASS (no HIP target; replace on ROCm only, CUDA path untouched)

| file | function | reached by | ROCm replacement |
|---|---|---|---|
| `kernels/gemm_f16.cu` | `gemm_f16` (alpha, bias) | `GEMM_F16` in `Linear.cpp` (Sana `kv_linear`; LoRA of `GEMV_AWQ` norm linears in Flux) | hipBLASLt or hipBLAS `GemmEx` with bias added by an epilogue or a small kernel; `hipblaslt` epilogues differ per type (libraries.md) |
| `kernels/dwconv.cu` | `dwconv_f16` (depthwise 3x3) | `DWCONV` in Sana | small direct HIP depthwise kernel (preferred, no MIOpen handle) or `at::conv2d` through the torch interop |
| `kernels/gemm_batched.cu` | `gemm_batched_fp16` | only `Attention::forward(qkv, pool_qkv, sparsityRatio)`, which every caller invokes with `sparsityRatio = 0.0f` (`FluxModel.cpp:1289,1392`) | Q2 = a: hipBLAS strided-batched GEMM |
| `kernels/gemm_w8a8.cu` | `gemm_w8a8_fp16` | no caller anywhere in `src/` or `nunchaku/` | Q2 = a: left out of the ROCm source list |
| `kernels/dispatch_cutlass.h` | type dispatch | the above | not compiled on ROCm |

### F. Block-Sparse-Attention (`third_party/Block-Sparse-Attention`, flash-attention-2 fork on CuTe/CUTLASS, sm80 instantiations)

- `mha_fwd` (dense): the default `AttentionImpl::FlashAttention2` of the C++ `QuantizedFluxModel` (`FluxModel.cpp:120, 246`, `Attention::forward(qkv)`). ROCm replacement: `at::scaled_dot_product_attention` through `src/interop/torch.*` (torch's ROCm flash backend: AOTriton/CK), same `[B, T, H, D]` view and scale. The Python v1 models already use torch SDPA.
- `mha_fwd_block` (block-sparse): only from the sparsity path above, unreachable with `sparsityRatio = 0`. Q2 = a: ROCm throws "Block-sparse attention is not supported on ROCm".
- Porter addition (2026-11-24): `mha_varlen_fwd` (dense, variable length) is also reached, from `MultiHeadCrossAttention::forward` in `SanaModel.cpp:166` (the Sana cross attention); the planner's inventory missed it. ROCm: per-sample torch SDPA over the `cu_seqlens` offsets, in the same replacement file as `mha_fwd` (`src/interop/flash_api_rocm.cpp`).
- `topk` (misc kernels) feeding the block mask: plain CUDA, ports with the rest.

### G. Libraries and runtime

- cuBLAS: `cublasHgemm` x4 in `Linear.cpp` (LoRA up/down in `GEMM_W4A4`), handle in `common.h`, `Linear.cpp:120`: hipify maps to hipBLAS; link `hipblas`. Note `cublasGetStatusString` -> `hipblasStatusToString`. Watch rocBLAS f16 output rounding on long k (faults-numerics.md) only if LoRA output drifts.
- `Tensor.h`: `cudaHostAlloc(..., cudaHostAllocPortable)`, `cudaMallocAsync` on the current stream; HIP has both (`hipMallocAsync` needs a mempool-capable device; all gates are). `cudaMemcpy2DAsync`: fine.
- `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` in `gemm_w4a4_launch_impl.cuh:104`, `attention.cu:74`, `gemm_w8a8.cu:29,163` (92160 bytes), `gemm_awq.cu`: HIP accepts the call; the real limit is 64 KB LDS on gfx90a/gfx942/RDNA (160 KB on gfx950). Compute each kernel's `SHMEM_SIZE` for the configurations actually launched and record it; any kernel above 64 KB needs a smaller tile on ROCm (`platform-limits.md`). The zgemm W8A8 92160-byte case is in the unused `GEMM_W8A8` path.
- NVTX (`nvtx3/nvToolsExt.h`, `nvtxRangePushA/Pop`) in `Linear.cpp`, `SanaModel.cpp`, `FluxModel.cpp`: roctx (`roctracer/roctx.h`) or no-op macros on ROCm; do not add a roctracer link dependency if a no-op is enough.
- `utils.set_cuda_stack_limit` (`cudaDeviceSetLimit(cudaLimitStackSize)`): HIP supports it.

### H. Python

- `nunchaku/utils.py` `get_precision`, `is_turing`, `check_hardware_compatibility` key on `torch.cuda.get_device_capability()`; ROCm reports gfx numbers that collide with NVIDIA SMs (gfx90a -> (9,0), gfx942 -> (9,4), gfx1100 -> (11,0); platform-limits.md). Add a `torch.version.hip` arm: precision `int4` on every AMD arch except gfx950 and gfx1250, where Q1 = c makes `fp4` available (whether `auto` picks `fp4` there, as upstream does on FP4-capable GPUs, is an open question for jeff in notes; until it is ruled and that path is validated, `auto` stays `int4`), `is_turing` False, compatibility check accepts the supported gfx list.
- `setup.py` version tag and arch selection (above).
- Everything else (diffusers pipelines, `nunchaku/models`, caching, LoRA conversion, `app/`, `examples/`) is device-agnostic PyTorch and needs no change; the tests exercise it.

## Risk list

1. **Fragment-layout correctness of the primitive layer.** A wrong lane mapping gives plausible garbage. Mitigation: a kernel-level golden test (below) comparing each helper and each full op against a torch reference, before any end-to-end run.
2. **Wave64: logical warps sharing a wavefront.** Any default-width shuffle, any 64-lane ballot, or any MFMA that reads all 64 lanes mixes two warps' data. Mitigation: width-32 sweep (section C), stage-2 MFMA only in 4-lane-block form or proven otherwise, validation on both widths.
3. **Register pressure.** Each lane holds a 32x128 per-warp accumulator tile (128 fp32 plus psum) by design; on RDNA (wave32, 256 VGPRs) the stage-1 gather path may spill. Check `-Rpass-analysis=kernel-resource-usage` / the ISA for scratch use; tile changes are forbidden (they change the checkpoint layout), so reduce temporaries instead.
4. **LDS above 64 KB** for some epilogue/attention configurations (section G).
5. **Numerics.** f16 LoRA and attention accumulate in a different order than NVIDIA tensor cores; gfx11/gfx12 WMMA drift over long k (faults-numerics.md). The end-to-end gate is LPIPS against a 16-bit reference with upstream's thresholds, which tolerates this; op tests use relative bounds for float paths and exact compare for integer paths.
6. **Approximate math replacements** (`tanh.approx`, `ex2.approx.ftz`, `rcp.approx.ftz`): ftz behaviour differs; GELU/SiLU/sigmoid outputs may shift slightly. Covered by LPIPS; do not chase bit-exactness.
7. **`cvt.pack.sat` nibble order and saturation.** The packed activation layout written by `quantize_w4a4_act_fuse_lora` is consumed by the GEMM in the same run, but the order must match `packer.py` for weights; test both quantize helpers against a CPU reference including saturation at +-8/15 and negative zero.
8. **Fresh allocations are not zeroed on ROCm** (faults-memory-streams.md): check every `Tensor::allocate` / `cudaMallocAsync` whose first use is an accumulate (`reduce_add` into LoRA activations, split-k in AWQ with `semaphore.h`).
9. **hipBLASLt / MIOpen handle creation under graph capture** if a user captures; not exercised by the tests, note only.
10. **Windows**: upstream ships Windows wheels (MSVC flags in `setup.py`, `scripts/build_windows_wheel.cmd`). On ROCm Windows the host compiler is clang-cl via TheRock; the MSVC flag set (`/Zc:__cplusplus`, `/bigobj`, `/FS`) must still reach clang-cl, and `-Xcompiler` forms must not reach hipcc (windows.md, "hipcc.exe and hiprtc traps for setuptools/Python builds").
11. **Model size on validation hosts.** FLUX.1-dev 16-bit reference runs need about 24 GB of GPU memory and network access to Hugging Face, and FLUX.1-dev is a gated repository (an HF token with the licence accepted).

## File-by-file change list

- `setup.py`: ROCm branch (arch list, flags, version tag, drop Block-Sparse-Attention and `gemm_w8a8.cu` sources per Q2).
- `.gitignore`: hipified mirror outputs.
- `src/kernels/zgemm/mma_hip.cuh` (new): primitive layer, stage 1 plus stage-2 arch paths.
- `src/kernels/zgemm/gemm_base.cuh`: include selection; `invoke_kernel` HIP arm; `broadcast_*` shuffle widths.
- `src/kernels/zgemm/gemm_utils.cuh`: HIP arms for every `asm` helper (section A).
- `src/kernels/zgemm/gemm_w4a4.cuh`: `mma` (INT4) via primitive layer; `mma_fp4`/`quantize_*_fp4` trap on ROCm (Q1); i2f selection arm; `__nanosleep`.
- `src/kernels/zgemm/gemm_w8a8.cuh`: `mma` via primitive layer.
- `src/kernels/zgemm/epilogues.cuh`, `lora.cuh`, `attention.cuh`: shuffle widths and masks; `attention.cuh:63` arch gate.
- `src/kernels/zgemm/gemm_w4a4_launch_impl.cuh`, `attention.cu`, `gemm_w8a8.cu`, `gemm_w4a4_test.cu`: LDS limits if any exceed 64 KB.
- `src/kernels/utils.cuh`: `cvt.rni.sat.s8` arms, `__CUDA_ARCH__` gates, `__nanosleep`.
- `src/kernels/reduction_utils.cuh`: mask/width.
- `src/kernels/awq/gemm_awq.cu`, `dequantize.cuh`, `semaphore.h`: ldmatrix/mma via the primitive layer (the AWQ kernel uses the same m16n8k16 f16 fragment layout), cp.async, lop3, atomics.
- `src/kernels/awq/gemv_awq.cu`: shuffle width and mask.
- `src/kernels/gemm_f16.cu`, `dwconv.cu`, `gemm_batched.cu`: `#ifdef USE_ROCM` implementation beside the CUTLASS one (or a sibling `_hip` source selected by `setup.py`).
- `src/FluxModel.cpp`: `mha_fwd` -> SDPA on ROCm; `mha_fwd_block` -> error on ROCm; `flash_api.h` include guard; NVTX.
- `src/SanaModel.cpp`, `src/Linear.cpp`: NVTX, `flash_api.h` guard.
- `src/common.h`: `cublasGetStatusString` arm if hipify leaves it.
- `nunchaku/utils.py`: ROCm arms of `get_precision`, `is_turing`, `check_hardware_compatibility`.
- `tests/ops/` (new, shipped with the port): kernel-level golden tests (below). Upstream asks contributors not to modify existing tests; new tests are welcome.
- `src/interop/flash_api_rocm.{h,cpp}` (new, ROCm builds only): `mha_fwd` and `mha_varlen_fwd` through torch SDPA, `mha_fwd_block` raising.
- README / docs installation section: a "Building for ROCm" paragraph with the exact command.

## Build commands

Linux (any gate), inside the job venv with a ROCm PyTorch (>= 2.7) installed and submodules initialised:

```
git submodule update --init --recursive
export PYTORCH_ROCM_ARCH=gfx90a        # or gfx942, gfx950, gfx1100, gfx1151, gfx1201 ...
export MAX_JOBS=<physical cores>
pip install -e . --no-build-isolation -v
```

Windows: same command from a TheRock ROCm PyTorch environment (`windows.md`, torch wheel with the matching device extra), Developer prompt so clang-cl/lld-link resolve.

Wrap with `utils/timeit.sh nunchaku compile -- ...`. A clean rebuild after editing a `.cu`/`.cuh` must re-hipify (delete the hipified mirror).

CUDA no-regression: `nvcc` compile of the CUDA build on the mi355 host with a CUDA-built torch (validation.md); expected to hit the ROCm-torch header wall on fleet images, in which case record `cuda-not-validated: <error>` and compile the touched `.cu` files standalone with nvcc for sm_80 and sm_120a at least.

## Test plan

There are no kernel-level tests upstream; every upstream test is an end-to-end diffusion run compared to a 16-bit reference image by LPIPS. So the order is:

1. **Op-level golden tests (new, required before any end-to-end run, every gate).** `tests/ops/`:
   - each primitive (INT4 s4/u4 m16n8k64, s8 m16n8k32, f16/bf16 m16n8k16) through a tiny test kernel or `gemm_w4a4_test.cu`'s existing entry points, against a torch reference: exact for integer, relative bound for float;
   - `quantize_w4a4_act_fuse_lora` against a torch reference of SVDQuant activation quantization (scales, nibble packing, LoRA-down output);
   - `gemm_w4a4` (int4, fp16 and bf16, with and without LoRA, with the norm/rope/QKV-pack epilogues via `test_rmsnorm_rope`, `test_pack_qkv`) against dequantize-then-matmul in torch, using weights packed by `nunchaku/lora/flux/packer.py`'s packer so the layout contract is tested end to end;
   - `attention_fp16` against torch SDPA;
   - `gemm_awq` / `gemv_awq` against dequantized torch matmul (and each other row by row);
   - `gemm_f16`, `dwconv_f16` against torch.
2. **End-to-end, Linux gates (wave64 MI-class and wave32 RDNA hosts with >= 24 GB):**
   - `pytest -v tests/v1/flux/test_flux1_schnell.py tests/v1/sdxl tests/v1/z_image` (ungated models first),
   - `pytest -v tests/flux/test_flux_schnell.py tests/flux/test_flux_dev.py tests/flux/test_flux_dev_loras.py tests/flux/test_flux_qencoder.py tests/flux/test_flux_kontext.py` (C++ `QuantizedFluxModel` path, both `flashattn2` and `nunchaku-fp16` attention cases; qencoder exercises the AWQ T5 kernels),
   - `pytest -v tests/sana` (C++ `QuantizedSanaModel`: gemm_f16, dwconv, LiteLA epilogue),
   - `tests/v1/flux/test_flux1_dev.py`, `tests/v1/qwenimage/test_qwenimage.py` where the host has the memory and the HF token (Qwen-Image 16-bit reference is about 40 GB; on a 64 GB GCD it fits);
   - pass criterion: upstream's own `expected_lpips` for `int4-bf16` (fp16 on a host without bf16 is not expected on any gate). Never loosen a threshold; if an AMD result misses by a small margin, record the measured LPIPS and bring it to jeff.
   - `tests/flux/test_flux_speed.py` skips by GPU name (no AMD entry) and `test_turing.py` skips off sm_75; record them as skipped, do not edit them. `test_device_id.py` needs two GPUs; run it on a multi-GPU host with two devices exposed, otherwise record skipped. `test_flux_memory.py` asserts upstream's NVIDIA memory budget; run it and record the value.
3. **Windows gate:** build, op-level tests, and the small models only (`tests/v1/sdxl`, `tests/sana`); FLUX-scale runs are not for the gfx1151 laptop (memory rule), and run on a Windows RDNA desktop host if one is available.
4. Non-GPU regression set: the package imports without a GPU (`python -c "import nunchaku"` fails today without `_C`; no change expected), and `pre-commit`/lint config untouched.

Record LPIPS values, op-test pass counts and per-image latency per gate in notes.

## Open questions

All three planner questions are ruled (notes, "Rulings on the planner's questions (Jeff Daily, 2026-11-23)"). Open items the porter records in notes as they arise.
