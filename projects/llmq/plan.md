# llmq port plan (lead platform: linux-gfx90a, MI250X, ROCm 7.2.1)

## Project
- Name: llmq (LLM.Q / `llm.q`)
- Upstream: https://github.com/IST-DASLab/llmq
- Default branch: `dev`
- What it is: quantized large-language-model TRAINING in pure CUDA/C++20, inspired by llm.c. Single-node multi-GPU. Trains Llama/Qwen-class models with FP8 (e4m3/e5m2) and bf16 matmuls. Not an inference/quant-kernel library -- it is a full training engine (dataloader, AdamW, NCCL all-reduce, cuDNN flash attention, checkpointing).

## Disposition: BLOCKED (reimplement-not-port; Hopper/sm89+ PTX + CUTLASS-class hot path)
This is the CPM.cu / SpargeAttn class called out in the dispatch brief. The performance-critical custom kernels are written directly in NVIDIA tensor-core PTX and Hopper-only features that have no ROCm/HIP spelling and no mechanical translation. A correctness-first mechanical port is NOT possible here; the hot path must be REWRITTEN against AMD-native primitives (rocWMMA / MFMA / Composable Kernel) and the Hopper-specific kernels must be re-architected for CDNA. That is a multi-week kernel-authoring effort, not a CUDA->HIP conversion, so the lead platform is set to `blocked` with this concrete reason rather than `planned`.

## Existing AMD support
None. No HIP/ROCm path, no makefile.hip, no branches (`dev` + `HEAD` only), no fork. The only AMD-adjacent token in the tree is a lone `#ifndef __HIP__` in `src/kernels/kernel_utils.cuh:11` guarding a `<cooperative_groups/reduce.h>` include -- a token gesture, not a build path; the surrounding code hardcodes 32-lane warps (`assert(__activemask() == 0xffffffff)`, `__reduce_max_sync(0xffffffff, ...)`, `threadIdx.x % 32`). Classification per the guide: "no HIP path" -> a port would add value IF the kernels were portable. They are not (see below). Decision: blocked.

## Build classification: cmake (pure CMake, NOT a torch extension)
Evidence: top-level `CMakeLists.txt:9` `project(LLMQ CUDA CXX)`; `enable_language(CUDA)` implicit; `.cu` sources compiled directly (`CMakeLists.txt:164-183`); no `find_package(Torch)`, no `torch.utils.cpp_extension`, no torch in `pyproject.toml` (it uses scikit-build-core + nanobind for an OPTIONAL python binding, default OFF at `CMakeLists.txt:20`). So if it were portable it would be Strategy A (compat header + `.cu` LANGUAGE HIP). ext_type = `cmake`.

## Why it is a rewrite, not a port -- CUDA surface that does not translate

### 1. Custom quant/bf16 GEMM is raw sm80/sm89 tensor-core PTX (the hot path)
`src/kernels/gemm_mma.cu` + `src/kernels/tensor_core_utils.cuh` implement the FP8/bf16 matmul by hand in inline PTX:
- `tensor_core_utils.cuh:76` `ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16`
- `tensor_core_utils.cuh:103,108` `mma.sync.aligned.m16n8k32.row.col.f32.<e4m3|e5m2>.<...>.f32` (FP8 tensor-core MMA, sm_89)
- `tensor_core_utils.cuh:130,141` `mma.sync.aligned.m16n8k32 ... f16` (bf16/f16 MMA)
- swizzled `__cvta_generic_to_shared` shared-memory layout + `cuda_pipeline_primitives.h` (`cp.async`) double-buffered pipeline, warp-specialized A/B loaders.
These PTX opcodes are NVIDIA ISA. ROCm has NO `mma.sync`/`ldmatrix`/`cp.async`. The AMD-native equivalent is MFMA intrinsics (`__builtin_amdgcn_mfma_*` / rocWMMA) with a completely different fragment layout, K-tiling, and LDS swizzle, plus FP8 on CDNA uses `__hip_fp8` types and gfx94x MFMA (gfx90a/MI250X has only limited/no native FP8 MFMA -- FP8 GEMM on MI250X would need emulation or upcast). This is exactly the CUTLASS-class rewrite the guide forbids translating (Changelog 2026-05-30: "CUTLASS does NOT port to ROCm ... reimplement against CK ... ck_tile preferred").

### 2. fused_classifier_tma.cu is Hopper sm_90-only
`src/kernels/fused_classifier_tma.cu`: `__cluster_dims__(8,1,1)` (`:43`), `cg::cluster_group` / `cluster.block_rank()` / `cluster.map_shared_rank()` distributed-shared-memory (`:58,72,194`), `cluster.barrier_arrive()` (`:201`), `<cuda/barrier>` mbarrier. Thread-block clusters and DSMEM are Hopper-exclusive; CDNA has no cluster concept. No HIP spelling exists. (There is a non-TMA `fused_classifier.cu` fallback, so this specific kernel could be dropped on AMD, but it is part of the surface.)

### 3. Hard dependency on NVIDIA libraries with no drop-in on this code path
- cuDNN + cudnn-frontend v1.12.0 (`CMakeLists.txt:28-33,169,186`): the ENTIRE fast/flash attention path is cuDNN-frontend graphs (`src/kernels/cudnn_att.cpp`). MIOpen is not API-compatible with cudnn-frontend; this is a from-scratch reimplementation (the in-tree `attention.cu` is a slow fp32 baseline that explicitly exists "because cuDNN does not support 32-bit" -- it is not a full replacement for training throughput).
- cuBLASLt (`src/kernels/matmul.cpp`): the non-custom matmul path and ALL fallbacks go through `cublasLtMatmul` with FP8 scaling (`CUBLASLT_MATMUL_DESC_A_SCALE_POINTER`), epilogue bias, transpose modes. hipBLASLt exists on ROCm and is the closest swap, BUT FP8 GEMM support on hipBLASLt is gfx94x-centric; on gfx90a (MI250X) FP8 cuBLASLt-equivalent matmul is not available. The whole point of the project (`--matmul-dtype=e4m3`) does not map to the lead arch.
- NCCL (`CMakeLists.txt:99-106`): swap to RCCL is mechanical (RCCL is API-compatible) -- not a blocker by itself.
- cuFile (gpu-direct storage), NVML, NVTX, curand: cuFile/NVML already have CMake-optional fallbacks (`USE_CUFILE`, `USE_NVML`); curand_kernel in `random.cu` maps to rocRAND; NVTX maps to roctx. These are not blockers.

### 4. FP8 + wave64 fault-class exposure (would still need fixing even after a rewrite)
- FP8 e4m3/e5m2 used in 18 files via `__nv_fp8_*` (`cuda_fp8.h`). HIP has `__hip_fp8_e4m3_fnuz`/`__hip_fp8_e5m2_fnuz` -- but CDNA2/gfx90a FNUZ has a DIFFERENT bias/encoding than NVIDIA OCP e4m3, so bit values and scale semantics differ; a port must re-derive every FP8 scale/cast.
- Hardcoded 32-lane warp everywhere: `kernel_utils.cuh:20` `assert(__activemask() == 0xffffffff)`, `__reduce_max_sync(0xffffffff,...)`, `threadIdx.x % 32`, `__shfl_xor_sync(0xFFFFFFFFu, ...)` in `fused_classifier_tma.cu:29` and across `bias/adamw/global_norm/qk_norm/attention`. On gfx90a (wave64) these 32-bit masks and `%32` lane math are the standard warp-size fault class (guide Fault classes). Fixable, but only after the kernels exist on AMD.
- `cg::tiled_partition<32>` and `<16>` (attention.cu:25-28): width-32 logical-warp partitions are arch-agnostic per the guide, but the `__activemask()==0xffffffff` assert and `__reduce_max_sync` 32-bit-mask intrinsics are not.
- `rcp.approx.ftz.f32` PTX in `kernel_utils.cuh:97` (minor; has a HIP intrinsic equivalent).

## CUDA surface inventory (summary)
- Kernels (`.cu`): 21 files. Portable-ish (elementwise/reduction; bf16): encoder, swiglu, rope, rmsnorm, quant, global_norm, bias, adamw, transpose, vector_add, random, fill, convert, qk_norm, fused_classifier (non-TMA), attention (fp32 baseline).
- NOT portable without rewrite: gemm_mma.cu (PTX MMA), tensor_core_utils.cuh (PTX), fused_classifier_tma.cu (Hopper clusters/TMA), cudnn_att.cpp (cuDNN-frontend), matmul.cpp (cuBLASLt FP8).
- Libraries: cuBLASLt (FP8 -> hipBLASLt, no FP8 on gfx90a), cuDNN-frontend (-> reimplement), NCCL (-> RCCL ok), curand (-> rocRAND ok), cuFile (optional fallback), NVML (optional fallback), NVTX (-> roctx).
- Warp intrinsics: `__shfl_xor_sync`, `__activemask`, `__reduce_max_sync`, `atomicMax_block`, `cg::reduce`, `cg::tiled_partition` across 9 kernel files.
- No textures/surfaces. cudaMallocManaged used in tests.

## Risk list
- FP8 on gfx90a/MI250X: native FP8 MFMA is gfx94x (MI300); on MI250X FP8 matmul must upcast/emulate -> changes numerics and defeats the project's purpose. This is the deepest risk: the lead arch may be the WRONG AMD target for this project (gfx942 would be the natural fit).
- FNUZ vs OCP FP8 encoding mismatch -> every scale and cast re-derived.
- Hopper cluster/DSMEM kernel has no CDNA analogue (drop or re-architect the cluster reduction).
- cuDNN-frontend attention -> full reimplementation (CK fused attention / ck_tile, or a hand MFMA flash kernel).
- wave64 32-bit lane masks / `%32` / `__activemask()==0xffffffff` asserts across many kernels.
- C++20 + cooperative_groups/reduce under hipcc; `__reduce_max_sync` has no HIP builtin (must hand-roll the warp reduce).
- Multi-arch standard cannot even be approached until the kernels exist; gfx90a;gfx1100 build is moot at this stage.

## GPU-validatable slice (if a future scoped port were attempted)
A correctness-first subset COULD be brought up on AMD without the tensor-core path: build `llmq-common` with `backend=CuBLAS` only (skip gemm_mma), drop fused_classifier_tma (use fused_classifier), replace cuDNN attention with the in-tree fp32 `attention.cu` baseline, swap NCCL->RCCL, hipBLASLt for bf16 (NOT FP8) matmul. The unit tests (`src/testing/`, Catch2): test-swiglu, test-rope, test-classifier, test-gemm. test-gemm compares `matmul_cublaslt` against an fp32 reference (`cudaMallocManaged`), so a hipBLASLt bf16 path is independently checkable. That bf16-only training slice is a legitimate first deliverable IF jeff wants to invest, but it is NOT a "port of llmq" -- it discards FP8 quant (the project's reason to exist) and the tuned GEMM, so it should be a deliberate scoping decision, not the default. Recorded here so the porter does not silently ship a gutted build as "ported".

## Build commands (for reference; not runnable as-is on AMD)
- CUDA (upstream): `cmake -S . -B build && cmake --build build --target train`
- Tests: `cmake --build build --target unit-tests && ctest --test-dir build` (or `./build/unit-tests`).
- A future HIP bringup would gate on `USE_HIP`, RCCL/hipBLASLt/rocRAND/roctx, `-DCMAKE_HIP_ARCHITECTURES` defaulted from the cache var (never literal gfx90a), and likely target gfx942 not gfx90a for FP8.

## Test plan
- GPU tests: Catch2 `unit-tests` (swiglu, rope, classifier, gemm). Non-GPU regression set: the python reference checks under `src/binding/python/tests/` (torch_reference) gate the optional binding, not the core.
- None of these can run on AMD today because `llmq-common` does not build on ROCm (PTX MMA + cuDNN + cuBLASLt FP8 link).

## Open questions (for jeff)
1. Is the intended AMD target really gfx90a/MI250X, or gfx942/MI300? FP8 training (the project's purpose) needs gfx942 MFMA; on gfx90a FP8 must be emulated. Strongly recommend retargeting this project's lead to gfx942 if it is to be done at all.
2. Scope: full AMD-native rewrite (rocWMMA/CK FP8 GEMM + CK flash attention + cluster re-arch) -- weeks of kernel authoring -- versus a bf16-only, cublasLt(hipBLASLt)-backed correctness slice that drops the tuned FP8 GEMM and cuDNN attention. The latter is achievable but is a different, smaller product.
3. Given MOAT's mechanical-port cadence, recommend deferring llmq alongside the other CUTLASS/PTX-rewrite-class and OptiX-gated projects (raft/CPM.cu class) until a CK kernel-authoring track is funded.

## Recommendation
Keep linux-gfx90a `blocked` with reason "Hopper/sm89+ tensor-core PTX (mma.sync/ldmatrix/cp.async) + thread-block clusters/TMA + cuDNN-frontend attention + FP8-on-gfx90a unsupported: AMD-native rewrite, not a mechanical CUDA->HIP port; recommend retarget to gfx942 and fund a CK kernel track". Do not open anything upstream. Followers (gfx1100/gfx1151) remain blocked-needs-gfx90a.
