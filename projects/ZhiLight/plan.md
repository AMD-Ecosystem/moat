# ZhiLight -- ROCm/HIP port plan (lead: linux-gfx90a)

## Project
- Name: ZhiLight
- Upstream: https://github.com/zhihu/ZhiLight (Zhihu + ModelBest LLM inference engine)
- Default branch: main
- Cloned read-only at projects/ZhiLight/src (depth=1)

## Existing AMD support (finding + decision)
CUDA-only. There is NO HIP build path: top-level CMakeLists and 3rd/bmengine both `enable_language(CUDA)`, hardcode `CMAKE_CUDA_ARCHITECTURES "80;89;90a"`, find `cublas/cublasLt/curand/culibos/nccl`, and the Dockerfile is `nvidia/cuda:12.4.1`. No gfx/hipcc/USE_HIP anywhere in the build.

The ONLY ROCm trace is vestigial: the GPTQ kernels `src/nn/quant/gptq/{q_gemm.cu,compat.cuh}` carry a handful of `#if defined(USE_ROCM)` guards and a `#include <hipblas/hipblas.h>` shim. These were copied verbatim from the vllm/exllama GPTQ kernel lineage; `USE_ROCM` is never defined by this project's build, so they are dead. This is not an existing port, abandoned or otherwise.

Decision: PROCEED with a correctness-first mechanical HIP port of the portable surface, and EXPLICITLY DISABLE the NVIDIA-tensor-core-PTX quant/attention fast paths on the first AMD pass (they are reimplement-not-port; see Risk list). A ROCm/HIP build of ZhiLight adds clear value: there is none today.

## Build classification: cmake (NOT a torch extension)
Evidence:
- setup.py defines `CMakeExtension` and a custom `CMakeBuild(build_ext)` that shells out to `cmake` + `cmake --build` (setup.py lines 14-89). It does NOT use `torch.utils.cpp_extension` (`CUDAExtension`/`BuildExtension`).
- The build is pure CMake: top-level CMakeLists.txt `enable_language(CUDA)` (line 13), `file(GLOB_RECURSE BACKEND_SOURCES src/*.cpp src/*.cu)` (lines 73-76), `pybind11_add_module(C ...)` (line 160). pybind11 is used directly, not torch's extension machinery.
- Torch is a *link* dependency only: it links `${Python_SITELIB}/torch/lib` and includes `torch/include` (CMakeLists 137,146) to satisfy the flash-attn `.so` ABI, and the tests `find_package(Torch)`. Torch hipify is NOT in the loop.
Per PORTING_GUIDE "Build classification": no `CUDAExtension`/`torch.utils.cpp_extension` -> treat as pure CMake. ext_type = `cmake` (set in upstream.json + status.json).

## Port strategy: A (compat header) + CMake HIP language gating
Rationale: pure CMake with `.cu` sources -> Strategy A is the minimal-footprint fit and matches the compat-header approach in the guide. Torch's hipify (Strategy B) does not apply because nothing goes through `CUDAExtension`.

Concretely:
1. Add `USE_HIP` option to top-level CMakeLists and to `3rd/bmengine/bmengine/CMakeLists.txt`. When ON: `enable_language(HIP)`, set `CMAKE_HIP_ARCHITECTURES` from the cache var defaulting to `gfx90a` only when unset (NEVER a literal -- guide lesson, so followers need no source edit), mark the existing `.cu` sources `LANGUAGE HIP`, and define `-DUSE_HIP -DUSE_ROCM` (the latter activates the existing gptq guards). When OFF: existing `enable_language(CUDA)` path is untouched, NVIDIA build intact.
2. Add ONE compat header (e.g. `src/utils/cuda_to_hip.h` and a parallel one usable from bmengine) that, under HIP, includes `<cstring>/<cstdlib>` BEFORE `<hip/hip_runtime.h>` (guide gpuRIR lesson: host memcpy/memset must resolve to libc, not HIP device overloads), then aliases the cuda* spellings the project uses. Library handles map cublas->hipblas, cublasLt->hipblasLt, curand->hiprand, nccl->rccl, `<mma.h>`->guarded out. Use pytorch's `cuda_to_hip_mappings.py` as the authoritative name table.
3. Swiss-army: gate the genuinely NVIDIA-only translation units (marlin, awq, fp8 PTX, the `mma.h` tensor-core attention paths, deep_gemm/flash_mla H20 calls) out of the HIP build with `#if !defined(USE_HIP)` and route their call sites to the portable fallback kernels that already exist in the dispatch (see Risk list).

The host C++ (model/, generator/, most of bmengine/core) is untouched by HIP; only `.cu` TUs see hipcc.

## CUDA surface inventory
- Scale: 31 `.cu` + 10 `.cuh` in src/, 12 `.cu` + 2 `.cuh` in 3rd/bmengine; ~18.5k `.cu`/`.cuh` LOC total.
- Elementwise / norm / softmax / rope / embedding / cat / index_select / typecast / arithmetic: bmengine/functions/*.cu and src/nn/{layernorm,functions,position,embedding,block}/*.cu. Straightforward HIP; the only catch is the shared warp/block reduce helper (below).
- Warp/block reductions (CENTRAL wave64 locus): `3rd/bmengine/.../functions/reduce.cuh` -- `warpReduce{Sum,Max,Min}[B]` start at `offset=16` with `__shfl_down_sync(0xFFFFFFFF,...)` (32-lane, 32-bit mask); `blockReduce*` hardcode `% 32` / `/ 32` and `__shared__ shared[33]` (<=32 warps). Reused by attention, softmax, ff, layernorm, cross_entropy, int8 quant. On wave64 these silently reduce only 32 of 64 lanes -> wrong results, no crash.
- Attention decode kernels: `src/nn/attention/attention_kernel.cu` -- `#define WARP_SIZE 32`, `assert(DIM_HEAD == WARP_SIZE*4)` (128-dim head == 4 warps), `g_q + laneId*4` indexing, per-lane `warpReduceSum`. Includes `<mma.h>` (nvcuda::wmma) for the tensor-core decode path. `quant_attention.cuh` uses `prmt.b32`/`sub.f16x2` PTX for int8->fp16 dequant.
- GEMM (central): `3rd/bmengine/.../functions/gemm.cpp` uses cublasLt with `cublasLtMatmulAlgo_t` heuristic autotuning + attribute setters, dtype matrix incl `CUDA_R_8F_E4M3`/`E5M2` fp8. `src/nn/linear/linear.cpp`, `gemm_grouped.cpp`, `embedding.cu`, `gptq/q_gemm.cu` also call cublas/cublasLt.
- Quantization kernels (NVIDIA tensor-core PTX, reimplement-not-port):
  - `src/nn/quant/marlin/{gptq_marlin,awq_marlin_repack,gptq_marlin_repack}.cu` + `marlin.cuh`: `mma.sync.aligned.m16n8k16`, `ldmatrix.sync`, `cp.async.cg.shared.global`, `__cvta_generic_to_shared`, `lop3`, `ld.global.acquire.gpu`/`fence.acq_rel.gpu`. Guarded `#if __CUDA_ARCH__ >= 800` (Ampere+).
  - `src/nn/quant/awq/gemm_kernels.cu` (+ dequantize.cuh): `ldmatrix`, `mma.sync.m16n8k8/k16`, sm75/sm80 guards.
  - `src/nn/quant/fp8/fp8_util.cu`: `cvt.rn.satfinite.e4m3x2.f16x2` PTX, `#if __CUDA_ARCH__ >= 890` (Ada+).
  - `src/nn/quant/gptq/{q_gemm,q_gemm_k_major,utils,groupwise_int8}.cu`: `#define WARP_SIZE 32`, warp shuffles; q_gemm already has partial USE_ROCM/hipblas shims (vllm lineage) -- the most portable of the quant set.
  - `src/nn/quant/int8/{quant_kernel,quant_reduce_kernel}.cu`: blockDim {32,32}, `% 32` grouping; portable but wave64-sensitive in the group reductions.
- Prebuilt NVIDIA-only static libs (git-LFS): `3rd/deep_gemm/libdeep_gemm.a` (`deep_gemm_fp8_block_h20_group` -- literally H20/Hopper), `3rd/flash_mla/libflash_mla.a` (DeepSeek MLA flash-attn, Hopper), `3rd/zmq/libzmq.a`. `3rd/tensorrt_llm/` is a link dir referenced in CMake but absent in the tree. deep_gemm + flash_mla are gated `ENABLE_DS_DEEP_GEMM`/`ENABLE_DS_FLASH_MLA` (set only on x86_64) with `USE_FLASH_MLA` runtime env -> disable on AMD.
- flash-attn: pulled from the pip `flash-attn` CUDA wheel; CMakeLists `FATAL_ERROR`s if no `flash_attn*.so` is in site-packages (lines 107-126). `ENABLE_FLASH_MHA` gates the prefill path. The `.so` is a CUDA binary -> must make this optional on AMD (ROCm flash-attn exists but ABI/symbol differ).
- NCCL: bmengine c10d + context/engine, src/generator, src/model, embedding.cu. 1:1 swap to RCCL (`-lrccl`, `rccl.h`); API matches NCCL.
- streams/events, cudaMalloc/Memcpy, curand (generator/random_util.cu): mechanical hip* swaps.

## Risk list
1. wave64 warp-size (HIGH, pervasive). reduce.cuh and every consumer assume 32-lane warps and 32-bit shuffle masks. Fix: a per-arch `kWarpSize` constant (`__GFX9__`->64 else 32, guide pattern), drive the shuffle loop from `kWarpSize/2`, size block-reduce `shared[]` to a `C10_WARP_SIZE_UPPER_BOUND`-style bound, replace `0xFFFFFFFF` with full-width masks (use the no-mask `__shfl_down` form on HIP). Re-derive attention_kernel's `DIM_HEAD==WARP_SIZE*4` assumption: on wave64 a 128-dim head is 2 wavefronts, not 4 warps -- the lane*4 load and per-warp tiling must be reworked or the kernel pinned to a 32-lane sub-group. This is the single biggest correctness risk and the reason gfx1100/gfx1151 (wave32) may behave differently from gfx90a.
2. NVIDIA tensor-core PTX in marlin/awq/fp8/quant_attention (HIGH, reimplement-not-port). `mma.sync`, `ldmatrix`, `cp.async`, `cvt.e4m3x2`, `prmt`, `lop3`, `cvta.shared` do not exist on AMD. Per guide (CUTLASS lesson, perf-kernel lesson): do NOT shim PTX. First AMD pass DISABLES these paths (`#if !defined(USE_HIP)`) and routes GPTQ/AWQ/Marlin/fp8 matmul to the cublasLt->hipBLASLt dequant+GEMM fallback that the dispatch already supports for unquantized/int8. A later AMD-native pass (rocWMMA / CK / MFMA, ck_tile preferred) can re-add quant fast paths. State which models this drops: Marlin GPTQ fast path, AWQ fast path, native fp8-block (DeepSeek-V3 FP8) -- these become unsupported-on-AMD or fall back to slower paths on the first port.
3. cublasLt -> hipBLASLt algo autotuning (HIGH). gemm.cpp caches `cublasLtMatmulAlgo_t` from heuristic queries with attribute setters. hipBLASLt's algo-get/heuristic API and enums differ (guide: watch hipBLAS v2 enums); the cached-algo serialization path likely needs an `#if defined(USE_HIP)` simpler heuristic. fp8 (`CUDA_R_8F_E4M3`) matmul support on hipBLASLt/gfx90a is limited -- fp8 GEMM may need to fall back to bf16/f16.
4. Prebuilt CUDA static libs deep_gemm/flash_mla (MED). Cannot link `libdeep_gemm.a`/`libflash_mla.a` (NVIDIA objects) on AMD. Gate `ENABLE_DS_DEEP_GEMM`/`ENABLE_DS_FLASH_MLA` OFF for the HIP build; the DeepSeek-V3/MLA fast paths are disabled (MLA falls back to the generic attention path already present in multi_head_latent_attention.cpp).
5. flash-attn .so hard dependency (MED). CMake FATAL_ERRORs without a CUDA flash_attn wheel. Make the flash-attn discovery non-fatal under USE_HIP (define `ENABLE_FLASH_MHA=0`); prefill then uses the in-tree attention/softmax kernels. Optionally wire ROCm flash-attn later, but not required for a correctness-first port.
6. `<mma.h>` / nvcuda::wmma include in attention_kernel.cu (MED). Replace with a guarded include; the wmma tensor-core decode path is disabled on AMD (fall back to the scalar multiply_q_k_block path), same class as risk 2.
7. NCCL->RCCL (LOW). 1:1 but confirm `ncclComm_t`/datatype enums and that the host-SIMD all-reduce (custom x86 AVX path in bmengine c10d) is x86-only -- on gfx90a hosts that is fine (x86_64), keep it.
8. fp8/bf16 host types (LOW-MED). `__nv_fp8_e4m3`/`__nv_bfloat16` (53 hits). Map to `__hip_fp8*`/`hip_bfloat16`/`__hip_bfloat16` in the compat header; verify the bf16 arithmetic operators exist on ROCm 7.2.1.
9. `-ffp-contract` / fast-math drift (LOW). clang(HIP) defaults to `-ffp-contract=fast`; the bit-exact-ish torch-reference test compares (test_softmax, test_attention) may drift ~1 ULP. Pin `-ffp-contract=on` in `CMAKE_HIP_FLAGS` (guide CV-CUDA lesson) and use `rtol/atol` already present in the python tests.
10. cub/hipCUB usage in bmengine sort.cu/topk.cu (LOW-MED). Swap Thrust/CUB->rocThrust/hipCUB; watch the wave64 block-collective TempStorage reuse race (guide CV-CUDA lesson) -- add `__syncthreads()` between reused collectives.
11. RelWithDebInfo strips `-DNDEBUG` (CMakeLists 59-60) so the many `assert(DIM_HEAD==WARP_SIZE*4)` are LIVE in debug builds -- they will fire on wave64 before producing wrong output, which is actually a useful tripwire during bringup.
12. Build-image size (LOW). Heavily-templated quant TUs; if the HIP link overflows the 2GiB reach, add `--offload-compress` (guide cudf lesson). Most quant TUs are disabled on the first pass, so unlikely.

## File-by-file change list (first AMD pass, correctness-first)
- CMakeLists.txt: add `USE_HIP` option; HIP language + `CMAKE_HIP_ARCHITECTURES` (default gfx90a when unset) + `set_source_files_properties(... LANGUAGE HIP)`; define `USE_HIP`/`USE_ROCM`; make flash-attn discovery non-fatal under HIP (ENABLE_FLASH_MHA=0); gate deep_gemm/flash_mla/tensorrt_llm link blocks behind `NOT USE_HIP`; `-ffp-contract=on` in HIP flags.
- 3rd/bmengine/bmengine/CMakeLists.txt: same USE_HIP gating; cublas/cublasLt/curand/culibos/nccl -> hipblas/hipblaslt/hiprand/(drop culibos)/rccl find+link under HIP.
- NEW src/utils/cuda_to_hip.h (+ a bmengine-visible variant): the single compat header (cstring/cstdlib before hip_runtime; symbol + handle aliases; bf16/fp8 type aliases; `kWarpSize` per-arch constant).
- 3rd/bmengine/.../functions/reduce.cuh: wave-agnostic warp/block reduce (kWarpSize-driven loop, full-width masks, sized shared[]). Highest-leverage single edit.
- 3rd/bmengine/.../functions/{sort,topk}.cu: Thrust/CUB -> rocThrust/hipCUB; sync between reused block collectives.
- src/nn/attention/attention_kernel.cu (+ .h): guard `<mma.h>` and the wmma tensor-core path off under HIP; rework or 32-lane-pin the `DIM_HEAD==WARP_SIZE*4` scalar path for wave64.
- src/nn/attention/quant_attention.cuh: guard the prmt/sub.f16x2 PTX off under HIP (int8 attention dequant -> portable path).
- src/nn/attention/{ds_flash_mla_api.cpp, multi_head_latent_attention.cpp}: ensure MLA falls back to the generic path when flash_mla disabled.
- src/nn/linear/linear.cpp: guard the `deep_gemm_fp8_block_h20_group` calls behind `ENABLE_DS_DEEP_GEMM` (already are) which is OFF on HIP.
- src/nn/quant/marlin/*.cu + marlin.cuh, src/nn/quant/awq/*.cu + dequantize.cuh, src/nn/quant/fp8/fp8_util.cu: `#if !defined(USE_HIP)` around the tensor-core/PTX kernels; route call sites to the hipBLASLt dequant+GEMM fallback. (Reimplement-not-port; deferred to a later AMD-native pass.)
- src/nn/quant/gptq/{q_gemm.cu,compat.cuh}: the existing USE_ROCM/hipblas shims become live; verify they compile on ROCm 7.2.1 (vllm lineage, likely close); wave64-check the WARP_SIZE-32 loops.
- src/nn/quant/int8/{quant_kernel,quant_reduce_kernel}.cu: wave64-check the {32,32} block grouping reductions.
- 3rd/bmengine/.../functions/gemm.cpp, src/nn/linear/gemm_grouped.cpp: cublasLt -> hipBLASLt under HIP (handle types, algo heuristic API, fp8 dtype fallback).
- 3rd/bmengine/.../core/{context,engine}.cpp, c10d/c10d.cpp: NCCL -> RCCL handle/enum swaps under HIP.
- Mechanical hip* swaps across the remaining elementwise/norm/rope/embedding/kvcache .cu via the compat header (no per-file logic change).

## Build commands (gfx90a)
Prereqs (planner-installable, not asking jeff): ROCm 7.2.1 toolchain, a ROCm PyTorch in the env (for torch/include + torch/lib ABI and the tests' find_package(Torch)), pybind11, ninja. flash-attn wheel NOT required under USE_HIP.
```
# configure (Strategy A: HIP language, arch from cache var)
ENABLE_NCCL_TP=on TESTING=1 \
cmake -S projects/ZhiLight/src -B agent_space/zhilight-build -GNinja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release -DWITH_TESTING=ON \
  -DPYTHON_EXECUTABLE=$(which python3)
cmake --build agent_space/zhilight-build -j32 --target C internals_
# or the project's own path:
CMAKE_GENERATOR=Ninja TESTING=1 CMAKE_ARGS="-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a" \
  python setup.py bdist_wheel   # run from src/, intermediates under agent_space
```
Followers reuse the same commit with only `-DCMAKE_HIP_ARCHITECTURES=gfx1100` / `gfx1151` (no source edit) per the configurable-arch lesson.

## Test plan
Real GPU functional tests (self-contained, torch-reference, NO model weights -- the validation gate):
- pytest tests/test_attention.py tests/test_attn_softmax.py tests/test_softmax.py tests/test_rotary_embedding.py tests/test_feedforward.py tests/test_linear.py tests/test_embedding.py tests/test_arthmetic.py tests/test_concat_tensor.py tests/test_index_along_dim.py tests/test_log_prob.py
  (these import `zhilight.internals_` -> the `internals_` pybind module that wraps the C++ kernels on GPU; they are the primary correctness gate and directly exercise the wave64-sensitive reduce/attention/softmax/ff paths).
- C++ gtest-style: agent_space/zhilight-build/.../test_attention_rag_buffer (src/nn/tests).
Model-level (only if weights available; not required to pass the gate): tests/test_llama.py, tests/test_mistral.py.
Non-GPU regression set (must not break): the pure-python zhilight/ package import, server entrypoint import, lazy_loader test (tests/test_lazy_loader.py). No CPU-only unit suite beyond these.
Note quant fast paths are disabled on this pass: any GPTQ/AWQ/Marlin/fp8-block specific test would be skipped/xfail on AMD until the AMD-native quant pass; the python tests above do not require them.

## Open questions
- Does ROCm 7.2.1 hipBLASLt on gfx90a expose the algo-heuristic + fp8 (E4M3) matmul features gemm.cpp relies on, or must fp8 GEMM fall back to bf16? (resolve at porter bringup.)
- Is a ROCm PyTorch the right torch to link for the flash-attn-free HIP build, and does dropping the flash_attn .so leave the prefill dispatch fully functional (the in-tree attention path covers it)? Confirm prefill path at runtime.
- DeepSeek-V3 FP8-block + MLA are the headline 2025 features and are exactly the Hopper-only (deep_gemm/flash_mla H20) parts being disabled -- confirm with jeff whether a correctness-first port that drops DeepSeek-V3 FP8/MLA on AMD is the intended first deliverable (it is the right call per the reimplement-not-port guidance, but it is a visible feature gap).
