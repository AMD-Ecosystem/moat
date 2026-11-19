# ntransformer -- ROCm/HIP Port Plan (lead: linux-gfx90a)

## Project
- Name: ntransformer
- Upstream: https://github.com/xaskasdf/ntransformer
- Default branch: main
- Head at clone: f2237be ("Async NVMe prefetch: overlap BAR1 reads with GPU compute")
- What it is: a from-scratch C++20/CUDA LLM inference engine (Llama architecture: RoPE, GQA, SwiGLU, RMSNorm, F16 KV cache) that streams GGUF model layers through limited VRAM via a 3-tier cache (VRAM-resident / pinned-RAM H2D / NVMe-or-mmap). All kernels are hand-written GEMV/softmax/attention/norm; "zero external dependencies beyond CUDA Toolkit (no PyTorch, no cuBLAS)".

## Existing AMD support -> decision
- None. The repo has no HIP/ROCm path, no `USE_HIP`, no `gfx`/wavefront references, no OpenCL/Vulkan/SYCL backend. It is CUDA-only and was recently ported Windows/MSVC/CUDA 12.4 -> Linux/gcc-14/CUDA 13.1.
- Decision: PROCEED with a fresh CUDA->HIP port. A ROCm/HIP path adds clear value (no AMD support exists today).
- Port-vs-rewrite: PORT (mechanical, correctness-first). The kernels are plain hand-written CUDA C (GEMV with warp-reduction dot products, online-softmax attention, RMSNorm, RoPE, SwiGLU). There is NO CUTLASS/CuTe, NO Hopper wgmma/MMA, NO cp.async/mma PTX, NO tensor-core/rocWMMA path -- so the "reimplement-not-port" flag does NOT apply. An AMD-native MFMA/CK GEMM rewrite is a possible later perf pass but is out of scope for the correctness port (and the engine is PCIe/NVMe-bandwidth bound by design, not compute bound, so GEMV perf is not the bottleneck).

## Build classification -> Strategy A (pure CMake)
- Evidence: top-level `CMakeLists.txt` line 2 `project(ntransformer LANGUAGES CXX CUDA)`, `find_package(CUDAToolkit REQUIRED)` (line 37), explicit `CORE_SOURCES` (.cpp, g++) vs `CUDA_SOURCES` (.cu, nvcc) split (lines 44-69), links only `CUDA::cudart` (line 87). No `find_package(Torch)`, no `torch.utils.cpp_extension`, no `CUDAExtension`, no setup.py/pyproject. -> pure CMake, Strategy A (colmap compat-header model).
- ext_type = cmake.

## Port strategy (Strategy A: compat header + LANGUAGE HIP)
1. Add one `src/core/cuda_to_hip.h` compat header. On HIP (`USE_HIP`/`__HIP_PLATFORM_AMD__`) include `<hip/hip_runtime.h>` and alias only the symbols this project uses (full list in CUDA surface below); on NVIDIA it is a no-op `#include <cuda_runtime.h>`. Force-include it on every HIP TU via `CMAKE_HIP_FLAGS -include .../cuda_to_hip.h` so it precedes each .cu's own includes (the .cu files include `<cuda_runtime.h>` / `<cuda_fp16.h>` directly).
2. CMake: add `option(USE_HIP ...)`; under it `enable_language(HIP)`, default `CMAKE_HIP_ARCHITECTURES` to `gfx90a` ONLY when unset (never a literal -- so followers build with `-DCMAKE_HIP_ARCHITECTURES=gfx1100` and no source edit, per guide), `set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE HIP)`, set the target's `HIP_ARCHITECTURES` from the cache var, link `hip::host` instead of `CUDA::cudart`. Keep the `else() enable_language(CUDA)` path byte-for-byte.
3. The two NVMe paths (`USE_GPUNVME`, `test_nvme_layer.cu`) stay OFF -- they need the external `gpu-nvme-direct` library plus VFIO/root/IOMMU-off host setup and are not GPU-validatable in this environment. The default build (`USE_GPUNVME=OFF`) covers all kernels and the unit tests.

## CUDA surface inventory
Runtime API (all 1:1 HIP spellings, add to compat header):
- Device/stream/event: cudaSetDevice, cudaGetDeviceCount, cudaGetDeviceProperties (+ cudaDeviceProp), cudaDeviceSynchronize, cudaStream_t, cudaStreamCreateWithFlags, cudaStreamNonBlocking, cudaStreamDestroy, cudaStreamSynchronize, cudaStreamWaitEvent, cudaEvent_t, cudaEventCreate, cudaEventCreateWithFlags, cudaEventDisableTiming, cudaEventDestroy, cudaEventRecord, cudaEventSynchronize, cudaEventElapsedTime.
- Memory: cudaMalloc, cudaFree, cudaMallocHost, cudaFreeHost, cudaMemcpy (+ kinds H2D/D2H/D2D), cudaMemcpyAsync, cudaMemset, cudaMemGetInfo, cudaHostRegister (+ cudaHostRegisterReadOnly).
- Func attr: cudaFuncSetAttribute, cudaFuncAttributeMaxDynamicSharedMemorySize.
- Error: cudaError_t, cudaSuccess, cudaErrorMemoryAllocation, cudaGetErrorString.
- FP16: `<cuda_fp16.h>` -> `<hip/hip_fp16.h>`; `half`, `__half2float`, `__float2half`, `half2` (gemm.cu vectorized F16 loads).
Kernels (all in `namespace nt::cuda`): gemv_q4_0/q8_0/q4_k/q5_k/q6_k/f16/f16_add/f32, gemm_f32, add_bias, silu_elementwise_mul, rmsnorm(+_f16), rope(+interleaved), softmax(+masked), attention_decode(fast HEAD_DIM-templated + generic), attention_prefill, copy_to_kv_cache, add/add_inplace/copy, cosine_similarity.
Warp intrinsic: `__shfl_xor_sync` only (no __ballot/__activemask/__any/__all/__popc).
Streams: STREAM_COMPUTE + 2 transfer streams. Events used for SLEP pipeline timing/ordering.
NOT present (confirmed by grep): cuBLAS, cuFFT, cuRAND, cuSPARSE, cuDNN, Thrust, CUB, NCCL, NVTX, textures/surfaces/cudaArray, cudaMallocManaged, cooperative groups / cooperativeLaunch, CUDA graphs, __launch_bounds__. -> no library swaps, no texture fault classes.

## Risk list (gfx90a wave64 first; RDNA wave32 noted for followers)
1. `__CUDACC__` vs `__HIPCC__` FP16 guard (HIGH, will break the build). `src/core/types.h:11-16` does `#ifdef __CUDACC__  #include <cuda_fp16.h>; using float16_t=half; #else using float16_t=uint16_t;`. hipcc defines `__HIPCC__`/`__HIP__`, NOT `__CUDACC__`, so under HIP every `.cu` would take the `uint16_t` branch while the kernels use `half`/`__half2float`/`__float2half` -> compile failure. Fix: change the guard to `#if defined(__CUDACC__) || defined(__HIPCC__)` and route the include through the compat header (`<hip/hip_fp16.h>` on HIP). Host .cpp keep `uint16_t` (they only use float16_t as a 2-byte KV-cache storage type, never half arithmetic -- verified in transformer.cpp), so the host path is unchanged. (MPPI-Generic __CUDACC__ trap.)
2. `__shfl_xor_sync(0xFFFFFFFF, v, offset)` masks (MEDIUM, verify on GPU). Two idioms:
   - rmsnorm/softmax/attention reductions start `offset = warpSize/2` and reduce across the FULL physical wavefront. These are correctly wave-agnostic (use the `warpSize` builtin, not a literal 32). On wave64 the first offset is 32, reducing all 64 lanes -- correct. The 32-bit mask `0xFFFFFFFF` is accepted by HIP (mask is effectively ignored / all-lanes on ROCm). Expected correct; confirm via test_gemm rmsnorm + a decode run.
   - GEMV kernels (gemm.cu) launch `dim3 block(32, GEMV_WARPS=8)` -- 32 threads per warp-row -- and reduce with a HARDCODED `offset = 16` start and `* 32` lane layout (`flat_id = warp_id*32+tid`, `b += 32`). This is the LOGICAL-warp-of-32 pattern: on wave64 two 32-rows pack into one 64-lane wavefront (lanes 0-31 / 32-63), but every shfl offset is <= 16 so the XOR butterfly never crosses the 32-lane boundary -> each 32-row reduces independently and correctly (the popsift "two 32-rows per wavefront" case, here SAFE because offset<=16 and the mask is full). This is arch-agnostic and fine on both wave64 and wave32. No change needed; this is the central thing the GPU test must confirm (test_gemv_f32/q4_0/q6_k check exact dot-product sums).
3. Cross-warp shared-mem sizing (LOW). rmsnorm/softmax/attention use `__shared__ float shared[32]` indexed by `wid = tid/warpSize`. Max block is 1024 threads -> wave32: 1024/32 = 32 warps (exactly fits 32); wave64: 1024/64 = 16. Safe at both. No change. (Do NOT shrink this array; 32 is the correct wave32 upper bound here.)
4. Dead hardcoded-32 array (LOW, cosmetic). attention.cu:50 `float acc[HEAD_DIM/32];` is unused (real accumulator is `local_acc[ELEMS_PER_THREAD]`). Harmless on any arch; leave it or drop it -- not a correctness issue.
5. `cudaHostRegister(..., cudaHostRegisterReadOnly)` on mmap'd GGUF region (LOW). 1:1 hipHostRegister + hipHostRegisterReadOnly; failure path already falls back to staging buffers. Exercised only in --streaming; not in unit tests.
6. `--use_fast_math` in CUDA flags (LOW). The HIP build should pass `-ffast-math` (or leave default) and, per the guide, consider `-ffp-contract=on` only if a bit-exact compare later drifts; the unit tests use loose tolerances (1e-3..0.5), so FMA-contraction drift is a non-issue here. rsqrtf is the only fast-math-sensitive call (rmsnorm) and the test tolerance is 0.01.
7. `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, 64KB)` (LOW). gfx90a LDS is 64 KB/CU; the 64 KB request is at the limit and the kernels already have a no-smem fallback for >64 KB (gemv_q*_k<false>). Confirm hipFuncSetAttribute accepts 64 KB on gfx90a (it should); the large-input test (test_gemv_q6_k_large) deliberately exercises the no-smem path.
8. C++ standard (LOW). Project is C++20 host + C++20 CUDA. hipcc/clang supports C++20; no rocThrust/hipCUB (which would force >=C++17) is used, so no std-bump conflict. Keep CMAKE_HIP_STANDARD 20.
9. Followers (wave32, gfx1100/gfx1151): items 2-3 are the watch points. The build is the test -- compile `-DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100"` and confirm both code objects emit; then validate a wave32 run. Delta plan to be appended on demand.

## File-by-file change list
- `CMakeLists.txt`: add `option(USE_HIP OFF)`; `if(USE_HIP) enable_language(HIP) ... else() enable_language(CUDA) endif()`; default+propagate CMAKE_HIP_ARCHITECTURES (no literal); `set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE HIP)`; link `hip::host` vs `CUDA::cudart` per backend; add `CMAKE_HIP_FLAGS -include .../cuda_to_hip.h`; keep test targets, gate test_nvme_layer behind USE_GPUNVME as today. (NEW small additive block, CUDA path unchanged.)
- `src/core/cuda_to_hip.h`: NEW compat header (runtime + fp16 aliases listed above), guarded `#if defined(USE_HIP)||defined(__HIP_PLATFORM_AMD__)`.
- `src/core/types.h`: change `#ifdef __CUDACC__` (line 11) to `#if defined(__CUDACC__) || defined(__HIPCC__)`; on HIP include `<hip/hip_fp16.h>` (via compat header). This is the only edit inside a shared header.
- (Optional) `src/cuda/attention.cu:50`: drop the unused `acc[HEAD_DIM/32]`.
- Everything else (.cu kernels, .cpp host): NO source edits expected -- runtime symbols are 1:1, warp code is wave-agnostic or correct-by-offset.

## Build commands (gfx90a)
```
cd projects/ntransformer/src
cmake -S . -B build-hip -DUSE_HIP=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build build-hip -j
```
Multi-arch correctness build (the warp-size test, per guide):
```
cmake -S . -B build-multi -DUSE_HIP=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100"
cmake --build build-multi -j
llvm-objdump --offloading build-multi/libntransformer_lib.a 2>/dev/null  # expect gfx90a AND gfx1100 (or check the test exe)
```
Note: drop `-march=native` from the HIP host flags if clang rejects it (it is in CMAKE_CXX_FLAGS_RELEASE / CMAKE_CUDA_FLAGS_RELEASE; clang accepts -march=native, so likely fine).

## Test plan
- GPU-validatable slice (this is the validator's gate): `build-hip/test_gemm` -- runs on real gfx90a, exercises gemv_f32, gemv_q4_0, gemv_q6_k (smem path), gemv_q6_k_large (no-smem path), silu_mul, rmsnorm with numeric tolerance checks. This directly covers the warp-reduction (`__shfl_xor_sync`) and the cross-warp `shared[32]` paths -- the only real risk surface. A clean PASS here is strong correctness evidence on wave64.
- `build-hip/test_tensor` -- tensor/allocator (CPU + GPU malloc/copy via nt_cuda_*). Must pass; covers the runtime-API aliases.
- These map to ctest targets `test_tensor` and `test_gemm` (CMakeLists 131-137); run `ctest --output-on-failure` (serially -- single assigned GPU, per guide).
- End-to-end (stretch, needs a small GGUF, e.g. a tiny Llama Q8_0): `./ntransformer -m <model>.gguf -p "Hello" -n 16` resident mode -- exercises rope, attention decode/prefill, softmax, the full forward loop. Compare token output / logits sanity vs the CUDA build if a CUDA GPU reference is available; otherwise assert coherent (non-garbage, non-NaN) generation. This is the broadest correctness check for the attention/rope/softmax kernels that the unit tests do not cover.
- Non-GPU regression set: the CPU portions of test_tensor (tensor metadata/shape/stride asserts) must not regress; they are vendor-independent.
- Out of scope for validation: NVMe direct path (USE_GPUNVME) and --streaming NVMe (external lib + VFIO/root); the mmap-staging streaming path could be smoke-tested with a small model but is bandwidth plumbing, not a kernel correctness gate.

## Open questions
- Is a small GGUF model (tiny Llama, Q8_0/Q6_K) available on the validator host for the end-to-end decode check? If not, the unit tests + a synthetic-weight forward smoke remain the gate; attention/rope/softmax then rest on code review + the online-softmax math being standard. Flag for the validator.
- `cudaFuncSetAttribute` 64 KB dynamic-smem request: confirm hipFuncSetAttribute returns success on gfx90a (64 KB == one LDS bank limit). If it errors, the kernels already fall back to the no-smem path, but the attribute call itself should be made non-fatal on HIP if needed.
