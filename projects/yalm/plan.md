# yalm -- ROCm/HIP port plan (lead: linux-gfx90a)

## Project
- Name: yalm (Yet Another Language Model) -- from-scratch LLM inference engine in C++/CUDA, no external inference deps.
- Upstream: https://github.com/andrewkchan/yalm
- Default branch: main
- Planned base sha: 6cd1ef6e7d6b9f724b6d98a63b273f903928ad2f
- ext_type: cmake (closest fit; see Build classification -- it is actually a hand-written Makefile + nvcc, no CMake and no torch).

## Existing AMD support
None. README states "An NVIDIA GPU is required"; build hardcodes `nvcc` and `-gencode arch=compute_80/compute_90`, links `-lcudart`. No OpenCL/Vulkan/SYCL/HIP path exists. A ROCm/HIP port adds clear value.
Decision: PROCEED with a fresh, correctness-first CUDA-to-HIP port.

## Build classification
- cmake-strategy class (Strategy A spirit), but there is NO CMake: the project builds via a hand-written `Makefile` that calls `nvcc` directly on `.cu` files and `$(CXX)` on `.cpp/.cc/.c`.
- Evidence: `Makefile` lines 4 (`NVCC?=nvcc`), 37 (`LDFLAGS+=-lcudart`), 44-51 (`CUFLAGS += -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90`), 88-90 (`$(BUILD)/%.cu.o: %.cu` -> `$(NVCC) ...`). `requirements.txt` has torch only for the offline `convert.py` weight converter, NOT for the build; no `find_package(Torch)`, no `CUDAExtension`. So this is NOT a pytorch extension -> Strategy B does not apply.
- Single CUDA translation unit: `src/infer.cu` (all device code + the CUDA host glue). Everything else is plain host C++.

## Port strategy: A (compat header) + Makefile HIP path -- mechanical, correctness-first
Rationale: it is a pure-CUDA standalone project with all device code in one `.cu` and a tiny set of CUDA runtime symbols. The minimal-footprint compat-header approach applies, adapted to a Makefile:
1. Add one compat header `src/cuda_to_hip.h`: on HIP (`USE_HIP`/`__HIP_PLATFORM_AMD__`) `#include <hip/hip_runtime.h>` + `<hip/hip_fp16.h>` and alias only the symbols yalm uses (cudaMalloc/Free/Memcpy/MemcpyAsync, cudaMallocHost/HostAlloc/HostRegister/HostUnregister, cudaSetDevice/DeviceGetAttribute and the attr enums cudaDevAttrWarpSize/cudaDevAttrMaxThreadsPerBlock, cudaStream*/cudaStreamCreate/cudaStreamSynchronize/cudaStreamLegacy/cudaStreamCaptureModeGlobal, the cudaGraph*/cudaKernelNodeParams/cudaStreamGetCaptureInfo_v2/cudaStreamUpdateCaptureDependencies graph API, cudaGetErrorString/Name/LastError/DeviceSynchronize, cudaSuccess/cudaError_t, cudaMemcpyHostToDevice/DeviceToHost, cudaHostRegisterDefault). Per PORTING_GUIDE, include `<cstring>`/`<cstdlib>` BEFORE `<hip/hip_runtime.h>` so host memcpy/memset libc decls win inside the .cu. On non-HIP it includes `<cuda_runtime.h>`+`<cuda_fp16.h>` and is a no-op -- NVIDIA build untouched.
   - `src/infer.cu` currently `#include <cuda_fp16.h>`; `src/model.h` `#include "cuda_runtime_api.h"`. Route both through the compat header (model.h needs the `cudaStream_t`/`cudaGraph_t`/`cudaKernelNodeParams` types for the host structs `CudaGraph`/`InferenceState`, so the header must provide them on HIP too -- it is included by host TUs, so keep the host-safe subset (types + enums) compilable without the HIP device path).
2. Makefile: add a `USE_HIP=1` path. When set, compile `.cu` with `hipcc` (`HIPCC?=hipcc`), `-x hip --offload-arch=$(HIPARCH)` (default gfx90a, overridable so gfx1100/gfx1151 need no edit), drop `-gencode`, link `-lamdhip64` instead of `-lcudart`, and pass `-DUSE_HIP`. Keep the host `.cpp` compile flags as-is (x86 `-mf16c -mavx2 -mfma` stay for the CPU backend on this x86 host). Do NOT rename `.cu` files.
3. Guard genuinely divergent code with `#if defined(USE_HIP)`; keep guards rare (the wave64 fixes below are mostly wave-agnostic rewrites, not guards).

Port vs rewrite: all kernels are simple hand-written GEMV/attention/softmax/RoPE/RMSNorm; NO CUTLASS/CuTe/wgmma/warp-specialization (confirmed: grep for cublas/cufft/curand/cusparse/thrust/cub/cutlass/cudnn = NONE). A mechanical HIP port is the correct first deliverable. An AMD-native (rocWMMA/MFMA/CK) perf rewrite of matmul/att is explicitly OUT OF SCOPE for correctness validation; note as possible follow-up only.

## CUDA surface inventory (all in src/infer.cu unless noted)
- Kernels (`__global__`): matmul, matmul_wide, fused_matmul_add_residuals, fused_qkv_matmul_clip, attn_dot, attn_softmax, att_mix, rmsnorm, fused_ffn_w1_w3_glu_act, copy_embedding_float, copy_embedding_half, fused_rope_and_cache_update, rotate_sink_tokens.
- Device helpers (`__device__`): blocktranspose, warp_reduce_sum, warp_all_reduce_max, block_all_reduce_max, warp_all_reduce_sum, block_all_reduce_sum, matmul_row (float + half overloads), rope (3 overloads), act<SILU/GELU>.
- Warp intrinsics: `__shfl_down_sync`, `__shfl_xor_sync`, all with `FULL_MASK = 0xffffffff` (32-bit). HIP accepts the `_sync` forms and ignores the mask (operates on the full wave); the reduction trip counts use `warpSize/2` (runtime) so the reductions are wave-width-agnostic AS WRITTEN. Risk is the block GEOMETRY around them, not the shuffles -- see Risk list.
- warpSize: runtime-queried into `static int warp_size` via `cudaDeviceGetAttribute(cudaDevAttrWarpSize)` (good -- 64 on gfx90a). Device code uses the builtin `warpSize`. Both correct on HIP. BUT three of the host launch sites and the `mha_cuda`/`matmul_cuda`/`ffn_cuda` test entry points hardcode `int warp_size = 32;` (lines 972, 1016, 1042) and `max_threads_per_block = 1024` -- these feed the TEST path the validator uses; must become 64-aware.
- fp16/bf16: device uses `cuda_fp16.h` `half`/`half2` and intrinsics `__half2float`, `__half22float2`, `__floats2half2_rn`, `__float2half`. HIP `hip_fp16.h` provides all 1:1. `f16_t` is just `uint16_t` (codec.h) -- a storage type, reinterpreted to `half*` at the boundary. No `__nv_bfloat16` in device code (DType::BF16 exists in the enum but the CUDA backend asserts F32/F16 only).
- cuBLAS/cuFFT/cuRAND/cuSPARSE/Thrust/CUB/CUTLASS/cuDNN: NONE. No library swaps needed.
- Textures/surfaces: NONE. (No texture pitch / layered-array / linear-filter fault classes apply.)
- Pinned/host memory: `cudaMallocHost`, `cudaHostAlloc`, `cudaHostRegister`/`Unregister` -> hipMallocHost/hipHostMalloc/hipHostRegister/Unregister (1:1).
- Streams: `cudaStreamCreate`, `cudaStreamSynchronize`, `cudaStreamLegacy`, `cudaStreamCaptureModeGlobal` -> hip* 1:1. `cudaStreamLegacy` -> `hipStreamLegacy` exists in ROCm.
- CUDA Graphs (the real runtime-feature risk): `cudaStreamBeginCapture`/`EndCapture`, `cudaGraphInstantiate`, `cudaGraphLaunch`, `cudaGraphExecKernelNodeSetParams`, `cudaGraphAddKernelNode`, `cudaStreamGetCaptureInfo_v2`, `cudaStreamUpdateCaptureDependencies`, `cudaKernelNodeParams`, `cudaGraphNode_t`. All have hip equivalents (hipGraph*, hipStreamGetCaptureInfo_v2, hipStreamUpdateCaptureDependencies, hipKernelNodeParams). Used only in the full-model `_forward_cuda` path (main binary), NOT in the test binary's kernel comparisons (`mha_cuda`/`matmul_cuda`/`ffn_cuda` launch kernels directly on `cudaStreamLegacy`). So the GPU validation slice does not depend on graph capture working -- de-risks the gate.
- Atomics: `atomicAdd` on `__shared__ float` in att_mix -- supported on gfx90a.
- Host-side x86 F16C (`immintrin.h`/`f16cintrin.h`, `_cvtss_sh`/`_mm_cvtss_f32`) in infer.cpp/model.cpp/test.cpp: CPU backend only, compiled by host CXX with `-mf16c`. Unaffected by the GPU port on this x86 host (would matter only on a non-x86 ROCm host; out of scope).

## Risk list (ordered by likelihood of a real fault)
1. HARD LAUNCH FAILURE -- `*32` block size overflows the 1024 thread cap on wave64. Lines 918, 958 (`fused_matmul_add_residuals<<<c.dim/32, warp_size*32>>>`) and 1191, 1197 (`matmul_wide<<<c.vocab_size/32, warp_size*32>>>`). On NVIDIA `warp_size*32 = 32*32 = 1024` (legal). On gfx90a `64*32 = 2048` > 1024 max threads/block -> `hipLaunchKernel` returns invalid configuration, kernel never runs. Root cause: the literal `32` is "warps per block" (WPB), chosen so blocktranspose coalesces 32 results into one 128B store; it was conflated with the assumption warp=32. Fix: introduce an explicit `WPB` (warps-per-block) that satisfies `WPB * warp_size <= max_threads_per_block` (e.g. WPB=16 on wave64 -> 1024 threads), use it consistently in BOTH the grid divisor (`c.dim/WPB`, `c.vocab_size/WPB`) and the block dim (`warp_size*WPB`), and in `blocktranspose`/`matmul_wide`/`fused_matmul_add_residuals` (`block_start_i = blockIdx.x * WPB`, predicate `threadIdx.x < WPB`, `sm[]` capacity >= WPB). NOTE the grid divisor must divide evenly -- `c.dim`/`c.vocab_size` are multiples of 32 upstream (32000 vocab, 4096 dim); pick WPB dividing them (16 works). This is the single highest-risk item and affects the classifier (logits) and the wo/w2 projections.
2. SHARED-MEM OOB + wrong pairing in `att_mix` (lines 424-522). `__shared__ float shared0[32]/shared1[32]` indexed by `threadIdx.x`, but the kernel is launched with `tpb.x = warp_size` (= 64 on gfx90a), so `shared0[threadIdx.x]` writes indices 0..63 into a 32-slot array -> shared-mem OOB / corruption. The loop stride `i += 2*warpSize` and the comment "NOTE: Assumes warpSize is 32" confirm the 32-lane assumption. Fix: size the shared arrays to `warp_size` (or a compile-time upper bound of 64 per PORTING_GUIDE C10_WARP_SIZE_UPPER_BOUND guidance) and keep the `threadIdx.x`-indexed addressing consistent with the wider wave; verify the 2-element-per-lane chunking still tiles head_dim correctly at lane count 64. This kernel feeds `mha_cuda` which the validator's `test_cuda_kernels()` directly checks against CPU -- a bug here fails the gate loudly (good).
3. Test-path hardcoded `int warp_size = 32` in `mha_cuda` (972), `matmul_cuda` (1016), `ffn_cuda` (1042) and `max_threads_per_block = 1024`. These are the entry points the GPU validation test calls. On gfx90a the kernels' device-side `warpSize` is 64, so a host launch with `warp_size=32` threads under-populates each logical warp: matmul (`<<<d,32>>>`) launches 32-thread blocks while `matmul_row` strides by `warpSize`(=64) and `warp_reduce_sum` reduces over 64 lanes -> half the lanes are absent, partial sums wrong. Fix: query warpSize at runtime in these helpers too (or pass the global `warp_size` set by `set_cuda_device`); the test harness must call `set_cuda_device(0)` first (confirm `test.cpp` does -- it does NOT currently; the test entry points self-initialize the hardcoded constants). Plan: replace the hardcoded 32/1024 in these three functions with a runtime query (hipDeviceGetAttribute) performed lazily on first use.
4. `blocktranspose` `sm[32]` and `block_all_reduce_max/sum` `shared[32]`: capacity 32 = "max warps per block". With WPB capped at 16 (item 1) and rmsnorm/softmax blocks <= 1024 threads / 64 = 16 warps, 32 is a safe upper bound on wave64 -- no change strictly required, but document the invariant `blockDim.x/warpSize <= 32`. Confirm after the WPB fix that no block exceeds 32 wavefronts.
5. CUDA Graph capture semantics on ROCm. `_forward_cuda` builds and replays a hipGraph via stream capture with manual `hipGraphAddKernelNode` + `hipGraphExecKernelNodeSetParams` and `hipStreamGetCaptureInfo_v2`/`hipStreamUpdateCaptureDependencies`. These exist in ROCm 7.2.1 but capture/replay edge cases (legacy-stream capture, node-param update) are a known divergence area. MITIGATION: the GPU validation gate (test binary) does NOT use graphs, so graph issues cannot block validation; they would surface only when running the full `./build/main` model. Treat full-model generation as a secondary check, not the gate. If graph replay misbehaves, a fallback is a non-graph dispatch path (the kernels are already individually launchable, as the test path proves).
6. `cudaStreamLegacy` -> `hipStreamLegacy`: confirm the symbol is provided by ROCm 7.2.1 hip_runtime; if absent, map to `0`/default stream in the compat header (the test path uses it).
7. FMA/`-ffast-math` numeric drift. Host Makefile uses `-ffast-math`; hipcc defaults to `-ffp-contract=fast`. The test compares GPU vs CPU with epsilon 1e-4 (`arrayEquals`), so it tolerates modest FMA drift -- low risk, but if matmul/ffn compares fail marginally, pin `-ffp-contract=on` per PORTING_GUIDE (CV-CUDA). Watch first before adding the flag.
8. Even-divisibility of grid: `c.dim/WPB` and `c.vocab_size/WPB` use integer division; upstream relied on dim%32==0. Choosing WPB that divides the upstream dims (16) preserves this; otherwise rows are dropped. Verify against the validated model configs (Mistral dim=4096, vocab=32000; head_dim=128).
9. `register_cuda_host` on a `std::vector::data()` then `unregister` (mha_cuda) -- hipHostRegister on pageable memory: fine, but note PORTING_GUIDE's pageable-async-copy class -- here copies are synchronous (`cudaMemcpy`, and `cudaMemcpyAsync` immediately followed by use within a sync'd path), low risk.

## File-by-file change list
- src/cuda_to_hip.h (NEW): compat header, HIP aliases + host-safe type/enum subset; libc includes before hip_runtime; `<hip/hip_fp16.h>` on HIP.
- src/model.h: replace `#include "cuda_runtime_api.h"` with `#include "cuda_to_hip.h"` (provides cudaStream_t/cudaGraph_t/cudaKernelNodeParams/cudaGraphNode_t/cudaGraphExec_t for the host structs).
- src/infer.cu: replace `#include <cuda_fp16.h>` with `#include "cuda_to_hip.h"`. Apply wave64 fixes: items 1 (WPB), 2 (att_mix shared sizing + stride), 3 (runtime warpSize in the 3 test entry points). Optionally guard cudaStreamLegacy mapping.
- Makefile: add `USE_HIP` branch (HIPCC, `-x hip --offload-arch=$(HIPARCH)` default gfx90a, `-DUSE_HIP`, `-lamdhip64`, drop `-gencode`/`-allow-unsupported-compiler`). Keep CUDA path as default/unchanged.
- src/notes.md (MOAT side): record build + test commands and the WPB invariant.

## Build commands (gfx90a)
CPU+CUDA build is unchanged (`make`). HIP build:
```
# from projects/yalm/src (do NOT commit build/ -- it is gitignored; build in a throwaway tree if needed)
make clean
make USE_HIP=1 HIPARCH=gfx90a            # builds ./build/main
make test USE_HIP=1 HIPARCH=gfx90a       # builds ./build/test (the GPU validation binary)
```
Followers reuse the same source with `HIPARCH=gfx1100` / `HIPARCH=gfx1151` -- no source edit, no new fork commit (the arch is a make var).

## Test plan
REAL GPU GATE (no model download needed):
```
make test USE_HIP=1 HIPARCH=gfx90a
./build/test          # runs test_attn() [CPU] + test_cuda_kernels() [GPU vs CPU]
```
`test_cuda_kernels()` (src/test.cpp:148-234) is a self-contained CPU-vs-GPU correctness comparison at epsilon 1e-4 for:
  - matmul (matmul_cuda<float>) vs matmul_cpu,
  - mha (mha_cuda -> attn_dot/attn_softmax/att_mix) vs mha_cpu,
  - ffn (ffn_cuda -> fused_ffn_w1_w3_glu_act + matmul) vs ffn_cpu.
This directly exercises the wave64-risk kernels (matmul reductions, att_mix shared-mem, softmax block reduce) against a CPU gold on real hardware -- it is the validation gate. Expected output: "All tests passed". A pass here means the warp-reduction, att_mix shared sizing, and matmul geometry are correct on wave64.

Kernel benches (optional perf sanity, run on GPU, not a correctness gate):
```
./build/test -bk matmul ; ./build/test -bk matmul-wide ; ./build/test -bk mha ; ./build/test -bk ffn
```
matmul-wide and the bench dims (vocab 32000, hidden 14336, dim 4096) additionally exercise the `matmul_wide`/WPB path (item 1) on realistic sizes.

SECONDARY (end-to-end, needs a converted model; do if a small model is available -- run only after the gate passes, NOT the gate because it depends on hipGraph replay item 5):
```
python convert.py --dtype fp16 model.yalm <hf-dir>     # offline, needs torch (requirements.txt)
./build/main model.yalm -d cuda -m perplexity -i "<prompt>"   # perplexity is a numeric correctness signal
./build/main model.yalm -d cuda -i "What is a large language model?" -m c   # generation
```
Perplexity mode (main.cpp run_perplexity) gives a quantitative CPU-vs-expected signal and is the best end-to-end correctness check if a tested model (Llama-3.2, Mistral-v0.2) is on hand.

NON-GPU REGRESSION SET (must not break): `test_attn()` (CPU attention, runs unconditionally in the default test invocation) and the CPU backend path (`-d cpu`) of main. The CPU build path and host `.cpp` flags are untouched by this port, so no CPU regression is expected; confirm `./build/test` still prints "All tests passed" and `./build/main ... -d cpu` still runs.

## Disposition
PROCEED -- mechanical HIP port, Strategy A adapted to a Makefile. Advance linux-gfx90a to `planned`.

## Open questions
- ROCm 7.2.1 hipGraph stream-capture + per-node `hipGraphExecKernelNodeSetParams` replay parity: validated only at porter/validator time on the full-model path; the gate does not depend on it. If it misbehaves, add a non-graph fallback dispatch (low effort -- kernels are already directly launchable).
- WPB choice (16 on wave64) must divide the model's dim and vocab_size; confirm against the exact configs of the model used for the optional end-to-end check (Mistral dim=4096/vocab=32000 -> 16 divides both; Llama-3.2 -- verify vocab divisibility, else clamp the last partial block).
- `hipStreamLegacy` availability in the installed ROCm; fall back to default stream `0` in the compat header if missing.
