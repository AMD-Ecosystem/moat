# lmdeploy port plan

Planner run 2026-11-20 on linux-gfx942, read-only, against upstream `InternLM/lmdeploy` `main` at 63549b282790f77ebf4bf2a0d6e44a70de14d85b (2026-11-20). No GPU was used and nothing was built.

Four questions for Jeff are at the top of the 2026-11-20 planner section of `notes.md`. Q1 (overlap with open upstream ROCm PRs) and Q2 (one project or two) must be answered before a porter starts. Q3 (DeepEP) and Q4 (CUDA-only accelerators in the PyTorch engine) are cuts inside the ruled scope, so per the 2026-11-11 ruling they are also Jeff's; the plan below marks the work each answer would add or remove.

## Project

- Upstream: https://github.com/InternLM/lmdeploy (Apache-2.0), default branch `main`.
- Fork: https://github.com/AMD-Ecosystem/lmdeploy, port branch `moat-port`.
- Scope (Jeff Daily, 2026-11-11): the whole project. The PyTorch engine on ROCm AND a HIP port of TurboMind (`src/turbomind/`), built for `LMDEPLOY_TARGET_DEVICE=rocm` beside the unchanged CUDA build. Staged; per-stage evidence; any further cut inside TurboMind is Jeff's call.

## Existing AMD support

- **Upstream, PyTorch engine: partial, improvable.** PR #3925 (merged 2025-09, AMD contributor Vivicai1005) added the ROCm install doc, `requirements/runtime_rocm.txt`, and one ROCm tiling branch (`_kernel_meta_rocm`, `lmdeploy/pytorch/kernels/cuda/flashattention.py:465,555`). No ROCm CI; the maintainer has no AMD hardware (issue #4022).
- **Upstream, open, since intake: two ROCm PRs by menglcai.** https://github.com/InternLM/lmdeploy/pull/5135 (`is_bf16_supported` on ROCm, opened 2026-11-15) and https://github.com/InternLM/lmdeploy/pull/5138 (W8A8 int8 on ROCm: portable `libdevice.round` plus 2-stage autotune configs that fit 64 KB LDS, opened 2026-11-16). Both open, no maintainer review yet. The author is an active ROCm-ecosystem contributor (open PRs on ROCm/aiter, ROCm/AMDMIGraphX, ROCm/torch_migraphx, vllm-omni ROCm fixes); profile states no affiliation. Judgment: a competent, current, but piecemeal third-party effort, not an authoritative port. It is the same work as part of stage P, so it changes how we deliver that stage (Q1). Their PR #5135 body also carries a useful baseline: the full `tests/pytorch/kernel` suite on upstream `main` gives 170 failed / 654 passed with 16 files crashing on gfx90a (torch 2.11, Triton 3.6), and about 345 failed / 700 passed with 3 files crashing on gfx1201/gfx1150/gfx1151 (Triton 3.8). The PyTorch engine is therefore far from "works on ROCm" today.
- **TurboMind: none.** No upstream branch, PR, or fork with a HIP TurboMind (checked `gh pr list --search "rocm OR hip OR amd"`, web search, fork list). `setup.py:139` builds TurboMind only for `LMDEPLOY_TARGET_DEVICE=cuda`, so on ROCm `lmdeploy.turbomind.is_available()` is False and `lmdeploy/archs.py` `autoget_backend` silently falls back to the PyTorch engine.
- **Delivery vehicle:** upstream merges vendor support directly (Ascend/MACA/Cambricon via dlinfer, ROCm via #3925), so upstream PRs are the right vehicle; they do not link platform forks.
- **Port vs rewrite (performance kernels):** TurboMind's GEMM and attention hot paths are hand-written NVIDIA tensor-core code (mma.sync, ldmatrix, cp.async; sm90 wgmma/TMA via CuTe). There is nothing for hipify to translate into, so these get **AMD-native implementations inside TurboMind's own frameworks** (an MFMA atom for gfx9 wave64, a WMMA atom for gfx11/gfx12 wave32), reusing the existing arch-neutral dispatch, mainloop, epilogue and weight-format layers. Dense fp16/bf16 GEMM goes through hipBLASLt via TurboMind's existing cuBLAS family first, so a model runs end to end before the native GEMM exists. The NVIDIA sm90/sm120 stacks (wgmma, TMA, clusters, setmaxnreg, PDL) are not ported; AMD dispatch never selects them.

## Build classification

Two build systems in one package, both decided by `setup.py`:

1. **PyTorch engine: pure Python + Triton, no compiled code** (`lmdeploy/pytorch/`, 41 Triton modules in `lmdeploy/pytorch/kernels/cuda/`). Installed with `LMDEPLOY_TARGET_DEVICE=rocm pip install .` (`setup.py:13,185` choose `requirements/runtime_rocm.txt`).
2. **TurboMind: pure CMake** (`CMakeLists.txt:18` `project(TurboMind LANGUAGES CXX CUDA)`, `find_package(CUDAToolkit REQUIRED)` at :41), driven from `setup.py:139-163` through `cmake_build_extension`, producing the pybind11 module `lmdeploy.turbomind._turbomind`. It does not use `find_package(Torch)` or `torch.utils.cpp_extension`; it exchanges tensors with torch through DLPack (`src/turbomind/python/bind.cpp`).

Recorded `ext_type`: `cmake` (the compiled part; the Python part has no build).

## Port strategy

- **PyTorch engine:** no build strategy; Python and Triton source fixes, keyed on `torch.version.hip` (the idiom upstream already uses in `flashattention.py:555`) or on the Triton backend (`kernels/cuda/utils.py:49` `is_cuda()`), never on `get_device_capability()` numbers, which collide between vendors.
- **TurboMind: Strategy A.** A `cuda_to_hip` compat header force-included on the HIP build only, `enable_language(HIP)` when `LMDEPLOY_TARGET_DEVICE=rocm` (a CMake option, say `TM_BUILD_HIP`/`USE_ROCM`, set by `setup.py`), `set_source_files_properties(<.cu> LANGUAGE HIP)`, sources keep CUDA spelling. Shim headers for `cuda_runtime.h`, `cublas_v2.h`, `curand_kernel.h`, `cub/*`, `nccl.h`, `nvtx3/*` on the HIP include path only, so shared sources stay byte-identical where possible. CUDA build untouched (no-op header on NVIDIA; every new branch guarded).
- **Logical warp width:** TurboMind's SIMT code is pervasively 32-lane (`kernels/core/common.h:68` `constexpr int WARP_SIZE = 32`, ~95 uses; `tid % 32` / `tid / 32`; xor-butterfly reductions with masks 1..16). Keep `WARP_SIZE = 32` as a **logical** warp on HIP: the compat layer pins every `*_sync` shuffle to width 32 and to a 64-bit mask, and the two absolute-lane shuffles (`kernels/core/array_ops.h:436` `blockSum` broadcast, `:458-459` transpose shuffle) get an explicit `width = 32`. That is correct on both wave32 and wave64 (cuda-to-rocm "Width-32 LOGICAL warp ops are arch-agnostic"). `__ballot_sync` with a 32-bit result (`kernels/gemm/moe_utils_v2.cu:428-1729`) must use the 64-bit ballot and select the lane group's 32 bits, not truncate. The native MFMA/WMMA kernels are written against the real wavefront (64 on gfx9, 32 on gfx11/12) and do not use `WARP_SIZE`.

## CUDA surface inventory

`utils/surface.py generate` census (floor): 253 CUDA files; markers kernel 191, launch 204, device_code 1128, warp_intrinsic 99, driver_api 1; libraries cutlass 329, nccl 268, thrust 111, curand 89, cublas 65, cub 28, cusparse 1.

### PyTorch engine (Python/Triton; the census cannot see it)

| Item | Where | ROCm action |
|---|---|---|
| bf16 misdetected | `lmdeploy/utils.py:430-468` `is_bf16_supported`: requires `torch.version.cuda`; callers `pytorch/config.py:117`, `check_env/model.py:60`, `configurations/{qwen,cogvlm,qwen3_5}.py`, `vl/engine.py:58`, 3 kernel tests | `torch.cuda.is_bf16_supported()` when `torch.version.hip` (= PR #5135; Q1) |
| `tl.extra.cuda.libdevice` (`__nv_*`, fails on AMD Triton) | `flashattention.py:17` (tanh, softcap), `pagedattention.py:29-30` (tanh, fast_dividef), `w8a8_triton_kernels.py:9` (round; = PR #5138), `kpool.py:5`, `dsv4/hc_pre_fused.py:11` | `triton.language.extra.libdevice` (portable; already used in `activation.py`, `gated_delta_preprocess.py`) |
| PTX inline asm in Triton | `awq_kernels.py:35,47` (`lop3` + `fma.f16x2` int4 dequant): **Triton AWQ unusable on ROCm**; `pagedattention.py:694-768` `bar.sync 0` (TURBO_QUANT KV only); `symm_mem_allgather.py` multimem (off by default); `dsv4/hc_pre_fused.py:72` gluon asm | pure-Triton dequant branch on HIP for AWQ; `tl.debug_barrier()` for the `bar.sync`; guard multimem and gluon off on HIP |
| Capability gates that collide (gfx90a=(9,0), gfx942=(9,4), gfx950=(9,5), gfx1100=(11,0), gfx1201=(12,0)) | `blockedf8_modules.py:238` `_has_gluon` (`==9` -> CDNA imports Hopper gluon wgmma/TMA kernel: **high risk**); `awq_modules.py:16` TurboMind AWQ provider on `(9,0)` (= gfx90a once TurboMind builds on ROCm; that provider is an sm90 prototype); `check_env/cuda.py:20` fp8 allowed on `>=9` (wrong for gfx90a, which has no fp8); `fused_moe.py:85,747,910`, `moe/blocked_fp8.py:611`, `compressed_tensors_w4a16.py:313` Hopper tuning; `kpool.py:116,217`; `moe_router.py:19`; `kernels/cuda/utils.py:7-29` `WARPS_PER_SM`/`BLOCKS_PER_SM`; `pagedattention.py:894-901` and `flashattention.py:559` nv_cap meta tables; `comm/symm_mem_*`, `flashinfer_allreduce.py` (off by default) | add `torch.version.hip` short-circuits; ROCm meta/autotune tables sized for the LDS (64 KB gfx90a/gfx942/RDNA, 160 KB gfx950) |
| fp8 dtype | `torch.float8_e4m3fn` hard-coded everywhere (`engine/cache_engine/schema.py:17,52`, `config.py:1010`, `nn/linear/blocked_fp8.py`, `static_fp8.py`, `moe/*`, `kpool`, `w8a8_triton_kernels.py:597`, `fill_kv_cache.py:79`, `messages.py:25`, `cli/utils.py:139`) | gfx942 needs `float8_e4m3fnuz` (weights reinterpreted with scale x2, the established conversion); gfx950 and RDNA4 use OCP `e4m3fn`; gfx90a/gfx11 have no fp8 and must refuse fp8 formats cleanly |
| Optional CUDA-only accelerators | deep_gemm, flash_mla, deep_ep, flashinfer, FA3 (`flash_attn_interface`, guarded by `torch.version.cuda`; `require_fa3_for_speculative_decoding` at `backends/cuda/attention/__init__.py:28` makes speculative decoding impossible on ROCm) | stay off on ROCm (they import-fail already). What replaces them, if anything, is Q4 |
| Portable optional deps | fla (Triton), tilelang 0.1.11 (+ apache-tvm-ffi; has an AMD backend), Dao causal_conv1d fallback | validate on ROCm; Qwen3-Next/Qwen3.5 need fla + tilelang (`configurations/qwen3_next.py:14`, `gated_delta_rule.py:321`) |
| CUDA graphs | `backends/cuda/op_backend.py:388`, `graph_runner/` (full and piecewise) | HIP graphs through torch; validate with `eager_mode=False` |
| Packaging | `requirements/runtime_rocm.txt` = common + accelerate + peft only | add aiohttp, prometheus_client, opencv-python-headless (vendor-neutral), flash-linear-attention, tilelang/apache-tvm-ffi (linux x86_64); no torch/torchvision/triton pins (ROCm wheels come from the ROCm index, `pytorch-triton-rocm`) |
| Docs | `docs/en/get_started/installation.md:47-65` (+ zh_cn) | validated scope, GPU list, TurboMind-on-ROCm build line |
| Device type | `messages.py:679` and `cli/utils.py:472` accept cuda/ascend/maca/camb; ROCm runs as `'cuda'` | keep (torch's HIP alias); no new `'rocm'` device type |

### TurboMind: build and host (`CMakeLists.txt`, `setup.py`, `src/turbomind/{core,engine,generation,memory,models,python,utils,comm}`)

| Item | Where | ROCm action |
|---|---|---|
| CUDA-only build | `CMakeLists.txt:18-41,250-285` (CUDA language, SM arch list, nvcc-only flags `--expt-*`, `--diag-suppress`, `-Xcompiler`, `-Xptxas=-v` in core/copy/models CMake, `--use_fast_math`); `setup.py:35,139` (nvcc version probe, extension only for `cuda`, nvidia-* wheels) | HIP branch: `enable_language(HIP)`, `CMAKE_HIP_ARCHITECTURES` default (gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1151;gfx1200;gfx1201), HIP equivalents of the flags (not `-ffast-math` for `--use_fast_math`), `setup.py` builds the extension for `rocm` too and adds no nvidia-* wheels |
| FetchContent cutlass | `CMakeLists.txt:84-96`; CuTe really used outside sm90 only in `kernels/copy/{copy,transpose}.cu` (layout/tensor only) | skip the fetch on HIP; rewrite the copy/transpose index math in plain C++ for HIP (small); `cutlass::FastDivmod` in `attention_params.h`, `reduce.cu` -> local divmod |
| Arch macros silently empty | `kernels/core/common.h:5-38` `TURBOMIND_ARCH_SM70/75/80/90` keyed on `__CUDA_ARCH__`: 0 under HIP, so every `#if TURBOMIND_ARCH_SM80` / `TURBOMIND_ARCH_DTYPE_GUARD` body compiles to an empty kernel with no error (`activation.cu:43,147`, `quantization.cu:35,124,176,...`); also `__CUDA_ARCH__ >= 800` in `ban_bad_words.cu:45`, `apply_token_bitmask_inplace_cuda.cu:36,57` | HIP device-pass definitions: bf16 on all targets, fp8 only on gfx942/gfx950/gfx12 (with the fnuz/OCP split), never "SM80/SM90" for MMA paths; a guard that fails loudly when a dtype path is compiled out |
| Host SM-version dispatch | `utils/cuda_utils.cc:241` `getSMVersion()` (callers `unified_attention_layer.cc:144,604`, `GatedDeltaNetLayer.cc:69` -> 940 selects sm90 linear-attn); `kernels/gemm/{gemm.cu:107,registry.cu:14,context.cu:81}`; `kernels/attention/registry.cu:109`; `core/copy.cc:90` | a HIP arch value derived from `gcnArchName` (gfx9 MFMA / gfx11 WMMA / gfx12 WMMA families) that can never equal an NVIDIA SM tag |
| Runtime API | `core/allocator.cc:22-70` mempools (`cudaMemPoolCreate`, `cudaMallocFromPoolAsync`, `ReleaseThreshold`, `TrimTo`), `cudaHostAlloc`, priority streams, `cudaMemcpy2D/3DAsync`, `cudaPointerGetAttributes`, `cudaFuncSetAttribute`, occupancy | all have HIP equivalents; mempool release-threshold behaviour differs (check against cuda-to-rocm's mempool note) |
| Driver API | `core/copy.cc:64-110` `cuMemcpyBatchAsync` via `cudaGetDriverEntryPoint` (sm100+); `comm/cuda_ipc/cuda_ipc_comm.cu:94,165-305` VMM (`cuMemCreate/Map/SetAccess/...`) and NVLS multicast (`cuMulticast*`); `kernels/gemm/tma.cu`, `kernels/linear_attn/kernel/tma_desc.cu` (`cuTensorMapEncode*`; tma_desc.cu is in the common linear_attn library) | serialized copy fallback; VMM -> `hipMem*`; multicast capability forced 0 on HIP; TMA files excluded from the HIP build |
| DLPack | `python/bind.cpp:83-93,183-196` accepts only `kDLCPU`/`kDLCUDAHost`/`kDLCUDA`; ROCm torch exports `kDLROCM` (10) | accept and export `kDLROCM`/`kDLROCMHost` on HIP |
| cuBLAS/cuBLASLt | `kernels/gemm/cublas.cu` (`cublasGemmEx`, production dense family, priority 260), `gpu_metric.cu:97` (peak measurement), `gemm/test/{reference,test_utils}.cu`; `utils/cuda_utils.h:28-47`, `core/cuda_data_type.h` (`CUDA_R_*`); cuBLASLt linked by `turbomind` but no calls | hipBLASLt (preferred) or hipBLAS `GemmEx`; `hipDataType`; drop the Lt link |
| cuRAND | `sampling_topk_kernels.cu` (`curand_init`), `sampling_kernels.cu` (`curand_uniform`), `llama_utils.cu` (`sizeof(curandState_t)`), gemm test RNG | hipRAND/rocRAND device API (state size by `sizeof`) |
| CUB / Thrust | `sampling_topp_kernels.cu` (`DeviceSegmentedRadixSort::SortPairsDescending`), BlockReduce/BlockScan in norm, llama, anomaly handler, `moe_utils_v2.cu` (`WarpScan`); thrust vectors in `llama_utils.cu`, `anomaly_handler.cu`, attention tests | hipCUB / rocThrust; check the rocPRIM segmented radix sort defects in cuda-to-rocm libraries.md against the top-p sort (float keys, NaN mixes) |
| NVTX | `utils/nvtx_utils.cc` | roctx, or `USE_NVTX=OFF` on HIP |
| NCCL | `comm/nccl/nccl.cu` (AllReduce/AllGather/ReduceScatter/Broadcast/Reduce/Group*/CommInitRank/CommSplit; optional `dlopen("libnccl.so.2")` for MemAlloc and window registration); `CMakeLists.txt:148-170` finds NCCL through the `nvidia.nccl` wheel | RCCL; dlopen `librccl.so.1`; window registration absent in RCCL, the code already tolerates null |
| DeepEP | `comm/nccl/deepep/` always added when NCCL is found (`comm/nccl/CMakeLists.txt:12`), hard-fails without `nccl_device.h` / NCCL >= 2.30.4 (`deepep/CMakeLists.txt:27-49`); DeepEP `dd758ca` + patch, NCCL device API (`ncclDevComm`, `ncclTeamLsa`, `ncclGetLsaDevicePointer`), and **runtime nvcc JIT** (`token_dispatcher.cc:148-155`, `FindCudaHome`); used only for `moe_a2a_backend='deepep'`, auto-selected when `nnodes > 1` (`lmdeploy/turbomind/turbomind.py:92-95`) | no RCCL device API exists. Q3 |
| Custom all-reduce (`cuda_ipc`) | `comm/cuda_ipc/*.cu`: one-process peer access via VMM (no `cudaIpc*`), system-scope `atom/ld.acquire/st.release` PTX (`semaphore.cuh:57-75`), LL16 packets with `st/ld.volatile.v4` flag-in-payload (`mscclpp.h:41,68`), named barriers `bar.sync id, n` (`fused_allreduce.cu:68,227,314`, `fused_allreduce_ex.cu:97,252`), `multimem.*` only under `TURBOMIND_ARCH_SM90` | HIP: `__hip_atomic_*` system scope, fine-grained / uncached peer memory (cuda-to-rocm: cross-GPU flags and data in ordinary device memory are never seen by the peer on MI250X), named barriers replaced (whole-block `__syncthreads` or an LDS counter barrier), NVLS paths compiled out; `communicator='nccl'` is the default and works without this, so this is a later stage |

### TurboMind: kernels outside GEMM/attention

Sampling (top-k, top-p, penalties, ban-bad-words, stop criteria, token bitmask), activation, quantization (fp8 e4m3, nvfp4; `cuda_fp8.h`), cross entropy, unfused attention softmax, decoding (embedding lookup), gpt kernels, norm (`rms_norm.cu`, `layer_norm.cu`; `cub::BlockReduce`, one `__shfl_xor_sync((uint32_t)-1)` at `rms_norm.cu:193`), copy (CuTe layouts), `models/llama/{llama_kernels,llama_utils,mla_utils,gated_delta_net_kernels}.cu` (`0xffffffff`, `&31`, `>>5` at `gated_delta_net_kernels.cu:118-181`), vision (`qwenvit_kernels.cu`, `internvit_kernels.cu`), `utils/{memory_utils,anomaly_handler}.cu`, `reduce_kernel_utils.cuh` (`FINAL_MASK 0xffffffff`, width 32, cooperative groups), `activation_ops.h` (`tanh.approx` asm). All SIMT; mechanical under the compat header plus the logical-warp rule, the arch-macro fix, and PTX replacements in `kernels/core`:

| PTX in shared headers | Where | HIP replacement |
|---|---|---|
| `mma.sync` m8n8k4/m16n8k8/m16n8k16 | `kernels/core/mma.h` | not translated: MFMA/WMMA atoms (stages T1, T3) |
| `ldmatrix` x1/x2/x4 (.trans), `movmatrix` | `kernels/core/smem.h`, `array_ops.h:478` | LDS loads matching the MFMA/WMMA operand layout; shuffle fallback already at `array_ops.h:454` |
| `cp.async` (+ `__pipeline_*`), `createpolicy` | `core/smem.h`, `array_ops.h`, `gemm/cp_async.h`, `attention/iterator_sm80.h`, `gemm/iterator_sm80.h:147` | register-staged loads (the sm70 mainloop shape) |
| `ld/st.shared.v{1,2,4}` | `core/array_ops.h`, `core/smem.h` | plain LDS loads/stores |
| `bar.sync`/`barrier.*` named, `ld.global.acquire.gpu`/`st.global.release.gpu`/`.cg` | `core/sync.h` (gemm sm90 only), `core/common.h` | `__syncthreads` / `__hip_atomic_*` agent scope |
| `lop3.b32` (16), `cvt.rni/rzi.sat.*` (12), `sub/mul/fma.f16x2`, `bf16x2` | `attention/quantization.h` (KV int4/int8 dequant, also used by `gemm/transform.h`) | plain bit ops (`(a&b)|c` compiles to `v_bfi`/`v_and_or`), `__float2uint_rn` + clamp, `__hmul2`/`__hfma2` |
| `__byte_perm` (19) | `attention/quantization.h`, `core/array_ops.h` | HIP provides `__byte_perm`; check selector semantics against `__builtin_amdgcn_perm` if hand-replaced |
| `mul.f32` (non-FTZ) | `core/floating_point.h:113` | plain C++; keep denormals |
| `tanh.approx` | `kernels/activation_ops.h` | `tanhf` / `__ocml_tanh_f32` |
| `__ldcs/__ldcg/__stcs/__stcg` | `core/array_ops.h` | HIP has them (or nontemporal builtins) |

### TurboMind: attention (`kernels/attention/`, 36 .cu)

- One templated kernel, `AttentionUniversal<Arch, Mainloop<..., Impl<MMA_tag,...>>, CacheIterFactory, CtaMap>`, registered per arch (`registrar.h`, `registry.cu`). Prefill over linear K/V (paged cache flattened and dequantized by `flattenKV_v2`, `kv_cache_utils_v2.cu:341`); decoding reads the paged cache directly (`block.h`, `block_iterator.h`); KV formats half/bf16, int8, int4 (inline per-token scale/zero), fp8 conversion (`quantization.h:788-863`); RoPE variants in `rotary_embedding.h` (default, dynamic, yarn, llama3, mrope); split-K reduce (`reduce.cu`) with context parallelism (`cp_utils.cu`).
- Per arch: sm70 prefill `MMA_884` / decode `MMA_SIMT`; sm75 `MMA_1688` / `MMA_81616`; sm80 `MMA_16816` (cp.async, 2 stages) / `MMA_SIMT` (Qh<=2) or `MMA_81616` (3 or 5 stages). Head dims 64/128/256/576 (+192 on sm80). No sm90-specific attention.
- **Portable base:** `MMA_SIMT` + `Mainloop<arch::Sm70>` (`impl_simt.h`, `mainloop_sm70.h`, `iterator_sm70.h`) uses only plain loads, `__shfl_xor_sync` and `__syncthreads`, and serves half/int8/int4 KV decoding. **There is no SIMT prefill**; every prefill impl is mma.sync. So the HIP prefill needs a new `Impl<MMA_MFMA>` (gfx9) and `Impl<MMA_WMMA>` (gfx11/12) in the attention framework, which also replaces `MMA_81616` for quantized-KV decoding performance later.
- LDS: `kernel_impl.h:72-75` opts in above 48 KB and swallows failure; HIP has no opt-in (`hipFuncSetAttribute(MaxDynamicSharedMemorySize)` does nothing and an over-limit launch fails). Prefill hd576 (~72 KB) and hd256 (~64 KB), sm80 decode hd576 (72-180 KB) exceed 64 KB; SIMT decode hd576 (~36 KB) fits. The registry must filter on `sharedMemPerBlock` and the AMD tile tables must be sized for 64 KB (160 KB on gfx950).
- Tests: `test_attention.cu` against an unfused reference (`reference.cu`: cuBLAS `GemmStridedBatchedEx`, mask, softmax, own RoPE; also compares the KV cache), `test_quant.cu` round trip; thrust `universal_vector`, cooperative groups.

### TurboMind: linear attention (`kernels/linear_attn/`, 15 .cu)

Gated delta rule (Qwen3-Next/Qwen3.5), recurrent and chunked. `SelectArchitecture` (`delta_rule.cu:43`): pre_sm90 below 900, sm_90 at 900, sm_120 at 1200. **pre_sm90** (~650 lines, SIMT, `__shfl_xor_sync(0xffffffff)` xor < 8, chunk16 smem exactly 64 KB with an f32 state, opt-in at `chunked.cu:291-295`) is the HIP base: route every AMD arch to it. sm_90 (~8.3K lines: CuTe GMMA/TMA/mbarrier/setmaxnreg/PDL) and sm_120 (~6.8K lines: CuTe SM80 atoms + SM90 TMA) are excluded from the HIP build; `tma_desc.cu` (common library) is guarded out, and `delta_rule.h`'s `state_tma_descs` stays unused on HIP. The Python test `tests/turbomind/linear_attn/test_gated_delta_rule.py:34-166` dispatches on `get_device_capability()` and would treat gfx942 as sm90: it needs a HIP mapping too.

### TurboMind: GEMM (`kernels/gemm/`, 39 .cu, ~1.26 MB)

- **Arch-neutral host layer (ports mechanically):** registrar/collector with `WHOLE_ARCHIVE` (`CMakeLists.txt:160-165`), `registry.cu` (device filter: `arch = major*100+minor*10`, smem vs `sharedMemPerBlockOptin`), `family.h/.cc` (formats, alignment, grouped, `pack()`), `gemm.cu` dispatch/tune/export (`TM_GEMM_TUNE`), `context.cu`, `dispatch_cache.cu`, `kernel.cu` cost model, `tuner/*` (event timing), `gpu_metric.cu`, `weight_plan.cc`, `cast.cu`, `convert_v3.cu`, `unpack.cu`, `moe_utils_v2.cu` (gating, CUB), epilogue (split-K, gated SiLU), `cta_map.h`, `thread_map.h`. Note `test/test_utils.cu` is compiled into production `gemm2_core` (`CMakeLists.txt:154-155`), so production links cuBLAS and cuRAND.
- **Arch-specific:** `arch/` configs (sm70 `s884`, sm75/sm80 `s16816`, SIMT), `core/mma.h`, `core/smem.h`, `cp_async.h`, `mainloop_sm80_v2.h`/`iterator_sm80.h`, `convert.cuh` fragment packing tables; the whole sm90 stack (`gemm_universal_sm90*.h`, `gmma_*`, `scaled_gmma_fp8_sm90.h`, `scheduler.cuh` clusters, `tma.cu`, `sm90_*`; CuTe atoms, wgmma, TMA, mbarrier, setmaxnreg, device-side `tensormap.replace`).
- **Formats by arch today:** fp16 dense (sm70/75/80 + cuBLAS), bf16 dense (sm80, sm90 grouped, cuBLAS), u4 AWQ/GPTQ g32/g128 dense+grouped (sm70-sm90), mxfp4 weight-only (sm70-sm90), fp8 e4m3 weight-only block 128 (sm70-sm90), **sm90-only:** fp8 w8a8 block-scaled dense+grouped (`sm90_64n32_8`), nvfp4 (`sm90_64n16_nvfp4`), mxfp4 x fp8 (`sm90_64n32_mxfp4_fp8_folded`). sm100/sm120 reuse the sm80 kernels (`arch.h:52`).
- **Correctness-first base:** a complete SIMT config exists but is dead (`arch/config_simt.h` `Sm75_Simt`, `mma_simt.h` FMA loop, `operand_simt.h`, `smem_copy_simt.h`, on `MainloopSm70`); nothing instantiates it. It hard-codes 32 threads (fine as a logical warp). Instantiating it on HIP gives every weight-only format (int4, fp8, mxfp4 through `TransformA/B`) a correct, slow path before the MFMA/WMMA atoms exist.
- **AMD-native target:** new `arch/` atoms for MFMA (gfx90a/gfx942/gfx950: `v_mfma_f32_16x16x16{f16,bf16}`, `32x32x8`, fp8 on gfx942/gfx950, scaled f8f6f4 on gfx950) and WMMA (gfx11/gfx12 16x16x16), with matching operand, smem-copy and `convert.cuh` packing, on the register-staged `MainloopSm70` and the generic epilogue. hipBLASLt covers dense fp16/bf16 through the existing cuBLAS family. No Composable Kernel dependency: TurboMind packs weights in its own layouts (`pack_u4`, block scales), so a CK path would add a re-pack and a heavy dependency for formats the native family must cover anyway.
- **Tests:** no CMake target builds a GEMM correctness test (`gemm/test/` has the cuBLAS reference and compare helpers but only `test_quantization` links them). The real GEMM gate is the Python suite `tests/turbomind/linear/test_linear.py` (uses `_tm`, skips if TurboMind is not built) plus `tests/turbomind/moe_gate/test_moe_gate_v2.py`; `gemm/test/test_moe_utils.cu` has a `main()` but no target.

## Risk list

1. **Silent empty kernels** from `TURBOMIND_ARCH_*` and `__CUDA_ARCH__` gates under HIP (above). The first build will "work" and produce garbage or zeros. Fix before any test is believed; add a loud failure for compiled-out dtype paths.
2. **Capability-number collisions** in both engines: gfx90a=(9,0) equals Hopper (TurboMind AWQ provider, gluon, symm-mem tables), gfx942/gfx950 (9,4)/(9,5) pass every `>= 9` Hopper gate, gfx1100/gfx1201 alias Blackwell-consumer table entries; TurboMind `getSMVersion()*10` gives 940 -> sm90 linear-attn path.
3. **Logical warp vs wavefront:** 32-bit `*_sync` masks fail to compile under current HIP (64-bit mask required); 32-bit `__ballot_sync` truncation in `moe_utils_v2.cu`; absolute-lane shuffles in `array_ops.h:436,458`. Validate on both wave64 and wave32.
4. **LDS 64 KB** (gfx90a/gfx942 and RDNA; 160 KB on gfx950): TurboMind attention/GEMM tiles and PyTorch-engine Triton autotune/meta tables built for 100-228 KB NVIDIA smem; HIP has no opt-in, so filter on `sharedMemPerBlock`.
5. **fp8 FNUZ vs OCP:** gfx942 is e4m3fnuz, gfx950/gfx12 OCP e4m3fn, gfx90a/gfx11 none; both engines hard-code e4m3fn (weights, KV cache, scales). Wrong format is silent numerical garbage, not an error.
6. **Cross-GPU memory model** in `comm/cuda_ipc`: system-scope flags and LL packets need fine-grained/uncached peer memory; named barriers have no HIP form.
7. **Fresh device allocations are not zeroed on ROCm:** audit accumulators/atomics in sampling, MoE gating (`moe_utils_v2.cu` scans/counters), split-K workspaces, allreduce flags.
8. **rocPRIM segmented radix sort defects** (cuda-to-rocm libraries.md) for the top-p sort on float keys.
9. **`-ffp-contract=fast` default and fast-math mapping** (`BUILD_FAST_MATH` -> do not translate to `-ffast-math`, which implies finite-math-only and breaks softmax masking with -inf).
10. **WMMA long-k drift on gfx11/gfx12** (cuda-to-rocm faults-numerics) for GEMM tolerances.
11. **Triton on RDNA:** bf16 multiply truncation in Triton <= 3.8 on gfx11/gfx12 (triton-lang/triton#11217, fixed in 3.9; cited in PR #5135) affects bf16 rotary tests; validate with Triton >= 3.9 and record the version.
12. **Graph capture + library handles:** HIP graph capture with hipBLASLt handle creation in another thread exits the process (cuda-to-rocm faults-memory-streams); PyTorch-engine CUDA graphs and TurboMind's hipBLASLt family must create handles before capture.
13. **Size of the effort:** TurboMind is ~2.8 MB of kernel source; stages T1 and T3 are multi-week native kernel work. Budget per stage, record partial value, stop at stage boundaries.

## Stages and file-by-file change list

Stage order follows the 2026-11-11 ruling. Every stage leaves the CUDA build byte-identical in effect (guards only) and ends with a recorded GPU result on this host's arch plus whatever other arch is available; stage evidence goes in `notes.md` under a dated heading per stage. If Jeff splits the project (Q2), stage P is one project and T0-T5 the other.

### Stage P: PyTorch engine on ROCm (Linux; no compiled code)

- `lmdeploy/utils.py`: HIP branch of `is_bf16_supported` (or carry PR #5135; Q1).
- `lmdeploy/pytorch/kernels/cuda/{flashattention,pagedattention,w8a8_triton_kernels,kpool}.py`, `dsv4/hc_pre_fused.py`: `triton.language.extra.libdevice` (w8a8 = PR #5138; Q1). ROCm meta for `pagedattention.py:894-901` like the existing `_kernel_meta_rocm`; 2-stage autotune configs that fit 64 KB.
- `lmdeploy/pytorch/kernels/cuda/awq_kernels.py`: pure-Triton int4 dequant when not on the CUDA backend.
- `lmdeploy/pytorch/kernels/cuda/pagedattention.py:694-768`: portable barrier for TURBO_QUANT.
- `lmdeploy/pytorch/kernels/cuda/utils.py:7-29`: HIP entries (or CU-count-based) for `WARPS_PER_SM`/`BLOCKS_PER_SM`.
- `lmdeploy/pytorch/backends/cuda/{blockedf8_modules,awq_modules,moe_router,kpool}.py`, `kernels/cuda/moe/{fused_moe,blocked_fp8}.py`, `compressed_tensors_w4a16.py`, `check_env/cuda.py`, `comm/{symm_mem_allreduce,symm_mem_allgather,flashinfer_allreduce}.py`: `torch.version.hip` guards so no Hopper-only path is selected on AMD; fp8 support check by arch (gfx942/gfx950/gfx12 only).
- fp8: a small helper choosing `float8_e4m3fnuz` on gfx942 (weights converted on load with the scale doubled; KV cache dtype) and `e4m3fn` elsewhere, used by `engine/cache_engine/schema.py`, `config.py`, `nn/linear/{blocked_fp8,static_fp8}.py`, `moe/*`, `fill_kv_cache.py`, `w8a8_triton_kernels.py`.
- Tests that skip on `get_device_capability()[0] < 9` (23 files under `tests/pytorch/kernel`) must skip on HIP where the feature is Hopper-only (gluon, deep_gemm-based), not run and fail.
- `requirements/runtime_rocm.txt`: parity for the vendor-neutral and AMD-capable entries; `docs/en/get_started/installation.md` + `docs/zh_cn/...`: validated GPUs and commands.
- Stage P is done when the full kernel suite has no ROCm-caused failure left unexplained (each remaining failure classified: Hopper-only feature skipped, upstream bug also on CUDA, or deferred with a record), and the engine smokes below pass on wave64 and wave32.

### Stage T0: TurboMind builds for ROCm; non-GEMM kernels; dense model runs

- `setup.py`: build the extension for `rocm`; skip `get_cuda_version`/nvidia-* wheels; pass the HIP option; RCCL from the ROCm install (no wheel).
- `CMakeLists.txt`, `src/**/CMakeLists.txt`, `src/turbomind/python/CMakeLists.txt` (own `project(... CXX CUDA)`): HIP language branch, `LANGUAGE HIP` on `.cu`, arch list, flags, no cutlass fetch, NCCL->RCCL find, `USE_NVTX` -> roctx or off, DeepEP off (Q3), sm90/sm120/TMA sources excluded, `CUDA::cublas`/`cublasLt`/`curand`/`cudart`/`cuda_driver`/`nvtx3` mapped to `roc::hipblaslt`/`hip::hiprand`/`hip::host`; RUNPATH for the ROCm libs on the pybind module (cuda-to-rocm build-cmake-link: a HIP-linked shared library gets no RUNPATH to the ROCm lib dir).
- New `src/turbomind/utils/cuda_to_hip.h` (compat header, HIP-only, force-included) and shim headers on the HIP include path.
- `kernels/core/common.h`: HIP definitions of the arch macros; `WARP_SIZE` stays 32 (logical); shuffle compat pinned to width 32.
- `kernels/core/{array_ops.h,floating_point.h,smem.h,sync.h}`, `kernels/activation_ops.h`, `kernels/attention/quantization.h`: PTX replaced under `__HIP_PLATFORM_AMD__` (table above).
- `utils/cuda_utils.{h,cc}`: HIP arch value from `gcnArchName`; `getSMVersion` callers use it.
- `core/copy.cc`: no driver entry point on HIP; serialized fallback. `core/allocator.cc`: verify mempool semantics.
- `python/bind.cpp`: `kDLROCM`/`kDLROCMHost`.
- `kernels/copy/{copy,transpose}.cu`: CuTe-free index math under HIP.
- `kernels/gemm/cublas.cu`, `gpu_metric.cu`, `gemm/test/*`: hipBLASLt (dense family, the only GEMM in T0), hipRAND.
- `kernels/{sampling_*,ban_bad_words,stop_criteria,apply_token_bitmask_inplace_cuda,activation*,quantization,cross_entropy,unfused_attention,decoding,gpt}_kernels.cu`, `kernels/norm/*`, `models/llama/*.cu`, `models/{qwenvit,internvit}/*.cu`, `utils/{memory_utils,anomaly_handler}.cu`, `kernels/reduce_kernel_utils.cuh`: compile under HIP with the compat layer; fix masks/ballots and uninitialized-allocation reads found by the sweep.
- `kernels/linear_attn/delta_rule.cu`: AMD -> pre_sm90; `tma_desc.cu` guarded.
- `comm/nccl/nccl.cu`: RCCL; `dlopen("librccl.so.1")` on HIP. `comm/cuda_ipc`: compiled with NVLS/multicast off, or excluded until T4 with `communicator='nccl'` forced on HIP.
- `lmdeploy/turbomind/*.py`: no CUDA-only assumption on load; `lmdeploy/archs.py` keeps choosing turbomind when available.
- T0 gate: TurboMind builds and imports on ROCm; Catch2 host tests, `test_quantization`, `test_cross_entropy`, `test_mrope_position_ids` pass; attention is not yet available, so no end-to-end model yet unless T1 lands in the same round.

### Stage T1: attention

- `kernels/attention/`: register `MMA_SIMT` decoding for all KV formats on HIP; new `impl_mfma.h` (gfx9) and `impl_wmma.h` (gfx11/12) `Impl` specializations for prefill (all head dims that fit LDS; hd576 with tiles sized for 64 KB, or a SIMT/MFMA split), then quantized-KV decoding; `arch.h`/`registry.cu`: HIP arch tags and `sharedMemPerBlock` filter; `kernel/*.cu`: HIP instantiation files; `reduce.cu`, `cp_utils.cu`: compile + local divmod.
- `test_attention.cu`, `test_quant.cu`: hipBLAS reference, built and run.
- T1 gate: `test_attention` (fp16, bf16, int8 KV, int4 KV; hd 64/128/256 and 576 where supported) and `test_quant` pass on wave64 and wave32; end-to-end `pipeline(..., backend_config=TurbomindEngineConfig())` on a dense fp16/bf16 model with hipBLASLt GEMM produces correct text.

### Stage T2: GEMM correctness for every weight format

- `kernels/gemm/arch/config_simt.h`, `kernel/` new HIP instantiation of `Sm75_Simt` (or a renamed HIP SIMT family) for u4 AWQ/GPTQ, fp8 weight-only, mxfp4, dense and grouped (MoE); `registry.cu`/`arch.h`/`family.cc` HIP tags; `moe_utils_v2.cu` ballot/shuffle fixes.
- T2 gate: `tests/turbomind/linear/test_linear.py` and `tests/turbomind/moe_gate/test_moe_gate_v2.py` pass on HIP for every format the SIMT family serves; AWQ and an MoE model run end to end.

### Stage T3: AMD-native GEMM

- `kernels/gemm/arch/`: `mma_mfma.h`, `operand_mfma.h`, `smem_copy_mfma.h`, `config_mfma.h` (gfx90a/gfx942/gfx950) and the WMMA equivalents (gfx11/gfx12); `convert.cuh` packing for those fragment layouts; `kernel/` instantiations for f16/bf16 dense, u4 g32/g128, fp8 weight-only, mxfp4, grouped; fp8 w8a8 block-scaled on gfx942 (fnuz) and gfx950 (OCP) to restore the format TurboMind serves only on sm90; nvfp4 and mxfp4 x fp8 on gfx950 (scaled f8f6f4 MFMA) last, with a dequant path elsewhere.
- T3 gate: `test_linear.py` across formats, compared with the T2 SIMT results and the reference; throughput recorded against hipBLASLt for dense shapes (performance is evidence, correctness is the gate).

### Stage T4: multi-GPU

- `comm/cuda_ipc/*`: HIP VMM, system-scope atomics, fine-grained peer memory, named-barrier replacement, NVLS off; `comm/device_comm.cc` selection; `test_comm.cu` (drop `-march=native` on HIP if it breaks), `test_ipc_comm`, `test_host_comm`.
- T4 gate: `test_comm` all-reduce/all-gather/fused-RMSNorm against RCCL; tp=2 (and tp=4/8 where the host has the GPUs) end to end with `communicator='nccl'` and `'cuda-ipc'`; an MoE model with the default a2a backend.

### Stage T5: Windows

- TurboMind on Windows, tp=1, `BUILD_MULTI_GPU=OFF` (as upstream does for CUDA, `setup.py:155-156`), TheRock clang toolchain (cuda-to-rocm windows.md); `CMakeLists.txt` MSVC branches (`/Zc:*`, `-Xcompiler`) adjusted for clang. The PyTorch engine is Linux-only upstream (Triton is pinned for Linux in `runtime_cuda.txt`), so the Windows gate is carried by TurboMind.
- T5 gate: build + `test_attention`, `test_linear.py`, and an end-to-end dense-model pipeline on a Windows RDNA GPU.

## Build commands

Python engine (any ROCm host with a ROCm torch + Triton):

```shell
LMDEPLOY_TARGET_DEVICE=rocm pip install -e .            # no compiled code before T0
```

TurboMind (from T0; the exact option name is the porter's):

```shell
LMDEPLOY_TARGET_DEVICE=rocm MAX_JOBS=<physical cores> pip install -e . -v
# or directly:
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_ROCM=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx942 -DBUILD_TEST=ON -DBUILD_PY_FFI=ON \
      -DBUILD_MULTI_GPU=ON -DUSE_NVTX=OFF
cmake --build build -j <physical cores>
```

CUDA no-regression (Jeff's rule: every round, on the nvcc fleet host; TurboMind needs no torch to compile, only nvcc + the `nvidia-nccl-cu12>=2.30.4` wheel):

```shell
pip install -e . -v                                   # LMDEPLOY_TARGET_DEVICE defaults to cuda
```

## Test plan

GPU gate per platform (wave64 = gfx942/gfx90a/gfx950; wave32 = gfx11/gfx12; windows = RDNA on Windows), run with `utils/timeit.sh` and recorded with pass/fail counts:

1. **PyTorch engine kernels:** `pytest -q -rA tests/pytorch/kernel` one file at a time with a per-file timeout (upstream baseline in PR #5135 for comparison: gfx90a 170 failed / 654 passed / 16 crashing files at main 442097e8). The gate is the before/after diff of failed test ids with every remaining failure classified.
2. **PyTorch engine smokes:** `pipeline('Qwen/Qwen3-1.7B', backend_config=PytorchEngineConfig(dtype='auto', eager_mode=True))` then `eager_mode=False` (graphs); a Qwen3.5 small model (fla + tilelang linear attention); an AWQ model; a W8A8 int8 model; an fp8 model on gfx942 and gfx950; `lmdeploy check_env`; `pipeline.get_ppl()` on a short WikiText-2 slice compared with fp16/bf16 for the quantized models.
3. **TurboMind C++:** `BUILD_TEST=ON` targets: `test_core`, `test_scope`, `test_logger`, `test_data_format`, `test_memory`, `test_prefix_trie` (Catch2), `test_quantization`, `test_cross_entropy`, `test_mrope_position_ids`, `test_attention`, `test_quant`, `test_comm`, `test_host_comm`, `test_ipc_comm`; `bench_conv1d_silu` as a run check.
4. **TurboMind Python:** `pytest tests/turbomind/linear/test_linear.py tests/turbomind/moe_gate tests/turbomind/linear_attn/test_gated_delta_rule.py` (the last after its capability mapping is fixed).
5. **TurboMind end to end:** `pipeline(<model>, backend_config=TurbomindEngineConfig(...))` for a dense bf16 model, an AWQ int4 model, an MoE model, a Qwen3.5 model (linear attention), an InternVL/Qwen-VL model (vision kernels); quantized KV (`quant_policy=8`, `4`); tp=2 from T4.
6. **Non-GPU regression set:** `pytest tests/test_lmdeploy` (100 files; model-free parts), `tests/pytorch/{engine,paging,config,spec_decode,kv_connector,nn}` CPU/mocked parts; these must give the same results as upstream `main` on the same host.
7. **CUDA path:** nvcc build of TurboMind on the mi355 fleet host each round; the Python changes are CUDA-neutral by construction (every branch keyed on `torch.version.hip`), confirmed by the reviewer.

## Open questions

Q1-Q4 are stated with options in `notes.md` (2026-11-20 planner section). Decisions the plan took itself, which Jeff may overrule: native MFMA/WMMA kernels inside TurboMind's frameworks rather than a Composable Kernel or aiter dependency; hipBLASLt for dense GEMM; logical 32-lane warps for TurboMind's SIMT code; the sm90/sm120 instruction stacks recorded as not applicable to AMD (their functions are served by the AMD families, including the sm90-only fp8 w8a8/nvfp4/mxfp4-fp8 formats in T3); PyTorch engine stays on `device_type='cuda'`.
