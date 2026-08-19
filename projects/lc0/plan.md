# lc0 (Leela Chess Zero) -- ROCm/HIP port plan

## Project
- Name: lc0
- Upstream: https://github.com/LeelaChessZero/lc0
- Default branch: master
- Base sha (depth-1 clone in projects/lc0/src): d8ce48258c39d331c119f8c8729374ceb3df8409
- Lead platform: linux-gfx90a (MI250X, CDNA2, wave64), ROCm 7.2.1, hipcc + hipBLAS 3.2 present, 4x gfx90a on host. Meson 1.11.1 + ninja present.

## Existing AMD support (assessment) + decision
lc0 ships TWO AMD paths today, neither a native HIP-language port of the CUDA backend:
1. SYCL backend `src/neural/backends/sycl/` (`-Dsycl=amd`): DPCT-migrated copy of the CUDA backend (files are `*.dp.cpp`, header `DPCT_COMPAT_RT_VERSION 12020`). It runs on AMD via DPC++/oneAPI's SYCL-over-HIP target (`-fsycl-targets=amd_gpu_gfx90a`, links `amdhip64`+`hipblas`). This is a *different programming model and toolchain* (Intel DPC++ compiler), not the ROCm/HIP C++ runtime path.
2. ONNX-ROCm backend (`onnx-rocm`): just onnxruntime's ROCm execution provider; the network runs inside onnxruntime, not lc0's own kernels.

There is NO `src/neural/backends/hip/` and no `-Dhip` option: lc0's own CUDA kernels (the `network_cuda`/`network_cudnn` backend) have never been compiled with hipcc. Per PORTING_GUIDE "AMD supported only via OpenCL/Vulkan/SYCL with no HIP path -> a ROCm/HIP port of the CUDA code still adds value." 

Decision: PROCEED with a new native HIP backend (Strategy A: compile the existing `.cu` with hipcc behind a compat header, keep CUDA spelling). It adds a first-class ROCm path that does not require the Intel DPC++ toolchain and that runs lc0's own optimized kernels directly on ROCm. The SYCL `.dp.cpp` files are a useful *semantic reference* (they show the intended GPU behaviour) but are NOT the porting base -- we port the CUDA `.cu`.

Port-vs-rewrite for perf-critical kernels: the only NVIDIA-tuned hot path is the optional CUTLASS fused-MHA (`cutlass_kernels.cu`, `AttentionKernel<..., cutlass::arch::Sm80, ...>`). CUTLASS does not port to ROCm (PORTING_GUIDE 2026-05-30). It is OPTIONAL (`#ifdef USE_CUTLASS`, gated on `-Dcutlass` + CUDA arch >= Sm80 + C++17) and lc0 has a full non-fused fallback (per-layer cuBLAS GEMM + custom softmax kernels). Correctness-first decision: leave CUTLASS OFF on HIP; the attention network still runs through the fallback. A later AMD-native fused-MHA via ck_tile is a possible follow-up, NOT part of the lead port. There are NO other CUTLASS/CuTe/wgmma/wmma/mma intrinsics in the backend (grep clean) -- the rest is cuBLAS + plain custom kernels, a mechanical port.

## Build classification (cmake | torch-extension) + evidence
Classification: cmake (standalone native GPU build -> Strategy A). NOT a pytorch extension.
- Evidence: build is Meson (`meson.build`, `meson_options.txt`); no `find_package(Torch)`, no `torch.utils.cpp_extension`, no `CUDAExtension`, no torch dep in `pyproject.toml` (pyproject is only a thin meson-python shim for the optional `lc0` python module). `ext_type` recorded as `cmake` (the MOAT enum's stand-in for "standalone native, Strategy A"; the actual generator is Meson, not CMake -- noted here so the porter is not surprised).
- How the CUDA backend builds (meson.build 452-610): each `.cu` is compiled to a `.o` by invoking `nvcc` through a `custom_target` (`command : [nvcc, nvcc_extra_args, nvcc_arguments, nvcc_io_arguments]`), then the host `.cc` (layers.cc, network_cuda.cc, network_cudnn.cc) are compiled as ordinary C++ and link the `.o` + `cublas`/`cudart`(+`cudnn`). There is no `enable_language(CUDA)`; the `.cu`->`.o` step is an explicit external-compiler custom_target. The HIP port mirrors this: a parallel `custom_target` path invoking `hipcc` on the SAME `.cu` files (CUDA spelling preserved, compat header force-included), behind a new `-Dhip` option.
- Defaults relevant to scope: `plain_cuda=true` (network_cuda.cc, cuBLAS + custom kernels), `cudnn=false` (network_cudnn.cc is OPT-IN), `cutlass=true` but auto-disabled below Sm80/C++17. So the *default* CUDA backend needs only cuBLAS -> hipBLAS; cuDNN -> MIOpen is NOT required for the lead port. `blas=true` and `gtest=true` by default (CPU reference backend + CPU regression tests both available on the same machine).

## Port strategy (A compat-header) + rationale
Strategy A, adapted to Meson:
1. Add one compat header `src/neural/backends/cuda/hip_compat.h` (only file that knows HIP). On HIP: `#include <hip/hip_runtime.h>`, `#include <hipblas/hipblas.h>`, and `#define` the CUDA spellings the backend uses to hip* (cudaMalloc->hipMalloc, cudaStream_t->hipStream_t, cublasSgemm->hipblasSgemm, CUDA_R_16F->HIP_R_16F, CUBLAS_OP_*->HIPBLAS_OP_*, etc.). Use `torch/utils/hipify/cuda_to_hip_mappings.py` as the authoritative name source. Include `<cstring>`/`<cstdlib>` BEFORE `<hip/hip_runtime.h>` (gpuRIR lesson: host memcpy/memset resolving to HIP device overloads). `cuda_common.h` already includes `<cublas_v2.h>`/`<cuda_fp16.h>`/`<cuda_runtime.h>`; add `#if defined(USE_HIP) #include "hip_compat.h" #else <the three CUDA headers> #endif` at its top so the rename happens once and all backend TUs inherit it.
2. Meson: add `option('hip', boolean, false)`. Under `if get_option('hip')`, reuse the EXISTING `-Dsycl=amd` discovery blocks already in meson.build (hipBLAS via `cc.find_library('hipblas', dirs: hip_libdirs)`, `amdhip64`, `hip_include`/`hip_libdirs` options, `amd_gfx` + `rocm_agent_enumerator` autodetect at meson.build 756-803). Compile the `.cu` with hipcc via custom_target: `command : [hipcc, '--offload-arch='+amd_gfx, '-x','hip','-std=c++17','-fPIC','-DUSE_HIP','-D__HIP_PLATFORM_AMD__', '-include', '.../hip_compat.h', '-I','src', hip_includes, '-c','@INPUT@','-o','@OUTPUT@']`. Build `network_cuda.cc` (+ layers.cc) as the HIP backend's host C++; do NOT build `network_cudnn.cc` or `cutlass_kernels.cu` on HIP. Register the backend under a new name (`hip`/`hip-fp16`) or reuse `cuda`/`cuda-fp16` -- decide in porter; a distinct `hip` name is cleaner and avoids colliding with a real CUDA build on a dual-stack box. Bake arch from `${amd_gfx}` / `-Damd_gfx` (already the option), defaulting gfx90a only when unset, so gfx1100/gfx1151 need no source/meson edit (PORTING_GUIDE configurable-arch lesson).
3. Guard genuinely divergent code with `#if defined(USE_HIP)`; keep rare (the L2-persistence block, the FP16/tensor-core capability gating, the CUTLASS include, math-mode).

Rationale: keeps the NVIDIA build byte-for-byte (new code is behind `-Dhip`/`USE_HIP`), minimal diff, reuses lc0's own AMD-discovery meson code, and lets the porter validate the fp32 `cuda`/`hip` path first then `cuda-fp16` (gfx90a has full FP16/MFMA).

## CUDA surface inventory (and HIP mapping)
Device kernels live in (~4500 lines): `common_kernels.cu` (fp32 + portable fp16), `fp16_kernels.cu` (fp16-specific SE/fused), `winograd_helper.inc` (warp reduce/max, atomics, vector load/store), `cutlass_kernels.cu` (optional, NVIDIA-only). Host: `layers.cc`, `network_cuda.cc`, `network_cudnn.cc`(opt-in), `cuda_common.h`, `kernels.h`, `layers.h`, `inputs_outputs.h`.

- Kernels / `__global__`/`__device__`: many custom kernels (winograd input/output transforms, SE, global avg-pool, global scale, policy map, softmax, layernorm, add-bias, activations). All plain CUDA C++ -> compile under hipcc unchanged except the warp/atomic fault classes below. `__forceinline__`/`__device__` map directly; device `assert()` is fine on HIP.
- Warp intrinsics (THE fault surface): `__shfl_xor_sync(0xFFFFFFFF, x, mask)` in `warpReduce`/`warpMax` (winograd_helper.inc:434,443); `__shfl_down_sync(0xFFFFFFFF,...)` (common_kernels.cu:642); `__shfl_sync(0xFFFFFFFF, v, 0)` broadcasts (common_kernels.cu:814,821). HIP: `__shfl_*_sync` REQUIRE a 64-bit mask (AutoDock-GPU lesson, ROCm 7.x static_asserts sizeof(mask)==8) -> the literal `0xFFFFFFFF` fails to COMPILE on HIP. Plus wave64 semantics (see risks).
- `warpSize` / hardcoded 32: pervasive `& 0x1F`, `>> 5`, `<< 5`, `* 32`, `for(mask=16;...)`, per-warp shared arrays sized `[.../32]`, block sizes `kWarpsPerBlock*32`, `int size = N*32` launch math. These encode "warp == 32 lanes". Handled per-site (risks below); NOT a blanket s/32/warpSize/.
- Textures / surfaces: NONE in the backend (grep clean). No texture/pitch/layered-array fault classes apply.
- cuBLAS: `cublasSgemm`, `cublasHgemm`, `cublasGemmStridedBatchedEx` (with `CUDA_R_16F`/`CUDA_R_32F`, `CUBLAS_GEMM_DEFAULT`), `cublasSgemmStridedBatched`, `cublasCreate/SetStream/SetMathMode`, `cublasHandle_t`, `CUBLAS_OP_T/N`. -> hipBLAS: hipblasSgemm/Hgemm/GemmStridedBatchedEx/SgemmStridedBatched, hipblasCreate/SetStream, HIPBLAS_OP_*, HIP_R_16F/HIP_R_32F (`<hip/library_types.h>`), HIPBLAS_GEMM_DEFAULT. WATCH hipBLAS v2 enums (ROCm 7.x default): GemmStridedBatchedEx compute-type + algo enum signatures differ slightly -- verify against `hipblas.h` (present at /opt/rocm/include/hipblas/hipblas.h).
- cuDNN: only `network_cudnn.cc` (+ a `typedef void* cudnnHandle_t` fallback in cuda_common.h when USE_CUDNN unset). cuDNN is OPT-IN and NOT built on HIP for the lead port -> no MIOpen swap needed now. (Follow-up: a `network_cudnn`->MIOpen port is possible but out of scope; MIOpen's API is not 1:1 with cuDNN.)
- cuFFT/cuRAND/cuSPARSE/Thrust/CUB: NONE used by the backend.
- Pinned/managed memory: none (`cudaMalloc`/`cudaMemcpyAsync` only; host buffers are plain malloc, copied via cudaMemcpyAsync). No `cudaMallocManaged`/`cudaHostAlloc` in the default path.
- Streams/events: `cudaStreamCreateWithFlags(cudaStreamNonBlocking)`, `cudaEventCreateWithFlags(cudaEventDisableTiming)`, `cudaEventRecord`, `cudaStreamWaitEvent`, `cudaMemcpyAsync` -> all 1:1 HIP. `cudaSetDevice`/`cudaGetDeviceProperties` -> hipSetDevice/hipGetDeviceProperties.
- NVIDIA-only APIs to GUARD OUT on HIP:
  - L2 persistence (network_cuda.cc 786-870, `#if CUDART_VERSION >= 11000`): `cudaStreamAttrValue`, `accessPolicyWindow`, `cudaAccessPropertyPersisting`, `cudaStreamSetAttribute(cudaStreamAttributeAccessPolicyWindow)`, `cudaCtxResetPersistingL2Cache`. No HIP equivalent. Guard `&& !defined(USE_HIP)` (it is a pure perf hint; `allow_cache_opt_` then stays false).
  - CUDA graph capture flags `cudaEventWaitExternal`/`cudaEventRecordExternal` (network_cuda.cc 50-51, 769) -- `#undef`'d then used in capture path; ensure they resolve to 0 / are guarded on HIP.
  - `cublasSetMathMode(CUBLAS_TENSOR_OP_MATH | CUBLAS_PEDANTIC_MATH)` (network_cuda.cc 292-298): hipblasSetMathMode exists but the math-mode enum semantics (TF32/pedantic tensor-op) do not map; simplest correct path is to guard the math-mode calls out on HIP and let hipBLAS pick defaults.
- FP16: no fp16 *intrinsics* (`__hmul`/`__hadd2` etc.) -- kernels compute in float and cast half<->float, so `cuda_fp16.h` -> `hip/hip_fp16.h` (via compat header) is the only fp16 dependency. `half` type maps.

## Risk list
1. wave64 vs 32 -- broadcast-after-32-lane-reduce (HIGH, correctness). `softmax_opt_64_kernel` (common_kernels.cu:776): launched `N*32` threads, each thread owns 2 of a 64-element C-plane, does `warpMax`/`warpReduce` (32-lane butterfly) then `__shfl_sync(0xFFFFFFFF, maxval, 0)` broadcasts LANE 0 of the wavefront. On wave64 lanes 32-63 (a DIFFERENT N row's data) get row-0's max/sum -> wrong softmax for half the rows. Fix (gsplat/popsift lesson): width-32 subgroup ops -- replace the broadcast with `__shfl(v, (laneId&~31), 32)` style (read lane 0 OF THE 32-LANE SUBGROUP) and make the butterfly width 32. Same pattern to audit in any kernel that reduces over 32 lanes then broadcasts.
2. wave64 -- `__shfl_*_sync` 64-bit mask (HIGH, compile-blocking). `0xFFFFFFFF` literal fails the ROCm static_assert sizeof(mask)==8 in HOST pass of the .cu (AutoDock-GPU). Define a compat full-warp mask `#if defined(USE_HIP) ... 0xFFFFFFFFFFFFFFFFULL` else `0xFFFFFFFF`, keyed on USE_HIP not on width, and route warpReduce/warpMax/broadcasts through it. NOTE: a 64-bit-mask + width-64 butterfly over masks 16..1 reduces only WITHIN 32-lane halves (bits 0-4), so the *value* math is the 32-lane reduction either way; the broadcast (risk 1) and the data-layout (risk 3) are the real correctness items.
3. wave64 -- 32-lane data layout vs 64-lane wavefront (HIGH, correctness). `globalAvgPool_kernel` (common_kernels.cu:619, 32 threads x 2 elems = 64-elem plane, `opIndex=tid>>5`, `laneId=tid&0x1F`, write at `laneId==0`) and the SE kernels in fp16_kernels.cu (`lane=k&0x1F; warp=k>>5`, per-warp shared sums `[.../32][...]`). On wave64 a 64-lane wavefront spans TWO logical 32-warps. For globalAvgPool/SE the two 32-lane halves are SELF-CONTAINED (each half reduces its own plane, lanes 0 AND 32 each write their own `opIndex`/`shared_sums[warp]`) so they are likely wave64-CORRECT as-is -- but the per-warp shared array dimension `kMaxResBlockFusingSeKFp16Ampere/32` and `C/32` loop counts assume 32-lane warps; on wave64 there are HALF as many wavefronts as the host launch math (`kBlockSize=kWarpsPerBlock*32`) assumes, so block/grid sizing and shared-array indexing must be re-derived for warpSize=64. Treat each such kernel individually (popsift: pick the axis by how partials recombine); verify by GPU output diff vs the blas backend, not by inspection.
4. wave64 -- atomic-combined per-warp partials (MED). `softmax_kernel` (845) and `shared_sum_for_layer_norm`: per-32-lane partials combined into `__shared__` via `atomicMaxFloat`/`atomicAdd` guarded `(c&0x1F)==0`. On wave64 lanes 0 AND 32 both fire the atomic -> TWO correct 32-lane partials combined -> VALUE still correct (AutoDock-GPU "atomicAdd-combine makes warp granularity flexible"). Low risk, but confirm on GPU.
5. int atomicMin/atomicMax silently dropped (MED, cudaKDTree lesson). `atomicMaxFloat` (winograd_helper.inc:449) reinterprets float as int and uses `atomicMax((int*)addr,...)`/`atomicMin((unsigned*)addr,...)`. cudaKDTree found these no-op on COARSE-GRAINED/managed memory on gfx90a. Here `addr` is `&maxval` in `__shared__` memory (device-local), NOT managed -- expected OK, but explicitly verify: if the softmax max comes out wrong, this is the cause; emulate with a CAS loop on HIP if so.
6. hipBLAS v2 enums (MED). `cublasGemmStridedBatchedEx` compute-type/algo enum + `CUDA_R_16F` mapping must use the v2 hipBLAS spelling (ROCm 7.x). Verify signatures against the installed hipblas.h; mis-mapped enum -> wrong-precision GEMM or a runtime status error.
7. Device-capability gating misfires on HIP (MED, functional). network_cuda.cc 242-278: FP16 support + `has_tensor_cores_` are decided from `deviceProp.major>=7` (CUDA SM numbers) and a `strstr(name,"GTX 16")` check; the FP16-unsupported branch THROWS. On HIP, `hipDeviceProp_t.major/minor`/`gcnArchName` do not follow CUDA SM numbering, so `cuda-fp16`/`hip-fp16` could wrongly throw "Your GPU doesn't support FP16" on gfx90a (which has full FP16/MFMA). Add a `#if defined(USE_HIP)` branch: enable FP16 unconditionally on gfx9+, set tensor-cores true (MFMA) so the fp16 path runs.
8. Rule-of-five on handles (LOW). No textures; streams/events/cublas handles are created once and destroyed in the network dtor. Confirm the dtor guards (don't double-destroy a 0 handle) -- AMD faults where CUDA tolerates. Low risk (single owner), but check the destructor.
9. Meson custom_target + hipcc interplay (LOW-MED, build). hipcc with `-include hip_compat.h` must not trip "cannot specify -o when generating multiple output files" (cupoch: use the FULL path to the force-include header, and `-c` single TU per custom_target as upstream already does). `-fPIC` is needed (lc0 already passes `-Xcompiler -fPIC` to nvcc; pass `-fPIC` to hipcc). The .o from hipcc links into a g++-built executable/library -> may need `-fPIE`/`-fPIC` consistency (AutoDock-GPU R_X86_64_32 PIE lesson). C++17 required (lc0's nvcc path negotiates 14/17/20; rocPRIM/hipBLAS headers and CUTLASS-off path are fine at 17).
10. CUTLASS (RESOLVED, not a risk if left off). Do NOT attempt to port `cutlass_kernels.cu` or the `subprojects/cutlass.wrap` (NVIDIA/cutlass v4.4.1) / `third_party/fused_multi_head_attention`. Keep `-Dcutlass` off and do not compile `cutlass_kernels.cu` on HIP; the `#ifdef USE_CUTLASS` fallbacks in layers.cc:1776 and network_cuda.cc:232 take the cuBLAS path.

## File-by-file change list
NEW:
- `src/neural/backends/cuda/hip_compat.h` -- the single CUDA->HIP alias header (runtime + hipBLAS + library_types), force-included on every HIP .cu, included by cuda_common.h under USE_HIP.

EDIT (all behind `#if defined(USE_HIP)` / new meson `if get_option('hip')`, NVIDIA path unchanged):
- `meson.build` -- add the `hip` build branch: hipBLAS/amdhip64 discovery (reuse the sycl=amd block), `hipcc` custom_targets for common_kernels.cu + fp16_kernels.cu (NOT cutlass_kernels.cu, NOT network_cudnn.cc), add network_cuda.cc(+layers.cc) to files, `-DUSE_HIP`, arch from `${amd_gfx}`.
- `meson_options.txt` -- add `option('hip', boolean, false)`. Reuse existing `amd_gfx`, `hip_libdirs`, `hip_include` options (already defined for sycl=amd).
- `src/neural/backends/cuda/cuda_common.h` -- top: `#if defined(USE_HIP) #include "hip_compat.h" #else #include <cublas_v2.h>/<cuda_fp16.h>/<cuda_runtime.h> #endif`. Map `cublasStatus_t`/error helpers if names differ.
- `src/neural/backends/cuda/winograd_helper.inc` -- warpReduce/warpMax: 64-bit full-warp mask; width-32 shuffles where the reduce is logically 32-lane. atomicMaxFloat: keep (verify, CAS-loop fallback only if GPU shows breakage).
- `src/neural/backends/cuda/common_kernels.cu` -- the `__shfl_*_sync` masks; the lane-0 broadcasts in softmax_opt_64_kernel -> width-32 subgroup broadcast; audit globalAvgPool / softmax launch+index math for warpSize=64.
- `src/neural/backends/cuda/fp16_kernels.cu` -- SE kernel per-warp shared-array dims and `k&0x1F`/`k>>5` for warpSize=64; same `__shfl` mask fix.
- `src/neural/backends/cuda/network_cuda.cc` -- guard out L2-persistence block and cudaEventWaitExternal/RecordExternal and cublasSetMathMode on HIP; HIP-aware FP16/tensor-core capability gating; verify handle dtor.
- `src/neural/backends/cuda/kernels.h`, `layers.h`, `inputs_outputs.h` -- only if a symbol needs a HIP alias not covered by the compat header.

NOT touched on HIP: `network_cudnn.cc`, `cutlass_kernels.cu`, `subprojects/cutlass.wrap`, `third_party/fused_multi_head_attention`.

## Build commands (gfx90a)
Toolchain present: /opt/rocm-7.2.1, hipcc, hipBLAS 3.2, meson 1.11.1, ninja, 4x gfx90a.
Subproject deps fetched by meson at configure (git): abseil-cpp, protobuf, zlib, eigen, gtest (all on the wraps). Configure + build:
```
cd projects/lc0/src   # in the FORK clone, once the porter creates it -- planner is read-only
meson setup build-hip \
  -Dhip=true -Damd_gfx=gfx90a \
  -Dplain_cuda=false -Dcudnn=false -Dcutlass=false \
  -Dgtest=true -Dblas=true \
  -Dhip_libdirs=/opt/rocm/lib -Dhip_include=/opt/rocm/include
ninja -C build-hip
```
(`-Dplain_cuda=false` disables the nvcc path so meson does not require nvcc on a ROCm-only host; the new `-Dhip=true` provides the GPU backend. Exact option names to be finalized by the porter when the meson branch is written.)
Same commit builds gfx1100/gfx1151 with only `-Damd_gfx=gfx1100|gfx1151` (configurable-arch lesson) -- no source edit.
Optional CPU-only compile smoketest: `rocm/dev-ubuntu-24.04:7.2.4-complete` (compile check only, never a validation gate).

## Test plan
The gtest suite (ChessBoard, FP16, HashCat, PositionTest, OptionsParserTest, SyzygyTest, EncodePositionForNN, EngineTest) is CPU-only -- it does NOT exercise any GPU backend. It is the NON-GPU REGRESSION SET: `meson test -C build-hip` must stay green (no regression). Build with `-Dgtest=true`.

GPU validation (exercises the HIP kernels + hipBLAS on real gfx90a):
1. Obtain a small/medium Leela network (`.pb.gz`) from lczero.org (a small T-net or the bundled benchmark net). The validator fetches it once into agent_space.
2. Backend run -- `./build-hip/lc0 benchmark --backend=hip --weights=<net> --nodes=<N>` (and `--backend=hip-fp16`). `lc0 benchmark` runs real searches through the backend (nps + node count); a non-crashing run that produces sane nps and legal bestmoves proves the kernels + GEMMs execute. Also `./lc0 backendbench --backend=hip --weights=<net>` (BackendBenchmark, raw backend throughput, no search).
3. CORRECTNESS cross-check (the real proof): run the SAME net + SAME positions through the trusted CPU `blas` backend (built in by default) and through `hip`/`hip-fp16`, and compare the network OUTPUT (policy distribution + value/WDL) per position. Use `lc0`'s own backend-compare path if available, else a small UCI harness feeding a fixed FEN set and diffing the eval. Acceptance: fp32 `hip` vs `blas` value within ~1e-3 and top policy moves identical; `hip-fp16` within fp16 tolerance (looser, ~1e-2 on value, same argmax policy). This catches the wave64 softmax/broadcast bugs (risks 1,3) which a non-crashing benchmark would NOT.
4. Determinism: same FEN + fixed nodes twice on `hip` -> identical bestmove/eval (the backend forward pass is deterministic; non-determinism here fingerprints a wave64 reduction race).
5. Confirm device execution via `AMD_LOG_LEVEL=3` (named HIP kernel dispatches + hipBLAS calls), and run ctest/benchmark SERIALLY on the single assigned GPU (MPPI lesson: parallel processes on one GPU cause false transient failures).

Validation is complete only when: (a) `hip` and `hip-fp16` benchmark/backendbench run clean on gfx90a, (b) network outputs match the blas backend within tolerance over a FEN set, (c) determinism holds, (d) the CPU gtest set does not regress.

## Inter-project MOAT deps
NONE. lc0's HIP backend needs only hipBLAS (system ROCm, /opt/rocm) and meson-fetched host libs (abseil/protobuf/zlib/eigen/gtest). No MOAT-ported dependency (no rmm/raft/etc.). `depends_on` left empty.

## Open questions
1. Backend NAME: register the HIP backend as `hip`/`hip-fp16` (clean, distinct) vs reuse `cuda`/`cuda-fp16` (smaller user-facing change). Recommend `hip` to avoid clashing on a dual-stack machine; porter decides.
2. Whether to also build `network_cudnn.cc` against MIOpen later. Out of scope for the lead port (cuDNN is opt-in; default backend needs only hipBLAS). Flag as a possible follow-up only.
3. Whether the porter folds the HIP backend INTO the existing cuda meson block (sharing nvcc-vs-hipcc selection) or adds a parallel `if get_option('hip')` block. Parallel block is lower-risk for keeping the NVIDIA path byte-identical; revisit if it duplicates too much.
4. AMD-native fused-MHA (ck_tile) to replace the disabled CUTLASS path -- explicitly a FUTURE perf pass, not the correctness-first lead port.
