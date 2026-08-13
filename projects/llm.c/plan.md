# Port plan: llm.c (lead platform linux-gfx90a)

## Project
- Name: llm.c
- Upstream: https://github.com/karpathy/llm.c
- Default branch: master
- Pinned upstream HEAD at planning: f1e2ace651495b74ae22d45d1723443fd00ecd3a (clone in projects/llm.c/src, gitignored, read-only)
- What it is: GPT-2/GPT-3 LLM training in raw C/CUDA, no PyTorch runtime dependency for the C/CUDA path. ~30k stars. The CUDA path is hand-written kernels in `llmc/*.cuh` plus three driver translation units: `train_gpt2.cu` (mixed precision: BF16 default / FP16 / FP32, the main one), `train_gpt2_fp32.cu` (standalone FP32, older, cooperative-groups based), and the test drivers `test_gpt2.cu` / `test_gpt2_fp32.cu`.

## Existing AMD support -> decision: PROCEED (finish/modernize a stale port; correctness-first mechanical HIP port)
- Upstream README (line 205-206) points to an external AMD fork: `anthonix/llm.c` ("support for AMD devices, such as the 7900 XTX"). Upstream itself has ZERO HIP/ROCm code (only matches are the README link and incidental `chip`/`Ampere` substrings).
- `anthonix/llm.c` facts (via gh api):
  - Supports gfx906 (Radeon VII), gfx90a (MI250X -- our lead arch), gfx1100 (7900 XTX), gfx942/MI300. Uses ROCm 6.2.
  - Approach: a Makefile `train_gpt2amd` target that runs `hipify-perl` to translate each `.cu` into `build/hip/%.cu`, then compiles with `hipcc --offload-arch=<gfx>`, linking `-lamdhip64 -lhipblaslt` (+ `-lrccl -lmpi` for multi-GPU). cuBLASLt -> hipBLASLt.
  - Last pushed 2024-09-23 -- roughly 20 months stale. Never upstreamed (the README only links the fork). Not archived.
- Classification per PORTING_GUIDE "assess existing AMD support": this is an "abandoned / incomplete ROCm/HIP port" that is also "below best practices" (in-place hipify-perl translation of every `.cu`, not the minimal-footprint compat-header model). It targets ROCm 6.2 and will have bitrotted against the ROCm 7.2.1 in use here (the GPUMD lesson: stale HIP makefiles bitrot against the current ROCm; the MOAT value is to build + GPU-validate + fix the rot). It is NOT mature upstream support, so it is NOT an already-supported skip.
- Decision: PROCEED. Do a clean MOAT port on top of CURRENT upstream master (Strategy A compat-header, minimal footprint), informed by but not adopting the anthonix tree (its hipify-everything model churns every file and is the opposite of MOAT's small-diff goal). The anthonix fork is a useful reference for which library swaps and arch list work, and as a cross-check that gfx90a runs.
- Port-vs-rewrite for perf-critical kernels: CORRECTNESS-FIRST MECHANICAL PORT. llm.c's attention/matmul are NOT CUTLASS/CuTe/wgmma; matmul is plain cuBLASLt GEMM (-> hipBLASLt, 1:1) and the kernels are hand-written shuffle/reduction CUDA, which translate to HIP directly. The optional cuDNN flash-attention path is the only NVIDIA-library-locked kernel and it is OFF by default; the in-repo `attention.cuh` (manual flash attention) is the default and ports mechanically. No AMD-native (rocWMMA/CK) rewrite is needed for a first correct port; a later MFMA/CK tuning pass on matmul/attention is possible follow-up but explicitly out of scope here.

## Build classification -> Makefile (non-CMake, non-pytorch) => Strategy A semantics
- Evidence: top-level `Makefile` drives everything; no `CMakeLists.txt`, no `find_package(Torch)`, no `setup.py`/`pyproject.toml`, no `torch.utils.cpp_extension`/`CUDAExtension`. `train_gpt2.py` is only a reference/data-gen script (PyTorch), not a build of the C/CUDA code.
- The Makefile hardcodes `NVCC := $(shell which nvcc)`, `NVCC_LDFLAGS = -lcublas -lcublasLt -lnvidia-ml`, optional `-lcudnn` (USE_CUDNN), optional `-lnccl` (NCCL/MULTI_GPU), optional `-lmpi` (USE_MPI). `.cu` are compiled directly by `nvcc` (Makefile lines 273-286).
- ext_type recorded as `makefile` (closest to the Strategy-A / non-torch family; the repo's existing token `cmake` would be misleading since there is no CMake). Strategy B (torch hipify) does NOT apply -- this is not a torch extension.

## Port strategy: A (compat header), Makefile-flavored
Goal: NVIDIA build byte-for-byte unchanged; only the HIP build sees HIP; small additive diff.

1. Add one compat header `llmc/cuda_to_hip.h` guarded by `#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)`:
   - `#include <hip/hip_runtime.h>`, and FIRST `#include <cstring>`/`#include <cstdlib>` BEFORE the HIP runtime so libc host `memcpy`/`memset` win over HIP `__device__` overloads inside `.cu` compiled as HIP (gpuRIR lesson) -- relevant because `Packed128`/`get_bits` use host+device `memcpy`.
   - Alias the CUDA spellings the project actually uses to HIP: runtime (`cudaMalloc/Free/Memcpy/MemcpyAsync/MallocHost/FreeHost/Stream*/Event*/DeviceProp/GetDeviceProperties/...`), bf16/fp16 headers (`cuda_bf16.h`->`hip/hip_bf16.h`, `cuda_fp16.h`->`hip/hip_fp16.h`; type aliases `__nv_bfloat16`->`__hip_bfloat16` etc.), and the full-warp mask (see fault classes). Use `torch/utils/hipify/cuda_to_hip_mappings.py` as the authoritative name source.
   - On NVIDIA the header is a near no-op (include `<cuda_runtime.h>`), so the CUDA path is untouched.
   - Rather than touch every angle-bracket toolkit include (`<cuda_runtime.h>`, `<cublas_v2.h>`, `<cublasLt.h>`, `<nvtx3/...>`, `<cuda_profiler_api.h>`, `<nvml.h>`), prefer forwarding shim headers of those exact names in a `llmc/hip_shims/` dir added to the include path ONLY for the HIP build (MPPI-Generic lesson), each `#include`-ing the compat header / the right HIP header (e.g. `<hip/hip_runtime_api.h>`, hipBLAS/hipBLASLt). Keeps `cuda_common.h`/`cublas_common.h` include lines unchanged -> zero NVIDIA churn.

2. Makefile: add a HIP toolchain branch (mirror of the nvcc branch), do NOT rename `.cu` files.
   - Detect ROCm (`HIPCC := $(shell which hipcc)`); add an AMD target set `train_gpt2amd test_gpt2amd train_gpt2fp32amd test_gpt2fp32amd` OR (cleaner) reuse the existing target names by switching the compiler when `USE_HIP=1`/`hipcc` is found. Prefer: keep existing target names, select `NVCC := hipcc` semantics behind a `USE_HIP` switch so README commands (`make test_gpt2cu`) work unchanged on AMD.
   - HIP arch must be CONFIGURABLE, never a literal: `HIPCC_FLAGS += $(addprefix --offload-arch=,$(AMDGPU_TARGETS))` with `AMDGPU_TARGETS ?= gfx90a` (default the lead arch only when unset). This lets followers build with only `AMDGPU_TARGETS=gfx1100`/`gfx1151` and NO source/Makefile edit -> no head_sha churn, no forced revalidation (CudaSift/Gpufit lesson).
   - HIP link flags: `-lhipblas -lhipblaslt` (replaces `-lcublas -lcublasLt`); drop `-lnvidia-ml` (NVML) unless an amdsmi shim is added (see risks); `-lrccl` in place of `-lnccl` when multi-GPU; `-lmpi` unchanged.
   - `--use_fast_math -std=c++17` are accepted by hipcc; `-t=0/--threads=0` and `--generate-code arch=...` are nvcc-only and must be guarded out of the HIP branch. Keep `-std=c++17` (rocPRIM/hipCUB would need it, though llm.c uses neither -- still safest).
   - Compile `.cu` directly with `hipcc` (no separate hipify-perl step; hipcc compiles `.cu` as HIP given the compat header handles symbol names). This is the minimal-footprint divergence from anthonix's hipify-translate-then-compile model.

3. Guard genuinely divergent code with `#if defined(USE_HIP)`; keep guards rare (warp size, full-warp mask, NVML, NVTX).

## CUDA surface inventory
- Custom kernels (the bulk): hand-written kernels across `llmc/{adamw,attention,encoder,fused_classifier,gelu,global_norm,layernorm,matmul,zero}.cuh` and `cuda_utils.cuh`. All standard CUDA C++; translate to HIP via the compat header.
- Warp intrinsics: `__shfl_xor_sync(0xFFFFFFFF, ...)` and `__shfl_down_sync(0xffffffff, ...)` in `cuda_utils.cuh` (warpReduceSum/Max, blockReduce) and `matmul.cuh` (sub-warp width-4 shuffles). `train_gpt2_fp32.cu` uses Cooperative Groups (`cg::`) reductions (`thread_block_tile`, `cg::reduce`). 15 occurrences of the 32-bit `0xffffffff` mask.
- Warp size: `#define WARP_SIZE 32U` in `cuda_common.h`, used in reductions, shared-array sizing (`__shared__ float shared_val[WARP_SIZE]`), `num_warps = blockDim.x/WARP_SIZE`, and in encoder stochastic-rounding RNG seeds (`bucket*WARP_SIZE+threadIdx.x`).
- Libraries:
  - cuBLAS + cuBLASLt (`cublas_common.h`, `matmul.cuh`, `attention.cuh`, both train/test drivers): matmuls and the LtMatmul path -> **hipBLAS + hipBLASLt** (libs present: /opt/rocm/lib/libhipblaslt.so). Watch the v2 enum/handle differences (PORTING_GUIDE library-swap note) and hipBLASLt epilogue/`cublasLtMatmulDesc` differences.
  - cuDNN flash attention (`cudnn_att.cpp/.h`, gated by USE_CUDNN, OFF by default) depends on the header-only `cudnn-frontend` graph API (`<cudnn_frontend.h>`) -> **no clean MIOpen equivalent of the frontend graph API**. Plan: leave USE_CUDNN off for the AMD port (ship the in-repo `attention.cuh` manual flash attention, which is the default). A MIOpen attention path is a large separate effort, explicitly out of scope.
  - NCCL (`zero.cuh`, train/profile/test drivers, gated by MULTI_GPU) -> **RCCL** (librccl.so present); API is 1:1 (`ncclAllReduce` etc.). Multi-GPU is optional; single-GPU validation does not need it.
  - MPI (OpenMPI, gated by USE_MPI) -> unchanged (host-side, vendor-neutral).
  - NVML (`mfu.h`, already `#if __has_include(<nvml.h>)` guarded) -> no nvml.h on ROCm, so it compiles out gracefully (MFU/clock readout disabled). Optional amdsmi/rocm_smi shim is a nice-to-have, not required.
  - NVTX (`cuda_common.h`, `zero.cuh`: nvtxRangePush/Pop) -> roctx (`roctracer/roctx.h`) or stub to no-ops under USE_HIP. Stub is simplest; profiling-only.
- bf16/fp16: `__nv_bfloat16`/`half`, `__ldcs`/`__stcs` streaming loads (HIP provides both), `__float2bfloat16` etc. -> `hip/hip_bf16.h`, `hip/hip_fp16.h`. The `__ldcs`/`__stcs` over `int4`/bf16 work on HIP.
- Packed128 (`cuda_utils.cuh`): `alignas(16)` 128-bit packed load/store via `int4` reinterpret + `memcpy` (host+device). Needs the `<cstring>`-before-hip_runtime ordering (gpuRIR lesson).
- `__launch_bounds__` (cuda_common.h, layernorm.cuh, fused_classifier.cuh, fp32 driver): HIP supports `__launch_bounds__`; the `MAX_1024_THREADS_BLOCKS` macro keys on `__CUDA_ARCH__ == 800 || >= 900` -- `__CUDA_ARCH__` is NOT defined under HIP device compilation (cudaKDTree lesson), so the macro falls to the `#else` (=1) on AMD, which is a safe occupancy default. Consider a USE_HIP branch if 2-blocks/SM matters for perf later.
- Pinned/WC memory: `cudaMallocHost`/`cudaHostAllocWriteCombined`/`cudaFreeHost` (cuda_common.h double-buffered file IO) -> hipHostMalloc/hipHostFree; `cudaHostAllocWriteCombined` -> hipHostMallocWriteCombined flag.
- Streams/events: main_stream + per-op streams, `cudaEvent` timing in train driver -> 1:1 hip; note hipEvent timing calls are nodiscard in HIP headers (amgcl lesson) -- wrap in the existing cudaCheck macro if a warning appears.
- No CUB / no Thrust (confirmed: zero matches). No textures/surfaces. No managed memory (`cudaMallocManaged`) in the core path. These remove several fault classes entirely.

## Risk list (ranked; cite PORTING_GUIDE fault classes)
1. **Full-warp mask fails to COMPILE on HIP** (PORTING_GUIDE "HIP __shfl_sync requires 64-bit mask" + AutoDock-GPU): every `__shfl_xor_sync(0xFFFFFFFF,...)`/`__shfl_down_sync(0xffffffff,...)` literal is a 32-bit mask; ROCm 7.x static_asserts `sizeof(MaskT)==8`. Surfaces during hipcc's HOST pass of the `.cu`. Fix: a compat full-warp mask macro `FULL_MASK` = `0xffffffffffffffffULL` under USE_HIP, `0xffffffffu` on CUDA; substitute at the call sites in `cuda_utils.cuh`/`matmul.cuh`. NOT keyed on wave width.
2. **Warp size 32-vs-64, reduction correctness** (PORTING_GUIDE Warp size; AutoDock-GPU "pick the reduction axis"): `warpReduceSum/Max` loop `offset=16;offset>0;offset/=2` reduces only 32 of 64 lanes on gfx90a (wave64) -> WRONG numerically (silent, not a compile error). Decide axis: these are full-warp reductions recombined cross-warp via shared mem, so port to NATIVE 64 lanes (start offset = warpSize/2 = 32) on gfx90a; set `WARP_SIZE` from a per-arch constant (64 on __GFX9__, else 32) for device code and from `deviceProp.warpSize` for host launch/shared sizing. matmul.cuh sub-warp shuffles use explicit `width=4` (logical sub-groups of 4) -- those are wave-size-agnostic in count but still need the 64-bit mask; verify width-4 semantics hold on wave64 (they should, width<=warpSize).
3. **`__shared__ float shared_val[WARP_SIZE]` sizing** (PORTING_GUIDE: static shared arrays sized by warp count need a compile-time upper bound): `blockReduce` sizes the cross-warp scratch to WARP_SIZE entries and indexes by warp_id (num_warps = blockDim.x/WARP_SIZE). With WARP_SIZE=64 a 1024-thread block has 16 warps (fits 64). Safe if WARP_SIZE is the device constant 64 on gfx90a; do NOT leave the array at 32 while warps compute as if 64. Use a compile-time upper bound to be safe across followers (gfx1100 wave32 -> 32 warps in a 1024 block, also fits 64).
4. **Cooperative-groups path in `train_gpt2_fp32.cu`** (`cg::thread_block_tile<32>`, `cg::reduce`, `tiled_partition`): HIP supports cooperative groups, but a hardcoded `tile<32>` partition on wave64 is the same 32-vs-64 question; HIP's `hip/hip_cooperative_groups.h` tiled_partition<32> is valid (sub-wave tile) so this likely ports as-is, but VERIFY numerically -- the fp32 test (`test_gpt2fp32cu`) is the guard.
5. **hipBLASLt API/enum drift** (PORTING_GUIDE library swaps): `cublasLtMatmul`, `cublasLtMatmulDesc*`, epilogue enums, `cublasComputeType_t CUBLAS_COMPUTE_32F`, `CUDA_R_16BF`/`CUBLAS_LOWP` -> hipBLASLt equivalents (`HIPBLAS_COMPUTE_32F`, `HIP_R_16BF` from `<hip/library_types.h>`). hipBLASLt is the higher-rot-risk dependency; mismatched epilogue/scale-type handling is the most likely first build/numerical break.
6. **Determinism / reproducibility shift, not a bug**: encoder stochastic-rounding seed uses WARP_SIZE and threadIdx; changing 32->64 changes the RNG stream, so AMD output will not be bitwise-identical to CUDA. This is expected; validate against the C reference tolerances, not against CUDA bitwise. (Aligns with the MPPI/GPUMD determinism lessons -- validate by tolerance/physics, not bit-equality across vendors.)
7. **Warp-synchronous reductions without __syncwarp** (PORTING_GUIDE / MPPI-Generic): audit each reduction for an unsynced warp-synchronous tail on wave64. The shuffle reductions here use `__shfl_*_sync` (carry their own sync semantics) so are lower risk than a `volatile` sdata tail, but check `layernorm.cuh`/`global_norm.cuh`/`fused_classifier.cuh` reductions specifically.
8. **memcpy host/device overload collision in Packed128** (gpuRIR): include `<cstring>` before `<hip/hip_runtime.h>` in the compat header.
9. **NVTX/NVML/`__CUDA_ARCH__` macro divergence** (cudaKDTree): `__CUDA_ARCH__`-keyed macros (`MAX_1024_THREADS_BLOCKS`, the `__ldcs`/`__stcs` bf16 fallback in cuda_common.h) behave differently under HIP (no `__CUDA_ARCH__`); confirm the fallbacks pick the right branch. Stub NVTX, compile out NVML.
10. Lower risk / likely-clean (no work expected): no textures/surfaces (avoids the entire texture pitch / linear-filter / layered-array fault-class family), no CUB/Thrust, no managed memory, no atomicMin/Max-on-int (encoder backward is bucketed-deterministic, not atomicAdd). int atomics that DO exist are atomicAdd (unaffected by the cudaKDTree coarse-grained atomicMin/Max bug).

## File-by-file change list (planned; porter executes)
- ADD `llmc/cuda_to_hip.h` -- the single compat header (runtime aliases, bf16/fp16 header redirect, `FULL_MASK`, per-arch `WARP_SIZE`/`kWarpSize` constant, NVTX stubs).
- ADD `llmc/hip_shims/{cuda_runtime.h,cublas_v2.h,cublasLt.h,cuda_profiler_api.h,nvml.h,nvtx3/...}` -- forwarding shims on the HIP-only include path so existing include lines are untouched (MPPI-Generic pattern); OR, if simpler, minimal guarded edits to `cuda_common.h`/`cublas_common.h` includes. Decide during port.
- EDIT `Makefile` -- add HIP toolchain branch: detect hipcc, `AMDGPU_TARGETS ?= gfx90a` -> `--offload-arch`, swap link flags (`-lhipblas -lhipblaslt`, `-lrccl`, drop `-lnvidia-ml`), guard nvcc-only flags, keep target names working under USE_HIP. No `.cu` rename.
- EDIT `llmc/cuda_utils.cuh` -- `FULL_MASK` at the two `__shfl_xor_sync` sites; reduction start offset from warpSize; `blockReduce` shared-array upper bound. Guarded by USE_HIP.
- EDIT `llmc/matmul.cuh` -- `FULL_MASK` at the four `__shfl_down_sync` sites; verify width-4 sub-warp on wave64.
- EDIT `llmc/cublas_common.h` -- hipBLAS/hipBLASLt types/enums under USE_HIP (`CUBLAS_LOWP`, compute type, handle).
- EDIT `llmc/cuda_common.h` -- WARP_SIZE per-arch constant; `MAX_1024_THREADS_BLOCKS` HIP branch; NVTX stub; bf16 `__ldcs`/`__stcs` fallback guard.
- POSSIBLY EDIT `llmc/{layernorm,global_norm,fused_classifier,zero,adamw,encoder}.cuh` -- only where a reduction/mask/warp-size assumption surfaces during build or numerical validation. Keep edits minimal and USE_HIP-guarded.
- `train_gpt2_fp32.cu` -- cooperative-groups tiles; edit only if the fp32 numerical test fails.
- Files NOT expected to change: the C-only `train_gpt2.c`/`test_gpt2.c`, `dataloader.h`, `tokenizer.h`, all of `dev/`. cuDNN files left as-is (USE_CUDNN off on AMD).

## Build commands (gfx90a)
Prereqs present on this host: ROCm 7.2.1, hipcc, hipify-perl/clang, libhipblas.so + libhipblaslt.so + librccl.so, ROCm PyTorch 2.13 (hip 7.2, torch.cuda.is_available()==True).
- Data/reference (once): `pip install -r requirements.txt` (torch already ROCm); `python dev/data/tinyshakespeare.py`; `python train_gpt2.py` (writes gpt2_124M.bin + gpt2_124M_debug_state.bin; runs on the MI250X via ROCm torch, or `--device=cpu`). Simpler: `./dev/download_starter_pack.sh` ships these .bin files pre-generated, avoiding the torch run entirely.
- Build (planned interface): `USE_HIP=1 AMDGPU_TARGETS=gfx90a make test_gpt2cu train_gpt2cu test_gpt2fp32cu train_gpt2fp32cu` (BF16 default), and `PRECISION=FP32 USE_HIP=1 AMDGPU_TARGETS=gfx90a make ...` for the FP32 build. NO_MULTI_GPU=1 to skip RCCL for the single-GPU bringup. USE_CUDNN stays 0 on AMD.
- CPU regression targets (must keep building/passing): `make train_gpt2 test_gpt2` and `cd dev/test && make`.

## Test plan
GPU correctness (real MI250X / gfx90a). Mirror the upstream ci_gpu.yml recipe, minus cuDNN:
1. `./test_gpt2fp32cu` -- FP32 forward/backward vs the PyTorch reference debug state, per-tensor tolerances in `test_gpt2.cu` (`check_tensor`, grad_thresholds, logit_accuracy_threshold). Strictest correctness gate (tight tolerances).
2. `./test_gpt2cu` (BF16 default) and its variants `-r 0` (no recompute GeLU), `-r 2` (recompute LN), `-w 0` (no master weights), `-b 32` (larger batch) -- the matrix the CI runs.
3. Training loss check: `./train_gpt2cu -b 1 -t 64 -d 256 -l 0.0001 -v 200 -s 200 -a 1 -x 10 -r 0 -f 0 -e gpt2_124M.bin > out.txt` then `python dev/loss_checker_ci.py -f out.txt -s 20 -e 28 -a 5.0` (asserts loss curve matches within tolerance). Run for both FP32 and BF16 builds.
4. Full tinyshakespeare smoke: `OMP_NUM_THREADS=8 ./train_gpt2cu` (40 steps, validation loss + sampling) -- proves end-to-end training stability and that the loss decreases / no NaN.
5. Determinism sanity: two identical `./test_gpt2cu` runs should pass the SAME tolerance check (bitwise-vs-CUDA is NOT expected due to wave-size RNG seed shift; the C-reference tolerance is the bar).
6. Multi-GPU (optional, only if a 2nd GPU is available on the host): `mpirun -np 2 ./train_gpt2cu` with RCCL to exercise the zero.cuh/RCCL path. Not a blocker for lead validation.
Non-GPU regression set (must not regress): `make train_gpt2 test_gpt2 && ./test_gpt2` (the C/CPU forward/backward vs reference) and `cd dev/test && make && ./test_dataloader` + `./test_outlier_detector`.

## Inter-project MOAT deps
NONE. llm.c vendors no MOAT-tracked library; it links only system ROCm libs (hipBLAS, hipBLASLt, RCCL, MPI). No rmm/raft/etc. `set-deps` will be left empty.

## Open questions
- hipBLASLt epilogue/scale-type parity for the bf16 GEMM path (matmul.cuh / attention.cuh) -- most likely first numerical/build break; resolve during port by aligning compute type and `CUBLAS_LOWP` -> hipBLASLt enums.
- Whether matmul.cuh's `__shfl_down_sync(...,width=4)` sub-warp reductions need any change on wave64 (expected fine since width<=warpSize, but verify numerically via test_gpt2cu).
- Whether to keep README target names (`make test_gpt2cu` with USE_HIP=1) vs add explicit `*amd` targets like anthonix. Leaning to reuse names behind USE_HIP for a smaller diff and unchanged docs; finalize in port.
- cuDNN/MIOpen flash attention parity is intentionally deferred (out of scope); revisit only if a perf pass is requested.

## Delta plan: follower platforms (placeholder)
gfx1100/gfx1151 are RDNA (wave32). The wave-size abstraction above (per-arch WARP_SIZE = 32 on RDNA) and configurable `AMDGPU_TARGETS` mean the same curated commit should build with only `AMDGPU_TARGETS=gfx1100`/`gfx1151` -- no source change expected. Followers validate first, delta-port only on failure. anthonix already ran gfx1100 (7900 XTX), a positive signal for the RDNA path. (To be expanded on demand.)
