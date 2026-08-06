# CUDA-L2 -- ROCm/HIP port plan (lead: linux-gfx90a)

## Project
- Name: CUDA-L2
- Upstream: https://github.com/deepreinforce-ai/CUDA-L2 (deepreinforce-ai)
- Default branch: main
- What it is: an LLM+RL system whose deliverable is a corpus of ~3,700 auto-generated, per-(M,N,K)-tuned half-precision GEMM (HGEMM) CUDA kernels that claim to beat cuBLAS/cuBLASLt on specific NVIDIA GPUs (RTX 3090 SM80, A100 SM80, H100 SM90). The repo ships the kernels plus a torch.utils.cpp_extension JIT harness and cuBLAS/cuBLASLt baselines for the benchmark comparison. Paper: arXiv 2512.02551.

## Existing AMD support
None. No HIP path, no OpenCL/Vulkan/SYCL path, no stale ROCm branch or fork. Decision: a ROCm/HIP port is NOT warranted as a mechanical translation -- see disposition below. This is the CUTLASS/CuTe + Hopper-PTX reimplement-not-port class.

## Build classification: torch-extension
Evidence:
- `tools/utils.py` `build_from_sources` -> `torch.utils.cpp_extension.load(name="hgemm_lib", sources=[...], extra_cuda_cflags=[...])` (JIT build, no CMake, no setup.py).
- `compile.py` and `eval_one_file.sh` drive that JIT load at eval time.
- Every kernel and the pybind glue `#include <torch/extension.h>`; the module is exposed via `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` (pybind/hgemm_*.cc).
- README requires `pip` PyTorch >= 2.6.0 and `TORCH_CUDA_ARCH_LIST` ("8.0", "9.0a").
Normally this -> Strategy B (build against a ROCm torch; hipify the .cu at build time). That classification is moot here because of the CUTLASS/CuTe core (next section).

## Port strategy: NONE viable as a mechanical port -> recommend BLOCKED (cant-port / reimplement-not-port)
Hard dependency on NVIDIA CUTLASS/CuTe and NVIDIA-tensor-core PTX, with no portable hand-written path that is the actual product:
- README mandates `git clone -b v4.2.1 https://github.com/NVIDIA/cutlass.git` and `CUTLASS_DIR`; `tools/utils.py get_build_cuda_cflags` hard-injects `-I $CUTLASS_DIR/include` and `-I $CUTLASS_DIR/tools/util/include`. CUTLASS/CuTe does NOT port to ROCm and never will (PORTING_GUIDE, 2026-05-30). There is no CuTe-on-ROCm shim.
- 3560 of 3736 kernels are pure CuTe: `#include <cute/tensor.hpp>`, `using namespace cute`, `TiledMMA`/`make_tiled_mma`, `MMA_Atom<MMA_Traits<SM80_16x8x16_F16F16F16F16_TN>>` / `SM80_16x8x16_F32F16F16F32_TN`, `partition_fragment_A/B/C`, CuTe copy atoms `SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>` (Ampere cp.async), `SM75_U32x4_LDSM_N` (ldmatrix). These are NVIDIA-PTX MMA/copy atoms expressed through CuTe's type system; there is no HIP/rocWMMA spelling and hipify cannot translate the CuTe template machinery.
- The h100 kernels add SM90-class warp-specialized producer/consumer structure (`kProducerWarpId`, `is_producer = warp_id==0`, `cp_async_fence`, `cooperative_groups`) -- Hopper-tuned scheduling that is meaningless on CDNA/RDNA.
- The remaining ~134 kernels use bare `nvcuda::wmma` (`wmma::fragment`, `wmma::mma_sync`) with hardcoded `threadIdx.x/32` warp arithmetic and 16x16x16 NVIDIA-tensor-core tile shapes. `nvcuda::wmma` does have a ROCm analogue (rocWMMA), but: (a) these are a tiny minority and only one or two per (M,N,K) point, (b) they are themselves wave32/16x16x16-NVIDIA-shaped and would need a wave64/MFMA rewrite, and (c) they are NOT the project's product -- they are not the kernels the paper's "beats cuBLAS" claim rests on. Porting only these would not constitute a meaningful CUDA-L2 port and the per-(M,N,K) coverage would collapse.

There is no single hand-written portable GEMM that is "the kernel"; the value of this repo IS the CuTe corpus and the RL pipeline that produced it (which is not in this repo). A correctness-first mechanical HIP port is impossible (CuTe will not compile under hipcc), and an AMD-native rewrite would mean re-deriving thousands of per-shape kernels with Composable Kernel / rocWMMA / MFMA -- effectively re-running the authors' RL search on AMD hardware, which is a research project, not a MOAT port. This is squarely the CPM.cu / SpargeAttn "no portable validatable slice" class the planner brief calls out.

## CUDA surface inventory
- Tensor-core MMA: CuTe `SM80_16x8x16_F16F16F16F16_TN` / `SM80_16x8x16_F32F16F16F32_TN` MMA atoms (3560 files); bare `nvcuda::wmma` 16x16x16 (134 files). AMD map: rocWMMA / MFMA (`__builtin_amdgcn_mfma_*`) -- a rewrite, not a translation. CuTe MMA atoms: no map.
- Async copy: CuTe `SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>`, `cp_async_fence`/`cp_async_wait` (Ampere cp.async). AMD map: no direct cp.async; would use `__builtin_amdgcn` global-load-to-LDS or plain vectorized loads -- rewrite.
- Shared-to-register load: `SM75_U32x4_LDSM_N` (ldmatrix). AMD map: none; manual LDS loads -- rewrite.
- Hopper warp specialization: producer/consumer split, `cooperative_groups`, named-barrier-style scheduling (h100). AMD map: not applicable.
- Libraries: cuBLAS (`cublas_v2.h`), cuBLASLt (`cublasLt.h`) in the cublas/ BASELINES only (not the product). hipBLAS exists; hipBLASLt has partial/asymmetric coverage of the cuBLASLt algo-search API the auto-tuning baseline uses (`find_best_algo_*`). These are the comparison harness, not the kernels under port.
- Datatypes: `cuda_fp16.h`, `cuda_bf16.h`, `cuda_fp8.h` -> hip_fp16/hip_bf16/hip_fp8 (the only truly mechanical part).
- No cuFFT/cuRAND/cuSPARSE/Thrust/CUB; no textures/surfaces; no pinned/managed memory; streams only implicitly via torch.

## Risk list (academic, since BLOCKED -- recorded for completeness)
- warpSize 32 vs 64: the wmma kernels hardcode `/32` and `%32` warp arithmetic and 16x16x16 NVIDIA tile shapes -- wave64/MFMA-shape (16x16x16 / 32x32x8) rewrite needed; multi-arch (gfx90a;gfx1100) would require wave-agnostic warpSize handling that does not exist in these kernels.
- CuTe template instantiation under hipcc: will not compile at all (no `cute/tensor.hpp` on ROCm).
- hipBLASLt algorithm-enumeration parity for the auto-tuning baseline.
- No serialized-format / texture-pitch / rule-of-five / OOB-neighbor exposure (no such surface here).

## File-by-file change list
N/A -- no port. (Were one ever attempted: a from-scratch CK/rocWMMA HGEMM library plus an AMD RL search, which is out of MOAT scope.)

## Build commands (gfx90a)
N/A. The upstream build (`./eval_one_file.sh ... --device_type a100`) requires CUTLASS_DIR + an NVIDIA GPU and cannot be configured for ROCm without rewriting the kernels.

## Test plan
- Real GPU tests: none portable. Upstream "tests" are correctness+speedup benchmarks (`zero_one_correctness_check.py`, `benchmarking_*.py`, `eval_one_file.sh`) that JIT-compile a CuTe kernel and compare against torch.matmul / cuBLAS on an NVIDIA card. None of this builds on ROCm.
- Non-GPU regression set: none (no CPU test suite).
- There is no GPU-validatable slice on AMD.

## Decision / disposition
Recommend BLOCKED for linux-gfx90a with reason: CUTLASS/CuTe + Ampere/Hopper-PTX (cp.async/ldmatrix/SM80-SM90 MMA atoms) core; 3560/3736 kernels are pure CuTe and CUTLASS does not port to ROCm; the ~134 nvcuda::wmma kernels are a non-representative minority requiring an MFMA/wave64 rewrite and do not constitute the project. No portable, GPU-validatable slice exists. A real "AMD CUDA-L2" would be an independent CK/rocWMMA-based RL kernel-search effort, out of MOAT's port scope.
Suggested triage skip once confirmed by jeff: `python3 utils/triage.py skip deepreinforce-ai/CUDA-L2 --reason cant-port --note "CUTLASS/CuTe + Ampere cp.async/ldmatrix + Hopper warp-spec; 3560/3736 kernels pure CuTe, no ROCm path; nvcuda::wmma minority is non-representative and needs MFMA rewrite; no validatable slice"`.

## Open questions
- Does jeff want the ~134 nvcuda::wmma kernels carved out as a separate, tiny rocWMMA proof-of-concept? They are wave32/16x16x16 NVIDIA-shaped and not the paper's product, so I recommend against it; flagging for an explicit call.
