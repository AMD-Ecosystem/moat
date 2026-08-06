# mHC.cu -- ROCm/HIP Port Plan (lead: linux-gfx90a / MI250X, ROCm 7.2.1)

## Project
- Name: mHC.cu
- Upstream: https://github.com/AndreSlavescu/mHC.cu (AndreSlavescu)
- Default branch: main; clone HEAD a426939 ("fix build")
- What it is: an unofficial CUDA implementation of mHC (Manifold-Constrained Hyper-Connections, DeepSeek-AI, arXiv 2512.24880). A focused fused-kernel library (despite the `.cu` repo name it is a multi-kernel project, not a single kernel): RMSNorm fwd/bwd, Sinkhorn-Knopp doubly-stochastic projection fwd/bwd (single + batched), "stream ops" (aggregate / distribute-mix-add) fwd/bwd in static and dynamic-H variants, and a fused RMSNorm+matmul path. Exposed both as a PyTorch C++/CUDA extension (`mhc_cuda`) consumed by a small Transformer trainer, and as a standalone CMake suite of C++/CUDA correctness tests and benchmarks.

## Existing AMD support
- None. No HIP/ROCm/gfx references anywhere in the tree (Makefile, CMake, sources, Modal harness all NVIDIA-only: `nvidia/cuda:12.8` image, `nvcc`, `-gencode sm_*`, README advertises H100/B200). This is a fresh CUDA-to-HIP port.
- Decision: PROCEED with a correctness-first mechanical HIP port. Value is clear (no AMD path exists).
- Port-vs-rewrite: the hot kernels are hand-written CUDA (cooperative-groups reductions, vectorized bf16 loads, shared-memory tiling) plus ONE cuBLAS/cuBLASLt bf16 GEMM in the dynamic-H projection. There is NO CUTLASS/CuTe and NO Hopper wgmma/MMA/tcgen PTX. The GEMM is a library call, not a bespoke tensor-core kernel. So a mechanical HIP translation is appropriate and will be correct; the cuBLASLt GEMM maps to hipBLASLt (rocBLAS fallback path already present via `cublasGemmEx`). An AMD-native MFMA/CK rewrite is NOT warranted for a first pass -- defer any perf-tuning to a later optional pass and note it. The only NVIDIA-arch-specific feature is PDL (programmatic dependent launch, sm_90+), which is an optional optimization gated by `MHC_ENABLE_PDL` and must simply be left OFF on AMD (see Risks).

## Build classification: torch-extension (with a secondary standalone CMake test/bench harness)
Evidence:
- `setup.py` lines 47-78: `from torch.utils.cpp_extension import BuildExtension, CUDAExtension`; builds `CUDAExtension(name="mhc_cuda", sources=["src/python/bindings.cu"], libraries=["cublas","cublasLt"])`, `cmdclass={"build_ext": BuildExtension}`. This is the primary deliverable and the path the Python tests/trainer use. -> Strategy B.
- `pyproject.toml` build-system requires `torch>=2.0.0`; `Makefile` `install:` target runs `pip install -e .`.
- Secondary: `src/csrc/CMakeLists.txt` is a pure-CMake project (`project(mHC LANGUAGES CXX CUDA)`, `find_package(CUDAToolkit)`, `CUDA::cudart CUDA::cublasLt`) that compiles the 8 `tests/test_*.cu` and 9 `benchmarks/bench_*.cu` standalone (NO Torch). `Makefile` `build:`/`test:` drive it via `cmake -B build -S src/csrc -DCMAKE_CUDA_ARCHITECTURES=...`.
- ext_type set to `torch-extension` (the product). The CMake harness is the cleaner GPU-validation slice (self-contained CPU references, no Torch/ROCm-torch dependency); it gets a Strategy-A treatment.

## Port strategy
Hybrid, because the two build systems share the same `src/csrc/{kernels,include}` headers:
- Strategy B for the extension: build `bindings.cu` against a ROCm PyTorch. Torch's `BuildExtension` auto-runs `torch.utils.hipify` on the extension's `.cu` and links `amdhip64`/`c10_hip`/`torch_hip`. Keep sources in CUDA spelling; hipify does the `cudaXxx->hipXxx` rename. `setup.py` needs ROCm branches: gate the `-gencode`/`--use_fast_math`/`-U__CUDA_NO_*` nvcc args and the `cublas`/`cublasLt` libs on `torch.version.hip`, emitting `--offload-arch=gfx*` and `hipblas`/`hipblaslt` instead; skip the `MHC_ENABLE_PDL` define on ROCm.
- Strategy A for the CMake harness: add `option(USE_HIP)`; under it `enable_language(HIP)`, default `CMAKE_HIP_ARCHITECTURES` to `gfx90a` only when unset (never a literal that overrides `-DCMAKE_HIP_ARCHITECTURES`), mark the `tests/*.cu` + `benchmarks/*.cu` `LANGUAGE HIP`, link `hip::host` + `roc::hipblaslt`. Force `MHC_ENABLE_PDL` OFF on HIP.
- The shared kernel headers are edited ONCE, guarded by `defined(__HIP_PLATFORM_AMD__)` / `USE_ROCM`, so both build systems pick up the same fixes. hipify handles symbol renames inside the extension; for the CMake harness the same `cudaXxx` spellings compile directly under `hipcc -x hip` because ROCm ships `<cuda_runtime.h>`-free HIP and the headers already use CUDA names -- so the CMake path needs the symbols available: prefer a single tiny `cuda_to_hip.h` compat shim force-included on the HIP CMake TUs (and a no-op on the Torch path where hipify already renamed). Keep the shim minimal: it only needs to map the handful of symbols hipify would have (`cudaXxx` runtime, the `nv_bfloat16`/`__bfloat1622float2` type+intrinsic names, `cublas*`->`hipblas*`/`hipblaslt`), plus define-away PDL and provide the profiler intrinsic shims below.

## CUDA surface inventory
Kernels / device code (all hand-written, fp32 compute with bf16 I/O):
- rmsnorm.cuh: RMSNorm fwd (`rmsnorm_forward_with_rms`), bwd, `compute_rms`; cg block reductions; vectorized bf16 weight load via `nv_bfloat162` reinterpret + `__bfloat1622float2`.
- sinkhorn_knopp.cuh: several Sinkhorn-Knopp variants (warp-per-row 32x32, warp-optimized, tiled, batched fwd, single/batched bwd); cg `tiled_partition<32>` row reductions; `__frcp_rn`; `N_MAX=32` cap.
- stream_ops.cuh: aggregate (bf16 fused sigmoid / dynamic), distribute-mix-add (static/dynamic), and their backward kernels; `MAX_N` static shared arrays; cg warp + shared cross-subgroup block reductions; vectorized float4 loads (`C4 = C/4`).
- fused_rmsnorm_matmul.cuh: fused RMSNorm+matmul; includes `<cublasLt.h>` and a `FusedRMSNormMatmulPDL` helper; cg reductions.
- utils.cuh: `fused_h_activations`, `fused_h_backward_pre/post`, `fused_rms_correction`; cg block reduction with `threadIdx.x/32` warp partition + `__syncthreads`; **inline NVPTX asm**: `globaltimer()` (`mov.u64 %0,%%globaltimer`) and `get_smid()` (`mov.u32 %0,%%smid`); `DeviceProfiler` struct.
- profiling.cuh: `MHC_PROFILE_*` macros, all `if constexpr (DO_PROFILE)` with default `DO_PROFILE=false`.
- bindings.cu: the Torch glue. `CublasLtCache` (cuBLASLt row-major bf16 A * bf16 B^T -> fp32 C, heuristic-selected, with a `cublasGemmEx` `CUBLAS_GEMM_DEFAULT_TENSOR_OP` fallback) and `MHCWorkspace` (arena + a side `cudaStream_t`/`cudaEvent_t` for Sinkhorn/main overlap).
Warp intrinsics / width: NO `__shfl*`/`__ballot`/`__activemask`/`__popc`/`warpSize`. All warp work is via `cg::thread_block_tile<32>` + `cg::reduce` (logical 32-lane subgroups) and integer `threadIdx.x/32`, `%32`, `BLOCK_SIZE/32` partitioning -- all keyed on the LITERAL 32, never on physical warpSize.
Libraries: cuBLAS + cuBLASLt (bf16 GEMM) -> hipBLAS + hipBLASLt. No cuFFT/cuRAND/cuSPARSE/Thrust/CUB.
Streams/events: one side stream + event for fwd overlap (`cudaStreamCreate`/`cudaEventCreate`/`cudaEventRecord`/`cudaStreamWaitEvent`) -> 1:1 hip. No textures/surfaces, no managed/pinned memory, no cooperative-grid launch.
PDL: `cudaLaunchAttributeProgrammaticStreamSerialization` + `cudaTriggerProgrammaticLaunchCompletion()` at ~10 launch sites and 3 device-side triggers, all under `#ifdef MHC_ENABLE_PDL`. sm_90+ only.

## Risk list
1. PDL (Hopper programmatic dependent launch) -- there is NO HIP equivalent of `cudaLaunchAttributeProgrammaticStreamSerialization` / `cudaTriggerProgrammaticLaunchCompletion()`. FIX: never define `MHC_ENABLE_PDL` on ROCm (drop it from the ROCm `setup.py` defines and force the CMake `option(MHC_ENABLE_PDL)` OFF when `USE_HIP`). PDL is purely a latency optimization; the non-PDL code path is the functional baseline and is selected by `#ifdef`. Confirm every `#ifdef MHC_ENABLE_PDL` has a complete `#else`/fall-through (the launches use `cudaLaunchKernelEx` under PDL vs plain `<<<>>>` otherwise -- verify the non-PDL branch builds standalone).
2. Inline NVPTX asm (`%%globaltimer`, `%%smid`) in utils.cuh -- will not compile under hipcc. Even though the profiler is `if constexpr(false)` by default, the `__device__` functions `globaltimer()`/`get_smid()` are still PARSED and CODEGEN'd (they are `__forceinline__` free functions, not templates). FIX: `#if defined(__HIP_PLATFORM_AMD__)` branch -> `globaltimer()` returns `__builtin_amdgcn_s_memrealtime()` (s_memrealtime, a free-running clock; ns-scale differs from CUDA globaltimer but profiling output is not validated), `get_smid()` returns the AMD hardware ID via `__builtin_amdgcn_s_getreg(...)` (HW_ID, bits for CU/SE) or simply 0 if unused -- profiler is off in all tests/benches by default.
3. cuBLASLt row-major bf16 path (bindings.cu CublasLtCache) -> hipBLASLt. hipBLASLt's heuristic/algo API mirrors cuBLASLt but watch: `CUBLASLT_ORDER_ROW`/layout-order attribute support, `CUBLAS_COMPUTE_32F` vs `HIPBLASLT_COMPUTE_F32`, and whether the heuristic returns >0 results on gfx90a for these (B x nC) by (out_dim x nC)^T shapes. The code ALREADY has a `use_lt=false` fallback to `cublasGemmEx` (-> `hipblasGemmEx`, rocBLAS-backed) when no Lt heuristic is found -- so a hipBLASLt miss degrades gracefully to hipBLAS. Verify the fallback compiles and that the bf16 A/B + f32 C + `CUBLAS_GEMM_DEFAULT_TENSOR_OP` (-> `HIPBLAS_GEMM_DEFAULT`) maps cleanly.
4. Warp-size / wave64 -- LOW risk but must be confirmed. All "warp" ops are width-32 LOGICAL: `cg::tiled_partition<32>` + `cg::reduce`, and block reductions partition by literal `/32`,`%32`,`BLOCK_SIZE/32` with `__syncthreads()` between the cross-subgroup shared write and read (utils.cuh:225-231, stream_ops.cuh:918-926, rmsnorm/fused tails). Per the guide these width-32 logical ops are arch-agnostic and correct on wave64. There is NO hardcoded physical-warp assumption, no `__shfl(...,32)` mask, no `s_warp_sums` undersizing (sized [8] for <=256 threads = <=8 subgroups of 32; max block 256). RISK to verify on the FOLLOWER (wave32) only: HIP's `cg::tiled_partition<32>` on a 32-wide wavefront and the `__syncthreads`-bounded cross-subgroup reductions -- there is a barrier at every cross-logical-warp shared boundary (icicle lesson), so this should hold; re-confirm on gfx1100.
5. `cg::reduce(tile<32>, plus)` correctness under rocPRIM-backed cooperative groups -- HIP cooperative_groups provides `cg::reduce`; confirm the `<cooperative_groups/reduce.h>` include resolves and `cg::plus<float>` is available on ROCm 7.2.1 (it is, but it is the one cg feature most likely to need a header tweak).
6. bf16 type + intrinsic names -- `nv_bfloat16`/`nv_bfloat162`/`__bfloat1622float2` (mhc_types.h `using floatX = nv_bfloat16`). hipify maps these to `__hip_bfloat16`/`hip_bfloat162`/`__bfloat1622float2`; for the CMake (non-hipify) path the compat shim must alias them (or include `<hip/hip_bf16.h>` and `#define nv_bfloat16 __hip_bfloat16` etc.). Verify the `reinterpret_cast<nv_bfloat162*>` vectorized loads keep 4-byte alignment semantics on HIP.
7. `--use_fast_math` (setup.py) vs `__frcp_rn`/`__frcp_rn`-style rounded reciprocals + the Sinkhorn `__frcp_rn`. On HIP, fast-math + `-ffp-contract=fast` (clang default) can drift fp32 ~1 ULP from CUDA. The C++ tests compare against an fp32 CPU reference with a tolerance (allclose-style), and the bf16 I/O already dominates error, so bit-exactness is NOT required -- but if any test uses a tight atol, pin `-ffp-contract=on` (CV-CUDA lesson) and keep fast-math parity. Decide per-test tolerance during validation; do not pre-emptively change math.
8. `cudaTriggerProgrammaticLaunchCompletion()` appears device-side in sinkhorn/fused even outside obvious guards (sinkhorn_knopp.cuh:289,622; fused_rmsnorm_matmul.cuh:402) -- confirm each is inside an `#ifdef MHC_ENABLE_PDL` region; if any is NOT guarded, add the guard (it is a no-op intrinsic absent on HIP).

## File-by-file change list
- `setup.py`: add a `torch.version.hip` branch -- replace `get_cuda_arch_flags()`/`get_extra_defines()` outputs (drop `-gencode`, drop `MHC_ENABLE_PDL`, keep `-O3`; `--use_fast_math` -> `-ffast-math` is implied, add `--offload-arch` from env `HIP_ARCH`/`PYTORCH_ROCM_ARCH` or default gfx90a) and swap `libraries=["cublas","cublasLt"]` -> `["hipblas","hipblaslt"]`. Keep CUDA branch byte-identical.
- NEW `src/csrc/include/cuda_to_hip.h`: the single compat shim. `#if defined(__HIP_PLATFORM_AMD__)` include `<hip/hip_runtime.h>`, `<hip/hip_bf16.h>`, `<hipblas/hipblas.h>`, `<hipblaslt/hipblaslt.h>`; alias the `cudaXxx` runtime symbols, the cublas/cublasLt symbols + enums used by bindings.cu/fused_rmsnorm_matmul.cuh, the bf16 type/intrinsic names; `#undef MHC_ENABLE_PDL`. NVIDIA: `#include <cuda_runtime.h>` only (no-op). Force-include it on the HIP CMake TUs via `CMAKE_HIP_FLAGS -include`. (For the Torch path hipify already renames; include the shim but make its aliases idempotent / `#ifndef`-guarded so it does not double-define.)
- `src/csrc/include/utils.cuh`: HIP branch for `globaltimer()`/`get_smid()` (risk 2). No other change (the `/32` reduction is arch-safe).
- `src/csrc/include/mhc_types.h`: route `<cublasLt.h>` + `nv_bfloat16` through the compat shim under HIP; keep CUDA path unchanged.
- `src/csrc/kernels/{rmsnorm,sinkhorn_knopp,stream_ops,fused_rmsnorm_matmul}.cuh`: only if a PDL `#ifdef` is found incomplete (risk 1/8) or a bf16 intrinsic needs a guard; otherwise untouched (hipify/shim covers names). Audit `<cublasLt.h>` includes -> shim.
- `src/csrc/CMakeLists.txt`: add `option(USE_HIP OFF)`; under it `enable_language(HIP)`, default `CMAKE_HIP_ARCHITECTURES=gfx90a` only when unset, retag `tests/*.cu`+`benchmarks/*.cu` `LANGUAGE HIP`, set `HIP_ARCHITECTURES` from the cache var, link `hip::host`/`roc::hipblaslt`, and force `MHC_ENABLE_PDL OFF` when `USE_HIP`. CUDA path unchanged.
- `src/python/bindings.cu`: route the cuBLAS/cuBLASLt + bf16 includes through the shim (or rely on hipify); confirm `at::cuda::getCurrentCUDAStream()` -> works under ROCm torch (it does; ATen abstracts it). Likely no hand edits beyond the include if hipify is clean.
- DO NOT touch `runmodal.py` (NVIDIA Modal harness, out of scope) or add any GitHub Actions.

## Build commands (gfx90a)
Standalone CMake test/bench harness (primary validation slice; no Torch needed):
```
cmake -B agent_space/mhc_build -S projects/mHC.cu/src/src/csrc \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build agent_space/mhc_build -j$(nproc)
```
Multi-arch correctness build (the warp-size test, per guide):
```
cmake -B agent_space/mhc_build_fat -S .../src/csrc -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" ... ; cmake --build ...
# confirm both code objects: llvm-objdump --offloading agent_space/mhc_build_fat/test_rmsnorm
```
Torch extension (needs a ROCm PyTorch in the env):
```
cd projects/mHC.cu/src && PYTORCH_ROCM_ARCH=gfx90a pip install -e . -v
```

## Test plan
GPU-validatable slice (real correctness, on-device, no Torch dependency) -- the standalone CMake tests, each ships a CPU fp32 reference and asserts agreement:
- test_rmsnorm, test_rmsnorm_backward
- test_sinkhorn_knopp (also asserts doubly-stochastic: row/col sums -> 1)
- test_mhc_layer
- test_stream_ops, test_stream_ops_backward
- test_fused_rmsnorm_matmul, test_fused_rmsnorm_matmul_backward
Run via the Makefile loop on the HIP build dir: each `agent_space/mhc_build/test_*` must exit 0 ("All C++ tests passed"). This is the lead-platform validation gate.
Python GPU tests (require ROCm torch; secondary, exercises the extension + cuBLASLt->hipBLASLt path and the dynamic-H trainer integration):
- `pytest src/python/tests -v` (test_ops.py: sinkhorn doubly-stochastic + gradient finite, rmsnorm shape/grad; test_layer.py, test_model.py: MHCLayer fwd/bwd vs a naive PyTorch reference).
Non-GPU regression set to NOT break: the CUDA build path must stay byte-identical (all ROCm changes behind `__HIP_PLATFORM_AMD__`/`USE_HIP`/`torch.version.hip`); `make lint` (ruff + cppcheck) and `make format` (clang-format) unaffected.
Benchmarks (bench_*) are perf, not correctness -- build them but do not gate on numbers; useful to confirm hipBLASLt path runs.

## Open questions
- Does the validation host have a ROCm-built PyTorch importable in the env? If not, the Python extension tests cannot run; the standalone CMake C++ tests are then the sole (and sufficient) GPU correctness gate -- they cover every kernel with CPU references and need only ROCm + hipBLASLt, not torch. (Lead validation should not be blocked on a ROCm torch.)
- hipBLASLt heuristic coverage for the row-major bf16 (B x nC)*(out_dim x nC)^T -> f32 shapes on gfx90a: if it returns 0 candidates the code falls back to hipBLAS `hipblasGemmEx`; confirm that fallback's bf16+f32-accumulate is supported on gfx90a (expected yes via rocBLAS). Decide at porter/validator time, not now.
- Tolerance of the C++ tests under HIP fast-math + clang fp-contract: if any uses a tight atol, pin `-ffp-contract=on`; otherwise leave math at parity with the CUDA `--use_fast_math` build.

## Delta plan: follower platforms
(Placeholder -- fill on demand.) gfx1100/gfx1151 are wave32 (RDNA3/3.5). Primary follower risk is risk 4/5: the literal-32 logical-warp partitioning and `cg::tiled_partition<32>` + `__syncthreads`-bounded cross-subgroup reductions must hold when each 32-lane logical warp is its OWN wavefront (no implicit lockstep). The existing `__syncthreads()` barriers at every cross-logical-warp shared boundary should make this safe; re-validate by running the same C++ test suite on the gfx1100 host with no source change (reuse the gfx90a fork branch, build `-DCMAKE_HIP_ARCHITECTURES=gfx1100`). Only delta-port if a determinism/correctness failure appears. gfx1151: additionally confirm hipBLASLt availability on the Windows HIP SDK; if absent, the hipBLAS `hipblasGemmEx` fallback or a CPU-reference-only C++ slice still validates the kernels.
