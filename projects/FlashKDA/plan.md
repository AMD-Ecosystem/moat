# FlashKDA -- ROCm/HIP port plan (lead: linux-gfx90a, MI250X, ROCm 7.2.1)

## Project
- Name: FlashKDA (Flash Kimi Delta Attention)
- Upstream: https://github.com/MoonshotAI/FlashKDA
- Default branch: master
- Planned at upstream sha: d2ff19a6a0c82f39f796f637ebd1c36090b1268f
- One sentence: a bf16 linear-attention (Kimi Delta Attention) forward kernel implemented entirely on NVIDIA CUTLASS/CuTe, requiring SM90+ (Hopper/Blackwell), shipped as a PyTorch CUDAExtension.

## Existing AMD support
- Classification: NONE. Single upstream branch (master), zero hip/rocm/amd/gfx references anywhere in the tree (csrc/, flash_kda/, setup.py, README). README explicitly states "SM90 and above" and "CUDA 12.9 and above". No OpenCL/Vulkan/SYCL path either.
- Decision: a ROCm/HIP build would add value in principle (real kernel, AMD has none), BUT the implementation is 100% CUTLASS/CuTe Hopper-native. Per the standing MOAT rule (PORTING_GUIDE changelog 2026-05-30): "CUTLASS does NOT port to ROCm and never will: do not attempt a CUTLASS->ROCm port or a CUTLASS->ROCm shim. Reimplement against Composable Kernel (CK), preferring ck_tile." There is no mechanical-port first step available here -- the only kernel sources do not compile under hipcc at all.
- This is therefore a reimplement-not-port project. See "Port strategy" for the recommendation and the explicit halt.

## Build classification
- torch-extension (Strategy B family). Evidence: setup.py line 4 imports `from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME`; lines 55-86 declare a single `CUDAExtension(name='flash_kda_C', sources=['csrc/flash_kda.cpp','csrc/smxx/fwd_launch.cu'], ...)` with `cmdclass={"build_ext": BuildExtension}`. No CMake. There is a `config.yaml`/`.clangd.template` but those are clangd IntelliSense config, not a build system.
- ext_type set to `torch-extension` in upstream.json/status.json. NOTE: the normal Strategy-B consequence (torch auto-hipifies the .cu at build) does NOT rescue this project, because torch hipify cannot translate CUTLASS/CuTe (see strategy).

## Port strategy
- Nominal: Strategy B (pytorch extension; torch hipifies .cu on a ROCm torch). REJECTED as sufficient: the extension's only device TU `csrc/smxx/fwd_launch.cu` and its headers pull in `cute/tensor.hpp`, `cute/arch/cluster_sm90.hpp`, `cute/arch/copy_sm90_tma.hpp`, `cute/arch/mma_sm80.hpp`, `cutlass/cluster_launch.hpp`, `cutlass/arch/barrier.h`, `cutlass/pipeline/sm90_pipeline.hpp`. hipify rewrites symbol spellings; it does not and cannot produce a CuTe/CUTLASS that compiles for AMD. CUTLASS has no ROCm backend.
- Actual required strategy: AMD-native REIMPLEMENTATION of the KDA forward kernel using Composable Kernel (ck_tile preferred). Keep the Python package, the C++ entry (`csrc/flash_kda.cpp`, the `flash_kda.fwd` ATen-tensor signature, dispatch, argument checks) and the torch_ref test harness; replace the entire `csrc/smxx/*` CUTLASS kernel + launcher with a CK/ck_tile implementation behind the same `run_fwd(...)`-style launch entry that `flash_kda.cpp` calls. Build it as a `CUDAExtension` on ROCm torch (the .cpp host glue hipifies trivially) but with the device kernel written natively for CDNA, not hipified from CUTLASS.
- Rationale: this is exactly the perf-critical-attention case the guide calls out (attention/GEMM, CUTLASS/CuTe/Hopper-tuned). A mechanical port is not even an available correctness-first step here because the source will not compile under hipcc. The work is: understand the KDA chunked-recurrence algorithm from the two kernels + torch_ref, then express its tiling/MMA/epilogue with CK on MFMA. This is a multi-day kernel-engineering effort, not a translation pass.

## CUDA surface inventory
Two device translation units (compiled), built on CuTe/CUTLASS:
- `csrc/smxx/fwd_launch.cu` (231 lines): host launcher; builds TMA descriptors (TMAQKLayout, TMAGLayout, TMAVOLayout, TMAStateSmemLayout, ...), configures cluster/persistent launch, sets dynamic shared memory, dispatches kernel1 + kernel2.
- `csrc/smxx/fwd_kernel1.cuh` (570) and `fwd_kernel2.cuh` (834): the two-stage KDA forward. Warp-specialized (WarpRole MMA / LOAD_QKG / STORE / NonParticipant), persistent, cluster-launched.
- `csrc/smxx/utils.cuh` (381): CuTe helpers, `cooperative_gemm` over `make_tiled_mma` with `SM75_U32x4_LDSM_N` / `SM90_U32x4_STSM_N`, `elect_one_sync`, pipeline factories.
- `csrc/fwd.h`, `csrc/flash_kda.cpp` (229): host-side ATen glue + arg validation; this part is portable.

Hopper-only / CUTLASS-only features in use (each has NO HIP/ROCm equivalent; each forces reimplementation, not translation):
- TMA (Tensor Memory Accelerator): `cute/arch/copy_sm90_tma.hpp`, `get_tma_tensor`, `make_load_pipeline` -> `PipelineTmaAsync`, `ClusterTransactionBarrier::arrive_and_expect_tx`, `kTmaTransactionBytes`, `fence_view_async_shared`. CDNA has no TMA; bulk async copy must be reimplemented with `__builtin_amdgcn` global-load-to-LDS / plain async DMA patterns or simple loads.
- wgmma / GMMA: `GMMA::Layout_K_INTER_Atom`, `GMMA::Layout_MN_INTER_Atom`, `cooperative_gemm` lowering to SM90 warpgroup MMA. CDNA uses MFMA (`v_mfma_*`); replace with CK's MFMA pipelines (rocWMMA or ck_tile gemm).
- Thread-block clusters / distributed shared memory: `cute/arch/cluster_sm90.hpp`, `cutlass/cluster_launch.hpp`. No cluster concept on CDNA; collapse to single-CTA tiling.
- Warp-specialized async pipeline: `cutlass/pipeline/sm90_pipeline.hpp`, `cutlass::PipelineState`, `PipelineTmaAsync`, `PipelineAsync`, `make_producer_start_state`, `cutlass::arch::NamedBarrier`. Reimplement producer/consumer staging with LDS double-buffering + `__syncthreads()`/named-barrier equivalents (`__builtin_amdgcn_s_barrier`) on CDNA.
- `cute::elect_one_sync()` leader election (PTX `elect.sync`). Replace with a lane-predicate computed from `__lane_id()` / wavefront ballot (wave64-aware).
- mbarrier / async-transaction barriers: `cutlass/arch/barrier.h`. No mbarrier on CDNA; use LDS atomics / `s_barrier`.
- Library deps: only CUTLASS/CuTe (header-only submodule from github.com/NVIDIA/cutlass). NO cuBLAS/cuBLASLt, NO cub/thrust, NO curand/cufft. So the dependency story is simple once the kernel is rewritten -- the only "dependency" is the GEMM/MMA machinery, which becomes CK.
- No textures/surfaces, no managed/pinned memory, no streams/events beyond the torch-provided current stream.

## Risk list
- PRIMARY RISK (gates the whole project): CK/ck_tile reimplementation effort. CK is not a 1:1 API for CuTe; per guide must first fully understand the KDA tiling/epilogue/fusion (gate activation, qk l2norm, beta sigmoid, A_log/dt_bias, lower_bound clamp, chunked delta-rule recurrence with carried state) then build the equivalent from CK examples. High effort; correctness bar is "exact match vs torch_ref / flash-linear-attention" per tests.
- wave64 fault-class exposure (will bite the CK rewrite, not the dead CUTLASS code):
  - `kWarpSize` is referenced in fwd_kernel2.cuh (warp_id = threadIdx.x / kWarpSize; warp-role partition). The CUTLASS code assumes 32; the CK rewrite must use a wave-size-correct constant (64 on gfx90a / GFX9, 32 on RDNA). Use the C10_WARP_SIZE pattern.
  - `elect_one_sync` / any ballot/shfl in the rewrite must use 64-bit lane masks on gfx90a (`unsigned long long`, `__ballot(...)` returns 64-bit).
  - Warp-specialization thread budgeting (kComputeThreads / kWarpSize, +1 LOAD warp, +1 STORE warp) changes arithmetic under wave64 -- a CK design should re-derive thread layout from the AMD wave size, not inherit the 32-lane partition.
  - No cub/hipcub block collectives in the current code, so the TempStorage-reuse race class does not apply unless the CK rewrite introduces hipCUB.
- Numeric exactness: tests claim "exact match against the torch reference". bf16 MMA accumulation order on MFMA differs from wgmma; an exact bitwise match is unlikely. Expect to validate within a tolerance (allclose) against torch_ref rather than bit-exact; confirm with the test author intent (test_fwd.py uses torch.testing / allclose -- verify thresholds during bringup). `--use_fast_math` is set on the CUDA side; the CK build should decide fp-contract/fast-math to balance speed vs the reference tolerance.
- `K = V = 128` only (README). The rewrite can hardcode head-dim 128 initially, matching upstream's own restriction.
- Varlen path (cu_seqlens, B==1) and stateful path (initial_state/final_state bf16 or fp32, transpose_state_layout) add kernel variants -- the recurrence carries state across chunks. Plan to bring up the dense non-varlen stateless path first, then add state and varlen.

## File-by-file change list (reimplement scope)
- KEEP (host glue, hipifies cleanly under torch on ROCm):
  - `csrc/flash_kda.cpp` -- ATen tensor checks + dispatch into the launcher. Minor: any SM90 capability assertion to relax/redirect; bf16 type unchanged.
  - `csrc/fwd.h` -- signature header; `<cutlass/bfloat16.h>` include to be replaced with a CK/torch bf16 type.
  - `flash_kda/__init__.py`, `tests/*` -- unchanged (Python API + reference are the contract the rewrite must satisfy).
- REPLACE ENTIRELY (CUTLASS/CuTe device code):
  - `csrc/smxx/fwd_launch.cu` -> CK/ck_tile launcher (CDNA tiling, LDS staging, MFMA gemm, kernel grid; no TMA/cluster).
  - `csrc/smxx/fwd_kernel1.cuh`, `csrc/smxx/fwd_kernel2.cuh` -> CK/ck_tile KDA forward (two stages or fused) on MFMA.
  - `csrc/smxx/utils.cuh` -> CK-based gemm/copy helpers, wave-size-aware leader election and barriers.
- setup.py: add a ROCm branch -- when `torch.version.hip`, drop the `cutlass/` includes and the `-gencode` arch flags, point includes at the CK headers (/opt/rocm/include for installed CK, or a CK submodule), set `--offload-arch=gfx90a` via `os.environ['PYTORCH_ROCM_ARCH']` or `extra_compile_args` for ROCm, and remove the `git submodule update --init cutlass` requirement on the HIP path. Keep the CUDA path byte-for-byte for upstream.
- New: a CK dependency. Composable Kernel ships with ROCm (/opt/rocm) and/or as a submodule. Decide ck_tile (preferred) vs classic CK at porter time based on what the KDA gemm shapes map to most cleanly.

## Build commands (gfx90a)
ROCm torch required. The extension build is driven by setup.py, not raw cmake.
- Build (after the CK rewrite lands):
  - `export PYTORCH_ROCM_ARCH=gfx90a`
  - `pip install -v --no-build-isolation -e .` from the repo root (uses BuildExtension on a ROCm torch; the rewritten device sources compile with hipcc against CK headers).
- A CPU-only compile smoketest is NOT meaningful for a GPU kernel and is not a gate.
- Do NOT pass `FLASH_KDA_CUDA_ARCHS` on the ROCm path (it is CUDA -gencode plumbing); arch is selected via `PYTORCH_ROCM_ARCH`/`--offload-arch`.

## Test plan
- GPU correctness (the real gate): `python tests/test_fwd.py` (and the fuller `tests/test_fwd_full.py` / `tests/run_test_full.sh`). These compare `flash_kda.fwd` against the torch reference in `tests/torch_ref.py` and against `flash-linear-attention`'s `chunk_kda`. Requires `flash-linear-attention>=0.5.0` + matplotlib (see tests/test.sh). On AMD, verify whether FLA's `chunk_kda` reference path (Triton) runs on ROCm; if not, compare directly against `tests/torch_ref.py` (pure torch) which is backend-agnostic. Validate with the test's tolerance (allclose), not bitwise -- MFMA accumulation will not bit-match wgmma.
- Cover the variants the API exposes: stateless / with initial_state (bf16 and fp32) / final_state output / varlen (cu_seqlens) / the in-kernel gate, qk-l2norm, beta-sigmoid options.
- Non-GPU regression set: none of consequence -- there is no CPU compute path to regress. The Python package import and arg-validation in flash_kda.cpp are the only non-kernel surface; keep them intact.
- Benchmarks (`benchmarks/bench_fwd.py`) are perf, not correctness; useful to size the CK rewrite's gap vs upstream but not a pass/fail gate.

## Open questions
1. Effort vs value: this is a from-scratch CK/ck_tile attention kernel (KDA chunked delta-rule with gating + carried state), one of the harder MOAT items. Is FlashKDA in scope for a full AMD-native reimplementation now, or should it be deferred behind a "CK reimplementation backlog" alongside the raft CUTLASS items and OptiX-gated ports? Recommend confirming with jeff before a porter spends multi-day effort.
2. Reference availability on ROCm: does `flash-linear-attention`'s Triton `chunk_kda` run on ROCm for the cross-check, or do we validate solely against `tests/torch_ref.py`?
3. Numeric tolerance: upstream advertises "exact match". Confirm the acceptable allclose tolerance for an MFMA implementation (bf16 accumulation order differs), so validation is not held to an unachievable bit-exact bar.
4. ck_tile vs classic CK for the KDA gemm shapes (head-dim 128, chunked) -- a porter-time decision; ck_tile preferred per guide.

## Recommendation
Do NOT mark skip/already-supported (AMD genuinely has nothing here). Mark planned with the explicit understanding that the porter stage is a CK/ck_tile reimplementation, not a hipify. Flag open question 1 to jeff at the porter handoff, since the effort profile matches the deferred-CK class (raft/cuvs/cuml) and OptiX-gated reimplementations already on the backlog.
