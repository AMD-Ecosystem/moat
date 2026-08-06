# CPM.cu -- ROCm/HIP port plan (lead: linux-gfx90a)

## Project
- Name: CPM.cu
- Upstream: https://github.com/OpenBMB/CPM.cu
- Default branch: main
- HEAD at clone: 23aa7b7fefc537113166691cec63d5baa5209ebe
- What it is: a standalone CUDA LLM-inference engine for OpenBMB's MiniCPM4 family. Headline features are a sparse attention path, EAGLE / FR-Spec speculative decoding (tree-verified), and W4A16 quantized inference via Marlin GPTQ/AWQ kernels. It is a Python package `cpmcu` wrapping one native pybind11 extension `cpmcu.C` built from the CUDA sources under `src/`.

## Existing AMD support
- None. Zero `hip`/`rocm`/`amd`/`USE_ROCM`/`gfx`/`__HIP__` tokens anywhere in the tree (code, build, docs, docker). Docker images are `cuda12.6`/`cuda12.8` only; `setup.py` validates `CPMCU_CUDA_ARCH` against compute-capability 80-120 (Ampere..Blackwell) and the kernels are written for `sm_80`. Pure NVIDIA.
- Decision: a HIP port would add value in principle (no AMD path exists), BUT the two performance-critical kernel families are NVIDIA-architecture-locked and cannot be mechanically hipified (see Port strategy). This is a port-vs-rewrite decision, and the honest answer is a substantial multi-week AMD-native reimplementation, not a port. Recommended disposition: see "Recommendation / disposition" at the bottom. Default action taken: advance to `planned` with the reimplementation scope documented so a porter can decide whether to take the rewrite; if MOAT wants only mechanical ports, this should be skipped with reason `cant-port` (CUTLASS/CuTe + raw sm80 PTX).

## Build classification
- torch-extension (Strategy B family, but see the strategy caveat).
- Evidence:
  - `pyproject.toml` build-system requires `torch>=2.0.0` and `ninja>=1.10.0`.
  - `setup.py` imports `torch.utils.cpp_extension` (`CUDAExtension`, `BuildExtension`) and builds a single `CUDAExtension(name='cpmcu.C', ...)` (setup.py lines 384, 422, 483-502). No CMake, no Makefile.
  - The extension compiles `src/entry.cu`, `src/utils.cu`, `src/signal_handler.cu`, `src/perf.cu`, all of `src/qgemm/gptq_marlin/*.cu`, and the selected `src/flash_attn/src/*hdim{64,128}*.cu`; links `cublas` + `dl`.
- ext_type set to `torch-extension`.

## Port strategy
Strategy B (hipify-at-build-time) is the correct *mechanical* mechanism for a torch CUDAExtension, and the portable parts of the tree will go through it cleanly. BUT Strategy B alone is insufficient here: two kernel families do not survive hipify and must be reimplemented AMD-native. This is a "mechanical port for the easy 60%, AMD-native rewrite for the hot 40%" plan, and the rewrite is the long pole.

The single `cpmcu.C` extension is monolithic: `entry.cu` `#include`s every model variant, and the base (non-quantized) attention path itself (`src/model/attn.cuh:4`) hard-includes `src/flash_attn/flash_api.hpp` and calls `mha_fwd_kvcache`. So the FlashAttention core is a MANDATORY compile + link + runtime dependency for the entire module -- there is no model configuration that links without it. Likewise the test suite downloads and runs the Marlin-quantized MiniCPM4-8B model, so the Marlin GEMM is exercised at runtime by validation.

### Kernel family 1: FlashAttention-2 on CUTLASS/CuTe (reimplement, do not port)
- `src/flash_attn/` is a vendored fork of Dao-AILab flash-attention v2.6.3 (`README.md` line 281). It is built directly on the bundled NVIDIA CUTLASS submodule (`.gitmodules` -> NVIDIA/cutlass; `src/cutlass/include` on the include path).
- `flash_fwd_kernel.h` includes `<cute/tensor.hpp>`, `<cutlass/cutlass.h>`, `<cutlass/array.h>`, `<cutlass/numeric_types.h>`. `kernel_traits.h` instantiates `MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>`, `Copy_Atom<SM75_U32x4_LDSM_N>`, `SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>`, `TiledMMA<...>` -- CuTe sm75/sm80 MMA and cp.async atoms.
- PORTING_GUIDE is explicit: CUTLASS does NOT port to ROCm and never will; do not attempt a CUTLASS->ROCm port or shim. The fix is to reimplement the kernel against Composable Kernel (prefer ck_tile) -- understand the FA2 tiling/softmax/epilogue/KV-cache-append/tree-mask, then build the equivalent from CK examples. Note ROCm ships its own flash-attention (CK-based, in `aotriton` / the `flash-attention` ROCm fork); the realistic path is to retarget `mha_fwd_kvcache` to a CK/aotriton attention with the project's required extras (paged/append KV cache, tree/block mask for speculative verify, sparse mask). The custom tree-verification mask (`flash_blockmask.h`) is the hard, project-specific piece that stock CK attention will not provide.

### Kernel family 2: Marlin W4A16 GPTQ/AWQ GEMM (reimplement, do not port)
- `src/qgemm/gptq_marlin/` is the vLLM Marlin kernel. It is raw NVIDIA sm80+ inline PTX with no HIP equivalent: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` and the bf16 variant (`marlin_device_ops.cuh:27,34`), `ldmatrix.sync.aligned.m8n8.x4.shared.b16` (:51), `cp.async.cg.shared.global` + `cp.async.commit_group` + `cp.async.wait_group` (`marlin.cuh:48-77`), `lop3.b32`, `prmt.b32`, `__cvta_generic_to_shared`, `red.relaxed.gpu.global.add.s32`, `ld.global.acquire.gpu.b32`, `fence.acq_rel.gpu`. hipify does not translate inline PTX; clang(HIP) cannot assemble PTX. Must be reimplemented with MFMA intrinsics + CK, matching Marlin's weight prepack/dequant-into-MMA layout, OR replaced by an existing ROCm W4A16 kernel and a matching weight-repack (note: the model weights are pre-repacked into Marlin's specific layout by `scripts/model_convert/gptq2marlin.py`, so swapping the kernel forces a matching dequant/repack change).

### Portable remainder (mechanical, Strategy B + fault-class fixes)
The model glue and the elementwise/norm/rope/topk/embedding/kvcache/sampling kernels (`src/model/*.cuh`) are ordinary CUDA C++ and hipify cleanly EXCEPT for the warp-size fault class below. `cublasGemmEx` in `linear.cuh`/`utils.cuh` maps to hipBLAS (`hipblasGemmEx`) -- torch's hipify handles the symbol rename, but the `cublas` link library in setup.py must become `hipblas`/`rocblas` under ROCm.

## CUDA surface inventory
- Kernels: norm (RMSNorm + fused), elementwise/activation (SiLU/GELU), rotary embedding, embedding gather, KV-cache append, attention mask build, topk/sampling, EAGLE tree drafter/verify, plus the two hot families above.
- Warp intrinsics (FAULT CLASS -- see Risks): `__shfl_xor_sync` / `__shfl_down_sync` with hardcoded 32-bit masks (`0xffffffff`, `0x0000ffff`) and step-16 reduction trees in `src/model/norm.cuh` (many, e.g. lines 24-28, 72-84, 150-175), `src/model/topk.cuh` (14-35), `src/model/drafter.cuh` (13-16). These bake warpSize==32.
- Libraries: cuBLAS (`cublasGemmEx`, `cublasCreate`, `cublasSetStream`, `cublasHandle_t`, `cudaDataType_t`, `CUBLAS_COMPUTE_32F`) -> hipBLAS/rocBLAS. No NCCL (single-GPU), no cuFFT/cuRAND/cuSPARSE, no Thrust/CUB outside CUTLASS.
- PTX / arch atoms: CuTe SM75/SM80 MMA+cp.async (flash); raw mma.sync/ldmatrix/cp.async/lop3/prmt PTX (Marlin). No Hopper wgmma, no warp-specialization (it is sm80-class), but still un-portable as written.
- Memory/streams: `cudaMalloc`/`cudaMemcpy*`/`cudaStream_t`/`cudaGetDeviceProperties` (flash_api reads `dprops.multiProcessorCount` for split-kv heuristic; on AMD use the CU count, fine after hipify). pybind11 module (`entry.cu`); watch the IPO/LTO-vs-pybind11 trap from the guide when building the HIP .so (disable IPO).
- Dtypes: `__half` / `__nv_bfloat16` selected by `ENABLE_DTYPE_FP16`/`ENABLE_DTYPE_BF16`; `--use_fast_math` and the `-U__CUDA_NO_HALF*` unblockers in setup.py. HIP half/bf16 headers cover these after hipify.

## Risk list
- WAVE64 WARP-SIZE (highest portable-code risk): every warp reduction in norm.cuh/topk.cuh/drafter.cuh assumes a 32-lane warp -- step-16 `__shfl_down_sync` trees and 32-bit `0xffffffff`/`0x0000ffff` masks. On gfx90a (wave64) a 32-bit mask is wrong and a 32-step tree reduces only half the wavefront. These are LOGICAL-32-warp reductions, so the correct fix is width-32 logical-warp ops (`__shfl_*_sync(mask, v, d, 32)` / arch-agnostic 32-lane subgroup), NOT a step-32 tree. Apply the MULTI-ARCH standard: device per-arch `kWarpSize` via `__GFX9__`->64 else 32 guards (NO `__AMDGCN_WAVEFRONT_SIZE__` in ROCm 7.2.1), host launch geometry from `hipDeviceAttributeWarpSize` / `hipGetDeviceProperties().warpSize`; never a single shared `-DWARP_SIZE` constant. Where the algorithm truly wants the whole physical wavefront, size for 64. The `0x0000ffff` 16-lane masks in drafter/norm need the same logical-width care.
- BUILD AS THE TEST: build the (eventually-portable) extension for `gfx90a;gfx1100` and confirm both code objects emit; do not bake a single wave width. Flag any porter that injects `-DWARP_SIZE=64`.
- CUTLASS/CuTe FlashAttention: non-portable; reimplement on CK/ck_tile or retarget to ROCm flash-attention/aotriton. Largest single risk and effort; the tree-verification block mask (speculative decoding) is project-specific and absent from stock CK attention.
- Marlin raw PTX W4A16: non-portable; reimplement with MFMA+CK and match the on-disk Marlin weight-repack format, OR swap to a ROCm W4A16 kernel and change the repack. The pre-quantized HF model assets are in Marlin layout, so the kernel and the repack must stay consistent.
- monolithic module: `entry.cu` includes all variants and the base attention pulls in flash_api -- you cannot link or run anything until family 1 builds. There is no "portable-only" runnable slice to validate independently; a CPU-only docker compile is not a gate (guide policy).
- cuBLAS link lib: setup.py `libraries=["cublas","dl"]` must become rocBLAS/hipBLAS under ROCm; hipify renames the calls but not the linker list.
- IPO/LTO vs pybind11 under HIP (guide): the HIP link may leave the .so as LTO bitcode with no `PyInit_*`; keep IPO off for the HIP build.
- `--use_fast_math` on HIP can change ULP results; the test is semantic (does the model emit "B"/"Blue"), not bit-exact, so this is low-risk for validation but note it.
- Test needs large HF downloads (MiniCPM4-8B + Marlin-quantized + Eagle/FR-Spec draft) and real GPU memory; size the gfx90a (64 GB MI250X GCD) accordingly.

## File-by-file change list (when/if the rewrite is undertaken)
- setup.py: detect ROCm torch (`torch.version.hip`); swap `libraries=["cublas","dl"]` -> rocBLAS/hipBLAS; drop `-gencode`/`sm_` and `--use_fast_math`/nvcc-only flags on HIP, emit `--offload-arch=gfx90a` (configurable, multi-arch capable); disable IPO; keep the dtype `-DENABLE_DTYPE_*` defines.
- New `src/cuda_compat.cuh` (device) + a small host warp-size helper: per-arch `kWarpSize` guards and the logical-warp shuffle width fixes; force-included on HIP TUs.
- src/model/norm.cuh, src/model/topk.cuh, src/model/drafter.cuh: convert hardcoded-32 shuffle trees to width-32 logical-warp ops (guarded), fix 32/16-bit masks.
- src/model/linear.cuh, src/utils.cuh, src/utils.cu, src/trait.cuh: cuBLAS->hipBLAS handle/type/enum review (hipblasGemmEx, hipDataType, HIPBLAS_COMPUTE_32F); hipify covers most, verify enum coverage.
- src/flash_attn/**: replace the CuTe FA2 kernels with a CK/ck_tile (or ROCm-flash-attention/aotriton) attention exposing the same `mha_fwd_kvcache` signature, including KV-append, causal, and the tree/block + sparse mask (`flash_blockmask.h`, `mask.cuh`). Remove the cutlass submodule from the HIP include path.
- src/qgemm/gptq_marlin/**: replace the PTX Marlin GEMM with an MFMA/CK W4A16 kernel; reconcile with `scripts/model_convert/gptq2marlin.py` repack (or add a ROCm repack).
- entry.cu / model glue: minimal; mostly unchanged once the two families build.

## Build commands (gfx90a)
- Prereq: a ROCm PyTorch (gfx90a) in the env; ninja installed.
- Portable-path bringup (will FAIL to link until families 1+2 build, expected):
  `CPMCU_DTYPE=fp16,bf16 PYTORCH_ROCM_ARCH=gfx90a pip install --no-build-isolation -e .`
  (no-build-isolation so it uses the host ROCm torch; setup.py's `CPMCU_CUDA_ARCH` numeric check must be bypassed/adapted for HIP -- it currently rejects non-numeric arch.)
- Multi-arch correctness build for the warp work: `PYTORCH_ROCM_ARCH="gfx90a;gfx1100"` and confirm `llvm-objdump --offloading build/.../C*.so` shows both gfx90a and gfx1100.

## Test plan
- Real GPU test (the validation gate): `pytest tests/test_model_generation.py` (pytest.ini -> testpaths=tests). It shells out to `python -m cpmcu.cli` against MiniCPM4-8B and the Marlin-quantized + Eagle/FR-Spec models (HuggingFace `openbmb/MiniCPM4-8B`, `openbmb/MiniCPM4-8B-marlin-cpmcu`, `openbmb/MiniCPM4-8B-Eagle-FRSpec*`), `--num-generate 32 --temperature 0.1`, and asserts the run succeeds + output contains the expected answer ("B"/"Blue"). This exercises base attention (flash), W4A16 (Marlin), and speculative decoding end-to-end -- it cannot pass until both hot families are reimplemented.
- Validatable slice: there is NO GPU slice that avoids family 1, because base attention itself uses flash_api. The earliest meaningful GPU milestone is "base (non-quant) MiniCPM4 prefill+decode emits coherent text" once the CK attention lands; Marlin/W4A16 is the second milestone; speculative decoding (tree mask) is the third.
- Non-GPU regression set: `tests/test_environment_setup` and `tests/test_gpu_memory_detection` are import/infra checks (must keep passing); `examples/` are smoke scripts, not asserted tests.

## Open questions
- Port-vs-rewrite acceptance: the FlashAttention(CUTLASS/CuTe) + Marlin(raw PTX) reimplementation is a large AMD-native effort, not a mechanical port. Does MOAT want to take this rewrite, or skip CPM.cu as `cant-port` (mechanical-port scope)? This is the decisive question.
- For attention, reuse ROCm flash-attention/aotriton vs hand-write ck_tile -- which gives the tree/sparse mask the speculative path needs?
- For W4A16, reimplement Marlin's exact layout in MFMA/CK vs adopt an existing ROCm W4A16 kernel + new repack -- which minimizes the model-asset conversion churn?

## Recommendation / disposition
Lead platform advanced to `planned` with the rewrite scope above so the decision is explicit and a porter can pick it up. If MOAT's bar is mechanical CUDA->HIP ports only, set `python3 utils/triage.py skip OpenBMB/CPM.cu --reason cant-port --note "FlashAttention on CUTLASS/CuTe + Marlin raw sm80 PTX; both require AMD-native (CK/MFMA) reimplementation, not a hipify port; monolithic module so no portable-only validatable slice"`. The portable model glue + warp fixes are straightforward, but they cannot be validated in isolation.

## Delta plans
(Followers gfx1100/gfx1151: not planned yet -- gated on a successful gfx90a build, which is gated on the two reimplementations. The warp-size work above already targets wave32 via the logical-32 fix + multi-arch build, so a future delta is mainly RDNA occupancy/CK-config tuning.)
