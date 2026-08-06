# fp6_llm Porting Plan (linux-gfx90a)

## Project

- **Name**: fp6_llm
- **Upstream**: https://github.com/usyd-fsalab/fp6_llm
- **Default branch**: main
- **Description**: FP6-LLM provides efficient GPU support for LLM inference with 6-bit (FP6) and 5-bit (FP5) quantization. It uses custom CUDA kernels with Tensor Core support for mixed-precision (FP6/FP5 weights + FP16 activations) matrix multiplication.

## Existing AMD Support

**Assessment**: No existing AMD/ROCm support.

- Grep of upstream docs for amd/rocm/hip: No matches.
- No AMD/ROCm forks found in the repository's fork list.
- No upstream branches with rocm/hip/amd names.
- No issues or PRs related to AMD/ROCm.
- Web search: No existing port found.

**Related AMD work**: AMD ROCm 7.0+ added native FP6/MXFP6 support for MI350/MI355X series GPUs via MFMA instructions, but this is a different hardware-accelerated path, not a port of fp6_llm's SIMT-dequant + Tensor-Core-MMA approach. The fp6_llm kernel design is fundamentally tied to NVIDIA Tensor Core architecture.

**Decision**: This project is NOT a candidate for mechanical HIP translation. The kernels are fundamentally NVIDIA Tensor-Core-specific and require an AMD-native rewrite, not a port. **Recommend SKIP with disposition `cant-port`.**

## Build Classification

- **Type**: pytorch extension (Strategy B)
- **Evidence**:
  - `setup.py` line 2: `from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension`
  - `setup.py` lines 32-40: Uses `CUDAExtension` to build `fp6_llm_cuda` from `pybind.cpp` and `fp6_linear.cu`
  - NVCC-specific flags: `-maxrregcount=255`, `-gencode=arch=compute_80,code=sm_80`

## Port Strategy

**Assessment: NOT PORTWORTHY via mechanical HIP translation.**

This project cannot be ported to HIP via the standard Strategy A or Strategy B approaches. The core GEMM kernel is built entirely around NVIDIA-specific features:

### 1. NVIDIA Tensor Core PTX (No HIP Equivalent)

The entire kernel is designed around NVIDIA's `mma.sync.aligned.m16n8k16` instruction (ptx_mma.cuh:62-70):
```cuda
asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
             "{ %0, %1, %2, %3},"
             "{ %4, %5, %6, %7 },"
             "{ %8, %9 },"
             "{ %10, %11, %12, %13 };"
             ...);
```
AMD CDNA (gfx90a) has MFMA (Matrix Fused Multiply-Add) instructions, but:
- Different ISA: MFMA is not a 1:1 mapping from `mma.sync`
- Different tile shapes: NVIDIA m16n8k16 vs AMD's mfma_f32_16x16x16f16, etc.
- Different register layouts and data movement patterns
- Requires complete kernel redesign, not symbol substitution

### 2. PTX ldmatrix Instructions (No HIP Equivalent)

The kernel uses `ldmatrix.sync.aligned.x2/x4.m8n8.shared.b16` (ptx_mma.cuh:43-56) for efficient shared-memory-to-register loads. These are Ampere+ PTX instructions with no HIP/AMDGPU equivalent. The entire shared memory staging and register allocation strategy is designed around ldmatrix.

### 3. PTX cp.async Instructions (Partial HIP Support)

Uses `cp.async.cg.shared.global` for async global-to-shared copies (ptx_cp.async.cuh:27-34). HIP has `__builtin_nontemporal_load` and async copy APIs, but the commit/wait_group semantics differ. This is the only component with a partial HIP path.

### 4. `__cvta_generic_to_shared` (No Direct HIP Equivalent)

Used in ptx_mma.cuh:41 and ptx_cp.async.cuh:26 for address space conversion. HIP has no direct equivalent; shared memory pointers work differently.

### 5. Hardcoded WARP_SIZE = 32

configs.h:9 defines `#define WARP_SIZE 32`. While this could be made wave-size-aware, the entire tiling scheme (WARP_M=64, WARP_K=64, MMA_16x8x16 tiles) is designed for 32-lane warps feeding Tensor Core instructions. The tiling arithmetic assumes 32 threads per warp.

### 6. Architecture-Specific Design

The kernel is explicitly designed for A100/sm80:
- setup.py: `-gencode=arch=compute_80,code=sm_80`
- README: "only tested and verified on A100 GPUs"
- The entire TC-FPx design scheme is built around NVIDIA Tensor Core characteristics

## CUDA Surface Inventory

| Category | Elements | HIP Equivalent | Risk |
|----------|----------|----------------|------|
| PTX mma.sync | mma.sync.aligned.m16n8k16 | **NONE** - requires MFMA rewrite | BLOCKER |
| PTX ldmatrix | ldmatrix.sync.aligned.x2/x4 | **NONE** | BLOCKER |
| PTX cp.async | cp.async.cg.shared.global, commit_group, wait_group | Partial via HIP async APIs | HIGH |
| Address conversion | __cvta_generic_to_shared | **NONE** direct | HIGH |
| Warp intrinsics | __shfl_sync (1 occurrence) | hipShflSync (needs 64-bit mask) | LOW |
| Hardcoded warp size | WARP_SIZE=32 | Needs wave-size abstraction | HIGH |
| CUDA headers | cuda.h, cuda_fp16.h, cuda_runtime.h | HIP equivalents exist | LOW |
| PyTorch extension | CUDAExtension, torch/extension.h | HIPExtension via ROCm torch | LOW |
| cuBLAS | None (uses custom kernel) | N/A | NONE |
| Streams | cudaStream_t | hipStream_t | LOW |

## Risk List

1. **BLOCKER: NVIDIA Tensor Core PTX** - The `mma.sync.aligned.m16n8k16` instruction is the computational core. No HIP equivalent; requires AMD MFMA rewrite.

2. **BLOCKER: ldmatrix PTX** - The `ldmatrix.sync` instructions for efficient shared-to-register data movement have no HIP equivalent. Requires redesign of data movement.

3. **HIGH: Architectural mismatch** - The entire kernel design (tiling, register allocation, shared memory layout, software pipelining) is optimized for NVIDIA Ampere Tensor Cores. A HIP port would require a ground-up rewrite using AMD's MFMA or rocWMMA.

4. **HIGH: No ROCm torch integration** - While torch.utils.cpp_extension supports HIPExtension, the PTX assembly blocks would fail at hipcc compile time.

5. **MEDIUM: __cvta_generic_to_shared** - HIP shared memory works differently; needs address space handling redesign.

6. **LOW: Wave size** - Hardcoded WARP_SIZE=32, but this is moot given the Tensor Core dependency.

7. **LOW: __shfl_sync mask** - Single usage needs 64-bit mask for HIP, but again moot.

## Recommendation

**SKIP this project with disposition `cant-port`.**

### Rationale

This is not a mechanical porting candidate. The fp6_llm kernels are a showcase of NVIDIA-specific features:
- PTX inline assembly for Tensor Core MMA
- PTX ldmatrix for efficient data staging
- PTX cp.async for async memory transfers
- Tiling and pipeline design tuned for Ampere Tensor Core characteristics

A "port" would require reimplementing the FP6 quantized GEMM kernel from scratch using:
- AMD MFMA intrinsics (rocWMMA or direct amdgcn intrinsics)
- AMD-appropriate tiling (e.g., 32x32x8 or 16x16x16 for MFMA)
- AMD async copy patterns
- Wave64-appropriate data layouts

This is a REWRITE, not a port, and is out of scope for MOAT's mechanical HIP translation mission.

### Alternative AMD Path

For FP6 quantization on AMD GPUs, users should consider:
1. **Native ROCm FP6** (ROCm 7.0+): MI350/MI355X GPUs have native MXFP6 hardware support via MFMA scale instructions
2. **AMD Quark**: AMD's quantization toolkit supports FP6/MXFP6 for vLLM/SGLang inference
3. **Composable Kernel (CK)**: AMD-native tensor operations library for writing efficient GEMM kernels

## File-by-file Change List

N/A - Project is not a porting candidate.

## Build Commands

N/A - Project is not a porting candidate.

## Test Plan

N/A - Project is not a porting candidate.

## Open Questions

1. Should MOAT track "rewrite candidates" separately from "port candidates"? Projects like fp6_llm require AMD-native implementations rather than HIP translations.

2. Should we document the AMD-native alternatives (Quark, native MXFP6, CK) in a "recommended alternatives" section for cant-port projects?
