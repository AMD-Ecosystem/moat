# FlashRT ROCm Port Plan

## Project

- **Name**: FlashRT
- **Upstream**: https://github.com/LiangSu8899/FlashRT
- **Default branch**: main
- **Description**: A general CUDA kernel library for realtime AI inference -- VLA control (Pi0, Pi0.5, GROOT), LLM serving (Qwen3.6-27B), and world-model image generation. Hand-written kernels cover norm/activation/fusion/RoPE/FP8/NVFP4 GEMM/attention primitives.

## Existing AMD Support

**Finding**: No existing AMD/ROCm/HIP support exists.

- Grepped `README* docs/` for `amd|rocm|hip|gfx`: no matches related to actual AMD support.
- The docs **explicitly acknowledge future HIP work** (`exec/backend/backend.h` comment: "backend/hip/... later"; `docs/extension/attention_backend.md` mentions "RDNA, AMD MI300" as a future porting target).
- No forks under ROCm/AMD/GPUOpen orgs, no rocm/hip branches upstream.
- Web search for "FlashRT ROCm" or "FlashRT AMD GPU" returned no relevant results.

**Decision**: Proceed with a fresh HIP port. The project's architecture (explicit backend abstraction in `exec/backend/`) is designed for multi-vendor support, making this a well-structured porting target.

**Merge policy**: The upstream appears to be a standard GitHub repo with no indication of a link-forks-instead-of-merge policy. An upstream PR is the appropriate delivery vehicle.

## Build Classification

**Classification**: Pure CMake project (Strategy A)

**Evidence**:
- `CMakeLists.txt` line 2: `project(flash_rt CUDA CXX)`
- `setup.py` is a trivial wrapper that calls `setuptools.setup()` -- the CUDA kernels are built separately via CMake, not via `torch.utils.cpp_extension`.
- `pyproject.toml` has no torch build dependency.
- No `find_package(Torch)`, no `CUDAExtension`, no `torch.utils.cpp_extension` usage.

The project builds three pybind11 extension modules via CMake:
1. `flash_rt_kernels.so` -- main kernel library
2. `flash_rt_fa2.so` -- Flash Attention 2 vendored kernels (SM80/SM89/SM120)
3. `flash_rt_fp4.so` -- NVFP4 FP4 kernels (SM100/SM110/SM120)

## Port Strategy

**Strategy A: Pure CMake with compat header** -- but with significant scope reduction.

### Scope Assessment

This is a **very large** project with 139 `.cu` files and 88 `.cuh` files. The CUDA surface is extensive:
- CUTLASS 4.4.2 for GEMM kernels (SM80/SM100/SM110/SM120 specific)
- cuBLAS/cuBLASLt for decomposed attention and GEMM
- Flash Attention 2 vendored source (SM80-family specific)
- FlashInfer XQA kernels
- Sage2 attention kernels
- Numerous NVFP4/NVFP8 (Blackwell) specific kernels

**Critical Reality**: The overwhelming majority of this codebase is **NVIDIA-architecture-specific**:
- SM100/SM110/SM120 (Thor/Hopper/Blackwell) CUTLASS kernels with TMA, warp specialization
- SM80/SM89 Flash Attention 2 kernels
- NVFP4/W4A8 block-scaled GEMM for Blackwell
- NVIDIA-specific PTX/SASS generation

A mechanical port of CUTLASS/FA2/NVFP4 kernels to HIP would **not work** -- CUTLASS does not port to ROCm (per PORTING_GUIDE). These would require AMD-native rewrites using Composable Kernel (CK) or similar.

### Recommended Approach

**Phase 1 (MVP for gfx90a)**: Port the **exec/ backend abstraction** and **basic runtime kernels** only.

1. **exec/ backend**: The `exec/backend/backend.h` abstraction layer is explicitly designed for HIP. Create `exec/backend/hip/hip_backend.cpp` implementing the same interface.

2. **Basic memory-bound kernels**: Port the simpler kernels that do not use CUTLASS:
   - `csrc/kernels/softmax.cu`
   - `csrc/kernels/rms_norm*.cu`
   - `csrc/kernels/rope_*.cu`
   - `csrc/fused_fp16/rms_norm_noweight_fp16.cu`

3. **cuBLAS decomposed attention path**: Replace cuBLAS/cuBLASLt with hipBLAS/hipBLASLt for the attention decomposed path (used by Thor SM110 today).

4. **Skip for now**:
   - All CUTLASS GEMM kernels (require CK rewrite)
   - Flash Attention 2/4 vendored source (use ROCm flash-attention or CK attention)
   - NVFP4/FP8 quantization kernels (require Blackwell-specific hardware)

**This is a BLOCKED port** -- the meaningful validation requires attention and GEMM kernels that would need AMD-native implementations (Composable Kernel), which is substantial new development, not a mechanical port.

## CUDA Surface Inventory

### Warp Intrinsics (HIGH RISK)

Found ~40 usages of warp sync primitives:
- `__shfl_sync`, `__shfl_xor_sync`: sage2, flashinfer_xqa, rms_norm, fused_fp4 kernels
- `__ballot_sync`: flashinfer_xqa_src/mha.cu (5 usages)
- `__activemask`: flashinfer_xqa_src/utils.cuh

**Hardcoded warp size (32)**: Multiple files define `#define WARP_SIZE 32` or `#define SM_WARP_SIZE 32`:
- `csrc/attention/sage2/qattn/attn_utils.cuh:26`
- `csrc/kernels/softmax.cu:9`
- `csrc/kernels/fp4_w4a4_matvec_sm120.cu:85`
- `csrc/kernels/attention_dit_bf16.cu:14`

All of these would need wave64 adaptation for gfx90a.

### CUDA Libraries

| Library | Usage | HIP Equivalent |
|---------|-------|----------------|
| cuBLAS | `cublas_v2.h`, decomposed attention | hipBLAS |
| cuBLASLt | FP8 GEMMs, `cublasLt.h` | hipBLASLt |
| CUTLASS 4.4.2 | All GEMM kernels (SM80/100/110/120) | **No equivalent** -- requires CK rewrite |

### CUDA Runtime

- `cudaMalloc`, `cudaFree`, `cudaMemcpy` -- standard runtime
- `cudaStream_t`, `cudaEvent_t` -- stream/event management
- `cudaGraphInstantiate`, `cudaGraphLaunch` -- CUDA graphs for capture/replay
- No textures/surfaces found
- No Thrust/CUB usage found
- No cooperative groups usage found

### CUTLASS Usage (NOT PORTABLE)

Extensive CUTLASS 4.4.2 usage for:
- FP8 GEMM (SM100/SM110)
- FP16/BF16 GEMM (SM100)
- NVFP4 W4A16 GEMM (SM100/SM120)
- Fused MHA (SM100)
- Block-scaled GEMMs

The CMakeLists.txt requires CUTLASS v4.4.2 cloned into `third_party/cutlass`.

### Flash Attention 2 (NOT PORTABLE AS-IS)

The vendored `csrc/attention/flash_attn_2_src/` contains SM80-specific Flash Attention 2 kernels. For ROCm, use the existing ROCm flash-attention (CK-based) or Triton-based implementation instead.

## Risk List

1. **CUTLASS dependency (BLOCKING)**: All GEMM and fused MHA kernels use CUTLASS extensively. CUTLASS does not port to ROCm. These would require Composable Kernel (CK) rewrites -- substantial new development.

2. **Warp size hardcoding (HIGH)**: Multiple kernels hardcode `WARP_SIZE=32`. Needs wave64 abstraction for gfx90a.

3. **Architecture-specific kernels (HIGH)**: SM100/SM110/SM120 specific code (TMA, warp specialization) has no HIP equivalent.

4. **Flash Attention 2 vendor source (MEDIUM)**: Would need to be replaced with ROCm flash-attention (CK or Triton backend).

5. **FP8/NVFP4 data types (MEDIUM)**: `cuda_fp8.h` types need mapping to HIP equivalents; block-scaled NVFP4 is Blackwell-specific.

6. **cuBLASLt FP8 GEMM (MEDIUM)**: hipBLASLt FP8 support varies by ROCm version.

## File-by-File Change List (Partial Port)

### New Files

1. `csrc/cuda_to_hip.h` -- compat header with CUDA->HIP symbol mappings
2. `exec/backend/hip/hip_backend.cpp` -- HIP backend implementation

### Modified Files

1. `CMakeLists.txt`:
   - Add `option(USE_HIP "Build with HIP for AMD GPUs" OFF)`
   - Add HIP language enable under USE_HIP
   - Gate CUDA-specific targets (CUTLASS, FA2, NVFP4) on `NOT USE_HIP`
   - Replace `CUDA::cudart`, `CUDA::cublas`, `CUDA::cublasLt` with HIP equivalents
   - Add `set_source_files_properties(... LANGUAGE HIP)` for portable .cu files

2. `exec/CMakeLists.txt`:
   - Add HIP backend build target

3. Basic kernel files (softmax, rms_norm, rope):
   - Include cuda_to_hip.h
   - Abstract warp size to use compile-time constant based on arch

### Excluded Files (Require AMD-Native Rewrite)

All of:
- `csrc/gemm/*.cu` (CUTLASS-based)
- `csrc/attention/fmha_*.cu` (CUTLASS FMHA)
- `csrc/attention/flash_attn_2_src/` (FA2 vendor)
- `csrc/attention/flash_attn_4_src/` (FA4 vendor)
- `csrc/gemm/fp4/` (NVFP4 CUTLASS)

## Build Commands

```bash
# Clone CUTLASS (required for NVIDIA build, not needed for HIP-only)
git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git third_party/cutlass

# Configure for HIP (partial port -- basic kernels only)
cmake -B build -S . \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++

# Build
cmake --build build -j$(nproc)
```

## Test Plan

### GPU Tests

With the partial port (basic kernels only), validation would be limited to:
1. `tests/test_install_smoke.py` -- import verification (would fail without GEMM kernels)
2. Unit tests for softmax, RMS norm kernels (if isolated tests exist)

### Non-GPU Tests

The Python-only tests (precision_spec, norm_stats_adapter, weight_loader) should not regress.

## Open Questions

1. **Is a partial port valuable?** The exec/ backend abstraction + basic kernels (softmax, norm) without GEMM or attention is insufficient for any real workload. The project's value is in its optimized CUTLASS GEMM and FA2 attention kernels.

2. **Composable Kernel rewrite scope**: What is the effort to rewrite the CUTLASS GEMM templates in CK? This is the only path to a functional port.

3. **Upstream receptiveness**: Would the maintainer accept a partial ROCm enablement (runtime only) or only a complete port?

4. **ROCm flash-attention integration**: The ROCm ecosystem has CK-based and Triton-based flash-attention implementations. Would the maintainer accept a different attention backend for AMD?

## Recommendation

**BLOCK this project** with reason: "CUTLASS-heavy codebase requires Composable Kernel (CK) reimplementation of all GEMM and attention kernels for meaningful AMD support. This is substantial new development (~months of kernel engineering), not a mechanical HIP port (~days). The exec/ backend abstraction is porting-ready, but the core kernel library is architecture-locked to NVIDIA."

The project explicitly designed its backend abstraction for future HIP support (`backend/hip/... later`), but the kernel implementations are deeply NVIDIA-specific. A meaningful port requires:
1. CK-based GEMM kernels replacing CUTLASS
2. ROCm flash-attention (CK or Triton) replacing FA2 vendor
3. Wave64 adaptation of all warp-level kernels

This is better characterized as "build AMD-native kernels using the project's abstractions" rather than "port CUDA code to HIP."
