# Port plan: tiny-vllm

## Project

- **Name**: tiny-vllm
- **Upstream**: https://github.com/jmaczan/tiny-vllm
- **Default branch**: main
- **Description**: Educational project for building your own LLM inference engine in C++ and CUDA. Implements core vLLM concepts (paged attention, continuous batching, KV cache) for Llama 3.2 1B.

## Existing AMD support

**None found.** Searched:
- Grepped upstream docs for AMD/ROCm/HIP references: only mention of "AMD CPU (Ryzen 7 9800X3D)" as dev machine, no GPU support
- Web search for "tiny-vllm ROCm/AMD/HIP": no results (web search returned hits for main vLLM project which has mature ROCm support, not this educational tiny-vllm)
- GitHub fork scan (`gh api repos/jmaczan/tiny-vllm/forks`): 30 forks, none with rocm/hip/amd in name
- No upstream ROCm/HIP branches or issues

**Decision**: Proceed with a from-scratch HIP port using Strategy A (pure CMake, compat-header model). The main vLLM project has mature ROCm support, but this is a separate educational project with hand-written CUDA kernels -- the port adds educational value by demonstrating how to write portable HIP kernels for LLM inference.

## Build classification

**Pure CMake project** (Strategy A)

Evidence (CMakeLists.txt):
- Line 4: `project(tiny-vllm LANGUAGES CXX CUDA)`
- Line 23: `find_package(CUDAToolkit REQUIRED)`
- Line 33-36: Links `CUDA::cublas`, `CUDA::cudart`
- No `find_package(Torch)`, no `CUDAExtension`, no setup.py

## Port strategy

**Strategy A: pure CMake, compat-header approach**

Rationale: This is a standalone CMake build with `.cu` sources and CUDA libraries (cuBLAS). Not a PyTorch extension. The port will:
1. Add a `cuda_to_hip.h` compat header with CUDA-to-HIP symbol aliases
2. Modify CMakeLists.txt to add `USE_HIP` option, `enable_language(HIP)`, mark `.cu` as LANGUAGE HIP
3. Swap cuBLAS for hipBLAS via compat header

## CUDA surface inventory

### Files
- `src/kernels.cu` (449 lines): All custom CUDA kernels
- `src/kernels.cuh` (19 lines): Kernel declarations
- `src/main.cpp` (1000+ lines): Host code with CUDA runtime + cuBLAS calls

### Kernels (all in kernels.cu)
| Kernel | Lines | Purpose |
|--------|-------|---------|
| `embeddingGatherKernel` | 25-33 | Token embedding lookup |
| `rmsNormKernel` | 48-73 | RMS normalization with tree reduction |
| `ropeKernel` | 89-102 | Rotary position embedding |
| `causalMaskKernel` | 125-138 | Causal attention mask |
| `softmaxKernel` | 158-200 | Softmax with online max/sum |
| `residualKernel` | 221-226 | Residual connection add |
| `siluKernel` | 241-248 | SiLU activation with elementwise multiply |
| `embeddingGatherKernelDecode` | 257-266 | Decode-phase embedding |
| `ropeKernelDecode` | 281-293 | Decode-phase RoPE |
| `softmaxKernelDecode` | 318-359 | Decode-phase softmax |
| `pagedAttentionKernel` | 382-444 | Paged attention with online softmax |

### Warp intrinsics
- `__shfl_down_sync(0xffffffff, qk, delta)` at lines 410-414 in `pagedAttentionKernel`
  - Used for tree reduction of dot product within a warp (32 threads)
  - The kernel is launched with `HEAD_DIM=64` threads per block
  - The shuffle uses hardcoded 32-bit mask `0xffffffff`

### __syncthreads usage
- Multiple uses for shared memory synchronization in tree reductions (rmsNormKernel, softmaxKernel, softmaxKernelDecode, pagedAttentionKernel)
- These are block-wide barriers with proper shared memory and are wave-size agnostic

### Data types
- `__nv_bfloat16` throughout (AMD: `__hip_bfloat16` via `hip/hip_bf16.h`)
- Standard float for intermediate computations

### Libraries
- **cuBLAS**: `cublasGemmEx` for matrix multiplications (Q/K/V/O projections, attention, MLP)
  - ROCm equivalent: hipBLAS `hipblasGemmEx`
- **CUDA runtime**: `cudaMalloc`, `cudaMemcpy`, `cudaFree`, `cudaGetDeviceCount`, `cudaGetDeviceProperties`, `cudaMemGetInfo`, `cudaError_t`
  - ROCm equivalent: Direct `hip*` equivalents via compat header

### Memory patterns
- Simple device allocations via `cudaMalloc`
- `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToDevice`
- No managed memory, no pinned memory, no streams/events
- No textures/surfaces

### No usage of
- cuRAND, cuFFT, cuSPARSE
- Thrust, CUB
- Cooperative groups
- Dynamic parallelism
- CUDA graphs

## Risk list

### High
1. **Warp shuffle mask width** (lines 410-414 in pagedAttentionKernel)
   - Issue: `__shfl_down_sync(0xffffffff, ...)` uses 32-bit mask; HIP requires 64-bit mask
   - Fix: Define compat macro `#define HIP_FULL_MASK 0xffffffffffffffffULL` for HIP, `0xffffffff` for CUDA
   - Per PORTING_GUIDE: "HIP `__shfl_sync`/... REQUIRE a 64-bit mask"

2. **Warp shuffle reduction on HEAD_DIM=64 threads**
   - The kernel launches 64 threads per block, uses two 32-thread warp reductions, then combines via shared memory
   - On wave64 (gfx90a): All 64 threads form ONE wavefront; the reduction pattern (16/8/4/2/1) still works correctly because it operates within 32 lanes
   - On wave32 (gfx1100): Two warps of 32 threads each; same pattern works
   - Verdict: The current code IS wave-size agnostic because it does logical-warp operations (32-wide shuffles) and combines via shared memory + `__syncthreads()`

### Medium
3. **bfloat16 type mapping**
   - Issue: `__nv_bfloat16` needs mapping to `__hip_bfloat16`
   - Fix: `#define __nv_bfloat16 __hip_bfloat16` in compat header + include `<hip/hip_bf16.h>`
   - Also need `nv_bfloat16` -> `__hip_bfloat16` (used without prefix in some places)

4. **cuBLAS to hipBLAS**
   - `cublasGemmEx` -> `hipblasGemmEx`
   - `cublasCreate` -> `hipblasCreate`
   - `cublasHandle_t` -> `hipblasHandle_t`
   - `cublasStatus_t` -> `hipblasStatus_t`
   - `CUBLAS_STATUS_SUCCESS` -> `HIPBLAS_STATUS_SUCCESS`
   - Compute type: `CUBLAS_COMPUTE_32F` -> `HIPBLAS_COMPUTE_32F`
   - Data types: `CUDA_R_16BF` -> `HIP_R_16BF` (from `<hip/library_types.h>`)

5. **Hardcoded CUDA arch in CMakeLists.txt**
   - Line 11: `set(CMAKE_CUDA_ARCHITECTURES 120)` (Blackwell)
   - Fix: For HIP, use `CMAKE_HIP_ARCHITECTURES` with default fallback pattern

### Low
6. **Hardcoded NVCC path**
   - Lines 2-3: `set(CMAKE_CUDA_COMPILER "/opt/cuda/bin/nvcc")`
   - Fix: Only set these when not USE_HIP; CMake finds hipcc automatically

7. **CUDA include in main.cpp**
   - Line 4-5: `#include <cuda_runtime.h>`, `#include <cublas_v2.h>`
   - Fix: Route through compat header

## File-by-file change list

### New files
1. **src/cuda_to_hip.h** - CUDA-to-HIP compat header
   - Include `<hip/hip_runtime.h>` and `<hip/hip_bf16.h>` on HIP
   - bfloat16 mappings: `__nv_bfloat16` -> `__hip_bfloat16`
   - CUDA runtime mappings: `cudaMalloc`, `cudaMemcpy`, `cudaFree`, etc.
   - cuBLAS -> hipBLAS mappings
   - 64-bit warp mask constant

### Modified files
1. **CMakeLists.txt**
   - Add `option(USE_HIP "Build with HIP for AMD GPUs" OFF)`
   - Conditional `enable_language(HIP)` vs `enable_language(CUDA)`
   - Move hardcoded NVCC paths inside `if(NOT USE_HIP)` guard
   - `set_source_files_properties(src/kernels.cu PROPERTIES LANGUAGE HIP)` when USE_HIP
   - Link `hip::host` and `roc::hipblas` instead of `CUDA::cublas` and `CUDA::cudart`
   - Set `CMAKE_HIP_ARCHITECTURES` with fallback to gfx90a

2. **src/main.cpp**
   - Replace `#include <cuda_runtime.h>` and `#include <cublas_v2.h>` with `#include "cuda_to_hip.h"`

3. **src/kernels.cu**
   - Replace `#include "kernels.cuh"` with `#include "cuda_to_hip.h"` followed by `#include "kernels.cuh"`
   - Replace `0xffffffff` in `__shfl_down_sync` calls with `WARP_FULL_MASK` macro

4. **src/kernels.cuh**
   - Replace `#include <cuda_bf16.h>` with a guard that includes the right header based on platform

## Build commands

### Configure (gfx90a)
```bash
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -G Ninja
```

### Build
```bash
cmake --build build
```

### Alternative for other archs
```bash
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -G Ninja
cmake --build build
```

## Test plan

### Primary validation
The project has no formal test suite. Validation is via inference output:

1. **Build and run inference**:
   ```bash
   cd build && ./tiny-vllm
   ```
   This requires Llama 3.2 1B model weights in safetensors format (download from HuggingFace).

2. **Reference comparison** (full_test.sh):
   ```bash
   echo "128000 128006 9125 128007 271 38766 1303 33025 2696 25 6790 220 2366 18 198 15724 2696 25 220 2304 2947 220 2366 21 271 128009 128006 882 128007 271 3923 374 279 6864 315 9822 30 128009 128006 78191 128007 271" | ./build/tiny-vllm
   ```
   Compare generated tokens to reference.txt (Python reference implementation output).

3. **Determinism check**: Run the same prompt twice, verify identical output.

### Non-GPU tests
None -- this is a GPU-only inference engine.

### Validation criteria
- Builds successfully with hipcc
- Runs inference without GPU faults
- Generates coherent text output matching the expected token sequence
- Run-to-run deterministic output (same prompt -> same tokens)

## Open questions

1. **Model weights access**: The validator will need access to Llama 3.2 1B safetensors weights from HuggingFace. This may require HF token/login if model is gated.

2. **Reference output tolerance**: The reference.txt shows intermediate tensor values from PyTorch. Due to floating-point differences between CUDA/HIP, the HIP output may differ at ULP level. The key validation is that generated tokens match, not intermediate tensors.

3. **Performance**: This is an educational project, not a production inference engine. Performance validation is not a primary goal, but gross performance issues (orders of magnitude slower) would indicate a bug.
