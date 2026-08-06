# FlashMoE ROCm Port Plan

## Project

- **Name**: FlashMoE
- **Upstream**: https://github.com/osayamenja/FlashMoE
- **Default branch**: main
- **Description**: Distributed MoE (Mixture of Experts) in a single fused kernel for high-throughput inference (NeurIPS'25)

## Existing AMD support

**Assessment**: None. The upstream README.md line 227 explicitly lists "AMD support" as a TODO item. No AMD-related forks exist (checked gh api repos/osayamenja/FlashMoE/forks). Web search found no existing ROCm/HIP port.

**Decision**: This project CANNOT be ported via a mechanical CUDA-to-HIP translation. It has hard dependencies on NVIDIA-proprietary libraries that have no AMD equivalents. See Risk List below for details.

## Build classification

**Classification**: Pure CMake project (not a PyTorch extension)

**Evidence**:
- `CMakeLists.txt` at root and `csrc/CMakeLists.txt` use `project(flashmoe LANGUAGES CXX CUDA)`
- No `find_package(Torch)`, `torch.utils.cpp_extension`, or `CUDAExtension`
- Pure C++20/CUDA codebase with pybind11 bindings for Python API

## Port strategy

**Recommendation**: BLOCKED -- cannot port. This project requires a ground-up AMD-native rewrite, not a CUDA-to-HIP translation.

**Rationale**: FlashMoE is built entirely on NVIDIA-proprietary infrastructure that has no AMD equivalents:

1. **cuBLASDx** (NVIDIA MathDx device-side GEMM library): The entire tile GEMM infrastructure uses `cublasdx::` extensively for device-side blocked matrix multiplication. There is NO AMD equivalent -- rocBLAS is a host-side library, not a device-side template library. A port would require replacing all cuBLASDx usage with Composable Kernel (CK) / rocWMMA, which is a complete rewrite of the compute kernels.

2. **NVSHMEM** (NVIDIA GPU-aware SHMEM): The project uses NVSHMEM for device-initiated asynchronous communication (`nvshmem_putmem_signal_nbi`, `nvshmem_signal_fetch`, `nvshmem_uint64_test`, etc.). AMD has no production-ready NVSHMEM equivalent -- there is experimental ROC-SHMEM but it is not feature-complete or API-compatible. The communication architecture would need to be redesigned.

3. **CUTLASS** (NVIDIA template library): Used for numeric conversion (`cutlass::NumericConverter`), alignment utilities (`cutlass::round_up`, `cutlass::AlignedArray`), and epilogue operations (`cutlass::epilogue::thread::ReLU`). Per PORTING_GUIDE.md: "CUTLASS does NOT port to ROCm and never will: do not attempt a CUTLASS->ROCm port."

4. **NVIDIA CCCL** (CUDA C++ Core Libraries): Used for `cuda::std::`, `cuda::atomic`, cooperative groups. While some of this has HIP equivalents, the deep integration with cuBLASDx/CUTLASS makes it inseparable.

5. **NVIDIA-specific arch features**: Code uses `sm_modifier::arch_specific` for SM90+, PTX inline assembly (`asm("cvt.rna.tf32.f32 %0, %1;" ...)`), and `__CUDA_ARCH__` macros throughout.

## CUDA surface inventory

### Libraries (all NVIDIA-proprietary, no AMD equivalent)

| CUDA Library | Usage | AMD Equivalent | Status |
|-------------|-------|----------------|--------|
| cuBLASDx (MathDx) | Device-side blocked GEMM, all compute | None (rocBLAS is host-side) | **BLOCKER** |
| NVSHMEM | Device-initiated async communication | ROC-SHMEM (experimental, not API-compatible) | **BLOCKER** |
| CUTLASS | Numeric conversion, alignment, epilogues | None (incompatible) | **BLOCKER** |
| CCCL (cuda::std) | Atomics, type traits, bit_cast | Partial HIP equivalents | Entangled |

### Warp intrinsics

- `__shfl_sync` at subscriber.cuh:318 -- Could be mapped to HIP but irrelevant given blockers
- `__syncwarp` at subscriber.cuh:316 -- Could be mapped to HIP but irrelevant given blockers
- Hardcoded `WARP_SIZE = 32` at infra/constants.cuh:14 -- Would need wave64/wave32 abstraction

### Data types (NVIDIA-specific)

- `__nv_bfloat16`, `__nv_bfloat162` -- Would need mapping to `hip_bfloat16`
- `cublasdx::tfloat32_t` -- cuBLASDx-specific type

### PTX/ISA

- `asm("cvt.rna.tf32.f32 %0, %1;" ...)` at tile.cuh:216 -- PTX instruction with no HIP equivalent

## Risk list

1. **FATAL: cuBLASDx dependency** -- The entire tile GEMM infrastructure (`tile::CollectiveMainloop`, `fGET`, `fGET_gated`) is built on cuBLASDx templates. There is no drop-in replacement. AMD would need rocWMMA/CK rewrite.

2. **FATAL: NVSHMEM dependency** -- The kernel's communication model relies on NVSHMEM for device-initiated put/signal operations. ROC-SHMEM is experimental and not API-compatible.

3. **FATAL: CUTLASS dependency** -- Numeric conversion, alignment, and epilogue utilities from CUTLASS are used throughout. CUTLASS cannot be ported.

4. **Hardcoded WARP_SIZE=32** -- Would break on CDNA wave64, but irrelevant given other blockers.

5. **Arch-specific tuning** -- SM70/80/90/100 specific optimizations baked into heuristics.

## File-by-file change list

N/A -- Cannot port. Would require complete rewrite.

## Build commands

N/A -- Project cannot be built for ROCm.

## Test plan

N/A -- No port possible.

## Open questions

None. This project is architecturally incompatible with ROCm due to its deep reliance on NVIDIA-proprietary compute and communication libraries.

## Recommendation

Set project state to `blocked` with reason: "Depends on cuBLASDx/NVSHMEM/CUTLASS which have no AMD equivalents; requires ground-up AMD-native implementation, not a CUDA-to-HIP port."

If AMD GPU support for FlashMoE is desired, it would require:
1. Rewriting the tile GEMM infrastructure using Composable Kernel (CK) or rocWMMA
2. Redesigning the communication layer without NVSHMEM (possibly using RCCL or a custom approach)
3. Replacing all CUTLASS utilities with AMD-compatible alternatives
4. This is effectively a new implementation inspired by FlashMoE's algorithms, not a port
