# accelerated-scan ROCm Port Plan

## Project

- **Name**: accelerated-scan
- **Upstream**: https://github.com/proger/accelerated-scan
- **Default branch**: main
- **Description**: First-order parallel associative scan implementation for state space models and linear RNNs. Provides both a pure C++ CUDA warp-shuffle kernel (`warp.cuh`) and a Triton-based implementation (`scalar.py`).

## Existing AMD Support

**Assessment**: None found.

**Searches performed**:
- Upstream docs grep (`grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/`): No matches
- WebSearch for "accelerated-scan ROCm", "accelerated-scan AMD GPU", "accelerated-scan HIP", "accelerated-scan MI300/gfx9": No existing AMD port
- GitHub forks check (`gh api repos/proger/accelerated-scan/forks`): 9 forks found, none with rocm/hip/amd branches
- Upstream PRs/issues: No AMD/ROCm/HIP related PRs or issues

**Decision**: Proceed with a fresh HIP port. No prior AMD work exists.

**Merge policy**: Standard GitHub project -- PRs welcome, no indication of a link-to-forks policy.

## Build Classification

**Type**: pytorch extension (Strategy B)

**Evidence**:
- `accelerated_scan/warp.py` line 5: `from torch.utils.cpp_extension import load_inline`
- `accelerated_scan/warp.py` lines 13-28: Uses `load_inline()` with `cuda_sources=[cuda_source]` for JIT compilation
- `pyproject.toml` line 8-10: `dependencies = ["torch>=2.9.0"]`
- No CMake, setup.py with `CUDAExtension`, or `find_package(Torch)` -- purely inline JIT extension

**ext_type**: `torch-extension`

## Port Strategy

**Strategy B** (pytorch extension) applies:

1. The Triton-based implementations (`scalar.py`, `complex.py`) should work on AMD with minimal changes since Triton has ROCm support. Test these first.

2. The C++ CUDA kernel (`warp.cuh`) uses `torch.utils.cpp_extension.load_inline()` with `cuda_sources`. On a ROCm PyTorch build, this should automatically hipify via torch's internal hipify infrastructure, BUT the kernel has significant warp-size assumptions that require manual fixes.

3. The ref.py is pure PyTorch and needs no changes.

## CUDA Surface Inventory

### warp.cuh (C++ CUDA kernel - 425 lines)

**Headers**:
- `#include <cuda.h>` (line 1)
- `#include <cuda_runtime.h>` (line 2)
- `#include <ATen/cuda/CUDAContext.h>` (line 4)
- `#include <torch/extension.h>` (line 5)

**CUDA Runtime API**:
- `at::cuda::getCurrentCUDAStream().stream()` (line 278) -> `at::cuda::getCurrentHIPStream()`

**Warp intrinsics** (CRITICAL - lines 140-141, 176-177):
- `__shfl_up_sync(0xffffffff, ...)` with width=32 implicit
- Uses `warpSize` at line 175: `for (int delta = 1; delta < warpSize; delta *= 2)`

**Hardcoded warp size constants**:
- Line 280: `constexpr int kNThreadsPerWarp = 32;` -- CRITICAL: hardcoded 32
- Lines 282-380: All dispatch cases compute thread counts assuming 32-thread warps
- Line 140-141, 176-177: `__shfl_up_sync(0xffffffff, ...)` uses 32-bit mask

**Data types**:
- `__nv_bfloat16` (line 385) -> `hip_bfloat16`
- `__half` (line 387) -> `__half` (same in HIP)

**No textures/surfaces, no cuBLAS/cuFFT/cuRAND, no managed memory, no streams/events beyond getCurrentCUDAStream**.

### scalar.py (Triton kernel - 145 lines)

Pure Triton with `tl.associative_scan`. Should work on AMD via Triton ROCm backend without code changes. Uses `enable_fp_fusion=False` flag for numerical stability.

### complex.py (Triton kernel - 240 lines)

Pure Triton with `tl.associative_scan` for complex numbers. Same as scalar.py, should work on ROCm via Triton.

### ref.py (reference implementation - 122 lines)

Pure PyTorch, no CUDA primitives. No changes needed.

## Risk List

### HIGH RISK: Hardcoded warpSize = 32 in warp.cuh

The entire `warp.cuh` kernel is designed around 32-thread warps:
- `kNThreadsPerWarp = 32` is a compile-time constant
- All thread/block calculations assume 32-thread warps
- The shuffle-based scan within warps uses width-32 semantics
- Shared memory arrays `warpLastGate[kNWarpsPerBlock]` assume kNWarpsPerBlock = seqlen / (kNStepsPerThread * 32)

**Impact on wave64 (gfx90a)**:
- The 32-thread logical warp operations (`__shfl_up_sync` with delta up to 31) are actually valid on wave64 because HIP's warp shuffles support width=32 logical warps within a 64-lane wavefront.
- The critical issue is block configuration: with `kNThreadsPerWarp=32`, the kernel launches blocks sized for 32-thread warps. On gfx90a, these threads form partial 64-lane wavefronts.
- The level-2 scan (lines 169-189) that scans across warps within a block assumes `warpSize` = 32, but `warpSize` is a device variable = 64 on gfx90a. This causes the shuffle loop to iterate too many times, potentially reading garbage.

**Fix approach**:
1. Replace `constexpr int kNThreadsPerWarp = 32` with a template parameter or runtime query.
2. For wave64, either:
   - (a) Keep 32-lane logical warps (width=32 shuffles are valid) but fix the level-2 scan to use the actual warp count, OR
   - (b) Fully wave64-ize the kernel to use 64-lane shuffles
3. The 32-bit shuffle mask `0xffffffff` must become `0xffffffffffffffffULL` for HIP.

### MEDIUM RISK: Triton ROCm backend maturity

- Triton's `tl.associative_scan` primitive should work on ROCm, but less battle-tested than NVIDIA.
- The `enable_fp_fusion=False` flag is critical for numerical accuracy -- verify it's honored on ROCm.

### LOW RISK: load_inline() hipify

- torch's `load_inline()` with `cuda_sources` should auto-hipify on ROCm PyTorch.
- May need to add `extra_hip_cflags` alongside `extra_cuda_cflags` in warp.py.

### LOW RISK: bfloat16/half type mappings

- `__nv_bfloat16` -> `hip_bfloat16` (torch hipify handles this)
- `__half` is the same spelling in HIP

## File-by-File Change List

### accelerated_scan/warp.py

1. Add ROCm detection and set appropriate flags:
   ```python
   import torch
   is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
   ```

2. Add `extra_hip_cflags` for HIP builds (equivalent to existing `extra_cuda_cflags`).

3. Consider fallback to Triton implementation on ROCm if warp.cuh porting proves too complex for the initial pass.

### accelerated_scan/warp.cuh

**Critical changes for wave64 correctness**:

1. **Lines 140-141, 176-177**: Change shuffle mask from `0xffffffff` to a 64-bit literal for HIP:
   ```cpp
   #if defined(__HIP_PLATFORM_AMD__)
   #define FULL_WARP_MASK 0xffffffffffffffffULL
   #else
   #define FULL_WARP_MASK 0xffffffff
   #endif
   ```

2. **Line 175**: The loop `for (int delta = 1; delta < warpSize; delta *= 2)` uses the device `warpSize` variable. On gfx90a this is 64, causing extra iterations. Options:
   - Keep width-32 logical warps: change to `for (int delta = 1; delta < kNThreadsPerWarp; delta *= 2)`
   - OR full wave64: change `kNThreadsPerWarp` to 64 and all downstream calculations

3. **Line 280**: `constexpr int kNThreadsPerWarp = 32;` -- Keep this for now (32-lane logical warps work on wave64 via width parameter).

4. **Lines 1-2**: Add HIP compatibility:
   ```cpp
   #if defined(__HIP_PLATFORM_AMD__)
   #include <hip/hip_runtime.h>
   #else
   #include <cuda.h>
   #include <cuda_runtime.h>
   #endif
   ```

5. **Line 278**: `at::cuda::getCurrentCUDAStream()` -- torch handles this on ROCm builds.

6. **Lines 385-387**: Type dispatch `__nv_bfloat16` -> `hip_bfloat16`:
   ```cpp
   #if defined(__HIP_PLATFORM_AMD__)
   warpscan<hip_bfloat16, at::BFloat16>(gates, __VA_ARGS__);
   #else
   warpscan<__nv_bfloat16, at::BFloat16>(gates, __VA_ARGS__);
   #endif
   ```

### accelerated_scan/scalar.py

No changes expected. Triton should handle ROCm automatically. Test to verify `tl.associative_scan` works correctly.

### accelerated_scan/complex.py

No changes expected. Same as scalar.py.

### accelerated_scan/ref.py

No changes needed. Pure PyTorch.

## Build Commands

### Configure + Build (gfx90a)

```bash
# Activate ROCm PyTorch environment
source /path/to/rocm-pytorch-venv/bin/activate

# Set arch
export PYTORCH_ROCM_ARCH=gfx90a

# Install in editable mode (JIT compiles on first import)
pip install -e .

# Test import triggers JIT compilation
python -c "from accelerated_scan.warp import scan; print('warp import OK')"
python -c "from accelerated_scan.scalar import scan; print('scalar import OK')"
```

### Verify build for gfx90a

```bash
# Check device
rocm-smi --showproductname

# Simple smoke test
python -c "
import torch
from accelerated_scan.scalar import scan
gates = torch.rand(1, 512, 1024, device='cuda')
tokens = torch.rand(1, 512, 1024, device='cuda')
out = scan(gates, tokens)
print('output shape:', out.shape)
print('output sum:', out.sum().item())
"
```

## Test Plan

### Primary test suite

```bash
cd projects/accelerated-scan/src
pytest tests/test_eq.py -v
```

This runs:
- `test_eq_forward`: Forward pass correctness vs reference (multiple dtypes: float32, bfloat16, float16; multiple seqlens: 32 to 65536 + non-power-of-2)
- `test_eq_backward`: Backward pass gradient correctness vs reference
- `test_eq_ref_reverse`: Reference implementation reverse-scan correctness

### Triton-first validation strategy

1. **Phase 1**: Validate Triton implementations (scalar.py, complex.py) work on ROCm without any changes. These use `tl.associative_scan` which should be wave-size agnostic.

2. **Phase 2**: Port warp.cuh with the wave64 fixes above.

### Tests to run

```bash
# Run all tests with scalar (Triton) backend
pytest tests/test_eq.py -v

# Run complex tests
pytest tests/tests_eq_complex.py -v

# Benchmark (optional, for performance comparison)
python tests/bench.py
```

### Determinism check

Run tests twice with fixed seed to ensure deterministic results:
```bash
pytest tests/test_eq.py -v --tb=short 2>&1 | tee run1.log
pytest tests/test_eq.py -v --tb=short 2>&1 | tee run2.log
diff run1.log run2.log
```

### Non-GPU tests

None identified. All tests require GPU.

## Open Questions

1. **Triton ROCm associative_scan**: Does `tl.associative_scan` work correctly on ROCm? This is the lowest-risk path and should be validated first before investing in warp.cuh porting.

2. **warp.cuh necessity**: If scalar.py (Triton) performs well on ROCm, is the C++ warp kernel even needed for the AMD port? The benchmarks show Triton is competitive with the C++ kernel for most sequence lengths.

3. **Width-32 shuffles on wave64**: Confirm that HIP's `__shfl_up_sync` with width=32 works correctly within a 64-lane wavefront (expected: yes, operates on 32-lane logical warps).

4. **torch load_inline hipify behavior**: Verify torch's `load_inline()` auto-hipifies the `cuda_sources` correctly on a ROCm build. If not, manual hipify or a separate HIP source may be needed.
