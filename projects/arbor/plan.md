# Plan: arbor

## Project

- **Name**: arbor
- **Upstream**: https://github.com/arbor-sim/arbor
- **Default branch**: main
- **Description**: High-performance library for computational neuroscience simulations with morphologically-detailed neurons

## Existing AMD Support

**Status: Authoritative native HIP backend exists upstream -- needs validation and improvement**

Arbor has native HIP support introduced in [PR #1007](https://github.com/arbor-sim/arbor/pull/1007) (merged April 2020), maintained by the arbor-sim project. The documentation explicitly states "experimental support for AMD GPUs" and lists tested hardware as Mi50/Mi60 (gfx906).

Evidence of native HIP support:
- `arbor/include/arbor/gpu/hip_api.hpp` -- HIP-specific API wrapper
- `arbor/include/arbor/gpu/cuda_api.hpp` -- CUDA-specific API wrapper  
- `arbor/include/arbor/gpu/gpu_api.hpp` -- unified GPU API dispatcher
- CMake `ARB_GPU=hip` option with `ARB_HIP_ARCHITECTURES` cache variable

The HIP backend is authoritative (maintained by upstream) but has known deficiencies:
1. Hardcoded wave64 assumption breaks RDNA (gfx1100/gfx1151)
2. Default architectures are outdated (gfx906/gfx900)
3. Not tested on current ROCm (7.x) or MI200-class (gfx90a) hardware

**Decision**: Validate and improve the existing HIP backend, not a from-scratch port.

## Build Classification

**Type**: Pure CMake (Strategy A not applicable -- HIP already integrated)

Evidence (CMakeLists.txt):
- Line 190: `elseif(ARB_GPU STREQUAL "hip")`
- Line 191: `set(ARB_WITH_HIP_CLANG TRUE)`
- Line 194: `set(ARB_HIP_ARCHITECTURES gfx906 gfx900 CACHE STRING ...)`
- No PyTorch/Torch dependencies

## Port Strategy

**Strategy: Validate existing HIP backend, fix wave-size bug, update arch defaults**

This is not a CUDA-to-HIP translation -- the translation was already done upstream. The MOAT value is:
1. Build and test on gfx90a with ROCm 7.x
2. Fix the wave-size hardcode that breaks RDNA platforms
3. Update default architectures to include modern hardware

## CUDA/HIP Surface Inventory

Already ported upstream. Key abstractions:

| Component | Location | Notes |
|-----------|----------|-------|
| GPU API wrapper | `arbor/include/arbor/gpu/{hip,cuda}_api.hpp` | Runtime API (malloc, memcpy, sync) |
| Warp primitives | `{hip,cuda}_api.hpp:120-150` | `ballot`, `shfl`, `shfl_up/down`, `any`, `active_mask` |
| Atomics | `{hip,cuda}_api.hpp:100-120` | `atomicAdd` for float/double |
| Warp size | `gpu_common.hpp:18-24` | **HARDCODED 64 for HIP -- BUG** |
| Reduce by key | `reduce_by_key.hpp` | Uses `threads_per_warp()` -- affected by bug |

GPU kernels (15 `.cu` files):
- `arbor/backends/gpu/*.cu` -- matrix operations, diffusion, threshold, stimulus
- `arbor/memory/fill.cu` -- memory utilities
- Test files: `test/unit/*.cu`, `test/ubench/*.cu`

## Risk List

1. **CRITICAL: Wave-size hardcode breaks RDNA**
   - Location: `arbor/include/arbor/gpu/gpu_common.hpp:18-24`
   - Bug: `threads_per_warp()` returns 64 for ALL HIP builds
   - Impact: `reduce_by_key.hpp` uses this for warp-level reductions; wrong on wave32
   - Fix: Use per-arch `__GFX9__` guard (64) vs default (32), per PORTING_GUIDE

2. **Outdated default architectures**
   - Location: CMakeLists.txt:194
   - Current: `gfx906 gfx900` (Vega-era, 2017-2019)
   - Fix: Update to include `gfx90a` (MI200), `gfx1100` (RDNA3)

3. **ROCm version compatibility**
   - Docs reference non-release hip-clang; ROCm has matured significantly since
   - The `pow(double, int)` workaround was already removed (PR #1247)
   - Verify clean build on ROCm 7.x

4. **HIP warp intrinsics signature differences**
   - `hip_api.hpp` uses `__shfl(x, lane)` without mask parameter
   - `hip_api.hpp` uses `__ballot(is_root)` without mask
   - These map correctly to HIP intrinsics but differ from CUDA signatures

5. **Generated mechanism code**
   - `modcc/printer/gpuprinter.cpp` generates code using `reduce_by_key`
   - Wave-size fix propagates through generated code automatically

## File-by-File Change List

### Fix wave-size hardcode

`arbor/include/arbor/gpu/gpu_common.hpp` (lines 17-24):
```cpp
// Current (broken):
#ifdef ARB_HIP
    return 64u;
#else
    return 32u;
#endif

// Fix:
#if defined(ARB_HIP)
  #if defined(__GFX9__)
    return 64u;   // CDNA: gfx90a, gfx94x (wave64)
  #else
    return 32u;   // RDNA: gfx10xx, gfx11xx (wave32)
  #endif
#else
    return 32u;   // CUDA
#endif
```

### Update default architectures

`CMakeLists.txt` (line 194):
```cmake
# Current:
set(ARB_HIP_ARCHITECTURES gfx906 gfx900 CACHE STRING ...)

# Fix:
set(ARB_HIP_ARCHITECTURES gfx90a CACHE STRING ...)
```

(Keep as cache variable so followers can override without source change)

## Build Commands

### Configure (gfx90a)
```bash
cd projects/arbor/src
mkdir -p build && cd build
export CC=clang
export CXX=hipcc
cmake .. \
  -DARB_GPU=hip \
  -DARB_HIP_ARCHITECTURES="gfx90a" \
  -DARB_WITH_PYTHON=OFF \
  -DCMAKE_BUILD_TYPE=Release
```

### Build
```bash
cmake --build . -j$(nproc)
```

### Multi-arch build (validation)
```bash
cmake .. -DARB_HIP_ARCHITECTURES="gfx90a;gfx1100"
```

## Test Plan

### GPU Tests
```bash
cd build
# Run all unit tests
ctest --output-on-failure

# GPU-specific tests (from test/unit/CMakeLists.txt):
./bin/unit --gtest_filter="*gpu*"
./bin/unit --gtest_filter="*reduce_by_key*"
./bin/unit --gtest_filter="*event_stream*"
./bin/unit --gtest_filter="*stack*"
./bin/unit --gtest_filter="*vector*"
./bin/unit --gtest_filter="*spikes*"
```

### Non-GPU regression set
```bash
# All non-GPU unit tests
./bin/unit --gtest_filter="-*gpu*"

# Python tests (if ARB_WITH_PYTHON=ON)
pytest ../python/test/
```

### Validation criteria
1. Build succeeds for gfx90a
2. GPU unit tests pass (especially reduce_by_key which exercises warp ops)
3. Non-GPU tests do not regress
4. Multi-arch build (gfx90a;gfx1100) produces valid code objects for both

## Open Questions

1. **Python bindings**: Should we enable `ARB_WITH_PYTHON=ON` for validation? It adds pybind11 complexity but is the primary user interface.

2. **MPI support**: `ARB_WITH_MPI=ON` enables distributed simulation. Skip for initial validation?

3. **Performance baseline**: No NVIDIA reference to compare against on this hardware. Focus on correctness (tests pass) not perf parity.

4. **Upstream PR strategy**: The wave-size fix is a clear bug fix. Should we validate first then PR, or open an issue immediately?
