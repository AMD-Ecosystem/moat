# GOMC ROCm/HIP Port Plan

## Project

- **Name**: GOMC (GPU Optimized Monte Carlo)
- **Upstream**: https://github.com/GOMC-WSU/GOMC
- **Default branch**: main
- **Version**: 2.80 (Dec 2025)

GOMC is a parallel molecular simulation code for high-performance Monte Carlo simulations of phase equilibria using Metropolis Monte Carlo algorithms. It supports multiple ensembles (NVT, NPT, GCMC, GEMC) and runs on both CPUs (via OpenMP) and NVIDIA GPUs (via CUDA).

## Existing AMD support

**Decision: Proceed with HIP port (no existing native ROCm support)**

Searches performed:
- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/` -- no results
- WebSearch "<GOMC> ROCm", "<GOMC> AMD GPU" -- no native HIP port found
- GitHub forks: no ROCm/AMD/GPUOpen org forks
- Upstream PRs/issues: no AMD-related PRs

Third-party support found: SCALE (a CUDA translation layer, not a native HIP port) reportedly runs GOMC on AMD GPUs with minor patches (libc++ flag, OpenMP checks, atomicAdd architecture conflict). SCALE intercepts CUDA calls at runtime rather than porting to HIP. This does NOT constitute existing HIP/ROCm support.

The upstream deliberately supports "CPUs with or without OpenMP, and NVIDIA GPUs" -- no AMD-native path exists. A HIP port is the right approach.

## Build classification

**Classification: Pure CMake project (Strategy A)**

Evidence:
- Root `CMakeLists.txt` (lines 141-149): uses `check_language(CUDA)`, `enable_language(CUDA)`, no PyTorch/Torch dependency
- No `find_package(Torch)`, no `torch.utils.cpp_extension`, no setup.py
- GPU sources in `src/GPU/*.cu`, `src/GPU/*.cuh`
- Build via `metamake.sh` which wraps CMake

This is a standalone CMake + CUDA project with no PyTorch extension machinery.

## Port strategy

**Strategy A: Pure CMake (compat header + LANGUAGE HIP)**

Approach:
1. Add `src/GPU/cuda_to_hip.h` compat header with HIP aliases for all CUDA symbols used
2. Add `USE_HIP` CMake option; when ON, `enable_language(HIP)` + mark `.cu` files as `LANGUAGE HIP`
3. Replace `cub/cub.cuh` include with hipCUB: `#include <hipcub/hipcub.hpp>`
4. Replace `using namespace cub;` with `using namespace hipcub;`
5. Gate the `__CUDA_ARCH__ < 600` double atomicAdd polyfill under `!defined(__HIP_PLATFORM_AMD__)` (HIP natively supports double atomicAdd)
6. Handle NVTX profiling (disabled on HIP, or use ROCm's rocTX if desired)

Rationale: The codebase uses a small, focused CUDA surface (cudaMalloc/cudaMemcpy/cudaFree/cudaDeviceSynchronize, basic error checking, CUB DeviceReduce). No textures, no streams/events, no advanced features. A clean compat-header port is straightforward.

## CUDA surface inventory

### Kernel attributes
- `__global__` kernels: 10+ (TranslateParticles, RotateParticles, BrownianMotion, energy/force/Ewald kernels)
- `__device__` functions: ~50 (Calc*GPU energy/coulomb/force functions)
- `__shared__` memory: not explicitly used (CUB handles internally)
- `__syncthreads()`: not directly used (CUB handles internally)

### Warp intrinsics
**None found** -- no `warpSize`, `__shfl*`, `__ballot`, `__activemask`, or hardcoded 32/0x1f. CUB's DeviceReduce abstracts warp-level reductions.

### Memory operations
- `cudaMalloc` / `cudaFree` -- via CUMALLOC/CUFREE macros (CUDAMemoryManager wrapper)
- `cudaMemcpy` (Host<->Device) -- direct calls
- `cudaDeviceSynchronize` -- one call
- `cudaMemGetInfo` -- GPU memory reporting
- No managed memory, no pinned memory, no async memcpy

### Device properties
- `cudaGetDeviceCount`, `cudaGetDeviceProperties`, `cudaSetDevice` -- in Main.cpp
- `cudaDevAttrMemoryClockRate` -- CUDA 13+ path

### CUB library usage
- `#include "cub/cub.cuh"` (expects system CUB or bundled for CUDA < 11)
- `using namespace cub;`
- `DeviceReduce::Sum` -- used extensively for energy/force reductions
- `CubDebugExit` macro for error checking

### Atomic operations
- `atomicAdd(double*, double)` -- many calls for force/energy accumulation
- Custom double atomicAdd polyfill for `__CUDA_ARCH__ < 600`
- `atomicCAS` -- in the polyfill only

### Profiling (optional)
- NVTX markers via `<nvtx3/nvToolsExt.h>` when `GOMC_NVTX_ENABLED`
- `cudaProfilerStart/Stop`

### Error handling
- `cudaGetLastError`, `cudaGetErrorString`
- `gpuErrchk` macro wrapping `gpuAssert`

## Risk list

1. **CUB -> hipCUB**: Direct swap; hipCUB API matches CUB. Namespace change `cub::` -> `hipcub::`.

2. **Double atomicAdd polyfill**: The `#if __CUDA_ARCH__ < 600` guard must become `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600 && !defined(__HIP_PLATFORM_AMD__)`. HIP natively supports double atomicAdd on all AMD architectures.

3. **NVTX profiling**: ROCm equivalent is rocTX (`<roctracer/roctx.h>`). For now, disable under HIP; profiling is optional (`-DGOMC_NVTX_ENABLED` off by default).

4. **CUDART_VERSION checks**: `CUDART_VERSION < 13000` check for memoryClockRate needs HIP equivalent. HIP defines `HIP_VERSION` but the memoryClockRate path is simpler on HIP (prop.memoryClockRate always available).

5. **CUDA_SEPARABLE_COMPILATION**: GPU targets use `CUDA_SEPARABLE_COMPILATION ON`. HIP equivalent is `HIP_SEPARABLE_COMPILATION` for device-code linking. Verify this works with hipCUB DeviceReduce.

6. **Architecture detection**: GOMCCUDASetup.cmake hardcodes `CMAKE_CUDA_ARCHITECTURES`. The HIP build should use `CMAKE_HIP_ARCHITECTURES` with gfx90a as default, but accept any value for follower platforms.

7. **warpSize safety**: While no explicit warp primitives are used, CUB/hipCUB internally handle wave32/wave64 differences. hipCUB is tested on both CDNA (wave64) and RDNA (wave32). No code changes needed.

8. **Test linking**: GPU gtests link against gtest_main. Ensure CMake HIP language properly links C++ and HIP objects. May need explicit `target_link_libraries` adjustments.

## File-by-file change list

### New files
- `src/GPU/cuda_to_hip.h` -- CUDA-to-HIP compat header

### Modified files
- `CMakeLists.txt` -- add `USE_HIP` option, HIP language enable
- `CMake/GOMCCUDASetup.cmake` -> `CMake/GOMCGPUSetup.cmake` -- rename and split CUDA/HIP paths, or add HIP counterpart `CMake/GOMCHIPSetup.cmake`
- `src/GPU/VariablesCUDA.cuh` -- include `cuda_to_hip.h` instead of raw CUDA headers
- `src/GPU/CUDAMemoryManager.cuh` -- include compat header
- `src/GPU/CUDAMemoryManager.cu` -- include compat header
- `src/GPU/CalculateEnergyCUDAKernel.cu` -- change `#include "cub/cub.cuh"` to hipCUB, `using namespace hipcub;` under HIP
- `src/GPU/CalculateForceCUDAKernel.cu` -- same CUB -> hipCUB changes
- `src/GPU/CalculateEwaldCUDAKernel.cu` -- same CUB -> hipCUB changes
- `src/GPU/CalculateMinImageCUDAKernel.cuh` -- guard double atomicAdd polyfill for CUDA-only
- `src/GPU/*.cu, *.cuh` (all) -- include compat header at top
- `src/Main.cpp` -- ifdef device enumeration for HIP (hipGetDeviceCount, hipGetDeviceProperties, hipSetDevice)
- `src/GOMCEventsProfile.h` -- disable NVTX under HIP or add rocTX path
- `test/BuildGPUTests.cmake` -> `test/BuildHIPTests.cmake` (add) -- HIP test build

### Keep unchanged
- All CPU-only `.cpp` files
- `lib/` directory
- `test/src/` unit test source files (they test CPU logic)

## Build commands

### Configure (gfx90a)
```bash
mkdir build && cd build
cmake .. \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DGOMC_GTEST=ON
```

### Build
```bash
make -j$(nproc) GPU_NVT GPU_NPT GPU_GCMC GPU_GEMC
```

Or via metamake (once modified for HIP):
```bash
./metamake.sh GPU
```

### Quick validation build (one ensemble)
```bash
make -j$(nproc) GPU_NVT
```

## Test plan

### Unit tests (non-GPU, must not regress)
```bash
make -j$(nproc) GOMC_CPU_NVT_Test GOMC_CPU_NPT_Test GOMC_CPU_GCMC_Test GOMC_CPU_GEMC_Test
ctest --output-on-failure
```

### GPU unit tests
```bash
make -j$(nproc) GOMC_GPU_NVT_Test GOMC_GPU_NPT_Test GOMC_GPU_GCMC_Test GOMC_GPU_GEMC_Test
ctest --output-on-failure
```

### Integration tests (GPU validation)
The upstream uses `GOMC_Examples` integration tests:
```bash
cd test
bash Setup_Examples.sh  # clones GOMC_Examples, builds both branches
python Run_Examples.py   # runs all examples, compares new vs ref
cat integration/IntegrationTest.log
```

For HIP validation, a simpler approach:
1. Build GPU executables (GOMC_GPU_NVT, GOMC_GPU_NPT, GOMC_GPU_GCMC, GOMC_GPU_GEMC)
2. Clone GOMC_Examples: `git clone https://github.com/GOMC-WSU/GOMC_Examples`
3. Run a representative example from each ensemble:
   - NVT: `./GOMC_GPU_NVT +p4 GOMC_Examples/NVT/<example>/in.conf`
   - NPT: `./GOMC_GPU_NPT +p4 GOMC_Examples/NPT/<example>/in.conf`
   - Compare energy outputs to CPU reference (within Monte Carlo statistical tolerance)

### Determinism check
Run the same simulation twice with identical random seeds; outputs should be bit-identical.

## Open questions

1. **Integration test timing**: The full GOMC_Examples suite can take significant time. Identify a minimal subset for CI validation (e.g., one fast example per ensemble).

2. **HIP device linking**: Verify `CUDA_SEPARABLE_COMPILATION` equivalent (`set_property(TARGET ... PROPERTY HIP_SEPARABLE_COMPILATION ON)` or rely on hipcc's default behavior with device code).

3. **MPI support**: GOMC supports MPI for parallel tempering. The HIP port should maintain MPI compatibility; no MPI-specific CUDA code identified, but verify MPI + HIP interop in a follow-up.

4. **NVTX -> rocTX**: Decide whether to add rocTX profiling support or simply disable profiling on HIP. rocTX is available in ROCm (`roctracer`).
