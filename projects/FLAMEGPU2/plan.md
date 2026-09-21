# Plan: FLAMEGPU2

## Project

- **Name:** FLAMEGPU2
- **Upstream:** https://github.com/FLAMEGPU/FLAMEGPU2
- **Default branch:** main
- **Description:** GPU-accelerated agent-based modeling framework for CUDA C++ and Python

## Existing AMD support

**None found.** No AMD/ROCm/HIP support exists upstream or in any fork.

- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/` returned no matches
- Web search for "FLAMEGPU2 ROCm", "FLAME GPU 2 AMD GPU", "FLAMEGPU2 HIP" found no existing ports
- No forks with rocm/hip/amd in the name found via `gh api repos/FLAMEGPU/FLAMEGPU2/forks`
- No ROCm/HIP issues or PRs exist in the upstream repository
- The project's documentation explicitly requires "CUDA 12.0+" and an "NVIDIA GPU with Compute Capability >= 3.5"

**Decision:** Proceed with a from-scratch ROCm/HIP port -- this is valuable new AMD support for a widely-used agent-based modeling framework.

**NOTE:** This project presents a MAJOR porting complexity due to its use of **NVIDIA Jitify + nvrtc** for runtime kernel compilation (user-defined agent behavior is compiled at runtime). See Risk List below.

## Build classification

**Pure CMake** (Strategy A applies)

Evidence:
- `CMakeLists.txt:45-48`: `check_language(CUDA)` + `enable_language(CUDA)` -- no PyTorch dependency
- No `find_package(Torch)`, no `torch.utils.cpp_extension`, no CUDAExtension
- Project builds as a static library + executables with CMake's native CUDA language support
- Test suite uses GoogleTest, not pytest

## Port strategy

**Strategy A (pure CMake, compat-header approach) with significant Jitify/RTC challenges**

Rationale: This is a pure CMake project with CUDA sources. However, the standard Strategy A approach faces a major obstacle: FLAMEGPU2's core feature is **runtime compilation** of user-defined agent functions via Jitify2 + nvrtc.

The port has two major work items:
1. **AOT path:** The library and examples build offline -- standard Strategy A with a cuda_to_hip.h compat header and `LANGUAGE HIP` marking.
2. **RTC (Runtime Compilation) path:** Jitify uses NVRTC (NVIDIA Runtime Compiler) which has no direct ROCm equivalent. This requires replacing Jitify with hiprtc (HIP Runtime Compiler) or comgr (Code Object Manager).

### Proposed approach for RTC:

ROCm provides **hiprtc** (HIP Runtime Compilation) as the equivalent of nvrtc. The Jitify2 library is NVIDIA-specific and will not work. Options:

1. **Replace Jitify with hiprtc directly:** Rewrite JitifyCache.cu to use hiprtc APIs (`hiprtcCreateProgram`, `hiprtcCompileProgram`, `hiprtcGetCode`, etc.) on HIP. This is substantial work but cleanest.

2. **Conditional compilation:** Keep Jitify for CUDA, implement a parallel hiprtc path for HIP behind `#if defined(USE_HIP)`.

3. **Alternative: pre-compilation of common kernels:** If RTC is used only for performance (not user-defined code), pre-compile kernels AOT. However, FLAMEGPU2's RTC is a core feature for user-defined agent behaviors -- users write agent functions in C++/CUDA that are compiled at runtime.

**Recommendation:** Implement a dual-path JitifyCache that uses Jitify on CUDA and hiprtc on HIP. This is significant work but preserves the framework's core functionality.

## CUDA surface inventory

### Kernels and device functions
- ~50+ `__global__` kernel definitions across src/flamegpu/
- Many `__device__` helper functions
- `__host__ __device__` combined functions

### CUDA Runtime API
- ~546 uses of cudaStream/cudaEvent/cudaMalloc/cudaMemcpy/cudaDevice APIs
- Standard memory management: cudaMalloc, cudaFree, cudaMemcpy, cudaMemcpyAsync
- Stream management: cudaStream_t, cudaStreamCreate, cudaStreamSynchronize
- Device queries: cudaGetDeviceProperties, cudaDeviceGetAttribute

### cuRAND
- Heavy usage (~80 references)
- `curandStatePhilox4_32_10_t` / `curandStateMRG32k3a_t` / `curandStateXORWOW_t`
- `curand_init`, `curand_uniform`, etc.
- File: `include/flamegpu/detail/curand.cuh`, `src/flamegpu/simulation/detail/RandomManager.cu`
- **ROCm equivalent:** hipRAND (rocRAND) -- mostly 1:1 API mapping

### CUB
- ~55 uses of `cub::` namespace
- `cub::DeviceScan` (18 uses)
- `cub::DeviceRadixSort` (10 uses)
- **ROCm equivalent:** hipCUB (rocPRIM underneath) -- API compatible

### Thrust
- ~32 uses of `thrust::` namespace
- `thrust::device_vector`, sorting, etc.
- **ROCm equivalent:** rocThrust -- drop-in compatible (same headers)

### NVRTC / Jitify (CRITICAL)
- Jitify2 library for runtime kernel compilation
- Direct nvrtc.h includes and API calls
- NVRTC functions: `nvrtcCreateProgram`, `nvrtcCompileProgram`, `nvrtcGetNumSupportedArchs`, etc.
- Files: `src/flamegpu/detail/JitifyCache.cu`, `src/flamegpu/detail/compute_capability.cu`
- **ROCm equivalent:** hiprtc -- similar API but significant rewrite needed

### Compute Capability
- `cudaDevAttrComputeCapabilityMajor/Minor`
- `__CUDA_ARCH_LIST__` macro usage
- Files: `src/flamegpu/detail/compute_capability.cu`
- **Risk:** CC numbers collide between NVIDIA and AMD; need HIP-aware detection

### `__ldg` intrinsic
- 14 uses for cached reads
- **ROCm equivalent:** `__ldg` is available in HIP (same spelling)

### Textures
- `cudaBindTexture` references are commented out (not active)
- No active texture object usage found
- **Status:** Not a concern

### Synchronization
- `__syncthreads` (2 uses)
- No `__syncwarp` or `__ballot` usage found
- **Status:** No warp intrinsic concerns

## Risk list

### HIGH RISK

1. **Runtime Compilation (Jitify/nvrtc -> hiprtc):** FLAMEGPU2's core feature is runtime compilation of user-defined agent functions. Jitify is NVIDIA-specific and must be replaced with hiprtc-based implementation. This is substantial work (~500-1000 lines to rewrite JitifyCache.cu).

2. **Compute Capability collision:** The project uses compute capability for kernel selection and nvrtc architecture flags. AMD's hipDeviceAttributeComputeCapabilityMajor returns values that collide with NVIDIA (gfx90a reports cc=9 like Hopper). Need to guard arch selection with `__HIP_PLATFORM_AMD__`.

### MEDIUM RISK

3. **cuRAND state types:** The project allows configurable cuRAND generators (Philox, MRG32k3a, XORWOW). hipRAND supports these but state struct names differ slightly; verify compatibility.

4. **CUB DeviceRadixSort begin_bit:** PORTING_GUIDE notes hipCUB DeviceRadixSort with nonzero begin_bit may not sort correctly. Check if FLAMEGPU2 uses this.

5. **Warp size (wave64 vs wave32):** No explicit warp intrinsics found in the codebase, but the RTC-compiled user code could contain them. Document that user agent functions should use `warpSize` not literal 32.

### LOW RISK

6. **__ldg intrinsic:** Available in HIP; no change needed.

7. **Thrust/CUB:** rocThrust and hipCUB are drop-in replacements.

## File-by-file change list

### Core porting (Strategy A)

1. **New file: `include/flamegpu/detail/cuda_to_hip.h`**
   - Compat header with CUDA->HIP aliases
   - Include hipRAND mappings
   - Include hipCUB mappings

2. **`CMakeLists.txt`** (root)
   - Add `option(USE_HIP "Build with HIP for AMD GPUs" OFF)`
   - Gate `enable_language(CUDA)` vs `enable_language(HIP)`
   - Set `CMAKE_HIP_ARCHITECTURES` default to gfx90a

3. **`src/CMakeLists.txt`**
   - Add HIP language handling
   - Mark CUDA sources as `LANGUAGE HIP` when USE_HIP
   - Link hiprtc instead of CUDA::nvrtc
   - Link hipCUB instead of CUB
   - Link hipRAND instead of cuRAND

4. **`include/flamegpu/detail/curand.cuh`**
   - Add `#if defined(USE_HIP)` path for hipRAND includes
   - Map curandState types to hiprand equivalents

5. **`src/flamegpu/detail/compute_capability.cu`**
   - Replace nvrtc calls with hiprtc equivalents under USE_HIP
   - Replace cudaGetDeviceCount/Properties with hip equivalents
   - Handle AMD arch detection instead of compute capability

### Jitify replacement (major work)

6. **`src/flamegpu/detail/JitifyCache.cu`** (MAJOR REWRITE for HIP)
   - Replace Jitify2 with hiprtc on HIP path
   - `jitify2::Program` -> `hiprtcCreateProgram`
   - `program->preprocess()` -> custom preprocessing
   - `preprocessed_program->compile()` -> `hiprtcCompileProgram`
   - `compiled_program->link()` -> `hiprtcGetCode` + `hipModuleLoadData`
   - Cache management similar to current implementation

7. **`include/flamegpu/detail/JitifyCache.h`**
   - Abstract interface to support both Jitify and hiprtc backends

8. **`cmake/dependencies/Jitify.cmake`**
   - Gate Jitify fetch on NOT USE_HIP

9. **`cmake/common.cmake`**
   - Gate CUDA::nvrtc requirement on NOT USE_HIP
   - Add hip::hiprtc requirement for USE_HIP

### Tests

10. **`tests/CMakeLists.txt`**
    - Add HIP language support
    - Mark test .cu files as LANGUAGE HIP

11. **Test files (78 .cu files)**
    - Should work with compat header; may need minor fixes for hipRAND state types

## Build commands

### Configure (gfx90a)
```bash
cmake -S . -B build \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DFLAMEGPU_BUILD_TESTS=ON \
  -DFLAMEGPU_BUILD_ALL_EXAMPLES=ON \
  -DCMAKE_BUILD_TYPE=Release
```

### Build
```bash
cmake --build build -j$(nproc)
```

## Test plan

### GPU tests (must pass)
```bash
# Run full test suite
cd build && ctest -VV

# Or run directly
./bin/Release/tests --gtest_filter="*"
```

### Individual example validation
```bash
# Run boids example
./bin/Release/boids_spatial3D

# Run circles example
./bin/Release/circles_spatial3D
```

### RTC validation (critical)
```bash
# RTC examples compile and run user-defined agent functions
./bin/Release/rtc_boids_bruteforce
./bin/Release/rtc_boids_spatial3D
```

### Non-GPU tests
- The model/description tests are largely host-side
- Exception tests should pass

### Validation criteria
- All 78+ test files pass
- All examples execute without error
- RTC agent functions compile and execute correctly
- Random number generation produces valid distributions

## Open questions

1. **Jitify replacement scope:** Is a full hiprtc implementation required, or can RTC be optional on AMD? The framework's core value proposition is runtime-compiled agent functions, so disabling RTC would significantly limit the AMD port's usefulness.

2. **hiprtc API parity:** Are all nvrtc features used (preprocessing, architecture targeting, caching) available in hiprtc? Initial research suggests hiprtc has similar but not identical capabilities.

3. **User agent function compatibility:** User-written agent functions (in CUDA syntax) would need to be compatible with HIP. This means:
   - Users should not use NVIDIA-specific intrinsics
   - Warp primitives should use `warpSize` not 32
   - The documentation should note AMD compatibility requirements

4. **Python bindings (pyflamegpu):** The SWIG Python bindings link against nvrtc. The HIP port would need to link hiprtc and handle the differences.

5. **Visualisation support:** The optional FLAMEGPU2 visualiser uses OpenGL. This should work on AMD but needs verification.

6. **MPI ensemble support:** The MPI-parallel ensemble feature should be architecture-agnostic but needs testing.

## Summary

FLAMEGPU2 is a significant porting effort primarily due to its **runtime compilation** architecture. The AOT code (library, examples, tests) follows standard Strategy A patterns. The RTC subsystem requires replacing NVIDIA's Jitify/nvrtc with AMD's hiprtc.

Estimated effort:
- AOT porting: Medium (~1-2 days)
- hiprtc JitifyCache replacement: High (~3-5 days)
- Testing and validation: Medium (~1-2 days)

The port adds genuine value: FLAMEGPU2 is a mature, well-documented framework with academic and research users who would benefit from AMD GPU support.
