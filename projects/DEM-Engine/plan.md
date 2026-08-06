# DEM-Engine ROCm Port Plan

## Project

- **Name**: DEM-Engine (DEME)
- **Upstream**: https://github.com/projectchrono/DEM-Engine
- **Default branch**: main
- **Description**: A dual-GPU Discrete Element Method solver with complex grain geometry support

## Existing AMD support

**Finding**: No existing AMD/ROCm/HIP support found.

- Grepped upstream docs (README*, docs/) for `amd|rocm|hip|gfx`: no results
- Searched web for "DEM-Engine ROCm", "DEM-Engine AMD GPU", "DEM-Engine HIP": no results
- Checked GitHub forks via API: 29 forks, none with rocm/hip/amd in name
- No AMD-related PRs or issues
- No upstream rocm/hip branches
- No USE_HIP or `__HIP` guards in codebase

**Decision**: Proceed with a from-scratch ROCm/HIP port.

## Build classification

**Classification**: Pure CMake project (Strategy A)

**Evidence** (CMakeLists.txt:17-21, 44):
```cmake
project(
    Chrono-DEM-Engine
    VERSION ${DEME_VERSION_MAJOR}.${DEME_VERSION_MINOR}.${DEME_VERSION_PATCH}
    LANGUAGES CXX CUDA
)
...
find_package(CUDAToolkit REQUIRED)
```

No `find_package(Torch)`, no `torch.utils.cpp_extension`, no `CUDAExtension`. This is a standalone CMake project with .cu sources and CUDA runtime/driver libraries.

## Port strategy

**Strategy A (pure CMake)** with modifications for runtime compilation.

**Key challenge**: The project uses NVIDIA's **Jitify** library for runtime kernel compilation (NVRTC). This is the defining architectural feature -- kernels are compiled at runtime, not at build time, to allow user-customizable force models and particle definitions.

**Solution**: ROCm provides a HIP-ported Jitify at https://github.com/ROCm/jitify. This is a single-header library compatible with hipRTC that aims to maintain API compatibility with NVIDIA's Jitify.

**Rationale**:
1. The project compiles `.cu` kernel sources at runtime using Jitify/NVRTC
2. All kernel sources use CUDA spelling (`__global__`, `cudaStream_t`, etc.)
3. ROCm's Jitify port can compile HIP sources at runtime using hipRTC
4. A `cuda_to_hip.h` compat header can alias CUDA symbols to HIP for both AOT and JIT-compiled code

## CUDA surface inventory

### Runtime compilation (Jitify/NVRTC)
- `jitify::JitCache`, `jitify::Program` (src/core/utils/JitHelper.{h,cpp})
- Program::kernel().configure().launch() pattern throughout
- Runtime-compiled kernel files: `src/kernel/*.cu` (19 files)
- Customizable policies: `src/kernel/DEMCustomizablePolicies/*.cu` (17 files)
- User scripts: `src/kernel/DEMUserScripts/*.cu` (7 files)

### CUB library
- Device algorithms (src/algorithms/DEMCubWrappers.cu):
  - `cub::DeviceSelect::Flagged`
  - `cub::DeviceScan::ExclusiveScan`
  - `cub::DeviceRadixSort::SortPairs`
  - `cub::DeviceSelect::Unique`
  - `cub::DeviceRunLengthEncode::Encode`
  - `cub::DeviceReduce::ReduceByKey`
  - `cub::DeviceReduce::Reduce`
  - `cub::DeviceReduce::Max`
  - `cub::DeviceReduce::Min`
- Block collectives (commented out in contact kernels)

### CUDA Runtime API
- Memory: `cudaMalloc`, `cudaMallocManaged`, `cudaFree`, `cudaMallocHost`, `cudaHostAlloc`, `cudaFreeHost`
- Streams: `cudaStream_t`, `cudaStreamCreate`, `cudaStreamDestroy`, `cudaStreamSynchronize`
- Memory transfer: `cudaMemcpy`, `cudaMemPrefetchAsync`
- Device: `cudaGetDevice`, `cudaGetDeviceProperties`, `cudaDeviceProp`
- Error handling: `cudaError_t`, `cudaSuccess`

### CUDA Driver API (via Jitify)
- Indirect through jitify: `cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel`

### Library includes (JIT)
- `#include <curand_kernel.h>` (in kernel_includes string for user force models)

### Atomic operations
- `atomicAdd` (float, int) - used extensively in force collection and contact detection

### No warp intrinsics
- No `__shfl*`, `__ballot`, `__activemask`, `warpSize` found
- No hardcoded warp size 32

### No textures/surfaces
- No texture or surface memory usage

## Risk list

1. **Jitify HIP port maturity (HIGH)**
   - ROCm's Jitify is marked "early-access technology preview"
   - Some features are partially supported due to hipRTC issues
   - Windows is not supported (Linux-only for ROCm)
   - Must validate that DEM-Engine's usage patterns work with ROCm Jitify

2. **hipRTC include paths (MEDIUM)**
   - Jitify passes `-I` flags for CUDA toolkit headers
   - JitHelper.cpp hardcodes CUDA paths (`CUDA_HOME`, `/usr/local/cuda`)
   - Must add ROCm paths for hipRTC (`/opt/rocm/include`, etc.)
   - Must ensure `cuda_to_hip.h` compat header is available to JIT compiler

3. **curand_kernel.h for JIT (MEDIUM)**
   - User force models may include `curand_kernel.h` at runtime
   - Need to map to `hiprand_kernel.h` for hipRTC
   - Or provide a shim header

4. **cudaMemPrefetchAsync API change (LOW)**
   - CUDA 13.0+ changed the signature (5-arg vs 4-arg)
   - HIP uses the 4-arg form (device int, not cudaMemLocation)
   - Code already handles both via `CUDART_VERSION` check

5. **atomicMin/atomicMax on managed memory (LOW)**
   - Per PORTING_GUIDE: silently dropped on coarse-grained memory on gfx90a
   - Project uses atomicAdd primarily, which is unaffected
   - Verify no atomicMin/atomicMax on managed memory paths

6. **Multi-GPU dual-GPU mode (LOW)**
   - Project is designed for dual-GPU operation
   - Need to verify hipSetDevice works correctly for peer access

## File-by-file change list

### New files
1. `src/core/utils/cuda_to_hip.h` - CUDA-to-HIP compat header with symbol aliases for:
   - Runtime API symbols (`cudaMalloc` -> `hipMalloc`, etc.)
   - Stream/event types
   - Error types
   - `#define cub hipcub` for CUB -> hipCUB
   - Device attribute constants

2. `src/core/utils/cuda_to_hip_jit.h` - Slimmer compat header for JIT-compiled kernels:
   - Device-side CUDA->HIP aliases
   - `curand_kernel.h` -> `hiprand_kernel.h` mapping

### Modified files

1. **CMakeLists.txt** (root)
   - Add `option(USE_HIP "Build with HIP for AMD GPUs" OFF)`
   - Conditional `enable_language(HIP)` vs `enable_language(CUDA)`
   - Find hipCUB instead of CUB on HIP
   - Link `hip::hiprtc` and `hip::amdhip64` instead of CUDA libraries
   - Set `CMAKE_HIP_ARCHITECTURES` (default gfx90a when unset)

2. **src/core/CMakeLists.txt**
   - Conditional Jitify source: NVIDIA/jitify vs ROCm/jitify
   - Install cuda_to_hip.h and cuda_to_hip_jit.h

3. **src/algorithms/CMakeLists.txt**
   - Mark .cu sources as `LANGUAGE HIP` when USE_HIP
   - Link hipCUB instead of CUB

4. **src/DEM/CMakeLists.txt**
   - Mark .cu sources as `LANGUAGE HIP`

5. **src/core/utils/JitHelper.cpp**
   - Include cuda_to_hip.h (for runtime API aliases)
   - Add ROCm include paths for hipRTC
   - Pass cuda_to_hip_jit.h as a Jitify header for JIT compilation
   - Change arch detection to use `hipGetDeviceProperties` and `gcnArchName`
   - Change `-arch=compute_XY` to `--gpu-architecture=gfx90a` style

6. **src/core/utils/CudaAllocator.hpp**
   - Include cuda_to_hip.h

7. **src/core/utils/ManagedMemory.hpp**
   - Include cuda_to_hip.h
   - Adjust `cudaMemPrefetchAsync` signature handling

8. **src/core/utils/GpuManager.{h,cpp}**
   - Include cuda_to_hip.h

9. **src/algorithms/DEMCubWrappers.cu**
   - Include cuda_to_hip.h
   - `#define cub hipcub` via compat header handles CUB->hipCUB

10. **src/DEM/API.h**
    - Change `#include <curand_kernel.h>` to conditional for HIP

11. **src/kernel/*.cu** (runtime-compiled kernels)
    - No changes needed if cuda_to_hip_jit.h is passed to Jitify
    - The JIT compiler will translate CUDA spelling via the compat header

### Submodule change
- **thirdparty/jitify** - Change from NVIDIA/jitify to ROCm/jitify (or add conditional handling)

## Build commands

### Configure (Linux gfx90a)
```bash
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx90a \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
```

### Build
```bash
cmake --build . -j$(nproc)
```

### Follower archs
```bash
# gfx1100
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 ...
# Windows gfx1101/gfx1201 - not supported by ROCm Jitify (Linux-only)
```

## Test plan

### No formal test suite
The project has no gtest/ctest infrastructure. Validation relies on demo programs.

### GPU validation demos
The following demos exercise core functionality and should be run for validation:

1. **DEMdemo_SingleSphereCollide** - Basic contact detection, force calculation
2. **DEMdemo_BallDrop** - Gravity, collision, settling
3. **DEMdemo_Repose** - Angle of repose, steady-state
4. **DEMdemo_ContactChain** - Force propagation through contacts
5. **DEMdemo_RotatingDrum** - Dynamic simulation with clumps

### Validation criteria
- Demos compile and link
- Demos run without GPU errors/crashes
- Visual/physical plausibility of output (qualitative)
- Compare particle counts, energy conservation against CUDA reference if available

### Non-GPU tests
None identified. All functionality requires GPU.

## Open questions

1. **ROCm Jitify stability**: The ROCm Jitify fork is marked "early-access". Need to validate:
   - Does it handle the `program.kernel("name").configure().launch()` pattern?
   - Does it correctly pass headers to hipRTC?
   - Does it handle include path resolution?

2. **Windows support**: ROCm Jitify explicitly states "Windows unsupported". This may block the Windows gfx1101/gfx1201 platforms, requiring a potential redesign (AOT compilation of kernels for Windows).

3. **curand for JIT**: User force models can include curand. Need to:
   - Verify hiprand_kernel.h works with hipRTC
   - May need to map curand types/functions in the JIT compat header

4. **Dual-GPU operation**: Project is designed for dual-GPU. Verify HIP peer access works correctly on MI250 (two GCDs).

5. **Jitify submodule management**: How to handle the NVIDIA/jitify vs ROCm/jitify submodule? Options:
   - Fork ROCm/jitify to jeffdaily and make that the submodule
   - Add ROCm/jitify as a separate path with CMake conditional
   - Fetch ROCm/jitify via FetchContent when USE_HIP=ON
