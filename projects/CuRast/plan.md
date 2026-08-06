# CuRast ROCm Port Plan

## Project

- **Name**: CuRast
- **Upstream**: https://github.com/m-schuetz/CuRast
- **Default branch**: main
- **Description**: CUDA-based software rasterization for billions of triangles, a 3-stage pipeline that outperforms Vulkan HW rasterization for small/dense triangles (Nanite-style workloads)

## Existing AMD support

**None found**. No AMD/ROCm references in the upstream README or docs. No ROCm/HIP forks in the fork list. No upstream issues or PRs mentioning AMD. Web searches for "CuRast ROCm/AMD/HIP" returned no results. The related compute_rasterizer project by the same author explicitly states "Requires Windows and NVIDIA GPUs. Pull requests for AMD support are welcome."

**Decision**: Proceed with a from-scratch ROCm/HIP port.

## Build classification

**Pure CMake project** (Strategy A)

Evidence:
- `CMakeLists.txt` line 4: `project(${TARGET_NAME} LANGUAGES CXX CUDA)` -- native CMake CUDA, no PyTorch dependency
- No `find_package(Torch)`, no `torch.utils.cpp_extension`, no setup.py, no pyproject.toml
- Uses CUDA Driver API (`cuda.h`, `cuLaunchKernel`, `cuModuleLoadData`) plus nvrtc and nvJitLink for runtime compilation
- Dependencies: GLFW, Vulkan, GLM, ImGui, TurboJPEG (all pure C/C++ or headers), LASzip (CMake subdirectory)

**ext_type**: `cmake`

## Port strategy

**Strategy A** with significant library substitutions:

1. Add `src/cuda_to_hip.h` compat header aliasing CUDA spellings to HIP
2. Gate language on `USE_HIP` CMake option; mark `.cu` files `LANGUAGE HIP`
3. Substitute CUDA Driver API -> HIP Driver API (`hipModuleLoadData`, `hipLaunchKernel`, etc.)
4. Substitute nvrtc/nvJitLink -> hiprtc for runtime compilation
5. Replace cooperative_groups with HIP cooperative groups (mostly compatible, but `memcpy_async` needs attention)
6. Replace curand_kernel -> hiprand_kernel (jpeg.cu only, may be commented out)
7. Handle CUDA-Vulkan interop -> HIP-Vulkan interop (HIP has `hipExternalMemoryHandleTypeOpaqueWin32` equivalents)
8. Handle surf2D surface objects -> HIP surface write equivalents

## CUDA surface inventory

### Kernel files (`.cu`)
| File | Symbols/Features |
|------|------------------|
| `src/kernels/triangles_visbuffer.cu` | `cooperative_groups` (this_grid, this_thread_block, tiled_partition<32>), `warp.shfl`, `warp.ballot`, `atomicMin`, `atomicAdd`, `__fdividef`, `__float_as_uint`, `__uint_as_float`, `__constant__`, memcpy_async include (unused in main code) |
| `src/kernels/resolve.cu` | `cooperative_groups`, `surf2Dwrite`, `cudaSurfaceObject_t`, `atomicAdd`, `atomicOr`, `atomicMin`, `atomicMax`, `__shfl`, `__ldg`, `__ffs`, warp collective: `match_any` |
| `src/kernels/triangles_common.cuh` | `cooperative_groups`, `atomicAdd`, same patterns |
| `src/kernels/lines.cu` | `cooperative_groups`, memcpy_async include |
| `src/kernels/textureTools.cu` | Standard CUDA patterns |
| `src/kernels/triangles_heightmap.cu` | `cooperative_groups`, memcpy_async include |
| `src/jpeg/jpeg.cu` | `curand_kernel.h` (may be commented), `cooperative_groups`, `tiled_partition<32>`, `warp.shfl`, `warp.ballot`, `__ldg`, `__ffs` |

### Host-side CUDA usage (C++ headers)
| File | API |
|------|-----|
| `CudaModularProgram.h` | nvrtc (`nvrtcCreateProgram`, `nvrtcCompileProgram`, `nvrtcGetLTOIR`), nvJitLink (`nvJitLinkCreate`, `nvJitLinkAddData`, `nvJitLinkComplete`), cuModule (`cuModuleLoadData`, `cuLaunchKernel`, `cuLaunchCooperativeKernel`), `CUmodule`, `CUfunction`, `CUdeviceptr` |
| `CURuntime.h` | `cuda.h`, `cuda_runtime.h`, `cuCtxGetDevice`, `cuDeviceGetAttribute` |
| `Timer.h` | `CUevent`, `cuEventCreate`, `cuEventRecord`, `cuEventElapsedTime` |
| `CudaVulkanSharedMemory.h` | Virtual memory API (`cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, `cuMemExportToShareableHandle`), Vulkan external memory (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT`) |
| `VulkanCudaSharedMemory.h` | Vulkan-CUDA interop (similar) |
| `CudaVirtualMemory.h` | CUDA virtual memory management |
| `main.cpp` | `cuInit`, `cuCtxCreate`, `cuDeviceGet` |

### Warp intrinsics
- `tiled_partition<32>` -- width-32 explicitly, safe on wave64 (operates as 32-lane subgroup)
- `warp.shfl(..., 0)` / `warp.shfl(..., winningLane)` -- width-32 shuffles, needs verification
- `warp.ballot()` -- returns `uint32_t` mask from width-32 partition, OK
- `cg::coalesced_threads().match_any(key)` -- cooperative groups match, HIP supports this
- `__ffs(mask)` -- find first set bit in 32-bit mask, HIP has this

### Surfaces
- `surf2Dwrite(color, gl_desktop, x * 4, y)` -- surface write to CUDA-Vulkan shared texture
- `cudaSurfaceObject_t` -- surface object handle

### CUDA Driver API mapping
| CUDA | HIP |
|------|-----|
| `cuInit` | `hipInit` |
| `cuCtxCreate` | `hipCtxCreate` |
| `cuDeviceGet` | `hipDeviceGet` |
| `cuModuleLoadData` | `hipModuleLoadData` |
| `cuLaunchKernel` | `hipModuleLaunchKernel` |
| `cuLaunchCooperativeKernel` | `hipModuleLaunchCooperativeKernel` |
| `cuEventCreate/Record/Elapsed` | `hipEventCreate/Record/ElapsedTime` |
| `cuMemAddressReserve` | `hipMemAddressReserve` |
| `cuMemCreate` | `hipMemCreate` |
| `cuMemMap` | `hipMemMap` |
| `cuMemExportToShareableHandle` | `hipMemExportToShareableHandle` |

### Runtime compilation (nvrtc -> hiprtc)
| nvrtc | hiprtc |
|-------|--------|
| `nvrtcCreateProgram` | `hiprtcCreateProgram` |
| `nvrtcCompileProgram` | `hiprtcCompileProgram` |
| `nvrtcGetLTOIR` | N/A -- HIP uses code objects, not LTO IR |
| `nvrtcGetPTX` | `hiprtcGetCode` (returns HSACO) |
| nvJitLink | N/A -- HIP compiles directly to device code, no separate linking step for non-RDC |

**Note**: The nvrtc + nvJitLink LTO IR workflow does not have a direct HIP equivalent. HIP compiles `.hip` sources directly to device code (HSACO) using hiprtc. This requires restructuring the runtime compilation path.

## Risk list

### High risk

1. **Runtime compilation (nvrtc/nvJitLink)**: The project uses nvrtc to compile `.cu` sources at runtime to LTO IR, then links with nvJitLink. HIP's hiprtc compiles directly to device code objects; there is no LTO IR intermediate or separate link step. The `CudaModularProgram.h` abstraction must be redesigned for HIP's single-step compilation model.

2. **CUDA-Vulkan interop**: The virtual memory sharing (`cuMemExportToShareableHandle` -> Vulkan external memory) uses Win32 handle types. HIP supports similar APIs (`hipMemExportToShareableHandle`) but the exact handle types and Vulkan integration need verification on Linux with AMD GPUs. This is platform-specific and will differ between Windows and Linux.

3. **Cooperative kernels (`cuLaunchCooperativeKernel`)**: HIP supports cooperative kernels (`hipModuleLaunchCooperativeKernel`) but there may be occupancy differences. The grid-wide synchronization (`grid.sync()`) patterns are used throughout the rasterization pipeline.

### Medium risk

4. **Surface write (`surf2Dwrite`)**: HIP supports surface operations but the CUDA-Vulkan surface interop path for the OpenGL-shared desktop texture needs porting. This interfaces with the Vulkan swapchain.

5. **Warp size assumptions**: The code explicitly uses `tiled_partition<32>`, which is correct for both wave64 and wave32 (operates as a 32-lane logical warp). The ballot masks are 32-bit matching the partition width. However, verify no implicit warpSize=32 assumptions in shared memory sizing or launch configs.

6. **`__ldg` intrinsic**: Load through texture cache. HIP provides `__ldg` in recent ROCm versions, but verify availability. Can fall back to regular loads if unavailable.

### Low risk

7. **cooperative_groups::memcpy_async**: Included but appears commented/unused in main kernels. If used, HIP has `cg::memcpy_async` equivalents.

8. **curand_kernel.h**: Only included in jpeg.cu; appears to be for future use or debugging. Map to hiprand_kernel.h if needed.

9. **Standard atomics**: `atomicMin`, `atomicMax`, `atomicAdd`, `atomicOr`, `atomicCAS` all have direct HIP equivalents. The 64-bit atomicMin on framebuffer (`atomicMin(&args.target.framebuffer[pixelID], pixel)` with `uint64_t`) is supported on gfx90a.

10. **`__debugbreak`**: Used in error handling. Replace with `__builtin_trap()` or platform-appropriate alternative.

## File-by-file change list

### New files
| File | Purpose |
|------|---------|
| `src/cuda_to_hip.h` | Compat header: CUDA spellings -> HIP, driver API aliases |
| `src/HipModularProgram.h` | HIP replacement for CudaModularProgram.h using hiprtc |
| `src/HipRuntime.h` | HIP replacement for CURuntime.h |

### Modified files (CMake)
| File | Changes |
|------|---------|
| `CMakeLists.txt` | Add `USE_HIP` option; `enable_language(HIP)` under it; set `.cu` files to `LANGUAGE HIP`; swap CUDA::* targets for HIP targets; arch via `CMAKE_HIP_ARCHITECTURES` |
| `cmake/common.cmake` | Add `ADD_HIP()` function parallel to `ADD_CUDA()`; find hiprtc instead of nvrtc |

### Modified files (kernel/device code)
| File | Changes |
|------|---------|
| `src/kernels/triangles_visbuffer.cu` | Include compat header; GLM_FORCE_CUDA -> GLM_FORCE_HIP under guard |
| `src/kernels/resolve.cu` | Include compat header; surf2D -> HIP surface equivalents |
| `src/kernels/utils.cuh` | Include compat header; nanotime() asm -> HIP clock intrinsic |
| `src/kernels/*.cu` | Include compat header at top |
| `src/jpeg/jpeg.cu` | Include compat header; curand -> hiprand if used |

### Modified files (host code)
| File | Changes |
|------|---------|
| `src/CudaModularProgram.h` | Guard CUDA path; HIP ifdef -> include HipModularProgram.h |
| `src/CURuntime.h` | Guard CUDA path; HIP ifdef for HIP driver API |
| `src/Timer.h` | `CUevent` -> `hipEvent_t` under HIP guard |
| `src/CudaVulkanSharedMemory.h` | HIP external memory APIs; may need significant rework |
| `src/VulkanCudaSharedMemory.h` | Same as above |
| `src/main.cpp` | `cuInit` -> `hipInit` etc. under guard; include cuda_to_hip.h |

### Platform-specific
| File | Notes |
|------|-------|
| `src/unsuck_platform_specific.cpp` | Windows mmap/unbuffered IO -- unchanged for Linux port |
| `src/MappedFile.h` | Windows API -- needs Linux mmap implementation (noted as TODO in README) |

## Build commands

### Configure (gfx90a)
```bash
mkdir build && cd build
cmake .. -DUSE_HIP=ON \
         -DCMAKE_HIP_ARCHITECTURES=gfx90a \
         -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
```

### Build
```bash
cmake --build . -j$(nproc)
```

### Notes
- The project currently only supports Windows (uses Win32 API for memory mapping and unbuffered IO)
- Linux support is TODO per README: "Main challenge: We're using the windows API for memory mapping and unbuffered IO"
- The port needs to address Linux mmap anyway before GPU testing is possible
- Vulkan is available on Linux; GLFW is cross-platform

## Test plan

### No automated test suite
The project has no unit tests or automated test suite. It is an interactive renderer with drag-and-drop model loading.

### Visual validation approach
1. Build and launch the application
2. Load the included `example_donaukanal_urania.glb` model (69 MB)
3. Verify:
   - Model renders correctly (triangles visible, no corruption)
   - Navigation works (orbit controls)
   - Performance is reasonable (check frame time in-app)
   - No GPU faults or crashes

### Downloadable test models
Per README, test with:
- `example_donaukanal_urania.glb` (included, 60M triangles)
- Komainu Kobe (60M triangles)
- Hakone Lantern (1M triangles)
- Sponza (262k triangles)

### Non-GPU tests
None -- this is a pure GPU rendering application.

### Validation command
```bash
./CuRast  # Drag and drop example_donaukanal_urania.glb
```

### Performance baseline
Per README: Venice (400M triangles) renders in 7.98ms on RTX 5090 at 1080p. The gfx90a baseline will differ but should be in the same order of magnitude for reasonable-sized models.

## Open questions

1. **Linux platform support**: The README explicitly lists Linux support as TODO due to Windows-specific memory mapping and unbuffered IO. The port must implement Linux mmap (trivial) and decide on fast sequential reads (io_uring is suggested but `posix_fadvise` + regular read may suffice). This is a prerequisite for GPU testing.

2. **HIP-Vulkan interop**: The CUDA-Vulkan shared memory path is sophisticated (sparse binding, virtual memory, external handles). HIP has equivalent APIs but the Linux/AMD equivalent of `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT` is `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`. Need to verify the HIP external memory path works with Vulkan on AMD GPUs.

3. **Runtime compilation redesign**: The nvrtc+nvJitLink LTO workflow has no direct HIP equivalent. Options:
   - a) Ahead-of-time compilation only (remove runtime compilation, compile all kernels at build time)
   - b) hiprtc single-step compilation (rewrite CudaModularProgram for HIP)
   - Option (a) is simpler and may be sufficient; option (b) preserves the runtime compilation feature

4. **glb loading performance**: The loader is optimized for Zorah (18.9B triangles) and uses 16 parallel threads with large host allocations. May need tuning for AMD GPUs / Linux memory.

5. **CUDA 13.1 requirement**: The project requires CUDA 13.1 with C++23 features (`<print>`, `<format>`, `<stacktrace>`). ROCm clang supports C++23; verify all features used are available.

6. **No test suite**: Validation is visual only. Consider adding a non-interactive render-to-file mode for CI/automated validation, or use screenshot comparison against a reference image.
