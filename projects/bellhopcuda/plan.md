# Plan: bellhopcuda

## Project

- **Name:** bellhopcuda
- **Upstream:** https://github.com/A-New-BellHope/bellhopcuda
- **Default branch:** main
- **Description:** C++/CUDA port of BELLHOP/BELLHOP3D underwater acoustics simulator (ray tracing for ocean acoustics).

## Existing AMD support

**None found.** Checked:
- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/`: no matches
- Web search for "bellhopcuda ROCm AMD GPU HIP": no results
- `gh api repos/A-New-BellHope/bellhopcuda/forks`: no AMD/ROCm/HIP-named forks
- `gh api repos/A-New-BellHope/bellhopcuda/pulls?state=all`: no ROCm/HIP PRs
- The upstream repo explicitly mentions only NVIDIA GPUs (GeForce RTX 20xx/30xx/40xx, A100, GH200) in docs.

**Decision:** Proceed with a from-scratch HIP port.

**Merge policy:** The project appears to accept contributions normally; no "notable forks" or "platform links" pattern. A single upstream PR is the appropriate delivery vehicle.

## Build classification

**Pure CMake (Strategy A)**

Evidence:
- `CMakeLists.txt` line 18: `project(bellhopcuda LANGUAGES CXX CUDA)` in config/cuda/CMakeLists.txt
- No `find_package(Torch)`, no `torch.utils.cpp_extension`, no setup.py, no pytorch dependency
- The project builds two variants: bellhopcxx (CPU-only, C++ threads) and bellhopcuda (CUDA GPU)
- Build gated by `BHC_ENABLE_CUDA` option (ON by default)

The build uses CMake's `enable_language(CUDA)` and generates template instantiations via `configure_file()` for different run/influence/SSP combinations:
- `fieldimpl.cu.in` -> many `field_*_*.cu` files for CUDA kernel variants
- `fieldimpl.cpp.in` -> many `field_*_*.cpp` files for CPU worker variants

## Port strategy

**Strategy A: pure CMake, compat-header approach**

Rationale:
- The project is a pure CMake build with minimal CUDA surface
- The CUDA code is well-isolated: only the generated `fieldimpl.cu.in` templates and `UtilsCUDA.cuh` contain CUDA-specific code
- Most source files are `.cpp`/`.hpp` and use the `HOST_DEVICE` macro (defined as `__host__ __device__` on CUDA, empty otherwise)
- No warp intrinsics (`__shfl*`, `__ballot`), no textures/surfaces, no cuBLAS/cuFFT/cuRAND/Thrust/CUB usage

Implementation:
1. Add a `cuda_to_hip.h` compat header with the ~15 CUDA API symbols used
2. Add a `config/hip/` subdirectory with `CMakeLists.txt` and `SetupHIP.cmake` mirroring the CUDA structure
3. Mark `.cu` sources as `LANGUAGE HIP` when `USE_HIP=ON`
4. Handle the NVIDIA-specific intrinsics in `common_setup.hpp` (`__nv_isinf*`, `__nv_isnan*`) with HIP equivalents

## CUDA surface inventory

### Runtime API calls (need alias in compat header)
| Location | Symbol |
|----------|--------|
| api.cpp:148 | `cudaGetDeviceCount` |
| api.cpp:160,182 | `cudaGetDeviceProperties`, `cudaDeviceProp` |
| api.cpp:188 | `cudaSetDevice` |
| mode/field.cpp:136 | `cudaSetDevice` |
| common_setup.hpp:171 | `cudaFree` |
| common_setup.hpp:191 | `cudaMallocManaged` |
| fieldimpl.cu.in:56,63 | `cudaMallocManaged`, `cudaFree` |
| UtilsCUDA.cuh | `cudaError_t`, `cudaSuccess`, `cudaGetErrorName`, `cudaGetLastError`, `cudaGetErrorString`, `cudaPeekAtLastError`, `cudaDeviceSynchronize`, `cudaDeviceReset` |

### Kernel qualifiers
| Location | Type |
|----------|------|
| common.hpp:56 | `HOST_DEVICE` macro -> `__host__ __device__` |
| common.hpp:124-129 | `__device__ __forceinline__` for `dev_bail()` |
| common.hpp:286-300 | `__device__` for math functions |
| common_setup.hpp:32-35 | `__device__` for `__isinf`, `__isnan` wrappers |
| fieldimpl.cu.in:29,33 | `__global__` kernel |

### NVIDIA-specific intrinsics (need HIP alternatives)
| Location | Symbol | HIP equivalent |
|----------|--------|----------------|
| common_setup.hpp:32-35 | `__nv_isinfd`, `__nv_isinff`, `__nv_isnand`, `__nv_isnanf` | `__hisinf`, `__hisnan` or just use `isinf`/`isnan` from `<cmath>` under HIP |
| common.hpp:126 | `__trap()` | `__builtin_trap()` (same) |
| common.hpp:318-319 | `__int_as_float`, `__float_as_int` | HIP has these (same names) |
| common.hpp:331-332 | `__longlong_as_double`, `__double_as_longlong` | HIP has these (same names) |

### Libraries (none actively used)
The `UtilsCUDA.cuh` file has error string mappings for cuBLAS, cuFFT, cuRAND, cuSPARSE, cuSOLVER, nvJPEG, NPP -- but these are UNUSED. The headers are guarded by `#ifdef CUBLAS_API_H_` etc., and no source file includes those headers or links those libraries. These can be removed or left as dead code.

### Memory model
- Uses `cudaMallocManaged` (unified memory) for `ErrState` and data structures
- HIP equivalent: `hipMallocManaged` (1:1 mapping)

### Streams/events
- None used

### Textures/surfaces
- None used

### Warp intrinsics
- None used. The code stores `warpSize` from device properties but never uses warp-level primitives.

### Thread indexing
- Standard `blockIdx.x`, `blockDim.x`, `threadIdx.x`, `gridDim.x` (1:1 HIP mapping)

## Risk list

1. **Low risk: NVIDIA-specific math intrinsics.** The `__nv_isinf*`/`__nv_isnan*` intrinsics in `common_setup.hpp` are NVCC-specific. HIP uses standard `isinf`/`isnan` from `<cmath>` in device code. The existing code has a non-`__NVCC__` fallback using `std::isinf`/`std::isnan`, so the HIP path may "just work" if `__NVCC__` is not defined under HIP. However, hipcc often defines `__NVCC__` for compatibility, so we may need an explicit `__HIP_PLATFORM_AMD__` guard.

2. **Low risk: `cuda/std/complex` and `cuda/std/atomic`.** The code uses libcu++ (`cuda::std::complex`, `cuda::std::atomic`). Under HIP, we use standard C++ (`std::complex`, `std::atomic`). The existing code already has a fallback via the `#else` branch in `common.hpp:97-101` when `BHC_BUILD_CUDA` is not defined. We need to ensure `BHC_BUILD_HIP` takes the same path or use ROCm's `hip/std/` headers if available.

3. **No risk: warp size.** The code stores `cudaProperties.warpSize` in `d_warp` but never uses it for kernel logic, ballot, shuffle, or shared memory sizing. No warp-size-dependent code paths.

4. **No risk: textures/surfaces.** Not used.

5. **No risk: CUDA libraries.** No actual usage of cuBLAS, cuFFT, cuRAND, etc.

6. **Low risk: GLM submodule.** The project uses GLM (OpenGL Mathematics library) as a submodule. GLM is header-only and platform-agnostic; no porting needed.

7. **Low risk: `__CUDA_ARCH__` guards.** The code uses `#ifdef __CUDA_ARCH__` to select device vs. host code paths in `common.hpp` and `util/atomics.hpp`. Under HIP, use `__HIP_DEVICE_COMPILE__` instead. Add a compat define.

## File-by-file change list

### New files
| File | Purpose |
|------|---------|
| `src/util/cuda_to_hip.h` | Compat header with CUDA->HIP aliases |
| `config/hip/CMakeLists.txt` | HIP build configuration (mirrors cuda/) |
| `config/hip/SetupHIP.cmake` | HIP compiler flags and options |
| `src/mode/fieldimpl.hip.in` | HIP kernel template (copy of fieldimpl.cu.in with minor tweaks, or just reuse .cu.in with LANGUAGE HIP) |

### Modified files
| File | Changes |
|------|---------|
| `CMakeLists.txt` | Add `BHC_ENABLE_HIP` option |
| `config/CMakeLists.txt` | Add `add_subdirectory(hip)` when `BHC_ENABLE_HIP` |
| `config/SetupCommon.cmake` | Minor: add HIP-aware clang detection |
| `config/GenTemplates.cmake` | Add `hip` extension option for HIP builds |
| `src/common.hpp` | Add `#elif defined(__HIP_PLATFORM_AMD__)` branches for HIP runtime includes and `STD` namespace |
| `src/common_setup.hpp` | Handle `__HIP_PLATFORM_AMD__` for isinf/isnan intrinsics |
| `src/util/atomics.hpp` | Add `__HIP_DEVICE_COMPILE__` as alternative to `__CUDA_ARCH__` |
| `src/api.cpp` | Include compat header; no other changes (CUDA API calls aliased) |
| `src/mode/field.cpp` | Include compat header |
| `src/util/UtilsCUDA.cuh` | Add HIP equivalents for error checking macros, or create separate `UtilsHIP.cuh` |
| `doc/compilation.md` | Add ROCm/HIP build instructions |

## Build commands

### Configure (gfx90a)
```bash
mkdir build && cd build
cmake .. \
  -DBHC_ENABLE_CUDA=OFF \
  -DBHC_ENABLE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_BUILD_TYPE=Release
```

### Build
```bash
cmake --build . -j$(nproc)
```

### Follower platforms
Replace `-DCMAKE_HIP_ARCHITECTURES=gfx90a` with:
- gfx1100: `-DCMAKE_HIP_ARCHITECTURES=gfx1100`
- gfx1101: `-DCMAKE_HIP_ARCHITECTURES=gfx1101`
- gfx1201: `-DCMAKE_HIP_ARCHITECTURES=gfx1201`

## Test plan

### GPU tests (validation required)
The project has an extensive test suite comparing outputs against the original Fortran BELLHOP:

```bash
# Run test suite comparing bellhopcuda (HIP) against bellhopcxx (CPU) and BELLHOP (Fortran)
# From repo root:
./run_tests.sh tl2D tl_match
./run_tests.sh tl3D tl3d_match_short
./run_tests.sh ray2D ray_tests_pass
./run_tests.sh arr2D arrivals_match
./run_tests.sh eigen2D eigen_tests
```

The test script:
1. Runs the Fortran BELLHOP reference
2. Runs bellhopcxx (CPU, single-threaded)
3. Compares CPU results to Fortran reference
4. Runs bellhopcxx (CPU, multi-threaded) and compares
5. Runs bellhopcuda (GPU) and compares

For HIP validation, we need to:
1. Build bellhopcxx (CPU) as reference
2. Build bellhopcuda with HIP
3. Run the same test matrix, replacing CUDA with HIP

### Non-GPU tests (must not regress)
- bellhopcxx (CPU-only) build and tests should remain unchanged
- The existing comparison scripts (`compare_shdfil.py`, `compare_ray_2.py`, `compare_arrivals.py`) are Python-based and platform-agnostic

### Test matrix
| Test type | 2D | 3D | Nx2D |
|-----------|----|----|------|
| TL (transmission loss) | tl_match.txt | tl3d_match_short.txt | tl_Nx2D_match.txt |
| Ray tracing | ray_tests_pass.txt | ray_3d_pass.txt | ray_Nx2D_pass.txt |
| Eigenrays | eigen_tests.txt | eigen_3d_tests.txt | N/A |
| Arrivals | arrivals_match.txt | N/A | N/A |

### Validation criteria
- Numerical results must match the Fortran BELLHOP reference within the existing tolerances (set in the compare_*.py scripts)
- No new failures in the "pass" test lists
- No regressions in "match" comparisons

## Open questions

1. **libcu++ vs std.** The code uses `cuda::std::complex` and `cuda::std::atomic` under CUDA. Under HIP, should we use `std::complex`/`std::atomic` (the current fallback) or ROCm's `hip/std/` headers if available? The fallback is likely sufficient since the code is not performance-critical in these paths.

2. **GLM submodule initialization.** The build requires `git submodule update --init --recursive`. Ensure this is documented for the HIP build path.

3. **Fortran BELLHOP dependency for tests.** The full test suite requires the original Fortran BELLHOP to be built and available. For HIP validation, we can compare against bellhopcxx (CPU) instead if Fortran BELLHOP is unavailable.
