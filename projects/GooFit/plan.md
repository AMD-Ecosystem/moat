# Plan: GooFit

## Project

- Name: GooFit
- Upstream: https://github.com/GooFit/GooFit
- Default branch: main
- Description: Massively-parallel fitting framework using Thrust for CUDA/OpenMP, for maximum-likelihood fits in High Energy Physics

## Existing AMD support

**Assessment**: No existing ROCm/HIP support found.

Searches performed:
- Grep of upstream docs for AMD/ROCm/HIP: no references found (only noise matches: "amphi", "warp" in physics context)
- Web search "<project> ROCm/AMD/HIP/MI300": no results
- `gh api repos/GooFit/GooFit/forks`: ~40 forks, none with rocm/hip/amd in name, none under ROCm/AMD/GPUOpen orgs
- No rocm/hip branches in upstream
- No PRs/issues mentioning ROCm/HIP/AMD
- CMakeLists.txt has no HIP/ROCm references

**Merge policy**: Standard GitHub model (accepts PRs). No indication of a "link platform forks" policy.

**Decision**: Proceed with a HIP port targeting ROCm. The project is a clean Strategy A candidate.

## Build classification

**Classification**: Pure CMake project (Strategy A)

**Evidence**:
- `CMakeLists.txt` lines 1-6: `project(GOOFIT VERSION 2.3.0 LANGUAGES CXX)`
- Line 328-329: `enable_language(CUDA)` inside `GOOFIT_OPTIONAL_CUDA()` macro
- Lines 500-548: `GOOFIT_ADD_LIBRARY()` function that sets `CUDA_SEPARABLE_COMPILATION ON`
- Lines 550-573: `GOOFIT_ADD_EXECUTABLE()` function
- `setup.py` uses scikit-build to invoke CMake (not a PyTorch extension; no torch.utils.cpp_extension)
- No `find_package(Torch)`, no `CUDAExtension`

The project uses Thrust's compile-time backend selection (`THRUST_DEVICE_SYSTEM`) to support CUDA, OMP, TBB, and CPP backends. On CUDA, it enables the CUDA language and compiles `.cu` files; on non-CUDA backends, it recompiles `.cu` as C++ with `-x c++`.

## Port strategy

**Strategy A: Pure CMake, compat-header approach**

Rationale:
1. GooFit is a standalone CMake project with `.cu` sources and heavy Thrust usage
2. No PyTorch dependency
3. The project already abstracts device/host via Thrust macros, making HIP integration straightforward
4. GooFit's `GlobalCudaDefines.h` already provides CUDA stub implementations for non-CUDA backends

Approach:
1. Add a `cuda_to_hip.h` compat header for CUDA runtime API symbol mapping
2. Add `USE_HIP` CMake option, enable HIP language, mark `.cu` files `LANGUAGE HIP`
3. Use rocThrust (ships with ROCm) instead of the bundled thrust submodule
4. The `THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA` define already exists; keep it (rocThrust maps this)
5. Minimal guards for HIP-specific includes

## CUDA surface inventory

### Kernels and device code
- **108 `.cu` files** across src/PDFs/, examples/, tests/, python/PDFs/
- **~766 `__global__`/`__device__`/`__host__` annotations**
- No manual `<<<>>>` launches found in user code; Thrust handles execution

### Thrust usage (primary GPU interface)
- **~572 thrust:: call sites**
- `thrust::transform`, `thrust::reduce`, `thrust::transform_reduce`
- `thrust::device_vector`, `thrust::counting_iterator`, `thrust::zip_iterator`
- `thrust::random::default_random_engine`, `thrust::uniform_real_distribution`
- `thrust::remove_if`, `thrust::fill`, `thrust::max_element`
- `thrust::norm`, `thrust::conj` (complex math)
- All will work via rocThrust without modification

### Warp intrinsics
- **None found**: No `__shfl*`, `__ballot`, `__activemask`, `warpSize` in kernels
- The project does not use warp-level primitives directly
- No wave64/wave32 concern

### Shared memory
- `__shared__` used in one file: `ConvolutionPdf.cu:94` (`__shared__ fptype modelCache[CONVOLUTION_CACHE_SIZE]`)
- Already stubbed for non-CUDA: `GlobalCudaDefines.h:29` defines `__shared__` as empty

### Constant memory
- `__constant__` arrays used for physics constants (e.g., `gpuDebug`, `debugParamIndex`, `AmpIndices[500]`, mass constants)
- `MEMCPY_TO_SYMBOL`/`MEMCPY_FROM_SYMBOL` macros wrapping `cudaMemcpyToSymbol`/`cudaMemcpyFromSymbol`
- HIP equivalent: `hipMemcpyToSymbol`/`hipMemcpyFromSymbol` (same semantics)

### CUDA runtime API
- `cudaMalloc`/`cudaFree`: wrapped in `gooMalloc`/`gooFree` (PdfBase.cu)
- `cudaMemcpy`: via `MEMCPY` macro (GlobalCudaDefines.h:50)
- `cudaDeviceSynchronize`: direct calls (5 sites) + stub for non-CUDA
- `cudaGetDeviceCount`/`cudaGetDeviceProperties`/`cudaSetDevice`: in Application.cpp for device selection
- `cudaError_t`, `cudaSuccess`: error handling

### CUDA libraries
- **None**: No cuBLAS, cuFFT, cuRAND, cuSPARSE, cuDNN
- Thrust is the only GPU library dependency

### Textures/surfaces
- **None**: Only commented-out references to `cudaArray`

### Pinned/managed memory
- **None**: No `cudaMallocManaged`, `cudaMallocHost`, `cudaHostAlloc`

### Streams/events
- `cudaStream_t` referenced in disabled ThrustOverride.h (`#if 0`)
- No active stream usage

### Read-only cache
- `__ldg` via `RO_CACHE(x)` macro (GlobalCudaDefines.h:70)
- HIP equivalent: `__ldg` is supported on HIP

## Risk list

| Risk | Assessment | Mitigation |
|------|------------|------------|
| rocThrust vs bundled Thrust | Low | ROCm ships rocThrust; detect via `find_package(rocthrust)` or use rocm include paths directly. The project already has `GOOFIT_FORCE_LOCAL_THRUST` option. |
| `__ldg` intrinsic | Low | HIP supports `__ldg`; no change needed. |
| Constant memory symbols | Low | `hipMemcpyToSymbol`/`hipMemcpyFromSymbol` are 1:1 with CUDA equivalents. |
| CUDA version macros | Low | `CUDART_VERSION` appears in info output only; can be stubbed or gated on `__HIPCC__`. |
| Complex number support | Low | Uses `thrust::complex` for `thrust::norm`/`thrust::conj`; rocThrust supports these. |
| C++11 standard | Low | The project uses C++11 by default; rocThrust/hipCUB require C++17. Must bump to `-std=c++17`. |
| MCBooster submodule | Medium | MCBooster (GooFit/MCBooster) is a Thrust-based phase-space generator. Likely works on HIP but needs verification. |
| Submodule initialization | Low | Shallow clone with `--depth=1` does not init submodules; need `git submodule update --init` before build. |

## File-by-file change list

### New files
- `include/goofit/detail/cuda_to_hip.h` -- CUDA-to-HIP compat header

### Modified files
1. **CMakeLists.txt**
   - Add `option(USE_HIP "Build with HIP for AMD GPUs" OFF)`
   - Add HIP language enable and arch configuration under `USE_HIP`
   - Set `CMAKE_HIP_ARCHITECTURES` default to gfx90a when unset
   - Mark `.cu` sources as `LANGUAGE HIP` when `USE_HIP`
   - Bump `CMAKE_CXX_STANDARD` and `CMAKE_HIP_STANDARD` to 17 (rocThrust requires C++17)
   - Find rocThrust when USE_HIP

2. **include/goofit/GlobalCudaDefines.h**
   - Add `#include "cuda_to_hip.h"` at top when `USE_HIP` or `__HIP_PLATFORM_AMD__`
   - No other changes needed; CUDA runtime symbols will be remapped via the compat header

3. **src/goofit/Application.cpp**
   - Add HIP equivalents for device query (`hipGetDeviceCount`, `hipGetDeviceProperties`, `hipSetDevice`)
   - Guard `CUDART_VERSION` references

4. **include/goofit/PDFs/MetricTaker.h**
   - Ensure `cudaError_t`/`cudaSuccess` are mapped via compat header

### Files unchanged
- All `.cu` files remain unchanged (symbol mapping via compat header)
- All Thrust usage unchanged (rocThrust is API-compatible)

## Build commands

### Configure (gfx90a)
```bash
cd projects/GooFit/src
git submodule update --init --recursive
mkdir build && cd build
cmake .. \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_CXX_STANDARD=17 \
  -DGOOFIT_DEVICE=CUDA \
  -DGOOFIT_TESTS=ON \
  -DGOOFIT_EXAMPLES=ON \
  -DGOOFIT_CERNROOT=OFF
```

Note: `GOOFIT_DEVICE=CUDA` is kept because the Thrust backend define `THRUST_DEVICE_SYSTEM_CUDA` is what rocThrust uses on HIP. The CMake detects CUDA language but we override to HIP.

### Build
```bash
cmake --build . -j$(nproc)
```

### Follower platforms
```bash
# gfx1100
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 ...

# gfx1101/gfx1201 (Windows)
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1101 ...
```

## Test plan

### GPU tests (must pass)
```bash
cd build
ctest --output-on-failure
```

Test categories:
- `tests/simple/`: VectorsTest.cu, SimpleTest, NormalizeTest, MonteCarloTest, BinningTest, BlindTest, Minuit1Test
- `tests/convert/`: ~25 PDF conversion tests (Gaussian, Argus, BW, etc.)
- `tests/PDFs/`: GenArgusTest, GenGaussianTest

All tests exercise GPU execution via Thrust.

### Non-GPU regression set
The same tests run on CPU when built with `GOOFIT_DEVICE=OMP` or `GOOFIT_DEVICE=CPP`. These do not use HIP and should not regress:
- Must still compile with `-DGOOFIT_DEVICE=OMP` (OpenMP backend)
- Must still compile with `-DGOOFIT_DEVICE=CPP` (single-threaded)

### Example programs (smoke test)
```bash
./examples/simpleFit/simpleFit
./examples/exponential/exponential
./examples/convolution/convolution
```

These examples run fits on synthetic data; successful completion indicates GPU execution works.

## Open questions

1. **MCBooster compatibility**: MCBooster is a Thrust-based phase-space generator bundled as a submodule. It should work with rocThrust, but needs GPU validation.

2. **Python bindings**: The scikit-build setup.py invokes CMake. The Python bindings should work with the HIP port if CMake is configured correctly, but need testing.

3. **ROOT integration**: Tests skip ROOT if not found (`GOOFIT_CERNROOT=OFF`). The Minuit2 fallback is pure C++ and should work unchanged.

4. **Thrust version**: The bundled Thrust (extern/thrust) is old. ROCm's rocThrust is more recent. The project has `GOOFIT_FORCE_LOCAL_THRUST` to prefer the bundled one, but on HIP we should use system rocThrust. May need to disable this option.
