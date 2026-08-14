# Plan: GooFit

> **Status: this plan has been superseded by what shipped.** The inventory,
> risk and test sections below held up, but the original design (a `USE_HIP`
> option layered on `GOOFIT_DEVICE=CUDA`, and dropping the bundled Thrust)
> was not what got built. The "Port strategy", "File-by-file change list"
> and "Build commands" sections have been rewritten to describe the built
> port; `notes.md` (attempt 5 onward) is the authoritative record.

## Project

- Name: GooFit
- Upstream: https://github.com/GooFit/GooFit
- Default branch: master
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

**Pure CMake, compat-header approach (as built)**

Rationale:
1. GooFit is a standalone CMake project with `.cu` sources and heavy Thrust usage
2. No PyTorch dependency
3. The project already abstracts device/host via Thrust macros, making HIP integration straightforward
4. GooFit's `GlobalCudaDefines.h` already provides CUDA stub implementations for non-CUDA backends

What shipped, which differs from the original sketch in four ways:

1. **HIP is a first-class value of the existing `GOOFIT_DEVICE` switch**, not a
   separate `USE_HIP` option layered on `GOOFIT_DEVICE=CUDA`. `GOOFIT_DEVICE=HIP`
   sits beside CUDA/OMP/TBB/CPP, which is where a GooFit reader expects to find a
   backend and avoids a second, contradictory way to say the same thing. A helper
   `IS_NOT_HIP` keeps the CUDA-only branches readable.
2. **`cuda_to_hip.h` is force-included, not `#include`d.** No `.cu` file names it;
   `GOOFIT_ADD_LIBRARY`/`GOOFIT_ADD_EXECUTABLE` pass
   `-include <cuda_to_hip.h>` on HIP translation units only, so the CUDA spelling
   of every source is left untouched.
3. **The bundled Thrust is left alone.** The plan was to disable
   `GOOFIT_FORCE_LOCAL_THRUST`; instead the HIP branch simply keeps
   `extern/thrust` off the include path so rocThrust's own headers are not
   shadowed. Upstream's `option()` and both of its branches are byte-identical to
   upstream, which keeps the diff at four added lines.
4. **The physics PDFs are scoped out behind a new `GOOFIT_PHYSICS` option**,
   defaulting ON everywhere except HIP. They depend on MCBooster and on a
   device-side Eigen complex matrix inverse, neither of which compiles under
   hipcc yet. This was not foreseen in the risk list below, which rated MCBooster
   "Medium / likely works".

Also load-bearing, and not in the original sketch: `GlobalCudaDefines.h` had to be
split rather than left intact. Upstream's single non-CUDA block would otherwise
have redefined `__shared__` to empty on top of the definition clang preincludes
via `__clang_hip_runtime_wrapper.h`. That is a warning, not an error, and it
silently turns `__shared__ fptype modelCache[...]` in `ConvolutionPdf.cu` into
per-thread private scratch with no LDS, leaving the surrounding `__syncthreads()`
meaningless.

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
- `include/goofit/detail/cuda_to_hip.h` -- CUDA-to-HIP runtime symbol map, force-included on HIP translation units

### Modified files
1. **CMakeLists.txt**
   - `GOOFIT_DEVICE=HIP` as a value of the existing backend switch, with `enable_language(HIP)`
   - `goofit_mark_hip_sources()` marks both `.cu` and the rocThrust-including `.cpp` as `LANGUAGE HIP`; without this CMake drops the `.cu` objects from the device link
   - `-fgpu-rdc` on compile *and* on the final link (see notes.md; the link side is the non-obvious half)
   - `-include cuda_to_hip.h` on HIP translation units only
   - New `GOOFIT_PHYSICS` option, declared next to `GOOFIT_KMATRIX`, default ON except on HIP
   - Keeps `extern/thrust` off the include path on HIP so it cannot shadow rocThrust
   - IPO exclusion for HIP alongside the existing CUDA one -- precautionary only; CMake 3.31 applies `INTERPROCEDURAL_OPTIMIZATION` to C/CXX/CUDA/Fortran and never to HIP

2. **include/goofit/GlobalCudaDefines.h**
   - Split upstream's single non-CUDA block so the half that redefines `__shared__`, `__constant__`, `__host__`, `__device__` does not fire under hipcc, which already defines them
   - The CPU-fallback half (including the `cudaError_t` enum) is unchanged for CPP/OMP/TBB

3. **include/goofit/detail/CudaCompat.h**
   - `GOOFIT_DEVICE_IS_GPU`, true for the CUDA and HIP Thrust systems alike; the guards that used to test `THRUST_DEVICE_SYSTEM_CUDA` now test this

4. **src/goofit/Application.cpp**
   - HIP version banner, and device-info output that reports the backend name and the gfx architecture rather than "CUDA" and a compute capability

5. **src/PDFs/MetricTaker.cu**
   - Fixed `MAX_NUM_OBSERVABLES` scratch array in place of device-side `new[]`/`delete[]`, matching upstream's own #384 change to `CompositePdf.cu`

6. **src/PDFs/CMakeLists.txt, src/PDFs/physics/CMakeLists.txt**
   - `utilities/DebugTools.cu` moves from `PDFCore` to `PDFPhysics` on every backend: it reads the `AmpIndices` `__constant__` defined by `Amp4BodyGlobals.cu`, so it cannot link with the physics sources excluded

7. **src/PDFs/basic/StepPdf.cu**
   - Drops an unused host-side global taking the address of a `__device__` function; invalid on HIP, dead code on CUDA

8. **python/goofit/CMakeLists.txt, python/PDFs/{,basic/,combine/,utilities/}CMakeLists.txt** (added at `e8dca9151`)
   - The pybind11 modules are declared with a plain `add_library()`/`target_sources()`, so their sources kept the CXX language while `roc::rocthrust` handed every consumer `hip::device`'s `-x hip --offload-arch=`; the host compiler then rejects the flags and the default configuration does not build
   - Two new macros in the top-level `CMakeLists.txt` (`goofit_adopt_hip_target`, `goofit_adopt_hip_sources`) give those targets the same `LANGUAGE HIP` + `-include cuda_to_hip.h` treatment as everything else, and are inert off the HIP backend

### Files unchanged
- Every other `.cu` keeps its CUDA spelling (symbol mapping via the force-included compat header)
- All Thrust usage unchanged (rocThrust is API-compatible)

## Build commands

### Configure and build
```bash
cd projects/GooFit/src
git submodule update --init --recursive
cmake -S . -B build-hip -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGOOFIT_DEVICE=HIP -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DGOOFIT_TESTS=ON -DGOOFIT_EXAMPLES=ON -DGOOFIT_CERNROOT=OFF
cmake --build build-hip -j32
```

`GOOFIT_DEVICE=HIP` selects the backend on its own; there is no `USE_HIP` option and
no `GOOFIT_DEVICE=CUDA` override. `GOOFIT_PHYSICS` defaults OFF here, so passing it
is unnecessary. HIP is never chosen by `GOOFIT_DEVICE=Auto` and must be named.
`-DGOOFIT_PYTHON=OFF` is no longer needed either, and leaving it off is now the
better check: the option defaults ON where Python development files exist, and the
default is what a user runs. Add `-DGOOFIT_PYTHON=OFF` only to shorten a build.

Other architectures differ only in `CMAKE_HIP_ARCHITECTURES`; nothing in the
backend is architecture-specific.

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

## Open questions (resolved)

1. **MCBooster compatibility**: does not compile under hipcc. Together with a
   device-side Eigen complex matrix inverse, this is what put the physics PDFs
   behind `GOOFIT_PHYSICS`. Registered as scoped-out work, not a defect.

2. **Python bindings**: build and pass on CPP, and since `e8dca9151` on HIP too
   (gfx90a: 6/6 `python/tests`, which are real 100k-event unbinned NLL fits). The
   earlier reading -- that the bindings export physics classes `GOOFIT_PHYSICS=OFF`
   excludes -- was wrong; `python/CMakeLists.txt` and `python/PDFs/CMakeLists.txt`
   already gate the physics bindings on that option. What actually broke them was
   the pybind11 targets being declared with a plain `add_library()`, which left
   their sources in the CXX language while `roc::rocthrust` handed every consumer
   `hip::device`'s `-x hip --offload-arch=`. They were never exercised because every
   recorded session passed `-DGOOFIT_PYTHON=OFF`, while the option defaults ON.

3. **ROOT integration**: unchanged, as expected. Built with `GOOFIT_CERNROOT=OFF`;
   the Minuit2 fallback is pure C++ and needed no attention.

4. **Thrust version**: resolved without touching `GOOFIT_FORCE_LOCAL_THRUST`. The
   HIP branch keeps `extern/thrust` off the include path, so rocThrust supplies
   Thrust and upstream's option keeps its exact meaning on every other backend.
   Note that upstream has since moved `extern/thrust` to modern CCCL (#391), which
   is one of the commits the rebase picked up.
