# tsne-cuda port plan

## Project

- Name: tsne-cuda (CannyLab/tsne-cuda; maintainer David Chan)
- Upstream: https://github.com/CannyLab/tsne-cuda, default branch `main`, planned at `244db8f` ("4.0.1: modernize Dockerized wheel packaging + pip index (#141)")
- Fork: https://github.com/AMD-Ecosystem/tsne-cuda, port branch `moat-port`. Per jeff's standing rule the fork's GitHub default branch is `moat-port`; `status.json.fork_default_branch` stays the upstream mirror (`main`).
- What it is: FIt-SNE (FFT-accelerated interpolation t-SNE) on the GPU. One shared library, `libtsnecuda.so`, built from 14 `.cu` files (about 3.5k lines), driven from Python through `ctypes` (`src/python/tsnecuda/TSNE.py`, sklearn-style `TSNE.fit_transform`). KNN comes from GPU FAISS; the rest is plain CUDA kernels plus cuBLAS, cuFFT and Thrust.
- Hard dependency: `faiss` (MOAT project, `AMD-Ecosystem/faiss` `moat-port`, completed on linux-gfx90a, linux-gfx1100 and windows-gfx1151 at `01a09e8`). Build it per its notes.md "Install as a dependency".
- Planned on windows-gfx1151. There is no lead platform; this plan is arch-independent and each gate (wave64, wave32, windows) validates on its own.

## Existing AMD support

Classification: **none authoritative** -> fresh HIP port.

- Upstream docs: no `amd|rocm|hip|gfx` hit (intake grep). No "notable forks" section; upstream does not run a link-platform-forks policy, so an upstream PR is the right vehicle.
- `gh pr list --state all --search "ROCm OR HIP OR AMD"`: only #127 "Contributing a SYCL version" (Intel, DPCT-converted, open and unanswered by maintainers since 2023-09). Not a HIP path. Its stall is a signal that the maintainer may be slow to take a vendor backend; it does not change the vehicle.
- oneapi-src/Velocity-Bench carries a `tsne/HIP` variant (https://github.com/oneapi-src/Velocity-Bench/tree/main/tsne/HIP): a benchmark-harness copy of an older tsne-cuda, renamed to `.cpp`, linking `-lhipblas -lhipfft -lhipsparse` and apparently without FAISS. Judgement: **community/benchmark snapshot, non-authoritative** -- a hint at most ("the library swaps are the whole port"), never code to inherit.
- RAPIDS cuML's TSNE derives from this project; it is a downstream consumer, not an AMD port (intake).

Port-vs-rewrite: the kernels are scalar float arithmetic with global atomics -- no tensor cores, no CUTLASS/CuTe, no warp-level primitives, no shared memory. A mechanical HIP port is the right and final form; there is no AMD-native rewrite to consider.

## Build classification

**cmake** (`ext_type: cmake`). Evidence:

- `CMakeLists.txt:11` `enable_language(CUDA)`; `:55` `find_package(CUDAToolkit REQUIRED)`; `:189-211` the `SOURCES` list (all `.cu`); `:225-238` `add_library(tsnecuda SHARED ...)` linking `CUDA::cudart/cublas/cufft/cufftw/cusparse`, OpenMP, pthread, `${FAISS_LIBRARIES}`.
- No `find_package(Torch)`, no `torch.utils.cpp_extension`. The Python package is a pure-Python `ctypes` loader (`TSNE.py:139` `N.ctypeslib.load_library('libtsnecuda', ...)`); `setup.py` only packages the prebuilt `.so`. CMake copies `src/python` next to the library when `BUILD_PYTHON=ON`.

## Port strategy

**Strategy A, shim-header variant.** Every CUDA dependency enters through a named header in `src/include/common.h:13-18` (`cuda.h`, `cublas_v2.h`, `cuda_runtime.h`, `cusolverDn.h`, `cusparse.h`, `cufft.h`) plus `cufft.h` again in `src/include/kernels/nbodyfft.h:5`. A `hip_compat/` directory of shadow headers, prepended to the include path on the HIP build only, carries all the aliasing, and the `.cu` tree stays untouched for the ROCm port itself. Thrust comes from rocThrust under its own `<thrust/...>` names. The CUDA build never sees `hip_compat/`.

CMake: `option(USE_HIP ... OFF)`; under it `enable_language(HIP)`, `find_package(hip/hipblas/hipfft/hipsparse/rocthrust)`, `set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE HIP)`, link `hip::host roc::hipblas hip::hipfft roc::hipsparse roc::rocthrust`; the existing `enable_language(CUDA)`, `CUDAToolkit`, the `-gencode` ladder and `CUDA_OPTS` move into the `else()` branch unchanged. No literal gfx arch in the committed build.

Windows is part of the port, not a follow-up: the HIP build must configure, build and load from Python on Windows (required `windows` gate). Upstream has never supported Windows (issues #139, #131, #55 ask for it), so this needs a handful of LLP64/POSIX fixes listed below; they are behaviour-neutral on Linux. How to carry them is open question 3.

## CUDA surface inventory

Regex floor: `projects/tsne-cuda/surface.json` (38 files incl. headers; 31 `__global__`, 24 launches; libraries cublas 63, cufft 31, cusparse 43, thrust 502, cusolver 1 -- counts include comments and the dead test/CLI files).

| surface | where | HIP/ROCm mapping | note |
|---|---|---|---|
| runtime: `cudaMalloc/Free/Memcpy/MemsetAsync/DeviceSynchronize/SetDevice/GetDeviceProperties/DeviceReset/GetErrorString`, `cudaDeviceProp`, `cudaError_t`, `cudaSuccess`, `cudaErrorInvalidConfiguration` | `fit_tsne.cu`, `options.h:225-232`, `cuda_utils.cu` | hip* 1:1 | `hip_compat/cuda_runtime.h` |
| `<cuda.h>` | `common.h:13` | include `hip/hip_runtime.h` | no driver-API calls anywhere (`cu[A-Z]...(` census empty) |
| cuBLAS: `cublasCreate`, `cublasSgemm` (`distance_utils.cu:45`), `cublasSgemv`, `cublasSsyrk`, `cublasHandle_t`, `CUBLAS_OP_*`, 10 `CUBLAS_STATUS_*` | `distance_utils.cu`, `math_utils.cu`, `fit_tsne.cu`, `cuda_utils.cu:64-92` | hipBLAS | `CUBLAS_STATUS_LICENSE_ERROR` has no hipBLAS value: alias it to a distinct status (`HIPBLAS_STATUS_UNKNOWN`) or the `switch` gets a duplicate case label |
| cuFFT: `cufftCreate/MakePlan2d/MakePlanMany/ExecR2C/ExecC2R/Destroy`, `cufftHandle/Real/Complex/Result`, all `CUFFT_*` result values | `fit_tsne.cu:326-339,582-584`, `nbodyfft.cu:289,406,416`, `cuda_utils.cu:166-230` | hipFFT | `cuda_utils.cu:203,209,224` gate three enum cases on `CUDART_VERSION < 13000`; undefined on HIP -> 0 -> cases compiled, and hipFFT does define `HIPFFT_INCOMPLETE_PARAMETER_LIST/PARSE_ERROR/LICENSE_ERROR`, so no source edit. Do NOT define `CUDART_VERSION` in the shim (rocThrust/hipCUB key on it) |
| cuFFTW | `CMakeLists.txt:232` link only | none needed | no fftw symbol is used; do not link `hipfftw` |
| cuSPARSE: `cusparseCreate`, `cusparseCreateMatDescr/SetMatType/SetMatIndexBase`, handle/descr/status types, 9 `CUSPARSE_STATUS_*` | `fit_tsne.cu:56-69`, `attr_forces.cu:41,105` (signature only), `cuda_utils.cu:112-160` | hipSPARSE | handle is created and threaded through but no live sparse op (the legacy `csrgeam/csr2csc/Sgthr` path in `math_utils.cu:77-220` is commented out) |
| cuSOLVER | `common.h:16` include only | empty stub `hip_compat/cusolverDn.h` | no symbol used |
| Thrust: `device_vector`, `complex<float>`, transform/reduce/sort/gather/permutation/strided ranges, `thrust::random` | everywhere (`thrust_utils.cu`, `math_utils.cu`, `nbodyfft.cu`, `fit_tsne.cu`) | rocThrust | `reinterpret_cast<cufftComplex*>` of `thrust::complex<float>` storage: `hipfftComplex` is `float2`-layout, same as CUDA |
| kernels | `apply_forces`, `attr_forces` (V1/V2/V3 + `reduce_sum_kernel`), `rep_forces`, `perplexity_search`, `nbodyfft` (9), `util/*` (`AssembleDistances`, `PostprocessNeighborIndicesKernel`, `Csr2CooKernel`, `syv2k`, broadcast) | plain HIP | no `__shared__`, no `__syncthreads`, no `__shfl/__ballot/__syncwarp`, no textures/surfaces, no streams/events, no pinned/managed memory |
| atomics | `attr_forces.cu:35-36`, `nbodyfft.cu:136,170` | float `atomicAdd` on global | correct on all archs; summation order nondeterministic on both vendors |
| `warpSize` | `options.h:232` host read into `GpuOptions::warp_size`, never used | none | the only `warp_intrinsic` hit in the census |
| compute-capability dispatch | `options.h:248-345` keys launch sizes on `device_properties.major` | none | AMD `major` (9 on gfx90a, 11/12 on RDNA) lands in the `>= 7` branch, whose values equal the sm_80 branch; harmless collision, no edit |
| GPU FAISS | `distance_utils.cu:79-121` (`StandardGpuResources`, `index_cpu_to_gpu`, `IndexIVFFlat`, k < 1024 on GPU, CPU FAISS for k >= 1024) | FAISS ROCm build (`FAISS_ENABLE_ROCM`) | installed FAISS GPU headers are hipified at configure time; `FindFAISS.cmake` finds them through `CMAKE_PREFIX_PATH` unchanged |
| `register` storage class | 44 uses in 7 `.cu` files | -- | clang rejects `register` in C++17 by default (`-Wregister`, default-error); nvcc accepts it. Add `-Wno-register` to the HIP compile options rather than editing 44 lines |

Not in the surface: no runtime PTX, no NVRTC, no CUDA driver API, no cuRAND (init RNG is host `std::default_random_engine`), no CUB, no cuDNN, no NCCL, no graphics interop.

## Risk list

1. **Wave size (32 vs 64)**: no warp-level code, so no wave64/wave32 hazard. The FFT kernels launch 32-thread blocks (`nbodyfft.cu:255`) and the broadcast kernels 32 (`matrix_broadcast_utils.cu:62`): half-empty wavefronts on CDNA, a performance note only.
2. **Unzeroed allocations**: no live raw `cudaMalloc` (all commented out); buffers are `thrust::device_vector` (value-initialised) or explicitly `cudaMemsetAsync`-zeroed before accumulation (`fit_tsne.cu:411-413,529`, `attr_forces.cu:200`). Low risk; validator runs with `TSNE_SYNC_KERNELS=1` once to localise any async fault.
3. **OOB reads**: `compute_point_box_idx` clamps box indices; launch guards are `TID >= N` throughout. A NaN coordinate would hit an undefined float->int cast before the clamp on both vendors. Low risk; covered by the finite-output check.
4. **FP contraction / iterative divergence**: t-SNE is a 1000-iteration gradient descent with atomics; results are not bitwise reproducible even on CUDA. Validate with embedding-quality metrics, not exact values. RDNA3.5 (gfx1151) is known to diverge from gfx1100 in iterative solvers -- thresholds must be metric-based and identical across archs.
5. **Library swaps**: hipBLAS `LICENSE_ERROR` alias (above); hipBLAS/hipFFT/hipSPARSE handle types are distinct here (no `void*`-typedef overload collisions, because the project has no wrapper overloaded on handle type). `CUBLAS_POINTER_MODE` default host -- same on hipBLAS.
6. **FAISS GPU limits on ROCm**: GPU path is used for `num_neighbors < 1024`; confirm the FAISS ROCm build's `GPU_MAX_SELECTION_K` is >= 1024 on both wave sizes (default `num_neighbors` is 32, so the default path is far inside any limit). `index_cpu_to_gpu` of an IVFFlat index exercises FAISS IVF on ROCm, which FAISS's own suite validated.
7. **`-Wl,--allow-multiple-definition` / `-Xlinker muldefs`** (`CMakeLists.txt:145,236`): upstream suppresses duplicate-symbol errors. The HIP link on Linux keeps the flag; on Windows lld-link has no equivalent short of `/FORCE:MULTIPLE`. Porter: find whether a real duplicate exists before adding anything.
8. **Windows (LLP64 / POSIX)**, none of which upstream ever compiled:
   - `long` used as FAISS's 64-bit index type: `fit_tsne.cu:100` (`new long[]`), `:119` (`device_vector<long>`), `distance_utils.h:100` and `distance_utils.cu:127` (`const long*` kernel arg). `long` is 32-bit on Windows, so this fails to compile against `int64_t*` and would halve the stride if forced. Fix: `int64_t` (identical type on Linux LP64).
   - `ssize_t` in the exported `pymodule_tsne` signature (`pymodule_ext.h:4,14`, `pymodule_ext.cu:8,39-40`): not defined by the Windows SDK. Python passes `c_intp` (pointer-sized signed), so a Windows-only typedef to `SSIZE_T`/`ptrdiff_t` is ABI-exact.
   - `#include <sys/time.h>` in `common.h:47`: absent on Windows and unused (no `gettimeofday`/`timeval` anywhere) -- guard with `#ifndef _WIN32`.
   - CMake: `pthread` and `-Wl,--allow-multiple-definition` link items, GNU `OPT_FLAGS` in `CMAKE_CXX_FLAGS`, and `find_package(OpenMP REQUIRED)` (OpenMP is linked but never used: no `#pragma omp`, no `omp_*`). Guard these for WIN32 on the HIP path.
   - DLL: CMake names it `tsnecuda.dll`, Python loads `libtsnecuda` -> set `PREFIX "lib"` on WIN32; `pymodule_tsne` needs exporting (`WINDOWS_EXPORT_ALL_SYMBOLS` on the target); `setup.py` `package_data` lists only `libtsnecuda.so`.
   - HIP link through CMake on Windows needs the `hip_link_win.py` wrapper (vendor `assets/hip_link_win.py` from the skill to `cmake/hip_link_win.py`, WIN32-guarded); all languages on the GNU-driver clang++ per the skill's Windows notes. The clang-cl `/Fo` bundler fix only if hit.
   - Python >= 3.8 does not search `PATH` for a ctypes DLL's dependencies: the validator stages `faiss.dll`, `hipblas.dll`, `hipfft.dll`/`rocfft.dll`, `hipsparse.dll`/`rocsparse.dll`, `rocblas.dll`, `amdhip64_7.dll`, `rocm_kpack.dll`, `amd_comgr*.dll`, OpenBLAS next to `libtsnecuda.dll` (harness work, not a source edit). TheRock 7.14 Windows SDK on this host does ship `hipfft.dll`, `hipsparse.dll`, `hipblas.dll` and rocThrust headers (checked).
9. **Texture pitch, rule-of-five, layered arrays, `__fsqrt_rn`, CC-dispatch on NVIDIA arch numbers**: not applicable (no textures, no RAII handle wrappers -- handles are created/destroyed once in `fit_tsne`, no correctly-rounded intrinsics; CC dispatch collision is benign as noted).
10. **FAISS + tsnecuda both statically pulling hipBLAS handles**: separate handles, no shared state; low risk.

## File-by-file change list

New files (fork):
- `hip_compat/cuda.h` -- includes `cuda_runtime.h` shim.
- `hip_compat/cuda_runtime.h` -- `hip/hip_runtime.h` + `#define` aliases for exactly the runtime symbols in the table.
- `hip_compat/cublas_v2.h` -- `hipblas/hipblas.h` + handle/status/op/function aliases; `CUBLAS_STATUS_LICENSE_ERROR` -> a distinct hipBLAS status.
- `hip_compat/cufft.h` -- `hipfft/hipfft.h` + type/plan/exec/result aliases (`CUFFT_R2C/C2R`, all 17 result values).
- `hip_compat/cusparse.h` -- `hipsparse/hipsparse.h` + the handle/descr/status/setter aliases.
- `hip_compat/cusolverDn.h` -- empty stub.
- `cmake/hip_link_win.py` -- vendored copy of the skill asset, used only on WIN32.

Edited:
- `CMakeLists.txt` -- `USE_HIP` option; CUDA language/toolkit/arch ladder/`CUDA_OPTS` into the `else()`; HIP: `enable_language(HIP)`, `find_package` of hip/hipblas/hipfft/hipsparse/rocthrust, `LANGUAGE HIP` on `SOURCES`, `target_include_directories(tsnecuda BEFORE PRIVATE hip_compat)`, `$<$<COMPILE_LANGUAGE:HIP>:-Wno-register>`, HIP link list; WIN32 guards for pthread/`--allow-multiple-definition`/GNU flags/OpenMP, `PREFIX "lib"`, `WINDOWS_EXPORT_ALL_SYMBOLS`, link wrapper.
- `src/fit_tsne.cu:100,119`, `src/include/util/distance_utils.h:100`, `src/util/distance_utils.cu:127` -- `long` -> `int64_t` (Windows only in effect).
- `src/include/ext/pymodule_ext.h` -- Windows `ssize_t` typedef.
- `src/include/common.h:47` -- `#ifndef _WIN32` around `<sys/time.h>`.
- `src/python/setup.py` -- add the Windows DLL name to `package_data` (and arguably the OS classifier).
- `INSTALL.md` -- a "Building for AMD GPUs (ROCm)" subsection: FAISS ROCm prerequisite, configure line with `-DUSE_HIP=ON -DCMAKE_PREFIX_PATH=/opt/rocm;<faiss-prefix>`, arch via `-DCMAKE_HIP_ARCHITECTURES`, Windows notes. `README.md` one line pointing to it.

Untouched: every kernel and Thrust body; `options.h`; `TSNE.py`; `examples/`; `visualization/`; `cmake/benchmarks/tune_kernels.py` (vendor-neutral, reads per-phase timers).

## Build commands

FAISS first (per `projects/faiss/notes.md` "Install as a dependency"; same `CMAKE_HIP_ARCHITECTURES`), installed to `$FAISS_PREFIX`.

Linux (gfx90a; any other target changes only the arch):
```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/opt/rocm;$FAISS_PREFIX" \
  -DBUILD_PYTHON=ON
cmake --build build -j
```
Windows (gfx1151; source `agent_space/win_rocm_env.sh` first; all languages on the GNU-driver clang++):
```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_C_COMPILER=$ROCM_ROOT/lib/llvm/bin/clang.exe \
  -DCMAKE_CXX_COMPILER=$ROCM_ROOT/lib/llvm/bin/clang++.exe \
  -DCMAKE_HIP_COMPILER=$ROCM_ROOT/lib/llvm/bin/clang++.exe \
  -DCMAKE_PREFIX_PATH="$ROCM_ROOT;$FAISS_PREFIX" -DBUILD_PYTHON=ON
cmake --build build
```
CUDA no-regression: the same tree with `USE_HIP` off must configure and build with nvcc exactly as before (fleet CUDA host if one is free; otherwise the reviewer confirms every CMake change sits inside `if(USE_HIP)`/`if(WIN32)` and every source edit is type-identical on Linux).

## Test plan

Upstream has **no runnable automated test suite**: `src/test/test.cu` is a gtest file whose every `TEST` is commented out and which includes headers deleted long ago (`naive_tsne.h`, `bh_tsne.h`); INSTALL.md:105 says "the current set of tests do not compile"; v4.0.0 removed gtest and the test target. The upstream-documented check is `tsnecuda.test()` (5000x50 random points, "no errors and no hang"). That is too weak alone, so the validator runs a metric harness, kept in `agent_space/` and recorded verbatim in notes.md (not added to the fork; see open question 4). All runs `PYTHONPATH=build/python`, `ulimit -c 0`, one pinned GPU.

GPU tests (all required on every gate):
1. **Smoke**: `python -c "import tsnecuda; tsnecuda.test()"` exits 0, prints the per-iteration log, output finite.
2. **Quality, small**: sklearn `load_digits` (1797x64), `TSNE(perplexity=30, learning_rate=200, n_iter=1000, random_seed=0)`: all finite; `sklearn.manifold.trustworthiness(X, Y, n_neighbors=10) >= 0.97`; 5-fold 10-NN classifier accuracy on `Y` >= 0.95; and within 0.02 trustworthiness of `sklearn.manifold.TSNE` on the same data (CPU reference).
3. **Quality, scale**: `make_blobs(n_samples=50000, centers=10, n_features=50, cluster_std=2.0, random_state=0)` (20000 on the gfx1151 laptop): finite; `adjusted_rand_score(labels, KMeans(10, n_init=10).fit_predict(Y)) >= 0.95`; wall time recorded.
4. **Async-fault sweep**: rerun 2 with `TSNE_SYNC_KERNELS=1` -- same thresholds, no HIP error.
5. **Launch-config overrides**: rerun 2 with `TSNE_FFT_THREADS=64 TSNE_ATTR_THREADS=256 TSNE_REP_THREADS=256 TSNE_INTEG_THREADS=256` -- same thresholds (exercises non-default block sizes on both wave sizes).
6. **CPU-FAISS branch**: 3000x50 blobs with `num_neighbors=1024` (the `k >= 1024` path in `distance_utils.cu:111`) completes, finite, ARI >= 0.9.
7. **Benchmark (record, not gate)**: `python cmake/benchmarks/tune_kernels.py --n 20000 --iters 300 --repeats 1` runs to completion; record its table.

Thresholds are identical on every arch; if a gate misses one, the first diagnostic is the same harness against sklearn's CPU TSNE, not a loosened bound.

Non-GPU regression set: there are no CPU tests upstream. The regression gate is (a) the CUDA build path unchanged (above) and (b) `import tsnecuda` on Linux still loading `libtsnecuda.so` by the same name.

## Scope and surface accounting

`surface.json` floor: `library:tsnecuda` and `option:BUILD_PYTHON` (both in scope), plus the library census. Planner additions: `python:tsnecuda` (ctypes package, in scope), `build:windows` (in scope), `test:tsnecuda.test` (in scope), `benchmark:tune_kernels` (run, not modified), `example:mnist/cifar` (vendor-neutral Python, covered by inspection), `doc:install-rocm`, and four **proposed scope-outs pending jeff** -- deliberately NOT written into `scoped_out`, so the surface gate keeps them open until jeff rules:

- `exe:src/exe/main.cu` -- the C++ CLI. Upstream removed it from the build in v4.0.0 (`CMakeLists.txt:246-249`, INSTALL.md:53) and it depends on the dropped cxxopts/gflags path. Proposed: not ported (nothing upstream builds it).
- `test:src/test/test.cu` + `src/include/test/*` -- the gtest suite. All tests commented out, includes deleted headers, no CMake target, documented as non-compiling. Proposed: not ported; replaced by the harness above.
- `packaging:` `packaging/Dockerfile`, `conda/meta.yaml`, `build_and_deploy.sh`, `run_all_packaging.sh`, `pypi_extra_hosts/` -- the CUDA wheel/conda pipeline. Proposed: no ROCm wheel or conda recipe in this port (a distribution decision for upstream after the build lands).
- `viz:visualization/` -- CPU Python ZMQ viewers for a feature upstream compiles out (`-DNO_ZMQ`). Proposed: untouched (no GPU code; nothing to port).

## Open questions (for jeff)

1. Scope cuts: confirm the four proposed scope-outs above (CLI, dead gtest suite, CUDA packaging pipeline, ZMQ visualization). None removes anything upstream currently builds or ships beyond the CUDA-specific packaging; the porter does not start until this is answered (standing rule on partial ports).
2. Windows in the same port: plan includes Windows (required gate, and it answers upstream issues #139/#131/#55). Confirm, versus validating Linux only and seeking a windows waiver.
3. The Windows fixes (`long`->`int64_t`, `ssize_t`, `<sys/time.h>`, CMake WIN32 guards, DLL prefix/exports) are not HIP-specific -- they are what any Windows build needs. Upstream is active (four commits 2026-07-22). Carry them on `moat-port` as separate commits in the one PR (recommended: Windows is only reachable through the HIP build, since upstream's CUDA build is Linux-only), or split them into a separate upstream PR first?
4. Test harness location: keep the metric harness out of the fork (recommended; recorded in notes.md), or add a small pytest under the existing empty `src/python/tsnecuda/test/` package so upstream gains a real test?
