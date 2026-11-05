# h2o4gpu porting plan

## Project

- Name: h2o4gpu
- Upstream: https://github.com/h2oai/h2o4gpu, default branch `master` (planned against tip `97279de`, 2026-10-29)
- Fork: https://github.com/AMD-Ecosystem/h2o4gpu. Today it carries only `master`, at `5e733c6` (one commit behind upstream). There is no `moat-port` yet.
- Licence: Apache-2.0 (intake). One licence action is still open and it comes first. See "Licence precondition" below.

### Facts that changed since intake (2026-08-07)

- **Upstream is active again.** h2oai merged eight PRs between 2026-09-16 and 2026-10-29: CUDA 12.8 and Blackwell support (#889), setuptools 83 (#890), dropping the `cub`, `py3nvml` and `nccl` submodules (#892), version 0.6.0 (#893), a "DAI build configuration" (#894), and xgboost/LightGBM bumps (#896-#898). The driver is Driverless AI (DAI), which consumes `TruncatedSVD`/`KMeans` from the GPU library (upstream `CLAUDE.md`). An upstream PR therefore has a live reviewer, not a dormant queue. Expect that reviewer to weigh the change against DAI's NVIDIA-only packaging.
- **The dead `h2oai/cub` submodule is gone.** #892 dropped it, and the sources now use the CUDA toolkit's CUB. hipCUB replaces it on ROCm, so intake's "fresh clone fails" warning no longer applies. Only four submodules are left: `xgboost`, `xgboost_prev`, `LightGBM` and `tests/googletest`.
- **`status.json.fork_default_branch` says `main`, but upstream and the fork both use `master`.** The tooling uses that field as the PR base and the mirror branch. Correcting it is a person's ruling (`moatlib.py set-base-branch h2o4gpu master --by <who>`). See open question 7.
- **Upstream deleted its Python KMeans/TSVD/PCA/GLM tests in #879 (2024).** The shipped Python suite now has GLM tests that need datasets, gbm tests for the bundled xgboost/LightGBM, and two non-GPU system tests. See the test plan.

## Existing AMD support

**None.** I re-ran the checks at planning time:
- `grep -niE '\b(amd|rocm|hip)\b|gfx[0-9]'` over README.md, docs/ and FAQ.md: no hits.
- `gh pr list -R h2oai/h2o4gpu --state all --search "ROCm OR HIP OR AMD"`: only #408 "Monika pydaal new2", a false substring hit.
- AMD-Ecosystem and ROCm orgs: no h2o4gpu effort (intake).

Classification: no AMD support, so this is a fresh HIP port. Upstream does not link platform forks, so **an upstream PR is the right vehicle**. The licence commit's vehicle is a separate question (open question 4).

Port vs rewrite: **mechanical port.** The heavy lifting is in vendor libraries (BLAS, sparse, dense solvers, sort/reduce). The project's own kernels are simple elementwise, tile and Cholesky kernels with no tensor-core, CUTLASS or warp-specialised code. Nothing calls for an AMD-native rewrite.

## Licence precondition (counsel ruling, must be the first fork commit)

`deferred.json` item `h2o4gpu-nvidia-proprietary-rescan` (decided `now` by jeffdaily, counsel ruling 2026-08-24) records that 12 files carry the pre-2017 CUDA Samples EULA-pointer notice. The ruling: our tip must never host them.

Those files are the ten `src/include/helper_*.h` headers, `src/include/exception.h` and `src/gpu/bwcheck.cu`. `src/gpu/p2pbwcheck.cu` (NVIDIA 1993-2015 copyright, includes `helper_cuda.h`) depends on them.

The ruling was meant to run "before any fork is created". The fork already exists, but its `master` is an unmodified upstream mirror and no MOAT tip exists yet. The ruling can still be met for **our** tip: the first commit on `moat-port` removes the files. Whether the existing mirror is acceptable is jeff's call (open question 4).

Findings that decide what "remove" means:
- `bwcheck.cu` and `p2pbwcheck.cu` are compiled into `gpuh2o4gpu` by `FILE(GLOB_RECURSE GPU_SOURCES src/*.cu ...)`. Nothing calls them: `bwcheck()` and `p2pbwcheck()` have no caller in src/, the SWIG `.i` files, Python or tests.
- Only those two files include the helper headers, directly or transitively (`helper_cuda.h`, `helper_functions.h` -> `helper_timer.h`/`helper_string.h`/`helper_image.h`/`exception.h`). Nothing at all includes `helper_cuda_drvapi.h`, `helper_cuda_gl.h`, `helper_gl.h`, `helper_math.h` or `helper_cusolver.h`.

Recommended action: **delete all 13 files** (the 12 flagged plus `p2pbwcheck.cu`). That changes no reachable behaviour, and the glob picks up the removal with no CMake edit. The alternative, swapping in BSD-3 cuda-samples headers, keeps dead code alive and still leaves the H2O-modified `bwcheck.cu` body under the old notice. Close the deferred item once the commit lands.

## Build classification

**cmake** (`ext_type` set to `cmake`). Evidence:
- Top-level `CMakeLists.txt` (CMake >= 3.18) builds every native artifact. Legacy FindCUDA does it: `FIND_PACKAGE(CUDA 8.0 REQUIRED)`, `CUDA_ADD_LIBRARY(gpuh2o4gpu ... STATIC)` with `CUDA_SEPARABLE_COMPILATION ON`, and `CUDA_ADD_EXECUTABLE(h2o4gpu_test ...)`. All of it sits under `if(USE_CUDA)` (`OPTION(USE_CUDA ... ON)`).
- SWIG produces the Python extension modules (`SWIG_ADD_LIBRARY(ch2o4gpu_gpu ...)`, `ch2o4gpu_cpu`), not `torch.utils.cpp_extension`. There is no `find_package(Torch)` and no torch dependency in `src/interface_py/setup.py` or the requirements files.
- The top-level `Makefile` drives `cmake` (`cpp`, `testunit`) and then `setup.py bdist_wheel` (`py`).

## Port strategy

**Strategy A, the shim-header method, plus a small set of guarded source edits.**

Rationale:
- Sources include CUDA headers by name: `cuda_runtime.h`, `cuda.h`, `cuda_runtime_api.h`, `cublas_v2.h`, `cusparse.h`, `cusolverDn.h`, `cusolver_common.h`, `curand.h`, `curand_kernel.h`, `cub/cub.cuh` and `thrust/system/cuda/error.h`. Shadow headers in a `hip_compat/` directory, prepended to the include path on the HIP build only, cover the runtime and library renames with no edits to the `.cu` files. Thrust headers resolve to rocThrust unchanged.
- The FindCUDA block cannot carry HIP. Add an `elseif(USE_HIP)` branch, or a separate `if(USE_HIP)` block with `USE_CUDA` forced off. Inside it: `enable_language(HIP)`, the same `GPU_SOURCES` glob with `.cu` marked `LANGUAGE HIP`, plain `add_library`/`add_executable`, the shim include dir, and `find_package(hip hipblas hipsparse hipsolver hiprand hipcub rocthrust)` links. The NVIDIA path stays byte-identical.
- Some things a shim cannot reach and need guarded source edits, kept to a minimum:
  1. Seven `__CUDACC__` gates (`prox_lib.h:13,685`, `util.h:145,262,368`, `glm.h:310`). HIP-Clang does not define `__CUDACC__`, so `__DEVICE__` and the thrust paths would silently compile host-only. Change each to `defined(__CUDACC__) || defined(__HIPCC__)`.
  2. NVML in `src/gpu/utils.cu` `get_gpu_info_c`. See the risk list.
  3. Whatever the HIP compiler rejects on first build (expected small).

Do not pass `-D__CUDACC__` globally. rocThrust/rocPRIM headers test it.

## CUDA surface inventory

Scope is the h2o4gpu-owned native code: 45 `.cu`/`.cuh`/`.h` files under `src/gpu`, plus 7 test TUs in `tests/cpp`. The census is in `surface.json`: 64 kernels, 93 launches, 2 warp-intrinsic hits, both in the excluded factorization code.

| Area | CUDA usage | ROCm mapping | Notes |
|---|---|---|---|
| Runtime | malloc/memcpy/memset (sync + async), streams (`cuda_stream[MAX_NGPUS]` per device in kmeans), `cudaGetDeviceProperties` (major/minor/name/`singleToDoublePrecisionPerfRatio`), `cudaDeviceReset`, `cudaHostRegister` (test_utilities.h only) | HIP runtime via `hip_compat/cuda_runtime.h` | `singleToDoublePrecisionPerfRatio` exists in `hipDeviceProp_t`; confirm at compile time |
| cuBLAS | ~50 entry points, S/D gemm, gemv, geam, dgmm, trsm, trsv, syrk, ger, nrm2, dot, axpy, scal, asum, i?amax, getrf/getrsBatched, plus complex `cublasCaxpy`/`Zaxpy`/`Scnrm2`/`Dznrm2` and `cublasSetStream`/`GetStream` | hipBLAS (`hip_compat/cublas_v2.h`) | Complex types: `cuComplex`/`cuDoubleComplex` map to `hipComplex`/`hipDoubleComplex` (ROCm 7 hipBLAS uses the HIP complex types natively) |
| cuSPARSE | Generic API only in the built code: `cusparseCreateCsr`, `CreateDnVec`, `SpMV` + `SpMV_bufferSize`, `Csr2cscEx2` + bufferSize, legacy `MatDescr` setters; `cudaDataType` `CUDA_R_32F/64F`, `CUDA_C_64F` (`cgls.cuh`) | hipSPARSE generic API, `HIP_R_32F/64F`, `HIP_C_64F` | `csrmm2` appears only in the excluded factorization code |
| cuSOLVER | Dense: `gesvd`, `syevd`, `geqrf`, `ormqr` (S/D) + buffer sizes, handle create/destroy | hipSOLVER (`hip_compat/cusolverDn.h`, `cusolver_common.h`) | TSVD/PCA/ARIMA/GLM-direct depend on it; check `gesvd` jobu/jobvt char-vs-enum signature in hipSOLVER |
| cuRAND | Device API `curand_init`/`curand_uniform(_double)`/`curandState` in `cml/cml_rand.cuh` (reached via `equil_helper.cuh`); host API `curandGenerate` only in `test_utilities.h` (NCCL test helper, compiled out) | hipRAND device API (`hiprand/hiprand_kernel.h`) | Sequences differ from cuRAND, so tests must not compare bit patterns of random init |
| Thrust | 745 hits: device_vector, transform/reduce/inner_product/sort/extrema/fill/random, `thrust/system/cuda/error.h` | rocThrust | Check that rocThrust ships `thrust/system/cuda/error.h`; otherwise shim it to the hip error header |
| CUB | `cub::DeviceRadixSort::SortPairs`, `DeviceReduce::Sum`, `DeviceSegmentedReduce::Sum`, `DoubleBuffer` (kmeans, data/matrix.cu) | hipCUB via `hip_compat/cub/cub.cuh` (`namespace cub = hipcub;`) | Device-wide only, no block collectives, so the TempStorage-reuse fault class does not apply |
| NVML | `get_gpu_info_c` in `src/gpu/utils.cu`: count, memory, name, utilisation, per-pid usage, CC | HIP runtime queries (see risk 1) | Load-bearing for the Python GPU path |
| NCCL | `h2o4gpuglm.cu`, `warmstart.cu`, `glm.h` under `USE_NCCL`/`USE_NCCL2` | (RCCL if ever enabled) | No build path defines either macro, so compiled out everywhere |
| NVTX | `USE_NVTX` blocks; CMake links `nvToolsExt` when `$ENV{USENVTX}` | (roctx) | `USE_NVTX` is never defined; HIP branch ignores `USENVTX` (open question 5) |
| Kernels | Own kernels in cml (`set_vector`, `any_isnan`, `strided_memcpy`, `set_matrix`, block Cholesky/trsv with `__shared__` tiles of `kTileSize=32`), equil_helper, matrix_dense/sparse (`SetSign`/`MultRow`/`MultCol`), kmeans (`self_dots`, `all_dots` with 32x32 shared tiles, `matmul` with dynamic shared memory under 32 KiB, `calculate_centroids`), arima (differencing, residuals) | HIP, unchanged | No warp-width assumption in built kernels: `32` here is a tile edge, not a lane count |
| Textures/surfaces, managed memory, driver API, NVRTC/PTX | none in project code (driver-API census hits are in the deleted helper headers) | n/a | |
| Excluded (upstream) | `src/gpu/factorization/*` ALS: `cusparseScsrmm2`/`Dcsrmm2`, `#define WARP_SIZE 32`, `__shfl_down_sync(0xFFFFFFFF, ..., WARP_SIZE)`, `__shfl` | n/a | Excluded by upstream's own CMake filter (open question 2) |

## Risk list

1. **NVML is load-bearing (highest risk).** `h2o4gpu.util.gpu.device_count()` calls `get_gpu_info_c()[0]`, and every solver (`kmeans.py`, `elastic_net.py`, `pogs.py`, `types.py`) uses that count to decide between GPU and CPU. With NVML stubbed or failing, every Python solver silently runs sklearn's CPU code and every Python test still passes. On ROCm, implement `get_gpu_info_c` with HIP runtime calls under `#if defined(__HIP_PLATFORM_AMD__)` (after `hip_runtime.h` is included, through the shim): `hipGetDeviceCount`, `hipMemGetInfo` per device, `hipDeviceProp_t.name/major/minor`. Utilisation and per-pid usage take the existing `NVML_ERROR_NOT_SUPPORTED` path (0, with a warning). amdsmi is an option for utilisation, but it is Linux-only and adds a dependency; the porter should not add it unless a test needs it. **The validator must prove the GPU path ran** (see the test plan), not just that tests passed.
2. **The `__CUDACC__` gates** (above). If missed, `prox_lib.h` functors lose `__device__` and fail to compile in device code. If a gate selects only a type alias, it can compile and silently pick a host path.
3. **`get_gpu_info_subprocess` uses `pynvml`.** It returns 0 GPUs on AMD, silently. It is reached only from `get_gpu_info()`, which no solver calls (the solvers use the C path). Leave it as is unless the Python tests show a caller. If one turns up, guard it the same way as risk 1.
4. **Header-defined `__global__`:** `kmeans_labels.h:95` `extern __global__ void debugMark(){};` is a non-inline, non-template kernel defined in a header included by several TUs. Under HIP non-RDC this can produce duplicate host-stub symbols at link. The fix is to make it `static`/inline-equivalent, or drop the unused definition behind the HIP guard. Decide when it fires.
5. **Separable compilation.** The CUDA build sets `CUDA_SEPARABLE_COMPILATION ON`, but the census shows no cross-TU `extern __device__`/`__constant__`. Build HIP **without** `-fgpu-rdc` first. Add `HIP_SEPARABLE_COMPILATION` only if the link proves it is needed: RDC on Windows also hits the bundler bug, and it slows the link.
6. **Invalid launch configuration in dead-ish code:** `cml_matrix.cuh` `_set_matrix` launches `dim3(kBlockSize, kBlockSize)` = 256x256 threads. That is illegal on CUDA and HIP alike, so it fails whenever it runs. Pre-existing and probably unreached. Do not fix it unless a test reaches it; if one does, the fix belongs in a separate commit.
7. **Fresh device allocations are not zeroed on ROCm.** kmeans/GLM allocate many `thrust::device_vector`s (which value-initialise) but also raw `cudaMalloc` buffers (`cml_rand` states, CUB temp storage, `d_key_alt_buf`). Any upstream code that reads a raw buffer before writing it will diverge only on AMD. Check this first if kmeans labels or GLM objectives look nondeterministic.
8. **FP contraction.** clang HIP defaults to `-ffp-contract=fast` across expressions. GLM/POGS iterates to tolerance, and kmeans distance uses the `||x||^2 + ||y||^2 - 2x.y` expansion. Expect last-ulp differences. Use tolerance-based checks against the CPU path, and do not loosen any existing test tolerance without recording why.
9. **Library signature drift:** hipSOLVER `gesvd` takes `signed char jobu/jobvt` in the hipSOLVER-native API but has a compat-API variant. hipSPARSE generic-API enum names (`HIPSPARSE_SPMV_ALG_DEFAULT`, `HIPSPARSE_CSR2CSC_ALG1`). The status enums differ, which affects `cublasGetErrorString`/`cusparseGetErrorString` switch tables in `cml_utils.cuh` and `cuda_utils*.h`. A `case CUBLAS_STATUS_LICENSE_ERROR` style label with no HIP equivalent needs a guard.
10. **Wave64 vs wave32:** no built kernel depends on warp width (the only warp code is in the excluded factorization). Re-check this if factorization is ever brought in (open question 2): `WARP_SIZE 32` + `__shfl_down_sync(0xFFFFFFFF, ...)` and the `blockDim.x / WARP_SIZE` shared reduction would both need the 64-lane treatment.
11. **Compute-capability collision:** `get_compute_capability` and `get_gpu_info_c` report `prop.major/minor`. Nothing in the native code or Python dispatches on them (checked: no consumer outside `util/gpu.py` reporting). Report HIP's values as they are.
12. **`CUDA_MAJOR`** is passed by `-DCUDA_MAJOR=${CUDA_VERSION_MAJOR}` and used in `elastic_net_ptr.cpp` (a printed "blasnumber"). The HIP branch must define it, for example as `HIP_VERSION_MAJOR`. `HAVECUDA` must stay defined on the HIP build: it means "GPU build" to the host code.
13. **`verify_gpu_lib.sh`** checks the `.so` with `cuobjdump`, and `make cpp` runs it after `strip`. On ROCm it must either check the `.hip_fatbin` section (`llvm-readelf`/`roc-obj-ls`) or skip with a message. Otherwise `make cpp` fails on ROCm.
14. **Rule-of-five / handles:** cublas/cusparse/cusolver handles are created and destroyed per object in several places (`cublasCreate` x15). A double destroy is tolerated on CUDA but faults on ROCm. Watch for it in teardown crashes.

## File-by-file change list (fork `moat-port`, cut from upstream `97279de`)

Commit 1, licence (before any port code):
- Delete `src/include/helper_cuda.h`, `helper_cuda_drvapi.h`, `helper_cuda_gl.h`, `helper_cusolver.h`, `helper_functions.h`, `helper_gl.h`, `helper_image.h`, `helper_math.h`, `helper_string.h`, `helper_timer.h`, `src/include/exception.h`, `src/gpu/bwcheck.cu`, `src/gpu/p2pbwcheck.cu`. Check that the CUDA build still compiles (glob-driven; nothing references them).

Port commits:
- `CMakeLists.txt`: `OPTION(USE_HIP ... OFF)`; HIP branch as above. Build only the targets the CUDA branch builds (`gpuh2o4gpu`, `ch2o4gpu_gpu`, `h2o4gpu_test`), with the same factorization filter. Use `-DCUDA_MAJOR=${HIP_VERSION_MAJOR} -DHAVECUDA`. Do not pin `CMAKE_HIP_ARCHITECTURES`. Use the same `-std=c++17`; device lambdas need no flag on HIP. The NVIDIA branch stays untouched.
- `hip_compat/` (new, HIP include path only): `cuda_runtime.h`, `cuda_runtime_api.h`, `cuda.h` (runtime aliases only), `cublas_v2.h`, `cusparse.h`, `cusolverDn.h`, `cusolver_common.h`, `curand.h`, `curand_kernel.h`, `cub/cub.cuh`, and `thrust/system/cuda/error.h` only if rocThrust lacks it. Keep host-safe includes unconditional and gate device-only ones behind `__HIPCC__`: host `.cpp` TUs (`src/common`, the SWIG wrapper) reach some of these through `glm.h`/`util.h`.
- `src/include/prox_lib.h`, `src/include/util.h`, `src/include/solver/glm.h`: `__CUDACC__` -> `__CUDACC__ || __HIPCC__` (7 sites).
- `src/gpu/utils.cu`: HIP-runtime implementation of `get_gpu_info_c` behind `__HIP_PLATFORM_AMD__`; the NVML path is unchanged; `#include "nvml.h"` guarded.
- `src/gpu/kmeans/kmeans_labels.h`: `debugMark` fix only if the link requires it.
- `Makefile`: forward `USE_HIP` (`cmake -DDEV_BUILD=... -DUSE_HIP=$(USE_HIP) ../`) in `cpp` and `testunit`.
- `scripts/verify_gpu_lib.sh`: ROCm branch (risk 13).
- `src/interface_py/setup.py`: treat a ROCm toolchain (`hipcc`/`amdclang++` on PATH, or `USE_HIP=ON`) like `nvcc` when deciding to package `_ch2o4gpu_gpu`.
- `README.md` (or `DEVEL.md`, matching where upstream documents the build): a short "Building for AMD GPUs (ROCm)" subsection with the commands below, including `-DCMAKE_PREFIX_PATH=/opt/rocm`.

Not touched: `src/gpu/factorization/**`, NCCL/NVTX code, `xgboost`/`xgboost_prev`/`LightGBM`, Dockerfiles, `ci/`, `examples/`, `src/interface_r`.

## Build commands (Linux, gfx90a; substitute the arch for other hosts)

Prerequisites: ROCm 7.x with hipBLAS, hipSPARSE, hipSOLVER, hipRAND/rocRAND, rocThrust, hipCUB; OpenBLAS (`cblas.h`); SWIG; Python 3.11 + numpy; `git submodule update --init tests/googletest`.

```
cmake -S . -B build-hip -DUSE_CUDA=OFF -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j"$(nproc)"
```

Equivalent through upstream's driver: `make cpp USE_HIP=ON` and `make testunit USE_HIP=ON`. Then `make apply-sklearn && make py` builds the wheel. That last step needs network access for the scikit-learn overlay.

CUDA no-regression gate: configure the untouched default (`-DUSE_CUDA=ON`) with nvcc on a CUDA-capable build host if one is available. Otherwise state that the NVIDIA path is unchanged by construction (an additive `USE_HIP` branch, deleted files unreferenced).

## Test plan

GPU (required, every validating platform):
1. **C++ googletest:** `./build-hip/h2o4gpu_test`. Upstream builds `tests/cpp/test_main.cu` plus `gpu/arima/{test_arima,test_least_squares_solver}.cu` and `gpu/kmeans/{test_kmeans_centroids,test_kmeans_h2o4gpu,test_kmeans_labels}.cu`. Record the pass/fail/skip counts. `test_als.cu` is excluded as upstream excludes it.
2. **Python solver oracle (the GPU path through SWIG):** install the wheel, then check `h2o4gpu.util.gpu.device_count(-1)[1] >= 1` **first**; this guards risk 1. Then run the KMeans/TruncatedSVD/PCA/ElasticNet(GLM) tests upstream deleted in #879, taken from `a22da27^:tests/python/open_data/{kmeans,svd,glm}` and run out of tree (they are not shipped, so do not add them to the fork). Each test must log that it ran on the GPU (`n_gpus > 0`, or the solver's verbose backend line). Where those files need data, use sklearn-generated datasets.
3. **Shipped GLM tests:** `tests/python/small/test_glm_{hyatt,ipums,paribas}.py` after `make sync_small_data` / `sync_open_data` (public S3, `--no-sign-request`). Record it as an environment gap if the buckets are gone.
4. **Multi-GPU** (only where the host has 2+ GPUs): KMeans with `n_gpus=2` exercises the per-device stream and CUB-buffer arrays.

Non-GPU regression set (must not regress):
- `cpuh2o4gpu` and `ch2o4gpu_cpu` build in the same configure (they are outside the GPU branch).
- `tests/python/open_data/system/test_import.py` and `test_metrics.py`.
- The fork's NVIDIA build is unchanged (above).

Not in the gate: `tests/python/open_data/gbm/*` and `tests/python/xgboost/*`, which test the bundled xgboost/LightGBM (open question 1); `tests/python/big`, which is a large-data variant of the small GLM tests (run it if time allows).

Platform order: Linux wave64 and wave32 first; Windows only per open question 3.

## Open questions (need jeff)

1. **Bundled xgboost/LightGBM GPU builds: proposed cut, pending jeff.** The h2o4gpu wheel bundles h2oai forks of xgboost (`xgboost`, `xgboost_prev`) and LightGBM, each with its own large CUDA surface, built by separate Makefile targets. The proposal is to port only h2o4gpu's own solvers (the CMake build) and leave those submodules untouched: a ROCm wheel would ship them CPU-only, or with whatever the user installs. They are separate codebases and would be separate MOAT candidates. Approve the cut, or say which of them must be in scope.
2. **Factorization/ALS solver: proposed scoped out, pending jeff.** Upstream already excludes `src/gpu/factorization` (and `test_als.cu`) from its CUDA 12 build, because `cusparse?csrmm2` was removed. Bringing it back on ROCm means rewriting it to the generic SpMM API and fixing its hardcoded `WARP_SIZE 32`/`__shfl` for wave64. That is feature work upstream has not done for CUDA either. The plan mirrors upstream's exclusion.
3. **Windows: proposed out of scope, waiver suggestion pending jeff.** Upstream is Linux-only: a Rocky 8/glibc toolchain, a bash Makefile, `sys/time.h`/`unistd.h`/`sys/resource.h` in kmeans, CPU and common sources, `.so` SWIG modules and `strip`. A Windows build would be a POSIX-to-Win32 port of host code unrelated to HIP. This host (windows-gfx1151) cannot validate the Linux build. An agent may only suggest a `windows` gate waiver; whether to request one is jeff's decision.
4. **Licence commit vehicle and the existing mirror.** (a) The ruling said "before any fork is created". The fork now exists, and its `master` mirrors upstream including the 12 flagged files. Is an unmodified upstream mirror acceptable under the ruling, given that our own tip (`moat-port`) will never carry the files? (b) Should the deletion travel in the upstream PR as its own commit ("remove unused CUDA Samples bandwidth-test code"), go upstream separately, or stay fork-only? Recommendation: its own first commit inside the upstream PR. It is dead code, and h2oai is actively maintaining the tree.
5. **NVTX: proposed ignored on ROCm, pending jeff.** `USE_NVTX` is never defined by any build path; `USENVTX` only links `nvToolsExt`. The HIP branch ignores it rather than mapping to roctx.
6. **Docker/packaging: proposed out of scope, pending jeff.** No ROCm counterpart to `Dockerfile.build-cuda12` / `make docker-build-cuda12`. The README documents the native ROCm build instead.
7. **`fork_default_branch` correction.** The record says `main`; upstream and fork use `master`. Needs `python3 utils/moatlib.py set-base-branch h2o4gpu master --by jeffdaily` (a person's ruling by design). Separately, under the standing rule the fork's GitHub default branch becomes `moat-port` once the porter creates it. That is a fork setting for a person; the recorded mirror field stays `master`.
8. **Upstream reception.** h2oai is active again, but its current work is driven by DAI's NVIDIA packaging, and an AMD path is an added maintenance surface for them. Proceed toward an upstream PR as usual; it is flagged here only so the expectation is set.

## Rulings (jeff, 2026-11-04)

1. Bundled GBM libraries are in scope, without disturbing the CUDA build. LightGBM: upstream (lightgbm-org/LightGBM) has `USE_ROCM`; use it, bumping or rebasing the ROCm build onto a version that has it if h2oai's fork predates it. xgboost: upstream dmlc/xgboost has no HIP build, so under `USE_HIP=ON` only, build xgboost from AMD-Ecosystem/xgboost (AMD's ROCm fork; ROCm branches `release_3.0.0-rocm`, `dev-rocm`, ...) instead of the h2oai submodule. Check whether h2oai's xgboost fork carries h2o-specific patches the ROCm branch lacks, and port them if so. The CUDA path keeps the existing submodules unchanged. Validate the gbm/xgboost Python tests on the GPU.
2. Factorization/ALS: mirror upstream's exclusion.
3. Windows: attempt last, when it is the only thing left. If it changes too much or is too cumbersome, stop and report. Otherwise it lands as one final commit on moat-port (a nice-to-have that can be split into its own PR if asked).
4. The 12 EULA-notice dead files: delete them in the first commit on moat-port.
5. NVTX: map to roctx (`roctxRangePushA`/`roctxRangePop`, link `roctx64`) under the same `USE_NVTX`/`USENVTX` switch, and test it: build with the switch on, run a GLM solve under rocprofv3 (or rocprof) with marker tracing, and confirm the ranges appear in the trace.
6. Docker/packaging: out of scope.
7. Done: `fork_default_branch` corrected to `master` (`set-base-branch`).


## Ruling 2026-11-04 (jeff)
- xgboost_prev under USE_HIP: keep it, built CPU-only, so models saved by older releases still load in the ROCm wheel as they do in the CUDA wheel.
- xgboost: stay on AMD-Ecosystem/xgboost release_2.1.3-rocm (closest to h2oai's 2.1.4, same generation as the CUDA wheel) with its one rocThrust compat patch; do not move to release/3.2.0.
- xgboost_prev packaging (jeff 2026-11-05): fix the missing libxgboost.so in the shared packaging step for both the CUDA and ROCm wheels, as its own non-ROCm commit that can be split out.
