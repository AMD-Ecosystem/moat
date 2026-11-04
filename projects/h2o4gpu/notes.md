# h2o4gpu notes

## Intake screen (2026-08-07)

### Licence
- `python3 utils/licenses.py check h2oai/h2o4gpu` -> `Apache-2.0 tier=1`, cleared to
  contribute. GitHub's field matched, and reading `LICENSE` directly confirms plain
  Apache-2.0 text. Recorded in `status.json.license_spdx`.
- Vendored submodules (`.gitmodules`): `cub` (h2oai fork of NVIDIA CUB), `nccl`
  (upstream `NVIDIA/nccl`, Apache-2.0/BSD dual), `xgboost` + `xgboost_prev` (h2oai
  forks, Apache-2.0), `LightGBM` (h2oai fork, MIT), `py3nvml` (h2oai fork),
  `tests/googletest` (Google, BSD-3). No NVIDIA proprietary-licensed file found in any
  of these -- NCCL and CUB are both permissively licensed upstream. **The `h2oai/cub`
  submodule repo no longer exists on GitHub (404, `git ls-remote` confirms "Repository
  not found")** -- a fresh `git clone --recurse-submodules` of h2o4gpu fails today,
  independent of any ROCm work. Worth the planner knowing before they try to build the
  CUDA baseline.

### Duplicate effort
- No AMD/ROCm/HIP/gfx mention anywhere in README.md or docs/.
- No repo matching `h2o4gpu` in AMD-Ecosystem or ROCm orgs (`gh search repos`).
- No upstream PR/issue mentioning ROCm/HIP/AMD (`gh pr list --search "ROCm OR HIP OR
  AMD"` returned one false-positive substring match, "Monika pydaal **new2**" -> no
  "AMD" in a meaningful sense).
- Verdict: no existing AMD effort, nothing to coordinate with.

### Maintenance status of upstream (stated plainly, not left implicit)
- **Not archived** (`isArchived: false`), and a PR was merged as recently as
  2026-07-27 -- but that PR was "Add secret scanning alert notification workflow", CI
  tooling, not code.
- Real feature/fix merges: last one before that was `#879 python 3.11 support`,
  merged 2024-02-22. Before THAT, the merge history goes straight back to 2021
  (`#874` "Address CRAN comments", 2021-05-15) and earlier -- i.e. the last
  substantial engineering work landed roughly five years ago; the repo has since seen
  exactly one real PR (#879) in the last ~5.5 years plus incidental CI housekeeping.
- 157 open issues; oldest open PR (`#886` "DCGM Support") has sat unmerged since
  2026-06-24 (6+ weeks); other open PRs date back to 2018-2020 and were never closed
  either way.
- Read plainly: h2o4gpu is alive in the sense that the repo is not archived and an
  occasional housekeeping PR still lands, but it is dormant for actual development --
  there is no active maintainer cadence to expect a timely review of a ROCm PR
  against. This is not "abandoned" per the `already-supported`/fork-only test (no
  archive banner, no README pointer to a successor), but a person deciding to take
  this up should expect the upstream PR to sit, possibly for a long time, rather than
  assume normal-cadence review.

### Viability -- CUDA surface
- Genuine, substantial own CUDA code under `src/gpu/`: elastic-net GLM solver (a POGS
  derivative: `h2o4gpuglm.cu`, `include/cml/*`, `include/cgls.cuh`,
  `projector/projector_cgls.cu`, `projector/projector_direct_dense.cu`), k-means
  (`kmeans/kmeans_h2o4gpu.cu`, `kmeans_labels.cu`), PCA/truncated-SVD (`pca/pca.cu`,
  `tsvd/tsvd.cu`), matrix factorization / ALS (`factorization/factorization.cu`),
  ARIMA (`arima/arima.cu`), dense/sparse matrix ops (`matrix/matrix_dense.cu`,
  `matrix_sparse.cu`), plus a C++ test suite under `tests/cpp/gpu/`. ~30 own
  `.cu`/`.cuh` files outside the vendored trees, using cuBLAS/cuSPARSE/cuSOLVER/Thrust
  directly and CUB (vendored) and NCCL for multi-GPU.
- What is NOT ours to port: the vendored `xgboost`/`xgboost_prev` and `LightGBM` trees
  carry their own (much larger) CUDA/GPU surfaces belonging to those upstream
  projects, not to h2o4gpu. Neither is tracked as a MOAT project today, so no
  `depends_on` was recorded (there is nothing to depend on yet); if either becomes a
  MOAT target later this would be worth revisiting, but the h2o4gpu-owned surface
  above (GLM/k-means/PCA/ALS/ARIMA/matrix) does not require them to be ported first --
  those libraries are consumed as prebuilt GPU backends, not compiled as part of the
  `src/gpu` code we would touch.
- `daal` (Intel MKL-derived, CPU-only, x86_64-only per README) is a CPU path, out of
  scope for a ROCm/HIP port either way.

### Recommendation
Fork-worthy: permissive top-level licence, no NVIDIA-proprietary text found, a real
and reasonably self-contained CUDA surface (classic GPU ML solvers: GLM/k-means/
PCA/ALS/ARIMA) that does not require porting the vendored xgboost/LightGBM trees
first, and no existing AMD/ROCm effort to duplicate. The caveat is entirely about
upstream review cadence, not about the code or the licence: expect a slow-to-silent
merge process, so budget for a fork-only outcome by default and treat an actual
upstream merge as a pleasant surprise rather than the plan. Also flag for whoever
plans this: the `h2oai/cub` submodule is a dead link right now, so the CUDA baseline
itself needs a workaround (or `hipCUB` swap-in makes it moot) before anything else.

## Planning (2026-11-04, windows-gfx1151 host)

- Plan written: Strategy A (shim headers + 7 `__CUDACC__` gate edits + HIP `get_gpu_info_c` in place of NVML). See plan.md.
- Corrections to intake: upstream is active again (8 merges 2026-09..10, driven by Driverless AI); the dead `h2oai/cub` submodule was dropped upstream in #892; upstream deleted its Python KMeans/TSVD/PCA/GLM tests in #879 (recover from `a22da27^` for out-of-tree validation).
- Fork `AMD-Ecosystem/h2o4gpu` has only `master` at 5e733c6 (upstream tip 97279de); record says `fork_default_branch: main` -- needs a person's `set-base-branch ... master`.
- Licence: of the 12 EULA-notice files, only `bwcheck.cu`/`p2pbwcheck.cu` include the helper headers, and nothing calls `bwcheck()`/`p2pbwcheck()`; plan deletes all 13 as the first moat-port commit.

## Porting (2026-11-04, linux-gfx90a, MI250X/MI250, ROCm 10.1 pip SDK)

Following jeff's 2026-11-04 rulings in plan.md. Fork branch `moat-port` cut from upstream `97279de`.

### Commit 1 `3c713ae` -- licence (ruling 4)
- Deleted the 12 EULA-notice files plus `src/gpu/p2pbwcheck.cu` (13 total, as plan.md recommended): `p2pbwcheck.cu` is a CUDA-Samples-derived file that `#include <helper_cuda.h>`, so deleting the 12 alone breaks the glob-driven GPU build. `bwcheck()`/`p2pbwcheck()` have no callers outside `examples/cpp` under `DOBWCHECK`/`DOP2PCHECK` hard-coded to 0. Deferred item `h2o4gpu-nvidia-proprietary-rescan` set `done`.

### Commit 2 `ac069e2` -- HIP port of h2o4gpu's own solvers
- `USE_HIP` CMake branch (`enable_language(HIP)`, `.cu` -> `LANGUAGE HIP`, `-include hip_compat/cuda_runtime.h` on HIP sources only), `hip_compat/` shim headers (runtime, hipBLAS, hipSPARSE, hipSOLVER, hipRAND, hipCUB, `thrust/system/cuda/error.h`, empty `nvToolsExt.h` because `src/interface_c/h2o4gpu_c.cpp` includes it under `HAVECUDA`). Factorization excluded (ruling 2).
- Gotchas hit:
  - Do NOT define `CUDART_VERSION` in a shim: hipSPARSE/hipSOLVER headers test it to detect the NVIDIA backend (`hipsparseStatus_t` became undefined).
  - rocThrust ships `thrust/system/cuda/error.h` but it is CUDA-only (uses `cudaErrorXxx` enums); shimmed to `thrust/system/hip/error.h` + `thrust::cuda_category() -> hip_category()`.
  - `hip::hipcub` / `roc::rocthrust` targets carry `hip::device` flags (`--offload-arch`) that leak into the g++ host sources of a mixed target; used their include dirs only.
  - Device-side `cudaGetDevice` in two kmeans|| lambdas (no HIP device equivalent): captured the host loop index (the loop just did `cudaSetDevice(i)`).
  - `ProxEvalF` ctor was `__device__` only but constructed on host; nvcc tolerated it, clang does not.
  - cuSOLVER headers pull in `cublas_api.h`; `arima.h` relied on that for `cublasHandle_t`, so the `cusolver_common.h` shim includes `cublas_v2.h`.
  - OpenMP: g++ objects use libgomp; clang `-fopenmp` emits libomp (`__kmpc_*`). HIP TUs are compiled without OpenMP, so the per-GPU `#pragma omp parallel for` loops in kmeans run serially (correct, slower multi-GPU). `gpuh2o4gpu` links `OpenMP::OpenMP_CXX` publicly so the HIP-linked test binary resolves `GOMP_*`.
  - `get_gpu_info_c` on ROCm: HIP runtime (`hipGetDeviceCount`, `hipGetDeviceProperties`, `hipSetDevice`+`hipMemGetInfo`, restores the current device). Utilization/per-pid usage = 0 with a warning. Names are not returned (the NVML path's `gpu_name` calloc never reached the caller either).
  - ARIMA `d_0_p_2_q_2_iter_1` / `d_1_p_1_q_1_iter_1` failed `ASSERT_FLOAT_EQ` (4 ULP) against cuSOLVER golden values. Triangulated with a float64 run of the same fits (`agent_space/arima/drv.cpp` linking `libgpuh2o4gpu.a`): d1p1q1 phi float(ROCm) -1.03694952, float64 -1.03695150, golden -1.0369391; d0p2q2 theta[1] ROCm -0.616015851, float64 -0.616015982, golden -0.61601555. ROCm is closer to float64 than the golden values -> rounding, not a defect. ROCm-only `ASSERT_NEAR(.., 2e-5*|expected|)` for those two tests; CUDA keeps `ASSERT_FLOAT_EQ`.
- Build (exact):
  ```
  R=$(rocm-sdk path --root); export PATH=$R/bin:$PATH
  cmake -S projects/h2o4gpu/src -B projects/h2o4gpu/src/build-hip -DUSE_CUDA=OFF -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=$R -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$FLEET_VENV/bin/python
  cmake --build projects/h2o4gpu/src/build-hip -j$MAX_JOBS
  # upstream driver (arch auto-detected; OMP_NUM_THREADS caps the Makefile's `make -j$(nproc)`):
  CMAKE_PREFIX_PATH=$R OMP_NUM_THREADS=$MAX_JOBS make cpp USE_HIP=ON
  ```
  Host deps installed: `apt-get install swig libopenblas-dev`; `git submodule update --init tests/googletest`.
- Results: `./build-hip/h2o4gpu_test` 24/24 pass, 3 consecutive runs (ARIMA 13, KMeansCentroids 4, KMeans h2o4gpu/labels rest). `get_gpu_info_c` via the SWIG module: `HIP_VISIBLE_DEVICES=0` -> n_gpus=1, `0,1,2,3` -> 4, 63.98 GiB total each, cc (9,0). `make cpp USE_HIP=ON` -> verify_gpu_lib.sh PASS (gfx90a code object after strip).
- No nvcc on this host: CUDA no-regression is by construction (additive `if(USE_HIP)` branch; the shared-source edits are `__HIPCC__` additions, a `__host__` qualifier, a captured loop index, one include, and test macros that expand to `ASSERT_FLOAT_EQ` on CUDA).
