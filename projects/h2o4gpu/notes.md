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

### Python GPU path (plan test-plan items 2-3), at `ac069e2`, re-run at `e7727e7`
- Wheel: `make apply-sklearn PYTHON=$FLEET_VENV/bin/python` (clones scikit-learn 1.5.2; also inits all submodules), then `make py PYTHON=...` -> `h2o4gpu-0.6.0+local...-cp311-linux_x86_64.whl`; `pip install --no-deps` into `$FLEET_VENV`. Extra pip in FLEET_VENV: cython==3.0.11 threadpoolctl psutil tabulate future joblib pytest-timeout meson-python ninja feather-format "pyarrow<15". Fleet guard allows pip only in `$FLEET_VENV`.
- `h2o4gpu.util.gpu.device_count(-1)` -> (1, 1) with `HIP_VISIBLE_DEVICES=0`.
- Out-of-tree oracle `agent_space/oracle/oracle.py` (vs scikit-learn 1.9.1 CPU, sign-aligned SVD comparisons): ALL PASS. KMeansH2O f32/f64 v_measure 1.000000, max center dist 6e-6 / 1e-14; TSVD f32 sv rel err 1.4e-6, comp 7.4e-4; f64 9e-15 / 1.3e-8; PCA f32 ev 4.9e-7 comp 2.8e-5; f64 3e-15 / 1.5e-8; ElasticNetH2O ridge r2 0.999972 (f32) / 0.999971 (f64) vs sklearn 0.999975.
- GPU proof: same oracle under `rocprofv3 --kernel-trace`: 8782 dispatches on the GPU agent incl. h2o4gpu's own `kmeans::detail::{self_dots,all_dots,matmul,make_new_labels,calculate_centroids,...}`, `cml::*`, `h2o4gpu::*`, plus rocBLAS/rocSOLVER/rocPRIM.
- Upstream's Python KMeans/SVD/GLM tests deleted in #879, recovered from `a22da27^` into `agent_space/oldtests` (not added to the fork), imports remapped to current names (`h2o4gpu.solvers.*`, CPU references from real `sklearn`), plus a conftest re-exporting solvers at `h2o4gpu.*` (the pre-#879 API):
  - kmeans+svd: 21 pass / 48 fail / 1 skip. All 18 kmeans fails: `KMeansSklearn.__init__() got an unexpected keyword argument 'precompute_distances'` (sklearn >= 1.0 API drift in the public `KMeans` wrapper's CPU fallback). All 30 svd/pca fails: components equal sklearn's up to sign (singular values and first component pass rtol; e.g. comp[1] 0.0582 vs -0.0582) -- sklearn >= 1.5 changed `svd_flip` to V-based sign; not a port issue.
  - glm (with open_data simple.txt/creditcard.csv/credit.feather): 49 pass / 14 fail / 13 skip. Fails: sklearn `normalize=` kwarg removed (10), pandas `Series.append` removed (1), `h2o4gpu.linear_model.logistic` module path (1), and 2 `test_elastic_net_ptr_driver` that pass after `pyarrow<15` (numpy 1.26).
- Shipped tests `tests/python/open_data/system` + `tests/python/small` (with conftest shim; shipped `test_glm_ipums.py` itself uses `h2o4gpu.ElasticNetH2O`, which upstream no longer exports -- upstream breakage): 11 pass / 5 fail; the 5 are missing datasets `data/xfullhyatt_2k.csv` and `data/bnp.feather` -- not in `s3://h2o-public-test-data` (listed 5000 keys under `h2o4gpu/`, and `smalldata/` prefixes) -> environment gap. `ipums_1k.csv` is at `h2o4gpu/open_data/ipums_1k.csv`. No aws cli: fetch with `curl https://h2o-public-test-data.s3.amazonaws.com/h2o4gpu/open_data/<file>`.

### Commit 3 `e7727e7` -- GPU module linked with -fopenmp via clang
- The `_ch2o4gpu_gpu` SWIG module links with the HIP compiler (it contains HIP objects); its `LINK_FLAGS ${OpenMP_CXX_FLAGS}` made clang add `libomp.so` + `libomptarget.so` NEEDED next to `libgomp`. Removed for the HIP module only. Side effect: clang's `-fopenmp` link had also injected RUNPATHs to the ROCm lib dirs; without it the module needs the ROCm lib dir on the loader path. `/opt/rocm` installs register it via ld.so.conf.d; with the pip SDK use `LD_LIBRARY_PATH=$(rocm-sdk path --root)/lib`.

### Commit 4 `2012add` -- roctx (ruling 5)
- `hip_compat/nvToolsExt.h` -> `roctracer/roctx.h`: `nvtxEventAttributes_t`, `nvtxRangePushEx/PushA/Pop` -> `roctxRangePushA/roctxRangePop`. CMake `if($ENV{USENVTX})` in the HIP branch: `find_library(roctx64)`, include dir, link into gpuh2o4gpu and cpuh2o4gpu (the CPU lib also compiles glm.h with USE_NVTX). Same switches as CUDA: `USENVTX` links, `USE_NVTX` (never defined by upstream's build; pass via CMAKE_CXX_FLAGS/CMAKE_HIP_FLAGS) emits ranges.
- Test: `USENVTX=1 cmake -S src -B src/build-nvtx -DUSE_CUDA=OFF -DUSE_HIP=ON -DCMAKE_PREFIX_PATH=$R -DCMAKE_CXX_FLAGS=-DUSE_NVTX -DCMAKE_HIP_FLAGS=-DUSE_NVTX`; GLM ElasticNetH2O fit (`agent_space/nvtxpkg/glm_solve.py`, package copy with the build-nvtx module) under `rocprofv3 --marker-trace --kernel-trace --output-format csv`: `glm_marker_api_trace.csv` has 1087 `MARKER_CORE_RANGE_API` records (MDnew, P1r_cholesky_svx, project, resid, adaprho, ... 72 each), equal to the 1087 `START:` lines the PUSH_RANGE macro prints; fit converged (corr 0.99999). Legacy `libroctx64` markers are captured by rocprofv3 on ROCm 10.1. Default build: no roctx NEEDED.
- After commit 4: `h2o4gpu_test` 24/24 at `2012add`.

### Commit 5 `39984a9` -- LightGBM (ruling 1)
- h2oai/LightGBM (`H2O_v4.6.0.2` = v4.6.0 + h2o) has no `USE_ROCM`; upstream lightgbm-org/LightGBM added it in v4.7.0 (#6086, #7039, #7185 all in v4.7.0). New submodule `LightGBM_rocm` -> upstream v4.7.0 (`8f7036f`). `patches/LightGBM_rocm/0001-h2o-customizations.patch` = the fork's C++ diff (GetTrees/Tree accessors for MOJO, `DAI_LIGHTGBM_AVOID_LOGGER`, FindGroups default-bin count). Fork's Python diff (no dask import, no pkg_resources) dropped: v4.7.0 compat.py no longer imports dask and dask.py imports lazily (verified `import lightgbm` leaves `dask` out of sys.modules).
- Makefile: `lightgbm_gpu`/`lightgbm_cpu` dispatch to `lightgbm_rocm`/`lightgbm_rocm_cpu` under `USE_HIP=ON` (`sh build-python.sh bdist_wheel --rocm`; `--precompile` silently packaged a CPU lib, do not use). `LIGHTGBM_DIR` in config.mk; install glob `lightgbm*-py3-none-*.whl` (upstream's wheel is `py3-none-linux_x86_64`).
- ROCm LightGBM serves `device_type='cuda'` (not the OpenCL `gpu`). 200000x30 regression, 100 rounds: cuda vs cpu RMSE 0.10112081684 / 0.10112081687, max pred diff 3.4e-8; rocprofv3: LightGBM::SplitKernel, FindBestFromAllSplitsKernel, ... Arch auto-detect wrote `CMAKE_HIP_ARCHITECTURES=gfx90a;gfx90a;gfx90a;gfx90a` (one per GPU; harmless).

### Commit 6 `ccde3a9` -- xgboost (ruling 1)
- AMD-Ecosystem/xgboost: no push permission (cannot host the h2o patches there); MOAT has no record for xgboost/LightGBM. Branches: release_2.1.3-rocm (2025-04), release_3.0.0-rocm, dev-rocm, master-rocm, release/3.1.1, release/3.2.0 (default, 2026-07). Chose `release_2.1.3-rocm` (`ee5234f`): closest to h2oai's 2.1.4 (same Python API, h2o patch applies unchanged).
- h2o patches the ROCm branch lacked (h2oai/xgboost `h2oai` vs dmlc v2.1.4, per its PORT.md): between-bin quantile (quantile.cc/.cu), XGBFI (`src/analysis/xgbfi.*`, c_api, core.py `get_feature_interactions`), `DAI_XGBOOST_AVOID_LOGGER`, sklearn pred_leaf/pred_contribs/approx_contribs, dask ntree_limit -> `patches/xgboost_rocm/0002`.
- Build failure #1 on ROCm 10.1: `reference to 'hip' is ambiguous` in rocThrust `cross_system.h` -- the branch's `src/common/cuda_to_hip.h` declared an empty `namespace thrust { namespace hip {} namespace cuda = thrust::hip; }`, colliding with rocThrust's versioned inline namespace. Fix `patches/xgboost_rocm/0001`: include `thrust/system/hip/detail/execution_policy.h`, alias inside `THRUST_NAMESPACE_BEGIN/END`. That was the only error; full build passes. (release/3.2.0 has dropped that block.)
- Python wheel: `python -m build --wheel --config-setting use_hip=True python-package` (packager reuses `lib/libxgboost.so`). `xgb.build_info()['USE_HIP'] == True`.
- Check: per-device sketched cuts differ by <= 3.3e-4 (same 7710 cuts), so plain DMatrix cuda vs cpu train RMSE 0.10149 vs 0.10114; with a shared `QuantileDMatrix` held-out RMSE 0.1041118556 (cuda) vs 0.1041118550 (cpu); binary acc 0.97306 vs 0.97324. XGBFI DataFrame and pred_leaf (5,20)/pred_contribs (5,31) (sum == margin) on a cuda model.
- xgboost_prev (h2oai legacy 1.5 fork, loader for old models): no ROCm branch exists; under `USE_HIP=ON` config.mk sets `XGBOOST_TARGET=libxgboost-cpu`. Its lib builds; its wheel step fails on current setuptools (`InvalidVersion: '1.4.0-SNAPSHOT'`) -- pre-existing, independent of ROCm (the CUDA path hits it too). Not fixed (CUDA path's submodule). **Question for jeff:** is a CPU-only xgboost_prev under USE_HIP acceptable, or should it be dropped from the ROCm wheel?
- Upstream note: the CUDA `make xgboost` recipe calls `xgboost/Makefile2`, which the current h2oai 2.1.4 submodule no longer has (it builds via `scripts/build_cuda128.sh`) -- pre-existing upstream staleness, untouched.

### Full ROCm wheel (at `ccde3a9`)
```
R=$(rocm-sdk path --root); export PATH=$FLEET_VENV/bin:$R/bin:$PATH CMAKE_PREFIX_PATH=$R OMP_NUM_THREADS=$MAX_JOBS CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
make cpp USE_HIP=ON
make lightgbm_gpu install_lightgbm_gpu lightgbm_cpu install_lightgbm_cpu USE_HIP=ON PYTHON=$FLEET_VENV/bin/python
make xgboost install_xgboost USE_HIP=ON PYTHON=$FLEET_VENV/bin/python
make py USE_HIP=ON PYTHON=$FLEET_VENV/bin/python      # after make apply-sklearn
```
Wheel bundles `_ch2o4gpu_gpu.so`, `lightgbm_gpu/lib/lib_lightgbm.so` (HIP), `lightgbm_cpu/...`, `xgboost/lib/libxgboost.so` (HIP).
- `tests/python/open_data/gbm` (LD_LIBRARY_PATH=$R/lib): 16 pass / 4 fail / 17 skip (data). Pass: test_xgboost covtype[0,1], dtinput hist + gpu_hist, 12 test_gpu_prediction_pickledmodel. Fail: 4 test_lightgbm `cv() got an unexpected keyword argument 'early_stopping_rounds'` (LightGBM >= 4.0 API; same on CUDA 4.6).
- Adapted copy `agent_space/shipped/adapted/test_lightgbm_rocm.py` (callbacks, `device_type='cuda'`, + lightgbm_dynamic selects lightgbm_gpu) + test_xgboost.py under rocprofv3: 7 passed; 334014 dispatches incl. 255381 `LightGBM::*` and 7968 xgboost kernels.
