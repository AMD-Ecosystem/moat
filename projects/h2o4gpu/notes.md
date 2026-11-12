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

### State at end of this porter round (2026-11-04, linux-gfx90a)
- Fork `moat-port` tip `ccde3a9` (6 commits on upstream `97279de`). Fresh configure+build of the tip: `h2o4gpu_test` 24/24. Tracked tree clean (`git status --ignore-submodules=untracked`).
- Lessons promoted to `cuda-to-rocm` `references/fault-classes.md` (Headers, includes and build) + SKILL.md index: CUDART_VERSION in shims, rocThrust `cuda/error.h` / `namespace thrust::hip`, hipcub/rocthrust targets leaking `--offload-arch`, one OpenMP runtime in mixed g++/HIP targets.
- `utils/check.py` `commits` gate fails on `97279de` only: the fork mirror `AMD-Ecosystem/h2o4gpu:master` is at `5e733c6`, one commit behind upstream `97279de`, so base..moat-port includes upstream's own commit. **Needs a person:** fast-forward the fork's `master` to upstream `97279de` (not a port branch, so this round did not push it). Also per the standing rule the fork's GitHub default branch should become `moat-port` (a person's fork setting).
- Windows (ruling 3): not attempted. Windows goes last, only after wave64 and wave32 pass at the same head; wave32 has not run, and this Linux host cannot build or validate Windows. Expected size per plan open question 3: POSIX host code (`sys/time.h`, `unistd.h`, `sys/resource.h` in kmeans/CPU/common), a bash Makefile, `.so` SWIG modules and `strip`, plus Windows builds of LightGBM 4.7 ROCm and AMD's xgboost ROCm branch -- likely the "too cumbersome" case of ruling 3; decide on the windows-gfx1151 host.
- Open questions for jeff: (1) xgboost_prev CPU-only under USE_HIP acceptable? (2) xgboost ROCm branch choice `release_2.1.3-rocm` (version-matched to h2oai 2.1.4, needs one rocThrust compat patch) vs AMD's maintained `release/3.2.0` (current ROCm, but a 2.1 -> 3.2 Python API jump and a re-port of the h2o patches).
- For the reviewer/validator: GPU runs need `LD_LIBRARY_PATH=$(rocm-sdk path --root)/lib` with the pip ROCm SDK (see commit 3 note). Recovered/adapted out-of-tree tests and the oracle are in `agent_space/` on this host only (container-local); their content and results are recorded above.

### Porter round close-out (2026-11-05, linux-gfx90a)
- No fork change: `moat-port` stays at `ccde3a9`. jeff's 2026-11-04 rulings (plan.md) settle the two open questions above: xgboost_prev stays CPU-only under `USE_HIP`, and xgboost stays on `release_2.1.3-rocm`.
- surface.json accounted for: 20 components covered with the commit and evidence from these notes; `vendored:nvidia-eula-helpers` (removed in 3c713ae, ruling 4), `interface:r` (no native code) and `platform:windows` (ruling 3 sequences it last; the windows gate is still owed) added to scoped_out beside the planner's six. `python3 utils/surface.py check h2o4gpu` clean with the stage at `ported`.

## CUDA compile check 2026-11-05 (ccde3a9)

CUDA no-regression compile check on linux-gfx950 (mi355 host), nvcc 12.8 (V12.8.93), gcc 13.3.0, cmake 3.31.6, SWIG 4.2.1, OpenBLAS 0.3.26. No role taken, no state changed, no GPU run.

- Trees: fork `moat-port` at `ccde3a9` and the control upstream `97279de` (the fork's `master` is now at `97279de`). The CUDA submodule pointers are identical in both: `xgboost` `4235811`, `xgboost_prev` `0ad357a`, `LightGBM` `42043b9`, `tests/googletest` `2fe3bd9`; `.gitmodules` keeps their h2oai URLs. `LightGBM_rocm`/`xgboost_rocm` are only added, and the Makefile/config.mk route to them only under `USE_HIP=ON` (`LIGHTGBM_DIR`/`XGBOOST_DIR` default to `LightGBM`/`xgboost`). Only `tests/googletest` was initialized; the gbm submodules were not built.
- One arch: `GPU_COMPUTE_VER` is a plain `SET` (not a cache var), so both trees got the same scratch-only sed: `60;61;70;75;80;90` -> `80`, and the Blackwell `;100;120` append dropped. Result `-gencode=arch=compute_80,code=sm_80;-gencode=arch=compute_80,code=compute_80`.
- Commands (same for both, `t` = ctrl|port):
  ```
  export PATH=/usr/local/cuda-12.8/bin:$FLEET_VENV/bin:$PATH
  cmake -S $t -B $t/build -DCMAKE_BUILD_TYPE=Release -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 -DPython_EXECUTABLE=$FLEET_VENV/bin/python -DSWIG_EXECUTABLE=$FLEET_VENV/bin/swig
  cmake --build $t/build -j32 -- -k
  ```
- Result: both PASS. All targets build: `libcpuh2o4gpu.a`, `libgpuh2o4gpu.a`, `_ch2o4gpu_cpu.so`, `_ch2o4gpu_gpu.so` (NVML stub linked), `h2o4gpu_test` (gtest; arima test macros compile to `ASSERT_FLOAT_EQ`). Build time about 25 s each.
- Errors: none from the port. Environment pitfall that hits both trees the same way: SWIG 4.5.1 (pip `swig` latest) fails `_ch2o4gpu_cpu` with `'PyString_Check' was not declared` / `'PyInt_Check'` (the `.i` files use Python 2 API that SWIG 4.3+ no longer shims). Use SWIG <= 4.2 (Ubuntu 24.04 apt `swig` is 4.2; pip `swig==4.2.1` into `$FLEET_VENV` works).
- Warnings: the project compiles with `-w` (CXX and NVCC), so compiler warnings are suppressed in both. Normalized warning sets (13 unique lines each) are identical apart from a line shift: CMake `CMP0146` FindCUDA dev warning at CMakeLists.txt:106 -> :111 (the 5 added lines above it), CMake deprecations from `cmake_policy(CMP0078 OLD)` and googletest's `cmake_minimum_required`, and the CUDA 12.8 NVTX2 `#pragma message` deprecation from `nvToolsExt.h`. No new warnings.
- The CUDA build sees no HIP pieces: no `hip_compat` in any include path or build file, and no reference to `LightGBM_rocm`/`xgboost_rocm` in the build tree.
- Binary diff (`libgpuh2o4gpu.a`): objects `bwcheck.cu.o` and `p2pbwcheck.cu.o` are gone (3c713ae), and so are their symbols (`bwcheck`, `p2pbwcheck`, `output*Matrix`, `printResults*`, `checkP2Paccess`, `delay`, `sMemory*`). Otherwise the symbol sets are the same apart from anonymous-namespace hash renames and the kmeans|| lambda closure types (`kmeans_parallel` symbols: 54 vs 54). `kmeans_h2o4gpu.cu.o` sm_80 fatbin 2976360 -> 3219224 B (+8%), PTX 2979409 -> 3134411 B. The growth sits in the four CUB `DeviceReduce{,SingleTile}Kernel` instantiations over the counting lambda (about 1875 -> 3010 PTX lines each), and it is the minstd_rand `discard` modular arithmetic (`mul.lo ... 2147483647`) emitted twice as often. Cause: `device` is now a captured host loop index and no longer a device-side `cudaGetDevice` (ac069e2). It is the same value, because the loop has just done `cudaSetDevice(i)`, so behaviour is unchanged; the size change is nvcc's codegen. h2o4gpuglm.cu.o +64 B, projector_direct_dense.cu.o -32 B.
- Not done: GPU test run (none on this host by instruction); gbm submodule builds (`make xgboost`/`lightgbm_gpu` CUDA recipes are untouched by the port, and the notes above record that the `xgboost/Makefile2` recipe is already stale upstream).
- Build trees deleted after the check.
## Review 2026-11-05 (linux-gfx942, MI300, ROCm 10.1 pip SDK)

### Port Review: h2o4gpu (moat-port `ccde3a9` vs master `97279de`)

#### Summary
A HIP port of h2o4gpu's GPU solvers via `hip_compat/` CUDA-named headers and a `USE_HIP` CMake branch, plus ROCm builds of LightGBM (upstream v4.7.0) and XGBoost (AMD-Ecosystem release_2.1.3-rocm) with h2o's patches. Verdict: **Request Changes**, three findings.

#### Port Correctness
1. **xgboost_prev ships without its library in the ROCm wheel, so it cannot be imported** (`Makefile:108-111`, `Makefile:266-268`, `DEVEL.md:213`). jeff's 2026-11-04 ruling keeps xgboost_prev under USE_HIP "so models saved by older releases still load in the ROCm wheel". `make xgboost_prev USE_HIP=ON` builds `xgboost_prev/lib/libxgboost.so` (CPU) and then fails in its own `python setup.py bdist_wheel` (`InvalidVersion: '1.4.0-SNAPSHOT'`). As a result `install_xgboost_prev` has no wheel to install, and `make py` packages `xgboost_prev/*.py` with no `lib/`. From the installed ccde3a9 wheel on this host: `import xgboost_prev` -> `XGBoostLibraryNotFound` (candidates `site-packages/xgboost_prev/lib/libxgboost.so`, ...). The cause is pre-existing and not ROCm-specific: upstream's CUDA builder pins `setuptools~=83.0` (`Dockerfile.build-cuda12:37`), so the CUDA recipe very likely produces the same lib-less package. Fix (ROCm-scoped, verified): under `USE_HIP=ON`, copy `xgboost_prev/lib/libxgboost.so` into `xgboost_prev/python-package/xgboost/lib/` after the lib builds, instead of depending on the broken wheel step. With the lib in place, `xgboost_prev` 1.4.0-SNAPSHOT trains, saves and reloads a model on the CPU (prediction max diff 0.0); `gpu_hist` raises as expected for a CPU build. **Question for jeff:** is the ROCm-only packaging fix acceptable, given the CUDA path probably has the same gap? Or should the fix cover both paths, or should the ruling be revisited?

#### Build System
2. **Without the ROCm lib dir on the loader path, the GPU module fails to load and every solver silently falls back to the CPU** (`CMakeLists.txt:357`, commit `e7727e7`). Dropping the HIP module's `-fopenmp` link flag also dropped the RUNPATH entries clang had been adding, so `_ch2o4gpu_gpu.so` now has `RUNPATH [/usr/lib/gcc/x86_64-linux-gnu/13]` only. On this host, the installed wheel without `LD_LIBRARY_PATH` gives `device_count(-1) == (0, 0)`. `GPUlib.get()` swallows the load error, so this is the silent-CPU-fallback failure the plan ranks as its top risk (risk 1); the shipped tests still pass. e7727e7's message and the new `fault-classes.md` paragraph ("a `/opt/rocm` install registers its lib dir with ld.so.conf") present loader registration as automatic. That is not verified here (no /opt/rocm on either host), and it is false for the pip/TheRock SDK this port was built and tested with. Fix: give the HIP SWIG module a RUNPATH to the ROCm library directory it was built against (for example `BUILD_RPATH`/`INSTALL_RPATH` from `$<TARGET_FILE_DIR:hip::amdhip64>`); at minimum, document the loader requirement in DEVEL.md's USE_HIP section. Then reword the commit body and the skill lesson to match. Confidence that `/opt/rocm` packages do not register the path themselves: low. The pip-SDK failure is confirmed.

#### Records
3. **surface.json lists `platform:windows` under `scoped_out`** (`projects/h2o4gpu/surface.json:403`). Its reason says "Not a cut", but `scoped_out` is the list of things the port deliberately does not do. Ruling 3 keeps Windows in scope (attempt last; one final commit, or report that it is too cumbersome), and the `windows` gate is required and unwaived. Record it as owed work, not a scope cut. If `surface.py check` cannot express "in scope, not yet done" at `ported`, raise that as a tooling question for jeff rather than filing the item under scoped_out.

#### CUDA path
- No finding. The concurrent `## CUDA compile check 2026-11-05 (ccde3a9)` above (mi355 host, nvcc 12.8) builds every CUDA target of ccde3a9 and of upstream 97279de, with identical warning sets and no hip_compat in any CUDA include path. That settles the CUDA no-regression question for this head. My own inspection agrees: every shared-source edit is a `__HIPCC__`/`__HIP_PLATFORM_AMD__` gate, a `__host__` qualifier, a captured loop index equal to the old device-side `cudaGetDevice` result, an include utils.cuh already uses, or a test macro that expands to `ASSERT_FLOAT_EQ`. The CUDA submodule pins are unchanged. The fix round for findings 1-2 needs the same nvcc check at its new head.

### Verification evidence (this host, gfx942, HIP_VISIBLE_DEVICES=1, `ulimit -c 0`)
- Build: `cmake -S src -B src/build-hip -DUSE_CUDA=OFF -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_PREFIX_PATH=$R -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$FLEET_VENV/bin/python && cmake --build src/build-hip -j56`: clean. Environment note: pip `swig` 4.5.1 breaks upstream's untouched `ch2o4gpu_cpu` wrapper (`PyString_Check`/`PyInt_Check` undeclared; SWIG >= 4.3 dropped the Python 2 compat macros), so use SWIG 4.2.x (`pip install swig==4.2.1`, or apt's 4.2.0). Pre-existing and not port-related.
- `h2o4gpu_test`: 24/24 pass, 3 consecutive runs.
- ARIMA tolerance fact-check (`agent_space/arima/drv.cu` linking `libgpuh2o4gpu.a`, float and double fits): d1p1q1 phi float -1.03694952 / double -1.0369515 / golden -1.0369391; theta 1.11547148 / 1.11547344 / 1.1154615; d0p2q2 theta[1] -0.616015851 / -0.616016071 / -0.61601555. These values are bit-identical to gfx90a. ROCm float sits about 2e-6 from float64, while the cuSOLVER golden sits about 1.2e-5 from it, so the ROCm result is the more accurate one. The 2e-5 relative bound (2.07e-5 absolute for phi) leaves about 2x headroom over the 1.04e-5 deviation. Justified.
- `make cpp USE_HIP=ON`: arch auto-detected gfx942; verify_gpu_lib.sh PASS. `make lightgbm_gpu/cpu install_* USE_HIP=ON`, `make xgboost install_xgboost USE_HIP=ON`: both patch sets applied cleanly on first build, and `git apply --reverse --check` then passes, so rebuilds skip re-applying. GPU lib_lightgbm and libxgboost link libamdhip64; lightgbm_cpu does not. `make apply-sklearn && make py USE_HIP=ON` -> wheel (needs `cython==3.0.11` etc. in the venv as the porter recorded).
- Patch fidelity: the +/- lines of `patches/xgboost_rocm/0002` equal `git diff v2.1.4 4235811` over the ten C++/Python source files (the remaining diff is CI/Docker/version only). The +/- lines of `patches/LightGBM_rocm/0001` equal `git diff v4.6.0 42043b9` over the five C++ files. The submodule pins match `v4.7.0` (8f7036f) and `release_2.1.3-rocm` (ee5234f).
- Python GPU oracle from the installed wheel (`agent_space/oracle/oracle.py`): `device_count(-1)` (1, 1). KMeansH2O v_measure 1.000000, center dist 6.8e-6 (f32) / 1.0e-14 (f64). TSVD sv rel err 1.3e-7 / 4.4e-16. PCA ev 4.5e-7 / 1.4e-15. ElasticNetH2O r2 0.999962 = sklearn ridge 0.999962. ALL PASS. rocprofv3 kernel trace: 10006 dispatches, including kmeans:: 94, cml:: 1442, h2o4gpu:: 230, rocBLAS/rocSOLVER/rocPRIM.
- `tests/python/open_data/gbm`: 16 pass / 4 fail / 17 skip (same as gfx90a; the 4 are `lgb.cv(early_stopping_rounds=)`, removed in LightGBM 4.0). `tests/python/open_data/system`: 2/2.
- h2o GBM patches on the GPU (`agent_space/gbm/gbm_check.py`, under rocprofv3): xgboost cut points for feature 0 are the between-bin midpoints `[0.5, 1.5, ..., 8.5]` on both cpu and cuda. `get_feature_interactions` returns a DataFrame; `pred_contribs` sums to the margin; sklearn `pred_leaf`/`pred_contribs` work; `DAI_XGBOOST_AVOID_LOGGER=1` prints `[XGBoost]` lines. LightGBM `device_type='cuda'` vs cpu: 100000x20, 50 rounds, max pred diff 7.9e-7. Trace: 24306 LightGBM:: and 370 xgboost kernels. The LightGBM C++ MOJO accessors (`GetTrees`, Tree split getters) compile, but nothing in-tree exercises them.
- roctx (ruling 5): `USENVTX=1` + `-DUSE_NVTX` build links `libroctx64.so.4`; the default build does not. ElasticNetH2O fit under `rocprofv3 --marker-trace`: 1115 marker records = 1115 `START:` lines, corr 0.99999.

## Porter fix round 2026-11-05 (linux-gfx90a, MI250X/MI250, ROCm 10.1 pip SDK) -- answers Review 2026-11-05

Fork `moat-port` `ccde3a9` -> `f2ecfb0` (3 new commits, no amend, no force-push). No review PR is recorded, so there are no threads to answer.

### Finding 1 -> `4e5e4331` "Package the previous XGBoost library without its wheel step" (no `[ROCm]`; jeff's ruling: shared packaging fix for both the CUDA and ROCm wheels, split-out-able)
- Reproduced at ccde3a9: `make -C src xgboost_prev USE_HIP=ON` builds `xgboost_prev/lib/libxgboost.so` (79 MB) then `Makefile2:47: libxgboostp3` fails, `packaging.version.InvalidVersion: Invalid version: '1.4.0-SNAPSHOT'` (venv setuptools 79.0.1).
- Fix (Makefile only): the `xgboost_prev` recipe passes `-o libxgboostp3` to the submodule's `make -f Makefile2`, so the wheel step is skipped and the target succeeds for every `XGBOOST_TARGET` (libxgboost / libxgboost2 / libxgboost-cpu all end in p3); `install_xgboost_prev` now does `mkdir -p xgboost_prev/python-package/xgboost/lib && cp xgboost_prev/lib/libxgboost.so` there instead of pip-installing the missing wheel. `src/interface_py/Makefile` symlinks that package dir as `xgboost_prev`, and `setup.py get_packages` walks every subdir, so `xgboost_prev/lib/libxgboost.so` ships in the h2o4gpu wheel. The old wheel install had only put the same .py files plus `lib/libxgboost.so` there.
- Split-out check: `git cherry-pick 4e5e4331` onto `origin/master` (97279de) applies cleanly (Makefile 6+/3-).
- `f2ecfb0c` adds `make xgboost_prev install_xgboost_prev USE_HIP=ON` to DEVEL.md's ROCm command block (the prose already named the target).
- Proof from the installed f2ecfb0 wheel, LD_LIBRARY_PATH unset (`agent_space/oracle/xgbprev.py`): `xgboost_prev 1.4.0-SNAPSHOT`, lib `site-packages/xgboost_prev/lib/libxgboost.so`; binary:logistic hist 5000x10, 30 rounds, train acc 0.9876; save/reload `.model` and `.json` max pred diff 0; `gpu_hist` raises (CPU build, as ruled).

### Finding 2 -> `074608c0` "[ROCm] Give the GPU Python module a runpath to the ROCm libraries"
- Reproduced at ccde3a9: `_ch2o4gpu_gpu.so` RUNPATH `[/usr/lib/gcc/x86_64-linux-gnu/13]` only, while `h2o4gpu_test` (an executable clang links) has `.../_rocm_sdk_devel/lib`. Cause: the ROCm lib dir is a HIP-compiler implicit link dir, so CMake drops it from the runpath of HIP-linked targets; clang adds it for executables only, and clang's `-fopenmp` link was what had added it for the module.
- Fix: `SET_TARGET_PROPERTIES(${SWIG_MODULE_ch2o4gpu_gpu_REAL_NAME} PROPERTIES BUILD_RPATH $<TARGET_FILE_DIR:hip::amdhip64> INSTALL_RPATH ...)` in the USE_HIP branch. OpenMP fix kept: NEEDED has libgomp.so.1 and no libomp/libomptarget. After: RUNPATH `[<venv>/_rocm_sdk_devel/lib:/usr/lib/gcc/x86_64-linux-gnu/13]` (the CMAKE_PREFIX_PATH spelling of the SDK root); `strip --strip-unneeded` in `make cpp` keeps it.
- The new commit body states the loader requirement correctly; e7727e7's body is not amended (rule: new commits only). Its "loader configuration that a ROCm install provides" sentence stays in history, superseded by 074608c0. The skill lesson in `references/fault-classes.md` is rewritten: a separate paragraph on HIP-linked shared libraries getting no ROCm RUNPATH, without the unverified ld.so.conf claim. SKILL.md index gets one line.

### Verification at f2ecfb0 (this host, gfx90a, HIP_VISIBLE_DEVICES=0, `ulimit -c 0`, LD_LIBRARY_PATH unset for every run)
```
R=$($FLEET_VENV/bin/python -m rocm_sdk path --root); export PATH=$FLEET_VENV/bin:$R/bin:$PATH CMAKE_PREFIX_PATH=$R OMP_NUM_THREADS=$MAX_JOBS CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
cmake -S src -B src/build-hip -DUSE_CUDA=OFF -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=$R -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$FLEET_VENV/bin/python && cmake --build src/build-hip -j$MAX_JOBS
make cpp USE_HIP=ON PYTHON=$FLEET_VENV/bin/python
make lightgbm_gpu install_lightgbm_gpu lightgbm_cpu install_lightgbm_cpu USE_HIP=ON PYTHON=...
make xgboost install_xgboost USE_HIP=ON PYTHON=...
make xgboost_prev install_xgboost_prev USE_HIP=ON PYTHON=...
make apply-sklearn PYTHON=... && make py USE_HIP=ON PYTHON=...
$FLEET_VENV/bin/python -m pip install --no-deps --force-reinstall src/interface_py/dist/h2o4gpu-0.6.0+local.f2ecfb0c.dirty-cp311-cp311-linux_x86_64.whl
```
(`.dirty` = the h2o patches the ROCm make targets apply inside LightGBM_rocm/xgboost_rocm; reverted with `git -C <sub> checkout -- .` afterwards. Tracked fork tree clean with `--ignore-submodules=untracked`.)
- Fresh configure + build: `h2o4gpu_test` 24/24, 3 consecutive runs.
- Wheel contents: `h2o4gpu/libs/_ch2o4gpu_gpu.so`, `lightgbm_gpu/lib/lib_lightgbm.so`, `lightgbm_cpu/lib/lib_lightgbm.so`, `xgboost/lib/libxgboost.so`, **`xgboost_prev/lib/libxgboost.so`** (new).
- Oracle (`agent_space/oracle/oracle.py`, rewritten on this host; exits nonzero if `GPUlib.get()` is None or no GPU): `LD_LIBRARY_PATH = None`, `device_count(-1)` (1, 1). KMeansH2O v_measure 1.000000, center dist 1.6e-5 (f32) / 4.3e-14 (f64); TSVD cusolver sv rel err 1.4e-7 / 5.0e-16, power (tol 1e-5) 4.9e-5 / 3.1e-5; PCA ev 2.2e-7 / 2.2e-15; ElasticNetH2O r2 0.999956 / 0.999920 (sklearn ridge 0.999997 / 0.999993). ALL PASS. (A first draft compared TSVD power at 1e-10 in f64 and failed at 2.7e-5. That is the power method's own tol=1e-5 convergence, not precision; cusolver gives 4e-16.)
- GPU proof: the same oracle under `rocprofv3 --kernel-trace --output-format csv`, LD_LIBRARY_PATH unset: 42818 dispatches on one GPU agent, incl. kmeans:: 108, cml:: 6050, h2o4gpu:: 262, rocBLAS 15994, rocSOLVER 636, rocPRIM 12556.
- `pytest tests/python/open_data/system tests/python/open_data/gbm`: 18 pass / 4 fail / 17 skip, the same as the previous round (4 = `lgb.cv(early_stopping_rounds=)`, removed in LightGBM 4.0; skips = unsynced data / "Huge memory footprint"). xgboost gpu_hist and the GPU pickled-model predict tests pass without LD_LIBRARY_PATH.
- Environment: fresh job venv needed `cython==3.0.11 threadpoolctl psutil tabulate future joblib pytest-timeout meson-python ninja feather-format "pyarrow<15"` (pip, into `$FLEET_VENV` only); apt swig 4.2.0 already present.
- Not run here: the CUDA compile check at f2ecfb0 (runs separately on mi355).

### Finding 3 (records)
- `surface.json` `platform:windows` stays under `scoped_out` as a placeholder. Its reason now opens with "PLACEHOLDER, NOT A SCOPE CUT" and says Windows is still owed under ruling 3, the windows gate is required and unwaived, and the entry moves when the tooling gains an owed state (that change is being handled separately). `surface.py check h2o4gpu`: clean.

## CUDA compile check 2026-11-05 (f2ecfb0)

CUDA no-regression compile check on linux-gfx950 (mi355 host), nvcc 12.8 (V12.8.93), gcc 13.3.0, cmake 3.31.6, SWIG 4.2.1 (pip, into the job venv), apt OpenBLAS 0.3.26. No role taken, no state changed, no GPU run. Nothing needed building from source.

- Trees: fork `moat-port` at `f2ecfb0` and control upstream `97279de` (fork `master`), as two worktrees of one clone under the job directory. CUDA submodules unchanged in both: `xgboost` `4235811`, `xgboost_prev` `0ad357a`, `LightGBM` `42043b9`, `tests/googletest` `2fe3bd9`, all still on their h2oai/google URLs in `.gitmodules`; the port only adds `LightGBM_rocm` (`8f7036f`) and `xgboost_rocm` (`ee5234f`). Only `tests/googletest` was initialized.
- Delta since the ccde3a9 check: `4e5e4331` (Makefile `xgboost_prev`/`install_xgboost_prev`, shared by both paths), `074608c0` (CMake RUNPATH, inside `if(USE_HIP)` only), `f2ecfb0c` (DEVEL.md).
- One arch, same scratch-only sed in both trees: `SET(GPU_COMPUTE_VER 60;61;70;75;80;90)` -> `80`, the `;100;120` append commented out. Result `-gencode=arch=compute_80,code=sm_80;-gencode=arch=compute_80,code=compute_80`.
- Commands (`t` = ctrl|port):
  ```
  export PATH=/usr/local/cuda-12.8/bin:$FLEET_VENV/bin:$PATH
  git -C $t submodule update --init tests/googletest
  cmake -S $t -B $t/build -DCMAKE_BUILD_TYPE=Release -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 -DPython_EXECUTABLE=$FLEET_VENV/bin/python -DSWIG_EXECUTABLE=$FLEET_VENV/bin/swig
  cmake --build $t/build -j32 -- -k
  ```
- Result: both PASS (configure rc 0, build rc 0, 0 lines matching `error`). Both build `libcpuh2o4gpu.a`, `libgpuh2o4gpu.a`, `_ch2o4gpu_cpu.so`, `_ch2o4gpu_gpu.so`, `h2o4gpu_test` (24 tests listed by `--gtest_list_tests`; not run).
- Errors: none.
- Warnings: no new warnings. The project builds with `-w`; the normalized warning/deprecation sets (14 unique lines each, paths and line numbers stripped) are identical. The only shift is the CMake `CMP0146` FindCUDA dev warning at CMakeLists.txt:106 -> :111.
- No HIP pieces in the CUDA build: no `hip_compat`, `amdhip`, `LightGBM_rocm` or `xgboost_rocm` in any CMake-generated file. `_ch2o4gpu_gpu.so` NEEDED and RUNPATH (`/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/lib64/stubs`) are identical in both trees, so 074608c0 does not touch the CUDA module.
- Binary diff (`libgpuh2o4gpu.a`, 19266318 -> 19399008 B): `bwcheck.cu.o` and `p2pbwcheck.cu.o` gone with their symbols (3c713ae); every other object is byte-size identical except `kmeans_h2o4gpu.cu.o` 4027424 -> 4270608 B (the captured-device CUB instantiations recorded at ccde3a9; `kmeans_parallel` symbols 54 vs 54) and `h2o4gpuglm.cu.o` +64 B. Symbol differences after hash normalization are only those two groups. Same picture as the ccde3a9 check.
- Makefile CUDA path (4e5e4331), dry run only (`make -n xgboost_prev install_xgboost_prev`, default `USE_HIP` off): the recipe now runs `make -f Makefile2 ... -o libxgboostp3 libxgboost` on `xgboost_prev`, and the install step does `mkdir -p xgboost_prev/python-package/xgboost/lib && cp xgboost_prev/lib/libxgboost.so ...` instead of the pip wheel install. Cosmetic: the two new `#` lines are tab-indented inside the recipe, so make echoes them as shell comments when the target runs. `xgboost_prev` itself was not CUDA-built here: its `Utils.cmake` sed adds `35;37`, which nvcc 12.8 does not accept, the same in both trees.
- Build trees, worktrees and clone deleted after the check.
## Review 2026-11-05 round 2 (linux-gfx942, MI300, ROCm 10.1 pip SDK) -- moat-port f2ecfb0

### Port Review: h2o4gpu (moat-port `f2ecfb0` vs master `97279de`; delta `ccde3a9..f2ecfb0` reviewed first)

#### Summary
A HIP port of h2o4gpu's GPU solvers via `hip_compat/` CUDA-named headers and a `USE_HIP` CMake branch, plus ROCm builds of LightGBM (upstream v4.7.0) and XGBoost (AMD-Ecosystem release_2.1.3-rocm) with h2o's patches; this round packages xgboost_prev's library (4e5e433) and gives the GPU module a ROCm RUNPATH (074608c). Both previous findings are fixed and verified on this host. Verdict: **Request Changes**, one finding (plus one CUDA-side check still owed).

#### Port Correctness
1. **With both XGBoost libraries in one process, the process segfaults at exit** (`Makefile:113`, the `xgboost_prev` recipe; exposed by 4e5e433, which makes `xgboost_prev` importable for the first time in this wheel). Both `xgboost/lib/libxgboost.so` (2.1.3) and `xgboost_prev/lib/libxgboost.so` (1.4.0-SNAPSHOT) export `dmlc::ThreadLocalStore<xgboost::XGBAPIThreadLocalEntry>::Get()::inst` (and other ThreadLocalStore statics) as `STB_GNU_UNIQUE` TLS (`readelf -Ws --dyn-syms`). ld.so merges GNU_UNIQUE symbols process-wide even under ctypes' RTLD_LOCAL, so the two versions share one thread-local object with two different layouts. Crash: `__call_tls_dtors -> xgboost::XGBAPIThreadLocalEntry::~XGBAPIThreadLocalEntry() (xgboost/lib/libxgboost.so) -> free(0x86f)` (gdb). Reproducer, from the installed f2ecfb0 wheel, LD_LIBRARY_PATH unset, exit 139 in 3/3 runs, 3/3 again after restoring the shipped lib, and 6/6 for a longer script that also saves and reloads models:
   ```
   import numpy as np, xgboost_prev as xp, xgboost
   xgboost.build_info()
   rng = np.random.RandomState(0); X = rng.randn(5000, 10); y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
   d = xp.DMatrix(X, label=y)
   b = xp.train({"objective": "binary:logistic", "tree_method": "hist"}, d, 30); b.predict(d)
   try: xp.train({"tree_method": "gpu_hist"}, d, 1)   # CPU-only build raises
   except xp.core.XGBoostError as e: str(e)
   ```
   The trigger is order- and call-dependent (several simpler sequences exit 0), but the pattern is the one h2o4gpu's own `util/xgboost_migration.py:load_pkl` creates: the new xgboost is loaded and used, then xgboost_prev is imported and used in the same process, and on ROCm xgboost_prev is CPU-only, so old GPU-trained models are more likely to make it raise. Fix (verified): build xgboost_prev with `-fno-gnu-unique` (e.g. `CXXFLAGS=-fno-gnu-unique` in the recipe's environment so its first CMake configure picks it up). A rebuilt lib (`cmake -S xgboost_prev -B build -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=RELWITHDEBINFO -DCMAKE_CXX_FLAGS=-fno-gnu-unique`) has 0 UNIQUE dynamic symbols; with it swapped into the installed wheel the reproducer exits 0 in 5/5 runs, and train/save/reload, new-xgboost loading the old `.model`/`.json` (max pred diff 0) and `load_pkl` migration all still work. **Question for jeff:** the CUDA wheel very likely has the same hazard (both libs built with g++ there; not verified, no CUDA GPU here). Should the flag go in as a shared, non-ROCm commit next to 4e5e433 (my recommendation, same reasoning as that ruling), or ROCm-only?

#### Testing
2. **4e5e433 changes the CUDA wheel's xgboost_prev build and has not been exercised with the CUDA target.** `-o libxgboostp3` and the copy-based `install_xgboost_prev` run for `XGBOOST_TARGET=libxgboost` too, and the fix round's nvcc check at f2ecfb0 is not yet recorded. When the mi355 CUDA compile check runs at the next head, include `make xgboost_prev install_xgboost_prev` with the default (CUDA) target and confirm `xgboost_prev/python-package/xgboost/lib/libxgboost.so` exists. Validate before presenting, per the standing CUDA-compile rule.

#### Records (not blocking)
- `surface.json` `platform:windows` under `scoped_out` now says plainly that it is a placeholder, Windows is owed under ruling 3, and the gate is required and unwaived; status.json has no waiver and windows-gfx1151 is unblocked. The records do not misstate the scope. One gap: the reason says the tooling change for an "owed" state "is being handled separately", but there is no deferred item, branch, or surface.py change for it on any ref (checked `origin/main` log and `deferred.py pending`). Register it (`utils/deferred.py add ...`) so it is not invisible.

### Verification evidence (this host, gfx942, HIP_VISIBLE_DEVICES=1, `ulimit -c 0`, LD_LIBRARY_PATH unset everywhere)
- Delta review: CUDA submodule pins unchanged vs master (`xgboost` 4235811, `xgboost_prev` 0ad357a, `LightGBM` 42043b9, `tests/googletest` 2fe3bd9); 074608c's RUNPATH is inside `if(USE_HIP)`. The CMake CUDA branch is untouched. `-o libxgboostp3` is valid: `libxgboostp3` is a non-phony target of `xgboost_prev/Makefile2`, a prerequisite of all three `XGBOOST_TARGET`s, and its only action was `python setup.py bdist_wheel`. No other script consumes `xgboost_prev/python-package/dist`.
- Build: `cmake -S src -B src/build-hip -DUSE_CUDA=OFF -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_PREFIX_PATH=$R -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$FLEET_VENV/bin/python && cmake --build src/build-hip -j56`: clean. `_ch2o4gpu_gpu.so` RUNPATH `[<job>/venv-base-links/_rocm_sdk_devel/lib:/usr/lib/gcc/x86_64-linux-gnu/13]` (the CMAKE_PREFIX_PATH spelling), NEEDED has libgomp only (no libomp/libomptarget); RUNPATH survives `make cpp`'s strip and is in the wheel copy.
- `h2o4gpu_test`: 24/24 pass, 3 consecutive runs (ARIMA ROCm tolerance tests included; the tolerance fact-check from the previous review stands, the code is unchanged).
- Full wheel (same commands as the porter's fix round: `make cpp`, `lightgbm_gpu/cpu`, `xgboost`, `xgboost_prev`, `apply-sklearn`, `py`, all `USE_HIP=ON`): all succeed; verify_gpu_lib.sh PASS (gfx942); both patch sets apply cleanly (only git's trailing-whitespace warnings); `make xgboost_prev` now succeeds. Wheel ships `_ch2o4gpu_gpu.so`, `lightgbm_gpu/lib/lib_lightgbm.so`, `lightgbm_cpu/...`, `xgboost/lib/libxgboost.so`, `xgboost_prev/lib/libxgboost.so`. Submodule working trees reverted afterwards; fork tracked tree clean at f2ecfb0.
- Python GPU oracle (out of tree, vs scikit-learn 1.9.1 / numpy float64): `GPUlib.get()` loaded, `device_count(-1)` (1, 1) with no LD_LIBRARY_PATH. KMeansH2O v_measure 1.000000, center dist 5.0e-6 (f32) / 7.4e-15 (f64); TSVD cusolver sv rel err 4.1e-7 / 4.4e-16; TSVD power (n_iter=5000, tol=1e-9) 7.4e-6 / 8.8e-7 (at the default n_iter=100, tol=1e-5 it is 3.1e-3 / 1.4e-3 on this near-degenerate spectrum and converges as n_iter grows, so convergence, not precision); PCA ev 4.9e-7 / 1.8e-15; ElasticNetH2O r2 1.000000 = Ridge. ALL PASS. rocprofv3 kernel trace: 89486 dispatches on one GPU agent incl. kmeans:: 94, cml:: 1474, h2o4gpu:: 230, rocBLAS 26236, rocSOLVER 756, rocPRIM 14488.
- xgboost_prev from the wheel: 1.4.0-SNAPSHOT loads `site-packages/xgboost_prev/lib/libxgboost.so`; hist train acc 1.0; `.model`/`.json` save/reload max diff 0, also loaded by the new xgboost with max diff 0; `gpu_hist` raises (CPU build, as ruled); `load_pkl` migrates an emulated old pickle to `XGBClassifier`. Exit crash: finding 1.
- `pytest tests/python/open_data/system tests/python/open_data/gbm`: 18 pass / 4 fail / 17 skip, same as both previous rounds (the 4: `lgb.cv(early_stopping_rounds=)`, removed in LightGBM 4.0).
- GBM patches on the GPU (rocprofv3: 24306 LightGBM:: and 2728 xgboost:: kernels): xgboost cut points for an integer feature are the between-bin midpoints `0.5 ... 8.5` for both cpu and cuda sketching (plain DMatrix, so `quantile.cu` ran); `get_feature_interactions` returns a DataFrame; `pred_contribs` sums to the margin; sklearn `pred_leaf` (5, 10); LightGBM `device_type='cuda'` vs cpu max pred diff 1.4e-8.
- roctx (ruling 5): `USENVTX=1` + `-DUSE_NVTX` build links `libroctx64.so.4`, default build does not; ElasticNetH2O fit with that module under `rocprofv3 --marker-trace`: 1101 `MARKER_CORE_RANGE_API` records = 1101 `START:` lines, corr 0.9999999.
- Environment, not the port: on this 224-core host, scipy's bundled OpenBLAS 0.3.18 segfaults in `blas_shutdown` at exit about half the time with no h2o4gpu involved (a plain scikit-learn KMeans/PCA/Ridge script: 5/10 runs exit 139). `OPENBLAS_NUM_THREADS=32` stops it (10/10 exit 0). The validator should set it on large hosts so it does not mask finding 1 or a real crash.
- Not run here: CUDA compile check at f2ecfb0 (mi355; see item 2).

- 2026-11-05: jeff filed https://github.com/h2oai/h2o4gpu/issues/899 (xgboost_prev no longer builds with CUDA 12 or current setuptools).
## Porter fix round 2026-11-05 (linux-gfx90a, MI250X/MI250, ROCm 10.1 pip SDK) -- answers Review round 2 under jeff's xgboost_prev reversal

jeff reversed the xgboost_prev keep ruling (plan.md, last entry): drop it from the ROCm build. Fork `moat-port` `f2ecfb0` -> `9607b09` (2 new commits, no amend, no force-push). No review PR is recorded, so no threads to answer.

### `24e61690` Revert "Package the previous XGBoost library without its wheel step"
- Plain `git revert` of 4e5e433 (shared, non-[ROCm]) with a rationale body. The CUDA `xgboost_prev`/`install_xgboost_prev` recipes are upstream's again: `make -n xgboost_prev install_xgboost_prev` (no USE_HIP) prints the `make -f Makefile2 ... libxgboost` recipe and the wheel install from `python-package/dist`, and `git diff 97279de -- Makefile` has no added/removed xgboost_prev line at that commit.

### `9607b099` [ROCm] Do not build or bundle the previous XGBoost with USE_HIP
- Makefile: under `USE_HIP=ON`, `xgboost_prev` and `install_xgboost_prev` only echo "XGboost previous version is not built with USE_HIP=ON" (same `ifeq` pattern as `xgboost`/`lightgbm_gpu`), so `deps-install-with-*` still run through them. `src/interface_py/Makefile`: the `xgboost_prev` symlink is made only when `USE_HIP != ON`, so `setup.py get_packages` never sees it. `make/config.mk`: the USE_HIP `XGBOOST_TARGET=libxgboost-cpu` override (it existed only for xgboost_prev) removed. DEVEL.md: the `make xgboost_prev ... USE_HIP=ON` line removed, plus a sentence that load_pkl cannot convert 0.90 pickles in a ROCm build. CUDA path: dry run identical to upstream (above).
- Review round 2 finding 1 (GNU_UNIQUE TLS crash with both XGBoost libraries in one process) is closed: xgboost_prev is no longer in the ROCm process at all, so `-fno-gnu-unique` is not needed, and the question to jeff is moot. Finding 2 (4e5e433 not exercised with the CUDA target) is moot: 4e5e433 is reverted, the CUDA recipe is upstream's. The records note (register the surface owed-state tooling item) is done: deferred `surface-owed-state-for-in-scope-platforms`.

### Verification at 9607b09 (this host, gfx90a, HIP_VISIBLE_DEVICES=0, `ulimit -c 0`, LD_LIBRARY_PATH unset, OPENBLAS_NUM_THREADS=32 for every run)
Fresh clone; the wheel was built from the working tree identical to 9607b09 (committed after the build, no edits in between):
```
R=$($FLEET_VENV/bin/python -m rocm_sdk path --root); export PATH=$FLEET_VENV/bin:$R/bin:$PATH CMAKE_PREFIX_PATH=$R OMP_NUM_THREADS=$MAX_JOBS CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
git submodule update --init --recursive tests/googletest LightGBM_rocm xgboost_rocm
cmake -S src -B src/build-hip -DUSE_CUDA=OFF -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=$R -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$FLEET_VENV/bin/python && cmake --build src/build-hip -j$MAX_JOBS
make cpp USE_HIP=ON PYTHON=$FLEET_VENV/bin/python
make lightgbm_gpu install_lightgbm_gpu lightgbm_cpu install_lightgbm_cpu USE_HIP=ON PYTHON=...
make xgboost install_xgboost USE_HIP=ON PYTHON=...
make xgboost_prev install_xgboost_prev USE_HIP=ON PYTHON=...     # prints the not-built line twice, builds nothing
make apply-sklearn PYTHON=... && make py USE_HIP=ON PYTHON=...
$FLEET_VENV/bin/python -m pip install --no-deps --force-reinstall src/interface_py/dist/h2o4gpu-0.6.0+local.24e61690.dirty-cp311-cp311-linux_x86_64.whl
```
(the wheel's describe is the revert commit + dirty = the uncommitted 9607b09 edits and the h2o patches applied inside LightGBM_rocm/xgboost_rocm; submodule trees reverted with `git -C <sub> checkout -- .` afterwards; fork tracked tree clean with `--ignore-submodules=untracked`.)
- All steps rc 0. `xgboost_prev/lib` never created (the xgboost_prev submodule source does get checked out by `apply-sklearn`'s submodule init, and is still not bundled).
- Wheel: `unzip -l | grep -c xgboost_prev/` = 0. Ships `h2o4gpu/libs/_ch2o4gpu_gpu.so`, `lightgbm_gpu/lib/lib_lightgbm.so`, `lightgbm_cpu/lib/lib_lightgbm.so`, `xgboost/lib/libxgboost.so`. Installed venv: no `xgboost_prev*` in site-packages; `importlib.util.find_spec('xgboost_prev')` None.
- `_ch2o4gpu_gpu.so`: RUNPATH `[<job>/venv-base-links/_rocm_sdk_devel/lib:/usr/lib/gcc/x86_64-linux-gnu/13]`, NEEDED libgomp only (074608c / e7727e7 unchanged).
- `h2o4gpu_test`: 24/24 pass, 3 consecutive runs.
- `import h2o4gpu` OK (0.6.0). Oracle (`agent_space/oracle/oracle.py`, exits nonzero if GPUlib is None or no GPU): `device_count(-1)` (1, 1). KMeansH2O v_measure 1.000000, center dist 4.1e-6 (f32) / 6.7e-15 (f64); TSVD cusolver sv rel err 1.8e-7 / 3.0e-16; PCA ev 4.6e-7 / 3.8e-16; ElasticNetH2O (alpha 0, lambda 1e-6) r2 1.000000 = Ridge. ALL PASS. Under `rocprofv3 --kernel-trace`: 48519 dispatches on one GPU agent, kmeans:: 110, cml:: 7242, h2o4gpu:: 226, rocBLAS 18071, rocSOLVER 636, rocPRIM 14722.
- GBM (`agent_space/oracle/gbm.py`, 100000x20 regression, 80/20 split, 50 rounds): xgboost 2.1.3 `USE_HIP` True, shared QuantileDMatrix held-out RMSE cuda 0.1309159382 / cpu 0.1309159387; `get_feature_interactions` present. lightgbm_gpu 4.7.0 `device_type='cuda'` RMSE 0.12325394 / cpu 0.12329379. Under rocprofv3: 24286 `LightGBM::` and 5332 `xgboost::` kernels.
  - Observation for the reviewer (LightGBM_rocm unchanged this round, so not from this change): LightGBM cuda vs cpu predictions are not ulp-close row by row on this data -- median |diff| 7.5e-10 but max 4.1e-2, 0.04% of rows > 1e-3 (200000x30, 100 rounds: max 6.7e-2, 0.2% of rows; RMSE 0.12260690 vs 0.12254017). That is the pattern of occasional near-tie splits resolving differently (different histogram summation order); RMSE agrees to < 1e-3 relative. Earlier rounds' "max pred diff 3.4e-8 / 7.9e-7 / 1.4e-8" were on other data (y = linear + noise); a first strict 1e-4 max-diff check here failed for this reason and was replaced by the RMSE check.
- `pytest tests/python/open_data/system tests/python/open_data/gbm`: 18 pass / 4 fail / 17 skip, same as the three previous rounds (4 = `lgb.cv(early_stopping_rounds=)`, removed in LightGBM 4.0; skips = unsynced data / memory).
- What a user sees for xgboost_migration (`agent_space/oracle/migration.py`, from the installed wheel): (1) a pickle from the current xgboost: `load_pkl` returns the XGBClassifier, max pred diff 0.0 -- unaffected. (2) an emulated XGBoost 0.90 pickle (a Booster pickled with its handle replaced by `save_raw(raw_format='deprecated')`, which has no serialisation header, written with a reducer override) reaches exactly the migration branch and fails with a chained traceback:
  ```
  xgboost.core.XGBoostError: [..] xgboost_rocm/src/learner.cc:1186: Check failed: header == serialisation_header_: If you are loading a serialized model (like pickle in Python, RDS in R) or
  configuration generated by an older version of XGBoost, please export the model by calling
  `Booster.save_model` from that version first, then load it back in current version. See: ...
  During handling of the above exception, another exception occurred:
    File ".../h2o4gpu/util/xgboost_migration.py", line 23, in load_pkl
      import xgboost_prev  # pylint: disable=unused-import
  ModuleNotFoundError: No module named 'xgboost_prev'
  ```
  Judged clear enough (names the cause and the remedy, re-save with the old version), so xgboost_migration.py is unchanged (shared code; on CUDA the package is still bundled). DEVEL.md states the limitation.
- Guard note: the fleet guard blocks any Bash command whose text contains "pip install" (it matched a heredoc editing the Makefile's existing recipe and a commit message); file edits went through the editor tool, commit messages through files. No pip ran outside `$FLEET_VENV`.
- Not run here: the CUDA compile check at 9607b09 (mi355, separately). Its delta vs the f2ecfb0 check: the revert restores upstream's Makefile recipe; 9607b09 touches only `ifeq ($(USE_HIP),ON)` branches, a `USE_HIP != ON` shell guard around the xgboost_prev symlink, config.mk's USE_HIP block, and DEVEL.md.

### Upstream issue draft (jeff's ruling: "a short upstream issue tells the maintainers"; filing it is a person's action, not done here)
Title: `make xgboost_prev` cannot build with CUDA 12 or current setuptools
Body: The `xgboost_prev` target cannot produce the previous XGBoost on the current toolchain. Its recipe edits `xgboost_prev/cmake/Utils.cmake` to add compute capabilities 35 and 37, which CUDA 12 no longer accepts, and the submodule's wheel step (`python setup.py bdist_wheel` in `Makefile2`, target `libxgboostp3`) fails on current setuptools with `packaging.version.InvalidVersion: Invalid version: '1.4.0-SNAPSHOT'`. `install_xgboost_prev` then has no wheel to install, so `make py` bundles `xgboost_prev` without `lib/libxgboost.so`, and `import xgboost_prev` raises `XGBoostLibraryNotFound`; `h2o4gpu.util.xgboost_migration.load_pkl` needs it to convert XGBoost 0.90 pickles. Seen with CUDA 12.8 (nvcc rejects the 35;37 gencode) and setuptools 79.0.1.
- `utils/check.py` `commits` gate at 9607b09 flags two titles without `[ROCm]`: `4e5e433` (jeff's 2026-11-05 ruling made it a shared non-ROCm commit) and its revert `24e6169` (jeff asked for a plain `git revert`; the revert of a non-ROCm commit keeps git's title). Both deliberate. At PR prep the pair cancels out and can be dropped by the squash; until then this is a known gate message, not a new defect.

## CUDA compile check 2026-11-05 (ccde3a9)

Rerun requested at ccde3a9 (fleet job h2o-cuda3); the head is now `9607b09`, so it was also built in the same pass. CUDA no-regression compile check on linux-gfx950 (mi355 host), nvcc 12.8 (V12.8.93), gcc 13.3.0, cmake 3.31.6, SWIG 4.2.1 (pip, into the job venv), apt OpenBLAS 0.3.26. No role taken, no state changed, no GPU run. Nothing needed building from source.

- Trees: one clone of `AMD-Ecosystem/h2o4gpu` with three worktrees under the job directory: control `97279de` (fork `master`), `ccde3a9`, and `9607b09` (`moat-port` tip). CUDA submodules identical in all three: `xgboost` `4235811`, `xgboost_prev` `0ad357a`, `LightGBM` `42043b9`, `tests/googletest` `2fe3bd9`, still on their h2oai/google URLs in `.gitmodules`; the port only adds `LightGBM_rocm` `8f7036f` and `xgboost_rocm` `ee5234f`. Only `tests/googletest` was initialized.
- One arch, same scratch-only sed in all three: `SET(GPU_COMPUTE_VER 60;61;70;75;80;90)` -> `80`, the `;100;120` append commented out. Result `-gencode=arch=compute_80,code=sm_80` + `code=compute_80`.
- Commands (`t` = ctrl|port|head):
  ```
  export PATH=/usr/local/cuda-12.8/bin:$FLEET_VENV/bin:$PATH
  git -C $t submodule update --init tests/googletest
  cmake -S $t -B $t/build -DCMAKE_BUILD_TYPE=Release -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 -DPython_EXECUTABLE=$FLEET_VENV/bin/python -DSWIG_EXECUTABLE=$FLEET_VENV/bin/swig
  cmake --build $t/build -j32 -- -k
  ```
- Result: all three PASS (configure rc 0, build rc 0, 0 lines matching `error`). Each builds `libcpuh2o4gpu.a`, `libgpuh2o4gpu.a`, `_ch2o4gpu_cpu.so`, `_ch2o4gpu_gpu.so`, `h2o4gpu_test` (24 tests listed; not run).
- Errors: none.
- Warnings: no new warnings. The project builds with `-w`; the normalized warning/deprecation sets (14 unique lines each: CMake `CMP0146` FindCUDA dev warning, `cmake_policy(CMP0078 OLD)` and googletest `cmake_minimum_required` deprecations, CUDA 12.8 NVTX2 `#pragma message`) are identical across the three. Only shift: `CMP0146` at CMakeLists.txt:106 (control) -> :111 (ccde3a9, 9607b09).
- No HIP pieces in the CUDA build: no `hip_compat`, `amdhip`, `LightGBM_rocm` or `xgboost_rocm` in any CMake-generated file. `_ch2o4gpu_gpu.so` NEEDED (11) and RUNPATH (`/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/lib64/stubs`) identical in all three.
- Binary diff vs control: `libgpuh2o4gpu.a` 19266318 -> 19399008 B in both ccde3a9 and 9607b09; `bwcheck.cu.o`/`p2pbwcheck.cu.o` gone (3c713ae), `kmeans_h2o4gpu.cu.o` 4027424 -> 4270608 B (captured-device CUB instantiations, recorded at the first ccde3a9 check), `h2o4gpuglm.cu.o` +64 B. Same picture as the two earlier checks.
- ccde3a9 vs 9607b09: every object in both static libs has the same size and the same symbol set after anonymous-namespace hash normalization; the only byte differences are the per-compile `_GLOBAL__N__<hash>` names (path-dependent), e.g. `.rodata..._GLOBAL__N__9378f319_6_pca_cu...` vs `..._d59ce2d9_...`. So the f2ecfb0 and 9607b09 rounds do not change the CMake CUDA build.
- Makefile CUDA path, dry run (`make -n xgboost_prev install_xgboost_prev xgboost install_xgboost lightgbm_gpu install_lightgbm_gpu`, no USE_HIP): ccde3a9 and 9607b09 print identical recipes, so 9607b09's `xgboost_prev`/`install_xgboost_prev` are upstream's again (the 24e6169 revert), building `xgboost_prev` via `Makefile2 ... libxgboost` and installing the wheel from `python-package/dist`. Against the control, the only differences are the config.mk line numbers in the `$(warning)` lines (USE_HIP block added above them) and the install glob `lightgbm*-py3-none-any.whl` -> `lightgbm*-py3-none-*.whl` (39984a9, recorded in Porting). The new glob is a superset of the old, so a CUDA `py3-none-any` wheel still matches; not a regression. `xgboost_prev` itself was not CUDA-built (its `Utils.cmake` sed adds `35;37`, which nvcc 12.8 rejects, the same upstream).
- Build trees, worktrees and the clone deleted after the check.
## Review 2026-11-05 round 3 (linux-gfx942, MI300, ROCm 10.1 pip SDK) -- moat-port 9607b09

### Port Review: h2o4gpu (moat-port `9607b09` vs master `97279de`; delta `ccde3a9..9607b09` reviewed first)

#### Summary
A HIP port of h2o4gpu's GPU solvers via `hip_compat/` CUDA-named headers and a `USE_HIP` CMake branch, plus ROCm builds of LightGBM (upstream v4.7.0) and XGBoost (AMD-Ecosystem release_2.1.3-rocm, pin `ee5234f` = branch tip) with h2o's patches. The ROCm build leaves out xgboost_prev under jeff's 2026-11-05 reversal (which supersedes the 2026-11-04 "CPU-only" ruling), and the GPU module gets a ROCm RUNPATH. Verdict: **Approve**. No findings. Every item from rounds 1 and 2 is closed or moot at this head.

#### Not a finding, recorded for the pipeline
- CUDA path at 9607b09: the nvcc 12.8 compile check that landed concurrently (`## CUDA compile check` entry from fleet job h2o-cuda3, commit f2ddb8da) builds every CUDA target at 9607b09, with the same objects and symbol sets as ccde3a9 and identical warning sets. That covers the standing CUDA-compile rule for this head; my dry-run comparison below agrees. (When this review was drafted, that entry had not yet reached this branch.)
- `utils/check.py` `commits` keeps flagging 4e5e433 and its revert 24e6169 (no `[ROCm]`), which is deliberate (recorded above). The pair cancels out; drop both at the PR-prep squash.

#### Judgement: LightGBM GPU vs CPU per-row differences (porter note: up to 4e-2)
Benign, not a port defect. Reproduced and explained here (`agent_space/oracle/lgbprobe{2,3,4}.py`, lightgbm_gpu 4.7.0 from the wheel, `deterministic=True`, 50-100 rounds):
- On linear, Friedman1, `make_regression` (100k and 200k rows) and integer-feature data, `device_type='cuda'` vs cpu max per-row diff is 1.9e-8 to 7.5e-7, with no split differing in any tree. CUDA run-to-run differs by up to 3e-7 (atomic histogram order). CPU row-wise vs column-wise and 1 thread vs all threads: bit-identical.
- Reproduced on a nonlinear target with 0.1 noise (`sin(2x0) + x1*x2 + sign(x3)*x4`, 100k x 20): held-out max 0.83, 0.13% of rows > 1e-3, RMSE 0.89386533 (cpu) vs 0.89387357 (cuda). First differing node (tree 1, path LRLRLRL): same feature f1, same gain 583.828, thresholds 0.0686407 (cuda) vs 0.0791445 (cpu). On the **training** rows the two models agree to 2.4e-8 (0 rows > 1e-3). So both learners partition the training data identically. They differ only in where they put the threshold inside a gap that holds no training row of that node, and a held-out row in that gap (87 of 20000 by f1 alone) goes the other way. The same pattern shows on a 200k x 30 mixed target (tree 88: same feature, same gain, different threshold; max 8e-2).
- This is threshold placement in LightGBM's own CUDA tree learner, which the port builds unmodified (the h2o patch touches only MOJO accessors, the logger switch and the FindGroups default). It is not a ROCm numeric fault. An accuracy check that compares GPU and CPU LightGBM should compare RMSE or training-row predictions, not held-out rows at ulp tolerance. Not verified on NVIDIA (no CUDA GPU here); the code path is the same source.

### Verification evidence (this host, gfx942, HIP_VISIBLE_DEVICES=1, `ulimit -c 0`, LD_LIBRARY_PATH unset, OPENBLAS_NUM_THREADS=32)
- Fresh clone of the fork at `9607b09`. CUDA submodule pins equal master (`LightGBM` 42043b9, `xgboost` 4235811, `xgboost_prev` 0ad357a, `tests/googletest` 2fe3bd9), with no `-` lines in `.gitmodules`.
- `cmake -S src -B src/build-hip -DUSE_CUDA=OFF -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_PREFIX_PATH=$R -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$FLEET_VENV/bin/python && cmake --build src/build-hip -j56`: clean (pip `swig==4.2.1` in the venv). `_ch2o4gpu_gpu.so` RUNPATH `[<job>/venv-base-links/_rocm_sdk_devel/lib:/usr/lib/gcc/x86_64-linux-gnu/13]`; NEEDED has libgomp and no libomp/libomptarget.
- `h2o4gpu_test`: 24/24 pass, 3 consecutive runs.
- ARIMA ROCm tolerance, re-derived (`agent_space/arima/drv.cu` linking `libgpuh2o4gpu.a`, float and double fits). d1p1q1 phi f32 -1.03694952 / f64 -1.0369515 / golden -1.0369391; theta 1.11547148 / 1.11547344 / 1.1154615. p2q2 phi -2.95895433, 2.88284779 / f64 -2.95895464, 2.88284823; theta 3.95986366, -0.616015851 / f64 3.95986376, -0.616016071. Bit-identical to rounds 1 and 2 and to gfx90a. The ROCm float result is closer to float64 than the cuSOLVER golden value (d1p1q1: 2e-6 vs 1.2e-5), and the worst deviation from golden (1.04e-5 on phi) is half the 2e-5 relative bound (2.07e-5). Justified; CUDA keeps `ASSERT_FLOAT_EQ`.
- Wheel, all rc 0: `make cpp`, `make lightgbm_gpu install_lightgbm_gpu lightgbm_cpu install_lightgbm_cpu`, `make xgboost install_xgboost`, `make xgboost_prev install_xgboost_prev`, `make apply-sklearn`, `make py`, all with `USE_HIP=ON PYTHON=$FLEET_VENV/bin/python`, `OMP_NUM_THREADS=56` (caps the recipes' `nproc`). `verify_gpu_lib.sh`: gfx942 code objects intact after strip. Both patch sets apply cleanly (git's trailing-whitespace warnings only: LightGBM 0001:47, xgboost 0002:384,738). `xgboost_prev` targets print the not-built line twice and build nothing. The wheel ships `_ch2o4gpu_gpu.so`, `lightgbm_gpu/lib/lib_lightgbm.so`, `lightgbm_cpu/lib/lib_lightgbm.so` and `xgboost/lib/libxgboost.so`, with `xgboost_prev/` entries 0. After install, `find_spec('xgboost_prev')` is None. Submodule trees reverted afterwards; tracked fork tree clean.
- Python GPU oracle from the installed wheel (`agent_space/oracle/oracle.py`, exits nonzero if GPUlib is None or there is no GPU): `device_count(-1)` (1, 1). KMeansH2O 1-v_measure 0 (f32, f64). TSVD cusolver sv rel err 8.2e-7 / 1.0e-15. PCA ev rel err 6.6e-7 / 2.1e-15. ElasticNetH2O (alpha 0, lambda 1e-6) r2 0.999993, gap to sklearn Ridge 2.8e-8 / 2.5e-13. ALL PASS. Under `rocprofv3 --kernel-trace`: 57341 dispatches on one GPU agent, including kmeans:: 120, cml:: 8570, h2o4gpu:: 226, rocBLAS 21359, rocSOLVER 636, rocPRIM 17368.
- `pytest tests/python/open_data/system tests/python/open_data/gbm`: 18 pass / 4 fail / 17 skip, the same as every previous round (the 4 failures are `lgb.cv(early_stopping_rounds=)`, removed in LightGBM 4.0).
- XGBoost h2o patches on the GPU (`agent_space/oracle/gbm.py`, under rocprofv3: 7262 `xgboost::` kernels): xgboost 2.1.3, `USE_HIP` True. Integer feature f0 thresholds are the between-bin midpoints `0.5 ... 8.5` on both cpu and cuda. Shared-QuantileDMatrix held-out RMSE 0.2909730735 (cpu) / 0.2909730746 (cuda), max pred diff 9.5e-7. `get_feature_interactions` returns a DataFrame (20 rows). `pred_contribs` sums to the margin. Booster and sklearn `pred_leaf` give (5, 10).
- roctx (ruling 5): a `USENVTX=1` configure with `-DCMAKE_CXX_FLAGS=-DUSE_NVTX -DCMAKE_HIP_FLAGS=-DUSE_NVTX` prints "Building with NVTX support on (roctx)" and the module NEEDs `libroctx64.so.4`; the default build has no roctx. An ElasticNetH2O fit using that module (copy of the installed package) under `rocprofv3 --marker-trace` gives 7424 marker records = 7424 `START:` lines (Evaluate_fg, gapoptvaltol, orelax, ... 525 each), corr 0.9999963.
- CUDA path dry run (no nvcc here): `make -n` for `cpp testunit xgboost xgboost_prev install_xgboost install_xgboost_prev lightgbm_gpu lightgbm_cpu install_lightgbm_gpu install_lightgbm_cpu` (default USE_HIP) against a master worktree. The only differences are an extra space from the empty `$(CMAKE_GPU_FLAGS)` and the LightGBM install glob `py3-none-*`, which still matches upstream's `-any` wheel. `src/interface_py` `make -n all`: the xgboost_prev symlink runs behind `[ "OFF" != "ON" ]`. So the CUDA recipes are upstream's.
- Records: surface.json `platform:windows` sits under `scoped_out` with its "PLACEHOLDER, NOT A SCOPE CUT" reason. Ruling 3 keeps Windows in scope, the windows gate is required and unwaived, windows-gfx1151 is unblocked, and deferred `surface-owed-state-for-in-scope-platforms` is registered and open. The records do not misstate the scope. `surface.py check`: clean. `jargon.py --port h2o4gpu`: clean.

## Ruling 2026-11-05 (jeff): PR-prep squash authorized
At PR prep, once every gate is satisfied at the validated head, squash moat-port into compliant commit(s) whose tree is identical to the validated head (fixing the commit-gate failures and the overstated messages recorded in notes), record it with `moatlib.py squash-carry-forward h2o4gpu <new_sha>`, and replace moat-port with the squash. That one force-push of h2o4gpu moat-port is authorized by this ruling (no upstream PR is open). Same terms as plvs.

## Validation 2026-11-06 (linux-gfx1100, Radeon Pro W7800 48GB, wave32, ROCm 10.1 pip SDK)

Validator at `9607b099` (moat-port tip = head_sha). Fresh clone of the fork, `git submodule update --init --recursive tests/googletest LightGBM_rocm xgboost_rocm`. HIP_VISIBLE_DEVICES=3 (dispatcher pin), `ulimit -c 0`, LD_LIBRARY_PATH unset and OPENBLAS_NUM_THREADS=32 for every build and run. Extra pip into the job venv: `swig==4.2.1 cython==3.0.11 threadpoolctl psutil tabulate future joblib pytest-timeout meson-python ninja feather-format "pyarrow<15" build wheel` (scikit-learn 1.9.1, numpy 1.26.4, setuptools 79.0.1 already present). apt `libopenblas-dev` already present.

### Build (rc 0 for every step, 605 s wall at MAX_JOBS=32)
```
R=$($FLEET_VENV/bin/python -m rocm_sdk path --root); P=$FLEET_VENV/bin/python
export PATH=$FLEET_VENV/bin:$R/bin:$PATH CMAKE_PREFIX_PATH=$R OMP_NUM_THREADS=$MAX_JOBS CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
cmake -S . -B build-hip -DUSE_CUDA=OFF -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH=$R -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$P
## Validation 2026-11-06 (linux-gfx90a, AMD Instinct MI250X / MI250, ROCm 10.1 pip SDK) -- moat-port 9607b09

Validator role, fresh clone of `AMD-Ecosystem/h2o4gpu` checked out at `9607b099f3e0128c99eb8964c32462892b086967` (= `origin/moat-port`). `HIP_VISIBLE_DEVICES=0`, `ulimit -c 0`, `LD_LIBRARY_PATH` unset and `OPENBLAS_NUM_THREADS=32` for every build and run. Submodule pins: `LightGBM_rocm` 8f7036f, `xgboost_rocm` ee5234f, `tests/googletest` 2fe3bd9. Extra venv packages (into `$FLEET_VENV` only): `cython==3.0.11 meson-python feather-format "pyarrow<15" pytest-timeout`; apt `swig` 4.2.0 and `libopenblas-dev` 0.3.26 already present. No dependency had to be built from source beyond the in-tree submodules.

### Build (all rc 0; wheel `h2o4gpu-0.6.0+local.9607b099.dirty`, "dirty" = the h2o patches applied inside the two ROCm submodules during the build)
```
R=$($FLEET_VENV/bin/python -m rocm_sdk path --root); P=$FLEET_VENV/bin/python
export PATH=$FLEET_VENV/bin:$R/bin:$PATH CMAKE_PREFIX_PATH=$R OMP_NUM_THREADS=$MAX_JOBS CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
cd projects/h2o4gpu/src
git submodule update --init --recursive tests/googletest LightGBM_rocm xgboost_rocm
cmake -S . -B build-hip -DUSE_CUDA=OFF -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=$R -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$P
cmake --build build-hip -j$MAX_JOBS
make cpp USE_HIP=ON PYTHON=$P
make lightgbm_gpu install_lightgbm_gpu lightgbm_cpu install_lightgbm_cpu USE_HIP=ON PYTHON=$P
make xgboost install_xgboost USE_HIP=ON PYTHON=$P
make xgboost_prev install_xgboost_prev USE_HIP=ON PYTHON=$P
make xgboost_prev install_xgboost_prev USE_HIP=ON PYTHON=$P    # prints "XGboost previous version is not built with USE_HIP=ON" twice
make apply-sklearn PYTHON=$P
make py USE_HIP=ON PYTHON=$P
$P -m pip install --no-deps --force-reinstall src/interface_py/dist/h2o4gpu-0.6.0+local.9607b099.dirty-cp311-cp311-linux_x86_64.whl
```
- `verify_gpu_lib.sh`: "AMD GPU code objects intact after strip: amdgcn-amd-amdhsa--gfx1100". xgboost_prev targets print "XGboost previous version is not built with USE_HIP=ON" twice and build nothing.
- Wheel ships `h2o4gpu/libs/_ch2o4gpu_gpu.so`, `lightgbm_gpu/lib/lib_lightgbm.so`, `lightgbm_cpu/lib/lib_lightgbm.so`, `xgboost/lib/libxgboost.so`; `unzip -l | grep -c xgboost_prev/` = 0, and no `xgboost_prev*` in site-packages after install. ("dirty" = the h2o patches the Makefile applies inside LightGBM_rocm/xgboost_rocm; reverted with `git -C <sub> checkout -- .` afterwards.)
- Installed `_ch2o4gpu_gpu.so`: RUNPATH `[<job>/venv-base-links/_rocm_sdk_devel/lib:/usr/lib/gcc/x86_64-linux-gnu/13]`, NEEDED libgomp and no libomp/libomptarget. It loads with LD_LIBRARY_PATH unset.

### Results
- `build-hip/h2o4gpu_test`: 24/24 PASS on 3 consecutive runs (rc 0 each).
- Python GPU path vs scikit-learn 1.9.1 (`agent_space/oracle.py`, recreated on this host; exits nonzero if GPUlib is None, no GPU, or any check fails): `device_count(-1)` (1, 1). ALL PASS:
  - KMeansH2O (20000x10, 5 blobs): v_measure 1.000000 both; max center dist to sklearn 3.8e-6 (f32) / 8.2e-15 (f64).
  - TruncatedSVDH2O cusolver (5000x50, 5 comps): sv rel err vs numpy SVD 4.4e-7 / 1.8e-15; components vs sklearn arpack (sign-aligned) 8.8e-6 / 2.5e-8.
  - PCAH2O: explained_variance rel err 5.0e-7 / 3.0e-15; components 3.5e-6 / 3.0e-8.
  - ElasticNetH2O (alpha 0, lambda 1e-6, 20000x20): r2 0.999994 / 0.999996, gap to sklearn Ridge 1.9e-8 / 1.2e-13.
  - Under `rocprofv3 --kernel-trace --output-format csv`: 104164 dispatches, all on one GPU agent, including kmeans:: 110 (all_dots, calculate_centroids, make_new_labels, ... float and double), h2o4gpu:: 226, cml:: 15482 (mostly rocPRIM kernels over cml functors), rocBLAS 39677, rocSOLVER 552, other rocPRIM/rocThrust 15482.
- GBM GPU vs CPU (`agent_space/gbm.py`, 100000x20, integer f0, 80/20 split, 50 rounds), under rocprofv3: 66184 dispatches on the GPU agent, `LightGBM::` 49792, `xgboost::` 5618. ALL PASS:
  - xgboost 2.1.3, `build_info()['USE_HIP']` True. Shared QuantileDMatrix held-out RMSE cpu 0.6310918444 / cuda 0.6310918563, max pred diff 1.9e-6. h2o between-bin cuts on f0 `0.5 ... 8.5` (cuda model). `get_feature_interactions` DataFrame (311, 8). `pred_contribs` (5, 21) sums to the margin. `pred_leaf` (5, 50).
  - lightgbm_gpu 4.7.0 `device_type='cuda'` vs cpu (deterministic): held-out RMSE 0.60913774 / 0.60917006; training-row max pred diff 9.9e-8 (the reviewer's round-3 guidance: compare RMSE or training rows, not held-out rows).
- `pytest -p no:cacheprovider tests/python/open_data/system tests/python/open_data/gbm -rfs -q`: 18 passed / 4 failed / 17 skipped, the same as every earlier round. The 4 failures are all `TypeError: cv() got an unexpected keyword argument 'early_stopping_rounds'` (test_lightgbm gpu/cpu x dart/gbdt; removed in LightGBM 4.0). Skips: data dir not synced (15), huge memory footprint (2).
- roctx (ruling 5): `USENVTX=1 cmake -S . -B build-nvtx ... -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_CXX_FLAGS=-DUSE_NVTX -DCMAKE_HIP_FLAGS=-DUSE_NVTX` prints "Building with NVTX support on (roctx)" and `_ch2o4gpu_gpu.so` NEEDs `libroctx64.so.4` (default build: no roctx). A copy of the installed package with that module (`agent_space/glm_nvtx.py`, ElasticNetH2O f64 fit, corr 0.9999213) under `rocprofv3 --marker-trace --kernel-trace`: 2580 `MARKER_CORE_RANGE_API` records = 2580 `START:` lines printed by PUSH_RANGE (Evaluate_fg, gapoptvaltol, orelax, P1alloc, ... 179 each), 10178 kernels.
- CUDA compile gate: not rerun; `## CUDA compile check 2026-11-05` above already covers `9607b09` (nvcc 12.8, all targets PASS).
- `python3 utils/jargon.py --port h2o4gpu`: clean. ROCm build documented in DEVEL.md beside the CUDA build.
- Integrity: `git status --porcelain --ignore-submodules=untracked` at `9607b099` shows only untracked build output (build-hip/, build-nvtx/, sklearn_download.lock, src/interface_py/lightgbm_{cpu,gpu}).
- Verdict: linux-gfx1100 (wave32) completed at `9607b099`.
- Wall clock about 15 min (most of it the recursive submodule clone). `verify_gpu_lib.sh`: "PASS: _ch2o4gpu_gpu.so survived strip". Both patch sets applied (only git's trailing-whitespace warnings, LightGBM 0001:47, xgboost 0002:384,738).
- Wheel ships `h2o4gpu/libs/_ch2o4gpu_gpu.so`, `lightgbm_gpu/lib/lib_lightgbm.so`, `lightgbm_cpu/lib/lib_lightgbm.so`, `xgboost/lib/libxgboost.so`; `xgboost_prev/` entries 0. Installed `_ch2o4gpu_gpu.so`: RUNPATH `[<job>/venv-base-links/_rocm_sdk_devel/lib:/usr/lib/gcc/x86_64-linux-gnu/13]`, NEEDED hipblas/hipsparse/hipsolver/hiprand/rocrand/amdhip64/openblas/libgomp (no libomp, no libomptarget, no roctx).

### Tests (exact results)
- C++ googletest `build-hip/h2o4gpu_test`: **24/24 pass, 3 consecutive runs** (rc 0 each).
- Python GPU path vs scikit-learn 1.9.1 (`agent_space/oracle/oracle.py` from the previous gfx90a rounds, unchanged; exits nonzero if `GPUlib.get()` is None or no GPU): `device_count(-1)` (1, 1). KMeansH2O v_measure 1.000000, center dist 3.7e-6 (f32) / 9.2e-15 (f64); TSVD (cusolver) sv rel err 1.8e-7 / 3.0e-16; PCA ev rel err 4.6e-7 / 3.8e-16; ElasticNetH2O (alpha 0, lambda 1e-6) r2 1.000000 = sklearn Ridge in both dtypes. **ALL PASS**.
- Same oracle under `rocprofv3 --kernel-trace --output-format csv`: ALL PASS again; 48519 dispatches, all on one GPU agent: `kmeans::` 110, `cml::` 7242, `h2o4gpu::` 226, rocBLAS 18071, rocSOLVER 636, rocPRIM 14722.
- xgboost and LightGBM, GPU vs CPU (`agent_space/oracle/gbm.py`, 100000x20, 80/20 split, 50 rounds): xgboost 2.1.3 (`USE_HIP` True), shared QuantileDMatrix held-out RMSE cuda 0.1309159382 / cpu 0.1309159387, `get_feature_interactions` present. lightgbm_gpu 4.7.0 `device_type='cuda'` RMSE 0.12325394 / cpu 0.12329379 (per-row max 4.05e-2, median 7.3e-10, 0.04% of rows > 1e-3: the near-tie threshold placement judged benign in Review round 3). ALL PASS. Under rocprofv3: 36807 dispatches on the GPU agent, `LightGBM::` 24286, `xgboost::` 5332.
- Shipped `pytest -q -p no:cacheprovider tests/python/open_data/system tests/python/open_data/gbm`: **18 passed / 4 failed / 17 skipped** in 26 s. The 4 failures are the known `test_lightgbm.py::test_lightgbm_{gpu,cpu}[{dart,gbdt}]`, all `TypeError: cv() got an unexpected keyword argument 'early_stopping_rounds'` (removed in LightGBM 4.0; upstream test drift, same as every earlier round). Skips = data not synced / memory.
- roctx with the NVTX switch on: `USENVTX=1 cmake -S . -B build-nvtx -DUSE_CUDA=OFF -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=$R -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$P -DCMAKE_CXX_FLAGS=-DUSE_NVTX -DCMAKE_HIP_FLAGS=-DUSE_NVTX && cmake --build build-nvtx -j64` prints "Building with NVTX support on (roctx)"; its `_ch2o4gpu_gpu.so` NEEDs `libroctx64.so.4` (the default build: none). A copy of the installed package with that module, ElasticNetH2O fit (`agent_space/glm_solve.py`) under `rocprofv3 --marker-trace --kernel-trace`: **1087 marker records = 1087 `START:` lines** (Evaluate_fg, gapoptvaltol, orelax, P1alloc, ... 72 each), 3848 kernels, fit corr 0.99999.

### Gates for completion
- CUDA no-regression: already recorded at this head (`## CUDA compile check 2026-11-05 (ccde3a9)`, second entry, which also built 9607b09 with nvcc 12.8: PASS, no new warnings). Not re-run.
- `python3 utils/jargon.py --port h2o4gpu`: clean. Documentation: DEVEL.md `##### USE_HIP (AMD GPUs)` documents the ROCm build next to the CUDA one, including the LightGBM/xgboost ROCm submodules and the xgboost_prev limitation.
- Integrity: submodule trees reset (`git -C LightGBM_rocm checkout -- .`, same for `xgboost_rocm`); `git status --porcelain --ignore-submodules=untracked` shows only untracked build output (`build-hip/`, `build-nvtx/`, `sklearn_download.lock`, `src/interface_py/lightgbm_{cpu,gpu}`). `moatlib.py audit-clean h2o4gpu`: OK.
- Verdict: linux-gfx90a **completed** at 9607b09.

## Porter 2026-11-11 (windows-gfx1151, Radeon 8060S, TheRock ROCm 7.14.0a20260612 pip SDK) -- ruling 3 Windows attempt: STOPPED, nothing pushed

Outcome: the native build works and the C++ suite passes, but the Python ElasticNet (GLM) path is broken on Windows (f32 leaves the heap corrupted, f64 never returns). I could not localize it within the budget (no debugger on this host), so nothing was landed: `moat-port` stays at `9607b099`, and gfx90a/gfx1100 evidence is untouched. Question for jeff at the end.

### What the Windows port takes (63 added / 3 removed lines, 10 files; Linux and CUDA paths unchanged by construction)
Kept as local commit `a949256` on branch `win-wip` in this host's fork clone (`projects/h2o4gpu/src`), and as `agent_space/h2o_win/windows-port-wip.patch`. Changes:
- `CMakeLists.txt`: one top-level `if(WIN32)` block -- `FIND_PATH(CBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas)` (vcpkg layout); `-D_HAS_AUTO_PTR_ETC=1` (MSVC STL hides `std::unary_function` in C++17, used by `src/cpu/include/equil_helper.h`); `-DNOMINMAX -DWIN32_LEAN_AND_MEAN -Dpopen=_popen -Dpclose=_pclose`; with OpenMP, `-D_OPENMP_LLVM_RUNTIME` (MSVC's `omp.h` is OpenMP 2.0 and lacks `omp_set_max_active_levels`; this selects its `omp_llvm.h`, matching the `libomp140` runtime clang links) plus `LINK_LIBRARIES(${OpenMP_CXX_LIBRARIES})` (the `-Xclang -fopenmp` link flag never reaches lld-link, so `_ch2o4gpu_cpu` had undefined `__kmpc_*`); `hip_compat/win32` on the include path (an empty `unistd.h`); `LINK_DIRECTORIES(${Python_LIBRARY_DIRS})` (Python.h auto-links `python3XY.lib`); `SET(CMAKE_POLICY_VERSION_MINIMUM 3.5)` (CMake 4 refuses the bundled googletest). In the HIP branch, the forced include is `/FI` when `CMAKE_HIP_COMPILER_FRONTEND_VARIANT` is MSVC (clang-cl ignores `-include` and compiles the header as a second source: "cannot specify /Fo when compiling multiple source files").
- `hip_compat/cuda_runtime.h`: `#include <windows.h>` under `_WIN32` (upstream's `Timer` in `src/gpu/include/utils.cuh` already has a `_WIN32` branch using `LARGE_INTEGER`/`QueryPerformanceCounter` but never included the header). `hip_compat/nvToolsExt.h`: no-op roctx stubs when `<roctracer/roctx.h>` is absent (TheRock Windows ships no roctracer).
- Shared sources, all portable: `src/include/timer.h` `_WIN32` branch using `std::chrono` (the POSIX `gettimeofday` branch unchanged); missing standard includes `<functional>` (`equil_helper.h`) and `<numeric>` (`std::iota` in `kmeans_h2o4gpu.cu`, `h2o4gpukmeans.cpp`); `src/swig/solver/elastic_net.i` `PyLong_AsLong` -> `PyLong_AsLongLong` for the device-pointer arguments (32-bit `long` on Windows overflowed: "OverflowError: argument 37 of type 'int'"; identical on LP64).
- `DEVEL.md`: Windows paragraph + cmake command under `##### USE_HIP (AMD GPUs)`.
- Scope limits: the Makefile, `make py`/wheel, the sklearn overlay, LightGBM/xgboost builds are not ported (bash/Linux). Only the CMake build (GPU lib, both SWIG modules, `h2o4gpu_test.exe`) is.

### Build (rc 0, fresh tree)
Toolchain: CMake 4.4.4 + Ninja from `agent_space/h2o_venv` (Python 3.12.10 venv: numpy 1.26.4, swig 4.2.1, scikit-learn 1.9.1, scipy, pandas, tabulate, future, psutil), all three compilers = TheRock `lib/llvm/bin/clang-cl.exe` (clang 23), OpenBLAS from vcpkg (`D:/Develop/vcpkg`), MSVC 14.44 libs.
```
source agent_space/win_rocm_env.sh   # ROCM_ROOT=TheRock _rocm_sdk_devel, MSVC link first on PATH
cmake -G Ninja -S . -B build-win -DUSE_CUDA=OFF -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_HIP_COMPILER=$CXX -DCMAKE_PREFIX_PATH=$ROCM_ROOT \
  -DCMAKE_TOOLCHAIN_FILE=D:/Develop/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE=agent_space/h2o_venv/Scripts/python.exe
cmake --build build-win -j 12
```
Gotchas: swig 4.5 drops `PyInt_Check`/`PyString_Check` that the wrappers use -> pin swig 4.2.1 (Linux used 4.2 too); numpy 2 breaks the kmeans SWIG call ("Cannot cast scalar from dtype('float64') to dtype('float32')") -> numpy 1.26 on Python 3.12 (as Linux). Runtime: `rocm_stage_runtime build-win` (amdhip64_7/comgr/kpack/hiprtc next to the exe) and the ROCm `bin` on PATH in /d/... form (a `D:/` entry breaks MSYS PATH and gives exit 127 "hipblas.dll"). The Python test package (`agent_space/h2o_win/mkpkg.sh`) is a copy of `src/interface_py/h2o4gpu` with the built modules in `libs/` and a stub `__init__` that calls `os.add_dll_directory` for `libs` and the ROCm bin and aliases `h2o4gpu.{linear_model,cluster,decomposition,utils}` to sklearn's; it loaded the SDK's `amdhip64_7.dll` (checked with GetModuleFileNameW), not the broken System32 one.

### Results vs Linux
- `h2o4gpu_test.exe`: **24/24 PASS, 3 consecutive runs** (same as gfx90a/gfx1100).
- Python oracle (`agent_space/h2o_win/oracle.py`, sklearn 1.9.1 references, same datasets as the Linux oracle): `device_count(-1)` (1, 1). float32 ALL PASS: KMeansH2O v_measure 1.0, center dist 3.4e-6; TSVD (cusolver) sv rel err 2.7e-7, comp 2.5e-6; PCA ev 5.8e-7, comp 1.2e-5; ElasticNetH2O (alphas=[0], lambdas=[1e-6], n_folds=1) r2 0.999989 vs Ridge 1.000000. float64 (fresh process) KMeans center dist 8.3e-15, TSVD sv 5.0e-16 / comp 2.9e-8, PCA ev 4.7e-15 / comp 3.0e-8: PASS. `_ch2o4gpu_cpu` loads.
- **FAIL 1, f64 ElasticNetH2O never returns**: 2000x20 or 200x20, with alphas/lambdas lists or with alpha_min=alpha_max=0, any max_iter (even 5). Prints the POGS header, then no iteration row ever (f32 prints rows and stops early at iteration 122 in 1.5 s). The process is not deadlocked: `AMD_LOG_LEVEL=4` shows it launching rocBLAS axpy/gemv/trsv/dot, rocPRIM reductions over `cml::Square<double>` and the prox transforms, plus 8-byte D2H copies, continuously (~100 launches per cycle) at 100% CPU. Linux f64 GLM passed on gfx90a and gfx1100 with the same code.
- **FAIL 2, f32 ElasticNetH2O corrupts the heap**: it returns the right answer, but the next GPU solver call in the same process dies with `0xc0000374` (STATUS_HEAP_CORRUPTION; seen twice, at the f64 KMeans fit after the f32 GLM). KMeans/TSVD/PCA alone in a process are clean.
- Ruled out: the `long` -> `long long` SWIG pointer bug (fixed above); `long`/`%ld` in the GLM sources (none outside tests). Not yet ruled out: OpenMP region in `elastic_net_ptr.cpp` (LLVM libomp140 via `_OPENMP_LLVM_RUNTIME`; the HIP TUs are compiled without OpenMP), the `free_sols`/`modelfree2` buffers, and the lambda setup (`lambdaslocal[0]` is never copied from `lambdas[0]` when a list is passed -- a pre-existing upstream bug, platform-independent). Next step needs a native debugger or Application Verifier (pageheap) on this host, neither installed.

### Question for jeff
Ruling 3 says stop if Windows is too cumbersome. The build port is small (above), but the GLM path has an unexplained Windows memory fault. Options: (a) authorize a further debugging round on this laptop (install the Windows debugging tools / gflags pageheap; GLM-only, small data), then land `a949256` (+fix) as the one Windows commit; (b) land the build commit now with the GLM failure recorded, letting the windows validator hold the arch (moving head re-owes gfx90a/gfx1100 revalidation for Linux-neutral changes); (c) stop Windows here and suggest a `windows` gate waiver. Not done either way: the CUDA nvcc compile check on the mi355 host would be owed for whatever new head lands.

## Porter 2026-11-11 round 2 (windows-gfx1151, Radeon 8060S, TheRock ROCm 7.14.0a20260612 pip SDK) -- jeff's ruling "(a)": GLM debugging round; FIXED, landed as `2c22a86`

Outcome: two separate defects, both found and fixed; `a949256` + fixes landed as ONE commit `2c22a86` on `moat-port` (9607b099f..2c22a86). Review covers that range. gfx90a and gfx1100 now owe revalidation (shared kmeans source changed). **CUDA nvcc compile check on the mi355 host is owed for `2c22a86`** (the kmeans change touches the CUDA path: `thrust::copy` from `device_vector` into a raw host pointer). Fork branch `wip-windows` (a949256) is now stale; nothing needs it.

Tools (no admin): WinDbg was already installed per-user (`%LOCALAPPDATA%\Microsoft\WindowsApps\cdbX64.exe`). The venv `python.exe` is a launcher that spawns the real interpreter: run cdb with `-o` (debug children) or attach to the big child process (`tasklist`, ~370 MB). A Release build with symbols that keeps Release codegen: `-DCMAKE_{CXX,HIP}_FLAGS_RELEASE="-O2 -Ob2 -DNDEBUG -Z7"` and `-DCMAKE_{MODULE,EXE}_LINKER_FLAGS_RELEASE="-INCREMENTAL:NO -DEBUG"` (dash spelling: MSYS rewrites `/O2` into a path). Heap checks from Python without pageheap: `ctypes.windll.ntdll.RtlValidateHeap(GetProcessHeap(), 0, None)`. Exact NTSTATUS exit codes need a Python `subprocess` wrapper (`agent_space/h2o_win/rc2.py`); MSYS bash folds `0xc0000374` into exit 127. Pageheap/gflags/AppVerifier (admin) were not needed.

### Defect 1: heap corruption came from KMeans, not ElasticNet (misattributed last round)
- Evidence: `seq.py km32` alone exits `0xc0000374` (at interpreter shutdown); `ts32`, `pca32`, `en32`, `en64` alone exit 0. Last round's oracle ran KMeans first, so the corruption surfaced later during the f64 KMeans call. cdb stack at the break: `ntdll!RtlFreeHeap <- ucrtbase!free_base <- _multiarray_umath` (NumPy freeing an array's data).
- Root cause: `kmeans.i` applies the h2o4gpu-specific `(T** INPLACE_ARRAY1)` typemap (`src/swig/numpy.i`), which passes `&PyArrayObject->data`; `kmeans_fit/predict/transform` (GPU) and the CPU back end then set `*pred_labels = (new thrust::host_vector<int>(rows))->data()`, so NumPy later `free()`s a `new`-allocated buffer. MSVC's `std::allocator` over-aligns allocations >= 4096 bytes (`_Allocate_manually_vector_aligned`), so the pointer is not the start of a heap block and `free()` corrupts the heap. 20000 labels = 80 KB triggers it; 5x10 centroids (200 B) do not.
- Fix: `alloc_result<T>()` helpers returning `calloc`/`malloc` buffers filled by `thrust::copy` (GPU) / `std::copy` (CPU), in `src/gpu/kmeans/kmeans_h2o4gpu.cu` and `src/cpu/h2o4gpukmeans.cpp`. Same pointer-handoff contract, so Python/R/C callers are unchanged; the transform buffer is now filled at deterministic per-GPU offsets instead of a racy `insert` under `omp parallel for`.
- Linux: latent only. libstdc++ `std::allocator` returns the `malloc` pointer, so glibc tolerates the `new`/`free()` mismatch (formally UB). No Linux crash expected, and Linux results should be unchanged; the gfx90a/gfx1100 revalidation confirms that.

### Defect 2: f64 ElasticNet "hang" is a GPU kernel that never completes (Windows codegen, inlining-sensitive)
- Evidence: symbol-carrying Release build: main thread always at `Solve` (`h2o4gpuglm.cu:583`) -> `FuncEval` (`prox_lib.h:712`, `thrust::inner_product` over `FuncEvalF<double>`) -> rocPRIM reduce -> `hipMemcpyWithStream` waiting. `AMD_LOG_LEVEL=3`: last launch is `trampoline_kernel<..., reduce_config_selector<double>, reduce_impl<true, transform_iterator<zip_function<FuncEvalF<double>>...` (grid 1, block 256, 64 MB scratch), followed by endless `PAL fence isn't ready! result:3`. So the GPU never finishes that kernel. Last round's "continuous launches" reading came from earlier iterations.
- Inlining-sensitive: hangs with HIP `/O2 /Ob2` (the Release default) with or without `/Z7`. Passes with `/O2 /Ob1` (RelWithDebInfo, and Release with only `-Ob1`). Device asm (`--cuda-device-only -S`): with `/Ob2`, the double kernel inlines everything into one 248-VGPR body with 128 B private segment; with `/Ob1`, `zip_detail::apply_impl<FuncEvalF<double>>` stays a call (81 VGPRs). No source-level loop or UB found in `FuncEval` (straight-line switch). Linux (ROCm 10.1 hipcc `-O3`, gfx90a/gfx1100) never hung, so I suspect a TheRock clang-23 gfx1151 (or Windows-target) device miscompile. Not proven: no minimal reproducer yet. The upstream tool bug is not drafted; next step for that would be extracting the kernel into a standalone .hip and diffing the ISA.
- Workaround (in the build files, guarded): in the HIP branch under `CMAKE_HIP_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"`, `SET_SOURCE_FILES_PROPERTIES(src/gpu/h2o4gpuglm.cu PROPERTIES COMPILE_OPTIONS "/Ob1")`. It comes after the config `/Ob2` on the command line, so it wins. Linux/hipcc and CUDA builds are untouched.

### Results at 2c22a86 (Release, `agent_space/h2o_win/build.sh`, 70 s build)
- `h2o4gpu_test.exe`: 24/24 PASS, 3 consecutive runs (same as gfx90a/gfx1100).
- `oracle.py f32,f64` in ONE process (`agent_space/h2o_win/oracle_fix.log`): exit 0, ALL PASS. f32: KMeans v_measure 1.0, center dist 4.8e-6; TSVD sv 2.7e-7, comp 2.5e-6; PCA ev 5.8e-7, comp 1.2e-5; EN r2 0.999989 vs Ridge 1.000000. f64: KMeans center dist 7.5e-15; TSVD sv 5.0e-16, comp 2.9e-8; PCA ev 4.7e-15, comp 3.0e-8; EN r2 0.999980 vs Ridge 1.000000 (gap 2e-5). These match the Linux oracle numbers recorded for gfx90a/gfx1100.
- Heap stress in one process: `seq.py km32,ts32,pca32,en32,en64,km64,ts64,pca64,en32,en64,km32` exits 0 (no corruption, EN f64 twice). `kmpt.py`: KMeans predict == nearest-centroid argmin 1.0 and == labels_ 1.0; transform (20000,5) argmin agreement 1.0, f32 and f64.
- CPU KMeans back end (`_ch2o4gpu_cpu`): compiles and loads. Not exercised, because upstream marks it "not working".
- Separate pre-existing upstream bug, not fixed (platform-independent): when `lambdas` is passed as a list, `elastic_net_ptr.cpp` copies `lambdas[1..]` but never `lambdas[0]`, so the first lambda stays 0 (the verbose log shows "lambda = 0").

## CUDA compile check at 2c22a86 (2026-11-12)

CUDA no-regression compile check on linux-gfx950 (mi355 host), nvcc 12.8 (V12.8.93), gcc 13.3.0, cmake 3.31.6, SWIG 4.2.1 (pip, into the job venv), numpy 1.26.4, apt OpenBLAS 0.3.26. No role taken, no state changed. No NVIDIA GPU on this host, so nothing was run on a GPU: this is compile and link evidence only.

- Trees: one clone of `AMD-Ecosystem/h2o4gpu` (fetched explicitly; `moat-port` tip is `2c22a862`), worktrees at `9607b099f` (base, the last checked head) and `2c22a86` (head), under the job directory. Only `tests/googletest` initialized. Same scratch-only sed in both, as in the earlier checks: `SET(GPU_COMPUTE_VER 60;61;70;75;80;90)` -> `80`, the `;100;120` append commented out (sm_80 + compute_80; 40 nvcc invocations, all with that pair).
- Commands (`t` = base|head):
  ```
  export PATH=/usr/local/cuda-12.8/bin:$FLEET_VENV/bin:$PATH
  git -C $t submodule update --init tests/googletest
  cmake -S $t -B $t/build -DCMAKE_BUILD_TYPE=Release -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 -DPython_EXECUTABLE=$FLEET_VENV/bin/python -DSWIG_EXECUTABLE=$FLEET_VENV/bin/swig
  cmake --build $t/build -j32 -- -k VERBOSE=1
  ```
- Result: both PASS (configure rc 0, build rc 0, no compiler or linker errors). Both build and link `libcpuh2o4gpu.a`, `libgpuh2o4gpu.a`, `_ch2o4gpu_cpu.so`, `_ch2o4gpu_gpu.so`, `h2o4gpu_test` (24 tests listed by `--gtest_list_tests`; not run).
- Configure summary: identical except the CMake `CMP0146` FindCUDA dev warning at CMakeLists.txt:111 -> :134 (the 23-line `if(WIN32)` block above it).
- Command lines: all 333 compile/link/archive/SWIG commands from the verbose log, sorted with the tree path normalized, are identical, including the nvcc line for `src/gpu/kmeans/kmeans_h2o4gpu.cu` and the host `c++` line for `src/cpu/h2o4gpukmeans.cpp`. The FindCUDA-generated `cuda_compile` scripts (which hold the nvcc flags) are identical too. So the `if(WIN32)` block and the MSVC-frontend `/FI` and `/Ob1` branch do not reach the Linux CUDA build; no `hip_compat`, `amdhip`, `/FI`, `Ob1`, `LightGBM_rocm` or `xgboost_rocm` appears in any generated build file.
- Warnings: the project builds with `-w`. Normalized warning sets from configure+build are identical (14 unique lines: CMP0146, `cmake_policy(CMP0078 OLD)` and googletest `cmake_minimum_required` deprecations, CUDA 12.8 NVTX2 `#pragma message`). Because `-w` hides compiler warnings, the two changed KMeans sources were also recompiled in both trees with their exact build commands, `-w` removed and `-Wall -Wextra` added (nvcc via `-Xcompiler`): 110 warning lines in each tree, the same 32 unique after line-number normalization, all pre-existing (unused parameters, sign-compare, `random_shuffle` deprecation). No new warning from `alloc_result`, the `thrust::copy`/`std::copy` calls, or `<numeric>`.
- Objects (`ar x` of both static libs, sizes and `nm` after hash normalization): only three objects differ.
  - `kmeans_h2o4gpu.cu.o` 4270672 -> 4256792 B. Device code unchanged: `.nv_fatbin` 3219224 B in both, and the PTX (`cuobjdump -ptx`) is identical once the per-compile `_<hash>_17_kmeans_h2o4gpu_cu_<hash>` names are normalized. Host `.text` 15361 -> 15297 B. Host symbols: gone are the `thrust::vector_base<{int,float,double}>::copy_insert`/`allocate_and_copy` instantiations, the `thrust::copy` into a `host_vector<int>` iterator and two `sequential::trivial_copy_n` clones (the old `host_vector::insert` path); new import `calloc`. 14 `kmeans_fit` symbols in both.
  - `h2o4gpukmeans.cpp.o` (CPU back end, host gcc) 86808 -> 85608 B; only symbol change is the new `malloc` import.
  - `projector_direct_dense.cu.o` -32 B (`.nv_fatbin` -8 B): it includes `timer.h`, whose content changed (`_WIN32` branch), so the per-compile internal-linkage hash changed; PTX identical after hash normalization. No code change.
- Modules: `_ch2o4gpu_gpu.so` (11 NEEDED), `_ch2o4gpu_cpu.so` (6) and `h2o4gpu_test` (10) have the same NEEDED and RUNPATH (`/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/lib64/stubs`) in both trees. The only change in undefined dynamic symbols is `PyLong_AsLongLong` added to both SWIG modules (the `elastic_net.i` typemap change; `PyLong_AsLongLong` is in every Python 3 C API).
- Not done: any NVIDIA GPU run (no NVIDIA GPU on this host), the Makefile gbm recipes (untouched by 2c22a86).
- Build trees and the clone stay in the job directory (scratch); nothing was pushed.
## Review 2026-11-11 (windows-gfx1151) -- moat-port 2c22a86 (delta 9607b099f..2c22a86) plus cuda-to-rocm lessons in f2b94d0b

### Port Review: h2o4gpu (moat-port `2c22a86` vs `9607b099f`)

#### Summary
One `[ROCm]` commit adds the Windows CMake/clang-cl HIP build, moves the KMeans result buffers to `malloc`/`calloc`, and compiles `h2o4gpuglm.cu` with `/Ob1` under the clang-cl HIP front end. Verdict: **Request Changes** (3 items, all small).

#### Port Correctness
1. `src/gpu/kmeans/kmeans_h2o4gpu.cu:947-948` -- `kmeans_transform` allocates `(rows / n_gpu) * n_gpu * k` elements, but the buffer replaces the data pointer of the NumPy array that `src/interface_py/h2o4gpu/solvers/kmeans.py:407` created with `rows * k` elements, and `kmeans.py:420` reshapes it to `(rows, k)`. With `n_gpu > 1` and `rows % n_gpu != 0`, NumPy reads `(rows % n_gpu) * k` elements past the end of the `calloc` block. The old `insert` vector had the same size, so this is inherited, but this commit rewrites exactly this allocation to satisfy the NumPy handoff contract: size it to what the consumer views, `alloc_result<T>(rows * k)` (calloc zero-fills the tail, as `kmeans_fit` already does for labels with `alloc_result<int>(rows)`). Single-GPU behavior is unchanged.

#### Build System
2. `CMakeLists.txt:346-347` -- the comment states as fact that the kernel "never completes on ROCm for Windows". The evidence (notes, "Defect 2") is one arch and one toolchain (gfx1151, ROCm 7.14 / clang 23), with no source-level cause found and no reduced reproducer. Say that: e.g. "With /Ob2 the double-precision FuncEval reduction kernel never completes on gfx1151 with ROCm 7.14 for Windows. The cause is not established (suspected device code generation); /Ob1 avoids it." Apply the same qualifier to the matching sentence in the commit message ("`h2o4gpuglm.cu` is compiled with `/Ob1`: ..."), which also reads as an established cause. The guard itself is right: it sits inside `if(USE_HIP)` (never reached by the CUDA build), under `CMAKE_HIP_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"` (GNU on Linux hipcc/clang), and `build-win/build.ninja` shows `/Ob2` followed by `/Ob1` on the `h2o4gpuglm.cu` line only. `FuncEval` is instantiated only in that TU (plus the CPU `.cpp`).

#### Records / lessons
3. The `/Ob1` workaround has no durable removal tracker. The notes say the minimal reproducer and the upstream tool bug draft are "not drafted", and the new lesson itself tells porters to "record that a compiler reproducer is still owed". A notes paragraph is not that record: register it with `utils/deferred.py add <kind> --project h2o4gpu` (reduce the `FuncEvalF<double>` reduction to a standalone `.hip`, diff `/Ob1` vs `/Ob2` ISA, draft the compiler issue for jeff, then drop the `/Ob1` property). In the same edit, move the second lesson ("A Windows "hang" that keeps the host blocked in `hipMemcpy`...", `.claude/skills/cuda-to-rocm/references/fault-classes.md`, under `## Memory and lifetime`) to `references/validation.md` beside the other `## Windows:` diagnostics. It is a hang-diagnosis and codegen-workaround recipe, not a memory/lifetime fault class, and a reader with a hung Windows run looks in validation.md. Change "is often one kernel" to "can be one kernel": it rests on one case. The first lesson (malloc for buffers NumPy frees) is correct (MSVC's x86/x64 `_Allocate_manually_vector_aligned` for >= 4096 bytes) and correctly placed. Both are MOAT-free (`check.py jargon`: ok).

### Not blocking (record only)
- CUDA nvcc compile check of `2c22a86` on the mi355 host is owed before the port goes to jeff (the kmeans change touches the CUDA path: `thrust::copy` from `device_vector` into a raw host pointer). gfx90a and gfx1100 revalidation is owed at the new head.
- The `if(WIN32)` block assumes clang-cl: `_OPENMP_LLVM_RUNTIME` plus `LINK_LIBRARIES(${OpenMP_CXX_LIBRARIES})` would not suit MSVC `cl /openmp`. Upstream never supported Windows and DEVEL.md documents clang-cl only, so this is not a regression.
- Inherited and not touched: the `T** INPLACE_ARRAY1` typemap leaks the NumPy array's original data buffer each time it swaps the pointer. The `kmeans_transform` verbose print reads `rows * cols` elements of a `rows * k` buffer.
- The commit Test Plan gives the C++ run as literal commands but describes the Python oracle only in prose.

Checked with no finding: the free paths (only Python owns the KMeans buffers: `kmeans.i` -> NumPy `free()`; R goes through the Python package, `interface_c` and the gtests do not call the kmeans API, and `examples/cpp/kmeans_driver.cpp` frees nothing); no thrust/std objects leak out of the API any more; `kmeans_fit` sizes labels to `rows` with uniform `rows / n_gpu` chunks; CPU back end `alloc_result(std::vector)` uses `malloc` + `std::copy` with `<cstdlib>`/`<algorithm>` included. `elastic_net.i` `PyLong_AsLongLong`: identical on LP64 and fixes Win64's 32-bit `long`. `timer.h`, the `__has_include` roctx fallback, `windows.h` under `_WIN32` and `NOMINMAX`: the Linux and CUDA preprocessing is unchanged. Commit title is 58 chars, AI disclosure is in prose, no Co-Authored-By. `jargon.py --port h2o4gpu`: clean.

## Porter fix round 2026-11-12 (linux-gfx942, MI300X, ROCm 10.1 pip SDK) -- answers Review 2026-11-11

jeff's request: the three must-fix items of Review 2026-11-11. `moat-port` 2c22a86 -> `d0b78f3d` (two new commits on top, plain push to AMD-Ecosystem/h2o4gpu; nothing amended or force-pushed). Review covers `2c22a86..d0b78f3d`. HIP_VISIBLE_DEVICES=2,3 (dispatcher pin, 2 MI300X), `ulimit -c 0`, LD_LIBRARY_PATH unset, OPENBLAS_NUM_THREADS=32 for every run.

### Item 1 -> `24cd051e` "[ROCm] Allocate the KMeans transform result for every input row"
- `kmeans_transform`: `alloc_result<T>(chunk_distances * n_gpu)` -> `alloc_result<T>(rows * k)`. Other result allocations checked: `kmeans_fit` centroids (`d_centroids[0]->size()` = `k*cols`, matches `kmeans.py`) and labels (`rows`), `kmeans_predict` labels (`rows`) were already right. CPU back end (`src/cpu/h2o4gpukmeans.cpp`) fixes `n_cpu = 1`, so its `n / n_cpu` sizes equal `rows`: no change.
- Allocation evidence (valgrind/ASan not used: the NumPy reader is uninstrumented, so ASan cannot see its read; a `calloc` interposer `agent_space/km/libcalloc_log.so` via `LD_PRELOAD` logs requested sizes instead). `n_gpus=2`, 20001 rows, k=5: NumPy `np.zeros` requests 400020 B (f32) / 800040 B (f64); the library then requested `calloc(100000, 4)` / `calloc(100000, 8)` at 2c22a86 (20 B / 40 B short of the view NumPy reads) and `calloc(100005, 4)` / `calloc(100005, 8)` at d0b78f3d.
- Python API (`agent_space/km/km_multi.py <n_gpus> <rows> <dtype>`, in-tree package: `make apply-sklearn`, `build-hip/_ch2o4gpu_*pu.so` + `ch2o4gpu_*pu.py` copied into `src/interface_py/h2o4gpu/libs/`, `PYTHONPATH=src/interface_py`; checks fit labels, predict and transform against NumPy nearest-centroid and squared distance):
  - `n_gpus=1`, 20001 and 20000 rows, f32 and f64: ALL PASS (labels/predict agreement 1.0; transform max rel err 1.5e-7 f32 / 3.8e-16 f64).
  - `n_gpus=2`: shapes correct and no out-of-bounds read, but values FAIL, from two inherited upstream bugs that the review item does not cover (same code at 9607b099 and on CUDA; behaviour identical at 2c22a86 and d0b78f3d):
    - transform: every row wrong for n_gpus>1, even with 20000 rows (argmin agreement 0.337; 0.262 at 20001). Each GPU's `chunk x k` block is column-major and the blocks are concatenated, while `kmeans.py:420` reshapes the whole buffer as one column-major `(rows, k)`. `agent_space/km/layout.py`: each block read as its own `(chunk, k)` F-order matrix matches NumPy to 3.8e-16.
    - remainder rows: with 20001 rows the last row is never processed by fit, predict or transform (`rows / n_gpu` per GPU; the code's own "TODO: that may ignore up to n_gpu - 1 rows"): label 0 (ref 4), distances 0 (calloc zero-fill, now inside the allocation).
    - Registered for a person's ruling (fix in the port vs upstream report only): `deferred.py` id `h2o4gpu-kmeans-multi-gpu-transform-and-remainder`. Not fixed here: outside the review item and upstream behaviour on CUDA too.
  - `n_gpus=3` not tested: the dispatcher pin exposes two GPUs and `device_count` caps the request at 2.
- C++ `build-hip/h2o4gpu_test`: 24/24 PASS, three runs.

### Item 2 -> `d0b78f3d` "[ROCm] Qualify the comment on the Windows /Ob1 GLM workaround"
Comment only: observed on one GPU (gfx1151) with one toolchain (TheRock ROCm on Windows, clang-cl); suspected device code generation issue, not yet reduced to a reproducer; `/Ob1` avoids it. Guard unchanged. Built in the same Linux tree as item 1 (the build used the working tree that became both commits).

### Item 3 (records)
- `deferred.py add rocm-bug-report --id h2o4gpu-ob2-gfx1151-windows-device-miscompile --project h2o4gpu --component llvm-amdgpu` (reproducer extraction, ISA diff, compiler issue draft for jeff, then drop `/Ob1`).
- `cuda-to-rocm` skill: the Windows hang lesson moved from `references/fault-classes.md` (## Memory and lifetime) to `references/validation.md` as `## Windows: a hang can be one kernel that never finishes`, after the other `## Windows:` sections; "is often" -> "can be". This branch's `utils/check.py` has no `skills` gate; `check.py jargon`: ok.

### Commit-message corrections owed at the PR-prep squash
- `2c22a86` ("[ROCm] Build the GPU library and Python modules on Windows"): the sentence "`h2o4gpuglm.cu` is compiled with `/Ob1`: with `/Ob2` the double-precision `FuncEval` reduction kernel never completes on Windows (gfx1151, ROCm 7.14), so the double-precision ElasticNet fit hung." overstates the evidence. Reword as: with `/Ob2` the kernel was observed never to complete on one GPU (gfx1151) with one toolchain (TheRock ROCm on Windows, clang-cl); the suspected cause is a device code generation issue not yet reduced to a reproducer; `/Ob1` avoids it.
- `2c22a86` Test Plan describes the Python oracle only in prose (Review 2026-11-11, not blocking): give literal commands at the squash.
- (added 2026-11-12, linux-gfx1100 porter; corrected 2026-11-12 per Review 2026-11-12 item 1) `2c22a86`: the `/Ob1` sentence (either wording above) misstates the cause now that it is known, and must not say the fit "hangs on RDNA GPUs". At the squash say: at the default inlining level the GLM reduction (float and double) compiles into a large device function whose long branches clobber the return address (LLVM AMDGPU bug llvm/llvm-project#224205); the double-precision fit was observed to hang on gfx1151 (Windows); the same code pattern is present in Linux RDNA builds, although no hang was seen there; limiting inlining (`/Ob1` for clang-cl, `-finline-hint-functions` otherwise) removes those long branches. Same wording as the `CMakeLists.txt` comment at `0e70429b`.
- (added 2026-11-12, Review 2026-11-12 item 3) `24cd051e` Test Plan runs `PYTHONPATH=src/interface_py python km_multi.py 2 20001 float32`, but `km_multi.py` is not in the tree (it was a porter scratch script). At the squash give literal in-tree commands or an inline `python -c`/heredoc script that builds the data, fits `h2o4gpu.KMeans(n_gpus=2)`, and checks `transform()` has shape `(rows, k)`; the single-GPU sizes can use the reproducers from h2oai/h2o4gpu#900 and #901.
- (added 2026-11-12, Review 2026-11-12 item 3) `1e22b819` Test Plan gives the asm long-branch count and the ElasticNet/sklearn check only in prose. At the squash give the literal commands: the device-only asm compile (the real `h2o4gpuglm.cu` compile line from `compile_commands.json` plus `--cuda-device-only -S --offload-arch=<gfx>`) with an inline count of `s_getpc_b64 s[30:31]`/`s_setpc_b64 s[30:31]` long branches per non-kernel function, and an inline script that fits ElasticNet (float64, 2000x20 and 20000x20) and compares with sklearn.
- (added 2026-11-12) `d0b78f3d` ("[ROCm] Qualify the comment on the Windows /Ob1 GLM workaround"): its body says the cause is a suspected device code generation issue not yet reduced to a reproducer, observed on one GPU and one toolchain. Superseded by `6c756448`; at the squash the two collapse into the single llvm/llvm-project#224205 explanation above.

### Gates and what is owed at d0b78f3d
- `python3 utils/jargon.py --port h2o4gpu`: clean (after fetching the fork's `master` into the clone). `utils/prose.py` on both messages: clean. Fork clone status: only untracked `build-hip/`.
- Owed: CUDA nvcc compile check on the mi355 host at `d0b78f3d` (the kmeans change is in the shared CUDA source); revalidation on linux-gfx90a, linux-gfx1100 and windows-gfx1151 (and the first linux-gfx942 validation).

## Porter 2026-11-12 (linux-gfx1100, Radeon PRO W7800, wave32, ROCm SDK clang 24.0.0git d6f6cb69) -- jeff's ruling on the Windows /Ob1 workaround: keep it, cite llvm/llvm-project#224205, check Linux RDNA

### Question for jeff (Linux RDNA shows the #224205 pattern in the asm, but nothing hangs)
The exact Linux compile line of `src/gpu/h2o4gpuglm.cu` (hipcc-style clang++ `-O3`, no `/Ob` equivalent) produces the bug pattern for gfx1100 and gfx1201: the rocPRIM single-block reduce body over `FuncEvalF<double>` is one large non-kernel function, called with `s_swappc_b64 s[30:31]`, that long-branches through `s_getpc_b64 s[30:31]` ... `s_setpc_b64 s[30:31]` 237 times (gfx1100) / 235 times (gfx1201) with no save or restore of s30/s31 anywhere in the function, and returns through `s_setpc_b64 s[30:31]`. Those are the same counts as the Windows cross-compile. The `FuncEvalF<float>` instance has it too (241 / 434). gfx90a: none through s[30:31]. Yet the float64 GLM never hung on this GPU: 3/3 Windows-reproducer runs and 24 more runs over sizes, both families and both dtypes, all matching sklearn, and the kernel trace shows the affected kernel running and finishing. Per the ruling I added no Linux workaround. Proposed guarded change if you want Linux covered (both verified on this host in asm and at runtime, see below): in the HIP branch's non-MSVC `else()`, `SET_SOURCE_FILES_PROPERTIES(src/gpu/h2o4gpuglm.cu PROPERTIES COMPILE_OPTIONS "SHELL:-mllvm -amdgpu-long-branch-factor=0")` (the long branches stay but use a reserved SGPR pair, so no s[30:31] long branch is left in any non-kernel function), or `-finline-hint-functions` (clang's `/Ob1`: the callee stays out of line and no long branch is left at all). I lean to `-finline-hint-functions`, because it matches the Windows workaround and removes the long branches instead of moving them; the cost is one more device call per reduce, with no measurable change here. Options: (a) leave Linux alone (latent: the pattern exists, but nothing observed it hang); (b) add one of the two guarded options for non-Windows HIP builds and have the next validator round confirm it on gfx1100/gfx90a. State left at `ported` either way.

### State handling
`set-state h2o4gpu linux-gfx1100 porting` refused: `illegal transition ported -> porting` (STAGE_TRANSITIONS has only `ported -> reviewing`). The lock was free, so the round took it with `moatlib.py port-lock h2o4gpu --take linux-gfx1100` (jeff's request authorized this porter round; no holder was displaced), committed and pushed the fork, ran `advance-head`, and released the lock at the end. The stage stayed `ported` the whole time, so the next review covers `2c22a86..6c756448` (the review at 2c22a86 requested changes; 24cd051e, d0b78f3d and 6c756448 have not been reviewed).

### Commit `6c756448` "[ROCm] Cite the LLVM bug behind the Windows /Ob1 GLM workaround"
Comment only, `CMakeLists.txt` in the `CMAKE_HIP_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"` branch: at the default inlining level the double-precision GLM kernel hangs on RDNA GPUs because of llvm/llvm-project#224205 (long-branch relaxation clobbers the return address in a large inlined device function); `/Ob1` keeps that function small enough to avoid it. Guard and `/Ob1` unchanged. Plain push `d0b78f3d..6c756448` to AMD-Ecosystem/h2o4gpu `moat-port`. `utils/prose.py` on the message: clean. `jargon.py --port h2o4gpu`: clean (after `git fetch origin master:master` in the clone). Fork clone status: only untracked `build-hip/`.

### Build on this host (rc 0)
This container has no `libopenblas-dev`; FindBLAS picked ATLAS, whose `/usr/include/x86_64-linux-gnu/cblas.h` has no `extern "C"`, so `h2o4gpu_test` failed to link (`undefined symbol: cblas_dgemm(CBLAS_ORDER, ...)`, "did you mean: extern "C" cblas_dgemm"). An environment gap, not a port defect (earlier validators had apt OpenBLAS). OpenBLAS v0.3.26 built from source into the job directory (`make -j32 USE_OPENMP=1 NO_FORTRAN=1 TARGET=HASWELL NUM_THREADS=64 && make PREFIX=<job>/deps/openblas install`). Extra pip into the job venv: `swig==4.2.1 cython==3.0.11 threadpoolctl psutil tabulate future joblib pytest pytest-timeout`. HIP_VISIBLE_DEVICES=0 (dispatcher pin), `ulimit -c 0`, LD_LIBRARY_PATH unset, OPENBLAS_NUM_THREADS=32.
```
R=$($FLEET_VENV/bin/python -m rocm_sdk path --root); P=$FLEET_VENV/bin/python; OB=<job>/deps/openblas
export PATH=$FLEET_VENV/bin:$R/bin:$PATH CMAKE_PREFIX_PATH=$R OMP_NUM_THREADS=$MAX_JOBS CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
git submodule update --init tests/googletest
cmake -S . -B build-hip -DUSE_CUDA=OFF -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH="$R;$OB" -DBLA_VENDOR=OpenBLAS -DCBLAS_INCLUDE_DIR=$OB/include -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$P -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-hip -j$MAX_JOBS
```
`build-hip/h2o4gpu_test`: 24/24 PASS, 3 runs (at d0b78f3d's code; 6c756448 only changes a comment in a branch Linux never reaches, rebuilt rc 0).

### Asm check
Device asm from the real compile line (`compile_commands.json` entry for `src/gpu/h2o4gpuglm.cu`), with `-o ...o` replaced by `--cuda-device-only -S -o <file>.s` and `--offload-arch` swapped per target (script `agent_space/asm/mkasm.sh <gfx> <out.s> [extra flags]`). Counter `agent_space/asm/lb.py`: splits the file into functions (`.type ...,@function` .. `.Lfunc_end`), marks kernels from `.amdhsa_kernel`, counts a long branch as `s_getpc_b64 s[a:b]` followed by `s_add_u32`/`s_addc_u32` (gfx12: `s_add_co_u32`/`s_add_co_ci_u32`) on a label difference and `s_setpc_b64 s[a:b]`, and looks for any save of s30/s31 (`v_writelane_b32 vN, s30|s31`, `s_mov_b32 sN, s30|s31`, `s_mov_b64 s[..], s[30:31]`) and restore.

| target / flags | non-kernel functions | long branches in them | through s[30:31] | saved s30/s31 | kernel long branches (s[30:31]) |
|---|---|---|---|---|---|
| gfx1100 default | 2 (reduce body, `FuncEvalF<double>` and `<float>`) | 478 | 478 (double 237, float 241) | no (0 saves, 0 restores) | 0 |
| gfx1201 default | 2 (same) | 669 | 669 (double 235, float 434) | no | 13 (0) |
| gfx90a default (control) | 1 | 116 | 0 | n/a | 220 (0) |
| gfx1100 `-mllvm -amdgpu-long-branch-factor=0` | 2 | 478 | 0 | n/a | 0 |
| gfx1201 `-mllvm -amdgpu-long-branch-factor=0` | 2 | 669 | 0 | n/a | 13 (5; kernels have no return address) |
| gfx1100 `-finline-hint-functions` | 44 | 0 | 0 | n/a | 0 |
| gfx1201 `-finline-hint-functions` | 44 | 0 | 0 | n/a | 0 |

The double function on gfx1100: 53185 asm lines, NumVgprs 248 (Windows `/Ob2` reported a 248-VGPR body too), one real return `s_setpc_b64 s[30:31]`, called from its kernel by `s_swappc_b64 s[30:31], s[0:1]`. The Linux layout therefore matches Windows; the difference is not in inlining.

### Runtime (this GPU, in-tree package)
Package: a copy of `src/interface_py/h2o4gpu` in `agent_space/pkg` with `build-hip/_ch2o4gpu_*pu.so` and `ch2o4gpu_*pu.py` in `libs/` and a stub `__init__.py` that aliases `h2o4gpu.{linear_model,cluster,decomposition,utils,metrics}` to sklearn 1.9.1, then appends `__init__.base.py` (the Windows approach; the `make apply-sklearn` overlay builds h2o's scikit-learn fork, which this check does not need). Reproducer `agent_space/en.py <dtype> <rows> [cols]`: `ElasticNetH2O(n_gpus=1, alphas=[0.0], lambdas=[1e-6], n_folds=1, max_iter=5000, tol=1e-6, double_precision=1)` fit + predict vs sklearn `Ridge(alpha=1e-6)` r2 (the Windows `en64` call and the Linux oracle's GLM check); `agent_space/logit.py`: `family="logistic"`, alphas [0.5], lambdas [1e-4] vs `LogisticRegression` accuracy. Every run is `timeout 300 $FLEET_VENV/bin/python agent_space/<script> ...`.
- Windows reproducer f64 2000x20: 3/3 rc 0, 0.8 s each, r2 0.999995 = Ridge (gap 3.8e-13).
- f64 20000x20 (the Linux validation's GLM check) 3/3, gap 1.8e-13; f32 20000x20 3/3, gap 0; f64 200x20 3/3; f64 20001x7 3/3; f64 100000x50 3/3 (4.5 s); logistic f64 2000 3/3 (acc 0.8775 = sklearn), logistic f64 20000 3/3 (0.8907 vs 0.8908), logistic f32 20000 3/3. 27 runs, all rc 0, none near the watchdog.
- `rocprofv3 --kernel-trace` on the f64 2000x20 run: 17605 dispatches; the `reduce_impl<true, ... zip_function<FuncEvalF<double>> ...>` kernel (grid 256, block 256: one block), whose callee is the function with the 237 unsaved branches, ran twice and finished.
- No GLM pytest ran: `tests/python/open_data` has only `system/` and `gbm/` (no GLM), and `tests/python/{small,big}/test_glm_*.py` need the Kaggle datasets (not synced). The earlier Linux validations covered GLM only through the oracle's ElasticNetH2O f32/f64 20000x20, repeated above.
- Why no hang on Linux (hypothesis, not verified): the bug fires only when a long branch is actually taken before the return, and which blocks run depends on the data path (the `FuncEval` switch on the function type, tail handling of the reduce). These inputs apparently never take one of the 237 branches on gfx1100, while the Windows gfx1151 run took one. Not tested: instrumenting which long-branch blocks execute.
- Variants (both built by recompiling only `h2o4gpuglm.cu.o` with the extra flag and relinking `_ch2o4gpu_gpu.so` from `link.txt`, `agent_space/variant.sh <tag> <flags>`): `-mllvm -amdgpu-long-branch-factor=0` and `-finline-hint-functions` each pass en f64 2000 3/3, en f64 20000 3/3, logistic f64 20000 3/3, en f32 20000 3/3, with the same r2/accuracy and timings within 0.2 s of the default build. They cannot show a hang being avoided here, because the default build does not hang on this GPU.

### Records
- The deferral `h2o4gpu-ob2-gfx1151-windows-device-miscompile` (reproducer, ISA diff, compiler issue draft, then drop `/Ob1`) now has its compiler issue: llvm/llvm-project#224205 (related LLVM PR #229427). Not edited here (it is a person's ruling); dropping `/Ob1` waits for a TheRock/ROCm compiler with the fix.

### Owed at 6c756448
- CUDA nvcc compile check on the mi355 host at `6c756448` (last checked at 2c22a86; 24cd051e changed shared KMeans source).
- Revalidation on linux-gfx90a, linux-gfx1100 and windows-gfx1151 (and the first linux-gfx942 validation).
- Review of `2c22a86..6c756448`.

## KMeans multi-GPU reproducers 2026-11-12 (linux-gfx942, 2x MI300X, ROCm 10.1 pip SDK)

jeff's request: verified output of two reproducers for the upstream h2oai/h2o4gpu issues he will file on deferral `h2o4gpu-kmeans-multi-gpu-transform-and-remainder`. No role, no state change, no fork change.

- Build: fresh clone of `https://github.com/AMD-Ecosystem/h2o4gpu.git` `moat-port` at `d0b78f3d7749c7bf36f9eeda6d138acb3f6d2673` in the job directory. `HIP_VISIBLE_DEVICES=6,7` (dispatcher pin), `ulimit -c 0`, `LD_LIBRARY_PATH` unset, `OPENBLAS_NUM_THREADS=32`.
  ```
  P=$FLEET_VENV/bin/python; R=$($P -m rocm_sdk path --root); export PATH=$FLEET_VENV/bin:$R/bin:$PATH
  $P -m pip install swig==4.2.1 cython==3.0.11 meson-python ninja threadpoolctl joblib
  git submodule update --init tests/googletest
  cmake -S . -B build-hip -DUSE_CUDA=OFF -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_PREFIX_PATH=$R -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$P
  cmake --build build-hip -j$MAX_JOBS
  make apply-sklearn PYTHON=$P
  cp build-hip/_ch2o4gpu_*pu.so build-hip/ch2o4gpu_*pu.py src/interface_py/h2o4gpu/libs/
  PYTHONPATH=<clone>/src/interface_py $P repro_transform.py
  ```
  Recipe note: SWIG from pip defaults to 4.5.1, whose generated wrapper fails to compile against the project's `numpy.i` (`PyString_Check`/`PyInt_Check` not declared); 4.2.1 builds (apt's 4.2.0 did in earlier rounds). `device_count(-1)` = (2, 2).
- Scripts run exactly as jeff wrote them; `KMeansH2O.__init__` already takes `n_clusters, n_gpus, random_state, max_iter`, so no change.

repro_transform.py:
```python
# KMeans.transform with n_gpus=2 vs NumPy distances to the fitted centers.
import numpy as np
from h2o4gpu.solvers.kmeans import KMeansH2O

rng = np.random.RandomState(0)
X = rng.rand(1000, 4)  # 1000 rows, divisible by 2
for n_gpus in (1, 2):
    km = KMeansH2O(n_clusters=3, n_gpus=n_gpus, random_state=0, max_iter=50).fit(X)
    C = km.cluster_centers_
    got = km.transform(X)
    want = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(-1))
    print("n_gpus=%d max|got-want|=%.3g argmin agreement=%.3f"
          % (n_gpus, np.abs(got - want).max(),
             (got.argmin(1) == want.argmin(1)).mean()))
```

repro_remainder.py:
```python
# With rows % n_gpus != 0, fit and predict never process the last rows.
import numpy as np
from h2o4gpu.solvers.kmeans import KMeansH2O

rng = np.random.RandomState(0)
X = rng.rand(1001, 4)  # 1001 rows, n_gpus=2 -> 1 row left over
for n_gpus in (1, 2):
    km = KMeansH2O(n_clusters=3, n_gpus=n_gpus, random_state=0, max_iter=50).fit(X)
    C = km.cluster_centers_
    # Last row placed exactly on center 1 and center 2: predict must return 1 and 2.
    X1 = X.copy(); X1[-1] = C[1]
    X2 = X.copy(); X2[-1] = C[2]
    nearest = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)).argmin(1)
    print("n_gpus=%d fit labels_[-1]=%d (nearest center %d)  predict last row on C[1]: %d, on C[2]: %d"
          % (n_gpus, km.labels_[-1], nearest[-1], km.predict(X1)[-1], km.predict(X2)[-1]))
```

Raw output (each script twice; identical both times):
```
=== repro_transform.py run 1 (HIP_VISIBLE_DEVICES=6,7) ===
n_gpus=1 max|got-want|=0.25 argmin agreement=1.000
n_gpus=2 max|got-want|=1.06 argmin agreement=0.423
rc=0
=== repro_transform.py run 2 (HIP_VISIBLE_DEVICES=6,7) ===
n_gpus=1 max|got-want|=0.25 argmin agreement=1.000
n_gpus=2 max|got-want|=1.06 argmin agreement=0.423
rc=0
=== repro_remainder.py run 1 (HIP_VISIBLE_DEVICES=6,7) ===
n_gpus=1 fit labels_[-1]=0 (nearest center 0)  predict last row on C[1]: 1, on C[2]: 2
n_gpus=2 fit labels_[-1]=0 (nearest center 0)  predict last row on C[1]: 0, on C[2]: 0
rc=0
=== repro_remainder.py run 2 (HIP_VISIBLE_DEVICES=6,7) ===
n_gpus=1 fit labels_[-1]=0 (nearest center 0)  predict last row on C[1]: 1, on C[2]: 2
n_gpus=2 fit labels_[-1]=0 (nearest center 0)  predict last row on C[1]: 0, on C[2]: 0
rc=0
```

Reading the output:
- Transform: the 2-GPU bug shows (argmin agreement 0.423 vs 1.000). But the 1-GPU line is not exact either: `max|got-want|=0.25` because `transform` returns **squared** Euclidean distances and the script compares with `np.sqrt(...)`. A separate diagnostic (`diag.py`, scripts untouched) on the same data: n_gpus=1 `max|got - squared|` = 8.88e-16; n_gpus=2 `max|got - squared|` = 1.22, argmin 0.423, and re-reading the flat buffer as two per-GPU `(500, 3)` column-major blocks (GPU 0 then GPU 1) matches squared distances to 1.22e-15. For the issue, compare with squared distances (drop the `np.sqrt`), or the 1-GPU line looks wrong too.
- Remainder: predict shows the bug (2-GPU last row returns 0 for both C[1] and C[2]; 1-GPU returns 1 and 2). The fit half does not show it with this seed: the last row's nearest center happens to be 0, the same as the never-written zero-filled label. The diagnostic's second 1001-row draw (the RNG state after the 1000-row draw) has nearest center 1: n_gpus=1 `labels_[-1]=1`, n_gpus=2 `labels_[-1]=0`.

Source check (read-only), upstream h2oai/h2o4gpu master `97279defc47329147fd20d62381608c20cd3efd7`, `src/gpu/kmeans/kmeans_h2o4gpu.cu`:
- Confirmed. `kmeans_transform` (line 893) gathers the per-GPU blocks with `#pragma omp parallel for` at line 945 and `h_pairwise_distances->insert(h_pairwise_distances->end(), d_pairwise_distances[q]->begin(), d_pairwise_distances[q]->end())` at lines 947-949, into one `thrust::host_vector<T>` created at line 944. Concurrent `insert(end(), ...)` on one vector is a data race (reallocation and size update unsynchronised), and the order of the blocks depends on thread scheduling. The pragma is live on CUDA: `CMAKE_CXX_FLAGS` gets `${OpenMP_CXX_FLAGS}` (lines 37-39) and the GPU library is built with FindCUDA's `CUDA_ADD_LIBRARY` (line 208), whose host-flag propagation (`CUDA_PROPAGATE_HOST_FLAGS`, default ON) passes it to the host compiler (inferred from CMake defaults, not observed with an nvcc build here).
- The port no longer has that loop: `2c22a862` replaced it. `moat-port` `d0b78f3d` `src/gpu/kmeans/kmeans_h2o4gpu.cu` lines 947-955: `alloc_result<T>(rows * k)` then, under the same `#pragma omp parallel for` (line 949), `thrust::copy(..., h_pairwise_distances + q * chunk_distances)` into disjoint offsets: no race, fixed block order. (HIP TUs are compiled without OpenMP anyway, so the loop is serial on ROCm.) The column-major block layout bug (`kmeans.py:420` reshape) and the remainder bug remain, as the output shows.

## Porter 2026-11-12 round 2 (linux-gfx1100, Radeon PRO W7800, wave32, ROCm 10.1 pip SDK, clang 24.0.0git d6f6cb69) -- jeff's ruling: Linux gets -finline-hint-functions

### Ruling
jeff 2026-11-12: Linux gets -finline-hint-functions. Option (b) of the question in "Porter 2026-11-12": Linux HIP builds get the guarded `-finline-hint-functions` workaround for llvm/llvm-project#224205 on `src/gpu/h2o4gpuglm.cu`, matching the Windows clang-cl `/Ob1`.

### Commit `1e22b819` "[ROCm] Limit inlining in the GLM file on Linux HIP builds too"
`CMakeLists.txt`, HIP branch only (the CUDA path is untouched): the non-MSVC `else()` gains `SET_SOURCE_FILES_PROPERTIES(src/gpu/h2o4gpuglm.cu PROPERTIES COMPILE_OPTIONS "-finline-hint-functions")`, the same mechanism as the Windows `/Ob1` line. No other `COMPILE_OPTIONS` is set on that file (its only other property is `LANGUAGE HIP`, from the `GPU_CU_SOURCES` line), so nothing is clobbered. The comment moved above the `if()` and covers both branches: the LLVM AMDGPU bug, why limiting inlining avoids it, `/Ob1` is the clang-cl spelling. Plain push `6c756448..1e22b819` to AMD-Ecosystem/h2o4gpu `moat-port` (fresh clone from `https://github.com/AMD-Ecosystem/h2o4gpu.git`, `moatlib.py protect-fork` installed before the push). `utils/prose.py` on the message: clean. `jargon.py --port h2o4gpu`: clean. Fork clone status: only untracked `build-hip/`.

### Build (rc 0, both trees)
New head in `projects/h2o4gpu/src`, baseline `6c756448` in a detached worktree beside it. Same recipe as "Porter 2026-11-12" (OpenBLAS v0.3.26 built from source into this job's directory, same make line; pip `swig==4.2.1 cython==3.0.11 threadpoolctl psutil tabulate future joblib pytest-timeout`). HIP_VISIBLE_DEVICES=0 (dispatcher pin), `ulimit -c 0`, LD_LIBRARY_PATH unset, OPENBLAS_NUM_THREADS=32.
```
cmake -S . -B build-hip -DUSE_CUDA=OFF -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH="$R;$OB" -DBLA_VENDOR=OpenBLAS -DCBLAS_INCLUDE_DIR=$OB/include -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$P -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-hip -j$MAX_JOBS
```

### Compile commands, 1e22b819 vs 6c756448
`compile_commands.json` of both trees, source path normalized: 37 entries each, same file set; exactly one entry differs, `src/gpu/h2o4gpuglm.cu`, and only by the added token `-finline-hint-functions`.

### Asm (device-only asm from each tree's real compile line, `--offload-arch` swapped; same `mkasm.sh`/`lb.py` method as "Porter 2026-11-12")

| target | 6c756448: non-kernel funcs / their long branches / through s[30:31] (unsaved) / kernel long branches | 1e22b819: same columns |
|---|---|---|
| gfx1100 | 2 / 478 / 478 (478; double 237, float 241) / 0 | 44 / 0 / 0 / 0 |
| gfx1201 | 2 / 669 / 669 (669; double 235, float 434) / 13 | 44 / 0 / 0 / 0 |
| gfx90a | 1 / 116 / 0 / 220 (0 through s[30:31]) | 44 / 0 / 0 / 0 |

Baseline numbers reproduce the previous section exactly. Deviation from the expected "gfx90a unchanged in branch count": the flag applies to the file for every offload arch, so gfx90a also goes from 336 long branches (116 in its one non-kernel function, 220 in kernels, none through s[30:31]) to 0. The s[30:31] count on gfx90a is 0 before and after; the gfx90a runtime effect (if any) belongs to its revalidation. gfx942 asm was not generated.

### Runtime on gfx1100, 1e22b819 vs 6c756448
- `build-hip/h2o4gpu_test`, 3 runs each: 24/24 PASS every run; 0.37-0.39 s new, 0.37-0.38 s base.
- GLM reproducers (`en.py`, `logit.py` from "Porter 2026-11-12", in-tree package per build, each run `timeout 300`), 3 runs each per build, 42 runs, all rc 0, results identical between builds. Times (new / base, seconds, three runs):

| case | result | new | base |
|---|---|---|---|
| en f64 2000x20 (Windows reproducer) | r2 0.999995 = Ridge, gap 3.8e-13 | 0.75 0.80 0.83 | 0.83 0.81 0.75 |
| en f64 20000x20 | gap 1.8e-13 | 1.79 1.81 1.78 | 1.79 1.58 1.79 |
| en f32 20000x20 | gap 0 | 1.56 1.53 1.52 | 1.51 1.56 1.53 |
| en f64 100000x50 | gap 1.9e-13 | 4.27 4.31 4.33 | 4.29 4.50 4.32 |
| logistic f64 2000 | acc 0.8775 = sklearn | 0.67 0.69 0.70 | 0.68 0.69 0.69 |
| logistic f64 20000 | acc 0.8907 vs 0.8908 | 1.11 1.10 1.20 | 1.21 1.20 1.21 |
| logistic f32 20000 | acc 0.8907 vs 0.8908 | 0.97 0.91 0.99 | 0.98 0.97 0.98 |

No measurable cost. As before, the baseline does not hang on this GPU either, so this shows no regression, not a hang avoided.

### State handling
Same as the previous round: lock taken with `moatlib.py port-lock h2o4gpu --take linux-gfx1100` (free; jeff's dispatch authorized it), fork pushed, `advance-head h2o4gpu 1e22b819bd1d84e7d801099148fef4c24d20b430`, lock released. Stage stays `ported`, so the next review covers `2c22a86..1e22b819`.

### Owed at 1e22b819
- Review of `2c22a86..1e22b819`.
- CUDA nvcc compile check on the mi355 host at `1e22b819` (last at 2c22a86; the CMake change is inside the HIP branch, but 24cd051e changed shared KMeans source).
- Revalidation on linux-gfx90a, linux-gfx1100 and windows-gfx1151, plus the first linux-gfx942 validation.

## Review 2026-11-12 (linux-gfx1201) -- moat-port 1e22b819 (delta 2c22a86..1e22b819)

### Port Review: h2o4gpu (moat-port `1e22b819` vs `2c22a86`)

#### Summary
Four commits answer Review 2026-11-11: the KMeans transform result is sized `rows * k` (24cd051e), the Windows `/Ob1` comment is qualified (d0b78f3d) and then cites llvm/llvm-project#224205 (6c756448), and Linux HIP builds get `-finline-hint-functions` on `src/gpu/h2o4gpuglm.cu` per jeff's ruling (1e22b819). Verdict: **Request Changes** (3 items; one comment line in the fork, the rest records and the skill lesson).

#### Build System
1. `CMakeLists.txt:345-347` -- the shared comment says "at the default level the double-precision GLM kernel hangs on RDNA GPUs". The evidence does not support the general present tense: the hang was observed only on gfx1151 (Windows, clang-cl); on gfx1100 the default Linux build ran 69 GLM runs without a hang ("Porter 2026-11-12" and round 2), and 1e22b819's own message says "No hang was observed on gfx1100". The asm shows the defect pattern in both precisions (`FuncEvalF<double>` 237/235 and `<float>` 241/434 long branches through s[30:31] on gfx1100/gfx1201), not only double. Reword to what is shown, e.g. "at the default level the GLM reduction is compiled into a large non-kernel device function that LLVM AMDGPU miscompiles on RDNA GPUs (llvm/llvm-project#224205: long-branch relaxation clobbers the return address); the double-precision fit hung on gfx1151." Apply the same correction to the 2c22a86 squash text in "Commit-message corrections owed at the PR-prep squash" (the entry added 2026-11-12 says "hangs on RDNA GPUs" and claims to match 6c756448, whose message correctly says "hung on an RDNA GPU"). The rest of the comment matches the asm: limiting inlining leaves 44 non-kernel functions and 0 long branches on gfx1100/gfx1201/gfx90a.

#### Records / lessons
2. `.claude/skills/cuda-to-rocm/references/validation.md:126-128` ("## Windows: a hang can be one kernel that never finishes") is now stale against the known cause. It tells the reader to scope the `/Ob1` workaround "under the MSVC HIP front end" and to record that a compiler reproducer is still owed, and attributes the case to gfx1151/TheRock only. The cause is now an identified LLVM AMDGPU bug that is present in Linux RDNA builds too (gfx1100/gfx1201 asm), and this port carries the workaround on both front ends. Update the lesson (on this branch, MOAT-free): name the bug (llvm/llvm-project#224205), give the asm signature that confirms it (in a non-kernel function, `s_getpc_b64 s[30:31]` ... `s_setpc_b64 s[30:31]` long branches with no save/restore of s30/s31, found with `--cuda-device-only -S`), and give both spellings of the per-file workaround (`/Ob1` for clang-cl, `-finline-hint-functions` for clang/hipcc) through `COMPILE_OPTIONS` on that source. Keep the Windows hang-diagnosis half (`AMD_LOG_LEVEL=3`, `PAL fence isn't ready!`) where it is, or split the codegen half into its own section where a Linux reader looks.
3. Test Plans that are not literal, reproducible commands are missing from the squash corrections list: `24cd051e` runs `PYTHONPATH=src/interface_py python km_multi.py 2 20001 float32`, but `km_multi.py` is not in the tree (it lives in a porter's `agent_space/`), and `1e22b819` gives the asm count and the ElasticNet/sklearn check only in prose. Add both to "Commit-message corrections owed at the PR-prep squash", with the literal commands to use at the squash (inline the check script, or use the reproducers from h2oai/h2o4gpu#900/#901 for the single-GPU sizes).

### Checked with no finding
- Review 2026-11-11 item 1: `kmeans_h2o4gpu.cu:948` `alloc_result<T>(rows * k)`, matching `kmeans.py`'s `(rows, k)` view; the per-GPU copies write at `q * chunk_distances` (disjoint, inside the allocation). Siblings: `kmeans_fit` centroids `d_centroids[0]->size()` (`k*cols`) and labels `rows` (:774, :779), `kmeans_predict` labels `rows` (:845); CPU back end `alloc_result(std::vector)` with `n_cpu = 1` (`h2o4gpukmeans.cpp:252/366/434`), so its vector sizes are `rows`. Item 2: the d0b78f3d qualification is superseded by the cited bug (6c756448, 1e22b819) and recorded for the squash. Item 3: deferral `h2o4gpu-ob2-gfx1151-windows-device-miscompile` registered; lesson moved to `validation.md` with "can be".
- 1e22b819 CMake: the new `SET_SOURCE_FILES_PROPERTIES` is in the `else()` of `CMAKE_HIP_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"` inside `if(USE_HIP)` (`CMakeLists.txt:299`), and `USE_CUDA`/`USE_HIP` are mutually exclusive (:129-130), so the CUDA path (FindCUDA `CUDA_ADD_LIBRARY`, :236) never sees it. It names only `src/gpu/h2o4gpuglm.cu`; that file's only other property is `LANGUAGE HIP` (:341), and no other `COMPILE_OPTIONS` is set on any source in the tree (`git grep h2o4gpuglm` outside sources: the two lines only). The target is created in the same directory (:361), so the directory-scoped property applies. Porter's `compile_commands.json` diff: one entry changed by the one token. The Makefile's HIP path calls the same CMake (`Makefile:123`).
- Commit messages 24cd051e, d0b78f3d, 6c756448, 1e22b819: `[ROCm]` titles of 61-63 chars, rationale, fenced Test Plan, AI disclosure in prose, no Co-Authored-By. 1e22b819's numbers match the round-2 asm table (478/669 -> 0 through s[30:31]; gfx90a 336 -> 0, none through s[30:31]). The corrections list covers 2c22a86 and d0b78f3d (the two messages that now misstate the cause); 6c756448's message is accurate. `python3 utils/jargon.py --port h2o4gpu`: clean. `utils/check.py` commits gate flags only `24e6169`/`4e5e433` (pre-existing, covered by the 2026-11-05 squash ruling). Fork clone at 1e22b819 clean.

### Not blocking (record only)
- Owed, already dispatched or known: the mi355 CUDA compile check at 1e22b819 and revalidation on linux-gfx90a, linux-gfx1100, windows-gfx1151 plus the first linux-gfx942 validation. The flag also changes gfx90a/gfx942 device code (336 -> 0 long branches on gfx90a), so their revalidation covers it.

### Records (jeff, 2026-11-12)
- Deferral `h2o4gpu-kmeans-multi-gpu-transform-and-remainder`: jeff reported the inherited multi-GPU KMeans bugs upstream on 2026-11-12 as https://github.com/h2oai/h2o4gpu/issues/900 (transform block layout and the `omp parallel for` insert race) and https://github.com/h2oai/h2o4gpu/issues/901 (remainder rows). jeff's ruling: report them upstream only; do not fix them in the port. Both issues verified open, authored by jeffdaily (read-only `gh issue view`). Deferral status not changed here (a person's ruling to record through `deferred.py decide`).
- Deferral `h2o4gpu-ob2-gfx1151-windows-device-miscompile`: its compiler bug is llvm/llvm-project#224205 ("[AMDGPU] Long-branch reserved SGPR pair is the return address s[30:31] in a callable function ...", open); jeff commented there on 2026-11-12. Deferral status not changed here. The `/Ob1` and `-finline-hint-functions` properties can go once the ROCm compilers the port supports carry the fix.
## CUDA compile check at 1e22b819 (2026-11-12)

CUDA no-regression compile check on linux-gfx950 (mi355 host), nvcc 12.8 (V12.8.93), gcc 13.3.0, cmake 3.31.6 (pip, job venv), SWIG 4.2.1 (pip, job venv), numpy 1.26.4, apt OpenBLAS 0.3.26. No role taken, no state changed. No NVIDIA GPU on this host, so nothing ran on a GPU: this is compile and link evidence only.

- Trees: one clone of `https://github.com/AMD-Ecosystem/h2o4gpu.git` (`moat-port` tip `1e22b819bd1d`), detached worktrees at `2c22a86` (base, last checked head) and `1e22b819` (head) under the job directory. Delta: `24cd051e` (`kmeans_transform` result allocated `rows * k`, `src/gpu/kmeans/kmeans_h2o4gpu.cu`), `d0b78f3d`/`6c756448`/`1e22b819` (CMakeLists.txt, all inside `if(USE_HIP)`; 1e22b819 adds `-finline-hint-functions` on `h2o4gpuglm.cu` in the non-MSVC HIP branch). Only `tests/googletest` initialized. Same scratch-only sed in both as in the earlier checks: `SET(GPU_COMPUTE_VER 60;61;70;75;80;90)` -> `80`, the `;100;120` append commented out (sm_80 + compute_80).
- Commands (`t` = base|head), the 9607b099f recipe:
  ```
  export PATH=/usr/local/cuda-12.8/bin:$FLEET_VENV/bin:$PATH
  git -C $t submodule update --init tests/googletest
  cmake -S $t -B $t/build -DCMAKE_BUILD_TYPE=Release -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 -DPython_EXECUTABLE=$FLEET_VENV/bin/python -DSWIG_EXECUTABLE=$FLEET_VENV/bin/swig
  cmake --build $t/build -j32 -- -k VERBOSE=1
  ```
- Result: both PASS (configure rc 0, build rc 0; the only `error` strings in the logs are the gtest flag `-Wno-error=dangling-else`). Both build and link `libcpuh2o4gpu.a`, `libgpuh2o4gpu.a`, `_ch2o4gpu_cpu.so`, `_ch2o4gpu_gpu.so`, `h2o4gpu_test` (24 tests listed by `--gtest_list_tests`; not run).
- Configure summary: identical after the tree path is normalized (this time not even the `CMP0146` line moves; the CMake edits are inside the HIP branch below it).
- Command lines: all 258 command lines of the verbose log (compile, nvcc `-M`/`-c`, link, archive, SWIG, cmake helpers; 40 nvcc lines), sorted with the tree path normalized, are identical between the trees, including the nvcc line for `src/gpu/kmeans/kmeans_h2o4gpu.cu` (same flags, the changed source is the only difference). All 20 FindCUDA `*.Release.cmake` compile scripts (which carry the nvcc flags) are identical. No `-finline-hint-functions`, `Ob1`, `hip_compat`, `amdhip`, `LightGBM_rocm` or `xgboost_rocm` in any generated build file.
- CPU back end: 24cd051e changes no CPU source; `src/cpu/h2o4gpukmeans.cpp` (host `c++`) has the identical command line and `h2o4gpukmeans.cpp.o` is byte-identical. The KMeans `.cu` file's host side goes through nvcc with `-ccbin /usr/bin/cc`.
- Warnings: the project builds with `-w`. Normalized configure+build warning sets are identical (14 unique lines: CMP0146, `cmake_policy(CMP0078 OLD)` and googletest `cmake_minimum_required` deprecations, CUDA 12.8 NVTX2 `#pragma message`). Because `-w` hides compiler warnings, `kmeans_h2o4gpu.cu` was also recompiled in both trees with its exact nvcc line, both `-w` removed and `-Xcompiler -Wall,-Wextra` added: rc 0, 70 warning lines in each tree, the same 18 unique after line-number normalization, all pre-existing. No new warning from line 948.
- Objects (`ar x` of both static libs; sizes, `size -A`, `nm`, `cuobjdump -ptx` and `objdump -d` compared after normalizing the per-compile `_INTERNAL_<hash>`/`_GLOBAL__N__<hash>` and `tmpxft_<pid>` names, which depend on the tree path and PID):
  - PTX of all 14 nvcc objects identical, so no device-code change anywhere (SASS not compared; this CUDA install has no `nvdisasm`).
  - The only code difference in either library is host `.text` of `h2o4gpukmeans::kmeans_transform<float>` and `<double>` (`<double>` 1939 -> 1955 B, `<float>` same size): the `imul` sequence feeding `calloc` now computes `rows * k` instead of `(rows / n_gpu * k) * n_gpu`. Every other function in `kmeans_h2o4gpu.cu.o` disassembles identically; symbol sets are identical.
  - `.nv_fatbin` -8 B in `h2o4gpuglm.cu.o`, `matrix_sparse.cu.o` and `projector_direct_dense.cu.o` (sources unchanged, PTX identical after normalization): length of the path-dependent hash names. `libgpuh2o4gpu.a` 19384650 -> 19384618 B; `libcpuh2o4gpu.a`, both modules and `h2o4gpu_test` have the same sizes.
- Modules: `_ch2o4gpu_gpu.so`, `_ch2o4gpu_cpu.so` and `h2o4gpu_test` have identical NEEDED and RUNPATH in both trees; `_ch2o4gpu_gpu.so` undefined dynamic symbols identical.
- Not done: any NVIDIA GPU run (no NVIDIA GPU on this host); the Makefile gbm recipes (untouched by this delta).
- Build trees and the clone stay in the job directory (scratch); nothing was pushed to the fork.

## Porter fix round 2026-11-12 (linux-gfx1100, Radeon RX 7900 host, ROCm SDK pip, job venv) -- answers Review 2026-11-12

jeff's request: address the three must-fix items of Review 2026-11-12 (scope `2c22a86..1e22b819`). No upstream or review PR exists, so `moat-port` is not frozen; plain push on top.

### State handling
Stage was `changes-requested` (not `ported` as in the earlier rounds), so the legal path applied: `port-lock h2o4gpu --take linux-gfx1100` (lock free, no holder displaced), `set-state h2o4gpu linux-gfx1100 porting`, fork commit and plain push, `advance-head`, `set-state ... ported` (releases the lock). The next review covers `1e22b819..0e70429b` plus this round's answers to the review's findings.

### Item 1: commit `0e70429b` "[ROCm] State only the observed GLM hang in the inlining comment"
Comment only, `CMakeLists.txt` above the `CMAKE_HIP_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"` block: at the default level the GLM reduction (float and double) compiles into a large device function whose long branches clobber the return address (llvm/llvm-project#224205); the double-precision fit was observed to hang on gfx1151 (Windows); the same code pattern is present in Linux RDNA builds, although no hang was seen there; limiting inlining removes those long branches. `/Ob1` and `-finline-hint-functions` lines and guards unchanged. Fresh clone from `https://github.com/AMD-Ecosystem/h2o4gpu.git`, `moatlib.py protect-fork h2o4gpu` before the push; plain push `1e22b819..0e70429b` to `moat-port`. The 2c22a86 entry of "Commit-message corrections owed at the PR-prep squash" now carries the same wording.

Verification:
- `python3 utils/moatlib.py classify h2o4gpu 1e22b819 0e70429b`: `class=comment-only arch_independent=True inert=True` (`CMakeLists.txt: comment-only`).
- Configure once in each tree (new head in `projects/h2o4gpu/src`, `1e22b819` in a detached worktree in the job directory), no build:
  ```
  R=$($FLEET_VENV/bin/python -m rocm_sdk path --root); export PATH=$FLEET_VENV/bin:$R/bin:$PATH CMAKE_PREFIX_PATH=$R
  git submodule update --init tests/googletest
  cmake -S . -B build-hip -DUSE_CUDA=OFF -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$FLEET_VENV/bin/python -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  ```
  Both rc 0 (pip `swig==4.2.1` into the job venv; no OpenBLAS needed for configure). `compile_commands.json`, tree path normalized: 37 entries each, same file set, 0 differing entries; the `src/gpu/h2o4gpuglm.cu` line still ends `-fPIC -finline-hint-functions ... -x hip -c`.
- No GPU run: a comment-only change produces an identical build, so it needs none.
- `utils/prose.py` on the message: clean. `python3 utils/jargon.py --port h2o4gpu`: clean. Fork clone status after commit: only untracked `build-hip/`.

### Item 2: skill lesson
`.claude/skills/cuda-to-rocm/references/validation.md`: the Windows hang-diagnosis section keeps `AMD_LOG_LEVEL=3` / `PAL fence isn't ready!` and the symbol-build advice, then points to a new section "A hang that depends on inlining: long branches clobber the return address": names llvm/llvm-project#224205, the asm signature (in a non-kernel function, `s_getpc_b64 s[30:31]` ... `s_setpc_b64 s[30:31]` long branches with no save/restore of s30/s31, counted per function in `--cuda-device-only -S` output), present in Linux RDNA builds too and not always hanging, and both per-file spellings (`/Ob1` for clang-cl, `-finline-hint-functions` for clang/hipcc on Linux) through `COMPILE_OPTIONS` on the one source. The "reproducer still owed" sentence is gone. `utils/check.py` on this branch has no `skills` gate; `python3 utils/check.py jargon`: ok; `jargon.py -` on the added lines: clean.

### Item 3
"Commit-message corrections owed at the PR-prep squash" gains the `24cd051e` entry (`km_multi.py` not in the tree; give literal in-tree commands or an inline script) and the `1e22b819` entry (asm count and sklearn check only in prose; give literal commands).

### Owed at 0e70429b
- Review of `1e22b819..0e70429b` plus this round's answers.
- CUDA: the nvcc compile check at `1e22b819` passed ("CUDA compile check at 1e22b819"); `0e70429b` is comment-only inside `if(USE_HIP)`, so no new CUDA check is needed.
- Revalidation on linux-gfx90a, linux-gfx1100 and windows-gfx1151, plus the first linux-gfx942 validation, at the new head.

## Review 2026-11-12 round 2 (linux-gfx1201) -- moat-port 0e70429b (delta 1e22b819..0e70429b) plus records be4c652b and skill df72c7a5

### Port Review: h2o4gpu (moat-port `0e70429b` vs `1e22b819`)

#### Summary
One comment-only fork commit (0e70429b) restates the GLM inlining comment to the observed evidence, and the MOAT records and cuda-to-rocm lesson answer Review 2026-11-12 items 2 and 3. Verdict: **Approve** (no findings).

### Checked with no finding
- Item 1: `CMakeLists.txt:343-351` at `0e70429b` says the GLM reduction (float and double) compiles into a large device function whose long branches clobber the return address, cites llvm/llvm-project#224205 (verified open, title "[AMDGPU] Long-branch reserved SGPR pair is the return address s[30:31] in a callable function ...", read-only `gh issue view`), the double-precision fit was observed to hang on gfx1151 (Windows), the pattern is present in Linux RDNA builds without an observed hang, and limiting inlining removes the long branches. Matches the asm records ("Porter 2026-11-12" and round 2: `FuncEvalF<double>` 237/235 and `<float>` 241/434 s[30:31] long branches on gfx1100/gfx1201, 0 after limiting inlining; gfx90a none through s[30:31], so "Linux RDNA" is the right scope). `/Ob1` and `-finline-hint-functions` lines and guards unchanged. `python3 utils/moatlib.py classify h2o4gpu 1e22b819 0e70429b`: `class=comment-only arch_independent=True inert=True`. Commit title 63 chars with `[ROCm]`, AI disclosure in prose, no Co-Authored-By, fenced Test Plan.
- Item 2: `.claude/skills/cuda-to-rocm/references/validation.md:126-132`: the Windows hang-diagnosis section keeps `AMD_LOG_LEVEL=3` / `PAL fence isn't ready!` and now forwards to a new, OS-neutral section "A hang that depends on inlining: long branches clobber the return address" that names #224205, gives the asm signature (non-kernel callee of `s_swappc_b64 s[30:31]`, `s_getpc_b64 s[30:31]` ... `s_setpc_b64 s[30:31]` with no save/restore of s30/s31, from `--cuda-device-only -S`), says it appears in Linux RDNA builds and does not always hang, and gives both per-file spellings (`/Ob1` clang-cl, `-finline-hint-functions` clang/hipcc) via `COMPILE_OPTIONS`. The stale "reproducer still owed" sentence is gone. `python3 utils/check.py jargon`: ok; `jargon.py -` on lines 126-132: clean. No MOAT tool, record or state named.
- Item 3: "Commit-message corrections owed at the PR-prep squash" (notes.md:624) has entries for `2c22a86` (wording identical to the 0e70429b comment), `d0b78f3d` (superseded, collapses into the #224205 explanation), `24cd051e` (`km_multi.py` not in tree; literal in-tree or inline script) and `1e22b819` (asm count and sklearn check owed as literal commands).
- `python3 utils/jargon.py --port h2o4gpu`: clean (fresh clone of `https://github.com/AMD-Ecosystem/h2o4gpu.git` at `0e70429b`, fork `master` fetched). Clone status clean. `utils/check.py commits` flags only `24e6169`/`4e5e433` (pre-existing, covered by the 2026-11-05 squash ruling).

### Not blocking (record only)
- Owed, known: revalidation on linux-gfx90a, linux-gfx1100, windows-gfx1151 and the first linux-gfx942 validation at `0e70429b`. The CUDA compile check at `1e22b819` passed and covers this comment-only delta inside `if(USE_HIP)`.
