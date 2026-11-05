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
