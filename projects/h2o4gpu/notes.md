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
