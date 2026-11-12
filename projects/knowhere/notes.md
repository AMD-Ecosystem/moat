# knowhere notes

## Intake screen (2026-11-12, windows-gfx1151 host, intake agent)

Claimed on jeff's ruling of 2026-11-12: knowhere is the port; milvus-io/milvus is NOT claimed or scaffolded and is recorded below as the downstream validation target. jeff will check with ROCm-DS before any porting starts, so MOAT builds on hipVS rather than competing with it.

Screened at zilliztech/knowhere `main` (shallow clone, 2026-11-12). Upstream is active (pushed 2026-11-12), 393 stars, not archived.

### Licence

- Top level: Apache-2.0 (`utils/licenses.py check` -> tier 1, cleared to contribute).
- Vendored: `thirdparty/faiss` MIT (Meta), `thirdparty/DiskANN` MIT (Microsoft), `thirdparty/hnswlib` Apache-2.0. All tier 1. No submodules.
- `utils/licenses.py scan-nvidia` on the clone: no NVIDIA proprietary licence text.
- Cardinal (`WITH_CARDINAL`, Zilliz's alternative CPU/GPU engine) is fetched from `github.com/zilliztech/cardinal`, which is private (404 to us). It is closed source and out of scope; the port covers the open cuVS backend only (`WITH_GPU=True` without Cardinal selects cuVS).
- hipVS and hipRaft (the ROCm dependencies) are Apache-2.0.

### CUDA surface (actual, not the headline count)

79 `.cu` files in the tree: 75 under `thirdparty/faiss/faiss/gpu` (vendored Faiss GPU, NOT compiled, see below) and 4 in knowhere proper (`src/common/cuvs/integration/{brute_force,cagra,ivf_flat,ivf_pq}_index.cu`, ~20 lines each, instantiating templates from the 823-line `cuvs_knowhere_index.cuh`). No `__global__` kernels and no `<<<>>>` launches anywhere in `src/`. The knowhere-owned GPU code is roughly 1.7 kLOC of cuVS/RAFT glue.

## hipVS and Faiss assessment

### 1. hipVS coverage of knowhere's GPU indexes

hipVS repo: `AMD-Ecosystem/hipVS` (`ROCm-DS/hipVS` redirects there), Apache-2.0, default branch `release/rocmds-26.03`, VERSION 1.0.0 (CHANGELOG dated 2026-04-01), ROCm 7.2.3, tracks upstream cuVS `branch-25.10` (`.upstream/RAPIDS_BRANCH`). RAFT comes from `AMD-Ecosystem/hipRaft` (`ROCm-DS/hipRaft`) at `release/rocmds-26.03`. Header layout and namespaces (`cuvs::neighbors::...`, `raft::...`) match cuVS, so knowhere's includes resolve unchanged. CHANGELOG: gfx950 support, CK pairwise distance, wave32 "experimental", multi-GPU experimental.

knowhere pins cuVS/RAFT `v26.02.00` (`cmake/libs/libcuvs.cmake`, `librapids.cmake`; bumped from 25.x in knowhere #1557, 2026-04-06, "for C++20 compatibility") and CUDA 13.2. So knowhere is one-plus RAPIDS releases ahead of hipVS (26.02 vs 25.10).

| knowhere index / use | cuVS API used | hipVS 25.10 base | gap |
|---|---|---|---|
| GPU_CUVS_BRUTE_FORCE (Milvus GPU_BRUTE_FORCE) | brute_force build/search/serialize | present | none found |
| GPU_CUVS_IVF_FLAT (GPU_IVF_FLAT) | ivf_flat build/extend/search/serialize, bitset filter | present | none found; community fork validated on gfx1100 |
| GPU_CUVS_IVF_PQ (GPU_IVF_PQ) | ivf_pq build/extend/search, codebook_gen, refine | present | none found; community fork validated on gfx1100 |
| GPU_CUVS_CAGRA (GPU_CAGRA) | cagra build/search/serialize, serialize_to_hnswlib, search_algo, hash_mode, graph_build_params {ivf_pq, nn_descent, iterative_search, **ace**} | present except ACE | `graph_build_params::ace_params` does not exist in hipVS (added upstream after 25.10); knowhere names it unconditionally in a `std::variant`. Filtered CAGRA search (bitset) has the API in hipVS, but the community port reports it unsupported at runtime on gfx1100; needs verification |
| DiskANN GPU build (`thirdparty/DiskANN/src/diskann_gpu.cpp`, knowhere #1460) | vamana::build, cluster::kmeans fit/predict incl. balanced_params | vamana and kmeans (balanced_params) present | none found at API level |
| Common | cuvs::core::bitset, filtering::bitset_filter, refine, raft device_resources_manager, raft linalg/matrix, thrust | present | none found |

API gaps to close: (a) ACE graph build -- either hipVS moves to a 26.x base or the port compiles ACE out under HIP; (b) any 25.10->26.02 signature drift, unknown until a compile; (c) filtered CAGRA correctness on AMD. Multi-GPU (`BUILD_MG_ALGOS`) is OFF in knowhere, so the experimental hipVS MG path does not matter.

### 2. Vendored Faiss GPU: reuse verdict

Moot for knowhere's shipped GPU path. The vendored Faiss is v1.15.1 (2026-09-15, current) plus knowhere additions under `faiss/cppcontrib/knowhere/`; it already carries upstream Faiss's ROCm support (`faiss/gpu/hipify.sh`, 39 `USE_AMD_ROCM` sites). But `cmake/libs/libfaiss.cmake` globs only CPU sources and never adds `faiss/gpu`, and the top-level CMakeLists unconditionally removes `src/index/gpu/{flat_gpu,ivf_gpu}` (the old Faiss-GPU index wrappers, plus RMM in `gpu_res_mgr.h`) from the build. knowhere's GPU path is cuVS only. Nothing to port in the Faiss GPU tree; leave it untouched. MOAT's validated upstream Faiss ROCm work is relevant only as proof the vendored copy would hipify if Zilliz ever re-enabled it.

### 3. Existing AMD / community work

- **ROCm-DS does not appear to own knowhere or Milvus.** No knowhere or milvus repository under ROCm, ROCm-DS or AMD-Ecosystem; hipVS's own docs mention Milvus only in the inherited upstream `.upstream/docs/source/integrations/milvus.rst`. AMD blogs cover hipVS/hipRAFT and a hipVS RAG demo, not Milvus.
- No ROCm/HIP/AMD issues, PRs or branches on zilliztech/knowhere; on milvus-io/milvus only #5654 "ROCM support" (2021, closed).
- **Community port exists:** `konkolchin/knowhere-hip` (branch `2.5`, now served as krainov/knowhere-hip) and `konkolchin/milvus-hip` (Milvus v2.5.4), with harness `konkolchin/ann-harness-amd`. Created 2026-07/08, last pushed 2026-10-21. Built on hipVS/hipRaft `release/rocmds-25.10` plus a gfx1100 IVF packer patch, opt-in `WITH_HIP`. Claims: GPU_IVF_FLAT and GPU_IVF_PQ validated on RX 7900 XTX (gfx1100); GPU_CAGRA unfiltered only (NN_DESCENT build; bitset filtered search unsupported); Instinct untested. One commit references a merged PR from an `llmkb-internal` org ("AMD/DXC port"), so it may be contractor work for AMD; it is not upstream-facing and targets the old knowhere 2.5 line (pre-26.02). Treat as a hint, not code to inherit; jeff may want to ask ROCm-DS whether they know of it.
- Small third-party repos (`ieuies/amd-bench`, `xyandys/amd-cagra-bench`) reference a hipVS "AMD challenge"/competition; benchmark helpers only.

### 4. CUDA surface outside cuVS/Faiss

- thrust in `cuvs_knowhere_index.cuh` (device_ptr, fill, for_each, partition, zip/counting iterators) -> rocThrust.
- `cuda_fp16.h` half types -> `hip_fp16.h`.
- `cuda_runtime_api.h` -> HIP runtime.
- Link line: CUDA::cublas, cusparse, cusolver -> hipBLAS, hipSPARSE, hipSOLVER (as hipVS itself links).
- RMM: only in the dead `src/index/gpu/gpu_res_mgr.h`; live code uses raft resources. hipVS ships its RMM equivalent through hipRaft.
- Build system: `cmake/utils/cuda.cmake` force-installs/forces CUDA 13.2 at `/usr/local/cuda-13.2`; `librapids.cmake` downloads rapids-cmake 26.02 and uses `rapids_cuda_init_architectures`. The port needs a HIP branch here (find hipVS/hipRaft packages instead of CPM-fetching rapidsai/cuvs). This is the main build work.
- No custom kernels, so the warp-size fault classes apply only inside hipVS, not knowhere.

### 5. GPU gate and Milvus-level validation

- Gate: knowhere's Catch2 UT built with `make WITH_GPU=True WITH_UT=True`; GPU cases in `tests/ut/test_gpu_search.cc` ("Test All GPU Index", "Test GPU cuVS indexes pad filtered results", CAGRA serialization tests), plus the full CPU UT suite to show no regression. Filtered-search cases cover the bitset path that the community port could not run on CAGRA.
- Milvus downstream target (not claimed): milvus-io/milvus (Apache-2.0) pins knowhere by commit (`internal/core/thirdparty/knowhere/CMakeLists.txt`, `KNOWHERE_VERSION`). After the knowhere port, build Milvus GPU against the ported knowhere and run the GPU index cases in `tests/python_client/testcases/test_index.py` / `test_query.py` (index types GPU_IVF_FLAT, GPU_IVF_PQ, GPU_CAGRA, GPU_BRUTE_FORCE from `common/common_params.py`), the Go `client/index/gpu_test.go`, and a recall/QPS check with a standard ANN dataset.

### 6. Windows viability

Upstream has no Windows build: README targets Linux (and macOS CPU), the GPU path installs CUDA only on Ubuntu, dependencies include folly and opentelemetry via conan, and hipVS/hipRaft have no Windows support. Milvus server is Linux-only. Suggest a `windows` gate waiver (only jeff can approve).

### Recommendation

Fork. Small knowhere-owned GPU surface (4 `.cu`, ~1.7 kLOC glue, no kernels), all four cuVS indexes plus the DiskANN GPU build covered by hipVS at the API level, with the ACE build mode and the 25.10 vs 26.02 version skew as the known gaps. Depends on hipVS/hipRaft (not MOAT projects; no `set-deps`). Hold porting until jeff's ROCm-DS conversation settles whether hipVS moves to a 26.x base.
