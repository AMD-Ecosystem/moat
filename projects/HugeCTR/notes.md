# HugeCTR notes

## Intake screen (2026-11-05, windows-gfx1151 host, intake role)

Source: third-party "AMD AI Software Gap Matrix" seed, entry "NVIDIA Merlin (NVTabular, HugeCTR)", verdict not-yet, AMD path none. Upstream: https://github.com/NVIDIA-Merlin/HugeCTR (default branch `main`, screened at the shallow-clone tip of 2026-11-05, last commit 61b4d539 dated 2026-08-03).

### Licence

- `python3 utils/licenses.py check NVIDIA-Merlin/HugeCTR`: Apache-2.0, tier 1.
- `python3 utils/licenses.py scan-nvidia agent_space/HugeCTR-screen` (shallow clone, submodules not populated): "no NVIDIA proprietary licence text".
- Vendored trees inside the main repo: `HugeCTR/include/hashtable/cudf/LICENSE`, `sparse_operation_kit/LICENSE.txt`, `sparse_operation_kit/.../dynamic_embedding_table/{cuCollections,cudf}/LICENSE` are all Apache-2.0. `third_party/dynamic_embedding_table/cuCollections` has no LICENSE file, but every file in it carries an Apache-2.0 header (`grep -rL 'Apache License'` returns nothing).
- Submodules at their pinned commits (checked through the GitHub API):
  - googletest f8d7d77c: BSD-3-Clause
  - nlohmann/json 35daa5c0: MIT
  - rapidsai/cuml 4323a5d3 (2020-08): Apache-2.0, no nested `.gitmodules`
  - NVIDIA/cutlass cf0301e0: CUTLASS 1.0.1 (2018-06); `LICENSE.TXT` is the 3-clause BSD text (GitHub reports NOASSERTION because of the file name); nested googletest only
  - aminnj/cpptqdm 04c733fd: MIT (LICENSE read)
  - greg7mdp/parallel-hashmap 7684faf1: Apache-2.0
  - p-ranav/argparse 997da925: MIT
  - NVIDIA-Merlin/HierarchicalKV 28046ba0: Apache-2.0; nested googletest only
- Verdict: clean file by file. `license_spdx` = Apache-2.0. NVIDIA-owned, so an upstream PR is unlikely; the fork would be the deliverable.

### Duplicate effort

- Upstream PRs/branches/issues: `gh pr list --state all --search "hip OR rocm OR amd"` is empty; there are no hip/rocm/amd branches; the only "AMD" issue hits are unrelated (#344 CPU-only, #199 non-DGX build). The README/docs make no mention of AMD or ROCm.
- `gh search repos HugeCTR` under ROCm and AMD-Ecosystem, and "HugeCTR rocm" globally, found nothing. The web search found no HugeCTR port anywhere. The `amd` and `AMD-AGI` orgs sit behind SAML SSO for this token, so a private effort there cannot be ruled out. (AMD-AGI does publish the public DLRMBenchmark repo, see below.)
- HierarchicalKV (the hard submodule dependency) has Ascend ports (Ascend/HierarchicalKV-ascend) but no AMD port.
- The workload is covered on AMD by a different stack. AMD's own path for GPU recommender training is TorchRec + FBGEMM_GPU on ROCm:
  - ROCm/FBGEMM is an AMD-org fork; upstream FBGEMM_GPU publishes ROCm nightly wheels (`https://download.pytorch.org/whl/nightly/rocm7.1/`, per https://docs.pytorch.org/FBGEMM/fbgemm_gpu/development/InstallationInstructions.html).
  - AMD ROCm blog, 2026-03-02, "Streamlining Recommendation Model Training on AMD Instinct GPUs" (https://rocm.blogs.amd.com/artificial-intelligence/recsys-training-docker/README.html): the `rocm/primus:v26.1` training docker ships TorchRec and FBGEMM. It trains DLRMv2 on 8x MI300X with TorchRec table-wise, row-wise and column-wise embedding sharding, and its reference is AMD-AGI/DLRMBenchmark. HugeCTR is not mentioned.
  - The DLRM on AMD GPUs blog (https://rocm.blogs.amd.com/artificial-intelligence/dlrm/README.html) covers the same ground on MI210.
  - For HugeCTR's SOK (TensorFlow plugin) side, TensorFlow has a ROCm build, but nothing ports SOK itself.

### Maintenance status (stated plainly)

HugeCTR is in de facto maintenance mode, though nothing upstream uses that wording:
- Releases are now annual: v23.12, v24.04, v24.06, v25.03.00 (2025-03-14), v26.03.00 (2026-03-12).
- From 2025-03 to 2026-11 there are about 10 commits on GitHub, all mirrored from an internal GitLab: CUDA 13.0 / GB300 update (2025-11), a pybind pin, an "unsafe input parsing" security fix (2026-06), and ONNX loader validation (2026-08). No feature work.
- Subsystems have been deprecated one after another: offline inference (v23.09), the Triton backend (v23.08, removed from Triton), the Hierarchical Parameter Server (v24.06), and the embedding cache components (v25.03; users are pointed to HierarchicalKV).
- No issue opened since mid-2025 has a maintainer reply: #477 to #482 all have 0 comments. #481 asks why HPS was deprecated and #482 says the DLRM tutorial is "severely outdated" for current containers.
- The repo is not archived.

### Viability (cheap checks)

- It is real CUDA: about 2.57 MB of `.cu`/`.cuh`. That is HugeCTR/src 1.05 MB, HugeCTR/embedding 0.56 MB, test/utest 0.26 MB, sparse_operation_kit 0.22 MB, gpu_cache 0.12 MB, plus vendored cuCollections.
- Hard dependencies: CUDA driver API, cuBLAS + cuBLASLt (to hipBLAS/hipBLASLt), cuRAND, NVML (to amdsmi), cuDNN (to MIOpen), NCCL (to RCCL), CUB/Thrust (to hipCUB/rocThrust), and HierarchicalKV (no AMD port; it would be its own MOAT project and a `depends_on`). Optional dependencies: cuDF + RMM + Parquet for the Parquet reader (behind `DISABLE_CUDF`), and SHARP/UCX/MPI/HWLOC for multi-node. The vendored CUTLASS is the 2018 1.0.1 release.
- Fault-class hints: inline `asm`/`mma_sync`/`wmma` in 5 files under HugeCTR/, and about 19 sites of `__shfl_*_sync`, `__ballot_sync` or `0xffffffff` masks (wave64 work).
- Porting it is technically possible but heavy: a multi-library port of a large framework plus an unported HierarchicalKV, for software its owner no longer develops.

### Recommendation

Decline, with SKIP_REASON `already-supported`. Nothing ports HugeCTR itself to AMD, but its workload is already covered by AMD's own supported stack: TorchRec + FBGEMM_GPU on ROCm, shipped in the rocm/primus training docker, with an AMD-AGI DLRMv2 benchmark on MI300X. A HugeCTR fork would duplicate that path, and it would serve a framework NVIDIA has mostly stopped developing (annual releases, deprecated subsystems, unanswered issues). No upstream PR is likely, so the fork alone would have to justify about 2.5 MB of CUDA plus a HierarchicalKV port. If the matrix entry is to be answered at all, it is better answered by pointing at TorchRec/FBGEMM on ROCm. `not-a-target` is the fallback reason if a person reads "already-supported" as requiring a port of this exact repo.

Question for jeff: Does an AMD-supported alternative stack for the same workload (TorchRec + FBGEMM_GPU on ROCm) count as `already-supported` for HugeCTR, or should a decline here use `not-a-target` (maintenance-mode NVIDIA framework, no upstream destination)? If you would rather take it up, HierarchicalKV has to be screened first as a hard dependency.

## Take-up ruling (2026-11-11)

Jeff Daily ruled that HugeCTR needs a canonical home in AMD-Ecosystem, overriding intake's decline (already-supported: TorchRec + FBGEMM_GPU on ROCm cover the workload). The project is recorded as a take-up (`set-intake HugeCTR fork`) with `depends_on: HierarchicalKV`, its hard build dependency, which has no AMD port and is screened as its own MOAT project (claimed and in intake 2026-11-11; the Hygon DCU port is not to be used as a base). State `awaiting-fork`: a person creates AMD-Ecosystem/HugeCTR; the selector will not start it before HierarchicalKV is done.
