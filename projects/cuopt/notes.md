# cuopt notes

## Intake screen (2026-11-05)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict not-yet, AMD path none). Screened at upstream `1af75d9e60a6137813f40a175157fe683f06ea1b` (VERSION 26.12.00, RAPIDS_BRANCH main), shallow clone in `agent_space/cuopt-screen`.

### Licence

- Top level `LICENSE`: Apache-2.0 (`utils/licenses.py check NVIDIA/cuopt` -> tier 1). Recorded with `set-license-spdx`.
- Every per-package `python/*/LICENSE` is Apache-2.0.
- `python3 utils/licenses.py scan-nvidia agent_space/cuopt-screen`: no NVIDIA proprietary licence text anywhere in the tree. No submodules (`.gitmodules` absent).
- `thirdparty/THIRD_PARTY_LICENSES` lists cxxopts (vendored, MIT), PSLP and papilo (Apache-2.0, fetched), bzip2, zlib, KaMinPar, LZ4, SIMDe, highway: all permissive.
- File-by-file bar for an NVIDIA-owned repo is met. The licence does not block anything.

### Duplicate effort

- No ROCm/HIP/AMD mention in upstream README or docs (the only "AMD" hits are the Approximate Minimum Degree ordering and a Threadripper CPU in a benchmark).
- Upstream PRs and issues searched for hip/rocm/amd: nothing relevant (hits are cuDSS PRs).
- `gh search repos cuopt --owner ROCm` and `--owner AMD-Ecosystem`: nothing. No AMD port of cuOpt found by web search either.
- The `amd` and `AMD-AGI` orgs are behind SAML SSO for this token and were not searched; that check stays open.
- cuOpt is not one of the ROCm-DS components (hipDF, hipMM, hipRAFT, hipVS, hipGRAPH per the ROCm-DS 25.10/26.03 release notes), but it is built on the RAPIDS stack and released on the RAPIDS train (RAPIDS_BRANCH, rapids-cmake, version 26.12). MOAT's dispositions say "RAPIDS: the ROCm-DS team owns this domain" for raft, rmm, cudf, cuml, cugraph, cuvs, cucim, cuCollections.

### What it is and the CUDA surface

GPU LP (PDLP, barrier), MILP heuristics and vehicle routing solver. `cpp/`: 200 `.cu` + 189 `.cuh`, about 6.4 MB. Python layer (Cython) plus a server, MCP and Java bindings.

Library use (files mentioning each, cpp/): cuSPARSE 58, cuBLAS 24, Thrust 194, CUB 79, `cuda::std` 36, NVTX 108, cooperative_groups 4; no cuSOLVER, cuRAND, CUTLASS or inline asm. The mappings hipSPARSE / hipBLAS / rocThrust / hipCUB / libhipcxx / roctx exist. Warp-size hazards look modest: about 18 hits on `0xffffffff` masks, `*_sync` shuffles/ballots and `warp_size = 32` constants across 10 files, mostly in `src/routing/` (route.cuh, ges/*), plus `mip_heuristics/feasibility_jump/load_balancing.cuh` and `presolve/load_balanced_bounds_presolve_kernels.cuh`.

### Hard dependencies

From `cpp/CMakeLists.txt` and `dependencies.yaml`:

| Dependency | Use | ROCm side |
|---|---|---|
| RAFT (`raft::raft`, 35 distinct headers: core, util, sparse, linalg, random) | required | `AMD-Ecosystem/hipRaft` (ROCm-DS, release/rocmds-26.03). Its CHANGELOG says the upstream RAFT baseline is **25.10**; cuOpt sets `CUOPT_MIN_VERSION_raft` to its own RAPIDS version, **26.12**. A year of RAFT API drift to bridge or wait out. |
| RMM (`rmm::rmm`) | required | `AMD-Ecosystem/hipMM` 4.0.0 (ROCm-DS 26.03; same baseline question) |
| CCCL (libcudacxx/Thrust/CUB) | required | libhipcxx 3.0.2 + rocThrust + hipCUB (via ROCm-DS) |
| **cuDSS** (`find_package(CUDSS REQUIRED)`) | required; sparse Cholesky/LDLT for the barrier solver (`src/barrier/sparse_cholesky.cuh`, 976 lines) | **No ROCm equivalent found.** Closed NVIDIA library. Barrier is one of the three methods the default `Concurrent` LP mode runs, so it cannot simply be dropped without a scope ruling. |
| NCCL | multi-GPU PDLP | RCCL |
| cuDF (Python layer: routing data model, wrappers, tests) | required by the Python package, not by libcuopt | hipDF (ROCm-DS) |
| cupy, numba-cuda | Python layer | cupy has a ROCm build; numba-cuda has no ROCm backend (planner to check how much is used) |

`depends_on` is set to `raft`, `rmm`. Both are dispositioned `ported-elsewhere` (ROCm-DS), so the selector treats them as satisfied; that is accurate for existence but hides the 25.10 vs 26.12 baseline gap, which is the real blocker. cuDSS is not a MOAT project and is recorded here only.

Platform: cuOpt and ROCm-DS are both Linux-only, so the `windows` gate would need a waiver from a person.

### Recommendation

Decline for now, reason `other` (dependency-blocked). The licence is clean and the deliverable would be the fork (an NVIDIA-owned upstream will not take a HIP PR), but the port cannot be built today without either (a) porting cuOpt's RAFT/RMM usage back to a RAFT 25.10 API or upgrading hipRaft/hipMM to 26.x, which is ROCm-DS team territory, and (b) replacing cuDSS with something that does not exist on ROCm yet, or a person ruling that the barrier solver is deferred on HIP. Revisit when ROCm-DS ships a RAPIDS 26.x baseline. If a person prefers to take it up anyway, the shape is: PDLP + dual simplex + MILP heuristics + routing on hipRaft/hipMM, barrier deferred (deferred.py, person rules), Linux gates only with a windows waiver.

### Questions for jeff

1. cuOpt sits on the RAPIDS train and depends on RAFT/RMM at 26.12, while ROCm-DS's hipRaft is at a RAFT 25.10 baseline. Should this go to the ROCm-DS team (is cuOpt on their roadmap, and when do hipRaft/hipMM move to 26.x), rather than MOAT porting cuOpt against an older hipRaft?
2. cuDSS has no ROCm equivalent. If cuOpt is taken up, may the barrier LP method (and its cuDSS dependency) be deferred on HIP, leaving PDLP and dual simplex in Concurrent mode? That is a defer-vs-now ruling for a person.
3. The `amd` and `AMD-AGI` orgs could not be searched (SAML SSO on this token), so an internal AMD cuOpt effort cannot be ruled out.
