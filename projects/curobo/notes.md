# curobo notes

## Intake screen (2026-11-05, intake agent for jeff; seed: AMD AI Software Gap Matrix)

Screened upstream `NVlabs/curobo` main at the shallow clone of 2026-11-05 (tag v0.9.0, "cuRoboV2-Fast", committed 2026-11-05T02:26Z). Matrix lead: verdict not-yet, AMD path none. The screen agrees: no authoritative AMD path exists.

### Recommendation: take up (fork). The deliverable is the AMD-Ecosystem fork.

NVIDIA-owned (NVlabs), so an upstream PR is unlikely to be accepted (CONTRIBUTING.md does accept DCO-signed contributions, so a PR is not impossible, but the expectation is fork-as-deliverable). The licence bar for NVIDIA-owned repos is met file by file (below).

### Licence

- Top level: Apache-2.0 (`licenses.py check NVlabs/curobo` -> tier 1). Evidence: https://github.com/NVlabs/curobo/blob/main/LICENSE
- History: the LICENSE file has three revisions. `07e6ccfc91` (2023-10-26, "release repository") was the restrictive "NVIDIA License" (NVIDIA Source Code License style, non-commercial). `457b658dbb` (2026-04-18, "curobov2 initial release", tag v0.8.0) replaced it with Apache-2.0. `76c2f4b6d3` (2026-11-05, v0.9.0) only added a copyright line and the Apache appendix. So everything from v0.8.0 on is Apache-2.0; v0.7.x and earlier (`src/curobo/` layout) remain under the old non-commercial licence and must never be used as a port base or source of code.
- Per-file headers on main: 917 SPDX `Apache-2.0` headers; one `BSD-3-Clause` (`curobo/_src/curobolib/kernels/third_party/helper_math.h`, CUDA Samples, BSD notice retained) and one `BSD-2-Clause` (`docs/Makefile`, Sphinx). No kernel or library file carries a proprietary header. Only two files lack an SPDX header, and both are empty `__init__.py` files under `curobo/tests/`. No submodules (`.gitmodules` absent); the only vendored directory with third-party code is `kernels/third_party/` (helper_math.h, BSD-3).
- `python3 utils/licenses.py scan-nvidia agent_space/curobo-screen` flagged 3 files; each read in full and cleared:
  - `THIRD_PARTY_NOTICES.md`: dependency notices, not cuRobo code. The proprietary text is the reproduced NVIDIA Software License of `cuda-bindings` 13.0.3 (an optional `cu13`/`cu13-torch` extra, not bundled; the file itself notes cuda-bindings >= 13.4.0 is Apache-2.0), plus table rows naming RADIO (NVIDIA Source Code License, non-commercial), the C-RADIO weights (NVIDIA Open Model License) and two CC BY-NC-SA datasets, all listed under "Example-only integrations and downloaded materials ... not required by cuRobo's implementation ... not bundled".
  - `curobo/examples/getting_started/feature_mapping.py` and `curobo/examples/reference/lidar_volumetric_mapping.py`: both are Apache-2.0 files whose docstrings warn that `torch.hub` downloads RADIO source (non-commercial) and weights at run time. Nothing is vendored.
  - RADIO is not needed by the core library: `curobo/_src` never imports RADIO or calls `torch.hub`; the three `perception/mapper` hits are docstrings that use "RADIO -> SigLIP" as an example of a feature space. A port should leave those two examples untouched and not use them as validation tests.
- The old matrix concern about nvblox (proprietary headers) is stale: v0.8.0+ ships its own Warp/CUDA mapper. Remaining "nvblox" strings are a CHANGELOG history entry, a docstring ("nvblox-style" weighting) and a comment. No nvblox import or dependency exists.
- Isaac Sim is optional: one loader file mentions it; no runtime dependency.
- Scope note: this licence finding is for contribution and fork distribution. Optional runtime extras (`cuda-bindings` < 13.4, `nvidia-cuda-runtime*` under the CUDA EULA) are CUDA-path-only and would not be part of a ROCm install.

### Duplicate effort

- Upstream docs: `grep -rniE 'amd|rocm|hip|gfx' README* docs/` finds only binary video files. No AMD mention.
- Upstream PRs (`gh pr list --state all --search "hip OR rocm OR amd"`): none. Branches matching hip/rocm/amd: none. Issues: none relevant.
- `ROCm/curobo` and `AMD-Ecosystem/curobo`: do not exist (404). No fork of NVlabs/curobo owned by a ROCm/AMD/EmbeddedLLM org.
- Community effort found: https://github.com/ZiguanWang/curobo_rocm_wrapper (created 2026-10-23, 0 stars, personal account, Chinese docs). A patch-and-submodule wrapper over upstream v0.8.0 (Apache, pybind-extension path) and v0.7.8 (old non-commercial licence), validated only on gfx1201 (wave32): "15 passed" on seven migration test files. Judgment: NON-authoritative community fork. Do not inherit its patches; at most a hint (it confirms wave masks and wave32 compile issues, and uses AMD-Ecosystem/warp). Its v0.7.8 profile is also licence-tainted.
- The `amd` and `AMD-AGI` orgs are behind SAML SSO for this token, so a duplicate check inside those two orgs is still open.

### Viability: yes, substantial work

- CUDA surface: 52 `.cu`/`.cuh` files, about 652 KB. Kernels in `curobo/_src/curobolib/kernels/` (kinematics, dynamics RNEA/CRBA, self-collision, L-BFGS, line search, B-spline and time-optimal trajectory, parallel banding) plus a C++ runtime in `cpp_runtime/` (CMake, `.cu` launchers, mapper using CUB DeviceScan/DeviceReduce).
- Build model changed under us: on main (v0.9.0, today) the Python package compiles kernels at run time ONLY through `cuda.core` (NVRTC, `Program(...).compile("cubin")`, `cuda.pathfinder` to find NVRTC headers; `curobo/_src/curobolib/backends/cuda_core_backend/`). v0.8.0 also had an optional `pybind` torch `CUDAExtension` backend via `setup.py`; v0.9.0 deleted `setup.py` and the pybind backend. A ROCm port therefore needs a HIP runtime-compilation backend (hiprtc, for example through ROCm `hip-python`) alongside the cuda.core one, or a reinstated torch-extension backend; the planner has to pick. This is the main design question.
- Wave-size hazards are pervasive: 37 `warpSize` uses, a `fullMask = 0xffffffffu` constant, `0xffffffff` masks on `__shfl*_sync` / `__ballot_sync`, `threadIdx.x & 31`, `/ 32` and `% 32` warp indexing in block reductions, `constexpr int warp_size = 32` in `bspline_kernel.cuh`, `__popc`/`__ffs` on 32-bit ballot results (peer election in `rnea_helpers.cuh`, `self_collision_helper.cuh`). wave64 (gfx90a/gfx942) needs real work; wave32 RDNA is easier.
- Hard dependency on NVIDIA Warp (`warp-lang`; 80 files under `curobo/_src` import it, and `cpp_runtime/robot/src/warp_abi.h` links Warp's ABI). Not a MOAT project (`moatlib.py show warp`: no record). The AMD path is https://github.com/AMD-Ecosystem/warp (fork of NVIDIA/warp, default branch `amd-integration`, pushed 2026-11-05). Not recorded with `set-deps` because it is not a MOAT project; the planner must confirm which Warp revision builds on each gate.
- Other deps are pure Python (numpy, scipy, trimesh, viser, yourdfpy, ...) plus torch. Optional: usd-core, Isaac Sim examples.
- Upstream is very active (pushed 2026-11-05, ~2k stars); expect sync rounds.
- Suggested scope for the planner to weigh: phase 1 is the Python package (HIP runtime-compile backend + wave-size fixes + packaging extras for ROCm), validated with upstream's GPU pytest suite; `cpp_runtime/` (CMake, CUB -> hipCUB) as a possible second phase. Whether to cut `cpp_runtime/` is jeff's call, per the partial-port rule.

### Question for jeff

The port base moved today: upstream v0.9.0 (2026-11-05) removed the torch-extension (pybind/setup.py) backend and now builds kernels only by NVRTC runtime compilation through `cuda.core`. Should the port add a hiprtc runtime-compilation backend that mirrors cuda.core (matches upstream's design, more new code), or reinstate a torch `CUDAExtension`-style compiled backend for ROCm only (closer to the v0.8.0 path the community wrapper used, but diverges from upstream)? And is `cpp_runtime/` (C++ deployment runtime and native mapper) in scope for the first port, or deferred?
