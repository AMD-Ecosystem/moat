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

## Planner (2026-11-20, planner agent for jeff, linux-gfx942; analysis only, no GPU used)

### Rulings needed from jeff before a porter starts

**Q1. Runtime backend for ROCm.** Upstream v0.9.0 compiles every kernel at run time with NVRTC through `cuda.core`; the torch-extension backend is gone. Options:
- (a) **Recommended:** a hipRTC + HIP-module launch path implementing only the slice of `cuda.core` curobo uses (compile/get_kernel, LaunchConfig, launch, stream from the torch handle, last error, shared-memory attribute), selected when `torch.version.hip` is set, bound with **ctypes to the HIP runtime and hipRTC that torch already loaded**. No new dependency, one HIP runtime per process, works with pip-SDK ROCm on Linux and Windows. Matches upstream's design; CUDA path unchanged.
- (b) Same as (a) but bound through AMD's `rocm-bindings-hip` (`hip-python` 10.0) wheel. Official bindings; adds a dependency whose version must track the ROCm major torch was built with.
- (c) Reinstate a torch `CUDAExtension` for ROCm only (v0.8.0 style). Not recommended: kernels are now template-specialised at run time on robot size, so an ahead-of-time build needs a fixed instantiation list or a dispatch rewrite, and it diverges from upstream.

**Q2. Scope: is `cpp_runtime/` (C++ robot/calibration/mapper runtime, ~35k lines, 26 CTest targets, plus the Python exporters that write cubins and Warp AOT modules for it) in the first port?** Options:
- (a) **Recommended:** phase 1 ports the Python package only (the library users install, with its full pytest suite and examples); `cpp_runtime/` and its exporters follow in a second round after phase 1 validates and AMD Warp's AOT support for code objects is known. `surface.json` lists the cpp_runtime components; under (a) they would be recorded `owed` (not `scoped_out`) so `pr_ready` refuses until they are done, or `scoped_out` if you rule the cut is permanent for this delivery.
- (b) Port both now. Adds Strategy A CMake work, hipCUB for the mapper, HIP graph-node APIs, a code-object export path, and a hard dependency on Warp AOT code-object export, likely several porter rounds.

**Q3. Warp dependency.** curobo needs NVIDIA Warp (tile API, Mesh BVH, graph capture). AMD Warp is 1.13.0+amd0 on `amd-integration` and 1.14.0 on `release/1.14.0.amd0`, which MOAT is porting as `warp-1.14` (stage `porting`, no platform validated). curobo's notices assess warp-lang 1.16.0; it pins nothing. Options:
- (a) Record `depends_on: [warp-1.14]` now (`moatlib.py set-deps`): the selector holds the curobo porter until warp-1.14 completes on this OS. Cleanest evidence chain; delays the port.
- (b) **Recommended:** no hard dependency yet; the porter builds AMD Warp from `amd-integration` (or the 1.14 line's current head) in the job venv as environment, records the exact Warp commit with each validation, and registers any Warp defect against warp-1.14. Revisit (a) at PR time.
- (c) Wait for an AMD Warp at >= 1.16 before porting.

Until these are ruled, the plan in `plan.md` is written for Q1(a), Q2(a), Q3(b).

### Summary of the plan

- Build class: runtime-compiled kernels from Python (`ext_type` = `python-nvrtc-runtime`); neither CMake Strategy A nor torch hipify applies to the package. `cpp_runtime/` is Strategy A if in scope.
- Upstream analysed at `0f04ee18066822ec0c6751fa085d7b033734c4da` (2026-11-19).
- Device strategy: keep curobo's logical 32-lane warp on every AMD target (host launch geometry, shared scratch and kernels all assume 32), via a new `kernels/common/warp_ops.cuh` of project-owned wrappers: native `*_sync` on CUDA, width-pinned `__shfl*(..., 32)` and half-wave 32-bit ballot/activemask/match_any masks on HIP. About 120 call sites in 24 headers switch to it.
- Other device fixes: `helper_math.h` operator block guard, `<cuda_runtime.h>` guard in the pose refiner, an fp32 path for the test-only `nvcuda::wmma` tf32 compact L-BFGS kernel.
- No textures, PTX, cooperative groups, CUB or Thrust on the Python path.
- surface.json: generated floor (62 CUDA files, 29 cpp_runtime components) plus 24 planner-added components covering the runtime-compile backend, the kernel groups, Warp, the pytest suite, examples, benchmarks, packaging and docs. `utils/surface.py check curobo`: fully accounted for.

### Facts checked during planning (for the porter)

- `cuda.core` usage is confined to `curobo/_src/curobolib/backends/cuda_core_backend/*` plus `curobo/_src/trajectory/timeopt/timeopt_solver.py` (imports `LaunchConfig` and `launch_kernel` directly). Other modules import backend modules, not cuda.core.
- `torch.cuda._compile_kernel` exists on ROCm torch 2.16 and uses hipRTC, but packs Python ints as C `int` (truncates `data_ptr()` pointers) and is private: not usable as the launch path.
- `hip-python` 10.0.0.0.1.2 on PyPI is a compatibility shim over `rocm-bindings-hip`/`rocm-bindings-core` 10.0 (low-level `hip`, `hiprtc` only; no cuda.core equivalent). `hip-python-as-cuda` is not on PyPI (only test.pypi.org, latest 7.2.2.562.43); that wheel maps `cuda.bindings.{driver,runtime,nvrtc}` and the old `cuda.cuda`/`cudart`/`nvrtc` onto HIP but has no `cuda.core`, so it would cover only curobo's three `cuda.bindings.runtime` calls.
- All 35 device `warpSize` uses are `curobo::common::warpSize` (= 32), not HIP's built-in.
- `lbfgs_compact_step_kernel.cuh` (wmma tf32) is reachable only from `LBFGSCompactCu` in `curobo/tests/_src/optim/gradient/test_lbfgs.py`; `lbfgs.py:102` says tensor-core compact is no longer dispatched.
- No `wp.Texture`/`wp.Volume` use, so AMD Warp's texture gap does not apply.
- `.github/workflows` has only docs deploy and commit signing; no GPU CI to mirror.
