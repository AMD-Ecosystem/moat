# libuipc notes

## Intake 2026-11-24 (linux-gfx942 host, no GPU used)

Screened at upstream `spiriMirror/libuipc` main `e06f8ee1b0ab7287cf1968ede4478c736a090370` (2026-11-24), shallow clone in `agent_space/libuipc-screen`. Nothing was built.

### Recommendation: fork

The upstream author asked for this port in public, the licence is Apache-2.0, no AMD effort exists anywhere, the CUDA backend is self-contained (CUB is its only CUDA library, no cuBLAS/cuSPARSE/cuSOLVER, no separate CUDA framework repository), and the GPU_IPC port already solved the hardest shared code. It is a large port (about 50k lines of `.cu`, 106k lines in the backend) but a mechanical one, with a real test suite to validate against.

### Upstream engagement (evidence)

Our HIP port of GPU_IPC was merged by its author on 2026-11-24 (https://github.com/KemengHuang/GPU_IPC/pull/4, merged 04:53Z by KemengHuang). One minute earlier he commented (https://github.com/KemengHuang/GPU_IPC/pull/4#issuecomment-7122416498):

> We have a more complete simulator, StiffGIPC, which is much faster and supports strong coupling between rigid and deformable bodies. Even better is our user-friendly version, libuipc. Especially libuipc is more popular, and I believe AMD support could greatly advance libuipc applications, since FP64 on AMD is more powerful than on NVIDIA, and our algorithm is almost entirely FP64-based. If you are interested in libuipc, we would also welcome your PR.

Kemeng Huang is the first-listed author and maintainer in `pyproject.toml` and the project lead named in `NOTICE`. This is an explicit invitation; the PR is expected and welcome.

### Which repository is upstream

`spiriMirror/libuipc` is the live upstream. `KemengHuang/libuipc` is a GitHub fork of it (`fork: true`, `parent`/`source` = spiriMirror/libuipc), created 2025-08-04, 0 stars, last push 2026-02-16. spiriMirror/libuipc is not a fork, was created 2024-06-02, has 361 stars and 82 forks, was pushed 2026-11-24, and hosts the docs site (spirimirror.github.io/libuipc-doc), the samples repository and the PR history. The PR goes to spiriMirror/libuipc; KemengHuang/libuipc is the author's personal fork for his own PRs.

### Licence

Fact: Apache-2.0 for everything a port touches. `utils/licenses.py check` -> Apache-2.0, tier 1, cleared to contribute. `scan-nvidia` -> no NVIDIA proprietary licence text anywhere in the tree.

Per-file read:

- `LICENSE`: Apache-2.0 full text. `README.md` says Apache-2.0. Both `pyproject.toml` files carry the Apache classifier.
- `NOTICE`: credits GPU IPC (Kemeng Huang), HKU (Taku Komura), CMU (Minchen Li), TransGP; states `src/geometry/metis` is derived from METIS and GKlib, both Apache-2.0, notices preserved.
- `src/geometry/metis/LICENSE-METIS`, `LICENSE-GKlib`: Apache-2.0 (University of Minnesota).
- `src/geometry/graph_coloring/LICENSE.md`: MIT (Brian Crites, 2014).
- `src/geometry/tetrahedralization/*`: `SPDX-License-Identifier: Apache-2.0` headers.
- `integrations/blender/libuipc_blender/`: mixed, explicitly scoped. The Blender-specific files are GPL-3.0-or-later, the reusable ones Apache-2.0, and the LICENSE there states that libuipc itself stays Apache-2.0. A port has no reason to touch this directory.
- Submodules: `scripts/SymEigen` (MuGdxy/SymEigen, a codegen script) and `libuipc-samples` (spiriMirror). Neither is in the CUDA build.

One inconsistency, recorded as a question and not a blocker: `python/LICENSE` is the GPL-3.0 full text (added 2024-09-16 in commit f7373727df, "add pyuipc whl"), and `python/pyproject.toml` points `license = { file = "LICENSE" }` at it, while the same file's classifier says Apache and the README says the project is Apache-2.0. It looks like a wrong file copied into the wheel directory, but that is a guess. The CUDA backend, CMake and C++ tests a port changes are all under the top-level Apache-2.0, so contribution is unaffected unless the port edits `python/` (it should not need to beyond, at most, a backend name in a test). Question for jeff: is the upstream `python/LICENSE` mismatch something to raise with the maintainer (a person's call, it is an upstream-visible remark), or ignore?

### Existing AMD / HIP / ROCm effort: none

- Upstream PRs and issues, all states: searched `hip`, `rocm`, `amd`, `radeon`; zero hits. Full title scan of all issues and PRs found only #215 "support mac, but no cuda backend" (a CPU-only build) and CUDA refactors.
- Upstream branches (22): none named hip/rocm/amd.
- 82 forks: none named for AMD/HIP/ROCm. KemengHuang/libuipc is the author's fork, not a port.
- Docs: `grep -rniwE 'amd|rocm|hip|gfx|radeon'` over README, docs/, agent_docs/, CLAUDE.md markdown: no hits.
- AMD-Ecosystem, ROCm, AMD-AGI, GPUOpen orgs: no libuipc or StiffGIPC repository (AMD-Ecosystem/libuipc does not exist yet).
- Web search (libuipc ROCm / HIP / AMD GPU): nothing.

Classification: no existing support; port from scratch, reusing the techniques (not the code) from our own GPU_IPC port.

### CUDA backend structure

Backends are runtime-loaded modules (`src/backends/common`, `uipc_add_backend`). Two exist: `none` (a CPU no-op) and `cuda`. Only `cuda` does simulation, so the port is one new build mode of `src/backends/cuda`, not a new backend directory.

- Size: 198 `.cu` files in `src/backends/cuda` (~50.1k lines of `.cu`; ~106k lines counting `.h`/`.inl`/`.cpp` in that directory), plus 19 `.cu` test files under `apps/tests/backends/cuda`. No `.cuh`.
- CUDA libraries: CUB only (`cub/device/*` radix sort, scan, reduce, select, partition, run-length encode; `cub/warp/warp_reduce`, `warp_scan`). No Thrust, cuBLAS, cuSPARSE, cuSOLVER, cuRAND, cuFFT, NCCL, CUTLASS or inline PTX. `linear_system.h` comments note the cusparse descriptors were deliberately dropped. Maps to hipCUB/rocPRIM.
- muda: no longer a dependency. libuipc used to pull MuGdxy/muda through vcpkg; that entry is commented out in `scripts/gen_vcpkg_json.py`, and the backend now carries `cuda_tool/` ("self-written device utilities": buffers, views, launch, cub wrappers, linear-system views, small-matrix Eigen helpers, graph wrapper). Only one leftover test, `apps/tests/backends/cuda/hello_muda.cu`, references the name. So no separate CUDA repository needs its own port; `depends_on` stays empty.
- Eigen on device: yes, extensively (109 `.cu` files use Eigen fixed-size types). `cuda_tool/eigen/evd.h` calls `SelfAdjointEigenSolver::computeDirect()` for N<=3 and the general `compute()` for N>3. On HIP, `compute()` in device code fails to compile (it was the GPU_IPC blocker, see below); the planner must find which call sites instantiate N>3 in device code. `svd` already uses an in-tree QR-SVD, not JacobiSVD.
- CUDA graphs: `cuda_tool/graph.h` uses graph capture, and WHILE conditional nodes (`cudaGraphConditionalHandle`, `cudaGraphSetConditional`) for the fused PCG loop in `linear_system/linear_fused_pcg.cu`. The conditional path is gated `CUDART_VERSION >= 12040` and `capture()` reports Unsupported otherwise, so a HIP build has a fallback to the host-driven loop; whether HIP's graph API gives conditional nodes is a planner question, not a blocker.
- libcu++: `#include <cuda/atomic>` with one `cuda::atomic_thread_fence(acquire, system)` in `collision_detection/details/linear_bvh.inl`. Map to libhipcxx or a HIP/clang fence.
- Driver API: `#include <cuda.h>` in `cuda_tool/eigen/svd_impl.h` only; no `cu*` driver calls found.
- Build flags: RDC is required (`CUDA_SEPARABLE_COMPILATION ON`; `affine_body/utils.cu` device functions called across TUs), so HIP needs `-fgpu-rdc`. `--extended-lambda` and `--expt-relaxed-constexpr` throughout (clang HIP accepts both semantics natively). C++20 in device code. Sources are split across domain OBJECT libraries (`components.cmake`) with one device-link in the final target.
- Warp-width assumptions (the main correctness risk on wave64): 75 literal `0xffffffff` masks, 60 `__shfl*` and 2 `__ballot` uses, `threadIdx.x & 31` / `>> 5` lane math in `collision_detection/details/stackless_bvh.inl` and `info_stackless_bvh.inl` (the same BVH code family as GPU_IPC's `mlbvh.cu`), `fast_segmental_reduce.inl` with `cub::WarpReduce<T, WarpSize>`, and `cub::WarpReduce<double>` with the default width in `finite_element/mas_preconditioner_engine.cu` (hipCUB's default is the hardware wavefront, 64 on CDNA, which changes the reduction's shared-memory layout and lane grouping). `cuda_device/builtin.h` redeclares the warp intrinsics but only under `#ifdef __INTELLISENSE__`, so it does not collide with HIP headers.
- FP64 atomics: 48 `atomicAdd`, mostly double. Fine on CDNA; on RDNA check whether `-munsafe-fp-atomics` matters for performance.
- `printf` from device code: 108 uses (fine on HIP).

### Build system

- CMake (primary, `CMakePresets.json`) and xmake (`xmake.lua`, a parallel build with `cuda` toolchain and rules). A port must cover CMake; whether to also cover xmake is a planner/scope question for jeff (rule: `feedback_rocm_build_just_works` asks the documented command to work, and the README documents both).
- Options: `UIPC_WITH_CUDA_BACKEND` (ON by default), `UIPC_CUDA_ARCHITECTURES` (default `native`), `UIPC_BUILD_PYBIND`, `UIPC_BUILD_TESTS`, `UIPC_WITH_USD_SUPPORT`/`UIPC_WITH_VDB_SUPPORT` (OFF). A port adds a HIP backend switch and `CMAKE_HIP_ARCHITECTURES`.
- Dependencies come from vcpkg (manifest generated at configure time by `scripts/gen_vcpkg_json.py`, with spiriMirror's own vcpkg registry for `octree` and `ftetwild`): eigen3, fmt, spdlog, nlohmann-json, libigl, tbb, cppitertools, magic-enum, cpptrace, dylib, catch2, urdfdom, tinygltf (local port), optional openvdb and usd. All CPU-only. conda env (`conda/env.yaml`) pins `cuda-toolkit=12.8`, cmake 3.26, python 3.11.
- Python bindings: pybind11 (`src/pybind`), wheel `pyuipc` via scikit-build (`pyproject.toml`).
- Windows/MSVC is a supported upstream platform (CUDA 13 `/Zc:preprocessor` handling in the CMake), so the windows gate is meaningful.

### Tests (validation surface)

Much better than GPU_IPC, which had none: 167 C++ test sources under `apps/tests` (Catch2; `backends/`, `sim_case/`, `regression/`, `sanity_check/`, `core/`, `geometry/`) and 25 pytest files under `python/tests` (contact finite-difference checks, sim cases). Upstream PR #507 adds an optional workflow running the CUDA backend tests on a simulated H100, so there is a CUDA baseline to compare pass counts against.

### How the GPU_IPC lessons carry over

Same author, same algorithms (GIPC/IPC barrier, MAS preconditioner, stackless/linear BVH, PCG). From `projects/GPU_IPC/notes.md` on `port/GPU_IPC`:

1. Warp intrinsics: never map `__shfl_*_sync`/`__ballot_sync` to ROCm's native forms, which default `width` to the wavefront (64 on CDNA) and return a 64-bit ballot. GPU_IPC pins width 32 and shifts the wavefront ballot to the calling lane's 32-lane half, implemented as real functions rather than function-like macros (macros of those names break ROCm's own declarations in any TU that includes `hip_bf16.h`, which Thrust/CUB pulls in). libuipc's BVH and segmented-reduce code needs the same treatment, plus the hipCUB `WarpReduce` default-width case GPU_IPC did not have.
2. Eigen `SelfAdjointEigenSolver::compute()` does not compile in HIP device code (host-only Householder tridiagonalization products); `computeDirect()` for 2x2/3x3 does. GPU_IPC replaced the one 12x12 call site with a cyclic Jacobi sweep, HIP-only, keeping the CUDA path on upstream's Eigen. libuipc's `evd.h` has exactly this split already (N<=3 direct, else `compute()`), and it already carries a device-safe `evd_jacobi` (`evd.h:162`), so the fix may be routing N>3 to it under HIP.
3. Do not translate nvcc `--use_fast_math` to clang `-ffast-math`: in FP64 it enables finite-math-only and reassociation, fatal to geometric predicates. Use `-ffp-contract=on`. libuipc's CMake and xmake set no fast-math flag today (grep at screen time), so the port only needs not to add one.
4. Uninitialized device memory: ROCm does not return zeroed pages; GPU_IPC's worst-case-sized collision buffers made this a live hypothesis (ruled out there, but the test method, memset 0 then 0xFF on every allocation and watching whether behaviour moves, is reusable).
5. GPU_IPC's line search had an uncapped `while(isIntersected)` loop that wedged on all backends after ~30 frames. libuipc is a rewrite; the planner should check whether the equivalent loop is bounded before reading a hang as a port defect.
6. wave64 evidence needs real CDNA hardware; `-mwavefrontsize64` on RDNA breaks rocPRIM (it takes its wave size from the arch macro), so it is useless for a whole app that uses CUB.
7. The Windows gfx1151 laptop bugchecked (CrowdStrike `csagent.sys`) under GPU_IPC's GPU work and the windows gate was waived for GPU_IPC on 2026-11-20. Expect the same question for libuipc's windows gate.

### Size estimate

Large but mechanical. Strategy A (CMake HIP language with a compat header, sources keep CUDA spelling), similar shape to GPU_IPC but roughly 5x the device code and a more structured build (OBJECT libraries, RDC, xmake twin, vcpkg). Expected work items: CMake HIP mode with `-fgpu-rdc` across the OBJECT targets; compat header for runtime/graph/CUB->hipCUB/libcu++ fence; warp-width audit of ~140 sites; Eigen `compute()` for N>3 device instantiations; conditional-graph fallback check; Windows clang/lld link. Rough guess: 2-4 porter rounds, with most risk in wave64 correctness and the RDC device-link on Windows. Nothing found that has no ROCm equivalent.

### Dependencies

`depends_on`: none. No hard dependency is itself a CUDA project (all vcpkg deps are CPU libraries; muda is no longer used). Optional: OpenVDB and USD (off by default, CPU).

### Fork request row (for jeff)

| upstream | licence determination | fork to create |
|---|---|---|
| https://github.com/spiriMirror/libuipc | Apache-2.0 (full scan: top-level LICENSE + NOTICE; METIS/GKlib Apache-2.0, graph_coloring MIT, Blender integration GPL-3.0-or-later/Apache-2.0 scoped to that directory and not touched by a port; `python/LICENSE` GPL-3.0 text contradicts the package's own Apache classifier, question above; no NVIDIA proprietary text) | AMD-Ecosystem/libuipc (does not exist yet; fork default branch `main`, port branch `moat-port`) |

### Open questions for jeff

1. `python/LICENSE` (GPL-3.0 text vs Apache classifier): raise upstream or ignore? Does not block the CUDA-backend port.
2. Scope: CMake only, or CMake plus the xmake build? The README documents both.
3. StiffGIPC (https://github.com/KemengHuang/Stiff-GIPC, MPL-2.0, "email the authors for commercial use") was also mentioned by the author; not screened here. libuipc is the one he called more popular and asked for.
