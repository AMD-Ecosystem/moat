# Stiff-GIPC notes

## Intake 2026-11-24 (linux-gfx942 host, no GPU used)

Screened upstream `KemengHuang/Stiff-GIPC` at bb2849a7 (2026-08-31, "fix(math): avoid copysign in Wilkinson shift"), shallow clone into `agent_space/`. Nothing built.

### Why this was screened

Our HIP port of GPU_IPC (https://github.com/KemengHuang/GPU_IPC/pull/4) merged upstream 2026-11-24T04:53Z. One minute before, the author wrote (https://github.com/KemengHuang/GPU_IPC/pull/4#issuecomment-7122416498): "We have a more complete simulator, StiffGIPC, which is much faster and supports strong coupling between rigid and deformable bodies. Even better is our user-friendly version, libuipc. Especially libuipc is more popular, and I believe AMD support could greatly advance libuipc applications, since FP64 on AMD is more powerful than on NVIDIA, and our algorithm is almost entirely FP64-based. If you are interested in libuipc, we would also welcome your PR."

Upstream engagement with this author is established (one merged PR). Note the exact wording: the explicit PR invitation names libuipc; Stiff-GIPC is mentioned as the faster simulator, not as a PR target. A Stiff-GIPC PR is plausible but not pre-invited.

### Licence: MPL-2.0 (tier 2), a fact, not a guess

The premise that GitHub reports no licence is out of date or was wrong: the GitHub API reports `mpl-2.0` today, and the file has been there from the start.

- `LICENSE` at the repo root is the verbatim Mozilla Public License 2.0 text (16725 bytes). The GitHub commit history for that path has one commit, 3c02443852 "init project" (2025-10-26), so the repository has been MPL-2.0 since its first commit. Evidence: https://github.com/KemengHuang/Stiff-GIPC/blob/bb2849a7b292099581907937860d96ecfdf42588/LICENSE
- `python3 utils/licenses.py check KemengHuang/Stiff-GIPC`: `license=MPL-2.0 tier=2`. Recorded with `set-license-spdx`.
- README: "this software is released under the MPLv2.0 license. For commercial use, please email the authors for negotiation." The upstream `AGENTS.md` repeats this. That sentence is a request layered on MPL-2.0 (which itself permits commercial use); it does not affect contributing a fix upstream. It would matter only if AMD were to ship or depend on Stiff-GIPC in its own product, which is a separate consult.
- File headers: 24 files under `StiffGIPC/` carry `// Copyright (c) 2024 Kemeng Huang. All rights reserved.` (one adds Jiming Ruan). The same style as GPU_IPC; no conflicting licence text, no SPDX tags.
- Vendored code, `MeshProcess/External/METIS` and `MeshProcess/External/GKlib` (KemengHuang forks of KarypisLab METIS/GKlib, vendored as plain directories, no `.gitmodules`): both carry the University of Minnesota Apache-2.0 notice. Clean.
- `python3 utils/licenses.py scan-nvidia agent_space/Stiff-GIPC-screen`: no NVIDIA proprietary licence text.

Comparison: GPU_IPC is recorded as `license_spdx: MPL-2.0` too (same author, same headers), so this is the identical situation that already went upstream cleanly.

No licence request to the author is needed, so none is drafted.

### Duplicate effort: none

- README / AGENTS.md: no AMD, ROCm or HIP mention; "Hardware requirements: Nvidia GPUs".
- Upstream PRs (all states): one, #1 "add vcpkg.json" by MuGdxy (open since 2025-11-18). Issues: one, closed. No HIP/ROCm/AMD PR or issue.
- Upstream branches: `main`, `main-old`, `nomuda`, `test`, `windows-cuda-13.2+vs-2026`. None HIP-related.
- 16 forks; none has a branch matching `hip|rocm|amd`. Also checked `haoxiangNtu/stiff-physics`, `haoxiangNtu/stiffgipc-isaaclab`, `KLBSTL/StiffGIPC_Learning` (by name): no AMD branches in the first; the others are learning/integration repos.
- AMD-Ecosystem and ROCm orgs: no repo or code matching gipc / StiffGIPC.
- The only AMD work for this family is our own: GPU_IPC (merged upstream) and the `port/libuipc` claim being screened in parallel.

### Overlap with GPU_IPC and libuipc

Measured with a line-set comparison (substantive lines: whitespace stripped, longer than 25 characters, comments excluded) and per-file `difflib` matching against GPU_IPC at 4fb6019 (upstream immediately before our port merged) and libuipc at e06f8ee (`src/backends/cuda`).

| Comparison | Shared substantive lines |
| --- | --- |
| Stiff-GIPC lines also in GPU_IPC | 6114 / 14558 (42%) |
| Stiff-GIPC lines also in libuipc CUDA backend | 788 / 14558 (5.4%) |
| GPU_IPC lines also in libuipc CUDA backend | 337 / 8111 (4.2%) |

**Stiff-GIPC is a superset-by-descent of GPU_IPC**: the same author's code base, reorganised into directories (`core/`, `collision/`, `fem/`, `solver/`, `math/`, `io/`, `app/`) with large additions. Every GPU_IPC source except `gl_main.cpp` (now `app/gl_main.cu`), `cuda_cub_wrappers.cu`, `device_buffer.h` and `device_utils.*` has a same-named descendant here. Per-file overlap with GPU_IPC: GIPC.cu 76% (7457 -> 10261 lines), GIPC_PDerivative.cuh 47%, MASPreconditioner.cu 58% (1501 -> 2186), mlbvh.cu 59%, femEnergy.cu 55% (1149 -> 2491), ACCD.cu 65%, FrictionUtils.cuh 66%, load_mesh.cpp 84%, QRSVD.hpp/givens.hpp 87-100%. The two upstreams also receive the same fixes in parallel (bb2849a7 here and 4fb6019 in GPU_IPC carry the identical "avoid copysign in Wilkinson shift" commit two days apart).

New relative to GPU_IPC (roughly 58% of the code): affine body dynamics (`abd_system/`, ~4.5k lines including `abd_energy.inl` 1909), a block linear-system framework (`linear_system/`, global matrix, subsystems, preconditioners, `converter.cu` with warp code), an in-repo CUDA utility layer (`cuda_tools/`, buffer views, dense vectors, cub wrappers; it replaced a vendored muda, per the upstream AGENTS.md and the `nomuda` branch), and `gipc/utils/parallel_algorithm/` (segmented reduce, unique-reduce, scatter).

**libuipc is a different code base**: a rewrite on muda with its own backend architecture. It credits GPU IPC in its NOTICE and calls Stiff-GIPC "the prototype implementation of Libuipc ... as our performance benchmark reference" (README line 204), and it reimplements the StiffGIPC MAS preconditioner (`src/backends/cuda/finite_element/mas_preconditioner_engine.cu`), but only 2.3% of Stiff-GIPC's MASPreconditioner.cu lines appear verbatim there. A libuipc port does not port Stiff-GIPC and vice versa. What transfers between the two is knowledge (MAS warp/bank logic on wave64, device eigensolvers, FP64 behaviour), not patches.

So: Stiff-GIPC is not subsumed by libuipc. It is the natural sequel to GPU_IPC, and most of the GPU_IPC port's fixes carry over almost mechanically.

### What the GPU_IPC port carries into

The merged GPU_IPC port (940bdae, 23 files, +480/-91) has direct counterparts here:

- CMake HIP language path with `-fgpu-rdc` for `CUDA_SEPARABLE_COMPILATION` and the `cuda_to_hip.h` shim -> same top-level `CMakeLists.txt` shape (single `gipc` executable, `file(GLOB_RECURSE)` sources, `CUDA_SEPARABLE_COMPILATION ON`, `--extended-lambda`, `--expt-relaxed-constexpr`, `--default-stream=per-thread`, `CUDA_ARCHITECTURES native`).
- The blocker GPU_IPC hit -- Eigen's `SelfAdjointEigenSolver` in device code, replaced by a HIP-only Jacobi eigensolver -> present here in `cuda_tools/cuda_eigen.h:177` (`evd`, a `CT_GENERIC` helper), `fem/femEnergy.cu:68,1697`, `core/GIPC.cu:134,162`.
- The MAS preconditioner warp/bank logic with full-mask intrinsics (wave64 hazard) -> `solver/MASPreconditioner.cu` (also uses `cooperative_groups` and `cooperative_groups/reduce.h`), `core/GIPC.cu`, plus the new `linear_system/utils/converter.cu`. About 140 `__shfl/__ballot/__popc/...` uses and 158 `0xffffffff`/warp-size literals across those three files.
- mlbvh, ACCD, friction, femEnergy fixes -> the descendant files above.

### CUDA surface, dependencies, build, platforms

- Size: `StiffGIPC/` has 106 source files, 46.5k lines (31 `.cu`, 10 `.cuh`, 51 `.h`, 8 `.inl`, 2 `.hpp`, 4 `.cpp`); 36.7k lines in `.cu`/`.cuh`. Roughly 1.6x GPU_IPC (28.1k lines).
- CUDA libraries: CUB (~50 uses in 7 files, warp reduce/scan and device primitives; maps to hipCUB/rocPRIM), cooperative groups (MASPreconditioner.cu). cuBLAS and cuSPARSE are linked, but the code only creates, streams and destroys handles (`cuda_tools/cuda_linear_system.h:148-179`), so hipBLAS/hipSPARSE are a near-trivial substitution. cuSOLVER is linked but its only use (`cusolverDnDsyevd`, `collision/mlbvh.cu:2360-2430`) is commented out. No Thrust in active code, no inline PTX, no textures, no CUDA graphs, no managed memory, no `__launch_bounds__`. 403 `atomicAdd`, mostly FP64 (fine on gfx90a/gfx942; check `unsafe-fp-atomics` choices as in GPU_IPC).
- Host dependencies: Eigen 3.4.0 (host and device), nlohmann_json, FreeGLUT, GLEW, OpenGL (interactive viewer); METIS + GKlib vendored and built from source under `MeshProcess/`. All in apt. No dependency on another MOAT project (`depends_on` stays empty).
- Build: CMake >= 3.18, `project(StiffGIPC LANGUAGES CXX CUDA)`, CUDA >= 11.0 (CUDA 13 CCCL include path handled), Linux and Windows (vcpkg or apt). FP64 throughout, `--use_fast_math`.
- Tests: none. The commit 42e69645 (2026-08-29) "fix global reduction outputs and remove tests" removed what existed. As in GPU_IPC, validation will be the headless Xvfb + xdotool run of the GLUT app (`gl_main.cu:73` starts with `bool stop = true`, same as GPU_IPC) with frame-count and Newton-convergence evidence, preferably on several of the bundled scenes (`Assets/scene`, cloth, mat twisting, box pile).

### Activity

Created 2025-10-26; 35 stars, 16 forks; last push 2026-08-31 (memory management refactor and MAS buffer fixes on 2026-08-29); about 60 commits, all by the author. Not archived. Active but bursty; the author maintains GPU_IPC alongside it.

### Size estimate for the port

Medium. A GPU_IPC-style port (CMake HIP path, shim header, device eigensolver replacement, wave64-safe MAS) is the first 60% of the work and can be reused nearly verbatim. The remainder is the new `abd_system/`, `linear_system/`, `cuda_tools/` and `parallel_algorithm/` code, which a planner should sweep for warp-size and lane-mask assumptions (`converter.cu`, segmented-reduce code) and for CUB usage. Expect roughly one and a half to two times the GPU_IPC effort, mostly in validation of the new rigid-deformable coupling scenes, not in compile fixes.

### Recommendation: fork

Worth taking up as its own port: MPL-2.0 (tier 2, same as GPU_IPC), no duplicate effort, genuinely CUDA, 42% shared with the already-merged GPU_IPC port, and not subsumed by libuipc (5% verbatim overlap; libuipc is a separate muda-based rewrite). The author explicitly welcomed a libuipc PR and only described Stiff-GIPC, so if only one of the two can be staffed now, libuipc is the stronger upstream case (361 stars vs 35, explicit invitation, Apache-2.0); Stiff-GIPC is the cheaper port and the paper's reference implementation, and a person may reasonably want both. Sequencing the two is a person's call.

### Fork request row

| Repository | Upstream | Licence (full intake scan) |
| --- | --- | --- |
| Stiff-GIPC | https://github.com/KemengHuang/Stiff-GIPC | MPL-2.0 (root LICENSE since first commit; vendored METIS/GKlib Apache-2.0; no NVIDIA proprietary text; README asks commercial users to contact the authors) |
