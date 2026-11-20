# PhysX notes

## Intake screen (2026-11-06)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: partial, AMD fork exists).

### Licence
`utils/licenses.py check NVIDIA-Omniverse/PhysX`: BSD-3-Clause, tier 1. LICENSE.md read by hand: "BSD 3-Clause License, Copyright (c) 2008-2025, NVIDIA Corporation". No per-file `scan-nvidia` run and no clone (the repo is very large and the recommendation is a decline); run the scan before any take-up.

### Existing AMD support
- `AMD-Ecosystem/PhysX` (GitHub fork of NVIDIA-Omniverse/PhysX, AMD-Ecosystem org) has a `rocm-enablement` branch with one commit, d09e25323d, 2026-09-16, "[ROCm] AMD Instinct (ROCm) enablement for PhysX", signed off by three AMD engineers (amd.com addresses). Its `main` is an unmodified mirror. Not a MOAT project (no MOAT records; `scaffold` accepted the name).
- Scope: 47 files. PhysX 5 GPU simulation through a `cuda_to_hip.h` shim plus `hip_compat/` header stubs: CMake for every GPU module (articulation, broadphase, common, narrowphase, simulation controller, solver, CUDA context manager), `CudaKernelWrangler.cpp` (+204), warp helpers, atomics, SDF collision, and an `INSTALL_ROCm.md`. Built and tested on MI300X (gfx942) and MI350X/MI355X (gfx950), ROCm 10.0.0, Ubuntu 24.04; only the Linux GPU CMake path is touched.
- Upstream: no ROCm/HIP/AMD PRs on NVIDIA-Omniverse/PhysX; the branch has not been offered upstream and is 3 commits behind upstream main (ovphysx updates).

### Gaps left by the AMD fork (for the record, not a reason to compete)
- Instinct-only, Linux-only: no wave32 (RDNA) or Windows validation, and the Windows GPU CMake path is untouched.
- Header-shim approach (`hip_compat/` stubs named like CUDA headers) rather than hipify-in-place; footprint not reviewed.

### Recommendation
Decline, reason `ported-elsewhere`: an AMD-owned fork (AMD-Ecosystem/PhysX, `rocm-enablement`) already ports PhysX 5 GPU simulation and validates it on MI300X/MI355X. RDNA wave32 and Windows coverage are differentiators to contribute to that branch in coordination with its owners, not a parallel MOAT port.

### Question for jeff
AMD-Ecosystem/PhysX is in our own org but is not a MOAT project. Should MOAT adopt it as an existing fork (validate wave32 and Windows on the `rocm-enablement` branch) in coordination with its authors, or leave it to them?

## Take-up ruling (2026-11-10)

Jeff Daily ruled on 2026-11-10 that MOAT takes PhysX up, building on the AMD team's existing ROCm work, overriding intake's decline (ported-elsewhere) recommendation:

- Fork: https://github.com/AMD-Ecosystem/PhysX. `main` is the upstream mirror and the recorded `fork_default_branch`. `rocm-enablement` (d09e25323d8eea94117a74d7c2bd25309da527d5, one squashed commit "[ROCm] AMD Instinct (ROCm) enablement for PhysX", 2026-09-16, 47 files) is the AMD team's Instinct/Linux port. Both branches stay as they are.
- Port base: the porter creates `moat-port` from the `rocm-enablement` tip, keeping that commit and its authorship, then merges upstream NVIDIA-Omniverse/PhysX `main` (3 commits newer at the time of the ruling). Our work goes on top as separate commits. Unlike MONAI, this is not an independent port: PhysX has no upstream ROCm support and Jeff judged the diff close to what MOAT would produce with hip_compat headers.
- Clean-up in scope: the rocm-enablement commit carries many whitespace-only changes. Revert them on `moat-port` (a separate commit on top), so the delta is the ROCm change alone. The reviewer should check for any other noise of the same kind.
- In scope, not deferred: the CDNA SDF limitation. INSTALL_ROCm.md, "Known limitation on CDNA (gfx942/gfx950)", says SDF mesh collision traps on CDNA because there is no hardware 3D-texture sampler (`tex3D`), and names "a software-trilinear sampler over a linear device copy of the grid" as the planned follow-up. Jeff ruled that follow-up part of this port: implement it, so SDF collision works on gfx942/gfx950 and the trap goes away, with tests that exercise SDF collision on a CDNA host.
- Also in scope: Radeon (wave32, gfx1100/gfx1201) and Windows gfx1151, with every gate validated (Linux wave64 and wave32 first, Windows last).
- Delivery vehicle (PR into the AMD fork's rocm-enablement, a handoff to the AMD team, or an upstream PR) is Jeff's call later.

Recorded: `set-intake PhysX fork`, `set-state PhysX linux-gfx942 awaiting-fork`, `release-forks` (fork exists, stage `screened`, `fork_url` set). No `fork_url_ruling` block yet: `set-fork-url` needs `moat-port` to exist on the fork; re-run `set-fork-url PhysX https://github.com/AMD-Ecosystem/PhysX --by "Jeff Daily"` once the porter has pushed it.

## Planning (2026-11-20, linux-gfx1100, planner)

### Questions for Jeff (Question 1 must be answered before a porter starts)

**Question 1 (blocking): is ovphysx part of this port?** The repository is no longer just the SDK. At upstream 5.13 it holds `physx/` (the SDK; all GPU simulation code), `ovphysx/` (NVIDIA's USD-native Python/C product built on the SDK), `blast/` and `flow/`. blast and flow have no CUDA (flow is D3D12/Vulkan shaders), so they need nothing. ovphysx does have CUDA of its own: 4 `.cu` files (`ovruntime/.../particles/TypeConversionUtility.cu`, `ovruntime/.../tensors/gpu/CudaKernels.cu`, `src/ovphysx/OvStageOutputCuda.cu`), about 30 host files calling the CUDA runtime, and Python output as Warp arrays. It builds against NVIDIA Carbonite and ovstage binaries that packman fetches, and `licenses.py scan-nvidia` finds two NVIDIA-proprietary licence texts in it (`ovphysx/licenses/LICENSE-binary.txt`, `ovphysx/tools/internal-licenses/Omniverse-Extensions-LICENSE.txt`). The AMD rocm-enablement branch does not touch it.
- Option A (planner's recommendation): port the `physx/` SDK only (CPU SDK, the GPU library, snippets, pvddom), and record ovphysx as deferred (`deferred.py add` against PhysX) for a later round once the SDK port is validated. Reason: ovphysx is a separate product with closed binary dependencies and a Warp dependency; the SDK port is the prerequisite for it either way.
- Option B: include ovphysx now. That adds `depends_on: [warp]` (MOAT's `warp` project), a check that the Carbonite/ovstage packman binaries can be used in a ROCm build at all, a ROCm path for its 4 `.cu` files and CUDA-runtime host code, and its C/Python test suites (c_unittests, c_samples, python tests, benchmarks) to the validation bar.
- Option C: some other cut (for example ovphysx's GPU tensor path only).
Until ruled, the 70 ovphysx components are `owed` in surface.json (in scope, not done), so nothing is cut silently and `pr_ready` refuses. The SDK work in plan.md is needed under every option, so a porter dispatched before the ruling would not waste work, but the plan does not assume the cut. A hold (`set-hold PhysX on`) is a person's action, so this run did not set one.

**Question 2 (does not block): licence record.** Upstream relicensed between the AMD base and today: every `physx/` source file now says `SPDX-License-Identifier: Apache-2.0` and `physx/LICENSE.md` is Apache 2.0; the root `LICENSE.md` is still BSD-3-Clause, which is what `licenses.py check` (tier 1, cleared) and `license_spdx` say. Should the record move to Apache-2.0 (`set-license-spdx`)? Separately, the AMD-added files (`cuda_to_hip.h`, `hip_compat/*`, `cutil_math.h`) carry BSD-3 headers with an AMD "Modifications Copyright" line; whether they switch to Apache-2.0 SPDX headers before any delivery is a call for you and the AMD authors.

**Question 3 (does not block): parity harness.** PhysX's unit tests are not in the open-source tree and the snippets assert nothing, so plan.md has the porter author a GPU-vs-CPU parity harness under `physx/snippets/`, as its own commit on `moat-port`. It ships with the port; whether it goes into an upstream PR or a handoff is your call at delivery.

### What this run did
- `orient.sh` on `port/PhysX`: next = PhysX, state screened, planner. Took the lock: `set-state PhysX linux-gfx1100 planning --agent planner`. `set-ext-type PhysX cmake`.
- Clones: `projects/PhysX/src` = fork with `upstream` remote, detached at upstream `main` `7d6f6ac8` (PhysX 5.13.0), `GIT_LFS_SKIP_SMUDGE=1` (the repository uses LFS for ovphysx media; a merge without it fails in the smudge filter).
- Trial merge, read only: `git merge-tree --write-tree --name-only d09e2532 7d6f6ac8` -> 24 conflicts; 21 are the licence header block only, 3 carry content (`physx/CMakeLists.txt`, `cmakegpu/linux/CMakeLists.txt`, `PxgGeometryManager.cpp`). With `-X ignore-all-space`: 23.
- `python3 utils/surface.py generate PhysX` against `7d6f6ac8`: 162 CUDA files, 131 components; planner added 21 (driver-API seam, registration seam, SDK GPU loader, SDF sampler, Windows GPU CMake, top-level `PX_USE_HIP` build, parity harness, 14 GPU snippets), 70 ovphysx components owed, `cuda-library:thrust` removed from the floor (vehicle "thrust" variables). `surface.py check PhysX`: fully accounted for.
- `licenses.py scan-nvidia projects/PhysX/src` (owed since intake): 2 hits, both licence texts inside `ovphysx/` (Question 1). Nothing under `physx/`.

### Findings that change the port (details in plan.md)
- Upstream moved from 5.10.0 to 5.13.0 since the AMD base (4 commits, 5240 files), not the 3 small ovphysx commits the ruling saw. No new kernel TUs (61 `.cu`, 495 kernels); one new driver call (`cuOccupancyMaxActiveBlocksPerMultiprocessor`) the shim lacks.
- Two silent no-ops in the AMD branch: `atomicAddSpatialVector` (`PxgArticulationBlockData.h:188`, used by `rigidDeltaAccum.cu`) and the narrowphase overflow `atomicOr` (`cudaNpCommon.h:107`) sit under `__CUDA_ARCH__` gates that HIP leaves undefined.
- The SDK's GPU module loader (`PxPhysXGpuModuleLoader.cpp`) requires `libcuda.so.1` before it opens `libPhysXGpu_64.so`, and the AMD branch did not change it, so no SDK-level GPU simulation can have run on that branch; snippets fall back to the CPU silently. Its evidence (fatbin/ldd/objdump checks plus a load-and-dispatch ladder) is consistent with that.
- The AMD SDF change compiles the texture path out on every HIP target, RDNA included, not only CDNA. The ruled software sampler will be the single HIP path.
- `__hipRegisterFatBinary` interposition can lose to the runtime's own symbol when the HIP runtime is already loaded globally; Windows needs its own forwarding.
- `__syncwarp` maps to a bare `wave_barrier` with no memory fence.
- The `bar.sync N` -> `__syncthreads()` mapping is exact (upstream's comments say each is a whole-block barrier).
