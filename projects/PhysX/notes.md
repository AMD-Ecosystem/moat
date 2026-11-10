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
