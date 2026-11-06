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
