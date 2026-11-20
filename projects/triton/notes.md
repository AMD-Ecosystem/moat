# triton notes

## Intake screen (2026-11-05, gap-matrix seed; matrix: ready, upstream)

- Licence: MIT, tier 1 (`utils/licenses.py check triton-lang/triton`).
- What it is: the Triton GPU kernel language and compiler (MLIR/LLVM). Not a CUDA-source project: it has per-vendor backends under `third_party/nvidia` and `third_party/amd`.
- Existing AMD support, upstream and AMD-maintained:
  - `third_party/amd/` is the in-tree AMD backend; `CODEOWNERS` assigns it to AMD engineers (`third_party/amd/ @antiagainst @zhanglx13`).
  - Upstream CI runs `.github/workflows/integration-tests-amd.yml` on self-hosted `amd-gfx90a`, `amd-gfx942` and `amd-gfx950` runners, with a `tritonci/triton-amd-ci` ROCm image (rocm10.1 preview, gfx942/gfx950/gfx1250).
  - `third_party/amd/backend/compiler.py` handles CDNA (gfx942, gfx950, gfx1250) and RDNA (gfx11xx incl. gfx1151, gfx120x) targets, so wave64 and wave32 are both covered.
  - README documents `LLVM_BUILD=amd`, the AMDGPU LLVM target, and an AMD-only address-sanitizer mode pointing at rocm.docs.amd.com.
  - PyTorch's ROCm wheels depend on Triton built for ROCm (`pytorch-triton-rocm`), so torch.compile/inductor on ROCm already rides this backend.
- Upstream: very active (pushed 2026-11-05; releases v3.8.0, v3.9.0). About 30 open PRs match "AMD", which is the AMD team's ongoing backend work rather than an outside port.
- Windows: not in this repo; see the separate `triton-windows` screen.

## Recommendation: decline, `already-supported`

Triton's AMD backend is in-tree, owned by AMD engineers via CODEOWNERS, and gated by upstream CI on gfx90a/gfx942/gfx950 runners with RDNA targets in the compiler; it is the backend PyTorch ROCm ships for torch.compile. There is no CUDA-to-HIP port to do and MOAT would only duplicate AMD's own team.
