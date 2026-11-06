# MONAI notes

## Intake screen (2026-11-05, gap-matrix seed)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: partial, AMD fork).

### Licence
- `python3 utils/licenses.py check Project-MONAI/MONAI` -> Apache-2.0, tier 1 (cleared to contribute).

### Existing AMD support (decisive)
- AMD maintains MONAI on ROCm: org fork https://github.com/AMD-Ecosystem/MONAI (forked from Project-MONAI/MONAI, default branch `amd-develop`, tip f9f65c9 dated 2026-09-28, repo pushed 2026-11-05). AMD-official and actively maintained.
- Shipped as `amd-monai` on AMD's PyPI index as part of ROCm-LS (ROCm Toolkit for Life Sciences): https://rocm.docs.amd.com/projects/monai/en/latest/index.html and https://rocm.blogs.amd.com/artificial-intelligence/monai-rocm/README.html. Supported: MI300X / MI325X / MI355X, ROCm 7.0.2 / 7.2.0, Ubuntu 24.04. Labelled early access.
- Upstream is absorbing it too: Project-MONAI/MONAI#9153 "Add ROCm support for AMD Instinct GPUs" MERGED 2026-10-06 (setup.py accepts ROCM_HOME, JIT cache key falls back to torch.version.hip, Dockerfile.rocm). #9161 "Resolve accelerator-vendor dependency variants from pyproject.toml" OPEN since 2026-10-09 (makes `monai[all]` / `monai[cucim]` installable on ROCm). Earlier, #6161 "Add supports for AMD GPU" merged.

### Remaining gap
- The matrix's "partial" is coverage: ROCm-LS documents only Instinct (gfx94x/gfx950) on Linux. Radeon (RDNA, wave32) and Windows are not listed as supported.
- That gap belongs to the ROCm-LS team that owns the fork and the upstream PRs, not to a parallel MOAT port. If wider coverage is wanted, the useful contribution is Radeon/Windows validation evidence handed to that team, coordinated with them.

### Viability
- MONAI is mostly pure PyTorch with a small optional C++/CUDA extension set (monai/csrc), which the merged upstream PR already builds through CUDAExtension hipification. Viable, but already done.

### Recommendation
Decline, SKIP_REASON `ported-elsewhere`: AMD's own team maintains the ROCm port (AMD-Ecosystem/MONAI `amd-develop`, `amd-monai` in ROCm-LS) and is upstreaming it. Note the Radeon/Windows coverage gap for the ROCm-LS team rather than as MOAT work.
