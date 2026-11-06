# triton-windows notes

## Intake screen (2026-11-05, gap-matrix seed; matrix: ready, AMD fork path "Triton on Windows")

- Licence: MIT, tier 1 (`utils/licenses.py check triton-lang/triton-windows`).
- What it is: a fork of triton-lang/triton (GitHub `fork=true`, parent `triton-lang/triton`) that adds Windows support, now hosted in the `triton-lang` org. Releases (`v3.9.0-windows.post30` 2026-10-28, `v3.8.0-windows.post29`, ...) are cut by woct0rdho, a community maintainer (the repo was formerly `woct0rdho/triton-windows`); not an AMD or ROCm-org effort. Active (pushed 2026-11-02), not archived. Default branch is `readme`; code lives on release branches.
- Existing AMD support, in this repo:
  - README states "Recent Nvidia and AMD GPUs are supported" and lists RDNA4 (gfx120x), RDNA3.5 (gfx115x, Strix Halo), RDNA3 (gfx110x) and RDNA2 (gfx103x) as supported, plus a per-dtype table that also covers CDNA3 (gfx94x).
  - The AMD backend is built with clang-cl (MSVCRT) on Windows; README points AMD users at TheRock ROCm + PyTorch wheels (https://github.com/ROCm/TheRock/blob/main/RELEASES.md).
  - It carries RDNA2 support that official Triton dropped (triton-lang/triton#12042; triton-windows#41), so it already goes beyond upstream on AMD.
  - The maintainer also upstreams shared fixes that touch the AMD backend (triton-lang/triton#12083, "Catch LLVM diagnostic ... in both Nvidia and AMD backends", open).
- The AMD backend itself is triton-lang/triton's in-tree, AMD-owned `third_party/amd` (see the `triton` screen).
- Open AMD gaps: no open triton-windows issue matching AMD/ROCm/gfx. Proton and GSan are documented as not maintained on Windows for any vendor; that is not an AMD-specific gap. TheRock's RELEASES.md does not list a Triton wheel, so the Windows AMD Triton path is this fork's wheels.
- MOAT relevance: this repo is the "AMD fork path" the matrix names, and it already works for the hardware MOAT's windows gate covers (gfx1151). A MOAT port would duplicate a maintained, working Windows AMD path.

## Recommendation: decline, `already-supported`

triton-windows already supports AMD on Windows (RDNA2 through RDNA4 including Strix Halo gfx1151, with TheRock ROCm), released monthly by its community maintainer under the triton-lang org on top of upstream's AMD-owned backend. Nothing is left to port; any Windows-specific AMD fix MOAT finds belongs as a PR to this repo, not a port project.
