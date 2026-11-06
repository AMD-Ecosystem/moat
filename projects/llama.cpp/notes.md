# llama.cpp notes

## Intake screen (2026-11-05, gap-matrix seed)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: ready, upstream).

- Licence: MIT (`utils/licenses.py check ggml-org/llama.cpp` -> tier 1), https://github.com/ggml-org/llama.cpp/blob/master/LICENSE
- Upstream active (pushed 2026-11-06), not archived.
- Existing AMD support, upstream and mature:
  - `GGML_HIP=ON` CMake backend, documented for Linux and Windows in `docs/build.md` ("## HIP", lines ~387-420).
  - Release workflow `.github/workflows/release.yml` has a `windows-rocm` job (ROCm 10.0.0); every release ships ROCm binaries, e.g. b12141: `llama-b12141-bin-ubuntu-rocm-10.0-x64.tar.gz`, `llama-b12141-bin-win-rocm-10.0-x64.zip`.
  - CI: `hip-quality-check.yml`; open PR #31567 runs a gfx1151 ROCm job on Strix Halo runners.
  - Open HIP PRs are tuning/maintenance by active contributors (#31746 RDNA3 D256 MMA FA, #31625 HIP fast-math, #30348 spill check on every AMD arch, #31617 bump ROCm to 6.3, #26592 hipCUB path, #16915 TheRock build docs, #26745 RDNA4 docs). No porting gap for MOAT to fill.

## Recommendation

Decline, reason `already-supported`: llama.cpp ships a first-class HIP backend with ROCm release binaries for Linux and Windows and an active AMD contributor stream; a MOAT port would duplicate it.
