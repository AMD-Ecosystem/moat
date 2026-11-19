# whisper.cpp notes

## Intake screen (2026-11-05, gap-matrix seed)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: ready, upstream).

- Licence: MIT (`utils/licenses.py check ggml-org/whisper.cpp` -> tier 1), https://github.com/ggml-org/whisper.cpp/blob/master/LICENSE
- Upstream active (pushed 2026-11-05), not archived.
- whisper.cpp has no CUDA code of its own: all GPU work goes through the vendored ggml (`ggml/src/ggml-hip/CMakeLists.txt` present), which is synced from llama.cpp and carries the same mature HIP backend (see the llama.cpp screen).
- Upstream README has an "AMD ROCm GPU support" section: `cmake -B build -DGGML_HIP=1 -DAMDGPU_TARGETS="gfx1201"`, naming gfx1100/gfx1101/gfx1201. It also documents an AMD Ryzen AI NPU (VitisAI) path.
- Gap: upstream CI and releases carry no ROCm build (workflows: no hip/rocm job; v1.9.5 release has no assets). That gap is already being worked:
  - Open upstream PR #3757 "feat: add ROCm GPU support for AMD hardware" (Geramy, 2026-04-17, not draft): Linux and Windows ROCm CI workflows plus release artifacts for gfx1151/gfx1150/gfx110X/gfx120X.
  - `lemonade-sdk/whisper.cpp-rocm` (AMD Lemonade fork of upstream) ships ROCm release binaries per arch for Linux and Windows (v1.8.4: `whisper-v1.8.4-{linux,windows}-rocm-gfx{110X,1150,1151,120X}`).
- Duplicate-effort check: no ROCm or AMD-Ecosystem repo named whisper.cpp found; `amd`/`AMD-AGI` orgs not searched (SAML SSO), but the Lemonade fork already covers the packaging angle.

## Recommendation

Decline, reason `already-supported`: the HIP backend is upstream (shared ggml) and documented; the only missing piece, ROCm CI/release packaging, is covered by open PR #3757 and the Lemonade fork's ROCm binaries, so a MOAT port would duplicate existing effort.
