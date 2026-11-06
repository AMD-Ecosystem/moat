# DALI notes

## Intake screen (2026-11-05)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: ready, via AMD alternative rocAL; NVIDIA declined non-NVIDIA support).

### Licence
- `licenses.py check NVIDIA/DALI`: Apache-2.0, tier 1. Recorded `license_spdx = Apache-2.0`.
- No per-file `scan-nvidia` run: the recommendation is a decline, so nothing is forked or offered upstream. (DALI is built on nvJPEG / nvImageCodec / NVDEC / NPP; a port would be a library-substitution rewrite, not a hipify, so the per-file licence question was not worth opening.)

### Existing AMD support
- No AMD fork: `ROCm/DALI` does not exist, and NVIDIA/DALI issue and PR searches for AMD / ROCm / HIP return only unrelated CUDA error reports. The matrix's "NVIDIA declined non-NVIDIA support" was not independently located in this screen; DALI is in any case CUDA-only by design (NVIDIA codec and NPP libraries throughout).
- AMD's answer is a separately-named project: https://github.com/ROCm/rocAL (ROCm Augmentation Library), AMD-owned, part of ROCm with docs at rocm.docs.amd.com/projects/rocAL, MIT, very active (three commits on 2026-11-05). It covers DALI's role: a decode-and-augment pipeline graph for images, video and audio with rocJPEG / rocDecode backends, a C API, and Python plugins for PyTorch, TensorFlow and JAX (`rocAL_pybind/amd/rocal/plugin/{pytorch,tf,jax}.py`) -- the same framework-iterator shape as DALI's plugins.
- Per `assess-existing-support.md`, a mature separate AMD project (ROCm-DS style) means we do not duplicate AMD's own work.

### Remaining gap
- rocAL is not API-compatible with DALI, and its operator set is smaller than DALI's. Code written against `nvidia.dali` does not run unchanged on AMD. Closing that would mean either a DALI-API compatibility layer over rocAL or adding operators to rocAL; both are rocAL-team work, not a MOAT CUDA-to-HIP port of DALI.

### Recommendation
Decline, SKIP_REASON `ported-elsewhere`: AMD covers DALI's role with its own maintained library, rocAL, and DALI itself is a CUDA-library-bound NVIDIA product with no upstream route for AMD support.
