# cudnn-frontend notes

## Intake screen (2026-11-05, gap-matrix seed; matrix: partial, AMD alternative MIOpen / hipDNN)

Recommendation: **decline**, SKIP_REASON `ported-elsewhere`.

- Licence: Apache-2.0 (tier 1, `utils/licenses.py check NVIDIA/cudnn-frontend`); LICENSING.md says some files keep MIT or BSD-3-Clause terms for pre-existing code (per-file SPDX tags), all permissive. No per-file NVIDIA scan run, because the recommendation is a decline.
- What it is: a header-only C++ and Python front end to the cuDNN graph (backend) API. The graph API lowers onto closed `libcudnn` engines, so there is no kernel source behind it to port. The repo also now ships "frontend-only" Python kernels (`python/cudnn/{gemm,sdpa,linear_attention,moe_ep,...}`), but those are CuTe DSL kernels targeting Hopper/Blackwell (and Rubin), with no HIP target; only 9 `.cu` files in a 3724-path tree (8 under `python/cudnn/linear_attention`, 1 legacy sample).
- AMD equivalent: hipDNN, AMD's graph-based DNN library with a cuDNN-frontend-shaped frontend, Python bindings and plugin backends (MIOpen, hipBLASLt), owned by the ROCm team in `ROCm/rocm-libraries/projects/hipdnn` (the standalone `ROCm/hipDNN` repo is deprecated in favour of it). Published docs: https://rocm.docs.amd.com/projects/hipdnn/en/latest/index.html . Its README states it is early stage with limited operation support; that gap is AMD's ROCm team's roadmap, not a MOAT port. The fused attention / GEMM kernels on AMD live in AITER and Composable Kernel.
- Remaining gap: none MOAT can close. A HIP build of cudnn-frontend would still need cuDNN; reimplementing the CuTe DSL kernels for AMD is a new-kernel effort against AITER/CK, not a port.
- Duplicate check: upstream PR search for hip/rocm/amd returns only unrelated matches; no AMD branch upstream.
