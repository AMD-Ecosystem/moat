# cuEquivariance notes

## Origin

Added 2026-06-04 at a colleague's suggestion (relayed by jeff) as a strategic target: cuEquivariance is "a major moat that NVIDIA has" -- a math library (segmented tensor products / equivariant neural-network ops: spherical harmonics, symmetric contractions, used by MACE-style ML interatomic potentials and equivariant GNNs). The core repo is open source, "but the performance-critical CUDA ops packages appear to rely on closed-source binary kernels. If we can overcome this gap, it would be a strong step forward."

## Viability assessment (2026-06-04) -- NOT a hipify/translation port

Repo inspection (github.com/NVIDIA/cuEquivariance):
- The repo is Apache-2.0 and contains only the PYTHON FRONTENDS: `cuequivariance/`, `cuequivariance_torch/`, `cuequivariance_jax/`. These build/describe the ops (the segmented-polynomial descriptor system) but do not contain kernels.
- The performance-critical kernels ship as CLOSED prebuilt binary wheels: `cuequivariance-ops-torch-cu12`, `cuequivariance-ops-jax-cu12` / `-cu13` (pip-installed, CUDA-version-pinned). There is NO `.cu`/`.cuh`/kernel source in the repo.
- No AMD / ROCm / HIP / SYCL support anywhere.

Implication: there is NOTHING TO HIPIFY. MOAT's normal CUDA->HIP translation (Strategy A/B) does not apply -- the CUDA source that would be translated is not public. "Overcoming the gap" means a GREENFIELD ROCm REIMPLEMENTATION of the kernels from the published math + the open frontend's descriptor specifications (write a ROCm executor backend for the open `cuequivariance` descriptors). That is research/engineering-grade scope, in the class of the OptiX->HIPRT and greenfield hipCIM/hipML efforts, NOT a routine port. Effort and risk are high; correctness/perf parity with NVIDIA's tuned closed kernels is the hard part.

## Merge policy / outcome

NVIDIA project -> NO upstream PR (standing rule + colleague's own note). Best achievable outcome is `fork` / greenfield: a standalone open ROCm implementation (e.g. a jeffdaily repo providing a ROCm backend for the open cuequivariance frontend). Publishing a new public repo and any AMD-facing step are GATED on explicit jeff go-ahead (same as hipCIM).

## Decision: PENDING (do NOT auto-start)

This is a strategic-scope decision, not a routine port. Two honest dispositions:
1. PURSUE as a greenfield ROCm backend for the open cuequivariance descriptor frontend (large, multi-session; would need its own plan, likely starting from the simplest descriptor ops and validating numerics against the CPU/JAX reference the frontend can produce without the closed ops).
2. BLOCK as out-of-translation-scope (closed binary kernels, no source) and record it as a documented NVIDIA-moat gap.

The planner should NOT treat this as a buildable hipify target. Await jeff's call on pursue-vs-block before any porting work. See [[moat-no-duplicate-amd-ports]] (existing-AMD assessment: none exists) and the greenfield workflow notes.
