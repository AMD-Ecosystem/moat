# DCGM notes

## Intake screen (2026-11-05, gap-matrix seed; matrix: ready via AMD alternative AMD SMI, ROCm Device Metrics Exporter, AMD GPU Operator)

Recommendation: **decline**, SKIP_REASON `ported-elsewhere`.

- Licence: Apache-2.0 (tier 1, `utils/licenses.py check NVIDIA/DCGM`). No per-file NVIDIA scan run, because the recommendation is a decline.
- What it is: NVIDIA's data-centre GPU manager -- a host engine, `dcgmi` CLI, field-collection library, the NVVS diagnostic suite and `dcgmproftester`, all built on NVML and NVIDIA's profiling interfaces. It is device-management software, not CUDA compute: porting it means re-targeting every NVML call at a different management library, which is a rewrite of the product.
- AMD equivalent, owned by AMD teams and maintained:
  - RDC (ROCm Data Center tool), AMD's direct DCGM counterpart: `ROCm/rocm-systems/projects/rdc` (standalone `ROCm/rdc` deprecated in favour of it).
  - AMD SMI (`ROCm/rocm-systems/projects/amdsmi`, also `ROCm/amdsmi`): the NVML counterpart underneath.
  - ROCm Validation Suite (`ROCm/ROCmValidationSuite`): the counterpart of DCGM diagnostics (NVVS).
  - ROCm Device Metrics Exporter (`ROCm/device-metrics-exporter`) and AMD GPU Operator (`ROCm/gpu-operator`): the counterparts of dcgm-exporter and NVIDIA GPU Operator in Kubernetes.
- Remaining gap: none MOAT can close; the coverage is complete across the AMD stack.
- Duplicate check: upstream PR search for rocm/amd/hip returns nothing; NVIDIA would not take an AMD backend.
