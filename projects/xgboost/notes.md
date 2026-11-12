# xgboost notes

## Intake screen (2026-11-06)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: partial, AMD fork).

- Licence: Apache-2.0, tier 1 (`utils/licenses.py check dmlc/xgboost`), https://github.com/dmlc/xgboost/blob/master/LICENSE. Not cloned; no submodule/NVIDIA-file scan done because the recommendation is a decline.
- Upstream dmlc/xgboost (latest release v3.4.2): no HIP build path. Top-level CMakeLists.txt mentions HIP only in a compiler-option help string; no `USE_HIP`. PR search for rocm/hip/amd returns no ROCm-enablement PR, open or merged.
- AMD fork: https://github.com/ROCm/xgboost (ROCm org, fork of dmlc/xgboost, AMD-owned). Default branch `release/3.2.0`, last pushed 2026-10-19; adds `USE_HIP` / `USE_RCCL` CMake options, `src/common/cuda_to_hip.h`, `device_helpers.hip.h`, HIP tests. Recent commits by AMD staff: "Enable for ROCm7 and update documentation" (2025-12-23), "add rocm 7.1.1 tab" (2026-01), "Update to release/3.2.0" (2026-07-24), plus docs/26.01 and docs/26.05 branches (ROCm docs site releases). Older `master-rocm` carries "fix navi wave32" (2025-04), so RDNA/wave32 was addressed.
- Judgement: authoritative and maintained AMD-owned port. Remaining gap is only that it trails upstream by a couple of minor releases (3.2.0 vs 3.4.2) and is not upstreamed; that is the ROCm team's release cadence, not a MOAT port.

Recommendation: decline, SKIP_REASON `ported-elsewhere` (AMD-owned ROCm/xgboost carries the HIP port and is actively released with ROCm).
