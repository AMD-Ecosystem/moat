# FBGEMM notes

## Intake screen (2026-11-05, gap-matrix seed)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: partial, upstream).

### Licence
- `python3 utils/licenses.py check pytorch/FBGEMM` -> UNPARSED (GitHub NOASSERTION). Read the LICENSE file: BSD 3-clause text ("Neither the name Facebook nor the names of its contributors may be used to endorse..."), copyright Meta Platforms. Recorded BSD-3-Clause, tier 1 (cleared to contribute). Submodules not scanned: the recommendation is a decline, so nothing is offered upstream from MOAT.

### Existing AMD support (decisive)
- ROCm is a first-class upstream target: FBGEMM_GPU build docs have a "Set Up for ROCm Build" section (fbgemm_gpu/docs/src/fbgemm_gpu/development/BuildInstructions.rst), and upstream CI carries `fbgemm_gpu_ci_rocm.yml`, `_fbgemm_gpu_rocm_build.yml`, `_fbgemm_gpu_rocm_test.yml`, `fbgemm_gpu_benchmark_rocm.yml`.
- AMD maintains it: AMD-org fork https://github.com/ROCm/FBGEMM (forked from pytorch/FBGEMM, pushed 2026-11-02); a steady stream of open AMD/ROCm PRs upstream (e.g. #6724 ROCm CI matrix, #6600 native gather/scatter on HIP, #6515 complete wave32 TBE codegen, #6405 link TheRock runtime, #6496 MI350/MI450 benchmark harness, #5561 ROCm SSD TBE inference).
- jeff's own ROCm work landed upstream (PRs close via Meta's internal merge): #5696 / #5804 / #6123 support warpSize 32 and 64 in one build, #6103 per-build wave-size flags, #6024 / #6278 kWarpSizeHost(), #6033 cache associativity from warp size, #6014 gate DPP warp reduce to gfx942/gfx90a/gfx950, plus hipify v2 fixes (#4854, #4705, #4028) and ROCm 6.0 / gen_ai enablement (#2086, #2610). So wave64 and wave32 (RDNA) are both handled upstream.

### Remaining gap
- The matrix's "partial" reflects work still in flight (wave32 TBE codegen completion #6515, SSD TBE inference #5561, some gen_ai/CK paths), all owned by the AMD team upstream. Windows is not an FBGEMM_GPU target on CUDA either, so there is no Windows gap specific to ROCm.
- Nothing for MOAT that would not compete with the owners' open PRs.

### Recommendation
Decline, SKIP_REASON `already-supported`: ROCm support is upstream, in CI, and AMD-maintained (ROCm/FBGEMM fork, jeff's wave32/wave64 PRs, ongoing AMD PRs).
