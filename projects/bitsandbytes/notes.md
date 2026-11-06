# bitsandbytes notes

## Intake screen (2026-11-05, gap-matrix seed; matrix: ready, upstream)

- Licence: MIT, tier 1 (`utils/licenses.py check bitsandbytes-foundation/bitsandbytes`).
- CUDA surface: `csrc/` holds `kernels.cu`, `ops.cu`, the `gemm_4bit_*.cu` family, `common.cuh`, `compat.cuh`, `compat_device.cuh`. There is no separate HIP tree: the ROCm build compiles the same `.cu` sources through the `compat*.cuh` shims (single-source HIP port, already upstream).
- Existing AMD support, upstream:
  - README's support matrix lists "AMD GPU" (`cuda` device) with LLM.int8(), QLoRA 4-bit and 8-bit optimizers all marked supported on Linux x86-64 for CDNA gfx908/gfx90a/gfx942/gfx950/gfx1250 and RDNA gfx101X/103X/110X/115X/120X, and on Windows 11 x86-64 for CDNA gfx908/gfx90a and the same RDNA list. So all three feature families are ROCm-complete per upstream's own table, wave64 and wave32, Linux and Windows.
  - `.github/workflows/python-package.yml` has a `build-rocm` job building the ROCm shared library for ROCm 6.4.4, 7.0.2, 7.1.1, 7.2.4, 7.14.0 and 10.0.0 via `.github/scripts/build-rocm.sh`, packaged into the release wheels.
  - Open upstream PRs show active AMD-side maintenance by several parties, including an AMD engineer: #2089 "[ROCm] Preserve float bits in SIMT DPP reduction" (liminfei-amd), #2101 "Cache the ROCm warp size per device", #2113 "Fix fp32 4-bit GEMV test tolerance for ROCm/gfx1200", #2073 "[ROCm] Allow explicit GPU architecture override", #2018 int8 GEMM BLAS Lt handle on ROCm/CUDA, and draft #2102 "Make the Windows ROCm load path work and fix has_imma on ROCm" (verified on gfx1201 Windows 11, ROCm 7.2).
  - ROCm/bitsandbytes (branch `rocm_enabled`, last push 2026-08-19) is the older AMD fork that ROCm docs tutorials build from; upstream now carries the support itself.
- Remaining gaps: upstream PR CI (`tests-pr.yml`) runs GPU tests only on CUDA; ROCm is build-only in upstream CI, so AMD GPU correctness rests on contributors' own runs. The open PRs above are the residual Windows/RDNA fixes and already have owners; a MOAT effort here would compete with them. Not a port gap.
- Upstream: active (pushed 2026-11-02), not archived.

## Recommendation: decline, `already-supported`

bitsandbytes already ships a single-source HIP build upstream with all three feature families (LLM.int8(), 4-bit QLoRA, 8-bit optimizers) marked supported on CDNA and RDNA, on Linux and Windows, and builds ROCm 6.4 through 10.0 libraries in its release workflow. What is left are small ROCm fixes that AMD and community contributors already have open as PRs; MOAT should not open competing ones.
