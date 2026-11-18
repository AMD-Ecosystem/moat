# numba-cuda notes

## Intake screen (2026-11-05, gap-matrix seed; matrix: partial, AMD fork Numba HIP ROCm/numba-hip)

Recommendation: **decline**, SKIP_REASON `ported-elsewhere`.

- Licence: BSD-2-Clause (tier 1, `utils/licenses.py check NVIDIA/numba-cuda`). No per-file NVIDIA scan run, because the recommendation is a decline.
- What it is: the CUDA target of Numba, split out of numba/numba and now maintained by NVIDIA -- a Python-to-NVVM/PTX JIT compiler plus driver bindings. It is a compiler back end, not a kernel library; MOAT already recorded numba/numba as `cant-port` ("LLVM-based CUDA JIT compiler, not a kernel port", 2026-05-29), and the same reasoning applies to the split-out target.
- AMD equivalent: Numba HIP (`numba.hip`), AMD's ROCm target for Numba, owned by the ROCm team. The matrix's `ROCm/numba-hip` is archived (default branch `deprecated-repo`, README: "Development has been moved to the monorepo https://github.com/ROCm/hip-python"); the live code is `ROCm/hip-python/packages/numba-hip`, built and tested with the HIP Python wheels (README: "`numba-hip` -- the ROCm HIP target for Numba"). Active as of 2026-11-02.
- Remaining gap: any missing `numba.cuda` API coverage in `numba.hip` is AMD's ROCm team's roadmap in hip-python, not a MOAT port of NVIDIA's compiler.
- Duplicate check: AMD's own effort above; nothing upstream at NVIDIA/numba-cuda would accept a HIP target.
