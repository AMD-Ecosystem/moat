# exllamav3 notes

## Intake screen (2026-11-05, gap-matrix seed)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict "partial, AMD path upstream"). Upstream https://github.com/turboderp-org/exllamav3 at bd1f5191 (2026-10-31), not archived, pushed 2026-11-04, ~1.8k stars.

### Licence

- `licenses.py check`: MIT, tier 1, cleared to contribute. Evidence: https://github.com/turboderp-org/exllamav3/blob/master/LICENSE
- `licenses.py scan-nvidia agent_space/exllamav3-screen`: no NVIDIA proprietary licence text.
- Vendored `exllamav3/exllamav3_ext/rocm/vendor/rocm_wmma_gemm/LICENSE`: MIT (Adel Johar). No submodules.

### Existing AMD support: mature, upstream, maintainer-owned

- Upstream has a native in-tree ROCm/HIP backend: `exllamav3/exllamav3_ext/rocm/` (compat.h, RDNA WMMA wrappers incl. emulated int8 WMMA, RDNA-specific EXL3 GEMV/GEMM/MoE sibling kernels, DPP reductions, vendored rocm_wmma_gemm with gfx1100/gfx1151 tuning configs, autotuner). Landed 2026-10-03/04 in commits da7983bc..cc50bbb2, 14 of 15 by the maintainer (turboderp), consolidating earlier community efforts (#283, #365-367, #441-443).
- Every release ships ROCm wheels: v1.6.6 (2026-10-31) has `exllamav3-1.6.6+rocm10.0.torch2.13.0-cp312/313/314-linux_x86_64.whl`.
- `doc/env_vars.md` documents ROCm-specific defaults (EXL3_GRAPHS=0 on ROCm due to a HIP graph-exec kernarg leak, EXL3_HC_FOLD=1, EXL3_RDNA_OCC_MAX, offload-compress, etc.).
- Very active ROCm contribution stream: merged #451, #458, #493 (gfx1150), #503, #519; open #542, #547, #548, #549 (SelfRef, RDNA3 kernels from phoenixhaxor/exllamav3-rocm, maintainer said "I'm happy to include more" on #543), #554 (gfx1151 DSA prefill), #581, #482 (ROCm test oracles).
- Inline PTX / MMA question is moot: the maintainer already wrote RDNA-native sibling kernels on WMMA (`rocm/rdna_wmma.cuh`, `rocm/quant/*_rdna.cu*`) rather than a mechanical translation of the mma.sync/PTX trellis GEMMs.

### Gaps that remain upstream (deliberate scope, not missing work)

- **CDNA / wave64 is explicitly rejected**: `exllamav3/util/arch_list.py` raises "exllamav3 on ROCm supports wave32 (RDNA) GPUs only"; `rocm/compat.h` notes the kernels assume 32-lane warps throughout. No issue or PR requests MI200/MI300 support (searched MI300, MI250, wave64, CDNA, gfx942, gfx90a, Instinct: zero hits). ExLlama targets consumer single-user inference; datacenter Instinct users are served by vLLM/SGLang. A wave64 port would mean re-deriving every RDNA-specific kernel for MFMA, i.e. an AMD-native rewrite of the hot path that the maintainer has not asked for.
- **Windows ROCm**: no Windows ROCm wheels; open community issue #544 (gfx1201, "8 build blockers") with a working stack. Open #459/#479: Linux JIT build fails with stock g++ host (clang required). These are small community-tracked build fixes, already in flight upstream.

### Duplicate effort

- No ROCm/exllamav3 or AMD-Ecosystem/exllamav3 (404). `amd` and `AMD-AGI` orgs are behind SAML SSO for this token, so a fork there is not ruled out; given upstream support, it does not change the answer.
- Community: phoenixhaxor/exllamav3-rocm (pre-upstream RDNA3 port), SelfRef/exllamav3-rdna (being upstreamed via #542-#549).

### Recommendation

Decline, SKIP_REASON `already-supported`. Upstream ships a maintainer-owned, released ROCm backend for RDNA (wave32) with per-arch tuning and an active contributor pipeline; MOAT would be duplicating it. The only uncovered surface is CDNA/wave64, which upstream excludes by design and nobody has requested; that would be an AMD-native kernel rewrite, not a port, and is a product decision for a person rather than a MOAT intake. The Windows ROCm build gaps (#544) are already being worked by the community upstream.

Question for jeff: Do you want CDNA/wave64 (MI300) support for exllamav3 pursued as a separate effort? Upstream rejects wave64 by design (`arch_list.py`), so it would be an AMD-native MFMA rewrite of the EXL3 GEMM/GEMV/MoE kernels rather than a port, and would need the maintainer's agreement first. Absent that, the recommendation is decline `already-supported`.
