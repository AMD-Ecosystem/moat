# exllamav2 notes

## Intake screen (2026-11-05)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict "partial", AMD path "upstream"). Screened at upstream master tip (last commit 2026-03-04, README only).

### Licence

MIT (tier 1), read from `LICENSE` at the repo root: https://github.com/turboderp-org/exllamav2/blob/master/LICENSE. No submodules. `python3 utils/licenses.py scan-nvidia agent_space/exllamav2-screen`: no NVIDIA proprietary licence text.

### Maintenance

README line 3: "This project is archived for now. Development continues on ExLlamaV3." The GitHub repository is not archived (PRs can still be opened), but development has stopped: last release v0.3.2 (2025-07-13); since then only CI workflow updates (torch 2.8/2.9, 2025-08 to 2025-12) and one README edit (2026-03-04). The successor turboderp-org/exllamav3 is screened separately.

### Existing AMD support: mature, upstream, first-party

- HIP path is upstream since 2023 (PRs #5, #22, #137, #154, #417 merged). The extension builds through torch's hipify; `exllamav2_ext/cuda/compat.cuh` and `graph.cuh` carry the `USE_ROCM` shims (half atomics, shfl, graph API), `ext.py` adds `-DHIPBLAS_USE_HIP_HALF`.
- Release wheels for ROCm are published by upstream CI: v0.3.2 ships rocm5.6/torch2.2.2, rocm6.0/2.3.1, rocm6.1/2.4.0, rocm6.2/2.5.0, rocm6.2.4/2.6.0, rocm6.3/2.7.0, rocm6.4/2.8.0 and rocm6.4/2.9.0, cp310-cp313, linux_x86_64 (workflows `build-wheels-release-rocm*.yml`).
- No kernel is disabled under HIP. The only HIP-specific behaviour changes: GEMM autotune off (`q_gemm_autotune.cuh`, "Autotune seems unreliable on ROCm", fixed block size 64), and the norm kernels size their reduction by `warpSize`.
- No ROCm/AMD fork under ROCm, AMD-Ecosystem or EmbeddedLLM (`gh search repos exllamav2 --owner ...`: empty). `amd` and `AMD-AGI` orgs not checked (SAML SSO for this token).

### Remaining gap: ROCm 7 build, already being fixed by AMD contributors

`rms_norm.cu` and `layer_norm.cu` use `1024 / warpSize` as a compile-time array size under `USE_ROCM`; since ROCm 7.0 `warpSize` is not a constant expression, so the extension fails to compile on ROCm 7.x. Two open upstream PRs, both by AMD employees, fix exactly this and both await maintainer review:

- #807 "Fix ROCm 7.0 build" (amd-vlarakic, 2025-09-25, +12/-5, ping 2025-11-01, no maintainer reply)
- #815 "fix: compile error on RDNA4 (gfx1201) ROCm" (Kaihui-AMD, 2026-05-15, +16/-6, validated on gfx1201 with ROCm 7.2.3 / torch 2.9.1, ping 2026-05-18, no maintainer reply)

Other open issues mentioning ROCm/AMD (#328, #397, #662, #685) are 2024 user reports (flash-attn on ROCm, OOM, spec-decode perf on 7900 XTX), not port gaps.

### Recommendation: decline, `already-supported`

The main path (quantized GEMM, attention, norms, MLP) is HIP-enabled upstream with first-party ROCm wheels through ROCm 6.4. The one real gap, the ROCm 7 `warpSize` build break, is a ~15-line fix already submitted twice by AMD engineers; a MOAT port would duplicate their PRs against a maintainer who has moved development to ExLlamaV3 and is not reviewing V2 PRs. Any effort is better spent on exllamav3. If someone wants V2 working on ROCm 7 regardless, the action is coordinating with the #807/#815 authors (or reviewing one of them), not a port.
