# Liger-Kernel notes

## Intake screen (2026-11-05, windows-gfx1151 host, no GPU run)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict "partial", AMD path "upstream"). Screened at upstream `8c7a5078318da1295dc3ca445f9d85e9e9e153f5` (2026-11-02).

### Licence

`python3 utils/licenses.py check linkedin/Liger-Kernel` -> BSD-2-Clause, tier 1, cleared to contribute. `liger_cute_kernels/LICENSE` is the same LinkedIn BSD-2-Clause. Vendored third-party licences under `licenses/`: Apache-2.0, MIT (AutoAWQ, Efficient-Cross-Entropy, llm.c, triton). No submodules. `python3 utils/licenses.py scan-nvidia agent_space/Liger-Kernel-screen` -> no NVIDIA proprietary licence text.

### What the ~3 MB of `.cu` is

All 171 `.cu`/`.cuh` files (3.19 MB) are in `liger_cute_kernels/`, a separate, optional package ("LigerCute -- native CUTLASS + NVSHMEM kernels"; own `pyproject.toml`, own publish workflow `publish-liger-cute-kernels.yml`). It holds expert-parallel MoE (102 files under `csrc/core/src/moe`) and tensor-parallel fused scaled linear cross entropy (42 files, incl. `*_sm100.cuh`), plus 16 C++ tests. Hand-written, not vendored or generated. It targets Hopper/Blackwell only: CuTe/CUTLASS warp-specialised MMA, NVSHMEM one-sided RDMA, NVLS multicast reductions. The main `liger_kernel` package is pure Triton (plus optional CuTe-DSL, cuTile, Ascend backends) and does not need LigerCute: `ops/fused_linear_scaled_cross_entropy.py:111,367` returns False for `torch.version.hip`, so ROCm takes the Triton fallback. `liger_cute_kernels.configure` raises `UnsupportedDeviceError` on unsupported devices and the frontend falls back.

LigerCute has nothing hipify can translate (CUTLASS/CuTe + NVSHMEM + NVLS); an AMD equivalent would be a from-scratch reimplementation on CK/rocSHMEM/MFMA (see `references/no-hip-equivalent.md`), not a port, for two optional multi-GPU training ops that already have a Triton path on ROCm.

### Upstream ROCm support today

- README: "Full AMD support" since v0.4.0 (2024-11); ROCm install section (`torch` from the rocm7.2 index, `triton>=3.0.0`); AMD and EmbeddedLLM credited in Sponsorship ("Providing AMD GPUs for our AMD CI", "Making Liger Kernel run fast and stable on AMD"). `setup.py:get_platform()` detects ROCm via `rocm-smi` and picks ROCm deps.
- Code: `liger_kernel.ops.utils.is_hip()`; HIP-specific `num_warps=16` (vs 32) in CE, FLCE, fused_linear_jsd, fused_linear_kl_div, kl_div, score_centering, vocab_parallel CE, `utils.py:69`. FLCE `_can_use_mm_out` disables the `mm(out=)` buffer reuse fast path on HIP (`ops/fused_linear_cross_entropy.py:37`; test `requires_nvidia_cuda` skip at `test/ops/test_fused_linear_cross_entropy.py:317`): a small perf delta, not a correctness gap. `utils._AMD_ARCH_BY_GFX` lists gfx908/90a/940-942/1100-1102 only (no gfx950, gfx115x, gfx12xx), but on AMD it is only consumed by Blackwell checks, so it is cosmetic.
- AMD CI: `.github/workflows/amd-ci.yml`, MI300 runner `linux-mi300-gpu-1`, ROCm 7.1, `make test` + `make test-convergence`. Since upstream #1515 (2026-10-07) it is `workflow_dispatch` only: "AMD runners are currently unavailable". In the retained Actions history (1983 runs, 2025-10 .. 2026-10) there is not a single successful run: the newest are cancelled (queued with no runner) or failed with an empty job list, the older ones failed at the checkstyle gate with `tests` skipped. So upstream has had no observed AMD test signal for over a year, while the code path remains supported.
- Merged ROCm work: #506 ROCm 6.3 CI (tjtanaa, EmbeddedLLM), #587 HIP device inference, #646 ROCm 6.3 tolerances, #887 AMD CI fix, #1243 ROCm install extras (kashif, maintainer, 2026-06).
- Open: PR #1297 (kashif, maintainer, 2026-07-11) "[FlyDSL] Add optional ROCm/AMD FlyDSL backend with cross-entropy and fused linear cross-entropy" -- an AMD-native (ROCm/FlyDSL) backend being written upstream by a maintainer. Issue #624 (2025-03, MI300X/ROCm 6.3: one JSD test and one qwen2_vl convergence test needed relaxed atol; addressed by #646). Issue #538 (2025-01, ROCm source build). Issue #126 (2024, Triton flash attention on AMD).

### Duplicate effort

- EmbeddedLLM/Liger-Kernel (AMD partner fork, default branch `mi300-support`, pushed 2024-12-06): 16 ahead / 745 behind linkedin:main, merge base 2024-11-02. Its MI300 work landed upstream as v0.4.0 "Full AMD support" and #506. Stale, superseded.
- No ROCm/Liger-Kernel or AMD-Ecosystem/Liger-Kernel. The `amd` and `AMD-AGI` orgs are behind SAML SSO for this token, so a fork there could not be checked; given upstream support, it would not change the recommendation.

### Is there a real gap?

Not one MOAT should port. ROCm is a first-class, maintainer-supported path in upstream Triton code, with HIP-specific tuning and an upstream maintainer actively building an AMD-native backend (#1297). The remaining items are:

1. No live AMD CI (runners unavailable). That is an AMD-provided-runner problem, not a code gap; the fix is hardware/runner access for upstream, which is an AMD relationship question, not a port.
2. No evidence on RDNA/wave32 or gfx950: upstream CI only ever targeted MI300. A MOAT validation run on wave32/gfx950 could surface failing tests, but without a run there are no specific failing tests to name, and the Triton kernels carry no wave64-specific autotune configs (`num_warps=16` on HIP is a total-thread choice that is valid on both wave widths: 1024 threads on wave64, 512 on wave32).
3. LigerCute (all of the `.cu`): NVIDIA-only by design, optional, falls back to Triton on ROCm; an AMD version would be a reimplementation, not a port.

Recommendation: decline, `SKIP_REASON` = `already-supported`.

## Questions for jeff

- Upstream's AMD CI has been dark since 2025 (no successful run in the retained history; made manual-only on 2026-10-07 because "AMD runners are currently unavailable", and AMD is credited for providing those GPUs). Do you want this routed to whoever owns the AMD CI runner sponsorship for linkedin/Liger-Kernel, instead of MOAT doing anything? And if you want test evidence on wave32/gfx950 anyway, say so and the project can be taken up as a validate-only screen (run `make test` on a gfx1100 and a gfx950 fleet host, name failing tests) rather than declined.
