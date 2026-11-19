# Megatron-LM notes

## Intake screen (2026-11-05)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: ready, AMD fork ROCm/Megatron-LM).

### Licence
- GitHub licence field: NOASSERTION. Read LICENSE directly: BSD-3-Clause for NVIDIA code ("applies to all files unless otherwise noted"), plus third-party files under their own notices (Apache-2.0 / MIT / BSD from Hugging Face, Google Research, Facebook, Microsoft, Mamba, Triton, vLLM and others). Recorded `license_spdx = BSD-3-Clause`.
- No per-file `scan-nvidia` run: the recommendation is a decline, so nothing is forked or offered upstream.

### Existing AMD support
- https://github.com/ROCm/Megatron-LM -- AMD-owned (ROCm org) fork of NVIDIA/Megatron-LM, default branch `rocm_dev`, actively maintained: regular IFU syncs with upstream (latest `core_r0.18.0` IFU, 2026-09), commits through 2026-09-30, pushed 2026-11-06. AMD ships it as the ROCm Megatron-LM training container (`rocm/megatron-lm`) documented on rocm.docs.amd.com.
- Upstream itself has some ROCm awareness (`torch.version.hip` in `megatron/core/ssm/ops/gdp/common.py`) and an open ROCm PR (#7631, async checkpoint staging on ROCm), so AMD contributors also work upstream directly.
- Authoritative, AMD-official, maintained: no gap for MOAT to fill. Megatron is mostly Python over PyTorch / Transformer Engine; its GPU kernel surface is TE and apex, which AMD ports separately (ROCm/TransformerEngine, ROCm/apex).

### Recommendation
Decline, SKIP_REASON `ported-elsewhere`: AMD's own ROCm/Megatron-LM fork covers it and is actively synced with upstream.
