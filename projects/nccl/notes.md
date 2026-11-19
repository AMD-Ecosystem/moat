# nccl notes

## Intake screen (2026-11-06)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: partial, AMD fork RCCL; entry about "NCCL advanced features (NVLink SHARP, symmetric memory)").

### Licence
`utils/licenses.py check NVIDIA/nccl`: GitHub NOASSERTION. LICENSE.txt read by hand: "Most of this project is licensed under the Apache-2 license ... Parts of the project retain their original BSD license", followed by the full Apache-2.0 text and a 3-clause BSD text; borrowed files carry their own licence. Recorded `Apache-2.0 AND BSD-3-Clause` (tier 1). No per-file NVIDIA proprietary scan run: the recommendation is a decline, and the repo is large enough that the brief asked for no clone.

### Existing AMD support
- RCCL is AMD's own NCCL port. `ROCm/rccl` is retired (README CAUTION) in favour of the ROCm monorepo `ROCm/rocm-systems` at `projects/rccl`, last commit 2026-11-06, maintained by AMD.
- It tracks NCCL closely: `makefiles/version.mk` is NCCL 2.32 (upstream latest release v2.32.3-1, 2026-09-17), and `src/` has the same layout as NCCL's plus RCCL extras (`rccl_wrap.cc`, `rccl_dev_runtime.cc`, `rccl_arch_thresholds.cc`, `algorithms/`, `commDump.cc`).
- Symmetric memory: RCCL carries NCCL's symmetric-memory stack -- `sym_kernels.cc`, `src/device/symmetric/` (all_gather, all_reduce, reduce_scatter, including GIN variants), `dev_runtime*.cc`, `gin/`, `rma/`, `ce_coll.cc`. Present and ported.
- NVLink SHARP (NVLS): `src/transport/nvls.cc` is stubbed on the AMD path ("NVLS (NVLink SHARP) is NVIDIA-only; provide stubs so callers in init.cc link"; nvlsSupport is 0). That is a hardware feature (NVSwitch in-network multicast reduction) with no AMD equivalent, not a software gap MOAT could close. The matrix's "partial" reflects this hardware difference.
- Upstream NVIDIA/nccl will not take a HIP backend; AMD's path is RCCL.

### Recommendation
Decline, reason `ported-elsewhere`: AMD maintains RCCL at NCCL 2.32 parity in ROCm/rocm-systems, including symmetric memory; the remaining gap (NVLS) is NVIDIA switch hardware, not software. Any RCCL feature gap belongs to the RCCL team, not to a parallel MOAT port.
