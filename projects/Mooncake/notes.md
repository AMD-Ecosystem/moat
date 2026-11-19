# Mooncake notes

## Intake screen (2026-11-05, gap-matrix seed)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict partial, AMD path upstream). Screened at upstream main, shallow clone, 2026-11-05.

### Licence
- Apache-2.0 (tier 1), `LICENSE-APACHE`; GitHub field agrees.
- `utils/licenses.py scan-nvidia agent_space/Mooncake-screen`: no NVIDIA proprietary licence text.
- Only submodule is `extern/pybind11` (BSD-style), not a concern.

### Existing AMD support upstream (mature, AMD-staffed)
- `-DUSE_HIP=ON` (`mooncake-common/common.cmake:86`, defines `USE_HIP __HIP_PLATFORM_AMD__`) builds Transfer Engine (classic and TENT, `tent/src/platform/rocm`), the `hip` transport (HIP IPC / shareable handles, intra-node XGMI), RDMA with HIP dma-buf MRs (`USE_HIP_DMABUF`, default ON), Mooncake Store with `hip_accelerator_device.cpp`, Python integration, benchmarks and tests. Documented in `docs/source/getting_started/build.md:241,401` and `supported-protocols.md` ("AMD GPU Clusters: rdma + hip").
- Merged ROCm work by AMD people: #3184 ROCm wheel build, CI and release parity (andyluo7), #3338 self-hosted MI350X ROCm CI tier, #4266 ROCm nightly wheels, #4949 py3.14 ROCm wheels, #4657/#5050/#3548 HIP dma-buf / P2PDMA / IPC fixes; #1354 makes amd-arozanov CODEOWNER of `hip_transport`. Open AMD-side PRs: #3977 (junyyang-amd), #4953/#5040/#5044 HipTransport async/batched copies, #5072 TENT on MI350X CI; RFC #5041 (fast intra-node XGMI).
- GPUDirect RDMA on AMD is done through dma-buf, so no nvidia-peermem equivalent is needed. cuFile/GDS has no counterpart required for the KV-transfer path.

### What is still CUDA-only
- `mooncake-pg` (~186 KB `.cu`/`.cuh`): `mooncake-pg/src/CMakeLists.txt` FATAL_ERRORs for HIP; ROCm CI and release pass `-DWITH_EP=OFF`. AMD owns this gap: RFC kvcache-ai/Mooncake#5202 "Mooncake PG on AMD ROCm (HIP) for elastic EP" by AMD-yanfeiwang (2026-11-03), with a prototype passing the upstream PG test suite on MI355X and elastic recovery across nodes over RDMA; maintainer UNIDY2002 replied 2026-11-04. Collective v2 stays CUDA-only by design there.
- `mooncake-ep` (~460 KB, DeepEP-style dispatch/combine with IBGDA, PTX in `include/elastic/mooncake_ep_elastic_ptx.cuh`, runtime nvcc JIT): no HIP path. RFC #5202 lists it as a non-goal. Open PR #4810 (Hygon DCU via DTK, which is HIP-based) adapts EP and PG kernels to 64-wide wavefronts under `USE_HYGON`, active (updated 2026-11-05): the closest thing to a HIP EP and the coordination point for anyone doing EP on ROCm. AMD's own MoE dispatch stack for ROCm is separate (MORI / ROCm DeepEP), so IBGDA-on-AMD for mooncake-ep is AMD-native work that needs multi-node Instinct plus RDMA NICs to validate, not a MOAT-sized port.
- MNNVL / intra-NVLink transports: NVIDIA-fabric specific; the AMD analogue (intra-node XGMI) is the existing `hip` transport.

### EmbeddedLLM/Mooncake fork
- EmbeddedLLM is a vLLM-on-ROCm partner org. Both `main` and `fix-rocm` are 0 ahead / 741 behind upstream main (`gh api repos/kvcache-ai/Mooncake/compare/main...EmbeddedLLM:Mooncake:fix-rocm`): it carries nothing upstream lacks. Not an active parallel effort.

### Viability
- Genuinely CUDA (643 KB `.cu`), but the KV-transfer engine and store (the product's core) already build and ship on ROCm with MI350X CI. The remaining PG gap is being upstreamed by AMD now; EP is AMD-native territory with a Hygon HIP PR in flight.
- MOAT's fleet has no multi-node RDMA Instinct setup to validate the transfer paths.

### Recommendation
Decline, `SKIP_REASON` `already-supported`: upstream has mature, AMD-staffed ROCm/HIP support with CI and wheels for the transfer engine and store; the PG gap is owned by AMD's RFC #5202 with a working MI355X prototype, and a MOAT PG port would compete with it. The only uncovered piece is mooncake-ep on HIP, which should be coordinated with the #5202 author and Hygon PR #4810 rather than started independently.
