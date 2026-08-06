# FlashKDA notes

## 2026-06-04 -- Porter analysis (linux-gfx90a)

### Classification: cant-port (CUTLASS/CuTe)

FlashKDA is a CUTLASS/CuTe-based Flash Kimi Delta Attention implementation targeting SM90+ (Hopper/Blackwell) exclusively. The entire device code path uses Hopper-specific features that have no ROCm/HIP equivalent:

- TMA (Tensor Memory Accelerator): `PipelineTmaAsync`, `ClusterTransactionBarrier`, `get_tma_tensor`
- GMMA/wgmma: warpgroup MMA via `cooperative_gemm`, `GMMA::Layout_K_INTER_Atom`
- Thread-block clusters + distributed shared memory: `cluster_sm90.hpp`, `cluster_launch.hpp`
- Warp-specialized async pipelines: `sm90_pipeline.hpp`, `PipelineState`, `NamedBarrier`
- SM90-only PTX: `elect_one_sync`, `mbarrier`

Per PORTING_GUIDE (2026-05-30): "CUTLASS does NOT port to ROCm and never will: do not attempt a CUTLASS->ROCm port or a CUTLASS->ROCm shim. Reimplement against Composable Kernel (CK)."

### What would be required for AMD support

A ground-up reimplementation of the KDA forward kernel using Composable Kernel (ck_tile preferred):
1. Understand the chunked delta-rule recurrence algorithm from fwd_kernel1.cuh (570 LOC) + fwd_kernel2.cuh (834 LOC) + the torch_ref.py reference
2. Translate the tiling, MMA, epilogue (gate activation, qk l2norm, beta sigmoid, A_log/dt_bias, lower_bound clamp, carried state) to CK/MFMA primitives
3. Replace TMA with explicit global-to-LDS async copy patterns
4. Replace warp-specialization with CK's producer/consumer staging
5. Handle wave64 correctly (the original code bakes in 32-lane warp assumptions)

This is multi-day kernel engineering work, not a translation pass.

### Decision

Blocked as `cant-port`. The pattern matches SpargeAttn, CUDA-L2, CPM.cu, NATTEN, llmq, and mirage -- all blocked for the same CUTLASS/CuTe reason. An AMD-native CK reimplementation could be valuable (linear attention on AMD GPUs has real demand) but falls outside the scope of a mechanical port pipeline.

The plan.md explicitly raised this as open question #1: whether to defer to a "CK reimplementation backlog." This decision confirms that deferral.
