# sparser-faster-llms Port Plan

## Project

- **Name**: sparser-faster-llms
- **Upstream**: https://github.com/SakanaAI/sparser-faster-llms
- **Default branch**: master
- **Description**: Custom CUDA kernels for sparse LLM inference and training using the TwELL packing format, designed for H100 GPUs.

## Existing AMD support

**Status**: None found

Checked:
- Upstream docs: `grep -rniE 'amd|rocm|hip|gfx[0-9]'` on README.md and docs/ returned no hits
- Forks: `gh api repos/SakanaAI/sparser-faster-llms/forks` shows 20+ forks, none with AMD/ROCm/HIP in name or description
- Web search: "<project> ROCm", "<project> AMD GPU", "TwELL sparse kernel AMD ROCm" -- no results
- Issues/PRs: no AMD-related discussions

**Merge policy**: Standard -- upstream appears to accept contributions directly.

**Decision**: BLOCKED -- the port is infeasible without a kernel rewrite (see Risk List).

## Build classification

**Type**: PyTorch extension (torch-extension)

**Evidence**:
- `custom_models/twell_modules/twell.py` line 9: `from torch.utils.cpp_extension import CUDA_HOME, load`
- `custom_models/twell_modules/twell.cpp` uses `#include <torch/extension.h>` and `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`
- Uses `torch.utils.cpp_extension.load()` for JIT compilation
- No CMakeLists.txt or setup.py; extension is built dynamically at runtime

## Port strategy

**Recommended**: BLOCKED (cannot proceed with standard Strategy B or Strategy A)

**Rationale**: The TwELL kernels are not portable CUDA code that can be hipified. They use NVIDIA Hopper (SM90+) hardware-specific features that have **no AMD equivalent**:

1. **WGMMA (Warpgroup Matrix Multiply Accumulate)**: The `wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16` PTX instruction is Hopper's tensor core instruction for warpgroup-level matrix operations. AMD CDNA has MFMA but with different semantics, shapes, and programming model.

2. **TMA (Tensor Memory Accelerator)**: The kernels use `cp.async.bulk.tensor.{2d,3d}` instructions for hardware-accelerated tensor copies with multicast to thread clusters. AMD has no equivalent hardware TMA unit.

3. **Thread Block Clusters**: The `__cluster_dims__` attribute and `barrier.cluster.*` instructions coordinate multiple thread blocks. AMD has no cluster abstraction.

4. **Dynamic Register Allocation**: `setmaxnreg.{inc,dec}.sync.aligned.u32` instructions dynamically adjust register allocation per warpgroup. AMD has no runtime register reallocation.

5. **Asynchronous Barriers (mbarrier)**: `mbarrier.init`, `mbarrier.arrive.expect_tx`, `mbarrier.try_wait.parity` instructions for producer/consumer synchronization with TMA. AMD barriers are different.

A mechanical HIP translation will not compile -- these are not CUDA runtime API calls that hipify can map; they are inline PTX assembly for hardware that does not exist on AMD.

## CUDA surface inventory

### Files

| File | Lines | Description |
|------|-------|-------------|
| `matmul_d2t.cu` | ~1200 | Dense-to-TwELL matmul with WGMMA + TMA |
| `matmul_t2d.cu` | ~200 | TwELL-to-dense downprojection |
| `matmul_gated_t2d.cu` | ~2000 | Gated TwELL-to-dense projection |
| `mlp_twell.cu` | ~230 | Fused MLP wrappers |
| `hilbert.cu` | ~140 | Host-only Hilbert curve scheduling |
| `twell.cpp` | ~890 | PyTorch/pybind11 bindings |
| `twell.py` | ~1100 | Python API and JIT loader |

### Hopper-specific features (fatal for AMD)

- `wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16` (WGMMA tensor core op)
- `cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes` (TMA 2D load)
- `cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster` (TMA multicast)
- `cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group` (TMA 3D store)
- `mbarrier.init.shared::cta.b64`, `mbarrier.arrive.expect_tx.release`, `mbarrier.try_wait.parity.acquire` (async barriers)
- `barrier.cluster.arrive`, `barrier.cluster.wait` (cluster sync)
- `setmaxnreg.dec.sync.aligned.u32`, `setmaxnreg.inc.sync.aligned.u32` (dynamic register allocation)
- `__cluster_dims__()` kernel attribute
- `%clusterid.x`, `%cluster_ctarank` PTX registers
- `mapa.shared::cluster.u32` (cross-CTA shared memory mapping)
- `CUtensorMap` descriptor type for TMA

### Standard CUDA features (portable if the above were not present)

- `__nv_bfloat16`, `__nv_bfloat162` types (mapped to `__hip_bfloat16`)
- `__shfl_sync`, `__shfl_xor_sync` warp shuffles (portable)
- `__ldcs`, `__stcs` cache-streaming loads/stores (have HIP equivalents)
- `__syncthreads()` (portable)
- `cudaMalloc`, `cudaMemcpy`, streams (hipified)

### cuBLAS / library usage

- `#include <cublas_v2.h>` is included in `matmul_d2t.cu` but appears to be a leftover include -- no cuBLAS calls are made; the kernels are hand-written.

### Warp size

- `#define WARP_SIZE 32` is used in `matmul_t2d.cu`, `matmul_gated_t2d.cu`
- `#define WARP_GROUP_SIZE 128` (4 warps per warpgroup) in `matmul_d2t.cu`
- Hardcoded warp-width-32 shuffles with `0xFFFFFFFFu` masks

For a future AMD-native rewrite, wave64 (gfx90a) vs wave32 (gfx1100) would need conditional handling.

## Risk list

1. **FATAL: Hopper-only PTX** -- WGMMA, TMA, clusters, mbarrier, setmaxnreg have no HIP/ROCm equivalent. A port requires a complete AMD-native kernel rewrite using rocWMMA/MFMA, explicit global-to-LDS copies, and standard thread block synchronization.

2. **FATAL: Architecture target** -- The code explicitly targets SM90a (`os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")` in `twell.py`). Even setting aside the PTX, the code is written for Hopper's SM count (132), warpgroup model, and memory hierarchy.

3. **Kernel complexity** -- The D2T kernel alone is ~1200 lines with complex producer/consumer pipelines, Hilbert-curve scheduling, and multi-stage buffering. An AMD rewrite is a substantial engineering effort requiring deep expertise in both Hopper and CDNA/RDNA matrix cores.

4. **No fallback path** -- Unlike many CUDA projects, there is no scalar or portable fallback. The extension either uses the Hopper kernels or fails. The sparse model training (torch path) is portable, but the TwELL inference kernels are the project's core contribution.

5. **Sparse format coupling** -- The TwELL packed format (uint32 with embedded index+value pairs) and the tile shapes (128x256x64) are co-designed with WGMMA's 64x256x16 native shape. An AMD kernel would need to rethink the format or accept suboptimal tile shapes.

## Alternative: AMD-native sparse kernel development

If this project were to be made AMD-compatible, the correct approach is **not** a port but a **parallel AMD implementation**:

1. Use rocWMMA/MFMA for matrix operations (similar to how Composable Kernel does)
2. Use explicit `__builtin_amdgcn_*` intrinsics for LDS and global memory operations
3. Design tile shapes around AMD's MFMA (e.g., 16x16, 32x32) rather than WGMMA's 64x256
4. Implement cooperative group synchronization using standard HIP barriers

This is a research/development project, not a porting task.

## File-by-file change list

N/A -- port is blocked.

## Build commands

N/A -- port is blocked.

For reference, the CUDA build (on Hopper) is:
```bash
cd /var/lib/jenkins/moat/projects/sparser-faster-llms/src
pip install torch==2.9.1
pip install -r requirements.txt
# Extension is JIT-compiled on first import of twell module
python -c "from custom_models.twell_modules import twell; twell.get_ext(algorithms=['twell_mlp'])"
```

## Test plan

N/A -- port is blocked.

For reference, the upstream tests are:
```bash
# Inference benchmark (requires sparse checkpoint + H100)
python benchmark_inference.py --model-path SakanaAI/SparseLM1.5B --reps 50

# Training (uses torch sparse path, not TwELL kernels -- this IS portable)
./launch.sh 8 sparsity_gated_1p5b zero1
```

The sparse training path (`train.py`, `sparse_models.py`) uses standard PyTorch and is fully portable to ROCm. Only the TwELL inference kernels (`custom_models/twell_modules/`) are Hopper-specific.

## Open questions

1. **Is a partial port valuable?** The training code is portable. The TwELL inference kernels are not. Would it be useful to port just the training path and document that TwELL inference requires NVIDIA Hopper? This would allow users to train sparse models on AMD and export them for inference on NVIDIA.

2. **AMD-native kernel interest?** Is there interest in developing AMD-native sparse MLP kernels using rocWMMA/MFMA/Composable Kernel? This would be a new development effort, not a port.

3. **Upstream communication?** Should we open an issue on the upstream repo documenting the Hopper-only limitation and expressing interest in AMD compatibility? The authors (Sakana AI + NVIDIA collaboration) may have thoughts on portability.

## Disposition

**BLOCKED**: The TwELL CUDA kernels use NVIDIA Hopper PTX instructions (WGMMA, TMA, thread clusters) that have no AMD equivalent. A mechanical HIP translation is impossible. Supporting AMD would require a ground-up kernel rewrite using AMD-native matrix intrinsics (rocWMMA/MFMA), which is a research project rather than a porting task.

The sparse training path (standard PyTorch) is portable; only the TwELL inference kernels are blocked.
