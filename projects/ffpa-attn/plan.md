# Plan: ffpa-attn ROCm Port

## Project

- Name: ffpa-attn
- Upstream: https://github.com/xlite-dev/ffpa-attn
- Default branch: main
- Commit: 67e3fff78e27a5ae6762aa177318d6f5a263625d

## Existing AMD support

**Status: None found -- proceed with port**

Web searches for "ffpa-attn ROCm", "ffpa-attn AMD GPU", "ffpa-attn HIP", "ffpa-attn MI300/gfx9" returned no results specific to this project. The README explicitly lists support for NVIDIA architectures only: Ampere (sm80), Ada (sm89), Hopper (sm90), and Blackwell (sm100/sm120).

Grep of upstream docs (`grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/`) returned no matches.

GitHub forks checked via `gh api repos/xlite-dev/ffpa-attn/forks`:
- 19 forks found, none with AMD/ROCm/HIP in name or description
- BruceXcluding/ffpa-attn-mma checked -- no AMD-specific branches, standard NVIDIA-only fork

**Decision**: No existing AMD support. A HIP port adds value for AMD GPU users who need FlashAttention-2 style attention with large headdim (>256) support.

## Build classification

**pytorch extension (Strategy B)**

Evidence:
- `setup.py` lines 112-144: Uses `torch.utils.cpp_extension.CUDAExtension` and `BuildExtension`
- `pyproject.toml` exists with torch dependency
- `env.py` handles CUDA arch list via `FFPA_BUILD_ARCH` env var
- Build invoked via: `pip install -e .` or `ENABLE_FFPA_CUDA_IMPL=1 pip install -e .`

The project has THREE backends:
1. **Triton backend** (Python/Triton, works on AMD via Triton-AMD -- NO CUDA porting needed)
2. **CuTeDSL backend** (NVIDIA CuTeDSL, sm90+ only -- NOT portable, skip)
3. **CUDA backend** (hand-written PTX/MMA kernels -- requires porting, forward-only)

## Port strategy

**Recommendation: Port Triton backend first (low effort), assess CUDA backend value later**

### Primary path: Triton backend (already AMD-capable)

Triton natively supports AMD GPUs via Triton-AMD. The project's Triton backend (`src/ffpa_attn/triton/`) should work on ROCm PyTorch with minimal or no changes. This is the fastest path to AMD support covering both forward AND backward passes.

Validation steps:
1. Install ROCm PyTorch
2. `pip install -e .` (pure Python/Triton install, CUDA extension disabled by default)
3. Run `pytest tests/test_ffpa_fwd.py tests/test_ffpa_bwd.py`

### Secondary path: CUDA backend port (high effort, questionable value)

The CUDA backend (`csrc/cuffpa/`) uses NVIDIA-specific PTX inline assembly that has NO direct HIP equivalent:
- **MMA intrinsics**: `mma.sync.aligned.m16n8k16.row.col.f16/f32/bf16` -- NVIDIA tensor core ops
- **ldmatrix**: `ldmatrix.sync.aligned.x4.m8n8.shared.b16` -- NVIDIA-specific shared memory matrix load
- **cp.async**: `cp.async.cg.shared.global.L2::128B` -- NVIDIA async copy with L2 hints
- **TMA**: Hopper TMA descriptors (`CUtensorMap`, `cp.async.bulk.tensor`) -- sm90+ only

AMD equivalent would require MFMA/WMMA intrinsics (different register layouts), explicit `__builtin_amdgcn_global_load_lds` for async copies, and completely different data movement patterns. This is NOT a mechanical port -- it is a REWRITE.

**Decision**: Skip CUDA backend port. The Triton backend provides both forward and backward passes with AMD support via Triton-AMD. The CUDA backend only provides forward pass and would require substantial AMD-native kernel development using rocWMMA/MFMA/Composable Kernel -- a significant engineering effort beyond mechanical porting.

## CUDA surface inventory (CUDA backend -- for reference)

| Component | CUDA Usage | HIP/ROCm Equivalent | Status |
|-----------|-----------|---------------------|--------|
| MMA intrinsics | `mma.sync.aligned.m16n8k16` | MFMA/rocWMMA | Rewrite required |
| ldmatrix | `ldmatrix.sync.aligned.x4/x2` | None (manual LDS load) | Rewrite required |
| cp.async | `cp.async.cg/ca.shared.global` | `__builtin_amdgcn_global_load_lds` | Rewrite required |
| TMA | `CUtensorMap`, `cuTensorMapEncodeTiled` | None (sm90+ only) | Skip |
| Warp shuffles | `__shfl_xor_sync`, `__shfl_sync` | `__shfl_xor`, `__shfl` | Direct mapping |
| WARP_SIZE | Hardcoded 32 | 64 on CDNA, 32 on RDNA | Risk: entire kernel design assumes 32 |
| cuda::barrier | C++ barrier for TMA | Not needed (skip TMA path) | N/A |
| `__half` / `__nv_bfloat16` | NVIDIA types | `hip_fp16.h` / `hip_bfloat16.h` | Direct mapping |
| `__fmaf_rn`, `__frcp_rn`, `expf`, `__logf` | Math intrinsics | Same in HIP | Direct mapping |

### Critical warp-size issue

The CUDA backend is designed around `WARP_SIZE=32`:
- `warp.cuh:14`: `#define WARP_SIZE 32`
- All kernel launch bounds, shared memory layouts, and warp shuffles assume 32-lane warps
- On gfx90a (wave64), this would require fundamental redesign of thread-to-data mapping

## Risk list

1. **No risk for Triton path**: Triton-AMD handles codegen
2. **CUDA backend warp-size**: Kernel assumes 32-lane warps throughout; wave64 (gfx90a) incompatible without redesign
3. **CUDA backend MMA intrinsics**: No 1:1 HIP mapping; requires AMD-native MFMA rewrite
4. **CUDA backend ldmatrix**: NVIDIA-specific; no HIP equivalent
5. **CuTeDSL backend**: sm90+ only, uses Hopper-specific features, not portable
6. **Performance**: Even if ported, mechanical translation of NVIDIA-tuned kernels often underperforms AMD-native implementations

## File-by-file change list (Triton validation path)

Likely no source changes needed for Triton backend. Potential issues to verify:

| File | Change | Reason |
|------|--------|--------|
| `src/ffpa_attn/triton/_ffpa_fwd.py` | Verify | Triton should auto-generate AMD code |
| `src/ffpa_attn/triton/_ffpa_bwd.py` | Verify | Triton should auto-generate AMD code |
| `src/ffpa_attn/triton/_ffpa_fwd_sm90.py` | Skip/disable | sm90-specific path |
| `src/ffpa_attn/triton/_ffpa_bwd_sm90.py` | Skip/disable | sm90-specific path |
| `src/ffpa_attn/functional.py` | May need guards | Backend selection logic may need AMD awareness |
| `src/ffpa_attn/autotune.py` | May need tuning | Autotuned configs are NVIDIA-specific |

## Build commands (gfx90a)

### Triton-only install (recommended)
```bash
# Ensure ROCm PyTorch is installed
pip install torch --index-url https://download.pytorch.org/whl/rocm7.0

# Clone and install
git clone https://github.com/xlite-dev/ffpa-attn.git
cd ffpa-attn
pip install -e . --no-build-isolation
```

### With CUDA extension (if attempting CUDA backend port)
```bash
# NOT RECOMMENDED -- requires MFMA rewrite
ENABLE_FFPA_CUDA_IMPL=1 MAX_JOBS=32 pip install -e .
```

## Test plan

### GPU tests (real validation)

```bash
# Core forward/backward tests with Triton backend
pytest tests/test_ffpa_fwd.py -v -x
pytest tests/test_ffpa_bwd.py -v -x

# Autotune tests (may need AMD-specific configs)
pytest tests/test_triton_autotune_mode.py -v -x

# Monkey-patch integration
pytest tests/test_monkey_patch.py -v -x
```

### Non-GPU tests (should not regress)

```bash
# Logger/utility tests
pytest tests/test_logger.py -v
```

### Performance benchmarks (optional, post-validation)

```bash
# Under bench/ directory
python bench/bench_ffpa_fwd.py
python bench/bench_ffpa_bwd.py
```

## Open questions

1. **Triton-AMD autotune configs**: The project uses NVIDIA-specific autotune configurations in `src/ffpa_attn/triton/configs/`. Will these work on AMD, or do we need AMD-specific configs? Triton-AMD should auto-tune, but NVIDIA-specific block sizes may not be optimal.

2. **sm90 code paths**: The project has sm90-specific Triton kernels (`_ffpa_fwd_sm90.py`, `_ffpa_bwd_sm90.py`). These use Hopper TMA via Triton's tma_descriptor API. Need to verify the dispatcher correctly falls back to non-sm90 paths on AMD.

3. **Upstream interest**: Would xlite-dev accept an AMD validation PR showing Triton backend works on ROCm? The CUDA backend is NVIDIA-only by design.

4. **Performance expectations**: The project's 1.5-3x speedup claims are benchmarked on NVIDIA GPUs with NVIDIA-tuned kernels. AMD performance will depend on Triton-AMD codegen quality. Expectation should be correctness first, performance tuning later.

## Summary

This is primarily a **validation project**, not a traditional CUDA-to-HIP port. The Triton backend should work out-of-the-box on ROCm PyTorch via Triton-AMD. The CUDA backend uses deep NVIDIA-specific PTX (tensor core MMA, ldmatrix, cp.async) that would require a ground-up AMD-native rewrite using MFMA/rocWMMA/Composable Kernel -- far beyond mechanical porting.

Recommended approach:
1. Validate Triton backend on gfx90a with ROCm PyTorch
2. Fix any dispatch logic that incorrectly selects NVIDIA-specific paths
3. Document that CUDA/CuTeDSL backends are NVIDIA-only
4. Potentially contribute AMD-specific autotune configs

This provides AMD users with working FlashAttention-2-style attention for large headdim (>256) via Triton, covering both forward and backward passes.
