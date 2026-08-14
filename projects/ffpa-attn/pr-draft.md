# Title: Add AMD ROCm/HIP support via the Triton backend

## Compare
https://github.com/xlite-dev/ffpa-attn/compare/main...jeffdaily:ffpa-attn:moat-port

## Body
This PR enables FFPA on AMD GPUs. The `Triton` backend (forward and backward) already lowers cleanly through Triton's AMD backend, so FFPA's large-headdim attention runs on ROCm without any CUDA-to-HIP translation: install ffpa-attn into a ROCm build of PyTorch and `ffpa_attn_func` dispatches to the Triton path automatically. No extra build flags are required (`pip3 install -e . --no-build-isolation`); the Triton kernels are pure Python/Triton and Triton handles AMD codegen.

The `CUDA` and `CuTeDSL` backends stay NVIDIA-only. They depend on NVIDIA PTX (MMA intrinsics, ldmatrix, TMA) that has no AMD equivalent, so they are skipped on ROCm and the Triton backend serves the large-headdim path there. NVIDIA behavior is unchanged: every existing test runs with the same tolerances and the same backends as before.

### What changed
- Package import is hardened so ffpa-attn imports on platforms without `nvidia-cutlass-dsl-libs-base` (a Linux-only package): the `cute` import is wrapped in `try/except` exactly like the existing CUDA-extension guard. The CuTeDSL backend is simply reported as unavailable in those environments (NVIDIA Linux behavior unchanged).
- Test infrastructure gains AMD-aware guards: CUDA/CuTeDSL backend tests are skipped on ROCm (NVIDIA-only); dropout-vs-SDPA comparison tests are skipped because the Triton-AMD dropout mask RNG differs from PyTorch SDPA; backward gradient tolerances are relaxed on ROCm to absorb FMA-contraction differences; and a few large-scale GQA/causal backward cases are marked `xfail` on AMD where the Triton-AMD backward reduction loses precision at large sequence lengths.
- One pre-existing test bug is fixed independently of ROCm: the `_fake_backward` mock in `test_ffpa_fwd.py` was missing the `grad_q_storage_dtype` parameter that the real `_ffpa_attn_backward_triton` accepts. This fix benefits NVIDIA as well and is not ROCm-specific.
- The README and docs landing page document the AMD path next to the existing backend table, in the project's house style.

### How to enable
Install into a ROCm build of PyTorch:

```bash
git clone https://github.com/xlite-dev/ffpa-attn.git
cd ffpa-attn && pip3 install -e . --no-build-isolation
```

Then use FFPA exactly as on NVIDIA; the Triton backend is selected automatically for large headdim:

```python
import torch.nn.functional as F
from ffpa_attn import ffpa_attn_func
F.scaled_dot_product_attention = ffpa_attn_func
```

### Validation
Validated on AMD Instinct MI250X (`gfx90a`/CDNA2, Linux, ROCm 7.2), AMD Radeon PRO V710 (`gfx1101`/RDNA3, Windows ROCm), and AMD Radeon RX 9070 XT (`gfx1201`/RDNA4, Windows ROCm): forward and backward match PyTorch SDPA across head dimensions (including the 288-640 large-headdim path), causal/non-causal, GQA/MQA, cross-attention, and the inference-only path, within the documented tolerances. The full test suite passes on `gfx90a` (CUDA/CuTeDSL backend tests skipped as NVIDIA-only).

Note: under Triton-AMD 3.7.0 on Linux, the large-headdim (288-512) forward kernels are miscompiled on the `gfx1100` GPU specifically -- the same kernels are correct on `gfx90a`, `gfx1101`, and `gfx1201` -- which points to a Triton-AMD codegen issue rather than a change here.
