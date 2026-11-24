# nunchaku: gfx90a evidence run scripts (linux-gfx90a, 2026-11-24)

Scripts of the gfx90a evidence run of the CDNA f16/bf16 MFMA path at 75e2528 in `notes.md`. `mma_dump.py` is in `notes.md`; `f16_dump.py`, `reldiff.py`, `speed.py` and `build.sh` are in `notes-rdna-scripts.md`; `test_cdna.hip` is in `notes-gfx950-scripts.md` (all unchanged). Run the Python ones from the fork root with the extension built in place and `PYTHONPATH=.`.

## denorm.hip

Smallest reproducer of the gfx90a f16 MFMA subnormal-input flush (no nunchaku code). Build: `hipcc -O3 --offload-arch=gfx90a denorm.hip -o denorm`. On gfx90a it prints 1024/1024 outputs wrong (0 instead of 4a) for A = 2^-20 and 2^-24, and 0/1024 for A = 2^-14. The same program with `__builtin_amdgcn_mfma_f32_16x16x4bf16_1k` and bf16 operands gives the same picture at 2^-130 (bf16 subnormal) against 2^-126.

```cpp
// One v_mfma_f32_16x16x4f16 (gfx90a; four 16x16x4 blocks) with every A = 2^-20 (an f16 subnormal), every B = 1, C = 0.
// Exact result: 4 * 2^-20 = 3.815e-06 in every output.  Also the same with A = 2^-14 (smallest normal).
#include <hip/hip_runtime.h>
#include <cstdio>
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
typedef float float16_ __attribute__((ext_vector_type(16)));
__global__ void k(float *out, float a) {
    half4 A = {(_Float16)a, (_Float16)a, (_Float16)a, (_Float16)a};
    half4 B = {(_Float16)1, (_Float16)1, (_Float16)1, (_Float16)1};
    float16_ C = {};
    float16_ D = __builtin_amdgcn_mfma_f32_16x16x4f16(A, B, C, 0, 0, 0);
    for (int i = 0; i < 16; i++) out[threadIdx.x * 16 + i] = D[i];
}
int main() {
    float *d, h[1024];
    hipMalloc(&d, sizeof(h));
    for (float a : {0x1p-20f, 0x1p-14f, 0x1p-24f}) {
        hipLaunchKernelGGL(k, 1, 64, 0, 0, d, a);
        hipMemcpy(h, d, sizeof(h), hipMemcpyDeviceToHost);
        int wrong = 0;
        for (float v : h) wrong += v != 4 * a;
        printf("A = %a (f16 %s): expected %g, got %g (lane 0); %d/1024 outputs wrong\n", a, a < 0x1p-14f ? "subnormal" : "normal", 4 * a, h[0], wrong);
    }
    return 0;
}
```

## test_cdna.hip with FTZ_INPUTS

`test_cdna.hip` with one line changed, so that `FTZ_INPUTS=1` sets every f16 test input whose f16 value is subnormal to a signed zero before both the device and the host reference see it (bf16 inputs untouched):

```cpp
    auto enc = [&](float x) { uint16_t v = bf ? f2bf(x) : f2h(x); if (getenv("FTZ_INPUTS") && !bf && (v & 0x7c00) == 0) v &= 0x8000; return v; };
```

## gemm_ftz.py

The 10 float16 W4A4 GEMM cases of `mma_dump.py` (same seeds and inputs), with (`ftz`) or without (`plain`) the LoRA-up inputs whose float16 value is subnormal zeroed beforehand. Run with each build and compare with `ulp_where.py`.

```python
"""The 10 float16 W4A4 GEMM cases of mma_dump.py, with (ftz) or without (plain) the LoRA-up inputs whose float16 value
is subnormal set to zero beforehand.  python gemm_ftz.py plain|ftz OUT.pt   (fork root, PYTHONPATH=.)"""
import sys

import torch

from nunchaku.lora.flux.packer import NunchakuWeightPacker
from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda
from tests.ops.test_gemm_w4a4 import pack_act, pack_ascales

ftz = sys.argv[1] == "ftz"


def flush(t):  # zero the values whose float16 rounding is subnormal (|x| < 2^-14 after rounding)
    return torch.where(t.half().float().abs() < 2.0**-14, torch.zeros_like(t), t) if ftz else t


G, dtype, packer, res = 64, torch.float16, NunchakuWeightPacker(bits=4), {}
for act_unsigned in (False, True):
    for m, n, k in [(256, 128, 128), (512, 384, 1024), (256, 3072, 3072), (1024, 3072, 768), (4352, 256, 3072)]:
        g = torch.Generator().manual_seed(m + 3 * n + 7 * k + int(act_unsigned))
        lo, hi = (0, 16) if act_unsigned else (-8, 8)
        qact = torch.randint(lo, hi, (m, k), dtype=torch.int32, generator=g)
        qwgt = torch.randint(-8, 8, (n, k), dtype=torch.int32, generator=g)
        qact[:3] = hi - 1
        qact[3:5] = lo
        qwgt[:2] = -8
        qwgt[2:4] = 7
        ascales = (torch.rand(m, k // G, generator=g) * 0.02 + 0.001).to(dtype)
        wscales = (torch.rand(n, k // G, generator=g) * 0.02 + 0.001).to(dtype)
        lora_up = flush((torch.randn(n, 32, generator=g) * 0.1).to(dtype))
        lora_act = flush(torch.randn(m, 32, generator=g) * 0.1)
        bias = (torch.randn(n, generator=g) * 0.1).to(dtype)
        out = torch.empty(m, n, dtype=dtype, device="cuda")
        svdq_gemm_w4a4_cuda(act=pack_act(qact).cuda(), wgt=packer.pack_weight(qwgt).cuda(), out=out,
                            ascales=pack_ascales(ascales).cuda(), wscales=packer.pack_scale(wscales, group_size=G).cuda(),
                            lora_act_in=lora_act.cuda(), lora_up=packer.pack_lowrank_weight(lora_up, down=False).cuda(),
                            bias=packer.pack_scale(bias, group_size=-1).cuda(), act_unsigned=act_unsigned)
        torch.cuda.synchronize()
        res[f"gemm_{dtype}_{act_unsigned}_{m}x{n}x{k}"] = (out.cpu(),)
torch.save(res, sys.argv[2])
```

## quant_ftz.py

The float16 quantize case of `mma_dump.py` with (`ftz`) or without (`plain`) subnormal float16 inputs zeroed; compare the saved `lora_act_out` of two builds with `maxrel.py` (max |diff| / max |x| over differing fp32 outputs).

```python
"""The float16 quantize case of mma_dump.py, with (ftz) or without (plain) the subnormal float16 values of x and
lora_down zeroed beforehand; prints nothing, saves lora_act_out.  python quant_ftz.py plain|ftz OUT.pt"""
import sys

import torch

from nunchaku.lora.flux.packer import NunchakuWeightPacker
from nunchaku.ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda

ftz, dtype, packer = sys.argv[1] == "ftz", torch.float16, NunchakuWeightPacker(bits=4)


def flush(t):
    return torch.where(t.float().abs() < 2.0**-14, torch.zeros_like(t), t) if ftz else t


x = flush((torch.randn(512, 3072, generator=torch.Generator().manual_seed(5)) * 2).to(dtype)).cuda()
lora_down = flush((torch.randn(32, 3072, generator=torch.Generator().manual_seed(6)) * 0.1).to(dtype))
smooth = (torch.rand(3072, generator=torch.Generator().manual_seed(7)) + 0.5).to(dtype)
q, s, la = svdq_quantize_w4a4_act_fuse_lora_cuda(x, lora_down=packer.pack_lowrank_weight(lora_down, down=True).cuda(),
                                                 smooth=packer.pack_scale(smooth, group_size=-1).cuda())
torch.cuda.synchronize()
torch.save({"quantize_float16": (q.cpu(), s.cpu(), la.cpu())}, sys.argv[2])
```

## maxrel.py

```python
import sys, torch
a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
w = 0.0
for n in a:
    for x, y in zip(a[n] if isinstance(a[n], tuple) else (a[n],), b[n] if isinstance(b[n], tuple) else (b[n],)):
        if x.dtype == torch.float32 and not torch.equal(x, y):
            w = max(w, ((x.double() - y.double()).abs().max() / x.double().abs().max()).item())
print(f"max |diff| / max |x| over differing fp32 outputs: {w:.3g}")
```

## ulp_where.py

Where the ulp gaps between two `mma_dump.py` / `f16_dump.py` / `gemm_ftz.py` dumps sit.

```python
"""For each differing 16-bit output of two mma_dump/f16_dump files: max ulp gap, the magnitude where it sits,
the largest gap among elements with |ref| >= 1% of max |ref|, and max |diff| / max |ref|."""
import sys, torch
a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
def ordered(t):
    i = t.contiguous().view(torch.int16).int()
    return torch.where(i < 0, -(i & 0x7FFF), i)
for name in a:
    va, vb = a[name], b[name]
    outs = [(va[0], vb[0])] if name.startswith(("attention", "awq")) else list(zip(va if isinstance(va, tuple) else (va,), vb if isinstance(vb, tuple) else (vb,)))
    for i, (x, y) in enumerate(outs):
        if x.dtype not in (torch.float16, torch.bfloat16) or torch.equal(x, y):
            continue
        fin = torch.isfinite(x) & torch.isfinite(y)
        d = (ordered(x) - ordered(y)).abs()[fin]
        xf = x[fin].double(); yf = y[fin].double()
        j = d.argmax()
        big = xf.abs() >= 0.01 * xf.abs().max()
        print(f"{name}[{i}]: max {d.max().item()} ulp at |x|={xf[j].abs().item():.3g} (max |x| {xf.abs().max().item():.3g}); "
              f"max {d[big].max().item()} ulp where |x| >= 1% of max; max|diff|/max|x| {((xf-yf).abs().max()/xf.abs().max()).item():.3g}; {(d!=0).sum().item()}/{d.numel()} differ")
```

## fp4_refuse.py

FP4 refusal checks (an independent equivalent of the gfx942 run's script; this one calls the FP4 test functions directly, bypassing their skip).

```python
"""How a GPU without NVFP4 kernels refuses FP4.  From the fork root, PYTHONPATH=.

    python fp4_refuse.py check   # Python-level checks: precision choice, checkpoint compatibility, test skip
    python fp4_refuse.py gemm    # the FP4 GEMM kernel called directly (expected: device trap)
    python fp4_refuse.py quant   # the FP4 activation quantize kernel called directly (expected: device trap)
"""
import sys

import torch

mode = sys.argv[1]
if mode == "check":
    from nunchaku.utils import ROCM_FP4_ARCHS, check_hardware_compatibility, get_precision, get_rocm_arch

    print("arch", get_rocm_arch(), "ROCM_FP4_ARCHS", ROCM_FP4_ARCHS)
    print("get_precision('auto') ->", get_precision("auto"))
    check_hardware_compatibility({"weight": {"dtype": "int4"}})
    print("check_hardware_compatibility(int4): accepted")
    try:
        check_hardware_compatibility({"weight": {"dtype": "fp4_e2m1_all"}})
        print("check_hardware_compatibility(fp4): ACCEPTED")
    except ValueError as e:
        print("check_hardware_compatibility(fp4): ValueError:", e)
    from tests.ops.test_gemm_w4a4_fp4 import fp4_supported

    print("tests.ops.test_gemm_w4a4_fp4.fp4_supported() ->", fp4_supported())
elif mode == "gemm":
    from tests.ops.test_gemm_w4a4_fp4 import test_gemm_w4a4_fp4

    test_gemm_w4a4_fp4(256, 256, 256, torch.bfloat16)
    print("FP4 GEMM RETURNED")
elif mode == "quant":
    from tests.ops.test_gemm_w4a4_fp4 import test_quantize_w4a4_act_fp4

    test_quantize_w4a4_act_fp4(256, 256, torch.bfloat16)
    print("FP4 QUANTIZE RETURNED")
```
