# W4A16 regression for the already-ported path: WQLinear GEMV (M < 8) and GEMM (M >= 8) vs a
# PyTorch fake-quant reference built from the same scales/zeros.
import sys, torch
sys.path.insert(0, sys.argv[1])
from awq.quantize.qmodule import WQLinear
torch.manual_seed(0)
res = []
for (IC, OC) in [(256, 128), (768, 768), (768, 3072), (3072, 768), (4096, 11008)]:
    lin = torch.nn.Linear(IC, OC, bias=False).half().cuda()
    w = lin.weight.data.float(); g = 128
    wg = w.reshape(OC, IC // g, g); mx, mn = wg.amax(-1), wg.amin(-1)
    s = ((mx - mn).clamp(min=1e-5) / 15).half().float(); z = torch.round(-mn / s).clamp(0, 15)
    q = torch.round(w / s.repeat_interleave(g, 1) + z.repeat_interleave(g, 1)).clamp(0, 15)
    wref = ((q - z.repeat_interleave(g, 1)) * s.repeat_interleave(g, 1)).half().float()
    # from_linear does not clamp q to [0, 15]; give it weights already on the 4-bit grid
    lin.weight.data = wref.half()
    ql = WQLinear.from_linear(lin, 4, g, scales=s.half(), zeros=z.half())
    for M in (1, 3, 7, 8, 16, 64):
        x = torch.randn(1, M, IC, device="cuda").half()
        y = ql(x).float(); r = x.float() @ wref.t()
        err = (y - r).abs().max().item(); tol = 2e-3 * r.abs().max().item() + 2e-2
        ok = err <= tol and torch.isfinite(y).all().item(); res.append(ok)
        print(f"{'PASS' if ok else 'FAIL'} IC={IC} OC={OC} M={M} max_err={err:.3e} tol={tol:.2e}", flush=True)
print(f"{sum(res)}/{len(res)} passed"); sys.exit(0 if all(res) else 1)
