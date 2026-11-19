# Usage: python compare.py a.pt b.pt  -- per-tensor max/mean abs diff and relative L2 error
import sys
import torch
a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
worst = 0.0
for k in a:
    x, y = a[k].double(), b[k].double()
    if x.shape != y.shape:
        print(f'{k:45s} SHAPE {tuple(x.shape)} vs {tuple(y.shape)}')
        continue
    bad = (~torch.isfinite(y)).sum().item()
    d = (x - y).abs()
    rel = (d.norm() / x.norm().clamp_min(1e-30)).item()
    worst = max(worst, rel)
    frac = (d > 1e-4 * x.abs().max().clamp_min(1e-30)).double().mean().item()
    print(f'{k:45s} max={d.max().item():.3e} mean={d.mean().item():.3e} relL2={rel:.3e} frac>1e-4={frac:.2e} nonfinite={bad} |ref|max={x.abs().max().item():.3e}')
print('worst relL2', worst)
