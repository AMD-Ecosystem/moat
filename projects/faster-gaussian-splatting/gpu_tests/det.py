# Usage: python det.py <FasterGSCudaBackend dir> -- run-to-run determinism of both forward paths on large scenes
import sys
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, __file__.rsplit('/', 1)[0])
import torch
from FasterGSCudaBackend.torch_bindings import diff_rasterize, rasterize
from run_outputs import scene, settings
for n, seed, w, h in [(200000, 7, 1920, 1080), (100000, 8, 1920, 1080), (50000, 9, 1280, 720)]:
    params = scene(n, seed); s = settings(w, h, 16)
    imgs, infs = [], []
    for r in range(4):
        with torch.no_grad():
            imgs.append(diff_rasterize(*params, torch.zeros(2, n, device='cuda'), s))
            infs.append(rasterize(*params, s, to_chw=True, clamp_output=False))
    dt = max((imgs[0] - i).abs().max().item() for i in imgs[1:])
    di = max((infs[0] - i).abs().max().item() for i in infs[1:])
    npx = max(((imgs[0] - i).abs().sum(0) > 0).sum().item() for i in imgs[1:])
    print(f'n={n} {w}x{h}: train fwd max run-to-run diff {dt:.3e} ({npx} px), inference {di:.3e}')
