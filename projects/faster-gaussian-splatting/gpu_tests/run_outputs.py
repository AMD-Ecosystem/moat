# Usage: python run_outputs.py <FasterGSCudaBackend dir> <out.pt>
# Runs every rasterizer entry point on fixed scenes and saves the results, so two builds
# (pre-merge port head vs merged head) can be compared output for output.
import sys, math
if __name__ == '__main__':
    sys.path.insert(0, sys.argv[1])
import torch
from FasterGSCudaBackend.torch_bindings import diff_rasterize, rasterize, update_pruning_scores, RasterizerSettings
import inspect
from FasterGSCudaBackend.torch_bindings import update_3d_filter

dev = 'cuda'


def scene(n, seed, spread=1.0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    xy = (torch.rand(n, 2, generator=g) * 2 - 1) * 2.2 * spread
    # distinct depths: the depth sort is keyed on depth alone, so tied depths order nondeterministically
    z = ((torch.randperm(n, generator=g).double() + 0.5) / n * 6 + 2).float().unsqueeze(1)
    means = torch.cat([xy * z / 4, z], 1)
    scales = torch.log(torch.rand(n, 3, generator=g) * 0.08 + 0.005)
    rot = torch.randn(n, 4, generator=g)
    opa = torch.randn(n, 1, generator=g) * 1.5
    sh0 = torch.randn(n, 1, 3, generator=g) * 0.5
    shr = torch.randn(n, 15, 3, generator=g) * 0.2
    return [t.to(dev).contiguous() for t in (means, scales, rot, opa, sh0, shr)]


def settings(w, h, sh_bases, aa=True):
    w2c = torch.eye(4, device=dev)[:3, :].contiguous()
    f = 0.9 * w
    return RasterizerSettings(
        w2c=w2c, cam_position=torch.zeros(3, device=dev), bg_color=torch.tensor([0.1, 0.2, 0.3], device=dev),
        active_sh_bases=sh_bases, width=w, height=h, focal_x=f, focal_y=f,
        center_x=w / 2, center_y=h / 2, near_plane=0.2, far_plane=100.0, proper_antialiasing=aa)


CASES = [
    # name, n, seed, w, h, sh_bases, aa, spread
    ('n37_256', 37, 1, 256, 256, 1, True, 1.0),
    ('n1000_256_sh4', 1000, 2, 256, 256, 4, True, 1.0),
    ('n1001_800x600_sh16', 1001, 3, 800, 600, 16, True, 1.0),
    ('n5000_257x131_sh9', 5000, 4, 257, 131, 9, False, 1.0),
    ('n20000_800x600_sh16', 20000, 5, 800, 600, 16, True, 1.0),
    ('n20000_offscreen_640x480', 20000, 6, 640, 480, 16, True, 2.5),
    ('n200000_1920x1080_sh16', 200000, 7, 1920, 1080, 16, True, 1.0),
]

def main():
    out = {}
    for name, n, seed, w, h, shb, aa, spread in CASES:
        params = scene(n, seed, spread)
        s = settings(w, h, shb, aa)
        with torch.no_grad():
            out[name + '/inference'] = rasterize(*params, s, to_chw=True, clamp_output=False).cpu()
        leaves = [p.clone().requires_grad_(True) for p in params]
        dinfo = torch.zeros(2, n, device=dev)
        img = diff_rasterize(*leaves, dinfo, s)
        g = torch.Generator(device='cpu').manual_seed(100 + seed)
        weight = torch.randn(img.shape, generator=g).to(dev)
        (img * weight).sum().backward()
        torch.cuda.synchronize()
        out[name + '/train_image'] = img.detach().cpu()
        for pn, p in zip(['means', 'scales', 'rotations', 'opacities', 'sh0', 'shrest'], leaves):
            out[name + '/grad_' + pn] = p.grad.detach().cpu()
        out[name + '/dinfo'] = dinfo.cpu()
        scores = torch.zeros(n, device=dev)
        update_pruning_scores(scores, *params, s)
        out[name + '/pruning'] = scores.cpu()
        # 3d filter (far_plane argument exists only on the merged head)
        filt = torch.full((n,), float("inf"), device=dev)
        vis = torch.zeros(n, dtype=torch.bool, device=dev)
        kw = dict(width=w, height=h, focal_x=s.focal_x, focal_y=s.focal_y, center_x=s.center_x, center_y=s.center_y,
                  near_plane=s.near_plane, clipping_tolerance=0.15, distance2filter=0.2)
        if 'far_plane' in inspect.signature(update_3d_filter).parameters:
            kw['far_plane'] = 1e6
        update_3d_filter(params[0], s.w2c, filt, vis, **kw)
        out[name + '/filter3d'] = filt.cpu()
        out[name + '/filter3d_vis'] = vis.cpu()

    torch.save(out, sys.argv[2])
    print('saved', len(out), 'tensors to', sys.argv[2])


if __name__ == '__main__':
    main()
