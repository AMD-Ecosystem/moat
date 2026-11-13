# Usage: python train_check.py <FasterGSCudaBackend dir>
# End-to-end: fit a random Gaussian scene to a target rendering of another scene with FusedAdam,
# injecting MCMC noise, using the absolute-gradient densification buffer. Checks FusedAdam against
# torch.optim.Adam, relocation_adjustment sanity, and that the loss falls.
import sys
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, __file__.rsplit('/', 1)[0])
import torch
from FasterGSCudaBackend.torch_bindings import diff_rasterize, rasterize, FusedAdam, add_noise, relocation_adjustment
from run_outputs import scene, settings

ok_all = True


def check(name, ok, detail=''):
    global ok_all
    ok_all &= bool(ok)
    print(f'{"PASS" if ok else "FAIL"}  {name}  {detail}')


# FusedAdam vs torch Adam on the same gradients
torch.manual_seed(0)
p1 = torch.randn(100003, 3, device='cuda', requires_grad=True)
p2 = p1.detach().clone().requires_grad_(True)
o1 = FusedAdam([{'params': [p1], 'lr': 1e-2}], lr=0.0, eps=1e-15)
o2 = torch.optim.Adam([p2], lr=1e-2, eps=1e-15)
for _ in range(20):
    g = torch.randn_like(p1)
    p1.grad = g.clone(); p2.grad = g.clone()
    o1.step(); o2.step()
d = (p1 - p2).abs().max().item()
check('FusedAdam matches torch.optim.Adam over 20 steps', d < 1e-5, f'max abs diff {d:.2e}')

# relocation adjustment
opa = torch.rand(1000, 1, device='cuda') * 0.98 + 0.01
sc = torch.rand(1000, 3, device='cuda') * 0.1 + 0.01
cnt = torch.randint(1, 30, (1000,), device='cuda', dtype=torch.int64)
na, ns = relocation_adjustment(opa, sc, cnt)
check('relocation_adjustment finite, opacity in (0,1], scale shrinks',
      torch.isfinite(na).all() and torch.isfinite(ns).all() and (na > 0).all() and (na <= 1).all() and (ns <= sc + 1e-6).all())

# fit
target_params = scene(3000, 11)
s = settings(320, 240, 16)
with torch.no_grad():
    target = rasterize(*target_params, s, to_chw=True, clamp_output=True)
params = [p.clone().requires_grad_(True) for p in scene(3000, 12)]
lrs = [2e-3, 5e-3, 1e-3, 5e-2, 2.5e-3, 1e-4]
opt = FusedAdam([{'params': [p], 'lr': lr} for p, lr in zip(params, lrs)], lr=0.0, eps=1e-15)
losses = []
for it in range(300):
    dinfo = torch.zeros(3, params[0].shape[0], device='cuda')
    img = diff_rasterize(*params, dinfo, s)
    loss = (img - target).abs().mean()
    loss.backward()
    opt.step()
    for p in params:
        p.grad = None
    with torch.no_grad():
        add_noise(params[1], params[2], params[3], params[0], 1e-4)
    losses.append(loss.item())
first, last = sum(losses[:10]) / 10, sum(losses[-10:]) / 10
check('training loss decreases (L1, 300 iters, with MCMC noise)', last < 0.7 * first and all(map(lambda v: v == v, losses)),
      f'first10 {first:.4f} last10 {last:.4f}')
sys.exit(0 if ok_all else 1)
