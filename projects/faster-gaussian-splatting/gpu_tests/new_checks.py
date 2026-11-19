# Usage: python new_checks.py <FasterGSCudaBackend dir>
# Checks of features that exist only on the merged head: absolute-gradient densification,
# fused MCMC noise (Philox4x32-10 + Box-Muller), inference/training forward agreement,
# and run-to-run determinism of the training forward.
import sys, math
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import torch
from FasterGSCudaBackend.torch_bindings import diff_rasterize, rasterize, RasterizerSettings, add_noise
from run_outputs import scene, settings  # noqa: E402

results = []


def check(name, ok, detail=''):
    results.append(ok)
    print(f'{"PASS" if ok else "FAIL"}  {name}  {detail}')


dev = 'cuda'

# ---------- Philox4x32-10 reference (Random123) ----------
M0, M1, W0, W1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85
MASK = 0xFFFFFFFF


def philox(c, k):
    c = list(c); k = list(k)
    for r in range(10):
        if r:
            k = [(k[0] + W0) & MASK, (k[1] + W1) & MASK]
        p0, p1 = M0 * c[0], M1 * c[2]
        c = [((p1 >> 32) ^ c[1] ^ k[0]) & MASK, p1 & MASK, ((p0 >> 32) ^ c[3] ^ k[1]) & MASK, p0 & MASK]
    return c


# published Random123 known-answer vectors for philox4x32_10
kat = [
    ((0, 0, 0, 0), (0, 0), (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)),
    ((MASK,) * 4, (MASK, MASK), (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)),
    ((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344), (0xa4093822, 0x299f31d0), (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1)),
]
check('python philox reference matches Random123 KATs', all(tuple(philox(c, k)) == tuple(e) for c, k, e in kat))


def box_muller(bu, bv):
    u = bu * 2.3283064e-10 + 1.1641532e-10
    r = math.sqrt(-2.0 * math.log(u))
    t = bv * 1.4629181e-9
    return r * math.cos(t), r * math.sin(t)


# ---------- fused MCMC noise: device output vs reference, element by element ----------
n = 100003
raw_scales = torch.zeros(n, 3, device=dev)
raw_rot = torch.zeros(n, 4, device=dev); raw_rot[:, 0] = 1.0
raw_opa = torch.full((n, 1), -30.0, device=dev)  # opacity ~0 -> op_sigmoid = 1/(1+exp(-0.5))
factor = 1.0 / (1.0 + math.exp(-0.5))
seed = 1234567
torch.manual_seed(seed)
means = torch.zeros(n, 3, device=dev)
add_noise(raw_scales, raw_rot, raw_opa, means, 1.0)
torch.cuda.synchronize()
noise = (means / factor).double().cpu()
# first call after manual_seed: offset 0 -> counter 0
idx = list(range(0, 64)) + list(range(n - 64, n)) + list(range(1000, n, 997))
ref = []
for i in idx:
    b = philox((0, 0, i, 0), (seed & MASK, seed >> 32))
    x, y = box_muller(b[0], b[1])
    z, _ = box_muller(b[2], b[3])
    ref.append((x, y, z))
ref = torch.tensor(ref, dtype=torch.float64)
err = (noise[idx] - ref).abs().max().item()
check('add_noise samples match Philox4x32-10 + Box-Muller reference', err < 2e-4, f'max abs err {err:.2e} over {len(idx)} primitives')

mu, sd = noise.mean(0), noise.std(0)
corr = torch.corrcoef(noise.T)
offdiag = (corr - torch.eye(3, dtype=torch.float64)).abs().max().item()
check('add_noise is standard normal per axis', mu.abs().max().item() < 0.02 and (sd - 1).abs().max().item() < 0.02,
      f'mean {mu.tolist()} std {sd.tolist()}')
check('add_noise axes uncorrelated', offdiag < 0.02, f'max |corr| {offdiag:.3e}')
lanes = noise[:64 * (n // 64)].view(-1, 64, 3)
lo, hi = lanes[:, :32].std().item(), lanes[:, 32:].std().item()
check('add_noise lanes 0-31 and 32-63 of each 64-wide wave identical in distribution', abs(lo - hi) < 0.02, f'std lo {lo:.4f} hi {hi:.4f}')
means2 = torch.zeros(n, 3, device=dev)
add_noise(raw_scales, raw_rot, raw_opa, means2, 1.0)
check('second add_noise call draws fresh samples (offset advances)', not torch.equal(means, means2))
torch.manual_seed(seed)
means3 = torch.zeros(n, 3, device=dev)
add_noise(raw_scales, raw_rot, raw_opa, means3, 1.0)
check('add_noise reproducible under torch.manual_seed', torch.equal(means, means3))

# ---------- rasterizer checks on the merged head ----------
for name, n, seed_, w, h, shb in [('n1001_800x600', 1001, 3, 800, 600, 16), ('n20000_800x600', 20000, 5, 800, 600, 16),
                                   ('n200000_1920x1080', 200000, 7, 1920, 1080, 16), ('n5000_257x131', 5000, 4, 257, 131, 9)]:
    params = scene(n, seed_)
    s = settings(w, h, shb)
    with torch.no_grad():
        inf_img = rasterize(*params, s, to_chw=True, clamp_output=False)
    runs = []
    for rows in (2, 3, 3):
        leaves = [p.clone().requires_grad_(True) for p in params]
        dinfo = torch.zeros(rows, n, device=dev)
        img = diff_rasterize(*leaves, dinfo, s)
        g = torch.Generator(device='cpu').manual_seed(100 + seed_)
        (img * torch.randn(img.shape, generator=g).to(dev)).sum().backward()
        runs.append((img.detach(), dinfo, [l.grad for l in leaves]))
    img2, d2, g2 = runs[0]
    img3, d3, g3 = runs[1]
    check(f'{name}: inference image == training forward image', (inf_img - img2).abs().max().item() < 1e-5,
          f'max {(inf_img - img2).abs().max().item():.2e}')
    check(f'{name}: training forward deterministic', torch.equal(runs[1][0], runs[2][0]))
    # gradients accumulate through float atomics, so compare against the run-to-run noise of two identical runs
    g3b, d3b = runs[2][2], runs[2][1]
    rel = lambda a, b: ((a - b).norm() / a.norm().clamp_min(1e-30)).item()
    noise = max(max(rel(a, b) for a, b in zip(g3, g3b)), rel(d3[1], d3b[1]))
    delta = max(max(rel(a, b) for a, b in zip(g2, g3)), rel(d2[1], d3[1]))
    check(f'{name}: abs-grad run leaves rows 0/1 and all grads as without it',
          torch.equal(d2[0], d3[0]) and delta <= 10 * noise + 1e-6,
          f'relL2 delta {delta:.2e} vs run-to-run {noise:.2e}')
    viol = (d3[1] - d3[2] - 1e-3 * d3[1].abs() - 1e-4).clamp_min(0).max().item()
    check(f'{name}: abs-grad row >= grad-norm row (triangle inequality)', viol == 0 and d3[2].sum() > 0,
          f'violation {viol:.2e}, row2 sum {d3[2].sum().item():.3e}, row1 sum {d3[1].sum().item():.3e}')
    finite = all(torch.isfinite(t).all() for t in [img3, d3] + g3)
    check(f'{name}: all outputs finite', bool(finite))
    # every 32-lane subtile of the 192-thread blend block has image content (exercises both halves of each wave)
    H, W = img2.shape[1], img2.shape[2]
    fg = (img2 - s.bg_color.view(3, 1, 1)).abs().sum(0) > 1e-3
    ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing='ij')
    warp = ((ys % 12) // 4) * 2 + (xs % 16) // 8
    cov = [fg[warp == k].float().mean().item() for k in range(6)]
    check(f'{name}: foreground present in every subtile slot 0..5', min(cov) > 0.05, 'coverage ' + ' '.join(f'{c:.2f}' for c in cov))

print(f'{sum(results)}/{len(results)} passed')
sys.exit(0 if all(results) else 1)
