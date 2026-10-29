# single_query_attention (FasterTransformer masked MHA) vs a PyTorch reference that applies the
# same rotary embedding, writes the same KV-cache slot and attends over the cache.
import math, sys, itertools
import torch
import awq_inference_engine as eng

torch.manual_seed(0)
dev = "cuda"


def rotary(x, pos, rot_dim, base, scale, neox):
    # x: [B, Hx, D] float, pos: [B] int
    if rot_dim == 0:
        return x
    x = x.clone()
    half = rot_dim // 2
    i = torch.arange(half, device=x.device, dtype=torch.float32)
    inv = 1.0 / torch.pow(torch.tensor(base, dtype=torch.float32, device=x.device), (2 * i) / rot_dim)
    ang = (pos.float() * scale)[:, None] * inv[None, :]  # [B, half]
    c, s = torch.cos(ang)[:, None, :], torch.sin(ang)[:, None, :]
    if neox:
        a, b = x[..., :half].clone(), x[..., half:rot_dim].clone()
        x[..., :half] = a * c - b * s
        x[..., half:rot_dim] = b * c + a * s
    else:
        a, b = x[..., 0:rot_dim:2].clone(), x[..., 1:rot_dim:2].clone()
        x[..., 0:rot_dim:2] = a * c - b * s
        x[..., 1:rot_dim:2] = a * s + b * c
    return x


def reference(q, k, v, kc, vc, lens, alibi, t, rot_dim, base, scale, neox):
    B, H, D = q.shape
    Hkv, L = vc.shape[1], vc.shape[2]
    x = kc.shape[-1]
    pos = lens.long() if lens is not None else torch.full((B,), t, device=q.device, dtype=torch.long)
    qf = rotary(q.float(), pos, rot_dim, base, scale, neox)
    kf = rotary(k.float(), pos, rot_dim, base, scale, neox)
    Kc = kc.float().permute(0, 1, 3, 2, 4).reshape(B, Hkv, L, D).clone()
    Vc = vc.float().clone()
    for b in range(B):
        Kc[b, :, pos[b] % L] = kf[b].to(q.dtype).float()
        Vc[b, :, pos[b] % L] = v[b].float()
    out = torch.empty(B, H, D, device=q.device)
    rep = H // Hkv
    for b in range(B):
        tl = int(pos[b])
        first = max(0, tl + 1 - L)
        ti = torch.arange(first, tl + 1, device=q.device)
        idx = ti % L
        for h in range(H):
            kv = h // rep
            sc = (Kc[b, kv, idx] @ qf[b, h]) / math.sqrt(D)
            if alibi is not None:
                sc = sc + alibi[h] * (ti - tl).float()
            p = torch.softmax(sc, 0)
            out[b, h] = p @ Vc[b, kv, idx]
    kc_new = Kc.reshape(B, Hkv, L, D // x, x).permute(0, 1, 3, 2, 4)
    return out, kc_new, Vc, pos


def run_case(dtype, B, H, Hkv, D, L, t, rot_dim=None, neox=True, base=10000.0, scale=1.0,
             lens=None, alibi=False, fused_qkv=False):
    rot_dim = D if rot_dim is None else rot_dim
    x = 4 if dtype == torch.float32 else 8
    if fused_qkv:
        # q, k and v as views of one [B, H + 2*Hkv, D] buffer share the batch stride the kernel uses.
        qkv = torch.randn(B, H + 2 * Hkv, D, device=dev, dtype=dtype)
        q, k, v = qkv[:, :H], qkv[:, H:H + Hkv], qkv[:, H + Hkv:]
    else:
        q = torch.randn(B, H, D, device=dev, dtype=dtype)
        k = torch.randn(B, Hkv, D, device=dev, dtype=dtype)
        v = torch.randn(B, Hkv, D, device=dev, dtype=dtype)
    kc = torch.randn(B, Hkv, D // x, L, x, device=dev, dtype=dtype)
    vc = torch.randn(B, Hkv, L, D, device=dev, dtype=dtype)
    lens_t = torch.tensor(lens, device=dev, dtype=torch.int32) if lens is not None else None
    slopes = (torch.rand(H, device=dev) * 0.5) if alibi else None
    ref_out, ref_kc, ref_vc, pos = reference(q, k, v, kc, vc, lens_t, slopes, t, rot_dim, base, scale, neox)
    kc_k, vc_k = kc.clone(), vc.clone()
    out = eng.single_query_attention(q, k, v, kc_k, vc_k, lens_t, slopes, t, rot_dim, base, scale, neox)
    torch.cuda.synchronize()
    err = (out.float() - ref_out).abs().max().item()
    # Compare the cache slot each sample wrote (and that nothing else moved).
    kerr = (kc_k.float() - ref_kc).abs().max().item()
    verr = (vc_k.float() - ref_vc).abs().max().item()
    tol = {torch.float16: 1e-2, torch.bfloat16: 4e-2, torch.float32: 1e-4}[dtype]
    # The rotated key is stored in T; for fp32 the bound is the rotary angle's own rounding at large
    # positions (t * inv_freq ~ 1e4 rad carries ~1e-3 absolute error in fp32 on either vendor).
    ktol = {torch.float16: 4e-3, torch.bfloat16: 3e-2, torch.float32: 1e-4 + 2e-7 * t}[dtype]
    ok = err < tol and kerr < ktol and verr == 0 and torch.isfinite(out).all().item()
    return ok, err, kerr, verr


results = []
def case(name, **kw):
    ok, err, kerr, verr = run_case(**kw)
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name}: out_err={err:.2e} kcache_err={kerr:.2e} vcache_err={verr:.2e}", flush=True)

HEADS = [(8, 8), (32, 8), (8, 1)]  # MHA, llama GQA, falcon MQA
DIMS = [32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]
for dtype in (torch.float16, torch.bfloat16):
    for D in DIMS:
        for (H, Hkv) in HEADS:
            for t in (0, 1, 31, 32, 517, 2048):
                case(f"{str(dtype)[6:]} D={D} H={H}/{Hkv} t={t} B=1",
                     dtype=dtype, B=1, H=H, Hkv=Hkv, D=D, L=2304, t=t)
for D in (64, 128, 256):
    case(f"float32 D={D} t=300", dtype=torch.float32, B=1, H=8, Hkv=8, D=D, L=512, t=300)
    case(f"float32 D={D} GQA t=2100", dtype=torch.float32, B=1, H=32, Hkv=8, D=D, L=2304, t=2100)

# batch > 1: separate MHA tensors, and GQA/MQA through views of a fused qkv buffer.
for dtype in (torch.float16, torch.bfloat16):
    for D in (64, 80, 96, 128):
        case(f"{str(dtype)[6:]} D={D} B=4 MHA t=77", dtype=dtype, B=4, H=16, Hkv=16, D=D, L=256, t=77)
        case(f"{str(dtype)[6:]} D={D} B=3 GQA fused t=600", dtype=dtype, B=3, H=32, Hkv=8, D=D, L=1024, t=600, fused_qkv=True)
        case(f"{str(dtype)[6:]} D={D} B=2 MQA fused t=40", dtype=dtype, B=2, H=8, Hkv=1, D=D, L=128, t=40, fused_qkv=True)

# rotary variants, per-sample lengths, alibi (no rotary), wrap-around cache, partial rotary dim.
case("fp16 GPT-J rotary", dtype=torch.float16, B=2, H=8, Hkv=8, D=128, L=256, t=100, neox=False)
case("fp16 rope theta 5e5 scale 0.5", dtype=torch.float16, B=1, H=32, Hkv=8, D=128, L=256, t=200, base=500000.0, scale=0.5)
case("fp16 partial rotary 64/128", dtype=torch.float16, B=1, H=8, Hkv=8, D=128, L=256, t=50, rot_dim=64)
# The kernel sizes shared memory from timestep, so callers pass the largest length as timestep.
case("bf16 length_per_sample", dtype=torch.bfloat16, B=3, H=8, Hkv=8, D=128, L=256, t=200, lens=[5, 100, 200])
case("fp16 length_per_sample", dtype=torch.float16, B=3, H=8, Hkv=8, D=64, L=256, t=200, lens=[200, 3, 64])
case("fp16 alibi no rotary", dtype=torch.float16, B=2, H=8, Hkv=8, D=128, L=256, t=150, rot_dim=0, alibi=True)
case("fp16 cache wrap t>=L", dtype=torch.float16, B=1, H=8, Hkv=8, D=64, L=64, t=100)

# Long contexts around the shared-memory limit.
props = torch.cuda.get_device_properties(0)
print("device", props.name, getattr(props, "gcnArchName", ""), "smem/block", props.shared_memory_per_block if hasattr(props, "shared_memory_per_block") else "?")
# Last timestep that fits 64 KiB of LDS (dynamic logits buffer + the kernel's static shared memory).
EDGE = {torch.float16: 10867, torch.bfloat16: 10889, torch.float32: 16239}
for dtype, D in ((torch.float16, 128), (torch.bfloat16, 64), (torch.float32, 128)):
    tmax = None
    e = EDGE[dtype]
    for t in (4000, 8191, e - 1, e, e + 1, 20000):
        try:
            ok, err, kerr, verr = run_case(dtype=dtype, B=1, H=8, Hkv=8, D=D, L=t + 8, t=t)
            results.append(ok)
            print(f"{'PASS' if ok else 'FAIL'} {str(dtype)[6:]} D={D} long t={t}: out_err={err:.2e} kcache_err={kerr:.2e}", flush=True)
        except RuntimeError as e:
            print(f"REFUSED {str(dtype)[6:]} D={D} t={t}: {e}", flush=True)
            # A refusal is only acceptable past the limit: record that t and expect larger ones to refuse too.
            if tmax is None:
                tmax = t
        else:
            if tmax is not None:
                print("FAIL: ran after an earlier refusal")
                results.append(False)

n = len(results); p = sum(results)
print(f"{p}/{n} passed")
sys.exit(0 if p == n else 1)
