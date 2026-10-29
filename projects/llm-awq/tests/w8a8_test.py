# w8a8_gemm_forward_cuda / w8a8_gemm_fuse_bias_forward_cuda vs an exact int64 matmul reference
# with the same epilogue: out = half(acc * wscale[n] * ascale[m] (+ bias[n])).
# The profiler reports which int8 GEMM ran (hipBLASLt kernels vs the SIMT fallback).
import sys
import torch
import awq_inference_engine as eng
from torch.profiler import profile, ProfilerActivity

torch.manual_seed(0)
dev = "cuda"


def ref(a, w, ws, as_, bias):
    acc = (a.long().cpu() @ w.long().cpu().t()).to(torch.float64)  # exact int64 product
    out = acc * ws.double().cpu()[None, :] * as_.double().cpu()[:, None]
    if bias is not None:
        out = out + bias.double().cpu()[None, :]
    return out, acc


def case(M, N, K, bias, results, kernels):
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=dev)
    w = torch.randint(-128, 128, (N, K), dtype=torch.int8, device=dev)
    ws = (torch.rand(N, device=dev) * 1e-3 + 1e-4).half()
    as_ = (torch.rand(M, device=dev) * 1e-2 + 1e-3).half()
    b = torch.randn(N, device=dev).half() if bias else None
    out = torch.empty(M, N, dtype=torch.float16, device=dev)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        if bias:
            eng.w8a8_gemm_fuse_bias_forward_cuda(a, w, ws, as_, out, b)
        else:
            eng.w8a8_gemm_forward_cuda(a, w, ws, as_, out)
        torch.cuda.synchronize()
    names = [e.key for e in prof.key_averages() if "epilogue" not in e.key and "Memcpy" not in e.key and "Memset" not in e.key]
    path = "simt" if any("w8a8_int8_gemm_simt" in n for n in names) else ("hipblaslt" if names else "?")
    kernels[path] = kernels.get(path, 0) + 1
    r, acc = ref(a, w, ws, as_, b)
    # fp16 output: allow 1 ulp-ish relative error plus the fp32 epilogue's own rounding
    err = (out.double().cpu() - r).abs()
    tol = r.abs() * 2e-3 + 1e-3
    ok = bool((err <= tol).all()) and torch.isfinite(out).all().item()
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} M={M} N={N} K={K} bias={bias} path={path} max_err={err.max().item():.3e} max|out|={r.abs().max().item():.2f}", flush=True)


results, kernels = [], {}
shapes = [(1, 64, 64), (7, 128, 256), (16, 4096, 4096), (17, 4096, 11008), (129, 1024, 512), (200, 384, 768),
          (3, 13, 7), (33, 50, 100), (65, 72, 37), (1, 8, 4), (300, 130, 520), (512, 4096, 4096)]
for M, N, K in shapes:
    for bias in (False, True):
        if bias and N % 2:
            # the CUDA fused-bias kernel reads bias in pairs; keep to even N as callers do
            pass
        case(M, N, K, bias, results, kernels)

# stream semantics: run on a side stream and check the output is ordered with that stream.
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    a = torch.randint(-128, 128, (64, 256), dtype=torch.int8, device=dev)
    w = torch.randint(-128, 128, (512, 256), dtype=torch.int8, device=dev)
    ws = torch.full((512,), 1e-3, device=dev).half(); as_ = torch.full((64,), 1e-2, device=dev).half()
    out = torch.empty(64, 512, dtype=torch.float16, device=dev)
    eng.w8a8_gemm_forward_cuda(a, w, ws, as_, out)
    y = out.float().sum()
s.synchronize()
r, _ = ref(a, w, ws, as_, None)
ok = abs(y.item() - r.sum().item()) <= 1e-2 * abs(r).sum().item() / 64
results.append(ok)
print(f"{'PASS' if ok else 'FAIL'} side-stream ordering")

print("paths:", kernels)
print(f"{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
