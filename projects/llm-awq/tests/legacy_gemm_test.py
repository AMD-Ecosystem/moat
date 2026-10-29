# gemm_forward_cuda (legacy 4-bit GEMM, quantization/gemm_cuda_gen.cu) on packed weights in its
# layout: kernel [OC, IC/8] int32 (nibble i of word j = input channel 8j+i), zeros [OC, zeros_w]
# int32 (8 groups per word), scales [OC, zeros_w*8] fp16.
# Checked against (1) a PyTorch dequantize + matmul reference, independent of the kernels, and
# (2) the legacy gemv_forward_cuda, row by row, on the same packed weights.
import sys
import torch
import awq_inference_engine as eng

torch.manual_seed(0)
dev = "cuda"


def ceil_div(a, b):
    return (a + b - 1) // b


def pack(OC, IC, G):
    q = torch.randint(0, 16, (OC, IC), dtype=torch.int32)
    ngroups = IC // G
    z = torch.randint(0, 16, (OC, ngroups), dtype=torch.int32)
    s = (torch.rand(OC, ngroups) * 0.02 + 0.001).half()
    mult = 128 // G
    zeros_w = ceil_div(ceil_div(ngroups, 8), mult) * mult
    kernel = torch.zeros(OC, IC // 8, dtype=torch.int64)
    for i in range(8):
        kernel |= (q[:, i::8].long() & 0xF) << (4 * i)
    zeros = torch.zeros(OC, zeros_w, dtype=torch.int64)
    for g in range(ngroups):
        zeros[:, g // 8] |= (z[:, g].long() & 0xF) << (4 * (g % 8))
    scales = torch.zeros(OC, zeros_w * 8, dtype=torch.float16)
    scales[:, :ngroups] = s
    to_i32 = lambda t: ((t + 2**31) % 2**32 - 2**31).to(torch.int32)
    # dequantized weight exactly as the kernels compute it: fp16(scale * (q - z))
    w = (s.float().repeat_interleave(G, 1) * (q.float() - z.float().repeat_interleave(G, 1))).half()
    return to_i32(kernel).to(dev), scales.to(dev), to_i32(zeros).to(dev), w.to(dev)


results = []
for (M, OC, IC, G) in [(1, 64, 1024, 128), (8, 128, 2048, 128), (16, 4096, 4096, 128), (33, 256, 1024, 64),
                       (128, 512, 4096, 64), (129, 1024, 2048, 128), (300, 640, 3072, 128), (7, 11008 // 64 * 64, 4096, 128)]:
    kernel, scales, zeros, w = pack(OC, IC, G)
    x = (torch.randn(M, IC, device=dev) * 0.5).half()
    out = eng.gemm_forward_cuda(x, kernel, scales, zeros, G, 8)
    torch.cuda.synchronize()
    ref = x.float() @ w.float().t()
    err = (out.float() - ref).abs().max().item()
    tol = 2e-3 * ref.abs().max().item() + 1e-2
    gerr = float("nan")
    ok = out.shape == (M, OC) and err <= tol and torch.isfinite(out).all().item()
    if IC % 1024 == 0:
        # the legacy GEMV handles one row per batch entry and needs IC to be a multiple of 1024
        gv = eng.gemv_forward_cuda(x, kernel, scales, zeros, G)
        gerr = (out.float() - gv.float()).abs().max().item()
        ok = ok and gerr <= tol
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} M={M} OC={OC} IC={IC} G={G}: vs dequant ref max_err={err:.3e} (tol {tol:.2e}), vs gemv max_err={gerr:.3e}", flush=True)

print(f"{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
