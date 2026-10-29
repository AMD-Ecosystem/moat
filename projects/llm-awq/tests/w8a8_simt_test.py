# Builds the SIMT int8 fallback kernel text from w8a8_gemm_cuda.cu in isolation (under /tmp, not the
# checkout) and checks it against an exact int64 matmul; on GPUs with hipBLASLt int8 support the
# extension itself never reaches this kernel.
import sys, re, os, tempfile
import torch
from torch.utils.cpp_extension import load_inline
src = open(sys.argv[1]).read()
start = src.index("template <int TILE_MN, int TILE_K>")
end = src.index("// out[m][n] = C[m][n]")
kernel = src[start:end]
cuda = "#include <torch/extension.h>\n" + kernel + r'''
torch::Tensor simt(torch::Tensor A, torch::Tensor B) {
  int M = A.size(0), K = A.size(1), N = B.size(0);
  auto C = torch::empty({M, N}, A.options().dtype(torch::kInt32));
  dim3 grid((N + 15) / 16, (M + 15) / 16);
  w8a8_int8_gemm_simt<16, 64><<<grid, dim3(16, 16)>>>(A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), C.data_ptr<int32_t>(), M, N, K);
  return C;
}
'''
d = tempfile.mkdtemp(prefix="w8a8simt_")
mod = load_inline("w8a8_simt", cpp_sources="torch::Tensor simt(torch::Tensor A, torch::Tensor B);",
                  cuda_sources=cuda, functions=["simt"], build_directory=d, verbose=False)
ok_all = []
for M, N, K in [(1, 64, 64), (3, 13, 7), (33, 50, 100), (65, 72, 37), (17, 4096, 11008), (300, 130, 520), (128, 256, 4096)]:
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device="cuda")
    w = torch.randint(-128, 128, (N, K), dtype=torch.int8, device="cuda")
    c = mod.simt(a, w).cpu().long()
    r = a.long().cpu() @ w.long().cpu().t()
    ok = torch.equal(c, r)
    ok_all.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} simt M={M} N={N} K={K} exact={ok}")
print(f"{sum(ok_all)}/{len(ok_all)} passed")
sys.exit(0 if all(ok_all) else 1)
