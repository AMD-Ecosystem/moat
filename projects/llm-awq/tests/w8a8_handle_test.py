# Builds the per-device hipBLASLt handle cache from w8a8_gemm_cuda.cu in isolation (under /tmp, not the
# checkout) and races its first use from 16 native threads, spread over every visible device. The
# extension's pybind entry holds the GIL, so Python threads cannot exercise this race.
# PASS: every thread on a device gets the same non-null handle, devices get distinct handles, and a
# later call returns the cached one.
import sys, tempfile
import torch
from torch.utils.cpp_extension import load_inline
src = open(sys.argv[1]).read()
start = src.index("// One hipBLASLt handle per device")
end = src.index("// C[m][n] = sum_k")
fn = src[start:end]
cuda = "#include <torch/extension.h>\n#include <hipblaslt/hipblaslt.h>\n#include <atomic>\n#include <thread>\n" + fn + r'''
std::vector<int64_t> race(int64_t nthreads) {
  int ndev = 0;
  TORCH_CHECK(hipGetDeviceCount(&ndev) == hipSuccess && ndev > 0, "no device");
  std::vector<hipblasLtHandle_t> got(nthreads, nullptr);
  std::atomic<int> ready{0};
  std::vector<std::thread> ts;
  for (int i = 0; i < nthreads; ++i)
    ts.emplace_back([&, i] {
      TORCH_CHECK(hipSetDevice(i % ndev) == hipSuccess, "hipSetDevice");
      ready.fetch_add(1);
      while (ready.load() < nthreads) {}
      got[i] = w8a8_hipblaslt_handle();
    });
  for (auto &t : ts) t.join();
  std::vector<int64_t> out;
  for (int i = 0; i < nthreads; ++i) out.push_back(i % ndev), out.push_back(reinterpret_cast<int64_t>(got[i]));
  for (int d = 0; d < ndev; ++d) {
    TORCH_CHECK(hipSetDevice(d) == hipSuccess, "hipSetDevice");
    out.push_back(d), out.push_back(reinterpret_cast<int64_t>(w8a8_hipblaslt_handle()));
  }
  return out;
}
'''
d = tempfile.mkdtemp(prefix="w8a8handle_")
mod = load_inline("w8a8_handle", cpp_sources="std::vector<int64_t> race(int64_t nthreads);",
                  cuda_sources=cuda, functions=["race"], build_directory=d, extra_ldflags=["-lhipblaslt"],
                  verbose=False)
n = 16
v = mod.race(n)
pairs = list(zip(v[0::2], v[1::2]))
by_dev = {}
for dev, h in pairs:
    by_dev.setdefault(dev, set()).add(h)
ok = all(len(hs) == 1 and 0 not in hs for hs in by_dev.values())
ok = ok and len({next(iter(hs)) for hs in by_dev.values()}) == len(by_dev)
print(f"{'PASS' if ok else 'FAIL'} {n} threads over {len(by_dev)} device(s): handles per device "
      f"{ {d: len(hs) for d, hs in by_dev.items()} }")
sys.exit(0 if ok else 1)
