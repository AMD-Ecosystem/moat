# Quest prefill check and timing scripts (linux-gfx950 validation, 2026-11-23)

Written for the linux-gfx950 validation at fork `9cb8bee` (notes.md, "Validation 2026-11-23 (linux-gfx950, at `9cb8bee`)") because the porter's harness (`prefill_harness.cu`, 2026-11-20) was never recorded and its container is gone. Kept here so the Radeon and Windows validators can run the same check. They are not part of the port and do not go to the fork.

## `prefill_harness.cu`: fp64 reference check (CFG 0-11) and kernel-only timing (CFG 100)

Calls `BatchPrefillWithPagedKVCacheDispatched` directly with a forced `num_frags_x`, shuffled pages in the NHD layout (unused page slots hold garbage), and checks output and LSE against an fp64 CPU reference with the bottom-right causal rule. Tolerances: 5e-3 + 5e-3 |ref| (fp16), 2e-2 + 3e-2 |ref| (bf16), LSE 2e-2 + 2e-3 |ref| in log2 units. CFG 10 and 11 (head 64 with two query tiles per warp) were added by the 2026-11-24 linux-gfx942 review, because no earlier configuration or test reached that instantiation. Ten shapes per configuration: 1x17, 13x33, 16x16, 51x400, 63x700, 77x129, 128x129, 244x1110, 311x311, 502x1110 (qo x kv), 4 KV heads. `-DNEGCTRL=true` makes the reference use the opposite causal flag (negative control). CFG 100 times the kernel for 32 heads x 128, fp16, causal, page size 16, nfx 2 (forced, also for 1 x 4096 where the operator would pick nfx 1), mean of 20 launches after 3 warmups.

```cpp
// fp64 reference check and timing of Quest's paged prefill kernel, calling
// BatchPrefillWithPagedKVCacheDispatched with an explicit num_frags_x.
// Build one configuration per binary: -DCFG=<0..9> (check) or -DCFG=100 (timing).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "prefill/prefill.cuh"

#ifndef NEGCTRL
#define NEGCTRL false
#endif

using namespace flashinfer;

#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess) {                                                          \
      printf("HIP error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      exit(2);                                                                        \
    }                                                                                 \
  } while (0)

template <typename T> float to_f(T x);
template <> float to_f<half>(half x) { return __half2float(x); }
template <> float to_f<__hip_bfloat16>(__hip_bfloat16 x) { return __bfloat162float(x); }
template <typename T> T from_f(float x);
template <> half from_f<half>(float x) { return __float2half(x); }
template <> __hip_bfloat16 from_f<__hip_bfloat16>(float x) { return __float2bfloat16(x); }

constexpr uint32_t kNumWarps = 4;
static uint32_t kNumKvHeads = 4;

// Returns number of failing elements (output or LSE); prints max errors.
template <typename DT, uint32_t NFX, uint32_t PS, uint32_t GS, uint32_t HD, bool CAUSAL>
int run_case(uint32_t qo_len, uint32_t kv_len, uint32_t seed, int iters, double* ms_out) {
  const uint32_t num_qo_heads = kNumKvHeads * GS;
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  const uint32_t num_pages = (kv_len + PS - 1) / PS;
  const uint32_t last_page_len = kv_len - (num_pages - 1) * PS;
  const uint32_t pool = num_pages + 7;
  std::vector<int32_t> perm(pool);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);
  std::vector<int32_t> indices(perm.begin(), perm.begin() + num_pages);

  std::vector<DT> q_h(size_t(qo_len) * num_qo_heads * HD);
  for (auto& x : q_h) x = from_f<DT>(nd(rng));
  // NHD pages: [pool, 2, PS, num_kv_heads, HD]; unused slots hold garbage.
  std::vector<DT> kv_h(size_t(pool) * 2 * PS * kNumKvHeads * HD);
  for (auto& x : kv_h) x = from_f<DT>(nd(rng) * 4.f);
  std::vector<double> k_r(size_t(kv_len) * kNumKvHeads * HD), v_r(k_r.size());
  for (uint32_t n = 0; n < kv_len; ++n) {
    for (uint32_t h = 0; h < kNumKvHeads; ++h)
      for (uint32_t d = 0; d < HD; ++d) {
        size_t page = indices[n / PS], e = n % PS;
        size_t ko = ((page * 2 * PS + e) * kNumKvHeads + h) * HD + d;
        size_t vo = (((page * 2 + 1) * PS + e) * kNumKvHeads + h) * HD + d;
        float kf = nd(rng), vf = nd(rng);
        kv_h[ko] = from_f<DT>(kf);
        kv_h[vo] = from_f<DT>(vf);
        k_r[(size_t(n) * kNumKvHeads + h) * HD + d] = to_f(kv_h[ko]);
        v_r[(size_t(n) * kNumKvHeads + h) * HD + d] = to_f(kv_h[vo]);
      }
  }

  // Tiles as split_qo_indptr makes them, with num_frags_x forced.
  const uint32_t rows_per_cta = NFX * kNumWarps * 16;
  std::vector<int32_t> req, tile;
  for (uint32_t j = 0; j < qo_len * GS; j += rows_per_cta) {
    req.push_back(0);
    tile.push_back(j / rows_per_cta);
  }
  const uint32_t num_tiles = req.size();
  std::vector<int32_t> qo_indptr = {0, int32_t(qo_len)};
  std::vector<int32_t> kv_indptr = {0, int32_t(num_pages)};

  DT *q_d, *kv_d, *o_d;
  float* lse_d;
  int32_t *idx_d, *kvp_d, *qop_d, *req_d, *tile_d;
  CK(cudaMalloc(&q_d, q_h.size() * sizeof(DT)));
  CK(cudaMalloc(&kv_d, kv_h.size() * sizeof(DT)));
  CK(cudaMalloc(&o_d, q_h.size() * sizeof(DT)));
  CK(cudaMalloc(&lse_d, size_t(qo_len) * num_qo_heads * sizeof(float)));
  CK(cudaMalloc(&idx_d, num_pages * 4));
  CK(cudaMalloc(&kvp_d, 8));
  CK(cudaMalloc(&qop_d, 8));
  CK(cudaMalloc(&req_d, num_tiles * 4));
  CK(cudaMalloc(&tile_d, num_tiles * 4));
  CK(cudaMemcpy(q_d, q_h.data(), q_h.size() * sizeof(DT), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(kv_d, kv_h.data(), kv_h.size() * sizeof(DT), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(idx_d, indices.data(), num_pages * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(kvp_d, kv_indptr.data(), 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(qop_d, qo_indptr.data(), 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(req_d, req.data(), num_tiles * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(tile_d, tile.data(), num_tiles * 4, cudaMemcpyHostToDevice));
  CK(cudaMemset(o_d, 0xff, q_h.size() * sizeof(DT)));

  paged_kv_t<PageStorage::kIndices, QKVLayout::kNHD, DT, int32_t> paged_kv(
      kNumKvHeads, PS, HD, 1, 0, last_page_len, indices.back(), kv_d, idx_d, kvp_d);
  auto launch = [&]() {
    CK((BatchPrefillWithPagedKVCacheDispatched<PageStorage::kIndices, QKVLayout::kNHD, NFX, PS,
                                               GS, HD, RotaryMode::kNone, false, CAUSAL, DT, DT,
                                               int32_t>(q_d, req_d, tile_d, qop_d, paged_kv, o_d,
                                                        nullptr, lse_d, num_tiles, 1.f, 1e4f,
                                                        nullptr)));
  };
  launch();
  CK(cudaDeviceSynchronize());
  if (iters > 0) {
    for (int i = 0; i < 3; ++i) launch();
    CK(cudaDeviceSynchronize());
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    CK(hipEventRecord(a));
    for (int i = 0; i < iters; ++i) launch();
    CK(hipEventRecord(b));
    CK(hipEventSynchronize(b));
    float ms;
    CK(hipEventElapsedTime(&ms, a, b));
    *ms_out = ms / iters;
  }
  std::vector<DT> o_h(q_h.size());
  std::vector<float> lse_h(size_t(qo_len) * num_qo_heads);
  CK(cudaMemcpy(o_h.data(), o_d, o_h.size() * sizeof(DT), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(lse_h.data(), lse_d, lse_h.size() * 4, cudaMemcpyDeviceToHost));
  cudaFree(q_d); cudaFree(kv_d); cudaFree(o_d); cudaFree(lse_d); cudaFree(idx_d);
  cudaFree(kvp_d); cudaFree(qop_d); cudaFree(req_d); cudaFree(tile_d);
  if (iters > 0 && qo_len * size_t(kv_len) > 4096u * 1024u) return 0;  // timing only

  const bool is_bf16 = std::is_same<DT, __hip_bfloat16>::value;
  const double atol = is_bf16 ? 2e-2 : 5e-3, rtol = is_bf16 ? 3e-2 : 5e-3;
  const double sm_scale = 1.0 / std::sqrt(double(HD));
  std::vector<int> bad(num_qo_heads, 0);
  std::vector<double> max_o(num_qo_heads, 0), max_l(num_qo_heads, 0);
  std::vector<std::thread> th;
  for (uint32_t hq = 0; hq < num_qo_heads; ++hq) {
    th.emplace_back([&, hq]() {
      const uint32_t hk = hq / GS;
      std::vector<double> s(kv_len), acc(HD);
      for (uint32_t i = 0; i < qo_len; ++i) {
        double m = -INFINITY;
        uint32_t lim = (CAUSAL != NEGCTRL) ? (kv_len - qo_len + i + 1) : kv_len;
        for (uint32_t n = 0; n < lim; ++n) {
          double t = 0;
          for (uint32_t d = 0; d < HD; ++d)
            t += double(to_f(q_h[(size_t(i) * num_qo_heads + hq) * HD + d])) *
                 k_r[(size_t(n) * kNumKvHeads + hk) * HD + d];
          s[n] = t * sm_scale;
          m = std::max(m, s[n]);
        }
        double den = 0;
        std::fill(acc.begin(), acc.end(), 0.0);
        for (uint32_t n = 0; n < lim; ++n) {
          double p = std::exp(s[n] - m);
          den += p;
          for (uint32_t d = 0; d < HD; ++d) acc[d] += p * v_r[(size_t(n) * kNumKvHeads + hk) * HD + d];
        }
        for (uint32_t d = 0; d < HD; ++d) {
          double ref = acc[d] / den;
          double got = to_f(o_h[(size_t(i) * num_qo_heads + hq) * HD + d]);
          double err = std::fabs(got - ref);
          if (!(err <= atol + rtol * std::fabs(ref))) ++bad[hq];
          if (!(err <= max_o[hq])) max_o[hq] = std::isnan(err) ? INFINITY : err;
        }
        double lref = (m + std::log(den)) / std::log(2.0);
        double lerr = std::fabs(double(lse_h[size_t(i) * num_qo_heads + hq]) - lref);
        if (!(lerr <= 2e-2 + 2e-3 * std::fabs(lref))) ++bad[hq];
        if (!(lerr <= max_l[hq])) max_l[hq] = std::isnan(lerr) ? INFINITY : lerr;
      }
    });
  }
  for (auto& t : th) t.join();
  int nbad = std::accumulate(bad.begin(), bad.end(), 0);
  printf("  %4u x %-5u bad=%d max|dO|=%.3e max|dLSE|=%.3e\n", qo_len, kv_len, nbad,
         *std::max_element(max_o.begin(), max_o.end()),
         *std::max_element(max_l.begin(), max_l.end()));
  return nbad;
}

#ifndef CFG
#define CFG 0
#endif

#if CFG == 0
#define CASE half, 1, 16, 1, 128, true
#define NAME "fp16 hd128 nfx1 causal p16"
#elif CFG == 1
#define CASE half, 2, 16, 1, 128, true
#define NAME "fp16 hd128 nfx2 causal p16"
#elif CFG == 2
#define CASE half, 2, 16, 1, 128, false
#define NAME "fp16 hd128 nfx2 non-causal p16"
#elif CFG == 3
#define CASE half, 1, 16, 1, 64, true
#define NAME "fp16 hd64 nfx1 causal p16"
#elif CFG == 4
#define CASE half, 2, 16, 4, 128, true
#define NAME "fp16 hd128 GQA4 nfx2 causal p16"
#elif CFG == 5
#define CASE __hip_bfloat16, 2, 16, 1, 128, true
#define NAME "bf16 hd128 nfx2 causal p16"
#elif CFG == 6
#define CASE __hip_bfloat16, 1, 16, 4, 64, false
#define NAME "bf16 hd64 GQA4 nfx1 non-causal p16"
#elif CFG == 7
#define CASE half, 2, 1, 1, 128, true
#define NAME "fp16 hd128 nfx2 causal p1"
#elif CFG == 8
#define CASE half, 1, 8, 1, 128, true
#define NAME "fp16 hd128 nfx1 causal p8"
#elif CFG == 9
#define CASE half, 2, 32, 8, 128, true
#define NAME "fp16 hd128 GQA8 nfx2 causal p32"
#elif CFG == 10
#define CASE half, 2, 16, 1, 64, true
#define NAME "fp16 hd64 nfx2 causal p16"
#elif CFG == 11
#define CASE __hip_bfloat16, 2, 8, 4, 64, false
#define NAME "bf16 hd64 GQA4 nfx2 non-causal p8"
#elif CFG == 100
#define CASE half, 2, 16, 1, 128, true
#define NAME "timing fp16 hd128 nfx2 causal p16, 32 heads"
#endif

int main(int argc, char** argv) {
#if __has_include("prefill/prefill_matrix_core.cuh")
  matrix_core::mma_shape_t sh;
  CK(matrix_core::get_prefill_mma_shape(&sh));
  printf("%s  [device mma shape wave=%u op_elems=%u]\n", NAME, sh.wave_size, sh.op_elems);
#else
  printf("%s  [tree without the matrix-core header: emulated path]\n", NAME);
#endif
#if CFG == 100
  kNumKvHeads = 32;  // 32 heads x 128, as the operator-level table
  const uint32_t shapes[][2] = {{1, 4096}, {128, 4096}, {1024, 1024}, {1024, 4096}, {4096, 4096}, {8192, 8192}};
  for (auto& s : shapes) {
    double ms = 0;
    run_case<CASE>(s[0], s[1], 1, 20, &ms);
    printf("  time %u x %u: %.4f ms\n", s[0], s[1], ms);
  }
  return 0;
#else
  const uint32_t shapes[][2] = {{1, 17}, {13, 33}, {16, 16}, {51, 400}, {63, 700},
                                {77, 129}, {128, 129}, {244, 1110}, {311, 311}, {502, 1110}};
  int fails = 0, n = 0;
  for (auto& s : shapes) {
    double ms;
    fails += run_case<CASE>(s[0], s[1], 1234 + n++, 0, &ms) != 0;
  }
  printf("%s: %d/%d cases pass\n", NAME, n - fails, n);
  return fails ? 1 : 0;
#endif
}
```

## `build.sh`

Built against a fork checkout (`<tree>`), or against the `kernels/include` of an older commit (`git archive <sha> kernels/include` plus a link to the checkout's `kernels/3rdparty`) to time the emulated path. Same flags as the extension's `batch_prefill.cu` compile line.

```bash
#!/bin/bash
# build.sh <quest tree> <output> <CFG> [extra flags]; ARCH (default gfx950) and SDK (ROCm root
# holding lib/llvm/bin/clang++; default: the pip ROCm SDK of the active python) may be set.
T=$1; OUT=$2; CFG=$3; shift 3
SDK=${SDK:-$(python -c 'import _rocm_sdk_devel,os;print(os.path.dirname(_rocm_sdk_devel.__file__))')}
K=$T/kernels/include
exec $SDK/lib/llvm/bin/clang++ -x hip -DUSE_HIP -DUSE_ROCM -D__HIP_PLATFORM_AMD__=1 -DCUDA_HAS_FP16=1 \
  -D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1 -DHIP_ENABLE_WARP_SYNC_BUILTINS \
  -I$K/hip_compat -I$K -I$T/kernels/3rdparty/flashinfer/include \
  -O3 -DNDEBUG -std=gnu++20 --offload-arch=${ARCH:-gfx950} -ffp-contract=on \
  -include $K/hip_compat/flashinfer/math.cuh -include $K/hip_compat/flashinfer/mma.cuh \
  -include $K/hip_compat/flashinfer/vec_dtypes.cuh \
  -DCFG=$CFG "$@" $(dirname $0)/prefill_harness.cu --hip-link -o $OUT
```

Usage: `bash build.sh <tree> bin/mc_<n> <n>` for n in 0-9 and 100; run each binary under `ulimit -c 0`; a check binary exits 0 when all 10 cases pass.

## `prefill_time.py`: operator time against SDPA

`quest.utils.prefill_forward` against `F.scaled_dot_product_attention` with the same alignment (`is_causal` where qo = kv, `causal_lower_right` where qo < kv). With no environment variables it is the 2026-11-20 MI300X method (mean of 20, 5 for the large shapes, 3 warmups). `MINMED=1 LOOP_MS=100 REPS=7 WARM_MS=1000` (used on gfx950, where GPU clocks vary between runs) warms up for 1 s, sizes each timed loop to about 100 ms, repeats 7 times and prints `min (median)`.

```python
# Quest prefill_forward (the operator the model calls) against torch SDPA with the same
# bottom-right causal alignment. Causal, 32 heads x 128, fp16, page size 16.
import os
import statistics
import time

import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

import quest.utils

H, D, PS = 32, 128, 16
dev = torch.device("cuda:0")
dt = torch.float16
SHAPES = [(1, 4096), (128, 4096), (1024, 1024), (1024, 4096), (4096, 4096), (8192, 8192), (16384, 16384)]


REPS = int(os.environ.get("REPS", "1"))
WARM_MS = float(os.environ.get("WARM_MS", "0"))
MINMED = os.environ.get("MINMED") == "1"
LOOP_MS = float(os.environ.get("LOOP_MS", "0"))


def timed(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    # Optional warmup by wall time, so the GPU clock has ramped before measuring.
    t0 = time.time()
    while (time.time() - t0) * 1e3 < WARM_MS:
        fn()
    torch.cuda.synchronize()
    if LOOP_MS:
        # Size each timed loop to about LOOP_MS of work.
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        iters = max(iters, int(LOOP_MS / max(a.elapsed_time(b), 1e-3)))
    out = []
    for _ in range(REPS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) / iters)
    return (min(out), statistics.median(out)) if MINMED else statistics.median(out)


print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).gcnArchName, torch.__version__)
print("| qo x kv | Quest prefill_forward | torch SDPA | ratio | max abs diff vs SDPA |")
print("|---|---|---|---|---|")
for qo, kv in SHAPES:
    iters = 5 if qo * kv >= 8192 * 8192 else 20
    torch.manual_seed(0)
    q = torch.randn(qo, H, D, dtype=dt, device=dev)
    k = torch.randn(kv, H, D, dtype=dt, device=dev)
    v = torch.randn(kv, H, D, dtype=dt, device=dev)
    ctl = quest.utils.InferenceController(1, H, D, PS, 1024, kv, dt, dev)
    ctl.prepare_metadata(kv)
    ctl.begin_forward(kv)
    quest.utils.append_kv(k, v, ctl, 0)
    with torch.inference_mode():
        o = quest.utils.prefill_forward(q, ctl, 0)
        t_q = timed(lambda: quest.utils.prefill_forward(q, ctl, 0), iters)
        qt, kt, vt = (x.transpose(0, 1).unsqueeze(0) for x in (q, k, v))
        if qo == kv:
            sdpa = lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
        else:
            bias = causal_lower_right(qo, kv)
            sdpa = lambda: F.scaled_dot_product_attention(qt, kt, vt, attn_mask=bias)
        ref = sdpa()
        t_s = timed(sdpa, iters)
    diff = (o - ref.squeeze(0).transpose(0, 1)).abs().max().item()
    if MINMED:
        print(f"| {qo} x {kv} | {t_q[0]:.3f} ({t_q[1]:.3f}) ms | {t_s[0]:.3f} ({t_s[1]:.3f}) ms | {t_q[0] / t_s[0]:.2f} | {diff:.1e} |")
    else:
        print(f"| {qo} x {kv} | {t_q:.3f} ms | {t_s:.3f} ms | {t_q / t_s:.2f} | {diff:.1e} |")
    del ctl
    torch.cuda.empty_cache()
```
