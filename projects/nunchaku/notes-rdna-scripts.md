# nunchaku: RDNA stage 2 proof scripts (porter, linux-gfx1100, 2026-11-24)

Scripts used by the RDNA porter round in `notes.md`, kept here so a gfx1201 job can rerun them. `mma_dump.py` itself is in `notes.md`. Run the Python ones from the fork root with the extension built in place and `PYTHONPATH=.`.

## test.hip

`test.hip`: the m16n16k16 WMMA helpers against the lane-gather reference and a float64 host reference (2000 random 16x16x16 problems per mode and scale, plus one exact small-integer problem).

```cpp
// Checks mma_m16n16k16_* (WMMA on RDNA) against the reference m16n8k16 lane-gather functions and a float64 host
// reference, on random 16x16x16 problems in the mma.sync fragment layouts.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include "kernels/zgemm/mma_rocm.cuh"

using namespace nunchaku::kernels;

#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        hipError_t e = (x);                                                                                            \
        if (e != hipSuccess) {                                                                                         \
            printf("%s: %s\n", #x, hipGetErrorString(e));                                                              \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

// mode 0: f32 accumulate f16, 1: f32 accumulate bf16, 2: f16 accumulate
template<int MODE>
__global__ void kern(const uint4 *A, const uint4 *B, const uint4 *C, uint4 *Dref, uint4 *Dw, int ntests) {
    for (int it = 0; it < ntests; it++) {
        const int l   = threadIdx.x;
        const uint4 a = A[it * 32 + l];
        const uint4 b = B[it * 32 + l];
        if constexpr (MODE == 2) {
            const uint4 c  = C[(it * 32 + l) * 2];
            const uint2 r1 = mma_m16n8k16_f16f16f16f16(a, uint2(b.x, b.y), uint2(c.x, c.y));
            const uint2 r2 = mma_m16n8k16_f16f16f16f16(a, uint2(b.z, b.w), uint2(c.z, c.w));
            Dref[(it * 32 + l) * 2] = uint4(r1.x, r1.y, r2.x, r2.y);
            Dw[(it * 32 + l) * 2]   = mma_m16n16k16_f16f16f16f16(a, b, c);
        } else {
            constexpr bool bf = MODE == 1;
            uint4 c0 = C[(it * 32 + l) * 2], c1 = C[(it * 32 + l) * 2 + 1];
            Dref[(it * 32 + l) * 2]     = mma_m16n8k16_f32f16f16f32<bf>(a, uint2(b.x, b.y), c0);
            Dref[(it * 32 + l) * 2 + 1] = mma_m16n8k16_f32f16f16f32<bf>(a, uint2(b.z, b.w), c1);
            mma_m16n16k16_f32f16f16f32<bf>(a, b, c0, c1);
            Dw[(it * 32 + l) * 2]     = c0;
            Dw[(it * 32 + l) * 2 + 1] = c1;
        }
    }
}

static uint16_t f2h(float f) {
    _Float16 h = (_Float16)f;
    uint16_t u;
    memcpy(&u, &h, 2);
    return u;
}
static float h2f(uint16_t u) {
    _Float16 h;
    memcpy(&h, &u, 2);
    return (float)h;
}
static uint16_t f2bf(float f) { // round to nearest even
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return u >> 16;
}
static float bf2f(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static float u2f(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static uint32_t f2u(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

template<int MODE>
static int run(int ntests, float scale) {
    const bool bf = MODE == 1;
    std::mt19937 rng(1234 + MODE);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto enc = [&](float x) { return bf ? f2bf(x) : f2h(x); };
    auto dec = [&](uint16_t x) { return bf ? bf2f(x) : h2f(x); };

    std::vector<uint4> hA(ntests * 32), hB(ntests * 32), hC(ntests * 64);
    std::vector<std::vector<double>> ref(ntests, std::vector<double>(256));
    std::vector<std::vector<double>> mag(ntests, std::vector<double>(256));
    for (int it = 0; it < ntests; it++) {
        uint16_t a[16][16], b[16][16]; // a[m][k], b[k][n]
        float c[16][16];
        for (int i = 0; i < 16; i++)
            for (int j = 0; j < 16; j++) {
                a[i][j] = enc(nd(rng) * scale);
                b[i][j] = enc(nd(rng) * scale);
                c[i][j] = nd(rng) * scale * scale;
                if (MODE == 2)
                    c[i][j] = h2f(f2h(c[i][j]));
            }
        if (it == 0) { // exact small integers: any layout or ordering difference is visible
            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 16; j++) {
                    a[i][j] = enc((float)((i * 7 + j * 3) % 11 - 5));
                    b[i][j] = enc((float)((i * 5 + j * 13) % 9 - 4));
                    c[i][j] = (float)(i * 16 + j);
                }
        }
        auto pk = [](uint16_t lo, uint16_t hi) { return (uint32_t)lo | (uint32_t)hi << 16; };
        for (int l = 0; l < 32; l++) {
            const int g = l / 4, t = l % 4;
            hA[it * 32 + l] = uint4(pk(a[g][2 * t], a[g][2 * t + 1]),
                                    pk(a[g + 8][2 * t], a[g + 8][2 * t + 1]),
                                    pk(a[g][2 * t + 8], a[g][2 * t + 9]),
                                    pk(a[g + 8][2 * t + 8], a[g + 8][2 * t + 9]));
            hB[it * 32 + l] = uint4(pk(b[2 * t][g], b[2 * t + 1][g]),
                                    pk(b[2 * t + 8][g], b[2 * t + 9][g]),
                                    pk(b[2 * t][g + 8], b[2 * t + 1][g + 8]),
                                    pk(b[2 * t + 8][g + 8], b[2 * t + 9][g + 8]));
            if (MODE == 2) {
                hC[(it * 32 + l) * 2] = uint4(pk(f2h(c[g][2 * t]), f2h(c[g][2 * t + 1])),
                                              pk(f2h(c[g + 8][2 * t]), f2h(c[g + 8][2 * t + 1])),
                                              pk(f2h(c[g][2 * t + 8]), f2h(c[g][2 * t + 9])),
                                              pk(f2h(c[g + 8][2 * t + 8]), f2h(c[g + 8][2 * t + 9])));
            } else {
                for (int f = 0; f < 2; f++)
                    hC[(it * 32 + l) * 2 + f] = uint4(f2u(c[g][2 * t + 8 * f]),
                                                      f2u(c[g][2 * t + 1 + 8 * f]),
                                                      f2u(c[g + 8][2 * t + 8 * f]),
                                                      f2u(c[g + 8][2 * t + 1 + 8 * f]));
            }
        }
        for (int m = 0; m < 16; m++)
            for (int n = 0; n < 16; n++) {
                double s = c[m][n], ms = fabs(c[m][n]);
                for (int k = 0; k < 16; k++) {
                    s += (double)dec(a[m][k]) * dec(b[k][n]);
                    ms += fabs((double)dec(a[m][k]) * dec(b[k][n]));
                }
                ref[it][m * 16 + n] = s;
                mag[it][m * 16 + n] = ms;
            }
    }
    uint4 *dA, *dB, *dC, *dR, *dW;
    CHECK(hipMalloc(&dA, hA.size() * 16));
    CHECK(hipMalloc(&dB, hB.size() * 16));
    CHECK(hipMalloc(&dC, hC.size() * 16));
    CHECK(hipMalloc(&dR, hC.size() * 16));
    CHECK(hipMalloc(&dW, hC.size() * 16));
    CHECK(hipMemcpy(dA, hA.data(), hA.size() * 16, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dB, hB.data(), hB.size() * 16, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dC, hC.data(), hC.size() * 16, hipMemcpyHostToDevice));
    CHECK(hipMemset(dR, 0, hC.size() * 16));
    CHECK(hipMemset(dW, 0, hC.size() * 16));
    kern<MODE><<<1, 32>>>(dA, dB, dC, dR, dW, ntests);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<uint4> hR(hC.size()), hW(hC.size());
    CHECK(hipMemcpy(hR.data(), dR, hR.size() * 16, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(hW.data(), dW, hW.size() * 16, hipMemcpyDeviceToHost));

    // unpack both results into [m][n]
    long identical = 0, total = 0, bad_ref = 0, bad_w = 0, first_exact_bad = 0, wmma_inexact = 0;
    double max_err_w = 0, max_err_r = 0, max_diff = 0; // errors relative to the magnitude sum
    for (int it = 0; it < ntests; it++) {
        float R[16][16], W[16][16];
        for (int l = 0; l < 32; l++) {
            const int g = l / 4, t = l % 4;
            for (int f = 0; f < 2; f++) {
                float r4[4], w4[4];
                if (MODE == 2) {
                    const uint4 r = hR[(it * 32 + l) * 2], w = hW[(it * 32 + l) * 2];
                    const uint32_t rr[2] = {f ? r.z : r.x, f ? r.w : r.y}, ww[2] = {f ? w.z : w.x, f ? w.w : w.y};
                    for (int q = 0; q < 2; q++) {
                        r4[2 * q] = h2f(rr[q] & 0xffff), r4[2 * q + 1] = h2f(rr[q] >> 16);
                        w4[2 * q] = h2f(ww[q] & 0xffff), w4[2 * q + 1] = h2f(ww[q] >> 16);
                    }
                } else {
                    const uint4 r = hR[(it * 32 + l) * 2 + f], w = hW[(it * 32 + l) * 2 + f];
                    r4[0] = u2f(r.x), r4[1] = u2f(r.y), r4[2] = u2f(r.z), r4[3] = u2f(r.w);
                    w4[0] = u2f(w.x), w4[1] = u2f(w.y), w4[2] = u2f(w.z), w4[3] = u2f(w.w);
                }
                const int n = 2 * t + 8 * f;
                R[g][n] = r4[0], R[g][n + 1] = r4[1], R[g + 8][n] = r4[2], R[g + 8][n + 1] = r4[3];
                W[g][n] = w4[0], W[g][n + 1] = w4[1], W[g + 8][n] = w4[2], W[g + 8][n + 1] = w4[3];
            }
        }
        for (int m = 0; m < 16; m++)
            for (int n = 0; n < 16; n++) {
                const double r = ref[it][m * 16 + n], ms = mag[it][m * 16 + n] + 1e-30;
                const double er = fabs(R[m][n] - r) / ms, ew = fabs(W[m][n] - r) / ms;
                const double d = fabs((double)R[m][n] - W[m][n]) / ms;
                max_err_r = fmax(max_err_r, er);
                max_err_w = fmax(max_err_w, ew);
                max_diff  = fmax(max_diff, d);
                identical += f2u(R[m][n]) == f2u(W[m][n]);
                total++;
                // bound: 17 additions in fp32 (2^-24 each, doubled for slack), plus the final f16 rounding
                const double tol = (MODE == 2) ? 17 * 2 * ldexp(1.0, -24) + (ldexp(1.0, -11) * fabs(r) + ldexp(1.0, -25)) / ms
                                               : 17 * 2 * ldexp(1.0, -24);
                bad_ref += er > tol;
                bad_w += ew > tol;
                // exact integer case: the reference sums exactly; gfx11 WMMA does not round to nearest and may be
                // an ulp off, so it is only counted
                if (it == 0) {
                    first_exact_bad += R[m][n] != (float)r;
                    wmma_inexact += W[m][n] != (float)r;
                }
            }
    }
    printf("mode %d scale %g: %ld/%ld bitwise identical to reference, exact-integer case: reference wrong %ld, wmma inexact %ld/256, "
           "max |err|/sum|terms|: reference %.3g, wmma %.3g, between them %.3g; over bound: reference %ld, wmma %ld\n",
           MODE, scale, identical, total, first_exact_bad, wmma_inexact, max_err_r, max_err_w, max_diff, bad_ref, bad_w);
    return first_exact_bad + bad_ref + bad_w;
}

int main() {
    int bad = 0;
    for (float scale : {1.0f, 0.01f, 30.0f}) {
        bad += run<0>(2000, scale);
        bad += run<1>(2000, scale);
        bad += run<2>(2000, scale == 30.0f ? 4.0f : scale);
    }
    printf(bad ? "FAIL\n" : "PASS\n");
    return bad != 0;
}
```

## build.sh

`build.sh <gfx>`: applies the one substitution the build's hipify makes to `mma_rocm.cuh` (`__nv_bfloat162` to `__hip_bfloat162`), then compiles `test.hip` with the extension's flags. It expects `inc/src/common.h` and `inc/src/kernels/utils.cuh` to be the hipified `common_hip.h` and `kernels/utils_hip.cuh` copied from any built tree. Paths are this job's; adjust `J`.

```bash
#!/bin/bash
# usage: build.sh <arch>   (hipifies the fork's mma_rocm.cuh the way the build does, then compiles test.hip)
J=/var/lib/jenkins/fleet/jobs/rx7900-20261123-210849-port2-nunchaku-rdna-prompt
T=$J/moat/agent_space/nunchaku/wmma16
export PATH=$J/venv/bin:$PATH
rm -f $T/inc/src/kernels/zgemm/mma_rocm.cuh
sed 's/__nv_bfloat162/__hip_bfloat162/g' $J/moat/projects/nunchaku/src/src/kernels/zgemm/mma_rocm.cuh > $T/inc/src/kernels/zgemm/mma_rocm.cuh
cd $T && hipcc -O3 -std=c++20 --offload-arch=$1 -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DCUDA_HAS_FP16=1 -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 -DENABLE_BF16=1 -DFMT_CONSTEVAL= -I inc/src -I $J/ref-src/third_party/spdlog/include $2 test.hip -o test_$1$3
```

## f16_dump.py

`f16_dump.py`: outputs of every kernel that uses the 16-bit product (it imports `mma_dump.py` from the same directory for the GEMM cases).

```python
"""Outputs of every kernel that uses the 16-bit floating-point m16n8k16 product, for comparing two builds.

    python f16_dump.py dump OUT.pt           # from the repository root, PYTHONPATH=., with one build
    python f16_dump.py compare REF.pt NEW.pt  # REF: reference build, NEW: matrix-instruction build

Cases: the W4A4 GEMM cases of mma_dump.py (their LoRA-up and LoRA-down epilogues use the product), the
nunchaku-fp16 attention kernel, and the AWQ W4A16 GEMM. Where a float64 reference is cheap it is stored too,
so that compare reports each build's own error next to the difference between the builds.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def attention_cases():
    from tests.ops.test_attention_fp16 import HEAD_DIM, pack_k, pack_q, pack_v

    from nunchaku._C.ops import attention_fp16

    for out_dtype in (torch.float16, torch.bfloat16):
        for tokens, valid in [(256, 256), (1024, 1024), (512, 500), (4608, 4608)]:
            g = torch.Generator().manual_seed(tokens + valid + (out_dtype == torch.bfloat16))
            b, h = 1, 4
            q, k, v = (torch.randn(b, h, tokens, HEAD_DIM, generator=g).half() for _ in range(3))
            k[:, :, valid:] = float("nan")
            v[:, :, valid:] = 0
            scale = HEAD_DIM**-0.5
            o = torch.empty(b, tokens, h * HEAD_DIM, dtype=out_dtype, device="cuda")
            attention_fp16(pack_q(q).cuda(), pack_k(k).cuda(), pack_v(v).cuda(), o, scale)
            ref = torch.nn.functional.scaled_dot_product_attention(
                q.cuda().double(), k[:, :, :valid].cuda().double(), v[:, :, :valid].cuda().double(), scale=scale
            )
            ref = ref.transpose(1, 2).reshape(b, tokens, h * HEAD_DIM)
            yield f"attention_{out_dtype}_{tokens}_{valid}", o, ref


def awq_cases():
    from tests.ops.test_awq import make_layer

    from nunchaku._C.ops import gemm_awq

    for dtype in (torch.float16, torch.bfloat16):
        for m, n, k in [(33, 512, 1024), (150, 512, 1024), (512, 4096, 4096), (1024, 1024, 4096)]:
            weight, qweight, qscale, qzero = make_layer(n, k, 128, dtype, seed=m + n + k)
            x = torch.randn(m, k, generator=torch.Generator().manual_seed(m + 1)).to(dtype)
            out = gemm_awq(x.cuda(), qweight, qscale, qzero)
            ref = x.cuda().double() @ weight.cuda().double().T
            yield f"awq_{dtype}_{m}x{n}x{k}", out, ref


def dump(path):
    import mma_dump

    results = {}
    for name, value in mma_dump.cases():
        torch.cuda.synchronize()
        results[name] = tuple(v.cpu() for v in value) if isinstance(value, tuple) else (value.cpu(),)
    for gen in (attention_cases(), awq_cases()):
        for name, out, ref in gen:
            torch.cuda.synchronize()
            results[name] = (out.cpu(), ref.cpu())
    torch.save(results, path)
    print(f"saved {len(results)} results to {path}")


def ordered(t):
    """16-bit floats as integers in value order, so that differences are distances in ulps."""
    i = t.contiguous().view(torch.int16).int()
    return torch.where(i < 0, -(i & 0x7FFF), i)


def compare(ref_path, new_path):
    a, b = torch.load(ref_path), torch.load(new_path)
    assert a.keys() == b.keys(), "different case lists"
    worst = {}
    for name in a:
        va, vb = a[name], b[name]
        has_ref = name.startswith(("attention", "awq"))
        outs = [(va[0], vb[0])] if has_ref else list(zip(va, vb))
        for i, (x, y) in enumerate(outs):
            same = torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))
            kind = name.split("_")[0] + (f"[{i}]" if not has_ref else "")
            if x.dtype in (torch.float16, torch.bfloat16):
                fin = torch.isfinite(x) & torch.isfinite(y)
                assert torch.equal(torch.isfinite(x), torch.isfinite(y)), f"{name}: non-finite pattern differs"
                d = (ordered(x) - ordered(y)).abs()[fin]
                ndiff = (d != 0).sum().item()
                maxd = d.max().item() if d.numel() else 0
                line = f"max {maxd} ulp, {ndiff}/{x.numel()} differ"
                worst[kind] = max(worst.get(kind, 0), maxd)
            elif x.dtype == torch.uint8:  # packed 4-bit activations
                nx = torch.stack([x & 0xF, x >> 4], -1).int()
                ny = torch.stack([y & 0xF, y >> 4], -1).int()
                d = (nx - ny).abs()
                ndiff = (d != 0).sum().item()
                line = f"{ndiff}/{nx.numel()} nibbles differ, max step {d.max().item()}"
                worst[kind] = max(worst.get(kind, 0), d.max().item())
            else:
                rel = ((x.double() - y.double()).abs().max() / x.double().abs().max()).item()
                line = f"max rel diff {rel:.3g}"
                worst[kind] = max(worst.get(kind, 0), rel)
            if has_ref:
                r = va[1].double()
                fin = torch.isfinite(r)
                scale = r[fin].abs().max()
                ea = ((x.double() - r).abs()[fin].max() / scale).item()
                eb = ((y.double() - r).abs()[fin].max() / scale).item()
                line += f"; error vs float64 (max rel to max |ref|): reference build {ea:.3g}, new build {eb:.3g}"
            print(f"{'SAME' if same else 'DIFF'} {name}{'' if has_ref else f'[{i}]'} {tuple(x.shape)} {x.dtype}: {line}")
    print("worst per output kind:", {k: (round(v, 9) if isinstance(v, float) else v) for k, v in worst.items()})


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2])
    else:
        compare(sys.argv[2], sys.argv[3])
```

## reldiff.py

`reldiff.py REF.pt NEW.pt`: max |difference| / max |output| per case for two `f16_dump.py` dumps.

```python
"""max |new - ref| / max |ref build output| per case, plus where the worst ulp gap sits (f16_dump files)."""
import sys, torch
a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
for name in a:
    va, vb = a[name], b[name]
    outs = [(va[0], vb[0])] if name.startswith(("attention", "awq")) else list(zip(va, vb))
    for i, (x, y) in enumerate(outs):
        if not x.is_floating_point():
            continue
        fin = torch.isfinite(x) & torch.isfinite(y)
        x, y = x[fin].double(), y[fin].double()
        d = (x - y).abs()
        j = d.argmax()
        print(f"{name}[{i}] {x.dtype if False else ''}max|diff|/max|out| {(d.max() / x.abs().max()).item():.3g}  at value {x[j].item():.4g} vs {y[j].item():.4g}")
```

## speed.py

`speed.py`: kernel timings with one build.

```python
"""Time the kernels that use the 16-bit m16n8k16 product with one build (run from a repository root, PYTHONPATH=.)."""

import torch

from nunchaku._C.ops import attention_fp16, gemm_awq
from nunchaku.lora.flux.packer import NunchakuWeightPacker
from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda
from tests.ops.test_attention_fp16 import HEAD_DIM, pack_k, pack_q, pack_v
from tests.ops.test_awq import make_layer
from tests.ops.test_gemm_w4a4 import pack_act, pack_ascales


def bench(fn, iters=20):
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


packer = NunchakuWeightPacker(bits=4)
for dtype in (torch.float16, torch.bfloat16):
    m, n, k, rank, G = 4096, 3072, 3072, 32, 64
    g = torch.Generator().manual_seed(0)
    act = pack_act(torch.randint(-8, 8, (m, k), dtype=torch.int32, generator=g)).cuda()
    wgt = packer.pack_weight(torch.randint(-8, 8, (n, k), dtype=torch.int32, generator=g)).cuda()
    asc = pack_ascales((torch.rand(m, k // G, generator=g) * 0.02).to(dtype)).cuda()
    wsc = packer.pack_scale((torch.rand(n, k // G, generator=g) * 0.02).to(dtype), group_size=G).cuda()
    for r in (rank, 128):
        if True:
            kw = dict(
                lora_act_in=(torch.randn(m, r) * 0.1).cuda(),
                lora_up=packer.pack_lowrank_weight((torch.randn(n, r) * 0.1).to(dtype), down=False).cuda(),
            )
        out = torch.empty(m, n, dtype=dtype, device="cuda")
        t = bench(lambda: svdq_gemm_w4a4_cuda(act=act, wgt=wgt, out=out, ascales=asc, wscales=wsc, **kw))
        print(f"gemm_w4a4 {dtype} {m}x{n}x{k} lora rank {r}: {t:.3f} ms")

    weight, qweight, qscale, qzero = make_layer(4096, 4096, 128, dtype, seed=1)
    x = torch.randn(512, 4096).to(dtype).cuda()
    print(f"gemm_awq {dtype} 512x4096x4096: {bench(lambda: gemm_awq(x, qweight, qscale, qzero)):.3f} ms")

b, h, tokens = 1, 24, 4608
q, kk, v = (torch.randn(b, h, tokens, HEAD_DIM).half() for _ in range(3))
q, kk, v = pack_q(q).cuda(), pack_k(kk).cuda(), pack_v(v).cuda()
o = torch.empty(b, tokens, h * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
print(f"attention_fp16 24 heads x {tokens} tokens: {bench(lambda: attention_fp16(q, kk, v, o, HEAD_DIM**-0.5)):.3f} ms")
```
