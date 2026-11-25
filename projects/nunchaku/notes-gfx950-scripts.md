# nunchaku: gfx950 round scripts (porter, linux-gfx950, 2026-11-24)

Scripts of the gfx950 porter round in `notes.md`. `mma_dump.py` is in `notes.md`; `f16_dump.py`, `reldiff.py`, `speed.py` and `build.sh` are in `notes-rdna-scripts.md` (unchanged). Run the Python ones from the fork root with the extension built in place and `PYTHONPATH=.`.

## test_cdna.hip

The RDNA round's `test.hip` (in `notes-rdna-scripts.md`) adapted to 64-lane wavefronts: each 32-lane warp computes its own problem, the launch is 64 threads, and `NANTEST=1` puts NaN and infinity into the upper warp's problems only and checks the lower warp's (the cross-warp isolation check). Build exactly as `build.sh` there, with `--offload-arch=gfx950` (or gfx942, gfx90a).

Changed 2026-11-25 (jeff's ruling on the gfx90a f16 MFMA subnormal flush): when the GPU it runs on is gfx90a, every f16 A and B input whose f16 value is subnormal is set to a signed zero before both the device and the host reference see it, because the MI200 FP16 matrix instructions flush subnormal inputs to zero (documented behavior; PyTorch's numerical-accuracy notes, "Reduced Precision FP16 and BF16 GEMMs and Convolutions on AMD Instinct MI200 devices"). It prints `gfx90a: subnormal f16 inputs zeroed` when it does. bf16 inputs, C, and every other GPU keep the strict check unchanged. `FTZ_INPUTS=1` (the gfx90a evidence run's switch) still forces the same zeroing on any GPU.

```cpp
// Checks mma_m16n16k16_* (WMMA on RDNA) against the reference m16n8k16 lane-gather functions and a float64 host
// reference, on random 16x16x16 problems in the mma.sync fragment layouts.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include "kernels/zgemm/mma_rocm.cuh"
#ifndef NTHREADS
#define NTHREADS 64
#endif

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
    for (int it = threadIdx.x / 32; it < ntests; it += blockDim.x / 32) {
        const int l   = threadIdx.x % 32;
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

// MI200 (gfx90a) FP16 matrix instructions read subnormal f16 inputs as zero (documented hardware behavior); there the
// test data carries no subnormal f16 inputs, so the strict bound below still applies to everything else
static bool ftz_f16_inputs = false;

template<int MODE>
static int run(int ntests, float scale) {
    const bool bf = MODE == 1;
    std::mt19937 rng(1234 + MODE);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto enc = [&](float x) {
        uint16_t v = bf ? f2bf(x) : f2h(x);
        if (ftz_f16_inputs && !bf && (v & 0x7c00) == 0)
            v &= 0x8000;
        return v;
    };
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
        if (getenv("NANTEST") && it % 2 == 1) { // non-finite values in the upper warp's problems only
            a[3][5] = enc(NAN);
            b[7][2] = enc(INFINITY);
            b[1][12] = enc(NAN);
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
    kern<MODE><<<1, NTHREADS>>>(dA, dB, dC, dR, dW, ntests);
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
        if (getenv("NANTEST") && it % 2 == 1)
            continue;
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
    printf("mode %d scale %g: %ld/%ld bitwise identical to reference, exact-integer case: reference wrong %ld, mfma inexact %ld/256, "
           "max |err|/sum|terms|: reference %.3g, mfma %.3g, between them %.3g; over bound: reference %ld, mfma %ld\n",
           MODE, scale, identical, total, first_exact_bad, wmma_inexact, max_err_r, max_err_w, max_diff, bad_ref, bad_w);
    return first_exact_bad + bad_ref + bad_w;
}

int main() {
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    ftz_f16_inputs = strncmp(prop.gcnArchName, "gfx90a", 6) == 0 || getenv("FTZ_INPUTS");
    if (ftz_f16_inputs)
        printf("%s: subnormal f16 inputs zeroed\n", prop.gcnArchName);
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

## force_fp4.py

pytest plugin (`-p force_fp4`, its directory on `PYTHONPATH`) that makes `get_precision()` return `fp4` for `auto`, including in modules that imported it by name, so the unchanged tests run on the NVFP4 checkpoints.

```python
"""pytest plugin: make nunchaku.utils.get_precision() pick fp4 for "auto", to run the tests on the NVFP4 checkpoints."""
import nunchaku.utils as u

_orig = u.get_precision


def get_precision(precision="auto", device="cuda", pretrained_model_name_or_path=None):
    return _orig("fp4" if precision == "auto" else precision, device, pretrained_model_name_or_path)


u.get_precision = get_precision

# modules that imported the function by name before this plugin ran
import sys

for _name, _mod in list(sys.modules.items()):
    if _name.startswith("nunchaku") and getattr(_mod, "get_precision", None) is _orig:
        _mod.get_precision = get_precision
```

## fp4_speed.py

Kernel-level W4A4 GEMM time, INT4 against NVFP4.

```python
"""W4A4 GEMM time, INT4 against NVFP4, same shapes, LoRA rank 32, bf16 output.  From the fork root, PYTHONPATH=."""

import torch

from nunchaku.lora.flux.packer import NunchakuWeightPacker
from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda
from tests.ops.test_gemm_w4a4 import pack_act, pack_ascales
from tests.ops.test_gemm_w4a4_fp4 import pack_act_fp4, pack_ascales_fp4, random_scales


def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


dtype, rank = torch.bfloat16, 32
packer = NunchakuWeightPacker(bits=4)
g = torch.Generator().manual_seed(0)
for m, n, k in [(4096, 3072, 3072), (4096, 12288, 3072), (4096, 3072, 12288)]:
    out = torch.empty(m, n, dtype=dtype, device="cuda")
    lora = dict(
        lora_act_in=torch.randn(m, rank, device="cuda") * 0.1,
        lora_up=packer.pack_lowrank_weight((torch.randn(n, rank, generator=g) * 0.1).to(dtype), down=False).cuda(),
    )
    int4 = dict(
        act=pack_act(torch.randint(-8, 8, (m, k), dtype=torch.int32, generator=g)).cuda(),
        wgt=packer.pack_weight(torch.randint(-8, 8, (n, k), dtype=torch.int32, generator=g)).cuda(),
        ascales=pack_ascales((torch.rand(m, k // 64, generator=g) * 0.02).to(dtype)).cuda(),
        wscales=packer.pack_scale((torch.rand(n, k // 64, generator=g) * 0.02).to(dtype), group_size=64).cuda(),
    )
    fp4 = dict(
        act=pack_act_fp4(torch.randint(0, 16, (m, k), generator=g)).cuda(),
        wgt=packer.pack_weight(torch.randint(0, 16, (n, k), dtype=torch.int32, generator=g)).cuda(),
        ascales=pack_ascales_fp4(random_scales((m, k // 16), g)).cuda(),
        wscales=packer.pack_scale(random_scales((n, k // 16), g).to(dtype), group_size=16).cuda(),
        fp4=True,
        alpha=0.5,
        wcscales=packer.pack_scale(torch.ones(n, dtype=dtype), group_size=-1).cuda(),
    )
    t_int4 = bench(lambda: svdq_gemm_w4a4_cuda(out=out, **int4, **lora))
    t_fp4 = bench(lambda: svdq_gemm_w4a4_cuda(out=out, **fp4, **lora))
    print(f"W4A4 GEMM {m}x{n}x{k} bf16 rank {rank}: int4 {t_int4:.3f} ms, fp4 {t_fp4:.3f} ms ({t_fp4 / t_int4:.2f}x)")
```

## zimage_speed.py

Z-Image Turbo per-image latency, INT4 against NVFP4 of the same rank.

```python
"""Per-image latency of the Z-Image Turbo pipeline (1024x1024, 9 steps, bf16) with the INT4 and NVFP4 nunchaku
transformers of the same rank, and the time spent in the transformer.  Run from the fork root, PYTHONPATH=.

    python zimage_speed.py <rank> <precision> [<precision> ...]
"""

import sys
import time

import torch
from diffusers import ZImagePipeline

from nunchaku import NunchakuZImageTransformer2DModel

rank = int(sys.argv[1])
prompt = "Table Mountain, South Africa, covered in clouds on a hot, bright summers day."
for precision in sys.argv[2:]:
    # the loader asks get_precision() for the device's precision instead of reading the checkpoint
    import nunchaku.models.transformers.transformer_zimage as tz

    tz.get_precision = lambda *args, **kwargs: precision
    path = f"nunchaku-tech/nunchaku-z-image-turbo/svdq-{precision}_r{rank}-z-image-turbo.safetensors"
    transformer = NunchakuZImageTransformer2DModel.from_pretrained(path, torch_dtype=torch.bfloat16)
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", transformer=transformer, torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    spent = []
    forward = pipe.transformer.forward

    def timed(*args, **kwargs):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = forward(*args, **kwargs)
        torch.cuda.synchronize()
        spent.append(time.perf_counter() - t)
        return out

    pipe.transformer.forward = timed
    kwargs = dict(width=1024, height=1024, num_inference_steps=9, guidance_scale=0.0)
    pipe(prompt, generator=torch.Generator().manual_seed(0), **kwargs)  # warm-up
    totals = []
    for i in range(3):
        spent.clear()
        torch.cuda.synchronize()
        t = time.perf_counter()
        pipe(prompt, generator=torch.Generator().manual_seed(i), **kwargs)
        torch.cuda.synchronize()
        totals.append((time.perf_counter() - t, sum(spent), len(spent)))
    for total, tr, n in totals:
        print(f"{precision} r{rank}: image {total:.3f} s, transformer {tr:.3f} s over {n} calls")
    del pipe, transformer
    torch.cuda.empty_cache()
```

## FLUX loader round (2026-11-25): scripts

Scripts of "Porter round 2026-11-25, gfx950: FLUX loader precision, gfx90a f16 flush ruling" in `notes.md`. Paths are this job's; adjust `J`. `test_zimage_fp4_public.py` is in `notes-gfx1250-scripts.md` (unchanged).

### flux_loader_matrix.py

`python -I flux_loader_matrix.py <tree> <out.json>` with `HF_HOME` set; run once with the 555105f tree and once with the new tree (each with the same built `_C`), then compare the two JSON files row by row.

```python
"""Load the public FLUX checkpoint types the tests use through NunchakuFluxTransformer2dModel.from_pretrained on faked
GPUs (CUDA capabilities, ROCm architectures) and unfaked, and record the precision the loader builds the quantized
model with and whether the load succeeds.

    python -I flux_loader_matrix.py <tree> <out.json>

<tree> is a built nunchaku checkout. The C++ model can only be built on a GPU, so the fakes cover the loader's
precision choice and its check_hardware_compatibility call; they are undone where the loader builds the quantized
model (load_quantized_module), which then runs unfaked on this host's GPU with the precision the loader chose. Each case
runs in its own process (a model built for the wrong precision may abort instead of raising).
"""

import json
import subprocess
import sys

CHECKPOINTS = [
    ("nunchaku-tech/nunchaku-flux.1-schnell", "svdq-int4_r32-flux.1-schnell.safetensors"),
    ("nunchaku-tech/nunchaku-flux.1-schnell", "svdq-fp4_r32-flux.1-schnell.safetensors"),
    ("mit-han-lab/nunchaku-shuttle-jaguar", "svdq-int4_r32-shuttle-jaguar.safetensors"),
    ("mit-han-lab/nunchaku-shuttle-jaguar", "svdq-fp4_r32-shuttle-jaguar.safetensors"),
]
DEVICES = (
    [["cuda", list(c)] for c in [(7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0), (12, 1)]]
    + [["rocm", a] for a in ["gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1201", "gfx1250"]]
    + [["real", "here"]]
)

CHILD = r"""
import json, sys, types
tree, repo, name, kind, device = json.loads(sys.argv[1])
sys.path.insert(0, tree)
import torch
from huggingface_hub import hf_hub_download
import nunchaku
from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.models.transformers import transformer_flux
assert nunchaku.__file__.startswith(tree), nunchaku.__file__
path = hf_hub_download(repo, name)
orig = (torch.version.hip, torch.cuda.get_device_capability, torch.cuda.get_device_properties)
def restore():
    torch.version.hip, torch.cuda.get_device_capability, torch.cuda.get_device_properties = orig
if kind == "cuda":
    torch.version.hip = None
    torch.cuda.get_device_capability = lambda *a, **k: tuple(device)
    def no_props(*a, **k):
        raise AssertionError("ROCm architecture asked on a CUDA device")
    torch.cuda.get_device_properties = no_props
elif kind == "rocm":
    torch.cuda.get_device_properties = lambda *a, **k: types.SimpleNamespace(gcnArchName=f"{device}:sramecc+:xnack-")
built = []
original = transformer_flux.load_quantized_module
def spy(*args, **kwargs):
    built.append("fp4" if kwargs["use_fp4"] else "int4")
    restore()
    return original(*args, **kwargs)
transformer_flux.load_quantized_module = spy
try:
    model = NunchakuFluxTransformer2dModel.from_pretrained(path, torch_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    status = "ok"
except Exception as e:
    status = f"{type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"
print("RESULT " + json.dumps(dict(built=built, status=status)), flush=True)
"""

tree, out = sys.argv[1], sys.argv[2]
results = []
for repo, name in CHECKPOINTS:
    for kind, device in DEVICES:
        arg = json.dumps([tree, repo, name, kind, device])
        try:
            p = subprocess.run(
                [sys.executable, "-I", "-c", CHILD, arg], capture_output=True, text=True, timeout=900, cwd="/tmp"
            )
            lines = [line for line in p.stdout.splitlines() if line.startswith("RESULT ")]
            if lines:
                r = json.loads(lines[-1][7:])
            else:
                tail = (p.stderr.strip().splitlines() or ["?"])[-1][:160]
                r = dict(built=None, status=f"process exit {p.returncode}: {tail}")
        except subprocess.TimeoutExpired:
            r = dict(built=None, status="timeout")
        row = dict(checkpoint=name, device=f"{kind}:{device}", **r)
        print(json.dumps(row), flush=True)
        results.append(row)
json.dump(results, open(out, "w"), indent=1)
```

### no_token.py

pytest plugin (`-p no_token` with its directory on `PYTHONPATH`): the FLUX tests read their prompts through `datasets.load_dataset(..., token=True)`, which fails with no Hugging Face login although `playgroundai/MJHQ-30K` is public. The dataset script also downloads the 3.6 GB image archive even when no ground-truth images are requested.

```python
"""pytest plugin: the FLUX tests read their prompts with datasets.load_dataset(..., token=True), which fails without a
Hugging Face login although the dataset (playgroundai/MJHQ-30K) is public. Drop that one argument; nothing else changes.

    python -m pytest -p no_token ...   (with this directory on PYTHONPATH)
"""

import datasets

_load_dataset = datasets.load_dataset


def _load_dataset_public(*args, **kwargs):
    if kwargs.get("token") is True:
        kwargs["token"] = None
    return _load_dataset(*args, **kwargs)


datasets.load_dataset = _load_dataset_public
```

### test_shuttle_fp4_public.py

`python -m pytest -v -s -p no_token <path>/test_shuttle_fp4_public.py` from the fork root.

```python
"""Upstream's shuttle-jaguar FLUX test (tests/flux/test_shuttle_jaguar.py) on its NVFP4 checkpoint.

The upstream test passes get_precision() ("int4" on AMD GPUs) to run_test, which picks the checkpoint file and the
LPIPS threshold from it. Here run_test gets "fp4" and upstream's fp4 threshold, with the test's other parameters
unchanged. nunchaku is not patched: NunchakuFluxTransformer2dModel.from_pretrained picks the precision itself; the
precision it builds the quantized model with is recorded and checked.

Run from the fork root:  python -m pytest -v -s -p no_token <this file>
"""

import pytest

from nunchaku.models.transformers import transformer_flux
from tests.flux.utils import run_test

chosen = []
_load_quantized_module = transformer_flux.load_quantized_module


def _record(*args, **kwargs):
    chosen.append("fp4" if kwargs["use_fp4"] else "int4")
    return _load_quantized_module(*args, **kwargs)


@pytest.fixture(autouse=True)
def loader_precision(monkeypatch):
    chosen.clear()
    monkeypatch.setattr(transformer_flux, "load_quantized_module", _record)
    yield
    print(f"loader precision: {chosen}")
    assert chosen == ["fp4"]


@pytest.mark.parametrize(
    "height,width,attention_impl,cpu_offload,expected_lpips", [(1024, 1024, "nunchaku-fp16", False, 0.148)]
)
def test_shuttle_jaguar_fp4(height, width, attention_impl, cpu_offload, expected_lpips):
    run_test(
        precision="fp4",
        model_name="shuttle-jaguar",
        height=height,
        width=width,
        attention_impl=attention_impl,
        cpu_offload=cpu_offload,
        expected_lpips=expected_lpips,
    )
```

### flux_e2e.sh

```bash
#!/bin/bash
# usage: flux_e2e.sh <tree>   (upstream's shuttle-jaguar test on int4, then the NVFP4 wrapper; prompts via no_token)
J=/var/lib/jenkins/fleet/jobs/mi355-20261125-131653-port6-nunchaku-flux-ftz-prompt
cd "$1" || exit 1
ulimit -c 0
export PATH=$FLEET_VENV/bin:$PATH HF_HOME=$J/hf HF_HUB_DISABLE_XET=1 NUNCHAKU_TEST_CACHE_ROOT=$J/test_cache PYTHONPATH=$J/scripts
timeout 5400 $FLEET_VENV/bin/python -m pytest -v -s -p no_token tests/flux/test_shuttle_jaguar.py > $J/w/e2e_shuttle_int4.log 2>&1
r1=$?
timeout 5400 $FLEET_VENV/bin/python -m pytest -v -s -p no_token $J/scripts/test_shuttle_fp4_public.py > $J/w/e2e_shuttle_fp4.log 2>&1
r2=$?
echo "int4 exit $r1, fp4 exit $r2"
[ $r1 -eq 0 ] && [ $r2 -eq 0 ]
```

### tcdna/build.sh

Compiles `test_cdna.hip` (above) against the built tree's hipified headers: `inc/src/common.h` = the tree's `src/common_hip.h`, `inc/src/kernels/utils.cuh` = `src/kernels/utils_hip.cuh`, `inc/src/kernels/zgemm/mma_rocm.cuh` = the fork's `mma_rocm.cuh` with `__nv_bfloat162` replaced by `__hip_bfloat162`.

```bash
#!/bin/bash
# usage: build.sh <arch>   (compiles test_cdna.hip with the extension's flags against the built tree's hipified headers)
J=/var/lib/jenkins/fleet/jobs/mi355-20261125-131653-port6-nunchaku-flux-ftz-prompt
export PATH=$J/venv/bin:$PATH
cd $J/scripts/tcdna && $J/venv/lib/python3.11/site-packages/_rocm_sdk_devel/bin/hipcc -O3 -std=c++20 --offload-arch=$1 -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DCUDA_HAS_FP16=1 -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 -DENABLE_BF16=1 -DFMT_CONSTEVAL= -I inc/src -I $J/w/src/third_party/spdlog/include test_cdna.hip -o test_$1
```
