# nunchaku: gfx1250 WMMA and loader-precision round scripts (porter, linux-gfx950, 2026-11-25)

Scripts of "Porter round 2026-11-25, gfx950: checkpoint precision (Q1 a), gfx1250 WMMA (Q2 b)" in `notes.md`. `mma_dump.py` is in `notes.md`; `f16_dump.py`, `reldiff.py` and `speed.py` are in `notes-rdna-scripts.md`; `fp4_speed.py` and `zimage_speed.py` in `notes-gfx950-scripts.md` (all unchanged). Paths name this job's directory (`J`); adjust them. The Python scripts that use the extension run from the fork root with the extension built in place and `PYTHONPATH=.`.

## test_gfx1250.hip

The wave32 primitive test for the gfx1250 forms, to run first on the MI450. The f16/bf16 modes are `test_cdna.hip`'s (`notes-gfx950-scripts.md`), unchanged except for 32 threads (one wavefront); on gfx1250 they check `mma_m16n16k16_*` on `v_wmma_f32_16x16x32_{f16,bf16}`. The INT4 modes (signed and unsigned activations, 2000 random problems each, an extreme-value problem and a one-nonzero-per-row sign/nibble-order problem) load the fragments exactly as `gemm_w4a4_block_wmma` does and call `mma_rocm::wmma_i32_16x16x64_int4` (`v_wmma_i32_16x16x64_iu8`), checked bit for bit against the reference `mma_m16n8kx_s32common` and an exact host product. On other targets the INT4 WMMA part reports "not compiled"; `REQUIRE_INT4=1` makes that a failure. Expected on gfx1250: `PASS`, both INT4 lines `exact`. On a CDNA GPU build it with `-DNTHREADS=64` (the CDNA path needs the whole wavefront).

```cpp
// gfx1250 primitive test. Checks mma_m16n16k16_* (16x16x32 WMMA on gfx1250) against the reference m16n8k16
// lane-gather functions and a float64 host reference on random 16x16x16 problems in the mma.sync fragment layouts,
// and the INT4 16x16x64 product of the W4A4 GEMM main loop (mma_rocm::wmma_i32_16x16x64_int4, iu8 WMMA), with
// operands loaded exactly as gemm_w4a4_block_wmma loads them, bit for bit against the reference
// mma_m16n8kx_s32common and an exact host reference.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include "kernels/zgemm/mma_rocm.cuh"
#ifndef NTHREADS
#define NTHREADS 32
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
    printf("mode %d scale %g: %ld/%ld bitwise identical to reference, exact-integer case: reference wrong %ld, matrix-instruction path inexact %ld/256, "
           "max |err|/sum|terms|: reference %.3g, matrix path %.3g, between them %.3g; over bound: reference %ld, matrix path %ld\n",
           MODE, scale, identical, total, first_exact_bad, wmma_inexact, max_err_r, max_err_w, max_diff, bad_ref, bad_w);
    return first_exact_bad + bad_ref + bad_w;
}


// INT4: activations (rows m) as the m16n8k64 A fragment, weights (columns n) as the B fragments of columns 0-7
// ({w.x, w.y}) and 8-15 ({w.z, w.w}), as the GEMM stores them.
template<bool ACT_UNSIGNED>
__global__ void kern_int4(const uint4 *Act, const uint4 *Wgt, uint4 *Dref, int *Dw, int *compiled, int ntests) {
    using AT = std::conditional_t<ACT_UNSIGNED, mma_helper::u4, mma_helper::s4>;
    for (int it = threadIdx.x / 32; it < ntests; it += blockDim.x / 32) {
        const int l   = threadIdx.x % 32;
        const uint4 a = Act[it * 32 + l];
        const uint4 w = Wgt[it * 32 + l];
        Dref[(it * 32 + l) * 2]     = mma_m16n8kx_s32common<AT, mma_helper::s4>(a, uint2(w.x, w.y), uint4(0, 0, 0, 0));
        Dref[(it * 32 + l) * 2 + 1] = mma_m16n8kx_s32common<AT, mma_helper::s4>(a, uint2(w.z, w.w), uint4(0, 0, 0, 0));
#if defined(__gfx1250__) && defined(NUNCHAKU_ROCM_WMMA_INT4)
        // the loads of gemm_w4a4_block_wmma (gfx12 and gfx1250): step s takes chunk 2s + l / 16
        const uint32_t *aw = reinterpret_cast<const uint32_t *>(&Act[it * 32]);
        const uint32_t *ww = reinterpret_cast<const uint32_t *>(&Wgt[it * 32]);
        const int col = l % 16, hw = l / 16;
        uint32_t wa[4], wb[4];
        for (int s = 0; s < 4; s++) {
            const int c = 2 * s + hw;
            wb[s]       = aw[(4 * (col % 8) + c % 4) * 4 + (c < 4 ? 0 : 2) + (col < 8 ? 0 : 1)];
            wa[s]       = ww[(4 * (col % 8) + c % 4) * 4 + (col < 8 ? 0 : 2) + (c < 4 ? 0 : 1)];
        }
        const auto acc = mma_rocm::wmma_i32_16x16x64_int4<false, ACT_UNSIGNED>(wa, wb);
        for (int v = 0; v < 8; v++) {
            Dw[it * 256 + col * 16 + v + 8 * hw] = acc[v]; // [activation row][weight column], as the GEMM stores it
        }
        if (it == 0 && l == 0)
            *compiled = 1;
#endif
    }
}

template<bool ACT_UNSIGNED>
static int run_int4(int ntests) {
    std::mt19937 rng(77 + ACT_UNSIGNED);
    std::uniform_int_distribution<int> da(ACT_UNSIGNED ? 0 : -8, ACT_UNSIGNED ? 15 : 7), dw(-8, 7);
    std::vector<uint4> hA(ntests * 32), hW(ntests * 32);
    std::vector<int> ref(ntests * 256);
    for (int it = 0; it < ntests; it++) {
        int A[16][64], W[16][64]; // A[m][k], W[n][k]
        for (int i = 0; i < 16; i++)
            for (int k = 0; k < 64; k++) {
                A[i][k] = da(rng);
                W[i][k] = dw(rng);
                if (it == 0) { // largest magnitudes everywhere
                    A[i][k] = ACT_UNSIGNED ? 15 : -8;
                    W[i][k] = -8;
                }
                if (it == 1) { // sign and nibble order: one nonzero per row and column, at distinct k
                    A[i][k] = (k == 4 * i + 1) ? (ACT_UNSIGNED ? 9 : -7) : 0;
                    W[i][k] = (k == 4 * i + 1) ? 3 + i % 4 : (k == (4 * i + 6) % 64 ? -1 : 0);
                }
            }
        // word of row r, chunk c (k 8c .. 8c + 7, low nibble first)
        auto word = [](const int (&M)[16][64], int r, int c) {
            uint32_t x = 0;
            for (int e = 0; e < 8; e++)
                x |= (uint32_t)(M[r][8 * c + e] & 0xf) << (4 * e);
            return x;
        };
        for (int l = 0; l < 32; l++) {
            const int g = l / 4, t = l % 4;
            hA[it * 32 + l] = uint4(word(A, g, t), word(A, g + 8, t), word(A, g, t + 4), word(A, g + 8, t + 4));
            hW[it * 32 + l] = uint4(word(W, g, t), word(W, g, t + 4), word(W, g + 8, t), word(W, g + 8, t + 4));
        }
        for (int m = 0; m < 16; m++)
            for (int n = 0; n < 16; n++) {
                int s = 0;
                for (int k = 0; k < 64; k++)
                    s += A[m][k] * W[n][k];
                ref[it * 256 + m * 16 + n] = s;
            }
    }
    uint4 *dA, *dW, *dR;
    int *dD, *dC;
    CHECK(hipMalloc(&dA, hA.size() * 16));
    CHECK(hipMalloc(&dW, hW.size() * 16));
    CHECK(hipMalloc(&dR, hA.size() * 32));
    CHECK(hipMalloc(&dD, ref.size() * 4));
    CHECK(hipMalloc(&dC, 4));
    CHECK(hipMemcpy(dA, hA.data(), hA.size() * 16, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dW, hW.data(), hW.size() * 16, hipMemcpyHostToDevice));
    CHECK(hipMemset(dR, 0, hA.size() * 32));
    CHECK(hipMemset(dD, 0x55, ref.size() * 4));
    CHECK(hipMemset(dC, 0, 4));
    kern_int4<ACT_UNSIGNED><<<1, NTHREADS>>>(dA, dW, dR, dD, dC, ntests);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<uint4> hR(hA.size() * 2);
    std::vector<int> hD(ref.size());
    int compiled = 0;
    CHECK(hipMemcpy(hR.data(), dR, hR.size() * 16, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(hD.data(), dD, hD.size() * 4, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&compiled, dC, 4, hipMemcpyDeviceToHost));
    long bad_ref = 0, bad_w = 0;
    for (int it = 0; it < ntests; it++)
        for (int l = 0; l < 32; l++) {
            const int g = l / 4, t = l % 4;
            for (int f = 0; f < 2; f++) {
                const uint4 r = hR[(it * 32 + l) * 2 + f];
                const int n   = 8 * f + 2 * t;
                const int *R  = &ref[it * 256];
                bad_ref += (int)r.x != R[g * 16 + n];
                bad_ref += (int)r.y != R[g * 16 + n + 1];
                bad_ref += (int)r.z != R[(g + 8) * 16 + n];
                bad_ref += (int)r.w != R[(g + 8) * 16 + n + 1];
            }
        }
    if (compiled)
        for (size_t i = 0; i < ref.size(); i++)
            bad_w += hD[i] != ref[i];
    printf("int4 %s x s4, %d problems: reference wrong %ld/%zu, iu8 wmma %s\n",
           ACT_UNSIGNED ? "u4" : "s4", ntests, bad_ref, ref.size(),
           compiled ? (bad_w ? "WRONG" : "exact") : "not compiled for this target (gfx1250 only)");
    if (compiled && bad_w)
        printf("  wmma wrong in %ld/%zu outputs\n", bad_w, ref.size());
    return bad_ref + bad_w + (compiled ? 0 : getenv("REQUIRE_INT4") != nullptr);
}

int main() {
    int bad = 0;
    for (float scale : {1.0f, 0.01f, 30.0f}) {
        bad += run<0>(2000, scale);
        bad += run<1>(2000, scale);
        bad += run<2>(2000, scale == 30.0f ? 4.0f : scale);
    }
    bad += run_int4<false>(2000);
    bad += run_int4<true>(2000);
    printf(bad ? "FAIL\n" : "PASS\n");
    return bad != 0;
}
```

## t1250/build.sh

`t1250/build.sh <built tree> <gfx> [extra flags] [suffix]`: takes the converted `common_hip.h` and `kernels/utils_hip.cuh` from a tree that has been built once, applies the one substitution the build's hipify makes to `mma_rocm.cuh`, and compiles `test_gfx1250.hip` with the extension's flags.

```bash
#!/bin/bash
# usage: build.sh <tree> <arch> [extra flags]  -> test_<arch>: hipifies the tree's mma_rocm.cuh the way the build does
# (one substitution) and compiles test_gfx1250.hip with the extension's flags; the tree must have been built once
# (for common_hip.h and kernels/utils_hip.cuh)
set -e
T=$(cd "$(dirname "$0")" && pwd)
S=$1
mkdir -p $T/inc/src/kernels/zgemm
cp $S/src/common_hip.h $T/inc/src/common.h
cp $S/src/kernels/utils_hip.cuh $T/inc/src/kernels/utils.cuh
sed 's/__nv_bfloat162/__hip_bfloat162/g' $S/src/kernels/zgemm/mma_rocm.cuh > $T/inc/src/kernels/zgemm/mma_rocm.cuh
cd $T && $FLEET_VENV/lib/python3.11/site-packages/_rocm_sdk_devel/bin/hipcc -O3 -std=c++20 --offload-arch=$2 \
    -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DCUDA_HAS_FP16=1 -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 -DENABLE_BF16=1 \
    -DFMT_CONSTEVAL= -I inc/src -I $S/third_party/spdlog/include ${3:-} test_gfx1250.hip -o test_$2${4:-}
```

## cocmp.py

Device code object comparison of two builds, per object and target: instruction text (no addresses or encodings) identical or not, strings present in only one build, matrix-instruction counts, scratch instruction counts, largest private segment, kernels with scratch and largest VGPR count (from the code object notes).

```python
"""Compare the device code objects of two builds, per object file and GPU architecture.

    python -I cocmp.py <tree_a> <tree_b> <gfx> [<gfx> ...]

For every .o under build/temp*/ present in both trees: unbundle the code object of each architecture from the
.hip_fatbin section, disassemble it (llvm-objdump -d, no addresses or encodings) and compare the instruction text;
compare the read-only data strings separately and print the differing strings (line numbers of assert messages).
Also prints, per object and architecture, the counts of matrix instructions and of scratch instructions in each tree.
"""

import collections
import os
import re
import subprocess
import sys
import tempfile

LLVM = os.path.join(os.environ["FLEET_VENV"], "lib/python3.11/site-packages/_rocm_sdk_devel/lib/llvm/bin")
a_root, b_root, archs = sys.argv[1], sys.argv[2], sys.argv[3:]


def objects(root):
    out = {}
    for dirpath, _, files in os.walk(os.path.join(root, "build")):
        for f in files:
            if f.endswith(".o") and "/temp." in dirpath:
                out[os.path.relpath(os.path.join(dirpath, f), root).split("/", 2)[2]] = os.path.join(dirpath, f)
    return out


def code_object(obj, gfx, tmp):
    fb = os.path.join(tmp, "fb")
    co = os.path.join(tmp, f"co-{gfx}")
    if subprocess.run([f"{LLVM}/llvm-objcopy", f"--dump-section=.hip_fatbin={fb}", obj, os.devnull], capture_output=True).returncode:
        return None
    r = subprocess.run(
        [f"{LLVM}/clang-offload-bundler", "--type=o", "--unbundle", f"--input={fb}", f"--output={co}",
         f"--targets=hipv4-amdgcn-amd-amdhsa--{gfx}"],
        capture_output=True,
    )
    if r.returncode or not os.path.getsize(co):
        return None
    return co


def disasm(co, gfx):
    text = subprocess.run(
        [f"{LLVM}/llvm-objdump", "-d", f"--mcpu={gfx}", "--no-show-raw-insn", "--no-leading-addr", co],
        capture_output=True, text=True,
    ).stdout
    lines = [l.split("//")[0].rstrip() for l in text.splitlines()[2:]]
    return [l for l in lines if l]


def strings(co):
    return subprocess.run(["strings", "-n", "6", co], capture_output=True, text=True).stdout.splitlines()


MATRIX = re.compile(r"\b(v_(?:mfma|wmma|swmmac|smfmac)[a-z0-9_]*)")
SCRATCH = re.compile(r"\b(scratch_(?:load|store)[a-z0-9_]*|buffer_(?:load|store)[a-z0-9_]*)")


def counts(lines):
    m = collections.Counter(x for l in lines for x in MATRIX.findall(l))
    s = sum(1 for l in lines if SCRATCH.search(l))
    return m, s


def private_segment(co):
    """Largest private (scratch) segment size of any kernel in the code object, from its notes."""
    t = subprocess.run([f"{LLVM}/llvm-readelf", "--notes", co], capture_output=True, text=True).stdout
    sizes = [int(x) for x in re.findall(r"\.private_segment_fixed_size:\s+(\d+)", t)]
    vgprs = [int(x) for x in re.findall(r"\.vgpr_count:\s+(\d+)", t)]
    return (max(sizes) if sizes else 0, sum(1 for x in sizes if x), len(sizes), max(vgprs) if vgprs else 0)


oa, ob = objects(a_root), objects(b_root)
with tempfile.TemporaryDirectory() as tmp:
    os.makedirs(tmp + "/a")
    os.makedirs(tmp + "/b")
    for name in sorted(set(oa) & set(ob)):
        for gfx in archs:
            ca = code_object(oa[name], gfx, tmp + "/a")
            cb = code_object(ob[name], gfx, tmp + "/b")
            if ca is None and cb is None:
                continue
            if ca is None or cb is None:
                print(f"{name} {gfx}: code object only in {'b' if ca is None else 'a'}")
                continue
            da, db = disasm(ca, gfx), disasm(cb, gfx)
            same = da == db
            sa, sb = strings(ca), strings(cb)
            only_a = sorted(set(sa) - set(sb))
            only_b = sorted(set(sb) - set(sa))
            (ma, xa), (mb, xb) = counts(da), counts(db)
            pa, pb = private_segment(ca), private_segment(cb)
            status = "IDENTICAL" if same else f"DIFFERENT ({sum(1 for x, y in zip(da, db) if x != y)} lines, {len(da)} vs {len(db)})"
            print(f"{name} {gfx}: instructions {status}; strings only in a: {len(only_a)}, only in b: {len(only_b)}")
            for s in only_a[:6]:
                print(f"    a: {s[:160]}")
            for s in only_b[:6]:
                print(f"    b: {s[:160]}")
            print(f"    matrix a: {dict(ma)}")
            print(f"    matrix b: {dict(mb)}")
            print(
                f"    scratch instructions a/b: {xa}/{xb}; max private segment a/b: {pa[0]}/{pb[0]} bytes, kernels with "
                f"scratch a/b: {pa[1]}/{pa[2]} / {pb[1]}/{pb[2]}; max vgpr a/b: {pa[3]}/{pb[3]}"
            )
            sys.stdout.flush()
```

## pp.sh and ppall.sh

`pp.sh <tree> <file.hip> <gfx> <out>`: the device-only preprocessed source of one converted source for one target, with the build's flags, no line markers (`-P`) and the tree path mapped to `/T` (so `__FILE__` does not differ between trees). `ppall.sh <tree_a> <tree_b> <gfx...>` compares every `.hip` source of two built trees that way. This, not the disassembly, is the identity proof for gfx11/gfx12: the compiler's output for the large WMMA kernels varies between two compiles of the same source (see `notes.md`).

```bash
#!/bin/bash
# usage: pp.sh <tree> <file.hip relative to tree> <gfx> <out>: device-only preprocessed source without line markers,
# with the build's flags and the tree path mapped to /T (so __FILE__ does not differ between trees)
J=/var/lib/jenkins/fleet/jobs/mi355-20261125-084210-port5-nunchaku-loader-gfx1250-prompt
T=$1; V=$FLEET_VENV
$V/lib/python3.11/site-packages/_rocm_sdk_devel/bin/hipcc -E -P --cuda-device-only --offload-arch=$3 -fmacro-prefix-map=$T=/T \
  -I$T/src -I$V/lib/python3.11/site-packages/torch/include -I$V/lib/python3.11/site-packages/torch/include/torch/csrc/api/include \
  -I$V/lib/python3.11/site-packages/torch/include/THH -I$V/lib/python3.11/site-packages/_rocm_sdk_devel/include -I$V/include -I/usr/include/python3.11 \
  -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC -DCUDA_HAS_FP16=1 -D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1 \
  -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 -DENABLE_BF16=1 -DBUILD_NUNCHAKU=1 -std=c++20 -UNDEBUG -O3 -U__HIP_NO_HALF_OPERATORS__ -U__HIP_NO_HALF_CONVERSIONS__ \
  -DFMT_CONSTEVAL= -I$T/third_party/json/include -I$T/third_party/mio/include -I$T/third_party/spdlog/include \
  -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_C -fno-gpu-rdc $T/$2 -o $4
```

```bash
#!/bin/bash
# usage: ppall.sh <tree_a> <tree_b> <gfx...>: for every converted .hip source of the build, compare the device-only
# preprocessed code (pp.sh: no line markers, tree path mapped) of the two trees per target
J=/var/lib/jenkins/fleet/jobs/mi355-20261125-084210-port5-nunchaku-loader-gfx1250-prompt
A=$J/moat/agent_space/nunchaku
a=$1; b=$2; shift 2
out=$(mktemp -d)
for f in $(cd $a && find src -name "*.hip" | sort); do
    for g in "$@"; do
        $A/pp.sh $a $f $g $out/a.ii 2>/dev/null; ra=$?
        $A/pp.sh $b $f $g $out/b.ii 2>/dev/null; rb=$?
        if [ $ra -ne 0 ] || [ $rb -ne 0 ]; then echo "$f $g: preprocess failed ($ra/$rb)"; continue; fi
        if cmp -s $out/a.ii $out/b.ii; then echo "$f $g: identical ($(wc -l < $out/a.ii) lines)"; else echo "$f $g: DIFFERENT"; diff $out/a.ii $out/b.ii | head -6; fi
    done
done
rm -rf $out
```

## cc1.sh

`cc1.sh <tree> <file.hip> <gfx> <out.o>`: compiles one converted source of a built tree for one target with the build's flags (used to show that two compiles of the same source differ).

```bash
#!/bin/bash
# usage: cc1.sh <tree> <file.hip relative to tree> <gfx> <out.o>: compiles one source of a built tree for one target
# with the build's flags (host and device), logging to <out.o>.log
T=$1; V=$FLEET_VENV
$V/lib/python3.11/site-packages/_rocm_sdk_devel/bin/hipcc \
  -I$T/src -I$V/lib/python3.11/site-packages/torch/include -I$V/lib/python3.11/site-packages/torch/include/torch/csrc/api/include \
  -I$V/lib/python3.11/site-packages/torch/include/THH -I$V/lib/python3.11/site-packages/_rocm_sdk_devel/include -I$V/include -I/usr/include/python3.11 \
  -c $T/$2 -o $4 -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 -fPIC -DCUDA_HAS_FP16=1 -D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1 \
  -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 -DENABLE_BF16=1 -DBUILD_NUNCHAKU=1 -std=c++20 -UNDEBUG -O3 -U__HIP_NO_HALF_OPERATORS__ -U__HIP_NO_HALF_CONVERSIONS__ \
  -DFMT_CONSTEVAL= -I$T/third_party/json/include -I$T/third_party/mio/include -I$T/third_party/spdlog/include \
  -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_C --offload-arch=$3 -fno-gpu-rdc > $4.log 2>&1
echo "EXIT $?" >> $4.log
```

## cmpbuild.sh

`cmpbuild.sh <archs> <jobs> <tree> <name> [...]`: builds each tree's tracked files in turn at the same path (`trees/cmp`), renaming each result to `trees/cmp-<name>`, so that source paths compiled into the code objects are identical.

```bash
#!/bin/bash
# usage: cmpbuild.sh <archs> <jobs> <tree> <name> [<tree> <name> ...]
# Builds each tree's tracked files (plus untracked, not ignored ones) in turn at the same path, trees/cmp, so that the
# source paths compiled into the code objects are identical, and renames each result to trees/cmp-<name>.
set -u
J=/var/lib/jenkins/fleet/jobs/mi355-20261125-084210-port5-nunchaku-loader-gfx1250-prompt
archs=$1; jobs=$2; shift 2
dst=$J/trees/cmp
while [ $# -ge 2 ]; do
    tree=$1; name=$2; shift 2
    rm -rf "${dst:?}"
    mkdir -p "$dst"
    (cd "$tree" && git ls-files -z --cached --others --exclude-standard | tar --null -T - -cf -) | (cd "$dst" && tar xf -)
    $J/moat/agent_space/nunchaku/build.sh "$dst" "cmp-$name" "$archs" "$jobs"
    mv "$dst" "$J/trees/cmp-$name"
done
```

## build.sh, t.sh, incd.sh

`build.sh <tree> <name> <archs> <jobs> [ref]`: detached in-place build wrapped in `timeit.sh`. `t.sh <tree> <log> <timeout> <command...>`: runs a command from a tree root (pinned GPU, `ulimit -c 0`, timeout, `HF_HOME` and the test cache in the job directory) wrapped in `timeit.sh` (test phase); `incd.sh` changes into the tree because `timeit.sh` runs from the MOAT checkout.

```bash
#!/bin/bash
# usage: build.sh <tree> <name> <archs> <jobs> [ref]   detached in-place build, logged to build-<name>.log
set -u
J=/var/lib/jenkins/fleet/jobs/mi355-20261125-084210-port5-nunchaku-loader-gfx1250-prompt
tree=$1; name=$2
export PATH=$FLEET_VENV/bin:$PATH PYTORCH_ROCM_ARCH="$3" MAX_JOBS=$4 MOAT_PYTHON=/usr/bin/python3
if [ "${5:-}" = ref ]; then export NUNCHAKU_ROCM_MMA_REFERENCE=1; else unset NUNCHAKU_ROCM_MMA_REFERENCE; fi
log=$J/moat/agent_space/nunchaku/build-$name.log
start=$(date +%s)
$J/moat/utils/timeit.sh nunchaku compile -- bash -c 'cd "$0" && exec "$FLEET_VENV/bin/python" setup.py build_ext --inplace' "$tree" > $log 2>&1
echo "EXIT $? $(( $(date +%s) - start ))" >> $log
```

```bash
#!/bin/bash
# usage: t.sh <tree> <log> <timeout s> <command...>   from the tree root, in-place build, pinned GPU, timed (test phase)
J=/var/lib/jenkins/fleet/jobs/mi355-20261125-084210-port5-nunchaku-loader-gfx1250-prompt
tree=$1; log=$2; tmo=$3; shift 3
export PATH=$FLEET_VENV/bin:$PATH PYTHONPATH=$tree:${EXTRA_PYTHONPATH:-} HF_HOME=$J/hf NUNCHAKU_TEST_CACHE_ROOT=${CACHE:-$J/test_cache} MOAT_PYTHON=/usr/bin/python3
ulimit -c 0
start=$(date +%s)
$J/moat/utils/timeit.sh nunchaku test -- $J/moat/agent_space/nunchaku/incd.sh "$tree" timeout "$tmo" "$@" > "$log" 2>&1
echo "EXIT $? $(( $(date +%s) - start ))" >> "$log"
```

```bash
#!/bin/bash
# usage: incd.sh <dir> <command...>   runs the command in <dir> (timeit.sh runs from the MOAT checkout)
cd "$1" && shift && exec "$@"
```

## loader_matrix.py

Loads every public checkpoint type the tests use through the four v1 loaders, on faked GPUs (CUDA capabilities, ROCm architectures) onto the CPU, or with `real` on this host's GPU unfaked (then moved to the GPU), and records the precision each loader passed to the model and whether the load succeeded. Run once with the 75e2528 tree and once with the new tree; `python -I loader_matrix.py <tree> <out.json> [real]` with `HF_HOME` set.

```python
"""Load every public checkpoint type the tests use through the v1 loaders, on faked GPUs and on the real one, and
record the precision each loader picks and whether the load succeeds.

    python -I loader_matrix.py <tree> <out.json> [real]

<tree> is a built nunchaku checkout (its own nunchaku/ and _C). Loads go to the CPU (the loaders' default device), so
the faked GPU is only asked for its capability (CUDA) or architecture (ROCm); with "real" the GPU of this host is used
unfaked and the loaded model is moved onto it.
"""

import json
import sys
import traceback
import types

tree, out = sys.argv[1], sys.argv[2]
real = len(sys.argv) > 3 and sys.argv[3] == "real"
sys.path.insert(0, tree)

import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

import nunchaku  # noqa: E402,F401
from nunchaku.models.transformers.transformer_flux_v2 import NunchakuFluxTransformer2DModelV2  # noqa: E402
from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel  # noqa: E402
from nunchaku.models.transformers.transformer_zimage import NunchakuZImageTransformer2DModel  # noqa: E402
from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel  # noqa: E402

assert nunchaku.__file__.startswith(tree), nunchaku.__file__

CHECKPOINTS = [
    (NunchakuZImageTransformer2DModel, "nunchaku-tech/nunchaku-z-image-turbo", "svdq-int4_r32-z-image-turbo.safetensors"),
    (NunchakuZImageTransformer2DModel, "nunchaku-tech/nunchaku-z-image-turbo", "svdq-fp4_r32-z-image-turbo.safetensors"),
    (NunchakuZImageTransformer2DModel, "nunchaku-tech/nunchaku-z-image-turbo", "svdq-int4_r128-z-image-turbo.safetensors"),
    (NunchakuZImageTransformer2DModel, "nunchaku-tech/nunchaku-z-image-turbo", "svdq-fp4_r128-z-image-turbo.safetensors"),
    (NunchakuSDXLUNet2DConditionModel, "nunchaku-tech/nunchaku-sdxl", "svdq-int4_r32-sdxl.safetensors"),
    (NunchakuSDXLUNet2DConditionModel, "nunchaku-tech/nunchaku-sdxl-turbo", "svdq-int4_r32-sdxl-turbo.safetensors"),
    (NunchakuFluxTransformer2DModelV2, "nunchaku-tech/nunchaku-flux.1-schnell", "svdq-int4_r32-flux.1-schnell.safetensors"),
    (NunchakuFluxTransformer2DModelV2, "nunchaku-tech/nunchaku-flux.1-schnell", "svdq-fp4_r32-flux.1-schnell.safetensors"),
    (NunchakuQwenImageTransformer2DModel, "nunchaku-tech/nunchaku-qwen-image", "svdq-int4_r32-qwen-image.safetensors"),
    (NunchakuQwenImageTransformer2DModel, "nunchaku-tech/nunchaku-qwen-image", "svdq-fp4_r32-qwen-image.safetensors"),
]

DEVICES = [("cuda", c) for c in [(7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0), (12, 1)]] + [
    ("rocm", a) for a in ["gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1201", "gfx1250"]
]

orig_hip = torch.version.hip
orig_cap = torch.cuda.get_device_capability
orig_props = torch.cuda.get_device_properties


def fake(kind, device):
    if kind == "cuda":
        torch.version.hip = None
        torch.cuda.get_device_capability = lambda *a, **k: device

        def no_props(*a, **k):
            raise AssertionError("ROCm architecture asked on a CUDA device")

        torch.cuda.get_device_properties = no_props
    else:
        torch.version.hip = orig_hip or "7.0"
        torch.cuda.get_device_capability = orig_cap
        torch.cuda.get_device_properties = lambda *a, **k: types.SimpleNamespace(gcnArchName=f"{device}:sramecc+:xnack-")


def restore():
    torch.version.hip = orig_hip
    torch.cuda.get_device_capability = orig_cap
    torch.cuda.get_device_properties = orig_props


picked = []
for cls in {c for c, _, _ in CHECKPOINTS}:
    original = cls._patch_model

    def spy(self, *args, _original=original, **kwargs):
        picked.append(kwargs.get("precision"))
        return _original(self, *args, **kwargs)

    cls._patch_model = spy

results = []
devices = [("real", "here")] if real else DEVICES
for cls, repo, name in CHECKPOINTS:
    path = hf_hub_download(repo, name)
    for kind, device in devices:
        picked.clear()
        if kind != "real":
            fake(kind, device)
        try:
            # as the tests do: no device argument (the Z-Image loader passes its keyword arguments on to the layers,
            # so device= fails there before and after), then move the model to the GPU
            model = cls.from_pretrained(path, torch_dtype=torch.bfloat16)
            if kind == "real":
                model = model.to("cuda")
            status = "ok"
            del model
        except Exception as e:  # noqa: BLE001
            status = f"{type(e).__name__}: {str(e).splitlines()[0][:150] if str(e) else ''}"
            if "-v" in sys.argv:
                traceback.print_exc()
        finally:
            restore()
            if kind == "real":
                torch.cuda.empty_cache()
        row = dict(loader=cls.__name__, checkpoint=name, device=f"{kind}:{device}", precision=picked[:], status=status)
        print(json.dumps(row), flush=True)
        results.append(row)

json.dump(results, open(out, "w"), indent=1)
```

## test_zimage_fp4_public.py

Runs upstream's `test_zimage_turbo` (unchanged) on the NVFP4 checkpoints: only the test module's own `precision` variable (its checkpoint name and threshold choice) is set to `fp4`; nunchaku is not patched, the loader picks the precision from the checkpoint, and the precision it passed to the model is recorded and checked. `python -m pytest -v -s <path>/test_zimage_fp4_public.py` from the fork root.

```python
"""Runs upstream's Z-Image Turbo test (tests/v1/z_image/test_z_image_turbo.py, unchanged) on the NVFP4 checkpoints.

The test chooses its checkpoint file and LPIPS threshold from its module-level ``precision`` (get_precision(), which
is "int4" on AMD GPUs); only that variable of the test module is set to "fp4" here. nunchaku is not patched: the
checkpoint is loaded through NunchakuZImageTransformer2DModel.from_pretrained, which picks the precision itself. The
precision the loader passed to the model is recorded and checked.

Run from the fork root:  python -m pytest -v -s <this file>
"""

import pytest

import tests.v1.z_image.test_z_image_turbo as upstream
from nunchaku import NunchakuZImageTransformer2DModel

upstream.precision = "fp4"

chosen = []
_patch_model = NunchakuZImageTransformer2DModel._patch_model


def _record(self, *args, **kwargs):
    chosen.append(kwargs["precision"])
    return _patch_model(self, *args, **kwargs)


@pytest.fixture(autouse=True)
def loader_precision(monkeypatch):
    chosen.clear()
    monkeypatch.setattr(NunchakuZImageTransformer2DModel, "_patch_model", _record)
    yield
    print(f"loader precision: {chosen}")
    assert chosen in ([], ["nvfp4"])  # [] for the rank without an NVFP4 checkpoint, where the test returns early


test_zimage_turbo = upstream.test_zimage_turbo
```
