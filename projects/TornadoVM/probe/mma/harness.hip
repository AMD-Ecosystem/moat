// Independent check of the HIP KernelContext MMA helpers that TC emits (tc_mma_m16n8k16, _bf16,
// tc_mma_m16n8k32_s8, _e4m3, _e5m2). The fragments are built here from the documented PTX m16n8
// layouts, not with TC's load helpers, and every result is compared with a CPU reference in double
// (int32 with wraparound for s8). run.sh generates tc_helpers.hip from helpers.tcir with the SDK's TC.
//
// Layouts (lane = 4g + t of a 32-lane warp; a k-word is 2 halves or 4 bytes, low element first):
//   A int4: x = row g kword t, y = row g+8 kword t, z = row g kword 4+t, w = row g+8 kword 4+t
//   B int2: x = column g kword t, y = column g kword 4+t
//   C/D   : element e is row g + 8(e/2), column 2t + e%2 of the 16 x 8 tile
//
// Usage: harness [--mutate]. --mutate checks every result against the next column's reference; every
// case must then fail, which shows the comparison can see a wrong result. Exit status 0 means every case passed (or, with --mutate, failed).
//
// Tolerances, relative to the sum of the products' magnitudes: s8 is always exact. The exact f16/bf16 cases
// (small integers) are bit-exact except on gfx11, whose WMMA is one or two float ulps off even then (1e-6). The
// FP8 every-byte cases are bit-exact on every arch, except the e5m2 negative subnormals 0x81-0x83 on gfx11
// (1e-6, measured one or two float ulps off). The random float cases allow 1e-5, and 1e-4 for FP8 on gfx1250,
// whose FP8 WMMA keeps about 14 to 16 bits while summing. A wrong layout or FP8 decoding is off by far more
// than any of these. The every-byte cases also list each byte whose result is not bit-exact.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// The helpers route on the GPU architecture macros and stop with #error without one, so the host pass
// sees only their declarations.
#if defined(__HIP_DEVICE_COMPILE__)
#include "tc_helpers.hip"
#else
__device__ float4 tc_mma_m16n8k16(int4 a, int2 b, float4 c);
__device__ float4 tc_mma_m16n8k16_bf16(int4 a, int2 b, float4 c);
__device__ int4 tc_mma_m16n8k32_s8(int4 a, int2 b, int4 c);
__device__ float4 tc_mma_m16n8k32_e4m3(int4 a, int2 b, float4 c);
__device__ float4 tc_mma_m16n8k32_e5m2(int4 a, int2 b, float4 c);
#endif

enum Kind { F16, BF16, S8, E4M3, E5M2 };
static const char *kKindName[] = {"f16", "bf16", "s8", "e4m3", "e5m2"};

#define CHECK_HIP(x)                                                                                         \
    do {                                                                                                     \
        hipError_t e_ = (x);                                                                                 \
        if (e_ != hipSuccess) {                                                                              \
            fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_));              \
            exit(2);                                                                                         \
        }                                                                                                    \
    } while (0)

// One 16 x 8 tile per warp. A: steps x (16 rows x 8 k-words); B: steps x (8 columns x 8 k-words);
// C and D: 16 x 8 row-major 32-bit values (float bits, or int32 for s8).
template <int KIND>
__global__ void tiles(const int *A, const int *B, const int *C, int *D, int ntiles, int steps, int activeWarps) {
    int warp = (int) (threadIdx.x / 32u), lane = (int) (threadIdx.x % 32u);
    int tile = (int) blockIdx.x * (int) (blockDim.x / 32u) + warp;
    if (warp >= activeWarps || tile >= ntiles) {
        return;
    }
    int g = lane >> 2, t = lane & 3;
    const int *c = C + tile * 128;
    int ce[4];
    for (int e = 0; e < 4; e++) {
        ce[e] = c[(g + 8 * (e >> 1)) * 8 + 2 * t + (e & 1)];
    }
    int4 iacc = make_int4(ce[0], ce[1], ce[2], ce[3]);
    float4 facc = make_float4(__int_as_float(ce[0]), __int_as_float(ce[1]), __int_as_float(ce[2]), __int_as_float(ce[3]));
    for (int s = 0; s < steps; s++) {
        const int *a = A + (tile * steps + s) * 128;
        const int *b = B + (tile * steps + s) * 64;
        int4 fa = make_int4(a[g * 8 + t], a[(g + 8) * 8 + t], a[g * 8 + 4 + t], a[(g + 8) * 8 + 4 + t]);
        int2 fb = make_int2(b[g * 8 + t], b[g * 8 + 4 + t]);
        if constexpr (KIND == F16) {
            facc = tc_mma_m16n8k16(fa, fb, facc);
        } else if constexpr (KIND == BF16) {
            facc = tc_mma_m16n8k16_bf16(fa, fb, facc);
        } else if constexpr (KIND == S8) {
            iacc = tc_mma_m16n8k32_s8(fa, fb, iacc);
        } else if constexpr (KIND == E4M3) {
            facc = tc_mma_m16n8k32_e4m3(fa, fb, facc);
        } else {
            facc = tc_mma_m16n8k32_e5m2(fa, fb, facc);
        }
    }
    int out[4];
    if constexpr (KIND == S8) {
        out[0] = iacc.x, out[1] = iacc.y, out[2] = iacc.z, out[3] = iacc.w;
    } else {
        out[0] = __float_as_int(facc.x), out[1] = __float_as_int(facc.y), out[2] = __float_as_int(facc.z), out[3] = __float_as_int(facc.w);
    }
    int *d = D + tile * 128;
    for (int e = 0; e < 4; e++) {
        d[(g + 8 * (e >> 1)) * 8 + 2 * t + (e & 1)] = out[e];
    }
}

// ---- host-side element decoding ------------------------------------------------------------------

static double half_value(uint16_t h) {
    int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    double v = e == 0 ? std::ldexp((double) m, -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp(1024.0 + m, e - 25);
    return s ? -v : v;
}

static double bf16_value(uint16_t h) {
    uint32_t u = (uint32_t) h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static double e4m3_value(uint8_t b) {
    int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    double v = (e == 15 && m == 7) ? NAN : e == 0 ? std::ldexp((double) m, -9) : std::ldexp(8.0 + m, e - 10);
    return s ? -v : v;
}

static double e5m2_value(uint8_t b) {
    return half_value((uint16_t) (b << 8));
}

static int perWord(int kind) {
    return kind == F16 || kind == BF16 ? 2 : 4;
}

// Element k of a row of A (or a column of B) stored as 8 k-words.
static double element(int kind, const int *words, int k) {
    uint32_t w = (uint32_t) words[k / perWord(kind)];
    switch (kind) {
    case F16:
        return half_value((uint16_t) (w >> (16 * (k & 1))));
    case BF16:
        return bf16_value((uint16_t) (w >> (16 * (k & 1))));
    case S8:
        return (double) (int8_t) (w >> (8 * (k & 3)));
    case E4M3:
        return e4m3_value((uint8_t) (w >> (8 * (k & 3))));
    default:
        return e5m2_value((uint8_t) (w >> (8 * (k & 3))));
    }
}

// ---- cases -----------------------------------------------------------------------------------------

struct Case {
    std::string name;
    int kind, steps, warpsPerBlock, activeWarps, ntiles;
    bool exact, everyByte = false;
    std::vector<int> A, B, C;
};

static float as_float(int x) {
    float f;
    memcpy(&f, &x, 4);
    return f;
}

static int float_bits(float f) {
    int x;
    memcpy(&x, &f, 4);
    return x;
}

static std::mt19937 rng(12345);

static uint16_t random_half(double limit) {
    for (;;) {
        uint16_t h = (uint16_t) rng();
        double v = half_value(h);
        if (std::isfinite(v) && std::fabs(v) <= limit) {
            return h;
        }
    }
}

// The half bits of a small integer v (|v| <= 4).
static uint16_t half_of_int(int v) {
    static const uint16_t bits[] = {0xc400, 0xc200, 0xc000, 0xbc00, 0x0000, 0x3c00, 0x4000, 0x4200, 0x4400};
    return bits[v + 4];
}

static uint8_t random_fp8(int kind, double limit) {
    for (;;) {
        uint8_t b = (uint8_t) rng();
        double v = kind == E4M3 ? e4m3_value(b) : e5m2_value(b);
        if (std::isfinite(v) && std::fabs(v) <= limit) {
            return b;
        }
    }
}

// Fills one k-word of the operand for `kind` with random elements; `ints` keeps them small integers.
static int random_word(int kind, bool ints) {
    uint32_t w = 0;
    if (kind == F16 || kind == BF16) {
        for (int i = 0; i < 2; i++) {
            uint16_t h;
            if (kind == F16) {
                h = ints ? half_of_int((int) (rng() % 9) - 4) : random_half(4.0);
            } else {
                float f = ints ? (float) ((int) (rng() % 9) - 4) : std::ldexp((float) ((int) (rng() % 255) - 127), -5);
                uint32_t u;
                memcpy(&u, &f, 4);
                h = (uint16_t) (u >> 16);
            }
            w |= (uint32_t) h << (16 * i);
        }
    } else {
        for (int i = 0; i < 4; i++) {
            uint8_t b = kind == S8 ? (uint8_t) rng() : random_fp8(kind, ints ? 8.0 : 448.0);
            w |= (uint32_t) b << (8 * i);
        }
    }
    return (int) w;
}

static Case random_case(int kind, int steps, int wpb, int active, bool ints, bool wrap) {
    Case c;
    c.kind = kind, c.steps = steps, c.warpsPerBlock = wpb, c.activeWarps = active, c.ntiles = 8 * wpb, c.exact = (ints && (kind == F16 || kind == BF16)) || kind == S8;
    c.name = std::string(ints ? "ints" : wrap ? "wrap" : "random") + " steps=" + std::to_string(steps) + " warps=" + std::to_string(active) + "/" + std::to_string(wpb);
    c.A.resize((size_t) c.ntiles * steps * 128);
    c.B.resize((size_t) c.ntiles * steps * 64);
    c.C.resize((size_t) c.ntiles * 128);
    for (auto &x : c.A) {
        x = random_word(kind, ints);
    }
    for (auto &x : c.B) {
        x = random_word(kind, ints);
    }
    for (auto &x : c.C) {
        if (kind == S8) {
            x = wrap ? (int) (rng() % 2 ? 0x7fff0000u + rng() % 65536u : 0x80000000u + rng() % 65536u) : (int) (rng() % 2001) - 1000;
        } else {
            x = float_bits(ints ? (float) ((int) (rng() % 65) - 32) : (float) ((int) (rng() % 20001) - 10000) * 1e-3f);
        }
    }
    return c;
}

// Every FP8 byte value v, in tile v: A row v % 16 holds v at k = (v * 7) % 32; B column v % 8 holds 1.0 at
// that k. The result is v's value (NaN and infinities included) at (v % 16, v % 8), and v * 0 elsewhere in
// the row.
static Case fp8_bytes(int kind) {
    Case c;
    c.kind = kind, c.steps = 1, c.warpsPerBlock = 4, c.activeWarps = 4, c.ntiles = 256, c.exact = true;
    c.name = "every byte", c.everyByte = true;
    c.A.assign((size_t) c.ntiles * 128, 0);
    c.B.assign((size_t) c.ntiles * 64, 0);
    c.C.assign((size_t) c.ntiles * 128, 0);
    uint8_t one = kind == E4M3 ? 0x38 : 0x3c;
    for (int v = 0; v < 256; v++) {
        int tile = v, r = v % 16, k = (v * 7) % 32, n = v % 8;
        int *a = &c.A[(size_t) tile * 128 + r * 8];
        a[k / 4] |= v << (8 * (k % 4));
        int *b = &c.B[(size_t) tile * 64 + n * 8];
        b[k / 4] |= one << (8 * (k % 4));
    }
    return c;
}

template <int KIND>
static void launch(const Case &c, std::vector<int> &D) {
    int *dA, *dB, *dC, *dD;
    CHECK_HIP(hipMalloc(&dA, c.A.size() * 4));
    CHECK_HIP(hipMalloc(&dB, c.B.size() * 4));
    CHECK_HIP(hipMalloc(&dC, c.C.size() * 4));
    CHECK_HIP(hipMalloc(&dD, c.C.size() * 4));
    CHECK_HIP(hipMemcpy(dA, c.A.data(), c.A.size() * 4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB, c.B.data(), c.B.size() * 4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dC, c.C.data(), c.C.size() * 4, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(dD, 0xff, c.C.size() * 4));
    int blocks = (c.ntiles + c.warpsPerBlock - 1) / c.warpsPerBlock;
    hipLaunchKernelGGL(tiles<KIND>, dim3(blocks), dim3(32 * c.warpsPerBlock), 0, 0, dA, dB, dC, dD, c.ntiles, c.steps, c.activeWarps);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    D.resize(c.C.size());
    CHECK_HIP(hipMemcpy(D.data(), dD, D.size() * 4, hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(dA));
    CHECK_HIP(hipFree(dB));
    CHECK_HIP(hipFree(dC));
    CHECK_HIP(hipFree(dD));
}

// Returns the number of mismatching elements; tiles of inactive warps must be left untouched.
static double gExactBound = 0.0, gFp8Bound = 1e-5, gE5m2SubnormalBound = 0.0;

// The bound for one element: every-byte tile v holds byte v's own result at (v % 16, v % 8).
static double bound_of(const Case &c, int tile, int r, int n) {
    if (c.everyByte) {
        bool sub = c.kind == E5M2 && r == tile % 16 && n == tile % 8 && tile >= 0x81 && tile <= 0x83;
        return sub ? gE5m2SubnormalBound : 0.0;
    }
    return c.exact ? gExactBound : (c.kind == E4M3 || c.kind == E5M2) ? gFp8Bound : 1e-5;
}

static long check(const Case &c, const std::vector<int> &D, bool mutate, double &worst) {
    long bad = 0;
    int K = 8 * perWord(c.kind);
    for (int tile = 0; tile < c.ntiles; tile++) {
        bool active = tile % c.warpsPerBlock < c.activeWarps;
        for (int r = 0; r < 16; r++) {
            for (int n = 0; n < 8; n++) {
                int got = D[(size_t) tile * 128 + r * 8 + n];
                if (!active) {
                    bad += got != -1;
                    continue;
                }
                int cv = c.C[(size_t) tile * 128 + r * 8 + n];
                int nn = mutate ? (n + 1) % 8 : n;
                if (c.kind == S8) {
                    uint32_t ref = (uint32_t) cv;
                    for (int s = 0; s < c.steps; s++) {
                        const int *a = &c.A[((size_t) tile * c.steps + s) * 128 + r * 8];
                        const int *b = &c.B[((size_t) tile * c.steps + s) * 64 + nn * 8];
                        for (int k = 0; k < K; k++) {
                            ref += (uint32_t) (int32_t) (element(S8, a, k) * element(S8, b, k));
                        }
                    }
                    bad += (uint32_t) got != ref;
                    continue;
                }
                double ref = as_float(cv), mag = std::fabs(ref);
                for (int s = 0; s < c.steps; s++) {
                    const int *a = &c.A[((size_t) tile * c.steps + s) * 128 + r * 8];
                    const int *b = &c.B[((size_t) tile * c.steps + s) * 64 + nn * 8];
                    for (int k = 0; k < K; k++) {
                        double p = element(c.kind, a, k) * element(c.kind, b, k);
                        ref += p;
                        mag += std::fabs(p);
                    }
                }
                double g = as_float(got), bound = bound_of(c, tile, r, n);
                if (c.everyByte && !mutate && !(std::isnan(ref) && std::isnan(g)) && g != ref) {
                    printf("      byte 0x%02x (row %d col %d): got %.9g, want %.9g, rel %.2g, bound %g\n", tile, r, n, g, ref,
                           mag > 0 ? std::fabs(g - ref) / mag : std::fabs(g - ref), bound);
                }
                if (std::isnan(ref) || std::isnan(g)) {
                    bad += std::isnan(ref) != std::isnan(g);
                } else if (std::isinf(ref) || std::isinf(g)) {
                    bad += ref != g;
                } else {
                    double err = std::fabs(g - ref), rel = mag > 0 ? err / mag : err;
                    worst = std::max(worst, rel);
                    bad += bound == 0.0 ? err != 0.0 : rel > bound;
                }
            }
        }
    }
    return bad;
}

static long run(const Case &c, bool mutate) {
    std::vector<int> D;
    switch (c.kind) {
    case F16:
        launch<F16>(c, D);
        break;
    case BF16:
        launch<BF16>(c, D);
        break;
    case S8:
        launch<S8>(c, D);
        break;
    case E4M3:
        launch<E4M3>(c, D);
        break;
    default:
        launch<E5M2>(c, D);
        break;
    }
    double worst = 0.0;
    long bad = check(c, D, mutate, worst);
    printf("%-5s %-28s %s (%ld wrong, worst rel %.2g)\n", kKindName[c.kind], c.name.c_str(), bad ? "FAIL" : "ok", bad, worst);
    return bad;
}

int main(int argc, char **argv) {
    bool mutate = argc > 1 && strcmp(argv[1], "--mutate") == 0;
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    printf("device: %s (%s), wavefront %d\n", prop.name, prop.gcnArchName, prop.warpSize);
    if (strncmp(prop.gcnArchName, "gfx11", 5) == 0) {
        gExactBound = 1e-6;
        gE5m2SubnormalBound = 1e-6;
    } else if (strncmp(prop.gcnArchName, "gfx1250", 7) == 0) {
        gFp8Bound = 1e-4;
    }
    printf("bounds: f16/bf16 exact cases %g, random fp8 %g, other random 1e-05\n", gExactBound, gFp8Bound);
    printf("bounds: every byte e4m3 0, e5m2 0 except bytes 0x81-0x83 %g\n", gE5m2SubnormalBound);
    struct Config {
        int steps, wpb, active;
    } configs[] = {{1, 1, 1}, {3, 2, 2}, {2, 4, 4}, {1, 2, 1}, {5, 4, 3}};
    long failedCases = 0, cases = 0;
    for (int kind = F16; kind <= E5M2; kind++) {
        for (auto cfg : configs) {
            for (int ints = 0; ints < 2; ints++) {
                cases++;
                failedCases += run(random_case(kind, cfg.steps, cfg.wpb, cfg.active, ints, false), mutate) != 0;
            }
        }
        if (kind == S8) {
            cases++;
            failedCases += run(random_case(S8, 2, 2, 2, false, true), mutate) != 0;
        }
        if (kind == E4M3 || kind == E5M2) {
            cases++;
            failedCases += run(fp8_bytes(kind), mutate) != 0;
        }
    }
    printf("%ld of %ld cases failed%s\n", failedCases, cases, mutate ? " (reference mutated: every case must fail)" : "");
    if (mutate) {
        return failedCases == cases ? 0 : 1;
    }
    return failedCases ? 1 : 0;
}
