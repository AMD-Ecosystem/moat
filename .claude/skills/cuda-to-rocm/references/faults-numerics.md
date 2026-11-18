# Faults: floating point and integer arithmetic

## Contraction and rounding

**clang for HIP defaults to `-ffp-contract=fast`** and forms FMAs ACROSS statements. nvcc
contracts by default too (`--fmad=true`, and NVRTC likewise), so a single `a * b + c` is an
FMA on both; what differs is where else contraction happens. The extra contraction drifts
results ~1 ULP from the CUDA build and CPU gold. That nvcc contracts only within one
expression is inferred from the result below (`-ffp-contract=on` reproduced nvcc's numbers),
not taken from nvcc's documentation. `-ffp-contract=off` is the `--fmad=false` equivalent. MATCH UPSTREAM's setting rather than
pinning one unconditionally: CV-CUDA needed `-ffp-contract=on` to match nvcc, while
LC-framework's nvcc recipe uses `-fmad=false -mno-fma -ffp-contract=off`, so pinning `on`
there would diverge. Integer-only components are unaffected. (CV-CUDA OpWarpPerspective;
LC-framework.)

**One contraction setting cannot satisfy both kinds of bit-exact test.** A project can have
tests that compare a kernel against a CPU gold (which want the contraction nvcc and the host
use) AND tests that compare two GPU code paths against each other byte-for-byte (which want
the same contraction in both instantiations). clang contracts a `float` instantiation and a
`float4` instantiation of the same template differently, so a per-plane path and a per-pixel
path that are bit-identical on NVIDIA differ by 1 ULP on ROCm under either setting. Measure
before choosing, and clear unrelated defects out of the sample first: in one suite a third of
the apparent contraction failures were an allocator ordering race and a test-fixture
use-after-free, and a decision taken on the raw count would have rested on the wrong
evidence. Treat the setting as a reviewed decision with numbers attached, and re-measure
after every functional fix that lands in the same suite. (CV-CUDA: `-ffp-contract=off`
removed most planar-versus-interleaved parity failures but regressed the cubic-versus-gold
case, so `on` was kept and the parity residual documented.)

**`__fsqrt_rn` is not always correctly rounded on gfx90a** (1 ULP high) where CUDA's
`sqrt.rn.f32` is IEEE correctly-rounded, so bit-exact tests pass on NVIDIA and fail on
gfx90a. Route the f32 device sqrt through the correctly-rounded f64 `__dsqrt_rn` and cast
back; CDNA has fast f64 sqrt so the cost is negligible. (CV-CUDA: OpNormalize, L2
PairwiseMatcher.)

**Device `cuda::min`/`cuda::max` NaN-selection can differ from host `std::min`/`std::max`.**
Host `std::min(a,b)` is `b<a?b:a` and `std::max(a,b)` is `a<b?b:a`; a device version written
`a<b?a:b` picks the OPPOSITE operand on a NaN compare, so tests filling buffers with random
bytes and comparing host gold against device output differ bit-for-bit. Spell the device
ternaries to match the host forms exactly. (CV-CUDA: OpMorphology CLOSE on RGBAf32.)

**An exact float-equality branch fed by approximate division can HANG, not just drift.**
HIP's `__fdividef` differs ~1 ULP from CUDA, so a downstream exact-equality test on the
quotient -- selecting a loop start or end index -- can flip on AMD. When that index feeds an
UNSIGNED loop bound via subtraction, the out-of-range value underflows to a huge count and
the loop never terminates: a data-dependent hang, not a wrong-by-1-ULP number. Do not gate
control flow on exact equality of approximate fp; use a tolerance, or cap the derived bound
at its provable geometric maximum. This is the class where numeric drift becomes a
CONTROL-FLOW failure. (TIGRE Siddon_projection: `Np` derived from `__fdividef` comparisons,
a 0.2s kernel ran >10 min at some angles; capped at `Nx+Ny+Nz+3`, a no-op for valid rays and
identical on CUDA.)

## Host math under the HIP compiler

**Host `min`/`max` on floating-point values truncate to int under hipcc.** CUDA declares
host `min`/`max` for int, unsigned, long, long long, float and double. HIP's
`__clang_hip_math.h` declares only `min`/`max(int, int)` on the host. So host code in a
`.cu` file such as `max(0.25, min(100.0, fwhm))` or `max(atan(a), atan(b))` compiles
cleanly, converts both arguments to int and returns an int. Device code is unaffected,
because the device overloads cover float and double. Expect parameters silently rounded
down, a value below 1 clamped to 0, and geometry decisions taken on a zero angle. The only
build symptom is `-Wliteral-conversion` on fractional literals ("changes value from 0.25
to 0"), and only when a literal is involved. Detection: add
`-D__HIP_NO_HOST_MIN_MAX_IN_GLOBAL_NAMESPACE__ -include poison.h` to the real hipcc
command with `--cuda-host-only -fsyntax-only`. `poison.h` redefines the default pair with
`__attribute__((diagnose_if(true, "HOSTMINMAX", "warning")))`, so every host call that
lands on `(int, int)` is reported with its location. Define CUDA's full overload set the
same way, each overload naming its own signature, to see which calls really take floating
point. Use `diagnose_if`, not `deprecated`: a `deprecated` version missed LEAP's two
`ramp_filter.cu` sites (cause not traced). Fix: change the call sites to what CUDA's
overload calls. `crt/math_functions.hpp` implements `min`/`max(double, double)` and the
mixed float/double forms with `fmin`/`fmax`, and `(float, float)` with `fminf`/`fmaxf`.
Cast a float argument to double where CUDA took a mixed overload, e.g.
`fmax(0.25, fmin(100.0, double(fwhm)))`. That keeps CUDA's results, NaN handling included,
on every ROCm and on CUDA, without an `#ifdef`. Leave int calls alone. Do NOT rely on
`-D__HIP_DEFINE_EXTENDED_HOST_MIN_MAX__=1`. That macro adds the floating-point host
overloads only from ROCm 7.2.0 (LLVM 0248d277ca), defaults to 0, and does not exist in
7.1.x and earlier, where the define silently does nothing and the truncation stays. Rerun
the detection pass with no define: no host call may still land on `(int, int)` with
non-int arguments. Then confirm that the device code objects did not change (`.text`/`.rodata`
of the unbundled `.hip_fatbin`; the `__hip_cuid_*` names change with path and flags).
(LEAP: bilateral FWHMs truncated, intensity FWHM < 1 gave the identity; the modular-beam
Joseph backprojector chose its parallel kernel for cone data, failing adjointness and FBP;
the FBPlowpass window zeroed all but DC.) `std::min`/`std::max` are NOT safe either once a
hipify pass has run: hipify_torch (and torch's hipify, same table) rewrites them to
`::min`/`::max`, which land on the same `(int, int)` host pair, so a correct
`std::min(float, float)` in CUDA source truncates in the hipified copy. Run the detection
on the hipified sources, not the originals. (torch's hipify stopped rewriting `std::` math in
pytorch#200419, 2026-10; a torch older than that, and hipify_torch, still do.) When a build compiles host `.cpp` files as
HIP too, those files are in scope, but unqualified calls there usually bind to `std::min`
through `using namespace std` (a template exact match beats the int pair); to tell a real
truncation from that, report only calls that land on `(int, int)` with ROCm's default
pair AND on a non-int overload with CUDA's full set. (LEAP version two: 27 sites, 10 of
them new, including `1.0 / max(fabs(v_min), fabs(v_max))` dividing by zero in the cone
geometry setup.)

**A HIP workaround in shared host code must not change CUDA's numbers.** HIP declares
`rsqrtf` for the device only and host `min`/`max` only for `(int, int)` (the entry above),
and HIP's `__half`
converts to every floating type through a template, so a port that rewrites a host call for
HIP (`rsqrtf(x)` to `1.0f / std::sqrt(x)`, `std::pow(a, b)` to a cast through `double`,
`__half` to an integer through `float`) silently changes the CUDA build too: `1.0f / sqrtf`
rounds twice and moved 9 of 51 coefficients by 1 ULP, `pow` through double differed in the
last bit, and the float route made NaN/Inf half-to-int undefined where CUDA's
`__half2int_rz` saturates. Gate the rewrite on `__HIPCC__` and keep upstream's
expression in the `#else`. Do not narrow the gate to the toolchain where the error first
showed: `std::pow(__half, __half)` was ambiguous in the HIP host pass on Linux too (glibc
`double` vs libstdc++ `float`) once a `__half` kernel instantiated it. A `_MSC_VER` gate
was chosen because an older Linux build had compiled upstream's call, but that build had no
gate and predated the kernels that instantiate it, and the first Linux build with the gate
failed; key the gate to the cause (HIP's `__half`), not to the first OS that
failed; where HIP needs a substitute, write the formula CUDA's host math
uses (`(float)(1.0 / sqrt((double)x))` for host `rsqrtf`), and give conversions CUDA's
saturation and NaN rules. A compile-only CUDA build with a per-TU preprocessed diff against
upstream is what found these; HIP runs cannot. (LichtFeld-Studio)

## Conversions and multi-word integers

**Converting a NEGATIVE float to an UNSIGNED integer diverges silently between the back
ends** -- it is undefined behaviour, and the two hardware conversions pick different
answers. Measured on gfx90a, `(uint64_t)(-2.28)` is `0x00000000fffffffe`: the AMD lowering
of a 64-bit `fptoui` keeps only the low 32 bits of the two's-complement value. NVIDIA's
`cvt.rzi.u64.f64` saturates a negative source to zero. So a wrapped-negative idiom that
works on CUDA -- `(T)z + (flag & modulus)` to reduce a signed sample into `[0, q)`, or any
`(unsigned)x` relying on modular wraparound -- produces values near 2^32 on AMD. Grep for
`static_cast<unsigned...>(` / `(uint64_t)` applied to anything that can be negative, and
route the truncation through the signed type first: `(T)(std::make_signed_t<T>)x`. Note the
32-BIT case does NOT diverge: `(uint32_t)(-2.28)` clamps to 0 on both, so only the 64-bit
conversion needs the fix and a blanket sweep of the 32-bit sites would gratuitously change
CUDA behaviour. Symptom to recognise: an output whose magnitude clusters near 2^32 with
roughly half the values affected, i.e. exactly the negative half of a signed distribution.
The fix is NOT behaviour-neutral upstream (it replaces a saturated zero with the correct
negative representative), so say so in the commit message rather than presenting it as a
pure port change. (HEonGPU: RNGonGPU's Box-Muller Gaussian sampler; every ciphertext
decrypted to noise while plaintext encoding, which samples no error, passed -- which is the
tell that the fault is in a SAMPLER and not in the transform everything shares.)

### PTX carry-chain arithmetic (multi-word integers)

Big-integer and modular-reduction code reaches for PTX because CUDA exposes the carry flag
and C++ does not: `add.cc.u64` / `addc.cc.u64` / `subc.u64` chains, and `mul.lo.u64` +
`mul.hi.u64` pairs. None of it survives hipify, so it is hand-translated, and the
translation is easy to get subtly wrong in a way that compiles, runs, and silently corrupts
results. Guard the original with `#ifdef __HIP_PLATFORM_AMD__` and leave the NVIDIA path
byte-identical rather than replacing it for both.

The two replacements, with the predicates spelled out, since these are what get fumbled:

    // mul.lo/mul.hi -> 128-bit product
    lo = a * b;
    hi = __umul64hi(a, b);

    // add.cc/addc chain -> carry out of a + b + carry_in
    sum   = a + b + carry;
    carry = (sum < a || (carry && sum == a)) ? 1 : 0;

    // sub.cc/subc chain -> borrow out of a - b - borrow_in
    diff   = a - b - borrow;
    borrow = (a < b || (a == b && borrow)) ? 1 : 0;

The add predicate needs both clauses. With `carry_in = 1` and `b == 2^64 - 1` the sum equals
`a` exactly and a carry is nevertheless produced; testing only `sum < a` drops it. The
symmetric case is why the borrow predicate tests `a == b && borrow` and reads the ORIGINAL
`a` and `b`, not the computed difference.

**Establish the limb order from the asm operand numbering before writing anything.** Outputs
are numbered before inputs, so in

    : "=l"(v.y), "=l"(v.x)        // %0 = v.y, %1 = v.x
    : "l"(a), "l"(b)              // %2 = a,   %3 = b
    mul.lo.u64 %1, %2, %3;        // -> v.x is the LOW limb

a struct whose members are named `x` and `y` can carry low in either one, and the surrounding
C++ shift operators are the cross-check. Getting this backwards produces plausible-looking
garbage. (HEonGPU)

### Shift counts of 0 or >= 64 in hand-rolled 128-bit arithmetic

The same hand-rolled `uint128_t` (a pair of 64-bit limbs) that carries the PTX above almost
always implements `operator<<` / `operator>>` as three unguarded limb shifts:

    result.lo = value.lo >> shift;
    result.lo = (value.hi << (64 - shift)) | result.lo;   // shift == 0 -> hi << 64
    result.hi = value.hi >> shift;                        // shift >= 64 -> UB

Shifting a 64-bit value by 64 or more, or by `64 - 0`, is undefined in C++ and **the two back
ends resolve it differently**: PTX `shr.u64` / `shl.u64` clamp the count to the register
width and so yield 0 for any count >= 64, while AMDGPU's `v_lshrrev_b64` uses only the LOW 6
BITS of the count, so a shift of 64 acts as a shift of 0 and of 65 as a shift of 1. AMD
returns the wrong limb OR-ed into the right one at every count >= 64. This is the 64-bit
sibling of the 32-bit mask fault in faults-wavefront.md (a shift count that wraps at 32
because the mask was built in `int`).

This is data-dependent, not arch-dependent, so it hides until the shift count crosses 64.
Barrett reduction shifts by `modulus.bit + 3`, which reaches 64 at a 61-bit modulus: on
HEonGPU every modular multiplication was correct up to 60 bits and garbage at 61 and 62, so
CKKS (37-bit moduli) passed while BFV, whose decryption uses a 61-bit correction modulus,
decrypted to noise. **Sweep the modulus/exponent width, not just the input values, when
validating modular arithmetic** -- a probe that only tests the widths a passing scheme uses
proves nothing about the one that fails.

Fix by branching on `shift == 0`, `shift < 64` and `shift < 128` explicitly. No `#ifdef` is
needed, but **do not tell the maintainer his CUDA numbers are unchanged**: the equivalence
with PTX holds only for counts 0 through 64. At exactly 64 the clamp happens to give the
right answer -- `lo >> 64` is 0, `hi << (64 - 64)` is `hi`, so the low limb already comes out
as `hi` and the high limb as 0, which is what the explicit branch computes. From 65 up it
does not: `64 - shift` underflows, PTX clamps every term and returns zero for the whole
value, while the correct result (and what the explicit branch computes) is
`hi >> (shift - 64)` in the low limb. So the guard is a CUDA-side fix too, at any count
above 64 -- on HEonGPU that is a 62-bit modulus, which the class documents as supported.
Check the reachable count range against the type's documented limits before writing
"behaviour-preserving" in a commit message, and tell the maintainer about the CUDA-visible
change. Audit `ROTL64`-style rotate macros in the same file for the `n == 0` case.
(HEonGPU: GPU-NTT `modular_arith.cuh`.)

Before carrying such a fix, and again at every upstream sync, check whether upstream has
fixed the defect itself. A fix written in inline PTX does not reach the HIP build: give the
HIP branch a helper with exactly the result of upstream's PTX for every shift count the
callers can produce, and drop your own fix rather than keeping a second, different one.
(HEonGPU: upstream GPU-NTT later fixed the 61-bit case with PTX shift helpers, and the port
replaced its explicit branches with HIP helpers that return zero for counts of 64 or more.)

## Half-precision GEMM accuracy

**rocBLAS `gemm_ex` rounds a 16-bit output once per k-chunk when k is long.** rocBLAS 5.7
(ROCm 10.1) splits k exactly when `max(lda, ldb) * k >= 2^30` for 2-byte types
(`gemm_tensile.hpp`: `safe_k = (2^31 - 1) / (max(ld_a, ld_b) * sizeof(Ti))`, split iff
`safe_k < k`), with the leading dimensions as passed whatever the transposes, ignoring strides
and batch count, into chunks of `safe_k`, and accumulates each chunk's product into D with
beta = 1. ROCm 10.2 (rocm-libraries #12219) narrows the split to the operands k strides
through, with the same 2^31-byte limit, so the 10.1 condition remains a safe superset. With a float16/bfloat16 D (`cublasGemmEx` -> `hipblasGemmEx` with compute type
f32) the output is rounded about k / 1023 times for a contiguous k-major operand, so the error
grows linearly with k: at k = 2^20 it is 25-50x a single rounding (fp16 7e-3, bf16 5e-2 relative),
and the 1025 chunk launches also take ~8 ms. `ROCBLAS_LAYER` logs only the outer call; a kernel
trace or `ROCBLAS_USE_HIPBLASLT=1` with the hipBLASLt log shows the chunks. A float D
(written to scratch, cast once) restores accuracy but is ~10x slower still; hipBLASLt
(`hipblasLtMatmul`, COMPUTE_32F, a few MiB of workspace) accumulates in float, rounds once and
runs these shapes in 45-85 us on MI300X, which is what torch uses there. Route only 16-bit
GEMMs with `max(lda, ldb) * k >= 2^30` to hipBLASLt so ordinary shapes keep their rocBLAS
path; strided batched (including stride 0 broadcast) works the same way. A regression test
needs k = 2^20 to clear a 4-ulp bound robustly; k = 2^15 splits only once. (oneflow: matmul
primitive, grouped_matmul_bias, fused self-attention.)

**gfx11 and gfx12 WMMA drift over long k, in every library, with any output type.** The float
accumulator of `v_wmma_f32_16x16x16_f16` does not round to nearest: one wave carrying it through
2^20 products is off by 3.4e-3 relative on gfx1100 and 1.7e-3 on gfx1201, against 3.1e-5 for an
fp32 FMA chain, and the bias is toward zero (it grows linearly with k). So on gfx1100 every
hipBLASLt algorithm for a long-k fp16 GEMM gives the same error (all 298 that fit 4 MiB, float D
too: about 6-8 fp16 ulp at k = 2^20; gfx1201 2.5-3.7), torch matches it digit for digit, and
rocBLAS does too wherever it does not chunk k itself (cuBLAS N,T, where neither operand is
k-major). Choosing another algorithm does not help; shortening the chain does. Do the chunking
yourself rather than asking hipBLASLt for a split-k: cut k into equal chunks of at most 2^15, run
them as batched GEMMs (batch stride = chunk along k) that write each chunk's product to its own
slice IN THE 16-BIT TYPE OF D, and sum the slices in float into D, rounding once. A 16-bit slice
limits each chunk's sum to the range and precision of that type: a float16 chunk of 8192 tokens
whose products average above 8 (loss-scaled weight gradients) is inf even when the chunks cancel to
a finite result, and finite chunk sums far larger than the result they cancel to are rounded
relative to their own size (x = 4, dy = +-1.9 in 8192-token blocks: chunk sums near 62000, results
near 250, 820 ulp; one unchunked WMMA GEMM gives 89). In the kernel that sums the slices, compute
a block of outputs again from A and B as float dot products over all of k when any element's slices
are not finite or a BOUND on their rounding error exceeds 3 units of roundoff of the output type (3
x 2^-11 float16, 3 x 2^-8 bfloat16) times the block's largest result. The bound is the sum of half
an ulp of each slice in its type (the float's exponent bits times the unit roundoff; 0 below the
smallest normal, so all-zero inputs never recompute, at the price of half a subnormal step per
slice left out) plus half an ulp of each float partial sum. That needs no host sync, works under
graph capture, keeps true inf/NaN inputs non-finite, and holds in the worst case. Do not use a
root-sum-square test (rss of the slices against sqrt(3) times the block maximum): it bounds only
the EXPECTED error, and chunk sums that all round the same way pass it and lose 4.2 / 4.5 / 9.4
ulp at k = 2^16 / 2^19 / 2^20 (a dot product of ones whose chunk sums sit just short of a rounding
midpoint: 8.8 ulp in float16 and bfloat16; oneflow's regression test). The bound never fires on
full 32 x 8 blocks of random data up to k = 2^20, but it does in blocks with few valid elements
(uniform random data, k = 2^16 / 2^20: 39 / 72% of 1-element blocks, 21 / 57% of 2, 7 / 38% of 4,
1.4 / 19% of the 1 x 8 edge blocks of a matrix-vector product, 0 / 0.8% of 32), so the recompute
must be cheap: one thread block walking k in 32-step tiles takes k / 32 serial steps (28 ms for one
element at k = 2^20). Spread k over the block's threads with 16-byte loads and, when the output has
few blocks, over several thread blocks per output block whose partial sums the last one to finish
adds in a fixed order (deterministic); that brought one recomputed element at k = 2^20 to about 20
us. A recomputed 1 x 8 block still reads its 8 rows of A again, so the bound's cost is the fired
fraction of a second read of A: on gfx1100 (oneflow) (1024, 2^20) @ (2^20, 1) took
5.1 ms against 4.2 without any recompute (+20%; root-sum-square +1-16%), (4096, 65536) @ (65536,
33) and (64, 2^20) @ (2^20, 1) +3-4% over root-sum-square, and every other shape measured,
including large weight gradients, the same as root-sum-square. Float slices for such thin
outputs would stop those firings, but hipBLASLt has no float-D algorithm for 1 x 1 outputs and
rocBLAS splits k into many launches for them, and per-shape costs went both ways (oneflow, not
adopted).
Watch the register count while doing it: a fully unrolled 32-load loop (which hipcc otherwise
serializes, waiting on every load) took the kernel from 66 to 248 VGPRs and the common case 11%
slower, and even 98 VGPRs (two 16-byte runs in flight) cost large GEMMs 2-3% and full-block
recomputes 9-34%; one run in flight per operand kept 79 VGPRs. An output where every full block
recomputes is still 30-120x slower. Chunking bounds the chain
length, not cancellation: WMMA's error grows with the size of the partial sums, so ordinary
unchunked gfx11 GEMMs are also tens of ulp off when large partial sums cancel (29-52 ulp at k =
8192-32768 with x = 4, dy = +-3 in 4096-token blocks), and cancellation inside one chunk is
invisible to the slice check (x = 4, dy = +-3 in 4096-token blocks of a 64 x k x 32 x^T @ dy:
78 / 182 / 269 ulp at k = 2^16 / 2^18 / 2^20, where torch's single hipBLASLt GEMM, which splits k
for such small outputs, gives 20 / 45 / 68; only a float accumulation without WMMA gets about 1,
at 7-200x the cost). Realistic loss-scaled gradients (random data, results near 30000) stay at
0.55-0.92 ulp. Do not give the chunks a float D: on gfx1100 hipBLASLt picks a much smaller tile for a float D (an fp16 4096 x
40000 x 4096 GEMM takes 139 ms with a float D against 16 ms with an fp16 one; 23-28x through
torch's `out_dtype=float32`), which made x^T @ dy weight gradients with 40000-65536 tokens
10-28x slower than one unchunked GEMM. One rounding per slice costs little: chunks of about 2^13
(at most 16 slices unless that would exceed 2^15) measured 0.9-1.2 ulp for large outputs and <=
1.2 ulp in every layout up to k = 2^20 on gfx1100, more accurate than 2^15 chunks (1.2-1.5 ulp)
and no slower. Run each chunk on the library ordinary GEMMs use (rocBLAS, about 1.5x faster than
hipBLASLt for large outputs on gfx1100) unless rocBLAS would split it again
(`max(lda, ldb) * chunk >= 2^30` in ROCm 10.1, see the rocBLAS entry above, where it rounds a
16-bit D per inner chunk). With that, large weight gradients run within
about 1.1x of one plain GEMM. Fold the batch into the chunk calls, tile the slices over m and n to
a memory budget with balanced tiles, and if a library has no algorithm for one tile, run that
tile alone through rocBLAS with a float D rather than failing the call after earlier tiles were
written. Two reasons not to rely on `GemmTuning::setSplitK`: its partials need m * n * split
floats of workspace, and on gfx1201 the heuristic's first kernel for k-major A and small outputs
(`MT16x16x192`) is wrong by tens of ulp for many k above 2^17 and stays wrong with the split,
which it accepts and runs. A GEMM whose k never exceeds 2^15 stays out of both. The gfx1100 costs
above were not measured on gfx1201. Gate on the k length on the WMMA families, not on
`ld * k`. gfx9 MFMA keeps fp32 accuracy. (oneflow: long-k matmul on gfx1100 and gfx1201.)

**A GEMM workaround covers every entry point that reaches the library, fused epilogues included.**
oneflow routed its matmul, bmm and grouped GEMMs to the chunked long-k path but its cuBLASLt
compat (`cublasLtMatmul` on hipBLASLt, used by the fused linear and MLP kernels and their
backward) still sent the whole k to hipBLASLt: 5.7-8.4 float16 ulp on gfx1100 at k = 2^20 and
98.75 ulp on gfx1201 at k = 163840 (the ROCm 10.1 hipBLASLt DepthU-192 kernels, fixed in hipBLASLt
1.5.0). Grep for every call into the BLAS libraries, including Lt/fused wrappers, and put the
selection in the one compat function they share. Run the GEMM through the same path and apply
the epilogue (bias, ReLU, ReLU bit mask, masked gradient, bias gradient) afterwards in separate
kernels; the result then equals the unfused ops bit for bit, which also gives the test a check
that the path was taken (gate it on ROCm: cuBLASLt applies the epilogue before rounding). If the
long-k path needs another library's handle (rocBLAS here) and the compat only receives the Lt
handle, create the second handle with the first: a kernel may first run inside a graph capture,
where the 32 MiB a rocBLAS handle allocates at creation cannot be. (oneflow, 0e606373b.)
