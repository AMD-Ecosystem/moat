# HEonGPU notes

## Build

Library and tests build successfully on linux-gfx90a:

```bash
cd projects/HEonGPU/src
mkdir build && cd build
cmake -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release -DHEonGPU_BUILD_TESTS=ON ..
cmake --build . -j$(nproc)
```

Output: `src/libheongpu.a`, test executables in `bin/test/`, plus dependencies (`libntt-1.0.a`, `libfft-1.0.a`, `librngongpu-1.0.a`)

## Submodule Updates

GPU-FFT, GPU-NTT, and RNGonGPU (all by the same author, Alisah Ozcan) were updated to HIP-compatible commits. These submodules contain CUDA-specific code that needed adaptation:

1. **GPU-NTT** (abbb2c3): PTX inline assembly in `modular_arith.cuh` replaced with HIP-compatible `__umul64hi` intrinsic for 128-bit multiplication. Chained comparison syntax fixed for clang.

2. **GPU-FFT** (00d3f8b): CMake HIP support added (USE_HIP option, HIP language, hipcc compilation).

3. **RNGonGPU** (50558fe): CUDA->HIP compatibility header added. `hipPointerAttribute_t.type` member used (differs from CUDA's `.memoryType`). Links hiprand instead of curand.

The submodules link `hip::host` (not `hip::device`) to avoid propagating HIP compile flags to downstream consumers. This allows pure C++ test executables to link against the HIP library without requiring HIP compilation themselves -- though they still need HIP compilation to use rocThrust headers.

## Port Details

### Key adaptations:

1. **cuda_to_hip.h header**: Central compatibility header mapping CUDA runtime symbols to HIP equivalents, including cuRAND->hipRAND mappings and warp size abstraction.

2. **rmm_hip_stub/**: Minimal RMM implementation for HIP since the real RMM does not support HIP. Implements `device_uvector`, `device_buffer`, `pool_memory_resource`, `statistics_resource_adaptor`, `pinned_memory_resource`.

3. **hip_compat/**: Shim headers for thirdparty code that includes `cuda_runtime.h` and `curand_kernel.h` directly.

4. **PTX inline assembly**: Replaced in `bigintegerarith.cuh` and `GPU-NTT/modular_arith.cuh` with portable C++ using `__umul64hi` intrinsic.

5. **Device function linking**: Made `SmallForwardNTT`/`SmallInverseNTT` inline in header to avoid cross-TU device linking issues (HIP doesn't support CUDA's CUDA_SEPARABLE_COMPILATION the same way).

6. **Warp shuffles**: HIP uses `__shfl_down(val, offset)` without mask; CUDA uses `__shfl_down_sync(mask, val, offset)`.

7. **Warp size**: Changed hardcoded `32` to runtime `warpSize` for gfx90a wave64 compatibility.

8. **HostVector**: Added explicit copy/move assignment operators (clang stricter than NVCC about std::vector inheritance).

9. **hipPointerAttribute_t**: HIP uses `.type` member, not `.memoryType`.

10. **CMake HIP flag propagation fix**: Changed `hip::device` to `hip::host` in GPU-NTT, GPU-FFT, and rmm_hip_stub CMakeLists.txt. The `hip::device` target propagates HIP compile flags (`-x hip --offload-arch=gfx90a`) via INTERFACE properties to all downstream targets, causing g++ to fail on pure C++ files. The `hip::host` target provides the HIP runtime library without compile-time flags.

11. **Test compilation**: Test .cpp files are compiled as HIP sources (`set_source_files_properties(... LANGUAGE HIP)`) because they transitively include rocThrust headers via heongpu.hpp. rocThrust requires HIP compilation context.

12. **RMM HIP stub error checking**: Added `hipError_t` return value checking to all allocation functions in the RMM stub. Throws `std::runtime_error` on allocation failure. Deallocation ignores errors (cannot throw in destructors).

## Review 2026-06-05

### Test Results
- Pass: bfv_encoding, ckks_encoding (2/20)
- Fail: All other tests (18/20) -- encryption, decryption, addition, multiplication, relinearization, rotation, TFHE

### Root Cause Analysis

The porter suspected warp size handling. However:

1. **GPU-NTT `<< 5` patterns are NOT warp-size bugs**: The `<< 5` shifts in `thirdparty/GPU-NTT/src/lib/ntt_4step/ntt_4step.cu` are algorithmic indexing into 32-element shared memory rows (`__shared__ T sharedmemorys[32][32+1]`), not warp-lane operations. The NTT algorithm inherently uses 32-element blocks regardless of hardware warp width.

2. **Warp-level fixes look correct**: The `warp_reduce` function in `src/include/heongpu/util/util.cuh:312` correctly uses runtime `warpSize`. The warp index calculations in encryption/decryption/keygeneration kernels (`wid = idx / warpSize`) are also correct.

3. **Encoding passes because it only uses NTT/INTT**: The encoding tests call `GPU_NTT` and `GPU_INTT` which work correctly. All failing tests additionally involve:
   - Random number generation (AES-based, not curand/hiprand)
   - Key generation
   - Modular arithmetic with biginteger operations

### Potential Issues Requiring Investigation

1. **bigintegerarith.cuh carry/borrow logic** (`src/include/heongpu/util/bigintegerarith.cuh:77-169`): The HIP path uses manual overflow detection. Example at line 85:
   ```cpp
   carry = (sum < a || (carry && sum == a)) ? 1 : 0;
   ```
   This logic appears correct on paper but should be verified against the PTX version for edge cases.

2. **AES RNG differences**: HEonGPU uses `rngongpu::RNG<rngongpu::Mode::AES>` for random number generation, not curand/hiprand. The AES implementation should produce identical results on CUDA and HIP, but this needs verification.

3. **Missing hypothesis**: The fact that ALL cryptographic operations fail (encryption, addition, multiplication, relinearization, rotation across BFV, CKKS, and TFHE schemes) while encoding works suggests a common code path issue, likely in either:
   - Random polynomial generation
   - NTT-domain modular multiplication
   - Key generation

### Verdict

**changes-requested**: The port builds and encoding works, but cryptographic operations produce incorrect results. This requires debugging before validation. The warp-size hypothesis is unlikely; the issue is more subtle and requires:
1. Unit testing the biginteger arithmetic against known values
2. Comparing RNG output between CUDA and HIP builds
3. Step-by-step comparison of a simple encrypt/decrypt operation

## Porter Debug Attempt 1 (2026-06-05)

### Components Verified Working

1. **Barrett modular multiplication**: Tested with 1024 random 36-bit inputs, all correct including edge cases (max values, zero, one).

2. **128-bit subtraction**: Tested 8 cases including borrow propagation, all correct.

3. **128-bit multiplication** (`__umul64hi`): Implicitly verified through Barrett test passing.

4. **Ternary RNG**: `modular_ternary_random_number` produces valid ternary values (-1, 0, 1 mod q). Distribution is skewed (50% -1, 25% 0, 25% 1) but this matches the original algorithm logic and doesn't cause complete failure.

5. **hiprand device API**: `hiprand_init`/`hiprand` produce reasonable random values with good distribution.

6. **Encode/decode roundtrip**: Works perfectly, confirming NTT/INTT with the library's own tables is functional.

### Key Finding

Tested `pk[0] + pk[1]*sk` (pointwise in NTT domain): values are NOT small. Expected: small Gaussian error (|e| < 50 for std_dev=3.2). Actual: values like 34 billion (half of 68B modulus).

This means the public key relationship `pk[0] = -a*s + e, pk[1] = a` is NOT being satisfied. The bug is in either:
- Public key generation kernel (`publickey_gen_kernel`)
- NTT transformation of the involved polynomials
- Data layout mismatch (wrong RNS component being used)

Note: Cannot directly verify pk[0]+pk[1]*sk=e in NTT domain because e in NTT domain is NOT small (NTT spreads energy). The relationship only holds algebraically for polynomial multiplication. However, the observed values are so large (50% of modulus) that even accounting for this, something is fundamentally wrong.

### What the Error Pattern Tells Us

Encryption of ALL-ZEROS produces decrypted values of magnitude ~50% of plain_modulus (max error ~515840 out of 1032193). This is NOT noise, it's complete garbage. If only the noise were wrong, we'd see small but incorrect noise. The magnitude indicates the core algebraic relationship is broken.

### Remaining Hypotheses

1. **NTT table generation bug**: Tables are generated on CPU. If the primitive roots or powers are wrong, all NTT-domain operations fail.

2. **Inconsistent RNS indexing**: The various kernels (keygen, encrypt, decrypt) might use different conventions for indexing RNS components.

3. **modulus_->data() ordering**: The array of moduli on GPU might not match what kernels expect.

4. **NTT layout mismatch**: `ntt_rns_configuration.ntt_layout = gpuntt::PerPolynomial` might behave differently than expected.

### Continued Analysis

**Key test**: Verified that `pk[0] + pk[1]*sk` in NTT domain produces values that are ~50% of the modulus (should be small Gaussian error). This means the fundamental relationship is broken.

The public key formula is `pk[0] = -(s*a + e), pk[1] = a` (verified by reading publickey_gen_kernel). The relationship `pk[0] + pk[1]*sk = -e` should hold in NTT domain.

**Verified components**:
1. CPU Barrett multiplication (used in NTT table generation): works
2. GPU Barrett multiplication: works
3. 128-bit subtraction/multiplication: works
4. hiprand device API: works
5. Encode/decode (plaintext NTT): works

**Open question**: The GPU-NTT RNS API call semantics. The keygenerator calls `GPU_NTT_Inplace(errors_a.data(), ..., Q_prime_size, Q_prime_size)` where errors_a has 2*Q_prime_size*n elements but the call pattern suggests only Q_prime_size*n elements are processed (only error_poly, not a_poly). This is the ORIGINAL code pattern -- if this were wrong, upstream CUDA would also fail. Need to understand the batch_size/mod_count semantics to confirm.

**Hypotheses**:
1. NTT table generation produces wrong values on HIP (despite CPU Barrett working)
2. There's a subtle difference in GPU kernel execution (memory layout, register usage) between CUDA and HIP that corrupts NTT results
3. The `__umul64hi` intrinsic or 128-bit subtraction has edge cases we haven't tested

### Next Steps (for attempt 2)

1. Create a direct NTT roundtrip test using the coefficient moduli tables to verify NTT/INTT work correctly

2. Add debug output to publickey_gen_kernel to print intermediate values on GPU

3. Test with a simpler configuration (smaller polynomial, single modulus) to isolate the bug

4. Binary comparison of GPU memory after NTT between CUDA (if accessible) and HIP builds

## Porter Debug Attempt 2 (2026-06-05)

### Additional Component Verification

1. **GPU Barrett mult/add/sub in isolation**: All pass on gfx90a (test_gpu_barrett.cu)

2. **NTT with single modulus (N=16, N=4096)**: PASS when using `GPU_NTT_Inplace` API with correctly generated tables (test_ntt_small.cu). The NTT kernel itself works correctly.

3. **NTT with separate in/out buffers**: Initially FAILED (test_ntt_coeff.cu), but after switching to `GPU_NTT_Inplace` it PASSES. This suggests either:
   - A bug in how I was calling `GPU_NTT` (less likely)
   - Or the test tables were mismatched (I generated them manually instead of using HEonGPU's table generator)

### Key Finding

The NTT kernels work correctly when:
- Tables are generated with correct primitive roots
- `GPU_NTT_Inplace` API is used (same pattern HEonGPU uses)
- Single modulus configuration is tested

This narrows the bug to one of:
1. **Table generation in HEonGPU context.cu**: The `generate_ntt_table` / `generate_intt_table` functions use `generate_primitive_root_of_unity` to find psi. If this finds the wrong primitive root for some modulus, all crypto operations fail.

2. **RNS table layout mismatch**: HEonGPU concatenates tables for multiple moduli `[mod0_table, mod1_table, ...]`. If the kernel indexes into the wrong modulus's table, it uses wrong twiddle factors.

3. **modulus array vs table array ordering**: The `modulus_->data()` array and `ntt_table_->data()` must have matching order. A mismatch would cause polynomial_i to be NTT'd with modulus_j's table.

4. **Something specific to RNS API**: The RNS version `GPU_NTT_Inplace(data, table, modulus_array, cfg, batch_size, mod_count)` may have a HIP-specific bug that the single-modulus version doesn't have.

### Test Files Created

- `test_gpu_barrett.cu`: GPU Barrett mult/add/sub test (PASS)
- `test_ntt_small.cu`: Single-modulus NTT test (PASS with N=16, N=4096)
- `test_ntt_coeff.cu`: NTT roundtrip with 36-bit prime (PASS with Inplace API)
- `test_ntt_rns.cu`: RNS multi-modulus test (incomplete - needs correct primes)

### Next Steps (for attempt 3)

1. **Compare HEonGPU's generated tables with independently computed tables**: Dump the `ntt_table_` contents from context after generation and verify they match what we expect for the given primitive roots.

2. **Add debug kernel to verify NTT table ordering**: Before calling keygen, verify that `modulus_[i]` corresponds to `ntt_table_[i*n : (i+1)*n]` by checking `psi^N = -1 mod q` for each modulus.

3. **Trace the RNS NTT path in detail**: The failing keygen uses the RNS API. Test the exact same call pattern with known-good tables to isolate whether it's the tables or the RNS kernel.

4. **Check primitive root finding**: `find_minimal_primitive_root` in util.cu may have an issue. Verify it finds the correct psi for each coefficient modulus.

## Porter Debug Attempt 3 (2026-06-11)

### IMPORTANT: prior submodule HIP-port work was LOST (never pushed)

**WRONG -- see "Porter Debug Attempt 4". The submodules' AMD GPU support is carried
as patch files on this branch and the pinned SHAs all resolve upstream.**

The local clone `projects/HEonGPU/src` had been deleted between attempt 2 and
attempt 3. Re-cloning the AMD-Ecosystem/HEonGPU fork's `moat-port` branch revealed
that the branch pins three submodule commits that exist NOWHERE:

- thirdparty/GPU-FFT  -> ac4b587 (not in upstream, no fork)
- thirdparty/GPU-NTT  -> 99cb3cc (not in upstream, no fork)
- thirdparty/RNGonGPU -> 50558fe (not in upstream, no fork)

`.gitmodules` still points at the upstream Alisah-Ozcan URLs, and no jeffdaily
forks of these submodules were ever created. So the entire submodule HIP port
(the PTX->intrinsic edits, the CMake HIP gating, the curand->hiprand swap) was
only ever committed in the now-deleted local clone and is unrecoverable from
git. The main HEonGPU repo's port (cuda_to_hip.h, hip_compat/, rmm_hip_stub/,
CMake) IS preserved on the fork at b91755d.

### Build reconstruction (done this attempt)

I reconstructed the submodule HIP support from scratch (saved as patches in
agent_space/HEonGPU-attempt3/{GPU-NTT,GPU-FFT,RNGonGPU,parent}.patch, gitignored
-- reapply on attempt 4, and create the jeffdaily submodule forks so this is
never lost again). The tree now builds cleanly for gfx90a (library + all 15
test executables). Changes:
- GPU-NTT/GPU-FFT/RNGonGPU CMakeLists: option(USE_HIP), project(... HIP ...),
  set_source_files_properties(... LANGUAGE HIP), CUDA::cudart->hip::host,
  CUDA::curand->hip::hiprand, hip_compat on the include path, HIP_ARCHITECTURES.
- modular_arith.cuh: PTX mul.lo/mul.hi -> a*b + __umul64hi(a,b); PTX sub.cc/subc
  -> manual borrow subtraction. Lo/hi mapping verified correct (value.x=lo,
  value.y=hi). This was a clean reconstruction; it is NOT the crypto bug.
- nttparameters.cu: clang rejects chained comparisons `0 < logn <= 25` as an
  error (-Wparentheses); rewrote to `0 < logn && logn <= 25` (10 sites).
- hip_compat/cuda_runtime.h: define CUDART_VERSION=10000 so base_rng.cu selects
  the hipPointerAttribute_t `.type` path (HIP has no `.memoryType`).
- hip_compat/curand_mtgp32_host.h + curand_mtgp32dc_p_11213.h shims (the
  RNGonGPU cuda_rng path includes them but never uses MTGP; hipRAND has the
  _host header, no dc_p header -> empty shim).
- thirdparty/build.sh: no longer force `git submodule update --init` (it would
  discard the local submodule HIP edits and fails on the missing pinned SHAs).
- Build deps installed: libssl-dev, libgmp-dev, libntl-dev.

### ROOT-CAUSE BISECTION -- first divergence pinpointed

Built a probe (agent_space/HEonGPU-attempt3/moat_probe.cpp) that links the real
library, generates a BFV context (N=4096, coeff {36,36},{37}), and exercises the
RNS NTT path with HEonGPU's OWN generated tables via temporary public accessors
on HEContextData. Findings, in order:

1. NTT tables are CORRECT. For modulus 0, psi = ntt_table[brev(1)] satisfies
   psi^N == q-1 and psi^(2N) == 1; ntt_table[brev(2)] == psi^2; the inverse
   table satisfies intt_table[brev(1)] == psi^-1 and psi*psi^-1 == 1; the stored
   n_inverse == N^-1 mod q. So context.cu table/primitive-root generation is
   NOT the bug (this overturns the attempt-2 hypothesis).

2. FORWARD NTT is CORRECT. GPU forward NTT output matches an independent O(n^2)
   CPU negacyclic reference EXACTLY when compared in bit-reversed order
   (0 mismatches / 4096; the GPU emits bit-reversed order, which is normal for
   the merge-NTT). Barrett mult, GentlemanSande/CooleyTukey units all fine.

3. INVERSE NTT is BROKEN on gfx90a. INTT(NTT(x)) != x for every coefficient:
   4096/4096 mismatches in BOTH natural and bit-reversed orderings, with the
   recovered values large and unrelated to the input by any constant factor
   (genuine corruption, not a scale/order offset). Single-modulus (batch=1) and
   multi-modulus (batch=3) both fail identically. Forward alone is correct;
   only the inverse miscomputes.

FIRST DIVERGENCE: the GPU-NTT inverse merge kernel (gpuntt::InverseCore /
GentlemanSande path, thirdparty/GPU-NTT/src/lib/ntt_merge/ntt.cu, e.g. line
1089) produces wrong results on ROCm/gfx90a, while the forward kernel in the
same file is correct. Tables, n_inverse, and Barrett arithmetic are all verified
correct, so the fault is inside the inverse kernel's execution on HIP, not in
table generation or modular arithmetic. The logn=12 inverse kernel-param table
(ntt.cuh CreateInverseNTTKernel) uses two kernels with blockdims 256 and 64x4,
512*sizeof(T) shared -- all within HIP limits and wave-agnostic by inspection,
so the param table is not obviously the cause. Note the build emitted
"loop not unrolled [-Wpass-failed]" warnings ONLY on the inverse kernels
(InverseCoreModulusOrdered / InverseCorePolyOrdered) -- worth checking whether
an inverse-specific #pragma unroll over a runtime-bounded loop miscompiles, or
whether the non-last inverse kernel's in-place write-back / global addressing
breaks under the amdclang code generation.

### Next Steps (for attempt 4) -- start from the inverse kernel

**SUPERSEDED. Both premises below were disproved in attempt 4: the submodule work
was never lost, and the inverse kernel is correct. See "Porter Debug Attempt 4".**

1. FIRST: create jeffdaily forks of GPU-NTT, GPU-FFT, RNGonGPU, reapply the
   agent_space patches, commit+push there, and re-point .gitmodules + the
   moat-port gitlinks so the build is never lost again.
2. Bisect the inverse kernel: test InverseCore at small N that uses a SINGLE
   inverse kernel (no multi-kernel split) vs N=4096 (two kernels). If single
   works and multi fails, the bug is the multi-kernel inverse decomposition /
   in-place buffering on HIP.
3. Inspect the inverse `#pragma unroll for (lp < loops)` (loops = runtime
   outer_iteration_count) -- the loop-not-unrolled warning is inverse-only.
   Try removing the unroll pragma on the inverse path and re-test.
4. Diff the SASS/ISA or run with -O0 on the inverse kernel to see if it is an
   optimizer miscompile (amdclang) rather than a logic port bug.
5. The probe in agent_space reproduces the failure in seconds without running
   the full gtest -- use it as the inner loop.

## Resuming (2026-08-07)

**SUPERSEDED by "Porter Debug Attempt 4" below.**

The port continues: this was judged a port worth finishing rather than an unportable
codebase, so linux-gfx90a is no longer marked blocked. The blocker as last recorded, which
is where to pick it up:

Attempt 3: tree rebuilt for gfx90a (lib + all 15 tests compile); crypto still fails. ROOT CAUSE ISOLATED: GPU-NTT INVERSE merge kernel (GentlemanSande/InverseCore, thirdparty/GPU-NTT/src/lib/ntt_merge/ntt.cu) miscomputes on gfx90a -- INTT(NTT(x))!=x (4096/4096 corrupt). Verified CORRECT: ntt/intt tables (psi^N==q-1, intt psi==psi^-1), n_inverse==N^-1, forward NTT (matches CPU ref exactly, bit-reversed), Barrett mult. Forward fine, inverse broken. ALSO: submodule HIP forks (GPU-NTT/FFT/RNGonGPU) were never pushed and the moat-port branch pins lost SHAs -- recovery patches in agent_space/HEonGPU-attempt3/. Attempt 4: bisect inverse single- vs multi-kernel, check inverse-only #pragma unroll loop-not-unrolled / amdclang miscompile.

## Porter Debug Attempt 4 (2026-08-11)

### CORRECTION: two claims from attempt 3 are wrong. Do not act on them.

**1. The submodule HIP work was NOT lost, and no submodule forks are needed.**
`.gitmodules` points at the three UPSTREAM repos (`Alisah-Ozcan/{GPU-NTT,
GPU-FFT,RNGonGPU}`) and the pinned gitlink SHAs (`8a4daf11c7`, `b743607c11`,
`d9aaa6b5d7`) all resolve upstream. The submodules' AMD GPU support lives on our
branch as `thirdparty/patches/{GPU-NTT,GPU-FFT,RNGonGPU}.patch`, applied
idempotently by `thirdparty/build.sh ON` (the reverse-apply check makes repeated
configures a no-op). Nothing is lost and `agent_space/HEonGPU-attempt3/` being
gone costs nothing. A submodule fix goes into the patch file, never into a
submodule checkout: edit the checkout, then regenerate with
`git -C thirdparty/<name> diff > thirdparty/patches/<name>.patch`.

**2. The GPU-NTT inverse merge kernel is NOT broken on gfx90a.** Attempt 3's
"INTT(NTT(x)) != x, 4096/4096 corrupt" does not reproduce. A probe linking
`libntt` directly and comparing against GPU-NTT's own `NTTCPU` reference gives
0 mismatches for logn 4..14, both reduction polynomials, forward and inverse,
round trip, in-place and out-of-place, single-modulus and RNS with mod_count
1/2/3. The likely cause of the attempt-3 result is that the RNS *inverse* is
reached through `GPU_INTT_Inplace`, not `GPU_NTT_Inplace`: the latter dispatches
the FORWARD path regardless of `cfg.ntt_type`, so calling it with an inverse
config silently runs a forward transform with the inverse tables and corrupts
every coefficient. That exact mistake reproduces attempt 3's numbers exactly.
GPU-NTT is exonerated; stop looking there.

Also verified clean this attempt, so do not re-verify:
- the device Barrett operators (`mult`, `add`, `sub`, `reduce_forced`) against a
  128-bit host reference at 20-, 36-, 37-, 59- and 60-bit moduli: 0 mismatches;
- the ternary and uniform samplers (values in range, RNS-consistent).

### ROOT CAUSE FOUND AND FIXED: negative float to unsigned conversion

`box_muller_kernel` in `RNGonGPU/src/lib/common/base_rng.cu` wrote the Gaussian
error as `static_cast<T>(z0) + (flag0 & modulus.value)`, where `T` is the
unsigned residue type and `z0` a signed double. The `flag` term only makes sense
if the cast wraps a negative value in two's complement, but converting an
out-of-range float to an unsigned integer is undefined, and the back ends
disagree. Measured on gfx90a:

```
(uint64_t)  -1.00 = 4294967295  (0x00000000ffffffff)
(uint64_t)  -2.28 = 4294967294  (0x00000000fffffffe)
(uint64_t) -20.00 = 4294967276  (0x00000000ffffffec)
(uint64_t)   2.28 =          2
```

AMD keeps only the low 32 bits; NVIDIA's `cvt.rzi.u64.f64` saturates a negative
source to zero. So on gfx90a roughly half of every error polynomial was ~2^32
instead of single digits. Before the fix, for a 4096 ring at std_dev 3.2:
`max|centered| = 4294967295`, `mean|centered| = 1.6e9`, 2031/4096 values >= q.
After: `max|centered| = 13`, `mean|centered| = 2.07`. (The residual "out of
range" count of ~518 is a sample in (-1,0) truncating to 0 and then storing
exactly `q`; that is upstream behaviour, is congruent to 0, and is harmless.)

Fix: a `truncate_signed<T>(U)` helper in `RNGonGPU/src/include/rngongpu/common/
aes.cuh` that casts through `std::make_signed_t<T>` first, used at the twelve
Box-Muller sites in `base_rng.cu` and the six curand-backed sites in
`cuda_rng_kernels.cu`. NOT behaviour-neutral on NVIDIA -- it replaces the
saturated zero with the correct negative representative, which is what the
surrounding arithmetic already assumes. Call that out to the maintainer.

Note the mirror case does NOT diverge: `static_cast<uint32_t>(negative double)`
clamps to 0 on gfx90a exactly as on NVIDIA (measured). So the TFHE sites at
`keygeneration.cu:1142,1202,1203` are the same latent upstream issue but not a
port divergence -- leave them alone.

### Result: 2/20 -> 11/20 suites passing

All seven CKKS suites now pass, plus both encoding suites. Still failing (9):
the seven BFV suites, BFV encryption/decryption, and TFHE_Gate_Boots.

Fork head after this attempt: `14c2b5162938d527b754ac361aaa17f7078860d4`.

This fix is expected to help the wave32 host too: the divergence is in the AMDGPU lowering
of a 64-bit `fptoui`, not in the wavefront width, which is consistent with the gfx1100
attempt-4 finding below that gfx1100 failed the same 18 suites as gfx90a. It was measured
on gfx90a only, so a gfx1100 run should confirm the same 9 remaining failures rather than
assume them. The gfx1100 session's conclusion -- "start at the RNG, not the NTT" -- was
right, and this is what was there.

### Where attempt 5 should start

**BFV (8 of the 9 failures).** Encode/decode round-trips exactly (0/4096
mismatches) but encrypt-then-decrypt corrupts every plaintext coefficient
(4096/4096 differ). CKKS passing rules out the RNG, both NTT directions,
`enc_div_lastq_ckks_kernel`'s modulus-switching logic, and the Barrett
operators, so the fault is in BFV-specific code:
`enc_div_lastq_bfv_kernel` (`src/lib/kernel/encryption.cu:91`) and
`decryption_kernel` (`src/lib/kernel/decryption.cu:44`), or the host constants
they consume (`Q_mod_t_`, `upper_threshold_`, `coeeff_div_plainmod_`, `gamma_`,
`Qi_t_`, `Qi_gamma_`, `Qi_inverse_`, `mulq_inv_t_`, `mulq_inv_gamma_`,
`inv_gamma_`, all computed on the host in `src/lib/host/bfv/context.cu`). Two
concrete suspects not yet ruled out: `fix = int(fix / plain_mod.value)` in
`enc_div_lastq_bfv_kernel` narrows a `Data64` quotient through `int` (it should
fit for the tested parameters, but check it for large plain moduli), and the
host-side constant generation, which no probe has compared against an
independent computation. The cheapest next bisect is a secret-key encryptor (if
BFV exposes one) versus the public-key path, which splits keygen from encrypt.

**TFHE (1 failure).** The three `warp_reduce` call sites are all TFHE:
`decrypt_lwe_kernel` (`decryption.cu:441`), `encrypt_lwe_kernel`
(`encryption.cu:280`) and `tfhe_generate_switchkey_kernel`
(`keygeneration.cu:1079`). `warp_reduce` in `util.cuh:308` loops on runtime
`warpSize` and uses maskless `__shfl_down`, which is right for wave64, and the
`sdata[wid]` sizing over-allocates at wave64 rather than under-allocating, so
the obvious reading is clean. The unexamined piece is the host-side shared
memory byte count passed to those three launches -- check it, and check whether
`encrypt_lwe_kernel`'s per-thread `curandState_t` count matches the launch
geometry (hipRAND state is not the same size as cuRAND's).

### Probes (regenerate them; they are in gitignored scratch)

Four standalone probes were used and each runs in seconds. Build them against
the already-built static libraries rather than through CMake; copy the flags out
of `build/test/CMakeFiles/bfv_encoding_testcases.dir/{flags.make,link.txt}`,
which is much quicker than adding a CMake target. Compile with
`-x hip -DUSE_HIP --offload-arch=gfx90a` and pass the `.a` files in a SEPARATE
link step (`-x hip` applies to every following input, so an archive listed after
it is fed to the compiler as source).

1. `probe.cu` -- GPU-NTT forward/inverse against `NTTCPU`, in-place and
   out-of-place, plain and RNS. Links `libntt-1.0.a` only.
2. `rngprobe.cu` -- the three modular samplers via
   `RandomNumberGenerator::instance()`, reporting max/mean centred magnitude,
   out-of-range count and RNS consistency. Needs a generated `HEContext` first
   or the singleton segfaults.
3. `bfv.cu` -- encode, encrypt, decrypt, decode with the plaintext coefficients
   compared at each hop.
4. `mod.cu` -- the device Barrett operators against a `__uint128_t` reference.
## Porter Attempt 4 (2026-08-08, linux-gfx1100)

### The build is now durable in git (the main deliverable of this attempt)

The attempt-3 patches in `agent_space/` were gone again, as predicted -- that path is
gitignored and did not survive. The submodule port has now been reconstructed a third time
and committed to the fork so this cannot recur:

- The gitlinks are reset to the real upstream commits the branch was based on
  (GPU-FFT `b743607`, GPU-NTT `8a4daf1`, RNGonGPU `d9aaa6b`). The branch is clonable again;
  previously `git submodule update --init` failed on three SHAs that exist nowhere.
- The submodule HIP support lives in `thirdparty/patches/{GPU-NTT,GPU-FFT,RNGonGPU}.patch`,
  applied by `thirdparty/build.sh` (which CMake already invokes at configure time) and only
  when `USE_HIP=ON`. Applying is skipped when `git apply --reverse --check` succeeds, so
  repeated configures are idempotent.
- `.gitmodules` carries `ignore = dirty` per submodule, because applying the patches
  permanently dirties those working trees and would otherwise read as an unclean tree.
- `thirdparty/hip_compat/` gained `curand_mtgp32_host.h` and `curand_mtgp32dc_p_11213.h`
  shims and a `CUDART_VERSION 10000` define (steers `base_rng.cu` to the
  `hipPointerAttribute_t.type` branch).

A fresh clone of `moat-port` now configures and builds with no manual repair.

### Build result (gfx1100, ROCm)

Configure + build clean, 100%: `libheongpu.a`, `libntt-1.0.a`, `libfft-1.0.a`,
`librngongpu-1.0.a` and all 15 test executables. Build deps needed: `libgmp-dev`,
`libntl-dev` (plus `libssl-dev`, already present).

```bash
cmake -S projects/HEonGPU/src -B projects/HEonGPU/src/build -DUSE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release -DHEonGPU_BUILD_TESTS=ON
cmake --build projects/HEonGPU/src/build -j$(nproc)
ctest --test-dir projects/HEonGPU/src/build --output-on-failure
```

### DECISIVE FINDING: the fault is NOT wavefront-width dependent

`ctest` on gfx1100: **2/20 pass, 18/20 fail, in 11.5s** -- the SAME two passes
(`BFV_Encoding`, `CKKS_Encoding`) and the same 18 failures as the gfx90a run in the
2026-06-05 review.

gfx1100 is **wave32**, identical in wavefront width to a CUDA warp. Every warp-size
hypothesis carried since attempt 1 -- the `& 31` / `>> 5` patterns, the 32 vs 64 lane masks,
the shared-memory warp-count sizing -- is therefore **ruled out as the cause of the current
failure**. Whatever is wrong is in shared port logic and is arch-independent. Stop looking
at wave width.

### The attempt-3 root cause is contradicted by the test results and should not be inherited

Attempt 3 concluded the GPU-NTT inverse merge kernel miscomputes (INTT(NTT(x)) != x,
4096/4096 corrupt). That cannot be globally true: the two tests that PASS are the encoding
tests, and encode/decode round-trips through that same inverse NTT. Attempt 1 independently
recorded "Encode/decode roundtrip: works perfectly". The attempt-3 probe built its own
tables and accessors and most likely mismatched the library's conventions rather than
finding a real kernel fault.

The honest common factor across the 18 failures, and the 2 passes, is: **encoding is the
only path that does no random polynomial generation and no key generation.** Everything that
fails goes through `rngongpu::RNG<Mode::AES>` and/or keygen. That is where attempt 5 should
start, not the NTT.

### Verified correct by inspection this attempt (do not re-audit)

- `bigintegerarith.cuh` HIP carry/borrow predicates. Case analysis confirms both are exactly
  right, including the `carry_in=1, b=2^64-1` edge where the sum equals `a` and a carry is
  still produced. Attempt 1 also verified these empirically.
- GPU-NTT `modular_arith.cuh` limb order. From the PTX operand numbering (outputs numbered
  before inputs), `value.x` is the LOW limb and `value.y` the HIGH limb; the C++ shift
  operators agree. The `__umul64hi` and manual-borrow replacements match.

### Known gfx90a-only risk, not the current bug

`RNGonGPU/src/lib/common/aes.cu` uses `int warpThreadIndex = threadIdx.x & 31;` at three
sites (174, 381, 557). On wave32 this is exactly the lane id, so it is inert on gfx1100 and
cannot explain today's failure -- but on wave64 it is a genuine warp-size assumption and
will need an arch-unified fix (correct on both widths) once the shared bug is found.

### Next steps (attempt 5)

1. Start at the RNG, not the NTT. Compare `rngongpu::RNG<Mode::AES>` output on ROCm against
   the AES vectors / a CPU reference; the AES round keys and T-tables are uploaded with
   plain `cudaMalloc` + copy in `aes_rng.cu`, so also check for any buffer read before it is
   written (ROCm does not zero fresh allocations; CUDA often appears to).
2. Then key generation, which is the other thing every failing test shares.
3. Build GPU-NTT's own `example/ntt_merge/test_merge_ntt.cu` and `test_merge_intt.cu`
   (upstream's own CPU-vs-GPU checks) for a clean, independent verdict on the NTT rather
   than a hand-written probe. This needs the examples' CMakeLists taught about HIP, which
   the current patches do not cover.
