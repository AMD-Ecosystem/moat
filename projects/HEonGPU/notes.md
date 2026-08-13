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
port divergence, and they are out of scope for this port.

**Out of scope is not "nothing to see".** The 2026-08-11 review established that
those sites are wrong on BOTH platforms, because the host implementation of the
same quantity (`double_to_torus32`, `src/lib/host/tfhe/encryptor.cu:102-108`)
routes through `std::llround` and WRAPS while the device sites saturate: every
negative noise sample in the key-switching and bootstrapping keys becomes exactly
zero. `TFHE_Gate_Boots` passes BECAUSE of that, so no test here can catch it.
Registered as `heongpu-tfhe-torus32-saturates` in `projects/HEonGPU/deferred.json`
-- read that before deciding this is settled.

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

**WRONG, corrected by the 2026-08-11 review; no fix is needed and none was made.**
`RNGonGPU/src/lib/common/aes.cu` uses `int warpThreadIndex = threadIdx.x & 31;` at three
sites (174, 381, 557). This attempt read it as a warp-size assumption. It is not one:
`aes.cu:180-186` fills all 32 columns of `__shared__ Data32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE]`
with the same value, so `& 31` selects a bank replica of an identical table, never a lane.
It is correct at any wavefront width and costs at most bank conflicts on wave64.

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

## Porter Attempt 5 (2026-08-11, linux-gfx90a) -- 20/20, all suites pass

Fork head after this attempt: `39d678d24f2e899da7e9a78e170dac4c414cf5a4`.

### ROOT CAUSE of the eight BFV failures: 128-bit shift by 64 in Barrett

`modular_operation_gpu::BarrettOperations::mult` (GPU-NTT
`common/modular_arith.cuh`) shifts its hand-rolled `uint128_t` right by
`modulus.bit + 3`. The shift operator expands that to `lo >> shift`,
`hi << (64 - shift)` and `hi >> shift` with no guard, so at `modulus.bit >= 61`
the count reaches 64 and every limb shift is undefined. PTX yields 0 for any
count >= 64; AMDGPU keeps only the low 6 bits, so a shift of 64 acts as a shift
of 0 and the high limb is OR-ed into the low one instead of replacing it. NVIDIA
gets the right answer by accident of the PTX semantics.

Measured against a `__uint128_t` host reference, 20000 random pairs plus the
boundary values, per modulus width:

| modulus | mult | reduce_forced | add | sub |
| --- | --- | --- | --- | --- |
| 20, 36, 37, 60 bit | 0 bad | 0 | 0 | 0 |
| 61 bit (BFV gamma) | 19999/20000 bad | 0 | 0 | 0 |
| 62 bit | 19999/20000 bad | 0 | 0 | 0 |

That is why CKKS passed and BFV did not: the CKKS test moduli are all 37 bits or
fewer, while BFV decryption's gamma correction term is a generated 61-bit prime
(2305843009213554689 for the 4096/{36,36},{37} parameter set). Fixed by handling
`shift == 0`, `shift < 64` and `shift < 128` explicitly. Behaviour-preserving on
NVIDIA, so no `#ifdef`. Carried in `thirdparty/patches/GPU-NTT.patch`.

Note attempt 4's device-Barrett audit swept 20/36/37/59/60-bit moduli and found
nothing; the fault begins at 61. Sweep the WIDTH, not only the values.

### ROOT CAUSE of the TFHE failure: a 32x over-indexed state buffer

`HEEncryptor<Scheme::TFHE>` allocated `context_->n_` (512) `curandState`
entries, then initialized and used `total_state = 512 * 32` of them --
`initialize_random_states_kernel` and `encrypt_lwe_kernel` both index by a
global thread id over a 32-block, 512-thread launch. A 32x heap overrun that
NVIDIA absorbs. LWE encrypt followed by decrypt with no bootstrapping went from
35/64 bits wrong to 0/64 once the allocation matched the launch.

The two suspects the attempt-4 handoff named for TFHE (the `warp_reduce` shared
memory byte count, and a hipRAND state size differing from cuRAND's) were both
innocent: `(THREADS/32 + 1) * sizeof(uint32_t)` over-allocates at wave64 rather
than under-allocating, and the state size never mattered because the count was
wrong by 32x either way.

### BFV_..._Addition_Subtraction was flaky for a reason of its own

Not a port fault. `test_bfv_addition.cpp` reduced its expected value with
`(a + b > t) ? a + b - t : a + b`, so a coefficient pair summing to exactly `t`
was expected to decode as `t` rather than 0. Over 40 encrypt-add-decrypt rounds
of 4096 coefficients, mismatches against `(m1 + m2) mod t` were 0 and against the
test's own reference 1 (957528 + 74665 = 1032193). Changed to `>=`. Roughly one
run in eighty hit it, which reads exactly like a flaky GPU failure -- if a single
suite fails intermittently here, check the test's reference before the kernel.

### The installed package was unusable on AMD (found by verifying the docs)

Writing the downstream-consumer CMake snippet for `docs/advanced_topics.rst` and
then actually building against an installed tree exposed four separate breaks in
the install/export path, all invisible to the in-tree tests. Fixed together in
`39d678d`; the general form is in the `cuda-to-rocm` skill under Strategy A.
`cmake --install --prefix` does not override this project's baked-in prefix --
reconfigure with `-DCMAKE_INSTALL_PREFIX`.

### Test result

`ctest --test-dir build` on gfx90a / ROCm 7.2.1: **20/20 passed**, three
consecutive runs, plus a fourth after the install/export commit.

### Probes (gitignored scratch, `agent_space/HEonGPU-a5/`)

`build.sh <name>` compiles one against the already-built static libraries in
`build/` -- seconds, versus a CMake target. Flags were lifted from
`build/test/CMakeFiles/bfv_encoding_testcases.dir/{flags.make,link.txt}`; `-x hip`
applies to every following input, so the archives go in a separate link step.

1. `bfv.cu` -- encode/encrypt/decrypt/decode over several message patterns,
   comparing the raw plaintext buffers at each hop as well as the decoded values.
2. `dec.cu` -- **the decisive one.** Overwrites a valid ciphertext with a trivial
   `(Delta*m, 0)` built on the host with `__uint128_t`, then decrypts. It isolates
   the decryption path from encryption entirely and needs only the public API.
3. `consts.cu` -- dumps the BFV host constants (`Qi_t`, `Qi_gamma`, `Qi_inverse`,
   `mulq_inv_t`, `mulq_inv_gamma`, `inv_gamma`, gamma itself) and checks each
   against an independent computation, then replicates `decryption_kernel` on the
   host. All constants were correct and the host replica gave 0 mismatches, which
   is what pointed at the device operators. Reaches the private members with
   `#define private public` around the `heongpu.hpp` include -- no library edit,
   no rebuild.
4. `mod61.cu` -- device Barrett `mult`/`reduce_forced`/`add`/`sub` against a
   `__uint128_t` reference, swept across modulus widths. This is the probe that
   found it.
5. `tfhe.cu` -- LWE encrypt/decrypt with no bootstrapping, then NOT, then a
   bootstrapped NAND, so a failure lands on one of the three.
6. `add.cu` -- 40 rounds of encrypt-add-decrypt compared against both `(m1+m2) mod t`
   and the test's own reference, which is how the flake was attributed.

## Review 2026-08-11

Port review of `moat-port` (`39d678d`) against the fork's `main` (`1928a14`), on
linux-gfx90a. Verdict: **changes-requested**. The three crypto fixes (Gaussian
cast, 128-bit Barrett shift, TFHE state buffer) are each correct and correctly
scoped; the test-reference change is right; the install/export work is
AMD-relevant and does not touch the CUDA path. What follows is what has to
change. Problems only.

Verified independently this round and NOT re-raised below: the twelve Box-Muller
and six curand-backed `truncate_signed` sites are the complete set of
float-to-unsigned conversions in RNGonGPU (the remaining `static_cast` hits are
integer-to-integer or write float-typed outputs); `T` instantiates only as
`uint32_t`/`uint64_t`, both of rank >= int, so the modular wraparound the fix
relies on is well defined; the three patches reverse-apply cleanly against the
pinned upstream SHAs and `git -C <sub> diff` regenerates each patch byte for
byte; `>= plain_modulus` is the right reference, since the test draws messages
from `[0, t-1]` so a pair can sum to exactly `t` and BFV decodes that as 0
(`test/test_bfv_addition.cpp:39,55`); `total_state` matches the launch, because
`encrypt_lwe_kernel` indexes `states[blockIdx.x * blockDim.x + threadIdx.x]`
over a 32-block, 512-thread launch (`src/lib/kernel/encryption.cu:287`,
`src/lib/host/tfhe/encryptor.cu:76-84`), so growing the allocation is the only
possible direction; `(THREADS / 32 + 1)` shared entries cover 16 warps at wave32
and 8 at wave64 at all three TFHE launch sites; `jargon.py --port HEonGPU` is
clean over the whole branch; commit titles, trailers and ASCII are clean.

### 1. "NVIDIA behaviour is unchanged" is false above a 61-bit modulus

`thirdparty/patches/GPU-NTT.patch` (the `shift < 128` arms of `operator>>` and
`operator<<`), the body of `f30493c`, and the new skill section all assert that
the guarded shifts reproduce what PTX already computed. That holds for counts 0
through 64 and I confirmed it arm by arm: PTX clamps a shift count to the
register width, so at `shift == 64` the original `(value.x >> 64) | (value.y <<
0)` already yields `value.y` in the low limb and 0 in the high, which is exactly
the new branch, and 1..63 and 0 are unchanged. It does NOT hold at `shift >= 65`.
There `64 - shift` underflows to `0xFFFFFFFF`, PTX clamps both shifts to the
width and yields zero for the whole result, while the new branch yields
`value.y >> (shift - 64)`. `mult`/`reduce` shift by `modulus.bit + 3`, so this is
reached at a 62-bit modulus, and the class documents Data64 as working to 62 bits
(`thirdparty/GPU-NTT/src/include/gpuntt/common/modular_arith.cuh:178-179`). The
porter's own sweep records 62-bit as broken before the fix and clean after; that
result is a CUDA fix as well, not a no-op.

Reword the commit body to scope the equivalence ("unchanged for counts up to 64,
i.e. moduli of 61 bits and below; at 62 bits PTX's clamp returned zero and this
also corrects the CUDA result"). A maintainer of a homomorphic-encryption library
will act differently on "your CUDA numbers do not move" than on "your CUDA
numbers move at 62-bit moduli, in the right direction". Register it as a deferral
next to `heongpu-negative-gaussian-cast` so it reaches him.

### 2. The promoted shift lesson carries the same wrong claim, and that one ships to every port

`.claude/skills/cuda-to-rocm/references/fault-classes.md`, "Shift counts of 0 or
>= 64 in hand-rolled 128-bit arithmetic": "Fix by branching on `shift == 0`,
`shift < 64` and `shift < 128` explicitly; that reproduces the PTX result, so it
is behaviour-preserving on NVIDIA and needs no `#ifdef`." The same paragraph
already says PTX yields 0 for any count >= 64, which contradicts it above 64.
"NVIDIA therefore computes the mathematically right answer by accident" is true
at exactly 64 and false at 65 and above. Merging the branch publishes this to
every future port, and a porter who follows it will tell a maintainer his CUDA
path is untouched when it is not. Fix the scope in the entry; the rest of that
section (the AMDGPU low-6-bits masking, the sweep-the-width advice, the
Barrett-at-61-bits worked example) checks out and should stay.

The other four promoted entries were checked against their sources and are
accurate: the allocation-versus-launch-geometry entry, the negative-float-to-
unsigned entry (including the claim that the 32-bit conversion does not diverge
-- the measured `(uint64_t)(-2.28) == 0xfffffffe` is exactly what the AMDGPU
fptoui-f64-to-i64 expansion produces when its hi `v_cvt_u32_f64` clamps and the
residue comes out as `2^32 - 2`, which is also why the single-instruction 32-bit
conversion clamps), the PTX carry-chain predicates (both the `carry_in=1,
b=2^64-1` and the `a == b && borrow` edges are right), the install-and-consume
entry (including that quoted `if("@USE_HIP@")` is false when empty and a bare
`if()` is an error), and the submodule-patch entry.

### 3. The TFHE torus conversion is the same defect and the note closes it too early

`src/lib/kernel/keygeneration.cu:1142` (`tfhe_generate_switchkey_kernel`) and
`:1202-1203` (`tfhe_generate_bootkey_random_numbers_kernel`) compute
`static_cast<uint32_t>(floor(x + 0.5))` where `x = frac * 2^32` and `frac` is in
`(-1, 1)`, so the argument reaches `-2^32`. Notes lines 392-395 justify leaving
them by "the same latent upstream issue but not a port divergence", which is
correct as far as it goes -- both back ends saturate a negative source to zero
here, so the port introduces nothing. What the note misses is the evidence that
these sites are wrong on both platforms: the host implementation of exactly this
quantity, `HEEncryptor<Scheme::TFHE>::double_to_torus32`
(`src/lib/host/tfhe/encryptor.cu:102-108`), routes through `std::llround` and so
WRAPS, while the device sites saturate. Same computation, two different answers
for a negative noise sample; every negative sample in the key-switching and
bootstrapping-key noise becomes exactly zero.

Leaving the code alone is a defensible scope call for this port and I am not
asking for a code change. Recording it is not optional: `TFHE_Gate_Boots` passes
BECAUSE zeroing half the noise makes a gate more likely to decode, so no test in
this repository can ever catch it, and in an FHE library a noise distribution
missing its negative half is a security fault, not a numerical curiosity. Add it
to `projects/HEonGPU/deferred.json` with the host-versus-device evidence, and
amend the attempt-4 paragraph so the next reader does not inherit "leave them
alone" as "nothing to see".

### 4. The examples were half-converted and cannot build with USE_HIP=ON

`example/basic/CMakeLists.txt:28-33`, `example/bootstrapping/CMakeLists.txt:18-23`
and `example/mpc/CMakeLists.txt:17-22` each grew a `USE_HIP` branch that links
`hip::host` but omits all three things `test/CMakeLists.txt:44-52` does and
explains it must: `set_source_files_properties(... LANGUAGE HIP)`, the `USE_HIP`
compile definition, and `thirdparty/hip_compat` on the include path. Reproduced
by compiling `example/basic/1_basic_bfv.cpp` the way those targets would:

```
g++ -std=gnu++17 -c example/basic/1_basic_bfv.cpp -Isrc/include \
    -Ithirdparty/{GPU-NTT,GPU-FFT,RNGonGPU}/src/include \
    -Ithirdparty/rmm_hip_stub/include -D__HIP_PLATFORM_AMD__=1
-> gpuntt/common/common.cuh:9: fatal error: cuda_runtime.h: No such file
```

and after adding `-isystem thirdparty/hip_compat`:

```
-> thrust/system/cuda/config.h:40: fatal error:
   cub/detail/detect_cuda_runtime.cuh: No such file
```

which is precisely the rocThrust-needs-a-HIP-compile reason the test file cites.
`benchmark/CMakeLists.txt:16` was not touched at all and links `CUDA::cudart`
unconditionally, so `-DHEonGPU_BUILD_BENCHMARKS=ON -DUSE_HIP=ON` fails at
configure time. Both default OFF, which is why this was never hit.

Give examples and benchmarks the same treatment the tests got and build each
once with the option on, or revert the example CMake edits entirely. A branch
that fails the moment a documented option is enabled is worse than one with no
branch, because it reads as tested.

### 5. Dead code in the compat header and the root CMake

`src/include/heongpu/cuda_to_hip.h:72-87` and `:95-100` define `kWarpSize` and
`FULL_WARP_MASK` on both paths. Neither is referenced anywhere in the tree
outside the header; every warp fix that landed uses runtime `warpSize`. The CUDA
arm is also `#if defined(__CUDA_ARCH__)` / `#else` with the identical
`kWarpSize = 32` in both branches. Delete both symbols and the dead conditional.

`CMakeLists.txt:57-58` sets `HIP_RUNTIME_LIB` and `HIP_INCLUDE_DIRS` as cache
variables that nothing reads. Remove them.

### 6. Two comments describe something other than the code beneath them

`src/include/heongpu/util/util.cuh:315` says "HIP requires a 64-bit mask for
`__shfl_down_sync`" directly above line 316, which calls maskless `__shfl_down`
and passes no mask at all. Say what the code does: HIP's `__shfl_down` defaults
to the wavefront width, which is why no mask constant is needed on either width.

`src/lib/kernel/small_ntt.cu:8` says the file "is kept for CUDA builds that may
still use explicit instantiation". The explicit instantiations were deleted in
the same commit; the file is now an empty translation unit on both platforms and
is still listed at `src/CMakeLists.txt:104`. Drop the sentence, or drop the file
from `HEONGPU_KERNEL_SOURCES`.

### 7. The memory-manager stub narrows an installed public API

`thirdparty/rmm_hip_stub/include/rmm/device_uvector.hpp:84-85` deletes the copy
constructor and offers no `(const device_uvector&, stream, mr)` overload, but the
installed public header `src/include/heongpu/util/devicevector.cuh:31-37`
declares `DeviceVector(const DeviceVector&, stream, Source)` forwarding to
exactly that. Nothing in the library instantiates it, which is why the build is
green; a consumer that copy-constructs a `DeviceVector` compiles on CUDA and
fails on the HIP build. Real RMM provides that constructor. Add it to the stub.

### 8. An attempt-4 claim in these notes is wrong and will be inherited

Notes line 533 records `RNGonGPU/src/lib/common/aes.cu`'s `int warpThreadIndex =
threadIdx.x & 31` as "a genuine warp-size assumption" needing an arch-unified fix
on wave64. It is not one. `aes.cu:180-186` fills all 32 columns of
`__shared__ Data32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE]` with the same value,
so `& 31` selects a bank replica of an identical table, never a lane; it is
correct at any wavefront width and costs at most bank conflicts on wave64. This
branch has already lost two sessions to an unreconciled claim in this file
(attempt 3's inverse-NTT verdict); correct this one in place rather than leaving
a third.

## Porter Attempt 6 (2026-08-11, linux-gfx90a) -- review findings addressed

Fork head after this attempt: `4b0d53dc9811826a6666fafc03c458fccddaf68e`
(five commits on top of `39d678d`, none of them an amend). 20/20 suites still
pass, three runs, and the examples and benchmarks now build and run too.

### The examples and benchmarks were never built with the option on

The review's finding 4 reproduced exactly. `HEonGPU_BUILD_EXAMPLES=ON` with
`USE_HIP=ON` failed to compile (the targets linked `hip::host` but were still
plain C++, so `cuda_runtime.h` then rocThrust went missing) and
`HEonGPU_BUILD_BENCHMARKS=ON` failed at configure (`CUDA::cudart` linked
unconditionally). Both now get what `test/CMakeLists.txt` gets: `LANGUAGE HIP`,
the `USE_HIP` definition, `thirdparty/hip_compat` on the include path.

Two extra steps the tests did not need:

- `OpenMP::OpenMP_CXX` guards its compile options with `$<COMPILE_LANGUAGE:CXX>`,
  so a source switched to `LANGUAGE HIP` gets no `-fopenmp` and its `#pragma omp`
  is ignored. `${OpenMP_CXX_FLAGS}` has to go on both `$<COMPILE_LANGUAGE:HIP>`
  and `$<LINK_LANGUAGE:HIP>`. ROCm's clang finds its own `libomp.so`; no explicit
  library is needed. (CORRECTED in attempt 7: this attempt left the imported
  target on the HIP branch as well, which is a defect, and the account of the
  symptom here was wrong in two ways -- see attempt 7 for what actually happens.)
- The old `LINKER_LANGUAGE CXX` on the example targets had to go, since the link
  is a HIP link now. (CORRECTED in attempt 7: the reason is that
  `$<LINK_LANGUAGE:HIP>` is false under a CXX link, not that device objects would
  be stranded -- they are not.)

The benchmark targets link no OpenMP: `benchmark_bfv.cpp` includes `<omp.h>` but
uses nothing from it, so the AMD branch mirrors the CUDA one exactly.

Verified by running, not only building: `1_basic_bfv`, `9_multi_stream_usage_way1`
(the OpenMP one), `15_basic_tfhe`, `1_multiparty_computation_bfv` and
`tfhe_benchmark` all produce correct output on gfx90a. Promoted to the
`cuda-to-rocm` skill (`references/strategy-a-cmake.md`) as a general rule --
turn every documented option ON once and build it.

### The Barrett shift claim was too broad, in both the branch and the skill

The guard is a CUDA fix as well, not a no-op. PTX clamps a shift count to the
register width, so the unguarded form is right at exactly 64 and returns zero for
the whole 128-bit value at 65 and above, where the answer is
`value.y >> (shift - 64)`. `mult`/`reduce` shift by `modulus.bit + 3`, so 65 is
reached at a 62-bit modulus and `BarrettOperations` documents 62 bits as
supported for Data64. The submodule patch's code comment now says this, a new
commit corrects the record without amending `f30493c`, and the skill entry
(`references/fault-classes.md`) has had the "behaviour-preserving on NVIDIA"
claim scoped. Registered as `heongpu-barrett-shift-cuda-side` in
`projects/HEonGPU/deferred.json` so the maintainer conversation is not lost.

### The memory-manager stub narrowed an installed public API

`DeviceVector(const DeviceVector&, stream, Source)` in the installed headers
forwards to an RMM constructor the stub did not have. A consumer translation unit
copy-constructing a `DeviceVector<Data64>` failed with "no matching constructor
for initialization of `rmm::device_uvector<unsigned long>`" at
`devicevector.cuh:35`, and now compiles, copies and reads back correctly. The
stream is deliberately NOT defaulted in the added constructor: with a default it
becomes ambiguous against the deleted copy constructor, which is also why real
RMM leaves it required.

### Also done

Dead `kWarpSize`/`FULL_WARP_MASK` and two unread CMake cache variables removed;
the `warp_reduce` shuffle comment and the `small_ntt.cu` comment now describe the
code (the file was an empty translation unit and is gone, its device functions
being inline in `small_ntt.cuh` for both back ends); the TFHE torus-conversion
asymmetry registered as `heongpu-tfhe-torus32-saturates`; the attempt-4 claims
about the TFHE sites and about `aes.cu`'s `& 31` corrected in place above.

### How the submodule patch was regenerated

Only the shift comment changed, but the procedure is worth having written down:

```bash
# edit thirdparty/GPU-NTT/... in the checkout, then
git -C thirdparty/GPU-NTT diff > thirdparty/patches/GPU-NTT.patch
git -C thirdparty/GPU-NTT checkout -- .          # back to pristine
git -C thirdparty/GPU-NTT apply --check "$PWD/thirdparty/patches/GPU-NTT.patch"
bash thirdparty/build.sh ON && bash thirdparty/build.sh ON   # idempotent
diff <(git -C thirdparty/GPU-NTT diff) thirdparty/patches/GPU-NTT.patch
```

`git apply` resolves a relative patch path against the `-C` directory, so pass an
absolute one.

## Review 2026-08-11 (round 2)

Second review of `moat-port`, focused on the delta `39d678d..4b0d53d` but re-checking
what it touches. Verdict: **changes-requested**, on two findings, both about the same
piece of the examples' OpenMP handling: the CMake is subtly wrong, and the lesson
promoted from it states three mechanisms that do not survive a test. Everything else
from the 2026-08-11 review is properly addressed. Problems only.

Verified this round and NOT re-raised. The corrected Barrett-shift text is right across
the whole reachable count range: 0 and 1..63 unchanged, exactly 64 already correct under
the PTX clamp (`value.x >> 64` is 0 and `value.y << 0` is `value.y`, which is what the
new `shift < 64`/`shift < 128` split computes), 65 and above previously zero under the
clamp and now `value.y >> (shift - 64)`; `mult`/`reduce` shift by `modulus.bit + 3`
(`modular_arith.cuh:384-386,414-416`) and the class documents 62 bits as the Data64 limit
(`:178-179`), so 65 is reachable and the CUDA-side claim is correct. A count of 128 or
more falls through to the default-constructed `uint128_t`, whose constructor zeroes both
limbs (`:188-192`), which matches what the clamp used to return. The `fault-classes.md`
entry now scopes the equivalence to 0..64 and no longer contradicts itself. The rmm stub
constructor matches `DeviceVector`'s forwarding call
(`devicevector.cuh:31-36` passes base-converted `other`, `cudaStream_t`,
`rmm::mr::device_memory_resource*`), is unambiguous against the deleted copy constructor
because the stream has no default, and initializes in declaration order. `kWarpSize`,
`FULL_WARP_MASK`, `HIP_RUNTIME_LIB` and `HIP_INCLUDE_DIRS` have no remaining reference
anywhere in the tree including the submodule checkouts, and `small_ntt.cu` was an empty
translation unit whose device functions are unconditionally `__device__ inline` templates
in `small_ntt.cuh:14,82` (only `bootstrapping.cuh:12` and `keygeneration.cuh:14` include
it), so dropping it from `HEONGPU_KERNEL_SOURCES` is right on both back ends. The `& 31`
correction is itself correct: `aes.cu:180-186` writes the same `t0G[threadIdx.x]` into
all 32 columns of `t0S`, and every one of the 28 uses of `warpThreadIndex`
(`aes.cu:236-330`) is the second (bank) subscript, never a lane. Benchmarks include
`<omp.h>` but reference no OpenMP symbol, and only `9_multi_stream_usage_way1.cpp` uses
one, so the benchmark branch omitting OpenMP is right. All three submodule patches still
regenerate byte for byte from their checkouts against the pinned upstream SHAs
(`8a4daf1`, `b743607`, `d9aaa6b`), so the reverse-apply idempotence holds. Both new
deferrals state their defect accurately and neither closes something this port should
have fixed. `jargon.py --port HEonGPU` is clean over the whole branch; commit titles,
trailers, ASCII and the absence of added copyright lines are clean; the CUDA branches of
all four CMakeLists are byte-identical to their pre-delta form.

### 1. Every example binary links two OpenMP runtimes

`example/basic/CMakeLists.txt:32`, `example/bootstrapping/CMakeLists.txt:22` and
`example/mpc/CMakeLists.txt:21` keep `OpenMP::OpenMP_CXX` on the HIP branch while lines
37-38 (and 27-28, 26-27) also put `${OpenMP_CXX_FLAGS}` on the HIP compile and link
lines. On a GNU C++ toolchain that imported target contributes nothing else to a HIP
target: `FindOpenMP.cmake:676-692` (checked in both CMake 3.28 and the 4.0 actually used
here) guards `INTERFACE_COMPILE_OPTIONS` with `$<COMPILE_LANGUAGE:CXX>`, sets
`INTERFACE_LINK_OPTIONS` only for Fujitsu and IntelLLVM, and sets
`INTERFACE_LINK_LIBRARIES` to `OpenMP_CXX_LIBRARIES` unguarded -- that is, to GCC's
`libgomp`. The generated link line ends with
`/usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so` while the objects, compiled by clang with
`-fopenmp`, need LLVM's `__kmpc_*`, so `readelf -d` on
`build/bin/examples/basic/9_multi_stream_usage_way1` shows both `libgomp.so.1` and
`libomp.so` in DT_NEEDED.

It works here only by accident of loader search order: DT_RUNPATH lists
`/opt/rocm-7.2.1/lib/llvm/lib`, where `libgomp.so.1` is a symlink to `libomp.so`, so both
names land on one runtime. Put GNU's `libgomp` earlier and both load -- `ldd` under
`LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu` reports `libgomp.so.1 => /usr/lib/...` and
`libomp.so => /opt/rocm-7.2.1/...` together. The consequence, reproduced with a minimal
HIP+OpenMP binary linked exactly the same way, is that `libomp` forks the team while
`omp_get_thread_num` resolves out of `libgomp`: "thread ids: 0 1 2 3" normally, "thread
ids: 0 0 0 0" with `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1`. In
`9_multi_stream_usage_way1` every worker would then take `s[0]` and the multi-stream
example silently stops being multi-stream.

Drop `OpenMP::OpenMP_CXX` from the three `if(USE_HIP)` branches; the explicit flags on
`$<COMPILE_LANGUAGE:HIP>` and `$<LINK_LANGUAGE:HIP>` already give clang its own runtime,
`libheongpu.a` references no OpenMP symbol at all (checked with `nm` over the archive),
and the CUDA branch, where `libgomp` is the correct runtime for an nvcc/g++ build, is
untouched. Write the flags the way `FindOpenMP` itself does,
`$<$<COMPILE_LANGUAGE:HIP>:SHELL:${OpenMP_CXX_FLAGS}>`, so a compiler whose OpenMP flag
is more than one token is not passed as a single argument.

### 2. The promoted CMake lesson gets the mechanism wrong three times, and the truth is worse

`.claude/skills/cuda-to-rocm/references/strategy-a-cmake.md`, the "Two things bite when a
target that was plain C++ becomes a HIP target" paragraph. Merging this branch publishes
it to every port, and each of its three factual claims fails a direct test.

"`OpenMP::OpenMP_CXX` ... [guards] its link options with `$<LINK_LANGUAGE:CXX>`" -- it
sets no link options at all on a GNU or Clang toolchain (`FindOpenMP.cmake:679-683`
restricts that to Fujitsu and IntelLLVM). What it actually contributes is the CXX
compiler's OpenMP library, unguarded, which is finding 1 above and the thing the entry
should be warning about.

"compiles the `#pragma omp` away silently and then fails to link with undefined
`__kmpc_*`" -- those two cannot both happen. Compiled without `-fopenmp` there are no
`__kmpc_*` references to be undefined; verified by compiling a HIP source with
`clang++ -x hip` and no OpenMP flag, linking it against `libgomp`, and getting a clean
link that prints "thread ids: 0 0 0 0". The undefined `__kmpc_*` link error belongs to
the intermediate state where the compile flag was added and the link flag was not. State
it correctly, because the default symptom is worse than a link error: a port that forgets
this ships examples whose parallel regions were silently compiled away, and a porter who
was told to watch for a link failure will not find one.

"drop any `LINKER_LANGUAGE CXX` ... or the HIP device objects never reach the link" --
not the mechanism. A non-RDC HIP object carries its own device image, and a plain `g++`
link of one produces a working binary: `clang++ -x hip --offload-arch=gfx90a -fPIE -c`
then `g++ k.o -lamdhip64` runs the kernel and returns the right values. The real reason
`LINKER_LANGUAGE CXX` had to go is narrower and worth naming: under a CXX link
`$<LINK_LANGUAGE:HIP>` is false, so the OpenMP link flag never applies, and `g++` plus
GNU `libgomp` cannot resolve the `__kmpc_*` the HIP-compiled object needs.

Correct all three in the skill entry. The same "otherwise the OpenMP example fails to
link" account is in the attempt-6 notes above and in `0da0c07`'s body; fix the notes in
place, and when the finding-1 change lands, let its commit body carry the accurate
version rather than amending the earlier commit.

## Porter Attempt 7 (2026-08-11, linux-gfx90a) -- round-2 findings addressed

Fork head after this attempt: `5d99b8f447895f5b34b35f856e654d65e69b390a`
(one commit on top of `4b0d53d`, not an amend). 20/20 suites pass, four runs.

### Finding 1: every example binary linked two OpenMP runtimes

Reproduced before touching anything: `readelf -d` on
`bin/examples/basic/9_multi_stream_usage_way1` listed both `libgomp.so.1` and
`libomp.so`, and the generated `link.txt` ended with
`/usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so` (the cache has
`OpenMP_CXX_LIB_NAMES=gomp;pthread`). `OpenMP::OpenMP_CXX` is gone from the three
`if(USE_HIP)` branches; the explicit `${OpenMP_CXX_FLAGS}` stays, now in the
`SHELL:` form. After the rebuild no executable under `bin/` links `libgomp` at
all (scanned all of them), and `bin/examples/basic/9_multi_stream_usage_way1`,
`1_basic_bfv`, `15_basic_tfhe`, `bootstrapping/3_ckks_bit_bootstrapping` and
`mpc/1_multiparty_computation_bfv` all run correctly.

`SHELL:` inside the genex behaves: `flags.make` shows a single `-fopenmp` and
`link.txt` one `-fopenmp`, with no literal `SHELL:` token.

### What the probes actually showed (all four measured this attempt)

Probe source in gitignored `agent_space/HEonGPU-a7/`, one HIP TU with a kernel
and a four-iteration `#pragma omp parallel for`, compiled
`hipcc -x hip --offload-arch=gfx90a -fPIE [-fopenmp] -c`.

- clang link, `-fopenmp`, no GNU library: DT_NEEDED has `libomp.so` only,
  `omp ids: 0 1 2 3`.
- same plus `/usr/lib/gcc/.../libgomp.so` on the link line (what the imported
  target added): both in DT_NEEDED, correct by luck; under
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1` it prints
  `omp ids: 0 0 0 0` with **four distinct `SYS_gettid` values**. The team really
  is forked (only LLVM's runtime can serve the `__kmpc_*` the object calls);
  `omp_get_thread_num` is exported by both and answers from `libgomp`, which
  knows nothing about that team. So the failure is not "serial" -- it is four
  threads all selecting resource 0, which for `9_multi_stream_usage_way1` means
  one shared stream with correct-looking output.
- object compiled WITHOUT `-fopenmp`, linked `g++ probe.o -lamdhip64 libgomp.so`:
  links clean, kernel result correct, `thread ids: 0 0 0 0`. This is the silent
  default symptom, and it disproves both halves of the attempt-6 account: no
  undefined `__kmpc_*`, and a plain `g++` link of a non-RDC HIP object runs the
  kernel fine, so `LINKER_LANGUAGE CXX` never stranded device code.
- object compiled WITH `-fopenmp`, linked by `g++` with no OpenMP library:
  undefined `__kmpc_global_thread_num` / `__kmpc_fork_call` /
  `__kmpc_for_static_init_4`. That is the intermediate state attempt 6 hit and
  then described as if it were the default.

`/opt/rocm-7.2.1/lib/llvm/lib/libgomp.so.1` is a symlink to `libomp.so`, which is
why the two-runtime binary resolves to one runtime under the ROCm RUNPATH.

### Finding 2: the promoted lesson

`references/strategy-a-cmake.md` rewritten as "OpenMP on a target you just
switched to HIP: two runtimes, one binary". The `$<LINK_LANGUAGE:CXX>` claim, the
"compiled away AND undefined `__kmpc_*`" claim and the stranded-device-objects
claim are all gone; each statement in the replacement is one of the measurements
above. The "turn every option ON" rule above it is unchanged. The attempt-6
bullets in these notes are annotated in place rather than rewritten, so the
correction is visible.

## Review 2026-08-11 (round 3)

Third review of `moat-port`, scoped to `4b0d53d..5d99b8f` (three example CMakeLists,
24 insertions) plus the rewritten skill section. Verdict: **review-passed**. Nothing
blocking; one inaccuracy in this file's own account of itself, corrected below rather
than bounced, since it is internal to MOAT and changes no fork content.

Every factual claim the delta and the skill section rest on was re-measured this round on
gfx90a / ROCm 7.2.1 rather than taken from attempt 7, because the skill text publishes to
every port when this branch merges.

The two-runtime fix holds and the mechanism is as stated. `readelf -d` on
`build/bin/examples/basic/9_multi_stream_usage_way1` now lists `libomp.so` alone, and a
scan of all 42 executables under `build/bin` finds no `libgomp` in any DT_NEEDED. The
generated `flags.make` carries exactly one `-fopenmp` (`HIP_FLAGS = ... -fPIE -fopenmp`)
and `link.txt` exactly one, with no literal `SHELL:` token and no `libgomp.so` on the link
line, so the build tree matches the committed source. The CUDA branch of all three files
(`example/basic/CMakeLists.txt:46-54` and the same block in bootstrapping and mpc) is
untouched by the delta, `find_package(OpenMP REQUIRED)` at `CMakeLists.txt:67` is upstream
and unmodified so `${OpenMP_CXX_FLAGS}` cannot silently be empty, and `nm -u` over
`build/src/libheongpu.a` shows no `__kmpc_*`, `GOMP_*` or `omp_*` undefined symbol, which
is the commit body's claim.

Independent reproduction of the four probes, with a fresh HIP TU (kernel plus a
four-iteration `#pragma omp parallel for` recording `omp_get_thread_num` and
`syscall(SYS_gettid)`), in gitignored `agent_space/rev3/`:

- clang link with `-fopenmp`, no GNU library: DT_NEEDED `libomp.so` only, `omp ids: 0 1 2 3`,
  four distinct tids.
- clang link plus `/usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so` (what the imported target
  added): both names in DT_NEEDED, `0 1 2 3` unforced. Under
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1`, and equally under a bare
  `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu` with no preload, it prints `omp ids: 0 0 0 0`
  with **four distinct tids**. So attempt 7's sharpening is right and round 2's "the team is
  serial" reading was not: the team forks and the ids are wrong. The reason is symbol-level:
  `nm -D` finds 0 `__kmpc_*` in GNU `libgomp.so.1` against 492 in ROCm's `libomp.so`, while
  `omp_get_thread_num` is defined in both, so the fork call can only bind to LLVM's runtime
  and the id query binds to whichever loaded first.
- object compiled WITHOUT `-fopenmp`, linked `g++ probe.o libgomp.so -lamdhip64`: clean
  link, kernel result correct, `omp ids: 0 0 0 0` with **one** tid repeated four times --
  genuinely serial, which is the skill's "runs SERIALLY" sentence and is a different state
  from the one above.
- object compiled WITH `-fopenmp`, linked by `g++` with no OpenMP library: undefined
  `__kmpc_global_thread_num`, `__kmpc_fork_call`, `__kmpc_for_static_init_4`,
  `__kmpc_for_static_fini`, `__kmpc_push_num_threads`, `omp_get_thread_num`.

The rest of the skill section checks out. `FindOpenMP.cmake:676-692` in the CMake actually
configured here (4.0, from `CMAKE_ROOT` in the cache; round 2 checked 3.28) guards
`INTERFACE_COMPILE_OPTIONS` with `$<COMPILE_LANGUAGE:${LANG}>`, restricts
`INTERFACE_LINK_OPTIONS` to Fujitsu and IntelLLVM, and sets `INTERFACE_LINK_LIBRARIES`
unguarded, exactly as the entry says; the `SHELL:` spelling the entry recommends is
verbatim what line 678 uses. `/opt/rocm-7.2.1/lib/llvm/lib/libgomp.so.1` is a symlink to
`libomp.so`, and because that file's SONAME is `libomp.so` the loader satisfies both
DT_NEEDED entries with one mapping -- `ldd` on the two-runtime probe shows a single
`libgomp.so.1 => /opt/rocm-7.2.1/lib/llvm/lib/libgomp.so.1` and no second OpenMP object,
which is the entry's "both names resolve to one runtime". The `LINKER_LANGUAGE` paragraph
is right in both halves: a CMake probe project with `LINKER_LANGUAGE CXX` on a
`LANGUAGE HIP` source drops a `$<$<LINK_LANGUAGE:HIP>:...>` link option entirely from
`link.txt` while the same target without it keeps the option, and the plain `g++` link of
the non-RDC HIP object above ran the kernel and returned the right values, so device code
is not what goes missing.

The `SHELL:` form is correct and in scope. It was asked for in round 2 (this file, finding
1, "write the flags the way `FindOpenMP` itself does"), so it is not extra scope; and it is
not merely defensible but load-bearing for the general case: a probe target given a
two-token value with `SHELL:` emits `-Xarch_host -DTWOTOKEN=1` as two arguments, while the
same value without it emits `"-Xarch_host -DTWOTOKEN=1"` quoted as one. For `-fopenmp` it
is a no-op, it changes one word on lines the finding required changing anyway, and the
generated build proves it emits nothing extra. Keep it.

### The record of what attempt 7 did to the attempt-6 account is wrong

`notes.md:1093-1094` states "the attempt-6 bullets in these notes are annotated in place
rather than rewritten, so the correction is visible". The first bullet was rewritten: MOAT
commit `7163b0d` replaced the original text ("`OpenMP::OpenMP_CXX` only decorates the CXX
language, so a source switched to `LANGUAGE HIP` compiles the `#pragma omp` away and then
fails to link with undefined `__kmpc_global_thread_num` / `__kmpc_fork_call`") with the
corrected sentence now at `notes.md:856-859`, and only then appended the annotation. Round
2 asked for exactly that ("fix the notes in place"), so the edit is right and the reader
hazard the annotation guards against is gone -- a porter reading `notes.md:839-876` straight
through now gets the correct account. What is left is that the annotation at `:860-862`
says "the account of the symptom here was wrong in two ways" while "here" no longer holds
any wrong account, so the two ways are unrecoverable from this file. They were: the claim
that the imported target "only decorates the CXX language", which misses its unguarded
`INTERFACE_LINK_LIBRARIES` and so misses the two-runtime defect entirely; and the claim
that the compile-away and the undefined `__kmpc_*` link error happen together, which the
probes above separate into two mutually exclusive states. That is recorded here, so no
further edit is owed.

Nothing else raised. Commit hygiene is clean over the whole branch (all 15 titles carry
`[ROCm]` and none exceeds 61 characters, no `Co-Authored-By`/noreply trailer, no internal
account reference, the only non-ASCII in the branch diff is the upstream author's name in
pre-existing headers), `jargon.py --port HEonGPU` reports clean, the fork tree is clean,
the delta touches no submodule patch and no kernel, and the fault classes are untouched by
a change that adds and removes only build flags.

## Validation 2026-08-11 (linux-gfx90a, MI250X, ROCm 7.2.1) -- completed

Real-GPU validation of `moat-port` at `5d99b8f447895f5b34b35f856e654d65e69b390a`
(the `review-passed` head). GPU: `rocm-smi`/`rocminfo` report three MI250X dies
(GFX Version gfx90a) on this host; `cat /opt/rocm/.info/version` reports 7.2.1.
Clean build from scratch (`rm -rf build` first, tree was otherwise as checked
out, `git -C src status --porcelain` empty before and after):

```bash
cmake -S projects/HEonGPU/src -B projects/HEonGPU/src/build \
    -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release \
    -DHEonGPU_BUILD_TESTS=ON -DHEonGPU_BUILD_EXAMPLES=ON -DHEonGPU_BUILD_BENCHMARKS=ON
cmake --build projects/HEonGPU/src/build -j128
ctest --test-dir projects/HEonGPU/src/build --output-on-failure
```

Both wrapped in `utils/timeit.sh HEonGPU compile -- ...` / `utils/timeit.sh
HEonGPU test -- ...`; configure 176s, build 65s (128 cores), no `error:` in
either log. `ctest`: **20/20 passed**, run twice back to back (13.3s and
13.3s), reproducing the porter's 20/20 claim independently on a from-scratch
build rather than trusting the porter's build directory.

### 1. Barrett 128-bit shift (`thirdparty/patches/GPU-NTT.patch`)

Independent probe (`agent_space/HEonGPU-val/mod61.cu`, gitignored), compiled
against the built `libntt-1.0.a`'s header directly (no library code changed),
sweeping modulus widths 20/36/37/59/60/61/62 bits with a random 64-bit prime
near each width, 20000 random pairs plus three boundary values per width,
compared against a `__uint128_t` host reference:

```
bits=20 q=907367               mult: 0/20000 bad
bits=36 q=50345224429          mult: 0/20000 bad
bits=37 q=86597943119          mult: 0/20000 bad
bits=59 q=460319146180475921   mult: 0/20000 bad
bits=60 q=871991169372668971   mult: 0/20000 bad
bits=61 q=1705080445570786637  mult: 0/20000 bad
bits=62 q=3302937310097410133  mult: 0/20000 bad
```

0 bad at every width, including 61 and 62 bits where attempt 5 measured
19999/20000 bad before the fix. The <= 60-bit paths remain 0 bad, confirming
the fix did not disturb them.

### 2. `truncate_signed<T>()` Gaussian sampler (`thirdparty/patches/RNGonGPU.patch`)

Independent probe (`agent_space/HEonGPU-val/rngprobe.cu`), linked against the
real `libheongpu.a`, calling the public
`heongpu::RandomNumberGenerator::instance().modular_gaussian_random_number_generation`
at a 4096-ring against the same 61-bit BFV-gamma-sized modulus attempt 4 used
(`2305843009213554689`), `std_dev=3.2`:

```
n=4096 requested_std_dev=3.20
mean(centered)=0.1101 std(centered)=2.8044 max|centered|=11 out_of_range(>=q)=517
```

`max|centered|=11` (attempt 4 measured 13 with a different seed; both are
small integers consistent with a std_dev-3.2 Gaussian, not the pre-fix
4294967295). `std(centered)=2.80` is in the right neighborhood of the
requested 3.2 for n=4096 samples. `out_of_range=517` matches attempt 4's
"~518" and is the documented harmless upstream artifact (a sample in (-1,0)
truncating to 0 then storing exactly `q`, congruent to 0). This is a shape
check, not just "decryption succeeded": a distribution collapsed to
`{0, q-2}` as in the pre-fix bug would still often decrypt correctly for
small plaintexts, which is exactly why this needed its own probe per the
dispatch instructions.

### 3. TFHE random-state buffer (`1b5952e`)

Read `src/lib/host/tfhe/encryptor.cu`: allocation `total_state = 512 * 32 =
16384` `curandState` entries; `initialize_random_states_kernel<<<(total_state
+ 511) >> 9, 512>>>` launches exactly `32 * 512 = 16384` threads, one state
per thread, matching the allocation exactly (this is the fixed geometry --
attempt 3/4's bug was allocating only `context_->n_ = 512` states while
indexing up to 16384). `encrypt_lwe_kernel`'s `block_count` is capped at 32,
so its indexing never exceeds the same 16384.

Independent probe (`agent_space/HEonGPU-val/tfhe_lwe.cu`), LWE
encrypt-then-decrypt with no bootstrapping, 40 rounds x 1024 bits (bigger than
attempt 5's 64-bit check, to stress the 32x-launch-vs-1x-allocation geometry
harder):

```
LWE encrypt-then-decrypt (no bootstrap): 0/40960 bits wrong (rounds=40, size=1024)
```

### Examples and benchmarks

Built with `-DHEonGPU_BUILD_EXAMPLES=ON -DHEonGPU_BUILD_BENCHMARKS=ON` in the
same configure above (both compiled clean, no separate step needed). Ran a
sample and checked output by inspection: `1_basic_bfv`, `2_basic_ckks`,
`9_multi_stream_usage_way1`, `15_basic_tfhe`,
`bootstrapping/3_ckks_bit_bootstrapping`, `mpc/1_multiparty_computation_bfv`
all exit 0 with plausible decrypted output (matches the values recorded in
attempt 6/7's own runs); `benchmark/tfhe_benchmark` runs all eight gates and
reports sane per-op timings (NAND/AND/NOR/OR/XNOR/XOR ~16.5-16.6ms, NOT
~0.012ms, MUX ~30ms).

`readelf -d` on every executable under `build/bin` (42 binaries: 15 tests, 24
examples, 3 benchmarks) was scanned for `libgomp`: **none found**. The
9_multi_stream_usage_way1 binary (the one review round 3 focused on) links
`libomp.so` and no `libgomp.so.*`, confirming the single-OpenMP-runtime fix
holds on a from-scratch build, not just the reviewer's build directory.

### CUDA no-regression gate: cuda-not-validated (environmental wall)

Attempted with `/opt/conda/envs/cuda-12.8/bin/nvcc` (12.8.93), host `gcc-13`,
`-DCMAKE_CUDA_ARCHITECTURES=80` pinned explicitly (no `native` autodetection
risk), `-DUSE_HIP=OFF`, `-DTHRUST_INCLUDE_DIR=<conda toolkit's thrust>` (the
project's `FindThrust.cmake` does not search a conda toolkit layout by
default, so this alone is a configuration-path fix, not a source edit, and
was not committed). Configure got past compiler detection and Thrust, then
hit the CUDA path's real dependency: `find_package`/CPM for RMM
(`rapidsai/rmm` branch-25.08, upstream and pre-existing -- this is the same
RMM the plan.md called out as needing a HIP-side stub because the real thing
is CUDA-only) transitively `FetchContent`-clones `rapids-cmake`, `spdlog`, and
`NVIDIA/cccl` from GitHub with full history. The `cccl` clone alone reached
73MB in the ~17 minutes I let it run and was not close to done (that repo's
full-history clone is commonly several hundred MB to >1GB), so this is a
network/time cost intrinsic to the upstream CUDA build's own dependency
fetching, not anything the port touched -- the `USE_HIP` path exists
precisely to avoid this by using the local `rmm_hip_stub/` instead of real
RMM. Killed the build (`pkill`, `rm -rf` the throwaway build dir) rather than
let it run past the ~15-minute budget for this secondary, compile-only gate.
No project file was modified for this attempt; `git -C
projects/HEonGPU/src status --porcelain` was empty before and stayed empty
after (the `-D` flags left no trace). Recording `cuda-not-validated:
upstream CUDA path's RMM dependency requires a full-history clone of
NVIDIA/cccl over the network, which did not complete inside the compile-only
budget; not attempted further`. Not a gate; does not block this arch.

### Deferrals

All three deferred items in `projects/HEonGPU/deferred.json`
(`heongpu-negative-gaussian-cast`, `heongpu-tfhe-torus32-saturates`,
`heongpu-barrett-shift-cuda-side`) were already recorded by the porter/review
rounds before this validation; nothing new to add. `jargon.py --port HEonGPU`
re-run clean. Documentation confirmed present in the project's own house
style: `README.md` ("AMD GPUs (ROCm)" section, `-D USE_HIP=ON` build
instructions), `docs/getting_started.rst` (ROCm prerequisite, HIP configure
line, note that tests/examples/benchmarks all build and run on AMD), and
`docs/advanced_topics.rst` (downstream-consumer CMake snippet for a HIP
build).

### Verdict

`linux-gfx90a`: **completed** at `5d99b8f447895f5b34b35f856e654d65e69b390a`.
Suite pass is reproduced from a clean build; the three fixes named invisible
to the test set were each independently exercised and hold at runtime on
real gfx90a hardware, not just re-read from the porter's notes. CUDA gate is
`cuda-not-validated` (environmental wall, not a regression finding -- the
port was never built as CUDA here, so no comparison against upstream
breakage was possible or necessary).

## Validation 2026-08-11 (linux-gfx1100, Radeon Pro W7800, ROCm 7.2.3) -- completed

Real-GPU validation of `moat-port` at `5d99b8f447895f5b34b35f856e654d65e69b390a`
(the `review-passed` head; this arch's first validation, `validated_sha` was
null beforehand). GPU: `rocminfo` reports four "AMD Radeon Pro W7800 48GB"
(gfx1100, RDNA3/wave32) on this host; `/opt/rocm/.info/version` (via
`rocm-smi`) reports 7.2.3.

The local fork clone at `projects/HEonGPU/src` was still on `043ff00` from an
abandoned 2026-08-08 session; fast-forwarded it to `origin/moat-port` at
`5d99b8f` (`git merge --ff-only`), then reset the three thirdparty submodules
to pristine (`git checkout -- . && git clean -fdx` in each) so
`thirdparty/build.sh ON`, invoked by CMake at configure time, applied the
CURRENT patches from a clean base rather than layering onto an already-dirty
tree left by the old commit's patches.

Clean build from scratch (`rm -rf build` first):

```bash
cmake -S projects/HEonGPU/src -B projects/HEonGPU/src/build \
    -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release \
    -DHEonGPU_BUILD_TESTS=ON -DHEonGPU_BUILD_EXAMPLES=ON -DHEonGPU_BUILD_BENCHMARKS=ON
cmake --build projects/HEonGPU/src/build -j64
ctest --test-dir projects/HEonGPU/src/build --output-on-failure
```

Both wrapped in `utils/timeit.sh HEonGPU compile -- ...` / `utils/timeit.sh
HEonGPU test -- ...`. Build produced `libheongpu.a`, `libntt-1.0.a`,
`libfft-1.0.a`, `librngongpu-1.0.a` and all 15 test executables, no `error:`
in the log (warnings only: `-Wunused-value` on `[[nodiscard]]` HIP error
codes and `-Wpass-failed` loop-unroll notices, same classes as gfx90a).
`ctest`: **20/20 passed**, run twice back to back (11.70s and 11.59s).
`git -C projects/HEonGPU/src status --porcelain` empty before and after (the
three submodules show only the expected patch-applied diffs under
`ignore = dirty`).

### Wave-width coverage note

This is the second arch to run these three crypto fixes on real hardware,
and the first at wave32. Attempt 4 (2026-08-08) had already ruled out
wavefront width as the cause of the pre-fix 18/20 failures by reproducing
the identical failure set on this same gfx1100 card; this run confirms the
converse -- the fixes that resolved gfx90a (wave64) also resolve gfx1100
(wave32), which is expected since none of the three root causes (a
128-bit shift count, a float-to-unsigned cast, an allocation-vs-launch
geometry mismatch) has any wavefront-width dependence, and is worth having
measured rather than assumed.

### Independent re-check: Barrett 128-bit shift, and a probe-methodology pitfall

Reused the gfx90a validation's approach: an independent probe compiled
against the built `libntt-1.0.a`'s header directly (no library code
changed), sweeping modulus widths with random pairs against a
`__uint128_t` host reference. First attempt used moduli of the form
`2^bits - small_offset` (offset up to 1000) and got 19998/20000 bad at 61
and 62 bits -- which looked like a wave32 regression the gfx90a run had
missed. It was not: `Modulus::bit_generator()` computes `bit =
static_cast<T1>(log2(value) + 1)` on the HOST in plain double arithmetic,
and a modulus within about 1024 of an exact power of two (at 61-62 bits,
double's mantissa only resolves increments of 512-1024 at that magnitude)
rounds `log2(value)` up to exactly the power-of-two boundary, so
`bit_generator` returns one more than the true bit length. That
mis-sized `bit` field, not the shift fix, is what produced the wrong
`mu` and wrong Barrett reduction -- confirmed by checking the SAME q's
`mod.bit` field directly (62 for a 61-bit modulus) and by rerunning with
moduli drawn uniformly across each target bit width instead of clustered
against the boundary, which restored `0/20000 bad` at every width:

```
bits=20  q=711789                 bit=20 mu=3089431              mult: 0/20000 bad
bits=36  q=46751029023            bit=36 mu=202021926856          mult: 0/20000 bad
bits=37  q=101323417063           bit=37 mu=372854893350          mult: 0/20000 bad
bits=59  q=500003406430292181     bit=59 mu=1329218940001591548   mult: 0/20000 bad
bits=60  q=817179969723198473     bit=60 mu=3253207481909186472   mult: 0/20000 bad
bits=61  q=1775933377289357583    bit=61 mu=5987738111274164963   mult: 0/20000 bad
bits=62  q=3202365455985161811    bit=62 mu=13282461495960627051  mult: 0/20000 bad
```

This is host-side, standard-library floating point, identical on any
platform and any back end -- not a HIP or wave32 effect, and not present
in the gfx90a validation because that probe (and this project's own
NTT-friendly prime search) never lands a modulus quite that close to a
power of two. Recorded here as a diagnostic method: when a probe for a
fixed defect newly fails on a second arch, check whether the PROBE's own
inputs are pathological before concluding the fix regressed.

### TFHE random-state buffer and the negative-float cast

Not re-probed standalone here (the gfx90a validation already isolated and
measured both independently, and neither has any wavefront-width
dependence by construction: the state-buffer geometry is a scalar
allocation-vs-launch-size count, and the cast is a per-thread scalar
conversion). Both are exercised end-to-end by `TFHE_Gate_Boots` (state
buffer) and by all seven BFV and seven CKKS suites (the Gaussian sampler
runs in every keygen), all of which passed on this arch. `15_basic_tfhe`
(below) additionally cross-checks the TFHE gates against their Boolean
truth tables, which the ctest suite does not do explicitly.

### Examples and benchmarks

Built with `-DHEonGPU_BUILD_EXAMPLES=ON -DHEonGPU_BUILD_BENCHMARKS=ON` in
the same configure. `readelf -d` scanned over all 42 executables under
`build/bin` (15 tests, 24 examples, 3 benchmarks): **no `libgomp` in any
binary**, confirming the round-3-review OpenMP fix holds on a from-scratch
gfx1100 build.

Ran and checked by inspection:
- `1_basic_bfv`: exits 0, decrypted results match the expected squared/
  scaled values shown in the example's own printed check.
- `9_multi_stream_usage_way1` (the OpenMP example): exits 0.
- `15_basic_tfhe`: exits 0; manually verified all eight gate outputs
  against their Boolean truth tables from the printed inputs (NAND, AND,
  NOR, OR, XNOR, XOR, NOT, MUX all correct bit-for-bit).
- `benchmark/tfhe_benchmark`: exits 0, all eight gates report sane
  per-op timings (NAND/AND/NOR/OR/XNOR/XOR ~12.9ms, NOT ~0.015ms, MUX
  ~23.6ms -- same shape as the gfx90a run's ~16.5/0.012/30ms, faster here
  as expected for a different card).

### CUDA no-regression gate

Already recorded at this exact `head_sha` (`5d99b8f`) by the gfx90a
validation above as `cuda-not-validated` (upstream CUDA path's RMM
dependency needs a full-history network clone of `NVIDIA/cccl`). Per the
validator's per-head_sha rule, not re-attempted on this arch.

### Jargon and documentation

`python3 utils/jargon.py --port HEonGPU`: clean (required creating a local
`main` branch tracking `origin/main` in the fork clone alongside the
existing local `moat-port`, since the tool resolves the range by local
branch name; deleted again after the check, no fork content touched).
Documentation confirmed present in `README.md` ("AMD GPUs (ROCm)" section)
and `docs/getting_started.rst` (ROCm prerequisite, HIP configure line,
`USE_HIP=ON`/`CMAKE_HIP_ARCHITECTURES` build instructions) in the checked-
out tree at this head.

### Verdict

`linux-gfx1100`: **completed** at `5d99b8f447895f5b34b35f856e654d65e69b390a`.
First validation on this arch (previously unvalidated); wave32 gate now
satisfied for this head. 20/20 reproduced twice from a clean build; the
Barrett-shift fix re-verified independently (after correcting a
pathological-input artifact in the probe itself, see above); examples and
benchmarks run correctly including a manual truth-table check of the TFHE
gates; no `libgomp` in any of 42 built binaries. CUDA gate already recorded
at this `head_sha`, not re-run.
## CUDA no-regression check 2026-08-12 (linux-gfx90a): TWO REGRESSIONS FOUND

The check deferred as `heongpu-cuda-no-regression-unrun` was run and the port
FAILS to build for NVIDIA. Upstream builds clean with the identical toolchain,
so both faults are ours.

Toolchain: nvcc 12.8.93, gcc-13 host, `-DCMAKE_CUDA_ARCHITECTURES=80`,
`-DUSE_HIP=OFF`, tests/examples/benchmarks ON. Configure succeeded (rc=0), so
the CMake work including the example/benchmark edits is fine for CUDA. The
build fails compiling `src/lib/heongpu.cpp` (a host CXX TU).

Getting past the dependency wall that stopped the 2026-08-11 attempt: read the
CCCL pin out of `rapids-cmake-src/rapids-cmake/cpm/versions.json` (3.0.3, sha
8c04b6539859932f5602e86d38314e4d87f96420), fetched that one commit with
`git fetch --depth 1`, and passed `-DCPM_CCCL_SOURCE=` and
`-DFETCHCONTENT_SOURCE_DIR_CCCL=`. 60MB and a few minutes instead of a
full-history clone at the 62 KB/s this host sustains. Same commit, no version
drift. The other deps (rmm, spdlog, rapids-cmake, rapids_logger, nvtx3,
googletest) clone in reasonable time and were left alone.

### Baseline: upstream compiles, we do not

Built upstream `1928a14` (parent of the first port commit) in
`agent_space/heongpu-upstream`, submodules restored to pristine by
reverse-applying `thirdparty/patches/*.patch`, same compilers, same cached
deps, target `heongpu_host_core`: **rc=0, zero errors**, and it compiles
`lib/heongpu.cpp` as a CXX object exactly as our build does. The only edit to
the baseline tree was replacing `thirdparty/build.sh` (a bare
`git submodule update --init --recursive`) with `exit 0`, because the
submodule content was placed by hand in a detached worktree. That changes no
compiled source.

### Regression 1: the shim pulls curand into every host TU

`src/include/heongpu/cuda_to_hip.h:75` includes `<curand_kernel.h>` on the
CUDA path. `util.cuh:9` includes `cuda_to_hip.h`, and `util.cuh` is reached
from `heongpu.hpp`, so every host `.cpp` gets it. Upstream includes
`<curand_kernel.h>` only from `kernel/*.cuh`, which are reached from `.cu`
TUs that nvcc compiles.

`curand_mtgp32_kernel.h:123` declares `extern const dim3 blockDim;` with C++
linkage; `device_launch_parameters.h:71,73` then declare `threadIdx`/`blockDim`
with C linkage, and g++ rejects the conflict. Minimal repro, g++-13 -std=c++17
against the 12.8 toolkit headers: `cuda_runtime.h` + `device_launch_parameters.h`
gives 0 errors; adding `<curand_kernel.h>` before them gives exactly the two
errors seen in the real build.

Fix direction: the CUDA branch of `cuda_to_hip.h` should include only
`<cuda_runtime.h>`. The include is redundant there, since the kernel headers
that need curand already include it themselves, and on HIP
`thirdparty/hip_compat/curand_kernel.h` satisfies those same includes.
Verified: removing it clears both errors.

### Regression 2: inlined device bodies must be parsed by the host compiler

Removing the curand include leaves four errors, so this one is independent.
Upstream's `small_ntt.cuh` only DECLARES `SmallForwardNTT`/`SmallInverseNTT`
(`__device__ void f(...);`) and the 145-line bodies live in
`src/lib/kernel/small_ntt.cu`, compiled by nvcc. A host CXX TU including the
header therefore sees declarations only.

The port inlined the bodies into the header ("to avoid cross-TU linking issues
on HIP") and `d1b149b` dropped `small_ntt.cu` from `HEONGPU_KERNEL_SOURCES`.
Now `heongpu.hpp:10 -> host/bfv/secretkey.cuh:11 -> kernel/keygeneration.cuh:14
-> small_ntt.cuh` makes g++ parse device bodies, and it cannot see
`__syncthreads` (`small_ntt.cuh:53,79,121,129`).

This is the change round 2 of review cleared as safe. The reading that the
device functions are "unconditionally `__device__ inline` templates" was
correct; what it could not catch is what that does to a host TU on the CUDA
path, because no CUDA build had ever been run. Two reviews and a validation
passed over it.

### Bearing on the port

Both are CUDA-path only and neither can change HIP compiled output, so the
gfx90a validation at `5d99b8f` still describes the AMD build. But the port as
it stands breaks the build it claims not to touch, which is the first thing a
maintainer would hit. Fix before any upstream submission.

Artifacts (gitignored): `agent_space/heongpu-cuda-check.sh` and `.log`,
`agent_space/HEonGPU-cuda-build/`, `agent_space/HEonGPU-upstream-build/`,
`agent_space/heongpu-upstream/`, `agent_space/cccl-3.0.3/`.

## Both CUDA-path regressions fixed 2026-08-12 (linux-gfx90a)

Fixed at `d7d609e` (cuRAND include) and `4ceabb2` (small-NTT guard), on top of
`5d99b8f`. Every claim in the diagnosis above was re-verified before touching
anything; nothing was taken on trust.

### What was verified before the fix

- The cuRAND linkage conflict, standalone: `g++-13 -std=c++17 -fsyntax-only`
  on `cuda_runtime.h` + `device_launch_parameters.h` is clean (rc=0); adding
  `<curand_kernel.h>` in front gives exactly the two conflicting-declaration
  errors from the real build.
- The include is redundant on the CUDA path: `kernel/{addition,bootstrapping,
  decryption,encoding,encryption,keygeneration}.cuh` each include
  `<curand_kernel.h>` themselves, and `util.cuh` is the only file in the tree
  that includes `cuda_to_hip.h`. Removing it restores upstream's include set.
- Why the small-NTT bodies were inlined, which the earlier note asserted
  without evidence: two `.hip` TUs, one defining a `__device__` template with
  an explicit instantiation and one calling it, `hipcc --offload-arch=gfx90a`
  with no `-fgpu-rdc`, fails at link with `lld: error: undefined hidden
  symbol: void addone<int>(int*)`. The reason is real. The HIP branch of
  `heongpu_set_gpu_properties` sets no separable-compilation property, while
  the CUDA branch sets `CUDA_SEPARABLE_COMPILATION ON`.
- Every caller is a device TU: `SmallForwardNTT`/`SmallInverseNTT` are called
  only from `lib/kernel/bootstrapping.cu` and `lib/kernel/keygeneration.cu`.
  `__HIPCC__` and `__HIP__` are both defined by hipcc, with or without an
  explicit `-x hip`; `__CUDACC__` is defined in both of nvcc's passes.

### The fix

`cuda_to_hip.h` CUDA branch keeps `<cuda_runtime.h>` and drops
`<curand_kernel.h>`. `small_ntt.cuh` restores upstream's two declarations
unconditionally and wraps the definitions in
`#if defined(__CUDACC__) || defined(__HIPCC__)`. The rejected alternative was
restoring `lib/kernel/small_ntt.cu` for CUDA: it would hold a second copy of
145 lines of NTT butterfly code that must stay in step with the header's.

### Evidence

- CUDA: `agent_space/heongpu-cuda-check.sh` (unchanged from the run that found
  the regressions), configure rc=0, build rc=0, 64 targets, zero errors in the
  log -- tests, examples and benchmarks. Compile-only; no NVIDIA GPU here, so
  this says nothing about NVIDIA runtime behaviour.
- AMD: full rebuild for gfx90a, `ctest` 20/20 in 13.17s.
- AMD codegen unchanged: the binaries built at `5d99b8f` were kept, and
  `python3 utils/codeobj_diff.py <baseline>/bin src/build/bin` reports
  `verdict=identical` over all 42 binaries (exported symbols and device ISA).
  So gfx90a's validation at `5d99b8f` still describes `4ceabb2`. gfx1100 was
  not measured here; the same argument should hold, but a wave32 host has to
  confirm it rather than inherit this.

### Gotcha for the next person running the CUDA check

`agent_space/heongpu-cuda-check.log` is appended to (`tee -a`), so it holds
every run. Find the last `=== START` before reading errors out of it -- the
failing run's four `__syncthreads` errors are still in the file above the
clean one.

## Review 2026-08-12 (round 4)

Scope: the delta `5d99b8f..4ceabb2` (`d7d609e`, `4ceabb2`) plus the lessons
promoted in MOAT `d4f3532`. Verdict: **changes-requested**. Both code fixes are
correct and their evidence holds; what is wrong is the stated MECHANISM, in two
places that ship -- an upstream-visible source comment and a skill entry every
future port reads.

### 1. The shim comment states a condition the fixed build itself disproves

`src/include/heongpu/cuda_to_hip.h:74-78` says the conflict arises "when a host
compiler sees both" curand_mtgp32_kernel.h's C++-linkage `threadIdx`/`blockDim`
and device_launch_parameters.h's C-linkage ones. In the FIXED tree a host TU
still sees both, and compiles clean. Preprocessing `src/lib/heongpu.cpp` with
the CUDA build's own g++-13 flags puts the C-linkage `threadIdx` at output line
58825 (device_launch_parameters.h, reached from GPU-NTT's
`common/modular_arith.cuh:16`) and the C++-linkage one at 188408
(curand_mtgp32_kernel.h, reached from `kernel/encryption.cuh:12` via
`heongpu.hpp -> host/tfhe/encryptor.cuh:10`). `g++-13 -fsyntax-only` on that TU
is rc=0.

The real condition is ORDER, and only one direction fails. Isolated, against the
same 12.8 headers:

```
curand_kernel.h, then cuda_runtime.h + device_launch_parameters.h  -> the 2 errors
cuda_runtime.h + device_launch_parameters.h, then curand_kernel.h  -> clean
```

Why the shim was the wrong place is then specific and worth saying: the shim is
the FIRST thing every host TU pulls in (`util.cuh:9`), ahead of whatever brings
device_launch_parameters.h in, and its own `#include <cuda_runtime.h>` on the
line below does not protect it because cuda_runtime.h includes
device_launch_parameters.h only under `#if defined(__CUDACC__)`
(cuda_runtime.h:107-119) -- so in a g++ TU the shim's curand landed first with
nothing before it. Rewrite the comment to say that. As written a maintainer can
disprove it with one command, and the next reader could conclude the kernel
headers' own curand includes are equally unsafe (they are not; they are what the
fixed build relies on).

### 2. The promoted fault-class entry repeats the same overbroad claim

`.claude/skills/cuda-to-rocm/references/fault-classes.md:321`: "**cuRAND's
device header is not host-includable at all**". False, and contradicted by the
port it is drawn from -- six `kernel/*.cuh` include `<curand_kernel.h>` and all
six are reachable from `heongpu.hpp`, so every host `.cpp` in the FIXED build
includes it. The entry's own prescription ("leave the RNG include in the kernel
headers that actually use it") is only safe because of the ordering above, which
the headline sentence hides. The reproducer sentence further down is correct
about order; it is the claim it is attached to that is not. Reword the headline
to the order-dependent rule and keep the reason the shim specifically is the
wrong home for it.

### 3. The device-bodies-in-a-header class is not in the SKILL.md index

`SKILL.md:98-102` "Headers, includes and build" gains no line, so a porter
scanning the always-loaded index before making exactly this move ("HIP will not
link my `__device__` function across TUs, I will inline it into the header")
does not see it and never opens fault-classes.md. This branch indexed its
earlier negative-float-cast lesson (`SKILL.md:96`), so this is inconsistent with
its own practice as well as with the point of promoting it. Add one line.

### 4. The pinned-sha recipe's CPM variable is case-sensitive and the entry does not say so

`references/validation.md:13` tells the reader to set both `-DCPM_<DEP>_SOURCE=`
and `-DFETCHCONTENT_SOURCE_DIR_<DEP>=`. Those two are not spelled the same way:
CPM reads `CPM_${CPM_ARGS_NAME}_SOURCE` (`CPM_0.40.0.cmake:686`), i.e. the
package name exactly as passed to `CPMAddPackage` and case-sensitive, while
CMake uppercases the name for `FETCHCONTENT_SOURCE_DIR_<NAME>`. It worked here
only because rapids-cmake names the package `CCCL`. On a lowercase-named package
(`spdlog`, `fmt`) `CPM_SPDLOG_SOURCE` is silently ignored and the full clone
happens anyway -- the exact failure the recipe exists to prevent. Say which name
each takes. The rest of the recipe checks out: the pin really is at
`rapids-cmake-src/rapids-cmake/cpm/versions.json`, key `CCCL`, `git_tag`
`8c04b6539859932f5602e86d38314e4d87f96420`, matching what the script passed.

### 5. The design rationale never addresses the alternative a maintainer will raise first

`4ceabb2`'s message weighs the guard against exactly one alternative (restoring
`lib/kernel/small_ntt.cu` for NVIDIA and duplicating 145 lines). The obvious
third option is missing: enable relocatable device code on the HIP side, which
is the direct analogue of the `CUDA_SEPARABLE_COMPILATION ON` the CUDA branch of
`heongpu_set_gpu_properties` already sets (`src/CMakeLists.txt:23`). That option
is a build-flag change that would have left `small_ntt.cuh` untouched and kept
`small_ntt.cu` in the tree -- a strictly smaller footprint on upstream sources
than deleting a file and editing a header, which is how the reviewer on the
other side will see it. There is a good answer (`-fgpu-rdc` blocks the inlining
of an NTT butterfly across the call, so the guarded-header build is likely
faster than upstream's own `-rdc` CUDA build, and the AMD result is measured
while an `-fgpu-rdc` build is not), but it is nowhere in the branch or in
notes.md. Put the reason in the commit message; do not change the design.

### Explicit verdicts on the four claims put to this review

1. **cuRAND include removal -- sound.** `util.cuh:9` is the only includer of
   `cuda_to_hip.h` in the tree; the six `kernel/*.cuh` that use cuRAND each
   include it themselves; no `curand_*` use in `src/` reaches the symbol only
   through the shim. The HIP branch is untouched, still includes
   `<hiprand/hiprand_kernel.h>`, and `thirdparty/hip_compat/curand_kernel.h`
   still satisfies those six includes (`src/CMakeLists.txt:19` puts it SYSTEM
   BEFORE). Only the explanation is wrong (finding 1).
2. **`__CUDACC__ || __HIPCC__` -- correct, and verified rather than assumed.**
   `nvcc -arch=sm_80` on a probe with `#if !defined(__CUDACC__) #error` compiles,
   and `#warning` fires in both passes with `__CUDA_ARCH__` undefined in the host
   one, so the entry's rejection of `__CUDA_ARCH__` is right. `hipcc -dM -E`
   predefines `__HIPCC__` with and without `-x hip`, in both passes, with no
   header needed -- so this does not repeat the `__HIP_PLATFORM_AMD__`
   include-order trap. g++ defines neither. The only callers are
   `lib/kernel/bootstrapping.cu` and `lib/kernel/keygeneration.cu`, both in
   `HEONGPU_KERNEL_SOURCES`, so both are `LANGUAGE HIP` on AMD and nvcc TUs on
   CUDA; nothing outside a device compilation needs the definitions. The
   declarations at `small_ntt.cuh:13-22` are byte-identical to upstream
   `1928a14`, trailing missing newline included.
3. **Design -- guard beats restoring `small_ntt.cu`.** Agreed, for the stated
   reason. See finding 5 for the missing third option.
4. **Binary equivalence -- sound, and the comparison is real.** `codeobj_diff.py`
   re-run independently here over `agent_space/heongpu-amd-baseline/bin` vs
   `src/build/bin`: `verdict=identical`, 42/42. `roc-obj-ls` on the compared
   binaries lists non-empty `hipv4-amdgcn-amd-amdhsa--gfx90a` slices (e.g.
   size=86352 in `test/bfv_addition_testcases`), so the ISA compare is not the
   vacuous empty-vs-empty case. Baseline binaries are timestamped 19:56 and the
   post-fix ones 20:00 on the same day with the same build dir layout. The a
   priori argument is independently sufficient anyway: every TU needing the
   definitions is compiled with a device compiler on AMD, so the guard is
   always true there. Withholding the claim for gfx1100 is correct, not
   over-cautious -- the carry-forward rule is per-arch and measured, and gfx1100
   was not measured.

### Also checked

CUDA build evidence is genuine and not stale: the last `=== START` in
`agent_space/heongpu-cuda-check.log` (19:58:45) is configure rc=0, build rc=0,
64 targets, zero "error" lines. 26 objects survive from the earlier failing run,
but none of them lists `cuda_to_hip.h` or `small_ntt.cuh` in
`compiler_depend.make` -- every object that depends on either changed header was
rebuilt after both edits (file mtimes 19:58:20 and 19:58:36). `jargon.py --port
HEonGPU` clean over the branch. Both titles are `[ROCm]`-prefixed and under 72
chars (64, 58); Claude named in both bodies, no noreply trailer. No submodule or
patch file touched by the delta; fork tree clean. No AMD-internal account
references.

## Porter Attempt 9 (2026-08-12, linux-gfx90a) -- round-4 findings addressed

All five findings addressed. Round 4 was a writing round: no design or code
behaviour changed, and the reviewer's verdicts on the four round-3 claims stand
as written. Fork head `4ceabb2` -> `4925df1`, two new commits, nothing amended,
`5d99b8f` still an ancestor.

### 1 + 5 (fork, upstream-visible)

`0e027a4` rewrites the `cuda_to_hip.h` comment. The old text named a condition
the fixed tree disproves ("when a host compiler sees both"); the new text names
the ORDER, which is the actual condition, plus why this header specifically was
the wrong home for the include: `util.cuh:9` pulls it in first of all, and its
own `#include <cuda_runtime.h>` does not put `device_launch_parameters.h` in
front because cuda_runtime.h includes that only under `__CUDACC__`. Re-verified
the reproducer here in BOTH directions against the CUDA 12.8 headers rather than
trusting the review, since the text ships:

```
g++-13 -fsyntax-only -I$CUDA/include probe.cpp
  curand_kernel.h, then cuda_runtime.h + device_launch_parameters.h -> 2 errors
    (device_launch_parameters.h:71 conflicting 'const uint3 threadIdx' with C
     linkage; previous declaration with C++ linkage at
     curand_mtgp32_kernel.h:124)
  the reverse order -> rc=0, silent
```

`4925df1` records the `-fgpu-rdc` rationale that finding 5 says a maintainer
will ask for first. It is a new commit, not an amend, and it carries a short
version of the reason into `small_ntt.cuh` next to the guard as well, since that
is where the question gets asked: enabling relocatable device code for HIP is
the analogue of the `CUDA_SEPARABLE_COMPILATION ON` at `src/CMakeLists.txt:23`
and would have left the header alone, but it leaves each butterfly a cross-TU
call the compiler cannot inline into the kernel running it, and the measured
20/20 result is from the header-definition build while an `-fgpu-rdc` build was
never measured. The design is unchanged.

### 2, 3, 4 (MOAT-side, this branch)

- `fault-classes.md`: the headline "cuRAND's device header is not host-includable
  at all" is replaced by the order-dependent rule, and the entry now says
  explicitly that the FIXED build has six `kernel/*.cuh` including
  `<curand_kernel.h>` reachable from the umbrella header, so every host `.cpp`
  includes it and compiles. Kept the reason a shim in particular cannot hold it.
- `SKILL.md`: one index line added under "Headers, includes and build" for the
  device-bodies-in-a-header class, so a porter scanning the index before making
  that exact move reaches the reference.
- `validation.md`: the CPM/FetchContent override recipe was wrong and would have
  cost the next porter the multi-hour clone it exists to prevent. Verified
  against the CPM copy in `agent_space/HEonGPU-upstream-build/cmake/`:

  ```
  CPM_0.40.0.cmake:686
    if(NOT CPM_ARGS_FORCE AND NOT "${CPM_${CPM_ARGS_NAME}_SOURCE}" STREQUAL "")
  ```

  so CPM takes the package name case-sensitively as `CPMAddPackage` received it
  while CMake upper-cases it for `FETCHCONTENT_SOURCE_DIR_<NAME>`. The entry now
  says which name each takes and where to read the real spelling
  (`rapids-cmake/cpm/versions.json`, or the first argument of the project's own
  `CPMAddPackage`). It was moot here only because the package is named `CCCL`.

### Verification of this round

AMD, gfx90a, ROCm 7.2.1, AMD Instinct MI250X / MI250 (said "MI210" until round 6;
this host has no MI210):

```
cmake --build projects/HEonGPU/src/build -j64        # rc=0, 0 "error:" lines
ctest --test-dir projects/HEonGPU/src/build          # 100% passed, 20/20, 13.27s
```

CUDA no-regression check re-run because finding 1 edits a header on the CUDA
include path (`bash agent_space/heongpu-cuda-check.sh`; last `=== START` is
20:32:17Z): configure rc=0, build rc=0, zero error lines. 82 objects rebuilt,
including `lib/heongpu.cpp.o` -- the host TU that reaches both edited headers --
so the check genuinely covered the edits rather than reusing stale objects.

Binary equivalence, measured not asserted:

```
python3 utils/codeobj_diff.py agent_space/heongpu-amd-baseline/bin \
                              projects/HEonGPU/src/build/bin
  verdict=identical, 42/42 (exported symbols + device ISA)
roc-obj-ls .../bin/test/bfv_addition_testcases
  hipv4-amdgcn-amd-amdhsa--gfx90a ... size=8584   (non-empty slice, not a
                                                   vacuous empty-vs-empty compare)
```

### What this means for the two archs (read the classifier, not the state word)

Both archs read `revalidate` after `advance-head`, and that was ALREADY true
before this round -- it is owed for the round-3 code fixes, not for anything
here. `changeclass.classify` on the two spans makes the distinction:

```
5d99b8f (validated_sha) .. 4925df1 -> mixed, arch_independent=False
4ceabb2 (previous head) .. 4925df1 -> comment-only, arch_independent=True
```

So this round's delta is comment-only and arch-independent by the classifier, on
top of binary-identical on gfx90a by measurement; a validator revalidating either
arch is proving the round-3 fixes (`d7d609e`, `4ceabb2`), and nothing in
`0e027a4`/`4925df1` adds to what has to be re-run. gfx1100 is deliberately not
claimed from the gfx90a measurement -- carry-forward is per-arch and measured --
but the comment-only classification is arch-independent by construction.

`jargon.py --port HEonGPU`: clean over the whole branch.
`git -C projects/HEonGPU/src status --porcelain`: empty.

## Review 2026-08-12 (round 5)

Scope: the delta `4ceabb2..4925df1` (`0e027a4`, `4925df1`) plus the three MOAT
documents the same round edited. Verdict: **changes-requested**. The code is
untouched and stays correct; the classifier and CUDA evidence check out. What is
wrong is again the prose, and in the same way as rounds 2, 3 and 4: two claims
stated as established fact are false against this machine, and both of them ship
-- one in an upstream source comment, one in an upstream commit message.

### 1. The rewritten shim comment still overstates, and the project's own build disproves it

`src/include/heongpu/cuda_to_hip.h:77-81`:

```
// only in one order: curand first. Host translation units reach this header
// first of all (util.cuh includes it ahead of everything else), and the
...
// here therefore always arrived first and broke every host .cpp.
```

The ORDER rule itself is right and I reproduced it independently (`g++-13
-fsyntax-only -I/opt/conda/envs/cuda-12.8/targets/x86_64-linux/include`: curand
first gives `device_launch_parameters.h:71` conflicting `const uint3 threadIdx`
with C linkage against `curand_mtgp32_kernel.h:124` plus the `blockDim` pair,
the reverse order is rc=0). So is the `__CUDACC__` point: `cuda_runtime.h:113`
and `:119` are both inside `#if defined(__CUDACC__)`.

The two sentences built on top of it are not.

*"Host translation units reach this header first of all"* -- not for
`src/lib/kernel/contextpool.cpp`. `src/include/heongpu/kernel/contextpool.hpp:9`
includes `gpuntt/common/nttparameters.cuh` BEFORE `util.cuh` on line 10, so that
TU reaches `device_launch_parameters.h` at preprocessed line 58806 and
`cuda_to_hip.h` only at 59557 -- the safe order. The parenthetical is true of
util.cuh; the conclusion drawn from it is not true of host TUs generally,
because a TU can reach the launch parameters through a path that does not go
through util.cuh, and one in this project does.

*"broke every host .cpp"* -- it broke one of four. Compiling all four host TUs
against the PRE-FIX shim (`git show d7d609e^:src/include/heongpu/cuda_to_hip.h`
dropped into an overlay include dir ahead of `src/include`, with each target's
own `flags.make` defines/includes/flags from the CUDA build):

```
src/lib/kernel/contextpool.cpp     rc=0  conflicting-decl-errors=0
src/lib/heongpu.cpp                rc=1  conflicting-decl-errors=2
src/lib/util/defaultmodulus.cpp    rc=0  conflicting-decl-errors=0
src/lib/util/serializer.cpp        rc=0  conflicting-decl-errors=0
```

The branch's own failing run agrees: in `agent_space/heongpu-cuda-check.log` the
19:26:51 build reports `Error 1` for `heongpu_host_core.dir/lib/heongpu.cpp.o`
and for nothing else, while contextpool/serializer/defaultmodulus all built.
`serializer.cpp` and `defaultmodulus.cpp` do not reach the shim at all (their
`.o.d` files list no `cuda_to_hip.h`).

Say what is true: the shim is reached ahead of the launch parameters by the host
TU that broke (`lib/heongpu.cpp`, via `heongpu.hpp`), and any TU that reaches
the shim before them is exposed -- which is enough to condemn the shim as the
home for the include without claiming a universal that a maintainer can refute
with one `g++ -fsyntax-only`. `0e027a4`'s body carries the same inference ("so a
curand include here reached the compiler before anything had declared the
builtins"); fix it there too.

### 2. The -fgpu-rdc rationale is false, and it is the stated basis for the design

`src/include/heongpu/kernel/small_ntt.cuh:31-32` and `4925df1`'s body both
assert that `-fgpu-rdc` "leaves each butterfly a cross-unit call that cannot be
inlined into the kernel running it" / "keeps each butterfly a call across
translation units that the compiler cannot inline".

Measured on this host (ROCm 7.2.1, `/opt/rocm/bin/hipcc`, gfx90a), it is not
true. HIP's `-fgpu-rdc` device link is a bitcode link followed by full
optimization, so cross-TU `__device__` calls ARE inlined. Built the shape this
port actually has -- a `__device__` function template defined with an explicit
instantiation in one TU and called from a `__global__` in another:

```
hipcc -O3 --offload-arch=gfx90a -fgpu-rdc -c tdev.hip -o tdev.o     # explicit instantiation
hipcc -O3 --offload-arch=gfx90a -fgpu-rdc -c tmain.hip -o tmain.o   # kernel calls it
hipcc -O3 --offload-arch=gfx90a -fgpu-rdc tdev.o tmain.o -o app3
llvm-objdump -d <extracted gfx90a bundle>
  functions in the linked device image: 0000000000001600 <_Z2k3Pyy>   (only the kernel)
  s_swappc_b64 (calls): 0
  s_barrier: 21                     (the callee's barriers, inline in the kernel)
```

The callee's standalone body is gone entirely; a 4110-instruction non-template
callee inlines the same way. The `cannot` is not a heuristic that happened to
fire -- there is no structural barrier to inline. (The intuition is right for
NVIDIA: nvcc `-rdc=true` needs `-dlto` for cross-TU device inlining. AMD's RDC
path is different, and that difference is exactly the thing a ROCm port comment
should not get backwards.)

The other half of the same comment checks out and should stay:
`small_ntt.cuh:24-26` is correct, and without the flag the link really does fail
with the error the skill quotes -- `lld: error: undefined hidden symbol: void
SmallFwd<unsigned long long>(...)`, referenced by the kernel.

Two consequences for the text:

- `4925df1` says "It was rejected on performance", which asserts both a decision
  that did not happen in that order (round 4 found the option was never
  considered) and a mechanism that is false. The adjacent "an -fgpu-rdc build
  was never measured" is prominent enough as placement, but it does not rescue
  an untrue mechanism stated as fact in the sentence before it.
- Either measure an `-fgpu-rdc` build and say what it cost, or drop the
  performance claim and give the reason that survives -- e.g. that RDC changes
  the link model for every TU in the library and for consumers, and the
  configuration the 20/20 result comes from is the header-definition one. Do not
  change the design; only stop justifying it with something untrue.

### 3. fault-classes.md repeats finding 1's overstatement into every future port

`.claude/skills/cuda-to-rocm/references/fault-classes.md:327-328`: "A shim is by
construction the FIRST header a TU pulls in, so a curand include there always
lands in the failing order." Not by construction, and not always -- the
counterexample is in the project the entry is drawn from
(`kernel/contextpool.hpp:9` ahead of `:10`, above). This is a skill entry, so it
is asserted about every project, where the claim is weaker still.

`fault-classes.md:334`: "so every host `.cpp` includes it and compiles." Of the
four host `.cpp` here, only `lib/heongpu.cpp` reaches `curand_kernel.h` --
verified from the CUDA build's dependency files, where `heongpu.cpp.o.d` lists
`curand_kernel.h` and all six kernel headers while `contextpool.cpp.o.d`,
`serializer.cpp.o.d` and `defaultmodulus.cpp.o.d` list none. The six-header
count and their reachability from `heongpu.hpp` ARE correct; the "every host
.cpp" is not. The rule the entry wants -- put the RNG include where it is
reached after the launch parameters, never in the shim -- survives without
either overstatement.

`references/validation.md:12` carries the same false universal ("a cuRAND
include the compat header leaked into every host TU"). It is round 3's text
rather than this round's, but it is on this branch and merges with it, so fix it
in the same pass.

### 4. The skill tells the next porter that -fgpu-rdc does not exist

`SKILL.md:103`: "to work around HIP's missing `-fgpu-rdc` link", and
`validation.md:12`: "the usual workaround for HIP having no
relocatable-device-code link". HIP has `-fgpu-rdc`; this project's build simply
does not enable it, which is what `fault-classes.md:343` correctly says
("Without `-fgpu-rdc` a HIP build cannot resolve that call across TUs"). The
always-loaded index line is the one a porter reads before making this exact
move, and as worded it removes a working option from their choices.

With finding 2, `fault-classes.md:340-367` should also stop presenting two
options and calling the file-restore "the other option": enabling `-fgpu-rdc` is
the third, it leaves the sources untouched, cross-TU inlining does happen on
AMD, and the honest reason to prefer the guard is the link-model change and the
fact that nobody has measured the alternative -- not an inlining barrier.

### Checked and sound -- do not redo

- **CPM correction (validation.md:13).** Verified independently, not from the
  porter's note. `CPM_0.40.0.cmake:686` is
  `if(NOT CPM_ARGS_FORCE AND NOT "${CPM_${CPM_ARGS_NAME}_SOURCE}" STREQUAL "")`,
  and nothing in that file upper-cases `CPM_ARGS_NAME`; `FetchContent.cmake:1745`
  does `string(TOUPPER ${contentName} contentNameUpper)`. The hazard is real:
  `rapids-cmake/cpm/versions.json` holds `spdlog`, `fmt`, `rmm`, `cuco`,
  `benchmark` in lower case alongside `CCCL`, so the corrected entry would have
  saved the clone on any of those. The `CCCL` pin
  `8c04b6539859932f5602e86d38314e4d87f96420` matches what the script passes.
- **SKILL.md index line placement** -- it does lead to the right entry; only its
  wording is finding 4.
- **Carry-forward.** `changeclass.classify` reproduced here:
  `5d99b8f..4925df1 -> mixed, arch_independent=False`;
  `4ceabb2..4925df1 -> comment-only, arch_independent=True`. Both archs sit at
  `validated_sha=5d99b8f` against head `4925df1`, so both genuinely owe a
  re-run, and it is owed for the round-3 code fixes rather than for this round.
  Not claiming gfx1100 from the gfx90a measurement is right.
- **CUDA no-regression evidence.** Last `=== START` in
  `agent_space/heongpu-cuda-check.log` is 20:32:17Z: configure rc=0, build rc=0
  at 20:33:09Z, 82 `Building ... object` lines including
  `heongpu_host_core.dir/lib/heongpu.cpp.o`, zero lines matching `error`. That
  object's `.o.d` is timestamped 20:32 and lists both edited headers, so the run
  covered the edits.
- **Hygiene.** Titles 58 and 56 chars, both `[ROCm]`-prefixed; Claude named in
  both bodies; no noreply trailer; delta is ASCII; `jargon.py --port HEonGPU`
  clean over the branch; fork tree clean; no AMD-internal account references.
- **Whether the small_ntt.cuh note belongs upstream at all.** It does -- it is
  design rationale for a non-obvious structure and carries no MOAT vocabulary.
  The objection is finding 2, that it is not true.

### Recommendation

**Request Changes.** Findings 1 and 2 ship to the upstream maintainer; findings
3 and 4 publish to every agent when this branch merges. No code change is
needed and the design should not move.

### 5. (added from a second reviewer, windows-gfx1151) Four commit bodies name an MI210; the only gfx90a hardware this project has recorded is an MI250X

`14c2b51`, `d7d609e`, `4ceabb2` and `4925df1` each state the AMD evidence as an
"MI210 (gfx90a)" -- `4925df1`: "measured 20 of 20 test suites passing on an
MI210". The only recorded identification of gfx90a hardware for this project is
`notes.md:1192,1195`, where the validator read `rocm-smi`/`rocminfo` on the
linux-gfx90a host and found three MI250X dies at the same ROCm 7.2.1.
`config/jargon.toml` uses "MI210 (gfx90a)" as its worked example of naming a
GPU, which is a plausible source for the string.

I cannot substantiate which card the porter's runs used -- this reviewer ran on
windows-gfx1151 with no access to that host, and the attempt-5/6/7 entries name a
platform but not a card -- so this is an inconsistency to reconcile, not a proven
error. Confirm the part on the machine that produced the runs and make the four
bodies say what it is. It is a claim about evidence, in text a maintainer reads
as evidence, and it is the only line in the branch that names a specific piece of
hardware. Whoever fixes findings 1-4 is already editing three of those four
bodies.

Reviewed independently on windows-gfx1151 (read-only clone, no build or GPU run;
this host has no HIP toolchain). That pass reached the same two conclusions as
rounds 4 and 5 -- the "sees both" cuRAND mechanism is disproved by
`heongpu.hpp:17 -> host/bfv/encryptor.cuh:10 -> kernel/encryption.cuh:12`, and
the `-fgpu-rdc` "cannot be inlined" claim is wrong because the HIP RDC device
link is a bitcode link that inlines across TUs -- and it is recorded here only as
independent corroboration, reasoned rather than measured. Round 5's gfx90a
measurement (`s_swappc_b64` count 0 in the RDC-linked image) settles it; the
duplicate write-ups were discarded rather than appended.
## Porter Attempt 10 (2026-08-12, linux-gfx90a) -- round-5 findings addressed

Fork head `4925df1` -> `81176c9`, two new commits, nothing amended, `5d99b8f`
still an ancestor. **This round DOES change compiled output** (finding 2 was
settled by measurement and the measurement changed the design), so unlike round
4 nothing here carries forward: the device ISA of every binary differs.

### Finding 2: -fgpu-rdc measured, and it wins. The guard is gone.

`8ef207a` enables `-fgpu-rdc` for the HIP build and restores
`src/lib/kernel/small_ntt.cu` and `src/include/heongpu/kernel/small_ntt.cuh` to
byte-identical upstream text (`git diff main -- <either> | wc -l` = 0). The
whole change is now fifteen lines of `src/CMakeLists.txt`, of which three are
the flag.

The reviewer's mechanism finding reproduced independently here (probe in
`agent_space/heongpu-rdc/`, a `__device__` function template with its explicit
instantiation in `tdev.hip`, called from a `__global__` in `tmain.hip`):

```
hipcc -O3 --offload-arch=gfx90a -fgpu-rdc {tdev,tmain}.hip -> links
  functions in the linked device image: only _Z2k3Pyy (the kernel)
  s_swappc_b64: 0        s_barrier: 5 (the callee's, inline in the kernel)
without -fgpu-rdc -> lld: error: undefined hidden symbol:
  void SmallFwd<unsigned long long>(...), referenced by k3(...)
```

So both halves hold: the link genuinely fails without the flag, and with it the
cross-TU device call is inlined and the callee's standalone body is gone. The
"cannot be inlined" rationale in `4925df1` was false and is retracted in
`8ef207a`'s body.

Then the cost, measured rather than asserted (gfx90a, ROCm 7.2.1, AMD Instinct
MI250X / MI250 -- said "MI210" until round 6, see below -- `-j64`,
each configuration built clean from an empty build dir):

| | guard (4925df1) | -fgpu-rdc (8ef207a) |
| --- | --- | --- |
| ctest | 20/20 | 20/20 |
| tfhe_benchmark NAND | 16.52, 16.58 ms | 16.52, 16.68 ms |
| tfhe_benchmark MUX | 29.67, 29.81 ms | 29.89, 29.99 ms |
| clean build, default (lib only) | 53.7 s | 52.1 s |
| clean build, tests+examples+benchmarks | 193.6 s | 326.6 s |
| device code in test/bfv_multiplication_testcases | 1,575,096 B | 1,684,264 B |
| of which this library's | 864,312 B (8 objects) | 973,480 B (1 image) |

The build-time cost is entirely the device link, which each of the 42
executables performs over the whole library (said "34" until round 6; the
configuration builds 15 + 15 + 5 + 4 + 3); the default build (no tests,
examples or benchmarks -- all three default OFF) is unchanged. Runtime on the
code path that actually calls the two functions is equal within run-to-run
noise. tfhe_benchmark is the right measurement target because every
`SmallForwardNTT`/`SmallInverseNTT` call site is TFHE (`keygeneration.cu:1240+`,
`bootstrapping.cu:807+`).

Consumer check, which is what made this safe to adopt: the flag rides on
`INTERFACE_COMPILE_OPTIONS`/`INTERFACE_LINK_OPTIONS` and is exported, so
`HEonGPUTargets.cmake` carries it and the downstream snippet in
`docs/advanced_topics.rst` still builds unchanged. (This round said the snippet
"needs no edit"; round 6 showed that covers only consumers whose own sources are
HIP, and the docs now say what the others need -- see the round-6 response.)
Verified by installing to a prefix and
building + running a BFV encrypt/decrypt program with that snippet verbatim:
`roundtrip OK`. This is the analogue of the CUDA side, where the static library
already sets `CUDA_RESOLVE_DEVICE_SYMBOLS OFF` and the documented consumer
snippet turns `CUDA_SEPARABLE_COMPILATION` on.

Where the design note went: with the header back to upstream text, a comment
there would be a gratuitous edit to a file we no longer touch. The rationale now
sits next to the flag in `src/CMakeLists.txt` (why RDC, and the interface/export
half separately). That satisfies round 4's "a maintainer will ask why" without
reopening a file.

CMake has NO `HIP_SEPARABLE_COMPILATION` property (checked, CMake 4.0.3 property
list), which is why this is hand-wired rather than a one-line property.

### Finding 1: the shim comment now names the TU it broke

`81176c9`. Measured the include order per host TU by preprocessing with each
target's own `flags.make` defines/includes/flags and grepping the line numbers:

```
lib/heongpu.cpp            cuda_to_hip@18      device_launch_parameters@58820  <- failing order
lib/kernel/contextpool.cpp cuda_to_hip@59557   device_launch_parameters@58806  <- safe
lib/util/defaultmodulus.cpp (no shim)          device_launch_parameters@58807
lib/util/serializer.cpp    (neither)
```

which is the reviewer's result exactly. The comment now says a host TU that
reaches this header before the launch parameters fails, names `lib/heongpu.cpp`
as the one that did, and claims nothing about "every host .cpp". Note for later:
the example and benchmark `.cpp` files DO reach both (their `.o.d` files list
`cuda_to_hip.h` and `curand_kernel.h`), so "one of four" is a statement about
the library's own host TUs, not about the whole build -- the failing build
simply never got as far as the examples.

### Findings 3 and 4: the skill

- `fault-classes.md`, cuRAND entry: "by construction the FIRST header a TU pulls
  in ... always lands in the failing order" and "every host `.cpp` includes it
  and compiles" are both gone. It now says a shim TENDS to be reached early,
  that one exposed TU is enough, and gives the `g++ -E | grep -n` recipe for
  settling the order per TU, with the four HEonGPU numbers as the worked example
  including the counterexample TU.
- `fault-classes.md`, device-bodies entry: rewritten around `-fgpu-rdc` as the
  normal answer, with the CMake wiring (compile AND link, and the interface half
  for consumers), the measured inlining result, the measured costs, and the
  header-move demoted to what you reach for when you do not know the flag
  exists. The closing paragraph now distinguishes the guard (codegen-neutral on
  AMD) from the RDC switch (changes every binary's ISA, so every arch owes a
  fresh run).
- `SKILL.md:103` and `validation.md:12` no longer say HIP lacks
  relocatable-device-code linking.

### Verification of this round

```
cmake --build .../build -j64                # rc=0, 0 error lines (both commits)
ctest --test-dir .../build                  # 100% passed, 20/20, 13.44s then 13.33s
bash agent_space/heongpu-cuda-check.sh      # last START 21:22:04Z: configure rc=0,
                                            # build rc=0, 81 objects, 0 error lines
```

The CUDA check was run TWICE, once per commit, because both commits touch files
on the CUDA include path; the first run (21:15:34Z) is the one that proves
`lib/kernel/small_ntt.cu` compiles and links under nvcc again (it appears at log
line 230 as a CUDA object, and 42 executables link). The CUDA path is now
upstream's own arrangement for these two files, so this is the least
CUDA-divergent the branch has been.

```
python3 utils/codeobj_diff.py agent_space/heongpu-amd-baseline/bin .../build/bin
  verdict=differ, all 42 binaries "device ISA differs", no symbol differences
```

That is EXPECTED and is the point: the device link re-optimizes the whole image.
Do not attempt a carry-forward on this delta. Both archs must re-run the 20
suites at `81176c9`; gfx1100 has never run an RDC build of this project and its
result cannot be inferred from gfx90a's.

If `codeobj_diff.py` reports "binary set differs" with names like
`*.0.hipv4-amdgcn-amd-amdhsa--gfx90a`, those are leftover bundles from someone
running `llvm-objdump --offloading` inside `bin/test`; delete them and re-run.

`jargon.py --port HEonGPU`: clean over the whole branch.
`git -C projects/HEonGPU/src status --porcelain`: empty.

## Review 2026-08-12 (round 6)

Scope: the delta `4925df1..81176c9` (`8ef207a`, `81176c9`), the skill edits it
carries, and the standing bar over the whole branch. Verdict:
**changes-requested**. The design change is right and I am not asking for it to
move -- I reproduced the mechanism and measured the two benchmarks the porter
did not. What is wrong is again upstream-visible text: one factual error that
round 5 already raised and this round did not touch, one wrong count that the
same round promoted into the skill, and one claim about what a consumer has to
know that the branch's own docs contradict.

### 1. Five commit bodies still say MI210; this host has no MI210

Round 5's finding 5 (notes.md:2075) flagged this as an inconsistency it could
not settle from windows-gfx1151. I am on the linux-gfx90a host that produced
every AMD run, so I can settle it: it is wrong.

```
rocminfo   -> Name: gfx90a / Marketing Name: AMD Instinct MI250X / MI250   (x3)
rocm-smi --showproductname -> Card Series: AMD Instinct MI250X / MI250
                              Card SKU: D65209   GFX Version: gfx90a
```

That matches the validator's own reading at notes.md:1192. `git log main..HEAD
--format="%h %b" | grep MI210` returns five hits, in `14c2b51`, `d7d609e`,
`4ceabb2`, `4925df1` and -- new this round -- `8ef207a`:

```
20/20 suites pass on an MI210 (gfx90a) with ROCm 7.2.1.
```

`config/jargon.toml` uses "MI210 (gfx90a)" as its worked example of naming a
GPU, which is the likely source. Fix all five. `rocminfo` reports the part only
as "MI250X / MI250", so the honest string is that one verbatim, or just
"gfx90a" -- do not substitute a second guess. notes.md:2135 carries the same
"MI210" and should be corrected in the same pass.

### 2. "34 executables" is wrong; the build has 42, and the skill now carries the wrong number

`8ef207a` body: "Tests, examples and benchmarks all enabled (34 executables)".
notes.md:2148: "each of the 34 executables". The configuration described builds
42:

```
test/CMakeLists.txt                  15
example/basic/CMakeLists.txt         15
example/bootstrapping/CMakeLists.txt  5
example/mpc/CMakeLists.txt            4
benchmark/CMakeLists.txt              3
                                     --
                                     42
find build/bin -type f -executable | wc -l          -> 42
"Linking ... executable" in each of the last two
  nvcc runs in agent_space/heongpu-cuda-check.log   -> 42
```

The 193.6 s -> 326.6 s figure is attributed to per-executable device links, so
the executable count is load-bearing for the one cost number in the commit. It
also went into the skill: `fault-classes.md:372` says "once 34
test/example/benchmark executables are enabled", while `fault-classes.md:407`,
in the same entry, says "all 42 gfx90a binaries". The entry contradicts itself,
and 42 is the right one in both places.

### 3. -fgpu-rdc changes what a consumer must do, and both the comment and the docs say it does not

`src/CMakeLists.txt:255-259`:

```
# ... keeps that a detail of this library rather than something every
# consumer has to know, exactly as CUDA_RESOLVE_DEVICE_SYMBOLS OFF below leaves
# the CUDA device link to the consumer.
```

Both halves of that need fixing.

*It is not a detail the consumer can be unaware of.* Relocatable device code
leaves no complete device image in the archive, so the final link must go
through the HIP driver. Measured here (ROCm 7.2.1, gfx90a), a `__device__`
function in one TU called from a `__global__` in another:

```
hipcc -fgpu-rdc -c {lib,app}.hip ; hipcc -fgpu-rdc app.o librdc.a -o a_hip
  ./a_hip -> val=42, and the linked device image holds only _Z1kPy,
             s_swappc_b64 count 0   (the inlining claim reproduces)
g++ -no-pie app.o librdc.a -lamdhip64
  /usr/bin/ld: undefined reference to `__hip_gpubin_handle_4760757e0f294a70'
  /usr/bin/ld: undefined reference to `__hip_fatbin_4760757e0f294a70'
same source compiled WITHOUT -fgpu-rdc, single TU, g++ -no-pie solo.o -lamdhip64
  ./s_gcc -> val=42                 (a non-RDC HIP object links fine under g++)
```

So a consumer that could previously link `libheongpu.a` with any host linker
now cannot. For a CMake consumer the exported
`INTERFACE_LINK_OPTIONS "-fgpu-rdc"` (confirmed present in the generated
`HEonGPUTargets.cmake:66`) mostly hides this, but it is unconditional, so a
consumer whose target resolves to a CXX link -- an app `.cpp` linking their own
HIP library that links `HEonGPU::heongpu` -- gets
`g++: error: unrecognized command-line option '-fgpu-rdc'` instead.

To be explicit, because it is the obvious wrong fix: do NOT wrap the link
option in `$<LINK_LANGUAGE:HIP>`. Guarding it does not make that case work, it
only swaps a message that names the flag for the `__hip_fatbin_*` message that
does not. The flag should stay unconditional; what is missing is prose.

*The CUDA analogy is backwards.* On the CUDA side the consumer IS told:
`docs/advanced_topics.rst:52` instructs them to put
`CUDA_SEPARABLE_COMPILATION ON` on their own target. The HIP block this branch
added at `docs/advanced_topics.rst:54-75` has no counterpart line, so a reader
comparing the two snippets concludes the HIP side needs nothing. Following the
snippet verbatim happens to work only because its single `main.cpp` is marked
`LANGUAGE HIP` and the link language follows.

Two edits: reword `src/CMakeLists.txt:255-259` to drop the "consumer has to
know" claim and state what the flag actually requires, and add one sentence to
the HIP block in `docs/advanced_topics.rst` mirroring line 52 -- the consuming
target must be compiled and linked as HIP, because the installed archive holds
relocatable device code.

### 4. The skill's -fgpu-rdc recipe passes the same gap to every future port

`fault-classes.md:352-362` gives the three-line CMake block that every future
port will copy, with the comment "whoever links the archive performs the device
link, including a consumer of the installed export, so put it on the interface
rather than in everyone's build". It does not say the consequence, which is
finding 3: after this the archive cannot be linked by anything but the HIP
driver, and a plain `g++` link fails with `undefined reference to
__hip_fatbin_*`. Add that, with the error string, and say the project's own
downstream docs need a line about it. Also fix the 34 at `fault-classes.md:372`.

### Checked and sound -- do not redo

- **The CMake wiring.** The flag reaches the six object libraries through
  `heongpu_set_gpu_properties` (`src/CMakeLists.txt:24`) and reaches
  tests/examples/benchmarks through the interface, guarded by
  `$<COMPILE_LANGUAGE:HIP>` so the `.cpp` sources compiled as CXX never see it.
  The CUDA path is untouched: everything new is inside `if(USE_HIP)`, and
  `CUDA_SEPARABLE_COMPILATION ON` at `src/CMakeLists.txt:30`/`266` still governs
  there. `lib/kernel/small_ntt.cu` is back in `HEONGPU_KERNEL_SOURCES` at the
  same position upstream has it (`git show main:src/CMakeLists.txt:88`).
  Applying the flag to all six object libraries rather than only
  `heongpu_kernel` is the right call -- a mixed RDC/non-RDC archive buys nothing
  measurable here and is fragile; do not "optimize" it later.
- **No `HIP_SEPARABLE_COMPILATION` property.** Verified, not accepted:
  `cmake --version` is 4.0.3 and `cmake --help-property-list | grep -i separable`
  returns only `CUDA_RESOLVE_DEVICE_SYMBOLS` and `CUDA_SEPARABLE_COMPILATION`.
  The only HIP properties are `HIP_ARCHITECTURES`, `HIP_EXTENSIONS`,
  `HIP_STANDARD`, `HIP_STANDARD_REQUIRED`. Hand-wiring is necessary.
- **Cross-TU device inlining under -fgpu-rdc.** Reproduced a third time, above:
  linked device image contains only the kernel, `s_swappc_b64` count 0.
- **The export carries the flags.** `INTERFACE_COMPILE_OPTIONS
  "$<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>"` and `INTERFACE_LINK_OPTIONS
  "-fgpu-rdc"` are both in the generated `HEonGPUTargets.cmake` (lines 63 and
  66). `docs/advanced_topics.rst` needs no *correction*, only the addition in
  finding 3. (Note for whoever retests the install: `cmake --install --prefix`
  does not override this project's baked-in absolute destinations; reconfigure
  with `-DCMAKE_INSTALL_PREFIX`, as the porter did.)
- **The performance gap I was worried about is closed -- I measured it.** RDC
  changes the ISA of every kernel, not just the small-NTT callers, so TFHE gates
  alone were not enough. Both remaining benchmarks exist in both builds
  (`agent_space/heongpu-amd-baseline/bin` is the pre-RDC 4925df1 build), so this
  cost nothing to settle. gfx90a, ms:

  | | guard (4925df1) | -fgpu-rdc (81176c9) |
  | --- | --- | --- |
  | BFV 32768 multiplication | 2.677 | 2.449 |
  | BFV 32768 relinearization | 0.9559 | 0.9548 |
  | BFV 32768 decryption | 0.2802 | 0.2679 |
  | BFV 65536 multiplication | 10.006 | 10.012 |
  | BFV 65536 relinearization | 6.677 | 6.666 |
  | BFV 65536 rotate row | 6.662 | 6.657 |
  | CKKS 32768 relinearization | 1.4171 | 1.4085 |
  | CKKS 32768 rotate row | 1.6026 | 1.6028 |
  | CKKS 32768 decode | 1.3169 | 1.2966 |
  | CKKS 32768 encryption | 0.7303 | 0.7071 |

  No regression anywhere; RDC is marginally ahead on several. I also re-ran
  `tfhe_benchmark` on the RDC build: NAND 16.5245, MUX 30.0051, inside the
  reported range. The decision to adopt `-fgpu-rdc` is supported. Worth adding
  one line to `8ef207a`'s body saying BFV and CKKS were measured too, since the
  body currently reads as if only TFHE was.
- **The shim comment.** `src/include/heongpu/cuda_to_hip.h:74-84` is true and
  complete. `lib/heongpu.cpp` includes only `<heongpu/heongpu.hpp>`, so "reaches
  this header through heongpu.hpp" is right; six kernel headers include
  `curand_kernel.h` themselves (`kernel/{addition,bootstrapping,decryption,`
  `encoding,encryption,keygeneration}.cuh`) and the build compiles, so "reached
  after the launch parameters" holds. The comment names one TU as an example and
  claims nothing universal or exclusive, so omitting the example/benchmark `.cpp`
  nuance is honest -- those TUs do not contradict anything it says.
- **CUDA no-regression evidence.** Last `=== START` in
  `agent_space/heongpu-cuda-check.log` is 21:22:04Z (line 1854): CONFIGURE rc=0,
  BUILD rc=0, 81 objects, 42 executables linked, 0 lines matching `error`. The
  previous run (START 21:15:30Z, CONFIGURE rc=0 21:15:34Z) rebuilt 64 objects
  including `heongpu_kernel.dir/lib/kernel/small_ntt.cu.o` and linked 42
  executables, which is the run that proves the restored file compiles and links
  under nvcc.
- **Revalidation accounting.** `moatlib classify HEonGPU 5d99b8f 81176c9` ->
  `class=mixed arch_independent=False inert=False`. Both archs sit at
  `validated_sha=5d99b8f` and genuinely owe a fresh 20-suite run at `81176c9`.
  Nothing in the branch claims otherwise; notes.md:2233-2236 says so explicitly.
- **Hygiene.** Longest title on the branch is 64 chars, all `[ROCm]`-prefixed;
  both new bodies disclose Claude and carry a Test Plan; no `Co-Authored-By` and
  no noreply trailer anywhere; the only non-ASCII in the delta is the upstream
  author's name in the byte-identical restore of `small_ntt.cu`;
  `jargon.py --port HEonGPU` clean over the whole branch; fork tree clean; no
  AMD-internal account references.

### Recommendation

**Request Changes.** No code change is needed and the design must not move.
Findings 1-3 ship to the maintainer; finding 4 publishes to every agent when
this branch merges. Finding 1 is a repeat of round 5's finding 5.

## Round 6 response (linux-gfx90a, 56615ec)

All four findings addressed in one fork commit, `56615ec` "[ROCm] Say what the
relocatable device code asks of consumers", plus notes and skill edits here.
`-fgpu-rdc` did not move.

### 1. The GPU is an MI250X/MI250, not an MI210

Confirmed on this host: `rocminfo` gives `Name: gfx90a` /
`Marketing Name: AMD Instinct MI250X / MI250` for both agents, and
`rocm-smi --showproductname` gives `Card Series: AMD Instinct MI250X / MI250`,
`Card SKU: D65209`, `GFX Version: gfx90a`. No MI210 anywhere. The five earlier
commit bodies (`14c2b51`, `d7d609e`, `4ceabb2`, `4925df1`, `8ef207a`) cannot be
amended, so `56615ec`'s body states the right part and says to read those
results as gfx90a on an MI250X/MI250. Corrected in place here at the two lines
that asserted it as fact (the round-4 verification header and the round-5 cost
table); the round-5/round-6 review sections still quote "MI210" because they are
the record of the finding.

### 2. 42 executables, not 34

`set(EXECUTABLES ...)` pairs: test 15, example/basic 15, example/bootstrapping
5, example/mpc 4, benchmark 3 = 42, matching `find build/bin -type f
-executable | wc -l` = 42 and the 42 links in each nvcc run. The cost claim is
re-derived from 42: 326.6 - 193.6 = 133 s of extra device link over 42
executables, about 3.2 s each. Fixed in `56615ec`'s body, in the round-5 section
above, and in the skill (`fault-classes.md`, which said 34 in one line and 42 in
another of the same entry).

### 3 and 4. What -fgpu-rdc asks of a consumer

Measured against a real installed export (`cmake -DCMAKE_INSTALL_PREFIX=...`,
`agent_space/heongpu-consumer/`, gitignored) rather than a toy:

| consumer | result |
| --- | --- |
| documented snippet verbatim (`main.cpp` LANGUAGE HIP) | links, `roundtrip OK` |
| same program split: HIP static lib + CXX `app.cpp` executable | `ld.lld: error: undefined hidden symbol: __hip_gpubin_handle_<hash>`, `undefined symbol: __hip_fatbin_<hash>` |
| that one + `target_link_options(app PRIVATE --hip-link)` | links, `roundtrip OK` |
| that one + `LINKER_LANGUAGE HIP` instead | still fails, identical errors |
| `g++ -no-pie` over the installed archive | `undefined reference to __hip_gpubin_handle_<hash>` |
| `hipcc` without `-fgpu-rdc` | same undefined symbols |
| `hipcc -fgpu-rdc` | links, `roundtrip OK` |

The mechanism is narrower than "the link language must be HIP". In the failing
CMake case the link language ALREADY resolves to HIP -- CMake picks
`/opt/rocm/lib/llvm/bin/clang++` because a linked static library has HIP sources
-- but the link line lacks `--hip-link`, which CMake adds only for a target that
has HIP sources of its own. Compare the two generated `link.txt`:

```
consumer_a: clang++ ... --offload-arch=gfx90a --hip-link --rtlib=compiler-rt ... -fgpu-rdc
consumer_b: clang++ ... --offload-arch=gfx90a                                   -fgpu-rdc
```

`set_target_properties(... LINKER_LANGUAGE HIP)` does not change that line.
`target_link_options(... --hip-link)` does, and that is what the docs now tell a
consumer to add. As the reviewer said, `$<LINK_LANGUAGE:HIP>` around the
interface flag was not tried as a fix: it addresses the wrong failure.

Edits: `src/CMakeLists.txt` comment no longer claims consumers are unaffected
and names the failure; `docs/advanced_topics.rst` gains a paragraph after the
HIP snippet, mirroring the CUDA `CUDA_SEPARABLE_COMPILATION` line, plus the
non-CMake case (`hipcc -fgpu-rdc`). The skill's `-fgpu-rdc` entry gains the same
consequence, including the `--hip-link` trap, so the next port copies the recipe
with the contract attached.

### Nothing compiled changed

Comment and `.rst` only. Verified rather than asserted: SHA-256 of all 42 AMD
binaries before and after the rebuild are identical (`hashes-before.txt` vs
`hashes-final.txt` in the scratch dir), and the CUDA reconfigure+build had
nothing to do.

```
cmake --build projects/HEonGPU/src/build -j64   # rc=0, no error/warning lines
ctest --test-dir projects/HEonGPU/src/build     # 100% passed, 20/20, 13.41s
bash agent_space/heongpu-cuda-check.sh          # START 22:40:47Z: CONFIGURE rc=0,
                                                # BUILD rc=0, 0 error lines
```

The 20-suite run and the CUDA check were re-run anyway. Both archs still owe a
fresh GPU run at the new head for the `-fgpu-rdc` switch itself (unchanged from
round 5). `moatlib classify HEonGPU 81176c9 56615ec` ->
`class=comment-only arch_independent=True inert=True` (the `.rst` is not source
at all); `classify HEonGPU 5d99b8f 56615ec` is still `class=mixed inert=False`,
which is the `-fgpu-rdc` switch, so the 20 suites are owed at `56615ec` on both
archs.

`jargon.py --port HEonGPU`: clean over the whole branch.
`git -C projects/HEonGPU/src status --porcelain`: empty.

Gotcha worth remembering here: `advance-head` accepts an abbreviated sha
verbatim, so a typo in the abbreviation is recorded silently. Pass
`git rev-parse HEAD`.

## Review 2026-08-12 (round 7, linux-gfx90a, 56615ec)

Scope: the delta `81176c9..56615ec` (one comment-and-docs commit) plus this
branch's `fault-classes.md` and notes edits. Rounds 1-6 cleared everything
earlier, including `-fgpu-rdc` itself. Every claim below was re-measured on this
host (ROCm 7.2.1, gfx90a, CMake 4.0.3) rather than taken from the round-6
record; the reproduction lives in `agent_space/rdc-check/` and
`agent_space/rdc-check2/` (gitignored).

Re-derived and confirmed, so do not redo them: the failing CMake consumer is
already linked by `/opt/rocm/lib/llvm/bin/clang++` even though
`CMAKE_CXX_COMPILER` is `/usr/bin/c++`, i.e. the escalation comes from a linked
target that has HIP sources and the missing piece really is `--hip-link`, which
CMake adds only for a target with HIP sources of its own; `LINKER_LANGUAGE HIP`
does not add it; `$<LINK_LANGUAGE:HIP>` around the interface flag does trade the
unrecognized-option error for the undefined-symbol one (round 6 flagged it as
untried, it is now measured); 42 executables (`find build/bin -type f
-executable` = 42, and 15+15+5+4+3 from the `set(EXECUTABLES ...)` pairs) with
326.6 - 193.6 = 133 s over 42 = 3.2 s each, and `fault-classes.md:390` and
`:426` now both say 42; `rocminfo`/`rocm-smi` give MI250X / MI250, SKU D65209,
no MI210; `classify 81176c9 56615ec` is comment-only/inert and
`classify 5d99b8f 56615ec` is still mixed/not inert, and nothing on the branch
claims the owed 20-suite runs are discharged; `jargon.py --port HEonGPU` clean;
fork tree clean; title `[ROCm] ...` 61 chars, AI disclosure present, no agent
trailer, ASCII.

### 1. `docs/advanced_topics.rst:76-81` gives a remedy that fails for the consumer shape its own sentence describes

The paragraph says "A target that links the library but has no HIP source of its
own does not get the HIP link and fails with undefined references to
`__hip_fatbin_*`" and offers exactly one fix,
`target_link_options(<your-target> PRIVATE --hip-link)`. That fix works only
when something else in the target's link chain is a CMake target with HIP
sources, because that is what makes CMake drive the link with the HIP compiler
in the first place. The measurement behind the round-6 change had such a target
(`agent_space/heongpu-consumer/b/CMakeLists.txt` links a local `wrap` static
library whose source is LANGUAGE HIP), so the shape where the consumer links
only the imported `HEonGPU::heongpu` was never covered. Measured here, an
imported rdc archive with `INTERFACE_LINK_OPTIONS -fgpu-rdc` standing in for the
installed export:

| consumer target | link driver | result |
| --- | --- | --- |
| one HIP source (the documented snippet) | rocm `clang++`, CMake adds `--hip-link` | links, runs |
| only `.cpp`, links a local HIP-source lib that links the archive | rocm `clang++`, no `--hip-link` | `undefined hidden symbol: __hip_gpubin_handle_*` |
| that + `target_link_options(... --hip-link)` | rocm `clang++` | links, runs |
| that + `LINKER_LANGUAGE HIP` instead | rocm `clang++`, still no `--hip-link` | same undefined symbols |
| only `.cpp`, links only the imported archive | `/usr/bin/c++` | ``g++: error: unrecognized command-line option `-fgpu-rdc'`` |
| that + `target_link_options(... --hip-link)` | `/usr/bin/c++` | same, plus `unrecognized command-line option '--hip-link'` |
| that + `LINKER_LANGUAGE HIP` and `--hip-link` | rocm `clang++` with `--hip-link` | links, runs |
| that + one HIP source instead | rocm `clang++` with `--hip-link` | links, runs |

So in the last four rows the documented remedy makes the error worse rather than
fixing it, and `LINKER_LANGUAGE HIP` -- which the same round concluded "does not
help" -- is precisely what is missing there. Both readings of "no HIP source of
its own" are reachable; the doc states one symptom and one fix for two different
link drivers.

Fix in prose, no code change: either scope the sentence to a target that is
already linked by the HIP compiler (say why: something in its link chain has HIP
sources) and add the second case, or give the one remedy that covers both --
compile one of the target's own sources as HIP, exactly as the snippet does, and
fall back to `LINKER_LANGUAGE HIP` plus `--hip-link` when that is impossible.
Both were measured to work against an imported archive.

Two companion lines carry the same over-generalization and should move with it:
`src/CMakeLists.txt:257`, "any other link fails with undefined `__hip_fatbin_*`
and `__hip_gpubin_handle_*` references", which is true of a hand-written `g++`
link but not of a CMake target that receives the interface `-fgpu-rdc` and is
linked by gcc; and `fault-classes.md:375-376`, "Setting `LINKER_LANGUAGE HIP`
does not add it either; `target_link_options(app PRIVATE --hip-link)` does",
which holds only inside the case the sentence before it sets up and reads as
general advice to the next port copying that recipe.

### Recommendation

**Request Changes** on finding 1 alone. Nothing else in the delta needs to
change: `-fgpu-rdc` must not move, the MI250X correction and its split between
corrected-in-place facts and quoted review record is right, the counts and the
no-op claim check out, and the CUDA path is untouched (the diff is entirely
inside `if(USE_HIP)` plus additive `.rst`). The fix is prose in three files, no
rebuild and no new GPU obligation: both archs already owe the 20 suites at head
for the `-fgpu-rdc` switch, and a docs-only commit on top does not add to that.

## Round 7 fix 2026-08-12 (linux-gfx90a, f657723)

Addresses the single round-7 finding. Prose only in three files
(`docs/advanced_topics.rst`, `src/CMakeLists.txt`, and the skill's
`fault-classes.md`); `moatlib classify 56615ec f657723` reports
`class=comment-only arch_independent=True inert=True`, so no rebuild of the
library and no new GPU obligation: both archs still owe the 20 suites at head
for the `-fgpu-rdc` switch itself, and this commit adds nothing to that.

Rebuilt the whole consumer matrix from scratch rather than reusing the round-6
or the review reproduction (`agent_space/round7/`, gitignored: `mini/` is a
two-TU `-fgpu-rdc` static library with a plain C++ entry point standing in for a
hand-imported archive, `consumers/` holds one CMake target per row and also
links the real installed export at
`agent_space/heongpu-consumer/prefix`). ROCm 7.2.1, gfx90a, CMake 4.0.3,
`CMAKE_CXX_COMPILER=/usr/bin/c++`.

The measurement that changes the answer: the installed export sets
`IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "HIP"` in
`lib/cmake/HEonGPU-1.1/HEonGPUTargets-release.cmake`, which CMake writes on its
own for a static library built from HIP sources. So a consumer that reaches the
library through `find_package` is ALWAYS driven by
`/opt/rocm/lib/llvm/bin/clang++` and always receives the interface `-fgpu-rdc`,
even when its own sources are plain C++ and even when it reaches the library
through a plain C++ intermediate library. The review's plain-consumer row was
measured on a stand-in imported target without that property, which is why it
saw `/usr/bin/c++` and a rejected `--hip-link`; that shape is real but it is not
what `find_package` produces.

Rows, all rebuilt here (`rc`, link driver from `link.txt`):

| consumer target | substrate | driver | result |
| --- | --- | --- | --- |
| one own source as HIP (the documented snippet) | real export | rocm clang++ | links, `context OK` |
| plain C++, links only `HEonGPU::heongpu`, references a device-carrying member via `-Wl,-u,_ZN7heongpu10modInverseEmm` | real export | rocm clang++ | undefined `__hip_fatbin_1f6ece...` / `__hip_gpubin_handle_` |
| that + `--hip-link` | real export | rocm clang++ | links, runs |
| that + `LINKER_LANGUAGE HIP` only | real export | rocm clang++ | same undefined symbols (the property never adds `--hip-link`) |
| that + `LINKER_LANGUAGE HIP` and `--hip-link` | real export | rocm clang++ | links, runs |
| plain C++ exe over a plain C++ static lib that links the export | real export | rocm clang++ | undefined symbols; `--hip-link` fixes it |
| plain C++ exe over a local HIP-source wrapper lib | real export | rocm clang++ | undefined symbols; `--hip-link` fixes it; `LINKER_LANGUAGE HIP` + `--hip-link` also works |
| plain C++, links a hand-imported archive carrying only `INTERFACE_LINK_OPTIONS -fgpu-rdc` | `mini` | `/usr/bin/c++` | `unrecognized command-line option '-fgpu-rdc'` |
| that + `--hip-link` | `mini` | `/usr/bin/c++` | plus `unrecognized command-line option '--hip-link'` |
| that + `LINKER_LANGUAGE HIP` only | `mini` | rocm clang++ | undefined `__hip_fatbin_*` |
| that + `LINKER_LANGUAGE HIP` and `--hip-link` | `mini` | rocm clang++ | links, prints `mini_run 42` |
| archive named by path, no imported target | real export | `/usr/bin/c++` | undefined `__hip_gpubin_handle_*` (the flag never reaches the line) |
| that + `LINKER_LANGUAGE HIP` + `--hip-link` | real export | rocm clang++ | still undefined; `-fgpu-rdc` is missing too |
| that + `LINKER_LANGUAGE HIP` + `--hip-link` + `-fgpu-rdc` | real export | rocm clang++ | links, runs |

So the shipped `.rst` keeps `--hip-link` as the remedy for a `find_package`
consumer, which is now stated with the reason (the export supplies the driver
and the flag; CMake supplies HIP link mode only for a target with its own HIP
source), and gains one sentence for the by-path shape, which needs all three by
hand. `src/CMakeLists.txt` names both failure modes instead of one. The
generalizable half went to the skill's `fault-classes.md` as a driver/line split
with the same table, since the next port copying that recipe is exactly who was
misled.

Not taken, for a person to rule on: `target_link_options(heongpu INTERFACE
--hip-link)` would make every CMake consumer of the export work with no action
at all, because the export already forces the HIP driver. It is a code change
with its own risk (a consumer that pins `LINKER_LANGUAGE CXX` would then get an
option gcc rejects), so it is out of scope for a prose round.

## Review 2026-08-12 (round 8, linux-gfx90a, f657723)

Scope: the delta `56615ec..f657723` (one comment-and-docs commit) plus this
branch's `fault-classes.md` edit and the new deferral. Everything earlier is
cleared. Measured on this host against the REAL installed export at
`agent_space/heongpu-consumer/prefix` (ROCm 7.2.1, gfx90a, CMake 4.0.3,
`CMAKE_CXX_COMPILER=/usr/bin/c++`); reproduction in `agent_space/revcheck/`
(gitignored).

Confirmed, do not redo: `HEonGPUTargets-release.cmake:11` really does set
`IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "HIP"`, and with that property a
plain-C++ consumer target that links `HEonGPU::heongpu` -- directly or through
a plain C++ static library -- is driven by `/opt/rocm/lib/llvm/bin/clang++`,
receives the interface `-fgpu-rdc`, and needs only `--hip-link`
(`undefined hidden symbol: __hip_gpubin_handle_1f6ece6550d50f9a`). The round-7
`/usr/bin/c++` row was measured on a stand-in without the property; the porter's
correction of it is right. `classify 56615ec f657723` =
`comment-only arch_independent=True inert=True`, so no rebuild and no new GPU
obligation; `classify 5d99b8f f657723` is still `mixed`/not inert and both
platforms are recorded validated at `5d99b8f`, so both archs still owe the 20
suites at head and nothing on the branch claims otherwise. `jargon.py --port
HEonGPU` clean over the whole branch, `prose.py` clean on the body, title 56
chars with `[ROCm]`, AI-assistance disclosure present, no agent trailer, ASCII
only, no internal references, fork tree clean.

### 1. Doc, comment and skill all assert the HIP link driver unconditionally, but it needs the HIP language enabled in the consumer project

`docs/advanced_topics.rst:76` says the export's link interface language means
"every CMake target that links it, directly or through another library of your
own, is driven by the HIP compiler and gets that flag without asking", and that
a target with no HIP source of its own "is therefore driven by the HIP compiler
but not in HIP link mode". `src/CMakeLists.txt:260-262` makes the same claim
("covers the first half for every consumer ... hands the HIP driver even to a
target with no HIP sources"), and `fault-classes.md:377` and `:392-394` state it
as the general rule ("Because a consumer reaching the archive through
`find_package` always lands in the HIP-driver rows").

CMake can only select the HIP linker driver if HIP is an enabled language in the
consuming project. Same source tree, only `project(...)` differing:

| consumer project | link driver | result |
| --- | --- | --- |
| `project(x LANGUAGES CXX HIP)`, plain C++ exe linking `HEonGPU::heongpu` | rocm clang++ | undefined `__hip_gpubin_handle_1f6ece6550d50f9a`; `--hip-link` fixes it |
| same, through a plain C++ static lib | rocm clang++ | same |
| `project(x LANGUAGES CXX)`, plain C++ exe linking `HEonGPU::heongpu` | `/usr/bin/c++` | ``unrecognized command-line option `-fgpu-rdc'`` |
| same, through a plain C++ static lib | `/usr/bin/c++` | same |
| same plus `target_link_options(app PRIVATE --hip-link)` | `/usr/bin/c++` | both `-fgpu-rdc` and `--hip-link` unrecognized |
| `project(x LANGUAGES CXX)` plus `find_package(hip REQUIRED)` and `hip::host` | `/usr/bin/c++` | ``unrecognized command-line option `-fgpu-rdc'`` |

The installed `HEonGPUConfig.cmake` does not enable the HIP language and
configures fine in a CXX-only project, and `find_package(hip)` does not enable it
either (last row) -- only `project(... LANGUAGES CXX HIP)` or
`enable_language(HIP)` does. So the failing consumer is not only the by-path
shape that `:83` blames: it is reachable straight through `find_package`, and it
is reachable by exactly the reader the paragraph addresses, since a target with
no HIP source of its own is the one with no reason to have listed HIP in
`project()`. For that reader the documented remedy adds a second rejected option
to a link that already failed on the first, which is the round-7 defect in a new
place.

Fix in prose, no code change, three spots: scope the claim at `.rst:76` to a
consumer project that enables the HIP language and say what to add when it does
not (`enable_language(HIP)`, or `LINKER_LANGUAGE HIP` together with
`--hip-link`); make `:83` about the missing language rather than only about
naming the archive by path; add the same condition to `src/CMakeLists.txt:261`;
and in `fault-classes.md` put the condition on the driver list at `:374-379`, on
the table row at `:386`, and replace "always lands in the HIP-driver rows" at
`:392` with the enabled-language qualifier. The skill entry is otherwise the
right shape -- the driver/line split and the warning about generalizing from one
measured shape are what was missing in rounds 6 and 7 -- but as written it
publishes an unconditional rule that a future porter will copy into their own
project's docs and get wrong the same way.

### 2. The deferral's stated cost is a consumer that the shipped interface already breaks

`deferred.json:63` (`heongpu-hip-link-interface-option`) and `notes.md` under
"Not taken, for a person to rule on" price the option as
"adding it would make every CMake consumer work unaided, at the cost of a
consumer that pins `LINKER_LANGUAGE CXX`". Measured against the real export, that
consumer is already broken today: `target_link_options(heongpu INTERFACE
-fgpu-rdc)` at `src/CMakeLists.txt:269` carries no generator expression, so it
lands on every consumer link line, and a target with `LINKER_LANGUAGE CXX` in a
HIP-enabled project fails now with ``unrecognized command-line option
`-fgpu-rdc'`` before `--hip-link` would ever be reached. Adding `--hip-link` to
the interface costs that consumer a second message on an already-failing link,
not a working build.

Deferring the code change itself is the right call for a prose round -- it
changes what every consumer's link line contains and would want its own consumer
rebuild -- but the rationale as recorded understates the case for doing it and
misses the larger question it is entangled with, which finding 1 measures: the
unguarded interface `-fgpu-rdc` is what breaks every CXX-driven consumer, so the
decision a person actually faces is whether the interface options should be
conditioned on the link driver at all, not just whether `--hip-link` joins them.
Re-register the item with that framing (`utils/deferred.py`), citing the rows
above, so the ruling is made on the measured trade.

### Recommendation

**Request Changes.** Both findings are prose or registry only; the port code is
unchanged and remains correct. Round 8 should be a single edit pass over
`.rst:76`, `.rst:83`, `src/CMakeLists.txt:261`, `fault-classes.md:374-394`, and
the deferral summary.

## Round 8 fix 2026-08-12 (linux-gfx90a, 9f9fd0b)

Addresses both round-8 findings. Prose only in the fork
(`docs/advanced_topics.rst`, `src/CMakeLists.txt`), plus the skill's
`fault-classes.md` and a reframed deferral in this repository.
`moatlib classify HEonGPU f657723 9f9fd0b` reports
`class=comment-only arch_independent=True inert=True`, so no rebuild of the
library and no new GPU obligation. `classify 5d99b8f 9f9fd0b` is still
`mixed`/not inert and both platforms are recorded validated at `5d99b8f`, so
both archs still owe the 20 suites at head for the `-fgpu-rdc` switch itself;
this commit adds nothing to that and nothing on the branch claims otherwise.

Rebuilt the consumer matrix from scratch in `agent_space/round8/` (gitignored):
twelve independent source trees, each configured and built on its own, all
against the REAL installed export at `agent_space/heongpu-consumer/prefix`
(ROCm 7.2.1, gfx90a, CMake 4.0.3, `CMAKE_CXX_COMPILER=/usr/bin/c++`,
`main.cpp` = `int main(){return 0;}` with
`-Wl,-u,_ZN7heongpu10modInverseEmm` to force the device-carrying object in).
Driver read from `CMakeFiles/app.dir/link.txt`.

| # | consumer project | shape | driver | result |
| --- | --- | --- | --- | --- |
| 1 | `LANGUAGES CXX HIP` | plain C++ exe, links `HEonGPU::heongpu` | rocm clang++ | undefined `__hip_fatbin_1f6ece6550d50f9a` / `__hip_gpubin_handle_` |
| 2 | `LANGUAGES CXX HIP` | + `--hip-link` | rocm clang++ | links |
| 3 | `LANGUAGES CXX` | same target | `/usr/bin/c++` | ``unrecognized command-line option `-fgpu-rdc'`` |
| 4 | `LANGUAGES CXX` | + `--hip-link` | `/usr/bin/c++` | both options unrecognized |
| 5 | `LANGUAGES CXX` | + `find_package(hip REQUIRED)` + `hip::host` | `/usr/bin/c++` | ``unrecognized command-line option `-fgpu-rdc'`` |
| 6 | `LANGUAGES CXX` | + `LINKER_LANGUAGE HIP` + `--hip-link` | -- | GENERATE FAILS: `Missing variable is: CMAKE_HIP_LINK_EXECUTABLE` |
| 7 | `LANGUAGES CXX` + `enable_language(HIP)` | + `--hip-link` | rocm clang++ | links |
| 8 | `LANGUAGES CXX HIP` | target pins `LINKER_LANGUAGE CXX` | `/usr/bin/c++` | ``unrecognized command-line option `-fgpu-rdc'`` |
| 9 | `LANGUAGES CXX HIP` | one own source `LANGUAGE HIP` (the documented snippet) | rocm clang++ | links |
| 10 | `LANGUAGES CXX` | archive named by path | `/usr/bin/c++` | undefined `__hip_fatbin_*` / `__hip_gpubin_handle_*` |
| 11 | `LANGUAGES CXX HIP` | by path + `LINKER_LANGUAGE HIP` + `--hip-link` + `-fgpu-rdc` | rocm clang++ | links |
| 12 | `LANGUAGES CXX` | same as 11 | -- | GENERATE FAILS, as row 6 |

The review's finding 1 reproduces exactly (rows 1, 3, 4, 5). Row 6 is the piece
that was not measured before and it decides how the fix had to be written: in a
project that never enabled HIP, `LINKER_LANGUAGE HIP` is not a remedy at all --
CMake has no HIP link rule to name, so generation fails before any compiler
runs. Rows 4 and 6 together mean a CXX-only consumer has NO target-level
remedy, and rows 2 and 7 mean the only remedy is to enable the language. So the
`.rst` now states the language requirement first, as a requirement rather than
a suggestion, keeps `--hip-link` as the remedy for a target with no HIP source
of its own, and says plainly that the library cannot rescue a consumer that
will not enable HIP. Row 11 keeps the by-path sentence true, now explicitly "in
a project where HIP is enabled" (row 12 is why). Every row of the round-7 matrix
was measured in `project(round7 LANGUAGES CXX HIP)`, which is why the condition
was invisible in that round.

Row 8 is finding 2: `target_link_options(heongpu INTERFACE -fgpu-rdc)` carries
no generator expression while the matching compile option is guarded by
`$<COMPILE_LANGUAGE:HIP>`, so a target pinning `LINKER_LANGUAGE CXX` already
fails today. The deferral `heongpu-hip-link-interface-option` was reframed
around that: the question is whether the interface options should be
conditioned on the link driver at all, and what a consumer that cannot enable
HIP should do. `deferred.py` has no edit or supersede subcommand and `add`
refuses a duplicate id, so the summary and refs of the (open, unruled) item were
rewritten in place through `deferred._load_project` / `_save_project`, i.e. the
same code path the CLI uses. Worth a small tooling follow-up if reframing an
unruled deferral happens again.

Promoted to the skill (`fault-classes.md`), since the next porter writing their
own consumer docs is exactly who repeats this: the enabled-language
precondition ahead of the driver/line split, a table row for the CXX-only
project with no target-level remedy, and the note that an INTERFACE link option
is unguarded unless you guard it, so `-fgpu-rdc` reaches a `LINKER_LANGUAGE
CXX` consumer while the compile option beside it does not.

## Review 2026-08-12 (round 9, linux-gfx90a, 9f9fd0b)

Scope: the delta `f657723..9f9fd0b` (one comment-and-docs commit) plus this
branch's `fault-classes.md` edit and the reframed deferral. Everything earlier
is cleared. Measured on this host against the REAL installed export at
`agent_space/heongpu-consumer/prefix` (ROCm 7.2.1, gfx90a, CMake 4.0.3,
`CMAKE_CXX_COMPILER=/usr/bin/c++`, `main.cpp` = `int main(){return 0;}`,
`-Wl,-u,_ZN7heongpu10modInverseEmm`); reproduction in `agent_space/revcheck9/`
(gitignored).

Confirmed, do not redo. Row 6 reproduces exactly: `project(t6 LANGUAGES CXX)` +
`LINKER_LANGUAGE HIP` + `--hip-link` fails at generate time with
`Missing variable is: CMAKE_HIP_LINK_EXECUTABLE`, so `LINKER_LANGUAGE HIP` is
indeed no fallback in a CXX-only project. Row 7 reproduces: `LANGUAGES CXX` +
`enable_language(HIP)` + `--hip-link`, no HIP source of its own, links (driver
`/opt/rocm/lib/llvm/bin/clang++`), which is the measurement behind "enabling the
HIP language does not oblige you to compile any of your own sources as HIP". Row
4 reproduces: `c++: error: unrecognized command-line option '--hip-link'` and
the same for `-fgpu-rdc`. `jargon.py --port HEonGPU` clean over the whole
branch, `prose.py` clean on the body, title 54 chars with `[ROCm]`,
AI-assistance disclosure present, no agent trailer, ASCII only, no internal
account references, fork tree clean, exactly one
`heongpu-hip-link-interface-option` entry.

### 1. "the library cannot rescue the link from its side" is false as stated; the producer-side remedy is `cmake/Config.cmake.in`

`docs/advanced_topics.rst:83` says "If your project does not enable HIP, the
library cannot rescue the link from its side"; `fault-classes.md:377` says
"neither does anything the producing library can put on its export";
`fault-classes.md:400-403` concludes "a consumer that cannot enable the HIP
language cannot link the archive at all, and the honest thing for the library's
docs to say is that enabling it is a requirement"; the body of `9f9fd0b` says
"The library cannot rescue such a consumer from its own side, so the
documentation now says so plainly".

The producing package can enable the language in the consumer's scope, because
`HEonGPUConfig.cmake` is included at the caller's file scope. Measured: a copy
of the installed config with `enable_language(HIP)` as its first line, consumed
by `project(tcfg LANGUAGES CXX)` with a plain C++ executable and only
`target_link_options(app PRIVATE --hip-link)`:

| consumer | config | driver | result |
| --- | --- | --- | --- |
| `LANGUAGES CXX`, `--hip-link` | as shipped | `/usr/bin/c++` | ``unrecognized command-line option `-fgpu-rdc'`` |
| `LANGUAGES CXX`, `--hip-link` | + `enable_language(HIP)` | rocm clang++ | configure rc=0, build rc=0, links |
| same, `find_package` called from `add_subdirectory` | + `enable_language(HIP)` | rocm clang++ | configure rc=0, build rc=0, links |

Only the config file differed; both runs used the same real prefix on
`CMAKE_PREFIX_PATH`. That line would live in `cmake/Config.cmake.in`, in this
repository, one file away from the code the sentence is about. There are real
reasons a maintainer might refuse it -- `enable_language` may not be called in a
function call and must be called in the highest directory common to all targets
using the language (CMake 4.0 `enable_language.rst:18,21`), so a consumer that
wraps `find_package` in a function hard-errors, and every consumer then pays HIP
compiler detection -- but those are reasons to decline an option, not grounds
for saying it does not exist.

`src/CMakeLists.txt:263-265` is the one place that scopes this correctly ("cannot
be helped from here"), and that is accurate: the remedy is not a target property.

Fix, smallest form. In the `.rst`, say what the shipped package does rather than
what the library can do ("the installed package does not enable the language for
you"), keeping the consumer-facing advice exactly as it is. In
`fault-classes.md`, drop the "neither does anything the producing library can put
on its export" clause and the "cannot link the archive at all" conclusion, and
replace them with the measured pair: nothing on the CONSUMER's target rescues
this, while the PRODUCER can enable the language from its package config, at the
documented cost -- a future porter is on the producer side, so the option they
are told does not exist is the one they own. Do not amend `9f9fd0b`; let the
follow-up commit body carry the corrected statement.

### 2. The deferral asks the person a question whose option set is missing the producer-side answer

`deferred.json` `heongpu-hip-link-interface-option` asks "what should a consumer
that cannot enable the HIP language do, given that nothing set on its own target
rescues the link". The target-level half is accurate and measured. But the
person ruling this also owns `cmake/Config.cmake.in`, and with the option above
missing the natural ruling is "document the requirement", which is a decision
taken without knowing an alternative exists. Add one sentence recording the
measurement and its cost. Everything else about the reframing is right: the
question really is whether the interface options should be conditioned on the
link driver at all, `$<LINK_LANGUAGE:HIP>` really is not the answer alone, and
the old cost estimate really was wrong.

### 3. The `deferred.py` tool gap is recorded only in notes prose

Rewriting the summary and refs of an open, unruled item in place through
`deferred._load_project`/`_save_project` was the right call: two overlapping
entries in front of the person ruling would be worse than one current one, the
item was unruled, the CLI has no edit or supersede path, and the refs still
point at both round-7 and round-8 notes, so the history is not lost. No change
wanted there. But "worth a small tooling follow-up" in a 2800-line `notes.md` is
not registered work, and this gap is control-plane, not HEonGPU's. Register it
in the global registry (`utils/deferred.py add`, no `--project`) so it survives
this port: `deferred.py` has no way to correct an open item, and `add` refuses a
duplicate id.

### What I would have changed but am not asking for

`fault-classes.md:393-394` -- the imported-archive and by-path rows carry no
"in a HIP-enabled project" qualifier while the three rows above them do, and
their remedy (`LINKER_LANGUAGE HIP`) is exactly the one that is unavailable
without the language. The sentence at `:396-399` states that immediately below
the table, so a reader gets it; I would still have put the qualifier in both
rows, since an unqualified row copied out of a table is how rounds 6 and 7 went
wrong. `fault-classes.md:411-414` has a ragged rewrap ("Do NOT try to fix / any
of this by / wrapping the interface flag") left over from the edit.
`docs/advanced_topics.rst:76` still says "every CMake target that links it ... is
driven by the HIP compiler", which row 8 (a target pinning `LINKER_LANGUAGE
CXX`) contradicts; that is the subject of the open deferral and does not belong
in consumer documentation before it is ruled.

Otherwise the round-8 fix is correct: the `.rst` leads with the language
requirement, keeps `--hip-link` scoped to a HIP-enabled project's target with no
HIP source, quotes the real generate-time message, and scopes the by-path
sentence; `src/CMakeLists.txt` matches it without overstating; and the skill's
enabled-language precondition ahead of the driver/line split, its new CXX-only
row, and its unguarded-INTERFACE-link-option lesson are all measured and in the
place a reader with that problem would look.

Verdict: changes-requested. Round 9 is a single edit pass over `.rst:83`,
`fault-classes.md:377` and `:400-403`, one sentence in the deferral summary, and
one `deferred.py add` in the global registry.

## Round 9 fix 2026-08-12 (linux-gfx90a, beba427)

Addresses all three round-9 findings. Docs, skill and deferrals only; no code.
`moatlib classify HEonGPU 9f9fd0b beba427` reports
`class=doc-only arch_independent=True inert=True`, so no rebuild and no new GPU
obligation. `classify 5d99b8f beba427` is still `mixed`/not inert and both
platforms are recorded validated at `5d99b8f`, so both archs still owe the 20
suites at head for the `-fgpu-rdc` switch; this commit adds nothing to that.

Verified the producer-side claim myself before writing it, in
`agent_space/round9/` (gitignored), against the REAL installed export at
`agent_space/heongpu-consumer/prefix` (ROCm 7.2.1, gfx90a, CMake 4.0.3,
`CMAKE_CXX_COMPILER=/usr/bin/c++`, `main.cpp` = `int main(){return 0;}`,
`-Wl,-u,_ZN7heongpu10modInverseEmm`). The alternate prefix is that same tree
with every entry symlinked except `lib/cmake/HEonGPU-1.1`, which is a copy whose
`HEonGPUConfig.cmake` differs from the shipped one by a single first line,
`enable_language(HIP)` (`diff` of the remainder is empty). Consumer sources
identical across cases; driver read from `CMakeFiles/app.dir/link.txt`.

| case | consumer | config | configure | build | driver |
| --- | --- | --- | --- | --- | --- |
| A | `project(consumer LANGUAGES CXX)`, plain C++ exe, `--hip-link` | as shipped | rc=0 | rc=2, ``unrecognized command-line option `-fgpu-rdc'`` | `/usr/bin/c++` |
| B | same source tree | + `enable_language(HIP)` | rc=0 | rc=0, 1067096-byte binary | `/opt/rocm/lib/llvm/bin/clang++` |
| C | same, `find_package` from `add_subdirectory` | + `enable_language(HIP)` | rc=0 | rc=0, same binary size | `/opt/rocm/lib/llvm/bin/clang++` |
| D | same, `find_package` wrapped in a `function()` | + `enable_language(HIP)` | rc=0 | rc=2, `--hip-link` AND `-fgpu-rdc` unrecognized | `/usr/bin/c++` |

So the review's finding 1 reproduces exactly: the producing package CAN enable
the language in the consumer's scope, and the sentence that said otherwise was
false. Case D is the one place my measurement differs from the review's
description of the cost: CMake 4.0's `enable_language.rst:18` says the call must
be in file scope "not in a function call", and the review expected a hard error,
but CMake 4.0.3 does not error here -- configure still returns 0 and the
language simply does not take effect, so the consumer gets the same failing C++
link as case A with no diagnostic pointing at the cause. A silent no-op is a
worse cost than a hard error, and the deferral and the skill both say "silently
gets no language" rather than "hard-errors" because of D. The other cost is
measured too: configure of the same tree goes from 0.88-0.91s (shipped) to
2.92-3.01s (with `enable_language(HIP)`), i.e. roughly two seconds of HIP
compiler detection imposed on every consumer, twice each, cold build dir.

Changes made:

- `docs/advanced_topics.rst:83` now says "the installed package does not enable
  the language for you, and nothing set on the target itself makes up for it",
  which is what is true; the consumer-facing advice after it is untouched. The
  producer-side option is not offered in consumer documentation because it is
  unruled -- it lives in the deferral, where the person who owns
  `cmake/Config.cmake.in` will see it.
- `fault-classes.md:377` drops "neither does anything the producing library can
  put on its export" and points at the producer paragraph; `:400-417` replaces
  "cannot link the archive at all" with the measured producer remedy, its two
  costs, and the note that if you decline it your docs owe the requirement. The
  next porter is on the producer side, so this is the option they own.
- `fault-classes.md:394-395`: added the "in a HIP-enabled project" qualifier to
  the imported-archive and by-path rows (the reviewer's optional item; agreed --
  their remedy is `LINKER_LANGUAGE HIP`, which is exactly what needs the
  language) and rewrapped the ragged lines left from round 8.
- `deferred.json` `heongpu-hip-link-interface-option`: one added passage giving
  the producer-side option, its measurement and its costs, so the ruling is not
  made blind to it. Rewritten in place again through
  `deferred._load_project`/`_save_project`, still exactly one entry, still open
  and unruled, refs extended to this section.
- The `deferred.py` tool gap is now registered GLOBALLY as
  `deferred-py-no-edit-or-supersede-path` (control-plane, not this port's): no
  edit or supersede subcommand, `add` refuses a duplicate id, statuses are only
  open/filed/done, so correcting an open unruled item has no supported path.

The commit body of `9f9fd0b` repeats the false statement and cannot be amended
(it is upstream-visible history and `5d99b8f` must stay an ancestor), so
`beba427`'s body carries the corrected statement explicitly and says so.

## Round 10: the -fgpu-rdc design is reverted, by ruling (2026-08-13, linux-gfx90a, 6ac06d0)

Not a review finding. Jeff Daily ruled the deferral
`heongpu-hip-link-interface-option` `now` (`--choice now --by jeffdaily`,
recorded in `deferred.json`): drop `-fgpu-rdc` entirely and go back to the
compiler guard, rather than condition the interface options. The reasoning as
given, and it is the reasoning to quote to anyone who asks why the smaller diff
was not kept:

- `-fgpu-rdc` bought a smaller footprint -- `small_ntt.cuh` and `small_ntt.cu`
  byte-identical to the unmodified project -- and paid with a requirement on
  downstream consumers that the project does not otherwise impose. `git log -S`
  on `target_link_options(heongpu INTERFACE` finds its first appearance in
  `8ef207a`, i.e. ours; nothing had ever been on this library's link interface,
  and the CUDA path puts nothing there either.
- Rounds 6-9 measured the consequence exactly: a consumer whose own project does
  not enable the HIP language cannot link the archive at all and has no
  target-level remedy (round-8 matrix rows 3-6, notes.md:2830).
- The guard has zero CMake surface and therefore zero consumer impact. Two
  modified files is the cheaper of the two prices.

The measured knowledge from rounds 5-9 stays true and stays recorded; only the
choice changed. Do not re-open the deferral and do not re-argue `-fgpu-rdc` on
this project.

### What was reverted, and what was kept

Worked out by diffing `4925df1` (the last guard commit) against `beba427`, not
from a list. Reverted in `6ac06d0`:

- `src/CMakeLists.txt`: the `PRIVATE -fgpu-rdc` compile option in
  `heongpu_set_gpu_properties`, the `INTERFACE` compile/link options and their
  comment block, and `lib/kernel/small_ntt.cu` back out of
  `HEONGPU_KERNEL_SOURCES`. The file is now byte-identical to `4925df1`.
- `src/include/heongpu/kernel/small_ntt.cuh`: restored from `4925df1` (the guard
  form), so `small_ntt.cu` is deleted again.
- `docs/advanced_topics.rst`: the three consumer-facing HIP paragraphs about
  enabling the HIP language, `--hip-link` and the by-path shape. The file is now
  byte-identical to `4925df1`. The CUDA `CUDA_SEPARABLE_COMPILATION` guidance is
  the project's own and was never touched.

Kept, because it has nothing to do with the design: the cuRAND-include removal
and its rewritten comment in `cuda_to_hip.h` (`d7d609e`, `81176c9`), everything
from rounds 1-4, and the MI250X/MI250 hardware naming (no fork file ever named a
GPU; that correction lives in commit bodies and in these notes).

One deliberate departure from a verbatim restore: the guard comment in
`small_ntt.cuh` at `4925df1` justified the guard by claiming `-fgpu-rdc` leaves
each butterfly an uninlinable cross-unit call. That is FALSE -- measured in
round 5, retracted in `8ef207a`, reproduced twice by reviewers -- so restoring
it verbatim would have re-shipped a known-false statement. The comment now
carries the real reason: RDC works and the device link still inlines, but it
leaves no complete device image in the archive, so every consumer has to enable
the HIP language and link through the HIP driver, and keeping the definitions
here asks nothing of consumers.

### The plain consumer, measured (this is what the ruling was about)

Fresh install of `6ac06d0` at `agent_space/heongpu-r10/prefix` (configured with
`-DCMAKE_INSTALL_PREFIX`; `cmake --install --prefix` does not override this
project's baked-in destinations). No `-fgpu-rdc` and no `INTERFACE_LINK_OPTIONS`
appear anywhere under `prefix/lib/cmake` now;
`IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "HIP"` is still there, as CMake
writes it for any static library built from HIP sources, and it is harmless
without the flag.

| consumer | shape | driver | result |
| --- | --- | --- | --- |
| `project(doc_consumer LANGUAGES CXX HIP)`, `main.cpp` `LANGUAGE HIP`, BFV encrypt/decrypt (the documented snippet verbatim) | real export | `/opt/rocm/lib/llvm/bin/clang++` | configure rc=0, build rc=0, `roundtrip OK` |
| `project(plain_consumer LANGUAGES CXX)`, plain C++ exe, `target_link_libraries(app PRIVATE HEonGPU::heongpu)` and nothing else -- no `enable_language(HIP)`, no `--hip-link`, no `LINKER_LANGUAGE` | real export | `/usr/bin/c++` | configure rc=0, build rc=0, runs, exit 0 |

The plain row is rows 3/4/5 of the round-8 matrix, which failed with
``unrecognized command-line option `-fgpu-rdc'`` under the RDC design and had no
target-level remedy. It now works with nothing added.

The plain consumer cannot include the HEonGPU headers (they pull in rocThrust,
which the project's own `.rst` already states), so its `main.cpp` declares two
host functions and calls them: `heongpu::is_prime(97)=1`, `is_prime(100)=0`,
`calculate_bit_size(1024)=11`. That still exercises the thing that matters,
because those symbols live in `util.cu.o`, which carries a device bundle -- and
`llvm-objdump --offloading` on the resulting executable shows
`app.0.hipv4-amdgcn-amd-amdhsa--gfx90a`, i.e. a complete gfx90a code object
reached the binary through a link driven by `/usr/bin/c++`. That is the property
`-fgpu-rdc` destroys.

Reproduction: `agent_space/r10-consumers/{doc,plain}` (gitignored).

### Verification of this round

```
cmake -S projects/HEonGPU/src -B projects/HEonGPU/src/build -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release \
  -DHEonGPU_BUILD_TESTS=ON -DHEonGPU_BUILD_EXAMPLES=ON \
  -DHEonGPU_BUILD_BENCHMARKS=ON \
  -DCMAKE_INSTALL_PREFIX=agent_space/heongpu-r10/prefix   # rc=0, empty build dir
cmake --build projects/HEonGPU/src/build -j64             # rc=0, 0 error lines, 64.7s
ctest --test-dir projects/HEonGPU/src/build               # 100% passed, 20/20, 13.25s
bash agent_space/heongpu-cuda-check.sh                    # see below
python3 utils/codeobj_diff.py agent_space/heongpu-amd-baseline/bin \
                              projects/HEonGPU/src/build/bin
```

`codeobj_diff.py` reports **verdict=identical**, all 42 binaries, exported
symbols and device ISA both. The baseline snapshot is the `4925df1` guard build
(round-5 measurement, still on disk), so this is direct evidence that the revert
reproduces the pre-RDC codegen exactly rather than approximately. It also
retires the round-5 concern in the other direction: whatever the RDC switch did
to every binary's ISA is now undone.

CUDA no-regression: run twice. The first run was incremental and rebuilt only
what the source change touched, so it was re-run after `touch`ing
`src/include/heongpu/cuda_to_hip.h` to force every TU on that include path.
That run (`=== START 2026-08-12T23:57:58Z ===` in
`agent_space/heongpu-cuda-check.log`): CONFIGURE rc=0, BUILD rc=0, **37 CUDA +
44 CXX objects rebuilt, 42 executables linked, 0 lines matching `error`**. Both
regressions this branch fixed stay fixed and both were actually recompiled in
that run: `lib/heongpu.cpp.o` (regression 1, the cuRAND include order) is in the
rebuild list, and every host TU reaching `small_ntt.cuh` (regression 2, device
bodies parsed by the host compiler) was rebuilt with the guard back in place.
Stale `heongpu_kernel.dir/lib/kernel/small_ntt.cu.o*` from the RDC-era build was
deleted first; CMake would not have linked it, but it confused the file listing.

### Revalidation accounting

`advance-head` -> `6ac06d0575ec210f8dbfa1123aa890d2a04a9938`.

```
moatlib classify HEonGPU 5d99b8f 6ac06d0 -> class=mixed arch_independent=False inert=False
moatlib classify HEonGPU beba427 6ac06d0 -> class=mixed arch_independent=False inert=False
```

Both archs sit at `validated_sha=5d99b8f` and owe a fresh 20-suite run at
`6ac06d0`. Do NOT carry forward the round-5..9 obligation as if nothing moved:
the code at head is not the code either arch last validated, and it is also not
the RDC code the last four rounds discussed. The one shortcut that IS available
to a validator: the AMD binaries at `6ac06d0` are bit-identical in device ISA to
the `4925df1` build, which linux-gfx90a ran at 20/20 on 2026-08-12 -- that is
evidence, not a substitute for the run.

`jargon.py --port HEonGPU`: clean over the whole branch.
`prose.py` on the commit body: clean. Title 63 chars.
`git -C projects/HEonGPU/src status --porcelain`: empty.

### Skill

`cuda-to-rocm` kept every measured fact from rounds 5-9 and was restructured so
it no longer reads as "`-fgpu-rdc` is the answer". `fault-classes.md` now opens
the class with two numbered options, says performance does not decide between
them (both measured, 20/20 and equal benchmarks), and says what does: whether
the project ships an installed library to strangers. The consumer-contract entry
is unchanged in substance -- it is the most expensive knowledge on this branch
and the next porter WILL reach for RDC -- but it is now framed as the cost of
option 1 and ends with what that cost bought here. The header-move entry records
the full guard -> RDC -> guard path and its lesson, and the codegen entry gains
the revert-check trick (`codeobj_diff.py` against a pre-switch `bin/` snapshot
should say `identical`). `SKILL.md`'s one-liner presents both options with the
deciding question.

## Review 2026-08-13 (round 11, linux-gfx942, 6ac06d0)

Scope: the unreviewed delta `9f9fd0b..6ac06d0` on the fork (`beba427`, the round-9
docs correction, and `6ac06d0`, the ruled `-fgpu-rdc` revert), plus this branch's
`cuda-to-rocm` edits (`c325161`, `2360d9f`, `c1747cc`) and both deferral writes.
Reviewed from a fresh clone of `moat-port` on a gfx942 host, so no build and no
GPU run here; both required gates owe a 20-suite run at head regardless (see the
round-10 revalidation accounting). Verdict: **review-passed**, no change round.

Confirmed independently, do not redo:

- The revert is exactly what the round-10 section claims. `git diff 4925df1 HEAD`
  is two files: `cuda_to_hip.h` (the kept cuRAND-include work from `d7d609e` /
  `81176c9`) and the rewritten guard comment in `small_ntt.cuh:29-35`.
  `src/CMakeLists.txt` and `docs/advanced_topics.rst` are byte-identical to the
  last guard commit, so nothing of the RDC design survives at head.
- The restored definitions match the deleted `small_ntt.cu` bodies line for line,
  with `inline` added at the definitions and the declarations left unconditional
  (`small_ntt.cuh:13-22` vs `:38-154`) -- the shape the skill documents, and legal
  as a later inline definition after a non-inline declaration.
- Nothing anywhere in the fork still references `-fgpu-rdc`, `--hip-link`,
  `__hip_fatbin_*` or `lib/kernel/small_ntt.cu` outside that one comment; the two
  includers of the header (`keygeneration.cuh:14`, `bootstrapping.cuh:12`) both
  reach `util.cuh` -> `cuda_to_hip.h` first, so the round-10 CUDA no-regression run
  (touch `cuda_to_hip.h`, 81 objects, 42 executables) genuinely recompiled every
  host TU that parses this header. That is the one class an AMD build cannot catch.
- Hygiene over the delta: titles 63 and 52 chars, both `[ROCm]`; AI-assistance
  disclosure and Test Plan in both bodies; no agent trailer; no non-ASCII in the
  added comment; `jargon.py --port HEonGPU` clean over the whole branch;
  `prose.py` clean on both bodies; no AMD-internal account references.

### Not blocking, but fix at the next write on this branch

1. `projects/HEonGPU/deferred.json` `heongpu-hip-link-interface-option` carries
   the ruling and the ruled work has landed, yet `status` is still `open`. It no
   longer appears in `deferred.py pending` (that filters on `decided`), but it does
   appear in `deferred.py list --open` alongside the three genuinely open
   upstream-defect items, describing interface options that no longer exist in the
   code. `python3 utils/deferred.py set-status heongpu-hip-link-interface-option
   done --project HEonGPU`, the way `heongpu-cuda-no-regression-unrun` was closed.
   This branch has twice paid for a stale record left in place; this is a one-line
   version of the same hazard.

2. `fault-classes.md:367-370` sends the reader to "two entries below" for the guard
   shape and the two-phase-lookup reason; counting the bold entries after it, that
   material is three below (consumer contract, cross-TU inlining, then the
   header-move entry).

3. The deciding question in `fault-classes.md:373-379` -- does the project ship an
   installed library to strangers -- is right as far as it goes, but the evidence
   behind it comes from a consumer that cannot call this library at all: the plain
   C++ consumer measured in round 10 links and runs precisely because it declares
   two host functions instead of including the headers, which pull rocThrust and
   must be compiled as HIP (`docs/advanced_topics.rst:54`). Any consumer that
   actually uses the API therefore enables the HIP language anyway and lands on the
   table's first row, where `-fgpu-rdc` costs nothing. Worth one sentence in the
   entry: ask also whether the library's public headers already force a HIP compile
   on consumers, because where they do, option 1's contract is nearly free. This is
   an addition to the lesson, not a correction of a false claim, and it is not a
   reason to re-open the ruling on this port.

4. `small_ntt.cuh:33-34` states the strong form of the consumer requirement ("every
   consumer has to enable the HIP language"). It survives round 9's correction --
   the language does have to be enabled in the consumer's scope either way, and the
   producer-side `enable_language(HIP)` in `cmake/Config.cmake.in` only moves who
   writes the line -- so no change is asked for. `6ac06d0`'s body already scopes it
   correctly ("nothing set on the consuming target makes up for it"). Noted only so
   the next reader does not re-derive it as a contradiction.

### For whoever prepares the upstream PR, not for a porter round

The branch is 26 commits and seven of them (`8ef207a`, `56615ec`, `f657723`,
`9f9fd0b`, `beba427`, the `4925df1` comment they corrected, and `6ac06d0`) are an
internal design excursion that nets out to one rewritten comment. Every body is
honest and self-contained, and `moat-port` has never been published, so the history
could still be curated. I am not asking for a rewrite: presentation of a design the
maintainer never saw is a publication-time decision, and rewriting would move
`head_sha` again for no change in the tree. Decide it deliberately when the PR body
is written rather than by default.

## Validation 2026-08-13 (linux-gfx942, MI300X HF, ROCm 7.14) -- completed

Real-GPU validation of `moat-port` at `6ac06d0575ec210f8dbfa1123aa890d2a04a9938`
(the `review-passed` head after round 11; this arch's first validation,
`validated_sha` was null beforehand). GPU: `rocminfo`/`rocm-smi
--showproductname` report eight "AMD Instinct MI300X HF" dies (GFX Version
gfx942) on this host. ROCm toolchain is the conda `_rocm_sdk_devel` SDK
(`hipcc --version` -> `HIP version: 7.14.60850-0000000`, amdclang 23.0.0git),
already on `PATH`/`CMAKE_PREFIX_PATH` via the container environment; no manual
activation needed. Installed `libntl-dev` (missing on this host; `libgmp-dev`
and `libssl-dev` were already present) with `sudo apt-get install -y
libntl-dev`.

Local fork clone at `projects/HEonGPU/src` was already fast-forwarded to
`origin/moat-port` at `6ac06d0` before this session started. Clean build from
scratch (`rm -rf build` first):

```bash
cmake -S projects/HEonGPU/src -B projects/HEonGPU/src/build -DUSE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
    -DHEonGPU_BUILD_TESTS=ON -DHEonGPU_BUILD_EXAMPLES=ON \
    -DHEonGPU_BUILD_BENCHMARKS=ON \
    -DCMAKE_INSTALL_PREFIX=agent_space/heongpu-gfx942/prefix
## Validation 2026-08-13 (linux-gfx1100, Radeon Pro W7800, ROCm 7.2.3) -- completed

Revalidation of `moat-port` at `6ac06d0575ec210f8dbfa1123aa890d2a04a9938` (this
arch's `validated_sha` was `5d99b8f447895f5b34b35f856e654d65e69b390a`, the round-10
`-fgpu-rdc` revert moved head). GPU: `rocminfo` reports four "AMD Radeon Pro
W7800 48GB" (gfx1100, RDNA3/wave32); `rocm-smi`/`/opt/rocm/.info/version` reports
7.2.3.

`moatlib classify HEonGPU 5d99b8f 6ac06d0` -> `class=mixed arch_independent=False
inert=False`; `git diff --stat 5d99b8f 6ac06d0` touches exactly two files
(`cuda_to_hip.h`, `small_ntt.cuh`, the net of the whole guard->RDC->guard
excursion). This is a real structural change to `small_ntt.cuh` (device
definitions now behind `#if defined(__CUDACC__) || defined(__HIPCC__)`, with
always-visible forward declarations added), not a rename or comment reflow, so
per the carry-forward rule and this branch's own round-10 note ("owe a fresh
20-suite run... that is evidence, not a substitute for the run") this was a full
rebuild and real-GPU run, not a carry-forward.

```bash
cd projects/HEonGPU/src && git checkout moat-port && git merge --ff-only origin/moat-port  # -> 6ac06d0
for d in thirdparty/GPU-FFT thirdparty/GPU-NTT thirdparty/RNGonGPU; do (cd $d && git checkout -- . && git clean -fdx); done
rm -rf build
cmake -S projects/HEonGPU/src -B projects/HEonGPU/src/build \
    -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release \
    -DHEonGPU_BUILD_TESTS=ON -DHEonGPU_BUILD_EXAMPLES=ON -DHEonGPU_BUILD_BENCHMARKS=ON
cmake --build projects/HEonGPU/src/build -j$(nproc)
ctest --test-dir projects/HEonGPU/src/build --output-on-failure
```

Both wrapped in `utils/timeit.sh HEonGPU compile -- ...` / `utils/timeit.sh
HEonGPU test -- ...`. Configure rc=0; build rc=0, zero `error:` lines (warnings
only, same classes as the other two archs: `-Wunused-value` on `[[nodiscard]]`
HIP error codes and `-Wpass-failed` loop-unroll notices on the small-NTT
kernel, expected now that the guard restores the header-inlined definitions).
All 42 executables built (`find build/bin -type f -executable | wc -l` = 42:
15 tests, 24 examples, 3 benchmarks). `git -C projects/HEonGPU/src
status --porcelain` empty before and after (integrity gate).

`ctest`: **20/20 passed**, run twice back to back (13.21s and 15.45s).

### Examples, benchmarks, and the OpenMP/RDC regressions this branch fixed

`readelf -d` scanned over all 42 binaries under `build/bin`: **no `libgomp` in
any DT_NEEDED**, confirming the round-6/7 two-OpenMP-runtime fix still holds
on a from-scratch gfx942 build (this project has no wave-width or arch
dependence in that fix, but it had never been checked on this arch's build
tree before).

Ran and checked by inspection:
- `1_basic_bfv`, `2_basic_ckks`: exit 0, plausible decrypted/decoded values.
- `9_multi_stream_usage_way1` (the OpenMP example): exit 0.
- `15_basic_tfhe`: exit 0; manually verified all eight gate outputs bit-for-bit
  against their Boolean truth tables from the printed inputs (`Input1: 1,1,0,1,
  0,1,0,0`, `Input2: 1,0,1,0,1,1,1,0`, `Input3(control): 0,0,0,0,1,1,1,1`) --
  NAND, AND, NOR, OR, XNOR, XOR, NOT and MUX (control=0 selects Input2,
  control=1 selects Input1) all correct.
- `bootstrapping/3_ckks_bit_bootstrapping`: exit 0, decrypted values match
  expected within float noise (e.g. `EXPECTED:1 - ACTUAL:1.00002`).
- `mpc/1_multiparty_computation_bfv`: exit 0, plausible output.
- `benchmark/tfhe_benchmark`: exit 0, all eight gates report sane per-op
  timings (NAND/AND/NOR/OR/XNOR/XOR ~10.9-11.2ms, NOT ~0.006ms, MUX ~19.3ms --
  same shape as the gfx90a/gfx1100 runs, fastest of the three cards as
  expected for MI300X).

This is the third arch (third distinct wavefront-family/vendor-card
combination) to run the full suite at the `-fgpu-rdc` guard design (reverted
back from RDC in round 10), and the first at wave64 since the revert. Nothing
in the revert has any wavefront-width dependence by construction (it is a
link-model choice, not a kernel change), and this run confirms that rather
than assuming it.

### CUDA no-regression gate

Already recorded at this exact `head_sha` (`6ac06d0`) by the linux-gfx90a
round-10 session above ("CUDA no-regression: run twice ... 37 CUDA + 44 CXX
objects rebuilt, 42 executables linked, 0 lines matching `error`"). Per the
validator's per-`head_sha` rule, not re-run here. This host also has no CUDA
toolchain set up (only the conda `_rocm_sdk_devel` HIP toolchain is present),
consistent with the rule's expectation that the gate lands on whichever Linux
arch validates first.

### Jargon and documentation

`python3 utils/jargon.py --port HEonGPU`: clean.

Documentation confirmed present in the checked-out tree at this head, in the
project's own house style: `README.md` ("AMD GPUs (ROCm)" section, `-D
USE_HIP=ON` build instructions) and `docs/getting_started.rst` (ROCm
prerequisite, `-D USE_HIP=ON -D CMAKE_HIP_ARCHITECTURES=<target>` configure
line, note that tests/examples/benchmarks all build and run on AMD).

### Verdict

`linux-gfx942`: **completed** at `6ac06d0575ec210f8dbfa1123aa890d2a04a9938`.
First validation on this arch (previously unvalidated); wave64 gate now has a
second independent measurement at the current, reverted (guard) design head,
distinct from the earlier gfx90a wave64 evidence which was measured at the
pre-revert `5d99b8f`. 20/20 reproduced twice from a clean build; no `libgomp`
in any of 42 built binaries; TFHE gates independently truth-table-checked.
CUDA gate already recorded at this `head_sha` by gfx90a, not re-run.
HEonGPU test -- ...`. Build: 0 lines matching `error:`, `libheongpu.a` plus all
15 test executables plus examples/benchmarks produced. `ctest`: **20/20 passed**,
run twice back to back (11.65s and 11.47s). `git -C projects/HEonGPU/src
status --porcelain` empty before and after (submodules show only the
patch-applied diffs under `ignore = dirty`, as expected).

### CUDA no-regression gate

Already recorded at this exact `head_sha` (`6ac06d0`) by the round-10 session on
linux-gfx90a (notes.md, "Verification of this round": CONFIGURE rc=0, BUILD rc=0,
37 CUDA + 44 CXX objects rebuilt, 42 executables linked, 0 lines matching
`error`). Per the validator's per-head_sha rule, not re-run here.

### Jargon and documentation

`python3 utils/jargon.py --port HEonGPU`: clean, after creating (then deleting) a
local `main` tracking `origin/main` in the fork clone, same as the prior
gfx1100 validation required (the tool resolves the range by local branch name).
Documentation confirmed present and unchanged at this head: `README.md` "AMD
GPUs (ROCm)" section (`USE_HIP=ON`, `CMAKE_HIP_ARCHITECTURES=gfx90a` example) and
`docs/getting_started.rst` (ROCm prerequisite, HIP configure line, GPU
architecture mapping table).

### Verdict

`linux-gfx1100`: **completed** at `6ac06d0575ec210f8dbfa1123aa890d2a04a9938`.
20/20 reproduced twice from a clean build of the `-fgpu-rdc` revert; CUDA gate
already recorded at this `head_sha` by another arch, not re-run; jargon and docs
clean. No regression versus the prior `5d99b8f` completion on this arch.

## Validation 2026-08-13 (windows-gfx1151, Radeon 8060S, TheRock ROCm 7.13 dev) -- validation-failed

First build attempt on Windows for this project (previous windows-gfx1151 activity was
a read-only review, notes.md round-6/round-10 corroboration; no build had been tried).
`moat-port` at `6ac06d0575ec210f8dbfa1123aa890d2a04a9938`, cloned fresh to
`projects/HEonGPU/src`, `moat-port` checked out, verified at the expected head.

### Summary

The Windows toolchain gap the dispatch flagged ("rocThrust and OpenMP... the interesting
part") is now fully solved and repeatable -- see below. Configuration and HIP device
compilation for real GPU code (GPU-NTT, GPU-FFT, RNGonGPU) succeed. The build then hits
a genuine, first-time-discovered **source** portability bug in HEonGPU's own code: two
headers pull in the Linux-only `<sys/sysinfo.h>` and `src/lib/util/memorypool.cu` calls
`struct sysinfo` / `sysinfo(&memInfo)` unconditionally, which does not exist on Windows.
This is porter work, not something a validator edits. Setting `validation-failed`;
`failed_sha` = `6ac06d0575ec210f8dbfa1123aa890d2a04a9938`.

### The toolchain gap, solved (record this so nobody repeats the archaeology)

TheRock's installed Windows ROCm SDK on this host (`D:/Develop/TheRock/.venv`,
`rocm-sdk-core`/`rocm-sdk-devel`/`rocm-sdk-libraries-gfx1151` 7.13.0a20260511) is missing,
for Windows specifically:

- `lib/cmake/hip/` exists but is **empty** -- no `hip-config.cmake` at all, even though
  `amdhip64.dll`/`amdhip64.lib` and the HIP headers are present and working. So
  `find_package(hip REQUIRED)` fails out of the box.
- `hiprand_kernel.h` etc. are present under `include/hiprand`, but there is no
  `hiprand-config.cmake`/`hiprandConfig.cmake`.
- rocThrust (headers or CMake package) is **absent entirely**. `lib/cmake/hip-lang/`
  *does* have a working `hip-lang-config.cmake` -- a hand-authored stub, dated
  2026-06-18, clearly left by an earlier (catboost) validation session on this same
  host; it is real and reusable, not mine.
- Filed as `heongpu-windows-rocm-sdk-cmake-packages-missing` in `deferred.json`
  (`rocm-bug-report`, component `rocm-sdk-core`) -- this blocks ANY HIP CMake project
  that uses `find_package(hip/hiprand/rocthrust)` on Windows with this SDK, not just
  HEonGPU.

Workaround (throwaway local CMake package shims, `agent_space/heongpu-win/cmake-shims/`,
gitignored, not part of the port):
- `hip/hipConfig.cmake`: defines `hip::amdhip64` (SHARED IMPORTED, `IMPORTED_IMPLIB` =
  the real `_rocm_sdk_core/lib/amdhip64.lib`), `hip::host` (INTERFACE, links
  `hip::amdhip64`, `__HIP_PLATFORM_AMD__=1`), `hip::device` (INTERFACE, links
  `hip::host`) -- the minimal subset of the real
  `rocm-systems/projects/clr/hipamd/hip-config-amd.cmake.in` (found in the local
  TheRock source checkout) that this port actually uses (only `hip::host`, confirmed by
  grep across the whole tree).
- `hiprand/hiprandConfig.cmake`: `hip::hiprand` as an INTERFACE target with **no**
  library, because HEonGPU/RNGonGPU only use the header-only device RNG API
  (`hiprandState_t`, `hiprand_init`, `hiprand`), never the host generator API that would
  need a real `hiprand`/`rocrand` runtime lib.
- `rocthrust/rocthrustConfig.cmake`: `roc::rocthrust` as an INTERFACE include-only
  target pointing at the local TheRock super-repo checkout's
  `rocm-libraries/projects/{rocthrust,rocprim}` (rocThrust is a header-only Thrust
  fork; HEonGPU only pulls `thrust/host_vector.h`). No target in the tree links
  `roc::rocthrust` (only `find_package` gates on it), so the include path also has to
  be injected globally via `CMAKE_HIP_FLAGS` (see below) -- the shim target exists in
  case something ever does link it.
Passed via `-DCMAKE_PREFIX_PATH=...;<shims>/hip;<shims>/hiprand;<shims>/rocthrust`
plus explicit `-Dhip_DIR=`, `-Dhiprand_DIR=`, `-Drocthrust_DIR=` (belt and suspenders;
`find_package` in CONFIG mode needs the `_DIR` var or the package literally sitting at
`<prefix>/<name>Config.cmake`, and shim folders are named to match).

GMP, NTL, OpenSSL, ZLIB (`find_package`/`find_library` targets `gmp`, `ntl`,
`OpenSSL::Crypto`/`SSL`, `ZLIB::ZLIB`) have no TheRock/vcpkg-reachable MSVC-ABI build on
this host either (vcpkg has `gmp`/`openssl`/`zlib` ports but `ftpmirror.gnu.org` /
`ftp.gnu.org` / `gmplib.org` are unreachable from this network -- `curl` times out;
`github.com`, `pypi.org`, `conda.anaconda.org`, `api.anaconda.org` all work fine, so this
is a specific-domain block, not a general network outage). Fix: fetch the prebuilt
conda-forge win-64 packages directly (MSVC-toolchain-built, so MSVC-ABI compatible with
clang-cl) via the `api.anaconda.org` redirect (`.conda` files are zip archives containing
`pkg-*.tar.zst`; Windows's built-in `tar.exe` extracts `.zst` natively, no extra tool
needed):
```
curl -sSL -o gmp.conda "https://api.anaconda.org/download/conda-forge/gmp/6.3.0/win-64/gmp-6.3.0-hfeafd45_2.conda"
unzip -o gmp.conda -d gmp_extract && cd gmp_extract && tar -xf pkg-*.tar.zst
# -> Library/{include,lib,bin}: gmp.h, gmp.lib, libgmp-10.dll
```
Same recipe for `openssl` (4.0.1, `Library/lib/{libcrypto,libssl}.lib`) and `zlib`
(1.3.2, `Library/lib/{z,zdll,zlib,zlibstatic}.lib`; vcpkg's own build of zlib also
works and was already present on this host, no conflict). `GMP_ROOT`, `OPENSSL_ROOT_DIR`,
`ZLIB_ROOT` env vars point HEonGPU's root `CMakeLists.txt`'s GMP `find_library` and
CMake's own `FindOpenSSL`/`FindZLIB` at these trees (CMake 3.12+'s `CMP0074` makes
`find_path`/`find_library` honor `ENV{<Package>_ROOT}` automatically). `REQUESTS_CA_BUNDLE`/
`CURL_CA_BUNDLE=D:/cla-bundle.pem` was needed for `conan`/vcpkg's own https clients
(the corporate proxy cert) but not for `curl` itself once pointed at reachable hosts.

NTL has **no** Windows package anywhere (checked vcpkg's port list, conda-forge -- its
`win-64` platform list is conspicuously absent even though `linux-64`/`osx-*` are
current at 11.6.0). Built from source, entirely with clang-cl, in about a minute:
```
git clone --depth 1 https://github.com/libntl/ntl   # the project's own upstream mirror,
                                                       # NOT gnu.org/shoup.net (both unreachable)
```
NTL's own `configure`/`DoConfig` (a Perl script driving many small feature-probe
compiles with Unix-style flags) will not run under clang-cl, but NTL ships its own
**Windows/MSVC fallback path** that nobody has to invent: `src/dosify` (the script NTL's
own maintainer used to build the historical "WinNTL" distribution) reveals the whole
recipe --
- `src/mach_desc.win`: a precomputed machine-description header for 32-bit `long` (the
  Windows LLP64 model, which clang-cl also uses) -> copy to `include/NTL/mach_desc.h`,
  skipping the `MakeDesc` runtime probe entirely.
- `src/ResetFeatures <dir> "<FEATURES list from src/mfile>"`: generates all-off
  `HAVE_*.h` feature headers (`ALL_FEATURES.h` `#include`s each unconditionally, so they
  must exist even if none apply) -- run directly against `include/NTL`, no `dos/` copy
  needed.
- `include/NTL/PackageInfo.h` = `#define NTL_WINPACK (1)` (mirroring `dosify`'s own
  `echo`) -- this is the flag that activates NTL's already-written MSVC/no-`long long`
  fallback code paths in `ctools.h`/`tools.h` (`#if (!defined(NTL_HAVE_LL_TYPE) &&
  defined(NTL_WINPACK) && defined(_MSC_VER))`), so nothing else needs guessing.
- `GetTime.cpp`/`GetPID.cpp` = copies of `GetTime4.cpp` (portable `<ctime>` wall clock)
  and `GetPID2.cpp` (`return 0`) -- the same substitution `dosify` performs.
- `config.h`: hand-substitute `src/cfile`'s `@{...}` template (it is literally
  `s/@\{NAME\}/$ConfigSub{NAME}/` in `DoConfig`, no compiler probing involved for this
  part) with everything `0` except `NTL_STD_CXX14=1`.
- Compile the ~76 files in `src/mfile`'s `$(SRC)` list (plus `GetTime.cpp`/`GetPID.cpp`)
  with `clang-cl /std:c++14 /EHsc /O2 /MD -DNDEBUG` (the `/MD` matters -- one file built
  without it first, `lld-link` then refused to link the mismatched CRT with
  `/failifmismatch: mismatch detected for 'RuntimeLibrary'`), archive with `llvm-lib.exe`.
- Smoke-tested (`NTL::ZZ`, `NTL::RR`, `conv`, `%`, `to_long`) linking the resulting
  `ntl.lib` from a standalone `.exe`: correct output, exit 0.

Reusable pieces for a future session on this host: `agent_space/heongpu-win/env.sh`
(the full env: vcvars64 INCLUDE/LIB, the shim `CMAKE_PREFIX_PATH` entries, `GMP_ROOT`
etc.) and `agent_space/heongpu-win/cmake-shims/`. Both gitignored by design (scratch),
so they do not survive to a different host/clone -- this writeup is the durable copy of
the recipe. The GMP/OpenSSL/ZLIB/NTL trees themselves and the two `.conda` downloads
live under the session scratchpad
(`C:\Users\jdaily\AppData\Local\Temp\claude\...\scratchpad\deps`), not under
`agent_space/`, since scratchpad is the tool's designated temp area; a future session
should just re-run the recipe above rather than expect that path to exist.

### Two more Windows/clang-cl/HIP-language CMake quirks worth recording (workarounds, not blockers)

1. **`CMAKE_HIP_COMPILER_FORCED` breaks the HIP compile rule; the ABI-detection failure
   it works around has a narrower, correct fix.** `enable_language(HIP)` with
   `CMAKE_HIP_COMPILER=clang-cl.exe` hits `CMake Error ... MSVC compiler version not
   detected properly` inside CMake's OWN internal ABI-detection scratch project
   (`CMakeDetermineCompilerABI.cmake`'s `try_compile`, at
   `Modules/Platform/Windows-MSVC.cmake:69`). Root cause, confirmed by reading the
   module: `Windows-MSVC.cmake`'s `_compiler_version` fallback chain checks
   `CMAKE_C_SIMULATE_VERSION`, `CMAKE_CXX_SIMULATE_VERSION`, `CMAKE_Fortran_...`,
   `CMAKE_CUDA_...` -- **there is no `CMAKE_HIP_SIMULATE_VERSION` branch at all**. In a
   real project (`LANGUAGES C CXX HIP ASM`, like HEonGPU's), `CMAKE_CXX_SIMULATE_VERSION`
   is already set by the time HIP is processed, so this never fires. But CMake's own
   internal ABI-detection scratch project enables *only* HIP, so every branch in the
   chain is unset and it hits the `FATAL_ERROR`. Setting `-DCMAKE_HIP_COMPILER_FORCED=1`
   "fixes" this by skipping ABI detection entirely -- but that also skips whatever
   normally seeds `-x hip` into the HIP compile rule, so HIP sources silently compile as
   plain C++ (`-TP`, no `-x hip`) and every device intrinsic (`threadIdx`, `blockIdx`,
   `__syncthreads`) is "undeclared identifier". The actual fix: don't use `FORCED`; add
   `-x hip` directly to `CMAKE_HIP_FLAGS` (it appears after the rule's fixed `-TP` on the
   command line and clang resolves the language from the last `-x`/`-TP`-equivalent
   flag, so it wins) -- keep `FORCED` too since it's still needed to dodge the
   `Windows-MSVC.cmake` bug, just don't rely on it for anything else.
2. **`MSVC_RUNTIME_LIBRARY` target property has no entry for HIP in CMake 3.31 on
   Windows.** `CMake Error ... MSVC_RUNTIME_LIBRARY value 'MultiThreadedDLL' not known
   for this HIP compiler` at generate time, on the first HIP static-library target
   evaluated (`thirdparty/GPU-FFT/src/CMakeLists.txt`'s `fft` target). Confirmed this is
   not a policy (CMP0091 OLD/NEW) or generator-expression issue -- both were tried and
   both fail identically. `Help/prop_tgt/MSVC_RUNTIME_LIBRARY.rst` documents C, CXX,
   CUDA, OBJC, OBJCXX, Fortran; **HIP is absent from the property's documented language
   list entirely** in this CMake version, even though `Platform/Windows-Clang.cmake`'s
   `__windows_compiler_clang(HIP)` call does populate the
   `CMAKE_HIP_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_*` module-level table -- the
   property's internal (C++-implemented) consumer apparently does not route HIP through
   it regardless. Fix: `-DCMAKE_MSVC_RUNTIME_LIBRARY=""` (empty, not just unset) --
   this disables the property mechanism for every language rather than leaving CMake to
   pick per-language defaults, so the runtime library is instead selected by the plain
   `-D_DLL -D_MT ...` flags `Windows-Clang.cmake` already bakes into
   `CMAKE_<LANG>_FLAGS_RELEASE_INIT` etc. when `CMAKE_MSVC_RUNTIME_LIBRARY_DEFAULT` is
   unset (confirmed those flags are present in the surviving build; the NTL smoke test
   and the real HIP objects that DID compile all link the DLL CRT correctly).

Both are CMake-Windows-HIP-language quirks, not HEonGPU-specific and not ROCm-specific
(the second is arguably a CMake upstream gap, filed nowhere since `utils/deferred.py`'s
`rocm-bug-report` kind is for ROCm components, not CMake itself) -- promoted to the
`cuda-to-rocm` skill (see below) so the next Windows CMake-HIP port does not rediscover
either.

### The line-ending trap in `thirdparty/build.sh`'s patch application

Independent of everything else: `git config core.autocrlf` was `true` on this host (the
common Git-for-Windows default), which CRLF-converts every file on checkout including
`thirdparty/patches/*.patch` AND the three submodule checkouts the patches apply to.
This should be a no-op (both sides converted the same way) but is not, because the
*first* `cmake` configure attempt cloned the three submodules automatically
(`thirdparty/build.sh`'s `git submodule update --init --recursive`) while my own
`git checkout -- .` commands (run to fix an unrelated, since-superseded hypothesis)
re-checked-out files at different times under a config that had already been changed
mid-session, desynchronizing the two sides. `git -C thirdparty/RNGonGPU apply
thirdparty/patches/RNGonGPU.patch` failed with `patch does not apply` starting at a
real hunk (not a whitespace-only diff). Fixed by setting `core.autocrlf false` locally
in the fork clone and all three submodules (plus the nested
`RNGonGPU/thirdparty/GPU-NTT`), then forcing every patch file and every submodule
checkout to be re-materialized from the index (`rm <file>; git checkout -- <file>` --
plain `git checkout -- .` does not always rewrite a file whose working-tree content
already byte-matches under the OLD line-ending convention). This is a description of
what went wrong in THIS session's investigation, not a defect in `thirdparty/build.sh`
itself -- a fresh clone with a consistent `core.autocrlf` (any single value, applied
before the first configure) does not hit this; recorded so a future Windows session
does not have to re-derive it if the same symptom appears from a differently-timed
config change.

### The blocking finding: `<sys/sysinfo.h>` is Linux-only, used unconditionally

```
D:/.../src/include/heongpu/util/random.cuh(12,10): fatal error: 'sys/sysinfo.h' file not found
D:/.../src/include/heongpu/util/memorypool.cuh(13,10): fatal error: 'sys/sysinfo.h' file not found
```
`src/lib/util/memorypool.cu:59-60` is the actual use: `struct sysinfo memInfo;
sysinfo(&memInfo);` (glibc-only; queries total/free system RAM for a memory-pool sizing
heuristic). `random.cuh`'s `#include <sys/sysinfo.h>` at line 12 has no corresponding
call in that header itself -- worth the porter checking whether it is a leftover
unneeded include or feeds something in `random.cu` before assuming it is dead.
Nothing upstream (CUDA path) is Windows-relevant here (upstream never targeted Windows
either), so this is not a CUDA-vs-HIP regression, just a Linux-only host API that no
previous validator round could see (every prior validation was Linux). Needs an actual
`#ifdef`/platform-abstraction fix in the port (e.g. `GlobalMemoryStatusEx` from
`<windows.h>` on `_WIN32`, matching the existing `sysinfo()` call's fields) -- out of
scope for a validator to patch. Everything up to this point (submodule build,
configure, real HIP device compilation of GPU-NTT/GPU-FFT/RNGonGPU) is solid evidence
the toolchain side is no longer the blocker; this one small, precisely located gap is.

Secondary, not yet investigated past first sight: once past the above, `_deps/googletest-build`
(FetchContent) fails to build with `clang-cl: error: argument unused during
compilation: '-O3' [-Werror,-Wunused-command-line-argument]`. Plausible cause (not
confirmed): `CMakeLists.txt:44` appends `-O3` to `CMAKE_CXX_FLAGS_RELEASE` even though
`Platform/Windows-Clang.cmake` already appended `-O3` to the `_INIT` value for Release,
so the flag appears twice on every Release-config CXX command line; harmless everywhere
else (a redundant-flag warning) but GoogleTest's own bundled CMakeLists apparently turns
on `-Werror` for its own build, making the redundancy fatal there. Would need
confirming (and, if real, is a one-line CMakeLists.txt de-duplication) before treating
as a second blocking finding -- recorded so the porter checks it in the same round
rather than being surprised by a second error immediately after fixing the first.

### Verdict

`windows-gfx1151`: **validation-failed** at `6ac06d0575ec210f8dbfa1123aa890d2a04a9938`.
Not a GPU fault, not a Windows-permanent-impossibility (both `heongpu-windows-rocm-sdk-cmake-packages-missing`
and the two CMake-HIP-language quirks have describable fixes elsewhere, and the
`sys/sysinfo.h` fix is an ordinary small portability patch) -- this is a real port-source
gap discovered for the first time because no arch before this one ever attempted a
Windows build. No GPU test ran; do not read "20/20" anywhere above as applying here.
`wave64`/`wave32` gates remain satisfied by `linux-gfx942`/`linux-gfx1100`;
`pr_ready` is unaffected by this arch failing since `windows` is the only gate this
arch could have contributed and it did not close.

## Porter round 12 (2026-08-13, windows-gfx1151) -- Windows host portability, b12dc98

Narrow round answering the 2026-08-13 windows-gfx1151 validation failure. Fork commit
`b12dc985654c35a9ebf16ffefd6c7eb29da0e6c4` on top of the validated `6ac06d0` (no amend,
so `linux-gfx942`/`linux-gfx1100` flip to revalidate with their evidence already recorded).

### The diff (4 concerns, 8 files, +31/-11)

1. `src/lib/util/memorypool.cu` `get_host_avaliable_memory()`: added a `#ifdef _WIN32`
   branch using `MEMORYSTATUSEX` + `GlobalMemoryStatusEx()` (`ullAvailPhys`), with the
   POSIX `sysinfo()` branch left byte-for-byte identical under `#else`. `<windows.h>` is
   included at the very top of the .cu, before the project header, behind
   `WIN32_LEAN_AND_MEAN`/`NOMINMAX` guards -- it has to precede everything because
   memorypool.cuh drags in Thrust/rmm and the `min`/`max` macros would break them.
   `MEMORYSTATUSEX` is zero-initialized so a failed call yields 0 rather than garbage
   (upstream ignores `sysinfo()`'s return; the Windows branch matches that shape).
2. `src/include/heongpu/util/memorypool.cuh`: `#include <sys/sysinfo.h>` wrapped in
   `#ifndef _WIN32`. Kept in the header (not moved into the .cu) so no Linux translation
   unit loses a transitive include.
3. `src/include/heongpu/util/random.cuh`: the `<sys/sysinfo.h>` include was **deleted**,
   not guarded. Verified first: a grep for `sysinfo|freeram|mem_unit|totalram|get_nprocs|
   get_phys_pages|get_avphys_pages|SI_LOAD_SHIFT` over src/ test/ example/ benchmark/
   matches only memorypool.{cuh,cu} -- nothing in random.cuh, random.cu, or the three
   context.cuh files that include random.cuh uses anything from that header. Deleting a
   dead include is smaller than guarding it and cannot drift.
4. `CMakeLists.txt` (USE_HIP branch only, the CUDA `else()` untouched): the four
   `-g`/`-O3` appends now sit inside `if(NOT MSVC)`.

Plus, found while building and fixed in the same commit because the build cannot proceed
without it:

5. `u_int32_t` -> `uint32_t` (6 sites, 4 files: bfv/ckks `evaluationkey.{cuh,cu}`).
   `u_int32_t` is a BSD/glibc `<sys/types.h>` typedef with no Microsoft equivalent; it is
   the identical type on Linux, so the CUDA path is unaffected.

### Build result: the Windows build now completes

`cmake --build build -j6` exits 0. The library, the three thirdparty HIP static libs
(ntt-1.0, fft-1.0, rngongpu-1.0) and **all 15 gtest executables** in `build/bin/test/`
compile and link for gfx1151. This is the first time HEonGPU has built on Windows at all.
Examples and benchmarks were OFF for this round.

The failures were hit and cleared in this order (each a distinct class; recorded so
nobody re-derives them):

| # | Failure | Class | Fix |
|---|---------|-------|-----|
| 1 | `'sys/sysinfo.h' file not found` | port source | items 1-3 above |
| 2 | `clang-cl: error: argument unused during compilation: '-O3' [-Werror,...]` building `gtest-all.cc` | port build file | item 4 above |
| 3 | `unknown type name 'u_int32_t'` | port source | item 5 above |
| 4 | `lld-link: error: undefined symbol: __udivti3` linking the test exes | toolchain | link `clang_rt.builtins-x86_64.lib` from `lib/llvm/lib/clang/23/lib/windows/`; clang-cl does not link compiler-rt builtins by default and the host modular arithmetic uses 128-bit division |
| 5 | `/failifmismatch: mismatch detected for 'RuntimeLibrary'` (twice: HIP objs MT vs NTL MD, then HIP MD vs gtest MT) | toolchain | see below |

Failure 5 is the interesting one and is a direct consequence of the previous session's
`-DCMAKE_MSVC_RUNTIME_LIBRARY=""` workaround for the CMake 3.31 `MSVC_RUNTIME_LIBRARY`
HIP gap: emptying it disables CRT selection for **every** language, and clang-cl's
default with no flag is the *static* CRT, so nothing in the build agreed with the
dynamic-CRT conda-forge GMP/OpenSSL/ZLIB or the `/MD`-built NTL. The previous session's
claim that `Windows-Clang.cmake` still bakes `-D_DLL -D_MT` into the `_INIT` flags is
**wrong** for this CMake/compiler pair -- verified from the cache
(`CMAKE_CXX_FLAGS_RELEASE:STRING=/O2 /Ob2 /DNDEBUG`, no CRT flag anywhere) and from the
object directives lld-link reported. Correct workaround: pass the CRT flag explicitly to
all three languages -- `-DCMAKE_HIP_FLAGS="-x hip /MD ..."`,
`-DCMAKE_CXX_FLAGS="-DWIN32 -D_WINDOWS -EHsc -MD"`,
`-DCMAKE_C_FLAGS="-DWIN32 -D_WINDOWS -MD"`. `-Dgtest_force_shared_crt=ON` alone does NOT
help: googletest's logic only rewrites an existing `/MD` into `/MT`, so with no CRT flag
present there is nothing for it to rewrite. Use the dash spellings (`-MD`, `-EHsc`,
`-DWIN32`) from Git Bash: a leading-slash flag is mangled into a path by MSYS argument
conversion (`/DWIN32` became `C:/Program Files/Git/DWIN32`), and
`MSYS_NO_PATHCONV=1`/`MSYS2_ARG_CONV_EXCL='*'` is not a usable global escape here because
it also breaks `thirdparty/build.sh`'s `git -C` invocations.

Items 4 and 5 are toolchain/environment, not port defects: both are handled on the
configure command line and nothing in the fork changed for them. Promoted to the
`cuda-to-rocm` skill's Windows reference since they hit any CMake + HIP + clang-cl project.

### Runtime: hipMemGetInfo is unsupported on this gfx1151 APU (platform gap, NOT fixed here)

Smoke run of one built executable:
```
./bin/test/bfv_addition_testcases.exe
unknown file: error: C++ exception with description "CUDA Error in
  .../src/lib/util/memorypool.cu at line 90: invalid argument" thrown in the test body.
[  FAILED  ] HEonGPU.BFV_Ciphertext_Ciphertext_Addition_Subtraction (438 ms)
```
Lines 89-90 are `MemoryPool::get_decive_avaliable_memory()`'s `cudaMemGetInfo` ->
`hipMemGetInfo` followed by `HEONGPU_CUDA_CHECK(cudaGetLastError())`. Confirmed to be the
runtime and not this port with a 10-line standalone HIP program that links nothing of
HEonGPU (`agent_space/heongpu-win/meminfo.hip`):
```
hipMemGetInfo rc=1 (invalid argument) free=0 total=0
hipGetLastError rc=1 (invalid argument)
```
So `hipMemGetInfo` fails unconditionally on this integrated RDNA3.5 part with TheRock
ROCm 7.13.0a20260511 -- the same APU gap seen on this host in other projects.
Deliberately NOT worked around in this round: any fallback (for example sizing the device
pool from `hipDeviceProp_t::totalGlobalMem` when `hipMemGetInfo` fails) is a design
decision touching the shared CUDA path and belongs to a triaged round, not to a Windows
portability fix. It is the next thing standing between this port and a Windows GPU test
pass.

### Environment reuse

`agent_space/heongpu-win/env.sh` from the previous session worked unchanged except for one
addition (the compiler-rt builtins directory appended to `LIB`). The scratchpad `deps`
tree (GMP/OpenSSL/ZLIB/NTL) and `cmake-shims/` both survived. Effective configure line:
```
cmake -S . -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DHEonGPU_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release -G Ninja \
  -DCMAKE_C_COMPILER=<rocm>/lib/llvm/bin/clang-cl.exe \
  -DCMAKE_CXX_COMPILER=<rocm>/lib/llvm/bin/clang-cl.exe \
  -DCMAKE_HIP_COMPILER=<rocm>/lib/llvm/bin/clang-cl.exe -DCMAKE_HIP_COMPILER_FORCED=1 \
  -DCMAKE_MSVC_RUNTIME_LIBRARY="" \
  -DCMAKE_HIP_FLAGS="-x hip /MD -D_USE_MATH_DEFINES -DWIN32_LEAN_AND_MEAN -DNOMINMAX -D_WIN32_WINNT=0x0601 -I<rocthrust> -I<rocprim>/rocprim/include" \
  -DCMAKE_CXX_FLAGS="-DWIN32 -D_WINDOWS -EHsc -MD" \
  -DCMAKE_C_FLAGS="-DWIN32 -D_WINDOWS -MD" \
  -DCMAKE_EXE_LINKER_FLAGS="/machine:x64 clang_rt.builtins-x86_64.lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="/machine:x64 clang_rt.builtins-x86_64.lib" \
  -DCMAKE_PREFIX_PATH="<rocm>;<shims>/hip;<shims>/hiprand;<shims>/rocthrust" \
  -Dhip_DIR=<shims>/hip -Dhiprand_DIR=<shims>/hiprand -Drocthrust_DIR=<shims>/rocthrust
```
Line endings: this clone's working tree is CRLF while the index is LF (a stale stat cache
from the earlier `core.autocrlf` flip makes git report those files clean). Every file
edited this round was rewritten to LF before staging so the commit carries only the
intended hunks -- check `git diff --stat` after any edit in this clone before committing.

## Review 2026-08-13 (round 12, windows-gfx1151, b12dc98)

Scope: `git diff 6ac06d0 b12dc98` (+37/-11, 8 files), in the context of the whole
`moat-port` branch. Rounds 1-11 are review-passed and the `-fgpu-rdc` / `small_ntt`
question is settled by ruling at `6ac06d0`; neither was reopened. Verdict:
**changes-requested**. The code in the delta is sound and should stay as committed --
what fails review is the recorded analysis attached to it, plus a correctness failure
this review measured for the first time.

### Ruling 1: the Windows host-portability delta belongs in `moat-port`. Do not split it.

Explicitly asked and explicitly answered. Every hunk is host portability rather than
CUDA-to-HIP translation, and would equally serve a hypothetical CUDA-on-Windows build.
It stays anyway, for three reasons.

`windows` is a required gate in `config/arches.toml`, and without these lines there is no
ROCm build on Windows at all -- not a degraded one, none: compilation stops in the
preprocessor at `src/include/heongpu/util/memorypool.cuh:14`. A change without which the
port cannot exist on a required platform is part of the port.

The standing "scope-separate non-ROCm fixes" lesson (libSGM/OpenCV) is about a different
shape and does not reach this. There, the workaround papered over a defect that existed
for, and was visible to, the project's current users independently of the port; carrying
it in `moat-port` hid an upstream bug inside an unrelated change. Here nothing in the
delta fixes anything an existing user experiences. Upstream targets Linux only -- there
is no CUDA-on-Windows build to repair -- so these lines repair no pre-existing defect;
they extend the platform set the port itself must reach. Split out, they would be a
Windows-portability PR standing alone against a Linux-only project, with no build that
exercises it and no reason for the maintainer to take it, and the ROCm PR would depend on
it landing first.

The blast radius is nil, which is what makes bundling safe rather than merely convenient.
`git diff 6ac06d0 b12dc98 -- src/lib/util/memorypool.cu src/include/heongpu/util/memorypool.cuh`
contains zero deletion lines: the POSIX branch is not edited, it is fenced. The CUDA
`else()` arm of `CMakeLists.txt:59-71` still appends `-g`/`-O3` unconditionally at lines
64-67 and was correctly left alone. `u_int32_t` -> `uint32_t` is the same type on glibc. So on every
configuration that works today, this commit is a no-op.

For completeness on "what would split if it had to": nothing. The one hunk with any
Linux reach is the `random.cuh` include deletion, and the answer there is to keep it and
prove it (see finding 3), not to move it elsewhere.

### Ruling 2: `hipMemGetInfo` is NOT an APU platform gap. The porter's isolation is invalid.

notes.md:3788-3811 records that `hipMemGetInfo` "fails unconditionally on this integrated
RDNA3.5 part", isolated with `agent_space/heongpu-win/meminfo.hip`, and concludes it is a
platform gap to be worked around or waived. Measured this review, on the same host, with
the porter's own untouched `meminfo.exe`:

```
$ ./meminfo.exe                                          # bare shell
hipMemGetInfo rc=0 (no error) free=72761692160 total=72924151808
$ source agent_space/heongpu-win/env.sh; ./meminfo.exe   # porter's exact env
hipMemGetInfo rc=0 (no error) free=72761692160 total=72924151808
```

The cited binary, in the cited environment, returns success and 67.8 GiB free. The
"unconditional failure" claim is false as written.

The isolation is invalid for a structural reason worth keeping: `meminfo.hip` contains no
kernel, so it never makes the runtime load a code object. The failure lives in code-object
loading -- the system Adrenalin `C:\WINDOWS\System32\amdhip64_7.dll` cannot JIT-link the
runtime's internal blit/transfer kernels against TheRock's device bitcode, the transfer
manager is never created, and the free-memory query alone fails while `hipDeviceTotalMem`
keeps working. A device-code-free reproducer cannot see any of that, which is exactly why
it looked like clean isolation. On Windows the loader prefers the executable's own
directory and then System32 over `PATH`, so `env.sh` prepending TheRock's `bin` never
displaced the driver copy; the real binary and the probe were both running on System32's
DLL and only the real binary cared.

Demonstrated by changing nothing but the DLLs beside the executable:

```
# before: only System32's runtime reachable
$ ./bfv_addition_testcases.exe
... "CUDA Error in .../src/lib/util/memorypool.cu at line 90: invalid argument"
[  FAILED  ] HEonGPU.BFV_Ciphertext_Ciphertext_Addition_Subtraction (91 ms)

$ cp <venv>/Lib/site-packages/_rocm_sdk_core/bin/{amdhip64_7.dll,amd_comgr0713.dll} \
     build/bin/test/
$ ./bfv_addition_testcases.exe
[  FAILED  ] HEonGPU.BFV_Ciphertext_Ciphertext_Addition_Subtraction (3779 ms)
```

The exception at `memorypool.cu:90` is gone and the test runs 40x longer, reaching real
FHE work. So:

- **No source fallback.** Do not add a `hipDeviceProp_t::totalGlobalMem` path, or any
  other graceful degradation, to `MemoryPool::get_decive_avaliable_memory()`. It would be
  a change to shared code buying nothing once the right DLLs are co-located, and it would
  bake a local driver-deployment defect into an upstream-visible port.
- **No block and no waiver.** `windows-gfx1151` is not platform-gapped here and the
  `windows` gate does not need a maintainer decision on this account.
- **The remedy is the validator's run procedure**: copy TheRock's `amdhip64_7.dll` and
  `amd_comgr*.dll` into the directory containing the test executables before `ctest`.
  Promoted to the `cuda-to-rocm` validation reference this round, since it will hit every
  future Windows validation on this host and is invisible from the build side.

One collateral note for whoever reads `memorypool.cu:90` next: the line is
`HEONGPU_CUDA_CHECK(cudaGetLastError())`, which reports the *sticky* error, not
`cudaMemGetInfo`'s return value at line 89. Naming an API from that line is a guess. It
happened to be right here only because `MemoryPool::instance()` clears the error state at
line 51 and nothing between the two makes another runtime call.

Checked and not a problem, recorded so it is not re-raised: the HUMA over-allocation
hazard does not apply to this port. `hipMemGetInfo` reports the full 72.9 GB pool and
`initial_device_memorypool_size` is 0.9, but
`thirdparty/rmm_hip_stub/include/rmm/mr/device/pool_memory_resource.hpp:14-17` is a
pass-through constructor that stores the size and pre-allocates nothing, so no 65 GB
allocation is ever attempted.

### 1. Newly measured: BFV encryption and addition give wrong results on windows-gfx1151

This is the finding that actually blocks the platform, and it was invisible behind the
`hipMemGetInfo` throw. With the runtime DLLs co-located, under
`agent_space/heongpu-win/env.sh`:

| executable | result |
|---|---|
| `bfv_encoding_testcases.exe` | `[ OK ] HEonGPU.BFV_Encoding_Decoding (1108 ms)`, exit 0 |
| `bfv_addition_testcases.exe` | exit 1, all 5 sub-cases fail, 10 assertions |
| `bfv_encryption_testcases.exe` | exit 1, `BFV_Encryption_Decryption` fails |

Every failure is a value mismatch, not an exception:
`test/test_bfv_addition.cpp:100,105,199,204,299,304,399,404,501,506` all report
`std::equal(...) Which is: false`. Encoding/decoding round-trips correctly, so the memory
pool, the device code objects and the NTT path used by encoding are all working; the break
is in the encryption/key-generation path. Note this is not a wavefront question on its
face -- `linux-gfx1100` is also wave32 and passes 20/20 -- so the likely suspects are the
Windows build configuration itself (the empty `CMAKE_MSVC_RUNTIME_LIBRARY`, the forced
`-x hip`, the hand-built NTL, the `clang_rt.builtins` 128-bit division used by the host
modular arithmetic) or something in RNGonGPU's AES path under clang-cl.

Porter to triage. Suggested cheapest first cut, in order: run
`bfv_relinearization`/`ckks_*` to see whether the failure tracks key generation
specifically; then check whether the host-side 128-bit modular precomputation
(`__udivti3` from `clang_rt.builtins-x86_64.lib`) agrees with the Linux result for one
known modulus, since that is the only piece of arithmetic in the chain that is
newly-linked on this platform and silently wrong if mis-linked rather than absent.

This is not held against the delta -- `b12dc98` is what made the failure reachable at all.

### 2. The commit body of `b12dc98` overstates the Windows/Linux equivalence

The body asserts that `MEMORYSTATUSEX::ullAvailPhys` "is what `freeram * mem_unit`
reports". Not the same quantity: glibc's `freeram` is free RAM excluding page cache and
buffers, while `ullAvailPhys` counts memory available to the process including the
reclaimable standby list, so Windows reports the more optimistic number for the same
machine state. They are the right analogues to pair, and the code is fine -- the pool
sizes derived from it are compared, not preallocated -- but this is upstream-visible prose
asserting an equality a maintainer may know is false. Reword to "the closest Windows
analogue" and drop the "which is what ... reports" clause. Same bar this project already
applied at `beba427`.

The `MEMORYSTATUSEX` handling itself checks out (`src/lib/util/memorypool.cu:71-76`):
`memInfo{}` at line 72 zero-initializes before `dwLength = sizeof(memInfo)` at line 73,
which is the classic trap and is not fallen into, and an ignored `GlobalMemoryStatusEx`
failure leaves `ullAvailPhys` at 0, which reaches the existing `"resolved to 0 bytes"`
`std::invalid_argument` at `memorypool.cu:165-169` rather than propagating garbage -- the same
shape as upstream ignoring `sysinfo()`'s return, so it is consistent by design.

### 3. `src/include/heongpu/util/random.cuh` (deleted `<sys/sysinfo.h>`, old line 12) -- the one hunk with Linux reach, unbuilt on Linux

Re-grepped independently rather than taking the porter's word:
`sysinfo|freeram|mem_unit|totalram|get_nprocs|get_phys_pages|get_avphys_pages|SI_LOAD_SHIFT`
across `src/ test/ example/ benchmark/` matches only `memorypool.cuh:14` and
`memorypool.cu:78-80`. A sweep for other BSD `<sys/types.h>` spellings
(`u_char|u_short|u_int|u_long|ushort|ulong`) and for any surviving `u_int32_t` both come
back empty, so item 5 of the round-12 diff is complete. The deletion is correct on the
evidence available.

It is still the only line in this delta that changes what a Linux translation unit sees,
and `b12dc98` has never been compiled on Linux -- if `<sys/sysinfo.h>` was transitively
supplying a declaration to `random.cu` or to the three `context.cuh` consumers, only a
Linux build finds out. Risk is low and the fix if it fires is one include. Flagged so the
`linux-gfx942`/`linux-gfx1100`/`linux-gfx90a` revalidations treat a compile error in
`random.cu` as this hunk rather than as a mystery, not as a reason to change the code now.

### Checked clean, not re-raised

`if(NOT MSVC)` is the correct predicate: `project()` at `CMakeLists.txt:11` enables
`C CXX` before the `USE_HIP` block, so CMake has set `MSVC` for the MSVC-like clang-cl by
then, and it is false for hipcc/gcc on Linux, leaving the `-g`/`-O3` appends exactly as
they were. The `<windows.h>` inclusion cannot leak: it is in a `.cu` translation unit, not
a header, so only `memorypool.cu` sees it, and `WIN32_LEAN_AND_MEAN`/`NOMINMAX` are set
with `#ifndef` guards that agree with the `-D` spellings the configure line already passes.
Commit hygiene passes: title 55 chars with the `[ROCm]` prefix, body carries rationale, the
AI-assistance disclosure and a Test Plan in fenced blocks, no `Co-Authored-By` and no
`noreply` trailer, `python3 utils/jargon.py --port HEonGPU` reports `jargon: clean` over
the branch, all added lines are ASCII, and `git -C projects/HEonGPU/src status --porcelain`
is empty.

### Handoff

`b12dc98` needs no code revert. Round-13 porter work is: correct the `hipMemGetInfo`
analysis in the round-12 notes section so no later agent requests a waiver on it, reword
the commit body per finding 2, and triage finding 1. The two runtime DLLs are currently
copied into `projects/HEonGPU/src/build/bin/test/` (untracked build output); a clean
reconfigure drops them, so re-copy before any test run.
