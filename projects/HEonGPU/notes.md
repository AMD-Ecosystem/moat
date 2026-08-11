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
