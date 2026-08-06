# icicle notes

## Fork / port
- Fork: https://github.com/AMD-Ecosystem/icicle (Actions disabled), branch `moat-port`.
- Base sha: 625532a624e5aaa6e9d31a1c92587f1fcc30dc76 (upstream main, 2025-08-06).
- Port head: d66e92a7733d958f2c5ec118cfa98e0e6de4c683.
- Strategy A, additive sibling backend: new `icicle/backend/hip_pqc/` registering device "HIP-PQC", mirroring `backend/cuda_pqc/`. NVIDIA "CUDA-PQC" path untouched.

## Out of scope (do NOT attempt from this repo)
- ICICLE's main compute backends -- MSM, NTT, ECNTT, EC, big-integer field arithmetic (the PTX carry-chain field math) -- are CLOSED-SOURCE. The top CMake pulls CUDA_BACKEND / METAL_BACKEND / VULKAN_BACKEND from PRIVATE repos at configure time; none of that source is in the tree. Only `backend/cuda_pqc/` (post-quantum ML-KEM) is open and portable. cuda_pqc's field type Zq (q=3329) is pure portable C++ (no inline asm), so no PTX reimplementation arises.

## Build (gfx90a, ROCm 7.2, GCD 0)
```
export HIP_VISIBLE_DEVICES=0
cmake -S icicle -B build -DCMAKE_BUILD_TYPE=Release \
  -DCPU_BACKEND=ON -DHIP_PQC_BACKEND=ON -DCUDA_PQC_BACKEND=OFF \
  -DPQC=ON -DBUILD_TESTS=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build -j
```
Followers: re-run with `-DCMAKE_HIP_ARCHITECTURES=gfx1100` / `gfx1151`, no source change (kernels are wave-agnostic). Build scratch was kept under agent_space/, not in the repo.

## Validation (real gfx90a, GCD 0) -- all PASS
- Backend gtests: `ctest --test-dir build/backend/hip_pqc/tests` -> 58/58 pass, including the bit-exact NIST KAT vectors (test_ml_kem_keygen compares ek/dk against ml_kem_{512,768,1024} reference files; encaps/decaps/pke also KAT-checked). Crypto is exact, not tolerance-based.
- Frontend dispatch: `ctest --test-dir build/tests -R PqcTest` -> 6/6 pass on "HIP-PQC" (shared-secret consistency + on-device path, batch 4096) through the full device abstraction.
- Dispatch confirmed: AMD_LOG_LEVEL=3 shows ml_kem_{keygen,encaps,decaps}_kernel launching on gfx90a via HIP-PQC.

## The wave64 work (the make-or-break)
The ML-KEM kernels run 128-thread blocks organized as four logical 32-thread warps via threadIdx.x/32 and threadIdx.x%32. On gfx90a (wave64) one 64-lane wavefront carries two logical warps. Fixes are localized to the hip_pqc copies; the cuda_pqc tree is untouched.
- `cuda_to_hip.h` provides wave-agnostic 32-lane-subgroup primitives and maps the CUDA warp-sync spellings onto them:
  - `__ballot_sync` -> per-32-lane ballot: take the full-wave `__ballot`, shift by `32 * ((tid & (warpSize-1)) >> 5)`, keep the low 32 bits.
  - `__shfl_sync` -> `__shfl(val, src, 32)` (width-32 confines to the caller's own subgroup; src is a within-group lane 0..31).
  - `__any_sync` -> nonzero per-32-lane ballot.
  - `__reduce_add_sync` -> width-32 `__shfl_xor` butterfly (avoids HIP's `__reduce_add_sync`, which reduces over the FULL 64-lane wavefront and would break MASK28 semantics, not just width).
  - `__syncwarp` -> left as HIP's native (wavefront barrier with memory ordering; a safe superset of a 32-lane fence). Do NOT replace it with a bare `__builtin_amdgcn_fence` -- a release-only fence is too weak for the cross-lane shared reads in keccakf / generate_k_r and produces wrong KATs for k=3/4.
- SHAKE coefficient counter (`load_coeffs_shake`): lanes 28..31 must now CONTRIBUTE A ZERO COUNT and participate in the width-32 reduction instead of `return`-ing early; the early stores are guarded with `if(!active) return;` AFTER the reduce.

## Gotchas (record for followers / reviewers)
- Two latent races that 32-lane lockstep hid but wave64 timing exposed. Both fixes are wave-agnostic (also correct on NVIDIA):
  1. encaps/decaps: warp 0 produces `k_r`, but `r = k_r + 32` is read by ALL warps inside `pke::encrypt`. Added a `__syncthreads()` between the producer and `pke::encrypt`. Symptom without it: K mismatch for k=2 (512).
  2. `generate_k_r`: `H(ek)` writes shared `hashed_ek` with lanes 0-3, `G_m_ek` reads it with lanes 4-7. Added `__syncwarp()` between them. Symptom without it: K mismatch for k=3/4 (768/1024) only -- 512's shorter H happened to win the race.
- `Zq` needs a trivial default constructor (`Zq() = default;`) under clang HIP: a value-initializing ctor makes `__shared__ Zq arr[...]` an error ("initialization is not supported for __shared__ variables"). All shared Zq buffers are written before read, so uninitialized is safe.
- `cuda::barrier<thread_scope_block>` in pke decrypt (init to 128, only warp 0 waits) -> replaced with `__syncthreads()` (equivalent: all 128 arrive, warp 0 then encodes).
- Missing HIP intrinsics: `__ldcs`/`__stcs`/`__ldlu` -> shimmed with `__builtin_nontemporal_load/store` (streaming/last-use hints; plain access is correct). `__restrict__` placed before the type on array params (`const __restrict__ uint8_t x[...]`) is rejected by clang HIP; rewrite as `const uint8_t* __restrict__ x`.
- Registration: the upstream `REGISTER_ML_KEM{512,768,1024}_BACKEND` macros each emit an anonymous-namespace `[]()->bool` static-init lambda; expanding all three in one HIP TU gives them colliding closure manglings under clang, so the static initializer invokes one variant's registrar twice and throws a duplicate-registration error at load. Replaced the three macros with one named `register_hip_pqc_ml_kem_backends()` calling the `register_*` entry points directly.
- Linking: the shared HIP-PQC backend is consumed via runtime dlopen (icicle_load_backend_from_env_or_default, RTLD_LOCAL), NOT linked into icicle_pqc -- linking it as a NEEDED dep would load it a second time and double-register. For ICICLE_STATIC_LINK the interface lib whole-archives it. Tests likewise rely on the runtime load (BACKEND_BUILD_DIR) and do not link the backend directly.
- `.gitignore` ignores `icicle/backend/*` except cpu/cuda_pqc; added `!icicle/backend/hip_pqc` so the new backend is tracked.
- `test_pqc_api.cpp` was hard-coded to "CUDA-PQC"; changed to `IcicleTestBase::main_device()` so it runs on the registered GPU backend on both NVIDIA and ROCm.
- error_translation.h: HIP lacks distinct `hipErrorInvalidHostPointer` / a `SyncDepthExceeded` analog; the hip copy maps hipError_t -> eIcicleError directly (do not `#define` cuda error enums to colliding hip values -> duplicate case labels).

## Validation 2026-06-02 (validator, linux-gfx90a, fork moat-port d66e92a7733d958f2c5ec118cfa98e0e6de4c683)

GPU: AMD Instinct MI250X / MI250 (gfx90a, amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-), GCD 0 (HIP_VISIBLE_DEVICES=0). ROCm 7.2.

### Build

Configure (fresh, includes Taskflow + GTest fetch via FetchContent):
```
export HIP_VISIBLE_DEVICES=0
cmake -S projects/icicle/src/icicle -B agent_space/icicle_val_build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCPU_BACKEND=ON -DHIP_PQC_BACKEND=ON -DCUDA_PQC_BACKEND=OFF \
  -DPQC=ON -DBUILD_TESTS=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=/opt/rocm
# Configuring done (369.1s -- includes network fetch)
```

Compile (via timeit.sh):
```
cmake --build agent_space/icicle_val_build -j128
# [100%] Built target test_ml_kem  -- exit 0, ~25 s
```

### Backend gtests (58 KAT-exact tests, run twice)

Run 1:
```
ctest --test-dir agent_space/icicle_val_build/backend/hip_pqc/tests -V --output-on-failure
# 100% tests passed, 0 tests failed out of 58 -- Total Test time: 20.04 s
```

Run 2:
```
ctest --test-dir agent_space/icicle_val_build/backend/hip_pqc/tests -V --output-on-failure
# 100% tests passed, 0 tests failed out of 58 -- Total Test time: 20.27 s
```

Both runs: 58/58 PASS. Includes bit-exact NIST KAT vectors for ML-KEM-512/768/1024 keygen, encaps, decaps, and PKE encrypt/decrypt. Main-device=HIP-PQC, Reference-device=CPU printed in each test.

### Frontend dispatch tests (6 tests, run twice)

Run 1:
```
ctest --test-dir agent_space/icicle_val_build/tests -R PqcTest -V --output-on-failure
# 100% tests passed, 0 tests failed out of 6 -- Total Test time: 1.40 s
```

Run 2:
```
ctest --test-dir agent_space/icicle_val_build/tests -R PqcTest -V --output-on-failure
# 100% tests passed, 0 tests failed out of 6 -- Total Test time: 1.45 s
```

Both runs: 6/6 PASS (MLkemSharedSecretConsistencyTest + MLkemSharedSecretConsistencyTestOnDevice for Kyber512/768/1024, batch 4096). Main-device=HIP-PQC confirmed.

### Kernel dispatch on gfx90a (AMD_LOG_LEVEL=3)

```
export HIP_VISIBLE_DEVICES=0
AMD_LOG_LEVEL=3 ./test_ml_kem_keygen --gtest_filter="MLKemKeygenTest.ML_KEM_Internal_Keygen512"
```
Output:
```
Using native code object for device: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-
ShaderName : void icicle::pqc::ml_kem::ml_kem_keygen_kernel<...>  hipLaunchKernel: Returned hipSuccess
```

```
AMD_LOG_LEVEL=3 ./test_ml_kem_encaps --gtest_filter="MLKemEncapsTest.MlKemEncaps512Batch"
```
Output:
```
Using native code object for device: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-
ShaderName : void icicle::pqc::ml_kem::ml_kem_encaps_kernel<...>
```

```
AMD_LOG_LEVEL=3 ./test_ml_kem_decaps --gtest_filter="MLKemDecapsTest.MlKemDecaps512Batch"
```
Output:
```
Using native code object for device: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-
ShaderName : void icicle::pqc::ml_kem::ml_kem_decaps_kernel<...>
```

All three ML-KEM kernels (keygen, encaps, decaps) confirmed launching native code on gfx90a. Deterministic across both runs.

### Verdict

PASS. 58/58 backend KAT gtests + 6/6 frontend dispatch tests, both deterministic. ML-KEM kernels run native gfx90a code. No non-GPU regressions (CPU backend gtests are not built in this configuration; the CPU backend itself is unmodified and the closed-source MSM/NTT/EC backends are out of scope and not built).

## Validation 2026-06-02 (gfx1100) -- FAIL: 12/58 backend tests, wave32 race exposed

GPU: AMD Radeon Pro W7800 48GB (gfx1100, wave32, RDNA3). ROCm 7.2.1. HIP_VISIBLE_DEVICES=0.

### Build (gfx1100)

Clone fork moat-port @ d66e92a7733d:
```
git clone --branch moat-port --single-branch https://github.com/AMD-Ecosystem/icicle /var/lib/jenkins/moat/agent_space/icicle_src
```

Configure:
```
cmake -S agent_space/icicle_src/icicle -B agent_space/icicle_gfx1100_build \
  -DCMAKE_BUILD_TYPE=Release -DCPU_BACKEND=ON -DHIP_PQC_BACKEND=ON -DCUDA_PQC_BACKEND=OFF \
  -DPQC=ON -DBUILD_TESTS=ON -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH=/opt/rocm
# Configuring done (10.6s -- FetchContent deps already cached)
```

Compile (timeit.sh):
```
cmake --build agent_space/icicle_gfx1100_build -j128
# [100%] Built target test_ml_kem -- exit 0, 17.7s
```

Architecture verified:
```
roc-obj-ls agent_space/icicle_gfx1100_build/backend/hip_pqc/libicicle_backend_hip_pqc.so
# hipv4-amdgcn-amd-amdhsa--gfx1100  (473032 bytes)
```
No gfx90a code present. Correct.

### Backend gtests (58 tests) -- FAIL 12/58

```
ctest --test-dir agent_space/icicle_gfx1100_build/backend/hip_pqc/tests
# 79% tests passed, 12 tests failed out of 58 -- Total Test time: 18.9s
```

Failing tests (consistent across runs):
- 16 - KyberTest.PkeKeygen768
- 24 - KyberTest.ML_KEM_Internal_Keygen768
- 31 - MLKemTest.KeyCheckTest768Batch
- 32 - MLKemTest.KeyCheckTest1024Batch
- 38 - MLKemKeygenTest.ML_KEM_Internal_Keygen768
- 41 - MLKemKeygenTest.ML_KEM_Internal_Keygen768_Batch
- 45 - MLKemEncapsTest.MlKemEncaps768Batch
- 46 - MLKemEncapsTest.MlKemEncaps1024Batch
- 48 - MLKemDecapsTest.MlKemDecaps768Batch
- 49 - MLKemDecapsTest.MlKemDecaps1024Batch
- 54 - PkeEncryptTest.Encrypt768Batch
- 55 - PkeEncryptTest.Encrypt1024Batch

All k=512 tests pass. All 1024 single-instance KAT tests pass. The failures are concentrated in k=768 and batch k=1024.

### Wave32 diagnosis

The failures are NON-DETERMINISTIC: `PkeKeygen768` (KAT test with fixed input d="3c53596d...") produces a DIFFERENT wrong dk_pke on every run, even though the input is fixed. This is the definitive signature of a shared-memory race condition. On gfx90a (wave64), wavefront 0 carries threads 0-63 (logical warps 0+1) and wavefront 1 carries threads 64-127 (logical warps 2+3). Intra-wavefront lockstep hides cross-warp shared memory races within a wavefront. On gfx1100 (wave32), each of the 4 logical warps (warp 0..3, threads 0..31, 32..63, 64..95, 96..127) is its own independent wavefront; there is NO automatic lockstep synchronization across warps.

Pattern analysis:
- k=512: `generate_error_vector<2, 4, eta=3, start=1, end=1>` -- ONE warp (warp 1). Passes.
- k=768: `generate_error_vector<3, 6, eta=2, start=0, end=1>` -- TWO warps (0 and 1). FAILS non-deterministically.
- k=1024 single KAT: `generate_error_vector<4, 8, eta=2, start=0, end=0>` -- ONE warp (warp 0). Passes.
- k=1024 batch: fails. Involves encaps/decaps which run `pke::encrypt` with `generate_matrix_A<4,1,3>` (warps 1-3) and `generate_error_vector<4,4,eta=2,0,0>` (warp 0) concurrently without a barrier -- and with 8192 blocks running simultaneously, the timing-dependent race becomes reliable.

The race is a genuine wave32 issue not fixed by the two existing race fixes (encaps `__syncthreads` and `generate_k_r __syncwarp`). Those fixes address cross-warp DATA races in specific functions. The new failure exposes a DIFFERENT class of race: multiple independent wavefronts accessing shared memory within the same block without sufficient ordering.

The porter needs to audit `pke::keygen` (k=768 two-warp error vector), `pke::encrypt` (concurrent matrix_A + error vector generation), and the encaps/decaps kernels for missing `__syncthreads()` between concurrent warp assignments that access or are followed by reads of the same shared memory regions.

### Verdict

FAIL. 46/58 backend gtests pass; 12/58 fail with non-deterministic results indicating a shared-memory race exposed uniquely on wave32 (gfx1100). No HSA faults (0x1016), no hang. The gfx90a wave64 lockstep hid this race. State set to validation-failed; escalated to porter.

## Review 2026-06-02 (reviewer, linux-gfx90a, fork moat-port d66e92a)

Verdict: review-passed. Additive sibling HIP-PQC backend; reviewed `git diff 625532a...HEAD` in full plus the cuda_to_hip shim, all wave64-touched kernels, the two race fixes (against the untouched cuda_pqc originals), registration, CMake wiring, and commit hygiene. No changes requested. The GPU gate (58/58 KAT-exact backend gtests + 6/6 frontend dispatch) is recorded above as porter-run on real gfx90a; the validator re-runs it next.

No defects found. Fault-class analysis verified correct:
- Wave64 subgroup mapping (cuda_to_hip.h): per-32-lane ballot extracted from the 64-bit `__ballot` (shift by `32 * ((tid & (warpSize-1)) >> 5)`), width-32 `__shfl`/`__shfl_xor`, and a width-32 butterfly that replaces HIP `__reduce_add_sync` (which would fold lanes 28..63 into MASK28). Confinement is by hardware lane group, not mask width. Sound; the two logical warps in a wavefront index distinct shared slots (`threadIdx.x/32`) so they stay independent.
- SHAKE coeff counter (cuda_sample_utils.cuh load_coeffs_shake): lanes 28-31 now compute `active=false`, contribute `valid_count=0` to the butterfly, participate in the reduction, then `if(!active) return;` after. Equivalent to the masked 28-lane CUDA reduce. Correct.
- Both races are genuine and pre-existing latently in cuda_pqc (confirmed byte-diff: cuda_pqc encaps has no `__syncthreads` between warp-0 `generate_k_r` and all-warp `pke::encrypt` read of `r=k_r+32`; cuda_pqc `generate_k_r` has no `__syncwarp` between `H` write and `G` read of shared `hashed_ek`). Fixes are wave-agnostic (`__syncthreads`, `__syncwarp`) and correct on NVIDIA too. Port (correctly per its additive strategy) leaves cuda_pqc untouched; the upstream PR ships only HIP-PQC.
- keccakf (cuda_sha3_32threads.cuh) is byte-identical to cuda_pqc; relies on intra-32-lane lockstep, valid within a wavefront on both wave widths; `sha3_state_raw` sized for 4 logical warps (max), fits wave64 and wave32.
- Registration: named `register_hip_pqc_ml_kem_backends()` avoids the three-anonymous-lambda mangling collision; device API uses single `REGISTER_DEVICE_API("HIP-PQC")` in a separate TU. Shared backend consumed via runtime dlopen (RTLD_LOCAL), not NEEDED-linked, so no double-registration; static build whole-archives. Sound.
- Shims `__ldcs`/`__stcs`/`__ldlu` -> `__builtin_nontemporal_load/store`; `cuda::barrier`->`__syncthreads`; `Zq() = default` for `__shared__` arrays. `__funnelshift_l`/`__signbit`/`ROTL1` are HIP builtins, KAT-validated.

Hygiene: cuda_pqc and cpu trees untouched (verified empty diff). `[ROCm]` title 46 chars, mentions Claude, Test Plan present, no noreply trailer, no ghstack, no em-dash, no AMD-internal account refs. Fork main is a clean mirror at base 625532a; moat-port single curated commit at d66e92a. Actions disabled on AMD-Ecosystem/icicle (`enabled:false`). Closed MSM/NTT/ECNTT/EC out-of-scope documented in commit, README, plan, notes.

Observational only (no action required):
- icicle/cmake/backend_include.cmake:37 sets `BACKEND_BUILD_DIR` for HIP-PQC for the frontend test loader, matching how the existing CUDA_BACKEND/METAL/VULKAN blocks set it (last-wins). A combined CUDA_PQC_BACKEND+HIP_PQC_BACKEND build would have the HIP value win, but that combo is not a MOAT target and the last-wins pattern is pre-existing upstream; the validated HIP-only build is correct.

## Delta-port 2026-06-02 (gfx1100 wave32 barrier fix)

Fixes the validator's 12/58 wave32 FAIL. Fork moat-port d66e92a -> d8e04fd.

### The race (one shared-memory LDS producer->consumer fence)

File: `icicle/backend/hip_pqc/include/ml_kem/samplers/cuda_samplers.cuh`, in `generate_error_vector`, at the `if constexpr (ntt)` boundary (the new `__syncwarp()` sits just before the unrolled `ntt_inplace_32threads` loop, ~line 116).

Data path: `samplePolyCBD_2_5threads` (eta=2, used by k=768/1024) writes each sampled polynomial into the shared error-vector buffer (`s_e`), then the in-place NTT reads that same shared poly back. The eta=2 sampler had NO fence between the CBD write and the NTT read; the eta=3 sampler (`samplePolyCBD_3_5threads`, used by k=512) already ends with a `__syncwarp()`, which is why k=512 passed and k=768/1024 failed.

On gfx90a (wave64) a wavefront carries two logical 32-warps in lockstep, so the LDS write->read happened to be ordered implicitly and the missing fence was invisible. On gfx1100 (wave32) each logical warp is its OWN wavefront; the LDS write->read across the divergent CBD branch into the NTT is not implicitly ordered, so the first error-vector polynomial (element 0, the warp-0 poly) was read before its CBD write completed. Symptom: `dk_pke` byte 0..383 (= s_hat[0]) was a DIFFERENT wrong value on every run while bytes 384+ (s_hat[1], s_hat[2]) stayed bit-exact -- the definitive non-deterministic single-poly signature. The two-warp k=768 keygen and the high-concurrency k=768/1024 batch encaps/decaps (which all route through `generate_error_vector` with ntt=true) exposed it; k=512 (eta=3, already fenced) and k=1024 single-instance KAT did not.

### The barrier added

```
if constexpr (ntt) {
  __syncwarp();   // order CBD shared write before NTT shared read
  for (int t = 0; t < min_hashes_per_warp; t++) ntt_inplace_32threads(...);
  ...
}
```

`__syncwarp()` (HIP wavefront barrier with memory ordering) is the right scope: the dependency is intra-logical-warp (warp 0 writes and warp 0 reads), and on wave32 the wavefront IS the logical warp. Wave-agnostic: on wave64 it is a full-wavefront barrier (a sound superset; gfx90a re-passes), on CUDA a warp barrier with a memory fence. Matches the existing eta=3 fence exactly. One-line fix, fixes all 12 failing tests at the single shared producer->consumer boundary.

### Build (gfx1100, ROCm 7.2.1, AMD Radeon Pro W7800, wave32)

```
export HIP_VISIBLE_DEVICES=0
cmake -S icicle -B build -DCMAKE_BUILD_TYPE=Release \
  -DCPU_BACKEND=ON -DHIP_PQC_BACKEND=ON -DCUDA_PQC_BACKEND=OFF \
  -DPQC=ON -DBUILD_TESTS=ON -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build -j
```

### Gate (gfx1100) -- PASS, stable

- `ctest --test-dir build/backend/hip_pqc/tests`: 58/58 PASS, run 3x (all 58/58).
- `ctest --test-dir build/tests -R PqcTest`: 6/6 PASS, run 3x (all 6/6).
- Determinism (the race-fix proof), 5x each, all bit-exact every run:
  - `KyberTest.PkeKeygen768` (the previously flaky NIST KAT): 5/5 OK.
  - `MLKemTest.KeyCheckTest768Batch` + `KeyCheckTest1024Batch`: 5/5 OK.
  - `MLKemEncapsTest.MlKemEncaps768Batch` + `MlKemEncaps1024Batch`: 5/5 OK.
  - `MLKemDecapsTest.MlKemDecaps768Batch` + `MlKemDecaps1024Batch`: 5/5 OK.
- No HSA 0x1016, no hang.

advance-head d8e04fd flipped gfx90a completed -> revalidate (expected; the barrier is wave-agnostic so gfx90a re-passes unchanged).

## Review 2026-06-02 (reviewer, linux-gfx1100, fork moat-port d8e04fd)

Verdict: review-passed (delta-ported -> review-passed). Reviewed the amend delta d66e92a..d8e04fd via /pr-review. The validator re-runs the gfx1100 GPU gate next.

No problems found. The delta is exactly one file, +7 lines: a single unguarded `__syncwarp()` plus comment in `icicle/backend/hip_pqc/include/ml_kem/samplers/cuda_samplers.cuh:123`, at the `if constexpr (ntt)` boundary. No unrelated change slipped into the amend; commit title/body unchanged except as expected.

Fact-checked against the code (not the writeup):
- Barrier scope is correct as `__syncwarp` (NOT `__syncthreads`). In `generate_error_vector` the CBD writes target `error_vector_result[start_idx_in_warp]` where `start_idx` is per-warp (cuda_samplers.cuh:93-94,102), and the NTT reads back `error_vector_result[t + start_idx]` for the SAME warp (cuda_samplers.cuh:126,130). Each logical warp owns a disjoint `[start_idx, start_idx+warp_hash_count)` poly range; there is no cross-warp read inside this helper, including the two-warp k=768 case (start=0,end=1). On wave32 the wavefront IS the logical warp, so a wavefront-scope fence orders the producer->consumer LDS dependency; `__syncthreads` would be an unnecessary superset.
- `__syncwarp` is a real LDS fence here: the shim leaves it as HIP-native (cuda_to_hip.h:148-151, "wavefront barrier with the right memory ordering"), not a weak release-only fence.
- The eta=3 path already fenced, eta=2 did not: `samplePolyCBD_3_5threads` ends with a trailing `__syncwarp()` (cuda_sample_helpers_5threads.cuh:117); `samplePolyCBD_2_5threads` ends at the last poly write with no fence (cuda_sample_helpers_5threads.cuh:69-70). Confirmed.
- Single fence covers all 12 failures: every ntt=true callsite routes through this boundary -- pke_keygen.cuh:31 (k=512, start=1,end=1, one warp, already eta=3-fenced -> passed), :35 (k=768, start=0,end=1, two warps -> failed), :39 (k=1024 single, start=0,end=0 -> passed), pke_encrypt.cuh:31 (encaps/decaps batch -> failed under concurrency). eta1=3 for Kyber-512 and eta1=2 for Kyber-768/1024 per spec, matching the eta=2-lacked-fence diagnosis.
- Wave-agnostic + minimal: no literal-arch hardcode, no warpSize/32 conditional, no per-arch branch; safe superset on wave64 (gfx90a re-passes, hence the expected advance-head revalidate flip) and on CUDA.

Hygiene: `[ROCm]` title 46 chars; body mentions Claude and now documents the third (wave32) race; Test Plan present; no Co-Authored-By/noreply trailer, no ghstack, no AMD-internal refs, no CI yaml. Scope confined to backend/hip_pqc (cuda_pqc/cpu/cuda trees untouched).

## Validation 2026-06-03 (revalidate, linux-gfx90a, fork moat-port d8e04fd116ff60770d1d04bc731562f2659af7a9)

GPU: AMD Instinct MI250X / MI250 (gfx90a, amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-), GCD 2 (HIP_VISIBLE_DEVICES=2). ROCm 7.2.

Delta from prior validated sha d66e92a7: one `__syncwarp()` added in `icicle/backend/hip_pqc/include/ml_kem/samplers/cuda_samplers.cuh` at the `if constexpr (ntt)` boundary in `generate_error_vector` (the wave32 race fix). Classified as `mixed` by moatlib.py -- real-GPU re-run required (no carry-forward).

### Build (incremental from existing build dir)

Source updated to d8e04fd1 via `git reset --hard <fork-remote>/moat-port` on the local checkout. Incremental rebuild only recompiled the changed TU.

```
export HIP_VISIBLE_DEVICES=2
bash utils/timeit.sh icicle compile -- cmake --build agent_space/icicle_val_build -j128
# [100%] Built target test_ml_kem -- exit 0
```

### Backend gtests (58 KAT-exact tests)

```
bash utils/timeit.sh icicle test -- ctest --test-dir agent_space/icicle_val_build/backend/hip_pqc/tests -V --output-on-failure
# 100% tests passed, 0 tests failed out of 58 -- Total Test time: 20.45 sec
```

58/58 PASS. Includes bit-exact NIST KAT vectors for ML-KEM-512/768/1024 keygen, encaps, decaps, PKE encrypt/decrypt. The previously added `__syncwarp()` (wave32 fix) is a safe superset on wave64 -- all k=512/768/1024 variants pass.

### Frontend dispatch tests (6 tests)

```
ctest --test-dir agent_space/icicle_val_build/tests -R PqcTest -V --output-on-failure
# 100% tests passed, 0 tests failed out of 6 -- Total Test time: 1.48 sec
```

6/6 PASS (MLkemSharedSecretConsistencyTest + MLkemSharedSecretConsistencyTestOnDevice for Kyber512/768/1024, batch 4096). Main-device=HIP-PQC confirmed.

### Verdict

PASS. 58/58 backend KAT gtests + 6/6 frontend dispatch tests. Wave32 `__syncwarp()` fix is wave-agnostic; gfx90a wave64 re-passes unchanged. No non-GPU regressions (CPU backend and closed-source MSM/NTT/EC backends out of scope, not built).

## Validation 2026-06-03 (gfx1100) -- barrier fix final gate at d8e04fd

GPU: AMD Radeon Pro W7800 48GB (gfx1100, amdgcn-amd-amdhsa--gfx1100, wave32, RDNA3). ROCm 7.2.1. HIP_VISIBLE_DEVICES=1. Fork clone projects/icicle/src at d8e04fd, clean.

### Build (fresh, gfx1100)

```
export HIP_VISIBLE_DEVICES=1
cmake -S projects/icicle/src/icicle -B agent_space/icicle_val_gfx1100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCPU_BACKEND=ON -DHIP_PQC_BACKEND=ON -DCUDA_PQC_BACKEND=OFF \
  -DPQC=ON -DBUILD_TESTS=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH=/opt/rocm
# Configuring done (10.2s -- FetchContent deps cached)
bash utils/timeit.sh icicle compile -- cmake --build agent_space/icicle_val_gfx1100 -j128
# [100%] Built target test_ml_kem -- exit 0
```

Architecture verified:
```
roc-obj-ls agent_space/icicle_val_gfx1100/backend/hip_pqc/libicicle_backend_hip_pqc.so
# hipv4-amdgcn-amd-amdhsa--gfx1100  (472776 bytes)
```
No gfx90a code present. Correct.

### Kernel dispatch on gfx1100 (AMD_LOG_LEVEL=3)

```
HIP_VISIBLE_DEVICES=1 AMD_LOG_LEVEL=3 ./test_ml_kem_keygen --gtest_filter="MLKemKeygenTest.ML_KEM_Internal_Keygen512"
# Using native code object for device: amdgcn-amd-amdhsa--gfx1100
# ShaderName : void icicle::pqc::ml_kem::ml_kem_keygen_kernel<...>
HIP_VISIBLE_DEVICES=1 AMD_LOG_LEVEL=3 ./test_ml_kem_encaps --gtest_filter="MLKemEncapsTest.MlKemEncaps512Batch"
# Using native code object for device: amdgcn-amd-amdhsa--gfx1100
# ShaderName : void icicle::pqc::ml_kem::ml_kem_encaps_kernel<...>
```

All three ML-KEM kernels (keygen, encaps, decaps) confirmed launching native code on gfx1100.

### Backend gtests (58 KAT-exact tests, run twice)

Run 1:
```
bash utils/timeit.sh icicle test -- ctest --test-dir agent_space/icicle_val_gfx1100/backend/hip_pqc/tests --output-on-failure
# 100% tests passed, 0 tests failed out of 58 -- Total Test time: 20.96 sec
```

Run 2:
```
HIP_VISIBLE_DEVICES=1 ctest --test-dir agent_space/icicle_val_gfx1100/backend/hip_pqc/tests --output-on-failure
# 100% tests passed, 0 tests failed out of 58 -- Total Test time: 20.97 sec
```

Both runs: 58/58 PASS. Bit-exact NIST KAT vectors for ML-KEM-512/768/1024 keygen, encaps, decaps, PKE encrypt/decrypt.

### Frontend dispatch tests (6 tests)

```
HIP_VISIBLE_DEVICES=1 ctest --test-dir agent_space/icicle_val_gfx1100/tests -R PqcTest --output-on-failure
# 100% tests passed, 0 tests failed out of 6 -- Total Test time: 1.24 sec
```

6/6 PASS (MLkemSharedSecretConsistencyTest + MLkemSharedSecretConsistencyTestOnDevice for Kyber512/768/1024, batch 4096). Main-device=HIP-PQC confirmed.

### Determinism (race-fix proof) -- previously non-deterministic cases, 3 runs each

KyberTest.PkeKeygen768 (the previously flaky NIST KAT, non-deterministic wrong dk_pke):
- Run 1: PASSED (112 ms), Run 2: PASSED (122 ms), Run 3: PASSED (109 ms) -- 3/3 bit-exact.

MLKemTest.KeyCheckTest1024Batch:
- Run 1: PASSED (755 ms), Run 2: PASSED (695 ms), Run 3: PASSED (713 ms) -- 3/3 bit-exact.

MLKemEncapsTest.MlKemEncaps768Batch + MlKemEncaps1024Batch:
- Run 1: PASSED (991 ms / 752 ms), Run 2: PASSED (811 ms / 684 ms), Run 3: PASSED (823 ms / 679 ms) -- 3/3 each.

MLKemDecapsTest.MlKemDecaps768Batch + MlKemDecaps1024Batch:
- Run 1: PASSED (739 ms / 542 ms), Run 2: PASSED (646 ms / 529 ms), Run 3: PASSED (640 ms / 510 ms) -- 3/3 each.

All previously non-deterministic cases are now bit-exact across 3 runs. The wave32 LDS race is confirmed fixed.

### Verdict

PASS. 58/58 backend KAT gtests + 6/6 frontend dispatch tests on gfx1100 (AMD Radeon Pro W7800, wave32) at d8e04fd. The previously-failing k=768/1024 cases (PkeKeygen768, KeyCheckTest1024Batch, MlKemEncaps768/1024Batch, MlKemDecaps768/1024Batch) are now bit-exact stable across 3 runs -- the single `__syncwarp()` in generate_error_vector at the eta=2 CBD->NTT boundary fixes the wave32 LDS race. No HSA 0x1016, no hang. Fork clean at d8e04fd.

## Validation 2026-06-04 (windows-gfx1151) -- BLOCKED: icicle host framework has no Windows support

GPU: AMD Radeon 8060S (gfx1151, RDNA3.5, wave32). Windows 11. TheRock ROCm (venv-gsplat). Fork moat-port @ d8e04fd.

### Attempt

Clone: `git clone --branch moat-port --single-branch https://github.com/AMD-Ecosystem/icicle projects/icicle/src` -- OK.

Configure:
```
cmake -S projects/icicle/src/icicle -B agent_space/icicle_win_build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=<rocm>/lib/llvm/bin/clang-cl.exe \
  -DCMAKE_CXX_COMPILER=<rocm>/lib/llvm/bin/clang-cl.exe \
  -DCMAKE_HIP_COMPILER=<rocm>/lib/llvm/bin/clang++.exe \
  -DCPU_BACKEND=ON -DHIP_PQC_BACKEND=ON -DCUDA_PQC_BACKEND=OFF \
  -DPQC=ON -DBUILD_TESTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_PREFIX_PATH=<rocm_win> \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=agent_space/googletest
```
Configure succeeded (28.5s, FetchContent resolved via local googletest + fetched Taskflow).

Build (cmake --build ... -j6) -- FAIL at first compilation:
```
D:/Develop/moat/projects/icicle/src/icicle/src/runtime.cpp:2:10: fatal error: 'dlfcn.h' file not found
```
AND (device-code TU):
```
D:/Develop/moat/projects/icicle/src/icicle/include/icicle/utils/utils.h:4:10:
  fatal error: 'cxxabi.h' file not found
```

### Root cause

The icicle host framework (the `icicle_device` library) has zero Windows support:

1. `icicle/src/runtime.cpp` (the backend loader): hard-depends on POSIX `dlfcn.h`/`dlopen`/`RTLD_LOCAL`/`RTLD_GLOBAL`, `dirent.h`/`opendir`/`readdir`/`closedir`, and `sys/stat.h`. Line 285 contains a literal `#error "Unsupported operating system"` for any platform that is not Linux or macOS.
2. `icicle/include/icicle/utils/utils.h`: includes `cxxabi.h` (GCC ABI demangling) for the `demangle<T>()` utility; Windows/MSVC has no equivalent.
3. `icicle/tests/test_base.h`: uses `setenv()` (POSIX) to set `ICICLE_BACKEND_INSTALL_DIR`.

The backend-loading architecture -- tests call `icicle_load_backend_from_env_or_default()` which scans a directory with `opendir`/`readdir` and loads each backend .dll via `dlopen` -- has no Windows (LoadLibrary) implementation. The `#error` in runtime.cpp is explicit that Windows is unsupported.

### Scope

This is not a delta to our HIP-PQC backend (backend/hip_pqc/). The HIP device kernels, cuda_to_hip.h shim, and wave-agnostic fixes are all correct and compile-clean. The blocker is entirely in the upstream icicle host framework code, which would require a separate multi-file Windows port (runtime.cpp LoadLibrary replacement, cxxabi.h demangling alternative, setenv -> _putenv_s, SHARED_LIB_EXTENSION ".dll"). That is upstream-scope work beyond the MOAT port boundary.

### Verdict

BLOCKED. No GPU run possible. Reason: `no-windows-support-in-icicle-host-framework`. The `icicle_device` runtime.cpp explicitly errors at compile time on non-Linux/macOS with `#error "Unsupported operating system"` and uses POSIX-only dlopen/dirent APIs throughout the backend loader. This is an upstream icicle limitation, not a HIP-PQC backend issue. The HIP-PQC GPU kernels and wave-agnostic shims are correct (validated on gfx90a and gfx1100). Escalate to upstream icicle for Windows support.

## Validation 2026-06-06 (windows-gfx1201) -- BLOCKED: same Windows framework blockers as gfx1151

GPU: RX 9070 XT (gfx1201, RDNA4, wave32). Windows 11. TheRock ROCm. Fork moat-port @ d8e04fd.

### Attempt

Shallow-cloned the fork at the validated head (d8e04fd) to probe Windows compatibility:

```
git clone --depth 1 --branch moat-port https://github.com/AMD-Ecosystem/icicle agent_space/icicle_win_probe
cd agent_space/icicle_win_probe
git checkout FETCH_HEAD
# HEAD: d8e04fd116ff60770d1d04bc731562f2659af7a9 -- confirmed same SHA
```

### Blockers verified at d8e04fd

The same structural Windows blockers found on gfx1151 are present unchanged at the current validated head:

1. `icicle/src/runtime.cpp:2-4`: `#include <dlfcn.h>`, `#include <dirent.h>`, `#include <sys/stat.h>` (POSIX only)
2. `icicle/src/runtime.cpp:280-286`: `#ifdef __linux__ ... #elif __APPLE__ ... #else #error "Unsupported operating system" #endif` -- the `#error` fires unconditionally on Windows
3. `icicle/src/runtime.cpp:323-336`: `dlopen()`, `opendir()`/`readdir()` -- POSIX backend loader, no LoadLibrary path
4. `icicle/include/icicle/utils/utils.h:4`: `#include <cxxabi.h>` (GCC ABI demangler, no Windows equivalent)
5. `icicle/tests/test_base.h:32`: `setenv()` (POSIX only, no `_putenv_s` equivalent)

These are in upstream icicle host framework code, not in the HIP-PQC backend (backend/hip_pqc/). The GPU kernels, cuda_to_hip.h shim, and wave-agnostic fixes are all correct and unaffected. A Windows port of the icicle host framework (LoadLibrary, FindFirstFile/FindNextFile, demangler alternative, _putenv_s) is upstream-scope work, not within the MOAT HIP-PQC port boundary.

### Verdict

BLOCKED. No GPU run possible. Reason: `no-windows-support-in-icicle-host-framework`. Identical to the gfx1151 finding at the same source SHA (d8e04fd). The blocker is upstream framework, not the HIP-PQC GPU port.

## Rework 2026-06-16 (porter, linux-gfx1100): drop duplicate test data + jargon scrub

Fork moat-port d8e04fd -> 036c92c (commit on top, no amend of the validated d8e04fd).

Two review defects fixed:

A. Jargon scrub (tracked files), technical meaning kept:
- icicle/backend/hip_pqc/CMakeLists.txt:3 ("Additive HIP-PQC backend (jeffdaily MOAT port)." -> "Additive HIP-PQC backend."); :29 ("Strategy A: keep the CUDA spelling..." -> "Keep the CUDA spelling...").
- icicle/backend/hip_pqc/include/cuda_to_hip.h:3 (dropped "(jeffdaily MOAT port)"); :7 (dropped "(Strategy A)").
- icicle/backend/hip_pqc/README.md:21 (dropped "(Strategy A)"; "runs bit-for-bit as on 32-wide hardware" -> "runs identically to 32-wide hardware").
- git grep for MOAT / Strategy A|B / bit-for-bit / byte-for-byte / curated commit / moat-port at HEAD: zero tracked hits.

B. Removed 85.5 MiB duplicated ML-KEM KAT test data:
- git rm'd the 9 files under icicle/backend/hip_pqc/tests/ml_kem/test_data/ (ml_kem_{512,768,1024}_data x {dk,ek,random_bytes}). They were byte-identical (same git blob SHAs) to icicle/backend/cuda_pqc/tests/ml_kem/test_data/. Total dropped: 89,653,248 bytes (85.5 MiB).
- Repointed icicle/backend/hip_pqc/tests/CMakeLists.txt copy_test_data: source dir ${CMAKE_CURRENT_SOURCE_DIR}/ml_kem/test_data -> ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda_pqc/tests/ml_kem/test_data, destination unchanged (${CMAKE_CURRENT_BINARY_DIR}/test_data). Tests still read ./test_data at runtime; behavior identical.
- git ls-tree -r -l moat-port -- icicle/backend/hip_pqc/tests/ml_kem/test_data: empty.

Device code / HIP kernels unchanged.

### Verify (gfx1100, HIP_VISIBLE_DEVICES=0, ROCm 7.2.1, AMD Radeon Pro W7800)
Fresh configure + build:
```
cmake -S projects/icicle/src/icicle -B agent_space/icicle_rework_build \
  -DCMAKE_BUILD_TYPE=Release -DCPU_BACKEND=ON -DHIP_PQC_BACKEND=ON -DCUDA_PQC_BACKEND=OFF \
  -DPQC=ON -DBUILD_TESTS=ON -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build agent_space/icicle_rework_build -j128   # Built target test_ml_kem, exit 0
```
Confirmed test_data copied into the build dir from the cuda_pqc source path (86M, all 9 files present).
- ctest --test-dir agent_space/icicle_rework_build/backend/hip_pqc/tests --output-on-failure -> 58/58 PASS (23.78s); the KAT tests read the repointed test_data and pass.
- ctest --test-dir agent_space/icicle_rework_build/tests -R PqcTest --output-on-failure -> 6/6 PASS (1.27s).

advance-head 036c92c classified the delta FUNCTIONAL (build/layout change): linux-gfx90a and linux-gfx1100 flipped completed -> revalidate (correct; do NOT force back to completed). Windows platforms stay blocked (unchanged).

## Validation 2026-06-16 (revalidate, linux-gfx90a, fork moat-port 036c92cfffbd339e89d4be44a27a8e0509b9f53a)

GPU: AMD Instinct MI250X / MI250 (gfx90a, amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-), GCD 0 (HIP_VISIBLE_DEVICES=0). ROCm 7.2.

Delta from prior validated sha d8e04fd: jargon scrub (3 comment-only files) + removed 9 duplicated ML-KEM test data files (85.5 MiB) from hip_pqc/tests/ml_kem/test_data/ and repointed the CMakeLists.txt copy_test_data target to read from cuda_pqc/tests/ml_kem/test_data/ instead. Device code unchanged. moatlib.py classify returned `mixed` (functional build/layout change) -> full GPU revalidation required.

### Build (fresh configure + compile)

```
export HIP_VISIBLE_DEVICES=0
cmake -S projects/icicle/src/icicle -B agent_space/icicle_val_build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCPU_BACKEND=ON -DHIP_PQC_BACKEND=ON -DCUDA_PQC_BACKEND=OFF \
  -DPQC=ON -DBUILD_TESTS=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=/opt/rocm
# Configuring done (15.0s -- FetchContent deps cached)
bash utils/timeit.sh icicle compile -- cmake --build agent_space/icicle_val_build -j128
# [100%] Built target test_ml_kem -- exit 0
```

Test data verified present in build dir: build/backend/hip_pqc/tests/test_data/{ml_kem_512,768,1024}_data/, copied from cuda_pqc source path (86M, all 9 files).

### Backend gtests (58 KAT-exact tests)

```
bash utils/timeit.sh icicle test -- ctest --test-dir agent_space/icicle_val_build/backend/hip_pqc/tests -V --output-on-failure
# 100% tests passed, 0 tests failed out of 58 -- Total Test time: 19.57 sec
```

58/58 PASS. Includes bit-exact NIST KAT vectors for ML-KEM-512/768/1024 keygen, encaps, decaps, PKE encrypt/decrypt.

### Frontend dispatch tests (6 tests)

```
ctest --test-dir agent_space/icicle_val_build/tests -R PqcTest -V --output-on-failure
# 100% tests passed, 0 tests failed out of 6 -- Total Test time: 1.40 sec
```

6/6 PASS (MLkemSharedSecretConsistencyTest + MLkemSharedSecretConsistencyTestOnDevice for Kyber512/768/1024, batch 4096). Main-device=HIP-PQC confirmed.

### Kernel dispatch on gfx90a (AMD_LOG_LEVEL=3)

```
AMD_LOG_LEVEL=3 ./test_ml_kem_keygen --gtest_filter="MLKemKeygenTest.ML_KEM_Internal_Keygen512"
# Using native code object for device: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-
# ShaderName : void icicle::pqc::ml_kem::ml_kem_keygen_kernel<...>
```

Native gfx90a code confirmed. hipLaunchKernel returns hipSuccess.

### Verdict

PASS. 58/58 backend KAT gtests + 6/6 frontend dispatch tests. Repointed test_data path (cuda_pqc shared source) works correctly -- all NIST KAT vectors load and pass. Device code unchanged; wave-agnostic fixes from d8e04fd remain intact. No non-GPU regressions (CPU backend and closed-source MSM/NTT/EC backends out of scope, not built). Fork clean at 036c92c.

## Validation 2026-06-17 (revalidate, linux-gfx1100, fork moat-port 98d8bcab96ebb63309c4fbdd5a66bdf2c5315285)

GPU: AMD Radeon Pro W7800 48GB (gfx1100, amdgcn-amd-amdhsa--gfx1100, wave32, RDNA3). ROCm 7.2.1. HIP_VISIBLE_DEVICES=3.

Delta from prior validated sha d8e04fd: two commits on top -- 036c92c (jargon scrub + drop 85.5 MiB duplicated test data, repoint CMakeLists.txt copy_test_data to cuda_pqc source path) and 98d8bca (comment-only: rewording in hip_pqc/tests/CMakeLists.txt). Device/kernel code unchanged. moatlib.py classify 036c92c returned `mixed` (functional build/layout change) -> full GPU revalidation required. 98d8bca classified `comment-only / inert` -> auto-carried after gfx1100 pass.

### Build (incremental from agent_space/icicle_rework_build, built for gfx1100 at 036c92c)

The rework build was already present and built for gfx1100. Incremental compile at current HEAD (98d8bca, comment-only) required no recompilation:

```
export HIP_VISIBLE_DEVICES=3
bash utils/timeit.sh icicle compile -- cmake --build agent_space/icicle_rework_build -j128
# [100%] Built target test_kyber -- exit 0
```

Architecture verified: `roc-obj-ls .../libicicle_backend_hip_pqc.so` -> `hipv4-amdgcn-amd-amdhsa--gfx1100 (472776 bytes)`. No gfx90a code present.

Test data verified in build dir: agent_space/icicle_rework_build/backend/hip_pqc/tests/test_data/ (86M, ml_kem_512_data/, ml_kem_768_data/, ml_kem_1024_data/) -- copied from cuda_pqc source path, not the hip_pqc tree.

### Backend gtests (58 KAT-exact tests)

```
export HIP_VISIBLE_DEVICES=3
bash utils/timeit.sh icicle test -- ctest --test-dir agent_space/icicle_rework_build/backend/hip_pqc/tests -V --output-on-failure
# 100% tests passed, 0 tests failed out of 58
```

58/58 PASS. Bit-exact NIST KAT vectors for ML-KEM-512/768/1024 keygen, encaps, decaps, PKE encrypt/decrypt. Main-device=HIP-PQC, Reference-device=CPU printed in each test.

### Frontend dispatch tests (6 tests)

```
export HIP_VISIBLE_DEVICES=3
ctest --test-dir agent_space/icicle_rework_build/tests -R PqcTest -V --output-on-failure
# 100% tests passed, 0 tests failed out of 6 -- Total Test time: 2.98 sec
```

6/6 PASS (MLkemSharedSecretConsistencyTest + MLkemSharedSecretConsistencyTestOnDevice for Kyber512/768/1024, batch 4096). Main-device=HIP-PQC confirmed.

### Verdict

PASS. 58/58 backend KAT gtests + 6/6 frontend dispatch tests on gfx1100 (AMD Radeon Pro W7800, wave32) at 98d8bca. Repointed test_data (cuda_pqc shared source) correctly provides all 9 KAT data files to the hip_pqc tests. The previously-fixed wave32 LDS race (d8e04fd __syncwarp in generate_error_vector) remains intact; all k=768/1024 tests pass deterministically. No non-GPU regressions. Fork clean.
