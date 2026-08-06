# sppark notes

## Validation Summary (linux-gfx90a)

sppark has authoritative upstream HIP support from Supranational. Validation on gfx90a (MI250X, ROCm 7.2.1) shows **partial** functionality:

### What Works: NTT (Number Theoretic Transform)

All 7 field types pass correctness tests against arkworks reference:

```
NVCC=off HIPCC=/opt/rocm/bin/hipcc cargo test --release --features=<field> -- --nocapture
```

| Field | Test | Status |
|-------|------|--------|
| bls12_381 | test_against_arkworks | PASS |
| bls12_377 | test_against_arkworks | PASS |
| bn254 | test_against_arkworks | PASS |
| pallas | test_against_arkworks | PASS |
| vesta | test_against_arkworks | PASS |
| gl64 (Goldilocks) | gl64_self_consistency | PASS |
| bb31 (Baby Bear) | bb31_self_consistency | PASS |

### What Does Not Work: MSM (Multi-Scalar Multiplication)

The MSM POC does not compile on HIP due to missing infrastructure:

1. **Build system gap**: poc/msm-cuda/build.rs only detects nvcc, not hipcc (unlike sppark crate's build.rs which detects both)

2. **mont_t.hip incompleteness**: Only ~25% of mont_t.cuh is ported to mont_t.hip:
   - mont_t.hip: 311 lines
   - mont_t.cuh: 1221 lines
   - Missing critical functions: `shfl_xor`, `cneg`, `vt_inverse_mod_x`, and warp-shuffle-based Montgomery arithmetic

3. **fp2 extension field not ported**: bls12-381-fp2.hpp, bls12-377-fp2.hpp, and alt_bn128-fp2.hpp are guarded by `#ifdef __CUDA_ARCH__` with no HIP equivalent. The fp2 device code (required for G2 MSM) uses mont_t functions not available in mont_t.hip.

4. **MSM kernel headers not HIP-ready**: pippenger.cuh, batch_addition.cuh use `#ifdef __CUDA_ARCH__` guards and include `<cuda.h>`, `<cooperative_groups.h>` without HIP alternatives.

### Root Cause

Supranational's HIP support focuses on NTT which uses basic Montgomery field operations. The MSM algorithm requires more advanced warp-shuffle-based routines (inverse modulo, vectorized shifts with ballot/shuffle) that were not ported to mont_t.hip. This is likely intentional -- MSM has CUDA-specific optimizations (PTX inline assembly, cooperative groups) that would require significant rewrite for HIP.

### Recommendation

The project is **already-supported for NTT**. For MSM:
- Completing the mont_t.hip port requires ~900 lines of complex code with GCN inline assembly
- The fp2 device code needs full implementation for HIP
- MSM kernels need cooperative_groups -> hip/hip_cooperative_groups.h conversion

This is beyond validation scope. Options:
1. **Document as partial support**: NTT works, MSM requires additional work
2. **Upstream issue/discussion**: Report findings to Supranational
3. **Full port**: Complete mont_t.hip and fp2 HIP paths (substantial effort)

## MSM HIP Port Progress (2026-06-05)

Work in progress to enable MSM on HIP/ROCm.

### Completed Changes

1. **poc/msm-cuda/build.rs**: Added hipcc detection (NVCC=off pattern from rust/build.rs)
2. **ff/bls12-381-fp2.hpp, ff/bls12-377-fp2.hpp, ff/alt_bn128-fp2.hpp**: Changed `#ifdef __CUDA_ARCH__` to `#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)`
3. **ff/mont_t.hip**: Extended significantly (~500 lines added):
   - `operator<<=` and `operator>>=` (bit shifts)
   - `cneg` (conditional negation)
   - `abs` (sign handling)
   - `is_one`, `is_zero` checks
   - `one(int or_zero)` selector
   - `shfl` member function
   - `shfl_xor` template
   - `dot_product` (simple a*b + c*d form)
   - `vt_inverse_mod_x` (modular inverse using Euclidean algorithm)
   - `reciprocal` and division operators
   - Supporting infrastructure: `approx_t`, `factorx_t`, `sfp_t` classes
   - Helper functions: `ab_approximation_n`, `inner_loop_x`, `cneg_v`, `smul_n_shift_x`
4. **msm/pippenger.cuh, msm/batch_addition.cuh**: Added HIP header guards, PTX-to-HIP conversions for laneid and shuffle operations, cooperative_groups polyfill
5. **msm/sort.cuh**: Added HIP versions for `pack` (lop3 alternative) and `sum_up` (shuffle-based sum)
6. **util/cuda2hip.hpp**: Added `HIP_VERSION_MAJOR < 7` guard for `__ballot_sync` polyfill (ROCm 7+ has native)
7. **rust/src/build.rs**: Added `-x hip` flag for proper HIP compilation

### Blocking Issue

ROCm 7.2.x's hipcc performs dual-pass compilation: first a host pass, then a device pass. During the HOST pass, `__HIP_DEVICE_COMPILE__` is NOT defined, but the file is still parsed and template functions are instantiated. This causes `__device__` inline functions from HIP headers (like `__shfl_up`, `__syncthreads`, `atomicAdd`) to fail with "call to __device__ function from __host__ function" errors.

This is NOT a cc-rs issue or a `-x hip` flag issue -- it's fundamental to hipcc's architecture.

Confirmed behavior:
```
/opt/rocm/bin/hipcc -E --offload-arch=gfx90a -x hip test.hip
```
produces BOTH `HIP_DEVICE_COMPILE_IS_DEFINED` and `HIP_DEVICE_COMPILE_IS_NOT_DEFINED` because preprocessor runs twice.

Error example:
```
/var/lib/jenkins/moat/projects/sppark/src/msm/sort.cuh:61:22: error: no matching function for call to '__shfl_up'
    61 |         uint32_t v = __shfl_up(sum, off, WARP_SZ);
       |                      ^~~~~~~~~
note: candidate function not viable: call to __device__ function from __host__ function
```

### Possible Solutions

1. **Make functions `__host__ __device__`**: But HIP intrinsics only work on device, need stubs for host
2. **Use `__HIPCC_RTC__` or other undocumented macros**: May have better host/device discrimination
3. **Restructure code**: Separate host-visible declarations from device-only implementations
4. **Use `.hip` extension with header-only guards**: May change hipcc's parsing behavior
5. **Report upstream to Supranational**: They may have solved this for other projects

### Files Modified (not yet committed)

- ff/mont_t.hip (+502 lines)
- msm/pippenger.cuh, msm/batch_addition.cuh, msm/sort.cuh
- poc/msm-cuda/build.rs, poc/msm-cuda/Cargo.toml, poc/msm-cuda/cuda/pippenger_inf.cu
- ff/bls12-381-fp2.hpp, ff/bls12-377-fp2.hpp, ff/alt_bn128-fp2.hpp
- rust/src/build.rs, util/cuda2hip.hpp

## MSM HIP Port Progress (2026-06-05 continued)

### Fixed Issues

1. **`__device__` macro stripped by affine_t.hpp**: ec/affine_t.hpp was redefining `__device__` to empty for non-CUDA compilers. Fixed by checking for `__HIPCC__` as well.

2. **cooperative_groups duplicate definitions**: pippenger.cuh and batch_addition.cuh both defined cooperative_groups polyfills. Added include guards (`__SPPARK_COOP_GROUPS_POLYFILL__`).

3. **ROCm 7 `__shfl_sync` 64-bit mask requirement**: ROCm 7's native `__shfl_sync` requires 64-bit masks. cuda2hip.hpp polyfills conflicted. Fixed by:
   - Guarding old polyfills with `#if HIP_VERSION_MAJOR < 7`
   - Adding 32-to-64-bit mask wrapper functions for ROCm 7+

4. **`class` vs `typename` in template defaults**: Template parameters like `class bucket_h = class bucket_t::mem_t` fail with clang because `mem_t` is a type alias. Changed to `typename bucket_t::mem_t`.

5. **Preprocessor guard alignment for host vs device types**: Fixed ff/*.hpp files to use `__HIP_DEVICE_COMPILE__` instead of `__HIPCC__` to properly distinguish host and device compilation phases.

6. **Device-only code in sort.cuh/pippenger.cuh/batch_addition.cuh**: Added `#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)` guards around function bodies that use device intrinsics.

### Remaining Blocking Issue

**Host functions using device-only operators during device pass**

hipcc's dual-pass compilation model differs from nvcc:
- nvcc: Device pass only compiles device code; host code is skipped
- hipcc: Device pass processes ALL code, including host functions

The pippenger.cuh `collect()` function runs on CPU (post-GPU-kernel processing) and performs field arithmetic using mont_t operators. These operators are `__device__`-only (defined via `#define inline __device__ __forceinline__` in mont_t.hip).

During hipcc device pass (compiling for gfx90a):
1. Host function `collect()` is parsed and type-checked
2. It instantiates field operations (`operator^`, `operator+=`, etc.) on mont_t types
3. These operators are `__device__`-only, causing "call to __device__ function from __host__ function"

For CUDA, this works because mont_t.cuh is only included when `__CUDA_ARCH__` is defined, so during host compilation, the types come from blst_t.hpp (CPU implementations). For HIP, we aligned to the same pattern (mont_t.hip only when `__HIP_DEVICE_COMPILE__`), but the issue persists because hipcc still parses host code during device pass.

### Potential Solutions (require deeper refactoring)

1. **Make mont_t operators `__host__ __device__`**: Would require implementing CPU-side field arithmetic in mont_t.hip (significant work).

2. **Guard host-only code paths**: Add `#if !defined(__HIP_DEVICE_COMPILE__)` around `collect()` and similar host functions. But these functions are needed at runtime.

3. **Use `__attribute__((host))` explicitly**: Mark all host-only functions to prevent device pass from trying to resolve their symbols.

4. **Split host and device templates**: Have separate host and device versions of msm_t class and related templates.

None of these are trivial; the codebase assumes nvcc's compilation model.

## MSM HIP Port Attempt: `__host__ __device__` with Host Stubs (2026-06-05)

Attempted to resolve the hipcc dual-pass compilation issue by making mont_t operators `__host__ __device__` with host stubs for device intrinsics.

### Approach

1. Changed `#define inline __device__ __forceinline__` to `#define inline __host__ __device__ __forceinline__` in mont_t.hip (HIP only)

2. For each function using device intrinsics (asm, `__builtin_amdgcn_*`, `threadIdx.x`, etc.), wrapped the body with:
```cpp
#if defined(__HIP_DEVICE_COMPILE__)
    // Real implementation with device intrinsics
#else
    return ...; // Host stub - compiles but never executed
#endif
```

3. Applied same pattern to fp2_t functions in bls12-381-fp2.hpp

4. Updated is_inf() and uadd() in ec/xyzz_t.hpp and ec/affine_t.hpp to be `__host__ __device__`

### Result

Build SUCCEEDED but tests CRASHED with GPU memory access fault:
```
Memory access fault by GPU node-2 (Agent handle: 0x...) on address 0x...
Reason: Unknown.
```

### Analysis

The host stubs approach has a fundamental flaw: making functions `__host__ __device__` allows the WRONG version to be selected at runtime. The compiler may choose the host stub version for device code paths, leading to uninitialized memory access.

The issue is subtle:
- During compilation, both host and device versions exist
- Template instantiation may select the wrong overload
- At runtime, calling a host stub from device code crashes

### Conclusion

This approach is NOT viable. The problem requires a structural solution:
1. Complete separation of host and device type systems (like CUDA with blst_t.hpp vs mont_t.cuh)
2. OR architectural changes to how hipcc handles dual-pass compilation
3. OR upstream fixes to sppark to support HIP's compilation model

The NTT tests continue to work because NTT doesn't have the host-function-calling-device-operator pattern that MSM has.

### State

Reverted all changes to mont_t.hip and field headers. Source tree reset to clean NTT-working state. Project remains blocked for MSM on HIP.

## MSM HIP Port Analysis: Compilation Unit Separation Required (2026-06-05)

### Problem Diagnosis

The previous attempt to use `__host__ __device__` stubs failed because hipcc's template instantiation picks the wrong overload at runtime. A deeper analysis reveals that the core issue requires a structural solution, not just macro guards.

hipcc's dual-pass compilation model:
1. **Host pass**: All code is parsed, including template instantiations. Device intrinsics (`__shfl_up`, `__syncthreads`, `atomicAdd`) are defined as `__device__` functions but referenced from templates instantiated for host code.
2. **Device pass**: `__HIP_DEVICE_COMPILE__` is defined, device intrinsics work.

The problem: sort.cuh, pippenger.cuh, and batch_addition.cuh have `__device__` helper functions that call HIP intrinsics. These helpers are instantiated during the host pass because the msm_t class template includes them. Even with proper guards (`#ifdef __HIP_DEVICE_COMPILE__`), the HIP system headers provide inline `__device__` functions that hipcc parses in both passes.

### Issues Found

1. **HIP cooperative_groups header**: `<hip/hip_cooperative_groups.h>` has inline functions that call `__ockl_grid_sync()` etc., which are `__device__`-only. Including this header causes errors in the host pass.

2. **__ballot_sync ambiguity**: ROCm 7+ has native `__ballot_sync` returning `uint64_t`, conflicting with the cuda2hip.hpp polyfill returning `uint32_t`.

3. **sort.cuh device functions**: Functions like `sum_up()` and `pack()` use HIP intrinsics directly. Even when guarded, the HIP headers they depend on cause issues.

4. **fp2 requires vt_inverse_mod_x**: The fp2 reciprocal uses `vt_inverse_mod_x()` which requires ~400 lines of complex Montgomery ladder code with shuffle operations. This is not in mont_t.hip.

### Required Solution: Compilation Unit Separation

The only viable solution is to separate host-only and device code into different compilation units:

1. **Device code** (.cu/.hip files compiled by hipcc):
   - mont_t, kernels, `__device__` helpers
   - Compiled with full HIP intrinsic support

2. **Host-only code** (.cpp files compiled by g++/clang):
   - `msm_t::collect()`, `integrate_row()`, `sum_up()`
   - Uses POD types (raw pointers, uint32_t*) for data exchange
   - Does NOT include device headers

3. **Interface**:
   - Device code exports kernel launch wrappers
   - Host code calls wrappers, never touches mont_t directly

### Implementation Outline

```
msm/
  pippenger.cuh          # Device-only: kernels, bucket_t ops
  pippenger_host.cpp     # Host-only: collect(), integrate_row()
  pippenger_interface.h  # POD struct for host/device boundary
```

The host functions (`collect`, `integrate_row`) would work with raw coordinate arrays instead of `bucket_t` objects. The type system divergence happens at compilation boundary, not runtime.

### Effort Estimate

- ~2-3 hours to refactor MSM to separate compilation units
- ~1 hour to implement minimal fp2 support (without reciprocal) or stub it out
- Testing and debugging

### Recommendation

The G1 MSM (primary use case) could be made to work with this refactor. G2 MSM (fp2) requires additional mont_t.hip work or should be disabled for HIP.

NTT continues to work because it doesn't have the host-function-calling-device-operator pattern that MSM has.

## MSM HIP Port BREAKTHROUGH (2026-06-11, porter)

MSM (G1 / Pippenger) now COMPILES and PASSES correctness on gfx90a (ROCm 7.2.1,
MI250X, wave64). The prior "compilation unit separation required" blocker was a
misdiagnosis: no separate .cpp TUs are needed. Two root causes, both fixable
in-header.

### Root cause 1: wrong host/device macro on HIP
sppark's field headers selected the device mont_t type with `__HIPCC__` (defined
in BOTH hipcc passes) where CUDA uses `__CUDA_ARCH__` (device pass only). So on
HIP the device field type leaked into host code. The correct HIP analog of
`__CUDA_ARCH__` is `__HIP_DEVICE_COMPILE__`. Fixed every `__CUDA_ARCH__`-style
guard across ff/{bls12-381,bls12-377,alt_bn128}.hpp, ec/{affine_t,xyzz_t,
jacobian_t}.hpp:
  - device-side field types and device EC algorithm: `defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)`
  - host-side field types (blst): `!defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)`
  - compiler-presence (mem_t classes etc.): `defined(__CUDACC__) || defined(__HIPCC__)`
ec/affine_t.hpp also stripped `__device__`/`__host__` to empty under `#ifndef __CUDACC__` (true on HIP!) and never restored them -- catastrophic; re-guarded to `!__CUDACC__ && !__HIPCC__`.

With this, the HIP HOST pass uses the blst host field (host-accessible Montgomery
constants) and the HIP DEVICE pass uses mont_t.hip -- exactly CUDA's model. This
is REQUIRED: mont_t's modulus/RR/ONE live in `__device__ __constant__` memory and
are GARBAGE if read from host, so the host driver (collect/integrate_row, which
does real field+point arithmetic on the CPU after the GPU phases) MUST use blst.
A "device-type-everywhere on HIP" attempt builds but returns the identity (inf)
because host reads junk constants -- verified with a host unit test.

### Root cause 2: clang instantiates __global__ template bodies in the host pass
nvcc drops unreferenced/launch-only `__global__` bodies from its host pass;
clang/hipcc semantically instantiates them at the `<<<>>>` / launch site with the
host (blst) field type, which then can't call the device-only kernel internals.
Fix: wrap each `__global__` kernel BODY in `#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)` so the host-pass instantiation is an empty stub (only its address is needed for the launch); the real body compiles in the device pass. Applied to breakdown/accumulate/integrate (pippenger.cuh), the batch_addition `add` helper + the digits-variant kernel (batch_addition.cuh). The xyzz_t `#else` host `uadd` wrappers were made `__host__ __device__` so the kernel body (instantiated in the host pass) can still resolve them.

### PTX -> HIP translations (msm/, the inline-asm fault class)
- `mov.u32 %laneid` -> `threadIdx.x % WARP_SZ` (logical-32-warp lane).
- predicated `shfl.sync.up.b32 %v|%pred` (clamp) -> `__shfl_up_sync(...,WARP_SZ)` + an explicit lane-range predicate.
- `lop3.b32 ...,0xb8` (a&~mask | b&mask) -> the plain C++ bit-select.
- `prefetch.global.L2` -> `__builtin_prefetch`.
All guarded `#if defined(__HIPCC__)` HIP / `#else` PTX so the CUDA build is byte-identical.

### mont_t.hip: extended + made __host__ __device__
`# define inline __host__ __device__ __forceinline__` (was `__device__`), and gave
every GCN-asm method (`+= -= * << >> final_sub czero csel`) a portable C++ host
fallback under `#else` of `__HIP_DEVICE_COMPILE__`. Added the MSM-needed methods
absent from the NTT-trimmed mont_t.hip: `is_zero`, `is_zero(a)`, `is_one`, `cneg`,
`abs`, `one(int)`, `shfl`, `operator<<=/>>=`, `operator-`. Cross-lane ops (shfl,
shfl_bfly) are device-only with a no-op host body (host never reaches them).

### cuda2hip.hpp
- `#define HIP_DISABLE_WARP_SYNC_BUILTINS` before `<hip/hip_runtime.h>`: ROCm 7's
  native `__shfl_*_sync`/`__ballot_sync` static_assert a 64-bit mask; sppark passes
  CUDA's 32-bit 0xffffffff with logical-32-warp semantics, so use the in-header
  32-bit-mask polyfills. `__syncwarp()` polyfill made unconditional (it lived in the
  same disabled block).
- Minimal `cooperative_groups::this_grid().sync()` shim -> `__ockl_grid_sync()`. The
  full `<hip/hip_cooperative_groups.h>` fails to compile when pulled into a TU with
  host code (its helpers reference __device__-only __ockl_* from __host__ __device__
  inline fns). Replaced the include with this shim.

### Build wiring
- poc/msm-cuda/build.rs: detect hipcc (NVCC=off) parallel to nvcc, compile via
  `sppark::build::ccmd()` (reads DEP_SPPARK_TARGET=rocm), emit `feature="rocm"`.
- `class X::mem_t` default template args -> `typename X::mem_t` (clang rejects the
  elaborated-type spec on a type alias; valid on nvcc too).

### G2 / fp2 scoped OUT on HIP (deferred)
fp2 (G2 MSM) reciprocal needs `vt_inverse_mod_x` (~400 lines of shuffle-based
Montgomery-ladder inverse) not in mont_t.hip. The msm-cuda POC's G1 path is the
primary use case. On HIP, pippenger_inf.cu defines SPPARK_MSM_FP2 only for CUDA;
`mult_pippenger_fp2_inf` and the fp2 Rust test/bench are `#[cfg(not(feature="rocm"))]`.

### Build (gfx90a, ROCm 7.2.1)
```
cd poc/msm-cuda
NVCC=off HIPCC=/opt/rocm/bin/hipcc HIP_VISIBLE_DEVICES=0 \
  cargo build --release --features=bls12_381    # or bls12_377 / bn254
```
Rust toolchain installed via rustup (was absent on this host).

### Validation (real gfx90a, HIP_VISIBLE_DEVICES=0) -- G1 MSM vs arkworks
- bls12_381 msm_correctness: PASS at TEST_NPOW=10,12,15,17
- bls12_377 msm_correctness: PASS at TEST_NPOW=12
- bn254: see latest run
NTT continues to pass (unchanged path).

### Critical gotcha: the __noinline__ barrier (NTT regression caught during port)
Changing mont_t.hip's `# define inline __device__ __forceinline__` to
`__host__ __device__ __forceinline__` (needed so the MSM host driver can call
the field) silently inlined away `noop()`, which upstream declares
`__device__ __noinline__` with an empty `asm("")`. That noop is an OPTIMIZATION
BARRIER the Montgomery reduction (operator-=, final_sub) relies on; forcing it
inline made NTT compile and run but produce WRONG results (test_against_arkworks
failed, all curves). Fix: keep `noop()` `__device__ __noinline__` on the device
path (`#if __HIP_DEVICE_COMPILE__`), host stub otherwise. Lesson: when you widen
a project's `inline` macro to host+device, audit every `__noinline__`/barrier it
overrides. Bisected by reverting files one at a time and re-running NTT.

### bn254 G1 MSM hangs (open issue, not blocking)
bls12_381 and bls12_377 G1 MSM pass; bn254 G1 MSM HANGS (GPU at 100%, no
completion) at TEST_NPOW>=12. bls curves finish in <1s. Likely a bn254-specific
wave64 issue (254-bit field, different limb count / sort partitioning) or a
cooperative-grid-sync edge case for that field. Needs separate investigation;
does not block the bls12_381/bls12_377 G1 result. Recorded so a resume starts
here, not from zero.

### Validation summary (linux-gfx90a, fork moat-port d7cb381, ROCm 7.2.1, MI250X)
- MSM G1 bls12_381 msm_correctness: PASS (TEST_NPOW 12,14,15,17)
- MSM G1 bls12_377 msm_correctness: PASS (TEST_NPOW 12,14)
- MSM G1 bn254: HANGS (see above)
- MSM G2/fp2: scoped out on HIP (deferred sppark-g2-fp2-msm)
- NTT bls12_381 test_against_arkworks: PASS (regression-checked, unchanged path)
- NTT bls12_377/bn254/pallas/vesta: build + run clean
- 0 test failures across the suite.

## Review 2026-06-12 (reviewer)

Verdict: changes-requested. The port is well-engineered and the CUDA path is preserved byte-for-byte (the device/host field guards reduce to the original expressions when only `__CUDA_ARCH__` is defined and HIP macros are absent; jargon-clean; commit hygiene clean). One genuine defect in the enabled surface blocks PR-readiness.

### Must fix

1. bn254 G1 MSM is enabled on ROCm but hangs the GPU. poc/msm-cuda/tests/msm.rs:19 `msm_correctness` runs unconditionally for bn254, and poc/msm-cuda/src/lib.rs:9-10 / Cargo.toml:17 leave `--features=rocm,bn254` buildable, but the notes record bn254 G1 hanging the GPU at TEST_NPOW>=12 (notes.md:386). A user who builds the bn254 feature gets a binary that wedges the device. Either fix the hang or gate bn254 G1 off on ROCm the way fp2/G2 is gated (a `#[cfg(not(feature="rocm"))]` on the bn254 path, mirroring poc/msm-cuda/cuda/pippenger_inf.cu's `SPPARK_MSM_FP2` include switch), so the shipped ROCm surface is all-passing. Shipping a feature flag that hangs the GPU is not upstream-PR-ready, even though the bls12_381/bls12_377 G1 results are unaffected.

### Verified clean (no action)

- Fault classes: no hardcoded warpSize that breaks wave64. The MSM uses upstream's logical-32-lane-warp model on wave64; every shuffle the porter ADDED uses the explicit width-32 4-arg `__shfl_*_sync(...,WARP_SZ)` variant (msm/pippenger.cuh:195, msm/sort.cuh:63,189), which lowers to the real ROCm `__shfl_*(...,32)` primitive rather than the cheaper 3-arg bpermute polyfill. The pre-existing 3-arg calls (msm/sort.cuh:199, msm/batch_addition.cuh:60 etc.) are upstream code and rely on the `idx += threadIdx.x & ~31` warp-base in the polyfill (util/cuda2hip.hpp:173).
- 64-bit lane mask: handled by suppressing ROCm 7's native `__shfl_*_sync`/`__ballot_sync` (which static_assert a 64-bit mask) in the MSM TU only via `SPPARK_DISABLE_NATIVE_WARP_SYNC` -> `HIP_DISABLE_WARP_SYNC_BUILTINS` (util/cuda2hip.hpp:7-9 of the new block, build.rs:120), so the 32-bit-mask CUDA-compat polyfills apply. NTT keeps the native `__syncwarp` ordering.
- PTX -> HIP translations are semantically faithful: `%laneid` -> `threadIdx.x % WARP_SZ`; predicated `shfl.sync.up` clamp -> 4-arg `__shfl_up_sync` + explicit lane-range zeroing (msm/pippenger.cuh:195-197, msm/sort.cuh:187-191); `lop3.b32 0xb8` -> the `(a&~mask)|(mask&b)` bit-select (msm/sort.cuh:46); `prefetch.global.L2` -> `__builtin_prefetch` (ec/xyzz_t.hpp:131). All guarded so the CUDA build is byte-identical.
- mont_t.hip host fallbacks (operator<<=/>>=, +=/-=, final_sub, cneg, abs, one(int), csel, czero) match the CUDA mont_t.cuh reference: >>= funnel-shifts (lo>>1)|(hi<<31) with the N%32 top-word handling; final_sub subtracts MOD iff incoming carry OR val>=MOD; cross-lane shfl/shfl_bfly are device-only with no-op host bodies (host never reaches them). The `noop()` __noinline__ Montgomery optimization barrier is correctly preserved on the device path (ff/mont_t.hip:70-73), avoiding the NTT-wrong-results regression documented in notes.
- Minimal footprint / BC: 17 files, all MSM/field/build. fp2 (G2) headers untouched and stay CUDA-only via the include-level `SPPARK_MSM_FP2` switch; no orphaned half-port from the reverted earlier attempts. NTT field selection on HIP is unchanged (NTT does not define `SPPARK_HIP_HOST_FIELD`, so it keeps the device field in both passes as before).
- Build: reuses the pre-existing multi-arch `sppark::build::ccmd()` (rust/src/build.rs, gfx90a at line 104, native-arch for build-poc); arch handling is unified, no per-arch hack. `class X::mem_t` -> `typename X::mem_t` default-arg fixes are valid on nvcc too.
- Commit hygiene: `[ROCm]` title 49 chars, no em-dash, no noreply trailer, mentions Claude, Test Plan present, copyright added parallel to upstream in mont_t.hip with Jeff Daily author tag (substantial extension warrants it). No AMD-internal account references.

Note: a real-GPU run is the validator's job; the bls12_381/bls12_377 G1 PASS already recorded is encouraging, but the bn254 defect above is a code/scope issue independent of the GPU run.

## Changes-requested fix: gate bn254 G1 off on ROCm (2026-06-12, porter)

Resolved the reviewer must-fix. Rather than ship a `--features=rocm,bn254`
binary that hangs the GPU, bn254 G1 MSM is now refused at build time on the
ROCm/HIP backend, mirroring how G2/fp2 is scoped out via the include-level
`SPPARK_MSM_FP2` switch. The bls12_381/bls12_377 G1 surface is unchanged and
still passes; the CUDA path is byte-identical (both guards are HIP/rocm-only).

Two guards (defense in depth), both committed on TOP of the validated commit
d7cb381 (not amended):
- poc/msm-cuda/build.rs: when the detected backend is rocm and the bn254
  feature is set, panic with a clear message before any GPU object is compiled.
  This is the user-visible gate (`rocm` is build.rs's own emitted cfg, so it
  is not available during build.rs; the backend is resolved from hipcc/nvcc
  detection, which is what build.rs keys on).
- poc/msm-cuda/cuda/pippenger_inf.cu: `#error` under
  `defined(__HIPCC__) && defined(FEATURE_BN254)` as a backstop for any other
  build path.

The Rust test (poc/msm-cuda/tests/msm.rs) `msm_correctness` is left as-is; it
can no longer be reached with bn254 on rocm because the build fails first, so
no `#[cfg]` change is needed there.

Deferred the underlying hang fix: `sppark-bn254-g1-msm-rocm`
(utils/deferred.py). Resume by root-causing the wave64 / sort-partitioning /
cooperative-grid-sync edge case for the 254-bit field, then drop both gates.

### Re-validation of the enabled ROCm surface (gfx90a, ROCm 7.2.1, MI250X, GCDs 2-3)
- `cargo build --release --features=bls12_381` (NVCC=off, hipcc): OK
- bls12_381 msm_correctness TEST_NPOW=15: PASS
- bls12_377 msm_correctness TEST_NPOW=14: PASS
- `--features=bn254` on rocm: build correctly REFUSED (build.rs panic), no
  binary produced, device never touched.

## Review 2026-06-12 (reviewer, re-review of commit 8688269)

Verdict: review-passed. The bn254 ROCm-gating fix resolves the prior must-fix and introduces no new problems; no findings.

Scope reviewed: git diff d7cb381..HEAD (commit 8688269 only, +19 lines across 2 files). Confirmed d7cb381 (validated SHA) is a reachable ancestor of HEAD (committed on top, not amended), so the prior validation lineage is intact; the delta is a functional build-gate change that correctly triggers revalidation of the enabled surface (porter re-ran: bls12_381 NPOW=15 PASS, bls12_377 NPOW=14 PASS, bn254 build refused).

Verified (no action): (a) the gate prevents a hanging binary -- poc/msm-cuda/build.rs:123 panics when `backend == "rocm"` (set at build.rs:111-112 on hipcc detection) and `cfg!(feature = "bn254")`, BEFORE the GPU object is compiled at build.rs:142, so no wedging binary is produced; `cfg!(feature = "bn254")` is the project's established idiom (already used at build.rs:30 to set FEATURE_BN254). The pippenger_inf.cu:30-32 `#error` under `defined(__HIPCC__) && defined(FEATURE_BN254)` is a correct backstop for direct-compile paths. (b) CUDA byte-identical -- both guards are HIP/rocm-only: the .cu block is inert under nvcc (`__HIPCC__` undefined) and the build.rs panic is inert when backend is cuda. (c) bls12_381/bls12_377 ROCm surface untouched -- the diff adds only the two gates, no edits to the validated G1 path. (d) Commit hygiene clean -- `[ROCm]` title 47 chars, no noreply/co-author trailer, mentions Claude, no em-dash, Test Plan present, no MOAT jargon, no AMD-internal account references.

Test (msm_correctness, poc/msm-cuda/tests/msm.rs) left unchanged is correct: it runs bn254 only when the bn254 feature is set, and that build now fails first on ROCm, so the test is unreachable there -- no `#[cfg]` change needed. Deferred entry sppark-bn254-g1-msm-rocm is registered and accurate (resume by root-causing the hang, then drop both gates).

## Validation 2026-06-12 (linux-gfx90a)

GPU: gfx90a (AMD Instinct MI250X, GCD 1), ROCm 7.2.1. HIP_VISIBLE_DEVICES=1. Fork HEAD 8688269.

### Build

```
cd poc/msm-cuda
NVCC=off HIPCC=/opt/rocm/bin/hipcc cargo build --release --features=bls12_381   # exit 0
NVCC=off HIPCC=/opt/rocm/bin/hipcc cargo build --release --features=bls12_377   # exit 0
```

Both succeed with only pre-existing unused-parameter and precedence warnings (upstream code, not port regressions).

### GPU tests (MSM correctness)

```
# bls12_381 G1 Pippenger MSM vs arkworks reference
HIP_VISIBLE_DEVICES=1 NVCC=off HIPCC=/opt/rocm/bin/hipcc TEST_NPOW=15 \
  cargo test --release --features=bls12_381 -- --nocapture
# Result: test msm_correctness ... ok  (1 passed, 0 failed)

# bls12_377 G1 Pippenger MSM vs arkworks reference
HIP_VISIBLE_DEVICES=1 NVCC=off HIPCC=/opt/rocm/bin/hipcc TEST_NPOW=14 \
  cargo test --release --features=bls12_377 -- --nocapture
# Result: test msm_correctness ... ok  (1 passed, 0 failed)
```

### bn254 build-time gate

```
HIP_VISIBLE_DEVICES=1 NVCC=off HIPCC=/opt/rocm/bin/hipcc \
  cargo build --release --features=bn254
# Result: exit 101 -- build.rs panics with
#   "the bn254 curve is not yet supported on the ROCm/HIP MSM backend;
#    use bls12_381 or bls12_377, or the CUDA backend for bn254"
# No GPU object compiled, no binary produced, device never touched.
```

Gate confirmed: --features=rocm,bn254 is refused at build time. No hang.

### Summary

| Test | Result |
|------|--------|
| bls12_381 msm_correctness (TEST_NPOW=15) | PASS |
| bls12_377 msm_correctness (TEST_NPOW=14) | PASS |
| bn254 build-time gate | PASS (build refused, exit 101) |

0 test failures. Source tree clean (git status --porcelain: empty). Advancing to completed.

## Validation 2026-06-12 (linux-gfx1100)

GPU: gfx1100 (AMD Radeon Pro W7800 48GB), ROCm 7.2.1. HIP_VISIBLE_DEVICES=1. Fork HEAD 8688269.

### Build

```
cd poc/msm-cuda
NVCC=off HIPCC=/opt/rocm/bin/hipcc cargo build --release --features=bls12_381   # exit 0
NVCC=off HIPCC=/opt/rocm/bin/hipcc cargo build --release --features=bls12_377   # exit 0
```

Both succeed with only pre-existing unused-parameter and precedence warnings (upstream code, not port regressions). gfx1100 code confirmed in libblst_cuda_msm.a (strings shows `amdgcn-amd-amdhsa--gfx1100`).

### GPU tests (MSM correctness)

```
# bls12_381 G1 Pippenger MSM vs arkworks reference
HIP_VISIBLE_DEVICES=1 NVCC=off HIPCC=/opt/rocm/bin/hipcc TEST_NPOW=15 \
  cargo test --release --features=bls12_381 -- --nocapture
# Result: test msm_correctness ... ok  (1 passed, 0 failed)

# bls12_377 G1 Pippenger MSM vs arkworks reference
HIP_VISIBLE_DEVICES=1 NVCC=off HIPCC=/opt/rocm/bin/hipcc TEST_NPOW=14 \
  cargo test --release --features=bls12_377 -- --nocapture
# Result: test msm_correctness ... ok  (1 passed, 0 failed)
```

### bn254 build-time gate

```
HIP_VISIBLE_DEVICES=1 NVCC=off HIPCC=/opt/rocm/bin/hipcc \
  cargo build --release --features=bn254
# Result: exit 101 -- build.rs panics with
#   "the bn254 curve is not yet supported on the ROCm/HIP MSM backend;
#    use bls12_381 or bls12_377, or the CUDA backend for bn254"
# No GPU object compiled, no binary produced.
```

### NTT regression check (poc/ntt-cuda)

```
HIP_VISIBLE_DEVICES=1 NVCC=off HIPCC=/opt/rocm/bin/hipcc GPU=1 \
  cargo test --release --features=<field> -- --nocapture
```

All 7 field types pass:

| Field | Test | Status |
|-------|------|--------|
| bls12_381 | test_against_arkworks | PASS |
| bls12_377 | test_against_arkworks | PASS |
| bn254 | test_against_arkworks | PASS |
| pallas | test_against_arkworks | PASS |
| vesta | test_against_arkworks | PASS |
| gl64 (Goldilocks) | gl64_self_consistency | PASS |
| bb31 (Baby Bear) | bb31_self_consistency | PASS |

### Summary

| Test | Result |
|------|--------|
| bls12_381 msm_correctness (TEST_NPOW=15) | PASS |
| bls12_377 msm_correctness (TEST_NPOW=14) | PASS |
| bn254 build-time gate | PASS (build refused, exit 101) |
| NTT bls12_381 test_against_arkworks | PASS |
| NTT bls12_377 test_against_arkworks | PASS |
| NTT bn254 test_against_arkworks | PASS |
| NTT pallas test_against_arkworks | PASS |
| NTT vesta test_against_arkworks | PASS |
| NTT gl64 gl64_self_consistency | PASS |
| NTT bb31 bb31_self_consistency | PASS |

0 test failures. Source tree clean (git status --porcelain: empty).

## Validation 2026-06-12 (windows-gfx1201, RX 9070 XT, RDNA4)

GPU: gfx1201 (AMD Radeon RX 9070 XT), TheRock ROCm 7.14.0a20260604 (Windows 11).
HIP_VISIBLE_DEVICES=1 (gfx1201 at device index 0 when masked to 1). Fork HEAD 00cb1a7 (includes all_gpus.cpp fix below).

### Windows ROCm toolchain recipe

```
ROCM_SDK = B:\develop\TheRock\external-builds\pytorch\.venv\Lib\site-packages\_rocm_sdk_devel
HIPCC     = $ROCM_SDK/bin/hipcc.exe   # HIP 7.14, clang 23
MSVC_VER  = 14.44.35207
MSVC_BIN  = C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/$MSVC_VER/bin/Hostx64/x64
WINSDK    = C:/Program Files (x86)/Windows Kits/10/  (10.0.26100.0)
Rust      = stable-x86_64-pc-windows-msvc (1.96.0)

# Environment:
export LIB="$MSVC_WIN\lib\x64;$WINSDK\Lib\10.0.26100.0\ucrt\x64;$WINSDK\Lib\10.0.26100.0\um\x64"
export INCLUDE="$MSVC_WIN\include;$WINSDK\Include\10.0.26100.0\ucrt;..."
export PATH="$MSVC_BIN:$ROCM_SDK/bin:$ROCM_SDK/lib/llvm/bin:$PATH"  # MSVC link.exe before Git's

# GPU discovery via offload-arch.exe: needs TheRock runtime DLLs in the same directory.
# Copy from _rocm_sdk_core/bin/ to _rocm_sdk_devel/lib/llvm/bin/:
#   amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc-builtins0714.dll, hiprtc0714.dll
# Without these DLLs in llvm/bin, offload-arch.exe fails with "cannot determine hip architecture"
# and the build silently falls back to --offload-arch=native (which then fails).

# Test binary DLL placement: the test .exe links amdhip64_7.dll.
# System32 has an Adrenalin amdhip64_7.dll that loads first (app dir > System32 > PATH).
# Fix: copy the same 5 TheRock DLLs into target/release/deps/ before each test run.
```

### Required source fix: util/all_gpus.cpp cooperativeLaunch filter

gfx1201 under TheRock ROCm 7.14 reports `cooperativeLaunch=0` in device properties.
The original `all_gpus.cpp` filtered out GPUs without cooperativeLaunch, silently
excluding gfx1201 from the GPU list -> every test failed with cudaErrorNoDevice.

Fix (commit 00cb1a7): remove `&& prop.cooperativeLaunch` from the GPU selection filter.
On Linux gfx90a/gfx1100 (ROCm 7.2.1), cooperativeLaunch=1 so those platforms are
unaffected. The fix lets NTT tests run on gfx1201; MSM operations that call
hipLaunchCooperativeKernel fail gracefully at the call site with "unspecified launch
failure" (hipErrorLaunchFailure 719) rather than silently at GPU detection.

### NTT tests (poc/ntt-cuda)

Build + test command pattern:
```
source sppark_build_env.sh   # sets LIB, INCLUDE, PATH, HIPCC
cd poc/ntt-cuda
cargo clean
NVCC=off HIPCC=<hipcc.exe> HIP_VISIBLE_DEVICES=1 cargo build --release --features=<field>
# Copy 5 TheRock DLLs to target/release/deps/ (System32 amdhip64 loader workaround)
NVCC=off HIPCC=<hipcc.exe> HIP_VISIBLE_DEVICES=1 GPU=1 \
  cargo test --release --features=<field> -- --nocapture
```

| Field | Test | Result |
|-------|------|--------|
| bls12_381 | test_against_arkworks | PASS |
| bls12_377 | test_against_arkworks | PASS |
| bn254 | test_against_arkworks | PASS |
| pallas | test_against_arkworks | PASS |
| vesta | test_against_arkworks | PASS |
| gl64 (Goldilocks) | gl64_self_consistency | PASS |
| bb31 (Baby Bear) | bb31_self_consistency | PASS |

7/7 NTT fields PASS on gfx1201 RDNA4.

### MSM tests (poc/msm-cuda) - BLOCKED by cooperative launch

gfx1201 under TheRock ROCm 7.14 on Windows does NOT support hipLaunchCooperativeKernel.
`hipLaunchCooperativeKernel` returns hipErrorLaunchFailure (719) even for a 1x1 grid.
The MSM Pippenger algorithm uses cooperative launch extensively (sort kernel, accumulate
kernel). Both bls12_381 and bls12_377 MSM tests fail:

```
thread 'msm_correctness' panicked at src\lib.rs:77:9:
cudaLaunchCooperativeKernel(...) failed: "unspecified launch failure"
test msm_correctness ... FAILED
```

This is a platform limitation of TheRock ROCm 7.14 on Windows gfx1201, not a port
defect. On Linux (ROCm 7.2.1), both gfx90a and gfx1100 pass MSM correctness.

### bn254 build-time gate

```
NVCC=off HIPCC=<hipcc.exe> HIP_VISIBLE_DEVICES=1 cargo build --release --features=bn254
# exit 101, panics: "the bn254 curve is not yet supported on the ROCm/HIP MSM backend..."
```

Gate confirmed working on Windows: exit 101, correct panic message. PASS.

### Summary

| Test | Result |
|------|--------|
| NTT bls12_381 test_against_arkworks | PASS |
| NTT bls12_377 test_against_arkworks | PASS |
| NTT bn254 test_against_arkworks | PASS |
| NTT pallas test_against_arkworks | PASS |
| NTT vesta test_against_arkworks | PASS |
| NTT gl64 gl64_self_consistency | PASS |
| NTT bb31 bb31_self_consistency | PASS |
| MSM bls12_381 msm_correctness (TEST_NPOW=15) | FAIL (hipLaunchCooperativeKernel 719) |
| MSM bls12_377 msm_correctness (TEST_NPOW=14) | FAIL (hipLaunchCooperativeKernel 719) |
| bn254 build-time gate | PASS (exit 101, correct panic) |

Outcome: validation-failed. NTT (7/7 fields) and bn254 build-gate pass. MSM is blocked
by hipLaunchCooperativeKernel returning error 719 on gfx1201 (TheRock ROCm 7.14 Windows,
cooperativeLaunch=0 in device properties). This is a platform limitation, not a port
defect. The all_gpus.cpp fix (commit 00cb1a7) is required for any GPU tests to run on
gfx1201 and has been pushed to the fork.

## Porter assessment of the gfx1201 MSM failure (2026-06-12, porter)

Assessed whether any in-scope, behavior-preserving port-side fix can run MSM on
gfx1201 without cooperative launch. Conclusion: none. Marked windows-gfx1201
blocked.

Re-confirmed the hardware reality on this host:
`HIP_VISIBLE_DEVICES=1 hipInfo` -> device 0 gcnArchName=gfx1201,
cooperativeLaunch=0. So the masked device is the RX 9070 XT and it reports no
cooperative-launch support, matching the documented limitation.

Why there is no port-side fix:
- The failure is in the runtime LAUNCH primitive (hipLaunchCooperativeKernel
  returns 719 even for a 1x1 grid), not in any kernel body. No edit to the
  kernels changes that.
- MSM (Pippenger) fundamentally relies on grid-wide cooperative synchronization:
  its multi-phase sort/accumulate kernels use cooperative_groups
  this_grid().sync() (__ockl_grid_sync) for grid-wide barriers between phases.
  Removing that dependency means reimplementing MSM as separate non-cooperative
  launches with host-side / global-memory inter-phase barriers -- a substantial
  divergence from upstream and out of scope here. Already registered as deferred
  work sppark-msm-noncoop-fallback.
- The port code is correct: gfx90a (CDNA) and gfx1100 (RDNA3) on Linux ROCm
  7.2.1 report cooperativeLaunch=1 and pass MSM bls12_381/bls12_377. The gap is
  specific to TheRock ROCm 7.14 on Windows gfx1201 (RDNA4).

No source change made. The redundant Windows tier (gfx1101/gfx1201/gfx1151) is
satisfied by one Windows arch completing; gfx1101 (RDNA3, supports cooperative
launch) is the intended path and is validated in a separate stage. The
windows-gfx1201 NTT 7/7 + bn254-gate PASS results above stand as a partial
Windows-ROCm proof.

### Cooperative-launch wall: wrong-runtime hypothesis ruled out, gap is Windows-wide (2026-06-12)

jeff asked whether the 719 / cooperativeLaunch=0 reading came from loading the System32
Adrenalin amdhip64 instead of TheRock's runtime. Verified directly with a standalone 1x1
cooperative-launch probe (agent_space/coop_probe/coop2.cpp, hipcc all-clang, built for
gfx1101+gfx1201):

- Ran under all THREE amdhip64_7.dll present on the host:
  - System32 Adrenalin (Mar 2025) -- printed "HIP Library Path: C:\WINDOWS\SYSTEM32\amdhip64_7.dll"
  - TheRock _rocm_sdk_core 7.14 (Jun 2026, the torch-matching runtime; DLLs copied into exe dir)
  - TheRock build/bin 7.0.2 (Apr 2026)
- Result is identical for all three and for BOTH GPUs (gfx1201 RDNA4 device 1, gfx1101 RDNA3 device 0):
  prop.cooperativeLaunch=0, hipDeviceGetAttribute(CooperativeLaunch)=0, and
  hipLaunchCooperativeKernel returns 719 even for a PLAIN 1x1 kernel with no grid.sync at all.
  So the launch API gates on the device property; it is not the grid barrier executing.
- On Windows there is no separate libhsa-runtime DLL (HSA is rolled into amdhip64), so "use
  TheRock's HSA runtime" reduces to "which amdhip64 loaded" -- all three tested.

Conclusion: cooperative launch is unsupported on Windows ROCm 7.14 for both gfx1101 and gfx1201.
The same RDNA3 silicon (gfx1100) reports cooperativeLaunch=1 and passes MSM on Linux ROCm 7.2.1,
so this is a Windows HIP runtime capability gap, not a port defect, not RDNA4-specific, and not a
wrong-runtime artifact. Both windows-gfx1201 and windows-gfx1101 are recorded blocked for MSM;
NTT 7/7 and the bn254 gate pass on Windows. The Windows tier cannot be satisfied for sppark's MSM
on this host. Non-cooperative MSM fallback remains deferred (sppark-msm-noncoop-fallback).

### Root cause of the cooperative-launch failure (AMD maintainers, 2026-06-12)

Traced to a known, AMD-acknowledged issue: ROCm/hip#3803 -> ROCm/rocm-systems#401 (OPEN,
labels "Under Investigation" / "project: hip" / "status: assessed"). AMD's @cjatin gave the
mechanism: hipLaunchCooperativeKernel allocates a cooperative queue whose grid-wide barrier
relies on Global Wave Sync (GWS) hardware. Two distinct causes apply to our two Windows GPUs:

- gfx1201 (RDNA4/Navi4): "Navi4 does not have GWS" -- a genuine HARDWARE gap. Reported
  cooperativeLaunch=0 on Linux too (issue #401 is an Ubuntu RX 9070 XT report), so it is NOT
  Windows-specific and NOT fixable by a runtime version bump. AMD recommends avoiding
  cooperative launch on Navi4 (use hipLaunchKernel + a non-GWS grid sync).
- gfx1101 (RDNA3): the silicon HAS GWS (Linux gfx1100 reports cooperativeLaunch=1 and runs
  MSM), but the Windows driver/runtime under-reports coop queues as 0. cjatin expected this
  "sorted in rocm 7.0", yet it is still 0 on our TheRock ROCm 7.14 + Adrenalin stack.

Conclusion: this is NOT a HIP runtime defect we can patch and NOT a sppark port defect. For
RDNA4 it is a hardware capability gap; for RDNA3-on-Windows it is an unresolved driver/runtime
reporting gap. sppark MSM validates correctly on Linux gfx90a + gfx1100 (both have GWS); the
Windows tier cannot be satisfied for MSM on this host. The non-cooperative MSM fallback
(deferred: sppark-msm-noncoop-fallback) remains the only path to MSM on RDNA4/Windows.

## Non-cooperative-launch fallback IMPLEMENTED + MSM validated on Windows (2026-06-12)

Implemented the deferred `sppark-msm-noncoop-fallback`: a runtime fallback that runs
the Pippenger MSM without cooperative launch, so MSM now works on the no-GWS Windows
GPUs. Fork commit 868aa18 (on top of the validated 00cb1a7), pushed to
AMD-Ecosystem/sppark moat-port. Design: projects/sppark/noncoop-fallback-design.md.

What changed (additive; cooperative path byte-for-byte unchanged):
- msm/sort.cuh: upper_sort factored into upper_count_emit + upper_scan_scatter phase
  helpers (the cooperative upper_sort still calls them with grid.sync between, so its
  behavior is unchanged). New fallback kernels sort_upper_count -> sort_upper_scatter
  -> sort_lower run the two-level cross-block radix pass as three ordinary kernels
  (kernel boundaries replace the grid barriers); the single-block sort_row path reuses
  sort with gridDim.x == 1. This is tractable because sppark already routes all
  cross-block state through the global histogram[]; shared counters[] is per-block
  scratch recomputed after each barrier.
- msm/pippenger.cuh: accumulate gains a COOP template flag; COOP=false drops the
  trailing grid.sync + in-kernel work-queue reset (its counter hoisted to namespace
  scope and zeroed by reset_accumulate_counter before each launch). msm_t::launch_sort
  / launch_accumulate dispatch on gpu.props().cooperativeLaunch (overridable with
  SPPARK_FORCE_NONCOOP). Requires C++17 (if constexpr) -- satisfied by the ROCm clang
  and modern CUDA toolchains.
- poc/msm-cuda/tests/msm.rs: added msm_correctness_sizes, a multi-size sweep vs
  arkworks that exercises the fallback (or, with SPPARK_FORCE_NONCOOP=1, forces it on a
  cooperative-launch-capable GPU).

Not touched: the bitmap-based batch_addition/batch_diff (add<>) cooperative overload is
NOT on the MSM correctness path (the digit-list batch_addition overload is, and it has
no grid barrier), so it was left as-is. If a non-MSM-correctness entry point that uses
add<> is ever run on no-GWS hardware it would need the same easy treatment as accumulate.

### Validation (real GPU, msm_correctness + msm_correctness_sizes vs arkworks)

Build/test recipe: agent_space/sppark_build_env.sh (MSVC 14.44.35207, WinSDK
10.0.26100.0, hipcc 7.14/clang 23), build per arch under the matching
HIP_VISIBLE_DEVICES mask (offload-arch bakes in the masked device's arch), copy the 5
TheRock runtime DLLs into target/release/deps/.

- windows-gfx1201 (RX 9070 XT, RDNA4, device 1): bls12_381 PASS at TEST_NPOW 8,10,12,
  15,17; bls12_377 PASS at TEST_NPOW 14; msm_correctness_sizes PASS; SPPARK_FORCE_NONCOOP
  PASS. MSM previously failed here with hipLaunchCooperativeKernel 719.
- windows-gfx1101 (Radeon PRO V710, RDNA3, device 0): bls12_381 PASS at TEST_NPOW 10,
  15,17 (built for gfx1101 under mask 0).

windows-gfx1201 and windows-gfx1101 are now `completed` at 868aa18 (MSM + NTT both pass).
The functional change advanced head 00cb1a7 -> 868aa18, so linux-gfx90a / linux-gfx1100
flipped to `revalidate`: the cooperative path they exercise is behavior-identical, but
the device binary gained the fallback kernels, so a binary-equivalence carry-forward does
not apply and a Linux GPU should re-run msm_correctness at 868aa18 to reconfirm (expected
PASS). This is required before the upstream PR (REQUIRED Linux tier).

## Validation 2026-06-12 (linux-gfx90a, revalidate at 868aa18)

GPU: gfx90a (AMD Instinct MI250X, GCD 0), ROCm 7.2.1. HIP_VISIBLE_DEVICES=0. Fork HEAD 868aa18.

This is a FULL GPU re-run: commit 868aa18 adds the non-cooperative-launch fallback (device kernels changed in msm/sort.cuh +188, msm/pippenger.cuh +127) and the new msm_correctness_sizes test. Binary-equivalence carry-forward does not apply.

### Build

```
cd poc/msm-cuda
HIP_VISIBLE_DEVICES=0 NVCC=off HIPCC=/opt/rocm/bin/hipcc cargo build --release --features=bls12_381  # exit 0
HIP_VISIBLE_DEVICES=0 NVCC=off HIPCC=/opt/rocm/bin/hipcc cargo build --release --features=bls12_377  # exit 0
```

Both succeed. Warnings are pre-existing upstream code (unused-parameter, precedence) -- not port regressions.

### GPU tests (MSM correctness + new msm_correctness_sizes)

```
# bls12_381 G1 Pippenger MSM -- cooperative path (gfx90a has cooperativeLaunch=1)
HIP_VISIBLE_DEVICES=0 NVCC=off HIPCC=/opt/rocm/bin/hipcc TEST_NPOW=15 \
  cargo test --release --features=bls12_381 -- --nocapture
# Result: msm_correctness ... ok, msm_correctness_sizes ... ok  (2 passed, 0 failed)

# bls12_377 G1 Pippenger MSM -- cooperative path
HIP_VISIBLE_DEVICES=0 NVCC=off HIPCC=/opt/rocm/bin/hipcc TEST_NPOW=14 \
  cargo test --release --features=bls12_377 -- --nocapture
# Result: msm_correctness ... ok, msm_correctness_sizes ... ok  (2 passed, 0 failed)

# Non-cooperative fallback path forced on gfx90a (which has cooperative launch)
HIP_VISIBLE_DEVICES=0 NVCC=off HIPCC=/opt/rocm/bin/hipcc TEST_NPOW=12 SPPARK_FORCE_NONCOOP=1 \
  cargo test --release --features=bls12_381 msm_correctness_sizes -- --nocapture
# Result: msm_correctness_sizes ... ok  (1 passed, 0 failed)
```

### bn254 build-time gate

```
HIP_VISIBLE_DEVICES=0 NVCC=off HIPCC=/opt/rocm/bin/hipcc cargo build --release --features=bn254
# exit 101, panics: "the bn254 curve is not yet supported on the ROCm/HIP MSM backend..."
```

Gate confirmed: --features=bn254 on ROCm is refused at build time (exit 101). No GPU object compiled.

### Summary

| Test | Result |
|------|--------|
| bls12_381 msm_correctness (TEST_NPOW=15) | PASS |
| bls12_381 msm_correctness_sizes (npow 4,8,10,12,14,16) | PASS |
| bls12_377 msm_correctness (TEST_NPOW=14) | PASS |
| bls12_377 msm_correctness_sizes (npow 4,8,10,12,14,16) | PASS |
| bls12_381 msm_correctness_sizes SPPARK_FORCE_NONCOOP=1 (fallback path) | PASS |
| bn254 build-time gate | PASS (exit 101, correct panic) |

0 test failures. Source tree clean (git status --porcelain: empty). Advancing to completed.

## Validation 2026-06-15 (linux-gfx1100, revalidate at 868aa18)

GPU: gfx1100 (AMD Radeon Pro W7800 48GB), ROCm 7.2.1. HIP_VISIBLE_DEVICES=0. Fork HEAD 868aa18.

Full GPU re-run: commit 868aa18 adds the non-cooperative-launch fallback (device kernels changed: msm/sort.cuh +188, msm/pippenger.cuh +127) and the new msm_correctness_sizes test. Binary-equivalence carry-forward does not apply (device code changed). gfx1100 has cooperativeLaunch=1 so it takes the cooperative path; the fallback kernels are compiled but not invoked for msm_correctness.

### Build

```
cd poc/msm-cuda
HIP_VISIBLE_DEVICES=0 NVCC=off HIPCC=/opt/rocm/bin/hipcc cargo build --release --features=bls12_381  # exit 0
HIP_VISIBLE_DEVICES=0 NVCC=off HIPCC=/opt/rocm/bin/hipcc cargo build --release --features=bls12_377  # exit 0
```

Both succeed. Warnings are pre-existing upstream code (unused-parameter, precedence) -- not port regressions.

### GPU tests (MSM correctness + msm_correctness_sizes)

```
# bls12_381 G1 Pippenger MSM -- cooperative path (gfx1100 has cooperativeLaunch=1)
HIP_VISIBLE_DEVICES=0 NVCC=off HIPCC=/opt/rocm/bin/hipcc TEST_NPOW=15 \
  cargo test --release --features=bls12_381 -- --nocapture
# Result: msm_correctness ... ok, msm_correctness_sizes ... ok  (2 passed, 0 failed)

# bls12_377 G1 Pippenger MSM -- cooperative path
HIP_VISIBLE_DEVICES=0 NVCC=off HIPCC=/opt/rocm/bin/hipcc TEST_NPOW=14 \
  cargo test --release --features=bls12_377 -- --nocapture
# Result: msm_correctness ... ok, msm_correctness_sizes ... ok  (2 passed, 0 failed)
```

### bn254 build-time gate

```
HIP_VISIBLE_DEVICES=0 NVCC=off HIPCC=/opt/rocm/bin/hipcc cargo build --release --features=bn254
# exit 101, panics: "the bn254 curve is not yet supported on the ROCm/HIP MSM backend..."
```

Gate confirmed: --features=bn254 on ROCm is refused at build time (exit 101). No GPU object compiled.

### NTT regression check (poc/ntt-cuda)

All 7 field types pass:

| Field | Test | Status |
|-------|------|--------|
| bls12_381 | test_against_arkworks | PASS |
| bls12_377 | test_against_arkworks | PASS |
| bn254 | test_against_arkworks | PASS |
| pallas | test_against_arkworks | PASS |
| vesta | test_against_arkworks | PASS |
| gl64 (Goldilocks) | gl64_self_consistency | PASS |
| bb31 (Baby Bear) | bb31_self_consistency | PASS |

### Summary

| Test | Result |
|------|--------|
| bls12_381 msm_correctness (TEST_NPOW=15) | PASS |
| bls12_381 msm_correctness_sizes | PASS |
| bls12_377 msm_correctness (TEST_NPOW=14) | PASS |
| bls12_377 msm_correctness_sizes | PASS |
| bn254 build-time gate | PASS (exit 101, correct panic) |
| NTT bls12_381 test_against_arkworks | PASS |
| NTT bls12_377 test_against_arkworks | PASS |
| NTT bn254 test_against_arkworks | PASS |
| NTT pallas test_against_arkworks | PASS |
| NTT vesta test_against_arkworks | PASS |
| NTT gl64 gl64_self_consistency | PASS |
| NTT bb31 bb31_self_consistency | PASS |

0 test failures. Source tree clean (git status --porcelain: empty). Advancing to completed.
