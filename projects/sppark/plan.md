# sppark ROCm/HIP Port Plan

## Project
- **Name:** sppark
- **Upstream:** https://github.com/supranational/sppark
- **Default Branch:** main
- **Description:** High-performance primitives for zero-knowledge proofs (MSM, NTT, finite fields) in CUDA/C++ with Rust and Go bindings

## Existing AMD Support

**Authoritative, upstream-maintained HIP support already present.**

sppark includes first-party ROCm/HIP support from Supranational. The README.md states: "A limited support for AMD's RDNA and CDNA GPUs is provided."

Evidence of existing HIP infrastructure:
- `util/cuda2hip.hpp` -- comprehensive CUDA-to-HIP shim header with runtime API mappings and warp intrinsic polyfills
- `ff/mont_t.hip`, `ff/gl64_t.hip`, `ff/mont32_t.hip` -- HIP-specific finite field implementations with inline GCN assembly for CDNA (gfx9) and RDNA (gfx10/11/12)
- `rust/src/build.rs` -- full ROCm build path with multi-arch support (`--offload-arch=gfx90a,gfx942,gfx908,gfx1100-1102,gfx1030-1034,gfx1200-1201`)
- `rust/build.rs` -- runtime detection of `hipcc`, version check (>= 5.7), automatic HIP toolchain engagement
- Architecture-aware wavefront size handling (`__GFX9__` -> 64, `__GFX10__`/`__GFX11__`/`__GFX12__` -> 32)
- Architecture-specific inline assembly (`v_addc_co_u32` vs `v_add_co_ci_u32` for CDNA vs RDNA)

The HIP support is **authoritative** (upstream, from the original Supranational developers) and **mature** (covers multiple GCN architectures, includes proper warp size handling, and has been tested on RDNA and CDNA).

**Decision:** This project already has authoritative ROCm/HIP support from upstream. MOAT value shifts from "fresh port" to "validation and improvement": build and run the existing HIP path on gfx90a, verify correctness, and contribute fixes for any rot or failures back to upstream.

## Build Classification

**Build type:** Rust/Cargo with `cc` crate + runtime compiler detection (nvcc/hipcc)

This is NOT a pure CMake project and NOT a pytorch extension. sppark is a header-only CUDA/HIP library with Rust crate wrappers:
- `rust/` -- main sppark crate with `build.rs` that detects nvcc/hipcc
- `rust/src/build.rs` -- provides `ccmd()` function returning either CUDA or ROCm `cc::Build`
- `poc/msm-cuda/` and `poc/ntt-cuda/` -- proof-of-concept Rust crates with tests and benchmarks

Evidence:
- `rust/Cargo.toml` exists with `[features] cuda, rocm`
- `rust/build.rs` lines 54-63: checks for `NVCC` env var or `which nvcc`; lines 59-63: checks for `HIPCC` or `which hipcc`
- `rust/build.rs` lines 141-214: full ROCm/hipcc build path including lib linking (`amdhip64`)
- `rust/src/build.rs` lines 69-113: ROCm `ccmd()` path with multi-arch `--offload-arch` flags

**Port Strategy:** N/A (validate existing)

Since authoritative HIP support exists upstream, this is a **validation project**, not a fresh port. The MOAT value is:
1. Build the existing HIP path on gfx90a
2. Run the NTT and MSM test suites
3. Identify and fix any bitrot or failures
4. Contribute fixes upstream if needed

## CUDA Surface Inventory

The existing HIP support already handles most of the CUDA surface. Key mappings in `util/cuda2hip.hpp`:

### Runtime API (already ported)
- `cudaMalloc` -> `hipMalloc` (template wrapper)
- `cudaFree` -> `hipFree`
- `cudaMemcpy` -> `hipMemcpy`
- `cudaMemcpyAsync` -> `hipMemcpyAsync`
- `cudaStream_t` -> `hipStream_t`
- `cudaEvent_t` -> `hipEvent_t`
- `cudaDeviceSynchronize` -> `hipDeviceSynchronize`
- `cudaLaunchCooperativeKernel` -> `hipLaunchCooperativeKernel`

### Warp Intrinsics (custom polyfills in cuda2hip.hpp)
- `__shfl_sync` -> custom using `__builtin_amdgcn_ds_bpermute`
- `__shfl_up_sync` -> custom using `__builtin_amdgcn_ds_bpermute`
- `__shfl_down_sync` -> custom using `__builtin_amdgcn_ds_bpermute`
- `__shfl_xor_sync` -> custom using `__builtin_amdgcn_ds_bpermute`
- `__ballot_sync` -> custom with wave64 ballot splitting
- `__syncwarp` -> `__builtin_amdgcn_wave_barrier` (HIP < 7)

The polyfills use `WARP_SZ = 32` logical warp semantics on top of wave64, which is appropriate for the NTT/MSM algorithms.

### Finite Field Math (architecture-specific implementations)
- `mont_t.hip` -- Montgomery multiplication with GCN inline assembly
- `gl64_t.hip` -- Goldilocks field with GCN assembly
- `mont32_t.hip` -- 32-bit Montgomery field with GCN assembly

These use architecture-specific instructions:
- CDNA (gfx9): `v_addc_co_u32`, `v_subb_co_u32`, `s_orn2_b64`
- RDNA (gfx10/11/12): `v_add_co_ci_u32`, `v_sub_co_ci_u32`, `s_orn2_b32`

### PTX Assembly (needs translation or gating)
Some files contain inline PTX that is CUDA-only:
- `msm/batch_addition.cuh:42` -- `mov.u32 %laneid` (lane id query)
- `msm/pippenger.cuh:160` -- `mov.u32 %laneid`
- `msm/pippenger.cuh:185-190` -- PTX predicated shuffle
- `msm/sort.cuh:45` -- `lop3.b32` (ternary logic op)
- `msm/sort.cuh:57-60`, `169-172` -- PTX predicated shuffle
- `ff/baby_bear.hpp` -- extensive PTX inline assembly

These are likely gated by `__NVCC__` / `__HIPCC__` or compiled only in CUDA mode. The `.hip` variants provide HIP-specific implementations.

### Libraries
- No cuBLAS/cuFFT/cuRAND/cuSPARSE usage
- No Thrust/CUB usage
- Pure custom kernels for all operations

## Risk List

1. **Build system complexity** -- Rust/Cargo build with runtime compiler detection; need to verify `HIPCC` detection and `DEP_SPPARK_TARGET=rocm` propagation work correctly on ROCm 7.2+

2. **Multi-arch fat binary** -- The build.rs specifies many `--offload-arch` targets; need to verify gfx90a is properly included and builds correctly

3. **PTX inline assembly in CUDA-only paths** -- Files like `ff/baby_bear.hpp` contain PTX assembly that must be gated to CUDA builds only (verify `__NVCC__` guards are correct)

4. **Wave64 ballot splitting** -- The `__ballot_sync` polyfill splits 64-bit ballot into two 32-bit halves based on `threadIdx.x & WARP_SZ`. Verify correctness on wave64 gfx90a.

5. **GCN instruction encoding differences** -- CDNA vs RDNA instruction variants (`v_addc_co_u32` vs `v_add_co_ci_u32`); verify `__GFX9__` macro is correctly defined on gfx90a

6. **ROCm version compatibility** -- Build.rs requires HIP >= 5.7; need ROCm 7.2.x which satisfies this

7. **Test coverage on AMD** -- The POC tests use arkworks as reference; verify the HIP path produces identical results

## File-by-File Change List

**No source changes expected.** This is a validation of existing HIP support. Potential fixes if issues arise:

| File | Potential Issue |
|------|-----------------|
| `rust/build.rs` | Verify HIPCC detection on ROCm 7.2 |
| `rust/src/build.rs` | Verify `--offload-arch=gfx90a` is included in the list |
| `util/cuda2hip.hpp` | Any missing API mappings or warp intrinsic issues |
| `ff/mont_t.hip` | GCN assembly correctness on gfx90a |
| `ntt/kernels.cu` | HIP-specific codepaths (`__HIP_DEVICE_COMPILE__` guards) |

## Build Commands

### Prerequisites
```bash
# ROCm 7.2.x with hipcc
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH

# Rust toolchain
rustup default stable

# blst (BLS12-381 library, required dependency)
# Will be fetched by Cargo
```

### Build NTT POC (gfx90a)
```bash
cd projects/sppark/src/poc/ntt-cuda

# Build with ROCm/HIP for gfx90a, using bls12_381 field
HIPCC=/opt/rocm/bin/hipcc \
cargo build --release --features=rocm,bls12_381
```

### Build MSM POC (gfx90a)
```bash
cd projects/sppark/src/poc/msm-cuda

# Build with ROCm/HIP for gfx90a, using bls12_381 curve
HIPCC=/opt/rocm/bin/hipcc \
cargo build --release --features=rocm,bls12_381
```

### Alternative: Use build-poc feature (native arch detection)
```bash
# Uses --offload-arch=native for current GPU
HIPCC=/opt/rocm/bin/hipcc \
cargo build --release --features=rocm,build-poc,bls12_381
```

## Test Plan

### NTT Correctness Tests
```bash
cd projects/sppark/src/poc/ntt-cuda

# Test against arkworks reference for multiple fields
HIPCC=/opt/rocm/bin/hipcc cargo test --release --features=rocm,bls12_381 -- --nocapture
HIPCC=/opt/rocm/bin/hipcc cargo test --release --features=rocm,bls12_377 -- --nocapture
HIPCC=/opt/rocm/bin/hipcc cargo test --release --features=rocm,bn254 -- --nocapture
HIPCC=/opt/rocm/bin/hipcc cargo test --release --features=rocm,pallas -- --nocapture
HIPCC=/opt/rocm/bin/hipcc cargo test --release --features=rocm,vesta -- --nocapture

# Self-consistency tests for Goldilocks and Baby Bear fields
HIPCC=/opt/rocm/bin/hipcc cargo test --release --features=rocm,gl64 -- --nocapture
HIPCC=/opt/rocm/bin/hipcc cargo test --release --features=rocm,bb31 -- --nocapture
```

### MSM Correctness Tests
```bash
cd projects/sppark/src/poc/msm-cuda

# Test against arkworks MSM reference
HIPCC=/opt/rocm/bin/hipcc cargo test --release --features=rocm,bls12_381 -- --nocapture
HIPCC=/opt/rocm/bin/hipcc cargo test --release --features=rocm,bls12_377 -- --nocapture
HIPCC=/opt/rocm/bin/hipcc cargo test --release --features=rocm,bn254 -- --nocapture
```

### Non-GPU Tests
The tests use arkworks as a CPU reference, so the non-GPU portion is the arkworks computation. No separate non-GPU test suite exists.

### Benchmarks (optional performance validation)
```bash
cd projects/sppark/src/poc/msm-cuda
HIPCC=/opt/rocm/bin/hipcc cargo bench --release --features=rocm,bls12_381
```

## Open Questions

1. **Upstream PR or just validation?** Since sppark already has authoritative HIP support, should MOAT:
   - Simply validate it works on gfx90a and report success?
   - Contribute any discovered fixes back to upstream?
   - Open a PR with validation evidence (test results)?

2. **Performance baseline:** Should we capture benchmarks to compare NVIDIA vs AMD performance?

3. **Multi-field testing:** The test matrix (7 fields x 2 algorithms) is large. Should we test all combinations or focus on the most common (bls12_381)?

4. **Go bindings:** The `go/` and `poc/go/` directories have Go/cgo bindings. Should these be tested on ROCm as well?

## Recommendation

**Disposition: Validate and improve (not skip)**

While sppark has existing authoritative HIP support, it still provides MOAT value:
1. **Real GPU validation** -- The upstream HIP code may not have been tested recently on gfx90a; MOAT can confirm it works
2. **CI/test evidence** -- Provide concrete test results for MI200 hardware
3. **Bitrot detection** -- If issues are found, fix them and contribute upstream
4. **Documentation** -- Document the exact build/test commands that work on ROCm 7.2.x

Proceed to porter phase for validation (not porting). The porter should:
1. Build both NTT and MSM POCs with `--features=rocm`
2. Run the correctness tests against arkworks
3. Document any failures and fix them
4. If all tests pass, mark as validated

If validation passes with no changes needed, the "port" is simply confirming upstream's HIP support works. If fixes are needed, contribute them upstream.
