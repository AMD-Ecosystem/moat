# mahout -- Implicit Hadamard Ozaki tensor-core engine: AMD (ROCm/HIP) follow-up plan

Internal MOAT planning doc (not upstream-facing). Targets a FOLLOW-UP feature PR
that adds an AMD matrix-core implementation of apache/mahout PR #1390's
"implicit Hadamard Ozaki" engine on top of our already-landed QDP HIP
infrastructure (PR #1399, AMD-Ecosystem/mahout `moat-port`). This plan is read-only on
code; no fork edits are made by the planner.

## Project
- Name: mahout (QDP Rust workspace under `qdp/`; NOT JVM Mahout).
- Upstream: https://github.com/apache/mahout, default branch `main`.
- This feature's upstream source: PR #1390 by aloha1357
  (`pr5-implicit-hadamard-engine`, head e25ce590), OPEN/unmerged, MERGEABLE.
- Our HIP base: PR #1399 (open), fork `moat-port`. Lead linux-gfx90a validated;
  the unified HIP build (`QDP_USE_HIP=1`, Cargo `hip` feature, hipcc-compiled `.cu`,
  `hip_compat/` shim, `gpu_rt`/`cuda_ffi` indirection) is the substrate this rides on.

## Existing AMD support assessment
- #1390 is a pure CUDA implementation: `nvcuda::wmma` (`<mma.h>`), raw PTX
  `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`, `ldmatrix.sync` PTX, and
  `__cvta_generic_to_shared`. There is NO AMD/ROCm path in the PR and none exists
  upstream for this engine (it is brand-new, unmerged CUDA code). So this is a
  genuine from-scratch HIP/matrix-core contribution, not a duplicate of any AMD
  effort. No skip.
- The Ozaki accurate-matmul scheme itself (int8 error-free transform for an
  accurate FWT) is a known technique; AMD has no drop-in library for it. We
  reimplement the kernel against AMD matrix cores.

## Build classification
- ext_type `rust-cc-cuda` (unchanged from #1399). The kernels are hand-written
  `.cu` compiled by the Rust `cc` crate; on AMD our `build_hip()` compiles the same
  `.cu` with hipcc + `--offload-arch`. This is NOT a CMake or torch-extension
  project; Strategy A/B do not apply verbatim. The relevant substrate is #1399's
  `build.rs` HIP branch + `hip_compat/` shim + `gpu_rt`/`cuda_ffi` Rust indirection.
- Evidence: `qdp/qdp-kernels/build.rs` (`build_hip`, `KERNEL_SOURCES`,
  `--offload-arch` from `QDP_HIP_ARCH_LIST`); `#1390` adds
  `.file("src/iqp_tc.cu").file("src/ImplicitHadamardOzaki.cu")` to the CUDA build
  list.

## What #1390 actually does (CUDA reference, the spec to reproduce)

The IQP "encode_batch_tc" path computes a batched complex IQP state vector whose
core is a Fast Walsh-Hadamard Transform (FWT). For num_qubits <= 12 it uses a
shared-memory butterfly FWT (`iqp_phase_fwt_normalize_tc_kernel` in iqp_tc.cu --
no tensor cores, pure double, ports trivially under HIP). For larger N it does a
**Kronecker decomposition** of the Hadamard: H_N = H_{n1} (x) H_{n2}, realized as
two batched matmuls against Hadamard factor matrices with a transpose between
(iqp_tc.cu launch_iqp_encode_tc, the `else` branch):
  1. phase split into real/imag double buffers (iqp_phase_split_kernel),
  2. Z = X * H_{n2}  (execute_implicit_hadamard),
  3. transpose (iqp_tc_batch_transpose_kernel, bank-conflict-padded),
  4. Y_T = Z_T * H_{n1} (execute_implicit_hadamard, applies norm_factor),
  5. transpose back, recombine real+imag into cuDoubleComplex.

The tensor-core heart is `execute_implicit_hadamard` (ImplicitHadamardOzaki.cu),
an **Ozaki int8 error-free matmul** that computes an accurate double-precision
`C = (X * H) * norm * inv` WITHOUT ever materializing H and using only int8
tensor-core matmul:

- **Ozaki / residue split (precompute_modulo_kernel_p26_implicit):** each double
  input element v is scaled by 2^30 and rounded to a 64-bit-ish integer iv, then
  reduced modulo 7 primes pr[] = {127,113,109,107,103,101,97}. Each prime's
  residue is stored as one int8 plane (7 planes total), in a 128x32 tile-packed
  layout (`get_A8_offset`). This is a Residue Number System (RNS) split: the int8
  matmul is done independently per prime, and CRT reconstructs the exact integer.
- **Implicit Hadamard (matrix-free):** the B operand (the Hadamard factor) is
  generated on the fly in shared memory from the sign rule
  `parity = popcount((kk+r) & (tile_n+c)) & 1; val = parity? h_neg[p] : h_pos[p]`
  where h_pos[p]=1 mod pr[p], h_neg[p]=(-1) mod pr[p]. No H matrix in DRAM.
- **Tensor-core matmul (the NVIDIA-specific core):** per prime, per 64x64 output
  tile, it accumulates over K in steps of 32 using
  `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (int8 m16n8k32 -> int32),
  with `ldmatrix.sync.aligned.m8n8.x4/.x2.shared.b16` to load A/B fragments from
  shared memory into the warp's registers. Accumulators are
  `int32 prime_acc[7][4][8][4]` -- 7 primes x a 4x8 grid of 16x8 sub-tiles x 4
  int32 regs each. 8 warps (256 threads), warp grid wr=warp/4, wc=warp%4; each
  warp owns a 32x16 region, looping mt in {0,1} (two m16 rows) and nt in {0,1}
  (two n8 cols).
- **CRT reconstruction + recentre (post-matmul):** per output element, combine the
  7 int32 residues via fixed multipliers f[] mod M=168897325606883 (a 47-bit
  CRT modulus), recentre to signed (subtract M if > M/2), convert to double, and
  scale by `norm_factor * inv` (inv = 2^-30). This is the error-free-transform
  reconstruction that makes the int8 matmul accurate.

Two launch variants exist: a persistent work-queue kernel (dynamic shared mem
~58 KiB, default) and a one-tile-per-block grid kernel (28 KiB static,
`OZAKI_NCU_PROFILE`). They share `implicit_ozaki_process_one_tile<SingleBuffer>`.

Rust/host integration (the glue we mirror): `qdp-kernels/src/lib.rs` declares
`extern "C" launch_iqp_encode_tc(...)` + a non-Linux/no-cuda stub returning 999;
`build.rs` adds the two new `.cu`; `qdp-core/src/gpu/encodings/iqp.rs` adds
`IqpEncoder::encode_batch_tc`; `qdp-core/src/lib.rs` adds
`QdpEngine::encode_batch_tc -> *mut DLManagedTensor`; `qdp-python/src/engine.rs`
exposes `encode_batch_tc`; `testing/qdp/test_iqp_tc_path.py` validates
normalization (N in {8,12}) and finiteness/smoke (N=14) against the FWT path.

## Port strategy: reimplement the matrix-core kernel, mechanically port the rest

Split #1390 into two classes:

1. **Mechanically portable under #1399's HIP build (no rewrite):** iqp_tc.cu
   except `execute_implicit_hadamard` -- the shared-memory FWT kernel, phase
   split, bank-conflict-padded transpose, recombine, and the host
   `launch_iqp_encode_tc` dispatcher (cudaMalloc/Free/sync, all already aliased by
   `hip_compat/cuda_runtime.h`). `cuComplex`/`cuCadd`/`cuCsub`/`make_cuDoubleComplex`
   are already shimmed (notes.md: hip_compat/cuComplex.h, double helpers). `sincos`,
   `__popcll`, `__restrict__` port directly. AdaptiveOzaki.h / ImplicitHadamardOzaki.h
   are plain host headers (drop `<cuda_runtime.h>` via the shim).

2. **AMD-native rewrite (the matrix-core core):** `execute_implicit_hadamard` +
   `implicit_ozaki_process_one_tile` cannot port mechanically -- `nvcuda::wmma`,
   `mma.sync ... m16n8k32 ... s8`, `ldmatrix.sync`, and `__cvta_generic_to_shared`
   are NVIDIA PTX/intrinsics with NO HIP equivalent. PORTING_GUIDE is explicit:
   the planner decides port-vs-rewrite for tensor-core kernels, and CUTLASS/PTX-MMA
   does not port. We REWRITE this kernel against AMD matrix cores (MFMA), keeping
   the Ozaki RNS split, implicit-Hadamard sign generation, and CRT reconstruction
   byte-for-byte at the algorithm level (those are arch-independent integer math)
   and re-deriving ONLY the tensor-core tiling + fragment layout for MFMA/wave64.

The CUDA path stays untouched: the new AMD kernel lives behind the same
`#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)` split, with the CUDA
`mma.sync`/`ldmatrix` code in the `#else`.

## MFMA mapping (gfx90a / CDNA2 int8 matrix core)

Authoritative shapes (ROCm amd_matrix_instruction_calculator; AMD matrix-cores
blog; rocWMMA gfx90a table). gfx90a int8 (I8 in -> I32 acc) MFMA instructions:
- `v_mfma_i32_32x32x8i8`   (M=N=32, K=8)
- `v_mfma_i32_16x16x16i8`  (M=N=16, K=16)
- plus smaller 4x/8x variants (4x4x4, 16x16x4, 32x32x4).
**Maximum int8 K on gfx90a is 16** -- there is NO K=32 int8 MFMA on CDNA2
(K=32/K=64 int8 only appear on gfx950/CDNA3+). MFMA is a 64-lane (wave64)
collective: A/B operands and the C/D accumulator are distributed across all 64
lanes of the wavefront, NOT 32 as on an NVIDIA warp.

rocWMMA confirms `int8_t -> int32_t` fragments on gfx90a at BlockM/N/K = 16/16/16
and 32/32/8 (its API maps 1:1 onto the two MFMA shapes above; larger K is handled
by looping the fragment over the K dimension).

### Recommendation: rocWMMA first, with a builtins fallback escape hatch
Use **rocWMMA `16x16x16` int8 fragments** (matrix_a row_major, matrix_b col_major
to match #1390's `.row.col`, accumulator int32) as the primary implementation:
- Closest 1:1 to #1390's `nvcuda::wmma` mental model and to its `m16n8k32` tile
  (n8 just becomes n16; see re-derivation), lowest-risk path to correctness.
- rocWMMA hides the wave64 lane distribution and the load_matrix_sync layout, so
  we do NOT hand-write the `ldmatrix` equivalent or the 64-lane operand mapping.
- int32 accumulate is native; no accuracy loss vs. a hand MFMA (same hardware op).
Tradeoff: rocWMMA adds a header-only dep (ships with ROCm; add to the HIP include
path in build_hip) and gives less control over shared-memory staging than raw
`__builtin_amdgcn_mfma_i32_16x16x16i8`. If profiling later shows rocWMMA's
fragment loads dominate, drop to the builtin for the inner loop (same 16x16x16
op, explicit AGPR accumulators) as a perf follow-up -- NOT required for the
correctness-first PR. Recommend rocWMMA for the first landed version.

(RDNA note below: rocWMMA int8 also exists on gfx11/gfx12 via WMMA, 16x16x16,
wave32 -- so a single rocWMMA-based kernel can target both CDNA and RDNA, unlike a
gfx9-only MFMA builtin. This is a strong additional reason to prefer rocWMMA.)

## Fragment-shape re-derivation (m16n8k32 s8 -> AMD 16x16x16 i8, wave64)

The CUDA tile math is built around NVIDIA's `m16n8k32` op and 32-lane warps; every
constant tied to those must be re-derived. Concretely:

- **K step 32 -> two K=16 MFMA.** The CUDA inner loop advances `kk += 32` and does
  one m16n8k32 (K=32) per step. gfx90a int8 max K is 16, so each 32-wide K slab
  becomes TWO 16x16x16 accumulate calls (K in {0..15},{16..31}) into the same
  int32 accumulator. The RNS/CRT math is unchanged; only the contraction is split.
  Net integer result is identical (int32 accumulation is associative and exact
  here -- see accuracy).
- **N 8 -> 16.** CUDA uses n8 (m16n8k32) and packs two n8 halves
  (`bf[nt*2..]`) per 16-wide B load. AMD 16x16x16 is square (n16), so one MFMA
  covers what CUDA did in two `nt` iterations. The 64x64 output tile becomes a
  4x4 grid of 16x16 sub-tiles (vs CUDA's 4x8 grid of 16x8). Re-derive
  `prime_acc` to `int32 acc[7][TM][TN]` where TM=TN=4 sub-tiles per warp-tile and
  each MFMA produces a 16x16 block (accumulator is the rocWMMA fragment's
  per-lane register array, not a flat [4] like CUDA's m16n8 4-reg layout).
- **Warp/wavefront grid.** CUDA: 256 threads = 8 warps of 32, wr=warp/4 (2),
  wc=warp%4 (4), each warp owns 32x16. AMD wave64: 256 threads = 4 wavefronts of
  64. Re-tile the 64x64 output across 4 wavefronts (e.g. wr in {0,1}, wc in {0,1},
  each wavefront owns a 32x32 quadrant = 2x2 grid of 16x16 MFMA), or keep 64x64
  per block with 4 wavefronts each doing a 64x16 column strip -- choose the layout
  that keeps shared-memory A/B reuse high. Key rule (PORTING_GUIDE AutoDock-GPU):
  gate every wavefront-stride / lane span on `warpSize` (64 on CDNA), never a
  literal 32 or 64.
- **Shared-memory staging + loads.** Drop `ldmatrix.sync` /
  `__cvta_generic_to_shared` entirely; rocWMMA `load_matrix_sync(frag, smem_ptr,
  ld)` replaces them. Keep the implicit-Hadamard B generation (popcount sign rule)
  and the Ozaki A residue tiling, but the in-tile A/B shared layout (CUDA packs
  `sA_p[r*32+c]`, `sB_p[r*64+c]`) is re-chosen to feed rocWMMA's expected
  row/col-major fragment load (16-row tiles, ld = tile width). The
  `get_A8_offset` 128x32 packed global layout from precompute can stay (it is a
  pure addressing convention) as long as the staging copy into shared matches the
  new fragment loads.
- **Shared-memory budget.** CUDA uses 7 primes x 2048 B x {1,2} buffers ~28-58 KiB.
  gfx90a has 64 KiB LDS/CU; verify the re-tiled buffers fit (7 planes is the
  multiplier to watch). If 58 KiB double-buffered does not fit alongside rocWMMA
  fragment overhead, start single-buffered (the `SingleBuffer=true` grid path) and
  add double-buffering as a perf follow-up.

## Accuracy (the whole point: preserve the error-free transform)

The Ozaki scheme is exact by construction provided the int8 matmul produces the
EXACT integer dot product per prime (no overflow, no rounding) before CRT. The
AMD int32 MFMA accumulator is the same width and the same integer semantics as
NVIDIA's s32 m16n8k32 accumulator, so the error-free property is preserved IF:
- **No int32 accumulator overflow.** Per prime, operands are residues in
  [0,127) (int8), K up to 1<<(n1 or n2) <= 1<<15 for N<=30 but here K=dim<=128 per
  Hadamard factor (n1,n2 ~ N/2, dim=1<<n_i). Max partial = 127*127*K. For K=128
  that is ~2.06e6, well under int32 max (2.1e9); even K=16384 (~2.6e11) would
  overflow, but the Kronecker split keeps K = dim2/dim1 <= a few thousand at most
  for N<=30 -- CONFIRM the max K per execute_implicit_hadamard call against
  MAX_QUBITS=30 and clamp/guard. This bound is identical on CUDA and AMD, so #1390
  already relies on it; we inherit the same guarantee. Flag: document the K bound.
- **Splitting K=32 into 2x K=16 does not change the integer sum.** int32 add is
  associative for these magnitudes (no overflow), so 2x K=16 == 1x K=32 exactly.
- **No implicit fp path.** rocWMMA int8->int32 is a pure integer op (the
  "accumulate in 32-bit then convert" note applies to float fragments, not int);
  confirm via `llvm-objdump -d --mcpu=gfx90a | grep v_mfma_i32` that the emitted
  op is `v_mfma_i32_16x16x16i8` and there is no float coercion.

Validation of bit-accuracy:
1. **MFMA microkernel vs CPU int reference:** a standalone test multiplies a
   random int8 [16xK] x [Kx16] and checks the int32 result == a CPU integer
   triple loop, for K=16 and K=32(=2x16). Bit-exact required.
2. **execute_implicit_hadamard vs CPU RNS+CRT reference:** feed a known double X,
   compute C on GPU and compare to a CPU implementation of the SAME RNS split +
   implicit-Hadamard + CRT (recentre, *norm*inv). Expect agreement to f64 rounding
   of the final scale (the integer core is exact; only the final
   `(double)(cv%M)*norm*inv` introduces representable f64 rounding, identical
   formula on both arches).
3. **vs #1390 CUDA output (cross-arch):** run the CUDA path on an NVIDIA box (or
   capture reference vectors from CI/the PR author) for the same inputs; the
   integer C before the final f64 scale must be IDENTICAL (same RNS, same CRT).
   The post-scale double may differ only in the last ULP if the multiply order
   differs -- keep the exact `fv * norm_factor * inv` order to match.
4. **End-to-end via test_iqp_tc_path.py:** N in {8,12} must stay normalized
   (atol 1e-6) and the TC path must match the shared-memory FWT path; N=14
   Kronecker path finite + normalized. This is #1390's own gate; it must pass on
   gfx90a unchanged (it is arch-neutral Python).
Risk flag: the n8->n16 re-tiling and the wave64 lane mapping are where a
correctness bug would hide (wrong accumulator->output-element mapping), so test 1
and 2 are mandatory gates BEFORE wiring the full path.

## Build / integration

- **Compile:** add `iqp_tc.cu` + the new AMD kernel TU to BOTH the CUDA file list
  and `KERNEL_SOURCES` (build_hip). Two viable file layouts:
  (a) keep #1390's filenames (`ImplicitHadamardOzaki.cu`) and put the AMD kernel
      behind `#if defined(__HIP_PLATFORM_AMD__)` inside it (one file, two code
      paths) -- mirrors the amplitude.cu approach and keeps the CUDA diff identical
      to #1390; OR
  (b) add a sibling `ImplicitHadamardOzaki_hip.cu` compiled only by build_hip and
      exclude it from the nvcc list.
  Recommend (a) for a minimal, reviewable diff and to keep the CUDA path
  byte-identical to #1390. The hipcc include path already has `hip_compat/`
  (cuda_runtime.h, cuComplex.h, vector_types.h) and `src/`; ADD the rocWMMA
  include dir (ships in ROCm, e.g. `$ROCM_PATH/include`, already reachable via
  hipcc) and `#include <rocwmma/rocwmma.hpp>` under the HIP guard.
- **`--offload-arch`:** unchanged mechanism -- `QDP_HIP_ARCH_LIST` (default gfx90a
  when unset). rocWMMA is header-only; no extra link lib beyond amdhip64.
- **Host glue:** the Rust side is arch-neutral and ports verbatim from #1390 --
  `qdp-kernels/src/lib.rs` `extern "C" launch_iqp_encode_tc` + the
  `#[cfg(any(not(qdp_gpu_platform), qdp_no_cuda))]` stub (note: rebase #1390's
  `target_os="linux"` cfgs onto #1399's `qdp_gpu_platform` cfg -- see #1399's
  Windows delta), `iqp.rs encode_batch_tc`, `lib.rs encode_batch_tc`,
  `qdp-python engine.rs`. No HIP-specific Rust is needed: the kernel ABI is
  identical.
- **Gating:** CUDA path selected when not HIP; AMD path under
  `defined(__HIP_PLATFORM_AMD__)`. The `extern "C"` symbol `launch_iqp_encode_tc`
  is the only thing Rust sees, identical on both.

## Base / branch strategy

#1390 is unmerged and may change; #1399 is our open PR on `moat-port`. The
follow-up needs #1399's HIP infra + #1390's feature + the new AMD kernel.
Recommended approach:
- Create a NEW topic branch off `AMD-Ecosystem/mahout` `moat-port` (which already has
  #1399's HIP build, hip_compat shim, gpu_rt indirection, and the
  `qdp_gpu_platform` cfg). Name it e.g. `moat-ozaki-tc` (no MOAT jargon needed in
  the branch since it is fork-internal; the eventual upstream PR branch name is
  cosmetic).
- Cherry-pick / replay #1390's commits onto it, resolving conflicts: the `.cu`
  feature files are new (clean add); `build.rs`, `qdp-kernels/src/lib.rs`,
  `iqp.rs`, `qdp-core/src/lib.rs`, `engine.rs` need the #1390 hunks ADAPTED to
  #1399's state (the `DEFAULT_*_ARCHES` arch-list edit in #1390's build.rs is a
  CUDA-only cosmetic change -- keep or drop independently; the `target_os="linux"`
  cfgs in #1390 must become `qdp_gpu_platform` to match #1399's Windows delta).
- THEN add the AMD kernel rewrite + rocWMMA include + the file-list entry.
- Tracking risk: if #1390 evolves (it is OPEN), our branch must re-sync. Mitigate
  by keeping the #1390-derived commits as a clean, separable lower layer and the
  AMD kernel as the top commit, so a #1390 rebase only replays the lower layer.
  Pin the #1390 head SHA (e25ce590) in notes and diff against it on each resync.
- Upstream delivery: this is a SEPARATE follow-up PR to apache/mahout, opened only
  after #1390 merges (or coordinated with it) and after #1399 merges -- otherwise
  the AMD kernel references a feature not yet in main. Do not open the upstream PR
  until both dependencies land; until then it lives on the fork. (User gate
  applies before any upstream PR.)

## Phasing + validation (each phase gated on real gfx90a)

Phase 0 -- Mechanical port of the non-TC parts. Port iqp_tc.cu's FWT/phase/
  transpose/recombine + host dispatcher under the HIP build; stub
  `execute_implicit_hadamard` (e.g. fall back to the existing scalar/global FWT or
  return unsupported for large N). Gate: builds with hipcc on gfx90a; N<=12
  test_iqp_tc_path normalized (uses only the shared-mem FWT, no TC).

Phase 1 -- MFMA microkernel correctness. Standalone rocWMMA 16x16x16 int8 kernel;
  validate int32 result bit-exact vs CPU integer GEMM for K=16 and K=32(=2x16),
  random int8 in [0,127). Confirm `v_mfma_i32_16x16x16i8` emitted
  (llvm-objdump --mcpu=gfx90a). Gate: bit-exact on real GPU.

Phase 2 -- Implicit-Hadamard tiling + RNS single prime. Wire the residue-split A
  staging, implicit-Hadamard B sign generation, and the 64x64-tile/wave64 loop for
  ONE prime; check the int32 per-prime accumulator vs a CPU RNS reference (single
  prime). Gate: bit-exact single-prime tile on GPU.

Phase 3 -- Full Ozaki accurate FWT (7 primes + CRT). Add all 7 prime planes and
  CRT reconstruction + recentre + f64 scale; validate execute_implicit_hadamard
  output vs the CPU RNS+CRT reference and vs the shared-mem FWT path for N where
  both run. Gate: matches CPU reference to f64 ULP; TC path == FWT path (atol
  1e-6) for N in {8,12} via direct host calls.

Phase 4 -- Rust/Python integration. Replay #1390's `launch_iqp_encode_tc` glue,
  `encode_batch_tc` on engine + encoder + python, rebased onto `qdp_gpu_platform`.
  Gate: `cargo test -p qdp-core -p qdp-kernels --features hip` green (no
  regression of #1399's suites) AND test_iqp_tc_path.py passes (N=8,12 normalized,
  N=14 Kronecker finite/normalized) on gfx90a with the ROCm-torch env.

Phase 5 -- PR prep (only when #1390 + #1399 are landed/coordinated and user
  approves). AMD copyright + `Jeff Daily` author header on the
  new/substantially-extended files; ROCm build doc in house style; scrub MOAT
  jargon from the diff; squash; pr-ready across required platforms.

## Followers (RDNA gfx1100 / gfx1201)

WMMA int8 (16x16x16, int8->int32) exists on RDNA3 (gfx1100/1151) and RDNA4
(gfx1201) and rocWMMA targets them with the SAME 16x16x16 fragment API -- so a
rocWMMA-based kernel should build and run on RDNA with NO new fragment geometry.
The wave32 difference is the catch: RDNA WMMA is a 32-lane collective vs CDNA
MFMA's 64-lane. Because we route everything through rocWMMA (which encapsulates
the lane mapping) and gate strides on `warpSize`, the per-arch lane distribution
is handled by the library; the block-level tiling (256 threads -> wavefronts) must
read `warpSize` (64 on CDNA, 32 on RDNA) for the wavefront count. Follower plan:
validate on gfx1100/gfx1201 first (no re-port), delta only if a wave32 tiling
constant was hardcoded. NOTE distinct from the AutoDock-GPU fp32 trap: that was
fp32-input WMMA (no RDNA intrinsic); int8 WMMA DOES exist on RDNA, so this engine
is not subject to the fp32 silent-no-op hazard.

## Feasibility verdict

VIABLE on gfx90a now. There is no hardware blocker: gfx90a has native int8 matrix
cores (int8->int32 MFMA) and rocWMMA exposes them; the Ozaki integer/CRT math is
arch-independent; the non-TC parts ride #1399's proven HIP build unchanged.
Difficulty: MEDIUM-HIGH, concentrated in ONE kernel rewrite. The two real risks
are (1) the fragment-shape re-derivation -- m16n8k32 (K=32, n8, 32-lane warp) ->
16x16x16 (K=16x2, n16, 64-lane wavefront) requires re-deriving the accumulator
<-> output-element mapping and the wavefront tiling, where a silent indexing bug
is easy and only a bit-exact int test catches it; and (2) accuracy preservation --
must confirm no int32 overflow at the max K reachable for N<=30 and that the K-split
sum is exact (both true given the magnitude bounds, but must be asserted, not
assumed). rocWMMA removes the lowest-level lane/ldmatrix risk and gives a single
kernel for CDNA + RDNA. No CUTLASS/CuTe/wgmma is involved (it is hand PTX MMA), so
the "CUTLASS does not port" wall does not apply -- a clean MFMA reimplementation is
the right and sufficient move. Estimate: the kernel rewrite + 5 validation gates
is the bulk; the Rust/Python glue is near-free (replayed from #1390).

## Open questions
- Confirm the max K (= Hadamard factor dim) actually reachable per
  execute_implicit_hadamard call for MAX_QUBITS=30 and that 127*127*K stays < 2^31
  (int32). If a large-N config can exceed it, #1390's CUDA path has the same
  latent overflow -- raise upstream rather than diverge.
- rocWMMA version shipped with our ROCm 7.2.1 / TheRock 7.14: confirm int8
  16x16x16 fragments are present and the col_major matrix_b layout matches #1390's
  `.row.col` operand orientation (else transpose the B staging).
- Whether to keep #1390's persistent work-queue + double-buffer variant on AMD or
  ship single-buffer first (LDS budget at 7 primes on 64 KiB gfx90a LDS).
- Reference vectors: obtain #1390 CUDA output for cross-arch bit-compare (ask the
  PR author or run on an NVIDIA box) to gate Phase 3 against the real CUDA result,
  not only the CPU reference.
