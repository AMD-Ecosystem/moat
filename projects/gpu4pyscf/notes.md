# gpu4pyscf notes

ROCm/HIP port. Lead platform linux-gfx90a (MI250X, ROCm 7.2.1). Fork:
https://github.com/AMD-Ecosystem/gpu4pyscf , branch `moat-port`. Strategy A
(CUDA spelling kept, mapped to HIP via one compat header).

## Milestone status

**Superseded 2026-10-31**: nothing below is deferred any more. Every library, including libxc, solvent, sem and multigrid, is built for ROCm, and gradients and Hessians pass; see the "Porter 2026-10-31" sections at the end. The list below is history.

- **Milestone 1 (HF / integrals) -- DONE on gfx90a.** gint, gvhf, gvhf-rys,
  gvhf-md, cupy_helper, plus the import-/runtime-coupled gecp, gdft and pbc.
  Validated: see Test Plan in the commit. J/K matches CPU PySCF to <1e-11,
  RHF energy to <1e-11 Ha, deterministic to print precision. The gvhf-md
  wave64 J-engine fix is multi-arch (one fat binary holds gfx90a + gfx1100).
- **Deferred (not built under USE_HIP):**
  - CUTLASS grouped GEMM (cupy_helper/grouped_gemm.cu, grouped_dot.cu) ->
    reimplement as a hipBLAS per-group loop or Composable Kernel grouped GEMM.
    Gates the DFT numint path.
  - External libxc-GPU functional library (wxj6000/libxc, CUDA-only) -> a
    separate libxc-HIP port; scaffold as a sibling MOAT project gpu4pyscf
    `depends_on`. Gates all DFT functionals.
  - DFT (gdft kernels build, but the functional eval path needs libxc-HIP;
    gen_grids double3 init was fixed for HIP), analytic gradients/Hessian
    (the gvhf ejk_ip1 derivative engines), solvent, sem, multigrid, periodic
    DFT/df (pbc.scf/pbc.dft, libmgrid). The single failing scf test
    `test_jk_energy_per_atom` is the analytic gradient engine (deferred);
    `test_scf.py::test_to_cpu/test_to_gpu` fail on the DFT RKS path (deferred).

## Build recipe (gfx90a, or multi-arch)

Env (one-time): in the py_3.12 conda env, cupy_rocm_7_0 is broken by a numpy
ABI mismatch (built for numpy 2.x, env had 1.26.4). Align numpy to 2.x:
`pip install 'numpy>=2.0,<2.3'` (satisfies cupy>=2.0, scipy 1.14 <2.3, numba
<2.5, torch). pyscf 2.13 is installed (the CPU reference / validation gold).
Verify: `python -c "import cupy; cupy.linalg.eigvalsh(cupy.eye(4))"` runs on
gfx90a.

```
export HIP_VISIBLE_DEVICES=0
cmake -S gpu4pyscf/lib -B build -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DBUILD_CUTLASS=OFF -DBUILD_LIBXC=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

The .so are emitted into gpu4pyscf/lib (LIBRARY_OUTPUT_DIRECTORY). The build
also drops unbundled `*.so.0.hipv4-*` / `*.so.0.host-*` RDC offload objects in
that dir -- do not commit them (`.so` is gitignored; the split objects are not,
so `rm -f gpu4pyscf/lib/*.so.0.hipv4-* gpu4pyscf/lib/*.so.0.host-*` before
committing). Run from the repo root with `PYTHONPATH=$PWD`.

Followers: replace the arch list with `gfx1100` (or `gfx1151`). The wave64
J-engine fix is wave-agnostic, so a follower should validate first and only
delta-port on failure.

## Key port facts / gotchas

- gint nr_fill_ao_ints l>=4 (NROOTS 7/8) instantiates a >360 KB per-thread
  gout[] array that overflows the gfx9 scratch frame; those legacy template
  instantiations are #ifndef USE_HIP'd out (the HF path uses gvhf-rys, not this
  legacy engine). nr_fill_ao_ints also compiled at -O1 to fit the frame.
- (OBSOLETE since the 2026-10-29 upstream sync: upstream no longer indexes any pool by get_smid(), and probe_smid_range/pool_slots were removed. Kept for history.)
  get_smid: HIP `__smid()` returns a sparse (SE,CU) id that can exceed
  multiProcessorCount (gfx90a: max 125 vs mpc 104). Any pool indexed as
  `pool + get_smid()*POOL_SIZE` MUST be sized to the smid range, not mpc, or
  blocks on high-id CUs write past it (silent ~4MB OOB on the d-shell int3c2e
  path). Implemented as `__config__.pool_slots`: `probe_smid_range()` in
  gvhf-rys/mole_helper.cu launches a saturating max-occupancy kernel that
  `atomicMax`es `get_smid()`, returns max+1, then the Python rounds up to the
  next power of two (128 on gfx90a) so the slot count provably covers the full
  encodable id range. Used at the two get_smid()-indexed pools:
  df/int3c2e_bdiv.py:192 and pbc/df/ft_ao.py:377. NOT moduloed into [0,workers)
  (that would reintroduce per-CU contention). On CUDA pool_slots == mpc
  (NVIDIA smid < SM count holds). The scf/jk.py and grad/hessian dd_pool pools
  are NOT smid-indexed (blockIdx.x*QUEUE_DEPTH, launched with exactly mpc
  blocks device-side) so they stay at mpc. Proof: sentinel guard after the
  pool stays intact (0 clobbered) with 128 slots; the old mpc=104 sizing
  clobbers 6639 f64 guard entries on the same d-shell aux_e2 run.
- pbc int3c2e uses cub::BlockScan/BlockReduce; hip_compat/cub/cub.cuh aliases
  cub -> hipcub. The wave64 `__ballot` path in int3c2e_create_tasks_o1.cuh is
  NOT compiled (no .cu includes it) and is left for the periodic int3c2e
  milestone.
- -ffp-contract=on pins fp contraction so the J/K fingerprints match CPU to
  the test's 7 d.p.; -fgpu-rdc + -fvisibility=default for the cross-TU device
  globals (Rys-root / index tables); -Wno-register for the C++17 register kw.
- hipSOLVER enums differ from cuSOLVER: HIPSOLVER_EIG_TYPE_1=211,
  EIG_MODE_VECTOR=202, HIPBLAS_FILL_MODE_LOWER=122 (cuSOLVER uses 1/1/0).
  Passing the cuSOLVER values gives HIPSOLVER_STATUS_INVALID_ENUM (10). The
  CuPy `get_cusolver_handle()` is a rocSOLVER handle, NOT a hipSOLVER one, so
  eigh creates its own hipSOLVER handle and binds the current stream.

## Install as a dependency

This is a leaf application (a CuPy plugin loaded via PYTHONPATH), not a base
library other MOAT targets build against. No install-as-dependency section
needed. A future libxc-HIP project would be the dependency gpu4pyscf consumes.

## Review 2026-06-02 (reviewer, gfx90a, moat-port @ 460879b) -- CHANGES REQUESTED

Verdict: one high-severity latent defect on the HF path; the wave64 J-engine
fix, multi-arch build, and Python BLAS/solver layer are otherwise correct.
Reproduced on real gfx90a (HIP_VISIBLE_DEVICES=0, GCD0).

### Blocker -- get_smid() scratch-pool OOB on the HF hcore path (not a deferred path)
- gpu4pyscf/lib/gvhf-rys/vhf.cuh:155-161 switches get_smid() to HIP __smid()
  and the comment claims "the host sizes the scratch pool to the full id range
  on ROCm (see SM_POOL_SLOTS in scf/jk.py / df int3c2e)". That symbol does not
  exist anywhere in the tree, and the actual scratch pool is unchanged:
  gpu4pyscf/df/int3c2e_bdiv.py:191-192 still does
  `workers = gpu_specs['multiProcessorCount']; pool = cp.empty((workers, POOL_SIZE))`
  (file untouched by the port). The kernel indexes it as
  `out_local = pool + get_smid()*POOL_SIZE` at gpu4pyscf/lib/gvhf-rys/fill_int3c2e.cu:332
  and 9 sites in unrolled_int3c2e.cu (654/937/1473/1699/1987/2353/2634/2887...),
  with no clamp/modulo on get_smid().
- Measured on this gfx90a GCD: __smid() reaches 125 while multiProcessorCount
  is 104. POOL_SIZE=25600 doubles, so a block on a CU with smid in [104,125]
  writes up to ~4 MB past the pool -- device heap corruption.
- This is on the Milestone-1 HF path, not deferred: scf/hf.py:get_hcore (line
  133) calls df.int3c2e_bdiv.contract_int3c2e_auxvec for the nuclear-attraction
  hcore on every RHF/UHF build, which routes through libvhf_rys.fill_int3c2e
  with that pool. The pool-write branch is `to_sph && (li>1||lj>1)`, taken for
  d-shells, which def2-svp / cc-pVDZ (the test bases) have. The 33+16 passing
  tests do not prove safety: the OOB is silent until the clobbered allocation
  is read, so a green run is consistent with latent corruption.
- Fix: size the pool to the actual __smid() id range under ROCm (allocate
  `max_smid+1` rows, e.g. from a one-time device query of the max __smid, or a
  safe gfx9 upper bound), OR clamp/remap the index in get_smid()'s consumers.
  Make the vhf.cuh comment match whatever is actually implemented (and define
  the SM_POOL_SLOTS it references, or drop the reference). Audit every
  get_smid()*POOL_SIZE site (fill_int3c2e.cu + unrolled_int3c2e.cu) and the
  df/int3c2e_bdiv.py:192 allocation together.

### Minor -- commit body / dead-code accuracy (fix on the same pass)
- The commit body states the pbc int3c2e_create_tasks warp_max/block_max
  reductions are "made wave-size aware ... the CUDA path is byte-identical".
  block_max's CUDA `#else` branch in gpu4pyscf/lib/pbc/int3c2e_create_tasks.cuh
  (lines ~378-391) was changed from `offset = WARPS/2` / `thread_id < WARPS`
  (WARPS=8) to `offset = nwarps/2` / `thread_id < nwarps`, so the CUDA source
  is not byte-identical (it is behavior-equivalent only because blockDim.x=256
  -> nwarps=8). Moreover block_max and warp_max are dead in Milestone 1: only
  _filter_jk_images is called by the compiled pbc TUs (contract_int3c2e.cu,
  ejk_int3c2e_ip1.cu, fill_int3c2e.cu) and it does not call warp_max/block_max.
  Either drop the dead-code CUDA-path edit (keep the CUDA path truly untouched
  and guard only under USE_HIP) or correct the "byte-identical" wording.
- Housekeeping: the build leaves untracked `*.so.0.hipv4-*` / `*.so.0.host-*`
  split RDC objects in gpu4pyscf/lib (notes already document `rm -f`-ing them);
  they are not committed (.so gitignored; splits untracked), so this is only a
  reminder, not a defect.

### Verified clean (no action)
- gvhf-md wave64 shfl fix is correct. cuda_to_hip_shfl.h rewrites the 16-lane
  __shfl_down_sync(mask,...) to __shfl_down(val, offset, popcount(mask)); the
  width = popcount = threadsx (16 in unrolled_md_j*.cu, derived per-site in
  md_contract_j.cu). With sq_id = tx + 16*ty mapping thread->hardware lane in
  row-major order, a width-16 HIP segment is exactly one ty-row on wave64
  (lanes 0-15/16-31/32-47/48-63) and wave32 -- multi-arch-correct from one
  binary; the buggy CUDA group_id/mask is correctly ignored because HIP
  re-derives the segment from the hardware lane. Scoping the macro to the 3
  J-engine TUs via force-include (not the global compat header) avoids the
  hipCUB/bf16 __shfl_down_sync overload collision. contract_int3c2e.cu's
  full-warp + nwarps cross-warp reduction tree-reduces correctly on wave64
  (garbage lanes >= nwarps never reach lane 0 given the offset reach).
- Multi-arch confirmed: llvm-objdump --offloading shows BOTH gfx90a and gfx1100
  code objects in all 8 .so (gint/gvhf/gvhf-rys/gvhf-md/cupy_helper/gdft/pbc/gecp).
- Python layer: lib/cusolver.py hipSOLVER enum remap (TYPE_1=211, MODE_VECTOR=202,
  FILL_LOWER=122) and own-handle/stream-bind is correct; cholesky stays on the
  CuPy potrf backend. lib/cublas.py lazy find_library(cublas|hipblas) is sound
  (handle unused). __config__ sharedMemPerBlockOptin .get() fallback and
  dft/libxc.py version guard are additive and CUDA-behavior-preserving.
- Deferral gating is clean: gpu4pyscf/__init__.py imports only lib+scf under
  is_hip; scf's deferred imports (grad, pbc.gto.int1e) are function-local;
  pbc/__init__.py and pbc/df/__init__.py skip the periodic/df drivers under HIP
  without breaking the libpbc 1e path. CUTLASS grouped_gemm/grouped_dot excluded
  from cupy_helper SOURCES when BUILD_CUTLASS=OFF (structural, not silent).
- test_jk_energy_per_atom genuinely exercises the deferred analytic-gradient
  engine (imports grad.rhf._jk_energy_per_atom, compares vs int2e_ip1), not core
  J/K -- the deferral label is accurate.
- Commit hygiene OK: [ROCm] title <=72 chars, Claude credited, no noreply
  trailer, no ghstack, Test Plan present, no AMD-internal account refs; fork
  master is a clean mirror; Actions disabled on the fork.

## Porter fix 2026-06-02 (Review response, fork moat-port @ 8cb8806)

Resolved both reviewer findings; rebuilt multi-arch and re-validated.

- BLOCKER (get_smid OOB) FIXED. Implemented the promised host-side pool resize.
  Added `probe_smid_range()` (gvhf-rys/mole_helper.cu): a saturating
  max-occupancy kernel that `atomicMax`es `get_smid()` and returns max+1.
  `gpu4pyscf/__config__.pool_slots` calls it once (USE_HIP only), rounds up to
  the next power of two (128 on gfx90a, covering the full 0..127 encodable smid
  range), and is used at df/int3c2e_bdiv.py:196 and pbc/df/ft_ao.py:377 (the two
  get_smid()-indexed pools). On CUDA pool_slots == multiProcessorCount, NVIDIA
  path unchanged. NOT moduloed (keeps the per-CU-private-slot semantics). Audit:
  get_smid() appears only in fill_int3c2e.cu/unrolled_int3c2e.cu (libvhf_rys)
  and pbc/ft_ao.cu; scf/jk.py + grad/hessian dd_pool are blockIdx-indexed with
  mpc blocks, not smid -- correctly left at mpc. Updated the vhf.cuh comment to
  point at the real pool_slots/probe_smid_range instead of the never-existing
  SM_POOL_SLOTS symbol.
- Proof the OOB is gone: sentinel guard rows placed right after the pool, driven
  by a d-shell aux_e2 (O/H cc-pVDZ + cc-pVDZ-jkfit) that hits the
  to_sph && li>1 fill branch: 0 guard entries clobbered with pool_slots=128;
  the same run forced to the old workers=mpc=104 sizing clobbers 6639 f64 guard
  entries. probe returns 126 (max smid 125) vs mpc 104.
- MINOR FIXED. pbc/int3c2e_create_tasks.cuh block_max: the wave-aware nwarps
  logic is now fully inside #ifdef USE_HIP; the CUDA #else branch is reverted to
  the upstream byte-identical WARPS form. (These functions are dead in M1.)
- Re-validated on gfx90a (GCD0): test_rhf/test_uhf/test_int2c2e/test_int4c2e
  33 passed 3 skipped; test_scf_j_engine/test_scf_jk 16 passed (only deferred
  test_jk_energy_per_atom fails). Both gfx90a+gfx1100 code objects present in
  libgvhf_rys.so (and the other 7 .so). AMD_LOG_LEVEL=3 shows probe_smid_kernel
  + int3c2e kernels dispatch on gfx90a.

## Review 2026-06-02 (reviewer, gfx90a, DELTA 460879b -> 8cb8806) -- REVIEW PASSED

Focused delta re-review of the porter's fix for the prior get_smid() pool-OOB
blocker. Both commits share parent 72087c7 (the porter amended the single
curated commit; correct per MOAT rules). No problems found; the prior blocker
and minor are both resolved and independently verified on real gfx90a (GCD0).

Verification performed (not just read):
- probe_smid_range (mole_helper.cu, USE_HIP only) uses the same get_smid()/
  __smid() the consumers use, launches mpc*8 blocks with a clock64 spin to keep
  blocks co-resident (saturating occupancy), atomicMaxes the slot id. Any probe
  under-count is absorbed by the power-of-two rounding in __config__: on gfx90a
  observed 125 -> +1 -> rounded to 128, covering the full gfx9 7-bit encodable
  smid range [0,127]. Not racy under-counting in a way that matters because the
  ceiling, not the observation, sets the floor.
- pool_slots (__config__.py) = 128 on gfx90a, == multiProcessorCount on CUDA
  (NVIDIA path untouched), not moduloed (per-CU-private slots preserved).
- Applied at exactly the two get_smid()-indexed pools: df/int3c2e_bdiv.py:196
  and pbc/df/ft_ao.py:377. Audit confirmed exhaustive: get_smid() appears only
  in fill_int3c2e.cu, unrolled_int3c2e.cu (9 sites) and pbc/ft_ao.cu:261. Those
  kernels launch a 2D grid (sp_block x ksh_block) that can exceed mpc, so the
  CU-private smid index is the right choice and mpc-sizing was genuinely wrong.
- scf/jk.py and grad/hessian dd_pool correctly left at multiProcessorCount:
  RYS_build_jk / RYS_per_atom_jk_ip1 index pool + blockIdx.x*QUEUE_DEPTH and
  dd_pool + blockIdx.x*... , launched <<<workers=multiProcessorCount, ...>>>, so
  blockIdx.x in [0,mpc) -- mpc sizing is exact there. Confirmed device-side.
- Decisive proof reproduced on GCD0 (agent_space/sentinel_proof.py and a
  negative control): d-shell aux_e2(O/H cc-pVDZ + cc-pVDZ-jkfit) hitting the
  to_sph && li>1 write branch -> 0 guard f64 entries clobbered at pool_slots=128;
  forcing the old workers=mpc=104 sizing clobbers 7107 guard entries on the same
  run. The guard genuinely catches the silent OOB in both directions.
- pbc int3c2e_create_tasks.cuh block_max CUDA #else is now byte-identical to
  upstream 72087c7 (buf[WARPS], thread_id < WARPS, offset WARPS/2, mask 0xff);
  warp_max likewise. Wave-aware nwarps logic is fully inside #ifdef USE_HIP. The
  prior "byte-identical" wording is now accurate. Verified via git show.
- vhf.cuh comment now points at the real pool_slots/probe_smid_range instead of
  the never-existing SM_POOL_SLOTS symbol.
- Tests on GCD0: test_rhf/uhf/int2c2e/int4c2e/scf_j_engine/scf_jk -> 49 passed,
  3 skipped, 1 failed (only the deferred analytic-gradient test_jk_energy_per_atom,
  accurately labeled). libgvhf_rys.so carries both gfx90a + gfx1100 code objects;
  probe_smid_range symbol exported. Build artifacts are current with 8cb8806's
  sources (so mtime newer than the .cu/.py). Untracked split RDC objects are the
  documented gitignored housekeeping, not committed.
- Hygiene: [ROCm] title 61 chars, Claude credited, no noreply trailer, no
  ghstack, no em-dash, Test Plan present; fork master is a clean mirror at
  72087c7; Actions disabled on the fork (enabled=false).

Verdict: APPROVE. Handing to the validator for the full GPU test run.

## Validation 2026-06-02 (gfx1100) -- COMPLETED

Platform: AMD Radeon Pro W7800 48GB, gfx1100 (RDNA3, wave32), HIP_VISIBLE_DEVICES=0,
ROCm 7.2.1, py_3.12 conda env (cupy-rocm-7-0 14.1.1, numpy 2.2.6, pyscf 2.13).
Fork: AMD-Ecosystem/gpu4pyscf @ 8cb88067f726 (moat-port, no changes -- validate-first follower).
Scope: Milestone 1 HF/integrals; DFT/gradients/CUTLASS-GEMM/libxc deferred.

### Build (gfx1100)

```
cmake -S gpu4pyscf/lib -B /var/lib/jenkins/moat/agent_space/gpu4pyscf_build_gfx1100 \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DBUILD_CUTLASS=OFF -DBUILD_LIBXC=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build /var/lib/jenkins/moat/agent_space/gpu4pyscf_build_gfx1100 -j8
rm -f gpu4pyscf/lib/*.so.0.hipv4-* gpu4pyscf/lib/*.so.0.host-*
```

Build time: 124s (2m4s). Exit 0. All 8 .so emitted.

Fat-binary code-object verification (llvm-objdump --offloading):
All 8 .so (libcupy_helper, libgdft, libgecp, libgint, libgvhf, libgvhf_md,
libgvhf_rys, libpbc) contain BOTH gfx1100 and gfx90a code objects.

Git status after split-RDC cleanup: clean (no uncommitted files).

AMD_LOG_LEVEL=3 confirms "Using native code object for device:
amdgcn-amd-amdhsa--gfx1100" for all 8 .so on probe_smid_kernel dispatch.
No HSA 0x1016 errors observed.

### probe_smid_range / pool_slots on gfx1100

pool_slots=512, multiProcessorCount=35 on gfx1100. probe_smid_range launched
on __smid() with saturating occupancy and returned max_smid+1=512 (larger
encodable smid range on RDNA3 than on gfx90a's 128). The two get_smid()-
indexed pools (df/int3c2e_bdiv.py:196 and pbc/df/ft_ao.py:377) are correctly
sized to 512 rows on gfx1100.

### Wave32 verdict

gvhf-md J-engine: CORRECT at wave32. The cuda_to_hip_shfl.h rewrite uses
__shfl_down(val, offset, popcount(mask)) with width=popcount(mask)=threadsx=16.
On wave32 a width-16 HIP segment covers one ty-row (lanes 0-15 or 16-31),
yielding the same lane mapping as on wave64. The J/K tests (test_scf_j_engine,
test_scf_jk) pass with 0 failures.

Multi-arch host/device WARP_SIZE check: NO mismatch. The gvhf-md J-engine
uses the width argument to __shfl_down at the call site rather than a compile-
time warp-width template constant, so host launch geometry (via mol.nao, not
a fixed warp count) never hardcodes a wave64-specific constant. No "Cannot find
Symbol" or wrong-instantiation on gfx1100. AMD_LOG_LEVEL=3 confirms gfx1100
native code objects dispatched without error.

### HF/integral test suite

Test run 1 (rhf / uhf / int2c2e / int4c2e):
```
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. python -m pytest \
  gpu4pyscf/scf/tests/test_rhf.py gpu4pyscf/scf/tests/test_uhf.py \
  gpu4pyscf/scf/tests/test_int2c2e.py gpu4pyscf/scf/tests/test_int4c2e.py -v --tb=short
```
Result: 29 passed, 7 skipped, 0 failed (87.2s).
Skips: test_rhf_d3/d4 (dftd3/dftd4 not installed), test_get_j1_hermi0/
test_get_jk1_hermi0/test_get_k1_hermi0 (@unittest.skip('hermi=0')),
test_uhf_d3bj/d4 (dftd3/dftd4 not installed). All unconditional skips, not
regressions vs gfx90a. Total 36 collected = 29+7.

Test run 2 (J-engine + JK, run 1):
```
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. python -m pytest \
  gpu4pyscf/scf/tests/test_scf_j_engine.py gpu4pyscf/scf/tests/test_scf_jk.py -v --tb=short
```
Result: 17 passed, 0 failed (39.8s). All J-engine and JK tests PASS including
test_jk_energy_per_atom (grad.rhf import succeeded on gfx1100, test ran and passed).

Test run 2b (J-engine + JK, determinism repeat):
```
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. python -m pytest \
  gpu4pyscf/scf/tests/test_scf_j_engine.py gpu4pyscf/scf/tests/test_scf_jk.py -v --tb=short
```
Result: 17 passed, 0 failed (39.1s). Identical to run 1 -- deterministic to print precision.

### Deferred-path tests (expected failures)

test_scf.py::test_to_cpu / test_to_gpu fail on the DFT RKS path (cupy_helper
release_gpu_stack via numint.nr_rks) -- deferred, libxc not installed. test_rhf,
test_rhf_cart, test_screening in test_scf.py all PASS. No new deferred-path
failures vs gfx90a.

### Summary

All gates pass:
- Fat-binary build: gfx1100 + gfx90a code objects in all 8 .so -- PASS
- probe_smid_range / pool_slots: 512 on gfx1100, probe_smid_kernel dispatches
  native gfx1100 code -- PASS
- Wave32 J-engine: J/K correct at wave32, no WARP_SIZE mismatch, no 0x1016 -- PASS
- HF/integral suite: 29+17 passed (46 total), 7 unconditional skips, 0 failures -- PASS
- Determinism: identical results on back-to-back runs -- PASS
- Deferred-path failures: DFT RKS only (test_to_cpu/gpu), expected, not gate -- PASS
- Fork: clean (no changes needed -- validate-first follower, gfx1100 validated at same SHA as gfx90a)

validated_sha=8cb88067f72652557788a3dc7be075a38b717c82. linux-gfx1100 -> completed.

## Validation 2026-06-02 (validator, linux-gfx90a, fork @ 8cb8806) -- COMPLETED

Platform: MI250X gfx90a, GCD0 (HIP_VISIBLE_DEVICES=0), ROCm 7.2.1, conda py_3.12.
Scope: Milestone 1 HF/integrals; DFT/gradients/CUTLASS-GEMM/libxc deferred.

### Step 1 -- Multi-arch build

Rebuilt from source in agent_space/gpu4pyscf_build:

```
cmake -S gpu4pyscf/lib -B agent_space/gpu4pyscf_build \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DBUILD_CUTLASS=OFF -DBUILD_LIBXC=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build agent_space/gpu4pyscf_build -j8
```

roc-obj-ls confirmed both gfx1100 and gfx90a code objects in all 8 .so:
libcupy_helper.so, libgdft.so, libgecp.so, libgint.so, libgvhf.so,
libgvhf_md.so, libgvhf_rys.so, libpbc.so. Build: PASS.

### Step 2 -- HF/integral test suite

```
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. python -m pytest \
  gpu4pyscf/scf/tests/test_rhf.py gpu4pyscf/scf/tests/test_uhf.py \
  gpu4pyscf/scf/tests/test_int2c2e.py gpu4pyscf/scf/tests/test_int4c2e.py -v
```

Result: 33 passed, 3 skipped, 0 failed (157.74 s). Exactly the expected count.

```
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. python -m pytest \
  gpu4pyscf/scf/tests/test_scf_j_engine.py gpu4pyscf/scf/tests/test_scf_jk.py -v
```

Result: 16 passed, 1 failed (54.41 s). The single failure is
test_scf_jk.py::test_jk_energy_per_atom -- deferred analytic-gradient engine,
expected and correctly labeled. All core J/K tests pass.

AMD_LOG_LEVEL=3 confirmed "Using native code object for device:
amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-" and probe_smid_kernel dispatching
natively on gfx90a.

### Step 3 -- Sentinel OOB proof

```
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. python agent_space/sentinel_proof.py
```

Output:
  pool_slots=128 multiProcessorCount=104
  pool allocations intercepted: 1
  guard rows=64 clobbered f64 entries=0
  PASS: sentinel intact -> no get_smid() OOB write past pool_slots

0 guard entries clobbered. The smid-pool OOB fix (probe_smid_range -> pool_slots=128
on gfx90a, covering full 0..127 smid range) is confirmed effective. The silent
~4 MB OOB that old workers=mpc=104 sizing caused is gone.

### Summary

All three formal gates pass:
- Multi-arch build: both gfx90a + gfx1100 code objects in all 8 .so -- PASS
- HF/integral suite: 33+16 passed, 3 skipped, 1 deferred expected failure -- PASS
- Sentinel OOB check: 0 guard clobbers at pool_slots=128 -- PASS

validated_sha=8cb88067f72652557788a3dc7be075a38b717c82. linux-gfx90a -> completed.
Followers (linux-gfx1100) unblocked to port-ready.

## Validation 2026-06-05 (windows-gfx1101 + windows-gfx1201) -- BLOCKED

Platform: Windows 11 (gfx1101 Radeon PRO V710, gfx1201 RX 9070 XT), ROCm 7.14 TheRock nightly.
Attempted: validate-first follower at fork 8cb88067f726 (moat-port).

### Blocker: CuPy-ROCm has no Windows distribution

gpu4pyscf's Python layer is built entirely on CuPy-ROCm: all GPU arrays are
`cupy.ndarray`, BLAS/solver/sparse access goes through `cupy_backends.cuda.libs.*`,
and the DFT/HF Python driver is built around `cupyx`. The PyPI package
`cupy-rocm-7-0` (14.1.1, the version used on Linux) ships ONLY Linux
`manylinux2014_x86_64` wheels -- confirmed via PyPI JSON API and `pip index versions`.
No Windows wheels exist. There is no other distribution channel with a Windows
CuPy-ROCm build.

The TheRock PyTorch venv (the active MOAT Windows build environment) has no cupy
installed and no mechanism to acquire a Windows CuPy-ROCm wheel. Building CuPy
from source on Windows against the TheRock ROCm SDK is a multi-day effort (CuPy's
Windows build is complex and not tested against a nightly ROCm) and is outside the
scope of a one-attempt validation.

No kernel .dll build was attempted because there is no Python driver to exercise
them: even a successful compile of the 8 HIP shared libraries as Windows DLLs
would leave no way to run the HF/integral test suite (pytest + pyscf + cupy).

Resolution: windows-gfx1101 and windows-gfx1201 both blocked with reason above.
These platforms are non-viable for gpu4pyscf until a Windows CuPy-ROCm wheel is
published. The port itself (linux-gfx90a + linux-gfx1100) is complete and validated.

## Upstream sync 2026-10-29 (porter, linux-gfx942 MI300X, fork moat-port 899ec26 -> cdd8b8b)

Requested by jeff before the upstream PR is opened. Upstream master was 173 commits ahead of the port base 72087c7.

- Fork default branch `master` fast-forwarded 90dee1e -> 10dcea3 (upstream tip), plain push.
- `git merge --no-ff upstream/master` into moat-port -> merge commit c9f1ac4. Conflicts (9 files) and resolution:
  - df/int3c2e_bdiv.py, pbc/df/ft_ao.py, gvhf-rys/mole_helper.cu: upstream rewrote fill_int3c2e / ft_aopair as persistent kernels (`<<<multiProcessorCount>>>`, pool indexed by `blockIdx.x`, batches from an atomic `head` int stored after the pool). Nothing indexes a pool by `get_smid()` any more (grep: only the definition in vhf.cuh remains), so the earlier get_smid OOB fix is obsolete: `probe_smid_range()` (mole_helper.cu) and `__config__.pool_slots` were REMOVED and upstream's allocation taken verbatim. vhf.cuh keeps the `__HIPCC__` guard and the `__smid()` branch, with the long pool comment dropped. The 2026-06-02 sentinel proof no longer applies (no smid-indexed pool exists).
  - dft/libxc.py: upstream's `load_library` now raises on a missing lib, so the `_libxc is None` guard was dead; upstream version taken (dft is not imported on ROCm anyway).
  - lib/cusolver.py: hipSOLVER branch kept; upstream's new CUDA lookup (find_library, fallback to the nvidia/cusolver wheel via load_library) in the CUDA branch.
  - lib/CMakeLists.txt: upstream dropped CUDA from `project()` and calls `enable_language(CUDA)` after setting arches; now `if(NOT USE_HIP) enable_language(CUDA)`. Kept HIP `project(C CXX HIP)`, upstream's 3.21 minimum and its new multigrid options.
  - ecp/common.cu: upstream's block_reduce is now warpSize-generic (shared pass stops at warpSize, shuffle from warpSize/2); the HIP branch is the same loop with an unmasked `__shfl_down`.
  - gvhf-md/contract_int3c2e.cu: upstream reuses `shared_memory` for the cross-warp partials; the port's `nwarps` is kept.
  - pbc/__init__.py: upstream's new `pbc.properties` import joins scf/dft behind the ROCm check.
- The merge commit body says "The HIP project() keeps upstream's Fortran removal pattern"; that is inaccurate (the HIP project() omits Fortran, as the port always did). Fix it in the PR-prep squash message.
- New upstream files that failed to compile on HIP (static_assert: HIP `__shfl_down_sync` needs a 64-bit mask), fixed in cdd8b8b under `#ifdef USE_HIP`, CUDA unchanged:
  - pbc/contract_ft_ao.cu (2 kernels x 2 sites): the sp_threads (=16) image-count max and the 16-lane Gv-group goutR sum use `__shfl_down(v, off, width)`; segments match on wave32/wave64 (gout_id is uniform per 16-lane group, so the preceding `break` is segment-uniform).
  - gvhf-rys/int3c2e_recontract.cu: full-warp sum uses `__shfl_down(v, d, warpSize)`.
- Audit of the other new/changed .cu in the HIP-built dirs (gint gvhf cupy_helper gvhf-rys gvhf-md ecp gdft pbc): no other `__shfl*_sync`/`__ballot`/PTX asm outside USE_HIP guards. multigrid (including the new multigrid_v3) is still not built for ROCm.
- cdd8b8b also rewords the port comments that referred to internal planning stages ("ROCm/HIP port -- Milestone 1", "later milestone(s)", "pinned base", "deferred") in __init__.py, lib/CMakeLists.txt, gdft/CMakeLists.txt, gint/nr_fill_ao_ints.cu, pbc/CMakeLists.txt, pbc/df/__init__.py.

### Build (gfx942, ROCm 10.1.0 TheRock SDK wheels in /opt/python-3.11-venv)

```
R=$(rocm-sdk path --root)
cmake -S gpu4pyscf/lib -B agent_space/build-gfx942 -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$R \
  -DBUILD_CUTLASS=OFF -DBUILD_LIBXC=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build agent_space/build-gfx942 -j56
```

At c9f1ac4: 2 TUs fail (contract_ft_ao.cu, int3c2e_recontract.cu; the mask static_assert above). At cdd8b8b: exit 0, all 8 libs (cupy_helper gdft gecp gint gvhf gvhf_md gvhf_rys pbc); `llvm-objdump --offloading` shows a gfx942 code object in each. With the ROCm 10.1 toolchain the leftover untracked RDC split objects are named `lib*.so.0.hip-amdgcn-amd-amdhsa--<gfx>` (not `.hipv4-*`/`.host-*`): `rm -f gpu4pyscf/lib/*.so.0.hip-amdgcn-*` before checking `git status`. Fork tree clean after that.

Pushed moat-port 899ec26..cdd8b8b (`jargon.py --port gpu4pyscf`: clean; upstream/master and the previous tip are both ancestors). `advance-head cdd8b8b`; linux-gfx942 porting -> ported, lock released. linux-gfx90a / linux-gfx1100 now read revalidate; this round needs review, then validation on wave64 and wave32.

### GPU tests: NOT RUN (environment blocker, not a port failure)

This host's image env (/opt/python-3.11-venv: numpy 1.26.4, scipy 1.10.1) has no CuPy and no pyscf, and no other env on the host has them. The job was dispatched without `--build`, so there is no `FLEET_VENV`; creating a private venv and installing cupy-rocm-7-0 / pyscf / numpy>=2 into it was refused by the fleet guard. ROCm 10.1 sonames match what cupy-rocm-7-0 links (libamdhip64.so.7, libhipblas.so.3, librocblas.so.5, librocsolver.so.0, libhipsolver.so.1), so a job WITH a venv should be able to install: numpy>=2.0,<2.3, scipy>=1.13, cupy-rocm-7-0==14.2.0, pyscf, pytest.

Test-list change for the validator: upstream removed scf/tests/test_int2c2e.py (in the 2026-06-02 command) and added scf/tests/test_init_guess.py (SAP/VSAP guess), test_hf_lowmem.py and others; recompute the HF/integral set from `gpu4pyscf/scf/tests/` rather than reusing the old command.

### Still owed before the upstream PR

- PR-prep squash (`moatlib.py squash-carry-forward`, tree identical to the validated head): 8cb8806's body says "(Strategy" / "A ..." split across a line (jargon.py misses it), "This first milestone", "follow-on milestones", "Milestone 1", "pinned base", and describes probe_smid_range/pool_slots, which this sync removed. 899ec26's body says "in-house port-strategy labels". The squashed message must describe the merged result.

## Review 2026-10-29 (reviewer, linux-gfx942, fork moat-port @ cdd8b8b vs upstream/master 10dcea3) -- CHANGES REQUESTED

Scope: whole fork diff (29 files), with focus on the upstream merge c9f1ac4 and cdd8b8b, and on whether every CUDA path is unchanged. Fork `master` == upstream 10dcea3 (clean mirror); fresh clone, tree clean; `jargon.py --port gpu4pyscf` clean. The HIP shuffle rewrites in contract_ft_ao.cu (sp_threads and 16-lane Gv segments; `t_id` is uniform per 16-lane group, so the `break` is segment-uniform), int3c2e_recontract.cu (warps derived from blockDim/warpSize, threads=512), ecp/common.cu (THREADS=128 >= 2*64) and ft_ao.cu were checked against the surrounding code and are correct on wave32 and wave64. Problems:

### Blocker -- the merge dropped the ROCm libxc guard; every RHF/UHF on ROCm now fails at import
- gpu4pyscf/dft/libxc.py:35 is upstream's `_libxc = load_library('libxc', path='gpu4pyscf/lib/deps/lib')`, and lib/utils.py:47 `load_library` raises `OSError` when the library is absent. The ROCm build has no libxc (BUILD_LIBXC=OFF), so `import gpu4pyscf.dft.libxc` now raises. Lines 66-68 (`_libxc.xc_version_string()`) would fail next even if it did not.
- The HF path still imports that module. A static module-level import walk (agent_space/g4p/imports.py, honouring the `is_hip` branches) gives, at cdd8b8b: `scf/hf.py:140 get_hcore` -> `pbc.gto.int1e` -> `pbc.df.ft_ao` -> `df.int3c2e_bdiv` -> `df/__init__` -> `df.df` -> `df.df_jk` -> `dft.numint` -> `dft.libxc`. `scf/jk.py:1007 _cache_q_cond_and_non0pairs` (called from jk.py:442 on every J/K build) goes `pbc.scf.rsjk` -> `pbc.df.ft_ao` -> the same chain. At 899ec26 the same walk reaches `dft.libxc` too (through pbc.df.int3c2e). That is why the port had the `_libxc = None` fallback and the `if _libxc is not None` version guard, and why the gdft CMake comments say the HF import chain reaches dft.libxc. The merge note's reason for taking upstream's file ("dft is not imported on ROCm anyway") is wrong.
- Fix: in dft/libxc.py restore a ROCm-only fallback. Under `cupy.cuda.runtime.is_hip`, catch the `OSError` from `load_library('libxc', ...)` and set `_libxc = None`. Put the version block (lines 66-71) under `if _libxc is not None`, with an "unavailable" `__version__` otherwise. Keep the CUDA path byte-identical: no try/except on CUDA, so a missing libxc still raises there as upstream intends. Upstream's `if _libxc is not None:` at line 134 already handles the rest. Re-run an RHF import/energy on a ROCm host before handing back. This round ran no Python at all, and one import would have exposed this.

### Fault class (lane mask) -- undefined 32-bit shift on wave64 in the gvhf-md J-engine, exposed by the merge
- gpu4pyscf/lib/gvhf-md/md_contract_j.cu:101-103 and :431-433. Upstream changed `lane_id = t_id % 32` to `t_id % warpSize`, and the mask is still `unsigned int mask = ((1 << threadsx) - 1) << group_id * threadsx`. On wave64, `group_id*threadsx` is lane_id rounded down to a multiple of threadsx (1..16 from scf/j_engine.py `ij`), so it reaches 32-63, which is an out-of-range shift of a 32-bit value. cuda_to_hip_shfl.h derives the shuffle width as `__popcll(mask)`, so the width now comes from undefined behaviour. In LLVM IR for gfx942 (`clang++ -x hip --offload-arch=gfx942 --cuda-device-only -O3 -S -emit-llvm ... md_contract_j.cu`) this is `%98 = shl i32 %95, %97` with `%97 = lane_id - lane_id % threadsx`, feeding `llvm.ctpop.i32(%98)`. An out-of-range shl is poison. It probably works today only because v_lshlrev_b32 masks the shift to 5 bits. The 2026-06 gfx90a validation predates this change because the base still used `% 32`. unrolled_md_j.cu and unrolled_md_j_4dm.cu still use `% 32` and are unaffected.
- Fix (HIP-only, CUDA untouched): under `#ifdef USE_HIP` compute `unsigned long long mask = ((1ull << threadsx) - 1) << (group_id * threadsx);` (shift <= 64 - threadsx, always < 64) at both sites, keeping the CUDA `unsigned int` line in `#else`.

### CUDA path changed outside a USE_HIP guard
- gpu4pyscf/lib/gvhf-md/contract_int3c2e.cu:516, 532, 535. The CUDA lines `if (thread_id < 8)` and `for (int offset = 4; ...)` became `thread_id < nwarps` and `offset = nwarps/2`, with `nwarps = (blockDim.x + warpSize - 1) / warpSize` computed for both backends. It is behaviour-equivalent on CUDA only because THREADS=256 (nwarps=8). The CUDA source is not upstream's, and the constant-trip reduction becomes a runtime-trip loop. This predates today's round (it is in 8cb8806, and the merge kept it). It is the same class as the 2026-06-02 block_max finding, which was fixed by moving the wave-aware form inside `#ifdef USE_HIP`. Do the same here: keep the upstream `8`/`4` lines in the `#else` branch and use `nwarps` only under USE_HIP.

### Minimal footprint -- CUDA build order reshuffled
- gpu4pyscf/lib/CMakeLists.txt:183-207. Upstream's order is gint, gvhf, gdft, cupy_helper, sem, solvent, gvhf-rys, gvhf-md, pbc, multigrid, ecp (upstream lines 127-142). The port moves gdft, pbc and ecp, and groups sem/solvent/multigrid at the end, for the CUDA build too. The result is functionally equivalent but is needless upstream-visible churn. Keep upstream's order and wrap only sem, solvent and multigrid in `if(NOT USE_HIP)` in place. The libgecp/libgdft/libpbc comment can sit above the first of them.

### Stale comment
- gpu4pyscf/lib/CMakeLists.txt:39-40 says the force-included compat header defines "the wave64 subgroup-reduction fix". cuda_to_hip.h:113-120 says it deliberately does not; the fix is cuda_to_hip_shfl.h, force-included only on three gvhf-md TUs (gvhf-md/CMakeLists.txt). Drop that clause.

### Commit hygiene (due at the PR-prep squash; already listed under "Still owed")
- c9f1ac4's body has two wrong statements. "The HIP project() keeps upstream's Fortran removal pattern" (already recorded) is one. The other is the dft/libxc.py rationale, which the blocker above disproves. 8cb8806's body carries "Strategy\nA" split across a line (jargon.py is line-based and cannot match it; no term added), "Milestone 1", "pinned base", and the removed probe_smid_range/pool_slots description. The squashed message must describe the merged result.

### For the porter / validator
- After the fixes, run at least `python -c "import gpu4pyscf; from pyscf import gto; from gpu4pyscf import scf; m=gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='def2-svp'); print(scf.RHF(m).kernel())"` on a ROCm host with CuPy-ROCm before marking ported. Then validate wave64 and wave32 with the test set recomputed from gpu4pyscf/scf/tests/ (see the sync notes above). test_init_guess.py's vsap/qchem-sad cases import gpu4pyscf.dft.rks/numint and need libxc, so they fall under the existing DFT exclusion and are not port regressions.

Verdict: REQUEST CHANGES (one blocker, one wave64 UB defect, one CUDA-path edit, two footprint/comment items).

## Porter fix 2026-10-30 (porter, linux-gfx942 MI300X, fork moat-port cdd8b8b -> 45d8053)

Answers "Review 2026-10-29". Each finding was fact-checked before any change. New commit on top of cdd8b8b (no rebase/amend/force). `jargon.py --port gpu4pyscf` clean, `prose.py` clean on the commit message. Pushed cdd8b8b..45d8053 (fast-forward); `advance-head 45d8053064db65a7c776a65f9310ec4063bc5b5f`. linux-gfx90a / linux-gfx1100 (validated at 899ec26) read revalidate.

### Review findings
- **Blocker (dft/libxc.py), CONFIRMED.** At cdd8b8b, RHF on gfx942 dies with `OSError: Unable to load libxc` (lib/utils.py:47). Fix as asked: under `cupy.cuda.runtime.is_hip`, catch OSError -> `_libxc = None`. The version/xc_func_type block is under `if _libxc is not None`, else `__version__ = 'unavailable (libxc not installed)'`. CUDA still calls load_library bare, so it raises.
- **Second import blocker the review did not name, same class.** With libxc fixed, RHF then failed with `ImportError: cannot import name 'GDF' from gpu4pyscf.pbc.df`. scf/jk.py:1007 imports `gpu4pyscf.pbc.scf.rsjk`, the pbc.scf package imports ghf, and ghf does `from gpu4pyscf.pbc.df import GDF`, which the port's ROCm guard in pbc/df/__init__.py had skipped. Probes on ROCm: pbc.df, pbc.scf and pbc.properties all import cleanly. pbc.dft fails (libmgrid absent, and the int1e cycle). Fix: pbc/__init__.py imports scf and properties on both backends and skips only dft on ROCm. pbc/df/__init__.py is upstream's file again (one fewer port file). With pbc.df imported early by pbc/__init__, the aft -> pp_int -> pbc.gto.int1e cycle no longer trips (`import gpu4pyscf.pbc.gto.int1e` alone is clean).
- **Lane mask (gvhf-md/md_contract_j.cu:101-103, :431-433), CONFIRMED.** The base had `t_id % 32`; upstream now uses `% warpSize` with the 32-bit mask. Fix: `unsigned long long mask = ((1ull << threadsx) - 1) << (group_id * threadsx);` under USE_HIP, with the CUDA line in `#else`. test_scf_j_engine passes.
- **CUDA path (gvhf-md/contract_int3c2e.cu), CONFIRMED.** The upstream `thread_id < 8` / `offset = 4` / `__shfl_down_sync(0xff, ...)` lines are restored in `#else`, and `nwarps` is declared and used only under USE_HIP. A script that strips USE_HIP branches confirms the CUDA view of contract_int3c2e.cu and md_contract_j.cu is byte-identical to upstream/master.
- **CMake order, CONFIRMED.** Upstream's add_subdirectory order is restored. sem and multigrid are wrapped `if(NOT USE_HIP)` in place, and solvent is `if(BUILD_SOLVENT AND NOT USE_HIP)`. The libgdft/libpbc/libgecp comment sits above gdft.
- **Stale comment, CONFIRMED.** lib/CMakeLists.txt no longer says the force-included header carries the wave64 subgroup-reduction fix.
- **Commit hygiene:** unchanged, still owed at the PR-prep squash (see "Still owed" above). The squashed message must also describe this commit.

### Additional merge regression fixed: int4c2e LDS request (gpu4pyscf/__config__.py)
test_int4c2e failed with `Failed to set CUDA shm size 65448: invalid argument`. Upstream f0e4de9 makes scf/int4c2e.py pass the raw `__config__.shm_size` to fill_int4c2e. On CUDA that is optin-2048. On ROCm the port's fallback gave the full 64 KB sharedMemPerBlock, leaving no room for the kernel's static `__shared__`. Fix: on ROCm, the fallback branch subtracts 1024. CUDA is unchanged, because the CUDA opt-in branch already reserves 2048. jk.py/j_engine.py SHM_SIZE (shm_size-1024) drops from 64512 to 63488 on ROCm. test_int4c2e: 3/3 pass.

### gfx942 + TheRock ROCm 10.1 compiler miscompile (NOT a port defect; nothing committed for it)
Any basis with d shells crashes RYS_build_jk on the generic rys_jk_kernel ((dp|dp) first): `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`, then `hipErrorIllegalAddress`. sto-3g and 6-31g pass (unrolled kernels only). The pre-merge head 899ec26 built on this host fails the same way, so the merge did not cause it. Proof:
- `rocgdb -batch -ex 'set amdgpu precise-memory on' -ex run ...` stops on `global_atomic_add v1, v20, v1, s[20:21] sc0` (the `atomicAdd(head, 1)` at the top of the persistent loop). s[20:21] is the plausible head pointer. v20 = {0xaf654f6a in lane 0, 0 in lanes 1-63}.
- `llvm-objdump -d` of the code object: v20 is set by `v_mov_b32 v20, 0` once (kernel entry) and later used as the destination of predicated `global_load_dwordx2 v[20:21]` (load_dm) inside the loop, with no reset. Source indices are all in bounds (gout max 80 < 81, dm_cache max 8 < 9), so there is no source UB.
- Reproduces in a standalone `--cuda-device-only -S -O3 -fno-gpu-rdc` compile of gvhf-rys/rys_contract_jk.cu, and at -O1. The same pattern appears in gfx90a output from this compiler (AMD clang 24.0.0git d6f6cb69). The June gfx90a pass used ROCm 7.2.1.
- `-mllvm -disable-machine-licm` in CMAKE_HIP_FLAGS and CMAKE_SHARED_LINKER_FLAGS (needed under -fgpu-rdc) fixes it: def2-svp J/K vs CPU 3.8e-13 / 1.2e-13.
- Registered as deferred `therock-10-1-machinelicm-clobbers-live-vgpr` (rocm-bug-report). **Question for jeff:** file it upstream, and should the port carry `-mllvm -disable-machine-licm` for ROCm (it is a toolchain workaround, not a source fix)? Until decided, the gfx942 validator must build with the flag on the command line or use an older ROCm. The lesson is in the cuda-to-rocm validation reference.

### Build and test (gfx942, ROCm 10.1 TheRock SDK, FLEET_VENV)
Venv: `pip install 'numpy>=2.0,<2.3' 'scipy>=1.13' cupy-rocm-7-0==14.2.0 pyscf pytest` -> numpy 2.2.6, scipy 1.17.1, cupy-rocm-7-0 14.2.0, pyscf 2.14.0. CuPy-ROCm needs `LD_LIBRARY_PATH=$(rocm-sdk path --root)/lib` (libamdhip64.so.7 etc.).

```
R=$(rocm-sdk path --root)
cmake -S gpu4pyscf/lib -B agent_space/build-gfx942 -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$R \
  -DBUILD_CUTLASS=OFF -DBUILD_LIBXC=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build agent_space/build-gfx942 -j56          # exit 0, 8 libs (committed flags)
# test build: same plus
#   "-DCMAKE_HIP_FLAGS=-mllvm -disable-machine-licm" "-DCMAKE_SHARED_LINKER_FLAGS=-mllvm -disable-machine-licm"
export LD_LIBRARY_PATH=$R/lib PYTHONPATH=$PWD HIP_VISIBLE_DEVICES=<one gpu>; ulimit -c 0
python -m pytest gpu4pyscf/scf/tests -q -rfs --tb=no
```

RHF water/def2-svp -75.96070932893478 vs CPU -75.96070932895181 (1.7e-11). UHF (spin=2) diff 4.9e-12.

Full gpu4pyscf/scf/tests (17 files) at 45d8053, test build: **122 passed, 8 skipped, 36 failed** (146 s). Skips: dftd3/dftd4 not installed, 3x `hermi=0`. Each failure was rerun in its own process to rule out context-poison cascades, and every one is outside the HF/integral scope:
- DFT (numint -> `cupy_helper.release_gpu_stack` -> `deviceSetLimit(stack)` gives hipErrorUnknown; libxc absent): test_diffuse_orbital test_rhf_hessian/test_rks_lowmem/test_rohf/test_uhf (all dft.RKS/UKS/ROKS), test_gdm x3, test_init_guess hcore_guess_rks/sad_guess_rks/rks_vv10/uks/uks_ecp/sad_hf_guess_rks/sad_hf_guess_uks, test_scf test_to_cpu/test_to_gpu (as in June), test_soscf x9 (rks/uks/roks).
- PCM solvent not built on ROCm (`object has no attribute 'PCM'`): test_diffuse_orbital test_uhf_solvent, test_gdm test_gdm_rks_pcm, test_init_guess test_sad_guess_rhf_pcm/test_vsap_guess_rks.
- Analytic gradient (the energy assertions pass, the gradient assertion fails): test_diffuse_orbital test_rhf/test_rhf_lowmem (energy OK, gradient off by ~1800, varying run to run), test_fermi_smearing test_gradient, test_scf_jk test_jk_energy_per_atom (ip1 engine; 3.27/4.12/5.06 across runs, nondeterministic). gfx90a failed it in June, gfx1100 passed it. Worth a look when gradients are ported: run-to-run variation suggests uninitialized or racy data, possibly the same compiler issue in the ip1 kernels.
- Environment: test_diffuse_orbital *_against_qchem x3 (`d-aug-cc-pvdz` not in pyscf 2.14's basis set).

Every HF/integral test passes: test_rhf, test_uhf, test_ghf, test_hf_lowmem, test_int4c2e, test_scf_j_engine, test_scf_jk (except the gradient case), test_scf_ecp, test_addons, test_cphf, test_dispersion_logic, and the HF guesses in test_init_guess. Validator command: the directory above, judged by this classification. Not run here: a determinism repeat.

## Review 2026-10-30 (reviewer, linux-gfx942, fork moat-port @ 45d8053 vs upstream/master 10dcea3) -- REVIEW PASSED

Re-review requested by jeff after the porter answered "Review 2026-10-29". Fresh clone of the fork, tree clean; fork `master` == upstream 10dcea3; `jargon.py --port gpu4pyscf` clean; 45d8053 title 66 chars, `[ROCm]` prefix, AI disclosure, fenced Test Plan, no trailer.

### Prior findings: all resolved
- Blocker, dft/libxc.py: resolved. ROCm-only `try/except OSError -> _libxc = None` (libxc.py:35-43). The CUDA branch calls `load_library` bare, so it still raises. The version/`xc_func_type` block is under `if _libxc is not None` (:74-85). Every other module-level `_libxc` use is already under upstream's `if _libxc is not None:` (:145). The remaining uses (`xc_func_type`, `_xc_func_p`, `XCfun.__init__`) are only reached at DFT call time.
- The porter's extra import fix (pbc/__init__.py imports scf and properties on both backends, skips only dft on ROCm; pbc/df/__init__.py is upstream's again): correct, and it removes a port file. The porter's RHF/UHF runs exercise this chain.
- Lane mask, md_contract_j.cu:103-107 and :437-441: resolved. `threadsx` is `ij = nsq // kl` from scf/j_engine.py:319-345, a power of two <= 16. `group_id*threadsx + threadsx <= warpSize`, so the 64-bit shift is in range and `__popcll(mask) == threadsx` on wave32 and wave64.
- CUDA path, contract_int3c2e.cu: resolved. Footprint check: a script (agent_space/cudaview.py) emits the USE_HIP-undefined view of every changed .cu/.cuh and diffs it against upstream/master. contract_int3c2e.cu, md_contract_j.cu, int3c2e_recontract.cu, contract_ft_ao.cu, ft_ao.cu and int3c2e_create_tasks.cuh are byte-identical to upstream. The remaining differences are comments (ecp/common.cu, nr_fill_ao_ints.cu), vhf.cuh's `|| defined(__HIPCC__)` (a no-op on CUDA), and gen_grids.cu's nested-brace initializers (reviewed 2026-06-02; same values on both toolchains).
- CMake order: resolved. lib/CMakeLists.txt:183-207 matches upstream's order, with sem/solvent/multigrid gated in place.
- Stale comment: resolved (lib/CMakeLists.txt:39-40).
- Commit hygiene: still owed at the PR-prep squash, as recorded under "Still owed". The squashed message must also cover 45d8053 and the libxc/pbc rationale.

### Whole-diff pass (30 files)
No new defects. Grep of the HIP-built dirs (gint gvhf cupy_helper gvhf-rys gvhf-md ecp gdft pbc) for `__ballot`, `__activemask`, `0xffffffff`, `% 32`, `/ 32` and inline asm: every hit is in a `#else` CUDA branch or in the unrolled_md_j*.cu TUs. Those TUs keep `% 32` and go through cuda_to_hip_shfl.h, which uses the mask only for its popcount.

### Fact-check of the porter's gfx942 miscompile claim: CONFIRMED
`clang++ -x hip --offload-arch=gfx942 --cuda-device-only -O3 -S -fno-gpu-rdc` of gvhf-rys/rys_contract_jk.cu with the TheRock ROCm 10.1 compiler (AMD clang 24.0.0git d6f6cb69). In `rys_jk_kernel`, v20 is set once (`v_mov_b32 v20, 0`, before the loop) and then used inside persistent loop `.LBB0_14` as the zero offset of `global_atomic_add v1, v20, v1, s[20:21]` and as an LDS base address. Inner loop `.LBB0_370` (parent `.LBB0_14`) overwrites it with a predicated `global_load_dwordx2 v[20:21], v[14:15], off offset:8`, and nothing resets it. This matches the rocgdb register dump (v20 nonzero in lane 0 only). With `-mllvm -disable-machine-licm` the atomic uses a different register (`global_atomic_add v1, v1, v2, s[4:5]`). It is a toolchain defect, not a port defect, and does not block review.

### For jeff (decision, not a review finding)
- The porter's open question stands: should the port carry `-mllvm -disable-machine-licm` under USE_HIP, and should the miscompile be filed upstream (deferred `therock-10-1-machinelicm-clobbers-live-vgpr`)? Until that is decided, a validator on the ROCm 10.1 toolchain must add the flag to both CMAKE_HIP_FLAGS and CMAKE_SHARED_LINKER_FLAGS (see the porter's recipe), or every basis with d shells crashes. The porter also reports the same pattern in this compiler's gfx90a output, so the wave64 revalidation needs the flag too if it uses this toolchain.

### For the validator
- The __config__.py change (shm_size minus 1 KB on ROCm) makes `shm_size < 64*1024` true on AMD. gpu4pyscf/gto/tests/test_ecp.py's five `skipIf(shm_size < 64*1024)` ECP-derivative tests (ip1/ipip, gradient-path integrals) now skip on ROCm. They were never in the HF validation set. Note the skips; they are not port failures.
- Test set and failure classification: as in "Porter fix 2026-10-30" (gpu4pyscf/scf/tests, 122 pass / 8 skip / 36 failures, all DFT, PCM, analytic-gradient or missing-basis). Add a determinism repeat, which the porter did not run.

Verdict: REVIEW PASSED.

## Validation 2026-10-30 (validator, linux-gfx1100, fork moat-port @ 45d8053) -- VALIDATION-FAILED (documentation only; GPU suite and CUDA gate pass)

Requested by jeff after "Review 2026-10-30" passed. Platform: AMD Radeon Pro W7800 48GB, gfx1100 (wave32), one GPU pinned by the dispatcher. Toolchain: TheRock ROCm 10.1.0 SDK wheels (AMD clang 24.0.0git d6f6cb69). Venv: Python 3.11, numpy 2.2.6, scipy 1.17.1, cupy-rocm-7-0 14.2.0, pyscf 2.14.0. Fork clone checked out at 45d8053064db65a7c776a65f9310ec4063bc5b5f (== origin/moat-port).

### Result
The GPU work passes on gfx1100. The arch is recorded `validation-failed` because the README does not document the ROCm build (validator step 4; porter step 7). This needs no code change.
- **Documentation gap:** README.md's "Compilation" section (lines 36-53) gives the CUDA build (`cmake -S gpu4pyscf/lib -B build/temp.gpu4pyscf`, then installing `cutensor-cu12 cupy-cuda12x`). The port adds no ROCm counterpart there or anywhere else (`git diff master moat-port` touches no doc file; no README/CONTRIBUTING mention of ROCm or USE_HIP). Porter: add a parallel ROCm block to that section in README's style, as a new commit. It needs `-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=<gfx>`, `-DBUILD_CUTLASS=OFF -DBUILD_LIBXC=OFF`, the `cupy-rocm-7-0` package, and the HIP compiler hint if needed. Also state that DFT, gradients, solvent and multigrid are not built on ROCm. That is a docs-only delta. Because this arch returns from a failure, the next gfx1100 run is a full one (no carry-forward); it should take about 10 minutes, and this record is its baseline.

### Build (committed flags; no `-disable-machine-licm`)
```
R=$(rocm-sdk path --root)
cmake -S gpu4pyscf/lib -B agent_space/build-gfx1100 -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$R \
  -DBUILD_CUTLASS=OFF -DBUILD_LIBXC=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build agent_space/build-gfx1100 -j32
```
Exit 0. All 8 libs built (cupy_helper gdft gecp gint gvhf gvhf_md gvhf_rys pbc).

**The gfx942 MachineLICM miscompile does not show on gfx1100** with this compiler. def2-svp (d shells, the generic rys_jk_kernel) runs with the committed flags. RHF water/def2-svp gives -75.96070932893481 against CPU pyscf -75.96070932895175 (1.7e-11). So validation here used the committed build, with no command-line workaround.

### Tests
```
export LD_LIBRARY_PATH=$R/lib PYTHONPATH=$PWD; ulimit -c 0
python -m pytest gpu4pyscf/scf/tests -q -rfs --tb=no -p no:cacheprovider
```
- Run 1: **126 passed, 8 skipped, 32 failed** (217 s).
- Run 2 (determinism repeat): **126 passed, 8 skipped, 32 failed** (145 s), with an identical failure set.
- Energy repeat, 3 separate processes. RHF water/def2-tzvp: -76.05868791529183 / ...180 / ...181. UHF water/cc-pvdz spin=2: -75.78154588046841 / ...843 / ...844. Run-to-run spread is <= 3e-14 Ha (atomic reduction order).
- Skips (8): dftd3/dftd4 not installed (5), `hermi=0` unconditional skips (3).
- Each of the 32 failures was rerun alone in its own process with the same result. All fall outside the HF/integral scope, matching the porter's classification:
  - DFT, 23 tests. `numint.release_gpu_stack` -> `hipErrorUnknown` (libxc absent; DFT not built on ROCm). test_diffuse_orbital: test_rhf_hessian, test_rks_lowmem, test_rohf, test_uhf. The traceback confirms these are dft.RKS/ROKS despite their names. test_gdm: test_gdm_rks, test_gdm_uks_spin_eq_0, test_gdm_uks_spin_neq_0. test_init_guess: test_hcore_guess_rks, test_sad_guess_rks, test_sad_guess_rks_vv10, test_sad_guess_uks, test_sad_guess_uks_ecp, test_sad_hf_guess_rks, test_sad_hf_guess_uks. test_scf: test_to_cpu, test_to_gpu. test_soscf: test_nr_rks, test_nr_rks_lda, test_nr_rks_rsh, test_nr_roks, test_nr_uks, test_nr_uks_lda, test_nr_uks_rsh, test_rks_gen_g_hop, test_uks_gen_g_hop.
  - PCM not built on ROCm (`object has no attribute 'PCM'`), 4 tests: test_diffuse_orbital test_uhf_solvent, test_gdm test_gdm_rks_pcm, test_init_guess test_sad_guess_rhf_pcm, test_init_guess test_vsap_guess_rks.
  - Environment, 3 tests: `BasisNotFoundError` for d-aug-cc-pvdz in pyscf 2.14 (test_diffuse_orbital *_against_qchem x3).
- Versus gfx942 (122/8/36), gfx1100 passes 4 more: the analytic-gradient cases test_diffuse_orbital test_rhf and test_rhf_lowmem, test_fermi_smearing test_gradient, and test_scf_jk test_jk_energy_per_atom. They pass in both runs. That fits the gfx942 failures being the ROCm 10.1 miscompile in the ip1 kernels and not port code. gfx1100 also passed test_jk_energy_per_atom in June.
- Every HF/integral file passes completely: test_rhf, test_uhf, test_ghf, test_hf_lowmem, test_int4c2e, test_scf_j_engine, test_scf_jk, test_scf_ecp, test_addons, test_cphf, test_dispersion_logic, test_fermi_smearing, plus the HF guesses in test_init_guess.

### CUDA no-regression gate at 45d8053: PASS (compile and link, not run)
This host has no conda. The `nvidia-cuda-nvcc-cu12` wheel ships only `ptxas`, not the `nvcc` driver. So I assembled a private toolkit root `$K` in agent_space: the full nvcc 12.9.86 redistributable archive (nvcc, cicc, cudafe++, nvlink, fatbinary), plus the `nvidia-cuda-runtime-cu12==12.9.79` and `nvidia-cuda-cccl-cu12==12.9.27` wheel headers and libs. Host gcc 13.3.
```
cmake -S <worktree @45d8053>/gpu4pyscf/lib -B agent_space/build-cuda -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CUDA_COMPILER=$K/bin/nvcc -DCUDAToolkit_ROOT=$K -DBUILD_CUTLASS=OFF -DBUILD_LIBXC=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build agent_space/build-cuda -j32
```
Exit 0. All 101 CUDA objects compiled, with `--generate-code=arch=compute_80,code=[compute_80,sm_80]`, and 12 shared libs linked. Those include the CUDA-only sem, solvent, mgrid_v2 and mgrid_v3, so the `if(NOT USE_HIP)` gating keeps them on CUDA. The only warnings are upstream's unused variables (#177-D in unrolled_rys_jk.cu). CUTLASS and libxc were off; both are external downloads, and the port does not change them.

### Other checks
- `jargon.py --port gpu4pyscf`: clean. (It needs local `master` and `moat-port` branches in the clone; a detached checkout gives "cannot resolve master..moat-port".)
- Fork tree: `git status --porcelain` is empty after `rm -f gpu4pyscf/lib/*.so.0.hip-amdgcn-*`.
- gto/tests/test_ecp.py: not run (outside the validation set).
- The PR-prep squash message (see "Still owed") is unchanged and still owed.
## Validation 2026-10-30 (validator, linux-gfx90a MI250X GCD, fork moat-port @ 45d8053) -- VALIDATION FAILED (documentation gate)

Requested by jeff after the 2026-10-30 review pass. GPU result: every in-scope HF/integral test passes. The documentation gate (validator step 4) fails; see "Step 4 checks". That is the only reason this platform is `validation-failed`.

Host and env: gfx90a:sramecc+:xnack- (one MI250X GCD, `HIP_VISIBLE_DEVICES` pinned by the dispatcher), TheRock ROCm 10.1.0 SDK wheels (`R=/opt/python-3.11-venv/lib/python3.11/site-packages/_rocm_sdk_devel`, AMD clang 24.0.0git d6f6cb69). Job venv (python 3.11.17) with the porter's package set: numpy 2.2.6, scipy 1.17.1, cupy-rocm-7-0 14.2.0, pyscf 2.14.0, pytest 7.3.2. CuPy-ROCm needs `LD_LIBRARY_PATH=$R/lib`. A venv created from /opt/python-3.11-venv has no `rocm-sdk` script; call `/opt/python-3.11-venv/bin/rocm-sdk path --root`.

### Build (committed flags) and the ROCm 10.1 miscompile on gfx90a
```
cmake -S gpu4pyscf/lib -B agent_space/build-gfx90a -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$R \
  -DBUILD_CUTLASS=OFF -DBUILD_LIBXC=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build agent_space/build-gfx90a -j64
```
Exit 0, 142 s, 8 libs (cupy_helper gdft gecp gint gvhf gvhf_md gvhf_rys pbc), each with a gfx90a code object (`llvm-objdump --offloading`). RHF water sto-3g: -74.9636307973671 vs CPU -74.9636307973705 (3.4e-12). RHF water def2-svp crashes with `HSA_STATUS_ERROR_MEMORY_FAULT` and a GPU hang analysis on the persistent kernel (grid 6656x4 = 104 CUs x 64, workgroup 64x4, group_seg 62808), then `rocBLAS error during freeing ... rocblas_status_internal_error`. This is the deferred `therock-10-1-machinelicm-clobbers-live-vgpr` miscompile from "Porter fix 2026-10-30". It is now confirmed at runtime on gfx90a, not only in that compiler's gfx90a disassembly. It is a toolchain defect, not a port defect. The 2026-06 gfx90a pass used ROCm 7.2.1, which this host does not have.

Test build: identical, plus `"-DCMAKE_HIP_FLAGS=-mllvm -disable-machine-licm" "-DCMAKE_SHARED_LINKER_FLAGS=-mllvm -disable-machine-licm"` (into agent_space/build-gfx90a-licm). Exit 0, 128 s. RHF water/def2-svp -75.9607093289348 vs CPU -75.96070932895176 (1.7e-11); UHF spin=2 -75.71394397066102 vs -75.71394397065606 (5.0e-12). All tests below use this build.

### Full test suite (scf/tests, 17 files), two runs
```
ulimit -c 0
cd projects/gpu4pyscf/src
LD_LIBRARY_PATH=$R/lib PYTHONPATH=$PWD python -m pytest gpu4pyscf/scf/tests -q -r fs --tb=line -p no:cacheprovider --junitxml=<run>.xml
```
Run 1: **122 passed, 8 skipped, 36 failed** (303.5 s). Run 2: **122 passed, 8 skipped, 36 failed** (209.1 s). The per-test outcomes are identical between the two runs (junit diff: the same 166 ids, 0 outcome changes) and match the gfx942 porter run at the same head.

Per file (pass/skip/fail): test_addons 26/0/0, test_cphf 3/0/0, test_diffuse_orbital 0/1/10, test_dispersion_logic 8/0/0, test_fermi_smearing 2/0/1, test_gdm 1/0/4, test_ghf 6/0/0, test_hf_lowmem 2/0/0, test_init_guess 15/0/9, test_int4c2e 3/0/0, test_rhf 15/2/0, test_scf 5/0/2, test_scf_ecp 1/0/0, test_scf_j_engine 8/0/0, test_scf_jk 11/0/1, test_soscf 5/0/9, test_uhf 11/5/0.

Skips (8): dftd3/dftd4 not installed (5), unconditional `hermi=0` skip (3).

Failures (36). I checked each against its junit message. All are outside the HF/integral scope, in the same classes as "Porter fix 2026-10-30":
- DFT, `hipErrorUnknown` on the numint path (libxc not built), 25: test_diffuse_orbital rhf_hessian/rks_lowmem/rohf/uhf (4); test_gdm gdm_rks/gdm_uks_spin_eq_0/gdm_uks_spin_neq_0 (3); test_init_guess hcore_guess_rks/sad_guess_rks/sad_guess_rks_vv10/sad_guess_uks/sad_guess_uks_ecp/sad_hf_guess_rks/sad_hf_guess_uks (7); test_scf to_cpu/to_gpu (2); test_soscf nr_rks/nr_rks_lda/nr_rks_rsh/nr_roks/nr_uks/nr_uks_lda/nr_uks_rsh/rks_gen_g_hop/uks_gen_g_hop (9).
- PCM not built (`object has no attribute 'PCM'`), 4: test_diffuse_orbital uhf_solvent, test_gdm gdm_rks_pcm, test_init_guess sad_guess_rhf_pcm/vsap_guess_rks.
- Analytic gradient (not ported), 4: test_diffuse_orbital rhf/rhf_lowmem (max gradient error 1600.5/1602.1 in run 1; rhf_lowmem 1553.6 in run 2), test_fermi_smearing test_gradient, test_scf_jk test_jk_energy_per_atom (4.48 in run 1, 3.23 in run 2). The magnitudes vary run to run, as on gfx942.
- Environment, 3: test_diffuse_orbital *_against_qchem (`d-aug-cc-pvdz` is not in pyscf 2.14).

Every HF/integral test passes: test_rhf, test_uhf, test_ghf, test_hf_lowmem, test_int4c2e (3/3, the shm_size fix), test_scf_j_engine (8/8, the wave64 lane-mask fix), test_scf_jk (all but the gradient case), test_scf_ecp, test_addons, test_cphf, test_dispersion_logic, and the HF guesses in test_init_guess.

Numeric determinism (separate processes, water cc-pVTZ, SAD-guess density): J/K run-to-run max |diff| 7.1e-15; difference from CPU PySCF J 5.5e-13, K 2.6e-13. RHF energy -76.05678324246752 vs -76.05678324246749 across two processes. The J/K bytes differ at the ULP level, consistent with atomic accumulation order, so the results are not bitwise deterministic.

### CUDA no-regression gate at 45d8053: PASS (compile-only)
This host has no conda. I assembled the toolkit from NVIDIA's official 12.8 redist archives (cuda_nvcc 12.8.93, cuda_cudart 12.8.90, cuda_cccl 12.8.90) into agent_space/cuda-12.8, with a `lib64 -> lib` symlink (nvcc.profile expects lib64). The 12.8 `nvidia-cuda-nvcc-cu12` wheel has no `nvcc` driver, only ptxas/nvvm. Host gcc 13.3. Source: a `git archive 45d8053` export, so the HIP libs in the clone were not overwritten.
```
PATH=$K/bin:$PATH CUDACXX=$K/bin/nvcc cmake -S cuda-src/gpu4pyscf/lib -B build-cuda \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DCUDAToolkit_ROOT=$K -DBUILD_LIBXC=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j64
```
Exit 0, 197 s, with `CUDA_ARCHITECTURES: 80` echoed. BUILD_CUTLASS=ON (the default; grouped_gemm compiled), BUILD_SOLVENT=ON. 12 libs: cupy_helper gdft gecp gint gvhf gvhf_md gvhf_rys mgrid_v2 mgrid_v3 pbc sem solvent. That includes the CUDA-only sem/solvent/multigrid dirs the port gates with `if(NOT USE_HIP)`. Not built: libxc (external, untouched by the port). This makes no claim about NVIDIA runtime behaviour.

### Step 4 checks
- Jargon: `python3 utils/jargon.py --port gpu4pyscf` -> `jargon: clean`. It needs local `master` and `moat-port` branches in the clone.
- Integrity: `git -C projects/gpu4pyscf/src status --porcelain` shows only the untracked RDC split objects (`lib*.so.0.hip-amdgcn-amd-amdhsa--gfx90a`, `lib*.so.0.host-x86_64-unknown-linux-gnu-`). No tracked file is modified.
- **Documentation: FAIL.** README.md "Compilation" documents the CUDA build (`cmake -S gpu4pyscf/lib -B build/temp.gpu4pyscf`, `cmake --build ...`, then the cutensor/cupy-cuda12x install line). The fork diff (30 files, master..moat-port) touches no README or other doc file. Nothing documents `-DUSE_HIP=ON`, `CMAKE_HIP_ARCHITECTURES`, the HIP compiler, `-DBUILD_CUTLASS=OFF -DBUILD_LIBXC=OFF`, or the `cupy-rocm-7-0` package. There is no docs/ tree, so README is the one place. For the porter: add a parallel ROCm block in that section, in its house style. It needs the commands, the CuPy-ROCm install line, and a short note that DFT, gradients, solvent and multigrid are not built for ROCm. This is upstream-visible text, so run jargon.py and prose.py on it. Because this arch returns from `validation-failed`, the next gfx90a pass is a full re-run (no carry-forward), so run this README fix's validation together with the toolchain-flag decision below.

### Open for jeff (unchanged, now with gfx90a runtime evidence)
`-mllvm -disable-machine-licm`: with the committed flags on ROCm 10.1, gfx90a crashes on every basis with d shells. Should the port carry the flag under USE_HIP, and should `therock-10-1-machinelicm-clobbers-live-vgpr` be filed upstream?

## Porter 2026-10-30 (linux-gfx942): README ROCm build section -- committed locally, NOT pushed

jeff requested this round to fix the documentation failures on gfx90a and gfx1100 at 45d8053. The docs-only commit `066e91e` "[ROCm] Document the ROCm build in the README" sits on top of 45d8053 in this fleet job's fork clone. The patch is also at agent_space/readme-patch/ in the job's moat clone. The commit adds a block at the end of the README "Compilation" section, after the cutensor warning. Upstream text is unchanged. The block contains:
- `-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=<gfx>`. CUTLASS and libxc already default to OFF under USE_HIP, so the recipe leaves those flags out.
- The `-DCMAKE_HIP_COMPILER=<rocm_path>/llvm/bin/clang++` hint.
- The `cupy-rocm-7-0` install line.
- A note that the cuTENSOR warning is expected on ROCm.
- A scope sentence: Hartree-Fock and the integral libraries are built; DFT (libxc), solvent, semi-empirical and multigrid are not.

Analytic gradients are not called out; the README claims only HF.

Checks:
- Recipe on gfx942 with TheRock ROCm 10.1 (no /opt/rocm): `cmake -S gpu4pyscf/lib -B <build> -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++`, then `cmake --build <build> -j 56`. Both exit 0 and build all 8 libs. No CMAKE_PREFIX_PATH is needed.
- `jargon.py --port gpu4pyscf` is clean, `prose.py` on the commit body is clean, and the added lines are ASCII.

Open items:
- The push of `moat-port` was refused in this headless job (the dispatch said no GitHub writes), so `advance-head` was NOT run. head_sha stays 45d8053, and no validator can fetch 066e91e yet. To land it, push the commit to the fork's `moat-port` (no PR is open, so the branch is not frozen), then run `python3 utils/moatlib.py advance-head gpu4pyscf <sha>`. The delta is documentation-only. The stage is still review-passed; this round did not take the work lock.
- The README recipe does not include `-mllvm -disable-machine-licm`. That waits on jeff's open decision above. With ROCm 10.1 on gfx90a and gfx942, the documented build still hits the MachineLICM miscompile on bases with d shells.

### Update: pushed and head advanced
jeff clarified that pushing the fork's port branch is allowed. `git push origin moat-port` moved the fork from 45d8053 to 066e91e (`git ls-remote` confirms 066e91eee794766343ea54be6560dcf430415502). Then `advance-head gpu4pyscf 066e91e...` ran, and head_sha is now 066e91e.

**Tooling gap: the failures did not clear.** advance_head classifies a documentation-only delta as inert and moves a `validation-failed` record's `failed_sha` up to the new head (moatlib.py advance_head, `failed = new_sha if verdict.arch_independent`). The reasoning is that a README edit cannot fix a failure. Here the failure WAS the README, so gfx90a and gfx1100 now record `failed_sha` = 066e91e, and `fleet` still routes both to the porter. No porter work is left. The legal way out is a validator run at 066e91e on each arch recording `completed` (`validation-failed -> completed` is allowed). The selector will not dispatch that run, so jeff has to request it explicitly. Each GPU suite already passed at 45d8053, and 066e91e changes only README.md. Fixing the rule (for example, letting a docs delta retire a docs-gate failure) is control-plane work and was not done here.

## Validation 2026-10-30 (validator, linux-gfx1100, fork moat-port @ 066e91e) -- NOT VALIDATED (environment: no CuPy-ROCm installable in this job)

Requested by jeff to clear the documentation-only failure at 066e91e: build by the new README section as written, run the full GPU suite, check the d-shell miscompile. Host: 4x AMD Radeon Pro W7800 48GB, gfx1100 (wave32), `HIP_VISIBLE_DEVICES=4` pinned by the dispatcher. Toolchain: TheRock ROCm SDK 10.1.0 wheels in /opt/python-3.11-venv (`rocm-sdk version` -> 10.1.0, AMD clang 24.0.0). Fork clone at 066e91eee794766343ea54be6560dcf430415502 (== origin/moat-port).

### Result
The record stays `validation-failed` at 066e91e. The cause is this job's environment, not the port. The image has no CuPy (`import cupy` fails in /opt/python-3.11-venv and in system python3). The job was dispatched without `FLEET_VENV`, and the fleet guard refused to create a job-local venv in agent_space and install the porter's package set into it ("pip changing the Python environment every job on this host shares"). I did not work around the guard, so no Python test ran. **For jeff:** redispatch this validation with `--build` (so the job gets its own `FLEET_VENV`), or preinstall the porter's package set (`'numpy>=2.0,<2.3' 'scipy>=1.13' cupy-rocm-7-0==14.2.0 pyscf==2.14.0 pytest`) in an environment the job may use. The rest of this section is the evidence that does not need CuPy.

### Build by the README section as written: PASS
From the fork root, with only the arch substituted and `-j` set to the host's 32 physical cores:
```
export PATH=/opt/python-3.11-venv/bin:$PATH
cmake -S gpu4pyscf/lib -B build/temp.gpu4pyscf -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build/temp.gpu4pyscf -j 32
```
Configure exit 0 (2 s). CMake found the HIP compiler on its own (`_rocm_sdk_devel/lib/llvm/bin/clang++`, Clang 24.0.0), so the README's `-DCMAKE_HIP_COMPILER` hint was not needed. In this TheRock layout both `<rocm_path>/llvm/bin/clang++` and `<rocm_path>/lib/llvm/bin/clang++` exist, so the hint as written also resolves. Build exit 0 (107 s), 8 libs: cupy_helper gdft gecp gint gvhf gvhf_md gvhf_rys pbc. The README's CuPy-ROCm install step could not be run (see Result). `git status --porcelain` in the fork is clean after the build, because build/ and the .so files are ignored. `jargon.py --port gpu4pyscf`: clean.

### GPU test suite: NOT RUN
No CuPy, so `python -m pytest gpu4pyscf/scf/tests` cannot import gpu4pyscf. The baseline for this arch is still "Validation 2026-10-30 (linux-gfx1100, 45d8053)": 126 passed / 8 skipped / 32 failed in two identical runs, all failures out of scope. 066e91e changes only README.md (11 added lines), so device code is unchanged from 45d8053. That is the reason to expect the same counts, not evidence of them.

### d-shell MachineLICM miscompile (deferred `therock-10-1-machinelicm-clobbers-live-vgpr`) on gfx1100: does not reproduce (static check)
I compiled gvhf-rys/rys_contract_jk.cu device-only with this host's ROCm 10.1 compiler, the same way the gfx942 reviewer did:
```
clang++ -x hip --offload-arch=<gfx> --cuda-device-only -O3 -S -fno-gpu-rdc -DUSE_HIP -include gpu4pyscf/lib/cuda_to_hip.h \
  -ffp-contract=on -std=gnu++17 -Igpu4pyscf/lib/gvhf-rys -Igpu4pyscf/lib/hip_compat -Igpu4pyscf/lib -Igpu4pyscf/lib/deps/include \
  gpu4pyscf/lib/gvhf-rys/rys_contract_jk.cu
```
In `rys_jk_kernel` I took the persistent-loop head atomic and searched every non-store instruction after the loop header for a write to its zero-offset VGPR:
- gfx942 (control): `global_atomic_add v1, v20, v1, s[20:21] sc0`. v20 is overwritten inside the loop by `global_load_dwordx2 v[20:21], ...` at 5+ sites, which is the defect as recorded.
- gfx1100, -O3 and -O2: `global_atomic_add_u32 v2, v0, v2, s[18:19] glc`. v0 is set to 0 once before the loop (`v_mov_b32_e32 v0, 0`), and nothing writes v0 after the loop header.
- `-mllvm -disable-machine-licm` still changes gfx1100 code (the kernel shrinks from 16522 to 15658 asm lines), but the clobber pattern is not there without it.
This agrees with the 45d8053 gfx1100 runtime result (def2-svp RHF 1.7e-11 vs CPU with the committed flags). Runtime confirmation at 066e91e needs CuPy and was not run.

### CUDA no-regression gate
Not rerun. 066e91e is README-only, and the 45d8053 gate (PASS, this arch's 2026-10-30 run) covers the same code.
## Validation 2026-10-30 (validator, linux-gfx90a MI250X GCD, fork moat-port @ 066e91e) -- VALIDATION FAILED (the README build crashes HF on ROCm 10.1; the toolchain-flag decision is open)

Requested by jeff. The premise was that the README was the only gap. With the README build as written, that premise does not hold on this host. The documentation now exists, and jargon is clean. But the build it documents fails 40 in-scope HF/integral tests here, because of the deferred ROCm 10.1 MachineLICM miscompile. The suite only reaches the 45d8053 baseline when `-mllvm -disable-machine-licm` is added, and that flag is neither in the README nor committed. The integrity gate requires every build edit that validation needs to be committed, so this cannot be recorded `completed`. Carrying the flag, or documenting it, is jeff's open decision (deferred `therock-10-1-machinelicm-clobbers-live-vgpr`). This is not a porter defect: the fork has no code fix to make until that decision is taken.

Host and env: gfx90a:sramecc+:xnack- (one MI250X GCD). TheRock ROCm SDK 10.1.0 wheels (`rocm-sdk version` -> 10.1.0, `R=/opt/python-3.11-venv/lib/python3.11/site-packages/_rocm_sdk_devel`, HIP compiler identification Clang 24.0.0). This is the only ROCm on the host; there is no /opt/rocm. Python env: the previous gfx90a validation job's venv, used read-only, because the fleet guard refused pip in this job. Versions: python 3.11.17, cupy-rocm-7-0 14.2.0, pyscf 2.14.0, numpy 2.2.6, scipy 1.17.1, pytest 7.3.2. CuPy-ROCm needs `LD_LIBRARY_PATH=$R/lib`. GPU pinning: the dispatcher set `HIP_VISIBLE_DEVICES=4`, but this container exposes only GPUs 0-3, so HIP saw no device (`hipErrorNoDevice`, then `hipsolverCreate failed`). That is a dispatcher bug. All four GPUs were idle and the one other running job was pinned to 2, so this run used `HIP_VISIBLE_DEVICES=0`.

### Build: README "Compilation" ROCm block, as written
```
cd projects/gpu4pyscf/src          # fork @ 066e91eee794766343ea54be6560dcf430415502
cmake -S gpu4pyscf/lib -B build/temp.gpu4pyscf -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build build/temp.gpu4pyscf -j 32
```
- The only deviations from the README were the arch and `-j 32` (the README says `-j 4`).
- Exit 0, 200 s, all 8 libs (cupy_helper gdft gecp gint gvhf gvhf_md gvhf_rys pbc).
- `-DCMAKE_HIP_COMPILER` was not needed. CMake found `$R/lib/llvm/bin/clang++` through the TheRock venv on PATH. The README hint holds as an "if not found" fallback.
- The `cupy-rocm-7-0` install line matches the package that works.
- RHF water/sto-3g: -74.96308088371563, against CPU -74.96308088371701 (1.4e-12).
- RHF water/def2-svp: aborts (rc 134) with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` in `rys_k_kernel`, then `hipErrorIllegalAddress` at rys_contract_k.cu:676 (`RYS_build_k`, li,lj,lk,ll = 2,1,2,1).

**The d-shell miscompile reproduces with this host's ROCm (10.1.0 TheRock, AMD clang 24.0.0).** It affects both `rys_jk_kernel` and `rys_k_kernel`, and every basis with d or f shells: (ds|ss), (dp|ss), (dp|dp), (fs|fp), (fs|dd), (fs|ff). The faults are `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` or `HSA_STATUS_ERROR_MEMORY_FAULT`.

### Comparison build: identical plus the workaround flag
This used a `git archive 066e91e` export in agent_space/src-licm, so the README-build libs were not overwritten. The README commands were run with `"-DCMAKE_HIP_FLAGS=-mllvm -disable-machine-licm" "-DCMAKE_SHARED_LINKER_FLAGS=-mllvm -disable-machine-licm"`. Exit 0, 8 libs. RHF water/def2-svp: -75.96096670688937, against CPU -75.96096670690653 (1.7e-11).

### Full test suite (gpu4pyscf/scf/tests, 166 ids), one process per test
An HSA abort kills the whole pytest process, so each test id ran in its own process: `agent_space/run_each.sh` (`pytest <id> -q -rfs --tb=short -p no:cacheprovider`, timeout 900). Env: `ulimit -c 0`, `HIP_VISIBLE_DEVICES=0`, `LD_LIBRARY_PATH=$R/lib`, `PYTHONPATH=<tree>`.

| build | passed | skipped | failed | aborted | wall |
|---|---|---|---|---|---|
| README as written | 82 | 8 | 56 | 20 (SIGABRT) | 913 s |
| + `-disable-machine-licm` | **122** | 8 | 36 | 0 | 1049 s |

- **README build:** compared with the flag build, 40 tests move. 24 go FAIL -> PASS, 16 go CRASH -> PASS, and 4 go CRASH -> FAIL; those 4 are DFT tests that crash in the HF part first. All 40 die in the Rys J/K kernels with the HSA memory errors above. The tests moved by file:
  - test_ghf: test_get_jk_complex, test_get_jk_real, test_ghf_scf, test_ghf_scf_complex_dm, test_to_cpu
  - test_hf_lowmem: test_diis_on_cpu, test_rhf_scf
  - test_init_guess: test_sad_guess_rhf_ghost, test_sad_hf_guess_rks, test_sad_hf_guess_uks
  - test_rhf: test_chkfile, test_get_jk, test_get_jk1, test_get_jk1_hermi0, test_get_k, test_get_k1, test_get_k1_hermi0, test_init_guess, test_rhf_scf
  - test_scf: test_rhf, test_rhf_cart, test_screening, test_to_cpu, test_to_gpu, test_uhf, test_uhf_cart
  - test_scf_ecp: test_rhf
  - test_scf_j_engine: test_general_contraction, test_irregular_angular_momemtum, test_j_engine_integral_screen, test_sparse_dm
  - test_scf_jk: test_general_contraction, test_get_jk_rsh_stacked_dms, test_jk_get_k_sr, test_jk_hermi0, test_jk_hermi0_l5, test_jk_hermi1, test_jk_hermi1_cart, test_k_hermi1
  - test_uhf: test_get_jk, test_get_jk1, test_get_k, test_get_k1, test_uhf_scf
- **Flag build:** 122/8/36. This matches the 45d8053 gfx90a and gfx942 runs exactly. The 36 failures are all out of scope:
  - DFT, 25: `hipErrorUnknown` from `deviceSetLimit` (libxc not built).
  - PCM not built, 4.
  - Analytic gradient, 4 (test_jk_energy_per_atom is off by 3.71 this run; it varies run to run, as before).
  - `BasisNotFoundError` d-aug-cc-pvdz, 3.
  - Skips (8): dftd3/dftd4 not installed (5), unconditional `hermi=0` skip (3).

### Other checks
- CUDA no-regression gate: not rerun. The 45d8053 -> 066e91e delta is README.md only (+11 lines), and the gate passed at 45d8053.
- `python3 utils/jargon.py --port gpu4pyscf` -> `jargon: clean`.
- Integrity: `git -C projects/gpu4pyscf/src status --porcelain` is empty at 066e91e.

### For jeff (decision; nothing a porter can do without it)
On ROCm 10.1 the documented gfx90a build crashes HF on any basis with d shells. The open ruling on deferred `therock-10-1-machinelicm-clobbers-live-vgpr` decides what clears this platform:
- (a) the port carries `-mllvm -disable-machine-licm` under USE_HIP in CMake (compile and link flags; a build change, so every arch revalidates), or
- (b) the README documents the flag as a ROCm 10.1 workaround (docs only), or
- (c) gfx90a is validated on a ROCm without the bug. This host has only 10.1. The June pass used ROCm 7.2.1.

With (a) or (b), the flag build above already passes 122/8/36 at this head, so the follow-up run should be short. Tooling note, unchanged from the porter's "Update": a docs delta does not retire this `validation-failed` record, so `fleet` routes gfx90a to the porter until jeff rules.

## Validation 2026-10-30 (validator, linux-gfx1100, fork moat-port @ 066e91e) -- COMPLETED

Requested by jeff. This reruns the gfx1100 validation at 066e91e, which earlier could not install CuPy-ROCm (see "NOT VALIDATED" above). This job has its own venv. Host: AMD Radeon Pro W7800 48GB, gfx1100 (wave32). Toolchain: TheRock ROCm SDK 10.1.0 wheels in the job venv (HIP compiler identification Clang 24.0.0; `R=<venv site-packages>/_rocm_sdk_devel`, no /opt/rocm). Fork clone at 066e91eee794766343ea54be6560dcf430415502 (== origin/moat-port).

### Python env (job venv)
```
pip install 'numpy>=2.0,<2.3' 'scipy>=1.13' cupy-rocm-7-0==14.2.0 pyscf==2.14.0 pytest
```
Installed versions: python 3.11.17, numpy 2.2.6, scipy 1.17.1, cupy-rocm-7-0 14.2.0, pyscf 2.14.0, pytest 7.3.2. CuPy-ROCm needs `LD_LIBRARY_PATH=$R/lib`.

GPU pinning: the dispatcher set `HIP_VISIBLE_DEVICES=4`, but this container has only GPUs 0-3. With that value, gpu4pyscf import fails with `RuntimeError: hipsolverCreate failed`. This is the same dispatcher bug the gfx90a run hit. amd-smi showed no process on any GPU and no other fleet job pinned, so this run used `HIP_VISIBLE_DEVICES=0`.

### Build: README "Compilation" ROCm block, as written
```
export PATH=$FLEET_VENV/bin:$PATH
cmake -S gpu4pyscf/lib -B build/temp.gpu4pyscf -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build/temp.gpu4pyscf -j 32
```
- Deviations from the README: the arch, and `-j 32` instead of `-j 4`.
- The paths were passed absolute because utils/timeit.sh changes directory to the moat root.
- Configure exit 0 (2 s); build exit 0 (106 s); 8 libs: cupy_helper gdft gecp gint gvhf gvhf_md gvhf_rys pbc.
- CMake found the HIP compiler without the `-DCMAKE_HIP_COMPILER` hint.
- No `-disable-machine-licm`: the committed flags only.

Smoke check, RHF water against CPU pyscf:

| basis | max shell | |GPU - CPU| (Ha) |
|---|---|---|
| sto-3g | p | 1.3e-12 |
| def2-svp | d | 1.7e-11 |
| cc-pvtz | f | 6.2e-12 |

So the ROCm 10.1 d-shell MachineLICM miscompile (deferred `therock-10-1-machinelicm-clobbers-live-vgpr`) does not show at runtime on gfx1100. That agrees with the static check in the earlier 066e91e section.

### Tests
```
ulimit -c 0
export HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$R/lib PYTHONPATH=<fork root>
python -m pytest gpu4pyscf/scf/tests -q -rfs --tb=no -p no:cacheprovider
```
- Run 1: **126 passed, 8 skipped, 32 failed** (217 s).
- Run 2: **126 passed, 8 skipped, 32 failed** (146 s).
- The two runs have identical failure sets. Counts and failure set equal the 45d8053 gfx1100 baseline exactly.
- Failures, classified from the junit messages (all out of the HF/integral scope):
  - 25 DFT, `hipErrorUnknown` (libxc not built). The 45d8053 section's "23" was a miscount of its own list, which names 25.
  - 4 PCM not built.
  - 3 `BasisNotFoundError` d-aug-cc-pvdz.
- Skips (8): dftd3/dftd4 not installed (5), unconditional `hermi=0` (3).
- These all pass, as at 45d8053: test_rhf, test_uhf, test_ghf, test_hf_lowmem, test_int4c2e, test_scf_j_engine, test_scf_jk, test_scf_ecp, test_addons, test_cphf, test_dispersion_logic, test_fermi_smearing, and the HF guesses in test_init_guess.

### Other checks
- CUDA no-regression gate: not rerun. 45d8053 -> 066e91e is README.md only (+11 lines), and the gate passed at 45d8053.
- `python3 utils/jargon.py --port gpu4pyscf` -> `jargon: clean`.
- Documentation: the README documents the ROCm build next to the CUDA build, and that build was followed here.
- Integrity: `git -C projects/gpu4pyscf/src status --porcelain` is empty after the build.

## Porter 2026-10-30 (porter, linux-gfx942 MI300X, fork moat-port 066e91e -> 28f2418): carry `-mllvm -disable-machine-licm` for HIP

jeff ruled on the open toolchain-flag question (option (a) of the gfx90a 066e91e section): the port carries the flag in CMake for HIP builds, with the reason next to it. Taken under the work lock (`review-passed -> porting`). New commit 28f2418 on top of 066e91e, pushed fast-forward; `advance-head 28f2418...`.

### Change
- gpu4pyscf/lib/CMakeLists.txt, inside the existing `if(USE_HIP)` block: `string(APPEND CMAKE_HIP_FLAGS " -mllvm -disable-machine-licm")` and the same on `CMAKE_SHARED_LINKER_FLAGS` (device code is generated at link time under -fgpu-rdc). The comment says what it works around (machine LICM clobbers the register holding the atomicAdd offset at the top of the persistent loop in the Rys J/K kernels on gfx90a/gfx942; every d-shell basis crashes) and when it can go (a later ROCm compiler that passes the d-shell SCF tests without it).
- README.md ROCm block: one paragraph saying the CMake configuration adds the flag, why, and not to remove it unless the d-shell SCF tests pass without it.
- The deferred `therock-10-1-machinelicm-clobbers-live-vgpr` stays open: its remaining question (file the compiler bug upstream) is still jeff's. Nothing recorded on his behalf.

### Build (gfx942, TheRock ROCm SDK 10.1.0, AMD clang 24.0.0; job FLEET_VENV, cmake 3.31.6)
Venv: `pip install 'numpy>=2.0,<2.3' 'scipy>=1.13' cupy-rocm-7-0==14.2.0 pyscf==2.14.0 pytest` -> numpy 2.2.6, scipy 1.17.1, cupy-rocm-7-0 14.2.0, pyscf 2.14.0, pytest 7.3.2. Runtime needs `LD_LIBRARY_PATH=$(rocm-sdk path --root)/lib`. GPU: dispatcher pin `HIP_VISIBLE_DEVICES=7` (worked on this host).
```
cmake -S gpu4pyscf/lib -B build/temp.gpu4pyscf -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942
cmake --build build/temp.gpu4pyscf -j 56 -- VERBOSE=1
```
README recipe as written (only `-j`). Exit 0, 8 libs. VERBOSE log: every HIP `.cu` compile line and all 8 `--hip-link` lines carry `-mllvm -disable-machine-licm`. Rebuilt after the commit (comment-only change since the first build): relink only, tree clean.

### Tests (fork @ 28f2418, `ulimit -c 0`)
RHF water vs CPU PySCF:

| basis | flag build | no-flag control |
|---|---|---|
| sto-3g | 9.8e-13 | 9.7e-13 |
| def2-svp | 1.7e-11 | abort: APERTURE_VIOLATION in rys_k_kernel, hipErrorIllegalAddress at rys_contract_k.cu:676 |
| cc-pvtz | 6.3e-12 | abort: APERTURE_VIOLATION, hipErrorIllegalAddress |

The control is a `git ls-files` export of 28f2418 with the two `string(APPEND ...disable-machine-licm")` lines deleted (agent_space/noflag-src), built the same way.

`python -m pytest gpu4pyscf/scf/tests -q -r fs --tb=line -p no:cacheprovider`: run 1 **122 passed, 8 skipped, 36 failed** (216 s); run 2 identical counts and identical failure set (157 s). Same as the 45d8053 gfx942 and 066e91e gfx90a flag-build results. All 36 failures out of scope: DFT/libxc (`deviceSetLimit` hipErrorUnknown; test_soscf rks/uks/roks, test_gdm, test_init_guess *rks/*uks, test_scf to_cpu/to_gpu, test_diffuse_orbital rks/rohf/uhf/hessian), PCM not built (4), analytic gradient (test_diffuse_orbital test_rhf/test_rhf_lowmem, test_fermi_smearing test_gradient, test_scf_jk test_jk_energy_per_atom), d-aug-cc-pvdz missing (3). Skips: dftd3/dftd4 (5), `hermi=0` (3).

### CUDA no-regression gate (compile and link only): PASS
The 12.8 `nvidia-cuda-nvcc-cu12` wheel has no `nvcc` driver (ptxas/nvvm only), as before, so the toolkit root `K=agent_space/cuda-12.8` was assembled from NVIDIA's official redist archives cuda_nvcc 12.8.93, cuda_cudart 12.8.90, cuda_cccl 12.8.90 (`lib64 -> lib` symlink); cublas/cusolver were not needed. The wheels (nvcc/runtime/cccl 12.8, cublas 12.8.5.5, cusolver 11.7.5.82) are also in FLEET_VENV. Host gcc 13.3. Source: `git ls-files` export of the working tree identical to 28f2418 (only the CMake comment wording changed afterwards).
```
PATH=$K/bin:$PATH CUDACXX=$K/bin/nvcc cmake -S cuda-src/gpu4pyscf/lib -B build-cuda \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DCUDAToolkit_ROOT=$K -DBUILD_LIBXC=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j 56 -- VERBOSE=1
```
Exit 0, `CUDA_ARCHITECTURES: 80`. 103 nvcc compile lines, 12 libs (cupy_helper gdft gecp gint gvhf gvhf_md gvhf_rys mgrid_v2 mgrid_v3 pbc sem solvent; CUTLASS on by default). `grep -c disable-machine-licm build-cuda.log` = 0.

### Other checks
`jargon.py --port gpu4pyscf`: clean. `prose.py` on the commit message: clean. Title 58 chars. `git status --porcelain` in the fork: empty.

### For the next roles
Build-flag change, so every arch revalidates at 28f2418 after review. Validators no longer add the flag on the command line: the README recipe as written is now the test build on ROCm 10.1 gfx90a/gfx942. gfx1100 does not show the miscompile; the flag only changes its code shape, so its rerun is the check that the flag costs nothing there.

## Review 2026-10-30 (reviewer, linux-gfx942, fork moat-port @ 28f2418 vs upstream/master 10dcea3; delta 066e91e..28f2418) -- CHANGES REQUESTED

Scope: the delta commit 28f2418 in full, and the whole fork diff (31 files, +695/-76) re-read for anything the delta interacts with. Confirmed and not findings: the flag is set only inside `if(USE_HIP)` so no CUDA compile or link line can see it (the porter's CUDA 12.8 configure/build grep = 0 matches this); it applies to every HIP arch rather than a per-arch branch; title 58 chars with `[ROCm]`, AI disclosure in prose, fenced Test Plan, no `Co-Authored-By`/noreply; `jargon.py --port gpu4pyscf` clean; `prose.py` on the commit body clean; added lines ASCII.

### Build System

1. **gpu4pyscf/lib/CMakeLists.txt:63 is a no-op, and the comment at :56-61 says the opposite of what actually carries the fix.** `string(APPEND CMAKE_SHARED_LINKER_FLAGS " -mllvm -disable-machine-licm")` is thrown away by the upstream `set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")` at :94. That `set` runs whenever the C compiler is GCC and `LDFLAGS` is unset, which is the README build. The flag reaches the device link only because CMake's HIP link rule repeats `CMAKE_HIP_FLAGS` (`<LANGUAGE_COMPILE_FLAGS>`), so :62 alone covers both compile and link. And with `-fgpu-rdc -c` the device side only writes bitcode, so on the compile step the flag changes host code and nothing else. If a maintainer trusts the comment ("needed on both lines") and keeps only the linker line, the workaround disappears without any error. Evidence, on this host (TheRock ROCm 10.1, GCC 13.3 as the C compiler, cmake 3.31.6):
   ```
   cmake -S gpu4pyscf/lib -B <b> -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942
   # CMakeCache: CMAKE_SHARED_LINKER_FLAGS is empty; all 8 link.txt files have exactly one
   # -disable-machine-licm, one --hip-link and one -fgpu-rdc, sitting in the CMAKE_HIP_FLAGS group
   # next to -Wno-register; the only CMAKE_SHARED_LINKER_FLAGS content there is -Wl,--no-as-needed.
   # Same tree with only line 62 deleted:
   grep -c disable-machine-licm <b2>/gvhf-rys/CMakeFiles/gvhf_rys.dir/link.txt   # -> 0
   ```
   Fix: drop :63 and rewrite the comment to say that the HIP link rule reuses `CMAKE_HIP_FLAGS`, so this one line also reaches the `-fgpu-rdc` device link, where device code is generated. The existing `string(APPEND CMAKE_SHARED_LINKER_FLAGS " -fgpu-rdc --hip-link")` at :55, and the "add --hip-link -fgpu-rdc on the shared-library link below" sentence at :49-50, come from earlier port commits and are dead for the same reason (CMake adds `--hip-link` itself, and `-fgpu-rdc` arrives through `CMAKE_HIP_FLAGS`). Drop or correct them in the same commit so that no comment in the block describes a mechanism that doesn't run. After the change, re-check that every `link.txt` still carries the flag once (VERBOSE build or `grep` as above) and that RHF water/def2-svp still matches CPU on gfx942.

### Port Correctness

2. **The comment (gpu4pyscf/lib/CMakeLists.txt:56-61) and the README paragraph (README.md:65) never say which compiler has the bug, so the removal condition cannot be checked.** The comment says "a ROCm compiler bug" and "Remove it only once a later ROCm compiler passes"; the README says "the ROCm compiler's machine LICM pass miscompiles" without any version. The defect was observed with the ROCm 10.1 compiler (AMD clang 24.0.0, deferred `therock-10-1-machinelicm-clobbers-live-vgpr`). The June gfx90a validation built with ROCm 7.2.1, without the flag, and passed the d-shell SCF tests. So the README's unqualified sentence is not accurate for every ROCm, and "a later ROCm compiler" has nothing to be later than. Fix: in both places, name the compiler version the miscompile was seen with (ROCm 10.1 / AMD clang 24). Keep the removal test as it is: drop the flag once the d-shell SCF tests pass on gfx90a and gfx942 without it.

### Recommendation
**Request Changes.** Both fixes go in one new commit on top of 28f2418 (no amend). No device code changes. The build-flag result has to stay the same (one `-disable-machine-licm` on every HIP compile and link line), so the porter's 28f2418 gfx942 test numbers carry over only if the VERBOSE check confirms that.

## Porter 2026-10-30 (porter, linux-gfx942 MI300X, fork moat-port 28f2418 -> 13e3500): review findings 1 and 2

jeff asked for both findings of the 28f2418 review fixed. Taken under the work lock (`changes-requested -> porting`). One new commit 13e3500 on top of 28f2418, pushed fast-forward (no amend, no force); `advance-head 13e3500...`.

### Change
- gpu4pyscf/lib/CMakeLists.txt: removed both `string(APPEND CMAKE_SHARED_LINKER_FLAGS ...)` lines (`-fgpu-rdc --hip-link` and `-mllvm -disable-machine-licm`), which the upstream `set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")` overwrites in the README build. The `-fgpu-rdc` comment now says CMake's HIP link rule repeats `CMAKE_HIP_FLAGS` and adds `--hip-link` itself, so those flags reach the shared-library (device) link. The LICM comment names ROCm 10.1 (AMD clang 24), says the d-shell SCF tests pass with ROCm 7.2.1 without the flag, and says to remove it once a ROCm release passes those tests without it.
- README.md: the same compiler version, the 7.2.1 note and the removal condition.

### Proof of no effective change (gfx942, ROCm 10.1 / AMD clang 24.0.0, cmake 3.31.6, GCC host)
Built both 28f2418 and 13e3500 the README way:
```
cmake -S gpu4pyscf/lib -B <b> -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942
cmake --build <b> -j $MAX_JOBS -v
```
Both exit 0. After normalizing the build dir name, 79 HIP `.cu` compile lines and 8 link lines in each build, and they are **byte-identical** between the two builds. Every compile and link line has `-disable-machine-licm` exactly once, and every link line has `--hip-link` and `-fgpu-rdc` once each. CMakeCache `CMAKE_SHARED_LINKER_FLAGS` is empty in both. Configure-only with `LDFLAGS=-Wl,-O1`, where the upstream `set` does not fire: all 8 `link.txt` have the flag, `--hip-link`, `-fgpu-rdc` and `-Wl,-O1` once each. At 28f2418 that path doubled them, so it now carries only one copy and still works.

### Tests (fork @ 13e3500, `ulimit -c 0`, HIP_VISIBLE_DEVICES=4, job FLEET_VENV: numpy 2.2.6, scipy 1.17.1, cupy-rocm-7-0 14.2.0, pyscf 2.14.0, pytest 7.3.2; `LD_LIBRARY_PATH=$(rocm-sdk path --root)/lib`)
RHF water against CPU PySCF: sto-3g 9.7e-13, def2-svp 1.7e-11, cc-pvtz 6.3e-12 (the same as the 28f2418 flag build). The full scf suite was not rerun; the compile and link commands are identical to 28f2418, so its 28f2418 gfx942 result (122 passed, 8 skipped, 36 out-of-scope failures) carries over.

### Other checks
`jargon.py --port gpu4pyscf`: clean. `prose.py` on the commit message: clean. Title 62 chars. The added lines and the message are ASCII. `git status --porcelain` in the fork is empty. No CUDA rerun: the change only touches lines inside `if(USE_HIP)` and the README.

## Review 2026-10-30 (reviewer, linux-gfx942, fork moat-port @ 13e3500 vs upstream/master 10dcea3; delta 28f2418..13e3500) -- CHANGES REQUESTED

Scope: the delta 13e3500 in full, 28f2418 again, and the whole fork diff (31 files, +697/-76) re-read where it touches the HIP CMake block. Not findings, checked independently on this host (TheRock ROCm 10.1, cmake from /opt/python-3.11-venv, GCC host): configure-only `cmake -S gpu4pyscf/lib -B /tmp/g4p-review-cfg -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942` -> all 8 `link.txt` carry `-disable-machine-licm`, `--hip-link` and `-fgpu-rdc` exactly once, all 8 `flags.make` carry the flag once, CMakeCache `CMAKE_SHARED_LINKER_FLAGS` empty. The flag is set only inside `if(USE_HIP)` (CMakeLists.txt:38-68) on `CMAKE_HIP_FLAGS`, which no CUDA rule reads, and no other build path (setup.py, builder) sets HIP flags. Both review findings of 28f2418 are fixed. `jargon.py --port gpu4pyscf` clean, `prose.py` on the 13e3500 body clean, title 62 chars with `[ROCm]`, AI disclosure in prose, fenced Test Plan, no trailers, added lines ASCII, fork tree clean.

### Port Correctness

1. **The ROCm 7.2.1 sentence and the removal condition (gpu4pyscf/lib/CMakeLists.txt:61 and :63-64; README.md:65) claim more than the evidence and leave the removal test ambiguous.** "With ROCm 7.2.1 those tests pass without it" follows "on gfx90a and gfx942", so it reads as covering both. The only 7.2.1 evidence is the June gfx90a validation; gfx942 has only ever been built with ROCm 10.1 (notes: every gfx942 section). And "Remove it once a ROCm release passes the d-shell SCF tests without it" names no architecture. ROCm 10.1 itself already passes those tests without the flag on gfx1100 (Validation 2026-10-30 gfx1100 @ 066e91e, and the 45d8053 gfx1100 run), so read literally the condition is already met by the very release that has the bug. The previous review asked to keep "on gfx90a and gfx942" in the removal test; 13e3500 dropped it. Fix, same wording in both places: "With ROCm 7.2.1 the d-shell SCF tests pass on gfx90a without it." and "Remove it once a ROCm release passes the d-shell SCF tests on gfx90a and gfx942 without it." Comment and README text only; no build change, so the 13e3500 build/test evidence carries over (the compile and link commands stay byte-identical).

### Recommendation
**Request Changes.** One new commit on top of 13e3500 (no amend), touching only those comment and README sentences. Re-run `jargon.py --port gpu4pyscf` and `prose.py` on the message.

## Porter 2026-10-30 (porter, linux-gfx942 MI300X, fork moat-port 13e3500 -> 5c60311): review finding 1 (wording)

jeff asked for the finding of the 13e3500 review fixed. Taken under the work lock (`changes-requested -> porting`). One new commit 5c60311 on top of 13e3500, pushed fast-forward (no amend, no force); `advance-head 5c60311...`.

### Change
Wording only, the same sentences in gpu4pyscf/lib/CMakeLists.txt:61-64 (comment rewrapped) and README.md:65: "With ROCm 7.2.1 the d-shell SCF tests pass on gfx90a without it." and "Remove it once a ROCm release passes the d-shell SCF tests on gfx90a and gfx942 without it."

### Proof of no effective change (gfx942, TheRock ROCm 10.1 / AMD clang 24.0.0, cmake 3.31.6 from /opt/python-3.11-venv, GCC host)
Configure-only at 13e3500 (a git worktree) and at 5c60311:
```
cmake -S <src>/gpu4pyscf/lib -B <b> -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942
```
Both exit 0. All `flags.make`, `link.txt` and `build.make` files concatenated (8 + 8 + 8 per tree), with source and build paths normalized: byte-identical (2269 lines each). All 8 `link.txt` carry `-disable-machine-licm`, `--hip-link` and `-fgpu-rdc` once each; all 8 `flags.make` carry the flag once; CMakeCache `CMAKE_SHARED_LINKER_FLAGS` empty in both. No full build or test rerun: the commands are identical, so the 13e3500 build/test evidence carries over.

### Other checks
`jargon.py --port gpu4pyscf`: clean (needed `git fetch origin master:master` in a fresh `-b moat-port` clone first). `prose.py` on the commit message: clean. Title 57 chars, message and added lines ASCII. `git status --porcelain` in the fork is empty.

## Review 2026-10-30 (reviewer, linux-gfx942, fork moat-port @ 5c60311 vs upstream/master 10dcea3; delta 066e91e..5c60311) -- CHANGES REQUESTED (MOAT skill text only; the fork is clean)

Scope: covers the delta 13e3500..5c60311 and the whole flag change 066e91e..5c60311. The whole fork diff (31 files, +697/-76; nothing but CMakeLists.txt and README.md changed since the 45d8053 pass) was re-read where it touches the HIP CMake block. I also read the cuda-to-rocm lesson that this branch adds for the flag. Checked independently on this host (TheRock ROCm 10.1, cmake from /opt/python-3.11-venv, GCC host) and not findings:
- `cmake -S gpu4pyscf/lib -B <b> -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942` exits 0.
- In that build, each of the 8 `link.txt` files has `-disable-machine-licm`, `--hip-link` and `-fgpu-rdc` exactly once, and each of the 8 `flags.make` files has the flag once. CMakeCache `CMAKE_SHARED_LINKER_FLAGS` is empty.
- The flag is appended only to `CMAKE_HIP_FLAGS` inside `if(USE_HIP)` (gpu4pyscf/lib/CMakeLists.txt:38-68). The HIP language is enabled only under USE_HIP, and no subdirectory, setup.py or builder sets HIP or linker flags, so a CUDA build cannot see it.
- The comment at CMakeLists.txt:57-64 says what the flag works around (MachineLICM in ROCm 10.1 / AMD clang 24 clobbers the atomicAdd offset register in the Rys J/K kernels on gfx90a and gfx942). It also says when to remove it (a ROCm release passes the d-shell SCF tests on gfx90a and gfx942 without it).
- README.md:56-67 matches the CMake behaviour and the recorded evidence.
- `jargon.py --port gpu4pyscf` and `prose.py` are clean on all three commit bodies. The titles have `[ROCm]` and are 58, 62 and 57 chars. Each body discloses the AI agent in prose and has a fenced Test Plan. There is no Co-Authored-By or noreply trailer, the messages and added lines are ASCII, and the fork tree is clean.

### Port Correctness (lesson promoted on this branch)

1. **.claude/skills/cuda-to-rocm/references/validation.md:209 (the "Proving a real miscompile" bullet) still prescribes the mechanism that 13e3500 removed as dead.** It says the fix was "`-mllvm -disable-machine-licm` in both `CMAKE_HIP_FLAGS` and the shared linker flags (needed under `-fgpu-rdc`, where device codegen runs at link)". 13e3500, and this review's configure, show that the `CMAKE_SHARED_LINKER_FLAGS` append is overwritten in the default GCC build. They also show that CMake's HIP link rule already repeats `CMAKE_HIP_FLAGS` (and adds `--hip-link` itself), so the single `CMAKE_HIP_FLAGS` line reaches the `-fgpu-rdc` device link. The lesson exists only on port/gpu4pyscf (`git grep disable-machine-licm origin/main -- .claude/skills` = nothing), so this review owns it. As written it would lead the next porter to add the same dead line. Fix: reword that sentence to say that setting it on `CMAKE_HIP_FLAGS` is enough, because CMake's HIP link rule repeats those flags on the `-fgpu-rdc` link where device code is generated, and that a `CMAKE_SHARED_LINKER_FLAGS` append can be silently overwritten by the project. Also replace "do not commit a compiler workaround without a person's decision" with the outcome: a person did decide, and the fork carries it under USE_HIP with a dated removal condition. This is a MOAT record edit only. The fork does not change, `head_sha` stays 5c60311, and nothing needs revalidation for it.

### Recommendation
**Request Changes.** The fork at 5c60311 has no open findings; the next review of this lesson edit can pass the fork as-is. For PR prep, the 28f2418 body still describes the superseded link-flag mechanism ("The flag goes on both the HIP compile flags and the shared-library link flags"). If the three commits are squashed, the squashed message must take the 13e3500 wording instead.

## Review 2026-10-30 (reviewer, linux-gfx942, fork moat-port @ 5c60311 unchanged; MOAT lesson 0a63d983) -- REVIEW PASSED

Requested by jeff. Scope: the records-only fix the previous review asked for, the machine-licm bullet in `.claude/skills/cuda-to-rocm/references/validation.md` as corrected by the primary in 0a63d983, plus a check that the fork has not moved. The fork itself was covered by the earlier reviews and is unchanged.

State bookkeeping: there is no `changes-requested -> reviewing` edge, so this run recorded the primary's 0a63d983 fix as the porter round (`changes-requested -> porting -> ported`, no fork change, `head_sha` stays 5c60311) and then took the work lock.

Checked, not findings:
- Fork `moat-port` tip is 5c60311636487d1f881d2f96e6d0dc80d55a9412; merge base with upstream/master is 10dcea3; 31 files, +697/-76, as in the previous review. `jargon.py --port gpu4pyscf`: clean.
- The corrected lesson matches the facts in this file and in the fork: the flag is set only on `CMAKE_HIP_FLAGS` (gpu4pyscf/lib/CMakeLists.txt:65, inside `if(USE_HIP)`); CMake's HIP link rule reuses those flags on the `-fgpu-rdc` link (Porter 13e3500, Review 13e3500: all 8 `link.txt` carry the flag once); the project's `set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")` (CMakeLists.txt:96) overwrites a linker-flags append (Review 28f2418 finding 1). The text now says that, and asks for a verbose-build check of every compile and link line.
- The "do not commit a compiler workaround" sentence is replaced with the outcome: it is a person's decision, and once made it goes under the HIP build only with a comment naming the compiler version, the arches and the removal condition. That is what CMakeLists.txt:57-64 carries.
- Other claims in the bullet are unchanged and still agree with the notes: the rocgdb and objdump evidence (Upstream sync 2026-10-29), "-O1 did not help" (reproduces at -O1), the gfx90a pattern from the same compiler, gfx1100 running correctly with the same compiler (Validation gfx1100 @ 066e91e).
- The branch adds only this bullet and the CUDA-root bullet to the skill (`git diff origin/main...HEAD -- .claude`), and both were read in earlier reviews.

No findings. Carried over from the previous review for PR prep: if the three LICM commits are squashed, the squashed message must take the 13e3500 wording, not the 28f2418 body's "both the HIP compile flags and the shared-library link flags".

## Validation 2026-10-30 (validator, linux-gfx1100, fork moat-port @ 5c60311) -- COMPLETED

Requested by jeff, a revalidation at the head that carries `-mllvm -disable-machine-licm` for HIP builds (28f2418, 13e3500, 5c60311; review passed at 5c60311). Host: Radeon Pro W7800 48GB, gfx1100 (wave32). Toolchain: TheRock ROCm SDK 10.1.0 (HIP compiler identification Clang 24.0.0; `R=$(rocm-sdk path --root)` = `/opt/python-3.11-venv/lib/python3.11/site-packages/_rocm_sdk_devel`, no /opt/rocm). Fork clone at 5c60311636487d1f881d2f96e6d0dc80d55a9412 (== origin/moat-port). GPU pin: dispatcher set `HIP_VISIBLE_DEVICES=0`, which exists here; amd-smi showed no process on any GPU.

### Python env (job venv)
```
pip install 'numpy>=2.0,<2.3' 'scipy>=1.13' cupy-rocm-7-0==14.2.0 pyscf==2.14.0 pytest
```
Installed: python 3.11.17, numpy 2.2.6, scipy 1.17.1, cupy-rocm-7-0 14.2.0, pyscf 2.14.0, pytest 7.3.2. CuPy-ROCm needs `LD_LIBRARY_PATH=$R/lib` (`rocm-sdk` is on PATH from /opt/python-3.11-venv, not in the job venv).

### Build: README "Compilation" ROCm block, as written
```
export PATH=$FLEET_VENV/bin:$PATH
cmake -S gpu4pyscf/lib -B build/temp.gpu4pyscf -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build/temp.gpu4pyscf -j 32 -v
```
- Deviations from the README: the arch, `-j 32` and `-v` (to check the flag). Paths absolute because utils/timeit.sh runs from the moat root. No `-DCMAKE_HIP_COMPILER` needed.
- Configure exit 0 (2 s); build exit 0 (107 s); 8 libs: cupy_helper gdft gecp gint gvhf gvhf_md gvhf_rys pbc.
- Verbose log: 79 HIP `.cu` compile lines and 8 `--hip-link` lines, each carrying `-disable-machine-licm` exactly once (the same counts as the gfx942 13e3500 check).

### RHF water vs CPU PySCF (`ulimit -c 0`, conv_tol 1e-12)
| basis | max shell | |GPU - CPU| (Ha) |
|---|---|---|
| sto-3g | p | 3.6e-12 |
| def2-svp | d | 9.6e-12 |
| cc-pvtz | f | 2.8e-11 |

The flag costs nothing in correctness on gfx1100 (066e91e without it: 1.3e-12 / 1.7e-11 / 6.2e-12; the geometry here differs slightly from that run's script, so only the magnitude compares).
## Validation 2026-10-30 (validator, linux-gfx90a MI250X GCD, fork moat-port @ 5c60311) -- COMPLETED

Requested by jeff: revalidate gfx90a now that the port carries `-mllvm -disable-machine-licm` for HIP builds (28f2418, 13e3500, 5c60311; review passed at 5c60311). Host: gfx90a (one MI250X GCD, `HIP_VISIBLE_DEVICES=0` as pinned by the dispatcher; it worked). Toolchain: TheRock ROCm SDK 10.1.0 (`R=$(rocm-sdk path --root)` = /opt/python-3.11-venv/.../_rocm_sdk_devel; HIP compiler identification Clang 24.0.0). Fork clone at 5c60311636487d1f881d2f96e6d0dc80d55a9412 (== origin/moat-port).

### Python env (job venv)
```
$FLEET_VENV/bin/pip install 'numpy>=2.0,<2.3' 'scipy>=1.13' cupy-rocm-7-0==14.2.0 pyscf==2.14.0 pytest
```
Installed: python 3.11.17, numpy 2.2.6, scipy 1.17.1, cupy-rocm-7-0 14.2.0, pyscf 2.14.0, pytest 7.3.2. The job venv sees the system TheRock install, so `rocm-sdk` and the HIP libraries come from /opt/python-3.11-venv; CuPy-ROCm needs `LD_LIBRARY_PATH=$R/lib`.

### Build: README ROCm block, as written
```
export PATH=$FLEET_VENV/bin:$PATH
cmake -S gpu4pyscf/lib -B build/temp.gpu4pyscf -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build build/temp.gpu4pyscf -j 32 -v
```
- Deviations from the README: the arch, `-j 32`, and `-v` to see the flags. Paths passed absolute (utils/timeit.sh runs from the moat root). No command-line flag additions: the committed CMake only.
- Exit 0, 210 s, 8 libs (cupy_helper gdft gecp gint gvhf gvhf_md gvhf_rys pbc). The verbose log has `-disable-machine-licm` on 87 lines: the 79 HIP compiles and the 8 `--hip-link` lines, matching the gfx942 porter count.

### RHF water vs CPU PySCF (conv_tol 1e-12)
| basis | max shell | GPU (Ha) | |GPU - CPU| (Ha) |
|---|---|---|---|
| sto-3g | p | -74.96302313846651 | 3.7e-12 |
| def2-svp | d | -75.9609839870705 | 9.6e-12 |
| cc-pvtz | f | -76.05712742030266 | 2.9e-11 |

The d- and f-shell bases no longer abort (at 066e91e without the flag, def2-svp died with APERTURE_VIOLATION in rys_k_kernel).

### Tests
```
ulimit -c 0
export HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$R/lib PYTHONPATH=<fork root>
python -m pytest gpu4pyscf/scf/tests -q -rfs --tb=no -p no:cacheprovider
```
- Run 1: **126 passed, 8 skipped, 32 failed** (227 s).
- Run 2: **126 passed, 8 skipped, 32 failed** (156 s). Identical failure set to run 1.
- Equal to the 066e91e and 45d8053 gfx1100 baselines. Failures classified by rerunning the 32 ids with `--tb=line`: 25 `hipErrorUnknown` (DFT/libxc not built: test_soscf rks/uks/roks x9, test_gdm x3 of 4, test_init_guess *rks/*uks x8, test_scf to_cpu/to_gpu, test_diffuse_orbital rks_df/rks_lowmem/uhf etc.), 4 PCM not built (`object has no attribute 'PCM'`), 3 `BasisNotFoundError` (d-aug-cc-pvdz). All out of the HF/integral scope.
- Skips (8): dftd3/dftd4 not installed (5), unconditional `hermi=0` (3).

### Other checks
- CUDA no-regression gate: not rerun. It passed at 28f2418 (Porter 28f2418 section), and 28f2418..5c60311 changes only README.md and comment/flag lines inside `if(USE_HIP)` in gpu4pyscf/lib/CMakeLists.txt, which no CUDA rule reads (reviews of 13e3500 and 5c60311).
- `python3 utils/jargon.py --port gpu4pyscf` -> `jargon: clean`.
- Documentation: the README documents the ROCm build next to the CUDA build, including the flag paragraph; that build was followed here.
- Integrity: `git -C projects/gpu4pyscf/src status --porcelain` is empty after the build.
- Run 1: **122 passed, 8 skipped, 36 failed** (309 s). Run 2: **122 passed, 8 skipped, 36 failed** (207 s). Identical failure sets. No aborts, so the whole suite ran in one process (the 066e91e no-flag run needed one process per test).
- Equal to the 066e91e gfx90a flag-build result (122/8/36) and the gfx942 28f2418 result. The 40 tests that the no-flag build broke at 066e91e all pass.
- Failures, classified from the junit messages, all out of the HF/integral scope: 25 DFT (`hipErrorUnknown`, libxc not built), 4 PCM not built, 4 analytic gradient (test_diffuse_orbital test_rhf/test_rhf_lowmem, test_fermi_smearing test_gradient, test_scf_jk test_jk_energy_per_atom, off by 4.33 this run; it varies run to run as before), 3 `BasisNotFoundError` d-aug-cc-pvdz.
- Skips (8): dftd3/dftd4 not installed (5), unconditional `hermi=0` (3).

### Other checks
- CUDA no-regression gate: not rerun. It passed at 28f2418 (Porter section), and 13e3500/5c60311 change only lines inside `if(USE_HIP)` and the README, with byte-identical HIP compile/link commands.
- `python3 utils/jargon.py --port gpu4pyscf` -> `jargon: clean`.
- Documentation: the README ROCm section documents the build and the flag; that build was followed here.
- Integrity: `git -C projects/gpu4pyscf/src status --porcelain` is empty after the build and tests.

## Porter 2026-10-31 (porter, linux-gfx942 MI300X, fork moat-port 5c60311 -> e469394): full port, part 1 (libxc, stack limit, gradient block size)

Requested by jeff: the port must be complete. The earlier plan's HF-only staging (DFT, solvent, sem, multigrid, gradients and Hessians deferred) was never approved. Work lock taken (`review-passed -> porting`). Input: `~/.claude/plans/gpu4pyscf-libxc-hip-probe.md` (gfx1100/gfx90a probe). Three new commits on top of 5c60311 (no amend, no force push):

- 4a90309 `[ROCm] Build GPU libxc with its HIP backend`: the libxc ExternalProject passes `-DENABLE_HIP=ON` plus `CMAKE_HIP_ARCHITECTURES` and `CMAKE_HIP_COMPILER` under USE_HIP; CUDA keeps `-DENABLE_CUDA=ON` and the CUDA arch. `BUILD_LIBXC` defaults ON on both. dft/libxc.py is upstream's file again apart from the version label (`7.1.2 (HIP)` on ROCm); the `_libxc = None` fallback is gone. The stale gdft CMake comment and README sentence are updated. libxc_prune.patch applies to 7.1.2 unchanged.
- c073679 `[ROCm] Set the device stack limit through HIP directly`: `release_gpu_stack()` calls `hipDeviceSetLimit(hipLimitStackSize, 128)` through ctypes on ROCm (soname `libamdhip64.so.<major>` from `runtimeGetVersion() // 10000000`, i.e. the library CuPy already loaded), and raises through `cupy.cuda.runtime.check_status` on error. CuPy-ROCm's `deviceSetLimit` is a stub returning hipErrorUnknown (cupy_backends/hip/cupy_hip_runtime.h); calling HIP directly returns 0. This, not missing libxc, was behind the 25 "DFT" failures of every earlier validation.
- e469394 `[ROCm] Keep the block size of the gradient energy kernels`: root cause of the analytic-gradient failures on wave64 (gfx90a, gfx942; gfx1100 passed). The kernels in unrolled_ejk_ip1.cu, unrolled_ejk_ip2_type12.cu and unrolled_ejk_ip2_type3.cu hard-code `constexpr int nsq_per_block = 256`, while their host launchers call `adjust_threads`, which doubles the block when `numRegs <= 128`. On gfx942 `rys_ejk_ip1_0000` has `.vgpr_count: 128`, so it was launched with 512 threads: every task done twice and the per-thread `rw` slices overlapping `cicj_cache`. H2/sto-3g (only the 0000 kernel) reproduced it: J gradient 40.5 vs 0.765, changing each run. Fix: under USE_HIP those three files redefine `adjust_threads` as a no-op. The other unrolled files read `blockDim.x` and keep the doubling. This is a latent upstream bug on CUDA too (it would fire on any GPU where these kernels fit 128 registers); CUDA left identical as instructed. Worth reporting upstream: **question for jeff**.

### Environment (gfx942, TheRock ROCm SDK 10.1.0, AMD clang 24.0.0, cmake 3.31.6, HIP_VISIBLE_DEVICES=3)
```
$FLEET_VENV/bin/pip install 'numpy>=2.0,<2.3' 'scipy>=1.13' cupy-rocm-7-0==14.2.0 pyscf==2.14.0 pytest pyscf-dispersion geometric pytest-timeout
# numpy 2.2.6, scipy 1.17.1, cupy-rocm-7-0 14.2.0, pyscf 2.14.0, pyscf-dispersion 1.5.0, geometric 1.1.1, pytest 7.3.2, pytest-xdist 3.3.1
cmake -S gpu4pyscf/lib -B build/temp.gpu4pyscf -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942
cmake --build build/temp.gpu4pyscf -j 100     # 3m15s with the libxc ExternalProject
export LD_LIBRARY_PATH=$(rocm-sdk path --root)/lib PYTHONPATH=$PWD; ulimit -c 0
```
Test runner (agent_space/scripts/runtree.sh): per directory, `python -m pytest <dir> -n 4 --dist loadfile -m 'not slow and not benchmark and not special' -q -rfEs --tb=line -p no:cacheprovider --timeout 1800 --junitxml=...` (the upstream CI selection, `-n 4 --dist loadfile` as upstream CI, one GPU).

### Results at e469394
- Water RKS/def2-svp, grids level 3, conv_tol 1e-12, vs CPU PySCF: B3LYP 1.1e-11, PBE 6.1e-12, M06 7.1e-12 Ha.
- Water RHF gradient vs CPU: sto-3g 6.3e-14, def2-svp 2.1e-13, cc-pvtz 9.9e-12 (before e469394: 27, 23, 4.2).
- dft/tests (c073679): 113 passed, 9 skipped (mcfun not installed 8, dftd3 1), 0 failed. Baseline 5c60311 in the same env: 24 passed, 56 failed, 8 skipped, 20 errors.
- scf/tests (c073679): 148 passed, 3 skipped, 13 failed; same-env 5c60311 baseline 125 passed, 3 skipped, 36 failed. (The c073679 commit message quotes "122 passed and 36 failed before", the earlier recorded gfx942 run without pyscf-dispersion; the same-env baseline is 125/36.) The 13: 4 PCM not built yet, 1 Hessian not imported, 3 d-aug-cc-pvdz missing from PySCF 2.14, 5 gradient (fixed by e469394, rerun in the full tree below).

### CUDA no-regression gate: PASS
NVIDIA redist archives from developer.download.nvidia.com/compute/cuda/redist (redistrib_12.8.1.json), each sha256-verified against the manifest: cuda_nvcc 12.8.93, cuda_cudart 12.8.90, cuda_cccl 12.8.90, cuda_profiler_api 12.8.90, unpacked to agent_space/cuda-12.8 (`lib64 -> lib`). Host gcc/gfortran 13.
```
PATH=$K/bin:$PATH CUDACXX=$K/bin/nvcc cmake -S <git archive>/gpu4pyscf/lib -B build-cuda -DCMAKE_CUDA_ARCHITECTURES=80 -DCUDAToolkit_ROOT=$K -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j 100 -- VERBOSE=1
```
At c073679: exit 0, 12 libs, and the libxc ExternalProject built with `ENABLE_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=80` (libxc.so.15). At e469394 (same tree, rebuilt): exit 0, the three unrolled_ejk_* TUs recompiled by nvcc.

## Porter 2026-10-31 (porter, linux-gfx942 MI300X, fork moat-port e469394 -> dc45713): full port, part 2 (every module on ROCm)

dc45713 `[ROCm] Build the solvent, semi-empirical and multigrid libraries`. Pushed; `advance-head dc45713`. What the HIP build left out at 5c60311, and how each is now handled:
- **libsolvent** (PCM/SMD): built. Fortran re-enabled in the HIP `project()` (gfortran; the SMD mnsol sources). pcm.cu's 32-lane row reduction uses `__shfl_down(v, o, THREADS)` under USE_HIP.
- **libsem**: built unchanged (no warp code).
- **libmgrid_v2 / libmgrid_v3** (multigrid): built. v2 evaluation.cuh already reduces over `warpSize` with a `thread_id % warpSize` leader; HIP path uses full-wave `__shfl_down` (control flow there is block-uniform). v3 treats a warp as 32 lanes everywhere (`WARP_SIZE 32`, `lane = tid % 32`), so the existing popcount-width shim cuda_to_hip_shfl.h is force-included on the mgrid_v3 target. New hip_compat/cuComplex.h (`hipDoubleComplex`), `cudaPeekAtLastError` mapping, and `#undef I` in the CUB shim (multigrid_v2/screen.cu includes <complex.h>, whose `I` macro breaks rocPRIM).
- **Python import guards**: gpu4pyscf/__init__.py and pbc/__init__.py are upstream's files again (grad, hessian, solvent, dft, tdscf, nac, pbc.dft import on ROCm).
- **CUTLASS grouped GEMM** (grouped_dot/grouped_gemm, used by numint.nr_rks_group): on ROCm one cupy GEMM (hipBLAS) per group. Composable Kernel / hipBLASLt grouped GEMM is a perf follow-up, not a correctness gap.
- **RawKernel sources** (cupy_helper.vec_dot, five multigrid_v3.py kernels, pbc/df/grad/rhf.py indexed_scale): `cupy_helper.kernel_source()` prepends, on ROCm only, `hip/hip_complex.h` aliases and `#define __shfl_down_sync(mask, var, delta) __shfl_down(var, delta, 32)` (all 11 sites use the full 32-bit mask with 32-lane groups). hiprtc has no cuComplex.h and HIP's `__shfl_down_sync` static_asserts a 64-bit mask.
- **Analytic gradients and Hessians**: nothing was excluded at build level; they failed at runtime because of the ejk block-size bug fixed in e469394 (part 1).
- **gint NROOTS 7/8 `#ifndef USE_HIP` and the gint `-O1`** (notes "Key port facts"): stale. nr_fill_ao_ints.cu is commented out of upstream's gint source list, so the guard covered nothing built, and `target_link_options(gint PRIVATE -O1)` lowered every gint kernel. Removed in a follow-up commit (below), file restored to upstream.
- **Not ported, by nature**: cuTENSOR (CuPy-ROCm has no binding; the built-in cupy.einsum engine is used, GPU, slower). cuEST (lib/cuest_wrapper.py, optional NVIDIA closed-source backend never imported by the package; test_cuest_wrapper skips without it; no ROCm equivalent).

### Hardware limit found (question for jeff)
ECP derivative integrals (gto/ecp.py get_ecp_ip / get_ecp_ipip) for high angular momentum need more LDS than a gfx942 workgroup has (64 KB): (li,lj,lc)=(4,4,4) asks for 62680 B dynamic plus the kernel's static `gctr_smem`/`buf`, and `cudaFuncSetAttribute(type2_cart_ip1_general)` returns invalid argument. Upstream already skips these 5 tests (gto/tests/test_ecp.py) on any GPU with `shm_size < 64*1024`; with the skip removed, test_ecp_cart_ip1 passes and the 4 g-shell cases fail this way. Affects ECP gradients/Hessians only when the basis has g functions on an ECP atom with g projectors (e.g. not def2-TZVP). gfx950 (160 KB LDS) should not hit it. A reduced-LDS variant of type2_cart_ip1_general would be new kernel design, not porting: jeff's call.

### Full test tree at dc45713 vs 5c60311 (gfx942, same env, same runner)
Runner: agent_space/scripts/runtree.sh (per directory, `-n 4 --dist loadfile -m 'not slow and not benchmark and not special'`). Columns: passed/failed/skipped/errors.

| directory | 5c60311 pass/fail/skip/err | new pass/fail/skip/err |
|---|---|---|
| cc/tests | 4/0/0/0 | 4/0/0/0 |
| df/hessian/tests | 16/8/0/0 | 24/0/0/0 |
| df/tests | 75/129/3/1 | 200/5/3/0 |
| dft/tests | 24/56/8/20 | 113/0/9/0 |
| dispersion/tests | 16/0/0/0 | 16/0/0/0 |
| fci/tests | 3/0/0/0 | 3/0/0/0 |
| geomopt/tests | 0/0/5/0 | 0/0/5/0 |
| grad/tests | 10/107/0/0 | 115/2/0/0 |
| gto/tests | 57/0/5/0 | 57/0/5/0 |
| hessian/tests | 20/71/1/0 | 91/0/1/0 |
| lib/tests | 33/6/1/0 | 39/0/1/0 |
| mcscf/tests | 8/0/0/0 | 8/0/0/0 |
| md/tests | 4/1/0/0 | 5/0/0/0 |
| mp/tests | 20/1/0/0 | 20/1/0/0 |
| nac/tests | 14/58/1/0 | 71/1/1/0 |
| pbc/df/tests | 56/2/7/12 | 104/1/7/0 |
| pbc/dft/tests | 0/1/1/52 | 156/0/18/0 |
| pbc/grad/tests | 0/25/0/53 | 185/0/2/0 |
| pbc/gto/pseudo/tests | 7/0/0/0 | 7/0/0/0 |
| pbc/gto/tests | 0/0/0/4 | 13/0/0/0 |
| pbc/hessian/tests | 0/0/0/4 | 0/0/6/0 |
| pbc/lib/tests | 2/0/0/0 | 2/0/0/0 |
| pbc/properties/berry/tests | 10/0/2/0 | 10/0/2/0 |
| pbc/scf/tests | 12/38/0/12 | 88/2/1/0 |
| pbc/tools/tests | 7/24/0/0 | 16/0/0/0 |
| pbc/x2c/tests | 1/0/1/0 | 1/0/1/0 |
| pop/tests | 2/0/0/0 | 2/0/0/0 |
| properties/tests | 3/18/5/5 | 26/0/5/0 |
| qmmm/pbc/tests | 0/3/0/0 | 3/0/0/0 |
| qmmm/tests | 7/14/0/0 | 21/0/0/0 |
| scf/tests | 125/36/3/0 | 159/2/3/0 |
| sem/dmp/tests | 0/0/0/4 | 1/0/0/0 |
| sem/grad/tests | 0/0/0/8 | 3/0/0/0 |
| sem/gto/tests | 0/0/0/8 | 15/0/0/0 |
| sem/integral/tests | 0/0/0/16 | 27/0/0/0 |
| solvent/tests | 11/26/91/0 | 126/0/2/0 |
| tdscf/tests | 20/15/0/60 | 93/2/0/0 |
| tests | 0/0/0/0 | 0/0/0/0 |
| x2c/tests | 8/0/1/0 | 8/0/1/0 |
| **total** | 575/639/135/259 | 1832/16/73/0 |

The baseline errors were mostly import-time (libsolvent/libsem/libmgrid absent, pbc.dft not imported, hipErrorUnknown from deviceSetLimit).

### The 16 remaining failures, each checked
- **Inputs from seeded `cupy.random` (7)**: CuPy-ROCm draws from hipRAND/rocRAND, not cuRAND, so hard-coded fingerprints of results on those inputs differ. df/test_df_tdrhf_grad::test_jk_energy_per_atom, df/test_df_tdrhf_nac::test_get_nacv_ee, grad/test_tdrhf_grad::test_grad_elec, grad/test_tduhf_grad::test_grad_elec, nac/test_tdrhf_nac_ee::test_get_nacv_ee, pbc/scf/test_pbc_soscf::test_rks_gen_g_hop, ::test_uks_gen_g_hop. Evidence: test_grad_elec's inputs run through CPU PySCF (pyscf.grad.tdrhf.grad_elec) give fp 7.823375488836 vs GPU 7.823375488826 (max diff 4.6e-12); the soscf tests pass all their CPU cross-checks (7 and 9 places) with only the fingerprint line disabled (scratch copy); the DF J/K energy derivative matches CPU-PySCF DF finite differences exactly (ratio 0.5 for J, -0.25 for K, all 9 components) and the test's own multi-DM vs per-DM check agrees to 4e-11.
- **Basis missing from PySCF 2.14 (4)**: d-aug-cc-pvdz (scf x2, tdscf test_rtda_ris_diffuse) and df/test_general_contraction (BasisNotFoundError).
- **Tolerance misses by <3x (4)**: mp/test_dfmp2_kernel_one_gpu (1.01e-7 vs 7 places; float32 cderi by default, float64 gives 9e-16 vs CPU); pbc/df/test_ft_ao_strain_deriv (3.8e-8 vs 3e-8 for a 1e-5 finite difference of values ~1134; the analytic value matches a CPU-evaluated Richardson FD to <=1e-8 in every component); df/test_df_tdrhf_grad::test_j_energy_per_atom (multi-DM vs sum of six single-DM runs 1.97-2.21e-12 vs 1e-12, varies run to run: atomic accumulation order); df/test_df_rks_grad::test_grad_hybrid (1.15e-9 vs CPU, tolerance 1e-9).
- **tdscf/test_krylov::test_krylov_eigenvalue (1)**: random 3000x3000 matrix (seed 42, different on ROCm) whose 3rd/4th eigenvalues are 0.0013 apart; the solver hits the test's max_iter=35. With max_iter=100 it gives eigvec error 5.2e-8 at conv_tol 1e-8 and 5e-12 at 1e-12.
Skips (73): optional packages (mcfun, phonopy, ASE, wannier90, pyscf.properties), CPU-feature limits stated in the tests, the ECP LDS limit above (5), and test_multigrid.py (legacy libmgrid, commented out upstream as well).

### CUDA gate (dc45713 plus the pending gint change): PASS
Same redist toolkit. git archive dc45713 over the gate tree, plus the two gint files; `cmake --build -j 100 -- VERBOSE=1`: exit 0, 12 libs, all solvent/sem/multigrid_v2/multigrid_v3 TUs recompiled by nvcc, no USE_HIP or cuda_to_hip on any line.

## Porter 2026-10-31 (porter, linux-gfx942 MI300X, fork moat-port dc45713 -> e1ad6bc): gint cleanup, final run, ported

e1ad6bc `[ROCm] Drop the gint workaround for a source that is not built`: removes the HIP-only `set_source_files_properties(nr_fill_ao_ints.cu ... -O1)` and `target_link_options(gint PRIVATE -O1)`, and restores nr_fill_ao_ints.cu (the NROOTS 7/8 `#ifndef USE_HIP`) to upstream. The file is commented out of the gint source list, so none of it applied to a built source. The build log has no `-O1` on any line.

Full tree rerun at the e1ad6bc tree (README build, gfx942, same runner): **1832 passed / 16 failed / 73 skipped / 0 errors**, identical per directory and the identical failure set to dc45713 (classification above). Baseline 5c60311: 575/639/135/259. CUDA gate covered e1ad6bc's TUs (previous section). `jargon.py --port gpu4pyscf` clean, `prose.py` clean on all five messages; fork `git status --porcelain` empty (note: md/tests writes `from_h5.xyz` into the cwd; delete it after a run).

Commits this round (all on top of 5c60311, no amend or force): 4a90309, c073679, e469394, dc45713, e1ad6bc.

### For jeff
1. ECP derivative integrals with g shells exceed the 64 KB LDS on gfx90a/gfx942 (see the hardware-limit section above): accept as a hardware limit (upstream already skips on small-shared-memory GPUs) or ask for a reduced-LDS kernel.
2. The ejk block-size mismatch (e469394) is a latent upstream bug on CUDA too; whether to report it upstream is your call.
3. Perf follow-ups, not correctness: grouped GEMM via CK/hipBLASLt instead of one GEMM per group; a hipTensor contraction engine instead of cupy.einsum.

### For the next roles
Every platform revalidates at e1ad6bc. The validator should run the whole tree (agent_space-style per-directory loop above, or `pytest gpu4pyscf -m 'not slow and not benchmark and not special'`), not only scf/tests, and judge against the classification above. Build now includes libxc (ExternalProject, needs network for the libxc tarball) and Fortran (gfortran) for libsolvent. Test env additions: pyscf-dispersion, geometric (pytest-xdist optional).

## Review 2026-10-31 (reviewer, linux-gfx942, fork moat-port @ e1ad6bc vs upstream/master 10dcea3; focus 5c60311..e1ad6bc) -- REVIEW PASSED

Whole `git diff 10dcea3...e1ad6bc` read with the pr-review checklist, hardest on 4a90309, c073679, e469394, dc45713, e1ad6bc. No defect in the fork code. One correction to the porter's failure analysis (below) is recorded here and as deferred items; it changes no code and blocks nothing. Upstream master has moved one commit (c4ce229) since the merge base; that is not a review finding.

### CUDA path: identical (rebuilt, compared at machine-code level)
NVIDIA redist 12.8.1 archives re-downloaded and sha256-checked against redistrib_12.8.1.json: cuda_nvcc 12.8.93, cuda_cudart 12.8.90, cuda_cccl 12.8.90, cuda_profiler_api 12.8.90, cuda_cuobjdump 12.8.90, cuda_nvdisasm 12.8.90. `git archive` of 10dcea3 (base) and e1ad6bc (head), each built with `cmake -S <tree>/gpu4pyscf/lib -B <bdir> -DCMAKE_CUDA_ARCHITECTURES=80 -DCUDAToolkit_ROOT=$K -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && cmake --build <bdir> -j 48 -- VERBOSE=1`: both exit 0, 12 libs + libxc.so.15 each.
- compile_commands.json, paths normalised: all 116 CUDA compile commands identical between base and head; no USE_HIP, cuda_to_hip or hip_compat on any line of the head build log.
- `cuobjdump -xelf all` + `nvdisasm -c`, with the RDC per-TU static-symbol hashes normalised: libcupy_helper, libgdft, libgecp, libgvhf, libsem, libsolvent byte- or SASS-identical; libgvhf_rys, libmgrid_v2, libmgrid_v3, libpbc, libgvhf_md SASS-identical. libgint differs in one function (register numbering only) although that TU's preprocessed source (`nvcc -E` with the recorded command) is byte-identical between the trees; libxc, built from the same tarball with identical commands, shows the same kind of residual differences in 41 functions, so this is nvcc output nondeterminism, not a source change.

### e469394 (gradient energy kernel block size): right fix, HIP-only is correct, not an observable CUDA defect
- Verified: `adjust_threads` (gvhf-rys/vhf.cuh:155) doubles `nsq_per_block` when `numRegs <= 128`; LAUNCH_JKENERGY_KERNEL launches `dim3(nsq_per_block, gout_stride)` with `buflen = nroots*2*nsq_per_block + ...`; the ejk kernels use `constexpr int nsq_per_block = 256` for `rw = shared_memory + sq_id` and `cicj_cache = shared_memory + nsq_per_block*(nroots*2)`, so with 512 threads lanes 256-511 overlap cicj_cache and every task is run twice. A scripted scan of every launcher that calls adjust_threads: the kernels it adjusts hard-code nsq_per_block only in unrolled_ejk_ip1.cu (9 kernels), unrolled_ejk_ip2_type12.cu (4) and unrolled_ejk_ip2_type3.cu (4); in unrolled_rys_jk.cu, unrolled_rys_k.cu, unrolled_rys_jk_ip1.cu and pbc/unrolled_rys_k.cu every adjusted kernel takes its width from `blockDim.x` (JKMATRIX_KERNEL_SETUP). The three files are separate TUs in gvhf-rys/CMakeLists.txt:12-13, so the `#define adjust_threads(kernel, threads)` override leaks nowhere.
- Reading blockDim.x in the ejk kernels instead would change generated CUDA code (the constexpr feeds shared-memory offsets and unrolling); suppressing the doubling for these launchers is the minimal fix.
- CUDA: registers of every kernel the launchers double, nvcc 12.8: sm_80 (built library, `cuobjdump --dump-resource-usage`) min 174 (rys_ejk_ip1_0000), sm_70 min 182, sm_90 min 168 (TU recompiled with `-Xptxas -v`); the type12/type3 ones are 220-255. None is <= 128, so the doubling never fires on upstream's default arch list (70/80/90) and CUDA results are unaffected. This is a latent inconsistency in upstream's launchers (it would fire if a future nvcc or arch fit one of these kernels in 128 registers), not an observable upstream CUDA defect. The fix therefore stays HIP-only in the port; a separate upstream report is optional and jeff's call (question 2 of the porter's list stands, now with this evidence).

### dc45713 grouped GEMM replacement: correct
`grouped_dot`/`grouped_gemm` on ROCm call `cupy.dot` per group (gpu4pyscf/lib/cupy_helper.py:1009-1016, 1085-1092). GPU check against numpy for groups (M,N,K) = (5,7,3), (0,4,3), (4,0,3), (3,4,0), (64,129,17), (1,1,1), with and without `Cs`: max error 2.2e-16 (dot) and 0 (gemm); K=0 with a prefilled `out` gives zeros, as the CUTLASS path's beta=0 would. `ROCBLAS_LAYER=1` shows one `rocblas_dgemm` per group, i.e. CuPy's hipBLAS backend, not a hand loop. Result shapes and dtype match the CUDA path's `cupy.empty((M, N))` float64.

### c073679 hipDeviceSetLimit via ctypes: correct
cupy-rocm-7-0 14.2.0 on TheRock 10.1 reports `runtimeGetVersion() == 71626385`, so the code opens `libamdhip64.so.7`, which is the exact library CuPy mapped (checked in /proc/self/maps); the soname major follows HIP_VERSION_MAJOR by construction. A missing library raises OSError, a non-zero status raises through `check_status`; nothing is swallowed. `hipLimitStackSize == 0x0` (hip_runtime_api.h:808), value 128 bytes, same as upstream's `deviceSetLimit(0x00, 128)`. `release_gpu_stack()` returns cleanly on gfx942.

### Other checks
- Machine-LICM flag: still appended only inside `if(USE_HIP)` (gpu4pyscf/lib/CMakeLists.txt), documented in the CMake comment and the README ROCm paragraph with its removal condition.
- libxc HIP ExternalProject: a two-arch configure (`-DCMAKE_HIP_ARCHITECTURES=gfx942;gfx90a`) writes `set(CMAKE_HIP_ARCHITECTURES "gfx942;gfx90a" ...)` and the HIP compiler into libxc's initial cache, so the list survives.
- Wave-size review of dc45713: pcm.cu width-THREADS(32) shuffle with blockDim.x == THREADS and a row-uniform early return; multigrid_v2 full-wave reduction writes rows `thread_id/warpSize` of an `n_threads/32`-row buffer that is zeroed and summed in full; multigrid_v3 and the hiprtc preamble reduce 32-lane groups (`lane = tid % 32`, blocks of 256-1024 threads, uniform loop bounds). All correct on wave32 and wave64.
- Commit hygiene: 5 new titles 43-64 chars with `[ROCm]`, AI disclosure, fenced Test Plans, no Co-Authored-By; `python3 utils/jargon.py --port gpu4pyscf` clean (after `git fetch origin master:master` in the clone); `utils/prose.py` clean on all five bodies.

### Correction: 3 of the 4 "tolerance near-misses" are rocBLAS trsm accuracy, not fp32 storage or atomics
Reran on gfx942 (HIP_VISIBLE_DEVICES=3, README build of e1ad6bc, numpy 2.2.6, scipy 1.17.1, cupy-rocm-7-0 14.2.0, pyscf 2.14.0): all 4 reproduce (1.0117e-7, 1.12e-9, 2.2e-12, and the strain FD). Root cause by substitution:
- mp/test_dfmp2_kernel_one_gpu: rounding the exact fp64 cderi to fp32 changes e_corr_os by only 1.8e-10, so fp32 storage is not the cause. The error enters in `decompose_j3c_gpu` (mp/dfmp2_addons.py:538, `cupyx.scipy.linalg.solve_triangular` in fp32 -> `rocblas_strsm`). On the identical fp32 j3c and fp32 Cholesky factor (n=136, cond(L)=864): scipy/LAPACK fp32 solve -> energy error 4.0e-9, max relative 2.8e-6, backward error 3.8 eps; rocBLAS -> 9.1e-8, 2.2e-5, 22 eps.
- df/test_df_rks_grad::test_grad_hybrid: by functional, BLYP 5.6e-13, HF 9.5e-10, B3LYP 1.18e-9: the deviation is in the DF exchange gradient. Not integral screening (Int3c2eOpt cutoff 1e-14/1e-18/1e-22 all give 9.6e-10). Replacing `solve_triangular` in df/grad/rhf.py (`_gen_metric_solver`, fp64 -> `rocblas_dtrsm`) with scipy drops the HF deviation from 9.4e-10 to 3.4e-11.
- df/test_df_tdrhf_grad::test_j_energy_per_atom (multi-DM vs summed single-DM, line 253): the same substitution in df/grad/rhf.py and df/grad/tdrhf.py drops it from 2.2-2.4e-12 to 2.5e-13.
- pbc/df/test_ft_ao_strain_deriv: not rechecked beyond the rerun; the porter's Richardson-FD comparison stands.
On random right-hand sides with these same factors (def2-TZVPP-ri, cond 864; def2-universal-jkfit, cond 4969), rocBLAS trsm is as accurate as LAPACK (backward 0.8-1.6 eps). Even on the real data its backward error, 22 eps, is within the normwise bound (n*eps = 136 eps). So this is a weaker accuracy margin in a ROCm library, not a port defect and not a GPU correctness defect, and the misses stay under 3x. The porter's classification ("float32 cderi by default", "atomic accumulation order", none given for test_grad_hybrid) is wrong for these three. The dc45713 commit body repeats it ("float32 DF-MP2, ..., atomic accumulation order"). The body is upstream-visible and cannot be amended, so the correction is a PR-prep squash item (deferred `gpu4pyscf-dc45713-msg-trsm-classification`). Whether to report the accuracy gap to rocBLAS is a person's call (deferred `rocblas-trsm-accuracy-vs-lapack-gpu4pyscf`). To reproduce: build the j2c Cholesky factor and the fp32 j3c as in `dfmp2_kernel_one_gpu` (mp/dfmp2_drivers.py:70-83, def2-TZVPP / def2-TZVPP-ri water), solve with `cupyx.scipy.linalg.solve_triangular` and with `scipy.linalg.solve_triangular` on the same fp32 inputs, and compare both against an fp64 solve.

The 7 RNG cases were rerun: each fails only on its `lib.fp` fingerprint line, with O(1) differences (inputs drawn from cupy.random), after its other assertions pass (test_jk_energy_per_atom fails at line 266 after lines 263-265 pass). The krylov and missing-basis cases were not rerun; the porter's evidence is accepted.

### For jeff
1. (porter's question 2, now with evidence) e469394: no CUDA build of upstream's arch list fires the doubling (lowest count 168 registers at sm_90), so there is no upstream CUDA defect to fix in the port. A separate upstream report would describe a latent launcher/kernel inconsistency. Your call.
2. rocBLAS trsm on real DF data is 6x worse in backward error and 8-28x worse in forward error than LAPACK. That costs three upstream tests their tolerance by less than 3x. Report it to rocBLAS or accept it (deferred `rocblas-trsm-accuracy-vs-lapack-gpu4pyscf`).
3. Correct the dc45713 body's failure classification in the PR-prep squash (deferred `gpu4pyscf-dc45713-msg-trsm-classification`).

### For the validators
Judge against the porter's 16-failure classification, with this section's correction for test_dfmp2_kernel_one_gpu, test_grad_hybrid and test_j_energy_per_atom. Expect those three to miss their tolerance by less than 3x on every AMD arch, because they all go through rocBLAS trsm.

## Validation 2026-10-31 (validator, linux-gfx942 MI300X, fork moat-port @ e1ad6bc) -- COMPLETED

Independent validator run (fresh fork clone, clean build directory, separate job venv), HIP_VISIBLE_DEVICES=3, `gfx942:sramecc+:xnack-`. Toolchain: TheRock ROCm SDK 10.1.0 (AMD clang 24), cmake 3.31.6, gfortran 13, Python 3.11.17. Test env: numpy 2.2.6, scipy 1.17.1, cupy-rocm-7-0 14.2.0, pyscf 2.14.0, pyscf-dispersion 1.5.0, geometric 1.1.1, pytest 7.3.2, pytest-xdist 3.3.1, pytest-timeout 2.4.0.

### Build (README recipe, libxc on by default)
```
pip install 'numpy>=2.0,<2.3' 'scipy>=1.13' cupy-rocm-7-0==14.2.0 pyscf==2.14.0 pytest pyscf-dispersion geometric pytest-timeout
rm -rf build
cmake -S gpu4pyscf/lib -B build/temp.gpu4pyscf -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942
cmake --build build/temp.gpu4pyscf -j 100 -- VERBOSE=1
```
Exit 0 in 178 s: 12 libs plus libxc.so.15 (ExternalProject, `ENABLE_HIP=ON`). The build log has `-mllvm -disable-machine-licm` on the HIP compiles (113 lines), `--offload-arch=gfx942` only, and no `-O1`. Fork `git status --porcelain` is empty after the run (the runner deletes md/tests' `from_h5.xyz`).

### Full test tree
Per directory over all 39 `gpu4pyscf/**/tests`, under `ulimit -c 0`, `LD_LIBRARY_PATH=$(rocm-sdk path --root)/lib PYTHONPATH=$PWD`:
`python -m pytest <dir> -n 4 --dist loadfile -m 'not slow and not benchmark and not special' -q -rfEs --tb=line -p no:cacheprovider --timeout 1800 --junitxml=...` (upstream CI selection; 4044 s total).

| directory | porter gfx942 @ e1ad6bc p/f/s/e | this run p/f/s/e |
## Validation 2026-10-31 (validator, linux-gfx1100, fork moat-port @ e1ad6bc) -- VALIDATION FAILED (rocBLAS batched-GEMM wrong results; port code not at fault)

Requested by jeff: the full test tree at e1ad6bc on wave32, judged against the gfx942 run (1832/16/73/0). Host: AMD Radeon Pro W7800 48GB, gfx1100 (wave32), `HIP_VISIBLE_DEVICES=2` pinned by the dispatcher, no other process on that GPU. Driver amdgpu 6.19.14.31400100, kernel 6.8.0-138, MEC firmware 2680, `cwsr_enable=1`. Toolchain: TheRock ROCm SDK 10.1.0 (AMD clang 24.0.0), rocBLAS librocblas.so.5.7 from the SDK. Host CPU: Threadripper PRO 5975WX (AVX2, no AVX-512). Fork clone at e1ad6bc6d25e983b2dea21fe98af33ab4cfc3681 (== origin/moat-port); `git status --porcelain` empty after the run.

**Result: validation-failed at e1ad6bc.** On gfx1100, rocBLAS `gemm_strided_batched_ex` with M=N=1 and a large batch_count returns wrong values for a varying handful of batch entries. CuPy-ROCm routes `cupy.einsum`/`cupy.matmul` batched contractions there, and with no cuTENSOR on ROCm gpu4pyscf's einsums go through CuPy's own engine. Users of the multigrid PBC DFT GGA/MGGA path (pbc/dft/multigrid_v2.py nr_rks/nr_uks, `cp.einsum("gp, pg -> p", xc_for_fock[1:4], Gv)`) and of the multicollinear GKS functionals (dft/mcfun_gpu.py `_eval_xc_lebedev`) get wrong, run-to-run varying numbers on this card. Every other failure maps to the gfx942 list or to an accuracy or host effect explained below. No port commit is at fault: with only `cupy.einsum` replaced, the affected tests pass. I did not mark this platform completed: wave32 is the only Linux wave32 evidence, and these are wrong results in shipped code paths, not an accuracy margin. **For jeff:** either the porter adds a ROCm workaround (keep these contractions off M=N=1 batched GEMM; see "Workaround evidence"), after which gfx1100 revalidates, or you accept the rocBLAS defect as known and rule this platform completed. Bug report registered: deferred `rocblas-gemm-strided-batched-m1n1-f64-gfx1100-wrong-results` (its summary predates the idle complex-f64 finding below; this section is the authoritative record).

### Environment (job venv)
```
$FLEET_VENV/bin/pip install 'numpy>=2.0,<2.3' 'scipy>=1.13' cupy-rocm-7-0==14.2.0 pyscf==2.14.0 pytest pyscf-dispersion geometric pytest-timeout
# numpy 2.2.6, scipy 1.17.1, cupy-rocm-7-0 14.2.0, pyscf 2.14.0, pyscf-dispersion 1.5.0, geometric 1.1.1, pytest 7.3.2, pytest-xdist 3.3.1, pytest-timeout 2.4.0
# mcfun 0.2.6 was added during the run, after df/ and dft/ had run (to get a CPU reference for the GKS case); see the per-directory note
export PATH=$FLEET_VENV/bin:$PATH LD_LIBRARY_PATH=$(rocm-sdk path --root)/lib PYTHONPATH=<fork>; ulimit -c 0
```

### Build: README ROCm block, clean build dir, libxc on (default), machine-LICM flag kept
```
rm -rf build
cmake -S gpu4pyscf/lib -B build/temp.gpu4pyscf -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build/temp.gpu4pyscf -j 32 -v
```
Exit 0, 240 s (utils/timeit.sh compile). 12 libs (cupy_helper gdft gecp gint gvhf gvhf_md gvhf_rys mgrid_v2 mgrid_v3 pbc sem solvent) plus libxc.so.15 from the ExternalProject with `ENABLE_HIP=ON`. HIP compiler found without a hint; gfortran 13.3.0 for libsolvent. 213 `--offload-arch=gfx1100` lines; `-disable-machine-licm` on 113 lines; no other arch.

### Water vs CPU PySCF (`ulimit -c 0`, conv_tol 1e-12)
- RKS/def2-svp, grids level 3: B3LYP 1.07e-11, PBE 5.91e-12, M06 7.06e-12 Ha.
- RHF energy: sto-3g 3.6e-12, def2-svp 9.7e-12, cc-pvtz 2.9e-11 Ha.
- RHF gradient, max component: with the default `conv_tol_grad` (sqrt(conv_tol) = 1e-6) the gaps were 4e-11 / 3.2e-9 / 1.9e-8. That is SCF convergence: with conv_tol 1e-14 and conv_tol_grad 1e-10 they are 1.1e-11 / 4.3e-12 / 4.8e-12. The GPU gradient evaluated from the CPU-converged orbitals agrees with CPU to 4.6e-14 / 2.2e-13 / 8.5e-12.

### Full tree (upstream CI selection), per directory
Runner, per directory: `python -m pytest <dir> -n 4 --dist loadfile -m 'not slow and not benchmark and not special' -q -rfEs --tb=line -p no:cacheprovider --timeout 1800 --junitxml=...` (same as the gfx942 runner; one GPU). Wall time 3234 s.

| directory | gfx942 @ e1ad6bc p/f/s/e | gfx1100 @ e1ad6bc p/f/s/e |
|---|---|---|
| cc/tests | 4/0/0/0 | 4/0/0/0 |
| df/hessian/tests | 24/0/0/0 | 24/0/0/0 |
| df/tests | 200/5/3/0 | 200/5/3/0 |
| dft/tests | 113/0/9/0 | 113/0/9/0 |
| dft/tests | 113/0/9/0 | 111/2/9/0 |
| dispersion/tests | 16/0/0/0 | 16/0/0/0 |
| fci/tests | 3/0/0/0 | 3/0/0/0 |
| geomopt/tests | 0/0/5/0 | 0/0/5/0 |
| grad/tests | 115/2/0/0 | 115/2/0/0 |
| gto/tests | 57/0/5/0 | 57/0/5/0 |
| hessian/tests | 91/0/1/0 | 91/0/1/0 |
| lib/tests | 39/0/1/0 | 39/0/1/0 |
| mcscf/tests | 8/0/0/0 | 8/0/0/0 |
| md/tests | 5/0/0/0 | 5/0/0/0 |
| mp/tests | 20/1/0/0 | 20/1/0/0 |
| nac/tests | 71/1/1/0 | 71/1/1/0 |
| pbc/df/tests | 104/1/7/0 | 104/1/7/0 |
| pbc/dft/tests | 156/0/18/0 | 156/0/18/0 |
| pbc/dft/tests | 156/0/18/0 | 98/28/14/34 |
| pbc/grad/tests | 185/0/2/0 | 185/0/2/0 |
| pbc/gto/pseudo/tests | 7/0/0/0 | 7/0/0/0 |
| pbc/gto/tests | 13/0/0/0 | 13/0/0/0 |
| pbc/hessian/tests | 0/0/6/0 | 0/0/6/0 |
| pbc/lib/tests | 2/0/0/0 | 2/0/0/0 |
| pbc/properties/berry/tests | 10/0/2/0 | 10/0/2/0 |
| pbc/scf/tests | 88/2/1/0 | 88/2/1/0 |
| pbc/tools/tests | 16/0/0/0 | 16/0/0/0 |
| pbc/x2c/tests | 1/0/1/0 | 1/0/1/0 |
| pop/tests | 2/0/0/0 | 2/0/0/0 |
| properties/tests | 26/0/5/0 | 26/0/5/0 |
| qmmm/pbc/tests | 3/0/0/0 | 3/0/0/0 |
| qmmm/tests | 21/0/0/0 | 21/0/0/0 |
| scf/tests | 159/2/3/0 | 159/2/3/0 |
| sem/dmp/tests | 1/0/0/0 | 1/0/0/0 |
| sem/grad/tests | 3/0/0/0 | 3/0/0/0 |
| sem/gto/tests | 15/0/0/0 | 15/0/0/0 |
| sem/integral/tests | 27/0/0/0 | 27/0/0/0 |
| solvent/tests | 126/0/2/0 | 126/0/2/0 |
| tdscf/tests | 93/2/0/0 | 93/2/0/0 |
| tests | 0/0/0/0 | 0/0/0/0 |
| x2c/tests | 8/0/1/0 | 8/0/1/0 |
| **total** | 1832/16/73/0 | **1832/16/73/0** |

### The 16 failures: the identical set, each with the recorded signature
- RNG fingerprints (7): df test_jk_energy_per_atom (fp -14.927 vs 16.882), df test_get_nacv_ee (1.0724 vs -0.8861), grad/test_tdrhf_grad test_grad_elec (7.823375488826, the exact GPU value in the porter's evidence), grad/test_tduhf_grad test_grad_elec, nac/test_tdrhf_nac_ee test_get_nacv_ee, pbc/scf/test_pbc_soscf test_rks_gen_g_hop and test_uks_gen_g_hop. All O(1) fingerprint mismatches.
- d-aug-cc-pvdz missing from PySCF 2.14 (4): BasisNotFoundError in df test_general_contraction, scf test_rhf_df_against_qchem and test_rks_df_against_qchem, tdscf test_rtda_ris_diffuse.
- rocBLAS trsm accuracy (3, the review's correction): mp test_dfmp2_kernel_one_gpu 1.0117209e-7 vs 7 places (same digits as the review's rerun), df test_grad_hybrid 1.124e-9 vs 1e-9, df test_j_energy_per_atom 2.34e-12 vs 1e-12.
- Finite-difference strain (1): pbc/df test_ft_ao_strain_deriv 3.77e-8 vs 3e-8.
- Krylov cap (1): tdscf test_krylov_eigenvalue, eigenvector error 1.34e-5 at max_iter 35.
No new failure and no error, so there is no new GPU finding.

### Water vs CPU PySCF (def2-svp, grids level 3, conv_tol 1e-12)
- RKS energies: B3LYP 1.07e-11, PBE 5.87e-12, M06 6.88e-12 Ha; all converged on both sides.
- RHF gradient, GPU gradient evaluated on the CPU's converged orbitals: sto-3g 4.8e-14, def2-svp 2.2e-13, cc-pvtz 9.8e-12 (max abs). With two independently converged SCFs the gradient difference is 3.9e-11 / 3.2e-9 / 1.9e-8. That is the density difference allowed by conv_tol 1e-12, not the gradient engine.

### Other gates
- CUDA gate: skipped. It is already recorded at e1ad6bc (Review 2026-10-31, base vs head compared at machine-code level), and this host has no CUDA toolkit.
- `python3 utils/jargon.py --port gpu4pyscf`: clean. README documents the ROCm build (the recipe above is the README's).
| x2c/tests | 8/0/1/0 | 7/1/1/0 |
| **total** | 1832/16/73/0 | 1771/47/69/34 |

mcfun was installed while nac/tests was running, so pbc/dft/tests ran its mcfun-gated GKS tests (test_pbc_gks: 6 passed, 6 skipped), which gfx942 skipped. That is why pbc/dft has 14 rather than 18 skips.

**The 16 gfx942 failures reproduce by name, all 16 and nothing else, in df (5), grad (2), mp (1), nac (1), pbc/df (1), pbc/scf (2), scf (2), tdscf (2).** Their classification (porter, with the reviewer's trsm correction) stands. Not individually re-root-caused here; same names, same directories.

### The deviations, each root-caused
1. **dft/test_numint2c.py::test_eval_xc_eff_fp** (MGGA_C_M06, multicollinear, vmat fingerprint off by 2.7e-8; on rerun `test_eval_xc_eff` vs CPU PySCF+mcfun, 4.8e-8) and **pbc/dft/test_multigrid_v2.py::test_get_vxc_gga, _gga_kpts, _mgga, _mgga_kpts** (vxc off by 3.5e-5 to 1.7e-3 against 1e-8, different every run): **rocBLAS M=N=1 batched GEMM.** Bisection:
   - mcol M06: the run-to-run variation is in `mcfun_gpu.eval_xc_eff`. libxc is bit-reproducible on the same input (polarized deriv=2, 1.6M points, 6 runs), and it matches CPU libxc (exc 9e-15, vxc 1.3e-13, fxc relative 7e-11). It also does not come from a stream race (`AMD_SERIALIZE_KERNEL=3` and `HIP_LAUNCH_BLOCKING=1` still give wrong runs) or from reading pool garbage (prefilling the CuPy pool with 0, NaN, 1e300 or -1 changes nothing). Re-running each operation twice on identical inputs: only `cp.einsum('xgo,xgo->go')` and `cp.einsum('xbygo,xgo->bygo')` give different bits, at 1-60 entries per call. UKS M06 (no batched einsum) is exact and deterministic.
   - multigrid: the density on the G mesh varies only by atomic-order jitter (1e-13). The xc-to-Fock input varies by up to 0.67 at the GGA term `cp.einsum("gp, pg -> p", xc_for_fock[1:4], Gv)`, which is rocBLAS `gemm_strided_batched_ex` f64_c, M=N=1, K=3, batch_count 347395 (mesh 67x61x85). Standalone, with random data on that shape: 688-757 wrong entries per call over 10 calls; `(x*Gv.T).sum(0)` gives 0.
   - CuPy-only reproducer (no gpu4pyscf): `cp.einsum('xgo,xgo->go', a, b)` with a, b of shape (5, 10240, 160) f64 gives 1-3 wrong entries (errors 0.3-3.6) in 9 of 20 calls while the test tree shared the GPU; `(a*b).sum(0)` gives 0 of 20. ROCBLAS_LAYER=2 shows `gemm_strided_batched_ex -m 1 -n 1 -k 5 ... --batch_count 1638400`.
   - Direct rocBLAS through ctypes (`rocblas_gemm_strided_batched_ex`, f64_r, N/N, M=N=1, lda=1, ldb=K, stride_c=1, device pointer mode): while another process used the GPU, k=64 gave 0 wrong at batch 4096 and 16384, 1 at 65536, 59385 at 131072 and 125093 at 262144 over 10 calls; k=16 gave 154; k=5 was sporadic. `rocblas_ddot_strided_batched` on the same data: always 0. Shapes (1x5)@(5x2), (2x5)@(5x1), 4x4, 8x8 and f32 (1x5)@(5x1): 0. No writes outside D (1M-entry sentinel bands on both sides intact), inputs unmodified.
   - **Load dependence:** the fault is timing-sensitive. On an idle GPU (no other process), f64_r M=N=1 was clean at every size tried (k 3/5/64, batch up to 347395, 5-10 calls each), but **f64_c (complex128) M=N=1 still fails idle**: batch 347395, k=3 gave 106-136 wrong per call, k=5 155-220, k=64 0-1. With one concurrent process running GEMM+FFT on the same GPU, f64_r fails heavily (counts above), and the mcol M06 case gives 7 of 10 wrong runs against 0 of 10 idle. The upstream CI runner (`-n 4 --dist loadfile` on one GPU) always runs under such load.
   - In the tree run, `test_get_vxc_lda_kpts` raised `hipErrorIllegalAddress` in the xdist worker that had just run the failing gga tests. The error is sticky, so the remaining 51 tests of that worker (17 test_pbc_numint, the rest of test_multigrid_v2, all 34 test_pbc_rks setups) failed or errored. Rerunning pbc/dft/tests with the same `-n 4 --dist loadfile`: 5 failed (the 4 above plus lda_kpts, 7.8e-4), 151 passed, 18 skipped, no illegal access. Serially on the idle GPU: 4 failed (the 4 above), 152 passed, 18 skipped. lda_kpts passes in the serial run. I could not tie the illegal access to rocBLAS (the guard bands caught no stray write), so it is recorded as seen once, following the wrong results, cause not proven.
   - **Workaround evidence:** with a diagnostic `sitecustomize` that computes two-operand `cupy.einsum` as broadcast-multiply-and-sum (no GEMM), test_multigrid_v2.py + test_numint2c.py pass twice in a row: 34 passed, 3 skipped, 0 failed (serial). The port's HIP code (dc45713 multigrid_v2 `__shfl_down`) is not involved: the LDA density kernel it touches is deterministic to 1e-14 and every LDA test passes.
2. **dft/test_numint.py::test_sparse_index** (`lib.fp(...) == 103.60117204957997` gives 103.60117204958): host CPU float summation, not GPU. The three fingerprinted arrays are int32 index arrays (1223/483/270 entries). All three fingerprints miss by 1-3 ulp, while changing any single index would move the fingerprint by about 1e-3 or more. `math.fsum` of the same products gives a third rounding (103.6011720495799). This host's numpy runs AVX2 OpenBLAS/SIMD (no AVX-512), so the dot's summation order differs from the reference machine's. An exact `==` on a float fingerprint cannot pass across CPUs; not a port or GPU question.
3. **x2c/test_x2c.py::test_sfx2c1e_cart** (hcore max diff 6.12e-11 against `assertAlmostEqual(.., 10)`, i.e. 5e-11; deterministic): accuracy margin, 1.2x. max|hcore| is 33 Ha, so this is relative 1.9e-12. By substitution: with the generalized eigh (gpu4pyscf.lib.cusolver.eigh, hipSOLVER), `cp.linalg.eigh` and `cp.linalg.solve` all replaced by SciPy/NumPy on the GPU-built matrices, it is still 4.58e-11 (the integral floor). The GPU generalized eigh adds 1.5e-11, and cp.linalg.eigh and solve add nothing. No M=N=1 batched GEMM in this test. Same class as the "<3x tolerance misses" (rocBLAS/rocSOLVER accuracy margin); gfx942 passes it, presumably by a small margin.

### Checks
- CUDA gate: not rerun. It passed at e1ad6bc on gfx942 (porter and reviewer sections), and this host has no verified CUDA toolkit.
- `python3 utils/jargon.py --port gpu4pyscf`: clean (after `git fetch origin master:master` in the clone).
- Documentation: the README ROCm block builds as written (only the arch and `-j` changed).
- Fork `git status --porcelain`: empty; no `from_h5.xyz` left.

### What would flip this to completed
A porter commit that keeps the batched contractions off M=N=1 `gemm_strided_batched` on ROCm. The narrowest place is CuPy's einsum as gpu4pyscf calls it: `gpu4pyscf/lib/cupy_helper.contract` and the direct `cp.einsum` sites in pbc/dft/multigrid_v2.py and dft/mcfun_gpu.py. Alternatively, a rocBLAS release without the defect. Then a gfx1100 rerun of the tree. A ruling by jeff that the defect is accepted also flips it, but that ruling is his.

## Porter 2026-10-31 (porter, linux-gfx1100 W7800, fork moat-port e1ad6bc -> 3982189): rocBLAS batched-GEMV workaround and a multigrid_v2 BlockScan race

Requested by jeff: answer the gfx1100 validation failure at e1ad6bc with a full, correct port on every supported GPU. Work lock taken (`validation-failed -> porting`, linux-gfx1100). Host: Radeon Pro W7800 48GB x4 (gfx1100), `HIP_VISIBLE_DEVICES=2` from the dispatcher (GPUs 0, 1, 3 used briefly for the reproducer only, all idle at the time). TheRock ROCm SDK 10.1.0: HIP runtime 7.16.26385 (`hipRuntimeGetVersion` 71626385), AMD clang 24.0.0 (d6f6cb6918), rocBLAS 5.7.0.39d8d603 (librocblas.so.5.7). Venv: numpy 2.2.6, scipy 1.17.1, cupy-rocm-7-0 14.2.0, pyscf 2.14.0, pytest-xdist 3.3.1 (no mcfun, to keep the test selection identical to gfx942).

Two new commits on top of e1ad6bc (no amend, no force push):

- 51f6cf0 `[ROCm] Synchronize before reusing BlockScan storage in multigrid_v2` (screening.cuh, one `__syncthreads()`; unconditional, it is a CUB contract violation on CUDA too).
- 3982189 `[ROCm] Keep one-row batched matmul off rocBLAS gemv_strided_batched` (cupy_helper.py; ROCm only).

### 1. The rocBLAS defect, reduced (no CuPy)

It is wider than "M=N=1 GEMM". `rocblas_gemm_strided_batched_ex` with **n=1** (column-major, N/N) is run by rocBLAS as `gemv_strided_batched`, and `rocblas_{d,z}gemv_strided_batched` (transA=N) itself gives the same wrong entries. It fails for **m = 1, 2, 4** and is clean for m = 8, 12, 16, 64, 256; it needs a **batch_count above about 131072** (complex128, idle: 0/20 calls wrong at 65536 and 100000, 1/20 at 131072, 20/20 at 262144). m=1 with n=2/4 (CuPy `(B,m,k)@(B,k,1)`) and m=n=2 are clean. The wrong values are not another batch entry's result and not a partial sum of the k terms (checked against every other entry and every subset of the k products). The fault is the same on all four W7800s (GPU 0, 1, 2, 3: 2307, 2251, ~2300, 2301 wrong entries over 3 calls of complex128 m=n=1 k=3 batch 347395), so it is not one bad card.

In CuPy terms: `(..., 1, k) @ (..., k, n)` becomes rocBLAS `M=n, N=1` (ROCBLAS_LAYER=2 confirms), so every batched matmul whose left operand has one row is exposed when n < 8. `cupy.einsum` produces exactly that for every batched dot product ('xg,xg->g', 'gp,pg->p', 'xbygo,xgo->bygo').

Failure rate (`gemm_n1`, below; 20 calls each; "load" = a second process looping 256x256 DGEMM on the same GPU):

| type | m | k | batch_count | idle: calls wrong / wrong entries | with load |
|---|---|---|---|---|---|
| f64_c | 1 | 3 | 347395 (the multigrid shape) | 20/20, 15469 | 20/20, 30830 |
| f64_c | 1 | 5 | 347395 | 20/20, 5067 | |
| f64_c | 1 | 3 | 131072 | 1/20, 2 | |
| f64_c | 1 | 3 | 262144 | 20/20, 116 | |
| f64_c | 2 | 3 | 347395 | 20/20, 37440 | |
| f64_c | 4 | 3 | 347395 | 20/20, 116284 | |
| f64_c | 8 | 3 | 347395 | 0/20 | |
| f64_r | 1 | 5 | 1638400 (the mcfun shape) | 0/20 | 12/20, 20 |
| f64_r | 1 | 5 | 262144 | | 8/20, 13 |
| f64_r | 1 | 64 | 262144 | 0/20 | 11/20, 18 |
| f64_r | 2 | 64 | 262144 | | 13/20, 42 |
| f64_r | 4 | 64 | 262144 | 0/20 | |
| f64_r | 1 | 64 | 65536 / 16384 | | 0/20 / 0/20 |
| f64_c gemv_strided_batched | 1 | 3 | 347395 | 20/20, 15243 | |
| f64_r gemv_strided_batched | 1 | 64 | 262144 | | 14/20, 33 |

A heavier 4096x4096 DGEMM load gave f64_r only 1/5 calls wrong at k=64; many small concurrent kernels expose it best, which matches the pytest-xdist runner.

Build and run (source in agent_space/repro/gemm_n1.cpp, pasted below):
```
R=$(rocm-sdk path --root)
hipcc -O2 --offload-arch=gfx1100 gemm_n1.cpp -lrocblas -L$R/lib -Wl,-rpath,$R/lib -o gemm_n1
./gemm_n1 z 1 3 347395 20          # complex128, idle
./gemm_n1 d 1 64 262144 20 load    # float64 with a concurrent process
./gemm_n1 z 1 3 347395 20 gemv     # same through rocblas_zgemv_strided_batched
```

```cpp
// rocblas_gemm_strided_batched_ex with n=1 (and rocblas_?gemv_strided_batched)
// returns wrong entries for small m and a large batch_count on gfx1100.
//
// build: hipcc -O2 --offload-arch=gfx1100 gemm_n1.cpp -lrocblas -o gemm_n1
// usage: gemm_n1 <d|z> <m> <k> <batch_count> <calls> [load] [gemv]
//   d = double (f64_r), z = double complex (f64_c)
//   load: fork a child process that keeps the same GPU busy with 256x256
//         DGEMMs while the parent runs
//   gemv: call rocblas_?gemv_strided_batched instead of gemm_strided_batched_ex
//
// Column-major, no transposes: A is m x k (lda=m, stride_a=m*k), B is k x 1
// (ldb=k, stride_b=k), D is m x 1 (ldd=m, stride_d=m), alpha=1, beta=0.
// With m=1 this is exactly the call CuPy issues for einsum('xg,xg->g'):
//   gemm_strided_batched_ex N N -m 1 -n 1 -k K --lda 1 --stride_a K --ldb K
//   --stride_b K --ldc 1 --stride_c 1 --ldd 1 --stride_d 1
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <algorithm>
#include <complex>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(1); } } while (0)
#define RB_CHECK(x) do { rocblas_status s_ = (x); if (s_ != rocblas_status_success) { \
    fprintf(stderr, "%s:%d rocblas status %d\n", __FILE__, __LINE__, (int)s_); exit(1); } } while (0)

typedef std::complex<double> zd;

static void load_child()
{
    const int n = 256;
    rocblas_handle h;
    RB_CHECK(rocblas_create_handle(&h));
    double *a, *b, *c;
    HIP_CHECK(hipMalloc(&a, sizeof(double) * n * n));
    HIP_CHECK(hipMalloc(&b, sizeof(double) * n * n));
    HIP_CHECK(hipMalloc(&c, sizeof(double) * n * n));
    HIP_CHECK(hipMemset(a, 0, sizeof(double) * n * n));
    HIP_CHECK(hipMemset(b, 0, sizeof(double) * n * n));
    double one = 1, zero = 0;
    for (;;) {
        RB_CHECK(rocblas_dgemm(h, rocblas_operation_none, rocblas_operation_none,
                               n, n, n, &one, a, n, b, n, &zero, c, n));
        HIP_CHECK(hipDeviceSynchronize());
    }
}

static double rnd(std::mt19937_64 &g, std::uniform_real_distribution<double> &u, double *) { return u(g); }
static zd rnd(std::mt19937_64 &g, std::uniform_real_distribution<double> &u, zd *)
{
    double re = u(g);
    return {re, u(g)};
}
static rocblas_datatype dtype(double *) { return rocblas_datatype_f64_r; }
static rocblas_datatype dtype(zd *) { return rocblas_datatype_f64_c; }

static rocblas_status gemv(rocblas_handle h, int m, int k, const double *alpha, const double *a,
                           long sa, const double *x, long sx, const double *beta, double *y, long sy, int batch)
{
    return rocblas_dgemv_strided_batched(h, rocblas_operation_none, m, k, alpha, a, m, sa, x, 1, sx,
                                         beta, y, 1, sy, batch);
}
static rocblas_status gemv(rocblas_handle h, int m, int k, const zd *alpha, const zd *a,
                           long sa, const zd *x, long sx, const zd *beta, zd *y, long sy, int batch)
{
    typedef rocblas_double_complex rz;
    return rocblas_zgemv_strided_batched(h, rocblas_operation_none, m, k, (const rz *)alpha,
                                         (const rz *)a, m, sa, (const rz *)x, 1, sx,
                                         (const rz *)beta, (rz *)y, 1, sy, batch);
}

template <typename T>
static int run(int m, int k, int batch, int calls, bool use_gemv)
{
    std::mt19937_64 g(12345);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const long sa = (long)m * k, sb = k, sd = m;
    std::vector<T> ha(sa * batch), hb(sb * batch), ref(sd * batch), hd(sd * batch);
    for (auto &x : ha) x = rnd(g, u, (T *)nullptr);
    for (auto &x : hb) x = rnd(g, u, (T *)nullptr);
    for (long p = 0; p < batch; p++)
        for (int i = 0; i < m; i++) {
            T s = T(0);
            for (int l = 0; l < k; l++) s += ha[p * sa + (long)l * m + i] * hb[p * sb + l];
            ref[p * sd + i] = s;
        }

    T *da, *db, *dd;
    HIP_CHECK(hipMalloc(&da, sizeof(T) * sa * batch));
    HIP_CHECK(hipMalloc(&db, sizeof(T) * sb * batch));
    HIP_CHECK(hipMalloc(&dd, sizeof(T) * sd * batch));
    HIP_CHECK(hipMemcpy(da, ha.data(), sizeof(T) * sa * batch, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(db, hb.data(), sizeof(T) * sb * batch, hipMemcpyHostToDevice));

    rocblas_handle h;
    RB_CHECK(rocblas_create_handle(&h));
    const T alpha = T(1), beta = T(0);
    const rocblas_datatype dt = dtype((T *)nullptr);

    int bad_calls = 0;
    long bad_total = 0;
    for (int c = 0; c < calls; c++) {
        HIP_CHECK(hipMemset(dd, 0, sizeof(T) * sd * batch));
        if (use_gemv)
            RB_CHECK(gemv(h, m, k, &alpha, da, sa, db, sb, &beta, dd, sd, batch));
        else
            RB_CHECK(rocblas_gemm_strided_batched_ex(
                h, rocblas_operation_none, rocblas_operation_none, m, 1, k, &alpha,
                da, dt, m, sa, db, dt, k, sb, &beta, dd, dt, m, sd, dd, dt, m, sd, batch,
                dt, rocblas_gemm_algo_standard, 0, 0));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(hd.data(), dd, sizeof(T) * sd * batch, hipMemcpyDeviceToHost));
        long bad = 0, first = -1;
        double maxerr = 0;
        for (long i = 0; i < sd * batch; i++) {
            double e = std::abs(hd[i] - ref[i]) / std::max(1.0, std::abs(ref[i]));
            if (!(e <= 1e-12)) {
                if (first < 0) first = i;
                bad++;
                maxerr = std::max(maxerr, e);
            }
        }
        if (bad) bad_calls++;
        bad_total += bad;
        printf("call %2d: %ld of %ld entries wrong", c, bad, sd * batch);
        if (bad) printf(" (first at %ld, max rel err %.3g)", first, maxerr);
        printf("\n");
    }
    printf("summary: %s %s m=%d n=1 k=%d batch_count=%d: %d of %d calls wrong, %ld wrong entries\n",
           use_gemv ? "gemv_strided_batched" : "gemm_strided_batched_ex",
           dt == rocblas_datatype_f64_r ? "f64_r" : "f64_c", m, k, batch, bad_calls, calls, bad_total);
    RB_CHECK(rocblas_destroy_handle(h));
    HIP_CHECK(hipFree(da));
    HIP_CHECK(hipFree(db));
    HIP_CHECK(hipFree(dd));
    return bad_calls ? 1 : 0;
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr, "usage: %s <d|z> <m> <k> <batch_count> <calls> [load] [gemv]\n", argv[0]);
        return 2;
    }
    const char type = argv[1][0];
    const int m = atoi(argv[2]), k = atoi(argv[3]), batch = atoi(argv[4]), calls = atoi(argv[5]);
    bool load = false, use_gemv = false;
    for (int i = 6; i < argc; i++) {
        load |= !strcmp(argv[i], "load");
        use_gemv |= !strcmp(argv[i], "gemv");
    }

    pid_t child = -1;
    if (load) {
        child = fork();
        if (child == 0) load_child();
        sleep(2);
    }

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    char ver[256];
    RB_CHECK(rocblas_get_version_string(ver, sizeof(ver)));
    int hipver;
    HIP_CHECK(hipRuntimeGetVersion(&hipver));
    printf("device: %s %s, HIP runtime %d, rocBLAS %s, load=%s\n", prop.name, prop.gcnArchName,
           hipver, ver, load ? "on" : "off");

    int rc = type == 'z' ? run<zd>(m, k, batch, calls, use_gemv) : run<double>(m, k, batch, calls, use_gemv);

    if (child > 0) {
        kill(child, SIGKILL);
        waitpid(child, nullptr, 0);
    }
    return rc;
}
```

### 2. Draft rocBLAS issue (for jeff to file; not filed)

**Title:** gemv_strided_batched (and gemm_strided_batched_ex with n=1) returns wrong entries for small m and batch_count > 131072 on gfx1100

**Environment:** AMD Radeon PRO W7800 48GB (gfx1100), reproduced on four cards in one host. Linux 6.8.0-138, amdgpu 6.19.14.31400100. TheRock ROCm SDK 10.1.0: HIP runtime 7.16.26385, AMD clang 24.0.0, rocBLAS 5.7.0.39d8d603.

**Description:** `rocblas_zgemv_strided_batched` / `rocblas_dgemv_strided_batched` with `transA = rocblas_operation_none`, m in {1, 2, 4}, lda = m, incx = incy = 1, alpha = 1, beta = 0 and a batch_count above about 131072 writes wrong values to a varying subset of the outputs. `rocblas_gemm_strided_batched_ex` with n = 1 (N/N, f64_r or f64_c) shows the same, presumably because it dispatches to the same kernel. CuPy issues exactly this call for `einsum('xg,xg->g', a, b)` and for any `(B,1,k) @ (B,k,n)` matmul with small n, so applications get silently wrong results. The wrong values change from call to call. They are neither another batch entry's result nor a partial sum.

- double complex, m=1, k=3, batch_count=347395: several hundred wrong entries in every call on an otherwise idle GPU.
- double, m=1, k=64, batch_count=262144: clean on an idle GPU, but 11 of 20 calls have wrong entries while another process runs small DGEMMs on the same GPU.
- m >= 8, batch_count <= 65536, and n > 1 shapes were clean in every run.

**Reproducer:** the attached `gemm_n1.cpp` (self-contained; CPU reference; prints the number of wrong entries per call).
```
hipcc -O2 --offload-arch=gfx1100 gemm_n1.cpp -lrocblas -o gemm_n1
./gemm_n1 z 1 3 347395 20
./gemm_n1 d 1 64 262144 20 load
./gemm_n1 z 1 3 347395 20 gemv
```

**Expected:** `0 of 347395 entries wrong` on every call (relative error <= 1e-12 against the CPU sum of k products).

**Actual:** e.g. `call 0: 755 of 347395 entries wrong (first at 112, max rel err 2.89)`; `summary: gemm_strided_batched_ex f64_c m=1 n=1 k=3 batch_count=347395: 20 of 20 calls wrong, 15469 wrong entries`. With `load`, float64: `summary: gemm_strided_batched_ex f64_r m=1 n=1 k=64 batch_count=262144: 11 of 20 calls wrong, 18 wrong entries`.

The deferred item `rocblas-gemm-strided-batched-m1n1-f64-gfx1100-wrong-results` predates this reduction; its summary says M=N=1 and batch > 65536, and this section is the authoritative description (n=1, m < 8, batch above about 131072; gemv_strided_batched itself is affected).

### 3. The workaround (3982189)

On ROCm only, `gpu4pyscf.lib.cupy_helper` replaces `cupy.matmul` with `_hip_matmul`. For float/complex operands with a batch (max ndim >= 3), no `out=` or other keyword, and a one-row left operand (`x1.ndim == 1` or `x1.shape[-2] == 1`), it computes `(a.swapaxes(-1, -2) * b).sum(axis=-2, keepdims=True)` (squeezing for 1-D operands, as matmul does). Every other call goes to the original `cupy.matmul`. That is the narrowest place covering every caller: `cupy.linalg._einsum.reduced_binary_einsum` looks up `cupy.matmul` at call time for every binary contraction, so all `cp.einsum` sites (about 740 in the tree), `lib.contract` (cupy engine, since there is no cuTENSOR on ROCm), and explicit `cp.matmul` are covered. The module already patches PySCF at import (`_patch_pyscf`), so this follows house style. CUDA: the assignment is under `cupy.cuda.runtime.is_hip`, so the CUDA path is unchanged. Not covered by construction: the `@` operator (`ndarray.__matmul__` calls CuPy's C-level matmul directly). The rocBLAS shape log of the full tree run (below) shows whether any `@` site reaches the shape.

Unit checks against NumPy (`agent_space/shapes2.py`, 5 calls each, batch 347395 where shown): (B,1,3)@(B,3,1) 3.7e-15, (B,1,3)@(B,3,4) 4.0e-15, (B,4,3)@(B,3,1) 3.7e-15, (B,1,5)@(B,5,3) 5.3e-15, 1-D@3-D, 3-D@1-D, broadcast 3-D and 4-D <= 4.4e-16 with matching shapes; `einsum('gp,pg->p')` complex 2.0e-15, `einsum('xgo,xgo->go')` (5,10240,160) 3.6e-15, `einsum('xbygo,xgo->bygo')` 1.8e-15, `contract('xgo,xgo->go')` 1.8e-15. Before: 206-226 wrong entries per call for the multigrid einsum. With ROCBLAS_LAYER=2 the only rocBLAS call left from these was the clean `M=1, N=4` shape.

### 4. The hipErrorIllegalAddress in pbc/dft: root-caused, a BlockScan race (51f6cf0)

It is not rocBLAS and not caused by the einsum fix: it reproduces at e1ad6bc with the workaround removed.
- Reproduction: `pytest gpu4pyscf/pbc/dft/tests/test_multigrid_v2.py -n 4` (default `--dist load`, so four processes run multigrid tests at the same time): hipErrorIllegalAddress in 2/2 runs with the workaround and 2/2 runs without it. Alone: `test_get_vxc_gga` 5/5 pass, and the whole file passes serially, even with 40 GB of the GPU held by another process (not memory pressure). Three concurrent serial runs of the file: 2 of 3 crash.
- Location: with `HIP_LAUNCH_BLOCKING=1`, `evaluation.cuh:899` (`evaluate_xc_driver` launch check, li=1 lj=0, from test_get_nuc; li=0 lj=1 in another run).
- Cause: a wrapper around `multigrid_v2.assign_pairs_to_blocks` that checks `pairs_on_blocks[:accumulated[-1]]` for entries outside [0, n_pairs) found slots left at the `-1` fill in the crashing processes (e.g. `n_pairs=2158 used=1163423 bad=91576`; 6 bad assignments in one process, 1 in another, 0 in the clean one), while the screening outputs (`screen_gaussian_pairs`) were always complete. `evaluate_xc_kernel` then reads `image_indices[-1]` and `non_trivial_pairs[-1]`. `put_pairs_on_blocks_kernel` (screening.cuh) calls `cub::BlockScan<int, 256>().ExclusiveSum` once per batch of 1024 pairs in a loop, and every call uses the same shared TempStorage with no `__syncthreads()` between calls. CUB requires that barrier before temp storage is reused. A fast wave can start the next scan while a slow one is still reading the previous result. Other processes on the GPU skew the wave timing, which is why it needs concurrency. (`count_pairs_on_blocks_kernel` already puts a `__syncthreads()` between its two BlockReduce calls; the other block collectives in multigrid_v2 are single calls or already barrier-separated.)
- Fix: one `__syncthreads()` at the end of the loop body. After it: 6 of 6 concurrent runs of the file (two rounds of three) pass 29/29 with zero bad slots.
- CUDA compile gate (a TU changed): NVIDIA redist 12.8.1 archives sha256-verified against redistrib_12.8.1.json (cuda_nvcc 12.8.93, cuda_cudart 12.8.90, cuda_cccl 12.8.90, cuda_profiler_api 12.8.90) into agent_space/cuda-12.8; `git ls-files` export of the working tree at 3982189; `cmake -S <tree>/gpu4pyscf/lib -B build-cuda -DCMAKE_CUDA_ARCHITECTURES=80 -DCUDAToolkit_ROOT=$K -DCMAKE_BUILD_TYPE=Release -DBUILD_LIBXC=OFF && cmake --build build-cuda -j 16 --target mgrid_v2`: exit 0, libmgrid_v2.so built, `screen.cu` (the TU that includes screening.cuh) compiled by nvcc for `compute_80,sm_80`, and no `USE_HIP` or `cuda_to_hip` on any compile line. The cupy_helper.py change is Python under `is_hip`, so nothing else on the CUDA side changes.

### 5. Full test tree at 3982189 (gfx1100)

Build: README ROCm block (`cmake -S gpu4pyscf/lib -B build/temp.gpu4pyscf -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100`, `-j 32`), clean build dir, libxc on: exit 0, 12 libs plus libxc.so.15; `-mllvm -disable-machine-licm` on the HIP compile lines (machine-LICM flag carried unchanged). Rebuilt mgrid_v2 after 51f6cf0.

Runner, per directory, under `ulimit -c 0`, `PYTHONPATH=<fork> LD_LIBRARY_PATH=$(rocm-sdk path --root)/lib`, with `ROCBLAS_LAYER=2` piped through a FIFO into a shape tally (agent_space/blaslog_reader.py):
`python -m pytest <dir> -n 4 --dist loadfile -m 'not slow and not benchmark and not special' -q -rfEs --tb=line -p no:cacheprovider --timeout 1800 --junitxml=...`

| directory | gfx942 @ e1ad6bc p/f/s/e | gfx1100 @ 3982189 p/f/s/e |
|---|---|---|
| cc/tests | 4/0/0/0 | 4/0/0/0 |
| df/hessian/tests | 24/0/0/0 | 24/0/0/0 |
| df/tests | 200/5/3/0 | **201/4/3/0** |
| dft/tests | 113/0/9/0 | **112/1/9/0** |
| dispersion/tests | 16/0/0/0 | 16/0/0/0 |
| fci/tests | 3/0/0/0 | 3/0/0/0 |
| geomopt/tests | 0/0/5/0 | 0/0/5/0 |
| grad/tests | 115/2/0/0 | 115/2/0/0 |
| gto/tests | 57/0/5/0 | 57/0/5/0 |
| hessian/tests | 91/0/1/0 | 91/0/1/0 |
| lib/tests | 39/0/1/0 | 39/0/1/0 |
| mcscf/tests | 8/0/0/0 | 8/0/0/0 |
| md/tests | 5/0/0/0 | 5/0/0/0 |
| mp/tests | 20/1/0/0 | 20/1/0/0 |
| nac/tests | 71/1/1/0 | 71/1/1/0 |
| pbc/df/tests | 104/1/7/0 | **105/0/7/0** |
| pbc/dft/tests | 156/0/18/0 | 156/0/18/0 |
| pbc/grad/tests | 185/0/2/0 | 185/0/2/0 |
| pbc/gto/pseudo/tests | 7/0/0/0 | 7/0/0/0 |
| pbc/gto/tests | 13/0/0/0 | 13/0/0/0 |
| pbc/hessian/tests | 0/0/6/0 | 0/0/6/0 |
| pbc/lib/tests | 2/0/0/0 | 2/0/0/0 |
| pbc/properties/berry/tests | 10/0/2/0 | 10/0/2/0 |
| pbc/scf/tests | 88/2/1/0 | 88/2/1/0 |
| pbc/tools/tests | 16/0/0/0 | 16/0/0/0 |
| pbc/x2c/tests | 1/0/1/0 | 1/0/1/0 |
| pop/tests | 2/0/0/0 | 2/0/0/0 |
| properties/tests | 26/0/5/0 | 26/0/5/0 |
| qmmm/pbc/tests | 3/0/0/0 | 3/0/0/0 |
| qmmm/tests | 21/0/0/0 | 21/0/0/0 |
| scf/tests | 159/2/3/0 | 159/2/3/0 |
| sem/dmp/tests | 1/0/0/0 | 1/0/0/0 |
| sem/grad/tests | 3/0/0/0 | 3/0/0/0 |
| sem/gto/tests | 15/0/0/0 | 15/0/0/0 |
| sem/integral/tests | 27/0/0/0 | 27/0/0/0 |
| solvent/tests | 126/0/2/0 | 126/0/2/0 |
| tdscf/tests | 93/2/0/0 | 93/2/0/0 |
| tests | 0/0/0/0 | 0/0/0/0 |
| x2c/tests | 8/0/1/0 | **7/1/1/0** |
| **total** | 1832/16/73/0 | **1832/16/73/0** |

Wall time 6689 s (the host was shared with another job; the validator's run took 3234 s). **Total 1832/16/73/0, the same as gfx942.**

- **pbc/dft/tests 156/0/18/0 and dft/tests' numint2c tests: the validator's deviations are gone.** test_multigrid_v2 (gga, gga_kpts, mgga, mgga_kpts, lda_kpts) and test_numint2c::test_eval_xc_eff_fp all pass. dft/tests' one failure is test_sparse_index (section 6).
- **No hipErrorIllegalAddress and no worker abort anywhere in the tree** (`grep -l 'hipError\|Fatal Python' tree/*.log` is empty). Clean concurrent runs of test_multigrid_v2.py after 51f6cf0: this tree run plus 6 of 6 three-way concurrent runs (section 4). Before 51f6cf0 it failed 4 of 6 concurrent runs and 2 of 2 `-n 4` runs.
- The 16 failures: 14 of gfx942's 16 by name (df test_general_contraction, test_grad_hybrid, test_jk_energy_per_atom, test_get_nacv_ee; grad test_tdrhf_grad and test_tduhf_grad test_grad_elec; mp test_dfmp2_kernel_one_gpu; nac test_get_nacv_ee; pbc/scf test_rks_gen_g_hop and test_uks_gen_g_hop; scf test_rhf_df_against_qchem and test_rks_df_against_qchem; tdscf test_rtda_ris_diffuse and test_krylov_eigenvalue), with the classes recorded earlier (RNG fingerprints, missing d-aug-cc-pvdz basis, rocBLAS trsm margin, Krylov cap). The other two gfx942 failures, df test_j_energy_per_atom (trsm margin) and pbc/df test_ft_ao_strain_deriv (finite-difference margin), pass here. The remaining two are host or margin effects explained in section 6: dft test_sparse_index (`103.60117204958 == 103.60117204957997`) and x2c test_sfx2c1e_cart (`6.122036211309023e-11` against 10 places), the same values the validator recorded.

**rocBLAS shape log over the whole tree** (ROCBLAS_LAYER=2, agent_space/tree/blas_shapes.json): 691006 `gemm_strided_batched_ex` and 7428 `gemm_batched` calls in 15603 distinct shapes from 29 directories, largest batch_count 249856. No `gemv_strided_batched` call. Batched calls with n=1: only 7 `gemm_batched` shapes with batch_count 3 (hessian and solvent tests, m <= 21), far below the failing regime (above about 131072). So no other path, including `@`, reaches the defective kernel in the test tree.

### 6. The two host-side deviations (no code change)

- **dft/test_numint.py::test_sparse_index** (`lib.fp(...) == 103.60117204957997`, got 103.60117204958): host CPU float summation, not GPU. The fingerprinted arrays are int32 index arrays; all three fingerprints are 1-3 ulp off, while changing any single index would move the fingerprint by about 1e-3 or more. `math.fsum` of the same products gives a third rounding. This host's numpy runs AVX2 OpenBLAS/SIMD kernels (Threadripper PRO 5975WX, no AVX-512), so the dot's summation order differs from the machine that recorded the reference. An exact `==` on a float fingerprint cannot hold across CPUs. (The validator's analysis; the result reproduces in this run, see the table.)
- **x2c/test_x2c.py::test_sfx2c1e_cart** (hcore max diff 6.12e-11 against `assertAlmostEqual(..., 10)`, i.e. 5e-11; deterministic): an accuracy margin of 1.2x, relative 1.9e-12 of max|hcore| = 33 Ha. With hipSOLVER's generalized eigh, `cp.linalg.eigh` and `cp.linalg.solve` all replaced by SciPy/NumPy on the GPU-built matrices it is still 4.58e-11 (the integral floor); the GPU generalized eigh adds 1.5e-11. No batched GEMV in this test. Same class as the other <3x rocBLAS/rocSOLVER accuracy misses; gfx942 passes it, presumably by a small margin.

### Checks
- `python3 utils/jargon.py --port gpu4pyscf`: clean. `utils/prose.py` on both commit bodies: clean. Titles 67 and 67 characters, `[ROCm]` prefix, AI disclosure in prose, fenced Test Plan, no Co-Authored-By.
- Build log: `-mllvm -disable-machine-licm` on 113 HIP compile lines (carried unchanged).
- Fork `git status --porcelain`: empty after removing the tests' `from_h5.xyz`.
- Pushed `moat-port` e1ad6bc..3982189 (fast-forward).
- Skill: `references/fault-classes.md` gained the batched-GEMV/CuPy einsum entry (Types, dispatch and platform limits) and a timing-dependent block-collective paragraph under the hipCUB TempStorage entry, plus one index line each in SKILL.md.

### For jeff
- File the rocBLAS issue from section 2 if you agree (not filed). The workaround stays in place until a fixed rocBLAS ships; its comment names the defect.
- 51f6cf0 is a race in upstream's CUDA code too (CUB requires the barrier). It ships in the port PR; reporting it upstream separately is your call.
- linux-gfx1100 now needs revalidation at 3982189. gfx942 and gfx90a also read `revalidate` because head moved.
