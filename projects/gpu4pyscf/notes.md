# gpu4pyscf notes

ROCm/HIP port. Lead platform linux-gfx90a (MI250X, ROCm 7.2.1). Fork:
https://github.com/AMD-Ecosystem/gpu4pyscf , branch `moat-port`. Strategy A
(CUDA spelling kept, mapped to HIP via one compat header).

## Milestone status

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
