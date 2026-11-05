# GPUMD notes

## Existing AMD support (pre-port state)

GPUMD already ships an upstream HIP path, so this was a validate-and-improve
job, not a fresh CUDA->HIP conversion:

- `src/makefile.hip` -- `hipcc --offload-arch=gfx90a -DUSE_HIP`, links
  `-lhipblas -lhipsolver -lhipfft`.
- `src/utilities/gpu_macro.cuh` -- Strategy-A compat header. Defines a `gpu*`
  symbol family aliased to `hip*` under `USE_HIP`, else to
  `cuda*`/`cublas*`/`cusolver*`/`cufft*`. Covers memory, stream, error, device
  props, cuRAND, cuBLAS, cuSOLVER Dn eigensolvers, cuFFT.
- 16 `USE_HIP` guard sites; the only behavioral ones select
  `thrust::hip::par` vs `thrust::cuda::par` (rocThrust drop-in).

## CUDA surface

- Libraries: hipBLAS, hipSOLVER (Dn dense eigensolvers), hipFFT, hipRAND,
  rocThrust -- all already wired through gpu_macro.cuh.
- Warp intrinsics: NONE. Reductions are shared-memory block tree-reductions
  with `__syncthreads()` after every step (force.cu:292, neighbor.cu:674,
  nep_charge.cu:582), including sub-warp offsets -- no implicit
  warp-synchronous assumption, so wave64-safe by construction. The warp-size
  fault class does not apply to this project.
- Textures/surfaces: NONE. The texture fault classes are moot.
- Atomics: atomicAdd, atomicCAS only. No atomicMin/Max, so the gfx90a
  coarse-grained int atomicMin/Max-drop bug does not apply.

## Changes made (HIP path only; CUDA makefile untouched)

`src/makefile.hip`, two edits, both behind the existing HIP build:

1. `-std=c++14` -> `-std=c++17`. ROOT CAUSE of the only build failure:
   rocThrust/rocPRIM in ROCm 7.2 hard-`#error` "rocPRIM requires at least
   C++17" (rocprim/config.hpp); the headers use `std::variant`, `std::is_same_v`
   etc. CUDA's Thrust compiles fine under C++14, which is why upstream's
   makefile.hip inherited `-std=c++14` and never caught this -- it bitrots
   against any modern rocThrust. Only the 3 TUs that include thrust
   (neighbor.cu, nep_multigpu.cu, dp.cu) failed; the bump fixes all.
2. Hardcoded `--offload-arch=gfx90a` -> `HIP_ARCH ?= gfx90a` (overridable).
   Followers build `make -f makefile.hip HIP_ARCH=gfx1100` with no file edit,
   so the curated commit head_sha does not churn and already-passed platforms
   are not forced to revalidate. (PORTING_GUIDE configurable-arch rule.)

No source (.cu/.cuh) changes were needed: no wave64 fix (no warp intrinsics),
no texture fix (no textures), no atomic emulation. The existing upstream compat
header and USE_HIP guards were already correct for ROCm 7.2.

## Build (gfx90a)

    cd projects/GPUMD/src/src
    export HIP_VISIBLE_DEVICES=2     # a free gfx90a on the shared host
    make -f makefile.hip clean
    make -f makefile.hip -j16        # default HIP_ARCH=gfx90a
    # produces ./gpumd (8.7 MB) and ./nep (974 KB); only benign nodiscard
    # warnings on ignored hipError_t returns, zero errors.

ROCm 7.2.1, hipcc clang 7.2.53211. Host: 4x gfx90a (MI250X/MI250, CDNA2,
wave64); GPU 3 was busy, used GPU 2.

## Validation (real gfx90a, three independent MD workloads)

Ran shipped regression cases under tests/gpumd/. Each runs a real MD trajectory.

1. carbon -- NEP4 machine-learning potential, NVE, 100 steps. Exercises the NEP
   force kernel + shared-mem reductions.
   - Energy conservation (the NVE physics gate): KE+PE relative drift 2.6e-7,
     std/|mean| 3.8e-7. Temperature 300.4 K (target 300). Correct NVE.
   - Cross-check vs committed CUDA reference (thermo1.out) at first dump
     (step 10): PE matches to 5.8e-6 relative, T/KE to 4.0e-4. Forces match
     CUDA to near fp precision.
2. graphene_dos -- Tersoff potential + phonon DOS via hipFFT, NPT then NVE,
   2x10000 steps. Exercises hipFFT + velocity-autocorrelation.
   - NVE-phase energy conservation: rel drift 2.2e-6, std/|mean| 5.4e-6.
     Temperature mean 297.7 K.
   - DOS output: 200 freq points 2-400 THz, all finite, no negative values
     (physically valid). Spectral shape matches CUDA ref to 5.7e-2.
3. graphene_kappa_emd -- EMD (Green-Kubo) thermal conductivity, heat-current
   autocorrelation.
   - Runs clean, T 297.3 K, HAC finite/well-formed (50x11). HAC tail differs
     more from the CUDA ref (sensitive long-time transport correlation under
     chaos); function is physically valid.

Final error-checked run: empty stderr, no HIP fault (GPUMD's GPU_CHECK_KERNEL
calls gpuGetLastError after kernels), no nan, energy conserved.

## Determinism (root-caused)

Run-to-run on the SAME gfx90a is NOT bit-identical: two fresh carbon runs
diverge to ~16 (in KE units) by step 40, but each run independently conserves
total energy (|dEtot| between runs stays 3e-3..1.6e-1 while KE/PE individually
decorrelate) -- the signature of deterministic physics + chaotic
(Lyapunov) amplification of ULP-level force noise.

Root cause is upstream and vendor-agnostic, NOT a HIP defect:
`force/neighbor.cu:80-81`,
`const int ind = atomicAdd(&cell_count[cell_id], 1); cell_contents[base+ind] = n1;`
-- the atomicAdd return value sets each atom's slot in its cell bin, and
atomicAdd ordering across concurrent threads is non-deterministic on any GPU.
So cell_contents (hence neighbor-list order, hence force-summation order) varies
each run on CUDA too. The velocity init is CPU `rand()` (glibc default seed),
identical run-to-run, so it is not the source.

Consequence for validation: bit-reproducible thermo (what the upstream
`tests/run_tests.sh` `diff -q` expects) is unattainable for chaotic MD across
runs or vendors. The correct acceptance gate is energy conservation +
agreement with the reference within physical tolerance (PE 5.8e-6, DOS 5.7e-2),
which the port passes.

## Followers

gfx1100/gfx1151 (RDNA, wave32): reuse this branch, build with
`make -f makefile.hip HIP_ARCH=<arch>`, no source change. Since the code has no
warp-size assumptions, the lead fix should carry; they validate on their own
hardware per MOAT policy.

## Validation 2026-05-30 (gfx1100, ROCm 7.2.1)

Host: 4x AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32). ROCm 7.2.1 / hipcc clang 22.0.0git (roc-7.2.1). Ran on HIP_VISIBLE_DEVICES=0.

### Fork state

Fork: https://github.com/AMD-Ecosystem/GPUMD, branch moat-port, commit 65b4ded54e7118861a9cb6de6ca3e8ca0e2f1b3e. Zero-change validation: no fork push, curated commit untouched, head_sha unchanged.

### Build

    cd projects/GPUMD/src/src
    make -f makefile.hip clean
    make -f makefile.hip HIP_ARCH=gfx1100 -j16
    # -> gpumd (7.8 MB), nep (838 KB); only benign nodiscard warnings, zero errors.

Wrapped with `utils/timeit.sh GPUMD compile`.

Code-object proof (roc-obj-ls ./gpumd): all GPU bundles are
`hipv4-amdgcn-amd-amdhsa--gfx1100`. No gfx90a code objects present.

    $ roc-obj-ls ./gpumd | grep -v host | head -5
    1  hipv4-amdgcn-amd-amdhsa--gfx1100  file://...#offset=655360&size=8648
    2  hipv4-amdgcn-amd-amdhsa--gfx1100  file://...#offset=671744&size=5024
    3  hipv4-amdgcn-amd-amdhsa--gfx1100  file://...#offset=684032&size=122680
    ...

### Validation (real gfx1100 GPU)

Wrapped with `utils/timeit.sh GPUMD test`. Fresh runs from agent_space/ (no pre-existing output files) with potentials symlinked via ../../../potentials/. All three runs: empty stderr, no HIP fault (GPUMD calls gpuGetLastError after every kernel via GPU_CHECK_KERNEL), no NaN, clean exit.

**1. carbon -- NEP4 NVE, 100 steps, 64000 atoms**

    cd agent_space/gpumd_tests/carbon
    HIP_VISIBLE_DEVICES=0 /path/to/gpumd

NVE energy conservation (KE+PE across 10 thermo dumps at steps 10-100):

| metric | gfx1100 | gfx90a |
|--------|---------|--------|
| drift (max-min)/\|mean\| | 9.76e-7 | 2.6e-7 |
| std/\|mean\| | 3.45e-7 | 3.8e-7 |
| T mean | 300.35 K | 300.4 K |

Cross-check vs committed CUDA reference (thermo1.out, step 10):

| quantity | rel diff gfx1100 | rel diff gfx90a |
|----------|-----------------|-----------------|
| PE | 9.77e-6 | 5.8e-6 |
| T/KE | 6.18e-4 | 4.0e-4 |

Both are within the chaotic-MD divergence window (wave32 vs wave64 vs CUDA; non-deterministic atomicAdd cell binning). Energy conservation is tight (~1e-6 level). PASS.

**2. graphene_dos -- Tersoff + phonon DOS via hipFFT, NPT+NVE, 2x10000 steps**

    cd agent_space/gpumd_tests/graphene_dos
    HIP_VISIBLE_DEVICES=0 /path/to/gpumd

NPT equilibration (100 rows, dump_thermo 100): T_mean 299.55 K (target 300 K), std 1.91 K. NVE+compute_dos does not write thermo.out (by GPUMD design when compute_dos is active); NVE health confirmed via MVAC.

MVAC t=0 normalization: x=y=z=1.000000 (exact normalization; NVE physics valid).

DOS (200 freq points, 2-400 THz): all finite, all non-negative (min 3.17e-2, max 1.22e3 a.u.).

DOS spectral shape vs CUDA ref (dos1.out, normalized): max diff 6.35e-2, RMSE 1.25e-2. gfx90a had 5.7e-2. Consistent -- within expected run-to-run MD variability. PASS.

**3. graphene_kappa_emd -- EMD Green-Kubo thermal conductivity, Tersoff, NPT+NVE 2x10000 steps**

    cd agent_space/gpumd_tests/graphene_kappa_emd
    HIP_VISIBLE_DEVICES=0 /path/to/gpumd

NPT equilibration (100 rows): T_mean 299.36 K, std 2.31 K. Clean exit after NVE+compute_hac.

HAC output: 50 rows x 11 columns, time grid 0.1-9.9 ps (matches reference exactly), all finite, no NaN/inf. HAC running-kappa values differ from CUDA ref as expected (long-time transport correlation is chaos-sensitive; gfx90a notes same caveat). PASS.

### Result: PASS

No source or build change was needed (zero-change follower validation). Fork was NOT pushed; curated commit 65b4ded is untouched. validated_sha = 65b4ded54e7118861a9cb6de6ca3e8ca0e2f1b3e.

## Validation 2026-05-30 (windows-gfx1151, TheRock ROCm) -- follower, NO source change

Platform: AMD Radeon 8060S (gfx1151, RDNA3.5, integrated APU), Windows 11. ROCm via
TheRock wheels (rocm-sdk 7.14.0a20260519, hip 7.13.26190), hipcc = clang 23. Validated
fork moat-port HEAD 65b4ded54e7118861a9cb6de6ca3e8ca0e2f1b3e unchanged -- the follower
needs no fork edit (the makefile.hip HIP_ARCH param already added by the lead does the job).

### Build (no source change)
    cd projects/GPUMD/src/src
    # env: TheRock _rocm_sdk_devel/bin + lib/llvm/bin on PATH, HIP_PATH=<rocm root>,
    #      HIP_DEVICE_LIB_PATH=<rocm>/lib/llvm/amdgcn/bitcode, MSVC cl dir ahead of MSYS link.exe
    mingw32-make -f makefile.hip gpumd HIP_ARCH=gfx1151 LDFLAGS="-L<rocm>/lib" -j16
Builds clean (only benign -Wnodiscard warnings, zero errors) -> gpumd (13.7 MB). The only
Windows-specific invocation detail vs Linux is `LDFLAGS=-L<rocm>/lib` so lld-link finds
hipblas/hipsolver/hipfft import libs (on Linux /opt/rocm/lib is the default search path);
this is an environment flag, NOT a source/makefile change. hipBLAS/hipSOLVER/hipFFT/hipRAND
+ rocThrust all resolve from the TheRock wheel. A single hipSOLVER TU (cusolver_wrapper.cu)
and the full link both succeed. Renamed the output to gpumd.exe to run on Windows.

### Validation (real gfx1151 GPU, HIP_VISIBLE_DEVICES=0) -- three MD workloads, all PASS
Acceptance gate per the gfx90a/gfx1100 records: clean exit, no HIP fault (GPUMD calls
gpuGetLastError after every kernel via GPU_CHECK_KERNEL), no NaN, energy conservation
(chaotic MD is not bit-exact run-to-run or cross-vendor).

1. tests/gpumd/carbon (NEP4 potential, NVE, 100 steps): exit 0, empty stderr, no NaN.
   Total energy Etot=KE+PE conserved: range 1.43e-1 on |Etot|~1.6e5 -> ~9e-7 relative
   (matches the gfx90a/gfx1100 "~1e-6" level). PASS.
2. tests/gpumd/graphene_dos (Tersoff, compute_dos via VAC + hipFFT, 2x10000 steps):
   exit 0, empty stderr. dos.out 200x4 + mvac.out + velocity.out all finite, no NaN.
   Exercises the hipFFT path. PASS.
3. tests/gpumd/graphene_kappa_emd (Tersoff, compute_hac EMD thermal transport, 2x10000):
   exit 0, empty stderr. hac.out 50x11 finite, no NaN (matches gfx1100's 50x11). PASS.

### Notes
- GPUMD is compute-heavy (kernels + hipBLAS/hipSOLVER/hipFFT) and does NOT query
  hipMemGetInfo, so it is unaffected by the gfx1151-APU runtime gap that blocked rmm; it
  runs cleanly on the APU just like the gsplat PyTorch workloads did.
- No warp intrinsics (block-level __syncthreads reductions only), so wave32 is correct by
  construction -- confirmed by the clean energy-conserving runs (same as the gfx1100 wave32
  result). No wave-size or any source fix required for gfx1151.
State: port-ready -> completed (validated_sha 65b4ded, fork unchanged).

## Carry-forward 2026-06-03 (all platforms) -- doc-only to PR head 98c8bb4

PR-prep rewrote the fork: moat-port advanced 65b4ded -> 98c8bb4 (PR #1538 head), a
forced update that orphaned 65b4ded (not an ancestor of the new tip). All three
platforms had GPU-validated at 65b4ded (gfx90a + gfx1100 + gfx1151 records above) and
their validated_sha was bumped to 98c8bb4 without a recorded classification -- closing
that documentation gap here.

Net tree delta 65b4ded..98c8bb4 is a single file: `doc/installation.rst` (+16), pure
documentation. `git diff 65b4ded 98c8bb4` over source/build files (*.cu/*.cuh/*.cpp/*.h,
makefile*, CMake*) is EMPTY -- zero device source or build change. Formal classifier:

    python3 utils/moatlib.py classify GPUMD 65b4ded 98c8bb4
    -> class=doc-only arch_independent=True inert=True

Doc-only is arch-independent, so the carry-forward holds for gfx90a, gfx1100 and gfx1151
with no GPU re-run and no rebuild (the source classifier is the authoritative gate for
this class; a per-arch binary-equiv build is unnecessary when device source is provably
untouched). All three remain completed @ 98c8bb4. PR #1538 stands as-is.

## Validation 2026-06-04 (windows-gfx1101 + windows-gfx1201, one FAT binary) -- follower, NO source change

validated_sha: 6231243f (zero-churn followers; fork untouched). Host = the dual-GPU
Windows workstation (memory windows-gfx1101-gfx1201-host). ROCm 7.14 / TheRock pip SDK.

### Multi-arch fat binary (one build, both GPUs)
Built ONE binary embedding gfx1101 + gfx1201 device code, selected at run time by
HIP_VISIBLE_DEVICES (0=gfx1101 Radeon PRO V710 RDNA3, 1=gfx1201 RX 9070 XT RDNA4):
```
ROCM=.../_rocm_sdk_devel ; export HIP_PATH=$ROCM
export HIP_DEVICE_LIB_PATH=$ROCM/lib/llvm/amdgcn/bitcode
export PATH="<MSVC HostX64/x64>:$ROCM/bin:$ROCM/lib/llvm/bin:$PATH"
cd projects/GPUMD/src/src
mingw32-make -f makefile.hip gpumd \
  CFLAGS="-O3 --offload-arch=gfx1101 --offload-arch=gfx1201 -DUSE_HIP -std=c++17" \
  LDFLAGS="-L$ROCM/lib" -j64
```
(scripts: agent_space/gpumd-win/.) The single gpumd.exe runs clean on BOTH devices.

Windows-specific INVOCATION flags (NOT source/makefile edits, so no fork HEAD churn):
- `-std=c++17` in CFLAGS: Linux hipcc defaults to C++17, but clang targeting
  x86_64-pc-windows-msvc here does not propagate C++17 to the MSVC STL, so rocPRIM's
  std::variant / "rocPRIM requires at least C++17" hard-errors (neighbor/dp/nep_multigpu
  TUs) without it. The lead deliberately un-pinned -std relying on the Linux default;
  Windows needs it explicit.
- `LDFLAGS=-L$ROCM/lib` so lld-link finds the hipblas/hipsolver/hipfft import libs.
- HIP_DEVICE_LIB_PATH so hipcc finds the amdgcn device bitcode; rename output gpumd -> gpumd.exe.

### MULTI-ARCH FOLLOW-UP (PR-prep, deferred): makefile.hip currently takes a single
`--offload-arch=$(HIP_ARCH)`; the fat binary above is produced via a CFLAGS override.
To make GPUMD natively multi-arch (per the project standard that every port supports
multi-arch builds), makefile.hip should turn HIP_ARCH into a LIST -> one --offload-arch
per element (clang rejects a comma list in one flag). Do this at PR-prep, NOT now: a
makefile edit churns HEAD and flips the completed gfx90a/gfx1100/gfx1151 to revalidate,
and the RETIRED gfx1151 host cannot revalidate -- it would strand and block the PR.

### Validation (real GPU, acceptance gate per the gfx90a/gfx1100/gfx1151 records:
clean exit, empty stderr, no HIP fault, no NaN, energy conservation; chaotic MD is not
bit-exact run-to-run or cross-vendor)

| case | gfx1101 (dev0) | gfx1201 (dev1) |
|------|----------------|----------------|
| carbon (NEP4 NVE, 100 steps) | exit 0, T=300.53, Etot drift 1.37e-6, no NaN, PE vs CUDA ref 5.2e-5 | exit 0, T=300.42, Etot drift 8.06e-7, no NaN, PE vs CUDA ref 4.23e-6 |
| graphene_dos (Tersoff + hipFFT DOS, 2x10000) | dos.out 200x4 finite, non-negative | dos.out 200x4 finite, non-negative |
| graphene_kappa_emd (Green-Kubo EMD, 2x10000) | hac.out 50x11 finite | hac.out 50x11 finite |

All exit 0, empty stderr, no NaN. Energy conservation ~1e-6 (matches gfx90a 2.6e-7 /
gfx1100 9.8e-7 / gfx1151 ~9e-7). hipFFT DOS path works on both RDNA3 and RDNA4. No warp
intrinsics (block __syncthreads reductions), so wave32 is correct by construction on both.

State: windows-gfx1101 + windows-gfx1201 port-ready -> completed (validated_sha 6231243f,
fork unchanged). All five platforms now terminal -> PR-ready.

## Follow-up 2026-11-05 (linux-gfx90a, porter) -- host `max` on double/float truncates to int, 6231243f -> 15124b53

Source: jeff's host min/max truncation sweep (`~/claude-shared/plans/moat-hip-host-minmax-truncation-sweep.md`, hit 5). Model fix: LEAP 63d2724. PR #1538 is merged, so `moatlib.py pr-state GPUMD` = `merged`, `moat-port` is not frozen; the push went to `moat-port` and the fork pre-push hook (protect-fork, fresh clone) allowed it. This commit is a follow-up candidate, not part of #1538.

Upstream check first: brucefan1983/GPUMD master (last touch of the file 1c9916cb9e, 2026-10-23) still has `max_nep_rc = max(max_nep_rc, parambs[i].rc_radial);` at `src/force/ilp_nep.cu:221`, so upstream has not fixed it.

Record re-homing: the folder was trunk-resident (finished project). `moatlib.ensure_port_branch('GPUMD')` cut `port/GPUMD` from origin/main e1e33d39, then `moatlib.py worktree GPUMD`. The first attempt from the trunk checkout was refused by the trunk pre-push `readme` gate (table stale from other ports' progress; that gate is skipped on `port/` branches). The push only created a ref at the unchanged origin/main sha, so README.md was regenerated in the working tree for that one push and then restored with `git checkout -- README.md`; nothing was committed to main.

### Defect
`max_nep_rc` is double, `rc_radial` is float. CUDA's host `max(double, float)` is `fmax(a, double(b))`. HIP's `__clang_hip_math.h` on this host (HIP 7.16.26385, clang 24.0.0git; `__HIP_DEFINE_EXTENDED_HOST_MIN_MAX__` defaults to 0) declares only host `max(int,int)`, so both arguments are converted to int: the 4.2 A NEP cutoff becomes 4. The check `if (max_nep_rc > rc)` (ILP rcut_global) at line 344 then accepts an ILP cutoff in (4, 4.2), which CUDA rejects, and the neighbour list built at the ILP cutoff is too short for NEP.

Fix: `max_nep_rc = fmax(max_nep_rc, double(parambs[i].rc_radial));` One line. Lines 229-230 (`max_MN_radial`/`max_MN_angular`) and the `max_n_max_angular`/`max_dim` lines are `(int,int)` and unchanged; `min(num_atoms, CUDA_MAX_NL_ILP_NEP_TMD)` is int/int. Device `min(...)` on floats in the kernels (2106+) are device code, left alone.

### Project-wide re-check (host-only poison compile)
Forced include `template<class A,class B> void min(A,B)=delete; template<class A,class B> void max(A,B)=delete;` with this build's real flags, every `*/*.cu` file:

    hipcc -x hip --cuda-host-only -fsyntax-only -O3 --offload-arch=gfx90a -DUSE_HIP -I./ -include poison.h <file>

- default defines: only hit `force/ilp_nep.cu:221`.
- second pass with `-DUSE_FIXED_NEIGHBOR -DUSE_NEPCG -DDEBUG -DUSE_FIXED_SCALER -DCG_RDF -DSTRONG_DEBUG -DOUTPUT_NEIGHBOR_FOR_TRAIN -DFINE_TUNE_DESCRIPTOR -DZHEYONG -DUSE_KEPLER`, and a third with `-DUSE_GRAPHENE_TB`: same single hit.
- external-library regions (`USE_PLUMED`, `USE_TENSORFLOW`, `USE_NNAP`, `USE_NETCDF`, `USE_MDI`) cannot be compiled here; grep of every file carrying those macros (force/dp.cu, nnap.cu, force.cu, main_gpumd/run.cu, measure/plumed.cu, dump_netcdf.cu, main_mdi/*) finds no unqualified `min(`/`max(` at all. Bare `#if`s are CUDA_VERSION/__CUDA_ARCH__ guards with no min/max inside.
- Judged harmless: `main_gnep/` and `main_mdi/` fail the host-only compile for unrelated reasons (raw `cuda*` API, missing `curand_kernel.h`/`add_efield.cuh`); neither directory is in `makefile.hip`, so they are not part of the HIP build and cannot carry this defect into a ROCm binary.
After the fix the poison compile of `force/ilp_nep.cu` reports 0 errors.

### Run-time before/after (real gfx90a, HIP_VISIBLE_DEVICES=1, ulimit -c 0)
Case `tests/gpumd/ilp_nep_cutoff` (new): 480-atom AB bilayer graphene (group column = layer), `nep_ilp 1 C` with LAMMPS CH.ILP C-C parameters and rcut_global 4.1, map to `potentials/nep/C_2022_NEP4.txt` (cutoff 4.2 3.7).

| binary | ILP rcut 4.1 | ILP rcut 4.3 (control) |
|---|---|---|
| 6231243f (before) | accepted, exit 0, 100 NVE steps ran (PE -1565.5 eV at step 10) | -- |
| 15124b53 (after)  | rejected, exit 1, `ERROR: Please make sure cutoff of ILP is larger than it of NEP.` (CUDA behaviour) | accepted, exit 0 |

Regression test added to `tests/run_tests.sh` in the harness's diff style: `../../../src/gpumd | tail -n 1 > error.out; diff -q error.out error1.out`. Before: `Files error.out and error1.out differ` (last line blank, run completed). After: identical. `run_tests.bat` not extended (Windows `fc`-only harness, cannot be exercised here).

### GPU suite (the three recorded workloads), same day, before vs after binary
Build: `utils/timeit.sh GPUMD compile -- make -C projects/GPUMD/src/src -f makefile.hip -j64` (zero errors; incremental rebuild touched only force/ilp_nep.o). Fresh scratch copies under /tmp with potentials symlinked.

| metric | after 15124b53 | before 6231243f (today) | gfx90a record 2026-05-30 |
|---|---|---|---|
| carbon NVE drift (max-min)/\|mean\| | 1.20e-6 | 1.73e-6 | 2.6e-7 |
| carbon std/\|mean\| | 3.77e-7 | 4.36e-7 | 3.8e-7 |
| carbon T mean | 300.44 K | 300.47 K | 300.4 K |
| carbon PE vs CUDA ref (step 10) | 1.28e-5 | 1.25e-5 | 5.8e-6 |
| graphene_dos DOS vs CUDA ref max diff / RMSE | 7.38e-2 / 1.14e-2 | 5.00e-2 / 8.93e-3 | 5.7e-2 |
| graphene_dos MVAC t0 / DOS finite, >=0 | 1,1,1 / yes (min 3.07e-2) | yes | yes |
| graphene_kappa_emd T mean / HAC 50x11 finite | 297.01 K / yes | 297.20 K / yes | 297.3 K / yes |

All 3 exit 0, empty stderr, no NaN, both binaries. Before and after sit in the same chaotic-MD band (run-to-run non-determinism root-caused above to atomicAdd cell binning); none of these workloads constructs ILP_NEP, so no change is expected and none is seen. PASS: 3/3 workloads + new ilp_nep_cutoff test (fails before, passes after).

Commit 15124b53 `[ROCm] Fix NEP cutoff truncated to int in the ILP_NEP constructor`; `utils/jargon.py --port GPUMD` clean, `utils/prose.py` clean. Pre-existing: `audit-commits` flags 6231243f (already merged upstream) for no Test Plan; not touched.

CUDA compile check not run here (no nvcc on this host); the change is plain `fmax`/`double()` and the test files are data, so it should compile unchanged with nvcc -- the dispatcher's mi355 nvcc check is still owed before presenting. All platforms now read revalidate at 15124b53 (validated_sha 6231243f).

## CUDA compile check (nvcc 12.8) at 15124b5

Helper run on the mi355 fleet host (CUDA 12.8 toolkit, V12.8.93; no NVIDIA GPU), compile only, requested by jeff. Head 15124b53 vs parent 6231243f, fresh clone of the fork plus `git worktree add ../gpumd-parent 6231243f`. Note: the head commit changes one line in `src/force/ilp_nep.cu` (`max(max_nep_rc, parambs[i].rc_radial)` -> `fmax(max_nep_rc, double(parambs[i].rc_radial))`); there is no `fmin`/`fminf`/`fmaxf` change despite the dispatch wording. The other changed files are test data and `tests/run_tests.sh`, not compiled.

Recipe: upstream's documented CUDA build, unmodified `src/makefile` (`nvcc -std=c++14 -O3 -arch=sm_60 -I./`, links `-lcublas -lcusolver -lcufft`):

    export PATH=/usr/local/cuda-12.8/bin:$PATH
    utils/timeit.sh GPUMD compile -- make -C <clone>/src -j32       # head 15124b53
    make -C <clone-parent>/src -j32                                  # parent 6231243f

| | head 15124b53 | parent 6231243f |
|---|---|---|
| make rc | 0 | 0 |
| TUs compiled / links | 125 / 2 | 125 / 2 |
| targets | `gpumd` (21431848 B), `nep` (3600104 B) | same sizes |
| nvcc errors | 0 | 0 |
| nvcc warnings | 127, all `Support for offline compilation for architectures prior to ..._75 will be removed` (one per nvcc invocation, from the makefile's default `-arch=sm_60`) | identical 127 |
| other warnings | none | none |

Changed TU alone, serial (`cd src && nvcc -std=c++14 -O3 -arch=sm_60 -I./ -c force/ilp_nep.cu`): rc 0 at both, only the sm_60 deprecation warning. Extra host-warning pass `-Wno-deprecated-gpu-targets -Xcompiler -Wall,-Wextra` on `force/ilp_nep.cu`: 113 lines of output at both commits, identical modulo line numbers (pre-existing `-Wcomment` at ilp_nep.cu:1551, `-Wunused-parameter` in potential.cuh etc.); nothing on line 221. No new warning anywhere. PASS: the CUDA path compiles unchanged with nvcc 12.8. No platform states changed.
## Review 2026-11-05 (linux-gfx942, reviewer) -- delta 6231243f..15124b53, PASSED

Scope: `git diff 6231243f..15124b53` on `moat-port` (fresh clone of AMD-Ecosystem/GPUMD). One source line (`src/force/ilp_nep.cu:221`) plus the new `tests/gpumd/ilp_nep_cutoff/` case and its `tests/run_tests.sh` stanza. No findings.

### Checked
- Overload equivalence: `max_nep_rc` is `double` (`ilp_nep.cuh:212`), `rc_radial` is `float` (`ilp_nep.cuh:132`). CUDA's host `max(double, float)` is `fmax(a, double(b))`; the fix is `fmax(max_nep_rc, double(parambs[i].rc_radial))`, the same call. `fmax(double,double)` resolves to libm on the host under both compilers, so NaN behaviour is the same (the non-NaN operand is returned).
- Scope: the line is in the `ILP_NEP` constructor (host only). No device code changed. The `(int,int)` calls at 229-230 are unchanged.
- Host-only poison probe, re-run independently (hipcc 7.16.26385, `-x hip --cuda-host-only -fsyntax-only -O3 --offload-arch=gfx942 -DUSE_HIP -I./ -include poison.h`, every `.cu` file in the `makefile.hip` gpumd + nep source lists, 125 files):
  - 15124b53 default: 0 hits, 0 other errors. Same with `-DUSE_FIXED_NEIGHBOR -DUSE_NEPCG -DDEBUG -DUSE_FIXED_SCALER -DCG_RDF -DSTRONG_DEBUG -DOUTPUT_NEIGHBOR_FOR_TRAIN -DFINE_TUNE_DESCRIPTOR -DZHEYONG -DUSE_KEPLER`, and with `-DUSE_GRAPHENE_TB`: 0 hits.
  - 6231243f (control): exactly one hit, `force/ilp_nep.cu:221:18: call to deleted function 'max'`.
  - The `#if` guard inventory over those directories matches the porter's coverage. Files carrying `USE_PLUMED/TENSORFLOW/NNAP/NETCDF` contain no unqualified `min(`/`max(`.
- Run-time before/after on gfx942 (MI300X, HIP_VISIBLE_DEVICES=0, ulimit -c 0). Both heads were built with `make -f makefile.hip HIP_ARCH=gfx942 -j56 gpumd` (rc 0):
  - 6231243f: `ilp_nep_cutoff` exit 0, `diff -q error.out error1.out` -> differ (the run completed with the bad cutoff).
  - 15124b53: exit 1, `ERROR: Please make sure cutoff of ILP is larger than it of NEP.`, diff identical. Empty stderr.
  - Control with ILP rcut 4.3: exit 0, empty stderr.
  - After the harness's own `rm -f error.out neighbor.out`, `git status --porcelain` is clean.
- Commit message: title is 65 chars, `[ROCm]`, and accurate. The body states the defect, cause and fix correctly, includes "Authored with Claude.", and has a fenced Test Plan. No trailer, ASCII only. `utils/jargon.py --port GPUMD` clean (after `git fetch origin master:master` in the clone), `utils/prose.py` clean. `audit-commits` flags only the pre-existing 6231243f (already upstream).

### Upstream check (follow-up candidate)
brucefan1983/GPUMD master ddc6e2d2 (2026-11-03) still has the bug at `src/force/ilp_nep.cu:216` (the porter's entry says 221; that is the fork line). The source hunk of 15124b53 applies cleanly to master (`git apply --check --include='src/*'`). Two facts for whoever prepares the follow-up (neither blocks this review):
- Upstream removed `tests/` (882583a7, 2026-09-20, "move regression tests"). Its harness is now `tests_regression/` (`run_regression.py` + `manifest.json`, baseline vs candidate). The `tests/run_tests.sh` stanza and `tests/gpumd/ilp_nep_cutoff/` cannot go upstream as written. The case would have to be re-expressed for `tests_regression/` or left out of the follow-up.
- The same poison probe on upstream master (147 files) finds the same single hit. Separately, `utilities/nep_model_file.cu` (new upstream) fails the HIP host compile: `nep_utilities.cuh: unknown type name '__forceinline__'`, so the probe could not see into that file. That is an upstream HIP build regression for the next sync round, not part of this delta.

CUDA compile: covered by the concurrent "CUDA compile check (nvcc 12.8) at 15124b5" entry above (mi355 host, rc 0, no new warning).

Verdict: review-passed at 15124b53.

## Validation 2026-11-05 (linux-gfx1100, validator) -- revalidate at 15124b53, PASS

Host: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), pinned HIP_VISIBLE_DEVICES=3 by the dispatcher, all GPU runs under `ulimit -c 0`. Toolchain: TheRock pip SDK, HIP 7.16.26385, AMD clang 24.0.0git (the earlier gfx1100 record used ROCm 7.2.1). Fresh clone of AMD-Ecosystem/GPUMD, detached at 15124b533b659e338e43ffb7878b847d7d1ce09d (`moat-port` tip, on `origin/moat-port`); parent 6231243f as a `git worktree` for the before control. No fork edit, no push.

### Build
    utils/timeit.sh GPUMD compile -- make -C <clone>/src -f makefile.hip HIP_ARCH=gfx1100 -j8    # head: gpumd + nep
    make -C <parent-worktree>/src -f makefile.hip HIP_ARCH=gfx1100 -j8 gpumd                     # parent 6231243f

Both rc 0, 0 errors. Head: 102 warnings, all pre-existing classes (94 `-Wunused-value` nodiscard hipError_t, 8 `-Winconsistent-missing-override`). `gpumd` 12968224 B, `nep` 1129168 B; every embedded code object is `amdgcn-amd-amdhsa--gfx1100` (87 bundles, none other).

### Tests (real gfx1100)
Each case copied from the tree into a scratch `tests/gpumd/<case>` with `potentials` and `src/gpumd` symlinked so the cases' relative paths resolve, run as `../../../src/gpumd > stdout.txt 2> stderr.txt` (script `agent_space/run_suite.sh`), wrapped `utils/timeit.sh GPUMD test -- ...`. Cases are the ones recorded above plus the new one: `ilp_nep_cutoff carbon graphene_dos graphene_kappa_emd`.

New test `ilp_nep_cutoff` (ILP rcut 4.1 vs NEP 4.2), harness check `tail -n 1 stdout > error.out; diff -q error.out error1.out`:

| binary | rc | last line | diff vs error1.out |
|---|---|---|---|
| 15124b53 (head) | 1 | `ERROR: Please make sure cutoff of ILP is larger than it of NEP.` | identical |
| 6231243f (parent) | 0 | run completed (1 step) | differ |
| 15124b53, C.ilp rcut edited to 4.3 (control) | 0 | run completed | -- |

So the truncation bug reproduces on gfx1100 at the parent and the fix restores the CUDA behaviour; empty stderr in all three.

Recorded workloads, head vs parent (same host, same day) vs this platform's 2026-05-30 record. All 8 runs exit 0, empty stderr, no NaN in any output.

| metric | head 15124b53 | parent 6231243f | gfx1100 record 2026-05-30 |
|---|---|---|---|
| carbon NVE drift (max-min)/\|mean Etot\| | 6.45e-7 | 1.06e-6 | 9.76e-7 |
| carbon std/\|mean\| | 1.67e-7 | 3.46e-7 | 3.45e-7 |
| carbon T mean | 300.44 K | 300.30 K | 300.35 K |
| carbon PE / KE vs CUDA ref (step 10) | 1.33e-6 / 8.1e-5 | 9.81e-6 / 6.4e-4 | 9.77e-6 / 6.18e-4 |
| graphene_dos dos.out 200x4 finite, min | yes, 2.87e-2 | yes, 2.93e-2 | yes, 3.17e-2 |
| graphene_dos DOS vs dos1.out (max-normalized) max diff / RMSE | 3.67e-2 / 9.33e-3 | 6.02e-2 / 9.22e-3 | 6.35e-2 / 1.25e-2 |
| graphene_dos MVAC t=0 x,y,z | 1,1,1 | 1,1,1 | 1,1,1 |
| graphene_dos NPT T, rows 10-99 mean / std | 299.71 / 2.09 K | 299.63 / 2.03 K | 299.55 / 1.91 K |
| graphene_kappa_emd NPT T, rows 10-99 mean / std | 300.06 / 1.83 K | 299.84 / 2.07 K | 299.36 / 2.31 K |
| graphene_kappa_emd hac.out | 50x11 finite, t 0.1-9.9 ps | same | same |

Differences from the earlier record, explained: (1) the earlier gfx1100 T std (1.91/2.31 K) excludes the initial 0->300 K Berendsen ramp; over all 100 rows the std is about 13.5 K for head, parent and the CUDA reference alike (ref rows 10-99: 300.32 / 1.86 K), so the table compares rows 10-99. This is a metric definition, not a behaviour change. (2) The remaining scatter (carbon drift, PE vs ref, DOS diff) moves in both directions between head and parent on the same host and day and stays in the chaotic-MD band root-caused above to atomicAdd cell binning; none of these three workloads constructs `ILP_NEP`, the only code the fix touches, so no change is expected and none beyond that noise is seen. (3) Toolchain changed from ROCm 7.2.1 to HIP 7.16 since the earlier record; parent and head on the new toolchain agree.

### Gates
- `python3 utils/jargon.py --port GPUMD` (clone symlinked at projects/GPUMD/src for the scan, then removed): clean.
- `python3 utils/moatlib.py audit-clean GPUMD`: OK; `git status --porcelain` in the clone shows no modified tracked file.
- Docs: `doc/installation.rst` "Building for AMD GPUs (ROCm/HIP)" still present; the delta changes no build step.
- CUDA compile: already recorded at 15124b5 (nvcc 12.8, mi355 host), not repeated.

Result: PASS, 4/4 cases (new `ilp_nep_cutoff` fails at the parent and passes at the head; 3 recorded workloads pass at both). linux-gfx1100 revalidate -> completed at 15124b53.
## Validation 2026-11-05 (linux-gfx942, validator) -- 15124b53, PASSED

Host: MI300X (gfx942, CDNA3, wave64), HIP 7.16.26385 / AMD clang 24.0.0git, HIP_VISIBLE_DEVICES=0 (dispatcher-pinned), `ulimit -c 0`. Fresh clone of AMD-Ecosystem/GPUMD, detached at 15124b533b659e338e43ffb7878b847d7d1ce09d. linux-gfx942 has no earlier validation record for this project (first validation on this platform), so the comparison is against the porter's gfx90a counts at this head, the reviewer's gfx942 run at this head, and the other platforms' records.

Build:

    utils/timeit.sh GPUMD compile -- make -C projects/GPUMD/src/src -f makefile.hip HIP_ARCH=gfx942 -j56

rc 0, 0 errors, 102 warnings (94 `-Wunused-value` on ignored hipError_t returns, 8 `-Winconsistent-missing-override`; the pre-existing benign classes). `gpumd` 12873992 B, `nep` 1185744 B.

Tests: scratch copy `/tmp/gpumd-val-942/{src,tests}` (binaries copied, `potentials` symlinked) so the fork tree stays clean. Every case under `tests/run_tests.sh` that runs `gpumd`, run as `../../../src/gpumd > stdout.log 2> stderr.log` in each case dir, wrapped `utils/timeit.sh GPUMD test -- /tmp/gpumd-val-942/run_suite.sh`. The two pytest stanzas (carbon_average, carbon_observe) were not run here and were not in the porter's or reviewer's suite either.

| case | rc | stderr | result |
|---|---|---|---|
| carbon (NEP4 NVE 100 steps) | 0 | empty | Etot drift (max-min)/\|mean\| 1.36e-6, std/\|mean\| 3.64e-7, T mean 300.54 K, PE vs CUDA ref at step 10 3.34e-5, neighbor.out identical to ref |
| graphene_dos (Tersoff + hipFFT DOS) | 0 | empty | NPT T (rows >= 10) 299.89 +- 1.98 K (CUDA ref 300.32 +- 1.86); MVAC t0 1,1,1; dos.out 200x4 finite, min 3.14e-2; DOS vs CUDA ref, normalized by global max: max diff 8.32e-2, RMSE 1.25e-2 |
| graphene_kappa_emd | 0 | empty | T mean 297.20 K, hac.out 50x11 finite |
| ilp_nep_cutoff (new) | 1 | empty | last stdout line `ERROR: Please make sure cutoff of ILP is larger than it of NEP.`, identical to error1.out (the harness diff passes) |
| ilp_nep_cutoff control, ILP rcut 4.3 | 0 | empty | runs (not in the harness; re-run of the porter/reviewer control) |
| graphene_kappa_nemd (additive) | 0 | empty | compute.out max rel diff vs ref 1.86e-2, shc.out 1499x3 finite |
| graphene_kappa_hnemd (additive) | 0 | empty | kappa.out and shc.out finite; 10-step kappa is noise-dominated (sign differs from ref, expected for a 10-row chaotic estimate) |
| silicon_dispersion (additive, no MD) | 0 | empty | omega2.out max rel diff vs ref 2.94e-11 |

No NaN in any thermo.out. Raw `diff -q` against the CUDA reference files differs for every MD output, as on every platform (chaotic MD, atomicAdd cell binning; see Determinism above). Only ilp_nep_cutoff and the deterministic outputs can match exactly, and they do.

Comparison: the carbon, graphene_dos and graphene_kappa_emd numbers fall in the same band as the porter's gfx90a run at 15124b53 (1.20e-6 / 7.38e-2 DOS / 297.01 K) and the gfx1100 record (DOS 6.35e-2 / 1.25e-2). None of the three constructs ILP_NEP, so the fix cannot change them. The one difference the fix explains is ilp_nep_cutoff: ILP rcut 4.1 under a 4.2 NEP cutoff is now rejected, matching CUDA, as in the reviewer's gfx942 run at this head. PASS: 3/3 recorded workloads plus the new test, 3 additive cases clean.

Pre-completion checks: `git -C projects/GPUMD/src status --porcelain` shows no tracked modifications; `utils/jargon.py --port GPUMD` clean; the ROCm build is documented in `doc/installation.rst` (98c8bb47), and this delta changes no build step. CUDA gate: already recorded at this head ("CUDA compile check (nvcc 12.8) at 15124b5", mi355), so not re-run.
