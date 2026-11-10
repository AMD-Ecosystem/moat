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

## Validation 2026-11-05 (windows-gfx1151, validator) -- 15124b53, PASSED

jeff asked for this run. GPU: AMD Radeon 8060S (gfx1151, RDNA3.5 APU, wave32), `HIP_VISIBLE_DEVICES=0`, TheRock ROCm 7.14.0a20260519, Windows 11. Fresh depth-1 clone of AMD-Ecosystem/GPUMD `moat-port` in `agent_space/gpumd-win2`, HEAD 15124b533b659e338e43ffb7878b847d7d1ce09d (matches `git ls-remote`); `git status --porcelain`: clean (objects and binary are ignored). Nothing pushed. Each case run once, serially, `ulimit -c 0`.

### Build
    utils/timeit.sh GPUMD compile -- bash agent_space/gp_build_f77.sh
      # cd src; mingw32-make -f makefile.hip gpumd HIP_ARCH=gfx1151 LDFLAGS="-L<rocm>/lib" CC="<rocm>/bin/hipcc.exe" CFLAGS="-O3 --offload-arch=gfx1151 -DUSE_HIP -std=c++17" -j8
rc 0, `gpumd` 13.1 MB, 0 errors. Environment notes (no source or makefile change): (1) `CC=<rocm>/bin/hipcc.exe` must be given by full path, because `mingw32-make` does not find `hipcc` through the MSYS PATH. (2) The 2026-05-30 recipe built with the default language mode; with this SDK's rocPRIM, `force/neighbor.cu`, `force/potential.cu` and `force/nep_multigpu.cu` (the thrust users) fail with `rocPRIM requires at least C++17` because the Windows clang driver defaults lower, so those three were built with `-std=c++17` and the other objects with the default (the build was run as one -j16 pass that failed on those three, then re-run with `CFLAGS` carrying `-std=c++17`; make rebuilt only the failed objects). The Linux makefile comment says C++17 is the hipcc default there; an explicit `-std=c++17` in the Windows recipe is the fix and the makefile is unchanged. (3) At -j16 two hipcc launches failed with "failed to execute" (process-launch failure, not a compile error) and succeeded at -j8.

### Tests (real gfx1151)
Scratch tree `agent_space/gp_run/{src/gpumd.exe, potentials (needed files only), tests/gpumd/<case>}` with the ROCm runtime DLLs staged next to the exe and the SDK `bin` on PATH (first launch without the SDK bin on PATH exited 127: `hipfft.dll` not found; environment, fixed by PATH). Run as `../../../src/gpumd.exe > stdout.txt 2> stderr.txt` via `utils/timeit.sh GPUMD test -- bash agent_space/gp_run_suite.sh` (18 s for the four cases, 11 s of that GPU).

| case | rc | stderr | result |
|---|---|---|---|
| ilp_nep_cutoff (new) | 1 | empty | last stdout line `ERROR: Please make sure cutoff of ILP is larger than it of NEP.`; identical to `error1.out` once the CR is stripped (the checkout has CRLF `error1.out` and the bash harness `diff -q` is byte-wise, so `run_tests.sh` as written differs on this Windows checkout; `run_tests.bat` has no entry for this case) |
| ilp_nep_cutoff control, ILP rcut 4.3 | 0 | empty | runs to completion |
| carbon (NEP4 NVE 100 steps) | 0 | empty | thermo 10x18 finite; Etot drift (max-min)/\|mean\| 1.00e-6, std/\|mean\| 3.1e-7, T mean 300.37 K; PE at step 10 vs CUDA ref 4.7e-5 relative |
| graphene_dos (Tersoff, hipFFT DOS) | 0 | empty | dos.out 200x4 finite, mvac t0 1,1,1, NPT T rows >= 10 300.11 +- 2.24 K |
| graphene_kappa_emd | 0 | empty | hac.out 50x11 finite, NPT T rows >= 10 299.76 +- 2.09 K |

No NaN in any output. Against the previous record on this platform (6231243f..65b4ded, 2026-05-30: carbon ~9e-7 energy conservation, dos 200x4 and hac 50x11 finite, exit 0): the same three workloads pass with the same shapes and the same energy-conservation level (1.0e-6). The only new behaviour is `ilp_nep_cutoff`, which is the fix (ILP rcut 4.1 under a 4.2 NEP cutoff is rejected as on CUDA; it ran to completion at the parent on gfx90a and gfx1100). The parent binary was not rebuilt here, so the "fails before" half was not re-shown on this platform; it is recorded by the gfx90a, gfx1100 and gfx942 entries above. Counts: 4/4 cases plus the control, versus 3/3 previously.

### Exclusions
Not run (beyond what the earlier Windows record covered, kept out to limit load on the APU): graphene_kappa_nemd, graphene_kappa_hnemd, silicon_dispersion, the other tests/gpumd cases, the two pytest stanzas, and the `nep` executable. No case recorded as faulting or large-memory on this host was run.

### Gates
`jargon.py -` on the commit message and added lines of 15124b5: clean (the port-range scan needs a full-history clone; the Linux validators ran `--port GPUMD` clean at this head). Docs: `doc/installation.rst` ROCm section unchanged by the delta. CUDA gate: recorded at this head ("CUDA compile check (nvcc 12.8) at 15124b5"); not run on this host.

State: windows-gfx1151 `revalidate` -> `completed` at 15124b53.

## Sync round 2026-11-05 (linux-gfx942, porter) -- merge upstream c3d5d8ac, restore the HIP build, 15124b53 -> f3622149

jeff asked for this run: one follow-up PR that brings the ROCm build up to date with upstream and carries 15124b53. `pr-state GPUMD` = `merged` (#1538, normal merge; 6231243f is an ancestor of upstream master), so `moat-port` is not frozen; `protect-fork` installed on a fresh clone; the push went to `moat-port`. Host: MI300X (gfx942), HIP 7.16.26385 / AMD clang 24.0.0git, `HIP_VISIBLE_DEVICES=0` (dispatcher-pinned), all GPU work under `ulimit -c 0`.

### Merge
Upstream master fetched by URL (no remote added): **c3d5d8ac5f605095c7c3a5a3e310ccf55a899de9** (2026-11-05, "Merge pull request #1873"), 1238 commits past 6231243f. Merged into `moat-port` with a merge commit (no rebase), so #1538's history and 15124b53 are unchanged.

Conflicts: exactly one, `tests/run_tests.sh` (modify/delete: upstream 882583a7 deleted `tests/`, 15124b53 added the ilp_nep_cutoff stanza). Resolved by keeping upstream's deletion and also deleting our `tests/gpumd/ilp_nep_cutoff/` (5 files), so `tests/` is not resurrected; the case is re-expressed in `tests_regression/` below. `src/force/ilp_nep.cu` auto-merged and keeps the fmax fix (upstream line 216). After the merge the tree equals upstream c3d5d8ac plus that one line.

Merge message: the first local attempt used git's default (`Merge commit '<sha>' into moat-port`), which `jargon.py` flags. Nothing had been pushed, so the merge was redone on a fresh local branch with `Merge branch 'master' of https://github.com/brucefan1983/GPUMD` plus a one-line conflict note, and the two fix commits were cherry-picked onto it (tree byte-identical to the built and tested one; no reset, `branch -f` or force push; local branches renamed only). The discarded local commits were d8d2442e/1b06890f/4c648c54 (kept locally as `moat-port-first-merge`, never pushed).

Final `moat-port` (pushed 15124b53..f3622149, fast-forward):
- 8405f15e Merge branch 'master' of https://github.com/brucefan1983/GPUMD
- 070d3f53 [ROCm] Restore the HIP build of nep_model_file, NEP-spin and replica
- f3622149 [ROCm] Add a regression case for the ILP_NEP cutoff check

### HIP build restoration (070d3f53)
First `make -f makefile.hip HIP_ARCH=gfx942 -k -j56` on the merge: 4 TUs failed, everything else built.
1. `utilities/nep_model_file.cu` (new, host-only): `nep_utilities.cuh: unknown type name '__forceinline__'`. nvcc pre-includes cuda_runtime.h, hipcc does not, and the file includes nep_utilities.cuh before any runtime header. Fix: `#include "gpu_macro.cuh"` at the top of `nep_utilities.cuh` (it pulls `hip/hip_runtime.h` or the CUDA headers).
2. `force/nep_spin/nep_spin.cu`, `main_nep/nep_spin/nep_spin.cu`, `main_nep/nep_spin/spin_evaluate.cu` (via `force/nep_spin/nep_spin3_force.cuh`): `static assertion failed ... The mask must be a 64-bit integer` from `amd_warp_sync_functions.h`. The NEP-spin kernels (upstream 76bde565 and follow-ups) tile blocks into 32-lane warps (`warp = threadIdx.x >> 5`, `lane = threadIdx.x & 31`) with 32-bit sub-masks (`atom_mask`, `center_mask`, `row_mask`/`active_mask`). On HIP the mask must be 64-bit, and without `NDEBUG` `__hip_check_mask` traps unless the mask equals the active lanes, which for the second 32-lane tile of a wave64 wavefront are lanes 32-63. Fix: `gpuLaneMask(mask)` in `gpu_macro.cuh` -- CUDA `(mask)`, HIP `(static_cast<unsigned long long>(mask) << (__lane_id() & 32))` (shift 0 on wave32) -- wrapped around all 11 masked `__syncwarp`/`__shfl_down_sync`/`__shfl_sync` call sites. `apply_ann_one_layer_warp` (main_nep) also passed no width, so the default `warpSize` (64) would broadcast lane 0 of tile 0 to tile 1 in `__shfl_sync(.., 0)`; it now passes its `warp_size` (32) to both shuffles (equals CUDA's default). Groups exit only at tile granularity (`atom`/`n1`/`active` are per tile), so no partial-tile divergence at the sync points. Two lines re-wrapped with clang-format (ColumnLimit 100).
3. `gpumd_replica` (new `src/main_replica/`, own makefile, upstream builds it from the default `all`): missing from `makefile.hip`. Its own `utilities/gpu_macro.cuh` already maps HIP, and `make -C main_replica CC=hipcc CFLAGS="-O3 --offload-arch=gfx942 -DUSE_HIP" LIBS=-lhipfft` built it unchanged, so `makefile.hip` now has `gpumd_replica` / `clean_replica` targets mirroring the default makefile's.
Not built, as before: `main_gnep/` (not in either makefile's `all`; raw CUDA API). `main_nep/common/nep_compile.cu` (runtime nvcc specialization) is already `#if !defined(USE_HIP)` upstream, so HIP uses the generic kernels; porting the runtime specialization to hipcc is possible later work, not needed for a complete build.

Rebuild: rc 0, gpumd 14717192 B, nep 3473696 B, tb 315128 B, gpumd_replica 2401232 B; warnings only the pre-existing classes (84 `-Wunused-value` nodiscard, 12 `-Winconsistent-missing-override`). Every tracked `.cu` except the 7 in `main_gnep/` produced an object.

CUDA-visible changes (all in 070d3f53; 15124b53's one line is unchanged): the extra `#include "gpu_macro.cuh"` in `nep_utilities.cuh`; `gpuLaneMask(m)` = `(m)` around 11 masks; explicit width 32 (= CUDA default) on two shuffles in `main_nep/nep_spin/nep_spin.cu`. `makefile` untouched. No nvcc here; the mi355 nvcc check is still owed (the dispatcher runs it).

### Host min/max probe (deleted-template forced include), merged tree
`hipcc -x hip --cuda-host-only -fsyntax-only -O3 --offload-arch=gfx942 -DUSE_HIP -I./ -include poison.h <file>` over all 230 tracked `.cu` outside `main_gnep/` (main_replica files compiled from `main_replica/` with `-I./ -I./integrate`). Passes: default; `-DUSE_FIXED_NEIGHBOR -DUSE_NEPCG -DDEBUG -DUSE_FIXED_SCALER -DCG_RDF -DSTRONG_DEBUG -DOUTPUT_NEIGHBOR_FOR_TRAIN -DFINE_TUNE_DESCRIPTOR -DZHEYONG -DUSE_KEPLER`; `-DUSE_GRAPHENE_TB`; `-DUSE_CJ -DGNEP_TEST_DIAGNOSTICS` (new upstream guards). Result: 230/230 rc 0, **0 deleted-function hits** in every pass (nep_model_file.cu now reachable). Control: ilp_nep.cu with the fix reverted -> `216:18: call to deleted function 'max'`. `GPUMD_WPE_ENABLED` / `NEP_SPECIALIZED_RUNTIME_BUILD` need external/generated headers; `USE_DEEPMD/NNAP/PLUMED/NETCDF` files carry one unqualified `min(` (force/nnap.cu:409, in a `__global__` kernel, device code). No new fix needed.

### Regression case (f3622149)
`tests_regression/` conventions followed: case `ilp_nep_cutoff_below_nep` (suites invalid/potential/full, `expect: failure`, `expected_returncode: 1`, `stdout_contains: "ERROR: Please make sure cutoff of ILP is larger than it of NEP."` -- ILP_NEP prints that to stdout, not via PRINT_INPUT_ERROR; `outputs: []`, `input_style: intentional_invalid`), input `cases/bugfix/ilp_nep_cutoff_below_nep.in` (ilp_nep_compute's input minus dumps), fixture `ilp_nep_graphene_short_cutoff` = `bilayer_graphene_primitive.xyz` + new `fixtures/potentials/graphene_nep_ilp_short_cutoff.txt` (graphene_nep_ilp.txt with rcut_global 16.0 -> 4.1) + new `fixtures/potentials/bilayer_graphene_map_C_2022_NEP4.txt` + `repo:potentials/nep/C_2022_NEP4.txt` (cutoff 4.2 3.7; the existing C_2024_NEP4 cutoff 7 is an integer, so it cannot show the truncation). Fixture hashes pinned in `test_manifest.py`, case added to its behaviour-contract list, counts 255 -> 256 in README.md and CONTRACT_SUITES.md. The dispatch said "ILP rcut 4.2 with NEP 4.0"; the bug needs a non-integer NEP cutoff above the ILP one, so the case is ILP 4.1 under NEP 4.2. The harness expresses expected-error cases natively; no harness change.
- `--check-manifest`: `Manifest OK: 256 cases, 24 suites`; `test_manifest.py`: 17 pass; `unittest discover`: 29 OK.
- After the fix: PASS (both roles exit 1 with the message, no outputs). Fix reverted (ilp_nep.cu `max(...)`, rebuilt gpumd in a scratch copy): `FAIL: ilp_nep_cutoff_below_nep:baseline: expected exit 1, got 0` -- stdout "1 steps completed.", neighbor.out written.

### Upstream suites on gfx942
Builds for the suites: scratch copies of the tree (`agent_space/gpumd/dbg2`), `make -f makefile.hip gpumd nep CFLAGS="-O3 --offload-arch=gfx942 -DUSE_HIP -DDEBUG"`, because upstream's integration CI builds with `-DDEBUG` (fixed seeds) and the README requires it for nep. Self-comparison as the README prescribes (same binaries as baseline and candidate):

    python3 tests_regression/run_regression.py --repo-root $R --baseline $D/gpumd --candidate $D/gpumd --baseline-nep $D/nep --candidate-nep $D/nep --suite full

| run | cases | relations | failures |
|---|---|---|---|
| A: release gpumd (aborted at 125 cases) | 117/125 | -- | 7 clock-seeded RNG cases (nvt_bdp, npt_scr_iso, heat_bdp, pimd_scr, deform_npt_tri, replicate_default_velocity, replicate_velocity_unseeded): `mt19937(now)`/`srand(now)` unless `-DDEBUG` -- expected, wrong build |
| B: DEBUG (616 s) | **249/256** | 25/25 | replicate_velocity_unseeded, graphene_hac, graphene_nemd_shc, graphene_hnemd_shc, silicon_liquid_transport (byte diffs), qnep_pppm_future_bec (thermo 1.3e-7 vs atol 1e-7), predict_nep_pbte_batch (force 2.0e-7 vs atol 2e-7) |
| B rerun of those 7 with srand shim | 7/7 | -- | none |
| C: DEBUG + srand shim, concurrent with a pytest run on the same GPU (675 s) | 247/256 | 21 pass, 4 skipped (members failed) | float-atomic noise only: qnep_pppm_future_bec (bec 1.06e-5 vs atol 1e-5), 7 pppm_* thermo (1.2e-7..5.1e-7 vs atol 1e-7), observer_average_then_observe_control (last ulp of one force) |

The new case passed in every run. Every one of the 256 cases passed in at least one of B/C, and all 25 relations passed in B.

Root cause of the B byte diffs (AMD runtime, not the port; corrected after the 2026-11-05 gfx1100 review): on the first kernel launch the HIP runtime (CLR) compiles its internal blit kernels through comgr; each comgr action creates a temp directory via LLVM `createUniqueDirectory` -> `createUniquePath` -> `sys::Process::GetRandomNumber`, which calls `srand(<random seed>)` once and then `rand()` (38 times) because TheRock's libLLVM.so.24.0git was built without `arc4random` (glibc < 2.36 at LLVM build time; its newest glibc symbol is GLIBC_2.27). Not GPUMD's code objects loading: `rand()` after `hipMalloc` is still the default stream. Ubuntu 24.04's libamd-comgr2 imports `arc4random` and does not do this. GPUMD's `srand` (or DEBUG's implicit seed 1) in `print_compile_information()` precedes the first HIP call, so the reseed overrides it. GPUMD's DEBUG reproducibility assumes the default C `rand()` stream (`velocity` without `seed`, `add_random_force`, curand seeds in lan/bao/pimd/ti...). With a diagnostic shim making `srand` a no-op, two runs give identical output and all 7 pass. Registered as deferrals `gpumd-hip-runtime-reseeds-rand` (fix in GPUMD or accept) and `comgr-llvm-srand-perturbs-host-rand` (candidate ROCm report) for a person to rule. Pitfall hit while diagnosing: `dump_xyz` appends, so rereading frame 1 after repeated runs in one directory looks "deterministic per directory".

Float-noise failures: README says the PPPM/qNEP numeric bounds "must be checked by self-comparison on the target GPU"; on MI300X the float-atomic noise sits at or just above them (worst 5.1e-7 vs 1e-7 under a concurrent GPU load) and varies run to run. AMD baseline difference in atomic ordering, not a regression; widening a bound would need upstream's agreement and is not proposed in this PR.

`tests_pytest` (calorine, ase installed in the job venv): NEP-spin tests `test_spin_*.py` -- 110/114 with the release build, the 4 `test_corrector_preserves_force_modifiers` failing on `add_random_force` streams (same rand() cause); 114/114 with the DEBUG build + srand shim. These include the FP64 oracle checks of energy/force/virial/mforce against TorchNEP references, which exercise the masked warp kernels on wave64. Rest of the suite, `-m "not slow"`, release build: **692 passed, 480 skipped** (skips: missing qnep/netCDF fixtures for some structures, netCDF4 module), 0 failed.

`tests_regression/replica/test_replica.py` (gpumd_replica, release): 6/12 pass; the 6 failures all stop at `REMD/PRD explicit streams support energy NEP4/NEP5 and qNEP4 models` because upstream dd7d0ba8 (2026-10-23, "update example model files") moved `examples/nep_train/nep.txt` and `examples/qnep_train/nep.txt` to the `format_version 1` header that `main_replica` does not recognise. Upstream state, platform-independent (fails the same way on CUDA); not touched.

Comparison with 15124b53: the old `tests/gpumd/` workloads (carbon, graphene_dos, graphene_kappa_emd...) no longer exist; their successors in tests_regression (e.g. graphene_hac, graphene_nemd_shc) are now byte-compared baseline vs candidate rather than against CUDA reference files, and pass in run C. The ILP_NEP behaviour matches the 15124b53 records on gfx90a/gfx1100/gfx942/gfx1151 (rejected after, accepted before).

### Gates
`python3 utils/jargon.py --port GPUMD`: 2 hits, both in upstream's own 725fe762 "Merge pull request #1538 from jeffdaily/moat-port", which is in upstream master and in range only because the fork's `master` mirror is at c5597219 (2026-05-27); the dispatcher owns the mirror. Against the real base, `jargon.py -C src --commits c3d5d8ac..moat-port --diff c3d5d8ac...moat-port`: clean. `prose.py` on both commit bodies: clean. `audit-clean`: OK; `git status --porcelain` clean. `audit-commits` asks for the mirror to be fast-forwarded first. All platforms now read revalidate at f3622149. Skill lessons added: fault-classes (64-bit masks / lane shift for 32-lane tiles), validation (runtime reseeds `rand()`).

### Draft follow-up PR for jeff (title and body; not posted)

Title: `[ROCm] Restore the HIP build and fix a host max() truncation in ILP_NEP`

Body:

This follow-up to #1538 fixes a HIP-only bug in the ILP_NEP setup and brings the HIP build back in line with current master.

ILP_NEP: under HIP, the host call `max(max_nep_rc, parambs[i].rc_radial)` (double, float) resolved to the only host overload, `max(int, int)`, so a 4.2 A NEP radial cutoff became 4 and an ILP cutoff of 4.1 was accepted instead of rejected as on CUDA. It now calls `fmax`, which is what CUDA resolves the call to. A new `tests_regression` case, `ilp_nep_cutoff_below_nep`, checks the rejection; it fails with the old line and passes with the fix.

HIP build: `utilities/nep_model_file.cu` reached `nep_utilities.cuh` without a runtime header (nvcc pre-includes one, hipcc does not), so the header now includes `gpu_macro.cuh`. The NEP-spin kernels pass 32-bit masks to `__syncwarp` and `__shfl_*_sync`; HIP requires 64-bit masks, and on a 64-lane wavefront the second 32-lane tile sits in lanes 32-63. A `gpuLaneMask()` macro in `gpu_macro.cuh`, the identity on CUDA, moves the mask to the right lanes, and one warp kernel now passes its 32-lane width to its shuffles explicitly. `makefile.hip` also builds `gpumd_replica` like the default makefile. The CUDA build sees only the extra include, the identity macro and an explicit width equal to the default.

Testing on an AMD Instinct MI300X (gfx942) with `make -f makefile.hip HIP_ARCH=gfx942`: gpumd, nep, gpumd_replica and tb build. The full `tests_regression` suite (built with `-DDEBUG`, self-comparison) and the NEP-spin tests in `tests_pytest`, including the FP64 reference checks, pass. Cases that depend on the default `rand()` stream can differ between runs on some ROCm builds, because the runtime reseeds the C library's `rand()` when it prepares its internal kernels on the first launch; they pass when that reseed is suppressed.

(Superseded by the draft in "Fix round 2026-11-05 (linux-gfx942, porter)" below. Before presenting: replace the test paragraph with the final counts from every gate's revalidation at the PR head, the mi355 nvcc compile result, and jeff's ruling on the rand() deferral; the body's "pass" for tests_regression is per case across runs, see the table above.)

## CUDA compile check (nvcc 12.8) at f3622149

Helper run on the mi355 fleet host (CUDA 12.8 toolkit, V12.8.93; no NVIDIA GPU), compile only, requested by jeff. Fresh clone of the fork's `moat-port` at **f3622149** vs `git worktree add ../up c3d5d8ac` (upstream master merged by 8405f15e). CUDA-visible delta between the two trees: 070d3f53 (`#include "gpu_macro.cuh"` in `utilities/nep_utilities.cuh`; `gpuLaneMask()` around 11 masks -- 4 in `force/nep_spin/nep_spin.cu`, 6 in `force/nep_spin/nep_spin3_force.cuh`, 2 in `main_nep/nep_spin/nep_spin.cu`; explicit width `warp_size` (32) on the two shuffles in `apply_ann_one_layer_warp`) and the carried 15124b53 host line in `force/ilp_nep.cu` (`fmax`). f3622149 itself touches only `tests_regression/`.

Recipe: upstream's unmodified `src/makefile` (default `all` = gpumd nep gpumd_replica tb; `nvcc -std=c++14 -O3 -arch=sm_60 -Xcompiler="<WARN_FLAGS: -Wall -Wextra ...>" -I./`), both trees built concurrently with `-O` (per-target output grouping):

    export PATH=/usr/local/cuda-12.8/bin:$PATH
    utils/timeit.sh GPUMD compile -- make -C <clone>/src -j64 -O      # f3622149
    make -C <clone-up>/src -j64 -O                                    # c3d5d8ac

| | f3622149 | c3d5d8ac |
|---|---|---|
| make rc | 0 | 0 |
| nvcc invocations (compile / link) | 230 / 4 | 230 / 4 |
| objects | 230 `.o` (incl. `main_replica/`) | same 230 |
| executables | `gpumd` 24730448 B, `nep` 8004496 B, `tb` 1552152 B, `main_replica/gpumd_replica` 3777440 B | same sizes |
| nvcc errors | 0 | 0 |
| nvcc warnings | 234, all `Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed` (one per invocation, from `-arch=sm_60`) | identical 234 |
| host compiler warnings (makefile's -Wall -Wextra set) | none | none |

Whole logs, sorted and path-normalized: identical, so per-file diagnostics are identical. Changed TUs compiled alone with the makefile's flags (`force/ilp_nep.cu`, `force/nep_spin/nep_spin.cu`, `main_nep/nep_spin/nep_spin.cu`, `main_nep/nep_spin/spin_evaluate.cu` (includes `nep_spin3_force.cuh` via `spin_training_kernels.cuh`), `utilities/nep_model_file.cu`): rc 0 at both, one line of output each (the sm_60 notice), nothing else.

Device code. No sm_60-capable `nvdisasm` on the host (CUDA 12.8 here ships without it; the only one present is 13.1, which refuses SM60), so instead of SASS text: `cuobjdump -xelf all` / `-xptx all` on all 230 objects of each tree, then a SHA-1 of every `.text.*`, `.nv.constant*`, `.nv.shared*`, `.nv.global*` and `.nv.info*` section of each cubin plus the full PTX. Raw objects never match byte-for-byte because nvcc's per-compile anonymous-namespace / internal-linkage hashes (`_GLOBAL__N__<8hex>_`, `_INTERNAL_<8hex>_`) land in symbol names; after normalizing those two tokens: **230/230 objects have identical cubin sections (841 kernel `.text` sections) and identical PTX**. That covers every file 070d3f53 reaches: `force/nep_spin/nep_spin.o` (43 kernels, 118k PTX lines), `main_nep/nep_spin/spin_evaluate.o` (31), `main_nep/nep_spin/nep_spin.o` (2, incl. `apply_ann_one_layer_warp<199>`; its shuffles are `shfl.sync.{down,idx}.b32 ..., %r35, %r37` with `%r35 = 31` (clamp for width 32) and `%r37 = -1` (mask), the same as upstream's default-width call), `force/ilp_nep.o` (11, host-only change) and the 35 TUs that include `nep_utilities.cuh`. Host `.o`/executable bytes differ (the hashes above, and `__LINE__` values in nep_spin.cu error macros shifted by the two re-wrapped lines; ilp_nep's `fmax`), as expected.

`gpuLaneMask()` on CUDA: `nvcc -E -Xcompiler -P` of the three NEP-spin TUs at both trees; no `gpuLaneMask` token survives, and apart from `__LINE__` values the only differences are `__syncwarp(m)` -> `__syncwarp((m))`, `__shfl_down_sync(m, ...)` -> `__shfl_down_sync((m), ...)` (plus the two re-wrapped statements) and the two added `, warp_size` arguments, i.e. the macro expands to the original constant/variable mask (`0xffffffff`, `atom_mask`, `center_mask`, `active_mask`). `utilities/nep_model_file.cu` preprocesses identically at both trees (30153 lines): on CUDA `gpu_macro.cuh` was already in by the time `nep_utilities.cuh` is reached, so the new include is a no-op there.

Not covered: `main_nep/nep_spin/spin_specialized.cu` (only under `NEP_SPECIALIZED_RUNTIME_BUILD`, compiled at runtime by `nep_compile`); it includes the same `nep_spin3_force.cuh`, whose expansion is shown above. PASS: the CUDA path builds identically with nvcc 12.8 and produces identical device code at f3622149 and upstream c3d5d8ac. No platform states changed.

## Review 2026-11-05 (linux-gfx1100, reviewer) -- sync round 15124b53 -> f3622149, CHANGES REQUESTED

Scope: `moat-port` at f3622149 = 8405f15e (merge of upstream master c3d5d8ac) + 070d3f53 + f3622149. Upstream master tip re-fetched by URL during the review: still c3d5d8ac. Host: RX 7900 XTX (gfx1100, wave32), HIP 7.16.26385 / AMD clang 24.0.0git (TheRock pip SDK), `HIP_VISIBLE_DEVICES=0`, `ulimit -c 0`.

### Must fix

1. **`src/main_replica/force/ewald.cu:138-141`: async H2D copies from freed pageable host vectors.** `Ewald::find_k_and_G` fills four local `std::vector<float>` (`cpu_kx/ky/kz/G`), enqueues `copy_from_host_async` (a `gpuMemcpyAsync`, `main_replica/utilities/gpu_vector.cuh:138-141`) from each on `stream`, and returns with no synchronisation, destroying the vectors. It runs every step from `Ewald::find_force` (line 262), reached from the replica NEP-Charge Ewald path (`main_replica/force/nep_charge.cu:1403,1679`). CUDA stages pageable async copies before returning, so the CUDA build happens to work. HIP does not, so the copy can read freed memory and upload corrupt k-vectors and G (cuda-to-rocm fault class "cudaMemcpy*Async from a soon-freed pageable host buffer"). The file is new upstream code (since 6231243f), and 070d3f53 brings it into the HIP build by adding `gpumd_replica` to `makefile.hip`, so the HIP port now ships it. Fix in shared code, not HIP-only: use the synchronous `copy_from_host(ptr, num_kpoints)`, or `CHECK(gpuStreamSynchronize(stream))` after the four copies. Keeping the vectors as members without a sync is not enough, because the next step's host refill could race the previous copy. Then run a replica NEP-Charge case that selects Ewald on a GPU. The porter's replica run stopped before any force evaluation (the `format_version 1` model issue), so this path has no HIP evidence yet. Every other `*_async` site in `main_replica` was checked and uses pinned buffers or syncs first (`replica_common.cu:246,258,361,529`, `replica_prd.cu:660,801,841-851,901`, `force/nep.cu:442`, `force/nep_charge.cu:1289,1565`, `minimize/minimizer.cu:124,146`, `minimize/minimizer_fire.cu:90`). The main `src/` tree has no async memcpy.

2. **The rand() diagnosis in the notes, both deferrals and the draft PR body is wrong about the trigger, and silent about which builds are affected.** Reproduced independently with a 10-line HIP program (`hipMalloc`, one kernel launch, `rand()` before and after) and an `LD_PRELOAD` shim that logs a `backtrace()` on `srand`:
   - Trigger: not the loading of GPUMD's code objects. It happens at the **first kernel launch**, when the HIP runtime (CLR) builds its own internal blit kernels from OpenCL C source through comgr. With `AMD_COMGR_EMIT_VERBOSE_LOGS=1` the actions are `COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC` (`-x cl -std=cl1.2`), `LINK_BC_TO_BC`, `CODEGEN_BC_TO_RELOCATABLE` and `LINK_RELOCATABLE_TO_EXECUTABLE`. Each creates `/tmp/comgr-<pid>-<n>-<6 random hex>` through `llvm::sys::fs::createUniqueDirectory` -> `createUniquePath` -> `llvm::sys::Process::GetRandomNumber`. Backtrace: `hipLaunchKernel` -> libamdhip64 -> `amd_comgr_do_action` -> libLLVM.so.24.0git `GetRandomNumber` -> `srand`. `rand()` after `hipMalloc` is still glibc's first default value (1804289383); after the launch it differs on every run. The shim counts one `srand(<random seed>)` and 38 `rand()` calls (50 with `--offload-compress`). `AMD_COMGR_USE_VFS=0/1` and `AMD_COMGR_CACHE=1` (cold or warm) do not avoid it.
   - Which builds: LLVM's `GetRandomNumber` calls `arc4random()` when the build finds it, and otherwise `srand(GetRandomNumberSeed())` once followed by `rand()`. glibc added `arc4random` in 2.36. The TheRock SDK here imports `srand` and `rand` and has no `arc4random` (`nm -D libLLVM.so.24.0git`), which fits a manylinux (glibc 2.28) build. Ubuntu 24.04's own `libamd-comgr2` (comgr 2.6 on the distro `libLLVM-17.so.1`, built against glibc 2.39) imports `arc4random` and not `srand`, so it never touches the C PRNG. The behaviour therefore depends on the glibc that comgr's LLVM was built against, not on a ROCm version number. Not verified on this host: the AMD repo packages (`/opt/rocm`, comgr statically linked to LLVM). The test is `nm -D --undefined-only libamd_comgr.so* | grep -w -e srand -e arc4random`.
   - Effect on GPUMD: `print_compile_information()` (`utilities/main_common.cu:30-35`; `srand(time)` only when DEBUG is off, no `srand` at all under DEBUG) runs before `print_gpu_information()`, the first HIP call (`main_gpumd/main.cu:33-34`). The runtime's reseed therefore overrides both. Under DEBUG it replaces the implicit seed 1 with a random one. In release builds it replaces one random seed with another, which is harmless. `main_replica` calls `srand(1)` under DEBUG (`main_replica/utilities/main_common.cu:31`) and is overridden the same way. `velocity` with an explicit seed calls `srand(seed+...)` itself after GPU init and is unaffected.
   - Action for the porter: correct the "Root cause" paragraph and both deferral summaries (`gpumd-hip-runtime-reseeds-rand`, `comgr-llvm-srand-perturbs-host-rand`). Component: LLVM `sys::Process::GetRandomNumber`, reached through comgr's temp directories while CLR compiles its blit kernels. The report belongs with comgr or TheRock (build LLVM so that `arc4random` or `getrandom` is used), or with LLVM (a library should not reseed the process-global C PRNG). Reword the draft PR body's "because the HIP runtime itself calls `srand()` while loading kernels" accordingly, e.g. "because some ROCm builds reseed the C library's `rand()` when the runtime prepares its internal kernels on the first launch". Whether to change GPUMD or file the ROCm report remains a person's ruling (the deferrals stand).

3. **Draft PR test paragraph and 070d3f53's Test Plan.** The commit's Test Plan runs `cd src` and `make`, then uses `--repo-root "$PWD" --baseline "$PWD/src/gpumd"`, which only works from the repository root. Run as written, it fails. The commit is published and must not be rewritten, so fix it where it can be fixed: the PR body's test section should give working commands (`cd src && make -f makefile.hip ...`, then `cd ..` before `run_regression.py`). The PR body's own pending update (final counts per gate) is already noted by the porter.

### Analysis corrections (no code change)
- Porter notes, 070d3f53: "Groups exit only at tile granularity (`atom`/`n1`/`active` are per tile)" is wrong. In `find_potential_warp` (`force/nep_spin/nep_spin.cu:651`, `if (atom >= N) return;`), `find_force_angular_warp` (`n1 >= N`) and `accumulate_spin3_oc_native_forces` (`!active`), the exit is per atom group (`lanes_per_atom`/`edge_lanes`/`EdgeLanes` lanes), not per 32-lane tile. The code is still correct. Each mask (`atom_mask`, `center_mask`, `row_mask`) covers exactly its own group's lanes, so the masks at each call partition the active lanes the way `__hip_check_mask` requires (`amd_warp_sync_functions.h`: each mask must equal `__ballot(true)` of the lanes passing it). `row_mask` covers exactly the `edge_lane < C` lanes that reach `__syncwarp(gpuLaneMask(active_mask))` (`nep_spin3_force.cuh:343`; no return inside `build_spin3_oc_center_pull_row`).

### Verified (for the record)
- (1) Merge: `git show --cc 8405f15e` has no combined hunk. `git diff c3d5d8ac 8405f15e` is exactly the `ilp_nep.cu:216` fmax line, and the merged tree has no `tests/` entries. The only port change in 6231243f..15124b53 is 15124b53 (the fmax line plus the old `tests/` case), so nothing was lost. The old case is re-expressed in f3622149. Of #1538's port (`doc/installation.rst`, `src/makefile.hip`), only `makefile.hip` changed upstream since (spin/nep subdirectory wildcards, `tb`), and both are in the merged tree.
- (2) 070d3f53. `gpuLaneMask` on HIP, `mask << (__lane_id() & 32)`: blocks are 1-D with 128 threads (`<<<..., 128>>>`) and tiles are `threadIdx.x >> 5`, so tile parity equals bit 5 of the hardware lane on wave64, for any 1-D block size, including a partial last wavefront. On wave32 the shift is 0, and HIP also masks to the low 32 bits (`__hip_adjust_mask_for_wave32`). On CUDA it is `(mask)`. The mi355 nvcc record shows identical PTX/cubin for all 230 objects. The explicit width 32 in `apply_ann_one_layer_warp` equals CUDA's default, and on wave64 without it `__shfl_sync(.., 0)` would broadcast tile 0's lane 0 to tile 1. The unwrapped `__syncwarp()` at `nep_spin.cu:738` is reached by every lane, before the `n1 >= N` return. `nep_utilities.cuh` including `gpu_macro.cuh` is a no-op on CUDA (the nvcc record shows identical preprocessing). The `makefile.hip` `gpumd_replica`/`clean_replica` targets mirror the default makefile. Overriding `CFLAGS` drops `-std=c++14` and the warning flags, which is correct for hipcc.
- (3) Fault-class sweep of `git diff 6231243f c3d5d8ac -- 'src/*.cu' 'src/*.cuh'` (519 files), read-only, two parallel passes:
  - Warp and wave: no added `warpSize` use or ballot/popc. All 44 new shared-memory tree reductions keep `__syncthreads()` inside the loop. `find_charge_self_energy<<<..., 32>>>` is per-thread only. No cooperative groups, inline asm, `__smid` or `__ldg`. The only early return before a `__syncthreads` (`ensemble_pimd.cu:597`) is block-uniform. Every changed `.cu` outside `main_gnep/` is compiled by `makefile.hip` or by `main_replica/makefile`.
  - Memory and API: no uninitialised accumulator (each new atomic or `+=` target is filled, memset or written first). No one-past reads. Every raw CUDA API, cuFFT, cuRAND or CUB use is behind `USE_HIP` with a HIP or rocPRIM branch. `__CUDA_ARCH__ < 600` gates fall through correctly. No `__fdividef`, `__fsqrt_rn` or CC-keyed dispatch. Handle lifetimes are guarded. The only finding is item 1.
- (4) New case on gfx1100, DEBUG build of f3622149 (`make -f makefile.hip -j16 gpumd nep HIP_ARCH=gfx1100 CFLAGS="-O3 --offload-arch=gfx1100 -DUSE_HIP -DDEBUG"`, 27 min, rc 0): `run_regression.py --suite full --case ilp_nep_cutoff_below_nep` gives **PASS**. Same tree with `ilp_nep.cu:216` reverted to `max(max_nep_rc, parambs[i].rc_radial)` (only `ilp_nep.o` rebuilt): **`FAIL: ilp_nep_cutoff_below_nep:baseline: expected exit 1, got 0`**.
- (5) `jargon.py --port GPUMD` (after creating a local `master` at the fork mirror `origin/master` = c5597219): exactly 2 hits, both `725fe762 Merge pull request #1538 from jeffdaily/moat-port`, an ancestor of c3d5d8ac (upstream's own merge). `jargon.py -C projects/GPUMD/src --commits c3d5d8ac..moat-port --diff c3d5d8ac...moat-port`: clean. Titles are 57, 68 and 65 characters with the `[ROCm]` prefix, with no Co-Authored-By trailer and with "Authored with Claude." The merge title is git's standard upstream-merge form.

### Tests on gfx1100 (DEBUG build, self-comparison)
- `tests_regression --suite full`: **stopped at 178/256 after 60 min**. The graphene transport cases are FP64-heavy and very slow on this GPU, and the 3 h job limit did not allow the remainder. Failures in the 178: `qnep_ewald_future_bec` (dpdt.out 1.0e-6 vs atol 2e-7) and `qnep_pppm_future_bec` (bec 1.77e-5 vs atol 1e-5), the float-atomic noise class the porter saw on gfx942 (the ewald one is new on this list, same class), plus `replicate_velocity_unseeded` and `graphene_hac`, both rand()-dependent per item 2. All other started cases passed, including the new case.
- NEP-spin pytest (`tests_pytest/test_spin_*.py`, 7 files): **110 passed, 4 failed** (1053 s). The 4 failures are the `test_corrector_preserves_force_modifiers` cases (sib and glsd, small and large box) that depend on the `add_random_force` `rand()` stream. Rerun with an `LD_PRELOAD` shim that turns `srand` into a no-op: **4 passed**, so 114/114 overall. This exercises the `gpuLaneMask` kernels on wave32 with HIP's mask checks live: no `NDEBUG`, so `__hip_check_mask` would trap on a bad partition. The porter covered wave64 on gfx942.

### Recommendation
**Request Changes**: item 1 (code fix plus a replica Ewald GPU run), item 2 (corrected diagnosis in notes, deferrals and PR body), and item 3 (PR body test commands).

## Fix round 2026-11-05 (linux-gfx942, porter) -- review items 1-4, f3622149 -> 51018103

jeff asked for this run, answering the gfx1100 review above. Host: MI300X (gfx942), HIP 7.16.26385 (TheRock pip SDK), `HIP_VISIBLE_DEVICES=0` (dispatcher-pinned), GPU work under `ulimit -c 0`. `orient.sh` completed normally (no hang) and named porter; `set-state GPUMD linux-gfx942 porting` took the lock. Fresh fork clone, `protect-fork` installed. `pr-state` = merged, so `moat-port` is not frozen; pushed f3622149..51018103 (fast-forward). Telemetry note: one `compile` entry with exit 2 / 0.006 s is my mis-invocation (timeit.sh cds to the MOAT root), not a build failure.

### 1. Ewald async copies (51018103, `[ROCm] Wait for the Ewald k-vector uploads before freeing their source`)
`src/main_replica/force/ewald.cu`: `CHECK(gpuStreamSynchronize(stream));` after the four `copy_from_host_async` calls in `find_k_and_G`, before the local vectors die. Shared code, CUDA-visible: same uploaded values; the force call now waits for these copies and the work queued before them.

Sweep for the same pattern (async copy from a pageable host buffer that dies or is reused before a sync): every `*Async`/`*_async` site in the HIP build (`grep -rn 'Async\|_async'` over `src/` minus `main_gnep/`). Only `main_replica` has any; the main `src/` tree has none. All other sites are fine: `replica_prd.cu:660` (stream.synchronize right after), `:801` (sync in the next loop before reading), `:841/847` (pinned `hot_position_`, sync), `:901` (sync loop after), `replica_common.cu:246` (sync), `:258` (`GPU_Pinned_Vector`, callers sync), `:361/529` (event synchronize before reading), `force/nep.cu:442`, `force/nep_charge.cu:1289,1565`, `minimize/minimizer.cu:124,146`, `minimizer_fire.cu:90` (all `gpuStreamSynchronize` before reading; D2H into members). `fill_async` is a memset. No other instance, so one commit.

Builds (scratch copies under `agent_space/gpumd/`; the release one is the literal PR-body command in the fork clone): `make -C src -f makefile.hip HIP_ARCH=gfx942 -j16` rc 0, 49 s, 230 objects, gpumd 14717192 B, nep 3473696 B, tb 315128 B, gpumd_replica 2401472 B (+240 B for the sync); warnings only the known classes (88 `-Wunused-value`, 12 `-Winconsistent-missing-override`).

Reaching `find_k_and_G`: upstream's replica examples are unusable since dd7d0ba8 (`format_version 1` header), so the case uses `examples/gpumd_static_qnep/model.xyz` (250 atoms, Pb/Te) with the legacy-header qNEP model `git show eda8e04b:examples/qnep_train/nep.txt` (`nep4_charge2`) and `potential nep.txt / time_step 1 / ensemble nvt_bdp 300 400 100 / multi_replica remd replicas 2 exchange 10 temp 300 400 / kspace ewald / run 50` (`remd_restart.meta` records `kspace ewald`; Ewald `find_force` runs every step). Commands are in 51018103's Test Plan, run as written from the fork clone root: rc 0.
- Repeatability: DEBUG `gpumd_replica` builds of old (f3622149) and new code (`make -C src/main_replica CC=hipcc CFLAGS="-O3 --offload-arch=gfx942 -DUSE_HIP -DDEBUG" LIBS=-lhipfft`), 3 runs each: `remd_thermo.out` and `remd_replica_0_restart.xyz` byte-identical across all 6 runs (md5 5e17ceef... / 42f64256...), so old and new give the same result. Release builds seed from the clock, so they differ run to run by design (step-0 energy -930.7942203399289 in both).
- Freed-memory poisoning: an `LD_PRELOAD` `free()` that fills each block with 0xFF (`malloc_usable_size`) before releasing it: old and new, 3 runs each, still byte-identical to the unpoisoned runs. The old code does NOT read freed memory on this runtime, and this is why: a probe (`clock64()` spin kernel queued on the stream, then a pageable H2D `hipMemcpyAsync`, then overwrite the source immediately, then sync and compare) shows HIP 7.16 on gfx942 stages copies of at most 64 KiB at call time (call returns in ~10 us with the stream still busy, data correct) and blocks until the stream drains for anything larger (16384 floats: 0.014 ms, stream busy; 16385 floats: 95 ms, stream idle). Neither reads the source after return, so the hazard is latent on this runtime; the fix stands on the API contract (HIP documents no such guarantee, and the skill's CV-CUDA case saw the corruption). Lesson added to `references/fault-classes.md` (a clean run is not evidence; the probe recipe).
- Upstream `tests_regression/replica/test_replica.py` with the release `gpumd_replica`: as-is 6/12 (the same 6 `format_version 1` failures as recorded). In a scratch copy with `examples/nep_train/nep.txt` from cd7f757d (`nep4`) and `examples/qnep_train/nep.txt` from eda8e04b: **12/12 OK** (46.8 s), including `test_qnep_comments_and_restart_solver_validation` REMD and PRD with `ewald` and `pppm`. So the replica Ewald path now has HIP evidence on gfx942.

### 2. rand() diagnosis corrected
Rewrote the "Root cause" paragraph and the old draft sentence above, both deferral summaries, and the skill's `references/validation.md` section, per the reviewer's reproduction. Confirmed here independently: `nm -D --undefined-only` on TheRock's `libLLVM.so.24.0git` shows `rand srand` and no `arc4random` (newest glibc symbol GLIBC_2.27); `libamd_comgr.so.3.5.0` imports `rand`; a 15-line HIP program prints `rand()` = 1804289383 (glibc's seed-1 first value) after `srand(1)` following `hipMalloc`, and a different value on every run after `srand(1)` plus the first kernel launch. `utils/deferred.py` has no command that edits a summary, so the two `summary` fields in `projects/GPUMD/deferred.json` were rewritten with a json load/dump; ids, status, timestamps and `decided: null` are unchanged, and both stay unruled for jeff.

### 3/4. Tests (gfx942), commands as in the PR body, run literally from the fork clone root
- `python3 -m pytest tests_pytest/test_spin_*.py` (release, job venv with calorine 4.1, ase 3.29.0, pytest 7.3.2, hypothesis): **110 passed, 4 failed** in 95 s; the 4 are `test_corrector_preserves_force_modifiers` (sib/glsd x small/large box). With the `srand` no-op shim: 4 passed.
- `tests_regression --suite full`, DEBUG build, self-comparison, completed both times:

| run | cases | relations | time | failures |
|---|---|---|---|---|
| 1 (scratch copy of the same tree; GPU briefly shared with the short probes at its start) | **250/256** | 25/25 | 604 s | 5 rand(): replicate_velocity_unseeded, graphene_hac, graphene_nemd_shc, graphene_hnemd_shc, silicon_liquid_transport; noise: qnep_pppm_future_bec (thermo 1.085e-7 vs atol 1e-7) |
| 2 (fork clone, literal PR commands, GPU alone) | **248/256** | 21 pass, 4 skipped (observer_* relations; member case failed) | 705 s | the same 5 rand(); noise: qnep_pppm_future_bec (thermo 1.014e-7), pppm_round_5_7 (thermo 1.026e-7 vs 1e-7), observer_average_then_observe_control (one force component differs by 1.1e-16, last ulp) |

Classification by rerunning the failures with `--case`, twice each: the 5 rand() cases fail 2/2 plain and pass 2/2 under the shim (rand() class, item 2). `pppm_round_5_7` and `observer_average_then_observe_control` pass 2/2 plain (float-atomic noise). `qnep_pppm_future_bec` fails 2/2 plain (bec 1.204e-5 vs atol 1e-5; thermo one unit in the 6th printed decimal of -16011.045805) and 2/3 under the shim (bec 1.19e-5, 1.08e-5; one pass), each time on a different field at ~7e-7 relative: not rand(), float-atomic PPPM/BEC noise sitting at the absolute bound, failing in 6 of its 7 runs on MI300X. The README says those bounds "must be checked by self-comparison on the target GPU"; the case is in main `gpumd`, untouched by this round, and the prior round saw the same. Widening a bound needs upstream's agreement; not proposed. The new ILP_NEP case passed in both runs. No failure is attributable to the port.

### Gates
`python3 utils/jargon.py --port GPUMD`: the 2 known hits in upstream's 725fe762 only. `jargon.py -C projects/GPUMD/src --commits c3d5d8ac..moat-port --diff c3d5d8ac...moat-port`: clean. `prose.py` on 51018103's body and the PR body: clean; `jargon.py -` on the PR body: clean. Title 70 chars, "Authored with Claude.", no Co-Authored-By. `git status --porcelain` in the fork clone: clean (only ignored build output, `.pytest_cache`, `tests_regression/reports/`). nvcc on mi355 is still owed for 51018103 (the dispatcher runs it). All platforms now revalidate at 51018103.

### Draft follow-up PR for jeff (title and body; not posted, supersedes the earlier draft)

Title: `[ROCm] Restore the HIP build and fix a host max() truncation in ILP_NEP`

Body:

This follow-up to #1538 fixes a HIP-only bug in the ILP_NEP setup and brings the HIP build back in line with current master.

ILP_NEP: under HIP, the host call `max(max_nep_rc, parambs[i].rc_radial)` (double, float) resolved to the only host overload, `max(int, int)`, so a 4.2 A NEP radial cutoff became 4 and an ILP cutoff of 4.1 was accepted instead of rejected as on CUDA. It now calls `fmax`, which is what CUDA resolves the call to. A new `tests_regression` case, `ilp_nep_cutoff_below_nep`, checks the rejection; it fails with the old line and passes with the fix.

HIP build: `utilities/nep_model_file.cu` reached `nep_utilities.cuh` without a runtime header (nvcc pre-includes one, hipcc does not), so the header now includes `gpu_macro.cuh`. The NEP-spin kernels pass 32-bit masks to `__syncwarp` and `__shfl_*_sync`; HIP requires 64-bit masks, and on a 64-lane wavefront the second 32-lane tile sits in lanes 32-63. A `gpuLaneMask()` macro in `gpu_macro.cuh`, the identity on CUDA, moves the mask to the right lanes, and one warp kernel now passes its 32-lane width to its shuffles explicitly. `makefile.hip` also builds `gpumd_replica` like the default makefile, and compiles with `-ffp-contract=on`: hipcc's default (`fast`) also fuses multiply-adds across statements, which on an MI250X (gfx90a) rounds the per-slot and batched instantiations of the replica NEP kernels differently and fails the bitwise batched-versus-per-slot replica tests; `on` fuses only within one source expression, so both instantiations form the same multiply-adds.

Replica PRD: `PRD_Driver::initialize_slot` zero-filled the per-atom forces with a kernel on the default stream, unordered with the force computed on the slot's non-blocking stream, so some replicas could resume from a zero force; it now synchronizes the device after the allocation, which made batched and per-slot PRD runs identical on a Radeon 8060S (gfx1151, Windows), where they differed, and the race is timing-dependent on any GPU, CUDA included.

Besides the `fmax` line and that synchronization, the CUDA build sees only the extra include, the identity macro, an explicit width equal to the default, and a comment in `main_replica/force/ewald.cu` explaining why its uploads from local host vectors need no stream synchronization.

Testing on an AMD Instinct MI300X (gfx942), from the repository root:

```
make -C src -f makefile.hip HIP_ARCH=gfx942 -j16
GPUMD_REPLICA="$PWD/src/gpumd_replica" python3 tests_regression/replica/test_replica.py
python3 -m pytest tests_pytest/test_spin_*.py
make -C src -f makefile.hip clean
make -C src -f makefile.hip HIP_ARCH=gfx942 -j16 gpumd nep CFLAGS="-O3 --offload-arch=gfx942 -DUSE_HIP -ffp-contract=on -DDEBUG"
python3 tests_regression/run_regression.py --repo-root "$PWD" --suite full --baseline "$PWD/src/gpumd" --candidate "$PWD/src/gpumd" --baseline-nep "$PWD/src/nep" --candidate-nep "$PWD/src/nep"
```

gpumd, nep, gpumd_replica and tb build. The replica tests pass 16 of 16 on the MI300X and on an MI250X (gfx90a) when run with the NEP4 and qNEP4 example models from cd7f757d and eda8e04b, because gpumd_replica rejects the current example models. The pytest runs used calorine from its master branch, because calorine 4.1 cannot read the `format_version 1` model files now in `tests_pytest/fixtures`. The NEP-spin tests pass 110 of 114, including the FP64 reference checks, and `python3 -m pytest tests_pytest -m "not slow"` passes 866 with the same 4 failures. In the full regression suite, 247 of the 256 cases pass, including `ilp_nep_cutoff_below_nep`, and the relations pass 25 of 25. The remaining failures are of two kinds. Cases that depend on the default `rand()` stream (the four `test_corrector_preserves_force_modifiers` cases, `replicate_velocity_unseeded`, `graphene_hac`, `graphene_nemd_shc`, `graphene_hnemd_shc`, `silicon_liquid_transport`) differ between runs on some ROCm builds, because the runtime reseeds the C library's `rand()` when it prepares its internal kernels on the first launch; they pass when that reseed is suppressed. The others are float-atomic summation noise at or just above absolute bounds that the regression README asks to be checked by self-comparison on the target GPU: `qnep_pppm_future_bec` (BEC 1.17e-5 against 1e-5, or one unit in the last printed digit of a thermo value printed to six decimals), `pppm_round_exact` and `pppm_round_5_7` (thermo 1.1e-7 to 1.6e-7 against 1e-7) and `train_nep_pbte_ediff` (one unit in the last printed digit of a test force); they vary between runs, and `qnep_pppm_future_bec`, `train_nep_pbte_ediff` and `pppm_round_5_7` passed on a rerun. The same class covers `observer_average_two_runs` and `observer_average_then_observe_control` (when either fails, the four `observer_*` relations are skipped) and `train_charge_vdw_mixed`: the small-box NEP path accumulates per-atom virial with double `atomicAdd`, so the last digit of one atom's virial in second.xyz varies between runs, and the trainer's atomic accumulation likewise moves the last printed digit of a training loss.

(Before presenting: the head is now 26547d27 (3b0f1908 plus `-ffp-contract=on` in `makefile.hip` at 458683da, comment-only corrections at 5f6596ec, and the PRD slot-allocation sync at 26547d27, a separate vendor-neutral commit per jeff's 2026-11-10 ruling), so every gate needs revalidation there; refresh the MI300X counts from the gfx942 run at 5f6596ec (the quoted ones are from 3b0f1908, without the flag) and confirm the replica 16/16 there (the MI250X 16/16 is the porter's gfx90a run at 458683da, same binaries as 5f6596ec). The mi355 nvcc 12.8 full build at 26547d27 passes (porter 2026-11-10 entry). The PR-prep squash message must state the corrected reason for the flag (fuses only within one source expression, so both instantiations form the same multiply-adds), not 458683da's "as nvcc does by default". jeff's 2026-11-09 rulings on the rand() deferrals are applied: no GPUMD change, the body explains the 9 failures.)

## CUDA compile check (nvcc 12.8) at 51018103

Helper run on the mi355 fleet host (CUDA 12.8 toolkit, V12.8.93; no NVIDIA GPU), compile only, requested by jeff. Fresh clone of the fork's `moat-port` at **51018103** vs `git worktree add ../up c3d5d8ac` (upstream master). CUDA-visible delta: 070d3f53 (`gpu_macro.cuh` include in `utilities/nep_utilities.cuh`; `gpuLaneMask()` at 12 call sites -- 4 in `force/nep_spin/nep_spin.cu`, 6 in `force/nep_spin/nep_spin3_force.cuh`, 2 in `main_nep/nep_spin/nep_spin.cu`; explicit width `warp_size` on the two shuffles in `apply_ann_one_layer_warp`), the carried 15124b53 host line in `force/ilp_nep.cu`, and 51018103 (`CHECK(gpuStreamSynchronize(stream))` in `main_replica/force/ewald.cu` `Ewald::find_k_and_G`, host only). f3622149 is `tests_regression/` only.

Recipe: upstream's unmodified `src/makefile` (default `all` = gpumd nep gpumd_replica tb; `nvcc -std=c++14 -O3 -arch=sm_60 -Xcompiler="<makefile WARN_FLAGS>" -I./`), both trees concurrently:

    export PATH=/usr/local/cuda-12.8/bin:$PATH
    utils/timeit.sh GPUMD compile -- make -C <clone>/src -j64 -O      # 51018103, 20.4 s
    make -C <clone-up>/src -j64 -O                                    # c3d5d8ac

| | 51018103 | c3d5d8ac |
|---|---|---|
| make rc | 0 | 0 |
| nvcc invocations (compile / link) | 230 / 4 | 230 / 4 |
| objects | 230 `.o` | same 230 |
| executables | `gpumd` 24730448 B, `nep` 8004496 B, `tb` 1552152 B, `src/gpumd_replica` 3777440 B | same sizes |
| nvcc errors | 0 | 0 |
| nvcc warnings | 234, all `Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed` (one per invocation, `-arch=sm_60`) | identical 234 |
| host compiler warnings (makefile's -Wall -Wextra set) | none | none |

Whole logs, sorted and path-normalized: identical, so per-file diagnostics are identical, including `main_replica/force/ewald.cu` and the 070d3f53 files.

Device code (sm_60, the makefile build). No `nvdisasm` in CUDA 12.8 here, so `cuobjdump -sass` fails; instead `cuobjdump -xelf all` / `-xptx all` on all 230 objects of each tree, then a SHA-1 of every `.text.*`, `.nv.constant*`, `.nv.shared*`, `.nv.global*`, `.nv.info*` section of each cubin (read with a small Python ELF parser) plus the full PTX, after normalizing nvcc's per-compile `_GLOBAL__N__<8hex>_` / `_INTERNAL_<8hex>_` tokens. **All 3953 records identical: 841 kernel `.text` sections (8984448 B) and 230 PTX files.** Caution for whoever repeats this: `objcopy -O binary --only-section=... /dev/stdout` on a cubin silently yields empty output (every hash was SHA-1 of nothing on a first attempt), so check the hashes are not `da39a3ee...`. Per changed TU: `force/nep_spin/nep_spin.o` 43 kernels / 118328 PTX lines, `main_nep/nep_spin/spin_evaluate.o` 31 / 90782, `main_nep/nep_spin/nep_spin.o` 2 / 783, `force/ilp_nep.o` 11 / 15765, `main_replica/force/ewald.o` 2 / 907, `utilities/nep_model_file.o` 0 / 30 -- all identical.

What the comparison can see, by control (scratch copies of `main_nep/nep_spin/nep_spin.cu` compiled with the makefile flags, then deleted; the clone stays clean):
- Mask changed to `gpuLaneMask(0x0000ffff)` on one shuffle: **sm_60 and sm_70 SASS `.text` unchanged** (pre-Volta SHFL ignores the member mask, and on sm_70 ptxas emitted the same code here), but the PTX differs (`mov.u32 %r37, -1` -> `65535`). So for masks the PTX is the evidence, and it is identical across the trees.
- Width changed from `warp_size` to 16: `.text` hash of `apply_ann_one_layer_warp<199>` changes (d6247e05 -> 3f3100e2), so the section hashes do detect code changes.
- Extra: the six changed TUs also compiled at `-arch=sm_70` in both trees: rc 0, only the deprecation notice, cubin sections and PTX identical.

`gpuLaneMask()` on CUDA: `nvcc -E -Xcompiler -P` of `force/nep_spin/nep_spin.cu`, `main_nep/nep_spin/nep_spin.cu`, `main_nep/nep_spin/spin_evaluate.cu` (includes `nep_spin3_force.cuh`) and `utilities/nep_model_file.cu` in both trees: no `gpuLaneMask` token survives; the differences are `__shfl_*_sync(m, ...)` / `__syncwarp(m)` -> `(...((m)), ...)` with `m` = `0xffffffff`, `atom_mask`, `center_mask`, `active_mask` (the original masks), the two added `, warp_size` arguments, and re-wrapped lines. After stripping whitespace, the extra parentheses and `, warp_size`, the only remaining differences are 31 `fprintf(stderr, "Line: %d\n", N)` statements in `force/nep_spin/nep_spin.cu` with N shifted by +1 (CHECK macro `__LINE__`); the other three TUs are identical. `nep_model_file.cu` differs only by one blank line: `gpu_macro.cuh` is already included on CUDA before `nep_utilities.cuh`.

Host code (`objdump -d`, addresses and the hash tokens normalized): only `force/ilp_nep.o` (`fmax`), `force/nep_spin/nep_spin.o` (`__LINE__` constants) and `main_replica/force/ewald.o` differ. `ewald.o` gains one undefined `cudaStreamSynchronize` reference (0 -> 1) and 96 B of `.text` (35951 -> 36047), i.e. the added sync is the only change there; its kernels are identical.

Not covered: `main_nep/nep_spin/spin_specialized.cu` under `NEP_SPECIALIZED_RUNTIME_BUILD` (runtime-compiled by `nep_compile`); it uses the same `nep_spin3_force.cuh`, whose expansion is shown above. PASS: the CUDA path builds with nvcc 12.8 exactly as upstream c3d5d8ac does, with identical device code and PTX; the only CUDA behaviour change is the host-side stream sync in `Ewald::find_k_and_G`. No platform states changed.

## Review 2026-11-05 (linux-gfx1100, reviewer) -- fix round f3622149 -> 51018103, CHANGES REQUESTED

Scope: `git diff f3622149 51018103` (one line, `src/main_replica/force/ewald.cu:142`), the porter's notes, both rand() deferrals, the draft follow-up PR body, and skill commit e79b2c77. Fresh fork clone, `moat-port` = 51018103 = `origin/moat-port`. Host: RX 7900 XTX (gfx1100, wave32), HIP 7.16.26385 (TheRock pip SDK), `HIP_VISIBLE_DEVICES=0`, `ulimit -c 0`. `orient.sh` hit the 300 s cap with no further output; the lock was taken with `set-state GPUMD linux-gfx1100 reviewing --agent reviewer`. The nvcc 12.8 record at 51018103 (836d8245) is PASS, with the added sync as the only CUDA change.

### Must fix

1. **The Ewald sync fixes a hazard HIP does not have, and costs replica concurrency on CUDA and HIP. This corrects my own previous review's item 1, whose premise was wrong.** The HIP API documents the opposite of what the commit, the PR body and the skill entry say: `hip_runtime_api.h` (TheRock SDK, `hipMemcpyAsync`): "@note If host or dst are not pinned, the memory copy will be performed synchronously." CUDA documents the same staging for pageable sources. Probe on gfx1100 (`wall_clock64()` spin kernel queued on the stream, pageable H2D `hipMemcpyAsync`, overwrite the source at once, sync, compare): copies of 4 KiB to 64 KiB return while the stream is still busy (staged), copies larger than 64 KiB (16385 floats and up) block until the stream drains, and all 15 trials uploaded the original data. This matches the porter's gfx942 probe and the 0xFF free-poisoning runs. Upstream's `find_k_and_G` is therefore correct on both runtimes. The cost: `Replica_Runtime::advance_device` (`main_replica/replica_common.cu:485-520`) has one worker thread per GPU that interleaves every replica slot's stream step by step, and upstream deliberately syncs the NEP-charge path only every 1000 calls (`force/nep_charge.cu:1288-1291`). A per-call `gpuStreamSynchronize` inside `Ewald::find_force` makes that thread wait for slot i's queued work before it enqueues slot i+1. Measured with release builds of f3622149 (scratch worktree) and 51018103, qNEP model eda8e04b, `examples/gpumd_static_qnep/model.xyz`, `multi_replica remd replicas N exchange 100 temp 300 400`, `kspace ewald`, `run 2000`, alternating old/new after one warm-up run (production seconds):

| replicas | f3622149 | 51018103 |
|---|---|---|
| 2 | 7.39, 8.58, 7.34, 14.77 | 9.76, 8.27, 12.52, 12.10 |
| 4 | 25.30, 30.49, 28.65, 30.22 | 30.70, 28.94, 28.16, 27.14 |
| 8 | 51.81, 60.28, 52.21, 58.30 (mean 55.6) | 67.12, 63.69, 65.40, 72.61 (mean 67.2) |

At 8 replicas per GPU every new run is slower than every old one, about 21% on the means. 2 and 4 replicas are within noise. So "The results are unchanged" holds, but "no CUDA behaviour change" does not: the change is CUDA-visible, unneeded, and slower at high replica counts. Action: revert 51018103 with a new `[ROCm]`-titled revert commit on `moat-port` (no force push; upstream code is restored byte for byte, and a later PR-prep squash can drop the pair through `squash-carry-forward`, since the tree then equals f3622149's). Remove the PR body's `gpumd_replica:` paragraph, and change "and the added stream synchronization" in the CUDA-visible sentence. If jeff prefers to keep the sync as defensive code, the PR body must not claim "HIP does not promise that a pageable source has been read when `hipMemcpyAsync` returns", because HIP's header promises exactly that. 51018103's published message carries the same wrong claim, which is a further reason to revert rather than keep it. The sweep result itself is right: no other `*_async` copy from a dying pageable buffer exists in the HIP build. I rechecked the sites the porter's list does not name (`replica_common.cu:217-233`, D2D state save/restore; `minimizer_fire.cu:148,219`, `fill_async`). `Force` keeps one `NEP_Charge` (so one `Ewald`) per stream (`force/force.cuh` `charge_potentials_`), so the Ewald device buffers are not shared across slots.

2. **Skill entry `references/fault-classes.md` "cudaMemcpy*Async from a soon-freed pageable host buffer" (e79b2c77) now contradicts itself and the HIP documentation.** It still opens with "ROCm's is genuinely asynchronous, so the freed and reused buffer is read after the copy is enqueued", then adds that on HIP 7.16 neither path reads the source after return, then advises to "Fix by the API contract (sync the stream ...)", when HIP's API contract (the `hipMemcpyAsync` note above) says pageable copies are synchronous. `SKILL.md:83` still reads "CUDA stages synchronously, HIP does not." Rewrite both to the evidence: HIP documents pageable `hipMemcpyAsync` as synchronous; HIP 7.16 on gfx942 and gfx1100 stages copies of up to 64 KiB and blocks on larger ones, so freeing a pageable source after the call is safe there; a host sync added for it costs stream overlap (GPUMD replica, about 21% at 8 replicas). Mark the CV-CUDA observation as unexplained by this mechanism (other runtime version, or a different cause such as a pinned or registered source, or a D2H direction) unless someone re-verifies it. Probe-recipe nit: a `clock64()` spin of 2e8 ticks did not finish in 4.5 minutes on gfx1100 and had to be killed, while `wall_clock64()` (constant rate) worked. Write the recipe with `wall_clock64()`. The likely cause is gfx11's wrapping 20-bit shader-cycle counter behind `clock64()` (not verified).

### Resolved
- **rand() diagnosis (previous item 2): accurate** in the notes "Root cause" paragraph, both deferral summaries and the PR body sentence ("on some ROCm builds, because the runtime reseeds the C library's `rand()` when it prepares its internal kernels on the first launch"). Independently re-checked here with an `LD_PRELOAD` srand/rand counter on a HIP program (`hipMalloc`, one launch, `srand(1)`, then hipMemset/D2D/D2H/H2D/2D/MemsetD32/async copies, then `rand()` again). Exactly one `srand(<random>)` happens, followed by 38 `rand()`, all at the first launch, and the later blit paths consume no `rand()`, so a program's own `srand` after its first launch (GPUMD `velocity` with a seed) is unaffected, as stated. nm: TheRock `libLLVM.so.24.0git` imports `rand srand`, no `arc4random`; Ubuntu `libLLVM-17.so.1` (what `libamd_comgr.so.2.6.0` loads) imports `arc4random`, not `srand`. GPUMD's `print_compile_information()` (`utilities/main_common.cu:30-35`) precedes `print_gpu_information()` (`main_gpumd/main.cu:33-34`), and `main_replica/utilities/main_common.cu:31,34` behaves the same way. Wording nit for `references/validation.md` and the two deferrals: "Ubuntu 24.04's distro comgr imports `arc4random`" -- the import is in its `libLLVM-17.so.1`; `libamd_comgr.so.2.6.0` itself imports only `rand` (and so does TheRock's `libamd_comgr.so.3`). The nm check command already covers both files, so the conclusion stands. Also note in the check text that a libLLVM can import `srand` and `arc4random` together (Ubuntu `libLLVM.so.20.1` does), and that `arc4random` present is what decides.
- **PR body commands (previous item 3): run as written from the repository root**, with `HIP_ARCH`/`--offload-arch` gfx1100 and `-j$MAX_JOBS` substituted: see Tests.
- 51018103's own Test Plan (the two-replica Ewald case) run literally on gfx1100: rc 0, `kspace ewald` in `remd_restart.meta`.

### Tests on gfx1100
Commands from the draft PR body, run from the repository root with gfx1100 and `-j$MAX_JOBS` (10) substituted. The release build was in the fork clone; `make clean` and the DEBUG build ran in a detached worktree at 51018103 (`agent_space/dbg`), so the release binaries pytest was using stayed in place. pytest, the DEBUG build and the regression run partly overlapped on the same GPU and CPU, which slowed all of them.
- `make -C src -f makefile.hip HIP_ARCH=gfx1100 -j10`: rc 0, 1385 s; gpumd 15126912 B, nep 3867040 B, tb 309240 B, gpumd_replica 2299080 B.
- `python3 -m pytest tests_pytest/test_spin_*.py` (job venv: calorine 4.1, ase 3.29.0, pytest 7.3.2): **incomplete**. The command parses and runs; it was killed at my 40-min background limit after 54 of 114 tests: 50 passed, 4 failed. The 4 are positions 2-5 in collection order, the four known `test_corrector_preserves_force_modifiers` rand() cases (114/114 with the srand shim in the previous gfx1100 review).
- `make -C src -f makefile.hip clean`: rc 0. `make -C src -f makefile.hip HIP_ARCH=gfx1100 -j10 gpumd nep CFLAGS="-O3 --offload-arch=gfx1100 -DUSE_HIP -DDEBUG"`: rc 0, 2123 s.
- `python3 tests_regression/run_regression.py --repo-root "$PWD" --suite full --baseline "$PWD/src/gpumd" --candidate "$PWD/src/gpumd" --baseline-nep "$PWD/src/nep" --candidate-nep "$PWD/src/nep"`, given as written plus `--program gpumd --timeout-scale 2` and a subset of `--case` globs: `nve 'nvt_*' pimd_nvt 'ilp_*' 'pppm_*' 'ewald_*' 'qnep_*' 'invalid_*' 'parsing_*' 'parser_*' 'replicate_*' 'msd_*' train_nep_pbte_batch 'predict_*'` (129 cases). The command line is accepted as written. A 35-min cap stopped it after **45 cases: 43 PASS, 2 FAIL**. Ran: nve, all nvt_* ensembles, pimd_nvt, nvt_ber_fix_move_before, all 27 invalid_*, all 5 parser_*, qnep_ewald_future_bec, qnep_pppm_future_bec, qnep_actions_before_potential, replicate_first_after_comments. Failures: `invalid_pimd_change_beads` (the baseline run's stderr had an extra first line `pthread_setaffinity_np failed: No such process` from the HIP runtime on the error-exit path; same binary as candidate, so a runtime race, not the port); `qnep_actions_before_potential` (dpdt.out field 23: 0.161695 vs 0.161696, abs 1e-6 against atol 2e-7, the float-atomic qNEP noise class seen on both arches). ilp_*, pppm_*, msd_*, replicate_* (beyond the first) and the nep cases did not run here; the ILP_NEP case passed on gfx1100 in the previous review and on gfx942 in the fix round.
- 51018103's Test Plan: rc 0 (above). Old/new replica timings: item 1.

### Recommendation
**Request Changes**: item 1 (revert the sync and fix the PR body, or keep it and correct the rationale; jeff's call, revert recommended), item 2 (skill entry and `SKILL.md:83`). Nothing else in the round needs work.

## Fix round 2026-11-09 (linux-gfx942, porter) -- re-review items 1-2, 51018103 -> 374da0df

jeff asked for this run and ruled on the re-review's item 1: the `gpuStreamSynchronize` that 51018103 added to `Ewald::find_k_and_G` is not needed (HIP, like CUDA, finishes reading a pageable source before `hipMemcpyAsync` returns; `hip_runtime_api.h` notes on `hipMemcpyAsync` and `hipMemcpy2DAsync`: "If host or dst are not pinned, the memory copy will be performed synchronously") and costs replica concurrency, so remove it and leave a comment explaining why none is needed. Host: MI300X (gfx942), HIP 7.16.26385 (TheRock pip SDK), `HIP_VISIBLE_DEVICES=0` (dispatcher-pinned), GPU work under `ulimit -c 0`. `orient.sh` completed normally (about 2 min) and named porter; `set-state GPUMD linux-gfx942 porting --agent porter` took the lock. Fresh fork clone, `protect-fork` installed; `pr-state` = merged, so `moat-port` is not frozen. A local `master` was created from `origin/master` so `jargon.py --port` resolves its range.

### 1. 374da0df `[ROCm] Drop the Ewald upload sync and explain why none is needed`
`src/main_replica/force/ewald.cu`: the sync line is removed and replaced by a three-line comment (the `cpu_*` vectors are pageable, both CUDA and HIP finish reading a pageable source before the async copy returns, a sync would serialize the replica streams this thread interleaves). `git diff f3622149 374da0df` is those three comment lines only, so the code equals upstream's. The commit body says plainly that 51018103's premise was wrong, cites the header note and the 64 KiB staged / larger-blocks measurements, gives the gfx1100 cost (8 replicas, 67.2 s vs 55.6 s, about 21%), "Authored with Claude.", and a Test Plan (full `makefile.hip` build plus the two-replica Ewald case). The comment is new text, so `squash-carry-forward` cannot drop the 51018103/374da0df pair later (the tree differs from f3622149 by the comment).

### 2. Verification (gfx942)
- Release build, literal Test Plan command (`timeit.sh` with absolute path, `-j40`): `make -C src -f makefile.hip HIP_ARCH=gfx942` rc 0, 36 s; gpumd 14717192 B, nep 3473696 B, tb 315128 B, gpumd_replica 2401232 B (51018103's release gpumd_replica built in a scratch worktree: 2401472 B, i.e. -240 B, matching the +240 B recorded for the sync). No errors.
- Two-replica Ewald case, commit Test Plan run literally from the fork clone root (release): rc 0, `kspace ewald` in `remd_restart.meta`.
- Byte-identity: DEBUG `gpumd_replica` (`make -C src/main_replica CC=hipcc CFLAGS="-O3 --offload-arch=gfx942 -DUSE_HIP -DDEBUG" LIBS=-lhipfft`) of 51018103 and 374da0df in scratch worktrees, the same case three runs each, alternating: all 6 runs rc 0 with `remd_thermo.out` md5 5e17ceef03bf29187cb5e9ff60c9d90a and `remd_replica_0_restart.xyz` md5 42f6425664ed81470dfb1bd2f7631ca6, the same hashes recorded for 51018103 and f3622149 in the 2026-11-05 fix round.
- 8-replica timing, the re-review's case (`examples/gpumd_static_qnep/model.xyz`, qNEP model from eda8e04b, `ensemble nvt_bdp 300 400 100`, `multi_replica remd replicas 8 exchange 100 temp 300 400`, `kspace ewald`, `run 2000`), release builds, GPU otherwise idle: warm-up on 51018103 38.76 s, then **51018103 38.84 s, 374da0df 15.34 s** production time (`Time used for this REMD production run`). Single run each (plus the warm-up), so the magnitude is not a measured mean, but the direction is clear and the effect on MI300X (about 2.5x) is far larger than the 21% seen on gfx1100.

### 3. Skill (`port/GPUMD`)
- `references/fault-classes.md`: the entry "`cudaMemcpy*Async` from a soon-freed pageable host buffer" is rewritten as "lifetime and ordering: the hazard is a pinned source or another stream, not a pageable source". It states the documented rule (pageable async copies read the source before returning on CUDA and HIP; HIP 7.16 stages up to 64 KiB and blocks above that, on gfx942 and gfx1100), the GPUMD cost of a needless sync, the real hazards (a pinned or registered source freed or reused before the stream reaches the copy; cross-stream ordering, where another stream reads the destination before the copy lands), and a probe recipe using `wall_clock64()` with the gfx1100 `clock64()` hang, pointing at the existing 20-bit `clock64()` entry. The old text cited CV-CUDA (`InterpolationVarShapeWrap`), not TIGRE; that observation (ROCm 7.2.1, `hipMemcpy2DAsync` from a freed per-iteration `dstVec`, fixed by keeping it alive) contradicts the documented rule and is now marked unexplained and not re-verified on a current runtime. The self-contradiction ("genuinely asynchronous" plus "fix by the API contract") is gone.
- `SKILL.md:83`: the one-liner now states the same rule and the two real hazards.
- `grep -rn -i pageable .claude/skills/cuda-to-rocm`: only the two rewritten places remain; no other text states the old rule.
- Wording nit from the re-review: `references/validation.md` and both deferral summaries now say Ubuntu 24.04's comgr imports only `rand` itself and the `libLLVM-17.so.1` it loads imports `arc4random`, and that `arc4random` present is what decides (a libLLVM can import both). Deferral ids, status, timestamps and `decided: null` unchanged (two `summary` strings edited with a json load/dump).

### 4. Draft PR body
In the "Draft follow-up PR" above: the `gpumd_replica:` paragraph is removed, and the CUDA-visible sentence now ends with "a comment in `main_replica/force/ewald.cu` explaining why its uploads from local host vectors need no stream synchronization" instead of "the added stream synchronization". The body itself had no `arc4random`/comgr wording to fix. `prose.py`: clean; `jargon.py -`: clean.

### Gates
`python3 utils/jargon.py --port GPUMD`: the 2 known hits in upstream's 725fe762 only. `prose.py` and `jargon.py -` on 374da0df's message: clean; title 64 chars. `git status --porcelain` in the fork clone: clean. Pushed 51018103..374da0df (fast-forward), `advance-head` to 374da0df, stage `ported`. nvcc on mi355 is owed for 374da0df (the dispatcher runs it); the expected CUDA delta against c3d5d8ac is now 070d3f53, the 15124b53 `fmax` line and a comment, with `ewald.o` back to upstream's code. A re-review is next.

## CUDA compile check (nvcc 12.8) at 374da0df

Helper run on the mi355 fleet host (CUDA 12.8 toolkit, V12.8.93; no NVIDIA GPU), compile only, requested by jeff. Fresh clone of the fork's `moat-port` at **374da0df** vs `git worktree add ../up c3d5d8ac` (upstream master). CUDA-visible delta: 070d3f53 (`gpu_macro.cuh` include in `utilities/nep_utilities.cuh`, `gpuLaneMask()` masks, explicit `warp_size` widths), the carried 15124b53 `fmax` line in `force/ilp_nep.cu`, and in `main_replica/force/ewald.cu` `Ewald::find_k_and_G` only a three-line comment (374da0df removes the 51018103 `CHECK(gpuStreamSynchronize(stream))` and adds the comment in its place). The other `src/` change, `makefile.hip`, is not read by the CUDA build.

Recipe: same as the f3622149 and 51018103 entries (upstream's unmodified `src/makefile`, default `all`, `nvcc -std=c++14 -O3 -arch=sm_60 -Xcompiler="<makefile WARN_FLAGS>" -I./`), both trees at the same time:

    export PATH=/usr/local/cuda-12.8/bin:$PATH
    utils/timeit.sh GPUMD compile -- make -C <clone>/src -j64 -O      # 374da0df, 19.1 s
    make -C <clone-up>/src -j64 -O                                    # c3d5d8ac

| | 374da0df | c3d5d8ac |
|---|---|---|
| make rc | 0 | 0 |
| nvcc invocations (compile / link) | 230 / 4 | 230 / 4 |
| objects | 230 `.o` | same 230 |
| executables | `gpumd` 24730448 B, `nep` 8004496 B, `tb` 1552152 B, `src/gpumd_replica` 3777440 B | same sizes |
| nvcc errors | 0 | 0 |
| nvcc warnings | 234, all `Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed` (one per invocation) | identical 234 |
| host compiler warnings (makefile's -Wall -Wextra set) | none | none |

Whole logs, sorted and path-normalized: identical, so the nvcc diagnostics are identical for every file.

Device code: the same section-hash method (`cuobjdump -xelf all` / `-xptx all` on all 230 objects; SHA-1 of every `.text.*`, `.nv.constant*`, `.nv.shared*`, `.nv.global*`, `.nv.info*` cubin section, read by a Python ELF parser, plus the full PTX, with the `_GLOBAL__N__<8hex>_` / `_INTERNAL_<8hex>_` tokens normalized). **All 3953 records identical: 841 kernel `.text` sections (8984448 B) and 230 PTX files**, the same totals as the earlier entries. The SHA-1 of empty input appears only on the 144 NOBITS sections (123 `.nv.shared*`, 21 `.nv.global`), whose file body is empty by definition; no PROGBITS section hashed empty. Per changed TU: `force/nep_spin/nep_spin.o` 43 kernels / 118328 PTX lines, `force/ilp_nep.o` 11 / 15765, `main_nep/nep_spin/nep_spin.o` 2 / 783, `main_replica/force/ewald.o` 2 / 907; all identical.

Host code (`objdump -dr`). Only symbol/branch addresses and the hash tokens are normalized; immediates are kept. A first pass that also masked every hex immediate hid the `__LINE__` constants and showed only `ilp_nep.o`, so a repeat of this check should keep immediates. Three objects differ; the other 227 are identical:
- `force/ilp_nep.o`: 8 lines, a reordered load and a swapped xmm0/xmm1 pair around `cvtss2sd`, from the `fmax(max_nep_rc, double(...rc_radial))` line.
- `force/nep_spin/nep_spin.o`: 47 `mov $imm,%ecx` constants (upstream 1261..2196), each +1 (the `CHECK` macro's `__LINE__` after the added include line). Undefined symbols are identical.
- `main_replica/force/ewald.o`: **not byte-identical to upstream, but differs only by two `__LINE__` constants**, `0x117`/`0x114` and `0x12f`/`0x12c` (279/276 and 303/300). These are the two `GPU_CHECK_KERNEL` sites after `find_k_and_G`, shifted +3 by the new comment. `cudaStreamSynchronize` is undefined in neither object (the 51018103 reference is gone), the `.text` sizes match (10148 B each), and the two kernels are identical. So no executable code changed; only the line number that `CHECK` would print on an error moved.

Not covered (as before): `main_nep/nep_spin/spin_specialized.cu` under `NEP_SPECIALIZED_RUNTIME_BUILD`. PASS: at 374da0df the CUDA path builds with nvcc 12.8 exactly as upstream c3d5d8ac does, with identical device code and PTX. Host code differs only by the `fmax` codegen in `ilp_nep.o` and by `__LINE__` constants in `nep_spin.o` and `ewald.o`; there is no CUDA behaviour change against upstream. No platform states changed.

## Review 2026-11-09 (linux-gfx1100, reviewer) -- fix round 51018103 -> 374da0df and skill 38ce17bd, CHANGES REQUESTED (skill only)

Scope: `git diff 51018103 374da0df`, `git diff f3622149 374da0df`, the 374da0df message, skill commit 38ce17bd, the porter's 2026-11-09 entry, and the draft follow-up PR body (title and body under "Draft follow-up PR for jeff ... supersedes the earlier draft"). Fresh fork clone, `moat-port` = 374da0df = `origin/moat-port`. Host: RX 7900 XTX (gfx1100, wave32), HIP 7.16.26385 / AMD clang 24.0.0git (TheRock pip SDK), `HIP_VISIBLE_DEVICES=0` (dispatcher-pinned), `ulimit -c 0`. `orient.sh` completed normally and named reviewer; `set-state GPUMD linux-gfx1100 reviewing --agent reviewer` took the lock. The nvcc 12.8 record at 374da0df is PASS (ewald.o differs from upstream only by two `__LINE__` constants).

### Must fix

1. **Skill: the new rule "an async copy from a pageable source reads it before returning on CUDA and HIP" is false for `hipMemcpy2DAsync` on HIP 7.16, and the CV-CUDA case it calls unexplained is exactly that.** `references/fault-classes.md:177-203` (heading "the hazard is a pinned source or another stream, not a pageable source", the opening rule, "is safe on both platforms", and the "Unexplained by this rule: CV-CUDA" sentence) and `SKILL.md:83` ("`cudaMemcpy*Async` from a pageable host buffer reads it before returning on CUDA and HIP, so freeing it after the call is safe") state the rule for every async copy, but both probes (gfx942, gfx1100) measured only 1D `hipMemcpyAsync`; the 2D claim rests on the header note alone. Probe here (source below: pageable `std::vector` source filled 0x5a, optional 0.5 s `wall_clock64()` spin on the stream, the copy, then `memset(src,0xEE)` and `delete` at once, sync, D2H, byte histogram of the copied region; twice, plus a run with `dpitch == width`):

| copy | bytes | spin | call | sync | device sees |
|---|---|---|---|---|---|
| `hipMemcpy2DAsync` H2D, dpitch 256 | 234 (13x18) | no / yes | 0.04 / 0.02 ms | 0.1 / 500 ms | 0xEE x218 plus freed-heap bytes |
| same | 65536 (4096x16) | no / yes | 0.03 / 0.01 ms | 0.0 / 500 ms | 0xEE, all bytes |
| same | 524288 (8192x64) | no / yes | 0.05 / 0.01 ms | 0.1 / 500 ms | 0xEE, all bytes |
| same, dpitch = width | all three sizes | both | <= 0.07 ms | | 0xEE (and heap garbage at 234 B) |
| `hipMemcpyAsync` H2D | 234, 65536 | both | <= 0.02 ms | | 0x5a, correct |
| `hipMemcpyAsync` H2D | 524288 | yes | 500 ms (blocked) | 0.0 ms | 0x5a, correct |

Without the overwrite every case copies 0x5a correctly, so the 2D path itself works; it reads the pageable source after the call returns, even on an idle stream and at any size, contrary to the `hip_runtime_api.h` note. The 1D rows reproduce the 64 KiB staged / larger-blocks result. This is the CV-CUDA mechanism (`TestInterpolationVarShapeWrap.cpp`, per-iteration `dstVec` freed after `cudaMemcpy2DAsync`, garbage only in padding the kernel never writes, fixed by keeping `dstVec` alive with no sync change), now reproduced on a current runtime rather than "not re-verified". It also corrects my previous review, which took the header note as covering 2D. Action: scope the safe rule to 1D `hipMemcpyAsync` H2D (documented and measured on gfx942 and gfx1100, HIP 7.16); add that pageable `hipMemcpy2DAsync` on HIP 7.16 (measured gfx1100; gfx942 not measured) reads the source after return, so a pageable 2D source must stay alive and unmodified until the stream is synchronized, and that 3D and other copy variants are unmeasured, so probe before relying on them; rewrite the CV-CUDA sentence as the explained example; fix the heading and `SKILL.md:83` to match. Keep the GPUMD cost paragraph, the pinned-source and cross-stream hazards, and the `wall_clock64()` recipe (add "for 2D, issue `hipMemcpy2DAsync` the same way"). 38ce17bd's message states the same over-broad rule; a follow-up skill commit on `port/GPUMD` is enough (no rewrite).

Probe source (`hipcc --offload-arch=gfx1100 -O2 q.hip -o q`; for the `dpitch == width` run, replace the `dpitch` initialiser with `w`):

    #include <hip/hip_runtime.h>
    #include <chrono>
    #include <cstdio>
    #include <cstring>
    #include <map>
    #include <vector>
    #define C(x) do{hipError_t e=(x);if(e){printf("err %s line %d\n",hipGetErrorString(e),__LINE__);return 1;}}while(0)
    __global__ void spin(unsigned long long t){unsigned long long s=wall_clock64();while(wall_clock64()-s<t);}
    using clk=std::chrono::steady_clock;
    static double ms(clk::time_point a){return std::chrono::duration<double,std::milli>(clk::now()-a).count();}
    // mode: 0 = 2D, 1 = 1D ; ovw: overwrite+free source after call
    int run(hipStream_t st,int rate,int mode,int ovw,size_t w,size_t h,bool dospin){
      size_t dpitch=(w+255)/256*256; unsigned char* d; C(hipMalloc(&d,dpitch*h));
      C(hipMemset(d,0,dpitch*h)); C(hipDeviceSynchronize());
      auto* src=new std::vector<unsigned char>(w*h,0x5a);
      if(dospin) spin<<<1,1,0,st>>>((unsigned long long)rate*1000ull/2);
      auto t0=clk::now();
      if(mode==0) C(hipMemcpy2DAsync(d,dpitch,src->data(),w,w,h,hipMemcpyHostToDevice,st));
      else C(hipMemcpyAsync(d,src->data(),w*h,hipMemcpyHostToDevice,st));
      double tc=ms(t0);
      if(ovw){memset(src->data(),0xEE,src->size()); delete src;}
      auto t1=clk::now(); C(hipStreamSynchronize(st)); double ts=ms(t1);
      std::vector<unsigned char> out(dpitch*h); C(hipMemcpy(out.data(),d,out.size(),hipMemcpyDeviceToHost));
      std::map<int,long> hist; size_t rowlen=(mode==0)?w:w*h; size_t rows=(mode==0)?h:1; size_t pitch=(mode==0)?dpitch:w*h;
      for(size_t r=0;r<rows;r++)for(size_t c=0;c<rowlen;c++)hist[out[r*pitch+c]]++;
      printf("%s ovw=%d spin=%d bytes=%zu call=%.2fms sync=%.1fms vals:",mode?"1D":"2D",ovw,dospin,w*h,tc,ts);
      for(auto&p:hist)printf(" %02x:%ld",p.first,p.second); printf("\n");
      if(!ovw) delete src; C(hipFree(d)); return 0;
    }
    int main(){int rate; C(hipDeviceGetAttribute(&rate,hipDeviceAttributeWallClockRate,0)); printf("wallrate %d kHz\n",rate);
      hipStream_t st; C(hipStreamCreate(&st));
      size_t W[]={13,4096,8192},H[]={18,16,64};
      for(int k=0;k<3;k++) for(int mode=0;mode<2;mode++) for(int ovw=0;ovw<2;ovw++) for(int sp=0;sp<2;sp++) if(run(st,rate,mode,ovw,W[k],H[k],sp)) return 1;
      return 0;}

### Checked, no change needed
- **Tree:** `git diff f3622149 374da0df` is the three comment lines in `src/main_replica/force/ewald.cu:142-144` only; `git diff 51018103 374da0df` removes `CHECK(gpuStreamSynchronize(stream));` and adds them. Code equals upstream's.
- **Comment accuracy:** the four uploads are `GPU_Vector::copy_from_host_async` -> 1D `gpuMemcpyAsync` H2D (`src/main_replica/utilities/gpu_vector.cuh:138-141`), which CUDA documents as staging pageable sources before return and HIP documents and behaves as synchronous/staged (table above). `src/` has no `Memcpy2D`/`Memcpy3D` call, so the 2D finding does not touch GPUMD. "Would serialize the replica streams that this thread interleaves" matches `Replica_Runtime::advance_device` and the measured cost. Length (3 lines) and style (`//` full sentences, inside the 100-column `.clang-format` limit, lines 86-92 chars) fit the file; upstream comments here are sparse, and three lines for a non-obvious "why no sync" is proportionate.
- **374da0df message:** accurate (its claims are about `hipMemcpyAsync`); `[ROCm]` title 64 chars; "Authored with Claude."; no trailer; Test Plan literal.
- **Draft PR body:** accurate for the final tree; `prose.py`: clean; `jargon.py -` (body only): clean. Nit, optional: "The CUDA build sees only the extra include, ..." omits the `fmax` line, which CUDA also compiles (the ILP_NEP paragraph already says CUDA resolves the old call to `fmax`, so it is not misleading); "Besides that line, the CUDA build sees only ..." would be exact. The internal reminder under the body still says "add the mi355 nvcc result for 51018103"; it is 374da0df now (PASS, recorded).

### Tests on gfx1100, PR body commands from the repository root
Every command in the draft body was run from the repository root, with gfx1100 and `-j$MAX_JOBS` (16) substituted. The release build and pytest ran in the fork clone. `make clean`, the DEBUG build and the regression run ran in a detached worktree at 374da0df (`agent_space/dbg`), so the release binaries pytest was using stayed in place. pytest and the DEBUG build overlapped on the same host.
- `make -C src -f makefile.hip HIP_ARCH=gfx1100 -j16` (through `timeit.sh`): rc 0, 71 s, no errors (only upstream's `nodiscard` warnings). Sizes: gpumd 15128448 B, nep 3870840 B, tb 309904 B, gpumd_replica 2298824 B.
- `python3 -m pytest tests_pytest/test_spin_*.py` (job venv: calorine 4.1, ase 3.29.0, pytest 7.3.2): **110 passed, 4 failed** in 1053 s. The 4 failures are the four `test_corrector_preserves_force_modifiers` `rand()` cases, matching the body's "110 of 114".
- `make -C src -f makefile.hip clean`: rc 0. `make -C src -f makefile.hip HIP_ARCH=gfx1100 -j16 gpumd nep CFLAGS="-O3 --offload-arch=gfx1100 -DUSE_HIP -DDEBUG"`: rc 0, 920 s.
- `python3 tests_regression/run_regression.py --repo-root "$PWD" --suite full --baseline "$PWD/src/gpumd" --candidate "$PWD/src/gpumd" --baseline-nep "$PWD/src/nep" --candidate-nep "$PWD/src/nep"`, run as written plus `--timeout-scale 2` and a representative subset in place of the full suite: `--case 'ilp_*' 'qnep_*' 'kspace_*' ewald_no_force nve 'nvt_*' 'parser_*' 'observer_*' pppm_round_5_7`. Result: **29 passed, 0 failed**; relations 4 passed, 21 skipped (members not selected); rc 0, 336 s. The subset includes `ilp_nep_cutoff_below_nep`, all four qNEP Ewald/PPPM cases, `observer_average_then_observe_control` and `pppm_round_5_7`.
- Not in the body, but relevant because `ewald.cu` is in `gpumd_replica`: `python3 tests_regression/replica/test_replica.py` on the release binary gave 6/12. The 6 failures are the known `format_version 1` model-header failures (upstream dd7d0ba8, recorded in the sync round). In the scratch worktree with `examples/nep_train/nep.txt` from cd7f757d and `examples/qnep_train/nep.txt` from eda8e04b (`GPUMD_REPLICA` set to the release binary; the edit was reverted afterwards): **12/12 OK**, including the qNEP REMD/PRD Ewald and PPPM restart tests.
- 374da0df's Test Plan, run as written from the fork clone root on the release build: rc 0, `kspace ewald` in `remd_restart.meta`.
- `git status --porcelain` in the fork clone: clean; HEAD = `origin/moat-port` = 374da0df.

### Recommendation
**Request Changes, skill only (item 1).** The fork at 374da0df needs no change: the tree differs from f3622149 by the comment alone, the comment is accurate for the 1D uploads it describes, and the PR body is accurate and passes both gates. Item 1 is a follow-up commit on `port/GPUMD` correcting `references/fault-classes.md` and `SKILL.md:83` for the measured `hipMemcpy2DAsync` behaviour. No fork commit means no new nvcc or validation round. The next review only needs to check the skill delta.

## Skill fix 2026-11-09 (linux-gfx942, porter) -- review item 1 (pageable 2D/3D async copies), head unchanged at 374da0df

jeff asked for this skill-only run. Host: MI300X (gfx942), HIP 7.16.26385 (`hipRuntimeGetVersion` 71626385), AMD clang 24.0.0git (TheRock pip SDK), `HIP_VISIBLE_DEVICES=0` (dispatcher-pinned), probe under `ulimit -c 0`. `orient.sh` completed normally and named porter; `set-state GPUMD linux-gfx942 porting --agent porter` took the lock. No fork change: `moat-port` stays at 374da0df.

### Probe (gfx942)
The reviewer's probe above, extended: mode 2 issues `hipMemcpy3DAsync` (srcPtr `make_hipPitchedPtr(src,w,w,h/2)`, dstPtr pitch `dpitch`, extent `(w, h/2, 2)`); `ovw` 0 = untouched, 1 = `memset(src,0xEE)` then `delete` right after the call, 2 = `memset` only (no free, to separate "read late" from heap reuse); every case at padded (`dpitch` rounded to 256) and tight (`dpitch == w`) pitch; the histogram is reduced to counts of 0x5a, 0xEE and other. `hipcc --offload-arch=gfx942 -O2 q.hip -o q`, run twice: byte counts identical between runs, timings within 0.1 ms (the first call of a run pays about 16-24 ms of init).

| copy | bytes | spin | call | sync | device sees (ovw=2 / ovw=1) |
|---|---|---|---|---|---|
| 1D `hipMemcpyAsync` | 234, 65536 | no / yes | <= 0.02 ms | 0.0 / 501 ms | 0x5a, all bytes (staged) |
| 1D `hipMemcpyAsync` | 524288 | no | 0.03-0.05 ms | 0.0 ms | 0x5a, all bytes |
| 1D `hipMemcpyAsync` | 524288 | yes | 501.2 ms (blocked) | 0.0 ms | 0x5a, all bytes |
| `hipMemcpy2DAsync` | 234 | no / yes | <= 0.02 ms | 0.1 / 501 ms | 0xEE x234 / 0xEE x218 + 16 heap bytes |
| `hipMemcpy2DAsync` | 65536 | no / yes | <= 0.03 ms | 0.0 / 501 ms | 0xEE x65536 / 0xEE x65504 + 32 heap bytes |
| `hipMemcpy2DAsync` | 524288 | no / yes | <= 0.02 ms | 0.0 / 501 ms | 0xEE, all bytes (both) |
| `hipMemcpy3DAsync` | 234, 65536, 524288 | no / yes | <= 0.02 ms | as 2D | same as 2D at each size |

Padded and tight pitch give the same result. Without the overwrite (ovw=0) every case copies 0x5a correctly. So on gfx942, as the reviewer found on gfx1100: 1D is staged up to 64 KiB and blocks above it (on an idle stream a large 1D copy just completes inside the call); 2D and 3D return at once at every size, idle stream included, and read the pageable source later. 3D on gfx1100 is not measured.

### Skill (commit 38081568 on `port/GPUMD`)
- `references/fault-classes.md`: heading now "a pageable 1D source is safe, a pageable 2D or 3D source is not on HIP"; the safe rule is limited to 1D `hipMemcpyAsync`; the header note is quoted with "only the 1D half holds"; the measurements (both arches, HIP 7.16.26385, sizes, pitches, spin) are listed; a pageable 2D/3D source must stay alive and unchanged until the stream (or an event after the copy) is synchronized, while CUDA documents staging; CV-CUDA is now the explained example; D2H, symbol and array copies are named unmeasured; the GPUMD cost paragraph, the pinned-source and cross-stream hazards and the `wall_clock64()` recipe (now "for 2D or 3D, `hipMemcpy2DAsync` or `hipMemcpy3DAsync` the same way", "staged, blocked or deferred") are kept. The entry cites no TIGRE case, so there was none to keep.
- `SKILL.md:83`: CUDA documents staging every H2D variant; HIP stages only 1D; a pageable 2D/3D source must stay alive and unchanged until the stream is synchronized; pinned-source and cross-stream hazards on both platforms.
- `grep -rn -i 'pageable' .claude/skills/cuda-to-rocm`: hits only in the two rewritten passages; no other statement in the skill claims 2D/3D staging (`grep -rn 'Memcpy2D\|Memcpy3D\|before return'` likewise).

### Draft follow-up PR body nits
- "The CUDA build sees only ..." is now "Besides the `fmax` line, the CUDA build sees only ...".
- The internal reminder now says to redo the mi355 nvcc compile if the head moves past 374da0df (PASS there, recorded), instead of "add the mi355 nvcc result for 51018103".
- `prose.py` on the body: clean; `jargon.py -`: clean.

Stage `ported` with head unchanged at 374da0df; the next review only needs the skill delta (38081568) and the two body edits.

## Review 2026-11-09 (linux-gfx1100, reviewer) -- skill 38081568 and PR body nits, head unchanged at 374da0df, CHANGES REQUESTED (skill only)

Scope: skill commit 38081568 (`SKILL.md:83`, `references/fault-classes.md:177-214`), the porter's "Skill fix 2026-11-09" entry, and the two edits to the draft follow-up PR body. Fork `moat-port` = 374da0df (`git ls-remote`), unchanged, so no code review. Host: RX 7900 XTX (gfx1100), HIP 7.16.26385 (`hipRuntimeGetVersion` 71626385), `HIP_VISIBLE_DEVICES=0` (dispatcher-pinned), `ulimit -c 0`. `orient.sh` completed and named reviewer; `set-state GPUMD linux-gfx1100 reviewing --agent reviewer` took the lock.

### Must fix

1. **Skill: "CUDA documents that every H2D async copy from PAGEABLE host memory is staged before the call returns" (`fault-classes.md:178-179`) and "CUDA documents staging every H2D variant before return" (`SKILL.md:83`) overstate the CUDA documentation.** The CUDA Runtime API "API synchronization behavior" page (https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html), Memcpy, Asynchronous, says only: "For transfers between device memory and pageable host memory, the function might be synchronous with respect to host" and "If pageable memory must first be staged to pinned memory, the driver may synchronize with the stream and stage the copy into pinned memory". The "returns once the pageable buffer has been copied to the staging memory" guarantee is in the Synchronous section (`cudaMemcpy`), not the async one. The evidence that CUDA stages pageable 2D copies before return is behavioural (CV-CUDA's freed `dstVec` is correct on CUDA), not documented, and MOAT has not probed CUDA. Same issue at `fault-classes.md:193-194`, where "is correct on CUDA" should say it works on CUDA in practice. Action: say what the page says ("might be synchronous"; the driver "may" stage), say CUDA staging pageable H2D copies before return, 2D included, is observed (CV-CUDA) and not guaranteed, and draw the conclusion that holds on both platforms: keep a pageable 2D/3D source alive until the stream is synchronized. The HIP 1D advice ("do not add a sync after a 1D pageable upload for HIP") rests on the HIP measurements and stays as written.
2. **Skill: `fault-classes.md:188` "(gfx942; gfx1100 not measured)" for `hipMemcpy3DAsync` is stale.** Measured here (below): gfx1100 matches gfx942 exactly. Change it to "(both arches)", or fold it into the 2D wording, in the same commit as item 1. The commit message for 38081568 has the same parenthetical; a follow-up commit is enough.

### Nit (optional)
- `fault-classes.md:181` "but only the 1D half holds": a 1D copy of 64 KiB or less is not "performed synchronously". It is staged and returns while the stream is still busy, which the bullet below says. Only the safety consequence holds, so "but only for 1D is the source safe to reuse on return" would be exact.

### 3D probe on gfx1100 (requested)
The porter's extended probe (modes 1D/2D/3D; `ovw` 0 untouched, 1 `memset(0xEE)` + `delete`, 2 `memset` only; padded (256) and tight pitch; 234 B, 64 KiB, 512 KiB; with and without a 0.5 s `wall_clock64()` spin). The 3D case uses srcPtr `make_hipPitchedPtr(src,w,w,h/2)`, dstPtr pitch `dpitch`, extent `(w,h/2,2)`. Built with `hipcc --offload-arch=gfx1100 -O2 q.hip -o q` and run twice. Byte counts were identical between the runs (timing fields stripped and `cmp`).
- `hipMemcpy3DAsync`, every size, both pitches, spin or not: call 0.01-0.04 ms; sync 0.0-0.1 ms idle and 500 ms with spin. ovw=0 gives 0x5a for every byte. ovw=2 gives 0xEE for every byte. ovw=1 gives 0xEE x218 + 16 other (234 B), 0xEE x65504 + 32 other (64 KiB) and 0xEE for every byte (512 KiB). This is identical to 2D on this host and to the porter's gfx942 table.
- The 2D and 1D rows reproduce the earlier gfx1100 results: 2D reads late at every size, and 1D is staged at 64 KiB or less (call <= 0.02 ms with spin) and blocks above it (call 500.07 ms with spin), always 0x5a. The only call over 0.1 ms other than the blocked 1D copies is the first call of the run (2D 234 B, 8.8 ms, init).

### Checked, no change needed
- HIP header quote: `hip_runtime_api.h` (TheRock 7.16 SDK) carries "@note If host or dst are not pinned, the memory copy will be performed synchronously." on `hipMemcpyAsync` (line 5432) and `hipMemcpy2DAsync` (line 5924), and not on `hipMemcpy3DAsync` (line 6136). The skill quotes it for exactly those two.
- The measurement list in `fault-classes.md:182-191` matches both arches' tables (sizes, pitches, spin, "under 0.1 ms" excluding first-call init, freed-heap bytes on delete). D2H, symbol and array copies are named unmeasured. The CV-CUDA example matches the earlier review's account. The GPUMD cost paragraph, pinned-source and cross-stream hazards, and `wall_clock64()` recipe are kept and consistent.
- `grep -rn -i 'pageable\|Memcpy2D\|Memcpy3D' .claude/skills/cuda-to-rocm`: hits only `SKILL.md:83`, `fault-classes.md:177-212`, and `fault-classes.md:337` (layered textures via host `hipMemcpy3D`, unrelated). No other passage contradicts the entry. `SKILL.md:83` agrees with the reference apart from item 1.
- Draft PR body: "Besides the `fmax` line, the CUDA build sees only ..." is exact (the include, `gpuLaneMask` identity, explicit width, ewald comment; `makefile.hip` is HIP only). The internal reminder now names 374da0df, and the nvcc PASS there is recorded. `prose.py`: clean; `jargon.py -` (body without the internal reminder): clean.

### Recommendation
**Request Changes, skill only (items 1-2).** This is one follow-up skill commit on `port/GPUMD`. The fork, the PR body, the nvcc record and the validation picture are unaffected. The next review only needs that commit.

## Skill fix 2026-11-09 (linux-gfx1100, porter) -- review items 1-2 and nit, skill commit cde62cba, head unchanged at 374da0df

jeff asked for this skill-only run. Host: RX 7900 XTX (gfx1100). `orient.sh` completed and named porter; `set-state GPUMD linux-gfx1100 porting` took the lock. No fork change and no GPU work: `moat-port` stays at 374da0df.

- Item 1 (`fault-classes.md:177-206`, `SKILL.md:83`): the claim that CUDA documents staging every pageable H2D async copy before return is gone. The entry now quotes NVIDIA's API synchronization page (Memcpy, Asynchronous: "might be synchronous with respect to host"; the driver "may" stage into pinned memory), notes the "returns once ... copied to the staging memory" guarantee is in the Synchronous section, and says CUDA staging of pageable 2D copies before return is observed (CV-CUDA), not documented, and not probed by MOAT. HIP behaviour is labelled measured. Conclusion on both platforms: keep a pageable 2D/3D source alive until the stream (or an event after the copy) is synchronized; on CUDA do not rely on it being read before return unless documented. CV-CUDA's freed `dstVec` "works on CUDA in practice". The 1D advice (no extra sync for HIP) is unchanged.
- Item 2 (`fault-classes.md:196`): `hipMemcpy3DAsync` now "(both arches)", folded into the 2D bullet, per the reviewer's gfx1100 probe.
- Nit (`fault-classes.md:187-189`): the header note is no longer "only the 1D half holds"; it now says neither copy is synchronous, but for 1D the source can be reused or freed right after the call (staged up to 64 KiB, blocking above it, per the bullet), and for 2D it cannot.
- `grep -rn -i 'pageable\|Memcpy2D\|Memcpy3D' .claude/skills/cuda-to-rocm`: hits only `SKILL.md:83`, `fault-classes.md:177-222` and the unrelated layered-texture line (`fault-classes.md:347`); all consistent. New lines are ASCII; `utils/check.py` ok.

Stage `ported`; the next review only needs skill commit cde62cba.

## Review 2026-11-09 (linux-gfx1100, reviewer) -- skill cde62cba, head unchanged at 374da0df, CHANGES REQUESTED (skill only)

Scope: skill commit cde62cba (`SKILL.md:83`, `references/fault-classes.md:177-222`) and the porter's "Skill fix 2026-11-09 (linux-gfx1100)" entry. Fork `moat-port` = 374da0df (`git ls-remote`), equal to `head_sha`, so no code review. No probe rerun (the gfx942 and gfx1100 tables above are the evidence). `orient.sh` completed and named reviewer; `set-state GPUMD linux-gfx1100 reviewing --agent reviewer` took the lock. `HIP_VISIBLE_DEVICES=1` (dispatcher-pinned), `ulimit -c 0`.

### Must fix

1. **`fault-classes.md:178-179` "Neither platform documents that a pageable H2D async copy reads its source before the call returns" contradicts `fault-classes.md:186-188`.** The quoted HIP header note ("If host or dst are not pinned, the memory copy will be performed synchronously", `hip_runtime_api.h` 5432 and 5924 in the TheRock 7.16 SDK, rechecked) does document exactly that for `hipMemcpyAsync` and `hipMemcpy2DAsync`; it is the documentation that is wrong for 2D. Action: limit the sentence to CUDA, e.g. "CUDA does not document that ...; HIP's header claims it for 1D and 2D, and the measurements below show it holds (as source safety) only for 1D."
2. **`fault-classes.md:188` "neither copy is synchronous" overstates the probes for 1D above 64 KiB.** Both tables show a 512 KiB 1D copy blocking in the call until the stream drains (501 ms call, 0.0 ms sync; idle stream: completes inside the call), and the bullet at `fault-classes.md:194` says so. Only the 1D copy of at most 64 KiB and the 2D copy are shown not to be synchronous. Action: scope it, e.g. "a 1D copy of at most 64 KiB is staged rather than synchronous and a 2D copy is neither, but for 1D the source can be reused ...". (The previous review's nit was scoped to 64 KiB or less; the generalisation is new.)

### Checked, no change needed
- NVIDIA "API synchronization behavior" page refetched: Memcpy, Asynchronous says "might be synchronous with respect to host" and the driver "may synchronize with the stream and stage the copy into pinned memory"; "The function will return once the pageable buffer has been copied to the staging memory" is in the Synchronous section. The skill quotes and places both correctly, and `SKILL.md:83` matches.
- CUDA staging of 2D before return is labelled observed (CV-CUDA), not documented and not probed; "works on CUDA in practice" (`fault-classes.md:203-204`) is accurate.
- `hipMemcpy3DAsync` "(both arches)" (`fault-classes.md:196`) matches the gfx942 and gfx1100 probes; the measurement parameters (HIP 7.16.26385, 234 B / 64 KiB / 512 KiB, tight and padded pitch, 0.5 s `wall_clock64()` spin, freed-heap bytes on delete) match both tables. D2H, symbol and array copies stay named unmeasured.
- `grep -rn -i 'pageable\|Memcpy2D\|Memcpy3D' .claude/skills/cuda-to-rocm`: hits only `SKILL.md:83`, `fault-classes.md:177-222` and the unrelated layered-texture line 347. `SKILL.md:83` contains neither problem above and agrees with the reference; no other passage contradicts it.

### Recommendation
**Request Changes, skill only (items 1-2).** Two sentence edits in `fault-classes.md` in one follow-up skill commit on `port/GPUMD`; `SKILL.md`, the fork, the PR body, the nvcc record and validation are unaffected. The next review only needs that commit.

## Skill fix 3 (2026-11-09, primary session, windows-gfx1151)

Applied the fifth review's two sentence edits to `references/fault-classes.md` (CUDA-only "does not document" sentence; 1D <= 64 KiB staged, larger 1D blocks, 2D neither). Fork unchanged at 374da0df.

## Review 2026-11-09 (linux-gfx1100, reviewer) -- skill c611a120, head unchanged at 374da0df, REVIEW PASSED

Scope: skill commit c611a120 (`references/fault-classes.md:177-191`) and the "Skill fix 3" entry. Fork `moat-port` = 374da0df (`git ls-remote`), equal to `head_sha`, so no code review and no probe rerun (the gfx942 and gfx1100 tables above are the evidence). `orient.sh` completed and named reviewer; `set-state GPUMD linux-gfx1100 reviewing --agent reviewer` took the lock. `HIP_VISIBLE_DEVICES=1` (dispatcher-pinned), `ulimit -c 0`.

### Checked, no change needed
- Item 1 of the fifth review: `fault-classes.md:178-180` now limits "does not document" to CUDA and says HIP's header claims source-read-before-return for 1D and 2D, holding only for 1D. Matches the header (note present on `hipMemcpyAsync` and `hipMemcpy2DAsync`, absent on `hipMemcpy3DAsync`; rechecked in a local `hip_runtime_api.h`) and both arches' probes. No longer contradicts lines 187-191.
- Item 2: `fault-classes.md:189-191` now says a 1D copy of at most 64 KiB is staged, a larger 1D copy blocks in the call, and a 2D copy is neither. Matches both tables (512 KiB 1D: 501 ms call with spin, completes in the call when idle; 2D/3D: under 0.1 ms call at every size) and the bullet at lines 193-195. The 64 KiB threshold is bracketed by the measured 64 KiB and 512 KiB points, as the bullet already stated.
- Internal consistency: heading, opening sentences, bullets, the "must stay alive until synchronized" conclusion, the CV-CUDA example, the unmeasured list (D2H, symbol, arrays), the GPUMD cost paragraph and the probe recipe agree. `SKILL.md:83` is unchanged and agrees.
- `grep -rn -i 'pageable\|Memcpy2D\|Memcpy3D' .claude/skills/cuda-to-rocm`: hits only `SKILL.md:83`, `fault-classes.md:177-224` and the unrelated layered-texture line 349; no contradiction. No non-ASCII in `fault-classes.md`; `utils/check.py` ok.

### Nit (optional, no action needed)
- The c611a120 message says HIP's header claims the copies "are synchronous (the claim only holds for 1D)"; per the skill text, the safety consequence holds for 1D, while a 1D copy of at most 64 KiB is staged rather than synchronous. Control-plane commit only; the skill text is exact.

### Recommendation
**Review passed.** The fork at 374da0df was passed on code by the previous review; the skill text is now accurate, internally consistent and names what was not measured.

## Validation 2026-11-09 (linux-gfx1100, validator) -- revalidate at 374da0df, PASS

jeff asked for this run. Host: AMD Radeon RX 7900 XTX (gfx1100, RDNA3, wave32), `HIP_VISIBLE_DEVICES=0` (dispatcher-pinned), all GPU runs under `ulimit -c 0`. Toolchain: TheRock pip SDK, HIP 7.16.26385, AMD clang 24.0.0git. `orient.sh` completed normally and named validator (`revalidate`). Fresh clone of AMD-Ecosystem/GPUMD, detached at **374da0df0d0011867be8a709c3e912d54cdc4eb0** (= `origin/moat-port`); a detached worktree of the same commit for the DEBUG build, and a second one for the replica legacy-model run. No fork edit, no push. Job venv: calorine 4.1, ase 3.29.0, pytest 7.3.2, hypothesis.

### Builds (PR-body commands, gfx1100)
    utils/timeit.sh GPUMD compile -- make -C <clone>/src -f makefile.hip HIP_ARCH=gfx1100 -j16
    utils/timeit.sh GPUMD compile -- make -C <dbg-worktree>/src -f makefile.hip HIP_ARCH=gfx1100 -j24 gpumd nep CFLAGS="-O3 --offload-arch=gfx1100 -DUSE_HIP -DDEBUG"

Both rc 0, 0 errors (warnings: only upstream's `nodiscard hipError_t` and `-Winconsistent-missing-override` classes). Release, 47 s: gpumd 15128448 B, nep 3870840 B, tb 309904 B, gpumd_replica 2298824 B (byte sizes identical to the 2026-11-09 gfx1100 review's build of the same commit). DEBUG, 41 s: gpumd 15128312 B, nep 3870784 B. (A first launch of the two builds was stopped by me before finishing, because I had set a too-short background timeout; both trees were `make clean`ed and rebuilt from scratch. The DEBUG half of the second launch died at once with exit 127 because its subshell missed the `cd` to the MOAT root that `timeit.sh` needs (so it wrote no telemetry entry); not a build failure.)

### Tests (real gfx1100)
- **NEP-spin pytest** (release): `python -m pytest tests_pytest/test_spin_{fire,force_modifiers,model_format,restart,sib,tspin,validation}.py` -> **110 passed, 4 failed**, 55 s. The 4 are the known `test_corrector_preserves_force_modifiers` cases (sib/glsd x small/large box, `add_random_force` `rand()` stream). Rerun with an `LD_PRELOAD` shim that turns `srand` into a no-op: **4 passed**. So 114/114 when that reseed is suppressed, and the `gpuLaneMask` kernels pass with HIP's mask checks live (no `NDEBUG`) on wave32.
- **Full `tests_regression`** (DEBUG, self-comparison, PR-body command plus `--timeout-scale 3`):
  `python3 tests_regression/run_regression.py --repo-root $D --suite full --baseline $D/src/gpumd --candidate $D/src/gpumd --baseline-nep $D/src/nep --candidate-nep $D/src/nep --timeout-scale 3`
  -> **completed: 249/256 cases, relations 25/25**, 442 s (the whole suite finished on this host, including the FP64 graphene/silicon transport cases; it overlapped the not-slow pytest on the same GPU for its full length). `ilp_nep_cutoff_below_nep` PASS. The 7 failures, classified by reruns with `--case`:

| case | first run | rerun plain | rerun with srand shim | class |
|---|---|---|---|---|
| replicate_velocity_unseeded (final.xyz) | FAIL | FAIL | PASS | rand() reseed (known) |
| graphene_hac (hac.out) | FAIL | FAIL | PASS | rand() reseed (known) |
| graphene_nemd_shc (compute.out) | FAIL | FAIL | PASS | rand() reseed (known) |
| graphene_hnemd_shc (kappa.out) | FAIL | FAIL | PASS | rand() reseed (known) |
| silicon_liquid_transport (adf.out) | FAIL | FAIL | PASS | rand() reseed (known) |
| qnep_pppm_future_bec | bec 1.44e-5 vs atol 1e-5 | bec 1.54e-5 | FAIL, thermo field 9 1.04e-7 vs atol 1e-7 | float-atomic noise (known; different field per run, fails with the shim too) |
| qnep_actions_before_potential | dpdt.out 1.0e-6 vs atol 2e-7 | PASS | -- | float-atomic noise (known; same magnitude as the 2026-11-05 gfx1100 review) |

  No new failure class. Shim run of the 5 rand() cases: `Summary: 5 passed, 0 failed`.
- **`tests_pytest -m "not slow"`** (release, whole directory): **801 passed, 4 failed, 480 skipped, 10 deselected**, 1010 s. The 4 failures are the same four `test_corrector_preserves_force_modifiers` rand() cases (they are inside this selection); no other failure. Skips are the fixture/netCDF skips recorded on gfx942 (480, same count).
- **Replica** (`GPUMD_REPLICA=<clone>/src/gpumd_replica python3 tests_regression/replica/test_replica.py`, release): as-is **6/12**, the 6 failures all `REMD/PRD explicit streams support energy NEP4/NEP5 and qNEP4 models` (the known upstream dd7d0ba8 `format_version 1` model-header class). In the scratch worktree with `examples/nep_train/nep.txt` from cd7f757d and `examples/qnep_train/nep.txt` from eda8e04b: **12/12 OK** (18.8 s), including the qNEP REMD/PRD Ewald and PPPM restart tests, which run the `ewald.cu` path 374da0df comments.

### Gates
- CUDA compile gate: already recorded PASS at 374da0df (mi355, nvcc 12.8, entry above); not repeated.
- `python3 utils/jargon.py --port GPUMD` (local `master` from the fork's `origin/master`): the 2 known hits in upstream's own 725fe762 merge only. `jargon.py -C projects/GPUMD/src --commits c3d5d8ac..374da0df --diff c3d5d8ac...374da0df`: clean.
- Documentation: `doc/installation.rst` "Building for AMD GPUs (ROCm/HIP)" documents `make -f makefile.hip` and `HIP_ARCH`; README points to it.
- Integrity: `git status --porcelain` clean in the clone and the DEBUG worktree (only ignored build output and reports); HEAD 374da0df.

Result: **completed** at 374da0df on linux-gfx1100. Every failure is in a known class (rand() reseed by the runtime's comgr on the first launch; float-atomic qNEP/PPPM noise at absolute bounds; replica model headers upstream).
## Validation 2026-11-09 (linux-gfx942, validator) -- revalidate at 374da0df, PASSED

jeff asked for this run. Host: MI300X (gfx942, CDNA3, wave64), HIP 7.16.26385 / AMD clang 24.0.0git (TheRock pip SDK), `HIP_VISIBLE_DEVICES=0` (dispatcher-pinned), all GPU work under `ulimit -c 0`, GPU otherwise idle for every run (no concurrent GPU jobs). `orient.sh` completed normally and named validator (`state=revalidate`). Fresh clone of AMD-Ecosystem/GPUMD detached at **374da0df0d0011867be8a709c3e912d54cdc4eb0** (= `origin/moat-port`); DEBUG build in a detached `git worktree` at the same sha (`agent_space/dbg`) so the release binaries pytest uses stayed in place (the PR body's `make clean` + DEBUG rebuild in one tree, split across two trees, as the gfx1100 reviewer did). No fork edit, no push. Test environment: job venv with calorine 4.1, ase 3.29.0, pytest 7.3.2, hypothesis.

### Builds (PR body commands, gfx942, -j56)
    utils/timeit.sh GPUMD compile -- make -C <clone>/src -f makefile.hip HIP_ARCH=gfx942 -j56
    utils/timeit.sh GPUMD compile -- make -C <dbg>/src -f makefile.hip HIP_ARCH=gfx942 -j56 gpumd nep CFLAGS="-O3 --offload-arch=gfx942 -DUSE_HIP -DDEBUG"

Release: rc 0, 52 s, 0 errors, warnings only the known classes (88 `-Wunused-value` nodiscard hipError_t, 12 `-Winconsistent-missing-override`); gpumd 14717192 B, nep 3473696 B, tb 315128 B, gpumd_replica 2401232 B (same sizes as the porter's 374da0df build). DEBUG: rc 0, 37 s; gpumd 14717048 B, nep 3473624 B. (Telemetry: one `compile` entry with exit 2 / 0.004 s is my own race -- the DEBUG make started before its worktree existed -- and one `test` entry with exit 144 is the first `-m "not slow"` pytest, which I killed at ~30% to relaunch it detached; neither is a build or test failure.)

### Tests (real gfx942)
| command (from the repository root) | result | time |
|---|---|---|
| `python3 -m pytest tests_pytest/test_spin_*.py` (release) | **110 passed, 4 failed** | 84 s |
| same 4 failures, `LD_PRELOAD` srand no-op shim | 4 passed | 4 s |
| `python3 -m pytest tests_pytest -m "not slow"` (release) | **801 passed, 4 failed, 480 skipped, 10 deselected** | 1378 s |
| `python3 tests_regression/run_regression.py --repo-root "$PWD" --suite full --baseline "$PWD/src/gpumd" --candidate "$PWD/src/gpumd" --baseline-nep "$PWD/src/nep" --candidate-nep "$PWD/src/nep"` (DEBUG, self-comparison) | **251/256 cases, relations 25/25** | 633 s |
| the 5 failing cases (`--case` x5), srand shim, twice | 5/5, 5/5 | ~40 s each |
| the same 5 plain | 0/5 (same files differ) | ~60 s |
| `python3 tests_regression/replica/test_replica.py` (release gpumd_replica, as-is) | **6/12** | 18 s |
| same, scratch worktree with `examples/nep_train/nep.txt` from cd7f757d and `examples/qnep_train/nep.txt` from eda8e04b, `GPUMD_REPLICA` = release binary (reverted afterwards) | **12/12 OK** (incl. qNEP REMD/PRD ewald and pppm restart) | 38 s |
| 374da0df's Test Plan (two-replica qNEP REMD, `kspace ewald`, run in `agent_space/gpumd-ewald` instead of `/tmp` to stay out of other jobs' paths) | rc 0, `kspace ewald` in `remd_restart.meta` | <10 s |

Every failure classified, none new:
- **rand() reseed (runtime, deferrals `gpumd-hip-runtime-reseeds-rand` / `comgr-llvm-srand-perturbs-host-rand`)**: the 4 `test_corrector_preserves_force_modifiers` cases (sib/glsd x small/large box; the same 4 in both pytest runs, the only failures in `-m "not slow"`), and in `tests_regression` `replicate_velocity_unseeded`, `graphene_hac`, `graphene_nemd_shc`, `graphene_hnemd_shc`, `silicon_liquid_transport` (byte diffs). All 9 pass under the srand no-op shim (the regression 5 twice), and the regression 5 fail again plain.
- **float-atomic tolerance noise**: none this run. `qnep_pppm_future_bec`, `pppm_round_5_7` and `observer_average_then_observe_control`, which hit their bounds in the porter's runs, all passed, as did all 25 relations (none skipped).
- **replica model-header (upstream dd7d0ba8 `format_version 1`)**: the 6 replica failures, each stopping at `REMD/PRD explicit streams support energy NEP4/NEP5 and qNEP4 models`; 12/12 with the legacy-header models.

The new case `ilp_nep_cutoff_below_nep` passed. Compared with the porter's gfx942 runs at 51018103 (250/256 and 248/256, pytest 110/114) and the gfx1100 reviewer's at 374da0df (pytest 110/114, regression subset 29/29), the counts are the same or better and the failure set is a subset of the known classes.

### Pre-completion checks
- `git -C projects/GPUMD/src status --porcelain`: clean (worktree `agent_space/dbg` also clean after reverting the two model files).
- `python3 utils/jargon.py --port GPUMD`: only the 2 known hits in upstream's own 725fe762 ("Merge pull request #1538 from jeffdaily/moat-port", an ancestor of c3d5d8ac; the fork's `master` mirror is still c5597219). `jargon.py -C projects/GPUMD/src --commits c3d5d8ac..374da0df --diff c3d5d8ac...374da0df`: clean.
- Docs: `doc/installation.rst` "Building for AMD GPUs (ROCm/HIP)" documents `make -f makefile.hip [HIP_ARCH=...]`; this round changes no build step a user types.
- CUDA gate: already recorded at this head ("CUDA compile check (nvcc 12.8) at 374da0df", mi355, PASS), not re-run.

PASS at 374da0df on linux-gfx942.

## Validation 2026-11-09 (linux-gfx90a, validator) -- revalidate at 374da0df, PASS

jeff asked for this run. Host: AMD Instinct MI250X/MI250 (gfx90a, CDNA2, wave64), `HIP_VISIBLE_DEVICES=2` (dispatcher-pinned), all GPU runs under `ulimit -c 0`. Toolchain: TheRock pip SDK, HIP 7.16.26385, AMD clang 24.0.0git. `orient.sh` (on `port/GPUMD`) completed normally and named validator (`revalidate`). Fresh clone of AMD-Ecosystem/GPUMD at `projects/GPUMD/src`, detached at **374da0df0d0011867be8a709c3e912d54cdc4eb0**; a detached worktree of the same commit (`agent_space/dbg`) for the DEBUG build, and a second (`agent_space/rep`) for the replica legacy-model run (edit reverted afterwards). No fork edit, no push. Job venv: calorine 4.1, ase 3.29.0, pytest 7.3.2, hypothesis.

### Builds (PR-body commands, gfx90a, concurrently)
    utils/timeit.sh GPUMD compile -- make -C <clone>/src -f makefile.hip HIP_ARCH=gfx90a -j32
    utils/timeit.sh GPUMD compile -- make -C <dbg-worktree>/src -f makefile.hip HIP_ARCH=gfx90a -j32 gpumd nep CFLAGS="-O3 --offload-arch=gfx90a -DUSE_HIP -DDEBUG"

Both rc 0, 0 errors, 63 s each. Release: gpumd 15753424 B, nep 4268272 B, tb 322264 B, gpumd_replica 2532296 B. DEBUG: gpumd 15753272 B, nep 4268208 B.

### Tests (real gfx90a)
- **NEP-spin pytest** (release, from the clone root): `python -m pytest tests_pytest/test_spin_*.py` -> **110 passed, 4 failed**, 69 s. The 4 are the known `test_corrector_preserves_force_modifiers` cases (sib/glsd x small/large box). Rerun with `LD_PRELOAD` of a shim making `srand` a no-op: **4 passed** (114/114 with the reseed suppressed). The `gpuLaneMask` kernels pass on wave64 with HIP's mask checks live.
- **Full `tests_regression`** (DEBUG, self-comparison, PR-body command; overlapped the spin pytest and then the not-slow pytest on the same GPU for its whole length):
  `python3 tests_regression/run_regression.py --repo-root $D --suite full --baseline $D/src/gpumd --candidate $D/src/gpumd --baseline-nep $D/src/nep --candidate-nep $D/src/nep`
  -> **249/256 cases, relations 25/25**, 882 s, whole suite completed (FP64 transport cases included; no `--timeout-scale` needed). `ilp_nep_cutoff_below_nep` PASS. The 7 failures, all 7 rerun with `--case` plain (75 s) and with the srand shim (75 s):

| case | first run | rerun plain | rerun with srand shim | class |
|---|---|---|---|---|
| replicate_velocity_unseeded (final.xyz) | FAIL | FAIL | PASS | rand() reseed (known) |
| graphene_hac (hac.out) | FAIL | FAIL | PASS | rand() reseed (known) |
| graphene_nemd_shc (compute.out) | FAIL | FAIL | PASS | rand() reseed (known) |
| graphene_hnemd_shc (kappa.out) | FAIL | FAIL | PASS | rand() reseed (known) |
| silicon_liquid_transport (adf.out) | FAIL | FAIL | PASS | rand() reseed (known) |
| qnep_ewald_future_bec | dpdt.out field 16, 1.0e-6 vs atol 2e-7 (last printed digit) | PASS | PASS | float-atomic noise (known class; same dpdt.out magnitude as qnep_actions_before_potential on gfx1100) |
| train_nep_pbte_ediff | force_test.out, 2.0e-7 vs atol 2e-7 / rtol 1e-5 (last printed digit) | PASS | PASS | float-atomic noise (known class; same as predict_nep_pbte_batch on gfx942) |

  Plain rerun `Summary: 2 passed, 5 failed`; shim rerun `Summary: 7 passed, 0 failed`. No new failure class; the two numeric case names are new to the list but the signature (one unit in the last printed digit, intermittent, passes on rerun) is the recorded float-atomic class.
- **`tests_pytest -m "not slow"`** (release, whole directory): **801 passed, 4 failed, 480 skipped, 10 deselected**, 2273 s (overlapped the regression run for its first 12 min). The 4 failures are the same four `test_corrector_preserves_force_modifiers` rand() cases; nothing else. Counts identical to the gfx1100 run at this head.
- **Replica** (`GPUMD_REPLICA=<clone>/src/gpumd_replica python3 tests_regression/replica/test_replica.py`, release): as-is **6/12** (15.9 s); the 6 failures are all `Input Error: REMD/PRD explicit streams support energy NEP4/NEP5 and qNEP4 models` (replica_remd.cu:373 / replica_prd.cu:537), the known upstream dd7d0ba8 `format_version 1` model-header class. In the scratch worktree with `examples/nep_train/nep.txt` from cd7f757d and `examples/qnep_train/nep.txt` from eda8e04b: **12/12 OK** (61.7 s), including the qNEP REMD/PRD Ewald and PPPM restart tests that run the commented `ewald.cu` path.

### Gates
- CUDA compile gate: already recorded PASS at 374da0df (mi355, nvcc 12.8); not repeated.
- `python3 utils/jargon.py --port GPUMD` (local `master` from the fork's `origin/master`): only the 2 known hits in upstream's own 725fe762 merge. `jargon.py -C projects/GPUMD/src --commits c3d5d8ac..374da0df --diff c3d5d8ac...374da0df`: clean.
- Documentation: `doc/installation.rst` "Building for AMD GPUs (ROCm/HIP)" documents `make -f makefile.hip` and `HIP_ARCH`; README links it.
- Integrity: `git status --porcelain` empty in the clone and both worktrees; HEAD 374da0df.

Result: **completed** at 374da0df on linux-gfx90a. Every failure is in a known class (rand() reseed at the first launch; float-atomic noise at absolute bounds; replica model headers upstream).

## Validation 2026-11-09 (windows-gfx1151, validator) -- sync round 374da0df, PASSED

jeff asked for this run. GPU: AMD Radeon 8060S (gfx1151, RDNA3.5 APU, wave32), `HIP_VISIBLE_DEVICES=0`, TheRock ROCm SDK (same as the 2026-11-05 Windows record), Windows 11. Fresh clone of AMD-Ecosystem/GPUMD (`agent_space/gpumd-win3`) at **374da0df0d0011867be8a709c3e912d54cdc4eb0** (= `moat-port`, matches `git ls-remote`); `git status --porcelain` clean. Nothing pushed. Serial, one GPU process at a time, `ulimit -c 0`, each case run once.

### Build
    utils/timeit.sh GPUMD compile -- bash agent_space/gp3_build.sh
      # cd src; mingw32-make -f makefile.hip gpumd nep gpumd_replica HIP_ARCH=gfx1151 LDFLAGS="-L<rocm>/lib" CC="<rocm>/bin/hipcc.exe" CFLAGS="-O3 --offload-arch=gfx1151 -DUSE_HIP -std=c++17" -j8 -k
rc 0, 0 errors, one pass (about 9 min). gpumd 15434752 B, nep 4276224 B, gpumd_replica.exe 2537984 B: all three targets build, including the new nep_model_file.cu (`gpu_macro.cuh` include) and the NEP-spin TUs (`gpuLaneMask`, wave32 shift 0). The linker wrote `gpumd` and `nep` without `.exe` (hipcc `-o gpumd` on this SDK); copied to `gpumd.exe`/`nep.exe` for running. Same environment notes as 2026-11-05 (full-path hipcc.exe, `-std=c++17`, -j8, runtime DLLs next to the exe, SDK bin on PATH).

### Tests (real gfx1151)
tests/gpumd no longer exists upstream, so the four earlier cases ran from the scratch tree `agent_space/gp_run3` (copied from the previous record's `gp_run`) with the new binary: `utils/timeit.sh GPUMD test -- bash agent_space/gp3_run_suite.sh`.

| case | rc | stderr | result |
|---|---|---|---|
| ilp_nep_cutoff | 1 | empty | last stdout line `ERROR: Please make sure cutoff of ILP is larger than it of NEP.`, identical to error1.out modulo the CRLF of the Windows checkout (as before) |
| ilp_ctl (control, ILP rcut 4.3) | 0 | empty | completes |
| carbon (NEP4 NVE) | 0 | empty | thermo finite, Etot drift (max-min)/\|mean\| 1.6e-6, std 4.3e-7, T mean 300.40 K |
| graphene_dos | 1 first, 0 after edit | see below | after removing the stale line: dos.out 400x4 and mvac.out 400x4 finite, mvac t0 1,1,1, thermo finite |
| graphene_kappa_emd | 0 | empty | hac.out 100x11 finite, T mean 298.4 +- 10.0 K |

graphene_dos first exit 1: `Input Error: measure/measure.cu:120 dump_velocity has been removed. Use dump_xyz <interval> <filename> velocity instead.` The old test input predates upstream's removal of `dump_velocity`; not a port fault. The line was deleted from the scratch `run.in` and the case rerun once (rc 0, empty stderr).

Regression harness, serial, release binary as baseline and candidate, 8 short light cases, `--program gpumd --timeout-scale 3`:
`python3 tests_regression/run_regression.py --repo-root agent_space/gpumd-win3 --suite full --program gpumd --case <c>... --baseline gpumd.exe --candidate gpumd.exe`
cases: ilp_nep_cutoff_below_nep (new), neighbor_alias_aliased, neighbor_alias_control, nve, nvt_ber, nvt_nhc, npt_ber_iso, nvt_mttk -> **8 passed, 0 failed**; relation neighbor_alias_translation_invariance PASS (the other 24 skipped as their cases were not selected). The new case behaves as on gfx90a/gfx942/gfx1100.

### Not run
The full 256-case suite, the pytest suites, replica tests, `nep` training and the rand()-reseed, float-atomic and replica model-header cases (all known classes on Linux at this head, none run here), to keep load off the APU. No GPU fault, hang or 719 occurred. gpumd_replica was built but not run.

### Gates
`jargon.py -C <clone> --commits c3d5d8ac..374da0df --diff c3d5d8ac...374da0df`: clean (the `--port` scan needs the projects/GPUMD/src clone absent on this host; Linux validators ran it, 2 known hits in upstream's own 725fe762). Docs: `doc/installation.rst` ROCm section documents `makefile.hip`; no build-step change this round. CUDA gate recorded at this head (mi355, PASS). Integrity: clone `git status --porcelain` clean.

State: windows-gfx1151 -> `completed` at 374da0df.

## Sync round 2 2026-11-10 (linux-gfx942, porter) -- merge upstream d6cd2e36, 374da0df -> 3b0f1908

jeff asked for this run. Host: MI300X (gfx942), HIP 7.16.26385 / AMD clang 24.0.0git (TheRock pip SDK), `HIP_VISIBLE_DEVICES=0` (dispatcher-pinned), GPU work under `ulimit -c 0`. `orient.sh` completed normally (no hang) and reported nothing actionable on linux-gfx942 (stage was review-passed, all completed); per the dispatch, `set-state GPUMD linux-gfx942 porting --agent porter` took the lock. `pr-state GPUMD` = `merged` (#1538), so `moat-port` is not frozen; fresh fork clone, `protect-fork` installed.

### Merge
Upstream master fetched by URL (no remote added): **d6cd2e361b4740687162c8fcb347c234d86558f2** (2026-11-10, "Merge pull request #1886 from Liusrepo/dftd3-box-switch"), 42 commits past c3d5d8ac (two past the 8bf530101c named in the dispatch). Highlights: `src/force/`, `integrate/`, `measure/`, `utilities/` split into subfolders (accff11e; upstream updated `makefile.hip`'s wildcards itself); old nep.txt formats dropped and NEP reading routed through `NEP_Model_File` (`utilities/nep_io/`); fixtures moved to `format_version 1`; ILP extended (30be7a46, which also deletes `accumulate_f12` from `nep_utilities.cuh`); replica batching (af8b0971); dftd3 fixes. Merge commit **3b0f1908** `Merge branch 'master' of https://github.com/brucefan1983/GPUMD`, with a one-paragraph conflict note in the body.

Conflicts: exactly one, `src/force/ilp/ilp_nep.cu` (upstream moved it from `src/force/` and replaced the hand-rolled `cutoff` line parsing with `model.rc_radial[0]` etc.; ours changed the `max_nep_rc` line). Resolution: upstream's new code kept verbatim (including its removal of the `// save the max rc_radial` comment), with our line re-applied: `max_nep_rc = fmax(max_nep_rc, double(parambs[i].rc_radial));`. Types checked: `max_nep_rc` is still `double` (`ilp_nep.cuh:226`) and `parambs[i].rc_radial` still `float` (`ilp_nep.cuh:139`, assigned from `NEP_Model_File::rc_radial`, a `std::vector<float>`), so HIP's host `max` still resolves to `max(int,int)` and the fix is not a no-op. Control: the poison compile of the file with upstream's line gives `call to deleted function 'max'` at that line. `git diff d6cd2e36 3b0f1908`: 15 files, +103/-21, exactly the previous round's delta (070d3f53 build restoration, the `fmax` line, the ewald comment, the regression case) on the new tree. Everything else auto-merged (`nep_spin.cu`, `makefile.hip`, `nep_utilities.cuh`, `manifest.json`, `test_manifest.py`).

### HIP build re-check
- `makefile.hip` enumerates sources by wildcard, and upstream already extended them to `*/*.cu` for the new subfolders; `SOURCES_TB` (`utilities/error.cu`, `utilities/main_common.cu`) still exist. No makefile change needed.
- `nep_utilities.cuh` still starts with `#include "gpu_macro.cuh"` (upstream only deleted `accumulate_f12`); `nep_model_file.cu` moved to `utilities/nep_io/` and builds.
- `gpuLaneMask`: the same 11 masked `__syncwarp`/`__shfl_*_sync` sites (4 in `force/nep_spin/nep_spin.cu`, 6 in `nep_spin3_force.cuh`, 2 in `main_nep/nep_spin/nep_spin.cu` with the explicit `warp_size`), plus the pre-existing bare `__syncwarp()`; no new warp intrinsic (`__ballot`, `__activemask`, `__any_sync`, `warpSize`) anywhere outside `main_gnep/`.
- Builds (from the MOAT root, -j56 = physical cores per socket):

      utils/timeit.sh GPUMD compile -- make -C <clone>/src -f makefile.hip HIP_ARCH=gfx942 -j56
      utils/timeit.sh GPUMD compile -- make -C <dbg>/src -f makefile.hip HIP_ARCH=gfx942 -j56 gpumd nep CFLAGS="-O3 --offload-arch=gfx942 -DUSE_HIP -DDEBUG"

  Both rc 0, 34 s, 0 errors; warnings only the known classes (88/84 `-Wunused-value` nodiscard, 12 `-Winconsistent-missing-override`). Release: gpumd 15099704 B, nep 3447344 B, tb 315128 B, gpumd_replica 2604936 B; DEBUG gpumd 15099560 B, nep 3447296 B. Every tracked `.cu` outside `main_gnep/` produced an object (main_replica included).
- Host min/max probe (deleted-template forced include, `hipcc -x hip --cuda-host-only -fsyntax-only -O3 --offload-arch=gfx942 -DUSE_HIP -I./ -include poison.h`, main_replica files from `main_replica/` with `-I./ -I./integrate`), all 229 tracked `.cu` outside `main_gnep/`, four passes (default; the `-DUSE_FIXED_NEIGHBOR ... -DUSE_KEPLER` set; `-DUSE_GRAPHENE_TB`; `-DUSE_CJ -DGNEP_TEST_DIAGNOSTICS`): 229/229 rc 0, **0 deleted-function hits** in every pass. The only new guards in the upstream range are `GPUMD_WPE_ENABLED` and `_WIN32`, with no `min(`/`max(` inside. No new HIP fix, so no `[ROCm]` commit this round.

### Tests (real gfx942, release unless noted; calorine see below)
| command | result |
|---|---|
| `python3 -m pytest tests_pytest/test_spin_*.py` | **110 passed, 4 failed** (94 s); the 4 `test_corrector_preserves_force_modifiers` pass under the `LD_PRELOAD` srand no-op shim (4/4) |
| `python3 -m pytest tests_pytest -m "not slow"` | **866 passed, 4 failed, 638 skipped, 10 deselected** (1465 s); the 4 failures are the same rand() cases. Upstream grew the suite (1508 collected vs 1285 at 374da0df) |
| full `tests_regression` (DEBUG, self-comparison, the PR-body command) | **247/256 cases, relations 25/25** (about 10 min); `ilp_nep_cutoff_below_nep` PASS |
| the 9 failing cases rerun plain | 2 passed, 7 failed (the 5 rand() cases, `qnep_pppm_future_bec` thermo 1.0e-6 vs atol 1e-7 at -16011.045804, `pppm_round_exact` 1.6e-7) |
| the 9 rerun with the srand shim | 8 passed, 1 failed (`pppm_round_exact`, 1.16e-7 vs 1e-7, a different field) |
| `tests_regression/replica/test_replica.py` (`GPUMD_REPLICA` = release binary) | **4/16**; the 12 failures all stop at `REMD/PRD explicit streams support energy NEP4/NEP5 and qNEP4 models` |
| same, scratch worktree with `examples/nep_train/nep.txt` from cd7f757d and `examples/qnep_train/nep.txt` from eda8e04b (reverted after) | **16/16 OK** (includes upstream's new batch-choice tests and the commented ewald path) |

ILP_NEP case control: a scratch worktree with only the `fmax` line reverted to upstream's `max` (gpumd rebuilt) gives `FAIL: ilp_nep_cutoff_below_nep:baseline: expected exit 1, got 0`; with the fix, PASS. The case still expresses the bug and the fix: its fixtures (`graphene_nep_ilp_short_cutoff.txt`, the map, `repo:potentials/nep/C_2022_NEP4.txt`, now `format_version 1` with `cutoff 4.2 3.7 80 47`) are unchanged in the upstream range, `--check-manifest` reports `Manifest OK: 256 cases, 24 suites`, and `test_manifest.py` passes 17.

Classification, none new:
- **rand() reseed** (deferral `gpumd-hip-runtime-reseeds-rand`, ruled defer by jeff 2026-11-09: no GPUMD change): the 4 spin pytest cases and `replicate_velocity_unseeded`, `graphene_hac`, `graphene_nemd_shc`, `graphene_hnemd_shc`, `silicon_liquid_transport`; all pass under the shim and fail again plain.
- **float-atomic noise at absolute bounds**: `qnep_pppm_future_bec` (bec 1.165e-5 vs atol 1e-5; on rerun a thermo field instead), `train_nep_pbte_ediff` (force_test 2.0e-7 vs atol 2e-7 / rtol 1e-5, passed on both reruns; same signature as on gfx90a at 374da0df), `pppm_round_exact` (1.49e-7, 1.63e-7, 1.16e-7 in three runs on different fields; a new case name in the list, same class as the 7 `pppm_*` thermo hits in run C of the 2026-11-05 round), `pppm_round_5_7` (1.15e-7, passed on both reruns).
- **replica model header** (upstream `format_version 1`, platform-independent): the 12 replica failures.
Compared with the 374da0df gfx942 validation (251/256, 25/25; pytest 110/114 and 801 passed / 4 failed): the same rand() set and the same float-noise class; this run hit four float bounds where that one hit none.

Environment notes:
- calorine: upstream moved every `tests_pytest/fixtures/models` file to `format_version 1`, which calorine 4.1 (newest on PyPI) cannot parse (`OSError: Failed to parse line 6 from .../nep_C.txt` in `calorine/nep/model.py`); with it, `-m "not slow"` gave 751 failed / 28 errors, all that parse error. calorine master (gitlab materials-modeling/calorine at 1c94125, 2026-11-10; still reports version 4.1) reads the new format; it was built from a clone in `agent_space/` into this job's own venv. Platform-independent: upstream's integration CI takes `calorine` from PyPI, so reproducing needs the development version until calorine releases.
- The host's root overlay filled to 100% mid-run (another job; 206 MB free after I removed my scratch control worktree and the regression `.work`), which aborted the first full regression at case 15 (`OSError: [Errno 28]` in `prepare_workdir`) and lost tool output for a few minutes. Space recovered by itself (115 GB, later 349 GB) and the full suite was rerun from the start. Not a port issue.

### Gates
`python3 utils/jargon.py --port GPUMD`: the 2 known hits in upstream's own 725fe762 (the fork's `master` mirror is still c5597219; the dispatcher owns it). `jargon.py -C projects/GPUMD/src --commits d6cd2e36..moat-port --diff d6cd2e36...moat-port`: clean. `prose.py` on the merge message and the updated draft PR body, and `jargon.py -` on the body: clean. `git status --porcelain` in the clone: clean. Pushed 374da0df..3b0f1908 (fast-forward), `advance-head` to 3b0f1908, stage `ported`.

CUDA-visible changes this round: none beyond upstream's own and the previously reviewed delta (`git diff d6cd2e36 3b0f1908` is the 374da0df delta against c3d5d8ac, with `ilp_nep.cu` at its new path). The mi355 nvcc compile is owed at 3b0f1908 (expected: device code identical to d6cd2e36, host differences only the `fmax` codegen and `__LINE__` constants). Next: review of the merge, then revalidation of every gate at 3b0f1908. The draft follow-up PR body above (the one superseding the earlier draft) now carries this run's counts and the calorine note.

## CUDA compile check (nvcc 12.8) at 3b0f1908

Helper run on the mi355 fleet host (CUDA 12.8 toolkit, V12.8.93; no NVIDIA GPU), compile only, requested by jeff. Fresh clone of the fork's `moat-port` at **3b0f1908** (merge of upstream d6cd2e36 into 374da0df) vs `git worktree add ../up d6cd2e36` (upstream master, fetched by URL, no remote added). `git diff d6cd2e36 3b0f1908 -- src/` touches 8 files: the 070d3f53 delta (`gpuLaneMask` in `gpu_macro.cuh`, masks and explicit `warp_size` in `force/nep_spin/nep_spin.cu`, `nep_spin3_force.cuh`, `main_nep/nep_spin/nep_spin.cu`, the `gpu_macro.cuh` include in `utilities/nep_utilities.cuh`), the `fmax` line in `force/ilp/ilp_nep.cu` (new path), the three-line comment in `main_replica/force/ewald.cu`, and `makefile.hip`, which the CUDA build does not read.

Recipe: same as the 374da0df entry (upstream's unmodified `src/makefile`, default `all`, `nvcc -std=c++14 -O3 -arch=sm_60 -Xcompiler="<makefile WARN_FLAGS>" -I./`), both trees built at the same time:

    export PATH=/usr/local/cuda-12.8/bin:$PATH
    utils/timeit.sh GPUMD compile -- make -C <clone>/src -j64 -O      # 3b0f1908, 19.6 s
    make -C <clone-up>/src -j64 -O                                    # d6cd2e36

| | 3b0f1908 | d6cd2e36 |
|---|---|---|
| make rc | 0 | 0 |
| nvcc invocations (compile / link) | 229 / 4 | 229 / 4 |
| objects | 229 `.o` | same 229 paths |
| executables (in `src/`) | `gpumd` 24758096 B, `nep` 7972632 B, `tb` 1552152 B, `gpumd_replica` 4225736 B | same sizes |
| nvcc errors | 0 | 0 |
| nvcc warnings | 233, all `Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed` (one per invocation) | identical 233 |
| host compiler warnings (makefile's -Wall -Wextra set) | none | none |

The object count is down by one from the 374da0df entry (230) and the sizes changed because upstream reorganized `src/` in this range; both trees agree. Whole logs, sorted, with only the absolute tree prefix normalized: identical, so the nvcc diagnostics are identical for every file.

Device code: the same section-hash method (`cuobjdump -xelf all` / `-xptx all` on all 229 objects; SHA-1 of every `.text.*`, `.nv.constant*`, `.nv.shared*`, `.nv.global*`, `.nv.info*` cubin section, read by a Python ELF parser, plus the full PTX, with the `_GLOBAL__N__<8hex>_` / `_INTERNAL_<8hex>_` tokens normalized). **All 4019 records identical: 855 kernel `.text` sections (9214400 B) and 229 PTX files.** The SHA-1 of empty input appears only on NOBITS sections (483 `.nv.shared*`, 359 of them size 0; 21 `.nv.global`), whose file body is empty by definition; no PROGBITS section hashed empty. (The 374da0df entry's NOBITS breakdown was 123 `.nv.shared*`; it probably counted only the non-empty ones, 124 here, but I have no script from that run to confirm. The comparison itself does not depend on that count.) Per changed TU: `force/nep_spin/nep_spin.o` 43 kernels / 118328 PTX lines, `force/ilp/ilp_nep.o` 11 / 17403, `main_nep/nep_spin/nep_spin.o` 2 / 783, `main_replica/force/ewald.o` 2 / 907; all identical.

Host code (`objdump -dr --no-show-raw-insn`; only instruction addresses, branch/symbol offsets and the hash tokens are normalized, immediates kept). Three objects differ; the other 226 are identical:
- `force/ilp/ilp_nep.o` (the `fmax` line): 151 hunks, 354 lines, more than the 8 lines at 374da0df because upstream rewrote the constructor around the line (`NEP_Model_File` parsing), so the changed codegen now cascades through register allocation. Every differing line is in `ILP_NEP::ILP_NEP` (`C1` and its `.cold` part); the substance is the same xmm0/xmm1 swap around `cvtss2sd 0x14(%rax,%r14,1)` (`double(parambs[i].rc_radial)`), and the rest are %rcx/%rdi renamings, spill-slot moves and alignment padding. Relocations (as a multiset) and the `call` sequence are identical, and every section has the same size (`.text` 42495 B, object total 1008512 B in both).
- `force/nep_spin/nep_spin.o`: 47 `mov $imm,%ecx` constants (upstream 1261..2178), each +1. These are the `CHECK` macro's `__LINE__` after the line-wrapped `__shfl_down_sync` at line 834 (+5/-4 in that file). Undefined symbols are identical; `.text` 94875 B in both.
- `main_replica/force/ewald.o`: two `__LINE__` constants, `0x117`/`0x114` and `0x12f`/`0x12c` (279/276, 303/300), the `GPU_CHECK_KERNEL` sites after `find_k_and_G` shifted +3 by the comment. Undefined symbols are identical; `.text` 10148 B in both.
No other host differences: `main_nep/nep_spin/nep_spin.o` and every includer of `nep_utilities.cuh`/`gpu_macro.cuh` are identical (the edits there do not change any line count that `__LINE__` reaches, and `gpuLaneMask(m)` is `(m)` on CUDA).

Not covered (as before): `main_nep/nep_spin/spin_specialized.cu` under `NEP_SPECIALIZED_RUNTIME_BUILD`. PASS: at 3b0f1908 the CUDA path builds with nvcc 12.8 exactly as upstream d6cd2e36 does, with identical device code and PTX. Host code differs only by the `fmax` codegen in `ilp_nep.o` and by `__LINE__` constants in `nep_spin.o` and `ewald.o`; there is no CUDA behaviour change against upstream. No platform states changed.
## Review 2026-11-10 (linux-gfx1100, reviewer) -- sync round 2, merge 374da0df -> 3b0f1908, REVIEW PASSED

jeff asked for this run. Host: RX 7900 XTX (gfx1100, wave32), HIP 7.16.26385 / AMD clang 24.0.0git (TheRock pip SDK), `HIP_VISIBLE_DEVICES=0` (dispatcher-pinned), `ulimit -c 0`. `orient.sh` completed normally and named reviewer; `set-state GPUMD linux-gfx1100 reviewing --agent reviewer` took the lock. Fresh fork clone, `moat-port` = 3b0f1908 (parents 374da0df, d6cd2e36); upstream d6cd2e36 fetched by URL (no remote added).

No findings.

What was checked:
- **Conflict resolution** (`src/force/ilp/ilp_nep.cu:203`): `git diff d6cd2e36 3b0f1908 -- src/force/ilp/ilp_nep.cu` is the single line `max(max_nep_rc, parambs[i].rc_radial)` -> `fmax(max_nep_rc, double(parambs[i].rc_radial))`; the surrounding upstream code (`model.rc_radial[0]`, `model.rc_angular[0]`, `model.max_radial`) is verbatim. `max_nep_rc` is still `double` (`ilp_nep.cuh`, `double max_nep_rc;`), `ILP_NEP` `rc_radial` still `float` (`ilp_nep.cuh:139`), `NEP_Model_File::rc_radial` is `std::vector<float>` (`utilities/nep_io/nep_model_file.cuh:54`), so the fix is still needed.
- **Delta identity**: `git diff c3d5d8ac 374da0df` and `git diff d6cd2e36 3b0f1908`, with `index` lines and hunk headers stripped and `src/force/ilp_nep.cu` mapped to its new path, have identical `+`/`-` lines (15 files, +103/-21 both); only three context lines in `ilp_nep.cu` differ, which are upstream's rewrite. Nothing of upstream's is reverted: the merge's diff against d6cd2e36 contains nothing but the reviewed delta. The auto-merged `makefile.hip` carries upstream's new `*/*.cu`/`*/*.cuh` wildcards and our `gpumd_replica`/`clean_replica` targets; `main_replica/makefile` gained only `replica_batch.cu` upstream and still links `-lcufft` only (our `LIBS="-lhipfft"` override still covers it).
- **Upstream range spot-check** (`git diff -M c3d5d8ac d6cd2e36 -- src`, 42 commits, 35 non-merge): no added `__shfl*`, `__ballot`, `__activemask`, `__any`/`__all`, `__syncwarp`, `warpSize`, 32-bit lane masks, `& 31`/`% 32`/`>> 5` lane math, `__ldg`, inline asm, textures, or new CUDA libraries. The new replica-batch reductions (`main_replica/replica_batch.cu`) are `__syncthreads`-based tree reductions over `blockDim.x`. The new `gpuMemcpy2DAsync` (`main_replica/utilities/gpu_macro.cuh:31`, used at `replica_batch.cu:389`) is device-to-device only, so the pageable-2D-source hazard does not apply; the new `gpuMemcpyAsync` D2H copies in `main_replica/force/nep.cu` are followed by `gpuStreamSynchronize`. Host min/max in new code is `std::max`. The only `_WIN32` additions are in `utilities/nep_io/compact_nep.cu` (`<process.h>`/`_getpid` vs `<unistd.h>`/`getpid`), correctly guarded. The porter's full host min/max probe was not redone.
- **Merge commit message**: `jargon.py -`: clean; `prose.py`: clean. `jargon.py -C projects/GPUMD/src --commits d6cd2e36..moat-port --diff d6cd2e36...moat-port`: clean.
- **Draft follow-up PR** (title and body under "Draft follow-up PR for jeff ... supersedes the earlier draft", internal reminder excluded): `jargon.py -`: clean; `prose.py`: clean. Every count matches the porter's sync round 2 table: spin 110/114; `-m "not slow"` 866 passed with the same 4 failures; regression 247/256 and relations 25/25; 9 failures = 5 rand() cases + `qnep_pppm_future_bec`, `pppm_round_exact`, `pppm_round_5_7`, `train_nep_pbte_ediff`; magnitudes (BEC 1.17e-5, pppm 1.16e-7 to 1.63e-7 quoted as 1.1e-7 to 1.6e-7, force 2.0e-7) and "passed on a rerun" for the three named cases (two on the plain rerun, `qnep_pppm_future_bec` on the shim rerun). The calorine sentence matches the environment note (calorine 4.1 fails on `format_version 1`; master 1c94125 reads it). The reminder correctly says every gate and the mi355 nvcc compile are owed at 3b0f1908.

### Build and tests on gfx1100 at 3b0f1908
Release in the fork clone, DEBUG in a detached worktree at 3b0f1908 (`agent_space/dbg`), from the MOAT root:

    utils/timeit.sh GPUMD compile -- make -C projects/GPUMD/src/src -f makefile.hip HIP_ARCH=gfx1100 -j10
    utils/timeit.sh GPUMD compile -- make -C agent_space/dbg/src -f makefile.hip HIP_ARCH=gfx1100 -j10 gpumd nep CFLAGS="-O3 --offload-arch=gfx1100 -DUSE_HIP -DDEBUG"

Both rc 0, 0 errors; warnings only the known classes (88 `-Wunused-value`, 12 `-Winconsistent-missing-override`). Release: gpumd 15510928 B, nep 3848608 B, tb 309904 B, gpumd_replica 2494264 B.

calorine from source at the porter's commit (gitlab materials-modeling/calorine 1c94125), cloned into `agent_space/` and installed into this job's venv only.

| command | result | porter gfx942 |
|---|---|---|
| `python -m pytest tests_pytest/test_spin_*.py` (release) | **110 passed, 4 failed** (54 s) | 110/4 |
| the 4 `test_corrector_preserves_force_modifiers` cases under an `LD_PRELOAD` srand no-op shim | 4/4 passed | 4/4 |
| `run_regression.py --suite full` (DEBUG, self-comparison) with `--case` for `ilp_nep_cutoff_below_nep ilp_nep_compute ilp_tersoff_compute ilp_tmd_sw_compute qnep_ewald_future_bec qnep_pppm_future_bec qnep_force_accessors qnep_actions_before_potential kspace_single_late multi_nep_observer nep89_two_runs temperature_nep_two_run_ramp predict_nep_pbte_batch train_nep_pbte_batch train_nep_pbte_ediff train_qnep1_pbte_batch train_qnep2_water_fff` | **16 passed, 1 failed** (relations skipped as a subset); `ilp_nep_cutoff_below_nep` PASS | all 17 pass in the full run except `qnep_pppm_future_bec` and `train_nep_pbte_ediff` (float noise) |
| `qnep_ewald_future_bec` rerun 3x | PASS, FAIL, PASS | -- |

The one failure, `qnep_ewald_future_bec` (`dpdt.out` field 16, 0.161695 vs 0.161696, 1.0e-6 vs atol 2e-7: one unit in the sixth printed decimal), is the known float-atomic class: the same field and magnitude as on gfx1100 in the 2026-11-05 run and on gfx90a at 374da0df. It flips between runs and is not in the merge's code. `qnep_pppm_future_bec` and `train_nep_pbte_ediff`, which failed on gfx942, passed here.

The mi355 nvcc compile at 3b0f1908 is still owed before presenting (porter's note). State: `review-passed`.
## Validation 2026-11-10 (linux-gfx1100, validator) -- revalidate at 3b0f1908, PASS

jeff asked for this run. Host: AMD Radeon RX 7900 XTX (gfx1100, RDNA3, wave32), `HIP_VISIBLE_DEVICES=0` (dispatcher-pinned), all GPU runs under `ulimit -c 0`. Toolchain: TheRock pip SDK, HIP 7.16.26385, AMD clang 24.0.0git. `orient.sh` (on `port/GPUMD`) completed normally and named validator (`revalidate`). Fresh clone of AMD-Ecosystem/GPUMD detached at **3b0f1908a32b260e2446201833d4cc19dce6321e** (= `origin/moat-port`; parents 374da0df, d6cd2e36); detached worktrees of the same commit for the DEBUG build (`agent_space/dbg`) and the replica legacy-model run (`agent_space/rep`). No fork edit, no push. Job venv: calorine built from source at gitlab materials-modeling/calorine **1c9412504bf1** (the porter's commit; still reports version 4.1), from a clone in `agent_space/`, into this job's venv only; ase 3.29.0, pytest 7.3.2, hypothesis 6.56.4.

### Builds (PR-body commands, gfx1100, from the MOAT root)
    utils/timeit.sh GPUMD compile -- make -C projects/GPUMD/src/src -f makefile.hip HIP_ARCH=gfx1100 -j16
    utils/timeit.sh GPUMD compile -- make -C agent_space/dbg/src -f makefile.hip HIP_ARCH=gfx1100 -j16 gpumd nep CFLAGS="-O3 --offload-arch=gfx1100 -DUSE_HIP -DDEBUG"

Both rc 0, 0 errors; warnings only the known classes (88 `-Wunused-value` nodiscard `hipError_t`, 12 `-Winconsistent-missing-override`). Release 46 s: gpumd 15510928 B, nep 3848608 B, tb 309904 B, gpumd_replica 2494264 B (identical sizes to the 2026-11-10 gfx1100 review build of this commit). DEBUG 45 s: gpumd 15510800 B, nep 3844456 B.

### Tests (real gfx1100)
| command | result | 374da0df on gfx1100 (2026-11-09) |
|---|---|---|
| `python -m pytest tests_pytest/test_spin_*.py` (release) | **110 passed, 4 failed**, 54 s; the 4 are `test_corrector_preserves_force_modifiers` (sib/glsd x small/large box) | 110/4 |
| the 4 under an `LD_PRELOAD` srand no-op shim (`void srand(unsigned){}`) | **4 passed** | 4/4 |
| full `tests_regression` (DEBUG, self-comparison, PR-body command plus `--timeout-scale 3`): `python3 tests_regression/run_regression.py --repo-root $D --suite full --baseline $D/src/gpumd --candidate $D/src/gpumd --baseline-nep $D/src/nep --candidate-nep $D/src/nep --timeout-scale 3` | **completed: 248 passed, 8 failed of 256; relations 25/25**, 429 s (whole suite including the FP64 graphene/silicon transport cases; overlapped the not-slow pytest on the same GPU). `ilp_nep_cutoff_below_nep` PASS | 249/256, 25/25, 442 s |
| the 8 rerun plain (`--case` x8) | 2 passed, 6 failed (5 rand() cases + `qnep_ewald_future_bec`) | -- |
| the 8 rerun under the srand shim | 7 passed, 1 failed (`qnep_actions_before_potential`, a different field) | -- |
| `python -m pytest tests_pytest -m "not slow"` (release, whole directory) | **866 passed, 4 failed, 638 skipped, 10 deselected**, 1021 s; the 4 are the same rand() cases, no other failure | 801/4/480 (suite grew upstream; 866/4/638 matches the porter's gfx942 count at 3b0f1908) |
| `GPUMD_REPLICA=<clone>/src/gpumd_replica python3 tests_regression/replica/test_replica.py` (release) | **4/16**; all 12 failures stop at `REMD/PRD explicit streams support energy NEP4/NEP5 and qNEP4 models` (8 REMD, 4 PRD) | 6/12 (fewer tests then) |
| same in `agent_space/rep` with `examples/nep_train/nep.txt` from cd7f757d and `examples/qnep_train/nep.txt` from eda8e04b (reverted after) | **16/16 OK**, 28.9 s (includes upstream's new batch-choice/batched tests and the qNEP REMD/PRD Ewald and PPPM restart tests) | 12/12 |

Regression failures, classified:
| case | full run | plain rerun | shim rerun | class |
|---|---|---|---|---|
| replicate_velocity_unseeded (final.xyz) | FAIL | FAIL | PASS | rand() reseed (known) |
| graphene_hac (hac.out) | FAIL | FAIL | PASS | rand() reseed (known) |
| graphene_nemd_shc (compute.out) | FAIL | FAIL | PASS | rand() reseed (known) |
| graphene_hnemd_shc (kappa.out) | FAIL | FAIL | PASS | rand() reseed (known) |
| silicon_liquid_transport (adf.out) | FAIL | FAIL | PASS | rand() reseed (known) |
| qnep_ewald_future_bec | dpdt.out field 16, 1.0e-6 vs atol 2e-7 | same field, 1.0e-6 | PASS | float-atomic noise (known; same field and magnitude as the 2026-11-10 review) |
| qnep_pppm_future_bec | bec 1.80e-5 vs atol 1e-5; thermo field 7, 1.0e-6 vs 1e-7 | PASS | PASS | float-atomic noise (known) |
| qnep_actions_before_potential | dpdt.out field 23, 1.0e-6 vs atol 2e-7 | PASS | dpdt.out field 15, 1.0e-6 | float-atomic noise (known; same signature as 374da0df on gfx1100) |

No new failure class. Note for the draft PR body: on gfx1100 the float-noise set is `qnep_ewald_future_bec`, `qnep_pppm_future_bec` and `qnep_actions_before_potential` (dpdt one unit in the sixth printed decimal; BEC up to 1.8e-5 against 1e-5), not gfx942's `pppm_round_*`/`train_nep_pbte_ediff`; the body's class wording covers both, but its quoted BEC magnitude (1.17e-5) is gfx942's, and the body describes an MI300X run only.

### Gates
- CUDA compile gate: already recorded PASS at 3b0f1908 (mi355, nvcc 12.8, entry above); not repeated.
- `python3 utils/jargon.py --port GPUMD` (local `master` created from the fork's `origin/master` c5597219): the 2 known hits in upstream's own 725fe762 merge only. `jargon.py -C projects/GPUMD/src --commits d6cd2e36..3b0f1908 --diff d6cd2e36...3b0f1908`: clean.
- Documentation: `doc/installation.rst` documents `make -f makefile.hip` and `HIP_ARCH=gfx1100`; README points to the installation guide for ROCm/HIP.
- Integrity: `git status --porcelain` empty in the clone and both worktrees; HEAD 3b0f1908.

Result: **completed** at 3b0f1908 on linux-gfx1100. Every failure is in a known class (rand() reseed, float-atomic qNEP noise at absolute bounds, replica `format_version 1` model headers upstream).

## Validation 2026-11-10 (linux-gfx90a, validator) -- sync round 2 at 3b0f1908, VALIDATION FAILED (new class: batched replica not bit-identical on gfx90a)

jeff asked for this run. Host: AMD Instinct MI250X/MI250 (gfx90a, CDNA2, wave64), `HIP_VISIBLE_DEVICES=2` (dispatcher-pinned), all GPU runs under `ulimit -c 0`. Toolchain: TheRock pip SDK, HIP 7.16.26385, AMD clang 24.0.0git. `orient.sh` on `port/GPUMD` completed normally (no hang) and named validator (`revalidate`). Fresh clone of AMD-Ecosystem/GPUMD at `projects/GPUMD/src`, detached at **3b0f1908a32b260e2446201833d4cc19dce6321e** (= `origin/moat-port`; parents 374da0df, d6cd2e36). Detached worktrees of the same commit: `agent_space/dbg` (DEBUG build), `agent_space/rep` (replica legacy-model runs and flag experiments; model edits reverted), `agent_space/dbgon` (flag experiment). No fork edit, no push. Job venv only: ase 3.29.0, pytest 7.3.2, hypothesis 6.56.4, and calorine built from a source clone of gitlab materials-modeling/calorine at **1c94125** (the porter's commit; reports version 4.1), which parses the `format_version 1` fixtures.

### Builds (PR-body commands, gfx90a, concurrently)
    utils/timeit.sh GPUMD compile -- make -C projects/GPUMD/src/src -f makefile.hip HIP_ARCH=gfx90a -j32
    utils/timeit.sh GPUMD compile -- make -C agent_space/dbg/src -f makefile.hip HIP_ARCH=gfx90a -j32 gpumd nep CFLAGS="-O3 --offload-arch=gfx90a -DUSE_HIP -DDEBUG"

Both rc 0, 0 errors, 63 s / 64 s. Warnings only the known classes (88 `-Wunused-value`, 12 `-Winconsistent-missing-override`). Release: gpumd 16139976 B, nep 4241928 B, tb 322264 B, gpumd_replica 2756488 B. DEBUG: gpumd 16139832 B, nep 4241872 B.

### Tests (real gfx90a)
| command | result | 374da0df on gfx90a |
|---|---|---|
| `python3 -m pytest tests_pytest/test_spin_*.py` (release, clone root) | **110 passed, 4 failed** (68 s); the 4 `test_corrector_preserves_force_modifiers` (sib/glsd x small/large box) | 110/4 |
| the 4, `LD_PRELOAD` srand no-op shim (`void srand(unsigned){}`) | **4 passed** (3.5 s) | 4/4 |
| `python3 -m pytest tests_pytest -m "not slow"` (release) | **866 passed, 4 failed, 638 skipped, 10 deselected** (2603 s, overlapped the regression run and the flag experiments on the same GPU); the 4 failures are the same rand() cases | 801/4/480 (suite grew upstream); identical to the porter's gfx942 count at 3b0f1908 |
| full `tests_regression` (DEBUG, self-comparison, PR-body command: `python3 $D/tests_regression/run_regression.py --repo-root $D --suite full --baseline $D/src/gpumd --candidate $D/src/gpumd --baseline-nep $D/src/nep --candidate-nep $D/src/nep`) | **250/256 cases, relations 25/25** (881 s, whole suite incl. FP64 transport cases, no `--timeout-scale`); `ilp_nep_cutoff_below_nep` PASS | 249/256, 25/25 |
| the 6 failures rerun with `--case` plain | 1 passed, 5 failed (147 s) | -- |
| the 6 rerun with the srand shim | **6 passed** (204 s) | -- |
| `GPUMD_REPLICA=<clone>/src/gpumd_replica python3 tests_regression/replica/test_replica.py` (release, as-is) | **4/16** (12.6 s); the 12 failures all stop at `REMD/PRD explicit streams support energy NEP4/NEP5 and qNEP4 models` | 6/12 (suite grew upstream) |
| same in `agent_space/rep` with `examples/nep_train/nep.txt` from cd7f757d and `examples/qnep_train/nep.txt` from eda8e04b | **13/16** (60.7 s): FAIL `test_batched_matches_per_slot_remd`, `test_batched_matches_per_slot_prd`, `test_split_resume_is_exact (batch=1)` | 12/12 (these tests are new); porter gfx942 16/16 |

Regression failure classification (known):
- rand() reseed (deferral `gpumd-hip-runtime-reseeds-rand`, ruled defer): `replicate_velocity_unseeded`, `graphene_hac`, `graphene_nemd_shc`, `graphene_hnemd_shc`, `silicon_liquid_transport` (text diffs); fail again plain, pass under the shim. Same 5 as at 374da0df.
- float-atomic noise at absolute bounds: `qnep_pppm_future_bec` (bec.xyz field 22735, 1.07e-5 vs atol 1e-5, rel 6.5e-7); passed on the plain rerun.

### New failure: batched replica integration is not bit-identical to per-slot on gfx90a
The three failing replica tests are upstream's new ones (af8b0971, replica batching). They assert that `GPUMD_REPLICA_BATCH=1` reproduces the per-slot path bit for bit. On gfx90a they fail in 4 of 4 runs (the full suite plus 3 reruns of the three tests). Diagnosis (scratch probe `agent_space/scripts/batch_det.py`, which reuses the test class: one 20-step start, then four 40-step resumes b0a/b0b at batch 0 and b1a/b1b at batch 1):
- batch 0 vs batch 0: identical; batch 1 vs batch 1: identical; batch 0 vs batch 1: differ (remd and prd). So each path is deterministic and this is not atomic-order noise. It is a systematic difference between the paths.
- Magnitude: the first differing `remd_thermo.out` row has potential energy -7463.528158664703 vs -7463.528137683868 (2.8e-9 relative); the temperature columns are identical.
- Whole `gpumd_replica` rebuilt with `-ffp-contract=off`: batch 0 == batch 1 for remd and prd. Bisected by recompiling one TU at a time with `-ffp-contract=off` and the rest default: `replica_batch.cu` alone: still differs; `integrate/ensemble_bdp.cu` alone: still differs; **`force/nep.cu` alone: identical**. `force/nep.cu` with **`-ffp-contract=on`** alone (the nvcc `--fmad=true` equivalent): identical.
- Cause: the NEP kernels in `main_replica/force/nep.cu` (`find_neighbor_list_large_box`, `find_descriptor`, `find_force_radial`, `find_partial_force_angular`, `find_force_ZBL`) are templates on `Box_View`, instantiated for `Box` (per-slot) and `Batch_Box` (batched; `model/batch_box.cuh`). hipcc's default `-ffp-contract=fast` contracts across statements, and on gfx90a the two instantiations get different FMA formation, so the forces differ in the last bits and the trajectories diverge. nvcc contracts within an expression only. This is the recorded fault class in `cuda-to-rocm` references/fault-classes.md ("clang for HIP defaults to `-ffp-contract=fast`"; CV-CUDA needed `-ffp-contract=on`). The porter's gfx942 run passed 16/16 with the default flags, so the codegen difference is arch-dependent. The upstream code is not wrong. Its bitwise test only holds when both instantiations contract the same way.
- Whole `gpumd_replica` built with `-ffp-contract=on` (`make -f makefile.hip HIP_ARCH=gfx90a -j32 gpumd_replica CFLAGS="-O3 --offload-arch=gfx90a -DUSE_HIP -ffp-contract=on"`), legacy models: **16/16 OK** (55.9 s).
- Side observation: the `-ffp-contract=off` replica build makes the four `test_qnep_comments_and_restart_solver_validation` subtests stop at `multi_replica encountered a non-finite thermodynamic value` / `FIRE minimization encountered a non-finite value` (12/16). The default and `on` builds pass them. Not pursued; it argues against `off`.
- Main-suite effect of `on` (DEBUG gpumd/nep built with `-ffp-contract=on` in `agent_space/dbgon`, same regression command): **250/256, relations 25/25**. The failures are the same 5 rand() cases plus `train_charge_vdw_mixed` (loss.out text diff), which passed 2 of 2 reruns (intermittent training noise). `qnep_pppm_future_bec` passed. No deterministic regression from the flag.

Recommendation for the porter: add `-ffp-contract=on` to `makefile.hip`'s `CFLAGS`, matching nvcc's default `--fmad=true` contraction, as the skill's fault class advises ("match upstream's setting"). The `gpumd_replica` target inherits `CFLAGS`. The CUDA build does not read `makefile.hip`. This is a build-file change, so every gate revalidates. Also run the replica batch tests on gfx942/gfx1100 with the flag. Whether to take the fix or record the bitwise tests as a known gfx90a exception is jeff's call if the porter disagrees. This run counts them as a port failure because a documented, cheap build-file fix exists and makes upstream's test pass on this arch.

### Gates
- CUDA compile gate: recorded PASS at 3b0f1908 (mi355, nvcc 12.8); not repeated.
- `python3 utils/jargon.py --port GPUMD` (local `master` created from the fork's `origin/master`): only the 2 known hits in upstream's own 725fe762 merge. `jargon.py -C projects/GPUMD/src --commits d6cd2e36..3b0f1908 --diff d6cd2e36...3b0f1908`: clean.
- Documentation: `doc/installation.rst` "Building for AMD GPUs (ROCm/HIP)" documents `make -f makefile.hip` and `HIP_ARCH`; README links the installation guide.
- Integrity: tracked files unmodified in the clone and all three worktrees; HEAD 3b0f1908 in each.

Result: **validation-failed** at 3b0f1908 on linux-gfx90a. Everything else matches the 374da0df record and the porter's gfx942 counts at this head: spin 110/114 (114 with the shim), `-m "not slow"` 866 passed with the 4 rand() cases, regression 250/256 + 25/25 with the 6 failures in known classes. The only blocker is the new batched-replica bit-identity failure above.

## Validation 2026-11-10 (linux-gfx942, validator) -- revalidate at 3b0f1908, PASSED

jeff asked for this run. Host: MI300X (gfx942, CDNA3, wave64), HIP 7.16.26385 / AMD clang 24.0.0git (TheRock pip SDK), `HIP_VISIBLE_DEVICES=0` (dispatcher-pinned), all GPU work under `ulimit -c 0`. `orient.sh` completed normally and named validator (`state=revalidate`). Fresh clone of AMD-Ecosystem/GPUMD detached at **3b0f1908a32b260e2446201833d4cc19dce6321e** (= `origin/moat-port`); DEBUG build in a detached `git worktree` at the same sha (`agent_space/dbg`), as in the 374da0df validation. No fork edit, no push. Test environment: job venv with calorine built from source at gitlab materials-modeling/calorine **1c9412504bf1** (the porter's commit; reports version 4.1; needed for the `format_version 1` fixtures), ase 3.29.0, pytest 7.3.2, hypothesis 6.56.4. The carry-forward shortcut does not apply (an upstream merge changes code everywhere), so this was a full run.

### Builds (PR body commands, gfx942, -j56)
    utils/timeit.sh GPUMD compile -- make -C projects/GPUMD/src/src -f makefile.hip HIP_ARCH=gfx942 -j56
    utils/timeit.sh GPUMD compile -- make -C agent_space/dbg/src -f makefile.hip HIP_ARCH=gfx942 -j56 gpumd nep CFLAGS="-O3 --offload-arch=gfx942 -DUSE_HIP -DDEBUG"

Both rc 0 (36 s / 34 s, run concurrently), 0 errors; warnings only the known classes (88/84 `-Wunused-value` nodiscard hipError_t, 12 `-Winconsistent-missing-override`). Release: gpumd 15099704 B, nep 3447344 B, tb 315128 B, gpumd_replica 2604936 B; DEBUG gpumd 15099560 B, nep 3447296 B. Every size equals the porter's 3b0f1908 build.

### Tests (real gfx942, from the repository root, GPU otherwise idle unless noted)
| command | result | time |
|---|---|---|
| `python3 tests_regression/run_regression.py --repo-root "$PWD" --suite full --baseline "$PWD/src/gpumd" --candidate "$PWD/src/gpumd" --baseline-nep "$PWD/src/nep" --candidate-nep "$PWD/src/nep"` (DEBUG, self-comparison) | **250/256 cases, relations 25/25** (0 skipped); `ilp_nep_cutoff_below_nep` PASS | 599 s |
| the 6 failing cases (`--case` x6), plain | 0/6 (same 6 fail) | 44 s |
| same, `LD_PRELOAD` srand no-op shim (`void srand(unsigned){}`), twice | 5/6, 5/6 (only `qnep_pppm_future_bec` fails) | 42 s, 43 s |
| `python3 -m pytest tests_pytest/test_spin_*.py` (release) | **110 passed, 4 failed** | 79 s |
| the 4 failures (`-k test_corrector_preserves_force_modifiers`) under the srand shim | 4 passed | 5 s |
| `python3 -m pytest tests_pytest -m "not slow"` (release; the replica runs below overlapped its first minutes) | **866 passed, 4 failed, 638 skipped, 10 deselected** | 1478 s |
| `GPUMD_REPLICA=$PWD/src/gpumd_replica python3 tests_regression/replica/test_replica.py` (as-is) | **4/16** (12 failures) | 23 s |
| same, DEBUG worktree with `examples/nep_train/nep.txt` from cd7f757d and `examples/qnep_train/nep.txt` from eda8e04b, release replica binary (reverted afterwards, worktree clean) | **16/16 OK** | 56 s |

Every failure classified, none new:
- **rand() reseed** (deferral `gpumd-hip-runtime-reseeds-rand`, ruled defer): the 4 `test_corrector_preserves_force_modifiers` cases (sib/glsd x small/large box; the same 4 in both pytest runs, the only failures in `-m "not slow"`), and in `tests_regression` `replicate_velocity_unseeded`, `graphene_hac`, `graphene_nemd_shc`, `graphene_hnemd_shc`, `silicon_liquid_transport` (text diffs). All 9 pass under the srand shim (the regression 5 twice) and the regression 5 fail again plain.
- **float-atomic noise at absolute bounds**: `qnep_pppm_future_bec` only, which failed in all 4 runs above (full, plain rerun, both shim reruns), each time on different fields just past the bound: bec.xyz 1.15e-5 to 1.17e-5 against atol 1e-5 (fields 21, 22727), thermo.out 1.0e-7 to 1.6e-7 against atol 1e-7 (fields 9, 30). Because it passed in the 374da0df validation on this host, I ran an A/B to check whether the merge changed it: DEBUG 374da0df built in a second worktree with the same command, then the case alone (`--case qnep_pppm_future_bec`), alternating 3b0f1908/374da0df, 6 rounds. **3b0f1908: 0/6 passed; 374da0df: 2/6 passed.** The failing fields (bec 13/22727/22735, thermo 8/9/30/48) and magnitudes (bec 1.1e-5 to 1.8e-5, thermo 1.0e-7 to 2.0e-7) are the same in both builds, so this is the known class and predates the merge: 374da0df shows the same noise on the same fields. 3b0f1908 failed more often (10 of 10 runs vs 4 of 6), which is not enough to call a regression with the same signature. The port's delta does not touch PPPM or qNEP; upstream's range does change code around them. `train_nep_pbte_ediff`, `pppm_round_exact` and `pppm_round_5_7`, which hit their bounds in the porter's run, passed here.
- **replica model header** (upstream `format_version 1`, platform-independent): the 12 replica failures (8 `REMD explicit streams support energy NEP4/NEP5 and qNEP4 models.`, 4 `PRD ...`); 16/16 with the legacy-header models, including upstream's new batch-choice and batched-vs-per-slot tests and the qNEP ewald/pppm restart paths.

Compared with the 374da0df gfx942 validation (251/256, 25/25; spin 110/114; `-m "not slow"` 801 passed / 4 failed of the smaller pre-merge suite; replica 6/12 as-is, 12/12 legacy) and the porter's 3b0f1908 gfx942 run (247/256, 25/25; spin 110/114; 866/4/638/10; replica 4/16, 16/16): same rand() set, same replica class, pytest counts identical to the porter's, and the regression failure set a subset of the porter's.

### Pre-completion checks
- `git -C projects/GPUMD/src status --porcelain`: no modified tracked files (both worktrees removed after the run).
- `python3 utils/jargon.py --port GPUMD` (after creating local `master`/`moat-port` refs from origin in the detached clone): only the 2 known hits in upstream's own 725fe762 ("Merge pull request #1538 from jeffdaily/moat-port"). `jargon.py -C projects/GPUMD/src --commits d6cd2e36..3b0f1908 --diff d6cd2e36...3b0f1908`: clean.
- Docs: `doc/installation.rst` documents `make -f makefile.hip [HIP_ARCH=...]`; this round changes no build step.
- CUDA gate: already recorded at this head ("CUDA compile check (nvcc 12.8) at 3b0f1908", mi355, PASS), not re-run.
- Telemetry: one `-m "not slow"` attempt was stopped at ~4% (harness foreground timeout) and relaunched detached; it left no stats entry.

PASS at 3b0f1908 on linux-gfx942.

## Porter 2026-11-10 (linux-gfx90a) -- `-ffp-contract=on` in makefile.hip, 3b0f1908 -> 458683da

jeff asked for this run, answering the linux-gfx90a validation failure at 3b0f1908 (batched replica not bit-identical to per-slot). Host: AMD Instinct MI250X/MI250 (gfx90a), `HIP_VISIBLE_DEVICES=2` (dispatcher-pinned), `ulimit -c 0`, TheRock pip SDK (HIP 7.16.26385). `orient.sh` completed normally and named porter; `set-state GPUMD linux-gfx90a porting` took the lock. Fresh fork clone at `projects/GPUMD/src`, `moat-port` = 3b0f1908, `protect-fork` installed, local `master` from `origin/master`. Job venv: ase 3.29.0, calorine from source at 1c94125 (reports 4.1), pytest 7.3.2.

### Change (458683da, one new commit on 3b0f1908)
`src/makefile.hip`: `CFLAGS = -O3 --offload-arch=$(HIP_ARCH) -DUSE_HIP -ffp-contract=on`, with a three-line comment in the style of the existing `-std` comment above it. `gpumd_replica` receives `CFLAGS` through `$(MAKE) -C main_replica ... CFLAGS="$(CFLAGS)"`, and `main_replica/makefile` has no HIP flags of its own (its CFLAGS are nvcc-only), so all four targets get the flag: the build log shows it on all 230 compiles, `force/nep.cu` and `replica_batch.cu` included. CUDA: `makefile.hip` is read only by the HIP build; `src/makefile`, CMake and every source file are unchanged, so the nvcc build is byte-for-byte the 3b0f1908 one (mi355 nvcc PASS at 3b0f1908 stands). Docs: `doc/installation.rst` shows only `make -f makefile.hip` and `HIP_ARCH=...`, never CFLAGS, so a reader following it keeps the flag; no doc change. The one place a user overrides CFLAGS is the draft PR body's DEBUG command, now updated to include `-ffp-contract=on`.

### Builds (gfx90a, concurrently)
    utils/timeit.sh GPUMD compile -- make -C projects/GPUMD/src/src -f makefile.hip HIP_ARCH=gfx90a -j48
    utils/timeit.sh GPUMD compile -- make -C agent_space/dbg/src -f makefile.hip HIP_ARCH=gfx90a -j48 gpumd nep CFLAGS="-O3 --offload-arch=gfx90a -DUSE_HIP -ffp-contract=on -DDEBUG"

(`agent_space/dbg`: detached worktree at 3b0f1908 with the same diff applied.) Both rc 0, 63 s, 0 errors, warnings the known 88 `-Wunused-value` + 12 `-Winconsistent-missing-override`. Release: gpumd 15637464 B, nep 3926888 B, tb 320472 B, gpumd_replica 2645824 B. DEBUG: gpumd 15637304 B, nep 3926832 B.

### Tests (real gfx90a)
| command | 458683da | 3b0f1908 on gfx90a (validator) |
|---|---|---|
| `GPUMD_REPLICA=<clone>/src/gpumd_replica python3 agent_space/rep/tests_regression/replica/test_replica.py` (worktree with `examples/nep_train/nep.txt` from cd7f757d and `examples/qnep_train/nep.txt` from eda8e04b, as the validator did) | **16/16 OK** (49 s) | 13/16 (the 3 batched tests) |
| same, `-k test_batched_matches_per_slot -k test_split_resume_is_exact`, 3 runs | **3/3 OK each run** (15 s each) | failed 4 of 4 runs |
| `python3 -m pytest tests_pytest/test_spin_*.py` (release) | **110 passed, 4 failed** (66 s): the 4 `test_corrector_preserves_force_modifiers`; **4 passed** under the `srand` no-op `LD_PRELOAD` shim | 110/4, 4/4 |
| full `tests_regression` (DEBUG, self-comparison, PR-body command from the worktree root) | **248/256, relations 21 passed + 4 skipped** (807 s; the 4 skipped are the `observer_*` relations, skipped because member `observer_average_then_observe_control` failed) | 250/256, 25/25 |

The 8 regression failures: the 5 rand() cases (`replicate_velocity_unseeded`, `graphene_hac`, `graphene_nemd_shc`, `graphene_hnemd_shc`, `silicon_liquid_transport`: fail again plain, pass under the shim), plus `observer_average_then_observe_control` (second.xyz text), `predict_nep_pbte_batch` (force_train field 32186, -0.0105681 vs -0.0105679, abs 2.0e-7 vs atol 2e-7) and `train_charge_vdw_mixed` (loss.out `rmse_force_train` 0.62040 vs 0.62041). Rerun of the 8 plus `observer_average_two_runs`, plain: 3 passed, 6 failed (the 5 rand() + `observer_average_two_runs`); under the shim: 8 passed, 1 failed (`train_charge_vdw_mixed`). Every failure is a self-comparison of one binary against itself, so it is run-to-run nondeterminism, not a deterministic difference.

A/B to see whether the flag makes the noise more frequent: a second DEBUG build of 3b0f1908 without the flag (`agent_space/dbgdef`), then 10 alternating runs of `--case observer_average_two_runs --case observer_average_then_observe_control --case train_charge_vdw_mixed --case predict_nep_pbte_batch` on each build:

| case (fails out of 10) | with `-ffp-contract=on` | default (`fast`) |
|---|---|---|
| `observer_average_two_runs` / `_then_observe_control` (either) | 4 | 0 |
| `train_charge_vdw_mixed` | 4 | 2 |
| `predict_nep_pbte_batch` | 0 | 0 |

The observer diff (kept with `--keep-all`) is one field of one atom in second.xyz, 2.6824736640780875 vs 2.6824736640780866 (8.9e-16, a few ulp), from two runs of the same binary; the gfx942 porter recorded the same case failing by 1.1e-16 with the default flags at 51018103 (round of 2026-11-05), so the nondeterminism exists without the flag, and on gfx90a the flag only changes how often it reaches the 17th printed digit. `train_charge_vdw_mixed` already failed intermittently in the validator's `-ffp-contract=on` experiment and fails 2/10 here without the flag. None is a deterministic regression. For the reviewer and validators: these two cases are the ones to watch at 458683da; if a person wants the observer case quieter, that is upstream test tolerance (exact-text compare of double-precision output under float atomics), not something the flag should be bent around.

### Gates
- Commit 458683da `[ROCm] Contract FMAs within expressions only in makefile.hip` (61 chars), body: why, "Authored with Claude.", Test Plan with literal commands; ASCII, no Co-Authored-By. `prose.py` on the message: clean.
- `python3 utils/jargon.py --port GPUMD`: only the 2 known hits in upstream's own 725fe762. `jargon.py -C projects/GPUMD/src --commits 3b0f1908..moat-port --diff 3b0f1908...moat-port`: clean.
- Integrity: `git status --porcelain` in the clone shows no modified tracked file (build output only).
- Pushed 3b0f1908..458683da to `moat-port` (fast-forward, pre-push hook passed; `ls-remote` = 458683da). `advance-head GPUMD 458683da...`.
- Draft follow-up PR body (the one superseding the earlier draft) updated: the HIP-build paragraph names the flag and why; the DEBUG command carries `-ffp-contract=on`; the replica test command and its 16/16 (MI300X at 3b0f1908, MI250X at 458683da) are added; the reminder says every gate is owed at 458683da and the MI300X counts must be refreshed there. `prose.py` and `jargon.py -` on the body: clean.

Next: review of 458683da (one build-file line), then revalidation of every gate at 458683da, including the replica batch tests on gfx942/gfx1100/Windows with the flag.

## Review 2026-11-10 (linux-gfx1100, reviewer) -- delta 3b0f1908..458683da (`-ffp-contract=on` in makefile.hip), CHANGES REQUESTED (wording only; keep the flag)

jeff asked for this run. Host: RX 7900 XTX (gfx1100, wave32), TheRock pip SDK (HIP 7.16.26385, AMD clang 24.0.0git), `HIP_VISIBLE_DEVICES=0` (dispatcher-pinned), all GPU work under `ulimit -c 0`. `orient.sh` completed normally (no hang) and named reviewer; `set-state GPUMD linux-gfx1100 reviewing --agent reviewer` took the lock. Fresh fork clone, `moat-port` = 458683da. Detached scratch worktrees of 458683da: `agent_space/rep` (legacy-model replica run), `agent_space/dbgon` (DEBUG, with the flag), `agent_space/dbgdef` (DEBUG, flag removed via a CFLAGS override; gpumd/nep sizes 15510800/3844456 B, identical to the 3b0f1908 DEBUG build on this host, so it is the 3b0f1908 build), `agent_space/bis` (per-TU bisection). No fork edit, no push. calorine was not needed (no pytest run this review).

Scope: `git diff 3b0f1908 458683da` (only `src/makefile.hip`, +4/-1), the 458683da message, the porter's entry above and the draft follow-up PR body. CUDA: `src/makefile`, `main_replica/makefile` and all sources untouched; the mi355 nvcc PASS at 3b0f1908 stands.

### Builds
    utils/timeit.sh GPUMD compile -- make -C projects/GPUMD/src/src -f makefile.hip HIP_ARCH=gfx1100 -j12
    utils/timeit.sh GPUMD compile -- make -C agent_space/dbgon/src -f makefile.hip HIP_ARCH=gfx1100 -j10 gpumd nep CFLAGS="-O3 --offload-arch=gfx1100 -DUSE_HIP -ffp-contract=on -DDEBUG"
    utils/timeit.sh GPUMD compile -- make -C agent_space/dbgdef/src -f makefile.hip HIP_ARCH=gfx1100 -j10 gpumd nep CFLAGS="-O3 --offload-arch=gfx1100 -DUSE_HIP -DDEBUG"

All rc 0, 0 errors. The release log has `-ffp-contract=on` on all 230 compile lines (gpumd, nep, tb, and the `main_replica` objects via `CFLAGS="$(CFLAGS)"`); only the 4 link lines lack it. Release sizes: gpumd 15467192, nep 3836632, tb 309896, gpumd_replica 2473688 B.

### Replica (release, legacy models from cd7f757d / eda8e04b in `agent_space/rep`, reverted after)
- `GPUMD_REPLICA=<clone>/src/gpumd_replica python3 tests_regression/replica/test_replica.py`: **16/16 OK** (28.6 s).
- `-k test_batched_matches_per_slot -k test_split_resume_is_exact`, 3 runs: **3/3 OK each** (8.2-8.4 s).

### A/B (DEBUG, self-comparison, alternating dbgon/dbgdef)
`python3 $D/tests_regression/run_regression.py --repo-root $D --suite full --baseline $D/src/gpumd --candidate $D/src/gpumd --baseline-nep $D/src/nep --candidate-nep $D/src/nep --case observer_average_two_runs --case observer_average_then_observe_control --case train_charge_vdw_mixed --case predict_nep_pbte_batch` (`agent_space/scripts/ab.sh`), 10 rounds, then the two observer cases alone for 20 more rounds (`scripts/obs.sh`):

| case (runs failing) | `-ffp-contract=on` | default (`fast`) |
|---|---|---|
| observer_* (either case), 10 rounds | 7/10 | 0/10 |
| observer_* (either case), 20 more rounds | 14/20 | 0/20 |
| train_charge_vdw_mixed | 4/10 | 0/10 |
| predict_nep_pbte_batch | 0/10 | 0/10 |

So on gfx1100 too the flag makes these self-comparisons fail far more often (gfx90a porter: 4/10 vs 0/10 and 4/10 vs 2/10).

### Where the observer nondeterminism comes from (an upstream race the flag exposes, not a regression)
- The diff is always the same single number: second.xyz, atom line 136, field 11 = per-atom virial xx, 2.6824736640780875 vs 2.6824736640780866 (2 double ulp). Positions, velocities, forces, energies and every other file are identical. These are the exact two values the gfx90a porter recorded, so it is a two-way ordering choice, not codegen noise.
- The case's box is 10 A with NEP radial cutoff 4.2, so `get_expanded_box` (`src/force/nep/nep.cu:1049`, thickness <= 2.5*(rc+1)) takes the small-box path. There `find_force_radial_small_box` and `find_force_angular_small_box` scatter per-atom virial into neighbor `n2` with double `atomicAdd` from many threads (`src/force/nep/nep_small_box.cuh:435-443`, `555-563`; forces likewise at `425-430`). The summation order of those double atomics is unordered on any GPU, CUDA included.
- Why it is only sometimes visible: the addends are floats converted to double. A double sum of floats is exact (order-independent) unless the addends' exponents span too widely. The flag changes the float addends' low bits (different FMA formation), which for this atom makes the double sum inexact, so the order shows in the 17th digit. With default flags on gfx942 the same case already failed this way (`observer_average_then_observe_control`, one ulp, porter runs at 51018103 on 2026-11-05, entries above), so the race exists without the flag.
- Per-TU bisection (`agent_space/scripts/tu.sh`: objects of the default build, one TU recompiled, gpumd relinked): `force/nep/nep.cu` alone with `-ffp-contract=on`: observer fails **7/10**; the same relink with `nep.cu` recompiled without the flag: **0/10**. That is the TU that includes `nep_small_box.cuh`, which matches the race above.
- `train_charge_vdw_mixed` is the same class: the `nep` trainer accumulates with float/double atomics (`main_nep/nep_charge_vdw/nep_charge_vdw.cu` has 48 `atomicAdd`), and it already failed 2/10 without the flag on gfx90a.

### Recommendation: keep `-ffp-contract=on` in `makefile.hip`'s global CFLAGS
A narrower alternative (the flag only for `main_replica`, leaving gpumd/nep on `fast`) would make the observer case pass on gfx1100/gfx90a by luck of rounding while the upstream race stays, and gfx942 shows it fails without the flag anyway. Global `on` gives every target the same contraction rule, makes upstream's bitwise replica tests pass on gfx90a/gfx942/gfx1100, and causes no deterministic regression. The extra observer/training failures should be documented as run-to-run noise from upstream's atomics; making them quiet is an upstream test or kernel question (ordered virial accumulation or a tolerance on second.xyz), not something to bend the flag around.

### Problems (must fix)
1. **The claim that `-ffp-contract=on` is what nvcc does is unverified and probably wrong.** It appears in `src/makefile.hip:20-21` ("as nvcc does by default"), the 458683da message ("which is what nvcc does by default (--fmad=true)") and the draft PR body ("`on` fuses only within an expression, as nvcc does by default"); it comes from `.claude/skills/cuda-to-rocm/references/fault-classes.md:399-403`. Clang's own CUDA default is cross-statement contraction (set to match nvcc): `clang -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=sm_80 -O0 -S -emit-llvm` on `float d = a[1]*a[2]; a[0] = d + a[3];` gives `fmul contract` and `fadd contract` (two statements), the same as `-x hip`. nvcc's `--fmad=true` emits unrounded PTX `mul.f32`/`add.f32`, which ptxas may fuse with no statement scope. Either verify on the mi355 host (`nvcc -arch=sm_90 -cubin` that two-statement kernel, `cuobjdump -sass`: an FFMA means nvcc contracts across statements) or, simpler, justify `on` by what it does: it contracts only within one expression, so the per-slot and batched instantiations form the same FMAs. Fix the makefile comment and the PR body now; 458683da's message cannot be amended (no force-push), so the new commit's message states the corrected reason and the PR-prep squash message carries it. If the nvcc check confirms cross-statement FFMA, correct the fault-classes lesson on this branch too.
2. **A CFLAGS override silently drops the flag.** `src/makefile.hip` note 2 says "Do not remove -DUSE_HIP"; make it "Do not remove -DUSE_HIP or -ffp-contract=on". Anyone building with `CFLAGS=...` on the command line (the way `tests_regression/README.md:89` and `developer/spin_validation.md:71` ask for `-DDEBUG`) replaces the whole line, and `gpumd_replica` receives that CFLAGS, so a DEBUG replica build loses the flag and the batched tests fail again on gfx90a. Editing the line in place (the header's own `-DUSE_PLUMED` instruction, `doc/installation.rst`) keeps it.
3. **Draft PR body: the failure list must cover the noise the flag makes frequent.** Validators at 458683da will see `observer_average_two_runs`/`observer_average_then_observe_control` (and the 4 `observer_*` relations they skip) and `train_charge_vdw_mixed` fail often. Name them in the float-atomic class with the cause in one sentence: per-atom virial in the small-box NEP path is accumulated with double `atomicAdd`, so the last digit of one atom's virial in second.xyz varies between runs; the training loss likewise. Keep the reminder to refresh counts from the gfx942 run at 458683da.

### Checks that passed
`python3 utils/jargon.py -` on the 458683da message and on the draft body: clean; `utils/prose.py` on both: clean; `jargon.py -C projects/GPUMD/src --commits 3b0f1908..458683da --diff 3b0f1908...458683da`: clean. Title `[ROCm] Contract FMAs within expressions only in makefile.hip` (61 chars), "Authored with Claude.", Test Plan with fenced commands, no Co-Authored-By, ASCII. Integrity: `git -C projects/GPUMD/src status --porcelain` shows no modified tracked file.

## Porter 2026-11-10 (linux-gfx950, mi355) -- review items on 458683da, 458683da -> 5f6596ec

jeff asked for this run. Host: MI355 (gfx950) with CUDA 12.8 (V12.8.93) at `/usr/local/cuda-12.8`, TheRock pip SDK (AMD clang 24.0.0git), `HIP_VISIBLE_DEVICES=0` (dispatcher-pinned), `ulimit -c 0`. `orient.sh` (on `port/GPUMD`) completed normally and named porter; `set-state GPUMD linux-gfx950 porting` took the lock. Fresh fork clone at `projects/GPUMD/src`, `moat-port` = 458683da, `protect-fork` installed, local `master` from `origin/master`.

### 1. What nvcc `--fmad=true` does across statements (verified)
Kernels in `agent_space/fma`: two statements `float d = a[1] * a[2]; a[0] = d + a[3];` and the one-expression control `a[0] = a[1] * a[2] + a[3];`. `nvcc -arch=sm_90 -cubin` and `-ptx`, `--fmad=true` (default) and `--fmad=false`. CUDA 12.8 ships no `nvdisasm`, so `cuobjdump -sass` fails on its own; it worked with `NVDISASM_PATH` pointing at the nvdisasm 13.1 bundled in the image's triton package (read only).

| | two statements | one expression |
|---|---|---|
| nvcc `--fmad=true`, PTX | `fma.rn.f32` | `fma.rn.f32` |
| nvcc `--fmad=true`, SASS | `FFMA` | `FFMA` |
| nvcc `--fmad=false`, PTX / SASS | `mul.rn`+`add.rn` / `FMUL`+`FADD` | same |
| clang `-x hip --offload-arch=gfx950 -O3 -ffp-contract=on`, IR / ISA | `fmul`+`fadd` (no contract) / `v_mul_f32`+`v_add_f32` | `llvm.fmuladd` / `v_fmac_f32` |
| clang `-ffp-contract=fast`, IR / ISA | `fmul contract`+`fadd contract` / `v_fmac_f32` | same |

So nvcc fuses across statements (NVVM already emits the `fma` in PTX); `-ffp-contract=on` is stricter than nvcc, not a match. The reviewer was right. `on` stays for what it guarantees: it fuses only within one source expression, so the per-slot and batched instantiations of the replica NEP kernels form the same FMAs, which upstream's bitwise batched-vs-per-slot tests need on gfx90a.

### Changes
- Fork 5f6596ec `[ROCm] Correct the -ffp-contract=on comment in makefile.hip` (60 chars), comments only in `src/makefile.hip`: the flag comment now gives the instantiation reason with no nvcc claim; note 2 reads "Do not remove -DUSE_HIP or -ffp-contract=on", because a `CFLAGS=...` override replaces the whole line and is passed to gpumd_replica. Body states the correction to 458683da's message, "Authored with Claude.", Test Plan with fenced commands; ASCII, no Co-Authored-By, `prose.py` clean.
- Docs (item 2): no change. `doc/installation.rst:74,78` and `developer/README.md:33` show only `make -f makefile.hip [HIP_ARCH=...]`; `tests_regression/README.md:89` and `developer/spin_validation.md:71` say in prose to build with `-DDEBUG` and give no `CFLAGS=` command, so no documented command drops the flag. The makefile note covers anyone overriding CFLAGS.
- Skill (port/GPUMD, separate commit 3aff0376): `references/fault-classes.md` "clang for HIP defaults to `-ffp-contract=fast`" now says nvcc also contracts across statements (with the PTX/SASS evidence), that `on` is stricter, and when to choose it (same FMAs across instantiations; GPUMD).
- Draft follow-up PR body (item 3): the HIP-build sentence now ends "`on` fuses only within one source expression, so both instantiations form the same multiply-adds"; the float-atomic class names `observer_average_two_runs`/`observer_average_then_observe_control` (the four `observer_*` relations are skipped when either fails) and `train_charge_vdw_mixed`, with the double-`atomicAdd` per-atom virial cause; the DEBUG command already carries `-ffp-contract=on` (unchanged). The reminder names 5f6596ec and asks that the squash message carry the corrected reason. `prose.py` and `jargon.py -` on the body: clean.

### Build and replica (gfx950, first runtime evidence on this arch)
    utils/timeit.sh GPUMD compile -- make -C projects/GPUMD/src/src -f makefile.hip HIP_ARCH=gfx950 -j128

rc 0, 0 errors, `-ffp-contract=on` on all 230 compile lines. gpumd 14715984, nep 3251088, tb 314616, gpumd_replica 2494272 B.

Replica, from a detached worktree of 5f6596ec with `examples/nep_train/nep.txt` from cd7f757d and `examples/qnep_train/nep.txt` from eda8e04b (reverted, worktree removed): `GPUMD_REPLICA=<clone>/src/gpumd_replica python3 tests_regression/replica/test_replica.py`.
- First full run: **FAILED (11 failures, 6 errors), 1017 s**: 6 `TimeoutExpired` (120 s per run), 7 runs SIGSEGV (-11), 4 rc 1. The crashing runs print `GPU HANG ANALYSIS` then `HSA_STATUS_ERROR_OUT_OF_RESOURCES`, then the sticky `701 too many resources requested for launch` at `main_replica/minimize/minimizer.cu:147` (`GPU_CHECK_KERNEL` after `gpu_calculate_force_square_max<<<1, 1024>>>`). During this run the pinned GPU 0 was shared with another container's process at 100% GFX and 150 GB VRAM (`amd-smi process -g 0`), and an earlier attempt also hit a 20 min limit at a similar pace.
- `-k test_batched_matches_per_slot -k test_split_resume_is_exact` right after: **3/3 OK** (13.8 s).
- Full run again: **16/16 OK** (49.9 s), with the other process still on GPU 0.
- Not a deterministic failure of the port, but the hang is unexplained: the gfx950 validator should run the replica suite more than once and, if it hangs again, on an unshared GPU before treating it as GPU contention.

### Gates
- `python3 utils/jargon.py --port GPUMD`: only the 2 known hits in upstream's 725fe762. `jargon.py -C projects/GPUMD/src --commits 458683da..moat-port --diff 458683da...moat-port`: clean.
- Integrity: `git -C projects/GPUMD/src status --porcelain` shows no modified tracked file.
- Pushed 458683da..5f6596ec to `moat-port` (fast-forward, pre-push hook passed; `ls-remote` = 5f6596ec). `advance-head GPUMD 5f6596ec...`.

Next: review of 5f6596ec (comments only) and skill 3aff0376, then revalidation of every gate at 5f6596ec. CUDA: `makefile.hip` is not read by nvcc, so the nvcc PASS at 3b0f1908 stands.

## Review 2026-11-10 (linux-gfx1100, reviewer) -- delta 458683da..5f6596ec (comments only) and skill 3aff0376, REVIEW PASSED

jeff asked for this run. Host: RX 7900 XTX (gfx1100), `HIP_VISIBLE_DEVICES=0` (dispatcher-pinned), `ulimit -c 0`. `orient.sh` completed normally and named reviewer; `set-state GPUMD linux-gfx1100 reviewing --agent reviewer` took the lock. Fresh fork clone, `moat-port` = 5f6596ec. No test run (comments-only round).

Items from the previous review:
1. nvcc claim: gone from `src/makefile.hip:22-25`; the comment now gives the instantiation reason (fuses only within one source expression, so every instantiation forms the same FMAs). 5f6596ec's message states the correction to 458683da. The draft PR body says the same and the reminder asks the squash message to carry it. Skill 3aff0376 (`references/fault-classes.md`, `-ffp-contract` entry) matches the porter's nvcc 12.8 PTX/SASS table: `--fmad=true` gives `fma.rn.f32`/FFMA for the two-statement form, so `on` is stricter than nvcc; `on` is justified by same-FMAs-across-instantiations, with GPUMD cited.
2. Note 2 at `src/makefile.hip:6-8` now reads "Do not remove -DUSE_HIP or -ffp-contract=on" and explains that a CFLAGS override replaces the line and reaches gpumd_replica. Done.
3. Draft PR body: the float-atomic class now names `observer_average_two_runs`/`observer_average_then_observe_control` (with the skipped `observer_*` relations) and `train_charge_vdw_mixed`, with the small-box double-`atomicAdd` per-atom virial cause. `jargon.py -` and `utils/prose.py` on the body (lines up to the internal reminder): clean. The reminder paragraph itself contains "revalidation"; it is not upstream text and must stay out of the posted body.

Build identity: `git diff 458683da 5f6596ec` changes only `#` lines in `src/makefile.hip` (no non-comment +/- line), and `make -n -f makefile.hip HIP_ARCH=gfx1100 all` in 5f6596ec and a 458683da worktree gives the same 248 command lines (`cmp` identical), so the build is the 458683da build.

Gates: `jargon.py -C projects/GPUMD/src --commits 458683da..5f6596ec --diff 458683da...5f6596ec` clean; `jargon.py -` on the 5f6596ec message clean; title 59 chars, `[ROCm]`, "Authored with Claude.", no Co-Authored-By. Integrity: clone `status --porcelain` empty.

Nit (no action needed before squash): 5f6596ec's Test Plan runs `cuobjdump -sass`, which needs an `nvdisasm` that CUDA 12.8 on the mi355 host lacks (the porter used `NVDISASM_PATH`); the PR-prep squash message replaces this anyway.

Verdict: review passed. Next: revalidation of every gate at 5f6596ec (gfx950 replica suite more than once, per the porter's hang note).

## Validation 2026-11-10 (windows-gfx1151, validator) -- revalidate at 5f6596ec, VALIDATION FAILED (batched PRD not bit-identical to per-slot on gfx1151)

jeff asked for this run. GPU: AMD Radeon 8060S (gfx1151, wave32 APU), `HIP_VISIBLE_DEVICES=0`, TheRock ROCm SDK, Windows 11. Fresh clone `agent_space/gpumd-win4` at **5f6596ec7b656eac5f0c96df882f8bdda36565df** (= `ls-remote moat-port`), clean tracked tree (only staged runtime DLLs untracked). Nothing pushed. Serial, one GPU process at a time, `ulimit -c 0`.

### Build
    utils/timeit.sh GPUMD compile -- bash agent_space/gp4_build.sh
      # cd src; mingw32-make -f makefile.hip gpumd nep gpumd_replica HIP_ARCH=gfx1151 LDFLAGS="-L<rocm>/lib" CC="<rocm>/bin/hipcc.exe" CFLAGS="-O3 --offload-arch=gfx1151 -DUSE_HIP -ffp-contract=on -std=c++17" -j8 -k
rc 0, 0 errors, `-ffp-contract=on` on all 224 compile lines. gpumd 15762432 B, nep 4227584 B, gpumd_replica.exe 2703360 B. Linker again wrote gpumd/nep without .exe (copied). Runtime DLLs (amd_comgr, amdhip64_7, hiprtc*) copied next to the exe; the first harness run without them exited 0xC0000005 (environment, fixed by staging).

### Tests (real gfx1151)
Regression harness, serial, self-comparison, `--timeout-scale 3`, cases ilp_nep_cutoff_below_nep neighbor_alias_aliased neighbor_alias_control nve nvt_ber nvt_nhc npt_ber_iso nvt_mttk: **8 passed, 0 failed**, relation neighbor_alias_translation_invariance PASS (24 skipped, not selected).

Replica, scratch worktree `agent_space/gp4_rep` at 5f6596ec with `examples/nep_train/nep.txt` from cd7f757d and `examples/qnep_train/nep.txt` from eda8e04b, `GPUMD_REPLICA=<clone>/src/gpumd_replica.exe`, `-k test_batched_matches_per_slot -k test_split_resume_is_exact` (4x2000 atoms PbTe, 40 steps, about 11 s):
- test_batched_matches_per_slot_remd ok; test_split_resume_is_exact ok (both batch=0 and 1)
- **test_batched_matches_per_slot_prd FAIL**, twice (run twice, stopped): `prd_replica_2_restart.xyz differs between prd_batch0 and prd_batch1`. The differing values are not ulp-level and differ between the two runs: first run replica 2 coordinate 3.2690353 (batch 0) vs 3.2686072 (batch 1), diff ~4e-4; second run 3.2068339 vs 3.2058530, diff ~1e-3. The first file the test compares to differ is the replica_2 restart; the others were not inspected (assert stops at the first).
Read: this is the same class as the linux-gfx90a failure at 3b0f1908 (batched replica not bit-identical to per-slot), which `-ffp-contract=on` cured there (3/3 OK at 458683da on MI250X, 16/16). On gfx1151 the flag is present and REMD plus split-resume pass, but PRD (3000 K quench, event 5, dephase 10) diverges, with differences that grow to 1e-3 which suggests an amplified near-ulp nondeterminism after a PRD event rather than a gross fault. No GPU fault, hang or 719. Not root-caused (known hard class: one arch wrong numbers while others pass; stopped per discipline). Pass counts at this arch: 2 of 3 requested replica tests.

### Gates
`jargon.py -C gpumd-win4 --commits 3b0f1908..5f6596ec --diff 3b0f1908...5f6596ec`: clean. Docs: installation.rst ROCm section unchanged and still accurate. CUDA gate recorded at 3b0f1908; the delta touches only makefile.hip. Integrity: clone has no modified tracked file.

State: windows-gfx1151 -> `validation-failed` at 5f6596ec. For the porter: decide whether PRD batched bit-identity is expected on wave32 APUs (other gates are satisfied by Linux archs at this head or pending; this record is extra evidence on the windows gate only), or whether a gfx1151 determinism difference in the batched PRD path (replica_batch.cu) needs a look.
## Validation 2026-11-10 (linux-gfx1100, validator) -- revalidate at 5f6596ec, PASS

jeff asked for this run. Host: AMD Radeon RX 7900 XTX (gfx1100, RDNA3, wave32), `HIP_VISIBLE_DEVICES=0` (dispatcher-pinned; `amd-smi process` showed no other process on any GPU), all GPU runs under `ulimit -c 0`. Toolchain: TheRock pip SDK, HIP 7.16.26385, AMD clang 24.0.0git. `orient.sh` on `port/GPUMD` completed normally and named validator (`revalidate`). Fresh clone of AMD-Ecosystem/GPUMD detached at **5f6596ec7b656eac5f0c96df882f8bdda36565df** (= `origin/moat-port`); detached worktrees of the same commit for the DEBUG build (`agent_space/dbg`) and the legacy-model replica run (`agent_space/rep`, model files reverted, both worktrees removed after). No fork edit, no push. Job venv: calorine built from source at gitlab materials-modeling/calorine **1c9412504bf1** (the porter's commit; reports 4.1) into this job's venv only; ase 3.29.0, pytest 7.3.2, hypothesis 6.56.4. Carry-forward does not apply (458683da changes the compile flags), so this was a full run.

### Builds (PR-body commands, gfx1100, from the MOAT root)
    utils/timeit.sh GPUMD compile -- make -C projects/GPUMD/src/src -f makefile.hip HIP_ARCH=gfx1100 -j16
    utils/timeit.sh GPUMD compile -- make -C agent_space/dbg/src -f makefile.hip HIP_ARCH=gfx1100 -j30 gpumd nep CFLAGS="-O3 --offload-arch=gfx1100 -DUSE_HIP -ffp-contract=on -DDEBUG"

Both rc 0, 0 errors (45 s / 35 s); warnings only the known classes (88/84 `-Wunused-value` nodiscard `hipError_t`, 12 `-Winconsistent-missing-override`). `-ffp-contract=on` on all 230 compile lines of the release build. Release: gpumd 15467192 B, nep 3836632 B, tb 309896 B, gpumd_replica 2473688 B, identical to the reviewer's 458683da gfx1100 build (as expected: 5f6596ec is comments only). DEBUG: gpumd 15467024 B, nep 3832488 B.

### Tests (real gfx1100)
| command | 5f6596ec | 3b0f1908 on gfx1100 | 374da0df on gfx1100 |
|---|---|---|---|
| `GPUMD_REPLICA=<clone>/src/gpumd_replica python3 tests_regression/replica/test_replica.py` in `agent_space/rep` with `examples/nep_train/nep.txt` from cd7f757d and `examples/qnep_train/nep.txt` from eda8e04b | **16/16 OK**, 28.6 s | 16/16 | 12/12 |
| same, `-k test_batched_matches_per_slot -k test_split_resume_is_exact`, 3 runs | **3/3 OK each run** (8.2 s each); no hang, no crash | -- | -- |
| replica as-is (current example models, clone) | 4/16; all 12 failures stop at `REMD/PRD explicit streams support energy NEP4/NEP5 and qNEP4 models` | 4/16 | 6/12 |
| `python -m pytest tests_pytest/test_spin_*.py` (release) | **110 passed, 4 failed**, 53 s; the 4 `test_corrector_preserves_force_modifiers` (sib/glsd x small/large box) | 110/4 | 110/4 |
| the 4 under an `LD_PRELOAD` srand no-op shim (`void srand(unsigned){}`) | **4 passed** | 4/4 | 4/4 |
| full `tests_regression` (DEBUG, self-comparison): `python3 $D/tests_regression/run_regression.py --repo-root $D --suite full --baseline $D/src/gpumd --candidate $D/src/gpumd --baseline-nep $D/src/nep --candidate-nep $D/src/nep --timeout-scale 3` | **completed, all 256 cases: 251 passed, 5 failed; relations 25/25**, 423 s (FP64 transport cases included; overlapped the not-slow pytest from about minute 2). `ilp_nep_cutoff_below_nep` PASS; `observer_average_*` and `train_charge_vdw_mixed` passed in this run | 248/256, 25/25 | 249/256, 25/25 |
| the 5 failures rerun with `--case` x5, plain / under the srand shim | 0/5 / **5/5** | -- | -- |
| `python -m pytest tests_pytest -m "not slow"` (release, whole directory) | **866 passed, 4 failed, 638 skipped, 10 deselected**, 1021 s; the 4 are the same rand() cases, nothing else | 866/4/638/10 | 801/4/480 |

Noise-frequency check of the cases the flag makes noisier (reviewer's 458683da finding), DEBUG, 10 rounds of `--case observer_average_two_runs --case observer_average_then_observe_control --case train_charge_vdw_mixed`: some case failed in **10/10 rounds**: `observer_average_two_runs` 6/10, `observer_average_then_observe_control` 3/10, either observer case 8/10, `train_charge_vdw_mixed` 5/10 (reviewer on gfx1100 at 458683da: observer either 7/10 and 14/20, train 4/10). A failing run kept with `--keep-all` shows exactly the recorded signatures: second.xyz atom field 11 (per-atom virial) 2.6824736640780866 vs 2.6824736640780875, and loss.out field 5 `0.62040` vs `0.62041`. Same two-valued ordering choice as on gfx90a and in the review, so it is the known float-atomic class (small-box NEP double `atomicAdd` of per-atom virial; trainer atomics), and the draft PR body already names these cases. They passed in the full-suite run.

Every failure classified, none new:
- **rand() reseed** (deferral `gpumd-hip-runtime-reseeds-rand`, ruled defer): the 4 spin pytest cases (both pytest runs) and `replicate_velocity_unseeded`, `graphene_hac`, `graphene_nemd_shc`, `graphene_hnemd_shc`, `silicon_liquid_transport` (text diffs); all pass under the shim, the 5 fail again plain.
- **float-atomic noise**: no case in the full run; `observer_average_two_runs`, `observer_average_then_observe_control`, `train_charge_vdw_mixed` in the targeted repeat runs above. The qNEP cases that hit their bounds at 3b0f1908 on this host (`qnep_ewald_future_bec`, `qnep_pppm_future_bec`, `qnep_actions_before_potential`) passed this time.
- **replica model header** (upstream `format_version 1`, platform-independent): the 12 as-is replica failures; 16/16 with the legacy models.

### Gates
- CUDA compile: the mi355 nvcc 12.8 PASS at 3b0f1908 stands; 458683da and 5f6596ec change only `makefile.hip`, which the CUDA build does not read. Not repeated.
- `python3 utils/jargon.py --port GPUMD` (local `master` from the fork's `origin/master`): only the 2 known hits in upstream's own 725fe762. `jargon.py -C projects/GPUMD/src --commits 3b0f1908..5f6596ec --diff 3b0f1908...5f6596ec`: clean.
- Documentation: `doc/installation.rst` documents `make -f makefile.hip` and `HIP_ARCH=gfx1100`; the makefile's note 2 says to keep `-ffp-contract=on` in any CFLAGS override.
- Integrity: `git status --porcelain` empty in the clone and both worktrees, HEAD 5f6596ec in each.

Result: **completed** at 5f6596ec on linux-gfx1100.
## Validation 2026-11-10 (linux-gfx942, validator) -- revalidate at 5f6596ec, PASSED (one new test name in the float-atomic trainer class; see the flag note)

jeff asked for this run. Host: MI300X (gfx942, CDNA3, wave64), HIP 7.16.26385 / AMD clang 24.0.0git (TheRock pip SDK), `HIP_VISIBLE_DEVICES=0` (dispatcher-pinned), all GPU work under `ulimit -c 0`. `amd-smi process -g 0` showed no other process on the GPU at the start, and no replica run hung or crashed. `orient.sh` (on `port/GPUMD`) completed normally and named validator (`state=revalidate`). Fresh clone of AMD-Ecosystem/GPUMD detached at **5f6596ec7b656eac5f0c96df882f8bdda36565df** (= `origin/moat-port`). Detached worktrees of the same commit: `agent_space/dbg` (DEBUG build), `agent_space/rep` (replica legacy-model runs; models reverted), `agent_space/def` (a no-flag `nep` build for an A/B); all were removed afterwards. No fork edit, no push. Job venv only: calorine built from a source clone of gitlab materials-modeling/calorine at **1c9412504bf1** (the porter's commit; reports 4.1), ase 3.29.0, pytest 7.3.2, hypothesis 6.56.4. A full run: the carry-forward shortcut does not apply (458683da changes every compile line).

### Builds (PR-body commands, gfx942, concurrently)
    utils/timeit.sh GPUMD compile -- make -C projects/GPUMD/src/src -f makefile.hip HIP_ARCH=gfx942 -j56
    utils/timeit.sh GPUMD compile -- make -C agent_space/dbg/src -f makefile.hip HIP_ARCH=gfx942 -j56 gpumd nep CFLAGS="-O3 --offload-arch=gfx942 -DUSE_HIP -ffp-contract=on -DDEBUG"

Both rc 0, 33 s / 32 s, 0 errors; warnings only the known classes (88/84 `-Wunused-value`, 12 `-Winconsistent-missing-override`). `-ffp-contract=on` is on all 230 release compile lines. Release: gpumd 14732376 B, nep 3251096 B, tb 314616 B, gpumd_replica 2502464 B. DEBUG: gpumd 14732216 B, nep 3251040 B.

### Tests (real gfx942; GPU not shared between runs)
| command | result | time | 3b0f1908 on gfx942 (validator) |
|---|---|---|---|
| `GPUMD_REPLICA=<clone>/src/gpumd_replica python3 tests_regression/replica/test_replica.py` (clone, as-is) | **4/16**; all 12 failures stop at `REMD/PRD explicit streams support energy NEP4/NEP5 and qNEP4 models` | 18 s | 4/16 |
| same from `agent_space/rep` with `examples/nep_train/nep.txt` from cd7f757d and `examples/qnep_train/nep.txt` from eda8e04b | **16/16 OK** | 51 s | 16/16 |
| same, `-k test_batched_matches_per_slot -k test_split_resume_is_exact`, 3 runs | **3/3 OK in each run** | 14 s each | -- |
| `python3 $D/tests_regression/run_regression.py --repo-root $D --suite full --baseline $D/src/gpumd --candidate $D/src/gpumd --baseline-nep $D/src/nep --candidate-nep $D/src/nep` (DEBUG, self-comparison; the whole suite, including the FP64 transport cases) | **248/256 cases; relations 21 passed, 4 skipped** (the `observer_*` relations, skipped because their member cases failed); `ilp_nep_cutoff_below_nep` PASS | 570 s | 250/256, 25/25 |
| the 8 failures rerun (`--case` x8), plain | 0/8 | 42 s | 0/6 |
| the 8 rerun under the `LD_PRELOAD` srand no-op shim (`void srand(unsigned){}`) | 7/8 (only `qnep_pppm_future_bec` fails); relations 4/4 | 44 s | 5/6 |
| `python3 -m pytest tests_pytest/test_spin_*.py` (release, clone root) | **110 passed, 4 failed** (the 4 `test_corrector_preserves_force_modifiers`) | 96 s | 110/4 |
| the 4 under the srand shim | 4 passed | 5 s | 4/4 |
| `python3 -m pytest tests_pytest -m "not slow"` (release) | **865 passed, 5 failed, 638 skipped, 10 deselected**: the 4 rand() cases and `test_nep_seed.py::test_same_seed_gives_the_same_model` | 1413 s | 866/4/638/10 |

Regression failures, classified:
| case | full run | plain rerun | shim rerun | class |
|---|---|---|---|---|
| replicate_velocity_unseeded, graphene_hac, graphene_nemd_shc, graphene_hnemd_shc, silicon_liquid_transport | FAIL | FAIL | PASS | rand() reseed (known; deferral `gpumd-hip-runtime-reseeds-rand`, ruled defer) |
| qnep_pppm_future_bec | bec.xyz field 13, 1.37e-5 vs atol 1e-5 | field 22735, 1.04e-5 | field 17, 1.11e-5 | float-atomic noise (known; failed in every run at 3b0f1908 on this host as well) |
| observer_average_two_runs, observer_average_then_observe_control | FAIL | FAIL | PASS | float-atomic, small-box NEP double-`atomicAdd` per-atom virial (known; the reviewer's 458683da finding). The diff is the same single value the gfx90a porter and gfx1100 reviewer recorded: second.xyz, the `C` line, virial field 2.6824736640780875 vs 2.6824736640780866. It failed in 2 of 3 runs here; with default flags at 3b0f1908 on this host it failed in 0 of 4 |

### The one failure outside the PR body's list: `test_nep_seed.py::test_same_seed_gives_the_same_model`
Upstream's test (baa4bfbe/00c1b638) trains twice with `seed 1` (BaTiO3, 2 generations) and requires identical `loss.out` and `nep.txt`. The failure here: `loss.out` differs in one value's last printed digit, `2.28621` vs `2.28622`. The same binary with the same seed gives different output, so this is run-to-run nondeterminism, not a wrong result.
- A/B on this host, 10 alternating rounds of `python3 -m pytest tests_pytest/test_nep_seed.py::test_same_seed_gives_the_same_model`, from the clone (release nep with the flag) and from `agent_space/def` (5f6596ec `nep` built with `make -f makefile.hip HIP_ARCH=gfx942 -j56 nep CFLAGS="-O3 --offload-arch=gfx942 -DUSE_HIP"`, 3447344 B, the same size as the 3b0f1908 nep): **with `-ffp-contract=on` 6/10 failed; default (`fast`) 0/10**. The test passed in both gfx942 `-m "not slow"` runs at 3b0f1908.
- Per-TU check (default objects; listed TUs recompiled with the flag; nep relinked; 10 runs each; `agent_space/scripts/tu_ab.sh`): `main_nep/common/nep_common.cu` alone 0/10; `main_nep/nep/nep.cu` alone 0/10; the 9 top-level `main_nep/*.cu` 0/10; the 2 `main_nep/nep/` TUs 0/10; **the 19 `main_nep/common/` TUs 2/10**. So most of the effect comes from the shared trainer kernels in `main_nep/common/`, which scatter forces with float `atomicAdd` (`nep_common.cu:182-187`, `nep_evaluate.cu:318-323, 420-425`). The full-flag rate (6/10) is higher than any subset's, so more than one TU contributes. I stopped the bisection there (validator scope).
- Classification: this is the float-atomic trainer class the reviewer already named for `train_charge_vdw_mixed` ("the trainer's atomic accumulation likewise moves the last printed digit of a training loss"), with the same frequency pattern (often with the flag, never without it). The test name is new to the list, and the draft PR body's `-m "not slow"` sentence ("passes 866 with the same 4 failures") no longer holds at 5f6596ec on the MI300X.
- **For jeff (not a validator decision):** this test is the only check that the `seed` keyword gives reproducible training. With `-ffp-contract=on`, seeded training on the MI300X reproduces only 4 times in 10 for this input; with hipcc's default it reproduced 10 times in 10. The reviewer's case for keeping the flag global was about the observer self-comparisons. This test checks something a user relies on. Narrowing the flag to `gpumd_replica`, which the 458683da review considered and rejected, would keep `nep` on the default. Either way, the porter's PR-body refresh should name this test in the float-atomic class (or remove it from that class, if the flag is narrowed).

Compared with the 374da0df gfx942 record (251/256, 25/25; spin 110/114; 801/4 of the smaller suite; replica 6/12 as-is, 12/12 legacy) and the 3b0f1908 gfx942 record (250/256, 25/25; spin 110/114; 866/4/638/10; replica 4/16, 16/16): the rand() set is unchanged, replica is unchanged and the batched tests pass, and `qnep_pppm_future_bec` is unchanged. The new failures are both flag-frequency effects in the float-atomic class: the observer pair (reviewer's prediction) and the seed test above.

### Gates
- CUDA compile gate: the mi355 nvcc 12.8 PASS at 3b0f1908 stands. 458683da and 5f6596ec change only `src/makefile.hip`, which the CUDA build does not read (`git diff --stat 3b0f1908 5f6596ec`: that file only, +8/-2). Not re-run.
- `python3 utils/jargon.py --port GPUMD`: only the 2 known hits in upstream's own 725fe762. `jargon.py -C projects/GPUMD/src --commits 3b0f1908..5f6596ec --diff 3b0f1908...5f6596ec`: clean.
- Documentation: `doc/installation.rst:74-78` documents `make -f makefile.hip` and `HIP_ARCH=...`; the build commands a user types are unchanged.
- Integrity: `git -C projects/GPUMD/src status --porcelain --untracked-files=no` empty, HEAD 5f6596ec; the three scratch worktrees had no modified tracked files and were removed.
- Telemetry: two `test` entries with nonzero exit and under 0.5 s are my own command mistakes (a relative path under `timeit.sh`, and a pytest on a nonexistent `test_spin_basic.py`), not test results.

Result: **completed** at 5f6596ec on linux-gfx942. Every failure is in a known class: rand() reseed, float-atomic noise (qNEP PPPM BEC, small-box virial, trainer loss), and the replica `format_version 1` model headers. The flag question above goes to jeff.
## Validation 2026-11-10 (linux-gfx90a, validator) -- revalidate at 5f6596ec, PASS

jeff asked for this run. Host: AMD Instinct MI250X/MI250 (gfx90a, CDNA2, wave64), `HIP_VISIBLE_DEVICES=2` (dispatcher-pinned), every GPU run under `ulimit -c 0`. Toolchain: TheRock pip SDK, HIP 7.16.26385, AMD clang 24.0.0git. `orient.sh` on `port/GPUMD` completed normally (no hang) and named validator (`port-ready`, returning from the 3b0f1908 failure, so a full run with no carry-forward). Fresh clone of AMD-Ecosystem/GPUMD at `projects/GPUMD/src`, detached at **5f6596ec7b656eac5f0c96df882f8bdda36565df** (= `origin/moat-port`); detached worktrees of the same commit for the DEBUG build (`agent_space/dbg`) and the legacy-model replica run (`agent_space/rep`; model edits reverted, both worktrees removed after). No fork edit, no push. Job venv only: calorine built from a source clone of gitlab materials-modeling/calorine at **1c94125** (the porter's commit; reports 4.1), ase 3.29.0, pytest 7.3.2. `amd-smi process` showed no other process on any GPU when the replica runs started; no hang or crash occurred.

### Builds (PR-body commands, gfx90a)
    utils/timeit.sh GPUMD compile -- make -C projects/GPUMD/src/src -f makefile.hip HIP_ARCH=gfx90a -j48
    utils/timeit.sh GPUMD compile -- make -C agent_space/dbg/src -f makefile.hip HIP_ARCH=gfx90a -j64 gpumd nep CFLAGS="-O3 --offload-arch=gfx90a -DUSE_HIP -ffp-contract=on -DDEBUG"

Both rc 0, 0 errors (60 s / 58 s); `-ffp-contract=on` on all 230 release compile lines; warnings only the known 88 `-Wunused-value` + 12 `-Winconsistent-missing-override`. Release: gpumd 15637464 B, nep 3926888 B, tb 320472 B, gpumd_replica 2645824 B; DEBUG gpumd 15637304 B, nep 3926832 B. Every size equals the porter's gfx90a build at 458683da, as expected for a comments-only delta. (A first DEBUG attempt exited 2 in 0 s because it started before its worktree existed; my shell error, not a build failure.)

### Tests (real gfx90a)
| command | 5f6596ec | 3b0f1908 on gfx90a | 374da0df on gfx90a |
|---|---|---|---|
| `GPUMD_REPLICA=<clone>/src/gpumd_replica python3 tests_regression/replica/test_replica.py` (release, as-is, from the clone root) | **4/16** (11.7 s); all 12 failures stop at `REMD/PRD explicit streams support energy NEP4/NEP5 and qNEP4 models` (8 REMD, 4 PRD) | 4/16 | 6/12 |
| same from `agent_space/rep` with `examples/nep_train/nep.txt` from cd7f757d and `examples/qnep_train/nep.txt` from eda8e04b, GPU idle | **16/16 OK** (49.0 s) | 13/16 (the 3 batched tests) | 12/12 |
| same, `-k test_batched_matches_per_slot -k test_split_resume_is_exact`, 3 runs | **3/3 OK each run** (15.1 s each) | failed 4 of 4 | -- |
| `python3 -m pytest tests_pytest/test_spin_*.py` (release) | **110 passed, 4 failed** (68 s); the 4 `test_corrector_preserves_force_modifiers` (sib/glsd x small/large box) | 110/4 | 110/4 |
| those 4 under an `LD_PRELOAD` shim of `void srand(unsigned){}` | **4 passed** | 4/4 | 4/4 |
| full `tests_regression` (DEBUG, self-comparison, from the worktree root): `python3 tests_regression/run_regression.py --repo-root $D --suite full --baseline $D/src/gpumd --candidate $D/src/gpumd --baseline-nep $D/src/nep --candidate-nep $D/src/nep` | **completed, 250/256 cases; relations 21 passed, 0 failed, 4 skipped** (801 s, whole suite incl. FP64 transport cases, no `--timeout-scale`; overlapped the not-slow pytest). `ilp_nep_cutoff_below_nep` PASS. The 4 skipped relations are the `observer_*` ones, skipped because `observer_average_then_observe_control` failed | 250/256, 25/25 | 249/256, 25/25 |
| rerun plain: `--case` for the 6 failures plus `observer_average_two_runs` | 1 passed, 6 failed (186 s): the 5 rand() cases and `observer_average_two_runs` (second.xyz); `observer_average_then_observe_control` passed | -- | -- |
| same 7 under the srand shim | **7 passed, 0 failed**; the 4 `observer_*` relations now run: 4 passed (155 s) | -- | -- |
| `python3 -m pytest tests_pytest -m "not slow"` (release, whole directory) | **866 passed, 4 failed, 638 skipped, 10 deselected** (2183 s, overlapped the regression run); the 4 are the same rand() cases, nothing else | 866/4/638/10 | 801/4/480 (smaller pre-merge suite) |

Failure classification, none new:
- **rand() reseed** (deferral `gpumd-hip-runtime-reseeds-rand`, ruled defer): the 4 spin pytest cases (both pytest runs) and `replicate_velocity_unseeded` (final.xyz), `graphene_hac` (hac.out), `graphene_nemd_shc` (compute.out), `graphene_hnemd_shc` (kappa.out), `silicon_liquid_transport` (adf.out); all fail again plain and pass under the shim. Same set as at 3b0f1908 and 374da0df on this host.
- **float-atomic noise** (the upstream double-`atomicAdd` per-atom virial race in the small-box NEP path that the reviewer traced, made more frequent by `-ffp-contract=on`): `observer_average_then_observe_control` in the full run, second.xyz atom line 137 (diff hunk `@@ -134,7`), field 11 (virial xx), **2.6824736640780866 vs 2.6824736640780875**, the exact two values the gfx90a porter and gfx1100 reviewer recorded; `observer_average_two_runs` (second.xyz text) on the plain rerun, where `_then_observe_control` passed instead; both pass under the shim rerun (the shim does not touch atomics, so that is just the intermittency). `train_charge_vdw_mixed` and `qnep_pppm_future_bec` passed this time.
- **replica model header** (upstream `format_version 1`, platform-independent): the 12 as-is replica failures; 16/16 with the legacy models.

Compared with 3b0f1908 on this host: the blocker that failed it (batched-vs-per-slot replica not bit-identical) is fixed by 458683da's flag: 16/16 and the three batched tests 3 of 3 runs, against 4 of 4 failing before. Regression count unchanged at 250/256; the float-noise case moved from `qnep_pppm_future_bec` to the `observer_*` pair, as the porter's and reviewer's A/B predicted. Pytest counts identical.

### Gates
- CUDA compile gate: the mi355 nvcc 12.8 PASS at 3b0f1908 stands (458683da and 5f6596ec change only `src/makefile.hip`, which the CUDA build does not read: `git diff 3b0f1908 5f6596ec --stat` = `src/makefile.hip | 10 ++++++++--`); not repeated.
- `python3 utils/jargon.py --port GPUMD` (local `master` created from the fork's `origin/master`): only the 2 known hits in upstream's own 725fe762 merge. `jargon.py -C projects/GPUMD/src --commits 3b0f1908..5f6596ec --diff 3b0f1908...5f6596ec`: clean.
- Documentation: `doc/installation.rst:74,78` documents `make -f makefile.hip` and `HIP_ARCH=...`; no CFLAGS override is documented, so the flag is kept.
- Integrity: no modified tracked file in the clone or either worktree (HEAD 5f6596ec in each) before the worktrees were removed.

Result: **completed** at 5f6596ec on linux-gfx90a. Every failure is in a known class (rand() reseed, float-atomic noise in the observer virial, replica `format_version 1` model headers upstream).

## Diagnosis 2026-11-10 (windows-gfx1151) -- batched PRD replica

jeff asked for a bounded diagnosis of the `test_batched_matches_per_slot_prd` failure recorded at 5f6596ec on windows-gfx1151. No platform state changed, no fork edit or push. Binary: the validator's `agent_space/gpumd-win4/src/gpumd_replica.exe` (5f6596ec, `-ffp-contract=on`). Inputs: the test class itself, imported from `agent_space/gp4_rep/tests_regression/replica` (legacy models from cd7f757d / eda8e04b swapped in), PRD options `replicas 3 event 5 dephase 10 correlate 5 distance 0.15 quench 0.01 3000`, 2000-atom PbTe supercell. Scratch scripts in `agent_space/gp5diag/`. Serial, `ulimit -c 0`, 9 GPU runs total, no fault or hang. Runtime PATH needs the SDK `bin` in MSYS form (`export PATH="$(cygpath -u "$ROCM_ROOT")/bin:$PATH"`); with the `D:/...` form the exe exits 0xC0000135 (hipfft.dll not found).

### Determinism probe (one 20-step batch=0 start, then four 40-step resumes from copies of it)
    python agent_space/gp5diag/batch_det.py agent_space/gp4_rep/tests_regression/replica <win path>/gpumd-win4/src/gpumd_replica.exe agent_space/gp5diag/run1
- b0a vs b0b (both `GPUMD_REPLICA_BATCH=0`): all files identical.
- b1a vs b1b (both `=1`): all files identical.
- b0 vs b1 (each pairing): only `prd_replica_2_restart.xyz` differs (plus the extra batched line in `neighbor.out`). Replicas 0 and 1, basin, events and meta are bit-identical. Replica 2: all 18000 position/velocity values differ, max position diff 2.3e-3 A.
So neither path is nondeterministic. The difference is systematic and confined to the last replica.

### Earliest divergence (1-step resume, `agent_space/gp5diag/short.py ... 1`)
After one step replica 2 already differs (position 7.0e-5 A max, velocity up to 7e-4 in natural units, per-atom velocity ratio 0.963..1.028, so not a thermostat factor). Not ulp-level and not chaos amplification. Back-solving the first velocity-Verlet drift, x1 - x0 - v0*dt = 0.5*F0/m*dt^2, from the restart and the 1-step output gives the force each run started with:
- replica 0 and 1, both paths: implied F0 = 0 (residual 3e-15).
- replica 2, batch 1: implied F0 = 0.
- replica 2, batch 0: implied acceleration rms 2.8e-5 A/fs^2, a real force.
The resumed run starts from zero forces except for per-slot replica 2.

### Cause: unordered zero fill of `force_per_atom` on the legacy default stream (upstream race, PRD only)
- `Replica_Runtime::initialize` (`main_replica/replica_common.cu:520-523`) runs `initialize_slot(*slot)` and then `compute_force(device, *slot)` on `slot.stream`, which is created `gpuStreamNonBlocking` (`main_replica/utilities/gpu_stream.cuh:154,183`).
- `PRD_Driver::initialize_slot` (`main_replica/replica_prd.cu:656`) calls `allocate_memory_gpu`, whose `atom.force_per_atom.resize(N * 3, 0)` (`main_replica/model/read_xyz.cu:554`) zero-fills through `GPU_Vector::fill`, a `gpu_fill<<<>>>` launch on the legacy default stream (`main_replica/utilities/gpu_vector.cuh:240`). PRD then uploads velocities with `copy_from_host_async` on the slot stream and syncs only that stream (`replica_prd.cu:664-668`). Nothing orders the default-stream fill before `compute_force` on the non-blocking slot stream.
- On gfx1151/Windows the fill lands after the NEP force is computed and wipes it. The next blocking `hipMemcpy` flushes it: the next slot's `allocate_memory_gpu` copies for replicas 0 and 1, and the `Replica_Batch` constructor's `copy_from_host` (`main_replica/replica_batch.cu:444-446`) for replica 2 when batching is on. With batching off nothing flushes replica 2's fill before the first step, so replica 2 keeps its real force. That is the batch0/batch1 difference.
- REMD does not race: `REMD_Driver::initialize_slot` (`replica_remd.cu:471-473`) uploads velocities with the blocking `copy_from_host`, which orders after the fill. That is why REMD batched and split-resume pass here.
- On Linux (gfx1100/gfx942/gfx90a) the fill completes before `compute_force` in practice, so all replicas keep their forces and both paths agree. CUDA has the same unordered-stream pattern; it is an upstream bug that this platform's timing exposes.
- Not caused by atomics, a wave-size assumption or FP contraction (both paths are bitwise repeatable, the error is a whole force field and not ulps). The `-ffp-contract=off` build was not tried; it is irrelevant to this cause.

### Confirmation (scratch build, not a fork edit)
Copied `gpumd-win4/src/main_replica` to `agent_space/gp5diag/fix/main_replica`, added `CHECK(gpuDeviceSynchronize());` right after `allocate_memory_gpu(...)` in `PRD_Driver::initialize_slot`, and rebuilt (`mingw32-make CC=<rocm>/bin/hipcc.exe CFLAGS="-O3 --offload-arch=gfx1151 -DUSE_HIP -ffp-contract=on -std=c++17" LIBS=-lhipfft LDFLAGS=-L<rocm>/lib`, only replica_prd.obj recompiled). From the same 20-step start, 40-step resumes at batch 0 and batch 1: **all PRD outputs identical** (three replica restarts, basin, events, meta). The fixed run's replica 2 equals the original per-slot replica 2 bit for bit; replicas 0 and 1 now differ from the original runs (both paths), confirming that the original runs resumed those replicas from a zero force.
The two different magnitudes in the validator's runs (4e-4, 1e-3) came from two different fresh start runs; start-run repeatability was not checked.

### Recommendation
A one-line ordering fix: `CHECK(gpuDeviceSynchronize());` (or `gpuStreamSynchronize(0)`) after `allocate_memory_gpu` in `PRD_Driver::initialize_slot`, or once in `Replica_Runtime::initialize` before `compute_force` to cover both drivers. It is correct on CUDA too. The race is upstream's code, so per the scope rule it belongs in an upstream issue/PR (GPUMD is active); carrying it on moat-port as a separate commit is the alternative if the windows gate should pass before upstream takes it. jeff's call. Not a known-nondeterminism note: the per-slot result is the physically correct one and the batched (and replica 0/1) results are wrong on this platform.

Ruling (jeff, 2026-11-10): carry the PRD slot-allocation ordering fix on `moat-port` as its own vendor-neutral commit; no separate upstream PR.

## Porter 2026-11-10 (linux-gfx950, mi355) -- PRD slot-allocation race (windows-gfx1151 validation-failed), 5f6596ec -> 26547d27

jeff asked for this run. Host: MI355 (gfx950), CUDA 12.8 (V12.8.93) at `/usr/local/cuda-12.8`, TheRock pip SDK, `HIP_VISIBLE_DEVICES=1` (dispatcher-pinned), `ulimit -c 0`. `orient.sh` (on `port/GPUMD`) completed normally; `set-state GPUMD linux-gfx950 porting --agent porter` took the lock. Fresh fork clone, `moat-port` = 5f6596ec, `protect-fork` installed, local `master` from `origin/master`, `pr-state` merged.

### Change
Fork 26547d27 `Order replica slot allocation before slot-stream work` (53 chars, no `[ROCm]`: shared code, not ROCm-specific): `CHECK(gpuDeviceSynchronize());` after `allocate_memory_gpu(...)` in `PRD_Driver::initialize_slot` (`src/main_replica/replica_prd.cu:656-658`) with a one-line comment that the force zero fill runs on the default stream. No `USE_HIP` guard. `gpuDeviceSynchronize` wrapped in `CHECK` is the idiom `main_replica/main.cu:151,181` and `GPU_CHECK_KERNEL` use; the project has no default-stream-only sync idiom. Body: problem, cause, fix, "Authored with Claude.", Test Plan with fenced commands; `prose.py` and `jargon.py -` clean.

Other drivers checked: `replica_remd.cu:471-473` (`REMD_Driver::initialize_slot`) follows the allocation with the blocking `copy_from_host`, which is ordered after the default-stream fill (a pageable host-to-device copy syncs the stream before it starts), so REMD does not have the race; unchanged. `replica_batch.cu:443-458` resizes without a fill value (no kernel) and then uses blocking `copy_from_host`; `replica_common.cu:239-241` likewise resizes without a fill. Only PRD has the pattern.

### Build and tests (gfx950)
    utils/timeit.sh GPUMD compile -- make -C projects/GPUMD/src/src -f makefile.hip HIP_ARCH=gfx950 -j128

rc 0, 0 errors. gpumd 14715984, nep 3251088, tb 314616, gpumd_replica 2494368 B (+96 B over 5f6596ec).

Replica, from a detached worktree with `examples/nep_train/nep.txt` from cd7f757d and `examples/qnep_train/nep.txt` from eda8e04b (worktree removed after), `GPUMD_REPLICA=<clone>/src/gpumd_replica`:
- `python3 tests_regression/replica/test_replica.py -v`: **16/16 OK** (57.8 s), `test_batched_matches_per_slot_prd` ok.
- `-k test_batched_matches_per_slot -k test_split_resume_is_exact` (remd, prd, split-resume), 3 runs in a row: **3/3 OK each** (14.4, 15.4, 14.6 s). No hang this time (cf. the earlier gfx950 hang note).
This host never showed the race; the gfx1151 validator's rerun is the evidence that it is fixed there.

### CUDA compile (nvcc 12.8, compile only, no NVIDIA GPU)
    export PATH=/usr/local/cuda-12.8/bin:$PATH
    utils/timeit.sh GPUMD compile -- make -C <worktree of 26547d27>/src -j128 CUDA_ARCH=-arch=sm_90 all

rc 0, full build and link of gpumd 27322192, nep 8705816, tb 1613592, gpumd_replica 4750176 B; `replica_prd.o` compiled; no compile errors or host warnings; the 4 link lines print only the `prior to sm_75` deprecation notice (upstream's link lines carry no arch). PASS.

### Gates
- Draft follow-up PR body: one paragraph for 26547d27 (the race, the gfx1151 observation, timing-dependent on any GPU) and the CUDA-visible sentence now names the synchronization; the reminder names head 26547d27 and the nvcc pass here. `prose.py` and `jargon.py -` on the body: clean.
- `python3 utils/jargon.py --port GPUMD`: only the 2 known hits in upstream's 725fe762. `jargon.py -C projects/GPUMD/src --commits 5f6596ec..moat-port --diff 5f6596ec...moat-port`: clean.
- Integrity: `git -C projects/GPUMD/src status --porcelain` shows no modified tracked file.
- Pushed 5f6596ec..26547d27 to `moat-port` (fast-forward, pre-push hook passed; `ls-remote` = 26547d27). `advance-head GPUMD 26547d27...`.

Next: review of 26547d27, then revalidation of every gate at 26547d27 (windows-gfx1151 must rerun `test_batched_matches_per_slot_prd`).

## Review 2026-11-10 (linux-gfx1100, reviewer) -- delta 5f6596ec..26547d27 (PRD slot-allocation sync), REVIEW PASSED

jeff asked for this run. Scope: fork 26547d27 (`src/main_replica/replica_prd.cu`, +2 lines), its message, the diagnosis and porter entries above, and the one new paragraph in the draft follow-up PR body. Fresh fork clone, `moat-port` = 26547d27. Host: RX 7900 XTX (gfx1100, wave32), TheRock pip SDK, `HIP_VISIBLE_DEVICES=0` (dispatcher-pinned), `ulimit -c 0`. `orient.sh` completed and named reviewer; `set-state GPUMD linux-gfx1100 reviewing --agent reviewer` took the lock.

No problems found.
- Race: confirmed. `allocate_memory_gpu` (`model/read_xyz.cu:554`) does `force_per_atom.resize(N * 3, 0)`, whose `fill` launches `gpu_fill<<<>>>` with no stream (`utilities/gpu_vector.cuh:240`); `GPU_CHECK_KERNEL` syncs only in DEBUG builds. The fill is the last default-stream op in `allocate_memory_gpu` (the later `virial`/`potential`/`thermo` resizes are plain `gpuMalloc`). `Replica_Runtime::initialize` (`replica_common.cu:521-523`) then runs `compute_force` on the non-blocking slot stream with nothing ordering it after the fill; the PRD velocity upload is `copy_from_host_async` on the slot stream. The added `CHECK(gpuDeviceSynchronize())` at `replica_prd.cu:658` follows the only fill and all blocking copies in `initialize_slot` and precedes all slot-stream work. `allocate_memory_gpu` and `initialize_slot` have no other callers.
- Other drivers: REMD (`replica_remd.cu:471-480`) uses blocking `copy_from_host` in both branches (resume directly; fresh via `Velocity::initialize`, `velocity.cu:307`), which is ordered after the null-stream fill. `replica_batch.cu:443-458` and `replica_common.cu:239-241` resize without a fill. The only other default-stream fills in `src/main_replica/` are `Minimizer_FIRE::compute`'s `velocity_.fill(0.0)`, taken only when `stream == nullptr` (otherwise `fill_async` on the stream); `GPU_Pinned_Vector(size, value)` fills host memory. No other allocate-with-fill before slot-stream work.
- Message: no `[ROCm]` (vendor-neutral, correct), title 53 chars, "Authored with Claude.", fenced Test Plan, no Co-Authored-By, ASCII. `prose.py`: clean. `jargon.py --port GPUMD`: only the 2 known hits in upstream's 725fe762.
- Draft PR body paragraph ("Replica PRD: ..."): accurate; `prose.py` and `jargon.py -` on the body: clean. Its gfx1151 sentence rests on the diagnosis scratch build until the windows-gfx1151 revalidation at 26547d27 lands (the reminder already says every gate is owed).

Build and test (gfx1100):

    utils/timeit.sh GPUMD compile -- make -C projects/GPUMD/src/src -f makefile.hip HIP_ARCH=gfx1100 -j32

rc 0; gpumd_replica 2473784 B. Detached worktree at 26547d27 with `examples/nep_train/nep.txt` from cd7f757d and `examples/qnep_train/nep.txt` from eda8e04b (removed after):

    GPUMD_REPLICA=$PWD/projects/GPUMD/src/src/gpumd_replica python3 <worktree>/tests_regression/replica/test_replica.py -v -k test_batched_matches_per_slot_prd

**1/1 OK** (4.9 s). This host never showed the race, so this is a no-regression check, not evidence of the fix. Clone `status --porcelain`: clean.
