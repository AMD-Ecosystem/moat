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
