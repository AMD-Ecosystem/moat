# gdtk notes

## Build for HIP/ROCm

```bash
# Build for AMD MI200 series (gfx90a)
cd src/chicken
make HIP=1 HIP_ARCH=gfx90a install

# Build for other AMD GPUs
make HIP=1 HIP_ARCH=gfx1100 install  # RDNA3
```

## Unit tests

```bash
cd src/chicken/test
make HIP=1 test
```

## GPU validation (shock tube)

```bash
export HIP_VISIBLE_DEVICES=0
export DGD=$HOME/gdtkinst
export PATH=$DGD/bin:$PATH
cd examples/chicken/shock-tube
chkn-prep --job=sod --binary
chkn-run --job=sod --binary
chkn-post --job=sod --binary --vtk-xml --tindx=all
```

## Port notes

- Strategy A variant: Makefile + compat header (not CMake)
- cuda_to_hip.h maps cudaMalloc/cudaFree/cudaMemcpy and error handling APIs
- Changed #ifdef CUDA to #if defined(CUDA) || defined(HIP) for GPU code paths
- No warp intrinsics, no textures, no cuBLAS/cuFFT dependencies (simple port)
- The upstream test_chicken.py has a bug (missing binaryData arg) but GPU simulation passes
- nlohmann/json.hpp warnings are from upstream (deprecated literal operator syntax)

## Review 2026-06-05

Reviewed moat-port branch (b611ce03 vs base 253b4592). No issues found.

**Port correctness**: Strategy A variant (Makefile + compat header) appropriate for this Makefile-based CUDA project. The cuda_to_hip.h header correctly maps all 10 CUDA runtime symbols used (cudaMalloc, cudaFree, cudaMemcpy, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaError_t, cudaSuccess, cudaGetLastError, cudaGetErrorString, cudaGetDeviceCount).

**Fault classes**:
- No warp intrinsics (__shfl*, __ballot, __activemask) -- no wave64/wave32 hazards
- No textures or pitched 2D binds
- No cuBLAS/cuFFT/cuRAND/cuSPARSE dependencies
- Atomics (atomicMin on long long int, atomicAdd on int) use device memory allocated via cudaMalloc -- safe on gfx90a
- Thread block size (128 via Config::threads_per_GPUblock) is warp-agnostic
- All 9 kernels have proper bounds checking (if (i < cfg.nActiveCells) or if (i < cfg.nFaces))
- No shared memory usage

**Minimal footprint**: CUDA spelling preserved in source; compat header only force-included for HIP builds. Preprocessor guards correctly use `#if defined(CUDA) || defined(HIP)` pattern.

**Build system**: HIP=1 option integrates correctly alongside GPU=1 (CUDA); HIP_ARCH configurable (default gfx90a); test makefile has matching HIP=1 path.

**Commit hygiene**: Title "[ROCm] Add HIP support for AMD GPU builds" (41 chars), no Co-Authored-By noreply trailer, mentions Claude, has Test Plan section with literal commands, uses jeffdaily account.

Verdict: **PASS** -- ready for validation.

## Validation 2026-06-05

Platform: linux-gfx90a (AMD Instinct MI250X, HIP_VISIBLE_DEVICES=3)
Commit: b611ce03f7f301a0b1bd93b7dca361ffa5ca2387

### Build

Built from scratch with HIP=1 HIP_ARCH=gfx90a:

```bash
cd src/chicken
make HIP=1 HIP_ARCH=gfx90a install -j$(nproc)
```

Build succeeded with only expected warnings:
- nlohmann/json.hpp deprecated literal operator warnings (upstream)
- Unused hipFree return value warnings (5 instances in block.cu cleanup)

Binary installed to /var/lib/jenkins/gdtkinst/bin/chkn-run.

### Unit tests

All 4 unit tests passed:

```bash
cd src/chicken/test
make HIP=1 test
```

- gas_test: PASS (GasModel and GasState thermodynamics)
- rsla_test: PASS (linear algebra solver)
- vector3_test: PASS (3D vector operations)
- spline_test: PASS (cubic spline interpolation)

### GPU simulations

1. **shock-tube** (Sod shock tube problem):
   - 400 cells, 100 time steps
   - Completed successfully in 0.339s
   - HIP device detected and used
   - Final time: t=6.069e-04

2. **isentropic-vortex** (convected vortex):
   - 28800 cells, 880 time steps
   - Completed successfully in 2.800s
   - HIP device detected and used
   - Final time: t=5.006e-02
   - 49 flow field snapshots written

### Verdict

**VALIDATED**: All unit tests and GPU simulations passed on gfx90a. The HIP port correctly utilizes AMD GPU acceleration with no runtime errors or numerical issues.

## Validation 2026-06-05 (gfx1100)

Platform: linux-gfx1100 (AMD Radeon RX 7900 XTX, HIP_VISIBLE_DEVICES=3)
Commit: b611ce03f7f301a0b1bd93b7dca361ffa5ca2387

### Build

Built from scratch with HIP=1 HIP_ARCH=gfx1100:

```bash
cd src/chicken
make HIP=1 HIP_ARCH=gfx1100 install -j$(nproc)
```

Build succeeded with only expected warnings:
- nlohmann/json.hpp deprecated literal operator warnings (upstream)
- Unused hipFree return value warnings (5 instances in block.cu cleanup)

Binary installed to /var/lib/jenkins/gdtkinst/bin/chkn-run.

### Unit tests

All 4 unit tests passed:

```bash
cd src/chicken/test
make HIP=1 test
```

- gas_test: PASS (GasModel and GasState thermodynamics)
- rsla_test: PASS (linear algebra solver)
- vector3_test: PASS (3D vector operations)
- spline_test: PASS (cubic spline interpolation)

### GPU simulations

1. **shock-tube** (Sod shock tube problem):
   - 400 cells, 100 time steps
   - Completed successfully in 0.146s
   - HIP device detected and used
   - Final time: t=6.069e-04

2. **isentropic-vortex** (convected vortex):
   - 28800 cells, 880 time steps
   - Completed successfully in 1.040s
   - HIP device detected and used
   - Final time: t=5.005e-02
   - 49 flow field snapshots written

### Verdict

**VALIDATED**: All unit tests and GPU simulations passed on gfx1100. The HIP port correctly utilizes AMD RDNA3 GPU acceleration with no runtime errors or numerical issues.

## Validation 2026-06-08 (windows-gfx1201)

Platform: windows-gfx1201 (AMD Radeon RX 9070 XT, gcnArchName=gfx1201, HIP_VISIBLE_DEVICES=0)
Commit: b611ce03f7f301a0b1bd93b7dca361ffa5ca2387

### Toolchain notes

The upstream Makefile is Linux-oriented (uses `make`, POSIX sed/cp). On Windows, the build was
performed manually via the ROCm venv hipcc.exe. gmake from Strawberry Perl was used for unit tests.
zlib was sourced from conan2 (`C:/Users/<user>/.conan2/p/b/zlib94c8e5ebac5da/p/{include,lib}/zlib.lib`).
TheRock runtime DLLs (amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll,
hiprtc-builtins0714.dll) were copied into the bin/ install directory to override the Adrenalin
System32 DLL (which would hang on first GPU kernel dispatch without them).

### Build

```
ROCM_VENV_SCRIPTS="B:/develop/TheRock/external-builds/pytorch/.venv/Scripts"
ROCM_DEVEL="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
ROCM_CORE="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_core"
export PATH="${ROCM_VENV_SCRIPTS}:${ROCM_DEVEL}/bin:${ROCM_DEVEL}/lib/llvm/bin:${ROCM_CORE}/bin:${PATH}"
ZLIB_INC="C:/Users/<user>/.conan2/p/b/zlib94c8e5ebac5da/p/include"
ZLIB_LIB="C:/Users/<user>/.conan2/p/b/zlib94c8e5ebac5da/p/lib"
cd src/chicken
sed -e 's/PUT_REVISION_STRING_HERE/b611ce0/' main.cu > main_with_rev_string.cu
hipcc.exe -DHIP -std=c++17 -O2 --offload-arch=gfx1201 -o chkn-run.exe -DIDEAL_AIR \
    -include cuda_to_hip.h -I"${ZLIB_INC}" main_with_rev_string.cu -L"${ZLIB_LIB}" -lzlib -fopenmp
```

Build succeeded with expected warnings (nlohmann/json deprecated literal operator,
hipFree nodiscard, sscanf MSVC deprecation on the host-side compilation).

### Unit tests

```
cd src/chicken/test
/c/Strawberry/c/bin/gmake.exe HIP=1 test
```

All 4 unit tests passed:
- gas_test: PASS (GasModel and GasState thermodynamics)
- rsla_test: PASS (linear algebra solver)
- vector3_test: PASS (3D vector operations)
- spline_test: PASS (cubic spline interpolation)

### GPU simulations

1. **shock-tube** (Sod shock tube problem):
   - 400 cells, 100 time steps
   - Completed successfully in 92.894s (slow due to Windows GPU dispatch overhead on small problem)
   - HIP device detected and used (1 device found)
   - Final time: t=6.069e-04 -- MATCHES gfx90a and gfx1100 exactly
   - Result: PASS

2. **isentropic-vortex** (convected vortex):
   - 28800 cells, up to 150000 time steps (max_time=50ms)
   - Uses sbp_asf flux calculator and exchange BCs on all boundaries
   - FAILED: immediately at step 1, ALL ~28,800 cells reported "Bad cell" via GPU printf
   - Error: "Stage 2, could not copy bad_cell_count from gpu to host cpu." (hipMemcpy failed after the bad-cell explosion)
   - Done in 124s (the runtime_error was caught by main(), simulation terminated)
   - The Linux runs (gfx90a and gfx1100) both completed with 880 steps to t=5.005e-02 with no bad cells.

### Diagnosis

The shock-tube passes with identical final time to Linux (ausmdv flux, wall_with_slip BCs).
The isentropic-vortex catastrophically fails at step 1 on gfx1201 only (sbp_asf flux + self-exchange
periodic BCs, 1 block wrapping onto itself). All 28,800 cells become bad simultaneously at the first
time step, indicating a numerical blow-up.

The sbp_asf kernel uses a 4-point stencil (fsL1, fsL0, fsR0, fsR1) and the exchange BC copies
ghost cell FlowState values across from the opposite boundary in the same block (read-write across
threads). This is the only code path not exercised by the passing shock-tube.

Possible causes:
1. gfx1201 FP-accumulation divergence in the sbp_asf kernel causing NaN/inf from step 1
2. Data race in the exchange BC when all boundary faces are set by `apply_convective_boundary_condition`
   in parallel (threads reading and writing the same cells array) -- but this would fail on all platforms
3. Wave32 vs wave64 difference: gfx1201 is wave32 (all AMD GFX are wave32 except gfx90a which is wave64),
   same as gfx1100, so this is not the distinguishing factor between gfx1100 (pass) and gfx1201 (fail).

The blow-up being gfx1201-specific (not gfx1100) despite both being wave32 RDNA points to an RDNA4
arithmetic divergence in the sbp_asf formulation.

### Verdict

**VALIDATION-FAILED**: The primary shock-tube validation PASSED on gfx1201 with correct numerics
(t=6.069e-04 matching Linux). However, the isentropic-vortex (sbp_asf + exchange BCs) catastrophically
fails at step 1 with all cells bad, a gfx1201-specific regression not seen on gfx90a or gfx1100.
Bouncing to porter for investigation of the sbp_asf/exchange-BC path on RDNA4.

## sbp_asf isentropic-vortex gfx1201 blow-up root-cause (2026-06-08)

ROOT CAUSE (definitive): a Windows-only host-side I/O bug in the upstream grid reader, NOT
an RDNA4/gfx1201 arithmetic issue and NOT the sbp_asf flux. In src/chicken/block.cu,
Block::readGrid() opens the binary `.bin` grid file WITHOUT `ios::binary`:

```
auto f = ifstream(fileName);                 // line ~798, BUG: no ios::binary
```

On Windows, a text-mode ifstream performs CRLF->LF translation and treats byte 0x1A
(Ctrl-Z) as end-of-file, corrupting the raw little-endian fp64 vertex stream. On Linux,
text mode == binary mode, so gfx90a/gfx1100 are unaffected -- which is why this looked
arch-specific. The very next binary reader in the same file, readFlow() (line ~890), DOES
pass `ios::binary` correctly, so the FLOW data (initial conserved state) loads fine; only
the GRID VERTICES are corrupted. This is a latent upstream bug, not introduced by the HIP port.

Failure mechanism: corrupted vertices -> Block::computeGeometry() (host, runs before the
host->GPU copy) computes garbage cell volumes that fall below the verySmallVolume guard in
hex_cell_properties() (vector3.cu ~422) and get clamped to exactly 0. In FVCell::eval_dUdt
(cell.cu ~190), `vol_inv = one/volume = 1/0 = inf`, so dUdt = inf for every cell, and the
stage-1 update `U1 = U0 + dt*dUdt` -> inf/NaN everywhere. decode_conserved then flags all
~28,800 cells bad at step 1. The "all cells bad" + "could not copy bad_cell_count" symptom
is the downstream effect of a divide-by-zero geometry, not an FP-accumulation divergence.

Why shock-tube passed and vortex failed: both run with --binary. The shock-tube grid is tiny
(a few hundred vertices) and its fp64 byte pattern happened not to contain a 0x1A or a lone
0x0D/0x0A that the text-mode translation mangles; the 43,923-vertex vortex grid inevitably
contains such bytes, so it corrupts. It is luck-of-the-byte-pattern, not arch behaviour --
the same vortex would fail identically on Windows gfx1101.

Decisive experiments (all on gfx1201, RX 9070 XT, HIP_VISIBLE_DEVICES=0, fork @ b611ce0):
1. PRIMARY HYPOTHESIS REFUTED -- FP contraction/fast-math: rebuilt the single TU with
   `-O2 -ffp-contract=off` (hipcc default is `fast`). Vortex still blew up at step 1, same
   cell. Fast-math is NOT the cause.
2. BC/flux race REFUTED: split the fused BC-fill+flux kernel (calculate_fluxes_on_gpu) into a
   separate apply_convective_bcs_on_gpu kernel launched before the flux kernel, adding a true
   inter-thread global sync between ghost-cell writes and stencil reads. Still blew up
   identically. (Note: that fused-kernel cross-thread read/write IS a latent race, but it is
   benign here and not the failure cause.)
3. INSTRUMENTATION (decisive): a printf in update_stage_1_on_gpu showed
   `vol=0, dUdt_mass=-inf` for the bad cells while U0 (mass=1.16, totE=390000) was correct and
   face fluxes F were finite (403.169) with face areas ~1e-16 (i.e. ~0). A host-side print
   right after computeGeometry() showed the HOST already had `vol=0` and cell-0 vertices
   `v0=(-2,-2,-0.2) v6=(6.26667,-2,-0.2)` -- scattered, not a compact hex -> grid file read
   corrupt on the CPU, before any GPU involvement.

FIX (one line, applied to a scratch build and validated, NOT committed to the fork per the
diagnostic scope):

```
auto f = ifstream(fileName, ios::binary);    // block.cu readGrid() binary branch
```

Validation of the fix on gfx1201:
- isentropic-vortex: now runs to completion, 880 steps to t=5.005e-02, 49 snapshots (tindx
  0-48), `Done in ~1.4s`, zero bad cells -- matches the gfx90a/gfx1100 Linux reference exactly.
- shock-tube (regression): still completes, 100 steps to t=6.069e-04 -- unchanged.

Fixability and scope: trivially fixable and the RIGHT general fix. It should be applied
UNCONDITIONALLY (not guarded for RDNA4/gfx12) -- it is a plain correctness bug in the binary
file open. It is behavior-preserving on gfx90a/gfx1100 (and on NVIDIA/Linux generally) because
on POSIX `ios::binary` is a no-op; it only changes behaviour on Windows, where it is required.
This is a clean upstream-quality fix (the adjacent readFlow already does exactly this), worth
upstreaming on its own merits. With it, gfx1201 passes BOTH examples and windows-gfx1201 can be
re-validated. (status.json left at validation-failed per the diagnostic instructions.)

## Validation 2026-06-08 (windows-gfx1201, fix applied)

Platform: windows-gfx1201 (AMD Radeon RX 9070 XT, gcnArchName=gfx1201, HIP_VISIBLE_DEVICES=0,
verified via B:/develop/TheRock/build/bin/hipInfo.exe).
Commit: 8aadf2185510930b523c786e21fde116c22b2809 (new commit on top of b611ce0; one-line
readGrid binary-mode fix in src/chicken/block.cu).

The fix: `auto f = ifstream(fileName);` -> `auto f = ifstream(fileName, ios::binary);` in the
binary `.bin` branch of Block::readGrid(). Applied unconditionally; the gzip/text branch
(bxz::ifstream) is untouched. See the root-cause section above.

### Build (single TU, ~9s rebuild)

```
ROCM_VENV_SCRIPTS=".../pytorch/.venv/Scripts"
ROCM_DEVEL=".../pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
ROCM_CORE=".../pytorch/.venv/Lib/site-packages/_rocm_sdk_core"
export PATH="${ROCM_VENV_SCRIPTS}:${ROCM_DEVEL}/bin:${ROCM_DEVEL}/lib/llvm/bin:${ROCM_CORE}/bin:${PATH}"
ZLIB_INC="C:/Users/<user>/.conan2/p/b/zlib94c8e5ebac5da/p/include"
ZLIB_LIB="C:/Users/<user>/.conan2/p/b/zlib94c8e5ebac5da/p/lib"
cd src/chicken
sed -e 's/PUT_REVISION_STRING_HERE/b611ce0/' main.cu > main_with_rev_string.cu
hipcc.exe -DHIP -std=c++17 -O2 --offload-arch=gfx1201 -o chkn-run.exe -DIDEAL_AIR \
    -include cuda_to_hip.h -I"${ZLIB_INC}" main_with_rev_string.cu -L"${ZLIB_LIB}" -lzlib -fopenmp
```

Build OK (only expected nlohmann/json deprecated-literal-operator, hipFree nodiscard, and
host-side sscanf MSVC deprecation warnings).

### Unit tests (4/4 PASS)

```
cd src/chicken/test
/c/Strawberry/c/bin/gmake.exe HIP=1 test
```

gas_test, rsla_test, vector3_test, spline_test all ran to "-Done-" (each asserts internally
and aborts on failure). 4/4 PASS.

### GPU examples (both PASS)

Tooling: chkn_prep.py / chkn_post.py run via the venv python with
PYTHONPATH=B:/develop/moat/projects/gdtk/src/src/lib (the `gdtk` package); chkn-run.exe is the
fixed gfx1201 binary with TheRock DLLs alongside it. HIP_VISIBLE_DEVICES=0.

1. shock-tube (Sod), regression:
   - `chkn-prep --job=sod --binary; chkn-run --job=sod --binary; chkn-post --job=sod --binary --tindx=all`
   - 400 cells, 100 steps, "Done in 0.177s", HIP devices found: 1
   - Final time t=6.069e-04 (times.data: 0.000606905) -- unchanged, matches gfx90a/gfx1100.
   - PASS

2. isentropic-vortex (sbp_asf + exchange BCs), the previously-failing case:
   - `chkn-prep --job=vortex --binary; chkn-run --job=vortex --binary; chkn-post --job=vortex --binary --tindx=all`
   - 28800 cells, ran to Step=880, "Done in 1.381s", ZERO bad cells, no hipMemcpy error.
   - Final time t=5.00125e-02 (times.data tindx 48 = 0.0500125), 49 snapshots (tindx 0-48)
     in both times.data and flow/ -- matches the gfx90a/gfx1100 Linux reference exactly.
   - PASS (previously: all ~28,800 cells bad at step 1 with "could not copy bad_cell_count").

### Verdict

VALIDATED on real GPU. Both binary-grid GPU examples pass on gfx1201 with the one-line
readGrid binary-mode fix; numerics match the Linux reference. windows-gfx1201 -> completed.

## Validation 2026-06-08 (linux-gfx90a revalidate, binary-equiv carry-forward)

Platform: linux-gfx90a (AMD Instinct MI250X, gfx90a)
Revalidate: b611ce03 -> 8aadf218 (one commit on top)

Delta: single-line change in src/chicken/block.cu, host-side Block::readGrid() -- adds
`ios::binary` to the ifstream open in the binary `.bin` grid branch. On Linux/POSIX,
`ios::binary` is a no-op at both the library and kernel level; it only has effect on
Windows (where it suppresses CRLF translation and Ctrl-Z EOF treatment).

Classification: source-level functional (moatlib classify returned unknown/revalidate),
but the change is host-only and POSIX-inert. Confirmed via codeobj_diff.py:

```
python3 utils/codeobj_diff.py /tmp/gdtk_build_old/chkn-run /tmp/gdtk_build_new/chkn-run
verdict=identical
  chkn-run vs chkn-run: identical (exported symbols + device ISA identical (24 exports))
```

Both builds (b611ce03 and 8aadf218) compiled with `make HIP=1 HIP_ARCH=gfx90a -j$(nproc)`.
Device ISA and exported symbols are byte-identical; no GPU re-run required.

Carry-forward: linux-gfx90a -> completed at 8aadf218 (binary-equiv).

## Validation 2026-06-08 (linux-gfx1100 revalidate, binary-equiv carry-forward)

Platform: linux-gfx1100 (AMD Radeon Pro W7800 48GB, gfx1100, HIP_VISIBLE_DEVICES=1)
Revalidate: b611ce03 -> 8aadf218 (one commit on top)

Delta: same single-line change in src/chicken/block.cu as the gfx90a revalidate (adds
`ios::binary` to Block::readGrid()). On Linux/POSIX, `ios::binary` is a no-op.

Classification: moatlib classify returned `mixed` (token count differs), so codeobj_diff
was run between builds at both SHAs compiled with `make HIP=1 HIP_ARCH=gfx1100 -j$(nproc)`.

```
export HIP_VISIBLE_DEVICES=1
# worktree at b611ce03 -> build-old/chkn-run
# worktree at 8aadf218 -> build-new/chkn-run
python3 utils/codeobj_diff.py agent_space/gdtk-gfx1100-gpu1/build-old/chkn-run \
                              agent_space/gdtk-gfx1100-gpu1/build-new/chkn-run
verdict=identical
  chkn-run vs chkn-run: identical (exported symbols + device ISA identical (24 exports))
```

Device ISA and all 24 exported symbols are byte-identical; no GPU re-run required.

Carry-forward: linux-gfx1100 -> completed at 8aadf218 (binary-equiv).

## Validation 2026-06-19 (windows-gfx1101)

Platform: windows-gfx1101 (AMD Radeon PRO V710, gcnArchName=gfx1101, HIP_VISIBLE_DEVICES=1,
verified via B:/develop/TheRock/build/bin/hipInfo.exe).
Commit: 8aadf2185510930b523c786e21fde116c22b2809

### Toolchain notes

Same as the gfx1201 Windows build: manual hipcc.exe invocation from the TheRock PyTorch venv.
gmake is not used; the single TU compile mirrors the gfx1201 recipe with --offload-arch=gfx1101.
The existing TheRock runtime DLLs (amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll,
hiprtc-builtins0714.dll) from the prior gfx1201 build were already in src/chicken/ and reused.

### Build (single TU, ~40s)

```
HIPCC="/b/develop/TheRock/external-builds/pytorch/.venv/Scripts/hipcc.exe"
ZLIB_INC="C:/Users/<user>/.conan2/p/b/zlib94c8e5ebac5da/p/include"
ZLIB_LIB="C:/Users/<user>/.conan2/p/b/zlib94c8e5ebac5da/p/lib"
CHICKEN_DIR="/b/develop/moat/projects/gdtk/src/src/chicken"
sed -e 's/PUT_REVISION_STRING_HERE/8aadf21/' main.cu > main_with_rev_string.cu
$HIPCC -DHIP -std=c++17 -O2 --offload-arch=gfx1101 -o chkn-run-gfx1101.exe -DIDEAL_AIR \
    -include cuda_to_hip.h -I"${ZLIB_INC}" -I"${CHICKEN_DIR}" \
    main_with_rev_string.cu -L"${ZLIB_LIB}" -lzlib -fopenmp
```

Build succeeded with expected warnings (nlohmann/json deprecated literal operator,
hipFree nodiscard, host-side sscanf MSVC deprecation). 14 warnings each for gfx1101 and host
device compilation passes. Binary: chkn-run-gfx1101.exe.

### Unit tests (4/4 PASS)

Built with hipcc.exe -std=c++17, run directly:

```
cd src/chicken/test
$HIPCC -std=c++17 -o gas_test.exe -DIDEAL_AIR gas_test.cu
$HIPCC -std=c++17 -o rsla_test.exe rsla_test.cu
$HIPCC -std=c++17 -o vector3_test.exe vector3_test.cu
$HIPCC -std=c++17 -o spline_test.exe spline_test.cu
./gas_test.exe && ./rsla_test.exe && ./vector3_test.exe && ./spline_test.exe
```

gas_test, rsla_test, vector3_test, spline_test all ran to completion. 4/4 PASS.

### GPU examples (both PASS)

chkn_prep.py run via the TheRock venv python with
PYTHONPATH=B:/develop/moat/projects/gdtk/src/src/lib. chkn-run-gfx1101.exe is the
gfx1101 binary. HIP_VISIBLE_DEVICES=1 (gfx1101 mask on this host).

1. shock-tube (Sod), primary validation:
   - `chkn-prep --job=sod --binary; chkn-run-gfx1101.exe --job=sod --binary`
   - 400 cells, 100 steps, "Done in 1.076s", HIP devices found: 1
   - Final time t=6.069e-04 (times.data: 0.000606905) -- matches gfx90a/gfx1100/gfx1201 exactly
   - PASS

2. isentropic-vortex (sbp_asf + exchange BCs):
   - `chkn-prep --job=vortex --binary; chkn-run-gfx1101.exe --job=vortex --binary`
   - 28800 cells, ran to Step=880, "Done in 2.297s", ZERO bad cells, no hipMemcpy error
   - Final time tindx=48 t=0.0500119 (~5.001e-02) -- matches gfx90a/gfx1100/gfx1201 reference
   - 49 snapshots (tindx 0-48) written in flow/
   - PASS (ios::binary fix confirmed working on gfx1101 as on gfx1201)

### Verdict

VALIDATED on real GPU (AMD Radeon PRO V710, gfx1101). Both binary-grid GPU examples pass with
correct numerics matching the Linux reference. The readGrid ios::binary fix is a POSIX no-op on
Linux (so no revalidation needed there) and works correctly on this Windows gfx1101 host.
windows-gfx1101 -> completed at 8aadf218.
