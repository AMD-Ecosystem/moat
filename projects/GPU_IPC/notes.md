# GPU_IPC notes

## Port status

2026-11-23: fork `moat-port` @ 8ee59a5, one commit on top of jeff's squash
21bce04 (tree-identical to c44ed01). It makes the Jacobi eigensolver in `PDSNK`
HIP-only; the CUDA build compiles upstream's `Eigen::SelfAdjointEigenSolver`
code again. See "## Porter round 2026-11-23 (linux-gfx942)". The CUDA compile
of 8ee59a5 is owed (no nvcc on the gfx942 host).

2026-11-20: fork `moat-port` @ 29e2494, a merge of upstream main (4fb6019,
13 commits past the old base 46594e0) on top of f9604bb. CUDA-compiled, and
validated on gfx1100 (W7800) at 29e2494: 90 frames, no wedge (see
"## Validation 2026-11-20"). linux-gfx90a still reads `revalidate` at 29e2494.
Earlier history: 3798cb2 (port) and f9604bb (warp-intrinsic functions), both
validated on gfx90a and gfx1100.

## Build recipe (Linux)

```bash
sudo apt install libeigen3-dev libglew-dev freeglut3-dev
cmake -S . -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j16
```

ROCm 7.2.53211, cmake 3.31.6, Eigen 3.4.0, GLEW 2.2.0, freeglut 3.4.0.

## Running it headless

There is no test suite. `gl_main.cpp` puts the entire simulation inside the GLUT
display callback and starts with `bool stop = true` (gl_main.cpp:65), so the
program does nothing until it receives a space keypress. Two things are needed to
get any evidence out of it on a headless host:

- `Xvfb :77 -screen 0 1024x1024x24` with `LIBGL_ALWAYS_SOFTWARE=1`, so rendering
  falls to llvmpipe while the compute stays on the AMD GPU.
- `xdotool key --window $(xdotool search --name FEM) space` to start the sim, and
  `stdbuf -o0` so the `printf` progress is not lost in the block buffer when the
  run is killed.

The harness used for this session is reproduced in the commit's Test Plan.

## Results on gfx1100 (GPU index 2, ROCm 7.2)

Default scene: 38386 vertices, 159870 tetrahedra, 41664 faces, 20836 surface
vertices, 62496 surface edges, 4 preconditioner levels.

| run | flags | frames completed | ms/frame |
|-----|-------|------------------|----------|
| 3 | `-ffast-math` | 32 | 206-225 |
| 4 | `-ffp-contract=on` | 26 | 215-216 |
| 5 | `-ffp-contract=on` (repeat of 4) | 32 | 216-228 |

Newton converges in 2 to 26 iterations per frame; kappa stays near 2824 and steps
to 2955 once. No HIP runtime error, no crash, no NaN.

## Open issue: the line search wedges after ~30 frames

Somewhere between frame 26 and frame 33 the scene reaches a state that
`GIPC::lineSearch` reports as self-intersecting for one small cluster of cloth
elements (vertices 12866-12893, and in one run 7304-11662 / 17484-17581). The
loop at GIPC.cu:10024 is

```cpp
while(checkInterset && isIntersected(TetMesh)) { ... alpha /= 2.0; ... }
```

which has no iteration cap, so once the CURRENT configuration is already
intersecting, halving alpha cannot help and the process spins forever printing
"type 0 intersection happened". That unbounded loop is upstream code and is
backend-independent.

What is known:

- The frame it happens on VARIES between runs of the same binary (26 vs 32). The
  solver accumulates through `atomicAdd` on doubles throughout, so run-to-run
  reordering is expected and this is not by itself evidence of uninitialized
  memory or a race.
- It is not the eigensolver replacement. That was checked on the host against the
  Eigen original over 20000 random symmetric 12x12 matrices: worst relative
  difference 1.5e-14, worst absolute 4.0e-14, every result positive semi-definite.
  Test source: agent_space is not used; the check is reproduced from the commit.
- It is not `-ffast-math`. Removing it changed the flag-related numerics (and was
  removed on its own merits, see below) but the wedge still occurs.

Not yet ruled out, in the order worth trying (re-ranked at review; this is the
validator's to settle, not the porter's):

1. Whether upstream does the same thing on an NVIDIA GPU at some frame. There is
   still no baseline. Until it exists every other hypothesis is speculation, and
   one run settles port-versus-upstream. Cheap for anyone with a card.
2. Uninitialized device memory -- but test it DECISIVELY rather than by audit:
   temporarily make `hipMalloc` also `hipMemset(p, 0, n)`, then separately 0xFF,
   and see whether the stall frame moves or becomes reproducible. ROCm does not
   hand back zeroed pages where CUDA's allocator often does, and GIPC sizes its
   collision-pair and Hessian buffers for the worst case
   (`maxCollisionPairsNum_CCD: 2499840`) while relying on counters to say how
   much is live. Auditing every `cudaMalloc` for a matching `cudaMemset` is slow
   and easy to get wrong.
3. Whether the scene is simply meant to be stopped before this point. The demo has
   a `totalFrames` exit that upstream left commented out (gl_main.cpp:943-946).

Two mechanisms were TESTED AND RULED OUT at review; do not repeat them.

- The second-stage reduction in `add_reduction` and `PCG_add_Reduction_*` shuffles
  past `warpNum` after an early `return`, which reads like an uninitialized-register
  read. A faithful replica measured EXACT against a CPU sum at 13 sizes, including
  non-power-of-two `warpNum` (352, 416, 448, 480, 100000, 100001).
- The partial-block `__syncthreads()` reached only by threads that passed
  `if (idx >= numbers) return` neither hangs nor corrupts on this hardware.

Also not the cause: wave64. See the review response below -- the wave64
application failure is rocThrust under a forced wavefront size on RDNA, and a
build whose non-thrust kernels all run wave64 reaches this same wedge at the same
frame.

## Resolved: the Eigen device eigensolver blocker

The previous session stopped here, and the diagnosis in it was half right. Eigen
does support HIP device compilation -- `EIGEN_DEVICE_FUNC` expands to
`__host__ __device__` for hipcc exactly as it does for nvcc, and Eigen 3.4's
`SelfAdjointEigenSolver` header carries 23 of those annotations. Core arithmetic,
`Map`, `.cross()` and small products all compile fine.

The boundary is `computeDirect()` versus the general `compute()` path, NOT a
matrix size. An earlier version of this note said "above 3x3", which is wrong and
was corrected after the review: a 3x3 `SelfAdjointEigenSolver` CONSTRUCTOR also
fails to compile in a `__device__` function, at `Tridiagonalization.h:434`
(reconfirmed here directly with hipcc against Eigen 3.4). `computeDirect()`
exists only for 2x2 and 3x3 and is the only device-safe entry point; the general
`compute()` path -- which the constructor runs, at every size -- reduces through
a Householder tridiagonalization whose inner products are Eigen's general
matrix-matrix product kernels (`scaleAndAddTo`, `applyThisOnTheLeft`,
`BlasUtil::extract`, `extractScalarFactor`), and those are host-only. That is why
femEnergy.cu:1116-1117 survives: it is a `computeDirect` call.

The whole blocker was 8 errors, all with `note: called by
'_calculate_bending_gradient_hessian'`, i.e. ONE call site:
`PDSNK<double,12>` at femEnergy.cu:1335. The 9x9 path
(`__project_StabbleNHK_H_3D_makePD`) already used the project's own `qr_svd` and
compiled without complaint. Reading the `note: called by` lines rather than the
Eigen-internal error locations is what turned "Eigen is incompatible with HIP"
into a one-function fix.

Replacement is a cyclic Jacobi sweep over plain `Scalar[12][12]` arrays. Jacobi
needs no workspace and no matrix product, is backward stable on symmetric input,
and compiles identically for nvcc and hipcc, so there is no conditional fork. The
`V * D * V^T` reconstruction is written as explicit loops for the same reason --
a fixed-size 12x12 Eigen product re-enters the host-only path.

Correction 2026-11-23: "a fixed-size 12x12 Eigen product re-enters the
host-only path" is wrong. `V * D * V.transpose()` on `Matrix<double,12,12>`
compiles and device-links in a HIP kernel (hipcc 7.16.26385, Eigen 3.4.0,
gfx942, with and without `-fgpu-rdc`); the host-only `scaleAndAddTo` calls in
the solver are on the dynamic-size blocks of the tridiagonalization. That is
what lets the 2026-11-23 round share upstream's reconstruction on both back
ends. "No conditional fork" no longer holds either: the solver is now HIP-only.

Register cost: two 12x12 double arrays per thread (2304 bytes) spill to scratch.
The Eigen original also built 12x12 temporaries on the stack, so this is not new,
but it is worth measuring if anyone optimizes this kernel.

## Also fixed this session

`cuda_to_hip.h` mapped `__shfl_*_sync` to the UNQUALIFIED HIP forms, which take
the wavefront width. Correct on gfx1100 (wave32), wrong on gfx90a/gfx942
(wave64): PCG_SOLVER.cu and mlbvh.cu reduce with `delta = 1,2,4,8,16` over logical
32-element groups and would have crossed group boundaries on a 64-lane wavefront.
Width is now pinned to 32. The ballot is shifted to the calling lane's own 32-lane
half rather than truncated. Neither was observable on this arch -- they are
wave64 correctness fixes made blind, and gfx90a/gfx942 validation is what proves
them.

`-ffast-math` was dropped from the HIP flags. It was added to mirror the CUDA
build's `--use_fast_math`, but clang's version additionally enables
finite-math-only, reassociation and reciprocal math in DOUBLE precision, which is
not what nvcc's flag does and is not survivable for geometric predicates.
`-ffp-contract=on` replaces it to match nvcc's expression-scope contraction.

MASPreconditioner.cu was reviewed and cleared. **That clearance was wrong on its
facts and the file has since been changed** -- see the review response below.
`WARP_BALLOT` (line 1047) and `WARP_SHFL_DOWN` (893-895, 1054) are live in it,
not commented out. What remains true: `BANKSIZE 32` is a logical cluster size and
its lane indices come from `idx % BANKSIZE` rather than a hardware lane id.

## Not done

- No NVIDIA no-regression compile check with nvcc. The CUDA path is additive-only
  by inspection (compat header inert without `USE_HIP`, all CMake edits inside the
  `USE_HIP` branch, `PDSNK` now backend-neutral), but it has not been compiled.
- No wave64 run. The two warp fixes above are unexercised.

## Review 2026-08-08

Reviewed fork sha 291c612 (`moat-port`) against 46594e0. Verdict:
changes-requested. No review PR was opened: gates are wave64/wave32/windows and
none are satisfied, so `upstream.py --review` correctly refuses, and finding 1
means there is nothing to approve yet.

### Blocking

**1. The port does not run at wave64.** gfx1100 supports wave64, so this is
testable on the existing host without a CDNA card. Rebuilding this exact branch
with `-DCMAKE_HIP_FLAGS=-mwavefrontsize64` (confirmed to reach the device
compile) gives a binary that dies during startup with `hipErrorIllegalAddress`,
before frame 1, reproducibly across runs. The wave32 build from the same tree,
same GPU, same Xvfb harness reaches the known line-search wedge as documented.
Under `AMD_SERIALIZE_KERNEL=3 AMD_LOG_LEVEL=4` the last dispatch before the
memory fault is `_updateVertexes` (GIPC.cu:190), the mesh re-sort just after
`_reduct_max_box` and the thrust sort; `_updateVertexes` indexes everything
through `sortIndex[idx]`, so the corruption is upstream of it. gfx90a builds
clean (`-DCMAKE_HIP_ARCHITECTURES=gfx90a`, exit 0), so this is a runtime gap and
not a build gap. The two blind wave64 fixes are individually correct (below) but
they are not sufficient, and `-mwavefrontsize64` makes this closable by the
porter rather than something to discover on gfx90a later.

**2. `MASPreconditioner.cu` does need something, and the recorded reason for
clearing it is wrong.** notes.md:137-141 says the file is wavefront-size
independent because its cross-lane traffic goes through `__shared__` and "the
`WARP_SHFL` calls are commented out upstream". They are not: `WARP_SHFL_DOWN` at
893-895 and 1054 and `WARP_BALLOT` at 1047 are all live. The concrete defect is
MASPreconditioner.cu:886-896: `if(__popc(connectMsk) == BANKSIZE)` is uniform
across a 32-vertex bank but NOT across a 64-lane wavefront, so on wave64 one bank
can take the branch while its sibling does not, and the width-32
`WARP_SHFL_DOWN` reduction at 891-895 then executes with half the wavefront
inactive and reads inactive lanes. `__shfl` lowers to `ds_bpermute`, which
returns stale register contents rather than faulting, so this is silent. Same
class as the "Intra-wave barrier divergence" entry already in
references/fault-classes.md.

### Analysis accuracy (this text goes upstream)

**3. The "above 3x3" boundary is wrong**, in the commit message, notes.md:96-100
and the skill entry. Compiling `SelfAdjointEigenSolver<Matrix<double,3,3>>`
inside a `__device__` function with hipcc fails too, at
`Tridiagonalization.h:434`. The real boundary is `computeDirect()` (which exists
only for sizes up to 3x3) versus the general `compute()`/constructor path, which
is host-only at EVERY size. That is why femEnergy.cu:1116-1117 survives: it is a
`computeDirect` call. As written the entry tells the next porter a 3x3
`SelfAdjointEigenSolver` constructor is device-safe, and it is not. Fix all
three; the skill entry matters most.

The rest of the diagnosis is confirmed independently: `EIGEN_DEVICE_FUNC` does
expand to `__host__ __device__` for hipcc exactly as for nvcc (Macros.h:479-521,
974), the errors do all trace to one call site, and the replacement was necessary
rather than a workaround for a misread.

**4. Commit e66c864 would go upstream saying something now false.** Its title is
"[ROCm] Add HIP/ROCm build support (incomplete - blocked by Eigen)" and its body
claims "NVCC has special Eigen handling that HIP/clang lacks", which 291c612
disproves. It also bullet-lists individual changes, against the commit-message
rule. Squash the two commits or rewrite this one.

**5. `jargon.py` fails.** `python3 utils/jargon.py --commits 46594e0..HEAD -C
projects/GPU_IPC/src` exits 1: "Strategy A" in e66c864. The same squash fixes it.

**6. The Test Plan asserts the frame-30 stall "is independent of this change".**
No NVIDIA baseline exists, so that is unsupported. State what is known instead.

### Code

**7. femEnergy.cu:874-878, the convergence test overflows.** `off` and `on` are
sums of squares, so for a matrix with Frobenius norm above about 1.3e154 both
become inf and `off <= eps * eps * (on + off)` is `inf <= inf`, true. The sweep
exits at sweep 0 with V = I and returns `diag(max(A_ii, 0))`, quietly laundering
a blown-up Hessian into a plausible-looking PSD matrix instead of letting the
blow-up propagate. Measured at scale 1e155: the eigenvalue error is the full
magnitude of the input. Eigen scales by the max coefficient first, so the
original did not do this. Two lines: divide by `max|a_ij|` up front and scale the
eigenvalues back.

Verified fine, so nobody re-tests it: the sweep count is hard-bounded at 32 and
the observed maximum over 23000 matrices was 13 (7 for random symmetric, 13 for
deliberately clustered +/-1 spectra); exact repeated eigenvalues, tightly
clustered eigenvalues, the zero matrix, rank-deficient input, denormal
off-diagonals and theta overflow all produce no NaN and no non-convergence. The
rotation, the eigenvector accumulation and the `V D V^T` reconstruction are
algebraically correct, and the clamp semantics match the Eigen original exactly.
The two wave64 mappings in cuda_to_hip.h are also both correct: forcing wave64 on
gfx1100 shows the shifted ballot returning the right 32-lane half for lanes 32-63
where truncation returns the lower half's bits, and on the exact
`__PCG_Solve_AX3_b` segment logic the ported mapping differs from the naive one
on 31 of 64 lanes, with the ported values being the correct ones.

**8. CMakeLists.txt:16-18, the gfx90a default is dead code.** `project(... HIP)`
initialises `CMAKE_HIP_ARCHITECTURES` from the build host's GPUs before the
`NOT DEFINED` test runs, so an unqualified configure here prints
"HIP architectures: gfx1100;gfx1100;gfx1100;gfx1100" -- this machine's GPUs, one
duplicate per device -- and never gfx90a. Set the default before `project()`, or
drop the block and its comment.

**9. CMakeLists.txt:131 hardcodes `/opt/rocm/include`.** Use
`find_package(hip REQUIRED)` and `hip::host`, or at least `$ENV{ROCM_PATH}`. An
upstream maintainer will not take an absolute path.

**10. `cmake_minimum_required` was raised 3.18 -> 3.21 for BOTH builds.** Only
the HIP branch needs 3.21. An opt-in feature should not drop existing CUDA users
on cmake 3.18-3.20.

**11. `device_launch_parameters.h` was removed from the CUDA path**, in
femEnergy.cuh, mlbvh.cuh, MASPreconditioner.cu, PCG_SOLVER.cu and gl_main.cpp.
That changes the CUDA build's include set to fix the HIP build, and the CUDA
build has never been compiled (notes.md:145-147 admits it). Restore parity with
one line: `#include "device_launch_parameters.h"` in the non-HIP branch of
cuda_to_hip.h.

**12. Comment-only churn.** The `<< <` to `<<<` reflow was applied to about 20
commented-out kernel launches (GIPC.cu, PCG_SOLVER.cu, mlbvh.cu,
MASPreconditioner.cu:1655). Dead code does not need the fix and it pads the diff
a maintainer has to read. Revert the comment-only hunks.

### On the open wedge, and what the validator should do

`ported` was the right state to set. The wedge is real, unexplained, and in a
loop that is upstream's; the stage that decides whether the evidence bar is met
is validation, and stop discipline says leave partial value rather than grind.
What is not right is the hypothesis ranking.

Reorder it. (1) Get the NVIDIA baseline, currently ranked second. Until it exists
every other hypothesis is speculation, and it settles port-versus-upstream in one
run. (2) wave64, now a demonstrated defect (finding 1) and reproducible on this
RDNA host. (3) Uninitialized device memory, but test it decisively rather than by
audit: temporarily make `hipMalloc` also `hipMemset(p, 0, n)`, then separately
0xFF, and see whether the stall frame moves or becomes reproducible. Auditing
every `cudaMalloc` for a matching `cudaMemset` is slow and easy to get wrong.

Two mechanisms tested and ruled out here, so nobody repeats them. The second-stage
reduction in `add_reduction` and `PCG_add_Reduction_*` shuffles past `warpNum`
after an early `return`, which looks like an uninitialized-register read, but a
faithful replica measured exact against a CPU sum at 13 sizes including
non-power-of-two `warpNum` (352, 416, 448, 480, 100000, 100001). The
partial-block `__syncthreads()` reached only by threads that passed
`if (idx >= numbers) return` neither hangs nor corrupts on this hardware.

## Response to review 2026-08-08

Fork history was rewritten to a single commit, b474148 on top of 46594e0.
`validated_sha` was null on both Linux archs and there is no Windows arch, so
nothing was orphaned. The tree was checked identical across the rewrite
(`git diff --exit-code <pre> HEAD`).

### wave64: what was measured, and what it does and does not show

The two wave64 fixes in `cuda_to_hip.h` stand. The reviewer verified their lane
logic on gfx1100 in wave64 mode and that result holds: `-mwavefrontsize64` on
gfx1100 produces genuine wave64 kernels (measured here: `warpSize` 64, `__ballot`
popcount 64), so the semantics being compared are faithful.

The full-application wave64 failure is a different matter, and it is **not
attributable to this port**. Two measurements:

1. A standalone program doing exactly what `sortGeometry` does -- `thrust::sequence`
   then `thrust::sort_by_key` on 38386 uint64 morton keys -- returns a valid
   permutation when compiled for gfx1100 normally and a **corrupt** one when the
   only change is `-mwavefrontsize64`: 10156 of 38386 slots out of range and
   27913 duplicates, first bad slot holding 361654435 against a bound of 38386.
   `_updateVertexes` then does `sortMapIndex[sortIndex[idx]] = idx`, which at that
   index writes about 1.4 GB past the allocation. That is the reported
   `hipErrorIllegalAddress`, arriving one kernel downstream of its cause.
2. Running the full `-mwavefrontsize64` binary makes the same point directly: it
   dies before frame 1 with the error thrown out of `thrust::exclusive_scan`
   (MASPreconditioner.cu:1376) rather than out of any kernel in this project.

So the crash is rocPRIM/rocThrust under forced wave64 on an RDNA target, a
configuration ROCm does not validate. The mechanism is a compile-time constant,
not a prebuilt binary: rocPRIM and rocThrust are header-only (`/opt/rocm/lib`
holds no library for either, and both compile with the application's own flags),
and rocPRIM takes its wavefront constant from the target ARCHITECTURE macro
rather than from the effective wavefront mode. `min_size()` in
`/opt/rocm/include/rocprim/intrinsics/arch.hpp:68-77` returns `32u` whenever
`ROCPRIM_NAVI`, so under `-mwavefrontsize64` on gfx1100 `warpSize` is 64 while
`rocprim::arch::wavefront::min_size()` is still 32, and rocPRIM sizes its shared
memory and per-warp partials for the wrong wave. (An earlier version of this
section said "the libraries an application links are built for gfx11's native
wave32", which is wrong and would send the next reader hunting a rebuilt rocPRIM
that does not exist.)

Confirming it: a mixed build with `-mwavefrontsize64` everywhere EXCEPT the three
rocThrust translation units (GIPC.cu, mlbvh.cu, MASPreconditioner.cu, given
`-mno-wavefrontsize64`) starts normally and runs to **frame 32 at 225 ms/frame**,
reaching the same line-search wedge as the wave32 build. That build runs
PCG_SOLVER.cu, femEnergy.cu, ACCD.cu, gpu_eigen_libs.cu and device_fem_data.cu at
wave64 on real hardware, including the PCG shuffle reductions and the shifted
ballot the two fixes target.

State this plainly: **this is not evidence of a port defect and not evidence
against one.** The wave64 verdict is owed to a real gfx90a or gfx942 run. The
`-mwavefrontsize64`-on-RDNA trick is useful for checking lane LOGIC and useless
for judging a whole application that links rocPRIM.

To reproduce the mixed build (the cmake fragment is not committed):

```cmake
set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/GPU_IPC/GIPC.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/GPU_IPC/mlbvh.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/GPU_IPC/MASPreconditioner.cu"
    PROPERTIES COMPILE_OPTIONS "-mno-wavefrontsize64")
```
```bash
cmake -S . -B buildmix -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_FLAGS=-mwavefrontsize64 \
      -DCMAKE_PROJECT_INCLUDE=/path/to/that/fragment.cmake
```

### MASPreconditioner.cu, finding 2

SUPERSEDED: round 2 reverted this hoist; see "Response to review 2026-08-08
(round 2)" below. What follows is the record of what round 1 did.

Fixed. `__buildMultiLevelR_optimized` now computes the 32-lane bank sum BEFORE
the `if(__popc(connectMsk) == BANKSIZE)` branch and uses it inside; banks taking
the other branch discard it. The mask is uniform across a 32-vertex bank but not
across a 64-lane wavefront, so with the reduction inside the branch the shuffles
run with half the wavefront inactive. This is arch-unified -- identical result at
wave32, at wave64 and on CUDA -- and costs the else-path banks five shuffles.

An honesty note on the mechanism, because it changes what the fix is buying. The
reviewer's stated failure mode was the shuffle "reading inactive lanes", and with
the width pinned to 32 in `cuda_to_hip.h` that does not currently happen: a
BANKSIZE group is exactly one 32-lane half of a 64-lane wavefront, and
`__shfl_down(v, d, 32)` clamps its source index inside that half. A replica of
the exact kernel logic run at wave64 on gfx1100, with bank 0 of every wavefront
taking the shuffle path and bank 1 taking the other, produced the exact 32-lane
sum in all 128 shuffle-path banks. The fix was made anyway: the old form is
correct only because of an alignment coincidence plus `ds_bpermute` semantics
for inactive lanes, and neither is something to depend on.

### The other items

3. Eigen boundary corrected in all three places: the commit message, the
   "Resolved" section above, and the `cuda-to-rocm` skill entry. Reconfirmed by
   compiling a 3x3 `SelfAdjointEigenSolver` constructor in a `__device__`
   function with hipcc -- it fails at `Tridiagonalization.h:434`, exactly as the
   reviewer reported, while `computeDirect()` on the same type compiles clean.
4, 5. The two commits are squashed into one. `jargon.py --commits
   46594e0..moat-port` is clean over the whole branch, and the new message does
   not bullet-list changes -- it names the order to read the diff in.
6. The Test Plan no longer claims the wedge is independent of this change. It
   states the frame range, that the frame varies between runs, and that no
   NVIDIA measurement exists.
7. Jacobi overflow fixed: the matrix is normalized by its largest coefficient
   before the sweeps and the eigenvalues are scaled back at the end.
8. The dead gfx90a default is gone. CMake initialises `CMAKE_HIP_ARCHITECTURES`
   from the build host inside `project(... HIP)`, so the `NOT DEFINED` test could
   never fire; the README already documents passing the flag.
9. `/opt/rocm/include` is gone. `find_package(hip REQUIRED)` plus linking
   `hip::host` supplies the include path and `__HIP_PLATFORM_AMD__` to the plain
   C++ sources, so the manual `target_compile_definitions` went too.
10. `cmake_minimum_required` is back to 3.18 at the top, raised to 3.21 only
    inside the `USE_HIP` branch that needs it.
11. `device_launch_parameters.h` is restored on the CUDA side of
    `cuda_to_hip.h`, so the CUDA build's include set is unchanged.
12. All 18 comment-only launch-syntax lines reverted (GIPC.cu 15,
    PCG_SOLVER.cu 2, MASPreconditioner.cu 1). GIPC.cu is now byte-identical to
    upstream: every change that file carried was comment churn.

### Post-fix run, gfx1100 wave32

Same harness, build @ b474148: 25 frames at 206-229 ms/frame, then the same
line-search wedge on the same vertex cluster (12866-12893). No regression from
any of the above.

## Review 2026-08-08 (round 2)

Reviewed fork sha b474148 (`moat-port`) against 46594e0, single commit. Verdict:
changes-requested. Everything from round 1 that was code or build is fixed and
verified (items 3-12 below). What remains is one change whose stated reason its
own author measured to be false, and two global skill entries that carry a
mechanism that is wrong on the facts.

No pull request was opened anywhere; findings live here.

### Blocking

**1. MASPreconditioner.cu:889-902 -- the hoist is unnecessary, and the reason
given for it in code and in the commit message is the one the porter's own
experiment refuted.**

The branch `if(__popc(connectMsk) == BANKSIZE)` IS uniform across a 32-vertex
bank, at every wavefront size. `_preparePrefixSumL0` (MASPreconditioner.cu:101-113)
is the last writer of `_fineConnectMsk` before this kernel -- `ReorderRealtime`
calls `BuildCollisionConnection` at 1510 BEFORE `PreparePrefixSumL0` at 1512 --
and it replaces each lane's mask with the transitive closure of its bank, so
`popc == BANKSIZE` implies a single component spanning the bank and every lane
in that bank holding the identical full mask. With the shuffle pinned to width
32 in cuda_to_hip.h and `laneId = threadIdx.x % BANKSIZE`, the shuffle group is
exactly the bank and all 32 of its lanes are active. Nothing reads an inactive
lane at wave32, at wave64, or on CUDA. That matches what was measured and
written in the honesty note at notes.md:389-398.

The problem is that MASPreconditioner.cu:889-894 and the corresponding commit
message paragraph still present the inactive-lane mechanism as the reason for
touching the kernel. That text is what an upstream maintainer reads as the
justification for a change to their code, and it asserts a failure mode the
author has measured does not occur.

The hoist also introduces something upstream did not do. `totalNodes = vertNum`
(MASPreconditioner.cu:1482) is not padded to a multiple of BANKSIZE -- the
shipped scene has 38386 vertices, so the last bank has 18 live lanes and 14 that
returned at line 875. Upstream never shuffled in that bank (a partial bank
cannot reach `popc == 32`, since mask bits are only ever set for real vertices);
the hoisted form does, with the full `0xffffffff` member mask in
`gipc::WARP_SHFL_DOWN` naming the returned lanes. The result is provably
discarded, so this is not a live bug -- it is an undefined-source-lane read on
both back ends bought for no benefit.

Revert to the upstream form. If it is kept instead, the comment and the commit
message have to say what is true: that the guarded form is safe because BANKSIZE
equals the pinned shuffle width and the banks are aligned to it.

**2. The same claim was promoted to the global skill and generalizes the
mistake.** `.claude/skills/cuda-to-rocm/references/fault-classes.md`, the entry
"A branch uniform over a 32-lane GROUP is not uniform over a 64-lane wavefront",
tells the next porter to hoist collectives out of cluster-uniform branches "even
where it currently measures clean", describing the working form as "a
coincidence plus inactive-lane semantics". It is not a coincidence: `BANKSIZE ==
32 == the pinned shuffle width` and the lane id is `threadIdx.x % BANKSIZE`,
which is the same invariant that makes the reduction meaningful on CUDA in the
first place. And the transformation it prescribes is exactly the one that, in
this kernel, moves a full-mask collective into threads an earlier `return`
excluded.

Rewrite it as the opposite lesson, which is the durable one: a fixed-cluster
warp reduction is wavefront-size-safe precisely when the cluster size equals the
pinned shuffle width and clusters are aligned to it, so pin the width and check
that alignment rather than restructuring the branch; and before hoisting any
collective, check whether an earlier early-return leaves a partial group.

**3. The rocPRIM attribution is right and the stated mechanism is wrong**
(notes.md:348-350 and the new `references/validation.md` section).

Reproduced independently on gfx1100 (GPU 2, ROCm 7.2): with only
`-mwavefrontsize64` added, `warpSize` is 64 while
`rocprim::arch::wavefront::min_size()` still returns **32**, and
`thrust::sort_by_key` over 38386 uint64 keys returns 4197 out-of-range slots and
33760 duplicates where the default-width build returns a clean permutation. So
the conclusion stands and is now mechanically pinned.

The cause is `/opt/rocm/include/rocprim/intrinsics/arch.hpp:68-77`: `min_size()`
returns `32u` whenever `ROCPRIM_NAVI`, i.e. rocPRIM derives its compile-time
wavefront constant from the target ARCHITECTURE macro, not from the effective
wavefront mode, so `-mwavefrontsize64` desynchronizes rocPRIM's constant from
the hardware. Nothing is prebuilt: `/opt/rocm/lib` contains no rocPRIM or
rocThrust library, both are header-only and compile with the application's own
flags. "The libraries an application links are built for gfx11's native wave32"
sends the next porter looking for a rebuilt rocPRIM that does not exist, when
the actual rule is that rocPRIM's wave size follows the target arch and never
the flag.

**4. `references/validation.md`: "Mixing is legal -- the wavefront size is a
per-kernel field in the code object, not a per-binary one" is over-broad.**
Measured here: a wave32 TU calling a `__device__` function from a
`-mwavefrontsize64` TU under `-fgpu-rdc` links and runs, but the callee's
`warpSize` is folded to 64 while the dispatch is 32. The per-source recipe is
sound only when nothing crossing the boundary uses `warpSize`,
`__AMDGCN_WAVEFRONT_SIZE__` or a cross-lane op. That happens to hold for GPU_IPC
-- ACCD.cu, femEnergy.cu, gpu_eigen_libs.cu and device_fem_data.cu contain no
`WARP_*` calls -- which is why the mixed build is trustworthy here. State the
condition with the recipe.

Findings 2-4 are corrections to skill files that exist only on this branch
(added by 3a6ca69) and are not on moat main, so fix them here; no separate PR
against main is needed.

### Minimal footprint

**5. MASPreconditioner.cu:15-22 drops `<cooperative_groups/reduce.h>` from the
CUDA include set.** Same class as round-1 finding 11, which was fixed for
`device_launch_parameters.h` and missed here. Nothing in the file uses
cooperative groups at all, so keep both of upstream's includes on the CUDA side
of the guard and add only the HIP one.

### Verified, so nobody re-checks it

- The wave64 chain (round-1 finding 1) is sound and independently reproduced;
  see finding 3 for the one correction. The mixed build does cover both
  `cuda_to_hip.h` fixes: PCG_SOLVER.cu is one of the five wave64 TUs and is the
  only file outside the three thrust TUs that uses both `WARP_SHFL_DOWN`
  (51-367, 531-1141) and `WARP_BALLOT` (526-1128). It does NOT exercise the
  MASPreconditioner change, since that TU is compiled `-mno-wavefrontsize64`;
  notes.md:356-358 does not claim otherwise.
- The Jacobi replacement is numerically sound. Re-derived: the rotation
  annihilates `a[p][q]`, `A <- R A R^T` with `V <- V R^T` preserves
  `V A V^T == A0`, so columns of `eigenvectors` are eigenvectors and the
  `sum_k V[i][k] d[k] V[j][k]` reconstruction is `V D V^T`; the clamp semantics
  match Eigen's early-out exactly, and the result is order-independent so the
  loss of Eigen's ascending sort does not matter. Re-measured against Eigen on
  the host: 20000 random symmetric 12x12, worst relative 1.2e-14, worst absolute
  3.6e-14; at scale 1e155 relative 4.5e-15 (the normalization fix works); all
  zeros, zeros containing -0.0, negative identity, 1e-160 scaling, rank-1 and a
  1e-300 negative eigenvalue all exact against the Eigen path. All-zeros takes
  `scale == 0`, skips the division, converges at sweep 0 and returns the input
  untouched.
- The two missing-return fixes are behavior-neutral: every caller of
  `_checkPTintersection` / `_checkPTintersection_fullCCD` (mlbvh.cu:1068, 1083)
  discards the value, and `__project_StabbleNHK_H_3D_makePD`'s only call site
  (femEnergy.cu:1236) discards it too.
- GIPC.cu is byte-identical to upstream (`git diff 46594e0...b474148 --
  GPU_IPC/GIPC.cu` is empty).
- Eigen `computeDirect()` versus `compute()` is stated correctly in all three
  places.
- All four build items are as described: no gfx90a default, `find_package(hip)`
  + `hip::host` with no install prefix named, 3.18 at the top with 3.21 only
  inside `if(USE_HIP)`, `device_launch_parameters.h` restored on the CUDA side
  of cuda_to_hip.h (and reaching MASPreconditioner.cu and PCG_SOLVER.cu
  transitively via cuda_tools.h / PCG_SOLVER.cuh).
- `jargon.py` is clean in both the `--commits` and the `--diff` form over
  46594e0..moat-port.
- Commit hygiene: title `[ROCm] Add a HIP build path for AMD GPUs` (40 chars),
  Claude named, no trailers, ASCII, no em-dash, no internal references, no added
  copyright or author lines, fork tree clean and pushed.
- The project's pervasive `threadIdx.x % 32` / `>> 5` grouping is not the
  hardcoded-32 fault class: those are logical cluster sizes, and pinning the
  compat shuffles to width 32 is what makes them arch-unified. No textures,
  surfaces or pitched allocations exist in this codebase.

## Response to review 2026-08-08 (round 2)

Fork history rewritten once more: 3798cb2 on top of 46594e0, still a single
commit. `validated_sha` was null on both Linux archs, so nothing was orphaned.
The only tree change against b474148 is MASPreconditioner.cu.

### 1. MASPreconditioner.cu reverted to upstream's form

Done. `__buildMultiLevelR_optimized` is byte-identical to upstream again: the
bank reduction is back inside `if(__popc(connectMsk) == BANKSIZE)`, operating on
`r` rather than a hoisted `bankSum`, and the comment that asserted the
inactive-lane mechanism is gone. The corresponding paragraph in the commit
message now says what is true and much shorter: BANKSIZE is 32 and a bank's lane
index is `threadIdx.x % BANKSIZE`, so with the shuffle width pinned to 32 in
cuda_to_hip.h the shuffle group IS the bank on either wavefront size, and no
kernel needed restructuring.

The reviewer's second reason is the one that settles it independently of the
mechanism argument. `totalNodes = vertNum` (MASPreconditioner.cu:1482) is not
padded to BANKSIZE, so the shipped scene's last bank has 18 live lanes and 14
that returned at line 875. The guarded form never shuffles there (a partial bank
cannot reach `popc == 32`); the hoisted form did, with a full `0xffffffff`
member mask naming returned lanes. The value was discarded, so it was not a live
bug, but it is an undefined source-lane read on both back ends bought for
nothing.

### 3. rocPRIM mechanism corrected

Corrected in the wave64 section above and in the skill's `validation.md`. The
statement that "the libraries an application links are built for gfx11's native
wave32" was wrong: there are no such libraries. Verified here --
`/opt/rocm/lib` contains no rocPRIM or rocThrust binary, both are header-only,
and `min_size()` at
`/opt/rocm/include/rocprim/intrinsics/arch.hpp:68-77` returns `32u` under
`ROCPRIM_NAVI` from the target arch macro alone, with no reference to the
wavefront mode. So `-mwavefrontsize64` on gfx1100 desynchronizes rocPRIM's
compile-time wave constant (32) from `warpSize` (64) and its algorithms lay out
shared memory and per-warp partials for the wrong wave. The conclusion -- the
crash is not a port defect -- is unchanged and was reproduced independently by
the reviewer (4197 out-of-range slots, 33760 duplicates).

### 2, 4. Skill entries corrected in place

Both entries added by 3a6ca69 were rewritten on this branch rather than lifted
to main, so the reviewer sees them with the work that produced them and they
reach other projects when this branch's own PR merges.

- `references/fault-classes.md`: the "hoist the collective" entry now teaches
  the opposite and narrower rule -- a fixed-cluster warp reduction is
  wavefront-safe exactly when the cluster size equals the pinned shuffle width
  and the clusters are aligned to it, so verify that invariant and pin the width
  instead of restructuring the branch; and before hoisting any collective, check
  whether an earlier early-return leaves a partial group, with GPU_IPC's
  unpadded last bank as the worked example.
- `references/validation.md`: carries the corrected rocPRIM mechanism above; no
  longer claims mixing is unconditionally legal (it states the condition -- the
  per-source recipe is sound only when nothing reachable across the TU boundary
  uses `warpSize`, `__AMDGCN_WAVEFRONT_SIZE__` or a cross-lane op, which is why
  it held here); and states what the mixed build does and does not cover.
  PCG_SOLVER.cu is one of the five wave64 TUs and the only file outside the
  three rocThrust TUs using both `WARP_SHFL_DOWN` and `WARP_BALLOT`, so the run
  does exercise both cuda_to_hip.h lane fixes on 64-lane hardware; it does NOT
  cover MASPreconditioner, which was compiled `-mno-wavefrontsize64`.

### 5. MASPreconditioner.cu include set

Fixed. All three of upstream's cooperative-groups includes are back on the CUDA
side of the guard (`<cooperative_groups.h>`, `<cooperative_groups/reduce.h>` and
the duplicate quoted form); the HIP side has only
`<hip/hip_cooperative_groups.h>`, which is needed because `using namespace
cooperative_groups;` at line 24 requires the namespace to exist. The CUDA
include set for this file is now unchanged from upstream apart from
`device_launch_parameters.h`, which reaches it transitively through
cuda_tools.h -> cuda_to_hip.h as the reviewer confirmed.

### Post-revert run, gfx1100 wave32, GPU 2

Same harness, build @ 3798cb2: 36 frames at 205-238 ms/frame, then the same
line-search wedge on the same vertex cluster (12866-12893). No HIP error, no
NaN. No regression from the revert; the wedge frame continues to vary between
runs of the same binary (25, 26, 32, 32, 36 so far), which is expected from the
float `atomicAdd` accumulation order.

## Review 2026-08-08 (round 3)

Reviewed fork sha 3798cb2 (`moat-port`) against 46594e0, single commit. Verdict:
review-passed, no findings. No pull request was opened anywhere.

Narrow round. `git diff b474148 HEAD` is one file, 13 insertions / 18 deletions
in MASPreconditioner.cu, plus the commit-message rewrite; everything round 2
recorded as clean was left alone.

### The revert is genuinely byte-identical

`git diff origin/main...HEAD -- GPU_IPC/MASPreconditioner.cu` is 7 changed lines
and all 7 are includes, which is a stronger statement than inspecting the
function: `__buildMultiLevelR_optimized` cannot differ from upstream by a byte.
The hoisting comment is gone with it.

The include split is right on both paths. The `#else` arm carries all three of
upstream's cooperative-groups includes in upstream's order
(MASPreconditioner.cu:20-22); the HIP arm carries only
`<hip/hip_cooperative_groups.h>` (line 19), which is what `using namespace
cooperative_groups;` at line 26 needs, and nothing in the file names a cg symbol
so the missing `reduce.h` on the HIP side costs nothing. Dropping upstream's
`"device_launch_parameters.h"` is safe on CUDA: cuda_tools.h -> cuda_to_hip.h:78
supplies the angled form.

The commit message's new one-sentence reason is the true one and I confirmed
both halves at the source. `BANKSIZE 32` (MASPreconditioner.cu:43) and `laneId =
threadIdx.x % BANKSIZE` (line 883), and with the width pinned to 32 by
cuda_to_hip.h:70-73 a lane cannot read outside its own 32-aligned group -- HIP's
`__shfl_down` clamps the index to `self` when `(self & (width-1)) + delta >=
width` (amd_warp_functions.h:375-381). Measured rather than argued: a two-TU
`-fgpu-rdc` program, one TU `-mwavefrontsize64` and one `-mno-wavefrontsize64`,
sharing a header-defined `__forceinline__` wrapper around `__shfl_down(v, d,
32)`, returns the same 32-lane sum 496 from both TUs while `warpSize` reads 64
and 32 respectively (gfx1100, GPU 2). That also settles the mixed-build
soundness question the other way round from how validation.md states it: the
shared cross-lane wrapper is not deduped across the wavefront-mode boundary, so
mlbvh.cu and MASPreconditioner.cu keep their own wave32 copies.

### The two skill entries

Both hold up. Every claim was checked against the source it describes.

`references/fault-classes.md`, the fixed-cluster rule: the invariant is stated
correctly and the `self & ~(width-1)` group-base term is really in the HIP
headers (`__shfl` line 144, `__shfl_up` line 260, `__shfl_xor` line 494;
`__shfl_down` expresses the same containment as a different comparison). The
worked example is exact -- `totalNodes = vertNum` at MASPreconditioner.cu:1477,
38386 vertices, so the last bank has 18 live lanes and 14 that returned at line
875. "A partial bank cannot reach `popc == BANKSIZE`" is not an assumption: both
writers of the mask set bits only within a bank and only for real vertex ids
(`_buildCML0` lines 64-68, `BuildCollisionConnection` lines 1290-1293 and
1331-1334), so 18 is the ceiling there. Round 2's separate point that
`_preparePrefixSumL0` (lines 101-115) replaces each mask with its bank's
transitive closure is what makes the predicate bank-uniform, so the entry's
"predicate uniform over the cluster" is satisfied here rather than assumed.

`references/validation.md`, the rocPRIM mechanism: the quoted `min_size()` is
verbatim from `/opt/rocm/include/rocprim/intrinsics/arch.hpp:68-77` on this host
(ROCm 7.2.3), and `/opt/rocm/lib` has no rocPRIM or rocThrust binary, so
header-only is right and the retracted "prebuilt for wave32" wording is
retracted where it needs to be -- validation.md tells the reader there is
nothing to rebuild, and notes.md:357-360 quotes the old sentence and says why it
was wrong, which is the form that stops a re-derivation.

The coverage claim is exact. Eight `.cu` files, three name thrust (GIPC.cu,
mlbvh.cu, MASPreconditioner.cu), leaving five at wave64; of those five only
PCG_SOLVER.cu uses `WARP_SHFL_DOWN` and `WARP_BALLOT` (ACCD.cu, femEnergy.cu,
gpu_eigen_libs.cu, device_fem_data.cu have no `WARP_*` call at all, which is
also what makes the TU-boundary condition hold). MASPreconditioner.cu carries
the other `WARP_BALLOT` site that the shifted-ballot fix exists for
(lines 1049-1051, `__brev`/`__clz` on a 32-bit mark) and it is indeed
uncovered by the mixed build.

### The post-revert run

Reading it as no regression is right, and the reasoning is available without
leaning on the frame count. The revert restores upstream source byte for byte
and the delta is confined to one kernel's control flow, where the removed work
was five shuffles on the else-path banks whose result was discarded; there is no
mechanism by which it could regress. The run corroborates rather than carries
it: 36 frames is the top of the observed range, 205-238 ms/frame overlaps the
pre-revert 206-229, and the wedge signature and locus are unchanged. A wedge
frame that varies (25, 26, 32, 32, 36) while the failing vertex cluster stays
12866-12893 is consistent -- nondeterministic float accumulation moves when the
solver reaches the scene's contact hot spot, not where it is.

### Checked and clean, so nobody re-checks it

Fork tree clean and pushed (origin/moat-port == 3798cb2), `porting` lock null,
`jargon.py --commits origin/main..HEAD` clean over the whole branch, no
non-ASCII in the changed skill files or notes, no added copyright or author
lines, no textures/surfaces/pitched allocations anywhere in the codebase, and
the four CUDA math libraries the CMake still links on the NVIDIA path are
genuinely dead (every cuBLAS/cuSOLVER reference in the tree is commented out).
`timeCost.txt` never entered any commit on any branch (`git log --all --
'*timeCost*'` is empty) and is absent from the working tree; the main checkout
has no `projects/GPU_IPC` directory.

The only hardcoded 32s the diff adds are the four width pins in cuda_to_hip.h,
which are the arch-unified form rather than the fault class. `mlbvh.cu:517,543,548`
adds `return false`/`return true` to two functions that fell off the end; the
value is ignored at both live call sites (lines 1068, 1083) and the fullCCD
variant's call sites are commented out, so this is a compile fix with no
behavior change.

Not reviewed here, by scope: the line-search wedge itself. It is the validator's
item and the NVIDIA baseline ranks first, since the commit message already says
the state has not been measured on an NVIDIA build.

## Validation 2026-08-08

linux-gfx1100 (GPU index 2, W7800, ROCm 7.2.53211), fork @ 3798cb2, first
validation on this arch.

### Build

```bash
cmake -S . -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j16
```

Configures and links clean (warnings only, all `[[nodiscard]]` on
`cudaEvent*`/`cudaMemset` return values, pre-existing pattern, not new).

### Run

Same Xvfb/xdotool harness as notes.md "Running it headless".

```bash
Xvfb :77 -screen 0 1024x1024x24 &
HIP_VISIBLE_DEVICES=2 DISPLAY=:77 LIBGL_ALWAYS_SOFTWARE=1 stdbuf -o0 ./build/gipc > run.log 2>&1 &
xdotool search --name FEM   # get window id
DISPLAY=:77 xdotool key --window <id> space
```

37 frames completed, 209-234 ms/frame. Newton converges normally each frame
(observed 2-4 iterations). No HIP runtime error, no crash, no NaN anywhere in
the log. Then the documented line-search wedge: `GIPC::lineSearch` reports
"type 0 intersection happened" for a fixed vertex cluster (this run: triangles
4276-4282 / 14706-14736) with an unboundedly increasing counter, which is
`GIPC.cu:10024`'s `while(checkInterset && isIntersected(TetMesh))` spinning
because halving alpha cannot escape an already-intersecting configuration.
Same signature as every prior round: frame varies (25/26/32/32/36/**37** now),
cluster varies, no recovery.

### The wedge: hypothesis 2 tested decisively, ruled out

The review left two live hypotheses, ranked NVIDIA-baseline first and
uninitialized device memory second. There is no NVIDIA GPU on this host, so
the baseline is not runnable here -- said plainly rather than substituted
with anything weaker. Hypothesis 2 IS testable on this hardware, so it was
tested rather than left as an audit.

Method: `cuda_to_hip.h`'s `cudaMalloc` was temporarily wrapped (throwaway
patch, reverted before completion, never committed) so every fresh device
allocation gets `hipMemset` to a controlled fill immediately after
`hipMalloc`, gated by a `GIPC_DEBUG_MALLOC_FILL` compile define so the
unmodified build is a pure passthrough:

```cpp
#if defined(GIPC_DEBUG_MALLOC_FILL)
template <typename T>
inline hipError_t gipc_debug_hipMalloc(T** ptr, size_t size)
{
    hipError_t err = hipMalloc(ptr, size);
    if (err == hipSuccess) hipMemset(*ptr, GIPC_DEBUG_MALLOC_FILL, size);
    return err;
}
#define cudaMalloc(ptr, size) gipc_debug_hipMalloc(ptr, size)
#else
#define cudaMalloc hipMalloc
#endif
```

Three builds, same source, same GPU, same harness, only
`-DGIPC_DEBUG_MALLOC_FILL=<N>` (via `CMAKE_HIP_FLAGS`) differing:

| build | fill | frames | wedge frame | cluster (tri ids) | NaN | HIP error |
|---|---|---|---|---|---|---|
| unmodified | passthrough | 37 | 37 | 4276-4282 / 14706-14736 | 0 | 0 |
| zero-fill | `hipMemset(0)` | 31 | 31 | 4373-4386 / 26002-26248 | 0 | 0 |
| 0xFF-fill | `hipMemset(0xFF)` | 32 | 32 | 11655-11704 / 17570-20563 | 0 | 0 |

0xFF is the decisive pattern: as an IEEE-754 double or float it is NaN, and as
a signed or unsigned integer index it is -1 / UINT_MAX, so a genuine
live-uninitialized read anywhere in the hot path -- exactly the "worst-case
collision buffer sized by a live counter" shape the review flagged as worth
checking -- should show up as either an immediate NaN cascade, a crash on an
out-of-range index, or a wildly different frame count. None of that happened:
0xFF-fill ran 711 recorded intersection-counter lines with zero NaN
(`grep -ic nan`) and zero HIP errors, wedged in the same narrow frame window
(31-37, inside the pre-existing 25-36 range) as the unmodified and zero-fill
builds, and every reported triangle/edge id across all three builds was a
small, valid, in-range mesh index -- never a huge nonsense number, which is
what a read of a stale, oversized, uninitialized collision buffer would look
like once its contents actually mattered. This is what a compute engine that
correctly gates every read behind a live counter looks like under adversarial
fill, not what an uninitialized-memory bug looks like.

Verdict: hypothesis 2 is TESTED and RULED OUT, joining the two mechanisms
round 2/3 already ruled out (the `add_reduction` `warpNum` over-shuffle, the
partial-block `__syncthreads()`). The frame/cluster drift across all three
fill conditions is consistent with, and only with, the already-documented
float-`atomicAdd`-reordering explanation: the fill pattern changes ULP-level
accumulation order and therefore WHEN the solver reaches the scene's contact
hot spot, not whether an uninitialized read corrupts anything.

The diagnostic patch touched nothing else, was never committed, and was
reverted with `git checkout -- GPU_IPC/cuda_to_hip.h` before this validation
was recorded; `git status --porcelain` on the fork was confirmed clean
afterward.

### Verdict on the wedge for this arch: does not block `completed`

Weighing what is and is not attributable:

- The loop itself, `GIPC.cu:10024`, is byte-identical to upstream (round 3:
  "GIPC.cu is now byte-identical to upstream: every change that file carried
  was comment churn"). It is upstream's unbounded `while` with no iteration
  cap, unconditionally vulnerable once alpha-halving reaches an
  already-intersecting configuration, and that is true on any backend.
- Three review rounds have checked the port's own kernels hard: the Jacobi
  eigensolver (exact against Eigen at 20000+ cases plus edge cases), the
  `add_reduction`/`PCG_add_Reduction_*` shuffle (exact against CPU at 13
  sizes), the partial-block `__syncthreads()`, the wave64 shuffle/ballot
  fixes (lane-exact against a naive mapping), and the `MASPreconditioner.cu`
  bank reduction (reverted to upstream's own byte-identical form). None of
  that work touches or explains the wedge.
- Hypothesis 2, the one mechanism that was both live and testable here, is
  now negative evidence, not just unaudited.
- Hypothesis 1, the NVIDIA baseline, remains genuinely unresolvable on this
  host (no NVIDIA GPU) and is stated as such rather than answered with
  something weaker.

That leaves an upstream unbounded loop, in code the port did not touch,
reached through float nondeterminism the port did not introduce, with the one
port-adjacent explanation that was actually testable here tested and refuted.
Completing this arch and recording the wedge as a known upstream limitation
(a) is the honest read of the evidence; treating it as an unattributed port
defect (b) would overclaim against what was actually measured. This is not
new evidence that the port is defect-free on every arch -- gfx90a/gfx942
still owe their own real-GPU run -- it is a decision that this specific
open question does not block THIS arch.

### CUDA no-regression gate

Not previously recorded at 3798cb2. `/opt/conda/envs/cuda-12.8/bin/nvcc`
12.8.93, host gcc 13.3.0, `-DCMAKE_CUDA_ARCHITECTURES=80` passed on the
command line. The project's CMakeLists.txt hardcodes
`set(CUDA_ARCHITECTURES 86)` unconditionally in the non-`USE_HIP` branch (not
`native`, so the `-D` not taking effect is not the ancient-arch trap this
project already targets a real modern arch, Ampere/sm_86, unconditionally);
configure logged "CUDA architectures: 86" either way.

```bash
cmake -S . -B build_cuda -DUSE_HIP=OFF -DCMAKE_CUDA_ARCHITECTURES=80 \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/opt/conda/envs/cuda-12.8/bin/nvcc
cmake --build build_cuda -j16
```

Fails at compile: 16 errors, all in `GPU_IPC/load_mesh.h`/`load_mesh.cpp`,
`error: 'uint32_t' was not declared in this scope` cascading into template
and member-access errors on `tetrahedra_obj::targetIndex` /
`::surfId2TetId` / `::surfVerts`. Built the upstream base sha (`46594e0`,
before this port's `<cuda_runtime.h>` -> `"cuda_to_hip.h"` swap in
`load_mesh.h`) with the identical toolchain, identical pinned arch, identical
flags: **byte-identical 16 errors, same file:line, same messages.** This is
pre-existing breakage in the upstream tree against nvcc 12.8 / gcc 13.3 (a
missing `<cstdint>` upstream never needed against older toolchains), not a
regression from this port -- the port's header swap
(`load_mesh.h:14`) is a superset of upstream's `#include <cuda_runtime.h>`
on the CUDA side (`cuda_to_hip.h`'s `#else` arm includes both
`<cuda_runtime.h>` and `<device_launch_parameters.h>`), so it could not have
removed the transitive `uint32_t` upstream was already relying on. Recorded
verbatim, not counted as a gate failure. Both throwaway build directories
(`build_cuda` here and upstream's) were removed afterward; nothing from this
gate reached the fork tree.

### Jargon and documentation

`python3 utils/jargon.py --commits 46594e0..HEAD -C projects/GPU_IPC/src` and
`--diff 46594e0..HEAD` both clean. The ROCm build is documented in
`README.md` alongside the CUDA build (Dependencies table, per-platform build
section), in the project's own house style; nothing to send back.

### State

Fork tree clean and pushed at 3798cb2 throughout
(`git -C projects/GPU_IPC/src status --porcelain` empty; the diagnostic patch
above was reverted, not committed; `timeCost.txt` written by the run into the
fork root was removed, not committed). linux-gfx1100 -> `completed`,
`validated_sha` = 3798cb2. gfx90a still owes its own real-GPU run; nothing
here substitutes for it.

## Validation 2026-08-09

linux-gfx90a (GPU index 0, MI250X, ROCm 7.2.53211), fork @ 3798cb2, first
validation on this arch. `HIP_VISIBLE_DEVICES=0` pinned for every command;
`rocm-smi` confirmed index 0 is gfx90a before use.

### Build

```bash
sudo apt-get install -y freeglut3-dev xdotool   # not preinstalled on this host
cmake -S . -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Configures and links clean; same warning set as the gfx1100 build (`[[nodiscard]]`
on `cudaEvent*`/`cudaMemset` return values, pre-existing pattern, not new).
`timeit.sh GPU_IPC compile` wraps the full configure+build.

### Run

Same Xvfb/xdotool headless harness as notes.md "Running it headless", with
`AMD_LOG_LEVEL=3` to capture kernel dispatch:

```bash
Xvfb :77 -screen 0 1024x1024x24 &
HIP_VISIBLE_DEVICES=0 DISPLAY=:77 LIBGL_ALWAYS_SOFTWARE=1 AMD_LOG_LEVEL=3 stdbuf -o0 ./build/gipc > run.log 2>&1 &
DISPLAY=:77 xdotool search --name FEM   # window id
DISPLAY=:77 xdotool key --window <id> space
```

Kernel dispatch confirmed directly from the `AMD_LOG_LEVEL=3` trace: `hipLaunchKernel`
entries with `ShaderName` resolving to project kernels (`_selfQuery_ee`,
`_GroundCollisionDetect`, `_computeXTilta`, `_updateSurfaces`,
`__PCG_constraintFilter`, and others across the run) -- this is real device
execution, not a CPU-only or skipped run.

`vertNum: 38386      tetraNum: 159870      faceNum: 41664` -- same default
scene as every prior round on any arch. 31 frames completed (`frame id: 1`
through `frame id: 31` in the log), Newton converging normally each frame
(observed iteration k 2-3). `Kappa: 2824.037412` throughout, matching the
gfx1100 runs exactly (same scene, same physical constant, unaffected by
backend). Zero NaN in the run log (`grep -ic nan`), zero HIP runtime errors
(only benign `hipGetLastError()` trace lines, no failing status).

Then the same documented line-search wedge as every gfx1100 round:
`GIPC::lineSearch` (`GIPC.cu:10024`) reports "type 0 intersection happened"
with a monotonically increasing counter starting at frame 32 and never
recovering -- the process was killed after the counter passed 5000 with no
change in behavior, consistent with the known unbounded `while(checkInterset
&& isIntersected(TetMesh))` loop in upstream code. Wedge frame (32) falls
inside the range already observed across gfx1100 rounds (25/26/32/32/36/37),
and the mechanism (float `atomicAdd` accumulation-order nondeterminism
changing WHEN the solver reaches the scene's contact hot spot) is
backend-independent and already fully investigated in the gfx1100 validation
above -- see "Verdict on the wedge for this arch: does not block `completed`".
That reasoning is not backend-specific (the loop is byte-identical to
upstream, hypothesis 2 was tested and ruled out on the same fill-pattern
method, hypothesis 1 remains genuinely unresolvable without an NVIDIA GPU on
either host) and applies unchanged here: this arch also has no NVIDIA GPU, so
hypothesis 1 stays unresolvable, and there is nothing gfx90a-specific to add
-- the wedge reproduces with the same signature (same loop, same counter
growth, no crash, no NaN, no divergence in scene stats or convergence
behavior) as gfx1100.

### CUDA no-regression gate

Already recorded at this head_sha (3798cb2) in the linux-gfx1100 validation
above. Per policy this gate runs once per head_sha, not once per arch --
skipped here rather than re-run.

### Jargon and documentation

`python3 utils/jargon.py --port GPU_IPC`: clean. Documentation status
unchanged since the gfx1100 validation (README.md documents the ROCm build
alongside the CUDA build); nothing new to check since head_sha is unchanged.

### State

Fork tree clean throughout this session
(`git -C projects/GPU_IPC/src status --porcelain` empty before and after;
`run.log` and `timeCost.txt` were build/run artifacts in the fork working
tree, both untracked and removed, never committed). No source or build edit
was needed for this arch. linux-gfx90a -> `completed`, `validated_sha` =
3798cb295f5eca470b46ee817aab546227c15b5b. Both Linux arches (wave32 via
gfx1100, wave64 via gfx90a) are now validated; `windows` remains the only
open gate.

## Validation attempt 2026-08-14: windows-gfx1151 (BLOCKED -- host defect)

Not a port result. The port behaves correctly on Windows as far as it was
possible to observe; the host bugchecks under GPU work. Recorded so the next
Windows host can pick this up without redoing the discovery.

### Build (clean)

```
cmake -S projects/GPU_IPC/src -B projects/GPU_IPC/src/build -G Ninja -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER=D:/Develop/TheRock/.venv/Lib/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang++.exe \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/d/vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DVCPKG_TARGET_TRIPLET=x64-windows \
  -DCMAKE_PREFIX_PATH=D:/Develop/TheRock/.venv/Lib/site-packages/_rocm_sdk_devel
ninja -C projects/GPU_IPC/src/build -j6
```

Configure 12.1 s, build 29.6 s, exit 0. Eigen/GLEW/GLUT/OpenGL via vcpkg
x64-windows. Runtime DLLs staged next to `gipc.exe` per the usual TheRock
recipe (`amdhip64_7.dll`, `amd_comgr.dll`, `hiprtc*.dll`, `rocm_kpack.dll`,
`freeglut.dll`, `glew32.dll`).

### Headless harness (Windows equivalent of the Xvfb one)

The Linux harness is `Xvfb` + `LIBGL_ALWAYS_SOFTWARE=1` + `xdotool key space`.
Windows equivalents, all three needed:

- **Xvfb / LIBGL_ALWAYS_SOFTWARE** -> Mesa llvmpipe: drop `opengl32.dll` and
  `libgallium_wgl.dll` (mesa-dist-win release-msvc) next to `gipc.exe`; app-dir
  DLL search beats System32. Verify with `(Get-Process gipc).Modules` --
  `libgallium_wgl.dll` present and no AMD ICD.
- **xdotool key space** -> `PostMessage(hwnd, WM_CHAR, 32, 0)` to the "FEM"
  window, to clear `stop = true` (`gl_main.cpp:65`).
- **stdbuf -o0** -> no equivalent. `printf` to a redirected file is block
  buffered, so `Kill()` discards the whole run log. Close the window instead
  (`WM_CLOSE`); freeglut's default `GLUT_ACTION_ON_WINDOW_CLOSE` exits the
  program and the CRT flushes.

Two traps worth recording, both cost a run here:

- Do NOT minimize the window to get it out of the way. `IPC_Solver()` is called
  from `display()`, and freeglut skips the display callback for an iconified
  window -- minimizing silently stops the simulation. Move it off-screen with
  `SetWindowPos(-4000, -4000)` instead, which keeps it "shown".
- PowerShell variable names are case-insensitive, so a loop variable `$frames`
  overwrites a `$Frames` parameter.

Harness kept at `agent_space/gipc_headless.ps1` (gitignored).

### Result: host bugcheck, no frames

`windows-gfx1151` set `blocked` -- see `status.json` for the full reason. The
host bugchecks `HYPERVISOR_ERROR 0x00020001`, attributed by `!analyze -v` to
`csagent.sys` (CrowdStrike Falcon), `FAILURE_BUCKET_ID
0x20001_28_2_csagent!unknown_function`. Six occurrences on this host, four of
them 2026-06-03, before any of this work.

What was established about the port before the host died:

- Configures, builds and links clean under clang-cl/ninja for gfx1151.
- Scene loads correctly: `vertNum: 38386  tetraNum: 159870  faceNum: 41664`,
  `surfVertNum: 20836  surfEdgesNum: 62496`, 4 BVH levels, matching the Linux
  runs' scene.
- GLEW 2.3.1 initializes; the app renders and responds to mouse rotation under
  software GL.
- Zero frames of solver were observed, so nothing is claimed about numerical
  behaviour, the wavefront-32 path, or the documented `lineSearch` wedge on
  this arch.

Crash correlation, for whoever revisits this: hardware GL (unthrottled render
loop, solver paused) killed the host in ~30 s; software GL with the solver
actually running killed it within seconds; software GL with the solver stopped
ran 46 s clean. Both GPU paths trip it, so no harness arrangement avoids the
fault -- GPU compute is the thing being validated. This needs a working
Windows host, not a different harness.

## Windows warp-intrinsic rework, committed as f9604bb (2026-08-19)

Ruled `now` by Jeff Daily and committed; the two files below are no longer
uncommitted, and the integrity gap this host carried since 2026-08-14 is closed.

The Windows session of 2026-08-14 left `GPU_IPC/cuda_to_hip.h` and
`GPU_IPC/details/device_utils.inl` modified and never committed. That is why the
"Build (clean)" recipe recorded above under the 2026-08-14 attempt says head_sha
3798cb2: the build was real, but the tree it built was 3798cb2 plus these edits,
which no commit held. Read that build result as evidence for f9604bb, not 3798cb2.

What the change does: 3798cb2 translated CUDA's warp intrinsics with function-like
macros (`#define __shfl_sync(mask, var, srcLane) __shfl(var, srcLane, 32)` and
friends, plus a `hip_ballot_warp32` helper). ROCm's own headers declare genuine
overloaded functions of those same names -- confirmed on TheRock's SDK in
`hip/amd_detail/amd_warp_sync_functions.h` and `hip/amd_detail/amd_hip_bf16.h` -- so
a function-like macro of the same name mis-parses those declarations in any
translation unit that includes them transitively, which anything using Thrust does by
way of `hip_bf16.h`. f9604bb deletes the macros and implements the identical
translation as real functions (`WARP_BALLOT`, `WARP_SHFL`, `WARP_SHFL_DOWN`) in
`details/device_utils.inl`, guarded by
`#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)` with the original CUDA
definitions preserved under `#else`.

Do NOT substitute ROCm's native `__ballot_sync`/`__shfl_*_sync` for this translation.
Theirs return a 64-bit wavefront-wide ballot and default `width` to `warpSize`, which
is 64 on CDNA; CUDA's are always 32 lanes with a 32-bit ballot. On gfx90a and gfx942
the native forms would silently change behaviour.

Why the two Linux platforms should revalidate cheaply: the pinning is unchanged --
both forms shuffle at width 32 and shift a wavefront-wide ballot down to the calling
lane's own 32-lane half -- and every call site already went through the `WARP_*`
wrappers, so dropping the unused `__shfl_up_sync`/`__shfl_xor_sync` macros changes
nothing. Verified with:

```
grep -rn "__ballot_sync\|__shfl_sync\|__shfl_down_sync\|__shfl_up_sync\|__shfl_xor_sync"   --include=*.cu --include=*.h --include=*.inl --include=*.cuh GPU_IPC
```

No matches outside the explanatory comment in `cuda_to_hip.h`. This is a plausible
`codeobj_diff` carry-forward candidate on each Linux arch rather than a full rerun,
but that is the validator's call on its own hardware, not a claim made from here.

Windows compile check at f9604bb, run on this host 2026-08-19 (the CrowdStrike
bugcheck recorded above triggers under GPU work, not compilation, so a build is
still available here):

```
ninja -C projects/GPU_IPC/src/build -j6
```

Exit 0, 12 targets rebuilt after touching both headers, 0 error lines, `gipc.exe`
relinked. This is a compile check only -- `windows-gfx1151` stays `blocked` and
nothing is claimed about runtime behaviour on this host.

## Validation 2026-08-19: linux-gfx90a revalidation at f9604bb

linux-gfx90a (GPU index 0, MI250X, ROCm/TheRock SDK at
`/opt/conda/envs/py_3.12/lib/python3.12/site-packages/_rocm_sdk_devel`,
`HIP_VISIBLE_DEVICES=0` pinned throughout). Revalidation triggered by `head_sha`
moving from the previously validated `3798cb295f5eca470b46ee817aab546227c15b5b`
to `f9604bbd1e965106ee0752b46e48f8e7828c6401` (the Windows warp-intrinsic
function rework, see the section above).

### Carry-forward attempted, not possible

`python3 utils/moatlib.py classify GPU_IPC 3798cb2... f9604bb...` returns
`class=mixed`, so a `codeobj_diff` carry-forward was attempted per the shortcut
rather than assumed clean. Built the OLD sha (3798cb2) at gfx90a in a throwaway
worktree with the same recipe as every prior round
(`cmake -S . -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
-DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)`): it **fails to
compile** on this host's current ROCm SDK. `mlbvh.cu`/`MASPreconditioner.cu`/
`GIPC.cu` all error in `hip/amd_detail/amd_hip_bf16.h` around line 735-751,
"too many arguments provided to function-like macro invocation" against
`__shfl_up_sync`/`__shfl_down_sync` -- this is exactly the macro/real-function
collision that commit f9604bb's message describes and fixes (the same defect
independently discovered on the Windows host on 2026-08-14). Since the old sha
cannot be built here, `codeobj_diff` has nothing to compare against and the
shortcut does not apply; this is a full real-GPU revalidation. Worktree removed
before proceeding, no trace left in the fork tree.

This is itself useful confirmation that the fix is necessary, not just
Windows-specific busywork: this Linux host's current TheRock-based SDK hits the
identical failure mode the Windows session found, so f9604bb is a real
portability fix, not a Windows-only cosmetic change.

### Build (head_sha f9604bb)

```bash
sudo apt-get install -y freeglut3-dev xdotool   # not preinstalled on this host
cmake -S . -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Configures and links clean, exit 0. Only warnings are the pre-existing
`[[nodiscard]]` ignored-return-value pattern on `cudaEvent*`/`cudaMemset), same
as every prior round. `timeit.sh GPU_IPC compile` wraps configure+build.

### Run

Same Xvfb/xdotool headless harness as every prior round on this arch:

```bash
Xvfb :77 -screen 0 1024x1024x24 &
HIP_VISIBLE_DEVICES=0 DISPLAY=:77 LIBGL_ALWAYS_SOFTWARE=1 AMD_LOG_LEVEL=3 stdbuf -o0 ./build/gipc > run.log 2>&1 &
DISPLAY=:77 xdotool search --name FEM   # window id
DISPLAY=:77 xdotool key --window <id> space
```

Same default scene as every prior round on any arch:
`vertNum: 38386  tetraNum: 159870  faceNum: 41664  surfVertNum: 20836
surfEdgesNum: 62496`, `Kappa: 2824.037412` (matches exactly). Newton converges
normally each frame (observed iteration k 2-8). Real device execution
confirmed directly from the `AMD_LOG_LEVEL=3` trace (`hipLaunchKernel` entries
resolving to project kernels: `_prepareHessian`, `__inverse2_P96x96`,
`__PCG_constraintFilter`, `_aggregationKernel`, and others across the run).

Every single HIP call in the run log returned `hipSuccess`
(`grep -oP "Returned \K\S+" run.log | sort | uniq -c` -> 291195 hipSuccess,
zero of anything else). Zero NaN (`grep -ic nan`). 31 frames completed, then
the same documented line-search wedge as every prior round on every arch:
`GIPC::lineSearch` (`GIPC.cu:10024`) reports "type 0 intersection happened"
with a monotonically increasing counter, matching the byte-identical-to-upstream
unbounded `while(checkInterset && isIntersected(TetMesh))` loop already
investigated and ruled non-blocking in the 2026-08-08/2026-08-09 validation
rounds above (upstream code, not port code; hypothesis 2 -- uninitialized
device memory -- tested decisively and ruled out with the fill-pattern method;
hypothesis 1 -- NVIDIA baseline -- remains unresolvable on any host in this
fleet). Wedge frame (31) falls inside the previously observed range
(25/26/32/32/36/37/31), consistent with the documented float `atomicAdd`
accumulation-order nondeterminism. Nothing new to investigate: this run adds
another data point to an already-closed question, not a new one.

### CUDA no-regression gate

Not previously recorded at f9604bb (it was recorded at the prior head_sha,
3798cb2, and this gate is per-head_sha). Re-run rather than skipped:

```bash
cmake -S . -B build_cuda -DUSE_HIP=OFF -DCMAKE_CUDA_ARCHITECTURES=80 \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/opt/conda/envs/cuda-12.8/bin/nvcc
cmake --build build_cuda -j$(nproc)
```

Fails at compile: 16 errors, byte-identical file:line and messages to the
16 errors already recorded and diffed against the upstream base at 3798cb2
(`GPU_IPC/load_mesh.h`/`load_mesh.cpp`, `uint32_t` not declared, cascading
template/member-access errors -- pre-existing upstream breakage against nvcc
12.8/gcc 13.3, a missing `<cstdint>` upstream never needed against older
toolchains). The f9604bb..3798cb2 delta (`git diff --stat`) touches only
`cuda_to_hip.h` and `device_utils.inl`, neither of which `load_mesh.h`/
`load_mesh.cpp` include, so this head move cannot have changed that outcome;
confirmed by identical error count and signature rather than assumed. Not
counted as a gate failure. Throwaway `build_cuda` directory removed
afterward; nothing from this gate reached the fork tree.

### Jargon and documentation

`python3 utils/jargon.py --port GPU_IPC`: clean. `README.md` still documents
the ROCm build alongside CUDA (unchanged by this head move -- the delta
touches only `cuda_to_hip.h` and `device_utils.inl`).

### State

Fork tree clean throughout (`git -C projects/GPU_IPC/src status --porcelain`
empty before and after; `run.log` was an untracked build artifact, removed,
never committed; the throwaway old-sha worktree was removed before this
validation, never committed). No source or build edit was needed for this
arch at f9604bb. linux-gfx90a -> `completed`, `validated_sha` =
f9604bbd1e965106ee0752b46e48f8e7828c6401.

## Validation 2026-08-20: linux-gfx1100 revalidation at f9604bb (carry-forward)

linux-gfx1100 (W7800, GPU index 0 of 3 visible on this host, ROCm 7.2.53211).
Revalidation triggered by `head_sha` moving from the previously validated
`3798cb295f5eca470b46ee817aab546227c15b5b` to
`f9604bbd1e965106ee0752b46e48f8e7828c6401` (the warp-intrinsic macro-to-function
rework recorded above).

### Classification

`python3 utils/moatlib.py classify GPU_IPC 3798cb2... f9604bb...` returns
`class=mixed` (same as the gfx90a revalidation), so the carry-forward shortcut
was attempted per the skill rather than assumed.

### Carry-forward via codeobj_diff -- succeeded on this host

Unlike the gfx90a host (TheRock SDK), this host's standard ROCm 7.2.3 SDK
builds the OLD sha (3798cb2) cleanly -- it does not hit the
macro/real-function collision in `amd_hip_bf16.h` that f9604bb fixes, because
this SDK's headers apparently do not trip the same collision (or this build
never transitively includes the colliding header path in a way that surfaces
it under this SDK/version). Built both shas in throwaway git worktrees with
the project's own recipe:

```bash
git worktree add <old_dir> 3798cb295f5eca470b46ee817aab546227c15b5b
git worktree add <new_dir> f9604bbd1e965106ee0752b46e48f8e7828c6401
cmake -S . -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Both configure and build clean, exit 0, same warning set as every prior round
(`[[nodiscard]]` on `cudaEvent*`/`cudaMemset`, pre-existing, not new).

```bash
python3 utils/codeobj_diff.py <old_dir>/build <new_dir>/build
```

Reports `verdict=indeterminate`, but the only indeterminate entries are
CMake's own compiler-detection throwaways
(`CMakeFiles/3.31.6/CMakeDetermineCompilerABI_{CXX,HIP}.bin`,
`CMakeFiles/3.31.6/CompilerId{CXX,HIP}/a.out`) -- tiny compiler-check binaries
CMake generates on every configure, unrelated to the project, where device-code
extraction fails because they carry no device code at all. Pointing the tool
directly at the actual build product instead:

```bash
python3 utils/codeobj_diff.py <old_dir>/build/gipc <new_dir>/build/gipc
```

`verdict=identical` -- exported symbols and device ISA identical (20 exports).
This is the real evidence: the one binary the project produces is
bit-for-bit-equivalent in its device code and symbol table across the head
move on this arch. Consistent with the commit's own claim ("the translation
itself is unchanged... shuffles stay pinned to width 32... shifted down to the
calling lane's own 32-lane half") and with the grep already recorded above
showing no caller uses the removed macro spellings.

Carried forward rather than re-run:

```bash
python3 utils/moatlib.py carry-forward GPU_IPC linux-gfx1100 f9604bbd1e965106ee0752b46e48f8e7828c6401 \
  binary-equiv "codeobj_diff verdict=identical on gipc binary (exported symbols + device ISA) between 3798cb2 and f9604bb; macro-to-function warp-intrinsic translation is behavior-preserving on gfx1100"
```

No GPU run was needed or performed for this arch this round; the last actual
GPU pass on gfx1100 remains the 2026-08-08 validation recorded above (37
frames, no HIP error, no NaN, documented line-search wedge).

### Jargon and documentation

`python3 utils/jargon.py --port GPU_IPC`: clean (needed a local `main` branch
pointing at `origin/main` and the local `moat-port` branch fast-forwarded to
f9604bb in this worktree's fork clone before it could resolve `main..moat-port`;
no content issue). Documentation unchanged since the last recorded check --
the head move touches only `cuda_to_hip.h` and `device_utils.inl`.

### CUDA no-regression gate

Already recorded at f9604bb during the gfx90a revalidation above (per policy
this gate runs once per head_sha). Skipped here.

### State

Fork tree clean throughout (`git -C projects/GPU_IPC/src status --porcelain`
empty; both throwaway worktrees removed with `git worktree remove --force`
before completion, never committed). No source or build edit was needed for
this arch. linux-gfx1100 -> `completed`, `validated_sha` =
f9604bbd1e965106ee0752b46e48f8e7828c6401 via carry-forward. Both Linux arches
(wave32 via gfx1100, wave64 via gfx90a) are validated at the current head;
`windows` remains the only open gate, still `blocked` on the CrowdStrike host
defect recorded above.

## Upstream sync 2026-11-20 (porter, linux-gfx1100)

Why: `pr-candidates --freshness` reported the port NOT SENDABLE AS IS; upstream
main was 13 commits (2026-07-16 to 2026-09-01) past the port's base 46594e0 and
touched 12 of the port's 19 files. Lock taken with `set-state GPU_IPC
linux-gfx1100 porting`.

### Merge

```bash
git -C projects/GPU_IPC/src remote add upstream https://github.com/KemengHuang/GPU_IPC
git -C projects/GPU_IPC/src fetch upstream
git -C projects/GPU_IPC/src merge --no-ff --no-commit upstream/main   # 8 conflicted files
```

Committed as 29e2494 `[ROCm] Merge upstream main and port its new warp and CUB
code` (parents f9604bb, 4fb6019); plain push f9604bb..29e2494; `advance-head`.
Ancestry checked before pushing: `merge-base --is-ancestor upstream/main HEAD`
and `--is-ancestor f9604bbd HEAD` both true. Note the fork's `main` mirror is
still at 95b5a26 (behind upstream 4fb6019); the local `main` in the clone was
pointed at `upstream/main` for the jargon scan only, the fork mirror was not
pushed.

Conflicts (CMakeLists.txt, PCG_SOLVER.cu, cuda_tools.h, device_utils.inl,
gpu_eigen_libs.cuh, mlbvh.cu, mlbvh.cuh, README.md) were resolved by taking
upstream's text and re-applying the port's pattern. Upstream rewrote PCG_SOLVER.cu
and mlbvh.cu substantially, so both were taken whole from upstream and the port's
edits re-applied (drop `device_launch_parameters.h`, `<< <` -> `<<<` on code
lines). README.md takes upstream's new layout with the AMD block under Linux.

Port edits dropped because upstream fixed the same thing itself:
- femEnergy.cu `return tempMat;` in `__project_StabbleNHK_H_3D_makePD`
  (upstream made the function `void`; the auto-merge had kept our return and
  the HIP build failed with -Wreturn-mismatch until it was removed).
- mlbvh.cu `return false/true` in `_checkPTintersection*` (upstream made them `void`).
- The HIP-branch `cmake_minimum_required(VERSION 3.21)` and its "CUDA users on
  3.18" comment: upstream raised the global floor to 3.24.
- CMake CUDA-branch comment edits ("unused but kept") and trailing-whitespace
  edits were reverted to upstream's text to shrink the diff.

### New upstream CUDA surface, and how each was ported

Audited with `git diff 46594e0 upstream/main -- GPU_IPC` and a grep for every
warp intrinsic, `warpSize`, `cub::`, `cuda*` identifier in the merged tree.

1. **`warpSize` in logical 32-lane reductions (wave64 bug).** `__blockReduce`
   and `__blockReduceMax2` (GIPC.cu:137-138, 206-207) and `__PCG_blockReducePartial`
   (PCG_SOLVER.cu:54-55) index groups with `threadIdx.x >> 5` / `% 32` but
   computed `warp_begin = idof + warpId * warpSize` and
   `valid_lanes = min(warpSize, ...)`. On CDNA (warpSize 64) group 1's
   `warp_begin` lands 64 elements on and `valid_lanes` can exceed 32, so lanes
   would add their own value back (width-32 shuffle returns self) or skip live
   lanes. Changed to `32`; on CUDA warpSize is 32, so identical. Upstream already
   writes `min(32, ...)` in PCG_SOLVER.cu:247.
2. **`WARP_MASK_FROM_COUNT`** (new upstream helper, `count >= warpSize`) is in the
   HIP branch of details/device_utils.inl with `count >= 32`, avoiding
   `1U << count` for 32..63 on wave64. CUDA branch is upstream's text.
3. **`__syncwarp(mask)`** (14 calls in MASPreconditioner.cu). HIP's
   `__syncwarp(MaskT)` static_asserts a 64-bit mask, so 32-bit masks do not
   compile. New `gipc::SYNC_WARP(mask)`: CUDA -> `__syncwarp(mask)`, HIP ->
   `__syncwarp()` (wavefront fence + wave_barrier). All call sites go through it.
4. **`__match_any_sync(active_mask, v)`** (PCG_SOLVER.cu:250, 276, 302, fused
   Hessian SpMV). New `gipc::WARP_MATCH_ANY(v, mask)`: CUDA -> `__match_any_sync`,
   HIP -> `__match_any(v)` shifted down to the calling lane's 32-lane half, the
   same treatment as `WARP_BALLOT`. A 12-lane row straddling lane 31/32 is
   split exactly as two CUDA warps would split it.
5. **CUB.** New cuda_cub_wrappers.cu (DeviceScan/DeviceRadixSort/DeviceReduce)
   and `cub::BlockReduce<AABB, 256>` in mlbvh.cu: hipCUB headers plus
   `namespace cub = hipcub;` under `USE_HIP`. No CMake target: linking
   `hip::hipcub` pulls `hip::device`, whose `--offload-arch` then reaches the
   gcc-compiled .cpp files ("c++: error: unrecognized command-line option
   '--offload-arch=gfx1100'"). hipCUB/rocPRIM headers sit in the same include
   root as HIP, the same way rocThrust was already used.
6. `cudaGetLastError` added to cuda_to_hip.h; `<cuda_runtime.h>` in the new
   device_buffer.h and cuda_cub_wrappers.h -> `"cuda_to_hip.h"`.
7. README ROCm floor raised from 6.0 to 7.0: `__syncwarp()`/`__match_any` come
   from HIP's warp-sync header (introduced 6.2 as an opt-in preview per the
   header comment), and only ROCm 7.x has built and run this port.

Pre-existing items checked and left alone: `__ffs`/`__popc` on 32-bit masks of
32-lane banks (unchanged semantics), the `cooperative_groups` include guard
(still unused), Eigen `qr_svd` in `SVD()` (compiled before).

### Build (gfx1100, this host)

Host: TheRock pip SDK (`/opt/python-3.11-venv/.../_rocm_sdk_devel`, hipcc
reports 7.16.26385), cmake from the venv, Ninja. freeglut headers are not
installed on this image; `libglut-dev` was extracted into the job directory
(`apt-get download libglut-dev; dpkg -x`) and its `libglut.so` pointed at the
system `libglut.so.3.12`. Command (host-specific paths in angle brackets):

```bash
cmake -S . -B build -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=<pip ROCm SDK root> \
  -DGLUT_INCLUDE_DIR=<glut>/include -DGLUT_glut_LIBRARY=<glut>/lib/x86_64-linux-gnu/libglut.so
cmake --build build -j 10
```

Exit 0; warnings only `-Wunused-result` (the pre-existing `[[nodiscard]]` pattern).

### CUDA no-regression compile at 29e2494

No conda or CUDA toolkit on this host. The pip `nvidia-cuda-nvcc-cu12` 12.8
wheel ships only ptxas, so NVIDIA's redistributable archives
(`cuda_nvcc-linux-x86_64-12.8.93`, `cuda_cudart-...-12.8.90`,
`cuda_cccl-...-12.8.90` from developer.download.nvidia.com/compute/cuda/redist)
were unpacked into the job directory. CMake's CUDA configure needs cuBLAS etc.,
which those archives lack, so each translation unit was compiled directly with
the project's CMake flags, compile only, host gcc 13:

```bash
nvcc -std=c++17 -arch=sm_80 -rdc=true --use_fast_math --extended-lambda \
  --expt-relaxed-constexpr --default-stream=per-thread -O3 \
  -DGIPC_ASSETS_DIR=... -DGIPC_OUTPUT_DIR=... -DUSE_SNK -DADAPTIVE_KAPPA \
  -IGPU_IPC -I/usr/include/eigen3 -c GPU_IPC/<file>.cu
g++ -std=c++17 ... -I<cuda>/include -c GPU_IPC/<file>.cpp
```

Port tip 29e2494 and upstream 4fb6019 (worktree, same toolchain, same flags):
all 8 `.cu` files compile in both (ACCD, GIPC, MASPreconditioner, PCG_SOLVER,
cuda_cub_wrappers, device_fem_data, femEnergy, mlbvh), and `gl_main.cpp`.
`load_mesh.cpp` fails in BOTH with 16 byte-identical errors (`uint32_t` not
declared at load_mesh.h:51, the pre-existing missing `<cstdint>` recorded at
3798cb2 and f9604bb). nvcc warning counts equal per file except femEnergy.cu
(port 18, upstream 29: the port no longer instantiates
`SelfAdjointEigenSolver`). Gate: pass (no port-only error). Device link was not
attempted (no cuBLAS/cuSOLVER/cuSPARSE/cuFFT in the archives).

### GPU run: not possible this session (host condition, not the port)

The dispatcher pinned `HIP_VISIBLE_DEVICES=3`, which this host maps to PCI
26:00.0 (`amd-smi` GPU 1, sysfs card4). That card sat at `gpu_busy_percent`
100 with no owning process visible. First another fleet job's 16-way
parallel autotune was running, but the card stayed at 100% after that process
exited. With it in that state:

- `gipc` at 29e2494 printed scene setup up to `meanMass` and then waited forever
  in `hipMemcpy` (gdb backtrace: `initScene1` -> `hipMemcpy` ->
  `hsa_signal_wait_scacquire`); `AMD_LOG_LEVEL=3` shows the only kernel queued
  before it was ROCm's own `__amd_rocclr_fillBufferUnAligned` (a `hipMemset`).
- A 15-line HIP program (hipMemset 150 KB, one 64-thread kernel,
  `hipDeviceSynchronize`) on the same pin also never returned: 18 attempts of
  50 s each from 21:51 to 22:06, plus one of 180 s. So this is the card, not
  GIPC.

The pin was kept, per the job's rules (other jobs share this host); GPUs 0 and 2
were idle but not this job's to take. No frame counts exist at 29e2494 to compare
with the f9604bb record (gfx1100 2026-08-08: 37 frames then the documented
line-search wedge; gfx90a 2026-08-19: 31 frames then the wedge). Both Linux
platforms stay `revalidate`. Next session: run the "Running it headless" harness
on a healthy gfx1100 and a gfx90a card. Upstream's 2026-08-29 "harden GPU
solver" commits may change the wedge frame, or remove the wedge.

`jargon.py --port GPU_IPC`: clean (after pointing the clone's local `main` at
`upstream/main`).

## Review 2026-11-20 (delta f9604bb..29e2494, upstream sync)

Reviewed by an independent reviewer pass (pr-review skill, inline fact-check),
then each finding re-checked by the session. Verdict: **review-passed**. No code
defects. Two wording errors in the pushed merge message of 29e2494, which may
not be amended (pushed, history is evidence); correct them in the upstream PR
body (done in "## Validation 2026-11-20: linux-gfx1100 at 29e2494

GPU: `HIP_VISIBLE_DEVICES=1` (dispatcher pin, kept) = AMD Radeon Pro W7800
48GB, gfx1100, PCI 46:00.0. Not the wedged card from earlier today (26:00.0,
still at `gpu_busy_percent` 100 at 22:18; not touched). HIP 7.16.26385
(pip ROCm SDK), AMD clang 24.0.0git.

### Card health check first

A tiny HIP program (hipMalloc + hipMemset 150 KB, one 64-thread kernel,
hipMemcpy, hipDeviceSynchronize) on the pin returned in 0.12 s with
`no error` and the expected value. The card was trusted after that.

### Build

Fresh clone of `moat-port` at 29e249433e6b. Recipe from "## Upstream sync
2026-11-20" (pip SDK as `CMAKE_PREFIX_PATH`, `libglut-dev` extracted with
`apt-get download libglut-dev; dpkg -x` into the job directory, its
`libglut.so` linked to the system `libglut.so.3.12`), Ninja, `-j 10`. Exit 0.
Warnings: 15 `-Wunused-result` and 138 `-Wunused-value`, all `[[nodiscard]]`
`hipError_t` from `cudaFree`/`cudaMemset` etc. (136 of them one line,
device_buffer.h:124, included from many TUs; 2 in cuda_cub_wrappers.cu): the
pre-existing pattern, no new class of warning.

### Run

One run, `ulimit -c 0`, the "Running it headless" harness (Xvfb on `:141`,
`LIBGL_ALWAYS_SOFTWARE=1`, `stdbuf -o0`, `xdotool key space`), run from a
scratch directory so nothing landed in the fork tree; killed after 480 s.

```bash
Xvfb :141 -screen 0 1024x1024x24 &
DISPLAY=:141 LIBGL_ALWAYS_SOFTWARE=1 stdbuf -o0 ./build/gipc > run.log 2>&1 &
DISPLAY=:141 xdotool key --window $(DISPLAY=:141 xdotool search --name FEM) space
sleep 480; kill %2 %1
```

Result:

- Scene: `vertNum: 38386 tetraNum: 159870 faceNum: 41664`, `surfVertNum:
  20836 surfEdgesNum: 62496`, `level num: 4`, `bboxDiagSize2: 2.225734`,
  identical to every earlier record.
- 90 frames completed, still advancing when killed (frame 90 finished at the
  end of the window). Reported average time cost 240 ms at frame 1 rising to
  310 ms by frame 90 (old gfx1100 record: 209-234 ms over 37 frames).
- No NaN (`grep -ic nan` 0), no HIP error, no crash, no "intersection
  happened" line at all, `boundary alpha: 1.000000` every frame.
- Newton iterations per frame: 2-3 for frames 1-22, then the contact phase
  (frame 24 at 17; peak 67 at frame 35; 8-46 thereafter, 34 at frame 90).
  Kappa starts at 2824.037412 (the old records' constant value) and adapts
  upward in the contact phase (2957.18 at frame 90).

### Compared with f9604bb (the nearest platform's record is this platform's)

At f9604bb and 3798cb2 this card wedged in `GIPC::lineSearch` between frames
25 and 37 (37 on 2026-08-08; gfx90a 31-32), printing "type 0 intersection
happened" with an unbounded counter. At 29e2494 the same scene enters contact
at about the same frame (iterations jump at frame 24) and the solver works
through it with more Newton iterations instead of wedging, past frame 90.
That fits upstream's merged commits 34fc121 "harden GPU solver", d66c0ab and
2626a18 (Hessian reserves and dynamic headroom), which are in code the port's
merge took from upstream. So the behavior change is an upstream improvement
and not a regression: the earlier "known upstream limitation" (the wedge)
does not reproduce at this head on gfx1100. Whether gfx90a also clears it is
for its own revalidation.

### Gates

- CUDA no-regression compile at 29e2494: already recorded (pass) in
  "## Upstream sync 2026-11-20"; not repeated.
- `jargon.py --port GPU_IPC`: clean (local `main` in the clone pointed at
  `upstream/main` 4fb6019 first, as the porter did; the fork's `main` mirror is
  behind and was not pushed). Upstream main still 4fb6019.
- README documents the ROCm build (Dependencies table and the `USE_HIP` build
  block). Fork tree clean (`git status --porcelain` empty).

linux-gfx1100 -> `completed`, `validated_sha` = 29e249433e6b.

## Upstream PR draft" below):

1. Merge message paragraph 2 says "the bending Hessian projection is now `void`
   upstream". The dropped `return tempMat;` was in
   `__project_StabbleNHK_H_3D_makePD` (Stable Neo-Hookean Hessian projection),
   femEnergy.cu; confirmed against `git diff 46594e0 f9604bb -- GPU_IPC/femEnergy.cu`.
2. Merge message says `WARP_MASK_FROM_COUNT` "is capped at 32 lanes on HIP for
   the same reason" (64-bit HIP masks). The actual reason: upstream's helper
   tests `count >= warpSize`, so on a 64-lane wavefront the common full-group
   call (`count == 32`, from `min(32, ...)` GIPC.cu:138/207, PCG_SOLVER.cu:55/247
   and `__mm_min(BANKSIZE, ...)` in MASPreconditioner.cu) evaluates `1U << 32`,
   undefined behaviour.

Checked and clean, so nobody re-checks it: every caller discards values read
from lanes outside its member mask (GIPC.cu:143/162/212-237, PCG_SOLVER.cu:59/74/
254/280/306, MASPreconditioner.cu:850/1013); `WARP_MATCH_ANY` splits a 12/9/6-lane
row straddling lane 31/32 exactly as two CUDA warps would; `full_cluster_mask` is
balloted before divergence; HIP `__syncwarp()` is a wavefront fence plus
wave_barrier, a superset of a 32-lane sync and deadlock-free under divergence;
all launches use block sizes that are multiples of 32; HIP's `__syncwarp(MaskT)`
and `__match_any_sync` do static_assert a 64-bit mask; no stray CUDA-only surface
in the merged tree (only unmapped `cuda*` name is the local `cudaStatus`); hipCUB
API matches for DeviceScan/DeviceRadixSort/DeviceReduce Sum/Max/Reduce and
`BlockReduce<AABB,256>::Reduce(v, op, valid_items)`; rocPRIM never reports a
zero-byte temp size, so `cub_workspace` never returns null for the second call;
CUDA arms are upstream's text; per-file port patch equals f9604bb's except the
intended edits; upstream main still 4fb6019 (`ls-remote`), both ancestry checks
pass; jargon clean; title 61 chars, AI disclosure present, no Co-Authored-By.

## Validation attempt 2026-11-20: linux-gfx1100 at 29e2494 (NOT RUN -- pinned card wedged)

No verdict recorded; linux-gfx1100 stays `revalidate` (no `completed`, no
`validation-failed`, no `set-blocked`: the fault is one card on a host with
other healthy gfx1100 cards, not the platform). Build at 29e2494 passes (see
"## Upstream sync 2026-11-20"). The pinned card (`HIP_VISIBLE_DEVICES=3` =
PCI 26:00.0) was re-tested at 22:14: `gpu_busy_percent` 100, no owning process,
a hipMemset plus one 64-thread kernel did not finish within 120 s. Carry-forward
is not applicable (the delta is a 4600-line upstream merge with new kernels).
CUDA no-regression compile at 29e2494 is already recorded (pass) in the sync
section and need not be repeated by the next validator. Jargon clean; README
documents the ROCm build in upstream's new layout.

To resume: on a healthy gfx1100 card and on gfx90a, build with the recipe in
"## Upstream sync 2026-11-20" (or the plain Build recipe on a host with a
system ROCm), run the "Running it headless" harness, and compare against the
f9604bb records (gfx1100: 37 frames then the line-search wedge; gfx90a: 31
frames then the wedge). The default scene changed upstream: `initScene1` now
loads two `bunny2.msh` tetrahedral meshes instead of the cloth scene, so the
vertex/tet counts (38386/159870) WILL differ and are not a regression signal.

## Validation 2026-11-20: linux-gfx942 at 29e2494 (first run on this arch)

linux-gfx942, AMD Instinct MI300X HF (`gfx942:sramecc+:xnack-`), one card,
`HIP_VISIBLE_DEVICES=4` as pinned by the dispatcher. TheRock pip SDK
(`/opt/python-3.11-venv/lib/python3.11/site-packages/_rocm_sdk_devel`, hipcc
7.16.26385, AMD clang 24.0.0git), cmake and Ninja from the same venv. Fork
`moat-port` @ 29e249433e6b, first validation on gfx942 (no earlier record).

### Card check before trusting it

15-line HIP program (hipMalloc 150 KB, hipMemset, one 64-thread kernel,
hipMemcpy back) on the pinned card: returned in 0.64 s, correct result
(`h[63]=63`). `amd-smi metric -u` GPU 4: GFX_ACTIVITY 0 %, all
`gpu_busy_percent` 0. The card is healthy (not the wedged-card condition seen
on the gfx1100 host).

### Build

`sudo apt-get install -y freeglut3-dev xvfb xdotool` (not on this image; an
`apt-get update` was needed first, the cached mesa index 404'd).

```bash
cmake -S . -B build -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=<pip ROCm SDK root>
cmake --build build -j 56
```

Exit 0, wrapped in `timeit.sh GPU_IPC compile`. Warnings: 153, all the
`[[nodiscard]]` ignored-return pattern (138 `-Wunused-value` from
`cudaFree`/`cudaMemset` in upstream's new device_buffer.h and
cuda_cub_wrappers.cu, 15 `-Wunused-result`). No other diagnostics.

### Run

"Running it headless" harness, one run, under `ulimit -c 0`, wrapped in
`timeit.sh GPU_IPC test`: Xvfb on a free display, `LIBGL_ALWAYS_SOFTWARE=1`,
`AMD_LOG_LEVEL=3`, `stdbuf -o0 ./build/gipc`, `xdotool key --window <FEM> space`,
killed after 300 s.

- Scene: `vertNum 38386 tetraNum 159870 faceNum 41664 surfVertNum 20836
  surfEdgesNum 62496`, 4 preconditioner levels, initial Kappa 2824.037412:
  identical to the f9604bb records on gfx90a and gfx1100.
- **833 frames completed in the 300 s window, still running when killed. No
  line-search wedge**: zero "intersection happened" lines (gfx90a at f9604bb
  stopped at frame 31 in that loop, gfx1100 at frame 37).
- Newton iterations per frame 2 to 70 (median 4; 490 of 833 frames take 4).
  Adaptive kappa rises from 2824.04 to 2988.14 over the run (the f9604bb runs
  never got past the first few kappa steps before wedging).
- Running average frame time 40-52 ms (gfx1100 at f9604bb: 209-234 ms).
- `AMD_LOG_LEVEL=3`: 20782664 HIP calls, every one `hipSuccess`, nothing else;
  12759658 `hipLaunchKernel` entries resolving to project kernels
  (`__PCG_Solve_AXALL_b2`, `__inverse2_P96x96`, `__buildMultiLevelR_optimized`,
  `_GroundCollisionDetect`, `PCG_vdv_Reduction`, ...) and to the hipCUB/rocPRIM
  reduce from the new cuda_cub_wrappers.cu (`target_arch 942`, wave64).
- `grep -ic nan` on stdout and the trace: 0. No crash, no HIP error.

The trace grew to 8.3 GB in 300 s on a disk at 99 %; it was summarized and
deleted. A later run should drop `AMD_LOG_LEVEL=3` or cap the run.

### Comparison with the f9604bb records

Nearest platform is gfx90a (wave64, the f9604bb gfx90a run on 2026-08-19): same
scene, same initial kappa, same convergence pattern for the early frames
(Newton k 2-4), no HIP errors in either. The difference is that 29e2494 no
longer wedges in `GIPC::lineSearch`; upstream's 2026-08-29 solver-hardening
commits in the merge are the likely reason (not bisected; the wedge was
upstream code in both records). This is an improvement, not a regression.

Correction to "## Validation attempt 2026-11-20" above: the default scene did
NOT change. `initScene1` at the old base 46594e0 already loads the two
`bunny2.msh` meshes (`git show 46594e0:GPU_IPC/gl_main.cpp`, lines 655-663),
and the counts at 29e2494 are byte-identical to the old records. Vertex and tet
counts are a valid comparison signal.

### Other gates

- CUDA no-regression compile at 29e2494: already recorded (pass) in
  "## Upstream sync 2026-11-20"; not repeated.
- `jargon.py --port GPU_IPC`: clean (local `main` in the clone pointed at
  upstream/main 4fb6019, which is still upstream's tip; nothing pushed).
- README documents the ROCm build next to CUDA (Dependencies table and build
  section naming gfx942).
- Fork tree clean: `git -C projects/GPU_IPC/src status --porcelain` empty;
  only the ignored `build/`.

linux-gfx942 -> `completed` at 29e249433e6b.

## Upstream PR draft

Status: the AMD GPU test sentence below names only GPUs with a completed run
at 29e2494: so far the Instinct MI300X (gfx942, 2026-11-20). gfx1100 and gfx90a
are still `revalidate` at this head; add each one to the sentence only after
its run completes at the final head, before jeff reviews it.
Correction (validator, later the same day): that is wrong. `initScene1` loads
the same two `bunny2.msh` meshes at f9604bb and at 29e2494
(`git show f9604bb:GPU_IPC/gl_main.cpp`), and the run at 29e2494 printed the
same 38386 vertices / 159870 tets. The counts are directly comparable.

## Upstream PR draft

Status: the Testing paragraph names the GPUs with a completed run at
29e2494: the Radeon PRO W7800 (gfx1100), 2026-11-20. linux-gfx90a has not
revalidated at 29e2494; add the MI250X (gfx90a) to the sentence only after it
completes at the final head.

Title:

```
Add a HIP build for AMD GPUs (ROCm)
```

Body:

~~~~
This adds an opt-in build for AMD GPUs through ROCm/HIP. The CUDA build is unchanged and remains the default; `-DUSE_HIP=ON` selects the AMD build, which compiles the same `.cu` sources as HIP.

How it works: a small header, `GPU_IPC/cuda_to_hip.h`, maps the CUDA runtime names the code uses onto HIP and, on CUDA, just includes `<cuda_runtime.h>` and `<device_launch_parameters.h>`. CMakeLists.txt gains a `USE_HIP` branch that enables the HIP language and marks the `.cu` files as HIP; the CUDA branch is your existing text. CUB calls go through hipCUB, which has the same interface, via a `cub` namespace alias under HIP.

Warp operations: a CUDA warp is always 32 lanes, while an AMD wavefront is 32 lanes on RDNA GPUs and 64 on CDNA GPUs. The `gipc::WARP_*` helpers in `details/device_utils.inl` therefore pin shuffles to 32 lanes and shift ballots and `__match_any` results down to the calling lane's 32-lane half, so every 32-lane group behaves as on CUDA on either wavefront size. Two small helpers were added beside the existing ones, `gipc::SYNC_WARP` and `gipc::WARP_MATCH_ANY`, because HIP's `__syncwarp` and `__match_any_sync` require 64-bit masks; on CUDA they call `__syncwarp(mask)` and `__match_any_sync(mask, value)` exactly as before.

Two changes touch shared code and are no-ops on CUDA. The block reductions in GIPC.cu and PCG_SOLVER.cu index 32-element groups with `threadIdx.x >> 5` but computed each group's start and live-lane count with `warpSize`; they now use 32, the size of the groups they index, which is what `warpSize` is on every NVIDIA GPU. And on a 64-lane wavefront `WARP_MASK_FROM_COUNT(32)` would evaluate `1U << 32` (undefined), so the HIP version tests `count >= 32`.

The 12x12 Hessian projection in `PDSNK` now uses a cyclic Jacobi eigensolver over plain arrays instead of `Eigen::SelfAdjointEigenSolver`, whose general `compute()` path relies on host-only matrix product kernels and cannot be instantiated in a HIP kernel. It is one source for both back ends and matched the Eigen result to 1.5e-14 relative over 20000 random symmetric matrices. The AMD build uses `-ffp-contract=on` instead of a fast-math flag, because clang's `-ffast-math` also enables finite-math-only and reassociation in double precision, which the CCD and intersection predicates cannot tolerate.

Build (Linux, ROCm 7.0 or later):

```bash
cmake -S . -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Use the architecture of your GPU (`rocminfo | grep gfx` lists it), for example gfx90a, gfx942 or gfx1100.

Testing: on an AMD Instinct MI300X (gfx942) the default scene ran headless for 833 frames in five minutes with no HIP errors and no NaN. The CUDA sources were compiled with nvcc 12.8 (sm_80) before and after this change; every `.cu` file compiles, and `load_mesh.cpp` fails identically on current main without this change (load_mesh.h uses `uint32_t` without including `<cstdint>`). On Windows the AMD build compiles and links with clang and Ninja, but it was not run on a GPU.
Testing: on a Radeon PRO W7800 (gfx1100) the default scene ran headless for 90 frames with no HIP errors and no NaN. The CUDA sources were compiled with nvcc 12.8 (sm_80) before and after this change; every `.cu` file compiles, and `load_mesh.cpp` fails identically on current main without this change (load_mesh.h uses `uint32_t` without including `<cstdint>`). On Windows the AMD build compiles and links with clang and Ninja, but it was not run on a GPU.

This work was done with the assistance of an AI coding agent and reviewed by me.
~~~~

Notes for whoever finalizes this draft: the merge commit message of 29e2494
misnames the projection ("bending") and misstates the `WARP_MASK_FROM_COUNT`
reason; the body above carries the corrected wording (see "## Review 2026-11-20").
The Windows sentence rests on the 2026-11-20 compile and link at c44ed01
(see "## Windows link fix 2026-11-20"); re-check it if the head moves again. Body checked 2026-11-20: `utils/prose.py`
clean, `utils/jargon.py -` clean; re-run both after each edit (re-checked clean 2026-11-20 after naming the MI300X).
clean, `utils/jargon.py -` clean; re-run both after any further edit.

## Windows link fix 2026-11-20 (porter, windows-gfx1151, build only)

Diagnosis: agent_space/gpu-ipc-wincompile/RESULT.md (checkup session, same day).
At 29e2494 the documented Windows AMD build compiled all 10 translation units
but failed to link: `clang++: error: invalid linker name in argument
'-fuse-ld=lld-link'`. CMake 3.29+ (`Platform/Windows-Clang.cmake`,
`CMAKE_<LANG>_USING_LINKER_DEFAULT`) puts that flag on Windows clang link lines
and the `-fgpu-rdc` HIP link rejects it. The 2026-08 Windows links only worked
because of an unrecorded hand edit of `build.ninja`.

Fix, fork commit c44ed01 on moat-port ("[ROCm] Link the Windows HIP build with
clang's lld driver name"), CMakeLists.txt, directly after
`project(gpu_ipc LANGUAGES CXX HIP)` inside the existing `if(USE_HIP)`:

```cmake
    if(WIN32)
        # CMake 3.29+ passes -fuse-ld=lld-link, which the -fgpu-rdc HIP link rejects
        set(CMAKE_HIP_USING_LINKER_DEFAULT "-fuse-ld=lld")
    endif()
```

The CUDA configure never reaches it (else branch of `USE_HIP`) and a Linux HIP
configure skips it (`WIN32` false). Placement after `project()` is honoured:
CMake reads the variable at generate time, and the generated `build.ninja`
carries `-fuse-ld=lld` once and no `lld-link`.

Verification, fresh build dir, cmake 3.31.0, ninja 1.13.2, TheRock SDK
7.14.0a20260612 clang++ (GNU driver) for CXX and HIP, vcpkg x64-windows deps,
env ROCM_PATH/HIP_PATH=<sdk root>, HIP_PLATFORM=amd,
HIP_DEVICE_LIB_PATH=<sdk root>/lib/llvm/amdgcn/bitcode, no extra -D flags,
no hand edit:

```
R=D:/Develop/TheRock/.venv/Lib/site-packages/_rocm_sdk_devel
cmake -S src -B build3 -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151   -DCMAKE_CXX_COMPILER=$R/lib/llvm/bin/clang++.exe -DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++.exe   -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=D:/Develop/vcpkg/scripts/buildsystems/vcpkg.cmake   -DVCPKG_TARGET_TRIPLET=x64-windows -DCMAKE_PREFIX_PATH=$R
ninja -C build3 -j8
```

Configure exit 0, build exit 0, 0 errors, 165 warnings (the 164 pre-existing
plus `argument unused during compilation: '--rtlib=compiler-rt'` at link),
gipc.exe 3,465,728 bytes, freeglut.dll and glew32.dll copied beside it. gipc.exe
was NOT run (Windows GPU run waived; this host bugchecks under the project's GPU
work). README untouched: its Windows section covers only the CUDA build.

Pushed 29e2494..c44ed01 to moat-port (plain push, no upstream PR open).
jargon: `--port` needs projects/GPU_IPC/src, which this worktree lacks, so the
same range was scanned by hand: `jargon.py -C <clone> --commits
origin/main..moat-port` and `--diff origin/main..moat-port`, both clean.
Delta 29e2494..c44ed01 touches only the WIN32+USE_HIP branch of CMakeLists.txt,
so it cannot change a Linux build; carry-forward and review are left to the
checkup session.

## Review 2026-11-20 (delta 29e2494..c44ed01, Windows HIP link fix)

Scope: the one commit "[ROCm] Link the Windows HIP build with clang's lld driver name" (CMakeLists.txt +4 lines). The rest of the branch was reviewed at 29e2494 (see the earlier Review 2026-11-20 section). No findings.

Checked:
- Variable and placement: CMake 3.31 `Modules/Platform/Windows-Clang.cmake:59` sets `CMAKE_${lang}_USING_LINKER_DEFAULT "-fuse-ld=lld-link"` (included by `Windows-Clang-HIP.cmake` via `__windows_compiler_clang(HIP)`) during `project()`; the linker-type machinery (CMAKE_LINKER_TYPE, added in 3.29 per `Help/release/3.29.rst`) reads `CMAKE_<LANG>_USING_LINKER_<TYPE>` at generate time, so a `set()` after `project()` in the top-level directory overrides it. Evidence: build/ (29e2494) `build.ninja` has `-fuse-ld=lld-link` and fails to link; build3/ (c44ed01, no extra flags) has `-fuse-ld=lld` once and links gipc.exe. The variable is internal (undocumented, set by the platform module), not a documented user knob; acceptable for a build-only override, noted for future CMake changes.
- Older CMake: project minimum is 3.24. On 3.24-3.28 the platform module adds no `-fuse-ld` and nothing reads the variable; an unread `set()` is silently ignored (no error, no warning without `--warn-unused-vars`). No regression.
- Isolation: inside `if(USE_HIP)` (CUDA configure takes the else branch) and `if(WIN32)` (Linux HIP unaffected). Only HIP link lines are changed; gipc links with the HIP language.
- Commit: title 63 chars with `[ROCm]` (ROCm-specific), body states cause and fix, AI disclosure in prose, no Co-Authored-By/noreply, Test Plan with literal commands in a fenced block; diff and message ASCII only; `jargon.py -C <clone> --commits/--diff 29e2494..c44ed01` both clean. Code comment is accurate and ASCII.
- Not run: gipc.exe (Windows GPU run waived; host rule). Build-only evidence is sufficient for a link-line change.

## Validation 2026-11-21: linux-gfx1100 at c44ed01 (carry-forward, binary-equiv)

Delta 29e2494..c44ed01 is the Windows HIP link fix: CMakeLists.txt +4 lines,
`if(WIN32) set(CMAKE_HIP_USING_LINKER_DEFAULT "-fuse-ld=lld") endif()` inside
`if(USE_HIP)`. `moatlib.py classify GPU_IPC 29e2494... c44ed01...`:
`class=mixed` (CMakeLists.txt token count differs). It needs the fork clone at
`projects/GPU_IPC/src`; without one it answers `class=unknown`.

Built both commits at the SAME source path (a fresh fork clone in the job
directory: checkout 29e2494, build into `build-old`; checkout c44ed01, build into
`build-new`), with the recipe from "## Upstream sync 2026-11-20": TheRock pip SDK
(hipcc 7.16.26385) as `CMAKE_PREFIX_PATH`, cmake 3.31.6, Ninja, `-j 10`,
`libglut-dev` extracted with `apt-get download; dpkg -x` and its `libglut.so`
linked to the system `libglut.so.3.12`:

```bash
cmake -S . -B <build> -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=<pip ROCm SDK root> \
  -DGLUT_INCLUDE_DIR=<glut>/include -DGLUT_glut_LIBRARY=<glut>/lib/x86_64-linux-gnu/libglut.so
cmake --build <build> -j 10
```

Both exit 0, 165 warning lines each (the pre-existing `[[nodiscard]]` pattern).

Comparison:

- `python3 utils/codeobj_diff.py build-old build-new`: `verdict=identical`
  (gipc: exported symbols + device ISA identical, 14 exports). Same on the two
  `gipc` binaries directly.
- Host side: `gipc` is byte-identical (`cmp` equal, sha256 1a48133c...50c6 for
  both), all 10 object files under `CMakeFiles/gipc.dir/GPU_IPC/` are
  byte-identical, and the generated `build.ninja` files are identical once the
  build directory name is normalized. The Linux configure never evaluates the
  new lines, as expected.

No GPU run was needed. The last real GPU pass on this platform is the 29e2494
run in "## Validation 2026-11-20: linux-gfx1100 at 29e2494" (90 frames, no HIP
error, no NaN, no wedge).

```bash
python3 utils/moatlib.py carry-forward GPU_IPC linux-gfx1100 \
  c44ed01799bbcd932c8bb3e203c174ce21197521 binary-equiv "<summary above>" --host-checked
```

Fork tree untouched (nothing pushed); the throwaway clone lives in the job
directory only.
## Validation 2026-11-21: linux-gfx942 carry-forward 29e2494 -> c44ed01 (binary-equiv)

linux-gfx942 (MI300X, `HIP_VISIBLE_DEVICES=4` as pinned), same TheRock pip SDK and
recipe as "## Validation 2026-11-20: linux-gfx942 at 29e2494" (cmake 3.31.6 and
Ninja from `/opt/python-3.11-venv`). Delta 29e2494..c44ed01 is CMakeLists.txt +4,
the `if(WIN32)` block inside `if(USE_HIP)` setting `CMAKE_HIP_USING_LINKER_DEFAULT`;
the classifier calls it `mixed`, so it was built rather than asserted.

Both shas built in ONE clone at one absolute path (checkout 29e2494 -> `build-old`,
checkout c44ed01 -> `build-new`), each with

```bash
cmake -S src -B src/build-<x> -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=<pip ROCm SDK root>
cmake --build src/build-<x> -j 56
```

Both exit 0, 153 warnings each (the known `[[nodiscard]]` pattern). Comparison:

- `python3 utils/codeobj_diff.py src/build-old src/build-new`: `verdict=identical`,
  `gipc: exported symbols + device ISA identical (14 exports)`.
- Host side: `gipc` is byte-identical, sha256
  `bc1671d20a47446a20cd5eb1f6529158f2eb50ebf48a0357b6c4a6590dcf3a0f` in both; all 10
  object files `cmp`-identical; `ninja -t commands all` identical after replacing the
  build-dir name.

No GPU run needed: the executable validated at 29e2494 is the same file. Fork tree
clean apart from the untracked build dirs. Recorded
`carry-forward GPU_IPC linux-gfx942 c44ed01 binary-equiv --host-checked`.

## Porter round 2026-11-23 (linux-gfx942)

Request from jeff: under CUDA, `PDSNK` must be upstream's Eigen code exactly
(the port has never run on an NVIDIA GPU, the Jacobi solver is unmeasured
there, and the PR says the CUDA build is unchanged); the Jacobi solver becomes
HIP-only with the smallest guard. `set-state GPU_IPC linux-gfx942 porting` took
the lock with no warning (nobody held it).

### Change: fork commit 8ee59a5 on top of 21bce04

"[ROCm] Keep Eigen's eigensolver in PDSNK for the CUDA build", femEnergy.cu only.

- The Jacobi routine is now a class, `SelfAdjointJacobiSolver<MatrixType>`,
  defined only under `#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)`
  (the guard the port uses in cuda_to_hip.h, device_utils.inl,
  MASPreconditioner.cu, cuda_cub_wrappers.cu and mlbvh.cu). It has the two
  accessors `PDSNK` reads from Eigen's solver: `eigenvalues()` sorted in
  increasing order (a selection sort, as Eigen does, because upstream's early
  return tests `eigenvalues()[0] >= 0.0` as "all non-negative") and
  `eigenvectors()` as columns. The Jacobi sweep, the normalization and the
  convergence test are the reviewed code from 21bce04, with the eigenvector
  array replaced by the member matrix.
- `PDSNK` is upstream's function again, signature included (`__device__`, not
  the port's `__device__ __host__`; nothing calls it from the host). The only
  guarded lines are the solver's declaration:
  `SelfAdjointJacobiSolver<...> eigenSolver(symMtr);` under HIP,
  `Eigen::SelfAdjointEigenSolver<...> eigenSolver(symMtr);` otherwise. The early
  return, the clamp into `D` and `symMtr = eigenSolver.eigenvectors() * D *
  eigenSolver.eigenvectors().transpose();` are upstream's text, shared.
- Sharing the reconstruction was possible because the fixed-size 12x12 Eigen
  product compiles in a HIP kernel (checked with hipcc 7.16.26385, Eigen 3.4.0,
  gfx942, `-c` and a full `-fgpu-rdc` device link), contrary to the 2026-08 note
  (corrected in place above). The 12x12 `SelfAdjointEigenSolver` constructor
  still fails in a HIP kernel on this toolchain, with the same errors as before:
  `scaleAndAddTo` (ProductEvaluators.h:361, dynamic-size blocks), `extract` /
  `extractScalarFactor` (BlasUtil.h:453/455), `applyThisOnTheLeft`
  (HouseholderSequence.h:320/322).

CUDA view check: stripping every `#if defined(USE_HIP) ||
defined(__HIP_PLATFORM_AMD__)` block (keeping `#else` parts) from the new
femEnergy.cu and diffing against `git show main:GPU_IPC/femEnergy.cu` leaves
one added blank line before `template <typename Scalar, int size>` (between
`#endif` and `PDSNK`) and nothing else: the CUDA translation unit is
upstream's token for token.

Host numeric check (g++ 13.3, -O2, the class and both PDSNK variants pasted
from the file): 20000 random symmetric 12x12 matrices (every third shifted
by +8I, every seventh scaled by 1e6; 17536 needed projection). Eigenvalues vs
Eigen: worst 4.6e-15 relative. Projected matrix, HIP-side PDSNK vs upstream's:
worst 1.1e-14 relative (Frobenius).

README and other comments: nothing describes the solver as shared, so no
change. The PR draft and squash message below are updated.

### Build (gfx942, this host)

The image no longer has Eigen, GLEW, freeglut, Xvfb or xdotool, and the
build dependencies must stay out of the image's shared paths, so they were
unpacked into the job directory: `apt-get update` and `apt-get download` with
`-o Dir::State::Lists=<job>/deps/apt/lists -o Dir::Cache=<job>/deps/apt/cache
-o Debug::NoLocking=1` (no root, nothing installed), for `libeigen3-dev
libglew-dev libglut-dev libglu1-mesa-dev libgl-dev libglx-dev libegl-dev
libgles-dev libglvnd-dev xvfb xdotool libgl1-mesa-dri libglx-mesa0` and the
dependencies `apt-get -s install` named (53 debs), each `dpkg -x` into
`<job>/deps/root`. Dangling relative `.so` symlinks there (libGLX.so ->
libGLX.so.0, ...) were repointed to the system copies in
/usr/lib/x86_64-linux-gnu.

TheRock pip SDK (`/opt/python-3.11-venv/lib/python3.11/site-packages/_rocm_sdk_devel`,
package 10.1.0, hipcc 7.16.26385, AMD clang 24.0.0git), cmake 3.31 and Ninja
from the same venv:

```bash
cmake -S . -B build -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_BUILD_TYPE=Release "-DCMAKE_PREFIX_PATH=<SDK root>;<job>/deps/root/usr"
cmake --build build -j 56
```

Exit 0 (wrapped in `timeit.sh GPU_IPC compile`), 153 warnings, the known
`[[nodiscard]]` pattern, the same count as the 2026-11-20 gfx942 record.
21bce04 built the same way at the same source path into `build-old`.
`codeobj_diff.py build-old build`: `verdict=differ` (device ISA differs), as
expected: the HIP reconstruction is now Eigen's 12x12 product instead of the
port's explicit loops, and eigenvalues are sorted.

### Headless harness change on this host

Xvfb execs `/usr/bin/xkbcomp`, which the image lacks, and dies with "Failed to
activate virtual core keyboard". Xvfb was started inside a private mount
namespace (`unshare -rm`) that bind-mounts a job-local directory over /usr/bin
(symlinks to the real /usr/bin entries through a second bind, plus the
unpacked xkbcomp), with `-xkbdir <job>/deps/root/usr/share/X11/xkb`. Only that
process sees the overlay. gipc runs outside the namespace as before:
`LIBGL_ALWAYS_SOFTWARE=1`, `AMD_LOG_LEVEL=1` (errors only; level 3 wrote 8.3 GB
in five minutes last time), `stdbuf -o0`, `xdotool key --window <FEM> space`,
killed after the window, `ulimit -c 0`, `HIP_VISIBLE_DEVICES=5` as pinned by
the dispatcher (no processes on it, `amd-smi process -g 5`).

### GPU run, 300 s each, same partition, new head vs a 21bce04 control

The pinned device is a 38-CU compute partition of an MI300X (`amd-smi static
-g 5 --asic`: NUM_COMPUTE_UNITS 38; the host now lists 63 GPU devices). The
2026-11-20 record did not note its CU count; its 40-52 ms frames fit a
whole-device run. So throughput is not comparable with that record, and
21bce04 was run on the same partition as the control.

| build | frames in 300 s | avg ms/frame (end) | Newton k per frame min/median/max | kappa end | NaN | HIP errors |
|---|---|---|---|---|---|---|
| 8ee59a5 (new) | 116 | 136.4 | 2 / 16 / 77 | 2948.53 | 0 | 0 |
| 21bce04 (control) | 129 | 138.7 | 2 / 15 / 64 | 2952.39 | 0 | 0 |

Same scene in both (38386 vertices, 159870 tets, 41664 faces, 4 levels,
initial kappa 2824.037412, identical to every earlier record). No
"intersection happened" lines, nothing on stderr apart from GLEW's status line.

The per-frame Newton counts are identical for frames 1-30 (3,2,2,3 x19,4,17,18,
8,5,5,5,5,11), then diverge from frame 31 (32 vs 31, 53 vs 51, ...). That is
roundoff: the HIP projection now sums the reconstruction in Eigen's order and
sorts the eigenpairs, so `H` differs in the last bits (1e-14 relative on the
host test), and the contact-rich frames from 31 on amplify it. The frame-time
difference is under 2 %, and the 13-frame count difference follows from the
iteration counts after frame 31. Device code under HIP is therefore not the
same as before (the request expected it would be, assuming the HIP path kept
its own reconstruction); the difference is the shared upstream reconstruction
and is numerically equivalent.

Against the 2026-11-20 gfx942 record (833 frames, Newton median 4): that
median is over 833 frames, most of them after the scene settles; these
300 s runs end at frame 116-129, inside the contact-heavy phase, in both
builds. A longer run is in "## Validation 2026-11-23" below.

### Other places where the port changes what the CUDA build compiles or does

Beyond the `cuda_to_hip.h` includes and the documented no-ops (32 in the block
reductions, `gipc::SYNC_WARP` / `gipc::WARP_MATCH_ANY`), `git diff main
moat-port` has these. None is the same kind of problem as the Jacobi solver (a
different computation on CUDA), so none was changed; jeff decides.

1. GPU_IPC/gl_main.cpp:91: the `cudaSetDevice` failure message reads "Do you
   have a CUDA/HIP-capable GPU installed?" instead of upstream's
   "CUDA-capable". The CUDA binary prints different text on that path.
   Recommendation: revert to upstream's string. It is cosmetic, but it is the
   one change that makes "the CUDA build is unchanged" literally untrue at
   run time, and an AMD user is not misled by "CUDA-capable" in a CUDA
   program's message.
2. GPU_IPC/mlbvh.cu, 11 launch sites: upstream's `<< <grid, block >> >`
   spacing is rewritten as `<<<grid, block >>>`. nvcc parses both as the same
   launch, so the CUDA object is unaffected; clang's HIP front end does not
   accept the spaced form. Recommendation: keep (it is required for HIP and is
   a token-level no-op for nvcc, which the owed nvcc compile confirms).
3. `#include "device_launch_parameters.h"` was removed from
   MASPreconditioner.cu, PCG_SOLVER.cu, gl_main.cpp and mlbvh.cuh; on CUDA
   `cuda_to_hip.h` includes it, so each file still gets it, and the host-only
   files that include `cuda_to_hip.h` (load_mesh.h, device_buffer.h, ...) now
   also see it. Declarations only. Recommendation: keep.
4. CMakeLists.txt, CUDA configure: compiled flags, targets and libraries are
   upstream's text in the `else()` branches, and the new
   `$<$<BOOL:${USE_HIP}>:USE_HIP>` definition is empty on CUDA. The configure
   log differs: a new `-- Backend: CUDA` line, and `-- CUDA standard` now
   prints before `-- C++ standard`. A `USE_HIP` cache option appears.
   Recommendation: keep; restore upstream's message order only if jeff wants
   the CUDA configure output identical too.

Checked and unchanged on CUDA: `WARP_MASK_FROM_COUNT` keeps upstream's
`warpSize` test in the CUDA branch; the cooperative-groups and CUB includes
keep upstream's text in the `#else` branches; femEnergy.cu changes nothing
else outside `PDSNK`.

### CUDA no-regression compile: owed

This host has no nvcc. The CUDA compile of 8ee59a5 (both before/after, sm_80,
as in "## Upstream sync 2026-11-20") is owed and jeff will run it on the host
that has nvcc. What it should show: femEnergy.cu compiles; the last CUDA
compile (29e2494) had `PDSNK` on the Jacobi path, so this is the first nvcc
compile of the restored Eigen path in the port tree (upstream main compiles
it today).

### Push and records

```bash
git -C projects/GPU_IPC/src push origin moat-port     # 21bce04..8ee59a5, plain push
python3 utils/moatlib.py advance-head GPU_IPC 8ee59a5c5cc69491b65819577bcf4110d9891e41
python3 utils/jargon.py --port GPU_IPC                # jargon: clean
python3 utils/prose.py <commit message>               # prose: clean
```

### Correction to the run comparison above (same day)

A third run of the same 8ee59a5 binary (started as a 1200 s run, stopped at
frame 227 when the review below requested a change) also matches frames 1-30
exactly and then diverges at frame 31 (Newton k 12, where the first 8ee59a5
run had 11 and 21bce04 had 11). Two runs of ONE binary diverge at the same
frame, so the old/new divergence from frame 31 is run-to-run nondeterminism
(the order of floating-point atomic accumulations), not the code change; the
roundoff explanation given above cannot be told apart from it. That partial
run: 227 frames, Newton k 2 / 9 / 78 (min/median/max), the last 20 frames at
5-10 iterations (the scene settling), 146 ms/frame average, no NaN, no HIP
error.

## Review 2026-11-23 (delta 21bce04..8ee59a5)

Scope: `git diff 21bce04..8ee59a5` (femEnergy.cu only), with the pr-review
skill; the code reading and the compile and host tests were done by a child
agent and each finding was re-checked against the code before recording.

Verified: with the HIP guard not taken, femEnergy.cu is upstream's file token
for token (CUDA view diff against `git show main:GPU_IPC/femEnergy.cu`: one
extra blank line, nothing else); `PDSNK` is `__device__` only as upstream; it
calls only Eigen (`SelfAdjointEigenSolver`, `setZero`, the product), all
upstream. The class plus upstream's tail compiles for gfx942 with and without
`-fgpu-rdc`. Host comparison against upstream's Eigen-based PDSNK, 20000
random symmetric matrices: 8.6e-15 relative, eigen-residual 3.6e-15,
orthogonality 1.1e-14, zero sort violations, zero early-return mismatches;
degenerate inputs (zero, repeated eigenvalues, rank-deficient, identity, x1e200,
x1e-200) agree with Eigen to 4e-15 with the same early-return decision.
Rotation formulas match Golub-Van Loan sym.schur2; the sort swaps eigenvalue
and eigenvector column together. Commit message: title 59 chars, `[ROCm]`
correct, ASCII, AI disclosure in prose, no trailer, no internal vocabulary.

### Blocking

1. femEnergy.cu:850-858 (scale loop) with the shared tail: a non-finite input
   comes back as an all-zero Hessian. An inf entry makes `scale` inf, the
   normalization turns every finite entry to 0, no rotation runs, the
   eigenvalues are `0 * inf` = NaN, and upstream's tail then builds `D = 0`
   (`NaN > 0.0` is false) and returns `I * 0 * I`. Same for a NaN on the
   diagonal of an otherwise diagonal matrix (`mag > scale` skips the NaN).
   Eigen's solver returns NaN in both cases and 21bce04 returned H unchanged;
   the class's own comment says it exists so a blown-up matrix does not come
   back plausible-looking. Fix (HIP-only, CUDA text unaffected): take the
   scale with `!(mag <= scale)` so NaN propagates, and when `scale` is not
   finite fill the eigenvalues and eigenvectors with quiet NaN and return.
2. Commit-message Test Plan: the headless GPU run is described in prose with
   no command. The follow-up commit and the squash message must give the
   literal commands (Xvfb, the gipc launch, the xdotool keypress).

### Comment

3. femEnergy.cu:835: "Jacobi needs no workspace" is not literal; the
   constructor keeps a `Scalar a[Size][Size]` copy (1152 bytes at 12x12).
   Reword to "no workspace beyond a copy of the matrix".

Observation for the record, not a finding: scratch in
`_calculate_bending_gradient_hessian` (femEnergy.cu:1416, launched from
GIPC.cu:7005) grows from 4864 to 7136 bytes per lane at gfx942 (VGPRs stay
128, occupancy 4 waves/SIMD), from upstream's `D` and the `V * D * V^T`
temporaries that sharing the tail brings in. Frame time on this partition
moved under 2 % (136.4 vs 138.7 ms).

Verdict: changes requested.

## Response to review 2026-11-23: fork commit 04c121a

"[ROCm] Propagate NaN from the HIP Jacobi solver on non-finite input",
femEnergy.cu, HIP-only lines.

1. Non-finite input. The scale is now taken with `if(mag > scale || mag !=
   mag)`, and `if(!(scale <= Eigen::NumTraits<Scalar>::highest()))` fills
   `m_eivalues` and `m_eivec` with `quiet_NaN()` and returns; upstream's tail
   then yields NaN (`NaN * 0`). The review's suggested `if(!(mag <= scale))`
   was not used: it is not sticky (a later finite `mag` satisfies
   `!(mag <= NaN)` and overwrites a NaN scale), and the host test showed the
   NaN-on-the-diagonal case still returning a finite matrix with it. The HIP
   flags have no fast-math, so `mag != mag` is not folded away.
   Host test (g++ -O2, with and without -ffp-contract=on), HIP-side PDSNK vs
   upstream's: inf on the diagonal, -inf off the diagonal, NaN on the diagonal
   of a diagonal matrix, NaN off the diagonal: all four return NaN in both;
   finite random input unchanged (20000 matrices, 1.1e-14 relative, as before).
2. Test Plan: 04c121a's message gives the literal Xvfb / gipc / xdotool
   commands, and so does the squash message below.
3. Comment: "no workspace beyond a copy of the matrix".

CUDA view of femEnergy.cu against main: still one added blank line and no
token difference. Incremental build exit 0. 300 s run on partition 5 under the
same harness: 125 frames, Newton k 2 / 15 / 64, frames 1-31 identical to every
earlier run (..., 5, 5, 5, 5, 11), 137.7 ms/frame, no NaN, no HIP error.
`jargon.py --port GPU_IPC` clean; pushed 8ee59a5..04c121a (plain push);
`advance-head GPU_IPC 04c121a1b7999282341d609d8533be02854b75bb`.
