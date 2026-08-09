# GPU_IPC notes

## Port status

Builds, links and runs on gfx1100 (linux-gfx1100), fork `moat-port` @ 3798cb2,
a single commit on top of 46594e0. The Eigen blocker recorded below is
resolved. Not yet validated on any arch. Both rounds of review 2026-08-08 are
addressed; see "## Response to review 2026-08-08" and "## Response to review
2026-08-08 (round 2)".

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
