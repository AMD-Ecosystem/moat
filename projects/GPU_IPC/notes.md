# GPU_IPC notes

## Port status

Builds, links and runs on gfx1100 (linux-gfx1100), fork `moat-port` @ 291c612.
The Eigen blocker recorded below is resolved. Not yet validated on any arch.

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

Not yet ruled out, in the order worth trying:

1. Uninitialized device memory. ROCm does not hand back zeroed pages where CUDA's
   allocator often does. GIPC allocates collision-pair and Hessian buffers sized
   for the worst case (`maxCollisionPairsNum_CCD: 2499840`) and relies on counters
   to say how much is live; any read past the live count sees garbage on ROCm and
   zeros on CUDA. Audit `device_fem_data.cu` and `GIPC.cu` for a `cudaMalloc` with
   no matching `cudaMemset`.
2. Whether upstream does the same thing on an NVIDIA GPU at some frame. No NVIDIA
   hardware was available in this session, so there is no baseline. This is the
   single most valuable next datapoint and it is cheap for anyone who has a card.
3. Whether the scene is simply meant to be stopped before this point. The demo has
   a `totalFrames` exit that upstream left commented out (gl_main.cpp:943-946).

## Resolved: the Eigen device eigensolver blocker

The previous session stopped here, and the diagnosis in it was half right. Eigen
does support HIP device compilation -- `EIGEN_DEVICE_FUNC` expands to
`__host__ __device__` for hipcc exactly as it does for nvcc, and Eigen 3.4's
`SelfAdjointEigenSolver` header carries 23 of those annotations. Core arithmetic,
`Map`, `.cross()`, small products and the direct 2x2/3x3 solver path all compile
fine. What does not compile is the reduction path taken ABOVE 3x3: it goes through
a Householder sequence whose inner products are Eigen's general matrix-matrix
product kernels (`scaleAndAddTo`, `applyThisOnTheLeft`, `BlasUtil::extract`,
`extractScalarFactor`), and those are host-only.

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

MASPreconditioner.cu was reviewed and needs nothing: `BANKSIZE 32` is a logical
cluster size, its lane indices come from `idx % BANKSIZE` rather than a hardware
lane id, and its cross-lane communication goes through `__shared__` arrays (the
`WARP_SHFL` calls are commented out upstream). It is wavefront-size independent
as written.

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
