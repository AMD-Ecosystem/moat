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
