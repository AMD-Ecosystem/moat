# STRUMPACK port plan

## Classification: VALIDATE-AND-DOCUMENT existing ROCm support (not a fresh port)

STRUMPACK (pghysels/STRUMPACK, LBNL, BSD, v8.0.0) is a GPU sparse DIRECT solver
(multifrontal sparse LU + HSS/BLR low-rank). It ALREADY ships a mature HIP/ROCm
backend (runs on Frontier/OLCF). This is the ROCm substitute for NVIDIA cuDSS,
which is proprietary/closed-source and cannot be ported. STRUMPACK is RXMesh's
pending solver-path dependency (`RXMesh/status.json depends_on: ["STRUMPACK"]`).

Like fused-ssim, the MOAT work is: build the existing HIP backend on gfx90a,
GPU-validate a sparse factor+solve (residual at tolerance, matches a CPU
reference, deterministic, confirmed on-GPU), fix any bitrot, and -- crucially --
write the `## Install as a dependency` section so RXMesh can consume it.

## Existing ROCm support (from upstream CMake)

- `CMakeLists.txt:8` `option(STRUMPACK_USE_HIP ...)`.
- `CMakeLists.txt:140-154` HIP path: `enable_language(HIP)` + `find_package` of
  hip, hipblas, hipsparse, rocsolver, rocblas, rocprim, rocthrust (all ship with
  ROCm 7.2.1 here).
- `CMakeLists.txt:702-704` links `roc::hipblas roc::rocblas roc::rocsolver
  roc::hipsparse roc::rocthrust`.
- HIP sources: `src/dense/HIPWrapper.{hip,cpp,hpp}`, `src/sparse/fronts/FrontHIP.hip`,
  plus the backend-agnostic `FrontGPU.cpp` gated on CUDA|HIP|SYCL
  (`src/sparse/fronts/CMakeLists.txt:34`). `.hip` files compile via the HIP
  language; no hand symbol-renaming, no compat header -- this is a designed-in
  backend.
- Runtime: built with GPU, `use_gpu_` defaults true (`StrumpackOptions.hpp:1335`);
  factor() runs on the GPU automatically. `enable_gpu()/disable_gpu()` toggle it,
  so the SAME binary gives a CPU reference solve.

## Build strategy: sequential (non-MPI), single-GPU

MPI default ON pulls ParMetis/PTScotch/SLATE/ButterflyPACK; that is heavy and
not needed to validate the single-GPU sparse direct solver. Per the task and
`CMakeLists.txt:74-95` (MPI OFF auto-disables those TPLs), scope to a sequential
build:

`-DSTRUMPACK_USE_MPI=OFF -DSTRUMPACK_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a`
plus disabling the still-ON optional TPLs (SLATE/ParMetis/Scotch/PTScotch/BPACK/ZFP)
so only the required METIS + BLAS/LAPACK + ROCm libs are needed.

## Dependencies

- Metis (serial): REQUIRED (`CMakeLists.txt:442 find_package(METIS REQUIRED)`).
  Install `libmetis-dev` via apt (system path).
- BLAS/LAPACK: reference libblas/liblapack present; install OpenBLAS for a real
  BLAS.
- rocBLAS/rocSOLVER/rocSPARSE/rocPRIM/rocThrust: ship with ROCm 7.2.1.
- gfortran, MPI compilers present (MPI not used in the sequential build).

## Validation (HIP_VISIBLE_DEVICES=1, gfx90a)

- Run `examples/sparse/testPoisson3d` (ships a 3D Poisson regular-grid generator
  + `max_scaled_residual`). GPU solve vs CPU-reference solve (disable_gpu) on the
  same matrix.
- Assert: componentwise scaled residual at solver tolerance (converged);
  ||x_gpu - x_cpu|| ~ machine eps; bitwise determinism across repeated GPU runs.
- Confirm on-GPU: STRUMPACK GPU timers / AMD_LOG_LEVEL=3 kernel dispatches.
- Also run a Matrix-Market test (`testMMdouble`) on a shipped/generated matrix
  as a second system.

## Deliverable

Expect ZERO source changes (designed-in HIP backend on a current ROCm). Record
that in notes.md. Write the `## Install as a dependency` section (configure +
build + install commands, install prefix, and what RXMesh sets:
`find_package(STRUMPACK)` + `-DCMAKE_PREFIX_PATH=...`). Do NOT fork/push unless
source changes are needed. Append any generalizable lesson to PORTING_GUIDE.md.
