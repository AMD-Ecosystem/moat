# STRUMPACK notes

## Target type: validate existing ROCm support (NVIDIA cuDSS substitute)

STRUMPACK (LBNL, BSD) is a GPU sparse DIRECT solver (multifrontal sparse LU + HSS/BLR low-rank) that ALREADY supports HIP/ROCm and runs on AMD GPUs (Frontier at OLCF); a 2025 paper reports its single-GPU factorization is ~1.9x faster than NVIDIA cuDSS. So this is a VALIDATE-AND-DOCUMENT target (like fused-ssim), NOT a fresh CUDA->HIP port: build for gfx90a, confirm a sparse factorize+solve runs correctly and deterministically vs a CPU reference, and document it as the ROCm substitute for the proprietary/closed-source NVIDIA cuDSS (which cannot be ported). It is the (pending) dependency for RXMesh's solver/autodiff path.

## Validated: linux-gfx90a (MI250X, CDNA2, wave64, ROCm 7.2.1)

STRUMPACK v8.0.0 (upstream `e08ec96`). Built the existing HIP backend as a sequential (non-MPI) single-GPU library and GPU-validated a sparse multifrontal LU factor+solve. The factorization runs on the GPU, the solution matches a CPU-reference solve to machine eps, and it is bitwise deterministic across runs. TWO small CMake changes were needed to fix bitrot against ROCm 7.2.1's CMake target layout (no kernel/algorithm source change). Details below.

## Existing ROCm support (assessment)

A designed-in HIP backend, not a CUDA-only project:
- `CMakeLists.txt:8` `option(STRUMPACK_USE_HIP "Use HIP for AMD or NVIDIA GPU acceleration" OFF)`.
- `CMakeLists.txt:140-154` the HIP path: `enable_language(HIP)` + `find_package` of hip, hipblas, hipsparse, rocsolver, rocblas, rocprim, rocthrust (all ship with ROCm 7.2.1).
- `CMakeLists.txt:702-704` links the roc:: math libs.
- HIP sources: `src/dense/HIPWrapper.{hip,cpp,hpp}`, `src/sparse/fronts/FrontHIP.hip` (compiled via the HIP language by file extension), plus backend-agnostic `src/sparse/fronts/FrontGPU.cpp` gated on CUDA|HIP|SYCL (`src/sparse/fronts/CMakeLists.txt:34`). No compat header, no hand symbol-renaming.
- Runtime: when built with GPU, `StrumpackOptions.hpp:1335` defaults `use_gpu_ = true`, so `factor()` runs on the GPU automatically; `enable_gpu()`/`disable_gpu()` (`StrumpackOptions.hpp:745,750`) toggle it, which gives a CPU reference solve from the same binary.

## Source changes needed (bitrot vs ROCm 7.2.1) -- 2 CMake files, no kernel change

Both are NVIDIA-safe (gated on `hipblas_FOUND` / `STRUMPACK_USE_HIP`) and minimal.

1. **`CMakeLists.txt:702-704` -- drop the `roc::rocthrust` link target.**
   `roc::rocthrust` -> `roc::rocprim_hip` -> `hip::device`, and `hip::device`'s
   `INTERFACE_COMPILE_OPTIONS` carry `-x hip` (hip-config-amd.cmake), plus
   `--offload-arch=gfx90a`. Because strumpack links the roc libs PUBLIC, those
   flags propagate to EVERY CXX source of the strumpack target, so g++ is handed
   `-x hip --offload-arch=gfx90a` on the host .cpp TUs and dies with
   `c++: error: unrecognized command-line option '--offload-arch=gfx90a'`. The
   other math libs (rocblas/hipblas/hipsparse) correctly link only `hip::host`
   (link-only, no `-x hip`); only rocThrust pulls `hip::device`. rocThrust is a
   header-only INTERFACE target whose headers live on the HIP compiler's default
   include path (`/opt/rocm/include`), and only `FrontHIP.hip` (compiled as HIP)
   includes `<thrust/complex.h>`; no host .cpp/.hpp includes thrust. Fix: link
   `roc::hipblas roc::rocblas roc::rocsolver roc::hipsparse` only, and add
   rocThrust's include dir via `target_include_directories(... PRIVATE
   ${_rocthrust_inc})` (a plain -I, no -x hip). This is the cupoch
   `hip::device`-propagation fault class (PORTING_GUIDE).

2. **`cmake/strumpack-config.cmake.in` -- add a HIP `find_dependency` block.**
   The template has a CUDA block (`if(@STRUMPACK_USE_CUDA@) enable_language(CUDA);
   find_dependency(CUDAToolkit)`) but NO HIP equivalent. The installed
   `STRUMPACK::strumpack` target's `INTERFACE_LINK_LIBRARIES` references
   `roc::hipblas;roc::rocblas;roc::rocsolver;roc::hipsparse`, so a consumer's
   `find_package(STRUMPACK)` fails with `target "roc::hipblas" not found` /
   `cannot find -lroc::hipblas` unless those targets are imported. Fix: mirror the
   CUDA block with `if(@STRUMPACK_USE_HIP@) find_dependency(hip/hipblas/hipsparse/
   rocsolver/rocblas)`. Without this RXMesh cannot consume the HIP build via
   `find_package`. (Note: thanks to fix #1 the consumer no longer inherits
   rocThrust/hip::device, so a downstream that links `STRUMPACK::strumpack` also
   does not get the `-x hip` host-CXX trap.)

The full diff is 2 files, +20/-1 lines. These are real fixes (the HIP build does
not compile, and the install is not consumable, without them on ROCm 7.2.1), so
the parent should deliver them upstream. No source/kernel/algorithm edits.

## Dependencies installed

- `libmetis-dev` (Metis 5.1.0, serial) -- REQUIRED (`CMakeLists.txt:442
  find_package(METIS REQUIRED)`); apt, system path `/usr` (header `/usr/include/metis.h`).
- `libopenblas-dev` (0.3.26) -- real BLAS/LAPACK; apt.
- rocBLAS/rocSOLVER/rocSPARSE/hipBLAS/hipSPARSE/rocPRIM/rocThrust -- ship with ROCm 7.2.1.
- gfortran, OpenMP (gcc), cmake 3.x present. MPI present but not used (sequential build).

## Build (ROCm/gfx90a, sequential single-GPU)

MPI defaults ON and pulls ParMetis/PTScotch/SLATE/ButterflyPACK (heavy, and not
needed to validate the single-GPU sparse direct solver). `CMakeLists.txt:74-95`:
turning MPI OFF auto-disables those TPLs. The remaining still-ON optional TPLs
(SLATE/ParMetis/Scotch/PTScotch/BPACK/ZFP) are disabled explicitly so only the
required METIS + BLAS/LAPACK + ROCm libs are needed.

```
SRC=<strumpack source>
cmake -S $SRC -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=<prefix> \
  -DSTRUMPACK_USE_MPI=OFF \
  -DSTRUMPACK_USE_OPENMP=ON \
  -DSTRUMPACK_USE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DTPL_ENABLE_SLATE=OFF -DTPL_ENABLE_PARMETIS=OFF \
  -DTPL_ENABLE_SCOTCH=OFF -DTPL_ENABLE_PTSCOTCH=OFF \
  -DTPL_ENABLE_BPACK=OFF -DTPL_ENABLE_ZFP=OFF -DTPL_ENABLE_COMBBLAS=OFF \
  -DTPL_METIS_PREFIX=/usr \
  -DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libopenblas.so \
  -DLAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/libopenblas.so
cmake --build build -j16
```

Builds `libstrumpack.a`. `StrumpackConfig.h` then has `STRUMPACK_USE_HIP` +
`STRUMPACK_USE_GPU` defined, MPI undef. The `.hip` files build as HIP objects
(`Building HIP object .../HIPWrapper.hip.o`, `FrontHIP.hip.o`); host .cpp build
with g++.

Runtime note: the host has 64 cores; OpenBLAS prints a benign "NUM_THREADS
exceeded" warning when OpenMP spawns 128 threads. Set `OPENBLAS_NUM_THREADS=16
OMP_NUM_THREADS=16` to silence it; it does not affect correctness.

## Validation (HIP_VISIBLE_DEVICES=1, gfx90a) -- ALL PASS

GPU confirmed by `AMD_LOG_LEVEL=3`: the multifrontal factorization dispatches
STRUMPACK's own device kernels -- `strumpack::gpu::{assemble_kernel,
extend_add_kernel, LU_block_kernel_batched, Schur_block_kernel_batched,
solve_block_kernel_batched}` (LU + Schur-complement + triangular solve on device;
~26k HIP dispatches at n=40). Compile alone is not the gate.

1. **Custom harness `agent_space/strumpack_validate.cpp`** -- builds a 3D Poisson
   system, solves with GPU enabled then with `disable_gpu()` (CPU reference) on the
   SAME matrix, and runs the GPU solve twice. Built host-side (g++) and linked with
   hipcc against `libstrumpack.a`. At n=25/35/40/60 (15625..216000 unknowns):
   - GPU componentwise scaled residual ||Ax-b|| ~ 1.1e-15 .. 2.7e-15 (converged).
   - CPU reference residual ~ matches (1.5e-15 .. 2.7e-15).
   - ||x_gpu - x_cpu||inf/||x_cpu||inf ~ 2.6e-15 .. 4.7e-15 (GPU matches CPU to eps).
   - ||x_gpu - x_exact|| ~ 2e-15 .. 4.4e-15 (matches a known exact solution).
   - GPU run-to-run: BITWISE IDENTICAL (deterministic).

   ```
   g++ -std=c++17 -O2 -fopenmp -I$SRC/src -I build -I build/src \
     -c agent_space/strumpack_validate.cpp -o agent_space/strumpack_validate.o
   /opt/rocm/bin/hipcc --offload-arch=gfx90a agent_space/strumpack_validate.o \
     build/libstrumpack.a -L/opt/rocm/lib -lrocsolver -lrocblas -lhipblas \
     -lhipsparse /usr/lib/x86_64-linux-gnu/libopenblas.so \
     /usr/lib/x86_64-linux-gnu/libmetis.so -lgfortran -fopenmp \
     -o agent_space/strumpack_validate
   HIP_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 \
     ./agent_space/strumpack_validate 40
   ```

2. **Shipped example `examples/sparse/testPoisson3d`** (Poisson generator +
   `max_scaled_residual`): n=50 (125000 unknowns) -> COMPONENTWISE SCALED
   RESIDUAL = 2.53e-15, 1 Krylov iteration.

   ```
   HIP_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 \
     build/examples/sparse/testPoisson3d 50
   ```

3. **Shipped Matrix-Market matrix** `examples/sparse/data/pde900.mtx` (a real,
   non-Poisson sparse system) via `testMMdouble`: scaled residual 3.90e-16,
   relative error vs true solution 5.05e-16.

4. **STRUMPACK's own sequential test** `test/test_sparse_seq` on pde900.mtx (the
   ctest `user_test_sparse_seq` invocation): EXIT 0, residual 3.90e-16. Runs
   several solver configurations and asserts correctness internally.

   ```
   HIP_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 \
     build/test/test_sparse_seq $SRC/examples/sparse/data/pde900.mtx
   ```

5. **Downstream `find_package(STRUMPACK)` consumer** (the RXMesh consumption path,
   `agent_space/strumpack_consumer/`): a minimal project that does
   `find_package(STRUMPACK REQUIRED)` + links `STRUMPACK::strumpack`, compiled with
   plain g++ (no HIP driver), builds cleanly and solves on the GPU at n=35 (residual
   1.82e-15, GPU==CPU to 2.7e-15, bitwise deterministic). Confirms both CMake fixes:
   find_package resolves the roc:: targets, and the consumer's host CXX is NOT
   handed `-x hip`.

## Install as a dependency

This is what unblocks RXMesh's solver path (RXMesh substitutes cuDSS -> STRUMPACK;
RXMesh `status.json depends_on: ["STRUMPACK"]`). Per the MOAT `_deps` workflow,
clone the ported fork's `moat-port` branch, build+install with the HIP flags, and
point the dependent at the install prefix.

### Build + install (arch from `CMAKE_HIP_ARCHITECTURES`)

```
# 1. clone the ported dep (deliverable branch)
git clone -b moat-port https://github.com/AMD-Ecosystem/STRUMPACK _deps/STRUMPACK/src

# 2. configure the sequential HIP build (arch is a cache var; followers pass their own)
cmake -S _deps/STRUMPACK/src -B _deps/STRUMPACK/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/var/lib/jenkins/moat/_deps/STRUMPACK/install \
  -DSTRUMPACK_USE_MPI=OFF \
  -DSTRUMPACK_USE_OPENMP=ON \
  -DSTRUMPACK_USE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DTPL_ENABLE_SLATE=OFF -DTPL_ENABLE_PARMETIS=OFF \
  -DTPL_ENABLE_SCOTCH=OFF -DTPL_ENABLE_PTSCOTCH=OFF \
  -DTPL_ENABLE_BPACK=OFF -DTPL_ENABLE_ZFP=OFF -DTPL_ENABLE_COMBBLAS=OFF \
  -DTPL_METIS_PREFIX=/usr \
  -DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libopenblas.so \
  -DLAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/libopenblas.so

# 3. build + install
cmake --build _deps/STRUMPACK/build -j16 --target install
```

Requires `libmetis-dev` + `libopenblas-dev` (apt). For gfx1100/gfx1151 followers,
change only `-DCMAKE_HIP_ARCHITECTURES=<arch>` (no source change).

### Install prefix layout

```
<prefix>/include/                         StrumpackSparseSolver.hpp, sparse/CSRMatrix.hpp, StrumpackConfig.h, ...
<prefix>/lib/libstrumpack.a               the static library (HIP-built)
<prefix>/lib/cmake/STRUMPACK/strumpack-config.cmake   the package config (exports STRUMPACK::strumpack)
```

### What the dependent (RXMesh) sets to consume it

CMake (preferred -- the package config pulls METIS/OpenMP/OpenBLAS + the roc:: libs
via find_dependency, so the consumer only needs ROCm on `CMAKE_PREFIX_PATH`):

```cmake
find_package(STRUMPACK REQUIRED)              # provides STRUMPACK::strumpack
target_link_libraries(<rxmesh_target> PRIVATE STRUMPACK::strumpack)
```

Configure the dependent with:

```
-DCMAKE_PREFIX_PATH="/var/lib/jenkins/moat/_deps/STRUMPACK/install;/opt/rocm"
```

The consumer compiles with a normal CXX compiler (g++); it does NOT need the HIP
driver and does NOT inherit `-x hip` on its host sources (fix #1). Header to include:
`#include "StrumpackSparseSolver.hpp"`. cuDSS -> STRUMPACK API mapping for RXMesh is
factor/solve via `StrumpackSparseSolver<scalar,integer>`: `set_matrix(CSRMatrix)`,
`reorder()`, `factor()`, `solve(b, x)`; the factorization auto-runs on the GPU when
the lib is built with HIP (`use_gpu_` default true).

Manual include/link flags (if not using find_package):
`-I<prefix>/include` and link `<prefix>/lib/libstrumpack.a -L/opt/rocm/lib
-lrocsolver -lrocblas -lhipblas -lhipsparse <openblas> <metis> -lgfortran -fopenmp`.

## Deliverable

Two CMake fixes (`CMakeLists.txt`, `cmake/strumpack-config.cmake.in`; +20/-1),
both NVIDIA-safe, both required for the ROCm 7.2.1 build to compile and be
consumable. No kernel/algorithm source edits. Per MOAT boundaries this agent did
NOT fork/push; the parent delivers the diff. The diff lives in the working clone
`projects/STRUMPACK/src` (see `git diff` there). No README/gen_readme changes; no
GitHub Actions.

## Validation 2026-05-30 (gfx1100, ROCm 7.2.1)

Platform: AMD Radeon Pro W7800 48GB x2, gfx1100 (RDNA3, wave32), ROCm 7.2.1.
Fork: AMD-Ecosystem/STRUMPACK moat-port, commit `ab5396f0f71e2c16955b6676399a68b21a601159`.
No fork commit added (follower validation, no code change needed).

### Build

Cloned fork into `projects/STRUMPACK/src/`, configured and built sequentially:

```
cmake -S projects/STRUMPACK/src -B projects/STRUMPACK/build-gfx1100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=.../build-gfx1100/install \
  -DSTRUMPACK_USE_MPI=OFF \
  -DSTRUMPACK_USE_OPENMP=ON \
  -DSTRUMPACK_USE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DTPL_ENABLE_SLATE=OFF -DTPL_ENABLE_PARMETIS=OFF \
  -DTPL_ENABLE_SCOTCH=OFF -DTPL_ENABLE_PTSCOTCH=OFF \
  -DTPL_ENABLE_BPACK=OFF -DTPL_ENABLE_ZFP=OFF -DTPL_ENABLE_COMBBLAS=OFF \
  -DTPL_METIS_PREFIX=/usr \
  -DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libopenblas.so \
  -DLAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/libopenblas.so
cmake --build projects/STRUMPACK/build-gfx1100 -j64
```

Build time: 18.4s (library only; linking examples adds ~10s). Clean exit, no warnings about
`--offload-arch` on host TUs.

### gfx1100 code-object evidence

`llvm-objdump --offloading` on both HIP objects:
- `FrontHIP.hip.o`: `hipv4-amdgcn-amd-amdhsa--gfx1100`
- `HIPWrapper.hip.o`: `hipv4-amdgcn-amd-amdhsa--gfx1100`

No `gfx90a` literal anywhere in the source or CMake. The arch flows entirely from
`CMAKE_HIP_ARCHITECTURES=gfx1100` at configure time.

### Host CXX flag check (fix #1 verification)

From `CMakeFiles/strumpack.dir/flags.make`:
```
HIP_FLAGS = -O3 -DNDEBUG -std=c++17 --offload-arch=gfx1100 -fPIC
CXX_FLAGS = -O3 -DNDEBUG -fPIC -Wall -Wno-overloaded-virtual -fopenmp
```
CXX_FLAGS has no `-x hip` and no `--offload-arch`. The offload flag appears only in
HIP_FLAGS (applied to `.hip` TUs). Fix #1 confirmed effective on gfx1100.

### GPU factor+solve validation (ALL PASS)

Custom harness `agent_space/strumpack_validate.cpp` (same as gfx90a), built and linked:
```
g++ -std=c++17 -O2 -fopenmp -I.../src/src -I.../build-gfx1100 \
  -c agent_space/strumpack_validate.cpp -o agent_space/strumpack_validate_gfx1100.o
/opt/rocm/bin/hipcc --offload-arch=gfx1100 agent_space/strumpack_validate_gfx1100.o \
  build-gfx1100/libstrumpack.a \
  -L/opt/rocm/lib -lrocsolver -lrocblas -lhipblas -lhipsparse \
  /usr/lib/x86_64-linux-gnu/libopenblas.so /usr/lib/x86_64-linux-gnu/libmetis.so \
  -lgfortran -fopenmp -o agent_space/strumpack_validate_gfx1100
HIP_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 \
  agent_space/strumpack_validate_gfx1100 40
```

Results at n=40 (64000 unknowns):
- GPU relative residual ||Ax-b||/||b|| = 7.476e-15 (converged)
- CPU reference residual = 6.439e-15
- ||x_gpu - x_cpu||_inf / ||x_cpu||_inf = 8.216e-15 (GPU matches CPU to machine eps)
- ||x_gpu - x_exact||_inf / ||x_exact|| = 5.773e-14 (matches exact solution to near-eps)
- ||x_gpu1 - x_gpu2||_inf = 0.000e+00 (BITWISE IDENTICAL -- deterministic)

Test time (timeit): 4.2s. Exit 0. PASS.

Additional shipped examples:
- `testPoisson3d 50` (125000 unknowns): COMPONENTWISE SCALED RESIDUAL = 2.43e-15 (1 Krylov iter).
- `testMMdouble pde900.mtx`: COMPONENTWISE SCALED RESIDUAL = 3.90e-16, RELATIVE ERROR = 5.05e-16.
- `test_sparse_seq pde900.mtx`: EXIT 0, SCALED RESIDUAL = 3.90e-16.

All results match gfx90a (MI250X, wave64) closely. The rocSOLVER/rocSPARSE calls are
arch-agnostic; the multifrontal LU + triangular solve correct on gfx1100 (RDNA3, wave32)
confirms the port's wave64-to-wave32 portability.

## Validation 2026-06-05 (gfx1101, Windows 11 -- BLOCKED)

Platform: Radeon PRO V710 (gfx1101, RDNA3, wave32), Windows 11 Pro, TheRock ROCm 7.14.0a20260604.
Fork: AMD-Ecosystem/STRUMPACK moat-port, commit `ab5396f0f71e2c16955b6676399a68b21a601159`.

### Build attempt

Tried three compiler toolchain combinations:

1. **All-clang (TheRock clang++/clang + flang for Fortran)**: TheRock's `flang.exe` is the
   internal LLVM Fortran frontend binary (`flang -fc1` layer), not a user-facing driver. It
   rejects all driver-level flags (`-triple`, `-emit-obj`, etc.) with "unknown argument" errors
   and cannot compile `.f90` files directly. CMake also fails to detect it as a valid Fortran
   compiler (`CMAKE_Fortran_PREPROCESS_SOURCE` not set, Fortran ID = unknown).

2. **MinGW all-GNU (Strawberry gcc/g++/gfortran) + TheRock clang++ for HIP**: CMake's
   `Windows-Clang-HIP.cmake:187` error: "The current configuration mixes Clang and MSVC or
   some other CL compatible compiler tool. Use either Clang or MSVC for all of C, C++, and/or
   HIP." MinGW GNU ABI + MSVC-ABI Clang-HIP is refused.

3. **MSVC cl.exe for C/CXX + TheRock clang++ for HIP + gfortran for Fortran**: CMake passes
   MSVC linker flags (`/machine:x64`) to the gfortran link step, causing MinGW's ld to fail
   with "cannot find /machine:x64". MSVC + MinGW-ABI gfortran cannot share a linker.

### Root cause

STRUMPACK compiles 8 Fortran `.f` source files (`src/dense/lapack/[sdcz]geqp3tol.f`,
`[sdcz]lapmr.f`) into `libstrumpack.a` (custom QR pivot routines). No working MSVC-ABI
Fortran compiler exists on this host:
- **MinGW gfortran**: MinGW ABI, incompatible with Clang-HIP (MSVC ABI target).
- **TheRock flang.exe**: Internal frontend binary, not a functional driver.
- **Intel ifort/ifx**: Not installed.

### How to unblock

Install **Intel oneAPI Base Toolkit** + **HPC Toolkit** (provides ifort/ifx, MSVC-ABI
compatible) on this Windows host, or install a working LLVM flang driver
(`flang-new.exe` targeting MSVC ABI). With either compiler:
- Use `CMAKE_Fortran_COMPILER=ifx` (Intel) or `CMAKE_Fortran_COMPILER=flang-new` (LLVM).
- The rest of the build (clang++ for C/CXX/HIP, METIS from RXMesh deps, rocm-openblas from
  TheRock `_rocm_sdk_devel/lib/host-math/lib/rocm-openblas.lib`) is ready to go.
- METIS: `B:/develop/moat/projects/RXMesh/build/_deps/metis-build/metis.lib` + GKlib.lib.
- OpenBLAS: `_rocm_sdk_devel/lib/host-math/lib/rocm-openblas.lib` (DLL at
  `_rocm_sdk_devel/bin/rocm-openblas.dll`).
- ROCBLAS_TENSILE_LIBPATH=`_rocm_sdk_libraries/bin/rocblas/library` at runtime.
- All ROCm cmake packages in `_rocm_sdk_devel/lib/cmake/` (hip, hipblas, hipsparse,
  rocsolver, rocblas, rocprim, rocthrust).
