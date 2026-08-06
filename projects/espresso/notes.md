# espresso notes

ESPResSo molecular dynamics. Scoped CORE-GPU HIP port (P3M electrostatics +
dipolar direct sum + GpuParticleData + pinned-host allocator + cuda init).
waLBerla LB/EK GPU is explicitly OUT OF SCOPE.

- Fork: https://github.com/AMD-Ecosystem/espresso  (branch `moat-port`)
- head_sha: 6d36cb6aaf5425c1710b756d79f67ec7fcb9a0ca
- Based on upstream eafab258df3b (5.0.1 release notes)
- Strategy A (compat header + LANGUAGE HIP), NVIDIA path byte-identical.

## Build recipe (gfx90a, ROCm 7.2.1)

WITH_WALBERLA=OFF collapses the GPU surface to the 7 core .cu files and avoids
the egress-heavy waLBerla GitLab fetch. heffte/Kokkos/Cabana are still
FetchContent'd from GitHub (kokkos.git, ECP-copa/Cabana, icl-utk-edu/heffte) but
are CPU-only here.

```
export ROCM_PATH=/opt/rocm
cmake -S . -B build-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DESPRESSO_BUILD_WITH_CUDA=ON -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DESPRESSO_BUILD_WITH_WALBERLA=OFF -DESPRESSO_BUILD_WITH_HDF5=OFF \
  -DESPRESSO_BUILD_WITH_SCAFACOS=OFF -DESPRESSO_BUILD_WITH_STOKESIAN_DYNAMICS=OFF \
  -DESPRESSO_BUILD_WITH_GSL=OFF -DESPRESSO_BUILD_WITH_NLOPT=OFF \
  -DESPRESSO_BUILD_WITH_CALIPER=OFF -DESPRESSO_BUILD_WITH_FFTW=ON \
  -DESPRESSO_BUILD_WITH_PYTHON=ON -DESPRESSO_BUILD_TESTS=ON
cmake --build build-hip -j16
```

Build deps were all already present on the gfx90a Jenkins host (nothing to
apt/conda-install): CMake 4.0.3, Ninja, OpenMPI 4.1.6, full libboost-all-dev,
FFTW3, Python 3.12 + numpy/scipy/cython 3.2.5, libhwloc-dev, ROCm 7.2.1
(hipfft + thrust headers under /opt/rocm/include).

Multi-arch fat binary verified: both gfx90a and gfx1100 device code objects
emit in src/core/espresso_core.so (followers need no source change):
`llvm-objdump --offloading build-hip/src/core/espresso_core.so | grep gfx`.

## Test data targets (not built by default)

The python tests import shared helpers and the samples read script files that a
custom target copies into the build tree. ctest fails with
`ModuleNotFoundError: unittest_decorators` / `FileNotFoundError ...local_samples`
until you build them:
```
cmake --build build-hip --target python_test_data   # unittest_decorators, tests_common, data/
cmake --build build-hip --target cuda_test          # C++ unit test (EXCLUDE_FROM_ALL)
```
The `sample_*_with_gpu` / `benchmark_*_with_gpu` ctest failures are missing
`local_samples/*.py` (a samples copy target was not built), NOT GPU faults and
NOT regressions; they fail the same way on the CUDA build.

## Validation (real gfx90a, HIP_VISIBLE_DEVICES=0)

Core CPU-vs-GPU / GPU-vs-analytic consistency gates -- 11/11 PASS:
- coulomb_cloud_wall, coulomb_cloud_wall_duplicated (P3M GPU vs reference)
- p3m_madelung (P3M GPU vs analytic Madelung constant -- proves hipFFT
  R2C/C2R normalization parity; a factor-of-N error would fail here)
- p3m_electrostatic_pressure
- dawaanr-and-dds-gpu, dipolar_direct_summation (DipolarDirectSum GPU vs CPU
  forces+torques)
- dipolar_interface, coulomb_interface, dipole_field_tracking, gpu_availability
- cuda_test (C++ pinned-host allocator unit test)

Non-GPU baseline (`ctest -LE gpu`) unaffected (p3m_fft, mmm1d CPU tests pass).
LB/EK GPU tests (lb*, ek*) need waLBerla GPU -> out of scope, not regressions.

## Fault classes hit

1. __CUDACC__ vs __HIPCC__ (the big one). clang for HIP defines __HIPCC__/__HIP__
   but NOT __CUDACC__. ESPResSo gated `DEVICE_QUALIFIER` (= __host__ __device__)
   on __CUDACC__ only (src/utils/include/utils/device_qualifier.hpp), so the
   shared bspline/sinc helpers became host-only under HIP and could not be called
   from device code. Same trap in the "do not include CUDA headers" guard in
   cuda/utils.cuh. Fix: accept __HIPCC__ too.
2. Rule-of-five on cuFFT handles (colmap pattern). P3MGpuFftPlan destroyed its two
   cufftHandle plans unconditionally; a default-constructed / never-initialized
   holder destroyed garbage handles -> ROCm faults (CUDA tolerates). Fix: default
   handles to 0, guarded destroy that resets to 0. Arch-unified.
3. -ffp-contract=on pinned in CMAKE_HIP_FLAGS (clang-HIP defaults to fast) so the
   B-spline weight chains match CUDA/CPU FMA formation (CV-CUDA fault class).
4. hip::device interface-compile-options poisoning: linking roc::rocthrust pulls
   hip::device, whose INTERFACE_COMPILE_OPTIONS inject `-x hip` (wrapped in
   $<COMPILE_LANGUAGE:CXX>) into EVERY host .cpp of the consumer -> g++ chokes on
   `--offload-arch`. Fix: do not link roc::rocthrust (header-only, resolved from
   the ROCm include path); link only hip::hipfft (pulls hip::host, safe).
5. clang-HIP does not implicitly include the runtime header (nvcc does), so the
   .cu files needed the shim include explicitly; CudaHostAllocator.cu had no cuda
   include at all (relied on nvcc auto-include).
6. hipHostMalloc is a template(T**, size_t, flags=0) and would not bind the
   2-arg void** cudaMallocHost call (nor the nullptr probe in cuda_test). Provided
   an inline 2-arg forwarder in the shim instead of a macro.

No warp intrinsics / __shfl / hardcoded-32 anywhere in src/core, so no wave-size
fix was needed; the gfx90a/gfx1100 fat binary covers the followers.

## Environment notes

- Codebase moved well past the planner's view: P3M now uses heffte as its FFT
  backend (p3m_heffte.cpp), and Kokkos/Cabana are mandatory FetchContent deps.
  heffte's CUDA backend (Heffte_ENABLE_CUDA) and the OpenMP_CUDA component are
  both gated off on the HIP path (they are not what the GPU P3M uses; our GPU FFT
  is the hipFFT in p3m_gpu_cuda.cu).
- The HIP language path bypasses the nvcc>=12 / CMAKE_CUDA_COMPILER_ID gate
  entirely, so the C++20 CUDA-version checks never run.

## CMake option

`option(USE_HIP ...)` (default OFF). With ESPRESSO_BUILD_WITH_CUDA=ON it enables
language HIP, defaults CMAKE_HIP_ARCHITECTURES to gfx90a only-if-unset, flips the
7 core .cu to LANGUAGE HIP, links hip::hipfft, and pins -ffp-contract=on.

## Review 2026-06-04

Verdict: changes-requested (one defect; everything else verified sound). The
fault-class analysis (the __CUDACC__/__HIPCC__ device-qualifier fix and the
Thrust-header-only decision both confirmed correct) holds; the only blocker is
a commit-hygiene violation that the porter must fix before validation.

### Commit Hygiene
- Commit 00149db4 title is 75 chars, over the 72-char limit (CLAUDE.md / pr-review
  checklist 7): "[ROCm] Add AMD GPU (HIP) support for the core electrostatics/magnetostatics".
  Shorten, e.g. "[ROCm] Add AMD GPU (HIP) support for core electrostatics" (54).
  Amending is safe here: no platform has validated this sha (validated_sha=null),
  so no validated commit is orphaned.

### Verified sound (no action; recorded so the next reviewer need not re-derive)
- __CUDACC__ vs __HIPCC__ fix (device_qualifier.hpp:22, utils.cuh:22): correct.
  clang -x hip defines BOTH __HIPCC__ and __HIP__ (confirmed via clang++ -dM -E
  -x hip --offload-arch=gfx90a), so DEVICE_QUALIFIER now expands to
  __host__ __device__ and the shared bspline/sinc helpers (bspline.hpp:43,202)
  are device-callable under HIP. CUDA path unchanged (still keys on __CUDACC__).
  The remaining __CUDACC__-only spellings in the tree are all under
  src/walberla_bridge/, which is out of scope (WITH_WALBERLA=OFF) -- correct not
  to touch them.
- Minor inconsistency, not a defect: the shim and cuda_test guard on __HIP__
  (cuda_to_hip.h:31, cuda_test.cu:198) while device_qualifier/utils.cuh guard on
  __HIPCC__. Both macros are defined together by clang-HIP, so behavior is
  identical; left as-is.
- cuFFT->hipFFT (cuda_to_hip.h:74-87): 1:1 macro/alias swap (R2C/C2R/D2Z/Z2D,
  Plan3d, Destroy, Complex types, CUFFT_* enums). Round-trip 1/N is absorbed in
  the analytic G_hat influence function (p3m_gpu_cuda.cu:252) -- identical code
  both vendors, both libs leave transforms unnormalized. p3m_madelung compares
  GPU result to the analytic Madelung constant, so a factor-of-N would fail it;
  it passes -> normalization parity genuinely gated.
- Thrust header-only / no roc::rocthrust link (src/core/CMakeLists.txt:64): sound,
  not fragile. Traced roc::rocthrust -> roc::rocprim_hip -> "roc::rocprim;hip::device";
  hip::device's INTERFACE_COMPILE_OPTIONS inject "-x hip" (hip-config-amd.cmake:159),
  which would poison the host .cpp TUs. hip::hipfft links only hip::host
  (hipfft-targets.cmake:63), safe. thrust headers resolve from /opt/rocm/include
  (device_vector/reduce/copy/device_ptr/raw_pointer_cast only -- all header-only).
- Rule-of-five on P3MGpuFftPlan (p3m_gpu_cuda.cu:110-141): handles default to 0,
  guarded lambda destroy resets to 0. colmap pattern, arch-unified, CUDA-safe
  (avoids cufftDestroy(garbage); behavior-preserving for an initialized plan).
- -ffp-contract=on (CMakeLists.txt CMAKE_HIP_FLAGS), 2-arg cudaMallocHost inline
  forwarder (cuda_to_hip.h, returns hipError_t for the void** call at
  CudaHostAllocator.cu:27), cstdlib/cstring-before-hip_runtime ordering, and the
  clang-HIP explicit shim include in the .cu files: each behavior-preserving and
  CUDA-path-safe.
- cuda_test.cu:198 error-string fork (__HIP__): cudaErrorNotPermitted aliases to
  hipErrorNotSupported ("operation not supported"); the guarded expected-string
  matches the alias. Correct test adaptation, NVIDIA branch unchanged.
- Scoping: WITH_WALBERLA=OFF (LB/EK GPU out of scope) documented in commit body,
  plan.md, and notes.md. sample_*/benchmark_*_with_gpu -L gpu failures are missing
  local_samples test data (notes "Test data targets"), fail identically on CUDA --
  not regressions.
- No warp intrinsics / hardcoded 32 / warpSize anywhere in src/core (grep clean) --
  wave-size fault class has no trigger; gfx90a+gfx1100 fat binary covers followers.
- ESPRESSO_OPENMP_CUDA refactor (src/core/CMakeLists.txt:88, consumed in
  p3m/CMakeLists.txt): set before add_subdirectory(p3m) at line 104, visible in the
  child scope; empty on HIP (drops the CUDA-language OpenMP target, correct), still
  OpenMP::OpenMP_CUDA on a plain CUDA build -- behavior-preserving.

## Review 2026-06-04 (re-review after changes-requested)

Verdict: review-passed. The sole prior blocker (commit-title 75 > 72) is fixed.
- New title "[ROCm] Add HIP support for core electrostatics and magnetostatics" = 65 chars.
- 00149db4 -> 6d36cb6a is a title-only amend: git diff is empty, both commits share
  tree 44c92340e953cba40ce5d05a5fc8a5ae57a65fe1 (tree-identical), so the entire prior
  code approval carries unchanged.
- Body hygiene re-verified clean: Claude disclosed, Test Plan with literal fenced
  commands, no noreply/Co-Authored trailer, no MOAT jargon, no em-dash, [ROCm] prefix.
No new problems. Ready for the validator (real-GPU run pending, expected at this stage).

## Validation 2026-06-04 (linux-gfx90a)

State: review-passed -> completed. GPU: gfx90a (GCD1, HIP_VISIBLE_DEVICES=1).
Validated sha: 6d36cb6aaf5425c1710b756d79f67ec7fcb9a0ca.

### Build

```
export ROCM_PATH=/opt/rocm
cmake -S . -B build-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DESPRESSO_BUILD_WITH_CUDA=ON -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DESPRESSO_BUILD_WITH_WALBERLA=OFF -DESPRESSO_BUILD_WITH_HDF5=OFF \
  -DESPRESSO_BUILD_WITH_SCAFACOS=OFF -DESPRESSO_BUILD_WITH_STOKESIAN_DYNAMICS=OFF \
  -DESPRESSO_BUILD_WITH_GSL=OFF -DESPRESSO_BUILD_WITH_NLOPT=OFF \
  -DESPRESSO_BUILD_WITH_CALIPER=OFF -DESPRESSO_BUILD_WITH_FFTW=ON \
  -DESPRESSO_BUILD_WITH_PYTHON=ON -DESPRESSO_BUILD_TESTS=ON
cmake --build build-hip -j16
cmake --build build-hip --target python_test_data
cmake --build build-hip --target cuda_test
```

Build: PASS. gfx90a and gfx1100 device code objects confirmed in espresso_core.so via
`llvm-objdump --offloading`.

### GPU tests: 11/11 PASS

```
HIP_VISIBLE_DEVICES=1 ctest -L gpu --timeout 600 -j4 --output-on-failure \
  -R 'coulomb_cloud_wall|coulomb_cloud_wall_duplicated|p3m_madelung|p3m_electrostatic_pressure|dawaanr-and-dds-gpu|dipolar_direct_summation|dipolar_interface|coulomb_interface|dipole_field_tracking|gpu_availability'
HIP_VISIBLE_DEVICES=1 ctest -R cuda_test --timeout 120 --output-on-failure
```

Core CPU-vs-GPU / GPU-vs-analytic consistency gates (all PASS):
- coulomb_cloud_wall: PASS
- coulomb_cloud_wall_duplicated: PASS
- p3m_madelung (P3M GPU vs analytic Madelung constant; hipFFT R2C/C2R normalization gated): PASS
- p3m_electrostatic_pressure: PASS
- dawaanr-and-dds-gpu (DipolarDirectSum GPU vs CPU forces+torques): PASS
- dipolar_direct_summation: PASS
- dipolar_interface: PASS
- coulomb_interface: PASS
- dipole_field_tracking: PASS
- gpu_availability: PASS
- cuda_test (C++ pinned-host allocator unit test): PASS

5 sample_*/benchmark_*_with_gpu tests fail with FileNotFoundError (local_samples/*.py
missing -- a samples copy target not built). Confirmed NOT GPU faults; identical
failure on CUDA builds. Not regressions.

LB/EK GPU tests (lb*, ek*) need waLBerla GPU -- out of scope (WITH_WALBERLA=OFF).

### Non-GPU baseline: 294/382 passed; 88 pre-existing failures, no regressions

```
HIP_VISIBLE_DEVICES=1 ctest -LE gpu --timeout 300 -j8
```

Key CPU tests confirmed PASS: mmm1d, p3m_fft, coulomb_mixed_periodicity.
The 88 failures are ALL pre-existing environmental issues unrelated to the HIP port:
- ~52 sample_* tests: missing local_samples/*.py data (same pattern as GPU samples)
- ~8 benchmark_* (CPU) tests: missing local_benchmarks data
- test_convert: missing nbformat Python module (tutorial conversion test)
- test_install: cmake --install never run (reference file diff fails)
- ~25 C++ unit_test binaries: GTest not installed on host -> CMake disabled
  "Build unit tests (requires GTest)" -> mpiexec cannot launch missing executables

## Validation 2026-06-04 (linux-gfx1100, RDNA3 wave32)

State: port-ready -> completed. GPU: gfx1100 (AMD Radeon Pro W7800 48GB, HIP_VISIBLE_DEVICES=3).
ROCm 7.2.1. warpSize=32 confirmed via rocminfo. Wavefront Size: 32(0x20).
Validated sha: 6d36cb6aaf5425c1710b756d79f67ec7fcb9a0ca.

### Build (gfx1100-only, ROCm 7.2.1)

```
export ROCM_PATH=/opt/rocm
cmake -S . -B build-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DESPRESSO_BUILD_WITH_CUDA=ON -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx1100" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DESPRESSO_BUILD_WITH_WALBERLA=OFF -DESPRESSO_BUILD_WITH_HDF5=OFF \
  -DESPRESSO_BUILD_WITH_SCAFACOS=OFF -DESPRESSO_BUILD_WITH_STOKESIAN_DYNAMICS=OFF \
  -DESPRESSO_BUILD_WITH_GSL=OFF -DESPRESSO_BUILD_WITH_NLOPT=OFF \
  -DESPRESSO_BUILD_WITH_CALIPER=OFF -DESPRESSO_BUILD_WITH_FFTW=ON \
  -DESPRESSO_BUILD_WITH_PYTHON=ON -DESPRESSO_BUILD_TESTS=ON
cmake --build build-hip -j16
cmake --build build-hip --target python_test_data
cmake --build build-hip --target cuda_test
```

Build: PASS (234/234). gfx1100 device code objects confirmed in espresso_core.so:
`llvm-objdump --offloading build-hip/src/core/espresso_core.so | grep gfx`
-> 4 bundles: hipv4-amdgcn-amd-amdhsa--gfx1100 (all 7 core .cu kernels present).

### Host env note

This gfx1100 Jenkins host has a conda OpenSSL 3.6.2 mismatch against the system
libcrypto.so.3. The conda libssl.so.3 (OpenSSL 3.6.2) requires OPENSSL_3.3.0 from
libcrypto, but the system libcrypto.so.3 only has earlier symbols. Setting
`LD_LIBRARY_PATH=/opt/conda/envs/py_3.12/lib:$LD_LIBRARY_PATH` before ctest ensures
the conda libcrypto.so.3 is resolved first. This is a host environment issue unrelated
to the port; the gfx90a host resolves consistently without this workaround.

### GPU tests: 11/11 PASS

```
export LD_LIBRARY_PATH=/opt/conda/envs/py_3.12/lib:${LD_LIBRARY_PATH:-}
HIP_VISIBLE_DEVICES=3 ctest -L gpu --timeout 600 -j4 --output-on-failure \
  -R 'coulomb_cloud_wall$|coulomb_cloud_wall_duplicated|p3m_madelung|p3m_electrostatic_pressure|dawaanr-and-dds-gpu|dipolar_direct_summation|dipolar_interface|coulomb_interface|dipole_field_tracking|gpu_availability'
HIP_VISIBLE_DEVICES=3 ctest -R cuda_test --timeout 120 --output-on-failure
```

Core CPU-vs-GPU / GPU-vs-analytic consistency gates (all PASS):
- coulomb_cloud_wall: PASS
- coulomb_cloud_wall_duplicated: PASS
- p3m_madelung (P3M GPU vs analytic Madelung constant; hipFFT R2C/C2R normalization gated): PASS
- p3m_electrostatic_pressure: PASS
- dawaanr-and-dds-gpu (DipolarDirectSum GPU vs CPU forces+torques): PASS
- dipolar_direct_summation: PASS
- dipolar_interface: PASS
- coulomb_interface: PASS
- dipole_field_tracking: PASS
- gpu_availability: PASS
- cuda_test (C++ pinned-host allocator unit test): PASS

### Non-GPU baseline spot-check: PASS

```
HIP_VISIBLE_DEVICES=3 ctest -LE gpu -R "p3m_fft|mmm1d|coulomb_mixed" --timeout 300 -j4
```

coulomb_mixed_periodicity, mmm1d, p3m_fft: all PASS. No regressions vs gfx90a baseline.

## Validation 2026-06-08 (windows-gfx1201 -- BLOCKED)

State: port-ready -> blocked. GPU: gfx1201 confirmed present at HIP_VISIBLE_DEVICES=0
(AMD Radeon RX 9070 XT, gcnArchName gfx1201, verified via hipInfo.exe).
Fork cloned at 6d36cb6aaf5425c1710b756d79f67ec7fcb9a0ca; HEAD correct.

ESPResSo requires MPI 3.0 (`find_package(MPI 3.0 REQUIRED)`) and `Boost::mpi` to
build. On Windows the only viable MPI implementation is MS-MPI (Microsoft MPI).
The MS-MPI installer (msmpisetup.exe) requires UAC/administrator elevation; the
automated validation environment runs as a standard user and cannot install
system-level packages.

Attempted workarounds (all exhausted):
- `winget install Microsoft.msmpi`: downloads correctly but its installer requires
  elevation -> `WinError 740` (The requested operation requires elevation).
- Extracting the MS-MPI SDK MSI manually: extracted mpi.h + msmpi.lib for building,
  but the runtime msmpi.dll is inside a delta-compressed cabinet (XPRESS-delta
  or MSZIP CM=13 variant, not standard DEFLATE) that cannot be decompressed without
  the Windows Installer cabinet stack (expand.exe returns 0xca00a009
  DELTA_E_ILLEGAL_DELTA_FORMAT; FDI API segfaults on the same cab).
- pip install mpi4py: package installed but needs system msmpi.dll to load -> fails.
- Alternative MPI (MPICH, OpenMPI for Windows): not available without admin install.
- vcpkg mpi: no mpi port available.

Without msmpi.dll:
  - CMake configure would fail at `find_package(MPI 3.0 REQUIRED)`.
  - Even if header-only MPI were accepted, linking msmpi.lib produces an executable
    that tries to load msmpi.dll at startup, so espressomd module import fails.
  - The Boost.MPI compiled library also requires msmpi.dll.
  - The GPU tests (espressomd Python tests) all link transitively against msmpi.dll.

This is a host infrastructure blocker (missing system dependency requiring admin
install), NOT a port defect. The HIP port is correct as proven by linux-gfx90a
and linux-gfx1100 validation.

The linux-gfx1100 gfx1201 fat-binary path would apply equally to windows-gfx1201
if MPI were available (gfx1201 is wave32, same as gfx1100, and the port has no
warp-size assumptions). A future validation would succeed if MS-MPI is installed
by an admin before the session, or if the project is rebuilt on Linux where MPI
is available as a standard package.
