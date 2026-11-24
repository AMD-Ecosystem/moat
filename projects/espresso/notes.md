# espresso notes

ESPResSo molecular dynamics. Scoped CORE-GPU HIP port (P3M electrostatics +
dipolar direct sum + GpuParticleData + pinned-host allocator + cuda init).
waLBerla LB/EK GPU is explicitly OUT OF SCOPE.

- Fork: https://github.com/AMD-Ecosystem/espresso  (branch `moat-port`)
- head_sha: 6d36cb6aaf5425c1710b756d79f67ec7fcb9a0ca
- Based on upstream eafab258df3b (5.0.1 release notes)
- Strategy A (compat header + LANGUAGE HIP). CORRECTION 2026-11-19: the NVIDIA path was
  NOT byte-identical and was never nvcc-checked until 2026-11-19; from 6d36cb6aa to 87cbc046d
  the nvcc build failed (missing `<cufft.h>`, see "CUDA no-regression compile" below). Fixed
  in 01090739c. The CUDA build graph (flags, defines, includes, link libs, objects) is now
  identical to upstream's; the source still differs on the CUDA path (extra headers via the
  compat header, the cuFFT plan-destroy guard), see "Porter 2026-11-19 (CUDA path fix)".

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
   cufftHandle plans unconditionally. CORRECTED 2026-11-19 (review of the CUDA path
   fix): the handles were not garbage; the holder is value-initialized through
   `std::make_shared<P3MGpuParams>()`, so they start at 0. The hazard is destroying a
   plan that was never created: `cufftDestroy(0)` on CUDA (an ignored error code),
   `hipfftDestroy(nullptr)` on HIP (hipfftHandle is a pointer), plus stale handles
   if re-planning throws after `free_device_memory()`. Fix: default handles to 0,
   guarded destroy that resets to 0. Arch-unified.
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

`option(ESPRESSO_BUILD_WITH_HIP ...)` (default OFF; named `USE_HIP` before 4861f4839). With ESPRESSO_BUILD_WITH_CUDA=ON it stops at configure if ESPRESSO_BUILD_WITH_WALBERLA is ON, enables language HIP (an explicit CMAKE_HIP_ARCHITECTURES is honoured, otherwise the build host's GPUs are auto-detected; the gfx90a default was removed in 8e228b5d), flips the 7 core .cu to LANGUAGE HIP, links hip::hipfft, and pins -ffp-contract=on. No C++ macro is derived from it; the sources test `__HIP__` / `__HIPCC__`.

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

## Upstream sync 2026-11-19 (porter, linux-gfx942)

Directed by jeff: bring the port up to date with upstream before the PR is prepared. No upstream PR is open, so `moat-port` was pushed directly.

- Fork mirror `python` fast-forwarded 62c745e14 -> c0e34ecc0 (`moatlib.py mirror-branch espresso`; upstream's own commits, no merge).
- `moat-port` 8e228b5d -> 87cbc046d: merge commit 06adcac90 (upstream c0e34ecc0, 75 commits past the old base eafab258), then a15c89169 (code_info) and 87cbc046d (docs). No rebase; validated commits untouched.

### What upstream changed that mattered

- **Kokkos GPU infrastructure (#5400, #5391).** Upstream now turns on `Kokkos_ENABLE_CUDA` when `ESPRESSO_BUILD_WITH_CUDA` is on, waLBerla is off and the CUDA compiler is not nvcc (i.e. clang-CUDA), and `espresso_set_kokkos_properties_on_source_files` may recompile every core `.cpp` as CUDA. A HIP build has no CUDA compiler id, so the condition would have been TRUE and tried to build Kokkos with CUDA. Every Kokkos call in `src/core` still runs on `Kokkos::DefaultHostExecutionSpace` / `HostSpace` (grep: no `DefaultExecutionSpace`, `CudaSpace` or device policy anywhere outside walberla), and nvcc builds never enable the Kokkos CUDA backend, so no GPU code is lost by keeping it off. Gated with `AND NOT USE_HIP` (in the merge commit). Enabling Kokkos HIP would be a later decision if upstream ever moves kernels to a device execution space.
- **kokkos-fft (new, ON by default)** with `KokkosFFT_ENABLE_CUFFT ${ESPRESSO_BUILD_WITH_CUDA}`. kokkos-fft v2.0.0 reads that option only under `Kokkos_ENABLE_CUDA` (`cmake/KokkosFFT_tpls.cmake`), so it is inert on HIP (and on nvcc builds); P3MFFTKokkos uses the host execution space and FFTW. No change.
- **heFFTe/Kokkos/Cabana fetch rework, HDF5 block moved before Cabana, Thrust >= 2.5 check.** The Thrust check is inside the CUDA-compiler branch, which a HIP build never enters.
- **`device_qualifier.hpp`** gained `HOST_ONLY_QUALIFIER` and `#pragma once`; merged cleanly under the port's `__CUDACC__ || __HIPCC__` test.
- **`p3m_gpu_cuda.cu`**: launch topology factored into `p3m_calc_topology` (block <= 1024 threads, 2D grid); `KERNELCALL` uses `__VA_OPT__`. Both fine under clang-HIP, no change.
- **`GpuParticleData_cuda.cu`**: `resize_or_replace` thrust workaround removed in favour of plain `resize`; fine with rocThrust (GPU tests pass).
- **New `testsuite/python/code_info.py` toolchain test** asserts `CMAKE_CUDA_COMPILER_ID/VERSION` are set when the CUDA feature is on. Empty under HIP -> new commit a15c89169 fills them from `CMAKE_HIP_COMPILER_ID/VERSION`, scoped to `src/config/CMakeLists.txt` only (the directory that generates `cmake_config.hpp`), so no other CMake logic sees a CUDA compiler. Passes.
- Floors raised: CMake 3.31.6, GCC 13.3, Python 3.12, NumPy 2.2.4, Cython 3.0.11, Boost 1.83.

### Conflicts and resolutions

- `CMakeLists.txt`: git produced one large conflict because upstream moved the HDF5 block ahead of Cabana and rewrote the heFFTe/Kokkos/Cabana fetches. Took upstream's file whole and re-applied the four port hunks verbatim (USE_HIP option; HIP compiler branch before the CUDA branch; OpenMP CUDA component guard; heFFTe CUDA guard), plus the new Kokkos CUDA guard above.
- `src/core/unit_tests/CMakeLists.txt`: upstream added `Kokkos::kokkos` to `cuda_test`'s deps; added to both the HIP and CUDA branches.
- Everything else auto-merged; each auto-merged GPU file was read against upstream's delta (above).

### Port doc gap closed

The port had never documented the ROCm build. 87cbc046d adds an "AMD GPU acceleration" section next to "Nvidia GPU acceleration" in `doc/sphinx/installation.rst` and lists `USE_HIP` under the external-feature options.

### Host build recipe (gfx942, pip ROCm SDK 10.1.0, no /opt/rocm)

This host has no system MPI/Boost/FFTW/hwloc headers and no /opt/rocm. Built into a job-local prefix from source (hwloc 2.11.2, OpenMPI 5.0.8, FFTW 3.3.10 double+float with omp/threads, Boost 1.87.0 mpi/serialization/test with `--without-python`; a Boost built with Python makes Boost::mpi drag in libboost_python and `cuda_test` fails to link). Python 3.12 venv (numpy 2.5.5, scipy 1.18.1, Cython 3.3.1) created with `uv` inside the job venv.

```
cmake -S <src> -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DESPRESSO_BUILD_WITH_CUDA=ON -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_HIP_COMPILER=$ROCM_SDK/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="$DEPS_PREFIX;$ROCM_SDK" \
  -DMPI_C_COMPILER=$DEPS_PREFIX/bin/mpicc -DMPI_CXX_COMPILER=$DEPS_PREFIX/bin/mpicxx \
  -DBoost_ROOT=$DEPS_PREFIX -DPython_EXECUTABLE=<py312> \
  -DESPRESSO_BUILD_WITH_WALBERLA=OFF -DESPRESSO_BUILD_WITH_HDF5=OFF \
  -DESPRESSO_BUILD_WITH_SCAFACOS=OFF -DESPRESSO_BUILD_WITH_STOKESIAN_DYNAMICS=OFF \
  -DESPRESSO_BUILD_WITH_GSL=OFF -DESPRESSO_BUILD_WITH_NLOPT=OFF \
  -DESPRESSO_BUILD_WITH_CALIPER=OFF -DESPRESSO_BUILD_WITH_FFTW=ON \
  -DESPRESSO_BUILD_WITH_PYTHON=ON -DESPRESSO_BUILD_TESTS=ON
cmake --build build-hip -j 96
cmake --build build-hip -j 96 --target python_test_data cuda_test
```

Build: PASS (ROCm clang 24.0.0). Cache confirms `Kokkos_ENABLE_CUDA=OFF`, `Kokkos_ENABLE_HIP=OFF`, `Heffte_ENABLE_CUDA=OFF`. New with ROCm 10.1 headers: `-Wunused-value` warnings for discarded `hipError_t` (`nodiscard`) on existing upstream lines (e.g. `GpuParticleData_cuda.cu:240`, `init_cuda.cu:88`); warnings only, not addressed.

Porter smoke (before commit, identical tree): `ctest -L gpu -j 8` 69/74 pass; the 5 failures are `sample_*_with_gpu` / `benchmark_*_with_gpu` FileNotFoundError on `local_samples/*.py` / `local_benchmarks/*.py` (known test-data gap, see "Test data targets"). `cuda_test` and `code_info` pass.

CUDA no-regression compile: NOT run in this round (this host has no nvcc, and a headless fleet worker cannot dispatch to the nvcc host). The CUDA-path changes are CMake-only: the `NOT USE_HIP` guard (USE_HIP defaults OFF) and `Kokkos::kokkos` added to the CUDA branch exactly as upstream has it. It still needs an nvcc compile on the CUDA fleet host before jeff gets the PR.

## Review 2026-11-19 (upstream sync)

Reviewer, linux-gfx942. Scope: `git diff python...moat-port` (python = upstream c0e34ecc0; head 87cbc046d), the merge 06adcac90's conflict resolution, upstream's own delta eafab258..c0e34ecc0 on build and GPU files, and the two new commits a15c89169 and 87cbc046d.

Verdict: review-passed. No blocking defect. The items below are non-blocking; fix them when the PR body/commits are prepared if convenient.

### Non-blocking findings
- `doc/sphinx/installation.rst:305-307` (87cbc046d): "The GPU lattice-Boltzmann and electrokinetics solvers are not available on AMD GPUs, so waLBerla must be disabled" is true but under-states the cost: `ESPRESSO_BUILD_WITH_WALBERLA=OFF` also removes the CPU LB/EK solvers. Suggest "... so waLBerla must be disabled, which also removes the CPU lattice-Boltzmann and electrokinetics solvers."
- `CMakeLists.txt:1134-1139` (pre-existing since the first port, not introduced by this round): with `USE_HIP=ON` and waLBerla left at its default ON, configure stops with `variable CMAKE_CUDA_ARCHITECTURES is undefined`, which does not tell the user to disable waLBerla. Now that the docs name the requirement, a `message(FATAL_ERROR "USE_HIP requires ESPRESSO_BUILD_WITH_WALBERLA=OFF")` in the HIP branch (`CMakeLists.txt:277`) would make the failure self-explaining. Optional.
- Commit 6d36cb6a body still says "The build emits a multi-arch device binary (gfx90a and gfx1100 by default)"; 8e228b5d replaced the default with HIP arch auto-detection. Already-reviewed history, but it is upstream-visible text; correct it in the squash/PR body when the PR is prepared.
- Outstanding (already recorded by the porter, not a review blocker): no nvcc compile of this head. The CUDA-path delta is CMake-only (`AND NOT USE_HIP` at `CMakeLists.txt:801`, USE_HIP defaults OFF; `Kokkos::kokkos` in the CUDA `cuda_test` branch at `src/core/unit_tests/CMakeLists.txt:192-193`, exactly upstream's line), but the rule still needs the nvcc compile before the PR goes to jeff.

### Verified (no action)
- Conflict resolution: replaying the old port diff (`git diff eafab258 8e228b5d`) with `git apply -3` onto `python` and diffing against 06adcac90 leaves exactly two files: `CMakeLists.txt` (the one new `AND NOT USE_HIP` Kokkos guard) and `src/core/unit_tests/CMakeLists.txt` (upstream's `Kokkos::kokkos`, added to both branches). Per-file added/removed lines of the 12 other port files are identical before and after. Nothing dropped or duplicated. `python` and 8e228b5d are both ancestors of moat-port (merge, not rebase).
- Kokkos host-only claim: every `Kokkos::parallel_for/parallel_reduce` in `src/core` (outside walberla) uses `RangePolicy`/`MDRangePolicy` on `DefaultHostExecutionSpace` or an alias of it; all Views/ScatterViews/AoSoA views and the Cabana `ListType` use `HostSpace`; no `DefaultExecutionSpace`, `CudaSpace`, `SharedSpace` anywhere. Keeping `Kokkos_ENABLE_CUDA` off on HIP loses no GPU code. Porter's cache: `Kokkos_ENABLE_CUDA=OFF`, `Kokkos_ENABLE_HIP=OFF`, `Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=OFF`, so `espresso_set_kokkos_properties_on_source_files` is a no-op and no `.cpp` is recompiled as CUDA.
- kokkos-fft: `KokkosFFT_ENABLE_CUFFT=ON` is in the HIP cache, but `_deps/kokkos_fft-src/cmake/KokkosFFT_tpls.cmake:18-20` reads it only under `if(Kokkos_ENABLE_CUDA)`; inert. `Cabana_REQUIRE_CUDA` follows `Kokkos_ENABLE_CUDA` (OFF). `Heffte_ENABLE_CUDA=OFF`.
- Thrust >= 2.5 check, `CUDAToolkit`, and all `CUDA::` targets are only in the CUDA branch / `else()` branches; the HIP branch never reaches them. `espresso_add_cuda_rpaths` (now also called from `cmake/espresso_unit_test.cmake`) is defined empty in the HIP branch.
- a15c89169 (`src/config/CMakeLists.txt:22-26`): the override is directory-scoped after `add_subdirectory(include/config)`; the only later consumer besides `configure_file` is `espresso_set_common_target_properties(espresso_config)`, which returns early for the INTERFACE target, so the commit's "no other CMake logic sees a CUDA compiler" holds. Generated header has `ESPRESSO_CMAKE_CUDA_COMPILER_ID "Clang"`, version "24.0.0"; `code_info.py` (lines 83-90, 107-112) accepts that and allows a `None` host compiler.
- Upstream GPU source delta: `p3m_calc_topology` keeps blocks <= 1024 threads (valid on AMD), `__VA_OPT__` in `KERNELCALL` is fine under clang-HIP, no warp intrinsics / `warpSize` / hardcoded 32 / textures in any `src/core` `.cu`. New GPU test `elc_nonneutral.py` (GPU_SLOTS 1) runs under `-L gpu`; new walberla `.cu` tests are out of scope (waLBerla OFF). `-Werror` applies to CXX and CUDA only, so the new ROCm-10.1 `nodiscard` `-Wunused-value` warnings in HIP TUs cannot fail a warnings-as-errors build.
- Docs: rocThrust needs no CMake hint (headers come from the ROCm include dir that clang-HIP and `hip::hipfft` both add); waLBerla OFF is genuinely required (see finding above); `CMAKE_HIP_COMPILER` is optional on a standard /opt/rocm install but harmless; `|p3m|` substitution and the `:ref:` label resolve (conf.py.in:254, label at installation.rst:298).
- Commits: titles 62/56/42 chars; `[ROCm]` on the two ROCm-only commits, none on the merge; AI disclosure in prose; fenced Test Plans; no Co-Authored-By/noreply trailer; no AMD-internal references; `python3 utils/jargon.py --port espresso` clean; diff is ASCII.

## Validation 2026-11-19 (linux-gfx942, MI300X wave64)

State: port-ready -> completed. GPU: AMD Instinct MI300X (gfx942), HIP_VISIBLE_DEVICES=0. ROCm: pip ROCm SDK 10.1.0 (clang 24.0.0, no /opt/rocm). Validated sha: 87cbc046d27c1eabd33eec8c208014bface17b07 (upstream sync head). Built `-DCMAKE_HIP_ARCHITECTURES=gfx942`, the recipe in "Upstream sync 2026-11-19" above; tree clean at head.

```
cmake --build build-hip -j 96
cmake --build build-hip -j 96 --target python_test_data cuda_test local_samples local_benchmarks
ctest --test-dir build-hip -L gpu --timeout 900 -j 8
cmake --build build-hip -j 96 --target check_unit_tests
ctest --test-dir build-hip -LE gpu --timeout 900 -j 16
```

- GPU suite (`-L gpu`): **74/74 PASS** in one clean run. This includes every core consistency gate from earlier rounds (coulomb_cloud_wall, coulomb_cloud_wall_duplicated, p3m_madelung, p3m_electrostatic_pressure, dawaanr-and-dds-gpu, dipolar_direct_summation, dipolar_interface, coulomb_interface, dipole_field_tracking, gpu_availability), upstream's new and extended GPU tests (elc_nonneutral, the extended p3m_madelung and dipolar_direct_summation), coulomb_tuning, long_range_actors, the GPU checkpoint tests, and, for the first time on this port, the `sample_*_with_gpu` / `benchmark_*_with_gpu` tests. Those need the `local_samples` / `local_benchmarks` targets plus `tqdm` and `matplotlib` in the Python environment; the earlier "fails on CUDA too" note was a missing target and missing packages, not a property of the tests. The samples copy must be rebuilt before each ctest run: a second run fails with "already processed, cannot overwrite".
- `cuda_test` and `code_info` (the new toolchain test fixed by a15c89169): PASS.
- Unit tests (`check_unit_tests`, GTest-free Boost.Test executables, multi-rank MPI included): **136/136 PASS**.
- Non-GPU suite (`-LE gpu`, 403 tests): 374 pass. The 29 failures are all environmental: 26 `tutorial_*` Not Run (tutorial notebooks need Jupyter/nbconvert and the tutorials target), `test_convert` (no `nbformat`), `test_install` (`cmake --install` never run; it diffs files under /usr/local), and `benchmark_mc_acid_base_reservoir`, which needed `pint` and passes on rerun after installing it. No regression.
- Jargon: `utils/jargon.py --port espresso` clean. Docs: "AMD GPU acceleration" in doc/sphinx/installation.rst (87cbc046d).
- CUDA no-regression compile: `cuda-not-validated: this host has no CUDA toolkit (no nvcc, no /opt/conda to create the cuda-12.8 env)`. The CUDA-side delta of this round is CMake-only (`NOT USE_HIP` on the Kokkos CUDA gate; `Kokkos::kokkos` in the CUDA `cuda_test` branch, exactly as upstream has it). It still needs the nvcc compile on the CUDA fleet host before the PR goes to jeff.
- Still owed before PR prep: gfx90a (wave64) and gfx1100 (wave32) were validated at 8e228b5d, not at this head. This run satisfies wave64 at 87cbc046d; wave32 needs a run on an RDNA host.

## Validation 2026-11-19 (linux-gfx1201, RDNA4 wave32)

State: port-ready -> completed. GPU: gfx1201 (`rocminfo`: Wavefront Size 32), HIP_VISIBLE_DEVICES=0. ROCm: pip ROCm SDK 10.1.0 (HIP 7.16.26385, AMD clang 24.0.0, no /opt/rocm). Validated sha: 87cbc046d27c1eabd33eec8c208014bface17b07, the same head the gfx942 run validated. Fork clone at `projects/espresso/src` clean at head throughout.

### Build

Host has no system MPI/Boost/FFTW/hwloc and no CUDA toolkit, same as the gfx942 host. Deps built from source into a job-local prefix `$DEPS_PREFIX` with the versions of "Upstream sync 2026-11-19": hwloc 2.11.2 (`--disable-cairo --disable-opencl --disable-cuda --disable-nvml --disable-rsmi --disable-levelzero`), OpenMPI 5.0.8 (`--with-hwloc=$DEPS_PREFIX --without-cuda --without-rocm --disable-mpi-fortran`), FFTW 3.3.10 double and float (`--enable-shared --enable-openmp --enable-threads`), Boost 1.87.0 (`--with-libraries=mpi,serialization,test,filesystem,system`, b2 `--without-python link=shared`, `using mpi : $DEPS_PREFIX/bin/mpicxx ;`). Python 3.12.3 venv by `uv` inside the job venv: numpy 2.5.5, scipy 1.18.1, Cython 3.3.1, tqdm, matplotlib, pint. `PATH` and `LD_LIBRARY_PATH` carry `$DEPS_PREFIX/bin` and `$DEPS_PREFIX/lib` for build and ctest.

Configure line is the gfx942 one with `-DCMAKE_HIP_ARCHITECTURES=gfx1201` (`ROCM_SDK=$(rocm-sdk path --root)`):

```
cmake --build build-hip -j 24
cmake --build build-hip -j 24 --target python_test_data cuda_test local_samples local_benchmarks
```

Build: PASS, 236/236 plus the test-data targets. Cache: `Kokkos_ENABLE_CUDA=OFF`, `Kokkos_ENABLE_HIP=OFF`, `Heffte_ENABLE_CUDA=OFF`. `llvm-objdump --offloading build-hip/src/core/espresso_core.so`: 4 bundles, all `hip-amdgcn-amd-amdhsa--gfx1201`. Warnings: 20, all the known ROCm 10.1 `nodiscard` `hipError_t` `-Wunused-value`.

### Tests (default myconfig, the gfx942 recipe)

```
ctest --test-dir build-hip -L gpu --timeout 900 -j 8
cmake --build build-hip -j 24 --target check_unit_tests
ctest --test-dir build-hip -LE gpu --timeout 900 -j 16
```

- GPU suite (`-L gpu`): 74/74 with no failure ("100% tests passed"): 71 Passed and 3 reported by ctest as Skipped. All core consistency gates pass, re-run verbose to confirm what ran inside each: coulomb_cloud_wall (5 ran, 1 skipped: SCAFACOS), coulomb_cloud_wall_duplicated, p3m_madelung (6), p3m_electrostatic_pressure, elc_nonneutral, dawaanr-and-dds-gpu, dipolar_direct_summation, dipolar_interface, coulomb_interface, gpu_availability, coulomb_tuning, long_range_actors, the GPU checkpoint tests, `cuda_test` and `code_info`. The default config compiles out DIPOLE_FIELD_TRACKING and MMM1D, so dipole_field_tracking runs 0 of 3 cases there; the maxset run below covers them.
- The 3 ctest Skipped: `sample_p3m_with_gpu` and `benchmark_p3m_with_gpu` run their whole GPU script on import (P3M tuning and timed integration print; `pypresso maintainer/benchmarks/p3m.py --particles_per_core 400 --gpu` run directly exits 0). The test class has no test methods, so Python 3.12's unittest exits 5 ("no tests ran"), which these tests map to `SKIP_RETURN_CODE 5`. Expected; not a GPU fault. `benchmark_lb_with_gpu` raises `FeaturesError: Missing features WALBERLA` (out of scope). A run of `-L gpu` right after another fails these 3 with "already processed, cannot overwrite" until `local_samples local_benchmarks` are rebuilt (already in the notes).
- Unit tests (`check_unit_tests`): 136/136 PASS.
- Non-GPU suite (`-LE gpu`, 403 tests): 346 Passed, 29 exit-5 Skipped (samples and benchmarks whose test is the import, as above), 28 failed, all environmental: 26 `tutorial_*` Not Run (no Jupyter/tutorials target), `test_convert` (`ModuleNotFoundError: No module named 'nbformat'`), `test_install` (diffs files under /usr/local; never installed). Same set as gfx942, less its `pint` failure (pint was installed here up front). No regression.

### Extra wave32 coverage: maxset build

Upstream's maximal feature config, to run GPU cases the default config skips:

```
cmake ... -DESPRESSO_MYCONFIG_FILE=<src>/maintainer/configs/maxset.hpp   # otherwise the same configure line
cmake --build build-max -j 24
cmake --build build-max -j 24 --target python_test_data cuda_test local_samples local_benchmarks
ctest --test-dir build-max -L gpu --timeout 900 -j 8 --output-on-failure
ctest --test-dir build-max -L gpu -E 'sample_|benchmark_' -V --timeout 900 -j 8
```

Build: PASS. GPU suite: 74/74, 0 failures (the same 3 exit-5 Skipped). New cases that ran and passed: dipole_field_tracking test_dds, test_dds_gpu, test_dp3m; dipolar_direct_summation test_dds_gpu, test_inner_loop_consistency_gpu, test_min_image_convention_gpu, the pressure-tensor checks. Every unittest-level skip left in the verbose log names an out-of-scope or not-built feature (WALBERLA/WALBERLA_FFT, MMM1D, STOKESIAN_DYNAMICS, SCAFACOS, SCAFACOS_DIPOLES, H5MD, THERMAL_STONER_WOHLFARTH) or is a mode filter inside the test.

### Gates

- Jargon: `python3 utils/jargon.py --port espresso` clean. Docs: "AMD GPU acceleration" at doc/sphinx/installation.rst:300 and `USE_HIP` at :918.
- CUDA no-regression compile: `cuda-not-validated: this host has no CUDA toolkit (no nvcc, no /usr/local/cuda, no /opt/conda to create the cuda-12.8 env)`. Still owed on the nvcc fleet host before the PR goes to jeff (see gfx942 section).
- With gfx942 (wave64) and this run (wave32) at 87cbc046d, both required Linux gates are met at head; windows is waived.

## CUDA no-regression compile 2026-11-19 (nvcc, mi355 host) -- PORT REGRESSION

Evidence only, requested by jeff before the PR: compile and link the CUDA configuration of the port at head and of upstream at the merge base with real nvcc. No NVIDIA GPU on this host, so compile/link only, nothing run. Stage and platform records untouched; nothing pushed to the fork.

Verdict: **the port breaks the nvcc build.** Upstream base builds 542/542; the port tree stops at `p3m_gpu_cuda.cu`, 501/542 (the remaining 41 steps are `espresso_core.so` and everything that links it). Not fixed here, as instructed.

### Trees

- Port: `AMD-Ecosystem/espresso` at `87cbc046d27c1eabd33eec8c208014bface17b07` (record `head_sha`).
- Base: `espressomd/espresso` at `c0e34ecc0912dbd01467fe71295598579e1a8339` (`git merge-base 87cbc046d origin/python`).
- Both cloned with `--recurse` submodule init; Kokkos 5.0.2, kokkos-fft v2.0.0, Cabana 0.8.0, heFFTe v2.4.1, waLBerla 623a60c1 FetchContent'd by each tree's own configure.

### Toolchain

- CUDA 12.8 (`nvcc` release 12.8, V12.8.93) at /usr/local/cuda, host compiler GCC 13.3.0 (`CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++`), CMake 3.31.6, Ninja.
- Default CUDA archs from upstream's CMake: `ESPRESSO_CMAKE_CUDA_ARCHITECTURES=75;86;89`.
- Host deps built job-locally (the host has no MPI headers and no Boost.MPI): OpenMPI 5.0.8 (`--disable-mpi-fortran --disable-opencl --disable-cuda --disable-nvml --disable-gl --disable-rsmi --disable-levelzero --enable-mca-no-build=accelerator`; the bundled hwloc breaks on `--without-cuda`), and Boost 1.83 `mpi`/`serialization`/`unit_test_framework` compiled from the boostorg `boost-1.83.0` tags against the system 1.83 headers, found through a minimal hand-written `BoostConfig.cmake`. System FFTW 3.3.10.

### Recipe (upstream CI `cuda12-maxset` job, reduced to what this host has)

Upstream's CUDA CI (`.gitlab-ci.yml` `cuda12-maxset` -> `maintainer/CI/build_cmake.sh`) uses `myconfig=maxset`, `ESPRESSO_BUILD_WITH_CUDA=ON`, `ESPRESSO_WARNINGS_ARE_ERRORS=ON`, waLBerla ON, Ninja. Kept those; turned off the optional packages not on the host (ScaFaCoS, Stokesian dynamics, NLopt, Caliper, HDF5, GSL) and the Python bindings (Python 3.12 + NumPy would need a pip install into a venv, which the host's job guard refuses; the bindings are Cython host code that nvcc never compiles). With Python off, `Python_EXECUTABLE` must still be passed because `src/config` runs its generator scripts through it (stdlib only). Identical for both trees:

```
cp <tree>/maintainer/configs/maxset.hpp <build>/myconfig.hpp
CUDACXX=/usr/local/cuda/bin/nvcc CUDAHOSTCXX=/usr/bin/g++ \
cmake -G Ninja -S <tree> -B <build> \
  -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_EXTENSIONS=OFF \
  -D CMAKE_C_COMPILER=/usr/bin/gcc -D CMAKE_CXX_COMPILER=/usr/bin/g++ \
  -D CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -D CUDAToolkit_ROOT=/usr/local/cuda -D Kokkos_CUDA_DIR=/usr/local/cuda \
  -D CMAKE_PREFIX_PATH=$DEPS -D Boost_DIR=$DEPS/lib/cmake/Boost-1.83.0 \
  -D MPI_C_COMPILER=$DEPS/bin/mpicc -D MPI_CXX_COMPILER=$DEPS/bin/mpicxx \
  -D ESPRESSO_WARNINGS_ARE_ERRORS=ON -D ESPRESSO_BUILD_WITH_CUDA=ON \
  -D ESPRESSO_BUILD_WITH_WALBERLA=ON -D ESPRESSO_BUILD_WITH_FFTW=ON \
  -D ESPRESSO_BUILD_WITH_PYTHON=OFF -D Python_EXECUTABLE=/usr/bin/python3.12 \
  -D ESPRESSO_BUILD_TESTS=ON \
  -D ESPRESSO_BUILD_WITH_HDF5=OFF -D ESPRESSO_BUILD_WITH_SCAFACOS=OFF \
  -D ESPRESSO_BUILD_WITH_STOKESIAN_DYNAMICS=OFF -D ESPRESSO_BUILD_WITH_GSL=OFF \
  -D ESPRESSO_BUILD_WITH_NLOPT=OFF -D ESPRESSO_BUILD_WITH_CALIPER=OFF
ninja -C <build> -k 0 -j32 all <the 53 unit-test executables, cuda_test included>
```

The unit tests are `EXCLUDE_FROM_ALL`; their 53 executable targets were listed with `ninja -t targets all` (same list in both trees) and built explicitly, giving 542 Ninja steps per tree.

### Results

| | configure | build steps | first error |
|---|---|---|---|
| base c0e34ecc0 | PASS | **542/542** | none (`espresso_core.so` carries sm_75/86/89 cubins; `cuda_test` links) |
| port 87cbc046d | PASS | **501/542** | `p3m_gpu_cuda.cu(82): error: identifier "cufftComplex" is undefined` |

Both caches: `CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`, `Kokkos_ENABLE_CUDA=OFF` (upstream's own gate keeps the Kokkos CUDA backend off for nvcc; the port's `AND NOT USE_HIP` changes nothing here), `Heffte_ENABLE_CUDA=ON`, waLBerla ON; port has `USE_HIP=OFF`. All other port CUDA TUs (the other six core `.cu` files and all waLBerla CUDA objects) compile under `--Werror=all-warnings`.

Port failure, the only failing edge (`FAILED: src/core/CMakeFiles/espresso_core.dir/electrostatics/p3m_gpu_cuda.cu.o`), 24 errors, all cuFFT names: `cufftComplex`, `cufftHandle`, `cufftDestroy`, `cufftPlan3d`, `CUFFT_R2C`, `CUFFT_C2R`, `CUFFT_SUCCESS`, `cufftExecR2C`, `cufftExecC2R`, plus `name followed by "::" must be a class or namespace name` at lines 305-317.

Cause: the port deleted `#include <cuda.h>` and `#include <cufft.h>` from `src/core/electrostatics/p3m_gpu_cuda.cu` (upstream has `#include <cufft.h>` at line 66) and routed the TU through `src/core/cuda/cuda_to_hip.h` via `cuda/utils.cuh`. That header's NVIDIA branch includes only `<cuda.h>` and `<cuda_runtime.h>`, not `<cufft.h>`, although its comment says it "includes the CUDA runtime and cuFFT" on NVIDIA. Introduced in 6d36cb6aa ("[ROCm] Add HIP support for core electrostatics and magnetostatics", 2026-06-04; `git log -S'#include <cufft.h>'`), so the June port was already broken for nvcc and today's sync did not cause it; the earlier "NVIDIA path byte-identical" claim was never checked with nvcc. Fix belongs to the porter (restore `#include <cufft.h>` on the CUDA path, in the TU or in the header's NVIDIA branch), followed by re-review, an nvcc re-run of this recipe, and whatever HIP revalidation the head move needs.

Job-local artifacts (container-local, not durable): builds and logs under the job directory `espresso/` (`build-{base,port}-wal`, `build-{base,port}.log`, `build-{base,port}2.log`).

## Porter 2026-11-19 (CUDA path fix, linux-gfx950)

Directed by jeff after the nvcc regression above. Host: MI355X (gfx950) with CUDA 12.8 nvcc and pip ROCm SDK (HIP 7.16.26385, clang from `rocm-sdk path --root`), no /opt/rocm. `moat-port` 87cbc046d -> **01090739c** (new commit on top, nothing rewritten; pushed directly, no upstream PR open).

### Fix

01090739c "[ROCm] Restore the cuFFT include for the CUDA build": one line, `#include <cufft.h>` in the NVIDIA (`#else`) branch of `src/core/cuda/cuda_to_hip.h`. Chosen over restoring it in `p3m_gpu_cuda.cu` because the header is new in the port (so this adds no upstream-file diff), the in-file version needs a `!__HIP__` guard (3 lines), and the header's comment already promised cuFFT on NVIDIA. Side effect: on NVIDIA every TU that includes `cuda/utils.cuh` or the header now also sees `<cufft.h>` (header only; linking unchanged, `CUDA::cufft` is still linked only to `espresso_core` as upstream does).

### Audit of the whole port diff (`git diff python...moat-port`) for anything removed or moved off the CUDA build

- Includes removed from CUDA TUs: `<cuda.h>`/`<cuda_runtime.h>` from `common_cuda.cu`, `init_cuda.cu`, `utils.cuh`; `<cuda.h>` from `p3m_gpu_error_cuda.cu`, `dipolar_direct_sum_gpu_cuda.cu`, `GpuParticleData_cuda.cu`, `cuda_test.cu`; `<cuda.h>`+`<cufft.h>` from `p3m_gpu_cuda.cu`. Every one of these TUs includes `cuda/utils.cuh`, which includes `cuda/cuda_to_hip.h`, whose NVIDIA branch now has `<cuda.h>`, `<cuda_runtime.h>`, `<cufft.h>`. `<cufft.h>` was the only header that did not reach the CUDA build. No other definition is removed: the header's NVIDIA branch defines nothing.
- Added on the CUDA path (harmless, recorded so nobody calls it byte-identical again): `CudaHostAllocator.cu` now includes the header (upstream relied on nvcc's implicit `cuda_runtime.h`), so it gains `<cuda.h>`/`<cufft.h>`; `utils.cuh` and `device_qualifier.hpp` test `|| defined(__HIPCC__)` (false under nvcc); `cuda_test.cu` keeps upstream's expected string in the `#else` branch.
- CMake: every hunk is `USE_HIP`-guarded and `USE_HIP` defaults OFF. The one moved item, `OpenMP::OpenMP_CUDA` for `espresso_p3m` (upstream: generator expression in `src/core/p3m/CMakeLists.txt`; port: `${ESPRESSO_OPENMP_CUDA}` set at `src/core/CMakeLists.txt:94`), is set before `add_subdirectory(p3m)` (line 110), so it still reaches the CUDA link. Proof by generated files: normalising source/build paths, the port's CUDA `build.ninja` + `CMakeFiles/*.ninja` against the base tree's (previous job's `build-base-wal`, same recipe) give the same 524 object outputs and an identical set of all 1816 `FLAGS`/`DEFINES`/`INCLUDES`/`LINK_FLAGS`/`LINK_LIBRARIES`/`LINK_PATH` lines. The only differences are the CMake-module path lists of the regeneration rule.
- **Deliberate CUDA-path behaviour change, kept, question for jeff:** 6d36cb6aa default-initialises `P3MGpuFftPlan::forw_plan/back_plan` to 0 and destroys a plan only when non-zero, resetting it to 0 (rule-of-five fault class, "Fault classes hit" item 2). On CUDA this also changes upstream behaviour: upstream calls `cufftDestroy(0)` when no plan was made (the handles are zero-initialized by `make_shared`, not garbage, as the review below verified), and would destroy stale handles if re-planning threw after `free_device_memory()`. (Corrected: this paragraph first claimed uninitialised handles and a routine double destroy at mesh change; both were wrong, the plans are re-created in the same call.) The guard assumes cuFFT never hands out handle 0 for a live plan; if it ever did, that plan would leak on CUDA. It compiles under nvcc with `-Werror` and is arch-unified, as the skill requires for shared-code fault fixes. It could be made HIP-only with a `#if defined(__HIP__)`, which would restore upstream's `cufftDestroy(0)` calls on CUDA. Question: keep it shared (and say so in the PR body as a fix to upstream behaviour), or restrict it to HIP so the CUDA path stays upstream's? Not changed in this round.

### Builds at 01090739c on this host

CUDA (nvcc 12.8.93, GCC 13.3.0, CMake 3.31.6): the recipe of "CUDA no-regression compile 2026-11-19" verbatim (upstream `cuda12-maxset`, maxset myconfig, `ESPRESSO_WARNINGS_ARE_ERRORS=ON`, waLBerla ON, Python OFF), MPI/Boost from that job's prefix, build dir `work/build-cuda`:

```
bash agent_space/cfg-cuda.sh <src> <build>            # = the configure block in that section
ninja -C <build> -k 0 -j42 all <the same 53 unit-test executables>
```

**542/542 steps, 0 FAILED, 0 warnings** (same step count as base c0e34ecc0). `espresso_core.so`: 7 cubins each for sm_75/86/89; links `libcufft.so.11` and `libcudart.so.12`; `cuda_test` links. `Kokkos_ENABLE_CUDA=OFF`, `Heffte_ENABLE_CUDA=ON`, `USE_HIP=OFF`.

HIP (gfx950), default myconfig, the gfx942 recipe with `-DCMAKE_HIP_ARCHITECTURES=gfx950`, `CMAKE_C/CXX_COMPILER=/usr/bin/gcc,g++`, Boost 1.83 from the nvcc job's prefix, Python 3.12 venv by `uv` inside `$FLEET_VENV` (numpy 2.5.5, scipy 1.18.1, Cython 3.3.1, tqdm, matplotlib, pint): build 236/236 plus `python_test_data cuda_test local_samples local_benchmarks`. 28 warnings, all the known `nodiscard` `hipError_t` `-Wunused-value`. `espresso_core.so`: 4 offload bundles, all `hip-amdgcn-amd-amdhsa--gfx950`. Porter smoke `ctest -R 'cuda_test|code_info|p3m_madelung|gpu_availability'`: 4/4 pass.

Host gotcha: the nvcc job's OpenMPI prefix was configured with `--enable-mca-no-build=accelerator`, which drops even the `null` accelerator component. It compiles and links fine, but every `MPI_Init` aborts with "User specified an accelerator component via --mca accelerator ... failed to initialize". For running tests, OpenMPI 5.0.8 was rebuilt job-locally without that flag (`--disable-mpi-fortran --disable-sphinx --disable-opencl --disable-cuda --disable-nvml --disable-gl --disable-rsmi --disable-levelzero`); `ompi_info` then lists `MCA accelerator: null`. Boost.MPI 1.83 built against the other 5.0.8 prefix works with it (same `libmpi.so.40` ABI; `LD_LIBRARY_PATH` puts the new prefix first).

Jargon: `python3 utils/jargon.py --port espresso` clean; `utils/prose.py` on the commit body clean.

## Review 2026-11-19 (CUDA path fix delta)

Reviewer, linux-gfx950. Scope: the delta 87cbc046d..01090739c and its commit message, plus an independent re-audit of the whole port diff (`git diff python...moat-port`, python = upstream c0e34ecc0) for anything removed or moved off the nvcc build (USE_HIP=OFF).

Verdict: review-passed at 01090739c. The fix is correct and the porter's audit holds. The findings below are non-blocking (upstream-visible text and a recorded rationale that is factually wrong); fix them in the squash / PR body when the PR is prepared, under the no-amend rule.

### Non-blocking findings
- Commit 01090739c Test Plan says "HIP on an AMD Instinct MI355X (gfx950), ROCm 7.16". The host's ROCm is the pip ROCm SDK 10.1.0 (`rocm-sdk version` -> 10.1.0); 7.16 is the HIP runtime version (7.16.26385). Write "ROCm 10.1 (HIP 7.16)" in the squash/PR body.
- Rationale of the cuFFT plan-destroy guard (`src/core/electrostatics/p3m_gpu_cuda.cu:111-113, 135-142`) is mis-stated in three places: commit 6d36cb6aa body ("A default-constructed-but-never-initialized holder ... then destroyed garbage handles; CUDA tolerates this but ROCm faults"), "Fault classes hit" item 2 above ("never-initialized holder destroyed garbage handles"), and the porter's question in "Porter 2026-11-19" ("upstream calls `cufftDestroy` on an uninitialised handle ... and destroys the same handles twice when `free_device_memory()` runs at a mesh change and again in the destructor"). Upstream creates the holder only through `std::make_shared<P3MGpuParams>()` (upstream p3m_gpu_cuda.cu:565, the only construction site; `git grep P3MGpuParams python`). That is value-initialization of a class with no user-provided constructor, so upstream's handles are zero-initialized, not garbage. What upstream really does: (a) when no plan was ever made (destructor after an exception in `p3m_gpu_init` before `cufftPlan3d`, or mesh_size 0) it calls `cufftDestroy(0)`; on HIP that is `hipfftDestroy(nullptr)` (hipfftHandle is a pointer), which is the real ROCm hazard; (b) at a mesh change, `free_device_memory()` (upstream :608) destroys the plans and the same call immediately re-creates them (:634-636; `mesh_size > 0` whenever the mesh is positive), so the destructor later destroys the new plans, not the old ones. A double destroy happens only if a `cufftPlan3d` after that free fails and throws. Correct the 6d36cb6aa wording in the squash/PR body ("destroyed null handles when no plan had been created, and stale ones if re-planning threw"), and correct fault item 2 before it is promoted anywhere.
- `src/core/cuda/cuda_to_hip.h:28` comment "This is the only file that knows about HIP" is not true of the port: `cuda/utils.cuh:22`, `utils/device_qualifier.hpp:22`, `unit_tests/cuda_test.cu:198` and the CMake files also test for HIP. Pre-existing since 6d36cb6aa; reword (e.g. "This is the only file that maps CUDA names to HIP") in the same round as the items above.

### Question for jeff: keep the cuFFT plan-destroy guard shared, or HIP-only?
Facts checked: `cufftHandle` is `typedef int` in cufft.h; NVIDIA documents no reserved or invalid handle value, so "cuFFT never returns 0 for a live plan" cannot be proven from the docs, and with no NVIDIA GPU on this host it was not observed either. If cuFFT ever did hand out 0, the guard would leak that one plan (its work area) at teardown; nothing would crash or compute wrongly. Against that, the CUDA behaviour change is small: upstream already starts from 0 (value-initialization, above), so the guard on CUDA only drops the `cufftDestroy(0)` calls (which return an error code that upstream ignores) and makes the throw-after-free path safe. On HIP, hipfftHandle is a pointer and nullptr is a safe sentinel. Recommendation: keep it shared (arch-unified, as the shared-code rule asks) and describe it in the PR body as "skip destroying plans that were never created" with the corrected rationale above; do not make it HIP-only. If jeff wants zero reliance on 0 being invalid on CUDA, the alternative is an explicit `bool has_plans` set after both `cufftPlan3d` calls succeed and checked in `free_device_memory()`, which removes the sentinel question on both backends. Not blocking this review.

### Verified (no action)
- Delta 01090739c: one line, `#include <cufft.h>` in the `#else /* NVIDIA */` branch of `src/core/cuda/cuda_to_hip.h:94`; HIP branch untouched, so the HIP build is unaffected. Title 51 chars, `[ROCm]` justified (the header exists only for the ROCm port and the fix repairs that port's regression), AI disclosure in prose, fenced Test Plan, no Co-Authored-By/noreply trailer, ASCII, `python3 utils/jargon.py --port espresso` clean.
- Include audit, redone: the eight removed `<cuda.h>`/`<cuda_runtime.h>`/`<cufft.h>` includes are in `common_cuda.cu`, `init_cuda.cu`, `utils.cuh`, `p3m_gpu_cuda.cu`, `p3m_gpu_error_cuda.cu`, `dipolar_direct_sum_gpu_cuda.cu`, `GpuParticleData_cuda.cu`, `unit_tests/cuda_test.cu`. Every one of those TUs includes `cuda/utils.cuh` (common_cuda.cu:22, init_cuda.cu:21, p3m_gpu_cuda.cu:55, p3m_gpu_error_cuda.cu:29, dipolar_direct_sum_gpu_cuda.cu:26, GpuParticleData_cuda.cu:33, cuda_test.cu:27), and always earlier in the TU than the removed include stood, so nothing is now seen later than before. utils.cuh includes the header, whose NVIDIA branch now provides all three. The NVIDIA branch defines no macro or alias.
- CMake audit: every port hunk is behind `USE_HIP` (default OFF) except the moved `OpenMP::OpenMP_CUDA`. `ESPRESSO_OPENMP_CUDA` is set at `src/core/CMakeLists.txt:93-95` under `ESPRESSO_BUILD_WITH_CUDA AND NOT USE_HIP` (upstream's genex condition was `ESPRESSO_BUILD_WITH_CUDA`), before `add_subdirectory(p3m)` at :110, and `src/core/p3m/CMakeLists.txt:46` is its only other reader. `src/walberla_bridge` keeps upstream's genex unchanged.
- Generated-file identity claim, re-derived independently: from `work/build-cuda` (01090739c, `USE_HIP=OFF`, `CMAKE_HOME_DIRECTORY` = this fork clone, nvcc) and the other job's `build-base-wal` (c0e34ecc0), after normalising source, build and dependency-prefix paths, all 1842 `FLAGS`/`DEFINES`/`INCLUDES`/`LINK_FLAGS`/`LINK_LIBRARIES`/`LINK_PATH`/`LANGUAGE_COMPILE_FLAGS` lines are identical, the 576 object/library outputs are identical, and the per-output rule plus source (73 `CUDA_COMPILER` edges) is identical. (The porter's 1816/524 counts differ only in what was grepped.)
- Build evidence: `work/build-cuda.log` ends `[542/542] Linking CXX executable src/core/unit_tests/p3m_test_matrix`, 0 `FAILED`, 0 `warning` (warnings are errors); `ninja -n all` on the build dir shows only the always-run version/git-info steps, so the build is current with the checked-out tree. Fork clone clean at 01090739c.

## Validation 2026-11-19 (linux-gfx950, MI355X wave64)

State: port-ready -> completed. GPU: AMD Instinct MI355X (gfx950), `HIP_VISIBLE_DEVICES=5`, `ulimit -c 0`. ROCm: pip ROCm SDK 10.1.0 (HIP 7.16.26385, AMD clang from `rocm-sdk path --root`, no /opt/rocm). Validated sha: 01090739cf852d70549a5699c5984b0f17414214. Fork clone `projects/espresso/src` clean at head. Build: the HIP build of "Porter 2026-11-19 (CUDA path fix)" (default myconfig, `-DCMAKE_HIP_ARCHITECTURES=gfx950`, 236/236, 4 offload bundles all gfx950), the job-local OpenMPI 5.0.8 first on `PATH`/`LD_LIBRARY_PATH`.

```
cmake --build build-hip -j 42 --target local_samples local_benchmarks   # before each ctest run
ctest --test-dir build-hip -L gpu --timeout 900 -j 8
cmake --build build-hip -j 42 --target check_unit_tests
ctest --test-dir build-hip -LE gpu --timeout 900 -j 16
```

- GPU suite (`-L gpu`): **74/74, 0 failed** ("100% tests passed"): 71 Passed, 3 ctest Skipped by exit code 5 (`sample_p3m_with_gpu`, `benchmark_p3m_with_gpu`: script runs on import, no test methods; `benchmark_lb_with_gpu`: WALBERLA not built), the same three the gfx1201 run explains. Every core consistency gate passed (coulomb_cloud_wall, coulomb_cloud_wall_duplicated, p3m_madelung, p3m_electrostatic_pressure, elc_nonneutral, dawaanr-and-dds-gpu, dipolar_direct_summation, dipolar_interface, coulomb_interface, dipole_field_tracking, gpu_availability, coulomb_tuning, long_range_actors, GPU checkpoints, `cuda_test`, `code_info`).
- Unit tests (`check_unit_tests`): **136/136 PASS**.
- Non-GPU suite (`-LE gpu`, 403 tests): 346 Passed, 29 exit-5 Skipped, 28 failed, all environmental and the same set as gfx1201: 26 `tutorial_*` Not Run (no Jupyter/tutorials target), `test_convert` (`ModuleNotFoundError: No module named 'nbformat'`), `test_install` (diffs under /usr/local; never installed). No regression.
- **CUDA no-regression gate at 01090739c: PASS.** nvcc 12.8.93 (/usr/local/cuda), GCC 13.3.0, upstream `cuda12-maxset` recipe of "CUDA no-regression compile 2026-11-19" (maxset, warnings as errors, waLBerla ON, Python OFF): **542/542 Ninja steps, 0 FAILED, 0 warnings**, equal to upstream base c0e34ecc0 (542/542); 87cbc046d stopped at 501/542. Generated CUDA build files identical to the base's after path normalisation (porter section; independently re-derived by the review). Compile/link only; no NVIDIA GPU on this host, so no NVIDIA runtime claim.
- Jargon: `python3 utils/jargon.py --port espresso` clean. Docs: "AMD GPU acceleration" in `doc/sphinx/installation.rst` (87cbc046d), unchanged.
- Gates: wave64 is satisfied at 01090739c by this run. gfx942 and gfx1201 validated 87cbc046d; the delta 87cbc046d..01090739c is one line in the NVIDIA-only `#else` branch of `cuda_to_hip.h`, so the HIP preprocessed sources are unchanged; wave32 (gfx1201) owes revalidation or a carry-forward at this head (not this host's arch).
- Open for jeff before PR prep: the plan-destroy guard question (porter and review sections), and the review's three wording fixes for the squash/PR body (ROCm 10.1 not 7.16 in 01090739c's Test Plan; corrected guard rationale in 6d36cb6aa's body; "only file that knows about HIP" comment in `cuda_to_hip.h:28`).

## Validation 2026-11-19 (linux-gfx1201, RDNA4 wave32, revalidate at 01090739c)

State: revalidate -> completed. GPU: gfx1201 (`rocminfo`: Wavefront Size 32), `HIP_VISIBLE_DEVICES=1`, `ulimit -c 0`. ROCm: pip ROCm SDK 10.1.0 (`rocm-sdk version`; HIP 7.16.26385, AMD clang 24.0.0 from `rocm-sdk path --root`, no /opt/rocm). Validated sha: 01090739cf852d70549a5699c5984b0f17414214 (the CUDA path fix head). Fresh fork clone at `projects/espresso/src`, clean at head throughout. Full GPU rerun rather than a carry-forward, at jeff's request (`classify 87cbc046d 01090739c` -> `mixed`, `cuda_to_hip.h` token count differs).

### Build

Host deps rebuilt from source into a job-local prefix with the recipe and versions of the earlier gfx1201 section (hwloc 2.11.2, OpenMPI 5.0.8, FFTW 3.3.10 double+float, Boost 1.87.0 mpi/serialization/test/filesystem/system, `--without-python`). Python 3.12.3 venv by `uv` inside the job venv: numpy 2.5.5, scipy 1.18.1, Cython 3.3.1, tqdm, matplotlib, pint. `$DEPS_PREFIX/bin` and `$DEPS_PREFIX/lib` on `PATH`/`LD_LIBRARY_PATH` for build and ctest. Configure line: the gfx942 one in "Upstream sync 2026-11-19" with `-DCMAKE_HIP_ARCHITECTURES=gfx1201`, `-DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++`.

```
cmake --build build-hip -j 24
cmake --build build-hip -j 24 --target python_test_data cuda_test local_samples local_benchmarks
```

Build: PASS, 236/236 plus the test-data targets. 20 warnings, all the known ROCm 10.1 `nodiscard` `hipError_t` `-Wunused-value`. Cache: `Kokkos_ENABLE_CUDA=OFF`, `Kokkos_ENABLE_HIP=OFF`, `Heffte_ENABLE_CUDA=OFF`. `llvm-objdump --offloading build-hip/src/core/espresso_core.so`: 4 bundles, all `hip-amdgcn-amd-amdhsa--gfx1201`.

### Tests (default myconfig)

```
ctest --test-dir build-hip -L gpu --timeout 900 -j 8 --output-on-failure
cmake --build build-hip -j 12 --target check_unit_tests
cmake --build build-hip -j 24 --target local_samples local_benchmarks
ctest --test-dir build-hip -LE gpu --timeout 900 -j 16
```

- GPU suite (`-L gpu`): **74/74, 0 failed** ("100% tests passed"): 71 Passed, 3 ctest Skipped by exit code 5 (`sample_p3m_with_gpu`, `benchmark_p3m_with_gpu`: script runs on import, no test methods; `benchmark_lb_with_gpu`: WALBERLA not built), the same three as every earlier run. All core gates passed: coulomb_cloud_wall, coulomb_cloud_wall_duplicated, p3m_madelung, p3m_electrostatic_pressure, p3m_tuning_exceptions, elc, elc_nonneutral, elc_vs_analytic, dawaanr-and-dds-gpu, dipolar_direct_summation, dipolar_interface, coulomb_interface, dipole_field_tracking, gpu_availability, coulomb_tuning, long_range_actors, the GPU save/test checkpoint pairs.
- `cuda_test` and `code_info`: PASS.
- Unit tests (`check_unit_tests`): **136/136 PASS**.
- Non-GPU suite (`-LE gpu`, 403 tests): 346 Passed, 29 exit-5 Skipped, 28 failed, all environmental and the same set as the earlier gfx1201 and gfx950 runs: 26 `tutorial_*` Not Run (no Jupyter/tutorials target), `test_convert` (`ModuleNotFoundError: No module named 'nbformat'`), `test_install` (`diff: /usr/local/lib/python3.12/site-packages/espressomd/...: No such file or directory`; never installed). No regression.

### Extra wave32 coverage: maxset build

```
cmake ... -DESPRESSO_MYCONFIG_FILE=<src>/maintainer/configs/maxset.hpp   # otherwise the same configure line
cmake --build build-max -j 12
cmake --build build-max -j 12 --target python_test_data cuda_test local_samples local_benchmarks
ctest --test-dir build-max -L gpu --timeout 900 -j 8 --output-on-failure
ctest --test-dir build-max -L gpu -E 'sample_|benchmark_' -V --timeout 900 -j 8
```

Build: PASS. GPU suite: **74/74, 0 failed** (the same 3 exit-5 Skipped); verbose run 68/68. Cases the default config compiles out ran and passed: dipole_field_tracking test_dds, test_dds_gpu, test_dp3m; dipolar_direct_summation test_dds_gpu, test_inner_loop_consistency_gpu, test_min_image_convention_gpu. Every unittest-level skip names an out-of-scope or not-built feature (WALBERLA/WALBERLA_FFT, MMM1D, STOKESIAN_DYNAMICS, SCAFACOS, SCAFACOS_DIPOLES, H5MD, THERMAL_STONER_WOHLFARTH) or is a mode filter inside the test, the same profile as the 87cbc046d run.

### Gates

- Jargon: `python3 utils/jargon.py --port espresso` clean (needs the fork's `python` branch fetched in the clone; a fresh `-b moat-port` clone reports "cannot resolve python..moat-port" until it is). Docs: "AMD GPU acceleration" at doc/sphinx/installation.rst:300, `USE_HIP` at :918.
- CUDA no-regression compile: not rerun; already PASS at 01090739c (542/542, nvcc 12.8.93, gfx950 section). This host has no CUDA toolkit.
- With gfx950 (wave64) and this run (wave32) at 01090739c, both required Linux gates are met at head; windows is waived. The questions for jeff listed at the end of the gfx950 section are unchanged.

## Validation 2026-11-19 (linux-gfx942, MI300X wave64, revalidate at 01090739c)

State: revalidate -> completed. GPU: AMD Instinct MI300X (gfx942, `rocminfo` Wavefront Size 64), `HIP_VISIBLE_DEVICES=1`, `ulimit -c 0`. ROCm: pip ROCm SDK 10.1.0 (`rocm-sdk version`; HIP 7.16.26385, AMD clang 24.0.0, no /opt/rocm). Validated sha: 01090739cf852d70549a5699c5984b0f17414214, a fresh fork clone at `projects/espresso/src`, clean at head throughout. Full run, no carry-forward (the classifier calls 87cbc046d..01090739c `mixed`, and jeff asked for the real tests).

Build: the recipe of "Upstream sync 2026-11-19 (Host build recipe)". Deps were rebuilt from source into this job's own prefix (hwloc 2.11.2, OpenMPI 5.0.8, FFTW 3.3.10 double+float omp/threads, Boost 1.87.0 mpi/serialization/test/filesystem/system `--without-python link=shared`), with the configure flags listed in the gfx1201 section. Python 3.12.3 venv by `uv` inside the job venv: numpy 2.5.5, scipy 1.18.1, Cython 3.3.1, tqdm, matplotlib, pint. Configure line as recorded, with `-DCMAKE_HIP_ARCHITECTURES=gfx942`.

```
cmake --build build-hip -j 96
cmake --build build-hip -j 96 --target python_test_data cuda_test local_samples local_benchmarks
ctest --test-dir build-hip -L gpu --timeout 900 -j 8
cmake --build build-hip -j 96 --target check_unit_tests
cmake --build build-hip -j 96 --target local_samples local_benchmarks
ctest --test-dir build-hip -LE gpu --timeout 900 -j 16
```

- Build: PASS, 236/236 plus the test-data targets. Cache: `Kokkos_ENABLE_CUDA=OFF`, `Kokkos_ENABLE_HIP=OFF`, `Heffte_ENABLE_CUDA=OFF`. `llvm-objdump --offloading build-hip/src/core/espresso_core.so`: 4 bundles, all `hip-amdgcn-amd-amdhsa--gfx942`. 20 warnings, all the known ROCm 10.1 `nodiscard` `hipError_t` `-Wunused-value`.
- GPU suite (`-L gpu`): **74/74, 0 failed** ("100% tests passed", 297.5 s): 71 Passed, 3 ctest Skipped by exit code 5. The 3 are the same ones the gfx950 and gfx1201 runs explain: `sample_p3m_with_gpu` and `benchmark_p3m_with_gpu` run the whole script on import and have no test methods (re-run with `-V`: P3M tuning output, then "Ran 0 tests ... NO TESTS RAN"), and `benchmark_lb_with_gpu` needs WALBERLA, which is not built. Every core consistency gate passed: coulomb_cloud_wall, coulomb_cloud_wall_duplicated, p3m_madelung, p3m_electrostatic_pressure, elc_nonneutral, dawaanr-and-dds-gpu, dipolar_direct_summation, dipolar_interface, coulomb_interface, dipole_field_tracking, gpu_availability, coulomb_tuning, long_range_actors, and the 4 GPU checkpoint save/test cases. The earlier gfx942 run at 87cbc046d wrote "74/74 PASS". This run is the same result, with ctest's exit-5 Skipped shown separately.
- `cuda_test` (pinned-host allocator) PASS in both `check_unit_tests` and `-LE gpu`. `code_info` PASS.
- Unit tests (`check_unit_tests`): **136/136 PASS**.
- Non-GPU suite (`-LE gpu`, 403 tests): 346 Passed, 29 exit-5 Skipped, 28 failed, all environmental. That is the same set as gfx950 and gfx1201: 26 `tutorial_*` Not Run (no Jupyter, no tutorials target), `test_convert` (`ModuleNotFoundError: No module named 'nbformat'`) and `test_install` (`diff: /usr/local/lib/python3.12/site-packages/espressomd/...: No such file or directory`; it was never installed). No regression.
- CUDA no-regression gate: already recorded at 01090739c (nvcc 12.8.93, 542/542, "Validation 2026-11-19 (linux-gfx950)"), so it is not repeated here. This host has no CUDA toolkit anyway.
- Jargon: `python3 utils/jargon.py --port espresso` clean (needed `git -C projects/espresso/src branch python origin/python` in a fresh clone). Docs: "AMD GPU acceleration" at doc/sphinx/installation.rst:300 and `USE_HIP` at :918.
- Nothing was edited in the fork and nothing was pushed to it.

## Ruling 2026-11-19 (jeff): one polish round before the PR

jeffdaily ruled on 2026-11-19 that four small things a maintainer would comment on in the port diff are fixed in one more round before the upstream PR is sent, with no advance contact to upstream: (a) rename `USE_HIP` to upstream's `ESPRESSO_` option naming, (b) a clear early configure error when the HIP option is on and waLBerla is left on, (c) correct the "only file that knows about HIP" comment in `cuda_to_hip.h`, (d) say in the user guide that disabling waLBerla also removes the CPU LB/EK solvers. Nothing else. No upstream PR is open, so `moat-port` was pushed directly.

## Porter 2026-11-19 (polish round, linux-gfx950)

Host: MI355X (gfx950), CUDA 12.8 nvcc (V12.8.93) at /usr/local/cuda, pip ROCm SDK 10.1.0 (HIP 7.16.26385, clang from `rocm-sdk path --root`), GCC 13.3.0, CMake 3.31.6. `moat-port` 01090739c -> **4861f4839** "[ROCm] Rename the HIP build option and clarify its requirements" (one new commit on top, nothing rewritten).

### Decisions

- **Name `ESPRESSO_BUILD_WITH_HIP`.** Every external-feature option is `ESPRESSO_BUILD_WITH_<X>` (`_CUDA`, `_FFTW`, `_WALBERLA`, and the modifier `_WALBERLA_AVX`), declared with `option()` in the "User input options" block and listed in installation.rst "Options and Variables"; the option stays declared right after `ESPRESSO_BUILD_WITH_CUDA` and keeps its description. Alternatives considered and not taken: a compiler-selector string option (upstream removed `ESPRESSO_CUDA_COMPILER` in favour of `CUDACXX`, CHANGELOG #4642, so a selector cuts against their direction) and `ESPRESSO_BUILD_WITH_CUDA_HIP` (modifier style like `_WALBERLA_AVX`, but reads poorly). No compatibility alias for `USE_HIP`: it was never in an upstream release.
- **No C++ definition renamed or added.** Nothing derives a macro from the option: `cmake/espresso_cmake_config.cmakein` lists its `#cmakedefine ESPRESSO_BUILD_WITH_*` lines explicitly and the port never added one; the sources (`cuda_to_hip.h`, `cuda/utils.cuh`, `utils/device_qualifier.hpp`, `unit_tests/cuda_test.cu`) test the compiler macros `__HIP__` / `__HIPCC__`, which is what they need (whether this TU is being compiled by clang-HIP), so they do not change. The CUDA feature reported to Python is unaffected.
- **waLBerla check** is the first statement in the HIP branch (`CMakeLists.txt:279-284`), before `enable_language(HIP)` and before any FetchContent; it only errors and never touches the cache. Verified: `cmake -G Ninja -S <src> -B build-err -DESPRESSO_BUILD_WITH_CUDA=ON -DESPRESSO_BUILD_WITH_HIP=ON -DPython_EXECUTABLE=/usr/bin/python3.12` -> `CMake Error at CMakeLists.txt:280 (message): ESPRESSO_BUILD_WITH_HIP requires ESPRESSO_BUILD_WITH_WALBERLA=OFF, because the waLBerla GPU kernels cannot be compiled with HIP`; cache keeps `ESPRESSO_BUILD_WITH_WALBERLA:BOOL=ON`.
- **cmake-format output included (beyond the four items, formatter only).** Upstream's base c0e34ecc0 is clean under its pinned formatters (cmake-format 0.6.13 over every CMakeLists.txt, clang-format 22.1 over the GPU sources: empty diff), and its GitLab `style` job (`maintainer/CI/fix_style.sh` -> pre-commit) fails on any patch. 01090739c already failed the `cmake-format` hook on three port-added spots (blank line in the empty `espresso_add_cuda_rpaths` macro, the `LANGUAGE HIP` wrap, a comment reflow in `src/core/CMakeLists.txt`), and the rename makes more lines too long. The commit carries the formatter's output for the touched CMake files; nothing hand-edited beyond the four items.

### Style, docs and builds at 4861f4839

- Formatters (job venv by `uv`: cmakelang 0.6.13, clang-format 22.1.8, pre-commit): `pre-commit run --files <the 17 port files>` -> clang-format Passed, cmake-format Passed (autopep8, ex-flags, pylint: no files). Same command on 01090739c: cmake-format Failed.
- Docs: `cmake --build build-hip --target sphinx` (the target runs `sphinx -q -W`, warnings as errors), Sphinx 9.1.1 with upstream's requirements.txt sphinx block plus `ase`, `zndraw~=0.6.0`, `asebytes==0.1.7` (autodoc imports `espressomd.plugins.ase` and `espressomd.zn`; without them the build fails on import warnings, an environment gap): **PASS**, rendered installation.html carries the new option name and sentence. Not run: `maintainer/CI/dox_warnings.sh` (the doxygen part of the style job; the touched C++ comment is not a doxygen block).
- **CUDA, nvcc: 542/542 steps, 0 FAILED, 0 warnings** (warnings are errors), equal to base c0e34ecc0. Recipe of "CUDA no-regression compile 2026-11-19" verbatim (`agent_space/cfg-cuda.sh`, MPI/Boost from the nvcc job's prefix), `ninja -C build-cuda -k 0 -j64 all <the 53 unit-test executables>`. Cache: `ESPRESSO_BUILD_WITH_HIP=OFF`, `Kokkos_ENABLE_CUDA=OFF`, `Heffte_ENABLE_CUDA=ON`. `cuobjdump --list-elf espresso_core.so`: 7 cubins each for sm_75/86/89.
- **HIP gfx950: 236/236** plus `python_test_data cuda_test local_samples local_benchmarks`, the configure line of "Porter 2026-11-19 (CUDA path fix)" with `-DESPRESSO_BUILD_WITH_HIP=ON` and Python from this job's own py312 venv (numpy 2.5.5, scipy 1.18.1, Cython 3.3.1, tqdm, matplotlib, pint, sphinx stack). 28 warnings, all the known ROCm 10.1 `nodiscard` `hipError_t` `-Wunused-value`. `espresso_core.so`: 4 offload bundles, all gfx950. Cache: `Kokkos_ENABLE_CUDA/HIP=OFF`, `Heffte_ENABLE_CUDA=OFF`. Porter smoke (`HIP_VISIBLE_DEVICES=3`): `ctest -R 'cuda_test|code_info|p3m_madelung|gpu_availability'` 4/4.
- Gates: `utils/jargon.py --port espresso` clean, `utils/prose.py` on the commit body clean, fork clone clean at head.

## Review 2026-11-19 (polish round delta)

Verdict: **review-passed** at 4861f4839 (delta 01090739c..4861f4839, reviewer on linux-gfx950). No blocking defect. All four ruled items are present and nothing outside them changed except cmake-format output, which I confirmed is formatter-only.

What I checked myself:
- (a) Rename: `git grep USE_HIP 4861f4839` finds nothing in the fork tree. `ESPRESSO_BUILD_WITH_HIP` follows the `ESPRESSO_BUILD_WITH_<X>` naming of `_CUDA`/`_FFTW`/`_WALBERLA` and is listed in installation.rst beside them. No C++ macro needed renaming: `cmake/espresso_cmake_config.cmakein` lists its `#cmakedefine ESPRESSO_BUILD_WITH_*` lines explicitly (no HIP line, so `check_myconfig.py` and `featuredefs.cmakedefs` are unaffected), and the sources test only `__HIP__`/`__HIPCC__`.
- (b) waLBerla check: I configured head into `work/review-err` with only `-DESPRESSO_BUILD_WITH_CUDA=ON -DESPRESSO_BUILD_WITH_HIP=ON`, with no ROCm prefix or Python hints. It fails at `CMakeLists.txt:280` with the new message, before `enable_language(HIP)` and `find_package(hipfft)`, and the cache keeps `ESPRESSO_BUILD_WITH_WALBERLA:BOOL=ON`. With the full ROCm prefix (`work/review-err2`) the same error appears. I also checked the commit message's "used to stop" claim: a `git archive` of 01090739c configured with `-DUSE_HIP=ON` and the same prefix (`work/review-old`) stops at `CMakeLists.txt:1138` with "variable CMAKE_CUDA_ARCHITECTURES is undefined". The claim is accurate.
- (c) Comment: the other HIP-aware files (`cuda/utils.cuh`, `utils/device_qualifier.hpp`, `unit_tests/cuda_test.cu`) only test `__HIPCC__`/`__HIP__` or include the shim. None of them aliases a CUDA name, so "the only file that maps CUDA names to HIP" is true.
- (d) Docs: the sentence matches installation.rst:744 (`WALBERLA` enables the lattice-Boltzmann and EK solvers). The `sphinx -W` pass is in the porter's evidence (`work/sphinx.log`).
- cmake-format: with upstream's `.cmake-format` and cmakelang 0.6.13 (`venv/fmt`), base `python` is clean on all four touched CMake files, 01090739c is dirty in `CMakeLists.txt` and `src/core/CMakeLists.txt`, and head is clean. I compared the tokens of head against 01090739c with `USE_HIP` substituted, ignoring whitespace and comment markers. The only difference is the new waLBerla block, so the remaining changes are layout produced by the formatter, and they are justified by upstream's pre-commit style job.
- CUDA path (option OFF): every changed condition is the same expression with the new name, and the new error sits inside the HIP-only branch, so OFF behaves exactly as before. The porter's nvcc evidence (542/542, `work/build-cuda.log`) covers the compile.
- Commit message: the title is 63 characters, and `[ROCm]` is justified because the change touches only the HIP option. The AI disclosure is in prose, with no Co-Authored-By and no noreply address. The message is ASCII, the fenced Test Plan uses the new option name, and `python3 utils/jargon.py --port espresso` is clean.

Non-blocking, for whoever drafts the upstream PR body:
- The Test Plan's CUDA block starts with `cp maintainer/configs/maxset.hpp build-cuda/myconfig.hpp`, which runs before anything creates `build-cuda`, so it fails as written ("No such file or directory"). The 01090739c message has the same line. Fixing it in the commit would mean rewriting history, so do not; add `mkdir -p build-cuda` in the PR body's Test Plan. In the same block, the `pre-commit run --files` line names 3 of the 6 touched files. The command is correct but narrower than the change.
- Records: the `## CMake option` section near the top of this file still says `option(USE_HIP ...)` and mentions a gfx90a default (which 8e228b5d0 already dropped). It is a stale project record, not fork content.

## Validation 2026-11-19 (linux-gfx950, MI355X wave64, polish round 4861f4839)

State: revalidate -> completed. GPU: AMD Instinct MI355X (gfx950), `HIP_VISIBLE_DEVICES=3`, `ulimit -c 0`. ROCm: pip ROCm SDK 10.1.0 (HIP 7.16.26385, clang from `rocm-sdk path --root`, no /opt/rocm). Validated sha: 4861f483930b774eccb14a26125de323d1a33851, fork clone `projects/espresso/src` clean at head. Builds: the porter's `build-hip` of "Porter 2026-11-19 (polish round)" (default myconfig, gfx950, 236/236) and a new `build-max`, same configure line plus `-DESPRESSO_MYCONFIG_FILE=<src>/maintainer/configs/maxset.hpp` (236/236 plus `python_test_data cuda_test`; 30 warnings, all the known ROCm 10.1 `nodiscard` `-Wunused-value`; 4 offload bundles, all gfx950). OpenMPI 5.0.8 runtime prefix and Boost 1.83 from the earlier jobs on this host, first on `PATH`/`LD_LIBRARY_PATH`.

```
cmake --build <build> -j 32 --target local_samples local_benchmarks   # before each ctest run
ctest --test-dir build-hip -L gpu --timeout 900 -j 8
cmake --build build-hip -j 48 --target check_unit_tests
ctest --test-dir build-max -L gpu --timeout 900 -j 8 --output-on-failure
ctest --test-dir build-max -L gpu -E 'sample_|benchmark_' -V --timeout 900 -j 8
cmake --build build-max -j 48 --target check_unit_tests
ctest --test-dir build-hip -LE gpu --timeout 900 -j 16
```

Counts are passed / skipped / failed (ctest's "Skipped" is the exit-5 mapping; earlier records wrote 71 + 3 as "74/74").

- GPU suite, default myconfig (`-L gpu`, 74 tests): **71 passed / 3 skipped / 0 failed**. Skipped: `sample_p3m_with_gpu`, `benchmark_p3m_with_gpu` (script runs on import, no test methods, exit 5), `benchmark_lb_with_gpu` (WALBERLA not built). The same three as every earlier run.
- GPU suite, maxset myconfig (`-L gpu`, 74 tests): **71 passed / 3 skipped / 0 failed**, the same three skipped. **First maxset run on an Instinct (wave64) GPU**; it had run only on gfx1201. Verbose rerun without samples/benchmarks: 68/68 passed. Cases the default config compiles out ran and passed: dipole_field_tracking test_dds, test_dds_gpu; dipolar_direct_summation test_dds_gpu, test_inner_loop_consistency_gpu, test_min_image_convention_gpu; dawaanr-and-dds-gpu test_dds_gpu, test_dds_mixed_particles, test_dds_replica_cpu, test_dp3m_dlc/metallic/non_metallic. Every unittest-level skip names a feature not built or out of scope (WALBERLA, WALBERLA_FFT, MMM1D, STOKESIAN_DYNAMICS, SCAFACOS, SCAFACOS_DIPOLES, THERMAL_STONER_WOHLFARTH, H5MD) or is a mode or combination filter inside the test.
- Unit tests (`check_unit_tests`): default **136 passed / 0 skipped / 0 failed**; maxset **136 passed / 0 skipped / 0 failed**.
- Non-GPU suite, default (`-LE gpu`, 403 tests): 346 passed / 29 skipped (exit 5) / 28 failed, all environmental and the same set as every earlier run: 26 `tutorial_*` Not Run (no Jupyter/tutorials target), `test_convert` (no `nbformat`), `test_install` (never installed). No regression.
- CUDA no-regression at this head: PASS, nvcc 12.8.93, 542/542 steps, 0 FAILED, 0 warnings (porter section above).
- Jargon: `python3 utils/jargon.py --port espresso` clean. Docs: "AMD GPU acceleration" in `doc/sphinx/installation.rst`; the `sphinx` target (`-W`) builds.
- Gates: wave64 satisfied at 4861f4839 by this run. wave32 (gfx1201) validated 01090739c; the delta is CMake option rename + a configure-time error + a comment + docs + cmake-format, so compiled output with the option on should be unchanged, but it is `mixed` to the classifier: wave32 owes a revalidation or a binary-equivalence carry-forward on an RDNA host. windows is waived.

## Validation 2026-11-19 (linux-gfx1201, RDNA4 wave32, polish round 4861f4839)

State: revalidate -> completed. GPU: AMD Radeon AI PRO R9700 (gfx1201, wave32): `amd-smi static -g 1 --asic` MARKET_NAME "AMD Radeon AI PRO R9700", DEVICE_ID 0x7551, 64 CUs, TARGET_GRAPHICS_VERSION gfx1201; `rocminfo` Marketing Name the same; ESPResSo's own `CudaInitHandle().list_devices()` reports `{0: 'AMD Radeon AI PRO R9700'}`, 34208743424 bytes. `HIP_VISIBLE_DEVICES=1`, `ulimit -c 0`. ROCm: pip ROCm SDK 10.1.0 (HIP 7.16.26385, AMD clang 24.0.0 from `rocm-sdk path --root`, no /opt/rocm). Validated sha: 4861f483930b774eccb14a26125de323d1a33851, fresh fork clone at `projects/espresso/src`, clean at head throughout. Full GPU run, no carry-forward, at jeff's request (`classify 01090739c 4861f4839` -> `mixed`: `CMakeLists.txt` token count differs from the new waLBerla check, the rest rename-only or comment-only).

Build: deps rebuilt from source into this job's prefix with the recipe of "Validation 2026-11-19 (linux-gfx1201, RDNA4 wave32)" (hwloc 2.11.2, OpenMPI 5.0.8, FFTW 3.3.10 double+float, Boost 1.87.0 mpi/serialization/test/filesystem/system `--without-python`). Python 3.12.3 venv by `uv` inside the job venv: numpy 2.5.5, scipy 1.18.1, Cython 3.3.1, tqdm, matplotlib, pint. Configure line: the gfx942 one in "Upstream sync 2026-11-19" with `-DESPRESSO_BUILD_WITH_HIP=ON` in place of `-DUSE_HIP=ON`, `-DCMAKE_HIP_ARCHITECTURES=gfx1201`, `-DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++`; `build-max` adds `-DESPRESSO_MYCONFIG_FILE=<src>/maintainer/configs/maxset.hpp` (checked: `build-max/src/config/include/config/myconfig-final.hpp.in` is byte-identical to `maxset.hpp`, `build-hip`'s to `src/config/myconfig-default.hpp`).

```
cmake --build build-hip -j 12
cmake --build build-hip -j 12 --target python_test_data cuda_test local_samples local_benchmarks
ctest --test-dir build-hip -L gpu --timeout 900 -j 8 --output-on-failure
cmake --build build-hip -j 12 --target check_unit_tests
cmake --build build-max -j 12
cmake --build build-max -j 12 --target python_test_data cuda_test local_samples local_benchmarks
ctest --test-dir build-max -L gpu --timeout 900 -j 8 --output-on-failure
ctest --test-dir build-max -L gpu -E 'sample_|benchmark_' -V --timeout 900 -j 8
cmake --build build-max -j 12 --target check_unit_tests
cmake --build build-hip -j 12 --target local_samples local_benchmarks
ctest --test-dir build-hip -LE gpu --timeout 900 -j 16
```

- Builds: default 236/236 plus test-data targets, 20 warnings; maxset PASS, 30 warnings; every warning is the known ROCm 10.1 `nodiscard` `hipError_t` `-Wunused-value`. Cache: `ESPRESSO_BUILD_WITH_HIP=ON`, `Kokkos_ENABLE_CUDA=OFF`, `Kokkos_ENABLE_HIP=OFF`, `Heffte_ENABLE_CUDA=OFF`. `llvm-objdump --offloading espresso_core.so`: 4 bundles, all `hip-amdgcn-amd-amdhsa--gfx1201`, in both builds.

Counts are passed / skipped / failed (skipped = ctest exit-5 mapping).

- GPU suite, default myconfig (`-L gpu`, 74 tests, 94 s): **71 passed / 3 skipped / 0 failed**. Skipped: `sample_p3m_with_gpu`, `benchmark_p3m_with_gpu` (script runs on import, no test methods), `benchmark_lb_with_gpu` (WALBERLA not built); the same three as every earlier run.
- GPU suite, maxset myconfig (`-L gpu`, 74 tests, 93 s): **71 passed / 3 skipped / 0 failed**, the same three skipped. Verbose rerun without samples/benchmarks: 68/68 passed, no unittest `FAILED` line (the `ERROR:` lines in the log are ESPResSo runtime messages raised on purpose by the exception tests). Cases the default config compiles out ran and passed: dipole_field_tracking test_dds, test_dds_gpu, test_dp3m; dipolar_direct_summation test_dds_gpu, test_inner_loop_consistency_gpu, test_min_image_convention_gpu; dawaanr-and-dds-gpu test_dds_mixed_particles, test_dds_replica_cpu, test_dp3m_dlc/metallic/non_metallic. Unittest-level skips name only features not built or out of scope (WALBERLA, WALBERLA_FFT, MMM1D, STOKESIAN_DYNAMICS, SCAFACOS, SCAFACOS_DIPOLES, THERMAL_STONER_WOHLFARTH, H5MD) or are mode/combination filters inside the test.
- Unit tests (`check_unit_tests`): default **136 passed / 0 skipped / 0 failed**; maxset **136 passed / 0 skipped / 0 failed**.
- Non-GPU suite, default (`-LE gpu`, 403 tests): 346 passed / 29 skipped / 28 failed, all environmental and the same set as every earlier run: 26 `tutorial_*` Not Run, `test_convert` (no `nbformat`), `test_install` (`diff: /usr/local/lib/python3.12/site-packages/espressomd/...: No such file or directory`; never installed). No regression.
- CUDA no-regression at this head: not rerun; already PASS at 4861f4839 (nvcc 12.8.93, 542/542, "Porter 2026-11-19 (polish round)"). This host has no CUDA toolkit.
- Jargon: `python3 utils/jargon.py --port espresso` clean. Docs: "AMD GPU acceleration" at doc/sphinx/installation.rst:300, configure example with `ESPRESSO_BUILD_WITH_HIP=ON` at :312, option listed at :919.
- Gates: with gfx950 (wave64) and this run (wave32) at 4861f4839, both required Linux gates are met at head; windows is waived. Nothing was edited in the fork and nothing was pushed to it.
## Validation 2026-11-19 (linux-gfx942, MI300X wave64, polish round 4861f4839)

State: revalidate -> completed. GPU: `AMD Instinct MI300X HF` (`rocminfo` Marketing Name and `amd-smi static --asic` MARKET_NAME; gfx942, device id 0x74a9, 304 CUs, Wavefront Size 64), `HIP_VISIBLE_DEVICES=0`, `ulimit -c 0`. ROCm: pip ROCm SDK 10.1.0 (`rocm-sdk version`; clang from `rocm-sdk path --root`, no /opt/rocm). Validated sha: 4861f483930b774eccb14a26125de323d1a33851, a fresh fork clone at `projects/espresso/src`, clean at head throughout. Full run, no carry-forward.

Host deps differ from the earlier gfx942 runs: this host image now has the system packages, so nothing was built from source. OpenMPI 4.1.6 (`libopenmpi-dev`), Boost 1.83 (`libboost-mpi1.83-dev`, whose `boost_mpi` is separate from `boost_mpi_python`, so `cuda_test` links without libboost_python), FFTW 3.3.10 with omp/threads, GCC 13.3.0, CMake 3.31.6. Python 3.12.3 venv by `uv` inside the job venv: numpy 2.5.5, scipy 1.18.1, Cython 3.3.1, tqdm, matplotlib, pint.

Configure (both builds; `build-max` adds `-DESPRESSO_MYCONFIG_FILE=<src>/maintainer/configs/maxset.hpp`):

```
cmake -S <src> -B <build> -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DESPRESSO_BUILD_WITH_CUDA=ON -DESPRESSO_BUILD_WITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_HIP_COMPILER=$ROCM_SDK/llvm/bin/clang++ -DCMAKE_PREFIX_PATH="$ROCM_SDK" \
  -DPython_EXECUTABLE=<py312> \
  -DESPRESSO_BUILD_WITH_WALBERLA=OFF -DESPRESSO_BUILD_WITH_HDF5=OFF \
  -DESPRESSO_BUILD_WITH_SCAFACOS=OFF -DESPRESSO_BUILD_WITH_STOKESIAN_DYNAMICS=OFF \
  -DESPRESSO_BUILD_WITH_GSL=OFF -DESPRESSO_BUILD_WITH_NLOPT=OFF \
  -DESPRESSO_BUILD_WITH_CALIPER=OFF -DESPRESSO_BUILD_WITH_FFTW=ON \
  -DESPRESSO_BUILD_WITH_PYTHON=ON -DESPRESSO_BUILD_TESTS=ON
cmake --build <build> -j 56
cmake --build <build> -j 56 --target python_test_data cuda_test
```

Tests, the commands of the gfx950 run at this head:

```
cmake --build <build> -j 32 --target local_samples local_benchmarks   # before each ctest run
ctest --test-dir build-hip -L gpu --timeout 900 -j 8
cmake --build build-hip -j 48 --target check_unit_tests
ctest --test-dir build-max -L gpu --timeout 900 -j 8 --output-on-failure
ctest --test-dir build-max -L gpu -E 'sample_|benchmark_' -V --timeout 900 -j 8
cmake --build build-max -j 48 --target check_unit_tests
ctest --test-dir build-hip -LE gpu --timeout 900 -j 16
```

Counts are passed / skipped / failed (skipped is ctest's exit-5 mapping).

- Build: PASS both. `build-hip` 236/236 plus `python_test_data cuda_test`, 28 warnings; `build-max` 236/236 plus the same targets, 30 warnings; all warnings the known ROCm 10.1 `nodiscard` `hipError_t` `-Wunused-value`. `llvm-objdump --offloading build-hip/src/core/espresso_core.so`: 4 bundles, all `hip-amdgcn-amd-amdhsa--gfx942`. Cache: `ESPRESSO_BUILD_WITH_HIP=ON`, `Kokkos_ENABLE_CUDA=OFF`, `Kokkos_ENABLE_HIP=OFF`, `Heffte_ENABLE_CUDA=OFF`.
- GPU suite, default myconfig (`-L gpu`, 74 tests): **71 passed / 3 skipped / 0 failed** (576.7 s). Skipped: `sample_p3m_with_gpu`, `benchmark_p3m_with_gpu` (script runs on import, no test methods, exit 5), `benchmark_lb_with_gpu` (WALBERLA not built). The same three as every earlier run.
- GPU suite, maxset myconfig (`-L gpu`, 74 tests): **71 passed / 3 skipped / 0 failed**, the same three skipped. Verbose rerun without samples/benchmarks: 68/68 passed, 904 unittest cases run. Cases the default config compiles out ran and passed: dipole_field_tracking test_dds, test_dds_gpu, test_dp3m; dipolar_direct_summation test_dds_gpu, test_inner_loop_consistency_gpu, test_min_image_convention_gpu; dawaanr-and-dds-gpu test_dds_gpu, test_dds_mixed_particles, test_dds_replica_cpu, test_dp3m_dlc/metallic/non_metallic. Every unittest-level skip names a feature not built or out of scope (WALBERLA, WALBERLA_FFT, MMM1D, STOKESIAN_DYNAMICS, SCAFACOS, SCAFACOS_DIPOLES, THERMAL_STONER_WOHLFARTH, H5MD) or is a mode or combination filter inside the test. The `ERROR: DipolarP3M: requires periodicity ...` lines in the log are expected-exception output of `p3m_tuning_exceptions` (Passed).
- Unit tests (`check_unit_tests`): default **136 passed / 0 skipped / 0 failed**; maxset **136 passed / 0 skipped / 0 failed**. Multi-rank MPI unit tests pass under the system OpenMPI 4.1.6.
- Non-GPU suite, default (`-LE gpu`, 403 tests): 346 passed / 29 skipped / 28 failed, all environmental and the same set as every earlier run: 26 `tutorial_*` Not Run (no Jupyter/tutorials target), `test_convert` (`ModuleNotFoundError: No module named 'nbformat'`), `test_install` (never installed). No regression.
- CUDA no-regression gate: already recorded at this head (nvcc 12.8.93, 542/542, "Porter 2026-11-19 (polish round)"); not repeated, and this host has no CUDA toolkit.
- Jargon: `python3 utils/jargon.py --port espresso` clean. Docs: "AMD GPU acceleration" at doc/sphinx/installation.rst:300, `ESPRESSO_BUILD_WITH_HIP` in its configure example (:312) and the options list (:919).
- Nothing was edited in the fork and nothing was pushed to it.
- Gates: wave64 was already satisfied at 4861f4839 by gfx950; this is additional wave64 evidence. wave32 (gfx1201) is still at 01090739c and owes a revalidation or binary-equivalence carry-forward at this head; windows is waived.

## Rulings (jeffdaily, 2026-11-20)

- The deferral `espresso-walberla-lb-ek-gpu` is ruled "now" (recorded in deferred.json): ESPResSo's lattice-Boltzmann and electrokinetics solvers are to run on AMD GPUs.
- The upstream pull request is held and will be sent complete: the core GPU commit already validated (fork head bc8d73dd9) plus the waLBerla bridge work as separate commits on top, so the core stays reviewable on its own. No upstream PR is open; a review page on our fork exists and will be refreshed later. The fork's `moat-port` is private until then: new commits go on top with plain pushes, and bc8d73dd9 is never rewritten.

## Porter 2026-11-20 (waLBerla LB/EK GPU, linux-gfx942)

Plan update: "Scope extension 2026-11-20" in plan.md. The state machine has no `review-passed -> planning` edge, so that section was written without a planning stage. The project then went `review-passed -> porting` with `set-state espresso linux-gfx942 porting` (this host holds the lock). Nothing was hand-edited.

Host: AMD Instinct MI300X (gfx942), `HIP_VISIBLE_DEVICES=0`, `ulimit -c 0`, pip ROCm SDK 10.1.0 (HIP 7.16.26385, AMD clang 24.0.0, `rocm-sdk path --root`, no /opt/rocm), system OpenMPI 4.1.6, Boost 1.83, FFTW 3.3.10, GCC 13.3.0, CMake 3.31.6. Python 3.12 venv by `uv` inside `$FLEET_VENV/py312` (numpy 2.5.5, scipy 1.18.1, Cython 3.3.1, tqdm, matplotlib, pint). The fleet guard refuses `uv pip install` into a venv outside `$FLEET_VENV`; put every venv under it. Formatters: `$FLEET_VENV/fmt` (cmakelang 0.6.13, clang-format 22.1.8, autopep8 2.3.2 + pycodestyle 2.14.0, pre-commit), used as `PATH=$FLEET_VENV/fmt/bin:$PATH pre-commit run --files <files>`.

Configure, used for every round (`agent_space/cfg.sh <build-dir>`):

```
cmake -S <src> -B <build> -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DESPRESSO_BUILD_WITH_CUDA=ON -DESPRESSO_BUILD_WITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_HIP_COMPILER=$ROCM_SDK/llvm/bin/clang++ -DCMAKE_PREFIX_PATH="$ROCM_SDK" \
  -DPython_EXECUTABLE=$FLEET_VENV/py312/bin/python \
  -DESPRESSO_BUILD_WITH_WALBERLA=ON -DESPRESSO_BUILD_WITH_HDF5=OFF \
  -DESPRESSO_BUILD_WITH_SCAFACOS=OFF -DESPRESSO_BUILD_WITH_STOKESIAN_DYNAMICS=OFF \
  -DESPRESSO_BUILD_WITH_GSL=OFF -DESPRESSO_BUILD_WITH_NLOPT=OFF \
  -DESPRESSO_BUILD_WITH_CALIPER=OFF -DESPRESSO_BUILD_WITH_FFTW=ON \
  -DESPRESSO_BUILD_WITH_PYTHON=ON -DESPRESSO_BUILD_TESTS=ON
cmake --build <build> -j $MAX_JOBS -- -k 0
cmake --build <build> -j $MAX_JOBS --target python_test_data cuda_test local_samples local_benchmarks
cmake --build <build> -j $MAX_JOBS --target check_unit_tests
ctest --test-dir <build> -L gpu --timeout 1200 -j 8
```

### Round (a): decoupling -- fork e8fa4dfa5 "[ROCm] Keep waLBerla in a HIP build with its CPU solvers"

The intake's probe, made CUDA-neutral. The configure stop is gone. `WALBERLA_BUILD_WITH_CUDA` is set only when `ESPRESSO_BUILD_WITH_CUDA AND NOT ESPRESSO_BUILD_WITH_HIP`. `OpenMP::OpenMP_CUDA` reaches the bridge targets and tests through `ESPRESSO_OPENMP_CUDA` (set for a CUDA build only, as `src/core` does). The new `espresso_set_walberla_gpu_source_language()` gives the four `*_init*.cu` files LANGUAGE CXX in a HIP build; it does nothing when HIP is off. The intake probe keyed the language on `NOT WALBERLA_BUILD_WITH_CUDA`, which would also have changed an nvcc build with a preinstalled CPU waLBerla. The installation guide drops "waLBerla must be disabled".

- Cache: `WALBERLA_BUILD_WITH_CUDA=OFF`, `WALBERLA_BUILD_WITH_HIP=OFF`. Build: 0 FAILED, 21 warnings (20 known `nodiscard hipError_t`, 1 `-Wunused-macros` `HAS_HEFFTE`).
- `check_unit_tests`: **150/150 passed**, including every waLBerla LB/EK suite.
- cmake-format passes; jargon and prose clean.

### Round (b): LB/EK GPU with HIP -- fork 3862a5040 "[ROCm] Run the waLBerla LB and EK GPU solvers with HIP"

- CMake: `WALBERLA_BUILD_WITH_HIP on` for a HIP build (else CUDA as before). `walberla::gpu` is linked for either backend. The language helper picks HIP when waLBerla has HIP, and `espresso_set_walberla_codegen_kernel_source_properties()` applies it to the generated `.cu` files. The generated GPU kernel lists use `WALBERLA_BUILD_WITH_CUDA OR WALBERLA_BUILD_WITH_HIP`. The two bridge GPU tests get LANGUAGE HIP and link `hip::host` (`ESPRESSO_WALBERLA_GPU_TEST_RUNTIME`).
- Guards: 17 hand-written files (see plan decision 3).
- Generated kernels, four fault classes. Three are the intake's: `RESTRICT` was undefined under clang-HIP in `preprocessor.tmpl.cuh` and the FieldAccessors templates' own prelude; philox `QUALIFIERS`; `uint3` from HIP built-ins. The fourth is new: `ReactionKernelSelector.tmpl.h` guards `KernelTraitGPU`/`get_kernel_gpu` on `__CUDACC__ and WALBERLA_BUILD_WITH_CUDA`, which made `EKReactionImplBulk/Indexed.hpp` fail. The intake probe compiled kernels only, never the selector headers.
  - philox gets clang-HIP == clang-CUDA in all three places that test `__clang__ && __CUDA__` (qualifiers, `mulhilo32` host branch, SSE block), not only the qualifier.
  - `uint3`: HIP accepts only explicit construction from `.x/.y/.z` (probe: `set(blockIdx, threadIdx)`, `uint3(blockIdx)` and a `uint3` returned as `dim3` all fail; `make_uint3(blockIdx.x, ...)` and `uint3{...}` compile). The templates gain `block_index()`/`thread_index()`. `getLinearIndex()` reads the built-ins itself; all 21 call sites passed exactly `(blockIdx, threadIdx, gridDim, blockDim, n)`. `calculate_dim_grid()` returns `dim3(...)`. A helper taking the built-in by value or reference was avoided: clang-CUDA deletes the built-ins' copy constructor, and nvcc was not available to check binding by reference.
  - Hand-written `greens_function.hpp` has the same `uint3` issue in its 4 kernels and needs hipFFT data types (`cufftReal/...` aliased to `hipfft*` under `__HIPCC__` only).
- Regeneration check (plan decision 2). Script `agent_space/fix_gen.py`, run with `agent_space/regen.sh` over a scratch worktree, with a codegen venv `$FLEET_VENV/codegen` built from the Readme recipe: `uv pip install -c requirements.txt numpy cython sympy islpy jinja2 setuptools packaging`, then lbmpy@e9efe343 and pystencils@e851f4e3 from gitos. Versions print as `1.4+1.ge851f4e 1.4+1.ge9efe34 1.12.1`, matching the file headers. The waLBerla python comes from a plain clone at 007e77e0; a worktree fails because the generator reads `.git/HEAD` as a file. One regeneration takes 5 min. Unmodified templates touch 148 files (noise). With the fixed templates, after cancelling verbatim reorderings: 241 generated files, net delta identical in 228. The other 13 are the three `generated_kernels/CMakeLists.txt` (not generated; my CMake edits) and 10 files with sympy term order or random-hash temporaries, with no residual line on an edited construct.
- Build: 0 FAILED. Warnings: only the 20 known `nodiscard hipError_t` and 1 `-Wunused-macros`. `llvm-objdump --offloading espresso_walberla.so`: 56 `gfx942` code objects.
- `check_unit_tests`: **152/152 passed**, including `LBWalberlaImpl_field_accessors_tests_cuda` (macroscopic_accessors<float/double> ran on the GPU, 0.5 s) and `PoissonSolver_test_cuda`. Its FFT GPU case is compiled out until round (c).
- `ctest -L gpu` (74 tests, 447 s): **67 passed, 7 failed**: `ek_eof`, `ek_interface`, `ek_potential`, `ek_potential_periodic`, `lattice_vtk`, `save_/test_checkpoint__therm_lb__p3m_gpu__lj__lb_walberla_gpu_binary`. Every one stops at `RuntimeError: waLBerla was compiled without CUDA support` from `new_ek_poisson_fft_cuda`. In round (b) that factory keeps its CUDA-only guard and is compiled as C++ in a HIP build, so the intermediate commit builds and fails loudly rather than silently.
- Formatters pass (clang-format on hand-written sources and the two philox headers; generated `.cu` reformatted with the Readme's `-style "{Language: Cpp, ColumnLimit: 0}"`, which changed only edited lines). Jargon and prose clean.

### Round (c): EK FFT Poisson solver -- fork 17094fdf3 "[ROCm] Run the electrokinetics FFT Poisson solver with HIP"

- heFFTe v2.4.1 (FetchContent) gets `Heffte_ENABLE_ROCM` in a HIP build (else `Heffte_ENABLE_CUDA` as before). `PoissonSolverFFT` selects `heffte::backend::rocfft` under `__HIPCC__`. The FFT solver guards and its GPU factory follow the round (b) pattern, and the factory is compiled as HIP via the language helper.
- heFFTe's rocFFT backend rejects `hipfftComplex` ("Using either an unknown complex type", `check_types<rocfft, float, HIP_vector_type<float,2>>`): its CUDA backend registers `cufftComplex`, but its ROCm backend registers no hipFFT type. Fixed in ESPResSo, under `__HIPCC__`, through heFFTe's documented extension point: `heffte::is_ccomplex<hipfftComplex>` / `is_zcomplex<hipfftDoubleComplex>`. No heFFTe patch. (ESPResSo already carries `cmake/heffte.patch`, which would not reach a preinstalled heFFTe.)
- Gotcha: heFFTe's `option()` caches `Heffte_ENABLE_ROCM=OFF` on the first configure, and ESPResSo's non-forced `set(... CACHE)` cannot change it. Reconfiguring an existing build directory keeps the GPU FFT disabled. Use a fresh build directory, as with upstream's CUDA switch.
- **Runtime-path defect found and fixed in the same commit.** With heFFTe built for ROCm, every C++ unit test links rocFFT and the HIP runtime through the static heFFTe. 149 of 152 tests then failed with `librocfft.so.0: cannot open shared object file`. Cause: the pip ROCm SDK's `lib` is in `CMAKE_HIP_IMPLICIT_LINK_DIRECTORIES`, so CMake leaves it out of the build runtime path of those C++ executables (HIP-linked targets such as `cuda_test` get it through `hip::host`), and the directory is not on the loader path (`ldconfig -p` lists only a stray `libamdhip64.so.5`). The core commit's HIP branch had defined upstream's `espresso_add_cuda_rpaths()` empty. It now appends `$<TARGET_FILE_DIR:hip::amdhip64>` to BUILD_RPATH/INSTALL_RPATH, the parallel of what upstream does with the CUDA library directories for the same targets (unit tests and `espressomd__init`). After the fix: 152/152.
- Build (fresh `build-c`): cache `Heffte_ENABLE_ROCM=ON`, `Heffte_ENABLE_CUDA=OFF`, `WALBERLA_BUILD_WITH_HIP=on`. 0 FAILED. Warnings: only the 20 known `nodiscard hipError_t`; the `-Wunused-macros HAS_HEFFTE` warning is gone because the macro is now used.
- `PoissonSolver_test_cuda --log_level=test_suite`: `ek_poisson_solver_fft_cuda<float>` 0.92 s and `<double>` 0.87 s on the GPU, `ek_poisson_solver_none_cuda`, all pass.
- `ctest -L gpu` (74 tests): 74 passed in the run before the rpath change (502 s). After the rpath change: 69 passed and 5 `sample_*/benchmark_*` failed with "already processed, cannot overwrite" (stale processed copies; rebuild `local_samples local_benchmarks` before **every** ctest run). Rerun after the rebuild: all pass. `sample_p3m_with_gpu`, `benchmark_p3m_with_gpu` and `benchmark_lb_with_gpu` print "NO TESTS RAN" (the script runs on import, and `benchmark_lb` does run LB on the GPU and prints timings), so ctest reports them as Skipped. This is upstream behaviour.

### CUDA no-regression compile 2026-11-20 (nvcc on this host) -- PASS at 17094fdf3

This host's image now has CUDA 12.8 (`/usr/local/cuda/bin/nvcc`, release 12.8, V12.8.93) next to the ROCm SDK. The gfx942 runs of 2026-11-19 recorded none. So the owed nvcc compile ran here, with no NVIDIA GPU: compile and link only. Recipe: the "CUDA no-regression compile 2026-11-19" one (upstream `cuda12-maxset`: maxset myconfig, `ESPRESSO_WARNINGS_ARE_ERRORS=ON`, waLBerla ON, Python OFF), with system OpenMPI 4.1.6 and Boost 1.83 (no local prefix needed). Script `agent_space/cuda-build.sh <src> <build>`, run on a worktree of the fork at 17094fdf3:

```
cp <src>/maintainer/configs/maxset.hpp <build>/myconfig.hpp
CUDACXX=/usr/local/cuda/bin/nvcc CUDAHOSTCXX=/usr/bin/g++ cmake -G Ninja -S <src> -B <build> \
  -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_EXTENSIONS=OFF \
  -D CMAKE_C_COMPILER=/usr/bin/gcc -D CMAKE_CXX_COMPILER=/usr/bin/g++ -D CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -D CUDAToolkit_ROOT=/usr/local/cuda -D Kokkos_CUDA_DIR=/usr/local/cuda \
  -D ESPRESSO_WARNINGS_ARE_ERRORS=ON -D ESPRESSO_BUILD_WITH_CUDA=ON -D ESPRESSO_BUILD_WITH_WALBERLA=ON \
  -D ESPRESSO_BUILD_WITH_FFTW=ON -D ESPRESSO_BUILD_WITH_PYTHON=OFF -D Python_EXECUTABLE=/usr/bin/python3.12 \
  -D ESPRESSO_BUILD_TESTS=ON -D ESPRESSO_BUILD_WITH_HDF5=OFF -D ESPRESSO_BUILD_WITH_SCAFACOS=OFF \
  -D ESPRESSO_BUILD_WITH_STOKESIAN_DYNAMICS=OFF -D ESPRESSO_BUILD_WITH_GSL=OFF \
  -D ESPRESSO_BUILD_WITH_NLOPT=OFF -D ESPRESSO_BUILD_WITH_CALIPER=OFF
ninja -C <build> -k 0 -j 40 all <every unit-test executable: ninja -t targets all, src/**/tests|unit_tests, minus the two ctest-runner targets `*/test`>
```

- Cache: `WALBERLA_BUILD_WITH_CUDA=on`, `WALBERLA_BUILD_WITH_HIP=OFF`, `Heffte_ENABLE_CUDA=ON`, `Heffte_ENABLE_ROCM=OFF`, `ESPRESSO_BUILD_WITH_HIP=OFF`, `Kokkos_ENABLE_CUDA=OFF`.
- Result: every compile and link edge succeeds, 0 compiler warnings under `-Werror`, 73 nvcc compile commands (59 of them in `src/walberla_bridge`, i.e. the generated kernels and bridge with the round (b)/(c) edits), and `LBWalberlaImpl_field_accessors_tests_cuda` and `PoissonSolver_test_cuda` link. (The first `-k 0` run listed two FAILED edges. Both were the `test.util` ctest-runner targets my target grep had included, not compile errors.)
- Build-graph identity: bc8d73dd9 was configured with the same recipe (`build-cuda-base`, configure only) and the generated `build.ninja` + `CMakeFiles/*.ninja` were compared (`agent_space/ninja_cmp.py`). With source and build paths normalised and CMake's runpath `:` padding collapsed (its length follows the build-path length), all 1842 FLAGS/DEFINES/INCLUDES/LINK_* lines are identical, and so are all 2313 build edges (output, rule, first input). The three bridge commits do not change the CUDA build graph. The CUDA sources they touch (bridge headers, generated `.cu`, the two GPU tests) compile clean.
- 917d6f707 (Caliper) changes only lines inside `if(ESPRESSO_BUILD_WITH_CALIPER)`, which the recipe leaves OFF, and with HIP off it sets both switches to `ESPRESSO_BUILD_WITH_CUDA` as before. Not rebuilt for nvcc.

### Round (d): Python LB/EK suite -- no code change needed; Caliper fix 917d6f707

- With waLBerla's HIP backend, `espressomd.gpu_available()` is true and the GPU LB/EK factories exist, so `@utx.skipIfMissingGPU()` classes run. The "compiled without CUDA support" path is reached only by a HIP build with a waLBerla that has no GPU backend: the round (a) intermediate, or a preinstalled CPU-only waLBerla. That is the same as upstream's nvcc build with a CPU-only waLBerla, so the tests were left as upstream wrote them.
- 37 Python files carry LB/EK GPU classes. 33 have a `gpu`-labelled ctest entry, and the 4 with `GPU_SLOTS 0` (`lb_interpolation`, `lb_streaming`, `lb_thermostat`, `lb_mass_conservation`) are multi-rank CPU runs with a `1_core` GPU twin.
- Verbose run `ctest -L gpu -R 'lb|ek_|lattice|LB|checkpoint|tracers' -V -j 8` on `build-c` (17094fdf3): **52/52 passed** (365 s). Parsed unittest lines: 127 GPU-class cases `ok`, 0 FAIL/ERROR (the "ERROR:" strings are expected-exception messages printed by `ek_interface`'s `test_runtime_exceptions`, which passes), and 0 occurrences of "compiled without CUDA support". 6 GPU-class skips: `TestLBMomentumConservationRegularSinglePrecisionWalberlaGPU` (the test's own filter: "LB with regular decomposition already tested with 2 MPI ranks"), and 5 `lattice_vtk` GPU classes "missing python module espressomd.io.vtk". The latter is environmental: `espressomd.io.vtk` imports the `vtk` package. After `uv pip install --python $FLEET_VENV/py312/bin/python vtk` (VTK 9.7.2), `lattice_vtk` ran 34 tests, OK, with all 6 LB/EK VTK GPU classes `ok`. The remaining skips in the suite name features not in the default myconfig (STOKESIAN_DYNAMICS, SCAFACOS, SCAFACOS_DIPOLES, H5MD, THERMAL_STONER_WOHLFARTH, LB_ELECTROHYDRODYNAMICS), which the maxset build below covers.
- Non-GPU suite `ctest -LE gpu -j 16` (419 tests, 339 s): 391 passed, 28 failed. The failures are the same environmental set as every earlier run: 26 `tutorial_*` Not Run, `test_convert` (no `nbformat`), `test_install` (never installed). The total is up from 403 because the waLBerla CPU tests are now built.
- Found while writing the surface accounting: with `ESPRESSO_BUILD_WITH_CALIPER=ON`, the core commit's HIP build still passed `CALIPER_WITH_NVTX/CUPTI=${ESPRESSO_BUILD_WITH_CUDA}` (ON). Caliper then does `find_package(CUDAToolkit REQUIRED)`, which fails on a ROCm-only host; on this host it built NVTX/CUPTI from /usr/local/cuda into the AMD build ("Nvtx support: Yes, using /usr/local/cuda"). Fix 917d6f707: both switches are ON only for a CUDA build. Verified: configure reports "Nvtx support: No", "CUpti support: No"; `build-cal` (HIP, Caliper ON, waLBerla OFF) 384/384; `caliper`, `p3m_madelung`, `gpu_availability` pass.

### Maxset build (HIP, waLBerla HIP) -- tree 17094fdf3 + the Caliper change (inert with Caliper OFF), i.e. 917d6f707

`agent_space/cfg.sh build-max -DESPRESSO_MYCONFIG_FILE=<src>/maintainer/configs/maxset.hpp`, then the round commands (`agent_space/run-max.sh` in the job's `work/`).

- Build: 0 FAILED. `check_unit_tests`: **152/152 passed**.
- `ctest -L gpu -V -j 8` (74 tests, 499 s): **74/74 passed**, 74 "Ran" lines and no unittest FAILED. Parsed: 138 GPU-class cases `ok`, 1 GPU-class skip (the test's own "LB with regular decomposition already tested with 2 MPI ranks" filter), 0 "compiled without CUDA support". No `LB_ELECTROHYDRODYNAMICS` skip, so the feature the default config leaves out ran here. Every other skip is a feature not built (MMM1D, STOKESIAN_DYNAMICS, SCAFACOS(_DIPOLES), H5MD, THERMAL_STONER_WOHLFARTH) or one of upstream's mode filters in `test_checkpoint.py` ("Lees-Edwards not implemented for LB GPU", "LB not supported", "... not in modes"), which apply to CUDA as well.

## Lesson for cuda-to-rocm: heFFTe's rocFFT backend does not accept hipFFT complex types

Promoted to cuda-to-rocm/references/libraries.md in 35013557.

Belongs in the library-substitution reference (FFT section). heFFTe recognizes `cufftComplex`/`cufftDoubleComplex` in its cuFFT backend, but its rocFFT backend registers no hipFFT type. A CUDA code that keeps its FFT buffers as `cufftComplex` and maps the name to `hipfftComplex` (= `HIP_vector_type<float,2>`) fails with `static assertion ... check_types<rocfft, float, HIP_vector_type<float, 2>>: Using either an unknown complex type`. heFFTe documents `heffte::is_ccomplex<T>` / `is_zcomplex<T>` as the specialization point for custom complex types. Specializing them for `hipfftComplex` / `hipfftDoubleComplex` in the HIP branch fixes it without patching heFFTe; heFFTe then reinterprets the buffers as `std::complex`, whose layout they share.

## Lesson for cuda-to-rocm: the ROCm library directory is an implicit HIP link directory, so CMake drops it from C++ executables' runtime path

Promoted to cuda-to-rocm/references/build-cmake-link.md in 35013557.

Belongs in the Linux build and link reference. `CMAKE_HIP_IMPLICIT_LINK_DIRECTORIES` contains the ROCm `lib` directory, and CMake leaves implicit link directories out of the build runtime path. A plain C++ executable that links the HIP runtime or rocFFT only indirectly (through a static library built with HIP sources, here heFFTe with `Heffte_ENABLE_ROCM`) gets `NEEDED librocfft.so.0` / `libamdhip64.so.7` with no RUNPATH entry. It runs on a system ROCm whose lib is in `ld.so.conf`, and fails with "cannot open shared object file" on a ROCm that is not on the loader path (the pip ROCm SDK). Targets that link `hip::host` directly do get the directory. Fix: add `$<TARGET_FILE_DIR:hip::amdhip64>` to `BUILD_RPATH`/`INSTALL_RPATH` of such targets, as projects already do for the CUDA library directories.

## Lesson for cuda-to-rocm: HIP's built-in index variables do not convert to uint3, and HIP's uint3 does not convert to dim3

Promoted to cuda-to-rocm/references/build-headers-macros.md in 35013557.

Belongs in the strict-AMD fault-class reference. CUDA's `blockIdx`/`threadIdx` are `uint3` and `make_uint3()` yields a `uint3` that converts to `dim3`. In HIP the built-ins are distinct types: passing them to a function that takes `uint3` (waLBerla's `gpu::FieldAccessor::set(uint3, uint3)`), `uint3(blockIdx)`, and returning a `uint3` as `dim3` all fail to compile. Construct explicitly from the members (`make_uint3(blockIdx.x, blockIdx.y, blockIdx.z)`), and return `dim3(...)`. Do not write a helper that takes the built-in by value: clang-CUDA deletes the built-ins' copy constructor. (Heading corrected by the 2026-11-20 review: the built-ins do convert to `dim3`; `dim3(blockIdx)` compiles with HIP 7.16. What fails is built-in to `uint3` and `uint3` to `dim3`.)

## Owed next (2026-11-20)

- Review of the delta bc8d73dd9..917d6f707 (four commits: e8fa4dfa5, 3862a5040, 17094fdf3, 917d6f707); stage left at `delta-ported`.
- Validation at 917d6f707 on linux-gfx950 (MI355X, wave64) and on a 32-wide Radeon (linux-gfx1201). Use the configure above with waLBerla ON in a **fresh** build directory (heFFTe's ROCm switch is a cache option). Gates: `check_unit_tests`, `ctest -L gpu` (rebuild `local_samples local_benchmarks` first) and the verbose LB/EK subset; install `vtk` in the test venv so `lattice_vtk`'s GPU classes run. linux-gfx942 evidence for this head is in the porter rounds above. The porter is not the validator, so this host still owes its own validation record.
- nvcc no-regression compile: done on this host at 17094fdf3 (above). 917d6f707 is CMake inside the Caliper block only. A rerun on the mi355 host is optional.
- Windows: waived for this project (approved 2026-11-18).
- Not done, for jeff: the review page on the fork (AMD-Ecosystem/espresso#1) still shows bc8d73dd9 and its PR body describes the core only; refresh it when the PR is prepared. waLBerla upstream needs nothing for the pinned 623a60c1. A future waLBerla pin bump past 2026-11-02 needs upstream's HIP version check fixed first (draft report #1 in the waLBerla intake notes).

## Review 2026-11-20 (waLBerla LB/EK GPU delta bc8d73dd9..917d6f707, linux-gfx942)

Reviewer on the MI300X host, pr-review skill, fork `moat-port` at 917d6f707 (four commits e8fa4dfa5, 3862a5040, 17094fdf3, 917d6f707 on the validated core commit bc8d73dd9). Verdict: **review-passed**. There are no fork findings. One lesson heading in these notes was overstated, and this review corrected it (below). Each check was fact-checked by re-running it, not by reading the porter's notes.

### Finding (MOAT records only, fixed in place)

- The lesson heading "HIP's built-in index variables do not convert to uint3 or dim3" overstated it. A probe with HIP 7.16 (`clang++ -x hip --offload-arch=gfx942 -c`) shows `take(blockIdx, threadIdx)` with `take(uint3, uint3)` fails, `uint3(blockIdx)` fails, and returning `make_uint3(...)` from a `dim3` function fails. But `dim3(blockIdx)` compiles, and so do `make_uint3(blockIdx.x, ...)` and `uint3{threadIdx.x, ...}`. The heading now says built-in to `uint3` and `uint3` to `dim3`, and the body records the correction. The body's own claims all hold. The fork's comments and commit message 2 already state it correctly ("do not convert to `uint3`", "returns a `dim3` instead of converting a `uint3`").

### Verified (no action; recorded so the validator and the next reviewer need not re-derive)

- **CUDA build unchanged, recompiled at the head.** This host has nvcc 12.8 (V12.8.93). Ran the porter's `cuda12-maxset` recipe (maxset myconfig, `ESPRESSO_WARNINGS_ARE_ERRORS=ON`, waLBerla ON, Python OFF, Caliper OFF) on a worktree at **917d6f707**, with `all` plus all 154 unit-test executables: **0 FAILED, 0 compiler warnings under -Werror**. 70 nvcc compile commands. `PoissonSolver_test_cuda` and `LBWalberlaImpl_field_accessors_tests_cuda` link. Cache: `WALBERLA_BUILD_WITH_CUDA=on`, `WALBERLA_BUILD_WITH_HIP=OFF`, `Heffte_ENABLE_CUDA=ON`, `Heffte_ENABLE_ROCM=OFF`. The first run stopped with 8 FAILED edges, all "No space left on device" (the host disk hit 100% because of other jobs; 6.5-7.5 GB free). `ninja -k 0 -j 24` resumed and finished clean. The configured build graph at 917d6f707 matches a bc8d73dd9 configure of the same recipe: 1842 flag lines and 2313 edges, identical with paths normalised. The last commit cannot affect this recipe: it changes only lines inside `if(ESPRESSO_BUILD_WITH_CALIPER)`, and with HIP off `ESPRESSO_CALIPER_WITH_CUDA_TOOLS` equals `ESPRESSO_BUILD_WITH_CUDA`, as before.
- **CPU-only build unchanged.** Default options, CUDA OFF, waLBerla ON, FFTW ON, and **Caliper ON** so the last commit is exercised. Configured at bc8d73dd9 and 917d6f707: 1885 flag lines and 2800 edges, identical. Cache: `CALIPER_WITH_NVTX/CUPTI=OFF`, `WALBERLA_BUILD_WITH_GPU_SUPPORT=FALSE`. Source-level, a CPU compile sees only `#if` lines that add `__HIPCC__`, `__HIP__` or `__HIP_DEVICE_COMPILE__` (not defined by g++), or `WALBERLA_BUILD_WITH_GPU_SUPPORT` (not defined when waLBerla has no GPU backend).
- **Every change is behind the HIP option or keyed on waLBerla's own switches.** CMake: the waLBerla HIP/CUDA switch, the heFFTe ROCm/CUDA switch and the Caliper switches branch on `ESPRESSO_BUILD_WITH_CUDA AND ESPRESSO_BUILD_WITH_HIP`. `ESPRESSO_OPENMP_CUDA` is set exactly as in `src/core/CMakeLists.txt:94`. `espresso_set_walberla_gpu_source_language()` returns without doing anything unless `ESPRESSO_BUILD_WITH_HIP`. The kernel lists and GPU tests use `WALBERLA_BUILD_WITH_CUDA OR WALBERLA_BUILD_WITH_HIP`. The test runtime is `CUDA::cudart` unless `WALBERLA_BUILD_WITH_HIP`. Preprocessor: `WALBERLA_BUILD_WITH_GPU_SUPPORT` is `#cmakedefine`d by waLBerla at the pin 623a60c1 (`CMakeLists.txt:651-655`, `waLBerlaDefinitions.in.h:39`) whenever CUDA or HIP is on. The test files' `gpu*` calls are waLBerla `gpu/GPUWrapper.h` macros that map to `cudaGetDeviceCount`/`cudaDeviceProp`/`cudaGetDeviceProperties` for CUDA (lines 82-85). One corner was checked and judged not a port defect: waLBerla sets `WALBERLA_BUILD_WITH_GPU_SUPPORT` with a non-forced `CACHE`. So a build directory first configured without a GPU backend, then reconfigured with `-DWALBERLA_BUILD_WITH_CUDA=ON`, keeps it FALSE. waLBerla's own `gpu` module keys on the same macro, so that tree is inconsistent upstream as well.
- **Generated files: the regeneration claim reproduces independently.** Two fresh regenerations, run by this review: unmodified templates (worktree at bc8d73dd9) and fixed templates (worktree at 917d6f707). Readme recipe and the pinned generator (pystencils 1.4+1.ge851f4e, lbmpy 1.4+1.ge9efe34, sympy 1.12.1, waLBerla python 007e77e0), with the Readme's clang-format styles. For each file, the line-multiset delta between the two regenerations was compared with the committed bc8d73dd9 -> 917d6f707 delta. Result: **238 generated files (the porter's 241 counted the three `generated_kernels/CMakeLists.txt`), 230 with an identical net delta**, covering all 33 hand-edited files except the two DynamicUBB*CUDA.cu. The 8 residual files are sympy term order (6 `DiffusiveFluxKernelWithElectrostatic*`, 4 lines each, never hand-edited) and random-hash temporaries (`DynamicUBB{Single,Double}PrecisionCUDA.cu`, 8 lines each). **0 residual lines are preprocessor lines or touch an edited construct** (`block_index`, `thread_index`, `getLinearIndex`, `dim3`, `make_uint3`, `__HIP*`, `__CUDA*`, `GPU_SUPPORT`, `QUALIFIERS`). `philox_rand.h` is not generator output: it is hand-maintained, "Adapted from the pystencils source file" (upstream #5221 updated it by hand), so editing it by hand is the upstream practice. Scripts: `agent_space/regen.sh`, `agent_space/cmp_regen.py` (container-local).
- **Fourth template fix (`ReactionKernelSelector.tmpl.h`).** The guard becomes `(defined(__CUDACC__) or defined(__HIPCC__)) and defined(WALBERLA_BUILD_WITH_GPU_SUPPORT)` in both places. The regenerated `ReactionKernelBulk_all.h` / `ReactionKernelIndexed_all.h` match the hand edit exactly. For nvcc it is the old condition (see above). It matches the pattern in `EKReactionImplBulk/Indexed.hpp`.
- **heFFTe hook.** `heffte::is_ccomplex<hipfftComplex>` / `is_zcomplex<hipfftDoubleComplex>` are specialized after `<heffte.h>`, under `__HIPCC__` only, at namespace scope before any instantiation. heFFTe 2.4.1 documents these traits as the extension point (`heffte_utils.h:239-252`) and specializes only the cuFFT types (`heffte_backend_cuda.h:333,338`), so there is no redefinition today. A future heFFTe that adds the same specializations would fail to compile, loudly; heFFTe is pinned at v2.4.1.
- **heFFTe backend switch and a stale build directory.** `Heffte_ENABLE_ROCM` is a heFFTe `option()`, and ESPResSo sets it with a non-forced `CACHE`, so a reused build directory keeps its first value. This **cannot silently keep the wrong backend**. `PoissonSolverFFT` names `heffte::backend::rocfft` under `__HIPCC__`. `heffte::fft3d`'s constructor does `static_assert(backend::is_enabled<backend_tag>::value, "The requested backend is invalid or has not been enabled.")` (`heffte_fft3d.h:273,296`), and `is_enabled<rocfft>` is specialized only in `heffte_backend_rocm.h`, which is included only with `Heffte_ENABLE_ROCM`. So a stale OFF fails the compile of `ek_poisson_fft_init_cuda.cu`. A preinstalled heFFTe without ROCm (`find_package(Heffte 2.4.1 QUIET)`) fails the same way. Upstream's CUDA switch behaves identically (`Heffte_ENABLE_CUDA`, cached, the same static_assert for `cufft`) and has no CMake guard, so a guard would be HIP-only divergence for no silent-failure case. Not requested. Commit 3's Test Plan already tells the reader to use a new build directory.
- **Runtime path (`librocfft.so.0`).** `espresso_add_cuda_rpaths()` in the HIP branch appends `$<TARGET_FILE_DIR:hip::amdhip64>` to BUILD_RPATH and INSTALL_RPATH, the parallel of upstream's CUDA branch (`ESPRESSO_LIBCUDA_RPATH`/`ESPRESSO_LIBCUDART_RPATH`, absolute paths in both). It is a build-file change keyed on the imported HIP target, not on a layout. For a system ROCm it resolves to `/opt/rocm/lib`, where rocFFT also lives; there it is redundant but harmless (usually in `ld.so.conf`). For the pip SDK it resolves to `_rocm_sdk_devel/lib`, which holds `libamdhip64.so.7` and `librocfft.so.0` (checked on this host). In a split install where rocFFT has its own prefix, that prefix is not one of clang's implicit HIP link directories, and those are the only ones CMake drops, so CMake adds it to the runtime path itself. `hip::amdhip64` exists whenever the HIP branch runs, because `find_package(hipfft REQUIRED)` does `find_dependency(hip)` (`hipfft-config.cmake:90`).
- **Caliper commit.** NVTX/CUPTI follow `ESPRESSO_BUILD_WITH_CUDA AND NOT ESPRESSO_BUILD_WITH_HIP`. That is unchanged for CUDA and CPU (graph identity above, with Caliper ON). In a HIP build both are OFF.
- **Commit split and messages, read as an upstream maintainer would.** Four commits: decoupling, LB/EK GPU, EK FFT, Caliper. Each builds on its own, and each says what it leaves out. Commit 2 openly reports 7 `ctest -L gpu` failures in a HIP build (the FFT Poisson solver comes in commit 3). That is a configuration that did not exist before the series, and ESPResSo squash-merges PRs (`git log python` titles end in `(#NNNN)`), so it is not a bisect hazard upstream. Titles are 55-59 characters with the `[ROCm]` prefix, matching the core commit. Bodies are ASCII, one line per paragraph, carry the AI-assistance sentence and a Test Plan with literal commands, and have no Co-Authored-By. `jargon.py --port espresso`: clean. `prose.py` on each body: clean. The `pre-commit run --files <all 60 changed files>` hooks pass at 917d6f707 (clang-format 22.1.8, cmake-format 0.6.13) and change nothing.
- **Leftover CUDA-only guards.** No hand-written source still tests `WALBERLA_BUILD_WITH_CUDA`. The remaining `defined(__CUDACC__)` without `__HIPCC__` are `#if defined(__CUDACC__) && !defined(__clang__)` nvcc-diagnostic pragmas in generated `.cpp` files, which are correct for HIP as is. GPU-aware MPI for the EK FFT goes through upstream's `is_mpi_gpu_aware()`, which already queries `MPIX_Query_rocm_support()` (`src/core/communication.cpp:99`).

## Validation 2026-11-20 (linux-gfx942, MI300X wave64, waLBerla LB/EK GPU head 917d6f707)

Validator on the MI300X host: AMD Instinct MI300X (gfx942), `HIP_VISIBLE_DEVICES=0` (pinned by the dispatcher), `ulimit -c 0`, `MAX_JOBS=56`. Pip ROCm SDK 10.1.0 (HIP 7.16.26385, `rocm-sdk path --root`, no /opt/rocm), system OpenMPI 4.1.6, Boost 1.83, FFTW 3.3.10, GCC 13.3.0, CMake 3.31.6. Test venv `$FLEET_VENV/py312` (Python 3.12, numpy 2.5.5, scipy 1.18.1, Cython 3.3.1, tqdm, matplotlib, pint, vtk 9.7.2). Source: `projects/espresso/src` checked out at 917d6f707, clean tree. Every build is in a **new** build directory, configured with the porter's command above ("Porter 2026-11-20", `agent_space/cfg.sh`), with the source path changed to this checkout.

### Default myconfig (`agent_space/build-hip`)

- Build: `cmake --build build-hip -j 56 -- -k 0`, then the targets `python_test_data cuda_test local_samples local_benchmarks`. **0 FAILED.** Only warnings: 28 known `nodiscard hipError_t` (`-Wunused-value`). Cache: `WALBERLA_BUILD_WITH_HIP=on`, `WALBERLA_BUILD_WITH_CUDA=OFF`, `Heffte_ENABLE_ROCM=ON`, `Heffte_ENABLE_CUDA=OFF`. `llvm-objdump --offloading espresso_walberla.so`: 58 gfx942 code objects.
- `check_unit_tests`: **152/152 passed** (95 s). `PoissonSolver_test_cuda --log_level=test_suite`: `ek_poisson_solver_fft_cuda<float>` 0.95 s, `<double>` 1.01 s, `ek_poisson_solver_none_cuda` 0.16 s, no errors. `LBWalberlaImpl_field_accessors_tests_cuda`: `macroscopic_accessors<float>` 0.40 s, `<double>` 0.39 s, no errors. The timings show the GPU cases ran rather than being skipped by the `has_gpu` precondition.
- `ctest -L gpu --timeout 1200 -j 8 -V`: **74/74 passed** (498 s). Parsed with `agent_space/parse_gpu.py`, which collects unittest cases whose class name names GPU/CUDA and handles the docstring and warning lines that split a case's status onto a later line (it parses 907 of the 917 cases the logs report as run, so the counts are a lower bound). **243 GPU-class cases `ok`, 1 skipped, 0 FAIL/ERROR.** Of these, **214 `ok` are in 33 LB/EK/lattice scripts**: ek_boundary 6, ek_bulk_reactions 2, ek_diffusion 2, ek_diffusive_flux 2, ek_eof 2, ek_ext_efield 2, ek_fixeddensity 2, ek_fixedflux 2, ek_fluctuations 2, ek_friction_coupling 2, ek_indexed_reactions 2, ek_interface 36, ek_noflux 2, ek_potential 2, ek_potential_periodic 2, ek_slice 4, lattice_vtk 17, lb 38, lb_boundary 12, lb_boundary_ghost_layer 6, lb_boundary_volume_force 2, lb_buoyancy_force 2, lb_circular_couette 2, lb_interpolation 4, lb_momentum_conservation 1, lb_poiseuille 2, lb_poiseuille_cylinder 6, lb_pressure_tensor 2, lb_shear 2, lb_slice 4, lb_streaming 2, lb_thermostat 3. The four `save_/test_checkpoint__*lb_walberla_gpu*` tests (class `TestCheckpoint`, GPU mode for the whole script) pass. The single GPU-class skip is the test's own filter (`TestLBMomentumConservationRegularSinglePrecisionWalberlaGPU`: "LB with regular decomposition already tested with 2 MPI ranks"); its double-precision and 2-rank twins ran. 0 occurrences of "compiled without CUDA support". 71 unittest `OK` summaries, 0 `FAILED`. The other skips are features not in the default myconfig (MMM1D, STOKESIAN_DYNAMICS, SCAFACOS(_DIPOLES), DIPOLE_FIELD_TRACKING, THERMAL_STONER_WOHLFARTH, H5MD, LB_ELECTROHYDRODYNAMICS) or upstream's mode filters in the checkpoint tests. With vtk installed, `lattice_vtk`'s GPU classes ran (17 cases).
- Non-GPU suite `ctest -LE gpu -j 16` (419 tests, 614 s): **391 passed, 28 failed**, the same environmental set as every earlier run: 26 `tutorial_*` Not Run, `test_convert` (no `nbformat`), `test_install` (never installed). No regression.

### Maxset myconfig (`agent_space/build-hip-max`, `-DESPRESSO_MYCONFIG_FILE=<src>/maintainer/configs/maxset.hpp`)

- Build: **0 FAILED**. Only warnings: 30 known `nodiscard hipError_t`.
- `check_unit_tests`: **152/152 passed** (251 s).
- `ctest -L gpu --timeout 1200 -j 8 -V`: **74/74 passed** (964 s). **243 GPU-class cases `ok`, 1 skipped (the same 2-rank filter), 0 FAIL/ERROR, 214 `ok` in the LB/EK/lattice scripts**, 0 "compiled without CUDA support". `lb_electrohydrodynamics` (`LBEHTest`), skipped in the default config for the missing feature, ran and passed here. The remaining skips are features maxset does not build (MMM1D, STOKESIAN_DYNAMICS, SCAFACOS(_DIPOLES), THERMAL_STONER_WOHLFARTH, H5MD), upstream's own filters ("Momentum is not conserved on single precision" in the CPU swimmer classes, "LB with regular decomposition already tested with 2 MPI ranks"), and the checkpoint tests' mode filters ("Lees-Edwards not implemented for LB GPU", "LB not supported", "... not in modes", "missing combination"), which apply to CUDA as well.

No tolerance was changed, and no test missed.

### Gates

- CUDA no-regression compile at this head: done in the review above (nvcc 12.8, `cuda12-maxset` recipe at 917d6f707, 0 FAILED under -Werror, build graph identical to bc8d73dd9).
- `jargon.py --port espresso`: clean. Docs: "AMD GPU acceleration" in `doc/sphinx/installation.rst` now covers the LB/EK solvers (waLBerla HIP backend, heFFTe rocFFT backend, hipFFT/rocFFT/rocThrust).
- Integrity: `git -C projects/espresso/src status --porcelain` is empty at 917d6f707.
- Host note: the host disk sat at 100% (6.5-8.9 GB free) while other jobs ran. It interrupted the first nvcc build with "No space left on device", and the build was resumed. The HIP build directories are about 0.6 GB each.

## Owed next (2026-11-20, after review and gfx942 validation at 917d6f707)

- **linux-gfx950 (MI355X, wave64)**: revalidate at 917d6f707 (validated bc8d73dd9). wave64 is satisfied by this gfx942 record, so this is extra evidence on the newest wave64 part, but the platform reads `revalidate`. Use the configure above in a fresh build directory with waLBerla ON. Gates: `check_unit_tests`, `ctest -L gpu` (rebuild `local_samples local_benchmarks` first) with `-V` and a GPU-class case count, vtk in the test venv. A default and a maxset build.
- **wave32 (linux-gfx1201, Radeon RDNA4)**: required. The wave32 gate is unsatisfied at 917d6f707 (validated bc8d73dd9). Same recipe with `-DCMAKE_HIP_ARCHITECTURES=gfx1201`. The waLBerla HIP GPU kernels have never run on a 32-wide GPU before this head, so a full run, not a carry-forward.
- Windows: waived for this project (approved 2026-11-18).
- For jeff, once wave32 passes: the review page on the fork (AMD-Ecosystem/espresso#1) still shows bc8d73dd9, and its body describes the core only. Refresh it for the five-commit series before approval. No upstream PR is open.
- For a trunk session: promote the three `## Lesson for cuda-to-rocm` sections above (the uint3 one with its corrected heading).

## Validation 2026-11-20 (linux-gfx950, MI355X wave64, waLBerla LB/EK GPU head 917d6f707)

Validator on the MI355X host, reproducing the method of "Validation 2026-11-20 (linux-gfx942 ...)" above. Host: AMD Instinct MI355X (gfx950, wave64), `HIP_VISIBLE_DEVICES=0` (pinned by the dispatcher), `ulimit -c 0`, `MAX_JOBS=64`. Pip ROCm SDK 10.1.0 (HIP 7.16.26385, clang from `rocm-sdk path --root`, no /opt/rocm), GCC 13.3.0, CMake 3.31.6, FFTW 3.3.10, system Boost 1.83 headers. This host has no system MPI: OpenMPI 5.0.8 (runtime build with the `null` accelerator, see "Porter 2026-11-19 (CUDA path fix)") and Boost.MPI/serialization/unit_test 1.83 were copied from the earlier espresso jobs' prefixes into this job (`work/deps/ompi`, `work/deps/boostpfx`), with `OPAL_PREFIX` set, `ompi/bin` first on `PATH` and `ompi/lib:boostpfx/lib` first on `LD_LIBRARY_PATH`. Test venv `$FLEET_VENV/py312` by `uv` (Python 3.12, numpy 2.5.5, scipy 1.18.1, Cython 3.3.1, tqdm, matplotlib, pint, vtk 9.7.2; the same versions as the gfx942 run). Source: `projects/espresso/src` checked out at 917d6f707 (= `origin/moat-port`), clean tree.

Note on the dispatch prompt: it described this run as the first on a 32-wide Radeon. This host is the MI355X (wave64); orient dispatched `linux-gfx950` (`revalidate`, validated bc8d73dd9). The waLBerla HIP kernels had never run on gfx950 either, so this is a full run, not a carry-forward. It does not satisfy wave32; that gate still needs linux-gfx1201 at this head.

Configure: the porter's command ("Porter 2026-11-20") with `-DCMAKE_HIP_ARCHITECTURES=gfx950`, plus this host's MPI pins, each build in a **new** directory (`agent_space/cfg.sh <build> [extra]`):
## Validation 2026-11-20 (linux-gfx1201, RDNA4 wave32, waLBerla LB/EK GPU head 917d6f707)

Validator on a Radeon gfx1201 host (`rocminfo`: Wavefront Size 32), `HIP_VISIBLE_DEVICES=1` (pinned by the dispatcher; another job used a different GPU on the host), `ulimit -c 0`, `MAX_JOBS=12`. Pip ROCm SDK 10.1.0 (`rocm-sdk version`; `rocm-sdk path --root`, no /opt/rocm), GCC 13.3.0. This is the first run of the waLBerla HIP kernels (generated LB/EK kernels, heFFTe rocFFT Poisson solver) on a 32-wide GPU. The method is the MI300X validation above, reproduced; a full run, not a carry-forward. Source: `projects/espresso/src`, a fresh `-b moat-port` clone checked out at 917d6f70715158d1b8be1a961595073dcc3c597c (with the fork's `python` branch fetched for `jargon.py`). `git status --porcelain` stayed empty throughout.

### Host setup

The host has no system MPI, Boost or FFTW and no CUDA toolkit. Deps were rebuilt from source into a job-local prefix with the recipe and versions of the earlier gfx1201 sections: hwloc 2.11.2, OpenMPI 5.0.8 (`--without-cuda --without-rocm --disable-mpi-fortran`), FFTW 3.3.10 double+float (`--enable-shared --enable-openmp --enable-threads`), Boost 1.87.0 (mpi, serialization, test, filesystem, system; `--without-python link=shared`). Test venv `$FLEET_VENV/py312` by `uv` (Python 3.12.3, numpy 2.5.5, scipy 1.18.1, Cython 3.3.1, tqdm, matplotlib, pint, vtk 9.7.2). `$DEPS_PREFIX/bin` and `$DEPS_PREFIX/lib` on `PATH`/`LD_LIBRARY_PATH`. With the test venv first on `PATH`, `utils/timeit.sh` needs `MOAT_PYTHON` set to the MOAT interpreter (it refuses the Python 3.12 venv, which lacks jsonschema).

Configure: the porter's 2026-11-20 command with `-DCMAKE_HIP_ARCHITECTURES=gfx1201` and the job-local deps, each build in a **new** directory:

```
cmake -S <src> -B <build> -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DESPRESSO_BUILD_WITH_CUDA=ON -DESPRESSO_BUILD_WITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx950 \
  -DCMAKE_HIP_COMPILER=$ROCM_SDK/llvm/bin/clang++ -DCMAKE_PREFIX_PATH="$ROCM_SDK;<deps>/boostpfx" \
  -DPython_EXECUTABLE=$FLEET_VENV/py312/bin/python \
  -DESPRESSO_BUILD_WITH_CUDA=ON -DESPRESSO_BUILD_WITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_HIP_COMPILER=$ROCM_SDK/llvm/bin/clang++ -DCMAKE_PREFIX_PATH="$DEPS_PREFIX;$ROCM_SDK" \
  -DMPI_C_COMPILER=$DEPS_PREFIX/bin/mpicc -DMPI_CXX_COMPILER=$DEPS_PREFIX/bin/mpicxx \
  -DBoost_ROOT=$DEPS_PREFIX -DPython_EXECUTABLE=$FLEET_VENV/py312/bin/python \
  -DESPRESSO_BUILD_WITH_WALBERLA=ON -DESPRESSO_BUILD_WITH_HDF5=OFF \
  -DESPRESSO_BUILD_WITH_SCAFACOS=OFF -DESPRESSO_BUILD_WITH_STOKESIAN_DYNAMICS=OFF \
  -DESPRESSO_BUILD_WITH_GSL=OFF -DESPRESSO_BUILD_WITH_NLOPT=OFF \
  -DESPRESSO_BUILD_WITH_CALIPER=OFF -DESPRESSO_BUILD_WITH_FFTW=ON \
  -DMPI_C_COMPILER=<deps>/ompi/bin/mpicc -DMPI_CXX_COMPILER=<deps>/ompi/bin/mpicxx -DMPIEXEC_EXECUTABLE=<deps>/ompi/bin/mpiexec \
  -DESPRESSO_BUILD_WITH_PYTHON=ON -DESPRESSO_BUILD_TESTS=ON
# maxset: the same plus -DESPRESSO_MYCONFIG_FILE=<src>/maintainer/configs/maxset.hpp
cmake --build <build> -j 64 -- -k 0
cmake --build <build> -j 64 --target python_test_data cuda_test local_samples local_benchmarks
cmake --build <build> -j 64 --target check_unit_tests
ctest --test-dir <build> -L gpu --timeout 1200 -j 8 -V
ctest --test-dir build-hip -LE gpu --timeout 1200 -j 16
```

The MPI pins matter on this host: without them CMake picks `mpicc`/`mpiexec` from the Boost prefix, whose OpenMPI was built with `--enable-mca-no-build=accelerator` and aborts in `MPI_Init`. Cache in both builds: `WALBERLA_BUILD_WITH_HIP=on`, `WALBERLA_BUILD_WITH_CUDA=OFF`, `Heffte_ENABLE_ROCM=ON`, `Heffte_ENABLE_CUDA=OFF`, `ESPRESSO_BUILD_WITH_HIP=ON`.

GPU-class counts come from `agent_space/parse_gpu.py <log>` (written for this run; the gfx942 parser is container-local on that host). It splits the `-V` log by ctest test number, follows a case's status onto later lines (docstrings, warnings), and calls a case GPU-class when its class name names GPU/CUDA or the ctest name is a `*checkpoint*gpu*` test. Like the gfx942 parser, it parses 907 of the 917 cases the logs report as run (a lower bound). Per-script groupings below are by ctest name, so `*_1_core` twins are listed separately; the LB/EK/lattice total is the comparable number.

### Default myconfig (`work/build-hip`)

- Build: **0 FAILED**. Only warnings: 28 known `nodiscard hipError_t` (`-Wunused-value`), as on gfx942. `llvm-objdump --offloading espresso_walberla.so`: **58 gfx950 code objects** (gfx942 had 58).
- `check_unit_tests`: **152/152 passed**. `PoissonSolver_test_cuda --log_level=test_suite`: `ek_poisson_solver_fft_cuda<float>` 0.72 s, `<double>` 0.72 s, `ek_poisson_solver_none_cuda` 0.17 s, "No errors detected" (heFFTe rocFFT backend on gfx950). `LBWalberlaImpl_field_accessors_tests_cuda`: `macroscopic_accessors<float>` 0.47 s, `<double>` 0.40 s, no errors. The timings show the GPU cases ran.
- `ctest -L gpu --timeout 1200 -j 8 -V`: **74/74 passed** (419 s; ctest's 3 "Skipped" are `sample_p3m_with_gpu`, `benchmark_p3m_with_gpu`, `benchmark_lb_with_gpu`, the upstream "NO TESTS RAN" scripts, as on every earlier run). 74 `Ran` lines totalling **917 cases**, 71 `OK` summaries, 0 `FAILED`. Parsed: 847 ok / 60 skipped / **0 FAIL/ERROR**; GPU-class **311 ok, 37 skipped, 0 FAIL/ERROR**; **214 GPU-class `ok` in the LB/EK/lattice scripts, identical to gfx942's 214**: ek_boundary 6, ek_bulk_reactions 2, ek_diffusion 2, ek_diffusive_flux 2, ek_eof 2, ek_ext_efield 2, ek_fixeddensity 2, ek_fixedflux 2, ek_fluctuations 2, ek_friction_coupling 2, ek_indexed_reactions 2, ek_interface 38, ek_noflux 2, ek_potential 2, ek_potential_periodic 2, ek_slice 4, lattice_vtk 17, lb 56, lb_boundary 8, lb_boundary_1_core 8, lb_boundary_ghost_layer 6, lb_boundary_volume_force 2, lb_buoyancy_force 2, lb_circular_couette 2, lb_interpolation_1_core 6, lb_mass_conservation_1_core 2, lb_momentum_conservation_1_core 1, lb_poiseuille 2, lb_poiseuille_cylinder 6, lb_pressure_tensor 2, lb_shear 2, lb_slice 4, lb_streaming_1_core 2, lb_thermostat_1_core 12. The four `*checkpoint__therm_lb__{p3m,elc}_gpu__lj__lb_walberla_gpu_*` tests pass (5 + 5 save cases, 29 + 29 test cases ok). **0 occurrences of "compiled without CUDA support".** GPU-class skips: 1 `TestLBMomentumConservationRegularSinglePrecisionWalberlaGPU` (the test's own "LB with regular decomposition already tested with 2 MPI ranks" filter), and 36 `CheckpointTest` mode/feature filters (features not built: H5MD, STOKESIAN_DYNAMICS, SCAFACOS(_DIPOLES), THERMAL_STONER_WOHLFARTH; upstream mode filters "... not in modes", "missing combination", "Lees-Edwards not implemented for LB GPU", "LB not supported", which apply to CUDA as well). `lattice_vtk`'s GPU classes ran (17 ok) with vtk installed.
- Non-GPU suite `ctest -LE gpu -j 16` (419 tests, 208 s): **391 passed (29 of them ctest "Skipped" samples/benchmarks), 28 failed**: 26 `tutorial_*` Not Run, `test_convert` (`ModuleNotFoundError: No module named 'nbformat'`), `test_install` (never installed). Identical to the gfx942 record at this head (419 / 391 / 28, same set). No regression.

### Maxset myconfig (`work/build-max`)

- Build: **0 FAILED**. Only warnings: 30 known `nodiscard hipError_t`. 58 gfx950 code objects in `espresso_walberla.so`.
- `check_unit_tests`: **152/152 passed**.
- `ctest -L gpu --timeout 1200 -j 8 -V`: **74/74 passed** (485 s; the same 3 ctest Skipped). 917 cases ran, 71 `OK`, 0 `FAILED`. Parsed: 851 ok / 56 skipped / **0 FAIL/ERROR**; GPU-class **311 ok, 37 skipped (the same filters), 0 FAIL/ERROR; 214 ok in the LB/EK/lattice scripts**; 0 "compiled without CUDA support". `lb_electrohydrodynamics` (`LBEHTest.test ... ok`), skipped in the default build for the missing `LB_ELECTROHYDRODYNAMICS`, ran and passed here.

No tolerance was changed, and no test missed: there is no finding to classify at this head on gfx950.

### CUDA no-regression compile at 917d6f707 (nvcc on this host), redone as requested

nvcc 12.8 (V12.8.93, `/usr/local/cuda`), no NVIDIA GPU: compile and link only. The `cuda12-maxset` recipe of "CUDA no-regression compile 2026-11-20" (maxset myconfig, `ESPRESSO_WARNINGS_ARE_ERRORS=ON`, waLBerla ON, Python OFF, Caliper OFF), with this job's OpenMPI/Boost (`agent_space/cuda-build.sh <build>`), from the same clean checkout at 917d6f707 into `work/build-cuda`:

```
ninja -C <build> -k 0 -j 64 all <the 154 unit-test executables: ninja -t targets all, src/**/tests|unit_tests, minus */test>
```

- Cache: `WALBERLA_BUILD_WITH_CUDA=on`, `WALBERLA_BUILD_WITH_HIP=OFF`, `Heffte_ENABLE_CUDA=ON`, `Heffte_ENABLE_ROCM=OFF`, `ESPRESSO_BUILD_WITH_HIP=OFF`, `Kokkos_ENABLE_CUDA=OFF`.
- Result: **690/690 steps, 0 FAILED, 0 warnings under -Werror** (134 s). 73 nvcc compile commands (67 in `src/walberla_bridge`). `espresso_walberla.so` carries 61 cubins each for sm_75/86/89. `PoissonSolver_test_cuda` and `LBWalberlaImpl_field_accessors_tests_cuda` link. PASS; agrees with the review's run on the gfx942 host.

### Gates

- `python3 utils/jargon.py --port espresso`: clean (needs local `python` and `moat-port` branches in the clone; a fresh clone has neither, and jargon then says "cannot resolve python..moat-port").
- Docs: "AMD GPU acceleration" in `doc/sphinx/installation.rst` (:300), `ESPRESSO_BUILD_WITH_HIP` in the configure example (:311) and the options list (:917).
- Integrity: `git -C projects/espresso/src status --porcelain` empty at 917d6f707. Nothing was edited in the fork and nothing was pushed to it.
- Coverage: wave64 is satisfied at 917d6f707 by gfx942 and now gfx950. **wave32 is still owed**: linux-gfx1201 validated bc8d73dd9, and the waLBerla HIP kernels have not yet run on a 32-wide GPU. Windows is waived.
  -DESPRESSO_BUILD_WITH_PYTHON=ON -DESPRESSO_BUILD_TESTS=ON [-DESPRESSO_MYCONFIG_FILE=<src>/maintainer/configs/maxset.hpp]
cmake --build <build> -j 12 -- -k 0
cmake --build <build> -j 12 --target python_test_data cuda_test local_samples local_benchmarks
cmake --build <build> -j 12 --target check_unit_tests
cmake --build <build> -j 12 --target local_samples local_benchmarks   # before every ctest run
ctest --test-dir <build> -L gpu --timeout 1200 -V
ctest --test-dir build-hip -LE gpu --timeout 1200 -j 12
```

GPU-class cases were counted from the `-V` logs with a parser that collects unittest cases whose class name contains GPU or CUDA, including statuses that a docstring or warning line pushes onto a later line (it resolves 907 cases per suite, 0 unresolved).

### Default myconfig (`build-hip`)

- Build: 507/507, **0 FAILED**; only warnings are 20 known `nodiscard hipError_t` `-Wunused-value`. Cache: `WALBERLA_BUILD_WITH_HIP=on`, `WALBERLA_BUILD_WITH_CUDA=OFF`, `Heffte_ENABLE_ROCM=ON`, `Heffte_ENABLE_CUDA=OFF`, `Kokkos_ENABLE_HIP/CUDA=OFF`. 69 HIP objects compiled (no ccache). `llvm-objdump --offloading build-hip/src/walberla_bridge/espresso_walberla.so`: **58 code objects, all `hip-amdgcn-amd-amdhsa--gfx1201`** (the MI300X build has 58 gfx942).
- `check_unit_tests`: **152/152 passed**. Run directly with `--log_level=test_suite`: `PoissonSolver_test_cuda` `ek_poisson_solver_fft_cuda<float>` 1.53 s, `<double>` 1.37 s, `ek_poisson_solver_none_cuda` 0.56 s; `LBWalberlaImpl_field_accessors_tests_cuda` `macroscopic_accessors<float>` 1.04 s, `<double>` 0.79 s; "No errors detected" in both. The timings show the heFFTe rocFFT and accessor GPU cases ran.
- `ctest -L gpu --timeout 1200 -V` (serial, ctest's default): **74/74 passed** (1617 s). 71 unittest `OK`, 0 `FAILED`. **243 GPU-class cases `ok`, 1 skipped, 0 FAIL/ERROR** (MI300X: 243 / 1 / 0). **236 `ok` in 37 LB/EK/lattice scripts** (the ek_*, lb*, lattice_vtk, observable_*LB and propagation_lb scripts), among them ek_interface 38, lb 56, lattice_vtk 17 (vtk installed), lb_pressure_tensor 2, lb_thermostat_1_core 12, ek_eof 2, ek_potential 2, ek_potential_periodic 2 (the FFT Poisson solver). The four `save_/test_checkpoint__*lb_walberla_gpu*` tests pass. The one GPU-class skip is the test's own filter (`TestLBMomentumConservationRegularSinglePrecisionWalberlaGPU`: "LB with regular decomposition already tested with 2 MPI ranks"). **0 occurrences of "compiled without CUDA support".** The other skips are features this config does not build (MMM1D, STOKESIAN_DYNAMICS, SCAFACOS(_DIPOLES), DIPOLE_FIELD_TRACKING, THERMAL_STONER_WOHLFARTH, H5MD, LB_ELECTROHYDRODYNAMICS) or upstream's own filters ("Momentum is not conserved on single precision", the checkpoint tests' "... not in modes", "Lees-Edwards not implemented for LB GPU", "LB not supported", "missing combination"). ctest Skipped (exit 5, no test methods, as every earlier run): `sample_p3m_with_gpu`, `benchmark_p3m_with_gpu`, `benchmark_lb_with_gpu` (which now runs LB on the GPU on import instead of raising `FeaturesError: WALBERLA`).
- Non-GPU suite `ctest -LE gpu -j 12` (419 tests, 315 s): **391 passed (362 Passed + 29 exit-5 Skipped), 28 failed**: 26 `tutorial_*` Not Run, `test_convert` (`ModuleNotFoundError: No module named 'nbformat'`), `test_install` (`diff: /usr/local/lib/python3.12/site-packages/espressomd/...: No such file or directory`). This is the MI300X record's count (419 / 391 / 28) and failure set exactly. No regression.

### Maxset myconfig (`build-max`)

- Build: 507/507, **0 FAILED**; 22 known `nodiscard hipError_t` warnings. Same cache values.
- `check_unit_tests`: **152/152 passed**.
- `ctest -L gpu --timeout 1200 -V` (serial): **74/74 passed** (1540 s). 71 `OK`, 0 `FAILED`. **243 GPU-class `ok`, 1 skipped (the same 2-rank filter), 0 FAIL/ERROR, 236 `ok` in the 37 LB/EK/lattice scripts, 0 "compiled without CUDA support".** `lb_electrohydrodynamics` (`LBEHTest`, 4 ranks), skipped in the default config for the missing feature, ran and passed. dipole_field_tracking ran (no DIPOLE_FIELD_TRACKING skip). The remaining skips are features maxset does not build or the upstream filters listed above.

No tolerance was changed.

### Finding: `lb_pressure_tensor` times out under `ctest -L gpu -j 8` (environment: GPU shared by concurrent processes)

The first default-build GPU run used the MI300X invocation `ctest -L gpu --timeout 1200 -j 8 -V`. It ended 73/74 with `lb_pressure_tensor` at ***Timeout 1200 s*** after its two CPU classes passed (the 241 other GPU-class cases passed, 0 "compiled without CUDA support"). It is not a wrong result and not a 32-wide fault:

- The same build, the GPU class alone (`pypresso lb_pressure_tensor.py --verbose TestLBPressureTensorGPU`): both cases (`test_averages`, `test_gk_viscosity`) **ok in 77 s**. The whole script alone (`ctest -R '^lb_pressure_tensor$' -V`): 4/4 ok, **89 s**. In the serial suites: 147 s (default), 90 s (maxset).
- Two copies of the GPU class at once on the pinned GPU: **578 s each** (7.5x slower for 2x the work). The test's sampling loop does 10000 steps of `integrator.run(2)` plus a device-to-host read of the pressure tensor, so it synchronizes constantly.
- Smallest reproducer, plain HIP with no ESPResSo (`syncloop.hip`: four small kernels on a 12^3 x 19 double field, then a 72-byte `hipMemcpy` D2H, 20000 iterations; `clang++ -x hip --offload-arch=gfx1201 -O2`): one process 30.1 us per iteration; **two concurrent processes 138.5 us each**, four 135 us each. A process that only holds an idle context does not slow it (31.6 us). So each synchronizing step pays roughly 100 us once a second process submits work to the same GPU. Eight concurrent ctest jobs, several of them LB GPU scripts, push a sync-bound test past its timeout.

Class: environment (a host or driver property of sharing one GPU between processes), not 32-wide-specific (no result differs; the kernels pass), not a HIP-build gap, and not a test bug. The documented `ctest -L gpu` is serial and passes. Not compared with a wave64 host: the MI300X ran `-j 8` without the timeout, but this host did not test whether the cost is specific to RDNA4 or to this driver. No fork change is warranted.

### Gates

- CUDA no-regression compile: not rerun. This host has no CUDA toolkit (no nvcc, no /usr/local/cuda, no /opt/conda). It is already PASS at 917d6f707 (review section above: nvcc 12.8, `cuda12-maxset`, 0 FAILED under -Werror).
- `python3 utils/jargon.py --port espresso`: clean. Docs: "AMD GPU acceleration" in `doc/sphinx/installation.rst` covers the LB/EK solvers (unchanged since the gfx942 record).
- Integrity: `git -C projects/espresso/src status --porcelain` empty at 917d6f707.
- With gfx942 (wave64) and this run (wave32) at 917d6f707, both required Linux gates are met at head; windows is waived.

## Lesson for cuda-to-rocm: a sync-bound GPU test that passes alone can time out under `ctest -jN` because the GPU is shared between processes

Belongs in the validation or Linux host reference, beside "A GPU test that hangs only under `ctest -jN`". On a Radeon gfx1201 host (ROCm 10.1 SDK), a loop of small kernels plus a synchronous small device-to-host copy took 30 us per iteration in one process and 135-139 us per iteration in each of two or four concurrent processes on the same GPU. An idle second context cost nothing; only concurrent submission did. A test that synchronizes every step (ESPResSo's `lb_pressure_tensor`, 10000 sample steps) passed in 77-147 s alone and timed out at 1200 s under `ctest -L gpu -j 8`. Before treating such a timeout as a hang or a port fault, rerun the test alone and then two copies at once: a pass alone plus a superlinear slowdown with two copies points to GPU sharing, and the fix is a serial GPU ctest run (or a `RESOURCE_LOCK`), not a longer timeout or a code change.

## Owed next (2026-11-20, after gfx1201 validation at 917d6f707)

- wave64 (gfx942) and wave32 (gfx1201) are validated at 917d6f707; windows is waived. linux-gfx950 still reads `revalidate` (extra wave64 evidence, gates nothing).
- For jeff (unchanged): refresh the fork review page (AMD-Ecosystem/espresso#1, still at bc8d73dd9, body describes the core only) for the five-commit series before approval. No upstream PR is open.
- For a trunk session: promote the four `## Lesson for cuda-to-rocm` sections (the three from the porter round, and the `ctest -jN` GPU sharing one above).

## Ruling 2026-11-23 (jeffdaily): "I want the fix round"

The port was validated at 917d6f707 and its fork review page (AMD-Ecosystem/espresso#1) was about to be approved; no upstream pull request is open (`moatlib.py pr-state espresso`: `none`). Reading the diff, jeff found two things and ruled "I want the fix round" before the pull request opens:

1. **One flag, not two.** The documented AMD build needed `-D ESPRESSO_BUILD_WITH_CUDA=ON -D ESPRESSO_BUILD_WITH_HIP=ON`, and every HIP block in the top-level CMakeLists.txt tested both, so `ESPRESSO_BUILD_WITH_HIP=ON` alone silently gave a CPU-only build. `ESPRESSO_BUILD_WITH_HIP=ON` is to imply the GPU switch, in the project's CMake style (deliberate choice between a cache-forcing `set` and a normal variable), with the installation guide, option list and any message text updated; setting both must keep working and give the same build files.
2. **The duplicated-looking test string** in `src/core/unit_tests/cuda_test.cu`: keep one copy of the string and switch only the differing word, with a short comment saying why; check whether HIP has a closer error code than `hipErrorNotSupported`.

The round stays on these two points. Commits go on top of 917d6f707 (no amend, no forced push); jeff's session folds them into the existing five commits afterwards.

## Porter 2026-11-23 (fix round, linux-gfx942)

Host: AMD Instinct MI300X (gfx942), `HIP_VISIBLE_DEVICES=0` (pinned by the dispatcher), `ulimit -c 0`, `MAX_JOBS=37`. Pip ROCm SDK 10.1.0 (HIP 7.16, `rocm-sdk path --root`, no /opt/rocm), CUDA 12.8 (`/usr/local/cuda/bin/nvcc`, V12.8.93), system OpenMPI 4.1.6, Boost 1.83, FFTW 3.3.10, GCC 13.3.0, CMake 3.31.6. Test venv `$FLEET_VENV/py312` by `uv` (numpy 2.5.5, scipy, Cython, tqdm, matplotlib, pint, vtk; for the docs also sphinx 9.1.1, sphinx-toggleprompt 0.6.0, sphinxcontrib-bibtex, sphinx-tabs, numpydoc, pybtex, ase 3.29.0, asebytes 0.1.7, zndraw 0.6, h5py, requests -- the `requirements.txt` lines; without ase/zndraw the `sphinx` target fails with autodoc import warnings, an environment gap). Lock taken with `set-state espresso linux-gfx942 porting` (no warning).

Fork `moat-port` 917d6f707 -> **bbe25fdf8**, four commits, pushed plainly:

| commit | title | fold into |
|---|---|---|
| 3d1a31b5f | [ROCm] Let ESPRESSO_BUILD_WITH_HIP enable GPU support on its own | bc8d73dd9 (core) |
| a62828c7d | [ROCm] Select waLBerla's HIP backend from ESPRESSO_BUILD_WITH_HIP | 3862a5040 (LB/EK GPU) |
| ab466b714 | [ROCm] Select heFFTe's rocFFT backend from ESPRESSO_BUILD_WITH_HIP | 17094fdf3 (EK FFT) |
| bbe25fdf8 | [ROCm] Keep one expected message in the CUDA error test | bc8d73dd9 (core) |

e8fa4dfa5 (waLBerla decoupling) and 917d6f707 (Caliper) get no fixup, but see the fold conflicts below.

### Point 1: the implication, and why a normal variable

`CMakeLists.txt`, right after the HIP option:

```
option(ESPRESSO_BUILD_WITH_HIP
       "Build with GPU support for AMD GPUs, compiled with HIP (ROCm)" OFF)
if(ESPRESSO_BUILD_WITH_HIP)
  # a HIP build is a GPU build; the cache keeps ESPRESSO_BUILD_WITH_CUDA as set
  set(ESPRESSO_BUILD_WITH_CUDA ON)
endif()
```

The four `if(ESPRESSO_BUILD_WITH_CUDA AND ESPRESSO_BUILD_WITH_HIP)` (top-level HIP language block, heFFTe ROCm switch, waLBerla HIP switch, `src/config` toolchain metadata) become `if(ESPRESSO_BUILD_WITH_HIP)`, which is now the same condition. The `CUDA AND NOT HIP` conditions (nvcc-only paths: OpenMP CUDA component, Kokkos CUDA, Caliper NVTX/CUPTI, `ESPRESSO_OPENMP_CUDA`) and the nested `if(CUDA) ... if(HIP)` blocks are unchanged.

Project style: upstream never forces a user option and has no option that implies another. It combines options where they are used (`ESPRESSO_BUILD_WITH_KOKKOS_FFT AND ESPRESSO_BUILD_WITH_FFTW`), refuses conflicting ones (`ASAN AND MSAN` -> FATAL_ERROR), and uses normal variables for derived settings (`ESPRESSO_BUILD_WITH_WALBERLA_AVX_DEFAULT`). `FORCE` appears only for `CMAKE_BUILD_TYPE` (`espresso_option_enum`) and the detected `ESPRESSO_NUMPY_SITEARCH`; `CMakeDependentOption` is not used anywhere.

Choice: a normal variable.
- A forced cache entry (`set(... CACHE BOOL ... FORCE)`) would rewrite the user's own setting and ratchet: switching HIP off later in the same build directory (ccmake, or `-DESPRESSO_BUILD_WITH_HIP=OFF`) would leave `ESPRESSO_BUILD_WITH_CUDA=ON` cached, and the next configure would take the nvcc path (`enable_language(CUDA)`) on an AMD machine. It would also overwrite an explicit `-DESPRESSO_BUILD_WITH_CUDA=OFF` permanently.
- The normal variable shadows the cache only for this configure: every consumer (`if()`, `$<BOOL:...>`, `#cmakedefine ESPRESSO_BUILD_WITH_CUDA` in `cmake_config.hpp`, subdirectories, FetchContent projects) sees ON, and the cache keeps exactly what the user typed. Switching HIP off returns to the user's value.
- The cost: `ccmake`/`cmake-gui` show `ESPRESSO_BUILD_WITH_CUDA` as the user's value (OFF in a HIP-only configure) while GPU support is on. The HIP option's help text now says it builds with GPU support, and the option list in the guide says it implies the GPU option. `cmake_dependent_option(ESPRESSO_BUILD_WITH_CUDA ... "NOT ESPRESSO_BUILD_WITH_HIP" ON)` would hide the entry while HIP is on and restore it after, but it rewrites upstream's own CUDA option declaration and adds a module upstream does not use, for a display detail; not taken.
- Feature summary: `feature_summary(WHAT ALL)` lists packages and `add_feature_info` entries only (ESPResSo adds one, for HDF5); neither GPU option appears in it, so the choice does not change it. Verified: the configure output, feature summary included, is identical between HIP-only and two-flag at the new head, and between 917d6f707 and the new head for the two-flag, CUDA-only and neither cases (build-directory paths and timings normalised).

Other places with the two-flag form: the installation guide's code block (now `cmake .. -D ESPRESSO_BUILD_WITH_HIP=ON \`) and its option list (now "Build with GPU support for AMD GPUs, compiled with HIP; implies ``ESPRESSO_BUILD_WITH_CUDA``"). No CMake `message()` text names either option combination (the waLBerla FATAL_ERROR of bc8d73dd9 that named `ESPRESSO_BUILD_WITH_HIP` is removed by e8fa4dfa5). No CI config was touched by the port (`.gitlab-ci.yml`, `maintainer/CI/` are not in `git diff python..moat-port`). `cuda_to_hip.h`'s comment ("selected by the ESPRESSO_BUILD_WITH_HIP CMake option") stays true. `electrostatics.rst`, `magnetostatics.rst` and `running.rst` name only `-D ESPRESSO_BUILD_WITH_CUDA=ON` for the GPU features; that is upstream text and remains true (the HIP option now implies it), so left alone. The Test Plans of all five existing commit messages show `-DESPRESSO_BUILD_WITH_CUDA=ON -DESPRESSO_BUILD_WITH_HIP=ON` (bc8d73dd9 lines 25 and 56, e8fa4dfa5:20, 3862a5040:32, 17094fdf3:21, 917d6f707:15); the commands still work. bc8d73dd9's body also says the option is used "together with `ESPRESSO_BUILD_WITH_CUDA=ON`" (line 3) and that "`cuda_test.cu` expects the error string "operation not supported" under HIP" (line 11). jeff's fold should update those sentences.

### Point 2: the test string, and the HIP error codes

```
    // HIP has no "not permitted" error: it maps to "not supported"
#if defined(__HIP__)
    std::string const reason = "not supported";
#else
    std::string const reason = "not permitted";
#endif
    std::string const what = "CUDA error: \"operation " + reason +
                             "\" during memory operation in filename.cu:4";
```

The file builds strings with `std::string` and `+` elsewhere (`read_pending_cuda_errors`). clang-format passes.

Closer HIP code: none. The `hipError_t` enum of HIP 7.16 (`include/hip/hip_runtime_api.h`, ROCm 10.1 pip SDK) has no "not permitted" member. The nearest-sounding are stream-capture codes whose messages say "operation not permitted when stream is capturing" (`hipErrorStreamCaptureUnsupported` = 900, the counterpart of `cudaErrorStreamCaptureUnsupported`), which are a different error. hipify-perl in the same SDK lists `cudaErrorNotPermitted` in `%hash_HipOnlyUnsupportedFunctions` (line 5925), so the ROCm tools themselves have no mapping for it. `hipErrorNotSupported` (801) stays, and the `#if` stays, reduced to the one word.

### Evidence

Build-file comparison (`agent_space/cmpcfg.sh <build> hip1|hip2|cuda|none`: the porter's waLBerla-ON recipe, configure only, `cuda` with `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`; base and head configured from the same source path, `projects/espresso/src`, at 917d6f707 and then at the new tree; `agent_space/treecmp.py` compares every file of the two build trees except fetched sources, configure logs and object directories, with the build path normalised; 946-968 files per tree):

- two flags, 917d6f707 vs head: identical except `CMakeCache.txt` (the HIP option's new help string) and CMake's compiler-ABI probe binary `CMakeFiles/3.31.6/CMakeDetermineCompilerABI_HIP.bin`. Every `build.ninja`, `CMakeFiles/*.ninja`, generated header (`cmake_config.hpp`, `myconfig` files), CTest file and script is the same.
- CUDA only (nvcc), 917d6f707 vs head: the same plus CMake's nvcc compiler-id scratch files (`CompilerIdCUDA/tmp/*.cudafe1.cpp`, `*.module_id`: nvcc's random anonymous-namespace hash) and the ABI/OpenMP probe binaries.
- neither, 917d6f707 vs head: identical except the help string in `CMakeCache.txt`.
- HIP only vs two flags, both at head: identical except `CMakeCache.txt` `ESPRESSO_BUILD_WITH_CUDA:BOOL=OFF` (the user's value, by the choice above) and the HIP ABI probe binary.
- The defect, HIP only at 917d6f707: configures without error, cache `WALBERLA_BUILD_WITH_HIP=OFF`, `Heffte_ENABLE_ROCM=OFF`, **0** HIP objects in `build.ninja` (two flags: 146), and the only hint is "Manually-specified variables were not used by the project: CMAKE_HIP_COMPILER".

The probe binaries and the nvcc scratch files differ between any two configures (embedded build ids and random hashes), not because of the change.

HIP build of the HIP-only configuration (`agent_space/cmp/head-hip1`, from the working tree that became bbe25fdf8; `git status --porcelain` empty after the commits, so the tree is the committed one):

- `cmake --build -j 37 -- -k 0` then `python_test_data cuda_test local_samples local_benchmarks`: 0 FAILED; only warnings: 28 known `-Wunused-value` (`nodiscard hipError_t`), as at 917d6f707.
- `check_unit_tests`: **152/152 passed**. `cuda_test --log_level=test_suite`: `gpu_interface` ran, "No errors detected".
- `ctest -L gpu --timeout 1200 -j 8 -V`: **74/74 passed** (757 s). `agent_space/parse_gpu.py`: **243 GPU-class cases ok, 1 skipped** (`TestLBMomentumConservationRegularSinglePrecisionWalberlaGPU`, the test's own 2-rank filter), **0 FAIL/ERROR**, 214 ok in 33 LB/EK/lattice scripts, 71 unittest `OK`, 0 `FAILED`, 0 "compiled without CUDA support". Identical to the 917d6f707 record.
- `sphinx` target (the project's target runs `sphinx -q -W`, warnings as errors), after reconfiguring the same directory so CMake finds Sphinx 9.1.1: **passes**, no warning. The rendered `installation.html` shows `cmake .. -D ESPRESSO_BUILD_WITH_HIP=ON \` and the new option-list entry.

CUDA no-regression compile at the new tree (nvcc 12.8, the `cuda12-maxset` recipe of "CUDA no-regression compile 2026-11-20": maxset myconfig, `ESPRESSO_WARNINGS_ARE_ERRORS=ON`, waLBerla ON, Python OFF, Caliper OFF; `agent_space/cuda-build.sh <src> agent_space/build-cuda`, `all` plus the 154 unit-test executables, `-j 37`): **690/690 steps, 0 FAILED, 0 compiler warnings under -Werror**, 73 nvcc compile commands. Cache `ESPRESSO_BUILD_WITH_HIP=OFF`, `WALBERLA_BUILD_WITH_CUDA=on`, `WALBERLA_BUILD_WITH_HIP=OFF`, `Heffte_ENABLE_CUDA=ON`, `Heffte_ENABLE_ROCM=OFF`. `cuda_test.cu` is compiled by `/usr/local/cuda/bin/nvcc` (compile_commands.json) and `cuda_test` links; the binary contains the string "not permitted" and not "not supported", so the CUDA branch expects the same message as before.

Checks: `pre-commit run` (clang-format 22.1.8, cmake-format 0.6.13) on the four changed files passes and changes nothing. `jargon.py --port espresso`: clean. `prose.py` on the four commit bodies: clean. Titles 55-66 characters. `moatlib.py classify espresso 917d6f707 bbe25fdf8`: `mixed`, not inert (CMake and a test source), so the review and every platform's validation follow.

### Fold dry run (for jeff's session)

Done locally in a scratch worktree (nothing pushed): from c0e34ecc0, `bc8d73dd9 + 3d1a31b5f + bbe25fdf8`, `e8fa4dfa5`, `3862a5040 + a62828c7d`, `17094fdf3 + ab466b714`, `917d6f707`, each squashed with the original message. The final tree is **identical** to bbe25fdf8. Two folds conflict, both trivially:

- Folding 3d1a31b5f into bc8d73dd9: at bc8d73dd9 the HIP block still starts with the waLBerla FATAL_ERROR (`if(ESPRESSO_BUILD_WITH_WALBERLA) message(FATAL_ERROR "ESPRESSO_BUILD_WITH_HIP requires ESPRESSO_BUILD_WITH_WALBERLA=OFF, ...")`), and the guide's command has a second line `-D ESPRESSO_BUILD_WITH_WALBERLA=OFF \`. Resolution: `if(ESPRESSO_BUILD_WITH_HIP)` keeping the FATAL_ERROR block inside it; `cmake .. -D ESPRESSO_BUILD_WITH_HIP=ON \` followed by the `-D ESPRESSO_BUILD_WITH_WALBERLA=OFF \` line.
- Replaying e8fa4dfa5 on the folded commit 1: the same two hunks. Resolution: `if(ESPRESSO_BUILD_WITH_HIP)` without the FATAL_ERROR block; `cmake .. -D ESPRESSO_BUILD_WITH_HIP=ON \` without the waLBerla line.
- Stats after the fold: commit 1 is 17 files, +216/-45 (the review page says +209/-43); commits 2-5 keep 6 files +24/-19, 57 files +582/-444, 6 files +41/-24, 1 file +7/-2.

### Draft replacement lines for the review page body (AMD-Ecosystem/espresso#1; jeff's to apply)

Description, first bullet, replace "It is selected with a new CMake option `ESPRESSO_BUILD_WITH_HIP` (default OFF) on top of `ESPRESSO_BUILD_WITH_CUDA=ON`." with:

```
It is selected with a new CMake option `ESPRESSO_BUILD_WITH_HIP` (default OFF), which turns on GPU support (`ESPRESSO_BUILD_WITH_CUDA`) by itself.
```

Commit 1 heading, replace "(17 files, +209/-43)" with "(17 files, +216/-45)", and add a bullet after the `utils/device_qualifier.hpp` one:

```
- `ESPRESSO_BUILD_WITH_HIP=ON` sets `ESPRESSO_BUILD_WITH_CUDA` to ON as a normal variable, so one option selects the AMD build and setting both gives the same build files. The cache keeps the user's own value of `ESPRESSO_BUILD_WITH_CUDA`, so switching HIP off again in the same build directory does not leave an nvcc build configured.
```

"Building for AMD GPUs", replace the code block with:

```bash
cmake .. -D ESPRESSO_BUILD_WITH_HIP=ON \
  -D CMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -D CMAKE_PREFIX_PATH=/opt/rocm
```

"Style", replace "The `sphinx` target was built with warnings as errors for commit 1; it was not rebuilt after commits 2 and 3 changed the wording of the new section." with:

```
The `sphinx` target, which builds with warnings as errors, passes at the last commit (Sphinx 9.1.1).
```

The Testing section's numbers do not change if this round's validation reproduces them (MI300X: 152/152, 74/74, 243 / 1 / 0, below). The MI355X and R9700 runs were at 917d6f707; the round's build files are identical for the two-flag command those runs used, but they are not re-run here. The V100 CUDA runs were at 917d6f707 too; this round's CUDA build files are identical to 917d6f707's, and the CUDA 12.8 compile line ("690 of 690 steps") was redone at the new head.

## Review 2026-11-23 (fix round delta 917d6f707..bbe25fdf8, linux-gfx942)

Reviewer on the MI300X host, pr-review skill, fork `moat-port` at bbe25fdf8 (3d1a31b5f, a62828c7d, ab466b714, bbe25fdf8 on 917d6f707; 4 files, +16/-11). Verdict: **review-passed**. No fork finding. One commit-message wording point is left for jeff's fold (below), because a pushed message cannot change without amending.

### For jeff's fold (commit message wording)

- 3d1a31b5f's Test Plan says every generated file of the four build directories is the same "apart from the new help text of the HIP option in the cache and, for the HIP-only configuration, the cached `ESPRESSO_BUILD_WITH_CUDA:BOOL=OFF`". CMake's own compiler-detection probe binaries (`CMakeFiles/<ver>/CMakeDetermineCompilerABI_*.bin`, nvcc's `CompilerIdCUDA/tmp` scratch files) also differ between any two configures. When the text is folded into bc8d73dd9's message, say "every build file (`build.ninja`, generated headers, CTest files)" rather than "every generated file". Its "the same as the two-option build" rests on identical build files plus the 917d6f707 two-flag record (152/152, 74/74), not on a second build at the head. That is sound, and the fold can say so.

### Verified (no action)

- **The implication precedes every reader.** The normal variable is set at `CMakeLists.txt:109-112`, right after the option. Before it, only `FeatureSummary`, `GNUInstallDirs`, `ExternalProject`, `FetchContent`, `espresso_option_enum`, `espresso_enable_avx2_support` and `espresso_override_clang_tidy_checks` are included, and none of them reads the switch. Every other consumer runs in a directory scope that inherits the variable, or in a function called from one: the top-level blocks, `src/config` (the configure_file of `cmake_config.hpp` with `#cmakedefine ESPRESSO_BUILD_WITH_CUDA`), `src/core`, `src/core/unit_tests`, `src/walberla_bridge/**`, `src/python/espressomd:146`, `testsuite/CMakeLists.txt:85`, `testsuite/python:116`, `cmake/espresso_unit_test.cmake:52` (called from subdirectories). Nothing reads `$CACHE{ESPRESSO_BUILD_WITH_CUDA}` or the cache file (the only `CMakeCache.txt` reader, `maintainer/CI/run_lcov.sh`, reads the source dir). This is confirmed by the porter's tree comparison, re-read here: the HIP-only and two-flag trees have identical `build.ninja`, `CMakeFiles/*.ninja`, `cmake_config.hpp`, CTest files (GPU labels and resource locks included) and configure output.
- **The simplified conditions are equivalent.** After line 111, `ESPRESSO_BUILD_WITH_CUDA AND ESPRESSO_BUILD_WITH_HIP` equals `ESPRESSO_BUILD_WITH_HIP`. No such conjunction remains (`grep`). The `CUDA AND NOT HIP` conditions keep their meaning (nvcc-only paths). In a CUDA-only or CPU build the implication does nothing, so those configures are unchanged (porter's comparison: build files identical to 917d6f707, cache differs only in the help string).
- **Normal variable vs forced cache.** The reasoning in the porter section holds against the project: no upstream user option is forced or implied. FORCE is used only for `CMAKE_BUILD_TYPE` (`cmake/espresso_option_enum.cmake:36`) and the detected NumPy path (`CMakeLists.txt:412`). A forced entry would make `-DESPRESSO_BUILD_WITH_HIP=OFF` in the same directory configure an nvcc build. An explicit `-DESPRESSO_BUILD_WITH_HIP=ON -DESPRESSO_BUILD_WITH_CUDA=OFF` gives a HIP build, which matches the new help text. `maintainer/benchmarks/runner.sh:53` passes `-D ESPRESSO_BUILD_WITH_CUDA=OFF` for a CPU benchmark build in a new directory, where HIP is OFF by default, so it is unaffected.
- **Heffte/waLBerla.** At head the HIP-only cache has `Heffte_ENABLE_ROCM:BOOL=ON` and `WALBERLA_BUILD_WITH_HIP:BOOL=on`, as a62828c7d/ab466b714 state. The nvcc cache has `Heffte_ENABLE_CUDA=ON`, `WALBERLA_BUILD_WITH_CUDA=on`, `ESPRESSO_BUILD_WITH_HIP=OFF`.
- **Test string.** `"CUDA error: \"operation " + reason + "\" during memory operation in filename.cu:4"` builds the same two strings as before (`const char*` + `std::string`). The HIP run's `gpu_interface` passes, and the nvcc `cuda_test` binary holds "not permitted" only. The HIP error-code check was repeated here: the `hipError_t` enum in the SDK has no "not permitted" member, and hipify-perl lists `cudaErrorNotPermitted` as unsupported in HIP, so the `#if` cannot go. The comment explains the word.
- **Docs.** The rendered `installation.html` (sphinx `-W` passed) shows the single-option command, and the option list says the HIP option implies `ESPRESSO_BUILD_WITH_CUDA`. Other mentions of the two-flag form exist only in commit-message Test Plans and the review page body (listed in the porter section, drafts there).
- **Commit hygiene.** Four `[ROCm]` titles of 55-66 characters (every change names HIP). Bodies are ASCII, one line per paragraph, with the AI-assistance sentence and fenced Test Plans, and no Co-Authored-By. `jargon.py --port espresso` is clean. The fold dry run reproduces the head tree exactly, and its two trivial conflicts are written down for jeff.
- **Scope.** The round touches only the two ruled points. No source compiled into the GPU libraries changes. The only compiled change is `cuda_test.cu`, a test.

## Validation 2026-11-24 (linux-gfx942, MI300X wave64, fix round head bbe25fdf8)

Validator on the MI300X host: AMD Instinct MI300X (gfx942), `HIP_VISIBLE_DEVICES=0` (pinned by the dispatcher), `ulimit -c 0`, `MAX_JOBS=37`. Toolchain and venv as in "Porter 2026-11-23". Source: `projects/espresso/src` at bbe25fdf8 (= `origin/moat-port`), clean tree. The method and counts follow "Validation 2026-11-20 (linux-gfx942 ...)", with one difference: the configure is the newly documented **single-option** command (`agent_space/cmpcfg.sh <build> hip1`, i.e. the porter's waLBerla-ON recipe with `-DESPRESSO_BUILD_WITH_HIP=ON` and without `-DESPRESSO_BUILD_WITH_CUDA=ON`). Its build files are identical to the two-option command's (porter section). Each build is in a **new** directory, and the whole run is `agent_space/validate.sh`:

```
cmake -S <src> -B <build> -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DESPRESSO_BUILD_WITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_HIP_COMPILER=$ROCM_SDK/llvm/bin/clang++ -DCMAKE_PREFIX_PATH="$ROCM_SDK" \
  -DPython_EXECUTABLE=$FLEET_VENV/py312/bin/python \
  -DESPRESSO_BUILD_WITH_WALBERLA=ON -DESPRESSO_BUILD_WITH_HDF5=OFF \
  -DESPRESSO_BUILD_WITH_SCAFACOS=OFF -DESPRESSO_BUILD_WITH_STOKESIAN_DYNAMICS=OFF \
  -DESPRESSO_BUILD_WITH_GSL=OFF -DESPRESSO_BUILD_WITH_NLOPT=OFF \
  -DESPRESSO_BUILD_WITH_CALIPER=OFF -DESPRESSO_BUILD_WITH_FFTW=ON \
  -DESPRESSO_BUILD_WITH_PYTHON=ON -DESPRESSO_BUILD_TESTS=ON \
  [-DESPRESSO_MYCONFIG_FILE=<src>/maintainer/configs/maxset.hpp]
cmake --build <build> -j 37 -- -k 0
cmake --build <build> -j 37 --target python_test_data cuda_test local_samples local_benchmarks
cmake --build <build> -j 37 --target check_unit_tests
cmake --build <build> -j 37 --target local_samples local_benchmarks   # before every ctest run
ctest --test-dir <build> -L gpu --timeout 1200 -j 8 -V
ctest --test-dir <build> -LE gpu --timeout 1200 -j 16                   # default build only
```

### Default myconfig (`agent_space/val/build-hip`)

- Build: **0 FAILED**. Only warnings: 28 known `-Wunused-value` (`nodiscard hipError_t`), as at 917d6f707. Cache: `ESPRESSO_BUILD_WITH_HIP=ON`, `ESPRESSO_BUILD_WITH_CUDA=OFF` (the user's value; the build is a GPU build through the implication), `WALBERLA_BUILD_WITH_HIP=on`, `WALBERLA_BUILD_WITH_CUDA=OFF`, `Heffte_ENABLE_ROCM=ON`, `Heffte_ENABLE_CUDA=OFF`. `llvm-objdump --offloading espresso_walberla.so`: **58 gfx942 code objects** (as at 917d6f707).
- `check_unit_tests`: **152/152 passed** (164 s). Run directly with `--log_level=test_suite`: `PoissonSolver_test_cuda`: `ek_poisson_solver_fft_cuda<float>` 1.11 s, `<double>` 1.12 s, `ek_poisson_solver_none_cuda` 0.18 s. `LBWalberlaImpl_field_accessors_tests_cuda`: `macroscopic_accessors<float>` 0.53 s, `<double>` 0.43 s. Both print "No errors detected". The timings show the GPU cases ran.
- `ctest -L gpu --timeout 1200 -j 8 -V`: **74/74 passed** (547 s; the usual 3 ctest Skipped `sample_p3m_with_gpu`, `benchmark_p3m_with_gpu`, `benchmark_lb_with_gpu`). `agent_space/parse_gpu.py`: **243 GPU-class cases ok, 1 skipped, 0 FAIL/ERROR; 214 ok in 33 LB/EK/lattice scripts**; 71 unittest `OK`, 0 `FAILED`; 0 "compiled without CUDA support". The skip is the test's own filter (`TestLBMomentumConservationRegularSinglePrecisionWalberlaGPU`, "LB with regular decomposition already tested with 2 MPI ranks"). These are the 917d6f707 counts exactly.
- Non-GPU suite `ctest -LE gpu -j 16` (419 tests, 380 s): **391 passed, 28 failed**: 26 `tutorial_*` Not Run, `test_convert` (`ModuleNotFoundError: No module named 'nbformat'`), `test_install`. This is the same environmental set as every earlier run. No regression.

### Maxset myconfig (`agent_space/val/build-max`)

- Build: **0 FAILED**. Only warnings: 30 known `-Wunused-value`. `myconfig-final.hpp` carries `LB_ELECTROHYDRODYNAMICS` (absent from the default build's), so maxset was in effect.
- `check_unit_tests`: **152/152 passed**.
- `ctest -L gpu --timeout 1200 -j 8 -V`: **74/74 passed** (547 s). **243 GPU-class ok, 1 skipped (the same filter), 0 FAIL/ERROR, 214 ok in the LB/EK/lattice scripts, 0 "compiled without CUDA support"**. `lb_electrohydrodynamics` passed (skipped in the default build for the missing feature).

No tolerance was changed, and no test missed.

### Gates

- CUDA no-regression compile at this head: done in "Porter 2026-11-23" at the tree of bbe25fdf8 (nvcc 12.8, `cuda12-maxset`, 690/690, 0 warnings under -Werror, `cuda_test` compiled by nvcc and expecting "not permitted"). It runs once per head and is not repeated here.
- `python3 utils/jargon.py --port espresso`: clean.
- Docs: "AMD GPU acceleration" in `doc/sphinx/installation.rst` shows the single-option command, the options list describes `ESPRESSO_BUILD_WITH_HIP` as implying `ESPRESSO_BUILD_WITH_CUDA`, and the `sphinx` target builds with `-W` (porter section).
- Integrity: `git -C projects/espresso/src status --porcelain` is empty at bbe25fdf8. The validator edited and pushed nothing in the fork.

## Owed next (2026-11-24, after the fix round at bbe25fdf8)

- **wave32 (linux-gfx1201)**: the gate is unsatisfied at bbe25fdf8 (validated 917d6f707). The delta is CMake (no compiled-output change for the two-flag command; the build files are identical) plus `cuda_test.cu`, a test-only source. A carry-forward needs `binary-equiv --host-checked` from builds of both shas on that host, or a full run. Use the single-option configure in a new build directory.
- linux-gfx950 reads `revalidate` (extra wave64 evidence, gates nothing). Windows is waived.
- For jeff: fold the four commits into the five (table, conflict resolutions and new stats in "Porter 2026-11-23"). Update bc8d73dd9's body sentences (option "together with `ESPRESSO_BUILD_WITH_CUDA=ON`"; the `cuda_test.cu` bullet) and the reviewer's wording point. Apply the drafted review-page lines. The review page (AMD-Ecosystem/espresso#1) now shows bbe25fdf8 as its head. No upstream PR is open.
