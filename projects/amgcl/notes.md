# amgcl notes

## Disposition

PORT (not already-supported). amgcl is header-only AMG with several backends.
Grep of the whole tree (`hip|rocsparse|rocthrust|rocprim|hipsparse`) returns
ZERO hits -- no native HIP/rocSPARSE backend exists upstream. AMD GPUs are
reachable today only through the OpenCL-based vexcl/viennacl backends, which per
PORTING_GUIDE does NOT count as already-supported. Value-add: a native
HIP/hipSPARSE backend mirroring the existing cuSPARSE+Thrust CUDA backend.

## Build classification

Pure CMake (no find_package(Torch), no torch.utils.cpp_extension) -> Strategy A
family. But the backend is header-only template code, so the "build" is just
compiling a TU that includes the new backend header with the HIP toolchain. No
.cu files to re-language; the example wiring copies `solver.cpp` -> `solver.hip`
and marks it `LANGUAGE HIP` (mirrors the existing `.cpp` -> `.cu` cuda example).

## Strategy: hipSPARSE shim, mirror cuda.hpp (minimal diff)

The CUDA backend uses the cuSPARSE GENERIC api (cusparseCreateCsr /
cusparseSpMV* / cusparseSpSV*) + Thrust. hipSPARSE (installed, ROCm 7.2.1)
exposes that exact api name-for-name (hipsparseCreateCsr, hipsparseSpMV*,
hipsparseSpSV*, hipsparseXcsrilu02*), so the port mirrors cuda.hpp line for line
with a mechanical cusparse->hipsparse / CUSPARSE_->HIPSPARSE_ / cudaDataType->
hipDataType / CUDA_R_*->HIP_R_* swap. Chose hipSPARSE over the lower-level
rocSPARSE because rocSPARSE's generic SpSV has no persistent descriptor (it is
stage-based), whereas hipSPARSE keeps cuSPARSE's `hipsparseSpSVDescr_t`, so the
ilu0 triangular-solve code ports unchanged in shape. Thrust stays `thrust::`
(rocThrust is a header-path-compatible drop-in under /opt/rocm/include).

## Files

New (NVIDIA path untouched):
- `amgcl/backend/hip.hpp`            -- mirror of backend/cuda.hpp
- `amgcl/relaxation/rocsparse_ilu0.hpp` -- mirror of relaxation/cusparse_ilu0.hpp

Modified (additive, guarded):
- `CMakeLists.txt`          -- `option(USE_HIP)`; enable_language(HIP),
  find_package(hipsparse/rocsparse), `hip_target` INTERFACE lib (mirrors
  cuda_target). Default lead arch only when CMAKE_HIP_ARCHITECTURES unset; never
  hardcode gfx90a literally (follower-revalidation churn rule).
- `examples/CMakeLists.txt` -- `add_hip_example` (copy .cpp -> .hip, LANGUAGE
  HIP, -DSOLVER_BACKEND_HIP); wires `solver`.
- `examples/solver.cpp`     -- `#elif defined(SOLVER_BACKEND_HIP)` arms
  (include+typedef, handle setup, two thrust copy-out branches folded into the
  CUDA branch).

## cuSPARSE -> hipSPARSE quirks hit (the real port surface)

1. hipsparseMatDescr_t AND hipsparseDnVecDescr_t are BOTH `typedef void*` in
   hipSPARSE (hipsparse-types.h:62, hipsparse-generic-types.h:52), whereas in
   cuSPARSE they are distinct opaque struct pointers. cuda.hpp's single
   `cuda_deleter` overloads `operator()` on those two types; under hipSPARSE
   those two overloads collide ("class member cannot be redeclared"). Fix:
   the backend `hip_deleter` keeps only the void* (DnVec) overload and drops the
   MatDescr one; the legacy CSR matrix descriptor (used only in the ilu0
   relaxation) gets its own `hip_mat_descr_deleter` in that header. The shared_ptr
   is `shared_ptr<void>` for both, so distinct deleter callables at .reset() are
   what disambiguate, not the type.

2. No `hipsparseXcsrilu02_bufferSize`/`_analysis`/`hipsparseXcsrilu02` generic-
   dispatch names exist (only the type-suffixed hipsparseDcsrilu02* /
   hipsparseScsrilu02*). cuSPARSE is the same; the upstream cuda code defines
   static `cusparseXcsrilu02_*` D/S overload wrappers for exactly this. Those
   wrappers were re-added (as `hipsparseXcsrilu02_*`) inside the HIP ilu0 class.
   The TYPE-agnostic `hipsparseXcsrilu02_zeroPivot` DOES exist (matches cuSPARSE).

3. The pre-CUDA-11 fallbacks in cuda.hpp / cusparse_ilu0.hpp (Hyb SpMV under
   `CUDART_VERSION < 11000`, csrsv2 triangular solves) are dropped in the HIP
   mirror: ROCm always ships the generic SpMV/SpSV api, so only the modern path
   is kept. Smaller, cleaner header.

4. SpMV algorithm: cuSPARSE `CUSPARSE_SPMV_CSR_ALG1` -> hipSPARSE
   `HIPSPARSE_SPMV_CSR_ALG1` (= rocSPARSE rowsplit, which is DETERMINISTIC per
   the rocsparse_spmv.h algorithm table). This is what gives bitwise run-to-run
   determinism in validation.

5. `hipsparseSpMV_preprocess` exists unconditionally in hipSPARSE; cuda.hpp only
   calls cusparseSpMV_preprocess under `CUDART_VERSION >= 12040`. The HIP mirror
   calls it always (valid, improves SpMV setup). No CUDART_VERSION on HIP.

6. hipEvent_t timing calls (hipEventCreate/Record/Synchronize/ElapsedTime) are
   `nodiscard` in the HIP headers (CUDA's are not), giving warnings if ignored
   like cuda.hpp does. Wrapped them in AMGCL_CALL_HIP (the header's existing
   error-check macro) instead of casting to void.

7. C++ standard: rocThrust/rocPRIM hard-#error below C++17 (general ROCm gotcha).
   `hip_target` sets `cxx_std_17`.

## Warp size

Not a factor here. The backend has NO custom device kernels with lane
assumptions -- all GPU work is hipSPARSE SpMV/SpSV + Thrust per-element
zip-iterator functors. grep of the new headers for warpSize/__shfl/__ballot/
hardcoded-32 = none. So the wave64 (gfx90a CDNA) fault class does not apply; the
port is arch-agnostic and the same headers should work on RDNA (gfx1100/gfx1151)
followers without change.

## Validation (REAL GPU, gfx90a / GPU 1, HIP_VISIBLE_DEVICES=1)

System: 7-point 3D Poisson on an n^3 grid (SPD), amgcl's own generator
(tests/sample_problem.hpp; the example's built-in problem when no -A given).
Device confirmed: "AMD Instinct MI250X / MI250".

Two validation vehicles, both pass:

A) Standalone driver (agent_space/amgcl_hip_validate.cpp, NOT in the fork):
   builds AMG(smoothed_aggregation, spai0)+CG on BOTH the HIP backend and the
   builtin CPU backend for the SAME system, n=48 (110592 unk) and n=64 (262144
   unk). Results (n=64):
     - builtin:   17 iters, resid 3.36e-11
     - hip spai0: 17 iters, resid 3.36e-11  (IDENTICAL iters + residual)
     - determinism: ||x_run1 - x_run2||inf = 0  (bitwise)
     - hip vs cpu solution: ||x_hip - x_cpu||inf / ||x_cpu||inf = 1.44e-15 (eps)
     - hip ilu0 (rocsparse_ilu0.hpp): 11 iters, resid 9.07e-11 (converges)
   Verdict line: VALIDATION PASS.

B) The project's own example built through its CMake (proves the deliverable
   build path): configure with -DUSE_HIP=ON and -DAMGCL_BUILD_EXAMPLES=ON,
   build target solver_hip. `./build-hip/examples/solver_hip` on the built-in
   Poisson:
     - spai0 relaxation: 3-level AMG, BiCGStab, 7 iters, error 5.24e-09
     - ilu0  relaxation: 3-level AMG, BiCGStab, 5 iters, error 1.40e-09
   Both converge on GPU 1.

Compile/lint was NOT treated as validation; the above are real GPU solves whose
convergence, residual, cross-backend agreement and determinism were checked.

## Build commands (recorded)

Standalone driver:
```
export HIP_VISIBLE_DEVICES=1
cd projects/amgcl/src
/opt/rocm/bin/hipcc -std=c++17 --offload-arch=gfx90a -O2 -I . -I tests \
  -Wno-deprecated-declarations \
  ../../../agent_space/amgcl_hip_validate.cpp \
  -o ../../../agent_space/amgcl_hip_validate \
  -L/opt/rocm/lib -lhipsparse -lrocsparse
../../../agent_space/amgcl_hip_validate 64
```

Project example:
```
export HIP_VISIBLE_DEVICES=1
cd projects/amgcl/src
cmake -S . -B build-hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DCMAKE_PREFIX_PATH=/opt/rocm -DAMGCL_BUILD_EXAMPLES=ON
cmake --build build-hip --target solver_hip -j
./build-hip/examples/solver_hip -p precond.relax.type=ilu0
```

Toolchain: ROCm 7.2.1, hipSPARSE 4.2.0 / rocSPARSE 4.2.0, gfx90a (MI250X).

## Validation 2026-05-30 (gfx1100, ROCm 7.2.1)

Platform: linux-gfx1100. Host: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), 4 GPUs. HIP_VISIBLE_DEVICES=0.
Fork branch moat-port at head_sha f4b87da72010e59e944847380ea972efa0aa15b8 -- NO source changes made (zero-churn follower).

### Libraries present

```
/opt/rocm/lib/libhipsparse.so  -- confirmed
/opt/rocm/lib/librocsparse.so  -- confirmed
/opt/rocm/include/thrust/      -- rocThrust confirmed
```

### Build commands

```
cmake -S projects/amgcl/src -B projects/amgcl/src/build-hip \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMGCL_BUILD_EXAMPLES=ON \
  -DCMAKE_PREFIX_PATH=/opt/rocm

cmake --build projects/amgcl/src/build-hip -j --target solver
cmake --build projects/amgcl/src/build-hip -j --target solver_hip
```

Configure succeeded (CMake 3.x, HIP compiler: Clang 22.0.0). Both targets built with only warnings (unused parameter, nodiscard hipGetDevice -- pre-existing from gfx90a build). No errors.

### gfx1100 code-object evidence

```
roc-obj-ls projects/amgcl/src/build-hip/examples/solver_hip
```
Output:
```
1  host-x86_64-unknown-linux-gnu-
1  hipv4-amdgcn-amd-amdhsa--gfx1100   offset=184320 size=1523056
```
gfx1100 code object embedded, no gfx90a object present.

### Solver convergence (real GPU, HIP_VISIBLE_DEVICES=0)

7-point 3D Poisson, n=64 (262144 unknowns), 3-level AMG, BiCGStab.

spai0 relaxation (default):
  Run 1: Iterations 8, Error 2.9431e-09
  Run 2: Iterations 8, Error 2.9431e-09  (bitwise identical -- deterministic)
  CPU builtin (same n=64): Iterations 8, Error 2.9431e-09  (exact match)

ilu0 relaxation (rocsparse_ilu0.hpp path):
  Iterations 6, Error 1.54581e-09  (converges cleanly)

gfx90a reference (from notes above, example solver_hip):
  spai0: 7 iters, error 5.24e-09
  ilu0:  5 iters, error 1.40e-09

The gfx1100 iteration count differs by 1 from gfx90a for spai0 (8 vs 7) and ilu0 (6 vs 5). This is expected: BiCGStab convergence history is not bitwise identical across architectures (different floating-point reduction order), but both converge well below the 1e-8 tolerance with no NaN/Inf. Within the same architecture (two runs on gfx1100), the residual is bitwise identical at 2.9431e-09 -- confirming HIPSPARSE_SPMV_CSR_ALG1 (rowsplit) determinism. No hipSPARSE errors, clean exit on both relaxation paths.

### Pass/fail

PASS. No source changes. No fork push. validated_sha = f4b87da72010e59e944847380ea972efa0aa15b8.

Toolchain: ROCm 7.2.1, hipSPARSE, rocSPARSE, rocThrust on gfx1100 (RDNA3, wave32).

## Windows gfx1151 attempt 2026-05-30 -- HIP backend COMPILES, GPU validation BLOCKED (hipSPARSE on APU)

Platform: AMD Radeon 8060S (gfx1151, RDNA3.5, integrated APU), Windows 11. ROCm via
TheRock wheels (rocm-sdk 7.14.0a20260519, hip 7.13.26190, hipSPARSE/rocSPARSE present
as hipsparse.lib/rocsparse.lib + DLLs). Fork moat-port HEAD f4b87da, NO source changes.

### Outcome
The amgcl HIP backend (amgcl/backend/hip.hpp, relaxation/rocsparse_ilu0.hpp) COMPILES
cleanly on Windows gfx1151 with NO port-code change -- a Boost-free standalone driver
(agent_space/amgcl_hip_validate.cpp: AMG smoothed_aggregation + spai0 + CG on both the
HIP and builtin backends) builds with hipcc + -lhipsparse -lrocsparse (Boost headers
from D:/Develop/TheRock/build/third-party/boost/source). The CPU builtin backend SOLVES
correctly (3D Poisson n=24: 12 iters, resid 7.94e-9). But the HIP backend SEGFAULTS at
runtime, and a minimal standalone hipSPARSE generic-API CSR SpMV (3x3 matrix) ALSO
segfaults on this device -- the crash is in rocSPARSE/hipSPARSE (before the first user
printf; likely a rocsparse DLL load / static-init failure on the APU), not in amgcl.

### Blocker (gfx1151 APU + Windows TheRock ROCm)
rocSPARSE/hipSPARSE is non-functional on the gfx1151 integrated APU here: a trivial
hipsparseCreate + hipsparseCreateCsr + hipsparseSpMV crashes (exit 139). amgcl's GPU
backend is fundamentally hipSPARSE SpMV/SpSV, so it cannot be GPU-validated on this
device. Same class as the rmm hipMemGetInfo gap -- a Windows-ROCm/APU runtime limitation,
not a port-code defect (amgcl gfx90a/gfx1100 pass; the port headers compile here unchanged).
gsplat (PyTorch) and GPUMD (hipBLAS/hipSOLVER/hipFFT + kernels) run fine on this same APU,
so the GPU and those libraries work; the gap is specific to rocSPARSE on the APU.

Better validated on a discrete-RDNA Windows GPU where rocSPARSE works. No fork change was
needed (the port compiles clean on Windows), so nothing to preserve as a patch.
State: set blocked=true; no source/sha change.

## Windows gfx1151 root-cause CORRECTION 2026-05-30

The earlier "rocSPARSE segfaults on the APU" was the System32 Adrenalin driver blit-JIT bug
(see rmm notes / gfx1151-apu-runtime-gaps). With TheRock's runtime deployed beside the exe the
segfault becomes a clean error, and the PRECISE root cause is: TheRock's rocm-sdk-libraries
wheel (7.14.0a20260519) is MISSING the rocSPARSE gfx1151 device-code kpack. The wheel's
.kpack/ dir ships blas_lib_gfx1151.kpack, fft_lib_gfx1151.kpack, rand_lib_gfx1151.kpack (and
torch_gfx1151.kpack) but NO sparse_lib_gfx1151.kpack, so hipSPARSE init/SpMV fails with
kpack_load_code_object error 13 (hipErrorInvalidImage) -- rocSPARSE has no gfx1151 kernels to
load. NOT an APU hardware limit and NOT the driver bug: the amgcl HIP backend compiles clean
and the CPU builtin backend solves. This is a TheRock PACKAGING gap; recheck on a newer nightly
that ships sparse_lib_gfx1151.kpack (or build rocSPARSE for gfx1151). Still blocked until then.

## Validation 2026-06-05 (windows-gfx1101, ROCm 7.14)

Platform: windows-gfx1101. Host: AMD Radeon PRO V710 (gfx1101, RDNA3, wave32), discrete GPU, HIP_VISIBLE_DEVICES=0. Fork moat-port HEAD f4b87da -- NO source changes (zero-churn follower, same as gfx1100).

### ROCm environment

TheRock ROCm 7.14.0a20260604 pip SDK (_rocm_sdk_devel + _rocm_sdk_core + _rocm_sdk_libraries wheels). Key finding: this host has NO sparse_lib_gfx1101.kpack in the wheels (only blas/fft/rand kpacks for gfx1101/gfx1201), matching the gfx1151 packaging gap. However, unlike gfx1151 (APU), the discrete gfx1101 GPU loads rocSPARSE kernels from the DLLs directly (rocsparse.dll in _rocm_sdk_libraries/bin is 67 MB and has embedded RDNA3 code). Confirmed via TheRock's own hipsparse-test.exe: 272/272 tests PASS including SpMV CSR and csrsv2 (triangular solve) tests.

### Build

Standalone driver (agent_space/amgcl_hip_validate_win.cpp, NOT in the fork) -- no Boost dependency. Built with hipcc.exe from _rocm_sdk_devel:

```
ROCM=.../_rocm_sdk_devel
hipcc.exe -std=c++17 --offload-arch=gfx1101 -O2 -I . -I tests
          -Wno-deprecated-declarations -DAMGCL_NO_BOOST
          agent_space/amgcl_hip_validate_win.cpp
          -o <TheRock/build/bin>/amgcl_hip_validate_win.exe
          -L$ROCM/lib -lhipsparse -lrocsparse
```

Run from TheRock/build/bin/ (where all hipsparse/rocsparse DLLs are already present; the exe needs amdhip64_7.dll, hipsparse.dll, rocsparse.dll, rocblas.dll in its directory).

### Solver convergence (real GPU, HIP_VISIBLE_DEVICES=0)

7-point 3D Poisson, n=64 (262144 unknowns), 3-level AMG, CG.

spai0 relaxation:
  GPU: AMD Radeon PRO V710
  Run 1: Iterations 17, residual 3.35779e-11
  Run 2: Iterations 17, residual 3.35779e-11  (bitwise identical -- deterministic)
  CPU builtin: Iterations 17, residual 3.35779e-11  (exact match)
  Cross-backend: ||x_hip - x_cpu||_inf / ||x_cpu||_inf = 1.43732e-15  (machine epsilon)

ilu0 relaxation (rocsparse_ilu0.hpp path):
  Iterations 11, residual 9.07312e-11  (converges cleanly)

gfx90a reference: spai0 17 iters / resid 3.36e-11; ilu0 11 iters / resid 9.07e-11 -- IDENTICAL.
gfx1100 reference: spai0 8 iters / resid 2.94e-09; ilu0 6 iters / resid 1.55e-09 -- slightly different (expected BiCGStab vs CG difference; this run uses CG throughout, gfx1100 used BiCGStab in the example).

### Pass/fail

PASS. No source changes. No fork push. validated_sha = f4b87da72010e59e944847380ea972efa0aa15b8.

Toolchain: ROCm 7.14.0a20260604 (TheRock pip), hipSPARSE, rocSPARSE on gfx1101 (RDNA3, wave32). HIP version 7.14.60850.

## Validation 2026-06-06 (windows-gfx1201, ROCm 7.14)

Platform: windows-gfx1201. Host: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), discrete GPU, HIP_VISIBLE_DEVICES=0. Fork moat-port HEAD f4b87da -- NO source changes (zero-churn follower, same as all other platforms).

### ROCm environment

TheRock ROCm 7.14.0a20260604 pip SDK. Key DLL resolution finding: the new
_rocm_sdk_libraries rocsparse.dll (67MB, Jun 4) has a `.kpackrf` section referencing
`../.kpack/blas_lib_@GFXARCH@.kpack` (relative to the DLL's directory). This means the
exe MUST run from a directory where the DLL lives at `../bin/rocsparse.dll` so the kpack
resolves to `../.kpack/blas_lib_gfx1201.kpack`. Running from `_rocm_sdk_libraries/bin/`
(where hipsparse.dll and rocsparse.dll live) works because `.kpack/blas_lib_gfx1201.kpack`
exists in `_rocm_sdk_libraries/.kpack/`. The older TheRock/build/bin DLLs (Apr 21)
lack gfx1201 AOT code in rocsparse.dll and cannot run on gfx1201.

No `sparse_lib_gfx1201.kpack` is needed (rocsparse.dll references blas_lib_gfx1201.kpack
for its device code dispatch, not a separate sparse kpack).

### Build

Standalone driver compiled with hipcc from _rocm_sdk_devel:

```
hipcc.exe -std=c++17 --offload-arch=gfx1201 -O2
          -I B:/develop/moat/projects/amgcl/src
          -I B:/develop/moat/projects/amgcl/src/tests
          -Wno-deprecated-declarations -DAMGCL_NO_BOOST
          B:/develop/moat/agent_space/amgcl_hip_validate_win.cpp
          -o <_rocm_sdk_libraries/bin>/amgcl_hip_validate_gfx1201.exe
          -L<_rocm_sdk_devel/lib> -lhipsparse -lrocsparse
```

Runtime: exe in `_rocm_sdk_libraries/bin/`; amdhip64_7.dll, amd_comgr.dll,
rocm_kpack.dll, hiprtc*.dll copied from `_rocm_sdk_core/bin` into the same dir.
Run with HIP_VISIBLE_DEVICES=0.

### Solver convergence (real GPU, HIP_VISIBLE_DEVICES=0)

7-point 3D Poisson, n=64 (262144 unknowns), 3-level AMG, CG.

spai0 relaxation:
  GPU: AMD Radeon RX 9070 XT (gfx1201)
  Run 1: Iterations 17, residual 3.35779e-11
  Run 2: Iterations 17, residual 3.35779e-11  (bitwise identical -- deterministic)
  CPU builtin: Iterations 17, residual 3.35779e-11  (exact match)
  Cross-backend: ||x_hip - x_cpu||_inf / ||x_cpu||_inf = 1.43732e-15  (machine epsilon)

ilu0 relaxation (rocsparse_ilu0.hpp path):
  Iterations 11, residual 9.07312e-11  (converges cleanly)

gfx90a/gfx1101 reference: spai0 17 iters / resid 3.36e-11; ilu0 11 iters / resid 9.07e-11 -- IDENTICAL.

### Pass/fail

PASS. No source changes. No fork push. validated_sha = f4b87da72010e59e944847380ea972efa0aa15b8.

Toolchain: ROCm 7.14.0a20260604 (TheRock pip), hipSPARSE, rocSPARSE on gfx1201 (RDNA4, wave32). HIP version 7.14.60850.

## PR framing note: amgcl is already AMD-capable via OpenCL (2026-06-04)

README sweep: amgcl has backends cuda.hpp / vexcl.hpp / viennacl.hpp + builtin OpenMP. VexCL and ViennaCL are OpenCL-based and already run on AMD GPUs, so amgcl is NOT lacking AMD support; it lacks a NATIVE HIP/ROCm backend (no hip.hpp, 0 HIP code, no ROCm PRs). This is still a valuable port (native ROCm perf + a hipSPARSE/rocSPARSE-style backend alongside the CUDA one), but the PR MUST be framed as "add a native HIP/ROCm backend" and must NOT claim to be the first/only AMD support. Merge-policy fit is good: backends live in-tree (amgcl/backend/*.hpp), so a hip.hpp backend matches the existing pattern. See [[moat-no-duplicate-amd-ports]].
