# Port plan: espresso (ESPResSo molecular dynamics)

## Project
- Name: espresso
- Upstream: https://github.com/espressomd/espresso
- Default branch: main (HEAD at planning: eafab258df3b, 5.0.1 release notes)
- Lead platform: linux-gfx90a

## DISPOSITION
PROCEED -- but as a SCOPED core-GPU port (electrostatics/magnetostatics), waLBerla GPU EXCLUDED.
Effort class: HEAVY-BUT-TRACTABLE on the C++ build axis, MODERATE on the actual GPU-port axis.
Concurrency flag: this build is HEAVY and finicky (mandatory MPI + Boost + FFTW + Python +
a configure-time FetchContent of several external projects). Do NOT run it as a fourth
concurrent heavy porter without watching disk/egress. The GPU code surface to port is small;
the cost is the build environment, not the kernels.

## Existing AMD support
Finding: ESPResSo ONCE HAD experimental HIP/ROCm support and it was REMOVED.
- Old docs/changelog: a `WITH_CUDA_COMPILER` option accepted `nvcc | clang | hip`; the HIP-Clang
  path was experimental, documented as "do not base hardware purchases on it," and has since
  been dropped. Current main (5.0.1) CMake has ZERO hip/rocm references; the CUDA-compiler
  options are NVIDIA (nvcc), Clang (clang-as-CUDA), and NVHPC only
  (CMakeLists.txt:310-338). No `WITH_CUDA_COMPILER=hip` remains.
- Web/orgs: no ROCm/AMD/GPUOpen-org fork, no separately-named AMD project (no ROCm-DS analogue),
  no live rocm/hip branch or open HIP PR upstream. The espressomd/docker issue #156 "Support for
  ROCm 3.1.0" tracked a CI image, not a code port, and predates the removal.
- Judgment: NO authoritative current AMD port exists. The removed HIP path is historical and
  long stale (pre-C++20, pre-CUDA-12, pre-waLBerla-bridge architecture); it is NOT a base to
  resurrect. Decision: FROM-SCRATCH HIP port our way (Strategy A), targeting only the core GPU.
  The old removed code may be consulted as a non-authoritative hint for which symbols mattered,
  never inherited.
- The current Clang-as-CUDA-compiler support is a helpful tailwind: clang-CUDA and clang-HIP
  share the same front end, so the host/device split and warnings are already clang-clean.

## Build classification
cmake (pure CMake C++ core + Python interface). Evidence:
- Top-level CMakeLists.txt, `enable_language(CUDA)` at CMakeLists.txt:274; CUDA arch via
  `CMAKE_CUDA_ARCHITECTURES` (CMakeLists.txt:281-296); CUDA libs linked in
  src/core/CMakeLists.txt:62 (`CUDA::cuda_driver CUDA::cudart CUDA::cufft`).
- NOT a pytorch extension (no find_package(Torch), no CUDAExtension). Strategy A applies.
- ext_type = cmake.

### GPU surface is split into TWO subsystems -- this is the crux
1. ESPResSo CORE GPU (src/core, gated by ESPRESSO_BUILD_WITH_CUDA): exactly 5 compiled .cu TUs
   (src/core/CMakeLists.txt:55-61): cuda/common_cuda.cu, cuda/init_cuda.cu,
   cuda/CudaHostAllocator.cu, magnetostatics/dipolar_direct_sum_gpu_cuda.cu,
   electrostatics/p3m_gpu_cuda.cu, electrostatics/p3m_gpu_error_cuda.cu,
   system/GpuParticleData_cuda.cu. Small, self-contained, our code. cuFFT + Thrust + atomicAdd.
2. waLBerla bridge GPU (src/walberla_bridge, gated by ESPRESSO_BUILD_WITH_WALBERLA=ON default):
   ~70 code-generated .cu kernels (lattice-Boltzmann + electrokinetics). CRITICALLY, when
   WITH_WALBERLA=ON and CUDA=ON the build FETCHES the entire waLBerla framework from
   i10git.cs.fau.de (CMakeLists.txt:997-1028, GIT_TAG 3247aa73) and builds it with
   WALBERLA_BUILD_WITH_CUDA=on, pulling in `walberla::gpu` -- waLBerla's own CUDA backend.
   Porting that is porting a whole separate upstream (external, non-GitHub GitLab; egress-heavy).

### Minimal GPU build recipe (the scoped target)
Disable waLBerla to collapse the surface to the 5 core .cu files and avoid the waLBerla fetch:

    cmake -S . -B build \
      -G Ninja \
      -DESPRESSO_BUILD_WITH_CUDA=ON \
      -DESPRESSO_BUILD_WITH_WALBERLA=OFF \
      -DESPRESSO_BUILD_WITH_HDF5=OFF \
      -DESPRESSO_BUILD_WITH_SCAFACOS=OFF \
      -DESPRESSO_BUILD_WITH_STOKESIAN_DYNAMICS=OFF \
      -DESPRESSO_BUILD_WITH_GSL=OFF \
      -DESPRESSO_BUILD_WITH_NLOPT=OFF \
      -DESPRESSO_BUILD_WITH_CALIPER=OFF \
      -DESPRESSO_BUILD_WITH_FFTW=ON \
      -DESPRESSO_BUILD_WITH_PYTHON=ON \
      -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
    cmake --build build -j16

Mandatory deps (apt/conda, install without asking): a C++20 compiler, CMake>=3.25, Ninja,
OpenMPI or MPICH (find_package(MPI 3.0 REQUIRED) at CMakeLists.txt:849 is UNCONDITIONAL -- MPI
cannot be turned off), Boost (mpi, serialization, system, filesystem), FFTW3 (CPU P3M),
Python3 + numpy + cython + scipy + h5py-less, and the ROCm stack (hipfft, rocfft, rocthrust/
hipcub). Even the GPU-only configuration links a CPU FFTW for the non-GPU P3M path.

Note: ESPResSo's GPU library selection is single-precision by default
(ESPRESSO_P3M_GPU_FLOAT defined, p3m_gpu_cuda.cu:32) so cuFFT->hipFFT needs the R2C/C2R
(single) entry points first; double (D2Z/Z2D) is compiled-out.

## Port strategy: A (compat header)
Rationale: pure CMake, our own small .cu set, no Torch. Add `enable_language(HIP)` gated on a
`USE_HIP` option, mark the 5 core .cu sources `LANGUAGE HIP`, and add ONE compat header that
aliases the CUDA spellings the core uses to HIP and swaps cuFFT->hipFFT. Keep host C++ untouched.
On NVIDIA the header is a no-op and the existing nvcc/clang-CUDA paths are unchanged.

Port-vs-rewrite: MECHANICAL port is correct here. The core kernels are charge-assignment /
B-spline interpolation / direct dipole sums + an FFT -- no CUTLASS/wgmma/warp-specialized GEMM,
nothing that wants an AMD-native rewrite. cuFFT maps to hipFFT 1:1.

## CUDA surface inventory (CORE only; waLBerla excluded)
- Kernels: charge assignment, B-spline mesh interpolation, force back-interpolation, influence
  function, P3M error estimate; dipolar direct-sum forces/torques/energy. Plain __global__.
- cuFFT (electrostatics/p3m_gpu_cuda.cu): cufftHandle, cufftPlan3d, cufftExecR2C/C2R,
  cufftComplex, cufftDestroy, CUFFT_SUCCESS. Cleanly abstracted behind macros
  (FFT_TYPE_COMPLEX/FFT_FORW_FFT/... at p3m_gpu_cuda.cu:37-50). -> hipFFT: hipfftHandle,
  hipfftPlan3d, hipfftExecR2C/C2R, hipfftComplex, hipfftDestroy, HIPFFT_SUCCESS. Link hipfft.
- Thrust: device_vector / reduce / copy / raw_pointer_cast in p3m_gpu_error_cuda.cu and
  GpuParticleData_cuda.cu. -> rocThrust (thrust:: names unchanged under ROCm; just include path
  and link). No CUB, no DeviceReduce template-matrix bloat -> no --offload-compress needed.
- atomicAdd on float (p3m_gpu_cuda.cu:368/375/489-491) and into mesh/force arrays. Native on
  gfx90a (float atomics fine). No 64-bit/double atomics in the core force path.
- Streams: a single global cudaStream_t stream[1] (cuda/common_cuda.cu:34). -> hipStream_t.
- Pinned host memory: CudaHostAllocator.cu (cudaHostAlloc/cudaMallocHost-style). -> hipHostMalloc.
- Launch/error helpers: KERNELCALL macro `<<<>>>` + cudaGetLastError (cuda/utils.cuh:75-80),
  cuda_safe_mem (common_cuda.cu). `<<<>>>` is accepted by hipcc/clang-HIP; cudaGetLastError ->
  hipGetLastError via the compat header.
- cuRAND: NONE in the core. The thermostat/LB noise is a self-contained Philox header
  (walberla_bridge .../philox_rand.h + libs/Random123-1.09), counter-based, no cuRAND link.
  Since waLBerla is excluded from the port scope, no RNG library swap is needed at all.
- Textures / surfaces / cudaArray / layered arrays: NONE in the core. (The popsift texture/
  layered-array fault classes do not apply.)
- Warp intrinsics (__shfl/__ballot/__any/warpSize/__activemask) and hardcoded 32: NONE found in
  src/core. So the wave64-vs-wave32 fault class has NO direct trigger in the ported code.
- CUTLASS/CuTe/wgmma: none.

## Risk list
- waLBerla scope creep (PRIMARY RISK). If WITH_WALBERLA is left ON, the build fetches and
  HIP-compiles the entire waLBerla GPU backend -- out of scope, egress-heavy, a separate port.
  MITIGATION: build with -DESPRESSO_BUILD_WITH_WALBERLA=OFF; scope the PR claim to core GPU
  (P3M / dipolar / GpuParticleData). LB/EK GPU is explicitly future work.
- Build weight / environment. Mandatory MPI + Boost + FFTW + a C++20 toolchain + Python/Cython;
  even with options trimmed this is a multi-hundred-TU C++ build. Finicky: CUDA std is pinned to
  C++20 and CUDA version >=12 for nvcc; for the HIP path we sidestep the nvcc version gate
  (CMakeLists.txt:276-338 keys off CMAKE_CUDA_COMPILER_ID, which the HIP language path bypasses).
- hipFFT R2C/C2R scaling/normalization parity. cuFFT and rocFFT/hipFFT both leave transforms
  UNNORMALIZED, but verify the inverse-transform 1/N handling in p3m_gpu_cuda.cu matches; a
  silent factor-of-N would surface as wrong forces, caught by the CPU-vs-GPU gate below.
- Float atomic ordering / determinism. atomicAdd into the mesh is order-nondeterministic on both
  vendors; use tolerance compares (the tests already do), not bit-exact.
- ffp-contract drift (clang-HIP defaults to -ffp-contract=fast). The B-spline weight chains are
  multi-statement; pin -ffp-contract=on in CMAKE_HIP_FLAGS to match CUDA/CPU FMA formation and
  keep the CPU-vs-GPU tolerance gate green. (CV-CUDA fault class.)
- Rule-of-five on the cufftHandle/stream wrappers. cuFFT plans live in a struct destroyed in a
  dtor (p3m_gpu_cuda.cu:113-138); ensure default-init and no double-destroy (AMD faults where
  CUDA tolerates). Audit the P3M data struct lifetime.
- Single global stream[1] (common_cuda.cu:34): a translation-unit-static handle; confirm it is
  created before first use and not destroyed twice across re-init.
- Wave size: no direct trigger in ported code, but BUILD the fat binary
  -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" and confirm both code objects emit, so the follower
  gfx1100/gfx1151 validations need no source change.

## File-by-file change list (proposed; porter confirms)
- NEW src/core/cuda/cuda_to_hip.h (or reuse a single compat header): under USE_HIP, include
  <hip/hip_runtime.h> + <hipfft/hipfft.h>, alias the cuda*/cufft* symbols the 5 core TUs use
  (cudaMalloc/Free/Memcpy/MemcpyAsync, cudaStream_t, cudaError_t, cudaSuccess, cudaGetLastError,
  cudaHostAlloc/cudaFreeHost, cufftHandle/Plan3d/ExecR2C/ExecC2R/Complex/Destroy/SUCCESS, the
  CUFFT_* plan flags). Include <cstring>/<cstdlib> BEFORE hip_runtime (gpuRIR lesson).
- CMakeLists.txt: add option(USE_HIP ...); when USE_HIP, enable_language(HIP), default
  CMAKE_HIP_ARCHITECTURES to gfx90a only-if-unset (never hardcode the literal), pin
  -ffp-contract=on in the HIP flags.
- src/core/CMakeLists.txt: when USE_HIP, set_source_files_properties(the 5 .cu PROPERTIES
  LANGUAGE HIP); link hip::hipfft (+ rocthrust) instead of CUDA::cufft/cudart; set
  HIP_ARCHITECTURES on the target from CMAKE_HIP_ARCHITECTURES.
- Touch the 5 .cu files only as needed (include the compat header; guard any genuinely divergent
  line with #if defined(USE_HIP)). Expect this to be small.
- Leave src/walberla_bridge untouched (built CPU-only or omitted via WITH_WALBERLA=OFF).

## Build commands (gfx90a)
See "Minimal GPU build recipe" above. Multi-arch fat-binary check:
    llvm-objdump --offloading build/.../libespresso_core.so | grep -E "gfx90a|gfx1100"
(expect both). Single-arch run on gfx90a for the actual GPU tests.

## Test plan (real GPU, egress-feasible -- testsuite generates its own systems, no downloads)
Build with -DESPRESSO_BUILD_TESTS=ON. GPU tests carry the ctest `gpu` label
(testsuite/python/CMakeLists.txt:113-116). Primary CPU-vs-GPU consistency gates (core scope):
- testsuite/python/dawaanr-and-dds-gpu.py -- DipolarDirectSum GPU vs CPU: builds a random dipole
  system, compares per-particle forces AND torques GPU-vs-CPU within tolerance. IDEAL gate.
  (@skipIfMissingGPU; needs CUDA, DIPOLES, LENNARD_JONES features.)
- testsuite/python/p3m_madelung.py -- P3M GPU (gpu=True, single_precision) vs the analytic
  Madelung constant of an ionic crystal (built-in reference, no data file). Lines 230-271.
- testsuite/python/coulomb_cloud_wall.py -- P3M forces vs a shipped reference (cloud-wall), CPU
  and GPU variants; classic correctness gate.
- testsuite/python/p3m_electrostatic_pressure.py, p3m_fft.py -- additional P3M GPU coverage.
- C++ unit test: src/core/unit_tests/cuda_test.cu (CudaHostAllocator) -- a ctest under the core
  unit-test target; exercises the pinned-memory path on GPU.
Run:
    ctest --test-dir build -L gpu --output-on-failure
    # core-scope subset:
    ctest --test-dir build -R "dds_gpu|p3m_madelung|coulomb_cloud_wall|cuda_test" --output-on-failure
Non-GPU regression set (must not regress): the full CPU testsuite minus the `gpu` label, esp.
the CPU P3M/coulomb tests (coulomb_cloud_wall CPU variant, p3m_madelung CPU path) and the C++
unit tests. Run `ctest --test-dir build -LE gpu` for the non-GPU baseline.
Note: LB/EK GPU tests (lb*.py, the many lattice-Boltzmann tests) require waLBerla GPU, which is
OUT OF SCOPE; they are skipped/not-built in the WITH_WALBERLA=OFF configuration. Do not count
them as regressions.

## Open questions
1. Does building with WITH_WALBERLA=OFF still produce a usable espressomd Python module for the
   P3M/dipolar GPU tests? (Expected yes -- waLBerla only provides LB/EK; electrostatics is core.)
   Porter to confirm the trimmed config imports and the dds/p3m tests are collected.
2. hipFFT vs rocFFT: link hipFFT (CUDA-compatible API, minimal diff) or call rocFFT directly?
   Plan assumes hipFFT for a 1:1 macro swap.
3. Is the C++20 + CUDA-language interaction happy under the HIP language path (we bypass the
   nvcc>=12 gate, but confirm the core compiles as -x hip at C++20).
4. Future scope (NOT this port): waLBerla GPU LB/EK. That is a much larger, separate effort
   (port or upstream-coordinate waLBerla's own GPU backend); flag for a follow-up project.

## Scope extension 2026-11-20: waLBerla LB/EK GPU (`src/walberla_bridge`)

Requested by jeffdaily on 2026-11-20 after he ruled the deferral `espresso-walberla-lb-ek-gpu` "now" (rulings in notes.md). The upstream PR is held and sent complete: the validated core commit bc8d73dd9 first, then the waLBerla bridge work as separate commits on top. The fork's `moat-port` stays private until then: new commits on top, plain pushes, bc8d73dd9 never rewritten.

How this plan was recorded: the stage was `review-passed`, and the state machine has no way back to `planning` from there (`STAGE_TRANSITIONS["review-passed"] = {"porting", "reviewing"}`; planning is only reachable from `screened`). So this section is the plan update, written without a planning stage, and the project went `review-passed -> porting` (linux-gfx942 holds the lock). No state was hand-edited.

The starting point was the waLBerla intake's analysis (`origin/port/walberla:projects/walberla/notes.md`, Q1, Probe A, Q3a/3b, Q5). Each fact used here was re-checked in the espresso fork at bc8d73dd9 and against waLBerla at the pin.

### Decisions

1. **waLBerla version and its HIP build.** ESPResSo fetches `https://github.com/lssfau/walberla.git` at `623a60c1` ("v7.2 with patches", `git describe` = v7.2-72-g623a60c1). That commit already has `WALBERLA_BUILD_WITH_HIP` and calls a plain `find_package(hip REQUIRED)` (CMakeLists.txt:634-643). The HIP minimum-version check that rejects HIP 7.x (`25bcdd25` "Require minimum versions for MPI, OpenMP, CUDA and HIP", 2026-11-02) is **not** an ancestor of the pin (`git merge-base --is-ancestor 25bcdd25 623a60c1` is false). Built inside ESPResSo with ROCm 10.1 (HIP 7.16), the pin configures and compiles **as is**. So waLBerla needs no fix, patch step, version bump or fork for this work. Caveat for a future pin bump: any waLBerla after 2026-11-02 needs that version check fixed upstream first (the intake's draft report #1). waLBerla's `src/gpu/ShiftedPeriodicity.cu` is silently left out of a HIP build (open waLBerla issue #314); ESPResSo does not use it.
2. **Generated kernels: edit the templates and the generated files consistently by hand, verified against the generator; no regeneration.** A full regeneration with ESPResSo's scripts and the exact pinned versions (pystencils `e851f4e3` from the contributor fork on gitos, lbmpy `e9efe343`, sympy 1.12.1, waLBerla python at `007e77e0`, the commit the headers name) is **not reproducible**. Starting from unmodified templates, it rewrites 148 files: sympy term order, renamed temporaries and the recorded waLBerla commit (the checked-in files name both `3247aa73` and `007e77e0`). Upstream's Readme says the same thing: "The code generation is not deterministic ... If you only adapted a few lines in a specific template file, then you only need to commit the corresponding output files." Committing a regeneration would bury a ~30-line fix in thousands of noise lines. Instead, a script applies identical edits to the 4 templates, the 28 affected generated files and the vendored `philox_rand.h` (2 copies). Then two regenerations are compared, one with unmodified and one with fixed templates. Once pure reorderings are cancelled, the generator's net delta equals the hand edit's net delta in every one of the 241 generated files. The residual is run-to-run noise, never an edited construct.
3. **Guards in the hand-written bridge**: `defined(__CUDACC__)` becomes `(defined(__CUDACC__) or defined(__HIPCC__))`, and `WALBERLA_BUILD_WITH_CUDA` becomes `WALBERLA_BUILD_WITH_GPU_SUPPORT`, which waLBerla >= 7.2 (the `find_package` minimum and the pin) defines for either backend. This is the same pattern as the core commit's `|| defined(__HIPCC__)`. No `__CUDACC__` is defined for HIP, because rocThrust and HIP headers key on it.
4. **Kokkos does not matter.** The Kokkos CUDA backend is enabled only for clang-CUDA without waLBerla, the bridge links Kokkos only under `Kokkos_ENABLE_CUDA`, and all Kokkos code runs in host space. No Kokkos change is needed.
5. **EK FFT**: the bridge's cuFFT use is data types only (`cufftReal/Complex` and the double variants). Under HIP these are aliased to the hipFFT types in `greens_function.hpp`. The FFT itself is heFFTe's: `heffte::backend::cufft` becomes `heffte::backend::rocfft`, and heFFTe v2.4.1 (FetchContent) is built with `Heffte_ENABLE_ROCM` (rocFFT + `hip::host`).
6. **CUDA path**: every CMake change is behind `ESPRESSO_BUILD_WITH_HIP` or keyed on `WALBERLA_BUILD_WITH_HIP`. Preprocessor changes only add macros that nvcc and clang-CUDA never define. The few source edits visible to CUDA (`uint3` copies via `block_index()/thread_index()`, the `dim3` return, the `gpu*` wrapper in two tests) name the same values and functions. The nvcc compile (upstream's `cuda12-maxset` recipe on the mi355 host) is owed before jeff gets the PR.

### Rounds (porter)

- (a) decoupling: HIP core with waLBerla's CPU solvers. Configure stop removed, `WALBERLA_BUILD_WITH_CUDA` only for a CUDA build, `OpenMP::OpenMP_CUDA` via `ESPRESSO_OPENMP_CUDA`, and the four `.cu` init files get a language in a HIP build. Gate: the whole C++ unit-test suite.
- (b) `WALBERLA_BUILD_WITH_HIP` from ESPResSo's CMake. The 56 generated kernels and the bridge compile as HIP (the three intake fixes plus a fourth template, `ReactionKernelSelector.tmpl.h`), the guards change, and the two bridge GPU unit tests build as HIP. Gate: unit tests and `ctest -L gpu`.
- (c) EK FFT Poisson solver: hipFFT types, heFFTe rocFFT backend.
- (d) Python suite: GPU classes run, not skip, and never hit "compiled without CUDA support". Gate: LB and EK Python files and `ctest -L gpu`. A statistical-test miss is investigated, never covered by a looser tolerance.

### Owed after porting

Review of the three new commits; nvcc compile; validation on linux-gfx950 (wave64) and a wave32 Radeon (linux-gfx1201); Windows waived for this project.
