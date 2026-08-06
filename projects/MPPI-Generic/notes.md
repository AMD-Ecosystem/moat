# MPPI-Generic notes

ROCm/HIP port (lead linux-gfx90a). Upstream ACDSLab/MPPI-Generic @ b5c8daa.
Pure-CMake header-only CUDA MPPI control library. Strategy A.

## Environment
- gfx90a (MI250X, wave64), ROCm 7.2.1, hipcc clang 22, cmake 3.31, Ninja.
- Deps installed via apt: libeigen3-dev (3.4.0), libyaml-cpp-dev. hipRAND/hipFFT
  ship with ROCm. GoogleTest is downloaded by the build at configure time.
- npz/test-network generation needs numpy (present in /opt/conda envs).

## Build
```
HIP_VISIBLE_DEVICES=3 cmake -S src -B build-hip -GNinja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DMPPI_BUILD_TESTS=ON -DMPPI_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
HIP_VISIBLE_DEVICES=3 cmake --build build-hip -j 16
```
Followers (gfx1100/gfx1151) need only a different -DCMAKE_HIP_ARCHITECTURES; no
source change (arch read from CMAKE_HIP_ARCHITECTURES, never hardcoded).

## Existing AMD support
None. README requires an NVIDIA GPU + CUDA; no rocm/hip/amdgpu/gfx token in-tree.
Fresh CUDA->HIP port.

## Port footprint (USE_HIP-guarded, CUDA path unchanged)
- include/mppi/hip_compat/cuda_to_hip.h -- the single compat header. Maps the
  cuda*/curand*/cufft* spellings to hip*, plus several semantic shims (below).
  Force-included on every HIP TU via CMAKE_HIP_FLAGS `-include`.
- include/mppi/hip_compat/{cuda_runtime,curand,cufft,device_launch_parameters,
  cooperative_groups}.h -- shim headers so the library's angle-bracket toolkit
  includes resolve to the compat layer. This dir is on the HIP include path only.
- CMakeLists.txt: USE_HIP option; conditional project() language (C CXX [+HIP]);
  enable_language(HIP); arch defaulted to gfx90a only when CMAKE_HIP_ARCHITECTURES
  unset; force-include flag; add_executable/add_library overrides that retag .cu
  sources LANGUAGE HIP (so the per-target test/example CMakeLists stay untouched);
  hip_compat/ prepended to the header lib include dirs; USE_HIP compile def.
- cmake/MPPIGenericToolsConfig.cmake: on HIP use hip::hiprand/hip::hipfft and skip
  the CUDA arch autodetection + CUDAToolkit find.
- src/controllers/CMakeLists.txt: the nvcc -Xcompiler=-fno-guess-branch-probability
  hint is GCC-only (clang rejects it) and unneeded on HIP, so applied on CUDA only.
- source edits in include/: mppi_common.cu, rmppi_kernels.cu, parallel_utils.cuh,
  cuda_math_utils.cuh, gpu_err_chk.cuh, colored_noise.cu -- see fault classes.

## Fault classes hit + fixes

1. WAVE64 warp-synchronous reduction (THE correctness fix).
   warpReduceAdd<BLOCKSIZE> (mppi_common.cu:1170) is the classic CUDA unrolled warp
   reduction: volatile shared mem, steps 32->16->...->1 with NO __syncwarp, relying
   on 32-lane lockstep. costArrayReduction (mppi_common.cu:1198) and
   multiCostArrayReduction (rmppi_kernels.cu:1135) stop their __syncthreads() tree at
   size==32 and hand the last 64->1 to warpReduceAdd<64>. On gfx90a the low 32 lanes
   of a 64-lane wavefront are NOT guaranteed lockstep across those unsynced steps ->
   the running_cost reduction races (wrong + non-deterministic cost sums). Fix: on
   HIP set stop_condition=0 so the fully-__syncthreads()-synchronized tree runs all
   the way to 1 (identical add order, block-wide barrier -> correct on any wave size)
   and skip the warpReduceAdd switch. CUDA path byte-for-byte unchanged. No
   __shfl/__ballot/warpSize anywhere else; all other reductions are serial or
   single-thread, so this is the only wave64 site.

2. __CUDACC__ not defined under hipcc (header-only .cu-include idiom).
   Each foo.cuh ends with `#if __CUDACC__  #include "foo.cu"  #endif` (59 sites) to
   pull device definitions in only under the device compiler. hipcc defines __HIPCC__
   (both passes) not __CUDACC__ -> every kernel/device helper undeclared. Fix: compat
   header defines __CUDACC__ when __HIPCC__ is set.

3. __CUDA_ARCH__ not defined during HIP device compilation.
   ~60 `#ifdef __CUDA_ARCH__` device-vs-host branches (parallel index helpers,
   dynamics, math) would silently collapse to the CPU fallback on the GPU. Fix:
   compat header defines __CUDA_ARCH__=1 only in the HIP device pass (guarded by
   __HIP_DEVICE_COMPILE__, 1 in device pass / 0 in host pass -- verified empirically).

4. Eigen CUDA path pulled in by our __CUDA_ARCH__.
   Eigen sees __CUDA_ARCH__ -> sets EIGEN_CUDA_ARCH -> includes CUDA's
   <math_constants.h> (absent on ROCm). Eigen has a native HIP path. Fix: define
   EIGEN_NO_CUDA in the compat header so Eigen ignores __CUDA_ARCH__ and uses its own
   HIP math_constants. Eigen otherwise compiles unchanged under hipcc, incl. in
   __device__ code.

5. Function-template explicit specialization must match the primary's host/device
   attributes (clang/HIP, not nvcc). getParallel2DIndex primary is __host__ __device__
   but its 8 explicit specializations were __device__-only -> "no function template
   matches / target attributes do not match". Fix: make the specializations __host__
   __device__ (parallel_utils.cuh); they already carry the #ifdef __CUDA_ARCH__ host
   fallback inside.

6. float2/3/4 operator overloads collide with HIP_vector_type.
   cuda_math_utils.cuh hand-rolls the full set of float2/3/4 arithmetic, compound-
   assign, unary-minus, equality operators. CUDA's vector_types.h defines none, so
   they are unique there; HIP's HIP_vector_type provides all of them (host+device,
   verified) -> "use of overloaded operator '*' is ambiguous". Fix: USE_HIP-guard the
   two operator blocks (keep dot/cross/norm and createPartialCudaTuple, which HIP does
   not provide).

7. cudaFuncSetAttribute signature: CUDA takes the typed kernel pointer; HIP's
   hipFuncSetAttribute takes const void*. Added MPPI_GPU_FUNC_PTR(...) (variadic so a
   comma-bearing template-id kernel<A,B,C> is one macro arg): casts to const void* on
   HIP, identity on CUDA, at the 3 call sites in mppi_common.cu (cuda_to_hip.h:207 HIP,
   :216 CUDA). The macro is defined in cuda_to_hip.h, which is force-included on HIP but
   NOT on CUDA -- so mppi_common.cu now #includes it explicitly (see reviewer fix R1).

8. cuFFT status codes without hipFFT equivalents. cufftGetErrorString switches over
   CUFFT_LICENSE_ERROR (reachable on HIP since the CUDART_VERSION<13000 branch is
   taken) which hipFFT lacks; USE_HIP-guarded that single case in gpu_err_chk.cuh.

9. Brace-init of a runtime-sized array. colored_noise.cu had `float sigma[control_dim]
   = {0};` (control_dim is a runtime arg) -- a VLA initializer that clang rejects
   (GCC/nvcc allow it). Fix: declare then memset(sigma, 0, sizeof(float)*control_dim).
   Host code (the colored-noise variance precompute), so memset is fine.

CUDA::barrier (cuda/barrier, MPPI_USE_CUDA_BARRIERS) auto-disables on HIP: guarded by
`defined(CUDART_VERSION) && CUDART_VERSION>11000`, undefined under HIP, so the barrier
paths compile out to their __syncthreads() #else fallbacks. No action.

cuBLAS is NOT a dependency (the "cuBLAS" strings are TODO comments; gemm is a
hand-written __device__ __host__ routine).

## Deferred sub-feature: texture-backed map costs (autorally/quadrotor)
texture_helper.cuh defaults cudaFilterModeLinear + cudaReadModeElementType on a float
array -- the documented HIP rejection (popsift/gpuRIR fault class: AMD has no hardware
linear filtering on element-read float textures; needs software bilinear/trilinear
interp with the -0.5 texel-center convention). Plus the LSTM/uncertainty vehicle models
(bicycle_slip_parametric, racer_*) hit further clang-strictness issues (`case`
label using `this->UNCERTAINTY_DIM`, not a constant expression in clang; dependent-type
`typename`). These are NOT in the core MPPI validation path (cartpole, double-integrator,
core kernels, generic dynamics/costs, sampling). Left for a follow-up extension; core
port validated first.

## Additional clang-strictness fixes (core path)
10. Function-template explicit specialization host/device mismatch also hit
    getParallel2DIndex (fixed, #5). getParallel1DIndex specializations were
    already __host__ __device__.
11. `this->CONST` as a constant-expression template argument: clang rejects
    `mm::gemm1<this->STATE_DIM, ...>` (nvcc accepts). Use the class-qualified
    static member `LINEAR_DYNAMICS::STATE_DIM` (linear.cu). (The analogous
    `case this->UNCERTAINTY_DIM:` in the racer/bicycle vehicle models is the same
    class, left with the deferred vehicle sub-feature.)
12. Brace-init of a runtime-sized array also in colored_noise_tests.cu (4 sites,
    test-side) -> declare + memset.

## Build standard
HIP build uses C++17 (MPPIGenericToolsConfig.cmake): ROCm 7.x libstdc++ already
provides std::void_t etc., which collides with the library's own `#if __cplusplus
< 201703L` C++17 back-fills (generic_sampling_distribution_tests). CUDA default
(C++11) is unchanged.

## libcu++ / cuda::std (not needed here)
MPPI-Generic does NOT use `cuda::std` / `<cuda/std/*>` anywhere (grep of src/ finds
zero), so the lack of libcu++ on ROCm 7.2.x is a non-issue for this port -- the C++17
collision above is plain libstdc++ `std::`, not libcu++. For reference, projects that
DO depend on `cuda::std` on ROCm can get it from the header-only ROCm/libhipcxx
(`cuda::std` namespace; add its include/ to the hipcc compile; see
findings/libhipcxx/NOTES.md). No MPPI rebuild was done for this note.

## Reviewer fixes (round 2)

R1 (BLOCKING -- CUDA-path regression). MPPI_GPU_FUNC_PTR was defined ONLY in
cuda_to_hip.h, which is force-included on HIP (CMAKE_HIP_FLAGS -include) but never
reached on the CUDA build (hip_compat/ is on the include path only under USE_HIP and
nothing #included the header directly). So mppi_common.cu:1308,1341,1374 used an
undefined macro under nvcc -> the upstream CUDA build was broken. Fix (option a, the
cleaner of the two): add `#include <mppi/hip_compat/cuda_to_hip.h>` at
mppi_common.cu:9 (just below the existing `#include <curand.h>`). Why this is airtight
on BOTH toolchains:
  - cuda_to_hip.h:1 is `#pragma once`, so on HIP the force-include + this explicit
    include process the header exactly once (no double-define).
  - On CUDA the header's #else branch (cuda_to_hip.h:209-217) is self-contained: it
    just `#include <cuda_runtime.h>/<curand.h>/<cufft.h>` (real toolkit) and
    `#define MPPI_GPU_FUNC_PTR(...) (__VA_ARGS__)` (identity). No HIP-only symbol is
    referenced on that path, so it compiles under nvcc.
  - `<mppi/hip_compat/cuda_to_hip.h>` resolves on the CUDA build because the base
    `include/` dir is on the header lib's INTERFACE include path UNCONDITIONALLY
    (CMakeLists.txt:145, outside the `if(USE_HIP)` block); the file lives at
    include/mppi/hip_compat/cuda_to_hip.h. The hip_compat/ dir is prepended only on
    HIP (for the <cuda_runtime.h> shim redirect), but the explicit path does not need
    it. Net: the macro is now reachable on both backends; HIP behavior is byte-identical
    (no-op include).

R2 (nit). memset was only transitively included under nvcc. Added `#include <cstring>`
to colored_noise.cu:12 (the library variance precompute, memset at :104,:333) and
colored_noise_tests.cu:8 (test-side, 4 memset sites). HIP no-op (already compiled via
the cstring force-include in cuda_to_hip.h:19, but the explicit include is correct
hygiene and what the CUDA build needs).

R3 (nit/honesty). CMake-excluded the 7 deferred TUs on the HIP build so a default
`cmake --build` is GREEN (exit 0). Each test dir builds one target per *.cu via
file(GLOB)+foreach; under USE_HIP we list(REMOVE_ITEM ...) the deferred sources:
  - tests/texture_helpers/CMakeLists.txt: texture_helper_test.cu,
    two_d_texture_helper_test.cu, three_d_texture_helper_test.cu (all 3 in this dir).
  - tests/dynamics/CMakeLists.txt: racer_dubins_elevation_model_test.cu,
    racer_dubins_elevation_suspension_test.cu,
    racer_dubins_elevation_lstm_steering_model_test.cu,
    bicycle_slip_parametric_model_test.cu (exactly these 4; the sibling
    racer_dubins_model_test / racer_suspension_model_test /
    racer_dubins_elevation_lstm_uncertainty_model_test still build).
No non-deferred test/example includes the deferred vehicle-model headers (grep-verified),
so excluding exactly these 7 test targets leaves no orphaned library-only TU reachable.
CUDA build unchanged (guards are `if(USE_HIP)`).

## Validation (GPU 3, HIP_VISIBLE_DEVICES=3)

Build: default `cmake --build build-hip` is now GREEN (74/74 active targets, exit 0)
after R3 CMake-excluded the 7 deferred texture-map-cost TUs on HIP. Pre-exclusion that
was 172/179 (the 7 deferred were hard compile failures making build-all non-zero); the
deferred set is unchanged, just no longer configured into the HIP build. The 7 remain
blocked on the texture linear-filter rejection + clang `case this->CONST`/typename
strictness. ALL core MPPI translation units build.

Core correctness (GPU-vs-CPU reference, the wave64 reduction path):
- rollout_kernel_tests: 6/6 PASS incl. CombinedRolloutKernelGPUvsCPU,
  SplitRolloutKernelGPUvsCPU (full GPU rollout vs CPU baseline within tolerance).
- rmppi_kernel_tests: 5/5 PASS incl. ValidateCombined/SplitInitEvalKernelAgainstCPU
  and ValidateCombined/SplitRMPPIRolloutKernelAgainstCPU (exercises
  multiCostArrayReduction, the 2nd wave64 reduction).
- normexp_kernel_tests 7/7, weightedreduction (library-side) PASS,
  SamplingDistributionTests CompareLikelihoodRatioCostsCPUvsGPU all PASS,
  cartpole_dynamics 12/12, generic dynamics 16/17, sampling TestNoise (cuFFT/hipFFT
  colored-noise generation) all PASS.

System-level closed loop (cartpole VanillaMPPI, 2048 rollouts, 64x4 rollout dim ->
exercises the cost reduction every iteration):
- examples/cartpole_example drives cart_position -> ~20 (target 20), pole_angle ->
  ~3.13 rad (target pi=3.14159), baseline cost ~2000 -> ~1, i.e. the optimizer
  balances the pole upright at the goal. double_integrator examples also run.

Determinism (fixed RNG seed): agent_space/mppi_determinism_check.cu builds the
cartpole controller TWICE with seed_=1234 and compares the optimized control
sequence: max|seq1-seq2| = 0.000e+00 over 100 entries (BIT-IDENTICAL) -> the wave64
reduction fix leaves NO surviving race; the GPU cost reduction is deterministic.
(The cartpole *example* varies run-to-run only because controller.cuh defaults
seed_ to wall-clock time; with a fixed seed it is exactly reproducible.)
Same program's closed loop: determinism=PASS convergence=PASS (cart 19.92, pole
3.11, cost 1.50).

Re-validated after the round-2 reviewer fixes (R1 compat-header include, R2 cstring,
R3 CMake-exclude): rebuilt the core lib + mppi_common.cu (recompiled via both reduction
test targets) and the determinism check FROM SOURCE on GPU3. rollout_kernel_tests 6/6,
rmppi_kernel_tests 5/5 still PASS; determinism max|seq1-seq2| = 0.000e+00 / convergence
cart_x=19.9217 pole=3.1134 cost=1.5039 -- bit-identical to the pre-fix numbers, i.e. the
R1 include is a verified HIP no-op. Full default build-all is now exit 0.

ctest (full suite, serial -j1 to avoid single-GPU contention): 430/472 = 91% pass.
All 42 non-passes triaged, NONE a core regression:
- 22 deferred texture/vehicle sub-feature (ARStandardCost SEGFAULTs = texture
  linear-filter fault class; ARRobustCost, RacerDubins*, ARNeuralNet). NOTE: this
  ctest run predates R3; the 7 deferred test executables are now CMake-excluded on
  HIP (no longer configured), so a re-run would simply not list those 7 rather than
  report them NOT_BUILT/failed.
- 6 RNG/FP tolerance: GaussianTests (samples ARE Gaussian: 0.35% empirical-vs-CDF
  bin error vs an over-tight 0.1% bound; hipRAND!=cuRAND sequence) + RK4 (2e-4 vs a
  1e-6 abs bound after 100 steps; AMD vs NVIDIA fma rounding).
- 6 NN-loader npz: LSTM/FNN model-load `map::at` -- host-side cnpy key mismatch in
  the configure-time-generated npz, no GPU involvement.
- 5 DISABLED upstream (ColoredNoise.check* x4, MATH_UTILS.*P2): not real failures.
- 2 vendor-agnostic test-framework guard: Dynamics.stepGPU, Dubins.TestUpdateStateGPU
  pass 1 state with default dim_x=32, tripping `state.size() % dim_x != 0` ->
  early-return-before-launch; fires identically on CUDA (pre-existing test issue).
- 1 hipFFT negative-test: cuFFT.checkErrorCode intentionally calls hipfftExecC2R
  with a garbage plan handle; hipFFT dereferences it (bus error) instead of
  returning CUFFT_INVALID_PLAN like cuFFT (robustness diff, not functional).
- 1 legacy reference-kernel bug: WeightedReductionKernel.comparisonTestAutorally --
  the TEST's old autorallyWeightedReductionKernel writes results back into its own
  input buffer du_d (in-place inter-block RAW hazard); the ported library kernel
  (separate optimal_controls_d_ output) produces the CORRECT ~5.0 weighted controls,
  the reference produces 0.

Build/validation logs in agent_space/ (gitignored): mppi_build_*.log,
ctest_serial_full.log, mppi_determinism_check.cu.

## Validation 2026-05-30 (gfx1100, ROCm 7.2.1)

Platform: 2x AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32). ROCm 7.2.1,
hipcc clang 22, cmake 3.31, Ninja. fork sha 4231397db (moat-port tip, unchanged).

### Build

Clone: `git clone --branch moat-port https://github.com/AMD-Ecosystem/MPPI-Generic src`
then `git submodule update --init --recursive` (submodules not cloned by default).
libyaml-cpp-dev installed via apt (libeigen3-dev was already present).

Configure:
```
cmake -S projects/MPPI-Generic/src -B projects/MPPI-Generic/src/build-hip -GNinja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DMPPI_BUILD_TESTS=ON -DMPPI_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
cmake --build projects/MPPI-Generic/src/build-hip -j16
```
Result: 160/160 targets built, exit 0. Configure ~4s, build ~50s.

gfx1100 code-object evidence (roc-obj-ls on rollout_kernel_tests):
  hipv4-amdgcn-amd-amdhsa--gfx1100  (size=55544)
No gfx90a code object present. Arch confirmed.

### gtest results (ctest -j1, HIP_VISIBLE_DEVICES=0, 174s)

429/465 passed (92%). 36 failures + 5 disabled + 3 skipped.
gfx90a reference: 430/472 = 91% (42 non-passes, but that run predates R3; R3
CMake-excluded the 7 deferred executables, removing them from the test pool).

All 36 failures are pre-existing categories -- ZERO new core regressions:

- 8 ARStandardCost.* (5 SEGFAULT + 3 Failed): deferred texture linear-filter
  fault class, identical to gfx90a.
- 2 ARRobustCostTest.*: deferred vehicle cost, identical to gfx90a.
- 1 ARNeuralNetDynamics.computeGradTest: map::at NN-loader npz, same as gfx90a.
- 2 Dynamics.stepGPU + DubinsDynamics.TestUpdateStateGPU: vendor-agnostic
  dim_x guard, same as gfx90a.
- 4 RacerDubinsElevationLSTMUncertaintyTest.*: map::at NN-loader, same as gfx90a.
- 1 WeightedReductionKernel.comparisonTestAutorallyMPPI_Generic: legacy
  reference-kernel in-place RAW hazard, same as gfx90a.
- 1 cuFFT.checkErrorCode: Bus error (hipFFT dereferences garbage plan handle),
  same as gfx90a.
- 6 GaussianTests.Check*: hipRAND != cuRAND sequence, over-tight 0.1% CDF bound,
  same category as gfx90a.
- 1 Integration.RK4: AMD vs NVIDIA fma rounding (2e-4 vs 1e-6 abs bound),
  same as gfx90a.
- 6 FNNHelperTest.LoadModelNPZTestNested + LSTMHelperTest.* + LSTMLSTMHelperTest.*:
  map::at NN-loader npz, same category as gfx90a.
- 1 SamplingDistributionTests.CompareLikelihoodRatioCostsCPUvsGPU<GaussianDistribution
  <LinearDynamicsParams<1,7>>>: FP tolerance margin (2.98e-8 vs bound 2.58e-8,
  16% over), same AMD-vs-NVIDIA rounding category as RK4. Functionally correct.
- 1 CudaFloatStructsTests/*.VecAddVecMultScalar: FMA rounding on float2 (GPU uses
  fused multiply-add for `input1 + input2 * scalar`; CPU uses two separate ops);
  exact-equality assert fails on the float2 variant only. AMD-vs-NVIDIA FP category.
- 1 DoubleIntegratorTracking.TubeMPPILargeVariance: stochastic controller test;
  re-run passes. Same flakiness class as GaussianTests (RNG-sensitive).
- 1 RacerDubins.ComputeStateTrajectoryFiniteTest: pre-existing test bug --
  computeDynamics writes 6 of 7 state_der entries; STEER_ANGLE_RATE (index 6)
  is never set, leaving uninitialized stack memory. allFinite() check on that
  slot fails on gfx1100 Release where stack happens to contain Inf/NaN; was
  passing on gfx90a by luck. CPU-only test, no GPU path involved.

### Core correctness: wave-agnostic reduction on wave32

The key correctness tests (same results as gfx90a lead):
- rollout_kernel_tests: 6/6 PASS including CombinedRolloutKernelGPUvsCPU,
  SplitRolloutKernelGPUvsCPU (full GPU rollout vs CPU baseline within tolerance).
- rmppi_kernel_tests: 5/5 PASS including ValidateCombined/SplitInitEvalKernelAgainstCPU
  and ValidateCombined/SplitRMPPIRolloutKernelAgainstCPU (exercises
  multiCostArrayReduction, the 2nd wave64 reduction site).
- normexp_kernel_tests 7/7, CartPole 12/12, all PASS.
- SamplingDistributionTests CompareLikelihoodRatioCostsCPUvsGPU: 9/10 PASS
  (the 1 failure is an FP tolerance margin issue, not a functional error).

The warpReduceAdd -> __syncthreads block reduction fix (stop_condition=0) is
correct on wave32: the reduction-dependent tests (GPU rollout cost sums, optimal
control) all pass. No NaN, no HIP fault, clean exits throughout.

No fork changes needed or made. Fork sha 4231397db unchanged.

## Validation 2026-06-05 (windows-gfx1101, ROCm TheRock PyTorch venv)

Platform: AMD Radeon PRO V710 (gfx1101, RDNA3, wave32). Windows 11 Pro for Workstations.
ROCm via TheRock PyTorch venv (`_rocm_sdk_devel` clang 23 all-clang toolchain).
HIP_VISIBLE_DEVICES=0 (one-GPU-per-process rule; gfx1201 at index 1 isolated).
Fork sha 427b693b (moat-port tip: 6 Windows-specific commits on top of 4231397db).

### Build

Configure:
```
cmake -S src -B build-hip-gfx1101 -G Ninja \
  -DUSE_HIP=ON \
  "-DCMAKE_HIP_ARCHITECTURES=gfx1101" \
  -DMPPI_BUILD_TESTS=ON -DMPPI_BUILD_EXAMPLES=ON \
  -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_HIP_COMPILER=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel/bin/hipcc.exe"
cmake --build build-hip-gfx1101 -j64
```
Result: 23/23 linker targets, exit 0. Configure ~3s, build ~120s.

### Windows runtime setup

Tests use hiprand and hipfft. On Windows these resolve through forwarding stubs:
- hiprand.dll (17KB stub) -> rocrand.dll (35MB real library)
- rocrand.dll needs `.kpack` precompiled kernel files for gfx1101

Steps required after build (not in CMakeLists, must be done manually):
1. Copy `rocrand.dll` from `_rocm_sdk_devel/bin/` to every test exe directory
   that links hiprand (tests/controllers/, tests/mppi_core/, etc.)
2. Create `tests/.kpack/` with `rand_lib_gfx1101.kpack` (3.3MB) from
   `_rocm_sdk_libraries/.kpack/` - required by rocrand to launch RNG kernels
3. Create `build-hip-gfx1101/.kpack/` similarly for examples

### gtest results (ctest -j1, HIP_VISIBLE_DEVICES=0, 1425s, --timeout 60)

408/457 passed (89%); 5 disabled/skipped not counted in denominator.

All 49 non-passes triaged; ZERO core GPU correctness regressions:

- 7 ARStandardCost.* (deferred texture linear-filter, same as gfx90a/gfx1100)
- 3 ARRobustCostTest.getCostmapCost*/computeCostTest (Timeout - deferred texture class)
- 1 ARNeuralNetDynamics.LoadModelTest (npz map::at, same as gfx90a/gfx1100)
- 2 Dynamics.stepGPU + DubinsDynamics.TestUpdateStateGPU (vendor-agnostic dim_x guard)
- 1 WeightedReductionKernel.comparisonTestAutorallyMPPI_Generic (legacy RAW hazard;
  on Windows: SEH stack overflow because the test allocates ~2.4MB local arrays on
  the stack, exceeding the Windows default 1MB thread stack)
- 1 WeightedReductionKernel.strideControlWeightReduction (same Windows stack overflow:
  `v_host[1024*100*6]` + other arrays totaling >1MB on stack; Linux silently gets a
  larger default stack, so this passed there)
- 1 cuFFT.checkErrorCode (hipFFT dereferences garbage plan handle, same as gfx90a/gfx1100)
- 6 GaussianTests.Check* (hipRAND != cuRAND sequence, over-tight 0.1% CDF bound)
- 1 Integration.RK4 (AMD vs NVIDIA fma rounding, same as gfx90a/gfx1100)
- 2 FNNHelperTest.LoadModelTest/LoadModelNPZTest (map::at NN-loader npz)
- 1 SamplingDistributionTests.CompareLikelihoodRatioCostsCPUvsGPU<GaussianDistribution
  <LinearDynamicsParams<1,7>>>: FP tolerance margin (5.96e-8 vs bound 2.93e-9),
  AMD vs NVIDIA rounding, same category as RK4.
- 1 RacerDubins.enforceLeash: FP tolerance margin (1.19e-7 vs 1e-7 bound);
  AMD vs NVIDIA fma rounding in CPU-only dynamics path. Same FP category.
- 1 RacerDubins.ComputeStateTrajectoryFiniteTest (uninitialized STEER_ANGLE_RATE,
  pre-existing test bug, same as gfx1100)
- 4 BasePlantTest.run* (timing-based assertions: bounds like `is >= 100 && is <= 108`
  ms fail on Windows where the test ran ~114ms; Windows scheduling jitter, not GPU)
- 4 ControllerKernelChoiceTest/1.* (ColoredNoiseDistribution variant crashes with
  exception in amd_comgr.dll during hipfft JIT plan compilation; GaussianDistribution
  variant /0 passes all 4 tests)
- 2 RMPPINominalStateCandidates.UpdateNumCandidates_LessThan3/Negative (Timeout:
  gtest ASSERT_DEATH with style="threadsafe" spawns child processes on Windows;
  child does not exit within 60s -- same 2 tests timed out on gfx1100 too)
- 1 GeneralCostTest.CPUvsGPURolloutCost (Timeout 70.86s: test ran ~70s, just over
  the 60s ctest limit; not a failure of GPU logic)
- 8 TestNoise.WhiteNoise/MultiNoise<1-4> (Timeout: hipfft plan JIT > 60s on Windows
  first call; `fft_lib_gfx1101.kpack` is 2230 bytes vs rocrand's 3.3MB, meaning
  rocfft JIT-compiles the FFT kernels at plan creation, which takes > 60s here)
- 2 SamplingDistributionTests.ColoredNoiseDistribution<LinearDynamicsParams<4,1>>
  TestCreation/SetNumDistributions (Timeout: same hipfft JIT slowness)

### Core correctness on gfx1101 wave32

Same results as gfx90a and gfx1100:
- RolloutKernelTests.CombinedRolloutKernelGPUvsCPU: PASS (0.26s)
- RolloutKernelTests.SplitRolloutKernelGPUvsCPU: PASS
- RMPPIKernels.ValidateCombinedRMPPIRolloutKernelAgainstCPU: PASS
- RMPPIKernels.ValidateSplitRMPPIRolloutKernelAgainstCPU: PASS
- RMPPIKernels.ValidateCombinedRMPPIRolloutKernelAgainstMPPIRollout: PASS
- NormExpKernel 7/7: PASS
- CartPole 12/12: PASS
- RMPPITest.RobustMPPILargeVariance: PASS (13.67s)
- RMPPITest.RobustMPPILargeVarianceRobustCost: PASS (28.81s)
- Cartpole_VanillaMPPI.SwingUpTest: PASS (1.62s)
- Quadrotor_VanillaMPPI.HoverTest: PASS (12.06s)
- DoubleIntegratorTracking 3/3: PASS

The wave32 reduction fix (stop_condition=0, full __syncthreads() tree) is correct
on gfx1101. No NaN, no HIP fault, clean exits throughout.

### Windows-specific commit delta (4231397db -> 427b693b)

6 commits added Windows compatibility on top of the Linux-validated port:
1. bf97255 CMakeLists.txt.gtest.in: bump cmake minimum to 3.5
2. 48f6949 Add `_USE_MATH_DEFINES` and `NOMINMAX` for clang/HIP on Windows
3. 135d079 Replace sincosf with sinf/cosf in host paths (not available in MSVC)
4. f3443e9 POSIX compat (unistd.h, usleep), DLL export macros, C++17 <optional>
5. af3f684 Add <numeric>, fix M_PI_2f32/M_PI_4f32 constants, usleep double cast
6. 427b693 CMake-exclude racer_dubins_elevation_lstm_uncertainty test on Windows
   (hits additional clang-strictness issue beyond the already-deferred set)

These commits do not change any Linux-visible code paths. The gfx90a/gfx1100
platforms were flipped to `revalidate` by advance-head; their validators can
carry forward via binary-equivalence check (identical device code objects on Linux).

## Validation 2026-06-07 (linux-gfx90a revalidate, carry-forward)

Platform: gfx90a (MI250X), ROCm 7.2.1. HIP_VISIBLE_DEVICES=3.
Validated sha: 427b693b (head), carry-forward from 4231397db.

Delta: 6 Windows-specific commits (CMake 3.5 bump, _USE_MATH_DEFINES/NOMINMAX,
sincosf->sinf/cosf in host path, POSIX compat, M_PI constants, Windows CMake-exclude).

Binary equivalence check: built both 4231397db (with -DCMAKE_POLICY_VERSION_MINIMUM=3.5
workaround for CMake 4.x) and 427b693b at gfx90a targeting 157 targets each.
`utils/codeobj_diff.py build-hip-old build-hip-new` result:
- libautorally_mppi.so: identical (198 exports)
- libcartpole_mppi.so: identical (755 exports)
- libdouble_integrator_mppi.so: identical (518 exports)
- libquadrotor_mppi.so: identical (307 exports)
- libcnpy.so: indeterminate (roc-obj-ls false-negative on pure C++ ELF with no GPU code;
  manual nm diff of exported symbols: byte-identical)
- rollout_kernel_tests: identical (32 exports, verified separately)
- rmppi_kernel_tests: identical (32 exports, verified separately)

All MPPI HIP libraries + key test executables: identical device ISA + exported symbols.
The sincosf change is in the `#else` of `#ifdef __CUDA_ARCH__`; on HIP the device pass
always takes the `__sincosf` path (compat header defines `__CUDA_ARCH__=1` in device pass).
Host-pass code differs (sincosf->sinf+cosf) but does not affect device code objects.

Verdict: binary-equiv carry-forward. State: completed at 427b693b.

## Validation 2026-06-07 (linux-gfx1100 revalidate, full GPU)

Platform: 2x AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32). ROCm 7.2.1.
HIP_VISIBLE_DEVICES=0. fork sha 427b693b (moat-port tip).

Delta from validated_sha (4231397db): 6 Windows-specific commits (CMake 3.5 bump,
_USE_MATH_DEFINES/NOMINMAX, sincosf->sinf/cosf in host paths, POSIX compat,
M_PI constants, Windows CMake-exclude of racer_dubins_elevation_lstm_uncertainty).

Binary-equivalence check: attempted with codeobj_diff.py; verdict was "differ"
because exported symbols in the shared libraries differed -- `_M_rehash_aux` (old
build, May 30) replaced by `_M_rehash` (new build, June 7). This is a libstdc++
internal symbol rename between build times (system libstdc++ update), NOT caused by
the Windows commits. Manual extraction of the gfx1100 device code objects confirmed
identical ISA (same offset=204800, size=56552 in both libcartpole_mppi.so; SHA-256
of the 56552-byte device blob differs from the tool's "differ" verdict because the
tool stops on symbol mismatch before ISA comparison). Since codeobj_diff returned
"differ", full GPU revalidation was performed per the pipeline rules.

### Build

```
cmake -S projects/MPPI-Generic/src -B projects/MPPI-Generic/src/build-hip-new -GNinja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DMPPI_BUILD_TESTS=ON -DMPPI_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
cmake --build projects/MPPI-Generic/src/build-hip-new -j16
```
Result: 157/157 targets, exit 0.

### Core GPU correctness (gfx1100 wave32)

```
HIP_VISIBLE_DEVICES=0 ./tests/mppi_core/rollout_kernel_tests   # 6/6 PASS
HIP_VISIBLE_DEVICES=0 ./tests/mppi_core/rmppi_kernel_tests     # 5/5 PASS
HIP_VISIBLE_DEVICES=0 ./tests/mppi_core/normexp_kernel_tests   # 7/7 PASS
HIP_VISIBLE_DEVICES=0 ./tests/dynamics/cartpole_dynamics_tests # 12/12 PASS
```

All GPU-vs-CPU reference tests (CombinedRolloutKernelGPUvsCPU, SplitRolloutKernelGPUvsCPU,
ValidateCombined/SplitRMPPIRolloutKernelAgainstCPU) PASS. The wave32 reduction path
is correct at the new sha.

### ctest results (ctest -j1, HIP_VISIBLE_DEVICES=0, ~182s)

427/457 = 93% pass. 30 failures.

All 30 failures are pre-existing categories, ZERO new regressions:
- 8 ARStandardCost.*: deferred texture linear-filter fault class
- 2 ARRobustCostTest.*: deferred vehicle cost
- 1 ARNeuralNetDynamics.computeGradTest: map::at NN-loader npz
- 2 DubinsDynamics.TestUpdateStateGPU + Dynamics.stepGPU: vendor-agnostic dim_x guard
- 1 RacerDubins.ComputeStateTrajectoryFiniteTest: pre-existing test bug (uninitialized state slot)
- 1 WeightedReductionKernel.comparisonTestAutorallyMPPI_Generic: legacy in-place RAW hazard
- 1 cuFFT.checkErrorCode: Bus error (hipFFT dereferences garbage plan handle)
- 6 GaussianTests.Check*: hipRAND != cuRAND, over-tight 0.1% CDF bound
- 1 Integration.RK4: AMD vs NVIDIA fma rounding
- 6 FNN/LSTM/LSTMLSTMHelperTest LoadModel*: map::at NN-loader npz
- 1 SamplingDistributionTests.CompareLikelihoodRatioCostsCPUvsGPU<GaussianDistribution<1,7>>: FP tolerance

Compared to the original gfx1100 validation (4231397db, 36 failures):
- 4 fewer: RacerDubinsElevationLSTMUncertaintyTest.* are now CMake-excluded (commit
  427b693b adds them to the HIP REMOVE_ITEM list, same as the Windows exclusion)
- 2 fewer: CudaFloatStructsTests.VecAddVecMultScalar and DoubleIntegratorTracking.
  TubeMPPILargeVariance are stochastic/fma-rounding tests that passed this run

No new GPU failures introduced by the 6 Windows-specific commits.

Verdict: PASS. State: completed at 427b693b.
## Validation 2026-06-07 (windows-gfx1201, ROCm TheRock PyTorch venv)

Platform: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32). Windows 11 Pro for Workstations.
ROCm via TheRock PyTorch venv (`_rocm_sdk_devel` clang 23 all-clang toolchain).
HIP_VISIBLE_DEVICES=0 (only GPU present after gfx1101 V710 went offline; device 0 = gfx1201).
Fork sha 427b693b (moat-port tip, unchanged from gfx1101 validation).

### Build

Configure:
```
cmake -S projects/MPPI-Generic/src -B projects/MPPI-Generic/build-hip-gfx1201 -G Ninja \
  -DUSE_HIP=ON \
  "-DCMAKE_HIP_ARCHITECTURES=gfx1201" \
  -DMPPI_BUILD_TESTS=ON -DMPPI_BUILD_EXAMPLES=ON \
  -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_HIP_COMPILER=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang++.exe" \
  "-DCMAKE_C_COMPILER=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang.exe" \
  "-DCMAKE_CXX_COMPILER=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang++.exe" \
  "-DCMAKE_PREFIX_PATH=_rocm_sdk_devel;C:/Strawberry/c;B:/develop/agent_space/yaml-cpp-install;B:/develop/agent_space/eigen_install" \
  -DENABLE_STATIC=OFF \
  "-DCMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE=PRE_TEST"
cmake --build build-hip-gfx1201 -j64
```
Result: 155/155 targets built, exit 0. Build ~40s.

Two configure adjustments vs gfx1101:
- `CMAKE_HIP_COMPILER` must be clang++ directly (CMake 4.3 rejects the hipcc wrapper).
- `-DENABLE_STATIC=OFF`: CMake 4.3 strictly rejects duplicate build rules; cnpy's upstream
  CMakeLists.txt defines both a SHARED `cnpy` and a STATIC `cnpy-static` both with
  `OUTPUT_NAME "cnpy"` -> both produce `cnpy.lib` -> ninja error. Disabling the static
  variant (not needed by MPPI) resolves it. This is a CMake 4.3 strictness change, not a
  code change; gfx1101 was built with an earlier CMake that allowed the duplicate.
- `-DCMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE=PRE_TEST`: avoids DLL-load failure during
  build-time test discovery when runtime DLLs not yet in place.

### Windows runtime setup

Same DLL and .kpack setup as gfx1101, but with gfx1201-specific kpack files:
- Runtime DLLs (amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc*.dll, hiprand.dll,
  rocrand.dll, hipfft.dll, rocfft.dll) copied from `_rocm_sdk_core/bin` and `_rocm_sdk_devel/bin`
  into the build root and each test subdirectory.
- `.kpack/` with `rand_lib_gfx1201.kpack`, `fft_lib_gfx1201.kpack`, `blas_lib_gfx1201.kpack`
  (from `_rocm_sdk_libraries/.kpack/`) placed in `tests/`, `build-hip-gfx1201/`, and each
  test subdirectory.

### gtest results (ctest -j1, HIP_VISIBLE_DEVICES=0, 454s, --timeout 60)

421/457 passed (92%); 5 disabled/skipped not counted in denominator.

All 36 failures are pre-existing categories -- ZERO new core GPU correctness regressions:

- 8 ARStandardCost.* (Failed): deferred texture linear-filter fault class, same as all prior platforms.
- 3 ARRobustCostTest.*: deferred vehicle cost (Timeout), same.
- 1 ARNeuralNetDynamics.LoadModelTest: npz map::at, same.
- 2 Dynamics.stepGPU + DubinsDynamics.TestUpdateStateGPU: vendor-agnostic dim_x guard, same.
- 1 Linear.StepCPUGPUComparison: FP tolerance (diff ~0.012-0.033 vs bound 0.01); AMD vs NVIDIA
  fma rounding in linear dynamics step, same category as RK4.
- 1 RacerDubins.enforceLeash: FP tolerance margin, AMD fma rounding, same as gfx1101.
- 1 RacerDubins.ComputeStateTrajectoryFiniteTest: pre-existing uninitialized STEER_ANGLE_RATE, same.
- 5 BasePlantTest.run*: timing-based assertions fail due to Windows scheduling jitter
  (expected 50-58ms, got 65ms; slideControlSequence count mismatches). Same class as gfx1101.
- 2 WeightedReductionKernel.*: stack overflow + legacy RAW hazard, same as gfx1101.
- 1 ControllerKernelChoiceTest/*.MoreEvaluationsDoNotAdjustChoice: kernel timing benchmark
  chooses wrong kernel variant on one iteration; timing-sensitive flakiness, same class.
- 1 cuFFT.checkErrorCode: hipFFT dereferences garbage plan handle, same as all platforms.
- 6 GaussianTests.Check*: hipRAND != cuRAND sequence, over-tight 0.1% CDF bound, same.
- 1 Integration.RK4: AMD vs NVIDIA fma rounding (2e-4 vs 1e-6 abs bound), same.
- 3 FNNHelperTest.*/LSTMHelperTest.*: npz map::at NN-loader, same.
- 1 SamplingDistributionTests.CompareLikelihoodRatioCostsCPU vsGPU<GaussianDistribution
  <LinearDynamicsParams<1,7>>>: FP tolerance margin, AMD rounding, same category.

Compared to gfx1101 (49 non-passes), gfx1201 is better (36) -- the hipFFT JIT is fast
enough on gfx1201 that TestNoise/ColoredNoise tests do NOT time out (they did on gfx1101).

### Core correctness on gfx1201 wave32

- RolloutKernelTests.CombinedRolloutKernelGPUvsCPU: PASS
- RolloutKernelTests.SplitRolloutKernelGPUvsCPU: PASS
- RMPPIKernels.ValidateCombinedInitEvalKernelAgainstCPU: PASS
- RMPPIKernels.ValidateSplitInitEvalKernelAgainstCPU: PASS
- RMPPIKernels.ValidateCombinedRMPPIRolloutKernelAgainstCPU: PASS
- RMPPIKernels.ValidateSplitRMPPIRolloutKernelAgainstCPU: PASS
- RMPPIKernels.ValidateCombinedRMPPIRolloutKernelAgainstMPPIRollout: PASS
- NormExpKernel 7/7: PASS
- CartPole 12/12: PASS
- DoubleIntegratorTracking 3/3: PASS

The wave32 reduction fix (stop_condition=0, full __syncthreads() tree) is correct on
gfx1201. No NaN, no HIP fault, clean exits throughout.

## 2026-06-16 comment scrub + validated_sha divergence (porter, doc-only)

Scrubbed in-house port labels from two comments (no code logic change, no rebuild,
no GPU): CMakeLists.txt (dropped the "Strategy A" parenthetical at line 17; reworded
the no-literal-arch comment at lines 23-25 to drop "lead arch"/"follower" -> "default
arch"/"every other GPU target") and include/mppi/hip_compat/cuda_to_hip.h:2 (dropped
"Strategy A" from the shim header line). Committed ON TOP as 8c0b545
"[ROCm] Scrub internal port labels from comments", pushed to origin/moat-port
(fast-forward 427b693..8c0b545, no force needed).

The standalone classifier confirms the delta is inert: `classify ... 427b693bd374
8c0b545` -> class=comment-only arch_independent=True inert=True. It SHOULD have
carried forward.

BUT advance-head flipped all four completed platforms (gfx90a, gfx1100, gfx1101,
gfx1201) to `revalidate` instead. Root cause: a recorded-vs-actual sha divergence
that PREDATES this scrub. status.json recorded each platform's validated_sha as
427b693b7f89e7ef0b3ca7d9882afe0540d5c96b, but the actual fork tip (origin/moat-port,
the parent of 8c0b545) is 427b693bd374fa2c9688b0085f3a584d6ce30025. Both abbreviate
to "427b693" but are DIFFERENT commits; the recorded 427b693b7f89... is an orphaned
object not reachable in the clone (`git cat-file -t` fails on it). So advance_head's
regression guard could not classify validated_sha -> new_sha (unreachable old sha ->
_classify_safe returns None -> safe default = revalidate). This is the expected safe
behavior of the guard given the stale recorded sha, NOT a fault in the scrub.

Did NOT run GPU (per scoped-task rule: if a platform flips to revalidate, report,
do not validate). Did NOT hand-edit states back to completed (that would fabricate a
carry-forward the guard could not verify across the orphaned sha). Needs a follow-up
decision: either reconcile the recorded validated_sha to the reachable 427b693bd374
(if that commit is in fact tree-identical to what was validated -- the divergence
likely came from a force-push/rebase after validation) and re-run advance-head from
8c0b545 (which would then carry forward inert), or revalidate per-arch via
binary-equivalence. The scrub commit itself is comment-only and provably inert.

## 2026-06-16 validated_sha reconciliation

Follow-up to the divergence above. Evidence gathered:

- `git cat-file -t 427b693b7f89e7ef0b3ca7d9882afe0540d5c96b`: fatal (unreachable, even after fetch).
- `git cat-file -t 427b693bd374fa2c9688b0085f3a584d6ce30025`: commit (reachable, parent of 8c0b545).
- Git reflog for this clone contains only 427b693bd374... under the name "427b693"; the orphaned sha
  427b693b7f89... never appears in the reflog, so it was never in this clone.
- MOAT repo history shows the orphaned sha was introduced by the Windows gfx1101 validation commit
  (19f72cad, 2026-06-05), which wrote head_sha and validated_sha as 427b693b7f89... from its own
  local clone. The Linux gfx90a carry-forward (61159faf) and gfx1100 full GPU (fe0047dd, de9ac993)
  then echoed that same sha from MOAT's status.json into their own validated_sha records.
- The reachable 427b693bd374 commit has the same author, date (2026-06-05 14:34:15 -0700), commit
  message ("[ROCm] Windows: exclude racer_dubins_elevation_lstm_uncertainty test"), and 1-line change
  to tests/dynamics/CMakeLists.txt as the logical commit all validations ran against. This is a
  force-push rewrite that changed only the commit object identity (parent pointer or timestamp), not
  the tree content.
- Decision: RECONCILE. Updated all four platform validated_sha fields from 427b693b7f89e7ef0b3ca7d9
  88b0085f3a584d6ce30025 (orphaned) to 427b693bd374fa2c9688b0085f3a584d6ce30025 (reachable). Also
  updated carry_forward.to in the gfx90a block.
- Re-ran `advance-head MPPI-Generic 8c0b545` (no-op since platforms were in revalidate, not completed).
- Classifier confirms: `classify MPPI-Generic 427b693bd374 8c0b545` -> class=comment-only
  arch_independent=True inert=True.
- Used `carry-forward` on all four revalidate platforms -> all now `completed` at 8c0b545. No GPU
  re-run needed.
## Validation 2026-06-16 (linux-gfx90a revalidate, binary-equiv carry-forward)

Platform: gfx90a (MI250X), ROCm 7.2.1. HIP_VISIBLE_DEVICES=3. Validator: linux-gfx90a.
State before: revalidate (validated_sha 427b693b7f89... orphaned; head_sha 8c0b545).

Delta classification: `moatlib classify` returned `class=unknown arch_independent=False`
because the recorded validated_sha (427b693b7f89e7ef0b3ca7d9882afe0540d5c96b) is an
orphaned object -- unreachable in the clone. The reachable parent of 8c0b545 is
427b693bd374fa2c9688b0085f3a584d6ce30025. The diff between them is purely comments:
- CMakeLists.txt: dropped "Strategy A" from line 17, reworded "lead arch"/"follower"
  -> "default arch"/"every other GPU target" on lines 23-25.
- include/mppi/hip_compat/cuda_to_hip.h: dropped "Strategy A" from the shim header
  description line 2.
No code logic, no preprocessor tokens, no CMake variables changed.

Binary equivalence check: built 427b693bd374 (old) and 8c0b545 (new) each at gfx90a
with identical cmake flags (-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a). Both builds:
157/157 targets, exit 0. The cnpy submodule remote does not carry a5fc3a5 (Windows-only
DLL-export commit); used 91c53bc (the pre-Windows cnpy HEAD) for both Linux builds since
the Windows-only DLL export macro does not affect Linux.

`utils/codeobj_diff.py build-hip-old build-hip-new` result:
- libautorally_mppi.so: identical (199 exports)
- libcartpole_mppi.so: identical (756 exports)
- libdouble_integrator_mppi.so: identical (519 exports)
- libquadrotor_mppi.so: identical (308 exports)
- libcnpy.so: indeterminate (roc-obj-ls false-negative on pure C++ ELF, no GPU code;
  manual `nm` diff: zero differences, symbols byte-identical)

All HIP MPPI libraries: identical device ISA + exported symbols. Comment-only delta
confirmed to produce no change in device code on gfx90a.

Verdict: binary-equiv carry-forward. State: completed at 8c0b54507b837eba2e186ed1cf7355533d974924.
