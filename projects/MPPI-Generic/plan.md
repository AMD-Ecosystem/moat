# MPPI-Generic ROCm/HIP port plan (lead: linux-gfx90a)

Upstream: ACDSLab/MPPI-Generic @ b5c8daa (header-only C++/CUDA MPPI control library).

## Existing AMD support assessment

None. README requires "An NVIDIA GPU" + CUDA 10+. No occurrence of rocm/hip/amdgpu/gfx
anywhere in the tree (grep clean). This is a fresh CUDA->HIP port, not a finish-the-port
or improve-the-port case.

## Build classification + strategy

Pure CMake project, `project(... LANGUAGES C CXX CUDA)`, no Torch (`find_package(Torch)`
absent, no `torch.utils.cpp_extension`). => **Strategy A**: one compat
header, mark `.cu` as `LANGUAGE HIP` under a `USE_HIP` option, arch from
`CMAKE_HIP_ARCHITECTURES`. Host C++ (cnpy.cpp, test_main.cpp) stays untouched.

Layout: header-only INTERFACE lib `mppi_header_only_lib` (all logic in include/*.cuh,*.cu);
tests (GoogleTest, one exe per *.cu) and examples are the only compiled `.cu` translation
units. Deps: Eigen3, cnpy (submodule), yaml-cpp (tests/examples), cuRAND, cuFFT, GoogleTest
(downloaded at configure). cuBLAS is NOT used (the "cuBLAS" strings are TODO comments; GEMM
is a hand-written `__device__ __host__` routine in matrix_mult_utils.cuh).

## CUDA surface

- **Warp reductions: YES, and they carry the wave64 fault.** No `__shfl`/`__ballot`/
  `warpSize` anywhere; cost/weight reductions go through shared memory + `__syncthreads()`
  (`costArrayReduction`, mppi_common.cu:1198) which is wave-size agnostic in its tree loop.
  BUT the tail handoff `warpReduceAdd<BLOCKSIZE>` (mppi_common.cu:1170, also rmppi_kernels.cu)
  is the classic CUDA unrolled warp-synchronous reduction: `volatile float*`, steps of
  32->16->8->...->1 with NO `__syncwarp()` between them, relying on 32-lane lockstep. The
  caller switch stops the tree at 32 and dispatches `warpReduceAdd<64>` for the last 64->1
  (mppi_common.cu:1221, rmppi_kernels.cu:1158). On gfx90a (wave64) the low 32 lanes of a
  64-wide wavefront are NOT guaranteed lockstep across these unsynced steps -> races ->
  wrong/non-deterministic cost sums. This is THE correctness fix.
- **cuRAND: YES (random rollouts).** Host generator API only (`curandCreateGenerator`,
  `curandGenerateNormal/Uniform/LogNormal/Poisson`, `curandSetPseudoRandomGeneratorSeed`,
  `curandSetStream`, `curandSetGeneratorOffset`, `CURAND_RNG_PSEUDO_DEFAULT`,
  `curandStatus_t`, `CURAND_STATUS_*`). 1:1 to hipRAND host API (all symbols confirmed
  present in /opt/rocm/include/hiprand/hiprand.h). No device-side curand_kernel state.
- **cuFFT: YES (colored-noise sampling).** `cufftPlan1d` C2R, `cufftExecC2R`, `cufftSetStream`,
  `cufftDestroy`, `cufftComplex`, `cufftHandle`, `CUFFT_C2R`, `cufftResult`, `CUFFT_*`.
  1:1 to hipFFT (confirmed present).
- **Eigen: YES**, pervasive on host AND device (`Eigen::Matrix` in `__device__` code, with
  the EIGEN_MAX_STATIC_ALIGN note). Header-only, backend-agnostic; compiles under hipcc as
  it does under nvcc. Installed libeigen3-dev 3.4.0.
- **Textures: YES** (autorally + quadrotor MAP costs, and the texture_helper utils).
  `texture_helper.cuh:50` defaults `cudaFilterModeLinear` + `cudaReadModeElementType` on a
  float array -> the documented HIP rejection fault class (popsift/gpuRIR). 2D and 3D layered
  texture helpers. NOTE: cartpole and double-integrator (the core validation systems) do NOT
  use textures; textures are an autorally/quadrotor-only sub-feature.
- **cuda::barrier: YES but auto-disabled on HIP.** `cuda/barrier` is guarded by
  `defined(CUDART_VERSION) && CUDART_VERSION > 11000` (mppi_common.cu:8), which is undefined
  under HIP, so the barrier path compiles out and every use has a `#else __syncthreads()`
  fallback. No action needed; do not try to port libcu++ barriers.
- **atomics**: a single `atomicAdd` in racer_dubins_elevation_suspension_lstm.cu (float add,
  fine on AMD; not int min/max). Not in the core/validation path.
- Runtime symbols: streams, malloc/free (+Async), memcpy, events, deviceProps, funcAttrs --
  all 1:1 hip spellings.

## Fault classes in play + fixes

1. **wave64 warp-synchronous reduction (CRITICAL, core path).** `warpReduceAdd` assumes
   32-lane implicit lockstep. Fix: make the reduction wavefront-safe under HIP. Approach:
   add `__syncwarp()` between the unrolled steps when compiling for HIP (HIP `__syncwarp()`
   syncs the whole wavefront, which is correct and sufficient here), and route the caller's
   power-of-2 tail handoff through a warp width that matches the arch. Concretely: introduce
   a `kWarpSize` per-arch constant (64 on `__GFX9__`, else 32; 32 on CUDA) and stop the tree
   reduction at `kWarpSize` instead of literal 32, then `warpReduceAdd<2*kWarpSize-aware>`.
   Keep the CUDA path byte-for-byte (guarded by `#if defined(USE_HIP)`/`__HIP_PLATFORM_AMD__`).
   Simpler equivalent that is provably correct on both: drop the unsynced warp-tail and let
   the `__syncthreads()` tree run all the way down (size>0), i.e. replace the
   `stop_condition=32`/`warpReduceAdd` tail with a fully-synced reduction on HIP. Will pick
   the minimal correct variant during porting and record the exact choice in notes.md.
2. **Texture hardware linear filtering rejection (autorally/quadrotor only).** `texture_helper`
   default `cudaFilterModeLinear` + element-read float -> HIP rejects at create. Same fix as
   popsift/gpuRIR: on HIP create the texture `cudaFilterModePoint` and do software bilinear/
   trilinear interpolation honoring the -0.5 texel-center convention, behind the texture-fetch
   helper. Only needed if the texture tests/costs are in the validation set; gated to that
   sub-feature so the core port lands first.
3. Out-of-bounds / pitch / rule-of-five: re-check the texture array binds (2D pitch, layered
   coherency) IF texture path is validated; not expected in the core MPPI kernels.

## Compat header

Add `include/mppi/utils/cuda_to_hip.h` (Strategy A): on `USE_HIP`/`__HIP_PLATFORM_AMD__`
include `<cstring>`/`<cstdlib>` then `<hip/hip_runtime.h>` and `#define` the cuda* runtime,
curand*, cufft* spellings the project uses to hip*; else include the CUDA runtime/curand/cufft.
Include it from the central hub `include/mppi/utils/gpu_err_chk.cuh` (which every kernel TU
already pulls in and which is where `<cuda_runtime.h>`,`<cufft.h>`,`<curand.h>` live), so the
footprint is one new header + a few-line edit there. Use
torch/utils/hipify/cuda_to_hip_mappings.py as the authoritative name source.

## CMake changes (minimal, arch-configurable)

In MPPIGenericToolsConfig.cmake / top CMakeLists, add `option(USE_HIP ... OFF)`:
- `enable_language(HIP)`; default `CMAKE_HIP_ARCHITECTURES` to gfx90a ONLY when unset (never
  a literal that overrides the cache var -- followers gfx1100/gfx1151 must build with only
  `-DCMAKE_HIP_ARCHITECTURES=<arch>`).
- swap the extra-libs from `CUDA::curand CUDA::cufft` to `hip::hiprand hip::hipfft` (or the
  `/opt/rocm/lib` libs) and skip `find_package(CUDAToolkit)` on the HIP path.
- mark the project `.cu`/`.cuh-compiled` sources `LANGUAGE HIP` (tests/examples globs +
  any compiled lib TU); the INTERFACE header lib needs the include dirs but no language.
- keep `MPPI_USE_CUDA_BARRIERS` OFF-effect on HIP automatically (macro guard handles it);
  no need to force the option.
CUDA path stays exactly as-is when `USE_HIP=OFF`.

## Build + validation (GPU 3, HIP_VISIBLE_DEVICES=3)

Build: `cmake -S src -B build-hip -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
-DMPPI_BUILD_TESTS=ON -DMPPI_BUILD_EXAMPLES=ON
-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++` then `cmake --build build-hip`.
Build dir lives outside git (build-hip/, gitignored at repo root / not under src tree commit).

Validate FOR REAL on gfx90a:
1. **Core kernel CPU-vs-GPU tests** (the reduction path): `rollout_kernel_tests`,
   `rmppi_kernel_tests` from tests/mppi_core -- these compute rollouts on GPU and compare to a
   CPU reference with EXPECT_NEAR. Passing proves the wave64 reduction fix is numerically
   correct. Also run the broader mppi_core + controllers + math_utils + sampling_distributions
   suites (skip/scope texture+autorally+quadrotor suites if the linear-filter sub-feature is
   deferred).
2. **Example closed-loop rollout**: run `cartpole_example` (2048 rollouts, dynamics_rollout_dim
   64x4 -> exercises the cost reduction) and confirm the controller drives the cart toward the
   desired terminal state (cart_position -> 20, pole angle -> PI) with the baseline cost
   trending down -- i.e. the optimizer actually works, matching expected control behavior.
   double_integrator_example as a second system.
3. **Determinism**: fix the cuRAND seed (controllers expose a seed) and run a kernel test /
   example twice; assert identical optimized control/cost across runs. Non-determinism would
   flag a surviving wave64 race in the reduction.
Reference comparison = the library's own CPU baselines in the gtest CPU-vs-GPU cases +
expected physical convergence of the example. Compile/lint is explicitly NOT the gate.

## Out of scope / deferred (record in notes if hit)

- autorally/quadrotor MAP-cost texture path (linear-filter software-interp fix) -- separable
  sub-feature behind the documented fault class; land + validate the core MPPI first, then
  extend if the texture suites are needed for "real" validation. neural-net (LSTM) dynamics
  models pull the same texture/atomic edges.
- Do NOT add GitHub Actions; do NOT run gen_readme; commit only status.json/plan.md/notes.md
  + a PORTING_GUIDE changelog line. Parent delivers the fork.

## Final status (lead linux-gfx90a: ported)

Core MPPI ported and GPU-validated on gfx90a. 172/179 HIP TUs build; all core MPPI
TUs build and the core GPU-vs-CPU correctness suite, the cartpole closed-loop
example, and a fixed-seed bit-identical determinism check all PASS (details in
notes.md). The wave64 warp-synchronous reduction was the one real correctness fault;
the rest were clang-strictness + toolkit-mapping mechanics, all USE_HIP-guarded with
the CUDA path unchanged.

Deferred (separate follow-up, recorded in notes.md): the texture-backed map-cost
sub-feature -- the autorally/quadrotor costs + racer_dubins/bicycle_slip vehicle
dynamics that consume texture_helper. It needs (a) the texture linear-filter
software-interpolation fix (popsift/gpuRIR fault class -- texture_helper defaults
cudaFilterModeLinear+element-read float, rejected on AMD), (b) HIP texture-object is
a pointer not an int so gtest `EXPECT_EQ(tex,0)` comparisons need adjusting, and
(c) the `case this->CONST` / dependent-typename clang fixes across those models.
7 TUs (3 texture-helper tests + 4 vehicle models) do not build pending this; ~22
ctest cases are in this bucket. Not required for core MPPI correctness.
