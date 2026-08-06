# LichtFeld-Studio -- ROCm/HIP port plan (lead platform: linux-gfx90a)

## Project
- Name: LichtFeld-Studio
- Upstream: https://github.com/MrNeRF/LichtFeld-Studio
- Default branch: master
- Pinned at clone: 730b713 ("Vulkan barriers and tighten interop lifetimes")
- What it is: a full 3D Gaussian Splatting workstation -- COLMAP-dataset trainer
  (3DGS + MCMC + 3DGUT/unscented), real-time Vulkan viewer, gaussian-selection
  editor, Python/nanobind plugin system, MCP automation server, and exporters
  (PLY/SOG/SPZ/USD/HTML). C++23, CUDA 12.8+, vcpkg-driven, NVIDIA-only today.

## Existing AMD support
None. Decision: PROCEED (port adds clear value).
- Evidence: zero hip/rocm/gfx/__HIP_ tokens anywhere in CMake/src/docker. README:
  "LichtFeld Studio targets NVIDIA GPUs"; ships CUDA DLLs on Windows; CMake hard-
  requires `find_package(CUDAToolkit 12.8 REQUIRED)` and detects arch from
  `nvidia-smi`. No OpenCL/Vulkan-compute path for the splatting math (Vulkan here
  is the GUI/rasterized-preview layer, not a compute backend for training).
- The GPU compute (custom CUDA tensor library + 3DGS rasterizers + training
  kernels) has no AMD path at all -> a HIP port is genuinely new capability.

## Build classification: pure CMake (Strategy A)
Evidence:
- Top-level `project(LichtFeld-Studio ... LANGUAGES CUDA CXX C)` (CMakeLists.txt:33);
  `enable_language(CUDA)` (285); `find_package(CUDAToolkit 12.8 REQUIRED)` (286).
  Plain `.cu` translation units compiled by the CUDA language, NOT a torch extension.
- NO `find_package(Torch)`, no `torch.utils.cpp_extension`, no `CUDAExtension` in the
  main app or its libraries. The project ships its OWN libtorch-free tensor library
  (`src/core/tensor/`, `lfs_core` is documented "LibTorch-free core + tensor library",
  src/core/CMakeLists.txt:5; `lfs_training` "LibTorch-free API", src/training/CMakeLists.txt:148).
- libtorch appears ONLY in `tests/CMakeLists.txt:65-76` (`find_package(Torch REQUIRED)`,
  `Torch_DIR` -> `external/release/libtorch`). It is a TEST-ONLY parity oracle: the
  `test_tensor_*`/`test_torch_*`/`test_pytorch_install_proof` cases compare the project's
  own tensor lib against PyTorch. It is not a build dependency of the shipped binary.
=> Strategy A: one CUDA->HIP compat header, mark the project's `.cu`
   as LANGUAGE HIP under an `option(USE_HIP ...)`, guard the few genuinely divergent
   spots. NVIDIA build stays byte-for-byte. (PORTING_GUIDE "Strategy A".)

## Scope decision (IMPORTANT -- read before porting)
This is a very large desktop app. The full build pulls ~40 vcpkg packages (USD,
ffmpeg, OpenImageIO, SDL3, Vulkan, RmlUi, nanobind, assimp, Boost, glslang,
shader-slang, ...) plus a vendored NVIDIA-only codec (external/nvImageCodec =
nvjpeg/nvjpeg2k). Porting the entire app (GUI + Python + MCP + USD + nvjpeg +
CUDA<->Vulkan external-memory interop) to ROCm is out of proportion to MOAT's
goal of obliterating the CUDA moat on the GPU compute.

The compute that matters is cleanly isolated and libtorch-free:
- `lfs_core` (custom CUDA tensor library) -- deps: glm, CUDA, OpenMP, dl only.
- `lfs_core_cuda` (memory arena / cuda utils).
- `lfs_training` + `lfs_training_kernels` (Adam, schedulers, losses, SSIM, MCMC,
  densification/pruning, bilateral grid, ppisp) -- deps: CUDA, curand, glm, the
  rasterizer backends.
- `fastlfs_backend` (fastgs forward/backward rasterizer + fused Adam).
- `gsplat_backend_lfs` (vendored gsplat: projection, intersect/tile, rasterize
  fwd/bwd, SH, relocation) -- deps: CUDA, curand, glm.
- `edge_compute` rasterizer.
These compile against CUDA + glm + cub/thrust + curand only. The GUI/Vulkan/USD/
nvjpeg bloat lives in `lfs_visualizer`, `lfs_io`, `lfs_mcp`, `lfs_python_utils`.

PLAN OF RECORD: port the compute libraries above to HIP and GPU-validate the
compute test set + a headless training smoke. Treat the GUI/Vulkan-interop,
Python/MCP, USD, and nvImageCodec/nvjpeg layers as a SEPARATE, explicitly
deferred follow-on (documented in Open questions). Rationale precedent: a
correctness-first mechanical port of the hot/compute kernels is a valid first
step (PORTING_GUIDE, "Before porting"); the moat here is the splatting math, not
the GTK file dialog. The porter should add `option(USE_HIP ...)` such that
`-DUSE_HIP=ON` builds the compute libs + the compute tests and does NOT force the
GUI/USD/nvjpeg subtrees (gate `add_subdirectory(src/visualizer|mcp|python)` and
`external/nvImageCodec` behind `if(NOT USE_HIP)` for the lead bringup). This keeps
the NVIDIA build untouched and gives the validator a buildable compute target
without a multi-hour USD/ffmpeg vcpkg bootstrap on top of unported nvjpeg.

If, once configured, the compute test binary cannot be decoupled from the GUI
graph without invasive surgery (tests link one monolithic `lichtfeld_tests`
target -- see Test plan), fall back to a dedicated HIP compute-test executable
that links only {lfs_core, lfs_core_cuda, lfs_training, lfs_training_kernels,
fastlfs_backend, gsplat_backend_lfs, GTest, libtorch-rocm} and the ~93 compute
test sources. This is a build-file addition, not a source change.

## Port strategy: A (compat header + LANGUAGE HIP), per-subtree gated
Rationale: pure-CMake CUDA project; the compat-header approach keeps the diff minimal and
the NVIDIA path identical. Concretely:
1. Add `src/core/include/core/cuda/cuda_to_hip.h` (single compat header). On
   ROCm (`USE_HIP`/`__HIP_PLATFORM_AMD__`) include `<hip/hip_runtime.h>` and alias
   only the cuda* symbols actually used (cudaMalloc/Free/Memcpy*/Memset*,
   cudaStream_t, cudaEvent_*, cudaError_t/cudaSuccess, cudaDeviceSynchronize,
   cudaGetLastError, cudaSurfaceObject_t, curand* -> hiprand*, cub -> hipcub).
   Include `<cstring>`/`<cstdlib>` BEFORE the HIP runtime (gpuRIR host-overload
   lesson). Pull cuda->hip names from torch's cuda_to_hip_mappings.py.
2. CMake: `option(USE_HIP "Build with HIP for AMD GPUs" OFF)`; under it
   `enable_language(HIP)`, set `CMAKE_HIP_ARCHITECTURES` from the cache var with
   a gfx90a default ONLY when unset (NEVER a literal -- CudaSift/Gpufit lesson),
   and mark the compute `.cu` LANGUAGE HIP. Provide a top-scope
   add_library/add_executable override (MPPI lesson) so each rasterizer subdir's
   `.cu` is retagged without editing every leaf CMakeLists. Force-include the
   compat header on HIP TUs via `CMAKE_HIP_FLAGS -include .../cuda_to_hip.h`.
3. Provide forwarding shim headers (MPPI lesson) named `cuda.h`,
   `cuda_runtime.h`, `cooperative_groups.h`, `cooperative_groups/reduce.h`,
   `device_launch_parameters.h`, `curand_kernel.h` in a `hip_compat/` dir added
   to the include path ONLY under `if(USE_HIP)`. gsplat's `Common.h` hard-
   includes `<cuda.h>` (line 13) purely to define CUDA_VERSION for GLM -- the
   shim handles that (see GLM risk).
4. Guard genuinely divergent code with `#if defined(USE_HIP)`; keep rare.

This is mechanical/correctness-first. None of these kernels are CUTLASS/CuTe or
Hopper-wgmma tuned (no `wmma`, no `mma_sync`, no `__nv_*` MMA -- verified), so an
AMD-native rewrite is NOT warranted; a HIP translation is the right first pass.

## CUDA surface inventory (non-external)
- 92 `.cu`/`.cuh` files. Kernels across: custom tensor lib (`src/core/tensor/*`,
  ~13 .cu: pointwise/broadcast/reduction/matrix/masking/random/strided/dot, plus
  `tensor_warp_reduce.cu` and `internal/warp_reduce.cuh`), training kernels
  (`src/training/kernels/*`: ssim, l1_loss, mcmc, densification, pruning,
  regularization, bilateral_grid fwd/bwd/tv, grad_alpha, image, mrnf, ppisp,
  camera_loss_heatmap), three rasterizer backends (fastgs fwd/bwd + adam, gsplat
  7 kernels, edge_compute), core cuda kernels (kmeans, kdtree_kmeans, morton,
  sh_layout, selection_ops, lanczos_resize, undistort, memory_arena), io cuda
  (image_format, color_convert, kmeans, morton), rendering (point_cloud_raster,
  selection_ops, cuda_interaction_kernels, cuda_vulkan_interop), one viz
  (vksplat_input_packer_cuda).
- Warp intrinsics: `__shfl_xor_sync` (10 sites) and `__shfl_down_sync` (5 sites),
  ALL with the 32-bit `0xffffffff` mask. Hot in `core/tensor/internal/warp_reduce.cuh`,
  `tensor_warp_reduce.cu`, `mcmc_kernels.cu`, ssim/regularization/densification/l1_loss.
- cooperative_groups: 9 files. `cg::this_grid().thread_rank()` (grid-stride id,
  wave-agnostic, fine). `cg::tiled_partition<32>` + `cg::reduce(warp, v, plus/greater)`
  in gsplat (Utils.cuh warpSum/warpMax, RasterizeBwd:244) and fastgs/edge_compute
  block reductions. `block.thread_rank()` indexing into shared tile buffers.
- Hardcoded warp constants: `const int WARP_SIZE = 32` (mcmc_kernels.cu:631,
  tensor_debug.cu:14), `constexpr size_t WARP_SIZE = 32` (tensor_strided_ops.cu:570),
  `BLOCK_SIZE/32` warps-per-block (tensor_warp_reduce.cu:1191), `threadIdx.x % 32`
  lane id (mcmc).
- Libraries: cuRAND (random tensor ops, mcmc, kmeans) -> hipRAND. CUB+Thrust
  (DeviceRadixSort::SortPairs on int64 keys + DeviceScan::InclusiveSum in gsplat
  IntersectTile; thrust sort/scan/reduce across ~19 files) -> hipCUB/rocThrust.
  NVTX (profiling) -> roctx or no-op. cudaSurfaceObject_t (Vulkan interop write).
  cudaExternalMemory_t/cudaExternalSemaphore_t (CUDA<->Vulkan, viewer only).
  `__half` (ssim, tensor ops, packed128.cuh) -> hip fp16 (drop-in).
  external/nvImageCodec = nvjpeg/nvjpeg2k (NVIDIA-only; no ROCm equivalent).
- Math types in device code: gsplat + mcmc use GLM (`vec3`/`mat3`/`glm::dot`);
  fastgs/edge_compute use CUDA built-in `float3`/`make_float3`.
- No textures except the one Vulkan-interop surface write; no `cudaArrayLayered`;
  no `wmma`/`mma_sync`/CUTLASS/CuTe.

## Risk list (ranked)
1. wave64 raw-shuffle reductions (HIGH, compile + correctness).
   `core/tensor/internal/warp_reduce.cuh` `warp_reduce_{sum,max,min,prod}` do
   `for (offset=16; offset>0; offset/=2) val += __shfl_xor_sync(0xffffffff,...)`.
   Two faults at once (PORTING_GUIDE Fault classes + line 176):
   (a) `0xffffffff` 32-bit mask FAILS TO COMPILE on HIP (ROCm static_asserts a
       64-bit mask). Define a USE_HIP full mask `0xffffffffffffffffULL`.
   (b) offset starting at 16 only reduces 32 lanes -> on a 64-lane CDNA wavefront
       you get TWO independent half-sums, silently wrong wherever the caller
       expects a full-wave reduction. Fix with the warp-size abstraction: start
       offset at `kWarpSize/2` (kWarpSize = 64 on __GFX9__, else 32; PORTING_GUIDE
       warp-size class). Same pattern in mcmc_kernels.cu (WARP_SIZE=32, lane =
       threadIdx.x%32, __shfl_down_sync) and tensor_warp_reduce.cu (BLOCK_SIZE/32).
   Decide per call site whether the reduction is "one value per warp recombined
   by atomicAdd/shared" (native 64-lane is fine, AutoDock line 177) vs "rows
   packed N-per-block positionally" (must keep 32-lane halves, popsift line 141).
   For these tensor reductions the result feeds a block/shared recombine, so the
   native-64 fix (kWarpSize-parameterized) is correct; verify with a fixed-seed
   determinism check (two runs bit-identical) per MPPI line 157.
2. HIP cooperative_groups lacks cg::reduce / <cooperative_groups/reduce.h> (HIGH,
   compile). DIRECTLY CONFIRMED by the completed MOAT `gsplat` port (notes.md:127):
   the SAME `Utils.cuh` warpSum/warpMax + the `cg::reduce(warp,bin_final,greater)`
   sites must be shimmed -- a manual width-32 `__shfl_xor` reduction under USE_ROCM,
   `cg::reduce` on CUDA. GOOD NEWS from that port (notes.md:90-96, 300-306): the
   32-thread tiles are wave-size-agnostic by construction; `thread_block_tile<32>`
   on wave64 makes 32-lane tiles whose member-masked shuffles stay in-tile, so the
   rasterizer warp reductions need NO wave64 lane-math rework once cg::reduce is
   shimmed -- it is purely a HIP-CG API gap, not a correctness bug. (On gfx1100/
   wave32 the tile is exactly one wavefront; also fine.)
3. GLM in __device__ code (MEDIUM-HIGH, compile). gsplat (Utils.cuh, Cameras.cuh,
   RasterizeBwd) and mcmc_kernels.cu call `glm::dot`/use glm vec/mat in device fns.
   GLM emits __host__ __device__ on its math only when it thinks it is nvcc; hipcc
   defines neither __CUDACC__ nor CUDA_VERSION, so every glm:: call from a kernel
   errors "call to __host__ function from __device__ function" (3DUNDERWORLD line
   167). vcpkg pins glm 1.0.3 (>= 0.9.9.9), where `GLM_FORCE_CUDA` alone steers GLM
   into its CUDA qualifier path -- so define `GLM_FORCE_CUDA` in the compat header
   before GLM is parsed (force-included on HIP TUs). Note gsplat's Common.h:13
   includes `<cuda.h>` solely for CUDA_VERSION; the `cuda.h` shim header must define
   a plausible CUDA_VERSION (e.g. 12080) so GLM's version check is satisfied. If
   defining __CUDACC__ is also needed, pair it with `EIGEN_NO_CUDA` only if Eigen
   appears (it does not here) -- so GLM_FORCE_CUDA should suffice; verify with a
   device-side glm::dot probe before broad changes.
4. HIP float3 operator ambiguity (MEDIUM, compile). fastgs/edge_compute use
   `make_float3` and float3 arithmetic. HIP's `float3` is HIP_vector_type with
   member operators that CUDA's plain struct lacks; any hand-rolled
   `operator+/-/*`(float3) collides ("ambiguous"). USE_HIP-guard only the colliding
   overloads, keep helpers HIP does not provide (dot/cross/normalize) -- check each
   individually (MPPI line 160, Fast-Poisson line 182). Grep fastgs/edge_compute
   utils for operator overloads before/after first HIP compile.
5. CUB radix sort + DoubleBuffer (MEDIUM, correctness). gsplat IntersectTile.cu
   uses `cub::DeviceRadixSort::SortPairs` over `cub::DoubleBuffer<int64_t>` keys
   (tile<<32 | depth) and `DeviceScan::InclusiveSum`. hipCUB supports both. It sorts
   the FULL 64-bit key (begin_bit 0), so the cudaKDTree nonzero-begin_bit hipCUB bug
   (line 151) does NOT apply -- but confirm no begin_bit/end_bit is passed and that
   the DoubleBuffer selector ends up reading the correct (sorted) buffer on HIP.
6. cuRAND -> hipRAND (MEDIUM). Device curand_kernel.h (states, uniform/normal) used
   in mcmc/kmeans/tensor random. hipRAND mirrors the API; watch generator-state
   types and per-thread seeding. test_curand_buffer_overflow + test_tensor_random*
   gate it. Negative/robustness tests may differ (MPPI line 164) -- judge by parity
   to the libtorch oracle, not by exact RNG bitstream.
7. CUDA<->Vulkan external-memory interop (HIGH but VIEWER-ONLY -> deferred).
   cuda_vulkan_interop.cpp imports VkDeviceMemory via cudaImportExternalMemory +
   cudaExternalMemoryGetMappedMipmappedArray and a timeline semaphore via
   cudaImportExternalSemaphore (OpaqueFd on Linux), then a kernel writes a
   cudaSurfaceObject_t. HIP has hipImportExternalMemory/Semaphore and Vulkan-HIP
   interop, but mipmapped-array import + timeline-semaphore interop on ROCm is a
   real unknown and is the GUI preview path, not training. Out of scope for the
   lead compute port (see Scope). The hpp holds raw `cudaExternalMemory_t`/
   `cudaExternalSemaphore_t` members -> when it IS ported, apply rule-of-five RAII
   (explicit =0 init, move-only, guarded destroy) since AMD faults on double/zero
   destroy (colmap CuTexObj class).
8. nvImageCodec / nvjpeg / nvjpeg2k (MEDIUM, dependency -> deferred). Vendored
   NVIDIA GPU codec, no ROCm equivalent. Gate `external/nvImageCodec` and the
   io GPU-decode path behind `if(NOT USE_HIP)`; the io layer must fall back to the
   CPU OpenImageIO/stb decode path on ROCm. Belongs to the deferred io/GUI tranche.
9. NVTX profiling ranges (LOW). Map to roctx or no-op under USE_HIP. nvToolsExt
   target creation in CMake is NVIDIA-only; skip under USE_HIP.
10. __half / fp16 (LOW). hip fp16 is a drop-in (hip_fp16.h via the runtime); just
    ensure the header is reachable through the compat header.
11. CUB_WRAPPER_LFS uses raw cudaMalloc/cudaFree per call (Common.h:70) -- works on
    HIP via aliases; just a perf note, not a fault.

## File-by-file change list (lead compute tranche)
- NEW src/core/include/core/cuda/cuda_to_hip.h -- compat header (the only file that
  knows HIP): hip runtime include, cuda*->hip* aliases, 64-bit full-warp mask,
  kWarpSize constant (__GFX9__ -> 64 else 32), GLM_FORCE_CUDA, cub->hipcub.
- NEW src/hip_compat/{cuda.h,cuda_runtime.h,cooperative_groups.h,
  cooperative_groups/reduce.h,device_launch_parameters.h,curand_kernel.h} --
  forwarding shims, on HIP include path only; cuda.h defines CUDA_VERSION; the
  cooperative_groups/reduce shim provides the cg::reduce replacement entry point.
- EDIT CMakeLists.txt -- add option(USE_HIP); under it enable_language(HIP), set
  CMAKE_HIP_ARCHITECTURES (cache-var, gfx90a default when unset), top-scope
  add_library/add_executable override to retag .cu LANGUAGE HIP + set
  HIP_ARCHITECTURES, CMAKE_HIP_FLAGS `-include cuda_to_hip.h` + hip_compat include
  dir; gate add_subdirectory(src/visualizer|mcp|python) and external/nvImageCodec
  and the nvToolsExt block behind `if(NOT USE_HIP)`; replace nvidia-smi arch detect
  with the HIP path when USE_HIP. Keep all CUDA branches untouched.
- EDIT src/core/tensor/internal/warp_reduce.cuh -- kWarpSize-parameterized reductions
  + 64-bit mask (risk 1).
- EDIT src/core/tensor/tensor_warp_reduce.cu, tensor_strided_ops.cu, core/cuda/
  tensor_debug.cu -- replace hardcoded 32/BLOCK_SIZE/32 with kWarpSize / runtime
  warpSize (risk 1). Host-side launch sizing uses prop.warpSize.
- EDIT src/training/kernels/mcmc_kernels.cu -- WARP_SIZE/lane/__shfl_down_sync to
  the abstraction (risk 1) + glm device path via compat (risk 3).
- EDIT src/training/rasterization/gsplat/Utils.cuh (+ Common.h cuda.h include) --
  USE_ROCM cg::reduce shim for warpSum/warpMax (risk 2), GLM via GLM_FORCE_CUDA
  (risk 3). Mirror gsplat MOAT port's Utils.cuh fix.
- EDIT gsplat RasterizeToPixelsFromWorld3DGSBwd.cu / IntersectTile.cu -- cg::reduce
  sites through the shim; confirm CUB DoubleBuffer/RadixSort under hipCUB (risk 5).
- EDIT fastgs + edge_compute kernel_utils.cuh / kernels_*.cuh -- float3 operator
  guards if any collide (risk 4); cg block-reduction shim if they use cg::reduce.
- EDIT any NVTX usage -> roctx/no-op (risk 9).
- DEFER (separate tranche, documented): src/rendering/cuda_vulkan_interop.*,
  src/visualizer/*, src/mcp/*, src/python/*, src/io GPU-decode + external/nvImageCodec.
Expect the porter to discover a few more hardcoded-32 / cg::reduce / float3 sites
on first compile; the macro-expansion grep caveat (cudaKDTree line 155) applies if
any cuda call is hidden behind a paste macro (CUB_WRAPPER_LFS is the only one seen).

## Build commands (gfx90a)
Lead compute bringup (no GUI/USD/nvjpeg), ROCm 7.2.1, hipcc at /opt/rocm:
```
# Compute-only HIP configure (USE_HIP gates off GUI/USD/python/nvjpeg subtrees)
cmake -S projects/LichtFeld-Studio/src -B projects/LichtFeld-Studio/src/build-hip \
  -G Ninja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DBUILD_TESTS=ON \
  -DLFS_ENFORCE_LINUX_GUI_BACKENDS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build projects/LichtFeld-Studio/src/build-hip -j"$(nproc)"
```
Notes: pass `-DCMAKE_HIP_ARCHITECTURES=gfx1100` (or gfx1151) unchanged for
followers -- no source edit (CudaSift/Gpufit lesson). The test target needs a
ROCm libtorch; point `Torch_DIR` at the locally built ROCm torch
(`.../torch/share/cmake/Torch`) rather than the project's `external/libtorch`
download (which is a CUDA build). The vcpkg toolchain is still required for gtest/
glm even in the compute tranche -- if the full vcpkg bootstrap is too heavy, supply
gtest+glm via apt/conda and bypass vcpkg by setting CMAKE_TOOLCHAIN_FILE empty.
A CPU-only ROCm docker compile check (image rocm/dev-ubuntu-24.04:7.2.4-complete)
is a fine smoke for "does it compile", never a validation gate.

## Test plan
Real GPU tests (the validator's bar). The suite builds gtest cases into a single
`lichtfeld_tests` executable; `gtest_discover_tests` registers each case with
ctest (tests/CMakeLists.txt:479). ~179 test_*.cpp total; ~93 are GPU-compute and
are the MOAT gate, ~45 are GUI/MCP/Python (out of scope for the compute tranche).
GPU-compute tests to pass (representative, all in the compute libs' surface):
- Rasterizers: test_gsplat_rasterizer, test_fastgs_kernels,
  test_fastgs_analytical_gradients, test_fastgs_fuzz, test_gut_* (rasterizer
  gradients / manual-vs-autograd), test_rotated_sh_correctness, test_sh_swizzle_layout.
- Losses/SSIM: test_fused_l1_ssim, test_mask_loss, test_loss_gradients,
  test_activation_gradients, test_gradient_accumulation.
- MCMC/strategy: test_mcmc, test_mcmc_* (tensor_ops, memory_ops, nan_fix,
  dead_mask, logit_verification, relocate_optimizer_state_bug), test_mrnf_strategy,
  test_relocate_gs_edge_cases, test_densification_tensor_ops.
- Tensor library (the wave64-sensitive core): test_tensor_reduction,
  test_tensor_reduction_alignment, test_tensor_warp* / warp-reduce paths,
  test_tensor_random / test_tensor_random_advanced, test_tensor_math,
  test_tensor_matrix, test_tensor_ops, test_tensor_vs_torch, test_torch_comparisons,
  test_nan_inf_gpu_check, test_curand_buffer_overflow, test_sort_minimal.
- ppisp: test_ppisp_cuda_vs_torch, test_ppisp_regularization.
Run SERIALLY on the one assigned GPU (`ctest` not `ctest -jN`; MPPI line 165 --
parallel processes hammering one GPU cause false-failure flapping). Use a ctest
`-R` filter to the compute set; libtorch parity tests prove correctness against
PyTorch as the oracle.
Headless training smoke (exercises the whole rasterize+optimize pipeline on GPU):
```
./build-hip/LichtFeld-Studio -d <colmap_scene> -o /tmp/lfs_out \
   --headless --iterations 200    # confirm exact flags via --help
```
The repo ships NO dataset (`data/` absent), so the validator fetches a small
COLMAP scene (e.g. the standard tandt `truck` or a Mip-NeRF360 room). Success
criteria: trains N iterations without NaN/Inf (test_nan_inf_gpu_check logic),
loss decreases monotonically-ish, and a fixed-seed run is reproducible to ULP
(determinism check per MPPI line 157 -- two runs bit-identical or within the
ASCII-export precision bar per 3DUNDERWORLD line 169). If the headless path drags
in the GUI/io-nvjpeg graph, gate it or rely on the gtest compute set + a tiny
standalone harness that calls the rasterizer forward/backward and Adam directly.
Non-GPU regression set NOT to break: the CPU tensor tests
(test_cpu_dtype_conversions, test_cpu_large_tensor_bugs, test_tensor_serialization,
argument parser, logger, scheduler) -- these must still pass under the HIP build.

## Inter-project MOAT deps
None. `depends_on = []` (set via `moatlib.py set-deps LichtFeld-Studio`). LichtFeld
vendors its OWN gsplat (gsplat_backend_lfs) and SSIM kernels in-tree; it does not
build against the MOAT `gsplat`/`fused-ssim`/`gaussian_splatting` forks. Those
sibling ports are PyTorch-extension ports (Strategy B); LichtFeld's copies are
standalone libtorch-free CUDA. The shared asset is the LESSONS, already in
PORTING_GUIDE -- most directly the completed MOAT `gsplat` port's analysis of the
identical Utils.cuh `cg::reduce`/`tiled_partition<32>` pattern (its notes.md
lines 78-133, 300-306), which says the 32-lane tiles are wave-size-agnostic and the
only required change is the cg::reduce HIP-API shim. libtorch (ROCm) is a TEST
oracle, supplied by the validator's environment, not a MOAT-tracked dep.

## Open questions
1. Scope confirmation with jeff: is the lead deliverable the GPU compute tranche
   (tensor lib + 3DGS rasterizers + training/losses/MCMC) with the GUI/Vulkan-
   interop, Python/MCP, USD, and nvjpeg layers explicitly deferred? The plan
   assumes yes (a full-app ROCm port is out of proportion to the moat). If the
   whole app must run on ROCm, that is a much larger, multi-stage effort and the
   CUDA<->Vulkan external-memory interop (risk 7) needs its own spike.
2. Can `-DUSE_HIP=ON` cleanly gate off src/visualizer|mcp|python + external/
   nvImageCodec while still building the compute libs and a compute test binary?
   If the monolithic `lichtfeld_tests` target cannot be decoupled from
   lfs_visualizer/lfs_mcp without source surgery, add a dedicated compute-test
   executable (build-file only) per the Scope fallback.
3. ROCm libtorch availability/version for the parity tests: the project's
   `external/libtorch` is a CUDA build; the validator must point Torch_DIR at a
   ROCm torch. Confirm the project's tensor-vs-torch tests don't assume CUDA-only
   torch APIs.
4. Does the headless training entrypoint pull the io GPU-decode (nvjpeg) path for
   image loading? If so, ensure the CPU OpenImageIO/stb fallback is selected under
   USE_HIP so the smoke test does not require nvjpeg.
5. CUB DoubleBuffer selector correctness on hipCUB for the gsplat tile sort
   (risk 5) -- verify the sorted buffer is the one read after SortPairs.
