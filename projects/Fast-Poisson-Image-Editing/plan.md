# Fast-Poisson-Image-Editing -- ROCm/HIP port plan (linux-gfx90a lead)

## What the project is

fpie (Fast Poisson Image Editing) is a parallel Poisson image-editing / seamless-cloning
solver. It blends a source patch into a target image by solving a Poisson equation over the
masked region with Jacobi iteration. It ships many interchangeable backends behind one Python
API (`fpie.process.EquProcessor` / `GridProcessor`): numpy, numba, taichi, gcc, OpenMP, MPI,
and CUDA. Two algorithms: `equ` (sparse equation form, neighbors gathered through an index
array `A`) and `grid` (dense grid, 4-neighbor stencil).

## Existing AMD support assessment

None. The CUDA backend is NVIDIA-only: `find_package(CUDA)` + `cudart`, headers `<cuda.h>`,
`<cuda_runtime.h>`, `<driver_functions.h>` (fpie/core/cuda/CMakeLists.txt:1, utils.h:4-6).
No OpenCL/Vulkan/SYCL/HIP path exists; CI has no GPU job (.github/workflows/{lint,test}.yml).
This is a genuine, valuable fresh CUDA-to-HIP port. Disposition: PORT (not skip).

## Build classification -> Strategy A (pure CMake)

The build is a pybind11 CMake project, NOT a pytorch extension (no torch, no
`torch.utils.cpp_extension`, no `CUDAExtension`). setup.py drives a plain `cmake && make`
(setup.py:55-84, CMakeBuild). The CUDA backend is one pybind11 module `core_cuda` built from
`solver.cc equ.cu grid.cu utils.cu` and linked against `cudart`
(fpie/core/cuda/CMakeLists.txt:13-15). Per PORTING_GUIDE "Build classification", a pure-CMake
`.cu` project uses **Strategy A**: one compat header + mark the `.cu` `LANGUAGE HIP`, CUDA
path untouched, all HIP behind `USE_HIP`.

## CUDA surface (small, fully enumerated)

Runtime: cudaMalloc, cudaFree, cudaMemcpy, cudaMemset, cudaMemcpyHostToDevice,
cudaMemcpyDeviceToHost, cudaDeviceSynchronize, cudaGetDeviceCount, cudaGetDeviceProperties,
cudaDeviceProp, cudaError_t. Headers: <cuda.h>, <cuda_runtime.h>, <driver_functions.h>.
Vector types int4/float3 + make_int4/make_float3 (built into HIP, no alias needed; verified by
a standalone hipcc gfx90a compile+run). Kernel model: `__global__`, `__host__ __device__`,
blockIdx/threadIdx/blockDim/gridDim, __syncthreads, `<<<>>>`. No warp primitives, no textures,
no surfaces, no cuBLAS/cuFFT/cuRAND/Thrust/CUB. This is the cleanest possible Strategy A
surface.

## Fault-class analysis (PORTING_GUIDE)

- Warp size / wave64: NOT triggered. The error reduction does NOT use any warp primitive
  (`__shfl`/`__ballot`/`__activemask`) or assume 32 lanes. `error_sum_equ_kernel` (equ.cu:162)
  reduces via `__shared__` then a single-thread (id==0) serial loop; `error_grid_kernel`
  (grid.cu:73) reduces per-block by thread (0,0) and globally by block (0,0), both serial. No
  `kWarpSize` constant is needed. (Will still re-check determinism on wave64.)
- Out-of-bounds stencil reads: the real risk in grid.cu (`X - m3`, `X - 3`, `X + 3`, `X + m3`
  at iter_grid_kernel:64-67 / error_grid_kernel:83-85) is ALREADY guarded by the Python layer,
  which pads a >=1px inactive (mask==0) border on every side: `mask[0]=mask[-1]=mask[:,0]=
  mask[:,-1]=0` then crops `x0=x.min()-1 .. x1=x.max()+2` (process.py:242-252 equ, 371-379
  grid). The kernels only write where `mask[id]` is set (grid.cu:60,79), and every active pixel
  is >=1px inside the crop, so all 4 neighbors are in-bounds. equ.cu gathers neighbors through
  index `A` and guards id==0 with `if(id.x)`; slot 0 is allocated, so in-bounds. No clamp
  needed -- confirmed by reading the host code, not assumed.
- driver_functions.h: not present in ROCm (only provides pitched-ptr helpers, unused here).
  Must guard that include so the HIP build does not fail on a missing header.
- Rule-of-five handles, texture pitch/filtering, layered coherency, library swaps, atomics:
  none apply (no textures, no atomics, no libraries).

## Port design (minimal footprint, CUDA path unchanged, USE_HIP-guarded)

1. New `fpie/core/cuda/cuda_to_hip.h`: on `USE_HIP || __HIP_PLATFORM_AMD__`, include
   <hip/hip_runtime.h> and `#define` the ~11 cudaXxx symbols to their hip spellings; else
   include the CUDA runtime headers (<cuda.h>, <cuda_runtime.h>, <driver_functions.h>). This is
   the only file that knows about HIP. utils.h includes it instead of the raw CUDA headers.
2. `fpie/core/cuda/CMakeLists.txt`: add `option(USE_HIP ... OFF)`. USE_HIP branch:
   `enable_language(HIP)`, default `CMAKE_HIP_ARCHITECTURES` to gfx90a only when unset (never a
   literal that overrides the cache var -> avoids follower churn), mark equ.cu/grid.cu/utils.cu
   `LANGUAGE HIP`, link nothing extra (HIP runtime auto-links). else branch: the existing
   `find_package(CUDA)` + `enable_language(CUDA)` + `cudart`. Pass `-DUSE_HIP` compile def so
   the header switches.
3. Top-level `CMakeLists.txt`: `find_package(CUDA)` fails on a ROCm host, so the cuda subdir is
   never added. Gate it: if `USE_HIP`, `add_subdirectory(fpie/core/cuda)` directly; else keep
   the `find_package(CUDA)` guard. Keep USE_HIP defaulted OFF so a stock CUDA build is byte-for-
   byte unchanged.
4. Build via setup.py path with `-DUSE_HIP=ON` threaded through (CMAKE_ARGS / a small env hook),
   or configure the CMake directly for validation.

## Build + validation (GPU 1, HIP_VISIBLE_DEVICES=1)

- Build the `core_cuda` HIP module for gfx90a; keep build dirs out of git (agent_space/).
- VALIDATE FOR REAL on GPU 1: the CUDA(HIP) backend exposes the same Python API as numpy, so
  blend a real sample image pair (tests/data.py downloads them) through both `equ` and `grid`
  with the `cuda` backend and compare the output image against the `numpy` backend
  (reference) within tolerance, plus a determinism check (two cuda runs identical). Both
  backends run the identical Jacobi recurrence, so they must agree to within float round-off.
  Also keep the numpy/non-GPU smoke tests green (tests/test_smoke.py). Compile is not
  validation.

## Followers

gfx1100/gfx1151 are RDNA (wave32). Because no warp primitive is used and the arch is a build-
time `-DCMAKE_HIP_ARCHITECTURES=<arch>` with no source change, followers should validate on the
same commit with no delta port. The wave64-vs-32 axis is moot here (serial reductions).
