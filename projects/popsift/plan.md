# PopSift port plan (Strategy A, gfx90a lead)

## Project
- Upstream: alicevision/popsift, branch `develop` (commit b1c8199), https://github.com/alicevision/popsift
- A standalone CUDA implementation of SIFT (Lowe) feature extraction. Computes a
  Gaussian scale-space pyramid, DoG, extrema detection + refinement, orientation
  assignment, and descriptor extraction entirely on the GPU. Sibling of the
  already-ported CudaSift; same domain (textures, warp shuffles, descriptor
  reductions) but a larger and more modern CUDA surface.
- The build produces a library `libpopsift` plus two demo apps (`popsift-demo`,
  `popsift-match`). The library is the port deliverable; the demos are thin
  host-only CLI wrappers.

## Existing AMD support
None. `grep -ri hip/rocm/__GFX/amdgpu/wavefront` over all sources, CMake, and
`.cmake.in` templates returns nothing. AMD is not supported by any means (no
OpenCL/Vulkan/SYCL path either). Decision: genuine port (no HIP path exists).
Closely-related prior art reused as the template: the CudaSift port
(`projects/CudaSift`) and colmap's `rocm-sift-gpu`.

## Build classification + strategy
Pure CMake CUDA project -> **Strategy A** (colmap/CudaSift model, minimal footprint).
Evidence: top `CMakeLists.txt` `project(PopSift ... LANGUAGES CXX CUDA)`,
`set(CMAKE_CUDA_ARCHITECTURES ...)`, `find_package(CUDAToolkit)`, links
`CUDA::cudart`. No `find_package(Torch)`, no `torch.utils.cpp_extension`. Not a
pytorch extension.

Plan:
1. One compat header `src/popsift/cuda_to_hip.h`: on AMD includes `<hip/hip_runtime.h>`
   and `#define`s every CUDA symbol popsift uses to its HIP equivalent; on NVIDIA a
   no-op `#include <cuda_runtime.h>`. Adapted from CudaSift's header, with the extra
   symbols popsift needs (surfaces, layered/3D arrays, memcpy3D, async variants).
2. `hip_compat/` dir with `cuda_runtime.h` (and `cuda.h`) shims that redirect to
   `cuda_to_hip.h`. 12 source/header files `#include <cuda_runtime.h>` directly
   (including the public `popsift.h`); on ROCm that header does not exist, so the
   shim dir is put on the include path for the HIP build only.
3. CMake: `option(USE_HIP ...)`. When ON: `enable_language(HIP)`, set every `.cu`
   to `LANGUAGE HIP`, `HIP_ARCHITECTURES gfx90a`, force-include the compat header
   and add `hip_compat/` to the include path, link rocThrust. When OFF: the
   existing CUDA path is byte-for-byte unchanged.
4. rocThrust: `s_filtergrid.cu` uses Thrust (`thrust::cuda::par.on(stream)`, sort_by_key,
   transform, device_vector). ROCm ships rocThrust at `/opt/rocm/include/thrust`
   with a `thrust::cuda::par` compatibility namespace, so the Thrust code compiles
   unchanged; only need `find_package(rocthrust)` + link on the HIP path.

Authoritative cuda->hip name source: `torch/utils/hipify/cuda_to_hip_mappings.py`.

## CUDA surface inventory
- **Textures (read path).** 61x `cudaTextureObject_t`. Per octave: point + linear
  textures over the blur-data and intermediate layered arrays, plus a point texture
  over the DoG array (`sift_octave.cu` alloc_*_tex). Sampled with `tex2DLayered<float>`
  (pyramid build, gradient, descriptors via `readTex` in `assist.h`/`s_gradiant.h`)
  and `tex2D<float>`. Address mode clamp, point + linear filter. Backed by 3D layered
  cudaArrays.
- **Surfaces (write path) -- NEW vs CudaSift.** 35x `cudaSurfaceObject_t`, 17x
  `surf2DLayeredwrite` (pyramid build writes blurred levels and DoG into the layered
  arrays; `s_pyramid_build*.cu`, byte x-offset `write_x*4`, `cudaBoundaryModeZero`).
  `cudaCreateSurfaceObject`/`cudaDestroySurfaceObject`.
- **Layered / 3D arrays -- NEW.** `cudaMalloc3DArray` with
  `cudaArrayLayered | cudaArraySurfaceLoadStore`; `cudaChannelFormatKindFloat`;
  `cudaMemcpy3D` + `cudaMemcpy3DParms` + `make_cudaPitchedPtr`/`make_cudaExtent`
  for debug download (`sift_octave.cu`). `cudaExtent` members in `sift_octave.h`.
- **Pitched memory.** `cudaMallocPitch` (`plane_2d.cu`), `cudaMemcpy2D(Async)`.
- **Constant memory.** `__device__ __constant__` `d_consts` (`sift_constants.cu`),
  `d_gauss` (`gauss_filter.cu`), `dct`/`dbuf`/`dobuf` (`sift_pyramid.cu`) via
  `cudaMemcpyToSymbol(Async)` / `cudaMemcpyFromSymbol(Async)`.
- **Warp intrinsics.** `assist.h` wraps `__shfl{,_up,_down,_xor}_sync(0xffffffff,...)`,
  `__ballot_sync`, `__any_sync`, `__all_sync` (1- and 3-arg width forms), guarded by
  `POPSIFT_HAVE_SHFL_DOWN_SYNC`. `warp_bitonic_sort.h` `Warp32` does a 32-lane bitonic
  sort (`shuffle_xor` with `1<<shift` masks). `__popc` on a ballot result.
- **Thrust.** `s_filtergrid.cu` only (grid filter). No CUB, no cuBLAS/cuFFT/cuRAND/
  cuSPARSE, no nvtx/nvToolsExt.
- **Runtime API.** streams (per-octave + default-legacy), events, `cudaMallocHost`/
  `cudaFreeHost`, `cudaHostRegister/Unregister`, `cudaMemcpyAsync`, `cudaMemsetAsync`,
  `cudaStreamWaitEvent`, `cudaSetDevice`, `cudaGetDeviceProperties`, `cudaDeviceReset`.

## Risk list (fault classes)
- **Warp size 64 vs 32 (highest risk).** `s_orientation.cu` is hardwired to a 32-lane
  warp: launched with `block.x = 32`; histogram is 64 bins handled as
  `threadIdx.x+0` / `threadIdx.x+32`; `best_index = (threadIdx.x, threadIdx.x+32)`
  then `Warp32::sort64`; `__popc(ballot(...))` to count orientations. On gfx90a
  (wave64) a `block.x=32` launch fills only half a wavefront. The bitonic
  `shuffle_xor` masks are `1<<shift`, shift 0..4 (max 16), so they stay within the
  low 32 lanes and the network is self-consistent; `ballot` returns a 64-bit mask but
  inactive upper lanes are 0 so `__popc` is unchanged. This *probably* preserves the
  32-lane semantics, exactly like the CudaSift `width=32` case, but it MUST be
  validated on-device (orientation/descriptor numeric parity). If wrong, fix by
  abstracting the warp width (the `Warp32` log2(32)=5 loop counts and the `+32`
  histogram split would need a 64-aware variant). The `+32` / `1<<shift` constants
  are algorithmic (bins, bitonic stride), not warpSize -- do not blanket-replace.
- **`__ballot`/`__any`/`__all` mask width.** `assist.h` passes the CUDA 32-bit full
  mask `0xffffffff` to `*_sync`. HIP's `__*_sync` ignore the mask value (poll the
  active wavefront), so behavior is correct on wave64; the 64-bit return of `ballot`
  is consumed only by `__popc`, which is width-agnostic. Handle by mapping
  `__shfl*_sync`/`__ballot_sync`/`__any_sync`/`__all_sync` to the mask-free HIP
  builtins in the compat header (like CudaSift's `__any_sync` alias).
- **Texture rule-of-five.** Texture/surface handles in `sift_octave.cu` are class
  members, `{}`-initialized, created in `alloc_*_tex`, destroyed once in
  `free_*_tex`; arrays freed once in `free_*_array`. No default-constructed-then-
  destroyed or double-destroy path observed (Octave owns them for its lifetime).
  Low risk, but verify the resize path (`resetDimensions`) frees before re-alloc
  (it does: free_* then alloc_*).
- **Out-of-bounds reads.** Texture fetches at rotated offsets (`x +- cos/sin`) and
  pyramid neighbor reads (`off_x +- span`) rely on `cudaAddressModeClamp`, which the
  sampler honors on AMD too (unlike colmap's *linear-memory* reads). Lower risk than
  CudaSift's linear path, but confirm no raw global +-1 edge read in `s_extrema.cu`
  / DoG without clamp (the DoG is sampled via a clamped texture, so OK).
- **Texture pitch alignment (256B).** popsift textures are **cudaArray-backed**
  (`cudaMalloc3DArray`), not `cudaResourceTypePitch2D` binds, so the 256-byte row
  pitch rule does not apply (the array manages its own layout). The lone
  `cudaMallocPitch` (`plane_2d`) is plain device memory, never bound as a pitched
  texture. Per the CudaSift changelog lesson: confirmed the resource type is Array,
  not Pitch2D -- not subject to the fault class.
- **Deprecated array-copy APIs.** None. popsift already uses the modern
  `cudaMalloc3DArray` + `cudaMemcpy3D` + surface objects (no `cudaMemcpyToArray`),
  all of which have direct HIP equivalents. (CudaSift needed the
  `cudaMemcpyToArray -> hipMemcpy2DToArray` rewrite; popsift does not.)
- **Thrust execution policy.** `thrust::cuda::par.on(stream)` -- rocThrust provides
  the `thrust::cuda` compat namespace, so it routes to HIP unchanged. Risk: rocThrust
  version-specific API drift; resolve at first compile of `s_filtergrid.cu`.
- **Dependency: Boost (demos only).** `src/application/CMakeLists.txt` does
  `find_package(Boost 1.71.0 REQUIRED COMPONENTS filesystem program_options system)`
  and optional DevIL. Boost is NOT installed on this host. The **library** does not
  depend on Boost/DevIL, so the port builds with `PopSift_BUILD_EXAMPLES=OFF`. The
  demos are blocked-on-dep (Boost) but are not part of the GPU port surface.

## File-by-file change list
- `src/popsift/cuda_to_hip.h` (new): the alias header (adapt CudaSift's; add surface,
  layered/3D array, memcpy3D, async, host-register, from-symbol symbols).
- `src/popsift/hip_compat/cuda_runtime.h`, `.../cuda.h` (new): redirect `<cuda_runtime.h>`
  / `<cuda.h>` to the compat header; on the HIP include path only.
- `CMakeLists.txt` (top): add `option(USE_HIP)`; gate `project(... CUDA)` vs `... HIP)`;
  skip `find_package(CUDAToolkit)` / CUDA-only flags on the HIP path. Keep CUDA path intact.
- `src/CMakeLists.txt`: under USE_HIP, set the `.cu` sources to `LANGUAGE HIP`,
  set `HIP_ARCHITECTURES gfx90a`, force-include `cuda_to_hip.h`, add `hip_compat/`
  to includes, link `roc::rocthrust` instead of (or alongside dropping) `CUDA::cudart`.
- `src/application/CMakeLists.txt`: only built when `PopSift_BUILD_EXAMPLES=ON`
  (left OFF on AMD until Boost is present); no porting changes needed (host-only).
- Kernel sources: **no source edits expected** beyond the force-included header. If
  on-device validation shows the wave64 orientation/bitonic path diverges, a guarded
  `Warp32`/orientation fix in `warp_bitonic_sort.h` + `s_orientation.cu` follows.

## Build commands (gfx90a)
Library-only (no Boost needed), out-of-source:
```
utils/timeit.sh popsift compile -- cmake -S projects/popsift/src -B projects/popsift/src/build-hip \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release \
  -DPopSift_BUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON
cmake --build projects/popsift/src/build-hip -j
```
Build dir `build-hip/` is added to `.git/info/exclude` in the clone.

## Test plan (real GPU, gfx90a)
- Build `libpopsift` clean (this run's gate).
- GPU gate (validator, separate MOAT stage): once Boost is installed, build
  `popsift-demo`/`popsift-match` and run SIFT on `testImages/` (and an Oxford-dataset
  image); confirm keypoint counts, orientations, and descriptors match the CUDA-path
  expectation -- this is where the wave64 orientation/bitonic risk is actually
  observed (a CPU/compile-only build cannot see it).
- BC: confirm the NVIDIA build still configures with `USE_HIP=OFF` (unchanged path).

## Open questions
- Boost 1.71+ on the gfx90a host (blocks the demo apps only; library builds without it).
- rocThrust API parity for the `s_filtergrid.cu` sort/transform calls (determine at
  first compile).
