# TIGRE port plan (linux-gfx90a lead)

## Project
- Name: TIGRE
- Upstream: https://github.com/CERN/TIGRE
- Default branch: main
- Domain: CERN/University of Bath tomographic (CT/CBCT) iterative reconstruction toolbox. MATLAB + Python frontends over a CUDA backend (forward projection / back-projection, FDK, iterative TV/PICCS solvers).

## DISPOSITION
- **Clean from-scratch HIP port, Strategy A mechanics inside a setuptools/Cython build (hand-rolled nvcc driver, NOT torch).** Effort class: **medium**. Tractable. Set platform `planned`; dispatch a porter.
- The single material risk is the texture path (3D-array `cudaFilterModeLinear` over float, a known gfx90a rejection class); everything else is a near-mechanical cuda->hip rename. No CUTLASS, no cuFFT in the GPU path, no library reimplementation required.
- Scope: Python/CUDA path only. The MATLAB mex path (CompileCUDA.m, pCT `improvedForwardProjections*.cu`) is OUT OF SCOPE for GPU validation.

## Existing AMD support
- **None.** Confirmed by: WebSearch ("TIGRE ROCm", "TIGRE AMD GPU HIP", "TIGRE MI300/gfx9", "CERN TIGRE fork ROCm/OpenCL/Vulkan") returned only CUDA/MATLAB material; `gh api repos/CERN/TIGRE/forks --paginate` shows no fork under ROCm/AMD/GPUOpen or with rocm/hip/amd in the name; upstream topic is literally `cuda`; no rocm/hip branch. No separate ROCm-DS-style project. No OpenCL/Vulkan/SYCL backend either.
- Judgment: genuine from-scratch port, not a duplicate. No authoritative or community AMD effort to inherit or improve.

## Build classification: setuptools/Cython with a hand-rolled nvcc driver (Strategy A mechanics)
Evidence (setup.py):
- `from Cython.Distutils import build_ext` (line 11); `cmdclass={"build_ext": BuildExtension}` (521). NOT `torch.utils.cpp_extension`; no torch dependency anywhere (pyproject.toml deps: matplotlib/numpy/scipy/h5py/tqdm).
- `BuildExtension.build_extensions` monkey-patches `self.compiler._compile` (`unix_wrap_compile`, lines 200-228) to swap the compiler to `nvcc` for `.cu`/`.cuh` and inject `COMMON_NVCC_FLAGS` + `-gencode` (COMPUTE_CAPABILITY_ARGS, 32-36, 144-149). This is the project's OWN nvcc invocation, copied from an old pytorch cpp_extension, not pytorch's.
- `locate_cuda()` (66-112) + `get_cuda_version`/`get_cuda_cc` shell out to `nvcc` at import time; each `Extension` links `libraries=["cudart"]`, `include_dirs=[CUDA["include"], "Common/CUDA/"]`.
- So this is "Strategy A" in spirit (only `.cu` TUs see the GPU compiler; host `.cpp`/Cython untouched) but the language gate lives in setup.py, not CMake. The HIP path is a `BUILD_WITH_HIP`/`hipcc`-driver branch added to setup.py, plus a single compat header for the `.cu` sources.

Set `ext_type = "setuptools-cuda"` (a hand-rolled-nvcc setuptools build; closest to Strategy A, explicitly NOT Strategy B torch).

## Port strategy: Strategy A compat-header + setup.py HIP driver branch
1. Add `Common/CUDA/cuda_to_hip.h` (compat header). On `USE_HIP`/`__HIP_PLATFORM_AMD__`: `#include <hip/hip_runtime.h>` and alias the exact cuda* symbols the Python-built TUs use (histogram below). Include `<cstring>/<cstdlib>` BEFORE `<hip/hip_runtime.h>` (gpuRIR lesson: host memcpy/memset can otherwise resolve to HIP __device__ overloads inside a .cu compiled as HIP). On NVIDIA the header is a passthrough to `<cuda_runtime.h>`. This is the only file that names HIP. gpuUtils.cu explicitly includes `<cuda.h>` and `<cuda_runtime_api.h>` -- route those through the header too.
2. setup.py: add a `BUILD_WITH_HIP` branch (env-gated, mirroring the existing `BUILD_WITH_CUDA` cibuildwheel hook). When set:
   - skip `locate_cuda()` (it shells out to nvcc and will throw on a ROCm-only box) and the `-gencode` discovery; locate ROCm via `ROCM_PATH` (default `/opt/rocm`).
   - swap the per-`.cu` compiler to `hipcc` (or `/opt/rocm/llvm/bin/clang++ -x hip`), replace `COMMON_NVCC_FLAGS`/`-gencode` with `--offload-arch=$HIP_ARCH` (default gfx90a; comma-list for multi-arch), `-fPIC`, `-D USE_HIP`. Do NOT hardcode a single arch in a way a follower must edit -- read `HIP_ARCH`/`HCC_AMDGPU_TARGET` env with a gfx90a default.
   - link `amdhip64` instead of `cudart`; `library_dirs=[$ROCM_PATH/lib]`, `include_dirs += [$ROCM_PATH/include]`.
   - keep the NVIDIA path byte-identical when `BUILD_WITH_HIP` unset.
3. Guard genuinely divergent device code with `#if defined(USE_HIP)`; keep guards rare. The texture-creation fix (below) is the main guarded region.

## CUDA surface inventory (Python-built TUs only)
Built `.cu` (from setup.py ext_modules): Siddon_projection, Siddon_projection_parallel, ray_interpolated_projection, ray_interpolated_projection_parallel, voxel_backprojection, voxel_backprojection2, voxel_backprojection_parallel, tv_proximal, GD_TV, GD_AwTV, PICCS, gpuUtils, RandomNumberGenerator (+ host TIGRE_common.cpp, projection.cpp, GpuIds.cpp).

- Kernels: `__global__`/`__device__` forward-projection (Siddon ray-driven; ray-interpolated) and voxel back-projection (FDK + matched), plus TV/AwTV/PICCS/tv_proximal gradient-descent solvers and a curand Poisson/Gaussian noise generator. All standard.
- Runtime API histogram (1:1 hip* equivalents): cudaSetDevice(139), cudaMemcpyAsync(53), cudaMalloc(44), cudaDeviceSynchronize(44), cudaStream*(42+), cudaTextureObject_t(40), cudaArray(40), cudaFree(40), cudaMemcpy(32), cudaMallocHost(23)/cudaFreeHost(21) [pinned], cudaHostRegister*(13)/cudaHostUnregister(13), cudaMemcpyToSymbolAsync(12) [__constant__], cudaMemGetInfo(7), cudaCreate/DestroyTextureObject(7), cudaCreateChannelDesc(7), cudaDeviceGetAttribute(8, cudaDevAttrHostRegisterSupported), cudaGetDeviceProperties(3). All have direct hip spellings.
- Textures: `cudaTextureObject_t` over a NON-layered 3D `cudaMalloc3DArray` (`make_cudaExtent`, `cudaMemcpy3DParms`/`cudaMemcpy3DAsync`), `resType=cudaResourceTypeArray`, `readMode=cudaReadModeElementType`. Sampled via `tex3D<float>(...,+0.5f,...)`. Filter modes:
  - `cudaFilterModeLinear` in: voxel_backprojection.cu (707), voxel_backprojection2.cu, voxel_backprojection_parallel.cu, ray_interpolated_projection.cu (594), ray_interpolated_projection_parallel.cu. The interpolation IS the algorithm (comment at ray_interpolated_projection.cu:198 "this line is 94% of time").
  - `cudaFilterModePoint` in Siddon_projection*.cu (ray-driven, no HW interp) -- those are unaffected.
- `__constant__` memory: `projParamsArrayDev`, `projSinCosArrayDev` written via `cudaMemcpyToSymbolAsync` -- HIP 1:1.
- Warp intrinsics: `__shfl_down_sync(0xFFFFFFFF, mySum, offset, 32)` in PICCS/GD_TV/GD_AwTV block reductions, loop `for(offset=warpSize/2; offset>0; offset/=2)`. The shuffle WIDTH is an explicit literal 32 (a width-32 LOGICAL-warp op), which the guide says is arch-agnostic and FINE on wave64. BUT the loop bound `warpSize/2` uses the physical `warpSize` (64 on gfx90a) while the shuffle width is 32 -- on wave64 the first iteration shuffles offset=32 against a width-32 segment, which is out of the logical-warp width and reads lane+32 across the 32-boundary. RISK: see risk list; needs the loop to iterate to the SHUFFLE width (32), not `warpSize/2`. No `__ballot`/`__popc`/`__activemask`/lane-packing; no hardcoded wave64 geometry.
- Pinned/managed memory: `cudaMallocHost`/`cudaHostRegister` (gated by `cudaDevAttrHostRegisterSupported`). HIP 1:1; watch the pageable-async hazard (risk list).
- Streams/events: `cudaStreamCreate`/`Synchronize`/`Destroy`, `cudaMemcpyAsync` on streams. 1:1.
- Multi-GPU: splits image/projection across `gpuids` with per-device `cudaSetDevice` + 2 streams/device. Pure runtime-API, ports as-is; validate single-GPU first.
- Libraries: cuRAND (`curand_kernel.h`, `curand_init`/`curand_poisson`/`curand_normal`) in RandomNumberGenerator.cu ONLY -> hipRAND (`hiprand_kernel.h`). NO cuBLAS, NO cuFFT, NO cuSPARSE, NO Thrust/CUB in the Python path.
- FFT: the FDK ramp filter is done on CPU in Python (Python/tigre/utilities/filtering.py uses `scipy.fft`/`np.fft`), NOT cuFFT. NO hipFFT/rocFFT substitution needed. (scatter.py also CPU fft2, host-only.)
- Atomics: only in `improvedForwardProjections*.cu` (43 each) -- MATLAB-only pCT path, NOT in setup.py, OUT OF SCOPE.

## Risk list
1. **HW linear-filter texture rejection over float (HIGH, central).** gfx90a/ROCm 7.2.1 rejects `cudaCreateTextureObject` with `filterMode=cudaFilterModeLinear` + `readMode=cudaReadModeElementType` over a float array (popsift readTex / sift_octave findings, PORTING_GUIDE). This hits every backprojection TU and the interpolated forward projector -- i.e. FDK and matched-backprojection reconstruction. Fix (guide-prescribed): under HIP create the texture `cudaFilterModePoint` and do trilinear interpolation in software inside the `tex3D` fetch (point-sample the 8 neighbors of a 3D texture and lerp, CUDA unnormalized -0.5 texel-center convention: coord c -> floor(c-0.5), weight (c-0.5)-floor). Route through a single `tex3D` fetch helper (e.g. `tex3D_interp(...)` in the compat header) so the ~handful of call sites are unchanged. Siddon (point-filtered) is unaffected. Verify empirically on gfx90a first -- if a future ROCm accepts linear float 3D textures the helper can fall back, but plan for software trilinear. addressMode=cudaAddressModeBorder (out-of-array reads return 0) must be replicated in software (return 0 outside [0,dim)).
2. **warpSize/2 loop vs width-32 shuffle (MEDIUM, wave64).** Block reductions in PICCS/GD_TV/GD_AwTV: `for(int offset=warpSize/2; offset>0; offset/=2) mySum += __shfl_down_sync(0xFFFFFFFF, mySum, offset, 32);`. On gfx90a warpSize=64 so the first offset is 32 = the logical width; `__shfl_down_sync(...,32,32)` shifts by the full sub-warp width (returns the thread's own value / out-of-segment), and the reduction then sums across two 32-lane halves incorrectly OR double-counts depending on how the partial sums are laid out. Fix: drive the loop by the shuffle width, `for(offset=16; offset>0; offset/=2)` (or `WARP_REDUCE_WIDTH/2` with WARP_REDUCE_WIDTH=32), keeping width-32 shuffles -- arch-agnostic and correct on both wave64 and wave32. Confirm whether the surrounding launch uses a 32- or 64-thread reduction tile; size accordingly. Cross-arch consistency check on gfx1100 (wave32) at follower time.
3. **Rule-of-five on texture/array handles (MEDIUM).** TIGRE uses raw `cudaTextureObject_t*`/`cudaArray**` arrays (malloc'd), created and `cudaDestroyTextureObject`/`cudaFreeArray`'d explicitly per device -- no RAII wrapper, so the colmap double-destroy class is unlikely, but verify every created texture is destroyed exactly once across the multi-GPU loop and not on an error early-return (AMD faults on destroying a never-initialized handle). Initialize handle arrays to 0.
4. **Pageable cudaMemcpyAsync use-after-free (MEDIUM).** Heavy `cudaMemcpyAsync`/`cudaMemcpy3DAsync` from host buffers on streams; CUDA stages pageable copies synchronously, HIP may not (CV-CUDA lesson). TIGRE uses pinned host memory (`cudaMallocHost`/`cudaHostRegister`) on the hot paths which is genuinely async on both, so verify each async copy's source outlives the stream sync; corruption would surface as nondeterministic reconstruction noise. Most copies are followed by `cudaStreamSynchronize` before reuse -- audit, don't assume.
5. **OOB texture/border reads (LOW-MED).** Ray/voxel kernels sample tex3D at computed coords; with HW border addressing OOB returns 0. The software-trilinear replacement (risk 1) MUST replicate border-zero for any of the 8 neighbors outside the array, or it faults/reads garbage on AMD (colmap OOB class). projection.cpp `maxDistanceCubeXY` already bounds the ray entry to minimize OOB, but edges still sample the border.
6. **256B texture pitch (LOW).** N/A here: textures are `cudaArray`-backed (cudaMalloc3DArray), not `cudaResourceTypePitch2D`, so the pitch-alignment class does not apply.
7. **setup.py nvcc-at-import (LOW, build).** `locate_cuda()` runs at module import and calls `nvcc --version`/`--list-gpu-arch`; on a ROCm-only host this throws before reaching ext config. The `BUILD_WITH_HIP` branch must short-circuit before `locate_cuda()`/`get_cuda_cc()`. Also `cuda_version` float-parse (121-126) is CUDA-only -- skip under HIP.
8. **ffp-contract / fsqrt 1-ULP (LOW).** Reconstruction is iterative and compared with nRMSE tolerances (not bit-exact), so the clang-HIP `-ffp-contract=fast` and `__fsqrt_rn` 1-ULP classes are unlikely to fail the gate. If a tight-tolerance test appears, pin `-ffp-contract=on`.

## File-by-file change sketch
- `Common/CUDA/cuda_to_hip.h` (NEW): include guard; `<cstring>`/`<cstdlib>` then `<hip/hip_runtime.h>` (+ `<hiprand/hiprand_kernel.h>` for the RNG TU) under USE_HIP, else `<cuda_runtime.h>`/`<cuda.h>`. Alias the histogram symbols (cudaMalloc->hipMalloc, cudaMemcpy*->hip*, cudaTextureObject_t->hipTextureObject_t, cudaArray->hipArray, cudaMalloc3DArray->hipMalloc3DArray, cudaMemcpy3D*->hipMemcpy3D*, cudaCreate/DestroyTextureObject, cudaCreateChannelDesc, cudaResourceDesc/cudaTextureDesc/enums, cudaMemcpyToSymbolAsync->hipMemcpyToSymbolAsync, cudaMallocHost/cudaHostRegister*/cudaDevAttrHostRegisterSupported, cudaDeviceProp/cudaGetDeviceProperties, curand*->hiprand*). Provide a `TIGRE_tex3D(tex,x,y,z)` fetch helper: on CUDA -> `tex3D<float>`; on HIP -> software trilinear over a point-filtered texture with border-zero.
- Each Python-built `.cu`: add `#include "cuda_to_hip.h"` at top (before other cuda includes). Replace the `cudaFilterModeLinear` texture creation with point-filter under `#if defined(USE_HIP)` and route the 3 `tex3D<float>` interpolated reads through `TIGRE_tex3D`. (Siddon point-filter reads can also route through the helper for uniformity but need no change.)
- PICCS.cu / GD_TV.cu / GD_AwTV.cu: fix the reduction loop bound (risk 2) to the shuffle width.
- gpuUtils.cu: includes `<cuda.h>`/`<cuda_runtime_api.h>` -> route through compat header.
- RandomNumberGenerator.cu: curand -> hiprand via header aliases (`curandState`->`hiprandState`, etc.).
- `setup.py`: add the `BUILD_WITH_HIP` branch (compiler swap to hipcc, `--offload-arch`, link amdhip64, skip locate_cuda); keep NVIDIA path unchanged. No CMake changes (project has none for the Python build).
- No host-`.cpp`/Cython/`.pyx`/`.pxd` changes expected.

## Build commands (gfx90a)
Host has ROCm 7.2 (hipcc, /opt/rocm/llvm/bin/clang++), gfx90a, python3.12, Cython 3.2.5, numpy 1.26.4.
```
# from projects/TIGRE/src
python3 -m pip install -e Python --no-build-isolation \
  2>&1   # after setup.py HIP branch lands; driven by:
export BUILD_WITH_HIP=1 ROCM_PATH=/opt/rocm HIP_ARCH=gfx90a
cd Python && pip install -e . --no-build-isolation
# (multi-arch later: HIP_ARCH="gfx90a,gfx1100")
```
The package installs the `_Ax`, `_Atb`, `_minTV`, `_minPICCS`, `_AwminTV`, `_tv_proximal`, `_gpuUtils`, `_RandomNumberGenerator` extension modules under tigre.

## Test plan (real gfx90a)
TIGRE ships no pytest CI; the real numerical gates are example.py + the algorithm_test harness. Validation (run with `gpuids=None`/empty name so device-name matching is bypassed -- getGpuNames returns the AMD device string and an empty filter selects all):
1. **Round-trip + reconstruction (primary correctness gate):** adapt `Python/example.py` headless (Agg backend, no plt.show): load head phantom (256^3), `tigre.Ax(head,geo,angles,projection_type='Siddon')` and `'interpolated'`, `tigre.Atb(proj,...,backprojection_type='matched')` and `'FDK'`, `algs.fdk(...)`, `algs.ossart(...,niter=20)`. Assert `Measure_Quality(fdkout, head, ['nRMSE'])` and the OS-SART nRMSE are within tolerance of the reference CUDA behavior (FDK nRMSE typically ~0.1-0.2 for this phantom; OS-SART lower). Both the Siddon (point texture) and interpolated (software-trilinear) paths must produce finite, artifact-free volumes -- a failed linear-texture fix shows as all-zero or NaN reconstructions.
2. **Adjointness check (matched projector):** `<Ax(x), y> ~= <x, Atb(y)>` within float tolerance using random x,y -- exercises forward+back without a phantom reference; a strong, reference-free numerical gate ideal for cross-arch consistency at follower time.
3. **Per-algorithm sweep:** `Python/tests/algorithm_test.py` over `generate_configurations.py` configs (cone, 256^3, 100 proj) for the iterative algos (sart/ossart/sirt/cgls/fista/asd_pocs using minTV/AwminTV/PICCS/tv_proximal) -- exercises every built extension. Compare reconstruction quality metrics to a tolerance, not bit-exact.
4. **RNG path:** `tigre.utilities` Poisson noise call (the `_RandomNumberGenerator` module) -- smoke + finiteness.
- Non-GPU regression set: the build itself (all 8 extensions compile/link), `import tigre`, geometry construction, the CPU filtering.py ramp filter (must remain CPU/unchanged). No host-side tests are touched by the port.

## Open questions
1. Exact nRMSE tolerances for the gold gate -- derive empirically from a first gfx90a run (no NVIDIA reference available on this host); adjointness check (#2) gives a hardware-independent gate that doesn't need a gold value.
2. Does ROCm 7.2 accept a 3D-array linear-filter float texture, or must we software-trilinear unconditionally? Porter to probe `hipCreateTextureObject` with cudaFilterModeLinear on gfx90a first; the guide's popsift finding says it is REJECTED for element-read float, so plan for software trilinear but confirm.
3. Multi-GPU split correctness on a 4-GCD MI-class node: validate single-device first (set HIP_VISIBLE_DEVICES=0), then exercise the split path.
4. hiprand_kernel API parity for `curand_poisson`/`curand_normal` -- hipRAND wraps rocRAND; confirm the device-side distribution calls map 1:1 (they generally do).
