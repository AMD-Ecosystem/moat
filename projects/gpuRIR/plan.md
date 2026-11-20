# gpuRIR ROCm/HIP port plan

Upstream: https://github.com/DavidDiazGuerra/gpuRIR (master @ fd8af43)
Target: linux-gfx90a (MI250X, CDNA2, wave64). ROCm 7.2.1.

## What it is

GPU Room Impulse Response simulator. Image Source Method (ISM) for early
reflections + a stochastic diffuse-reverberation tail, plus an FFT-based
batched convolution for filtering signals along a trajectory. Exposed as a
pybind11 Python extension `gpuRIR_bind`, wrapped by the `gpuRIR` Python
package.

## Existing AMD support

None. Pure CUDA (`.cu` + cuFFT + cuRAND), no HIP path, no OpenCL/SYCL
alternative. Disposition: fresh CUDA->HIP port.

## Build classification

Pure CMake building a pybind11 MODULE library; not a pytorch extension
(no find_package(Torch), no torch.utils.cpp_extension). -> Strategy A
(one compat header, `.cu` marked LANGUAGE HIP) plus a
**library swap** cuFFT->hipFFT and cuRAND->hipRAND.

Build entry points: `pip install .` / `setup.py` drives CMake
(CMakeExtension/CMakeBuild). CMakeLists builds a static lib `gpuRIRcu`
from `src/gpuRIR_cuda.cu` and a MODULE `gpuRIR_bind` from
`src/python_bind.cpp`, linking pybind11, cuFFT, cuRAND.

## CUDA surface inventory (src/gpuRIR_cuda.cu, the only .cu)

- Kernels (no warp intrinsics, no __shfl/__ballot/__any, no __constant__):
  calcAmpTau, generateRIR (+_mp, +_lut), reduceRIR (+_mp),
  h2RIR_to_floatRIR, envPred, diffRev, complexPointwiseMulAndScale.
  Reductions use shared memory + __syncthreads only -> **no wave64 fault
  class** (the popsift warp-packing bug does not apply).
- cuFFT: cufftHandle, cufftComplex, cufftReal, cufftResult_t,
  cufftPlan1d, cufftExecR2C, cufftExecC2R, cufftDestroy,
  CUFFT_R2C/C2R/SUCCESS. Also includes <cufftXt.h> but uses none of its
  symbols (only the cufftComplex/Real types). -> drop cufftXt include on HIP.
- cuRAND (host API): curandGenerator_t, curandStatus_t,
  CURAND_STATUS_SUCCESS, curandCreateGenerator,
  curandSetPseudoRandomGeneratorSeed, curandGenerateUniform,
  CURAND_RNG_PSEUDO_DEFAULT.
- Runtime: malloc/free/memcpy/memset, memcpy2D/memset2D, **3D pitched
  copies** (cudaMemcpy3D, cudaMemcpy3DParms, make_cudaPitchedPtr,
  make_cudaExtent, make_cudaPos, cudaPitchedPtr), events not used,
  cudaGetDeviceProperties, cudaPeekAtLastError, cudaDeviceSynchronize.
- fp16: <cuda_fp16.h>, half2 intrinsics (mixed-precision path). HIP has
  <hip/hip_fp16.h> and the same half2 intrinsics; gate the include.
- **Texture LUT path** (default on: python_bind ctor lut=true,
  example.py activateLUT(True)): a 1D cudaArray with cudaFilterModeLinear
  + cudaReadModeElementType, sampled by tex1D<float> at a float coord
  computed with __fmaf_rz. Two HIP issues here:
  1. __fmaf_rz (round-toward-zero FMA) does not exist in HIP (only
     __fmaf_rn). CudaSift hit the same with __fmul_rz.
  2. Hardware linear filtering on element-read float textures is the
     popsift fault class (HIP rejects cudaFilterModeLinear +
     cudaReadModeElementType at create time on AMD).

## Fault-class decisions

- Warp size: N/A (no warp intrinsics; nThreadsGen_t=32 is just a block
  dim, not a 32-lane assumption).
- __fmaf_rz: alias to round-to-nearest. The value is only a LUT index
  coordinate, rounding mode is immaterial.
- Texture linear filter (LUT): on HIP, create the LUT texture with
  **point** filtering and do the linear interpolation manually in
  image_sample_lut (point-fetch the two neighbors, lerp), using CUDA's
  unnormalized -0.5 texel-center convention. This sidesteps both the
  __fmaf_rz issue and the linear-filter rejection in one move, and keeps
  the default LUT path working on AMD. CUDA path unchanged.
- cudaMemcpyToArray: still declared in ROCm 7.2 (deprecated) with the
  same signature -> alias directly, no call-site rewrite needed (unlike
  CudaSift on older ROCm).

## Changes (HIP-guarded only; CUDA path byte-for-byte unchanged)

1. New `src/cuda_to_hip.h`: on USE_HIP include hip/hip_runtime,
   hipfft, hiprand, hip_fp16; #define the cuda/cufft/curand symbols above
   to their hip spellings (names per torch hipify map). Alias __fmaf_rz
   to __fmaf_rn. Else include the original CUDA headers.
2. `src/gpuRIR_cuda.cu`: replace the cuda/cufft/curand/fp16 #includes
   with `#include "cuda_to_hip.h"`. Add a small
   `#if defined(USE_HIP)` branch in create_sinc_texture_lut (point
   filter) and image_sample_lut (manual lerp). Nothing else.
3. `CMakeLists.txt`: add `option(USE_HIP)`; when ON, enable_language(HIP),
   mark gpuRIR_cuda.cu LANGUAGE HIP, set HIP_ARCHITECTURES gfx90a,
   compile-define USE_HIP, force-include cuda_to_hip.h, and link
   hip::hipfft + hip::hiprand (roc::rocfft/rocrand transitively) instead
   of cuFFT/cuRAND. Guard find_package(CUDA) so it is skipped under HIP
   (no CUDA on the AMD host). project() LANGUAGES gated likewise.

## Build

`pip install . --no-build-isolation` with CMAKE args forwarded via env to
select USE_HIP. Since setup.py hardcodes the cmake args, pass HIP options
through `CMAKE_ARGS` env honored by a small addition, or invoke cmake
directly into a build dir and point pip at the result. Simplest: drive
cmake directly for validation (configure with -DUSE_HIP=ON
-DCMAKE_HIP_ARCHITECTURES=gfx90a), then copy/locate the built .so onto
PYTHONPATH next to the gpuRIR package. (Also wire CMAKE_ARGS into setup.py
so `pip install .` works hands-free.)

## Validation

rocm-smi to pick an idle gfx90a; HIP_VISIBLE_DEVICES=N. Run the shoebox
example (examples/example.py shape, headless): simulateRIR for a small
room, omni + a card mic. Assert the returned RIR is finite (no NaN/Inf),
non-zero, and the direct-path peak lands at the expected sample
round(dist/c*Fs) for a known src/rcv geometry. Exercise both LUT on and
off, and the gpu_conv FFT path (simulateTrajectory) to cover hipFFT.

## MOAT state

planned -> porting -> ported on success; set-blocked with a precise
reason only if an unresolvable hipFFT/plan incompatibility appears.
