# OCTproZ Porting Plan

## Project

- **Name:** OCTproZ
- **Upstream:** https://github.com/spectralcode/OCTproZ
- **Default branch:** main
- **Description:** Open source software for optical coherence tomography (OCT) raw data processing and visualization

## Existing AMD support

**Status: None**

- Upstream README and docs contain no references to AMD, ROCm, HIP, or gfx architectures
- No ROCm/HIP forks found in GitHub forks list (20 forks checked -- all appear to be personal forks)
- Web search for "OCTproZ ROCm" / "OCTproZ AMD GPU" / "OCTproZ HIP" returned no relevant results
- No ROCm/AMD-related issues or PRs in the upstream repo
- No existing branches with rocm/hip in name

**Decision:** Proceed with a fresh HIP port. This is a CUDA-only medical imaging application that would benefit from AMD GPU support for researchers using AMD hardware.

## Build classification

**Type:** qmake (Qt project with custom CUDA compilation rules)

**Evidence:**
- `octproz_project/octproz/octproz.pro` (lines 1-159): Qt qmake project file
- `octproz_project/octproz/pri/cuda.pri` (lines 1-173): Custom qmake CUDA compiler rules that invoke nvcc directly
- No CMakeLists.txt present in the project
- Build system uses `QMAKE_EXTRA_COMPILERS` to add nvcc as a custom compiler for `.cu` files

**Build flow:**
1. qmake processes the `.pro` files
2. Custom compiler rules in `cuda.pri` compile `.cu` files with nvcc
3. Links against `libcudart`, `libcuda`, `libcufft`

## Port strategy

**Strategy:** Modified Strategy A (qmake-based, compat-header approach)

**Rationale:**
- This is a standalone application (not a PyTorch extension), so Strategy B does not apply
- The project uses qmake instead of CMake, requiring adaptation of the Strategy A approach
- A single compat header (`cuda_to_hip.h`) will provide CUDA-to-HIP symbol mapping
- A parallel `hip.pri` file will configure hipcc compilation for `.cu` files
- The `.cu` sources remain in CUDA spelling; the compat header aliases the symbols

**Implementation:**
1. Create `src/cuda_to_hip.h` compat header with symbol mappings
2. Create `octproz/pri/hip.pri` as the HIP equivalent of `cuda.pri`
3. Modify `octproz.pro` to conditionally include `hip.pri` or `cuda.pri` based on a `USE_HIP` config option
4. Include the compat header at the top of `cuda_code.cu`
5. Fix any HIP-specific issues (surface objects, cuFFT->hipFFT, etc.)

## CUDA surface inventory

### Source files
- `octproz_project/octproz/src/cuda_code.cu` (1698 lines) -- the only CUDA source file
- `octproz_project/octproz/src/kernels.h` -- declarations for CUDA functions

### Kernels (`__global__` functions)
1. `inputToCufftComplex` -- data type conversion
2. `inputToCufftComplex_and_bitshift` -- data conversion with bit shifting
3. `rollingAverageBackgroundRemoval` -- uses `__shared__` memory
4. `klinearization` -- k-space resampling (linear interpolation)
5. `klinearizationQuadratic` -- k-space resampling (quadratic)
6. `klinearizationCubic` -- k-space resampling (cubic Hermite)
7. `klinearizationLanczos` -- k-space resampling (Lanczos-8)
8. `windowing` -- apply window function
9. `klinearizationAndWindowing` -- combined k-linearization + windowing
10. `klinearizationCubicAndWindowing` -- combined cubic k-lin + windowing
11. `klinearizationLanczosAndWindowing` -- combined Lanczos k-lin + windowing
12. `klinearizationAndWindowingAndDispersionCompensation` -- three-in-one
13. `klinearizationCubicAndWindowingAndDispersionCompensation` -- three-in-one cubic
14. `klinearizationLanczosAndWindowingAndDispersionCompensation` -- three-in-one Lanczos
15. `sinusoidalScanCorrection` -- sinusoidal scan pattern correction
16. `fillSinusoidalScanCorrectionCurve` -- initialize correction curve
17. `getMinimumVarianceMean` -- fixed pattern noise removal
18. `meanALineSubtraction` -- subtract mean A-line
19. `dispersionCompensation` -- phase-based dispersion compensation
20. `dispersionCompensationAndWindowing` -- combined dispersion + windowing
21. `fillDispersivePhase` -- initialize dispersion phase curve
22. `postProcessTruncateLog` -- magnitude, log scaling, truncation
23. `postProcessTruncateLin` -- magnitude, linear scaling, truncation
24. `getPostProcessBackground` -- compute background for removal
25. `postProcessBackgroundRemoval` -- subtract background
26. `cuda_bscanFlip_slow` -- flip B-scans (slow version)
27. `cuda_bscanFlip` -- flip B-scans (optimized)
28. `updateDisplayedBscanFrame` -- update B-scan display buffer
29. `updateDisplayedEnFaceViewFrame` -- update en-face display buffer
30. `updateDisplayedVolume` -- write to 3D texture for volume view
31. `floatToOutput` -- convert float to output bit depth

### Device functions (`__device__` / `__forceinline__ __device__`)
- `endianSwapUint32`, `endianSwapInt32`, `endianSwapUint16`, `endianSwapInt16` -- byte swap
- `cubicHermiteInterpolation` -- Hermite interpolation
- `lanczosKernel8` -- Lanczos-8 kernel evaluation
- `cuMultiply` -- complex multiplication

### Library usage
- **cuFFT:** `cufftPlan1d`, `cufftExecC2C`, `cufftSetStream`, `cufftDestroy`, `cufftComplex`, `CUFFT_C2C`, `CUFFT_INVERSE`
  - **HIP equivalent:** hipFFT (`hipfftPlan1d`, `hipfftExecC2C`, etc.)
- **CUDA Runtime:** `cudaMalloc`, `cudaFree`, `cudaMemcpy*`, `cudaMemset`, `cudaStream*`, `cudaEvent*`, `cudaGetDevice*`, `cudaHostRegister`, `cudaHostUnregister`, `cudaLaunchHostFunc`, `cudaDeviceSynchronize`, `cudaPeekAtLastError`, `cudaGetLastError`, `cudaMemGetInfo`
  - **HIP equivalents:** Direct 1:1 mappings via compat header

### CUDA-OpenGL interop
- `cudaGraphicsGLRegisterBuffer` -- register OpenGL buffer for CUDA access
- `cudaGraphicsGLRegisterImage` -- register OpenGL 3D texture
- `cudaGraphicsMapResources` / `cudaGraphicsUnmapResources` -- map/unmap resources
- `cudaGraphicsResourceGetMappedPointer` -- get device pointer
- `cudaGraphicsSubResourceGetMappedArray` -- get mapped array
- **HIP equivalents:** `hipGraphicsGLRegisterBuffer`, `hipGraphicsMapResources`, etc. (requires `-lhiprtc` linking)

### Surface objects
- Legacy surface binding (CUDA < 12): `surface<void, cudaSurfaceType3D> surfaceWrite;` + `cudaBindSurfaceToArray`
- Modern surface objects (CUDA >= 12): `cudaCreateSurfaceObject`, `cudaDestroySurfaceObject`
- `surf3Dwrite` -- write to 3D surface (used in `updateDisplayedVolume`)
- **HIP equivalents:** `hipCreateSurfaceObject`, `hipDestroySurfaceObject`, `surf3Dwrite` (via HIP surface API)

### Host-registered memory
- `cudaHostRegister`, `cudaHostUnregister` with `cudaHostRegisterPortable`
- **HIP equivalent:** `hipHostRegister`, `hipHostUnregister`

### Streams and events
- `cudaStreamCreate`, `cudaStreamCreateWithFlags`, `cudaStreamDestroy`
- `cudaEventCreateWithFlags`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventDestroy`
- `cudaStreamNonBlocking`, `cudaEventBlockingSync`
- **HIP equivalents:** Direct mappings

### Device intrinsics
- `__uint2float_rd` -- unsigned int to float with round-down
- `__saturatef` -- clamp float to [0,1]
- `__syncthreads` -- block synchronization (line 199)
- **HIP equivalents:** Direct mappings (HIP provides `__uint2float_rd`, `__saturatef`, `__syncthreads`)

### Warp intrinsics
- **None detected** -- no `__shfl*`, `__ballot`, `__activemask`, or explicit `warpSize` usage in kernels

### Other CUDA features
- `cudaDeviceProp` properties queried: `maxThreadsPerBlock`, `warpSize`, `name`, `major`, `minor`, etc.
- `CUDART_VERSION` macro check for CUDA 13+ compatibility
- `__CUDACC_VER_MAJOR__` macro check for surface object API selection
- `__aarch64__` checks for Jetson Nano support

## Risk list

1. **Surface objects / CUDA-OpenGL interop (MEDIUM):** The volume visualization uses `surf3Dwrite` and CUDA-GL interop. HIP supports this via `hipGraphicsGLRegisterBuffer` and surface objects, but the interop behavior may differ. The legacy surface binding path (CUDA < 12) uses `cudaBindSurfaceToArray` which is deprecated; on HIP, only the modern surface object API is available.

2. **cuFFT -> hipFFT (LOW):** The project uses `cufftPlan1d` and `cufftExecC2C` for 1D complex-to-complex FFT. hipFFT provides 1:1 compatible APIs. The handle type changes from `cufftHandle` to `hipfftHandle` and the complex type from `cufftComplex` to `hipfftComplex`.

3. **qmake build system (MEDIUM):** The project uses qmake with custom nvcc compiler rules. A parallel `hip.pri` must be created that invokes `hipcc` instead of `nvcc`. The architecture flags need to be adapted from CUDA sm_* to HIP gfx*.

4. **`__aarch64__` / Jetson Nano code paths (LOW):** Several `#ifdef __aarch64__` guards exist for Jetson Nano zero-copy memory. These are orthogonal to the HIP port and can be left as-is (they will not trigger on x86-64 ROCm).

5. **`CUDART_VERSION` / `__CUDACC_VER_MAJOR__` macros (LOW):** These are used for CUDA version detection. On HIP, define `__CUDACC_VER_MAJOR__` to 12 (or use HIP's version macros) to select the modern surface object path.

6. **helper_cuda.h (LOW):** The project includes `<helper_cuda.h>` from CUDA samples for `checkCudaErrors`. This macro needs to be redefined for HIP or replaced with inline error checking.

7. **`cuda_profiler_api.h` (LOW):** Included but not actively used in the code. Can be conditionally excluded on HIP.

8. **`driver_functions.h` (LOW):** CUDA driver types header. HIP has equivalent driver types; include conditionally.

9. **No warp-size concerns (NONE):** The kernels do not use warp-collective operations or hardcode warp size = 32. Block size is 128, which works on both wave64 (gfx90a) and wave32 (gfx1100) architectures.

## File-by-file change list

### New files
1. `octproz_project/octproz/src/cuda_to_hip.h` -- compat header with CUDA-to-HIP symbol mappings
2. `octproz_project/octproz/pri/hip.pri` -- qmake HIP compiler configuration (parallel to cuda.pri)

### Modified files
1. `octproz_project/octproz/octproz.pro` -- add `USE_HIP` config option, conditionally include hip.pri or cuda.pri
2. `octproz_project/octproz/src/cuda_code.cu` -- include cuda_to_hip.h, select modern surface object path on HIP
3. `octproz_project/octproz/src/kernels.h` -- include cuda_to_hip.h, adapt headers for HIP
4. `octproz_project/octproz/src/gpuinfo.cpp` -- uses cudaDeviceProp; will work via compat header mappings
5. `octproz_project/octproz/src/gpuinfo.h` -- may need compat header include

### Symbol mapping summary (cuda_to_hip.h)
```
cudaMalloc -> hipMalloc
cudaFree -> hipFree
cudaMemcpy -> hipMemcpy
cudaMemcpyAsync -> hipMemcpyAsync
cudaMemset -> hipMemset
cudaMemGetInfo -> hipMemGetInfo
cudaStream_t -> hipStream_t
cudaStreamCreate -> hipStreamCreate
cudaStreamCreateWithFlags -> hipStreamCreateWithFlags
cudaStreamDestroy -> hipStreamDestroy
cudaStreamNonBlocking -> hipStreamNonBlocking
cudaEvent_t -> hipEvent_t
cudaEventCreateWithFlags -> hipEventCreateWithFlags
cudaEventRecord -> hipEventRecord
cudaEventSynchronize -> hipEventSynchronize
cudaEventDestroy -> hipEventDestroy
cudaEventBlockingSync -> hipEventBlockingSync
cudaGetDevice -> hipGetDevice
cudaSetDevice -> hipSetDevice
cudaGetDeviceCount -> hipGetDeviceCount
cudaGetDeviceProperties -> hipGetDeviceProperties
cudaDeviceGetAttribute -> hipDeviceGetAttribute
cudaDeviceProp -> hipDeviceProp_t
cudaDeviceSynchronize -> hipDeviceSynchronize
cudaHostRegister -> hipHostRegister
cudaHostUnregister -> hipHostUnregister
cudaHostRegisterPortable -> hipHostRegisterPortable
cudaLaunchHostFunc -> hipLaunchHostFunc
cudaPeekAtLastError -> hipPeekAtLastError
cudaGetLastError -> hipGetLastError
cudaError_t -> hipError_t
cudaSuccess -> hipSuccess
cudaGraphicsResource -> hipGraphicsResource_t
cudaGraphicsGLRegisterBuffer -> hipGraphicsGLRegisterBuffer
cudaGraphicsGLRegisterImage -> hipGraphicsGLRegisterImage
cudaGraphicsMapResources -> hipGraphicsMapResources
cudaGraphicsUnmapResources -> hipGraphicsUnmapResources
cudaGraphicsResourceGetMappedPointer -> hipGraphicsResourceGetMappedPointer
cudaGraphicsSubResourceGetMappedArray -> hipGraphicsSubResourceGetMappedArray
cudaGraphicsRegisterFlagsWriteDiscard -> hipGraphicsRegisterFlagsWriteDiscard
cudaGraphicsRegisterFlagsSurfaceLoadStore -> hipGraphicsRegisterFlagsSurfaceLoadStore
cudaArray -> hipArray_t
cudaSurfaceObject_t -> hipSurfaceObject_t
cudaCreateSurfaceObject -> hipCreateSurfaceObject
cudaDestroySurfaceObject -> hipDestroySurfaceObject
cudaResourceDesc -> hipResourceDesc
cudaResourceTypeArray -> hipResourceTypeArray
cufftHandle -> hipfftHandle
cufftComplex -> hipfftComplex
cufftPlan1d -> hipfftPlan1d
cufftExecC2C -> hipfftExecC2C
cufftSetStream -> hipfftSetStream
cufftDestroy -> hipfftDestroy
CUFFT_C2C -> HIPFFT_C2C
CUFFT_INVERSE -> HIPFFT_BACKWARD
checkCudaErrors -> (inline HIP_CHECK macro)
```

## Build commands

### Configure and build for gfx90a (Linux)

```bash
# Prerequisites: ROCm 7.2.1+, Qt 5.12+, hipFFT

cd octproz_project

# Build with HIP enabled
qmake CONFIG+=USE_HIP octproz_project.pro

# Or specify arch explicitly
qmake CONFIG+=USE_HIP HIP_ARCH=gfx90a octproz_project.pro

make -j$(nproc)
```

### Alternative: specify HIP arch via environment
```bash
export HIP_ARCHITECTURES=gfx90a
qmake CONFIG+=USE_HIP octproz_project.pro
make -j$(nproc)
```

### Dependencies (Linux)
```bash
# ROCm stack
sudo apt install rocm-dev hipfft-dev

# Qt 5
sudo apt install qt5-default qtbase5-dev

# OpenGL
sudo apt install libgl1-mesa-dev libglu1-mesa-dev
```

## Test plan

### Real GPU tests

OCTproZ does not have a formal automated test suite. Validation requires manual functional testing with the application.

**Test procedure:**
1. Build OCTproZ with HIP support on gfx90a
2. Launch the application: `./OCTproZ`
3. Verify GPU is detected (Help -> About should show GPU info)
4. Load sample OCT data using the Virtual OCT System plugin (included)
5. Verify processing pipeline runs (B-scan view updates in real-time)
6. Verify volume view renders correctly (3D surface object writes)
7. Test en-face view display
8. Test different processing options:
   - K-linearization (linear, cubic, Lanczos)
   - Windowing
   - Dispersion compensation
   - Fixed pattern noise removal
   - Background removal
   - B-scan flip
   - Sinusoidal scan correction
9. Compare output visually with CUDA build on same data

### Non-GPU regression tests

No non-GPU tests identified. The application is GPU-centric; all processing happens on GPU.

### Virtual OCT System plugin

The Virtual OCT System plugin (`octproz_plugins/octproz-virtual-oct-system`) generates synthetic OCT data for testing without physical hardware. This is the primary means of testing without an actual OCT system.

## Open questions

1. **Surface object interop on HIP:** The CUDA-OpenGL interop path for the 3D volume texture uses `cudaGraphicsGLRegisterImage` + `surf3Dwrite`. While HIP supports this, the exact behavior with GL textures should be verified on real hardware. If issues arise, an alternative would be to use a pixel buffer object (PBO) approach instead of direct surface writes.

2. **hipFFT complex type:** cuFFT uses `cufftComplex` which is `float2`. hipFFT uses `hipfftComplex` which is also `float2`. Verify binary compatibility or use explicit casts if needed.

3. **Qt OpenGL context compatibility:** OCTproZ uses Qt OpenGL for rendering. The CUDA-GL interop requires the OpenGL context to be compatible with CUDA. On HIP, verify the GL context is compatible with `hipGraphicsGLRegisterBuffer`. This typically requires EGL or GLX with appropriate extensions.

4. **Windows port (gfx1101/gfx1201):** The HIP SDK on Windows has different library names and build procedures. The `hip.pri` will need Windows-specific adaptations. The CUDA-GL interop path may have different behavior on Windows DirectX/WGL backends.
