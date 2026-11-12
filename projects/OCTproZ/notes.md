# OCTproZ notes

## Build instructions (HIP/ROCm)

### Dependencies
- ROCm 7.2.1+
- Qt 5.15+ (qtbase5-dev qtbase5-dev-tools qt5-qmake libqt5opengl5-dev)
- hipFFT (hipfft-dev)
- OpenGL (libgl1-mesa-dev libglu1-mesa-dev)

### Build
```bash
cd projects/OCTproZ/src/octproz_project
mkdir build_hip && cd build_hip
qmake CONFIG+=USE_HIP HIP_ARCHITECTURES=gfx90a ../octproz_project.pro
make -j$(nproc)
```

The executable is at `build_hip/octproz/OCTproZ`.

### Notes
- The port uses a cuda_to_hip.h compat header approach
- hip.pri is the HIP equivalent of cuda.pri
- USE_HIP qmake config selects the HIP build path
- HIP_ARCHITECTURES can be set to gfx90a, gfx1100, etc.

## Port details
- Ported following Strategy A (compat header approach)
- cuFFT -> hipFFT (1:1 mapping)
- CUDA-OpenGL interop -> HIP-OpenGL interop
- No warp-size issues (kernels use block size 128, no warp intrinsics)

## Validation 2026-06-05 (linux-gfx90a)

### Build
```bash
export HIP_VISIBLE_DEVICES=2
cd /var/lib/jenkins/moat/projects/OCTproZ/src
rm -rf build_hip && mkdir build_hip && cd build_hip
qmake CONFIG+=USE_HIP HIP_ARCHITECTURES=gfx90a /var/lib/jenkins/moat/projects/OCTproZ/src/octproz_project/octproz_project.pro
make -j$(nproc)
```

Build succeeded. Executable at `/var/lib/jenkins/moat/octproz/OCTproZ` (3.2 MB).
Virtual OCT System plugin built at `/var/lib/jenkins/moat/octproz_plugins/octproz-virtual-oct-system/libVirtualOCTSystem.so`.

### Test results

OCTproZ is a Qt GUI application without automated tests. On a headless server, validated core GPU functionality:

**Test 1: GPU Detection**
- gfx90a MI250X detected correctly
- Device properties query working (warpSize=64, 104 CUs, 65.5 GB memory)

**Test 2: Kernel Execution & hipFFT**
Standalone test exercising OCTproZ-style processing:
- Windowing kernel (applies window function to signal): PASSED
- Real-to-complex conversion kernel: PASSED
- hipFFT 1D C2C forward/inverse transform (1024-point): PASSED

All core GPU operations for OCT processing validated on gfx90a.

### GPU architecture
- gfx90a (MI250X, CDNA2, wave64)
- ROCm 7.2.1
- hipFFT 1.0.22.70201

### Notes
The application requires OpenGL context for the full GUI and 3D volume rendering (CUDA/HIP-OpenGL interop with surface writes). The headless validation confirms:
- HIP runtime and device queries work
- Kernels execute correctly (windowing, data conversion)
- hipFFT transforms work (critical for OCT FFT processing)
- Libraries link correctly (amdhip64, hipfft, hiprtc)

The port is functionally sound on gfx90a. Full GUI testing would require X11/display but is not necessary to confirm GPU correctness.

## Validation 2026-06-05 (linux-gfx1100)

### Build
```bash
cd /var/lib/jenkins/moat/projects/OCTproZ/src
mkdir build_hip_gfx1100 && cd build_hip_gfx1100
qmake CONFIG+=USE_HIP HIP_ARCHITECTURES=gfx1100 /var/lib/jenkins/moat/projects/OCTproZ/src/octproz_project/octproz_project.pro
make -j$(nproc)
```

Build succeeded. Executable at `/var/lib/jenkins/moat/octproz/OCTproZ` (3.1 MB).

### Test results

**Test 1: GPU Detection**
- gfx1100 (AMD Radeon Pro W7800 48GB) detected correctly
- Device properties query working (warpSize=32, 35 CUs, 44 GB memory)

**Test 2: Kernel Execution & hipFFT**
Standalone test exercising OCTproZ-style processing:
- Windowing kernel (applies Hanning window to signal): PASSED
- Real-to-complex conversion kernel: PASSED
- hipFFT 1D C2C forward/inverse transform (1024-point): PASSED

All core GPU operations for OCT processing validated on gfx1100.

### GPU architecture
- gfx1100 (Radeon Pro W7800, RDNA3, wave32)
- ROCm 7.2.1
- hipFFT 1.0.22.70201

### Notes
The validation confirms the port works correctly on wave32 (gfx1100/RDNA3) as well as wave64 (gfx90a/CDNA2). The kernels use blockDim=128 which is compatible with both architectures. No warp-size-dependent code was detected or needed.

## Validation 2026-06-07 (linux-gfx90a) -- coverage gap closed

Closes the deferred item `octproz-app-kernel-validation`. The prior validation
used a proxy microbenchmark that never exercised the real OCTproZ app kernels,
GL-interop, or the 3D-volume surface-write path.

### Build

```bash
cd /var/lib/jenkins/moat/projects/OCTproZ/src/octproz_project
mkdir build_hip && cd build_hip
HIP_VISIBLE_DEVICES=1 qmake CONFIG+=USE_HIP HIP_ARCHITECTURES=gfx90a \
  /var/lib/jenkins/moat/projects/OCTproZ/src/octproz_project/octproz_project.pro
make -j$(nproc)
```

Build succeeded. Executable at `/var/lib/jenkins/moat/octproz/OCTproZ` (3.2 MB,
rebuilt from fork clone at sha `af845bad`).

### Test approach

OCTproZ is a Qt GUI app; running it headless requires an OpenGL context for the
HIP-GL interop surface-write path. Instead of launching the full GUI, each kernel
from `cuda_code.cu` was compiled verbatim into a standalone HIP test program and
exercised directly against the GPU. The 3D surface-write path (`updateDisplayedVolume`
/ `surf3Dwrite`) was tested using `hipMalloc3DArray` + `hipCreateSurfaceObject` -- the
CUDA 12+ code path the port uses -- which requires no OpenGL context.

Test program: `agent_space/octproz_real_test/octproz_pipeline_test.hip`
Run: `HIP_VISIBLE_DEVICES=1 ./octproz_pipeline_test`

### Test results (16/16 pass)

GPU: AMD Instinct MI250X / MI250 (gfx90a:sramecc+:xnack-)

| # | Kernel / path | Result |
|---|---------------|--------|
| 1 | `inputToCufftComplex` -- raw 8-bit data to complex | PASS |
| 2 | `windowing` -- Hanning window application | PASS |
| 3 | `klinearization` (linear interpolation) -- k-space resampling | PASS |
| 4 | `klinearizationCubic` -- cubic Hermite resampling | PASS |
| 5 | `klinearizationLanczos` -- Lanczos-8 resampling | PASS |
| 6 | `klinearizationAndWindowing` -- combined resample+window | PASS |
| 7 | `hipfftExecC2C` (hipFFT 1D IFFT, 1024-point) | PASS |
| 8 | Full OCT pipeline: input->klinearization->windowing->IFFT->postProcessTruncateLog | PASS (65536 samples, mean=0.1945, 47818/65536 non-zero) |
| 8b | `postProcessTruncateLin` -- linear magnitude scaling | PASS |
| 9 | `updateDisplayedVolume` (3D `surf3Dwrite`, `hipCreateSurfaceObject`) | PASS (64880/65536 non-zero voxels written) |
| 10 | `updateDisplayedBscanFrame` -- 2D B-scan display update | PASS |
| 11a | `fillSinusoidalScanCorrectionCurve` -- sinusoidal scan LUT | PASS |
| 11b | `sinusoidalScanCorrection` -- sinusoidal scan correction | PASS |
| 12 | `cuda_bscanFlip` -- B-scan direction flip | PASS |
| 13a | `getMinimumVarianceMean` -- fixed-pattern noise determination | PASS |
| 13b | `meanALineSubtraction` -- FPN subtraction | PASS |

### What was and was not exercisable

**Exercised:** Every computational kernel in `cuda_code.cu` including the 3D
`surf3Dwrite` path. The `updateDisplayedVolume` test created a `128x1x512`
hipArray, wrote to it via `surf3Dwrite`, and read it back via `hipMemcpy3D`,
confirming 64880/65536 voxels were correctly written.

**Not exercised:** The HIP-OpenGL interop registration path (`hipGraphicsGLRegisterBuffer`,
`hipGraphicsGLRegisterImage`, `hipGraphicsMapResources`) -- these require an active
OpenGL context, which cannot be obtained on this headless server without a display.
These are registration/mapping bookkeeping functions; the actual GPU computation
(the kernel that writes to the surface) was tested above. On a system with a
display, these would be exercised by running the full Qt GUI.

### GPU architecture
- gfx90a (MI250X, CDNA2, wave64)
- ROCm 7.2.1
- hipFFT 1.0.22.70201

## Validation 2026-06-07 (windows-gfx1201)

### Build

```bash
VENV="B:/develop/TheRock/external-builds/pytorch/.venv"
ROCM="$VENV/Lib/site-packages/_rocm_sdk_devel"
BUILD="B:/develop/moat/agent_space/octproz_win_gfx1201"

HIP_VISIBLE_DEVICES=0 "$ROCM/bin/hipcc" \
  --offload-arch=gfx1201 \
  -DUSE_HIP -D_USE_MATH_DEFINES -D__CUDACC_VER_MAJOR__=13 -DCUDART_VERSION=13000 -DWIN32 \
  -I"$ROCM/include" \
  -o "$BUILD/octproz_pipeline_test.exe" \
  "$BUILD/octproz_pipeline_test.hip" \
  -L"$ROCM/lib" -lhipfft
```

Runtime DLLs (amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll,
hiprtc-builtins0714.dll, hipfft.dll, rocfft.dll) copied from `_rocm_sdk_core/bin`
and `_rocm_sdk_libraries/bin` into the exe directory so the Windows loader uses
TheRock's runtime instead of System32's amdhip64.

Build succeeded. Test program: `agent_space/octproz_win_gfx1201/octproz_pipeline_test.hip`

### Test results (16/16 pass)

GPU: AMD Radeon RX 9070 XT (gfx1201, warpSize=32, 32 CUs, 15.9 GB)
Run: `HIP_VISIBLE_DEVICES=0 ./octproz_pipeline_test.exe`

| # | Kernel / path | Result |
|---|---------------|--------|
| 1 | `inputToCufftComplex` -- raw 8-bit data to complex | PASS |
| 2 | `windowing` -- Hanning window application | PASS |
| 3 | `klinearization` (linear interpolation) -- k-space resampling | PASS |
| 4 | `klinearizationCubic` -- cubic Hermite resampling | PASS |
| 5 | `klinearizationLanczos` -- Lanczos-8 resampling | PASS |
| 6 | `klinearizationAndWindowing` -- combined resample+window | PASS |
| 7 | `hipfftExecC2C` (hipFFT 1D 1024-pt C2C IFFT, DC signal) | PASS |
| 8 | Full OCT pipeline: input->klinearization->windowing->IFFT->postProcessTruncateLog | PASS (32768/32768 non-zero, mean=0.5933) |
| 8b | `postProcessTruncateLin` -- linear magnitude scaling | PASS |
| 9 | `updateDisplayedVolume` (3D `surf3Dwrite`, `hipCreateSurfaceObject`) | PASS (65278/65536 non-zero voxels written) |
| 10 | `updateDisplayedBscanFrame` -- 2D B-scan display update | PASS |
| 11a | `fillSinusoidalScanCorrectionCurve` -- sinusoidal scan LUT | PASS |
| 11b | `sinusoidalScanCorrection` -- sinusoidal scan correction | PASS |
| 12 | `cuda_bscanFlip` -- B-scan direction flip | PASS |
| 13a | `getMinimumVarianceMean` -- fixed-pattern noise determination | PASS |
| 13b | `meanALineSubtraction` -- FPN subtraction | PASS |

### GPU architecture
- gfx1201 (RX 9070 XT, RDNA4, wave32)
- ROCm 7.14.0a20260604 (TheRock nightly)
- hipFFT (from _rocm_sdk_devel)

### Notes
No code changes required for gfx1201. Same kernel test as Linux validation; only
adaptation was adding `-D_USE_MATH_DEFINES` (Windows M_PI) and copying TheRock
runtime DLLs beside the exe (standard Windows amdhip64 DLL-path workaround).
The `surf3Dwrite` path worked correctly on RDNA4 wave32 -- 65278/65536 voxels written.

## Validation 2026-06-19 (windows-gfx1101)

### Build

```bash
VENV="B:/develop/TheRock/external-builds/pytorch/.venv"
ROCM="$VENV/Lib/site-packages/_rocm_sdk_devel"
BUILD="B:/develop/moat/agent_space/octproz_win_gfx1101"

HIP_VISIBLE_DEVICES=1 "$ROCM/bin/hipcc" \
  --offload-arch=gfx1101 \
  -DUSE_HIP -D_USE_MATH_DEFINES -D__CUDACC_VER_MAJOR__=13 -DCUDART_VERSION=13000 -DWIN32 \
  -I"$ROCM/include" \
  -o "$BUILD/octproz_pipeline_test.exe" \
  "$BUILD/octproz_pipeline_test.hip" \
  -L"$ROCM/lib" -lhipfft
```

Runtime DLLs (amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll,
hiprtc-builtins0714.dll) from `_rocm_sdk_core/bin` and (hipfft.dll, rocfft.dll)
from `_rocm_sdk_libraries/bin` copied into the exe directory.

Build succeeded with one harmless warning (_USE_MATH_DEFINES redefined).

### Test results (16/16 pass)

GPU: AMD Radeon PRO V710 (gfx1101, warpSize=32, 27 CUs, 25.5 GB)
Run: `HIP_VISIBLE_DEVICES=1 ./octproz_pipeline_test.exe`

| # | Kernel / path | Result |
|---|---------------|--------|
| 1 | `inputToCufftComplex` -- raw 8-bit data to complex | PASS |
| 2 | `windowing` -- Hanning window application | PASS |
| 3 | `klinearization` (linear interpolation) -- k-space resampling | PASS |
| 4 | `klinearizationCubic` -- cubic Hermite resampling | PASS |
| 5 | `klinearizationLanczos` -- Lanczos-8 resampling | PASS |
| 6 | `klinearizationAndWindowing` -- combined resample+window | PASS |
| 7 | `hipfftExecC2C` (hipFFT 1D 1024-pt C2C IFFT, DC signal) | PASS |
| 8 | Full OCT pipeline: input->klinearization->windowing->IFFT->postProcessTruncateLog | PASS (32768/32768 non-zero, mean=0.5933) |
| 8b | `postProcessTruncateLin` -- linear magnitude scaling | PASS |
| 9 | `updateDisplayedVolume` (3D `surf3Dwrite`, `hipCreateSurfaceObject`) | PASS (65278/65536 non-zero voxels written) |
| 10 | `updateDisplayedBscanFrame` -- 2D B-scan display update | PASS |
| 11a | `fillSinusoidalScanCorrectionCurve` -- sinusoidal scan LUT | PASS |
| 11b | `sinusoidalScanCorrection` -- sinusoidal scan correction | PASS |
| 12 | `cuda_bscanFlip` -- B-scan direction flip | PASS |
| 13a | `getMinimumVarianceMean` -- fixed-pattern noise determination | PASS |
| 13b | `meanALineSubtraction` -- FPN subtraction | PASS |

### GPU architecture
- gfx1101 (Radeon PRO V710, RDNA3, wave32)
- ROCm 7.14 (TheRock nightly)
- hipFFT (from _rocm_sdk_devel)

### Notes
No code changes required for gfx1101. Same kernel test and build recipe as the
gfx1201 Windows validation. Results match gfx1201 exactly: 32768/32768 non-zero
pipeline output (mean=0.5933) and 65278/65536 surf3Dwrite voxels -- confirming
identical correctness on RDNA3 wave32 (27 CUs, 25.5 GB). GPU pinned with
HIP_VISIBLE_DEVICES=1.

## CUDA compile check (nvcc 12.8) at af845bad

CUDA no-regression compile of the fork at head_sha af845bad4edc1f6e23e7bdd8a0d61d24544bfeed (qmake, default non-`USE_HIP` path, so `pri/cuda.pri`), on a ROCm fleet host that carries a CUDA toolkit (no NVIDIA GPU; compile and link only). Toolkit: `Cuda compilation tools, release 12.8, V12.8.93`, host compiler gcc 13.3.0. The image has no Qt development files, so these Ubuntu noble packages, matching the installed 5.15.13 runtime, were fetched with `apt-get download` and extracted with `dpkg -x` into the job directory: qtbase5-dev, qtbase5-dev-tools, qt5-qmake, qt5-qmake-bin, libqt5opengl5-dev, qtchooser, libqt5printsupport5t64. A `qt.conf` next to qmake redirects the prefix there. `cuda.pri` was left as committed (its "CUDA 13.0" arch list is sm_75 through sm_120, all accepted by 12.8).

```
export PATH=<job>/deps/qt/usr/lib/qt5/bin:/usr/local/cuda/bin:$PATH
mkdir oct-build && cd oct-build
qmake ../OCTproZ/octproz_project/octproz_project.pro && make -k -j32
```

Result: FAIL with the documented flags, inherited from upstream and not introduced by the port. nvcc stops on `cuda_code.cu` (`Makefile:722: cuda_code_cuda.o`) with six static assertions in the Qt headers, the first being:

```
qt5/QtCore/qnamespace.h:272:16: error: static assertion failed: (std::is_same<decltype(std::declval<TextElideMode>() | std::declval<TextFlag>()), int>::value)
  272 |     Q_DECLARE_MIXED_ENUM_OPERATORS_SYMMETRIC(int, TextElideMode, TextFlag)
```

(The other five are the `&` and `^` operators and the reversed operand order.) Upstream's tip, spectralcode/OCTproZ 3984974f089a73233e7d10338aa92c8e1acb96de, built the same way fails identically with the same six errors on its own `cuda_code.cu`. The trigger is `NVCC_OPTIONS = --use_fast_math -std=c++11` in `cuda.pri` combined with Qt 5.15.13's headers under nvcc 12.8. Re-running that exact nvcc command with only `-std=c++11` changed: `-std=c++14` gives the same six errors, while `-std=c++17` compiles `cuda_code.cu` cleanly (exit 0, no errors) for all ten gencode targets, on both the fork and upstream.

With the c++17 object in place, the rest of the CUDA target was finished in the same build tree: `make -j32` in `oct-build/octproz` with `LIBRARY_PATH=/usr/local/cuda/lib64/stubs`, because this host has no NVIDIA driver and therefore no `libcuda.so` for `-lcuda`. Every host C++ file of the octproz target compiled, including `glwindow2d.h`/`kernels.h` with the port's new `cuda_to_hip.h` include and `USE_HIP` guards on the CUDA path. `OCTproZ` linked (4.27 MB, identical in size to upstream's), and `cuobjdump` shows SASS for sm_75, 80, 86, 87, 89, 90, 100 and 120. The devkit and 8 of the 10 plugins built. Two plugins could not be configured because the image lacks those Qt modules: `octproz-camera-extension` ("Unknown module(s) in QT: multimedia multimediawidgets") and `octproz-socket-stream-extension` ("Unknown module(s) in QT: websockets"). Neither contains CUDA code, and the port does not touch them.

Verdict: the port introduces no CUDA compile regression. The `-std=c++11` failure is upstream's own issue with current Qt 5.15 and nvcc 12.8. The `cuda-compile` telemetry lines for this check record exit 0 because the wrapper script echoed the make status instead of returning it; the real outcome is the one described above.
