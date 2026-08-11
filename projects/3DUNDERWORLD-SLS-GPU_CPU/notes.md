# 3DUNDERWORLD-SLS-GPU_CPU notes

ROCm/HIP port. Lead: linux-gfx90a (CDNA2, wave64), validated on GPU ordinal 2
(MI250X). Strategy A (compat header + `enable_language(HIP)` + `.cu` LANGUAGE HIP).

## Build classification + strategy

Pure CMake (no Torch). Legacy `find_package(CUDA)` + `cuda_add_library` /
`cuda_add_executable`, gated on `ENABLE_CUDA` (default OFF). -> Strategy A.

CUDA surface is tiny and clean: cudaMalloc/Free/Memcpy/Memset, H2D/D2H,
cudaPeekAtLastError, cudaGetErrorString, cudaError_t/cudaSuccess, cudaEvent_*
(profiling only), atomicInc (bucket insert), __threadfence + a trap. No
__shfl/__ballot/warpSize, no textures/surfaces, no curand/cublas/cufft/thrust.

## Existing AMD support

None (no HIP path, no OpenCL/Vulkan/SYCL backend, no stale ROCm branch). Genuine
CUDA->HIP port.

## Changes (all USE_HIP-guarded; CUDA path byte-for-byte unchanged)

1. `src/lib/ReconstructorCUDA/cuda_to_hip.h` (new) -- the single HIP-aware file.
   On USE_HIP: includes `<cstring>`/`<cstdlib>` then `<hip/hip_runtime.h>` and
   aliases the cuda* surface to hip*. Else plain `<cuda_runtime.h>`.
   Force-included (CMake `-include`) on the HIP GPU targets so it precedes GLM.
2. `CUDA_Error.cuh` -- include `cuda_to_hip.h` instead of bare `<cuda_runtime.h>`.
3. `DynamicBits.cuh` -- `asm("trap;")` (NVPTX, illegal on amdgcn) -> `gpuTrap()`,
   a per-backend macro in `cuda_to_hip.h`: `__builtin_trap()` on HIP,
   `asm("trap;")` on CUDA. Never-taken overflow guard; no output effect on
   either backend. (Corrected at 7dc3a24 -- the first attempt used a bare
   unconditional `__builtin_trap()`, which nvcc rejects in device code and
   which broke the CUDA build; see the 2026-08-09 section.)
4. `ReconstructorCUDA.cu` / `FileReaderCUDA.cu` -- guard the NVIDIA-only
   `<device_launch_parameters.h>` include behind `!USE_HIP` (HIP provides those
   builtins intrinsically).
5. CMake: root adds `USE_HIP` option (+ unified `USE_GPU` flag); on HIP
   `enable_language(HIP)`, arch from `${CMAKE_HIP_ARCHITECTURES}` (default
   gfx90a ONLY when unset -- never a literal). `ReconstructorCUDA/CMakeLists.txt`
   and `src/app/CMakeLists.txt` mark the existing `.cu` `LANGUAGE HIP`, set
   `HIP_ARCHITECTURES`, `cxx_std_17`, and force-include the compat header. CUDA
   branch keeps `cuda_add_*` unchanged.
6. Bit-rot repair (see Quirks) -- host-side, required for the GPU path to build
   on ANY backend: repoint `FileReaderCUDA` from the deleted `FileReader` /
   `core/FileReader.h` to the current `ImageFileProcessor`; make
   `ReconstructorCUDA` a standalone class (owns cameras_/projector_) instead of
   deriving the refactored pure-interface `Reconstructor`.

## Fault classes

- wave64 / warp size: NONE apply. No warp primitives, no hardcoded 32, no
  warp-sized shared arrays. Grid-stride, per-thread-independent kernels. (Implies
  gfx1100/gfx1151 RDNA wave32 should pass with no delta.)
- NVPTX inline asm: `asm("trap;")` -> per-backend `gpuTrap()` (fix #3). Both
  spellings are backend-only -- neither compiles under the other toolchain.
- OOB reads: audited, safe (add2Bucket clamps bktIdx; buildBuckets checks
  projector bounds + mask; color idx are in-bounds pixel indices; bucket reads
  loop `i < count_[idx]` with `count_ <= MAX_CNT_PER_BKT_`). No stencil/neighbor
  gathers, so no clamp fix needed.
- OOB write, upstream, NOT fixed by this port: `ReconstructorCUDA.cuh:45` passes
  `MAX_CNT_PER_BKT_` to `atomicInc` where the idiom needs `MAX_CNT_PER_BKT_-1`,
  so the 111th insert into a bucket returns 110 and writes slot 0 of bucket
  `bktIdx+1` -- one uint past the allocation for the last bucket. Verbatim in
  the pre-port merge-base c87fe37 and identical on both toolchains, so it is
  upstream behaviour, out of scope here, and registered as deferred work
  `sls-gpu-bucket-atomicinc-overrun` for the maintainer round. An earlier note
  here claimed the wrap kept the write in the row and cited the clean gfx90a run
  as confirmation; both were wrong. A clean run cannot confirm it: reaching the
  case needs >110 camera pixels decoding into one projector cell (the alexander
  dataset never does), and a one-uint overrun of a ~346 MB device allocation
  stays inside the same page, so it corrupts silently instead of faulting.
- atomicInc on managed memory dropped-RMW class: N/A (plain cudaMalloc device
  memory; only int/uint atomicMin/atomicMax are in that class, not atomicInc).

## Quirks (also see PORTING_GUIDE changelog)

- GLM (0.9.9.8, libglm-dev) only emits `__host__ __device__` on its math
  functions when it detects the NVIDIA CUDA compiler via `__CUDACC__` +
  `CUDA_VERSION>=7000` (glm/simd/platform.h); hipcc/clang defines neither, so
  every glm:: call (dot/length/normalize/mat*vec, used in the device
  triangulation helpers) is host-only and the kernels fail with "call to
  __host__ function from __device__ function". GLM's qualifier macros
  (GLM_FUNC_QUALIFIER etc.) are redefined UNCONDITIONALLY in detail/setup.hpp, so
  pre-defining them does NOT win. The working fix: in the compat header, AFTER
  `<hip/hip_runtime.h>` is fully parsed, define `__CUDACC__`, `CUDA_VERSION 8000`,
  and `GLM_FORCE_CUDA`, then let GLM be included (transitively) afterwards -- this
  steers GLM into its CUDA path so its qualifiers become `__device__ __host__`,
  matching how a real CUDA build gets device-callable glm. Force-include the
  compat header on the HIP targets so it precedes GLM. (Defining __CUDACC__ after
  the HIP runtime is parsed does not disturb HIP.)
- memcpy/memset in device helpers: inside a .cu compiled as HIP, these resolve to
  HIP device overloads once hip_runtime is in scope; include <cstring>/<cstdlib>
  BEFORE hip_runtime in the compat header so the libc host decls remain available
  (matches the gpuRIR lesson).
- Upstream GPU-path bit-rot: the shipped `FileReaderCUDA`/`ReconstructorCUDA`
  reference a `FileReader` base and `core/FileReader.h` that were renamed
  (`Camera->ImageProcessor`, `FileReader->ImageFileProcessor`, commit 82078e2)
  and a pre-refactor `Reconstructor` base with `cameras_`/`projector_`/`addCamera`
  that was later reduced to a pure `reconstruct(Buckets)` interface (23a06f1). So
  the CUDA GPU build did NOT compile on stock nvcc either. A half-finished
  `ImageFileProcessorCUDA.cuh` (header only, unused) shows the intended rename.
  Fixed minimally on the host side (fix #6) so the GPU pipeline -- which builds
  buckets on-device and is a standalone parallel path -- compiles and runs.

## Build (gfx90a, GPU 2)

```bash
cd projects/3DUNDERWORLD-SLS-GPU_CPU/src
# deps: apt install libglm-dev ; OpenCV (libopencv-dev) already present (4.6.0)
# data: git submodule path `data` is not in the dev HEAD tree; clone directly:
#   git clone --depth 1 https://github.com/theICTlab/3DUNDERWORLD-SLS-DATA.git data
rm -rf build_hip && mkdir build_hip
HIP_VISIBLE_DEVICES=2 cmake -S . -B build_hip \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DGTEST=ON
HIP_VISIBLE_DEVICES=2 cmake --build build_hip -j$(nproc)
# Outputs: build_hip/bin/{SLS,SLS_GPU}, build_hip/test/runCPUTest
# SLS_GPU links libamdhip64.so.7 and embeds 3 gfx90a code objects (roc-obj-ls).
```

For a follower arch (RDNA wave32), only the configure changes -- no source edit:
`-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100` (or gfx1151).

## Validation (real GPU, GPU 2 = MI250X gfx90a) -- PASS

Dataset: shipped `alexander` (1024x768 projector, 42 jpg/cam) -- the README demo
and gtest dataset.

```bash
DATA=projects/3DUNDERWORLD-SLS-GPU_CPU/src/data/alexander
BIN=projects/3DUNDERWORLD-SLS-GPU_CPU/src/build_hip/bin
# GPU run (x2 for determinism), CPU reference run -- same args
HIP_VISIBLE_DEVICES=2 $BIN/SLS_GPU --leftcam=$DATA/leftCam/dataset1 \
  --rightcam=$DATA/rightCam/dataset1 \
  --leftconfig=$DATA/leftCam/calib/output/calib.xml \
  --rightconfig=$DATA/rightCam/calib/output/calib.xml \
  --output=./output.ply --format=jpg --width=1024 --height=768
$BIN/SLS  ... (same args)   # CPU reference
```

Results (comparator: agent_space/sls_val/compare.py, not committed):
- Point counts identical across all runs: GPU run1 = GPU run2 = CPU = 146064.
- GPU vs CPU correspondence: 100% coverage both directions @tol=10 AND @tol=0.5;
  nearest-neighbor mean 3.2e-5, p99.9 1.0e-3, max 1.0e-3 world units. The 1e-3
  ceiling is the ASCII PLY print precision (6 sig figs) -- i.e. the GPU cloud is
  identical to the CPU reference to the file's representable precision. Project's
  own pass tolerance is MAX_DIFF=10, so this is ~4 orders of magnitude inside it.
- Determinism (GPU run1 vs run2): same 146064 points; NN mean 1.8e-5, max 1.0e-3
  (print-precision ceiling). Per-axis sorted distributions agree to 1e-3. The
  only run-to-run variation is last-ASCII-digit (~1e-4) float jitter from the
  bucket-fill order being atomicInc-dependent, which reorders the per-bucket
  avgPoint float sum (float non-associativity). Geometry/topology and point set
  are stable; this is well below output precision and the test tolerance.
- CPU gtest `runCPUTest` (Arch + Alexander vs in-tree reference clouds,
  MAX_DIFF=10): both PASS. No regression in non-GPU tests.

Verdict: HIP GPU reconstruction is numerically equal to the CPU reference (to
print precision) and to the gtest ground-truth, with stable correspondences and
output-precision determinism on gfx90a. Validated.

## Outstanding / follower notes

- gfx1100: VALIDATED 2026-05-30 (see section below).
- gfx1101: VALIDATED 2026-06-05 (see section below).
- gfx1151: host retired; validation superseded by gfx1101/gfx1201.
- gfx1201: VALIDATED 2026-06-06 (see section below).
- The `data` submodule gitlink is missing from the dev HEAD tree (`.gitmodules`
  references it but `git ls-tree HEAD data` is empty); clone the DATA repo
  directly as above. Not a port issue.

## Validation 2026-05-30 (gfx1100, ROCm 7.2.1)

Platform: linux-gfx1100. GPU: 2x AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32). ROCm 7.2.1, hipcc/clang++ 22.0.0. SHA validated: 3a506a202f999f97bfe93b080c55e188bd7a0e35.

No source or CMake changes were needed; only `-DCMAKE_HIP_ARCHITECTURES=gfx1100` differs from the gfx90a build.

### Build commands

```bash
cd projects/3DUNDERWORLD-SLS-GPU_CPU/src
# dep install (GLM not present on this host)
sudo apt-get install -y libglm-dev
# data: clone directly (submodule not in dev HEAD tree)
git clone --depth 1 https://github.com/theICTlab/3DUNDERWORLD-SLS-DATA.git data
rm -rf build_hip && mkdir build_hip
HIP_VISIBLE_DEVICES=0 cmake -S . -B build_hip \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release -DGTEST=ON
# wrapped with timeit:
bash utils/timeit.sh 3DUNDERWORLD-SLS-GPU_CPU compile -- \
  cmake --build build_hip -j$(nproc)
```

Build result: success, warnings only (nodiscard on cudaEvent aliases, unused variable in CPU path -- pre-existing). 3 HIP translation units compiled for gfx1100.

### Code-object evidence

```
roc-obj-ls build_hip/bin/SLS_GPU
# 3 code objects, all hipv4-amdgcn-amd-amdhsa--gfx1100; no gfx90a.
1  hipv4-amdgcn-amd-amdhsa--gfx1100  (6640 bytes)
2  hipv4-amdgcn-amd-amdhsa--gfx1100  (12392 bytes)
3  hipv4-amdgcn-amd-amdhsa--gfx1100  (27128 bytes)
```

### GPU reconstruction result

```bash
DATA=projects/3DUNDERWORLD-SLS-GPU_CPU/src/data/alexander
BIN=projects/3DUNDERWORLD-SLS-GPU_CPU/src/build_hip/bin
# Run 1
bash utils/timeit.sh 3DUNDERWORLD-SLS-GPU_CPU test -- bash -c \
  "HIP_VISIBLE_DEVICES=0 $BIN/SLS_GPU --leftcam=$DATA/leftCam/dataset1 \
   --rightcam=$DATA/rightCam/dataset1 \
   --leftconfig=$DATA/leftCam/calib/output/calib.xml \
   --rightconfig=$DATA/rightCam/calib/output/calib.xml \
   --output=/tmp/output_gpu_run1.ply --format=jpg --width=1024 --height=768"
# Run 2 (determinism)
HIP_VISIBLE_DEVICES=0 $BIN/SLS_GPU ... --output=/tmp/output_gpu_run2.ply ...
# CPU reference
$BIN/SLS ... --output=/tmp/output_cpu.ply ...
```

Point counts: GPU run1 = GPU run2 = CPU = 146064 (matches gfx90a reference exactly).

Coordinate stats (GPU run1):
- x: min=-119.898, max=135.822, mean=45.003
- y: min=-117.639, max=208.327, mean=16.143
- z: min=-116.617, max=134.884, mean=-55.124
- No NaN/Inf detected.

Matches gfx90a coordinate stats within print precision.

GPU vs CPU nearest-neighbor correspondence:
- CPU->GPU: mean=3.7e-5, p99.9=1.0e-3, max=2.5e-3; 100% coverage @tol=10.0 and @tol=0.5.
- GPU->CPU: same (symmetric). Reconstruction is numerically identical to CPU reference to file print precision.

Determinism (run1 vs run2, set-based NN):
- count match: True (146064 each)
- run1->run2: mean=4.7e-6, max=1.0e-3; 100% coverage @tol=1.0, 99.96% @tol=1e-3.
- The residual max (1.0e-3) is the ASCII PLY 6-sig-fig print-precision ceiling; identical to gfx90a behavior. The point SET is stable; only bucket-fill ordering (atomicInc) varies, producing last-digit float jitter -- well below the project's MAX_DIFF=10 tolerance.

### CPU gtest suite

```bash
cd build_hip && ctest --output-on-failure
# 3/3 passed: RunCPUTest.Arch, RunCPUTest.Alexander, CPU_TEST (44 sec total)
```

No regression in non-GPU tests.

### Verdict

PASS. gfx1100 (wave32, RDNA3) produces identical reconstruction results to gfx90a and CPU reference. No source changes were needed; the port is wave-size-agnostic. No fork interaction performed.

## Validation 2026-06-05 (windows-gfx1101, ROCm 7.14)

Platform: windows-gfx1101. GPU: Radeon PRO V710 (gfx1101, RDNA3, wave32). ROCm 7.14.0a20260604 (clang++ 23.0.0). SHA validated: 633065b857387209d619468a0f765ca7460c1ccd (commit on top of 3a506a2 adding the WIN32 OpenCV compat path).

### Windows-specific adaptations

1. OpenCV include path: upstream sources use `<opencv4/opencv2/...>` (Linux /usr/include/opencv4 layout). The Windows prebuilt OpenCV puts headers at `build/include/opencv2/` without the `opencv4/` prefix. Fixed by creating a compat directory with a junction `opencv4/ -> build/include/` and adding `-DOPENCV4_COMPAT_DIR=<compat>` to cmake (a new WIN32 conditional in CMakeLists.txt committed as 633065b).

2. cxxabi.h: `cmdline.h` (third-party header in the project) includes `<cxxabi.h>` for `abi::__cxa_demangle`. Not available in MSVC ABI (which clang++ targets on Windows). Provided a Windows stub `cxxabi.h` in the compat directory -- demangle returns the raw mangled name, acceptable for command-line error messages.

3. gtest: the test CMakeLists expects `libgtest.a` (Unix static lib naming); Windows links `gtest.lib`. Built without `-DGTEST=ON` on this host. CPU tests are already validated on Linux; skipping them here does not affect GPU correctness validation.

4. DLL loader order: copied TheRock DLLs (amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc*.dll) into the exe directory so they win over System32's Adrenalin amdhip64 (exe-dir beats System32 in the Windows DLL search order).

5. OpenCV DLL (opencv_world4110.dll) copied to exe directory.

### Build commands

```bash
SRC="B:/develop/moat/projects/3DUNDERWORLD-SLS-GPU_CPU/src"
ROCM="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
GLM_DIR="B:/develop/moat/agent_space/glm/glm"       # GLM 0.9.9.8 headers extracted here
OPENCV_DIR="B:/develop/opencv-install/extracted/opencv/build"
OPENCV4_COMPAT="B:/develop/moat/agent_space/opencv4_compat"  # opencv4/ junction + cxxabi.h stub
MSVC_VER="14.44.35207"
WINSDK_VER="10.0.26100.0"
MSVC_ROOT="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/$MSVC_VER"
WINSDK_ROOT="C:/Program Files (x86)/Windows Kits/10"

export LIB="$MSVC_ROOT/lib/x64;$WINSDK_ROOT/Lib/$WINSDK_VER/ucrt/x64;$WINSDK_ROOT/Lib/$WINSDK_VER/um/x64"
export INCLUDE="$MSVC_ROOT/include;$WINSDK_ROOT/Include/$WINSDK_VER/ucrt;$WINSDK_ROOT/Include/$WINSDK_VER/um;$WINSDK_ROOT/Include/$WINSDK_VER/shared"
export HIP_DEVICE_LIB_PATH="$ROCM/lib/llvm/amdgcn/bitcode"
export HIP_VISIBLE_DEVICES=0

# data: clone directly (submodule not in dev HEAD tree)
# git clone --depth 1 https://github.com/theICTlab/3DUNDERWORLD-SLS-DATA.git data

cmake -S "$SRC" -B "$SRC/build_gfx1101" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
  -DCMAKE_C_COMPILER="$ROCM/lib/llvm/bin/clang.exe" \
  -DCMAKE_CXX_COMPILER="$ROCM/lib/llvm/bin/clang++.exe" \
  -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/clang++.exe" \
  -DCMAKE_PREFIX_PATH="$ROCM" \
  -DGLM_INCLUDE_DIR="$GLM_DIR" \
  -DOpenCV_DIR="$OPENCV_DIR" \
  -DOpenCV_ARCH=x64 -DOpenCV_RUNTIME=vc16 \
  -DOPENCV4_COMPAT_DIR="$OPENCV4_COMPAT" \
  -DGTEST=OFF

bash utils/timeit.sh 3DUNDERWORLD-SLS-GPU_CPU compile -- cmake --build "$SRC/build_gfx1101" -j64

# Copy runtime DLLs to exe dir
ROCM_CORE="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_core"
for dll in amdhip64_7.dll amd_comgr.dll rocm_kpack.dll hiprtc-builtins0714.dll hiprtc0714.dll; do
    cp "$ROCM_CORE/bin/$dll" "$SRC/build_gfx1101/bin/"
done
cp B:/develop/opencv-install/extracted/opencv/build/x64/vc16/bin/opencv_world4110.dll "$SRC/build_gfx1101/bin/"
```

Build result: success, warnings only (fopen deprecated in DynamicBitset.cpp/Log.cpp -- pre-existing; nodiscard on cudaEvent aliases -- same as Linux). Binary embeds `hipv4-amdgcn-amd-amdhsa--gfx1101` code objects (confirmed via `strings SLS_GPU.exe | grep gfx1101`).

### GPU reconstruction result

```bash
DATA="B:/develop/moat/projects/3DUNDERWORLD-SLS-GPU_CPU/src/data/alexander"
BIN="B:/develop/moat/projects/3DUNDERWORLD-SLS-GPU_CPU/src/build_gfx1101/bin"
export HIP_VISIBLE_DEVICES=0

# Run 1
bash utils/timeit.sh 3DUNDERWORLD-SLS-GPU_CPU test -- \
  "$BIN/SLS_GPU.exe" \
    --leftcam="$DATA/leftCam/dataset1" --rightcam="$DATA/rightCam/dataset1" \
    --leftconfig="$DATA/leftCam/calib/output/calib.xml" \
    --rightconfig="$DATA/rightCam/calib/output/calib.xml" \
    --output="C:/Temp/output_gpu_run1.ply" --format=jpg --width=1024 --height=768
# Run 2 (determinism): same with --output=...run2.ply
# CPU reference: SLS.exe with same args
```

Point counts: GPU run1 = GPU run2 = CPU = 146064 (matches gfx90a and gfx1100 reference exactly).

Coordinate stats (GPU run1):
- x: min=-119.898, max=135.822, mean=45.003
- y: min=-117.639, max=208.327, mean=16.143
- z: min=-116.617, max=134.884, mean=-55.124
- No NaN/Inf detected. Matches gfx90a/gfx1100 stats exactly.

GPU vs CPU nearest-neighbor correspondence (set-based NN, compare.py):
- CPU->GPU: mean=3.69e-5, p99.9=1.0e-3, max=2.52e-3; 100% coverage @tol=0.5 and @tol=10.
- GPU->CPU: same (symmetric).

Determinism (run1 vs run2):
- count match: True (146064 each)
- run1->run2: mean=5.23e-6, max=1.0e-3; 100% coverage @tol=0.5 and @tol=10.
- Residual max (1.0e-3) is ASCII PLY 6-sig-fig print-precision ceiling; identical to gfx90a/gfx1100 behavior.

### Verdict

PASS. gfx1101 (wave32, RDNA3) produces numerically identical reconstruction results to gfx90a, gfx1100, and the CPU reference. The only changes from the Linux build are the Windows-specific include-path workarounds (compat dir for opencv4/ layout and cxxabi.h stub) and the DLL copy step. GPU kernels are unchanged.

## Validation 2026-06-06 (windows-gfx1201, ROCm 7.14)

Platform: windows-gfx1201. GPU: RX 9070 XT (gfx1201, RDNA4, wave32). ROCm 7.14.0a20260604 (clang++ 23.0.0). SHA validated: 633065b857387209d619468a0f765ca7460c1ccd. HIP_VISIBLE_DEVICES=0 (gfx1101 absent from bus; gfx1201 enumerated at device 0).

No source or CMake changes were needed; only `-DCMAKE_HIP_ARCHITECTURES=gfx1201` differs from the gfx1101 build. Same Windows-specific adaptations apply (OPENCV4_COMPAT_DIR, cxxabi.h stub, DLL copy).

### Build commands

```bash
SRC="B:/develop/moat/projects/3DUNDERWORLD-SLS-GPU_CPU/src"
ROCM="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
GLM_DIR="B:/develop/moat/agent_space/glm/glm"
OPENCV_DIR="B:/develop/opencv-install/extracted/opencv/build"
OPENCV4_COMPAT="B:/develop/moat/agent_space/opencv4_compat"
MSVC_VER="14.44.35207"
WINSDK_VER="10.0.26100.0"
MSVC_ROOT="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/$MSVC_VER"
WINSDK_ROOT="C:/Program Files (x86)/Windows Kits/10"

export LIB="$MSVC_ROOT/lib/x64;$WINSDK_ROOT/Lib/$WINSDK_VER/ucrt/x64;$WINSDK_ROOT/Lib/$WINSDK_VER/um/x64"
export INCLUDE="$MSVC_ROOT/include;$WINSDK_ROOT/Include/$WINSDK_VER/ucrt;$WINSDK_ROOT/Include/$WINSDK_VER/um;$WINSDK_ROOT/Include/$WINSDK_VER/shared"
export HIP_DEVICE_LIB_PATH="$ROCM/lib/llvm/amdgcn/bitcode"
export HIP_VISIBLE_DEVICES=0

cmake -S "$SRC" -B "$SRC/build_gfx1201" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_C_COMPILER="$ROCM/lib/llvm/bin/clang.exe" \
  -DCMAKE_CXX_COMPILER="$ROCM/lib/llvm/bin/clang++.exe" \
  -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/clang++.exe" \
  -DCMAKE_PREFIX_PATH="$ROCM" \
  -DGLM_INCLUDE_DIR="$GLM_DIR" \
  -DOpenCV_DIR="$OPENCV_DIR" \
  -DOpenCV_ARCH=x64 -DOpenCV_RUNTIME=vc16 \
  -DOPENCV4_COMPAT_DIR="$OPENCV4_COMPAT" \
  -DGTEST=OFF

bash utils/timeit.sh 3DUNDERWORLD-SLS-GPU_CPU compile -- cmake --build "$SRC/build_gfx1201" -j64

# Copy runtime DLLs to exe dir
ROCM_CORE="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_core"
for dll in amdhip64_7.dll amd_comgr.dll rocm_kpack.dll hiprtc-builtins0714.dll hiprtc0714.dll; do
    cp "$ROCM_CORE/bin/$dll" "$SRC/build_gfx1201/bin/"
done
cp B:/develop/opencv-install/extracted/opencv/build/x64/vc16/bin/opencv_world4110.dll "$SRC/build_gfx1201/bin/"
```

Build result: success, warnings only (fopen deprecated, nodiscard on cudaEvent aliases -- same as gfx1101 build). Binary embeds `hipv4-amdgcn-amd-amdhsa--gfx1201` code objects (confirmed via `strings SLS_GPU.exe | grep gfx1201`).

### GPU reconstruction result

```bash
DATA="B:/develop/moat/projects/3DUNDERWORLD-SLS-GPU_CPU/src/data/alexander"
BIN="B:/develop/moat/projects/3DUNDERWORLD-SLS-GPU_CPU/src/build_gfx1201/bin"
export HIP_VISIBLE_DEVICES=0

# Run 1
bash utils/timeit.sh 3DUNDERWORLD-SLS-GPU_CPU test -- \
  "$BIN/SLS_GPU.exe" \
    --leftcam="$DATA/leftCam/dataset1" --rightcam="$DATA/rightCam/dataset1" \
    --leftconfig="$DATA/leftCam/calib/output/calib.xml" \
    --rightconfig="$DATA/rightCam/calib/output/calib.xml" \
    --output="C:/Temp/output_gpu_gfx1201_run1.ply" --format=jpg --width=1024 --height=768
# Run 2 (determinism): same with --output=...run2.ply
# CPU reference: SLS.exe with same args
```

Point counts: GPU run1 = GPU run2 = CPU = 146064 (matches gfx90a, gfx1100, and gfx1101 reference exactly).

Coordinate stats (GPU run1):
- x: min=-119.898, max=135.822, mean=45.003
- y: min=-117.639, max=208.327, mean=16.143
- z: min=-116.617, max=134.885, mean=-55.124
- No NaN/Inf detected. Matches all prior platforms exactly.

GPU vs CPU nearest-neighbor correspondence (set-based NN, compare.py):
- CPU->GPU: mean=3.69e-5, p99.9=1.0e-3, max=2.52e-3; 100% coverage @tol=0.5 and @tol=10.
- GPU->CPU: same (symmetric). Identical to gfx1101 stats.

Determinism (run1 vs run2):
- count match: True (146064 each)
- run1->run2: mean=6.08e-6, max=1.0e-3; 100% coverage @tol=0.5 and @tol=10.
- Residual max (1.0e-3) is ASCII PLY 6-sig-fig print-precision ceiling; identical to all prior platforms.

### Verdict

PASS. gfx1201 (RDNA4, wave32) produces numerically identical reconstruction results to gfx90a, gfx1100, gfx1101, and the CPU reference. No source changes were needed; the port is wave-size-agnostic and arch-independent across all four validated platforms.

## PR conflict rebase 2026-07-02 (gfx90a, ROCm 7.2.1)

Upstream PR #33 went stale: the base branch `dev` advanced from 169f60a to
c87fe37 by merging upstream PR #32 (bashtavenko/synch -- a new `sync_main`
utility, an abseil FetchContent dependency, and a project-wide clang-format
pass over the two CMakeLists the port also edits). GitHub reported the PR
CONFLICTING/DIRTY.

Rebased moat-port (was 4584f0d) onto upstream/dev (c87fe37). Two conflicts,
both in CMake and both purely from upstream's clang-format reformat colliding
with our GPU-backend edits; no logic overlap:
- root `CMakeLists.txt`: kept our USE_HIP / USE_GPU backend-selection block
  (upstream only reformatted the surrounding CUDA detection whitespace).
- `src/app/CMakeLists.txt`: kept our USE_GPU/USE_HIP SLS_GPU target block;
  upstream's new `sync_main` target + abseil FetchContent merged cleanly and
  are preserved.
The net port diff vs upstream/dev is unchanged (same 11 files, byte-identical
port content). New head: 5eda3fd. Pushed with --force-with-lease (sanctioned
history rewrite for conflict resolution); this updates the open PR.

Rebuilt and re-validated on gfx90a (GPU 2, MI250X) at 5eda3fd -- the merge
brought new host code (sync_main/abseil) into the build, so a full build+run
was warranted, not just a compile check:
- Build: `-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DGTEST=ON`, success;
  SLS_GPU embeds 3 hipv4-amdgcn-amd-amdhsa--gfx90a code objects; new sync_main
  target links (unrelated abseil host util, warnings only).
- GPU reconstruction (alexander, 1024x768), 2 runs + CPU reference: all three
  = 146064 points. GPU vs CPU NN mean 3.59e-5, max 2.52e-3, p99.9 1.0e-3, 100%
  coverage @0.5 and @10. Determinism run1 vs run2: NN mean 1.77e-5, max 1.01e-3,
  100% @0.5. Identical to the original gfx90a record (print-precision ceiling).
- CPU gtest `ctest`: 3/3 pass (RunCPUTest.Arch, RunCPUTest.Alexander, CPU_TEST).

Dependency setup on this host (fresh checkout): apt universe was disabled;
after `apt-get update` installed `libopencv-dev` (4.6.0) and `libglm-dev`
(0.9.9.8), matching the original environment. Data cloned directly from the
DATA repo (submodule gitlink still absent from dev HEAD).

`advance-head 5eda3fd` reclassified the three completed followers (gfx1100,
gfx1101, gfx1201) to `revalidate`: the rebased base is a source-class change
(new sync_main.cc, abseil, reformatted CMake), so they should rebuild on the
new base to reconfirm, even though the GPU code objects are unchanged. The
gfx90a lead stays `pr-open` (its validated_sha stays frozen at PR-open time by
design); it was nonetheless re-validated on real GPU at 5eda3fd as recorded
above.

## Validation 2026-07-02 (linux-gfx1100, ROCm 7.2.1) -- revalidate at 5eda3fd

Platform: linux-gfx1100. GPU: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), HIP_VISIBLE_DEVICES=0. ROCm 7.2.1, hipcc/clang++ 22.0.0. SHA validated: 5eda3fd973f6a6f5c4a2008073eced6257126315.

Reason for revalidate: moat-port was rebased onto upstream/dev (c87fe37) which added sync_main.cc, abseil FetchContent, and a clang-format pass over the CMakeLists files the port also edits. Old validated_sha (4584f0d) orphaned by the force-push, so binary-equivalence carry-forward was not possible; full GPU re-run required.

No source or CMake changes to the port were needed; only the base changed. Build flags identical to the 2026-05-30 gfx1100 validation.

### Build

```bash
cd projects/3DUNDERWORLD-SLS-GPU_CPU/src
git reset --hard origin/moat-port  # to 5eda3fd
rm -rf build_hip && mkdir build_hip
HIP_VISIBLE_DEVICES=0 cmake -S . -B build_hip \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release -DGTEST=ON
bash utils/timeit.sh 3DUNDERWORLD-SLS-GPU_CPU compile -- \
  cmake --build /path/to/build_hip -j$(nproc)
```

Build result: success, warnings only (fopen deprecated, nodiscard on cudaEvent aliases, sync_main missing return -- all pre-existing or upstream). New sync_main binary also built. Code objects in SLS_GPU:

```
roc-obj-ls build_hip/bin/SLS_GPU
1  hipv4-amdgcn-amd-amdhsa--gfx1100  (6640 bytes)
2  hipv4-amdgcn-amd-amdhsa--gfx1100  (12392 bytes)
3  hipv4-amdgcn-amd-amdhsa--gfx1100  (27128 bytes)
```

Code object sizes are identical to the 2026-05-30 gfx1100 validation.

### GPU reconstruction result

Point counts: GPU run1 = GPU run2 = CPU = 146064 (matches all prior platforms exactly).

Coordinate stats (GPU run1):
- x: min=-119.898, max=135.822, mean=45.003
- y: min=-117.639, max=208.327, mean=16.143
- z: min=-116.617, max=134.884, mean=-55.124
- No NaN/Inf detected. Matches all prior platforms exactly.

GPU vs CPU nearest-neighbor correspondence:
- CPU->GPU: mean=3.693e-5, p99.9=1.007e-3, max=2.515e-3; 100% coverage @tol=0.5 and @tol=10.
- GPU->CPU: same (symmetric).

Determinism (run1 vs run2, NN-based):
- count match: True (146064 each)
- run1->run2: mean=5.04e-6, max=1.017e-3; 100% coverage @tol=1.0, 99.96% @tol=1e-3.
- Residual max (1.017e-3) is ASCII PLY 6-sig-fig print-precision ceiling; identical to prior gfx1100 record.

### CPU gtest suite

```bash
cd build_hip && ctest --output-on-failure
# 3/3 passed: RunCPUTest.Arch, RunCPUTest.Alexander, CPU_TEST (44 sec total)
```

No regression in non-GPU tests.

### Verdict

PASS. gfx1100 (wave32, RDNA3) produces numerically identical reconstruction results to all prior platforms and to the CPU reference at 5eda3fd. The new upstream base (sync_main, abseil) builds and links correctly; the GPU code objects are unchanged (same byte sizes as 2026-05-30 validation).

## CI HIP build job 2026-07-06 (gfx90a, ROCm 7.2.1)

Maintainer v3c70r agreed to merge PR #33 and asked for a GitHub Action that
builds the libraries against HIP, to serve as a reference for configuring HIP
on a local machine. Added `.github/workflows/hip.yml` (job `build-hip`, name
"ROCm build (HIP)") on top of moat-port. New head: 20e55bd.

Compile-only, mirroring the existing CUDA job (c-cpp.yml): runs in the
`rocm/dev-ubuntu-24.04:7.2.4-complete` container, installs the same host deps
(libglm-dev, libopencv-dev, libtiff-dev, cmake, build-essential), configures
`-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++`,
and builds only the `SLS_GPU` target. No GPU run (hosted runners have no AMD
GPU). The arch is pinned explicitly because `enable_language(HIP)` auto-detects
the host GPU when unset and would error on a GPU-less runner. Steps and comments
are written to double as a local ROCm/HIP config reference, per the request.

Validated the exact configure+build commands on this gfx90a host (ROCm 7.2.1;
the container is 7.2.4, both provide HIP), no docker available here:

```bash
cd projects/3DUNDERWORLD-SLS-GPU_CPU/src
rm -rf build_ci_check
cmake -S . -B build_ci_check -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build build_ci_check --target SLS_GPU -j$(nproc)
```

Result: success (warnings only -- nodiscard on cudaEvent aliases). SLS_GPU
embeds 3 `hipv4-amdgcn-amd-amdhsa--gfx90a` code objects (roc-obj-ls), sizes
7336/13424/27136 -- matching the prior gfx90a validations.

Regression guard: `advance-head 20e55bd` classified hip.yml as doc-only; the
completed gfx1100 follower carried forward (validated_sha -> 20e55bd, no GPU
re-run). windows-gfx1101/gfx1201 were already `revalidate` from the 5eda3fd
rebase (not newly flipped by this yml); when revalidated they can carry the yml
delta forward via codeobj_diff since it changes no compiled output.

Upstream reply to v3c70r drafted but NOT posted (orchestrator halted all
upstream-visible communication pending Jeff's review). Draft returned to Jeff.

## Validation 2026-08-08 (linux-gfx90a, ROCm) -- revalidate at 3626150ec2f -- CUDA regression, validation-failed

Platform: linux-gfx90a. GPU: AMD Instinct MI250X (gfx90a, CDNA2, wave64),
HIP_VISIBLE_DEVICES=2 (confirmed via `rocm-smi`, index 2 = gfx90a). Revalidate
trigger: recorded `validated_sha` (4584f0deed39a6abec9f9b85861bafc66b44c935)
was force-pushed away by the 2026-07-02 rebase (object no longer resolvable,
`classify` fails with `bad object`); the "PR-open freeze" note that had kept
this arch's record pinned to that orphaned sha is not part of the current
process, so this run re-validates for real rather than trusting the freeze.

### HIP GPU build + test -- PASS

```bash
cd projects/3DUNDERWORLD-SLS-GPU_CPU/src
git clone --depth 1 https://github.com/theICTlab/3DUNDERWORLD-SLS-DATA.git data
rm -rf build_hip && mkdir build_hip
HIP_VISIBLE_DEVICES=2 cmake -S . -B build_hip \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DGTEST=ON -DCMAKE_BUILD_TYPE=Release
bash utils/timeit.sh 3DUNDERWORLD-SLS-GPU_CPU compile -- cmake --build build_hip -j$(nproc)
```

Build: success. `roc-obj-ls build_hip/bin/SLS_GPU`: 3 code objects, all
`hipv4-amdgcn-amd-amdhsa--gfx90a`, sizes 7336/13424/27136 -- byte-identical to
every prior gfx90a record.

GPU reconstruction (alexander dataset, 1024x768), 2 runs + CPU reference:
point counts GPU run1 = GPU run2 = CPU = 146064 (matches every prior platform
exactly). GPU vs CPU nearest-neighbor (own compare.py, brute-force cKDTree,
not committed): mean 3.609e-5, p99.9 1.005e-3, max 2.516e-3 world units, 100%
coverage @tol=0.5 and @tol=10. Determinism run1 vs run2: mean 1.812e-5, max
1.020e-3, 100% coverage @tol=0.5/@tol=10 -- matches the original gfx90a record
to the ASCII-PLY print-precision ceiling.

CPU gtest (`ctest --output-on-failure`): 3/3 pass (RunCPUTest.Arch 9.96s,
RunCPUTest.Alexander 26.28s, CPU_TEST 37.67s). No regression in non-GPU tests.

Jargon (`python3 utils/jargon.py --port 3DUNDERWORLD-SLS-GPU_CPU`): clean.
Documentation: README.md already documents `-DUSE_HIP=ON` /
`CMAKE_HIP_ARCHITECTURES` alongside the CUDA build instructions.

### CUDA no-regression gate -- FAIL (genuine port regression, not pre-existing)

Not previously recorded at any sha reachable from head, so due to run. Built
port head (3626150) as CUDA: `conda create -n cuda-12.8 -c nvidia
cuda-toolkit=12.8` (nvcc 12.8.93), host gcc-13, arch pinned `-arch=sm_80` via
`CUDA_NVCC_FLAGS` (legacy `FindCUDA`, no `CMAKE_CUDA_ARCHITECTURES` in this
project -- confirmed no `native` autodetect trap applies here). The conda
toolkit ships `include/` only under `targets/x86_64-linux/`, not directly
under the env root, so legacy `FindCUDA` can't locate `CUDA_INCLUDE_DIRS`
until a `bin/include/lib/lib64` symlink tree is built pointing at it
(`CUDA_TOOLKIT_ROOT_DIR=<that tree>`) -- worth carrying into the skill as a
FindCUDA-vs-conda-toolkit layout note.

```bash
cmake -S . -B build_cuda -DENABLE_CUDA=ON \
  -DCUDA_TOOLKIT_ROOT_DIR=<symlink-tree-over-conda-env> \
  -DCUDA_HOST_COMPILER=/usr/bin/gcc-13 -DCUDA_NVCC_FLAGS="-arch=sm_80" \
  -DCMAKE_BUILD_TYPE=Release
bash utils/timeit.sh 3DUNDERWORLD-SLS-GPU_CPU cuda-compile -- \
  cmake --build build_cuda --target SLS_GPU -j$(nproc)
```

Fails:

```
.../src/lib/ReconstructorCUDA/./DynamicBits.cuh(88): error: calling a
__host__ function("__builtin_trap") from a __device__ function
("SLS::Dynamic_Bitset_Array_GPU::to_uint const") is not allowed
```

Root cause: commit 513385e ("Add AMD GPU support to the GPU reconstructor")
changed `DynamicBits.cuh`'s never-taken overflow guard from `asm("trap;")` to
`__builtin_trap()` **unconditionally** (no `USE_HIP` guard), on the claim it
is "portable" and "output is unaffected on either backend." That's true for
HIP/clang (where `__builtin_trap()` is device-callable) but false for nvcc:
nvcc does not treat `__builtin_trap()` as device-callable, so the CUDA
`ENABLE_CUDA=ON` path -- which the commit explicitly claims stays "unchanged"
-- no longer compiles. Confirmed this is new-on-the-port, not pre-existing
bit-rot: `git show c87fe37:.../DynamicBits.cuh` (the upstream merge-base
before any port commit) still has `asm("trap;")` at that line, which compiles
cleanly under nvcc. This is the exact "fix added for HIP, not defined/kept for
the CUDA branch" fault class in the validator's own instructions.

The fix belongs to the porter: guard fix #3 behind `USE_HIP` (or
`__HIP_DEVICE_COMPILE__`) so nvcc keeps `asm("trap;")` and clang/HIP gets
`__builtin_trap()`, matching how every other change in this port is already
`USE_HIP`-guarded. Not fixed here -- validator does not edit the fork.

### Verdict

HIP GPU validation on gfx90a PASSES on every measure (build, code objects,
reconstruction accuracy, determinism, CPU gtest, jargon, docs). The CUDA
no-regression gate FAILS: this specific port commit breaks the CUDA build,
violating "the CUDA build must be a pure passthrough." State set to
`validation-failed` for linux-gfx90a with this finding; sent back to the
porter. No source/CMake changes were made to the fork by this validation run
(`git status` clean in `src/` aside from gitignored `build_*`/`data`
directories).

## Port fix 2026-08-09 (linux-gfx90a, ROCm 7.2.1) -- CUDA regression closed at 7dc3a24

Answers the 2026-08-08 `validation-failed` above. GPU: MI250X gfx90a,
`HIP_VISIBLE_DEVICES=3` (index 3 confirmed gfx90a via `rocm-smi`).

Root cause, confirmed by rebuilding 3626150 under nvcc before touching anything:
change #3 of 513385e replaced `asm("trap;")` in `DynamicBits.cuh` with an
UNCONDITIONAL `__builtin_trap()`. clang makes that device-callable; nvcc treats
it as `__host__`, so `-DENABLE_CUDA=ON` died with

```
DynamicBits.cuh(88): error: calling a __host__ function("__builtin_trap") from a
__device__ function("SLS::Dynamic_Bitset_Array_GPU::to_uint const") is not allowed
```

Neither spelling compiles on both toolchains (`trap;` has no amdgcn form,
`__builtin_trap()` is host-only under nvcc), so the choice is necessarily
per-backend.

Fix: a `gpuTrap()` macro in `cuda_to_hip.h` -- the compat header already owns the
backend condition and the rest of the cuda*->hip* mapping, so this keeps the
`#if` in one file instead of open-coding a second copy of it in `DynamicBits.cuh`.
Matches the project's own `gpuErrchk` naming. The CUDA branch expands to
upstream's exact text.

### CUDA no-regression -- PASS

nvcc 12.8.93 (`/opt/conda/envs/cuda-12.8`), gcc-13 host compiler, `-arch=sm_80`.
`CUDA_TOOLKIT_ROOT_DIR` points at the symlink tree over the conda env described
in the skill's FindCUDA-vs-conda note (`bin`, `include ->
targets/x86_64-linux/include`, `lib`, `lib64`).

```bash
CT=<symlink-tree>
cmake -S . -B build_cuda -DENABLE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=$CT \
  -DCUDA_HOST_COMPILER=/usr/bin/gcc-13 -DCUDA_NVCC_FLAGS="-arch=sm_80" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build_cuda -j$(nproc)     # full `all`, not just SLS_GPU
```

Full `all` target builds clean (SLS, SLS_GPU, calibrateCamera, generateGraycode,
sync_main). Only warning is pre-existing upstream (`sync_main.cc:24`, missing
return). `cuobjdump -sass bin/SLS_GPU | grep -c BPT` -> 3, so the trap really is
emitted rather than optimized away.

Byte-level proof the CUDA path is RESTORED, not merely compiling: PTX of
`DynamicBits.cu` from this branch vs from the pre-port merge-base c87fe37 is
IDENTICAL (`diff` exit 0, 9328 bytes each).

```bash
git archive c87fe37 src/lib/ReconstructorCUDA | tar -x -C base
nvcc -arch=sm_80 -ptx -I base/src/lib -I $SRC/src/lib -o base.ptx \
  base/src/lib/ReconstructorCUDA/DynamicBits.cu
nvcc -arch=sm_80 -ptx -I src/lib -o port.ptx src/lib/ReconstructorCUDA/DynamicBits.cu
diff base.ptx port.ptx
```

The base tree needs the PORT's include path as a SECOND `-I` (it has no
`core/Log.hpp` of its own under the archived subtree); base's own `-I` comes
first so its headers still win.

### ROCm gfx90a -- PASS (unchanged by the fix)

```bash
export HIP_VISIBLE_DEVICES=3
cmake -S . -B build_hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DGTEST=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_hip -j$(nproc)
ctest --test-dir build_hip --output-on-failure
```

`roc-obj-ls build_hip/bin/SLS_GPU`: 3 code objects, all
`hipv4-amdgcn-amd-amdhsa--gfx90a`, sizes 7336/13424/27136 -- identical to every
prior gfx90a record, i.e. the device code is untouched (expected: `gpuTrap()`
expands to the same `__builtin_trap()` it had). Links `libamdhip64.so.7`.

alexander 1024x768, 2 GPU runs + CPU reference: 146064 points in all three, no
NaN/Inf. GPU vs CPU nearest-neighbor both directions: mean 3.578e-5, p99.9
1.005e-3, max 2.516e-3 world units, 100% coverage @tol=0.5 and @tol=10.
Determinism run1 vs run2: mean 1.771e-5, max 1.010e-3, 100% coverage. Matches
the 2026-08-08 numbers to the ASCII-PLY print-precision ceiling.

`ctest`: 3/3 pass (RunCPUTest.Arch 10.00s, RunCPUTest.Alexander 26.37s, CPU_TEST
39.54s).

Jargon scan clean. README already documents the ROCm build; no doc change needed.

Lesson promoted to the skill (fault-classes.md, extending the entry the failing
run had already opened): the repair SHAPE (per-backend macro in the compat header
rather than an `#if` at the use site) and the PTX-diff-against-merge-base method
for proving the CUDA path is byte-restored.

## Review 2026-08-11 (reviewer, linux-gfx942) -- changes-requested

Reviewed `git diff c87fe37...7dc3a24` on the fork `moat-port` branch (13 files,
+259/-69). No PR opened; findings below only. The strategy (compat header,
`.cu` kept and marked `LANGUAGE HIP`, `USE_HIP` option default OFF, CUDA branch
still on `cuda_add_*`) is the right one for a pure-CMake project and is applied
consistently; the wave-size analysis holds (no `warpSize`, `__shfl*`,
`__ballot`, `__popc`, or hardcoded 32 anywhere under
`src/lib/ReconstructorCUDA/` or `src/app/App_CUDA.cu`; all five kernels are
grid-stride and per-thread independent). Problems:

### 1. The OOB fault-class audit is wrong; the atomicInc bucket write can go one past the row

`notes.md` ("atomicInc wraps at bucket capacity so the row write stays
in-bounds") and `plan.md` ("`atomicInc(count, MAX_CNT_PER_BKT_)` wraps at 110 ==
bucket capacity, so the data write ... stays inside the bucket row") are both
incorrect.

`src/lib/ReconstructorCUDA/ReconstructorCUDA.cuh:45`

```
data_[atomicInc( &(count_[bktIdx]), MAX_CNT_PER_BKT_)+bktIdx*MAX_CNT_PER_BKT_] = val;
```

`atomicInc(address, val)` (identical in CUDA and HIP) stores
`((old >= val) ? 0 : (old+1))` and returns `old`. Starting from 0, the 110th
insert returns 109 and leaves `count_ == 110`; the 111th insert sees
`old == 110 >= val`, resets the counter to 0, and **returns 110**. The write is
then `data_[110 + bktIdx*110]`, i.e. slot 0 of bucket `bktIdx+1`, and for
`bktIdx == NUM_BKTS_-1` it is exactly one uint past the end of the
`MAX_CNT_PER_BKT_*NUM_BKTS_` allocation made at `ReconstructorCUDA.cuh:61`. The
correct capacity argument for this idiom is `MAX_CNT_PER_BKT_-1`.

The read side is fine (`getPointCloud2Cam` loops `i < count_[idx]` with
`count_ <= 110`, so `idx*110+i` stays in the row), so this is a write-only
overflow: a silent cross-bucket corruption in the general case and a
one-past-end global write for the last bucket.

This is unmodified upstream code, so fixing it is not required by "smallest
complete port" -- but recording it as "audited, already safe" is. Required:

- correct the claim in `notes.md` and `plan.md`;
- drop the supporting argument "confirmed at runtime -- AMD would have faulted
  on a stray read and the run completed clean". It does not hold for this case:
  the overflow needs >110 camera pixels decoding to one projector cell (the
  `alexander` dataset evidently never reaches it), and even when reached, a
  one-uint overrun of a ~346 MB device allocation lands inside the same page and
  would not fault;
- register it as deferred/upstream work with
  `python3 utils/deferred.py add --project 3DUNDERWORLD-SLS-GPU_CPU` rather than
  leaving it only as prose, so the decision to raise it with the maintainer is
  visible.

### 2. Upstream PR #33's body still describes the reverted `__builtin_trap()` change

The live PR body (fetched read-only) still contains:

> `DynamicBits.cuh`: the NVPTX-only `asm("trap;")` (illegal on amdgcn) becomes
> the portable `__builtin_trap()`. It is a never-taken overflow guard, so output
> is unaffected on either backend.

That is the exact claim 7dc3a24 disproved and reverted. The branch now uses
`gpuTrap()` (`DynamicBits.cuh:88`), defined per backend in `cuda_to_hip.h:60`
(HIP, `__builtin_trap()`) and `cuda_to_hip.h:66` (CUDA, `asm("trap;")`). A
reviewer reading the PR description against the diff will not find the change it
describes, and the description asserts portability that nvcc rejects.

Editing the PR body is an upstream write and a person's call, so the action here
is to draft the replacement bullet in `notes.md` so the checkup/maintainer round
can apply it. Suggested text:

> `DynamicBits.cuh`: the never-taken overflow guard used the NVPTX-only
> `asm("trap;")`, which has no amdgcn spelling. `__builtin_trap()` is
> device-callable under clang but host-only under nvcc, so neither spelling
> compiles under both toolchains. The compat header gains a `gpuTrap()` macro
> that expands to the original `asm("trap;")` on NVIDIA and `__builtin_trap()`
> on AMD; PTX for `DynamicBits.cu` is byte-identical to the pre-change sources.

The body's Test Plan also predates the current head (it lists the four GPU
validations but not the CUDA no-regression evidence added at 7dc3a24); fold that
in with the same edit.

### 3. `.gitignore` carries a local-workflow directory name upstream

`.gitignore:14-15` adds `build_ci_check/` and `build_hip/`. `build_hip/` matches
the name the README uses; `build_ci_check/` exists only in our local
verification of the CI commands (see the 2026-07-06 section above) and means
nothing to this project. Drop that line, or replace both with `build*/`.

### 4. `OPENCV4_COMPAT_DIR` is a public CMake option no document explains

`CMakeLists.txt:43-45` adds a WIN32-only `OPENCV4_COMPAT_DIR` include path. What
the directory must contain (an `opencv4` junction pointing at the prebuilt
OpenCV include root, because the sources hardcode `<opencv4/opencv2/...>`) is
recorded only in this file, not anywhere in the repository. A Windows user
cannot discover the option or construct the directory. Document it in `README.md`
beside the ROCm build instructions, or drop it from this change.

### 5. The README's ROCm configure line is not the one that was verified

`README.md:37-43` documents

```
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
```

Every configure recorded in this file, and the CI job at
`.github/workflows/hip.yml:53-59`, additionally passes
`-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++`. The bare form has never been
run, so the README publishes an untested command. Either run it on a stock ROCm
install and keep it, or document the compiler flag.

Same paragraph asserts ROCm "(7.2 or newer)". Nothing in the port needs 7.2;
only 7.2.1 (Linux) and 7.14 (Windows) were exercised, and the claim reads as a
hard floor that would turn away ROCm 6.x users. State what was tested rather
than a minimum that was not established.

### Checked and clean

Strategy A applied correctly (single compat header, no second HIP-aware file, no
`.cu` renames); no per-arch `#if` anywhere, so no wave32/wave64 divergence in
shared code; no textures, surfaces, pitched binds, or library swaps to get
wrong; the CUDA path is additive and guarded (`USE_HIP` default OFF, `USE_CUDA`
branch byte-equivalent, PTX diff on record); commit titles all `[ROCm]` and
<= 61 chars with AI-assistance disclosure and no `Co-Authored-By` trailer;
`jargon.py --port 3DUNDERWORLD-SLS-GPU_CPU` clean; no AMD-internal account
references in the diff or messages (the fork-only `moat-port` CI trigger present
in the workflow when it ran green on the fork was removed before this head).
The `fault-classes.md` lesson promoted on this branch was checked against the
code it describes and reproduces the fixed form, not the defect.

Not held against the port: no GPU run at this head from this reviewer (the
validator stage runs it next).

## Port fix 2026-08-11 (linux-gfx942, ROCm 7.14) -- review findings closed at bc3e4e9

Answers the 2026-08-11 review above. Host: AMD Instinct MI300X (gfx942, CDNA3,
wave64). ROCm 7.14.60850 came from an SDK wheel layout rather than `/opt/rocm`,
with the HIP compiler at `$(hipconfig --hipclangpath)/clang++` (clang 23.0.0) --
useful here because it is exactly the case the README's compiler fallback is
about.
Two new commits on top of 7dc3a24; plain fast-forward push, no history rewrite
(PR #33 is open on this branch, so a force-push was not an option).

### Finding 1 (OOB audit wrong) -- records corrected, code untouched

Re-derived it independently before editing: `atomicInc(p, val)` stores
`old >= val ? 0 : old+1` and returns `old`, so with `MAX_CNT_PER_BKT_ == 110`
(`ReconstructorCUDA.cu:25`) the counter does reach 110 and the next insert
returns 110, writing `data_[110 + bktIdx*110]` = slot 0 of bucket `bktIdx+1`,
one uint past the `110*NUM_BKTS_` allocation for the last bucket.
`git show c87fe37:src/lib/ReconstructorCUDA/ReconstructorCUDA.cuh` has the same
line (line 35 there), so it is pre-existing upstream and out of scope for a
minimal port. Read side re-checked and genuinely safe (`i < count_[idx]`,
`count_ <= 110`).

Corrected the audit in `notes.md` (Fault classes) and `plan.md` (Fault classes),
including dropping the "AMD would have faulted" argument: 1024x768 projector
cells and the alexander dataset never put >110 camera pixels in one cell, and a
one-uint overrun of the ~346 MB allocation stays in the same page. Registered as
deferred work `sls-gpu-bucket-atomicinc-overrun` (`projects/.../deferred.json`)
for the maintainer round. Promoted the general form to the skill
(`fault-classes.md`, Memory and lifetime): the `buf[atomicInc(&count[row],
CAPACITY) + row*CAPACITY]` idiom needs `CAPACITY-1`, is usually pre-existing
upstream, and cannot be cleared by a green run.

### Finding 2 (PR #33 body describes the reverted trap change) -- left for a person

No action taken here: editing the PR body is an upstream write. The replacement
bullet and the note about folding the CUDA no-regression evidence into the Test
Plan stay in the review section above for the checkup/maintainer round. The
same round should also mention the two commits added today.

### Finding 3 (`build_ci_check/` in .gitignore) -- fixed (90a8120)

Replaced the two port-added lines (`build_ci_check/`, `build_hip/`) with a
single `build*/`. Local `build_hip/`, `build_bare/`, `build_expl/` and `build/`
all stay ignored, so the integrity-gate `git status --porcelain` is still clean,
and no name from our own workflow ships upstream.

### Findings 4 + 5 (undocumented option, unverified README command) -- fixed (bc3e4e9)

`README.md` now documents `OPENCV4_COMPAT_DIR` beside the AMD build block: what
Windows OpenCV lays out, what the directory must contain, and an `mklink /J`
example. The CMake option itself is unchanged (the reviewer's alternative was to
drop it, but the two Windows validations depend on it).

The ROCm floor claim is gone, replaced by "tested with ROCm 7.2". The configure
line stays in its bare form -- BUT it is now a command that has actually been
run: on this host, with the ROCm SDK on PATH and no `/opt/rocm` at all,
`cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942` finds the HIP compiler
by itself. The prose now explains the `CMAKE_HIP_COMPILER` fallback (which is
what CI and the earlier host records used, and what a nonstandard install
needs) and why a GPU-less build machine must set `CMAKE_HIP_ARCHITECTURES`.

### Build on this host (gfx942) -- PASS

```bash
cd projects/3DUNDERWORLD-SLS-GPU_CPU/src
sudo apt-get install -y libglm-dev libopencv-dev libtiff-dev   # neither was present
rm -rf build_bare
cmake -S . -B build_bare -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942
bash utils/timeit.sh 3DUNDERWORLD-SLS-GPU_CPU compile -- cmake --build build_bare -j$(nproc)
```

Full `all` target builds clean (SLS, SLS_GPU, calibrateCamera, generateGraycode,
sync_main); only warning is the pre-existing `sync_main.cc:24` missing return.
`llvm-objdump --offloading build_bare/bin/SLS_GPU` -> 3 bundles, all
`hipv4-amdgcn-amd-amdhsa--gfx942`; links `libamdhip64.so.7`. The explicit-compiler
form (`-DCMAKE_HIP_COMPILER="$(hipconfig --hipclangpath)/clang++"`) configures
identically. No GPU run here (porter role); gfx942 validation is next.

Host quirk (promoted to the skill's `validation.md`): `roc-obj-ls` is not shipped
in the ROCm SDK wheel layout, so use
`$(hipconfig --hipclangpath)/llvm-objdump --offloading <binary>` to list embedded
code objects instead.

`python3 utils/jargon.py --port 3DUNDERWORLD-SLS-GPU_CPU`: clean.
`utils/prose.py` on both commit bodies and the new README paragraphs: clean.

## Review 2026-08-11 round 2 (reviewer, linux-gfx942) -- changes-requested

Re-reviewed the delta `7dc3a24..bc3e4e9` (2 files, +5/-5) and re-checked the full
`c87fe37...bc3e4e9` diff on the fork `moat-port` branch, plus the two skill
lessons this branch promotes. No PR opened.

Findings 1, 3, 4 and 5 of the 2026-08-11 review are genuinely closed; finding 2
is correctly deferred to the maintainer round and is not held against the fork.
One new problem, in a promoted lesson.

### 1. The promoted `roc-obj-ls` lesson misdiagnoses the failure it describes

`.claude/skills/cuda-to-rocm/references/validation.md:32` states `roc-obj-ls`
"is MISSING from SDK-wheel layouts (a host with no `/opt/rocm` at all)", and
line 35 tells the reader not to conclude a build is untargeted "because
`roc-obj-ls` is absent". `notes.md:1000` makes the same claim ("not shipped in
the ROCm SDK wheel layout"). On this host, which is exactly that layout and is
the host the lesson is attributed to, it is not absent:

```
$ which roc-obj-ls
/opt/conda/envs/py_3.12/bin/roc-obj-ls        # on the default PATH
$ roc-obj-ls build_bare/bin/SLS_GPU
Traceback (most recent call last):
  File "/opt/conda/envs/py_3.12/bin/roc-obj-ls", line 3, in <module>
    from rocm_sdk_core._cli import roc_obj_ls
ImportError: cannot import name 'roc_obj_ls' from 'rocm_sdk_core._cli'
```

The wheel ships the console-script shim but `rocm_sdk_core._cli` exports no
`roc_obj_ls` (confirmed by listing the module's names). So the symptom a future
agent meets is a Python traceback from a tool that is on PATH, not
command-not-found. A lesson that says "missing" invites the opposite
conclusion -- that the entry does not apply to their host, or that their ROCm
install is broken -- which is the failure mode the entry exists to prevent.

Required: reword both places to say the shim is present but its entry point is
broken in this layout (ImportError), so the tool is unusable rather than
absent. The `llvm-objdump --offloading` advice and the rest of the entry are
correct and should stay; I verified `$(hipconfig --hipclangpath)/llvm-objdump
--offloading` prints one `hipv4-amdgcn-amd-amdhsa--gfx942` bundle per device
translation unit on `build_bare/bin/SLS_GPU`.

### Carry-forward for the maintainer round (not a fork defect, no action here)

Extending finding 2 of the previous review: PR #33's body (fetched read-only)
is stale in three further ways beyond the `__builtin_trap()` bullet already
drafted above. It describes neither `OPENCV4_COMPAT_DIR` (`CMakeLists.txt:43`,
a Windows-only OpenCV include-path workaround that is a source change in the
diff), nor `.github/workflows/hip.yml` (added at the maintainer's request),
nor the `.gitignore` and README changes. Whoever applies the drafted
replacement should cover those too, so the description matches the diff a
maintainer reads.

### Checked and clean this round

- `.gitignore:13` is now a single `build*/`; `build` on line 3 is upstream's,
  and `git status --porcelain` in `src/` is clean with `build_bare/` and
  `build_expl/` present, so the integrity gate still holds.
- README's bare configure line was genuinely exercised:
  `build_bare/CMakeCache.txt:149` records `CMAKE_HIP_COMPILER:FILEPATH=` found
  by CMake itself, against `build_expl`'s `:STRING=` from the command line.
- The auto-detect path the README now documents was re-run here from scratch
  (`cmake -S . -B <tmp> -DUSE_HIP=ON`, no arch): CMake sets
  `CMAKE_HIP_ARCHITECTURES=gfx942;...` repeated once per visible GPU (8 on this
  host), the driver deduplicates, and the resulting `SLS_GPU` has the same 3
  gfx942 bundles and the same size (1833832 vs 1833808 bytes) as the pinned
  build. The documented default is safe on a multi-GPU node.
- `.github/workflows/hip.yml` is byte-identical to the version that ran green
  on the fork (run 28765809368, `ROCm build (HIP)`, success) except for the
  temporary `moat-port` trigger removal, so the container tag and every step
  are exercised evidence rather than a paste.
- The corrected `atomicInc` audit in `notes.md:60-71` and `plan.md:91-104`
  re-derives correctly, and `deferred.json` carries
  `sls-gpu-bucket-atomicinc-overrun` as open.
- The `fault-classes.md` additions (atomicInc capacity idiom, "portable is a
  claim", `gpuTrap()` repair shape, PTX-diff method, FindCUDA-vs-conda layout)
  check out against the code and against `cuda_to_hip.h:60,66`; each sits in
  the section a reader with that problem would open.
- Compat-header coverage is complete: the 15 `cuda*` symbols used across
  `src/lib/ReconstructorCUDA/*` and `src/app/App_CUDA.cu` are all aliased in
  `cuda_to_hip.h`, with no unguarded HIP spelling anywhere else.
- Upstream bit-rot repair re-verified as a repair, not a behaviour change:
  `git show c87fe37:src/lib/core/Reconstructor.h` is already a pure interface
  with only `reconstruct(const std::vector<Buckets>&)=0`, and `core/FileReader.h`
  does not exist at that commit, so the GPU path could not compile on nvcc
  either. `ReconstructorCUDA` owning `projector_` matches upstream's own
  destructor, and `cameras_` are stack objects in `App_CUDA.cu` (not owned).
- Commit hygiene on both new commits: `[ROCm]` titles at 51 and 54 chars, AI
  assistance disclosed, Test Plan with literal commands, no `Co-Authored-By`,
  ASCII. `jargon.py --port 3DUNDERWORLD-SLS-GPU_CPU` clean over the whole
  branch. Fork remote matches `status.json.fork_url` and `moat-port` is pushed
  at bc3e4e9.

Not held against the port: no GPU run at this head from this reviewer (the
validator stage runs it next).
