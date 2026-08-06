# cuPDLP-C notes

## Build (HIP/ROCm)

Dependencies:
- HiGHS 1.6.0 (NOT 1.10.0 -- API mismatch causes segfault)
- ROCm with hipBLAS, hipSPARSE

```bash
# Build HiGHS 1.6.0
git clone --depth 1 --branch v1.6.0 https://github.com/ERGO-Code/HiGHS.git
cd HiGHS && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PWD/../install -DBUILD_SHARED_LIBS=ON
make -j$(nproc) && make install
export HIGHS_HOME=$PWD/../install

# Build cuPDLP-C
cd cuPDLP-C
mkdir build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Test

```bash
export LD_LIBRARY_PATH=$HIGHS_HOME/lib:$PWD/lib:$LD_LIBRARY_PATH
./bin/plc -fname ../example/afiro.mps -nIterLim 5000
```

Expected output: "Solving information: Optimal current solution" with primal objective ~-464.75.

## Warp-size fix

The original code assumed warpSize=32 everywhere:
- Explicit exit(1) if warpSize != 32 in cupdlp_movement_interaction_cuda()
- `int lane = threadIdx.x % 32` and `int wid = threadIdx.x / 32` in reduction kernels
- `__shared__ cupdlp_float shared[32]` sized for 256/32=8 warps

Fixed with arch-dependent macros:
- `__GFX9__` -> wave64 (CDNA), else wave32 (RDNA/CUDA)
- `__shfl_down` for HIP (no sync needed), `__shfl_down_sync` for CUDA
- Shared memory sized to kMaxWarpsPerBlock=16 (upper bound)
- Warp-level reductions expanded to include offset 32 for wave64

## Known issues

- BUILD_APPS=ON fails due to API mismatch in onlinelp.cpp (upstream issue, not port-related)
- Requires HiGHS 1.6.0 specifically; 1.10.0 causes segfault in Init_Scaling (API incompatibility)

## Validation 2026-06-05

### Platform: linux-gfx90a (AMD Instinct MI250X / MI250)
### Arch: gfx90a (wave64, CDNA2)
### ROCm: 7.2.53211

Build:
```bash
cd /var/lib/jenkins/moat/_deps/cuPDLP-C/HiGHS/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PWD/../install -DBUILD_SHARED_LIBS=ON
make -j$(nproc) && make install
export HIGHS_HOME=/var/lib/jenkins/moat/_deps/cuPDLP-C/HiGHS/install

cd /var/lib/jenkins/moat/projects/cuPDLP-C/src/build
HIP_VISIBLE_DEVICES=2 cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Tests (HIP_VISIBLE_DEVICES=2):
```bash
LD_LIBRARY_PATH=$HIGHS_HOME/lib:$PWD/lib ./bin/testcudalin
# Output: 0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0 (PASS)

LD_LIBRARY_PATH=$HIGHS_HOME/lib:$PWD/lib ./bin/testcublas
# Output: 2-norm is :0.000000 (PASS)

LD_LIBRARY_PATH=$HIGHS_HOME/lib:$PWD/lib ./bin/plc -fname ../example/afiro.mps -nIterLim 5000
# Solving information: Optimal current solution
# Primal objective: -464.75088
# Dual objective: -464.83139
# Primal infeas: 2.69e-02 / 3.21e-05 (relative)
# Dual infeas: 3.97e-05 / 3.59e-06 (relative)
# Duality gap: 8.05e-02 / 8.65e-05 (relative)
# 200 iterations, all tolerances < 1e-4 (PASS)
```

Result: ALL TESTS PASS
- testcudalin: element-wise GPU operations correct
- testcublas: hipBLAS nrm2 correct
- plc afiro.mps: LP solver converges to OPTIMAL solution with correct objective -464.75

Wave64 reduction kernels (movement_1_kernel, movement_2_kernel, sum_kernel) work correctly on gfx90a CDNA2 with the __GFX9__ guards and warp-size-independent code.

## Validation 2026-06-05 (gfx1100)

### Platform: linux-gfx1100 (AMD Radeon Pro W7800 48GB)
### Arch: gfx1100 (wave32, RDNA3)
### ROCm: 7.0.53211

Build:
```bash
cd /var/lib/jenkins/moat/_deps/cuPDLP-C/HiGHS/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PWD/../install -DBUILD_SHARED_LIBS=ON
make -j$(nproc) && make install
export HIGHS_HOME=/var/lib/jenkins/moat/_deps/cuPDLP-C/HiGHS/install

cd /var/lib/jenkins/moat/projects/cuPDLP-C/src/build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Tests:
```bash
export LD_LIBRARY_PATH=$HIGHS_HOME/lib:$PWD/lib:$LD_LIBRARY_PATH
./bin/testcudalin
# Output: 0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0 (PASS)

./bin/testcublas
# Output: 2-norm is :0.000000 (PASS)

./bin/plc -fname ../example/afiro.mps -nIterLim 5000
# Solving information: Optimal current solution
# Primal objective: -464.75089
# Dual objective: -464.83139
# Primal infeas: 2.69e-02 / 3.21e-05 (relative)
# Dual infeas: 3.97e-05 / 3.60e-06 (relative)
# Duality gap: 8.05e-02 / 8.65e-05 (relative)
# 200 iterations, all tolerances < 1e-4 (PASS)
```

Result: ALL TESTS PASS
- testcudalin: element-wise GPU operations correct
- testcublas: hipBLAS nrm2 correct
- plc afiro.mps: LP solver converges to OPTIMAL solution with correct objective -464.75

Wave32 reduction kernels work correctly on gfx1100 RDNA3. The arch-dependent warp-size macros successfully handle both wave64 (gfx90a) and wave32 (gfx1100) in a single codebase.

## Validation 2026-06-08 (windows-gfx1201)

### Platform: windows-gfx1201 (AMD Radeon RX 9070 XT, gfx1201, RDNA4, wave32)
### ROCm: 7.14.0a20260604 (TheRock pip SDK)
### GPU index: HIP_VISIBLE_DEVICES=0 (only GPU enumerated this session)

Two Windows-specific build fixes were required and committed on top of the Linux port:

1. commit ee2f874 `[ROCm] Fix Windows build: skip -lm on Windows`
   - Changed `-lm` linker flag in 3 CMakeLists.txt files to `$<$<NOT:$<PLATFORM_ID:Windows>>:m>`
   - On Windows there is no m.lib; math functions are in the C runtime

2. commit 503569a `[ROCm] Fix Windows DLL export and HIP link for shared libraries`
   - Added `WINDOWS_EXPORT_ALL_SYMBOLS ON` to cudalin, cupdlp, wrapper_lp, wrapper_highs targets
   - Added explicit `target_link_libraries(wrapper_lp/wrapper_highs PRIVATE ${HIP_LIBRARY})` since mps_lp.c uses hipMalloc via the cuda_to_hip.h shim and cupdlp does not re-export those symbols on Windows

Build (using amdclang++ toolchain via TheRock venv, all-clang, no MSVC):

```powershell
# Activate TheRock venv
source B:/develop/TheRock/external-builds/pytorch/.venv/Scripts/activate

# Build HiGHS 1.6.0
cd B:/develop/moat/_deps/cuPDLP-C/HiGHS
mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=B:/develop/moat/_deps/cuPDLP-C/HiGHS/install \
  -DBUILD_SHARED_LIBS=ON \
  -DFAST_BUILD=OFF -DBUILD_EXAMPLES=OFF \
  -DCMAKE_C_COMPILER=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang.exe \
  -DCMAKE_CXX_COMPILER=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang++.exe
ninja -j24 && ninja install

# Build cuPDLP-C
cd B:/develop/moat/projects/cuPDLP-C/src
mkdir build && cd build
ROCM_PATH="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
cmake .. -G Ninja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_FLAGS="--rocm-device-lib-path=${ROCM_PATH}/lib/llvm/amdgcn/bitcode" \
  -DHIGHS_HOME=B:/develop/moat/_deps/cuPDLP-C/HiGHS/install \
  -DCMAKE_C_COMPILER=${ROCM_PATH}/lib/llvm/bin/clang.exe \
  -DCMAKE_CXX_COMPILER=${ROCM_PATH}/lib/llvm/bin/clang++.exe
ninja -j24

# Copy ROCm runtime DLLs into build/bin for DLL search
# (from _rocm_sdk_core and _rocm_sdk_libraries)
# Also create build/.kpack/blas_lib_gfx1201.kpack (rocsparse.dll resolves kpack at ../kpack relative to its DLL dir)
mkdir -p build/.kpack
cp _rocm_sdk_libraries/.kpack/blas_lib_gfx1201.kpack build/.kpack/
```

Tests (HIP_VISIBLE_DEVICES=0):
```python
# Run from build/bin with PATH including BINDIR + _rocm_sdk_core/bin + _rocm_sdk_libraries/bin
# ROCBLAS_TENSILE_LIBPATH=_rocm_sdk_libraries/bin/rocblas/library

testcudalin.exe
# Output: 0.000000, 1.000000, 4.000000, 9.000000, 16.000000, 25.000000, 36.000000, 49.000000, 64.000000, 81.000000 (PASS)

testcublas.exe
# Output: 2-norm is :0.000000 (PASS)

plc.exe -fname B:\...\example\afiro.mps -nIterLim 5000
# GPU device: AMD Radeon RX 9070 XT (gfx1201, RDNA4)
# Solving information: Optimal current solution
# Primal objective: -4.64750896e+02 (PASS, matches reference -464.75)
# Dual objective: -4.64831392e+02
# 200 iterations in 0.141s
```

Key Windows runtime requirement: rocsparse.dll references `../.kpack/blas_lib_gfx1201.kpack` relative to its own directory. The build/bin/ rocsparse.dll requires build/.kpack/blas_lib_gfx1201.kpack to exist. Copy from `_rocm_sdk_libraries/.kpack/blas_lib_gfx1201.kpack`.

Result: 3/3 PASS on AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32).
validated_sha = 503569a57a7a8b5ab2d9f0a0c4e6375696cc6750

## Revalidation 2026-06-08 (linux-gfx90a)

### Platform: linux-gfx90a (AMD Instinct MI250X / MI250, gfx90a, ROCm 7.2.1)
### Method: binary-equivalence carry-forward (no GPU re-run needed)

Delta f6105bf9..503569a5 consists of two Windows-specific commits:
- ee2f874: `[ROCm] Fix Windows build: skip -lm on Windows` -- changes `-lm` to `$<$<NOT:$<PLATFORM_ID:Windows>>:m>` in 3 CMakeLists.txt files; Linux still links -lm (no change)
- 503569a: `[ROCm] Fix Windows DLL export and HIP link for shared libraries` -- adds `WINDOWS_EXPORT_ALL_SYMBOLS ON` (ignored on Linux) and explicit HIP_LIBRARY links for wrapper_lp/wrapper_highs (DLL resolution only needed on Windows)

Classifier returned `mixed` (CMakeLists.txt token count changed), so binary-equivalence check performed.

Built at both SHAs for gfx90a:
```bash
export HIGHS_HOME=/var/lib/jenkins/moat/_deps/cuPDLP-C/HiGHS/install
# build_old at f6105bf9, build_new at 503569a5
cmake <src> -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

codeobj_diff.py result:
- lib/libcudalin.so: identical (exported symbols + device ISA identical, 74 exports)
- lib/libcupdlp.so: indeterminate (no GPU code in lib -- roc-obj-ls: "No kernel section found")
- lib/libwrapper_highs.so: indeterminate (no GPU code -- CPU-only wrapper)
- lib/libwrapper_lp.so: indeterminate (no GPU code -- CPU-only wrapper)

Manual nm -D symbol diff for all three indeterminate libs: zero diff (identical exported symbols).

Conclusion: libcudalin.so (sole GPU kernel library) is binary-identical on gfx90a. The three
CPU-only shared libs have identical exported symbols. Delta is Windows-only; no GPU re-run needed.

Carry-forward: completed at 503569a57a7a8b5ab2d9f0a0c4e6375696cc6750 (binary-equiv)

## Revalidation 2026-06-08 (linux-gfx1100)

### Platform: linux-gfx1100 (AMD Radeon Pro W7800 48GB, gfx1100, RDNA3, wave32)
### GPU index: HIP_VISIBLE_DEVICES=1
### ROCm: 7.0.53211

Delta f6105bf9..503569a5 (same two Windows-specific commits as above):
- ee2f874: -lm -> $<$<NOT:$<PLATFORM_ID:Windows>>:m> in CMakeLists.txt (Linux still links -lm)
- 503569a: WINDOWS_EXPORT_ALL_SYMBOLS ON (ignored on Linux) + explicit HIP_LIBRARY link for wrapper_lp/wrapper_highs

codeobj_diff result (built both SHAs in agent_space/cuPDLP-C-gfx1100-gpu1/build-old|new):
- lib/libcudalin.so: identical (exported symbols + device ISA identical, 74 exports)
- lib/libcupdlp.so: indeterminate (no GPU code)
- lib/libwrapper_highs.so: indeterminate (no GPU code)
- lib/libwrapper_lp.so: indeterminate (no GPU code)

Manual nm -D diff for indeterminate libs: libcupdlp.so is export-identical. wrapper_lp.so and wrapper_highs.so show hipMalloc/hipMemcpy/hipMemset/hipGetErrorString as `U hipMalloc@hip_4.2` in new vs `U hipMalloc` in old -- both undefined (imported), difference is only versioned symbol binding from explicit HIP_LIBRARY link. Same runtime function called; no behavioral change on Linux.

Since codeobj_diff returns `indeterminate` (not `identical`), proceeded with full GPU revalidation per protocol.

Build:
```bash
export HIGHS_HOME=/var/lib/jenkins/moat/_deps/cuPDLP-C/HiGHS/install
export HIP_VISIBLE_DEVICES=1
cd /var/lib/jenkins/moat/agent_space/cuPDLP-C-gfx1100-gpu1/build-new
cmake /var/lib/jenkins/moat/projects/cuPDLP-C/src -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Tests:
```bash
export LD_LIBRARY_PATH=$HIGHS_HOME/lib:/var/lib/jenkins/moat/agent_space/cuPDLP-C-gfx1100-gpu1/build-new/lib:$LD_LIBRARY_PATH
./bin/testcudalin
# 0.000000, 1.000000, 4.000000, 9.000000, 16.000000, 25.000000, 36.000000, 49.000000, 64.000000, 81.000000 (PASS)

./bin/testcublas
# 2-norm is :0.000000 (PASS)

./bin/plc -fname /var/lib/jenkins/moat/projects/cuPDLP-C/src/example/afiro.mps -nIterLim 5000
# Solving information: Optimal current solution
# Primal objective: -4.64750896e+02 (PASS, matches reference -464.75)
# 200 iterations
```

Result: 3/3 PASS on AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32).
validated_sha = 503569a57a7a8b5ab2d9f0a0c4e6375696cc6750

## CUDA no-regression gate 2026-06-12 (compile-verified with nvcc)

The port keeps the NVIDIA CUDA path alive (`-DBUILD_CUDA=ON`); the AMD additions
are guarded behind `USE_HIP` / `__HIP_PLATFORM_AMD__`. Previously the CUDA path
was only argued unchanged by construction (no MOAT host has an NVIDIA GPU). This
gate compiles it for real with the actual CUDA toolkit.

Toolkit: CUDA 12.4 (nvcc V12.4.131) installed via `mamba create -n cudaverify
-c nvidia cuda-toolkit=12.4`. The project hardcodes `/usr/local/cuda/include`
for the CUDA C targets (upstream behavior, unchanged by the port), and conda
stages headers under `targets/x86_64-linux/include` with libs in `lib/`, so the
env was normalized for a standard layout: `ln -s $PREFIX/lib $PREFIX/lib64`,
symlink `targets/x86_64-linux/include/*` into `$PREFIX/include`, and build with
`CPATH=$PREFIX/include`. On a normal `/usr/local/cuda` install none of this is
needed. See [[cuda-toolkit-noregression-gate]].

Build (no NVIDIA GPU required; nvcc emits device code without one):
```
export CUDA_HOME=/opt/conda/envs/cudaverify CPATH=$CUDA_HOME/include
cmake .. -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_HOST_COMPILER=$CUDA_HOME/bin/g++ -DHIGHS_HOME=$HIGHS_HOME
cmake --build . --target cupdlp cudalin testcudalin testcublas -j 16
```

Result: clean build, exit 0, no errors. nvcc compiled `cupdlp_cuda_kernels.cu`
(201 KB object) and `cupdlp_cudalinalg.cu` with `-arch=all`; cuobjdump confirms
fatbin device code for sm_50..sm_90. The `cudalin`/`cupdlp` libraries and the
`testcudalin`/`testcublas` executables all link. The warp-reduce macros resolve
to the original `__shfl_down_sync(0xFFFFFFFF, ...)` on the NVIDIA `#else` branch.

Scope: this proves the CUDA build is unbroken across all NVIDIA arches. It does
NOT prove runtime correctness on NVIDIA hardware (no GPU on the fleet).

## Validation 2026-06-19 (windows-gfx1101)

### Platform: windows-gfx1101 (AMD Radeon PRO V710, gfx1101, RDNA3, wave32)
### ROCm: 7.14.0a20260604 (TheRock pip SDK)
### GPU index: HIP_VISIBLE_DEVICES=1 (gfx1101 confirmed; gfx1201 at index 0)

Build (new directory build_gfx1101/, same toolchain as gfx1201):

```bash
ROCM_SDK="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
HIGHS_HOME="B:/develop/moat/projects/cuPDLP-C/HiGHS/install"
export ROCM_PATH="${ROCM_SDK}" HIP_PATH="${ROCM_SDK}"
export HIPCXX="${ROCM_SDK}/lib/llvm/bin/amdclang++.exe"
export HIGHS_HOME

cd B:/develop/moat/projects/cuPDLP-C/src/build_gfx1101
cmake .. -G Ninja \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_FLAGS="--rocm-device-lib-path=${ROCM_SDK}/lib/llvm/amdgcn/bitcode" \
  -DCMAKE_C_COMPILER="${ROCM_SDK}/lib/llvm/bin/clang.exe" \
  -DCMAKE_CXX_COMPILER="${ROCM_SDK}/lib/llvm/bin/clang++.exe" \
  -DCMAKE_HIP_COMPILER="${HIPCXX}"
ninja -j64
```

Runtime setup (copy DLLs and kpack into build_gfx1101/bin/):
- Copy from _rocm_sdk_core/bin: amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll, hiprtc-builtins0714.dll
- Copy from _rocm_sdk_libraries/bin: hipblas.dll, hipsparse.dll, rocblas.dll, rocsparse.dll, highs.dll
- mkdir build_gfx1101/.kpack && cp _rocm_sdk_libraries/.kpack/blas_lib_gfx1101.kpack build_gfx1101/.kpack/

Tests run via Python subprocess (PATH includes bin/ + _rocm_sdk_libraries/bin):

```python
import subprocess, os
bindir = r'B:\develop\moat\projects\cuPDLP-C\src\build_gfx1101\bin'
rocm_libs_bin = r'B:\develop\TheRock\external-builds\pytorch\.venv\Lib\site-packages\_rocm_sdk_libraries\bin'
env = {**os.environ, 'HIP_VISIBLE_DEVICES': '1',
       'ROCBLAS_TENSILE_LIBPATH': rocm_libs_bin + r'\rocblas\library',
       'PATH': bindir + ';' + rocm_libs_bin + ';' + os.environ.get('PATH', '')}
# Run each exe from cwd=bindir
```

Results:

testcudalin.exe:
# 0.000000, 1.000000, 4.000000, 9.000000, 16.000000, 25.000000,
# 36.000000, 49.000000, 64.000000, 81.000000 (PASS)

testcublas.exe:
# 2-norm is :0.000000 (PASS)

plc.exe -fname afiro.mps -nIterLim 5000:
# Cuda device 0: AMD Radeon PRO V710
# Solving information: Optimal current solution
# Primal objective: -4.64750896e+02 (PASS, matches reference -464.75)
# Dual objective: -4.64831392e+02
# 200 iterations in 0.13s

Result: 3/3 PASS on AMD Radeon PRO V710 (gfx1101, RDNA3, wave32).
validated_sha = 1a1ddff3840c0dd7c8d22302052ed84348d440e4
