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

## Upstream sync round 2026-11-04 (moat-fix-41, linux-gfx1100)

Why: upstream PR #41 was reported as conflicting with `main`. Upstream landed one large commit (b196976 "Prepare cuPDLP-C for COIN-OR", merged as f81f459). It bundles HiGHS 1.6.0 under `ThirdParty/HiGHS`, adds `build.sh`, `INSTALL`, `tests/` (CTest + NumPy checker), rewrites README.md, removes `example/` (afiro.mps now at `ThirdParty/HiGHS/check/instances/afiro.mps`), replaces the hardcoded `/usr/local/cuda/include` with `${CMAKE_CUDA_PATH}/include`, and converts every source and CMake file to CRLF. README.md, INSTALL, build.sh and other docs/scripts stay LF upstream (51 CRLF, 61 LF files outside ThirdParty).

Branch: `moat-fix-41` cut from published tip d380bcab.
- f10a1e80 merge of upstream/main f81f4592 (`git merge -X ignore-cr-at-eol upstream/main`). Subject rewritten to git's pull-style default `Merge branch 'main' of https://github.com/COPT-Public/cuPDLP-C` because jargon flagged the `moat-fix-41` branch name in the default subject.
- 09bb5323 `[ROCm] Add a HIP mode to build.sh and document it in INSTALL`.

Conflicts (10 raw, all whole-file CRLF hunks; 3 remained with ignore-cr-at-eol) and resolutions:
- `cupdlp/CMakeLists.txt`: kept the port's HIP branch, `(CUDA)` message and Windows `m` guard; took upstream's `${CMAKE_CUDA_PATH}/include` in the CUDA branch.
- `cupdlp/cuda/CMakeLists.txt`: kept the port's restructured CUDA_SOURCES / HIP / CUDA layout; replaced `/usr/local/cuda/include` in the CUDA branch with upstream's `${CMAKE_CUDA_PATH}/include`.
- `README.md`: took upstream's rewritten README and added one AMD line in its "Build, install and test" section (the old cmake-block section no longer fits; upstream defers build steps to INSTALL / build.sh). 09bb532 then points that line at `MODE=hip bash build.sh`.
- Silent fix: the ignore-cr merge dropped upstream's blank line before the new `# Installation checks` block in the top CMakeLists.txt; restored.
- No new CUDA code reached the files the port builds: upstream's .cu/.cuh changes were line endings only. Upstream's new top-level install rules work unchanged for HIP (`if(TARGET cudalin)`; `INSTALL_RPATH_USE_LINK_PATH` picks up the ROCm libs).

Line endings: every port-touched file is in upstream's convention for that file. All CMake, `.cmake`, `.cu`, `.cuh`, `.h` files the port adds or changes (including the port-added `FindHIPConf.cmake` and `cuda_to_hip.h`) are CRLF; README.md, INSTALL, build.sh stay LF like upstream. Check: `git diff upstream/main moat-fix-41 --stat` lists only the 12 previous port files plus build.sh and INSTALL, and its line content equals the port's pre-sync delta (`git diff --ignore-cr-at-eol 7b94c41 d380bca`) apart from the resolutions above. Every other file is byte-identical to upstream.

Build (TheRock pip SDK, hipconfig 7.16.26385, AMD Radeon Pro W7800 48GB gfx1100, HIP_VISIBLE_DEVICES=3):
```bash
export ROCM_PATH=$(hipconfig --rocmpath)
cmake -S src/ThirdParty/HiGHS -B highs-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PWD/highs -DBUILD_SHARED_LIBS=ON -DFAST_BUILD=ON
cmake --build highs-build -j10 && cmake --install highs-build
export HIGHS_HOME=$PWD/highs
cmake -S src -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j10
```
`ROCM_PATH` must be set on a pip-SDK host (no /opt/rocm); without it FindHIPConf cannot find hipblas.

Tests (ulimit -c 0, LD_LIBRARY_PATH=$HIGHS_HOME/lib:build/lib):
- `build/bin/testcudalin`: 0,1,4,...,81 PASS
- `build/bin/testcublas`: `2-norm is :0.000000` PASS
- `build/bin/plc -fname src/ThirdParty/HiGHS/check/instances/afiro.mps -nIterLim 5000`: Optimal, primal -4.64750896e+02, dual -4.64831392e+02, rel primal infeas 3.21e-05, rel dual infeas 3.60e-06, rel gap 8.65e-05, 200 iterations. PASS, identical to the 2026-06 gfx1100 figures.
- New upstream suite, `ctest --test-dir build`: 4/4 passed (installation_bounded, equality, mixed_bounds, compressed_mps).
- `cd src && python3 tests/run.py --project cuPDLP-C --mode gpu --binary build/bin/plc --output <tmp>`: pass=true, x=(1.00000001, 1.99999999), objective 7.99999999.
- `ROCM_PATH=... GPU_ARCH=gfx1100 HIGHS_HOME=... MODE=hip bash build.sh` in a clean worktree of 09bb532: exit 0. CTest 4/4, installed suite 4/4, checker pass=true (`Cuda device 0: AMD Radeon Pro W7800 48GB`), both checker-regression controls PASS. The installed plc resolves hipblas/hipsparse through its install rpath, with no LD_LIBRARY_PATH set.
Bar met: 3/3 recorded tests at the same counts, plus upstream's new tests.

Not done this round: CUDA no-regression compile (no nvcc on this host; the only CUDA-path change is upstream's own `${CMAKE_CUDA_PATH}/include` line); `MODE=hip` without GPU_ARCH (auto-detect, validated in d380bca, not rerun); wave64 and Windows revalidation (other hosts).

Jargon: `jargon.py --port cuPDLP-C` reports 2 hits, both in upstream's bundled `ThirdParty/HiGHS` ("strategy a..." in HEkkDual.h comments). They show up only because the fork's `main` mirror is still at 7b94c41; `git diff upstream/main moat-fix-41` has none. Fast-forwarding the fork mirror (`upstream.py --drift --apply`) clears them.

Scope note for jeff: 09bb532 (build.sh `MODE=hip` + INSTALL step) is the one addition beyond conflict resolution. The triage left it as a scope question. It is in because build.sh/INSTALL are now where upstream documents the CUDA build, and the porter contract puts the ROCm build doc in the same place. It is a separate commit; if you would rather leave build.sh alone, dropping it leaves the README line from the merge (`-DUSE_HIP=ON` in place of `-DBUILD_CUDA=ON`) as the only AMD doc.

### Copyright/author header finding (for jeff; unchanged this round)

`cupdlp/cuda/cupdlp_cuda_kernels.cu`, `cupdlp/cuda/cuda_to_hip.h` and `FindHIPConf.cmake` carry "Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. / Author: Jeff Daily". Upstream house style has no per-file headers: of the project's .c/.h/.cu/.cuh/.cpp files, only `cupdlp/cupdlp_cs.c/.h` (inherited CSparse, kept for attribution) carry one; the existing CUDA files have none. Attribution lives in AUTHORS, OWNERSHIP.txt (which declares Dongdong Ge the owner of project-owned code, MIT) and THIRD_PARTY_NOTICES. Under AGENTS.md's default this house style does not expect parallel attribution, and the header on `cupdlp_cuda_kernels.cu` claims "All rights reserved" on a file that is mostly upstream code, which sits awkwardly next to OWNERSHIP.txt. Recommendation: remove the three headers in a follow-up `[ROCm]` commit, or reduce them to a plain description line. The merge did not force a change, so they are unchanged. Your call.

### Proposed fix review PR body (for jeff)

Title: `[ROCm] Merge current upstream main into the HIP port`

```
Upstream PR COPT-Public/cuPDLP-C#41 showed as conflicting after upstream landed its COIN-OR preparation commit (b196976, merged in f81f459). That commit bundles HiGHS 1.6.0, adds build.sh, INSTALL and a tests/ suite, rewrites the README, and converts all source and CMake files to CRLF. No maintainer has commented on the PR; this round only brings it up to date.

What changed on moat-fix-41:
- f10a1e8: merge of upstream main (`-X ignore-cr-at-eol`). The three real conflicts were resolved keeping both sides: the HIP branches stay, the CUDA branches take upstream's `${CMAKE_CUDA_PATH}/include`, and the README takes upstream's text plus one AMD line. Every file the port adds or changes follows upstream's line endings (CRLF for sources and CMake, LF for README/INSTALL/build.sh); all other files are byte-identical to upstream, so the PR diff stays the port only.
- 09bb532: `MODE=hip` for build.sh and an INSTALL step, since those are now upstream's documented build path. Separate commit so it can be dropped.

Retested on AMD Radeon Pro W7800 (gfx1100): testcudalin, testcublas and plc on afiro (Optimal, -464.75, 200 iterations, same figures as before), upstream's CTest suite 4/4, the NumPy checker pass, and `MODE=hip bash build.sh` end to end. Wave64 and Windows revalidation follow on their hosts.

Open for you: whether to keep 09bb532, and the AMD copyright/author headers on three port files (see notes); unchanged here.

## Upstream reply

I merged current main into this branch, so it no longer conflicts, and every file the port touches now follows the repository's line endings (CRLF for sources and CMake files). I retested on an AMD Radeon Pro W7800 (gfx1100): the CUDA-linalg and BLAS tests, afiro, and your new CTest and checker suite all pass, and build.sh gains a `MODE=hip` option.

## Notes

Evidence and commands: projects/cuPDLP-C/notes.md, "Upstream sync round 2026-11-04".
```

## Review 2026-11-04 (moat-fix-41 at 09bb532, linux-gfx942)

Scope: `git diff d380bca...09bb532` (merge f10a1e8 plus 09bb532), and the full port `git diff upstream/main...09bb532` (14 files). Method: compared the CR-stripped added/removed lines of the pre-sync port delta (`git diff --ignore-cr-at-eol 7b94c41 d380bca`) with the post-sync delta (`git diff --ignore-cr-at-eol upstream/main 09bb532`). The only differences are the README text change, the INSTALL/build.sh additions and the two `/usr/local/cuda/include` -> `${CMAKE_CUDA_PATH}/include` resolutions, so the merge resolution is confirmed. The notes' line-ending and byte-identity claims also check out: all 14 touched files use upstream's convention for that file, nothing has mixed endings, and every other file equals upstream/main.

Verdict: changes requested (one finding).

1. `INSTALL:37-38` (09bb532) still says "Installed executables: .coinor-deps/cupdlp-cpu-install/bin/plc and/or .coinor-deps/cupdlp-gpu-install/bin/plc". With `MODE=hip`, build.sh installs to `.coinor-deps/cupdlp-hip-install` (`build.sh` `prefix="$PWD/.coinor-deps/cupdlp-$mode-install"`), so an AMD user is pointed at a directory that does not exist. Fix: list `.coinor-deps/cupdlp-hip-install/bin/plc` alongside the other two. In the same edit, `INSTALL:41` "Keep the external HiGHS and CUDA libraries at their configured locations" should read "CUDA or ROCm libraries", because the hip install depends on ROCm libraries through its install rpath in the same way. LF line endings, as in the rest of INSTALL.

Checked, not a finding: upstream's `SOURCE_MANIFEST.sha256` fails for the 12 port-modified files at head (on upstream/main every entry passes except the uninitialized pybind11 submodule). The file records the hashes of upstream's 20261001-r2 release archive (VALIDATION.md:4-6, SOURCE_PROVENANCE.json), and no build, test or CI step reads it. Regenerating it would rewrite upstream's release record, so the port leaves it alone. Mention it only if the maintainer asks.

Jargon: `jargon.py --port cuPDLP-C` (after creating a local `main` at origin/main in the clone) gives the same 2 hits as the porter, both in HiGHS comments under `ThirdParty/HiGHS` ("simplex strategy and", "weight strategy as"). They are false positives that come from the stale fork mirror.

### Proposed fix review PR body: review

The draft in "Upstream sync round 2026-11-04" fails `jargon.py -`: "moat-fix-41" (line 3) and "revalidation" (line 7). Other problems:
- "converts all source and CMake files to CRLF" is false. On upstream/main, `cupdlp/cupdlp_cs.c`, `cupdlp/cupdlp_cs.h`, `tests/CMakeLists.txt`, `third-party/CMakeLists.txt` and `cupdlp/test/CMakeLists.txt` are still LF. The notes' "converts every source and CMake file to CRLF" has the same error.
- The "## Notes" section and "(see notes)" point at this repository's records, which a fork reader cannot see.
- It names 09bb532 as the only build.sh/INSTALL commit, but finding 1 adds another.
- GPU naming: at 09bb532 no platform has a validation recorded. status.json gives linux-gfx1100 `validated_sha` = d380bca, the published tip, and the gfx1100 run at 09bb532 was the porter's own run, not a validation. The head also moves when finding 1 is fixed. Post the body and its `## Upstream reply` only after validation completes at the final head, and name only the GPUs whose platforms completed there. If linux-gfx1100 has not completed by then, take out the W7800 sentence. If wave64 (this host, gfx942) or Windows completes, those GPUs can be added.

Corrected body (jargon.py - clean, prose.py clean, ASCII). Title unchanged: `[ROCm] Merge current upstream main into the HIP port`

```
Upstream PR COPT-Public/cuPDLP-C#41 showed as conflicting after upstream landed its COIN-OR preparation commit (b196976, merged in f81f459). That commit bundles HiGHS 1.6.0, adds build.sh, INSTALL and a tests/ suite, rewrites the README, and converts most source and CMake files to CRLF. No maintainer has commented on the PR; this round only brings it up to date.

What changed on this branch:
- f10a1e8: merge of upstream main (`-X ignore-cr-at-eol`). The three real conflicts were resolved keeping both sides: the HIP branches stay, the CUDA branches take upstream's `${CMAKE_CUDA_PATH}/include`, and the README takes upstream's text plus one AMD line. Every file the port adds or changes keeps upstream's line endings for that file (CRLF for the CMake files and CUDA sources, LF for README.md, INSTALL and build.sh); all other files are byte-identical to upstream, so the PR diff stays the port only.
- `MODE=hip` for build.sh, with INSTALL documenting it (including where the hip install lands), since those are now upstream's documented build path. Separate commits, so they can be dropped.

Retested on AMD Radeon Pro W7800 (gfx1100): testcudalin, testcublas and plc on afiro (Optimal, -464.75, 200 iterations, same figures as before), upstream's CTest suite 4/4, the NumPy checker pass, and `MODE=hip bash build.sh` end to end.

Open for you: whether to keep the build.sh/INSTALL commits, and the AMD copyright/author headers on three port files (FindHIPConf.cmake, cuda_to_hip.h, cupdlp_cuda_kernels.cu); unchanged here.

## Upstream reply

I merged current main into this branch, so it no longer conflicts, and every file the port touches keeps the line endings the repository uses for it. I retested on an AMD Radeon Pro W7800 (gfx1100): testcudalin, testcublas, afiro, and your new CTest suite and NumPy checker all pass. build.sh also gains a `MODE=hip` option, documented in INSTALL.
```

## Fix round 2026-11-04 (moat-fix-41 -> 8f43bf8, linux-gfx1100): review finding 1

Finding 1 (INSTALL hip path) addressed in 8f43bf8 `[ROCm] List the hip install path in INSTALL`, a new commit on top of 09bb532 (no rebase, no amend). INSTALL now lists `.coinor-deps/cupdlp-hip-install/bin/plc` next to the cpu and gpu paths, and the library sentence reads "Keep the external HiGHS and CUDA or ROCm libraries at their configured locations." INSTALL stays LF (0 CR bytes). Not changed, per jeff's open questions: whether to keep 09bb532's build.sh/INSTALL HIP mode, and the AMD copyright headers.

Build (TheRock pip SDK, hipconfig 7.16.26385, AMD Radeon Pro W7800 48GB gfx1100), clean worktree of 8f43bf8, `ulimit -c 0`:
```bash
export ROCM_PATH=$(hipconfig --rocmpath)
GPU_ARCH=gfx1100 MODE=hip bash build.sh
```
exit 0. Bundled HiGHS built by build.sh (no HIGHS_HOME). CTest 4/4 passed, installed suite 4/4 (bounded -10, equality 3, mixed_bounds -1.000000015, compressed_mps -10), NumPy checker pass=true (x=(1.00000001, 1.99999999), objective 7.99999999), both checker-regression controls PASS. `.coinor-deps/cupdlp-hip-install/bin/plc` exists, so the INSTALL path is right.

Direct tests from `build-coinor-hip/bin`: testcudalin 0,1,4,...,81 PASS; testcublas `2-norm is :0.000000` PASS; installed plc on `ThirdParty/HiGHS/check/instances/afiro.mps -nIterLim 5000`: `Cuda device 0: AMD Radeon Pro W7800 48GB`, Optimal, primal -4.64750896e+02, dual -4.64831392e+02, 200 iterations. Same figures as at 09bb532 and d380bca. This is the porter's run, not a validation; every platform now reads revalidate at 8f43bf8.

Jargon: `jargon.py --port cuPDLP-C` gives the same 2 known false positives in bundled `ThirdParty/HiGHS` comments (stale fork `main` mirror). Pushed only `moat-fix-41` (09bb532..8f43bf8); `moat-port` stays at d380bca. advance-head 8f43bf8, stage ported.

Not done: review-PR thread replies (porter step 9), because no fix review PR is recorded and this run has no GitHub writes.

### Fix review PR body at 8f43bf8 (for jeff)

Supersedes the corrected body in "Review 2026-11-04". prose.py clean, jargon.py clean, ASCII. The two `<...>` placeholders stay until validation completes at 8f43bf8; then name only the GPUs whose platform records show a completed validation at that head. Title unchanged: `[ROCm] Merge current upstream main into the HIP port`

```
Upstream PR COPT-Public/cuPDLP-C#41 showed as conflicting after upstream landed its COIN-OR preparation commit (b196976, merged in f81f459). That commit bundles HiGHS 1.6.0, adds build.sh, INSTALL and a tests/ suite, rewrites the README, and converts most source and CMake files to CRLF. No maintainer has commented on the PR; this round only brings it up to date.

What changed on this branch:
- f10a1e8: merge of upstream main (`-X ignore-cr-at-eol`). The three real conflicts were resolved keeping both sides: the HIP branches stay, the CUDA branches take upstream's `${CMAKE_CUDA_PATH}/include`, and the README takes upstream's text plus one AMD line. Every file the port adds or changes keeps upstream's line endings for that file (CRLF for the CMake files and CUDA sources, LF for README.md, INSTALL and build.sh); all other files are byte-identical to upstream, so the PR diff stays the port only.
- 09bb532 and 8f43bf8: `MODE=hip` for build.sh, with INSTALL documenting it, listing where the hip install lands (`.coinor-deps/cupdlp-hip-install/bin/plc`) and noting that it needs the ROCm libraries kept in place. Separate commits, so they can be dropped.

<TESTED_ON: name only the GPUs whose platform records show a completed validation at 8f43bf8, with what passed: testcudalin, testcublas and plc on afiro (Optimal, -464.75, 200 iterations), upstream's CTest suite 4/4, the NumPy checker, and `MODE=hip bash build.sh` end to end.>

Open for you: whether to keep the build.sh/INSTALL commits, and the AMD copyright/author headers on three port files (FindHIPConf.cmake, cuda_to_hip.h, cupdlp_cuda_kernels.cu); unchanged here.

## Upstream reply

<UPSTREAM_REPLY: write once validation has completed at 8f43bf8. Say that current main is merged so the PR no longer conflicts, that every file the port touches keeps the repository's line endings for it, which AMD GPUs (only those with a completed validation at 8f43bf8) passed testcudalin, testcublas, afiro and the new CTest suite and NumPy checker, and that build.sh gains a `MODE=hip` option documented in INSTALL.>
```

## Review 2026-11-04 (moat-fix-41 at 8f43bf8, linux-gfx942): re-review

Scope: the new commit since the 09bb532 review, `git diff 09bb532 8f43bf8` (INSTALL only, +4/-3). upstream/main is still f81f459, so the merge resolution reviewed at 09bb532 stands. `git diff upstream/main 8f43bf8` still lists the same 14 files. Each file still uses one line-ending convention, the same one upstream uses (CRLF line count equals total lines for the CMake/CUDA/header files; INSTALL, README.md and build.sh have 0 CR).

Verdict: review passed. There are no findings.

- Finding 1 (INSTALL hip install path) is addressed. INSTALL:37-39 lists `.coinor-deps/cupdlp-hip-install/bin/plc`, which matches build.sh `prefix="$PWD/.coinor-deps/cupdlp-$mode-install"` for `MODE=hip`. INSTALL:42 reads "CUDA or ROCm libraries". INSTALL stays LF, and its longest line (85) is within upstream's line-width range for the file.
- 8f43bf8 commit message: `[ROCm]` title within 72 characters, rationale, "Authored with Claude.", fenced Test Plan, and no Co-Authored-By trailer.
- `jargon.py --port cuPDLP-C` (local `main` set to origin/main in the clone) gives the same 2 known false positives in bundled `ThirdParty/HiGHS` comments, which come from the stale fork mirror.
- Upstream PR #41, read-only query today: OPEN, CONFLICTING/DIRTY, head d380bca, 0 comments and 0 reviews. "No maintainer has commented" holds. No fork review PR exists yet.

### Fix review PR body at 8f43bf8: review

`jargon.py -` and `prose.py` both report clean, and the text is ASCII. I checked every factual claim against the head: the README diff is the one AMD line, the line-ending claim holds, "most" CRLF is right, 09bb532/8f43bf8 are separate commits, and the INSTALL description matches. The `<TESTED_ON>` and `<UPSTREAM_REPLY>` placeholders are correct for now, because no platform has a validation recorded at 8f43bf8 (all read revalidate). The body cannot be posted until validators complete at 8f43bf8. To reduce the risk of a placeholder being posted verbatim, here is a ready `## Upstream reply` with one slot to fill. Name only the GPUs whose platform records show `completed` with `validated_sha` 8f43bf8. If the head moves again, re-check. It passes `jargon.py -`, `prose.py` and the ASCII check:

```
## Upstream reply

I merged current main into this branch, so it no longer conflicts, and every file the port touches keeps the line endings the repository uses for it. I retested on <GPUs validated at 8f43bf8, e.g. "an AMD Instinct MI300X (gfx942)">: testcudalin, testcublas, afiro, and your new CTest suite and NumPy checker all pass. build.sh also gains a `MODE=hip` option, documented in INSTALL.
```

The open questions for jeff are unchanged (whether to keep the build.sh/INSTALL commits, and the AMD copyright headers).

## Validation 2026-11-04 (linux-gfx90a, moat-fix-41 at 8f43bf8)

Platform: linux-gfx90a, AMD Instinct MI250X / MI250 (gfx90a, wave64, CDNA2), TheRock pip SDK (hipconfig 7.16.26385), HIP_VISIBLE_DEVICES=0, `ulimit -c 0`. Full run (sync round changed the tree; no carry-forward). Fresh clone of the fork, `git checkout 8f43bf8401712e7ce342cc52a412872f09f22069` (tip of `moat-fix-41`), CRLF files left as checked out.

Build and upstream suites (bundled HiGHS built by build.sh, no HIGHS_HOME):
```bash
export ROCM_PATH=$(hipconfig --rocmpath)
GPU_ARCH=gfx90a MODE=hip bash build.sh
```
exit 0, `CMAKE_HIP_ARCHITECTURES: gfx90a`. CTest 4/4 passed (installation_bounded, equality, mixed_bounds, compressed_mps). Installed suite 4/4 (bounded -10, equality 3, mixed_bounds -1.000000015, compressed_mps -10, all OPTIMAL). NumPy checker `--mode gpu` pass=true, x=(1.00000001, 1.99999999), objective 7.99999999, primal residual 1.08e-08. Both checker-regression controls PASS (stale_success_with_noop, incorrect_fresh_solution).

Direct GPU tests:
```bash
cd build-coinor-hip/bin && ./testcudalin && ./testcublas
.coinor-deps/cupdlp-hip-install/bin/plc -fname ThirdParty/HiGHS/check/instances/afiro.mps -nIterLim 5000
```
- testcudalin: 0,1,4,...,81, rc=0. PASS
- testcublas: `2-norm is :0.000000`, rc=0. PASS
- plc afiro (installed binary, rpath, no LD_LIBRARY_PATH): `Cuda device 0: AMD Instinct MI250X / MI250`, Optimal, primal -4.64750884e+02, dual -4.64831390e+02, rel primal infeas 3.21e-05, rel dual infeas 3.59e-06, rel gap 8.65e-05, 200 iterations, rc=0. PASS

Comparison: same 3/3 direct tests as the 2026-06-05 gfx90a validation (primal -464.75088, dual -464.83139, same relative residuals, 200 iterations) and the porter's gfx1100 run at 8f43bf8 (CTest 4/4, installed 4/4, checker pass, 200 iterations). No regressions.

Jargon: `jargon.py --port cuPDLP-C` (local `main` and `moat-fix-41` branches created in the clone) gives the same 2 known false positives in bundled `ThirdParty/HiGHS` comments ("simplex strategy and", "weight strategy as") from the stale fork `main` mirror. Docs: ROCm build documented in README, INSTALL and build.sh `MODE=hip`, where upstream documents CUDA. Tree clean (only untracked `.coinor-deps/`, `tests/logs/`).

cuda-not-validated: no CUDA toolkit on this host (no conda; the fleet guard refuses pip installing nvidia-cuda-nvcc into a venv). The CUDA gate is still owed at 8f43bf8; the only CUDA-path change since the 2026-06-12 nvcc gate is upstream's own `${CMAKE_CUDA_PATH}/include` taken in the merge.

Result: completed at 8f43bf8401712e7ce342cc52a412872f09f22069 (wave64).

## Validation 2026-11-04 (linux-gfx1100, moat-fix-41 at 8f43bf8)

Platform: linux-gfx1100, AMD Radeon Pro W7800 48GB (gfx1100, wave32, RDNA3), TheRock pip SDK (`hipconfig --rocmpath`), HIP_VISIBLE_DEVICES=1, `ulimit -c 0`. Full run: this platform's previous `validated_sha` was d380bca, before the sync merge changed the tree, so no carry-forward. Fresh clone of the fork into `projects/cuPDLP-C/src`, `git fetch origin moat-fix-41`, `git checkout 8f43bf8401712e7ce342cc52a412872f09f22069`; CRLF files left as checked out.

Build and upstream suites (bundled HiGHS built by build.sh, no HIGHS_HOME):
```bash
export ROCM_PATH=$(hipconfig --rocmpath)
GPU_ARCH=gfx1100 MODE=hip bash projects/cuPDLP-C/src/build.sh
```
exit 0, `CMAKE_HIP_ARCHITECTURES: gfx1100`. CTest 4/4 passed (installation_bounded, equality, mixed_bounds, compressed_mps). Installed suite 4/4 (bounded -10, equality 3, mixed_bounds -1.000000015, compressed_mps -10, all OPTIMAL). NumPy checker `--mode gpu` pass=true, x=(1.00000001, 1.99999999), objective 7.99999999, primal residual 1.08e-08. Both checker-regression controls PASS (stale_success_with_noop, incorrect_fresh_solution).

Direct GPU tests:
```bash
cd build-coinor-hip/bin && ./testcudalin && ./testcublas
cd ../.. && .coinor-deps/cupdlp-hip-install/bin/plc -fname ThirdParty/HiGHS/check/instances/afiro.mps -nIterLim 5000
```
- testcudalin: 0,1,4,...,81, rc=0. PASS
- testcublas: `2-norm is :0.000000`, rc=0. PASS
- plc afiro (installed binary, rpath, no LD_LIBRARY_PATH): `Cuda device 0: AMD Radeon Pro W7800 48GB`, Optimal, primal -4.64750896e+02, dual -4.64831392e+02, rel primal infeas 3.21e-05, rel dual infeas 3.60e-06, rel gap 8.65e-05, 200 iterations, rc=0. PASS

Comparison: identical figures to the porter's gfx1100 run at 8f43bf8 (CTest 4/4, installed 4/4, checker pass, afiro -464.750896 in 200 iterations) and to this platform's earlier records at 503569a and d380bca (3/3 direct tests, same afiro objective and residuals). No regressions.

Jargon: `jargon.py --port cuPDLP-C` (local `main` created at origin/main in the clone) gives the same 2 known false positives in bundled `ThirdParty/HiGHS` comments from the stale fork `main` mirror. Docs: ROCm build documented in README, INSTALL and build.sh `MODE=hip`. Tree clean (only untracked `.coinor-deps/`, `tests/logs/`).

cuda-not-validated: no CUDA toolkit on this host (no conda, no nvcc). The CUDA compile gate is still owed at 8f43bf8.

Result: completed at 8f43bf8401712e7ce342cc52a412872f09f22069 (wave32).

## Fix round 2026-11-04 (moat-fix-41 -> f56941f, linux-gfx1100): copyright/author lines

jeff's rulings on the two open questions: keep 09bb532 (build.sh `MODE=hip` + INSTALL) unchanged; remove the AMD copyright and author lines the port added.

Search: `git diff upstream/main 8f43bf8 | grep -niE '^\+.*(copyright|author|\(c\)|daily|amd\.com|rights|license|spdx|contributor)'` finds exactly six added lines, the copyright and author pair in each of `FindHIPConf.cmake`, `cupdlp/cuda/cuda_to_hip.h` and `cupdlp/cuda/cupdlp_cuda_kernels.cu`. All three headers are port-added: `FindHIPConf.cmake` and `cuda_to_hip.h` are new files, and upstream's `cupdlp_cuda_kernels.cu` starts with `#include` and has no comment block. AUTHORS, OWNERSHIP.txt and THIRD_PARTY_NOTICES are untouched by the port.

f56941f `[ROCm] Drop per-file copyright and author lines from HIP files`, a new commit on top of 8f43bf8 (no amend, no rebase), removes the copyright line, the author line and the empty comment separator after them in each file. The description comments stay (`# Locate the ROCm/HIP toolchain...`, `CUDA-to-HIP compatibility header...`, `AMD GPU (HIP/ROCm) support for the cuPDLP-C CUDA kernels.`). Each of the three files is still all CRLF (62/62, 108/108, 474/474 lines).

`moatlib.py classify cuPDLP-C 8f43bf8 f56941f`: `class=comment-only arch_independent=True inert=True`, all three files comment-only. advance-head carried linux-gfx90a and linux-gfx1100 forward to f56941f (method source-class, class comment-only). The windows gate still owes a run (both Windows records are at d380bca, as they were at 8f43bf8), so `fix-ready` is False until then.

Rebuild and rerun at f56941f (AMD Radeon Pro W7800 48GB gfx1100, TheRock pip SDK, HIP_VISIBLE_DEVICES=1, `ulimit -c 0`), in the fresh clone:
```bash
export ROCM_PATH=$(hipconfig --rocmpath)
GPU_ARCH=gfx1100 MODE=hip bash build.sh
build-coinor-hip/bin/testcudalin
build-coinor-hip/bin/testcublas
.coinor-deps/cupdlp-hip-install/bin/plc -fname ThirdParty/HiGHS/check/instances/afiro.mps -nIterLim 5000
```
build.sh exit 0 (`CMAKE_HIP_ARCHITECTURES: gfx1100`), CTest 4/4, installed suite 4/4 (bounded -10, equality 3, mixed_bounds -1.000000015, compressed_mps -10), NumPy checker pass=true, both checker-regression controls PASS. testcudalin 0..81 rc=0; testcublas `2-norm is :0.000000` rc=0; afiro Optimal, primal -4.64750896e+02, dual -4.64831392e+02, rel infeas 3.21e-05 / 3.60e-06, rel gap 8.65e-05, 200 iterations. Identical to 8f43bf8. This is the porter's run.

Jargon: `jargon.py --port cuPDLP-C` gives the same 2 known false positives in bundled `ThirdParty/HiGHS` comments (stale fork `main` mirror). Pushed only `moat-fix-41` (8f43bf8..f56941f); `moat-port` stays at d380bca. Stage ported.

### Fix review PR body at f56941f (for jeff)

Supersedes the earlier drafts. prose.py clean, `jargon.py -` clean, ASCII. The GPUs named are the platforms with a completed record at f56941f: linux-gfx90a (MI250X) and linux-gfx1100 (W7800), both validated at 8f43bf8 and carried forward to f56941f as comment-only. If a Windows platform completes at f56941f before the PR opens, its GPU may be added; if the head moves again, re-check. Title unchanged: `[ROCm] Merge current upstream main into the HIP port`

```
Upstream PR COPT-Public/cuPDLP-C#41 showed as conflicting after upstream landed its COIN-OR preparation commit (b196976, merged in f81f459). That commit bundles HiGHS 1.6.0, adds build.sh, INSTALL and a tests/ suite, rewrites the README, and converts most source and CMake files to CRLF. No maintainer has commented on the PR; this round only brings it up to date.

What changed on this branch:
- f10a1e8: merge of upstream main (`-X ignore-cr-at-eol`). The three real conflicts were resolved keeping both sides: the HIP branches stay, the CUDA branches take upstream's `${CMAKE_CUDA_PATH}/include`, and the README takes upstream's text plus one AMD line. Every file the port adds or changes keeps upstream's line endings for that file (CRLF for the CMake files and CUDA sources, LF for README.md, INSTALL and build.sh); all other files are byte-identical to upstream, so the PR diff stays the port only.
- 09bb532 and 8f43bf8: `MODE=hip` for build.sh, with INSTALL documenting it, listing where the hip install lands (`.coinor-deps/cupdlp-hip-install/bin/plc`) and noting that it needs the ROCm libraries kept in place.
- f56941f: removes the AMD copyright and author lines from FindHIPConf.cmake, cuda_to_hip.h and cupdlp_cuda_kernels.cu, because the project keeps no per-file headers and records attribution in AUTHORS and OWNERSHIP.txt. Comment-only; each file keeps its description comment.

Retested at 8f43bf8 on AMD Instinct MI250X (gfx90a) and AMD Radeon Pro W7800 (gfx1100): testcudalin, testcublas and plc on afiro (Optimal, -464.75, 200 iterations), upstream's CTest suite 4/4, the installed suite 4/4, the NumPy checker, and `MODE=hip bash build.sh` end to end. f56941f changes only comments; the W7800 rebuild and rerun at f56941f gave the same results.

## Upstream reply

I merged current main into this branch, so it no longer conflicts, and every file the port touches keeps the line endings the repository uses for it. I retested on an AMD Instinct MI250X (gfx90a) and an AMD Radeon Pro W7800 (gfx1100): testcudalin, testcublas, afiro, and your new CTest suite and NumPy checker all pass. build.sh also gains a `MODE=hip` option, documented in INSTALL, and I removed the per-file copyright and author lines the port had added, since the project records attribution in AUTHORS and OWNERSHIP.txt.
```

## Review 2026-11-04 (moat-fix-41 at f56941f, linux-gfx90a): re-review

Scope: `git diff 8f43bf8 f56941f` (3 files, 9 deletions, 0 insertions), plus a re-check of the full port `git diff upstream/main f56941f`. upstream/main is still f81f459, so the merge resolution reviewed at 09bb532 stands.

Verdict: review passed. No findings.

- Only port-added lines removed. `FindHIPConf.cmake` and `cupdlp/cuda/cuda_to_hip.h` do not exist on upstream/main, and upstream's `cupdlp_cuda_kernels.cu` starts at `#include "cupdlp_cuda_kernels.cuh"` with no comment block, so all nine deleted lines (copyright, author, empty separator in each file) came from the port. No upstream text touched; each file keeps its description comment.
- Line endings kept: the deleted lines were CRLF, and the files are 62/62, 108/108, 474/474 CRLF at head. Across all 14 port files each uses the same single convention as upstream (CMake/CUDA all CRLF; INSTALL, README.md, build.sh 0 CR).
- No per-file attribution left: `git diff upstream/main f56941f | grep -iE '^\+.*(copyright|author|\(c\)|daily|amd\.com|rights reserved|spdx|contributor|written by|advanced micro)'` returns nothing. The remaining `AMD` mentions are technical (option text, `__HIP_PLATFORM_AMD__`, docs). AUTHORS, OWNERSHIP.txt and THIRD_PARTY_NOTICES are not in the port diff. Git commit author metadata is the only place the address appears, which is normal authorship, not a file header.
- f56941f commit message: `[ROCm]` title within 72 characters, rationale, "Authored with Claude.", fenced Test Plan, no Co-Authored-By trailer.
- Upstream PR #41, read-only query today: OPEN, DIRTY, head d380bca, 0 comments, 0 reviews. "No maintainer has commented" holds.
- Gates at f56941f: linux-gfx90a and linux-gfx1100 completed (validated at 8f43bf8, carried forward as comment-only); the windows gate is still owed (both Windows records at d380bca), so the body cannot be posted yet.

Minor wording, not a finding: f56941f's message and the drafted body say the project keeps no per-file copyright headers. Upstream's inherited CSparse files `cupdlp/cupdlp_cs.c` and `cupdlp/cupdlp_cs.h` carry them, and OWNERSHIP.txt says existing notices are preserved. Accurate for the project's own sources. The commit stays as is (no amend); the body below is narrowed to say "its own sources". At any PR-prep squash, apply the same narrowing to the commit message.

### Fix review PR body at f56941f: review

The draft in "Fix round 2026-11-04 (moat-fix-41 -> f56941f)" passes `jargon.py -` and `prose.py`, and it is ASCII. Claims checked against the head: README diff is the one AMD line; 14 files differ from upstream/main and every other file is byte-identical; the line-ending claim holds; "most" CRLF is right; the INSTALL description matches INSTALL:27-42; and the retest list matches the gfx90a and gfx1100 validator records at 8f43bf8 (testcudalin, testcublas, afiro Optimal -464.75 in 200 iterations, CTest 4/4, installed suite 4/4, NumPy checker, build.sh end to end), which both carry forward to f56941f. The `## Upstream reply` names only MI250X (gfx90a) and W7800 (gfx1100), the two GPUs with a completed record at f56941f. The only change is the narrowing above. If a Windows platform completes at f56941f before posting, its GPU may be added to both the retest paragraph and the reply; if the head moves, re-check. Corrected body (jargon.py - clean, prose.py clean, ASCII). Title unchanged: `[ROCm] Merge current upstream main into the HIP port`

```
Upstream PR COPT-Public/cuPDLP-C#41 showed as conflicting after upstream landed its COIN-OR preparation commit (b196976, merged in f81f459). That commit bundles HiGHS 1.6.0, adds build.sh, INSTALL and a tests/ suite, rewrites the README, and converts most source and CMake files to CRLF. No maintainer has commented on the PR; this round only brings it up to date.

What changed on this branch:
- f10a1e8: merge of upstream main (`-X ignore-cr-at-eol`). The three real conflicts were resolved keeping both sides: the HIP branches stay, the CUDA branches take upstream's `${CMAKE_CUDA_PATH}/include`, and the README takes upstream's text plus one AMD line. Every file the port adds or changes keeps upstream's line endings for that file (CRLF for the CMake files and CUDA sources, LF for README.md, INSTALL and build.sh); all other files are byte-identical to upstream, so the PR diff stays the port only.
- 09bb532 and 8f43bf8: `MODE=hip` for build.sh, with INSTALL documenting it, listing where the hip install lands (`.coinor-deps/cupdlp-hip-install/bin/plc`) and noting that it needs the ROCm libraries kept in place.
- f56941f: removes the AMD copyright and author lines from FindHIPConf.cmake, cuda_to_hip.h and cupdlp_cuda_kernels.cu, because the project keeps no per-file headers on its own sources and records attribution in AUTHORS and OWNERSHIP.txt. Comment-only; each file keeps its description comment.

Retested at 8f43bf8 on AMD Instinct MI250X (gfx90a) and AMD Radeon Pro W7800 (gfx1100): testcudalin, testcublas and plc on afiro (Optimal, -464.75, 200 iterations), upstream's CTest suite 4/4, the installed suite 4/4, the NumPy checker, and `MODE=hip bash build.sh` end to end. f56941f changes only comments; the W7800 rebuild and rerun at f56941f gave the same results.

## Upstream reply

I merged current main into this branch, so it no longer conflicts, and every file the port touches keeps the line endings the repository uses for it. I retested on an AMD Instinct MI250X (gfx90a) and an AMD Radeon Pro W7800 (gfx1100): testcudalin, testcublas, afiro, and your new CTest suite and NumPy checker all pass. build.sh also gains a `MODE=hip` option, documented in INSTALL, and I removed the per-file copyright and author lines the port had added, since the project records attribution in AUTHORS and OWNERSHIP.txt.
```
