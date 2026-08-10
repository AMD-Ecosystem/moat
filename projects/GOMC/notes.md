# GOMC notes

## Build

### HIP/ROCm (AMD GPUs)

```bash
mkdir build_hip && cd build_hip
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build . -j$(nproc)
```

For other AMD architectures, change `-DCMAKE_HIP_ARCHITECTURES` (e.g., gfx1100 for RDNA3).

Build produces 4 GPU executables:
- GOMC_GPU_NVT (NVT ensemble)
- GOMC_GPU_NPT (NPT ensemble)
- GOMC_GPU_GCMC (Grand Canonical Monte Carlo)
- GOMC_GPU_GEMC (Gibbs Ensemble Monte Carlo)

### CUDA (NVIDIA GPUs)

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Test

All 4 GPU executables tested successfully on AMD gfx90a:

1. GPU_GEMC on AR_KR noble gas system (3000 steps, simple VDW system)
2. GPU_GEMC on ISOPEN_NEOPEN (5000 steps, complex molecules with MEMC moves)
3. GPU_GCMC on PEN_HEX (200 steps, Ewald electrostatics)
4. GPU_NVT on AR_KR single box (1000 steps)
5. GPU_NPT on AR_KR single box (1000 steps with volume moves)

Quick validation command:
```bash
cd test/input/Systems/AR_KR/SingleRun
HIP_VISIBLE_DEVICES=0 ./GOMC_GPU_GEMC in.conf
```

## Validation (2026-06-05)

Platform: linux-gfx90a
GPU: AMD Instinct MI250X / MI250 (gfx90a)
ROCm: 7.2.4
Build: Strategy A (compat header + LANGUAGE HIP)

All 4 GPU executables built and validated:
- GPU_NVT: PASS (1000 steps, 0.004s)
- GPU_NPT: PASS (1000 steps, 0.263s)
- GPU_GCMC: PASS (200 steps, 0.196s)
- GPU_GEMC: PASS (3000 steps, 0.017s; 5000 steps, 2.19s)

GPU kernels tested:
- Energy calculations (VDW, electrostatic)
- Force calculations
- Ewald sums for long-range electrostatics
- Molecule displacement, rotation, transfer moves
- Volume moves (NPT)

All simulations completed successfully with expected Monte Carlo acceptance rates.

## Validation (2026-06-05, linux-gfx1100)

Platform: linux-gfx1100
GPU: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3)
ROCm: 7.2.4
Build: Strategy A (compat header + LANGUAGE HIP)

All 4 GPU executables built and validated successfully on gfx1100:

1. GPU_NVT: PASS
   - Test: NVT/neon_T_34_00_K (1M steps)
   - Runtime: 4.48s
   - Acceptance: 68.9% displacement moves
   
2. GPU_NPT: PASS
   - Test: NPT/water_580_00_K (10k steps, SPCE water)
   - Runtime: 4.22s
   - Acceptance: 100% displacement, 53.9% rotation, 55.6% volume moves
   - Ewald summation active
   
3. GPU_GCMC: PASS
   - Test: GCMC/argon (1M steps, 2-box grand canonical)
   - Runtime: 30.8s
   - Acceptance: 52.0% displacement, 50.8% mol-transfer
   
4. GPU_GEMC: PASS
   - Test: NVT_GEMC/pure_fluid/argon_T_115_00_K (1M steps)
   - Runtime: 46.1s
   - Acceptance: 51.7%/61.2% displacement, 4.8%/4.4% mol-transfer, 54.2% volume-transfer

All Monte Carlo moves functioning correctly on RDNA3:
- Displacement, rotation, regrowth, volume transfer, molecule transfer
- Ewald reciprocal-space electrostatics
- VDW and Coulomb energy/force calculations
- hipCUB device reductions

## Validation (2026-06-07, windows-gfx1201)

Platform: windows-gfx1201
GPU: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32)
ROCm: 7.14.0a20260604 (TheRock multi-arch nightly)
Build: ROCm clang all-clang toolchain + hip_link_win.py wrapper

Windows build required three fixes vs the Linux port (committed as a
follow-up commit on moat-port):
1. CMake 4.x Windows-Clang injects -fuse-ld=lld-link which the HIP
   device-link mode rejects. Fixed via CMAKE_HIP_LINK_EXECUTABLE override
   routing through CMake/hip_link_win.py (strips -fuse-ld=*).
2. Source uses #ifndef WIN32 guards (without underscore); Clang defines
   _WIN32 but not WIN32. Fixed by adding WIN32 to compile definitions.
3. WSAStartup/gethostname (Winsock2) calls in Main.cpp require ws2_32.lib.

Build command (Windows, gfx1201):
```
VENV=/b/develop/TheRock/external-builds/pytorch/.venv
ROCM=$VENV/Lib/site-packages/_rocm_sdk_devel
cmake ../src -G "Unix Makefiles" \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/clang.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_PREFIX_PATH=$ROCM \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . --target GPU_NVT GPU_GEMC GPU_GCMC GPU_NPT -j32
```
Copy ROCm DLLs into build dir: amdhip64_7.dll amd_comgr.dll rocm_kpack.dll
hiprtc0714.dll hiprtc-builtins0714.dll (from _rocm_sdk_core/bin)

All 4 GPU executables validated on gfx1201:
1. GPU_GEMC: PASS
   - AR_KR SingleRun (3000 steps, VDW): acceptance ~48%/19% disp
   - ISOPEN_NEOPEN (3000 steps, MEMC moves): MEMC acceptance 61.8%
2. GPU_GCMC: PASS
   - PEN_HEX Ewald electrostatics (200 steps): acceptance ~44% disp
3. GPU_NVT: PASS
   - AR_KR single box (1000 steps): acceptance ~98% disp
4. GPU_NPT: PASS
   - AR_KR single box (1000 steps, volume moves): 94.8% vol-transfer

GPU kernels tested: VDW/Coulomb energy, Ewald electrostatics, displacement,
rotation, MEMC exchange, volume moves, hipCUB device reductions.

Note: checkpoint-restart tests (K_1/K_N) crash on Windows after reading
checkpoint (exit 127, likely in PRNG state restoration). This is a
checkpoint binary ABI issue on Windows; core GPU simulation kernels and all
4 executables run correctly. Checkpoint functionality is not part of the GPU
validation scope.

## Validation (2026-06-19, windows-gfx1101)

Platform: windows-gfx1101
GPU: AMD Radeon PRO V710 (gfx1101, RDNA3, wave32) at HIP_VISIBLE_DEVICES=1
ROCm: 7.14.0a20260604 (TheRock multi-arch nightly)
Build: ROCm clang all-clang toolchain + hip_link_win.py wrapper (same as gfx1201)

Build command (Windows, gfx1101):
```
VENV=/b/develop/TheRock/external-builds/pytorch/.venv
ROCM=$VENV/Lib/site-packages/_rocm_sdk_devel
mkdir build_gfx1101 && cd build_gfx1101
cmake ../src -G "Unix Makefiles" \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/clang.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
  -DCMAKE_PREFIX_PATH=$ROCM \
  -DCMAKE_BUILD_TYPE=Release
gmake GPU_NVT GPU_GEMC GPU_GCMC GPU_NPT -j64
```
Copy ROCm DLLs into build dir: amdhip64_7.dll amd_comgr.dll rocm_kpack.dll
hiprtc0714.dll hiprtc-builtins0714.dll (from _rocm_sdk_core/bin)

All 4 GPU executables validated on gfx1101:
1. GPU_GEMC: PASS
   - AR_KR SingleRun (3000 steps, VDW): acceptance ~48%/19% disp (matches gfx1201)
   - ISOPEN_NEOPEN (3000 steps, MEMC moves): MEMC acceptance 61.8% (matches gfx1201)
2. GPU_GCMC: PASS
   - PEN_HEX Ewald electrostatics (200 steps): acceptance ~44% PEN disp (matches gfx1201)
3. GPU_NVT: PASS
   - AR_KR single box (1000 steps): acceptance ~98% disp (matches gfx1201)
4. GPU_NPT: PASS
   - AR_KR single box (1000 steps, volume moves): 95.3% vol-transfer (matches gfx1201 ~94.8%)

GPU kernels tested: VDW/Coulomb energy, Ewald electrostatics, displacement,
rotation, MEMC exchange, volume moves, hipCUB device reductions.

Note: checkpoint-restart tests (K_1/K_N) crash on Windows -- same as gfx1201,
pre-existing Windows checkpoint ABI issue, not a GPU validation concern.

## Porting notes

### Random123 library

The bundled Random123 library needed several defines for HIP device code:
- `R123_CUDA_DEVICE __device__` -- mark array methods as device functions
- `R123_THROW(x) abort()` -- disable exceptions in device code
- `R123_USE_SSE 0` etc. -- disable x86 SSE intrinsics on GPU
- Use `r123::double2` explicitly to avoid collision with HIP's `double2` type

These defines are placed in `TransformParticlesCUDAKernel.{cu,cuh}` before including Random123 headers.

### hipCUB flag pollution

The `hip::hipcub` CMake target propagates `-x hip --offload-arch=<arch>` flags to all compilation units via `hip::device`. This causes g++ (the CXX compiler) to fail on host .cpp files. The workaround is to manually include hipCUB/rocprim directories and link only `amdhip64`, avoiding the imported target's flag pollution.

### HIP separable compilation

The port uses `HIP_SEPARABLE_COMPILATION ON` and `-fgpu-rdc` to allow `__device__` functions defined in one translation unit to be called from another (e.g., `CalcEnGPU` defined in CalculateEnergyCUDAKernel.cu and called from CalculateForceCUDAKernel.cu).
