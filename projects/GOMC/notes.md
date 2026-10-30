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

## Retarget onto upstream `development` (2026-10-30, linux-gfx90a, porter)

Requested by jeff: upstream CONTRIBUTING.md asks for PRs against `development`, so the port now sits on it (103 commits ahead of upstream `main` at merge time). The base-branch change in the records and the fork's `development` mirror are handled separately by the primary.

Fork `moat-port`: 58d5496d -> 4c06552c (three new commits on top, nothing rewritten):
- `faacd96a` merge commit of upstream `development` @ 09ec304e (`git merge --no-ff upstream/development`). Textually clean. The 4 files changed on both sides were reviewed by hand against `upstream/development`: CMakeLists.txt (development's compiler-ID chain plus new `else() FATAL_ERROR` sits before the USE_HIP/CUDA selection; it does not affect HIP because CXX stays g++), src/GPU/VariablesCUDA.cuh (development's `printFreeMemory` rewrite, `exit(cuda_status)` compiles under the compatibility header), src/Main.cpp and src/MoleculeLookup.cpp (host-side changes next to the header swap / HIP memoryClockRate branch). development adds no .cu/.cuh files, kernels or CUDA runtime calls (`git diff upstream/main upstream/development | grep '^+' | grep -E 'cuda|__global__|<<<|cub::'` hits only README prose and the printFreeMemory lines). New header src/FFTabulated.h is host-only.
- `16cb7352` fix: GOMCHIPSetup.cmake used directory-scoped `add_compile_definitions(GOMC_CUDA USE_HIP __HIP_PLATFORM_AMD__)`, so the CPU ensembles (NVT/NPT/GCMC/GEMC) in a USE_HIP build were compiled with GOMC_CUDA and failed to link (undefined `UpdateGPULambda`, `CallTranslateParticlesGPU`, `CUDAMemoryManager::isFreed`, ...). Pre-existing since the original port (earlier rounds only built the GPU targets); a plain `cmake --build` of a USE_HIP config failed. Now per-target via `${GPU_HIP_defs}`.
- `4c06552c` README: "Building GOMC for AMD GPUs with ROCm" section (the port had never documented the HIP build in the repo, only in the review PR body).

### Build (this host, TheRock ROCm SDK 10.1.0 pip wheels, no /opt/rocm)

```bash
R=$(rocm-sdk path --root)
cmake -S projects/GOMC/src -B gomc-build-hip -DUSE_HIP=ON -DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=$R -DCMAKE_BUILD_TYPE=Release -DGOMC_GTEST=ON
cmake --build gomc-build-hip -j 128
```
All 8 executables (GOMC_CPU_{NVT,NPT,GCMC,GEMC}, GOMC_GPU_{NVT,NPT,GCMC,GEMC}) build. `utils/timeit.sh` cds to the MOAT root, so pass `-S/-B` absolute paths, never a relative `..`.

CPU gtests (`GOMC_GTEST=ON`: GOMC_{NVT,NPT,GCMC,GEMC}_Test) do NOT compile on upstream `development` itself: upstream commit bc0cd248 ("Add missing check, fix comment typo...") leaves an unbalanced brace in test/src/ConsistentTrajectoryTest.cpp `Check_Neo_Pen`, so the following TEST()s nest inside it (`local class ... shall not have static data member`). Port does not touch test/. Upstream bug, not fixed here. GPU gtests (test/BuildGPUTests.cmake) are gated on `CMAKE_CUDA_COMPILER` and so are not built in a HIP configuration (unchanged from earlier rounds).

### GPU runs (gfx90a, MI250X/MI250 GCD, HIP_VISIBLE_DEVICES=0, `ulimit -c 0`)

Script: copies test/input/Systems to a scratch dir, sets `ConsoleFreq true 100`, derives single-box AR_KR NVT (1000 steps, Dis 0.87/Rot 0.13) and NPT (1000 steps, Dis 0.86/Rot 0.13/Vol 0.01, Pressure 100) configs from AR_KR/SingleRun/in.conf, runs each with the GPU and the CPU executable of the same build and compares every ENER_n point.

| run | GPU exit | ENER points | GPU vs CPU max rel diff | acceptance (GPU = CPU) |
|---|---|---|---|---|
| GEMC AR_KR SingleRun 3000 steps | 0 | 62 | 0 (8 sig. figs) | disp AR 48.30% / KR 19.04% (box 0), matches recorded ~48/19 |
| GEMC ISOPEN_NEOPEN SingleRun 3000 steps (MEMC) | 0 | 62 | 0 | MEMC NEO-ISB 61.76%, matches recorded 61.8% |
| GCMC PEN_HEX SingleRun 200 steps (Ewald) | 0 | 6 | 0 | PEN disp 43.75%, matches recorded ~44% |
| NVT AR_KR box 0, 1000 steps | 0 | 11 | 0 | disp AR 97.2% / KR 98.4% |
| NPT AR_KR box 0, 1000 steps | 0 | 11 | 0 | disp 98.0/97.6%, volume 50.0% |

GPU timings: GEMC ISOPEN 2.52 s, PEN_HEX 0.158 s, others < 0.02 s. BPTI_TIP3/K_CHANNEL_TIP3 SingleRun configs are checkpoint-test fixtures and are not runnable as GCMC inputs (both CPU and GPU: "chemical potential is missing for PROTA").

### CUDA compile gate (CUDA 12.8)

The PyPI `nvidia-cuda-nvcc-cu12` 12.8.x wheels (12.8.61, 12.8.93; also on pypi.nvidia.com) contain only `ptxas` + nvvm, no `nvcc`/`nvlink`/`cudafe++`, and the runtime wheel has no `libcudadevrt.a`/`libcudart_static.a`. Installed the wheels into the job venv as asked, then used NVIDIA's official redist archives for the same release (sha256 verified against redistrib_12.8.1.json): cuda_nvcc 12.8.93, cuda_cudart 12.8.90, cuda_cccl 12.8.90, extracted into one root with `--strip-components=1`, plus `ln -s lib lib64` (nvcc links from `lib64`).

```bash
cmake -S projects/GOMC/src -B gomc-build-cuda -DCMAKE_CUDA_COMPILER=$C/bin/nvcc -DCUDAToolkit_ROOT=$C \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release
cmake --build gomc-build-cuda -j 128     # rc=0, 0 warnings, 138 s, all 8 executables
```
Upstream GOMCCUDASetup.cmake overrides `CMAKE_CUDA_ARCHITECTURES` to `all` on CMake >= 3.23 (flags show `-arch=all`), so `-DCMAKE_CUDA_ARCHITECTURES=80` is ignored; sm_80 is included in `all`. Upstream behavior, left alone.

### Jargon

`python3 utils/jargon.py --port GOMC` reports 3 hits, all upstream development's own comment "I'm the follower" in src/ParallelTemperingUtilities.cpp (lines 449/537/817), because `--port` scans from `fork_default_branch` (`main`), so development's 103 commits count as added. Against the real base: `jargon.py -C projects/GOMC/src --commits upstream/development..moat-port` and `--diff upstream/development...moat-port` are both clean. Becomes clean under `--port` once the records' base branch is `development`.

Update: after the primary recorded `fork_default_branch: development` (fork `development` mirror = upstream 09ec304e, the merged commit), `git -C projects/GOMC/src branch -f development origin/development` then `python3 utils/jargon.py --port GOMC` -> clean. A fresh fork clone needs that local `development` branch for `--port` to resolve.
