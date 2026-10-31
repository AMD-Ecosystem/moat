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

## Review 2026-10-30 (linux-gfx942, reviewer)

Scope: whole fork diff `git diff development...moat-port` at 4c06552c (base `development` = upstream 09ec304e; 22 files, +379/-49), with the merge faacd96a's four both-sides files read against `git diff 785c2ceb development` (merge base 785c2ceb). Verdict: review-passed, no blocking findings.

Checked:
- Merge faacd96a: `git show --cc` is empty for CMakeLists.txt, VariablesCUDA.cuh, Main.cpp and MoleculeLookup.cpp, so there are no hand-resolved hunks. development's changes there are host-only: the compiler-ID chain plus `else() FATAL_ERROR` (CXX stays GNU in a HIP configuration, so it never fires), the `printFreeMemory` rewrite (`exit(cuda_status)` takes an unscoped `hipError_t` enum, which converts implicitly to int), and `InputFileReader`, hostname and `operator=` changes away from the port's hunks. No CUDA symbol in `src/` or `lib/` is missing from `cuda_to_hip.h` (grep for `cuda[A-Z]*`, `CUDART_*`, `cub::`, `Cub*`, `__shfl*`, `__ballot*`, `warpSize`; the only CUDART_VERSION uses are the NVTX block, gated off under USE_HIP, and the Main.cpp `#elif` after the HIP branch).
- 16cb7352 CMake scoping: `GOMCHIPSetup.cmake` is only included under `USE_HIP`, and `GOMCCUDASetup.cmake`/`GOMCCPUSetup.cmake` are byte-identical to development, so the CUDA build's flags are unchanged (it still adds `-DGOMC_CUDA` through `CMAKE_HOST_COMP_FLAGS`/`CMAKE_CUDA_COMP_FLAGS` on the GPU targets). The CUDA compile gate (nvcc 12.8.93, all 8 executables) recorded above still stands. The leftover directory-scoped `add_compile_definitions(WIN32 _USE_MATH_DEFINES)` affects only Windows and matches what CMake's MSVC defaults give the CPU targets. The CPU targets already link ws2_32 on WIN32.
- Fault classes: no warp intrinsics, `warpSize`, textures, streams or lane masks in the GPU code; hipCUB reductions only; the double-`atomicAdd` polyfill is excluded on AMD; Random123 is configured for HIP device code. Nothing the merge added changes this.
- README section: commands match the build; no device-selection environment variables.

Evidence on this host (TheRock ROCm SDK 10.1.0 wheels, CMake via the job venv):
```bash
R=$(rocm-sdk path --root)
cmake -S projects/GOMC/src -B agent_space/gomc-hip -DUSE_HIP=ON -DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_PREFIX_PATH=$R -DCMAKE_BUILD_TYPE=Release
cmake --build agent_space/gomc-hip -j 64     # rc=0, all 8 executables (GOMC_CPU_* and GOMC_GPU_*)
grep -c GOMC_CUDA agent_space/gomc-hip/CMakeFiles/NVT.dir/flags.make      # 0
grep -c GOMC_CUDA agent_space/gomc-hip/CMakeFiles/GPU_NVT.dir/flags.make  # 2
```
Smoke run only, not the validation gate: GOMC_GPU_GEMC and GOMC_CPU_GEMC on AR_KR/SingleRun (MI300X, HIP_VISIBLE_DEVICES=4, `ulimit -c 0`) both exit 0 with identical ENER_0 output. `python3 utils/jargon.py --port GOMC` is clean, and so is `git -C projects/GOMC/src status --porcelain`.

Not blocking: `moatlib.py audit-commits GOMC` reports that 58d5496d and b1b67078 (both earlier rounds, already review-passed) have Test Plans with no fenced command block. Fixing that means rewriting their messages, so leave it to the PR-prep squash; no new commit can fix it.

## Validation 2026-10-30 (linux-gfx90a, validator)

Head 4c06552c (fork `moat-port`, fresh clone). Full real-GPU revalidation (prior gfx90a evidence was at 58d5496d; the upstream `development` merge is not a carry-forward candidate).

GPU: AMD Instinct MI250X / MI250 GCD (gfx90a, wave64), HIP_VISIBLE_DEVICES=0, `ulimit -c 0`. ROCm: TheRock ROCm SDK 10.1.0 pip wheels (HIP 7.16.26385), no /opt/rocm.

Build, following the README ROCm section with the SDK compiler path (no `-DCMAKE_HIP_ARCHITECTURES`, to exercise the README's autodetection: CMakeCache shows `gfx90a`):
```bash
R=$(rocm-sdk path --root)
utils/timeit.sh GOMC compile -- bash -c "cmake -S $PWD/projects/GOMC/src -B $PWD/agent_space/gomc-hip -DUSE_HIP=ON \
  -DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$R -DCMAKE_BUILD_TYPE=Release && cmake --build $PWD/agent_space/gomc-hip -j 32"
```
rc=0, all 8 executables (GOMC_CPU_{NVT,NPT,GCMC,GEMC}, GOMC_GPU_{NVT,NPT,GCMC,GEMC}). Warnings: 4978 `-Wunused-value` + 20 `-Wunused-result`, all `ignoring return value of type 'hipError_t' declared with 'nodiscard'` where upstream ignores the return of `CUMALLOC`/`cudaMemcpy` etc. in src/GPU/*.cu (HIP declares `hipError_t` nodiscard, CUDA does not). Cosmetic, not a gate.

Tests: `utils/timeit.sh GOMC test -- bash agent_space/gomc_val.sh <build> <src> <work>`. The script copies test/input/Systems, sets `ConsoleFreq true 100` (PEN_HEX: 40), and derives single-box AR_KR NVT (1000 steps, Dis 0.87/Rot 0.13) and NPT (1000 steps, Dis 0.86/Rot 0.13/Vol 0.01, Pressure 100) configs from AR_KR/SingleRun/in.conf by dropping the box-1, GEMC, SwapFreq and ChemPot lines. It runs each with the GPU and the CPU executable of the same build, then compares every ENER_n row field by field and every `% Accepted` line. The GPU log reports `Device Name: AMD Instinct MI250X / MI250`.

| run | GPU/CPU rc | ENER rows | GPU vs CPU max rel diff | acceptance lines GPU vs CPU | key acceptance |
|---|---|---|---|---|---|
| GEMC AR_KR SingleRun 3000 steps | 0/0 | 62 | 0 | identical | disp AR 48.30% / KR 19.04% (box 0) |
| GEMC ISOPEN_NEOPEN SingleRun 3000 steps (MEMC) | 0/0 | 62 | 0 | identical | MEMC NEO-ISB 61.76% |
| GCMC PEN_HEX SingleRun 200 steps (Ewald) | 0/0 | 12 | 0 | identical | PEN disp 43.75% |
| NVT AR_KR box 0, 1000 steps | 0/0 | 11 | 0 | identical | disp AR 97.22% / KR 98.39% |
| NPT AR_KR box 0, 1000 steps | 0/0 | 11 | 0 | identical | volume 50.0% |

5/5 pass. GPU and CPU trajectories are identical at printed precision, and the results match the porter's gfx90a numbers at this head and the earlier rounds' recorded acceptances. GPU wall times: ISOPEN 4.34 s, PEN_HEX 0.61 s, others ~0.3 s (the timings include HIP init; the host was heavily loaded).

Non-GPU: the CPU executables from the same build ran all five systems with rc=0. The CPU gtests do not compile on upstream `development` itself (unbalanced brace in test/src/ConsistentTrajectoryTest.cpp, recorded above). That is upstream breakage, the port leaves test/ untouched, and I did not re-run it here.

CUDA compile gate: already recorded at 4c06552c (nvcc 12.8.93, all 8 executables), so it was not re-run.

Completion checks: `git -C projects/GOMC/src status --porcelain` is empty, `python3 utils/jargon.py --port GOMC` is clean (after `git -C projects/GOMC/src branch -f development origin/development`), and the README carries the "Building GOMC for AMD GPUs with ROCm" section.
## Validation 2026-10-30 (linux-gfx1100, validator)

Requested by jeff after the `development` retarget passed review. Fork `moat-port` @ 4c06552c (= `origin/moat-port`). Previous gfx1100 evidence was at 58d5496d; `moatlib.py classify GOMC 58d5496d 4c06552c` -> `mixed` (the upstream `development` merge touches ~100 host files), so full re-run, no carry-forward.

Host: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), HIP_VISIBLE_DEVICES=0, `ulimit -c 0`. Toolchain: TheRock ROCm SDK 10.1.0 pip wheels (no /opt/rocm), AMD clang 24.0.0git, g++ 13.3.0, CMake 3.31.6.

### Build (README "Building GOMC for AMD GPUs with ROCm" recipe, compiler path adapted to the SDK, no arch given)

```bash
R=$(rocm-sdk path --root)
cmake -S $PWD/projects/GOMC/src -B $PWD/agent_space/gomc-hip -DUSE_HIP=ON -DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=$R -DCMAKE_BUILD_TYPE=Release
cmake --build $PWD/agent_space/gomc-hip -j 32    # rc=0, 38 s, all 8 executables
```
The README's claim that the arch is detected from the installed GPU holds: with no `-DCMAKE_HIP_ARCHITECTURES`, the cache holds `gfx1100;gfx1100;gfx1100;gfx1100` (one entry per GPU on the host, harmless) and the GPU targets compile with `--offload-arch=gfx1100` only. Warnings: 5078 `-Wunused-value` + 20 `-Wunused-result`, all "ignoring return value of type 'hipError_t' declared with 'nodiscard'" (ROCm 10 marks `hipError_t` nodiscard; upstream ignores the same `cudaError_t` returns). Not a port defect, not a gate.

### GPU runs vs CPU, same build

Same procedure as the gfx90a retarget round: copy test/input/Systems to a scratch dir, `ConsoleFreq true 100`, single-box AR_KR NVT (1000 steps, Dis 0.87/Rot 0.13) and NPT (1000 steps, Dis 0.86/Rot 0.13/Vol 0.01, Pressure 100) derived from AR_KR/SingleRun/in.conf by dropping the box-1, GEMC, ChemPot and swap lines; each case runs `GOMC_GPU_<ens> in.conf` and `GOMC_CPU_<ens> in.conf` from the config directory, and every `ENER_n` console line plus every `% Accepted` line is compared. GPU log reports `Device Name: AMD Radeon Pro W7800 48GB`.

| run | GPU / CPU exit | ENER points | GPU vs CPU max rel diff | acceptance (GPU = CPU, identical lines) | GPU wall |
|---|---|---|---|---|---|
| GEMC AR_KR SingleRun 3000 steps | 0 / 0 | 62 | 0 | disp AR 48.30% / KR 19.04% (box 0); same as gfx90a | 0.16 s |
| GEMC ISOPEN_NEOPEN SingleRun 3000 steps (MEMC) | 0 / 0 | 62 | 0 | MEMC NEO-ISB 61.76%; same as gfx90a | 1.76 s |
| GCMC PEN_HEX SingleRun 200 steps (Ewald) | 0 / 0 | 6 | 0 | PEN disp 43.75%; same as gfx90a | 0.30 s |
| NVT AR_KR box 0, 1000 steps | 0 / 0 | 11 | 0 | disp AR 97.22% / KR 98.39% | 0.14 s |
| NPT AR_KR box 0, 1000 steps | 0 / 0 | 11 | 0 | disp 98.00/97.55%, volume 50.0% | 0.15 s |
| GEMC ISOPEN_NEOPEN 30000 steps (extra, ConsoleFreq 1000) | 0 / 0 | 62 | 0 | MEMC 54.05%, mol-transfer ISB 41.01% | 16 s |
| GCMC PEN_HEX 2000 steps (extra, Ewald) | 0 / 0 | 42 | 0 | PEN disp 24.81%, mol-transfer PEN 19.05% | 2 s |

Result: 7/7 GPU runs pass with GPU output identical to CPU at the printed precision. CPU executables of the same HIP build (the non-GPU path) build and run on every case. CPU gtests: not built, upstream `development` itself fails to compile them (ConsistentTrajectoryTest.cpp brace, see the retarget section); GPU gtests are CUDA-only. Unchanged from the retarget round.

CUDA compile gate: already recorded at 4c06552c (nvcc 12.8.93, retarget section), not re-run.

Completion checks: `git -C projects/GOMC/src status --porcelain` empty; `moatlib.py audit-clean GOMC` clean; `python3 utils/jargon.py --port GOMC` clean (needs local `development` and `moat-port` branches in the fork clone: `git branch -f development origin/development; git branch -f moat-port origin/moat-port`); ROCm build documented in README.

Gotcha: `utils/timeit.sh` cds to the MOAT root, so GOMC runs (relative paths in in.conf) must be wrapped as `utils/timeit.sh GOMC test -- bash -c "cd <config dir> && <exe> in.conf"`.

## Validation 2026-10-30 (windows-gfx1151, validator) -- validation-failed at 4c06552c

Requested by jeff. Fork `moat-port` @ 4c06552c, fresh clone, tree clean. Host: Windows 11, AMD Radeon 8060S (gfx1151, RDNA3.5, wave32, Strix Halo APU); TheRock ROCm SDK (clang 23.0.0git, HIP 7.14 runtime DLLs staged next to each exe, `ulimit -c 0`, one GPU process at a time, `-j16`). The Windows gate was last met at 58d5496d (gfx1201/gfx1101); the `development` merge added upstream host code that the Windows all-clang toolchain rejects.

### Build at exactly 4c06552c: FAILS (host .cpp, upstream `development` code; GPU kernels compile)

```bash
R=D:/Develop/TheRock/.venv/Lib/site-packages/_rocm_sdk_devel
cmake -S $SRC -B $B -G 'Unix Makefiles' -DCMAKE_MAKE_PROGRAM=C:/Strawberry/c/bin/gmake.exe -DUSE_HIP=ON \
  -DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++.exe -DCMAKE_C_COMPILER=$R/lib/llvm/bin/clang.exe \
  -DCMAKE_CXX_COMPILER=$R/lib/llvm/bin/clang++.exe -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DCMAKE_PREFIX_PATH=$R -DCMAKE_BUILD_TYPE=Release
cmake --build $B -j16        # rc=2, no executable produced (GPU_* targets alone fail the same way)
```
Configure succeeds (the three 2026-06 Windows fixes are present). Three independent compile errors, none in port-touched code, all in lines upstream `development` added after the 58d5496d base:
1. `src/CalculateEnergy.cpp:332/334`: `variable 'atomCount' / 'molCount' must have explicitly specified data sharing attributes` -- `#pragma omp parallel for default(none)` with `aForcex[ : atomCount]` reductions (upstream f9bef633 / 239b6884); g++ accepts it, clang does not. Fix: add `atomCount, molCount` to `shared(...)`.
2. `src/CellList.cpp:28`: `array initializer must be an initializer list` -- copy ctor mem-initializes the array members `neighbors(other.neighbors), head(other.head)` (upstream 6e2420e2, 2026-06-24); a GCC extension, ill-formed C++. Fix: copy per box in the existing loop.
3. `src/Main.cpp:106`: `std::max(1, numThreads)` hits the `max` macro of `<windows.h>` (pulled in by winsock). Fix: `-DNOMINMAX` (no source edit needed), or `(std::max)`.
Linux never sees these because the host compiler there is g++.

### Diagnostic evidence from an UNCOMMITTED scratch copy (not the gate; source differs from 4c06552c by the 1+2 edits above plus `-DCMAKE_CXX_FLAGS=-DNOMINMAX`)

With those edits all 8 executables build (rc=0). Five cases, GPU = `GOMC_GPU_*`, CPU = `GOMC_CPU_*` of the same build; every ENER_n row and every acceptance line compared (`agent_space/gomc_val_win.py`). Device Name: AMD Radeon(TM) 8060S Graphics.

| run | GPU/CPU exit | ENER rows | max rel diff | acceptance lines | GPU wall |
|---|---|---|---|---|---|
| GEMC AR_KR SingleRun 3000 | 0/0 | 62 | 0 | 67/67 identical | 0.32 s |
| GEMC ISOPEN_NEOPEN 3000 (MEMC) | 0/0 | 62 | 0 | 70/70 identical (MEMC NEO-ISB 61.76%, as Linux) | 2.61 s |
| GCMC PEN_HEX 200 (Ewald) | 0/0 | 12 | 0 | 23/23 identical | 0.48 s |
| NVT AR_KR box 0, 1000 | 0/0 | 11 | 0 | 15/15 identical | 0.25 s |
| NPT AR_KR box 0, 1000 | 0/0 | 11 | 0 | 16/16 identical | 0.28 s |

So the GPU path is expected to pass on gfx1151 once the host source builds; this does not count as validation of 4c06552c. No hang, no crash, no bugcheck.

### K_1/K_N checkpoint exit 127 (CPU executables only, no GPU)

Reproduced: `Base` (CPU GEMC, rc=0) writes `base_restart.chk`; `K_1` (reads it) exits 127 right after `Random number seed: 123`; `K_N` then fails only because K_1 wrote no output. Not a missing DLL: `dumpbin /dependents GOMC_CPU_GEMC.exe` shows only WS2_32, MSVCP140, VCRUNTIME140, libomp140, UCRT api-sets, all resolved (Base runs with the same exe). Cause: `CheckpointSetup.cpp:47` `std::ifstream ifs(filename)` and `CheckpointOutput.cpp:58` `std::ofstream ofs(filename)` open the cereal `BinaryInputArchive`/`BinaryOutputArchive` streams in TEXT mode; on Windows the CRT translates `\n` (0x0A) and treats 0x1A as EOF, corrupting the binary checkpoint. With `std::ios::binary` added to both opens (scratch copy), AR_KR Base/K_1/K_N and ISOPEN_NEOPEN Base/K_1/K_N all exit 0. Verdict: UPSTREAM Windows bug in code the port does not touch (`git diff origin/development..moat-port -- src/CheckpointSetup.cpp src/CheckpointOutput.cpp` is empty; the lines are upstream's). Not an environment issue and not a port defect; the old "Windows checkpoint ABI" note was this. An upstream CPU-only comparison was not run: the same clang errors above block a clang build of `development`, and the cause is evident from the source. Fixing it is optional for the port; it is a candidate for a separate upstream PR (not bundled into moat-port).

### Result

validation-failed at 4c06552c: the Windows all-clang build breaks on upstream `development` host code (items 1 and 2 need small source edits in moat-port; item 3 is a flag, best set in `GOMCHIPSetup.cmake` next to the existing `WIN32 _USE_MATH_DEFINES` definitions as `NOMINMAX`). Instructed not to change source, so none pushed. After the porter commits those, this arch returns as `port-ready` for a full run.
