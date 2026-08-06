# KittenGpuLBVH notes

Fork: https://github.com/AMD-Ecosystem/KittenGpuLBVH (branch `moat-port`)
Upstream: https://github.com/jerry060599/KittenGpuLBVH (same author as YarnBall).
Lead: linux-gfx90a (MI250X, CDNA2, wave64), ROCm 7.2.1.

Standalone LBVH (linear BVH) collision library. CUDA-only upstream (the .cu
sources include `<cuda_runtime.h>`/`<device_launch_parameters.h>` unconditionally;
the only build is the Visual Studio solution).

## Why this is its own MOAT project

It is the `KittenGpuLBVH` git submodule of the `YarnBall` port
(`KittenEngine/KittenEngine/KittenGpuLBVH`). YarnBall's HIP build needs the
submodule's CUDA-only includes guarded for HIP, so the change cannot ship only
inside YarnBall's fork: an upstream YarnBall PR must point its submodule at an
UPSTREAM KittenGpuLBVH commit, not at our fork. So we upstream the LBVH HIP
support here first, then re-pin YarnBall's submodule to the merged upstream SHA
and restore YarnBall's `.gitmodules` to the upstream URL. See YarnBall notes
("KittenGpuLBVH submodule") and the deferred entry `yarnball-pr-blocked-on-lbvh`.

The port is based on current upstream main (50ecaabd "Disambiguate any() calls"),
one commit ahead of the old commit YarnBall pinned (8964555).

## Port shape

- `lbvh.cu` / `lbvh.cuh`: CUDA-only runtime includes guarded behind
  `#if !defined(USE_HIP)`; spaced `<< <`/`>> >` launches normalized to `<<<`/`>>>`.
- `cuda_to_hip.h` (new): shim that includes `<hip/hip_runtime.h>` and aliases the
  `cudaXxx` runtime/error surface to `hipXxx`. Force-included on each HIP TU.
- `KittenEngine/includes/modules/Common.h`: a `USE_HIP` branch (parallel to the
  `__has_include("cuda_runtime.h")` branch) decorating `KITTEN_FUNC_DECL` with
  `__device__ __host__` and providing `gpuAssert`.
- `CMakeLists.txt` (new): standalone build of the `testLBVH()` sample
  (`main.cpp` + `lbvh.cu`) as `lbvh_test`. CUDA by default, ROCm with `-DUSE_HIP=ON`.
  GLM 1.0.1 via FetchContent (system GLM 0.9.9.8 lacks the `__HIP__` device path).

## Build + validate (gfx90a, real GPU)

```bash
cd <fork clone>
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/lbvh_test
```

The sample self-validates: builds a 100K-object LBVH on the GPU, queries all
collision pairs, and compares against a brute-force CPU pass. Pass =
"CPU and GPU results match." Validated 2026-06-19 on MI250X gfx90a: 338 pairs
GPU == 338 pairs CPU, self check complete. Binary embeds
`hipv4-amdgcn-amd-amdhsa--gfx90a`.

Follower archs: only `-DCMAKE_HIP_ARCHITECTURES=<arch>` differs (gfx1100, gfx1201).

## Validation 2026-06-19

### windows-gfx1101 (Radeon PRO V710, RDNA3 gfx1101, TheRock ROCm 7.14, Windows 11)

GPU: AMD Radeon PRO V710, HIP_VISIBLE_DEVICES=1 (gfx1201 at mask 0). Verified via hipInfo.

Build (Ninja, all-clang from `_rocm_sdk_devel`, -j64):

```bat
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
cmake -G Ninja -S <src> -B build_gfx1101 -DUSE_HIP=ON \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/clang.exe \
  -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$ROCM
cmake --build build_gfx1101 -j64
```

Build succeeded (8 nodiscard/format warnings, no errors -- identical to gfx1201). TheRock
runtime DLLs (amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll,
hiprtc-builtins0714.dll) copied from `_rocm_sdk_core/bin` into `build_gfx1101/`.

Test run (HIP_VISIBLE_DEVICES=1):

```
Generating Data...
Building LBVH...
Querying LBVH...
Getting results...
356 collision pairs found on GPU.

Running brute force CPU collision detection...
356 collision pairs found on CPU.
CPU and GPU results match.

LBVH self check...
Max stack size: 19
Node size: 64
LBVH self check complete.
```

PASS. 356 GPU pairs == 356 CPU pairs; self-check passed. Result matches gfx1201 (356 pairs).
No TDR / card detach observed. sha 33c0f78. Status: PASS.

### linux-gfx1100 (W7800, gfx1100, ROCm 7.2.1, wave32)

Built at sha 33c0f78 with `-DCMAKE_HIP_ARCHITECTURES=gfx1100`. Binary embeds
`hipv4-amdgcn-amd-amdhsa--gfx1100`. Test run with `HIP_VISIBLE_DEVICES=0`:

```
338 collision pairs found on GPU.
338 collision pairs found on CPU.
CPU and GPU results match.
LBVH self check complete.
```

Result matches gfx90a (338 pairs). No wave-size issues: the BVH construction
uses Thrust sort/scan primitives which adapt to wave32 on RDNA3 without
warp-level assumptions. Status: PASS.

## Validation 2026-06-18

### windows-gfx1201 (RX 9070 XT, RDNA4 gfx1201, TheRock ROCm 7.14, Windows 11)

GPU: AMD Radeon RX 9070 XT, HIP_VISIBLE_DEVICES=0 (gfx1101 absent/detached this session).

Build (Ninja, all-clang from `_rocm_sdk_devel`):

```bat
ROCM=_rocm_sdk_devel
cmake -G Ninja -B build_gfx1201 -DUSE_HIP=ON \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/clang.exe \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$ROCM
cmake --build build_gfx1201 -j16
```

Build succeeded (8 nodiscard/format warnings, no errors). TheRock runtime DLLs
(amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll,
hiprtc-builtins0714.dll) copied from `_rocm_sdk_core/bin` into `build_gfx1201/`
so the exe's own directory search wins over System32's Adrenalin amdhip64.

Test run:

```
356 collision pairs found on GPU.
356 collision pairs found on CPU.
CPU and GPU results match.
LBVH self check complete.
```

PASS. 356 GPU pairs == 356 CPU pairs; self-check passed. No cooperative launch
used (plain kernel launches only, no hipLaunchCooperativeKernel), so the
gfx1201/TheRock-7.14 cooperativeLaunch=0 limitation does not apply.

## Install as a dependency

YarnBall consumes this as a vendored git submodule, not an installed package, so
there is no `CMAKE_PREFIX_PATH` install step. To finish YarnBall after this lands
upstream: in the YarnBall fork, set `.gitmodules` url back to
`https://github.com/jerry060599/KittenGpuLBVH` (drop the `branch` line), set the
submodule gitlink to the merged upstream KittenGpuLBVH SHA, then re-validate
YarnBall headless on each arch.
