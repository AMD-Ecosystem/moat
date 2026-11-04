# cuda_voxelizer notes

## What it is
`Forceflow/cuda_voxelizer` (aka cuda_voxelizer, by Jeroen Baert) is a small command-line tool that converts polygon meshes (.ply/.off/.obj/.3DS/.SM/RAY) into voxel grids (.vox/.binvox/.obj cubes/point cloud/morton grid). Not archived, actively maintained (last push 2026-06-16), 651 stars.

## Licence
Top-level: MIT (SPDX `MIT`), tier 1, cleared to contribute. Verified by reading `LICENSE` directly (`utils/licenses.py check` also reported `license=MIT tier=1` without an UNPARSED warning, and this was cross-checked against the file text).

**Per-file caveat found on inspection (not a top-level licence problem, but worth a person's eyes before the port touches these two files):** `src/libs/cuda/helper_cuda.h` and `src/libs/cuda/helper_string.h` are vendored copies of the classic NVIDIA CUDA Samples helper headers and carry the old (pre-2017) NVIDIA CUDA Samples header text:

```
Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
Please refer to the NVIDIA end user license agreement (EULA) associated
with this source code for terms and conditions that govern your use of
this software. Any use, reproduction, disclosure, or distribution of
this software and related documentation outside the terms of the EULA
is strictly prohibited.
```

This is restrictive EULA-style text, distinct from an ordinary NVIDIA copyright line under a permissive licence. It did NOT trip `utils/licenses.py`'s `scan_nvidia()` because the exact phrase doesn't match the configured `tier3.nvidia_proprietary.text_markers` strings (those target the newer "NVIDIA Source Code License" / "NVIDIA Software License Agreement" wording); this is the older EULA-referencing boilerplate that shipped with CUDA Samples before NVIDIA relicensed the helper headers to BSD-3-Clause around 2018-2019. A third file in the same directory, `src/libs/cuda/helper_math.h`, already carries the newer, permissive BSD-3-Clause NVIDIA header (`Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.` plus standard BSD-3 redistribution text) -- so the repo has a mix of old-EULA and new-BSD3 vendored NVIDIA headers side by side.

Scope/impact: these two files are pure utility boilerplate (CUDA error-check macros and command-line-arg helpers), not part of the project's own voxelization algorithm. The actual algorithmic core (`src/voxelize.cu`, `src/voxelize_solid.cu`, `src/voxelize.cuh`, `src/util_cuda.cpp`) is clean, carries no NVIDIA header, and is covered by the project's own MIT licence. Only `src/util_cuda.h` includes `helper_cuda.h`; `helper_string.h` is only pulled in transitively by `helper_cuda.h`. A HIP port would very likely replace both files outright (hipification typically swaps `checkCudaErrors` for a small custom macro rather than porting NVIDIA's own header), which would sidestep the issue entirely, but that is a porter-stage decision, not an intake one. Flagging per the "any file carrying an NVIDIA proprietary licence needs a decision before proceeding" rule -- recording it here rather than silently treating the top-level MIT tier as covering the whole tree.

Also checked: `src/libs/magicavoxel_file_writer/` (vendored third-party component, no submodules in the repo) carries its own MIT licence (Copyright (c) 2018 Aiekick) -- clean, no conflict.

No submodules (`.gitmodules` absent).

## Duplicate effort
Searched AMD-Ecosystem and ROCm orgs by name (`gh search repos`, `gh repo list AMD-Ecosystem`): no repo named cuda_voxelizer or containing "voxel". `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/` on upstream found no hits, and there is no "notable forks" section. A web search for an existing HIP/ROCm port of cuda_voxelizer turned up nothing. Conclusion: no duplicate or partial AMD effort found anywhere.

## Viability
Genuinely CUDA: `src/voxelize.cu`, `src/voxelize_solid.cu`, `src/voxelize.cuh`; CMake does `PROJECT(CudaVoxelize LANGUAGES CXX CUDA)`, `FIND_PACKAGE(CUDAToolkit REQUIRED)`, links `CUDA::cudart`; uses `cudaGetDeviceCount`, `checkCudaErrors`, `__global__`/`__device__` kernels, CUDA constant memory (Morton LUTs), and requires "Compute Capability 2.0 or higher". It falls back to a CPU path (`-cpu` flag) when no CUDA device is found, so the tool stays usable if a HIP build somehow lagged, but the point of porting is the GPU path. Not archived; actively maintained.

Build dependency: Trimesh2 (`Forceflow/trimesh2`) for mesh I/O, found via `find_package`/manual `Trimesh2_INCLUDE_DIR`/`Trimesh2_LINK_DIR` cache vars, not vendored in-tree and not itself a CUDA/GPU library -- no MOAT project dependency to record (`depends_on: []`). No other MOAT project dependencies.

## Recommendation passed to `set-intake`
```
python3 utils/moatlib.py set-intake cuda_voxelizer fork \
  --summary "Small, actively-maintained MIT CUDA mesh voxelizer with a real GPU kernel path and no existing AMD/ROCm port; two vendored NVIDIA CUDA-Samples helper headers (helper_cuda.h, helper_string.h) carry old restrictive EULA-style text and should be reviewed/replaced during the port even though the project's own code is clean MIT." \
  --duplicate "none" --viable yes
```

## Port (windows-gfx1151, 2026-11-03)

Fork: AMD-Ecosystem/cuda_voxelizer, branch `moat-port` cut from the mirror `main` (f9fc9a9). Per jeff's ruling the fork's GitHub default branch is now `moat-port` (`gh repo edit --default-branch moat-port`, confirmed with `gh repo view`); `status.json.fork_default_branch` stays `main`, the upstream mirror and PR base. Local clone remotes: `origin` = fork, `upstream` = Forceflow; protect-fork hook installed.

Commits on `moat-port` (all on top, none amended):

1. `94d1340` helper_cuda.h / helper_string.h replaced with NVIDIA/cuda-samples `v12.2` `Common/` (commit 1fa9c59db4ee8b34c4efa5b6e8fbd2a8c72a93ce), BSD-3-Clause 2022 headers. v12.2 chosen because upstream CI builds with CUDA 12.2. One local edit kept from the previous copy: `#include "helper_string.h"` (cuda-samples uses `<helper_string.h>`, which needs Common/ on the include path and would break the CUDA build). API used by the project -- `checkCudaErrors`, `findCudaDevice(int, const char**)` (which calls `checkCmdLineFlag`/`getCmdLineArgumentInt`) -- unchanged in signature and guards (`__DRIVER_TYPES_H__`, `__CUDA_RUNTIME_H__`); diff vs the old copy is the licence block, Ada/Hopper table rows, one printf newline. `utils/licenses.py scan-nvidia` on the tree: clean. Pushed alone first; deferred item `cuda-voxelizer-nvidia-proprietary-rescan` set `done`. Not verified with nvcc (no CUDA toolkit on this host) -- the CUDA no-regression compile is still owed by a host that has nvcc.
2. `8b84b91` `kernel << <g, b >> > (...)` -> `kernel<<<g, b>>>(...)` in both .cu files (clang: "expected expression").
3. `c593413` helper_math.h: arithmetic operator block (negate..divide) under `#if !defined(__HIP_PLATFORM_AMD__)` -- HIP_vector_type's member compound ops and friend vec*vec, vec/vec tie with helper_math's non-template free ones (ambiguous in util.h, cpu_voxelizer.cpp, voxelize.cu, smoothstep). Host math block guard extended to `!__CUDACC__ && !__HIPCC__` (clang HIP math header defines host+device min/max -> redefinition).
4. `527b087` `USE_HIP` option, `hip_compat/{cuda.h,cuda_runtime.h,device_launch_parameters.h}` shim on the include path (BEFORE), `.cu` as `LANGUAGE HIP`, `HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES}`, `LINKER_LANGUAGE CXX`, `hip::host`. No `static` was needed on the `__constant__` LUTs or `_DEBUG` `__device__` counters: Release and Debug both link and the morton path (cudaMemcpyToSymbol per TU) is correct.
5. `5cd065b` `cudaMemset(vtable, 0, vtable_size)` after `cudaMallocManaged` (risk 1). Garbage was not observed on gfx1151 before the fix; it is a correctness fix, not a reproduced failure.
6. `316616d` README "Build for AMD GPUs using CMake (ROCm)" subsection (Linux + Windows) + dependency line.

Risk 2/3 (bitwise atomics on managed memory, managed memory on Windows): hipMallocManaged works on gfx1151 Windows (APU) and atomicOr/atomicXor results match the CPU; no device-memory fallback was needed here. Risk 2 is still open for the wave64 gate (gfx90a discrete).

### Build recipe (Windows gfx1151, TheRock HIP 7.14, CMake 4.3.2 from the TheRock venv)

Trimesh2 2022.03.04 `libsrc` (minus GLCamera/GLManager) compiled with TheRock `clang-cl -MD -O2 -EHsc -std:c++17` into `trimesh.lib` (a `-MDd -Od` copy for the Debug build). Then (`agent_space/cuvox_build.sh`):

```
source agent_space/win_rocm_env.sh
cmake -S . -B build-hip -G Ninja -DUSE_HIP=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_C_COMPILER=$ROCM_LLVM_BIN/clang-cl.exe -DCMAKE_CXX_COMPILER=$ROCM_LLVM_BIN/clang-cl.exe -DCMAKE_HIP_COMPILER=$ROCM_LLVM_BIN/clang-cl.exe \
  -DCMAKE_PREFIX_PATH=$ROCM_ROOT -DTrimesh2_INCLUDE_DIR=<trimesh2>/include -DTrimesh2_LINK_DIR=<trimesh2>/lib.win64
cmake --build build-hip -j16
```

Gotchas: in Git Bash pass clang-cl flags as `-MD`/`-Fo` (a `/MD` is path-converted to `C:/Program Files/Git/MD`). FindOpenMP with clang-cl picks MSVC's `libomp.lib` (`-Xclang -fopenmp`); the exe then imports `libomp140.x86_64.dll`, found only under VC Redist `debug_nonredist/x64/Microsoft.VC143.OpenMP.LLVM/` -- stage it next to the exe along with `rocm_stage_runtime`. The HIP compile with clang-cl drops small `voxelize.exe`/`voxelize_solid.exe` files in the build dir (harmless side artifacts). `findCudaDevice` prints "MapSMtoCores for SM 11.5 is undefined" and names the device "Hopper" (the BSD-3 header's fallback); cosmetic, per plan no CC-keyed logic was added. clang warns `nodiscard` on the unchecked `cudaDeviceSynchronize`/`cudaOccupancyMaxPotentialBlockSize` returns (upstream code, left as-is).

### GPU-vs-CPU differential (porter sanity check, gfx1151, HEAD 316616d)

`python3 agent_space/cuvox_compare.py <exe> test_models/bunny.OBJ <workdir> [--sizes] [--formats] [--repeat]` runs GPU then `-cpu` on the same binary and compares voxel sets (binvox RLE decoded, morton .bin bits, obj_points vertex lines). Bunny: 5110 triangles.

| mode | size | filled (CPU) | diff (binvox = morton = obj_points) |
|---|---|---|---|
| surface | 64 | 13,802 | 0 |
| surface | 256 | 221,760 | 0 |
| surface | 512 | 887,315 | 9 (3 GPU-only, 6 CPU-only) |
| solid | 64 | 52,258 | 0 |
| solid | 256 | 3,345,384 | 0 |
| solid | 512 | 26,763,628 | 1 (GPU-only) |

- 3x repeated GPU runs at 512 surface + solid: identical run to run.
- `-o vox` and `-o obj` at 64: GPU and CPU files byte-identical (`cmp`).
- Debug build (`_DEBUG` device counters, cudaMemcpyFromSymbol) at 64, surface + solid, binvox + morton: 0 diff.
- Upstream CI smoke `-s 64 -cpu`: exit 0.
- Triangulation of the 512 residual (risk 8): `-DCMAKE_HIP_FLAGS=-clang:-ffp-contract=off` -> solid 0, surface 4 (2/2). Additionally replacing `rsqrtf` in helper_math `normalize()` with `1.0f / sqrtf()` (experiment build only, reverted) -> surface 0. So the residual is entirely FP: FMA contraction in the HIP TUs plus the device's approximate rsqrtf (host helper_math uses 1/sqrtf). Both apply on NVIDIA too (nvcc `--fmad=true`, approximate `rsqrtf`); not a port defect, and no flag change is shipped. Validators should expect exactly this class of tiny 512 residual; 64/256 must be 0.
