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
