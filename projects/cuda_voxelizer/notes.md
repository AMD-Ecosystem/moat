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

## Review 2026-11-04 (linux-gfx942, reviewer, moat-port 316616d vs main f9fc9a9)

Scope as ruled by jeff 2026-11-03: CMake-only (`.vcxproj` untouched), BSD-3-Clause v12.2 helpers as the first commit, fork default branch `moat-port`. Checked: `src/libs/cuda/helper_cuda.h` and `helper_string.h` are byte-identical to NVIDIA/cuda-samples `1fa9c59` `Common/` except the quoted `#include "helper_string.h"` (`helper_cuda.h:41`); `utils/jargon.py --port cuda_voxelizer` clean; fork tree clean at 316616d.

### Findings (must fix)

1. `README.md:71-74` (Windows build paragraph): it does not say the built executable needs `libomp140.x86_64.dll` at run time. With clang-cl, FindOpenMP selects MSVC's LLVM `libomp.lib`, so `cuda_voxelizer.exe` imports `libomp140.x86_64.dll`, which Visual Studio ships only under `VC\Redist\MSVC\<version>\debug_nonredist\x64\Microsoft.VC143.OpenMP.LLVM\` and puts on no search path (porter's gfx1151 notes above). A user who follows the README gets an executable that does not start. Ruling on the porter's open item: yes, the README should mention it. Add one sentence after the Windows command, for example: "The executable needs the ROCm runtime DLLs from ROCm's `bin` directory and `libomp140.x86_64.dll` (Visual Studio's LLVM OpenMP runtime, in `VC\Redist\MSVC\<version>\debug_nonredist\x64\Microsoft.VC143.OpenMP.LLVM`) on `PATH` or next to `cuda_voxelizer.exe`." This is a new commit on top (no amend).
2. CUDA no-regression build never run, while two commit Test Plans present it as run. `94d1340` (helper headers) shows `cmake -S . -B build ...` without `USE_HIP` plus a run at `-s 64`, and `8b84b91` (launch syntax) shows `cmake --build build` and a run; the porter's notes say the nvcc compile was not done (no CUDA toolkit on gfx1151). The plan's CUDA no-regression step (`CUDAARCHS=60`, upstream recipe, `USE_HIP` OFF) must be run at the head on an NVIDIA host before the upstream PR, so those Test Plans are true as written; if it cannot be run, correct the two Test Plans at PR-prep squash instead of shipping them. This cannot be done on an AMD host; it needs an nvcc host dispatched for it.

### Rulings on the porter's open items

- libomp140 in the README: yes (finding 1).
- 512 residual (gfx1151: 9 surface / 1 solid): acceptable, no code or flag change. Reproduced on gfx942 with the same class of residual and the same triangulation (below): `-ffp-contract=off` alone leaves 4 surface voxels, adding `1.0f / sqrtf` in `normalize()` leaves 0. Both factors exist in upstream's CUDA build too (nvcc `--fmad=true` by default, approximate device `rsqrtf` against the host `1.0f / sqrtf` in `helper_math.h:78`), so a GPU-vs-CPU boundary difference at fine grids is upstream behaviour, not a port defect. Validators: 64 and 256 must be 0 diff; at 512 expect single-digit boundary voxels whose count can differ by arch (gfx1151 9 surface, gfx942 7 surface), each disappearing under the triangulation.

### Reviewer evidence (linux-gfx942, MI300X, xnack-, TheRock ROCm 10.1.0 pip SDK, HIP 7.16.26385, CMake 3.31.6, g++ 13.3 host)

Not validation evidence; the validator still owns the gate. Trimesh2 2022.03.04 (`0b3d9e2`) `libsrc` minus GLCamera/GLManager compiled with `g++ -O2 -fPIC -fopenmp` into `libtrimesh.a`. The README Linux recipe with the default g++ host compiler builds clean (only upstream warnings):

```
cmake -S . -B build-hip -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_PREFIX_PATH=$(rocm-sdk path --root) -DCMAKE_BUILD_TYPE=Release -DTrimesh2_INCLUDE_DIR=<trimesh2>/include -DTrimesh2_LINK_DIR=<tm2build>
cmake --build build-hip -j28
```

Without `-DCMAKE_HIP_ARCHITECTURES`, CMake detected gfx942 (README and `527b087` claim confirmed). GPU-vs-CPU differential on `test_models/bunny.OBJ` (same binary, GPU then `-cpu`), all GPU logs show `## GPU VOXELISATION` and `Voxelization GPU time`:

| mode | size | filled (CPU) | diff, binvox = morton = obj_points |
|---|---|---|---|
| surface | 64 | 13,802 | 0 |
| surface | 256 | 221,760 | 0 |
| surface | 512 | 887,315 | 7 (3 GPU-only, 4 CPU-only) |
| solid | 64 | 52,258 | 0 |
| solid | 256 | 3,345,384 | 0 |
| solid | 512 | 26,763,628 | 1 (GPU-only) |

- 512 triangulation: `-DCMAKE_HIP_FLAGS=-ffp-contract=off` -> surface 4 (2/2), solid 0; plus `rsqrtf` -> `1.0f / sqrtf` in `normalize()` (copy of the tree in agent_space, not the fork) -> surface 0, solid 0.
- 3 repeated GPU runs at 512, surface + solid, binvox + morton: identical.
- Plan risk 2 (bitwise atomics on managed memory, discrete wave64) does not fire on gfx942: the differential above is exact at 64/256, and a standalone probe (`hipMallocManaged` 8 MiB, `hipMemset`, then atomicOr/atomicXor setting every bit) left 0 of 2,097,152 words wrong; the device reports `hipDeviceAttributeManagedMemory=1`, `ConcurrentManagedAccess=1`, `PageableMemoryAccess=0`, pointer type managed. gfx90a (no MI300-class coherence) is still unmeasured.
- Timings: surface 512 GPU 109 ms vs CPU 2.45 s; solid 512 GPU 4.8 s vs CPU 137 s.
- A Linux `CMAKE_BUILD_TYPE=Debug` does not define `_DEBUG` (only MSVC does), so the `_DEBUG` device counters and `cudaMemcpyFromSymbol` are covered only by the Windows Debug run.

## Porter round 2026-11-04 (linux-gfx950, answering the 2026-11-04 review)

### Finding 1: README libomp140 (fixed in `689b885`)

New commit on top of `316616d` (no amend): one paragraph after the Windows build text saying the executable needs the ROCm `bin` DLLs and `libomp140.x86_64.dll` (VS LLVM OpenMP runtime, `VC\Redist\MSVC\<version>\debug_nonredist\x64\Microsoft.VC143.OpenMP.LLVM`) on `PATH` or next to `cuda_voxelizer.exe`. `utils/jargon.py --port cuda_voxelizer` clean, `utils/prose.py` on the commit message clean.

### Finding 2: CUDA no-regression build (run; passes)

Host: Linux x86_64, CUDA 12.8 from apt (`nvcc` V12.8.93 at `/usr/local/cuda/bin`, not on the default `PATH`), GCC 13.3.0, CMake 3.31.6, Ninja. No NVIDIA GPU. Trimesh2 `2022.03.04` (`0b3d9e2`) built with its own Makefile (`make -C libsrc`) into `lib.Linux64/libtrimesh.a`, as upstream CI does. The upstream CI recipe, `USE_HIP` left at its default OFF:

```
export PATH=/usr/local/cuda/bin:$PATH CUDAARCHS=60
cmake -GNinja -DTrimesh2_INCLUDE_DIR=<trimesh2>/include -DTrimesh2_LINK_DIR=<trimesh2>/lib.Linux64 -S . -B build-cuda
cmake --build build-cuda --parallel 64
./build-cuda/cuda_voxelizer -f ./test_models/bunny.OBJ -s 64 -cpu
./build-cuda/cuda_voxelizer -f ./test_models/bunny.OBJ -s 64
```

- Configure: `The CUDA compiler identification is NVIDIA 12.8.93 with host compiler GNU 13.3.0`, `Found CUDAToolkit ... 12.8.93`, OpenMP 4.5.
- Build at `689b885` (head): exit 0, `[8/8] Linking CXX executable cuda_voxelizer`; both `.cu` files compiled by nvcc for sm_60. Warnings only: upstream `-Wformat` (`%llu` for `size_t`) in `util_cuda.cpp:34`, `util_io.cpp:21`, `main.cpp:67`, `-Wwrite-strings` in `main.cpp:28` (lines the port does not touch), and nvcc's "architectures prior to sm_75 deprecated" note for `CUDAARCHS=60`. No warning from the replaced helper headers or the HIP-only code.
- The same recipe also builds and links at `94d1340` (helper headers alone) and `8b84b91` (helpers + launch syntax), the two commits whose Test Plans show a CUDA build, each in a detached worktree; `-s 64 -cpu` exit 0 at both.
- `-s 64 -cpu` (the upstream CI smoke): exit 0, writes `bunny.OBJ_64.vox`. `-s 64` without `-cpu`: exit 0, `[CUDA] First call to CUDA Runtime API failed`, `CUDA GPU not found`, falls back to CPU voxelization -- the expected behaviour with no NVIDIA driver.

What this does not cover: the Test Plans of `94d1340` and `8b84b91` show `./build/cuda_voxelizer -f ./test_models/bunny.OBJ -s 64` as a GPU run; with no NVIDIA GPU on this host that ran the CPU fallback, so the CUDA kernels compiled and linked but did not execute. The build half of those Test Plans is now true; the NVIDIA GPU execution is still unobserved.

### Local HIP build (linux-gfx950, porter sanity, not validation)

TheRock pip SDK (`rocm-sdk path --root`), HIP Clang 24.0.0, same Trimesh2 (g++ build): `cmake -S . -B build-hip -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx950 -DCMAKE_PREFIX_PATH=$(rocm-sdk path --root) -DCMAKE_BUILD_TYPE=Release -DTrimesh2_INCLUDE_DIR=<trimesh2>/include -DTrimesh2_LINK_DIR=<trimesh2>/lib.Linux64`, `cmake --build build-hip --parallel 64`: exit 0. `-s 64 -o binvox` takes the GPU path (`CUDA device(s) found`, "MapSMtoCores for SM 9.5 is undefined", the known cosmetic fallback). The validator owns the differential.

## Review 2026-11-04 (re-review, linux-gfx942, reviewer, moat-port 689b885; delta 316616d..689b885)

Verdict: review-passed. Both findings of the first 2026-11-04 review are resolved; no new findings.

- Finding 1 (README libomp140): resolved by `689b885`, `README.md:77`. The paragraph sits right after the Windows command and the target sentence, names `libomp140.x86_64.dll`, the `debug_nonredist\x64\Microsoft.VC143.OpenMP.LLVM` location, the ROCm `bin` DLLs, and `PATH` or next to the exe. Documentation only; nothing else in the tree changed (`git diff 316616d..689b885` is the one README hunk). Commit: `[ROCm]` title within 72 chars, AI-assistance disclosure, no `Co-Authored-By`/noreply; Test Plan shows the nvcc build and the `-cpu` smoke only, which matches what was run. `utils/jargon.py --port cuda_voxelizer` clean, `utils/prose.py` on the message clean, fork tree clean at 689b885.
- Finding 2 (CUDA no-regression build): resolved for review. The porter's nvcc 12.8 / GCC 13.3 build+link with the upstream CI recipe (`USE_HIP` OFF, `CUDAARCHS=60`) at 689b885, 94d1340 and 8b84b91 makes the build half of the 94d1340 and 8b84b91 Test Plans true. Not independently reproduced here: building Trimesh2 on this host was refused by the session's permission policy, so this relies on the porter's recorded log. Static cross-check of what the CUDA build sees from the whole series (`git diff main..HEAD`): `CMakeLists.txt` keeps `ENABLE_LANGUAGE(CUDA)`, `FIND_PACKAGE(CUDAToolkit)` and `CUDA::cudart` in the `ELSE()` branch; `hip_compat/` is only on the include path under `USE_HIP`; `helper_math.h` changes are guarded by `__HIP_PLATFORM_AMD__` / `__HIPCC__` and are no-ops under nvcc; the remaining CUDA-visible changes are the BSD-3 helper headers, the `<<<...>>>` spacing in the two `.cu` files, and the `cudaMemset` after `cudaMallocManaged` (`src/main.cpp`). None is a plausible CUDA runtime regression.
- NVIDIA GPU run (no NVIDIA GPU yet; the primary is arranging a V100 run): does not block review-passed. The reviewer role does not request changes for GPU runs that have not happened, the CUDA code that executes on an NVIDIA GPU differs from upstream only as listed above, and the open claim is in commit Test Plans, which matter at the upstream PR. It does gate the upstream PR: registered as `cuda-voxelizer-nvidia-gpu-run-before-pr` in `deferred.json` (run `-s 64` on the GPU path at head and compare with `-cpu`, or else correct the 94d1340 / 8b84b91 Test Plans at the PR-prep squash). Whether it waits until PR prep or is done now is a person's ruling (`utils/deferred.py decide`).

## CUDA run on NVIDIA V100 (2026-11-04, primary session)

Closes the "kernels never executed on NVIDIA" gap left by the mi355 nvcc build. Node: Tesla V100-SXM2-32GB, driver 575.57.08, CUDA 12.9 (`/usr/local/cuda`), sm_70. Trimesh2 2022.03.04 built from `libsrc` without the GL objects (no GL headers on the node; `make -k -C libsrc` then `ar rcs lib.Linux64/libtrimesh.a libsrc/OBJ.Linux64/*.o`), cmake from pip. For each of 689b885 (head), 94d1340 and 8b84b91:

```
cmake -S . -B b -DTrimesh2_INCLUDE_DIR=$W/trimesh2/include -DTrimesh2_LINK_DIR=$W/trimesh2/lib.Linux64 -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build b -j8
./b/cuda_voxelizer -f ./test_models/bunny.OBJ -s 64
./b/cuda_voxelizer -f ./test_models/bunny.OBJ -s 64 -solid
./b/cuda_voxelizer -f ./test_models/bunny.OBJ -s 64 -cpu
```

All three configure, build, link and run to completion (exit 0). The GPU runs print `[CUDA] CUDA device(s) found, picking best one` and the same bounding box `(-2.966815,0.033993,-2.477184)-(1.915775,4.916584,2.405406)` at grid 64; total runtime 393-441 ms (GPU) and 100-117 ms (`-solid`). The Test Plans of 94d1340 and 8b84b91 are now true in full (build and run on an NVIDIA GPU). USE_HIP stayed at its default OFF.
