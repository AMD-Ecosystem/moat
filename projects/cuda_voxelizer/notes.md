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

## Validation 2026-11-04 (linux-gfx1201, validator, moat-port 689b885)

Verdict: completed. Satisfies the wave32 gate at head 689b885.

Host: AMD Radeon AI PRO R9700 (gfx1201, RDNA4, wave32, discrete PCIe, xnack off), one GPU pinned with `HIP_VISIBLE_DEVICES`. TheRock ROCm 10.1.0 pip SDK (`rocm-sdk path --root`), AMD clang 24.0.0git, amdgpu 6.19.14, CMake 3.31.6, g++ 13.3.0 host. All GPU runs under `ulimit -c 0`. Fork clone tree clean at 689b885 (`git status --porcelain` empty); `utils/jargon.py --port cuda_voxelizer` clean. README documents the ROCm build for Linux and Windows next to the CUDA build. CUDA no-regression gate not rerun: recorded at this head already (nvcc 12.8 build above, V100 run above).

### Build

Trimesh2 `2022.03.04` (`0b3d9e2`): `make -C trimesh2/libsrc -j24` -> `lib.Linux64/libtrimesh.a`, 0 errors. The README Linux recipe, verbatim apart from the ROCm prefix (no `/opt/rocm` on this host):

```
mkdir build && cd build
cmake -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_PREFIX_PATH=$(rocm-sdk path --root) -DTrimesh2_INCLUDE_DIR:PATH=<trimesh2>/include -DTrimesh2_LINK_DIR:PATH=<trimesh2>/lib.Linux64 <src>
cmake --build . --parallel 24
```

Exit 0; warnings only from upstream (`main.cpp:28` -Wwrite-strings, `%llu` -Wformat in `main.cpp:67`/`util_cuda.cpp:34`) and Trimesh2's `mathutil.h`. Also built: `-G Ninja -DCMAKE_BUILD_TYPE=Release`, `-DCMAKE_BUILD_TYPE=Debug`, and Debug with `-DCMAKE_CXX_FLAGS=-D_DEBUG -DCMAKE_HIP_FLAGS=-D_DEBUG` (Linux Debug does not define `_DEBUG`, so this is the only way to cover the device counters on Linux). All exit 0.

### GPU-vs-CPU differential

Harness: `agent_space/scripts/cuvox_diff.py <exe> test_models/bunny.OBJ <work> --sizes .. --repeat 3`. Each run is in its own directory with a copy of the model. The GPU log must show `## GPU VOXELISATION` and no CPU fallback. binvox RLE, the morton `.bin` bitset and obj_points are decoded to voxel sets and compared, giving GPU-only and CPU-only counts. Bunny has 5110 triangles. The GPU log names the device `AMD Radeon AI PRO R9700` and prints the known cosmetic "MapSMtoCores for SM 12.0 is undefined".

Release build (`-DCMAKE_BUILD_TYPE=Release`), 18 cases, diff identical across binvox / morton / obj_points:

| mode | size | filled (CPU) | filled (GPU) | diff (GPU-only / CPU-only) |
|---|---|---|---|---|
| surface | 64 | 13,802 | 13,802 | 0 |
| surface | 256 | 221,760 | 221,760 | 0 |
| surface | 512 | 887,315 | 887,312 | 9 (3 / 6) |
| solid | 64 | 52,258 | 52,258 | 0 |
| solid | 256 | 3,345,384 | 3,345,384 | 0 |
| solid | 512 | 26,763,628 | 26,763,629 | 1 (1 / 0) |

README-recipe build (no `CMAKE_BUILD_TYPE`), the same 18 cases: 64 and 256 are 0 everywhere; 512 surface 4 (2 / 2), 512 solid 0, for all three formats.

- 512 residual is the accepted FP class, same counts as gfx1151 Release (9 surface, 1 solid). Triangulation on Release, binvox: `-DCMAKE_HIP_FLAGS=-ffp-contract=off` -> surface 4 (2 / 2), solid 0; plus `rsqrtf` -> `1.0f / sqrtf` in helper_math `normalize()` (scratch copy, not the fork) -> surface 0, solid 0.
- Repeat stability: 3 extra GPU runs per 512 case (surface + solid x binvox / morton / obj_points, both builds = 36 repeats): 0 voxels differ from the first GPU run in every one.
- Debug build at 64, surface + solid, binvox + morton: 0 diff. `-D_DEBUG` build at 64: 0 diff, and the surface counters read back through `cudaMemcpyFromSymbol` (`Processed 5110 triangles`, `Tested 117367 voxels`, `Marked 45229 voxels`); solid prints 0 for all three because upstream's solid kernel never increments its counters. So the duplicate `__device__` names in two TUs link and resolve per TU.
- `-o obj` at 64: GPU and CPU files byte-identical (`cmp`). `-o vox` at 64: GPU and CPU files differ only in the last 4 bytes, and so do two CPU runs of the same binary. Cause is upstream: `VoxWriter.cpp` fills the palette with `for (i = 0; i < 255; i++)`, leaving `colors[255]` uninitialized stack memory, which is then written as the final RGBA entry. The first 56,469 of 56,473 bytes are identical across 3 GPU and 3 CPU runs. Not a port defect; the gfx1151 byte-identity was luck.
- Upstream CI smoke `cuda_voxelizer -f bunny.OBJ -s 64 -cpu`: exit 0, writes `bunny.OBJ_64.vox`, on both the README and Release builds.
- Timings (Release, 512, GPU vs `-cpu`, wall including I/O): surface binvox 0.46 s vs 0.60 s; solid binvox 0.47 s vs 14.4 s.

### Plan risk 2 (bitwise atomics on managed memory, discrete card)

Does not fire on this discrete PCIe gfx1201. The differential is exact at 64/256 and the 512 residual is fully explained by FP. Standalone probe `agent_space/scripts/managed_atomics.hip` (`hipcc --offload-arch=gfx1201 -O2`): `hipMallocManaged` 8 MiB, `hipMemset`, then one thread per bit doing `atomicOr` (and `atomicXor` x3 on a second buffer), then again after a host `memset` of the managed buffer. Results: atomicOr 0 of 2,097,152 words wrong, atomicXor 0 of 2,097,152, atomicOr after host touch 0 of 2,097,152. Device reports `ManagedMemory=1`, `ConcurrentManagedAccess=1`, `PageableMemoryAccess=0`, `Integrated=0`; pointer type managed. gfx90a (wave64 discrete without MI300-class coherence) is still unmeasured.
## Validation 2026-11-04 (linux-gfx90a, validator, moat-port 689b885) -- validation-failed

Verdict: **validation-failed** on plan risk 2. Every default build passes the differential, but only because of a compiler accident: in a build whose device code keeps the atomic's metadata (device `-O0`), the GPU output on this card is empty with no error. The plan's remedy for risk 2 applies: "Fix if it fires: allocate the voxel table with `cudaMalloc` + `cudaMemset` and copy it back with `cudaMemcpy` before file output". Back to the porter.

Host: AMD Instinct MI250X/MI250 (gfx90a:sramecc+:xnack-), one GCD (`HIP_VISIBLE_DEVICES=0`), Linux. TheRock ROCm 10.1.0 pip SDK (`rocm-sdk path --root`), HIP 7.16.26385, AMD clang 24.0.0git (d6f6cb6), CMake 3.31.6, Ninja, g++ 13.3. Trimesh2 2022.03.04 (`0b3d9e2`), `make -C libsrc` (full libsrc builds on this host), `lib.Linux64/libtrimesh.a`. All GPU runs under `ulimit -c 0`.

### Build (README Linux recipe)

```
cd src && mkdir build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=$(rocm-sdk path --root) -DTrimesh2_INCLUDE_DIR:PATH=<trimesh2>/include -DTrimesh2_LINK_DIR:PATH=<trimesh2>/lib.Linux64 ..
cmake --build . --parallel 64
```

Exit 0, `[8/8] Linking CXX executable cuda_voxelizer`, upstream warnings only. Device prints `AMD Instinct MI250X / MI250` (plus the known cosmetic "Hopper"/"compute capability 9.0" name from helper_cuda.h).

### GPU-vs-CPU differential (default Release build), `test_models/bunny.OBJ`

Harness: `agent_space/cuvox_compare.py <exe> <model> <workdir> [--sizes] [--formats] [--repeat N]` (GPU run then `-cpu` on the same binary; the GPU log must show `## GPU VOXELISATION` and `Voxelization GPU time` and no "Falling back"; binvox decoded from RLE, morton `.bin` as a bit set, obj_points as vertex lines). The binvox, morton and obj_points counts agree in every case.

| mode | size | filled CPU | filled GPU | differing (GPU-only / CPU-only) | GPU time (binvox) |
|---|---|---|---|---|---|
| surface | 64 | 13,802 | 13,802 | 0 | 2.5 ms |
| surface | 256 | 221,760 | 221,760 | 0 | 23.4 ms |
| surface | 512 | 887,315 | 887,313 | 8 (3 / 5) | 107.8 ms |
| solid | 64 | 52,258 | 52,258 | 0 | 13.8 ms |
| solid | 256 | 3,345,384 | 3,345,384 | 0 | 333 ms |
| solid | 512 | 26,763,628 | 26,763,629 | 1 (1 / 0) | 9.3 s |

- 512 residual is the accepted FP class, reproduced: `-DCMAKE_HIP_FLAGS=-ffp-contract=off` gives surface 4 (2/2), solid 0; also replacing `rsqrtf` in `normalize()` with `1.0f / sqrtf` (copy of the tree in agent_space, not the fork) gives surface 0, solid 0.
- Repeat stability: GPU runs at 512, surface and solid, binvox, morton and obj_points, 3 runs each: output byte-identical (sha256) run to run.
- `-o vox` and `-o obj` at 64: obj byte-identical GPU vs CPU. vox differs only in its last 4 bytes, and so do two `-cpu` runs of the same binary. This is pre-existing upstream behaviour in untouched vendored code: `VoxWriter.cpp:543` fills the palette with `i < 255`, so `colors[255]`, the file's final int32, is uninitialized stack. With those 4 bytes excluded, all 6 files (3 GPU, 3 CPU) are identical. Not a port issue.
- Debug: `-DCMAKE_BUILD_TYPE=Debug` links and runs at 64 (surface and solid, binvox and morton): 0 diff. Linux Debug does not define `_DEBUG`, so a second Debug build added `-DCMAKE_CXX_FLAGS=-D_DEBUG -DCMAKE_HIP_FLAGS=-D_DEBUG`: links (no duplicate-symbol error from the per-TU `__device__` counters), 0 diff, and `cudaMemcpyFromSymbol` reads the counters (`[Debug] Processed 5110 triangles on the GPU`, `Tested 117367 voxels`, `Marked 45229 voxels`).
- Upstream CI smoke `./build/cuda_voxelizer -f ./test_models/bunny.OBJ -s 64 -cpu`: exit 0, CPU path, writes the `.vox`. CPU counts match the porter's gfx1151 and the reviewer's gfx942 CPU counts exactly. There is no nvcc here, so an unmodified upstream binary could not be built for a CPU baseline.
- CUDA gate: not re-run here. It is already recorded at this head (porter nvcc 12.8 build at 689b885; V100 run at 689b885).
- `utils/jargon.py --port cuda_voxelizer`: clean. ROCm build is in the README. Fork tree clean at 689b885 (`git status --porcelain` empty; `build/` untracked).

### Risk 2: managed-memory atomicOr/atomicXor fire on gfx90a

Standalone probe (`agent_space/probe/p4.hip`, `hipcc --offload-arch=gfx90a -O2`): `hipMallocManaged` buffer, `hipMemset` 0, one thread per bit doing `atomicOr`/`atomicXor(&t[w], 1u << b)`. Device attributes: ManagedMemory=1, ConcurrentManagedAccess=1, PageableMemoryAccess=0.

| buffer | op | bits set / expected |
|---|---|---|
| hipMalloc | atomicOr / atomicXor (contended and spread) | 2,097,152 / 2,097,152 |
| hipMallocManaged | atomicOr / atomicXor (contended and spread) | **0 / 2,097,152**, no error from launch or sync |
| hipMallocManaged | `atomicOr_system` / `atomicXor_system` | 2,097,152 / 2,097,152 |

The same failure appears at 64, 4096, 65,536 and 2M words. `atomicAdd`, plain stores and `atomicOr(0xffffffff)` on managed memory work. So the native `global_atomic_or`/`global_atomic_xor` into managed memory is dropped silently: host-resident fine-grained memory over PCIe, where AtomicOps cover only add, swap and CAS.

Why the shipped binary still passes: its code objects contain no native OR/XOR, only one `global_atomic_cmpswap` retry loop per kernel (`llvm-objdump --offloading build/cuda_voxelizer`, `llvm-objdump -d` on the gfx90a bundles). At `-O0` the device IR of `voxelize.cu` and `voxelize_solid.cu` carries `atomicrmw or/xor ... !amdgpu.no.fine.grained.memory`, the same as the probe. `-mllvm -opt-bisect-limit` shows the metadata is dropped by pass 3893, `inline on _Z17voxelize_triangle7voxinfoPfPjb` (inlining `setBit`), after which the backend expands the RMW to CAS, which PCIe supports. Device codegen by optimization level, for both `.cu` files: `-O0` gives `flat_atomic_or`/`flat_atomic_xor`; `-O1`, `-O2`, `-O3` and `-Os` give `global_atomic_cmpswap`. The CMake Release, Debug and `-ffp-contract=off` builds all contain only the CAS.

End to end with device `-O0` (`-DCMAKE_BUILD_TYPE=Debug "-DCMAKE_HIP_FLAGS_DEBUG=-g -O0"`, otherwise the README recipe), size 64, binvox:

| mode | filled CPU | filled GPU | differing |
|---|---|---|---|
| surface | 13,802 | **0** | 13,802 |
| solid | 52,258 | **0** | 52,258 |

So correctness on discrete MI200 depends on an inliner side effect that a compiler change (or a user's `-O0` debug build) removes, with an empty grid and no error as the symptom. gfx942 (reviewer probe) and the gfx1151 APU (porter) do not drop the native atomics, which is why they passed.

Fix options for the porter (each also correct on CUDA):

1. The plan's remedy: allocate `vtable` with `cudaMalloc` + `cudaMemset`, and copy it back into a host buffer with `cudaMemcpy` after the kernel, before the writers. hipMalloc + native OR/XOR measured exact above.
2. Smaller: `atomicOr_system`/`atomicXor_system` in `setBit`/`setBitXor` on the HIP build. Measured exact on managed memory above, but it changes the two kernel sources and needs a HIP guard, because CUDA's `_system` atomics need sm_60.

Option 1 matches the plan and keeps the kernels untouched. After the fix, a validator should repeat this matrix and the device `-O0` build on gfx90a.

The fault class is promoted to the `cuda-to-rocm` skill (`references/fault-classes.md` Memory and lifetime, plus the SKILL.md index line) on this branch.

## Porter round 2026-11-04 (linux-gfx90a, answering the gfx90a validation-failed at 689b885)

Fix: `3625a1e` "[ROCm] Keep the voxel table in device memory", the plan's risk-2 remedy (validator's option 1). `src/main.cpp`: the GPU path allocates `device_vtable` with `cudaMalloc` + `cudaMemset`, passes it to `voxelize`/`voxelize_solid`, then `malloc`s the host `vtable`, `cudaMemcpy` DeviceToHost, and `cudaFree`s the device table before the writers. The log line now says "CUDA device memory". `hip_compat/cuda_runtime.h`: added `cudaMalloc`, `cudaFree`, `cudaMemcpy` mappings. Triangles stay in managed memory (kernels only read them; no atomics). No `#ifdef`: the change is portable and smaller than the `_system` alternative, which would edit both kernel files and need a HIP guard (CUDA `_system` atomics need sm_60). Kernels untouched. The file keeps its CRLF line endings.

**CUDA-visible change**: `src/main.cpp` is compiled by the CUDA build. No nvcc on this host, so the CUDA build was not compiled here. The primary should re-run the V100 check (build + `-s 64`, `-s 64 -solid`, `-s 64 -cpu`) at `3625a1e`.

Host: AMD Instinct MI250X/MI250 (gfx90a:sramecc+:xnack-), one GCD (`HIP_VISIBLE_DEVICES=0`). TheRock ROCm 10.1.0 pip SDK, CMake 3.31.6 + Ninja, g++ 13.3. Trimesh2 2022.03.04 `make -C libsrc`. All GPU runs under `ulimit -c 0`.

Builds (both exit 0, upstream warnings only):

```
cmake -S <src> -B b-rel  -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=$(rocm-sdk path --root) -DTrimesh2_INCLUDE_DIR:PATH=<tm2>/include -DTrimesh2_LINK_DIR:PATH=<tm2>/lib.Linux64
cmake -S <src> -B b-dbg0 -G Ninja -DCMAKE_BUILD_TYPE=Debug "-DCMAKE_HIP_FLAGS_DEBUG=-g -O0" (rest as above)
cmake --build <dir> --parallel 64
```

Emitted atomics (`llvm-objdump --offloading`, then `llvm-objdump -d` on each gfx90a bundle): Release has one `global_atomic_cmpswap` per kernel (as before); the -O0 Debug build has `flat_atomic_or` (voxelize.cu) and `flat_atomic_xor` (voxelize_solid.cu), the native instructions that set no bits on managed memory at 689b885. They now target `hipMalloc` memory, and the -O0 table below shows they set the bits.

GPU-vs-CPU differential, `test_models/bunny.OBJ`, validator's harness (`cuvox_compare.py <exe> <model> <work> --repeat 3`: GPU then `-cpu`, GPU log must show `## GPU VOXELISATION` and the GPU time, no fallback; binvox/morton/obj_points decoded to voxel sets). Counts identical across the three formats in every case; 3 repeats per case byte-identical to the first GPU run (36 cases + 108 repeats total):

| mode | size | filled CPU | Release GPU diff (GPU-only/CPU-only) | Release GPU ms (binvox) | -O0 GPU diff | -O0 GPU ms |
|---|---|---|---|---|---|---|
| surface | 64 | 13,802 | 0 | 0.4 | 0 | 24.7 |
| surface | 256 | 221,760 | 0 | 9.9 | 0 | 577 |
| surface | 512 | 887,315 | 8 (3/5) | 61.9 | 4 (2/2) | 3,878 |
| solid | 64 | 52,258 | 0 | 1.0 | 0 | 18.1 |
| solid | 256 | 3,345,384 | 0 | 50.6 | 0 | 698 |
| solid | 512 | 26,763,628 | 1 (1/0) | 395 | 0 | 5,497 |

- At 689b885 the -O0 build gave 0 filled voxels (surface and solid, size 64). Now it is exact at 64/256.
- 512 residual is the accepted FP class. Release matches the validator's 689b885 numbers exactly (8 / 1). -O0 matches the validator's `-ffp-contract=off` triangulation (4 / 0), as expected because -O0 does no FMA contraction.
- Solid 512 GPU kernel time fell from 9.3 s (managed table, CAS loop over PCIe) to 0.4 s with the table in device memory.
- Upstream CI smoke `-s 64 -cpu` (Release build): exit 0, writes `bunny.OBJ_64.vox`.
- `utils/jargon.py --port cuda_voxelizer` clean; `utils/prose.py` on the commit message clean; fork tree clean at 3625a1e.
- Not rerun: the `-D_DEBUG` device-counter build (counters and `cudaMemcpyFromSymbol` are untouched by this change), Windows, and the other arches. 3625a1e advances head, so every arch revalidates.

## CUDA run on NVIDIA V100 at 3625a1e (2026-11-04, primary session)

Same node, driver 575.57.08, CUDA 12.9, sm_70, and the same recipe as the earlier V100 section. At 3625a1e ("[ROCm] Keep the voxel table in device memory"; this changes main.cpp, which the CUDA build compiles), configure, build and link pass with USE_HIP OFF. `-s 64`, `-s 64 -solid` and `-s 64 -cpu` all exit 0, and the GPU run prints `[CUDA] CUDA device(s) found` with the same bounding box as before. The GPU `-s 64 -o binvox` output is byte-identical to the `-cpu` output (`cmp` clean). The CUDA no-regression gate holds at 3625a1e.

## Review 2026-11-04 (third review, linux-gfx942, reviewer, moat-port 3625a1e; delta 689b885..3625a1e)

Verdict: review-passed. The delta implements the plan's risk-2 remedy (validator's option 1) correctly; no code findings. Two upstream-visible wording corrections in the `3625a1e` commit body go to the PR-prep squash (registered as `cuda-voxelizer-3625a1e-body-wording`), because commits on the branch are never amended.

### Code check (`git diff 689b885..3625a1e`: `src/main.cpp` +12/-6, `hip_compat/cuda_runtime.h` +3)

- `src/main.cpp:212-229`: `device_vtable` is `cudaMalloc`ed and `cudaMemset` to 0, passed to `voxelize`/`voxelize_solid`, then a host `vtable` is `malloc`ed, filled by `cudaMemcpy` DeviceToHost and the device table is `cudaFree`d before the writers (`main.cpp:252-264`). No other host-side reader of the table exists: `voxelize.cu:192-218` and `voxelize_solid.cu:147-173` only launch and `cudaDeviceSynchronize`, and the writers in `util_io.cpp` take the host pointer. `cudaMemcpy` on the legacy stream is ordered after the kernel anyway, and it is now the first checked call after the kernels, so a sticky kernel fault surfaces through `checkCudaErrors` where before it did not. The unchecked `malloc` matches the CPU path's unchecked `calloc` (`main.cpp:236`). The device table is not leaked; the host table was never freed upstream either (process exit). CRLF line endings preserved on every changed line. No `#ifdef`; the kernels are unchanged.
- `hip_compat/cuda_runtime.h:40-44`: `cudaMalloc`, `cudaFree`, `cudaMemcpy` -> HIP names, alongside the existing `cudaMemset`/`cudaMemcpyDeviceToHost`. Nothing else in the tree needed a new mapping (all builds below link).
- Fault classes re-scanned for the delta: no warp-size, texture, OOB or FP surface touched. The memory-and-lifetime entries apply and are handled: fresh device allocation zeroed explicitly (`cudaMemset`), bitwise atomics now target device memory, the copy is synchronous from device to a live host buffer.
- Commit `3625a1e`: `[ROCm]` title 44 chars, AI-assistance disclosure, no `Co-Authored-By`/noreply, Test Plan has fenced literal commands. `utils/jargon.py --port cuda_voxelizer` clean (after creating the local `main` ref the checker needs in a fresh clone), `utils/prose.py` on the body clean, fork tree clean at 3625a1e.

### Wording corrections for the PR-prep squash (not blocking review)

1. The Test Plan's last paragraph says the 512 boundary residual is "from floating-point contraction and rsqrtf, as in the CUDA build". The cause is measured on four AMD arches, but no CUDA run has ever gone past `-s 64` (both V100 sections above), so "as in the CUDA build" is an inference stated as an observation. Either run the 512 matrix on the V100 or reword (for example, "nvcc also contracts FMAs and uses an approximate device rsqrtf by default").
2. The body says "on CUDA it replaces page migration of the table with one explicit copy". It omits the one CUDA behaviour change a maintainer would weigh: on Linux with Pascal or newer, a `cudaMallocManaged` table larger than free device memory oversubscribes and still runs; a `cudaMalloc` table that size now fails at `cudaMalloc` with an explicit error. Example: `-s 4096` needs an 8 GiB table (4096^3 / 8 bytes), more than the README's GTX 1050 Ti benchmark card has. Upstream does no memory-limit check (`util_cuda.cpp:33` only prints `cudaMemGetInfo`; README Todo lists "Memory limits test"), so the same grids already fail on Windows CUDA, where managed memory cannot oversubscribe. A sentence in the body is enough; whether the trade-off is acceptable or the porter should switch to validator option 2 (HIP-guarded `_system` atomics, table stays managed) is a person's call at PR prep. I do not request option 2: it edits both kernel files and still relies on a CUDA sm_60 requirement, and the extreme-grid case is untested by upstream either way.

### Fact-check of the porter's gfx90a evidence

Codegen reproduced from this host (no gfx90a GPU here; compile only, same TheRock 10.1.0 SDK). `llvm-objdump --offloading` then `llvm-objdump -d` on each bundle of the 3625a1e binary:

| build | voxelize.cu bundle | voxelize_solid.cu bundle |
|---|---|---|
| gfx90a Release | `global_atomic_cmpswap` | `global_atomic_cmpswap` |
| gfx90a Debug, `CMAKE_HIP_FLAGS_DEBUG="-g -O0"` | `flat_atomic_or` | `flat_atomic_xor` |
| gfx942 Release | `global_atomic_or` | `global_atomic_xor` |
| gfx942 Debug -O0 | `flat_atomic_or` | `flat_atomic_xor` |

This matches the porter's and validator's gfx90a claims exactly (optimized gfx90a builds lower the RMW to a CAS loop; -O0 keeps the native OR/XOR). On gfx942, Release already emits the native OR/XOR, so this card was always exercising native bitwise atomics on managed memory, and the first review's "does not fire on gfx942" agrees with that. That those native instructions set no bits on gfx90a managed memory, and that 3625a1e makes the -O0 build exact there, can only be seen on the MI250X; it is recorded consistently by two independent sessions (validator probe table and -O0 end-to-end at 689b885, porter -O0 matrix at 3625a1e) and the gfx90a validator owns it. The porter's Release 512 counts on gfx90a (8 / 1) equal the validator's at 689b885 and the -O0 counts (4 / 0) equal the `-ffp-contract=off` triangulation, which is the expected pattern.

### Fact-check of the primary's V100 run at 3625a1e

Not reproducible here (no NVIDIA GPU). Consistent with everything else: the CUDA-visible delta is exactly the `main.cpp` hunk above (the `hip_compat/` header is only on the include path under `USE_HIP`), and GPU `-s 64 -o binvox` byte-identical to `-cpu` is what every AMD arch also shows at 64. The V100 sections cover `-s 64` only, which is why wording item 1 exists.

### Reviewer evidence (linux-gfx942, AMD Instinct MI300X HF, one GPU pinned by `HIP_VISIBLE_DEVICES`, TheRock ROCm 10.1.0 pip SDK, HIP 7.16.26385, AMD clang 24.0.0git d6f6cb6, CMake 3.31.6, Ninja, g++ 13.3.0; all GPU runs under `ulimit -c 0`)

Not validation evidence; the gfx942 validator still owns the gate. Trimesh2 `0b3d9e2` (2022.03.04), `make -C libsrc` -> `lib.Linux64/libtrimesh.a`. Builds of fork 3625a1e, all exit 0, upstream warnings only (`main.cpp:28` -Wwrite-strings, `main.cpp:67` -Wformat, `nodiscard` on unchecked returns in the `.cu` files, Trimesh2 `mathutil.h`); no warning on the new lines:

```
common="-G Ninja -DUSE_HIP=ON -DCMAKE_PREFIX_PATH=$(rocm-sdk path --root) -DTrimesh2_INCLUDE_DIR:PATH=<tm2>/include -DTrimesh2_LINK_DIR:PATH=<tm2>/lib.Linux64"
cmake -S <src> -B rel942    $common -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx942
cmake -S <src> -B dbg942    $common -DCMAKE_BUILD_TYPE=Debug   -DCMAKE_HIP_ARCHITECTURES=gfx942
cmake -S <src> -B o0942     $common -DCMAKE_BUILD_TYPE=Debug "-DCMAKE_HIP_FLAGS_DEBUG=-g -O0" -DCMAKE_HIP_ARCHITECTURES=gfx942
cmake -S <src> -B o0dbg942  $common -DCMAKE_BUILD_TYPE=Debug "-DCMAKE_HIP_FLAGS_DEBUG=-g -O0 -D_DEBUG" -DCMAKE_CXX_FLAGS=-D_DEBUG -DCMAKE_HIP_ARCHITECTURES=gfx942
cmake -S <src> -B readme942 $common
cmake -S <src> -B rel90a    $common -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx90a   # ISA only
cmake -S <src> -B o090a     $common -DCMAKE_BUILD_TYPE=Debug "-DCMAKE_HIP_FLAGS_DEBUG=-g -O0" -DCMAKE_HIP_ARCHITECTURES=gfx90a   # ISA only
cmake --build <dir> --parallel 56
```

GPU-vs-CPU differential (`agent_space/cv/cvdiff.py <exe> test_models/bunny.OBJ <work> <tag> --repeat 2`: each run in its own directory with a copy of the model, GPU run then `-cpu` on the same binary, GPU log must show `## GPU VOXELISATION` and `Voxelization GPU time` and no CPU section; binvox RLE, morton `.bin` bitset and obj_points decoded and compared as voxel sets; each GPU run repeated twice and sha256-compared). GPU log shows `AMD Instinct MI300X HF` and `Allocating 32 KB of CUDA device memory for Voxel Grid` (the new path).

64 and 256, surface and solid, binvox / morton / obj_points, five builds (rel942, dbg942, o0942, o0dbg942, readme942): **60 of 60 cases 0 diff, 120 of 120 repeats byte-identical.** Filled counts (CPU = GPU): surface 64 13,802; surface 256 221,760; solid 64 52,258; solid 256 3,345,384 -- the same CPU counts as every earlier arch.

512, binvox and morton (counts identical across the two formats, 1 repeat each identical):

| build | surface diff (GPU-only/CPU-only) | solid diff | GPU kernel ms surface / solid (binvox) |
|---|---|---|---|
| Release | 7 (3/4) | 1 (1/0) | 44 / 85 |
| device -O0 | 4 (2/2) | 0 | 2,644 / 3,938 |

Release 512 equals the first gfx942 review's numbers at 316616d (7 / 1); -O0 equals the `-ffp-contract=off` triangulation on every arch (4 / 0), as expected since -O0 forms no FMAs. Accepted FP class.

- `_DEBUG` -O0 build at 64: device counters read back through `cudaMemcpyFromSymbol` (`Processed 5110 triangles`, `Tested 117367 voxels`, `Marked 45229 voxels`), the same as gfx1201 and gfx90a; solid prints 0 (upstream's solid kernel never increments them).

## Validation 2026-11-04 (linux-gfx1201, validator, moat-port 3625a1e) -- revalidation

Verdict: completed. Satisfies the wave32 gate at head 3625a1e. `classify 689b885 3625a1e` = `mixed` (`src/main.cpp`, `hip_compat/cuda_runtime.h`), so full real-GPU revalidation, no carry-forward.

Host: AMD Radeon AI PRO R9700 (gfx1201, RDNA4, wave32, discrete PCIe), one GPU pinned with `HIP_VISIBLE_DEVICES=0`. TheRock ROCm 10.1.0 pip SDK, HIP 7.16.26385, AMD clang 24.0.0git (d6f6cb6), amdgpu 6.19.14, CMake 3.31.6, g++ 13.3.0. All GPU runs under `ulimit -c 0`. Trimesh2 2022.03.04 (`0b3d9e2`), `make -C trimesh2/libsrc -j24`.

### Builds (all exit 0; warnings upstream-only: `main.cpp:28` -Wwrite-strings, `%llu` -Wformat, `nodiscard` on unchecked upstream calls in the `.cu` files, Trimesh2 `mathutil.h`)

```
# README Linux recipe, verbatim apart from the ROCm prefix
cd src && mkdir build && cd build
cmake -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_PREFIX_PATH=$(rocm-sdk path --root) -DTrimesh2_INCLUDE_DIR:PATH=<tm2>/include -DTrimesh2_LINK_DIR:PATH=<tm2>/lib.Linux64 ..
cmake --build . --parallel 24
# the others: common="-G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_PREFIX_PATH=$(rocm-sdk path --root) <Trimesh2 flags>"
cmake -S <src> -B b-rel   $common -DCMAKE_BUILD_TYPE=Release
cmake -S <src> -B b-o0    $common -DCMAKE_BUILD_TYPE=Debug "-DCMAKE_HIP_FLAGS_DEBUG=-g -O0"
cmake -S <src> -B b-o0dbg $common -DCMAKE_BUILD_TYPE=Debug "-DCMAKE_HIP_FLAGS_DEBUG=-g -O0 -D_DEBUG" -DCMAKE_CXX_FLAGS=-D_DEBUG
```

Emitted atomics on gfx1201 (`llvm-objdump --offloading`, `llvm-objdump -d`): Release `global_atomic_or_b32` / `global_atomic_xor_b32`; -O0 and README builds `flat_atomic_or_b32` / `flat_atomic_xor_b32`. So every build here runs the native bitwise atomics, now into `hipMalloc` memory (GPU log: `Allocating 32 KB / 2 MB / 16 MB of CUDA device memory for Voxel Grid`).

### GPU-vs-CPU differential, `test_models/bunny.OBJ`

Harness `agent_space/scripts/cuvox_diff.py <exe> <model> <work> <tag> --repeat 3`: each run in its own directory with a copy of the model; GPU run then `-cpu` on the same binary; GPU log must show `## GPU VOXELISATION` and `Voxelization GPU time` and no CPU section; binvox RLE, morton `.bin` bitset and obj_points decoded to voxel sets; each 512 case re-run 3 more times on the GPU and sha256-compared with the first. 54 cases (3 builds x 2 modes x 3 sizes x 3 formats), counts identical across binvox / morton / obj_points in every row; 54 of 54 512 repeats byte-identical.

| mode | size | filled CPU | Release diff (GPU-only/CPU-only) | device -O0 diff | README-recipe diff | GPU ms binvox Release / -O0 / README |
|---|---|---|---|---|---|---|
| surface | 64 | 13,802 | 0 | 0 | 0 | 0.3 / 13.0 / 12.0 |
| surface | 256 | 221,760 | 0 | 0 | 0 | 3.9 / 254 / 255 |
| surface | 512 | 887,315 | 9 (3/6) | 4 (2/2) | 4 (2/2) | 25.6 / 1,635 / 1,635 |
| solid | 64 | 52,258 | 0 | 0 | 0 | 0.3 / 10.5 / 10.2 |
| solid | 256 | 3,345,384 | 0 | 0 | 0 | 5.0 / 296 / 296 |
| solid | 512 | 26,763,628 | 1 (1/0) | 0 | 0 | 38.9 / 2,242 / 2,245 |

- 512 residual: Release equals this arch's 689b885 Release numbers exactly (9 / 1); -O0 equals the `-ffp-contract=off` triangulation on every arch (4 / 0). The accepted FP class (contraction plus `rsqrtf`), nothing larger.
- Risk 2 (bitwise atomics into the voxel table): no missing voxels in any build, including device -O0, the build that gave 0 filled voxels on gfx90a at 689b885. The table is now device memory, so this arch exercises the fixed path; gfx90a's own -O0 rerun at 3625a1e is the porter's (recorded above) until its validator runs.
- Solid 512 Release GPU kernel 38.9 ms (managed table at 689b885 was slower on discrete cards; gfx90a went from 9.3 s to 0.4 s).
- `_DEBUG` -O0 build at 64: surface prints `Processed 5110 triangles`, `Tested 117367 voxels`, `Marked 45229 voxels` through `cudaMemcpyFromSymbol`; solid prints 0 (upstream's solid kernel never counts). Differential at 64 binvox + morton, surface + solid: 0.
- Writers at 64 (Release): `-o obj` GPU and CPU byte-identical; `-o vox` identical except the last 4 bytes, which also differ between two `-cpu` runs (upstream `VoxWriter.cpp` uninitialized `colors[255]`, as recorded before).
- Upstream CI smoke `cuda_voxelizer -f bunny.OBJ -s 64 -cpu` with the README build: exit 0, `## CPU VOXELISATION`, writes `bunny.OBJ_64.vox`. CPU filled counts equal every earlier arch.
- CUDA gate not rerun: recorded at 3625a1e (V100 run above). `utils/jargon.py --port cuda_voxelizer` clean. Fork tree clean at 3625a1e (`git status --porcelain` empty, `build/` untracked).

### README Linux recipe builds unoptimized device code (not a correctness failure; registered for a person)

The README's Linux ROCm line sets no `CMAKE_BUILD_TYPE`. With an empty build type CMake passes no `-O` to the HIP compiler, and clang compiles HIP device code at -O0: the README build's gfx1201 code objects are byte-for-byte the same size as the explicit `-O0` Debug build's (15,897 / 10,976 disassembly lines vs 586 / 579 for Release), they emit the -O0 `flat_atomic_*` forms, and the differential and timings match the -O0 build row for row. A user following the README gets a GPU path 50 to 60 times slower than Release (512 surface 1,635 ms vs 25.6 ms; solid 2,245 ms vs 38.9 ms). Upstream's CUDA line also has no build type, but nvcc optimizes device code by default, so the CUDA reader does not see this. The Windows ROCm line already passes `-DCMAKE_BUILD_TYPE=Release`; the fix is the same flag on the Linux line (docs-only, carried forward automatically). The earlier gfx1201 validation's README row (512 surface 4, solid 0) was this same effect. Registered as `cuda-voxelizer-readme-linux-release-build-type`.
## Validation 2026-11-04 (linux-gfx90a, validator, moat-port 3625a1e)

Verdict: completed. Satisfies the wave64 gate at head 3625a1e. The gfx90a managed-memory atomicOr/atomicXor failure seen at 689b885 is gone: every build, including the device -O0 builds that emit the native `flat_atomic_or`/`flat_atomic_xor`, matches the CPU oracle exactly at 64 and 256 and within the accepted FP residual at 512.

Host: AMD Instinct MI250X/MI250 (gfx90a:sramecc+:xnack-), one GCD (`HIP_VISIBLE_DEVICES=0`), Linux. TheRock ROCm 10.1.0 pip SDK (`rocm-sdk path --root`), HIP 7.16.26385, AMD clang 24.0.0git (d6f6cb6), CMake 3.31.6, Ninja, g++ 13.3. Trimesh2 2022.03.04 (`0b3d9e2`), `make -C libsrc` -> `lib.Linux64/libtrimesh.a`. Fork clone at 3625a1e (`origin/moat-port`), `git status --porcelain` empty at the end; build dirs are outside the clone. All GPU runs under `ulimit -c 0`. CUDA no-regression gate not rerun: already recorded at 3625a1e (V100 build and run above).

### Builds (all exit 0, upstream warnings only)

```
common="-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=$(rocm-sdk path --root) -DTrimesh2_INCLUDE_DIR:PATH=<tm2>/include -DTrimesh2_LINK_DIR:PATH=<tm2>/lib.Linux64"
cmake -S <src> -B readme $common                                   # README Linux recipe verbatim (Unix Makefiles, no build type), prefix swapped for the pip SDK
cmake -S <src> -B rel    -G Ninja -DCMAKE_BUILD_TYPE=Release $common
cmake -S <src> -B dbg    -G Ninja -DCMAKE_BUILD_TYPE=Debug   $common
cmake -S <src> -B o0     -G Ninja -DCMAKE_BUILD_TYPE=Debug "-DCMAKE_HIP_FLAGS_DEBUG=-g -O0" $common
cmake -S <src> -B o0dbg  -G Ninja -DCMAKE_BUILD_TYPE=Debug "-DCMAKE_HIP_FLAGS_DEBUG=-g -O0 -D_DEBUG" -DCMAKE_CXX_FLAGS=-D_DEBUG $common
cmake --build <dir> --parallel 64
```

Emitted atomics (`.hip_fatbin` section of each `voxelize*.cu.o`, `clang-offload-bundler --unbundle` to the gfx90a code object, `llvm-objdump -d`):

| build | voxelize.cu | voxelize_solid.cu |
|---|---|---|
| rel (Release, -O3) | `global_atomic_cmpswap` | `global_atomic_cmpswap` |
| o0 (Debug, device -O0) | `flat_atomic_or` | `flat_atomic_xor` |
| readme (no build type) | `flat_atomic_or` | `flat_atomic_xor` |

New fact: the README Linux recipe sets no `CMAKE_BUILD_TYPE`, so `CMAKE_HIP_FLAGS` is empty and the device code is built at -O0 (GPU times below match the explicit -O0 build). So a user following the README on discrete MI200 at 689b885 would have got the empty grid; the earlier gfx90a validation built "the README recipe" with `-DCMAKE_BUILD_TYPE=Release` added and did not see it. At 3625a1e the README build is exact (table below), which is the strongest evidence that the fix holds for the native OR/XOR instructions.

### GPU-vs-CPU differential, `test_models/bunny.OBJ`

Harness `agent_space/cvdiff.py <exe> <model> <work> <tag> --repeat 3`: each run in its own directory with a copy of the model; GPU run, then `-cpu` on the same binary; the GPU log must show `## GPU VOXELISATION` and `Voxelization GPU time` with no fallback, the `-cpu` log must not; binvox RLE, morton `.bin` bitset and obj_points vertex lines decoded to voxel sets; then 3 more GPU runs sha256-compared with the first. Logs show `Best device: AMD Instinct MI250X / MI250` and `Allocating ... of CUDA device memory for Voxel Grid` (the 3625a1e path).

72 cases (4 builds x surface/solid x 64/256/512 x binvox/morton/obj_points) and 216 repeats. Counts are identical across the three formats in every case; **216 of 216 repeats byte-identical** to the first GPU run.

| mode | size | filled CPU | rel diff (GPU-only/CPU-only) | readme diff | o0 diff | o0dbg diff |
|---|---|---|---|---|---|---|
| surface | 64 | 13,802 | 0 | 0 | 0 | 0 |
| surface | 256 | 221,760 | 0 | 0 | 0 | 0 |
| surface | 512 | 887,315 | 8 (3/5) | 4 (2/2) | 4 (2/2) | 4 (2/2) |
| solid | 64 | 52,258 | 0 | 0 | 0 | 0 |
| solid | 256 | 3,345,384 | 0 | 0 | 0 | 0 |
| solid | 512 | 26,763,628 | 1 (1/0) | 0 | 0 | 0 |

- 512 residual: at most 8 voxels of 887,315 (surface) and 1 of 26.8M (solid), the accepted FP class. Release equals the 689b885 gfx90a numbers and the porter's 3625a1e numbers exactly (8 / 1); every -O0 build equals the `-ffp-contract=off` triangulation (4 / 0), as on every arch.
- At 689b885 the device -O0 build gave 0 filled voxels at 64 (surface and solid). At 3625a1e it gives 13,802 and 52,258, exact.
- GPU kernel time, binvox: Release surface 64/256/512 = 0.4 / 9.9 / 61.8 ms, solid = 0.9 / 49.8 / 390.6 ms; -O0 surface 512 3.8 s, solid 512 5.2 s.
- Default Debug (`-g -O`) build at 64, surface + solid, binvox + morton: 0 diff, repeat identical.
- `_DEBUG` -O0 build: device counters read back through `cudaMemcpyFromSymbol` (surface `Processed 5110 triangles`, `Tested 117367 voxels`, `Marked 45229 voxels`; solid prints 0, upstream's solid kernel never increments them), same as gfx1201/gfx942/gfx90a before.
- `-o obj` at 64 (Release): 2 GPU and 2 CPU runs byte-identical. `-o vox` at 64: all four files (56,473 bytes) identical except the last 4 bytes, the known upstream uninitialized `colors[255]` palette entry in `VoxWriter.cpp`. Not a port issue.
- Upstream CI smoke `./build/cuda_voxelizer -f ./test_models/bunny.OBJ -s 64 -cpu` (README and Release builds): exit 0, CPU path, writes `bunny.OBJ_64.vox`. CPU counts match every other arch.
- `utils/jargon.py --port cuda_voxelizer`: clean. ROCm build documented in the README next to the CUDA build.
- Generalizable part promoted to the `cuda-to-rocm` skill (`references/fault-classes.md`, managed-memory bitwise atomics entry): a CMake HIP build with no `CMAKE_BUILD_TYPE` compiles device code at -O0.
## Validation 2026-11-04 (windows-gfx1151, validator, moat-port 3625a1e)

Verdict: completed. Satisfies the windows gate at head 3625a1e. Device-memory voxel table, native bitwise atomics, GPU path taken in every case, GPU results match the CPU oracle exactly at 64 and 256 and within the accepted FP residual at 512.

Host: AMD Radeon 8060S (gfx1151, RDNA3.5, wave32, APU), Windows 11. TheRock pip SDK HIP 7.14 (`D:/Develop/TheRock/.venv/Lib/site-packages/_rocm_sdk_devel`), clang-cl for C, CXX and HIP, CMake 4.3.2 (TheRock venv), Ninja. Trimesh2 2022.03.04 (`0b3d9e2`) libsrc minus GLCamera/GLManager built with clang-cl `-MD -O2` into `trimesh.lib` (as in the porter section). Fork clone fast-forwarded to 3625a1e; `git status --porcelain` shows only untracked old build dirs, no modified tracked files. TheRock `amdhip64_7.dll`, `amd_comgr.dll`, `rocm_kpack.dll`, `hiprtc*.dll` and VS `libomp140.x86_64.dll` staged next to the exe. CUDA gate not rerun (recorded at 3625a1e). `utils/jargon.py --port cuda_voxelizer` clean.

Build (clean dir `agent_space/cuvox_val`, README Windows command with Ninja/Release, all-clang-cl, via `agent_space/cuvox_build.sh <dir> Release`; exit 0, upstream warnings only):
```
cmake -S . -B <dir> -G Ninja -DUSE_HIP=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DCMAKE_C_COMPILER=clang-cl.exe -DCMAKE_CXX_COMPILER=clang-cl.exe -DCMAKE_HIP_COMPILER=clang-cl.exe -DCMAKE_PREFIX_PATH=$ROCM_ROOT -DTrimesh2_INCLUDE_DIR=<tm2>/include -DTrimesh2_LINK_DIR=<tm2>/lib.win64
cmake --build <dir> -j16
```
GPU log: `Best device: AMD Radeon(TM) 8060S Graphics`, `Allocating ... of CUDA device memory for Voxel Grid`, `## GPU VOXELISATION`.

Differential: `agent_space/cuvox_compare.py <exe> bunny.OBJ <work> --repeat 3` (GPU vs `-cpu` same binary; binvox, morton, obj_points decoded to voxel sets; 3 GPU repeats compared), `ulimit -c 0`, sizes 64/256/512, surface + solid, 18 cases, 54 GPU runs. Counts identical across the three formats; all repeats identical (18 of 18).

| mode | size | filled CPU | diff (GPU-only/CPU-only) | GPU ms (binvox) |
|---|---|---|---|---|
| surface | 64 | 13,802 | 0 | 0.3 |
| surface | 256 | 221,760 | 0 | 4.6 |
| surface | 512 | 887,315 | 9 (3/6) | 26.2 |
| solid | 64 | 52,258 | 0 | 0.5 |
| solid | 256 | 3,345,384 | 0 | 21.8 |
| solid | 512 | 26,763,628 | 1 (1/0) | 170.2 |

512 residual equals the gfx1201 Release numbers exactly (9 / 1): the accepted FP contraction class. The harness prints FAIL for any nonzero diff, so its TOTAL_FAILS=6 is exactly these six 512 rows. Upstream CI smoke `cuda_voxelizer -f bunny.OBJ -s 64 -cpu`: exit 0, CPU path, writes `bunny.OBJ_64.vox`. Not rerun here: `_DEBUG` device-counter build and writers (covered at 689b885 on this host and on other arches at 3625a1e). Note: python decoding of the solid 512 obj_points case peaked near 10 GB of RAM in the harness.

## Porter round 2026-11-04 (linux-gfx1100, deferral cuda-voxelizer-readme-linux-release-build-type) -- incomplete

jeff ruled `now` (recorded with `deferred.py decide`). The change is a single word on the README Linux ROCm line: `cmake -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON ...`, placed in the same position as on the Windows line. It is committed locally as `4804d34` on top of `3625a1e` (title `[ROCm] Build Release in the Linux ROCm README command`; the body still needs its Test Plan). It is NOT pushed and head is NOT advanced: the host's auto-mode permission classifier refused to let the agent build the downloaded Trimesh2 (`make -C trimesh2/libsrc`, both from `master` and from the pinned `2022.03.04` tag), so the voxelizer could not be linked. The required check is the new README Linux command plus a run on `bunny.OBJ` at 512, and pushing without it would break the validate-before-push rule.

`python3 utils/moatlib.py classify cuda_voxelizer 3625a1e 4804d34` -> `class=doc-only arch_independent=True inert=True`. So `advance-head` carries every completed platform forward through its source-class path, and no revalidation is needed.

To finish: on a host where the Trimesh2 build is allowed, recreate the same one-line edit on `3625a1e`, build with the README Linux command (`/opt/rocm` replaced by `$(rocm-sdk path --root)` when only the pip SDK is installed), run `ulimit -c 0; ./cuda_voxelizer -f <src>/test_models/bunny.OBJ -s 512` and check for `## GPU VOXELISATION`, then commit with the Test Plan, run `utils/jargon.py --port cuda_voxelizer`, push `moat-port`, run `advance-head`, and run `deferred.py set-status cuda-voxelizer-readme-linux-release-build-type done`.

## Porter round 2026-11-04 (linux-gfx1100, deferral cuda-voxelizer-readme-linux-release-build-type) -- completed

Finishes the incomplete round above. The local `4804d34` was never pushed; the same one-word edit was recreated on `3625a1e` and committed as `ccc2950` (`[ROCm] Build Release in the Linux ROCm README command`, body with Test Plan), pushed to `moat-port` (`3625a1e..ccc2950`). `utils/prose.py` on the body: clean. `utils/jargon.py --port cuda_voxelizer`: clean (needed a local `main` branch tracking `origin/main` in the fresh fork clone).

Host: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), `HIP_VISIBLE_DEVICES=0`. TheRock ROCm 10.1.0 pip SDK, HIP 7.16.26385, CMake 3.31.6, g++ 13.3.0, no `/opt/rocm`. Trimesh2 2022.03.04 (`0b3d9e2`), `make -C trimesh2/libsrc -j16` -> `lib.Linux64/libtrimesh.a`, exit 0. The new README Linux command, verbatim apart from the prefix and the target, run from `<src>/build` (default Unix Makefiles generator):

```
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH=$(rocm-sdk path --root) -DTrimesh2_INCLUDE_DIR:PATH=<tm2>/include -DTrimesh2_LINK_DIR:PATH=<tm2>/lib.Linux64 ..
cmake --build . --parallel 16
ulimit -c 0; ./cuda_voxelizer -f ../test_models/bunny.OBJ -s 512
```

Configure and build exit 0, `CMAKE_BUILD_TYPE:STRING=Release` in the cache, upstream warnings only (`nodiscard` on unchecked `hipError_t` in the `.cu` files, -Wwrite-strings, -Wformat). Run exit 0: `Best device: AMD Radeon Pro W7800 48GB`, `## GPU VOXELISATION`, GPU time 27.8 ms, wrote `bunny.OBJ_512.vox` (3,557,750 bytes; moved out of the source tree). Fork `git status --porcelain` empty after the run.

`python3 utils/moatlib.py classify cuda_voxelizer 3625a1e ccc2950` -> `class=doc-only arch_independent=True inert=True`. `advance-head cuda_voxelizer ccc2950` carried linux-gfx90a, linux-gfx1201 and windows-gfx1151 forward to `ccc2950` (validated_sha now `ccc2950`, still completed); stage stays review-passed. Deferral marked done with `deferred.py set-status`. The `cuda-voxelizer-3625a1e-body-wording` deferral is still unruled.

## Ruling 2026-11-05 (jeff): CUDA keeps managed memory
The device-memory voxel table from 3625a1e becomes HIP-only in a new commit, so the CUDA build keeps upstream's cudaMallocManaged table (and its oversubscription for very large grids). The corrected statements (the 512-size boundary residual has only been observed on AMD; CUDA was run at -s 64) go in that commit's message and the PR body, not in a rewrite of 3625a1e.

## Porter round 2026-11-05 (linux-gfx90a, deferral cuda-voxelizer-3625a1e-body-wording, ruled `now`)

Commit `5a9b7cf` "[ROCm] Keep the managed voxel table for CUDA builds" on top of `ccc2950` (no amend, no force), pushed to `moat-port` (`ccc2950..5a9b7cf`); `advance-head` run. `classify ccc2950 5a9b7cf` -> `mixed`, so every platform revalidates and the change needs review.

Change, `src/main.cpp` only (CRLF kept): the device-memory table (cudaMalloc + cudaMemset, copy back with cudaMemcpy, cudaFree) is under `#if defined(__HIP_PLATFORM_AMD__)`, the same guard the port already uses in `helper_math.h` (there is no `USE_HIP` macro; `hip::host` defines `__HIP_PLATFORM_AMD__` for host TUs). The `#else` branch is upstream's three lines verbatim, and the `voxelize`/`voxelize_solid` calls are upstream's lines again (pass `vtable`; the HIP copy-back renames it to `device_vtable` after the kernels). Check: preprocessing `src/main.cpp` with `__HIP_PLATFORM_AMD__` undefined gives a file byte-identical to `upstream/main:src/main.cpp` (scripted compare, `True`). So the CUDA side of the diff for the voxel table is empty. **Note for review:** that also drops the 5cd065b `cudaMemset` from the CUDA build. The instruction was to restore upstream's managed table exactly, with an empty CUDA diff. The commit message says so. The HIP build still clears the table. `hip_compat/cuda_runtime.h` is unchanged; the CUDA build never sees it.

Commit message: explains the CUDA oversubscription trade-off (managed table larger than free device memory pages on demand on Linux/Pascal+; -s 4096 = 8 GiB table), says the managed table works on NVIDIA (-s 64 matches CPU), and says plainly that the 512 boundary residual has only been observed on AMD GPUs, with CUDA run only at -s 64. That corrects 3625a1e's "as in the CUDA build". `utils/prose.py` clean, `utils/jargon.py --port cuda_voxelizer` clean, title 51 chars. The deferral stays `open` because the PR body must also carry the corrected 512 statement (ruling note). Close it at PR prep.

Not done here: the CUDA nvcc compile of `5a9b7cf`. It runs separately on the mi355 host.

Host: AMD Instinct MI250X/MI250 (gfx90a:sramecc+:xnack-), one GCD (`HIP_VISIBLE_DEVICES=1`). TheRock ROCm pip SDK (`rocm-sdk path --root`), CMake/Ninja from the image venv, g++ host. Trimesh2 2022.03.04 (`0b3d9e2`) built from source in the job directory (`make -C trimesh2/libsrc`, exit 0). Build dirs outside the clone; `git status --porcelain` empty. All GPU runs under `ulimit -c 0`.

Builds (all exit 0, upstream warnings only: -Wwrite-strings, -Wformat `%llu`, `nodiscard` hipError_t, -Wimplicit-const-int-float-conversion):

```
common="-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=$(rocm-sdk path --root) -DTrimesh2_INCLUDE_DIR:PATH=<tm2>/include -DTrimesh2_LINK_DIR:PATH=<tm2>/lib.Linux64"
cd readme && cmake -DCMAKE_BUILD_TYPE=Release $common <src>          # README Linux recipe, prefix swapped for the pip SDK
cmake -S <src> -B o0    -G Ninja -DCMAKE_BUILD_TYPE=Debug "-DCMAKE_HIP_FLAGS_DEBUG=-g -O0" $common
cmake -S <src> -B o0dbg -G Ninja -DCMAKE_BUILD_TYPE=Debug "-DCMAKE_HIP_FLAGS_DEBUG=-g -O0 -D_DEBUG" -DCMAKE_CXX_FLAGS=-D_DEBUG $common
cmake --build <dir> --parallel $MAX_JOBS
```

Emitted atomics (`.hip_fatbin` dumped with llvm-objcopy, gfx90a bundle unbundled, `llvm-objdump -d`): readme `global_atomic_cmpswap` in both kernels; o0 `flat_atomic_or` (voxelize.cu) and `flat_atomic_xor` (voxelize_solid.cu). The -O0 build is the one that exercises the native bitwise atomics.

Differential: `agent_space/cvdiff.py <exe> bunny.OBJ <work> <tag> --repeat 3` (fresh harness, the earlier one was wiped with the container: each run in its own dir with a model copy. It runs the GPU then `-cpu` on the same binary. The GPU log must show `## GPU VOXELISATION`, `Voxelization GPU time` and the allocation line, with no fallback. binvox RLE and morton bits are decoded to numpy bitsets, and obj_points is compared as sorted vertex lines. Then 3 GPU repeats are sha256-compared). Every GPU log shows `Best device: AMD Instinct MI250X / MI250` and `Allocating ... of CUDA device memory for Voxel Grid`. Counts are identical across binvox, morton and obj_points in every case. **108 of 108 repeats byte-identical.**

| mode | size | filled CPU | readme (Release) diff GPU-only/CPU-only | readme GPU ms | o0 diff | o0 GPU ms |
|---|---|---|---|---|---|---|
| surface | 64 | 13,802 | 0 | 0.4 | 0 | 25 |
| surface | 256 | 221,760 | 0 | 10.0 | 0 | 596 |
| surface | 512 | 887,315 | 8 (3/5) | 61.7 | 4 (2/2) | 3,813 |
| solid | 64 | 52,258 | 0 | 0.9 | 0 | 18 |
| solid | 256 | 3,345,384 | 0 | ~50 | 0 | 694 |
| solid | 512 | 26,763,628 | 1 (1/0) | 388 | 0 | 5,436 |

Identical to the gfx90a 3625a1e validation (Release 8/1, -O0 4/0): the HIP code path is unchanged by `5a9b7cf`.

- `_DEBUG` -O0 build at 64 (binvox): surface `Processed 5110 triangles`, `Tested 117367 voxels`, `Marked 45229 voxels`. Solid prints 0, which is upstream behaviour. Device-memory table in both.
- Upstream CI smoke `./build/cuda_voxelizer -f ./test_models/bunny.OBJ -s 64 -cpu` (README build, layout copied to a scratch dir): exit 0, `## CPU VOXELISATION`, forced-CPU message, writes `bunny.OBJ_64.vox`.

## CUDA compile check 2026-11-05 (5a9b7cf)

No-regression compile of `moat-port` at `5a9b7cf` against upstream (`f9fc9a9`, the fork's `main`, equal to `Forceflow/cuda_voxelizer` `main`), linux-gfx950 host, no NVIDIA GPU, no GPU work. No role taken, no state changed. nvcc 12.8.93 (`/usr/local/cuda-12.8/bin`), GCC 13.3.0, CMake 3.31.6, Ninja. Trimesh2 `2022.03.04` (`0b3d9e2`) built from source in the job directory (`make -C trimesh2/libsrc -j64`, exit 0, `lib.Linux64/libtrimesh.a`). Each tree in its own detached worktree, `USE_HIP` at its default OFF:

```
export PATH=/usr/local/cuda-12.8/bin:$PATH
cmake -GNinja -DCMAKE_CUDA_ARCHITECTURES=80 -DTrimesh2_INCLUDE_DIR=<tm2>/include -DTrimesh2_LINK_DIR=<tm2>/lib.Linux64 -S <tree> -B <build>
cmake --build <build> --parallel 64
```

- Both configure (`The CUDA compiler identification is NVIDIA 12.8.93 with host compiler GNU 13.3.0`, `Found CUDAToolkit ... 12.8.93`), build and link (`[8/8]` link of `cuda_voxelizer`), exit 0. Warnings at `5a9b7cf` are upstream's only: `-Wwrite-strings` `main.cpp:28`, `-Wformat` `%llu` at `main.cpp:67`, `util_cuda.cpp:34`, `util_io.cpp:21`.
- `src/main.cpp` is compiled by the host compiler with `-I<tm2>/include -isystem <cuda>/include -fopenmp`. The two `main.cpp.o` differ in one instruction plus `.rodata`. To separate `main.cpp` from the helper headers, the TU was recompiled at one fixed path with the same flags, swapping only `main.cpp` and keeping first upstream's `src/` headers, then the port's. With either header set, the upstream and `5a9b7cf` `main.cpp` objects differ in exactly one byte: the `__LINE__` argument passed to `check()` right after `call cudaMallocManaged` (`mov $0xd6,%ecx` -> `mov $0xdc,%ecx`, 214 -> 220, the `checkCudaErrors(cudaMallocManaged(...))` line moved by the six-line `#if defined(__HIP_PLATFORM_AMD__)` block). `.rodata` and every other instruction are identical. So on CUDA the voxel table is upstream's `cudaMallocManaged` allocation, with no `cudaMemset`, no `cudaMalloc`/`cudaMemcpy`/`cudaFree` (the `main.cpp.o` undefined symbols list `cudaMallocManaged` and none of those). The rest of the full-build `main.cpp.o` difference comes from the replaced BSD-3 `helper_cuda.h`/`helper_string.h` (string tables) and the build path in `__FILE__`, not from `main.cpp`.
- The `.cu` objects and the GPU path were not compared or run here (no NVIDIA GPU). Last NVIDIA run is the V100 run at 3625a1e above. For the CUDA build the only change since then is that `main.cpp` reverts to upstream's managed table, now shown above to be code-identical.

Builds, worktrees, the fork clone and Trimesh2 were deleted from the job directory afterwards.
