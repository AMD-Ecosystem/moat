# PhysX port plan

Planner run 2026-11-20 on linux-gfx1100 (no GPU used, nothing built). Decisions that need Jeff are listed at the top of the 2026-11-20 planning section of `notes.md` and under "Open questions" below; Question 1 must be answered before the porter starts.

## Project

- Upstream: https://github.com/NVIDIA-Omniverse/PhysX, default branch `main`, tip `7d6f6ac8` ("update ovphysx 0.8.0", 2026-11-13), PhysX SDK 5.13.0.
- Fork: https://github.com/AMD-Ecosystem/PhysX. `main` is the upstream mirror (still at `4f2103c3`, PhysX 5.10.0) and the recorded `fork_default_branch`. `rocm-enablement` is `d09e2532`, the AMD team's single squashed commit on top of `4f2103c3`.
- Repository layout at 5.13: `physx/` (the SDK, the only part with GPU simulation code), `ovphysx/` (USD/Python product on top of the SDK), `blast/`, `flow/`, `schemas/`.
- Build type: `cmake` (recorded with `set-ext-type`).

## Existing AMD support

- `AMD-Ecosystem/PhysX@rocm-enablement` (d09e2532, 2026-09-16, signed off by three AMD engineers): authoritative, AMD-official, incomplete. It ports the Linux GPU CMake path of the SDK through a force-included `cuda_to_hip.h` plus `hip_compat/` shadow headers (shim-header method, sources mostly byte-identical). Its evidence is artifact checks (fatbin present, links amdhip64, 61 gfx942 code objects) and a module-load/dispatch ladder on MI300X/MI355X. INSTALL_ROCm.md and README.md name no PhysX snippet or simulation run.
- Decision (Jeff, 2026-11-10): build on it, do not re-port. `moat-port` is cut from `d09e2532` keeping its authorship; our work goes on top as separate commits.
- Not a performance-kernel project in the CUTLASS sense: no CUTLASS/CuTe/wgmma/tensor cores. A mechanical HIP port is the right vehicle; no AMD-native rewrite is planned.
- Upstream deliberately linking platform forks: no evidence either way (no ROCm/HIP/AMD PRs or issues upstream). Delivery vehicle stays Jeff's later call (ruling of 2026-11-10).

## What changed upstream since the AMD base

Upstream went from 5.10.0 (`4f2103c3`) to 5.13.0 (`7d6f6ac8`) in 4 commits touching 5240 files, 1505 of them under `physx/source` (305 modified files under `physx/source/gpu*`, 5 added: `PxgDeformableContactWriters.h`, `PxgDeformableFilterManager.h/.cpp`, `PxgAttachmentManager.cpp`, `deformableAndParticleUtils.cuh`; 1 removed: `PxgDeformableContactInfo.h`).

- Relicence: every `physx/` source file now carries `SPDX-License-Identifier: Apache-2.0` and `physx/LICENSE.md` is Apache 2.0. The repository root `LICENSE.md` still says BSD-3-Clause (which is what `licenses.py check` and the record report). See Open questions.
- Trial merge of `d09e2532` with `7d6f6ac8` (`git merge-tree`, in agent_space): 24 conflicting files. 21 conflict only in the licence header at line 1 (upstream replaced the BSD text with SPDX lines; AMD added a "Modifications Copyright" line or trimmed trailing whitespace in the same block). Three carry content: `physx/CMakeLists.txt` (8 hunks, 155 lines), `physx/source/compiler/cmakegpu/linux/CMakeLists.txt` (one 279-line hunk, AMD's HIP block against upstream's rewrite), and `physx/source/gpunarrowphase/src/PxgGeometryManager.cpp` (2 hunks). `-X ignore-all-space` removes only one conflict, so the whitespace clean-up does not change the merge much.
- GPU code at 5.13: 61 `.cu` translation units (same count as the base), 495 `__global__` kernels (497 at the base). No new kernel TU, so the AMD CMake source lists still apply; the new host `.cpp` files fall under the existing per-target HIP host-seam flags.
- New driver API since the base: `cuOccupancyMaxActiveBlocksPerMultiprocessor` (`PxgCudaBroadPhaseSap.cpp:206`), not in the AMD shim. New `__CUDA_ARCH__` gate: `broadphase.cu:1505` (`== 750` occupancy hint; the default branch is fine on HIP).

## Build classification

Pure CMake, Strategy A (compat header). Evidence: `physx/CMakeLists.txt` declares `project(PhysX ... LANGUAGES C CXX CUDA)` when `PX_GENERATE_GPU_PROJECTS` is on and adds `source/compiler/cmakegpu`; the GPU targets are plain `add_library` targets in `physx/source/compiler/cmakegpu/*.cmake` with `.cu` sources. No torch anywhere in `physx/`. PhysX launches every kernel through the CUDA driver API (`cuModuleLoadDataEx`, `cuModuleGetFunction`, `cuLaunchKernel` in `cudamanager/src/CudaContextManager.cpp` and `CudaKernelWrangler.cpp`), with the module table filled by intercepting the nvcc registration calls. That driver-API seam is the project-specific part of the port; the AMD branch maps it to `hipModule*`.

## Port strategy

Keep the AMD design: `PX_USE_HIP` option, `.cu` files marked `LANGUAGE HIP`, `-include cuda_to_hip.h` and the `hip_compat/` shadow headers on the HIP include path only, host seam `.cpp` files compiled by the host compiler with `-D__HIP_PLATFORM_AMD__`, no `-fgpu-rdc` (each TU is an independent module, as with nvcc). Warp model: PhysX is hard-wired to a 32-lane warp (`WARP_SIZE` 32, per-warp shared scratch, 32-bit masks); the shim pins every `*_sync` collective to a 32-lane logical warp and returns a subgroup-local 32-bit ballot, so a wave64 wavefront runs as two logical warps and wave32 runs natively. This is wavefront-neutral by construction and is kept; the risks below say what to check.

Commit order on `moat-port` (ruling of 2026-11-10):

1. `d09e2532` unchanged (AMD authorship).
2. Merge of upstream `main` (`7d6f6ac8`, 5.13.0). Header conflicts: take upstream's SPDX header and keep AMD's "Modifications Copyright" line under it. Content conflicts: re-apply AMD's HIP blocks onto upstream's rewritten CMake files and `PxgGeometryManager.cpp`.
3. Whitespace revert: undo every whitespace-only change of `d09e2532` (`git diff -w` shows 2089/51 lines against 2578/540 raw; the bulk is stripped indentation in `physx/CMakeLists.txt`, `cmakegpu/CMakeLists.txt`, `cmakegpu/linux/CMakeLists.txt`, `README.md`, and the code blocks of `INSTALL_ROCm.md`, plus a lost final newline in `CudaKernelWrangler.cpp`). After this commit `git diff -w` and `git diff` of the ROCm delta should agree.
4. Comment clean-up: the AMD comments cite internal records that mean nothing upstream ("See projects/PhysX/notes.md", "KB <slug>", "plan Risk #3", "projects/PhysX/plan.md Risk #1", and the device `printf` that names `projects/PhysX/notes.md`). Replace each with the hazard itself or delete it. `jargon.py --port PhysX` must be clean.
5. Onward, one concern per commit: the fixes and features listed in the change list.

## CUDA surface inventory (5.13, `physx/source`)

- 61 `.cu` TUs, 495 kernels across gpuarticulation (6 CUDA files), gpubroadphase (2), gpucommon (17), gpunarrowphase (48), gpusimulationcontroller (24), gpusolver (23).
- Warp intrinsics: `__shfl*` 555 hits/45 files, `__ballot*` 229/46, `__any_sync`/`__all_sync` 28/9, `warpSize`/`WARP_SIZE` 701/74, `__syncwarp` in about 44 files. No `__activemask`, `clock64`, `__nanosleep` or `__match_*`.
- Inline PTX: `bar.sync N` x5 (`convexMeshMidphase.cu`, mapped to `__syncthreads()`; upstream's own comments say each is a whole-block barrier, so this is exact), `red.global.add.f32` (`atomic.cuh`, mapped to `atomicAdd`), `shfl.up` scans (`warpHelpers.cuh`, rewritten with `__shfl_up_sync`), `ld.global.cg.v4` in `solverMultiBlock.cu` (commented out upstream; no action).
- Atomics: 206 hits/43 files (float `atomicAdd`, `atomicCAS`, `atomicMin/Max`, 64-bit `AtomicOr`).
- `__launch_bounds__`: 110/31. HIP reads the second argument as min warps per EU, not min blocks per SM; the AMD shim passes the CUDA value through as a hint. Occupancy only, not correctness.
- Textures: SDF only. `tex3D<float>` in `sdfCollision.cuh`, created as 3D `CUarray` + texture objects in `PxgGeometryManager.cpp` (`CU_AD_FORMAT_FLOAT`, `UNSIGNED_INT8`, `UNSIGNED_INT16`, clamp addressing, linear filter, normalized-float read for the unorm formats), carried as a 64-bit `CUtexObject` inside the geometry blob (`triangleMesh.cuh:162`). The SDF sampler is reached from `convexMeshMidphase.cu`, `particleSystemMeshMidphase.cu`, `femClothPrimitives.cu`, `softbodySoftbodyMidPhase.cu`. `bvh.cuh` `tex1Dfetch` is commented out upstream.
- `__constant__` tables defined in headers (8 files); no `cuModuleGetGlobal`/`cudaMemcpyToSymbol`, so the AMD `static __constant__` change in `cudaNpCommon.h` is safe on both toolchains.
- Driver API: 54 distinct `cu*` entry points (context, module, launch, stream, event, memory, peer copy, host-mapped memory, 3D array/texture). No CUDA graphs, IPC, external memory or graphics interop. No cuBLAS/cuFFT/cuRAND/cuSPARSE/cuSOLVER/cuDNN/Thrust/CUB/NCCL/NVRTC (the surface scan's 76 "thrust" hits are vehicle-engine variables).
- Pinned/mapped memory: `cuMemHostAlloc` with `PORTABLE`/`DEVICEMAP`/`WRITECOMBINED`, `cuMemHostGetDevicePointer`.
- Outside `physx/`: `ovphysx/` has its own CUDA (4 `.cu`: `TypeConversionUtility.cu`, `CudaKernels.cu`, `OvStageOutputCuda.cu`; about 30 CUDA-runtime host files; Warp arrays as output) and depends on NVIDIA Carbonite and ovstage binaries fetched by packman. `blast/` has no CUDA (one comment). `flow/` is D3D12/Vulkan shader compute (99 shader files), no CUDA. Scope of ovphysx is Question 1.

## Risk list

1. **Silent no-ops where HIP leaves `__CUDA_ARCH__` undefined (defects in the AMD branch, found by this plan).**
   - `gpusimulationcontroller/include/PxgArticulationBlockData.h:188`: `atomicAddSpatialVector` is `#if __CUDA_ARCH__ >= 200 ... #else PX_UNUSED(...)`, so on HIP it adds nothing. Called from `rigidDeltaAccum.cu:807/809` (articulation scratch and solver spatial impulses). Silent wrong physics, no error.
   - `gpunarrowphase/include/cudaNpCommon.h:107`: `#if __CUDA_ARCH__ atomicOr(&overflowError, ...)`: on HIP, narrowphase buffer overflows are never reported.
   - Fix: extend the guard for HIP device compilation (`|| defined(__HIP_DEVICE_COMPILE__)`), never re-key it. Then sweep every `__CUDA_ARCH__`, `__CUDACC__`, `__CUDACC_VER_MAJOR__` and `CUDART_VERSION` gate in `physx/` once more; the current list (`PxgIntrinsics.h`, `PxMath.h`, `particlesystem.cu`, `broadphase.cu`) falls back to correct generic code.
2. **The SDK never loads the HIP GPU library.** `physx/source/physx/src/gpu/PxPhysXGpuModuleLoader.cpp` (Linux) does `dlopen("libcuda.so.1")` and gives up when it fails, before opening `libPhysXGpu_64.so`; Windows delay-loads `nvcuda.dll` (`cmakegpu/windows/PhysXGpu.cmake:12-15`) and loads through `CmModuleUpdateLoader`. Every GPU snippet then prints a load error, `PxCreateCudaContextManager` returns null and the snippet silently runs on the CPU (`SnippetHelloGRB.cpp:70`, `SnippetSDF.cpp:153`). The AMD branch did not touch the loader, which is consistent with its evidence never including an SDK-level run. Needs a `PX_USE_HIP` branch (the CPU SDK must see the define) that loads the HIP runtime instead.
3. **Module registration by symbol interposition.** The AMD branch defines `__hipRegisterFatBinary`/`__hipRegisterFunction`/`__hipRegisterVar`/`__hipUnregisterFatBinary` in `CudaKernelWrangler.cpp` and forwards with `dlsym(RTLD_NEXT)`. On Linux the module constructors' calls bind through the global scope first, so if `libamdhip64` is already loaded globally when `libPhysXGpu_64.so` is opened (any host application that uses HIP itself: PyTorch, Warp, ovphysx), the runtime's definition wins, PhysX's module table stays empty and every kernel lookup fails. Bind the wrappers locally (protected visibility or `-Wl,-Bsymbolic-functions` on PhysXGpu) and test with the HIP runtime preloaded (`LD_PRELOAD=libamdhip64.so` on one snippet run). On Windows there is no `RTLD_NEXT`; a definition in an object file does win over the import library at link time, so the same wrappers can forward through `GetProcAddress` on the loaded `amdhip64` DLL. Prove it on gfx1151 rather than assume it.
4. **`__syncwarp` mapped to `__builtin_amdgcn_wave_barrier()` alone.** That is a scheduling barrier, not a memory fence. Where PhysX uses `__syncwarp` to order shared-memory writes by one lane before reads by another, add the wavefront-scope release/acquire fences around it (the form HIP's own `__syncwarp` uses). Applies on both wavefront widths.
5. **wave64 logical-warp model.** `ballotSync` takes the subgroup half with `__lane_id() & ~31`; fine on wave32 (always 0) and wave64. Check that no kernel launches a block whose size is not a multiple of 32 and that no code derives a lane id from `__lane_id()` on one path and `threadIdx.x & 31` on another. 32-lane `WARP_SIZE` geometry must stay (shared-memory layouts and RadixSort bucket math depend on it); do not "fix" it to 64.
6. **SDF sampler parity.** The software sampler must reproduce: texel-centre convention (callers already add +0.5), clamp-to-edge on all three axes, trilinear filtering, and normalized-float reads of the unorm8/unorm16 subgrid formats (value / 255 or / 65535 before `applySubgridSdfScale`). NVIDIA hardware filtering uses 8-bit fixed-point weights, so the software result is closer to exact than CUDA's; compare against the CPU SDF query with a tolerance, not bit-exact. The linear copy must be a copy (the source is pageable mesh memory), freed in `removeGeometry`, and keep the 8-byte handle slot in the geometry blob.
7. **Floating point.** The CUDA build uses `-use_fast_math -ftz=true -prec-div=false -prec-sqrt=false`; the AMD HIP flags are `-ffp-contract=on` only. GPU-vs-CPU comparisons need tolerances; do not reach for `-ffast-math`.
8. **Fresh allocations not zeroed on ROCm.** PhysX allocates many accumulation buffers through `cuMemAlloc` and clears them with `cuMemsetD32Async`. The reviewer checks the 5.13 additions (`PxgDeformableFilterManager`, `PxgAttachmentManager`) for a read-modify-write before a memset.
9. **RDNA specifics.** LDS is 64 KB per workgroup: check the largest static `__shared__` footprints and any dynamic shared memory sizing against it. `cuOccupancyMaxActiveBlocksPerMultiprocessor` is new in 5.13; a nonzero answer on RDNA does not prove the shared memory fits.
10. **Windows.** The upstream Windows GPU path is MSVC + nvcc. HIP needs clang for `.cu`, and CMake's HIP language on Windows wants one clang driver for every language, so the Windows ROCm build likely compiles the whole SDK with clang (TheRock). `PhysXGpu_64.dll` exports C functions but passes C++ interface objects across the boundary, so both sides must share the MSVC ABI (clang-cl is compatible). Windows goes last, after both Linux gates pass at the same head.
11. **CUDA path.** Unguarded shared edits (`static __constant__` in `cudaNpCommon.h`, `PxPreprocessor.h` `__HIPCC__` addition, `PxCudaTypes.h`) must stay pure passthrough on nvcc. Compile the CUDA path with real nvcc before presenting any round.
12. **Build "just works".** INSTALL_ROCm.md requires `-DCMAKE_HIP_COMPILE_OBJECT=...-x hip...` on CMake 4 with the pip SDK, and `-DCMAKE_SUPPRESS_REGENERATION=ON` for one arch. Move such workarounds into the build files, guarded to the case that needs them; the documented command must work as written.

## File-by-file change list (on top of the merge)

- `physx/source/gpusimulationcontroller/include/PxgArticulationBlockData.h`, `physx/source/gpunarrowphase/include/cudaNpCommon.h`: extend the `__CUDA_ARCH__` guards for HIP (Risk 1).
- `physx/source/gpucommon/include/cuda_to_hip.h`: add `cuOccupancyMaxActiveBlocksPerMultiprocessor` -> `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`; give `syncWarp` wavefront fences (Risk 4); remove internal-record references.
- `physx/source/gpunarrowphase/src/CUDA/sdfCollision.cuh`: replace the trap in `pxgSampleSdfTexture3D` with a software trilinear sampler on HIP; the CUDA branch stays `tex3D`. Use the software path on every HIP architecture (one code path, validated on all three gates; RDNA's texture hardware is not needed).
- `physx/source/gpunarrowphase/src/PxgGeometryManager.cpp` (+ `include/PxgGeometryManager.h` if a field is needed): on HIP, upload each SDF grid (dense, coarse, subgrid in its stored format) to linear device memory with a small device-side descriptor (pointer, dims, format), store the descriptor's device address in the `CUtexObject` slot, synchronize as the CUDA path does, and free both in `removeGeometry`.
- `physx/source/physx/src/gpu/PxPhysXGpuModuleLoader.cpp` and the CPU SDK CMake: `PX_USE_HIP` loader branch for Linux and Windows (Risk 2).
- `physx/source/cudamanager/src/CudaKernelWrangler.cpp`, `physx/source/compiler/cmakegpu/linux/PhysXGpu.cmake`: deterministic local binding of the registration wrappers; Windows forwarding (Risk 3).
- `physx/source/compiler/cmakegpu/windows/*.cmake`, `physx/source/compiler/cmakegpu/CMakeLists.txt`: Windows HIP path (`PhysXGpu_64.dll`, link `amdhip64`, no `nvcuda.dll` delay-load).
- `physx/CMakeLists.txt` (+ presets if needed): `PX_USE_HIP` through the documented top-level build with snippets; HIP compile rule inside the build files (Risk 12).
- Parity harness (new, own commit, under `physx/snippets/`, added to the GPU snippet list): runs fixed scenes on CPU and GPU and exits nonzero on divergence beyond tolerance. Scenes: rigid stack and GPU broadphase, articulation in contact with rigid bodies (exercises `rigidDeltaAccum.cu`), SDF mesh collision with dense and each subgrid format, PBD particles against an SDF mesh, one deformable volume. Fails, not skips, when a device is present but the context is invalid; skips with a clear message when no device is present.
- `INSTALL_ROCm.md`, `README.md`: correct the version (it says 5.6), drop the CDNA SDF limitation once fixed, add RDNA and Windows, describe what was actually validated, keep it short. No README-only workarounds.

## Build commands

Linux, any arch (`<gfx>` = `gfx942`, `gfx950`, `gfx1100`, `gfx1201`; ROCm from `/opt/rocm` or the pip SDK root `$(rocm-sdk path --root)`). The porter confirms the final spelling and records it in notes.md.

```bash
cmake -S physx -B build-hip -G Ninja \
  -DPHYSX_PRESET=linux-clang -DPX_USE_HIP=ON -DPX_GENERATE_GPU_PROJECTS=ON \
  -DPX_BUILDSNIPPETS=ON -DPX_BUILDPVDRUNTIME=ON \
  -DCMAKE_BUILD_TYPE=release -DCMAKE_HIP_ARCHITECTURES=<gfx>
cmake --build build-hip -j"$MAX_JOBS"
```

Also build the `checked` configuration once per platform (asserts on) for the GPU snippets and the harness. The standalone GPU-only configure from INSTALL_ROCm.md (`cmake -S physx/source/compiler/cmakegpu -DPX_USE_HIP=ON ...`) must keep working.

CUDA no-regression (nvcc fleet host): the same top-level configure with `-DPX_USE_HIP=OFF` and the host's nvcc, building `PhysXGpu` and the snippets.

Windows gfx1151: TheRock ROCm, clang throughout, `-DCMAKE_HIP_ARCHITECTURES=gfx1151`; recipe to be worked out by the porter on the Windows round.

## Test plan

PhysX's public unit tests (`physx/test/public`, behind `PX_BUILDUNITTESTS`) are not in the open-source tree, so the suite is the snippets, pvddom's CTest set, and the authored parity harness.

GPU (each platform):

1. All 14 GPU snippets (`GPU_SNIPPET_LIST`: HelloGRB, DeformableSurface, DeformableVolume, DeformableVolumeAttachment, DeformableVolumeKinematic, DeformableSurfaceSkinning, DeformableVolumeSkinning, SDF, PBF, PBFMultiMat, Isosurface, OmniPvdMultiGeom, RBDirectGPUAPI, DirectGPUAPIArticulation) exit 0 in release and checked builds, with no "Failed to load", "Failed to initialize cuda context" or "incompatible" message, and with GPU kernel dispatches counted (`rocprofv3 --kernel-trace`; a run with zero dispatches is a failure, not a pass). Snippets assert nothing, so this proves the GPU path runs, not that it is right.
2. The parity harness passes (correctness). SnippetSDF and the harness's SDF scenes on a CDNA host (gfx942 or gfx950) are required by the ruling: SDF collision must work there with no trap.
3. `ctest` in the build tree for pvddom, including `ovd_sdk_gpu_capture`, which must report PASS, not the "SKIP: GPU unavailable" skip.
4. One snippet run with the HIP runtime preloaded (`LD_PRELOAD=<rocm>/lib/libamdhip64.so`) to prove the registration binding (Risk 3).

Non-GPU regression set (must not regress): every non-GPU snippet exits 0 in the HIP build; the pvddom CTest set; a `PX_GENERATE_GPU_PROJECTS=OFF` (cpu-only preset) build still configures and builds without any ROCm or CUDA toolkit.

Gates: wave64 (gfx942/gfx950), wave32 (gfx1100/gfx1201), windows (gfx1151, last). Plus the nvcc compile of the CUDA path before each round is presented.

## Open questions

1. **ovphysx scope (blocks the porter).** See notes.md, Planning 2026-11-20, Question 1. Until ruled, its 70 components sit under `owed` in surface.json, so nothing is cut silently and `pr_ready` refuses.
2. **Licence record (does not block).** `physx/` moved from BSD-3-Clause to Apache-2.0 in 5.13; root `LICENSE.md` is still BSD-3. The AMD-added files carry BSD-3 headers. Updating `license_spdx` and choosing the header for new files are a person's calls.
3. **Parity harness delivery (does not block).** Planned on `moat-port` as its own commit so it ships with the port and can be dropped from an upstream PR if Jeff prefers.
