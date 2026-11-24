# PhysX notes

## Intake screen (2026-11-06)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: partial, AMD fork exists).

### Licence
`utils/licenses.py check NVIDIA-Omniverse/PhysX`: BSD-3-Clause, tier 1. LICENSE.md read by hand: "BSD 3-Clause License, Copyright (c) 2008-2025, NVIDIA Corporation". No per-file `scan-nvidia` run and no clone (the repo is very large and the recommendation is a decline); run the scan before any take-up.

### Existing AMD support
- `AMD-Ecosystem/PhysX` (GitHub fork of NVIDIA-Omniverse/PhysX, AMD-Ecosystem org) has a `rocm-enablement` branch with one commit, d09e25323d, 2026-09-16, "[ROCm] AMD Instinct (ROCm) enablement for PhysX", signed off by three AMD engineers (amd.com addresses). Its `main` is an unmodified mirror. Not a MOAT project (no MOAT records; `scaffold` accepted the name).
- Scope: 47 files. PhysX 5 GPU simulation through a `cuda_to_hip.h` shim plus `hip_compat/` header stubs: CMake for every GPU module (articulation, broadphase, common, narrowphase, simulation controller, solver, CUDA context manager), `CudaKernelWrangler.cpp` (+204), warp helpers, atomics, SDF collision, and an `INSTALL_ROCm.md`. Built and tested on MI300X (gfx942) and MI350X/MI355X (gfx950), ROCm 10.0.0, Ubuntu 24.04; only the Linux GPU CMake path is touched.
- Upstream: no ROCm/HIP/AMD PRs on NVIDIA-Omniverse/PhysX; the branch has not been offered upstream and is 3 commits behind upstream main (ovphysx updates).

### Gaps left by the AMD fork (for the record, not a reason to compete)
- Instinct-only, Linux-only: no wave32 (RDNA) or Windows validation, and the Windows GPU CMake path is untouched.
- Header-shim approach (`hip_compat/` stubs named like CUDA headers) rather than hipify-in-place; footprint not reviewed.

### Recommendation
Decline, reason `ported-elsewhere`: an AMD-owned fork (AMD-Ecosystem/PhysX, `rocm-enablement`) already ports PhysX 5 GPU simulation and validates it on MI300X/MI355X. RDNA wave32 and Windows coverage are differentiators to contribute to that branch in coordination with its owners, not a parallel MOAT port.

### Question for jeff
AMD-Ecosystem/PhysX is in our own org but is not a MOAT project. Should MOAT adopt it as an existing fork (validate wave32 and Windows on the `rocm-enablement` branch) in coordination with its authors, or leave it to them?

## Take-up ruling (2026-11-10)

Jeff Daily ruled on 2026-11-10 that MOAT takes PhysX up, building on the AMD team's existing ROCm work, overriding intake's decline (ported-elsewhere) recommendation:

- Fork: https://github.com/AMD-Ecosystem/PhysX. `main` is the upstream mirror and the recorded `fork_default_branch`. `rocm-enablement` (d09e25323d8eea94117a74d7c2bd25309da527d5, one squashed commit "[ROCm] AMD Instinct (ROCm) enablement for PhysX", 2026-09-16, 47 files) is the AMD team's Instinct/Linux port. Both branches stay as they are.
- Port base: the porter creates `moat-port` from the `rocm-enablement` tip, keeping that commit and its authorship, then merges upstream NVIDIA-Omniverse/PhysX `main` (3 commits newer at the time of the ruling). Our work goes on top as separate commits. Unlike MONAI, this is not an independent port: PhysX has no upstream ROCm support and Jeff judged the diff close to what MOAT would produce with hip_compat headers.
- Clean-up in scope: the rocm-enablement commit carries many whitespace-only changes. Revert them on `moat-port` (a separate commit on top), so the delta is the ROCm change alone. The reviewer should check for any other noise of the same kind.
- In scope, not deferred: the CDNA SDF limitation. INSTALL_ROCm.md, "Known limitation on CDNA (gfx942/gfx950)", says SDF mesh collision traps on CDNA because there is no hardware 3D-texture sampler (`tex3D`), and names "a software-trilinear sampler over a linear device copy of the grid" as the planned follow-up. Jeff ruled that follow-up part of this port: implement it, so SDF collision works on gfx942/gfx950 and the trap goes away, with tests that exercise SDF collision on a CDNA host.
- Also in scope: Radeon (wave32, gfx1100/gfx1201) and Windows gfx1151, with every gate validated (Linux wave64 and wave32 first, Windows last).
- Delivery vehicle (PR into the AMD fork's rocm-enablement, a handoff to the AMD team, or an upstream PR) is Jeff's call later.

Recorded: `set-intake PhysX fork`, `set-state PhysX linux-gfx942 awaiting-fork`, `release-forks` (fork exists, stage `screened`, `fork_url` set). No `fork_url_ruling` block yet: `set-fork-url` needs `moat-port` to exist on the fork; re-run `set-fork-url PhysX https://github.com/AMD-Ecosystem/PhysX --by "Jeff Daily"` once the porter has pushed it.

## Planning (2026-11-20, linux-gfx1100, planner)

### Questions for Jeff (Question 1 must be answered before a porter starts)

**Question 1 (blocking): is ovphysx part of this port?** The repository is no longer just the SDK. At upstream 5.13 it holds `physx/` (the SDK; all GPU simulation code), `ovphysx/` (NVIDIA's USD-native Python/C product built on the SDK), `blast/` and `flow/`. blast and flow have no CUDA (flow is D3D12/Vulkan shaders), so they need nothing. ovphysx does have CUDA of its own: 4 `.cu` files (`ovruntime/.../particles/TypeConversionUtility.cu`, `ovruntime/.../tensors/gpu/CudaKernels.cu`, `src/ovphysx/OvStageOutputCuda.cu`), about 30 host files calling the CUDA runtime, and Python output as Warp arrays. It builds against NVIDIA Carbonite and ovstage binaries that packman fetches, and `licenses.py scan-nvidia` finds two NVIDIA-proprietary licence texts in it (`ovphysx/licenses/LICENSE-binary.txt`, `ovphysx/tools/internal-licenses/Omniverse-Extensions-LICENSE.txt`). The AMD rocm-enablement branch does not touch it.
- Option A (planner's recommendation): port the `physx/` SDK only (CPU SDK, the GPU library, snippets, pvddom), and record ovphysx as deferred (`deferred.py add` against PhysX) for a later round once the SDK port is validated. Reason: ovphysx is a separate product with closed binary dependencies and a Warp dependency; the SDK port is the prerequisite for it either way.
- Option B: include ovphysx now. That adds `depends_on: [warp]` (MOAT's `warp` project), a check that the Carbonite/ovstage packman binaries can be used in a ROCm build at all, a ROCm path for its 4 `.cu` files and CUDA-runtime host code, and its C/Python test suites (c_unittests, c_samples, python tests, benchmarks) to the validation bar.
- Option C: some other cut (for example ovphysx's GPU tensor path only).
Until ruled, the 70 ovphysx components are `owed` in surface.json (in scope, not done), so nothing is cut silently and `pr_ready` refuses. The SDK work in plan.md is needed under every option, so a porter dispatched before the ruling would not waste work, but the plan does not assume the cut. A hold (`set-hold PhysX on`) is a person's action, so this run did not set one.

**Question 2 (does not block): licence record.** Upstream relicensed between the AMD base and today: every `physx/` source file now says `SPDX-License-Identifier: Apache-2.0` and `physx/LICENSE.md` is Apache 2.0; the root `LICENSE.md` is still BSD-3-Clause, which is what `licenses.py check` (tier 1, cleared) and `license_spdx` say. Should the record move to Apache-2.0 (`set-license-spdx`)? Separately, the AMD-added files (`cuda_to_hip.h`, `hip_compat/*`, `cutil_math.h`) carry BSD-3 headers with an AMD "Modifications Copyright" line; whether they switch to Apache-2.0 SPDX headers before any delivery is a call for you and the AMD authors.

**Question 3 (does not block): parity harness.** PhysX's unit tests are not in the open-source tree and the snippets assert nothing, so plan.md has the porter author a GPU-vs-CPU parity harness under `physx/snippets/`, as its own commit on `moat-port`. It ships with the port; whether it goes into an upstream PR or a handoff is your call at delivery.

### What this run did
- `orient.sh` on `port/PhysX`: next = PhysX, state screened, planner. Took the lock: `set-state PhysX linux-gfx1100 planning --agent planner`. `set-ext-type PhysX cmake`.
- Clones: `projects/PhysX/src` = fork with `upstream` remote, detached at upstream `main` `7d6f6ac8` (PhysX 5.13.0), `GIT_LFS_SKIP_SMUDGE=1` (the repository uses LFS for ovphysx media; a merge without it fails in the smudge filter).
- Trial merge, read only: `git merge-tree --write-tree --name-only d09e2532 7d6f6ac8` -> 24 conflicts; 21 are the licence header block only, 3 carry content (`physx/CMakeLists.txt`, `cmakegpu/linux/CMakeLists.txt`, `PxgGeometryManager.cpp`). With `-X ignore-all-space`: 23.
- `python3 utils/surface.py generate PhysX` against `7d6f6ac8`: 162 CUDA files, 131 components; planner added 21 (driver-API seam, registration seam, SDK GPU loader, SDF sampler, Windows GPU CMake, top-level `PX_USE_HIP` build, parity harness, 14 GPU snippets), 70 ovphysx components owed, `cuda-library:thrust` removed from the floor (vehicle "thrust" variables). `surface.py check PhysX`: fully accounted for.
- `licenses.py scan-nvidia projects/PhysX/src` (owed since intake): 2 hits, both licence texts inside `ovphysx/` (Question 1). Nothing under `physx/`.

### Findings that change the port (details in plan.md)
- Upstream moved from 5.10.0 to 5.13.0 since the AMD base (4 commits, 5240 files), not the 3 small ovphysx commits the ruling saw. No new kernel TUs (61 `.cu`, 495 kernels); one new driver call (`cuOccupancyMaxActiveBlocksPerMultiprocessor`) the shim lacks.
- Two silent no-ops in the AMD branch: `atomicAddSpatialVector` (`PxgArticulationBlockData.h:188`, used by `rigidDeltaAccum.cu`) and the narrowphase overflow `atomicOr` (`cudaNpCommon.h:107`) sit under `__CUDA_ARCH__` gates that HIP leaves undefined.
- The SDK's GPU module loader (`PxPhysXGpuModuleLoader.cpp`) requires `libcuda.so.1` before it opens `libPhysXGpu_64.so`, and the AMD branch did not change it, so no SDK-level GPU simulation can have run on that branch; snippets fall back to the CPU silently. Its evidence (fatbin/ldd/objdump checks plus a load-and-dispatch ladder) is consistent with that.
- The AMD SDF change compiles the texture path out on every HIP target, RDNA included, not only CDNA. The ruled software sampler will be the single HIP path.
- `__hipRegisterFatBinary` interposition can lose to the runtime's own symbol when the HIP runtime is already loaded globally; Windows needs its own forwarding.
- `__syncwarp` maps to a bare `wave_barrier` with no memory fence.
- The `bar.sync N` -> `__syncthreads()` mapping is exact (upstream's comments say each is a whole-block barrier).

## Rulings on the planner's questions (Jeff Daily, 2026-11-23)

- Question 1 (ovphysx): **A**. Port the `physx/` SDK only (CPU SDK, the GPU library, snippets, pvddom); ovphysx is deferred for a later round and its components are recorded `scoped_out` for this round with a registered deferral (`deferred.py add` against PhysX).
- Question 2 (licence record): moved to `Apache-2.0 AND BSD-3-Clause` (every `physx/` source file and `physx/LICENSE.md` are Apache-2.0; the root `LICENSE.md` is BSD-3-Clause). The headers of the AMD-added files are left as they are.
- Question 3 (parity harness): **keep it on the fork; do not send it upstream** (ruled later the same day).
- Standing ruling of the same day: Windows laptop GPU runs go last across all ports; they are queued, not started.

## Porter round 2026-11-24 (linux-gfx942, MI300X, headless fleet job)

### Questions for Jeff (none blocks the reviewer)

**Question A: the fork's `main` mirror is still at upstream 5.10.** `git ls-remote https://github.com/AMD-Ecosystem/PhysX main` is `4f2103c3` (5.10.0); upstream `main` is `7d6f6ac8` (5.13.0), and `moat-port` merges it. Anything that diffs `moat-port` against the recorded `fork_default_branch` (`main`) therefore also shows NVIDIA's four upstream commits (5240 files): `jargon.py --port PhysX` reports 14 hits, every one in upstream's own ovphysx text, and a fork review PR into `main` would carry the same noise. Our delta measured against upstream is clean (`jargon.py -C projects/PhysX/src --diff 7d6f6ac8..moat-port` and `--commits 4f2103c3..moat-port`: clean). Options: (a) a person fast-forwards the fork's `main` to `7d6f6ac8` (a plain fast-forward of the mirror, which no agent may push); (b) leave it and have the reviewer diff against `upstream/main`; (c) another base ruling. Recommendation: (a), before any fork review PR. This job pushed nothing to `main`.

### What this round did

- Re-entered the stale lock (`set-state PhysX linux-gfx942 porting`, WARNING re-stamp; the killed job mi300c-20261123-152800 was gone after the host reboot).
- Salvage: the killed job had left an unpushed 17-commit `moat-port` in its own clone. Imported it as a local branch, reviewed every commit against upstream `7d6f6ac8` and the AMD base, and kept it only after redoing every build and run below with this job's own toolchain. Nothing of its evidence is used. Checks made on the salvage: the merge commit differs from git's own auto-merge (`git merge-tree --write-tree d09e2532 7d6f6ac8`) only in the 24 conflicted files, and each resolution keeps upstream's SPDX header with AMD's "Modifications Copyright" line under it, plus the AMD HIP blocks; the whitespace-restore commit is empty under `git diff -w`.
- Corrected before the first push (nothing had been published, no arch had validated anything, so rewriting was safe): the atomics commit message overstated the articulation fix (see Findings); the parity snippet's no-GPU skip was a false pass (see Findings), and its message, the docs commit's "tested" line and two new harness scenes were folded into their commits. Tree of every rewrite checked against the commit it replaced.
- New commits: `Build the BV32 root lane masks without a shift by 32` (shared code, no prefix) and `[ROCm] Fix a mismatched ENDIF in the Linux GPU library CMake`.
- Fork `moat-port` pushed to `AMD-Ecosystem/PhysX` at `4764ca139605` (plain push with `GIT_LFS_SKIP_PUSH=1`: the branch carries upstream's ovphysx LFS pointers, whose objects live in upstream's LFS store; a clone of the fork needs `GIT_LFS_SKIP_SMUDGE=1` as the planner noted). Base: `d09e2532` (AMD, authorship kept) + merge of upstream `7d6f6ac8` (merge, not rebase).
- Surface: the 70 ovphysx components moved from `owed` to `scoped_out` citing the 2026-11-23 ruling (Question 1 = A); deferral `physx-ovphysx-port` registered with `deferred.py add feature-port --project PhysX`. `option:PX_BUILDUNITTESTS` scoped out (the public unit tests are not in the open tree; configure stops with "This source distribution does not include the public unit tests"). Three Windows items stay `owed`. Everything else `covered`, each with a note naming its evidence. `surface.py check PhysX`: fully accounted for.
- `set-fork-url` (mentioned in the take-up notes) was not run: it records a port living outside the organization fork, and `fork_url` already is `AMD-Ecosystem/PhysX`.

### Commit list on `moat-port` (oldest first, above `d09e2532`)

1. Merge upstream main (PhysX 5.13.0) into the ROCm port
2. [ROCm] Fix the HIP branch of the GPU language selection (`elif()` from the merge resolution; CMake rejected it)
3. [ROCm] Restore upstream whitespace in files touched by the ROCm port
4. [ROCm] Drop references to records that do not exist upstream
5. [ROCm] Map cuOccupancyMaxActiveBlocksPerMultiprocessor to HIP
6. [ROCm] Let CMake pass the HIP offload architectures (dropped the hand-written `--offload-arch`, dedups auto-detected arches)
7. [ROCm] Build the snippets against the ROCm GPU stack (`linux-gcc-rocm` preset, `USE_CUDA_INTEROP=0`)
8. [ROCm] Load the ROCm GPU library without libcuda.so.1 (plan Risk 2)
9. [ROCm] Bind the HIP registration wrappers inside the GPU library (protected visibility; plan Risk 3)
10. [ROCm] Keep two device atomics that were compiled out for HIP (plan Risk 1)
11. [ROCm] Fix the radix sort bucket shifts on AMD GPUs (found by the earlier job: memory fault on the first GPU frame)
12. [ROCm] Make __syncwarp a memory fence as well as a barrier (plan Risk 4)
13. [ROCm] Sample SDF grids in software instead of trapping (plan Risk 6; single HIP path on every AMD GPU)
14. Add a GPU-versus-CPU parity snippet (`physx/snippets/snippetgpuparity`, own commit; stays on the fork by the 2026-11-23 ruling)
15. [ROCm] Do not require the CUDA toolkit to use a ROCm PhysX package (`PhysXConfig.cmake.in`; pvddom's `ovd_sdk_package_coexistence` caught it)
16. [ROCm] Find the HIP compiler without ROCM_PATH (`rocm-sdk path --root`, then `/opt/rocm`; plan Risk 12)
17. [ROCm] Document the ROCm build as it now works (INSTALL_ROCm.md rewritten, README pointer, README_LINUX preset line)
18. Build the BV32 root lane masks without a shift by 32
19. [ROCm] Fix a mismatched ENDIF in the Linux GPU library CMake

### Toolchain (verified, not assumed)

- TheRock ROCm 10.2 from the nightly index, into the job venv only: `$FLEET_VENV/bin/python -m pip install --index-url https://nightly.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel]==10.2.0a20261022" "rocm-sdk-device-gfx942==10.2.0a20261022"`, then `rocm-sdk path --root` once to expand the devel payload.
- Root `$FLEET_VENV/lib/python3.11/site-packages/_rocm_sdk_devel`, `.info/version` = `10.2.0`, HIP 7.17.26422, AMD clang 24.0.0git (`2792f8ceeb5a`).
- Compiler the build actually used: `CMakeCache.txt` `CMAKE_HIP_COMPILER=.../venv/lib/python3.11/site-packages/_rocm_sdk_devel/lib/llvm/bin/amdclang++` (host C/C++: `/usr/bin/cc`, `/usr/bin/c++`, gcc 13.3.0). `ldd libPhysXGpu_64.so` resolves `libamdhip64.so.7` from the same `_rocm_sdk_devel/lib`. The image's ROCm 10.1 was never on the compile path.
- Snippet dependencies built from source into the job directory (`deps/prefix`, `CMAKE_PREFIX_PATH`): libglvnd v1.7.0 (meson, `-Degl=false -Dgles1=false -Dgles2=false -Dheaders=true`), GLU glu-9.0.3 (meson, `-Dgl_provider=glvnd`), freeglut v3.6.0 (CMake, demos and static off). meson came from the job venv. Packman cache: `PM_PACKAGES_ROOT=<job>/deps/packman-repo`.
- GPU runs: `HIP_VISIBLE_DEVICES=1` (the dispatcher's pin), `ulimit -c 0`, `timeout` around every run.

### Build recipe (as documented in INSTALL_ROCm.md, run as written)

```
cd physx
./generate_projects.sh linux-gcc-rocm
make -C compiler/linux-gcc-rocm-release -j192
make -C compiler/linux-gcc-rocm-checked -j192
```

Release and checked both build with 61 HIP objects (one per `.cu`) at the round head. Also built: the direct configure from INSTALL_ROCm.md with `ROCM_PATH` and `HIP_PATH` unset (`cmake -S physx -B build -DPHYSX_PRESET=linux-gcc -DPX_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx942;gfx1100" -DCMAKE_BUILD_TYPE=release`; compiler found through `rocm-sdk`; `llvm-objdump --offloading` shows 61 gfx942 and 61 gfx1100 code objects), the standalone GPU-library configure from INSTALL_ROCm.md, `PX_GENERATE_GPU_PROJECTS_ONLY=ON`, `PX_GENERATE_GPU_STATIC_LIBRARIES=ON` with `PX_GENERATE_STATIC_LIBRARIES=OFF` (`libPhysXGpu_static_64.a`; upstream itself refuses the static/static pair; no in-tree consumer runs the static GPU library), the FetchContent example (`physx/documentation/fetchcontent/examples/fetchcontent`, `-DFETCHCONTENT_SOURCE_DIR_PHYSX_LIB=<fork> -DPX_USE_HIP=ON`: builds, runs, prints "GPU acceleration is available"), and the CPU-only preset (`-DPHYSX_PRESET=linux-gcc-cpu-only`, PATH limited to `/usr/bin:/bin`, no ROCm or CUDA visible: configures and builds, 483/483 steps).

Snippets render through GLUT, and this host has no display or GL driver, so for the runs each snippet is relinked from the build's own compile commands with `-URENDER_SNIPPET` (upstream's non-render path, fixed step count); helper scripts are in the job's `agent_space` (`headless_snippets.py`, `run_snippets.sh`). Every GPU snippet run is under `rocprofv3 --kernel-trace`, and zero dispatches counts as a failure.

### Results at the round head (MI300X, gfx942, ROCm 10.2)

All at fork head `4764ca13`, release and checked built from scratch at that head.

- **SnippetGpuParity: 29/29 checks pass** in release, in checked, and in release with `LD_PRELOAD=<rocm>/lib/libamdhip64.so`; 559,794 kernel dispatches in the traced release run. Release GPU-vs-CPU differences: free fall 4.0e-4 m (limit 1e-3), stacks 9.5e-4 m (limit 0.01), articulation 4.7e-6 m, SDF mesh dense/8/16/32-bit 2.1e-4 / 7.5e-5 / 1.0e-3 / 8.7e-4 m (limit 5e-3), large SDF mesh 1.7e-3 m; rest-height errors at most 1.0e-4 m; particles on SDF lowest 1.083 m (table top 1.0); deformable lowest vertex -6.7e-8 m; lever extra bend 0.19 rad (static estimate 0.18); contact overflow reported 20 times.
- **GPU snippets (traced with rocprofv3 --kernel-trace, built without rendering):** release 13/13 exit 0, each with 18k to 117k dispatches; checked 13/13 exit 0 plus OmniPvdMultiGeom (checked only, upstream drops it from release; needs `--omnipvdfile=<path>`): exit 0, 443,294 dispatches, 120 frames written. SnippetSDF as shipped runs on the CPU (`gUseGPU = false` upstream; exit 0, 0 dispatches, expected); a local copy with `gUseGPU = true` (100 SDF instances) exits 0 with 53,939 dispatches including `triangleTriangleCollision`, the SDF path in `convexMeshMidphase.cu`. No run printed "Failed to load", "Failed to initialize cuda context" or "incompatible".
- **pvddom workflow CTest** (`cmake -S physx/pvddom/workflow -B capture-build -G Ninja -DCMAKE_BUILD_TYPE=checked -DPX_GENERATE_GPU_PROJECTS=ON -DPX_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942`; `ctest -C checked`): 5/5 pass; `ovd_sdk_gpu_capture` is a PASS with `GPU=ON` ("Independent captures equal ... GPU=ON"), not the SKIP.
- **pvddom unit CTest** (`cmake -S physx/pvddom -DPVDRUNTIME_ROOT=physx/pvdruntime -DPVDDOM_BUILD_TESTS=ON -DPVDDOM_BUILD_INSPECTION=ON -DPVDDOM_BUILD_TOOLS=ON`): 372/372 pass (includes `ovd_cli_installed`, which drives the consumer tests).
- **Non-GPU snippets (release, built without rendering):** 55/56 exit 0. The seven Vehicle snippets need `--vehicleDataPath=physx/snippets/media/vehicledata` and LoadCollection needs `--generateExampleFiles` then file arguments; with those they pass. SnippetProfilerConverter only prints its usage: it needs a raw profiler capture as input; not exercised (CPU tool, untouched by the port).
- **CPU-only build** with no ROCm or CUDA on PATH: builds.

### Teeth tests (run during the round on the head of the day; each a temporary revert, rebuild, run, restore, with the GPU library rebuilt from the restored tree afterwards)

| Revert | Harness result |
|---|---|
| SDF sampler returns 0 | 10 checks fail (all 8 SDF mesh checks, 2 particle-on-SDF checks) |
| radix shift guard removed | free fall GPU-vs-CPU 41.6 m off, then SIGSEGV (exit 139) |
| `setOverflowError` back under `__CUDA_ARCH__` only | contact overflow check fails (0 reports, 20 with the fix) |
| registration wrappers back to default visibility, run with `LD_PRELOAD=<rocm>/lib/libamdhip64.so` | "Could not find registered CUDA function ..." for every kernel, exit 1; without the preload it still passes, which is exactly the plan's Risk 3 |
| loader back to `libcuda.so.1` | "Could not load libcuda.so.1", GPU context check fails, exit 1 |
| `atomicAddSpatialVector` back under `__CUDA_ARCH__` only | still passes; no teeth possible (see Findings) |

The `__syncwarp` fence has no teeth test: lockstep wavefronts make the missing fence a compiler-ordering hazard that a fixed scene does not reliably expose.

### Findings

- **The articulation atomic fix is latent in the open tree.** `atomicAddSpatialVector`'s only caller is `accumulateDeltaVRigidSecondLaunchMultiStage2` (`rigidDeltaAccum.cu:641`), whose kernel id `ACCUMULATE_DELTAVEL_RIGIDBODY_SECOND_MULTI2` no host `.cpp` launches (the particle and FEM cores launch `..._FIRST` and `..._SECOND`). No traced run dispatched it. The fix stays (it is correct and the kernel is compiled and exported); the commit message now says it is latent. The overflow half of that commit is reachable and has teeth.
- **The salvaged harness's "SKIP: no GPU found" was a false pass.** `PhysXDeviceSettings::getSuggestedCudaDeviceOrdinal` returns `PHYSX_GPU_DEVICE` or 0 and never probes for a device, so `PxGetSuggestedCudaDeviceOrdinal` is negative only when the GPU library failed to load. With the loader reverted, the salvaged harness printed SKIP and exited 0. The harness now has no skip path: it fails whenever it cannot create a valid GPU context (checked with the loader reverted and with `HIP_VISIBLE_DEVICES=-1`, both exit 1). This deviates from the plan's "skips with a clear message when no device is present"; there is no portable device probe at that level. Reviewer: judge whether that is acceptable for a harness that stays on the fork.
- **BV32 root mask, a second shift-by-32.** `triangleTriangleOverlaps` (`convexMeshMidphase.cu:1443/1447`) builds `(1 << mNbNodes) - 1`, and a BV32 root can have 32 children. A temporary device `printf` showed `nb0 32 nb1 32 mask0 0 mask1 0` for the harness's large-mesh scene on the MI300X. The overlap pre-test then runs GJK on an empty (infinite) box, which today happens to report an overlap, so only the early-out is lost and the scene passed before the fix; the outcome rests on undefined behaviour. Fixed the way `reduction.cuh:503` already guards its element mask. The large-mesh scene stays in the harness as coverage of the full-root path; it has no teeth for this fix.
- **Sweeps.** Every `__CUDA_ARCH__`/`__CUDACC__`/`__CUDACC_VER_MAJOR__` gate in `physx/` besides the two fixed falls back to correct generic code (`PxgIntrinsics.h`, `PxMath.h`, `particlesystem.cu`, `broadphase.cu:1505`, `atomic.cuh`). Masks built from counts: `reduction.cuh:503` is already guarded; the polygon-clip masks in `cudaGJKEPA.cu` and `convexTriangle.cuh` (`(1 << nbVerts0) - 1`, `(1 << nbContacts) - 1`) depend on face vertex and contact counts staying below 32, which was not proven; reviewer item.
- **AMD-base noise left for the reviewer.** `d09e2532` added a root `.gitignore` (`build-hip/`, `physx/build-hip-test/`, `physx/lib/`) that upstream does not have.
- **Build warnings.** 2241 compiler warnings in the release build, mostly `-Wdeprecated-declarations` (hipCtx* in the compatibility header), `-Wunused-value`, `-Wunused-result`, `-Wattributes`, `-Wmacro-redefined`; none is an error and none was introduced by this round.

### Owed (not done here, exact state)

- **CUDA compile check: run elsewhere.** This host has no nvcc. The CUDA path must be compiled with real nvcc on the mi355 fleet host before the round is presented to Jeff: top-level configure with `-DPX_USE_HIP=OFF`, building `PhysXGpu` and the snippets. Shared edits that nvcc will see: `PxPreprocessor.h` (`__HIPCC__`), `PxCudaTypes.h`, `static __constant__` in `cudaNpCommon.h`/`epa.cuh`/`vector.cuh`/`marchingCubesTables.cuh`, `RadixSort.cuh` helpers (plain shifts on CUDA), the BV32 mask (same value on CUDA), `#ifndef USE_CUDA_INTEROP` in two snippet renderers, `PhysXConfig.cmake.in`, `compiler/public/CMakeLists.txt`, `SnippetGpuParity.cpp`.
- **wave32 (gfx1100/gfx1201):** compiles (61 gfx1100 code objects above); not run. The validator on a wave32 host owns it.
- **Windows (gfx1151):** nothing written. `build:cmakegpu-windows-hip` and the two Windows-only pvddom tests stay `owed`. What the Windows round needs (plan Risk 3 and 10): `cmakegpu/windows/*.cmake` HIP path producing `PhysXGpu_64.dll` linked to `amdhip64`, no `nvcuda.dll` delay-load, `PxPhysXGpuModuleLoader.cpp` Windows branch under `PX_USE_HIP`, and the registration wrappers forwarding through `GetProcAddress` on the loaded `amdhip64` DLL instead of `dlsym(RTLD_NEXT)`. Laptop runs go last by the standing ruling.

## Lesson for cuda-to-rocm: masks and counters that rely on a 32-bit shift by 32 being 0

Belongs in `references/faults-numerics.md`, beside the 64-bit shift entry (and cross-referenced from the 32-bit mask entry in `faults-wavefront.md`). Two forms in PhysX that are not lane-index masks: (1) a data-derived shift amount in packed counters, `1u << ((radix - first) << 4)` for radix values outside the current group, where the code relies on the result being 0 (NVIDIA's `shl` clamps the count, AMD uses its low five bits); on ROCm the counters picked up bogus increments and the ranks ran past the buffers (memory fault on the first GPU frame); (2) a lane mask from a count that can be exactly 32, `(1 << n) - 1` with `n` the number of children of a 32-wide BVH node: NVIDIA gives all 32 lanes, AMD gives 0. Grep for `<<` and `>>` whose count is computed from data or from a count that can reach the word width, not only from a lane index, and spell out the intended result (`n < 32 ? (1u << n) - 1 : ~0u`). Form (2) can be silent: in PhysX it fed a bounding-box reduction over no lanes, and the downstream GJK happened to report overlap on the resulting infinite box.

## Lesson for cuda-to-rocm: a library that interposes the HIP registration entry points must bind them locally

Belongs in `references/build-cmake-link.md`. A shared library that defines `__hipRegisterFatBinary`/`__hipRegisterFunction`/`__hipRegisterVar`/`__hipUnregisterFatBinary` to capture its own code objects (for a driver-API module table) and forwards to the runtime with `dlsym(RTLD_NEXT)` works only while nothing else made `libamdhip64` global first: with default visibility its module constructors' calls are resolved through the global scope at load time, so in a host process that already uses HIP (or under `LD_PRELOAD=libamdhip64.so`), the runtime's definitions win and the library's table stays empty. Give the wrappers `__attribute__((visibility("protected")))` so the static linker binds the calls inside the library; `readelf -rW lib.so | grep __hipRegister` should then show no relocations. Test it by running once with the runtime preloaded. (PhysX: every kernel lookup failed under the preload until the wrappers were protected.)
