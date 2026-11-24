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

## Review 2026-11-24 (linux-gfx942, MI300X, reviewer, headless fleet job)

Scope: `git diff 7d6f6ac8...4764ca13` (the whole delta against upstream 5.13: `d09e2532` plus the merge plus this round's 18 commits), read in full, with the code around every substantial hunk. The fork is not behind: `git rev-list --count moat-port..origin/main` = 0 and `..upstream/main` = 0. `jargon.py -C projects/PhysX/src --diff 7d6f6ac8..moat-port` and `--commits 4f2103c3..moat-port`: clean. Commit titles all at most 72 characters, every commit of this round discloses the agent and has a Test Plan, no Co-Authored-By trailer.

Toolchain used for the checks below: TheRock ROCm 10.2 wheels in this job's venv (`rocm[libraries,devel]==10.2.0a20261022`, `rocm-sdk-device-gfx942==10.2.0a20261022`); `rocm-sdk path --root` = `<job>/venv/lib/python3.11/site-packages/_rocm_sdk_devel`, `.info/version` 10.2.0; the compiler my builds invoked is `<root>/lib/llvm/bin/amdclang++` (AMD clang 24.0.0git, `2792f8ceeb5a`; `CMAKE_HIP_COMPILER` in the scratch build's cache, auto-discovered through `rocm-sdk`), host C++ `/usr/bin/c++` (gcc 13.3). GPU runs on `HIP_VISIBLE_DEVICES=1` under `ulimit -c 0` and `timeout`. Scratch programs are in this job's `agent_space/` (`radix/radix_test.cu`, `convex32/ConvexFaceCheck.cpp`, `convex32/MeshCheck.cpp`); the in-SDK checks link the porter's release static libraries at 4764ca13 (porter job directory, clean tree) and load either the porter's `libPhysXGpu_64.so` or one I built from a scratch worktree with only the fix in finding 1 applied (`cmake -S physx -B build -G Ninja -DPHYSX_PRESET=linux-gcc -DPX_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=release -DPX_BUILDSNIPPETS=OFF`, `cmake --build build --target PhysXGpu`, 147/147 steps).

Verdict: **changes requested**. Two GPU paths the harness never exercises are broken on the MI300X at 4764ca13.

### Port correctness

1. **The radix-sort shift fix is incomplete; the single-warp sort is still broken on AMD (this round, `d1714e8e`; the code is upstream's).** `physx/source/gpucommon/src/CUDA/RadixSort.cuh:513-516` (`radixSortPassSingleWarp`) and `:639-642` (`radixSortPassSingleWarpKeysOnly`) still build `accum0..3 = 1u << (radixRank << 3)` with `radixRank = radix - i`, which is 4 or more (or wrapped) for every key outside the current group of four buckets, and for the `0xff` padding keys. On AMD the counters pick up bogus increments, so the output offsets, the run sums (`:567-571`, `:689-693`) and the output order are wrong and writes can land past the bucket. Callers: `radixSortSingleWarp` from `gpubroadphase/src/CUDA/aggregate.cu:302` (aggregate broadphase) and `radixSortSingleWarpKeysOnly` from `gpunarrowphase/src/CUDA/convexMeshPostProcess.cu:624` (`sortTriangleIndices`, every convex/primitive-vs-triangle-mesh pair). Evidence on the MI300X: a standalone kernel calling `radixSortSingleWarpKeysOnly<1>` on random 8-bit keys (2 passes, block 32x1) returns 32/32, 99/100, 126/128, 396/400 keys out of place; the same test with those 8 shifts routed through `radixShiftLeft` sorts all of them. In the SDK, 12 boxes in one self-colliding `PxAggregate` dropped on a plane: CPU top box y = 0.500, GPU y = -4.616 (the stack falls through the ground); with the GPU library rebuilt with only those 8 shifts guarded, GPU y = 0.500. Fix: use `radixShiftLeft` at those 8 sites (the guarded `val += (1 << ...)` and `val >> ...` lines below them run only for `radixRank < 4` and are fine).

2. **Every primitive falls through a static triangle mesh on the GPU (origin not yet attributed).** `agent_space/convex32/MeshCheck.cpp`: a box, a convex cube, a sphere and a capsule (0.25 m) dropped onto a static triangle-mesh grid (1x1, 8x8 and 40x40 quads over 10 m): CPU rests at y = 0.2500 in all 12 cases; GPU ends at y = -19.16 (box, convex) and -19.34 to -19.38 (sphere, capsule) in all 12, with the porter's GPU library and with the one carrying the fix of finding 1, so this is a separate defect. 8, 16 and 31-sided convex prisms on a 2-triangle mesh fall through the same way, and PhysX prints `PxsContext::fillManagerTouchEvents: touch event bitmap contains more events than cached counters indicated` once per GPU mesh scene. The same prisms on a static box rest correctly. A kernel trace (`rocprofv3 --kernel-trace --stats`) shows the pipeline dispatching (`midphaseGeneratePairs` 126, `convexTrimeshNarrowphase` 72, `sphereTrimeshNarrowphase` 54, `sortTriangleIndices` 126, `convexTrimeshCorrelate`/`PostProcess`/`FinishContacts` 126 each), so the contacts are produced wrong or dropped, not skipped. The porter must find the cause (candidates on the path: the midphase in `convexMeshMidphase.cu` including the `bar.sync N` to `__syncthreads()` mapping, the convex/sphere-trimesh narrowphase, `convexMeshPostProcess.cu`), fix it, and say which commit introduced it. This is the most common GPU rigid-body case (anything on a mesh terrain) and nothing in the round's evidence covers it: the harness's mesh scenes are SDF mesh-vs-mesh (`triangleTriangleCollision`), which work.

3. **Masks built from a count that can be 32, outside RadixSort and BV32 (upstream code; the porter's sweep listed only the polygon-clip ones, as unproven).** Each is `(1 << n) - 1` (or a ballot ANDed with it) where `n` can reach 32: NVIDIA gives all lanes, AMD gives 0. Fix all of them with the guard already used at `reduction.cuh:503` and `convexMeshMidphase.cu:1444` (`n < 32 ? (1u << n) - 1 : 0xffffffff`), in shared code without a HIP guard (the CUDA result is the same), in a commit without the `[ROCm]` prefix like `49671488`:
   - `gpusolver/src/CUDA/solverMultiBlock.cu:720` and `solverMultiBlockTGS.cu:325`: `maskHigh = (1 << ((subWarpIndex + 1) * numThreadsPerBody)) - 1` with `numThreadsPerBody = min(nextPow2(numSlabs), 32)` (`:708`, `:311`). The shift is exactly 32 for the last sub-warp of every launch; when `numSlabs > 16` the mask is 0 on AMD and `hasRef = (__ballot_sync(mask, hasRef) & mask)` (`:778`, TGS `:376`) is always false, so the averaged body velocity is never written back. Reachable deterministically for a body with more than 16 slabs (more than 128 partitions at the default `gpuMaxNumPartitions` 8). For 16 or fewer the AMD mask gains bit 0 (lane 0's `hasRef` leaks into the last sub-warp); fix that too, since it is the same expression.
   - `gpunarrowphase/src/CUDA/cudaGJKEPA.cu:479` (`reduce(..., (1 << nbContacts) - 1)`) and `convexTriangle.cuh:1855` (`mask = (1 << numContactsGenerated) - 1`): `addContacts` (`cudaGJKEPA.cu:71-85`) stores at most 32 contacts but `nbContacts` keeps counting, so polyClip can return 32 or more; at 32 the AMD mask is 0 and `minIndex`/`maxIndex` in `reduce` return -1, at 33 to 63 it covers only the low lanes.
   - `cudaGJKEPA.cu:130`, `:152`, `:153` and `convexTriangle.cuh:1003`: masks from a face vertex count, which `ConvexMesh::isGpuCompatible()` (`geomutils/src/convex/GuConvexMesh.cpp:85`) and the kernel asserts (`cudaGJKEPA.cu:925-926`) allow to be 32. Not demonstrated here: cooking a 32-sided prism produced a 31-vertex largest face (62 vertices), and 8 to 31-sided prisms on a box match the CPU exactly; but 32 is legal input. `cudaGJKEPA.cu:2993` (`totalContacts`, at most 32) the same.
   No harness scene reaches any of these; the polygon-clip ones need faces with many vertices and the solver one needs a body with more than 16 slabs. Add a harness scene for the solver case (one body constrained by enough joints or contacts to exceed 16 slabs, compared with the CPU) if it can be built in reasonable size; otherwise state that it is fixed by inspection.

4. **The public `PxCudaTypes.h` pulls in an internal header on any HIP-aware consumer (AMD commit `d09e2532`).** `physx/include/cudamanager/PxCudaTypes.h:28` includes `cuda_to_hip.h` whenever `__HIP_PLATFORM_AMD__` or `__HIPCC__` is defined. `PxScene.h` includes `PxCudaTypes.h`, and `cuda_to_hip.h` lives in `source/gpucommon/include`, which is not installed. Checked: a host translation unit that includes `<hip/hip_runtime.h>` before `PxPhysicsAPI.h` (`amdclang++ -D__HIP_PLATFORM_AMD__ -I<rocm>/include -I physx/include`) fails with `PxCudaTypes.h:28:10: fatal error: 'cuda_to_hip.h' file not found`; even where it is found it would define `cuInit`, `__shfl_sync`, `__syncwarp` and dozens more macros in the consumer. Upstream's own guard already does the right thing for PhysX's GPU sources: `cuda_to_hip.h` defines `CUDA_VERSION`, so `#ifndef CUDA_VERSION` skips the forward declarations in every force-included TU. Keep the public header free of internal includes (drop the HIP branch, or key it on a macro only the GPU targets define) and check that a consumer that includes the HIP runtime first still compiles.

### Build system

5. **`CMAKE_HIP_FLAGS` is overwritten on every configure (AMD commit).** `physx/source/compiler/cmakegpu/linux/CMakeLists.txt:170` sets `CMAKE_HIP_FLAGS` to `CMAKE_HIP_FLAGS_INIT` with `CACHE ... FORCE`, so a user's `-DCMAKE_HIP_FLAGS=...` is silently discarded. Add the `--rocm-path`/`--rocm-device-lib-path` flags to the init value only, or append them, without forcing the cache. In the same file pair, amdhip64 is located twice by different methods (`physx/CMakeLists.txt:371-402` with `find_library`, and `source/compiler/cmakegpu/linux/PhysXGpu.cmake:65-108` with a glob) and linked twice; keep one. The `PX_THEROCK` branch (`linux/CMakeLists.txt`, about 50 lines) has never been configured by anyone on record; configure it once or drop it.

### Minimal footprint and comments

6. **AMD-base noise still in the delta (AMD commit).** The root `.gitignore` (`build-hip/`, `physx/build-hip-test/`, `physx/lib/`) is not upstream's and is local build-tree noise of the same kind the take-up ruling asked to revert. Remove it.
7. **Comments that are now wrong.** `gpucommon/include/cuda_to_hip.h:108-110` says the SDF texture path is compiled out on HIP because CDNA has no image hardware (stale since `07e78d64`, this round); `:59-66` says disabling HIP's native vector operators makes PhysX's `cutil_math.h` operators the single set, while `cutil_math.h` removes those operators on HIP in favour of HIP's (AMD commit); `cudamanager/src/CudaKernelWrangler.cpp:160` names `hipModuleLoadDataEx` where `CudaContextManager.cpp` calls `hipModuleLoadData`; `CudaContextManager.cpp:488,534` and `CudaKernelWrangler.cpp:328` say "gfx942" where any target applies (AMD commit). Correct them.

### Testing

8. **The harness misses the paths that are broken.** `SnippetGpuParity` has no primitive-vs-triangle-mesh scene, no aggregate scene, and no convex face with many vertices; the GPU snippets' "exit 0 with N dispatches" could not catch findings 1 and 2. Add, compared with the CPU: a box, a convex and a sphere resting on a static triangle mesh; a self-colliding aggregate stack; and a convex with large faces on a box and on another convex. Each must fail at 4764ca13 and pass with the fixes (record that teeth result).

### Points the porter asked a reviewer to judge

- Harness fails when it cannot get a GPU context: accepted. It is a GPU validation tool that stays on the fork, is only built for GPU builds, and the porter showed the plan's skip was a false pass (the ordinal is negative only when the GPU library failed to load). Failing is the right behaviour; the plan's wording is superseded.
- `atomicAddSpatialVector` fix with no launching host code: keep it. The guard extension is correct, matches the overflow fix beside it, and the commit message says it is latent.
- `__syncwarp` as release fence + `wave_barrier` + acquire fence at wavefront scope: correct; it is exactly HIP's own `__syncwarp()` (`include/hip/amd_detail/amd_warp_sync_functions.h:167-174` in the 10.2 SDK), and `wave_barrier` is not `s_barrier`, so divergent callers cannot deadlock.
- Radix sort fix: correct where applied (a wrapped `(radix - i) << 3` or `<< 4` is never below 32), incomplete in the same file (finding 1); the same assumption survives elsewhere (finding 3). The masks sweep found the rest bounded below 32 (lane ids, `threadIdx.x & 28`, literal or `isValid()`-bounded broadphase shifts, 64-bit articulation bit indices from `__clzll`/`__ffsll`); `algorithms.cu:152` can reach 32 at blockDim 1024 but only feeds a shuffle mask the compatibility header ignores.
- Wavefront width: the 32-lane logical-warp model holds on wave64 and wave32. No device use of the builtin `warpSize` (every use is a local `= 32` or `WARP_SIZE` shadow); no host query of the device warp size; every launch has blockDim.x a multiple of 32 (2D/3D launches use x = 32, or 128x2), so `threadIdx.x & 31` is the hardware lane within its 32-lane subgroup; HIP's `__shfl` takes `srcLane` modulo width like CUDA; `ballotSync` is zero-extended with subgroup base 0 on wave32. Not a defect today: HIP's `__shfl_xor` with width 32 and a lane mask of 32 or more would let the upper logical warp of a wave64 read the lower one (CUDA returns the caller's value); every PhysX call uses a mask of 16 or less. LDS from the porter's earlier gfx1100 code objects is within 64 KB, the closest being `sb_meshContactGenLaunch` at 64496 bytes; the wave32 validator should confirm on the current head.
- Lessons in this file: both read correctly against the code (PTX `shl` clamps the count; protected visibility binds the wrappers locally). Lesson 1 should add that the porter's own grep missed sites in the same file, and name the solver and contact-count masks of finding 3, which are its best examples.

## CUDA compile check 2026-11-24 (mi355 fleet host, nvcc 12.8, no GPU used)

Requested by Jeff: compile the CUDA path of fork `4764ca13` with real nvcc and compare it with upstream 5.13 `7d6f6ac8`, which the branch merges. Nothing was run on a GPU (this job had none and used none), nothing in the fork changed, nothing was pushed, and the project's stage and lock were left alone (the review above holds them). Builds lived in the job directory (`<job>/nvcc/`): `up/` at `7d6f6ac8`, `port/` at `4764ca13`, and `same/`, one worktree built at `7d6f6ac8` and then at `4764ca13` in place so that object files can be compared byte for byte (paths, and therefore `__FILE__` strings and debug info, are the same).

Verdict: **no CUDA regression.** Same objects, no errors, no warnings (the build compiles every host file with `-Werror` and every `.cu` with `-Werror=all-warnings`, so any new warning would have failed it), identical compile commands for every shared file, identical SASS on all 8 targets for 59 of the 61 kernel TUs, and the two TUs that differ are explained below and equivalent on NVIDIA hardware. One small CUDA behaviour change in host code (finding 1) is left for a person to judge; it does not break the build or a normal run.

### Toolchain and what was substituted

- nvcc `/usr/local/cuda-12.8/bin/nvcc`, "Cuda compilation tools, release 12.8, V12.8.93"; host gcc 13.3.0; CMake 3.31.6. Upstream pins CUDA 12.8 (`physx/documentation/platformreadme/linux/README_LINUX.md`), so no version substitution.
- Upstream's Linux build as documented: `cd physx && ./generate_projects.sh linux-gcc` with `PM_CUDA_PATH=/usr/local/cuda-12.8`, `CUDACXX=/usr/local/cuda-12.8/bin/nvcc`, `PM_PACKAGES_ROOT=<job>/nvcc/packman-repo`, then `make -C compiler/linux-gcc-release` (and `-checked`). Packman fetched upstream's public packages (`clang-physxmetadata 4.0.0.32489833_public`, `rapidjson 1.1.0-67fac85-073453e1_public`); CUDA is not a packman package on Linux.
- The `linux-gcc` preset builds the snippets, whose configure needs GLUT, which the image lacks: built freeglut v3.6.0 from source into `<job>/nvcc/deps/prefix` (`cmake -DFREEGLUT_BUILD_DEMOS=OFF -DFREEGLUT_BUILD_STATIC_LIBS=OFF`) and passed it with `CMAKE_PREFIX_PATH`. Same at both commits.
- The toolkit has no `nvdisasm`, and the triton wheel's (13.1.80) cannot decode sm_52 or sm_70, so `cuobjdump -sass` used NVIDIA's `cuda_nvdisasm-linux-x86_64-12.8.90-archive.tar.xz` from `developer.download.nvidia.com/compute/cuda/redist`, SHA-256 `8ac63f40...12ae`, matching `redistrib_12.8.1.json`.
- GPU targets: upstream's full list, SASS for sm_52, 70, 80, 86, 89, 90, 100, 120 plus compute_120 PTX (`PX_GENERATE_GPU_REDUCED_ARCHITECTURES=False`).

### Builds (all at both commits, all exit 0)

| Build | 7d6f6ac8 | 4764ca13 |
|---|---|---|
| preset `linux-gcc`, release, full tree (SDK, PhysXGpu, snippets) | 890 objects, 61 `.cu.o` | 892 objects (the two extra are `SnippetGpuParity.cpp.o` and its `ClassicMain.cpp.o`), 61 `.cu.o` |
| preset `linux-gcc`, checked, full tree | builds, 0 warnings | builds, 0 warnings |
| direct configure `cmake -S physx -B build-top -G Ninja -DPHYSX_PRESET=linux-gcc -DCMAKE_BUILD_TYPE=release`, build and `cmake --install` | 632/632 steps | 632/632 steps |

- Compile commands (`compile_commands.json`, paths normalized): preset release 734 shared files and direct configure 590 files, **0 differ** in flags, defines or include paths. No CUDA command carries `-include cuda_to_hip.h`, `PX_USE_HIP` or a `hip_compat` path, and none of the 961 dependency files of the 4764ca13 build names `cuda_to_hip.h`, `hip_compat/` or `PxgSdfLinearGrid.h`: the HIP compatibility layer never reaches nvcc or the CUDA host build.
- `libPhysXGpu_64.so`: same 210 exported dynamic symbols at both commits, no `__hipRegister*` symbol, `NEEDED` only libstdc++/libm/libc (libcuda is opened at run time by the loader, as upstream).
- Installed `lib/cmake/PhysX/PhysXConfig.cmake` from the direct configure: the condition becomes `if((ON OR OFF) AND NOT "OFF")`, which CMake evaluates true (checked with `cmake -P`), so a CUDA build's package still does `find_dependency(CUDAToolkit REQUIRED)` as upstream's does.

### Object comparison (same-path builds)

- 825 objects byte-identical, among them every host `.cpp` that includes the edited public headers (`PxPreprocessor.h`, `PxCudaTypes.h`, `PxParticleGpu.h`) and the two snippet renderers with the new `#ifndef USE_CUDA_INTEROP`.
- 67 differ: 61 kernel TUs rebuilt, 57 of them differ in bytes; 4 host objects of the GPU library; the 2 new parity-snippet objects.
- Host objects: `PxgGeometryManager.cpp.o`, `CudaKernelWrangler.cpp.o` and `PxPhysXGpuModuleLoader.cpp.o` differ only in `mov $imm` operands that are `__LINE__` values passed to the error reporters (the inserted HIP blocks move later lines; `objdump -dr` shows no other difference). So the CUDA `createTextureObject`/`removeGeometry` (texture objects, `cuArrayDestroy`), the CUDA kernel lookup by `moduleIndex` with the cudart registration stubs, and the `libcuda.so.1` preload in the loader all compile exactly as upstream. `CudaContextManager.cpp.o` also differs in code: finding 1.
- Kernels, compared per function and per target with `cuobjdump -sass` (whitespace-normalized, encodings included): **55 of the 57 differing TUs have identical SASS in every kernel on all 8 targets**. Comparing every section of their 8 cubins shows the bytes differ only in `.debug_line`, `.debug_str`, `.nv_debug_line_sass`, `.nv_debug_ptx_txt` and their relocations (the edited headers gained lines, `-lineinfo` records them), and in `.symtab`/`.strtab`, where the tables made `static __constant__` (`cudaNpCommon.h`, `epa.cuh`, `vector.cuh`, `marchingCubesTables.cuh`) get a TU-internal mangled name (`_ZN5physx32local_invalidateThresholdsConvexE` becomes `_ZN41_INTERNAL_..._cudaBox_cu_...5physx32local_invalidateThresholdsConvexE`), same section, already `LOCAL` in the cubin. No `.text`, `.nv.constant*`, `.nv.info` (registers, shared memory, stack) or global data differs. PhysX never looks a device variable up by name (no `cuModuleGetGlobal`/`cudaMemcpyToSymbol` in `physx/source`), so the rename has no effect.

The shared-source changes Jeff asked about, each confirmed to take its original code on CUDA:

- **Radix sort shift fix** (`RadixSort.cuh`, `radixShiftLeft`/`radixShiftRight` under `__HIP_PLATFORM_AMD__`): all 4 TUs that include it (`radixSortImpl`, `aggregate`, `convexMeshPostProcess`, `accumulateThresholdStream`) have identical SASS. The reviewer's finding 1 fix (the 8 single-warp sites through `radixShiftLeft`) keeps this property, since the helper is a plain shift on CUDA.
- **Re-enabled atomics**: `PxRedAddGlobal` (`atomic.cuh`, HIP branch first, then upstream's `__CUDA_ARCH__ >= 350` `red.global.add.f32`), `setOverflowError` (`cudaNpCommon.h`) and `atomicAddSpatialVector` (`PxgArticulationBlockData.h`): every including TU (11, 23 and 44 TUs) has identical SASS, `rigidDeltaAccum` included.
- **`__syncwarp` mapping**: lives only in `cuda_to_hip.h`, which no CUDA compile includes (above); upstream's `__syncwarp` reaches nvcc unchanged.
- **SDF sampler**: `pxgSampleSdfTexture3D` is `tex3D<float>` on CUDA. The 4 TUs that include `sdfCollision.cuh` keep identical SASS in every SDF kernel and the same texture instruction counts on sm_90 (`TEX*`: `convexMeshMidphase` 399, `particleSystemMeshMidphase` 120, `femClothPrimitives` 90, `softbodySoftbodyMidPhase` 93, at both commits).
- **Loader and registration**: see the host objects above; the HIP loader branch is behind `PX_USE_HIP`, which the CUDA build never defines, and the protected `__hipRegister*` wrappers and module scan are behind `__HIP_PLATFORM_AMD__ && !__HIPCC__`.
- `warpHelpers.cuh` (PTX `shfl.up` scan kept on CUDA), the `bar.sync N` macro in `convexMeshMidphase.cu`, the `__ffsll`/`__clzll` casts in `articulationImpulseResponse.cuh`, the `cutil_math.h` operator guards, the `float4` forward declarations, `PxPreprocessor.h` and `PxCudaTypes.h`: no SASS change in any including TU.

The two TUs whose SASS differs, both from unguarded shared edits that are equivalent on NVIDIA:

- **`convexMeshMidphase.cu`, kernel `triangleTriangleOverlaps` only** (commit `49671488`, BV32 root masks): upstream's `(1 << n) - 1` compiles to a plain `SHF.L`/`SHL`, which on NVIDIA returns 0 for a count of 32, so the mask is `0xffffffff`; the new `n < 32 ? (1u << n) - 1 : 0xffffffff` adds one `ISETP.GE.U32 ... 0x20` and one `SEL` per mask (2 of each on every target; sm_52 builds the same value as `SEL(~(0xffffffff << n), -1)`), plus register and scheduling changes (2368 vs 2376 instructions on sm_90). Same mask value for every `n` in 0..32; the other 3 kernels of the TU are identical.
- **`algorithms.cu`, kernels `scanPerBlockKernel`, `scanPerBlockKernel4x4`, `addBlockSumsKernel4x4`, `radixFourBitCountPerBlockKernel` on sm_52 to sm_90** (AMD commit `d09e2532`, `warpSize` replaced by `PXG_LOGICAL_WARP_SIZE = 32`): upstream's `id % warpSize` and `/ warpSize` are lowered as a generic unsigned division by a value nvcc only learns is 32 late (`I2F.U32.RP`/`MUFU.RCP`/`IMAD.HI` sequences); with the literal they become `& 0x1f` and shifts, and the loops bounded by `blockDim.x / 32` unroll differently (for example `scanPerBlockKernel` 152 to 120 instructions on sm_90). Same arithmetic as long as `warpSize` is 32, which it is on every NVIDIA GPU; on sm_100 and sm_120 nvcc folds `warpSize` itself and the SASS of all 7 kernels is identical at both commits, which is the direct evidence that the two forms are the same program.

### Finding 1 (for the porter and Jeff; not a build break)

`CudaContextManager.cpp` (AMD commit `d09e2532`) restructured the CUDA module-load loop, not only the HIP one. Upstream:

```
CUresult ret = CUDA_ERROR_UNKNOWN;
if (moduleTable[i])
{
    ret = mCudaCtx->moduleLoadDataEx(&mCuModules[i], moduleTable[i], 0, NULL, NULL);
}
if (ret != CUDA_SUCCESS && ret != CUDA_ERROR_NO_BINARY_FOR_GPU)
{ ... report "Failed to load CUDA module data. Cuda error code %i." ... }
```

At `4764ca13` the loop starts with `if (!moduleTable[i]) continue;` for both paths, so on CUDA a null module-table entry (one `__cudaUnregisterFatBinary` has zeroed) is now skipped silently where upstream reported error 999. Every non-null entry behaves as before, and the generated code differs only in this loop. Smallest fix, if the reviewer or Jeff wants the CUDA path untouched: move the `continue` inside the `#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)` branch and restore upstream's four lines above (with its blank line) in the `#else` branch. Not applied here.

### Reproduce

```
cd <clone>/physx
export PM_PACKAGES_ROOT=<dir> PM_CUDA_PATH=/usr/local/cuda-12.8 CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH=/usr/local/cuda-12.8/bin:$PATH CMAKE_PREFIX_PATH=<freeglut prefix>
./generate_projects.sh linux-gcc
make -C compiler/linux-gcc-release -j128
make -C compiler/linux-gcc-checked -j128
cuobjdump -sass -arch sm_90 <obj>.cu.o   # nvdisasm 12.8.90 on PATH
```

Comparison scripts (job-local, not kept): `<job>/nvcc/scripts/cmpcmds.py` (compile commands), `sasscmp.py` (per-kernel SASS, 8 targets), `cubincmp.py` (cubin sections), `fdiff.sh`/`opdiff.py` (one kernel's SASS and opcode counts).

## Porter round 2026-11-24, review fixes (linux-gfx942, MI300X, headless fleet job mi300c-20261124-035623)

Answers "Review 2026-11-24" (eight findings) and finding 1 of the CUDA compile check. Fork `moat-port` 4764ca13 -> **064e809b**, ten new commits on top (plain push, `GIT_LFS_SKIP_PUSH=1`; no amend, rebase or force push of anything published). One local correction before the push: the harness commit was first written with a 74-character title; that unpushed commit was soft-reset and recommitted with a 69-character title and an identical tree (checked with `git rev-parse HEAD^{tree}`), because the commit gate refuses titles over 72 characters.

### Toolchain (this job)
- `$FLEET_VENV/bin/python -m pip install --index-url https://nightly.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel]==10.2.0a20261022" "rocm-sdk-device-gfx942==10.2.0a20261022"`; `rocm-sdk path --root` = `<job>/venv/lib/python3.11/site-packages/_rocm_sdk_devel`, `.info/version` 10.2.0.
- Compiler the builds invoked: `CMAKE_HIP_COMPILER=<root>/lib/llvm/bin/amdclang++` (AMD clang 24.0.0git `2792f8ceeb5a`), taken from `CMakeCache.txt` of the preset build and auto-discovered through `rocm-sdk`; host `/usr/bin/c++` (gcc 13.3). `ldd libPhysXGpu_64.so` resolves `libamdhip64.so.7` from the same root. The image's ROCm 10.1 was not on the compile path.
- `$FLEET_VENV/bin` first on PATH, `MAX_JOBS=64`, `HIP_VISIBLE_DEVICES=1` (the dispatcher's pin), `ulimit -c 0`, `timeout` around every GPU run. Snippet dependencies (freeglut, GLU, libglvnd prefix and packman cache) copied from the porter job directory.

### Commits (oldest first)
1. `e1080687` [ROCm] Compile GPU code without strict aliasing or noundef arguments (finding 2)
2. `3ebaa20c` [ROCm] Guard the bucket shifts of the single-warp radix sort (finding 1)
3. `a2667442` Build lane masks from counts that can reach 32 without a shift by 32 (finding 3; shared code, no prefix)
4. `2935dab2` [ROCm] Keep the public CUDA types header free of internal includes (finding 4)
5. `638c5b36` [ROCm] Keep the caller's CMAKE_HIP_FLAGS and find amdhip64 once (finding 5)
6. `51d7478b` [ROCm] Drop the untested TheRock configuration branch (finding 5)
7. `8f41fd34` [ROCm] Remove the root .gitignore added with the ROCm port (finding 6)
8. `31e8c427` [ROCm] Correct comments in the HIP compatibility code (finding 7)
9. `0b3fd207` Restore upstream's CUDA handling of a null module table entry (CUDA compile check, finding 1)
10. `064e809b` Add mesh, aggregate, large-face and slab scenes to the parity snippet (finding 8)

### Finding 2: root cause (every primitive falls through a static triangle mesh)

Two independent miscompilations by clang of upstream code that nvcc compiles as intended. Neither is a `bar.sync` mapping, narrowphase-algorithm or post-process defect. **Introduced by the AMD enablement commit `d09e2532`**, which compiles the GPU code with amdclang++ without either flag. Not by this round's commits.

How it was narrowed:
1. Reproduced at 4764ca13 with the reviewer's `MeshCheck.cpp`, unchanged: all 12 cases at y = -19.16 to -19.38 (CPU 0.2500), with the touch-event warning.
2. Attribution: `d09e2532` plus only what is needed to run the GPU at all (cherry-picked loader `332132c7`, registration visibility `58229d69`, radix `d1714e8e`, the `compiler/public` CUDA-language skip from `0af51c4c`, offload arch `2039a7d8`; built through a temporary `linux-gcc` + `PX_USE_HIP` preset) fails the same way (-19.28 to -19.60 m). With the two flags below it passes 12/12. Only ROCm 10.2 was available here; the AMD commit says it used ROCm 10.0.
3. Minimal repro: one sphere (or box) on a two-triangle mesh. A device `printf` at the end of `midphaseCore` made the sphere rest correctly, a heisenbug. Printfs in the narrowphase, correlate and finish kernels did not. So the midphase output was suspect, and printing changes code generation.
4. Host readback instead of device printf: after the midphase launch, `streamSynchronize` plus `memcpyDtoH` of the `ConvexMeshPair` array and the pair stack (scratch tree only, under an env variable). Pair counts and triangle pairs were right, but **`aToB` was all zeros** (p = 0, q = (0,0,0,0)), and correct (p = (0.1, 0.287, 0.1), q = identity) whenever the device printf was compiled in. `ConvexMeshPair_WriteWarp` stores the register struct `pairInfo` by reading it back as `uint` (`reinterpret_cast<const uint*>(&inp)[laneIdx]`). Under clang's TBAA the float stores to `pairInfo.aToB` are dead, so they are dropped. The int fields survive because int and unsigned may alias. `-fno-strict-aliasing` on the HIP objects: the sphere and capsule rest. The upstream GCC host build already uses `-fno-strict-aliasing` (`source/compiler/cmake/linux/CMakeLists.txt:105`).
5. The box and convex then froze in mid-air at y = 0.2867, the height at first touch, while the CPU settled at 0.25. Finish-kernel contacts had `pA = (0,0,0)` and `pen = 0`. Host readback after `convexTrimeshNarrowphase`: the temporary-contact counter was 256 for 3 + 5 real contacts (32x), and the first stored contact was garbage `(0, 1, nan)`. Device asm of `convexTrimeshNarrowphase` (`--offload-device-only -S`): the atomic-optimizer loop before `global_atomic_add` sums `nbOutputContacts` over the whole `exec` mask, and there is no `s_and_saveexec` for `if (threadIdx.x == 0)`. Cause: `PxU32 startIndex;` in `convexTriangleContactGen` (`convexTriangle.cuh:1873`) is assigned only on lane 0 and then passed to `__shfl_sync`. Clang marks arguments `noundef`, so the undefined incoming value on the else edge lets LLVM delete that edge. Every lane then runs the atomic and the `cvxTriContacts[...].index` store. Initialising `startIndex = 0` restores the guard (`s_and_saveexec s[20:21], s[0:1]` before the loop), and so does `-Xclang -no-enable-noundef-analysis` with the code unchanged.
6. Flag matrix, all GPU targets, sphere / box after 40 frames: none: -2.24 / frozen; `-fno-strict-aliasing` only: 0.2500 / 0.2867 frozen; noundef opt-out only: -2.16 / -1.89; both: 0.2500 / 0.2500.

Fix (commit 1): `ADD_COMPILE_OPTIONS("$<$<COMPILE_LANGUAGE:HIP>:-fno-strict-aliasing;SHELL:-Xclang -no-enable-noundef-analysis>")` once in the HIP section of `cmakegpu/linux/CMakeLists.txt`. Verified in `compile_commands.json`: 61/61 `.cu` carry both, 0 host `.cpp`. Chosen over patching the two sites, because the GPU code reinterprets structures through other types throughout (`*_ReadWarp`/`*_WriteWarp`, shape copies) and the lane-0-assign-then-broadcast pattern recurs, so a site fix would leave the same class elsewhere. CUDA is untouched (HIP-only CMake).

### Findings 1 and 3, and the second sweep
- Finding 1: the eight `accum0..3` shifts in `radixSortPassSingleWarp` and `radixSortPassSingleWarpKeysOnly` now go through `radixShiftLeft` (plain shift on CUDA). The guarded shifts below them, and `:221`/`:337` (under `threadIndexInWarp < 2` / `< 4`), were checked and are bounded.
- Finding 3: `n < 32 ? (1 << n) - 1 : -1` (`0xffffffff` for unsigned), no HIP guard, same value on NVIDIA: solver `maskHigh` (both solvers), `cudaGJKEPA.cu` polyClip vertex masks (`vertMask0/1`, 3 uses), `:483`, `:2997`, `convexTriangle.cuh:1003`, `:1855`. **Two more sites the review did not list:** `cudaGJKEPA.cu:950` and `:984` (`int mask = (1 << nbContacts) - 1` after `polyClip` in the convex-convex stage-2 kernel), where `polyClip` returns the raw `addContacts` count, which can exceed 32.
- **Sweep method** (`agent_space/scratch/shiftsweep.py`, job-local): strip comments and strings from every `.cu`/`.cuh` and GPU header under `physx/source/gpu*` and `cudamanager`, find every `<<`/`>>`, take the right operand up to the first lower-precedence operator, and keep each one whose count is not an integer literal. That gives 280 sites, 52 of them shifting by a `& 31` lane id. Each of the other 228 was classified by the range of its count. The first sweep grepped for masks named after counts or lanes, which is why it missed shifts of a data value or of a count that can reach the word width. The 41 shifts by bare `threadIdx.x`/`tI` were also checked against their kernels' launch blockDim.x (sub-agent, read-only, 41/41 SAFE: blockDim.x 32 on every launch, or guarded by `warpIndex == 0` in the 1024-thread `triangleTriangleCollision`, or by `threadIdx.x < 4` in `sbClothCollision`). Everything else is bounded below 32: enum and bit-field constants, `% 32`/`& 31`, loop indices below 32, 64-bit articulation bit indices, and `in1 << nbVerts1` in polyClip, which ORs a value with itself when the count is 32 and so gives the same result on both vendors. The `1 << minIndex(...)` and `1 << lowestSetIndex(...)` family in the contact reductions shifts by -1 only on an empty mask. The callers keep the mask non-empty (cluster picks are guarded by `if (cluster)`, and the fill loop runs only when the input has more than TMaxPoints bits). `gjk.cuh:207` would shift by -1 only for a NaN support distance, which is broken on both vendors.
- Face vertex count 32 is legal but not reached: cooking a 31- or 32-sided prism gives a largest face of 31 (confirmed again in the harness). The vertex-count masks are fixed by inspection.

### Findings 4 to 7 and the CUDA module-load loop
- 4: `PxCudaTypes.h` restored byte-for-byte to upstream 7d6f6ac8. `cuda_to_hip.h` defines `CUDA_VERSION`, so every force-included GPU TU takes upstream's `#else` branch, and HIP's `hipHostMalloc{Portable,Mapped,WriteCombined}` = 1/2/4 satisfy its compile-time asserts. `cuda_to_hip.h` now includes `foundation/PxSimpleTypes.h`, which the skipped branch used to provide. Consumer check: `#include <hip/hip_runtime.h>` then `PxPhysicsAPI.h`, `amdclang++ -D__HIP_PLATFORM_AMD__ -I<rocm>/include -I physx/include -c`, and the same file as `-x hip --offload-arch=gfx942`. Both compile at 064e809b; at 4764ca13 the first fails with `'cuda_to_hip.h' file not found`.
- 5: the `CMAKE_HIP_FLAGS` FORCE was replaced by prepending the ROCm path flags to the caller's value, before `enable_language(HIP)`. Checked with `-DCMAKE_HIP_FLAGS=-DPX_EXTRA_FLAG_CHECK`: 61/61 HIP commands carry it and `--rocm-path`, the cache keeps the user's value, and a reconfigure leaves it unchanged. amdhip64 is now located once (`cmakegpu/linux/PhysXGpu.cmake`); the `find_library` block in `physx/CMakeLists.txt` was removed. `PX_THEROCK` was configured once before being dropped: `-DPX_THEROCK=ON -DCMAKE_HIP_COMPILER=<root>/lib/llvm/bin/amdclang++ -DCMAKE_PREFIX_PATH=<root>` configures, but `--target PhysXGpu` fails with `cuda_to_hip.h:71: hip/hip_runtime.h: No such file or directory` (the HIP include path only reaches PhysXGpu through `hip::host`). The branch was removed; the standalone path accepts a caller-supplied `CMAKE_HIP_COMPILER`.
- 6: root `.gitignore` removed. Untracked build output under `physx/lib/` and `physx/bin/` now shows in `git status`, as it does upstream.
- 7: corrected the SDF-texture, vector-operator, `hipModuleLoadDataEx` and "gfx942" comments. The `#define __HIP_USE_NATIVE_VECTOR__ 0` was dead: HIP's `amd_hip_vector_types.h:25` redefines it to 1, and it only selects how HIP implements its operators. Removing it took `-Wmacro-redefined` from 198 to 0 in the release build. Other warnings are unchanged by category (1049 deprecated-declarations, 732 unused-value, 342 unused-result, 312 attributes, 48 invalid-offsetof, 30 cuda-compat, 6 parentheses), plus 4 `-Wdangling-else` reported in upstream `convexMesh.cu:357/571`, which this round does not touch.
- CUDA compile check finding 1: `if (!moduleTable[i]) continue;` moved inside the HIP branch, and upstream's CUDA code restored verbatim in the `#else`.

### Finding 8: harness scenes and teeth
New scenes in `SnippetGpuParity` (all compared with the CPU): **primitives on mesh** (box, convex cube, sphere, capsule on an 8x8 static grid mesh: position difference and rest height); **aggregate pile** (the reviewer's 12 staggered 0.5 m boxes dropped into one self-colliding aggregate: lowest box against the CPU, on the ground, below start). A first version with an initially stacked, touching column passed at 4764ca13, so it was replaced; a tall column sways a few millimetres either way, so the positions are not compared. **large-face convex** (31-vertex-cap prism on a box, a second prism turned half a step on top). **many slabs** (dynamic 8x0.2x8 m plate carrying 13x13 boxes, more than 128 partitions on one body).

Teeth, all release, MI300X. The harness binary is the 064e809b one in every run; only `libPhysXGpu_64.so` differs.

| GPU library | Result |
|---|---|
| 064e809b | 47 checks, 0 failed (also checked build: 47/0) |
| 4764ca13 | 13 failed: all 8 primitives-on-mesh checks (15.7 to 19.6 m off), aggregate pile lowest box -43.66 m (2 checks), large-face convex 0.0215 m, many slabs 0.200 m, plus "PhysX errors reported" (the touch-event warning) |
| 064e809b without the compiler flags | 9 failed: the 8 primitives-on-mesh checks and the touch-event error |
| 064e809b without the radix fix | 2 failed: aggregate pile (lowest box -43.66 m) |
| 064e809b without the mask fix | 2 failed: large-face convex 0.0215 m, many slabs 0.200 m |

The solver more-than-16-slabs case was built in reasonable size (170 bodies, 120 frames) and has teeth.

### Builds at 064e809b (all exit 0)
- `./generate_projects.sh linux-gcc-rocm`, `make -C compiler/linux-gcc-rocm-{release,checked}` from scratch: 61 HIP objects each.
- Direct configure from INSTALL_ROCm.md with `ROCM_PATH`/`HIP_PATH` unset, `CMAKE_HIP_ARCHITECTURES="gfx942;gfx1100"`: 61 gfx942 + 61 gfx1100 code objects (`llvm-objdump --offloading`), `libamdhip64.so.7` linked.
- Standalone GPU configure from INSTALL_ROCm.md (`cmake -S physx/source/compiler/cmakegpu ...`, `--target PhysXGpu`).
- `PX_GENERATE_GPU_STATIC_LIBRARIES=ON PX_GENERATE_STATIC_LIBRARIES=OFF` (`libPhysXGpu_static_64.a`).
- CPU-only preset with `PATH=/usr/bin:/bin`.
- Reviewer's MeshCheck at 064e809b: 12/12 at y = 0.2500, no touch-event warning.
- `jargon.py -C projects/PhysX/src --diff 4764ca13..moat-port` and `--commits 4764ca13..moat-port`: clean. `jargon.py --port PhysX` still reports only upstream ovphysx text, because the fork's `main` mirror is at 5.10 (Question A of the previous porter round, still open).

### Shared (non-HIP-guarded) source lines changed this pass, for the nvcc compile check
Line numbers are at 064e809b:
- `physx/source/gpucommon/src/CUDA/RadixSort.cuh:513-516, 639-642`: calls to `radixShiftLeft`, a plain shift on CUDA. Expect identical SASS in the four including TUs.
- `physx/source/gpusolver/src/CUDA/solverMultiBlock.cu:720-722` and `solverMultiBlockTGS.cu:325-327`: `highShift`/`maskHigh` select. Same value for every count; SASS will differ by a compare and select.
- `physx/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu:130-134` (vertMask0/1 and use), `:156-157`, `:483`, `:950`, `:984`, `:2997`: mask selects, same values on NVIDIA.
- `physx/source/gpunarrowphase/src/CUDA/convexTriangle.cuh:1003`, `:1855`: mask selects.
- `physx/source/cudamanager/src/CudaContextManager.cpp:521-536` (the `#else` CUDA branch, restored to upstream's text) and `:474-477` (the `continue` moved under the HIP guard). Expect `CudaContextManager.cpp.o` to match upstream again apart from `__LINE__` immediates.
- `physx/include/cudamanager/PxCudaTypes.h`: now identical to upstream 7d6f6ac8.
- `physx/snippets/snippetgpuparity/SnippetGpuParity.cpp:601-758`, `:844-885`: new scenes and checks (fork-only file).
- HIP-only, which nvcc never sees: `cuda_to_hip.h`, `cmakegpu/linux/CMakeLists.txt` and `PhysXGpu.cmake` (inside `IF(PX_USE_HIP)`). Comment-only: `CudaKernelWrangler.cpp`. `physx/CMakeLists.txt` lost only the `PX_USE_HIP` amdhip64 block; the CUDA linking block is unchanged.

## Lesson for cuda-to-rocm: clang's strict aliasing and noundef rules miscompile CUDA code that nvcc accepts

Belongs in `references/faults-numerics.md` (or a compiler-semantics entry in `diagnosis.md`), with a pointer from the symptom index ("works with printf", "kernel output silently zero"). CUDA code compiled by nvcc often relies on two things clang's HIP front end does not guarantee:
1. **Type punning.** Code reads a struct as `uint` words to copy it lane by lane (`reinterpret_cast<const uint*>(&s)[lane]`). With type-based alias analysis, clang may treat the float stores into that struct as dead, so the copy writes zeros or garbage. In PhysX the triangle-mesh midphase emitted an all-zero transform, and every sphere and capsule fell through meshes.
2. **Undefined values passed to calls.** The pattern `T x; if (lane == 0) x = atomicAdd(...); x = __shfl_sync(mask, x, 0);` leaves `x` undefined on the other lanes. Clang marks call arguments `noundef`, so LLVM may delete the branch edge that carries the undefined value: the lane-0-only atomic then runs on every lane. In PhysX the contact counter advanced 32 times too fast.
Signs: the result changes when a `printf` is added in the kernel, and the device asm lacks the `s_and_saveexec` guard around a lane-0 atomic. Test with `-fno-strict-aliasing` and `-Xclang -no-enable-noundef-analysis` on the HIP sources. When the codebase uses either pattern widely, add both flags to the HIP compile (HIP sources only) rather than patching sites. To find the culprit, read device buffers back on the host instead of printing on the device, which perturbs code generation.

## Lesson for cuda-to-rocm: addendum to "masks and counters that rely on a 32-bit shift by 32 being 0"

As the reviewer asked: the first sweep in this port grepped for mask expressions and missed sites in the same file (`RadixSort.cuh`, eight more shifts) and elsewhere. The best examples of form (2) are the solver's last-sub-warp mask `(1 << ((subWarpIndex + 1) * numThreadsPerBody)) - 1`, whose count is exactly 32 on every launch (it breaks bodies with more than 16 slabs), and contact-count masks after polygon clipping, where the count can exceed 32 because the clipper keeps counting past its buffer. A sweep that would have caught them: list every shift whose count is not a literal, then classify each count's range. A script that strips comments and takes the right operand up to the first lower-precedence operator is enough; PhysX had 280 such sites.

## Review 2026-11-24, delta 4764ca13...064e809b (linux-gfx942, reviewer, headless fleet job mi300c-20261124-035623)

Scope: `git diff 4764ca13...064e809b`, the ten commits of the porter round above, read with the surrounding code. Three independent child reviews (CMake and flags; kernel shift fixes with a fresh sweep of their own; header, comments, harness and lessons) were run, and each finding was fact-checked against the code. The CMake review configured scratch builds (direct, standalone GPU, GPU-static) and inspected `compile_commands.json`, link lines and RUNPATH; nothing ran on a GPU.

Verdict: **changes requested** (two small fork changes). All eight findings of "Review 2026-11-24" are closed:
- 1: all 8 single-warp shifts go through `radixShiftLeft`; the counts were re-derived (the result is nonzero only for radix - i in 0..3).
- 2: the root cause holds up against the asm and the readback evidence. Both flags reach 61/61 HIP objects in the top-level, standalone and GPU-static configures, no host `.cpp`, and the SHELL-in-genex form expands correctly.
- 3: every listed site, plus `cudaGJKEPA.cu:950/984`, matches the NVIDIA value for every count including 33 to 63. An independent sweep (all non-literal shifts plus every ballot with a computed mask) found nothing new.
- 4: `PxCudaTypes.h` is identical to upstream, and a HIP-runtime-first consumer compiles.
- 5: the user's `CMAKE_HIP_FLAGS` is preserved and does not accumulate on reconfigure; amdhip64 still reaches shared, standalone and static consumers; `PX_THEROCK` has no leftovers.
- 6: `.gitignore` is gone.
- 7: the comments are accurate, and the removed define is unconditionally redefined by HIP at `amd_hip_vector_types.h:25`.
- 8: four scenes, with teeth recorded per fix. "many slabs" really reaches 169+ partitions and 22+ slabs (`PxgContext.cpp:1618`, `PxgConstraintPartition.cpp:921-958`).

The CUDA module-load loop now matches upstream line for line.

### Build system
1. **`HIPFLAGS` from the environment is ignored on the first configure.** `physx/source/compiler/cmakegpu/linux/CMakeLists.txt:112` builds a normal `CMAKE_HIP_FLAGS` from the cache value before `enable_language(HIP)` creates the cache entry from `$ENV{HIPFLAGS}`; the normal variable then shadows it. A fresh configure with `HIPFLAGS=-DPX_ENV_CHECK` gave 0/61 `.cu` commands with it; a plain reconfigure gave 61/61. (The old FORCE dropped it every time, so this is a new inconsistency rather than a regression, but this round wrote the line.) Fix: when `CACHE{CMAKE_HIP_FLAGS}` is not yet defined, use `$ENV{HIPFLAGS}` as the caller's value. Check with a fresh configure under `HIPFLAGS=...`.

### Testing
2. **The large-face scene does not check the face size it depends on.** `physx/snippets/snippetgpuparity/SnippetGpuParity.cpp:869-871` prints `largestFace` but never checks it. If cooking ever splits the 31-vertex cap, the scene silently loses the more-than-32-clipped-points coverage that is its purpose (its teeth come from the mask fix at `cudaGJKEPA.cu:950/984`). Add `check(largestFace >= 31, ...)`. In the same scene and `primitivesOnMesh`, a NULL cooked mesh (`cube`, `prism`) would crash at `setSolverIterationCounts` instead of failing a check. Guard it with a check as well.

### Commit hygiene (not fixable on the published branch; carry into the PR-prep squash message, which is a person's push)
3. `e1080687`'s Test Plan reports "1x1, 8x8 and 40x40 ... all 12 cases", but at that commit the snippet has no mesh scene. Even at 064e809b it has one 8x8 grid with 4 primitives; the 12 cases came from a scratch program. `3ebaa20c`'s Test Plan cites "the parity snippet's self-colliding aggregate stack", a scene added later (and renamed "aggregate pile"). `a2667442` says the convexTriangle.cuh contact count can exceed 32; that file's `addContacts` (`convexTriangle.cuh:905-927`) counts only stored contacts, so there it reaches exactly 32, never more. The fix is still needed. `0b3fd207`'s Test Plan exercises only ROCm; say that the CUDA branch is textually upstream's, or give the nvcc build line.

### Non-blocking notes for the porter (record, no reviewer verdict depends on them)
- `convexTriangle.cuh:1872` `PxU32 startIndex;` is still uninitialized; `-Xclang -no-enable-noundef-analysis` is an internal frontend option. Initializing it (`= 0`, the same result on CUDA) keeps the confirmed site correct if the option ever changes. The flag stays for the other sites.
- `638c5b36` says amdhip64 is now found once. The link line still names it twice, because CMake's HIP runtime library (`CMAKE_HIP_RUNTIME_LIBRARY_DEFAULT=SHARED`) adds it after the glob's. This is harmless.

### Lessons in this file
- "clang's strict aliasing and noundef rules": accurate, but "relies on two things clang's HIP front end does not guarantee" implies nvcc guarantees them. Both patterns are undefined behaviour that nvcc does not exploit. Reword before promotion.
- Addendum to "masks and counters": "breaks bodies with more than 16 slabs" is imprecise. `numSlabs` is scene-wide and the threshold scales with `gpuMaxNumPartitions` (16 x 8 = 128 partitions at the default). Once any body exceeds it, `numThreadsPerBody` is 32 for every body in the solve. "exactly 32 on every launch" holds for the last sub-warp of each warp. Reword before promotion.
