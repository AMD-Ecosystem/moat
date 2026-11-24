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

Belongs in `references/faults-numerics.md` (or a compiler-semantics entry in `diagnosis.md`), with a pointer from the symptom index ("works with printf", "kernel output silently zero"). CUDA code that nvcc compiles as intended often contains two kinds of undefined behaviour. nvcc does not exploit them, but clang's HIP compile does, so the same source that works on NVIDIA miscompiles on AMD:
1. **Type punning.** Code reads a struct as `uint` words to copy it lane by lane (`reinterpret_cast<const uint*>(&s)[lane]`). With type-based alias analysis, clang may treat the float stores into that struct as dead, so the copy writes zeros or garbage. In PhysX the triangle-mesh midphase emitted an all-zero transform, and every sphere and capsule fell through meshes.
2. **Undefined values passed to calls.** The pattern `T x; if (lane == 0) x = atomicAdd(...); x = __shfl_sync(mask, x, 0);` leaves `x` undefined on the other lanes. Clang marks call arguments `noundef`, so LLVM may delete the branch edge that carries the undefined value: the lane-0-only atomic then runs on every lane. In PhysX the contact counter advanced 32 times too fast.
Signs: the result changes when a `printf` is added in the kernel, and the device asm lacks the `s_and_saveexec` guard around a lane-0 atomic. Test with `-fno-strict-aliasing` and `-Xclang -no-enable-noundef-analysis` on the HIP sources. When the codebase uses either pattern widely, add both flags to the HIP compile (HIP sources only) rather than patching sites. To find the culprit, read device buffers back on the host instead of printing on the device, which perturbs code generation.

## Lesson for cuda-to-rocm: addendum to "masks and counters that rely on a 32-bit shift by 32 being 0"

As the reviewer asked: the first sweep in this port grepped for mask expressions and missed sites in the same file (`RadixSort.cuh`, eight more shifts) and elsewhere. The best examples of form (2) are the solver's last-sub-warp mask `(1 << ((subWarpIndex + 1) * numThreadsPerBody)) - 1`, whose count is exactly 32 for the last sub-warp of every warp at any slab count: AMD gives that body a mask with lane 0's bit added, and once the scene-wide slab count makes `numThreadsPerBody` 32 (more than 128 partitions on one body at the default `gpuMaxNumPartitions` of 8) every body's mask is empty. A second example is the contact-count mask after polygon clipping in `cudaGJKEPA.cu`, where the clipper keeps counting past its buffer, so the count can exceed 32. A sweep that would have caught them: list every shift whose count is not a literal, then classify each count's range. A script that strips comments and takes the right operand up to the first lower-precedence operator is enough; PhysX had 280 such sites.

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

## Porter round 2026-11-24, second review fixes (linux-gfx942, MI300X CPX slice, headless fleet job mi300c-20261124-045812)

Answers "Review 2026-11-24, delta 4764ca13...064e809b": finding 1 (HIPFLAGS), finding 2 (harness checks), the non-blocking `startIndex` note, the two lesson rewordings and the commit-hygiene record. Fork `moat-port` 064e809b -> **886c3529**, three new commits on top (plain push, `GIT_LFS_SKIP_PUSH=1`; nothing amended, rebased or force-pushed).

### Toolchain (this job)
- `$FLEET_VENV/bin/python -m pip install --index-url https://nightly.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel]==10.2.0a20261022" "rocm-sdk-device-gfx942==10.2.0a20261022"` into the job venv; `rocm-sdk path --root` (run once) = `<job>/venv/lib/python3.11/site-packages/_rocm_sdk_devel`, `.info/version` 10.2.0, `hipconfig --version` 7.17.26422.
- Compiler the builds invoked: `CMAKE_HIP_COMPILER=<root>/lib/llvm/bin/amdclang++` (AMD clang 24.0.0git `2792f8ceeb5a`) in `compiler/linux-gcc-rocm-release/CMakeCache.txt`, found through `rocm-sdk` with `ROCM_PATH`/`HIP_PATH` unset at configure time for the direct and standalone configures; host `/usr/bin/c++` (gcc 13.3).
- `$FLEET_VENV/bin` first on PATH, `MAX_JOBS=96` (exported by the dispatcher), `HIP_VISIBLE_DEVICES=1` (the dispatcher's pin), `ulimit -c 0`, `timeout` around every GPU run.
- Reused from job mi300c-20261124-035623 by copy: the fork clone (re-fetched; tree at 064e809b verified clean before editing, all build directories `physx/compiler/linux-gcc-rocm-*`, `physx/bin`, `physx/lib` deleted and rebuilt from scratch here) and the snippet dependency prefix and packman cache (`.pc` files re-pointed at this job's copy).

### Commits (oldest first)
1. `acba89ef` [ROCm] Take HIPFLAGS from the environment on the first HIP configure (finding 1)
2. `33c8a6c3` Check the cooked meshes and face size in the parity snippet scenes (finding 2; fork-only file)
3. `886c3529` Initialize the contact start index broadcast from lane 0 (non-blocking note; shared code, no prefix; the body says why CUDA is unaffected)

### Finding 1: HIPFLAGS
`cmakegpu/linux/CMakeLists.txt`: `IF(DEFINED CMAKE_HIP_FLAGS)` prepends the ROCm flags to the caller's value as before; otherwise (no cache entry and no normal variable yet, i.e. the first configure) the caller's value is `$ENV{HIPFLAGS}`. `DEFINED` without `CACHE{}` also respects a normal variable a parent project set. Checks (`compile_commands.json`, `.cu` entries):

| Configure | Result |
|---|---|
| fresh, `HIPFLAGS=-DPX_ENV_CHECK`, top-level `-DPHYSX_PRESET=linux-gcc -DPX_USE_HIP=ON` | 61/61 carry it, once each; 0 host entries; cache `CMAKE_HIP_FLAGS=-DPX_ENV_CHECK` |
| same tree reconfigured with `HIPFLAGS` set, then again without it | 61/61, once each; `--rocm-path=` once each (no accumulation) |
| fresh, `HIPFLAGS=-DPX_ENV_CHECK -DCMAKE_HIP_FLAGS=-DPX_EXTRA_FLAG_CHECK` | 61/61 `PX_EXTRA_FLAG_CHECK`, 0/61 `PX_ENV_CHECK` (the cache value wins, as in plain CMake) |
| fresh, no HIPFLAGS | 61/61 `--rocm-path=` once each, cache `CMAKE_HIP_FLAGS` empty |
| standalone GPU configure from INSTALL_ROCm.md, `HIPFLAGS` set, fresh and reconfigured | 61/61, once each |
| fresh, `HIPFLAGS` set, at 064e809b (fix stashed) | 0/61 (reproduces the finding) |

### Finding 2: harness checks
`largeFaceConvexes` and `primitivesOnMesh` now fail a check (`prism cooked`, `triangle mesh cooked`, `convex cube cooked`) and return instead of dereferencing a NULL cooked mesh; the largest-face printf became `check(largestFace >= 31, "large-face convex", "vertices in the largest cooked face", ...)`. Harness: **48 checks, 0 failed** (was 47; the new check reads 31). Teeth, release, each a temporary edit of the snippet only, restored and rebuilt afterwards:
- grid mesh and prism cooking replaced by NULL: exit 1, 44 checks, 6 failed (2x `triangle mesh cooked`, 2x `prism cooked`, `vertices in the largest cooked face` 0, upper prism height), no crash. The same edit without the guards had crashed with SIGSEGV (exit 139) at first try, when the edit also hit the SDF scene's mesh.
- convex cube cooking replaced by NULL: exit 1, 42 checks, 2 failed (`convex cube cooked`, CPU and GPU runs).

### Non-blocking note: `startIndex`
`convexTriangle.cuh:1872` `PxU32 startIndex = 0;`. Teeth on the GPU library (release, all 61 HIP objects rebuilt each time), with `-Xclang -no-enable-noundef-analysis` temporarily removed and `-fno-strict-aliasing` kept:
- with the initialization: 48 checks, 0 failed;
- without it: 4 failed (primitives on mesh, box and convex 0.0186 m above the CPU rest height: the frozen-at-first-touch symptom of the last round).
Restored (flag back, initialization kept): 48/0. So the initialization alone fixes the confirmed site; the flag stays for the other sites of the same pattern.

### Builds at 886c3529 (all exit 0)
- `./generate_projects.sh linux-gcc-rocm`, `make -C compiler/linux-gcc-rocm-release -j96`, `make -C compiler/linux-gcc-rocm-checked -j96`, from scratch.
- The configures in the table above.
- `jargon.py -C projects/PhysX/src --diff 064e809b..moat-port` and `--commits 064e809b..moat-port`: clean. `prose.py` on the three commit messages: clean.

### Corrections to commit messages, for the PR text (the published commits cannot change)
The PR description (or a squash message, which is a person's push) should state these instead of what the commits say:
- `e1080687` ([ROCm] Compile GPU code without strict aliasing or noundef arguments): its Test Plan's "1x1, 8x8 and 40x40 ... all 12 cases" came from a scratch program, not the parity snippet; at that commit the snippet has no mesh scene. The snippet's coverage of this fix is the primitives-on-mesh scene (one 8x8 grid mesh, box, convex, sphere, capsule) added in `064e809b`.
- `3ebaa20c` ([ROCm] Guard the bucket shifts of the single-warp radix sort): the scene its Test Plan cites was added later, in `064e809b`, and is named "aggregate pile" (staggered boxes dropped into one self-colliding aggregate), not "self-colliding aggregate stack".
- `a2667442` (Build lane masks from counts that can reach 32 without a shift by 32): the contact count in `convexTriangle.cuh` reaches exactly 32 (that file's `addContacts`, `:905-927`, counts only stored contacts); it can exceed 32 only in `cudaGJKEPA.cu` (`:950`, `:984`). The fix is needed in both.
- `0b3fd207` (Restore upstream's CUDA handling of a null module table entry): the Test Plan ran only the ROCm build. The CUDA branch is textually upstream 7d6f6ac8's; it was compiled with nvcc 12.8 on another host only for the earlier commits (see "CUDA compile check 2026-11-24"), and the recheck of this branch is owed (below).
- `886c3529`'s and `33c8a6c3`'s Test Plans were written to describe only what their own trees can run.

### Shared (non-HIP-guarded) source lines changed since 4764ca13, for the nvcc compile recheck
Line numbers at 886c3529. Everything from the list under "Porter round 2026-11-24, review fixes" (line numbers at 064e809b) still holds, at the same line numbers except in `SnippetGpuParity.cpp`: this round added lines to its new scenes, which moved from `:601-758`, `:844-885` to `:601-774`, `:860-901` (`git diff -U0 4764ca13 886c3529` hunks `+601,174` and `+860,42`). Of the other files this round touched, `convexTriangle.cuh` changed one line in place (`:1872`, below) and `cmakegpu/linux/CMakeLists.txt` is not seen by nvcc. The full list, with that one new line:
- `physx/source/gpucommon/src/CUDA/RadixSort.cuh:513-516`, `:639-642` (calls to `radixShiftLeft`, a plain shift on CUDA).
- `physx/source/gpusolver/src/CUDA/solverMultiBlock.cu:720-722`, `solverMultiBlockTGS.cu:325-327` (`highShift`/`maskHigh`).
- `physx/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu:130-134`, `:156-157`, `:483`, `:950`, `:984`, `:2997` (mask selects).
- `physx/source/gpunarrowphase/src/CUDA/convexTriangle.cuh:1003`, `:1855` (mask selects) and **`:1872` (new this round: `PxU32 startIndex = 0;`; nvcc may emit one extra move or none)**.
- `physx/source/cudamanager/src/CudaContextManager.cpp:521-536` (`#else` CUDA branch, upstream's text); `:474-477` move the `continue` under the HIP guard; `:488`, `:542` comment-only in the HIP branch.
- `physx/include/cudamanager/PxCudaTypes.h`: identical to upstream 7d6f6ac8.
- `physx/snippets/snippetgpuparity/SnippetGpuParity.cpp` (fork-only file): new scenes and checks, now `:601-774` and `:860-901`.
- Not seen by nvcc: `cuda_to_hip.h`; `cmakegpu/linux/CMakeLists.txt` and `PhysXGpu.cmake` (inside `IF(PX_USE_HIP)`); `CudaKernelWrangler.cpp` (comments only); `physx/CMakeLists.txt` (only the `PX_USE_HIP` amdhip64 block removed); root `.gitignore` removed.
Recheck to run: top-level configure with `-DPX_USE_HIP=OFF` at 886c3529, build `PhysXGpu` and `SnippetGpuParity`, and compare the objects that include `convexTriangle.cuh` (`convexMesh.cu`, `convexHeightfield.cu`, `convexMeshPostProcess.cu`, `femClothPrimitives.cu`) against 064e809b.

## Review 2026-11-24, delta 064e809b...886c3529 (linux-gfx942, reviewer, headless fleet job mi300c-20261124-045812)

Scope: `git diff 064e809b...886c3529` (three commits) and the notes this porter round wrote or reworded. One independent child review (all three commits, the notes, the lessons) plus an inline read; each finding below was fact-checked against the code. The child review configured scratch builds (top-level, standalone `cmakegpu`, a parent project using `add_subdirectory`, `-DCMAKE_HIP_FLAGS` precedence, reconfigure with a different `HIPFLAGS`); nothing ran on a GPU in the review.

Verdict: **changes requested, notes only.** The three fork commits (`acba89ef`, `33c8a6c3`, `886c3529`) have no findings, and neither do their messages. Both findings are in this file, and one of them is a lesson that would be promoted into the skill after review. By Jeff's instruction this was the last review cycle, so the round stops here for his decision. The validator did not run at 886c3529.

### Lessons in this file
1. **The reworded addendum to "masks and counters" ties the shift by 32 to the wrong condition** (the second paragraph of "Lesson for cuda-to-rocm: addendum to ..."). Old code (`4764ca13`, `solverMultiBlock.cu:720`, same at `solverMultiBlockTGS.cu:325`): `maskHigh = (1 << ((subWarpIndex + 1) * numThreadsPerBody)) - 1`, with `numThreadsPerBody` a power of two no larger than 32 and `subWarpIndex = threadIdx.x / numThreadsPerBody`. For the last sub-warp, `(subWarpIndex + 1) * numThreadsPerBody` is 32 for every slab count, not only when `numThreadsPerBody` is 32. Example with one slab (`numThreadsPerBody` 1), lane 31 on AMD: `maskHigh` 0, `maskLow` `0x7fffffff`, so `mask` is `0x80000001` (NVIDIA: `0x80000000`), and the ballot `& mask` near `:780` also picks up lane 0's body. Passing 128 partitions changes the symptom from a wrong mask on the last body of each warp to an empty mask on every body; it is not when the shift by 32 starts. The previous review's wording ("holds for the last sub-warp of each warp") was the accurate one. The same sentence also runs the second example on as a clause ("... each body's mask comes out 0 on AMD, and contact-count masks after polygon clipping ...") and says the clipping count "can exceed 32" without qualification; it can only in `cudaGJKEPA.cu:950/984`, while `convexTriangle.cuh`'s `addContacts` (`:906-927`) stops at the buffer. Suggested wording: "the solver's last-sub-warp mask `(1 << ((subWarpIndex + 1) * numThreadsPerBody)) - 1`, whose count is exactly 32 for the last sub-warp of every warp at any slab count: AMD gives that body a mask with lane 0's bit added, and once the scene-wide slab count makes `numThreadsPerBody` 32 (more than 128 partitions on one body at the default `gpuMaxNumPartitions` of 8) every body's mask is empty. A second example is the contact-count mask after polygon clipping in `cudaGJKEPA.cu`, where the clipper keeps counting past its buffer, so the count can exceed 32."

### Notes accuracy
2. "Shared (non-HIP-guarded) source lines changed since 4764ca13" says everything in the earlier list holds "same line numbers; this round changed no line above them in those files". That is false for `SnippetGpuParity.cpp`, which moved from `:601-758, :844-885` to `:601-774, :860-901` (`git diff -U0 4764ca13 886c3529` hunks `+601,174`, `+860,42`). The bullets under that sentence give the right numbers; only the sentence needs correcting before the nvcc recheck host uses it.

### What a pass needs
A notes-only porter edit of the two items above (no fork commit), then a review of that notes delta. Head 886c3529 would stay, and the validator would then run at it.

## Porter round 2026-11-24, notes only (linux-gfx942, headless fleet job mi300c-20261124-051935)

Answers "Review 2026-11-24, delta 064e809b...886c3529". No fork commit; `moat-port` and `head_sha` stay at **886c3529** (fresh clone of the fork in this job, `GIT_LFS_SKIP_SMUDGE=1`, clean tree at 886c3529).
1. Lesson addendum (second paragraph of "Lesson for cuda-to-rocm: addendum to ..."): the solver example and the clipping example are replaced by the review's suggested wording verbatim. Checked against the code: at 4764ca13 `solverMultiBlock.cu:720-722` and `solverMultiBlockTGS.cu:325-327` compute `maskHigh = (1 << ((subWarpIndex + 1) * numThreadsPerBody)) - 1`, `maskLow` likewise with `subWarpIndex`, `mask = maskHigh - maskLow`; `numThreadsPerBody = min(nextPow2(numSlabs), 32)` divides 32 and `subWarpIndex = threadIdx.x / numThreadsPerBody` (the warp index comes from `threadIdx.y`), so the last sub-warp's count is 32 at every slab count. With AMD's `shl` taking the low five bits, `maskHigh` is 0 there and `mask = -maskLow` is the sub-warp's lanes plus lane 0 (one slab: `0x80000001`; 16 threads per body: `0xffff0001`); at 32 threads per body `maskLow` is also 0 and the mask is empty. 886c3529 has the `highShift < 32` select at `solverMultiBlock.cu:721-722` (and TGS `:325-327` per the list below). The clipping clause now names only `cudaGJKEPA.cu` (`:950`, `:984`, `nbContacts` from `polyClip`); `convexTriangle.cuh`'s `addContacts` (`:906-927`) stores only `index < NUM_TMP_CONTACTS_PER_PAIR` and counts only stored contacts.
2. "Shared (non-HIP-guarded) source lines changed since 4764ca13": the false sentence is corrected. `git diff -U0 4764ca13 886c3529 -- SnippetGpuParity.cpp` gives hunks `+601,174` and `+860,42` (at 064e809b: `+601,158`, `+844,42`); `git diff --stat 064e809b 886c3529` touches only `SnippetGpuParity.cpp`, `cmakegpu/linux/CMakeLists.txt` and `convexTriangle.cuh` (one line in place, `@@ -1872 +1872 @@`), so every other line number in the list is unchanged.

## Review 2026-11-24, notes delta of the notes-only porter round (linux-gfx942, reviewer, headless fleet job mi300c-20261124-051935)

Scope: the notes-only porter commit on `port/PhysX` (`323f0164`), against the two findings of "Review 2026-11-24, delta 064e809b...886c3529" and the fork code at 4764ca13 and 886c3529. Fork head unchanged at 886c3529; no fork diff to review. One independent child review plus an inline check, both read-only, nothing built or run.

Verdict: **review passed.**
1. Lesson addendum: the text is the review's suggested wording verbatim and every claim holds. The kernels launch with `blockDim.x` 32 (`PxgCudaSolverCore.cpp:1200`, `PxgTGSCudaSolverCore.cpp:1208`, warp index from `threadIdx.y`), so the last sub-warp's count is 32 at any slab count; on AMD the mask is `-maskLow` (`0x80000001` at one slab, `0xffff0001` at 16 threads per body) and empty at 32 threads per body; `nbSlabs = max(1, ceil(nbPartitions / mNbMaxPartitions))` (`PxgContext.cpp:1617`) with `gpuMaxNumPartitions` defaulting to 8 (`PxSceneDesc.h:1070`), so 32 threads per body means more than 128 partitions; the clipping clause now names only `cudaGJKEPA.cu:950/984`, and `convexTriangle.cuh` `addContacts` (`:906-927`) counts only stored contacts. The run-on is gone.
2. Line-number sentence: `git diff -U0 4764ca13 886c3529` on `SnippetGpuParity.cpp` gives `+601,174` and `+860,42`; `git diff --stat 064e809b 886c3529` touches only the three files named; every bullet's cited lines were checked at 886c3529 and match.

Non-blocking, for the trunk session that promotes the lesson: the addendum's lead-in "The best examples of form (2) are the solver's last-sub-warp mask ..." is now followed by one example and a separate "A second example ..." sentence; write "The best example of form (2) is ..." when promoting.

## CUDA compile check at 886c3529 2026-11-24 (mi355 fleet host, nvcc 12.8, no GPU used)

Requested by Jeff: recheck the CUDA path of fork `886c3529` (the head; later commits ignored) with real nvcc against upstream 5.13 `7d6f6ac8`, which the branch merges, using the method of "CUDA compile check 2026-11-24". Nothing ran on a GPU (this job had none and used none), nothing in the fork changed, nothing was pushed, and the project's stage and lock were not touched. Builds lived in the job directory (`<job>/nvcc/`): `up/` at `7d6f6ac8`, `port/` at `886c3529`, and `same/`, one worktree built at `7d6f6ac8`, its build tree saved, then checked out at `886c3529` and rebuilt in place, so objects compare byte for byte with identical paths.

Verdict: **no CUDA regression.** Both commits build with no errors and no warnings beyond nvcc's deprecated-target note for sm_52/sm_70 (61 per build at both commits, one per `.cu`; every host file builds with `-Werror` and every `.cu` with `-Werror=all-warnings`). Compile commands are identical for every shared file, the CUDA module-load loop is upstream's line for line, the four changed host objects differ only in `__LINE__` immediates, and 54 of the 61 kernel TUs have identical SASS on all 8 targets. The other 7 differ only in kernels that use a shared-source edit, each traced to its source lines by recompiling with those lines reverted, and each edit computes the same values on NVIDIA. Nothing to fix.

### Toolchain and what was substituted

Same as the previous check: nvcc `/usr/local/cuda-12.8/bin/nvcc` "release 12.8, V12.8.93" (upstream pins CUDA 12.8, so no version substitution), gcc 13.3.0, CMake 3.31.6; packman fetched upstream's public `clang-physxmetadata` and `rapidjson` packages; freeglut v3.6.0 (built from source earlier, for the snippets' GLUT dependency) and NVIDIA's `cuda_nvdisasm-linux-x86_64-12.8.90` redist archive (the toolkit has no nvdisasm) were copied from the previous check's job directory. GPU targets are upstream's full list: SASS for sm_52, 70, 80, 86, 89, 90, 100 and 120, plus compute PTX (`PX_GENERATE_GPU_REDUCED_ARCHITECTURES=False`).

### Builds (all exit 0)

| Build | 7d6f6ac8 | 886c3529 |
|---|---|---|
| preset `linux-gcc`, release, full tree (SDK, PhysXGpu, snippets), `make -j64` | 890 objects, 61 `.cu.o` | 892 objects (plus `SnippetGpuParity.cpp.o` and its `ClassicMain.cpp.o`), 61 `.cu.o` |
| preset `linux-gcc`, checked | 899 objects, 61 `.cu.o` | 901 objects, 61 `.cu.o` |
| direct `cmake -S physx -B build-top -G Ninja -DPHYSX_PRESET=linux-gcc -DCMAKE_BUILD_TYPE=release -DPX_USE_HIP=OFF`, build, `cmake --install` | 632/632 steps | 632/632 steps |

- Compile commands (paths normalized): preset release 734 shared files and direct configure 590 files, **0 differ**. No `.cu` command carries `-fno-strict-aliasing`, `-no-enable-noundef-analysis`, `-include cuda_to_hip.h`, `PX_USE_HIP` or a `hip_compat` path, and none of the 1780 dependency files of the 886c3529 builds names `cuda_to_hip.h`, `hip_compat/` or `PxgSdfLinearGrid.h`. A fresh direct configure at 886c3529 with `HIPFLAGS=-DPX_ENV_CHECK` in the environment gives the same 590 commands with no trace of it: the two HIP-only compiler flags (inside `IF(PX_USE_HIP)` and a `$<COMPILE_LANGUAGE:HIP>` genex) and the HIPFLAGS handling never reach the CUDA build.
- All 87 `link.txt` files of the release build are identical (normalized); `SnippetGpuParity` adds one. `libPhysXGpu_64.so`: the same 210 exported symbols, no `hip` symbol, `NEEDED` only libstdc++/libm/libc, at both commits.
- Installed `PhysXConfig.cmake`: upstream `if(ON OR OFF)`, port `if((ON OR OFF) AND NOT "OFF")`. Both are true under CMP0012 NEW and false under an unset policy (`cmake -P`), so they always agree; a consumer project (`cmake_minimum_required(VERSION 3.16)`, `find_package(PhysX CONFIG)`) finds CUDAToolkit 12.8.93 with either install.

### Host objects

67 objects differ: 61 `.cu.o`, 4 host objects of the GPU library, and the 2 new snippet objects. 7 of the `.cu.o` (`broadphase`, `MemCopyBalanced`, `compressOutputContacts`, `sparseGridStandalone`, `constraintBlockPrePrep`, `preIntegration`, `preIntegrationTGS`) have identical cubins and PTX on every target and differ only in the host `.strtab` FILE symbol `tmpxft_<pid>_..._cudafe1.cpp`, nvcc's process-id temp name. It differs between any two builds, so it is build noise, not a port change.

The 4 host objects (`CudaContextManager.cpp.o`, `CudaKernelWrangler.cpp.o`, `PxgGeometryManager.cpp.o`, `PxPhysXGpuModuleLoader.cpp.o`): only `.text` sections differ (no data, rodata or symbol differences). In `objdump -dr` the instruction count is the same, and every differing instruction (82, 3, 7 and 3) differs only in one immediate that maps an upstream source line to its line at 886c3529 (checked mechanically against a line map of each file). These are `__LINE__` arguments to the error reporters, moved by the inserted HIP blocks.

**CUDA module-load loop.** Taking the `#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)` branches out of the loop at 886c3529 (`CudaContextManager.cpp:472-539`) leaves text identical to upstream's loop (`diff` empty, blank lines and comments included): `CUresult ret = CUDA_ERROR_UNKNOWN;`, `if (moduleTable[i])` around `moduleLoadDataEx`, and the error report for every result other than `CUDA_SUCCESS` and `CUDA_ERROR_NO_BINARY_FOR_GPU`. The `continue` is only in the HIP branch. With its code equal to upstream apart from line immediates, finding 1 of the previous check is closed.

### Kernels

Per-kernel `cuobjdump -sass` on all 8 targets (whitespace-normalized, encodings included): 54 TUs are identical in every kernel. Every TU that includes `RadixSort.cuh` (4), `atomic.cuh` (11) or `warpHelpers.cuh` (1) is among them. So are the `cudaNpCommon.h`, `PxgArticulationBlockData.h` and `sdfCollision.cuh` includers outside the 7 below, and those 7 are shown below to owe nothing to any header. The 7 TUs that differ, each attributed by recompiling it with the exact nvcc command line from a copy of the 886c3529 sources with only the named lines reverted (an unmodified copy reproduces the 886c3529 SASS exactly, so moving the source path changes nothing):

| TU, kernel(s) differing | Reverting only ... | gives SASS equal to upstream on 8/8 targets | Cause |
|---|---|---|---|
| `convexMeshMidphase.cu`, `triangleTriangleOverlaps` | the file | yes | BV32 root masks (as in the previous check: +2 `ISETP.GE.U32` and +2 `SEL` on sm_90, 2368 to 2376 instructions; the SDF kernels are identical) |
| `algorithms.cu`, `scanPerBlockKernel`, `scanPerBlockKernel4x4`, `addBlockSumsKernel4x4`, `radixFourBitCountPerBlockKernel` on sm_52 to sm_90 | the file | yes | `PXG_LOGICAL_WARP_SIZE = 32` for `warpSize` (as before; identical on sm_100 and sm_120, where nvcc folds `warpSize` itself) |
| `solverMultiBlock.cu`, `computeAverageSolverBodyVelocity` | the file (`:720-722` only differ) | yes | `maskHigh` select |
| `solverMultiBlockTGS.cu`, `computeAverageSolverBodyVelocityTGS` | the file (`:325-327` only differ) | yes | `maskHigh` select |
| `cudaGJKEPA.cu`, `convexConvexNphase_stage2Kernel`, `convexPlaneNphase_Kernel` | the file (`:130-134`, `:156-157`, `:483`, `:950`, `:984`, `:2997` only differ) | yes | mask selects |
| `convexMesh.cu`, `convexTrimeshNarrowphase` | `convexTriangle.cuh:1003`, `:1855` and `:1872` | yes | mask selects and `startIndex = 0`, each on its own |
| `convexHeightfield.cu`, `convexHeightfieldNarrowphase` | the same three lines | yes | same |

`convexMeshPostProcess.cu` and `femClothPrimitives.cu` also include `convexTriangle.cuh` but use neither `polyClip` nor `convexTriangleContactGen`; their SASS is identical. The SDF texture path is unchanged on CUDA: the same `TEX*` counts on sm_90 at both commits (`convexMeshMidphase` 399, `particleSystemMeshMidphase` 120, `femClothPrimitives` 90, `softbodySoftbodyMidPhase` 93).

**Mask selects.** Upstream's `(1 << n) - 1` reaches PTX as `shl.b32` by a register. PTX defines a shift by 32 or more to give 0, so upstream's mask is `0xffffffff` at n = 32 (and above), and the port's `n < 32 ? ... : -1` (`0xffffffff`) is the same value for every n. In PTX the change is `setp.lt.u32` plus `selp.b32` in place of the shift (seen in `computeAverageSolverBodyVelocity`). In SASS it is one `ISETP.GE.U32 ..., 0x20` (an `ISETP.GE` for the signed `int` counts in `cudaGJKEPA.cu`) and one `SEL` per mask, with register and scheduling changes around them. Per kernel on sm_90: PGS and TGS +1/+1; `convexPlaneNphase_Kernel` +1/+1; `convexConvexNphase_stage2Kernel` +10 `ISETP.GE.U32`, +2 `ISETP.GE`, +12 `SEL` (the `reduce` wrapper at `:483` is inlined at several call sites); `convexTrimeshNarrowphase` and `convexHeightfieldNarrowphase` +3/+3 and one more `BRA` from block layout on sm_70 and later. No atomic, shuffle, global/shared memory or barrier instruction is added or removed by the masks on any target. The total `SHFL` count per kernel is the same at both commits (sm_52/sm_90: `convexTrimeshNarrowphase` 193/389, `convexHeightfieldNarrowphase` 224/460, `convexConvexNphase_stage2Kernel` 967/1938), and so are the PTX `shfl.sync` and `atom.` counts of each TU (203/6, 243/6, 1233/24).

**`startIndex = 0` (`convexTriangle.cuh:1872`).** The PTX of `convexMesh.cu` and `convexHeightfield.cu` without the initialization and with it (all other lines at 886c3529) differs in 5 lines on sm_52 to sm_90 and 2 on sm_100/sm_120, apart from the TU hash in internal names. One `mov.u32 %r, 0` feeds the value that reaches `shfl.sync.idx` on the path that skips the lane-0 atomic, and the constant 0 is now shared with the shuffle's source-lane operand (sm_90, `convexTrimeshNarrowphase`):

```
popc.b32 %r228, %r1386;
mov.u32 %r1387, %r1230;        // new: %r1230 = 0
@%p32 bra $L__BB0_156;         // threadIdx.x != 0 skips the atomic, in both versions
atom.global.add.u32 %r1387, [%rd184], %r228;
...
$L__BB0_156:
shfl.sync.idx.b32 %r231|%p519, %r1387, %r1230, %r1234, %r1251;   // source lane 0
```

nvcc keeps the `@%p32 bra` guard around the lane-0 atomic in both versions (the clang miscompile this line fixes on HIP does not happen here), and the shuffle reads lane 0's value, which the initialization does not touch. The change is therefore inert on CUDA, and the SASS differences are ptxas register allocation and scheduling. On sm_52 `convexTrimeshNarrowphase` swaps one `NOP` for one `MOV` (3516 instructions both). On sm_70 to sm_120 instruction counts change by -8 to +16 (`IMAD.MOV`/`MOV`/`LEA`/`VIADD` mix), with no atomic, shuffle or memory instruction added. In `convexHeightfieldNarrowphase` the register allocation spills slightly less (2 `LDL` and 1 `STL` fewer on sm_86 to sm_120). The porter's expectation ("nvcc may emit one extra move or none") holds at the PTX level.

### Shared (non-HIP-guarded) source lines changed since 4764ca13, each checked

- `RadixSort.cuh:513-516`, `:639-642`: the text is the `radixShiftLeft` calls; SASS is identical in all 4 including TUs.
- `solverMultiBlock.cu:720-722`, `solverMultiBlockTGS.cu:325-327`: mask select, attributed and equivalent (above).
- `cudaGJKEPA.cu:130-134`, `:156-157`, `:483`, `:950`, `:984`, `:2997`: mask selects, attributed and equivalent (above).
- `convexTriangle.cuh:1003`, `:1855`, `:1872`: attributed and equivalent (above).
- `CudaContextManager.cpp:521-536`, `:474-477`, `:488`, `:542`: the CUDA loop is upstream's text, the `continue` and the two comments are in the HIP branch, and the object differs only in `__LINE__` immediates.
- `PxCudaTypes.h`: `git diff 7d6f6ac8 886c3529` is empty.
- `SnippetGpuParity.cpp` (fork-only file): compiles under `-Werror` in the release and checked builds.
- Not seen by nvcc: `cuda_to_hip.h` (in no dependency file); `cmakegpu/linux/CMakeLists.txt` and `PhysXGpu.cmake` (HIP branch only; the CUDA `ELSE()` branch is upstream's text, and the compile commands are identical); `CudaKernelWrangler.cpp` (object differs only in 3 `__LINE__` immediates); `physx/CMakeLists.txt` (link lines identical); root `.gitignore`.

### Reproduce

```
cd <clone>/physx
export PM_PACKAGES_ROOT=<dir> PM_CUDA_PATH=/usr/local/cuda-12.8 CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin CMAKE_PREFIX_PATH=<freeglut prefix>
./generate_projects.sh linux-gcc
make -C compiler/linux-gcc-release -j64
make -C compiler/linux-gcc-checked -j64
cuobjdump -sass -arch sm_90 <obj>.cu.o   # nvdisasm 12.8.90 on PATH
cuobjdump -xptx all <obj>.cu.o
```

Comparison scripts (job-local, not kept): the previous check's `cmpcmds.py`, `sasscmp.py` and `cubincmp.py`, plus `linecheck.py` (host disassembly differences against a source line map), `variant.sh` (one TU recompiled from an edited source copy with the build's exact nvcc command), `sass2.py`, `kdiff.sh`, `semops.py` (per-kernel SASS, opcode and memory/atomic/shuffle deltas) and `ptxdiff.sh`.
## Validation 2026-11-24 (linux-gfx942, validator, headless fleet job mi300c-20261124-051935)

Fork `moat-port` at **886c3529**, built from scratch in this job (fresh clone, clean tree; only untracked `physx/bin/` and `physx/lib/` build output afterwards). Device: the dispatcher's pin `HIP_VISIBLE_DEVICES=1`, one device visible. **This host is in CPX partition mode: the device is one 38-CU, 24 GiB (24574 MB VRAM) slice of an AMD Instinct MI300X (gfx942), not a whole card** (`amd-smi static`: `ACCELERATOR_PARTITION: CPX`, `NUM_COMPUTE_UNITS: 38`; `rocminfo` lists the slice as gfx942 with 38 CUs; `hipMemGetInfo` total 24.0 GiB).

### Toolchain
- `$FLEET_VENV/bin/python -m pip install --index-url https://nightly.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel]==10.2.0a20261022" "rocm-sdk-device-gfx942==10.2.0a20261022"` into the job venv; `rocm-sdk path --root` (run once) = `<job>/venv/lib/python3.11/site-packages/_rocm_sdk_devel`, `.info/version` 10.2.0, `hipconfig --version` 7.17.26422.
- **Compiler the builds invoked:** `CMAKE_HIP_COMPILER=<root>/lib/llvm/bin/amdclang++` (AMD clang 24.0.0git, `2792f8ceeb5a`) in both `compiler/linux-gcc-rocm-{release,checked}/CMakeCache.txt`, found through `rocm-sdk` on PATH with `ROCM_PATH`/`HIP_PATH` unset (the image has no `/opt/rocm`); every one of the 61 `.cu` entries in each `compile_commands.json` starts with that path and carries `--offload-arch=gfx942`, `-fno-strict-aliasing` and `-Xclang -no-enable-noundef-analysis`. Host `/usr/bin/c++` (gcc 13.3.0). `ldd libPhysXGpu_64.so` resolves `libamdhip64.so.7` from the same root.
- `$FLEET_VENV/bin` first on PATH (cmake, rocprofv3, rocm-sdk from the job venv), `MAX_JOBS=96`, `ulimit -c 0`, `timeout` around every GPU run. Snippet dependency prefix (freeglut, GLU, libglvnd) and packman cache copied from job mi300c-20261124-045812 (`.pc` files re-pointed); headless relink scripts as in the first porter round.

### Builds (all exit 0, from scratch)
```
cd physx
./generate_projects.sh linux-gcc-rocm
make -C compiler/linux-gcc-rocm-release -j96
make -C compiler/linux-gcc-rocm-checked -j96
```
61 HIP objects in each; `llvm-objdump --offloading libPhysXGpu_64.so`: 61 gfx942 code objects; 2522 / 2510 compiler warnings (same classes as recorded before), no errors. Also built with the other INSTALL_ROCm.md commands: the direct configure (`cmake -S physx -B <dir> -DPHYSX_PRESET=linux-gcc -DPX_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx942;gfx1100" -DCMAKE_BUILD_TYPE=release`, full build), the standalone GPU library (`cmake -S physx/source/compiler/cmakegpu ... --target PhysXGpu`), and the CPU-only preset (`-DPHYSX_PRESET=linux-gcc-cpu-only` under `env -i PATH=/usr/bin:/bin`, no ROCm or CUDA visible). These three write to `physx/lib/bin/`, separate from the `physx/bin/` the runs below used (checked by timestamps: nothing under `physx/bin/` changed after the snippet relinks).

### Parity harness (SnippetGpuParity, all 48 checks, every scene)
| Build | Runs | Result |
|---|---|---|
| release | 5 | 48 checks, 0 failed, each (about 10 s per run) |
| checked | 5 | 48 checks, 0 failed, each (about 330 s per run) |
| release, `LD_PRELOAD=<root>/lib/libamdhip64.so` | 1 | 48 / 0 |
| release, under `rocprofv3 --kernel-trace` | 1 | 48 / 0, 632,176 kernel dispatches |
| release, `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3` (experiment, below) | 5 | 48 / 0 each |

No skips: the harness has no skip path. 17 runs in all, 17 passes.

### Finding: three deformable checks change from run to run (the other 45 are bit-identical)
Compared value by value over all 17 runs (`agent_space/compare_parity.py`): 45 of 48 checks print the same value in every run, release and checked alike (free fall, stacks, articulation, all SDF and large-SDF checks, particles on SDF, primitives on mesh, aggregate pile, large-face convex, many slabs, contact overflow 20, PhysX errors 0). Three checks, all in the two deformable-volume scenes, differ in every build type:

| Check | Range over 17 runs | Limit |
|---|---|---|
| deformable: lowest vertex height (m) | -6.74481e-08 to -6.74336e-08 (3 distinct values) | > -0.02 |
| articulation lever: extra bend under deformable load (rad) | 0.190378 to 0.190741 (17 distinct values) | 0.06 to 0.5 |
| articulation lever: deformable load stays on the arm (m) | 0.840984 to 0.842479 (17 distinct values) | > 0.8 |

The values recorded in earlier rounds (0.190458 rad, 0.842092 m, -6.74481e-08 m) are inside these ranges. Investigation:
- **Kernel serialization does not remove it.** With `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3` (every launch and copy waits for the previous one), 5 release runs still give five different lever values (0.190378 to 0.190741) and two lowest-vertex values; the 45 other checks stay identical to the unserialized runs. So the variation comes from ordering inside kernels, not from a race between kernels, streams or copies, or from host timing.
- **The deformable path accumulates floats with atomics in upstream code.** Vertex deltas from contacts: `AtomicAdd(outputDeltas[...], delta*invMass...)` in `gpusimulationcontroller/src/CUDA/deformableAndParticleUtils.cuh:110-130`; rigid and articulation coupling: `atomicAdd(&solverBodyDeltaVel[...], ...)`, `atomicAdd(&tempDenom[...], denom)` and `atomicAdd(&artiLinks[...].mDeltaScale[...], denom)` in `rigidDeltaAccum.cu:627-835`; contact slots taken with `atomicAdd(writer.totalContactCount, 1)` in `gpunarrowphase/src/CUDA/softbodyPrimitives.cu`, so the contact order changes too. Float addition is not associative, so the sum depends on the order in which the waves reach the atomic. The port changed none of these lines (`git log 7d6f6ac8..886c3529 -- physx/source/gpusimulationcontroller` touches only the AMD base, the merge, a records cleanup and `96d8827e`, which re-enabled `atomicAddSpatialVector` and `setOverflowError` for HIP; the former's only caller is the latent `..._SECOND_MULTI2` kernel, per the first porter round). The rigid-body pipeline, which uses atomics only for integer counters and orders its contacts by partition, is the part that stays bit-identical.
- **Why this is not dismissed as noise:** the miscompiles fixed in this port (strict aliasing, noundef, the shifts by 32) produced large errors or crashes that showed in the rigid scenes; nothing like that shows here (the spread is 0.19% of the lever bend and 1.5 mm of the load height, against limits more than 2x away, and every rigid, SDF and mesh check is bit-stable). What is not established: that the CUDA build on NVIDIA varies by the same amount, because this host cannot run it. Upstream documents run-to-run determinism for rigid bodies; I found no statement either way for GPU deformables. **For Jeff:** I recorded the platform as completed on this evidence. Ways to close the question: (1) a CUDA run of these two scenes (5 runs) on an NVIDIA GPU, expected to vary similarly; or (2) rule that order-dependent float accumulation in upstream's deformable solver is accepted. If you rule the other way, set this platform back with `set-state PhysX linux-gfx942 validation-failed`; a code change would then belong to the porter, and none was made here.

### GPU snippets (traced with `rocprofv3 --kernel-trace`, relinked without rendering; 300 s timeout each)
- Release: HelloGRB 18,153 dispatches, DeformableSurface 33,954, DeformableVolume 93,827, DeformableVolumeAttachment 112,194, DeformableVolumeKinematic 104,145, DeformableSurfaceSkinning 35,359, DeformableVolumeSkinning 94,027, PBF 33,346, PBFMultiMat 28,625, Isosurface 32,113, RBDirectGPUAPI 18,538, DirectGPUAPIArticulation 116,765: **12/12 exit 0 with dispatches**. SDF as shipped: exit 0 with 0 dispatches, expected, because upstream sets `gUseGPU = false`. The local GPU copy (`gUseGPU = true`): exit 0, 53,939 dispatches. OmniPvdMultiGeom is not built in release (upstream drops it there).
- Checked: the same 12 exit 0 (18,154 to 116,766 dispatches), SDF as shipped exit 0 with 0 dispatches, the SDF GPU copy 53,940 dispatches, OmniPvdMultiGeom (`--omnipvdfile=<path>`) exit 0 with 443,245 dispatches: **14/14 GPU snippets pass** (13 as shipped plus the SDF GPU copy).
- HelloGRB release with `LD_PRELOAD=<root>/lib/libamdhip64.so`: exit 0 (registration binding, plan Risk 3), along with the preloaded parity run above.
- No log contains "Failed to load", "Failed to initialize cuda context" or "incompatible".

### Non-GPU regression set
- Non-GPU snippets, release: **55/56 exit 0**. The seven Vehicle snippets exit 0 with `--vehicleDataPath=physx/snippets/media/vehicledata`; LoadCollection exits 0 after `--generateExampleFiles` on `collection.bin`, `collection.xml`, `collectionA.bin collectionB.bin` and `collectionA.xml collectionB.xml`. **Skipped: SnippetProfilerConverter**, because it only prints its usage without a raw profiler capture (`--srcFile=`) as input; it is a CPU tool the port does not touch. Run without arguments, the Vehicle snippets, LoadCollection and ProfilerConverter exit 1 with usage or data-path messages.
- pvddom workflow CTest (`cmake -S physx/pvddom/workflow -B <dir> -G Ninja -DCMAKE_BUILD_TYPE=checked -DPX_GENERATE_GPU_PROJECTS=ON -DPX_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942`; `ctest -C checked`): **5/5 pass**. `ovd_sdk_gpu_capture` is a real PASS with `-DGPU=ON` ("Independent captures equal, ... GPU=ON"), not the skip.
- pvddom unit CTest (`cmake -S physx/pvddom -DPVDRUNTIME_ROOT=physx/pvdruntime -DPVDDOM_BUILD_TESTS=ON -DPVDDOM_BUILD_INSPECTION=ON -DPVDDOM_BUILD_TOOLS=ON`): **372/372 pass**.
- CPU-only preset with no ROCm or CUDA on PATH: configures and builds.

### CUDA gate
Not run here (no nvcc and no conda on this host). It is covered at this head by "CUDA compile check at 886c3529 2026-11-24" (mi355 fleet host, nvcc 12.8: no CUDA regression), which another job pushed while this validation ran.

### Content checks
- `jargon.py -C projects/PhysX/src --diff 7d6f6ac8..moat-port` and `--commits 7d6f6ac8..moat-port`: clean. `jargon.py --port PhysX` reports 14 hits, all in `ovphysx/` files that are identical to upstream 7d6f6ac8 (checked with `git diff --quiet 7d6f6ac8 886c3529` on the files). They show only because the fork's `main` mirror is at `4f2103c3` (ovphysx 0.5.11), behind upstream; this is Question A of the first porter round, still open.
- ROCm build documentation: `INSTALL_ROCm.md` (commands above, run as written).
- Integrity: `git -C projects/PhysX/src status --porcelain` shows only untracked `physx/bin/` and `physx/lib/`.

## Validation 2026-11-24 (linux-gfx1201, validator, headless fleet job gfx1201-20261124-062326-val-physx-w32-prompt)

Fork `moat-port` at **886c3529**, fresh clone (`GIT_LFS_SKIP_SMUDGE=1`) into `projects/PhysX/src`, built from scratch in this job; afterwards only untracked `physx/bin/` and `physx/lib/` (build output) in `git status --porcelain`. Device: the dispatcher's pin `HIP_VISIBLE_DEVICES=0`, one **AMD Radeon AI PRO R9700 (gfx1201, RDNA4, wave32, 64 CUs, 32 GB GDDR6)**. First run of this port on a wave32 GPU. Recipe as in "Validation 2026-11-24 (linux-gfx942 ...)" so the two compare line by line.

Verdict: **validation-failed.** Release passes everything, value for value like gfx942. The **checked build fails the parity harness in 5 of 5 runs** (16 to 26 of 48 checks; bodies fall through meshes and the ground, an articulation flies off), and the failure goes away whenever the host is made to wait for the GPU somewhere in the frame. It is an ordering race between host and GPU work that this card's timing exposes, and release is not immune: perturbing release's host timing with an interposer made it fail too (below). The cause is not located; no source change is proposed as a fix, only next steps for the porter. Nothing was pushed to the fork.

### Toolchain
- `$FLEET_VENV/bin/python -m pip install --index-url https://nightly.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel]==10.2.0a20261022" "rocm-sdk-device-gfx1201==10.2.0a20261022"` (no PyTorch involved, so no family wheel); `rocm-sdk path --root` (run once) = `<job>/venv/lib/python3.11/site-packages/_rocm_sdk_devel`, `.info/version` 10.2.0, `hipconfig --version` 7.17.26422. The image's venv layer (`venv-base-links`) also carries ROCm 10.1 device wheels and a 10.1 `_rocm_sdk_devel`; the job venv's site-packages come first, and nothing in either build tree names the 10.1 path (grep of `CMakeCache.txt`, `flags.make`, `link.txt`).
- **Compiler the builds invoked:** `CMAKE_HIP_COMPILER=<root>/lib/llvm/bin/amdclang++` (AMD clang 24.0.0git, `2792f8ceeb5a`) in both `compiler/linux-gcc-rocm-{release,checked}/CMakeCache.txt`, auto-discovered through `rocm-sdk` with `ROCM_PATH`/`HIP_PATH` unset (no `/opt/rocm` on this image). In each `compile_commands.json` all 61 `.cu` entries start with that path and carry `--offload-arch=gfx1201` (no other arch), `-fno-strict-aliasing` and `-Xclang -no-enable-noundef-analysis`; 0 of 829 host entries carry the noundef flag. Host `/usr/bin/c++` (gcc 13.3.0). `ldd libPhysXGpu_64.so` resolves `libamdhip64.so.7` from the same root, both configs.
- `$FLEET_VENV/bin` first on PATH, `MAX_JOBS=24`, `ulimit -c 0`, `timeout` around every GPU run. Snippet dependencies built from source into `<job>/deps/prefix`: libglvnd v1.7.0 (meson, `-Degl=false -Dgles1=false -Dgles2=false -Dheaders=true`), GLU glu-9.0.3 (meson, `-Dgl_provider=glvnd`), freeglut v3.6.0 (CMake, demos and static off). Packman cache `PM_PACKAGES_ROOT=<job>/deps/packman-repo`.

### Builds (all exit 0, from scratch)
```
cd physx
./generate_projects.sh linux-gcc-rocm
cmake -DCMAKE_HIP_ARCHITECTURES=gfx1201 compiler/linux-gcc-rocm-release   # the preset had already detected gfx1201; pinned explicitly
cmake -DCMAKE_HIP_ARCHITECTURES=gfx1201 compiler/linux-gcc-rocm-checked
make -C compiler/linux-gcc-rocm-release -j24
make -C compiler/linux-gcc-rocm-checked -j24
```
61 HIP objects each; `llvm-objdump --offloading libPhysXGpu_64.so`: 61 gfx1201 code objects. 2527 / 2526 compiler warnings, the classes recorded before (1049 deprecated-declarations, 732 unused-value, 342 unused-result, 312 attributes, 48 invalid-offsetof, 30 cuda-compat, 6 parentheses, 4 dangling-else, 2 shift-op-parentheses, 1 conversion-null), no errors. Also built: the direct configure from INSTALL_ROCm.md (`cmake -S physx -B <dir> -DPHYSX_PRESET=linux-gcc -DPX_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_BUILD_TYPE=release`, full build; writes `physx/lib/bin/`, `physx/bin/` untouched), the standalone GPU library (`cmake -S physx/source/compiler/cmakegpu ... --target PhysXGpu`, checked type, in the experiments below), and the CPU-only preset (`-DPHYSX_PRESET=linux-gcc-cpu-only` under `env -i PATH=/usr/bin:/bin`, outputs redirected to a scratch dir: configures and builds, `PX_USE_HIP=OFF`, `DISABLE_CUDA_PHYSX`).

### LDS, scratch and launches on gfx1201 (wave32 review points)
Per-kernel metadata of all 61 code objects (`.hip_fatbin` unbundled with `clang-offload-bundler`, `llvm-readelf --notes`), 488 kernels per config: **no kernel above 64 KB LDS.** Release, largest `.group_segment_fixed_size`: `sb_meshContactGenLaunch` 64496 bytes (same as the earlier gfx1100 build), `midphaseGeneratePairs` 61440, `sb_heightfieldContactGenLaunch` 60400, `convexHeightFieldMidphase` 57856, `convexCoreTetmeshNphase_Kernel32` 56128. Checked: largest `solveWholeIslandTGS` 45312. No run of release or checked reported a launch failure (`PhysX errors reported` 0 in every passing parity run; no "Failed to load", "Failed to initialize cuda context" or "incompatible" in any log).

**Checked device code is compiled at -O0** (not a finding about the race, see below): the port sets no `CMAKE_HIP_FLAGS_CHECKED`, so `.cu` commands in the checked build carry no `-O` (`flags.make`: `-std=gnu++17 --offload-arch=gfx1201 -fPIC -fno-strict-aliasing -Xclang -no-enable-noundef-analysis -ffp-contract=on -include cuda_to_hip.h ...`), and the code objects show it (e.g. `triangleTriangleCollision` 16016 bytes of scratch, 7 release kernels vs most checked kernels with dynamic stack). Upstream's CUDA checked build optimizes device code (nvcc's device default is -O3; `PHYSX_CUDA_FLAGS_CHECKED` adds `-lineinfo`). The same holds on gfx942 (same CMake). For the porter: give the HIP checked config `-O3` (and optionally `-gline-tables-only`) to match CUDA's checked semantics. This does not fix the failure: a checked GPU library rebuilt with `-DCMAKE_HIP_FLAGS_CHECKED=-O3` also failed (3 checks, one run).

Wavefront model: the shim's `ballotSync` takes `__lane_id() & ~31` (0 on wave32) and zero-extends the 32-bit ballot, `__shfl_*_sync` are pinned to width 32, `syncWarp` is fence + `wave_barrier` + fence; on wave32 each logical warp is a whole hardware wave. The harness evidence that this gives the wave64 results: in release, the 45 deterministic checks print here exactly what the gfx942 notes record at their stated precision (free fall 4.00586e-4 m, stacks 9.54951e-4 m, articulation 4.73016e-6 m, SDF dense/8/16/32-bit 2.06819e-4 / 7.45855e-5 / 1.04807e-3 / 8.6733e-4 m, large SDF 1.72706e-3 m, particles lowest 1.08333 m, contact overflow 20, largest cooked face 31, PhysX errors 0). The gfx942 job kept no full logs, so a digit-by-digit comparison beyond what its notes print was not possible.

### Parity harness (SnippetGpuParity, 48 checks)
| Build | Runs | Result |
|---|---|---|
| release | 5 (about 21 s each) | 48 / 0 each |
| release, 3 copies at once on the GPU | 3 | 48 / 0 each |
| release, `LD_PRELOAD=<root>/lib/libamdhip64.so` | 1 | 48 / 0 |
| release, under `rocprofv3 --kernel-trace` | 1 | 48 / 0, 632,218 kernel dispatches (gfx942: 632,176) |
| **checked** | **5** (223 to 380 s) | **16, 23, 26, 24, 20 failed** (exit 1 each) |
| checked, `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3` | 2 | 48 / 0 each |
| checked, `AMD_SERIALIZE_KERNEL=3` only | 1 | 48 / 0 |
| checked, `AMD_SERIALIZE_COPY=3` only | 1 | 48 / 0 |
| checked, `GPU_MAX_HW_QUEUES=1` | 2 | 17, 24 failed |

Release, value by value over its 12 runs (5 sequential, 3 concurrent, preload, traced, plus 2 shim runs below that passed; `agent_space/compare_parity.py`): **45 of 48 checks identical in every run; the same three deformable checks as on gfx942 vary, and only those:**

| Check | gfx1201, release runs | gfx942, 17 runs | Limit |
|---|---|---|---|
| deformable: lowest vertex height (m) | -6.74481e-08 to -6.74336e-08 (3 distinct) | -6.74481e-08 to -6.74336e-08 (3 distinct) | > -0.02 |
| articulation lever: extra bend under deformable load (rad) | 0.190497 to 0.190933 | 0.190378 to 0.190741 | 0.06 to 0.5 |
| articulation lever: deformable load stays on the arm (m) | 0.840010 to 0.842071 | 0.840984 to 0.842479 | > 0.8 |

The spread is the same order as on gfx942 (0.23% of the bend, 2.1 mm of the load height), consistent with the order-dependent float atomics in upstream's deformable solver recorded there. Checked runs that pass under serialization print the same 45 values as release.

What fails in checked (union over the 5 runs; each run a different subset): stacks (tower top 2.0 to 2.5 m low, speed 49 m/s), articulation (GPU vs CPU 112.9 to 122.4 m), all 8 SDF mesh checks (1.0 m), all 8 primitives-on-mesh checks (19.38 m, i.e. free fall for the whole run: no contacts at all), aggregate pile (lowest box -41.6 to -43.7 m), large-face convex (19.58 m), many slabs (19.63 m). Free fall, particles, deformable and contact overflow did not fail. No PhysX or HIP error is printed in these runs: the contacts are silently missing.

### Finding: an ordering race between host and GPU work (checked always, release when perturbed)
- **Not -O0, not wave-specific arithmetic.** Checked with an `-O3` GPU library still fails; release kernels are bit-stable run to run; any serialization makes checked pass with release's values.
- **Release is not immune.** An `LD_PRELOAD` interposer (`agent_space/shim/syncshim.cpp`, forwards every `hipMemcpy*Async`/`hipMemset*Async`/`hipModuleLaunchKernel` with `dlsym(RTLD_NEXT)`) that only counted calls and called `hipPointerGetAttributes` on each async copy's host pointer made release fail 1 of 48 (`articulation GPU vs CPU 112.931 m`, the exact value of checked run 1); 1 of 4 further perturbed release runs failed `contact overflow`, another `many slabs`. Release's normal timing hides the race on this card, not correctness.
- **Bisect with the interposer on checked** (each `hipStreamSynchronize(stream)` after the named call): after every `hipMemcpyHtoDAsync`: 18 failed (no fix). After every `hipMemcpyDtoHAsync`: 48 / 0. After every `hipModuleLaunchKernel` (that stream only): 48 / 0. After the device-to-host copies of only one caller group (backtrace-keyed): narrowphase result fetch (`PxgGpuNarrowphaseCore::fetchNarrowPhaseResults`, `fetchLostFoundPatchData`): 48 / 0; `PxgSimulationCore::gpuMemDmaBack` only: 48 / 0; broadphase/particle/soft-body/`copyDToHAsync` group only: 48 / 0; solver DMA-back group only: **GPU memory fault** (`HSA_STATUS_ERROR_MEMORY_FAULT`, faulting address 0x0, kernel `radixSortMultiBlockLaunch`, then error 700 at `PxgCudaUtils.h:55/61` `synchronizeStreams`). So it is not one missing synchronization at one site: a host wait anywhere in the frame closes the window, and the unsynchronized interleaving can also feed a null pointer to a broadphase kernel.
- Also seen: `GPU_ENABLE_WGP_MODE=0` (CU mode) on checked: memory fault in `sb_querySSContactReferenceCountLaunch` and the articulation check failed before it.
- **Not established:** whether the race is in upstream's stream/event usage (latent on NVIDIA and on gfx942 by timing), in the HIP shim's mapping of the driver API (events, `hipStreamWaitEvent`, mapped/write-combined pinned memory), or in the ROCm 10.2 runtime on gfx12. The files the bisect pointed at (`PxgNarrowphaseCore.cpp`, `PxgSimulationCore.cpp`, the solver cores) are unchanged from upstream 7d6f6ac8; `PxgCudaUtils.h` `synchronizeStreams` is upstream's record-then-wait on a reused event.
- **Proposed next steps for the porter** (not a fix): (1) on gfx942, run the release harness several times with the counting interposer (`SHIM_STATS=1 LD_PRELOAD=<shim>`) and the checked harness again: a failure there shows the race is not wave32 or RDNA specific; (2) trace one failing checked run with `rocprofv3 --kernel-trace --memory-copy-trace --hip-runtime-trace` and find the first frame whose narrowphase output count read on the host differs from a serialized run; (3) audit host reads of pinned/mapped buffers and host writes to pinned staging buffers against the event or stream sync that should precede them, and the shim's event and pinned-memory flags against CUDA semantics. Build: the interposer is `g++ -O2 -fPIC -shared -I<root>/include syncshim.cpp -o libsyncshim.so -L<root>/lib -Wl,-rpath,<root>/lib -lamdhip64 -ldl`; modes `SHIM_MODE=htod|dtoh|dtod|memset|launch`, `SHIM_DTOH_KEYS=<obj+off>|...`, `SHIM_CALLERS=1`, `SHIM_STATS=1`. (The source lives only in this job's `agent_space`; the description above is enough to rewrite it.)

### GPU snippets (traced with `rocprofv3 --kernel-trace`, relinked without rendering; 600 s timeout each)
- Release: HelloGRB 18,057 dispatches, DeformableSurface 33,954, DeformableVolume 93,827, DeformableVolumeAttachment 112,194, DeformableVolumeKinematic 104,112, DeformableSurfaceSkinning 35,359, DeformableVolumeSkinning 94,027, PBF 33,307, PBFMultiMat 28,625, Isosurface 32,113, RBDirectGPUAPI 18,538, DirectGPUAPIArticulation 116,765: **12/12 exit 0**; SDF as shipped exit 0 with 0 dispatches (upstream `gUseGPU = false`); SDF GPU copy (`gUseGPU = true`, local copy outside the fork) exit 0 with 53,879. OmniPvdMultiGeom is not built in release. Counts equal gfx942's except HelloGRB (18,153), Kinematic (104,145), PBF (33,346), SDF GPU (53,939), all within 0.6%.
- Checked: all 14 exit 0 (HelloGRB 12,670, DeformableSurface 33,202, DeformableVolume 93,828, Attachment 112,195, Kinematic 104,270, SurfaceSkinning 34,605, VolumeSkinning 94,028, SDF 0, SDF GPU 53,940, PBF 33,227, PBFMultiMat 28,626, Isosurface 32,114, RBDirectGPUAPI 17,830, DirectGPUAPIArticulation 108,514, OmniPvdMultiGeom 442,283). The snippets assert nothing; HelloGRB (gfx942 checked 18,154), RBDirectGPUAPI and DirectGPUAPIArticulation (gfx942 116,766) dispatched fewer kernels than on gfx942, consistent with contacts missing as in the harness.
- HelloGRB release with `LD_PRELOAD=<root>/lib/libamdhip64.so`: exit 0.

### Non-GPU regression set
- Non-GPU snippets, release: **55/56 exit 0**, the seven Vehicle snippets with `--vehicleDataPath=physx/snippets/media/vehicledata`, LoadCollection after `--generateExampleFiles` on `collection.bin`, `collection.xml`, `collectionA.bin collectionB.bin`, `collectionA.xml collectionB.xml`. **Skipped: SnippetProfilerConverter**, which only prints its usage without a raw profiler capture (`--srcFile=`); CPU tool the port does not touch (as on gfx942).
- pvddom workflow CTest (`cmake -S physx/pvddom/workflow -B <dir> -G Ninja -DCMAKE_BUILD_TYPE=checked -DPX_GENERATE_GPU_PROJECTS=ON -DPX_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201`; `ctest -C checked`): **5/5 pass**, `ovd_sdk_gpu_capture` a real PASS with `-DGPU=ON` ("Independent captures equal, ... GPU=ON").
- pvddom unit CTest (`cmake -S physx/pvddom -DPVDRUNTIME_ROOT=physx/pvdruntime -DPVDDOM_BUILD_TESTS=ON -DPVDDOM_BUILD_INSPECTION=ON -DPVDDOM_BUILD_TOOLS=ON`): **372/372 pass**.
- CPU-only preset with no ROCm or CUDA on PATH: configures and builds.
- Memory: nothing ran out of memory on this 32 GB card; nothing was shrunk.

### CUDA gate
Not run here (no nvcc on this host); covered at this head by "CUDA compile check at 886c3529 2026-11-24" (no CUDA regression).

### Content checks
- `jargon.py -C projects/PhysX/src --diff 7d6f6ac8..886c3529` and `--commits 7d6f6ac8..886c3529`: clean. (`--port PhysX` needs the fork's `main` in the clone; this single-branch clone lacks it.)
- Integrity: `git -C projects/PhysX/src status --porcelain` shows only untracked `physx/bin/` and `physx/lib/`; HEAD 886c3529.

## Porter round 2026-11-24, ordering race diagnosis (linux-gfx1201, headless fleet job gfx1201-20261124-083657-diag-physx-race-prompt)

Diagnosis of "Finding: an ordering race between host and GPU work" from the gfx1201 validation. **Verdict: a HIP runtime defect (ROCm 10.2 CLR), not the port's driver-API mapping and not upstream's stream or event usage.** A small async device-to-host copy (16 KiB or less) into host memory is run as a copy kernel whose writes are released only at device scope, so its bytes can still be in the GPU L2 cache when the next kernel on the stream runs; a later store to mapped host memory can reach the host first. PhysX's read-back pattern (copy, then the one-thread `bpSignalComplete` kernel stores 1 to a mapped flag, host spins on the flag and reads the copied data with no runtime call) therefore reads stale data. CUDA guarantees the copy has landed before the next stream operation runs. Workaround committed (HIP-only, changes no result): a system-scope fence in `bpSignalComplete`. Deferral `hip-small-d2h-copy-host-visibility` registered with the issue draft below for Jeff to file. Also committed: `-O3` for the HIP checked and profile configurations, and `cuCtxSynchronize` mapped to `hipDeviceSynchronize`.

### Questions for Jeff
- **File the ROCm issue** drafted under "Issue draft" below (rocm-systems, labels `bug`, `project: clr`). Nothing was posted.
- **The workaround touches upstream's `broadphase.cu`** (6 lines, under `#if defined(__HIP_PLATFORM_AMD__)`, so nvcc sees no change). It is confined to the HIP build and changes no value, so it was committed per the round's instructions; if you would rather carry it only until the runtime is fixed, say so and it can be reverted when a fixed ROCm ships.

### Toolchain (this job)
- `$FLEET_VENV/bin/python -m pip install --index-url https://nightly.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel]==10.2.0a20261022" "rocm-sdk-device-gfx1201==10.2.0a20261022"`; `rocm-sdk path --root` (once) = `<job>/venv/lib/python3.11/site-packages/_rocm_sdk_devel`, `.info/version` 10.2.0, `hipconfig --version` 7.17.26422. `therock_manifest.json`: TheRock `f93a7c76`, rocm-systems pin **`1a455a9bc1441b9762a50c6a8188534536e92025`** (2026-10-20) plus the patch `0001-rocr-honor-build-time-hotswap-default.patch`; all HIP runtime source cited below is at that commit. Driver amdgpu 6.19.14.31400100, kernel 6.11.0-26.
- **Compiler the builds invoked:** `CMAKE_HIP_COMPILER=<job>/venv/.../_rocm_sdk_devel/lib/llvm/bin/amdclang++` (AMD clang 24.0.0git `2792f8ceeb5a`) in release, checked and debug `CMakeCache.txt`; all 61 `.cu` entries of each `compile_commands.json` start with it and carry `--offload-arch=gfx1201` only. Host `/usr/bin/c++` (gcc 13.3.0).
- `HIP_VISIBLE_DEVICES=0` (the dispatcher's pin), `$FLEET_VENV/bin` first on PATH, `MAX_JOBS=24`, `ulimit -c 0`, `timeout` around every GPU run. Fork cloned from the validator's local clone (`GIT_LFS_SKIP_SMUDGE=1`, origin reset to `AMD-Ecosystem/PhysX`, `moat-port` at 886c3529), snippet dependencies and packman cache copied from the validator's job. Build recipe as in INSTALL_ROCm.md (`./generate_projects.sh linux-gcc-rocm`, `cmake -DCMAKE_HIP_ARCHITECTURES=gfx1201 compiler/linux-gcc-rocm-<cfg>`, `make -C ... -j24`).

### Step 1: the driver-API mapping against CUDA (`cuda_to_hip.h`, `CudaContextManager.cpp`)

| CUDA call as PhysX uses it | HIP mapping | Same semantics? (HIP source at 1a455a9b) |
|---|---|---|
| `cuEventCreate(CU_EVENT_DISABLE_TIMING)` | `hipEventCreateWithFlags(hipEventDisableTiming)` | Yes. New in this CLR: consecutive records of the same timing-disabled event on one stream with no work between are coalesced (`hip_event.cpp:217-226`, `rocvirtual.cpp:5352-5360`); the skipped record reuses the previous barrier's signal through `AttachHwEvent` (`rocvirtual.cpp:2945-2955`), so query, synchronize and waits stay correct. (I first misread `AttachHwEvent` as an empty stub because a `sed` range cut it off; it is complete.) |
| `cuEventRecord` | `hipEventRecord` | Yes. A barrier packet with a system-scope release (`recordCommand` uses `kCacheStateInvalid` unless `hipEventDisableSystemFence`, which PhysX never passes). |
| `cuEventQuery` / `cuEventSynchronize` | `hipEventQuery` / `hipEventSynchronize` | Yes. **ROCm/rocm-systems#15025** (use of a destroyed recording stream; fix `99ad3502`, 2026-11-18, not in the pin, verified with `git merge-base --is-ancestor`) needs an event queried after the stream it was last recorded on is destroyed. PhysX destroys streams only at shutdown and in `PxgSDFBuilder` (cooking), which records no event on its stream; every queried or synchronized event belongs to a core's long-lived stream. Not triggered. |
| `cuStreamWaitEvent` | `hipStreamWaitEvent` | Yes. `Event::streamWait` captures the event's current record at call time (`hip_event.cpp:153-185`) and enqueues a marker that waits on that record's signal; re-recording the event later does not change what an already-enqueued wait waits for, as in CUDA. An unrecorded event is a no-op in both. `synchronizeStreams`' reused event is therefore fine. A wait on a record that has no HSA signal would be skipped with `Waiting event(%p) doesn't have a HSA signal!` (`rocvirtual.cpp:2868-2875`); a failing checked run with `AMD_LOG_LEVEL=1` printed no such line (stderr empty). |
| `cuStreamCreate(CU_STREAM_NON_BLOCKING)`, `cuStreamCreateWithPriority` | `hipStreamCreateWithFlags`, `hipStreamCreateWithPriority` | Yes. |
| `cuCtxCreate(LMEM_RESIZE_TO_MAX \| SCHED_BLOCKING_SYNC \| MAP_HOST)` | `hipCtxCreate` | Yes for PhysX: returns the device's primary context and stores the flags (`hip_context.cpp:155-169`). |
| `cuCtxSynchronize` | was `hipCtxSynchronize` | **No.** `hipCtxSynchronize` returns `hipErrorNotSupported` and waits for nothing (`hip_context.cpp:325-328`). PhysX calls it only under `PX_STOMP_ALLOCATED_MEMORY` (PX_DEBUG builds, `CudaContextManager.cpp:872/876`), so it is not the checked/release race. Fixed: mapped to `hipDeviceSynchronize`. |
| `cuMemHostAlloc(PORTABLE)` / `(PORTABLE\|DEVICEMAP)` | `hipHostMalloc(hipHostMallocPortable[\|Mapped])` | Differs but not relevant: HIP returns fine-grained (coherent) memory only for flags 0, Coherent, Mapped, NumaUser or Uncached (`hip_memory.cpp:479-492`), so PhysX's Portable-only pinned heap is non-coherent; all CUDA page-locked memory is coherent. The reproducer below shows the race with every flag combination, coherent included, so this is not the cause; left unchanged. |
| `cuMemcpyDtoHAsync` from device to pinned host memory | `hipMemcpyDtoHAsync` | **No (runtime).** Copies of at most `GPU_FORCE_BLIT_COPY_SIZE` (16 KiB) to or from host memory take the shader-copy path (`useShaderCopyBufferPath`, `rocblit.cpp:2785-2796`). For an async copy the runtime adds no system scope ("For non blocking rely on the release scope issued by synchronization packet", `rocblit.cpp:3324-3334`; unchanged on develop `985bf6ca`, 2026-11-24), so the copy is complete for the stream while its bytes may still be in L2. In CUDA, completion means the bytes are in host memory. |
| `cuMemcpyHtoDAsync`, `cuMemsetD*Async`, synchronous copies and memsets | 1:1 | Yes (synchronous forms are followed by `cuStreamSynchronize` in the wrappers). The validator's "synchronize after every host-to-device copy" did not help, consistent with the defect being on the device-to-host side. |
| `cuLaunchKernel` | `hipModuleLaunchKernel` | Yes. |

### Step 2: the failing operation pair in PhysX
Temporary instrumentation in the local build only (`agent_space/diag-stale.patch`, reverted before any commit): `CudaCtx::memcpyDtoHAsync` logs (destination, size, stream, caller); a 1x1x1 launch with one pointer argument (the `bpSignalComplete` launches) attaches that stream's logged copies to the flag; after `spinWait` succeeds, the destinations are snapshotted, `cuStreamSynchronize(stream)` runs, and any byte that changed is reported. A change means the host was about to read bytes whose producing copy had not landed.

- Checked at -O0 (as 886c3529 builds it), one run: **5 stale of 15,421 checked copies**, all in `PxgSimulationCore::gpuMemDmaBack` on the simulation core's stream: the third copy (`deactiveArray <- mDeactivateBuffer`, `sizeof(PxU32)*numActiveBodies`, call at `libPhysXGpu_64.so+0x270f8d5`) 4 bytes twice and 48 bytes twice, the second copy (`activateArray <- mActivateBuffer`, `+0x270f851`) 48 bytes once (identified from the disassembly: the call before them passes size 0x98 = `sizeof(PxgSimulationCoreDesc)`). **The two operations:** the async device-to-host copy of the activate/deactivate lists (a runtime copy kernel, under 16 KiB) and the later `bpSignalComplete` kernel's store to `mEventMapped`, both on `PxgSimulationCore::mStream`; the host reads the lists in `PxgSimulationCore::syncDmaback` right after `spinWait(*mEventMapped)`. What was supposed to order them: stream order, i.e. CUDA's guarantee that the copy has landed in host memory before the next kernel on the stream runs.
- The same run still failed 6 checks, because the instrumentation only repairs copies attached to a flag: kernels on these streams also write mapped host memory directly, which the same L2 behaviour leaves unflushed. With a system-scope fence in `bpSignalComplete` (the committed workaround) and the instrumentation still in, checked at -O0: **3/3 runs 48/0, 0 stale of 42,210 checked copies.**
- `GPU_FORCE_BLIT_COPY_SIZE=0` (small device-to-host copies go to SDMA instead of a copy kernel) on the validator's unmodified checked binary: **2/2 runs 48/0**, values identical to release.
- Release with the instrumentation and the counting interposer (`SHIM_STATS=1 LD_PRELOAD=libsyncshim.so`), no fence: 5/5 48/0, 0 stale of 74,730; release timing on this card does not open the window at these sites today. Unfixed release under the same interposer, 20 runs: 20/20 48/0 (the validator saw failures in a few of its perturbed runs). Unfixed checked at -O3: 1 of 13 runs failed (`many slabs` 19.63 m, 2 checks). Unfixed checked at -O0: failed in every run (validator 5/5, 16 to 26 checks; this job 2/2, 7 and 6 checks).
- A rocprofv3 trace was not needed once the instrumentation named the copy and the reproducer isolated the runtime path.

### Step 3: standalone reproducer (no PhysX)
`agent_space/repro/d2h_visibility.cpp` (in the issue draft below) and the more general `agent_space/repro/flag_after_copy.cpp` (flags, sizes, signal kinds). Built with `hipcc -O2 --offload-arch=gfx1201`. Fill a device buffer with the iteration number, `hipMemcpyDtoHAsync` it into pinned host memory, then signal the host on the same stream; the host spins on the flag and checks the buffer.

| Signal after the copy (`flag_after_copy`, 200,000 iterations unless noted) | 512 B | 4 KiB | 64 KiB / 1 MiB (SDMA) |
|---|---|---|---|
| one-thread kernel stores 1 to a mapped flag (PhysX's pattern) | 12 stale | 15 stale | 0 / 0 (20,000 it.) |
| `hipStreamWriteValue32(flag, 1)` | 199,212 stale | 199,113 stale | 0 / 0 (20,000 it.) |
| `hipEventRecord` (default flags) then the kernel | 0 | 0 | |
| `__threadfence_system()` in the kernel before the store | 0 | 0 | |
| kernel, `GPU_FORCE_BLIT_COPY_SIZE=0` | 0 | 0 | |
| `hipStreamWriteValue32`, `GPU_FORCE_BLIT_COPY_SIZE=0` (20,000 it.) | 0 | | |

20,000-iteration sweep with the kernel signal over host flags `Portable`, `Portable|WriteCombined`, `NonCoherent`, `Portable|Coherent`, `Default` at 512 B / 4 KiB / 16 KiB / 64 KiB: stale reads (1 to 7) at 512 B to 16 KiB with every flag, coherent included; 0 at 64 KiB with every flag. Some stale reads had the first word new and the last word old (the copy only partly visible). With `hipStreamWriteValue32` the buffer usually still holds its initial 0xffffffff when the flag is seen. `submitBatchMemoryOperation` (`rocvirtual.cpp:4398-4404`) describes this hazard ("data left in this agent's L2 by a preceding agent-scope dispatch ... has not drained yet, so the value overtakes it") and counts on a system-scope acquire on the later packet; on gfx1201 that does not drain the earlier copy's dirty lines. Not run on gfx942 (no such GPU here; MI300X passed every PhysX run, so its cache policy for host memory evidently differs). The two-stream event pattern the round suggested was not needed: the failing pattern is a single stream plus a mapped flag.

### Decision
A HIP runtime defect. Smallest HIP-only workaround, committed: `__threadfence_system()` before `*signal = 1` in `bpSignalComplete` (`broadphase.cu`, under `__HIP_PLATFORM_AMD__`). Every host read of GPU results that is not preceded by a runtime synchronization goes through one of the three `spinWait` sites (`PxgCudaBroadPhaseSap::gpuDMABack`, `PxgCudaSolverCore`/`PxgTGSCudaSolverCore::syncDmaBack`, `PxgSimulationCore::syncDmaback`), and all three use this kernel. Cost: one system-scope release (an L2 write-back) in a one-thread kernel, three times per step; no host wait. Considered and rejected: forcing `GPU_FORCE_BLIT_COPY_SIZE=0` (process-wide, changes copy performance for the application), recording a system-fence event after every device-to-host copy in the compatibility header (a barrier per copy, 22,716 copies per harness run), mapping `Portable` to `Coherent` (does not help, see the reproducer). This is not blanket synchronization: no host wait is added, and the fence restores exactly the guarantee CUDA's stream order gives at these three points.

### Commits on `moat-port` (above 886c3529; pushed, new head 42d4c033)
1. `8f87766f` `[ROCm] Write back the GPU cache before signalling completion to the host` (`broadphase.cu`)
2. `eb3d5004` `[ROCm] Optimize device code in the checked and profile HIP builds` (`cmakegpu/linux/CMakeLists.txt`: `CMAKE_HIP_FLAGS_CHECKED`/`_PROFILE` default to `-O3` when the caller has not set them; CUDA's checked and profile flags are identical, so profile had the same gap)
3. `42d4c033` `[ROCm] Map cuCtxSynchronize to a device-wide wait` (`cuda_to_hip.h`)

### Results at the new head (Radeon AI PRO R9700, gfx1201)
| Build | Runs | Result |
|---|---|---|
| checked (-O3 + fence) | 10 | 48/0 each, about 24 s per run (was 220 to 380 s at -O0) |
| release + fence, `SHIM_STATS=1 LD_PRELOAD=libsyncshim.so` | 10 | 48/0 each |
| release + fence | 3 | 48/0 each |
| checked built at -O0 (`-DCMAKE_HIP_FLAGS_CHECKED=-O0`) + fence | 10 | 48/0 each (about 221 s per run; unfixed at -O0: failed every run, 7 of 7) |
| debug GPU library (`make ... PhysXGpu`) | build | exit 0; both `cuCtxSynchronize` calls resolve to `hipDeviceSynchronize` |

Across the 23 checked and release runs: 45 of 48 checks identical in every run and equal to the validator's unfixed release values (`compare_parity.py` against its `relpar-1.log`); the same three deformable checks as on gfx942 vary (bend 0.190478 to 0.190902 rad, load height 0.840338 to 0.84203 m, lowest vertex 3 distinct values). The fence changes no result.

### Owed
- **CUDA compile check at the new head**: the delta is preprocessor-guarded (`__HIP_PLATFORM_AMD__`), the HIP branch of the CMake, and the HIP-only compatibility header, so nvcc's input is unchanged; it still needs the nvcc build on the mi355 host before the round is presented.
- Review of the delta, then revalidation on linux-gfx942 (head moved) and linux-gfx1201.
- Run `agent_space/repro` (copied into the issue draft) on gfx942 to say in the issue whether CDNA3 is affected.

### Issue draft for Jeff (ROCm/rocm-systems; labels `bug`, `project: clr`; nothing posted)

**Title:** `[CLR] hipMemcpyDtoHAsync <= 16 KiB not host-visible before later stream work`

**Body:**

On gfx1201, an async device-to-host copy of 16 KiB or less followed on the same stream by work that signals the host (a kernel storing to mapped host memory, or `hipStreamWriteValue32`) lets the host observe the signal before the copied bytes. With `hipStreamWriteValue32` the host sees the flag before the copy's data in over 99% of iterations; with a one-thread kernel store, in about 1 of every 10,000 to 17,000. CUDA's stream order guarantees the copy has landed in host memory before the next operation on the stream runs, and applications rely on it (NVIDIA PhysX's GPU read-back spins on a mapped flag written by a kernel after its device-to-host copies, then reads the copied data; on gfx1201 it reads stale results and simulation goes wrong).

Cause as far as I can tell from the source: copies of at most `GPU_FORCE_BLIT_COPY_SIZE` (16 KiB) with a host-memory endpoint use the shader copy path (`KernelBlitManager::useShaderCopyBufferPath`), and for an async copy `KernelBlitManager::copyBuffer` adds no system scope ("For non blocking rely on the release scope issued by synchronization packet", `rocblit.cpp`), so the copy kernel's writes can remain in L2 after the copy is complete for the stream. `submitBatchMemoryOperation` describes the same hazard and relies on a system-scope acquire on the later packet, which does not write back the earlier dispatch's dirty lines. Forcing SDMA (`GPU_FORCE_BLIT_COPY_SIZE=0`), copies above 16 KiB, a `hipEventRecord` between the copy and the signal, or `__threadfence_system()` in the signalling kernel all make the problem disappear. The host-memory coherence flag does not matter (fails with `hipHostMallocDefault`, `Coherent`, `NonCoherent`, `Portable`, `WriteCombined`).

Reproducer (`hipcc -O2 --offload-arch=gfx1201 d2h_visibility.cpp -o d2h_visibility`):
```cpp
// hipMemcpyDtoHAsync (<= 16 KiB) followed on the same stream by a host-visible signal:
// the host can observe the signal before the copied bytes.
// usage: d2h_visibility <kernel|writevalue> [bytes=512] [iters=100000]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::printf("%s failed: %s\n", #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

__global__ void fill(unsigned* d, unsigned n, unsigned v) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = v;
}
__global__ void signal(unsigned* flag) { *flag = 1; }

int main(int argc, char** argv) {
  const bool writeValue = argc > 1 && !std::strcmp(argv[1], "writevalue");
  const size_t bytes = argc > 2 ? std::strtoull(argv[2], nullptr, 0) : 512;
  const int iters = argc > 3 ? std::atoi(argv[3]) : 100000;
  const unsigned n = unsigned(bytes / 4);

  hipStream_t s;
  CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  unsigned *d, *h, *flag, *dflag;
  CHECK(hipMalloc((void**)&d, bytes));
  CHECK(hipHostMalloc((void**)&h, bytes, hipHostMallocDefault));
  CHECK(hipHostMalloc((void**)&flag, 4, hipHostMallocMapped));
  CHECK(hipHostGetDevicePointer((void**)&dflag, flag, 0));
  std::memset(h, 0xff, bytes);

  int stale = 0;
  for (int it = 1; it <= iters; ++it) {
    *(volatile unsigned*)flag = 0;
    fill<<<(n + 255) / 256, 256, 0, s>>>(d, n, unsigned(it));
    CHECK(hipMemcpyDtoHAsync(h, d, bytes, s));
    if (writeValue) CHECK(hipStreamWriteValue32(s, dflag, 1, 0));
    else signal<<<1, 1, 0, s>>>(dflag);
    (void)hipStreamQuery(s);  // submit the work, as the application does
    while (*(volatile unsigned*)flag == 0) {}
    if (((volatile unsigned*)h)[0] != unsigned(it) || ((volatile unsigned*)h)[n - 1] != unsigned(it)) ++stale;
  }
  CHECK(hipStreamSynchronize(s));
  std::printf("%s bytes=%zu iters=%d stale=%d\n", writeValue ? "writevalue" : "kernel", bytes, iters, stale);
  return stale != 0;
}
```
Output on the R9700 (200,000 iterations each):
```
kernel bytes=512 iters=200000 stale=23
kernel bytes=4096 iters=200000 stale=12
writevalue bytes=512 iters=200000 stale=199296
kernel bytes=65536 iters=200000 stale=0
writevalue bytes=65536 iters=200000 stale=0
GPU_FORCE_BLIT_COPY_SIZE=0: writevalue bytes=512 iters=200000 stale=0
```

Expected: `stale=0` in every case (as with copies above 16 KiB, or with `GPU_FORCE_BLIT_COPY_SIZE=0`).

Environment: Radeon AI PRO R9700 (gfx1201), TheRock ROCm 10.2.0 wheels `10.2.0a20261022` (rocm-systems `1a455a9b`), HIP 7.17.26422, amdgpu 6.19.14.31400100, Linux 6.11.0-26, Ubuntu 24.04 userspace. The copy-path code is unchanged on `develop` as of 2026-11-24 (`985bf6ca`).

### Content checks
- `jargon.py -C projects/PhysX/src --diff 7d6f6ac8..moat-port` and `--commits 7d6f6ac8..moat-port`: clean. `jargon.py --port PhysX`: the 14 known hits in upstream's ovphysx text (Question A of the first porter round), none in our delta. `prose.py` on the three commit messages: clean.
- Integrity: `git -C projects/PhysX/src status --porcelain` shows only untracked build output; the diagnostic patch (`agent_space/diag-stale.patch`) was reverted before the commits.

## Lesson for cuda-to-rocm: a GPU-written host flag can arrive before an earlier small device-to-host copy (belongs in `faults-memory-streams.md`)
CUDA code often reads results back without a runtime synchronization: async device-to-host copies on a stream, then a tiny kernel (or a stream write-value) that sets a flag in mapped host memory, and the host spins on the flag and reads the copied data. CUDA's stream order makes the copied bytes land in host memory before the next stream operation runs. HIP (seen on ROCm 10.2, gfx1201) runs copies of 16 KiB or less that touch host memory as a copy kernel whose writes are released only at device scope, so they can still be in the GPU L2 cache when the flag reaches host memory: the host reads stale data, rarely with a kernel-written flag (about 1 in 10,000) and almost always with `hipStreamWriteValue32`. Host-memory coherence flags make no difference. Symptoms are timing-dependent wrong results that vanish with any `hipStreamSynchronize`, with `AMD_SERIALIZE_COPY`, or with `GPU_FORCE_BLIT_COPY_SIZE=0` (forces SDMA); unoptimized device code tends to expose it. Fix in the port: `__threadfence_system()` in the signalling kernel before the flag store (or record a default-flag event between the copies and the signal). Find candidates by searching for host spin loops on `volatile` pinned memory and for `cuStreamWriteValue`. Evidence: "Porter round 2026-11-24, ordering race diagnosis" above.

## Review 2026-11-24, delta 886c3529...42d4c033 (linux-gfx942, reviewer, headless fleet job mi300c-20261124-130700-review2-physx-prompt)

Scope: `git diff 886c3529...42d4c033` on `moat-port` (3 commits, 3 files, +18/-1): the `bpSignalComplete` fence (`8f87766f`), `-O3` for the HIP checked/profile configurations (`eb3d5004`), `cuCtxSynchronize` -> `hipDeviceSynchronize` (`42d4c033`). pr-review skill, every finding re-checked against the code; HIP runtime source read at the rocm-systems pin `1a455a9bc1441b9762a50c6a8188534536e92025` (confirmed as `submodules[rocm-systems].pin_sha` in this job's `_rocm_sdk_devel/share/therock/therock_manifest.json`, TheRock `f93a7c76`, `10.2.0a20261022`; blobless sparse clone of `projects/clr`). **Verdict: review-passed.** No defect in the delta; the items below are accuracy corrections to the records and the issue draft, none needs a fork change.

### The fence (`broadphase.cu:2625-2630`)
- **Correct for the documented mechanism.** Runtime source at the pin, re-read: `KernelBlitManager::useShaderCopyBufferPath` (`rocblit.cpp:2769-2796`) sends copies of at most `sdmaCopyThreshold_` = `GPU_FORCE_BLIT_COPY_SIZE` (default 16, `flags.hpp:179`) KiB with a host endpoint to the shader path; `copyBuffer` (`rocblit.cpp:3323-3341`) calls `addSystemScope()` only when the source is host memory or the copy is blocking, so an async device-to-host blit dispatch keeps the default header. With `AMD_OPT_FLUSH=1` (default, `flags.hpp:171`; `device.cpp:1649`) that default is agent-scope release (`rocvirtual.cpp:2448-2458`; on gfx12 system acquire with agent release, `sysAcquireAgentReleaseHBits`). So the copy is complete for the stream at device scope only, as the notes say. The fence then writes the L2 back before the flag store. ISA from this job's amdclang++ (`hipcc -O3 --cuda-device-only -S`): gfx942 and gfx950 `buffer_wbl2 sc0 sc1; s_waitcnt; buffer_inv sc0 sc1`, gfx1201 `global_wb scope:SCOPE_SYS; s_wait_storecnt 0; global_inv scope:SCOPE_SYS`. Both write back the whole L2 (not only this wave's lines), so lines left by the earlier copy kernel are covered.
- **Minimal enough.** `__threadfence_system()` is seq_cst, so it also emits the acquire half (`buffer_inv` / `global_inv`), which publishing to the host does not need. A release-only store (`__hip_atomic_store(signal, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM)`, which compiles to `buffer_wbl2 sc0 sc1` + `global_store ... sc0 sc1` on gfx942 and `global_wb` + `global_store scope:SCOPE_SYS` on gfx1201) would be tighter. On this gfx942 slice, measured with a 1M-element kernel plus a one-thread signal kernel, 2000 iterations x 3: no fence 11.6 to 11.9 us per iteration, `__threadfence_system` 18.1 us, release store 18.0 us. The L2 write-back costs about 6.4 us per signal and the acquire adds about 0.15 us, so the difference is not worth a round. The fence also costs gfx942 about 6 us x 3 per step it does not need (gfx942 is not affected, below); that is negligible against a step and keeps the fix arch-unified.
- **gfx10/gfx11 note (not a defect):** on gfx1100, `__threadfence_system()` compiles to `buffer_gl1_inv; buffer_gl0_inv` only, with no L2 write-back, because the LLVM memory model treats the gfx10/11 L2 as coherent with host memory. So on RDNA2/3 the fence does nothing for this mechanism, and the hardware model says it does not need to. When a gfx1100 host validates, run `d2h_visibility` there (kernel and writevalue, 512 B and 4 KiB) to confirm.
- **nvcc sees no change:** the fence is under `#if defined(__HIP_PLATFORM_AMD__)`, which nvcc never defines. `cuda_to_hip.h` is force-included only by the HIP CMake paths (`-include cuda_to_hip.h` in the `cmakegpu/*.cmake` HIP branches). The CMake hunk sits inside `IF(PX_USE_HIP)` after `ENABLE_LANGUAGE(HIP)` (`cmakegpu/linux/CMakeLists.txt:149-156`). The real nvcc build at 42d4c033 is still owed (mi355 host).

### Every host read-back site with the same shape (small device-to-host copy, then a GPU-written host flag the host spins on, no runtime sync in between)
Searched `physx/source` and `physx/include` for `spinWait`, `BP_SIGNAL_COMPLETE`, `volatile` host reads, mapped pinned memory (`DEVICEMAP`, `getMappedDevicePtr`, `cuMemHostGetDevicePointer`), `cuStreamWriteValue`/`WaitValue` (no use anywhere), and every file with `memcpyDtoHAsync` against its synchronization calls.
| Site | Shape | Verdict |
|---|---|---|
| `PxgCudaBroadPhaseSap::gpuDMABack` (`PxgCudaBroadPhaseSap.cpp:421-450`): `PxgBroadPhaseDesc` copy, plus the AABB manager's `mAggregateDesc` copy on the same `bpStream` (`PxgAABBManager.cpp:763`), and found/lost pairs written by kernels straight into mapped memory; then `bpSignalComplete`, then `spinWait` | same | covered by the fence |
| `PxgSimulationCore::gpuMemDmaBack` / `syncDmaback` (`PxgSimulationCore.cpp:2290-2345`): bounds, transforms, activate/deactivate lists, changed handles, then `bpSignalComplete`, then `spinWait` | same (the site the diagnosis caught) | covered |
| `PxgSolverCore::gpuMemDMAbackSolverBodies` (`PxgSolverCore.cpp:195-222`): three body copies on `mStream`, plus `mStream2` work joined through `synchronizeStreams` (its `hipEventRecord` already releases at system scope); spun on in `PxgCudaSolverCore::syncDmaBack` (`:771`) and `PxgTGSCudaSolverCore::syncDmaBack` (`:759`) | same | covered (4 spin loops, 3 launch sites, all through `bpSignalComplete`) |
| Narrowphase fetch (`PxgNarrowphaseCore.cpp:1767-2035`, including `mLostAndTotalReportedPairsCountMapped`), FEM cloth, soft body, particle system, FEM core, geometry manager, `CudaContextManager` | copies or mapped writes followed by `streamSynchronize` / `eventSynchronize` | not affected: `HostQueue::finish` (`commandqueue.cpp:159-200`) sends a marker when the fence is dirty, i.e. after any agent-scope dispatch |
| `PxgParticleAndDiffuseBuffer::readOmniPvdData` (`PxgParticleSystemCore.cpp:354`): host reads the mapped diffuse count written by `ps_diffuseParticleCopy` | kernel write to mapped memory read after the simulation's own synchronization; no copy-then-flag | different shape, upstream code; not this mechanism |
| Isosurface, anisotropy, smoothing, sparse-grid standalone utilities (`PxgIsosurfaceExtraction.cpp:701-707`, `PxgAnisotropy.cpp:140-160`, `PxgSmoothing.cpp:107/122`, `PxgSparseGridStandalone.cpp:225`) and `PxCudaHelpersExt.h` | copies onto the caller's stream; the caller synchronizes | not affected (no flag) |
No other site needs the workaround. Minor correction to the diagnosis's wording: there are four `spinWait` loops (broadphase, simulation core, two solver cores), signalled from three launch sites; all launch the same kernel, so the conclusion stands.

### Checked/profile `-O3` (`cmakegpu/linux/CMakeLists.txt:149-156`)
Correct: CMake defines no `CMAKE_HIP_FLAGS_CHECKED`/`_PROFILE`, and clang's default is -O0. Upstream's `PHYSX_CUDA_FLAGS_CHECKED` and `_PROFILE` (`:247-248`) are identical to release apart from nothing device-side (nvcc device default -O3), so profile had the same gap. `IF(NOT ...)` honors a caller's `-DCMAKE_HIP_FLAGS_CHECKED=-O0` (`-O0` is not a CMake false constant). No `-DNDEBUG` is added, so device asserts stay on in checked, as before. The Windows cmakegpu file has no HIP branch yet (the Windows gate is not ported), so there is nothing to mirror.

### `hipCtxSynchronize` stub (`cuda_to_hip.h:269-271`)
Re-read at the pin (`hip_context.cpp:325-328`): `HIP_RETURN(hipErrorNotSupported)`, no wait. It is still a stub on develop today (`3c589064`, `hip_context.cpp:337-340`). PhysX's only calls are `CudaContextManager.cpp:872/876` under `PX_STOMP_ALLOCATED_MEMORY` (debug), bracketing the 0xcd fill. **Yes, it should be mapped now, and the delta does it correctly:** `hipCtxCreate` returns the device's primary context, so a device-wide wait is exactly `cuCtxSynchronize` for PhysX. The other context calls PhysX makes (`cuCtxCreate/Destroy/GetDevice/PushCurrent/PopCurrent`, `cuCtxGetStreamPriorityRange` -> `hipDeviceGetStreamPriorityRange`) are implemented in this CLR; the other `hipErrorNotSupported` stubs (`hipCtxGetApiVersion`, `GetCacheConfig`, `SetSharedMemConfig`, `GetFlags`) are not called.

### Reproducer and issue draft (accuracy only; nothing posted)
- `d2h_visibility.cpp` extracted from the draft unchanged; it builds with this job's `hipcc -O2 --offload-arch=gfx942` (no errors). **On gfx942 (MI300X CPX slice, 38 CUs) it reports 0 stale in all 8 cases**, 200,000 iterations each: kernel and writevalue at 512 B, 4 KiB, 16 KiB, 64 KiB (`agent_space/repro/run-gfx942.log`). So CDNA3 is not affected; even `hipStreamWriteValue32`, stale in over 99% of gfx1201 iterations, is clean. Consistent with gfx942's agent-scope release writing back host-memory lines (the agent spans several L2s). **Add to the issue body:** "Not reproduced on MI300X (gfx942): stale=0 in all eight cases."
- Rate wording: the draft says a kernel-written flag beats the copy "in about 1 of every 10,000 to 17,000"; its own outputs (23, 12 and 15 of 200,000) are 1 in 8,700 to 16,700. Say "about 1 in 9,000 to 17,000".
- Source claims verified at the pin (line numbers match: `rocblit.cpp:3332` comment, `useShaderCopyBufferPath` at `:2769`, `submitBatchMemoryOperation` comment `rocvirtual.cpp:4399-4406`). The `copyBuffer` condition is unchanged at today's develop tip `3c589064` (`rocblit.cpp:3633-3640`; `985bf6ca` is its ancestor); the draft may cite the newer SHA. Labels `bug` and `project: clr` exist on ROCm/rocm-systems (`gh label list`). No existing issue found (`gh search issues` for "hipMemcpyDtoHAsync stale", "GPU_FORCE_BLIT_COPY_SIZE", "hipStreamWriteValue32 before copy", "small device to host copy visibility": none relevant).
- Title is 72 characters and fine. Keep "as far as I can tell" on the acquire explanation: the source shows why the release is missing, but not why a system-scope acquire on the later packet fails to drain it on gfx12.

### Commit messages and lesson
- Titles 49/65/72 characters, `[ROCm]` correct on all three (HIP-only changes), AI disclosure present, Test Plans fenced, no trailer; `jargon.py --diff` and `--commits 886c3529..42d4c033` clean.
- `8f87766f` says "HIP does not [guarantee this], for copies of 16 KiB or less"; the gfx942 result shows that holds on gfx1201, not on every AMD GPU. Not worth a history rewrite. If the round is squashed for the upstream PR, scope the sentence to "on some GPUs (seen on gfx1201)" and spell "signaling" (American) in the title and the code comment.
- **Lesson "a GPU-written host flag can arrive before an earlier small device-to-host copy":** accurate as qualified ("seen on ROCm 10.2, gfx1201"). When it is promoted, add: not reproduced on gfx942 (MI300X), and on gfx10/11 `__threadfence_system()` writes nothing back from L2 (so the advice is for gfx12; there a release-only system-scope store is enough).

## CUDA harness build for an NVIDIA run 2026-11-24 (mi355 fleet host, nvcc 12.8, no GPU used)

Requested by Jeff: build the CUDA path of fork `42d4c033` (the head) with the parity harness, packaged to run `SnippetGpuParity` on an NVIDIA Tesla V100 (sm_70, driver 575.57, Ubuntu 24.04, glibc 2.39) elsewhere, to see whether CUDA shows the same run-to-run variation in the three deformable checks (lever bend, load height, lowest deformable vertex) that gfx942 and gfx1201 show. Nothing ran on a GPU (none used on this host), nothing in the fork changed, nothing was pushed to it, and the project's stage and lock were not touched. Built in a fresh clone in the job directory (`<job>/cuda/src`, `GIT_LFS_SKIP_SMUDGE=1`, detached at `42d4c033`; afterwards `git status --porcelain` shows only untracked `physx/bin/`).

### Toolchain
- nvcc `/usr/local/cuda-12.8/bin/nvcc` "Cuda compilation tools, release 12.8, V12.8.93"; host gcc "Ubuntu 13.3.0-6ubuntu2~24.04.1" 13.3.0; CMake 3.31.6 from the job venv (`$FLEET_VENV/bin` first on PATH). Packman fetched upstream's public packages into `<job>/cuda/packman-repo`.
- Snippet dependency: freeglut v3.6.0 built from source into `<job>/cuda/deps/prefix` (`cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DFREEGLUT_BUILD_DEMOS=OFF -DFREEGLUT_BUILD_STATIC_LIBS=OFF`), only so that the `linux-gcc` preset's snippet configure finds GLUT. No relink was needed: the harness never renders, and Ubuntu's linker default `--as-needed` drops SnippetRender's GL/GLU/GLUT/X11, so the built `SnippetGpuParity_64` is packaged as built. Its `NEEDED` is libstdc++/libm/libc (checked adds ld-linux); PhysX is linked statically; it `dlopen`s `libPhysXGpu_64.so` and `libcuda.so.1` at run time. Its RUNPATH is `$ORIGIN` plus the job's freeglut path, which is harmless (missing on the target, never needed).

### Builds (both exit 0)
```
cd physx
export PM_PACKAGES_ROOT=<job>/cuda/packman-repo PM_CUDA_PATH=/usr/local/cuda-12.8 CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH=$FLEET_VENV/bin:/usr/local/cuda-12.8/bin:$PATH CMAKE_PREFIX_PATH=<job>/cuda/deps/prefix
./generate_projects.sh linux-gcc
make -C compiler/linux-gcc-release -j21
make -C compiler/linux-gcc-checked -j21
```
61 `.cu.o` per config, 0 errors, no warnings other than nvcc's deprecated-target note for sm_52/sm_70 (61 per build). `PX_GENERATE_GPU_PROJECTS=TRUE`, `PX_GENERATE_GPU_REDUCED_ARCHITECTURES=FALSE`: every `.cu` compile carries `arch=compute_N,code=[compute_N,sm_N]` for N = 52, 70, 80, 86, 89, 90, 100, 120 plus `compute_120` PTX. `cuobjdump --list-elf libPhysXGpu_64.so`: 61 cubins for each of the 8 targets, sm_70 included, in both configs (`--list-ptx` shows PTX for every target too, since the gencode list embeds `compute_N` beside each `sm_N`). `libPhysXGpu_64.so` `NEEDED` libstdc++/libm/libc only; `ldd -r` leaves exactly 51 undefined `cu*` symbols, which the loader's `dlopen("libcuda.so.1", RTLD_GLOBAL)` supplies. Highest symbol versions required: GLIBC_2.38, GLIBCXX_3.4.29 (checked harness), CXXABI_1.3.9, all within Ubuntu 24.04.

### Tarballs (`<job>/out/`)
Each holds one directory `physx-cuda-harness-<config>/` with `SnippetGpuParity_64`, `libPhysXGpu_64.so`, `run.sh` (sets `LD_LIBRARY_PATH` to its own directory and runs the harness with no arguments, as the AMD validations did; the harness ignores arguments), `README.txt` and `SHA256SUMS`. `libcuda.so.1` is not packaged; it comes from the target's driver.

| Tarball | Size | sha256 | Harness | libPhysXGpu_64.so |
|---|---|---|---|---|
| `physx-cuda-harness-release.tar.gz` | 111,388,826 B | `5c3e4cbaef6dfec95602c8c06ac5d34bd855ebf7d83e841df7bbab1411b1958e` | 8,335,032 B | 348,434,864 B |
| `physx-cuda-harness-checked.tar.gz` | 175,164,029 B | `28113e5ddb2926637c5828b5e6e0f0b6c6dd82212ad87ef99136343b07e313f3` | 190,921,416 B | 372,973,864 B |

Checked after unpacking, from `/tmp`, with `LD_LIBRARY_PATH=<unpacked dir>`: `ldd SnippetGpuParity_64` and `ldd -r libPhysXGpu_64.so` resolve only linux-vdso, libstdc++.so.6, libm.so.6, libc.so.6, libgcc_s.so.1 and ld-linux-x86-64.so.2 from the system, plus the 51 `cu*` symbols of `libcuda.so.1`; `sha256sum -c SHA256SUMS` OK. The harness was not run here (no NVIDIA GPU or driver, and GPUs were not to be used).

### For the NVIDIA run
`./run.sh` five times per config (release about 10 s per run on AMD, checked several minutes), pin with `CUDA_VISIBLE_DEVICES` on a multi-GPU node, and compare the three deformable checks across runs with the gfx942 and gfx1201 ranges recorded in the two "Validation 2026-11-24" sections; the other 45 checks are expected to be identical from run to run. Packaging script: `<job>/cuda/scripts/package.sh <config>` (job-local; the description above is enough to redo it).
