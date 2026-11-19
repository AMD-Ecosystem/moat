# diff-surfel-tracing port plan

## Project

- Name: `diff-surfel-tracing`
- Upstream: https://github.com/xbillowy/diff-surfel-tracing, default branch `main`, HEAD `ef6f24b`
- Fork: https://github.com/AMD-Ecosystem/diff-surfel-tracing; `main` mirrors upstream, `moat-port`
  is 2 commits (`5991683`, `415f0a4`) on top of upstream `9b86cbf`
- What it is: a differentiable 2D-Gaussian (surfel) ray tracer built on NVIDIA OptiX 7, published
  as the reflection backend of EnvGS and also used by LiDAR-RT. One PyTorch extension
  (`diff_surfel_tracing._C`) plus a CMake step that compiles the OptiX programs to PTX.
- Planned on linux-gfx942, 2026-08-13.

This is not a fresh port. A complete OptiX-to-HIPRT reimplementation already exists on
`moat-port` and was GPU-validated on four platforms -- but under **EnvGS's** record, against an
upstream base that has since moved, and in a shape that cannot be offered upstream. The plan
below covers what remains, and says explicitly what is already proven so nobody re-does it.

## Existing AMD support: improvable, and it is ours

Intake settled licence and duplicate effort. Applying the finer judgement from the skill's
`assess-existing-support.md`: there is no third-party AMD or ROCm work on this project at all
(upstream README has zero amd/rocm/hip/gfx matches, none of the five forks is an AMD port, no
ROCm-org repo). The only existing AMD support is MOAT's own, written during EnvGS Stage 2 and
committed to this fork before this project had a record.

Authoritativeness test: it is our own code, reviewed (`review-passed` under EnvGS) and GPU-proven.
**Decision: inherit it and extend it.** The tracer kernel logic, the HIPRT integration and the
build structure are not re-designed. What changes is everything that stands between that code and
a maintainer: the upstream base it sits on, the shape of the third-party dependency, and the fact
that the CUDA path no longer builds.

### What EnvGS Stage 2 already proves (do not re-derive)

From `projects/EnvGS/notes.md`, all at tracer commit `5991683`/`415f0a4`:

- **The OptiX-to-HIPRT mapping is correct and complete.** Triangle GAS `optixAccelBuild` ->
  `hiprtCreateGeometry`/`hiprtBuildGeometry` over the same host disk tessellation; raygen +
  closesthit + miss collapse into one HIP kernel; `__anyhit__` record-and-ignore ->
  a HIPRT filter functor (`surfelFilter`) returning true to enumerate all hits t-sorted;
  `optixLaunch` -> `oroModuleLaunchKernel` of a `hiprtBuildTraceKernels` JIT module with a disk
  cache. The public API (`OptiXStateWrapper`, `SurfelTracer`, `SurfelTracingSettings`) is
  byte-stable, so consumers need no change.
- **The build structure works and is non-obvious.** Orochi's hipew driver loader redeclares the
  HIP driver API and conflicts with torch's `<hip/hip_runtime.h>`; they cannot share a
  translation unit. The HIPRT/Orochi glue is therefore a standalone static library and the torch
  extension reaches it through a POD/`void*` boundary (`hiprt_tracer/hiprt_wrapper.h`). Anyone
  "simplifying" that boundary will produce an uncompilable TU.
- **Two genuine bug classes were found and fixed**, both latent in the upstream OptiX sources:
  a value-returning `__device__` function falling off the end (UB; poisons the surfel normal and
  NaNs every geometric gradient once inlined into the register-heavy backward traversal), and an
  uninitialized `cutoff` on the reflected bounces.
- **Numerical correctness on four platforms**: linux-gfx90a (wave64), linux-gfx1100 (wave32),
  windows-gfx1201 and windows-gfx1101. Forward renders a non-trivial image with genuine
  hits and misses; all backward gradients finite; FD-vs-analytic cosine ~1.00 on colors and
  opacities and ~0.997 on means3D/scales; cold-JIT reruns bit-identical.
- **No wavefront fault class in our device code.** Exhaustive grep of `hiprt_tracer/` finds no
  `warpSize`, `__shfl*`, `__ballot`, `__activemask`, `tiled_partition` or hardcoded 32. Shading is
  per-ray serial compositing plus `atomicAdd`; both kernels guard `h >= H || w >= W`. That is why
  wave64 and wave32 passed with no arch-specific edit, and it stays true after the rebase.

So the **technique** is proven and the **numerics** are proven. What is unproven is this
project's own record: `head_sha` is null, no evidence is tied to any commit of this repository,
and the code is two upstream commits behind an algorithm change.

### Delivery vehicle: an upstream PR is realistic, and it is the target

Recorded because it decides the shape of the work. Unlike the sibling `diff-surfel-rasterizations`
(tier-4 Inria licence, 2-star repo, fork-is-deliverable), this repository can take a patch:

- **Licence tier 1.** `LICENSE` at HEAD is verbatim MIT, copyright 2024 3D Vision Group, State Key
  Lab of CAD&CG, Zhejiang University (added 2025-10-14 in `ef6f24be`; intake verified by reading
  the file -- GitHub's NOASSERTION is a misparse).
- **The maintainer is responsive and acts.** Issue #2 ("License") was answered by `xbillowy`
  the same day and the LICENSE commit landed that same day. Issue #4 was answered and closed.
  Owner account active as of 2026-06-04.
- **The one closed PR is not a rejection.** PR #5 was opened at 14:52:45 and closed at 14:53:20 --
  35 seconds later, by its own author (`rhombus19`), not by the maintainer. There is no evidence
  of a maintainer turning contributions away.
- **Upstream does not link platform forks.** No notable-forks section, no "see the AMD fork"
  pointer. The karpathy-style exception does not apply; a PR is the right vehicle.
- 58 stars, 5 forks, not archived, last push 2025-10-14, and two downstream projects (EnvGS,
  LiDAR-RT) depend on it.

**But the port in its current shape is not offerable**, and that is the substance of this plan:

1. `setup.py` ends with `if not torch.version.hip: raise SystemExit(...)`. The fork replaced the
   OptiX build instead of adding to it. An additive port must leave the CUDA path a passthrough.
2. `moat-port` vendors 98 files of the HIPRT SDK (+49,970 lines), including a real 784 KB
   `hiprt0300164.dll` and a 10.5 KB import lib, and four of those vendored files carry our own
   local patches. Asking a maintainer to absorb a third-party SDK is a hard sell.
3. The vendored tree has no HIPRT or Orochi licence file (only cuew's), and one of our patches
   contains MOAT vocabulary (`third_party/hiprt/hiprt/impl/Compiler.cpp:655`, "MOAT probe patch"),
   which `utils/jargon.py` will refuse.

Correcting intake's summary on one point of fact: the three `contrib/Orochi/contrib/bin/win64/*.dll`
files are **git-LFS pointer stubs of 132-134 bytes**, not binaries -- and since the repository has no
`.gitattributes`, they will not resolve on clone at all. The genuinely committed binary is
`third_party/hiprt/dist/bin/Release/hiprt0300164.dll` (784,384 bytes) plus its `.lib`, added in
`415f0a4` for the Windows validation.

### Port versus AMD-native rewrite

Already settled by the code that exists, and worth restating for the reviewer. OptiX has no HIP
analogue, so this was never a hipify: it is a reimplementation against an AMD-native target, and
HIPRT **is** that target. There is no CUTLASS, CuTe, wgmma or warp-specialized kernel anywhere in
this project, so there is no second, more-native rewrite waiting behind a mechanical translation.
The one performance question that does exist is local and recorded under Open questions.

## Build classification: torch-extension

Evidence:

- `setup.py:4`: `from torch.utils.cpp_extension import CUDAExtension, BuildExtension`
- `setup.py`: `ext_modules=[CUDAExtension(name="diff_surfel_tracing._C", sources=[...])]`,
  `cmdclass={'build_ext': CustomBuildExtension}` subclassing `BuildExtension`
- `ext.cpp:6`: `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` binding `OptiXStateWrapper`,
  `build_acceleration_structure`, `trace_surfels`, `trace_surfels_backward`

`ext_type` set to `torch-extension` on this record.

There is a **second build path in the same project**, and it is not optional: `CustomBuildExtension
.build_extensions` shells out to `cmake .. && make` after the extension links, compiling
`optix_tracer/{forward,backward}.cu` to **PTX** (`CMake/PTXUtilities.cmake`,
`optix_tracer/CMakeLists.txt` -> targets `optix_tracer_forward`, `optix_tracer_backward`), then
copies `build/ptx/*.ptx` into the installed package. The OptiX device programs never link into
`_C`; they are loaded at runtime by `optixModuleCreate`. This is the runtime-PTX build class the
skill warns about, and it is the reason the HIPRT port replaces `optixModuleCreate` with a hiprtc
JIT rather than with a compiled kernel: on ROCm the equivalent artifact is compiled at runtime from
`hiprt_tracer/kernels.h` shipped inside the package.

## Port strategy: B (torch extension), with a per-backend device path

Strategy B in the classification sense -- it is a `CUDAExtension` and torch's build-time hipify
runs -- but hipify does almost nothing here, because the CUDA-shaped code is thin (`atomicAdd`,
`cudaMalloc`/`cudaFree`, a stream) and the substance is an OptiX pipeline with no symbol-level
translation. The real strategy is: **keep upstream's OptiX sources untouched, add a parallel HIPRT
backend, and select between them in `setup.py` on `torch.version.hip`.**

Two things follow, and they are the shape of the whole port:

- **The device kernel body is duplicated, and that is inherent.** `hiprt_tracer/kernels.h` (1,304
  lines) is a faithful transposition of `optix_tracer/forward.cu` (660) + `backward.cu` (1,209)
  into one JIT translation unit, because OptiX's multi-program-plus-SBT model and HIPRT's
  single-kernel-plus-functor model do not share a source structure. Factoring the shading math into
  a header shared by both backends would mean refactoring upstream's CUDA path, which the standing
  rules forbid. Duplication it is -- stated plainly in the PR, and raised as a question for the
  maintainer rather than decided for them.
- **The shared, non-traversal headers must NOT be duplicated.** They currently are, and it is pure
  drift risk. `hiprt_tracer/config.h` is byte-identical to `optix_tracer/config.h`;
  `hiprt_tracer/auxiliary.h` differs from `optix_tracer/auxiliary.h` by exactly three HIP-guarded
  blocks (`__trap` -> `__builtin_trap`; guarding out the float2/3/4 operator overloads that collide
  with HIP's `HIP_vector_type` under `__HIPCC_RTC__`; the missing-return fix). Every one of those
  guards is `__HIP*`-conditional, so folding them back into upstream's own headers leaves the CUDA
  build byte-identical and removes two files that would otherwise silently diverge from upstream at
  every future rebase. Do that.

## CUDA / OptiX surface inventory

Regex census of upstream HEAD (`utils/surface.py`): 5 CUDA-bearing files, 150 device-code markers,
4 `__global__`, 2 runtime-PTX markers. No cuBLAS/cuFFT/cuRAND/cuSPARSE/cuDNN, no Thrust/CUB, no
NCCL, no textures, no driver API beyond `CUdeviceptr`/`cuCtx` handles.

| upstream construct | count | ROCm/HIP mapping | status |
|---|---|---|---|
| `optixDeviceContextCreate` / `Destroy` | 1/1 | `hiprtCreateContext` over an Orochi context | done |
| `optixAccelComputeMemoryUsage` + `optixAccelBuild` (`OPTIX_BUILD_INPUT_TYPE_TRIANGLES`) | 1+2 | `hiprtCreateGeometry` + `hiprtBuildGeometry` with `hiprtTriangleMeshPrimitive`; HIPRT sizes its own scratch | done |
| `optixAccelCompact`, `OPTIX_PROPERTY_TYPE_COMPACTED_SIZE` | 1 | already commented out upstream; HIPRT compacts internally | n/a |
| `optixModuleCreate` from PTX | 1 | `hiprtBuildTraceKernels` -- hiprtc JIT of `kernels.h`, disk-cached | done |
| `optixProgramGroupCreate` x3, SBT records, `optixSbtRecordPackHeader`, `optixPipelineCreate`, stack-size utils | 3/3/1 | no analogue and none needed: HIPRT has no SBT and no pipeline object | dissolved |
| `__raygen__ot` (fwd, bwd) | 2 | the body of `forward_kernel` / `backward_kernel` | done |
| `__anyhit__ot` + `optixIgnoreIntersection` | 2+2 | `surfelFilter` device functor in a `hiprtFuncTable`, returns true to continue | done |
| `optixTrace` + `optixGetPayload*` | 7+4 | `hiprtGeomTraversalAnyHitCustomStack` loop; payload is a struct passed to the functor | done |
| `optixGetLaunchIndex` / `Dimensions` | 4+4 | `blockIdx/threadIdx` over the (H, W) grid | done |
| `optixGetRayTmax`, `optixGetPrimitiveIndex` | 2+2 | `hiprtHit::t`, `hiprtHit::primID` | done |
| `optixLaunch` | 2 | `oroModuleLaunchKernel` | done |
| `cudaMalloc`/`cudaFree`/`cudaMemcpy*` (9/13/7) | | `hipMalloc`/`hipFree`/`hipMemcpy*` via Orochi in the glue TU | done |
| `cudaStream` (4), `at::cuda::getCurrentCUDAStream` | | `c10::hip::getCurrentHIPStream()`, passed across the boundary as `void*` | done |
| `atomicAdd` (20 sites, float and float3-component) | | identical in HIP | done |
| warp intrinsics, `warpSize`, hardcoded 32 | **0** | nothing to do | n/a |
| textures/surfaces, managed/pinned memory | **0** | nothing to do | n/a |

Beyond the census, what the tooling cannot see and the port must still account for:

- **The runtime-PTX build path** (`CMakeLists.txt`, `CMake/FindOptiX7.cmake`,
  `CMake/PTXUtilities.cmake`, `optix_tracer/CMakeLists.txt`) -- NVIDIA-only, must keep working, and
  must not be invoked on ROCm.
- **The JIT include set**: `kernels.h` plus the headers it pulls in are staged into the installed
  package and compiled at runtime by hiprtc. Anything the kernel includes must be shipped by
  `package_data` or the extension imports and then fails at first trace.
- **`HIPRT_PATH`**: HIPRT's `Utility::getRootDir()` reads it to find its own BVH-builder kernel
  sources for JIT. Without it the BVH build fails at runtime, not at build time.

## Risk list

1. **The rebase changes the algorithm, silently.** Upstream `e0016a2` ("update: latest version")
   removed the 2D-projection low-pass filter path (`compute_transmat_xy`, `rho2d`, `cmd`, the
   `projmatrix`/`W`/`H` use) and the distortion accumulation, flipped `DUAL_VISIABLE`'s direction
   vector from `means3D - campos` back to `ray_d`, moved `STEP_EPSILON` into `payload.dpt`, changed
   `min_depth` for the non-`start_from_first` primary ray, and dropped the internal `normalize` in
   `computeColorFromSH`. Every one of those is a behavioural change that compiles fine and shows up
   only as a different image. Replay them hunk-by-hunk from upstream's own diff into `kernels.h`;
   do not re-transpose the file from scratch.
2. **New backward outputs are easy to leave at zero.** `e0016a2` added `dL_dgrads3D` and
   `dL_dgrads3D_abs` to `Params` and made `trace_surfels_backward` return 12 tensors instead of 10
   (`__init__.py` no longer computes `grad_grads3D` in Python). A HIPRT backward kernel that never
   accumulates them returns finite zeros and passes any "all gradients finite" check. The harness
   must assert these two are **nonzero** and that `grads3D` tracks `means3D`.
3. **Missing-return UB, still live at upstream HEAD.** A scan of upstream `ef6f24b` finds exactly
   one value-returning `__device__` function with no return: `optix_tracer/auxiliary.h:403
   quat_to_rotmat_transpose` (declared `float3`). It is on the failing path. The second site fixed
   during Stage 2, `compute_transmat_xy_backward`, no longer exists upstream -- `e0016a2` deleted
   it. Likewise the uninitialized `cutoff`: upstream now assigns it unconditionally at the top of
   the backward kernel (`backward.cu:405-411`), so that fix is subsumed. **Carry the first fix
   forward; drop the other two as fixed upstream, and say so rather than leaving dead diff.**
   hiprtc does not surface its own warnings: after the rebase, compile `kernels.h` standalone with
   `-Werror=return-type` and confirm it is clean.
4. **hiprtc/comgr environment failures look like port bugs.** `hipErrorInvalidImage (200)` on an
   arch missing from the JIT's default target set, a cache write failing because the device name
   contains `/`, a Windows DLL-load picking the display-driver copy over the ROCm SDK copy. All
   four are HIPRT-side and all four are patched locally today (see below).
5. **The chunk buffer must stay in global scratch.** Stage 2 found that a per-ray hit buffer held in
   a kernel stack array goes stale above a register-pressure threshold; `params.chunk_buffer +
   tidx * CHUNK_SIZE` is the recorded fix. A rebase that "tidies" it back onto the stack will
   reintroduce an intermittent wrong-image bug.
6. **The glue/torch TU boundary is load-bearing** (see Existing AMD support). Keep it POD/`void*`.
7. **The CUDA path cannot be tested by us.** No NVIDIA GPU and no OptiX SDK on the fleet. The
   mitigation is structural, not experimental: keep every CUDA-path file byte-identical to upstream
   except the single unconditional UB fix, so "unchanged" is a diff the reviewer can verify rather
   than a claim.
8. **De-vendoring HIPRT trades a licence problem for a dependency problem.** Our four HIPRT patches
   are all still absent from HIPRT HEAD (checked 2026-08-13 against
   `GPUOpen-LibrariesAndSDKs/HIPRT` `hiprt/impl/Compiler.cpp`, last pushed 2026-04-15): no device-name
   sanitize in `getCacheFilename`, no `--offload-arch` in `addCommonOpts`, no basename-only source
   name in `buildProgram`. Two of them are needed by platforms we must gate on (gfx90a's
   "AMD Instinct MI250X / MI250" name; gfx1201's absent default target). So a plain pinned submodule
   does not build a working tracer everywhere today.
9. **Wavefront:** no risk in our code (item above), but note HIPRT itself is wave-size aware --
   `WarpSize` 64 on the CDNA arm (`hiprt_common.h:202` lists `__gfx90a__`, `__gfx942__`,
   `HIPRT_RTIP 0`, software BVH traversal) and 32 on RDNA, selected inside HIPRT. gfx942 is
   explicitly supported. Do not attempt to force a wave size from our side.
10. **Licence, still a person's ruling, and materially improved by the rebase.** The 9
    NVIDIA-proprietary headers intake flagged live inside the `third_party/optix` submodule
    (`NVIDIA/optix-dev`), which the port neither modifies nor redistributes -- it is a gitlink
    pointing at NVIDIA's own repository, and it stays. Separately: the base commit's
    `optix_tracer/params.h` carried a pasted NVIDIA proprietary banner, and our
    `hiprt_tracer/params.h` derives from that file. Upstream **removed that banner itself** in
    `e0016a2`, so after the rebase the derivation is from a plain MIT-licensed file. The remaining
    in-tree NVIDIA text (`CMakeLists.txt`, `CMake/*.cmake`, `optix_tracer/common.{h,cpp}`,
    `optix_tracer/CMakeLists.txt`) is BSD-3-Clause, not proprietary, and is untouched.

## File-by-file change list

Numbers are versus upstream HEAD `ef6f24b`. "exists" = already on `moat-port` and needs rebasing
rather than writing.

### Rebase and re-derive

| file | change |
|---|---|
| `hiprt_tracer/kernels.h` | exists (1,304 lines). Replay `e0016a2` + `3b97d5d`: delete `compute_transmat_xy`, `compute_transmat_xy_forward`, `compute_transmat_xy_backward` and all `rho2d`/`cmd`/`P`/`splat2pixel`/`xy` state; drop the distortion accumulation (`dist`, `M1`, `M2`); `DUAL_VISIABLE` dir -> `ray_d`; `payload.dpt = dpt + STEP_EPSILON` and `ray_ot = ray_o + payload.dpt * ray_d`; `min_depth` ternary for the non-`start_from_first` primary; `ray_ot = E + START_OFFSET * ray_dt`; `computeColorFromSH` takes a pre-normalized dir; accumulate `dL_dgrads3D` and `dL_dgrads3D_abs`. Keep: global chunk scratch, the `surfelFilter` functor, the traversal loop. |
| `hiprt_tracer/params.h` | exists. Fold into upstream's `optix_tracer/params.h` under `#ifdef USE_ROCM` instead of keeping a copy: add `IntersectionInfo` and `chunk_buffer`, drop `OptixTraversableHandle handle` on the HIP side, pick up upstream's new `dL_dgrads3D`/`dL_dgrads3D_abs`. One file, one place to update at the next rebase. |
| `hiprt_tracer/config.h` | **delete** -- byte-identical to `optix_tracer/config.h`; include that instead. |
| `hiprt_tracer/auxiliary.h` | **delete** -- fold its three HIP-guarded blocks into `optix_tracer/auxiliary.h` (`__trap` -> `__builtin_trap`; `#if !defined(__HIPCC_RTC__)` around the float2/3/4 operator overloads; the `quat_to_rotmat_transpose` void fix, which is unconditional because the UB is real on CUDA too). CUDA preprocessed output is unchanged. |
| `hiprt_tracer/hiprt_wrapper.{h,cpp}` | exists (57 + 294 lines). Keep the POD/`void*` boundary as-is. Update only for the de-vendored HIPRT include path. |
| `trace_surfels.cpp` | exists but **rewritten in place, deleting the OptiX host glue**. Restore it: upstream's `optixAccelBuild`/`optixLaunch` bodies come back verbatim, the HIPRT bodies move under `#ifdef USE_ROCM`. Pick up upstream's 12-tensor backward signature and the `3b97d5d` variable removals. |
| `trace_surfels.h` | exists. Conditional include of the backend wrapper; adopt upstream's 12-element return tuple. |
| `ext.cpp` | exists. Conditional include only; bindings unchanged. |
| `diff_surfel_tracing/__init__.py` | exists. Keep the `__file__`-based `pkg_dir` (upstream's `site.getsitepackages()[0]` is wrong for editable installs on **both** backends -- offer it as a plain fix, not a ROCm one); keep `HIPRT_PATH` and the Windows DLL-directory block, both already `os.name`/`isdir` guarded; adopt upstream's 12-value unpack and the `SurfelTracingSettings` defaults. |
| `setup.py` | exists but **refuses to build on non-HIP**. Restore upstream's OptiX branch verbatim (OPTIX_HOME resolution, the CMake PTX step, the PTX copy) and put the whole HIPRT branch -- glue static lib, hipify ignore for the HIPRT tree, runtime staging, Windows `_winhip.cu` ABI workaround, link args -- behind `if torch.version.hip:`. |

### Third-party dependency

| file | change |
|---|---|
| `third_party/hiprt/**` (98 files, 49,970 lines) | **remove from the tree.** Replace with a `.gitmodules` entry pinning `GPUOpen-LibrariesAndSDKs/HIPRT` at tag `3.1.0.cb09c56` (the version currently vendored, `version.txt` = `3 1 0 cb09c56`), mirroring upstream's existing `third_party/optix` submodule pattern, plus a `HIPRT_HOME` environment fallback mirroring `OPTIX_HOME`. This deletes the committed `hiprt0300164.dll`/`.lib`, the three unresolvable LFS pointer stubs, and the licence-file gap in one move. |
| our 4 HIPRT patches | extract to a single patch file in the fork with a README line telling the user to apply it before building HIPRT, until the fixes land at GPUOpen. They are: `getCacheFilename` device-name sanitize (`/` -> `_`, needed for "AMD Instinct MI250X / MI250"); `addCommonOpts` `--offload-arch=<gcnArch>` (needed for gfx1201); `buildProgram` basename-only source name (needed on Windows: comgr fails silently on a `B:\...` source name); `hiprt.cpp`/`hiprt_libpath.h`/Orochi `hipew.cpp` full-path DLL loading (Windows). All four are genuine HIPRT bugs and all four are still absent from HIPRT HEAD. Contributing them to GPUOpen is the durable fix and is a separate, person-approved contribution -- register it with `utils/deferred.py add --project diff-surfel-tracing`. **Remove the "MOAT probe patch" comment** while extracting (`Compiler.cpp:655`); `utils/jargon.py` will otherwise refuse the branch. |
| `third_party/optix` | unchanged gitlink. Not modified, not redistributed. |

### Docs, build metadata, harness

| file | change |
|---|---|
| `README.md` | an installation section for ROCm alongside the OptiX one: HIPRT submodule/`HIPRT_HOME`, the HIPRT build command, `pip install -v .`. Same structure as the existing OptiX paragraph. |
| `.gitignore` | exists; keep the staged-artifact entries, drop the ones that only made sense with a vendored SDK. |
| `example/validate_rocm.py` | **new, and it ships in the fork.** The Stage 2 harnesses (`validate_stage2.py`, `validate_geom_fd.py`) were written under `agent_space/`, which is gitignored -- they are gone from every host that did not run that session, which is why the harness has to be a committed artifact somewhere. Rounds 1-5 kept it in MOAT on the reasoning that the project "cannot get one upstream without inventing a test framework for someone else's repository". The round-5 review ruled that reasoning does not hold: one standalone script beside the existing standalone `example/render.py` is not a framework, it imports only the standard library, torch and the package, and it is the answer to the maintainer's "how do I know this back end works on my machine" for a PR that adds ~2,400 lines of new back end to a repository with no tests. Moved in round 6, single copy, and referenced from the README's AMD section. |

Expected diff versus upstream after all of the above: roughly 6 new/changed source files and a
submodule entry, on the order of 2,000 added lines (of which ~1,300 is `kernels.h`), against
+49,970 today.

## Build commands

Reference platform gfx942 (this host); substitute `PYTORCH_ROCM_ARCH` per arch.

```bash
# 1. HIPRT (once per host; the host library is arch-agnostic, JIT handles the device)
git clone --depth 1 -b 3.1.0.cb09c56 https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT.git
cd HIPRT && git apply /path/to/diff-surfel-tracing/third_party/hiprt-rocm-fixes.patch
export HIP_PATH=/opt/rocm
cmake -DCMAKE_BUILD_TYPE=Release -DBITCODE=OFF -DNO_UNITTEST=ON -DHIP_PATH=/opt/rocm -S . -B build
cmake --build build --target hiprt03001 -j16      # -> dist/bin/Release/libhiprt0300164.so

# 2. the extension
export HIPRT_HOME=/path/to/HIPRT
export PYTORCH_ROCM_ARCH=gfx942
export HIP_VISIBLE_DEVICES=0
cd projects/diff-surfel-tracing/src
rm -rf build *.egg-info diff_surfel_tracing/hiprt_cache
bash ../../../utils/timeit.sh diff-surfel-tracing compile -- \
    pip install -e . --no-build-isolation --no-deps -v
```

`BITCODE=OFF` is HIPRT's tested default: the BVH-builder and traversal device kernels are JIT
compiled at first use and disk-cached. Clearing `diff_surfel_tracing/hiprt_cache/` forces a cold
JIT, which is a required part of the test plan below.

The CUDA path is unchanged and builds as upstream documents it (`pip install -v .` with the
`third_party/optix` submodule or `OPTIX_HOME` set) -- we cannot exercise it, see risk 7.

## Test plan

### The GPU gate

`example/validate_rocm.py` in the fork (moved there in round 6; MOAT keeps no copy),
self-contained, no external data, arch-agnostic. It must cover, at minimum:

1. **Import and API.** `import diff_surfel_tracing`; `_C` exports `OptiXStateWrapper`,
   `build_acceleration_structure`, `trace_surfels`, `trace_surfels_backward`.
2. **BVH build** from the disk tessellation used by EnvGS's `HardwareRendering.get_disks`, so the
   geometry path matches the real consumer.
3. **Forward.** Finite RGB, a non-trivial hit fraction (genuine hits *and* misses -- an all-hit or
   all-miss image passes a naive finiteness check and proves nothing), plausible depths, finite
   normals and accumulation. Bit-identical across repeated runs.
4. **Backward, all 12 outputs.** Every returned gradient finite; `grad_grads3D` and
   `grad_grads3D_abs` **nonzero** and consistent with `grad_means3D` (risk 2); colors gradient
   nonzero.
5. **Finite-difference vs analytic.** Colors is the exact linear gate (expect cosine 1.0000).
   Opacities, means3D and scales: cosine > 0.9 and slope in [0.5, 1.8]. Rotations: finite only --
   per-component cosine is quaternion-renormalization null-space noise, ruled acceptable on gfx90a
   and reconfirmed on gfx1100.
6. **A reflected bounce.** At least one case with `max_trace_depth >= 1` and a `specular_threshold`
   that actually spawns secondary rays. The reflection path is the reason this repository exists and
   is where the `cutoff` bug lived; a depth-0 run does not exercise it.
7. **Cold JIT.** Clear `diff_surfel_tracing/hiprt_cache/`, rerun, results bit-identical to the warm
   run and the cache repopulated. The Stage 2 heisenbug was masked by inlining changes, so a
   cold-cache rerun is a genuine gate here, not ceremony.

Run:

```bash
HIP_VISIBLE_DEVICES=0 bash utils/timeit.sh diff-surfel-tracing test -- bash -c \
    'cd projects/diff-surfel-tracing/src && python3 example/validate_rocm.py'
```

### Optional stronger evidence

`example/render.py` is a real end-to-end render (PLY 2DGS model + camera path -> images + video)
but needs the author's Google Drive archive (file id `1drKlXptpkht0ZVp6Ywh8ZSSXsi8RddKx`). If it
downloads, run it and record the result; it is the closest thing to the maintainer's own smoke
test. If it does not, that is not a gate failure -- record why and move on.

### Non-GPU regressions that must not break

- `python3 -c "import diff_surfel_tracing"` on a machine with no GPU visible must fail no earlier
  than upstream does (import must not require a live device).
- **CUDA-path integrity, by diff rather than by build.** `git diff upstream/main -- optix_tracer/
  CMake/ CMakeLists.txt` must show nothing but the single unconditional `quat_to_rotmat_transpose`
  fix and the HIP-guarded additions in `auxiliary.h`/`params.h`. `setup.py` must take the OptiX
  branch when `torch.version.hip` is falsy. This substitutes for the nvcc no-regression gate we
  cannot run, and the reviewer should check it explicitly.
- `python3 utils/jargon.py --port diff-surfel-tracing` clean -- currently it is not
  (`Compiler.cpp:655`).

### Gates

| gate | platform | status |
|---|---|---|
| wave64 | linux-gfx942 (this host) or linux-gfx90a | proven on gfx90a at `415f0a4` under EnvGS; **must be re-run at this project's own `head_sha` after the rebase**. gfx942 is new for this code -- HIPRT lists `__gfx942__` in the CDNA arm, and MI300-class device names contain no `/`, so the cache-filename patch may prove unnecessary there; verify rather than assume. |
| wave32 | linux-gfx1100 | proven at `415f0a4` under EnvGS; re-run after the rebase. |
| windows | windows-gfx1101 or windows-gfx1201 | proven at `415f0a4` under EnvGS; re-run after the rebase, and with attention -- de-vendoring removes the committed DLL, so the Windows install story changes from "it is in the tree" to "build HIPRT on Windows". If that proves impractical, say so and bring the trade-off back rather than silently re-vendoring. |

None of the prior evidence carries forward automatically: `head_sha` is null on this record, and
the rebase changes device code, so `carry-forward` does not apply. Every gate runs again.

## Open questions

1. **Does the maintainer want the duplicated kernel body, or a shared shading header?** Duplication
   keeps the CUDA path untouched (our standing rule) at the cost of two copies of the shading math
   drifting apart. The alternative refactors upstream's own sources. This is the maintainer's call;
   ask it in the PR rather than deciding it for them.
2. **Do the four HIPRT fixes go to GPUOpen, and when?** Until they land, the ROCm build instructions
   include "apply this patch", which is the weakest part of the PR. Contributing them upstream to
   HIPRT is a separate person-approved contribution; registered as a deferral.
3. **Are the `__launch_bounds__(64)` / 8x8-block / `__noinline__` choices still needed?** They were
   applied while chasing what turned out to be UB, and the notes say so explicitly ("they perturb
   inlining around the UB... the actual fix is the void return"). 64 threads per block is low
   occupancy for a shading kernel. After the rebase and with the UB fix in place, measure a 16x16
   block against the current one; keep whichever is correct and faster, and record the measurement.
   Correctness first -- this is not a reason to delay the gates.
4. **Windows without a committed DLL** -- see the windows gate above.
5. **The NVIDIA submodule ruling** remains a person's decision. The plan does not depend on it:
   nothing in the port derives from those headers, and the rebase removes the one pasted proprietary
   banner from the derivation chain.
