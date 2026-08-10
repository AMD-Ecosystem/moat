# EnvGS -- ROCm/HIP Port Plan (lead platform: linux-gfx90a, MI250X, ROCm 7.2.1)

## Project
- Name: EnvGS (CVPR 2025, "Environment Gaussian Splatting" -- 3DGS/2DGS + an environment-Gaussian component for view-dependent appearance / reflections)
- Upstream: https://github.com/zju3dv/EnvGS  (default branch `main`, base_sha 4dce5a41b2f614faca3169a94637ad02caba25b6)
- EnvGS is built on the `easyvolcap` framework (the repo's top-level Python package). The CUDA work is entirely in three git submodules; easyvolcap itself is pure Python/PyTorch glue.
- Host verified: torch 2.13.0a0+gitb5e90ff, HIP 7.2.53, `torch.cuda.is_available()=True`, device "AMD Instinct MI250X / MI250", `gcnArchName=gfx90a:sramecc+:xnack-`. Build cwd must stay OUTSIDE /var/lib/jenkins/pytorch.

## TL;DR for the porter
- The repo has TWO native GPU components, in submodules:
  1. `submodules/diff-surfel-rasterizations` -- the base 2D Gaussian (surfel) rasterizer. **EASY**, fused-ssim/gaussian_splatting class. Strategy B (torch hipify). NO warp primitives at all (uses only `cg::this_thread_block`/`cg::this_grid` + per-thread `atomicAdd`). EnvGS installs exactly 3 of its ~17 variant dirs: `diff-surfel-rasterization-wet`, `-wet-ch05`, `-wet-ch07`.
  2. `submodules/diff-surfel-tracing` -- the environment-Gaussian REFLECTION path. **HARD WALL: it is an NVIDIA OptiX 7.7 ray-tracing pipeline** (BVH `optixAccelBuild`, `optixTrace`, `__raygen__`/`__anyhit__`/`__closesthit__`/`__miss__` programs compiled to PTX, SBT). OptiX has no ROCm equivalent. This is the project's defining risk and is NOT a mechanical hipify.
  3. `submodules/StableNormal` -- pure-Python diffusion model for monocular-normal preprocessing (data prep). No CUDA. Out of scope for this port.
- Recommended disposition: PORT in two stages. Stage 1 (base 2DGS rasterizer, the 3 `-wet*` variants) is a near-trivial Strategy B port that mirrors gaussian_splatting/fused-ssim and delivers a real GPU-validated EnvGS diffuse-pass result. Stage 2 (the OptiX env-Gaussian reflection tracer) is gated on porting OptiX ray-Gaussian tracing to a ROCm ray-tracing backend (HIP-RT) -- a substantial separate effort; flag the risk now, decide port-vs-defer after Stage 1 validates.

## Existing AMD support
None. No ROCm/HIP path anywhere -- no `USE_ROCM`/`hip`/`amd` in either submodule, no AMD fork of `xbillowy/diff-surfel-rasterizations` or `xbillowy/diff-surfel-tracing`, and no OpenCL/Vulkan alternative. The base rasterizer is plain CUDA via torch `CUDAExtension`; the tracer is CUDA+OptiX. Decision: a fresh ROCm/HIP port adds clear value (Stage 1 immediately; Stage 2 is real new AMD-native work).

## Build classification: torch-extension (Strategy B) for BOTH submodules
Evidence:
- `diff-surfel-rasterization-wet*/setup.py`: `from torch.utils.cpp_extension import CUDAExtension, BuildExtension` building `diff_surfel_rasterization._C` from `cuda_rasterizer/{rasterizer_impl,forward,backward}.cu`, `rasterize_points.cu`, `ext.cpp`. Pure torch CUDAExtension -> Strategy B.
- `diff-surfel-tracing/setup.py`: also a `CUDAExtension` (`diff_surfel_tracing._C` from `optix_tracer/common.cpp`, `optix_tracer/optix_wrapper.cpp`, `trace_surfels.cpp`, `ext.cpp`), PLUS a `CustomBuildExtension` that, after the torch build, shells out to a SEPARATE CMake build (`cmake .. && make`) of `optix_tracer/{forward,backward}.cu` compiled to **PTX** (`CUDA_PTX_COMPILATION ON`, `CMake/PTXUtilities.cmake`) and copies `build/ptx/*.ptx` into the installed package. So the tracer is torch-extension for its host glue + a CMake/PTX OptiX device pipeline. The OptiX device programs are loaded at runtime from PTX by `optixModuleCreate` (not linked into `_C`).
- EnvGS top-level (`pyproject.toml`) is `easyvolcap`, a pure-Python setuptools package (no ext_modules). The CUDA build lives only in the submodules.
- `ext_type` set to `torch-extension` in upstream.json + status.json.

## Port strategy
- Stage 1 -- base 2DGS rasterizer (the 3 `-wet*` variants): Strategy B. Build each against the ROCm torch; torch auto-hipifies the `.cu`/`.cuh` and passes `-DUSE_ROCM=1`. Expect near-zero source edits (see CUDA surface). Mirrors gaussian_splatting (single repo, Strategy B) and fused-ssim (no warp primitives).
- Stage 2 -- OptiX env-Gaussian tracer: NOT a hipify. OptiX does not exist on ROCm. Options, in order of preference:
  (a) Reimplement the ray-Gaussian-disk tracing on AMD ray tracing via **HIP-RT** (AMD's ray-tracing library: BVH build + traversal; the closest analogue to OptiX GAS + `optixTrace`). The per-ray surfel intersection + alpha-compositing logic in `optix_tracer/{forward,backward}.cu` (the `__anyhit__`/`__raygen__` bodies) is portable math; the OptiX *plumbing* (module/pipeline/SBT/GAS) is what must be re-expressed in HIP-RT's API.
  (b) A custom HIP BVH-traversal kernel (build BVH on triangles = 2 triangles per 2DGS disk, traverse in a plain HIP kernel, run the same intersection math). More work than HIP-RT but no extra dependency; its own wave64 considerations (none today since the device code is per-ray serial + atomicAdd).
  (c) Defer Stage 2 and validate EnvGS's diffuse/rasterized path only (see Test plan): `use_optix_tracing=False` / `use_base_tracing=False` runs the model through the rasterizer, not the tracer.
  Decision: do Stage 1 first and GPU-validate it; bring HIP-RT feasibility (a) into a follow-up. A correctness-first rasterizer port is a valid, valuable first deliverable even if the OptiX reflection path is deferred -- but note (below) that the env-Gaussian reflection, EnvGS's headline feature, is fundamentally OptiX-bound in this codebase and has no software fallback.

## CUDA surface inventory
### A. Base rasterizer (`diff-surfel-rasterization-wet`, `-wet-ch05`, `-wet-ch07`; 536-line forward.cu each, identical structure, differ only in SH/specular channel count)
- Kernels: standard 2DGS forward (`preprocessCUDA`, `renderCUDA`) + backward (`preprocessCUDA`, `renderCUDA`), launched from `rasterizer_impl.cu`. `ext.cpp` exports `rasterize_gaussians`, `rasterize_gaussians_backward`, `mark_visible`.
- Cooperative groups: ONLY `cg::this_thread_block()` (block.sync, block.thread_rank) and `cg::this_grid().thread_rank()`. **No `cg::reduce`, no `tiled_partition<32>`, no `.shfl*`.** (Confirmed by grep across all 3 installed variants: zero warp-level calls.)
- Warp intrinsics: NONE. No `__shfl*`, `__ballot`, `__activemask`, `__any/__all`, `__reduce_*_sync`, `warpSize`, inline PTX, or `half2` masks anywhere. `auxiliary.h` defines `NUM_WARPS (BLOCK_SIZE/32)` but it is unused by the reduction (the block is 16x16=256 and all cross-thread exchange is via `__shared__` + `block.sync()`; gradients accumulate via direct per-thread `atomicAdd`).
- Atomics: `atomicAdd(float*)` for gradient accumulation (15 sites in backward.cu) -- portable 1:1 to HIP, fully supported on gfx90a device memory. Float-sum order across threads is non-deterministic on every GPU (CUDA included); this is the standard 3DGS gradient-tolerance situation, validated by gradcheck within tolerance, not bitwise.
- Includes the `<cooperative_groups/reduce.h>` header (HIP CG lacks it -- gsplat lesson) but never calls into it. If torch-hipify leaves it verbatim and clang errors on the unused include under HIP, guard it `#ifndef USE_ROCM`; most likely it is parsed harmlessly. Watch this at first compile.
- GLM: bundled at `submodules/diff-surfel-rasterizations/third_party/glm` (one shared `third_party/glm` for all variants; `setup.py` adds `-I../third_party/glm`). Used heavily in device code (`glm::mat3/vec3/mat3x4`, `quat_to_rotmat`, etc.). This is the gsplat GLM-in-a-submodule hipify trap (below).
- No cuBLAS/cuFFT/cuRAND/cuSPARSE/Thrust/CUB. No textures/surfaces. No pinned/managed memory. No `cudaMalloc`/streams in the extension (torch owns memory; kernels take raw tensor pointers). No `__CUDA_ARCH__` guards in the `.cu`.
- libcu++/`cuda::std`: none in the rasterizer. (No libhipcxx needed for Stage 1, unlike gsplat's 3DGUT path.)

### B. OptiX env-Gaussian reflection tracer (`diff-surfel-tracing`)
- Host glue (`trace_surfels.cpp`, `optix_tracer/optix_wrapper.cpp`, `common.cpp`): builds the OptiX context, modules, pipelines (forward + backward), SBT; builds the BVH GAS from triangles (`optixAccelBuild`, `OPTIX_BUILD_INPUT_TYPE_TRIANGLES`, FLOAT3 vertices / UINT3 indices, ALLOW_COMPACTION|UPDATE|RANDOM_VERTEX_ACCESS); launches via `optixLaunch(... H, W, 1)`. Uses `at::cuda::getCurrentCUDAStream()`, plain `cudaMalloc`/`cudaMemcpy`/`cudaFree`/`cudaStreamSynchronize` for the `Params` block and GAS buffers (these specific CUDA-runtime calls hipify 1:1).
- Device programs (`optix_tracer/forward.cu`, `backward.cu`): OptiX shader programs `extern "C" __global__ void __raygen__ot/__anyhit__ot/__closesthit__/__miss__`, using `optixTrace`, `optixGetPayload_*`, `optixGetLaunchIndex/Dimensions`, `optixGetRayTmax`, `optixGetPrimitiveIndex`, `optixIgnoreIntersection`, `OptixVisibilityMask`, `OPTIX_RAY_FLAG_*`. Compiled to PTX (not object), loaded by the OptiX runtime. **None of this exists on ROCm.** No warp primitives in the tracer device code either (per-ray serial alpha compositing + `atomicAdd` gradient scatter, ~19 atomicAdd sites in backward.cu).
- third_party submodule: `third_party/optix` = `NVIDIA/optix-dev.git` @ OptiX 7.7.0 (`OPTIX_VERSION 70700`) -- headers only; the OptiX implementation ships in the NVIDIA driver and is unavailable on AMD.
- Autograd: `diff_surfel_tracing/__init__.py` wraps a `torch.autograd.Function _TraceSurfels` (forward = trace, backward = trace-backward pipeline). The Python API is `SurfelTracer(SurfelTracingSettings)` and is consumed by `easyvolcap/utils/optix_utils.py::HardwareRendering`.

## Risk list
- **OptiX is a hard ROCm wall (Stage 2, the headline risk).** No OptiX on ROCm, and `diff-surfel-tracing` has NO software/CPU tracing fallback (grep: none). The env-Gaussian REFLECTION pass in EnvGS is wired ONLY through `HardwareRendering` (OptiX): `easyvolcap/models/samplers/envgs_sampler.py` lines 535-544 always call `self.diffop.render_gaussians(...)` (= OptiX) for the reflected rays once `iter >= render_reflection_start_iter`. So the reflections cannot be produced at all without either porting the tracer to HIP-RT/custom-BVH or accepting a diffuse-only result. Quantify HIP-RT feasibility (BVH build/update + traversal + custom intersection + a backward/gradient pass through traversal) before committing Stage 2.
- **GLM-in-a-submodule hipify corruption (Stage 1, gsplat precedent).** torch's AOT hipify walks every `.hpp` under the build cwd + include dirs and content-rewrites GLM headers, dropping GLM's `.inl` and breaking its `__CUDACC__`/`CUDA_VERSION` detection ("GLM requires CUDA 7.0", missing `.inl`, "host fn from device fn"). Fix per gsplat: monkeypatch `hipify_python.hipify` in each `-wet*/setup.py` to add the `third_party/glm` dir to `ignores` AND strip it from `header_include_dirs`; GLM 1.0.x already detects `__HIP__` and compiles verbatim under `-x hip`. MUST init the GLM submodule first (an empty submodule silently falls back to system GLM 0.9.9 with no HIP detection). Here the pinned GLM is g-truc/glm @ 5c46b9c -- verify it is a 1.0.x (HIP-aware) tag; if it is 0.9.9.x, the GLM `__CUDACC__`/`GLM_FORCE_CUDA` steering trick (3DUNDERWORLD-SLS lesson) applies instead.
- **`<cooperative_groups/reduce.h>` include present but unused (Stage 1).** HIP CG lacks it. Likely harmless if never instantiated; if clang errors, `#ifndef USE_ROCM`-guard the include. Low risk.
- **hipify std::min/std::max -> ::min/::max on the host pass (Stage 1, gsplat precedent).** If any `.cu` uses `std::min/std::max` reachable from the host compile, provide the host-only global forwarders. Grep at port time.
- **`cudaFuncSetAttribute`/`__grid_constant__`/`cudaFuncAttributeMaxDynamicSharedMemorySize`** if present in `rasterizer_impl.cu` -- cast kernel to `(const void*)` for `hipFuncSetAttribute` (gsplat lesson). Grep at port time.
- **Float gradient determinism (both stages).** Per-thread `atomicAdd` gradient accumulation is order-nondeterministic on all GPUs. Validate via gradcheck/finite-difference within tolerance and via output-image SSIM/PSNR against the CUDA reference or against a short-training convergence curve -- not bitwise. (gaussian_splatting/gsplat precedent.)
- **No warp-size (wave64) fault class in either component** -- confirmed by exhaustive grep (no `__shfl`/`__ballot`/`warpSize`/`__reduce_*_sync`/tiled_partition/shfl_xor anywhere). This is the fused-ssim outcome, NOT the LiteGS hard case. gfx1100/gfx1151 (wave32) followers should need no wave-related delta for the rasterizer; the tracer's wave behavior is moot until Stage 2 exists.
- **OOB neighbor reads / texture pitch / library swaps**: none applicable (no stencils, no textures, no CUDA math libs). The 256B-pitch and texture fault classes do not apply.

## File-by-file change list
Stage 1 (base rasterizer; per each of the 3 installed variants -- changes are identical, ideally factored):
- `submodules/diff-surfel-rasterizations/diff-surfel-rasterization-wet{,-ch05,-ch07}/setup.py`: add the gsplat GLM-ignore hipify monkeypatch (add `third_party/glm` to hipify `ignores`, strip from `header_include_dirs`). No other setup change (torch's HIP CUDAExtension already passes `-DUSE_ROCM=1`).
- `.../cuda_rasterizer/forward.cu`, `backward.cu`: expected NO edits. Contingency only: guard the unused `<cooperative_groups/reduce.h>` include and/or add host min/max forwarders IF first compile demands it. Confirm by building, not preemptively.
- `submodules/diff-surfel-rasterizations/third_party/glm`: initialize the submodule (no edit); verify version is HIP-aware (1.0.x).
- Likely total Stage 1 source edits: the 3 setup.py monkeypatches, plus at most a 1-line include guard. This matches fused-ssim ("torch hipify + build is the whole port").

Stage 2 (OptiX tracer -- only if pursued):
- New ROCm backend for `diff-surfel-tracing`: replace the OptiX context/module/pipeline/SBT/GAS plumbing (`optix_tracer/optix_wrapper.{cpp,h}`, `common.{cpp,h}`, `trace_surfels.cpp`, `params.h`) with a HIP-RT (or custom HIP BVH) equivalent; port the `__raygen__`/`__anyhit__` intersection + compositing math from `optix_tracer/{forward,backward}.cu` into HIP-RT trace callbacks or a HIP traversal kernel. New CMake to build the HIP device code (drop the PTX-compilation path; HIP-RT/HIP uses code objects, not OptiX-IR/PTX). Keep `diff_surfel_tracing/__init__.py` autograd API and `ext.cpp` bindings stable so `easyvolcap/utils/optix_utils.py` is unchanged. This is a rewrite, scoped as its own task after Stage 1.

## Build commands (gfx90a; cwd OUTSIDE /var/lib/jenkins/pytorch)
Stage 1 (per the gsplat recipe; run from each variant dir or a wrapper):
```
git -C projects/EnvGS/src submodule update --init --recursive   # GLM, etc.
export PYTORCH_ROCM_ARCH=gfx90a
# for each of the 3 installed rasterizer variants:
cd submodules/diff-surfel-rasterizations/diff-surfel-rasterization-wet
rm -rf build *.egg-info *_hip.* cuda_rasterizer/*_hip.*   # force re-hipify on edited .cu
python -m pip install -e . --no-build-isolation --no-deps -v
# repeat for -wet-ch05 and -wet-ch07
```
Followers (gfx1100/gfx1151): same, only `PYTORCH_ROCM_ARCH=gfx1100|gfx1151`, clean rebuild, expected no source edit.

Stage 2: deferred (HIP-RT toolchain + new CMake; define when Stage 2 is scoped).

## Test plan (real GPU; define "validated on gfx90a")
The repo's `tests/` are mostly easyvolcap research scripts (OpenGL/zmq/pulsar/cupy), not a CUDA unit suite; relevant ones: `tests/hardware_splatting_tests.py`, `tests/diff_gauss_tests.py`, `tests/bad_covariance_tests.py`, `tests/bvh_vs_pytorch3d_knn_tests.py`. Primary validation mirrors the splatting precedents:
- Stage 1 "validated on gfx90a" =
  1. All 3 `-wet*` extensions import and run a forward `rasterize_gaussians` on a small Gaussian set, producing a finite image (no illegal-memory-access; `tests/cuda_illegal_memory_*` clean).
  2. Backward gradcheck / finite-difference on the rasterizer outputs (means2D, opacity, colors, transMat) within 3DGS gradient tolerance (matches gaussian_splatting/gsplat).
  3. A short EnvGS training run on a small scene with `use_optix_tracing=False`/`use_base_tracing=False` (diffuse/rasterized path, e.g. a `2dgs` config) for a few hundred iterations: loss decreases monotonically-ish and PSNR rises -- proving the rasterizer forward+backward train end-to-end on GPU. Confirm device execution via `AMD_LOG_LEVEL=3` showing the rasterizer kernels dispatched.
  4. Determinism bar: run-to-run image/loss agreement to float/print precision (atomicAdd order varies), NOT bitwise.
- Stage 2 "validated on gfx90a" (only if pursued) = the full EnvGS reflection path: build BVH, trace reflected rays through the HIP-RT/custom backend, and reproduce the published reflective-scene result (e.g. ref_real / shiny scenes) -- forward render matching the CUDA/OptiX reference within image-metric tolerance, plus a short reflective-scene training showing convergence. Until Stage 2 exists, the reflection feature is unvalidated by construction.
- Non-GPU regression set: easyvolcap must still import and its CPU utilities run; do not regress the pure-Python framework. The OpenGL/zmq/pulsar tests are environment-dependent (headless GL) and are not GPU-correctness gates for this port.

## Open questions
- HIP-RT availability + maturity on this ROCm 7.2.1 host (is `hiprt` installed / installable?), and whether its BVH build/update + traversal + custom-intersection + a differentiable backward traversal can express the OptiX surfel tracer. This decides Stage 2 port-vs-defer. (Probe at Stage 2 start; `import hiprt` currently fails and no OptiX SDK is on the host.)
- Pinned GLM version (g-truc/glm @ 5c46b9c): confirm it is 1.0.x (HIP-aware) so the gsplat ignore-monkeypatch suffices; if 0.9.9.x, use the GLM `__CUDACC__`/`GLM_FORCE_CUDA` steering trick instead.
- Whether jeff wants Stage 1 (diffuse rasterizer) landed and validated as the deliverable now, with the OptiX reflection tracer (Stage 2) tracked as a separate HIP-RT task -- or whether EnvGS should wait until the reflection path is portable. Recommendation: land Stage 1 (real, validated AMD value, low risk), open Stage 2 as a scoped HIP-RT follow-up, and do not skip the project.
