# kaolin port plan

Planner run 2026-11-20 on linux-gfx1100 (read-only analysis, no build, no GPU). Source read: upstream `NVIDIAGameWorks/kaolin` `master` at 393740bc0d14, fork `AMD-Ecosystem/kaolin` `master` at fcc9bfcd7bbf (= `rocm-enablement` 74a1b181b894 merged onto upstream d52da9f8). Open questions for Jeff Daily are at the end and repeated at the top of the planning section of `notes.md`; Q1 and Q2 must be ruled before the porter starts.

## Project

- Name: kaolin. Upstream https://github.com/NVIDIAGameWorks/kaolin, default branch `master`. Licence Apache-2.0 with NVIDIA-restricted files handled by the 2026-11-11 licence ruling (notes.md).
- Fork: https://github.com/AMD-Ecosystem/kaolin. Recorded `fork_default_branch` = `master`, which is the AMD team's delivery branch (ruling 2026-11-10); neither `master` nor `rocm-enablement` is modified.
- Port branch: `moat-port`, created from fork `master` (fcc9bfcd, carrying 74a1b181 with its authorship), then a merge of upstream `master`. `git merge-tree origin/master upstream/master` is clean: upstream's 3 new commits (d52da9f8..393740bc) touch only Python and tests (`kaolin/io/usd/gaussians.py`, `kaolin/metrics/pointcloud.py`, `kaolin/metrics/tetmesh.py` and their tests).

## Existing AMD support

Improvable, authoritative. The fork's `master`/`rocm-enablement` is an AMD-engineer ROCm enablement of the whole `kaolin._C` extension (87 kernels) claimed validated on MI300X/MI355X. Decision (Jeff's ruling, 2026-11-10): build on it, reduce it to the minimal diff, and extend it to wave32 and Windows. Not a community fork; its commit stays in history.

What the AMD commit actually contains (35 files, +9170/-100), as read for this plan:

| group | files | verdict |
|---|---|---|
| 20 added `kaolin/csrc/**/*_gpu.cu` | beside each `*_cuda.cu` | 11 are byte-identical to their `_cuda.cu` sibling apart from an added AMD copyright line (sided_distance, unbatched_triangle_distance, mesh_intersection, convolution, feature_grids, point_utils, query, tile_to_packed, dibr_soft_mask, rasterization, raytrace). 9 carry the real port changes (gs_to_spc, mesh_to_spc, unbatched_mcube, packed_simple_sum, bf, recon, spc, deftet, unbatched_reduced_sg_inner_product). Remove all 20; fold the needed changes into the `_cuda.cu` originals. |
| headers | `rocm_cub_compat.h` (new), `spc_math.h`, `spc_utils.cuh`, `utils.h` | needed (CubDebugExit shim, HIP vector/math-constant headers). Keep, trim comments. |
| host C++ | `bf.cpp`, `mise.cpp`, `unbatched_reduced_sg_inner_product.cpp`, `minkowski_conv.cu` | bf.cpp texture-to-buffer: needed. mise.cpp: torch-version fix, keep as a separate shared commit if the porter confirms the old 6-arg factory fails on the build torch. sg .cpp `empty_like`: perf only, drop. minkowski_conv.cu split chevrons: needed. |
| setup.py | ROCm detection, `PYTORCH_ROCM_ARCH` default, `amd-kaolin` rename, cudart_static skip, third_party/cub skip | keep (ruled), trim comments; arch default is Q2. |
| Python | `kaolin/physics/__init__.py`, `kaolin/io/usd/__init__.py`, `kaolin/ops/pointcloud.py` | `try/except ImportError` around warp; see Q1. Whitespace-only hunks in pointcloud.py: drop. |
| docs | `README.md`, `docs/notes/installation.rst` | mostly whitespace damage (re-indented RST blocks, bibtex, image tags) plus an AMD section citing an "AMD PyPI index" with no URL, a ROCm 10.0 badge and MI300 numbers. Revert the whitespace; replace the AMD prose with a short ROCm source-build subsection (see file list). |
| `tools/windows/Dockerfile.install_wheel` | `amd_kaolin-*.whl` | follows from the ruled rename; keep. |

Two facts the porter must know before reusing the AMD content:

1. **As committed, fork `master` looks unbuildable on both back ends.** `setup.py` globs `kaolin/csrc/**/*.cu`, so each `_gpu.cu` compiles beside its `_cuda.cu` twin, and the twins define the same strong extern symbols (e.g. `kaolin::sided_distance_forward_cuda_impl`), which should fail the link. On ROCm the `_cuda.cu` originals of bf/recon/spc/gs_to_spc/mesh_to_spc also keep nvcc's split chevrons (`kernel << <grid, block >> >(...)`), which torch hipify does not rewrite (`processKernelLaunches` matches only `<<<`) and clang rejects. The AMD validation numbers must therefore come from a tree that differs from the commit. Nothing is inherited on the strength of those numbers; the porter may confirm the failure with one unmodified build for the record, but should not debug it.
2. **Several AMD wave64 claims do not match the code.** Read at the line level:
   - `packed_simple_sum_cuda.cu` `ReduceBlockIntoLanes`: upstream reduces shared memory to 64 entries with `__syncthreads`, then lanes 0..31 fold `x[tid]+x[tid+32]` and butterfly 16..1. Lane 0's result depends only on lanes 0..31, so this is a correct 32-lane logical reduction on wave64 too; the AMD comment's "silently HALVE the reduction" is not what the code does. The real HIP problem is the 32-bit mask `0xffffffff` passed to `__shfl_down_sync`, which HIP rejects at compile time (64-bit mask required). Minimal fix: mask/width only.
   - `unbatched_reduced_sg_inner_product_cuda.cu`: the same holds for the forward (32-wide, leader lane 0) and backward (16-wide, offsets 8..1, leader lane 0 of each group) reductions: no cross-row contamination reaches a leader. The needed change is the mask (and the forward's `0xfffffff`, a 28-bit literal, is an upstream bug on CUDA too). The AMD file additionally rewrites the backward kernel for MI300 occupancy (second grid axis, `atomicAdd` straight into the outputs, drops the scratch caches and `at::sum_out`), switches to `fmaf`/`_EXP` (`__expf`) and `1/x` reciprocals, and changes forward output to `empty_like`. That is unguarded shared code: it changes CUDA numerics and makes the backward non-deterministic. Under the minimal-diff ruling it is dropped.
   - `deftet_cuda.cu`: a real wave64 bug. The kernel launches `dim3(32,32)` and treats each 32-lane row as one pixel; on wave64 `__ballot` spans two pixels and `__popc` counts both. Keep the AMD fix in substance (HIP-only helper selecting the caller's 32-bit half of the 64-bit vote), trimmed.

## Build classification

`torch-extension`. Evidence: `setup.py:63` imports `BuildExtension, CppExtension, CUDAExtension, CUDA_HOME` from `torch.utils.cpp_extension`; `get_extensions()` (setup.py:194-231) builds one `CUDAExtension` named `kaolin._C` from `glob('kaolin/csrc/**/*.cpp') + glob('kaolin/csrc/**/*.cu')` when `torch.cuda.is_available() or FORCE_CUDA=1`, else a `CppExtension`. No CMake. Recorded with `set-ext-type kaolin torch-extension`.

## Port strategy

Strategy B (torch build-time hipify). Sources stay in CUDA spelling; fixes that hipify cannot make go in source under `USE_ROCM`/`__HIP_PLATFORM_AMD__` guards, CUDA branch byte-identical where possible. No compat header beyond the existing `rocm_cub_compat.h` (CubDebugExit, which hipCUB lacks). Torch here is hipify v2 (`torch.utils.hipify.__version__` 2.0.0 on torch 2.16): CUDA spellings in hipified TUs are correct under v1 and v2, so no generation gate is needed. Mechanical translation is correct for every kernel; none is a tuned tensor-core kernel (no CUTLASS/CuTe/wgmma), so no AMD-native rewrite is called for. Performance work is out of scope for this round.

## CUDA surface inventory

`surface.py generate` at 393740bc: 32 files, 422 device-code markers, 87 `__global__` kernels, 69 launches, 19 warp-intrinsic sites, 27 texture sites, cub 17, cublas 13.

| surface | where | HIP mapping / risk |
|---|---|---|
| 87 kernels, 23 `.cu` | `kaolin/csrc/{metrics,ops,render}` | hipify; the split-chevron launches in `bf_cuda.cu`, `recon_cuda.cu`, `spc_cuda.cu`, `gs_to_spc_cuda.cu`, `mesh_to_spc_cuda.cu`, `minkowski_conv.cu` must become `<<<`/`>>>` (portable, nvcc accepts both). |
| `__shfl_down_sync` 32-bit masks | `packed_simple_sum_cuda.cu:62`, `unbatched_reduced_sg_inner_product_cuda.cu:129-131,315-317,351-353,383,408-414` | compile error under HIP (mask must be 64-bit). Fix: HIP branch with maskless `__shfl_down(v, off, W)` with W = the logical group width (32 or 16), CUDA branch unchanged except the 28-bit forward mask fix. |
| `__ballot_sync` + `__popc` per pixel | `deftet_cuda.cu:170-179` | wave64 correctness bug (two pixels per wavefront). HIP helper returns the caller's 32-lane half. |
| `__popc` on 32-bit octree bytes | `spc_utils.cuh:50,91,128`, `raytrace_cuda.cu:118`, `scan_octrees.cu:40`, `convolution_cuda.cu:82` | data popcounts, not lane masks; arch-independent. |
| texture objects (`tex1D`, `tex1Dfetch`) | `bf_cuda.cu` + `bf.cpp` (profile curve, `cudaArray`, normalized-float read); `unbatched_mcube_cuda.cu` (5 linear LUTs) | gfx94x disables the HIP texture API. Use the AMD approach on every ROCm target: plain device buffers read with `__ldg`, normalizing the uchar4 curve in the kernel. One path for wave64 and wave32. |
| `CubDebugExit`, `cub::DeviceScan`, `cub::DeviceRadixSort`, `using namespace cub` | `spc_utils.cuh`, `recon_cuda.cu`, `bf_cuda.cu`, others | hipCUB; `rocm_cub_compat.h` supplies CubDebugExit; `using namespace hipcub` under ROCm where the code uses unqualified names. Check rocPRIM radix-sort defects (cuda-to-rocm libraries.md): keys here are uint32/uint64 Morton codes, so check any `end_bit`=64 call. |
| `third_party/cub` submodule (NVIDIA CUB) | `setup.py get_include_dirs()` | CUDA-only; skipped on ROCm (already in the AMD setup.py). |
| cuBLAS | `convolution.cuh:23` (`cublas_v2.h`), `minkowski_conv.cu`, `convolution_cuda.cu` (`cublasHandle_t` passed through, from `at::cuda::getCurrentCUDABlasHandle`) | hipBLAS via torch; check whether any `cublas*` call remains or only the handle type travels. |
| `<math_constants.h>`, `<vector_functions.h>` | 4 `.cu` files, `spc_math.h` | hipify maps neither. `spc_math.h` ROCm branch includes `hip/hip_vector_types.h` and `hip/hip_math_constants.h` and aliases `CUDART_NORM_HUGE_F`; guard the 4 direct includes. |
| `atomicAdd` (55 sites) | throughout | float/int; check any half/double overloads under HIP. Every fresh `at::empty` accumulator must be zeroed before an atomic RMW (ROCm does not hand back zeroed memory); upstream mostly uses `at::zeros`. |
| `cudaStream_t`, `c10::cuda::CUDACachingAllocator::raw_alloc_with_stream` | mcube, others | hipify v2 keeps the masquerading CUDA spelling. |
| `__constant__` (1 site) | | no `cudaMemcpyToSymbol`; fine. |
| CUDA-samples headers | `unbatched_mcube/{defines.h,helper_math.h,tables.h}`, `spc/point_utils.h` | replaced under the 2026-11-11 licence ruling. The new `helper_math.h` must be a drop-in on both back ends: under HIP the vector types already have built-in arithmetic operators, so defining them again is ambiguous (cuda-to-rocm build-headers-macros.md, "The CUDA Samples helper_math.h"); guard the operator overloads on `!__HIP_PLATFORM_AMD__` and keep `lerp`, `make_*` helpers that HIP lacks. |
| warp-lang (NVIDIA Warp) kernels | `kaolin/physics/**`, `kaolin/ops/pointcloud.py` (farthest_point_sampling), `kaolin/io/usd/physics_materials.py`, `kaolin/experimental/newton/**` | Python, runtime-compiled by Warp, invisible to the CUDA census. Needs a ROCm Warp (Q1). |
| nvdiffrast backend | `kaolin/render/mesh/{rasterization,dibr,nvdiffrast_context}.py`, `rep/surface_mesh.py`, `easy_render/mesh.py` | optional third-party backend, licence-blocked in MOAT's dispositions (NVlabs/nvdiffrast, NSCL); tests run it only with `KAOLIN_TEST_NVDIFFRAST=1` (Q3). |
| gsplat | `kaolin/render/camera/gsplats_*.py`, `kaolin/ops/gaussians/densifier.py` | optional; the AMD-Ecosystem gsplat port exists (disposition ported-elsewhere); consume it as environment. |

## Risk list

- Wavefront: deftet ballot (real bug, fixed); reductions with a 32-bit mask (compile); every 32x32 / 16x16 block geometry stays as is. Sweep for `blockDim.x / 32`, literal-32 shared arrays and lane masks once more during the fold, on both widths.
- Wave32 (gfx1100/gfx1201) has never run this code: the AMD fork validated only gfx942/gfx950 and its default arch list excludes RDNA. Textures-as-buffers removes the one RDNA/CDNA divergence; the deftet helper takes its `warpSize == 32` branch.
- Fresh-allocation zeroing: the sg backward rewrite depended on `at::zeros`; dropping it returns to upstream's caches (already `at::zeros`). Check `at::empty` buffers fed to `atomicAdd` in raytrace/spc/conversions during the fold.
- mcube LUT buffers: upstream leaks its `raw_alloc_with_stream` buffers on every call (`initTextures` per call, `freeTextures` destroys only the texture objects); the AMD ROCm branch keeps the leak. Free them with `raw_delete` in the ROCm `freeTextures` (a one-line fix in the new branch, CUDA untouched).
- hipCUB/rocPRIM radix sort on 64-bit Morton keys with `end_bit` 64 (cuda-to-rocm libraries.md); check `DeviceRadixSort` bit ranges in `spc`/`recon`/`bf`.
- Numerics: clang's default `-ffp-contract=fast` versus nvcc; tests with tight tolerances in SPC raytrace / sg lighting may need a `TEST_WITH_ROCM`-style tolerance only after the cause is understood.
- Hipify mirror in the tree: hipify writes `*.hip` and `*_hip.{h,cuh,cpp}` siblings next to the sources (no `cuda` directories in `kaolin/csrc`). Add the patterns to `.gitignore` in the first ROCm commit and verify with `git clean -fdx`, build, `git status --porcelain` empty.
- torch version window: `TORCH_MAX_VER = '2.12.1'`; fleet torch is newer, so builds need `IGNORE_TORCH_VER=1`. Do not change upstream's bound.
- Windows: `setup.py` appends `-Xcompiler /Zc:preprocessor` to the `nvcc` key on win32, which hipcc rejects; guard to non-ROCm. `CUDA_HOME`/nvcc probes are already skipped under `IS_ROCM`. Expect the Windows torch-extension traps in cuda-to-rocm windows.md (hipcc.exe and setuptools, DLL staging).
- Jargon: the AMD comments carry in-house tags (`KB: ...` slugs, "304 CUs", gfx942 occupancy prose). Remove them when folding; run `jargon.py --port kaolin` before every push.

## File-by-file change list (commits on `moat-port`, in order)

1. **Base**: create `moat-port` from fork `master` fcc9bfcd; merge upstream `master` (merge commit). Run `set-fork-url kaolin https://github.com/AMD-Ecosystem/kaolin --by "Jeff Daily"` after the first push (ruling).
2. **Licence commits (ruling 2026-11-11), one per item**:
   - delete `kaolin/csrc/ops/spc/point_utils.h`; write fresh declarations matching what `bindings.cpp` and `point_utils.cpp`/`point_utils_cuda.cu` use.
   - delete `unbatched_mcube/defines.h`, `helper_math.h`; write replacements (only what `unbatched_mcube_cuda.cu` uses: `NTHREADS`, `make_*`, `lerp`, int3/float3 arithmetic) that compile under nvcc and HIP.
   - delete `unbatched_mcube/tables.h`; generate the replacement programmatically from the marching-cubes case rules (Lorensen and Cline 1987), with the generator committed or described in the commit body, and check the generated tables equal the behaviour of the old ones by running `tests/python/kaolin/ops/conversions/test_unbatched_mcube*` on the CUDA and ROCm builds.
   - delete `kaolin/non_commercial/`, `tests/python/kaolin/non_commercial/`, `LICENSE.NSCL`, the NSCL section of `COPYRIGHT`, `from . import non_commercial` in `kaolin/__init__.py`, `docs/modules/kaolin.non_commercial.rst` and its references in `docs/index.rst`, `docs/kaolin_ext.py`, `docs/notes/overview.rst`. The Apache-licensed `kaolin/ops/conversions/flexicubes/` stays.
3. **Fold and remove the `_gpu.cu` copies** (one commit: the originals gain the needed edits, the 20 copies are deleted, so no tree ever has both). Per original:
   - `gs_to_spc_cuda.cu`, `mesh_to_spc_cuda.cu`, `recon_cuda.cu`, `bf_cuda.cu`: guard `#include <math_constants.h>` to non-ROCm (spc_math.h supplies the HIP constants); split chevrons to `<<<`/`>>>`.
   - `spc_cuda.cu`, `minkowski_conv.cu`: split chevrons only.
   - `recon_cuda.cu`, `bf_cuda.cu`: `using namespace hipcub` under ROCm, `using namespace cub` otherwise.
   - `bf_cuda.cu` + `bf.cpp`: profile curve as a `const uchar4*` device buffer on ROCm (`KAOLIN_PROFILE_T`, `__ldg` + /255 normalization), texture object on CUDA; `bf.cpp` includes `<cuda_runtime.h>` and frees by back end.
   - `unbatched_mcube_cuda.cu`: LUTs as device buffers on ROCm (`KAOLIN_LUT_T`/`KAOLIN_LUT_FETCH`), freed in `freeTextures`; the `helper_math.h` skip becomes unnecessary once the replacement header is HIP-clean.
   - `packed_simple_sum_cuda.cu`: HIP branch of the final butterfly uses `__shfl_down(final, i, 32)`; everything else upstream.
   - `unbatched_reduced_sg_inner_product_cuda.cu`: a width-pinned shuffle helper (HIP `__shfl_down(v, off, W)`; CUDA `__shfl_down_sync(0xffffffff, v, off, W)`), forward W=32, backward W=16; nothing else.
   - `deftet_cuda.cu`: `deftet_ballot32` helper (HIP: 32-bit half of `__ballot` chosen by `__lane_id() >= 32` on wave64; CUDA: `__ballot_sync(FULL_MASK, pred)`).
4. **Headers**: keep `rocm_cub_compat.h`, the `spc_math.h` HIP include branch, and the includes in `spc_utils.cuh`/`utils.h`; strip `KB:` tags and AMD-internal prose.
5. **setup.py**: keep `is_rocm_pytorch()`/`IS_ROCM`, the arch default (Q2 decides the list), the cudart_static and third_party/cub skips, `PACKAGE_NAME = 'amd-kaolin'`; shorten comments; Windows: no `/Zc:preprocessor` on the `nvcc` key under `IS_ROCM`.
6. **Separate shared fixes (no `[ROCm]` prefix)**: `mise.cpp` `TensorOptions` form, only if the 6-argument `at::zeros`/`at::full` overload really fails on the build torch (record the error); sg forward mask `0xfffffff` -> `0xffffffff` if not already inside step 3's helper.
7. **Python warp handling**: per Q1. Independent of Q1: first check whether `import kaolin` works on a ROCm host with upstream's `warp-lang` wheel installed (it is in `tools/requirements.txt`; the PyPI wheel imports CPU-only without an NVIDIA GPU). If it does, the `try/except ImportError` guards are not needed for import and are dropped; the "not available on ROCm" `RuntimeError` text in `farthest_point_sampling` is not accurate for a ROCm Warp and is dropped with them.
8. **Copyright lines**: remove the "Modifications Copyright (C) 2026 Advanced Micro Devices" lines from upstream files (MOAT default: no new copyright lines; upstream kaolin files carry only NVIDIA's). Wholly new files (`rocm_cub_compat.h`, the four licence-replacement headers) keep an Apache header with an AMD copyright line; the replacement headers' text goes through the legal review in ruling item 4.
9. **Docs**: revert every whitespace-only hunk in `README.md` and `docs/notes/installation.rst` (bibtex, RST indentation, image tags, the inline comment); drop the "Runs on AMD Instinct" badge/section and the `amd-kaolin` pip-from-index paragraph (no such index or release exists); add one short "AMD ROCm" subsection under "Installation from source" in `installation.rst` with the real source-build steps (ROCm torch, `PYTORCH_ROCM_ARCH`, `IGNORE_TORCH_VER` when outside the window, `python setup.py develop`), and one README line pointing to it. No device-selection variables in README commands.
10. **.gitignore**: hipify mirror patterns (`*.hip`, `*_hip.h`, `*_hip.cuh`, `*_hip.cpp`), verified by a clean build.

## Build commands

Linux (each executing host, in the job venv with a ROCm torch; fleet hosts carry one in the image Python, use `$FLEET_VENV` for a build):

```
git clone https://github.com/AMD-Ecosystem/kaolin -b moat-port && cd kaolin
python -m pip install -r tools/build_requirements.txt -r tools/requirements.txt
export IGNORE_TORCH_VER=1            # torch newer than setup.py's TORCH_MAX_VER
export PYTORCH_ROCM_ARCH=gfx1100     # this host's target (gfx942, gfx950, gfx1201, ...)
export MAX_JOBS=<physical cores>
python setup.py develop 2>&1 | tee build.log
python -c "import kaolin, torch; print(kaolin.__version__, torch.version.hip)"
git status --porcelain               # must be empty after the build
```
`FORCE_CUDA=1` is needed only to build without a visible GPU. The `third_party/cub` submodule is not needed on ROCm.

Windows gfx1151 (last, after both Linux gates pass at one head): TheRock ROCm SDK plus the matching ROCm torch wheel with `[device-gfx1151]`, `PYTORCH_ROCM_ARCH=gfx1151`, same `setup.py develop`; follow cuda-to-rocm windows.md for hipcc.exe/setuptools and DLL placement.

CUDA no-regression (ruling item 3: the licence replacements change the CUDA build): on the mi355 fleet host, a separate venv with a CUDA torch wheel (the host torch is ROCm-only, cuda-to-rocm validation.md "A torch extension needs a CUDA-built torch for this check"), `CUDA_HOME` pointing at a CUDA 12.x toolkit, `FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0;9.0" IGNORE_TORCH_VER=1 python setup.py build_ext`. A run on the V100 fleet host additionally executes the CUDA path (needed for the 88-failure comparison below).

## Test plan

Real suite: pytest over `tests/python/` (114 test files), as upstream CI does in `ci/gitlab_jenkins_templates/ubuntu_test_cuda_CI.jenkins`:

```
pytest --durations=50 --import-mode=importlib -rs \
  --log-disable=PIL.PngImagePlugin --log-disable=PIL.TiffImagePlugin \
  --log-disable=kaolin.rep.surface_mesh tests/python/
pytest --doctest-modules --ignore=kaolin/experimental kaolin
```

- GPU-heavy groups that exercise the ported kernels: `tests/python/kaolin/ops/spc`, `ops/conversions` (mcube, mesh_to_spc, gs_to_spc, flexicubes), `ops/mesh`, `ops/test_reduction.py` (packed_simple_sum), `ops/test_batch.py`, `metrics` (sided_distance, triangle distance), `render/mesh` (rasterization, dibr soft mask, deftet), `render/lighting` (sg inner product), `render/spc` (raytrace), `rep/test_rep_spc.py`. Run these first on each platform and report pass/fail/skip per group.
- Dataset-gated tests (ShapeNet, ModelNet, SHREC, scanned toys, tensor IR) skip without the `KAOLIN_TEST_*` paths; record them as skipped-by-design, not as coverage.
- nvdiffrast tests run only with `KAOLIN_TEST_NVDIFFRAST=1`; leave it unset (Q3).
- gsplat-dependent tests (`render/camera/test_gsplats_*`, `ops/gaussians/test_densifier.py`): install the AMD-Ecosystem gsplat port into the job venv if it builds in budget; otherwise record them skipped with the reason.
- warp-dependent groups (`tests/python/kaolin/physics/**`, `experimental/newton/**`, FPS in `ops/test_pointcloud.py`): per Q1.
- Non-GPU regression set that must not change: `io/` (obj, gltf, ply, off, usd), `math/quat`, `rep/test_tensor_container.py`, `utils/`, `visualize/`, and the doctests; upstream's CPU CI runs `io/test_gltf.py`, `io/test_obj.py` and the doctests.
- Baseline comparison (ruling): the AMD README's MI300X "88 failed (all non-port)" label is not accepted. Run the same suite on the V100 host against upstream CUDA kaolin at the same upstream base, and classify every ROCm failure as also-fails-on-CUDA, environment/dataset, or port; only the first two may ship.
- Bit-level checks for the three touched reductions: packed_simple_sum, sg inner product forward/backward and deftet outputs compared against the CPU/reference paths the tests already use, on wave64 and wave32. The sg backward stays deterministic (no atomics added), so repeat runs must match bit for bit.
- Gates: wave64 (gfx942 or gfx950), wave32 (gfx1100 or gfx1201) on Linux first, then windows (gfx1151).

## Open questions (for Jeff Daily; the porter must not assume answers)

**Q1. Warp-dependent features: port now, defer, or wait?** `kaolin.physics` (Simplicits, materials, common, utils), `ops.pointcloud.farthest_point_sampling`, `io.usd.physics_materials` and `experimental.newton` run Warp kernels on the GPU; on ROCm they need a ROCm Warp. MOAT's own Warp release line `warp-1.14` (AMD-Ecosystem/warp, `release/1.14.0.amd0`) is at stage `porting`, Phase 0 of its plan done on 2026-11-18 (runtime kernel compilation still a stub then), so it cannot carry these yet. Newton (`newton[examples]==1.2.0`) is a Warp application with no MOAT project. Options:
  - (a) Port and validate the `_C` extension and everything not using Warp now; record the warp-dependent features as `owed` in surface.json, so the port can complete its platforms but `pr_ready` refuses until they are validated with the `warp-1.14` build; Newton coupling stays out unless a Newton project is adopted.
  - (b) Scope the warp-dependent features out of this port (surface `scoped_out` with your ruling as the reason) and register a deferral against kaolin to revisit when `warp-1.14` ships.
  - (c) Hold kaolin's porter until `warp-1.14` completes, then port everything in one pass.
  The planner's recommendation is (a): the extension port does not depend on Warp at build time, so nothing blocks it, and `owed` keeps the gap visible instead of cutting it.

**Q2. `PYTORCH_ROCM_ARCH` default in setup.py.** Your ruling keeps AMD's default of `gfx942;gfx950` when the variable is unset and reserved its extension to wave32 for you. With wave32 and Windows in scope, a user building on a Radeon or Strix Halo machine without setting the variable gets a wheel with no code for their GPU. Options: (a) keep `gfx942;gfx950`; (b) extend it with the RDNA targets the gates cover (`gfx1100;gfx1101;gfx1151;gfx1200;gfx1201`); (c) drop the override and let torch's own default apply (`torch.utils.cpp_extension._get_rocm_arch_flags`: the architectures of the GPUs visible at build time, so a build on a Radeon machine targets that Radeon; on torch 2.16 a GPU-less `FORCE_CUDA=1` build then gets an empty `--offload-arch=` and has no fallback, so (c) would also need a fallback for that case). Validators set the variable explicitly either way, so this changes only what users get.

**Q3. nvdiffrast backend.** kaolin's mesh rasterization can use nvdiffrast as an optional backend; NVlabs/nvdiffrast is declined in MOAT as licence-blocked (NSCL). Plan default: leave kaolin's nvdiffrast code paths untouched, do not install nvdiffrast, keep `KAOLIN_TEST_NVDIFFRAST` unset, and record the backend in surface.json as `scoped_out` citing that disposition. Confirm, or say otherwise.

**Q4. AMD copyright lines and docs.** Under the minimal-diff ruling the plan removes the "Modifications Copyright ... Advanced Micro Devices" lines from upstream files, removes the AMD README section and the `amd-kaolin` pip-from-AMD-index paragraph, and adds a short ROCm source-build subsection. These are the AMD engineers' additions; confirm the removal is wanted, or name which to keep.

Not questions (already ruled, listed so nothing is a surprise): the sg inner-product backward performance rewrite and the `empty_like` change are dropped as unneeded by the port (2026-11-10 ruling); the delivery vehicle is decided later.
