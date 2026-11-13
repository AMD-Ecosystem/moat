# diff-surfel-rasterizations port plan

## Project

- Name: `diff-surfel-rasterizations`
- Upstream: https://github.com/xbillowy/diff-surfel-rasterizations, default branch `main`, HEAD `1aa433c`
- Fork: https://github.com/AMD-Ecosystem/diff-surfel-rasterizations; `main` is a clean mirror of upstream `1aa433c`; `moat-port` is 3 commits ahead at `4c95346`
- What it is: a collection of 14 differentiable 2D-Gaussian (surfel) rasterizers derived from
  Inria 3DGS / 2DGS, each a separate directory shipping the same kernels with a different
  `NUM_CHANNELS` (and one with a 1x1 tile). Each directory is an independent PyTorch
  extension package.
- Planned on linux-gfx942, 2026-08-13.

## Existing AMD support: improvable, and it is ours

Intake settled licence and duplicate effort; this section only records the finer judgement
the skill's `assess-existing-support.md` asks for.

There is no third-party AMD or ROCm effort. Upstream has two forks; one is ours. The
existing AMD support on `moat-port` is MOAT's own stage-1 EnvGS work, landed before this
project had a record of its own. Applying the authoritativeness test: it is not a community
hint to be treated as suspect, it is our own code, already GPU-exercised. **Decision:
inherit it unchanged and extend it.** Nothing on `moat-port` is re-done, re-designed or
rebased.

### What the existing 3-variant port already proves

`moat-port` covers `diff-surfel-rasterization-wet`, `-wet-ch05` and `-wet-ch07` -- 22 files,
+275/-48 -- and EnvGS (`review-passed`, `head_sha 7528e8db`) validated it on
**linux-gfx90a, linux-gfx1100, windows-gfx1101 and windows-gfx1201**. From EnvGS's notes,
that run established, for the wet family:

- the extensions compile with `-DUSE_ROCM=1 -x hip`, import, and export
  `rasterize_gaussians` / `rasterize_gaussians_backward` / `mark_visible`;
- forward renders a finite, non-trivial image (~1778/4000 surfels visible) with no illegal
  memory access, and is bit-identical run to run;
- backward gradients are finite everywhere, the opacity finite-difference matches the
  analytic gradient (slope ~1.00), and a short Adam fit drops the loss ~95% to PSNR 25-26 dB
  for all three channel counts;
- `AMD_LOG_LEVEL=3` confirms `preprocessCUDA`, `renderCUDA`, `duplicateWithKeys` and
  `identifyTileRanges` dispatch on the AMD device;
- CUB (`DeviceScan::InclusiveSum`, `DeviceRadixSort::SortPairs`) hipifies cleanly to
  hipCUB/rocPRIM, cooperative groups reduce to `this_grid()`/`this_thread_block()`, and GLM
  device math compiles verbatim.

So the *technique* is proven on both wavefront families and on Windows. What is unproven is
**coverage**: 11 of 14 variants have never been compiled on AMD, and the two code bodies
those 11 contain (`base` and `wet-abs`) have never been compiled on AMD at all. That, plus
the total absence of a test suite tying any of it to this repository, is what this port has
to fix.

### Delivery vehicle: the fork is the deliverable, not an upstream PR

Recorded here because it changes what "done" means. Two independent reasons there is no
upstream PR destination (intake's finding, restated because the plan depends on it):

1. Tier-4 Inria Gaussian-Splatting licence, non-commercial, contribution requires the
   copyright holder's written permission by email -- and the substantive copyright is
   Inria/MPII GRAPHDECO's, not the repository owner's.
2. The repository is 2 stars with zero pull requests and zero issues ever opened. It is one
   researcher's collection of channel-count variants, not a library accepting patches.

Consequence for the plan: the "minimal upstream-mergeable diff" constraint that normally
governs footprint does not bind here. It is still worth honouring for the ported sources --
we do not want a fork that has diverged in style from upstream -- but it does **not** forbid
adding a test directory, which is the only way this repository can ever be validated on its
own. See "Test plan".

### Port versus AMD-native rewrite

Not applicable in the interesting sense. These kernels are hand-written scalar CUDA -- no
CUTLASS, no CuTe, no wgmma, no warp specialization, no tensor-core path of any kind. There
is nothing an AMD-native rewrite (rocWMMA, Composable Kernel, MFMA) would have to replace.
A mechanical HIP translation is not a first step here, it is the right and final answer.

## Build classification: torch-extension

Evidence, per variant directory (identical in all 14):

- `<variant>/setup.py` line 13: `from torch.utils.cpp_extension import CUDAExtension, BuildExtension`
- `<variant>/setup.py`: `ext_modules=[CUDAExtension(name="<pkg>._C", sources=[... .cu, ext.cpp])]`,
  `cmdclass={'build_ext': BuildExtension}`
- `<variant>/ext.cpp`: `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` binding the three entry points

`ext_type` set to `torch-extension` on this record.

A second, non-authoritative build path exists and is scoped out: each variant also ships a
`CMakeLists.txt` building a static `CudaRasterizer` library (`project(DiffRast LANGUAGES CUDA CXX)`).
It is dead as shipped -- `target_include_directories(CudaRasterizer PRIVATE third_party/glm ...)`
points at a per-variant `third_party/` that contains only `stbi_image_write.h`, and the
per-variant `.gitmodules` declaring `third_party/glm` registers no gitlink (`git submodule
status` lists only the repository-root `third_party/glm`). The CMake path therefore cannot
configure on CUDA either. Scoped out in `surface.json`, and registered as a deferral so it is
reviewed with this port rather than silently dropped.

## Port strategy: B (torch hipify)

Torch's build-time hipify runs automatically for a `CUDAExtension` on a ROCm torch; sources
stay in CUDA spelling and only what hipify cannot handle is fixed in source, guarded by
`USE_ROCM`. That is exactly the shape of the existing `moat-port` work, so continuing with
Strategy B also means the 11 new variants get byte-identical treatment to the 3 proven ones.

hipify generation is a non-issue for this project, and the check is recorded so nobody
repeats it: the whole tree contains **no** `c10::`, `at::cuda`, `getCurrentCUDAStream`,
`CUDAGuard` or `AT_CUDA*` reference (grep over all `.cu`/`.cpp`/`.h`, zero hits). The
masquerading-API split between hipify v1 and v2 cannot bite code that never names those
classes -- including the Windows `ext_winhip.cu` TU, whose only torch include is
`<torch/extension.h>`. No `TORCH_HIPIFY_V2` branch is needed. (This host carries hipify
2.0.0 with torch 2.14; the original stage-1 work was done against hipify 1.x. Both are fine.)

### The five source edits, and why each is needed

These are the hunks already on `moat-port` for the wet variants. Each is a genuine
nvcc-versus-clang difference, not cosmetic, and each replicates unchanged to the other 11.

1. **Kernel launch chevrons.** Upstream writes `renderCUDA<NUM_CHANNELS> << <grid, block >> >(...)`.
   nvcc's EDG front end accepts the split spelling; clang lexes `<<<` as a single
   `lesslessless` token and rejects `<< <`. Respaced to `<<<grid, block >>>`. Semantically
   identical on nvcc, so the CUDA path is untouched. 7 launch sites per variant.
2. **`device_launch_parameters.h`.** No such header in HIP. Guarded `#if !defined(USE_ROCM)`.
   Its contents (`threadIdx` and friends) are intrinsic under hipcc.
3. **`<cooperative_groups/reduce.h>`.** No such header in HIP's cooperative groups. Guarded
   the same way; safe because the code uses only `cg::this_grid()`, `cg::this_thread_block()`
   and `thread_rank()` -- there is no `cg::reduce`, no `tiled_partition`, no
   `coalesced_threads` anywhere in the tree.
4. **`__trap()`.** Not declared by HIP's device runtime. `#define __trap __builtin_trap`
   under `USE_ROCM` in `auxiliary.h`. One call site (the `prefiltered` invariant violation).
5. **`setup.py`: keep bundled GLM out of hipify.** torch's hipify walks every `.hpp` under
   the extension include dirs into its file set and content-rewrites the GLM headers a source
   pulls in. That drops GLM's `.inl` files (hipify copies only `.hpp`/`.h`) and mangles GLM's
   own `__CUDACC__`/`__HIP__` compiler detection, breaking the build. Bundled GLM already
   detects `__HIP__` (`glm/simd/platform.h` -> `GLM_COMPILER_HIP`) and compiles verbatim under
   `-x hip`, so the fix is to leave it alone: monkeypatch `hipify_python.hipify` to add the
   GLM dir to `ignores` and drop it from `header_include_dirs`. Gated on `torch.version.hip`,
   so the CUDA path returns immediately.

Plus the Windows-only sixth edit, also already on `moat-port`:

6. **`setup.py`: Windows/HIP `ext.cpp` ABI.** On Windows with ROCm, MSVC `cl.exe` compiles
   `.cpp` but cannot link `c10::ValueError(SourceLocation, string)` out of the Clang-built
   `c10.dll` (the inherited constructor is absent from the MSVC import lib). `ext.cpp` is
   copied to `ext_winhip.cu` so `BuildExtension` routes it through hipcc/amdclang++, which
   shares the ABI. Guarded `os.name == 'nt' and torch.version.hip`.

### Replicate, do not factor out

Sixty lines of identical `setup.py` support code copied into 14 files is unattractive, and
factoring it into a shared top-level `rocm_build_support.py` would work (setup.py already
depends on `..` for GLM). **Decision: replicate verbatim anyway.** Three reasons. The three
existing variants are GPU-proven exactly as written and refactoring them churns proven code
for no functional gain. Upstream's own structure is 14 independent copies of everything, and
the standing rule is to preserve upstream structure. Most importantly, verbatim replication
makes completeness mechanically checkable -- see the equivalence-class invariant below --
whereas a shared helper would make "did variant N actually get the treatment?" a question
requiring judgement.

### Completeness invariant (the thing that stops a partial port passing)

Before the port, the 14 variants fall into exact md5 equivalence classes per file. Those
classes must be **unchanged after the port**:

| file | classes upstream |
|---|---|
| `cuda_rasterizer/auxiliary.h` | 1 (all 14 identical) |
| `cuda_rasterizer/forward.cu` | 2 (`base`+`tile1` = 6 dirs; `wet`+`wet-abs` = 8 dirs) |
| `cuda_rasterizer/backward.cu` | 2 (12 dirs; `wet-abs`* = 2 dirs) |
| `cuda_rasterizer/rasterizer_impl.cu` | 3 (6 / 6 / 2) |
| `rasterize_points.cu` | 3 (6 / 6 / 2) |
| `cuda_rasterizer/forward.h`, `backward.h`, `rasterizer.h` | 2 / 2 / 3 |
| `cuda_rasterizer/rasterizer_impl.h`, `ext.cpp`, `CMakeLists.txt` | 1 each |

If any post-port md5 table has more classes than the corresponding row, a variant was
missed or edited differently. `setup.py` is 14 distinct files before and after; there the
check is that each variant's port diff against its own upstream file is identical modulo the
package name. The reviewer runs both checks; they are cheap and they catch precisely the
"didn't go far enough" failure this project is most exposed to.

## CUDA surface inventory

Regex census (`utils/surface.py`): 70 CUDA-bearing files, 98 `__global__` sites, 770
device-code markers, `cub` in 56 files. Per variant that is 5 CUDA files and 7 kernels:

| kernel | file | mapping |
|---|---|---|
| `FORWARD::preprocessCUDA<C>` | `forward.cu` | direct, no change |
| `FORWARD::renderCUDA<C>` (`__launch_bounds__(BLOCK_X*BLOCK_Y)`) | `forward.cu` | direct |
| `BACKWARD::preprocessCUDA<C>` | `backward.cu` | direct |
| `BACKWARD::renderCUDA<C>` (`__launch_bounds__`) | `backward.cu` | direct |
| `checkFrustum` | `rasterizer_impl.cu` | direct |
| `duplicateWithKeys` | `rasterizer_impl.cu` | direct |
| `identifyTileRanges` | `rasterizer_impl.cu` | direct |

Everything else the port has to account for:

- **Warp intrinsics: none.** No `__shfl*`, no `__ballot`, no `__activemask`, no `warpSize`,
  no `cg::reduce`, no `tiled_partition`, no `coalesced_threads`. The single hardcoded 32 is
  `#define NUM_WARPS (BLOCK_SIZE/32)` in `auxiliary.h`, which is **dead** -- it is defined in
  all 14 variants and referenced nowhere. Leave it alone (touching it would be an unrelated
  refactor of the CUDA path). This is why the port is wavefront-neutral by inspection, and
  why gfx90a and gfx1100/1101/1201 all passed the stage-1 run with no arch-specific edit.
- **Block-level sync:** `__syncthreads_count(done)` and `block.sync()`. Both HIP-supported.
  The `done` early-exit is a uniform `break` on `num_done == BLOCK_SIZE`, evaluated by every
  thread -- no thread reaches a barrier its neighbours skip, so the intra-wave barrier
  divergence fault class does not apply.
- **Atomics:** 36 `atomicAdd` sites (float and float2/float4 component). Native on AMD.
  Non-determinism from reorder is expected and is the standard 3DGS bar (stage 1 measured
  backward agreement to ~3e-7, forward bit-identical).
- **CUB:** `cub::DeviceScan::InclusiveSum` and `cub::DeviceRadixSort::SortPairs` (uint64 keys,
  uint32 values), plus the nullptr-probe sizing calls. hipify -> hipCUB/rocPRIM. Proven.
- **Cooperative groups:** `this_grid().thread_rank()`, `this_thread_block()`,
  `group_index()`, `thread_index()`, `thread_rank()`, `sync()`. All HIP-supported without a
  cooperative launch.
- **Runtime API:** `cudaMemcpy` (D2H, one int), `cudaMemset` (tile ranges), `cudaDeviceSynchronize`,
  `cudaGetErrorString` -- all inside the `CHECK_CUDA` macro path. Straight hipify renames.
- **Memory:** all device memory comes from torch tensors resized through `resizeFunctional`
  and sub-allocated by `rasterizer_impl.h::obtain` with 128-byte alignment. No `cudaMalloc`,
  no pinned or managed memory, no streams, no events, no graphs. `imgState.ranges` is
  explicitly `cudaMemset` to zero, so the "ROCm does not hand back zeroed pages" fault class
  is already covered by upstream.
- **Textures / surfaces:** none. So no 256-byte pitch issue, no layered-array issue, no
  rule-of-five resource-handle issue.
- **Math intrinsics:** `rsqrtf` (x2), `exp`, `min`/`max`, `fabs`. No `__fsqrt_rn`, no
  `__fdividef`, no exact float-equality branch fed by a fast-math divide. The one `p.z == 0.0`
  guard precedes the division it protects.
- **Other CUDA libraries:** no cuBLAS, cuFFT, cuRAND, cuSPARSE, cuSOLVER, cuDNN, Thrust,
  CUTLASS, NCCL, driver API or NVRTC. Nothing needs a library substitution.
- **Non-C++ build paths:** none. No Go/cgo, no runtime PTX, no codegen.
- **Deprecated torch C++ API:** `x.type().is_cuda()` and `.data<float>()` are used throughout
  `rasterize_points.cu`. Checked against this fleet's torch 2.14 (`ATen/core/TensorBody.h`
  lines 230 and 247): both still present. **No change needed** -- recorded so the porter does
  not spend time on it.

## Risk list

1. **High-channel register and LDS pressure (`ch18`, `ch26`, `wet-ch18`, `wet-ch26`) --
   the top risk, and exactly the untested tail.** `renderCUDA<C>` keeps per-thread arrays
   `float C[26]`, and backward adds `accum_rec[26]`, `dL_dpixel[26]`, `last_color[26]` -- ~104
   floats per thread, dynamically indexed, so scratch-bound. Backward shared memory is
   `C*BLOCK_SIZE*4 + 16.4KB`: 43,008 bytes at C=26, under the 64KB per-workgroup LDS limit on
   both gfx9 and gfx11 but leaving one workgroup per CU. Expect long compile times and
   possible spill warnings. A hard failure would show as a compiler "local memory limit
   exceeded" or a launch failure, not wrong numbers. If it happens, do **not** retune the
   kernel; record it per-platform and raise it, because a tuning change would diverge from
   the CUDA path.
2. **`tile1` and `__launch_bounds__(1)`.** `diff-surfel-rasterization-tile1` sets
   `BLOCK_X = BLOCK_Y = 1`, so `renderCUDA` carries `__launch_bounds__(1)`, which clang maps
   to `amdgpu_flat_work_group_size(1,1)` -- a workgroup of one active lane in a 64- or 32-wide
   wavefront. Legal, but it is an unusual corner for the AMD backend and it makes rendering
   pathologically slow (one pixel per block). Build it, test it on a small image (32x24), and
   if the compiler rejects the attribute record the exact diagnostic rather than editing the
   attribute out.
3. **`wet-abs` gradient shape contract.** The two abs variants change `dL_dmeans2D` from
   `{P,3}` to `{P,4}` (columns 2-3 carry the homodirectional absolute gradient) and add an
   internal `{P,2}` buffer. Autograd matches gradient shape to input shape, so **`means2D`
   must be passed as `(P,4)` for `wet-abs` and `wet-abs-ch05`, and `(P,3)` for every other
   variant.** A `(P,3)` input trips autograd's shape check, which looks like a port bug and
   is not one. (Companion to the stage-1 finding that 2DGS `scales` are `(P,2)`, not `(P,3)`.)
4. **SH path is 3-channel only.** For `NUM_CHANNELS != 3`, `geomState.rgb` is sized `P*3`
   while `renderCUDA<C>` would index `features[id*C + ch]`. Upstream already guards this with
   an explicit `throw` in `rasterizer_impl.cu` when `colors_precomp == nullptr`, so it is not
   a live out-of-bounds read -- but the test harness must drive every `C != 3` variant through
   `colors_precomp`, and only `diff-surfel-rasterization`, `-tile1`, `-wet` and `-wet-abs` may
   use the SH path.
5. **FP contraction differences.** clang(HIP) defaults to `-ffp-contract=fast` and forms FMAs
   across expressions where nvcc contracts expression-only. This makes exact cross-variant
   image equality the wrong assertion; the harness compares with a tolerance (below). It also
   means CUDA-versus-HIP image comparison, if anyone attempts it, must be tolerant.
6. **Wavefront size: assessed and dismissed.** No warp-width-dependent code exists (see
   inventory). Recorded explicitly so the wave32 platform does not go looking for a delta plan
   it does not need. If a wave32 platform *does* fail, the cause is elsewhere and the failure
   is interesting.
7. **Stale hipified mirrors.** torch writes `.hip` mirrors next to the `.cu` sources; editing
   a `.cu` without clearing `build/` and the mirrors rebuilds stale code and validates the
   wrong thing. Every build command below starts with the clean step.
8. **14 builds is a real cost.** Each variant is an independent extension with 7 templated
   kernels. Budget accordingly per platform, and build with `MAX_JOBS` set; the high-channel
   variants dominate.
9. **nvcc no-regression gate is argued, not measured.** No CUDA toolkit or NVIDIA GPU is
   available in this fleet, so the "CUDA path still compiles" gate rests on the structure of
   the diff: every source edit is inside `#if !defined(USE_ROCM)` / `#if defined(USE_ROCM)`
   except the chevron respacing, which is whitespace inside a token sequence nvcc already
   accepts; both `setup.py` additions return early unless `torch.version.hip` is truthy. If a
   CUDA host ever becomes available, compile one variant of each family and record it.

## File-by-file change list

### Already on `moat-port` (inherit unchanged, do not re-edit)

- `README.md` -- AMD GPU support section
- `diff-surfel-rasterization-wet/`, `-wet-ch05/`, `-wet-ch07/` -- `setup.py` +
  `cuda_rasterizer/{auxiliary.h,backward.cu,backward.h,forward.cu,forward.h,rasterizer_impl.cu}`

### New work: the 11 remaining variants

For each of

`diff-surfel-rasterization`, `-ch05`, `-ch11`, `-ch18`, `-ch26`, `-tile1`, `-wet-abs`,
`-wet-abs-ch05`, `-wet-ch11`, `-wet-ch18`, `-wet-ch26`

apply the same seven-file treatment:

| file | change |
|---|---|
| `cuda_rasterizer/auxiliary.h` | `#if defined(USE_ROCM)` -> `#define __trap __builtin_trap` |
| `cuda_rasterizer/backward.h` | guard `#include "device_launch_parameters.h"` |
| `cuda_rasterizer/forward.h` | guard `#include "device_launch_parameters.h"` |
| `cuda_rasterizer/backward.cu` | guard `<cooperative_groups/reduce.h>`; respace 2 launches |
| `cuda_rasterizer/forward.cu` | guard `<cooperative_groups/reduce.h>`; respace 2 launches |
| `cuda_rasterizer/rasterizer_impl.cu` | guard `device_launch_parameters.h` and `<cooperative_groups/reduce.h>`; respace 3 launches |
| `setup.py` | GLM hipify-ignore monkeypatch; Windows `ext_winhip.cu` routing; `_ext_sources`/`GLM_DIR` |

Untouched everywhere: `config.h`, `rasterizer.h`, `rasterizer_impl.h`, `rasterize_points.h`,
`rasterize_points.cu`, `ext.cpp`, `CMakeLists.txt`, `LICENSE.md`, the per-variant READMEs,
and `diff_surfel_rasterization*/__init__.py`.

### New work: the test harness

- `tests/test_variants.py` -- new, the GPU test suite described below
- `tests/README.md` -- how to build all 14 and run it

A new top-level `tests/` directory touches no variant and no upstream file. It exists because
this repository currently has no test of any kind, so without it there is nothing a validator
can run and no way to distinguish a working port of 14 variants from a working port of 3. The
"no upstream PR destination" finding above is what makes adding it the right call rather than
a footprint violation.

### Lessons to promote to the `cuda-to-rocm` skill (on this branch)

None of the five fault classes this port depends on is currently in the skill, and all five
recur across the whole 3DGS/2DGS/gsplat rasterizer family, which MOAT keeps meeting. The
porter should add them where a reader with that problem would look:

1. clang/HIP requires `<<<` to lex as one token; nvcc's EDG front end accepts `<< <`.
   (Headers/includes/build, or a new "nvcc accepts, clang does not" entry.)
2. `device_launch_parameters.h` does not exist in HIP.
3. `<cooperative_groups/reduce.h>` does not exist in HIP's cooperative groups.
4. `__trap()` is not declared by HIP's device runtime; `__builtin_trap` is the equivalent.
5. torch hipify content-rewrites a bundled header-only library it can see (GLM), dropping
   `.inl` files and mangling the library's own compiler detection -- add the directory to
   hipify's `ignores` and remove it from `header_include_dirs`. Belongs in
   `references/strategy-b-torch.md`.

## Build commands

Per variant, from the variant directory. gfx942 shown; substitute `PYTORCH_ROCM_ARCH` per
platform (`gfx90a`, `gfx1100`, `gfx1101`, `gfx1201`).

```
export PYTORCH_ROCM_ARCH=gfx942
export MAX_JOBS=16
cd projects/diff-surfel-rasterizations/src/<variant>
rm -rf build *.egg-info *.hip cuda_rasterizer/*.hip   # force a full re-hipify
pip install -e . --no-build-isolation --no-deps -v
```

`--no-build-isolation` is load-bearing: without it pip builds in a fresh environment and
pulls a CUDA torch to compile against. `--no-deps` keeps pip from replacing the ROCm torch.
Editable installs work here (each package directory has an `__init__.py`).

All 14, wrapped for telemetry:

```
bash utils/timeit.sh diff-surfel-rasterizations compile -- bash -c '
  export PYTORCH_ROCM_ARCH=gfx942 MAX_JOBS=16
  cd projects/diff-surfel-rasterizations/src
  for v in diff-surfel-rasterization diff-surfel-rasterization-ch05 \
           diff-surfel-rasterization-ch11 diff-surfel-rasterization-ch18 \
           diff-surfel-rasterization-ch26 diff-surfel-rasterization-tile1 \
           diff-surfel-rasterization-wet diff-surfel-rasterization-wet-abs \
           diff-surfel-rasterization-wet-abs-ch05 diff-surfel-rasterization-wet-ch05 \
           diff-surfel-rasterization-wet-ch07 diff-surfel-rasterization-wet-ch11 \
           diff-surfel-rasterization-wet-ch18 diff-surfel-rasterization-wet-ch26; do
    ( cd $v && rm -rf build *.egg-info *.hip cuda_rasterizer/*.hip \
      && pip install -e . --no-build-isolation --no-deps ) || exit 1
  done'
```

Windows: same, from a ROCm-torch environment; the `ext_winhip.cu` routing engages
automatically via `os.name == 'nt' and torch.version.hip`.

The CMake path is not built. See "Build classification".

## Test plan

**Non-GPU regression set: there is none.** Upstream ships no tests, no CI, no CPU code path
and no non-GPU entry point (`find` over the tree: 28 Python files, all of them `setup.py` or
a package `__init__.py`; no workflow files). Nothing can regress that is not GPU behaviour.
The one no-regression obligation is that the CUDA build stay a pure passthrough, argued
structurally in risk 9.

**GPU tests: `tests/test_variants.py`, new, parameterised over all 14 variants.** The design
problem is that there is no reference implementation and no NVIDIA GPU to capture one from.
The solution is to use the 14 variants as each other's oracle: they are three code bodies
compiled at ten different channel counts, and the forward alpha-compositing arithmetic is
identical across all of them. That turns coverage into correctness.

Per variant:

1. **Import and symbol check.** The package imports and `_C` exports `rasterize_gaussians`,
   `rasterize_gaussians_backward`, `mark_visible`.
2. **`mark_visible` against a CPU reference.** Reimplement the frustum test in torch on CPU
   and assert exact boolean agreement. Cheap, exact, no tolerance argument.
3. **Forward sanity.** A fixed-seed scene (~4000 surfels, 200x150, `scales` shaped `(P,2)`,
   `means2D` shaped `(P,4)` for the abs variants and `(P,3)` otherwise, `colors_precomp` with
   `NUM_CHANNELS` columns) renders a finite, non-trivial image: no NaN/Inf, nonzero fraction
   above a floor, min/max/mean in range.
4. **Determinism.** Two identical forward calls are bit-identical.
5. **Backward finiteness.** Gradients to `means3D`, `means2D`, `opacities`, `colors_precomp`,
   `scales`, `rotations` are all finite and not all zero.
6. **Opacity finite difference.** Perturb opacity, compare the analytic gradient to the
   central difference: sign agreement and slope near 1.0. Stage 1 established this as the
   decisive geometric-blend gradient check for this kernel family.

Cross-variant (the coverage oracle, and the reason 14 variants can be proven without a
reference):

7. **First-three-channel image equality across all 14.** Render one scene through every
   variant, feeding each a `colors_precomp` whose first three columns are the same values.
   Assert every variant's output image agrees with `diff-surfel-rasterization`'s on channels
   0-2 within `rtol=1e-4, atol=1e-5` (tolerance rather than exactness because of FP
   contraction, risk 5). This single assertion catches a variant that compiled but computes
   garbage -- which is precisely the failure a per-variant smoke test would miss.
8. **`tile1` versus the base variant.** Same scene at 32x24. `tile1` differs only in tile
   decomposition (1x1 instead of 16x16); per-pixel compositing order is unchanged, so its
   image must match `diff-surfel-rasterization` within the same tolerance. Small image
   because a 1x1 tile makes it pathologically slow.
9. **`wet-abs` gradient structure.** `grad_means2D[:, 2:4]` (the homodirectional absolute
   gradient) is non-negative and not identically zero, and `grad_means2D[:, 0:2]` matches the
   corresponding `wet` variant's columns within tolerance.

Run:

```
bash utils/timeit.sh diff-surfel-rasterizations test -- \
  python -m pytest projects/diff-surfel-rasterizations/src/tests/test_variants.py -v
```

Evidence to record per platform in `notes.md`: ROCm and torch versions, `PYTORCH_ROCM_ARCH`,
the pass/fail count, the forward statistics and finite-difference slopes for at least the
three previously-untested code bodies (`base`, `tile1`, `wet-abs`), and any spill or
occupancy diagnostic from the `ch18`/`ch26` builds. Confirm dispatch on the AMD device at
least once per platform with `AMD_LOG_LEVEL=3`.

**Gates.** `wave64` from a gfx9 platform (gfx942 or gfx90a), `wave32` from gfx1100/1101/1201,
`windows` from a Windows platform. All three ran the 3-variant stage-1 code successfully via
EnvGS, so no gate is expected to need a waiver and none is requested. That prior evidence is
**not** carried forward to this record: it was taken at a different commit, against 3 of 14
variants, through a different project's harness. Each platform validates the 14 variants
against this project's own `head_sha`.

## Open questions

1. **Does `__launch_bounds__(1)` survive the AMD backend?** `tile1` is the only variant that
   asks. If it does not, the choice is a per-variant guard versus recording the variant as
   blocked; the plan's preference is to record and raise rather than to diverge the kernel.
2. **Do `ch26` and `wet-ch26` build within a sane time and register budget on every arch?**
   Register pressure is arch-dependent, so this can plausibly pass on gfx942 and fail on a
   wave32 arch even though nothing wavefront-dependent is in the source. That would be the
   first genuinely arch-specific finding in this project and would justify a delta plan.
3. **Should the fork's `README.md` grow an install matrix?** It currently says "each variant
   is a PyTorch CUDA extension ... `pip install -e .` just works", written when three
   variants were ported. Once all 14 are covered the statement is true without qualification;
   whether to enumerate them is the porter's call.
4. **`utils/surface.py` misses `CUDAExtension` declarations spread over multiple lines** --
   its regex requires the name as the first argument on the same line, so the generated floor
   for this project found zero Python extensions and one collapsed CMake library. The 14
   extension components were added to `surface.json` by hand. Every torch-extension project
   has the same hole in its generated floor. Fixing it is control-plane work and does not
   belong on this branch, but it should be raised.
