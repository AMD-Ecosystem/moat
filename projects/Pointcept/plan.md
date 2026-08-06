# Pointcept -- ROCm/HIP Porting Plan (lead: linux-gfx90a)

## Project
- Name: Pointcept
- Upstream: https://github.com/Pointcept/Pointcept
- Default branch: main
- Clone (read-only): projects/Pointcept/src/
- Domain: point-cloud perception framework (PTv1/2/3, SpUNet, OACNN, PointGroup). The
  CUDA surface is a set of in-tree PyTorch C++/CUDA extension ops under `libs/`.

## Existing AMD support
- Decision: PROCEED with a from-scratch HIP port (Strategy B).
- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/` -> no hits. No "AMD support" / "notable
  forks" section; the repo does not link a platform fork, so an upstream PR is an
  appropriate delivery vehicle (no karpathy-style merge policy).
- Web search ("Pointcept ROCm/AMD/HIP") -> no AMD port, no ROCm-DS-style separately-named project.
- `gh pr list --repo Pointcept/Pointcept --state all --search "ROCm OR HIP OR AMD"` -> empty.
- `gh api repos/Pointcept/Pointcept/forks` -> no fork under ROCm/AMD/GPUOpen orgs and none with
  rocm/hip/amd in the name (the lone loose grep hit, SamDeGeyter/Pointcept, is unrelated).
- Conclusion: no existing AMD effort, authoritative or community. Clean from-scratch port.

## Build classification: torch-extension (Strategy B)
Evidence -- every GPU lib builds via `torch.utils.cpp_extension`:
- `libs/pointops/setup.py`, `libs/pointops2/setup.py`, `libs/pointgroup_ops/setup.py`,
  `libs/pointrope/setup.py` all use `CUDAExtension` + `BuildExtension`.
- `libs/pointseg/setup.py` uses `CppExtension` (CPU-only, no `.cu`; no port needed).
No CMake, no `find_package(Torch)`. This is the canonical Strategy B shape: torch hipifies the
`.cu`/`.cpp` sources at build time on a ROCm torch.

## Port strategy: B (torch-hipify), correctness-first mechanical port
Rationale: these are simple data-parallel point-cloud ops (knn, ball-query, FPS, grouping,
interpolation, aggregation, subtraction, local attention + relative-position-encoding,
PointROPE rotary embedding). None are CUTLASS/CuTe/Hopper-tuned, so no AMD-native (CK/rocWMMA/MFMA)
rewrite is warranted; a mechanical hipify port is the right and sufficient first step.
- Keep sources in CUDA spelling; torch's hipify translates them at build time. Do NOT add a
  `cuda_to_hip.h` compat header and do NOT hand-rename symbols (that is Strategy A).
- Fix only what hipify cannot, guarded by `USE_ROCM` / `torch.version.hip`.

## CUDA surface inventory
GPU libs and their `.cu` counts: pointops (9), pointops2 (10), pointgroup_ops (1), pointrope (1).
- Kernels: knnquery, ballquery, FPS (`furthestsampling`/`farthest_point_sampling`), grouping,
  interpolation, aggregation, subtraction, attention step1/step2 (+ v2), relative-pos-encoding
  v2/v3, bfs_cluster (PointGroup), pointrope rotary kernel.
- Host glue: standard ATen extension (`#include <ATen/cuda/CUDAContext.h>`,
  `at::cuda::getCurrentCUDAStream()`); `torch::PackedTensorAccessor32` in pointrope.
- Warp intrinsics: NONE. No `__shfl*`, `__ballot`, `__activemask`, `__any/__all`, `__popc`,
  `__syncwarp`, `warpSize`, cooperative groups.
- Textures/surfaces/cudaArray: NONE.
- Libraries: NO cuBLAS/cuFFT/cuRAND/cuSPARSE/cuDNN/Thrust/CUB. Pure hand-written kernels.
- half/bf16/`__half`/`__nv_*`: NONE. `__CUDA_ARCH__`: NONE. THC legacy headers: NONE.
- Atomics: `atomicAdd` (float) in subtraction and rpe_v2 backward -- 1:1 on HIP, no change.
- Streams/events/pinned/managed: only `cudaStream_t` via `getCurrentCUDAStream()`; no events,
  no pinned/managed memory.
- ROCm/HIP -> CUDA mapping: every symbol is hipify-1:1 (`cudaStream_t`->`hipStream_t`,
  `cudaGetLastError`->`hipGetLastError`, `at::cuda::*` stays under hipify). No risk symbols.

## Risk list
1. **nvcc-only flags in pointrope/setup.py (BUILD BLOCKER, must fix).**
   `libs/pointrope/setup.py:33` passes `nvcc=["-O3", "--ptxas-options=-v", "--use_fast_math"]
   + cuda.get_gencode_flags()`. On a ROCm torch hipcc rejects `--ptxas-options=-v`,
   `--use_fast_math` (nvcc spelling), and the `-gencode arch=compute_*` flags from
   `cuda.get_gencode_flags()`. Fix: branch on `torch.version.hip` in setup.py -- on ROCm pass
   `["-O3", "-ffast-math"]` (or just `-O3`) and drop the gencode list (torch sets
   `PYTORCH_ROCM_ARCH`/`--offload-arch`). The other four setups pass only `-O2`/`-g`/`-O3`,
   which hipcc accepts unchanged.
2. **Warp size (gfx90a wave64 vs gfx1100/gfx1151 wave32).** No warp intrinsics anywhere, so
   the usual lane-mask/shfl hazards do not apply. The one place worth auditing is the FPS
   shared-memory reduction (`sampling_cuda_kernel.cu`): it is a power-of-two `__update` ladder
   in which EVERY step (including the `tid<32`, `tid<16`, ... tail) is followed by an explicit
   `__syncthreads()` -- there is NO implicit warp-synchronous tail and no `volatile`. So it is
   correct on both wave64 and wave32 with no change. The `<32>` kernel template instantiations
   in attention/rpe are BLOCK-TILE sizes (n_threads), not warp-width assumptions; arch-agnostic.
   Net: low warp-size risk, but confirm via the cross-arch consistency gate on followers.
3. **torch_scatter dependency in the in-tree tests.** The `libs/pointops2/functions/test_*.py`
   scripts import `torch_scatter`. On ROCm install it from the pyg-rocm-build wheels (do NOT let
   an unfiltered pip pull the CUDA `torch-scatter` wheel over the ROCm torch). Environment, not a
   source deliverable.
4. **spconv dependency is NOT yet ported (deps-first ordering).** status.json lists
   `depends_on: ["spconv"]` and spconv's lead state is `unclaimed`. spconv (traveller59) is the
   SparseUNet/OACNN/PointGroup-model backbone (a separate pccm-codegen build, `ext_type:
   pccm-codegen`, its own MOAT project). No ready ROCm spconv wheel was found. IMPORTANT: the
   `libs/` CUDA ops ported here do NOT require spconv to build, install, or run their tests --
   pointops/pointops2/pointgroup_ops/pointrope are self-contained PyTorch extensions. spconv is
   only needed to run the SPARSE-CONV model configs end-to-end. So this port's own validation
   (build the four extensions + run the in-tree op tests on gfx90a) is unblocked. Scope the PR
   claim to the `libs/` ops; end-to-end sparse-conv model validation waits on the spconv port.
5. **Fresh-allocation-not-zero / OOB-neighbor classes.** No texture pitch, no stencil edge reads,
   no partial-write-relies-on-zeroed-buffer pattern spotted; FPS writes `tmp` in full over its
   range. Low risk, but the validator should watch the numeric-diff tests for any ULP drift
   (hipcc default `-ffp-contract=fast`; `--use_fast_math`/`-ffast-math` in pointrope widens this).
   If a pointrope numeric test drifts, pin `-ffp-contract=on` and/or drop fast-math on the HIP path.

## File-by-file change list
- `libs/pointrope/setup.py` -- add `torch.version.hip` branch: on ROCm replace the nvcc-only
  `extra_compile_args["nvcc"]` (`--ptxas-options=-v`, `--use_fast_math`, gencode list) with a
  hipcc-safe list (`-O3`, optionally `-ffast-math`/`-ffp-contract=on`). The ONLY required build edit.
- `libs/pointops/setup.py`, `libs/pointops2/setup.py`, `libs/pointgroup_ops/setup.py` -- expected
  NO change (only `-O2`/`-g`, hipcc-safe). Confirm at build; touch only if a flag rejects.
- `.cu`/`.cpp`/`.cuh`/`.h` kernel sources -- expected NO source change (no warp intrinsics, no
  textures, no libraries, all symbols hipify-1:1). Add a `USE_ROCM`-guarded fix ONLY if a fault
  class surfaces at validation (e.g. an fp-contract numeric drift in pointrope).
- Docs (PR-prep phase): add a ROCm/AMD build note in the README "Installation" section
  (line ~157-219) in the project's house style, parallel to the CUDA `setup.py install` block.

## Build commands (gfx90a)
Against a ROCm PyTorch (Linux ROCm 7.2.1, torch main). For each GPU lib:
```
cd libs/<lib>
PYTORCH_ROCM_ARCH=gfx90a python setup.py install      # or: pip install . --no-build-isolation
```
`--no-build-isolation` is load-bearing if using pip (else pip pulls a CUDA torch to build against).
Pass `PYTORCH_ROCM_ARCH="gfx90a;gfx1100"` to emit a fat binary and prove both code objects
(`llvm-objdump --offloading <ext>.so` shows gfx90a AND gfx1100) for the warp-size guard check.
Build order: pointops, pointops2, pointgroup_ops, pointrope (pointseg is CppExtension/CPU, optional).

## Test plan
Real GPU tests (the in-tree op tests; this is the validator's GPU gate):
- `libs/pointops2/functions/test_attention_op_step1.py`, `test_attention_op_step1_v2.py`,
  `test_attention_op_step2.py`, `test_relative_pos_encoding_op_step1{,_v2,_v3}.py`,
  `test_relative_pos_encoding_op_step2{,_v2}.py` -- each builds inputs on `.cuda()` (ROCm torch
  maps `cuda`->HIP), runs the custom op forward+backward, and diffs against a PyTorch/torch_scatter
  reference. Run each with `python test_<...>.py` on gfx90a; a pass is reference-agreement
  (watch for ULP drift -> risk #5). These exercise pointops2 (attention + rpe + sort/knn paths).
- pointops / pointgroup_ops / pointrope: no shipped unit test. Validate via a small written
  driver per op (knn, ball-query, FPS, grouping, interpolation, aggregation, subtraction,
  pointrope) numerically against a CPU/PyTorch reference, tol ~1e-5 (1e-4 if fast-math kept).
  Register this driver in notes.md so followers reuse it.
- Cross-arch consistency (followers gfx1100/gfx1151): diff deterministic op outputs against the
  gfx90a result for the same seeded input (the warp-size correctness gate per PORTING_GUIDE).
Non-GPU regression set: the framework's pure-Python paths and pointseg (CppExtension) must still
import and build; do not regress them. No repo-wide pytest suite exists.
End-to-end model configs (SpUNet/OACNN/PointGroup via spconv) are OUT OF SCOPE for this port's
validation until the spconv ROCm port lands (risk #4).

## Open questions
- Does the host have a ready ROCm `torch_scatter` (pyg-rocm-build) wheel for the pointops2 tests,
  or must it be built `--no-build-isolation`? (Validator resolves at env setup.)
- Keep `--use_fast_math` semantics on ROCm (`-ffast-math`) for parity, or drop to `-O3` +
  `-ffp-contract=on` for tighter numeric agreement? Decide from the pointrope numeric-diff result.
- spconv ROCm port is a separate MOAT project; this plan deliberately scopes Pointcept's PR to the
  in-tree `libs/` ops so it is not blocked on spconv.
