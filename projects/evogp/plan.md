# evogp -- ROCm/HIP port plan (lead: linux-gfx90a)

## Project
- Name: evogp (EvoGP -- GPU-accelerated tree-based Genetic Programming framework, built on PyTorch with custom CUDA kernels)
- Upstream: https://github.com/EMI-Group/evogp
- Default branch: main (clone HEAD ee11f1e)
- Lead platform: linux-gfx90a (MI250X, CDNA2, wave64), ROCm 7.2.x

## Existing AMD support
None. README documents only the NVIDIA CUDA Toolkit + nvcc install path; no ROCm/HIP/AMD/OpenCL/SYCL/Vulkan mention anywhere in the repo. No HIP files, no AMD branch, single linear git history. Decision: PORT (fresh CUDA->HIP). A ROCm/HIP port adds clear value and there is nothing to finish or improve.

## Build classification
torch-extension (Strategy B). Evidence: setup.py:3 imports `from torch.utils.cpp_extension import CUDAExtension, BuildExtension`; setup.py:37-60 builds a single `CUDAExtension(name="evogp.evogp_cuda", sources=[torch_wrapper.cu, generate.cu, mutation.cu, forward.cu])`; pyproject.toml requires `torch` in build-system.requires. The Python package (src/evogp/{tree,algorithm,problem,pipeline}) is pure-Python orchestration over the one compiled op library, registered via `torch.ops.evogp_cuda.*` (TORCH_LIBRARY in torch_wrapper.cu).
ext_type set to `torch-extension` in upstream.json and status.json.

## Port strategy: B (torch hipify), minimal source edits
Building the CUDAExtension against a ROCm torch makes `torch.utils.cpp_extension` auto-run `torch.utils.hipify` on the `.cu`/`.h` sources and link the HIP runtime (amdhip64, c10_hip, torch_hip). Do NOT add a compat header and do NOT hand-rename symbols; keep CUDA spelling and let hipify translate. Fix only what hipify cannot, guarded by `USE_ROCM`. The CUDA surface here is small and entirely standard, so the expected source delta is near-zero.

Performance note: the kernels are bespoke tree/stack interpreters and a shared-memory fitness reduction -- NO CUTLASS/CuTe/wgmma, no GEMM/attention. A mechanical hipify port is the correct and complete approach; no AMD-native rewrite is warranted.

### setup.py nvcc-flag handling (the one likely build edit)
setup.py:48-57 passes nvcc-only flags that hipcc will reject or that are meaningless on HIP:
- `--ptxas-options=-v`, `-Xptxas=-O3`, `-lineinfo`, `-maxrregcount=32` -- nvcc/ptx-specific.
- `--expt-relaxed-constexpr` -- nvcc-only spelling.
- `-lcudart` -- a link flag in a compile list; harmless-ish but CUDA-named.
- `-use_fast_math` -- nvcc spelling; hipcc uses `-ffast-math`.
torch's hipify rewrites source, not setup.py compile args, so these flags are passed verbatim to hipcc and several will fail the build. Fix: gate the nvcc arg list on backend. Detect ROCm at setup time (`torch.version.hip is not None`) and supply a HIP-appropriate `nvcc` arg list (CUDAExtension keeps the key name `nvcc` for the device compiler on both backends): drop the ptxas/lineinfo/maxrregcount/relaxed-constexpr/-lcudart entries, keep `-O3`, and translate `-use_fast_math` -> `-ffast-math` if fast math is desired (see fast-math risk below -- prefer dropping it for first correctness bringup). This is the primary expected edit.

## CUDA surface inventory
Files: src/evogp/cuda/{defs.h, kernel.h, torch_wrapper.cu, generate.cu, mutation.cu, forward.cu}.
- Kernels (8 `__global__`): treeGPGenerate (generate.cu:17); treeGPEvalKernel (forward.cu:305); treeGPRegressionFitnessKernel (forward.cu:403); averageFitnessValueKernel (forward.cu:474); constant_tree_treeGPRegressionFitnessKernel (forward.cu:592); constant_vars_treeGPRegressionFitnessKernel (forward.cu:739); treeGPMutationKernel (mutation.cu:118); treeGPCrossoverKernel (mutation.cu:224). All 1-D grid/block index math (`blockIdx.x*blockDim.x+threadIdx.x`), plus one 2-D grid (nGP, nTB) in the fitness kernel.
- Warp intrinsics: NONE. No `__shfl*`, `__ballot`, `__any/__all`, `__activemask`, `__popc`, no `warpSize`, no cooperative groups, no `__syncwarp`. (This removes the entire warp-mask fault class.)
- Shared memory + reduction: `__shared__ float sharedFitness[SR_BLOCK_SIZE]` (SR_BLOCK_SIZE=1024) in the two regression-fitness kernels; tree-reduction `for (size = SR_BLOCK_SIZE/2; size>0; size>>=1){ if(threadId<size) sdata[t]+=sdata[t+size]; __syncthreads(); }`. Crucially the warp-synchronous tail is COMMENTED OUT (`// if (size >= 32)` at forward.cu:464 and :639) -- every step is `__syncthreads()`-guarded down to size 1. This is exactly the wave-safe pattern the guide prescribes; the classic unsynced wave32 tail is absent.
- Atomics: `atomicAdd(&fitnesses[...], sharedFitness[0])` float, one per block (forward.cu:470, :645). Float atomicAdd is fine on gfx90a. No int atomicMin/atomicMax (the cudaKDTree coarse-grained-memory trap does not apply).
- Constant memory: `__constant__` arrays (forward.cu:10-14, const_node_value/type, const_variables/labels, const_tree_size) populated via `cudaMemcpyToSymbolAsync(..., cudaMemcpyDeviceToDevice)` (forward.cu:670-672, :805-806). hipify maps `__constant__`, `cudaMemcpyToSymbolAsync`, and `cudaMemcpyDeviceToDevice` 1:1.
- Device stack: heavy use of `alloca(MAX_STACK*sizeof(...))` per thread (MAX_STACK=1024) in generate/mutation/forward -- a large per-thread local stack tree interpreter. Compiles on hipcc; watch register/stack pressure and scratch usage on gfx90a (see risks). `-maxrregcount=32` (an nvcc occupancy tuning) is dropped on HIP.
- Thrust: only `thrust::random` (taus88 engine + `uniform_real_distribution`) in generate.cu (kernel.h:8,20; generate.cu:41) -- DEVICE-side RNG, header-only. rocThrust is a drop-in (same `<thrust/random.h>` path under /opt/rocm/include); no source change expected. NO cub/hipCUB, NO curand, NO cuBLAS/cuFFT/cuSPARSE.
- Host runtime: `cudaGetLastError`, `cudaGetErrorString`, `cudaDeviceSynchronize` in torch_wrapper.cu error-check helper; all hipify 1:1. Kernels launch on the default stream (no explicit stream/event objects), so no `getCurrentCUDAStream` plumbing and no rule-of-five resource-handle exposure.
- Textures/surfaces: NONE. (No texture-pitch, layered-array, or linear-filter fault classes apply.)
- torch_wrapper.cu uses TORCH_LIBRARY / torch::Tensor / TORCH_CHECK -- backend-agnostic torch C++; unchanged on ROCm torch.

## Risk list (likelihood for this repo)
1. setup.py nvcc-only flags reach hipcc (HIGH / build-blocking): `--ptxas-options`, `-Xptxas`, `-lineinfo`, `-maxrregcount=32`, `--expt-relaxed-constexpr`, `-lcudart`, `-use_fast_math`. Fix as in Strategy section (backend-gated nvcc arg list). This is the main expected change.
2. `-use_fast_math` / clang(HIP) `-ffp-contract=fast` numerics (MEDIUM): GP fitness is a sum of squared/abs errors over data points; fast-math + cross-statement FMA can drift float results. Not bit-exact-validated upstream (it is a stochastic evolutionary run), so small drift is tolerable, but for a clean first bringup DROP fast-math on HIP (omit `-ffast-math`); optionally pin `-ffp-contract=on`. Re-add only if perf-needed and validated. Note the LOOSE_* ops and DELTA/MAX_VAL clamps in defs.h already guard div-by-zero / overflow, reducing NaN risk.
3. Partial-block `return` before a `__syncthreads()` (LOW, pre-existing on CUDA): constant_tree kernel does `if (dataPointId >= dataPoints) return;` (forward.cu:606) before the reduction's `__syncthreads()`; threads in a tail block can diverge at the barrier. This is latent UB on CUDA too and works in practice because SR_BLOCK_SIZE divides cleanly for typical dataPoints; on wave64 it is still a full-block 1024-thread barrier (16 wavefronts), same shape as CUDA's 32 warps. Watch for hangs/garbage on non-multiple-of-1024 data sizes; if it surfaces, set the early-out fit to 0 and fall through to the barrier (zero-pad) instead of returning. NOT expected to bite the sr_test (num_data=1000 < 1024 -> single partial block, threadId>=1000 return; same on CUDA).
4. `torch.uint32` tensor dtype (LOW): test_bind_success.py:28 and the keys path use `torch.uint32`, a newer torch dtype. Ensure the ROCm torch build is recent enough (it will be on ROCm 7.2.x torch). Pure-Python/torch concern, not a kernel issue.
5. wave64 reduction correctness (LOW): the reduction is fully `__syncthreads()`-guarded (no warp-synchronous tail), so it is correct on any wave width. Re-confirm bitwise/near-bitwise determinism across two fixed-seed runs at validation; float `atomicAdd` order across blocks is non-deterministic on every GPU (CUDA included), so accept ULP-level run-to-run variation, validate by convergence not bit-equality.
6. `alloca` stack/scratch pressure on gfx90a (LOW-MEDIUM): MAX_STACK=1024 -> ~4 KB float stack + ~4 KB infos + tree arrays per thread, dynamic alloca in a 1024-thread block. CDNA scratch sizing differs from NVIDIA local memory; could hit launch failure (too many resources) at large block sizes. If a launch fails, reduce block size on HIP or confirm scratch limits; expected to fit but verify at build/run.

## File-by-file change list
- setup.py: backend-gate the `nvcc` extra_compile_args (HIP list drops ptxas/lineinfo/maxrregcount/relaxed-constexpr/-lcudart, drops `-use_fast_math` for first bringup, keeps `-O3`; CUDA list unchanged). Detect via `torch.version.hip`.
- src/evogp/cuda/*.cu, *.h: expected NO hand edits (hipify handles cuda*->hip*, __constant__, cudaMemcpyToSymbolAsync, thrust). Only touch a source file if the build reveals a hipify gap; any such edit guarded by `#ifdef USE_ROCM`. Candidate-if-needed: forward.cu risk-3 zero-pad, and `-ffp-contract` is a flag (setup.py), not a source edit.
- No CMake (none exists). No compat header (Strategy B forbids it).

## Build commands (gfx90a)
Prereqs: a ROCm PyTorch (ROCm 7.2.x build) active in the env; ROCm toolchain at /opt/rocm. Confirm `python -c "import torch; print(torch.version.hip)"` is non-None before building.
```
# from projects/evogp/src (fork clone), build the extension in-place against ROCm torch
export PYTORCH_ROCM_ARCH=gfx90a
pip install -e . --no-build-isolation
# or: python setup.py build_ext --inplace
```
If hipify leaves a stale mirror after a source edit, clean (`rm -rf build/ src/evogp.egg-info src/evogp/*.so`) and rebuild (known incremental-hipify gotcha).
CPU-only compile smoketest (optional, NOT a validation gate): docker image rocm/dev-ubuntu-24.04:7.2.4-complete.

## Test plan (real GPU gate)
Primary GPU validation gate (end-to-end, exercises all 8 kernels + reduction):
```
python -m evogp.sr_test     # 1000-pop symbolic regression, 100 gens, fixed seed (manual_seed(0))
```
Pass criterion: completes without HIP fault AND the run makes evolutionary progress -- best/mean fitness (loss) decreases across generations and the final best fitness is well below the initial random population's. Capture the per-generation loss; a flat or NaN loss curve is a FAIL even if it does not crash. Because cross-block float atomicAdd order is non-deterministic, do not require bitwise run-to-run equality; require convergence and finite (non-NaN/inf) fitness. Run twice with the fixed seed and confirm both converge to comparable final loss.

Per-kernel correctness probe (faster triage, exercises each op against tiny known inputs):
```
python test/test_bind_success.py    # generate / crossover / evaluate / SR_fitness on small hand-set trees, prints results
```
Use this to localize any kernel that misbehaves before the full sr_test. Note it imports `evogp.evogp_cuda` and uses `torch.uint32` keys.

Broader end-to-end coverage (optional, after sr_test passes): example/regressor.py (SymbolicRegression with a known target func, 300 gens) and example/basic.py (XOR-3 SR) -- both should converge.

Non-GPU regression set: there is no CPU test suite; the Python orchestration (tree/algorithm/selection/mutation/pipeline) only runs meaningfully with the compiled GPU op. brax/jax/mujoco/genesis problem modules are optional extras (not installed) and out of scope. No non-GPU tests to regress.

## Disposition
PORT on linux-gfx90a via Strategy B (torch hipify). Small, standard CUDA surface; no warp intrinsics, no CUTLASS, no texture/surface, no cub/curand. Expected delta is essentially the setup.py nvcc-flag gating plus whatever (if anything) the build surfaces. Advance to `planned`.

## Follower outlook (not re-planned here)
gfx1100/gfx1151 are wave32. Because the only wave-width-sensitive code (the fitness reduction) is fully `__syncthreads()`-guarded with no warp-synchronous tail and there are no warp intrinsics or hardcoded 32 in device code, the same source is expected to validate on RDNA unchanged. Followers should validate first and only delta-port on failure (set PYTORCH_ROCM_ARCH per arch; rocThrust available on Linux ROCm, less certain on the Windows gfx1151 HIP SDK).
