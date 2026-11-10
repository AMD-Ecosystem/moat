# MONAI plan

## Project

- Upstream: https://github.com/Project-MONAI/MONAI (default branch `dev`), Apache-2.0.
- Fork: https://github.com/AMD-Ecosystem/MONAI, maintained by AMD's ROCm-LS team. GitHub default branch `amd-develop`; upstream mirror `dev` (recorded `fork_default_branch`).
- **Port base: `amd-develop` at f9f65c9437e131565f75d7a11266540ba51293fe** ("docs: update with new default branch (#122)", 2026-09-28). `moat-port` is cut from this commit, not from `dev`. amd-develop forked from upstream at eccefc5 (Release 1.6 doc updates, 2026-06-08) and carries 119 commits of its own; upstream `dev` has 162 commits that amd-develop lacks.
- Take-up: Jeff Daily, 2026-11-09. This is a validation project against the AMD team's existing port (see notes.md "Take-up ruling").
- Delivery: to the AMD team's fork (`amd-develop`), not a new upstream PR to Project-MONAI. The delivery vehicle (a PR from `moat-port` into `amd-develop`, or a handoff) is Jeff's call later.

## Existing AMD support

**Authoritative, incomplete coverage.** amd-develop is AMD-official and shipped as `amd-monai` (ROCm-LS, https://rocm.docs.amd.com/projects/monai/en/latest/), supported on MI300X/MI325X/MI355X (gfx942/gfx950), Ubuntu 24.04, ROCm 7.x. Upstream has absorbed part of it (Project-MONAI/MONAI#9153, merged 2026-10-06; #9161 open). Radeon (RDNA, wave32) and Windows are neither claimed nor tested.

Decision: validate and improve; do not re-port. MOAT's differentiator is wave32 (gfx1100/gfx1201) and Windows (gfx1151) evidence, plus whatever fixes those platforms need, on `moat-port`, coordinated with the AMD team. Upstream links no platform forks. There are no NVIDIA-tuned performance kernels (no CUTLASS/CuTe/wgmma), so the port-vs-rewrite question does not arise.

## Build classification

**torch-extension**, Strategy B.

- `setup.py` builds `monai._C` with `torch.utils.cpp_extension.CUDAExtension` from `monai/csrc/**/*.cpp` and `*.cu`, when `BUILD_MONAI=1` (`get_extensions`, setup.py:155-205).
- `monai/_extensions/loader.py` JIT-builds `monai/_extensions/gmm` with `torch.utils.cpp_extension.load` at runtime (`load_module`), whether or not `BUILD_MONAI` is set.
- Everything else is pure Python on PyTorch.

## Port strategy

B: torch's build-time hipify handles every `.cu` file, through both `CUDAExtension` and JIT `load`. amd-develop already relies on this, and AMD's documented build is `BUILD_MONAI=1 FORCE_CUDA=1 python3 setup.py bdist_wheel` (docs/install/installation.rst). Expect few or no source edits for Linux. The work is validation on new platforms, plus fixing what they expose.

amd-develop-specific facts the porter and validator must know:

- `FORCE_CUDA=1` is required on ROCm in amd-develop. `BUILD_CUDA = FORCE_CUDA or (torch.cuda.is_available() and CUDA_HOME is not None)`, and on a ROCm torch `CUDA_HOME` is None. Without `FORCE_CUDA=1`, `BUILD_MONAI=1` silently builds the CPU-only `_C`. Upstream #9153 fixed this with a `ROCM_HOME` fallback that amd-develop does not have (amd-develop predates it). The JIT cache key in `loader.py` likewise still uses `torch.version.cuda` (None on ROCm), which #9153 also fixed. Neither change is required for correctness, so neither is in scope for this plan unless the AMD team wants it. A future sync of amd-develop with upstream brings both.
- `setup.py` `_rocm_install_requires()` makes `install_requires` unconditionally `torch[device-gfx...]` plus a `rocm>=X.Y` pin, detected from `MONAI_ROCM_SERIES`, the `rocm` package, `rocm-sdk version`, or `hipcc --version`, with archs taken from `GPU_TARGETS`/`AMDGPU_TARGETS` (default gfx942,gfx950). Set `AMDGPU_TARGETS` to the host arch, and install the built wheel with `--no-deps`, as the AMD docs do.
- The package name is `amd-monai`; python_requires >= 3.12; numpy < 2.5.
- `tests/runner.py --rerun-failures N` (or `MONAI_TEST_RERUN_FAILURES`) re-runs failed tests in a fresh process. The AMD team added it for an intermittent MIOpen code-object load fault on gfx942 ("device kernel image is invalid" / "named symbol not found"). The validator records the first-pass failures and the after-rerun result separately. A test that passes only on rerun is a flake, recorded as such in notes.md, not a clean pass.

## CUDA surface inventory

13 files (surface.json floor): 24 kernels, 30 launches, 4 warp-intrinsic sites. No cuBLAS/cuFFT/cuRAND/cuSPARSE/cuDNN/Thrust/CUB/NCCL, no textures, no driver API, no NVRTC.

| component | files | notable surface | HIP mapping / note |
|---|---|---|---|
| bilateral (precise + PHL) | `csrc/filtering/bilateral/*_cuda*.cu` | `__constant__` buffers, `cudaMemcpyToSymbol` | hipify adds `HIP_SYMBOL`; fine |
| permutohedral lattice | `csrc/filtering/permutohedral/permutohedral_cuda.cu`, `hash_table.cuh` | `BLOCK_SIZE 32` blocks with shared memory, `atomicCAS`/`atomicExch` hash table, raw `cudaMalloc`/`cudaMemset`/`cudaFree`, `__constant__` pointers set with `cudaMemcpyToSymbol`, `#include <THC/THCAtomics.cuh>` | block-size 32 is not a warp assumption (one `__syncthreads` at :317); THC header -> THH via hipify, verify it ships in the ROCm torch wheel, Windows included |
| trainable (joint) bilateral | `csrc/filtering/trainable_*/*_gpu_{forward,backward}.cu` | 20+ `__constant__` symbols each, 256-entry kernel tables, `BLOCK_SIZE 32`, half dispatch | fine; half on RDNA |
| resample push/pull | `csrc/resample/pushpull_cuda.cu`, `csrc/utils/resample_utils.h` | 1024-thread launch with `C10_LAUNCH_BOUNDS_1(1024)`, `gpuAtomicAdd` incl. half/double | atomics on half/double on RDNA (gfx11/gfx12) are CAS-emulated; correctness OK, check speed only if it times out |
| LLTM | `csrc/lltm/lltm_cuda.cu` | 1024 threads, no launch bounds | HIP default flat-work-group max is 1024; fine |
| GMM (JIT) | `_extensions/gmm/gmm_cuda.cu`, `gmm_cuda_linalg.cuh` | warp reductions: `__shfl_down`/`__shfl_xor` (AMD branch: no `_sync`, default width = warpSize), `warp_index = tid >> 5`, `lane = tid & 31`, `block = warp_count*32`, `s_matrix_component[warp_count]` | **the one wavefront-sensitive site**, see risks |

Python-level CUDA-only dependencies (optional imports, all tests skip when absent):

| dependency | where | ROCm equivalent | plan |
|---|---|---|---|
| cuCIM | wsi_reader, transforms, metrics, auto3dseg | amd-hipcim (Linux), already in amd-develop's extras | Linux: install and exercise; Windows: none exists (question 2) |
| CuPy | ~11 modules | CuPy ROCm wheels (Linux) | validator installs on Linux; Windows skips (question 3) |
| pynvml | `utils/tf32.py` (Ampere detection), `utils/module.py` | amdsmi | inert: tf32 detection returns True when pynvml is missing; amd-develop removed nvidia-ml-py; no change |
| TensorRT / torch_tensorrt / polygraphy / cuda-python | `networks/trt_compiler.py`, `networks/utils.py` | MIGraphX (new backend) | inert on ROCm; proposed deferral `monai-tensorrt-to-migraphx`, awaiting Jeff (question 1) |
| NVTX | `handlers/nvtx_handlers.py`, `utils/nvtx.py` via `torch.cuda.nvtx` | torch maps to roctx on ROCm | run the NVTX tests |
| apex | `networks/layers/factories.py` (InstanceNorm3dNVFuser) | none needed | optional, skips |

## Risk list

1. **GMM warp reductions on wave64 (gfx942/gfx90a).** Reading the code, the reduction is correct on wave64. Each 32-lane half of a 64-wide wavefront reduces with offsets 16..1. Lane 0 and lane 32 each depend only on their own half, because `__shfl_down` with width 64 reads lane+offset <= lane+31. The second stage runs in threads 0..31 only. Upstream #6161 added the `__HIP_PLATFORM_AMD__` branch for this. It is still the highest-value wave64 check: `tests/networks/layers/test_gmm.py` must run, not skip (it needs the JIT build to succeed).
2. **GMM on wave32 (gfx1100/gfx1201).** There the code matches CUDA semantics exactly (32-wide shuffles), so the risk is low. But no AMD wave32 run exists, and the hardcoded `>> 5` / `& 31` are only safe because the code never assumes more than 32 lanes. Do not "fix" them to `warpSize`: on wave64 that would change which lanes `s_matrix_component` indexes. Any change here must stay correct on both widths (portability rule).
3. **`BLOCK_SIZE 32` launches** (permutohedral, trainable bilateral). These are half a wavefront on CDNA, so occupancy is poor but the result is correct. A performance note only; not in scope.
4. **Fresh allocations not zeroed.** `hash_table.cuh` memsets values, entries, and keys. `ranks` (`cudaMalloc(&ranks, ...)`, :69) and `zeros` (:72) are not memset. Check whether the kernels write before they read. If a permutohedral test is flaky on one arch only, suspect this first.
5. **Windows extension build (gfx1151).** Upstream CI compiles only the CPU `_C` on Windows with MSVC (`/openmp`, setup.py `omp_flags`). Nobody has built the device half on Windows. Under ROCm Windows torch, `CUDAExtension` compiles the hipified `.hip` files with the ROCm clang/hipcc and the `.cpp` files with the host compiler (MSVC cl, or clang-cl if the toolchain selects it), and links amdhip64. Expected failure classes: the THC/THH header path; `/openmp` with clang-cl (it needs `-openmp`/`/openmp` plus a libomp the ROCm SDK may not ship); a `setup.py bdist_wheel` versus `pip --no-build-isolation` difference; `_rocm_install_requires()` failing to find `hipcc`/`rocm-sdk` on PATH, then falling back to series 7.14 and emitting a `torch[device-gfx942,gfx950]` requirement that is wrong for gfx1151 (harmless with `--no-deps`, wrong metadata otherwise). Under the "just works" rule, any toolchain workaround goes in `setup.py`, guarded to win32, never only in a README.
6. **Windows JIT (GMM).** `torch.utils.cpp_extension.load` on Windows ROCm needs ninja and a working host compiler in the environment at test time, not only at build time. A JIT failure surfaces as test_gmm errors, not as a build failure. Classify DLL-load or exit-127 failures as environment failures first (stop-discipline rule).
7. **hipCIM/CuPy absent on Windows.** WSI and CuPy paths skip there (questions 2 and 3); expect a larger skip count on Windows than on Linux, and record it.
8. **MIOpen intermittent load fault (gfx942, known to the AMD team).** Use `--rerun-failures` and report flakes separately (see strategy).
9. **RDNA numerical tolerance.** Network tests with tight tolerances (dynunet with `use_gemm_transpose`, which is gated on `torch.version.hip` and therefore active on RDNA too; swin_unetr SDPA, which defaults on under ROCm; densenet pretrain consistency) may differ on gfx11/gfx12. Classify each before touching a tolerance; any bump is gated on ROCm, never loosening CUDA.
10. **Python 3.12+ only** (amd-develop setup.cfg). The validator's venv must use 3.12 or later on every platform.
11. **CUDA no-regression.** Any `moat-port` change to `.cu`/`setup.py` must also be compiled with real nvcc (mi355 fleet host) before it goes to Jeff (cuda-compile-before-presenting rule). A validation-only round with no source delta has nothing to compile.

## File-by-file change list

At the base, none is known to be required. The expected shape of the work:

- `moat-port` = amd-develop f9f65c9 plus only the fixes that wave32 or Windows validation proves necessary. Each fix is the smallest change, keeps the CUDA path and the AMD team's structure, and is correct on both wavefront widths.
- Likely touch points, if anything breaks: `setup.py` (Windows toolchain guards, `omp_flags` for clang-cl), `monai/_extensions/loader.py` (Windows JIT flags), `monai/csrc/filtering/permutohedral/*` (the THC include, uninitialized `ranks`/`zeros` if implicated), test tolerances gated on ROCm.
- Not in scope unless Jeff or the AMD team asks: syncing amd-develop with upstream `dev` (162 commits), back-porting #9153 (ROCM_HOME detection, JIT cache key), performance work on 32-thread blocks.

## Build commands

Linux (any arch; gfx942 shown). Use a job venv with Python >= 3.12 and a ROCm 7.x torch for the host arch from the AMD/TheRock index. Follow docs/install/installation.rst for the `rocm[devel]` environment variables (ROCM_PATH, ROCM_DEVEL_PATH, LIBRARY_PATH, CPATH).

```bash
export AMDGPU_TARGETS=gfx942          # gfx90a / gfx1100 / gfx1201 / gfx1151 per host
export PYTORCH_ROCM_ARCH=$AMDGPU_TARGETS
export OMP_NUM_THREADS=1
$PY -m pip install -r requirements-dev.txt -c amd-constraints.txt   # plus the AMD index flags from the install doc
BUILD_MONAI=1 FORCE_CUDA=1 utils/timeit.sh MONAI build -- $PY setup.py bdist_wheel
$PY -m pip install --no-deps dist/amd_monai-*.whl
$PY -c "import monai, monai._C, torch; print(monai.__version__, torch.version.hip)"
$PY -c "from monai._extensions import load_module; load_module('gmm', {'CHANNEL_COUNT':1,'MIXTURE_COUNT':2,'MIXTURE_SIZE':3}, verbose_build=True)"
```

The `monai._C` import must succeed, and `monai._C` must carry the GPU paths. Proof is that `tests/networks/layers/filtering/test_bilateral_approx_cuda.py` and `test_phl_cuda.py` run and pass. A `_C` built without `FORCE_CUDA=1` is CPU-only and still passes the import check; only the GPU tests expose it.

Windows (gfx1151): the same sequence in a TheRock ROCm Windows torch venv (PowerShell equivalents). Record the exact compiler that `CUDAExtension` selects for the `.cpp` and `.hip` files.

## Test plan

GPU tests that exercise the port (must run, not skip, on every platform):

```bash
$PY -m pytest -v tests/networks/layers/filtering tests/networks/layers/test_gmm.py \
  tests/networks/layers/test_grid_pull.py tests/networks/layers/test_lltm.py \
  tests/utils/test_enum_bound_interp.py tests/transforms/test_spatial_gpu_support.py
```

The full unit suite, as the AMD team runs it, with the fresh-process rerun:

```bash
utils/timeit.sh MONAI test -- $PY tests/runner.py -p '^(?!test_integration).*' --rerun-failures 2
```

Or `./runtests.sh --unittests`, without `--build`, because the wheel is already installed. Run integration tests with `./runtests.sh --net` on Linux wave64 and wave32. Linux needs hipCIM (`amd-hipcim`) and the CuPy ROCm wheel installed, so the cucim and CuPy paths are exercised.

Evidence per platform: passed/failed/errored/skipped counts for the first pass and after the rerun; the list of GPU tests that skipped, with the reason; the head sha; the torch and ROCm versions. Non-GPU regression set: the same suite's CPU cases. They must not regress relative to the amd-develop base on that host. Running the base first gives the baseline for any failure that might predate MOAT.

Platform order (Jeff's ruling): Linux wave64 (gfx942, gfx90a if available) and Linux wave32 (gfx1100, gfx1201) first; Windows gfx1151 last, only after both Linux gates pass at the same head.

## Open questions

Recorded in notes.md, "Planner (2026-11-10, linux-gfx942): questions for Jeff before `planned`": (1) TensorRT -> MIGraphX, proposed deferral `monai-tensorrt-to-migraphx`; (2) hipCIM on Windows; (3) CuPy on Windows; (4) the delivery vehicle. The plan is not marked `planned` until Jeff answers 1-3.
