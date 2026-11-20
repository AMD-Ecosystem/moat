# MONAI plan

Written 2026-11-20 (planner, linux-gfx942) for upstream `dev` at a85d75e11cdd9c6d6d13500dc7f3960dd07dd7d3 ("Update docs for 1.6.2 (#9289)", 2026-11-18). This replaces the 2026-11-10 plan, which was written against AMD-Ecosystem/MONAI `amd-develop` and was superseded by Jeff Daily's independent-port ruling of 2026-11-10 (notes.md). That plan's amd-develop facts (the f9f65c9 base, `FORCE_CUDA` on amd-develop, `_rocm_install_requires`, the `amd-monai` package name, `tests/runner.py --rerun-failures`) do not apply. amd-develop was not read for this plan.

## Project

- Upstream: https://github.com/Project-MONAI/MONAI, default branch `dev`, Apache-2.0.
- Fork: https://github.com/AMD-Ecosystem/MONAI, owned by AMD's ROCm-LS team. Its GitHub default branch stays `amd-develop` (Jeff's ruling, the one exception to the moat-port-default rule). `fork_default_branch` = `dev` (the upstream mirror; stale, so the porter fast-forwards or fetches upstream `dev` before cutting).
- **Port base: upstream `dev` at a85d75e1.** `moat-port` is cut from that commit (or a later `dev` tip, recorded) and pushed to the fork as a new branch.
- Scope: the whole project (Jeff, 2026-11-10 and 2026-11-18): `monai._C`, the runtime-built GMM extension, the optional GPU dependencies with their ROCm equivalents where they exist, and the GPU, unit and integration suites. Two newly found NVIDIA-only optional paths (nvImageCodec, kvikio) await Jeff's ruling; see Open questions.
- Hard dependency: `cupy` (MOAT project, review-passed, Windows ROCm build of CuPy), recorded in `depends_on` by the 2026-11-18 ruling.

## Existing AMD support

Upstream already carries part of a ROCm port, contributed by the AMD team, and that code is in scope as upstream code:

- Project-MONAI/MONAI#9153 (merged 2026-10-06, by an AMD engineer): `setup.py` accepts `ROCM_HOME` when `CUDA_HOME` is None, so `BUILD_MONAI=1` builds the device half of `_C` on a ROCm torch without `FORCE_CUDA`; `loader.py` keys the GMM JIT cache on `torch.version.hip`; `Dockerfile.rocm` (Instinct gfx942/gfx950, TheRock `rocm` wheels from `ROCM_INDEX_URL`, `amd-hipcim` from the AMD index); `test_densenet.py` skips bit-exactness on ROCm. #6161 (2023) added the `__HIP_PLATFORM_AMD__` shuffle branch in `gmm_cuda.cu`.
- Open upstream, by the same AMD team: issue #9285 (`monai[all]` / `monai[cucim]` not installable on ROCm), PR #9161 (a vendor mechanism that substitutes `cucim-cu*` -> `amd-hipcim`, `cupy-cuda*` -> a ROCm CuPy, and so on in the extras), PR #9290 (report `HIP version` in `print_config`/`check_env`). Both PRs were updated 2026-11-19.
- AMD-Ecosystem/MONAI `amd-develop` / `amd-monai` (ROCm-LS) exists and is authoritative, but is excluded from this plan by ruling.

Judgment: authoritative but partial. Upstream builds and documents Instinct on Linux only; nobody claims Radeon (wave32) or Windows. Nothing in the extension is an NVIDIA-tuned performance kernel (no CUTLASS, CuTe, wgmma or tensor-core code), so a mechanical port is the whole answer and no AMD-native rewrite is called for. Upstream does not link platform forks, so an upstream PR is a valid vehicle; whether we send one, given the AMD team's own open PRs, is Jeff's later call (Open questions).

## Build classification

**torch-extension**, Strategy B (recorded `ext_type: torch-extension`).

- `setup.py:95-155` `get_extensions()`: with `BUILD_MONAI=1`, `CUDAExtension` named `monai._C` from `monai/csrc/**/*.cpp` + `**/*.cu`, define `WITH_CUDA`; `CppExtension` (CPU only) when no toolkit is found. `BUILD_CUDA = FORCE_CUDA or (torch.cuda.is_available() and (CUDA_HOME or ROCM_HOME))` (setup.py:41-46).
- `monai/_extensions/loader.py` `load_module()`: `torch.utils.cpp_extension.load` JIT-builds `monai/_extensions/gmm` at first use (`GaussianMixtureModel`), with a 300 s build timeout. Its sources ship as package data.
- `pyproject.toml` `[build-system]` requires `torch>=2.8.0`; builds against an installed ROCm torch need `--no-build-isolation`, or pip builds against a CUDA torch from PyPI.
- Everything else is pure Python on PyTorch.

## Port strategy

B: torch's hipify runs on every `.cu`/`.cuh` file through both `CUDAExtension` and JIT `load`, and #9153 already makes the build pick the device path on ROCm. Sources stay in CUDA spelling. Expect few source edits on Linux; the work is the defects listed below, then building and testing on wave64, wave32 and Windows, plus whatever those platforms expose.

## CUDA surface inventory

surface.json floor (regex census at a85d75e): 13 files, 24 kernels, 30 launches, 131 device-code markers, 4 warp-intrinsic sites. No cuBLAS/cuFFT/cuRAND/cuSPARSE/cuSOLVER/cuDNN, no Thrust/CUB/CUTLASS, no NCCL, textures, driver API or NVRTC. This is the same surface the 2026-11-10 census found.

| component | files | surface | HIP mapping / verdict |
|---|---|---|---|
| bilateral (precise + PHL) | `csrc/filtering/bilateral/bilateralfilter_cuda{,_phl}.cu` | `__constant__` buffers, `cudaMemcpyToSymbol` | hipify adds `HIP_SYMBOL`; fine |
| permutohedral lattice | `csrc/filtering/permutohedral/permutohedral_cuda.cu`, `hash_table.cuh` | `BLOCK_SIZE 32` blocks, `atomicCAS`/`atomicExch` hash table, raw `cudaMalloc`/`cudaMemset`, `__constant__` pointers via `cudaMemcpyToSymbol`, `#include <THC/THCAtomics.cuh>` | hipify rewrites the THC include to `THH/THHAtomics.cuh`, which the ROCm torch wheel ships (checked: torch 2.16 dev, `include/THH/THHAtomics.cuh` -> `ATen/hip/Atomic.cuh`). `values`/`entries`/`keys` are memset. The un-memset `ranks`/`zeros` (hash_table.cuh:68-75) sit under `LINEAR_D_MEMORY`, which is commented out (hash_table.cuh:19): dead code, not a risk. `matrix` (permutohedral_cuda.cu:473) is not memset; `createMatrix` must write every entry before `splat` reads it (reviewer checks). |
| trainable (joint) bilateral | `csrc/filtering/trainable_*/*_gpu_{forward,backward}.cu` | many `__constant__` symbols, `BLOCK_SIZE 32`, half dispatch | fine |
| resample push/pull | `csrc/resample/pushpull_cuda.cu`, `csrc/utils/resample_utils.h` | 1024-thread launch with `C10_LAUNCH_BOUNDS_1(1024)`, `gpuAtomicAdd` incl. half/double, THC include | HIP max flat work-group is 1024; half/double atomics are CAS-emulated on RDNA, correct |
| LLTM | `csrc/lltm/lltm_cuda.cu` | 1024 threads, no launch bounds | HIP default flat-work-group max is 1024; fine |
| GMM (JIT) | `_extensions/gmm/gmm_cuda.cu`, `gmm_cuda_linalg.cuh` | `__SHFL_DOWN`/`__SHFL_XOR` (AMD branch: no `_sync`, default width = warpSize), `warp_index = tid >> 5`, `lane = tid & 31`, blocks of `WARPS << 5`, `dim3(32, MIXTURE_COUNT)` blocks | wavefront-sensitive, analysed below; correct on both widths by reading |

GMM on wave64, read 2026-11-20 at a85d75e: the `__SHFL_DOWN` reductions (offsets 16..1, gmm_cuda.cu:91-96 and :160-ff) leave each 32-lane half's sum in lane 0 / lane 32, which are the only lanes that write `s_matrix_component` (`lane_index == 0`); a down-shift by at most 16 from lane <= 15 never leaves its half for the lanes that matter. The second stage runs in `warp_index == 0` (threads 0..31) with offsets starting at `warp_count/2`, so lane 0 reads lanes `< warp_count` only. The `__SHFL_XOR` butterflies in `GMMFindSplit` and `GMMcommonTerm` (xor masks 1..16) never flip bit 5, so on a `dim3(32, MIXTURE_COUNT)` block each 32-thread row stays inside its own half of a 64-wide wavefront. On wave32 the code has CUDA semantics exactly. Do not "fix" `>> 5` / `& 31` to `warpSize`: that would change which lanes index `s_matrix_component` on wave64.

Python-level surface (all optional imports; missing modules make the paths and their tests skip):

| dependency | where | ROCm equivalent | plan |
|---|---|---|---|
| cuCIM | `data/wsi_reader.py`, `transforms/*`, `metrics/utils.py`, `auto3dseg/utils.py` | hipCIM (`amd-hipcim`, Linux only, AMD index) | Linux: install and exercise. Windows: CPU readers (openslide/tifffile), named as a CPU fallback (Jeff, 2026-11-18). |
| CuPy | 13 modules (`transforms`, `metrics`, `data`, `utils/type_conversion.py`, ...) | CuPy ROCm wheels (Linux); MOAT `cupy` port (Windows) | Linux: CuPy ROCm wheel. Windows: build CuPy from the MOAT cupy port per `projects/cupy/notes.md` "Install as a dependency", and run these paths (Jeff, 2026-11-18). |
| TensorRT / torch_tensorrt / polygraphy / cuda-python | `networks/trt_compiler.py`, `networks/utils.py` | MIGraphX (would be a new backend) | deferred (`monai-tensorrt-to-migraphx`, Jeff ruled `defer` 2026-11-18); stays inert, tests skip; the port's documentation names the gap |
| nvImageCodec pydicom plugin | `data/nvimgcodec_pydicom_plugin.py`, `NvImgCodecPydicomReader` in `data/image_reader.py`, `monai[nvimgcodec]` | rocJPEG via rocPyDecode, baseline JPEG only; nothing for JPEG Lossless / J2K / HTJ2K | **new since the last plan; awaiting Jeff** (question 1, deferral `monai-nvimgcodec-dicom-to-rocjpeg`). Until ruled: inert, reader reports unavailable, CPU pydicom/ITK decoding. |
| kvikio (RAPIDS GDS) | `GDSDataset` in `data/dataset.py`, `to_gpu` in `data/image_reader.py` | hipFile, reachable through `torch.cuda.gds` on ROCm; no ROCm kvikio | **awaiting Jeff** (question 2, deferral `monai-kvikio-gds-to-hipfile`). Until ruled: inert, tests skip. |
| nvsubquadratic (Hyena blocks) | `networks/blocks/hyena.py`, `pythonapp-hyena-gpu.yml` | not needed: the base package is pure PyTorch (py3-none-any wheel 0.2.0); only its optional `[cuda]` extra (`subquadratic-ops-torch-cu13`) has CUDA kernels | validator installs `nvsubquadratic --no-deps` (it pins `torch<2.15`, which would replace the ROCm torch) plus `omegaconf einops`, and runs the Hyena tests on the GPU |
| pynvml | `utils/tf32.py`, `utils/module.py` (`compute_capability_geq`), MLflow GPU metrics | amdsmi | inert: `has_ampere_or_later` returns False when `torch.version.cuda` is None; nothing to port |
| NVTX | `handlers/nvtx_handlers.py`, `utils/nvtx.py` via `torch.cuda.nvtx` | roctx through torch | run the NVTX tests |
| apex | `networks/layers/factories.py` | none needed | optional, skips |
| `nvidia-smi` | `tests/test_utils.py` `query_memory` | falls back to `range(n)` on OSError | no change |

## Risk list

1. **`_compiled_unsupported()` disables the compiled path on gfx12 (a real ROCm defect).** `monai/transforms/spatial/functional.py:58-96` decides whether `monai._C` supports the device from `torch.cuda.get_device_properties(device).major` (>= 12, or above `MONAI_MAX_COMPUTE_CAPABILITY` from `TORCH_CUDA_ARCH_LIST`). On ROCm, `major`/`minor` are the gfx version: gfx942 = 9.4, gfx1100 = 11.0, gfx1151 = 11.5, gfx1201 = 12.0. `TORCH_CUDA_ARCH_LIST` is unset on ROCm, so `max_compute_capability()` is 0, and the heuristic `major >= 12` marks every gfx12 device (gfx1201, gfx1250) unsupported. `Resample`, `SpatialResample` and `Warp` then fall back silently to `grid_sample` on RDNA4 and newer, so the ported `grid_pull` never runs there in normal use. Fix: return False when `torch.version.hip` is set. On ROCm, an extension built without the device's arch fails loudly with "no kernel image is available"; it does not silently produce wrong results, which is the CUDA hazard the function guards against. Update `tests/transforms/test_spatial_gpu_support.py` (`test_cuda_device_detection`, `test_resample_compilation_flag_respected`, `test_compiled_unsupported_logic` all encode the CUDA rule) so the ROCm expectation is `False`, without changing the CUDA cases.
2. **GMM JIT build timeout.** `load_module(..., build_timeout=300)` raises `TimeoutError` if hipcc takes more than 5 minutes; `test_gmm.py` builds several defines variants. hipcc of `gmm_cuda.cu` per arch is expected well under that on Linux, but measure it on every platform (Windows and the APU are slowest). If it is close, the fix is to raise the default in `loader.py` (and the porter records the measured time), not to skip.
3. **GMM test hides build failures as skips.** `test_gmm.py:286-291` calls `skipTest` on "Error building extension", and the class is `@skip_if_quick`. A JIT failure therefore reads as a skip, and `--quick` skips it entirely. The validator treats any GMM skip as a failure and runs without `QUICKTEST`.
4. **Compiled-path tests need `BUILD_MONAI=1` at test time.** `USE_COMPILED = HAS_EXT and BUILD_MONAI == "1"` (`monai/config/deviceconfig.py:30`), and `skip_if_no_cpp_extension` keys on `USE_COMPILED`. Without the variable in the test environment, every `_C` test skips and the run looks clean. The validator also checks that `monai._C` was built with the device path: `test_bilateral_approx_cuda.py` and `test_phl_cuda.py` must run and pass, not skip.
5. **GMM on wave64 (gfx942/gfx90a).** Correct by reading (above), and #6161 targeted it, but only a run proves it: `tests/networks/layers/test_gmm.py` must run and pass.
6. **GMM on wave32 (gfx1100/gfx1201/gfx1151).** CUDA semantics; low risk; no AMD wave32 evidence yet.
7. **`BLOCK_SIZE 32` launches** (permutohedral, trainable bilateral): half a wavefront on CDNA, correct but low occupancy. Performance only; out of scope.
8. **Windows extension build (gfx1151).** Upstream CI builds only the CPU `_C` on Windows. Under a ROCm Windows torch, `CUDAExtension` compiles `.cpp` with clang-cl and the hipified `.hip` with ROCm clang. Expected failure classes: `omp_flags()` returns `/openmp` for win32, which clang-cl accepts only with a libomp that the ROCm SDK may not ship (check the `torch_parallel_backend()` result first: OpenMP flags are added only when ATen's backend is OpenMP); the torch wheel's `[device-gfx1151]` extra; ninja and the host compiler on PATH at test time for the GMM JIT. Under the "just works" rule, any workaround goes in `setup.py`/`loader.py`, guarded to win32 + ROCm, never only in a README.
9. **Windows JIT (GMM).** `torch.utils.cpp_extension.load` needs the compiler environment at test time. Classify DLL-load or exit-127 failures as environment failures first.
10. **CuPy on Windows from source.** The Windows validator builds the MOAT cupy port into the MONAI venv before testing (Jeff, 2026-11-18). If that build proves too heavy for the gfx1151 host, record the measured reason and bring it to Jeff; never skip silently.
11. **Python dependency installs replace the ROCm torch.** `pip install -e '.[all]'` pulls `cucim-cu13`, `nvidia-ml-py`, `cupy-cuda13x` (via the `nvimgcodec` extra) and packages pinning CUDA torch. Install from a filtered list (`python monai/config/print_dependencies.py all testing | grep -vE '^(cucim-cu|nvidia-|cupy-cuda|nni|polygraphy|onnx_graphsurgeon)'`, as `Dockerfile.rocm` does), and check `torch.version.hip` after every install.
12. **RDNA numerical tolerance.** Tight network tests (dynunet `use_gemm_transpose`, swin_unetr with SDPA, densenet consistency, which upstream already skips on ROCm) may differ on gfx11/gfx12. Classify each failure before touching a tolerance; any bump is gated on `torch.version.hip`, never loosening CUDA.
13. **Intermittent MIOpen faults.** MIOpen conv tests on ROCm can fail with code-object load errors in long single-process runs. If seen, rerun the failing module in a fresh process and record first-pass and rerun results separately; a test that passes only on rerun is a flake, recorded as one.
14. **CUDA no-regression.** Any change to `.cu`, `setup.py` or `loader.py` is compiled with real nvcc (mi355 fleet host) before it goes to Jeff. A ROCm-only Python change (`functional.py`) gets the CUDA unit tests it touches run on that host.

## File-by-file change list

Required at the base:

- `monai/transforms/spatial/functional.py`: `_compiled_unsupported()` returns False on a ROCm torch (`torch.version.hip`), with a one-line docstring note. CUDA logic unchanged (risk 1).
- `tests/transforms/test_spatial_gpu_support.py`: ROCm branch of the expectations; CUDA cases unchanged.
- Documentation: add a short "AMD GPUs (ROCm)" install section to `docs/source/installation.md`, covering:
  - the build command (`BUILD_MONAI=1 pip install --no-build-isolation .` against a ROCm torch, with `PYTORCH_ROCM_ARCH`);
  - the `BUILD_MONAI=1` runtime requirement;
  - the optional ROCm equivalents (hipCIM on Linux; CuPy ROCm);
  - the CPU fallbacks named as such (whole-slide reading on Windows, and whatever Jeff rules for nvImageCodec/kvikio);
  - the TensorRT gap.
  Keep it out of the way of #9161's planned edits to the same file (Open question 3).

Likely, if validation exposes them:

- `setup.py`: Windows + ROCm guards (`omp_flags` under clang-cl), risk 8.
- `monai/_extensions/loader.py`: Windows JIT flags; the build timeout if it is measured close to 300 s (risk 2).
- Test tolerances gated on ROCm, per failure, with the measured differences recorded.

Not in scope unless Jeff rules otherwise: the dependency-extras substitution of #9161 and the version reporting of #9290 (question 3); a MIGraphX backend (deferred); a rocJPEG DICOM decoder (question 1); a hipFile/kvikio GDS backend (question 2); performance work on 32-thread blocks; `Dockerfile.rocm` changes beyond what a validated fix needs.

## Build commands

Linux (any arch; gfx942 shown). Job venv, Python >= 3.10 (3.12 matches `Dockerfile.rocm`), a ROCm 7.x/10.x torch for the host arch from the TheRock index, the `rocm[devel]` SDK environment as `Dockerfile.rocm` sets it (`ROCM_PATH`, `ROCM_HOME`, `CPATH`, `LIBRARY_PATH`, `LD_LIBRARY_PATH`, and the bitcode symlink), or a system `/opt/rocm`.

```bash
export PYTORCH_ROCM_ARCH=gfx942          # gfx90a / gfx1100 / gfx1201 / gfx1151 per host
export OMP_NUM_THREADS=1 MAX_JOBS=<physical cores>
$PY monai/config/print_dependencies.py build-system | xargs -d '\n' $PY -m pip install --no-build-isolation
$PY monai/config/print_dependencies.py all testing \
  | grep -vE '^(cucim-cu|nvidia-|cupy-cuda|nni|polygraphy|onnx_graphsurgeon)' > agent_space/rocm-req.txt
$PY -m pip install -r agent_space/rocm-req.txt pytest parameterized
$PY -c "import torch; print(torch.__version__, torch.version.hip)"   # still the ROCm torch
BUILD_MONAI=1 utils/timeit.sh MONAI compile -- $PY -m pip install --no-build-isolation --no-deps -e .
export BUILD_MONAI=1
$PY -c "import monai, monai._C, torch; from monai.config import USE_COMPILED; print(monai.__version__, torch.version.hip, USE_COMPILED)"
$PY -c "from monai._extensions import load_module; load_module('gmm', {'CHANNEL_COUNT':1,'MIXTURE_COUNT':2,'MIXTURE_SIZE':3}, verbose_build=True)"
# optional GPU deps
$PY -m pip install --extra-index-url https://pypi.amd.com/rocm-<series>.0/simple/ amd-hipcim && $PY -c "from cucim import CuImage"
$PY -m pip install <CuPy ROCm wheel for the host ROCm>          # check torch.version.hip afterwards
$PY -m pip install --no-deps nvsubquadratic && $PY -m pip install omegaconf einops
```

The setup log must print `BUILD_MONAI_CUDA=True`. The editable install hipifies in place: the `*.hip` / `*_hip.*` siblings and the JIT's hipified GMM files are untracked build output. They must not be committed; the porter adds them to `.gitignore` if upstream does not already ignore them. Also confirm that the documented non-editable route works from a fresh clone (`pip install --no-build-isolation .`, then run the GPU tests from outside the tree), because the GMM JIT sources ship as package data.

Windows (gfx1151): the same sequence in a TheRock ROCm Windows torch venv (PowerShell equivalents, `torch[device-gfx1151]`), plus CuPy built from the MOAT cupy port per `projects/cupy/notes.md` "Install as a dependency". No hipCIM. Record the compiler `CUDAExtension` selects for the `.cpp` and `.hip` files, and the GMM JIT build time.

## Test plan

GPU tests that exercise the port's compiled code (must RUN and pass, not skip, on every platform; `BUILD_MONAI=1`, no `QUICKTEST`):

```bash
BUILD_MONAI=1 utils/timeit.sh MONAI test -- $PY -m pytest -v -rs \
  tests/networks/layers/filtering tests/networks/layers/test_gmm.py \
  tests/networks/layers/test_grid_pull.py tests/networks/layers/test_lltm.py \
  tests/utils/test_enum_bound_interp.py tests/transforms/test_spatial_gpu_support.py \
  tests/networks/blocks/warp tests/transforms/test_resampler.py \
  tests/transforms/test_resample_backends.py tests/transforms/functional/test_resample.py
```

Every GPU test that skips, with its reason, goes in notes.md; a skip in this set is a failure unless it is an upstream skip for a reason unrelated to ROCm. On gfx12, also check that `Resample` takes the compiled path after the risk-1 fix (`_compiled_unsupported(torch.device('cuda'))` is False).

Optional-dependency paths (must run where the dependency exists):

```bash
# Linux: hipCIM + CuPy installed; the 29 test files that reference cucim or cupy
BUILD_MONAI=1 $PY -m pytest -v -rs $(grep -rlE "cucim|cupy" tests --include='test_*.py')
BUILD_MONAI=1 $PY -m pytest -v -rs tests/networks/blocks/test_hyena_block.py tests/networks/nets/test_hyena_nd_unetr.py
```

Full unit suite (upstream's runner), then integration:

```bash
BUILD_MONAI=1 utils/timeit.sh MONAI test -- ./runtests.sh --unittests      # no --build: already installed
BUILD_MONAI=1 utils/timeit.sh MONAI test -- ./runtests.sh --net            # integration (35 test files), Linux wave64 and wave32
```

Evidence per platform: passed/failed/errored/skipped counts (first pass, plus any fresh-process rerun, recorded separately); the GPU-test skip list with reasons; head sha; torch and ROCm/HIP versions; GMM JIT build time. Non-GPU regression set: the same suite's CPU cases. A failure there is checked against the same suite run at the `dev` base sha on the same host before it is charged to the port. The unit suite has 834 `test_*.py` files; 58 carry CUDA-, extension- or GPU-dependency skip gates.

Platform order (Jeff's ruling): Linux wave64 (gfx942; gfx90a if available) and Linux wave32 (gfx1100, gfx1201) first; Windows gfx1151 last, only after both Linux gates pass at the same head. gfx1201 is the platform that proves risk 1.

CUDA no-regression (before presenting): on the mi355 host with real nvcc, `BUILD_MONAI=1 pip install --no-build-isolation -e .` against a CUDA torch, then `pytest tests/transforms/test_spatial_gpu_support.py tests/networks/layers/test_grid_pull.py`. Record the nvcc command and result, or `cuda-not-validated: <exact error>` if the host has only a ROCm torch.

## Open questions

Recorded in notes.md, "Planner (2026-11-20, linux-gfx942): questions for Jeff before `planned`". The project stays in `planning` until Jeff answers 1-3:

1. nvImageCodec GPU DICOM decoding: defer, or a rocJPEG-backed decoder (baseline JPEG only).
2. kvikio GPUDirect Storage: defer, a kvikio port as its own project, or a `torch.cuda.gds` (hipFile) backend.
3. Overlap with the AMD team's open upstream PRs #9161 (dependency extras) and #9290 (HIP version reporting): stay clear of them, or cover the same ground independently.
4. (Not blocking the porter.) Delivery: an upstream PR to Project-MONAI, alongside the AMD team's open PRs, or comparison with amd-develop only. Jeff decides after validation.
