# rl (TorchRL) port plan

## Decision needed before the porter starts

One question for jeff; the rest of the plan does not depend on the answer.

**Should the upstream PR also turn on TorchRL's ROCm wheel lane?** `.github/workflows/build-wheels-linux.yml` defaults `with-rocm` to `disable` (and `release.yml` defaults `build_rocm: false`). The lane was switched off in pytorch/rl#3603 ("torchrl is CPU-only") and was never switched back on, although #3677 brought the CUDA lane back.

- (a) Include the flip in the same PR: `with-rocm` defaults to `enable`, `build_rocm` defaults to `true`. This gives ROCm users a GPU segment tree in the nightly and release wheels. MOAT cannot run pytorch/test-infra's ROCm wheel jobs, so the change would ship without CI evidence from us, and it adds wheel-build cost upstream has to accept.
- (b) The PR changes `setup.py`, docs and tests only, and its body proposes the lane flip as a follow-up for the maintainers. (Recommended: the smallest complete port, and the CI cost stays upstream's call.)
- (c) Two PRs: the source PR first, then a separate CI PR once the first merges.

The porter must not touch `.github/` until this is ruled. This is not a scope cut: under every option the whole CUDA surface (one `.cu`) is ported.

## Project
- Upstream: https://github.com/pytorch/rl (TorchRL, PyTorch org, MIT)
- Fork: https://github.com/AMD-Ecosystem/rl, default branch `main` (mirror); port branch `moat-port`
- Planned at upstream `1d94cfa5c35f9305b402fb71f0378f9cd7f09b28` (2026-11-19)

## Existing AMD support
**Improvable.** No upstream or AMD port of the CUDA extension exists. Intake found no ROCm/HIP PR or issue on pytorch/rl and no ROCm or AMD-Ecosystem fork. Upstream is ROCm-aware in two places only:
- `torchrl/modules/tensordict_module/_rnn_precision.py` selects `ieee` precision for its Triton RNN kernels when `torch.version.hip` is set.
- `packaging/pkg_helpers.bash` exports `FORCE_CUDA=1` for a `rocm*` `CU_VERSION`, a legacy path that the test-infra wheel builds may not use.

Everything else in TorchRL is pure Python on torch and tensordict, so it already runs on ROCm torch. The gap: under ROCm torch the prioritized replay sampler silently drops to the CPU segment tree. Upstream accepts platform work in-tree (it carries a ROCm wheel lane), so an upstream PR is the right vehicle. These kernels are not performance-tuned (no tensor cores, no CUTLASS), so this is a mechanical port, with no AMD-native rewrite.

## Build classification
**torch-extension.** Evidence:
- `setup.py:19-24` imports `BuildExtension`, `CppExtension`, `CUDA_HOME` and `CUDAExtension` from `torch.utils.cpp_extension`.
- `setup.py:238-239`: `extension = CUDAExtension if build_cuda else CppExtension`.
- `setup.py:322-324` adds `torchrl/csrc/*.cu` to the sources only when `build_cuda`.
- `pybind.cpp:13,28` registers the `CudaSum/MinSegmentTreeFp32/Fp64` bindings under `#ifdef WITH_CUDA`, which setup.py defines for both `cxx` and `nvcc`.

Recorded with `set-ext-type rl torch-extension`.

## Port strategy
**B: torch build-time hipify.** Under ROCm torch, `CUDAExtension` hipifies the sources and compiles `cuda_segment_tree.cu` as `cuda_segment_tree.hip` with hipcc. The extension uses only ATen/c10 CUDA APIs that hipify maps (`at::cuda::getCurrentCUDAStream`, `c10::cuda::CUDAGuard`, `C10_CUDA_KERNEL_LAUNCH_CHECK`, `<<<>>>`), so no kernel source change is expected.

The real change is the build gate. `setup.py::_should_build_cuda()` returns False on ROCm torch: it requires `torch.version.cuda` (None under ROCm) and an `nvcc` whose version matches it. The fix teaches the gate about ROCm, using torch's own `ROCM_HOME` and `torch.version.hip`, and keeps the CUDA path unchanged.

## CUDA surface inventory
One file: `torchrl/csrc/cuda_segment_tree.cu` (233 lines). The surface census finds 4 kernels, 1 `__device__` helper and 4 launches.

| item | where | HIP mapping / risk |
|---|---|---|
| `ApplyOp<T,IsMin>` `__device__ __forceinline__` | .cu:17 | direct |
| `SetLeavesKernel` `<<<1,1>>>` serial leaf writes (keeps duplicate-index order identical to the CPU tree) | .cu:26 | direct; deliberately single-thread |
| `RecomputeLevelKernel` 256-thread blocks, one launch per tree level | .cu:39 | direct; `kThreads=256` is a multiple of 32 and 64 |
| `QueryKernel` per-query range fold | .cu:51 | direct |
| `ScanLowerBoundKernel` prefix-sum descent | .cu:76 | direct |
| `at::cuda::getCurrentCUDAStream`, `c10::cuda::CUDAGuard`, `C10_CUDA_KERNEL_LAUNCH_CHECK` | .cu | hipify maps them to the HIP-masqueraded c10 APIs |
| `torch::Device(torch::kCUDA)` defaults, `values.is_cuda()` checks | `cuda_segment_tree.h`, .cu | true for ROCm devices (HIP masquerades as CUDA in torch) |
| tree storage | `cuda_segment_tree.h:37` `torch::full(..., identity)` | initialised, so the ROCm "fresh allocation is not zero" hazard does not apply |

The kernel has no warp intrinsics, `warpSize`, literal-32 lane logic, shared memory, atomics, textures, CUDA libraries (cuBLAS, Thrust, CUB and the like), managed or pinned memory, or graph capture. Every fault class in the `cuda-to-rocm` index was swept against this file and none applies.

Not CUDA, unchanged: `segment_tree.h` and `utils.cpp`, the CPU extension that is always built.

## Risk list
1. **Silent skip is the main validation risk.** Every GPU segment-tree test skips with "TorchRL was not built with CUDA segment tree support" when the binding is missing (`test/rb/test_prioritized.py:104-105` and four more). A green run therefore proves nothing unless the extension really has `CudaSumSegmentTreeFp32`. The validator must assert this (see Test plan) and count skips.
2. **The `DEBUG=1` path passes `-G` to the device compiler** (`setup.py:270,307`). nvcc reads it as device debug info. clang/hipcc either rejects it or reads it as an unrelated option. When `torch.version.hip` is set, the porter should pass `-g` (or no flag) instead and verify a `DEBUG=1` build. A release build never takes this path.
3. **The stale-build detector** (`_check_and_clean_stale_builds`) keys on the torch, Python and CUDA versions and on `FORCE_CUDA`. On ROCm the CUDA versions are None, so switching ROCm torch builds would not clean a stale `.so`. Add `torch.version.hip` to the recorded build info.
4. **hipify output names.** In extension mode torch renames `cuda_segment_tree.cu` to `cuda_segment_tree.hip`, and may write hipified copies of `pybind.cpp` and of the `cuda_segment_tree*.h` headers it includes. If hipify rewrites an include and leaves a stale name, the build fails at compile time, not at run time. The porter checks the generated tree once.
5. **C++ standard.** `_get_cxx20_compatibility_flag()` applies to `cxx` only, and torch's BuildExtension passes the language standard to hipcc. Under ROCm torch, check that hipcc receives the same standard as the host compiler.
6. **Windows.** Upstream's Windows wheels are CPU-only and the `win32` branch uses MSVC flags (`/O2`, `/EHsc`, `/DWITH_CUDA`). A Windows ROCm torch (TheRock) build of a `CUDAExtension` meets the setuptools and `hipcc.exe` traps in `cuda-to-rocm` `windows.md`. Expect the ROCm gate there to need `ROCM_HOME` resolution from the TheRock SDK. Any Windows-only fix goes in `setup.py`, guarded to win32 plus HIP; no README-only workaround.
7. **Numerics.** `test_cuda_segment_tree_parity` compares GPU results with the CPU tree. Fp32 sums are folded in a fixed tree order on both, so results should match. If they drift, check FMA contraction first (`faults-numerics.md`), though no multiply is fused here.

## File-by-file change list (fork `moat-port`)
- `setup.py`:
  - Import `ROCM_HOME` from `torch.utils.cpp_extension`.
  - `_should_build_cuda()`: keep `FORCE_CUDA=1` first. When `torch.version.hip` is set, return `ROCM_HOME is not None` and skip the nvcc version check. The CUDA branch stays byte-identical.
  - `_get_build_cuda_version()`: on ROCm, record the HIP version (or leave the CUDA version None and record HIP separately in build info). Do not let the stale check misfire on CUDA.
  - `_check_and_clean_stale_builds()`: add `torch.version.hip` (`hip_version`) to the compared and recorded fields.
  - Debug flags: no `-G` for the device compiler when building against HIP (risk 2).
  - Keep the `WITH_CUDA` define name; it gates the bindings and is what upstream's code reads.
- `torchrl/csrc/*`: no change expected. Change a file only if a ROCm compile error proves it necessary, and keep the CUDA path identical.
- `test/test_setup.py`: unit tests for the gate with monkeypatched `torch.version.hip`, `torch.version.cuda` and `ROCM_HOME` (ROCm present builds, ROCm without `ROCM_HOME` does not, CUDA behaviour unchanged). This matches the existing monkeypatch style there (`test_detect_torch_extension_cxx20`).
- Docs: `docs/source/index.rst` (the `python setup.py develop` section) and `CONTRIBUTING.md` get one or two lines. With ROCm PyTorch, the GPU prioritized-replay kernels build automatically when ROCm is found, and `FORCE_CUDA=1` still forces them.
- `.github/`: only if jeff rules option (a) or (c) above.

## Build commands
Build in a job venv with a ROCm PyTorch (never the shared image env). Install `pybind11[global]`, `ninja` and `tensordict` (the matching nightly or source) there first.

```
cd <fork checkout>
python -c "import torch; print(torch.__version__, torch.version.hip)"
python setup.py clean
MAX_JOBS=<physical cores> python setup.py develop      # or a no-build-isolation editable install
python -c "import torchrl._torchrl as e; assert hasattr(e, 'CudaSumSegmentTreeFp32'); print(e.__file__)"
```
Linux: `PYTORCH_ROCM_ARCH=<gfx>` limits the offload targets to the host arch. Windows: TheRock ROCm torch wheel with the matching `[device-<gfx>]` extra; set `ROCM_HOME` to the SDK root if torch does not find it.

Also check that a build with `FORCE_CUDA=0` on a host with no ROCm still produces the CPU-only extension (`CppExtension`).

## Test plan
GPU tests that exercise the port (required on every platform; the `hasattr` assertion above must pass first):
```
pytest -v test/rb/test_prioritized.py -rs
pytest -v test/rb/test_storages.py -k "cuda" -rs
TORCHRL_SMOKE_REQUIRE_CUDA_EXT=1 pytest -v test/smoke_test.py
pytest -v test/test_setup.py -k "not install_no_deps"
```
Pass criteria:
- `test_cuda_segment_tree_parity`, `test_cuda_prioritized_replay_buffer_samples_on_cuda`, `test_tensordict_prioritized_replay_buffer_memmap_storage_cuda_sampler`, `test_cuda_prioritized_replay_buffer_weight_matches_cpu_formula` and `test_prioritized_memmap_cuda_sampler_after_multiprocess_writes` all PASS.
- None of them may skip with "not built with CUDA segment tree support"; the validator records the `-rs` skip list.

GPU regression set (the rest of the replay-buffer suite on device, run as recorded evidence):
```
pytest test/rb -m gpu -rs
pytest test/rb -x -q
```
Optional wider evidence (pure-Python torch paths, not the port itself): `pytest -m gpu test/modules/test_rnn.py test/objectives test/envs -rs`. Record failures there with their cause. Do not count them against the port unless they touch the extension.

Benchmark (evidence only, no gate): `pytest benchmarks/test_replaybuffer_benchmark.py -k "cuda or Cuda" --benchmark-only`. It compares the GPU tree with the CPU tree; record the ratio on each platform.

Non-GPU regression set: `pytest test/rb -q` on CPU (`CUDA_VISIBLE_DEVICES=` / `HIP_VISIBLE_DEVICES=`) and `test/test_setup.py` must not regress, and the CPU segment tree tests must still pass.

CUDA no-regression: the `setup.py` change must leave a CUDA torch build unchanged. Run the `test_setup.py` gate tests (no GPU needed) and an nvcc compile of the extension on the fleet host with nvcc. A ROCm-only torch on that host blocks the torch-extension CUDA compile; if so, record `cuda-not-validated: <error>`.

Platforms: the wave64, wave32 and windows gates, as configured. The kernel has no wavefront-dependent code, so wave32 and wave64 are expected to behave the same.

## Open questions
- The ROCm wheel lane (top of this file) is jeff's ruling.
- Whether tensordict nightly wheels install cleanly next to a ROCm torch nightly on each platform. tensordict is pure Python, so this is expected but not yet checked.
