# torch-linear-assignment notes

## Port summary

Strategy B (pytorch extension, torch hipify) -- zero-source-edit port confirmed.

PyTorch's build-time hipify translates the CUDA sources automatically when building against ROCm PyTorch. No manual source changes required.

## Build (linux-gfx90a)

```bash
export HIP_VISIBLE_DEVICES=3
pip install --no-build-isolation -e /path/to/src
```

Verified `has_cuda: True` after build.

## Validation (linux-gfx90a)

GPU tests pass:
- `test_simple`: PASSED (deterministic assignment output matches expected)
- Batch test (100 x 20 x 40): PASSED (valid assignments, no duplicates)

The `test_cuda_equal_to_cpu` and `test_compare_to_scipy` tests fail due to a torch/numpy version incompatibility in the base environment (torch compiled with numpy 1.x, environment has numpy 2.x). This is an environment issue, not a port issue -- the GPU code path works correctly.

## Gotchas

1. The `SMPCores()` function returns CUDA cores per SM based on NVIDIA arch major/minor. On AMD it falls through to default 128. This affects block size selection but not correctness -- the algorithm is inherently sequential per batch element (Hungarian assignment).

2. Torch's hipify generates a `.hip` file at build time (`torch_linear_assignment_hip_kernel.hip`). This is a build artifact, not checked in.

## Review 2026-06-05

Zero-source-edit Strategy B port: fork branch `moat-port` (SHA 9c842e3) is byte-identical to upstream `main`. Torch's build-time hipify handles all CUDA-to-HIP translation automatically.

Fault-class analysis (upstream code):
- No warp intrinsics or hardcoded 32-lane assumptions (algorithm is per-thread sequential Hungarian)
- No textures, no RAII handle concerns (uses PyTorch tensor API)
- No OOB neighbor reads (element indexing is bounded by loop)
- `SMPCores()` defaults to 128 on unknown archs (affects block size selection, not correctness)

GPU tests verified:
- Simple 2x2 assignment: PASS
- Batch 100x20x40: PASS (unique assignments per batch, valid task indices)

No problems found. Ready for validator.

## Validation 2026-06-05 (linux-gfx90a)

Arch: gfx90a (MI250X, HIP_VISIBLE_DEVICES=3)
Head SHA: 9c842e3

Build command:
```bash
export HIP_VISIBLE_DEVICES=3
cd /var/lib/jenkins/moat/projects/torch-linear-assignment/src
pip install --no-build-isolation -e .
```

Build artifacts:
- HIP kernel: `src/torch_linear_assignment_hip_kernel.hip`
- Compiled backend: `torch_linear_assignment/_backend.cpython-312-x86_64-linux-gnu.so`

GPU validation (pytest tests/ + manual comprehensive test):
- `test_simple`: PASSED (4x3 deterministic assignment, exact match)
- Manual test 1 (simple deterministic 4x3): PASSED (result matches expected [0,2,-1,1])
- Manual test 2 (large batch 100x20x40): PASSED (valid indices 0-39 or -1, no duplicates per batch)
- Manual test 3 (empty batch 0x10x10): PASSED (correct shape handling)

Note: `test_cuda_equal_to_cpu` and `test_compare_to_scipy` fail with "RuntimeError: Numpy is not available" due to PyTorch compiled with numpy 1.x running on numpy 2.2.6 environment. This is an environment incompatibility, not a port issue. The GPU code path executes correctly as verified by the passing tests.

Validation result: PASS - GPU kernels execute correctly on gfx90a, assignments are valid and deterministic.

## Validation 2026-06-05 (linux-gfx1100)

Arch: gfx1100 (AMD Radeon Pro W7800 48GB, HIP_VISIBLE_DEVICES=3)
Head SHA: 9c842e3

Build command:
```bash
export HIP_VISIBLE_DEVICES=3
cd /var/lib/jenkins/moat/projects/torch-linear-assignment/src
pip install --no-build-isolation -e .
```

Build artifacts:
- HIP kernel: `src/torch_linear_assignment_hip_kernel.hip`
- Compiled backend: `torch_linear_assignment/_backend.cpython-312-x86_64-linux-gnu.so`

GPU validation (pytest + manual verification):
- `test_simple`: PASSED (4x3 assignment test)
- Manual verification: GPU result [0, 2, 1, -1] produces optimal cost 6.0 (verified via brute-force enumeration)
- Large batch test (100x20x40): PASSED (valid assignments, no duplicates per batch)
- Determinism test: PASSED (identical results across runs)

Note: `test_cuda_equal_to_cpu` and `test_compare_to_scipy` fail with "RuntimeError: Numpy is not available" due to PyTorch/numpy version incompatibility (same as gfx90a). The GPU code path executes correctly as verified by the passing tests and manual validation.

Validation result: PASS - GPU kernels execute correctly on gfx1100, assignments are valid, optimal, and deterministic.

## Validation 2026-06-08 (windows-gfx1201)

Arch: gfx1201 (AMD Radeon RX 9070 XT, RDNA4, wave32), HIP_VISIBLE_DEVICES=0 (only GPU present after gfx1101 V710 offline)
Head SHA: 4024f61 (adds Windows /ALTERNATENAME linker fix on top of 9c842e3)
ROCm: 7.14.0a20260604 (TheRock nightly), Python 3.12.10 (MSVC), numpy 2.4.6, scipy 1.17.1

### Windows delta (new commit 4024f61)

One Windows-specific linker fix required (not needed on Linux):

**`c10::ValueError` LNK2001**: `c10.dll` does not export the inherited constructor
`c10::ValueError(SourceLocation, string)` (MSVC does not re-export inherited
constructors even for `C10_API` classes). Headers included via `<torch/extension.h>`
(e.g. `ATen/TensorIndexing.h`) trigger `TORCH_CHECK_VALUE` which generates a
`__declspec(dllimport)` reference to that constructor, causing LNK2001. Fix:
`/ALTERNATENAME` linker directive in `setup.py` (Windows-only, `sys.platform=="win32"`)
redirects the dllimport thunk to `c10::Error(SourceLocation, string)`, which IS
exported. `ValueError IS-A Error` with no additional data members; semantically
identical constructors.

### Build command (gfx1201)

```python
# Environment: MSVC link.exe must precede Git's /usr/bin/link.exe on PATH
# Requires: ROCM_HOME, HIP_DEVICE_LIB_PATH, DISTUTILS_USE_SDK
# See /tmp/build_tla_fixed.py for the full wrapper used

HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1201 \
  ROCM_HOME=<venv>/Lib/site-packages/_rocm_sdk_devel \
  HIP_DEVICE_LIB_PATH=<venv>/Lib/site-packages/_rocm_sdk_devel/lib/llvm/amdgcn/bitcode \
  DISTUTILS_USE_SDK=1 \
  pip install --no-build-isolation -e . --no-deps
```

Build result: PASS (~29s, exit 0). gfx1201 code object confirmed present in .pyd (`hipFatB` section, `gfx1201` string found).

### Test command

```
HIP_VISIBLE_DEVICES=0 python -m pytest tests/ -v --tb=short
```

### Test results: 3/3 PASS

```
tests/test_assignment.py::TestAssignment::test_cuda_equal_to_cpu PASSED
tests/test_assignment.py::TestAssignment::test_simple PASSED
tests/test_to_indices.py::TestAssignmentToIndices::test_compare_to_scipy PASSED
3 passed in 1.45s
```

Note: All 3 tests pass on Windows (including `test_cuda_equal_to_cpu` and `test_compare_to_scipy` which fail on Linux due to numpy 1.x/2.x incompatibility). Windows venv has numpy 2.4.6 which is compatible.

Assignments deterministic and match expected values: simple 4x3 assignment [0,2,-1,1] exact match; CPU/GPU agreement verified; scipy reference comparison verified.

Validation result: PASS - GPU kernels execute correctly on gfx1201, assignments valid and deterministic.

## Validation 2026-06-20 (windows-gfx1101)

Arch: gfx1101 (AMD Radeon PRO V710, RDNA3, wave32), HIP_VISIBLE_DEVICES=1
Head SHA: c65c0b823fe81f022e060291a47374ac8483cfc8
ROCm: 7.14.0a20260604 (TheRock nightly), Python 3.12.10 (MSVC), numpy/scipy present

### Build command (gfx1101)

MSVC link.exe must precede Git's /usr/bin/link.exe on PATH. Exported env vars for the session:

```
MSVC_LINK="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
export PATH="$MSVC_LINK:$PATH"
export DISTUTILS_USE_SDK=1
export HIP_VISIBLE_DEVICES=1
export PYTORCH_ROCM_ARCH=gfx1101
export ROCM_HOME=<venv>/Lib/site-packages/_rocm_sdk_devel
export HIP_DEVICE_LIB_PATH=<venv>/Lib/site-packages/_rocm_sdk_devel/lib/llvm/amdgcn/bitcode

python -m pip install --no-build-isolation -e <src> --no-deps
```

Build result: PASS. gfx1101 code object confirmed in .pyd (`hipFatB` section, `gfx1101` string found).
The .hip compile stage succeeded (torch BuildExtension hipified the .cu -> .hip automatically).
No `.hip` extension issue hit here (torch 2.9.1 BuildExtension compiled .hip correctly on this build).

### Test command

```
HIP_VISIBLE_DEVICES=1 python -m pytest tests/ -v --tb=short
```

### Test results: 3/3 PASS

```
tests/test_assignment.py::TestAssignment::test_cuda_equal_to_cpu PASSED
tests/test_assignment.py::TestAssignment::test_simple PASSED
tests/test_to_indices.py::TestAssignmentToIndices::test_compare_to_scipy PASSED
3 passed in 1.40s
```

GPU confirmed: `torch.cuda.get_device_name(0)` = "AMD Radeon PRO V710" (gfx1101); `has_cuda()` = True (HIP backend active, not CPU fallback).

All 3 tests pass on Windows gfx1101 (same as gfx1201). Assignments deterministic and match expected values: CPU/GPU agreement verified; scipy reference comparison verified.

Validation result: PASS - GPU kernels execute correctly on gfx1101, assignments valid and deterministic.

## Fix round 2026-11-20: rework PR #31 to the Triton path (porter, linux-gfx90a)

### What the maintainer asked

On PR #31 jeff asked (2026-11-04) which line AMD support should target: the legacy compiled extension (now behind `TLA_BUILD_LEGACY_CUDA=1` since 0.1.0) or the Triton solver that is the 0.1.0 default. ivan-chai answered (2026-11-10): "I think, Triton is preferable to keep the code as simple as possible." No line-level review comments exist (`gh api repos/ivan-chai/torch-linear-assignment/pulls/31/comments` returns `[]`). The PR was `CONFLICTING` / `DIRTY` against upstream `main` (README.md, setup.py).

### Round setup

- `moatlib.py fix-branch torch-linear-assignment`: re-homed the trunk-resident folder onto `port/torch-linear-assignment`, staged `moat-fix-31` with base `c65c0b823fe8` (= the PR head GitHub showed, so no HEAD-MOVED).
- `set-state ... linux-gfx90a porting` took the lock cleanly (no takeover warning for this project).
- Fork clone at `projects/torch-linear-assignment/src`, `upstream` remote added, local `main` = `upstream/main` (42202a6) so `jargon.py --port` scans only our commits.

### Commits on moat-fix-31 (on top of published c65c0b8)

1. `c49c034` Merge upstream main (0.1.0 Triton solver). `git merge upstream/main`; conflicts in README.md and setup.py resolved with `git checkout --theirs` (upstream's favor). This drops our legacy-extension changes (Windows `/ALTERNATENAME` link directive, README hipify note). The merged tree is byte-identical to upstream `main` 42202a6 (`git diff upstream/main` empty).
2. `9470225` [ROCm] Use the Triton solver on AMD GPUs. Pure Python and docs, the kernel is untouched:
   - `torch_linear_assignment/assignment.py` `_cuda_uses_triton`: accept a ROCm torch (`torch.version.hip` set). The `torch.version.cuda` requirement and the capability >= 8.0 floor still apply when `torch.version.hip is None`, so the NVIDIA decision is logically identical to before. On ROCm `get_device_capability` reports the gfx version ((9, 0) for gfx90a), not an NVIDIA CC, so the floor is not applied there.
   - `tests/test_triton.py`: same eligibility change in `_compiled_triton_skip_reason` and `_compiled_device_params`, otherwise all 26 compiled-Triton tests skip on ROCm.
   - `torch_linear_assignment/_triton.py`: module docstring "Linux NVIDIA systems" -> "Linux NVIDIA and AMD systems".
   - README: AMD GPUs in the backend policy sentence, a dispatch-table row, and the validation-targets sentence (example: MI250X). Table re-aligned with the repo's own `mdformat --number --wrap=no` (mdformat 0.7.22 + mdformat-gfm, installed into FLEET_VENV only).

### Environment (linux-gfx90a)

AMD Instinct MI250X (gfx90a:sramecc+:xnack-), HIP_VISIBLE_DEVICES=1 (dispatcher pin; all 4 GPUs idle by `amd-smi metric --usage`). Image Python 3.11.17, torch 2.16.0a0+git6c3af02 (HIP 7.16.26385), triton 3.9.0+gitfc8d31a5 (dist name `triton`, has `backends/amd`), scipy 1.10.1, numpy 1.26.4. Triton cache: job-local `TRITON_CACHE_DIR`. Compiled kernel metadata confirms `"backend": "hip", "arch": "gfx90a", "warp_size": 64`.

The package is pure source in 0.1.0 (setup.py builds nothing without `TLA_BUILD_LEGACY_CUDA=1`), so tests ran from the checkout with `PYTHONPATH`; no install into the shared image Python. The fleet guard refuses `pip` (even `--dry-run`) against the image Python and against a job-local venv outside FLEET_VENV, so the documented `pip install -e . --no-build-isolation` route was NOT exercised on this host. That route needs a venv that has torch, which FLEET_VENV does not.

### Test command

```bash
export HIP_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=<job>/triton-cache
ulimit -c 0
cd projects/torch-linear-assignment/src
PYTHONPATH=$PWD python -m pytest tests/ -v -p no:cacheprovider -p no:rerunfailures
```

### Results

| tree | passed | skipped | failed | notes |
|---|---|---|---|---|
| c49c034 (merge only = upstream main) | 52 | 26 | 0 | every `test_triton.py` case skips ("requires a CUDA-enabled PyTorch build"); public API on GPU warns and falls back to SciPy |
| 9470225 (staging tip) | 78 | 0 | 0 | 26 test_assignment, 25 test_benchmark, 1 test_to_indices, 26 test_triton; no RuntimeWarning (fallback warning gone); 12.4 s |

Against the last record for this platform (c65c0b8, legacy extension, 2026-06-05: `test_simple` pass, 2 failures from a numpy 1.x/2.x mismatch in that environment): the suite is now upstream's 0.1.0 suite (78 tests vs 3), so the counts are not comparable; this environment has numpy 1.26.4 and all SciPy-reference tests pass. Spot check: `_cuda_uses_triton(cuda tensor)` is True and a 4x50x80 batch equals `batch_linear_assignment_cpu` exactly.

### Benchmark (staging tip 9470225)

```bash
PYTHONPATH=$PWD env -u CUDA_LAUNCH_BLOCKING python tests/benchmark.py --backends scipy,public_cuda --output <agent_space>/tla-bench.jsonl
```

Exit 0, 60 records: 30 `scipy` + 30 `public_cuda` with `implementation: triton`; every record `parity: {"exact": true, "oracle": "scipy_promoted"}` (exact in all 30 cases). Batch-208 float32 warm medians (transpose / square / direct = 300x100 / 300x300 / 300x600): Triton 1.4 / 12.3 / 4.3 ms, SciPy/Triton 39.7x / 48.8x / 74.7x (SciPy timing is this host's CPU; not a cross-GPU claim). Batch 624: 86x-142x. Batch 1 is slower than SciPy (0.3x-0.5x), as upstream's own NVIDIA numbers also show for small batches. First calls include Triton JIT (0.3-1.4 s cold per shape).

### CUDA path

No CUDA source changed: the legacy `.cu`/`.cpp` are upstream's (merge tree == upstream main), and the NVIDIA dispatch condition is logically unchanged (hip None -> same `cuda is None` and `capability < (8, 0)` rejections). No nvcc compile is relevant to a pure-Python delta; `cuda-not-validated: no NVIDIA GPU to run the Triton path; delta is Python dispatch logic only`.

### Open points for jeff (not decided by this round)

- **Windows gate.** Upstream's Triton dispatch is Linux-only (`sys.platform != "linux"` returns False) and the package declares Triton only on Linux, for NVIDIA as well. On Windows, AMD and NVIDIA GPU inputs both take the SciPy CPU fallback by upstream design, so a Windows validation can only show the fallback. This round did not touch that; whether to extend Triton to Windows (triton-windows) is beyond what the maintainer asked and would add code. The `windows` gate needs jeff's call (fallback-only validation, or a waiver suggestion).
- **wave32 untested here.** The kernel runs with `num_warps=1` (one 64-lane wavefront on gfx90a, 32 on RDNA). A wave32 host must validate before fix-ready.
- **ROCm torch wheels and the `triton` requirement.** `requirements.txt` has `triton; platform_system == "Linux"`. Here the installed Triton's distribution name is `triton`, so it is satisfied. Official ROCm torch wheels ship Triton as `pytorch-triton-rocm`; a plain `pip install torch-linear-assignment` there may pull PyPI `triton` over it. Not testable on this host (pip blocked); not changed, since the same marker applies to NVIDIA and the maintainer asked for simplicity. Worth a check on a host with a pip-installed ROCm torch.
