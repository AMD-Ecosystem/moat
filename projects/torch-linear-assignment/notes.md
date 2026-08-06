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
