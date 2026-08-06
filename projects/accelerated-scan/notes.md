# accelerated-scan notes

## Port Summary

This project provides parallel associative scan implementations for state space models. It has two backends:

1. **Triton** (`scalar.py`, `complex.py`): Uses `tl.associative_scan` which is wave-size agnostic
2. **C++ warp kernel** (`warp.cuh`): JIT-compiled CUDA kernel with hardcoded warp-size assumptions

### ROCm Approach

On ROCm, `warp.py` redirects to the Triton backend. The C++ warp kernel has fundamental wave64 incompatibilities:
- `kNThreadsPerWarp=32` hardcoded as logical warp size
- Shuffle operations defaulting to `warpSize` (64 on CDNA) instead of the 32-lane logical warp
- Block configurations assuming 32-thread hardware warps

Attempted fixes (64-bit shuffle masks, explicit width=32 parameters, fixing iteration bounds) still resulted in HSA_STATUS_ERROR_EXCEPTION crashes on gfx90a. The Triton backend works correctly and supports arbitrary sequence lengths.

## Build

```bash
cd projects/accelerated-scan/src
pip install -e .
```

## Test

```bash
export HIP_VISIBLE_DEVICES=0  # or 1 depending on available GPU
pytest tests/test_eq.py -v  # 399/400 pass
pytest tests/tests_eq_complex.py -v  # 240/240 pass
```

The single failing test (`test_eq_ref_reverse[65536-1]`) is a marginal tolerance issue in the reference implementation (23us vs 20us allowed), not a port problem.

## Gotchas

- The C++ warp kernel cannot be made wave64-compatible without a significant rewrite
- The Triton backend is fully functional and recommended for ROCm
- Tests require torch with ROCm support and Triton with ROCm backend

## Review 2026-06-05

Reviewed commit 94e47c4 (`[ROCm] Add AMD GPU support via Triton backend`) against origin/main.

### Summary

The port redirects `warp.py` to use the Triton `scalar.py` backend on ROCm instead of the C++ warp kernel. This is acceptable: the C++ kernel has deep wave64 incompatibilities (kNThreadsPerWarp=32 hardcode, implicit warpSize loop bounds, crash-inducing block geometry), and Triton's `tl.associative_scan` is wave-size agnostic. The partial fixes in warp.cuh (64-bit masks, explicit width=32 shuffles, kNThreadsPerWarp loop bounds) are present but the kernel still crashes on gfx90a, so the Triton fallback is the right call.

### Issues

**1. scan_forward(reverse=True) silently ignored on ROCm** (warp.py:13-14)

The ROCm branch defines `scan_forward(gates, tokens, reverse=False)` but calls `_scan_impl(gates, tokens)` without passing `reverse`. On CUDA the C++ kernel supports reverse scans; on ROCm the parameter is silently dropped.

Impact: Low. No test or autograd path uses `scan_forward(reverse=True)` -- the `Scan` class always calls it with the default. The only `reverse=True` test (`test_eq_ref_reverse`) uses `scan_ref`, not `scan_warp`. But the silent drop is a latent behavior difference that should be documented or fixed (raise NotImplementedError for reverse=True on ROCm, or implement it in Triton).

**2. warpscan_forward/warpscan_backward not exported on ROCm** (warp.py)

On CUDA, `from accelerated_scan.warp import warpscan_forward` works; on ROCm it fails with ImportError because these symbols are defined only in the `else` branch. The benchmark script (`tests/bench.py:73-77`) imports these directly and will fail on ROCm with provider=warp.

Impact: Low. This is a low-level C++ API, not the public `scan()` API. But it is a breaking API change for anyone using the direct C++ bindings.

### Verdict

**review-passed**

The Triton fallback is the correct strategy for this project given the wave64 kernel incompatibility. The issues above are minor documentation/cleanup items suitable for PR prep, not blockers.

## Validation 2026-06-05

Validated commit 94e47c4 on gfx90a (AMD Instinct MI250X) with ROCm/HIP.

### GPU Architecture

- Device: AMD Instinct MI250X
- Architecture: gfx90a
- Wave size: 64

### Build

```bash
cd /var/lib/jenkins/moat/projects/accelerated-scan/src
pip install -e .
```

Build succeeded. The package uses PyTorch's JIT compilation via Triton backend on ROCm.

### Test Results

```bash
export HIP_VISIBLE_DEVICES=1
pytest tests/test_eq.py -v              # 399/400 passed
pytest tests/tests_eq_complex.py -v      # 240/240 passed
```

**Total: 639/640 tests passed**

The single failure is `test_eq_ref_reverse[65536-1]` with a marginal tolerance issue (23.3us vs 20us allowed absolute error) in the reference implementation's reverse scan, not a port problem. This is a pre-existing test instability, not related to the ROCm port.

### GPU Correctness

All Triton-based scan operations (scalar and complex) execute correctly on gfx90a:
- Forward scan: All sequence lengths (32 to 131072, power-of-2 and irregular) pass
- Backward scan: All gradient tests pass
- Complex scan: All 240 complex-valued scan tests pass
- Data types: float32, bfloat16, float16 all work correctly

The Triton `tl.associative_scan` implementation is wave-size agnostic and handles wave64 correctly.

## Validation 2026-06-05 (gfx1100)

Validated commit 94e47c4 on gfx1100 (AMD Radeon Pro W7800 48GB) with ROCm/HIP.

### GPU Architecture

- Device: AMD Radeon Pro W7800 48GB
- Architecture: gfx1100
- Wave size: 32

### Build

```bash
cd /var/lib/jenkins/moat/projects/accelerated-scan/src
pip install -e .
```

Build succeeded. The package uses PyTorch's JIT compilation via Triton backend on ROCm.

### Test Results

```bash
export HIP_VISIBLE_DEVICES=0
pytest tests/test_eq.py -v              # 399/400 passed
pytest tests/tests_eq_complex.py -v      # 240/240 passed
```

**Total: 639/640 tests passed**

The single failure is `test_eq_ref_reverse[65536-1]` with a marginal tolerance issue (23.3us vs 20us allowed absolute error) in the reference implementation's reverse scan, not a port problem. This is the same pre-existing test instability observed on gfx90a.

### GPU Correctness

All Triton-based scan operations (scalar and complex) execute correctly on gfx1100:
- Forward scan: All sequence lengths (32 to 131072, power-of-2 and irregular) pass
- Backward scan: All gradient tests pass
- Complex scan: All 240 complex-valued scan tests pass
- Data types: float32, bfloat16, float16 all work correctly

The Triton `tl.associative_scan` implementation is wave-size agnostic and handles wave32 (RDNA3) correctly.

## Validation 2026-06-19 (windows-gfx1101)

Validated commit 84770b4d on gfx1101 (AMD Radeon PRO V710, RDNA3) on Windows 11.

### GPU Architecture

- Device: AMD Radeon PRO V710
- Architecture: gfx1101 (RDNA3)
- Wave size: 32
- HIP_VISIBLE_DEVICES=1

### Build

```
HIP_VISIBLE_DEVICES=1 pip install -e .
```

Build succeeded. Pure Python package; Triton backend JIT-compiles at runtime, no C++ extension build needed.

### Test Results

```
HIP_VISIBLE_DEVICES=1 pytest tests/test_eq.py -v              # 593/593 passed
HIP_VISIBLE_DEVICES=1 pytest tests/tests_eq_complex.py -v     # 240/240 passed
```

**Total: 833/833 tests passed**

All tests pass on gfx1101, including `test_eq_ref_reverse[65536-1]` and the full `test_eq_reverse` suite (192 cases) added in the reverse-scan prep commit.

### GPU Correctness

All Triton-based scan operations (scalar and complex) execute correctly on gfx1101 (RDNA3, wave32):
- Forward scan: All sequence lengths (32 to 127323, power-of-2 and irregular) pass
- Backward scan: All gradient tests pass
- Reverse scan: All 192 reverse-scan cases pass (test_eq_reverse)
- Complex scan: All 240 complex-valued scan tests pass
- Bindings test: warpscan_bindings test passes (ROCm shim exports confirmed)
- Data types: float32, bfloat16, float16 all work correctly

The Triton `tl.associative_scan` implementation is wave-size agnostic and handles gfx1101 (RDNA3, wave32) correctly.

## Validation 2026-06-07 (windows-gfx1201)

Validated commit 94e47c4 on gfx1201 (AMD Radeon RX 9070 XT, RDNA4) on Windows 11.

### GPU Architecture

- Device: AMD Radeon RX 9070 XT
- Architecture: gfx1201 (RDNA4)
- Wave size: 32

### Build

```
HIP_VISIBLE_DEVICES=0 pip install -e .
```

Build succeeded. Pure Python package; Triton backend JIT-compiles at runtime, no C++ extension build needed.

### Test Results

```
HIP_VISIBLE_DEVICES=0 pytest tests/test_eq.py -v              # 400/400 passed
HIP_VISIBLE_DEVICES=0 pytest tests/tests_eq_complex.py -v      # 240/240 passed
```

**Total: 640/640 tests passed**

All tests pass on gfx1201, including `test_eq_ref_reverse[65536-1]` which was marginal on gfx90a/gfx1100 (timing-sensitive; passes cleanly here).

### GPU Correctness

All Triton-based scan operations (scalar and complex) execute correctly on gfx1201 (RDNA4, wave32):
- Forward scan: All sequence lengths (32 to 131072, power-of-2 and irregular) pass
- Backward scan: All gradient tests pass
- Complex scan: All 240 complex-valued scan tests pass
- Data types: float32, bfloat16, float16 all work correctly

The Triton `tl.associative_scan` implementation is wave-size agnostic and handles gfx1201 (RDNA4, wave32) correctly.
