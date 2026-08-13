# Title: Add setup.py to build the _C extension (CUDA and ROCm)

## Compare
https://github.com/Luo-Yihao/FaithC/compare/main...jeffdaily:FaithC:moat-port

## Body
The tree ships the `_C` sources (`bindings.cpp`, `kernels.cu`) and `ops.py` hard-imports them (`from . import _C`), but the only build config is a `pyproject.toml` that declares a pure-Python package with no extension module. A clean source install (`pip install -e .`, or pixi) therefore never compiles `_C`, so `from . import _C` fails. This adds the missing build wiring.

`setup.py` builds `_C` with PyTorch's `CUDAExtension` and `BuildExtension`. On a CUDA PyTorch it compiles the original CUDA sources unchanged; on a ROCm PyTorch `BuildExtension` hipifies the same sources automatically, so one source tree builds for both backends. This complements the ROCm kernel/runtime support already in `main`: that made the kernels HIP-clean, and this makes the extension actually build (on CUDA and ROCm alike).

The kernels use only `atomicAdd`, `__syncthreads`, dynamic shared memory and float math, with no warp-level intrinsics, so they are wavefront-size agnostic and need no per-architecture changes.

Two further changes make the build correct on Windows (both are no-ops on Linux and CUDA):

- The int64 index/candidate buffers were typed `long`, which is 32-bit on Windows (LLP64) while `torch::kInt64` tensors are 64-bit; the kernel signatures, `data_ptr<>()` calls and casts now use `int64_t`, which is correct on every platform and identical to `long` on Linux LP64.
- `setup.py` adds a Windows-only `/ALTERNATENAME` link directive. `c10.dll`, built with clang-cl, does not export the `c10::ValueError(SourceLocation, std::string)` constructor inherited via `using Error::Error;`, so headers pulled in through `<torch/extension.h>` that expand `TORCH_CHECK_VALUE` fail to link (`LNK2001`); the directive aliases the missing import thunk to the exported `c10::Error(SourceLocation, std::string)` constructor. The same root cause was fixed upstream in pytorch/pytorch#175340 (explicit exported constructors for the affected `c10::Error` subclasses); this alias keeps the extension building on PyTorch releases from before that fix.

### Building

```bash
# CUDA (unchanged)
pip install -e . --no-build-isolation

# ROCm (set the arch(es) for your GPU)
PYTORCH_ROCM_ARCH=gfx90a pip install -e . --no-build-isolation
```

The README's Manual Setup section documents the ROCm path alongside the existing CUDA instructions. `.gitignore` is extended to cover the hipify build artifacts (`*.hip`, `*.prehip`, `*.so.*`).

### Validation

Built on an AMD Instinct MI250X (gfx90a, ROCm 7.2): a clean `setup.py build_ext` hipifies, compiles and links `_C` (the `.so` carries a native gfx90a code object). A synthetic-tensor harness drives all four `_C` bindings on the GPU and compares against a pure-torch CPU reference (the `atomicAdd` output-slotting kernels as order-independent `(a, t)` pair sets, the deterministic kernels exactly and for rerun stability); all checks pass, with Moller-Trumbore dot-product drift of 3.5e-7 within the kernels' eps thresholds. A `gfx90a;gfx1100` multi-architecture binary also builds with both code objects present.

The end-to-end demo additionally depends on `atom3d` and `torch_scatter` on the GPU; bringing those up on ROCm is left as a follow-up, so this change covers the `_C` kernel layer those higher-level paths call into.
