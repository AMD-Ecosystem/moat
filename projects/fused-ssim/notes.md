# fused-ssim notes

## Summary

`rahul-goel/fused-ssim` is a PyTorch CUDA extension (fused differentiable SSIM
for Gaussian-splatting). Strategy B: torch hipifies the extension's `.cu` at
build time. Ported and GPU-validated on linux-gfx90a (MI250X, CDNA2, wave64,
ROCm 7.2.1) with ZERO source changes. The kernels have no warp-level primitives,
so there is no wave64 fault class to fix.

## Existing AMD support (assessment)

Already AMD-aware, but only at the build-config level:
- `setup.py:29-32` `if torch.version.hip:` branch (adds `-ffast-math`, skips
  `--maxrregcount`/`-gencode`). The only AMD commit, `88169c5` (#29, amd.com
  author), changed `setup.py` only.
- README advertises "2D (CUDA, Metal, ROCm) and 3D (CUDA only)".

NOT skipped as already-supported, because the 3D path post-dates that
enablement and had never been GPU-checked on AMD: `ext.cpp:5,15` gate the 3D
bindings on `FUSED_SSIM_CUDA`, which `setup.py` defines for HIP too, so
`fusedssim3d` IS compiled and exported on ROCm despite the "CUDA only" README
note. `fused_ssim/__init__.py:6` imports it unconditionally on any CUDA-or-HIP
device. The MOAT value here was to build + GPU-validate the current (post-3D-
rework) code on a modern ROCm, including that 3D path.

## ROCm torch env

- torch `2.13.0a0+gitb5e90ff`, `torch.version.hip 7.2.53211`, locally built.
- Must run from a NON-source dir; importing torch from `/var/lib/jenkins/pytorch`
  shadows `torch/_C` and fails. All runs below are from `/tmp`.
- conda env `py_3.12`; no project `.venv`.

## CUDA surface / fault classes

- Both `ssim.cu` (2D) and `ssim3d.cu` (3D) are separable 11x11 Gaussian
  convolutions with shared-memory tiling; block `(16,16)`=256 threads. The
  per-pixel reduction goes entirely through `__shared__` arrays + `block.sync()`;
  the 3D Z-axis pass uses a per-thread register ring buffer (no cross-thread
  exchange).
- NO warp primitives: grep `__shfl|__ballot|__activemask|__any|__all|__reduce|
  __popc|tiled_partition` over both `.cu` -> nothing. The two "same warp"
  strings (`ssim.cu:167`, `ssim3d.cu:252`) are misleading comments on plain
  `ly + BLOCK_Y` shared-memory indexing.
- The only literal 32 (`ssim.cu:328-335`, 2D backward load:
  `warp_id=tid/32; lane_id=tid%32; num_warps=(T+31)/32; col+=32`) is a benign
  loop-tiling stride distributing 256 threads over the 26x26 shared tile. It
  does not read `warpSize` and does no cross-lane op, so it is correct unchanged
  on wave64 (validated by the backward-gradient checks passing).
- No texture, atomics RMW, cuBLAS/cuFFT/cuRAND, rule-of-five handles, or layered
  arrays. => no Fault-class fixes needed.

## Build (Strategy B, zero source changes)

From the fork clone `projects/fused-ssim/src`:

```
HIP_VISIBLE_DEVICES=1 python -m pip install -e . --no-build-isolation -v
```

Torch hipify rewrote the sources to `ssim.hip` / `ssim3d.hip` (gitignored)
cleanly: `cooperative_groups` -> `<hip/hip_cooperative_groups.h>`,
`c10/cuda/CUDAGuard.h` -> `c10/hip/HIPGuard.h`, `__constant__` preserved,
`at::cuda::OptionalCUDAGuard` kept (works on HIP). Linked
`-lamdhip64 -lc10_hip -ltorch_hip`. setup.py HIP branch reported NVCC args
`['-O3','-DFUSED_SSIM_CUDA','-ffast-math']`. Module exports all four symbols
(`fusedssim`, `fusedssim_backward`, `fusedssim3d`, `fusedssim_backward3d`);
`is_3D_supported == True`.

## Validation (GPU 1, gfx90a) -- ALL PASS

Reference: `pytorch_msssim` 1.0.0 (pip) and a pure-PyTorch conv-based SSIM.

1. Repo `tests/test.py` (2D, 100 iters, B=5 CH=5 H=1080 W=1920): forward
   `torch.isclose` and full-tensor gradient `.all()` vs reference SSIM AND
   pytorch_msssim. Exit 0 (asserts inside the loop). fused fwd 3.67ms vs ref
   197ms.

   ```
   cd projects/fused-ssim/src/tests && HIP_VISIBLE_DEVICES=1 python test.py
   ```

2. Repo `tests/test_3D.py` (3D, 10 iters, B=2 CH=1 D=H=W=96, rtol=1e-6
   atol=1e-8): forward + gradient `torch.allclose` vs conv3d reference AND
   pytorch_msssim(spatial_dims=3). Exit 0.

   ```
   cd projects/fused-ssim/src/tests && HIP_VISIBLE_DEVICES=1 python test_3D.py
   ```

3. Independent harness `agent_space/fused_ssim_validate.py` over configs the
   repo tests do not cover (2D CH=1/3/16, B=1..4, several sizes; 3D three
   shapes) + bitwise determinism:

   ```
   HIP_VISIBLE_DEVICES=1 python agent_space/fused_ssim_validate.py
   ```

   - 2D fwd vs ref/pm: max|diff| ~1e-8; 2D grad: ~1e-10..1e-12.
   - 3D fwd vs pm: max|diff| ~5e-9; 3D grad: ~1e-11.
   - Determinism: 2D and 3D fwd+grad bitwise identical across two runs.
   - All measured diffs are 4-5 orders tighter than the asserted tolerances.

## Deliverable

Zero kernel/source edits. The port is "build against ROCm torch + validate";
no fork commit carries a code change for gfx90a. No GitHub Actions added; no
README/gen_readme changes.

## Validation 2026-05-30 (gfx1100, ROCm 7.2.1)

**GPU:** AMD Radeon Pro W7800 48GB, gfx1100 (RDNA3, wave32). ROCm 7.2.1. torch 2.13.0a0+gitb5e90ff, hip 7.2.53211. Validated sha: `666987fb33dcc378d98e5a649e13f6c6d37da620`.

**Fork interaction:** none. Follower reuses the gfx90a branch unchanged; no source edit, no fork push, no CI workflow.

### Build

```
bash -c "cd /var/lib/jenkins/moat/projects/fused-ssim/src && HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1100 python -m pip install -e . --no-build-isolation -v"
```

Compile time: 37.5s. Torch hipify rewrote `ssim.cu` -> `ssim.hip`, `ssim3d.cu` -> `ssim3d.hip` with zero unsupported CUDA calls. setup.py HIP branch: `['-O3', '-DFUSED_SSIM_CUDA', '-ffast-math']`. Module imports cleanly; `is_3D_supported=True`; all four symbols present (`fusedssim`, `fusedssim_backward`, `fusedssim3d`, `fusedssim_backward3d`).

**gfx1100 code-object evidence** (`roc-obj-ls` on `fused_ssim_cuda.cpython-312-x86_64-linux-gnu.so`):

```
1  hipv4-amdgcn-amd-amdhsa--gfx1100  offset=208896 size=14896   (ssim.hip kernel)
2  hipv4-amdgcn-amd-amdhsa--gfx1100  offset=229376 size=23312   (ssim3d.hip kernel)
```

Both kernels compiled natively for gfx1100; no gfx90a or generic-arch code objects present.

### Test results (all from /tmp, HIP_VISIBLE_DEVICES=0)

**1. Repo tests/test.py** (2D, 100 iters, B=5 CH=5 H=1080 W=1920, `torch.isclose` fwd + full-tensor grad `.all()` vs ref SSIM and pytorch_msssim): PASS (exit 0). Fused fwd 4.6ms vs ref 157ms.

```
cd /tmp && HIP_VISIBLE_DEVICES=0 python /var/lib/jenkins/moat/projects/fused-ssim/src/tests/test.py
```

**2. Repo tests/test_3D.py** (3D, 10 iters, B=2 CH=1 D=H=W=96, `torch.allclose` rtol=1e-6 atol=1e-8 vs conv3d ref and pytorch_msssim(spatial_dims=3)): PASS (exit 0). Fused fwd 0.33ms vs ref 452ms.

```
cd /tmp && HIP_VISIBLE_DEVICES=0 python /var/lib/jenkins/moat/projects/fused-ssim/src/tests/test_3D.py
```

Both tests run twice; identical pass results.

**3. Independent harness** `agent_space/fused_ssim_validate_gfx1100.py` (2D: 4 configs CH=1/3/16, B=1..4; 3D: 3 shapes; NaN/Inf checks; bitwise determinism):

```
cd /tmp && HIP_VISIBLE_DEVICES=0 python /var/lib/jenkins/moat/agent_space/fused_ssim_validate_gfx1100.py
```

Results:
- 2D fwd vs ref (same-padding): max|diff| = 5.22e-08 (tolerance 1e-4; 4 orders margin)
- 2D fwd vs pm (valid-padding): max|diff| = 1.25e-07 (tolerance 1e-4; 3 orders margin)
- 2D grad vs ref: max|diff| = 1.75e-09; 2D grad vs pm: max|diff| = 2.10e-09
- 3D fwd vs pm: max|diff| = 1.32e-08; 3D grad vs pm: max|diff| = 3.12e-10
- NaN/Inf: none in fwd or grad, 2D and 3D
- Determinism: 2D and 3D fwd+grad bitwise identical across two calls within run

### Comparison to gfx90a

gfx90a (wave64): 2D fwd_ref ~1e-8, 2D grad ~1e-10..1e-12, 3D fwd ~5e-9, 3D grad ~1e-11.
gfx1100 (wave32): 2D fwd_ref 5.22e-08, 2D grad 1.75e-09, 3D fwd 1.32e-08, 3D grad 3.12e-10.
All differences between platforms are within ~1 order of magnitude and well within the asserted tolerances. The literal-32 stride in the 2D backward load is confirmed correct on wave32 (gradients pass at the same tolerance as wave64).

**RESULT: PASS. linux-gfx1100 -> completed.**

## Windows gfx1151 blocker: c10 inherited-ctor export (clang callee-cleanup gap)

fused-ssim registers via PYBIND11_MODULE (ext.cpp). The pybind11 at::Tensor type
caster inlines a TORCH_CHECK_VALUE -> constructs c10::ValueError(SourceLocation,
std::string). On Windows that ctor is dllimport from c10.dll, but c10.dll (built
with AMD clang-cl) does NOT export it -> LNK2001 __imp_??0ValueError@c10@@...

Root cause (proven on this host, rocm-sdk 7.14.0a20260531, AMD clang 23.0.0git):
clang does not export INHERITED constructors (using Error::Error) of a
__declspec(dllexport) class when the ctor has callee-cleanup parameters, i.e. a
non-trivially-destructible by-value param. Every c10::Error subclass ctor is
(SourceLocation, std::string) -- std::string by value -> callee-cleanup -> dropped.
  c10.lib DOES export ValueError's implicit copy/move ctors and the BASE
  Error(SourceLocation,std::string); it omits only the inherited ValueError ctor.
  Minimal repro: a dllexport Derived:Base{using Base::Base;} exports the inherited
  ctor when the param is `int` (trivial) but NOT when it is `std::string`, with
  warning "exporting inherited constructor is not yet supported; dllexport ignored
  on inherited constructor with callee-cleanup parameters".

Upstream state:
- Issue pytorch/pytorch#181892 (jeffdaily) documents the class.
- PR pytorch/pytorch#175340 (explicit exported constructors for the affected
  c10::Error subclasses, ValueError/NotImplementedError) LANDED on pytorch main
  as commit 03a98a4c4249 (2026-06-02). NOTE: the GitHub PR reads CLOSED/mergedAt=null
  because pytorchbot lands via Meta's internal flow and never sets the "Merged"
  flag -- it IS landed (see the pytorchbot merge-flow note in PORTING_GUIDE).
- The pytorch-side explicit-ctor fix is the real fix: clang at the compiler level
  does not export callee-cleanup inherited ctors (LLVM#182706, merged 2026-03-05,
  fixes only TRIVIAL-param inherited ctors; the by-value-std::string case is still
  unsupported). So the `/ALTERNATENAME` workaround is needed only until a TheRock
  Windows torch wheel ships a torch built after 03a98a4c, not because #175340 was
  rejected.

The fused-ssim HIP port itself is correct (kernels validate on gfx90a + gfx1100).
The Windows blocker is entirely an external torch-wheel/clang toolchain bug, out of
port scope (cf. LMCache POSIX, llm.c makefile). Unblocks automatically when either
(a) a TheRock torch wheel ships with #175340-style explicit ctors, or (b) clang
gains callee-cleanup inherited-ctor export. gsplat avoids this by using
TORCH_LIBRARY (no pybind Tensor caster instantiation).

## Audit/confirm 2026-06-03

**GPU:** AMD Instinct MI250X / MI250, gfx90a (CDNA2, wave64). ROCm 7.2.1. torch 2.13.0a0+gitb5e90ff, hip 7.2.53211.

**Upstream HEAD check:** upstream advanced to `a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38` (PR #39, "3D fused ssim documentation and tests update"). Diff vs validated sha `666987fb`: README.md only -- the upstream author rolled back the "3D supported on ROCm" claim to "CUDA only". No source file (.cu, .h, setup.py, ext.cpp) changed. The validated sha remains the assessment reference; the source code is identical.

**Warp-size re-verification (ssim.cu + ssim3d.cu at 666987fb):**

Grepped both `.cu` files for: `__shfl* __ballot __activemask __any __all __reduce __popc tiled_partition cg:: warpSize 0xffffffff`. Result: zero matches. The `cg::this_thread_block()` calls (lines ssim.cu:75,302; ssim3d.cu:142,457) are block-level cooperative groups (`block.sync()` = `__syncthreads()`), not warp-level.

The literal-32 pattern (ssim.cu:328-335: `warp_id=tid/32; lane_id=tid%32; num_warps=(T+31)/32; col+=32`) is purely loop-tiling stride distributing 256 block threads over the 26x26 shared tile. No cross-lane operation is present. Correct on wave64 unchanged (gradients confirm). The comment "same warp" at ssim.cu:167 and ssim3d.cu:252 describes shared-mem index proximity, not a warp intrinsic.

**Verdict: genuinely wave-agnostic. No warp-level primitive anywhere in either kernel.**

**Build (Strategy B):**

```
cd /var/lib/jenkins/moat/projects/fused-ssim/src && HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx90a python -m pip install -e . --no-build-isolation -v
```

Compile time: 53.6s. setup.py HIP branch: `['-O3', '-DFUSED_SSIM_CUDA', '-ffast-math']`. All four symbols present; `is_3D_supported=True`.

**Test results (all from /tmp, HIP_VISIBLE_DEVICES=0):**

1. `tests/test.py` (2D, 100 iters, B=5 CH=5 H=1080 W=1920, forward + full-tensor gradient `.all()` vs reference and pytorch_msssim): PASS (exit 0). Fused fwd 3.8ms vs ref 197ms.

2. `tests/test_3D.py` (3D, 10 iters, B=2 CH=1 D=H=W=96, `torch.allclose` rtol=1e-6 atol=1e-8): PASS (exit 0). Fused fwd 0.63ms vs ref 96ms.

3. Independent harness `agent_space/fused_ssim_validate.py` (2D: 4 configs CH=1/3/16, B=1..4; 3D: 3 shapes; determinism):
   - 2D fwd vs ref: max|diff| 5.6e-09 .. 3.4e-08 (tolerance 1e-4; 4+ orders margin)
   - 2D grad vs ref: max|diff| 2.3e-10 .. 5.2e-12
   - 3D fwd vs pm: max|diff| 2.8e-09 .. 5.1e-09
   - 3D grad vs pm: max|diff| 3.6e-11 .. 5.1e-11
   - NaN/Inf: none. Determinism: 2D and 3D fwd+grad bitwise identical.
   - ALL PASS.

**Best-practice assessment:**

| Criterion | Result |
|---|---|
| Footprint | Zero source changes (upstream already has AMD support at build-config level) |
| Strategy | Strategy B (torch hipify at build time) |
| Warp-size handling | Wave-agnostic: no warp primitives, literal-32 is benign loop-tiling |
| Fault classes | None (no texture, atomics RMW, cuBLAS/cuFFT/cuRAND, layered arrays) |
| CI | No GHA workflows added or modified |
| Wave32 soundness | gfx1100 completion is sound: no arch-specific device code, wave-agnostic kernels confirmed correct on gfx1100 gradients |
| gfx1151 blocked | c10 inherited-ctor export clang-cl bug (external toolchain, not port); port correct |

PORTING_GUIDE best practices: **MEETS**. This is the cleanest possible AMD port (build-config only, zero kernel edits, wave-agnostic, all real GPU tests pass).

**RESULT: PASS. linux-gfx90a remains completed.**

## Upstream-PR disposition: VALIDATED outcome, no PR (delta check 2026-06-04)

An earlier README-sweep note here said "DO NOT open a duplicate PR -- upstream already supports ROCm." Retracted as too blunt. Upstream does advertise a ROCm build (setup.py HIP branch, #29; README "AMD GPUs (ROCm)"; Anton Smirnov credit), but the delta check shows our work is real, just not code:

- CODE DELTA: NONE. Strategy B (torch hipify), arch-agnostic build, no fork commit -- we validated upstream's own commit 666987fb directly (fork_url/base_sha both null). There is nothing to contribute as a code PR, unlike CTranslate2's 2-line gfx90a arch-list add.
- VALIDATION DELTA (the value -> outcome=validated): first GPU validation of the ROCm path on gfx90a (CDNA2 wave64) and gfx1100 (RDNA3 wave32) at ROCm 7.2.1; upstream has no AMD GPU CI. Notably the 3D SSIM path: upstream's README disclaims "3D (CUDA only)" -- a claim the author ROLLED BACK TO in PR #39 -- yet fusedssim3d compiles, exports, and matches pytorch_msssim(spatial_dims=3) to ~1e-8 fwd / ~1e-11 grad on both AMD archs. Upstream's docs understate their own ROCm capability; we have the hardware evidence. Confirmed wave-agnostic (no warp primitives).

Outcome = validated (2 arch). No upstream PR: zero code delta, and the only conceivable contribution (a README note that 3D also works on ROCm) is exactly the claim the author deliberately removed in #39, so it would need AMD-CI-grade evidence to be welcome. See [[moat-no-duplicate-amd-ports]].

## Validation 2026-06-07 (gfx1201, Windows, ROCm 7.14.0a20260604)

**GPU:** AMD Radeon RX 9070 XT, gfx1201 (RDNA4, wave32). ROCm 7.14.0a20260604 (TheRock nightly). torch 2.9.1+rocm7.14.0a20260604, hip 7.14.60850-d34cbb64. Validated sha: `cdfe1fa6fced7898a9c8b3ba48e78633646ce28d` (adds Windows ext.cpp fix on top of the existing Windows shim commit).

**Fork interaction:** two commits on moat-port for Windows:
1. `767e3f6` -- routes ext.cpp through hipcc via `_winhip.cu` shim (c10 inherited-ctor workaround)
2. `cdfe1fa` -- fixes 3D symbols missing on Windows: `#ifdef FUSED_SSIM_CUDA` in ext.cpp became dead on the hipcc path because torch's `_hipify_compile_flags` renames `-DFUSED_SSIM_CUDA` -> `-DFUSED_SSIM_HIP` for all nvcc/hipcc-compiled files; changed to `#if defined(FUSED_SSIM_CUDA) || defined(FUSED_SSIM_HIP)` so both Linux (CXX path, FUSED_SSIM_CUDA) and Windows (hipcc path, FUSED_SSIM_HIP) compile all four symbols.

### Build

```python
# B:\develop\moat\agent_space\fused_ssim_build_gfx1201.py
# HIP_VISIBLE_DEVICES=0, PYTORCH_ROCM_ARCH=gfx1201
# ROCM_HOME=<venv>/_rocm_sdk_devel, HIP_DEVICE_LIB_PATH=<venv>/_rocm_sdk_devel/lib/llvm/amdgcn/bitcode
# DISTUTILS_USE_SDK=1, MSVC link.exe prepended to PATH
python B:\develop\moat\agent_space\fused_ssim_build_gfx1201.py
```

Compile time: ~60s. torch hipify: `ssim.cu` -> `ssim.hip`, `ssim3d.cu` -> `ssim3d.hip`, `ext_winhip.cu` (HIP path). All four symbols present (`fusedssim`, `fusedssim_backward`, `fusedssim3d`, `fusedssim_backward3d`); `is_3D_supported=True`. Linked against MSVC link.exe; `amdhip64`, `c10_hip`, `torch_hip`.

### Test results (all HIP_VISIBLE_DEVICES=0, non-source cwd)

**1. tests/test.py** (2D, 100 iters, B=5 CH=5 H=1080 W=1920, `torch.isclose` fwd + full-tensor grad `.all()` vs ref SSIM and pytorch_msssim): PASS (exit 0). Fused fwd ~4.0ms vs ref ~132ms.

```
HIP_VISIBLE_DEVICES=0 B:\develop\TheRock\external-builds\pytorch\.venv\Scripts\python.exe B:\develop\moat\projects\fused-ssim\src\tests\test.py
```

**2. tests/test_3D.py** (3D, 10 iters, B=2 CH=1 D=H=W=96, `torch.allclose` rtol=1e-6 atol=1e-8 vs conv3d ref and pytorch_msssim(spatial_dims=3)): PASS (exit 0).

```
HIP_VISIBLE_DEVICES=0 B:\develop\TheRock\external-builds\pytorch\.venv\Scripts\python.exe B:\develop\moat\projects\fused-ssim\src\tests\test_3D.py
```

**3. Independent harness** `agent_space/fused_ssim_validate_gfx1201.py` (2D: 4 configs CH=1/3/16, B=1..4; 3D: 3 shapes "valid" padding vs pytorch_msssim(spatial_dims=3); NaN/Inf; bitwise determinism):

```
cd C:\Windows\Temp && HIP_VISIBLE_DEVICES=0 B:\develop\TheRock\external-builds\pytorch\.venv\Scripts\python.exe B:\develop\moat\agent_space\fused_ssim_validate_gfx1201.py
```

Results:
- 2D fwd vs ref (same): max|diff| = 3.35e-08 (tolerance 1e-4; 4+ orders margin)
- 2D fwd vs pm (valid): max|diff| = 5.59e-09
- 2D grad vs ref: max|diff| = 1.63e-09
- 3D fwd vs pm (valid): max|diff| = 4.97e-09; 3D grad vs pm: max|diff| = 2.87e-11
- NaN/Inf: none in fwd or grad, 2D and 3D
- Determinism: 2D and 3D fwd+grad bitwise identical across two calls within run
- ALL PASS (exit 0)

### Comparison to Linux platforms

gfx90a (wave64): 2D fwd_ref ~1e-8, 2D grad ~1e-10, 3D fwd ~5e-9, 3D grad ~1e-11.
gfx1100 (wave32): 2D fwd_ref 5.22e-08, 2D grad 1.75e-09, 3D fwd 1.32e-08, 3D grad 3.12e-10.
gfx1201 (wave32): 2D fwd_ref 3.35e-08, 2D grad 1.63e-09, 3D fwd 4.97e-09, 3D grad 2.87e-11.
All platforms within ~1 order of magnitude; well within tolerances.

**RESULT: PASS. windows-gfx1201 -> completed.**

### Note on linux platform revalidate state

The `cdfe1fa` ext.cpp change (FUSED_SSIM_CUDA || FUSED_SSIM_HIP) is `arch_independent=False` per the classifier (token count differs), so linux-gfx90a and linux-gfx1100 are now in `revalidate`. However, on Linux the CXX compiler path gets `-DFUSED_SSIM_CUDA` (unchanged by `_hipify_compile_flags` which only touches nvcc flags), so the compiled ext.o is byte-identical to the 666987fb build. Linux validator agents can carry forward with binary-equiv check.

## Validation 2026-06-07 (gfx90a, carry-forward via binary-equiv)

**GPU:** AMD Instinct MI250X, gfx90a (CDNA2, wave64). ROCm 7.2.1. Head sha: `cdfe1fa6fced7898a9c8b3ba48e78633646ce28d`.

**Delta:** `666987fb..cdfe1fa` adds two commits:
- `767e3f6` -- Windows-only setup.py shim (gated on `os.name == 'nt'`; dead on Linux)
- `cdfe1fa` -- ext.cpp `#ifdef FUSED_SSIM_CUDA` -> `#if defined(FUSED_SSIM_CUDA) || defined(FUSED_SSIM_HIP)` (Linux CXX path still gets `-DFUSED_SSIM_CUDA`; condition evaluates identically)

**Binary-equivalence check:**

Built both SHAs with `HIP_VISIBLE_DEVICES=2 PYTORCH_ROCM_ARCH=gfx90a`:

```
conda run -n py_3.12 bash -c "cd /tmp/fused-ssim-old && pip install -e . --no-build-isolation -v"
conda run -n py_3.12 bash -c "cd /tmp/fused-ssim-new && pip install -e . --no-build-isolation -v"
```

Both builds reported: `NVCC args: ['-O3', '-DFUSED_SSIM_CUDA', '-ffast-math']. CXX args: ['-O3', '-DFUSED_SSIM_CUDA'].`

```
python3 utils/codeobj_diff.py /tmp/fused-ssim-build-old /tmp/fused-ssim-build-new
```

Output: `verdict=identical -- fused_ssim_cuda.cpython-312-x86_64-linux-gnu.so: identical (exported symbols + device ISA identical (161 exports))`

**RESULT: CARRY FORWARD. linux-gfx90a -> completed (binary-equiv at cdfe1fa).**

## Revalidation 2026-06-07 (gfx1100, carry-forward via binary-equiv)

**GPU:** AMD Radeon Pro W7800 48GB, gfx1100 (RDNA3, wave32). ROCm 7.2.1. torch 2.13.0a0+gitb5e90ff. Head sha: `cdfe1fa6fced7898a9c8b3ba48e78633646ce28d`. Revalidating from `666987fb`.

**Delta:** `666987fb..cdfe1fa` -- identical to gfx90a delta above: two Windows-only commits (setup.py `os.name == 'nt'` block; ext.cpp `#ifdef` widens to `|| defined(FUSED_SSIM_HIP)`). Linux CXX path still receives `-DFUSED_SSIM_CUDA`; both conditions evaluate identically on Linux.

**Binary-equivalence check:**

Built both SHAs with `HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1100`:

```
cd /tmp/fused-ssim-old && HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1100 python -m pip install -e . --no-build-isolation -v
cd /tmp/fused-ssim-new && HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1100 python -m pip install -e . --no-build-isolation -v
```

Both builds reported: `NVCC args: ['-O3', '-DFUSED_SSIM_CUDA', '-ffast-math']. CXX args: ['-O3', '-DFUSED_SSIM_CUDA'].`

```
python3 utils/codeobj_diff.py /tmp/fused-ssim-build-old /tmp/fused-ssim-build-new
```

Output: `verdict=identical -- fused_ssim_cuda.cpython-312-x86_64-linux-gnu.so: identical (exported symbols + device ISA identical (161 exports))`

**RESULT: CARRY FORWARD. linux-gfx1100 -> completed (binary-equiv at cdfe1fa).**

## Validation 2026-06-16 (gfx1101, Windows, ROCm 7.14.0a20260604)

**GPU:** AMD Radeon PRO V710, gfx1101 (RDNA3, wave32). ROCm 7.14.0a20260604 (TheRock nightly). torch 2.9.1+rocm7.14.0a20260604, hip 7.14.60850-d34cbb64. Validated sha: `6f9a6201df9cb774d8256cf802832c6cc1f6fb03`.

**Fork interaction:** none beyond gfx1201. The two Windows commits (767e3f6, cdfe1fa) already on moat-port cover gfx1101 identically; no additional source or build-config changes were needed.

### Build

```python
# B:\develop\moat\agent_space\fused_ssim_build_gfx1101.py
# HIP_VISIBLE_DEVICES=0, PYTORCH_ROCM_ARCH=gfx1101
# ROCM_HOME=<venv>/_rocm_sdk_devel, HIP_DEVICE_LIB_PATH=<venv>/_rocm_sdk_devel/lib/llvm/amdgcn/bitcode
# DISTUTILS_USE_SDK=1, MSVC link.exe prepended to PATH
python B:\develop\moat\agent_space\fused_ssim_build_gfx1101.py
```

torch hipify: `ssim.cu` -> `ssim.hip`, `ssim3d.cu` -> `ssim3d.hip`, `ext_winhip.cu` (HIP path). All four symbols present (`fusedssim`, `fusedssim_backward`, `fusedssim3d`, `fusedssim_backward3d`); `is_3D_supported=True`. Linked against MSVC link.exe; `amdhip64`, `c10_hip`, `torch_hip`. NVCC args: `['-O3', '-DFUSED_SSIM_CUDA', '-ffast-math']`.

### Test results (all HIP_VISIBLE_DEVICES=0, non-source cwd)

**1. tests/test.py** (2D, 100 iters, B=5 CH=5 H=1080 W=1920, `torch.isclose` fwd + full-tensor grad `.all()` vs ref SSIM and pytorch_msssim): PASS (exit 0). Fused fwd ~4.6ms (warmup incl.) vs ref 938ms.

```
HIP_VISIBLE_DEVICES=0 B:\develop\TheRock\external-builds\pytorch\.venv\Scripts\python.exe B:\develop\moat\projects\fused-ssim\src\tests\test.py
```

**2. tests/test_3D.py** (3D, 10 iters, B=2 CH=1 D=H=W=96, `torch.allclose` rtol=1e-6 atol=1e-8 vs conv3d ref and pytorch_msssim(spatial_dims=3)): PASS (exit 0). Fused fwd 6.3ms vs ref 8307ms.

```
HIP_VISIBLE_DEVICES=0 B:\develop\TheRock\external-builds\pytorch\.venv\Scripts\python.exe B:\develop\moat\projects\fused-ssim\src\tests\test_3D.py
```

**3. Independent harness** `agent_space/fused_ssim_validate_gfx1101.py` (2D: 4 configs CH=1/3/16, B=1..4; 3D: 3 shapes "valid" padding vs pytorch_msssim(spatial_dims=3); NaN/Inf; bitwise determinism):

```
cd C:\Windows\Temp && HIP_VISIBLE_DEVICES=0 B:\develop\TheRock\external-builds\pytorch\.venv\Scripts\python.exe B:\develop\moat\agent_space\fused_ssim_validate_gfx1101.py
```

Results:
- 2D fwd vs ref (same): max|diff| = 4.10e-08 (tolerance 1e-4; 4+ orders margin)
- 2D fwd vs pm (valid): max|diff| = 1.26e-07 (tolerance 1e-4; 3+ orders margin)
- 2D grad vs ref: max|diff| = 1.63e-09
- 3D fwd vs pm (valid): max|diff| = 5.96e-09; 3D grad vs pm: max|diff| = 2.65e-11
- NaN/Inf: none in fwd or grad, 2D and 3D
- Determinism: 2D and 3D fwd+grad bitwise identical across two calls within run
- ALL PASS (exit 0)

### Comparison to other platforms

gfx90a (wave64): 2D fwd_ref ~1e-8, 2D grad ~1e-10, 3D fwd ~5e-9, 3D grad ~1e-11.
gfx1100 (wave32): 2D fwd_ref 5.22e-08, 2D grad 1.75e-09, 3D fwd 1.32e-08, 3D grad 3.12e-10.
gfx1201 (wave32): 2D fwd_ref 3.35e-08, 2D grad 1.63e-09, 3D fwd 4.97e-09, 3D grad 2.87e-11.
gfx1101 (wave32): 2D fwd_ref 4.10e-08, 2D grad 1.63e-09, 3D fwd 5.96e-09, 3D grad 2.65e-11.
All platforms within ~1 order of magnitude; well within tolerances. gfx1101 results closely match gfx1201 (same wave32, both RDNA3/RDNA4).

**RESULT: PASS. windows-gfx1101 -> completed.**
