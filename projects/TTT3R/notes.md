# TTT3R notes

## Build (linux-gfx90a)

```bash
cd src/croco/models/curope
rm -rf build *.so *.hip
PYTORCH_ROCM_ARCH=gfx90a python setup.py build_ext --inplace
```

## Validation

```bash
cd src/croco/models/curope
python -c "import torch; import curope; print('curope imported successfully')"
```

GPU vs CPU correctness test:
```bash
python -c "
import torch, curope
tokens = torch.randn(2, 16, 8, 64, device='cuda')
pos = torch.randint(0, 10, (2, 16, 2), device='cuda', dtype=torch.int64)
ref = tokens.clone().cpu()
pos_cpu = pos.cpu()
curope.rope_2d(tokens, pos, 100.0, 1.0)
curope.rope_2d(ref, pos_cpu, 100.0, 1.0)
diff = (tokens.cpu() - ref).abs().max().item()
assert diff < 1e-5, f'diff={diff}'
print(f'PASS (diff={diff:.2e})')
"
```

## Port details

Changes:
1. setup.py: Detect ROCm via torch.version.hip, skip cuda.get_gencode_flags() on ROCm (fails on AMD), use HIP-compatible flags
2. kernels.cu: Fix deprecated tokens.type() -> tokens.scalar_type() (required for modern PyTorch)

The kernel is portable (no warp intrinsics, no CUDA libraries, only __syncthreads__ and standard math) so PyTorch's automatic hipify handles the CUDA->HIP translation at build time.

Gotchas:
- Cannot use -ffast-math with hipcc on the kernel file because PyTorch headers use std::isinf() which conflicts with -ffinite-math-only (part of -ffast-math)
- The cuRoPE2D module expects input in (B, N, H, D) layout or (B, H, N, D) created from transpose of (B, N, H, D) to maintain correct strides

## Review 2026-06-05

Reviewed branch `moat-port` against upstream `main`.

**Verdict**: Approve

**Summary**: Clean Strategy B (PyTorch extension) port. The setup.py detects ROCm via `torch.version.hip` and uses HIP-compatible compile flags; the CUDA path is preserved unchanged. The only kernel change (`tokens.type()` -> `tokens.scalar_type()`) is a legitimate PyTorch modernization that applies to both platforms.

The kernel uses only portable constructs (`__global__`, `__syncthreads__`, `extern __shared__`, standard math functions). No warp intrinsics, no CUDA libraries, no textures. PyTorch's automatic hipification handles the CUDA->HIP runtime symbol translation at build time.

GPU vs CPU validation on gfx90a: max diff 2.26e-06 (within 1e-5 tolerance).

No findings.

## Validation 2026-06-05

Platform: linux-gfx90a (MI250X, HIP_VISIBLE_DEVICES=1)
Commit: d162988b5af74e7939faaea422c004ddc1511503

Build:
```bash
cd /var/lib/jenkins/moat/projects/TTT3R/src/src/croco/models/curope
HIP_VISIBLE_DEVICES=1 PYTORCH_ROCM_ARCH=gfx90a python setup.py build_ext --inplace
```

Test results:
- Import test: PASS
- GPU vs CPU correctness: PASS (max diff=2.38e-06, tolerance 1e-5)

Validation commands:
```bash
HIP_VISIBLE_DEVICES=1 python -c "import torch; import curope; print('curope imported successfully')"

HIP_VISIBLE_DEVICES=1 python -c "
import torch, curope
torch.manual_seed(42)
tokens = torch.randn(2, 16, 8, 64, device='cuda')
pos = torch.randint(0, 10, (2, 16, 2), device='cuda', dtype=torch.int64)
ref = tokens.clone().cpu()
pos_cpu = pos.cpu()
curope.rope_2d(tokens, pos, 100.0, 1.0)
curope.rope_2d(ref, pos_cpu, 100.0, 1.0)
diff = (tokens.cpu() - ref).abs().max().item()
assert diff < 1e-5, f'diff={diff}'
print(f'PASS: GPU vs CPU match (max diff={diff:.2e})')
"
```

Real GPU validation passed.

## Validation 2026-06-07 (windows-gfx1201)

Platform: windows-gfx1201 (AMD Radeon RX 9070 XT, gfx1201, HIP_VISIBLE_DEVICES=0)
Commit: d064cc9a55b88763e5a3873dfa4a6b45e2751e00

Windows fix required: `c10::ValueError` LNK2001. On Windows, MSVC compiles
curope.cpp and generates a `__declspec(dllimport)` reference to
`c10::ValueError(SourceLocation, string)` which c10.dll (clang-built) does not
export. Fixed by adding a `/ALTERNATENAME` linker directive in setup.py that
aliases the missing ValueError thunk to `c10::Error(SourceLocation, string)`,
which IS exported. This fix is guarded by `sys.platform == "win32" and IS_ROCM`.

Build:
```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set HIP_VISIBLE_DEVICES=0
set PYTORCH_ROCM_ARCH=gfx1201
set ROCM_HOME=B:\develop\TheRock\external-builds\pytorch\.venv\Lib\site-packages\_rocm_sdk_devel
set DISTUTILS_USE_SDK=1
cd B:\develop\moat\projects\TTT3R\src\src\croco\models\curope
B:\develop\TheRock\external-builds\pytorch\.venv\Scripts\python.exe setup.py build_ext --inplace
```

Test results:
- Import test: PASS
- GPU vs CPU correctness: PASS (max diff=2.38e-06, tolerance 1e-5)

Validation commands:
```bash
HIP_VISIBLE_DEVICES=0 python -c "import torch; import curope; print('curope imported successfully')"

HIP_VISIBLE_DEVICES=0 python -c "
import torch, curope
torch.manual_seed(42)
tokens = torch.randn(2, 16, 8, 64, device='cuda')
pos = torch.randint(0, 10, (2, 16, 2), device='cuda', dtype=torch.int64)
ref = tokens.clone().cpu()
pos_cpu = pos.cpu()
curope.rope_2d(tokens, pos, 100.0, 1.0)
curope.rope_2d(ref, pos_cpu, 100.0, 1.0)
diff = (tokens.cpu() - ref).abs().max().item()
assert diff < 1e-5, f'diff={diff}'
print(f'PASS: GPU vs CPU match (max diff={diff:.2e})')
"
```

Real GPU validation passed.

## Revalidation 2026-06-08 (linux-gfx90a)

Platform: linux-gfx90a (MI250X, HIP_VISIBLE_DEVICES=3)
Revalidate: d162988b -> d064cc9a

Delta: one commit "[ROCm] Fix Windows c10::ValueError LNK2001 in curope extension"
Change: adds `import sys` and a `if sys.platform == "win32" and IS_ROCM:` block in setup.py that populates `extra_link_args` with a `/ALTERNATENAME` linker directive. On Linux this block is unreachable; `extra_link_args=[]` (no-op).

Method: binary-equivalence carry-forward. Built both SHAs with `PYTORCH_ROCM_ARCH=gfx90a` and ran `utils/codeobj_diff.py`.

Result: `verdict=identical` -- device ISA and all 66 exported symbols match.

Conclusion: carry-forward to `completed` at d064cc9a; no GPU re-run required.

## Validation 2026-06-05 (linux-gfx1100)

Platform: linux-gfx1100 (AMD Radeon Pro W7800)
Commit: d162988b5af74e7939faaea422c004ddc1511503

Build:
```bash
cd /var/lib/jenkins/moat/projects/TTT3R/src/src/croco/models/curope
rm -rf build *.so *.hip
PYTORCH_ROCM_ARCH=gfx1100 python setup.py build_ext --inplace
```

Test results:
- Import test: PASS
- GPU vs CPU correctness: PASS (max diff=2.38e-06, tolerance 1e-5)

Validation commands:
```bash
python -c "import torch; import curope; print('curope imported successfully')"

python -c "
import torch, curope
torch.manual_seed(42)
tokens = torch.randn(2, 16, 8, 64, device='cuda')
pos = torch.randint(0, 10, (2, 16, 2), device='cuda', dtype=torch.int64)
ref = tokens.clone().cpu()
pos_cpu = pos.cpu()
curope.rope_2d(tokens, pos, 100.0, 1.0)
curope.rope_2d(ref, pos_cpu, 100.0, 1.0)
diff = (tokens.cpu() - ref).abs().max().item()
assert diff < 1e-5, f'diff={diff}'
print(f'PASS: GPU vs CPU match (max diff={diff:.2e})')
"
```

Real GPU validation passed.

## Validation 2026-06-20 (windows-gfx1101)

Platform: windows-gfx1101 (AMD Radeon PRO V710, gfx1101, HIP_VISIBLE_DEVICES=1)
Commit: d064cc9a55b88763e5a3873dfa4a6b45e2751e00

Same Windows fix as gfx1201 (c10::ValueError /ALTERNATENAME linker directive) applied; setup.py already contains it guarded by `sys.platform == "win32" and IS_ROCM`.

Build:
```bash
VENV="B:/develop/TheRock/external-builds/pytorch/.venv"
ROCM_HOME="$VENV/Lib/site-packages/_rocm_sdk_devel"
MSVC_BIN="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
export PATH="$MSVC_BIN:$PATH"
export ROCM_HOME HIP_DEVICE_LIB_PATH="$ROCM_HOME/lib/llvm/amdgcn/bitcode"
export DISTUTILS_USE_SDK=1 HIP_VISIBLE_DEVICES=1 PYTORCH_ROCM_ARCH=gfx1101

cd projects/TTT3R/src/src/croco/models/curope
rm -rf build kernels.hip curope.cp312-win_amd64.pyd
python setup.py build_ext --inplace
```

Compiled with `--offload-arch=gfx1101` (confirmed in build log). The `/ALTERNATENAME` linker directive was included.

Test results:
- Import test: PASS (AMD Radeon PRO V710 confirmed as HIP device)
- GPU vs CPU correctness: PASS (max diff=2.38e-06, tolerance 1e-5)

Validation commands:
```bash
HIP_VISIBLE_DEVICES=1 python -c "
import sys
sys.path.insert(0, r'B:\develop\moat\projects\TTT3R\src\src\croco\models\curope')
import torch, curope
torch.manual_seed(42)
tokens = torch.randn(2, 16, 8, 64, device='cuda')
pos = torch.randint(0, 10, (2, 16, 2), device='cuda', dtype=torch.int64)
ref = tokens.clone().cpu()
pos_cpu = pos.cpu()
curope.rope_2d(tokens, pos, 100.0, 1.0)
curope.rope_2d(ref, pos_cpu, 100.0, 1.0)
diff = (tokens.cpu() - ref).abs().max().item()
assert diff < 1e-5, f'diff={diff}'
print(f'PASS: GPU vs CPU match (max diff={diff:.2e})')
"
```

Real GPU validation passed. Max diff 2.38e-06 matches gfx90a/gfx1201 results exactly.

## Revalidation 2026-06-08 (linux-gfx1100)

Platform: linux-gfx1100 (AMD Radeon Pro W7800, gfx1100, HIP_VISIBLE_DEVICES=0)
Revalidate: d162988b -> d064cc9a

Delta: one commit "[ROCm] Fix Windows c10::ValueError LNK2001 in curope extension"
Change: adds `import sys` and a `if sys.platform == "win32" and IS_ROCM:` block in setup.py that populates `extra_link_args` with a `/ALTERNATENAME` linker directive. On Linux this block is unreachable; `extra_link_args=[]` (no-op). The `extra_compile_args` and `kernels.cu` are unchanged.

Method: binary-equivalence carry-forward. Built both SHAs with `PYTORCH_ROCM_ARCH=gfx1100 HIP_VISIBLE_DEVICES=0` into agent_space/TTT3R-gfx1100-gpu0/{old,new} and ran `utils/codeobj_diff.py`.

```
python3 utils/codeobj_diff.py \
  agent_space/TTT3R-gfx1100-gpu0/old \
  agent_space/TTT3R-gfx1100-gpu0/new
```

Result: `verdict=identical` -- device ISA and all 66 exported symbols match.

Conclusion: carry-forward to `completed` at d064cc9a; no GPU re-run required.
