# FastGeodis notes

## Build (linux-gfx90a)

Strategy B PyTorch extension. The only required change was fixing setup.py to detect
`ROCM_HOME` when `CUDA_HOME` is not set, enabling the GPU extension build on ROCm PyTorch.

PyTorch's hipify automatically translates `fastgeodis_cuda.cu` at build time; no kernel-level
changes are needed. All 4 kernels use only standard CUDA runtime APIs and `__syncthreads()`
that map 1:1 to HIP.

Build command:
```bash
HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx90a pip install -e . --no-build-isolation
```

## Test (linux-gfx90a)

300 tests pass in ~72s:
```bash
HIP_VISIBLE_DEVICES=0 python -m pytest tests/ -v
```

Test coverage: 2D/3D geodesic distance transforms (GPU + CPU), signed/unsigned variants,
plus CPU-only algorithms (Toivanen, pixelqueue, fastmarch).

## Gotchas

- ROCm detection: Original setup.py only checks `CUDA_HOME`, which is None on pure ROCm
  systems. Fixed by also checking `ROCM_HOME` from `torch.utils.cpp_extension`.

## Review 2026-06-05

Reviewed by: reviewer agent

**Verdict: Approve**

No problems found. This is an exemplary minimal Strategy B port:
- setup.py change correctly detects ROCM_HOME as alternative to CUDA_HOME
- No kernel changes needed -- hipify handles cudaMemcpyToSymbol and __syncthreads() 1:1
- No warp intrinsics, textures, CUDA libraries, or RAII handles -- no fault classes apply
- CUDA build preserved (additive change)
- Commit message correct: [ROCm] prefix, Claude mention, Test Plan, no noreply trailer
- 300/300 tests passed on gfx90a

Ready for validation.

## Validation 2026-06-05

Platform: linux-gfx90a (AMD Instinct MI250X, gfx90a, ROCm 7.2.53211)
Validated at: bef0d9317d896706c5d36fbf5ec24ddf6909c876

Build:
```bash
cd projects/FastGeodis/src
HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx90a pip install -e . --no-build-isolation
```
Build time: 55.6s

Test:
```bash
cd projects/FastGeodis/src
HIP_VISIBLE_DEVICES=0 python -m pytest tests/ -v
```
Test time: 76.3s

Result: PASS
- 300/300 tests passed
- GPU tests (TestFastGeodis, TestFastGeodisSigned, TestGSF with cuda device) all passed
- CPU tests (TestToivanen, TestPixelQueue, TestFastMarch) all passed - no regression
- All test classes cover: shape validation, euclidean distance output correctness, zero input, ones mask, ill-formed input handling

The port is validated. PyTorch's auto-hipify successfully translated all CUDA kernels (cudaMemcpyToSymbol to constant memory, __syncthreads() barriers). No numeric divergence observed on gfx90a.

## Validation 2026-06-05 (linux-gfx1100)

Platform: linux-gfx1100 (AMD Radeon Pro W7800 48GB, gfx1100, ROCm 7.2.53211)
Validated at: bef0d9317d896706c5d36fbf5ec24ddf6909c876

Build:
```bash
cd projects/FastGeodis/src
HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1100 pip install -e . --no-build-isolation
```

Test:
```bash
cd projects/FastGeodis/src
HIP_VISIBLE_DEVICES=0 python -m pytest tests/ -v
```

Result: PASS
- 300/300 tests passed (matching gfx90a)
- GPU tests (TestFastGeodis, TestFastGeodisSigned, TestGSF with cuda device) all passed
- CPU tests (TestToivanen, TestPixelQueue, TestFastMarch) all passed - no regression
- All test classes cover: shape validation, euclidean distance output correctness, zero input, ones mask, ill-formed input handling

The port is validated on gfx1100 (RDNA3). PyTorch's auto-hipify successfully handles the RDNA3 wavefront size (32 lanes) -- the kernels use __syncthreads() only (no warp intrinsics), so they are wavefront-size agnostic. No numeric divergence observed.

## Validation 2026-06-07 (windows-gfx1201)

Platform: windows-gfx1201 (AMD Radeon RX 9070 XT, gfx1201, RDNA4, wave32, ROCm/HIP 7.14.0a20260604)
Validated at: 5fc19ecea7e516778ab4db264e584230c83676b4

A Windows build fix was necessary: c10.dll built with clang does not export the inherited
constructor c10::ValueError(SourceLocation, string), causing LNK2001 when MSVC compiles the
extension .cpp. Fixed by adding a /ALTERNATENAME linker directive in setup.py (Windows-only,
guarded by sys.platform == "win32") that redirects the missing thunk to c10::Error(SourceLocation,
string), which IS exported. This fix was committed as a new commit on top of the ROCm port
(5fc19ec); the Linux-validated bef0d93 remains reachable as an ancestor.

Build environment:
- MSVC link.exe prepended to PATH (before Git's /usr/bin/link.exe)
- ROCM_HOME=_rocm_sdk_devel (TheRock venv), DISTUTILS_USE_SDK=1
- HIP_VISIBLE_DEVICES=0, PYTORCH_ROCM_ARCH=gfx1201

Build command:
```
export PATH="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64:$PATH"
cd projects/FastGeodis/src
HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1201 \
  ROCM_HOME="<venv>/Lib/site-packages/_rocm_sdk_devel" \
  DISTUTILS_USE_SDK=1 \
  python.exe -m pip install -e . --no-build-isolation
```
Build time: ~31s

Test command:
```
cd projects/FastGeodis/src
HIP_VISIBLE_DEVICES=0 python.exe -m pytest tests/ -v
```
Test time: ~5s

Result: PASS
- 300/300 tests passed (matching gfx90a and gfx1100)
- GPU tests (TestFastGeodis, TestFastGeodisSigned, TestGSF with cuda device) all passed
- CPU tests (TestToivanen, TestPixelQueue, TestFastMarch) all passed - no regression
- All 2D/3D geodesic distance, signed/unsigned, shape/input validation tests passed

GPU confirmed: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32) at HIP_VISIBLE_DEVICES=0.
PyTorch auto-hipify + offload-arch=gfx1201. No numeric divergence on RDNA4.

Note for linux-gfx90a/gfx1100: the 5fc19ec commit adds only a sys.platform=="win32"-guarded
/ALTERNATENAME linker directive. Linux build inputs and kernels are unchanged. The Linux
validators can carry forward via binary-equivalence check (codeobj_diff.py) without re-running
GPU tests.

## Revalidation 2026-06-08 (linux-gfx90a)

Platform: linux-gfx90a (AMD Instinct MI250X, gfx90a, ROCm 7.2.x)
Revalidated at: 5fc19ecea7e516778ab4db264e584230c83676b4
Previous validated_sha: bef0d9317d896706c5d36fbf5ec24ddf6909c876

Delta: one commit (5fc19ec) adds a /ALTERNATENAME linker directive in setup.py gated by
`if sys.platform == "win32" and BUILD_CUDA`. That block never executes on Linux; the compiled
extension and device code are byte-identical to the bef0d93 build. Full GPU suite run as
ground truth rather than codeobj_diff (suite is cheap at ~71s).

Build:
```bash
HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx90a pip install -e . --no-build-isolation
```
(extension already built from prior validation; editable install confirmed)

Test:
```bash
HIP_VISIBLE_DEVICES=0 python -m pytest tests/ -v
```
Test time: 71.13s

Result: PASS
- 300/300 tests passed (matching all prior validations)
- GPU tests (TestFastGeodis, TestFastGeodisSigned, TestGSF with cuda device) all passed
- CPU tests (TestToivanen, TestPixelQueue, TestFastMarch) all passed - no regression
- 4 pre-existing SyntaxWarning (invalid escape sequence in test docstrings) - not new

## Validation 2026-06-19 (windows-gfx1101)

Platform: windows-gfx1101 (AMD Radeon PRO V710, gfx1101, RDNA3, wave32, ROCm/HIP 7.14.0a20260604)
Validated at: 5fc19ecea7e516778ab4db264e584230c83676b4

Device verified: HIP_VISIBLE_DEVICES=1 -> AMD Radeon PRO V710 (gfx1101, 25.48 GB VRAM, warpSize=32).
GPU health check passed before build and test.

Build environment:
- MSVC link.exe prepended to PATH (before Git's /usr/bin/link.exe)
- ROCM_HOME=_rocm_sdk_devel (TheRock venv), DISTUTILS_USE_SDK=1
- HIP_VISIBLE_DEVICES=1, PYTORCH_ROCM_ARCH=gfx1101

Build command:
```
export PATH="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64:$PATH"
HIP_VISIBLE_DEVICES=1 PYTORCH_ROCM_ARCH=gfx1101 \
  ROCM_HOME="<venv>/Lib/site-packages/_rocm_sdk_devel" \
  DISTUTILS_USE_SDK=1 \
  python.exe -m pip install -e /b/develop/moat/projects/FastGeodis/src --no-build-isolation
```
Build time: ~32s
Arch verification: gfx1101 string found in FastGeodisCpp.cp312-win_amd64.pyd; gfx1201 absent.

Test command:
```
HIP_VISIBLE_DEVICES=1 python.exe -m pytest tests/ -v
```
Test time: ~5s

Result: PASS
- 300/300 tests passed (matching gfx90a, gfx1100, and gfx1201)
- GPU tests (TestFastGeodis, TestFastGeodisSigned, TestGSF with cuda device) all passed
- CPU tests (TestToivanen, TestPixelQueue, TestFastMarch) all passed - no regression
- No TDR events; gfx1101 remained healthy throughout (kernels are short-running)

gfx1101 note: RDNA3 wavefront size is 32 (same as gfx1201 RDNA4). Kernels use only
__syncthreads() with no warp intrinsics, making them wavefront-size agnostic. No numeric
divergence observed vs gfx1201 results. All 2D/3D geodesic distance transforms matched.

## Revalidation 2026-06-08 (linux-gfx1100)

Platform: linux-gfx1100 (AMD Radeon Pro W7800 48GB, gfx1100, ROCm 7.2.53211)
Revalidated at: 5fc19ecea7e516778ab4db264e584230c83676b4
Previous validated_sha: bef0d9317d896706c5d36fbf5ec24ddf6909c876
Method: binary-equivalence carry-forward (codeobj_diff)

Delta: one commit (5fc19ec) adds a /ALTERNATENAME linker directive in setup.py gated by
`if sys.platform == "win32" and BUILD_CUDA`. That block never executes on Linux.

Binary-equivalence check:
- Built bef0d93 at agent_space/FastGeodis-gfx1100-gpu0/old-src with PYTORCH_ROCM_ARCH=gfx1100
- Built 5fc19ec at agent_space/FastGeodis-gfx1100-gpu0/new-src with PYTORCH_ROCM_ARCH=gfx1100
- python3 utils/codeobj_diff.py old-build new-build

Result: `verdict=identical`
- FastGeodisCpp.cpython-312-x86_64-linux-gnu.so: identical (exported symbols + device ISA identical, 552 exports)
- Device code objects and exported symbols are byte-identical between both SHAs on gfx1100.

Carry-forward: linux-gfx1100 advanced to 5fc19ec via binary-equiv without GPU re-run.
Prior GPU validation (300/300 tests on gfx1100 at bef0d93) remains the evidence base.
