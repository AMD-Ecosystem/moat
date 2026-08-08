# faster-gaussian-splatting notes

## Build (linux-gfx90a)

```bash
conda activate py_3.12  # ROCm torch env
HIP_VISIBLE_DEVICES=2 pip install -e projects/faster-gaussian-splatting/src/FasterGSCudaBackend --no-build-isolation
```

## Test

```python
import sys
sys.path.insert(0, 'projects/faster-gaussian-splatting/src/FasterGSCudaBackend')
import torch
from FasterGSCudaBackend.torch_bindings.rasterization import rasterize, RasterizerSettings

n = 500; device = 'cuda'; torch.manual_seed(42)
means = torch.randn(n, 3, device=device)
scales = torch.randn(n, 3, device=device)
rotations = torch.randn(n, 4, device=device)
opacities = torch.randn(n, device=device)
sh_0 = torch.randn(n, 3, device=device)
sh_rest = torch.randn(n, 15, 3, device=device)

w2c = torch.eye(4, device=device).unsqueeze(0)[:, :3, :]
settings = RasterizerSettings(
    w2c=w2c, cam_position=torch.zeros(3, device=device),
    bg_color=torch.zeros(3, device=device), active_sh_bases=1, width=256, height=256,
    focal_x=200, focal_y=200, center_x=128, center_y=128, near_plane=0.1, far_plane=100,
    proper_antialiasing=True
)

image = rasterize(means, scales, rotations, opacities, sh_0, sh_rest.view(n, -1), settings, to_chw=True, clamp_output=True)
print(f'Output: {image.shape}, range=[{image.min():.4f}, {image.max():.4f}]')
```

## Gotchas

1. **C++20 required**: PyTorch 2.x uses C++20 concepts/requires in headers. The extension must use `-std=c++20`.

2. **helper_math.h vector ops conflict**: HIP's HIP_vector_type provides all float2/3/4 operators. NVIDIA's helper_math.h defines the same operators, causing "ambiguous overload" errors on HIP. Fixed by guarding the operators with `HELPER_MATH_SKIP_VECTOR_OPS`.

3. **std::lerp conflict**: C++20 adds std::lerp(float,float,float). The helper_math.h scalar lerp conflicts with it when `using namespace std` is in effect (via PyTorch headers). Fixed by conditionally compiling the scalar lerp based on C++ version.

4. **cub::DoubleBuffer not mapped by hipify**: torch hipify maps `cub::Device*` functions to `hipcub::`, but not the `cub::DoubleBuffer` type. Fixed with `namespace cub = hipcub;` on HIP.

5. **rsqrt not available on host**: The rsqrt intrinsic is CUDA device-only. Host code in kernels_mcmc.cuh was calling it; changed to `1/std::sqrt()`.

6. **Backward pass gradient shape issue**: The diff_rasterize backward pass has a shape mismatch for opacity gradients (returns [N,1] but expects [N]). This is a pre-existing bug in the original code, not related to the ROCm port.

## Wave64 safety

The code uses `cg::tiled_partition<32>` for warp-level operations. This creates a 32-lane logical tile regardless of hardware wave width. The `constexpr warp_size = 32` matches the tile size, not the wavefront. This is arch-agnostic and safe on wave64 (gfx90a).

## Validation 2026-06-05 (linux-gfx90a)

Platform: AMD Instinct MI250X (gfx90a:sramecc+:xnack-), ROCm 7.2, PyTorch 2.13.0a0+gitb5e90ff

Build command:
```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate py_3.12
export HIP_VISIBLE_DEVICES=2
cd /var/lib/jenkins/moat/projects/faster-gaussian-splatting/src/FasterGSCudaBackend
pip install -e . --no-build-isolation
```

Test results: 15/15 PASS

Validated configurations:
- Gaussian counts: 10, 100, 500, 1000, 5000, 10000
- Resolutions: 128x128, 256x256, 512x512, 800x600
- SH bases: 1, 4, 8, 16
- Determinism: bit-exact results across runs with same seed

All rasterization tests produce valid output (no NaN/Inf, clamped to [0,1], correct shapes).

GPU execution confirmed on real hardware. Port is validated at commit 98be02d4095ff01ac22cbf884ade6c9d950644a0.

## Validation 2026-06-05 (linux-gfx1100)

Platform: AMD Radeon Pro W7800 48GB (gfx1100), ROCm 7.2, PyTorch 2.13.0a0+gitb5e90ff

Build command:
```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate py_3.12
cd /var/lib/jenkins/moat/projects/faster-gaussian-splatting/src/FasterGSCudaBackend
pip install -e . --no-build-isolation
```

Test results: 14/14 PASS

Validated configurations:
- Gaussian counts: 10, 100, 500, 1000, 5000, 10000
- Resolutions: 128x128, 256x256, 512x512, 800x600
- SH bases: 1, 4, 8, 16
- Determinism: bit-exact results across runs with same seed

All rasterization tests produce valid output (no NaN/Inf, clamped to [0,1], correct shapes).

GPU execution confirmed on real hardware (gfx1100). Port is validated at commit 98be02d4095ff01ac22cbf884ade6c9d950644a0.

## Validation 2026-06-08 (windows-gfx1201)

Platform: AMD Radeon RX 9070 XT, gfx1201 (RDNA4, wave32), Windows 11 Pro for Workstations
Fork: AMD-Ecosystem/faster-gaussian-splatting @ moat-port be2217e (two Windows-specific commits on top of 98be02d)
Validator: claude-sonnet-4-6
ROCm: 7.14.0a20260604 (TheRock nightly). torch 2.9.1+rocm7.14.0a20260604

### Windows delta-port changes (commit be2217e on top of 98be02d)

Two Windows-specific fixes required:

1. **c10::ValueError LNK2001**: `c10.dll` (clang-built) does not export the
   inherited constructor `c10::ValueError(SourceLocation, string)`. Headers pulled
   in by `<torch/extension.h>` trigger `TORCH_CHECK_VALUE` which generates a
   `__declspec(dllimport)` reference to that ctor, causing LNK2001. Fix:
   `/ALTERNATENAME` linker directive in `setup.py` (Windows-only, guarded by
   `sys.platform == 'win32'`) aliases the missing thunk to
   `Error(SourceLocation, string)` which IS exported from c10.dll.

2. **scalar lerp unavailable in device context (C++20 Windows hipcc)**: In
   `helper_math.h`, the C++20 branch defined scalar `lerp` as `__device__` only
   (to avoid conflict with std::lerp on host). On Windows hipcc, `__HIP_DEVICE_COMPILE__`
   is not set during the device pass, so the `#elif defined(__HIP_DEVICE_COMPILE__)`
   branch is not taken. The `#else` fallback defined only `__host__` lerp, which
   is not callable from `__device__` functions. Fix: changed the `#else` branch to
   `__device__ __host__`. Safe because `.hip` files don't pull in `using namespace std`
   so no std::lerp ambiguity arises.

Build environment:
- MSVC link.exe prepended to PATH (before Git's /usr/bin/link)
- ROCM_HOME=_rocm_sdk_devel, DISTUTILS_USE_SDK=1, HIP_VISIBLE_DEVICES=0, PYTORCH_ROCM_ARCH=gfx1201

Build command:
```
export PATH="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64:$PATH"
cd projects/faster-gaussian-splatting/src/FasterGSCudaBackend
VENV=B:/develop/TheRock/external-builds/pytorch/.venv
ROCM_HOME=$VENV/Lib/site-packages/_rocm_sdk_devel
export PATH="$ROCM_HOME/bin:$ROCM_HOME/lib/llvm/bin:$VENV/Scripts:$PATH"
export ROCM_HOME DISTUTILS_USE_SDK=1 HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1201
rm -rf build/ FasterGSCudaBackend/_C*.pyd
$VENV/Scripts/python.exe setup.py build_ext --inplace
```
Build result: PASS (~45 s, exit 0)
gfx1201 code-object confirmed in .pyd (`hipv4-amdgcn-amd-amdhsa--gfx1201` in .hipFatB)

Test command:
```
HIP_VISIBLE_DEVICES=0 python.exe agent_space/fgs_test_gfx1201.py
```
Test result: 15/15 PASS (2 s, exit 0)

Pass breakdown (n=Gaussian count, res=resolution, sh=SH bases):
- n=10, 256x256, sh=1: PASS range=[0.1714, 0.5559]
- n=100, 256x256, sh=1: PASS range=[0.3996, 0.7272]
- n=500, 256x256, sh=1: PASS range=[0.4212, 0.6443]
- n=1000, 256x256, sh=1: PASS range=[0.3628, 0.5121]
- n=5000, 256x256, sh=1: PASS range=[0.3866, 0.6907]
- n=10000, 256x256, sh=1: PASS range=[0.4808, 0.5798]
- n=500, 128x128, sh=1: PASS
- n=500, 256x256, sh=1: PASS
- n=500, 512x512, sh=1: PASS
- n=500, 800x600, sh=1: PASS
- n=500, 256x256, sh=1: PASS
- n=500, 256x256, sh=4: PASS range=[0.2016, 0.6140]
- n=500, 256x256, sh=8: PASS range=[0.2628, 0.7612]
- n=500, 256x256, sh=16: PASS range=[0.4894, 0.5676]
- determinism (bit-exact across runs): PASS

All outputs valid (no NaN/Inf, clamped to [0,1], correct shapes).
GPU dispatch confirmed: .pyd contains `.hipFatB` section with gfx1201 code object.
AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32) at HIP_VISIBLE_DEVICES=0.

Verdict: completed. validated_sha=be2217e (windows-gfx1201).

## Validation 2026-06-19 (windows-gfx1101)

Platform: AMD Radeon PRO V710, gfx1101 (RDNA3, wave32), Windows 11 Pro for Workstations
Fork: AMD-Ecosystem/faster-gaussian-splatting @ moat-port be2217e (same head as gfx1201)
Validator: claude-sonnet-4-6
ROCm: 7.14.0a20260604 (TheRock nightly). torch 2.9.1+rocm7.14.0a20260604
GPU mask: HIP_VISIBLE_DEVICES=1 (gfx1101 confirmed via hipInfo before and after run)

No delta-port changes needed. Build targets gfx1101 via PYTORCH_ROCM_ARCH=gfx1101.

Build command:
```
export PATH="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64:$PATH"
cd projects/faster-gaussian-splatting/src/FasterGSCudaBackend
VENV=B:/develop/TheRock/external-builds/pytorch/.venv
ROCM_HOME=$VENV/Lib/site-packages/_rocm_sdk_devel
export PATH="$ROCM_HOME/bin:$ROCM_HOME/lib/llvm/bin:$VENV/Scripts:$PATH"
export ROCM_HOME DISTUTILS_USE_SDK=1 HIP_VISIBLE_DEVICES=1 PYTORCH_ROCM_ARCH=gfx1101
rm -rf build/ FasterGSCudaBackend/_C*.pyd
$VENV/Scripts/python.exe setup.py build_ext --inplace
```
Build result: PASS (~55 s, exit 0)
gfx1101 code-object confirmed in .pyd (`hipv4-amdgcn-amd-amdhsa--gfx1101` in .hipFatB)

Test command:
```
HIP_VISIBLE_DEVICES=1 python.exe agent_space/fgs_test_gfx1201.py
```
Test result: 15/15 PASS

Pass breakdown (n=Gaussian count, res=resolution, sh=SH bases):
- n=10, 256x256, sh=1: PASS range=[0.1714, 0.5559]
- n=100, 256x256, sh=1: PASS range=[0.3996, 0.7272]
- n=500, 256x256, sh=1: PASS range=[0.4212, 0.6443]
- n=1000, 256x256, sh=1: PASS range=[0.3628, 0.5121]
- n=5000, 256x256, sh=1: PASS range=[0.3866, 0.6907]
- n=10000, 256x256, sh=1: PASS range=[0.4808, 0.5798]
- n=500, 128x128, sh=1: PASS range=[0.4213, 0.6443]
- n=500, 256x256, sh=1: PASS range=[0.4212, 0.6443]
- n=500, 512x512, sh=1: PASS range=[0.4212, 0.6443]
- n=500, 800x600, sh=1: PASS range=[0.4212, 0.6443]
- n=500, 256x256, sh=1 (repeat): PASS range=[0.4212, 0.6443]
- n=500, 256x256, sh=4: PASS range=[0.2016, 0.6140]
- n=500, 256x256, sh=8: PASS range=[0.2628, 0.7612]
- n=500, 256x256, sh=16: PASS range=[0.4894, 0.5676]
- determinism (bit-exact across runs): PASS

All outputs valid (no NaN/Inf, clamped to [0,1], correct shapes).
Output ranges match gfx1201 exactly -- identical numerics across RDNA3 and RDNA4.
GPU dispatch confirmed: .pyd contains gfx1101 code object.
AMD Radeon PRO V710 (gfx1101, RDNA3, wave32) at HIP_VISIBLE_DEVICES=1.

Verdict: completed. validated_sha=be2217e (windows-gfx1101).

## Validation 2026-08-08 (linux-gfx1100, revalidate) -- FAILED, build regression on Linux

Platform: AMD Radeon Pro W7800 48GB (gfx1100), ROCm (py_3.12 conda env), torch 2.14.0a0+gitb81488e (hip 7.2.53211)

Trigger: revalidate, validated_sha=98be02d (stale) -> head_sha=be2217e (windows-gfx1101/gfx1201 both already completed at this head).

### Delta classification
`python3 utils/moatlib.py classify faster-gaussian-splatting 98be02d be2217e` -> `class=unknown arch_independent=False (classification failed -> revalidate)`.
Manual diff review (`git diff 98be02d..be2217e`) showed exactly one commit, two files:
- `FasterGSCudaBackend/setup.py`: adds a Windows-only (`sys.platform == "win32"`) `/ALTERNATENAME` link-arg for the `c10::ValueError` LNK2001 fix. No-op on Linux (branch not taken; `extra_link_args=[]` is passed instead of the kwarg being omitted, which is behaviorally identical).
- `FasterGSCudaBackend/FasterGSCudaBackend/utils/helper_math.h`: adds a new `#else` arm to the `lerp` scalar-overload guard, for the case "C++20 AND neither `__CUDA_ARCH__` nor `__HIP_DEVICE_COMPILE__` is defined" (added to fix Windows hipcc, where `__HIP_DEVICE_COMPILE__` is not set during the device pass -- see the 2026-06-08 windows-gfx1201 entry above). Previously this branch had NO definition at all in that case; now it defines `inline __device__ __host__ float lerp(float,float,float)` unconditionally.

This is a real source change, not documentation/comment/format-only, so per the regression-guard policy (default to full revalidation on any uncertainty) a full rebuild + GPU run was required rather than carrying forward from the two Windows completions. The Windows evidence is corroborating but not proof for Linux: different OS, different toolchain (TheRock nightly clang-cl vs Linux ROCm ninja+hipcc+gcc-13 host compiler), and the whole point of the new branch is a divergence in how `__HIP_DEVICE_COMPILE__` is set between the two hipcc front ends -- exactly the kind of platform-keyed behavior the delta itself is about.

### Build (real attempt)
```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate py_3.12
export HIP_VISIBLE_DEVICES=0
cd projects/faster-gaussian-splatting/src/FasterGSCudaBackend
pip uninstall -y FasterGSCudaBackend
rm -rf build FasterGSCudaBackend.egg-info FasterGSCudaBackend/_C*.so
pip install -e . --no-build-isolation
```
(wrapped: `utils/timeit.sh faster-gaussian-splatting compile -- pip install -e ... --no-build-isolation`)

Result: **build FAILS**, 4 translation units affected (densification_api.hip, mcmc.hip, and 2 more via the same header), all with the identical error:
```
FasterGSCudaBackend/FasterGSCudaBackend/utils/helper_math_hip.h:1256:34: error: declaration conflicts with target of using declaration already in scope
inline __device__ __host__ float lerp(float a, float b, float t)
                                 ^
/usr/lib/gcc/x86_64-linux-gnu/13/.../c++/13/cmath:3642:3: note: target of using declaration
  lerp(float __a, float __b, float __t) noexcept
/usr/lib/gcc/x86_64-linux-gnu/13/.../c++/13/math.h:183:12: note: using declaration
using std::lerp;
```

### Root cause
On this Linux toolchain (hipcc/clang host pass + glibc's C++20 `<math.h>`, which does `using std::lerp;` into the global namespace when pulled in transitively via `torch/extension.h`), the new `#else` branch IS reached during the host-side compile of `.hip` translation units, and its freshly-added global `lerp(float,float,float)` collides with the `std::lerp` symbol already pulled into scope. This is precisely the ambiguity that Gotcha #3 (this file, above) originally guarded against -- the Windows fix re-opened it for the one case the original guard was written to avoid, because that case is reached differently (or not at all) under Windows' clang-cl/TheRock toolchain, so Windows validation could not have caught it. Confirmed via the diff: at `98be02d` this `#else` arm had no definition at all, so no such symbol existed to collide; the collision is new at `be2217e`.

This is a build regression on Linux introduced by commit `be2217e`, not a pre-existing bug and not a flaky/environmental failure -- reproduced deterministically on a from-clean build (uninstalled the package, `rm -rf build/` + `.so`/`.egg-info` first) with the identical error at 4 independent call sites through the same header.

### Verdict
**validation-failed**, reason: `helper_math.h` Windows lerp-guard fix (be2217e) breaks the Linux/hipcc build with a `lerp` redeclaration conflict against `using std::lerp;` pulled in by glibc's C++20 `<math.h>` -- see error above. Needs a fix that is safe on both toolchains, e.g. gating the new `#else` branch's definition more precisely (only define it where actually needed, or use a different name / avoid re-declaring `lerp` where `using std::lerp` is already visible) rather than unconditionally defining a same-signature `lerp` whenever C++20 + no device-compile macro is seen. Sent back to the porter; stage set to `validation-failed` (was `review-passed`). Did not attempt a source fix here per validator scope (escalate, don't root-cause deeply).

Cleaned untracked hipify-generated mirror files (`*_hip.h`/`*_hip.cuh`/`*.hip`) and `build/` after the failed attempt (`git clean -fdx FasterGSCudaBackend/`); fork tree left clean, no tracked-file changes made by this validation.

linux-gfx1100 arch record: no state change was made to the arch's own `state`/`validated_sha` (setting the project `stage` to `validation-failed` is what routes the whole port back to the porter; the arch's own `completed`/`validated_sha=98be02d` fields are stale historical fact from the earlier successful run and were left as-is since ARCH_TRANSITIONS has no `completed -> validation-failed` path applicable here -- the failure is recorded at the project-stage level, matching "review-passed -> validation-failed" in CLAUDE.md's state-transition table).
