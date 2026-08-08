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

## Port fix 2026-08-08 (linux-gfx1100) -- scalar lerp guard, commit 6b18628

Fixes the Linux build regression reported in the entry above. Head moves
be2217e -> 6b18628, so windows-gfx1101 and windows-gfx1201 (both completed at
be2217e) go stale and must re-run; that is expected, the change is functional.

### What changed

`utils/helper_math.h`: the whole `#if __cplusplus < 202002L / #elif device / #else`
ladder around the scalar `lerp` is gone. In its place:

```c++
inline __device__ __host__ float lerp_scalar(float a, float b, float t)
{
    return a + t*(b-a);
}
#if __cplusplus < 202002L
inline __device__ __host__ float lerp(float a, float b, float t)
{
    return lerp_scalar(a, b, t);
}
#endif
```

`rasterization/include/kernel_utils.cuh`: the two calls in
`will_primitive_contribute` (the only scalar-lerp call sites in the tree) now say
`lerp_scalar`. The float2/float3/float4 `lerp` overloads are untouched.

README: added the AMD/ROCm build documentation, which the port had never carried
(Requirements bullet + a `PYTORCH_ROCM_ARCH=gfx1100 pip install ...` block beside
the existing CUDA pip command, in the README's own style).

### Why the old guards could not work, on either toolchain

The guard was keyed on `__CUDA_ARCH__ || __HIP_DEVICE_COMPILE__`, i.e. on *which
compiler pass is running*. The actual constraint is *whether `std::lerp` is
visible as an unqualified global name*, which is a property of the standard
library, not of the pass. Those two axes are independent, and each OS failed on a
different one:

- Windows (clang-cl + MSVC STL): MSVC declares `lerp` in namespace `std` only; it
  does not inject it into the global namespace. In the host pass
  (`__HIP_DEVICE_COMPILE__` unset) there was therefore NO viable
  `lerp(float,float,float)` at all, and parsing the device function body failed.
  That is what be2217e's `#else` arm was fixing.
- Linux (hipcc + libstdc++ 13): glibc/libstdc++ `<math.h>` does `using std::lerp;`
  at global scope, and `<torch/extension.h>` pulls it in. So in the host pass there
  was already a global `lerp(float,float,float)`, and be2217e's new definition is a
  redeclaration conflict ("declaration conflicts with target of using declaration
  already in scope") at 4 call sites.

Any guard that keys on the pass macro fixes one and breaks the other. Keying on
`_WIN32` / `_MSC_VER` would "work" but encodes the wrong axis (OS as a proxy for
standard library) and would break again on, say, libc++ on Windows.

### Why this is safe on Windows (for the Windows host to verify)

`lerp_scalar` is defined unconditionally as `__device__ __host__`. Consequences:

1. It exists in BOTH compiler passes on every toolchain, so the pass-macro
   divergence that motivated be2217e is no longer load-bearing anywhere. This is
   strictly stronger than be2217e's fix, which only supplied a definition in the
   pass where `__HIP_DEVICE_COMPILE__` happened to be unset.
2. `lerp_scalar` is not a name any standard library declares, so nothing can
   collide with it regardless of whether the STL injects names into the global
   namespace. The Linux failure mode cannot recur, and it cannot appear on
   Windows either if TheRock ever switches standard library.
3. It is `__device__ __host__`, so calling it from `__device__ inline
   will_primitive_contribute` is valid in the device pass as well; there is no
   host-only fallback left to be silently selected.
4. Numerics are unchanged: same expression `a + t*(b-a)`, same type, same call
   sites. Previously the Linux host pass resolved these calls to `std::lerp`
   (a *different* algorithm, the two-point-guaranteed form) while the device pass
   resolved them to the local one; only the device pass generates code, so the
   emitted math is identical to before on both OSes. Expect bit-identical
   rasterizer output vs the be2217e Windows runs.
5. The `#if __cplusplus < 202002L` scalar `lerp` is retained purely so the header
   keeps its documented interface for pre-C++20 consumers. This extension always
   builds with C++20 (setup.py forces `-std=c++20` / `/std:c++20`), so that arm is
   not compiled here on any platform, Windows included.

The only Windows-visible risk is a build error, not a behavior change; there is no
code path where Windows takes a different definition than Linux does.

### Build + test (linux-gfx1100)

```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate py_3.12
export HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1100
pip install -e projects/faster-gaussian-splatting/src/FasterGSCudaBackend --no-build-isolation
```
Build: PASS, 45.2 s (from clean: no `build/`, no `_C*.so`). AMD Radeon Pro W7800
48GB (gfx1100), ROCm/torch 2.14.0a0+gitb81488e (hip 7.2.53211). Extension contains
exactly one code object, `amdgcn-amd-amdhsa--gfx1100`.

Test (`PYTHONPATH=.../src/FasterGSCudaBackend`, script kept at
scratch `fgs_test.py`, same shape as the earlier Windows script plus one case):
16/16 PASS -- n = 10/100/500/1000/5000/10000 at 256x256; 128x128, 256x256,
512x512, 800x600 at n=500; SH bases 1/4/8/16; bit-exact determinism across two
runs; and n=20000 spread over a 800x600 frame with sh=16 to drive the tile-overlap
path where `lerp_scalar` is actually called. All outputs finite, correctly shaped,
within [0,1]. The 256x256 sh=1 series reproduces the gfx1201/gfx1101 Windows
numbers to 4 decimals (e.g. n=10 range [0.1714, 0.5558] vs [0.1714, 0.5559]).

Note the sh=4/8/16 and non-256x256 ranges differ from the Windows log because this
script centers the principal point at width/2, height/2 while the Windows script
pinned it at (128,128) for every resolution. Same code, different scene.

### Gotcha #7

Do not guard a helper-function definition on `__CUDA_ARCH__` /
`__HIP_DEVICE_COMPILE__` when the thing you are avoiding is a NAME COLLISION with
the standard library. The pass macro answers "am I compiling for device", not "is
this name already taken", and the two standard libraries in the fleet answer the
second question differently. Give the helper a name the standard library does not
use and define it unconditionally as `__device__ __host__`. Promoted to the
`cuda-to-rocm` skill (references/fault-classes.md, "Headers, includes and build").

## Review 2026-08-08 (linux-gfx1100) -- changes-requested

Reviewed `moat-port` 6b18628 (on top of be2217e) against the fork's `main`.
Review PR: https://github.com/AMD-Ecosystem/faster-gaussian-splatting/pull/1
(review-only, never merged; carries the title and body the upstream PR will use).

### Blocking: 6b18628 commits 26 hipify-generated files

5,871 of the commit's 5,903 added lines are build output: `utils/helper_math_hip.h`,
`utils/utils_hip.h`, the six `rasterization/include/*_hip.h`, the six
`rasterization/include/*_hip.cuh`, the three `densification/include/*_hip.*`, and the
nine `*/src/*.hip`. These are what torch's hipify writes at build time
(`torch/utils/cpp_extension.py` -> `hipify_python.hipify(..., is_pytorch_extension=True)`;
`hipify_python.get_hip_file_path` is the `.cu` -> `.hip` and `.h`/`.cuh` -> `_hip.h`/`_hip.cuh`
naming). They were untracked at be2217e (`git ls-tree -r be2217e | grep -c _hip` -> 0) and the
2026-08-08 validation entry above records deleting them with `git clean -fdx`.

The port's real footprint is 6 files / 88 lines; as committed the branch reads as 29 files /
5,903 lines, most of it a machine translation of the maintainer's own kernels sitting beside
the originals. It is also a staleness trap: hipify rewrites those files in place on every
build, so they are byte-identical today (verified: a from-clean `setup.py build_ext --inplace`
leaves `git status` clean) but the next source edit dirties tracked files, which either get
committed mismatched or mask a dirty tree from the integrity gate.

Fix: no arch validated 6b18628, so amend it (or `git rm --cached` in a follow-up) down to the
six real files, and add `*.hip`, `*_hip.h`, `*_hip.cuh` to `.gitignore` beside the `*.so` and
`build/` entries it already has. Nothing here is a hand-written `.hip`. Re-check the commit
message afterwards; it already describes the three-file change it should have been.

### The lerp fix itself: correct, no changes needed

Verified independently rather than from the porter's summary.

- `lerp_scalar` (helper_math.h:1246) is defined once, unconditionally, `__device__ __host__`.
  It therefore exists in both compiler passes on every toolchain, which is what the
  `__HIP_DEVICE_COMPILE__`-keyed version could not guarantee, and no standard library declares
  that name, so neither the Linux collision nor the Windows missing-overload failure can recur.
  Strictly stronger than be2217e, not a trade of one OS for the other. Windows re-validation is
  still the evidence, but the argument does not depend on it.
- helper_math.h:1256-1268: the float2/float3/float4 overloads take parameter types `std::lerp`
  does not, so they never participated. Untouched, correctly.
- kernel_utils.cuh:84-85 are the only scalar-lerp call sites in the tree. `git grep lerp` over
  every tracked `.cu`/`.cuh`/`.h`/`.cpp` (excluding generated mirrors) finds no other caller;
  the vector overloads have no callers at all.
- Not keying on `_WIN32`/`_MSC_VER` is right, and is the generalizable part: the axis is which
  standard library is in scope, not which OS.
- README: accurate (`PYTORCH_ROCM_ARCH` is read by torch's `_get_rocm_arch_flags`; setup.py's
  `IS_ROCM` comes from `torch.version.hip`, so the documented command does build for ROCm) and
  it matches the file's own paragraph-per-line style with no hard breaks.
- Standing rules clean: `jargon.py --commits be2217e..6b18628` clean, ASCII only, no em-dash,
  title 52 chars with the `[ROCm]` prefix, Claude named in the body, no noreply trailer.
- Promotion confirmed: `references/fault-classes.md` "Headers, includes and build" states the
  rule generally (pass macro vs name visibility), names the two standard libraries and both
  failure directions, lists the exposed C++20 names, and rejects `_WIN32` as a proxy. Right
  file, right altitude.

### Non-blocking correction to the fix write-up

The claim that the `#if __cplusplus < 202002L` arm (helper_math.h:1250) is dead "on any
platform, Windows included" holds for the ROCm build only. be2217e's Windows failure is itself
the proof: it only occurs if the C++20 branch was taken in the Windows host pass, so hipcc's
clang reports `__cplusplus >= 202002L` there. Under nvcc on Windows, cl.exe reports
`__cplusplus == 199711L` unless `/Zc:__cplusplus` is passed and setup.py does not pass it, so
the arm is live in the nvcc host pass. That is harmless -- the MSVC standard library declares
`lerp` in namespace `std` only, so nothing collides, and the arm restores exactly the
definition the header carried before this series. No code change wanted; the reasoning is what
needed correcting, since "setup.py forces C++20 so `__cplusplus` is 202002" is not true of
MSVC.

### Verified while reviewing, no action

- The C++17 -> C++20 bump in setup.py is required by PyTorch itself, not by ROCm: compiling
  `bindings.cpp` against this torch at `-std=c++17` fails with `torch/all.h:5: #error C++20 or
  later compatible compiler is required to use PyTorch`. It fixes the CUDA build on modern
  torch too, and it is also what exposes the `std::lerp` collision at all (C++17 has no
  `std::lerp`), which means upstream's own CUDA build hits the same conflict the moment they
  bump. Worth saying in the upstream PR body, and it is.
- `sys.platform == "win32"` gates the `/ALTERNATENAME` link arg on OS rather than on vendor, so
  it applies to CUDA/Windows too. Correct as written: the missing `c10::ValueError` ctor export
  is an MSVC/torch defect, not an AMD one, and `/ALTERNATENAME` is inert when the symbol
  resolves.
- `kernels_mcmc.cuh:20` `rsqrt` -> `1 / std::sqrt` is in `init_relocation_coefficients()`,
  which is host code, so no device numerics change.
- No new wavefront assumptions. The `cg::tiled_partition<32>` / `warp.ballot()` /
  `previous_lanes_mask` code is untouched upstream code, tile-relative throughout, and already
  validated on wave64 (gfx90a).
