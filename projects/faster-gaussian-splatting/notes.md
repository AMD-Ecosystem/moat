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
   keeps its documented interface for pre-C++20 consumers. On the ROCm build that
   arm is dead on every platform: setup.py passes `-std=c++20` / `/std:c++20` and
   hipcc's clang reports `__cplusplus >= 202002L` accordingly, on Linux and on
   Windows alike (be2217e's Windows failure is itself the proof, since it can only
   have occurred if the C++20 branch was taken in the Windows host pass).

   CORRECTION (reviewer, 2026-08-08): the earlier wording here said the arm was
   dead "on any platform, Windows included", which is true of the ROCm build only.
   Under nvcc on Windows the arm is LIVE: cl.exe reports `__cplusplus == 199711L`
   unless `/Zc:__cplusplus` is passed, and setup.py does not pass it, so
   `/std:c++20` alone does not move the macro in the MSVC host pass. That is
   harmless -- the MSVC standard library declares `lerp` in namespace `std` only
   and never injects it into the global namespace, so nothing collides, and the arm
   restores exactly the definition the header carried before this series. No code
   change wanted; only the reasoning needed correcting, because "setup.py forces
   C++20, therefore `__cplusplus` is 202002" is not true of MSVC.

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

## Port fix 2026-08-08 (linux-gfx1100) -- drop committed hipify output, commit 932a8b7

Addresses the blocking finding in the review above. Head moves 6b18628 -> 932a8b7.

### Amended, not a new commit on top

Checked first: no arch had `validated_sha == 6b18628`. At the time of the fix the
four records were linux-gfx90a and linux-gfx1100 at 98be02d, windows-gfx1101 and
windows-gfx1201 at be2217e. Nothing validated the commit being rewritten, so
amending orphans nothing and every `validated_sha` stays a reachable ancestor; the
regression guard can still classify 98be02d..932a8b7 and be2217e..932a8b7. Had any
arch validated 6b18628, a new commit on top would have been mandatory instead.
Pushed with `--force-with-lease=moat-port:6b18628...`.

### What changed

`git rm --cached` on the 26 hipify-generated files, plus three patterns in the
project's existing `.gitignore` (`*.hip`, `*_hip.h`, `*_hip.cuh`, placed with the
other extension globs beside `*.so`). The tree delta from 6b18628 to 932a8b7 is
exactly those 26 deletions and the `.gitignore` edit; `helper_math.h`,
`kernel_utils.cuh` and `README.md` are byte-identical, so the reviewed and accepted
`lerp_scalar` fix is untouched.

    git diff --name-status 6b18628 932a8b7   # 26 D + M .gitignore, nothing else

Commit footprint: 4 files, +23 -18 (was 29 files, +5903 -18). The branch against
the fork's `main` is now 7 files, +91 -15.

The commit message gained one clause about ignoring the translated files; the rest
already described the three-file change it should always have been.

### Verification (from clean, GPU 0)

```bash
git clean -fdx                                     # removed exactly the 26, nothing else
source /opt/conda/etc/profile.d/conda.sh && conda activate py_3.12
export HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1100
pip uninstall -y FasterGSCudaBackend
rm -rf build/ *.egg-info FasterGSCudaBackend/_C*.so
pip install -e . --no-build-isolation
```
Build: PASS. All 26 files regenerated by the build and all 26 are ignored.
`git status --porcelain` prints NOTHING afterwards -- no tracked file modified, no
generated file untracked-and-unignored. `llvm-objdump --offloading` on the built
`_C.cpython-312-x86_64-linux-gnu.so` shows 7 bundles, all
`hipv4-amdgcn-amd-amdhsa--gfx1100`. Extension imports on AMD Radeon Pro W7800 and
exposes all 8 entry points (adam_step, add_noise, backward, forward, inference,
pruning_scores, relocation_adjustment, update_3d_filter). Full GPU numerics were
not re-run here: the tree differs from the already-tested 6b18628 only by
`.gitignore`, which the compiler never reads.

`jargon.py --commits be2217e..HEAD`: clean.

### Open item for whoever prepares the upstream PR (CLOSED 2026-08-08 -- see "Port 2026-08-08" below)

`jargon.py --commits origin/main..HEAD` reports one hit that is NOT in this commit:
the base commit 98be02d's message says "Strategy B (torch hipify): ...". That is
in-house vocabulary in upstream-visible text and must not ship. It was left alone
deliberately -- rewording it means rebasing 98be02d and be2217e, which are the
`validated_sha` of all four archs, and the standing rule is not to rewrite a commit
an arch has validated. The sanctioned fix is the optional pre-PR squash: collapse
`moat-port` to a single commit with a clean message, then
`moatlib.py squash-carry-forward`. Re-run `jargon.py --commits origin/main..HEAD`
before opening upstream, not just the range of the newest commit -- scoping the
check to the last commit is how this survived a review.

### Gotcha #8

Torch's build-time hipify writes its output as sibling files IN THE SOURCE TREE,
so a `git add -A` after a build commits a
machine translation of the project's own kernels. Worse, because hipify rewrites
them in place, a tracked mirror looks clean on a from-clean build and only dirties
the tree on the NEXT source edit -- which is exactly when the integrity gate is
supposed to be trustworthy. Gitignore the artifacts before the first commit, and
verify with `git clean -fdx` + build + an EMPTY `git status --porcelain`;
checking only "no tracked file modified" passes even when the artifacts are
tracked. `*.hip`, `*_hip.h`, `*_hip.cuh` is the whole list HERE only because no
path in this project contains `cuda`/`CUDA`/`THC`: `get_hip_file_path` appends
`_hip` only when its `cuda`->`hip` rewrite changes neither the directory nor the
filename, and otherwise renames (`src/cuda/foo.cuh` -> `src/hip/foo.cuh`,
`cuda_utils.cuh` -> `hip_utils.cuh`). Promoted to the `cuda-to-rocm` skill
(references/strategy-b-torch.md, "Never commit the hipified mirror"), with the
rename cases tabulated there.

## Review 2026-08-08 (re-review of 932a8b7, linux-gfx1100)

Focused re-review of the fix for the blocking finding above. The accepted content
(`lerp_scalar`, README, the three ROCm guards) was confirmed byte-identical and is not
re-litigated.

### Confirmed

- `git diff --name-status 6b18628 932a8b7` is exactly 26 `D` plus `M .gitignore`. Commit
  932a8b7 is 4 files, +23 -18; `origin/main...HEAD` is 7 files, +91 -15. Both numbers as
  claimed. No `*.hip` / `*_hip.*` path is tracked anywhere in the branch.
- **Amend safety.** No arch had `validated_sha == 6b18628`: linux-gfx90a and linux-gfx1100
  at 98be02d, windows-gfx1101 and windows-gfx1201 at be2217e. `git merge-base --is-ancestor`
  puts both of those shas under 932a8b7, so every `validated_sha` is still reachable and the
  regression guard can still classify. 6b18628 is orphaned and nothing pointed at it. The
  amend was legitimate.
- **The ignore set is exact, not over-broad.** `git clean -fdxn` on a built tree lists 27
  paths: the 26 hipify outputs plus `_C.cpython-312-x86_64-linux-gnu.so`, which upstream's
  own `*.so` already covered. Counts line up with the sources: 9 `.cu` -> 9 `.hip`,
  7 `.cuh` -> 7 `_hip.cuh`, 10 of the 17 `.h` -> 10 `_hip.h`. No tracked path contains
  `cuda`/`CUDA`/`THC` (case-sensitive), so `get_hip_file_path` appends `_hip` for every file
  here rather than renaming, and the three patterns cover the whole generated set.
- **The stronger verification holds now, independently.** The tree has been built (the
  regenerated `.hip`/`_hip.*` files and the `.so` are on disk) and `git status --porcelain`
  prints nothing. That is the empty-status check, not the weaker "no tracked file modified"
  that a tracked mirror also passes.
- Commit hygiene on 932a8b7: `[ROCm]` prefix, 52-char title, Claude named, no
  `Co-Authored-By` / noreply trailer, ASCII only, no em-dash, Test Plan with literal
  commands. `jargon.py --diff origin/main...HEAD` clean.
- The `__cplusplus < 202002L` correction now reads correctly (dead on the ROCm build; live
  in the nvcc host pass on Windows because setup.py does not pass `/Zc:__cplusplus`).

### Blocking: the deferred jargon hit has no mechanism behind it, and its window is closing

The verdict to not rebase 98be02d today was right in isolation, but the premise it rests on
is no longer true, and the recorded handoff points at a moment where the fix is no longer
free.

**Nothing mechanical will ever catch it.** `utils/upstream.py:494-499` (`publish_blockers`)
and `utils/upstream.py:429-433` (`open_review_pr`) run `jargon.scan_text` over the PR
**title and body only**. No gate anywhere scans the commit range, and nothing on the publish
path reads notes.md. So "Strategy B (torch hipify)" in 98be02d's message ships upstream
silently. This is not a note-and-move-on item; it is a standing-rule violation with no
backstop.

**Rewriting those commits is not the cost it was.** `moatlib.py pr-ready` reports all four
archs blocking: head is 932a8b7 and every `validated_sha` is 98be02d or be2217e. The delta
spans the Windows link fix and the `lerp_scalar` rename, which `advance_head` will not
classify as arch-independent-inert, so all four archs must revalidate at head regardless of
what happens to the history. Rewriting 98be02d and be2217e right now therefore forfeits no
validation that is not already forfeit -- the standing rule protects a `validated_sha` that
is still load-bearing, and neither of these is.

**The recorded plan names the wrong moment.** The note hands the job to "whoever prepares
the upstream PR", i.e. after approval. By then the review PR (AMD-Ecosystem#1, currently at
932a8b7 with no approval on it, only a COMMENTED review) has been approved against a
specific commit, and `moatlib.approval_currency` (utils/moatlib.py:813-816) refuses with
`stale-commits` the moment the tip differs from the approved sha. A squash after approval
invalidates the approval and sends it back to the person who gave it.
`squash_carry_forward` also must not run before revalidation: it advances every `completed`
arch to the new sha on tree-identity alone, which today would falsely certify the
`lerp_scalar` fix as validated on gfx90a and both Windows archs.

Required, before the validator runs: remove the in-house vocabulary from the branch's commit
messages. How is open -- squash the three commits into one, or reword 98be02d alone and
replay be2217e -- but do it now, while no arch is validated at head and no approval exists.
Then `python3 utils/jargon.py --commits origin/main..HEAD -C projects/faster-gaussian-splatting/src`
must be clean over the whole branch, and the validator validates the final sha once. Update
the "Open item" section to record what was done instead of what to do later.

### Also required

- `.claude/skills/cuda-to-rocm/references/strategy-b-torch.md:12` states the naming rule as
  `.h`/`.cuh` -> `_hip.h`/`_hip.cuh`. That is only the case this project hit.
  `get_hip_file_path` appends `_hip` **only when the cuda->hip substring rewrite leaves both
  the directory and the filename unchanged**; otherwise it renames, so `src/cuda/foo.cuh`
  becomes `src/hip/foo.cuh` and `cuda_utils.cuh` becomes `hip_utils.cuh` -- neither matched
  by any of the three patterns. `cuda/` directories and `cuda`-prefixed filenames are common
  in CUDA projects, so a porter who copies the three lines into such a project and skips the
  verification step gets a silently incomplete ignore list. Add the condition to that
  paragraph. The rest of the section is the right file and the right altitude.
- The meta-lesson (scoping the jargon check to the newest commit is how this survived a
  review) is recorded only in this project's notes, which is where it will die.
  `.claude/skills/pr-review/review-checklist.md:41` and `.claude/agents/validator.md:22`
  both already say `--commits <base>..HEAD`; the porter, the prior reviewer and the validator
  all read `<base>` as the previous head. Make it explicit in the pr-review checklist -- the
  base is the fork's default branch, the whole branch, never the previous head -- since that
  checklist is the one a reviewer actually runs.

### Verdict

Changes requested. The artifact fix itself is correct and complete; what blocks is the
jargon item, which is free to fix today and stops being free once validation or approval
lands on this branch.

## Port 2026-08-08 (history rewrite, linux-gfx1100)

Addresses the blocking finding of the re-review above: in-house vocabulary in the
branch's commit messages, which nothing on the publish path would have caught
(`upstream.py` scans the PR title and body only, never the commit range).

**What was done.** Squashed the three commits (98be02d, be2217e, 932a8b7) into one:
`git reset --soft 7bc0593` (the merge base with `origin/main`) followed by a single
commit, so the index -- and therefore the tree -- was never touched. New sha
`1b3716140ae6c6e01133d6aba8ad2ca380a9800c`, pushed with
`--force-with-lease=moat-port:932a8b7...`.

Squash rather than reword-and-replay because the second and third commits are fix-ups
of the first that never existed upstream, and 932a8b7's message narrated that internal
history ("Previous attempts keyed the definition on __CUDA_ARCH__ ..."), which reads as
a reference to nothing once the branch is a single commit. Squashing also removed
be2217e's model-name credit and gave the Windows link fix a Test Plan it lacked.

**Verification.**

```
git diff --stat 932a8b7 HEAD          # empty: tree identical, only history changed
python3 utils/jargon.py --commits origin/main..moat-port -C projects/faster-gaussian-splatting/src   # clean
python3 utils/jargon.py --diff origin/main...moat-port -C projects/faster-gaussian-splatting/src     # clean
git status --porcelain                # empty (built tree, artifacts ignored)
```

The whole-branch range is the one that matters: `origin/main..`, not the previously
reviewed head. GPU numerics were NOT re-run and did not need to be -- the tree is
byte-identical to 932a8b7, so the 16/16 gfx1100 result recorded above stands unchanged.

**Consequence for validation.** All four archs were already going to revalidate at head:
their `validated_sha` was 98be02d or be2217e, and the delta to head spans the Windows
link fix and the `lerp_scalar` rename, which `advance_head` will not classify as
arch-independent-inert. The rewrite therefore forfeited no validation that was not
already forfeit. The old shas are now unreachable, so `_classify_safe` returns None and
every arch falls to the safe default, which is the intended outcome. Do NOT run
`squash-carry-forward` here: it certifies on tree-identity alone and would falsely mark
the `lerp_scalar` fix validated on gfx90a and the two Windows archs.

**Skill correction shipped with this.** `references/strategy-b-torch.md` stated the
hipify output naming as a plain `_hip` suffix. `get_hip_file_path` appends `_hip` only
when its `cuda`->`hip` / `CUDA`->`HIP` / `THC`->`THH` rewrite leaves BOTH the directory
and the filename unchanged; otherwise it renames, so `src/cuda/kernel.cuh` becomes
`src/hip/kernel.cuh` and `cuda_utils.cuh` becomes `hip_utils.cuh`. The section now
tabulates both behaviours and tells the porter to derive the extra ignore entries from
the project's own paths, with the empty-`git status` check after `git clean -fdx` kept
as the thing that actually catches a rename nobody predicted. The pr-review checklist
and validator agent now state that the jargon `<base>` is the fork's default branch,
never the previously reviewed head -- scoping that check to the newest commit is how
this hit survived a review.

## Review 2026-08-08 (re-review of 1b37161 after the history rewrite, linux-gfx1100)

Narrow re-review: the round changed history only. The accepted substance (`lerp_scalar`,
README, hipify-artifact cleanup) is not re-litigated.

### Confirmed

- **The tree is unchanged, independently verified.** `git rev-parse 932a8b7^{tree}` and
  `git rev-parse HEAD^{tree}` are both `47d40ae3d821b880eba1248117f96eaff36d1851`, and
  `git diff --exit-code 932a8b7 HEAD` exits 0. Identical tree objects are a stronger
  statement than an empty diff (no mode, no rename, no submodule delta can hide in one),
  so not re-running the numerics was correct: a byte-identical tree compiled by the same
  toolchain produces the same code object, and the recorded 16/16 gfx1100 result carries.
- **The rewrite was safe.** The branch forked at `7bc0593`, so `git reset --soft 7bc0593`
  landed on the same base it already had rather than moving it. No arch had
  `validated_sha == 932a8b7`; all four were at `98be02d` / `be2217e`, already stale
  against head, and the delta to head is functional, so all four owed a revalidation
  either way and the rewrite forfeited nothing. `origin/moat-port` and local `HEAD` both
  read `1b37161`; the force-with-lease landed.
- **`squash-carry-forward` was not run.** status.json still carries the old
  `validated_sha` values and no `carry_forward` block on any platform. Correct: on
  tree-identity alone it would have certified the `lerp_scalar` fix as validated on
  gfx90a and both Windows archs, which never ran it.
- **Jargon is clean over the whole branch.** `jargon.py --commits origin/main..moat-port`
  and `--diff origin/main...moat-port` both report clean.
- **Commit hygiene.** Title 39 chars with the `[ROCm]` prefix, Claude named, no
  `Co-Authored-By` / noreply trailer, ASCII only, no em-dash, Test Plan with literal
  commands in fenced blocks, no in-house vocabulary, no branch-internal narration, no
  AMD-internal account references.
- **The wavefront claim in the commit message is right.** Checked against the ROCm 7.2
  headers rather than taken on trust: `thread_block_tile_base::ballot`
  (`amd_hip_cooperative_groups.h:868`) builds a tile mask from
  `(thread_rank() % warpSize) / numThreads` and passes it through
  `internal::helper::adjust_mask` (`hip_cooperative_groups_helper.h:113`), which COMPACTS
  the wave-relative mask down to tile-relative bits. So on wave64 a `tiled_partition<32>`
  ballot still returns a 32-bit tile-relative mask; storing it in a `uint`
  (kernels_forward.cuh:283), `__popc`-ing it, `__fns(mask, 0, n+1)`
  (kernels_forward.cuh:295, and HIP's `__fns` at `amd_device_functions.h:149` takes a
  32-bit mask) and `previous_lanes_mask = (1 << lane_idx) - 1` are all correct at both
  widths. `meta_group_rank()` is parent-relative, so `warp_start = warp_idx * 32` indexes
  the block-sized shared arrays correctly on wave64 too. No hardcoded-32 fault.
- **The .gitignore set is complete, reproduced from scratch.** Copied the tree to a
  scratch dir and ran the hipify pass under a ROCm torch: 26 generated files, every one
  matched by `*.hip` / `*_hip.h` / `*_hip.cuh`, and `git status --porcelain` prints
  nothing. `bindings.cpp` produces no mirror at all (hipify changes nothing in it, so
  `preprocess` returns `[skipped, no changes]` and writes no file), which is why the
  absence of a `*_hip.cpp` pattern is not a gap here.

### Blocking: the branch is based on pre-fix upstream code in the function it edits

`moat-port` forked at `7bc0593` and is 2 commits behind the fork's `main`, which is
byte-identical to upstream `nerficg-project/faster-gaussian-splatting@ae2bf80` (checked
via `gh api`). One of those two commits is not incidental:

    44a13d1  2026-07-11  Fix bug in tile-based culling logic

It changes `will_primitive_contribute` in
`FasterGSCudaBackend/FasterGSCudaBackend/rasterization/include/kernel_utils.cuh` -- the
same function, the same file, ten lines above the port's only kernel edit at
`kernel_utils.cuh:84-85`. Upstream turned `x_min_diff > 0.0f` into `>= 0.0f` (and the
same for `y_min_diff`) because a splat whose 2d mean lands bit-exactly on the pixel
centre at a tile's left or upper boundary was being culled from that tile.

The hunks do not textually collide, so `git merge-tree` reports zero conflicts and
`git diff origin/main...HEAD` (merge-base relative) shows nothing wrong. That is exactly
why three reviews have walked past it. The check that surfaces it is
`git rev-list --count moat-port..origin/main`.

Why it blocks now rather than later:

- The validation this port rests on was measured against the OLD culling predicate. The
  commit message's Test Plan specifically claims "a 20000-Gaussian spread scene that
  drives the tile boundary path where lerp_scalar is called" -- the tile boundary path is
  precisely what 44a13d1 changed, so the one test named as covering that path exercises
  semantics that no longer exist upstream. The bit-exact-repeatability and
  four-decimal cross-arch claims are on the same footing.
- Rebasing changes head_sha. Do it now and it costs nothing, because no arch is validated
  at head and no approval exists. Do it after the validator runs and all four archs
  revalidate a second time; do it after approval and `approval_currency` refuses with
  `stale-commits`. This is the same window argument that forced the squash last round, and
  it is open for the same reason.

Required: rebase `moat-port` onto `origin/main` (`ae2bf80`) before the validator runs,
confirm `git rev-list --count moat-port..origin/main` is 0, re-check both `jargon.py`
forms, and let the validator validate the rebased sha once.

While the commit is being touched anyway, one wording fix: the message opens with
"PyTorch's build system translates the .cu and .cuh sources to HIP as part of the build".
It also translates plain `.h` headers -- 10 of the 17 in this tree -- which the same
message's own `.gitignore` paragraph acknowledges with `*_hip.h`. "the .cu, .cuh and .h
sources" removes the inconsistency.

### Also required, but as a PR against moat main, not on this branch

`references/strategy-b-torch.md`, `references/fault-classes.md`,
`.claude/skills/pr-review/review-checklist.md` and `.claude/agents/validator.md` are
global files carrying lessons every agent needs, and on `port/faster-gaussian-splatting`
they are stranded until this port merges. They belong in a PR off `origin/main`.

The rewritten hipify naming rule in `strategy-b-torch.md` was checked line by line against
`torch/utils/hipify/hipify_python.get_hip_file_path` (torch 2.14.0a0). The stated
condition -- `_hip` is appended only when the `cuda`->`hip` / `CUDA`->`HIP` / `THC`->`THH`
rewrite leaves both the directory and the full filename unchanged -- is exactly the source
(`if is_pytorch_extension and dirpath == orig_dirpath and (root + ext) == orig_filename`),
and all six table rows reproduce. Three corrections to fold into that PR:

- `strategy-b-torch.md:14` and `:35`. The rewrite is a plain substring `replace` over the
  whole dirpath string, not a match on directory components, so `src/cuda_kernels/foo.cuh`
  becomes `src/hip_kernels/foo.cuh`. The derivation at :35 says "a source directory named
  `cuda`/`CUDA`", which reads as exact-match and under-covers that case -- the same class
  of incomplete ignore list the section exists to prevent. Say "a directory component
  CONTAINING `cuda`/`CUDA`/`THC`".
- `strategy-b-torch.md:14-21`. A mirror is written only when hipify actually CHANGES the
  file's content. `preprocess_file_and_save_result` returns `[skipped, no changes]` and
  writes nothing when the output is identical and the dirpath did not move
  (`hipify_python.py:966-974`), so a source hipify leaves alone produces no mirror at all.
  That is why this tree's 17 `.h` files yielded only 10 `_hip.h`, and why `bindings.cpp`
  yielded nothing. As written the table reads as unconditional, which over-predicts the
  ignore list -- harmless, but it also hides the reason the counts never line up.
- `strategy-b-torch.md:27`. The line counts are wrong. `git diff --numstat 7bc0593
  6b18628` gives 26 generated files at +5883 and 6 real files at +88/-15, total +5971.
  The correct sentence is "26 such files, 5,883 of 5,971 added lines, for an 88-line
  port".

### Verdict

Changes requested. The history rewrite itself is clean and complete -- tree identical,
jargon gone, no validation forfeited, `squash-carry-forward` correctly left alone. What
blocks is that the branch is not based on the code it will be merged into, in the one
function it edits, and the cost of fixing that only goes up from here.

## Port fix 2026-08-08 (linux-gfx1100) -- rebase onto fork main, commit b0d21d5

Addresses the blocking finding of the re-review above. Head moves
1b37161 -> b0d21d5 (`--force-with-lease=moat-port:1b37161`, no bare force).
No arch had `validated_sha == 1b37161` (all four were at 98be02d / be2217e,
already stale), so the rewrite forfeited nothing that was not already forfeit.

### The rebase

`git rebase origin/main moat-port` applied cleanly, no conflict. Afterwards
`git rev-list --count moat-port..origin/main` is 0, and the branch now sits on
ae2bf80, which is byte-identical to upstream nerficg-project@ae2bf80.

The two commits picked up:

- `44a13d1` "Fix bug in tile-based culling logic" -- rewrites
  `will_primitive_contribute` in `rasterization/include/kernel_utils.cuh`,
  turning `x_min_diff > 0.0f` into `>= 0.0f` and the same for `y_min_diff`, so a
  splat whose 2d mean lands bit-exactly on the pixel centre at a tile's left or
  upper boundary is no longer culled from that tile.
- `ae2bf80` "Allow disabling diffuse/specular color during rendering" -- Python
  only (Model.py, Renderer.py, utils.py, fastergs_garden.yaml). No CUDA source,
  so it cannot interact with the port.

### The rename and the upstream boundary change coexist

Both edits are in the same function and both survived, at their own lines:

    72:  const float x_min_diff = rect_min.x - mean.x;
    73:  const float x_left = static_cast<float>(x_min_diff >= 0.0f);   <- upstream 44a13d1
    ...
    84:      lerp_scalar(rect_max.x, rect_min.x, x_left),               <- port
    85:      lerp_scalar(rect_max.y, rect_min.y, y_above)               <- port

`grep -rn lerp` over the tree (excluding the hipify mirrors and helper_math.h
itself) returns exactly those two lines, so every scalar-lerp call site in the
new upstream code is renamed and none was missed. The float2/float3/float4
`lerp` overloads are untouched, as before. `git diff --stat origin/main..HEAD`
is the same 7 files / +91 / -15 as before the rebase.

### GPU re-run, before versus after (linux-gfx1100, HIP_VISIBLE_DEVICES=0)

The rebase changes the tree, so the earlier 16/16 no longer carried. Rebuilt
from clean and re-ran; then, to measure the upstream semantic change directly,
built 1b37161 (pre-rebase) from clean with the same toolchain and compared the
output tensors bit for bit against the rebased build.

```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate py_3.12
export HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1100
rm -rf FasterGSCudaBackend/build FasterGSCudaBackend/FasterGSCudaBackend/_C*.so
pip install -e projects/faster-gaussian-splatting/src/FasterGSCudaBackend --no-build-isolation
PYTHONPATH=.../src/FasterGSCudaBackend python agent_space/fgs_test.py
```

Build: PASS from clean, AMD Radeon Pro W7800 48GB (gfx1100), torch
2.14.0a0+gitb81488e (hip 7.2.53211).
Tests: **16/16 PASS**, same 16 cases as the pre-rebase run (Gaussian-count
sweep 10..10000 at 256x256; 128x128 / 256x256 / 512x512 / 800x600 at n=500;
SH bases 1/4/8/16; the 20000-Gaussian spread scene at 800x600 sh=16; bit-exact
determinism across two runs). All outputs finite, correctly shaped, in [0,1].

Bit-for-bit before/after over 13 rendered scenes:

| scene | before (1b37161) vs after (b0d21d5) |
| --- | --- |
| 12 of 13 scenes | bit-identical |
| n=20000, 800x600, sh=16 (tile-overlap) | 30 differing pixels of 480,000, max abs delta 1.19e-07 (1 ULP), 3 tiles touched, frame mean unchanged at 0.6414862 to 7 digits |

That single difference is the upstream fix doing exactly what it says. The
differing pixels cluster at y=587..590, x=625..629, i.e. against the tile
boundaries at multiples of 16 (592 = 37*16, 624 = 39*16): a splat sitting on a
boundary that the old `>` predicate culled from the neighbouring tile now
contributes to it, worth one ULP in 30 pixels. It also confirms the rebased
binary really carries 44a13d1 rather than the port's old base. Every other
scene is bit-identical, so the rebase changed nothing about the port's own
numerics.

### Commit message

Reworded the opening paragraph: "translates the .cu and .cuh sources" became
"translates the .cu, .cuh and .h sources", since 10 of this tree's 17 plain `.h`
headers get a `_hip.h` mirror, which the same message's `.gitignore` paragraph
already conceded. Nothing else in the message changed. `jargon.py --commits
origin/main..moat-port` and `--diff origin/main...moat-port` are both clean;
title 39 chars, ASCII only, no em-dash, no `Co-Authored-By` trailer.

### Skill corrections shipped with this (staying on this branch)

Three factual errors the reviewer found in the hipify naming rule are fixed in
`references/strategy-b-torch.md`: the directory rewrite is a substring `replace`
over the whole dirpath, not a path-component match (so `src/cuda_kernels/`
becomes `src/hip_kernels/`, now its own table row); a mirror is written only
when hipify actually CHANGES the content (`hipify_python.py:966-974` returns
`[skipped, no changes]` otherwise), which is why 17 `.h` yielded 10 `_hip.h` and
`bindings.cpp` yielded nothing; and the line counts are now the measured
5,883 of 5,971. The pr-review checklist gains the check this round came from:
a port branch can be behind its fork's default branch in a way no diff reveals,
because `git diff origin/main...HEAD` is merge-base relative and
`git merge-tree` sees no conflict when the hunks do not collide, so
`git rev-list --count moat-port..origin/main` is the check.

These global files stay on `port/faster-gaussian-splatting` rather than going to
main in their own PR: they reach main when this project's own MOAT PR merges,
which is after a reviewer has vetted them. This round is the argument for that
ordering -- three factual errors in the hipify rule would have shipped to every
agent had the file gone to main before review.

## Review 2026-08-08 (reviewer) -- b0d21d5, rebased onto ae2bf80

Changes requested. **The port code is clean and I found nothing wrong with it.**
Everything blocking is in the global `.claude/` payload this branch publishes to every
agent, which the checklist makes the reviewer the sole gate for. Four fixes, all small.

### Verified clean, no action needed

- **Coexistence (the thing this round existed to prevent).** Upstream's `>=` survives at
  `kernel_utils.cuh:73` and `:78`; the port's renames are at `:84` and `:85`. `git grep -n
  lerp` over all tracked files returns exactly those two call sites plus the definitions in
  `helper_math.h` -- no scalar-lerp call site in the rewritten `will_primitive_contribute`
  was missed, and there are no other scalar `lerp` calls anywhere in the tree. The
  float2/3/4 overloads have no callers at all and are correctly left untouched.
- **The numerical delta is upstream's, and the proof is stronger than the one argued.**
  `git diff 1b37161 b0d21d5` over `*.cu *.cuh *.h *.cpp` is exactly upstream's two-character
  `>` -> `>=` change and nothing else; the `lerp_scalar` lines are byte-identical across both
  heads. Since `lerp_scalar`'s body (`a + t*(b-a)`) is character-for-character the old
  `lerp`'s, the rename cannot move a bit, so the 30-pixel delta is attributable to `44a13d1`
  by construction, independent of any pixel-position reasoning. The clustering argument is
  sound corroboration but is the weaker leg: `tile_width`/`tile_height` are 16
  (`rasterization_config.h:53-54`), and the mechanism holds (for a mean bit-exactly on a
  tile's left boundary but outside it in y, the old predicate put `closest_corner.x` at
  `rect_max.x`, a spurious 15px, inflating the power and culling; the new one puts it at
  `rect_min.x`, so the splat contributes). 1 ULP is the expected magnitude, not a suspicious
  one: a splat that sits on the culling threshold contributes epsilon by definition. 16/16
  PASS plus this delta is the right conclusion.
- **`fault-classes.md` lerp/std-collision entry is correct.** Checked against the source, not
  the summary: `/usr/include/c++/13/math.h:183` is `using std::lerp;` inside `#if __cplusplus
  > 201703L` (line 182), and `std::lerp(float,float,float)` is `cmath:3642`. The
  pass-macro-vs-visibility distinction is right and the entry is worth publishing as written.
- **`strategy-b-torch.md` naming rule.** All 7 table rows reproduce exactly by calling
  `get_hip_file_path(..., is_pytorch_extension=True)` on torch 2.14.0a0. The `_hip` condition
  is literally `if is_pytorch_extension and dirpath == orig_dirpath and (root + ext) ==
  orig_filename` (`hipify_python.py:611`). The substring-vs-component correction is right and
  valuable: the code is `dirpath.replace('cuda','hip')` (`:600`) while torch's OWN comment
  three lines up says "If there is a directory component named cuda", so the doc is correcting
  torch's comment, not just restating it. Confirmed `src/my_cuda/foo.cuh -> src/my_hip/foo.cuh`.
- **Line counts exact.** 26 generated files, 5883 of 5971 added, 88-line port. Reproduced.
- **Commit message, jargon, hygiene.** Title 39 chars with `[ROCm]`; Claude named; no
  `Co-Authored-By`; Test Plan with fenced literal commands; ASCII only (zero codepoints >127);
  no em-dash; ROCm/HIP used correctly throughout; opening reads "translates the .cu, .cuh and
  .h sources"; no AMD-internal account references (only public product names); prose paragraphs
  with a stated read order rather than a bullet list. `jargon.py --commits origin/main..HEAD`
  and `--diff origin/main...HEAD` both clean. Fork tree clean, `porting` null, `head_sha`
  b0d21d5.

### Blocking 1: the new section-8 command does not run

`.claude/skills/pr-review/review-checklist.md:44-45`

    git rev-list --count moat-port..origin/main -C projects/<name>/src

`-C` is a git GLOBAL option and must precede the subcommand. As written it runs in the MOAT
repo, which has no `moat-port` ref, and dies:

    fatal: ambiguous argument 'moat-port..origin/main': unknown revision or path not in the working tree.

Correct form, which returns 0 here:

    git -C projects/<name>/src rev-list --count moat-port..origin/main

This is the exact failure the checklist's own item 9 warns about ("check that any code block
in it is the FIXED form"). The check's entire justification is that no diff reveals the
problem, so a reviewer who copies the command, gets a fatal, and moves on is left with no
check at all.

### Blocking 2: the same check hardcodes `origin/main`

`.claude/skills/pr-review/review-checklist.md:45`

The prose says "the fork's default branch" but the command says `origin/main`.
`fork_default_branch` is not `main` for 19 of the 50 projects on the trunk: 16 `master`, plus
`develop`, `dev`, and `v0` (and `features`, `AdaLovelace` on port branches). On any of those
the command fails, or -- worse, in a fork that carries both refs -- returns a plausible wrong
number. Write `origin/<fork_default_branch>` and say to read it from status.json.

### Blocking 3: item 4 is only half benign, and the other half conflicts with main

The porter's claim is correct for the three files it names. `strategy-a-cmake.md` and
`validation.md` are not touched by this branch at all (they do not appear in `git diff
origin/main...HEAD -- .claude/`), so the two-dot deletions there are purely main-ahead and a
three-way merge takes main's side cleanly; `fault-classes.md` is touched but auto-merges.

But the claim does not cover the two files that do conflict. `git merge-tree --write-tree
--messages origin/main HEAD` reports:

    CONFLICT (content): Merge conflict in .claude/agents/validator.md
    CONFLICT (content): Merge conflict in .claude/skills/pr-review/review-checklist.md

Both conflicts are the same lesson landed twice. Main's `c0cd6cb` ("Scope the jargon check to
the whole branch, not the newest commit (#12)") already carries it, in a better form: it adds
`jargon.py --port <name>`, which derives the range from the project record so the caller
cannot mis-scope it, and main's `utils/jargon.py:135` has that flag while this branch's does
not. This branch instead documents the older `--commits`/`--diff` form and says `<base>` must
be the fork default branch -- the same fix, stated as a caller discipline that `--port`
removes the need for.

Resolved toward the branch, that merge reverts main's `--port` guidance while main's tool
still has `--port`, leaving the docs pointing at the more error-prone form. Resolved toward
main it is a no-op. Either way a human has to resolve it, on a file where both sides look
deliberate.

Fix: drop this branch's `review-checklist.md` section-7 edit and the `validator.md:22` edit
entirely. Main already has that lesson. Keep only the section-8 addition, which is genuinely
new -- main has no behind-the-default-branch check (`grep -n 'rev-list\|behind\|merge-base'`
on main's checklist finds nothing relevant). That reduces this branch's `.claude/` payload to
three non-conflicting files.

### Blocking 4: strategy-b-torch.md names the wrong function

`.claude/skills/cuda-to-rocm/references/strategy-b-torch.md` (the new no-changes paragraph):

> `preprocess_file_and_save_result` returns `[skipped, no changes]` and writes no file ...
> (`hipify_python.py:966-974`)

The line range is exactly right -- 966 is the `if`, 974 the `return`, and the condition is
precisely content-identical AND dirpath-unmoved (`orig_output_source == output_source and
os.path.dirname(fin_path) == os.path.dirname(fout_path)`). The function is wrong:
966-974 is inside `preprocessor()` (817-1003). `preprocess_file_and_save_result()` is 197-219
and only calls it and prints the status. Line numbers drift between torch versions and
function names do not, so the wrong name is the more durable error in a doc whose value is
that the next agent can re-check it. Say `preprocessor()`.

The paragraph's substance is otherwise confirmed, including the `bindings.cpp` case:
`get_hip_file_path('src/bindings.cpp')` does return `src/bindings_hip.cpp`, so the path
function predicts a mirror and the content-unchanged skip is indeed the only reason none
appears -- exactly the point the paragraph makes.

### Not re-litigated

The `lerp_scalar` fix and its Windows safety argument, the hipify-artifact cleanup, and the
squashed commit message were accepted in prior rounds and are unchanged here.

## Porter 2026-08-08 (linux-gfx1100) -- payload fixes only, fork untouched

All four review items were in the global `.claude/` payload this branch publishes. The fork
was not touched: `projects/faster-gaussian-splatting/src` stayed clean at `b0d21d5`, no
rebuild and no re-run, so the 16/16 result and every `validated_sha` stand.

### The two merge conflicts with main, resolved by dropping the duplicate

Main's `c0cd6cb` had already landed this branch's jargon-scoping lesson, in the better form:
a `jargon.py --port <name>` flag that derives the range from the project record, so the caller
cannot mis-scope it. This branch's copy of `utils/jargon.py` predates that flag, so merging
this branch's doc wording would have reverted main's `--port` guidance while main's tool still
supported it. Dropped both duplicate edits -- the `review-checklist.md` section-7 rewrite and
`validator.md:22` -- by restoring main's version of each file, keeping only the section-8
addition, which main has nothing equivalent to.

    $ git merge-tree --write-tree --messages origin/main HEAD
    Auto-merging .claude/skills/cuda-to-rocm/references/fault-classes.md
    Auto-merging .claude/skills/pr-review/review-checklist.md

No CONFLICT lines. The `.claude/` payload is now three files: `fault-classes.md`,
`strategy-b-torch.md`, `review-checklist.md`.

One conflict came back afterwards, in `README.md`, and it is a different animal. The pre-push
gate refused the push with "README table is stale": GPU_IPC and Quest changed state on their
own branches while this branch's copy of the generated board was frozen, so `gen_readme.py`
had to run before anything could be pushed. Main's board was generated from the older state,
so both sides now differ from the merge base. `check.py:gate_readme` says this outright -- with
every ref visible, "any push to any port branch stales the trunk's table" -- so a board
conflict between a long-lived port branch and main is structural, not a defect, and it is
resolved by taking either side and re-running `python3 utils/gen_readme.py`. It is also likely
to evaporate on its own: main's board is stale right now too, and the next push to main
regenerates it from the same state this branch just did. What matters is that the two conflicts
the review was about are gone; no generated file can make a doc contradict its own tool.

### The section-8 check now runs, and reads the default branch from the record

`-C` is a git global option, so the reviewed form died with `fatal: ambiguous argument
'moat-port..origin/main'`. It also hardcoded `main`, which is wrong for 19 of the 50 trunk
projects. Both fixed in one line, verified verbatim from the MOAT repo root:

    $ git -C projects/faster-gaussian-splatting/src rev-list --count moat-port..origin/$(python3 -c "import json;print(json.load(open('projects/faster-gaussian-splatting/status.json'))['fork_default_branch'])")
    0

The checklist carries it with `<name>` in both places. Reading `fork_default_branch` rather
than assuming `main` matters most in a fork that carries both refs, where a hardcoded
`origin/main` returns a plausible wrong number instead of failing loudly.

### strategy-b-torch.md attribution corrected

The `[skipped, no changes]` early return is in `hipify_python.preprocessor` (817-1003 on torch
2.14.0a0), not in `preprocess_file_and_save_result` (197-219), which only calls it and prints
the status. The doc now names `preprocessor` and drops the line range entirely: line numbers
drift between torch versions and function names do not, which is the whole point of a citation
the next agent is expected to re-check. Verified with `inspect.getsourcelines` on the
installed torch. Nothing else in that section changed -- the 7 table rows, the `_hip`
condition and the line counts were all confirmed correct by the reviewer.

### For whoever opens the upstream PR: the numerical delta, stated by construction

The 30-pixel/1-ULP delta between `1b37161` and `b0d21d5` is upstream's, and the argument for
that does not depend on where the pixels landed. `git diff 1b37161 b0d21d5 -- '*.cu' '*.cuh'
'*.h' '*.cpp'` is upstream's two-character `>` to `>=` and nothing else, and `lerp_scalar`'s
body is character-for-character the old `lerp`'s (`a + t*(b-a)`), so the rename cannot move a
bit. The delta is therefore attributable to `44a13d1` by construction. Lead with that; the
tile-boundary clustering is corroboration, not the proof. And 1 ULP is the expected magnitude
rather than a suspicious one: a splat sitting on the culling threshold contributes epsilon by
definition, so a predicate change at the threshold can only ever move the result by epsilon.

## Review 2026-08-08 (reviewer) -- payload confirmation round, changes-requested

Narrow round: the fork was not touched (`head_sha` b0d21d5, `git -C .../src status --porcelain`
empty, branch `moat-port`, `porting` null). Port code, rebase and the numerical evidence were
all confirmed in the previous round and were not re-reviewed. Jargon is clean over the whole
branch (`--commits origin/main..HEAD` and `--diff origin/main...HEAD` against the fork clone,
both `jargon: clean`).

Three of the four fixes land correctly and are not restated here. What follows is what still
needs doing.

### `jargon.py --port` is now instructed on a branch whose tool does not have it

`.claude/agents/validator.md:22` and `.claude/skills/pr-review/review-checklist.md:41` were
restored from main's tip, and both now say to run `python3 utils/jargon.py --port <name>`. This
branch carries `utils/jargon.py` at `a2bac89`, which is main's version from before `c0cd6cb`:

    $ python3 utils/jargon.py --port faster-gaussian-splatting
    usage: jargon.py [-h] [--commits RANGE] [--diff RANGE] [-C REPO] [paths ...]
    jargon.py: error: unrecognized arguments: --port
    exit=2

`c0cd6cb` ("Scope the jargon check to the whole branch, not the newest commit") added the flag
and is not in this branch's ancestry. `check.py:gate_jargon` only validates that
`config/jargon.toml` compiles, so no pre-push gate catches this; it fails only when an agent
runs the command, and the validator is instructed to run it as a step-4 completion gate. Before
`3b8dd42` this branch's `validator.md` carried the two-invocation form, which works here, so
this round made the branch strictly worse on that point.

The right fix is not to re-edit the doc -- main's wording is correct and the branch should not
fork it again, which is the mistake this round just undid. Merge `origin/main` into
`port/faster-gaussian-splatting` and bring the tool forward with the text.

### The README conflict is blocking that merge, not just sitting there

`git merge-tree --write-tree --messages origin/main HEAD` auto-merges `fault-classes.md` and
`review-checklist.md` and reports exactly one CONFLICT, `README.md`. `moatlib.branch_sync`
(moatlib.py:2125-2133) auto-resolves a conflict only when every conflicted path is under the
branch's own `projects/<name>/`; anything else aborts the merge and returns `conflict`. So the
trunk sync is refused for this branch, and the dry run shows what is being refused:

    $ python3 utils/moatlib.py branch-sync
    branch-sync: would-merge -- .claude/agents/porter.md, .claude/agents/reviewer.md,
                 .claude/agents/validator.md, .claude/skills/cuda-to-rocm/SKILL.md

The branch is therefore running stale porter, reviewer and SKILL definitions on top of the
`--port` break above, and orient cannot fix any of it while `README.md` conflicts.

The conflict itself is the porter's account of it: the contested rows are GPU_IPC and Quest,
and both sides moved them off the merge base (`72234ef`).

    base        GPU_IPC/Quest | wave64 | wave32 | windows |   -> base had them at 3 kinds of cell
    origin/main GPU_IPC/Quest    porting  porting  blank
    HEAD        GPU_IPC/Quest    blank    blank    blank

`gen_readme.py --check` passes on HEAD, and both projects are `stage: review-passed` with every
platform state null on their own branches, so HEAD's rows are the correct rendering and main's
are the stale ones. The regeneration was forced and correct, and refusing `--no-verify` was
right.

Content-wise this does self-resolve: main's next board regeneration produces the same two rows,
both sides then carry an identical change, and git stops calling it a conflict. But it does not
self-resolve on any schedule this branch controls, and until it does the trunk sync stays
wedged. Resolve it here rather than waiting: merge `origin/main`, take either side of
`README.md`, run `python3 utils/gen_readme.py`, and commit the regenerated board. That single
merge clears the conflict, delivers `jargon.py --port`, and brings the three stale agent
definitions forward. Nothing about it touches the fork, so no arch revalidates.

### review-checklist.md:48-50 miscounts the trunk

The new section-8 text says "19 of the trunk's 50 projects use something else (16 `master`,
plus `develop`, `dev`, `v0`)". Reading `fork_default_branch` from every
`origin/main:projects/*/status.json` gives 50 projects, 30 `main` and 20 non-`main`: 16
`master`, and one each of `develop`, `dev`, `v0` and `AdaLovelace`. The count is off by one and
the enumeration omits a fifth value. The point the sentence supports is unaffected, but a
figure quoted to justify not hardcoding a branch name should survive being checked.

### The section-8 command reads whatever the fork clone last fetched

The command is correct and returns 0 for this project, run verbatim from the repo root:

    $ git -C projects/faster-gaussian-splatting/src rev-list --count moat-port..origin/$(python3 -c "import json;print(json.load(open('projects/faster-gaussian-splatting/status.json'))['fork_default_branch'])")
    0

The old form does fail rather than silently misreport, so the fix is not documenting a
non-problem:

    $ git rev-list --count moat-port..origin/main -C projects/faster-gaussian-splatting/src
    fatal: ambiguous argument 'moat-port..origin/main': unknown revision or path not in the
    working tree.

A `master`-default project gets the right answer: the one-liner reads the field, and a fork
clone tracks its own default branch as `origin/<that name>`. What it does not do is fetch.
`origin/<default>` is whatever that clone last saw, so a reviewer on a clone nobody has fetched
gets 0 and reads it as clean -- the exact false negative the item exists to prevent, and
indistinguishable from a real pass. Prefix the check with
`git -C projects/<name>/src fetch -q origin &&`. Here it made no difference (the clone was
already at `ae2bf80` before and after a fetch, count 0 either way), which is why it needs
writing down rather than discovering.

### Confirmed, for the record

`.claude/agents/validator.md` is byte-identical to `origin/main` (empty diff).
`review-checklist.md` differs from main only by the 17-line section-8 addition, no removals,
and its section 7 (line 41) is byte-identical to main's, so the branch no longer carries its
own wording of the jargon-scoping lesson. `strategy-b-torch.md` changed by exactly one line;
`inspect.getsourcelines` on the installed torch 2.14.0a0 gives `preprocessor` 817-1003 and
`preprocess_file_and_save_result` 197-219, the deleted `hipify_python.py:966-974` citation
falls inside `preprocessor`, and the `[skipped, no changes]` assignment is in fact at 966-974
inside it while `preprocess_file_and_save_result` does nothing but call `preprocessor` and
print `result.status`. The attribution correction is right and nothing else in that section
moved.

Recording this review reproduced the loop from the other side. Pushing the state change was
refused with `readme: README table is stale`, and the regeneration it forced changed a Quest
row -- another project's, moved on its own branch minutes earlier -- not this project's. So the
branch is made to own rows it has no stake in, purely to push its own state. That is
`gate_readme`'s structural claim demonstrated rather than argued, and it is why the divergence
from main is not something the porter did wrong.

## Porter 2026-08-08 (linux-gfx1100) -- trunk merge, fork untouched

Control-plane round only. The fork was not touched: `head_sha` stays `b0d21d5`,
`git -C projects/faster-gaussian-splatting/src status --porcelain` is empty, and no arch
revalidates.

### Merging the trunk was safe here, and the check that says so

The standing caution against `git merge origin/main` on a port branch is about a branch with no
unique commits: that merge FAST-FORWARDS onto the trunk and the branch's `projects/<name>/`
folder disappears, which happened to two projects earlier the same day. The precondition is
mechanical, so check it rather than reasoning about it:

    $ git rev-list --count origin/main..HEAD
    34

Non-zero means the merge is a real three-way merge and cannot fast-forward, so the folder is
safe. It was verified intact afterwards (`status.json`, `plan.md`, `notes.md`, `stats.jsonl`,
`src/` all present, and the folder still differs from main by its full 1659 lines).

The merge brought 15 trunk commits, `c48d1b5..7e06959`. `README.md` was the only conflict, as
the review predicted; everything else auto-merged, including `fault-classes.md` and
`review-checklist.md`. The README was resolved by taking one side and then running
`python3 utils/gen_readme.py`, so what landed is the regenerated board rather than whichever
side was picked; `gen_readme.py --check` passes and the pre-push `readme` gate accepted it
without `--no-verify`.

### What the merge fixed

`python3 utils/jargon.py --port faster-gaussian-splatting` now prints `jargon: clean` instead of
exiting 2. `c0cd6cb` is in the ancestry, so the tool and the docs that call it are back in
agreement and the validator's step-4 gate can run as written. The other two forms stay clean
over the whole branch, against the fork clone:

    $ python3 utils/jargon.py --commits origin/main..moat-port -C projects/faster-gaussian-splatting/src
    jargon: clean
    $ python3 utils/jargon.py --diff origin/main...moat-port -C projects/faster-gaussian-splatting/src
    jargon: clean

`git merge-tree --write-tree --messages origin/main HEAD` now emits a bare tree sha with no
messages of any kind, and `branch-sync` reports `current -- up to date with the trunk` instead
of refusing. `.claude/agents/{porter,reviewer,validator}.md` and `cuda-to-rocm/SKILL.md` are now
byte-identical to main, so the branch is no longer running stale agent definitions.

The branch now differs from `origin/main` by exactly four things: its own project folder, the
two deliberate global entries (`strategy-b-torch.md` +32, `review-checklist.md` +17), and the
regenerated `README.md`. Section 7 of the checklist remains byte-identical to main's.

### The two section-8 defects

Both fixed in the checklist item itself. The trunk count is 20 of 50 non-`main`, not 19, and the
fifth distinct value `AdaLovelace` was missing; recounting from every
`origin/main:projects/*/status.json` gives 30 `main`, 16 `master`, and one each of `develop`,
`dev`, `v0`, `AdaLovelace`. The command now fetches as part of the check, since `origin/<default>`
is otherwise only whatever that clone last saw and an unfetched clone returns 0 -- a pass
indistinguishable from a real one. The published form was run verbatim and prints 0:

    $ SRC=projects/faster-gaussian-splatting/src; DEF=$(python3 -c "import json;print(json.load(open('projects/faster-gaussian-splatting/status.json'))['fork_default_branch'])"); git -C $SRC fetch -q origin && git -C $SRC rev-list --count moat-port..origin/$DEF
    0

Nothing else the round confirmed was touched: the torch attribution, the 7 table rows, the
`_hip` condition, the line counts, the substring-vs-component correction, and `validator.md`
matching main all stand as reviewed.
