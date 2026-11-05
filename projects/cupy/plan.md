# cupy plan: Windows ROCm build

Planner run 2026-11-04 on windows-gfx1151 (Radeon 8060S, TheRock ROCm 7.14.0a20260612 pip SDK, Python 3.13). Read-only clone of upstream `main` at `6efc0a4`; the fork's `moat-port` is at `18b54de`, 4 commits behind, and none of those 4 touches a file in this plan (they change `cupyx/scipy/sparse/linalg/_eigen.py` and its test). Line numbers below are at `6efc0a4`.

## Project

- Name: cupy
- Upstream: https://github.com/cupy/cupy, default branch `main` (PR base, per jeff 2026-11-04)
- Fork: https://github.com/ROCm/cupy (AMD's staging fork for upstream CuPy PRs). `main` there mirrors upstream; the port branch is `moat-port`, cut from upstream `18b54de`.
- Licence: MIT (intake). Submodules `third_party/{cccl,dlpack,jitify,xsf}` are permissive. None of them is touched.

## Existing AMD support

**Mature ROCm on Linux, none on Windows. The Windows build is the improvable delta.**

- CuPy upstream ships an authoritative HIP backend for Linux (`CUPY_INSTALL_USE_HIP=1`, `cupy-rocm-7-x` wheels, ROCm CI in `.github/workflows/scripts/pretest-rocm-test.sh`). It is maintained in-tree, and AMD engineers send HIP PRs from ROCm/cupy (e.g. #9622, #9746, #9748). Nothing in it is to be redone.
- Windows is refused outright: `install/cupy_builder/_compiler.py:191-193` raises `RuntimeError('ROCm is not supported on Windows')`. No branch on ROCm/cupy or cupy/cupy implements Windows ROCm (intake searched both).
- Delivery vehicle: an upstream PR to cupy/cupy `main`. CuPy merges platform work in-tree, so an upstream PR is the right vehicle (this is not a reference repo that links platform forks). The code is a small build-system and runtime-loader delta, with no kernel work, so there is no port-vs-rewrite question.

## Build classification

**setuptools + Cython with a custom builder (`install/cupy_builder/`). It is neither Strategy A (CMake) nor Strategy B (torch extension).** `ext_type` is already `setuptools-cuda` in status.json; it stays.

Evidence:
- `setup.py:139`: `cmdclass={'build_ext': cupy_builder._command.custom_build_ext}`; no torch import anywhere in the build.
- `install/cupy_builder/_command.py:52-55`: `.cu` files go to `DeviceCompilerWin32` / `DeviceCompilerUnix`; everything else is Cython-generated C++ built by the distutils host compiler (`cupy_setup_build.py:537-538`, `ccompiler.new_compiler()`, which is `MSVCCompiler` -> `cl.exe` on Windows).
- `install/cupy_builder/_features.py:412-419`: `CUPY_INSTALL_USE_HIP` selects the HIP feature set.

## Port strategy

The project already has its compat layer: `cupy_backends/hip/*.h` implement the CUDA runtime, driver, cuBLAS, cuSPARSE, cuRAND and cuSOLVER entry points on top of HIP and the roc*/hip* libraries. That is Strategy A's compat-header shape, written by upstream. This port therefore changes **no kernel or library-mapping code**. It does four things:

1. Teach the builder to build HIP on Windows: host C++ through clang-cl, the three `.cu` files through `hipcc.exe`, ROCm root discovery, and a library set that matches what ROCm ships on Windows.
2. Supply the two pieces ROCm lacks on Windows (roctracer, roctx) with no-op paths.
3. Fix two Windows mechanics: Windows only exports symbols that are explicitly exported, and DLL search does not use `PATH`.
4. Make runtime compilation (hiprtc, plus runtime `hipcc` for CUB reductions) find its DLL, headers and target arch on Windows.

Every change is gated on `sys.platform == 'win32'` / `_WIN32`, or is semantically identical on Linux (`os.devnull` for `/dev/null`; an export macro that expands to nothing off Windows). The CUDA Windows path (`DeviceCompilerWin32` nvcc branch, `/std:c++17`, the NVTX logic) stays byte-identical. Jeff's constraint is that the upstream PR must not change CuPy's CUDA Windows build or its Linux ROCm build behaviour.

## Feasibility probe (this laptop, no CuPy build, no GPU work)

All against `D:/Develop/TheRock/.venv/Lib/site-packages/_rocm_sdk_devel` (`rocm-sdk path --root`):

| question | answer |
|---|---|
| Import libs in `<root>/lib` | `amdhip64 hipblas hipfft hiprand hipsparse rocfft rocblas rocsolver rocsparse hiprtc hipsolver` all present. **No** `roctracer64`, `roctx64`, `rccl`. |
| Headers | `hip/hiprtc.h`, `hip/hip_version.h`, `hipcub/hipcub_version.hpp`, `thrust/version.h` (rocThrust), `rocprim`, `hipblas/`, `hipsparse/`, `hipfft/`, `rocsolver/`, `hiprand/` present. **No** `roctracer/`, `roctx.h`, `rccl/`. |
| Runtime DLLs in `<root>/bin` | `amdhip64_7.dll`, `amd_comgr.dll`, `rocm_kpack.dll`, `hiprtc0714.dll`, `hiprtc-builtins0714.dll`, `hipblas.dll`, `hipfft.dll`, `hiprand.dll`, `hipsparse.dll`, `rocblas.dll`, `rocfft.dll`, `rocrand.dll`, `rocsolver.dll`, `rocsparse.dll` (hard links into the core/libraries wheels). |
| hiprtc DLL name | `hiprtc0714.dll` = `hiprtc{MAJOR:02d}{MINOR:02d}.dll`. It exports `hiprtcGetCode`/`GetCodeSize`/... (the names `_cnvrtc.pxi` uses for the hiprtc prefix) and imports only `KERNEL32.dll`. HIP_VERSION here is 71460850 (7.14.60850). |
| Host compile of HIP headers | **cl.exe fails** on `hiprand/hiprand_kernel.h`, which `cupy/random/cupy_distributions.cuh:37` pulls into the host TU of `cupy.random._generator_api`: C2676 on `uint4[]`, C3861 `__builtin_unreachable`, C2146 in `rocrand_discrete.h`. **clang-cl compiles it cleanly.** cl.exe does accept `hip_runtime_api.h`/`hipblas.h`/`hipsparse.h`/`hipfft.h`/`rocsolver.h` alone, but the random module rules it out. clang-cl is required, not a preference. |
| clang-cl and setuptools' flags | setuptools' MSVC compile options include `/GL`; clang-cl ignores it and emits a plain COFF object (`-emit-obj`, no `-flto`). `link.exe /LTCG` then only warns "no code generation required". So swapping `cc` alone is enough; the MSVC linker stays. |
| Link | A clang-cl object linked with `link.exe` against the import libs above links; the import table shows `amdhip64_7.dll`. |
| hipcc CRT | **`hipcc.exe -c` defaults to the static CRT**: the object carries `/FAILIFMISMATCH:"RuntimeLibrary=MT_StaticRelease"` and `/DEFAULTLIB:libcpmt.lib`. Linked with setuptools' `/MD` objects, that is LNK2038. With `-fms-runtime-lib=dll` it is `MD_DynamicRelease` + `msvcrt.lib`. The Windows device-compile flags must include it. |
| hipcc arch | Without `--offload-arch`, hipcc on Windows **silently targets gfx906**. `--offload-arch=native` fails ("cannot determine hip architecture: ...offload-arch.exe"). `HCC_AMDGPU_TARGET=gfx1151` is honoured. |
| `hipcc -x hip -E -v` | Works with `NUL` as the input. The stderr carries the same `#include <...> search starts here:` / `End of search list.` block the regex in `cupy/_environment.py:553` expects. On Windows the list includes the MSVC STL and Windows SDK include dirs (paths with spaces) in addition to clang's resource and `cuda_wrappers` dirs and `<root>/include`. |
| ROCm root discovery | `hipconfig --rocmpath` (venv `Scripts/hipconfig.exe`) prints the devel root. `shutil.which('hipcc')` finds `.venv/Scripts/hipcc.exe`, a working console-script shim, so the Linux heuristic `realpath(hipcc)/../..` yields `.venv`: **wrong root**. No `HIP_PATH`/`ROCM_PATH`/`ROCM_HOME` is set on this host. |

## CUDA surface inventory

`python3 utils/surface.py generate cupy` census (a floor): 56 files. Markers: device_code 578, kernel 20, launch 2, warp_intrinsic 8, driver_api 13, texture 1. Libraries: cusparse 1583, cusolver 619, cublas 489, thrust 191, nccl 115, cufft 109, curand 81, cub 3.

All of it already has a HIP mapping upstream. This port maps nothing new; what matters is how each item reaches Windows:

| CUDA surface | HIP mapping (upstream, Linux) | Windows status / action |
|---|---|---|
| CUDA runtime (`cupy_backends/cuda/api/runtime.pyx`) | `cupy_backends/hip/cupy_hip_runtime.h` stubs over `amdhip64` | Builds. `cudaRuntimeGetVersion` is looked up via `SoftLink(__file__)` -> **must be exported** (C5). |
| Driver API (`driver.pyx`, `_driver_extern.pxi`) | `cupy_hip.h` `cu*` stubs over `hipModule*` | 20 `cu*` stubs looked up via `SoftLink(__file__)` -> **must be exported** (C5). |
| cuBLAS | hipBLAS + rocBLAS (`cupy_hipblas.h`) | `hipblas.lib`/`rocblas.lib` present. |
| cuSOLVER | rocSOLVER directly (`cupy_rocsolver.h`); `get_cusolver_handle()` is a rocSOLVER handle | `rocsolver.lib` present. hipSOLVER is not used. |
| cuSPARSE | hipSPARSE + rocSPARSE (`cupy_hipsparse.h`) | Present. 16 generic-API stubs looked up via `SoftLink(__file__)` (`cusparse.pyx:1054-1057`) -> **must be exported** (C5). |
| cuRAND (host + device `cupy_distributions.cu`) | hipRAND/rocRAND | Present. Host TU needs clang-cl (probe above). |
| cuFFT | hipFFT + rocFFT (`cupy/cuda/cupy_cufft.h`) | Present. |
| NVRTC | hiprtc, soft-linked by Linux soname (`_cnvrtc.pxi:126-152`) | **Windows DLL name** (C6). |
| CUB (`cupy_cub.cu`, CUB block reduction at runtime) | hipCUB/rocPRIM; runtime reductions call `hipcc` (`_cub_reduction.pyx:57-60`) | Headers present. Runtime `hipcc` needs a full path and an explicit arch on Windows (C7). |
| Thrust (`cupy_thrust.cu`) | rocThrust | Headers present; compiled by hipcc. |
| NCCL | RCCL (`HIP_nccl`, optional) | **Absent on Windows.** The module is not `required`, so it drops out when `rccl.h` is missing; confirm at build time, no code change expected. |
| NVTX | roctx (`cupy_roctx.h`, lib `roctx64`) | **Absent on Windows.** No-op path (C4). |
| `cudaProfilerStart/Stop` | `roctracer_start/stop` (`cupy_profiler.h`, lib `roctracer64`) | **Absent on Windows.** No-op path (C4). |
| Textures (`cupy/cuda/texture.pyx`, cupyx ndimage) | HIP texture objects | No code change; runtime risk on this laptop (see risks). |
| cuTENSOR, cuSPARSELt, Jitify, cuFFTXt, cudart_static | not built on HIP | Unchanged; not part of the HIP feature set. Jitify in particular is CUDA-only: on HIP, include paths for runtime compilation come from `_get_hipcc_include_dirs()`, not jitify. |
| Warp intrinsics / warp size in `cupy/_core/include/cupy/*.cuh` | Already wavefront-aware upstream (`hip_workaround.cuh`, runtime `warpSize`) | Untouched. Wave32 (gfx1151, gfx1100) and wave64 coverage comes from the three gates, not from this delta. |
| Managed memory, IPC, pinned memory, streams/events/graphs | HIP runtime | No code change. Windows-HIP runtime limits are risks (below). |

## Risk list

Ordered by how likely each is to cost a round.

1. **Symbols looked up through `ctypes.CDLL(__file__)` are not exported on Windows.** On Linux every non-static `extern "C"` function in a `.so` is dynamically visible. A Windows `.pyd` exports only `PyInit_*`, so `SoftLink.get()` in `cupy_backends/cuda/_softlink.pyx:65-78` returns `_fail_not_found` for every `cu*` driver stub, `cudaRuntimeGetVersion` and the cuSPARSE generic stubs. The symptom is an `AssertionError`/"function not found" at first use (module load, kernel launch, `runtimeGetVersion`), not a link error. Fix in C5; verify with `llvm-objdump -p <module>.pyd` export table.
2. **CRT mismatch between hipcc objects and setuptools objects** (proved above). `-fms-runtime-lib=dll` is mandatory in the Windows device compile.
3. **Silent wrong arch.** Both the build-time `.cu` compile and the runtime `hipcc --genco` (CUB block reductions, `backend='hipcc'` RawModules) fall back to gfx906 on Windows. The result is `hipErrorNoBinaryForGpu` / "invalid device function" at launch, which looks like a broken port. The build must refuse to proceed on Windows without `HCC_AMDGPU_TARGET` (or an explicit `--offload-arch` in `HIPCC`), and the runtime must pass `--offload-arch=<gcnArchName>` of the current device unless the caller gave one. `offload-arch.exe` failing here may be specific to this host's broken System32 `amdhip64_7.dll`; the explicit arch is correct either way.
4. **hiprtc fed MSVC STL include dirs.** Since #9622, every hiprtc compile appends `-I` for each dir from `hipcc -E -v`. On Windows that list includes the MSVC STL and Windows SDK dirs. If a CuPy device header reaches `<type_traits>`, `<cmath>` or similar, hiprtc parses MSVC STL headers in device mode. This is the same thing that happens with libstdc++ on Linux, but the MSVC STL is less tested under hiprtc. The first ElementwiseKernel/ReductionKernel run proves or disproves it. If it fails, the fallback is to filter the Windows list to clang resource dirs + `<root>/include` (a Windows-only change in `_get_hipcc_include_dirs`). The porter decides on evidence; do not pre-filter.
5. **DLL resolution.** Python 3.8+ does not use `PATH` for an extension's dependent DLLs. This host's System32 `amdhip64_7.dll` is broken; the `os.add_dll_directory` user dirs are searched before System32 for extension-module loads, which is what we want. Secondary loads made by the ROCm DLLs themselves (hiprtc -> `amd_comgr.dll`, `hiprtc-builtins0714.dll`; `amdhip64_7.dll` -> `rocm_kpack.dll`) may use the classic search order, where System32 wins. Preloading `amdhip64_7.dll`, `amd_comgr.dll` and `hiprtc*.dll` by full path from `<root>/bin` makes later by-name loads reuse them, as `rocm_sdk.preload_libraries` does. A missing transitive DLL shows as exit 127 or `?` with no module name: environment, not port (skill: validation.md). Keep the `add_dll_directory` handles alive (module-level list), or they are removed on GC.
6. **Knowing the build is HIP before importing any native module.** Every CuPy extension on HIP links the ROCm DLLs, so the DLL directory must be set before the first `cupy_backends` import, at a point where `runtime.is_hip` cannot yet be asked. It must not change the CUDA Windows build. Chosen mechanism (C8): `custom_build_ext` already writes `cupy/.data/_depends.json` on Windows (`_command.py:241-260`, unchanged), and its DLL list contains `amdhip64_7.dll` for a HIP build. `_environment` keys the ROCm DLL setup on that. **Porter must verify that the file lands for both `pip install .` and `pip install -e .`** (editable builds write `build_lib` to a temp dir). If it does not land in the editable layout, use a small builder-written marker instead and record why.
7. **ROCm root discovery picks the wrong tree.** On Linux, `_environment._get_rocm_path` falls back to `realpath(which hipcc)/../..`. On Windows that hits the venv console-script shim. The Windows order should be explicit variables first, then `hipconfig --rocmpath`; never the `hipcc` heuristic. `HIP_PATH` is what AMD's Windows HIP SDK installer sets machine-wide; whether to honour it is open question 1.
8. **Host-compiler swap inside setuptools.** `MSVCCompiler.initialize()` hard-codes `cl.exe` (`setuptools/_distutils/compilers/C/msvc.py:303`; setuptools 81) and ignores `CC`/`CXX`. The swap has to happen after `initialize()` and on **both** compiler instances: the one `preconfigure_modules` uses for `check_library`/`check_hip_version` (`cupy_setup_build.py:537`), and `build_ext.compiler` (`_command.py:207-221`, which already calls `initialize()` and records `cc`). Missing the first gives a confusing "Include files not found: hiprand/hiprand.h"-style configure failure.
9. **`-std=c++14` reaches clang-cl.** `install_build.py:228-230` appends GNU `-std=c++14` for HIP. clang-cl warns "unknown argument ignored" and builds C++14. Windows HIP must use `/std:c++17`, as the CUDA Windows branch does; Linux HIP keeps `-std=c++14` (unchanged).
10. **roctx/roctracer headers and libs referenced unconditionally.** `_features.py:181,191-192` require `roctx.h`, `roctracer64`, `roctx64`. On Windows the include check and the link check fail, and `cuda` is a `required` module, so configure aborts. `get_compiler_setting` already defines `CUPY_NO_NVTX` on win32 (`install_build.py:239-244`, via the CUDA NVTX probe). With it, `cupy_roctx.h` skips `<roctx.h>` but still calls `roctxMarkA` and friends: a compile error today.
11. **Windows HIP runtime limits surfacing as test failures** (not port defects; classify, do not fix):
    - kernels on separate streams do not overlap on Windows (skill: validation.md), so timing/concurrency assertions fail;
    - `hipMallocManaged` and IPC handles may be unsupported or limited on Windows/APU; tests should already skip on the device attribute, and any that do not are reference failures to record;
    - this laptop bugchecked (0x119) under another project's HIP texture tests; see open question 3.
12. **APU memory.** The 8060S shares system RAM. Large allocations have locked this machine. Run with `-m "not slow"`, a pool cap (`CUPY_GPU_MEMORY_LIMIT`), one pytest process, and `ulimit -c 0`.
13. **LLP64.** `long` is 32-bit on Windows host and device. CuPy's CUDA Windows build already lives with this, and hiprtc on Windows uses the MSVC data model, so typed kernels should behave as on CUDA Windows. Watch the int64/`long` dtype tests as a canary only.
14. **`__builtin_unreachable` / `_MSC_VER` branches.** clang defines `_MSC_VER` on windows-msvc, so `cupy_cub.h:51` / `cupy_thrust.h:19` / `cupy_cub.cu:128` take their MSVC branches in hipcc device compiles. The `cupy_cub.h`/`cupy_thrust.h` guards also require `__CUDACC_VER_MAJOR__ == 11`, so they are inert. `cupy_cub.cu:128` selects `__builtin_bit_cast`, which clang has. Low risk; noted so a compile error there is not a surprise.
15. **Generic fault classes** (wave64/wave32, OOB reads, unzeroed allocations, `-ffp-contract=fast`) are owned by the existing Linux HIP backend and covered by the Linux gates. This delta adds no device code. A wave32 failure on gfx1151 that also fails on linux-gfx1100 at the baseline is a reference failure, not a Windows regression.

## File-by-file change list

All paths relative to the fork root. "Windows-only" means guarded by `PLATFORM_WIN32`/`sys.platform == 'win32'`/`_WIN32`.

| file | change | scope |
|---|---|---|
| `install/cupy_builder/install_build.py` `get_rocm_path` (53-62) | win32: `ROCM_HOME`, then (per Q1) `HIP_PATH`/`ROCM_PATH`, then `hipconfig --rocmpath` | Windows-only |
| `install/cupy_builder/install_build.py` `get_compiler_setting` (228-237) | HIP+win32 -> `/std:c++17`; HIP+Linux keeps `-std=c++14` | Windows-only branch |
| `install/cupy_builder/install_build.py` `get_compiler_setting` (239-244) | for HIP on win32, define `CUPY_NO_NVTX` from roctx absence (not from the CUDA NVTX probe) and define the no-roctracer macro (C4) | Windows-only |
| `install/cupy_builder/install_build.py` (new helper) | `get_hip_host_compiler()`: clang-cl from `HIPCC_CC`-style override or `<rocm>/lib/llvm/bin/clang-cl.exe`; `configure_msvc_for_hip(compiler)` calls `initialize()` then sets `compiler.cc` | Windows-only |
| `install/cupy_builder/cupy_setup_build.py` `get_ext_modules` (533-542) | apply `configure_msvc_for_hip` to the configure-time compiler when HIP+win32 | Windows-only |
| `install/cupy_builder/_command.py` `build_extensions` (207-221) | after `initialize()`, swap `cc` to clang-cl when HIP | Windows-only |
| `install/cupy_builder/_compiler.py` `DeviceCompilerWin32.compile` (189-193) | replace the raise with `_compile_win32_hipcc`: `hipcc.exe` + preprocess opts + `-c src -o obj -O2 -std=c++17 -include hip_runtime.h -fms-runtime-lib=dll -D_USE_MATH_DEFINES`; no `-fPIC`; raise a clear error if neither `HCC_AMDGPU_TARGET` nor an `--offload-arch` in `HIPCC` is set | Windows-only |
| `install/cupy_builder/_features.py` `HIP_cuda_nvtx_cusolver` (174-196) | on win32 drop `roctx.h` from `include` and `roctracer64`/`roctx64` from `libraries` (absent from ROCm on Windows) | Windows-only |
| `cupy_backends/hip/cupy_roctx.h` | under `CUPY_NO_NVTX`: `nvtxMarkA`/`nvtxRangePushA`/`nvtxRangePop` no-ops, `NVTX_VERSION 0`, so `nvtx.pyx`'s `NVTX_VERSION < 1` paths take over | macro-gated; Linux never defines it today |
| `cupy_backends/hip/cupy_profiler.h` | under a new no-roctracer macro (e.g. `CUPY_NO_ROCTRACER`), skip `<roctracer/roctracer_ext.h>`; `cudaProfilerStart/Stop` return per Q2 | macro-gated |
| `cupy_backends/hip/cupy_hip_common.h` | `CUPY_HIP_EXPORT`: `__declspec(dllexport)` under `_WIN32`, empty otherwise | empty off Windows |
| `cupy_backends/hip/cupy_hip.h`, `cupy_hip_runtime.h`, `cupy_hipsparse.h` | prefix only the stubs that `SoftLink` resolves by name with `CUPY_HIP_EXPORT`: the `cu*` names in `_driver_extern.pxi`, `cudaRuntimeGetVersion`, and the `cusparse*` names `cusparse.pyx` gets via `_lib.get` (list in the risk section; derive it mechanically from the `.get('...')` calls) | no-op off Windows |
| `cupy_backends/cuda/libs/_cnvrtc.pxi` `_get_softlink` (126-152) | win32 + ROCm 7.x: `libname = f'hiprtc{major:02d}{minor:02d}.dll'` from `runtimeGetVersion()` | Windows-only |
| `cupy/_environment.py` `_get_rocm_path` (214-231) | win32 order as in the builder; skip the `which hipcc` heuristic | Windows-only |
| `cupy/_environment.py` `_get_hipcc_include_dirs` (545-571) | `/dev/null` -> `os.devnull`; call hipcc by `get_hipcc_path()` on win32 | `os.devnull` is identical on Linux |
| `cupy/_environment.py` (new) | `_setup_win32_hip_dll_directory()`: when the build is HIP (risk 6), `os.add_dll_directory(<root>/bin)`, keep the handle, preload `amdhip64_7`/`amd_comgr`/`hiprtc*` by full path | Windows-only |
| `cupy/__init__.py` (12-13) | call it before `from cupy import _core` | Windows-only |
| `cupy/_environment.py` `_diagnose_win32_dll_load` (672-698) | optional: print the ROCm path for a HIP build instead of `CUDA Path: None` | Windows-only, cosmetic |
| `cupy/cuda/compiler.py` `compile_using_hipcc` (887-922), `_preprocess_hipcc` (927-939) | win32: `get_hipcc_path()` instead of bare `hipcc`; append `--offload-arch=<gcnArchName>` unless `HCC_AMDGPU_TARGET` or an `--offload-arch` option is given | Windows-only |
| `docs/source/install.rst` (~470-500, "Building CuPy for ROCm From Source") | Windows subsection: TheRock SDK, clang-cl, `HCC_AMDGPU_TARGET` required, runtime needs the devel tree | docs (Q5) |

No change to: kernels, `cupy/_core/include`, CUDA headers, the CUDA branches of `_compiler.py`/`install_build.py`, CI workflows, wheels/packaging.

## Ordered commit plan

Each commit builds on its own and is reviewable alone. Commits 1-8 are Windows-only in effect. Titles follow the repo rules (`[ROCm]`, at most 72 characters), and the bodies carry a Test Plan.

1. **`[ROCm] Find the ROCm installation on Windows`**: builder `get_rocm_path` and runtime `_get_rocm_path` win32 order. Purpose: every later step needs the right root; the Linux heuristic finds the venv shim.
2. **`[ROCm] Build host code with clang-cl for HIP on Windows`**: compiler swap on both compiler instances, `/std:c++17`. Purpose: cl.exe cannot compile `hiprand_kernel.h` (proved); clang-cl targets the same MSVC ABI, so link.exe and Python's CRT are unchanged.
3. **`[ROCm] Compile device code with hipcc on Windows`**: `_compile_win32_hipcc` with `-fms-runtime-lib=dll`, arch requirement. Purpose: removes the `RuntimeError('ROCm is not supported on Windows')`.
4. **`[ROCm] Build without roctracer and roctx on Windows`**: feature list, `CUPY_NO_NVTX` for HIP, no-op `cupy_roctx.h`/`cupy_profiler.h` paths. Purpose: ROCm on Windows ships neither; `cuda` is a required module.
5. **`[ROCm] Export the HIP stubs that are resolved at run time on Windows`**: `CUPY_HIP_EXPORT` on the soft-linked stubs. Purpose: `ctypes.CDLL(__file__)` lookups need exported symbols on Windows.
6. **`[ROCm] Load hiprtc by its Windows DLL name`**: `_cnvrtc.pxi`. Purpose: RawKernel/ElementwiseKernel/ReductionKernel compilation.
7. **`[ROCm] Run hipcc on Windows for runtime compilation`**: `os.devnull`, full hipcc path, explicit offload arch. Purpose: hiprtc include discovery and CUB block reductions.
8. **`[ROCm] Add the ROCm DLL directory on Windows before import`**: `_environment` + `cupy/__init__.py`. Purpose: Python 3.8+ ignores `PATH` for extension dependencies.
9. **`[ROCm] Document building for ROCm on Windows`**: install.rst (if Q5 says yes).

Order rationale: 1-5 make `pip install` succeed, 6-8 make `import cupy` and the first compiled kernel work, and 9 is docs. A porter can build after 5, import after 8, and attribute each failure to one commit. If the porter finds a needed change outside this list (likely candidates: risk 4's include filter, an extra Windows define), it gets its own commit in the matching position, not a fold-in.

## Build commands

### Windows (windows-gfx1151, Git Bash)

Use a dedicated venv on a space-free path; do not install into the shared TheRock venv. Pin the SDK to the version the other Windows work uses.

```bash
ulimit -c 0
PY=/c/Users/jdaily/AppData/Local/Programs/Python/Python313/python.exe
$PY -m venv D:/Develop/moat/agent_space/cupy_venv
source D:/Develop/moat/agent_space/cupy_venv/Scripts/activate
python -m pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  "rocm[libraries,devel]==7.14.0a20260612"
rocm-sdk init && python -m rocm_sdk test
python -m pip install "setuptools>=77" wheel "Cython>=3.2,<3.3,!=3.2.6" "numpy>=2.0,<2.7" \
  "scipy>=1.14,<1.19" pytest "hypothesis>=6.37.2,<6.55.0" mpmath packaging

export ROCM_HOME="$(rocm-sdk path --root)"
export CUPY_INSTALL_USE_HIP=1
export HCC_AMDGPU_TARGET=gfx1151
export HIP_DEVICE_LIB_PATH="$ROCM_HOME/lib/llvm/amdgcn/bitcode"   # hipcc wrapper may not pass --rocm-path; drop if unneeded and say so
export CUPY_NUM_BUILD_JOBS=8
export PATH="/c/Users/jdaily/.local/bin:/d/Develop/moat/agent_space/py3shim:$PATH"
cd D:/Develop/moat/agent_space/wt-cupy
utils/timeit.sh cupy compile -- python -m pip install -v --no-build-isolation projects/cupy/src
```

Do not set `DISTUTILS_USE_SDK`; let setuptools locate Visual Studio itself. If a later step needs it, record why. After the build, check the import tables and exports:

```bash
R="$ROCM_HOME"; M=$(python -c "import cupy_backends.cuda.api.driver as m; print(m.__file__)")
"$R/lib/llvm/bin/llvm-objdump.exe" -p "$M" | grep -E "DLL Name|cuModuleLoad"
```

### Linux (wave64 and wave32 gates; regression check for the existing Linux HIP build)

```bash
export ROCM_HOME=/opt/rocm               # or $(rocm-sdk path --root) on a TheRock host, plus LD_LIBRARY_PATH=$ROCM_HOME/lib
export CUPY_INSTALL_USE_HIP=1
export HCC_AMDGPU_TARGET=gfx942          # gfx90a / gfx1100 / gfx1201 per host
export CUPY_NUM_BUILD_JOBS=<physical cores>
utils/timeit.sh cupy compile -- python -m pip install -v --no-build-isolation projects/cupy/src
```

Build the baseline (upstream `18b54de`, or the fork's `main` at the merge base) in a second venv with the same commands. The comparison in the test plan needs it.

### CUDA no-regression (standing rule; mi355 fleet host, nvcc 12.8)

```bash
export CUPY_NVCC_GENERATE_CODE="arch=compute_90,code=sm_90"
utils/timeit.sh cupy compile -- python -m pip install -v --no-build-isolation projects/cupy/src
python -m pytest tests/install_tests -q
```

This proves the Linux CUDA build compiles and links at the port head. The CUDA **Windows** build cannot be compiled on any MOAT host (open question 4).

## Test plan

### Smoke (every platform, before pytest)

```bash
python -c "import cupy; cupy.show_config()"
python - <<'EOF'
import cupy as cp, numpy as np
x = cp.arange(1 << 20, dtype=cp.float32)
assert float((x * 2).sum()) == float((np.arange(1 << 20, dtype=np.float32) * 2).sum())          # ElementwiseKernel + reduction (hiprtc)
k = cp.RawKernel(r'extern "C" __global__ void f(float* a){a[threadIdx.x]+=1;}', 'f'); a = cp.zeros(64, cp.float32); k((1,), (64,), (a,)); assert int(a.sum()) == 64
cp._core.set_reduction_accelerators(['cub']); assert float(cp.ones((256, 256)).sum(axis=1)[0]) == 256.0   # runtime hipcc + hipCUB
cp.random.seed(0); cp.random.standard_normal(1000).mean()                                       # hipRAND + cupy_distributions.cu
cp.linalg.eigvalsh(cp.eye(4)); cp.linalg.solve(cp.eye(4), cp.ones(4))                          # rocSOLVER
cp.fft.fft(cp.ones(64)); (cp.ones((64, 64)) @ cp.ones((64, 64)))                                # hipFFT, hipBLAS
import cupyx.scipy.sparse as sp; (sp.random(100, 100, 0.1, format='csr') @ cp.ones(100))         # hipSPARSE
cp.sort(cp.random.rand(1000)); cp.cuda.nvtx.Mark('m'); cp.cuda.profiler.start(); cp.cuda.profiler.stop()  # rocThrust, roctx/roctracer paths
print('OK')
EOF
```

### GPU pytest suites

Common environment for every run: `ulimit -c 0`, `CUPY_TEST_GPU_LIMIT=1`, one process (no xdist), `-m "not slow and not multi_gpu"`. On the laptop also set `CUPY_GPU_MEMORY_LIMIT=25%`. Run per directory so a hang or a host fault loses one directory, not the run. Wrap each run with `utils/timeit.sh cupy test --`.

Windows subset (windows gate):

| tier | suites | why |
|---|---|---|
| W1 | `tests/cupy_tests/cuda_tests` (deselect `test_texture.py`, see Q3), `tests/cupy_tests/core_tests` | every piece this delta touches: runtime/driver stubs, hiprtc, compiler, memory, streams, nvtx/profiler, cuBLAS/cuFFT/cuRAND/cuSOLVER/cuSPARSE wrappers |
| W2 | `tests/cupy_tests/{creation,math,linalg,fft,random,sorting,statistics,manipulation,indexing,logic,binary,misc,padding,polynomial,lib,functional,io}_tests`, `tests/cupy_tests/test_*.py` | the array library proper, which kernel compile drives |
| W3 | `tests/cupyx_tests/{linalg_tests,jit_tests}`, `tests/cupyx_tests/scipy_tests/{linalg_tests,sparse_tests,special_tests}`, `tests/cupyx_tests/test_*.py` | what gpu4pyscf uses (cupyx.scipy.linalg/sparse, cusolver/cusparse wrappers) |
| deferred | `tests/cupyx_tests/scipy_tests/{ndimage,signal,interpolate,...}`, `signal_tests`, `distributed_tests` (needs NCCL; skips), `example_tests` | large, texture-heavy, or not reachable on Windows; see Q3 |

Linux subset (wave64, wave32 gates): W1 + W2 + W3, plus `tests/cupyx_tests/scipy_tests` in full and `tests/example_tests`. This is wider than Windows because these gates are the regression check for a build that already works.

**How a failure is classified:** run the same suites on the baseline build on the same host and ROCm. A Linux **regression** is a test that passes at baseline and fails at port head. On Windows there is no baseline. Use the linux-gfx1100 (wave32) baseline failures as the reference set: a Windows failure that also fails on gfx1100 baseline is a reference failure, not a port defect. Windows-only failures are triaged against risk 11 before being called port defects. Record pass/fail/skip/xfail counts per directory and the full failing-test list in notes.md, tied to the exact fork sha.

### Non-GPU regression set (must not regress anywhere)

```bash
python -m pytest tests/install_tests tests/import_tests -q
CUPY_INSTALL_USE_HIP=1 python -m pytest tests/install_tests -q     # exercises check_hip_version on a HIP host
```

`tests/install_tests/test_cupy_builder/test_features.py::test_CUDA_cuda` is CUDA-only, as in upstream's Windows CI (`.pfnci/windows/test.ps1:261`).

## Open questions (jeff)

1. **Which Windows ROCm distributions should the PR support?** TheRock (pip SDK) is what MOAT validates. AMD's Windows HIP SDK sets `HIP_PATH` machine-wide. Honouring `HIP_PATH` makes CuPy work for HIP SDK users without extra setup, but on a machine with both installed it can silently pick the HIP SDK, which the skill says fails at run time in misleading ways. The options are: (a) `ROCM_HOME` -> `HIP_PATH` -> `ROCM_PATH` -> `hipconfig --rocmpath`; or (b) `ROCM_HOME` -> `hipconfig --rocmpath` only, with docs telling HIP SDK users to set `ROCM_HOME`. Either way only TheRock is validated, and the docs would say so.
2. **`cudaProfilerStart/Stop` on Windows HIP**: a no-op returning `hipSuccess` (what a CUDA user sees with no profiler attached), or `hipErrorNotSupported` (honest, but `cupy.cuda.profile()` would then raise)? The plan assumes the no-op.
3. **Textures and the laptop.** This laptop bugchecked (0x119) during another project's HIP texture tests. Should `tests/cupy_tests/cuda_tests/test_texture.py` and the texture-backed cupyx ndimage tests run on windows-gfx1151 at all, or be scoped out of the windows gate and covered on Linux only? Separately, is the W1-W3 subset above enough to satisfy the windows gate, or should the deferred cupyx suites run too?
4. **CUDA Windows no-regression evidence.** No MOAT host can build CuPy's CUDA Windows path. Is "the CUDA win32 branches are byte-identical in the diff (reviewer-checked) plus the Linux nvcc build on mi355" acceptable, or should that gap be stated in the PR body for upstream's Windows CI to cover?
5. **Docs in the same PR?** A Windows subsection in `docs/source/install.rst` (commit 9), or code only with docs as a follow-up?
6. **Coordination with ROCm/cupy's owners.** The port is staged on AMD's own CuPy staging fork, next to that team's Linux HIP work (recent upstream HIP PRs come from gpinkert). Who tells them before an upstream PR opens, and does it go upstream from ROCm/cupy under their flow or ours?
7. **Linux baseline cost.** The regression comparison needs a baseline build and test run on each Linux gate host, which doubles the Linux test time. Accept, or limit the Linux baseline to W1-W3?
8. **Packaging is not in this plan.** A Windows `cupy-rocm` wheel, conda package, or CI job is not planned; the deliverable is a source build. Confirm that is the intended scope. Separately, once this validates, gpu4pyscf's windows waiver and a `depends_on: cupy` edge on gpu4pyscf are for jeff to revisit; that record is not edited here.
