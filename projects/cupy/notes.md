# cupy notes

## Intake screen (2026-11-04, linux-gfx1100, intake role)

Scope, as ruled by jeffdaily: cupy/cupy already supports ROCm on Linux (HIP backend, `cupy-rocm-7-0` wheels), so this project covers ONLY the Windows ROCm build: TheRock ROCm SDK on Windows with clang-cl as the host compiler. gpu4pyscf's `windows` gate is blocked on it (its record carries an approved windows waiver and `blocked_reason` "cupy-rocm has no Windows distribution"). The port lives on ROCm/cupy, AMD's staging fork for upstream CuPy PRs (`fork_url`, ruled by jeffdaily). `moat-port` already exists there at `18b54de`, which is upstream `main` HEAD as cloned for this screen (2026-11-03 commit). Recommendation: **fork** (the fork exists, so state goes to `screened`).

### Licence

- `python3 utils/licenses.py check cupy/cupy`: `license=MIT tier=1`, cleared to contribute. LICENSE confirms MIT (Preferred Infrastructure / Preferred Networks).
- Submodules (`git clone --depth 1 --recurse-submodules`): `third_party/cccl` (Apache-2.0 with LLVM exception, plus Thrust's listed exceptions), `third_party/dlpack` (Apache-2.0), `third_party/jitify` (BSD-3-Clause, NVIDIA copyright), `third_party/xsf` (BSD-3-Clause, SciPy, plus compatible bundled licences in `LICENSES_bundled.txt`). All permissive.
- `python3 utils/licenses.py scan-nvidia agent_space/cupy-screen`: no NVIDIA proprietary licence text (submodules included).

### Duplicate effort

- cupy/cupy PRs and issues, searched `windows rocm` and `windows hip` (all states): nothing implements Windows ROCm. The only direct request is issue #8374 "need window support for cupy plz" (2024), closed as inactive after kmaehashi asked for the use case. cupy/cupy has no hip/rocm/amd/win branches.
- Upstream code says so outright: `install/cupy_builder/_compiler.py` `DeviceCompilerWin32` raises `RuntimeError('ROCm is not supported on Windows')` when `CUPY_INSTALL_USE_HIP` is set.
- ROCm/cupy, 135 branches: none is Windows work (no branch name matches `win`). The branches are ROCm Linux enablement, IFU and fix branches (`hip/*`, `rocm*`, `IFU-*`, MIOpen integration) plus `moat-port`. ROCm/cupy is the AMD team's own staging fork, so this port sits beside that team's Linux work and does not compete with it. Coordinating with whoever owns ROCm/cupy's upstream PR flow (recent HIP PRs upstream: gpinkert, e.g. #9622, #9746, #9748) is a human question, not a race.
- AMD-Ecosystem: not searched or recommended, per jeff's ruling.

### Viability: what the Windows HIP build needs

The build is Cython + setuptools (`install/cupy_builder/`); `ext_type` is recorded as `setuptools-cuda`. Device sources compiled by the device compiler on HIP are only `cupy/cuda/cupy_cub.cu`, `cupy/cuda/cupy_thrust.cu` and `cupy/random/cupy_distributions.cu`; everything else is Cython-generated C++ built by the host compiler against the HIP headers and the `cupy_backends/hip/*.h` stubs. No MOAT `depends_on`.

Build side (`install/cupy_builder/`):

1. `_compiler.py` `DeviceCompilerWin32.compile` raises for HIP. A Windows `_compile_*_hipcc` path is needed: `hipcc.exe` (TheRock ships it, marked "Legacy") or TheRock's `clang.exe -x hip`. Flags need to match the host CRT: `DeviceCompilerUnix._compile_unix_hipcc` passes `-fPIC`, which does not apply on Windows.
2. Host compiler: setuptools' `msvc` compiler drives `cl.exe` by default. The scope calls for clang-cl, so the planner has to choose how to swap it in (a compiler subclass, or the `CC`/`CXX`-style override that distutils' MSVC compiler does not honour). `install_build.get_compiler_setting` appends GNU `-std=c++14` for HIP on every OS; Windows needs `/std:c++17` (or `-std:c++17` for clang-cl), as the CUDA branch already does.
3. ROCm root: `install_build.get_rocm_path` reads only `ROCM_HOME` (empty string when unset, never `None`). Windows needs `HIP_PATH` and/or the TheRock `rocm-sdk` wheel layout. `get_hipcc_path` already knows `bin/hipcc.exe`.
4. Libraries (`_features.py` `HIP_cuda_nvtx_cusolver`, `HIP_random`, `HIP_cub`, `HIP_thrust`, `HIP_nccl`): link names are `amdhip64 hipblas hipfft hiprand hipsparse rocfft roctracer64 roctx64 rocblas rocsolver rocsparse` plus optional `rccl`. TheRock's `docs/development/windows_support.md` lists rocBLAS, hipBLAS, hipBLASLt, rocSOLVER, hipSOLVER, rocRAND, hipRAND, rocFFT, hipFFT, rocSPARSE, hipSPARSE, hipCUB, rocThrust, rocPRIM and MIOpen as Supported on Windows. RCCL is Unsupported (automatically disabled), and so is the whole rocprofiler family; roctracer and roctx are not listed at all. So:
   - `roctracer64` / `roctx64` are not expected on Windows. `cupy_backends/hip/cupy_profiler.h` (`roctracer_start/stop` for `cudaProfilerStart/Stop`) and `cupy_roctx.h` need a Windows no-op path. `CUPY_NO_NVTX` already exists for the CUDA Windows build without NVTX (`get_compiler_setting`), and roctx can reuse it; the profiler stub needs its own guard.
   - `HIP_nccl` is not `required`, so rccl should drop out when `rccl.h` is absent; confirm at port time.
   - Import-library names on Windows (`amdhip64.lib`, `hipblas.lib`, ...) and the `lib` directory need confirming against a TheRock install; `cupy_setup_build.py` also prepends the old `<rocm>/hipfft/{include,lib}` layout for ROCm >= 4.1, which is harmless when absent.
5. `check_hip_version` / `check_library` compile and run small probes with the host compiler; they have to find the HIP headers and DLLs at build time on Windows too.

Runtime side (`cupy/`, `cupy_backends/`):

6. hiprtc: `cupy_backends/cuda/libs/_cnvrtc.pxi` soft-links hiprtc by Linux soname (`libhiprtc.so.7` for ROCm 7.x, `libamdhip64.so.N` before that). Windows needs the hiprtc DLL name from TheRock (to be read off an install; the HIP runtime DLL is `amdhip64_7.dll`). Without this, no RawKernel, ElementwiseKernel or ReductionKernel compiles, so it is on the critical path for gpu4pyscf.
7. `cupy/_environment.py` `_get_hipcc_include_dirs` runs `hipcc -x hip -E -v /dev/null`, and `cupy/cuda/compiler.py` `_compile_with_cache_hip` calls it on every hiprtc compile (since PR #9622). `/dev/null` does not exist on Windows (`NUL` or a temp file is needed), and the regex expects hipcc's Linux `-v` output format.
8. `_environment._get_rocm_path` falls back from `ROCM_HOME` to `realpath(which hipcc)` to `/opt/rocm`; Windows needs `HIP_PATH` / rocm-sdk discovery. `cupy/_core/_cub_reduction.pyx` calls hipcc at runtime for CUB-based reductions (`-I<rocm>/include`), so hipcc has to be reachable at runtime on Windows, or that path disabled there.
9. DLL loading: Python 3.8+ on Windows does not search `PATH` for an extension module's dependent DLLs. The CUDA Windows build relies on `cuda.pathfinder`, which does not apply to HIP. The HIP build needs an `os.add_dll_directory` on the ROCm `bin` (or the TheRock rocm-sdk wheel's equivalent) before the first `cupy_backends` import. The skill's Windows notes apply: build and run against TheRock, never the Windows HIP SDK; stage `rocm_kpack.dll` alongside `amdhip64_7.dll`; and read a first-launch DLL or 719 error as environment before blaming the port.
10. HIP on Windows does not overlap kernels on separate streams (skill validation reference). Any cupy test that asserts multi-stream concurrency will fail on Windows for that reason alone.

Shared-code note for the planner: items 1-5 and 9 touch `install/cupy_builder` and `_environment.py`, which the CUDA Windows build also uses. The CUDA Windows path must stay unchanged, and the standing rule to compile the CUDA path with real nvcc before presenting applies.

### Questions for jeff (recorded, not decided)

- `fork_default_branch` is recorded as `main`, but ROCm/cupy has no `main` branch: its GitHub default is `master` (`6c4b343`). Upstream cupy/cupy's default is `main`. Whichever branch is the upstream mirror and PR base needs recording; this screen did not change it.
- Gates: the project scope is Windows-only, but `config/arches.toml` requires `wave64`, `wave32` and `windows`. The build-script and `_environment.py` changes are shared with the Linux HIP build, so Linux revalidation is still real evidence against a regression. Whether the Linux gates are owed (or a waiver is suggested) is jeff's call.
- Once cupy-rocm builds on Windows, gpu4pyscf's approved windows waiver and its `windows-gfx1101` block can be revisited, and a `depends_on: cupy` edge could be recorded on gpu4pyscf. That is gpu4pyscf's record and is not edited here.

## Ruling 2026-11-04 (jeff): PR base and mirror
The PR base is upstream cupy/cupy `main`. ROCm/cupy had no `main` (its default `master` stopped at 2026-01-31 when the fork became staging-only), so a `main` branch mirroring upstream was pushed to ROCm/cupy at 6efc0a4. ROCm/cupy's `master` and its default branch are left as they are (not ours to reconfigure). `fork_default_branch` stays `main`. moat-port was cut from upstream main at 18b54de.
- Gates (jeff 2026-11-04): all three gates stay required (wave64, wave32, windows). The Windows work touches code the Linux HIP build shares, so the Linux validations are the regression check for CuPy's existing Linux ROCm build.

## Planner 2026-11-04 (windows-gfx1151)

plan.md and surface.json written against upstream main 6efc0a4 (moat-port 18b54de is 4 commits behind; none touch planned files). Laptop probes against TheRock 7.14.0a20260612 (`rocm-sdk path --root`), no CuPy build:
- hiprtc DLL is `hiprtc0714.dll` (`hiprtc{MAJOR:02d}{MINOR:02d}.dll`); no roctracer/roctx/rccl libs or headers on Windows; all other import libs present.
- cl.exe cannot host-compile `hiprand/hiprand_kernel.h` (C2676/C3861/C2146); clang-cl can. clang-cl ignores setuptools' `/GL`, so only `compiler.cc` needs swapping.
- `hipcc -c` defaults to the static CRT (`RuntimeLibrary=MT_StaticRelease`); `-fms-runtime-lib=dll` gives `MD_DynamicRelease`.
- hipcc on Windows silently targets gfx906 without `--offload-arch`; `--offload-arch=native` fails here; `HCC_AMDGPU_TARGET` is honoured.
- HIP stubs resolved via `ctypes.CDLL(__file__)` need explicit export on Windows (plan risk 1).
- `hipconfig --rocmpath` prints the devel root; `which hipcc` finds the venv shim, so the Linux root heuristic is wrong on Windows.
Eight open questions for jeff are in plan.md.

## Porter round 2026-11-05 (windows-gfx1151, Radeon 8060S, TheRock ROCm 7.14.0a20260612, Python 3.13, MSVC 14.44)

moat-port 18b54de -> f4f78903 (12 commits, fast-forward push to ROCm/cupy). Plan commits 1-9 landed in order; three needed changes found while building got their own commits (8, 9, 10 below), and plan commit 8 (DLL directory) grew the editable-install case.

| # | sha | title |
|---|---|---|
| 1 | d83e33a | Find the ROCm installation on Windows (ROCM_HOME, then `hipconfig --rocmpath`; HIP_PATH NOT honoured, Ruling 1) |
| 2 | 2d2388d | Build host code with clang-cl for HIP on Windows (both compiler instances; `/std:c++17`) |
| 3 | cf9406d | Compile device code with hipcc on Windows (`-fms-runtime-lib=dll`, HCC_AMDGPU_TARGET required) |
| 4 | 186c05c | Build without roctracer and roctx on Windows (`CUPY_NO_NVTX`, `CUPY_NO_ROCTRACER`; profiler start/stop succeed as no-ops, Ruling 2) |
| 5 | 94fa86b | Export the HIP stubs resolved at run time (`CUPY_HIP_EXPORT`: 25 cu*, cudaRuntimeGetVersion, 12 cusparse) |
| 6 | 850bb2f | Load hiprtc by its Windows DLL name (`hiprtc0714.dll`) |
| 7 | a331ca0 | Run hipcc on Windows for runtime compilation (os.devnull, full hipcc path, `--offload-arch=<gcnArchName>`) |
| 8 | b2cb1b3 | NEW: Fix two HIP host headers (`<string>` for std::to_string in cupy_hipblas.h; `rocblas_stride` not `long int` for geqrf in cupy_lapack.h -- LLP64) |
| 9 | 7eb7e38 | NEW: hipcc.exe does not quote forwarded args: drop `-DCYTHON_EXTERN_C=extern "C"` from the Windows device compile; on Windows pass hipcc's default -I dirs only to hiprtc, not to the hipcc backend (MSVC dirs under `Program Files (x86)` split) |
| 10 | a82fe0d | NEW: hiprtc on Windows fails (HIPRTC_ERROR_COMPILATION, empty log) when the program name is an absolute path; use the base name on HIP+win32 |
| 11 | 6fd6150 | Add the ROCm DLL directory before import (keyed on amdhip64 in `_depends.json`; no preloading needed) + copy `_depends.json` into the source tree for in-place/editable HIP builds |
| 12 | f4f7890 | Document building for ROCm on Windows (install.rst subsection, Ruling 5) |

Every change is gated on win32 / `_WIN32` / `ctx.use_hip`, or is identical on Linux (`os.devnull`; `<string>` already transitive; `rocblas_stride` == `long` on LP64; `CUPY_HIP_EXPORT` empty off Windows). CUDA branches untouched except an inserted HIP-only line in `_command.build_extensions`.

### Build recipe (Git Bash inside the x64 Native Tools prompt)

```bash
PY=/c/Users/jdaily/AppData/Local/Programs/Python/Python313/python.exe
$PY -m venv C:/Users/jdaily/cupy-work/venv
C:/Users/jdaily/cupy-work/venv/Scripts/python.exe -m pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ "rocm[libraries,devel]==7.14.0a20260612"
C:/Users/jdaily/cupy-work/venv/Scripts/rocm-sdk.exe init
C:/Users/jdaily/cupy-work/venv/Scripts/python.exe -m pip install "setuptools>=77" wheel "Cython>=3.2,<3.3,!=3.2.6" "numpy>=2.0,<2.7" "scipy>=1.14,<1.19" pytest "hypothesis>=6.37.2,<6.55.0" mpmath packaging
source agent_space/cupy_rocm_env.sh   # venv Scripts + MSVC bin first on PATH, ROCM_HOME=<rocm-sdk path --root>, CUPY_INSTALL_USE_HIP=1, HCC_AMDGPU_TARGET=gfx1151, CUPY_NUM_BUILD_JOBS=12
utils/timeit.sh cupy compile -- python -m pip install -v --no-build-isolation projects/cupy/src      # ~6 min incremental; also -e (editable) verified
```

- `HIP_DEVICE_LIB_PATH` is NOT needed: `hipcc.exe -c` from `<root>/bin` found the device libs without it. `ROCM_HOME` can be omitted (hipconfig fallback); `DISTUTILS_USE_SDK` not set.
- Configure summary: cuda Yes (71460850), cub Yes (400400), nccl No (`rccl.h` absent, optional), random Yes, thrust Yes, dlpack Yes.
- Exports verified: `llvm-objdump -p driver.cp313-win_amd64.pyd` lists cuLaunchKernel/cuModuleLoad..., runtime pyd exports cudaRuntimeGetVersion, cusparse pyd 15 cusparse exports.
- Plan risk 4 (hiprtc fed MSVC STL include dirs) did NOT materialise: elementwise/reduction kernels compile with the full `hipcc -E -v` list. Risk 5: no preloading needed; `GetModuleFileNameW` shows amdhip64_7/hipblas/hiprtc0714/amd_comgr all loaded from `<root>/bin`, not System32. Risk 6: `_depends.json` lands for `pip install .`, NOT for `pip install -e .` (setuptools builds into a temp build_lib) -- fixed in commit 11 and verified (editable import + smoke).
- Build failures hit before commits 8-9 (build1.log): `std::to_string` (cupy_hipblas.h:92), geqrf `long` vs `rocblas_stride` (cupy_lapack.h:139-143), `clang: error: no such file or directory: 'C'` (hipcc.exe quoting). Runtime failure before commit 10: every ElementwiseKernel raised CompileException with an empty log; isolated with a direct hiprtc probe (`k.cu` OK, `C:\...\k.cu` and `C:/.../k.cu` FAIL). Before the -I change in commit 9, the CUB block reduction failed with `clang: error: no such file or directory: 'Files'`.

### Smoke (plan test-plan smoke, all pass, non-editable and editable)

elementwise+reduction, RawKernel, CUB block reduction via runtime hipcc (`set_reduction_accelerators(['cub'])`), hipRAND, rocSOLVER eigvalsh/solve, hipFFT, hipBLAS matmul, hipSPARSE csr@vec, rocThrust sort, nvtx Mark/RangePush/RangePop, profiler start/stop, `cupy.cuda.profile()`. hiprtc compile costs ~1 s per kernel on this laptop; a runtime-hipcc CUB kernel ~1-3 min.

### CUDA Windows no-regression (Ruling 4)

CUDA 12.8 no-admin toolkit (`CUDA_PATH=C:\Users\jdaily\cuda-win\cuda128\Library`, `CUPY_LIBRARY_PATH=...\Library\lib`, `CUPY_NVCC_GENERATE_CODE=arch=compute_90,code=sm_90`, MSVC 14.44, separate venv `C:/Users/jdaily/cupy-work/cudavenv`, `source agent_space/cupy_cuda_env.sh`), `pip wheel -v --no-build-isolation --no-deps` of upstream 18b54de and of moat-port f4f78903 in two clones under C:/Users/jdaily/cupy-work:
- both build (EXIT 0); identical module summary (cuda Yes 12080, cusolver Yes, cub/thrust Yes 300102, jitify Yes, random Yes, dlpack Yes; nccl/nvtx/cutensor/cusparselt No -- headers not in this toolkit);
- identical set of 64 `.pyd` in the wheels;
- all 134 compile/link command lines (`INFO:compilers.C.base`, nvcc `Command:`) identical after normalising the clone path, temp dirs and `CUPY_CACHE_KEY`;
- same warning count (1684) and the same 5 LNK4098 LIBCMT warnings in both.
No GPU run possible. The single-arch gencode is a choice for the comparison only.

### Windows stage (ii) subset run in this round (CUPY_TEST_GPU_LIMIT=1, CUPY_GPU_MEMORY_LIMIT=25%, ulimit -c 0, one pytest process, `-m "not slow and not multi_gpu"`, test_texture.py always --ignore'd)

- `tests/cupy_tests/cuda_tests` at f4f78903: **188 passed, 3 failed, 81 skipped, 28 deselected** (15 s). Failures, none in code this delta touches:
  - `test_memory.py::TestSingleDeviceMemoryPool::test_get_limit` -- artefact of the test's own memory cap (`CUPY_GPU_MEMORY_LIMIT=25%` sets a pool limit; the test expects 0). Passes with the cap unset (verified).
  - `test_memory.py::test_managed_memory_madvise_basic` -- `hipErrorInvalidValue` from the madvise call: HIP managed-memory advice on Windows/APU (plan risk 11). Compare with the linux-gfx1100 baseline.
  - `test_runtime.py::test_pointer_get_memory_type` -- `hostAlloc` pinned memory reports `type == 2` (device) instead of 1 (host) on this Windows APU runtime (plan risk 11). Compare with the linux-gfx1100 baseline.
- First cuda_tests attempt was killed by me before it reached test_texture.py: the `--ignore=tests/...` path was relative to the wrong cwd. The test script (agent_space/cupy_test.sh) now always ignores test_texture.py by absolute path.
- `tests/cupy_tests/core_tests` (minus `fusion_tests` and `test_cub_reduction.py`, see below) at f4f78903, **stopped at 51% for the time budget**: **1145 passed, 25 failed, 170 skipped, 2 xfailed** through `test_ndarray_elementwise_op.py` (`-v`, so per-test lines survive the stop; log C:/Users/jdaily/cupy-work/test-w1-core.log on the laptop). Not yet run: the rest of core_tests from `test_ndarray_elementwise_op.py::test_broadcasted_*` on, `fusion_tests`, `test_cub_reduction.py`.
  - 23 x `test_elementwise.py::TestElementwiseGUFuncLike::*`: the kernels `#include <cupy/xsf/cupy.h>`; hiprtc on Windows is handed the MSVC STL include dirs (from `hipcc -E -v`) and `<complex>` fails: `MSVC ...\include\complex:575: error: call to 'signbit' is ambiguous` (candidates: `hiprtc_runtime.h:1083/1084` device `signbit(float/double)`). **Plan risk 4 materialised** for kernels that include C++ standard headers (xsf special functions, mdspan GUFunc-like kernels). The plan's fallback (filter to clang resource dirs + `<root>/include`) cannot work on its own: xsf's `config.h` includes `<complex>`, `<cmath>`, ... when `__CUDACC__` is not defined, and TheRock ships no libc++ headers. Plain elementwise/reduction kernels are unaffected. NOT fixed in this round -- needs a decision (deferred `hiprtc-win-msvc-stl-complex-signbit`).
  - 2 x `test_ndarray_elementwise_op.py::test_pow_python_scalar_fastpath[3|3.0]`: complex64 `a ** 3` is off by 4.2e-7 relative (`-124.99995-2.98e-6j` vs `-125`), over the test's atol 2e-5 at |216|. Device complex pow accuracy in shared code, not this delta; compare with the linux-gfx1100 baseline.
  - Speed: hiprtc costs ~1 s per kernel here (another workload was compiling on the host at the time), so core_tests alone is a multi-hour run on this laptop; `fusion_tests` managed ~20 tests in 16 min and was stopped. Each runtime-hipcc CUB kernel takes 1-3 min, so `test_cub_reduction.py` was left out (the smoke covers the CUB hipcc path).
- Non-GPU: `tests/install_tests tests/import_tests` with `CUPY_INSTALL_USE_HIP=1`: 31 passed, 1 skipped.
- Stage (ii) is therefore incomplete: rest of W1, all of W2 and W3 are owed by the windows-gfx1151 validator (after the Linux gates, Ruling 3). The test script `agent_space/cupy_test.sh` (laptop) wraps the run.
- Three ROCm defects registered as deferred rocm-bug-report on this project: `hiprtc-win-absolute-program-name`, `hipcc-exe-unquoted-forwarded-args` (both worked around in the port), `hiprtc-win-msvc-stl-complex-signbit` (not worked around).

## Porter round 2026-11-05b (windows-gfx1151): hiprtc + MSVC STL workaround (deferral hiprtc-win-msvc-stl-complex-signbit, ruled `now` by jeff, bounded)

Result: **worked.** Two new commits on the fork clone, on top of f4f78903, **committed locally and NOT pushed** (see "Lock" below):

| # | sha | title |
|---|---|---|
| 13 | 4ccded519 | [ROCm] Declare the as_mdspan friend of CArray as __device__ (`cupy/_core/include/cupy/carray.cuh`, one line) |
| 14 | 694db667f | [ROCm] Supply MSVC STL device helpers for hiprtc on Windows (`cupy/_core/include/cupy/hip_workaround.cuh`, +94 lines guarded by `__HIPCC_RTC__ && _MSC_VER`) |

What was found, in order (each layer only visible once the previous one was fixed):
1. `MSVC complex:575: call to 'signbit' is ambiguous`: `_Ctraits<long double>::_Signbit` calls `std::signbit(long double)`; hiprtc_runtime.h declares only `signbit(float)` and `signbit(double)` (lines 1083/1084). Fix: `static __device__ bool signbit(long double)` forwarding to `__signbit(double)`, plus `namespace std { using ::signbit; }` (the earlier using-declaration in std does not see a later overload). Must precede the first `<complex>` include; `hip_workaround.cuh` is in `cupy_header_list`, which `compile_with_cache` prepends to every CuPy-generated kernel, before the user preamble (`#include <cupy/xsf/...>`). hiprtc on Windows defines `_WIN32`, `_MSC_VER`, `__HIPCC_RTC__`, `__HIP_DEVICE_COMPILE__` (probed); `__CUDACC__` is not defined.
2. `xsf/cupy.h: 'shape_' is a private member of 'CArray<...>'` (21 of 31 GUFuncLike tests). NOT Windows-specific: the friend declaration in carray.cuh lacks `__device__` while `xsf::as_mdspan` is `__device__`; the ROCm 7.14 clang (23) no longer matches them. A minimal standalone repro fails the same with offline `clang++ -x hip --cuda-device-only` for BOTH `--target=x86_64-unknown-linux-gnu` and `x86_64-pc-windows-msvc`, and with `-fno-ms-compatibility -fno-delayed-template-parsing`; adding `__device__` to the friend fixes it. CUDA: nvcc 12.8 `-c` compiles both forms; NVRTC 12.8 compiling the real cupy headers + an `xsf::as_mdspan` kernel FAILS without the change ("A function without execution space annotations ... host functions are not allowed in JIT mode", unless `-default-device`) and PASSES with it. So this is shared code, deliberately unconditional (correct on CUDA, Linux HIP, Windows HIP). It deviates from "Windows-HIP-only" in jeff's brief; it is a separate commit so it can be dropped or sent separately. The Linux validators should check GUFuncLike on Linux ROCm 7.14 (prediction: it fails there without this commit).
3. `ld.lld: error: undefined hidden symbol: _Exp` (voigt_profile): MSVC's complex exp/cosh/sinh call the UCRT helpers `_Exp`, `_Cosh`, `_Sinh`, `_FExp`, `_FCosh`, `_FSinh` (ymath.h), host-only. A plain `extern "C" __device__ short _Exp(...)` definition is refused: `__host__ __device__ function '_Exp' cannot overload __device__ function '_Exp'` (clang's `cuda_wrappers/complex` wraps the STL in `force_cuda_host_device`). Working form: `__device__ __attribute__((weak)) short __cupy_msvc_Exp(double*, double, short) __asm__("_Exp")` plus its definition; same for the other five. Semantics follow the UCRT (microsoft/STL `stl/src/xexp.cpp`, `xcosh.cpp`, `xsinh.cpp`, read for behaviour, not copied): `_Exp` = y*e^x*2^eoff with a power-of-two split for |x|>700 (two-step ln2 reduction with fma), returns 0/1/-1 (zero/inf/finite); `_Cosh`/`_Sinh` = y*cosh/sinh with the exp path for |x|>=22 and the UCRT's NaN/inf/zero cases; the float forms go through double. `weak` so a second definition (e.g. a future multi-program link) cannot collide. Long double helpers (`_LExp`, `_LCosh`, `_LSinh`, `_LDtest`) are not needed by anything instantiated and are not provided.
- Not workable: filtering include dirs (TheRock ships no libc++; xsf `config.h` includes `<complex>`, `<cmath>`, ... whenever `__CUDACC__` is undefined); an MSVC STL macro to avoid `_Ctraits` (none exists). `<cmath>` alone compiles under hiprtc with the MSVC dirs (probed: signbit/exp/cosh on double).
- Scope limit: user `RawKernel`/`RawModule` code is not prefixed with `cupy_header_list` and still hits the bug if it includes `<complex>`; including `<cupy/hip_workaround.cuh>` first is the user-side workaround.

Accuracy probe (`C:/Users/jdaily/cupy-work/msvc_complex_probe.py`, RawModule including hip_workaround.cuh then `<complex>`): std::exp/cosh/sinh of complex<double> and complex<float>, 4008 inputs (re in [-30,30] and [-750,750], im in [-10,10], plus edge values; float re clipped to +-100). 0 mismatches vs NumPy (rtol 1e-13 double, 2e-6 float) over all outputs whose reference is finite and normal (11717 double, 6722 float). Subnormal outputs differ by double rounding (double) or flush to 0 (float; f32 subnormals are flushed on the device here). Where NumPy overflows in an intermediate (cosh/sinh of 710.27-4.72j and 710.5+0.5j) the device result is finite and correct (~1.6e308), which is the point of the helpers.

Tests at the local tip 694db667f (`agent_space/cupy_test.sh`: CUPY_TEST_GPU_LIMIT=1, CUPY_GPU_MEMORY_LIMIT=25%, ulimit -c 0, `-m "not slow and not multi_gpu"`, one pytest process at a time, test_texture.py ignored, no texture tests run):
- `core_tests/test_elementwise.py -k TestElementwiseGUFuncLike`: **31 passed** (was 23 failed / 8 passed at f4f78903; with only the signbit shim: 21 failed / 10 passed).
- cupyx.scipy.special smoke, one file at a time: voigt_profile 1/1 passed, wright_bessel 1/1, ellipk 5/5, spfun_stats 2 passed 1 skipped, dawsn 1 passed 1 failed, sici 2 passed 2 failed, erfi 1 passed 1 failed, lambertw 0/1, digamma 1 passed 6 failed, wofz 0/1. Baseline without the change: every xsf kernel fails at signbit (spfun_stats re-run at baseline: 2 failed, both `signbit is ambiguous`).
- All 12 remaining failures are one error, `no matching function for call to '<fn>'` with `thrust::complex<T>` arguments against `cxx::complex<T>` (= `std::complex<T>`) candidates: xsf `config.h` maps `cxx::complex` to `thrust::complex` only `#if defined(__CUDACC__)`, and no HIP compile defines `__CUDACC__`, so complex-typed xsf functions cannot take CuPy's complex arrays under HIP on ANY OS. Not Windows-specific and not in this delta; expected to reproduce on the Linux gates (to be confirmed by the linux-gfx1100 baseline, Ruling 7). The real-valued digamma tests fail for the same reason (the ufunc's complex loop is compiled with the rest).
- Changing the header invalidates the kernel cache: the first run of each file recompiles (ellipk took ~3 min).

### Lock: commits not pushed (needs a person)

`set-state cupy windows-gfx1151 porting` refused: `illegal transition ported -> porting` (the stage table only allows `ported -> reviewing`). The fork-write lock is free (`port-lock cupy`: no lock held), but there is no porter entry from `ported`, and `port-lock --take` is documented as a person's decision, so the porter did not push moat-port, did not advance head and did not change state. To land it: a person takes the lock (`python3 utils/moatlib.py port-lock cupy --take windows-gfx1151`) or otherwise rules how a porter adds commits at `ported`; then fast-forward push moat-port f4f78903 -> 694db667f, `advance-head`, release. The two commits live only in the laptop clone `D:/Develop/moat/agent_space/wt-cupy/projects/cupy/src` (branch moat-port, 2 ahead of origin/moat-port). jargon.py: no hits in the two new commits (its existing hits are in upstream history). The CUDA Windows build comparison (Ruling 4) was not rerun: carray.cuh is a run-time header, not compiled at build time, and hip_workaround.cuh is HIP-only; NVRTC evidence above.

### ROCm bug report drafts (not filed)

Reproducer files on the laptop: `D:/Develop/moat/agent_space/cupy-rocm-bugs/` (hiprtc_repro.cpp, repro_signbit.cu, repro_exp.cu, repro_cmath.cu, k.cu, shim.h, hipcc_quote.py, q.cu, `dir with space/hdr.h`). All three confirmed on this laptop 2026-11-05 with TheRock ROCm 7.14.0a20260612 (pip `rocm[libraries,devel]`, gfx1151), HIP 7.14.60850-1b2a555677, AMD clang 23.0.0git (llvm-project 52226beb248f), comgr daab70a62bfc, MSVC 14.44.35207, Windows SDK 10.0.22621, Windows 11.

The host program shared by drafts 1 and 2 (`hiprtc_repro.cpp`), built with `clang-cl /EHsc /MD /std:c++17 -D__HIP_PLATFORM_AMD__ -I<rocm>\include hiprtc_repro.cpp hiprtc.lib`:

```cpp
// hiprtc_repro <source.cu> <program name> [hiprtc options...]
#include <hip/hiprtc.h>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
int main(int argc, char** argv) {
  std::ifstream f(argv[1]);
  std::stringstream ss;
  ss << f.rdbuf();
  std::string src = ss.str();
  hiprtcProgram prog;
  hiprtcCreateProgram(&prog, src.c_str(), argv[2], 0, nullptr, nullptr);
  std::vector<const char*> opts(argv + 3, argv + argc);
  hiprtcResult rc = hiprtcCompileProgram(prog, (int)opts.size(), opts.data());
  size_t n = 0;
  hiprtcGetProgramLogSize(prog, &n);
  std::string log(n, '\0');
  hiprtcGetProgramLog(prog, &log[0]);
  std::printf("hiprtcCompileProgram: %s\n%s\n", hiprtcGetErrorString(rc), log.c_str());
  return rc != HIPRTC_SUCCESS;
}
```

#### Draft 1: hiprtc on Windows cannot compile `<complex>` from the MSVC STL

Title: [hiprtc][Windows] `#include <complex>` fails: `call to 'signbit' is ambiguous` (MSVC STL), then `undefined hidden symbol: _Exp`

On Windows, hiprtc compiles device code against whatever C++ standard library the include path provides, and ROCm ships no libc++ headers there, so a kernel that includes `<complex>` is compiled against the MSVC STL (the include dirs `hipcc -x hip -E -v NUL` reports: clang `cuda_wrappers`, the clang resource dir, MSVC `include`, the Windows SDK `ucrt`, ...). That fails in two ways. Libraries that compile C++ kernels at run time (CuPy, through SciPy's xsf special-function headers and its mdspan-based kernels) hit this on every kernel that includes a standard header.

Kernel `repro_signbit.cu`:

```cpp
#include <complex>
extern "C" __global__ void k(double* a) { a[0] = std::abs(std::complex<double>(a[0], a[1])); }
```

Run with the MSVC include dirs as `-I` options (here the 12 unique dirs from `hipcc -x hip -E -v NUL`): `hiprtc_repro.exe repro_signbit.cu repro_signbit.cu --offload-arch=gfx1151 -I<dir1> ... -I<dir12>`

```
hiprtcCompileProgram: HIPRTC_ERROR_COMPILATION
In file included from ...\comgr-XXXX\input\repro_signbit.cu:1:
In file included from ...\_rocm_sdk_devel\lib\llvm\lib\clang\23\include\cuda_wrappers\complex:82:
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\include\complex:575:16: error: call to 'signbit' is ambiguous
  575 |         return (::std:: signbit)(_Left);
...\comgr-XXXX\include\hiprtc_runtime.h:1083:75: note: candidate function
 1083 | static __attribute__((device)) inline __attribute__((always_inline)) bool signbit(float __x) { return ::__signbitf(__x); }
...\comgr-XXXX\include\hiprtc_runtime.h:1084:75: note: candidate function
 1084 | static __attribute__((device)) inline __attribute__((always_inline)) bool signbit(double __x) { return ::__signbit(__x); }
1 error generated when compiling for gfx1151.
```

The call is `_Ctraits<long double>::_Signbit`, i.e. `std::signbit(long double)`; hiprtc_runtime.h declares only the float and double overloads, so the long double argument is ambiguous. With that worked around (a `signbit(long double)` overload declared before `<complex>`, `shim.h`, passed as `-include shim.h`), the next failure is at link, from `repro_exp.cu`:

```cpp
#include <complex>
extern "C" __global__ void k(double* a) { auto z = std::exp(std::complex<double>(a[0], a[1])); a[0] = z.real(); a[1] = z.imag(); }
```

```
hiprtcCompileProgram: HIPRTC_ERROR_COMPILATION
ld.lld: error: undefined hidden symbol: _Exp
>>> referenced by ...\comgr-1844-1-15d9fb\input\repro_exp.o:(k)
```

MSVC's `std::exp`, `cosh` and `sinh` for `complex<float>` and `complex<double>` call the UCRT helpers `_Exp`, `_Cosh`, `_Sinh`, `_FExp`, `_FCosh`, `_FSinh` (declared in `ymath.h`), which have no device definition. `<cmath>` alone compiles (`repro_cmath.cu`: `std::signbit`, `std::exp`, `std::cosh` on double).

Expected: `<complex>` compiles and links for the device under hiprtc on Windows, as it does on Linux with libstdc++.

What a fix could look like: (a) in hiprtc_runtime.h (and the HIP clang wrapper headers, if offline compilation is affected the same way), when targeting the MSVC environment, add `signbit(long double)` and device definitions of the six UCRT helpers (`y * exp(x) * 2^eoff`, `y * cosh(x)`, `y * sinh(x)` with the UCRT's overflow avoidance and special-value handling), the way the clang CUDA/HIP wrapper headers already supply device versions of other C runtime functions; or (b) ship libc++ headers with ROCm on Windows and have hiprtc use them for device code. A user-side workaround that works today: a header included before `<complex>`, guarded by `__HIPCC_RTC__ && _MSC_VER`, providing the overload and the six helpers as `__device__ __attribute__((weak))` functions bound to the C names with `__asm__("_Exp")` etc. (a plain `extern "C" __device__ _Exp` definition is rejected, because `cuda_wrappers/complex` forces the STL declarations to `__host__ __device__`).

#### Draft 2: hiprtc on Windows fails with an empty log when the program name is an absolute path

Title: [hiprtc][Windows] `hiprtcCompileProgram` returns HIPRTC_ERROR_COMPILATION with an empty log when the program name is an absolute path

`hiprtcCreateProgram`'s `name` argument names the program (it is used in diagnostics). On Windows, an absolute path there makes the compile fail with no log; the same source compiles with a relative name. CuPy and similar libraries pass the full path of a cache file as the name.

Kernel `k.cu`:

```cpp
extern "C" __global__ void k(float* a) { a[threadIdx.x] += 1.0f; }
```

```
> hiprtc_repro.exe k.cu C:\Users\me\k.cu --offload-arch=gfx1151
hiprtcCompileProgram: HIPRTC_ERROR_COMPILATION

> hiprtc_repro.exe k.cu k.cu --offload-arch=gfx1151
hiprtcCompileProgram: HIPRTC_SUCCESS
```

`C:/Users/me/k.cu` fails the same way; `sub\k.cu` and `sub/k.cu` succeed. Same result with `AMD_COMGR_USE_VFS=0`. With `AMD_COMGR_EMIT_VERBOSE_LOGS=1 AMD_COMGR_REDIRECT_LOGS=stderr` the only signal is `AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE ... ReturnStatus: AMD_COMGR_STATUS_ERROR`, with no diagnostic. A successful compile reports its source as `...\comgr-XXXX\input\k.cu`, so the likely cause is joining comgr's temporary input directory with an absolute Windows name (a drive letter inside a path), which is harmless on Linux (`/tmp/x/input//home/...`) and invalid on Windows.

Expected: any name compiles, or at least an error says the name is invalid. A fix: use only the file name component (or a sanitized name) for the temporary file and keep the full name for diagnostics.

#### Draft 3: hipcc.exe does not quote the arguments it forwards to clang

Title: [hipcc][Windows] `hipcc.exe` splits forwarded arguments that contain spaces or quotes

`hipcc.exe` rebuilds the clang command line as one string without quoting the arguments it forwards, so an argument that contains a space or a double quote is split when clang parses its command line. A define such as `-DEXTERN_C=extern "C"` and an include directory under `C:\Program Files (x86)\...` (where the MSVC and Windows SDK headers live) both break.

`hipcc_quote.py` passes the same argument vector (a list, so each element is one argv entry) to `hipcc.exe` and to `clang++.exe`:

```python
args = ['-x', 'hip', '--offload-arch=gfx1151', '-c', 'q.cu', '-o', 'q.o',
        '-DEXTERN_C=extern "C"', '-I' + os.path.join(here, 'dir with space')]
```

with `q.cu` (and `dir with space/hdr.h` defining `HDR_OK`):

```cpp
#include <hip/hip_runtime.h>
#include "hdr.h"
EXTERN_C __global__ void k(float* a) { a[threadIdx.x] += HDR_OK; }
```

```
hipcc.exe rc 1
clang: error: no such file or directory: 'C'
clang: error: no such file or directory: 'with'
clang: error: no such file or directory: 'space'
failed to execute:""...\lib\llvm\bin\clang.exe"   --offload-arch=gfx1151 -O3  -x hip -c "q.cu" -o "q.o" -DEXTERN_C=extern "C" -IC:\Users\me\rtcbug\dir with space"

lib\llvm\bin\clang++.exe rc 0
```

Expected: hipcc forwards each argument unchanged, as clang++.exe receives it. A fix: quote each forwarded argument for the Windows command line (surround with `"` and escape embedded `"` and trailing backslashes per the CommandLineToArgvW rules), or start clang with an argument vector instead of a joined string.

## Review 2026-11-06 (linux-gfx1100, reviewer)

Scope: `git diff 18b54de...694db667f` on ROCm/cupy `moat-port` (14 commits), pr-review skill, with each finding re-checked against the code. Verdict: **changes requested** (1 finding).

### Port Correctness

1. **The Windows ROCm root is found only when `hipconfig` is on `PATH`.** See `cupy/_environment.py:240` and `install/cupy_builder/install_build.py:71`: both call `shutil.which('hipconfig')`. TheRock installs `hipconfig.exe` as a console script in the venv's `Scripts` directory (plan, feasibility probe), and that directory is on `PATH` only when the venv is activated. Suppose a user runs `C:\venv\Scripts\python.exe -c "import cupy"` without activating (an IDE, a scheduled task, a service) and has not set `ROCM_HOME`, which install.rst calls optional. Then `get_rocm_path()` returns None and `_setup_win32_hip_dll_directory` (`cupy/_environment.py:614-616`) returns silently without adding `<root>/bin`. The first extension import then fails with a bare `DLL load failed`, and nothing names ROCm or `ROCM_HOME`. hiprtc and runtime hipcc are left without a root as well. The same happens at build time with `--no-build-isolation` from an unactivated venv: it fails with `clang-cl could not be found: lib\llvm\bin\clang-cl.exe`. Fix, within Ruling 1 (still `ROCM_HOME`, then `hipconfig --rocmpath`): when `which` finds nothing on `PATH`, also look for `hipconfig` in `sysconfig.get_path('scripts')` of the running interpreter, in both helpers. Optionally, also make the HIP branch of the Windows DLL-load diagnostic (`_diagnose_win32_dll_load`) name `ROCM_HOME`. This was found by reading the code; it was not reproduced here, because this host has no Windows. The windows-gfx1151 porter should reproduce it with an unactivated venv before and after the fix.

### Checked, with no finding (evidence for the next stage)

- Every builder and runtime change is gated on win32, `_WIN32`, `ctx.use_hip` together with win32, or `__HIPCC_RTC__ && _MSC_VER`, or is identical on Linux: `os.devnull`; `CUPY_HIP_EXPORT` is empty off Windows; `rocblas_stride` is `int64_t`, which is `long` on LP64, and `cupy_lapack.h:136` sits inside the HIP-only branch; `<string>` was already pulled in transitively; adding the `cupy_hip_common.h` include to `cupy_hipsparse.h` leaves the build identical (below). The `_command.py` `_depends.json` copy sits inside `if sys.platform == 'win32'`. Every caller of `get_rocm_path` is HIP-gated, so the CUDA Windows build never runs `hipconfig`. The CUDA Windows branches (the nvcc `DeviceCompilerWin32.compile` path, the `/std:c++17` and NVTX logic) are byte-identical; the porter's laptop comparison at f4f78903 (134 identical command lines) covers every build-time change, and 4ccded519 and 694db667f touch only run-time headers (`carray.cuh` is read by NVRTC and hiprtc only; it is never compiled at build time).
- Export set compared against the lookups: every `.get()` name in `_driver_extern.pxi` and `_runtime_softlink.pxi`, and every name in `cusparse.pyx` that has a HIP stub, carries `CUPY_HIP_EXPORT`. The `cusparseSpGEAM*` names have no HIP stub on any OS, so behaviour is unchanged. The cuSOLVER and cuFFT SoftLinks do not resolve against `__file__` on HIP.
- The MSVC STL helpers in `hip_workaround.cuh` were compared with the UCRT behaviour (microsoft/STL `xexp.cpp`, `xcosh.cpp`, `xsinh.cpp`). The return codes are 0/1/-1 for zero, infinity and finite (as `_Dscale`). The ln2 hi/lo split constants are the fdlibm values. The NaN, infinity and zero cases of `_Cosh` and `_Sinh` match, and the switch to the exp path at |x| >= 22 adds error below double epsilon (e^-44). The block is inert on Linux and for CUDA (`__HIPCC_RTC__ && _MSC_VER`).
- Commit hygiene: all 14 titles start with `[ROCm]` and are at most 72 characters; no Co-Authored-By trailer; every body has an AI disclosure and a Test Plan; ASCII only; `python3 utils/jargon.py --port cupy` reports clean (it needs a local `main` ref in the clone: `git -C projects/cupy/src branch main origin/main`).

### Linux HIP no-change check (this host: TheRock pip ROCm, HIP 7.16.26385, AMD clang 24.0.0git, gfx1100 W7800, Python 3.11, Cython 3.2.9, numpy 2.4.6)

The commands below build the port and the baseline in-place, in two worktrees under `agent_space/` (`wt-port` at 694db667f and `wt-base` at 18b54de). The same command was run in each tree, with the job venv and `CUPY_NUM_BUILD_JOBS=16`:

```bash
export ROCM_HOME=/opt/python-3.11-venv/lib/python3.11/site-packages/_rocm_sdk_devel
export PATH="$ROCM_HOME/bin:$PATH" LD_LIBRARY_PATH="$ROCM_HOME/lib" CUPY_INSTALL_USE_HIP=1 HCC_AMDGPU_TARGET=gfx1100
utils/timeit.sh cupy compile -- bash -c "cd <wt> && python setup.py build_ext --inplace -v"
```

- Both trees build (EXIT 0) with the same configure summary: cuda Yes 71626385, cub Yes 400700, nccl Yes 23007, random Yes, thrust Yes, dlpack Yes. Both produce the same set of 128 `.so` files and the same warning count (472).
- All 132 compile and link command lines are identical, once the tree path and `CUPY_CACHE_KEY` (a source hash) are normalised. The dynamic symbol tables of all `.so` files are identical apart from the three path-derived `__hip_cuid_*` symbols.
- Smoke, run with `PYTHONPATH=<wt>` and `ulimit -c 0` on the pinned GPU: plan smoke (elementwise, reduction, RawKernel, CUB via runtime hipcc, hipRAND, rocSOLVER, hipFFT, hipBLAS, hipSPARSE, rocThrust, nvtx, profiler), plus complex exp at 700.5+0.5j against NumPy. **Pass on both.**
- `tests/install_tests tests/import_tests` with `CUPY_INSTALL_USE_HIP=1`: 31 passed, 1 skipped, on both.
- Shared commit 4ccded519, `python -m pytest tests/cupy_tests/core_tests/test_elementwise.py -k TestElementwiseGUFuncLike`: base **21 failed, 10 passed** (111 x `'data_'/'shape_' is a private member of 'CArray<...>'`), port **31 passed**. The porter's prediction is confirmed: the bug is not specific to Windows, and Linux ROCm needs the fix too. The full Linux baseline (Ruling 7) remains the validators' work.

## Porter round 2026-11-06 (windows-gfx1151): review finding 1 (hipconfig lookup without an activated venv)

moat-port 694db667f -> **24a62ef95** `[ROCm] Find hipconfig without an activated environment on Windows` (one new commit, fast-forward push, no amend). Files: `cupy/_environment.py`, `install/cupy_builder/install_build.py`, `docs/source/install.rst` (+40/-15).

Change (Ruling 1 kept: ROCM_HOME, then hipconfig; HIP_PATH still ignored): both `_get_win32_rocm_path` helpers fall back to `shutil.which('hipconfig', path=sysconfig.get_path('scripts'))` when hipconfig is not on PATH. When no root is found, a HIP build warns at import (`_setup_win32_hip_dll_directory`, text names ROCM_HOME and `rocm-sdk path --root`), `_diagnose_win32_dll_load` prints `ROCm Path:` plus the same hint for a HIP build (CUDA builds keep `CUDA Path:`), and `configure_msvc_for_hip` raises a ROCM_HOME-naming error instead of the relative clang-cl path. All in win32-only / HIP-only functions; Linux and CUDA builds unchanged (no CUDA compile-command change, so the nvcc comparison of Ruling 4 is not affected).

Reproduction (unactivated: `env -u ROCM_HOME -u VIRTUAL_ENV PATH="/c/Windows/System32:/c/Windows" C:/Users/jdaily/cupy-work/venv/Scripts/python.exe ...`; build PATH = MSVC bin + /usr/bin + Windows dirs):
- Symptom 1 differs from the review's prediction: there is no bare `DLL load failed` here. System32 carries the display driver's `amdhip64_7.dll` (HIP 7.12, runtime 71260602), so `import cupy` succeeds against that DLL and the first kernel fails with `RuntimeError: CuPy failed to load hiprtc0712.dll` (the hiprtc name is derived from the runtime version). Any machine with an AMD driver behaves this way, which is worse than a load failure: CuPy would run on the System32 runtime. After the fix: `amdhip64_7.dll` and `hiprtc0714.dll` load from `<root>\bin` (runtime 71460850), and kernels run.
- Symptom 2 reproduced exactly: `pip install -v --no-build-isolation --no-deps -e .` -> `RuntimeError: clang-cl could not be found: lib\llvm\bin\clang-cl.exe`. After: EXIT 0, clang-cl from `<root>\lib\llvm\bin` (log C:/Users/jdaily/cupy-work/build-unact-after.log).
- No-root path, checked by hiding hipconfig from `shutil.which` (C:/Users/jdaily/cupy-work/noroot_probe.py, noroot_build_probe.py): import warning shown, diagnostic prints `ROCm Path: None` + hint, the build helper raises the ROCM_HOME message.
- Gotcha: any setup.py run (even a failing build) clears `cupy/.data` (upstream `prepare_wheel_libs`), deleting `_depends.json` from an editable install's source tree. The HIP DLL directory is then silently not added and CuPy binds the System32 driver runtime (as above) until the next successful build. Seen here after the failed "before" build. Upstream behaviour, not changed.

Tests at 24a62ef95 (laptop, ulimit -c 0, one pytest process, no texture tests):
- smoke.py from the unactivated venv, ROCM_HOME unset: all pass (elementwise+reduction, RawKernel, CUB via runtime hipcc, random, eigvalsh/solve, FFT, matmul, sparse, sort, nvtx, profiler).
- `agent_space/cupy_test.sh cuda-hipconfig tests/cupy_tests/cuda_tests`: **188 passed, 3 failed, 81 skipped, 28 deselected**, same three known failures as at f4f78903 (test_get_limit from the memory cap, madvise, pointer memory type).
- `tests/install_tests tests/import_tests`: 31 passed, 1 skipped.
- flake8 on the two files: only pre-existing F824 in install_build.py.
- jargon.py --port cupy: clean, after pointing the clone's local `main` at the upstream base 18b54de. ROCm/cupy's own `main` is NOT an upstream mirror (32k commits off), so `git branch main origin/main`, as the review note suggests, gives 20 false hits from upstream history.
