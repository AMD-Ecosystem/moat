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

## Review 2026-11-06b (linux-gfx1201, reviewer)

Scope: `git diff 694db667f...24a62ef95` on ROCm/cupy `moat-port` (one commit, `[ROCm] Find hipconfig without an activated environment on Windows`), pr-review skill, each point re-checked against the code. Verdict: **review passed**, no findings. The commit resolves finding 1 of the 2026-11-06 review within Ruling 1: the lookup order is still ROCM_HOME, then `hipconfig --rocmpath`, and HIP_PATH is not read anywhere.

### Checked, with no finding

- Every behavioural change is Windows- or HIP-only:
  - `cupy/_environment.py`: `_get_win32_rocm_path` is called only under `_PLATFORM_WIN32` (`_get_rocm_path`, line 221). `_setup_win32_hip_dll_directory` returns early off win32, and the new warning sits after the `_is_win32_hip_build` check, so a CUDA Windows import is unchanged. `_diagnose_win32_dll_load` is reached only from `_diagnose_import_error` under `sys.platform == 'win32'`. For a CUDA build it emits the same lines as before (`''`, `''`, `CUDA Path: ...`, `DLL dependencies:`).
  - `install/cupy_builder/install_build.py`: `_get_win32_rocm_path` is reached only under `PLATFORM_WIN32` (line 63). `configure_msvc_for_hip` has two callers, `_command.py:221` (win32 MSVC and `ctx.use_hip`) and `cupy_setup_build.py:540` (`ctx.use_hip and PLATFORM_WIN32`). On every OS, the only new effect is `import sysconfig` (stdlib).
- `if not rocm_path` in `configure_msvc_for_hip` covers both the `''` the build helper returns and an empty ROCM_HOME. The runtime `get_rocm_path` caches None correctly: its sentinel is `''`.
- I exercised the logic on Linux with `sysconfig.get_path('scripts')` mocked to a directory holding a fake `hipconfig`, `PATH=/nonexistent` and ROCM_HOME unset (`agent_space/probe_lookup.py`):
  - Both helpers return the fake root.
  - With no scripts directory, the runtime helper returns None and the build helper returns `''`.
  - `configure_msvc_for_hip` raises the ROCM_HOME message.
  - `_setup_win32_hip_dll_directory`, forced to win32 with an `amdhip64_7.dll` depends list, emits the ROCM_HOME warning.
- `shutil.which(..., path=...)` applies PATHEXT on Windows. A venv's `sysconfig.get_path('scripts')` is `<venv>\Scripts`. Searching the current directory first is unchanged from the existing on-PATH lookup.
- Commit hygiene: the title is 65 characters with `[ROCm]`. There is no Co-Authored-By or noreply trailer. The body has an AI disclosure and a Test Plan with fenced commands. The body and the diff are ASCII. `python3 utils/jargon.py --port cupy` reports clean, with the clone's local `main` at 18b54de. That follows the porter's correction: ROCm/cupy's `main` is not an upstream mirror, so `branch main origin/main` is wrong.
- install.rst: the added sentence describes the lookup as the code implements it.

### Linux HIP no-change check (this host: gfx1201, TheRock pip ROCm, HIP 71626385, Python 3.11.17, Cython 3.2.9, numpy 2.4.6 in the job venv)

The same command was run in two worktrees, `agent_space/wt-base` (694db667f) and `agent_space/wt-port` (24a62ef95):

```bash
export ROCM_HOME=/opt/python-3.11-venv/lib/python3.11/site-packages/_rocm_sdk_devel
export PATH="$FLEET_VENV/bin:$ROCM_HOME/bin:$PATH" LD_LIBRARY_PATH="$ROCM_HOME/lib" CUPY_INSTALL_USE_HIP=1 HCC_AMDGPU_TARGET=gfx1201 CUPY_NUM_BUILD_JOBS=12
utils/timeit.sh cupy compile -- bash -c "cd <wt> && python setup.py build_ext --inplace -v"
```

- Both trees build (EXIT 0) with the same configure summary: cuda Yes 71626385, cub Yes 400700, nccl Yes 23007, random, thrust and dlpack Yes. Both have 455 warnings and 64 `.so` files.
- All 516 compile and link command lines are identical after normalising the tree path and `CUPY_CACHE_KEY`. The `nm -D --defined-only` output for every `.so` is identical (6930 lines), with `__hip_cuid_*` excluded.
- Smoke on the port tree, under `ulimit -c 0` with `HIP_VISIBLE_DEVICES=0`, importing cupy with warnings set to errors: **pass**. It covered elementwise, reduction, RawKernel, CUB, random, eigvalsh/solve, FFT, sparse matmul, sort, nvtx, profiler, and complex exp at 700.5+0.5j against NumPy.
- `tests/install_tests tests/import_tests`: 31 passed, 1 skipped.
- Host noise, unrelated to the port: when cupyx.scipy.sparse imports SciPy, the image's shared scipy 1.10 (built for numpy 1.x) prints `_ARRAY_API not found`. cupyx catches the error and carries on.
- No CUDA compile was run. The commit changes no compiled source and no CUDA build path, and the build helper's new branch is HIP-only, so the nvcc comparison of Ruling 4 still applies.

### Porter note: a failed setup.py run removes `cupy/.data/_depends.json`

Confirmed by reading the code. `prepare_wheel_libs` (`install/cupy_builder/cupy_setup_build.py:481-490`) runs from `setup.py:53` on every invocation, before any build, and runs `rmtree` on `cupy/.data`. The Windows-only copy in `_command.py:257-271` restores `_depends.json` only after a successful `build_ext`.

**The port should not guard against it** (no finding):
- It affects only a source tree used in place (editable or `--inplace` builds) after a failed run. Wheels and regular installs carry their own `cupy/.data` in site-packages, and the next successful build restores the file.
- This is upstream behaviour, and it already affects CUDA Windows the same way: with no `_depends.json`, the import diagnostic is silently dropped.
- A guard would need a HIP marker that survives the `rmtree`. Either another file outside `.data`, or importing a compiled module to ask `is_hip` before the DLL directory exists, which is the very load this function enables. That adds upstream-visible surface to fix a transient developer-tree state that upstream's own build creates.
- It is enough to keep this in notes as a gotcha for anyone validating on Windows: after a failed build, rebuild before testing, or CuPy binds the driver's System32 `amdhip64_7.dll`.

## CUDA compile check (Linux) at 24a62ef95

2026-11-06, linux-gfx950 fleet host, no role taken and no state changed (Ruling 4 no-regression evidence for the Linux CUDA path). CUDA 12.8 (`nvcc` V12.8.93, `/usr/local/cuda-12.8`), g++ 13.3.0, Python 3.11.17, Cython 3.2.9, numpy 2.4.6, setuptools 79.0.1 in the job venv. No NVIDIA driver or GPU on the host; nothing was run on a GPU.

Two worktrees of one ROCm/cupy clone, `wt-base` at upstream 18b54de and `wt-port` at moat-port 24a62ef95 (submodules cccl 4ee3a7b, dlpack cd0d5e4, jitify 1a0ca0e, xsf 2e1e697 in both), built in parallel with the same script:

```bash
export TMPDIR=<job>/cudachk/tmp-<wt>
export CUDA_PATH=/usr/local/cuda-12.8 PATH=$FLEET_VENV/bin:/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64 LIBRARY_PATH=/usr/local/cuda-12.8/lib64/stubs
export CUPY_NVCC_GENERATE_CODE=arch=compute_80,code=sm_80 CUPY_NUM_BUILD_JOBS=48
unset CUPY_INSTALL_USE_HIP ROCM_HOME
cd <wt> && python -m pip wheel -v --no-build-isolation --no-deps -w <job>/cudachk/whl-<wt> .
```

- `LIBRARY_PATH=.../lib64/stubs` is needed on a host without the NVIDIA driver: without `libcuda.so` the jitify configure probe fails to link and the build aborts with `Your CUDA environment is invalid` (seen identically in both trees on the first attempt).
- Both build (EXIT 0, about 135 s each). Identical configure summary: cuda Yes (12080), cusolver Yes, nvtx Yes, cub Yes (300102), jitify Yes (1a0ca0e), random Yes, thrust Yes (300102), dlpack Yes; nccl, cutensor, cusparselt No (headers not in this toolkit).
- All 133 compile and link command lines are identical after normalising the tree path, the temp dir and `CUPY_CACHE_KEY` (sorted, byte-compared): 65 g++ compiles, 65 g++ links, 3 nvcc compiles (`cupy_cub.cu`, `cupy_thrust.cu`, `cupy_distributions.cu`). `CUPY_CACHE_KEY` differs as expected (c5550207 vs bad50f9b), since the port changes `carray.cuh` and `hip_workaround.cuh`.
- Same warning lines (255 each, identical after path normalisation).
- Wheels: identical file list (65 `.so`). `nm -D --defined-only` of every `.so` identical (2247 lines), and the `NEEDED` entries are identical. The only files that differ are the expected run-time sources: `cupy/__init__.py`, `cupy/_environment.py`, `cupy/cuda/compiler.py`, `carray.cuh`, `hip_workaround.cuh` (plus `RECORD`).
- Verdict: **no CUDA Linux build regression at 24a62ef95.** Build trees, wheels and worktrees under the job directory were deleted afterwards.
## Validation 2026-11-06 (linux-gfx90a, validator): full upstream baseline per Ruling 7

Port: ROCm/cupy `moat-port` at **24a62ef95**. Baseline: upstream **18b54de** (merge base). Host: MI210-class gfx90a (`gfx90a:sramecc+:xnack-`, wave64), TheRock pip ROCm (rocm-sdk 10.1.0, HIP runtime 71626385, hipconfig 7.16.26385), Python 3.11.17, Cython 3.2.9, numpy 2.4.6, scipy 1.17.1, pytest 7.3.2, pytest-xdist 3.3.1, pytest-timeout, hypothesis 6.54.6, optuna (all in the job venv).

### Build (same toolchain and flags for both trees)

```bash
export ROCM_HOME=<venv>/../_rocm_sdk_devel     # rocm-sdk devel root of the venv
export PATH="$VENV/bin:$ROCM_HOME/bin:$PATH" LD_LIBRARY_PATH="$ROCM_HOME/lib"
export CUPY_INSTALL_USE_HIP=1 HCC_AMDGPU_TARGET=gfx90a CUPY_NUM_BUILD_JOBS=32
utils/timeit.sh cupy compile -- bash -c "cd projects/cupy/src && python setup.py build_ext --inplace -v"   # port
(cd agent_space/wt-base && python setup.py build_ext --inplace -v)                                          # baseline worktree at 18b54de
```

Both EXIT 0, identical configure summary (cuda Yes 71626385, cub Yes 400700, nccl Yes 23007, random Yes, thrust Yes, dlpack Yes), 472 warnings each, the same 64 `.so` files. Plan smoke (elementwise, reduction, RawKernel, CUB, hipRAND, rocSOLVER, hipFFT, hipBLAS, hipSPARSE, rocThrust, nvtx, profiler) passes on both. `tests/install_tests tests/import_tests` (CUPY_INSTALL_USE_HIP=1): 31 passed, 1 skipped on both.

### Full test suite

Every subdirectory of `tests/cupy_tests` and `tests/cupyx_tests` (and of `cupyx_tests/scipy_tests`), plus each top-level `test_*.py`, as its own pytest process, one process at a time, alternating port and baseline per target. No marker filter (slow tests included; multi_gpu tests skip under the GPU limit). The GPU (64 GB) peaked near 8 GB with 16 workers, so `-n 16` was used (`-n auto` would have meant 128 contexts on one GPU). Environment as upstream's ROCm CI (`.pfnci/linux/tests/rocm-7-14.sh`):

```bash
ulimit -c 0
export CUPY_TEST_GPU_LIMIT=1 CUPY_TEST_FULL_COMBINATION=0 CUPY_CACHE_DIR=<per-tree dir>
utils/timeit.sh cupy test -- bash -c 'cd "$1" && shift && exec "$@"' _ <tree> \
  timeout --signal INT --kill-after 60 14400 python -m pytest -n 16 -rfEX --timeout 600 -p no:cacheprovider -q --junitxml=<out>.xml <target>
```

Totals over the junit records (one record per test id, worst outcome kept):

| tree | passed | failed | error | skipped | xfail |
|---|---|---|---|---|---|
| port 24a62ef95 | 98727 | 385 | 1 | 7768 | 4726 |
| base 18b54de | 98705 | 407 | 1 | 7768 | 4726 |

Same test-id set on both trees. Per-directory counts were identical on both trees except core_tests (port 130 failed / base 151), special_tests (13 / 15) and ndimage_tests (1 / 0). Skip counts in sparse_tests (port 2 fewer) and ndimage_tests (port 2 more) differ in opposite directions and cancel in the totals.

### Classification

- **Identical failures (upstream's): 385** (384 failures + the 1 collection error `statistics_tests/test_histogram.py::TestCubHistogram`, which xdist reports as 16 errors). By class: test_bfloat16 127; interpolate TestInterp1D 105, TestNdBSpline 21, TestCubicSpline 14, TestRegularGridInterpolator 1; sparse TestSpsolveTriangular 24, sparray/int64-index/LOBPCG 17; signal TestChirp 8; cupy fft 6 and cupyx.scipy fft 6; TestCumsum 6, TestCumprod 6, TestBatchScanSizeOverInt32Max 4; TestSVD 4; test_cusparse TestCheckAvailabilityVersionSkew 4; test_init 4; test_optimize 3; TestPolyval 2; TestUfuncSlow 2; TestArrayElementwiseOp 2; one each in binary TestElementwise (`std::is_signed` not found under hiprtc), dlpack, cuda_tests test_runtime, random TestRandomState, testing TestPackageRequirements; and the 13 complex xsf calls below.
- **New on the port: none.** One test failed on the port run only: `cupyx_tests/scipy_tests/ndimage_tests/test_interpolation.py::TestZoom::test_zoom_float[_param_19_{cval=1.0, mode='nearest', order=0, output=None, prefilter=True, zoom=10}]` (48 of 1e6 elements wrong, e.g. 1.49e-10 instead of 6.66). It did not reproduce: `-k "TestZoom and test_zoom_float"` serially, twice on each tree, 54 passed every time; the whole `test_interpolation.py` with `-n 16` on the port: 5203 passed. On Linux the port changes only the `__device__` friend in `carray.cuh` (no code generated) and an MSVC-only block in `hip_workaround.cuh`, and the build is identical to the baseline, so this is an intermittent on this host, not a port regression.
- **Fixed by the port: 23**, all from 4ccded519 (`__device__` on the `as_mdspan` friend of CArray):
  - `core_tests/test_elementwise.py::TestElementwiseGUFuncLike`: **21 fail on the baseline, pass on the port** (all 31 tests of the class pass on the port). Baseline error: 111 x `'data_'/'shape_'/'strides_' is a private member of 'CArray<...>'`; 0 in the port log. This confirms the gfx1100 reviewer's finding on gfx90a.
  - `special_tests/test_spfun_stats.py::TestPoissonBinomCDF::test_values` and `::test_scalar_k`: fail on the baseline with the same CArray private-member error, pass on the port. These are new here: no earlier record lists them.
- **The 12 complex xsf calls** (the Windows porter's 12 remaining special failures) fail identically on Linux, on both trees: TestErfi::test_complex, TestDawsn::test_complex, TestLambertW::test_values, TestDigamma x6 (test_complex, test_scalar, test_arange, test_linspace_positive, test_linspace_negative, test_inf_and_nan), TestSici::test_complex, TestShichi::test_complex, TestWofz::test_complex. A 13th test with the same cause, `test_ufunc_dispatch.py::TestUfunc::test_dispatch[wofz]`, also fails on both trees. Every error is `no matching function for call to '<dawsn|digamma|erfi|lambertw|sici|shichi|wofz>'`, with candidates `cxx::complex<T>` (= `std::complex<T>`) and arguments `thrust::complex<T>`: xsf maps `cxx::complex` to `thrust::complex` only when `__CUDACC__` is defined, and no HIP compile defines it. This is upstream's, not in this delta, and the same on every OS.

### Other gates

- CUDA compile gate: not run on this host (no CUDA toolkit here). No commit since the Ruling 4 nvcc comparison changes a build-time CUDA path (reviews 2026-11-06 and 2026-11-06b).
- Jargon: `python3 utils/jargon.py --port cupy` reports clean (local `main` at 18b54de). Docs: `docs/source/install.rst` has the Windows ROCm section; the Linux ROCm build docs are upstream's.
- Integrity: `git -C projects/cupy/src status --porcelain` shows no modified tracked files (only in-place build output).

**Verdict: linux-gfx90a completed at 24a62ef95.** The port behaves as upstream on Linux, apart from the 23 tests that 4ccded519 fixes.
## Validation 2026-11-06 (linux-gfx1201, validator): full upstream baseline (Ruling 7)

Verdict: **pass, no port regression on Linux.** Port: ROCm/cupy `moat-port` 24a62ef95. Baseline: upstream 18b54de. Host: gfx1201 (Radeon, 32 GiB VRAM, one GPU visible, `HIP_VISIBLE_DEVICES=0`), TheRock pip ROCm `_rocm_sdk_devel` (HIP 7.16.26385, runtime 71626385), Python 3.11.17, job venv with Cython 3.2.9, numpy 2.4.6, scipy 1.17.1, pytest 7.3.2, hypothesis 6.54.6, pytest-xdist 3.3.1, pytest-timeout. Same submodule pins in both trees.

### Build (both trees, same command, in-place, worktrees `agent_space/wt-port` and `agent_space/wt-base`)

```bash
export ROCM_HOME=/opt/python-3.11-venv/lib/python3.11/site-packages/_rocm_sdk_devel
export PATH="$FLEET_VENV/bin:$ROCM_HOME/bin:$PATH" LD_LIBRARY_PATH="$ROCM_HOME/lib"
export CUPY_INSTALL_USE_HIP=1 HCC_AMDGPU_TARGET=gfx1201 CUPY_NUM_BUILD_JOBS=12
utils/timeit.sh cupy compile -- bash -c "cd <wt> && python setup.py build_ext --inplace -v"
```

- Both EXIT 0. Same configure summary (cuda Yes 71626385, cub Yes 400700, nccl Yes 23007, random Yes, thrust Yes, dlpack Yes), same 640 compile/link lines after normalising the tree path and `CUPY_CACHE_KEY`, 474 warnings each. `nm -D --defined-only` symbol names identical for every `.so` (`__hip_cuid_*` excluded); only `cupy_backends/cuda/libs/nvrtc.so` differs, in the addresses of seven `std::vector` weak symbols, from the win32-gated program-name branch in `_cnvrtc.pxi`.
- Smoke (`agent_space/smoke.py`: elementwise + reduction, RawKernel, CUB, random, eigvalsh/solve, FFT, sparse matmul, sort, complex exp 700.5+0.5j vs NumPy): pass on both.

### Test command (one pytest process at a time; port then base, each suite)

```bash
ulimit -c 0; export CUPY_CACHE_DIR=agent_space/kcache-<t> PYTHONPATH=agent_space/wt-<t>; cd agent_space/wt-<t>
utils/timeit.sh cupy test -- python -m pytest tests/cupy_tests  -p no:cacheprovider -n 8 --dist loadfile --timeout 1800 -rfE -q --junitxml=...
utils/timeit.sh cupy test -- python -m pytest tests/cupyx_tests -p no:cacheprovider -n 8 --dist loadfile --timeout 1800 --timeout-method=thread -rfE -q --junitxml=...
```

`-n 8`, not `-n auto`: auto is 48 workers on this host, under 0.7 GiB of VRAM each, and several tests allocate 2-8 GiB. Even at 8 workers a few large-memory tests OOM or self-skip under contention (listed below; all pass when run alone). No marker filter: slow tests included; multi-GPU tests run and fail with `hipErrorInvalidDevice` on both trees because only one GPU is visible.

### Raw results

| run | passed | failed | errors | skipped | xfailed | xpassed | wall |
|---|---|---|---|---|---|---|---|
| port cupy_tests | 26586 | 216 | 8 | 2113 | 105 | 0 | 43 min |
| base cupy_tests | 23137 | 3626 | 9 | 2113 | 105 | 0 | 96 min (interrupted, below) |
| port cupyx_tests | 71594 | 576 | 456 | 5146 | 4622 | 192 | 24 min |
| base cupyx_tests | 71914 | 256 | 456 | 5146 | 4622 | 192 | 24 min |

Two runs were polluted by the same harness fault, one on each tree, so the affected files were rerun on that tree (the rerun result supersedes):
- **base cupy_tests**: about 3390 tests failed with `hipErrorNoBinaryForGpu` / `hipErrorInvalidSource: device kernel image is invalid` (precompiled CUB/Thrust kernels and cached kernels), in random, arithmetic, statistics, sorting and testing_tests files. Afterwards every worker sat at 0% CPU with the controller waiting, and an nccl `TestCommSplit::test_4_processes_2_subcomm` spawn child spun for 77 min where `--timeout 1800` (signal method) could not fire. I killed the spinning child, then sent SIGINT to the controller, which wrote the junit XML (1 h 36 min). Rerun of the 15 affected files on base: 12694 passed, 42 failed, 8 errors, no `NoBinaryForGpu`.
- **port cupyx_tests**: 323 tests in `ndimage_tests/test_filters.py` and `test_interpolation.py` failed with the same `hipErrorNoBinaryForGpu`. Rerun of those two files on port: 13183 passed, 0 failed. The base run had one such failure (`test_morphology` tophat); rerun of `test_morphology.py` on base: 5466 passed.
- So a worker sometimes loses its kernel images after a long xdist run on this host, on either tree; the symptom is `hipErrorNoBinaryForGpu` on every later kernel in that worker. It is not tied to the port. cupyx runs used `--timeout-method=thread` so a hang inside a HIP call kills the worker instead of wedging the run.

### Classification (per test id, after the reruns)

| suite | pass both | fail both | port fixes (fail base, pass port) | port-only fail | skip both | xfail both |
|---|---|---|---|---|---|---|
| cupy_tests | 26565 | 215 | 21 | 0 (2 contention, below) | 2108 | 105 |
| cupyx_tests | 72110 | 706 | 2 | 0 | 5146 | 4622 |

- **Port-only after reruns, contention, not regressions**: `cupy_tests/cuda_tests/test_memory.py::TestMemInfo::test_mem_info` (asserts free memory > 0 while the other workers had the GPU full) and `math_tests/test_sumprod.py::TestCumprod::test_cumprod_huge_array` (OOM allocating 4 GiB). Rerun alone on port: 2 passed. Likewise 5 `TestReductionSizeOverInt32Max` tests self-skipped on port ("out of memory in test") and passed in the base rerun, and the 3 `TestSplineFilter1dLargeArray` tests OOM'd in both full runs; rerun alone on base: 3 passed.
- **No new failure on the port.** Every failure on the port also fails on the baseline at 18b54de, with the same error.

### 4ccded519 effect (`__device__` on the CArray friend declaration)

- `pytest tests/cupy_tests/core_tests/test_elementwise.py -k TestElementwiseGUFuncLike`, alone: base **21 failed, 10 passed** (111 x `'data_'/'shape_'/'strides_' is a private member of 'CArray<...>'`), port **31 passed**. Matches the gfx1100 review.
- The full run found two more tests with the same cause: `cupyx_tests/scipy_tests/special_tests/test_spfun_stats.py::TestPoissonBinomCDF::test_values` and `::test_scalar_k`. Base: 2 failed (`HIPRTC_ERROR_COMPILATION`, `'data_'/'shape_'/'strides_' is a private member of 'CArray<float, 1, true, true>'`); port: 2 passed, 1 skipped.
- So 4ccded519 fixes 23 tests on Linux ROCm. It is the only Linux behaviour change in the port.

### The complex xsf calls (identical on both trees, upstream's)

The 12 xsf failures the Windows porter recorded fail the same way on Linux, at port and at baseline, plus a 13th test with the same error: `test_dawsn::TestDawsn::test_complex`, `test_digamma::TestDigamma::{test_arange,test_complex,test_inf_and_nan,test_linspace_negative,test_linspace_positive,test_scalar}`, `test_erfi::TestErfi::test_complex`, `test_lambertw::TestLambertW::test_values`, `test_sici::TestSici::test_complex`, `test_sici::TestShichi::test_complex`, `test_wofz::TestWofz::test_complex`, and on Linux also `test_ufunc_dispatch::TestUfunc::test_dispatch[wofz]`. Each is `error: no matching function for call to '<fn>'`, the candidate note being `no known conversion from 'const thrust::complex<float>' to 'cxx::complex<float>' (aka 'std::complex<float>')`: xsf `config.h` maps `cxx::complex` to `thrust::complex` only under `__CUDACC__`, which no HIP compile defines. The error sets match test by test between port and base. This is not Windows-specific and is not in the port's scope.

### Failures common to both trees (upstream's at 18b54de on this host)

cupy_tests, 215 ids (222 junit entries; the 8 `TestCubHistogram` setup errors share one id):
- 1 x `binary_tests.test_elementwise.TestElementwise`: `AssertionError: Only cupy raises error`
- 1 x `core_tests.fusion_tests.test_misc.TestFusionMultiDevice`: `AssertionError: Both cupy and numpy raise exceptions`
- 127 x `core_tests.test_bfloat16`: `AssertionError: Only cupy raises error` (113); `NVRTCError: HIPRTC_ERROR_COMPILATION (6)` (14)
- 1 x `core_tests.test_dlpack.TestNewDLPackConversion`: `NVRTCError: HIPRTC_ERROR_COMPILATION (6)`
- 2 x `core_tests.test_elementwise.TestElementwise`: `CUDARuntimeError: hipErrorInvalidDevice: invalid device ordinal`
- 2 x `core_tests.test_ndarray.TestNdarrayCopy`: `CUDARuntimeError: hipErrorInvalidDevice: invalid device ordinal`
- 5 x `core_tests.test_ndarray.TestNdarrayDeepCopy`: `CUDARuntimeError: hipErrorInvalidDevice: invalid device ordinal`
- 2 x `core_tests.test_ndarray_elementwise_op.TestArrayElementwiseOp`: `AssertionError:`
- 1 x `core_tests.test_ndarray_get.TestArrayGet`: `CUDARuntimeError: hipErrorInvalidDevice: invalid device ordinal`
- 1 x `core_tests.test_ndarray_get.TestArrayGetWithOut`: `CUDARuntimeError: hipErrorInvalidDevice: invalid device ordinal`
- 19 x `core_tests.test_raw.TestRaw`: `CUDARuntimeError: hipErrorInvalidDevice: invalid device ordinal`
- 1 x `core_tests.test_scan.TestScan`: `CUDARuntimeError: hipErrorInvalidDevice: invalid device ordinal`
- 3 x `creation_tests.test_from_data.TestFromData`: `CUDARuntimeError: hipErrorInvalidDevice: invalid device ordinal`
- 4 x `cuda_tests.test_device.TestDeviceSwitch`: `CUDARuntimeError: hipErrorInvalidDevice: invalid device ordinal` (3); `Failed: Timeout (>N.0s) from pytest-timeout.` (1)
- 2 x `cuda_tests.test_memory.TestUnownedMemory`: `CUDARuntimeError: hipErrorInvalidDevice: invalid device ordinal`
- 1 x `cuda_tests.test_nccl.TestCommSplit`: `Failed: Timeout (>N.0s) from pytest-timeout.`
- 1 x `cuda_tests.test_nccl.TestNCCL`: `NcclError: NCCL_ERROR_INVALID_ARGUMENT: invalid argument (run with NCCL_DEBUG=WAR`
- 1 x `cuda_tests.test_runtime`: `assertion`
- 3 x `cuda_tests.test_stream.TestStream`: `CUDARuntimeError: hipErrorInvalidDevice: invalid device ordinal`
- 2 x `fft_tests.test_cache.TestPlanCache`: `Failed: Timeout (>N.0s) from pytest-timeout.` (1); `CUDARuntimeError: hipErrorInvalidDevice: invalid device ordinal` (1)
- 6 x `fft_tests.test_fft`: `AssertionError:`
- 4 x `linalg_tests.test_decomposition.TestSVD`: `AssertionError:`
- 5 x `manipulation_tests.test_basic.TestBasic`: `CUDARuntimeError: hipErrorInvalidDevice: invalid device ordinal` (4); `AssertionError: Both cupy and numpy raise exceptions` (1)
- 1 x `manipulation_tests.test_join.TestJoin`: `CUDARuntimeError: hipErrorInvalidDevice: invalid device ordinal`
- 4 x `math_tests.test_sumprod.TestBatchScanSizeOverInt32Max`: `CUDADriverError: hipErrorInvalidValue: invalid argument`
- 6 x `math_tests.test_sumprod.TestCumprod`: `AssertionError: Only cupy raises error`
- 6 x `math_tests.test_sumprod.TestCumsum`: `AssertionError: Only cupy raises error`
- 1 x `random_tests.test_generator.TestRandomState`: `Failed: DID NOT RAISE <class 'KeyError'>`
- 8 x `statistics_tests.test_histogram`: `assertion`
- 1 x `testing_tests.test_helper.TestPackageRequirements`: `AssertionError: assert False`

cupyx_tests, 706 ids:
- 114 x `distributed_tests.test_array_nccl.TestDistributedArray2Devices`: `CUDARuntimeError: hipErrorInvalidDevice: invalid device ordinal`
- 114 x `distributed_tests.test_array_nccl.TestDistributedArray4Devices`: `CUDARuntimeError: hipErrorInvalidDevice: invalid device ordinal`
- 114 x `distributed_tests.test_array_no_nccl.TestDistributedArray2Devices`: `CUDARuntimeError: hipErrorInvalidDevice: invalid device ordinal`
- 114 x `distributed_tests.test_array_no_nccl.TestDistributedArray4Devices`: `CUDARuntimeError: hipErrorInvalidDevice: invalid device ordinal`
- 16 x `distributed_tests.test_linalg.TestDistributedMatMul`: `CUDARuntimeError: hipErrorInvalidDevice: invalid device ordinal`
- 3 x `jit_tests.test_cub.TestCubWarpReduce`: `AssertionError:`
- 1 x `jit_tests.test_raw.TestRaw`: `CUDARuntimeError: hipErrorInvalidDevice: invalid device ordinal`
- 1 x `profiler_tests.test_benchmark.TestBenchmark`: `CUDARuntimeError: hipErrorInvalidDevice: invalid device ordinal`
- 6 x `scipy_tests.fft_tests.test_fft`: `AssertionError:`
- 21 x `scipy_tests.interpolate_tests.test_ndbspline.TestNdBSpline`: `AssertionError: Only cupy raises error`
- 14 x `scipy_tests.interpolate_tests.test_polyint.TestCubicSpline`: `AssertionError: Only cupy raises error`
- 105 x `scipy_tests.interpolate_tests.test_polyint.TestInterp1D`: `AssertionError: Only cupy raises error` (97); `assertion` (8)
- 1 x `scipy_tests.interpolate_tests.test_rgi.TestRegularGridInterpolator`: `assertion`
- 8 x `scipy_tests.signal_tests.test_waveforms.TestChirp`: `AssertionError: Only cupy raises error` (6); `NVRTCError: HIPRTC_ERROR_COMPILATION (6)` (2)
- 13 x `scipy_tests.sparse_tests.test_csr.TestCsrMatrixScipyComparison`: `AssertionError:` (8); `AssertionError: Only cupy raises error` (4)
- 1 x `scipy_tests.sparse_tests.test_linalg.TestLOBPCG`: `XPASS(strict)`
- 24 x `scipy_tests.sparse_tests.test_linalg.TestSpsolveTriangular`: `AssertionError: Only cupy raises error`
- 3 x `scipy_tests.sparse_tests.test_sparray.TestBoolSum`: `assertion` (1); `AssertionError:` (1)
- 1 x `scipy_tests.sparse_tests.test_sparray.TestMeanEmpty`: `AssertionError: Different types of errors occurred`
- 1 x `scipy_tests.sparse_tests.test_sparray.TestSetdiag2DRejected`: `ValueError: hipSPARSE currently cannot handle sparse matrices with null ptrs`
- 1 x `scipy_tests.sparse_tests.test_sparray.TestSparseArray1D`: `AssertionError:`
- 3 x `scipy_tests.sparse_tests.test_sparray.TestSparseArrayTypeIdentity`: `ValueError: hipSPARSE currently cannot handle sparse matrices with null ptrs`
- 2 x `scipy_tests.sparse_tests.test_sparray.TestSpsolveArray`: `assertion`
- 1 x `scipy_tests.sparse_tests.test_sparse_int64_indices.TestInt64Helpers`: `assertion`
- 1 x `scipy_tests.sparse_tests.test_sparse_int64_indices.TestInt64LinalgGuards`: `assertion`
- 3 x `scipy_tests.sparse_tests.test_sparse_int64_indices.TestInt64SumAxis`: `assertion`
- 1 x `scipy_tests.special_tests.test_dawsn.TestDawsn`: `AssertionError: Only cupy raises error`
- 6 x `scipy_tests.special_tests.test_digamma.TestDigamma`: `AssertionError: Only cupy raises error`
- 1 x `scipy_tests.special_tests.test_erfi.TestErfi`: `AssertionError: Only cupy raises error`
- 1 x `scipy_tests.special_tests.test_lambertw.TestLambertW`: `AssertionError: Only cupy raises error`
- 1 x `scipy_tests.special_tests.test_sici.TestShichi`: `AssertionError: Only cupy raises error`
- 1 x `scipy_tests.special_tests.test_sici.TestSici`: `AssertionError: Only cupy raises error`
- 1 x `scipy_tests.special_tests.test_ufunc_dispatch.TestUfunc`: `AssertionError: Only cupy raises error`
- 1 x `scipy_tests.special_tests.test_wofz.TestWofz`: `AssertionError: Only cupy raises error`
- 4 x `test_cusparse.TestCheckAvailabilityVersionSkew`: `ValueError: No available version information specified for __test_version_skew__`
- 2 x `test_cusparse.TestSpmmLargeColumns`: `RuntimeError: parallel_for failed: hipErrorInvalidConfiguration: invalid configurati`
- 1 x `test_time.TestRepeat`: `CUDARuntimeError: hipErrorInvalidDevice: invalid device ordinal`

Plus, on the base full run only, 3 `TestSplineFilter1dLargeArray` OOMs (pass alone, above). The multi-GPU `hipErrorInvalidDevice` rows come from one visible GPU. bfloat16 is `no viable overloaded '/='` (and `+=`, `*=`) in hiprtc. The timeouts are `TestDeviceSwitch::test_thread_safe`, `TestCommSplit::test_4_processes_2_subcomm` and `TestPlanCache::test_LRU_cache5`. Per-test outcomes: `agent_space/res-{port,base}-{cupy,cupyx}_tests.xml` and `*_rerun.xml` on this host (scratch, not durable); classification scripts `agent_space/parse.py`, `cmp.py`.

### Other gates

- Jargon: `python3 utils/jargon.py --port cupy` clean (local `main` at 18b54de, per the porter's note).
- Docs: install.rst documents the Windows ROCm build next to the CUDA build (checked by both reviews); the Linux HIP docs are unchanged.
- Integrity: `git -C agent_space/wt-port status --porcelain --untracked-files=no` and the same for `projects/cupy/src` are empty; HEAD 24a62ef95.
- **CUDA gate: not run on this host (no CUDA toolkit, no conda).** cuda-not-validated here. The port changes no Linux CUDA compile command (reviews 2026-11-06 and 2026-11-06b). An nvcc 12.8 compile at 24a62ef95 is still owed on a host with the toolkit (mi355), per the standing rule to compile the CUDA path before presenting.

## Validation 2026-11-06 (windows-gfx1151, validator): Ruling 3 stages (ii), (iii), (iv) at 24a62ef95

Port: ROCm/cupy `moat-port` at **24a62ef95** (fast-forward; `git status --porcelain` empty at the end). Host: Windows 11, Radeon 8060S (gfx1151, wave32 APU), TheRock ROCm 7.14.0a20260612 (hipconfig 7.14.60850), Python 3.13 venv `C:/Users/jdaily/cupy-work/venv`, MSVC 14.44 (x64 Native Tools), Cython 3.2.9, numpy 2.5.4, scipy 1.18.1, pytest 9.2.0, pytest-timeout (installed for this run), hypothesis 6.54.6. No xdist: one pytest process at a time.

### Build

```bash
source agent_space/cupy_rocm_env.sh      # venv Scripts + MSVC bin on PATH, HCC_AMDGPU_TARGET=gfx1151, CUPY_INSTALL_USE_HIP=1
utils/timeit.sh cupy compile -- python -m pip install -v --no-build-isolation -e projects/cupy/src      # EXIT 0
```

- Gotcha confirmed: a `python setup.py build_ext --inplace -v` run (EXIT 0) cleared `cupy/.data` and did not put `_depends.json` back in the source tree, so the next import bound System32's amdhip64 and failed with `CuPy failed to load hiprtc0712.dll`. The editable `pip install -e` run restores it, and `install.rst` documents `pip install`. Porter note for commit 6fd6150: its source-tree copy of `_depends.json` is not reached by a plain `setup.py build_ext --inplace` here. Not a gate.
- Smoke (`cupy-work/smoke.py`: elementwise, reduction, RawKernel, CUB, hipRAND, rocSOLVER, hipFFT, hipBLAS, hipSPARSE, rocThrust, nvtx, profiler): OK.

### Test command (per file, one pytest process at a time; driver `agent_space/cupy_val.sh` on the laptop)

```bash
ulimit -c 0
export CUPY_TEST_GPU_LIMIT=1 CUPY_GPU_MEMORY_LIMIT=25% CUPY_TEST_FULL_COMBINATION=0 PYTHONUTF8=1
utils/timeit.sh cupy test -- timeout --signal INT --kill-after 60 7200 python -m pytest -p no:cacheprovider -q -rfEX --timeout 1200 -m "not slow and not multi_gpu" --junitxml=<file>.xml <test file>
```

`CUPY_TEST_FULL_COMBINATION=0` matches upstream's ROCm CI and the Linux validators. No file approached the 2 h cap (longest: `ndimage_tests/test_filters.py` 55 min, `test_interpolation.py` 55 min, `test_arithmetic.py` 25 min, `test_ndarray_elementwise_op.py` 24 min). The machine stayed responsive: no TDR, no GPU hang, no bugcheck. Counts are junit records (skipped includes xfail as junit reports it).

### Counts per stage

| Stage | Scope | Files | passed | failed | error | skipped |
|---|---|---|---|---|---|---|
| (ii) W1-W3 | cuda_tests (texture ignored), all core_tests incl. fusion_tests and test_cub_reduction, every cupy_tests dir of W2 plus testing_tests and the top-level test_*.py, cupyx linalg_tests, jit_tests, scipy linalg/sparse/special, cupyx test_*.py | 229 | 50521 | 96 | 0 | 7990 |
| (iii) deferred cupyx | distributed, profiler, signal, tools, scipy fft/fftpack/interpolate/ndimage (minus test_interpolation.py)/signal/spatial/stats, get_array_module, example_tests | 71 | 42346 | 158 | 0 | 3070 |
| (iv) texture, last, one file each | `cuda_tests/test_texture.py` (module-level `pytest.skip('HIP texture support is not yet ready')`, 1 skipped); `ndimage_tests/test_interpolation.py` (its texture-backed class is `skipif is_hip`) | 2 | 5200 | 0 | 0 | 461 |
| total | | 302 | 98067 | 254 | 0 | 11521 |

Per-test lists: `C:/Users/jdaily/cupy-work/val-stage-ii.txt`, `val-stage-iii.txt`, junit under `cupy-work/val/` (laptop scratch).

### Classification of the 254 failures against the Linux baselines (gfx90a and gfx1201 above)

Identical failures on upstream's Linux base (not ours): **243**
- stage ii (88): binary TestElementwise 1; TestArrayElementwiseOp pow fastpath 2; cupy fft test_rfft_input_alignment 6; TestSVD 4; TestCumsum 6 and TestCumprod 6; TestRandomState 1; jit TestCubWarpReduce 3; sparse TestSpsolveTriangular 24, LOBPCG 1, sparray 11, int64-index 5; the 13 complex xsf calls (dawsn, digamma x6, erfi, lambertw, sici, shichi, wofz, ufunc_dispatch[wofz]); test_cusparse VersionSkew 4; `cuda_tests/test_runtime::test_pointer_get_memory_type` 1 (also fails on Linux).
- stage iii (155): scipy fft test_rfft_input_alignment 6, interpolate TestNdBSpline 21 + test_polyint 119 (TestInterp1D 105, TestCubicSpline 14) + test_rgi 1, waveforms TestChirp 8 (6 + 2 hiprtc compile).

Windows/APU runtime behaviour or harness (not the port): 4
- `cuda_tests/test_memory.py::TestSingleDeviceMemoryPool::test_get_limit`: artefact of `CUPY_GPU_MEMORY_LIMIT=25%` (the test expects 0). Passes with the cap unset (re-verified).
- `cuda_tests/test_memory.py::test_managed_memory_madvise_basic`: `hipErrorInvalidValue` from the managed-memory advise call on this Windows APU runtime (passes on both Linux hosts).
- `statistics_tests/test_histogram.py::TestCubHistogram::test_bincount_gh7698`: asserts `cub.device_histogram` is called, but `cupy/_statistics/histogram.py` skips CUB under `runtime.is_hip`. Linux reports the class as a setup/collection error.
- `signal_tests/test_signaltools.py::TestHilbert2::test_quadrant_values[shape3]`: 1 of 25 elements, 5.5e-16 vs -6.7e-16 against `atol=1e-15` (FFT round-off around exact zero); reproduced in isolation; scipy 1.18.1 here vs 1.17.1 on Linux.

Toolchain defect, not the port: **7 tests** (all double `nan("")`): `jit_tests/test_raw::TestRaw::test_nan`, `special_tests/test_zeta` (test_arange, test_inf_and_nan, test_linspace), `test_zetac::test_inf_and_nan`, `signal_tests/test_waveforms.py::TestSawtooth::test_sawtooth[3.0]` and `TestSquare::test_square[3.0]`. They pass on both Linux hosts. Cause, read in the SDK header `lib/llvm/lib/clang/23/include/__clang_hip_math.h`: `double nan(const char*)` has an `#else` branch for `_WIN32` that does `__val |= 0xFFF << 51;` with an int-typed `0xFFF`, so the shift is undefined and double `nan("")` returns 0.0. A hiprtc probe (`cupy-work/nan_probe.py`) gives `nanf("")` = nan, `__builtin_nan("")` = nan, `nan("")` = 0.0. The fix is `0xFFFULL << 51`. Registered as `rocm-bug-report clang-hip-math-win-nan-double-shift` on this project. CuPy emits `nan("")` in `cupyx/jit/_cuda_types.py:340` and `cupyx/scipy/special/_zeta.py`. A port-side workaround was not made (validation does not modify the fork); because the header declares `nan` itself it would have to be a macro in the `__HIPCC_RTC__ && _MSC_VER` block of `hip_workaround.cuh`, which is the porter's call.

New on the port: none. The two Linux-proven fixes work on Windows hiprtc: `core_tests/test_elementwise.py` 45 passed (TestElementwiseGUFuncLike, commit 4ccded519 plus the MSVC signbit helpers in 694db667f) and the xsf TestPoissonBinomCDF tests pass.

Notable passes (the porter's earlier stops): all of `core_tests` (fusion_tests included; `test_cub_reduction.py` 25 passed), `math_tests/test_arithmetic.py` 8783, `ndimage_tests/test_filters.py` 7980, `test_interpolation.py` 5200 (the Linux intermittent TestZoom case did not appear).

### Windows runtime observation

`core_tests/test_ndarray_copy_and_view.py` printed its summary (54 passed, 7 skipped) and then did not exit for 23 min (process-teardown hang on this APU); the driver's watcher killed it (rc=137), results intact. One occurrence in 302 files. hiprtc costs about 1 s per kernel on a cold cache; the whole run took about 11 h wall including a restart that moved fusion_tests to the end of stage (ii) (its first `test_array.py` attempt was discarded and rerun).

### Other gates

- CUDA compile gate: not run on this host; the Linux nvcc comparison at 24a62ef95 passed and Ruling 4's Windows CUDA comparison stands (f4f78903; later deltas change no CUDA branch).
- Jargon: `python3 utils/jargon.py --port cupy` clean. Docs: `docs/source/install.rst` has "Building CuPy for ROCm on Windows". Integrity: fork tree clean.

**Verdict: windows-gfx1151 completed at 24a62ef95**, with the one toolchain defect above registered.

## Porter round 2026-11-09 (windows-gfx1151): `_depends.json` missing after `setup.py build_ext --inplace`

jeff ruled `now` on the Windows validator's observation (2026-11-06) and authorized a lock takeover from the validated stage (`port-lock cupy --take windows-gfx1151`). moat-port 24a62ef95 -> **161d6e87e** `[ROCm] Restore the DLL dependency list for in-place builds on Windows` (one new commit, plain fast-forward push, no amend). File: `install/cupy_builder/_command.py` (+22/-9).

Cause: setuptools' `build_ext.run` (`setuptools/command/build_ext.py:105-111`) saves `inplace`, sets it to False, builds into `build_lib`, then calls `copy_extensions_to_source()` if it was set. So `self.inplace` is always False inside `build_extensions`, and 6fd6150's source-tree copy ran only for `editable_mode`. With `prepare_wheel_libs` clearing `cupy/.data` on every setup.py run, `build_ext --inplace` always left the source tree without `_depends.json`.

**Note for the PR-prep squash:** 6fd6150's message says "In-place and editable installs ... `build_ext` also copies it there in those modes"; at 6fd6150 that was true only for editable installs. With 161d6e87e it holds for both; a squashed message can keep the sentence.

Change: the editable copy stays at the end of `build_extensions` (now `getattr(self, 'editable_mode', False)` only); a `copy_extensions_to_source` override (called by setuptools only for `--inplace`) does the same copy when `sys.platform == 'win32'` and `ctx.use_hip`. Shared helper `_copy_depends_json_to_source`. Linux and CUDA builds unchanged: the override calls `super()` and adds nothing off Windows-HIP; no compile command changes, so Ruling 4's CUDA comparison is unaffected.

Evidence (laptop, `source agent_space/cupy_rocm_env.sh`, x64 Native Tools env, `ulimit -c 0`; probe `C:/Users/jdaily/cupy-work/probe_inplace.py`, GetModuleFileNameW):
- Before, at 24a62ef95: `utils/timeit.sh cupy compile -- bash -c "cd projects/cupy/src && python setup.py build_ext --inplace"` EXIT 0, source `cupy/.data` empty; import -> `dirs []`, `amdhip64_7` = `C:\WINDOWS\SYSTEM32\amdhip64_7.dll`, runtime 71260602, first kernel `RuntimeError: CuPy failed to load hiprtc0712.dll`. Reproduced exactly as the validator described (log `cupy-work/build-inplace-before.log`).
- After, same command: EXIT 0, `cupy/.data/_depends.json` present; `amdhip64_7` and `hiprtc0714.dll` both from `<venv>\Lib\site-packages\_rocm_sdk_devel\bin`, runtime 71460850, kernel ok (log `build-inplace-after.log`).
- `smoke.py` after the in-place build (`CUPY_GPU_MEMORY_LIMIT=25%`): OK (elementwise+reduction, RawKernel, CUB, random, solvers, FFT, matmul, sparse, sort, nvtx, profiler).
- Editable route still works: removed the source `_depends.json`, `python -m pip install -v --no-build-isolation --no-deps -e projects/cupy/src` EXIT 0, file restored, same DLL bindings, kernel ok (log `build-editable-after.log`).
- `tests/install_tests tests/import_tests`: 31 passed, 1 skipped. flake8 on `_command.py`: clean. `jargon.py --port cupy`: clean.
- Not run: the full suite (the change touches only build-time file placement; the three validated platforms derive `revalidate` at the new head), Linux/CUDA builds (no change reachable there).
