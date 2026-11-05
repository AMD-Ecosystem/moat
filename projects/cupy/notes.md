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
