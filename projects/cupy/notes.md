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
