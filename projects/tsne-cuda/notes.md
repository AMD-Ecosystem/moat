# tsne-cuda notes

## Intake (2026-08-07)

**What it is:** `CannyLab/tsne-cuda`, a CUDA implementation of the FIt-SNE algorithm with a Python (sklearn-compatible) binding. Active upstream: not archived, last push 2026-07-22, 1.9k stars, 18 open issues.

### Licence

`utils/licenses.py check CannyLab/tsne-cuda` reports `BSD-3-Clause`, tier 1 (GitHub parsed it cleanly, no NOASSERTION). Confirmed by reading the files directly rather than trusting the field alone: top-level `LICENSE`, `packaging/conda/LICENSE.txt`, and `src/python/LICENSE.txt` are all identical BSD-3-Clause text (Regents of the University of California, 2018). One git submodule, `third_party/cxxopts` (pinned at `c713b44`), carries its own MIT licence. No unlicensed vendored code found. `utils/licenses.py`'s NVIDIA-proprietary text-marker scan (`scan_nvidia`, matches licence TEXT not copyright lines) returned zero hits across the tree -- CUDA-specific code is present but not under an NVIDIA proprietary licence.

Recorded: `status.json.license_spdx = BSD-3-Clause` (tier 1, cleared to contribute).

### Duplicate effort

- No `tsne-cuda` (or fork of `CannyLab/tsne-cuda`) in AMD-Ecosystem or ROCm orgs (`gh search repos`/`gh repo list` against both, empty).
- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/ INSTALL.md` in the upstream tree: no hits (false-positive filtered for "chip"/"ship"/"whip"/"amdahl"). No "notable forks" section.
- RAPIDS cuML's `TSNE` implementation is explicitly derived from this project ("The CUDA implementation in RAPIDS cuML is derived from the excellent CannyLabs open source implementation") -- but that is a *downstream consumer*, not an AMD port, and there is no evidence RAPIDS cuML itself has ROCm support to point to instead. Not a duplicate-effort finding, just worth knowing tsne-cuda is the origin of a well-known derivative.

No existing or parked AMD effort found for this project.

### Viability

Genuine CUDA project: 18 `.cu` files (`src/kernels/*.cu` -- `apply_forces`, `attr_forces`, `rep_forces`, `nbodyfft`, `perplexity_search` -- plus `src/util/*.cu`, `src/fit_tsne.cu`, `src/ext/pymodule_ext.cu` for the Python extension). CMakeLists links `CUDA::cublas`, `CUDA::cufft`, `CUDA::cufftw`, `CUDA::cusparse`, uses Thrust (`src/util/thrust_utils.cu`), and hard-requires GPU FAISS (`find_package(FAISS REQUIRED)`, linked via `${FAISS_LIBRARIES}`) -- FAISS is not optional, the build fails without it. cuBLAS/cuFFT/cuSPARSE/Thrust all have direct hip*/roc* substitutes covered by the standard hipify path. The interactive ZMQ visualization path is already compiled out upstream (`-DNO_ZMQ`), so no extra third-party dependency there.

**MOAT dependency:** `faiss` is an existing MOAT project (`AMD-Ecosystem/faiss`) already `completed` on `linux-gfx90a`. This is a real, satisfiable dependency, not a blocker -- recorded with `set-deps tsne-cuda faiss`. A planner should build tsne-cuda against the validated FAISS ROCm port per its notes.md "Install as a dependency" section.

Not archived; a PR upstream has a real destination.

### Recommendation

Fork: the licence is clear (tier 1, no ambiguity, no NVIDIA-proprietary text, submodule licence checked), no duplicate AMD effort exists, the CUDA surface is real and standard (cuBLAS/cuFFT/cuSPARSE/Thrust + GPU FAISS, all with known hip*/roc* substitutes), and its one MOAT dependency (`faiss`) is already validated and completed. Recommending the planner take this up.

## Port round 2026-11-04 (porter, linux-gfx942) -> ported

Fork `AMD-Ecosystem/tsne-cuda`, branch `moat-port` cut from upstream `244db8f`, head `2e948b4`. Six commits, split per jeff's 2026-11-04 rulings:

| commit | title | kind |
|---|---|---|
| `6bab251` | [ROCm] Add a HIP build for AMD GPUs | HIP port (Linux), docs |
| `34ed72d` | [ROCm] Make the sources compile on Windows | non-HIP Windows fix (long->int64_t, ssize_t, sys/time.h, dllexport of pymodule_tsne) |
| `2adca53` | [ROCm] Name and package the library for Python on Windows | non-HIP Windows fix (lib prefix, package_data) |
| `2b32cb8` | [ROCm] Build the HIP library on Windows | Windows build (GNU-only flags/libs off WIN32), docs |
| `b11946a` | [ROCm] Build the reduction unit tests with -DBUILD_TESTS=ON | revived gtests (ruling 1) |
| `2e948b4` | [ROCm] Add GPU tests of embedding quality to the Python package | pytest harness (ruling 4) |

The Windows work separated cleanly. `2b32cb8` is the only one written for the HIP build in particular, and even it is only GNU/ELF hygiene: upstream's CUDA build never targeted Windows, so the guards are reached only through `USE_HIP`. The CMake guards for `-fPIC`/`-fopenmp`/`find_package(OpenMP)` sit in shared code under `if(NOT WIN32)`; the link-item guard sits inside the HIP branch, so the CUDA `target_link_libraries` block is byte-identical to upstream.

CUDA path: unchanged. Every CMake edit is inside `if(USE_HIP)` (CUDA config block moved into the `else()` without re-indenting so the diff stays small), or `if(WIN32)`/`if(NOT WIN32)` (Linux behaviour identical; only flag order in `CMAKE_CXX_FLAGS` moved, and there are no CXX sources). The `.cu` edits are type-identical on Linux (`int64_t` is `long` on LP64). No CUDA host was available to rebuild; the reviewer should confirm by inspection, as the plan foresaw.

### Deviations from plan.md

- **No `cmake/hip_link_win.py`.** The plan vendored the link wrapper; it was written for executables with `-fgpu-rdc` device links. tsnecuda is a non-RDC shared library, and FAISS's Windows validation (2026-08-20, gfx1151) linked its HIP shared library through CMake's own rule with the GNU-driver clang++ for all languages and no wrapper. If the windows validator hits the MSVC-flag mangling the skill describes, the fix is the wrapper set as `CMAKE_HIP_CREATE_SHARED_LIBRARY` on WIN32 -- send it back to the porter.
- **Explicit `__declspec(dllexport)` on `pymodule_tsne`** instead of `WINDOWS_EXPORT_ALL_SYMBOLS`: Thrust-heavy objects can exceed the 65535-export limit, and Python needs one symbol. Consequence: the gtest target compiles `cuda_utils.cu` and `reduce_utils.cu` in rather than linking the DLL.
- **hipFFT licence result.** The plan said hipFFT defines `HIPFFT_LICENSE_ERROR`; current hipFFT (1.0.26, ROCm 10.1) does not (the enum skips 15). The shim maps `CUFFT_LICENSE_ERROR` to `static_cast<hipfftResult>(15)` (in range of the enum, distinct from every other case label), so `cuda_utils.cu` is untouched.
- **`-Wl,--allow-multiple-definition` is not needed.** Relinking the HIP `libtsnecuda.so` without it succeeds (no duplicate symbol); kept on Linux for parity with the CUDA link line, omitted on Windows.
- **CPU-FAISS test size.** The plan's test 6 (3000 blobs, k=1024) crashes on an upstream bug (deferred `tsne-cuda-cpu-knn-missing-neighbors`): IVF with sqrt(N) cells / nprobe 20 returns -1 padding for 315 of 3000 clustered points (measured with CPU FAISS alone, `agent_space/ivfcheck/check.cpp`), then `syv2k` faults. The test uses 10000 points (0 missing at 5000, 10000, 20000).

### Host and dependency recipe (linux-gfx942)

ROCm here is TheRock pip SDK 10.1 (`rocm-sdk path --root` -> `.../site-packages/_rocm_sdk_devel`), not `/opt/rocm`. FAISS `moat-port` `01a09e8` built per faiss notes "Install as a dependency" with `-DCMAKE_HIP_ARCHITECTURES=gfx942`, the SDK's `llvm/bin/clang{,++}` for C/CXX/HIP, `-DCMAKE_PREFIX_PATH=$SDK`, and two local workarounds (deferred `faiss-rocm10-fp16-include-and-install-header`):
```
-DCMAKE_HIP_FLAGS="-include hip/hip_fp16.h -include hip/hip_bf16.h"
cp faiss/impl/InvertedListScannerStats.h install/include/faiss/impl/
```
tsne-cuda:
```
SDK=$(rocm-sdk path --root); FAISS_PREFIX=<faiss>/install
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_C_COMPILER=$SDK/llvm/bin/clang -DCMAKE_CXX_COMPILER=$SDK/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=$SDK/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="$SDK;$FAISS_PREFIX" -DBUILD_PYTHON=ON -DBUILD_TESTS=ON
cmake --build build -j 56
```
Also verified: the documented INSTALL.md form (system gcc 13 for C/CXX, HIP compiler found from `PATH`) configures and builds. Warnings only (`[[nodiscard]] hipError_t` ignored at `options.h:226,229`, `cuda_utils.cu:243`, `attr_forces.cu:200`, upstream code).

### Results at 2e948b4 (MI300X gfx942, one GPU via HIP_VISIBLE_DEVICES=0, ulimit -c 0)

Runtime: `LD_LIBRARY_PATH=$FAISS_PREFIX/lib PYTHONPATH=build/python`, Python 3.11, numpy 1.26.4, scikit-learn 1.9.1, pytest 7.3.2.

- `python -c "import tsnecuda; tsnecuda.test()"`: rc 0, 1000 iterations, total_time 2.3-5.2 s.
- `ctest --test-dir build --output-on-failure`: 1/1 (8 gtests `Reductions.*`); `./build/tsnecuda_test` 20 consecutive runs (clock-seeded inputs) all `PASSED 8 tests`.
- `python -m pytest -v build/python/tsnecuda/test`: 7 passed in 16-18 s, 4 consecutive clean runs plus the final clean-build run. Measured: digits trustworthiness 0.9924 (sklearn TSNE 0.9933), 5-fold 10-NN 0.9616; blobs 50000 ARI 1.0 (fit 1.1 s); blobs 10000 k=1024 (CPU FAISS) ARI 1.0. Sync-kernels and launch-override variants pass the same thresholds.
- `cmake/benchmarks/tune_kernels.py --n 20000 --iters 300 --repeats 1`: completes; recommends `{"TSNE_FFT_THREADS": 512, "TSNE_ATTR_THREADS": 128, "TSNE_REP_THREADS": 128, "TSNE_INTEG_THREADS": 512, "TSNE_INTEG_FACTOR": 4}` (record only).

### Gotcha: scikit-learn KMeans crashes on 224-thread hosts

Before `threadpool_limits(8)` the pytest run exited 134/139 about half the time, after all tests passed. Backtrace: SIGSEGV in `sgemm_incopy_PRESCOTT` (scipy's bundled OpenBLAS) called from `lloyd_iter_chunked_dense._omp_fn` on libgomp worker thread ~#507. KMeans alone, without tsnecuda loaded, crashes 4/6 runs on this host and 0/6 with `OMP_NUM_THREADS=16`. Environment, not the port; the test caps KMeans threads so it is robust on many-core validators.

### Other findings (deferred, for a person)

- `tsne-cuda-reducemean-axis1-divisor`: `ReduceMean` divides by N for both axes; square 512x512 tests hide it. Upstream bug.
- Per the standing rule, the fork's GitHub default branch should be `moat-port`; this fleet worker may not change repository settings, so a person needs to set it.

### For the validators

- windows: the three Windows commits were written on Linux and never compiled for Windows. Expect to stage `faiss.dll` and the ROCm DLLs next to `libtsnecuda.dll` (Python >= 3.8 ignores `PATH`), and to need the FAISS workarounds above if the Windows SDK is also ROCm >= 10. Size the large blobs test with `TSNECUDA_TEST_BLOBS_N=20000` on gfx1151.
- wave32: nothing wave-size-dependent in the code (no warp intrinsics, no `__shared__`); the launch-override test covers non-default block sizes.

## Review 2026-11-04 (reviewer, linux-gfx1201) -> changes-requested

Reviewed `moat-port` `6bab251..2e948b4` against upstream `244db8f` (fork `main` == upstream `main` == `244db8f`; fork GitHub default branch is already `moat-port`, so the "a person needs to set it" item above is done). `utils/jargon.py --port tsne-cuda`: clean. Titles <= 72 chars, `[ROCm]`, no trailer, ASCII only.

### Findings (must-fix)

1. **pytest subprocess tests depend on how the package reached `sys.path`.** `src/python/tsnecuda/test/test_tsne.py:80` builds the child environment from `os.environ` only, and the child runs `from tsnecuda.test import test_tsne`. pytest puts `build/python` on the parent's `sys.path` itself (rootdir insertion), not in `PYTHONPATH`. Running the documented command (INSTALL.md, `python -m pytest -v build/python/tsnecuda/test`) without an installed package or a `PYTHONPATH` therefore passes 5 tests and fails `test_digits_quality_synchronized_kernels` and `test_digits_quality_launch_overrides` with `ModuleNotFoundError: No module named 'tsnecuda'` (reproduced on gfx1201, `unset PYTHONPATH`, cwd `agent_space/tsne`). Fix: prepend the package's parent directory to the child's `PYTHONPATH`, e.g. `os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))` joined with any existing value by `os.pathsep`.
2. **With clang-cl as the C++ compiler, `-DBUILD_TESTS=ON` cannot build googletest.** `CMakeLists.txt:40-57` still appends GNU-driver flags (`-std=c++17 -O3 -Wall -Wextra -m64 -mpopcnt -msse4`) to `CMAKE_CXX_FLAGS` on Windows; only `-fopenmp -fPIC` moved under `if(NOT WIN32)` (`:49`). The library is unaffected (all its sources are HIP, which does not use `CMAKE_CXX_FLAGS`), but FetchContent googletest is C++, inherits `CMAKE_CXX_FLAGS`, and under `MSVC` (true for clang-cl) compiles with `-GS -W4 -WX` (`googletest/cmake/internal_utils.cmake:76`, v1.15.2). Checked with the SDK's `clang-cl --target=x86_64-pc-windows-msvc -### <those flags> -W4 -WX`: `error: unknown argument ignored in clang-cl: '-std=c++17' [-Werror,-Wunknown-argument]`, same for `-msse4`; `-Wall` also becomes `-Weverything`. The skill (validation.md, "keep one clang driver") says all-clang-cl is a valid choice with CMake 4.x, and FAISS's own gfx1151 Windows record was all-clang-cl, so a user building against an all-clang-cl FAISS can reasonably pick it. Fix: apply the `OPT_FLAGS` `CMAKE_CXX_FLAGS` append only `if(NOT MSVC)` (Linux and the documented GNU-driver Windows path unchanged; `MSVC` is false for GNU-driver clang++), keeping `-fopenmp -fPIC` under `NOT WIN32`. Belongs in `2b32cb8` (the Windows build commit), as a new commit on top. Not compiled on Windows; the mechanism is verified from the driver and googletest's CMake, not from a Windows build.

### Checks requested by jeff (results)

- **CUDA path unchanged (inspection; no nvcc here).** With `USE_HIP=OFF` on Linux: `enable_language(CUDA)` moved below the options (no dependency on order); the CUDA config block, arch ladder and `CUDA_OPTS` are byte-identical inside `else()`; the CUDA `target_link_libraries` list is byte-identical; `find_package(OpenMP REQUIRED)` still runs off Windows; `OPT_FLAGS` lose nothing on Linux (`-fopenmp -fPIC` reordered after `-DFINTEGER=int`, `CMAKE_CXX_FLAGS` only, no CXX sources in the target). `BUILD_TESTS` defaults OFF; `test_tsne.py` is added to `PYTHON_SOURCES` on both backends (device-generic test, correct place). Source edits are type-identical on LP64 (`int64_t` is `long`; `TSNECUDA_EXPORT` empty; `sys/time.h` still included). Not verified: the CUDA `tsnecuda_test` target (opt-in, never compiled with nvcc).
- **Each commit stands alone.** Clean configure + build + `tsnecuda.test()` at each of `6bab251`, `34ed72d`, `2adca53`, `2b32cb8`, `b11946a` (with `-DBUILD_TESTS=ON`) and `2e948b4`: all rc 0 on gfx1201. Splitting the non-HIP Windows fixes into their own PR: `34ed72d` cherry-picks cleanly onto `244db8f`; `2adca53` conflicts only in its 4-line CMake hunk (context is the HIP link block) and needs a trivial hand resolution. `2e948b4` cherry-picks cleanly onto upstream; `b11946a` does not (its CMake has HIP branches) and is not meant to.
- **Windows buildability by inspection.** Documented GNU-driver path: `-fPIC` (a hard error for `x86_64-pc-windows-msvc`, verified) is guarded; no other POSIX-only construct in the 16 library sources or the test sources (grep for `long`, `ssize_t`, `sys/`, `unistd`, `M_PI`, `drand48`, `pthread`, `getopt`); `pymodule_ext.cu` includes the header carrying the `dllexport`; numpy `load_library('libtsnecuda')` resolves `.dll` on Windows; no duplicate strong symbols across the 16 objects (`nm --defined-only`), so dropping `--allow-multiple-definition` hides nothing. clang-cl path: finding 2. Still open for the windows validator: the HIP-language executable link of `tsnecuda_test.exe` (the case `hip_link_win.py` was written for; the FAISS Windows evidence covers a HIP shared library and CXX test executables), and whether this SDK needs FAISS's `-DCMAKE_HIP_COMPILER_ROCM_ROOT` / `--rocm-device-lib-path` hints.
- **Skill lessons on this branch** (fault-classes.md hipFFT licence status, validation.md KMeans): checked; on this host (ROCm SDK 10.1.0, hipFFT 1.0.26) `hipfftResult` skips 15 and `HIPBLAS_STATUS_UNKNOWN` (11) has no case in `_cublasGetErrorEnum`. Accurate.

### GPU evidence at 2e948b4 (gfx1201, HIP_VISIBLE_DEVICES=0, ulimit -c 0)

FAISS `moat-port` `01a09e8` built for gfx1201 with the SDK clang and the two workarounds above. tsne-cuda: the porter's command with `-DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBUILD_TESTS=ON`. Warnings only (FAISS `Panorama.h`/`platform_macros.h`, `[[nodiscard]] hipError_t`, `-Wswitch` on the error-name switches).

- `ctest --test-dir build --output-on-failure`: 1/1; `./tsnecuda_test` 10 consecutive runs, all `PASSED 8 tests`.
- `python -c "import tsnecuda; tsnecuda.test()"`: rc 0, total_time 3.07 s.
- `PYTHONPATH=build/python python -m pytest -v build/python/tsnecuda/test` (scikit-learn 1.9.1, numpy 1.26.4, pytest 7.3.2): 7 passed in 23.6-23.8 s, 3 runs. Digits trustworthiness 0.9929 (scikit-learn 0.9933), 10-NN 0.9666; blobs 50000 ARI 1.0 (2.8 s incl. KMeans); blobs 10000 k=1024 ARI 1.0.
- With the image's Python (no scikit-learn) the module skips as designed (`1 skipped`).

## Fix round 2026-11-04 (porter, linux-gfx942) -> ported

Answers the two must-fix findings of the 2e948b4 review, as new commits on top (no amend), one per topic:

| commit | title | answers |
|---|---|---|
| `ab87296` | [ROCm] Keep GNU compiler flags out of clang-cl C++ builds on Windows | finding 2 (Windows HIP build topic, follows `2b32cb8`) |
| `4fedca7` | [ROCm] Let the subprocess tests find the package without PYTHONPATH | finding 1 (pytest topic, follows `2e948b4`) |

- `CMakeLists.txt`: the `string(REPLACE ...)`/`set(CMAKE_CXX_FLAGS ...)` append of `OPT_FLAGS` is now inside `if(NOT MSVC)`; `-fopenmp -fPIC` stay under `NOT WIN32`. On Linux with the SDK clang++ the googletest C++ objects still compile with `-std=c++17 -Wno-sign-compare -O3 -Wall -Wextra -DFINTEGER=int -fopenmp -fPIC -m64 -mpopcnt -msse4` (8 `FLAGS =` lines in `build/build.ninja`). The clang-cl path itself is not compiled here (no Windows host); the windows validator should build with `-DBUILD_TESTS=ON`.
- `test_tsne.py::_digits_in_subprocess`: prepends `dirname(dirname(dirname(abspath(__file__))))` (the directory holding `tsnecuda/`) to the child's `PYTHONPATH`, joined with any existing value by `os.pathsep`.

Rebuilt clean at `4fedca7` with the recipe above (`-DBUILD_TESTS=ON`, gfx942, FAISS `01a09e8` install), warnings only. MI300X, `HIP_VISIBLE_DEVICES=0`, `ulimit -c 0`, `LD_LIBRARY_PATH=$FAISS_PREFIX/lib`, Python 3.11 venv with scikit-learn 1.9.1, numpy 1.26.4, pytest 7.3.2:

- `ctest --test-dir build --output-on-failure`: 1/1; `./build/tsnecuda_test` 10 consecutive runs, all `PASSED 8 tests`.
- `cd build/python && PYTHONPATH= python -c "import tsnecuda; tsnecuda.test()"`: rc 0, total_time 2.35 s.
- INSTALL.md command from the source root, `PYTHONPATH` unset, package not installed (`python -c "import tsnecuda"` -> `ModuleNotFoundError`): `python -m pytest -v build/python/tsnecuda/test` 7 passed in 18.8 / 16.1 / 15.5 s (3 runs).
- Same setup with the 2e948b4 `test_tsne.py` copied into `build/python`: `-k "synchronized or overrides"` 2 failed, `ModuleNotFoundError: No module named 'tsnecuda'` in the child (the reviewer's reproduction, confirmed on gfx942). Build copy restored by `cmake --build build` and checked identical to the source.
- `utils/jargon.py --port tsne-cuda`: clean; `utils/prose.py` on both bodies: clean.

## Review 2026-11-04 (reviewer, linux-gfx1201, re-review) -> review-passed

Re-reviewed the fix delta `2e948b4..4fedca7` (`ab87296`, `4fedca7`) with the pr-review skill, fact-checked inline. No findings; both must-fix items of the previous review are resolved.

- Finding 2 (`ab87296`): `CMakeLists.txt:58` guards the `OPT_FLAGS` append with `if(NOT MSVC)`; `-fopenmp -fPIC` stay under `NOT WIN32`. The only C++ sources in the build are googletest (the library and `tsnecuda_test` sources are all HIP), so under clang-cl googletest now gets only its own MSVC flags. Checked with the SDK's `clang-cl --no-default-config --target=x86_64-pc-windows-msvc -### -c -GS -W4 -WX`: no error, default `-std=c++14`, which googletest v1.15.2 accepts. (Without `--no-default-config` the Linux SDK's `clang-cl.cfg` injects `-frtlib-add-rpath`, which `-WX` rejects; that is a Linux-SDK host artifact, not the port, and the Windows SDK has its own config.) Linux unchanged: 8 googletest `FLAGS = -std=c++17 ... -msse4` lines in `build/build.ninja`. Still not compiled on Windows; the windows validator should build with `-DBUILD_TESTS=ON`.
- Finding 1 (`4fedca7`): `test_tsne.py:_digits_in_subprocess` prepends the directory holding `tsnecuda/` (three `dirname`s of `__file__`) to the child's `PYTHONPATH`, keeping any existing value.
- Commit hygiene: titles 68 and 67 chars, `[ROCm]`, no trailer, ASCII; `utils/jargon.py --port tsne-cuda` clean (needed a local `main` branch from `origin/main` in the fresh clone first).

GPU evidence at `4fedca7` (gfx1201, `HIP_VISIBLE_DEVICES=0`, `ulimit -c 0`): FAISS `moat-port` `01a09e8` install from the earlier gfx1201 review (built per the recipe above), fresh tsne-cuda build with the porter's recipe, `-DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBUILD_TESTS=ON`, rc 0. Runtime `LD_LIBRARY_PATH=$FAISS_PREFIX/lib`, `PYTHONPATH` unset, package not installed (`import tsnecuda` from the source root -> `ModuleNotFoundError`), Python 3.11 venv with scikit-learn 1.9.1, numpy 1.26.4, pytest 7.3.2.

- `ctest --test-dir build --output-on-failure`: 1/1; `./build/tsnecuda_test` 10 consecutive runs, all `PASSED 8 tests`.
- `cd build/python && python -c "import tsnecuda; tsnecuda.test()"`: rc 0, total_time 2.79 s.
- `python -m pytest -v build/python/tsnecuda/test` (INSTALL.md form, from the source root): 7 passed in 23.74 / 23.68 / 23.55 s, 3 runs, including `test_digits_quality_synchronized_kernels` and `test_digits_quality_launch_overrides`.

## CUDA compile check 2026-11-04

Requested CUDA no-regression compile check (no role taken, no state changed) on the mi355 fleet host, fork `moat-port` `4fedca7`. **Result: not completed -- blocked before any `.cu` compiled, on the missing CUDA GPU FAISS dependency. This says nothing about a CUDA regression either way.**

Toolchain: `/usr/local/cuda-12.8/bin/nvcc` (not on `PATH` by default), `Cuda compilation tools, release 12.8, V12.8.93`, host GCC 13.3.0, CMake 3.31.6. The toolkit has cuBLAS, cuFFT, cuSPARSE and Thrust.

```
git clone https://github.com/AMD-Ecosystem/tsne-cuda.git tsne-cuda-cudacheck
cd tsne-cuda-cudacheck && git checkout 4fedca7
export PATH=/usr/local/cuda-12.8/bin:$PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=90 -DBUILD_TESTS=ON
```

Configure, rc 1: CUDA 12.8.93 detected, `tsnecuda: honoring CMAKE_CUDA_ARCHITECTURES=90`, OpenMP found, then

```
CMake Error at .../FindPackageHandleStandardArgs.cmake:233 (message):
  Could NOT find FAISS (missing: FAISS_INCLUDE_DIR FAISS_GPU_INCLUDE_DIR
  FAISS_LIBRARY)
  cmake/Modules/FindFAISS.cmake:21 (find_package_handle_standard_args)
  CMakeLists.txt:197 (find_package)
```

Cause: `find_package(FAISS REQUIRED)` needs a CUDA GPU FAISS (headers `faiss/gpu/*.h` plus `libfaiss`); `src/include/util/distance_utils.h` includes `faiss/gpu/GpuIndexFlat.h`, `StandardGpuResources.h` and `GpuCloner.h`, so a CPU-only FAISS would not do. None is installed on this host and there is no conda (upstream INSTALL.md recommends `conda install -c conda-forge libfaiss`). Building upstream FAISS v1.15.1 from source with nvcc 12.8 (`-DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_CUDA_ARCHITECTURES=90`, `cmake --build build --target faiss`, `cmake --install`) was refused by this session's permission policy as building third-party code the task did not name, so it was not run. No fork change is indicated; nothing was pushed to the fork.

To finish: provide a CUDA GPU FAISS (conda-forge `libfaiss` built for CUDA 12, or the source build above), then rerun the configure with `-DCMAKE_PREFIX_PATH=<faiss prefix>` (or `-DFAISS_ROOT_DIR`, `-DFAISS_INCLUDE_DIR`/`-DFAISS_LIBRARY`) and `cmake --build build -j 64`, which compiles the 16 library sources and `tsnecuda_test` (`src/test/test.cu`, `cuda_utils.cu`, `reduce_utils.cu`, GoogleTest v1.15.2 via FetchContent).

## Validation 2026-11-04 (validator, linux-gfx1201) -> validation-failed at 4fedca7

Radeon gfx1201 (`rocm_agent_enumerator`: gfx1201), `HIP_VISIBLE_DEVICES=0`, `ulimit -c 0`, ROCm SDK 10.1.0 (TheRock pip, no `/opt/rocm`). Fresh clone of `AMD-Ecosystem/tsne-cuda` at `4fedca7`, tree clean. FAISS `moat-port` `01a09e8` install for gfx1201 (the earlier gfx1201 review's build, recipe above; 168 gfx1201 code objects in `libfaiss.so`).

Build, INSTALL.md Linux form with the SDK root in place of `/opt/rocm` and the documented `-DCMAKE_HIP_COMPILER` hint (system GCC 13.3 for C/CXX, SDK clang 24 for HIP), rc 0, warnings only (`-Wunused-value` 52, `-Wswitch` 12, `-Wignored-pragmas` 12, `-Wmacro-redefined` 6):
```
SDK=$(rocm-sdk path --root); FAISS_PREFIX=<faiss install>
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_HIP_COMPILER=$SDK/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="$SDK;$FAISS_PREFIX" -DBUILD_TESTS=ON
cmake --build build -j 24
```
Runtime: Python 3.11 venv, scikit-learn 1.9.1, numpy 1.26.4, pytest 7.3.2; `PYTHONPATH` unset, package not installed (`import tsnecuda` from the source root -> `ModuleNotFoundError`, `pip show tsnecuda` -> not found).

### Why it failed: `tsnecuda_test` is flaky (test reference, not the GPU)

- First `./build/tsnecuda_test` x10: 8 `PASSED 8 tests`, 2 failed (`Reductions.ReduceSum_Row_512x512`). Then x30: 30 passed; x100: 98 passed, 2 failed (`Reductions.ReduceSum_Col_512x512`: `cpu -75.362464905` vs `gpu -75.362594604`, difference 1.297e-4 > 1e-4). Final x10: 10 passed. Failures come in adjacent pairs because the inputs are seeded with `time(NULL)` (one seed per second).
- Seed pinned with an `LD_PRELOAD` `time()` shim (`agent_space/seedshim`): seeds 1..300 -> only seed 217 fails (`ReduceSum_Row_512x512`, difference 1.068e-4), and it fails 10/10 times, seed 216 passes 5/5. Deterministic in the data, not a race.
- Seed 217, row 228, against a double-precision sum of the same inputs: CPU float sequential reference -59.540256500 (off by 1.06e-4), GPU (hipBLAS) -59.540149689 (off by 5.1e-7), double -59.540150203. The GPU is right; the test's float sequential CPU reference drifts past the 1e-4 absolute tolerance on some row/column sums of 512 N(0,1) values.
- The 1e-4 `EXPECT_NEAR` and the float CPU accumulation are upstream's (`src/include/test/test_reduce.h`); upstream never built these tests, and `b11946a` builds them, so the port now ships a test that fails at random on wall-clock seeds. Fix for the porter, as a new commit in the gtest topic: accumulate the CPU reference in `double` in the sum/mean tests (or compare with a tolerance relative to the magnitude). Every arch validates that commit, gfx942 included (its validation at this head is in flight).

### Everything else passed at 4fedca7

- `ctest --test-dir build --output-on-failure`: 1/1 (twice).
- `cd build/python && python -c "import tsnecuda; tsnecuda.test()"`: rc 0, total_time 2.90 s.
- `python -m pytest -v build/python/tsnecuda/test` from the source root: 7 passed in 23.41 / 23.48 / 23.54 s (3 runs).
- Metric harness (the `test_tsne.py` helpers called directly, `agent_space/metrics.py`), against the file's thresholds: digits trustworthiness 0.99297 (min 0.97; scikit-learn TSNE 0.99330, allowed gap 0.02), 5-fold 10-NN 0.96941 (min 0.95); `TSNE_SYNC_KERNELS=1` 0.99219 / 0.96831; launch overrides (FFT 64, ATTR/REP/INTEG 256) 0.99255 / 0.96607; blobs 50000 ARI 1.0 (min 0.95); blobs 10000 k=1024 CPU FAISS ARI 1.0 (min 0.9).
- `utils/jargon.py --port tsne-cuda`: clean (needs local `main` and `moat-port` branches in a fresh clone).

### Runtime finding: `libomp.so` with the documented GCC recipe

With the INSTALL.md form, `libtsnecuda.so` links GCC's `libgomp.so.1`, but a FAISS built with the ROCm clang needs `libomp.so` (from `$SDK/llvm/lib`) and its RUNPATH holds only `$SDK/lib`. Every Python test then fails at import: `OSError: libomp.so: cannot open shared object file` (7 pytest failures, `tsnecuda.test()` rc 1). The earlier runs built tsne-cuda with the SDK clang for C/CXX, whose libomp RUNPATH happened to satisfy FAISS. Worked around by `LD_LIBRARY_PATH=$FAISS_PREFIX/lib:$SDK/llvm/lib`; all results above use that. It is a property of the FAISS install, but the INSTALL.md recipe is what hits it: the porter may want one sentence there (make FAISS's OpenMP runtime loadable, or use the ROCm clang for C/CXX as well). That process also loads both libgomp and libomp; no problem seen in these runs.

### CUDA gate

`cuda-not-validated`: no CUDA toolkit or conda on this host, and the CUDA build also needs a CUDA-built GPU FAISS (`find_package(FAISS REQUIRED)`), which the conda toolkit does not provide. Still owed on the mi355 host before the port goes to jeff.
## Validation 2026-11-04 (validator, linux-gfx90a) -> completed

Head `4fedca7` (clean tree, `git status --porcelain` empty). AMD Instinct MI250X/MI250 (gfx90a, sramecc+ xnack-), one GCD via `HIP_VISIBLE_DEVICES=0`, `ulimit -c 0`. ROCm: TheRock pip SDK 10.1.0 (`rocm-sdk path --root`), no `/opt/rocm` on this host.

FAISS `moat-port` `01a09e8` built for gfx90a per faiss notes "Install as a dependency", with the SDK `llvm/bin/clang{,++}` for C/CXX/HIP, `-DCMAKE_PREFIX_PATH=$SDK` and the two workarounds above (`-include hip/hip_fp16.h -include hip/hip_bf16.h`, copy of `InvertedListScannerStats.h`). tsne-cuda: the INSTALL.md Linux recipe plus the HIP-compiler hint it documents (system gcc 13.3 for C/CXX):
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=$SDK/llvm/bin/clang++ -DCMAKE_PREFIX_PATH="$SDK;$FAISS_PREFIX" -DBUILD_TESTS=ON
cmake --build build -j 32
```
rc 0. 94 warnings: 52 `-Wunused-value` (`[[nodiscard]] hipError_t`), 12 `-Wswitch` (error-name switches), and 18 from FAISS headers (`Panorama.h` `#pragma float_control`, `Index.h` `VERSION_STRING` redefined).

Runtime gotcha (environment, not the port): a FAISS built with the SDK clang links LLVM `libomp.so`, which its runpath does not cover, so `import`/`fit` fails with `OSError: libomp.so: cannot open shared object file`. Fix: `LD_LIBRARY_PATH=$FAISS_PREFIX/lib:$SDK/lib/llvm/lib`. (`libtsnecuda.so` itself links `libgomp`; both runtimes then load into one process, and no test was affected.)

Results (Python 3.11.17 venv, numpy 1.26.4, scikit-learn 1.9.1, pytest 7.3.2; `PYTHONPATH` unset, package not installed: `import tsnecuda` -> `ModuleNotFoundError`, `pip show tsnecuda` -> not found):

- `ctest --test-dir build --output-on-failure`: 1/1 passed (0.59 s).
- `./build/tsnecuda_test` 10 consecutive runs: 10/10 rc 0, `PASSED 8 tests` each.
- `cd build/python && python -c "import tsnecuda; tsnecuda.test()"`: rc 0, total_time 10.56 s (run 1), 8.39 s (run 2). Slower than gfx942/gfx1201 (2.3-3.1 s), all of it one-time setup rather than iterations: `_time_knn` 3.55 s, `_time_norm` 2.8-3.8 s, `_time_init_fft` 3.79 s first run / 0.56 s second; `_time_nbodyfft` 0.05 s, `_time_attr` 0.02 s. Recorded only, not a gate.
- `python -m pytest -v build/python/tsnecuda/test`: 7 passed in 77.09 / 79.03 / 68.90 s (3 runs, absolute path, cwd MOAT root) and 61.31 s (INSTALL.md form, cwd source root). Includes `test_digits_quality_synchronized_kernels` and `test_digits_quality_launch_overrides`.
- Metric harness (`test_tsne.py` helpers called directly to print values) against its thresholds:

| metric | measured | threshold |
|---|---|---|
| digits trustworthiness | 0.99285 | >= 0.97 |
| digits 5-fold 10-NN accuracy | 0.96218 | >= 0.95 |
| scikit-learn TSNE trustworthiness (reference) | 0.99330 | ours >= ref - 0.02 |
| digits, `TSNE_SYNC_KERNELS=1` | trust 0.99135, kNN 0.96051 | 0.97 / 0.95 |
| digits, launch overrides (64/256/256/256) | trust 0.99254, kNN 0.96664 | 0.97 / 0.95 |
| blobs 50000, k=32 ARI | 1.0 (5.18 s) | >= 0.95 |
| blobs 10000, k=1024 (CPU FAISS) ARI | 1.0 (4.46 s) | >= 0.9 |

- `utils/jargon.py --port tsne-cuda`: clean. Documentation: INSTALL.md "Building for AMD GPUs (ROCm)" (Linux and Windows), linked from README.
- CUDA gate: `cuda-not-validated: no CUDA toolkit on this host (no conda, no nvcc)`; also needs a CUDA-built FAISS for `find_package(FAISS REQUIRED)`. Still owed on a CUDA-capable host at `4fedca7`.
