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
