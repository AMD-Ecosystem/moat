# CTranslate2 notes

## FINAL DECISION 2026-06-11: outcome=validated, NO upstream PR (jeff)

Resolved. CTranslate2 is delivered as a VALIDATION record, not an upstream contribution. Upstream already ships a mature HIP backend; our only delta is a 2-line arch-list add (prepend gfx90a) plus one doc line, which jeff judged too trivial to warrant a PR. `upstream.json` carries `outcome: validated` (README shows "validated (5 arch)"); all 5 platforms stay `completed` at 7bedc886 as the validation record. Do NOT open a PR and do NOT re-surface this in "next PR to prep" scans -- any such scan must skip projects whose upstream.json has a terminal `outcome` (validated/fork/superseded/blocked) or a pr_url. The 2026-06-04 "small-delta PR is warranted" note below and the PR-prep "ready for PR-open" note are SUPERSEDED by this decision; the PR-prep edits (doc + squash to 7bedc886) are kept as harmless fork content, no PR.

## PR-prep 2026-06-11 (porter, lead linux-gfx90a) -- SUPERSEDED (no PR; see FINAL DECISION above)

All 5 platforms were completed at dfa0d30 (gfx90a/gfx1100/gfx1101/gfx1201/gfx1151). Ran the PR-prep sequence per CLAUDE.md; no upstream PR opened (gated on user approval).

- JARGON SCRUB: the existing dfa0d30 commit message and the 2-line diff were already clean -- no MOAT in-house vocabulary (no lead/follower, Strategy A/B, head_sha, validated_sha, revalidate, curated/moat-port as jargon). The diff adds no code comments. Nothing to rewrite.
- DOCUMENTATION (the prep edit): `docs/hardware_support.md` "## GPU" section listed only NVIDIA GPUs despite upstream shipping ROCm wheels. Added an AMD GPU bullet naming the supported ROCm arches (gfx90a/CDNA2 MI250X + the RDNA set) in the page's house style (parallel to the NVIDIA bullet; MyST markdown). `docs/installation.md` already documents `-DWITH_HIP=ON` + the ROCm-libraries requirement (build-step location, complete -- arch is set via CMAKE_HIP_ARCHITECTURES / the prepare scripts, not a user-doc arch list, so nothing imposed there). README.md line 61 already notes AMD ROCm wheels. Doc edit is behavior-preserving.
- CMAKE ARCH AUTO-DETECT: N/A. CTranslate2 sets the ROCm arch via -DCMAKE_HIP_ARCHITECTURES (read from cache) + the docker ARG HIP_ARCHITECTURES default + the prepare_build_environment_*_rocm.sh PYTORCH_ROCM_ARCH export. There is no CMake auto-detect hook for the HIP path (the CUDA path's CUDA_ARCH_LIST=Auto is NVIDIA-only/upstream). Inventing one is out of scope and would change build behavior; not added.
- COPYRIGHT/AUTHORSHIP: N/A. The port created no new source file and did not substantially extend one; the whole delta is a 2-line build-config arch-list add plus a doc edit (trivial config/build-flag per CLAUDE.md). No AMD copyright line or \author tag applies.

Prep commit (doc-only): e83988fb18bac4eb5cccb6e941601995bc0dcbb4 -- classified arch-independent by advance-head; all 5 platforms carried forward to e83988fb without GPU re-run.

SQUASH: collapsed moat-port to one commit 7bedc88617d18ce8222309faae7eba883f161003 (tree-identical to e83988fb -- arch-list 2-line change + doc edit), upstream-quality message (no MOAT jargon, explains gfx90a enablement + first-GPU-validation, Test Plan with the literal gtest commands). Force-pushed-with-lease. squash-carry-forward carried all 5 completed platforms to 7bedc886 (no refusal). pr-ready=True. Ready for user's PR-open decision against OpenNMT/CTranslate2 (out of porter scope).

## Validation 2026-06-06 (windows-gfx1201, RX 9070 XT, TheRock ROCm 7.14.0a) -- PASS

Fork: AMD-Ecosystem/CTranslate2 @ moat-port, HEAD dfa0d30dd18c4e65863e091f4ac99d7b936a02f1 (no source change needed)
GPU: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), Windows 11 (HIP_VISIBLE_DEVICES=0, sole GPU on host)
Compiler: clang-cl 23.0.0 (TheRock ROCm 7.14.0a20260604), cmake 4.3.1, all-clang-cl CMake-HIP

### Build notes (windows-gfx1201)

ROCm SDK root: `_rocm_sdk_devel` (TheRock PyTorch venv, multi-arch gfx1101+gfx1201 device extras).
OpenBLAS: same as gfx1101 -- `_rocm_sdk_devel/lib/host-math/` (lib: `rocm-openblas.lib`, headers: `include/openblas/cblas.h`).

Fresh-configure difference vs gfx1101: for a clean build (CMakeFiles deleted), clang-cl 23.0.0 cannot
auto-discover the ROCm device libs from ROCM_PATH alone. The `--rocm-device-lib-path` flag must be
added to BOTH `CMAKE_HIP_FLAGS` AND `CMAKE_CXX_FLAGS` in the init-cache file, because `.cc` files with
`set_source_files_properties(LANGUAGE HIP)` are compiled via the CXX compiler rule (uses
`CMAKE_CXX_FLAGS`), not the HIP rule. On gfx1101 this was masked by reusing the old CMakeFiles/
directory from the gfx1151 configure (which had the device lib path already in the CMake internal
compiler ID cache). The gfx1201 fresh configure exposed the gap.

### Build command (windows-gfx1201)

```
ROCM_ROOT=<TheRock venv>/_rocm_sdk_devel
DEVLIB=$ROCM_ROOT/lib/llvm/amdgcn/bitcode

# Init-cache file /tmp/ct2_gfx1201_init.cmake:
#   set(CMAKE_CXX_FLAGS "/EHsc /wd4267 --rocm-device-lib-path=<DEVLIB> -Wno-deprecated-literal-operator -Wno-unused-command-line-argument" CACHE STRING "" FORCE)
#   set(CMAKE_C_FLAGS "-Wno-unused-command-line-argument" CACHE STRING "" FORCE)
#   set(CMAKE_HIP_FLAGS "--rocm-device-lib-path=<DEVLIB> -Wno-deprecated-literal-operator -Wno-unused-command-line-argument" CACHE STRING "" FORCE)

cmake -S projects/CTranslate2/src -B projects/CTranslate2/build \
  -G Ninja \
  -C /tmp/ct2_gfx1201_init.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$ROCM_ROOT/lib/llvm/bin/clang-cl.exe" \
  -DCMAKE_CXX_COMPILER="$ROCM_ROOT/lib/llvm/bin/clang-cl.exe" \
  -DCMAKE_HIP_COMPILER="$ROCM_ROOT/lib/llvm/bin/clang-cl.exe" \
  -DCMAKE_HIP_STANDARD=17 \
  -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DWITH_MKL=OFF -DWITH_OPENBLAS=ON \
  "-DOPENBLAS_INCLUDE_DIR=$ROCM_ROOT/lib/host-math/include/openblas" \
  "-DOPENBLAS_LIBRARY=$ROCM_ROOT/lib/host-math/lib/rocm-openblas.lib" \
  -DOPENMP_RUNTIME=NONE \
  -DBUILD_TESTS=ON -DBUILD_CLI=OFF \
  "-DCMAKE_PREFIX_PATH=$ROCM_ROOT" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build projects/CTranslate2/build -j64
```

Build time: ~58s. Warnings only (unused-variable, dllimport in spdlog). No errors.
Post-build: copy `build/ctranslate2.dll` into `build/tests/` (CMake copies it at link time but the
DLL in tests/ was stale from the prior gfx1101 build; overwrite with the fresh gfx1201 DLL).

Code-object arch evidence:
```python
data = open('ctranslate2.dll', 'rb').read()
# 23 hipfb files with gfx1201 device code; 41 gfx1201 hits in DLL
```

### GPU test run 1 (CUDA/* filter):
```
cd tests/ && HIP_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ROCBLAS_TENSILE_LIBPATH=.../rocblas/library \
  ./ctranslate2_test.exe <data_dir> --gtest_filter='CUDA/*'
```
Result: **164 PASSED, 3 SKIPPED** (Conv1DGroupNoBiasQuantized x3 dtypes). MATCHES bar.

### Full suite (CPU+GPU, exclude Conv1DGroupNoBiasQuantized):
```
cd tests/ && HIP_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ROCBLAS_TENSILE_LIBPATH=.../rocblas/library \
  ./ctranslate2_test.exe <data_dir> "--gtest_filter=-*Conv1DGroupNoBiasQuantized*"
```
Result: **350 PASSED, 1 SKIPPED** (CPU/OpDeviceFPTest.Conv1DDilation -- intentional). MATCHES bar. No non-GPU regressions.

wave32 verdict: gfx1201 is RDNA4 wave32. C10_WARP_SIZE=32 matches hardware exactly. No wave-size source change needed.

State transition: port-ready -> completed. validated_sha = dfa0d30dd18c4e65863e091f4ac99d7b936a02f1. No source change.

## Validation 2026-06-05 (windows-gfx1101, Radeon PRO V710, TheRock ROCm 7.14.0a) -- PASS

Fork: AMD-Ecosystem/CTranslate2 @ moat-port, HEAD dfa0d30dd18c4e65863e091f4ac99d7b936a02f1 (no source change needed)
GPU: AMD Radeon PRO V710 (gfx1101, RDNA3, wave32), Windows 11 (HIP_VISIBLE_DEVICES=0)
Compiler: clang-cl 23.0.0 (TheRock ROCm 7.14.0a20260605), cmake 4.3.1, all-clang-cl CMake-HIP

### Build notes (windows-gfx1101)

ROCm SDK root: `_rocm_sdk_devel` (same venv as gfx1201, multi-arch build with gfx1101+gfx1201 device extras).
OpenBLAS: same as gfx1151 -- `_rocm_sdk_devel/lib/host-math/` (lib: `rocm-openblas.lib`, headers: `include/openblas/cblas.h`).

CMake HIP compiler detection required adding `--rocm-device-lib-path` to `CMAKE_HIP_FLAGS` via a CMake init-cache file. Without it, clang-cl can't compile the HIP identification source (`CMakeHIPCompilerId.hip`) because the ROCm device libs are not auto-discovered from `ROCM_PATH` alone. The device libs are at `_rocm_sdk_devel/lib/llvm/amdgcn/bitcode/`.

### Build command (windows-gfx1101)

```
ROCM_ROOT=<TheRock venv>/_rocm_sdk_devel
DEVLIB=$ROCM_ROOT/lib/llvm/amdgcn/bitcode

# Init-cache file /tmp/ct2_gfx1101_init.cmake:
#   set(CMAKE_CXX_FLAGS "/EHsc /wd4267 -Wno-deprecated-literal-operator -Wno-unused-command-line-argument" CACHE STRING "" FORCE)
#   set(CMAKE_C_FLAGS "-Wno-unused-command-line-argument" CACHE STRING "" FORCE)
#   set(CMAKE_HIP_FLAGS "--rocm-device-lib-path=<DEVLIB> -Wno-deprecated-literal-operator -Wno-unused-command-line-argument" CACHE STRING "" FORCE)

cmake -S projects/CTranslate2/src -B projects/CTranslate2/build \
  -G Ninja \
  -C /tmp/ct2_gfx1101_init.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$ROCM_ROOT/lib/llvm/bin/clang-cl.exe" \
  -DCMAKE_CXX_COMPILER="$ROCM_ROOT/lib/llvm/bin/clang-cl.exe" \
  -DCMAKE_HIP_COMPILER="$ROCM_ROOT/lib/llvm/bin/clang-cl.exe" \
  -DCMAKE_HIP_STANDARD=17 \
  -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
  -DWITH_MKL=OFF -DWITH_OPENBLAS=ON \
  "-DOPENBLAS_INCLUDE_DIR=$ROCM_ROOT/lib/host-math/include/openblas" \
  "-DOPENBLAS_LIBRARY=$ROCM_ROOT/lib/host-math/lib/rocm-openblas.lib" \
  -DOPENMP_RUNTIME=NONE \
  -DBUILD_TESTS=ON -DBUILD_CLI=OFF \
  "-DCMAKE_PREFIX_PATH=$ROCM_ROOT" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build projects/CTranslate2/build -j64
```

Build time: ~90s. Warnings only (unused-variable, -Wnodiscard on benchmark_ops, dllimport in spdlog). No errors.

Windows-specific build notes (gfx1101 vs gfx1151 differences):
- cmake 4.3.1 works (gfx1151 had 4.3.2, not required).
- CMAKE_HIP_FLAGS must include `--rocm-device-lib-path=<bitcode_dir>`: without this, clang-cl 23.0.0 fails the HIP compiler ID test with "cannot find ROCm device library". The device libs are at `_rocm_sdk_devel/lib/llvm/amdgcn/bitcode/`.
- Flags with spaces (like `/EHsc ...`) must be passed via CMake init-cache file (-C flag), NOT as `-DCMAKE_CXX_FLAGS="..."` on the shell command line. Passing them directly on the shell command line causes cmake to treat `/EHsc` as a Windows path prefix (resolving to `C:/Program Files/Git/EHsc`).
- OPENMP_RUNTIME=NONE same as gfx1151 (avoid /openmp compile flag which has no linker support).
- OMP_NUM_THREADS=1 required at test time: on a 64-core machine, the BS::thread_pool CPU inference executor deadlocks with OpenMP's default thread count. Set OMP_NUM_THREADS=1 (in addition to OPENBLAS_NUM_THREADS=1) to prevent this.
- ROCBLAS_TENSILE_LIBPATH=<sdk_libraries>/bin/rocblas/library for hipBLAS GPU kernels.
- DLL runtime: copy to tests/ dir same as gfx1151: amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll, hiprtc-builtins0714.dll, rocm-openblas.dll, hipblas.dll, hiprand.dll, rocblas.dll, rocrand.dll, rocsolver.dll, libhipblaslt.dll, gtest.dll, gtest_main.dll, ctranslate2.dll.

Code-object arch evidence:
```python
data = open('ctranslate2.dll', 'rb').read()
# Found gfx1101 at offset: 3374178
# Found hip elf target (hipv4-amdgcn-amd-amdhsa--gfx1101): 6452334
```

### GPU test run 1 (CUDA/* filter):
```
cd tests/ && HIP_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ROCBLAS_TENSILE_LIBPATH=.../rocblas/library \
  ./ctranslate2_test.exe <data_dir> --gtest_filter='CUDA/*'
```
Result: **164 PASSED, 3 SKIPPED** (Conv1DGroupNoBiasQuantized x3 dtypes). MATCHES bar.

### GPU test run 2 (CUDA/* filter -- determinism check):
Same command. Result: **164 PASSED, 3 SKIPPED**. IDENTICAL to run 1 -- BIT-DETERMINISTIC.

wave32 verdict: gfx1101 is RDNA3 wave32. C10_WARP_SIZE=32 matches hardware exactly. No wave-size source change needed.

### Full suite (CPU+GPU, exclude Conv1DGroupNoBiasQuantized):
```
cd tests/ && HIP_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ROCBLAS_TENSILE_LIBPATH=.../rocblas/library \
  ./ctranslate2_test.exe <data_dir> "--gtest_filter=-*Conv1DGroupNoBiasQuantized*"
```
Result: **350 PASSED, 1 SKIPPED** (CPU/OpDeviceFPTest.Conv1DDilation -- intentional). MATCHES bar. No non-GPU regressions.

State transition: port-ready -> completed. validated_sha = dfa0d30dd18c4e65863e091f4ac99d7b936a02f1. No source change.

## Validation 2026-06-03 (windows-gfx1151, Radeon 8060S, TheRock ROCm 7.14.0a) -- PASS

Fork: AMD-Ecosystem/CTranslate2 @ moat-port, HEAD dfa0d30dd18c4e65863e091f4ac99d7b936a02f1 (no source change needed)
GPU: AMD Radeon 8060S (gfx1151, RDNA3.5, wave32), Windows 11
Compiler: clang-cl 23.0.0 (TheRock ROCm 7.14.0a20260531), cmake 4.3.2, all-clang-cl CMake-HIP

### OpenBLAS resolution

TheRock ships OpenBLAS at `_rocm_sdk_devel/lib/host-math/` (library: `rocm-openblas.lib`, headers: `include/openblas/cblas.h`). Passed explicitly as:
```
-DOPENBLAS_INCLUDE_DIR=<rocm>/lib/host-math/include/openblas
-DOPENBLAS_LIBRARY=<rocm>/lib/host-math/lib/rocm-openblas.lib
```

### Build command (windows-gfx1151)

```
ROCM_ROOT=<TheRock venv>/_rocm_sdk_devel
# set ROCM_PATH=$ROCM_ROOT (Windows env, backslash)
cmake -S projects/CTranslate2/src -B projects/CTranslate2/build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$ROCM_ROOT/lib/llvm/bin/clang-cl.exe" \
  -DCMAKE_CXX_COMPILER="$ROCM_ROOT/lib/llvm/bin/clang-cl.exe" \
  -DCMAKE_HIP_COMPILER="$ROCM_ROOT/lib/llvm/bin/clang-cl.exe" \
  -DCMAKE_HIP_STANDARD=17 \
  -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DWITH_MKL=OFF -DWITH_OPENBLAS=ON \
  "-DOPENBLAS_INCLUDE_DIR=$ROCM_ROOT/lib/host-math/include/openblas" \
  "-DOPENBLAS_LIBRARY=$ROCM_ROOT/lib/host-math/lib/rocm-openblas.lib" \
  -DOPENMP_RUNTIME=NONE \
  -DBUILD_TESTS=ON -DBUILD_CLI=OFF \
  "-DCMAKE_CXX_FLAGS=/EHsc /wd4267 -Wno-deprecated-literal-operator -Wno-unused-command-line-argument" \
  "-DCMAKE_C_FLAGS=-Wno-unused-command-line-argument" \
  "-DCMAKE_HIP_FLAGS=-Wno-deprecated-literal-operator -Wno-unused-command-line-argument" \
  "-DCMAKE_PREFIX_PATH=$ROCM_ROOT" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build projects/CTranslate2/build -j6  # power-safe cap
```

Build time: 126s. Warnings only (unused-variable, -Wnodiscard on benchmark_ops). No errors.

Windows-specific build notes:
- cmake 4.3.2 required (3.31.0 cannot detect clang-cl HIP ABI on Windows; working exe at C:\Users\<user>\AppData\Local\Temp\pip-uninstall-bkgoot79\cmake.exe)
- OPENMP_RUNTIME=NONE: CMakeLists.txt line 313 adds /openmp compile flag when MSVC=true but never links OpenMP runtime. Using NONE avoids the /openmp compile flag entirely (no omp_get_max_threads undefined at link).
- /EHsc in CMAKE_CXX_FLAGS: required for exceptions in test .cc files; clang-cl disables exceptions by default without it.
- -Wno-unused-command-line-argument: suppresses CMake-injected /d2FH4- flag that clang-cl rejects as -Werror.
- ROCM_PATH env must be set to the Windows SDK root so link_directories($ROCM_PATH/lib) resolves correctly (not /opt/rocm).
- Runtime deployment: copy amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprand.dll, hipblas.dll, rocsolver.dll, rocblas.dll, libhipblaslt.dll, rocrand.dll, rocm-openblas.dll, gtest.dll, gtest_main.dll beside ctranslate2_test.exe (Windows loader prefers exe dir over System32 for TheRock DLLs).
- OPENBLAS_NUM_THREADS=1 required at test time: rocm-openblas has internal thread spawning that conflicts with BS::thread_pool (no-OpenMP fallback), causing deadlock in CPU inference tests. Setting OPENBLAS_NUM_THREADS=1 prevents OpenBLAS from spawning worker threads.
- ROCBLAS_TENSILE_LIBPATH=<sdk_libraries>/bin/rocblas/library/gfx1151 for hipBLAS GPU kernels.

Code-object arch evidence:
```
strings ctranslate2.dll | grep gfx1151 | head -3
# hipv4-amdgcn-amd-amdhsa--gfx1151
```

### GPU test run 1 (CUDA/* filter):
```
cd tests/ && OPENBLAS_NUM_THREADS=1 ROCBLAS_TENSILE_LIBPATH=.../rocblas/library/gfx1151 \
  ./ctranslate2_test.exe <data_dir> --gtest_filter='CUDA/*'
```
Result: **164 PASSED, 3 SKIPPED** (Conv1DGroupNoBiasQuantized x3 dtypes). MATCHES bar.

### GPU test run 2 (CUDA/* filter -- determinism check):
Same command. Result: **164 PASSED, 3 SKIPPED**. IDENTICAL to run 1 -- BIT-DETERMINISTIC.

wave32 verdict: gfx1151 is RDNA3.5 wave32, same warp class as gfx1100 (already validated). C10_WARP_SIZE=32 matches hardware exactly. No wave-size source change needed.

### Full suite (CPU+GPU, exclude Conv1DGroupNoBiasQuantized):
```
cd tests/ && OPENBLAS_NUM_THREADS=1 ROCBLAS_TENSILE_LIBPATH=.../rocblas/library/gfx1151 \
  ./ctranslate2_test.exe <data_dir> "--gtest_filter=-*Conv1DGroupNoBiasQuantized*"
```
Result: **350 PASSED, 1 SKIPPED** (CPU/OpDeviceFPTest.Conv1DDilation -- intentional). MATCHES bar. No non-GPU regressions.

State transition: port-ready -> completed. validated_sha = dfa0d30dd18c4e65863e091f4ac99d7b936a02f1. No source change.

## Validation 2026-05-31 (linux-gfx1100, gfx1100/W7800, ROCm 7.2.1) -- PASS

Fork: AMD-Ecosystem/CTranslate2 @ moat-port, HEAD dfa0d30dd18c4e65863e091f4ac99d7b936a02f1 (unchanged, no follower fork push)
GPU: 4x AMD Radeon Pro W7800 48GB, gfx1100 (RDNA3, wave32). Tested on HIP_VISIBLE_DEVICES=0.
Compiler: amdclang++ 22.0.0 (roc-7.2.1), cmake 3.31

Build command (gfx1100, adapted from lead):
```
cmake -S projects/CTranslate2/src -B projects/CTranslate2/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=amdclang -DCMAKE_CXX_COMPILER=amdclang++ \
  -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DWITH_MKL=OFF -DWITH_OPENBLAS=ON -DOPENMP_RUNTIME=COMP \
  -DBUILD_TESTS=ON -DBUILD_CLI=OFF \
  -DCMAKE_CXX_FLAGS="-Wno-deprecated-literal-operator" \
  -DCMAKE_HIP_FLAGS="-Wno-deprecated-literal-operator"
bash utils/timeit.sh CTranslate2 compile -- cmake --build projects/CTranslate2/build -j 16
```
Result: compile CLEAN in 119.5s. Only -Wnodiscard on benchmark_ops (same as lead).

Code-object arch evidence (roc-obj-ls on libctranslate2.so):
All bundles: `hipv4-amdgcn-amd-amdhsa--gfx1100` (no gfx90a present). Confirmed via roc-obj-ls.

GPU test (CUDA/* filter, HIP_VISIBLE_DEVICES=0):
```
HIP_VISIBLE_DEVICES=0 bash utils/timeit.sh CTranslate2 test -- \
  projects/CTranslate2/build/tests/ctranslate2_test \
  projects/CTranslate2/src/tests/data --gtest_filter='CUDA/*'
```
Result: 164 PASSED, 3 SKIPPED (Conv1DGroupNoBiasQuantized x3 dtypes). MATCHES gfx90a bar.

Full suite (filter out Conv1DGroupNoBiasQuantized CPU abort):
```
HIP_VISIBLE_DEVICES=0 bash utils/timeit.sh CTranslate2 test -- \
  projects/CTranslate2/build/tests/ctranslate2_test \
  projects/CTranslate2/src/tests/data "--gtest_filter=-*Conv1DGroupNoBiasQuantized*"
```
Result: 350 PASSED, 1 SKIPPED (CPU/OpDeviceFPTest.Conv1DDilation). MATCHES gfx90a bar. No non-GPU regressions.

Determinism (wave32, agent_space/ct2_determinism_gfx1100.cc):
softmax/log_softmax/layer_norm/rms_norm across widths {15,32,96,1023,2048} run twice each on GPU: ALL BIT-IDENTICAL (20/20).

State transition: port-ready -> completed. validated_sha = dfa0d30. Fork untouched (no commit, no push).

## Validation 2026-05-31 (linux-gfx90a, gfx90a/MI250X, ROCm 7.2.1) -- PASS

Fork: AMD-Ecosystem/CTranslate2 @ moat-port, HEAD dfa0d30dd18c4e65863e091f4ac99d7b936a02f1
GPU arch: gfx90a (MI250X), device 2 (HIP_VISIBLE_DEVICES=2, 0% load at test time)
Compiler: amdclang++ 22.0.0 (roc-7.2.1), cmake 3.31

Compile:
```
cmake --build projects/CTranslate2/build -j 16
```
Result: all targets rebuilt from cache (no source change), compile CLEAN (0 errors, -Wnodiscard only on benchmark_ops, suppressed in build).

GPU test (CUDA/* filter, device 2):
```
HIP_VISIBLE_DEVICES=2 bash utils/timeit.sh CTranslate2 test -- \
  projects/CTranslate2/build/tests/ctranslate2_test \
  projects/CTranslate2/src/tests/data --gtest_filter='CUDA/*'
```
Result: 164 PASSED, 3 SKIPPED (Conv1DGroupNoBiasQuantized x3 dtypes -- intentional feature gap). MATCHES DOCUMENTED BAR.

Full suite (filter out Conv1DGroupNoBiasQuantized CPU abort):
```
HIP_VISIBLE_DEVICES=2 bash utils/timeit.sh CTranslate2 test -- \
  projects/CTranslate2/build/tests/ctranslate2_test \
  projects/CTranslate2/src/tests/data "--gtest_filter=-*Conv1DGroupNoBiasQuantized*"
```
Result: 350 PASSED, 1 SKIPPED (CPU/OpDeviceFPTest.Conv1DDilation -- intentional). MATCHES DOCUMENTED BAR. No non-GPU regressions.

Determinism (wave64 bar, agent_space/ct2_determinism.cc):
softmax/log_softmax/layer_norm/rms_norm/quantize_scale across widths {15,32,96,1023,2048} run twice each on GPU: ALL BIT-IDENTICAL (25/25).

State transition: review-passed -> completed. validated_sha = dfa0d30.

## Disposition: VERIFY/MODERNIZE -- mature upstream HIP backend, first CDNA2 (gfx90a/wave64) GPU validation

CTranslate2 ships a mature Strategy-A ROCm/HIP backend (WITH_HIP=ON: enable_language(HIP) +
LANGUAGE HIP source props + CT2_USE_HIP compat-header aliasing in src/cuda/{utils.h,primitives.cu,
allocator.cc,random.h,helpers.h}). Every upstream arch list targets RDNA only
(gfx1030;gfx1100;...;gfx1201); gfx90a/CDNA2 (wave64) was absent everywhere, and upstream CI runs
zero GPU tests on ROCm (CPU-container wheel builds only). The MOAT deliverable is the first CDNA2
GPU validation + any wave64 fix.

Result: the stock HIP backend builds for gfx90a with NO source change and the full GPU gtest suite
passes + is bit-deterministic on MI250X. No wave64 source fix was required. The fork change is the
minimal build-config enablement: add gfx90a to the two Linux ROCm arch lists.

## Lead platform: linux-gfx90a (MI250X, ROCm 7.2.1) -- VALIDATED

- Box: 4x MI250X GCDs (all gfx90a, devices 0-3). Pin a free GCD with HIP_VISIBLE_DEVICES (do not
  hardcode); validation ran on device 1.
- Build: amdclang/amdclang++ 22.0.0 (roc-7.2.1), cmake 3.31, OpenBLAS 0.3.26 (apt libopenblas-dev).
- Build out-of-the-box: configure succeeded with -DCMAKE_HIP_ARCHITECTURES=gfx90a (the library
  CMakeLists reads the cache var; no hardcoded literal arch list in CMakeLists.txt -- the RDNA lists
  live only in docker/Dockerfile_rocm + python/tools/prepare_build_environment_*_rocm.sh). Compile
  clean (158s, -j16); only -Wnodiscard warnings on benchmark_ops. hipBLAS enum/algo coverage
  (cublasGemmEx/StridedBatchedEx -> hipblas*, HIPBLAS_COMPUTE_16F/32F/32I) resolved at compile with
  no edit -- the plan's risk 3 is a non-issue on 7.2.1.

## Build script (gfx90a, BUILD_TESTS, OpenBLAS CPU backend)

```
cmake -S projects/CTranslate2/src -B projects/CTranslate2/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=amdclang -DCMAKE_CXX_COMPILER=amdclang++ \
  -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DWITH_MKL=OFF -DWITH_OPENBLAS=ON -DOPENMP_RUNTIME=COMP \
  -DBUILD_TESTS=ON -DBUILD_CLI=OFF \
  -DCMAKE_CXX_FLAGS="-Wno-deprecated-literal-operator" \
  -DCMAKE_HIP_FLAGS="-Wno-deprecated-literal-operator"
cmake --build projects/CTranslate2/build -j 16
```

Prereq: `git -C projects/CTranslate2/src submodule update --init --recursive` (the depth-1 clone did
not init submodules; googletest is needed for tests). third_party/{thrust,cutlass} are fetched but
unused on the HIP build. A follower (gfx1100/gfx1151) builds the same source by changing only
-DCMAKE_HIP_ARCHITECTURES (upstream already targets those RDNA arches).

## GPU validation (the gate) -- PASS

Test binary: `ctranslate2_test` (gtest), takes the data dir as argv[1]:
`./tests/ctranslate2_test <src>/tests/data`. Device-parametrized: CPU/* run on CPU, CUDA/* run on
the AMD GPU on a HIP build (the build defines CT2_WITH_CUDA so the CUDA instantiations compile).

- GPU-only set `--gtest_filter='CUDA/*'`: 164/167 PASS, 3 SKIPPED. The 167 CUDA cases cover the full
  wave64 reduction surface: SoftMax/LogSoftMax/MaskedSoftMax/MaskedSoftMaxTriangular, LayerNorm/
  LayerNormAxis/RMSNorm, TopK/TopPMask/Multinomial, QuantizeINT8, Gemm/GemmBias/GemmInt8/GemmGELU,
  Conv1D, plus CUDA/LayerDeviceFPTest (full transformer layers: attention/FFN/embedding, FP32+FP16+
  BF16 end-to-end on GPU), CUDA/PrimitiveTest (hipBLAS GEMM + device memory), CUDA/BiasedDecoding,
  CUDA/StorageViewDeviceTest. Real GPU execution confirmed via AMD_LOG_LEVEL=3 (15 hipLaunchKernel ->
  hipSuccess per SoftMax test).
- Full suite (CPU+GPU+inference), `--gtest_filter='-*Conv1DGroupNoBiasQuantized*'`: 350/351 PASS, 1
  SKIPPED (CPU dilated-conv, intentional). Includes TranslatorTest/ModelTest/DecodingTest/
  BatchingTest/CrossAttentionTest on the bundled tiny models under tests/data.

### Determinism (PORTING_GUIDE wave64 bar) -- PASS, bit-identical

The gtest pass/skip set is identical across two runs. Beyond that, a standalone harness
(agent_space/ct2_determinism.cc) runs softmax/log_softmax/layer_norm/rms_norm/quantize on the GPU
twice with fixed input over row widths {15,32,96,1023,2048} (sub-warp, exact-warp, multi-warp,
odd-ILP-epilogue, 32-partition) and memcmp's the raw output bytes: ALL BIT-IDENTICAL. The
__syncwarp(mask) tail in block_reduce introduces no wave64 race on gfx90a.

### Intentional skips (not regressions)

- CUDA/OpDeviceFPTest.Conv1DGroupNoBiasQuantized (x3 dtypes): the test itself GTEST_SKIPs for
  non-CPU devices ("Grouped quantized convolution is not implemented for CUDA", ops_test.cc:1325).
  This is a CUDA-path feature gap that predates the port; the HIP build inherits it. Not a defect.
- CPU/OpDeviceFPTest.Conv1DDilation: "Dilated convolution is not implemented for CPU" (intentional).

### Known CPU-backend artifact (orthogonal to the port)

`CPU/OpDeviceFPTest.Conv1DGroupNoBiasQuantized/float32` THROWS ("No INT8 GEMM backend for CPU") and
aborts the whole binary because this validation builds OpenBLAS-only (-DWITH_OPENBLAS=ON, no
oneDNN/MKL). The CPU int8 GEMM backend comes from MKL/DNNL, which we omit for a correctness build.
This would also throw on an x86 CUDA build with the same OpenBLAS-only CPU config; it is not a
GPU/HIP regression and the GPU variants of this test skip on their own. Run the suite with
`--gtest_filter='-*Conv1DGroupNoBiasQuantized*'` for a clean pass, or build with -DWITH_DNNL=ON to
get the CPU int8 path (not needed to validate the GPU port).

## Wave64 analysis: src/cuda/helpers.h C10_WARP_SIZE 32 -- left UNCHANGED (correct on wave64)

helpers.h:398 `#define C10_WARP_SIZE 32` is unconditional and drives the host `get_block_size()`
(:407, computes the launch dim3) and the device `block_reduce()` (:411-455) used by softmax_gpu.cu
and quantize_gpu.cu. The plan flagged this as the PRIMARY wave64 risk. Conclusion after analysis +
hardware validation: leave it at 32.

Why 32 is correct here (it is a logical shared-memory partition size, NOT the hardware warp size):
block_reduce writes all blockDim.x values to __shared__, __syncthreads(), then the first
C10_WARP_SIZE threads each reduce a contiguous C10_WARP_SIZE-stride group from shared memory
(smem[lane*32 + i]), write smem[lane], __syncthreads(), and thread 0 reduces smem[0..blockDim.x/32).
The coverage is exact for ANY partition size W that divides blockDim.x with blockDim.x/W <= W,
regardless of the hardware wavefront width, because every value is read from the shared memory the
whole block wrote (not from lane registers). get_block_size returns powers of two >= 32, so
blockDim.x is always a multiple of 32 and blockDim.x/32 <= 1024/32 = 32. So the reduced VALUE is
correct on wave64 unchanged.

Critically, do NOT key C10_WARP_SIZE on __GFX9__ here: that is the raft/cuvs HOST/DEVICE WarpSize
split trap (PORTING_GUIDE 2026-05-31). get_block_size is HOST code where __GFX9__ is undefined, so
it would read 32 and launch a 32-thread block, while the device block_reduce would partition by 64
and compute blockDim.x/64 = 0 warps -> the reduction loop never runs -> zero/garbage output. Host
and device MUST use the same W; 32 is the value that is simultaneously correct on CUDA, RDNA, and
CDNA, so the single unconditional define is the right abstraction.

The only genuine wave64 hazard was the `__syncwarp(mask)` tail (helpers.h:435, mask =
(1<<(blockDim.x/32))-1) executed by only the first blockDim.x/32 lanes of a 64-lane wavefront -- the
MPPI-Generic partial-mask warp-sync class. But here the mask explicitly names exactly the active
lanes (all within the first wavefront, since blockDim.x/32 <= 32 < 64) and the WaR hazard it guards
(lane 0 reading smem[1] vs lane 1 writing smem[1]) is among lanes that are all in that one
wavefront and all named in the mask. HIP/gfx90a handles this correctly: 164/167 GPU tests pass and
the determinism harness is bit-identical. So no MPPI-style "drop the tail" fix is needed (and it
would be unsafe here -- a bare drop reintroduces the inter-lane WaR race, and a __syncthreads inside
the divergent `if (threadIdx.x < C10_WARP_SIZE)` branch would hang).

## Out of scope on HIP (expected-unsupported, not defects)

- AWQ-int4 (src/ops/awq/*_gpu.cu) and FlashAttention: REMOVED from the HIP build (CMakeLists.txt
  708-723). These are the CUTLASS/sm80 kernels; CUTLASS does not port to ROCm. No tests reference
  them. The gemv_gpu.cu `#define WARP_SIZE 32` + __shfl_down_sync(0xffffffff) and the
  flash-attention utils.h __shfl_xor_sync(uint32_t(-1)) are in these excluded files -- moot.
- cuDNN conv1d (conv1d_cudnn_gpu.cu, WITH_CUDNN) is CUDA-only; HIP uses the generic im2col+GEMM
  conv1d_gpu.cu (validated: CUDA/OpDeviceFPTest.Conv1D* pass).
- NCCL/RCCL tensor parallelism (CT2_WITH_TENSOR_PARALLEL) is incompatible with WITH_HIP
  (CMakeLists.txt:687 FATAL_ERROR) -- out of scope.

## Fork change (minimal, NVIDIA-safe, RDNA-safe)

Library source is byte-identical to upstream (the wave64 fix was validated unnecessary). The only
change adds gfx90a to the two Linux ROCm arch lists so CDNA2 is a documented, wheel-buildable target:
- docker/Dockerfile_rocm:70 HIP_ARCHITECTURES default
- python/tools/prepare_build_environment_linux_rocm.sh:43 PYTORCH_ROCM_ARCH
Both prepend gfx90a; existing RDNA wheels and the NVIDIA/CUDA build are untouched. The Windows ROCm
prepare script is left alone (gfx90a is a Linux CDNA datacenter part; Windows targets RDNA gfx1151).

## Followers

gfx1100 (RDNA3, wave32) and gfx1151 (RDNA3.5, wave32) are arches upstream ALREADY targets and ships
wheels for. They build the same source with only -DCMAKE_HIP_ARCHITECTURES changed. On wave32 the
C10_WARP_SIZE 32 partition matches the hardware warp exactly (the historically-tested path), so no
new wave64 concern. Validate on their own hardware per MOAT follower flow.

## Review 2026-05-31 (reviewer, lead linux-gfx90a) -- PASS, no changes requested

Reviewed moat-port HEAD dfa0d30 vs upstream base 5dfc5d8 via /pr-review (local-branch mode).
No problems found. Verdict: review-passed; safe to proceed to GPU validation toward completed.

Verification trail (all load-bearing claims independently re-derived from the fork tree):
- Minimal footprint CONFIRMED: `git diff 5dfc5d8...HEAD` touches exactly 2 files, 1 line each --
  docker/Dockerfile_rocm:70 and python/tools/prepare_build_environment_linux_rocm.sh:43, each
  prepending `gfx90a;` to an RDNA-only list. src/ include/ CMakeLists.txt cmake/ third_party/ diff
  is empty: library source byte-identical to upstream.
- NVIDIA + RDNA builds unperturbed: docker `ARG HIP_ARCHITECTURES` (line 69) precedes the `ENV ...:-`
  default (line 70), so the new value is only a fallback default; any build passing --build-arg or a
  CUDA build (separate CMAKE_CUDA_ARCHITECTURES path) is unaffected. Change is additive (prepend), no
  prior target dropped. Windows ROCm prepare script (line 22) correctly left RDNA-only.
- block_reduce/C10_WARP_SIZE=32 wave64 analysis VALID. Only two callers of the hand-rolled
  cuda::block_reduce: softmax_gpu.cu:222/228 and quantize_gpu.cu:70, both launching blockDim =
  get_block_size() (power of two, >=32, <=max_threads=1024 -> blockDim.x/32 in [1,32]) with smem
  sized blockDim.x*sizeof(float). First-warp loop covers smem[0,blockDim.x) exactly (no gap/overlap/
  OOB); all inputs read from shared memory after the line-422 __syncthreads (no __shfl in the
  function) so the VALUE is wave-width-independent. The one genuine intra-wavefront WaR hazard (lane
  0 reading smem[1..N-1] at :433 vs lanes 1..N-1 writing smem[lane] at :436, all within the first 32
  lanes = first half of one wave64 wavefront) IS correctly ordered by __syncwarp(mask) at :435 with
  mask=(1<<(blockDim.x/32))-1 naming exactly those active lanes. So the tail must be KEPT (an
  MPPI-style "drop the __syncwarp" fix would reintroduce that race); leaving it is correct. On HIP
  __syncwarp(MaskT) (amd_warp_sync_functions.h:176) takes the uint64_t mask and syncs the wavefront,
  so it cannot under-synchronize. Keying C10_WARP_SIZE on __GFX9__ would break the host get_block_size
  (host has no __GFX9__) -> host/device split; the unconditional 32 is the correct shared abstraction.
- Other reduction-surface ops (rms_norm/layer_norm/topk/multinomial/mean) use cub::BlockReduce, and
  utils.h:15 `#define cub hipcub` routes them to rocPRIM (wave-size-aware by construction) -- no
  hardcoded-32 or hand-rolled shuffle path there. Library swaps correct (find_package hipblas/hiprand/
  rocprim/rocthrust/hipcub; Strategy A `set_source_files_properties(... LANGUAGE HIP)`).
- Out-of-scope exclusions are upstream's existing HIP CMake (unchanged by the fork): WITH_TENSOR_PARALLEL
  and WITH_FLASH_ATTN -> FATAL_ERROR, AWQ/CUTLASS sources REMOVE_ITEM'd. Consistent with the notes.
- Commit hygiene CLEAN: title "[ROCm] Add gfx90a (CDNA2/MI250X) to the ROCm build arch lists" = 61
  chars; body discloses Claude (Anthropic); Test Plan with literal commands; no Co-Authored-By/noreply,
  no ghstack, no AMD-internal account ref (the author is the user's own public commit
  identity, permitted). Fork Actions disabled ("enabled":false).
- GPU run not re-executed at review (validator's stage, per PORTING_GUIDE 2026-05-30); porter recorded
  CUDA/* 164/167 +3 skip, full 350/351 +1 skip, determinism bit-identical. Not a blocker for review.

## Upstream-PR disposition: SMALL DELTA PR is warranted (corrected 2026-06-04) -- SUPERSEDED 2026-06-11

SUPERSEDED by the FINAL DECISION at the top of this file: jeff judged the delta too trivial to PR, so outcome=validated and NO PR is opened. The analysis below is retained only for the technical wave64/arch-list facts; its "PR is warranted" conclusion no longer holds.

An earlier README-sweep note here said "DO NOT open a duplicate PR -- upstream already supports ROCm." That was WRONG and is retracted. Upstream has a mature HIP backend (the planner knew this: disposition was verify/modernize, not fresh port), but our port genuinely EXTENDS it, and the delta is contributable:

1. gfx90a (CDNA2) enablement -- the entire code delta is 2 lines: prepend `gfx90a` to the ROCm arch lists in docker/Dockerfile_rocm and python/tools/prepare_build_environment_linux_rocm.sh. Upstream's HIP backend is RDNA-only (gfx1030;gfx11xx;gfx12xx); gfx90a/MI250X is absent from every arch list, docker image, and CI, so MI-class users cannot use the prebuilt ROCm wheels.
2. First GPU validation of the ROCm path on ANY arch. Upstream CI compiles ROCm wheels in a CPU-only container and only builds docker images (zero GPU execution). We ran the full gtest suite on gfx90a (wave64, first CDNA run), gfx1100 (wave32), and gfx1151 (wave32, Windows/TheRock).
3. wave64 de-risk finding (no fix needed): `C10_WARP_SIZE 32` (helpers.h:398) LOOKS like the FBGEMM wave64 fault class but is benign here -- it feeds a shared-memory PARTITION reduction (`smem[lane*C10_WARP_SIZE+i]`, __syncthreads-bracketed), not __shfl masks/lane indexing, so it is a tiling constant, not a hardware-warp-width assumption; correct on a 64-wide wave, left untouched.

PR framing: "Add gfx90a (CDNA2) to the ROCm build arch lists; validated on MI250X." Minimal, NVIDIA/RDNA-safe, welcome to a project that actively maintains its HIP backend. NOT a duplicate to skip. See [[moat-no-duplicate-amd-ports]].
