# kaldifeat notes

## Environment (Linux gfx90a lead)
- Host arch: gfx90a (MI250X / CDNA2, wave64), ROCm 7.2.1.
- ROCm PyTorch env (shared with k2): `/opt/conda/envs/py_3.12`
  - torch 2.13.0a0+git8f9a6c8, `torch.version.hip` = 7.2.53211
  - `torch.cuda.is_available()` = True (ROCm), `torch.cuda.device_count()` = 4 (MI250X)
  - `torch.utils.hipify.__version__` = 2.0.0 (hipify v2). NOTE: not used by this build (custom CMake extension, not torch CUDAExtension) and no c10 stream sites, so TORCH_HIPIFY_V2 handling is N/A here.
- Build interpreter MUST be `/opt/conda/envs/py_3.12/bin/python` so `cmake/torch.cmake` resolves the ROCm torch (it runs `import torch` via PYTHON_EXECUTABLE).

## Build facts
- Pure CMake (Strategy A). `setup.py` uses a custom `cmake_extension`/`BuildExtension` (cmake/cmake_extension.py) that shells out to cmake+make; NOT torch's CUDAExtension. So torch build-time source-hipify never runs.
- `.cu`/`.cuh` count = 0. No CUDA/c10::cuda/CUDAStream/cublas/cufft/thrust/cub sites in csrc. GPU compute is delegated entirely to libtorch tensor ops; device is a `torch::Device` from a user string ("cuda"). Confirmed: on ROCm torch, `cuda:0` maps to "AMD Instinct MI250X / MI250" via the HIP backend, so the existing code runs on the AMD GPU unchanged.

## Source change required (small, build-only)
A pure rebuild was NOT quite enough: the C++ unit-test executables (`-Dkaldifeat_BUILD_TESTS=ON`) needed two ROCm-only build fixes. The library/python module itself needs no source change. All fixes are guarded on `torch.version.hip` (a ROCm torch) so CUDA/CPU builds are untouched.

1. cmake/torch.cmake -- detect a ROCm torch (`torch.version.hip` non-empty) and `find_library` the HIP runtime (`amdhip64`) and `rocprofiler-sdk`. On ROCm, `${TORCH_LIBRARIES}` drags in the static `libkineto.a` whose ROCm profiler references `hipRuntimeGetVersion@hip_4.2` (in `libamdhip64.so`); the shared torch libs do not re-export it, so the test exes fail to link without the HIP runtime on their own link line.
2. kaldifeat/csrc/CMakeLists.txt --
   - Link the HIP runtime into the shared `kaldifeat_core` (PUBLIC) so the test exes inherit it.
   - For the gtest executables, compile the core sources directly via an OBJECT library (`kaldifeat_core_obj`) instead of linking the shared `libkaldifeat_core.so`. Root cause: at process startup the ROCm profiler (`librocprofiler-sdk`, pulled in by `libkineto.a`) recursively `dlopen`s the executable's new shared dependency (`libkaldifeat_core.so`) during its own one-time `pthread_once` static init -> `__gnu_cxx::recursive_init_error`, aborting before any test runs. (gdb backtrace: `__hipRegisterFatBinary` in librocsparse -> `__gxx_personality_v0` throw inside rocprofiler-sdk's pthread_once.) Compiling the core into the exe removes the startup dlopen; the static-core exe then also needs `rocprofiler-sdk` on its link line (the shared `.so` had resolved it transitively).
   - The shared `kaldifeat_core` (used by the python module `_kaldifeat.so` and the install) is unchanged; the python module never hits the recursive-dlopen abort because it is imported into an already-initialized process.

These C++ gtests are CPU-only unit tests (feature-window math, RecyclingVector) and upstream CI does NOT build them (`kaldifeat_BUILD_TESTS` defaults OFF; the upstream CUDA workflow does not override it). They are not the GPU validation gate but are made to pass on ROCm for completeness.

## Version tag cosmetic note
`get_version.py:with_cuda()` keys on `nvcc` being on PATH. On ROCm there is no nvcc, so the wheel/egg local-version tag is `+cpu.torch2.13.0a0` even though the build is GPU-capable. This is cosmetic (packaging name only); the build runs on the AMD GPU regardless. Left as-is to avoid changing the PyPI naming scheme; a downstream consumer should not key on the `+cpu` suffix.

## Build commands (gfx90a, verified)
```
PYI=/opt/conda/envs/py_3.12/bin/python
cmake -S projects/kaldifeat/src -B projects/kaldifeat/src/build \
  -DPYTHON_EXECUTABLE=$PYI -Dkaldifeat_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build projects/kaldifeat/src/build -j
```
Python install (the supported consumer path):
```
cd projects/kaldifeat/src
KALDIFEAT_MAKE_ARGS="-j" /opt/conda/envs/py_3.12/bin/python setup.py install --prefix <prefix>
```
No `CMAKE_HIP_ARCHITECTURES` / `enable_language(HIP)` -- no device code is compiled here; GPU arch comes from the linked ROCm torch at runtime, so the build is arch-portable with no source change (followers just need a matching ROCm torch).

## Tests (verified on gfx90a, GPU exercised)
Pin one GCD: `HIP_VISIBLE_DEVICES=0` (then `device_count()`=1, so `get_devices()` returns `[cpu, cuda:0]`).
- ctest (`-Dkaldifeat_BUILD_TESTS=ON`): 12/12 pass. = 2 C++ gtests (feature-window-test, online-feature-test) + 10 registered python tests (test_fbank, test_fbank_options, test_frame_extraction_options, test_mel_bank_options, test_mfcc, test_mfcc_options, test_plp, test_plp_options, test_spectrogram, test_spectrogram_options). Each python test loops over `get_devices()` and compares the cuda:0 result against the precomputed CPU reference (`test_data/*.txt`) and the CPU run via `torch.allclose`.
- Whisper tests (NOT ctest-registered upstream, run standalone): test_whisper_fbank, test_whisper_v3_fbank -- both PASS. Self-contained (vendored mel-bank headers, no network/model download).
- GPU proof: `torch.cuda.get_device_name(0)` = "AMD Instinct MI250X / MI250"; a fbank computed on cuda:0 returns a tensor on cuda:0 with ~80 MB GPU memory allocated during compute; `torch.version.hip` non-None. Not a silent CPU fallback.
```
HIP_VISIBLE_DEVICES=0 \
PYTHONPATH=$PWD/kaldifeat/python:$PWD/build/lib \
ctest --test-dir build --output-on-failure
```

## Install as a dependency
k2/torch consumes `AMD-Ecosystem/kaldifeat @ moat-port` via `cmake/kaldifeat.cmake`. Verified recipe:
```
# 1. Clone the ROCm port branch
git clone -b moat-port https://github.com/AMD-Ecosystem/kaldifeat _deps/kaldifeat/src

# 2. Build + install against the ROCm torch env (produces _kaldifeat.so,
#    libkaldifeat_core.so, and kaldifeatConfig.cmake)
cd _deps/kaldifeat/src
KALDIFEAT_MAKE_ARGS="-j" /opt/conda/envs/py_3.12/bin/python setup.py install \
  --prefix $(pwd)/../install
```
This installs (verified paths under `<prefix>/lib/python3.12/site-packages/kaldifeat-*.egg/`):
- `_kaldifeat.cpython-312-x86_64-linux-gnu.so` (python extension)
- `kaldifeat/lib/libkaldifeat_core.so` (C++ core shared lib)
- `kaldifeat/share/cmake/kaldifeat/kaldifeatConfig.cmake` + `kaldifeatConfigVersion.cmake` (CMake package)

For a CMake consumer (e.g. k2/torch), point `find_package(kaldifeat)` at the installed
`kaldifeatConfig.cmake` (`-Dkaldifeat_DIR=<prefix>/.../kaldifeat/share/cmake/kaldifeat`)
instead of re-fetching upstream in `cmake/kaldifeat.cmake` (k2 pins FetchContent at tag
v1.25.4; redirect it to the installed jeffdaily build). For a python consumer, put the egg
dir on `PYTHONPATH` or `pip install .` into the env. The build env MUST be the ROCm torch
(`/opt/conda/envs/py_3.12`); kaldifeat inherits its GPU backend from the linked torch.
</content>
</invoke>

## Validation 2026-06-12

Platform: linux-gfx90a (MI250X GCD 2, ROCm 7.2.1, HIP 7.2.53211)
Validated SHA: 8d3c066348129b1f6957bc6c1383295f51b05d97
Result: PASSED

### Additional build fix applied during validation

torch 2.13.0a0+gitb5e90ff (a newer build than the porter used, +gitb5e90ff vs +git8f9a6c8) updated its headers to use C++20 `requires` constraints in TensorImpl.h. Building kaldifeat with the project's default C++17 produced:
```
error: 'requires' does not name a type
error: 'SetDimsTemplate' was not declared in this scope
```

Root cause: TorchConfig.cmake sets `CXX_STANDARD 20` on the torch interface target, but the project's CMakeLists.txt defaults to C++17. The cmake/torch.cmake was updated to read the CXX_STANDARD property from the torch target after find_package(Torch) and raise CMAKE_CXX_STANDARD to match if the current standard is lower. This is a no-op on older torch (C++17 or unset) and on CUDA builds. Committed as a follow-up to the porter's commit.

### Environment note

The host env (py_3.12) had NumPy 2.2.6 installed but torch was compiled against NumPy 1.x ABI, making torch.from_numpy() fail with "Numpy is not available". Fixed by installing numpy 1.26.4 in the user location (pip install --user "numpy<2"), which overrides the env's 2.x. This is a shared-env issue not specific to kaldifeat.

### Build commands
```
PYI=/opt/conda/envs/py_3.12/bin/python
cmake -S projects/kaldifeat/src -B projects/kaldifeat/src/build \
  -DPYTHON_EXECUTABLE=$PYI -Dkaldifeat_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build projects/kaldifeat/src/build -j
```
Configure output confirms: "Raising CMAKE_CXX_STANDARD from 17 to 20 (torch requirement)"

### Test results
```
HIP_VISIBLE_DEVICES=2 \
PYTHONPATH=$PWD/kaldifeat/python:$PWD/build/lib \
ctest --test-dir build --output-on-failure
# 100% tests passed, 12 tests passed out of 12
# 1: Test.feature-window-test -- Passed
# 2: Test.online-feature-test -- Passed
# 3-12: test_fbank, test_mfcc, test_plp, test_spectrogram and their _options variants -- all Passed

HIP_VISIBLE_DEVICES=2 PYTHONPATH=$PWD/kaldifeat/python:$PWD/build/lib \
  $PYI kaldifeat/python/tests/test_whisper_fbank.py    # PASS
HIP_VISIBLE_DEVICES=2 PYTHONPATH=$PWD/kaldifeat/python:$PWD/build/lib \
  $PYI kaldifeat/python/tests/test_whisper_v3_fbank.py  # PASS
```

GPU proof: torch.cuda.get_device_name(0) = "AMD Instinct MI250X / MI250", torch.version.hip = 7.2.53211, ~80 MB GPU memory allocated during fbank compute (kaldifeat.Fbank on cuda:0). Not a CPU fallback.

CUDA no-regression gate: not applicable for linux-gfx90a lead on this port; kaldifeat has zero .cu/.cuh device code.

## Validation 2026-06-12 (linux-gfx1100)

Platform: linux-gfx1100 (AMD Radeon Pro W7800 48GB, gfx1100 / RDNA3, ROCm 7.2.1, HIP 7.2.53211)
Validated SHA: 8d3c066348129b1f6957bc6c1383295f51b05d97
Result: PASSED (follower validation, no delta-port required)

### Build commands
```
PYI=/opt/conda/envs/py_3.12/bin/python
cmake -S projects/kaldifeat/src -B projects/kaldifeat/src/build \
  -DPYTHON_EXECUTABLE=$PYI -Dkaldifeat_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build projects/kaldifeat/src/build -j
```
Configure output: "Raising CMAKE_CXX_STANDARD from 17 to 20 (torch requirement)" (same as gfx90a lead). Build succeeded, no errors or warnings beyond CMake version deprecations.

### Test results
```
HIP_VISIBLE_DEVICES=3 \
PYTHONPATH=$PWD/kaldifeat/python:$PWD/build/lib \
ctest --test-dir build --output-on-failure
# 100% tests passed, 12 tests passed out of 12
#  1: Test.feature-window-test -- Passed
#  2: Test.online-feature-test -- Passed
#  3: test_fbank_py -- Passed
#  4: test_fbank_options_py -- Passed
#  5: test_frame_extraction_options_py -- Passed
#  6: test_mel_bank_options_py -- Passed
#  7: test_mfcc_py -- Passed
#  8: test_mfcc_options_py -- Passed
#  9: test_plp_py -- Passed
# 10: test_plp_options_py -- Passed
# 11: test_spectrogram_py -- Passed
# 12: test_spectrogram_options_py -- Passed

HIP_VISIBLE_DEVICES=3 PYTHONPATH=$PWD/kaldifeat/python:$PWD/build/lib \
  $PYI kaldifeat/python/tests/test_whisper_fbank.py    # PASS
HIP_VISIBLE_DEVICES=3 PYTHONPATH=$PWD/kaldifeat/python:$PWD/build/lib \
  $PYI kaldifeat/python/tests/test_whisper_v3_fbank.py  # PASS
```
Total: 14/14 pass (12 ctest + 2 whisper).

GPU proof: torch.cuda.get_device_name(0) = "AMD Radeon Pro W7800 48GB", torch.version.hip = 7.2.53211, kaldifeat.Fbank on cuda:0 allocated ~34 MB GPU memory and returned a tensor on cuda:0. Not a CPU fallback.

No CUDA no-regression gate required (follower platform, zero .cu/.cuh device code).

## Validation 2026-06-12 (windows-gfx1201, RX 9070 XT, RDNA4)

Platform: windows-gfx1201 (AMD Radeon RX 9070 XT, gfx1201 / RDNA4, wave32)
Fork: AMD-Ecosystem/kaldifeat @ moat-port, HEAD aa90013 (includes Windows build fixes)
GPU: HIP_VISIBLE_DEVICES=1 (RX 9070 XT, gfx1201 -- gfx1101 is present but wedged; 1=gfx1201 confirmed via hipInfo)
Result: PASSED

### Windows build fixes applied (committed as aa90013 on top of 8d3c066)

kaldifeat requires three Windows-specific cmake fixes to build with clang (non-MSVC)
against the TheRock ROCm PyTorch wheel (torch 2.9.1+rocm7.14):

1. CMakeLists.txt: `CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS` was gated on `MSVC` but clang on
   Windows has `MSVC=false`, so kaldifeat symbols were not exported from the DLL. Changed
   guard to `WIN32`. Also changed the `/wd4624` warning block from `if(WIN32)` to
   `if(MSVC)` -- MSVC-only slash flags are rejected by clang.

2. cmake/torch.cmake: The ROCm PyTorch wheel's `Caffe2Targets.cmake` bakes MSVC-only
   options (`/permissive-`, `/EHsc`, `/bigobj`) into the imported-target
   `INTERFACE_COMPILE_OPTIONS` via generator expressions. Clang (GCC-compat mode) rejects
   slash flags as unknown file arguments. Fixed by detecting `WIN32 AND NOT MSVC`, stripping
   slash flags from `TORCH_CXX_FLAGS`, and clearing `INTERFACE_COMPILE_OPTIONS` on all
   imported torch/caffe2 targets (non-slash flags come from TORCH_CXX_FLAGS and survive
   via CMAKE_CXX_FLAGS).

3. kaldifeat/python/tests/CMakeLists.txt: ctest's `ENVIRONMENT` property uses semicolons to
   separate `VAR=value` pairs. On Windows, Python uses `;` as the `PYTHONPATH` separator
   (not `:` as on Unix). Fixed the separator to be conditional on `WIN32`. Also added
   `PATH` to the test environment on Windows so that `kaldifeat_core.dll` and torch's
   `c10.dll`/`torch_cpu.dll` are findable when Python loads `_kaldifeat.pyd`.

These changes are all guarded on `WIN32`/`WIN32 AND NOT MSVC`/`MSVC` and are inert on
Linux and CUDA builds. Linux platforms (gfx90a, gfx1100) carried forward to aa90013 via
source-class.

### Build commands

```
VENV=B:/develop/TheRock/external-builds/pytorch/.venv
PYI=$VENV/Scripts/python.exe
ROCM=$VENV/Lib/site-packages/_rocm_sdk_devel
SRC=B:/develop/moat/projects/kaldifeat/src

ROCM_PATH=$ROCM HIP_VISIBLE_DEVICES=1 PYTORCH_ROCM_ARCH=gfx1201 \
  cmake -S $SRC -B $SRC/build \
  -G Ninja \
  -DPYTHON_EXECUTABLE=$PYI \
  -Dkaldifeat_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/clang.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_PREFIX_PATH="$VENV/Lib/site-packages/torch/share/cmake;$ROCM/lib/cmake" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5

HIP_VISIBLE_DEVICES=1 PYTORCH_ROCM_ARCH=gfx1201 cmake --build $SRC/build -j64
```

Configure confirms:
- "Building PyTorch for GPU arch: gfx1201"
- "PyTorch is a ROCm build (HIP 7.14.60850-d34cbb64)"
- "HIP runtime library: .../amdhip64.lib"

### Test results

Run environment (set once, used for all tests):
```
export PATH="$SRC/build/bin:$VENV/Lib/site-packages/torch/lib:$VENV/Lib/site-packages/_rocm_sdk_core/bin:$PATH"
export PYTHONPATH="$SRC/kaldifeat/python;$SRC/build/lib"
export HIP_VISIBLE_DEVICES=1
```

C++ gtests (working directory torch/lib per cmake config):
```
ctest --test-dir $SRC/build -R "feature-window-test|online-feature-test" --output-on-failure
# 1/2 Test.feature-window-test -- Passed
# 2/2 Test.online-feature-test -- Passed
```

Python tests (run directly due to ctest ENVIRONMENT semicolon escaping limitation on Windows):
```
python $SRC/kaldifeat/python/tests/test_fbank.py                    # PASS
python $SRC/kaldifeat/python/tests/test_fbank_options.py            # PASS
python $SRC/kaldifeat/python/tests/test_frame_extraction_options.py # PASS
python $SRC/kaldifeat/python/tests/test_mel_bank_options.py         # PASS
python $SRC/kaldifeat/python/tests/test_mfcc.py                     # PASS
python $SRC/kaldifeat/python/tests/test_mfcc_options.py             # PASS
python $SRC/kaldifeat/python/tests/test_plp.py                      # PASS
python $SRC/kaldifeat/python/tests/test_plp_options.py              # PASS
python $SRC/kaldifeat/python/tests/test_spectrogram.py              # PASS
python $SRC/kaldifeat/python/tests/test_spectrogram_options.py      # PASS
python $SRC/kaldifeat/python/tests/test_whisper_fbank.py            # PASS
python $SRC/kaldifeat/python/tests/test_whisper_v3_fbank.py         # PASS
```

Total: 14/14 pass (2 C++ gtests + 10 Python feature tests + 2 whisper tests).

GPU proof:
- torch.__version__ = 2.9.1+rocm7.14.0a20260604
- torch.version.hip = 7.14.60850-d34cbb64
- torch.cuda.is_available() = True
- torch.cuda.device_count() = 1 (with HIP_VISIBLE_DEVICES=1)
- torch.cuda.get_device_name(0) = "AMD Radeon RX 9070 XT"
- kaldifeat.Fbank on cuda:0 returned tensor on cuda:0, ~76 MB GPU memory allocated
- Not a CPU fallback

Note: ctest ENVIRONMENT property semicolon-in-value issue -- Windows Python uses ';'
as PYTHONPATH separator, but ctest's ENVIRONMENT list also uses ';' as item separator,
making it impossible to embed ';' in a PATH/PYTHONPATH value via cmake generator
expressions alone. The cmake fix (aa90013) provides the correct separator for builds
but the PATH embedding via $ENV{PATH} expansion still hits this limitation for the
PATH variable. The workaround for running tests is to set PATH/PYTHONPATH explicitly
in the shell before calling ctest (or run test scripts directly). The C++ tests
(working directory=torch/lib, no PYTHONPATH needed) run cleanly via ctest.

Venv torch verified intact after validation: 2.9.1+rocm7.14.0a20260604.

## Review 2026-06-12
Verdict: review-passed (build-only Strategy A port; no GPU device code; all ROCm changes guarded on kaldifeat_TORCH_IS_ROCM). Fault classes (warp size, rule-of-five, OOB reads, texture pitch, library swaps) verified N/A: 0 .cu/.cuh, 0 c10::cuda/CUDAStream/warp/cublas/cufft sites. Commit hygiene clean ([ROCm] title 57 chars, Claude named, no noreply trailer, ASCII, no MOAT jargon in diff). Docs added in both README.md and doc/source/installation/from_source.rst.

Non-blocking observations (not bounced; record for the prep phase):
- kaldifeat/csrc/CMakeLists.txt:39-40 -- `test_kaldifeat` (a build-only scratch exe with main(), never registered via add_test, never run by ctest) still links the SHARED kaldifeat_core. By the porter's own root-cause it would hit the recursive-dlopen __gnu_cxx::recursive_init_error abort if ever RUN on ROCm, but it is never run, so it does not affect validation or the build. Left for consistency consideration only; do not block on it.
- kaldifeat/csrc/CMakeLists.txt:70-84 -- under kaldifeat_BUILD_TESTS the core sources now compile twice (shared kaldifeat_core + OBJECT kaldifeat_core_obj). Minor build-time cost on every backend incl. CUDA/CPU; correctness unaffected. The unconditional kaldifeat_add_test rewrite (shared core -> obj + ${TORCH_LIBRARIES}) is behavior-equivalent on CUDA/CPU (same KALDIFEAT_TORCH_VERSION_* defs, csrc include via root include_directories, TORCH_INCLUDE_DIRS explicit).
- notes.md tail carries stray `</content></invoke>` artifact lines from an earlier write; harmless, clean up opportunistically.

## Validation 2026-06-20 (windows-gfx1101, Radeon PRO V710, RDNA3)

Platform: windows-gfx1101 (AMD Radeon PRO V710, gfx1101 / RDNA3, wave32)
Fork: AMD-Ecosystem/kaldifeat @ moat-port, HEAD aa90013
GPU: HIP_VISIBLE_DEVICES=1 (Radeon PRO V710, gfx1101 -- mapping verified: mask 0=gfx1201, mask 1=gfx1101)
Result: PASSED (optional platform, purely additive)

### Build commands

Identical to gfx1201 build recipe except separate build dir and PYTORCH_ROCM_ARCH=gfx1101:

```
VENV=B:/develop/TheRock/external-builds/pytorch/.venv
PYI=$VENV/Scripts/python.exe
ROCM=$VENV/Lib/site-packages/_rocm_sdk_devel
SRC=B:/develop/moat/projects/kaldifeat/src

ROCM_PATH=$ROCM HIP_VISIBLE_DEVICES=1 PYTORCH_ROCM_ARCH=gfx1101 \
  cmake -S $SRC -B $SRC/build_gfx1101 \
  -G Ninja \
  -DPYTHON_EXECUTABLE=$PYI \
  -Dkaldifeat_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/clang.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_PREFIX_PATH="$VENV/Lib/site-packages/torch/share/cmake;$ROCM/lib/cmake" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5

HIP_VISIBLE_DEVICES=1 PYTORCH_ROCM_ARCH=gfx1101 cmake --build $SRC/build_gfx1101 -j64
```

Configure confirms:
- "Building PyTorch for GPU arch: gfx1101"
- "PyTorch is a ROCm build (HIP 7.14.60850-d34cbb64)"
- "HIP runtime library: .../amdhip64.lib"
- Build: 42/42 steps, no errors.

### Test results

Run environment:
```
export PATH="$SRC/build_gfx1101/bin:$VENV/Lib/site-packages/torch/lib:$VENV/Lib/site-packages/_rocm_sdk_core/bin:$PATH"
export PYTHONPATH="$SRC/kaldifeat/python;$SRC/build_gfx1101/lib"
export HIP_VISIBLE_DEVICES=1
```

C++ gtests:
```
ctest --test-dir $SRC/build_gfx1101 -R "feature-window-test|online-feature-test" --output-on-failure
# 1/2 Test.feature-window-test -- Passed
# 2/2 Test.online-feature-test -- Passed
```

Python tests (run directly):
```
python $SRC/kaldifeat/python/tests/test_fbank.py                    # PASS
python $SRC/kaldifeat/python/tests/test_fbank_options.py            # PASS
python $SRC/kaldifeat/python/tests/test_frame_extraction_options.py # PASS
python $SRC/kaldifeat/python/tests/test_mel_bank_options.py         # PASS
python $SRC/kaldifeat/python/tests/test_mfcc.py                     # PASS
python $SRC/kaldifeat/python/tests/test_mfcc_options.py             # PASS
python $SRC/kaldifeat/python/tests/test_plp.py                      # PASS
python $SRC/kaldifeat/python/tests/test_plp_options.py              # PASS
python $SRC/kaldifeat/python/tests/test_spectrogram.py              # PASS
python $SRC/kaldifeat/python/tests/test_spectrogram_options.py      # PASS
python $SRC/kaldifeat/python/tests/test_whisper_fbank.py            # PASS
python $SRC/kaldifeat/python/tests/test_whisper_v3_fbank.py         # PASS
```

Total: 14/14 pass (2 C++ gtests + 10 Python feature tests + 2 whisper tests).

GPU proof:
- torch.__version__ = 2.9.1+rocm7.14.0a20260604
- torch.version.hip = 7.14.60850-d34cbb64
- torch.cuda.is_available() = True
- torch.cuda.device_count() = 1 (with HIP_VISIBLE_DEVICES=1)
- torch.cuda.get_device_name(0) = "AMD Radeon PRO V710"
- kaldifeat.Fbank on cuda:0 returned tensor on cuda:0, ~76 MB GPU memory allocated
- Not a CPU fallback

No CUDA no-regression gate (follower/optional platform, zero .cu/.cuh device code).
GPU health post-test: hipInfo on mask 1 = Radeon PRO V710, gfx1101, memFree 25.34 GB (99%) -- no TDR or wedge.
