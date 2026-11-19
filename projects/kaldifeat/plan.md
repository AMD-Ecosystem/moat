# kaldifeat porting plan (Linux gfx90a lead)

## Project
- Name: kaldifeat
- Upstream: https://github.com/csukuangfj/kaldifeat
- Default branch: master
- Fork (to be created by porter): AMD-Ecosystem/kaldifeat, branch `moat-port`
- Role in stack: Kaldi-compatible online/offline feature extraction (fbank/mfcc/plp/spectrogram/whisper) used by the Next-gen Kaldi stack (sherpa, icefall). It is the dependency that unblocks k2's deferred k2/torch decoder layer (deferred item `k2-torch-decoder-kaldifeat`); k2's build references it via cmake/kaldifeat.cmake at tag v1.25.4.

## Existing AMD support
None found. Disposition: a fresh ROCm build port that adds value.
- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* doc/` returns only Windows wheel filenames (`...win_amd64.whl`) -- the x86-64 "amd64" platform tag, not AMD-GPU support.
- No fork under ROCm/AMD/GPUOpen orgs (`gh api repos/csukuangfj/kaldifeat/forks` -> none matching rocm/amd/hip).
- No upstream ROCm/HIP/AMD PRs or branches (`gh pr list --search "ROCm OR HIP OR AMD"` -> empty).
- Web search ("kaldifeat ROCm AMD GPU HIP") finds no kaldifeat-specific AMD effort, no separately-named AMD project.
Authoritativeness: N/A (nothing exists). Proceed with a from-scratch port -- which here is almost entirely a build/CI exercise, not a code translation (see Port strategy).

## Build classification: pure CMake (Strategy A)
Evidence:
- `setup.py` line 37 uses a CUSTOM `cmake_extension("_kaldifeat")` and a CUSTOM `BuildExtension` imported from `cmake/cmake_extension.py` -- NOT `torch.utils.cpp_extension.CUDAExtension`/`BuildExtension`. `cmake/cmake_extension.py`'s `BuildExtension.build_extension` just shells out to `cmake` + `make _kaldifeat install`. So torch's build-time source-hipify machinery never runs.
- `CMakeLists.txt` line 65 `include(torch)` -> `cmake/torch.cmake` does `find_package(Torch REQUIRED)` against the installed torch's cmake dir. Standard CMake `find_package(Torch)`, the Strategy-A "CMake build that finds Torch" shape.
- `kaldifeat/csrc/CMakeLists.txt` line 17 links only `${TORCH_LIBRARIES}`; no CUDA/cuda-library targets named anywhere.
Per PORTING_GUIDE "Build classification": a CMake build that finds Torch but does not use torch's extension hipify is handled as Strategy A (build the CMake project against a ROCm torch), not Strategy B.

## Port strategy: Strategy A, build-only (no source GPU code to translate)
This is the minimal-footprint end of Strategy A: there is effectively NO CUDA surface to port.

GPU compute is 100% delegated to libtorch tensor ops. The only "device" concept in the code is a `torch::Device` value:
- Python users pass a device STRING (`"cpu"`, `"cuda"`, `"cuda:0"`) into the options object; `kaldifeat/python/csrc/*.cc` constructs `torch::Device(s)` from it (e.g. whisper-fbank.cc:29,45).
- C++ option structs default to `torch::Device{"cpu"}` (feature-fbank.h:41, feature-plp.h:58, feature-spectrogram.h:35, whisper-fbank.h:37).
- All movement/compute is generic: `tensor.to(device)`, `tensor.device()`, `torch::matmul`, FFT/`torch::stft`, etc. These dispatch to whatever backend the device maps to.

On a ROCm PyTorch, `torch.cuda.is_available()` is True, the device string stays literally `"cuda"` (PyTorch's deliberate ROCm-keeps-CUDA-names conflation), and `tensor.to("cuda")` runs on the AMD GPU via the HIP backend automatically. So the existing code already runs on AMD once linked against a ROCm torch -- no code edit is required for GPU functionality.

### TORCH_HIPIFY_V2 handling: NOT needed
The hipify v1/v2 c10-namespace axis (PORTING_GUIDE lines 194-198) only bites code that names `c10::cuda::...` / `c10::hip::...`, `at::cuda::getCurrentCUDAStream`, `CUDAStream`, or a hand-rolled stream shim in a NON-hipified TU. kaldifeat has ZERO such sites (grep for `c10::cuda|at::cuda|CUDAStream|getCurrentCUDAStream|getCurrentHIPStream` over `kaldifeat/csrc` and `kaldifeat/python/csrc` returns nothing). It never touches a CUDA/HIP stream, never includes a c10 cuda header. There is nothing to key on `torch.utils.hipify.__version__`, so no `-DTORCH_HIPIFY_V2` neutral define is introduced. (Recorded explicitly because the dispatching task flagged it as likely; verified it is not. The k2 fix does not transfer here because kaldifeat, unlike k2, has no c10 stream usage.)

### Why not Strategy B
Strategy B presupposes torch's `CUDAExtension` source-hipify; kaldifeat does not use it. Forcing torch hipify here would be wrong and unnecessary -- there are no `.cu`/`.cuh` to hipify (count = 0) and no CUDA-spelled symbols in `.cc` to rename.

## CUDA surface inventory
- Hand-written CUDA kernels (`.cu`/`.cuh`): 0 (confirmed `find . -name '*.cu' -o -name '*.cuh'` = 0).
- `__global__`/`__device__`, warp intrinsics (`__shfl*`, `__ballot`, `warpSize`, hardcoded 32): none.
- Textures/surfaces: none.
- cuBLAS/cuFFT/cuRAND/cuSPARSE/Thrust/CUB/cuDNN: none directly (any GEMM/FFT is inside libtorch, which on ROCm already uses hipBLAS/rocBLAS, hipFFT/rocFFT, MIOpen).
- `c10::cuda` / `at::cuda` / `CUDAStream` / `cudaXxx` runtime calls: none.
- Pinned/managed memory, streams, events: none.
- Device handling: `torch::Device` values only, constructed from user strings; pure libtorch tensor ops downstream.
Net: the porting surface is the BUILD ENVIRONMENT (link against ROCm torch) plus CI/docs, not the source.

## Risk list
The usual GPU fault classes (warp size 32-vs-64, rule-of-five on resource handles, OOB neighbor reads, 256B texture pitch, library swaps) DO NOT APPLY -- there is no device code in this repo; those risks live inside libtorch, which is already ROCm-validated upstream. Project-specific risks:
- CPU-only fallbacks in the algorithm: `mel-computations.cc` Durbin/LPC (lines ~290, 337-339) explicitly run on CPU and copy back to the saved device (`TODO: support cuda`). These already do `.to(cpu)` then `.to(saved_device)`, so they are correct on any device including ROCm "cuda"; PLP just incurs a host round-trip. Not a port blocker; note it so a PLP-on-GPU test that expects no host transfer is not misread.
- Bit-exactness: tests use `torch.allclose` against precomputed CPU reference text files (`test_data/*.txt`), default tolerances -- not zero-tolerance. fbank/mfcc/spectrogram GPU vs CPU should pass within allclose; watch only if a specific test tightens tolerance. No bit-exact gold compare that would trip the ULP-drift fault classes.
- Multi-GPU host: this box has 4 GPUs, so `get_devices()` returns `cpu`, `cuda:0`, AND `cuda:1`; the GPU tests will exercise both indices. That is fine (more coverage) but means a single-GPU assumption is not in play here.
- `find_package(Torch)` must resolve the ROCm torch's caffe2 targets cleanly. Expected to work (it is the standard torch cmake package), but the porter must configure with the SAME interpreter (`/opt/conda/envs/py_3.12/bin/python`) so `torch.cmake`'s `execute_process` picks the ROCm torch, not a stray CUDA torch.
- No source edit is expected, which means head_sha churn is purely from CI/docs/notes. Keep the fork commit minimal (CI workflow for ROCm if added, plus docs) so follower revalidation is cheap. If genuinely zero source change is needed, the port is "validated build + GPU tests green against ROCm torch"; the deliverable to upstream is then primarily CI/docs enabling the ROCm build path (frame it to the maintainer as "builds and tests pass on ROCm PyTorch unchanged").

## File-by-file change list (expected)
Source (`kaldifeat/csrc`, `kaldifeat/python/csrc`): NO changes expected -- device-agnostic libtorch code already runs on ROCm.
Build/CI/docs (the actual deliverable):
- `.github/workflows/`: leave upstream CI as-is on upstream; on the jeffdaily fork DISABLE Actions (`gh api -X PUT repos/AMD-Ecosystem/kaldifeat/actions/permissions -F enabled=false`). Do NOT add a CPU-only GHA smoketest (MOAT rule). If upstream wants a ROCm CI lane in the eventual PR, mirror `run-tests-ubuntu-cuda.yml` to a ROCm image as a SEPARATE prep-phase decision, not a fork smoketest.
- Docs: add a brief "running on AMD GPUs (ROCm)" note in the project's house style wherever the CUDA/GPU build is documented (`README.md` and `doc/source/installation/`), stating that kaldifeat builds and runs on a ROCm PyTorch unchanged and that the device string is `"cuda"` as usual. Match the project's reStructuredText doc-site style under `doc/source/`.
- `notes.md`: record the env facts (below) and the "## Install as a dependency" section (below).
If a build actually surfaces a needed source fix (e.g. an ABI flag or a torch-version guard), it goes behind a `USE_ROCM`/version guard per PORTING_GUIDE; only add it if the build demands it.

## Build commands (gfx90a)
Use the shared ROCm torch env that k2 uses: `/opt/conda/envs/py_3.12` (torch 2.13.0a0, HIP 7.2.53211, hipify 2.0.0, 4 GPUs visible).

Standalone CMake (for the test target):
```
export PYI=/opt/conda/envs/py_3.12/bin/python
cmake -S projects/kaldifeat/src -B projects/kaldifeat/src/build \
  -DPYTHON_EXECUTABLE=$PYI \
  -Dkaldifeat_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build projects/kaldifeat/src/build -j --target _kaldifeat
```
Python install (the supported path; mirrors what k2/sherpa consume):
```
cd projects/kaldifeat/src
export KALDIFEAT_MAKE_ARGS="-j"
/opt/conda/envs/py_3.12/bin/python setup.py install
```
Arch: kaldifeat compiles NO device code, so there is no `CMAKE_HIP_ARCHITECTURES` to pass and no per-arch fat binary -- the GPU arch is whatever the linked ROCm torch supports at runtime. This means the same build is arch-portable across gfx90a/gfx1100/etc with no source change (followers just need a matching ROCm torch); there is no warp-size dimension to validate. Do NOT add a `-DCMAKE_HIP_ARCHITECTURES`/`enable_language(HIP)` block -- it would be dead code here.

## Test plan
Real test suite: `kaldifeat/python/tests/` (the C++ gtests under `kaldifeat/csrc/*-test.cc` are CPU-window/online-feature unit tests).
- GPU-exercising tests (run on cpu AND cuda automatically via `utils.get_devices()`, which appends `cuda:0`/`cuda:1` when `torch.cuda.is_available()` -- True on ROCm): `test_fbank.py`, `test_mfcc.py`, `test_plp.py`, `test_spectrogram.py`, `test_fbank_options.py`, `test_mfcc_options.py`, `test_plp_options.py`, `test_spectrogram_options.py`, `test_whisper_fbank.py`, `test_whisper_v3_fbank.py`. These compare GPU output against precomputed CPU reference (`test_data/*.txt`) and against the CPU run with `torch.allclose`.
- Non-GPU regression set (must not regress): the C++ gtests `feature-window-test` and `online-feature-test` (built with `-Dkaldifeat_BUILD_TESTS=ON`, run via ctest), and the CPU branch of every python test.
Run:
```
cd projects/kaldifeat/src
export PYTHONPATH=$PWD/kaldifeat/python:$PWD/build/lib:$PYTHONPATH   # adjust to install path
for t in kaldifeat/python/tests/test_fbank.py kaldifeat/python/tests/test_mfcc.py \
         kaldifeat/python/tests/test_plp.py kaldifeat/python/tests/test_spectrogram.py \
         kaldifeat/python/tests/test_whisper_fbank.py kaldifeat/python/tests/test_whisper_v3_fbank.py; do
  /opt/conda/envs/py_3.12/bin/python $t || echo "FAIL $t"
done
```
Validation gate (per MOAT policy): GPU tests green on gfx90a real hardware (they print `device cuda:0`) AND the C++/CPU tests not regressed. A CPU-only run is NOT sufficient. The whisper tests may pull a mel-bank table or model -- the porter checks whether they need network/data; if a whisper test needs an unavailable download, scope it out explicitly and validate on the fbank/mfcc/plp/spectrogram GPU set (which is self-contained via test_data).

## Dependencies
- `depends_on`: none. kaldifeat needs only libtorch (provided by the ROCm torch env); no MOAT-project build deps. Leave `status.json.depends_on = []`.
- Reverse edge: k2/torch (k2's deferred decoder layer) consumes kaldifeat -- already tracked in the deferred registry (`k2-torch-decoder-kaldifeat`); not added here.

## Install as a dependency (porter: write this into notes.md)
k2/torch will consume `AMD-Ecosystem/kaldifeat @ moat-port`. The base-library obligation (CLAUDE.md "Project dependencies") applies: notes.md MUST carry a "## Install as a dependency" section so a downstream port can build+install kaldifeat into `_deps/kaldifeat/`. Expected recipe (porter to verify and record):
```
git clone -b moat-port https://github.com/AMD-Ecosystem/kaldifeat _deps/kaldifeat/src
cd _deps/kaldifeat/src
KALDIFEAT_MAKE_ARGS="-j" /opt/conda/envs/py_3.12/bin/python setup.py install \
  --prefix _deps/kaldifeat/install   # or pip install . into the env
```
k2 references kaldifeat through `cmake/kaldifeat.cmake` (FetchContent at tag v1.25.4); when wiring k2/torch, point that at the installed jeffdaily build / `kaldifeatConfig.cmake` rather than re-fetching upstream. Record the exact verified commands in notes.md.

## Open questions
- Do the whisper fbank tests require a network download or only the in-repo mel-bank headers (`whisper-mel-bank.h`, `whisper-v3-mel-bank.h` are vendored)? If self-contained, include them in the GPU validation set; if not, scope them out and note it.
- Does the eventual upstream PR want a ROCm CI lane (a ROCm mirror of `run-tests-ubuntu-cuda.yml`)? Decide in the prep phase; not required for MOAT validation.
- Confirm `find_package(Torch)` against ROCm torch 2.13 configures without pulling a CUDA caffe2 target; if it errors, that (not the source) is the only place a build fix could be needed.
