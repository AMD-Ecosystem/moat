# keops port plan

Planner: linux-gfx90a, 2026-11-20. Upstream analysed at `ef72bc907df7` (main, "Merge pull request #446 from getkeops/test_fp16"). The intake section of `notes.md` has the binder inventory this plan builds on; it is not repeated here except where the plan depends on it.

## Project

- Name: keops (KeOps: keopscore code generator + pykeops Python front end + rkeops R front end)
- Upstream: https://github.com/getkeops/keops, default branch `main`
- Fork: https://github.com/AMD-Ecosystem/keops, port branch `moat-port`
- Delivery: one upstream PR (ruled at intake). Windows gate waived (approved 2026-11-12; upstream supports Linux and macOS only).
- Scope: the whole project. Nothing is deferred: there is one GPU backend (the NVRTC binder) and both front ends reach the GPU only through it.

## Existing AMD support

None. Intake found no issue, PR, branch, fork or doc mention of ROCm/HIP/AMD on getkeops/keops, nor any AMD-owned or community port. There is nothing to inherit or to judge for authoritativeness. Upstream links no platform forks; an upstream PR is the right vehicle.

Port vs rewrite: KeOps' kernels are generic map-reduce loops (one thread per output row, tiles of the other variable staged in dynamic shared memory, sequential per-thread accumulation). There is no CUTLASS, no tensor-core or warp-level code. A mechanical port is the right answer; no AMD-native rewrite is planned or wanted.

## Build classification

Neither CMake nor a torch extension: a **runtime-compiled (NVRTC + CUDA driver API) Python package**, the third build class (`cuda-to-rocm` platform-limits.md, "NVRTC (CUDA C++ generated at run time) ports to hipRTC"). Recorded as `ext_type = nvrtc-driver-api`.

Evidence:
- No `.cu` file in the tree. `keopscore/setup.py` and `pykeops/setup.py` are pure setuptools packages that ship `binders/nvrtc/*.cpp` and `pykeops_nvrtc.cpp` as package data; nothing is compiled at install time.
- At first use, `Gpu_link_compile.compile_jit_compile_dll` (`keopscore/binders/nvrtc/Gpu_link_compile.py:105`) builds `nvrtc_jit.so` from `nvrtc_jit.cpp` with the host C++ compiler (`g++ -std=c++11 -O3 -flto=auto -fpermissive`, `config/CxxCompiler.py:129`) plus `cuda.get_preprocessing_options()`, `cuda.get_include_options()`, `cuda.get_linking_options()` (`config/Cuda.py:583-650`). `pykeops/common/keops_io/nvrtc/LoadKeOps_nvrtc.py:78` builds the pybind11 module `pykeops_nvrtc.so`, which `#include`s `binders/nvrtc/keops_nvrtc.cpp`.
- Per formula, Python (`keopscore/mapreduce/gpu/*.py`) emits CUDA C++ source, `nvrtc_jit.so:Compile` runs NVRTC and writes a CUBIN/PTX file, and `keops_nvrtc.cpp` loads it with `cuModuleLoadDataEx` and launches it with `cuLaunchKernel`.

So no hipcc is involved at any point: the port is a host-compiler build against HIP headers plus hipRTC at run time.

## Port strategy

Strategy A in shape, by the shim-header method, keeping the CUDA path byte-identical:

1. **Shim headers, HIP include path only.** Add a directory (proposed `keopscore/keopscore/binders/nvrtc/hip/`, name it so that it cannot shadow ROCm's own `hip/` include tree when added with `-I`, e.g. `binders/nvrtc/hip_shim/`) holding `cuda.h`, `nvrtc.h` and `cuda_fp16.h`. Each one includes the HIP header (`hip/hip_runtime_api.h`, `hip/hiprtc.h`, `hip/hip_fp16.h`) and maps the CUDA driver/NVRTC names KeOps uses to HIP: types (`CUresult`, `CUdevice`, `CUcontext`, `CUmodule`, `CUfunction`, `CUdeviceptr`, `nvrtcProgram`, `nvrtcResult`), constants (`CUDA_SUCCESS`, `NVRTC_SUCCESS`, `NVRTC_ERROR_INVALID_OPTION`, `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR/MINOR`) and functions (full list in the surface inventory below). The host compile gets `-D__HIP_PLATFORM_AMD__ -I<shim> -I<rocm>/include`; the CUDA compile never sees the directory. `keops_nvrtc.cpp`, `utils_pe.h`, `CudaSizes.h` and `pykeops_nvrtc.cpp` then stay unchanged.
2. **`nvrtc_jit.cpp`: one guarded block.** The arch flag cannot be derived through the shim: compute capability means nothing on AMD (and collides with NVIDIA numbers, platform-limits.md). Under `#ifdef __HIP_PLATFORM_AMD__` read `gcnArchName` with `hipGetDeviceProperties` and pass `--offload-arch=<gcnArchName>` (keep the full name, e.g. `gfx90a:sramecc+:xnack-`); skip the `-arch=` retry path. Map `use_fast_math` to no option on HIP (Risk R3). The extraction macros become `nvrtcGetTARGET=nvrtcGetCUBIN` mapped by the shim to `hiprtcGetCode`/`hiprtcGetCodeSize`.
3. **`config/`: a HIP backend behind the existing `config.cuda` object.** Every caller (pykeops `__init__`, `Gpu_link_compile`, `KeOpsPath`, `mapreduce/__init__`, `Cache.py`, and every GPU test via `pykeops.config.cuda.is_available()`) reads `keopscore.config.cuda`. Keep that name and interface. Add `config/Hip.py` with a `HipConfig` (subclass of `CudaConfig` or a sibling with the same public methods) that discovers `libamdhip64.so*`, `libhiprtc.so*` (and `libhiprtc-builtins.so*` beside it), `hip/hip_runtime_api.h`, `hip/hiprtc.h`; initialises with `hipInit`, `hipDriverGetVersion`, `hipGetDeviceCount`; reads per-device limits with `hipDeviceGetAttribute` using **HIP's enum values** (`hipDeviceAttributeMaxThreadsPerBlock` = 56, `hipDeviceAttributeMaxSharedMemoryPerBlock` = 74 in HIP 7.16 headers; CUDA's 1 and 8 are wrong on HIP); emits `-D__HIP_PLATFORM_AMD__`, the shim and ROCm include dirs, `-lamdhip64 -lhiprtc` (by path, plus `-Wl,-rpath`), the same `-DMAXIDGPU/-DMAXTHREADSPERBLOCKn/-DSHAREDMEMPERBLOCKn` flags, and returns the shim dir as the hipRTC include path. `config/__init__.py` selects: CUDA first exactly as today; HIP only if CUDA is not found (or an explicit env override, e.g. `KEOPS_GPU_BACKEND=hip|cuda`, if the porter wants one). The NVIDIA code path through `CudaConfig` is unchanged.
4. **Generated code: one shared guard.** `keopscore/utils/math_functions.py:49` emits `#ifdef __CUDACC__` around the GPU spelling of every math function. hipRTC does not define `__CUDACC__` (it defines `__HIPCC__` and `__HIPCC_RTC__`), so on HIP every function would silently take the CPU branch (Risk R1). Change it to `#if defined(__CUDACC__) || defined(__HIPCC__)`. Do not define `__CUDACC__` on the hipRTC command line: it would also reach the HIP builtin headers.
5. **User-facing text.** `check_health`/`print_all` and the "no properly configured cuda lib" messages should name the backend found (HIP/ROCm) rather than claiming CUDA; `doc/python/installation.rst` gets a short ROCm section (requirements: a ROCm install or the ROCm Python SDK providing `libamdhip64`, `libhiprtc` and the HIP headers; a ROCm build of PyTorch for the torch front end). No README device-selection env vars in example commands.

Rejected alternatives: a sibling `binders/hiprtc/` copy of the 650-line `keops_nvrtc.cpp` (duplicate maintenance for upstream, and every later upstream edit lands NVIDIA-only); `#ifdef` blocks throughout `keops_nvrtc.cpp` (larger upstream-visible diff for no gain over the shim). If the porter finds a call the shim cannot express cleanly, a small guarded block in the `.cpp` is acceptable; say why in the commit.

## CUDA surface inventory

All in host C++ compiled by the host compiler, plus generated device code compiled by NVRTC.

| CUDA use | Where | HIP equivalent | Note |
|---|---|---|---|
| `cuInit`, `cuDeviceGet`, `cuDeviceGetAttribute` | `nvrtc_jit.cpp`, `keops_nvrtc.cpp`, `config/Cuda.py` (ctypes) | `hipInit`, `hipDeviceGet`, `hipDeviceGetAttribute` | attribute enum values differ (R4) |
| `cuDevicePrimaryCtxRetain/Release`, `cuCtxGetCurrent`, `cuCtxPushCurrent`, `cuCtxSynchronize` | `keops_nvrtc.cpp` | `hipDevicePrimaryCtxRetain/Release`, `hipCtxGetCurrent`, `hipCtxPushCurrent`, `hipCtxSynchronize` (or `hipDeviceSynchronize`) | HIP ctx API deprecated but present in HIP 7.16; fallback is `hipSetDevice` (R6) |
| `cuMemAlloc`, `cuMemFree`, `cuMemcpyHtoD`, `cuMemcpyDtoH` | `keops_nvrtc.cpp`, `utils_pe.h` | `hipMalloc`, `hipFree`, `hipMemcpyHtoD`, `hipMemcpyDtoH` | signatures match (`const void*` src in HIP 7.16) |
| `cuModuleLoadDataEx`, `cuModuleGetFunction`, `cuModuleUnload`, `cuLaunchKernel` (4 sites, NULL stream, dynamic LDS) | `keops_nvrtc.cpp` | `hipModuleLoadDataEx`, `hipModuleGetFunction`, `hipModuleUnload`, `hipModuleLaunchKernel` | kernels are `extern "C"`, no name mangling issue |
| `cuGetErrorName` | `utils_pe.h` (`CUDA_SAFE_CALL*`) | `hipDrvGetErrorName` (or `hipGetErrorName`) | |
| `nvrtcCreateProgram/CompileProgram/GetProgramLog(Size)/GetErrorString/DestroyProgram`, `nvrtcGetCUBIN(Size)`/`nvrtcGetPTX(Size)` | `nvrtc_jit.cpp`, `utils_pe.h` (`NVRTC_SAFE_CALL`) | `hiprtc*`, `hiprtcGetCode(Size)` | |
| `-arch=sm_XY` from compute capability | `nvrtc_jit.cpp:60-70` | `--offload-arch=<gcnArchName>` | guarded block (strategy step 2) |
| `-use_fast_math` | `nvrtc_jit.cpp:72`; default on in pykeops (`torch/operations.py:231`, `numpy/operations.py:59`) | no option | R3 |
| `cuda_fp16.h`, `half2` host type | `keops_nvrtc.cpp` (template `KeOps_module<half2>`), generated code in half mode (`MapReduce.py:57`, `GpuAssignZero.py:26`, `GpuReduc1D_ranges.py:72`) | `hip/hip_fp16.h` via shim (host); hipRTC has fp16 built in (device) | g++ `-std=c++11 -D__HIP_PLATFORM_AMD__` compiles `hip_runtime_api.h` + `hip_fp16.h` + `hiprtc.h` (checked on this host, HIP 7.16) |
| fp16 device intrinsics: `__float2half2_rn`, `__floats2half2_rn`, `__float22half2_rn`, `__half22float2`, `h2exp/log/sin/cos/sqrt/rsqrt/rcp/floor/rint`, `__habs2`, `__hfma2`, `__hmul2`, `__heq2/__hne2/__hlt2/__hle2/__hgt2/__hge2`, half2 `+ - * /` | `utils/math_functions.py`, formulas | all declared in HIP 7.16 `amd_hip_fp16.h` (checked) | half2 operators need `__HIP_NO_HALF_OPERATORS__` unset, which is hipRTC's default |
| `__global__`, `extern __shared__`, `__syncthreads`, `blockIdx/blockDim/threadIdx` | generated by `mapreduce/gpu/*.py` (9 kernels: `GpuConv1DOnDevice`, `_ranges`, chunk/finalchunk variants, `GpuConv2DOnDevice`, `reduce2D`, `AssignZero`) | identical in HIP | |
| `#ifdef __CUDACC__` GPU/CPU split | `utils/math_functions.py:49` | add `__HIPCC__` | R1 |
| float math: `exp, log, sin, cos, sqrt, rsqrt, powf, fma, floor, round, atan2, acos, asin, atan, abs, fminf`, `1.0f/0.0f` infinities | formulas | HIP device math | infinities need IEEE semantics (R3) |
| Device discovery by ctypes: `libcuda.so.*`, `libnvrtc.so.*`, `libnvrtc-builtins.so*`, headers, pip `nvidia/*` packages, `CUDA_VISIBLE_DEVICES` | `config/Cuda.py` | `HipConfig` (strategy step 3) | R5 |

Not present (checked by grep over `keopscore/`, `pykeops/`): warp shuffles, ballots, `warpSize`, lane masks, atomics, textures, streams/events other than the NULL stream, pinned or managed memory, cuBLAS/cuFFT/cuRAND/cuSPARSE/Thrust/CUB, cooperative groups, `__CUDA_ARCH__`, `CUDART_VERSION`.

## Risk list

- **R1 (high, silent): generated math takes the CPU branch under hipRTC.** `#ifdef __CUDACC__` is false in hipRTC. Float formulas would still compile and mostly agree (the CPU spellings are valid device code), so most float tests would pass while `rsqrt` becomes `1.0f/sqrt`, `powf` becomes `pow`, and every half2 formula fails to compile (`exp(half2)`). Fix in strategy step 4. Validator: confirm `test_float16.py` runs on the GPU and passes, and spot-check one generated `.cu` in the build folder for the GPU spelling.
- **R2 (high, silent): GPU tests fall back to CPU.** Every GPU test gates on `pykeops.config.cuda.is_available()`; if HIP discovery fails, the suite runs CPU-only and is green. Validator must prove the GPU path ran: `pykeops.check_health()` output naming the HIP backend and device, `*_nvrtc.*` (or renamed) code-object files in the build folder after the run, and the count of GPU-gated tests that executed rather than skipped (`pytest -rs`).
- **R3: `use_fast_math` translation.** pykeops enables it by default. clang `-ffast-math` implies `-ffinite-math-only`, and KeOps builds its Min/Max/LogSumExp reductions on `( 1.0f/0.0f )` infinities (`utils/code_gen_utils.py:265`), so `-ffast-math` would break `Max_SumShiftExp` (logsumexp) and `Min`/`Max`/`KMin` reductions. Pass nothing for it on HIP (hipRTC already contracts FMA by default, as NVRTC does). Do not pass `-ffast-math` (faults-numerics.md). Test tolerances stay unchanged.
- **R4: device-attribute enum values differ.** `config/Cuda.py:31-32` hard-codes CUDA's `1` and `8`; on HIP those are other attributes. Use 56 and 74 (HIP 7.16 `hipDeviceAttribute_t`) or `hipGetDeviceProperties` through a compiled helper. AMD reports 1024 threads and 65536 bytes of LDS per block; block size is still capped by `CUDA_BLOCK_SIZE` = 192, and the chunk-mode constant 49152 stays below 64 KiB.
- **R5: two HIP runtimes in one process.** A process can hold one `libamdhip64.so.7`. If `HipConfig` loads an SDK runtime by path and PyTorch then brings its own (download.pytorch.org wheels ship `libamdhip64` in `torch/lib`), the second load reuses the first by soname and a version mismatch shows up as "no GPU" in whichever loads second (build-cmake-link.md, "A source-built HIP library and a PyTorch wheel in one process must load the same ROCm runtime"). Discovery order: explicit `ROCM_PATH`/`HIP_PATH`; a `libamdhip64` already mapped into the process (torch imported first); the ROCm Python SDK (`_rocm_sdk_core`, `_rocm_sdk_devel` in site-packages); `torch/lib`; `/opt/rocm`. Headers come from an SDK (`_rocm_sdk_devel/include` or `<rocm>/include`), never from torch. The no-torch smoke test in `pytest.sh` must work with no torch installed.
- **R6: deprecated HIP context API.** Primary-context retain/push works on HIP 7.16 but is deprecated and warns at compile time. Map 1:1 first; if it misbehaves, have the shim stand the device in for the context and make it current with `hipSetDevice` (platform-limits.md). The g++ warnings are acceptable; do not add `-Werror`.
- **R7: host compile flags.** `-std=c++11 -flto=auto -fpermissive` with g++ against HIP headers compiles (checked here). If a different host compiler is configured, check `__HIP_PLATFORM_AMD__` is defined before any HIP header (build-headers-macros.md).
- **R8: wavefront width.** No lane-level code, so no wave32/wave64 correctness exposure. Block sizes from the shared-memory bound may not be multiples of 64 (or 32): efficiency only. Validate on both widths anyway.
- **R9: uninitialised device memory.** Output buffers in host mode are fresh `hipMalloc`; the generated kernels assign every output element (`GpuAssignZero` for empty reductions, `reduce2D` writes all `nx*dimred`). No read-modify-write of fresh memory found; the validator's numeric tests cover it.
- **R10: JIT compile time.** hipRTC is clang and is slower per formula than NVRTC; the full suite compiles several hundred formulas cold. Keep the cache folder between runs and run long jobs under the fleet's 2-hour background limit (start detached, poll).
- **R11: `CUDA_VISIBLE_DEVICES`.** `Cuda.py:339` treats an empty value as CPU-only. On HIP honour `HIP_VISIBLE_DEVICES` the same way (and keep `CUDA_VISIBLE_DEVICES`, which HIP also honours).

## File-by-file change list

Fork branch `moat-port`, all additive or guarded:

- `keopscore/keopscore/binders/nvrtc/hip_shim/cuda.h`, `nvrtc.h`, `cuda_fp16.h` (new): the mapping headers.
- `keopscore/keopscore/binders/nvrtc/nvrtc_jit.cpp`: `__HIP_PLATFORM_AMD__` block for the arch option and fast-math option.
- `keopscore/keopscore/config/Hip.py` (new): `HipConfig`.
- `keopscore/keopscore/config/__init__.py`: backend selection, CUDA first.
- `keopscore/keopscore/config/Cuda.py`: only if needed to share helpers with `HipConfig` (prefer subclass overrides to edits).
- `keopscore/keopscore/utils/math_functions.py:49`: the `__HIPCC__` guard.
- `keopscore/keopscore/binders/nvrtc/Gpu_link_compile.py`: only if messages or the IR-file suffix need a backend branch (`get_ir_type()` drives the `.cubin`/`.ptx` suffix; `HipConfig` can return `CUBIN`, or a new `HSACO` value if the file name should say so).
- `keopscore/setup.py`, `pykeops/setup.py`: package data for the new shim headers (the `.cpp` sources are listed explicitly at `keopscore/setup.py:72`; the headers need the same).
- `pykeops/pykeops/__init__.py`, `pykeops/pykeops/config.py`: message wording only if they say CUDA where the backend is HIP.
- `doc/python/installation.rst`: ROCm section.
- `rkeops/`: no code change expected (it calls pykeops through reticulate); optionally the `rkeops_use_gpu` doc wording.

Unchanged: `keops_nvrtc.cpp`, `utils_pe.h`, `CudaSizes.h`, `pykeops_nvrtc.cpp`, every `mapreduce/` and `formulas/` file, the CPU binder.

## Build commands

Nothing is built ahead of time; the build is the first import.

```
python -m venv --system-site-packages <venv>     # inherits a ROCm PyTorch
<venv>/bin/pip install -e keopscore -e "pykeops[test]" --no-deps
<venv>/bin/pip install pytest
cd /tmp && <venv>/bin/python -c "import pykeops; pykeops.clean_pykeops(); pykeops.check_health()"
```

`check_health` builds `nvrtc_jit.so` and `pykeops_nvrtc.so` in the cache folder and must report the HIP backend, the device(s) and the per-device limits. With no `/opt/rocm` (pip ROCm SDK hosts), discovery must find `_rocm_sdk_core`/`_rocm_sdk_devel`, or set `ROCM_PATH` to the SDK root.

## Test plan

Real GPU gate, per platform (wave64 and wave32; Windows waived):

1. No-torch smoke test, as `pytest.sh` does it: a venv without torch, `pip install -e keopscore -e pykeops`, then `import pykeops; pykeops.check_health(); assert pykeops.test_numpy_bindings()`. Proves discovery and the numpy GPU path without PyTorch.
2. Full suite with ROCm PyTorch: `pytest -v -rs keopscore/keopscore/test/` and `pytest -v -rs pykeops/pykeops/test/` from the source tree (these are the two suites `pytest.sh` runs). `pytest.sh` itself creates a fresh venv and pip-installs torch from PyPI (a CUDA build), so on ROCm run its steps by hand in a venv that has a ROCm torch, or point pip at a ROCm torch index; record which.
3. Prove the GPU ran (R2): health-check output, code-object files in the cache folder, zero GPU tests skipped for "no GPU", and `test_float16.py` passing (R1). Report pass/fail/skip counts per file.
4. Tolerances unchanged. Any numeric failure is diagnosed (faults-numerics.md), not loosened; a ROCm-only tolerance bump needs a demonstrated hardware cause.
5. rkeops on one Linux platform: install R and the package's test dependencies in the job's own directory, point it at the same pykeops venv, run `rkeops/tests/testthat` with `rkeops_use_gpu()`; record results. This is evidence that the R front end inherits the port; the Python suites above are the gate.

Non-GPU regression set: the same suites on a CPU-only run (`HIP_VISIBLE_DEVICES=` / `CUDA_VISIBLE_DEVICES=` empty) pass as on upstream, `test_openmp.py` included, and `black --check` (upstream's `black.yml`) passes on changed Python files.

CUDA no-regression: the CUDA path must stay byte-identical except `math_functions.py` and `config/__init__.py`. On a host with a CUDA toolkit, build `nvrtc_jit.so` and `pykeops_nvrtc.so` through `CudaConfig` (that compile needs only the toolkit headers and libraries, no GPU) and confirm `CudaConfig` is still selected first. A run of `pytest.sh` on an NVIDIA GPU is supporting evidence when a host has one.

Benchmarks (`benchmarks/`, `pykeops/pykeops/benchmarks/`) are not a gate; one GPU vs CPU timing from `benchmarks/PyTorch_GPU.py` in notes is useful context.

## Open questions

None needs a person's ruling. The port covers the whole project (keopscore, pykeops, rkeops), delivery (upstream PR) and the Windows waiver are already ruled. Engineering choices left to the porter, with the default stated above: shim directory name; whether to offer a `KEOPS_GPU_BACKEND` override; whether the code-object file suffix says `hsaco`; context API 1:1 vs `hipSetDevice` (R6).
