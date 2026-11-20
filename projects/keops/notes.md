# keops notes

## Intake screen (2026-11-12)

Claimed for intake on a person's ruling of 2026-11-12: port, delivered as an upstream PR.

Upstream: getkeops/keops, default branch `main`, screened at ef72bc907df78cd18bff866d3dc7cb6375f11a4d (2026-06-24). Not archived; ~1.2k stars; last release v2.3 (2025-04-18).

### Licence

- `python3 utils/licenses.py check getkeops/keops` -> `license=MIT tier=1`, cleared to contribute.
- Read the files: `licence.txt`, `keopscore/keopscore/licence.txt`, `pykeops/licence.txt`, `pykeops/pykeops/licence.txt` are MIT (Charlier, Feydy, Glaunes, Durif, Collin, Vernay, Serre-Combe). `rkeops/LICENSE` is the CRAN MIT stub (YEAR/COPYRIGHT HOLDER).
- No submodules (`.gitmodules` absent), no vendored third-party trees.
- `python3 utils/licenses.py scan-nvidia agent_space/keops-screen` -> no NVIDIA proprietary licence text.

### Duplicate effort

- GitHub issue/PR search on getkeops/keops for `rocm`, `amd`, `hip`, `hiprtc`, `radeon`, `opencl`: zero hits (title, body and comments).
- No upstream branch matching hip/rocm/amd. No fork of getkeops/keops owned by an AMD/ROCm account; no keops repo in AMD-Ecosystem or ROCm; repository search for keops + rocm/hip/amd: nothing.
- Docs grep: no AMD/ROCm/HIP mention.
- Duplicate: none.

### Viability

- Genuine GPU project, but no `.cu` files in the tree: kernels are generated as CUDA C++ strings by `keopscore/mapreduce/gpu/*.py` and compiled at runtime with NVRTC. Only GPU binder: `keopscore/keopscore/binders/nvrtc/`.
- No MOAT dependencies (`depends_on` empty). Runtime needs only the driver API + runtime compiler (on ROCm: libamdhip64 + libhiprtc) and a host C++ compiler; PyTorch is optional (numpy path exists). Downstream: GPyTorch, GeomLoss.
- Upstream CI: `linux_cuda_test.yml` (self-hosted, `bash ./pytest.sh`), `linux_cpu_test.yml`, `macOs_test.yml`, `black.yml`.

## hiprtc port assessment

### Binder layout and every NVRTC / driver-API call

- `keopscore/binders/nvrtc/Gpu_link_compile.py`: compiles the "JIT engine" shared library once per cache folder with the host C++ compiler (`get_compile_command`: cxx + `cuda.get_preprocessing_options()` + include/link options for cuda/nvrtc), then loads it with `ctypes.CDLL` and calls its `Compile(target_file, cu_code, use_half, use_fast_math, device_id, include_paths)` for each formula. Writes `<hash>_nvrtc.{ptx,cubin}`.
- `keopscore/binders/nvrtc/nvrtc_jit.cpp` (146 lines): `cuDeviceGet`, `cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR/MINOR)` -> `-arch=sm_XY`/`compute_XY`; `--include-path=`; `-use_fast_math`; `nvrtcCreateProgram/CompileProgram/GetProgramLog(Size)/GetErrorString/DestroyProgram`; `nvrtcGetTARGET(Size)` macro-selected to `nvrtcGetPTX` or `nvrtcGetCUBIN`. Retries without `-arch` on `NVRTC_ERROR_INVALID_OPTION`.
- `keopscore/binders/nvrtc/keops_nvrtc.cpp` (649 lines): includes `<cuda.h> <nvrtc.h> <cuda_fp16.h>`; memory `cuMemAlloc/cuMemFree/cuMemcpyHtoD/cuMemcpyDtoH`; context `cuInit`, `cuDeviceGet`, `cuDevicePrimaryCtxRetain/Release`, `cuCtxGetCurrent`, `cuCtxPushCurrent`, `cuCtxSynchronize`; module `cuModuleLoadDataEx`, `cuModuleGetFunction` (`GpuConv1DOnDevice`, `GpuConv1DOnDevice_ranges`, `GpuConv2DOnDevice`, `reduce2D`), `cuModuleUnload`; `cuLaunchKernel` (4 sites, dynamic shared memory sized `blockSize_x * dimY * sizeof(TYPE)`). Block size bounded by `CUDA_BLOCK_SIZE` (192), `maxThreadsPerBlock`, `sharedMemPerBlock`; chunk mode hard-codes 1024 / 49152.
- `pykeops/common/keops_io/nvrtc/pykeops_nvrtc.cpp`: pybind11 wrapper that `#include <binders/nvrtc/keops_nvrtc.cpp>`; inherits the port automatically.
- `keopscore/config/Cuda.py` (754 lines): discovery by ctypes of `libcuda.so.*`, `libnvrtc.so.*`, `libnvrtc-builtins.so*`, `nvrtc.h`, `cuda.h` (system paths and pip `nvidia/cuda_nvrtc`), `cuInit`, `cuDriverGetVersion`, `cuDeviceGetCount`, per-device `MAX_THREADS_PER_BLOCK`/`MAX_SHARED_MEMORY_PER_BLOCK` baked in as `-DMAXTHREADSPERBLOCKn/-DSHAREDMEMPERBLOCKn`; chooses PTX vs CUBIN by version. `CUDA_VISIBLE_DEVICES` is read for device listing.
- `include/CudaSizes.h`: `CUDA_SAFE_CALL`/`NVRTC_SAFE_CALL` macros and per-device size globals.

Shape of the port: every API above has a 1:1 HIP/hiprtc equivalent (`hipModuleLoadData`, `hipModuleLaunchKernel`, `hipDevicePrimaryCtxRetain`, `hipCtx*` deprecated but present, `hiprtc*`, `hiprtcGetCode` replaces PTX/CUBIN). Arch flag becomes `--offload-arch=<gcnArchName>` from `hipDeviceProp_t.gcnArchName` (not compute capability). Cleanest design: a sibling `binders/hiprtc/` (or a thin header mapping `cu*`/`nvrtc*` to HIP under `USE_HIP`) plus a `config/Hip.py` (or a backend switch in `Cuda.py`) discovering `libamdhip64.so`/`libhiprtc.so` under ROCM_PATH, with `HIP_VISIBLE_DEVICES`. Keep the NVRTC path byte-identical.

### Warp-size / wave64 risk in generated code

Low. Grep of `keopscore/` (formulas, mapreduce, utils) finds no `__shfl*`, `__syncwarp`, `__ballot`, `__activemask`, `warpSize`, lane math or atomics. Reductions are per-thread: each thread owns one `i` and loops sequentially over `j` tiles staged in dynamic shared memory between `__syncthreads()`; `reduce2D` is a per-column serial pass. Default block 192 is a multiple of 64; the shared-memory bound can yield a block size that is not a multiple of 64 (efficiency only, not correctness). Shared memory: AMD reports 64 KiB/block, above the 48 KiB chunk-mode constant, so no overflow. fp16 path uses `half2` intrinsics (`__float2half2_rn`, `h2exp`, `h2log`, `h2sin`, `h2rsqrt`, `h2rint`, `__hlt2/__hgt2/__heq2`) and `#include <cuda_fp16.h>` in generated code: needs `hip/hip_fp16.h` and a check that each intrinsic exists in hiprtc's fp16 header. Watch `-use_fast_math` -> `-ffast-math` and fp tolerances in tests.

### Non-JIT C++, pykeops, rkeops

- `binders/cpp/` is the CPU (OpenMP) binder; untouched.
- pykeops: GPU path is the shared `keops_nvrtc.cpp` via pybind11; only config/discovery and `show_cuda_status`/`is_available` wording need a HIP branch. torch tensors are passed as raw device pointers, so ROCm PyTorch (`torch.cuda` on HIP) works unchanged.
- rkeops: R front end calls pykeops via reticulate; no own GPU code. Inherits the port; no separate work beyond `CUDA_VISIBLE_DEVICES` handling.
- No standalone C++/CMake/nvcc build remains in v2.x (old KeOps v1 CMake path is gone).

### Runtime-compilation headers under ROCm

Generated code includes only `cuda_fp16.h` (fp16 path) plus KeOps' own headers passed via `--include-path`. hiprtc ships its own builtin headers (no `hip_runtime.h` include needed); `hip/hip_fp16.h` needs `-I$ROCM_PATH/include`. Discovery today is Linux `.so` globbing; ROCm needs `libhiprtc.so*`, `libamdhip64.so*`, header `hip/hiprtc.h`. On Windows ROCm (TheRock) the names would be `amdhip64_7.dll`, `hiprtc0702.dll`-style versioned DLLs.

### GPU test gate

- `pykeops/pykeops/test/` (30 files: `test_torch.py`, `test_numpy.py`, `test_float16.py`, `test_chunks*.py`, `test_finalchunks*.py`, `test_block_sparse_reduction.py`, `test_lazytensor_*`, `test_gpu_cpu.py`, `test_vmap.py`, ...) run by `pytest.sh` (the upstream CUDA CI entry point).
- `keopscore/keopscore/test/test_op.py` (per-op formula checks) and `test_openmp.py`.
- Benchmarks under `benchmarks/` and `pykeops/pykeops/benchmarks/` for performance sanity (optional, not a gate).
- Proposed gate: full `pytest.sh` on ROCm PyTorch, GPU tests passing with tolerances unchanged, on wave64 and wave32.

### Windows

Upstream does not support Windows: install docs cover Linux (gcc) and macOS (clang) only; compiler selection, `-fPIC`/`-shared` and `.so` discovery are POSIX-only; issue #43 "Windows support" open since 2020, PR #281 (2022) and #421 "[Draft] Windows compatibility" (2025, external contributor, no maintainer action) unmerged. The code also uses `signed long int` for indices and device pointers (`keops_nvrtc.cpp`, generated kernels), which is 32-bit on LLP64 Windows. A Windows ROCm build would mean doing upstream's missing Windows port first. Suggest a `windows` gate waiver (agent suggestion only; needs a person's approval).

### Upstream activity and receptiveness

Active but small core team: bcharlier (MEMBER), joanglaunes, jeanfeydy; last merged PR #446 (2026-06-24, config/cpp compilation rework, which added pip-CUDA discovery); last push 2026-07-17. External PRs do get merged (gdurif, Louis-Pujol, kpoeppel, DavidLapous, rubenalv) but older drafts linger (#421, #281). The June 2026 config rework is recent: rebase-sensitive area, so the port should follow the new `config/Cuda.py` structure. No public stance on AMD; the backend-agnostic generated code and the new `pykeops.config.cuda.is_available()` abstraction make a well-scoped hiprtc binder plausible. Keep the diff additive.

## Planning (2026-11-20, linux-gfx90a)

**Decisions needed from Jeff: none.** The plan ports the whole project (keopscore binder and config, pykeops, rkeops through pykeops). Nothing is deferred, delivery (upstream PR) and the windows waiver were already ruled, and the remaining choices are engineering defaults listed under Open questions in `plan.md`.

Upstream analysed at `ef72bc907df7`. Build class: runtime-compiled NVRTC + driver API Python package (`ext_type = nvrtc-driver-api`); no hipcc anywhere, the port is a g++ build against HIP headers plus hipRTC at run time. Strategy: shim headers (`cuda.h`, `nvrtc.h`, `cuda_fp16.h`) on the HIP include path only, so `keops_nvrtc.cpp`, `utils_pe.h`, `CudaSizes.h` and `pykeops_nvrtc.cpp` stay unchanged; one guarded block in `nvrtc_jit.cpp` for `--offload-arch=<gcnArchName>`; a `HipConfig` behind the existing `keopscore.config.cuda` object, CUDA selected first.

Findings beyond the intake assessment:

- `keopscore/utils/math_functions.py:49` wraps every generated math call in `#ifdef __CUDACC__`; hipRTC does not define it, so GPU code would silently use the CPU spellings (float mostly passes, half2 fails to compile). Guard becomes `__CUDACC__ || __HIPCC__`.
- pykeops turns `use_fast_math` on by default, and Min/Max/LogSumExp reductions start from `( 1.0f/0.0f )`, so `-ffast-math` (finite-math-only) would break them. On HIP the option maps to nothing.
- `config/Cuda.py` hard-codes CUDA attribute ids 1 and 8; HIP 7.16 `hipDeviceAttributeMaxThreadsPerBlock` = 56, `hipDeviceAttributeMaxSharedMemoryPerBlock` = 74 (read from the headers with g++ on this host; no GPU used).
- Every GPU test gates on `pykeops.config.cuda.is_available()`, so failed HIP discovery gives a green CPU-only suite; the validator must prove the GPU ran.
- `pytest.sh` builds fresh venvs and installs torch from PyPI (CUDA); on ROCm run its steps in a venv with a ROCm torch.
- Header checks on this host (pip ROCm SDK 10.1, HIP 7.16.26385, no `/opt/rocm`): all fp16 intrinsics KeOps emits are declared in `amd_hip_fp16.h`; `g++ -std=c++11 -O3 -fpermissive -D__HIP_PLATFORM_AMD__` compiles `hip_runtime_api.h` + `hip_fp16.h` + `hiprtc.h`; `hipMemcpyHtoD` takes `const void*`.

## Porter round (2026-11-20, linux-gfx950)

**Decisions needed from Jeff: none.** The port follows `plan.md` with no scope change. Two test results the reviewer and validators should know about are explained below (one pykeops GPU failure that is a data tie, not a port defect; rkeops' compute tests skipping on upstream main for an upstream reason); neither needs a ruling.

Fork `AMD-Ecosystem/keops`, branch `moat-port`, head `2980874a` on upstream `ef72bc90`:

- `38c65e2b` [ROCm] Run the GPU backend on AMD GPUs through HIP and hipRTC
- `2980874a` [ROCm] Document running KeOps on AMD GPUs

### What changed (against the plan)

- Shim headers in `keopscore/keopscore/binders/nvrtc/hip_shim/` (`cuda.h`, `nvrtc.h`, `cuda_fp16.h`), on the ROCm include path only; `keops_nvrtc.cpp`, `utils_pe.h`, `CudaSizes.h`, `pykeops_nvrtc.cpp` unchanged, as planned. Added to `keopscore/setup.py` package data (checked: the built wheel contains them).
- R6 resolved the `hipSetDevice` way, not 1:1: the 1:1 mapping worked on HIP 7.17 but g++ printed ~100 lines of deprecation warnings, which KeOps echoes to the user on every binder compile. The shim makes `CUcontext` a `hipDevice_t`; `cuDevicePrimaryCtxRetain` stores the device, `cuCtxGetCurrent`/`cuCtxPushCurrent` are `hipGetDevice`/`hipSetDevice`, `cuDevicePrimaryCtxRelease` is a no-op. `cuCtxSynchronize` maps to `hipDeviceSynchronize` (hipCtxSynchronize returns NotSupported on some ROCm releases).
- `nvrtc_jit.cpp`: two `__HIP_PLATFORM_AMD__` blocks: `--offload-arch=<gcnArchName>` via `hipGetDeviceProperties` (fast-math not translated, R3), and include directories passed as `-I<dir>`: **hipRTC does not accept NVRTC's `--include-path=<dir>`**; it parsed it as `--include` + `-path=<dir>` and failed with `'-path=...hip_shim' file not found` (first GPU attempt).
- `config/Hip.py`: `HipConfig(CudaConfig)`. Discovery order: a `libamdhip64` already mapped in the process (`/proc/self/maps`), `ROCM_PATH`/`HIP_PATH`/`ROCM_HOME`, `CONDA_PREFIX`, site-packages `_rocm_sdk_core`, `_rocm_sdk_devel`, `torch`, then `/opt/rocm`. hipRTC is taken from the runtime's own install first. Attribute ids 56/74 (checked identical in HIP 7.16 and 7.17 headers). Version decoded as HIP's `major*10^7 + minor*10^5 + patch`. IR type `HSACO` (code objects are `<hash>_nvrtc.hsaco`). Empty `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES` means CPU only (R11). hipRTC include path is the shim dir only; host include options add the ROCm include root.
- `config/__init__.py`: `_gpu_config()` builds `CudaConfig` first (stdout captured), falls back to `HipConfig` only when CUDA is unusable, and prints only the reported backend's warnings (so an AMD host does not print "libcuda not found"). `CudaConfig` gains only `gpu_backend = "CUDA"`; the build folder name uses it (`..._HIP71726422_...` vs `..._CUDA<ver>_...`), which also keeps CUDA and HIP caches apart.
- `utils/math_functions.py`: `#if defined(__CUDACC__) || defined(__HIPCC__)` (R1).
- No `KEOPS_GPU_BACKEND` override (not needed; kept the diff small). `Gpu_link_compile` messages still say "cuda jit compiler engine" (left as upstream wording).
- Docs: `doc/python/installation.rst` gets "ROCm detection (AMD GPUs)" beside "CUDA toolkit detection" (requirements bullet too), and `doc/python/api/common/Config.rst` lists `HipConfig`. README defers install to the docs site, so it is unchanged.

### Environment (this host)

TheRock ROCm 10.2 nightly wheels into the job venv (`$FLEET_VENV`, Python 3.11.17), index `https://nightly.repo.amd.com/rocm/whl-next/`:

```
V=10.2.0a20261022
$FLEET_VENV/bin/python -m pip install --index-url https://nightly.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple \
  "rocm[libraries,devel]==$V" "rocm-sdk-device-gfx950==$V" "torch==2.16.0a0+rocm$V" "amd-torch-device-gfx950==2.16.0a0+rocm$V"
$FLEET_VENV/bin/python -m pip install --no-deps --no-build-isolation -e keopscore -e pykeops
```

HIP 7.17.26422, torch 2.16.0a0+rocm10.2.0a20261022, g++ 13.3.0, AMD Instinct MI355X (gfx950), one GPU (`HIP_VISIBLE_DEVICES=3`), `ulimit -c 0`. `10.2.0a20261022` is the newest 10.2 build on that index (10.3 nightlies follow). Host quirk, not a port issue: the job venv inherits the base image's amdsmi 10.1 Python package through a `.pth`, and with it `torch.cuda.device_count()` returned 0 while tensors worked; `PYTHONPATH=$SP/_rocm_sdk_devel/share/amd_smi` (the 10.2 copy) fixed the count. KeOps does not use the torch device count.

### Evidence (GPU, head 2980874a)

From the fork root, `KEOPS_CACHE_FOLDER` in the job dir, cache cleaned first:

```
python -c "import pykeops; pykeops.clean_pykeops(); pykeops.check_health()"   # run from /tmp: from the repo root, ./keopscore and ./pykeops shadow the packages
pytest -v -rs keopscore/keopscore/test/
pytest -v -rs pykeops/pykeops/test/
```

- `check_health`: "HIP Support: Enabled", 1 GPU, HIP 7.17, `-DMAXTHREADSPERBLOCK0=1024 -DSHAREDMEMPERBLOCK0=163840` (gfx950 reports 160 KiB LDS).
- keopscore: **68 passed** (test_op.py places tensors on `cuda` when the backend is available).
- pykeops: **92 passed, 1 failed, 0 skipped** (CPU-only run skips 6 GPU-only tests; here all ran). 247 `*_nvrtc.hsaco` code objects built during the run (R2). `test_float16.py` 2 passed on GPU (half2 formulas only compile through the GPU branch of the R1 guard).
- The failure, `test_lazytensor_clamp.py::test_lazytensor_clamp_bw`, is a tie in this GPU's random data, not a port defect: element (376,0,2) torch 181667.06 vs KeOps 182789.88. Float64 torch gives 182789.896, so KeOps is right (KeOps GPU vs float64: rel L2 1.5e-7; KeOps **CPU** backend on the same tensors gives the identical 182789.875; torch float32 on CPU gives the same wrong 181667.05). Cause (corrected 2026-11-20 after the review below, which measured it): `x[376,0,2]*y[0,664,2]` is exactly `1.54f` in float32 (1.5399999960 in float64, strictly inside the bound); torch's float32 clamp backward treats the bound as outside (gradient 0 at the tie), while KeOps' `Clamp.DiffT` keeps it inside (gradient 1 on both the GPU and CPU backends), as float64 torch does. `(p-a)/(b-a)` is exactly 1.0 in float32, not past 1. So torch float32 is the side that drops the gradient. The data come from `torch.manual_seed(0)` + `torch.randn(device="cuda")`; the generated values are expected to differ between GPU vendors (inferred, not checked on an NVIDIA GPU), which would explain why upstream's NVIDIA CI does not hit the tie. Tolerances left unchanged. Diagnosis scripts: `agent_space/diag/clamp.py`, `tie.py` (container-local).
- No-torch smoke test (fresh venv without torch, `ROCM_PATH` set to the SDK): `check_health()` + `test_numpy_bindings()` pass, `torch_found False`.
- Discovery: with `ROCM_PATH` unset the SDK is found through site-packages (`_rocm_sdk_core/lib/libamdhip64.so.7`, the doc example output is this run's); with torch imported first, KeOps reports the same library torch loaded, and a Gaussian kernel sum matched torch to 3.2e-7.

### Non-GPU regression

- Same suites with `HIP_VISIBLE_DEVICES=` (empty): keopscore 68 passed; pykeops 87 passed, 6 skipped (the GPU-only tests: test_conv2d x3, test_float16 x2, test_gpu_cpu x1), 0 failed. Warning shown: "HIP_VISIBLE_DEVICES is set to empty ... CPU only".
- `black` (26.10.1, upstream uses `psf/black@stable`): `black --check keopscore pykeops` clean, 376 files.

### CUDA path (this host has CUDA 12.8 headers and libraries, no NVIDIA GPU)

`agent_space/cudacheck/build.sh` (container-local): `nvrtc_jit.cpp` and `pykeops_nvrtc.cpp` compile and link with g++ using exactly the flags `CudaConfig` emits (`-DMAXIDGPU=0 ... -DnvrtcGetTARGET=nvrtcGetCUBIN -DARCHTAG=\"sm\" -std=c++11 -O3 -flto=auto -fpermissive -fPIC -I/usr/local/cuda/include`, linked to `lib64/stubs/libcuda.so` and `libnvrtc.so`); 40 generated formula sources from the GPU run compile with `nvcc -arch=sm_90 -cubin`. `CudaConfig` is selected first by construction (`_gpu_config`), and its code is unchanged apart from the `gpu_backend` attribute. Not run: an NVIDIA GPU pass of `pytest.sh` (no NVIDIA host available to this job).

### rkeops (evidence, not a gate)

R 4.4.3 and the rkeops dependencies from conda-forge into `agent_space/r-env` (micromamba, job dir); `R CMD INSTALL rkeops` into a job-local library; `RETICULATE_PYTHON=$FLEET_VENV/bin/python`, `TEST_GPU=1`; `testthat::test_dir` excluding `test-01_install.R` (it runs `reticulate::py_install("pykeops")`, which would replace the editable install with the PyPI release). Result: 38 tests, 7 failed, 2 skipped (test files 07-10 skip). All of it is upstream drift, independent of the port: rkeops' `check_pypkg()` reads `keopscore.verbose`, which keopscore on main no longer has (removed in 6bfa32a6), so `skip_if_no_keopscore()` skips every compute test on any platform; the 7 failures are `test-03_rkeops_options.R` expecting 6 options (there are 7, `formula_only` was added) and `test-04_keops_setup.R` expecting `pykeops$verbose` to be logical (it is now an integer). Direct GPU check instead (`agent_space/diag/rkeops_gpu.R`): `rkeops_use_gpu()`, `keops_kernel("Sum_Reduction(Exp(-SqNorm2(x-y))*b, 1)")`, 2000 x 3000 points, float32: built `<hash>_nvrtc.hsaco`, max rel err 5.0e-7 vs a dense R reference, 5.3e-7 vs rkeops on CPU.

### Benchmark (context, not a gate)

`python benchmarks/KeOps.py` (Gaussian convolution, D=3, float32) on the MI355X: 1e4 / 1e5 / 1e6 points 0.0019 / 0.0116 / 0.5226 s per call. Same formula on the OpenMP CPU backend (`HIP_VISIBLE_DEVICES=`), 1e5 points: 0.51 s. `benchmarks/PyTorch_GPU.py` was not run (its 1e6 case materializes a 1e6 x 1e6 matrix).

### For the reviewer and validators

- Prove the GPU ran (R2): "HIP Support: Enabled" in `check_health`, `*_nvrtc.hsaco` files in the build folder, 0 GPU tests skipped.
- Expect `test_lazytensor_clamp_bw` to fail on any AMD GPU whose `torch.randn` stream produces the same tie (diagnosis above); confirm it with the KeOps CPU backend on the same tensors before treating it as a port failure. wave32 hosts may draw different data.
- Run pytest from the fork root with the `pytest` executable (not `python -m pytest`, which puts the repo root on `sys.path` so `./keopscore` shadows the package), and run Python one-liners from another directory for the same reason.

## Lesson for cuda-to-rocm: hipRTC rejects NVRTC's --include-path= spelling

Belongs in `references/platform-limits.md`, in the "NVRTC (CUDA C++ generated at run time) ports to hipRTC" paragraph. NVRTC accepts `--include-path=<dir>` (and `-I<dir>`); hipRTC passes options to clang, which reads `--include-path=<dir>` as the prefix-matched `--include` (force-include a file) followed by `-path=<dir>`, so the compile fails with `fatal error: '-path=<dir>' file not found` from `<built-in>`. Pass `-I<dir>` on the ROCm side (NVRTC also accepts `-I`, so a single spelling works for both if the CUDA path may change). Seen in keops (runtime-generated kernels with a header directory on the hipRTC include path).

## Review 2026-11-20 (linux-gfx950, head 2980874a)

Reviewed `git diff ef72bc90...2980874a` on `moat-port` with the pr-review skill; findings fact-checked inline (re-read the code, reproduced on the GPU). Verdict: **changes requested**, one finding.

### Finding 1: the clamp_bw explanation has the mechanism backwards (upstream-visible)

The conclusion holds (a data tie, KeOps is the correct side, not a port defect); the stated mechanism is the reverse of what happens. Commit `38c65e2b` body, Test Plan: "KeOps' Clamp gradient treats the bound as outside, torch as inside". The porter section above says the same, and adds that `ClampInt((x-a)/(b-a),0,1)` "rounds just past 1 and treats it as outside".

Measured on this host (`agent_space/diag/tie2.py`, container-local; same seed and data as the test, product `p = x[376,0,2]*y[0,664,2]`):

```
p == 1.54f: True 1.5399999618530273
torch float32 d clamp/dp at tie: 0.0
torch float64 d clamp/dp:        1.0
keops cuda d clamp/dp at tie:    1.0
keops cpu  d clamp/dp at tie:    1.0
float32 (p-a)/(b-a): 1.0
```

So torch float32 drops the gradient at the bound (treats it as outside), and KeOps keeps it (inside), as float64 does (in float64 the product is 1.5399999960 < 1.54, strictly inside). `(p-a)/(b-a)` is exactly 1.0, not past 1. The rest of the diagnosis reproduces exactly: only element (376,0,2) violates the tolerance, torch32 181667.0625, KeOps GPU 182789.875, KeOps CPU 182789.875, torch float64 182789.896, KeOps GPU vs float64 rel L2 1.51e-7.

Fix: reword the commit body's Test Plan (and the notes) to: "with this GPU's random data one product x*y equals the clamp bound 1.54f exactly in float32; torch's float32 clamp gradient treats the bound as outside, KeOps' as inside. The KeOps result matches a float64 reference (where the product is just inside the bound) to 1.5e-7, and the KeOps CPU backend gives the same value on the same data." `moat-port` has no upstream PR yet, so the message can still be corrected. No code change needed.

### Fact-checked and confirmed (no finding)

- rkeops skips and failures are upstream drift, independent of the port: `rkeops/R/rkeops_check.R:54` `check_pypkg()` treats the package as unavailable unless `tmp_pypkg$verbose` imports; `keopscore` has no `verbose` attribute at this head (removed with the module-level `verbose = True` in `6bfa32a6`; checked `hasattr(keopscore, "verbose") == False`), so `skip_if_no_keopscore()` (`tests/testthat/setup.R:17`) skips 07-10 everywhere. `test-03_rkeops_options.R:5` expects 6 options and `rkeops_options.R:76` now has `formula_only` (7); `test-04_keops_setup.R:24` expects `pykeops$verbose` to be `TRUE`, and it is the int `1`. None of these files are touched by the port.
- GPU rerun at this head (`ulimit -c 0`, `HIP_VISIBLE_DEVICES=3`, fresh `KEOPS_CACHE_FOLDER`, TheRock 10.2.0a20261022 wheels and torch 2.16.0a0+rocm10.2.0a20261022 in the job venv): `check_health` "HIP Support: Enabled", 1 GPU, HIP 7.17; `pytest -q -rfs keopscore/keopscore/test/` 68 passed; `pytest -q -rfs pykeops/pykeops/test/` 92 passed, 1 failed (`test_lazytensor_clamp_bw`), 0 skipped; 247 `*_nvrtc.hsaco` built. Review evidence only; the validator owns the gate.
- Lesson "hipRTC rejects NVRTC's --include-path= spelling": reproduced with hipRTC from the 10.2 wheels (`--include-path=<dir>` -> result 6, `<built-in>:2:10: fatal error: '-path=<dir>' file not found`; `-I<dir>` -> 0), and NVRTC 12.8 accepts both spellings (both 0). Accurate as written.
- Shim mappings checked against the call sites in the unchanged `keops_nvrtc.cpp` (context retain/push/get, `cuModuleLoadDataEx(&module, target, 0, NULL, NULL)`, `cuMemAlloc` on a `CUdeviceptr`, launches on stream `NULL`, `cuCtxSynchronize` -> `hipDeviceSynchronize`); `jargon.py --port keops` clean; commit titles carry `[ROCm]`, no trailers.

### CUDA compile check (2026-11-20, this host: CUDA 12.8 toolkit at /usr/local/cuda, no NVIDIA driver or GPU)

Method: run KeOps' own configuration and `Gpu_link_compile.get_compile_command` from the fork at 2980874a, replacing only `CudaConfig._find_and_load_libcuda` (the libcuda driver probe, which needs a driver) through an import hook installed before `keopscore.config` runs `_gpu_config`. The stub reports CUDA 12.8 (12080), one device with 1024 threads and 49152 bytes shared memory per block, and points at `lib64/stubs/libcuda.so`. NVRTC, header discovery, option generation and backend selection are KeOps' own code. Script: `agent_space/cudacheck/cmds.py` (container-local); run from `/tmp`:

```
KEOPS_CACHE_FOLDER=$OUT/cache $FLEET_VENV/bin/python -I cmds.py <fork-root> $OUT
```

- Backend selection: `selected: CudaConfig CUDA CUBIN`, with a HIP runtime also present on the host (ROCm Python packages in the venv). So on a CUDA host `_gpu_config` still selects CUDA before HIP; `HipConfig` is never constructed when CUDA is usable (`config/__init__.py:24-43`). Build folder name keeps `CUDA<version>`.
- KeOps itself built both binders through this config: "Compiling cuda jit compiler engine ... OK", "Compiling nvrtc binder for python ... OK".
- The exact commands it emits, rerun by hand (paths shortened; `<k>` = `keopscore/keopscore`):

```
/usr/bin/c++ -DMAXIDGPU=0 -DMAXTHREADSPERBLOCK0=1024 -DSHAREDMEMPERBLOCK0=49152 -DnvrtcGetTARGET=nvrtcGetCUBIN -DnvrtcGetTARGETSize=nvrtcGetCUBINSize -DARCHTAG=\"sm\"  -std=c++11 -O3 -flto=auto -fpermissive -fPIC -I<k> -I/usr/local/cuda-12.8/targets/x86_64-linux/include -I/usr/local/cuda/include <k>/binders/nvrtc/nvrtc_jit.cpp -shared /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libnvrtc.so.12 -Wl,-rpath,/usr/local/cuda/lib64/stubs -Wl,-rpath,/usr/local/cuda/lib64 -o nvrtc_jit.so
/usr/bin/c++ <same flags> $(python -m pybind11 --includes) pykeops/pykeops/common/keops_io/nvrtc/pykeops_nvrtc.cpp -shared <same libs> -o pykeops_nvrtc.so
c++ <same -D/-std/-O3/-fpermissive/-fPIC/-I flags> -c <k>/binders/nvrtc/keops_nvrtc.cpp -o keops_nvrtc.o
```

  All three rc=0 with 0 warnings (g++ 13.3.0). No `hip_shim` include or `__HIP_PLATFORM_AMD__` appears in the CUDA commands. `nm -D --undefined-only`: `nvrtc_jit.so` needs `cuDeviceGet cuDeviceGetAttribute cuGetErrorName nvrtc*` (CUBIN getters), NEEDED `libcuda.so.1 libnvrtc.so.12`; `pykeops_nvrtc.so` needs `cuCtxGetCurrent cuCtxPushCurrent_v2 cuCtxSynchronize cuDevicePrimaryCtxRetain cuDevicePrimaryCtxRelease_v2 cuMemAlloc_v2 ... cuLaunchKernel cuModuleLoadDataEx`, NEEDED `libcuda.so.1`; no `hip*` symbols anywhere.
- Generated formula code through real NVRTC (`agent_space/cudacheck/nvrtc_gen.py`): the 247 `.cu` sources KeOps generated during the GPU run above (2 of them half2), compiled with libnvrtc 12.8 using the CUDA-side options `nvrtc_jit.cpp` builds (`--include-path=` the two CUDA include dirs, `-arch=sm_90`, `-use_fast_math`): 247 compiled, 0 failed. The formula generator emits the same source on both backends; the `math_functions.py` guard still selects the GPU spellings under `__CUDACC__`.
- Not run: an NVIDIA GPU execution (no NVIDIA hardware on this host).

## Porter fix round (2026-11-20, linux-gfx950, head 2980874a)

Answers Finding 1 of the review above. No code change and no new fork commit: the wrong mechanism appears only in the body of `38c65e2b` (checked: `git diff ef72bc90...2980874a` adds no line mentioning clamp, the bound or the tie, and the `2980874a` body does not mention it). Jeff's standing rule for fork branches is no amend, no rebase and no force-push, because rewritten history desynchronizes the recorded head, so `38c65e2b` is not rewritten here. Head stays `2980874a`.

- Porter section above: the "Cause" sentence is corrected to the measured facts from the review (torch float32 drops the gradient at the tie, KeOps keeps it, as float64 does; `(p-a)/(b-a)` is exactly 1.0).
- The `## For the reviewer and validators` bullet about the tie names no mechanism and stays as written.

### Correction for the final history

When the commits are prepared for the upstream pull request (a later step a person approves), the Test Plan of `38c65e2b` ("[ROCm] Run the GPU backend on AMD GPUs through HIP and hipRTC") must replace this wrong text:

```
pykeops: 92 passed, 1 failed (test_lazytensor_clamp_bw: with this GPU's random data one product x*y equals the clamp bound 1.54f exactly; KeOps' Clamp gradient treats the bound as outside, torch as inside. The KeOps result matches a float64 reference to 1.5e-7, and the KeOps CPU backend gives the same value on the same data).
```

with exactly this sentence:

```
pykeops: 92 passed, 1 failed (test_lazytensor_clamp_bw: with this GPU's random data one product x*y equals the clamp bound 1.54f exactly in float32; torch's float32 clamp gradient treats the bound as outside, KeOps' as inside. The KeOps result matches a float64 reference (where the product is just inside the bound) to 1.5e-7, and the KeOps CPU backend gives the same value on the same data).
```

The rest of the commit message is unchanged. If that preparation squashes the two commits, the same replacement applies to the squashed body. Until then the published branch keeps the wrong sentence; nothing upstream-visible exists yet (no upstream PR, no review PR).

## Review 2026-11-20 round 2 (linux-gfx950, head 2980874a)

Re-review at the same head (`moat-port` local and remote both `2980874a`, fork tree clean). Verdict: **review passed**. No open findings.

Finding 1 (clamp_bw mechanism reversed in the `38c65e2b` Test Plan): resolved by the recorded correction. Reasoning:

- The wrong sentence exists only in the commit body of `38c65e2b`. `git grep` for "treats the bound", "clamp bound" and "as outside" over the tree at `2980874a` finds nothing; the `2980874a` body does not mention the tie. A commit message cannot be fixed by a new commit, so under the no-amend, no-force-push rule for fork branches the only correct places for the fix are the notes and the PR-preparation step, where history is rewritten anyway (the tree-identical squash that `squash-carry-forward` exists for) and where a person approves the exact title, body and commits on the review PR. Rewriting now would buy nothing: nothing upstream-visible exists (no upstream PR, no review PR).
- The "wrong text" block under "Correction for the final history" matches the commit body verbatim, so the replacement is mechanical.
- The corrected sentence matches the facts, re-measured independently this round on the MI355X (`HIP_VISIBLE_DEVICES=3`, `ulimit -c 0`, fresh `KEOPS_CACHE_FOLDER`; the fork's `keopscore`/`pykeops` on `PYTHONPATH`, no install into the tree; base-image torch 2.16.0a0, since the job venv has none): `x[376,0,2]*y[0,664,2] == 1.54f` in float32 (1.5399999618530273); the float64 product is 1.539999995966653, just inside the bound; torch float32 clamp gradient at the tie 0.0, torch float64 1.0; KeOps d/dx at the tie equals y (-2.086639, gradient kept) on both the GPU and CPU backends; float32 `(p-a)/(b-a)` is exactly 1.0; element (376,0,2) torch32 181667.062, torch64 182789.896, KeOps GPU 182789.875, KeOps CPU 182789.875; KeOps GPU vs float64 rel L2 1.51e-7; (376,0,2) is the only element outside the test tolerance.
- The porter section's "Cause" text now states the measured mechanism and drops the "rounds just past 1" claim; the reviewer/validator bullet names no mechanism.

Carried condition (not a finding): whoever prepares the commits for the review PR must apply the replacement under "Correction for the final history" before the review PR opens; no tool enforces it, so the approver should see the corrected Test Plan on the review PR.

Sanity check of the rest: `git diff --stat ef72bc90...2980874a` unchanged (12 files, +537/-4); `jargon.py --port keops` clean; record changes since the first review touch only `notes.md`, `stats.jsonl` and `status.json`. The first review's code, CUDA compile check and lesson verification stand.

## Validation 2026-11-20 (linux-gfx950, head 2980874a)

Verdict: **completed** (real-GPU pass on AMD Instinct MI355X, gfx950). Fork clone `projects/keops/src` at `2980874acefdc7e905b3733141c8765d770180ed`, `git status --porcelain` empty before and after.

### Environment

Job venv `$FLEET_VENV` (Python 3.11.17), TheRock 10.2 wheels installed with the porter recipe (index `https://nightly.repo.amd.com/rocm/whl-next/`, PyPI extra): `rocm[libraries,devel]==10.2.0a20261022`, `rocm-sdk-device-gfx950==10.2.0a20261022`, `torch==2.16.0a0+rocm10.2.0a20261022`, `amd-torch-device-gfx950==2.16.0a0+rocm10.2.0a20261022`; then `pip install --no-deps --no-build-isolation -e keopscore -e pykeops` from the fork root. `rocm-sdk init` was needed to expand `_rocm_sdk_devel` (the 10.2 `share/amd_smi` does not exist until then). pip printed dependency-conflict warnings for the image's 10.1 device packages reached through the venv's base-links `.pth`; they are shadowed by the venv's own site-packages and are not loaded. amdsmi quirk handled as the porter describes: `PYTHONPATH=$SP/_rocm_sdk_devel/share/amd_smi` (10.2 copy), after which `torch.cuda.device_count()` = 1. Confirmed what runs: `pip list` shows rocm/rocm-sdk-core/devel/libraries/device-gfx950 10.2.0a20261022 and torch 2.16.0a0+rocm10.2.0a20261022; `torch.__version__` 2.16.0a0+rocm10.2.0a20261022, `torch.version.hip` 7.17.26422, device "AMD Instinct MI355X"; `/proc/self/maps` after `import pykeops` and `check_health` both name `$FLEET_VENV/lib/python3.11/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7` and `libhiprtc.so.7` (no 10.1 or /opt/rocm library). g++ 13.3.0. `HIP_VISIBLE_DEVICES=3`, `ulimit -c 0`, `KEOPS_CACHE_FOLDER` a fresh `keops-cache-val/` in the job directory (deleted and recreated first). Env file and scripts: `agent_space/val/` (container-local).

### GPU run (all wrapped in `utils/timeit.sh keops test -- ...`)

```
cd /tmp && python -c "import pykeops; pykeops.clean_pykeops(); pykeops.check_health()"
cd <fork root> && pytest -v -rfs keopscore/keopscore/test/
cd <fork root> && pytest -v -rfs pykeops/pykeops/test/
```

- `check_health`: "HIP Support: Enabled", 1 GPU, HIP 7.17, `-DMAXTHREADSPERBLOCK0=1024 -DSHAREDMEMPERBLOCK0=163840`, build folder `..._HIP71726422_VISIBLE_DEVICES3`.
- keopscore: **68 passed**, 0 skipped, 0 failed.
- pykeops: **92 passed, 1 failed (`test_lazytensor_clamp.py::TestCase::test_lazytensor_clamp_bw`), 0 skipped**. The GPU-only tests all ran and passed: test_conv2d x3, test_float16 x2, test_gpu_cpu x2.
- `find $KEOPS_CACHE_FOLDER -name '*_nvrtc.hsaco' | wc -l` after both suites: **247**.

### test_lazytensor_clamp_bw: how it was counted and judged

Raw pytest result, counted as is: 1 failed (`AssertionError: clamp_bw: ||ref-test||_2=1.122814e+03 and ||ref||_2=1.070351e+07`). Test and tolerances unchanged.

Judgement, on evidence reproduced in this run (`agent_space/val/clamp_diag.py`, `clamp_diag2.py`; same seed and data as the test, same GPU):

- Elements failing the test's criterion (`allclose(torch, keops, atol=0.01, rtol=0.001)`): only **(376,0,2)**. There: torch float32 181667.0625, torch float64 182789.8963, KeOps GPU 182789.8750, KeOps CPU backend (same values moved to CPU tensors; the CPU backend refuses cuda tensors) 182789.9219.
- `x[376,0,2]*y[0,664,2]` in float32 = 1.5399999619 == 1.54f exactly; float64 product 1.539999995967, just inside the bound. KeOps' `Clamp.DiffT` ratio with float32 operands `(p-a)/(b-a)` = exactly 1.0. (Computed in torch with Python-double `a`, `b` the ratio prints 0.99999994; the float32-operand form is the one KeOps evaluates.)
- Clamp derivative at the tie, d/dx of clamp(x*y): torch float32 **-0.0** (dropped); torch float64 -2.086639 (= y, kept); KeOps GPU -2.086639; KeOps CPU -2.086639.
- Whole gradient, relative L2: KeOps GPU vs torch float64 **1.510e-7**; KeOps CPU vs float64 1.322e-7; torch float32 vs float64 1.049e-4; KeOps GPU vs KeOps CPU 1.905e-7 (max abs diff 0.375 on values up to ~1e5; not bitwise equal). One difference from the earlier records: the porter and reviewer saw KeOps CPU give exactly 182789.875 at the tie element; here it gives 182789.922 (rel 2.6e-7). This is float32 reduction-order rounding between the backends (this host has 256 cores for the OpenMP reduction), three orders of magnitude below the 1122.8 torch discrepancy, and both backends keep the gradient at the tie and match float64.

Conclusion: the port is correct on this test. torch float32 is the side that drops the gradient at an exact float32 tie with the bound; KeOps on the HIP backend agrees with a float64 reference to 1.5e-7 and with its own CPU backend to float32 rounding. Not a port defect, so it does not block the verdict; it stays a failing test in the raw count.

### Non-GPU regression (same suites, `HIP_VISIBLE_DEVICES=` empty, wrapped in timeit)

- keopscore 68 passed; pykeops **87 passed, 6 skipped, 0 failed** (skips: test_conv2d x3, test_float16 x2, test_gpu_cpu x1, all "CUDA not available"/"Requires KeOps CUDA support"). Identical to the porter's counts. No regression.

### CUDA gate

Skipped: notes already record the CUDA compile check at this head (`### CUDA compile check (2026-11-20, ...)` under the first review, head 2980874a); the gate runs once per head_sha.

### Content checks

- `python3 utils/jargon.py --port keops`: clean.
- Docs: `doc/python/installation.rst` has "ROCm detection (AMD GPUs)" (`part.ROCmDetection`) beside the CUDA detection section and the requirements bullet; `doc/python/api/common/Config.rst` lists `HipConfig`. Present.
- The corrected Test Plan sentence for `38c65e2b` stays recorded under "### Correction for the final history" for the PR-preparation step; not a docs or jargon gate failure (reviewer's settled finding).

## Validation 2026-11-20 (linux-gfx90a, head 2980874a)

Verdict: **completed** (real-GPU pass on AMD Instinct MI250X / MI250, gfx90a:sramecc+:xnack-, wave64). Additional evidence on a second CDNA generation; the wave64 gate was already satisfied by linux-gfx950. Fork clone `projects/keops/src` freshly cloned and checked out at `2980874acefdc7e905b3733141c8765d770180ed`; `git status --porcelain` empty before and after.

### Environment

Job venv `$FLEET_VENV` (Python 3.11.17), TheRock 10.2 wheels with the porter recipe for this arch (index `https://nightly.repo.amd.com/rocm/whl-next/`, PyPI extra): `rocm[libraries,devel]==10.2.0a20261022`, `rocm-sdk-device-gfx90a==10.2.0a20261022`, `torch==2.16.0a0+rocm10.2.0a20261022`, `amd-torch-device-gfx90a==2.16.0a0+rocm10.2.0a20261022`; `pip install --no-deps --no-build-isolation -e keopscore -e pykeops` from the fork root; `rocm-sdk init`; `PYTHONPATH=$SP/_rocm_sdk_devel/share/amd_smi` (amdsmi quirk as before; `torch.cuda.device_count()` = 1). pip printed the same dependency-conflict warnings for the image's 10.1 device packages reached through the base-links `.pth`; shadowed, not loaded. Confirmed: `torch.__version__` 2.16.0a0+rocm10.2.0a20261022, `torch.version.hip` 7.17.26422, `warp_size` 64; `/proc/self/maps` after a KeOps GPU reduction names only `$FLEET_VENV/.../_rocm_sdk_core/lib/libamdhip64.so.7` and `libhiprtc.so.7`. g++ 13.3.0. `HIP_VISIBLE_DEVICES=0` (the job's pin), `ulimit -c 0`, fresh `KEOPS_CACHE_FOLDER` in the job directory. Note for the timeit wrapper: with the venv first on PATH, `utils/timeit.sh` needs `MOAT_PYTHON` set to the MOAT interpreter, and it does not keep the caller's cwd, so the pytest commands are wrapped as `bash -c "cd <fork root> && pytest ..."`.

### GPU run (wrapped in `utils/timeit.sh keops test -- ...`)

```
cd /tmp && python -c "import pykeops; pykeops.clean_pykeops(); pykeops.check_health()"
bash -c "cd <fork root> && pytest -v -rfs keopscore/keopscore/test/"
bash -c "cd <fork root> && pytest -v -rfs pykeops/pykeops/test/"
```

- `check_health`: "HIP Support: Enabled", 1 GPU, HIP 7.17, `-DMAXTHREADSPERBLOCK0=1024 -DSHAREDMEMPERBLOCK0=65536` (gfx90a reports 64 KiB LDS, against 160 KiB on gfx950, so the shared-memory-bounded block sizing differs from the gfx950 run), build folder `..._HIP71726422_VISIBLE_DEVICES0`.
- keopscore: **68 passed**, 0 skipped, 0 failed.
- pykeops: **92 passed, 1 failed (`test_lazytensor_clamp.py::TestCase::test_lazytensor_clamp_bw`), 0 skipped**. GPU-only tests all ran and passed: test_conv2d x3, test_float16 x2, test_gpu_cpu x2.
- `*_nvrtc.hsaco` in the cache after both suites: **247**.

### test_lazytensor_clamp_bw on this GPU, judged on its own evidence

Raw result counted as is: 1 failed, `AssertionError: clamp_bw: ||ref-test||_2=1.122814e+03 and ||ref||_2=1.070351e+07` (test and tolerances unchanged). Diagnosis re-run here (`agent_space/val/diag/clamp_diag.py`, container-local; same seed and data as the test):

- Data: `torch.manual_seed(0)` + `torch.randn(device="cuda")` on the MI250X gives x sha256 prefix `e545a9ac46dc0ff4`, y `6105642bdca5ec06`, and the same tie and the same norm as on gfx950, so this generator produces the same stream on both CDNA generations.
- Only element failing the test criterion (`isclose(torch, keops, atol=0.01, rtol=0.001)`): **(376,0,2)**: torch float32 181667.0625, torch float64 182789.8963, KeOps GPU 182789.8750, KeOps CPU backend (CPU tensors) 182789.8750.
- The single exact float32 tie in the whole x*y product: (i,j,k) = (376,664,2), x -0.7380291, y -2.0866387, product 1.5399999619 == 1.54f; float64 product 1.539999995967 (just inside); float32 `(p-a)/(b-a)` = 1.0.
- d clamp(x*y)/dx at the tie: torch float32 **-0.0** (dropped); torch float64 -2.0866387 (kept); KeOps GPU -2.0866387; KeOps CPU -2.0866387.
- Whole gradient, relative L2: KeOps GPU vs torch float64 **1.510e-7**; KeOps CPU vs float64 1.510e-7; torch float32 vs float64 1.049e-4; KeOps GPU vs KeOps CPU **0** (bitwise equal on this host, max abs diff 0.0).
- Wavefront: gfx90a is wave64 like gfx950, so this run does not exercise a 32-wide wavefront; what differed from gfx950 was the LDS size (64 vs 160 KiB) and therefore the tile/block sizing, and the KeOps result is unchanged (182789.875 at the tie element on both GPUs, bitwise equal to this host's CPU backend). That is consistent with the code read at intake: reductions are per-thread serial loops over j tiles, with no warp intrinsics, so the result does not depend on wavefront width or block size. A wave32 host may draw different random data, so the test may pass or fail there for data reasons alone.

Conclusion: not a port defect; torch float32 is the side that drops the gradient at an exact float32 tie with the bound. KeOps on the HIP backend matches float64 to 1.5e-7. It stays 1 failed in the raw count and does not block the verdict.

### Non-GPU regression (`HIP_VISIBLE_DEVICES=` empty, wrapped in timeit)

keopscore **68 passed**; pykeops **87 passed, 6 skipped, 0 failed** (test_conv2d x3, test_float16 x2, test_gpu_cpu x1: "CUDA not available"/"Requires KeOps CUDA support"). Identical to the porter's and the gfx950 counts.

### CUDA gate

Skipped: already recorded at this head (`### CUDA compile check (2026-11-20, ...)` under the first review); it runs once per head_sha.

### Content checks

`python3 utils/jargon.py --port keops`: clean (needed local `main` and `moat-port` branches in the fresh clone). Docs present: `doc/python/installation.rst` "ROCm detection (AMD GPUs)", `doc/python/api/common/Config.rst` lists `HipConfig`. The "Correction for the final history" condition for the `38c65e2b` Test Plan still stands for the PR-preparation step.
## Validation 2026-11-20 (linux-gfx1100, head 2980874a): blocked, GPU wedged

Verdict: **no verdict on the port; platform `blocked`**. The one GPU this job was pinned to (`HIP_VISIBLE_DEVICES=3`, AMD Radeon Pro W7800 48GB, gfx1100, warp size 32, PCI bus 0x26) never finishes any GPU work: a host-to-device copy hangs before any KeOps kernel runs, under two ROCm runtimes, with SDMA disabled, and from torch alone. That is the "wedged W7800" host state already described in `cuda-to-rocm` references/libraries.md (first H2D copy never completes, GFX activity high with no process), not evidence about the port. The wave32 gate is still open. `test_lazytensor_clamp_bw` could not be run on this GPU, so nothing is known yet about how a 32-wide wavefront GPU draws that test's data. Fork clone `projects/keops/src` at `2980874acefdc7e905b3733141c8765d770180ed`, `git status --porcelain` empty.

### Environment

Job venv `$FLEET_VENV` (Python 3.11.17), TheRock 10.2 wheels with the porter recipe, device package for this arch: `rocm[libraries,devel]==10.2.0a20261022`, `rocm-sdk-device-gfx1100==10.2.0a20261022`, `torch==2.16.0a0+rocm10.2.0a20261022`, `amd-torch-device-gfx1100==2.16.0a0+rocm10.2.0a20261022`; `rocm-sdk init`; `pip install --no-deps --no-build-isolation -e keopscore -e pykeops`; `PYTHONPATH=$SP/_rocm_sdk_devel/share/amd_smi`. Same pip conflict warnings for the image's 10.1 device packages as on gfx950. torch reports 2.16.0a0+rocm10.2.0a20261022, HIP 7.17.26422, 1 device, gcnArchName gfx1100, warp_size 32. amdgpu driver 7.1.9 (`/sys/module/amdgpu/version`), kernel 6.8. `ulimit -c 0`. Scripts and logs: `agent_space/val/` (container-local).

### What ran on the GPU side

- `check_health` (from /tmp, fresh `KEOPS_CACHE_FOLDER`): "HIP Support: Enabled", 1 GPU, HIP 7.17, `-DMAXTHREADSPERBLOCK0=1024 -DSHAREDMEMPERBLOCK0=65536` (gfx1100 reports 64 KiB LDS), build folder `..._HIP71726422_VISIBLE_DEVICES3`, libamdhip64/libhiprtc from the venv's `_rocm_sdk_core` (confirmed in `/proc/self/maps`). Both binders (`nvrtc_jit.so`, `pykeops_nvrtc`) built. This needs no GPU execution.
- `pytest -v -rfs keopscore/keopscore/test/` (wrapped in timeit): collected 68, hung on the first test (`test_formula_maths[Abs]`) for over 10 minutes at 100% CPU with no formula source generated yet, i.e. in torch's input tensor setup before KeOps. Killed. pykeops GPU suite not started.
- Minimal repro (`repro1.py`: two `torch.randn(device="cuda")` tensors and `Genred("Abs(x-y)", ...)`): hipRTC compiled the formula; the code object is valid for this GPU (`llvm-readelf`: e_flags gfx1100, `amdhsa.target amdgcn-amd-amdhsa--gfx1100`, `.wavefront_size: 32`, kernel `GpuConv1DOnDevice`). faulthandler shows the process then stuck inside the binder's `call_keops` (the C++ launch/synchronize) indefinitely.

### Isolation: the device, not KeOps

- Standalone C++ launcher (`hs/launch.cpp`, g++ against HIP headers, no KeOps code): `hipInit`, `hipSetDevice`, `hipModuleLoadData` of the KeOps code object, `hipModuleGetFunction`, four `hipMalloc` all succeed; the first `hipMemcpyHtoD` (12 bytes x 300) never returns (killed at 60 s). Same with the same source linked to the base image's ROCm 10.1 `libamdhip64` (so not the 10.2 nightly), and with `HSA_ENABLE_SDMA=0` (blit-kernel copy, so the compute queue itself does not complete, not only the DMA engine).
- torch alone: `torch.arange(10.).cuda()` never returns (killed at 60 s).
- `amd-smi` for this card (amd-smi index 1, BDF 26:00.0): GFX_ACTIVITY 100%, 28 MB VRAM used, no process listed, ECC counts 0, 43 C. Rechecked at 22:58Z: the copy still hangs, activity still 100%.
- The first operation that waits on this GPU in this job already hung (the pytest above, before any KeOps kernel was launched), so this job did not cause the state. Not checked: whether a previous job on the host did, and whether other cards on the host are healthy (this job may use only its own pin). No root, so no `dmesg` and no `amd-smi reset`.

### Non-GPU regression (`HIP_VISIBLE_DEVICES=` empty, fresh cache, wrapped in timeit)

- keopscore **68 passed**; pykeops **87 passed, 6 skipped, 0 failed** (skips: test_conv2d x3, test_float16 x2, test_gpu_cpu x1). Identical to the porter and gfx950 counts. No regression.

### What is needed

Reset or reboot the host so this W7800 completes work again (or pin a different, healthy gfx1100 or other wave32 GPU), then `set-blocked keops linux-gfx1100 --clear` and rerun this recipe in full, judging `test_lazytensor_clamp_bw` on that GPU's own evidence. No porter work follows from this run.

### Content checks

`python3 utils/jargon.py --port keops`: clean.
