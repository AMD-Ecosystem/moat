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
