# Mirage (MPK) -- ROCm/HIP port plan

## Project
- Name: mirage
- Upstream: https://github.com/mirage-project/mirage
- Default branch: `mpk` (the repo has pivoted to "Mirage Persistent Kernel": compiling LLMs into a single CUDA megakernel)
- Analyzed at HEAD `bde2dec` ("[Online Serving] ..." #660); re-confirmed 2026-06-16 at HEAD `2c87a75` ("Support DFlash for Kimi-K2.6" #728). The intervening work added only more NVIDIA-specific kernels (DFlash, fp8 sm100, 2-SM Blackwell branches), reinforcing -- never relaxing -- the NVIDIA-arch lock.

## Disposition: SKIP (cant-port)

Recommended triage:
`python3 utils/triage.py skip mirage-project/mirage --reason cant-port --note "CUTLASS/CuTe-centric tensor superoptimizer + NVIDIA-arch-locked megakernel codegen (PTX mma/wgmma/TMA/cp.async, sm_80/sm_90a/sm_100a, NVSHMEM); no tractable correctness-first HIP target"`

This is NOT a mechanical CUDA->HIP port. The CUDA surface is (a) CUTLASS/CuTe-centric, which per the PORTING_GUIDE does not port to ROCm and must be reimplemented against CK/ck_tile, and (b) locked to NVIDIA-specific architectures and runtime/library features at every layer that touches the GPU (raw PTX tensor-core asm, TMA, async copy, NVSHMEM, NVRTC/nvcc-at-runtime targeting `sm_90a`/`sm_100a`). There is no thin correctness-first slice that yields a working ROCm build without effectively rewriting Mirage's entire kernel library and code generator for AMD. Reasoning and evidence below.

## Existing AMD support
- Re-checked 2026-06-16 (no clone): upstream still has zero AMD/ROCm/HIP/gfx code (GitHub code search for `hipMalloc/gfx90a/rocblas/__HIP_PLATFORM` returns nothing); no AMD/ROCm/HIP PRs in the upstream queue under any author (`gh pr list --search "ROCm OR HIP OR AMD"` empty); no forks under the ROCm/AMD/GPUOpen orgs; no separately-named AMD project (no ROCm-DS-style effort); default branch is still `mpk`. Active branches (`2sm`, `fp8_sm100_kernel`, `cute_linear`, `linear_cutlass`) show continued CUTLASS/CuTe + sm100 Blackwell development. Determination unchanged.
- None. Zero references to HIP/ROCm/amdgcn/gfx*/MFMA/rocBLAS/hipBLAS anywhere in `src/`, `include/`, `python/`, CMake, or setup.py (the only "hip" substring hits are in `..._sm100...` Blackwell filenames). No stale AMD branch, no AMD PR, no abandoned port.
- The project DOES carry a non-CUDA backend abstraction (`MIRAGE_BACKEND_USE_CUDA` vs `MIRAGE_BACKEND_USE_NKI`, `MIRAGE_FINGERPRINT_USE_CUDA` vs `..._USE_CPU` in `include/mirage/config.h`), but the alternative is NKI = AWS Trainium / Neuron (`src/nki_transpiler/`, `NeuronArch`, `nki.language.*`), not AMD. The abstraction does not provide a HIP path or any seam an AMD backend could slot into cheaply; it only chooses between emitting CUDA C++ vs emitting Python NKI source.

## Build classification: CMake + Cython (NOT a pytorch extension), plus Rust + Z3
Evidence:
- `setup.py` builds a standalone C++ library `mirage_runtime` by invoking CMake/`make` directly (lines ~192-235), then builds a single Cython extension (`python/mirage/_cython/core.pyx`) that links `mirage_runtime`, `cudart`, `cudadevrt`, `cuda`, `z3`, plus two Rust `.so`s. It is not `torch.utils.cpp_extension`/`CUDAExtension`; torch is only a runtime dep (`requirements.txt: torch>=2.4`) used by the Python frontend and tests.
- `CMakeLists.txt`: `project(MIRAGE LANGUAGES C CXX CUDA)`, `set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")`, finds CUDA via `cmake/cuda.cmake` + `deps/cutlass/CUDA.cmake`, links `${CUDA_CUDART_LIBRARY}`/`${CUDA_CUDA_LIBRARY}`, hard-includes `deps/cutlass/include` and `deps/cutlass/tools/util/include`.
- Two Rust crates (`src/search/abstract_expr/abstract_subexpr`, `src/search/verification/formal_verifier_equiv`) built via cargo; Z3 solver (`z3-solver==4.16`) for formal verification. These are GPU-irrelevant (the search/verification engine) and not a porting concern, but they enlarge the build.
- Under the normal MOAT taxonomy this would be "Strategy A (pure CMake compat-header)". That classification is moot here because the GPU code is CUTLASS/PTX/NVIDIA-arch-locked (see below), which Strategy A cannot address.

## Port strategy: none viable (neither A nor B)
- Strategy A (compat header + `enable_language(HIP)` + `.cu` as `LANGUAGE HIP`) presupposes the `.cu`/`.cuh` is mostly 1:1-translatable CUDA with a handful of fault-class fixes. Mirage's GPU code is not: it is CUTLASS/CuTe templates + raw PTX tensor-core/async/TMA inline asm. A compat header cannot alias `mma.sync...`/`wgmma`/`cp.async`/`cute::`/CUTLASS collectives to anything on ROCm.
- Strategy B does not apply (not a torch extension).
- The honest path to ROCm would be an AMD-native reimplementation of (1) Mirage's runtime kernel library (the `persistent_kernel/tasks/*` megakernel building blocks) against CK/ck_tile + MFMA, and (2) the CUDA code generator/transpiler so it emits HIP instead of CUDA, plus (3) replacing NVSHMEM, NVRTC, and the on-device fingerprint verifier kernels. That is a ground-up backend, not a port, and is out of scope for MOAT's one-repo-at-a-time mechanical-port-first model.

## CUDA surface inventory
GPU code lives in three layers; all three are NVIDIA-locked.

1. Runtime library (the actual `pip install` artifact, `mirage_runtime`) -- 15 `.cu` + CUTLASS-using `.cc`.
   - On-device "fingerprint" verifier: the superoptimizer executes candidate tensor programs on GPU to check equivalence (`src/kernel/cuda/*.cu`: matmul, rms_norm, reduction, element_*, customized, all_reduce; `src/threadblock/cuda/{matmul,element_unary,input_executor}.cu`). These pull `cutlass/` headers (e.g. `src/kernel/cuda/customized_kernel.cu`, `matmul.cu` via the threadblock runtime). Gated on `MIRAGE_FINGERPRINT_USE_CUDA`.
   - `src/utils/cuda_helper.cu`: `cudaDataType_t`, `CUDA_R_16F/32F/...` -- these specific symbols map cleanly to hipDataType, but they are the trivial part.
   - `src/kernel/device_memory_manager.cu`: `cudaMalloc`/`cudaSetDevice`/streams -- 1:1 to HIP, again trivial relative to the rest.

2. Transpiler runtime headers (`include/mirage/transpiler/runtime/**`, shipped into the emitted CUDA) -- CUTLASS/CuTe everywhere: `threadblock/hopper_matmul.h`, `threadblock/pipeline.h`, `threadblock/blackwell_pipeline.h`, `transpiler_tb_hopper.cc`, `transpiler_tb_blackwell.cc`. 81 files include `cutlass/`/`cute/`.

3. Persistent-kernel (MPK) device task library (`include/mirage/persistent_kernel/tasks/**`) -- the megakernel building blocks, the perf core. NVIDIA-arch-specific by directory:
   - `ampere/` (sm80, the OLDEST supported): even here, `mma.cuh` is raw PTX `asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ...")` tensor-core ops; `linear_cutlass.cuh`, `smem_layout.cuh` use CUTLASS; `copy_sm80.cuh` uses `cp.async`. On AMD these need MFMA `__builtin_amdgcn_mfma_*` + LDS rewrites; PTX asm does not assemble for amdgcn.
   - `hopper/` (sm90): `wgmma.cuh`, `tma_2d/3d/4d.cuh`, `cute/hopper/*` warp-specialized GEMM. Hopper-only (wgmma, TMA, warpgroup) -- no AMD analog.
   - `blackwell/` (sm100): `sm100_*`, `tcgen`/`tmem`, 2-SM MLA. Newest NVIDIA -- no AMD analog.
   - `tma.cuh`, `tma_*`: Tensor Memory Accelerator (Hopper+). No AMD equivalent; would be plain loads/CK pipelines.

Other NVIDIA-locked GPU dependencies:
- NVRTC: `python/mirage/mpk/cuda_utils.py` imports `from cuda.bindings import driver, nvrtc` and uses the CUDA driver + NVRTC (this is the cudf/jitify lesson: would require hipRTC + the HIP driver API). `cuda-python` is a hard `requirements.txt` dep.
- Runtime nvcc invocation: `python/mirage/kernel.py` (JIT path) and `python/mirage/mpk/persistent_kernel.py` shell out to `shutil.which("nvcc")` and compile the generated `.cu` with `-arch=sm_90a -gencode=arch=compute_90a,code=sm_90a`, `-arch=sm_100a`, `-DMPK_ENABLE_TMA`, `-DMIRAGE_GRACE_HOPPER/BLACKWELL`, `-lcublas`. There is no codegen branch that emits HIP or targets gfx*.
- NVSHMEM: multi-GPU all-reduce in the megakernel links `-lnvshmem_host -lnvshmem_device` and bakes `io_category="nvshmem_tensor"` into the API (README). AMD analog would be ROCSHMEM, a separate undertaking.
- cuBLAS: linked by the JIT (`kernel.py`: `-lcublas`).
- Warp/lane intrinsics: ~18 `__shfl`/`__ballot`/`/32`/`& 31`/`0xffffffff` hits in `ampere/` alone -- the standard wave32-vs-wave64 fault class -- but this is moot given the PTX/CUTLASS blockers above.

## Risk list (why even a "just the runtime library" subset is not tractable)
- CUTLASS does not port to ROCm (PORTING_GUIDE, 2026-05-30): even the host-side fingerprint verifier `.cu` includes CUTLASS threadblock templates, so the minimal `pip install`-able library does not compile under hipcc without reimplementing those kernels against CK.
- Raw PTX tensor-core asm in the lowest-common-denominator `ampere/mma.cuh` -- not translatable by a compat header; needs an MFMA rewrite.
- NVRTC -> hipRTC (cudf jitify class): the runtime codegen pipeline is built on `cuda.bindings.nvrtc` + the CUDA driver module API; a HIP path needs hipRTC and the HIP module API and a HIP-emitting code generator.
- nvcc-at-runtime with `sm_90a`/`sm_100a`/`compute_*` hardcoded in two Python files; no gfx target path; `-arch=native` fallback still drives nvcc, not hipcc.
- NVSHMEM (multi-GPU) is in the public API surface and the megakernel link line.
- Wave32/wave64: real and pervasive, but irrelevant until the above are solved.
- Scope/footprint: a real port rewrites the kernel library (dozens of `.cuh` across ampere/hopper/blackwell/cute), the transpiler runtime headers, the CUDA codegen, and the NVRTC/nvcc driver -- i.e. essentially all GPU-facing code. This is an AMD-native backend project, not a MOAT mechanical port.

## What would make it portable (for the record, not proposed here)
A correctness-only subset (USE_CUDA fingerprint verifier + a HIP-emitting transpiler producing plain HIP without CUTLASS/TMA/wgmma, single-GPU, no NVSHMEM, hipBLAS for GEMM) could in principle run on gfx90a, but it would (a) require writing a new HIP transpiler backend and replacing the CUTLASS-based verifier kernels with CK/hipBLAS, and (b) abandon the entire MPK perf story (the megakernel is the project). That is upstream-scale feature work, not a port, and only upstream can reasonably own it.

## File-by-file change list
N/A -- recommending skip; no fork created, no code edited.

## Build commands (gfx90a)
N/A -- no tractable HIP configure/build exists. (For reference, the CUDA build is `pip install -e . -v` with nvcc + CUTLASS submodule + Rust + Z3; there is no `-DUSE_HIP`-style switch and adding one does not address the CUTLASS/PTX/NVRTC blockers.)

## Test plan
N/A for a port. For the record, the real GPU suites (all NVIDIA-arch-gated) are:
- `tests/runtime_python/` -- per-kernel correctness vs PyTorch references (`test_rmsnorm.py`, `test_attention/`, `test_linear/`, `test_paged_attention.py`, etc.); each builds a `CUDAExtension` via `tests/runtime_python/setup.py` with `-gencode=...sm_80/sm_90a/sm_100`. Subdirs `hopper/`, `blackwell/`, `cute/` are arch-specific.
- `tests/ci-tests/run_python_tests.sh` + `run_ci_tests_qwen3.sh` -- end-to-end Qwen2.5/Qwen3 megakernel latency/output (`.github/workflows/gpu-tests.yml` runs these on `gpu-nvidia` runners).
- `tests/python/{test_tensor_program.py,test_packaging.py}` -- frontend/superoptimizer (CPU-ish but still constructs CUDA graphs).
None of these can run on AMD without the (nonexistent) HIP backend.

## Inter-project deps
None to record via set-deps. Mirage bundles CUTLASS, nlohmann/json, and Z3 as git submodules and pulls cuda-python/torch/triton from pip; it does not depend on any other MOAT project, and (being skipped) nothing should depend on it.

## Open questions
- None blocking the decision. If MOAT later wants an AMD-native tensor-program superoptimizer, that is a from-scratch backend effort (CK/ck_tile kernel library + HIP transpiler + hipRTC + ROCSHMEM) better proposed to upstream than pursued as a mechanical MOAT port.
