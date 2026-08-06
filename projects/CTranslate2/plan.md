# CTranslate2 -- ROCm/HIP port plan (lead platform: linux-gfx90a)

## Project

- Name: CTranslate2
- Upstream: https://github.com/OpenNMT/CTranslate2
- Default branch: master
- Clone analyzed at: 5dfc5d8 ("Upgrade CI system dependencies and reduce build time (#2056)", 2026-05-29), version v4.7.2 line.
- Domain: C++/Python inference engine for Transformer models (NMT, Whisper, LLM decoding). CPU (MKL/oneDNN/Ruy/OpenBLAS) + GPU (CUDA, and an existing HIP backend) execution with quantization (int8/int16/fp16/bf16/int4-AWQ).

## Existing AMD support: MATURE upstream HIP backend -> disposition = VERIFY / MODERNIZE on gfx90a (proceed, do NOT skip)

CTranslate2 already ships a real, actively-maintained ROCm/HIP backend. This is NOT a fresh CUDA->HIP conversion.

Evidence:
- `option(WITH_HIP "Compile with AMD HIP GPU backend" OFF)` (CMakeLists.txt:17); full HIP branch CMakeLists.txt:686-771 using `enable_language(HIP)` + `set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE HIP)` -- i.e. exactly the PORTING_GUIDE Strategy A / compat-header approach.
- Compat-header aliasing under `CT2_USE_HIP` in `src/cuda/utils.h`, `src/cuda/primitives.cu`, `src/cuda/allocator.cc`, `src/cuda/random.h`, `src/cuda/helpers.h` (`#define cublasSgemm hipblasSgemm`, `#define cub hipcub`, `#define curand_init hiprand_init`, etc.).
- `find_package(hiprand|hipblas|rocprim|rocthrust|hipcub)`, links `hiprand roc::hipblas roc::rocprim roc::rocthrust hip::hipcub` (CMakeLists.txt:702-769).
- Official build path documented: `docs/installation.md:117,127` (`-DWITH_HIP=ON`), `docker/Dockerfile_rocm` (ROCm 7.2.1), `python/tools/prepare_build_environment_{linux,windows}_rocm.sh`.
- Ships official ROCm Python wheels (README.md:61; CI job `build-python-wheels-rocm`, ci.yml:235).
- Landed v4.7.0 (2026-02-03, #1989); ROCm bumped 7.2->7.2.1 in v4.7.2 (#2030, 2026-05-18). Maintained, not bitrot.

Why this is still a MOAT target (proceed as verify/modernize, per PORTING_GUIDE "Before porting" + the GPUMD/fused-ssim lessons "do NOT auto-skip; build+GPU-validate the CURRENT code"):
1. The upstream HIP build targets RDNA only. Every arch list is `gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201` (Dockerfile_rocm:70, prepare_build_environment_linux_rocm.sh:43). gfx90a (CDNA2 / MI250X, our LEAD platform, wave64) is ABSENT from every list, the docker images, and CI.
2. CI never runs GPU tests on ROCm at all -- the `build-python-wheels-rocm` job compiles in a CPU-only manylinux/cibuildwheel container and the docker job only builds images (ci.yml:235-274, 424-472). There is zero GPU execution of the ROCm path upstream. So the HIP backend has NEVER been GPU-validated, on any arch, let alone CDNA2/wave64.
3. There is a concrete, identifiable wave64 correctness risk on the live HIP path: `src/cuda/helpers.h:398` `#define C10_WARP_SIZE 32` is UNCONDITIONAL (no `__GFX9__` / runtime branch), and drives `get_block_size()` (helpers.h:407) and the hand-rolled `block_reduce()` (helpers.h:411-455) used by `softmax_gpu.cu` and `quantize_gpu.cu`. This is the PORTING_GUIDE "Warp size" fault class (the pytorch/FBGEMM `C10_WARP_SIZE` lesson). It may or may not produce wrong results (the reduction is shared-memory-partition-based, `__syncthreads`-bracketed, not `__shfl`-based -- see Risk list), but it is unverified on wave64 and is the prime suspect if any GPU test fails.

MOAT deliverable: make `-DCMAKE_HIP_ARCHITECTURES=gfx90a` build, GPU-validate the gtest suite on gfx90a (MI250X), and fix any wave64 fault uncovered (most likely a `C10_WARP_SIZE` per-arch fix). Keep the change minimal and NVIDIA-safe; the existing RDNA wheels must keep working. This is the finish/modernize disposition, not already-supported-skip.

If, after building, the full GPU gtest suite passes on gfx90a with NO source change (plausible -- the warp-sensitive code is narrow), then the MOAT contribution is the gfx90a build-enablement (add gfx90a to the arch list / confirm it builds) plus the first recorded CDNA2 GPU validation, and the disposition collapses toward "already-supported, now also verified on gfx90a". Decide that only after the validator runs on hardware; do NOT pre-skip.

## Build classification: pure CMake (Strategy A) -- already implemented upstream

- No `find_package(Torch)`, no `torch.utils.cpp_extension`, no `CUDAExtension`. The Python module (`python/setup.py`) is a pybind11 wrapper that links the prebuilt C++ `ctranslate2` shared lib; the GPU code is a standalone CMake build. => Strategy A.
- The HIP path is ALREADY the canonical Strategy A recipe (compat header + `LANGUAGE HIP` on the `.cu` set). The port work is therefore verification + targeted wave64/arch fixes, NOT introducing Strategy A.

## Port strategy: A (compat-header, already in place) + gfx90a enablement + wave64 audit

Rationale: the heavy lifting (compat header, CMake HIP language, library swaps, build-env scripts) is done and shipping. The remaining surface is (a) confirm/extend the build to gfx90a without churning the RDNA arch list, and (b) the wave64 fault classes that an RDNA-only project never had to confront. No port-vs-rewrite tradeoff for hot kernels here: the GPU GEMM (the perf-critical path) is delegated to hipBLAS (`primitives.cu`), not a hand-tuned CUTLASS/CuTe kernel; the AWQ-int4 and FlashAttention CUTLASS/sm80 kernels are explicitly EXCLUDED from the HIP build (see CUDA surface), so there is no CUTLASS-on-ROCm question to answer (cf. the PORTING_GUIDE rule "CUTLASS does NOT port to ROCm").

## CUDA surface inventory (and HIP disposition of each)

CMake `CUDA_SOURCES` (CMakeLists.txt:230-258) are marked `LANGUAGE HIP`. On the HIP build the following are REMOVED (CMakeLists.txt:708-720): `src/ops/awq/{gemm,gemv,dequantize}_gpu.cu` and their CPU dispatchers. `flash_attention_gpu.cu` and `nccl_ops_gpu.cu` STAY but compile to no-op/throw stubs (guards confirmed).

Custom kernels (live on HIP):
- `src/cuda/primitives.cu` -- GEMM via cuBLAS->hipBLAS (`cublasSgemm`/`cublasGemmEx`/`cublasGemmStridedBatchedEx` aliased to `hipblas*`), elementwise/transform via rocThrust. No warp intrinsics. Risk: low.
- `src/ops/*_gpu.cu` (alibi_add, bias_add, concat_split_slide, conv1d, dequantize, gather, gumbel_max, layer_norm, mean, median_filter, rotary, tile) -- elementwise/grid-stride kernels; no `__shfl`/`__ballot`. Risk: low (no wave-size dependence). `conv1d_cudnn_gpu.cu` is CUDA-only (cuDNN), not in HIP build.
- `src/ops/{softmax,quantize}_gpu.cu` -- use the LOCAL `ctranslate2::cuda::block_reduce` + `ilp_reduce` + `get_block_size` from `helpers.h`, which are keyed on `C10_WARP_SIZE 32`. PRIMARY wave64 risk (see Risk list).
- `src/ops/{topk,topp_mask,multinomial}_gpu.cu` -- reductions/sorts via `cub::BlockReduce`/`BlockScan`/`BlockRadixSort` templated on `num_threads`; on HIP `cub`==`hipcub`->rocPRIM, which is wave-size-aware internally. Risk: low-medium (rocPRIM block primitives handle wave64; `topp_mask` `max_num_classes = num_threads*32` is a capacity constant, not a warp op).
- `src/cuda/random.cu` -- curand->hiprand Philox states. Risk: low.

Warp/wave intrinsics found (whole tree):
- `src/cuda/helpers.h`: `C10_WARP_SIZE 32` (LIVE on HIP -- the one to fix/verify).
- `src/ops/awq/gemv_gpu.cu`: `#define WARP_SIZE 32` + `__shfl_down_sync(0xffffffff,...)` -- EXCLUDED from HIP build (moot).
- `include/ctranslate2/ops/flash-attention/utils.h`: `__shfl_xor_sync(uint32_t(-1),...)` -- flash-attention EXCLUDED from HIP build (moot). NOTE: if FlashAttention were ever enabled on HIP, the `uint32_t(-1)` mask would fail to compile (HIP `__shfl_*_sync` static_assert sizeof(mask)==8; PORTING_GUIDE AutoDock-GPU/raft lesson) AND the 32-lane reduction would be wave64-wrong. Out of scope while WITH_FLASH_ATTN is incompatible with WITH_HIP.

Libraries / runtime (all already mapped upstream):
- cuBLAS -> hipBLAS (`primitives.cu`, `utils.h`). hipBLAS v1 enum spellings in use (HIPBLAS_OP_*, HIPBLAS_COMPUTE_*); verify `cublasGemmEx`/`StridedBatchedEx` compute-type + algo enums resolve on ROCm 7.2.1 hipBLAS.
- cuBLASLt -> NOT used on HIP (no hipblasLt link). int8 GEMM goes through `cublasGemmEx`(`CUBLAS_COMPUTE_32I`)->hipblasGemmEx.
- cuDNN -> MIOpen: NOT used. cuDNN path (`conv1d_cudnn_gpu.cu`, `WITH_CUDNN`) is CUDA-only; HIP uses the generic `conv1d_gpu.cu` (im2col + GEMM). No MIOpen dependency.
- cuRAND -> hipRAND (`random.h`, kernel-side `hiprand_kernel.h`).
- Thrust -> rocThrust (`THRUST_CALL`, `thrust::transform/copy/permutation_iterator`); CUB -> hipCUB (`#define cub hipcub`). Both header-only on hipcc default path. (PORTING_GUIDE: rocThrust/hipCUB are drop-ins.)
- CUB CachingDeviceAllocator -> hipcub `util_allocator.hpp` `cub::CachingDeviceAllocator` (allocator.cc:12,55); async path `cudaMallocAsync`->`hipMallocAsync` gated on memory-pool support. GPU caching allocator works on HIP.
- NCCL -> RCCL (`rccl/rccl.h`) ONLY under `CT2_WITH_TENSOR_PARALLEL`, which is incompatible with WITH_HIP (CMakeLists.txt:687). Tensor parallelism is out of scope on HIP.
- CUTLASS (third_party/cutlass) -> only consumed by AWQ-int4 + FlashAttention, both excluded from HIP. No CUTLASS-on-ROCm work (correctly avoided).
- Streams/events: `hipStream_t`/`cudaStreamSynchronize`->`hipStreamSynchronize` aliased; no textures/surfaces; no managed memory beyond the optional async pool.

Device-capability gates: `src/cuda/utils.cc:205-219` hardcodes `gpu_supports_int8 / gpu_has_int8_tensor_cores / gpu_has_fp16_tensor_cores = true` on HIP for all ROCm 7 archs. Correct for gfx90a (CDNA2 has FP16/BF16 matrix cores + INT8). So the int8/fp16/bf16 GPU code paths and tests are exercised on gfx90a.

## Risk list

1. wave64 / `C10_WARP_SIZE 32` (PRIMARY; PORTING_GUIDE "Warp size", pytorch/FBGEMM lesson). `helpers.h` hardcodes 32. ANALYSIS: `block_reduce` is NOT a `__shfl` warp-shuffle reduction -- it writes all `blockDim.x` values to `__shared__`, `__syncthreads()`, then has the first `C10_WARP_SIZE` threads each reduce a contiguous `C10_WARP_SIZE`-stride group from shared memory (`smem[lane*C10_WARP_SIZE+i]`), then a final single-thread reduce. The partition math is self-consistent for `blockDim.x` a multiple of 32 with `blockDim.x/32 <= 32`, regardless of hardware wave width, because it reads shared memory the whole block wrote (not lane registers). So the reduced VALUE is plausibly CORRECT on wave64. The genuine concern is the `__syncwarp(mask)` at helpers.h:435 executed by only lanes 0..(blockDim.x/32-1) of a 64-lane wavefront -- a partial-mask warp sync (cf. PORTING_GUIDE MPPI-Generic warp-synchronous-tail hazard). On gfx90a `get_block_size` can also return 32 (half a wavefront active). VERDICT: must be GPU-tested; if softmax/quantize/layer_norm/rms_norm GPU tests fail or are non-deterministic, fix by making `C10_WARP_SIZE` a per-arch constant (64 on `__GFX9__`, else 32) per the PORTING_GUIDE template, and/or drop the `__syncwarp` tail in favor of the `__syncthreads`-tree (MPPI-Generic fix). Keep CUDA byte-identical via `#if defined(CT2_USE_HIP)`.
2. Determinism on wave64: softmax/quantize use a hand-rolled tree; topk/topp/multinomial use rocPRIM block ops. Run each GPU FP test twice and assert bit-identical (the PORTING_GUIDE determinism bar) to catch any unsynced wave64 race even if the mean passes.
3. hipBLAS enum/algo coverage on ROCm 7.2.1: `cublasGemmEx`/`cublasGemmStridedBatchedEx` with `HIPBLAS_COMPUTE_16F/32F/32I` + `HIPBLAS_GEMM_DEFAULT`. Verify they compile and the int8 (`COMPUTE_32I`) and true-fp16 (`COMPUTE_16F`) GEMM paths run on gfx90a (the `use_true_fp16_gemm` path, utils.cc:260). Low risk (PORTING_GUIDE: hipBLAS mirrors cuBLAS name-for-name) but the v1-vs-v2 enum surface is worth a compile check.
4. Arch-list churn (PORTING_GUIDE CudaSift/Gpufit lesson): do NOT replace the RDNA arch list with a literal `gfx90a`. The build already reads `-DCMAKE_HIP_ARCHITECTURES`; the lead validation just passes `-DCMAKE_HIP_ARCHITECTURES=gfx90a`. If a source fix is needed, gate it on arch macros, not a hardcoded literal, so followers (gfx1100/gfx1151 -- which upstream ALREADY targets) need no further edit. Ideal end state: one curated commit builds for gfx90a AND the upstream RDNA set with only the cache var changing.
5. `conv1d` on HIP uses the generic im2col+GEMM path (cuDNN path is CUDA-only). Validate Conv1D GPU op correctness (it is in ops_test). Low risk.
6. AWQ-int4 / FlashAttention absent on HIP: not a regression (they were never on the HIP path); no tests reference them by name (grep of tests/ for awq|int4|flash = empty). Document as expected-unsupported on HIP, do not treat as a defect.
7. ROCm 7.2.1 host-toolchain quirks already handled upstream (`-Wno-deprecated-literal-operator`, amdclang/amdclang++). The build uses `amdclang++` as CXX; expect the usual clang strictness (two-phase lookup, std::min/max) but the code already compiles for RDNA, so host C++ is clean.

## File-by-file change list (expected; minimal, NVIDIA-safe, RDNA-safe)

Best case (validator passes on gfx90a with no source change): zero source files; the deliverable is the recorded gfx90a GPU validation + (optionally) adding gfx90a to the documented arch lists in `docker/Dockerfile_rocm` / `prepare_build_environment_linux_rocm.sh` (NOTE: editing those is doc/build-config only and does NOT affect the library HEAD used for validation; weigh against churn). The core library build already accepts `-DCMAKE_HIP_ARCHITECTURES=gfx90a` with no edit.

If wave64 fault confirmed (most likely target):
- `src/cuda/helpers.h` -- replace `#define C10_WARP_SIZE 32` with a per-arch constant: 64 when `defined(CT2_USE_HIP) && defined(__GFX9__)`, else 32 (CUDA and RDNA unchanged). Re-check `get_block_size` lower-bound and `block_reduce` partition/`__syncwarp` for wave64 (drop the warp-synchronous tail under HIP if it races, letting the `__syncthreads` path finish). Single file, guarded, CUDA byte-identical.
- Possibly `src/ops/softmax_gpu.cu` / `src/ops/quantize_gpu.cu` only if a block-size launch assumption needs adjusting; prefer fixing it once in `helpers.h`.

Do NOT touch: the awq/flash-attention sources (excluded), the CUDA-only cuDNN/tensor-parallel paths, CPU backends, or `.github/workflows` (per CLAUDE.md: no GHA edits; they churn HEAD and trip the regression guard).

## Build commands (gfx90a)

Dependencies (ROCm 7.2.1 + the CPU math backend the runtime needs). Install via apt as needed:
`rocm-hip-runtime-dev hipblas-dev hipcub-dev hiprand-dev rocprim-dev rocrand-dev rocthrust-dev`, plus a CPU GEMM backend (oneDNN/OpenBLAS). For validation the simplest CPU backend is OpenBLAS (`-DWITH_OPENBLAS=ON`) to avoid the MKL+oneDNN docker dance; MKL is x86-perf-only and not required for correctness tests.

Configure + build (Strategy A, configurable arch, BUILD_TESTS on):
```
cmake -S projects/CTranslate2/src -B projects/CTranslate2/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=amdclang -DCMAKE_CXX_COMPILER=amdclang++ \
  -DWITH_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DWITH_MKL=OFF -DWITH_OPENBLAS=ON -DOPENMP_RUNTIME=COMP \
  -DBUILD_TESTS=ON -DBUILD_CLI=OFF \
  -DCMAKE_CXX_FLAGS="-Wno-deprecated-literal-operator" \
  -DCMAKE_HIP_FLAGS="-Wno-deprecated-literal-operator"
cmake --build projects/CTranslate2/build -j"$(nproc)"
```
(Submodules: the depth=1 clone did NOT init submodules; before configuring run `git -C projects/CTranslate2/src submodule update --init --recursive` so googletest/cpu_features/ruy/spdlog/cxxopts are present. third_party/thrust + third_party/cutlass are only needed for the CUDA build -- not the HIP build -- but `git submodule update` will fetch them; that is fine, they are unused on HIP.)
Same source builds gfx1100/gfx1151 by changing only `-DCMAKE_HIP_ARCHITECTURES` (followers; upstream already targets these RDNA arches).
Optional CPU-only compile smoketest of the HIP TUs: docker image `rocm/dev-ubuntu-24.04:7.2.4-complete` (compile check only, NOT a validation gate -- it cannot observe a wave64 fault).

## Test plan

Real GPU test suite (gtest), the validator's gate:
- Build target `ctranslate2_test` (tests/CMakeLists.txt; gated by `-DBUILD_TESTS=ON`). It is parametrized per device: every op/layer/primitive/storage test is instantiated for both `Device::CPU` and `Device::CUDA`, and on a `WITH_HIP=ON` build `Device::CUDA` runs on the AMD GPU (the `#ifdef CT2_WITH_CUDA` CUDA instantiations are compiled because the HIP build defines `CT2_WITH_CUDA`, ops_test.cc:1417).
- GPU test surface = the `CUDA`-suffixed gtest instantiations:
  - `ops_test.cc`: `OpDeviceTest`/`OpDeviceFPTest` over FLOAT32 (1e-2 tol on FP16, 4e-2 on BF16) -- covers Add/Mul/Sub/Tile/Concat/Split/Mean/Gather/Transpose/Softmax/LayerNorm/RMSNorm/Quantize/Dequantize/TopK/TopP/Multinomial/Conv1D/Rotary/Alibi/GumbelMax/Gemm etc. This is the core wave64 validation surface (softmax/quantize/layer_norm/rms_norm/topk/topp directly exercise the reductions).
  - `layers_test.cc`: `LayerDeviceFPTest` (CUDA, FP32/FP16/BF16) -- attention/feedforward/embedding layers end-to-end on GPU.
  - `primitives_test.cc`, `storage_view_test.cc`: `Device::CUDA` -- the hipBLAS GEMM primitive + device memory ops.
  - `translator_test.cc`, `model_test.cc`, `attention_test.cc`, `batching_test.cc`, `decoding_test.cc`: full inference on GPU using the bundled tiny models under `tests/data/models` (and `tests/data/marian`, `tests/data/audio`). These exercise the whole stack (GEMM + softmax + layer norm + decoding) on the AMD GPU.
- Run SERIALLY on the single assigned GPU (PORTING_GUIDE MPPI lesson): `cd projects/CTranslate2/build && ctest --output-on-failure` (NOT `-jN`; parallel test procs on one GPU cause spurious timeouts). Or run `./tests/ctranslate2_test` directly; filter GPU-only with `--gtest_filter='*CUDA*'` for a fast wave64 pass, then run the full binary for the inference/model tests.
- Determinism check (PORTING_GUIDE bar): run `--gtest_filter='*CUDA*'` twice and diff; any non-determinism in softmax/quantize fingerprints the `C10_WARP_SIZE`/`__syncwarp` wave64 hazard.
- Some bundled test models may require conversion artifacts; if `model_test`/`translator_test` need a downloaded model the bundled `tests/data/models` set covers the core path -- treat any missing large model as a test-data gap (PORTING_GUIDE RXMesh rocker-arm lesson), not a port failure, and validate the op/layer/primitive GPU suite as the correctness gate.

Non-GPU regression set that must not regress:
- The `CPU`-suffixed instantiations of the same gtest binary (all op/layer/primitive/storage tests on `Device::CPU`) -- they share the source and must stay green. Building `WITH_HIP=ON` must not perturb the CPU backends (the CMake explicitly re-marks the CPU `.cc` as `LANGUAGE CXX`, CMakeLists.txt:726-754, so they are untouched by HIP).

## Inter-project MOAT deps

None. CTranslate2 vendors all GPU deps it needs (rocThrust/hipCUB/rocPRIM are ROCm system packages, googletest/spdlog/cpu_features/ruy/cxxopts are submodules). It does not depend on any other MOAT project (no rmm/raft/etc.). `set-deps` not required (leave empty).

## Open questions

1. Does the full GPU gtest suite pass on gfx90a (wave64) with the upstream `C10_WARP_SIZE 32` unchanged? (Analysis says the partition-based reduction is plausibly correct, but the `__syncwarp(partial-mask)` tail on a 64-lane wavefront is the live risk -- resolved only by running on hardware.) This decides whether the port is a 1-file wave64 fix or a pure gfx90a-enablement+validation.
2. If it passes unchanged: is the right MOAT deliverable (a) just the recorded first-ever CDNA2 GPU validation (no fork commit needed beyond confirming `-DCMAKE_HIP_ARCHITECTURES=gfx90a` builds), or (b) also adding gfx90a to the upstream documented arch lists (Dockerfile_rocm / prepare_build_environment_linux_rocm.sh)? Lean (a) to avoid HEAD churn; surface (b) to jeff as an optional upstream-value-add at the PR gate.
3. hipBLAS `cublasGemmEx`/`StridedBatchedEx` compute-type + algo enum coverage on ROCm 7.2.1 -- confirm at first compile (expected fine).
