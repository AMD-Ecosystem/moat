# oneflow notes

ROCm/HIP port of oneflow's CUDA GPU backend. Lead platform linux-gfx90a (MI250X,
CDNA2, wave64), ROCm 7.2.1. Strategy A (pure CMake: USE_HIP option, one compat header,
.cu retagged LANGUAGE HIP). Fork: https://github.com/AMD-Ecosystem/oneflow (branch moat-port).

Older round-by-round records (2026-05-31 through the 2026-11-11 reviews and porter rounds: every review, porter round and validation before the c741328b7 baselines below) moved verbatim to `notes-archive.md` in this folder on 2026-11-14, because this file had reached the repository's 1 MB file limit. Search there for a dated section that a deferral, a commit or an older note cites.

## Key architectural decision: WITH_CUDA defined FOR the HIP build
oneflow's entire GPU backend is gated on `#ifdef WITH_CUDA` (204 files) and `kCUDA` is the
GPU DeviceType. Rather than introduce a `kROCM` enum (proto + every registry + thousands
of switch arms), the port keeps `BUILD_CUDA=ON` so `-DWITH_CUDA` is defined and `kCUDA`
keeps meaning "the GPU backend", and adds a parallel `USE_HIP` that routes the toolkit,
libraries, and the `.cu` source language to ROCm. `device="cuda"` in Python maps to the
HIP backend (mirrors how torch-ROCm keeps `torch.cuda`). AutoDock-GPU lesson, framework
scale.

## Monolith constraint
All ~220 `.cu` + all `.cpp` link into ONE `liboneflow.so` (`oneflow_add_library(oneflow
SHARED ${of_all_obj_cc})`); there is no per-op sub-target. The whole lib must compile +
link before any op runs, so the validation slice (ep/cuda/primitive tensor ops) is a
VALIDATION scope, not a build scope.

## Build script
`projects/oneflow/src/build_rocm.sh [config|tp|oneflow|all]`. gfx90a, -j16, ccache.
Configure flags: `-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DBUILD_CUDA=ON
-DWITH_CUTLASS=OFF -DWITH_MLIR=OFF -DBUILD_RDMA=OFF -DBUILD_CPP_API=OFF -DBUILD_HWLOC=OFF
-DBUILD_TESTING=OFF -DTREAT_WARNINGS_AS_ERRORS=OFF -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF`.
HIP compiler auto-detected at /opt/rocm/lib/llvm/bin/clang++ (no -DCMAKE_HIP_COMPILER
needed). Third-party (LLVM-16 prebuilt download, protobuf, grpc, opencv, oneDNN, half,
eigen, flatbuffers, etc.) builds from source via oneflow's own cmake/third_party; the CUDA
deps (NCCL/CUTLASS/flash-attn/cub) are skipped under USE_HIP (RCCL/hipCUB/rocThrust are
header drop-ins; CUTLASS/flash/MIOpen deferred).

## Followers (gfx1100/gfx1151)
Same commit; only `-DCMAKE_HIP_ARCHITECTURES=<arch>` changes (the wavefront width
OF_GPU_WARP_SIZE is derived from the arch in cmake/cuda.cmake, no source edit). RDNA is
wave32 -- the warp-size fixes below are arch-unified and correct for both 32 and 64.

## Where things are
- `oneflow/core/device/cuda_to_hip.h` -- the single compat header. Force-included on every
  HIP TU (`CMAKE_HIP_FLAGS -include ...`). Aliases the cuda*/cublas*/curand*/cufft*/cusolver*
  symbols oneflow uses to hip*; complex types/conv; CUDART_INF/NAN -> HIP_INF/NAN;
  `__trap`->abort; defines `__CUDACC__` and device-pass `__CUDA_ARCH__=1`; 64-bit warp mask
  `OF_SHFL_FULL_MASK`; libc-before-hip include order; handle-type aliases for
  cudnn/cusolver/cublasLt (operations deferred).
- `oneflow/core/device/hip_compat/` -- forwarding shim headers (cuda_runtime.h, cublas_v2.h,
  curand.h, cufft.h, cuComplex.h, nccl.h -> rccl/rccl.h, cub/cub.cuh -> `namespace cub =
  hipcub`, ...) on the HIP include path ONLY. The CUDA build never sees them.

## Fault classes -- status

### Monolith compile progress (whole-tree .cu under HIP)
A keep-going `ninja -k 0` of all 220 .cu objects drove the fault-finding. Starting from
138/220 compiling with just the compat header, the fixes below cleared the bulk; the
genuinely-deferred cuDNN/cuBLASLt/wmma TUs (~12) are list(FILTER EXCLUDE)-ed from the HIP
monolith glob in cmake/oneflow.cmake. The 30-TU ep/cuda/primitive validation slice all
compiles as HIP.

### CLEARED (validated by compiling the ep/cuda/primitive slice as HIP, 30/30 TUs)
1. wave64 warp size. `kWarpSize`/`kCudaWarpSize` were hardcoded 32 in
   core/cuda/{softmax,layer_norm,rms_norm}.cuh and core/device/cuda_util.h. Fixed to
   `OF_GPU_WARP_SIZE` (64 on gfx9xx, 32 on RDNA/CUDA), injected from the target arch by
   cmake/cuda.cmake into BOTH the host and device passes of every .cu via CMAKE_HIP_FLAGS.
   This is mandatory because kWarpSize flows from host launch config into device kernel
   template args, so it MUST be one compile-time constant -- `__GFX9__` is device-pass-only
   (proven: a static_assert keyed on __GFX9__ fails in the host pass) and cannot be used.
   RmsNormParamGrad's full-warp WarpReduceSum now starts its butterfly at kWarpSize/2 so all
   64 lanes fold in (was hardcoded 16).
2. 32-bit shfl mask. `__shfl_*_sync(0xffffffff, ...)` does not compile on ROCm 7.x
   (static_assert sizeof(mask)==8). Compat header defines `OF_SHFL_FULL_MASK`
   (0xffffffffffffffffULL on HIP, 0xffffffff on CUDA); the reduction headers route through
   it. Warp-tiled shuffles also pass explicit `thread_group_width` (confines the butterfly
   to the group on a 64-lane wavefront; matches the codebase's existing convention).
3. `__CUDA_ARCH__` undefined on HIP. Compat header defines `__CUDA_ARCH__=1` in the device
   pass only (guarded by `__HIP_DEVICE_COMPILE__`), so device-vs-host `#ifdef`s take the
   device branch and numeric `#if __CUDA_ARCH__ >= NNN` intrinsic gates (TanhApprox PTX,
   atomic.cuh native paths) naturally fall to the portable `#else` -- no per-site edits
   needed for the slice. Paired with `__CUDACC__` (defined when `__HIPCC__`) so header-only
   device code is selected. EIGEN_NO_CUDA set so Eigen takes its native HIP path.
4. Missing symbol aliases surfaced by compiling: cuComplex/cuDoubleComplex + conversions,
   cuComplexFloatToDouble/DoubleToFloat, CUDART_INF/INF_F/NAN/NAN_F, __trap, the cufft/
   cusolver enums and types.
5. cuda_pseudo_half.h / cuda_pseudo_bfloat16.h redefinition. HIP's __half (hip_fp16.h) and
   __hip_bfloat16 (hip_bf16.h) already ship the full operator/intrinsic set these files
   emulate for pre-sm53/pre-sm80 CUDA; with __CUDA_ARCH__=1 they would redefine them and
   clash ("__device__ function cannot overload __host__ __device__ function"). Gated both
   bodies with `&& !defined(USE_HIP)`.
6. bf16 type identity. HIP has two bf16 structs: __hip_bfloat16 (modern, what
   __float2bfloat16 etc. return) and hip_bfloat16 (legacy). oneflow's nv_bfloat16/
   __nv_bfloat16 must both alias __hip_bfloat16 so functor return/param types match the
   conversion results (else "no viable conversion from __hip_bfloat16 to hip_bfloat16").
7. More 32-bit shfl/ballot masks tree-wide: layer_norm_gpu_kernel.cu, data_shuffle_kernel.cu
   (per-arch kWarpSize + OF_SHFL_FULL_MASK), batch_norm_kernel_utils.h (WARP_SIZE per-arch +
   WARP_SHFL_XOR default mask 64-bit), lru_cache.cu (kept 32 ways = 32 lanes; 64-bit ballot
   folded with __ffsll/__popcll, lanes 32-63 idle). Also CUDART_INF/NAN, __trap->abort,
   cuComplex* + conversions, CUDA_VERSION defined to 11080, bare curand->hiprand,
   CUDNN_BN_MIN_EPSILON (numeric bound), cufft/cusolver enums.
8. clang strictness: `dst.template Mut(i)` where Mut is non-template -> drop the spurious
   `.template` (the -Wmissing-template-arg-list-after-template-kw family). The cub shim must
   make `cub` a real namespace (`namespace cub { using namespace hipcub; }`) not an alias,
   because ndarray_reduce_impl.cu REOPENS namespace cub to add its own functors.
   layer_norm.cuh MaximizeDynamicSharedMemorySize casts the kernel ptr to const void* for
   hipFuncGetAttributes/hipFuncSetAttribute (gsplat lesson).
9. Host GPU .cpp TUs (ep/cuda/*.cpp, device/cuda_util.cpp, ...) compile as CXX not hipcc, so
   they get the compat header force-included + shim dir + /opt/rocm/include + warp-size macro
   PER oneflow target (set_compile_options_to_oneflow_target), NOT globally -- a global
   force-include leaks into third-party subdir builds (TBB) which lack the ROCm include path.

### Third-party bringup fixes (not HIP-specific, needed for the monolith)
- flatbuffers 1.12.0 hardcodes -Werror; gcc-13 -Wstringop-overflow on libstdc++
  stl_algobase.h breaks its build. PATCH_COMMAND strips the -Werror flags.
- kineto auto-enables a roctracer backend on ROCm and demands ROCTRACER_LIBRARY from a wrong
  path; LIBKINETO_NOROCTRACER=ON under USE_HIP (profiler is off anyway).

### Compile tally (whole oneflow/ .cu tree, gfx90a)
203 of 220 .cu compile as HIP. The 17 deferred TUs (list(FILTER EXCLUDE) in
cmake/oneflow.cmake, each with a genuine library/API gap -- not a mechanical port):
- cuDNN, no MIOpen drop-in: grid_sample_kernel_util.cu (spatial-transformer),
  normalization_kernel.cu (cuDNN batchnorm). cuda_util.h/cudnn_util.h gate `<cudnn.h>` and
  the cudnn error-string helper behind `!USE_HIP`; the handle type is aliased so
  cuda_stream.h parses, but the operations are not. cudnn_conv_util.cpp also deferred.
- cuBLASLt fused-MLP / grouped-GEMM / matmul-bias family: cublas_fused_mlp_kernel.cu,
  cublas_fused_mlp_grad_kernel.cu, cublas_bias_add_relu_matmul_grad_kernel.cu,
  cublas_fused_matmul_bias_add_grad.cu, fused_matmul_bias_kernel.cu,
  fused_matmul_bias_add_relu_dropout.cu, grouped_matmul_bias.cu (hipBLASLt descriptor
  surface + hipblasGemmStridedBatchedEx signature differ from cuBLASLt).
- cuSOLVER: lu_decomposition_kernel.cu (getrf/getrs; deferred to hipSOLVER).
- CUTLASS/CuTe + wmma + TensorRT-flash: fused_dot_feature_interaction_kernel.cu (wmma),
  fused_self_attention_query_mul_key_and_value_kernel.cu (cublasGemmStridedBatchedEx),
  fused_attention_kernels/fused_glu_kernel/fused_multi_head_attention_inference (CUTLASS,
  already -DWITH_CUTLASS=OFF). CUTLASS does not port to ROCm (CK/ck_tile reimplementation).
- libcu++: lru_cache.cu (`cuda::std::semaphore`, no ROCm equivalent).
- P2P/IPC embedding shuffle: one_embedding_embedding_shuffle_p2p_kernel.cu,
  one_embedding_embedding_gradient_shuffle_p2p_kernel.cu.
- nvjpeg/npp image decode (NVIDIA-only) -- gated behind `!USE_HIP`.
The fixable-but-not-fixed bf16x2 broadcast (__float2bfloat162_rn -> a HIP helper in the
compat header) was added, clearing dropout_kernel.cu.

### LINKED + GPU-VALIDATED (continuation session 2)
liboneflow.so (1.3 GB) and the _oneflow_internal pybind module LINK and load; the
ep/cuda/primitive slice GPU-validates against the PyTorch-ROCm oracle. Key decision: the
whole monolith now compiles with the ROCm clang (CC=clang, CXX=clang++) instead of g++ --
oneflow's data_type.h / cuda_stream.h pull the HIP half/bf16/library handle headers (clang
builtins) into thousands of host .cpp, which only parse under clang. cuda.cmake links
hip::host (NOT hip::device, whose -x hip INTERFACE option would recompile every host .cpp
as device code) and roc::rccl (the nccl* collective-comm symbols).

### CLEARED in session 2 (host-.cpp + link + GPU faults)
1. Toolchain: build the monolith with the ROCm clang for C/C++/HIP (build_rocm.sh). The
   clang-only HIP math/library headers are gated on __clang__ in cuda_to_hip.h; the
   platform macro __HIP_PLATFORM_AMD__ is selected for the plain-C++ pass.
2. Eigen 3.90 + clang: Transpositions.h friend operator* calls trt.derived() on a class
   with no derived() -- g++ never instantiates it, clang rejects it. PATCH_COMMAND in
   cmake/third_party/eigen.cmake rewrites it to pass the Transpose directly. Not
   HIP-specific; any clang build needs it.
3. LLVM ABI-breaking-checks mismatch: ROCm clang ships llvm/Config/abi-breaking.h with
   LLVM_ENABLE_ABI_BREAKING_CHECKS=0 on its builtin path (ahead of oneflow's vendored
   LLVM headers =1), so every TU referenced llvm::DisableABIBreakingChecks while the
   linked libLLVMSupport (=1) defines EnableABIBreakingChecks -> liboneflow.so failed to
   load. Fix: LLVMSupportWithHeader INTERFACE_COMPILE_DEFINITIONS
   LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1 (LLVM's sanctioned escape hatch).
4. CPython 3.12: imp module removed -> importlib in framework/unittest.py + sysconfig.py
   (the autotest harness import path). _PyFrameEvalFunction frame param changed to
   _PyInterpreterFrame* in 3.11; custom_eval_frame.c casts the stale-typed hook at the
   registration site (the stack getter is off by default).
5. Excluded-op host references: NewLruCache (lru_cache.cu excluded) guarded under USE_HIP
   in embedding/cache.cpp (kFull cache still works); GridSampleKernelUtil<kCUDA> (excluded
   grid_sample_kernel_util.cu) -> grid_sample_kernel.cpp registers only kCPU on HIP. These
   were the only two dangling symbols (verified via nm -DC scan; everything else is proto,
   resolved by libof_protoobj.so at load).
6. Host cuda* symbols: cuda_util.cpp CudaDriverGetPrimaryCtxActive reimplemented with the
   native HIP driver API (hipDeviceGet/hipDevicePrimaryCtxGetState); CUBLAS_STATUS_LICENSE_ERROR
   and CUDART_VERSION handled; cuda_stream.cpp CheckCublasVersion no-op on HIP (no
   hipblasGetVersion / CUBLAS_VER_MAJOR); cuDNN/cuSOLVER/cuBLASLt/nvjpeg/NVTX host refs
   guarded under !USE_HIP in cuda_stream/env_global_objects_scope/profiler/image_decoder.
7. .template disambiguation (clang -Wmissing-template-arg-list-after-template-kw):
   add_n_kernel.cpp, one_embedding.cpp (pybind array::data).
8. clip_by_value_kernel.cpp: __double2half is device-only on HIP -> __float2half on host.
9. .cuh standalone-compile: oneflow's glob compiles every .cuh as a TU; fused_softmax.cuh,
   one_embedding_data_shuffle.cuh, cublas_fused_mlp_util.cuh are not self-contained (use
   their .cu includer's prior includes) -> excluded from the HIP glob (their code still
   builds inside the .cu that include them).

### CLEARED GPU CORRECTNESS faults (found by running the slice on GPU)
A. hipblasSetMathMode NOT_SUPPORTED on CDNA: hipblas accepts only HIPBLAS_DEFAULT_MATH;
   TENSOR_OP_MATH and DISALLOW_REDUCED_PRECISION_REDUCTION return status 7 and
   OF_CUBLAS_CHECK aborted (fp16 matmul + fp16 reduce). CublasMathModeGuard::SetMathMode is
   a no-op on HIP (rocBLAS picks precision from the data/compute types; no NVIDIA math-mode
   equivalent). cuda_util.cpp.
B. block-SMEM kernels missing __launch_bounds__: LayerNorm/RmsNorm/Softmax Block(SMem|
   Uncached)Impl and the LayerNorm/RmsNorm grad variants are launched at up to 1024
   threads via an occupancy heuristic; on gfx90a, without __launch_bounds__(1024) the
   register allocation is too large and the 1024-thread launch faults (HIP's occupancy API
   still returns >0). Added __launch_bounds__(1024) to all of them (the LayerNorm Uncached
   kernel already had it -- the others were an upstream oversight). core/cuda/{layer_norm,
   rms_norm,softmax}.cuh.
C. CUDNN_BN_MIN_EPSILON: the compat header hardcoded the pre-cuDNN-7.5 floor 1e-5;
   CHECK_GE(epsilon, CUDNN_BN_MIN_EPSILON) then aborted on the common eps=1e-6 layer norm.
   cuDNN 8 (CUDA 11.x, oneflow's target) uses 0.0 -> match it. cuda_to_hip.h.
D. WAVE64 reduction bug (the important one): layer_norm_gpu_kernel.cu LayerNormParamGrad
   (gamma/beta gradient) had an INCONSISTENT session-1 fix -- tile_size was tied to
   OF_GPU_WARP_SIZE (64) while block_dim_x, the launch dim3(32,...), and the shared
   transpose [32][33] all stayed 32, so WarpReduce butterflied from 32 and folded the
   neighbouring threadIdx.y row on a 64-lane wavefront -> weight_grad garbage (max abs diff
   ~21 vs torch). Fix: tile_size is a 32-wide tiling constant (NOT the warp width);
   WarpReduce reduces block_dim_x(=32) lanes via the 4-arg __shfl_down_sync(...,width=32),
   correct on wave32 AND wave64. Lesson: arch-unify the WHOLE kernel (launch dims + shared
   sizes + reduction width), never just the reduction constant.

## GPU validation (test oracle = PyTorch ROCm), session 2 RESULTS
torch 2.13 ROCm (torch.cuda.is_available()==True on HIP) is the oracle. 4 MI250X GCDs; run
SERIALLY on one free GCD via HIP_VISIBLE_DEVICES (rocm-smi --showmemuse to pick). From the
repo python/ tree with `source build/source.sh` (PYTHONPATH=.../python). Build first with
`./build_rocm.sh` then `cmake --build build -j16` (the default target builds liboneflow.so
+ _oneflow_internal + copies the module into python/oneflow/).
PASS (device="cuda" == HIP, fwd+bwd vs torch): test_add (11), test_cast, test_where,
test_masked_fill, test_broadcast_ops, test_transpose, test_softmax, test_layer_norm (4) =
64 passed; test_matmul core (fp32/mm/mv/broadcast/batch/tf32) = 9 passed.
KNOWN NON-ROCm test artifacts (not GPU-correctness): test_matmul fp16 subcase -- the GPU
result is BIT-IDENTICAL to torch (max abs diff 0.0 on controlled input), the pytest fails
only on a numpy/oneflow `__bool__` dtype-interop quirk (np.allclose returns a oneflow
scalar); test_matmul int subcase -- CPU path, op_kernel_not_found (oneflow has no int32
matmul primitive kernel; unrelated to GPU/ROCm).
RUN: cd python/oneflow/test/modules; HIP_VISIBLE_DEVICES=<free> python3 -m pytest
test_add.py ... test_layer_norm.py -q -p no:cacheprovider  (serial, never -n).

## CUDA compile check at c741328b7 (2026-11-12, mi355 host, nvcc 12.8.93)

No-regression compile of the CUDA path for the round 8d470dcc14..c741328b7 (19444bcdf, 3007f7951, c741328b7); no role, no state change, no GPU work. Full CUDA builds of `moat-port` 8d470dcc14 and c741328b7 with the recipe of "CUDA compile check at 8d470dcc14", then a comparison of configure arguments, the bundled-LLVM configure argv and link lines, compile and link commands, and diagnostics. **Result: PASS. Both configure and build with rc 0 and link liboneflow.so and `_oneflow_internal`. The bundled LLVM's configure argv is token-for-token identical (no `--version-script`, no `CMAKE_SHARED_LINKER_FLAGS` argument without USE_HIP), its CMakeCache and all 448 of its link lines are identical, all 1822 compile commands are identical, and the warning set is identical line for line. Nothing was run on an NVIDIA GPU; this host has none.**

### Toolchain and procedure (same as at 8d470dcc14)
- nvcc 12.8.93 (`/usr/local/cuda-12.8`), gcc/g++ 13.3.0, cmake 3.31.6. cuDNN 8.9.7.29 and NCCL 2.32.3 from their cu12 wheels, the same sha256 as at 8d470dcc14 (adf4f59e...7b02, bb94b234...a21c), extracted under the job directory with unversioned `.so` symlinks. `dev-requirements.txt` installed into the job venv; a python3 wrapper dropping `--user`.
- `cmake <src> -C cmake/caches/international/cuda.cmake -DCMAKE_CUDA_ARCHITECTURES=80 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 -DCUDNN_ROOT_DIR=<job>/deps/cudnn -DCMAKE_BUILD_TYPE=Release -DTREAT_WARNINGS_AS_ERRORS=OFF -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_COMPILER=nvcc -DCMAKE_CUDA_HOST_COMPILER=g++-13 -DUSE_SYSTEM_NCCL=ON -DNCCL_ROOT_DIR=<job>/deps/nccl -DPython3_EXECUTABLE=<wrapper> -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`, then `make -k -j64` (the dispatcher's `MAX_JOBS`; the earlier check used -j128), each wrapped in `utils/timeit.sh oneflow compile`. Release and `TREAT_WARNINGS_AS_ERRORS=OFF` for the reasons recorded at 8d470dcc14 (the gcc 13.3 `-Werror=maybe-uninitialized` at `oneflow/core/auto_parallel/sbp_graph.cpp:696` is upstream rot; both builds show its two warnings). WITH_CUTLASS at upstream's default (ON), so CUTLASS and flash-attention are compiled.
- Scratch-only, in both trees, never committed: `cmake/third_party/flash_attention.cmake` pointed at a local zip of oneflow-inc/flash-attention-v2 eed2e82 with its `csrc/cutlass` submodule (bbe579a9, v3.4.1) and `#include <cstdio>`/`<cstdlib>` added to `csrc/flash_attn/src/cuda_utils.cu`; the pinned aliyuncs mirror URL still returns HTTP 403.
- The CUDA build takes `LLVM_PROVIDER=install` (WITH_MLIR is OFF by default, CMakeLists.txt:50), so `oneflow/ir/install-llvm.cmake` runs in it; the bundled LLVM configure is an `execute_process` that logs nothing, so its expanded argv was captured afterwards with a traced reconfigure of each tree: `cmake --trace-expand --trace-source=<src>/oneflow/ir/install-llvm.cmake .` (rc 0 on both).

### Results
| | 8d470dcc14 | c741328b7 |
|---|---|---|
| configure / build rc | 0 / 0 | 0 / 0 |
| configure / total wall | 108 s / 688 s | 114 s / 678 s |
| liboneflow.so | 911,801,520 B, 222 sm_80 ELF | 911,801,520 B, 222 sm_80 ELF |
| `_oneflow_internal` | 14,099,776 B | 14,099,776 B |
| oneflow `.cu.o` (excluding CUTLASS, flash-attention) | 222 | 222 |
| compile_commands entries | 1822 | 1822 |
| error / warning diagnostic lines | 0 / 1352 | 0 / 1352 |

All comparisons normalize the job's `src-<tag>` and `build-<tag>` paths.
- Bundled LLVM configure argv (`install-llvm.cmake` `execute_process`, from the trace): identical token for token. The only textual difference is an extra space before `-G` where `${LLVM_HIP_ARGS}` expands to an empty list, which adds no argument. No `--version-script` and no `-DCMAKE_SHARED_LINKER_FLAGS=` in the c741328b7 argv; the trace shows `set(LLVM_HIP_ARGS )` and `if(BUILD_CUDA AND USE_HIP )` skipped. The rest of the file's trace (the `cmake --build . -j256 --target install-oneflow-tblgen install-mlir-headers` step, LLVM_INCLUDE_DIRS) is the same.
- Bundled LLVM: `llvm_monorepo-build/CMakeCache.txt` identical (`CMAKE_SHARED_LINKER_FLAGS:STRING=` empty in both); all 448 `link.txt` identical (the 3 that carry a `--version-script` are LLVM's own export maps and are present in both); no `oneflow_llvm.map` written; libLLVMSupport.so.17git has no VERDEF and no `ONEFLOW_LLVM` node, and liboneflow.so has no `ONEFLOW_LLVM` reference, in either build.
- Top-level: `CMakeCache.txt` identical. Configure output differs only in line numbers shifted by 19444bcdf's comment edits (`cmake/cuda.cmake:102` -> `:104` in one CMake warning, `cmake/third_party.cmake:152` -> `:153` in four include backtraces).
- Compile commands: 1822 of 1822 identical as argument sets, no TU added or removed. Main-build link lines: 531 of 533 identical; the other two (`cutlass_lib`, `cutlass_library_static`) differ only in CUTLASS's unity-batch object names, which are a SHA256 of the batch's absolute source paths (cutlass `CUDA.cmake:262`); with paths and those hashes normalized they are byte-identical (40 objects each).
- Diagnostics: ANSI-stripped `warning:`/`error:` lines compared as multisets with line numbers kept: identical, nothing only in either build. (The count differs from the 1364 recorded at 8d470dcc14 because this run's filter is different; the comparison here is between two runs of the same filter.)
- liboneflow.so: NEEDED entries identical; the dynamic symbol tables differ only in 4 nvcc anonymous-namespace names whose hash depends on the source path (`_GLOBAL__N__5276d210_19_random_generator_cu...` vs `..._58b7062f_...`, same symbols).
- Not covered: running anything on an NVIDIA GPU.
- Build trees, source clones and dependency extracts deleted after the run (host disk at 97%). Logs, the traces and the extracted argv/cache files stay in the job's `cw/logs/` (container-local); the comparison script is `cw/tools/compare.py`.

This settles the "CUDA compile check of c741328b7 on the mi355 host (nvcc)" item under "Porter round 2026-11-12 ... Owed".

## Validation 2026-11-11 (linux-gfx90a) at c741328b7 (validator, MI250X GCD, `HIP_VISIBLE_DEVICES=0` pinned by the dispatcher; jeff's request: revalidate at the new head after the 2026-11-11 PR-prep rounds, with NO LD_PRELOAD of libLLVM)

Host: AMD Instinct MI250X / MI250 (`gfx90a:sramecc+:xnack-`, wave64), one GPU visible, ROCm 10.1.0 SDK wheels (`$FLEET_VENV-base-links/_rocm_sdk_devel`), Python 3.11, numpy 1.26.4, protobuf 3.20.3 (`"$FLEET_VENV"/bin/python -m pip install -r dev-requirements.txt fsspec pytest-timeout pytest-repeat opencv-python-headless==4.10.0.84`; the fleet guard only accepts pip when the literal `$FLEET_VENV` appears in the command), torch 2.16.0a0+git6c3af02 symlinked read-only into `$FLEET_VENV-base-links`, `ulimit -c 0` on every run, **`LD_PRELOAD` unset on every run** (`env -u LD_PRELOAD` in each script), `PYTHONDONTWRITEBYTECODE=1`, no JIT extension builds. Fresh fork clone (`--reference` the prep-fixes job's clone, `--dissociate`), `protect-fork` installed. **Head confirmed: `moat-port` = `git ls-remote` = c741328b7b5e70f1b7dbb946d269f1605bf8c0ee; `git status --porcelain` empty before and after.** Scripts in `agent_space/` (gitignored): `env.sh`, `build.sh`, `suite.sh`, `suites.sh`, `one.sh`, `py1.sh`, `cap.sh`, `reg.sh`, `rr.sh`, `cg.sh`, `ctl.sh`, `cmp.py`, `t/{imp,cg_grad,longk,bce_probe,torch_bce,torch_bce_tgt}.py`.

### Verdict: **completed**. Without any preload, `import oneflow` works in both import orders; graph (with and without CUDA graphs) and tensor suites match the baseline id for id; modules differ by one id and one aborted file, both explained below as test-oracle randomness that the base build shows too; every capture, regression, long-k and cold-capture check passes.

### Build (documented procedure, clean, this job's directory)
`utils/timeit.sh oneflow compile -- bash agent_space/build.sh all` = `bash build_rocm.sh all` with `CMAKE_PREFIX_PATH=$R`, `ROCM_CLANG(_C)` from the SDK wheel, `CMAKE_HIP_ARCHITECTURES=gfx90a`, `JOBS=$MAX_JOBS` (128), job-local ccache (cold), `pywrap/python3` first on PATH (the cmake/python.cmake `pip install --user` venv quirk): **rc 0, 674 s**, `OF_GPU_WARP_SIZE: 64`. `readelf --dyn-syms`: libLLVMSupport.so.17git **3002** `@@ONEFLOW_LLVM`, libLLVMDemangle.so.17git **455**, liboneflow.so **4** `@ONEFLOW_LLVM` references (the porter's counts).

### import without preload (`agent_space/t/imp.py`, asserts `LD_PRELOAD` is absent)
| order | result |
|---|---|
| `import oneflow; import torch` | rc 0, `1.0.0+cu00.git.c741328b7`, `flow.cuda.is_available()` True, 64 x 1000 softmax sums to 64.0, 64 x 64 fp32 mm vs numpy 7.6e-6, torch mm 7.6e-6, torch -> oneflow round trip bitwise |
| `import torch; import oneflow` | rc 0, the same (softmax sum 63.999996) |
No `realloc(): invalid pointer` anywhere in this round's logs. The `Runtime version 71626.38 of CUDA Runtime incompatible with compiletime version 11.8` warning at import is pre-existing (in the 3007f7951 baseline logs as well).

### Suites (`agent_space/suite.sh`: one pytest per file, `--timeout=900` plus `timeout -k 30 1800`, junit per file, `ONEFLOW_TEST_DEVICE_NUM=1`, no preload; `utils/timeit.sh oneflow test`, 964 s total)
Baseline: no 8d470dcc1 gfx90a junit survives on this host, so the closest recorded baseline is the prep-fixes job's 3007f7951 build run WITH the preload (`mi200-20261111-153809-oneflow-prep-fixes/moat/agent_space/j/base_*`, read-only; its totals reproduce the 8d470dcc1 gfx90a record, and 3007f7951 differs from 8d470dcc1 by two comments and a test timeout). Also compared with the porter's c741328b7 no-preload run (`j/new_*`).
| suite | wall | pass / fail / error / skip | per id vs 3007f7951 baseline | per-file rc vs baseline |
|---|---|---|---|---|
| graph (92 files, 4 at a time) | 112 s | 152 / 2 / 3 / 102 | **259 of 259 same** | identical |
| graph, `ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1` | 112 s | 152 / 2 / 3 / 102 | **259 of 259 same** | identical; no timeout, no hang |
| tensor (22 files, 4 at a time) | 91 s | 366 / 3 / 0 / 22 | **391 of 391 same** | identical |
| modules (548 files, 8 at a time) | 649 s | 1614 / 49 / 5 / 315 | **1982 of 1983 shared same**; 5 ids only in the baseline | 1 file differs (test_higher_derivative_loss: 1 -> 139) |
Against the porter's c741328b7 run: graph 259/259, graph+CG 259/259, tensor 391/391 same; modules 1981 of 1983 (the two random ids below).

Every modules difference, explained:
- `test_global_higher_derivative_loss::test_smooth_l1_loss_grad_grad` fail -> pass. Random draws: in 3 reruns on this build it failed 2 times and passed once (run 2); the porter saw the same flip on 3007f7951 (its rerun 2). Not port-related.
- `test_special_ops::test_flow_psi_with_random_data` (porter's diff only): pass here, and 3 of 3 pass in reruns; `test_flow_digamma_with_random_data` failed once in 3 reruns. The random-draw digamma/psi item recorded since 021f4ad34.
- `test_higher_derivative_loss` aborted (rc 139), so its 5 ids are missing. **Root cause, found this round:** `generate_necessity_for_bce_loss` draws the BCE target from `[0, num_classes)` = [0, 2) on whichever device `random_device()` picks, and torch (the oracle) rejects targets above 1. On CPU it raises `RuntimeError: all elements of target should be between 0 and 1` (the test fails, `PyTorchDoesNotSupportError`); on the GPU the same check is `CUDA_KERNEL_ASSERT(target_val >= zero && target_val <= one)` (`aten/src/ATen/native/cuda/Loss.cu:90`), a device trap: `HSA_STATUS_ERROR_EXCEPTION ... code: 0x1016` in `binary_cross_entropy_out_cuda`, then a 719 at the next HIP call and SIGSEGV/SIGABRT while the process goes down. Evidence: `t/bce_probe.py` wraps torch's `F.binary_cross_entropy` and logs what it receives; the faulting call had input [0.0706, 0.9685] (valid) and **target [0.4187, 1.348]**. Torch alone, no oneflow imported (`t/torch_bce_tgt.py`, input 0.5): target 0.9 and 1.0 no fault, **target 1.348 -> the same HSA exception**. So abort vs fail depends only on which device the first out-of-range draw lands on. Counts: this build aborted 4 of 4 (suite plus 3 reruns of `rr.sh`); the 3007f7951 build (read-only, with preload, `ctl.sh`) aborted 1 of 3 here, and in the porter's round each build aborted 1 of 3. On every non-aborting run on either build the test reports `test_bce_loss_grad_grad` failed with the torch target error, so the file never passes on either build. Not port-related; an upstream test defect (target should be drawn in [0, 1] for BCELoss).
- Aborted files otherwise the same on both builds: test_dataset (`/dataset` missing), test_loss (the same BCE oracle); graph: test_alexnet_graph, test_graph_ofrecord_reader, test_run_global_graph_by_vm (`/dataset`).

### Capture regression tests (`agent_space/cap.sh`, one pytest process per run, no preload)
| test | `ROCBLAS_USE_HIPBLASLT` unset | `=1` |
|---|---|---|
| graph/test_graph_fused_mlp_cuda_graph.py | **5 of 5 pass** (7.3-7.4 s) | **5 of 5 pass** (7.4 s) |
| graph/test_graph_cuda_graph_stream_creation.py | **5 of 5 pass** (6.6-6.9 s) | **5 of 5 pass** (6.6-6.8 s) |
| graph/test_graph_cuda_graph_fft_plan.py | **5 of 5 pass** (23.8-24.8 s) | **5 of 5 pass** (21.5-22.6 s) |
Plus test_graph_fused_mlp_cuda_graph with outer `ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1`: 3 of 3 pass. No log has a `(906)`, `(901)` or `PARSE_ERROR` line. (A first `cap.sh` attempt clobbered its loop variable inside `run()` and ran nothing after the first test, rc 4 "file not found"; fixed with `local`, all rerun.)

### Regression tests of the recent rounds (`agent_space/reg.sh`, no preload)
test_matmul_long_k 4 p; test_conv_float64 6; test_grouped_matmul_bias 2; test_cross_entropy_float16 3; test_check_numerics_float16 2; test_pool2d_channels_last 1; test_pooling_channels_last 2; test_batchnorm 6 + 1 skip; test_conv_bias_grad 2; test_reduce_sum_like 2; test_fused_matmul_bias 1; test_bmm 2; test_rand 4 + 1 skip; test_randint 6 + 1 skip; test_optim_lbfgs 1; misc/test_import_without_imp 1; graph/test_graph_repeated_small_run 2; graph/test_graph_fused_mlp_cuda_graph 1; graph/test_graph_image_gpu_decoder 1 + 1 skip. **Identical to the 8d470dcc1 record.** test_cublas_fused_mlp fails 1 of 1 (`test_fused_matmul_op`, float32 elementwise 1e-4), the known tolerance item, the same as at 8d470dcc1.

### Long-k and cold-capture checks (no preload)
- `t/longk.py` (reconstructed; the earlier script did not survive): k = 2^20 fp16/bf16 eager, uniform(-1, 1), error = max abs / max |float64 ref|, oneflow vs torch on the same rounded inputs: **RESULT PASS, oneflow equals torch in all 12 cases**. fp16: mm 1/2/16 x 2^20 x 64 2.78e-4 / 2.79e-4 / 4.11e-4, transpose_b 2.01e-4, bmm 4 x (1 x 2^18 x 64) 2.54e-4, 1 x 2^16 x 64 2.61e-4 (the same numbers as the 2026-11-01 table); bf16 1.81e-3 to 2.79e-3 (different draws from the old table, same level).
- `t/cg_grad.py <k>` with `ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1 ONEFLOW_ONE_EMBEDDING_FUSED_MLP_ASYNC_GRAD=1` (two-layer fused_mlp fp16, fp32 master weights, SGD lr 1, weights reset before each step, graph called first; 4 steps x 4 parameters, graph update vs eager `before - grad`, bitwise): **k = 64, 1024 and 2^20, `ROCBLAS_USE_HIPBLASLT` unset and `=1`: rc 0, 16 of 16 bitwise equal, 0 mismatches, no 906, in all 6**.
  - A first reconstruction reset the graph's parameters with `param.data = new_tensor`; that swaps in a tensor the compiled graph never sees, so the graph kept training its own buffers while `before` read the reset values: step 0 was bitwise equal, steps 1-3 differed by up to 390. Positive check that this is the script and not c741328b7: the same v1 script on the 3007f7951 build (read-only, with preload) gives **the identical 4 of 16 / 12 mismatches**. Resetting in place (`param.copy_(...)` under `no_grad`) gives 16 of 16 on this build. Kept as `t/cg_grad_v1_dataassign.py`.

### Not run
- Two-GPU one_embedding / P2P / fused-harness tests: one GPU pinned. The delta 8d470dcc1..c741328b7 touches no kernel (two CMake comments, one test keyword, the LLVM sub-build's link flag).
- CUDA compile gate: no nvcc on this host (`/opt/conda/envs/cuda-12.8` absent). Recorded by another host concurrently with this round: "CUDA compile check at c741328b7" (nvcc 12.8, PASS).

### Other checks
- `python3 utils/jargon.py --port oneflow`: only the three known hits in 68718f038's message (for the PR-prep squash); nothing new.
- Documentation: no tracked file carries an LD_PRELOAD/libLLVM instruction; README's ROCm block and `build_rocm.sh` reproduced this build. From c741328b7 on, the `LD_PRELOAD=$R/lib/llvm/lib/libLLVM.so.24.0git` in this file's earlier run recipes is not needed on this platform: this whole round ran without it.
- `git -C projects/oneflow/src status --porcelain`: empty (no modified tracked file).
- Core dumps: `ulimit -c 0` on every run; this host's `kernel.core_pattern` pipes to systemd-coredump, which receives dumps regardless (bash printed "dumped core" for the test_higher_derivative_loss aborts and the deliberate torch-only BCE trap; no core file landed in the job directory).

## Validation 2026-11-11 (linux-gfx1100) at c741328b7 (validator, Radeon Pro W7800 48GB, `HIP_VISIBLE_DEVICES=0` pinned by the dispatcher; jeff's request: revalidate at the new head after the 2026-11-11 PR-prep rounds, with NO LD_PRELOAD of libLLVM)

Host: Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), one GPU visible, ROCm 10.1.0 SDK wheels (`$FLEET_VENV-base-links/_rocm_sdk_devel`), Python 3.11.17, numpy 1.26.4, protobuf 3.20.3 (`"$FLEET_VENV"/bin/python -m pip install -r dev-requirements.txt fsspec pytest-timeout pytest-repeat opencv-python-headless==4.10.0.84`), torch 2.16.0a0+git6c3af02 symlinked read-only into `$FLEET_VENV-base-links` (torch, its dist-info, torchgen, functorch), `ulimit -c 0` and **`LD_PRELOAD` unset on every head run** (`unset LD_PRELOAD` in `agent_space/env.sh`, and `t/imp.py` prints it: `None`), `OMP_NUM_THREADS=4`, `ONEFLOW_TEST_DEVICE_NUM=1`, `PYTHONDONTWRITEBYTECODE=1`, no JIT extension builds. Fresh fork clone into `projects/oneflow/src`, `protect-fork` installed. **Head confirmed: `moat-port` = `git ls-remote origin moat-port` = c741328b7b5e70f1b7dbb946d269f1605bf8c0ee; `git status --porcelain` empty before the build and after every run.** Scripts in `agent_space/` (gitignored): `env.sh`, `build.sh`, `run1.sh`, `suite.sh`, `cap.sh`, `reg.sh`, `cmp.py`, `t/{imp,cg_grad,longk}.py`.

### Verdict: **completed**. Without any preload, `import oneflow` works in both import orders. The graph suite (with and without CUDA graphs) and the tensor suite match a same-host 8d470dcc1 baseline id for id. Modules differs from that baseline in one id, the known random CPU float16 digamma draw. Every capture, regression, long-k and cold-capture check passes.

### Host load (recorded as asked)
While this job built, another job's pytorch build ran about 60 gcc processes on the same 64-thread host. Load average reached **156** at 04:02 and stayed above 80 until about 04:20. Heavy oneDNN TUs (`cpu_rnn_list.cpp`, `cpu_reorder_regular_f32_f32.cpp`, `ref_rnn.cpp`) each spent more than 25 minutes in cc1plus at about 60 % of a core. After 04:25 load fell to 15-28. Test timings were only mildly affected: graph 140 s here vs 133 s at 8d470dcc1. modules took 1377 s at 4 files at a time, vs 841 s for the same runner at fd94e199a; the same-host baseline below took 1307 s under the same conditions, so this is the host and the test count, not the head.

### Build (documented procedure, clean, this job's directory)
`utils/timeit.sh oneflow compile -- bash agent_space/build.sh <src>` = `bash build_rocm.sh all` with `CMAKE_PREFIX_PATH=$R`, `ROCM_CLANG(_C)` from the SDK wheel, `CMAKE_HIP_ARCHITECTURES=gfx1100`, `JOBS=$MAX_JOBS` (10, the dispatcher's cap; earlier rounds used 32), job-local ccache (cold, `CCACHE_BASEDIR` set), `pywrap/python3` first on PATH (drops cmake/python.cmake's `pip install --user`): **rc 0, 3587 s** (1133 s at JOBS=32 at 8d470dcc1), `CMAKE_HIP_ARCHITECTURES: gfx1100`, `OF_GPU_WARP_SIZE: 32`. `readelf --dyn-syms`: libLLVMSupport.so.17git **3002** `@@ONEFLOW_LLVM`, libLLVMDemangle.so.17git **455**, liboneflow.so **4** `@ONEFLOW_LLVM` references. These are the gfx90a porter's and validator's counts.

Baseline: a second clone at 8d470dcc14 (`agent_space/base/src`) built the same way after the head build (ccache warm, 44 % hits; rc 0, about 29 min). Its libLLVMSupport.so.17git and liboneflow.so have **0** `ONEFLOW_LLVM` symbols and no version node. Control: **without** the preload, `import oneflow` on that baseline aborts with `realloc(): invalid pointer`, rc 134. All baseline suite runs below used `LD_PRELOAD=$R/lib/llvm/lib/libLLVM.so.24.0git`, as every round up to 8d470dcc1 ran. The baseline was needed because the 8d470dcc1 gfx1100 junit lived in a job directory that no longer exists, and that round ran only the graph suite and the 133-file list, not tensor or modules.

### import without preload (`agent_space/t/imp.py`)
| order | result |
|---|---|
| `import oneflow; import torch` | rc 0, `1.0.0+cu00.git.c741328b7`, `flow.cuda.is_available()` True, 64 x 1000 softmax sums to 64.0, 64 x 64 fp32 mm vs float64 numpy 9.8e-6 (torch 9.8e-6), torch -> oneflow GPU round trip bitwise |
| `import torch; import oneflow` | rc 0, the same (mm 7.4e-6 on that draw) |
No `realloc(): invalid pointer` in any head log of this round (`grep -r` over `agent_space/j`). The `Runtime version 71626.38 ... incompatible with compiletime version 11.8` and `capability 110 is not compatible` warnings at import are pre-existing.

### Suites (`agent_space/suite.sh`: one pytest per file, `--timeout=900` plus `timeout -k 30 1800`, junit per file, 4 files at a time; `utils/timeit.sh oneflow test`; per-id comparison `agent_space/cmp.py`)
| suite | c741328b7, no preload: wall, pass / fail / error / skip (ids) | 8d470dcc1 + preload, same host | per id | per-file rc |
|---|---|---|---|---|
| graph (92 files) | 140 s, 152 / 2 / 3 / 102 (259) | 130 s, 152 / 2 / 3 / 102 | **259 of 259 same** | identical |
| graph, `ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1` | 136 s, 152 / 2 / 3 / 102 (259) | 131 s, the same | **259 of 259 same** | identical; slowest file 25 s, no timeout, no hang |
| tensor (22 files) | 73 s, 368 / 1 / 0 / 22 (391) | 68 s, the same | **391 of 391 same** | identical |
| modules (548 files) | 1377 s, 1614 / 53 / 5 / 316 (1988) | 1307 s, 1615 / 52 / 5 / 316 (1988) | **1987 of 1988 same** | 1 file differs (test_special_ops 0 -> 1) |
- Graph: the same totals and the same 10 nonzero files as "Validation 2026-11-09 (validator, linux-gfx1100 ...)" at 8d470dcc1. test_alexnet_graph, test_graph_ofrecord_reader and test_run_global_graph_by_vm abort on `/dataset/imagenette/ofrecord/part-00000` (missing). test_alexnet_auto_parallel, test_fx_fuse and test_fx_replace_ops error at collection. test_fx_symbolic_trace_module and test_run_graph_by_vm have 1 failure each. test_graph_session_env_destruct1 and test_util collect nothing.
- Tensor: only test_safetensors (`ModuleNotFoundError: safetensors`, not installed), as in every earlier gfx1100 round. The 8d470dcc1-era tensor total of 368 / 1 / 0 / 22 is reproduced exactly.
- Modules, the one difference: `test_special_ops::test_flow_psi_with_random_data` passed on the baseline and failed on the head. The failure is `op_kernel_not_found_error` for `digamma` with `DeviceType_Name: kCPU`, `kFloat16`: the autotest drew CPU float16, and no build has a CPU float16 digamma kernel (the class recorded since the f52f1b78e rounds). Three reruns of the file on the head build: run 1 failed the same way (CPU float16 again), runs 2 and 3 passed 16 of 16. Not port-related.
- Modules failures otherwise have the same ids on both builds, in the classes already explained in this file: the autotest "endless loop" files (avgpool 4, global_avgpool 1, conv1d 2, conv2d 4, deconv2d 3), the numpy `__bool__` autotest proxy (concat, copy 2, div, logical_and, matmul 2, replication_pad, slice 2, tensordot), `KeyError: 'ONEFLOW_CI'` (test_fft 5), `oneflow.char != int8` (tensor_ops 4, type_tensor 2), the higher-derivative torch oracles (bce target outside [0, 1], kl_div, smooth_l1, and the ZeroTensor abs grad_grad pair), the image ops on `/dataset`, 5 collection errors, functional docstr, codegeex, and the float32 tolerance pair test_cublas_fused_mlp / test_fused_matmul_bias_add_relu_dropout. Aborts on both builds: test_dataset (`/dataset`) and test_loss (the BCE oracle device trap), rc 134. test_higher_derivative_loss completed on both builds this time.

### Capture tests (`agent_space/cap.sh`, one pytest process per run, sequential, no preload)
| test | `ROCBLAS_USE_HIPBLASLT` unset | `=1` |
|---|---|---|
| graph/test_graph_fused_mlp_cuda_graph.py | **5 of 5 pass** (7-8 s) | **5 of 5 pass** (7-8 s) |
| graph/test_graph_cuda_graph_stream_creation.py | **5 of 5 pass** (6-7 s) | **5 of 5 pass** (6-7 s) |
| graph/test_graph_cuda_graph_fft_plan.py | **5 of 5 pass** (17 s) | **5 of 5 pass** (16-17 s) |
No log has a `(906)`, `(901)` or `PARSE_ERROR` line.

### Regression tests of the recent rounds (`agent_space/reg.sh`, no preload)
test_matmul_long_k 4 p; test_conv_float64 6; test_grouped_matmul_bias 2; test_cross_entropy_float16 3; test_check_numerics_float16 2; test_pool2d_channels_last 1; test_pooling_channels_last 2; test_batchnorm 6 + 1 skip; test_conv_bias_grad 2; test_reduce_sum_like 2; test_fused_matmul_bias 1; test_bmm 2; test_rand 4 + 1 skip; test_randint 6 + 1 skip; test_optim_lbfgs 1; misc/test_import_without_imp 1; graph/test_graph_repeated_small_run 2; graph/test_graph_fused_mlp_cuda_graph 1; graph/test_graph_image_gpu_decoder 1 + 1 skip. All pass. test_cublas_fused_mlp fails 1 of 1 (`test_fused_matmul_op`, the known float32 tolerance item), as at 8d470dcc1 and on gfx90a at c741328b7.

### Long-k and cold-capture checks (no preload)
- `t/longk.py` (written for this round; the earlier script did not survive): fp16 and bf16, uniform(-1, 1), error = max abs / max |float64 ref|, oneflow and torch on the same rounded inputs: **RESULT PASS**. oneflow fp16: mm 1/2/16 x 2^20 x 64 3.75e-4 / 3.15e-4 / 3.41e-4, transpose_b 3.75e-4, bmm 4 x (1 x 2^18 x 64) 3.37e-4, 1 x 2^16 x 64 2.88e-4. On the same inputs torch's fp16 results are 3.9e-3 / 1.1e-3 / 8.7e-4 / 3.3e-3 for the four 2^20 cases: torch on this ROCm still shows the long-k fp16 accuracy loss on gfx1100, and oneflow's long-k path does not. bf16: oneflow 2.5e-3 to 4.0e-3, torch 2.0e-3 to 5.0e-3.
- `t/cg_grad.py <k>` (rebuilt from the gfx90a description: two-layer fused_mlp fp16, fp32 master weights, SGD lr 1, weights reset in place before each step, graph called first; 4 steps x 4 parameters, graph update vs eager `before - grad`, bitwise) with `ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1 ONEFLOW_ONE_EMBEDDING_FUSED_MLP_ASYNC_GRAD=1`: **k = 64, 1024 and 2^20, `ROCBLAS_USE_HIPBLASLT` unset and `=1`: rc 0, 16 of 16 bitwise equal, 0 mismatches, no 906, in all 6**.

### Not run
- Two-GPU one_embedding / P2P tests: one GPU pinned. The delta 8d470dcc1..c741328b7 touches no kernel (two CMake comments, one test timeout, the bundled LLVM sub-build's link flag).
- CUDA compile gate: already recorded at this head ("CUDA compile check at c741328b7", mi355, nvcc 12.8.93, PASS). Not repeated.

### Other checks
- `python3 utils/jargon.py --port oneflow` (after `git branch -f master origin/master` in the clone): only the three known hits in 68718f038's message, left for the PR-prep squash; added lines are clean.
- Documentation: no tracked file carries an LD_PRELOAD or libLLVM instruction. The README ROCm block and `build_rocm.sh` reproduced this build unchanged. From c741328b7 on, the `LD_PRELOAD=$R/lib/llvm/lib/libLLVM.so.24.0git` in the earlier gfx1100 recipes of this file is not needed on this platform either.
- `git -C projects/oneflow/src status --porcelain`: empty.
- Core dumps: `ulimit -c 0` on every run. This host's `kernel.core_pattern` pipes to apport (`|/usr/share/apport/apport ...`), and a pipe handler receives the dump whatever the ulimit says. bash printed "core dumped" for the deliberate no-preload baseline import and for the known aborting files (test_dataset, test_loss, the three `/dataset` graph files). No core file landed in the job directory.

## PR-prep squash 2026-11-12

jeff approved the squash shape (gcc-13 commit, RMSNorm fix with its test, one HIP backend commit) and authorized the force-push. moat-port was force-pushed c741328b7 -> 14801d8a83043e852b72a3671bc88e5896aec904 with `--force-with-lease=moat-port:c741328b7...`; both trees are 0dac92194ed9bfc8acd94ba591866f8c9644ac82. `squash-carry-forward` carried linux-gfx90a and linux-gfx1100. linux-gfx942 stays validated at d9e41753e (jeff ruled not to revalidate it before the squash); wave64 is satisfied by gfx90a. The stale fork branch `moat-port-squash` (be1ef73f1) was deleted with jeff's authorization.

## Upstream 2026-11-12

- jeff approved the fork review PR (https://github.com/AMD-Ecosystem/oneflow/pull/1, `/moat approve`); `upstream.py --publish --apply --name oneflow` opened https://github.com/Oneflow-Inc/oneflow/pull/10697 at 14801d8a8. moat-port is now frozen.
- jeff filed the libLLVM interposition report as https://github.com/ROCm/TheRock/issues/9939 (deferral `therock-libllvm-symbol-interposition`; its stored summary is out of date, see "Porter round 2026-11-12" for the corrected mechanism).

## Validation 2026-11-13 (linux-gfx942) at 14801d8a8 (validator, MI300X, `HIP_VISIBLE_DEVICES=4,5` pinned by the dispatcher; jeff's request: revalidate gfx942 at the submitted fork head so the open upstream PR's gfx942 claim covers the code submitted, with NO LD_PRELOAD of libLLVM)

Host: AMD Instinct MI300X (gfx942, wave64), two GPUs visible, Intel Xeon Platinum 8480C, ROCm 10.1.0 SDK wheels (`$FLEET_VENV-base-links/_rocm_sdk_devel`), Python 3.11.17, numpy 1.26.4, protobuf 3.20.3 (`"$FLEET_VENV"/bin/python -m pip install -r dev-requirements.txt fsspec pytest-timeout pytest-repeat opencv-python-headless==4.10.0.84 portalocker protobuf==3.20.3 numpy==1.26.4`), torch 2.16.0a0+git6c3af02 symlinked read-only into `$FLEET_VENV-base-links` (torch, its dist-info, torchgen, functorch). `ulimit -c 0` and **`LD_PRELOAD` unset on every run** (`unset LD_PRELOAD` in `agent_space/env.sh`; `t/imp.py` prints `None`), `OMP_NUM_THREADS=4`, `PYTHONDONTWRITEBYTECODE=1`, `ONEFLOW_TEST_DEVICE_NUM=1` for one-GPU runs and `=2` for two-rank runs, no JIT extension builds. Fresh fork clone, `protect-fork` installed. **Head confirmed: `moat-port` = `git ls-remote origin moat-port` = 14801d8a83043e852b72a3671bc88e5896aec904 (tree 0dac92194e); `git status --porcelain` shows no modified tracked file before the build or after the runs.** d9e41753e is no longer reachable after the squash (`classify` fails with "bad object"), so this was a full run, not a carry-forward check. Scripts in `agent_space/` (gitignored): `env.sh`, `build.sh`, `suite.sh`, `one.sh`, `cap.sh`, `reg.sh`, `emb.sh`, `emb2.sh`, `fh10.sh`, `idsg.sh`, `seq.sh`, `probe.sh`, `cmp.py`, `t/{imp,longk,cg_grad,fused_harness,fhcmp,ids_probe,two_graphs,exit_raise,exit_raise_normal}.py`.

### Verdict: **validation-failed**, for one unexplained GPU fault. `expensive/test_id_shuffle.py` (one GPU) aborted **once in 90 runs**: `HSA_STATUS_ERROR_MEMORY_FAULT` in `oneflow::(anonymous namespace)::CudaGatherNd<int, int>`, then `cuda_device.cpp:122 Check failed: cudaSetDevice(device_index_) : an illegal memory access was encountered (700)`, rc 134. It did not reproduce in the other 89 runs, nor in 6000 in-process iterations of the same graph that check every index on the host. Every other check passes, or differs from the records only in ways explained below as not caused by the port. jeff asked for `completed` only if every difference is explained, and this one is not, so this platform records `validation-failed` at 14801d8a8. wave64 stays satisfied by linux-gfx90a, which `squash-carry-forward` carried to this head.

### The unexplained fault (for the porter, or for jeff's ruling)
- Where it happened: round 2 of `emb.sh`, at 07:42:54. It was the first one-GPU file on GPU 4, right after the two-rank fused-harness run with every P2P switch on had exited (`setsid -w`; the launcher had reaped both ranks). Nothing else of this job was on GPU 4 at the time (my last other GPU process ended at 07:30). Log: `agent_space/j/emb/r2/1gpu_test_id_shuffle.log`. Hang dump: `grid=[13312, 1, 1], workgroup=[256, 1, 1]` (13312 = 512 x 26, the `num_tables=26` cases), faulting address 0x763c6b57d000.
- Which code: the test's reference check `cur_rank_unique_ids[cur_rank_inverse_indices][inverse_unique_partition_indices]` is the gather_nd. A fault there means either id_shuffle produced an out-of-range index or gather_nd read out of bounds. The port does not modify `nd_index_slice_kernels.cu`, `nd_index_slice_util.h` or the one-GPU id_shuffle path. Its only change in `data_shuffle_kernel.cu` is `kWarpSize` from `OF_GPU_WARP_SIZE` and `OF_SHFL_FULL_MASK` in a warp max reduction (+8 -1). It is still a HIP build of those kernels, so I cannot call it non-port.
- Reruns, all pass: 20 + 20 runs of the file (GPU 4 and GPU 5 at the same time); 40 more between two-rank runs (`idsg.sh`); 8 replays of the exact preceding sequence (fused all-P2P two-rank run, then the file on GPU 4; `seq.sh`); round 1 and the d9e41753e record (5 rounds) never saw it either.
- `t/ids_probe.py`: the test's graph built once per configuration (table ids on/off x 1/26 tables), 3000 iterations per process, two processes at once. Before any gather it checks that every `cur_rank_inverse_indices` < len(`cur_rank_unique_ids`) and every `inverse_unique_partition_indices` < len(`cur_rank_inverse_indices`), then reconstructs ids on the host (GPU 4) or with the GPU gather_nd (GPU 5): **bad_index 0, wrong_values 0** on both.
- Suggested next step: run the file in a loop of a few hundred with `AMD_LOG_LEVEL` off and `HSA_ENABLE_SDMA=0` variations, or `AMD_SERIALIZE_KERNEL=3`, to get a second occurrence. If id_shuffle's outputs can be dumped before the gather when it fires, that decides between the two candidate kernels.

### Build (documented procedure, clean, this job's directory)
`utils/timeit.sh oneflow compile -- bash agent_space/build.sh all` = `bash build_rocm.sh all` with `CMAKE_PREFIX_PATH=$R`, `ROCM_CLANG(_C)` from the SDK wheel, `CMAKE_HIP_ARCHITECTURES=gfx942`, `JOBS=$MAX_JOBS` (37, the dispatcher's cap), job-local ccache (cold), `pywrap/python3` first on PATH: **rc 0, 790 s**, `CMAKE_HIP_ARCHITECTURES: gfx942`, `OF_GPU_WARP_SIZE: 64`, liboneflow.so 982 MB. `readelf --dyn-syms`: libLLVMSupport.so.17git **3002** `@@ONEFLOW_LLVM`, libLLVMDemangle.so.17git **455**, liboneflow.so **4** `@ONEFLOW_LLVM` references, the gfx90a and gfx1100 counts. (A first configure stopped after 25 s at cmake/python.cmake's `pip install --user`, because my wrapper did not yet drop `--user`; after fixing the wrapper the build was clean, from `rm -rf build`.)

### import without preload (`agent_space/t/imp.py`)
| order | result |
|---|---|
| `import oneflow; import torch` | rc 0, `1.0.0+cu00.git.14801d8a8`, `flow.cuda.is_available()` True, 64 x 1000 softmax sums to 64.0, 64 x 64 fp32 mm vs float64 numpy 7.9e-6 (torch 4.6e-6), torch -> oneflow round trip bitwise |
| `import torch; import oneflow` | rc 0, the same (mm 7.3e-6, torch 7.3e-6) |
No `realloc(): invalid pointer` in any log of this round. The `capability 94 is not compatible` and `Runtime version 71626.38 ... compiletime version 11.8` warnings at import are pre-existing.

### Suites (`agent_space/suite.sh`: one pytest per file, `--timeout=900` plus `timeout -k 30 1800`, junit per file; `utils/timeit.sh oneflow test`; `agent_space/cmp.py`)
The d9e41753e gfx942 junit and the c741328b7 gfx90a junit no longer exist on any host I can reach (both lived in finished job directories). So the comparison is against the per-id failure lists and totals recorded in this file.
| suite | wall | pass / fail / error / skip (ids) | gfx942 records | gfx90a c741328b7 |
|---|---|---|---|---|
| graph (92 files, 4 at a time, GPU 4) | 124 s | 152 / 2 / 3 / 102 (259) | 8d470dcc1 gfx942 porter round: 152 pass, 102 skip, 5 fail+error, the same 10 nonzero files | 152 / 2 / 3 / 102 |
| graph, `ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1` | 118 s | 152 / 2 / 3 / 102 (259); **259 of 259 ids the same as without** | | the same |
| tensor (22 files, 4 at a time, GPU 4) | 102 s | 368 / 1 / 0 / 22 (391) | d9e41753e: 368 / 1 / 0 / 22, test_safetensors only | 366 / 3 / 0 / 22 |
| modules (548 files, 8 at a time, GPU 5) | 699 s | 1610 / 53 / 5 / 315 (1983) | d9e41753e: 1603 / 54 / 5 / 315 (1977) | 1614 / 49 / 5 / 315 (1983) |
- Graph: the 10 nonzero files are the recorded ones. test_alexnet_graph, test_graph_ofrecord_reader and test_run_global_graph_by_vm abort on `/dataset/imagenette/ofrecord/part-00000` (checked in both modes). test_alexnet_auto_parallel, test_fx_fuse and test_fx_replace_ops error at collection. test_fx_symbolic_trace_module and test_run_graph_by_vm fail 1 each. test_graph_session_env_destruct1 and test_util collect nothing. No timeout or hang with CUDA graphs on.
- Tensor: only test_safetensors (module not installed), as at d9e41753e. gfx90a's extra two are its test_autocast fp16 ulp pair, which never failed on gfx942.
- Modules against d9e41753e gfx942, per id from the recorded lists. Failures went from 54 to 53. Gone: test_optim_lbfgs (`collections.Iterable`, fixed by aeaeac14b `Use collections.abc.Iterable`) and test_global_tril_with_diag (the CPU-reproducible flaky autotest). New: `test_matmul::test_flow_tensor_matmul_with_random_data_allow_tf32`, the gfx942 flake recorded at a12ca99e7 (4-5 of 30 standalone). **Here it failed 2 of 15 standalone** (`agent_space/j/rr/tf32_*`), the same rate, with the same float32 tolerance assert. The ids went from 1977 to 1983; the 6 new ids are tests added since d9e41753e (test_matmul_long_k 3 -> 4, plus additions in test_rand, test_randint, test_rmsnorm, test_layer_norm, test_softmax, test_sum, test_argmax, test_logical_reduce), and every one passes. The other 52 failures carry the d9e41753e ids and classes. They are 14 autotest "endless loop" (avgpool 4, global_avgpool 1, conv1d 2, conv2d 4, deconv2d 3), 10 numpy `__bool__` (concat, copy x2, div, logical_and, matmul fp16/int, slice x2, tensordot), fft `KeyError: 'ONEFLOW_CI'` 5, `oneflow.char != int8` 6, higher-derivative torch oracles 5 (global_higher_derivative_loss bce/kl_div/smooth_l1, math_op and global_math_op abs), `/dataset` 4, functional docstr, codegeex, the float32 tolerance pair cublas_fused_mlp / fused_matmul_bias_add_relu_dropout, argsort and sort (numpy's AVX-512 int32 argsort as the oracle; this host is the same 8480C), special_ops psi, and 5 collection errors. psi is the `op_kernel_not_found_error` for `digamma` on `kCPU` / `kFloat16`: in 3 reruns it failed 2 times the same way and passed 16/16 once.
- Aborted files, the same as at d9e41753e: test_dataset (`/dataset`), test_loss and test_higher_derivative_loss. Both name torch's own `binary_cross_entropy_out_cuda` in the `HSA_STATUS_ERROR_EXCEPTION ... code: 0x1016`: the BCE oracle device trap root-caused in "Validation 2026-11-11 (linux-gfx90a) at c741328b7". test_grid_sample passes (4), and ebd0b3938 fixed its race. test_fused_self_attention and test_shutting_down collect nothing (rc 5), as before.
- Against gfx90a c741328b7: 1610 vs 1614 passes. The only extra failures here are argsort and sort (host-CPU oracle), allow_tf32 (gfx942 flake), and psi and smooth_l1 grad_grad (random draws; gfx90a recorded both flipping between runs).

### Capture tests (`agent_space/cap.sh`, GPU 4, one pytest process per run, no preload)
| test | `ROCBLAS_USE_HIPBLASLT` unset | `=1` |
|---|---|---|
| graph/test_graph_fused_mlp_cuda_graph.py | **5 of 5 pass** (11-18 s) | **5 of 5 pass** (10-13 s) |
| graph/test_graph_cuda_graph_stream_creation.py | **5 of 5 pass** (14-16 s) | **5 of 5 pass** (12-14 s) |
| graph/test_graph_cuda_graph_fft_plan.py | **5 of 5 pass** (15-23 s) | **5 of 5 pass** (22-27 s) |
Plus test_graph_fused_mlp_cuda_graph with an outer `ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1`: 3 of 3. No log has `(906)`, `(901)` or `PARSE_ERROR`.

### Long-k and cold-capture checks (GPU 4, no preload)
- `t/longk.py` (k = 2^20 fp16/bf16 eager, uniform(-1, 1), error = max abs / max |float64 ref|, oneflow and torch on the same rounded inputs): **RESULT PASS; oneflow equals torch in all 12 cases.** fp16: mm 1/2/16 x 2^20 x 64 2.78e-4 / 3.08e-4 / 4.08e-4, transpose_b 2.87e-4, bmm 4 x (1 x 2^18 x 64) 2.39e-4, 1 x 2^16 x 64 3.46e-4. bf16: 1.97e-3 to 3.29e-3. 1 x 2^20 x 64 matches d9e41753e exactly (2.78e-4 fp16; bf16 2.53e-3 vs 2.48e-3 there, a different draw). modules/test_matmul_long_k: 4 passed.
- `t/cg_grad.py <k>` with `ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1 ONEFLOW_ONE_EMBEDDING_FUSED_MLP_ASYNC_GRAD=1`, rebuilt on the model of graph/test_graph_fused_mlp_cuda_graph.py: two-layer fused_mlp fp16 with fp32 master weights, SGD lr 1, graph called first, both models reset in place each step, 4 steps x 4 parameters, graph update vs `before - eager grad`, bitwise. **k = 64, 1024 and 2^20, `ROCBLAS_USE_HIPBLASLT` unset and `=1`: rc 0, 16 of 16 bitwise equal in all 6.** My first version called a nonexistent `config.enable_cuda_graph` and died with a Python AttributeError. That exposed the next item.
- **Teardown segfault after an uncaught Python exception: upstream behaviour, not the port.** When a script raises after queueing any CUDA op, oneflow segfaults during exit (rc 139 instead of 1). `t/exit_raise.py` (a cuda Parameter plus an SGD, then `raise`) did this 8 of 8, with CUDA graphs on or off; a clean exit is rc 0 in 6 of 6. gdb: the main thread is in `exit()` running `__hip_module_dtor` -> `__hipUnregisterFatBinary` from liboneflow.so. Meanwhile a VM worker thread, still running the queued op, lazily loads MIOpen (`libMIOpenCKGroupedConv_gfx942.so` -> `__hipRegisterFatBinary` -> `amd_comgr_do_action`) and faults in libLLVM.so.24 (`OverlayFileSystem::pushOverlay`). Cause: python/oneflow/__init__.py's ExitHook marks the exit abnormal, and `SwitchToShuttingDownPhase` (oneflow/api/python/env/env.cpp:60) runs `vm::ClusterSync()` and `CloseVMThreads()` only `if (is_normal_exit)`, so worker threads keep running into `exit()`. With oneflow's excepthook bypassed (`t/exit_raise_normal.py`, so the normal path runs), the same exception exits cleanly, rc 1, 5 of 5. The port does not change that file or the Python exit hook. It only turns an already failing process's exit code into a crash; no test result depends on it.

### Regression tests of the recent rounds (`agent_space/reg.sh`, GPU 4, one pytest per file, no preload)
test_matmul_long_k 4; test_conv_float64 6; test_grouped_matmul_bias 2; test_cross_entropy_float16 3; test_check_numerics_float16 2; test_pool2d_channels_last 1; test_pooling_channels_last 2; test_batchnorm 6 + 1 skip; test_conv_bias_grad 2; test_reduce_sum_like 2; test_fused_matmul_bias 1; test_bmm 2; test_rand 4 + 1 skip; test_randint 6 + 1 skip; test_optim_lbfgs 1; test_rmsnorm 6 + 1 skip; test_layer_norm 6; test_softmax 2; test_sum 8; test_argmax 3; test_logical_reduce 22; misc/test_import_without_imp 1; graph/test_graph_repeated_small_run 2; graph/test_graph_fused_mlp_cuda_graph 1; graph/test_graph_image_gpu_decoder 1 + 1 skip. All pass. test_cublas_fused_mlp fails 1 of 1 (`test_fused_matmul_op`, the known float32 tolerance item), as on gfx90a and gfx1100 at c741328b7.

### Two GPUs: one_embedding, id_shuffle and P2P (`agent_space/emb.sh` 2 rounds, `emb2.sh` 2 rounds; two-rank runs via `python3 -m oneflow.distributed.launch --nproc_per_node 2 --redirect_stdout_and_stderr`, `setsid -w timeout`)
`emb.sh` round 1 ran the two-rank pytest files with `ONEFLOW_TEST_DEVICE_NUM=1` inherited from env.sh, so every `skip_unless_1n2d` file skipped. Those results are discarded (the fused-harness and one-GPU results of that round are valid). From round 2 on, the two-rank launches set `=2`. I wanted to stop round 1 early with a `pkill -f` pattern, and the harness denied it (it could match other jobs' processes on this shared host). I did not work around it: the round finished on its own and `emb2.sh` re-ran the two-rank files. Valid two-rank rounds: `emb.sh` round 2 and `emb2.sh` rounds 1 and 2, 3 rounds in all.
| test | 3 valid rounds |
|---|---|
| expensive/test_id_shuffle_global.py, all P2P switches 0 | 3 passed on both ranks, 3 of 3 |
| expensive/test_id_shuffle_global.py, embedding, gradient **and id_shuffle** P2P 1 | **aborts 3 of 3**: an upstream defect, outside the recorded recipe (below) |
| modules/test_dynamic_allocation_gradient_shuffle_shuffle_global.py, P2P 0 and 1 | 1 skipped (upstream `@unittest.skip`) |
| expensive/test_oneembedding.py, 2 ranks | 1 passed on both ranks, 3 of 3 |
| expensive/test_id_shuffle_p2p_global.py (sets id_shuffle P2P itself) | 1 passed on both ranks, 3 of 3 |
| expensive/test_one_embedding_p2p_shuffle_global.py (fused interaction + embedding/gradient P2P, exact-value P2P op checks) | 1 passed on both ranks, 3 of 3 |
| modules/test_comm_ops.py -k bfloat16 | 2 passed on both ranks, 3 of 3 |
| graph/test_graph_nccl_logical_bfloat16.py | 3 passed on both ranks, 3 of 3 |
| one GPU (both `emb.sh` rounds): test_oneembedding, padding_idx, dynamic_allocation_gradient_shuffle, one_embedding_adagrad, sgd 1 each; adam, ftrl 1 skipped (upstream); test_fused_dot_feature_interaction 2 | passed in both rounds |
| one GPU: expensive/test_id_shuffle | 4 passed in round 1; **round 2 aborted** (the unexplained fault above); 89 of 90 runs pass overall |
- **The id_shuffle P2P abort is upstream's IPC key naming, the same on CUDA.** I set all three P2P switches for test_id_shuffle_global. The d9e41753e recipe set only the embedding and gradient switches there, and covered id_shuffle P2P with test_id_shuffle_p2p_global.py, which passes here. With pytest's capture off (`-s`), rank 0 dies with `rpc_server.cpp:134 CHECK_OR_THROW(kv_.emplace(k, v).second)`, and rank 1 dies in `IdShuffleP2PKernel` -> `RpcClient::PushKV`. (With capture on, both rank logs are empty, because pytest's fd capture swallows the abort.) `GetPtrs` in one_embedding_id_shuffle_p2p_kernel.cu publishes each rank's IPC handle under `ctx->op_name() + parallel_id`. Every nn.Graph names the op `id_shuffle-0`, so a second graph with this op in the same process pushes an existing key. `t/two_graphs.py` with only `ONEFLOW_ONE_EMBEDDING_ID_SHUFFLE_USE_P2P=1`: one graph rc 0; two graphs, both print `id_shuffle ops ['id_shuffle-0']`, then the same CHECK, rc 1. test_id_shuffle_global builds four graphs in one test. The port's diff to that kernel adds only HIP system-scope atomics and the uncached buffer allocation; `GetPtrs` is upstream's. Not a port fault and not a GPU fault.
- One environment failure: `emb.sh` round 1 `fh_fused_p2p1_idp2p` rc 1 after 2 s. Rank 0's gRPC bootstrap server could not bind its port ("No address added out of total 1 resolved"), so rank 1 hit `rpc_client.cpp:212 CHECK_LT_OR_THROW(retry_idx, rpc_client_max_retry_times())`. That is a port collision on a shared host before any GPU work; the same configuration then passed 19 of 19 (below).

### Fused embedding-interaction harness (`agent_space/t/fused_harness.py`, rebuilt from this file's description because no copy survives: 26 tables x 2000 rows, dim 128, device-mem store, seed 7, AMP, `fused_dot_feature_interaction([bottom(dense).unsqueeze(1), emb], output_concat=bottom(dense), output_padding=1)` -> Linear(480, 1) -> BCEWithLogits, SGD lr 0.5, 512 rows per rank, zipf ids, labels = parity of the first two ids, 50 steps, per-step print; each rank feeds its own slice)
- **`fh10.sh`, 10 runs, FUSE_EMBEDDING_INTERACTION=1 with embedding + gradient + id_shuffle P2P on, a run alive at 150 s counts as hung: 10 of 10 rc 0, 32-36 s, 0 hangs.** Plus `seq.sh`, 8 more runs of the same configuration: 8 of 8 rc 0. Plus `emb.sh` round 2's run: 19 of 19 with P2P on and id_shuffle P2P on.
- The fused pass fires: in every fused run the compiled job has `sparse_feature` and `skip_last_gather` true (false in the unfused runs).
- Against `emb.sh` round 1 unfused (`t/fhcmp.py`, both ranks identical): losses are identical at all 50 steps in all 16 compared runs (unfused rerun, fused P2P off x2, fused P2P on x2, fused all-P2P, and the 10 `fh10` runs; 0.6963 -> 0.6934). Top-weight relL2 is 4.0e-5 to 5.7e-5 and bottom-weight relL2 3.7e-6 to 4.7e-6; the unfused rerun differs from round 1 by the same amount (5.2e-5 / 4.1e-6). This harness learns little (the loss barely moves), so it shows the fused P2P paths agree with the unfused path, not training quality.

### Not run
- `cg_ws2.py` / `cg_ws3.py`: not in the dispatch list, and no copy survives. The capture tests and `cg_grad.py` cover capture.
- CUDA compile gate: already recorded for this tree ("CUDA compile check at c741328b7", nvcc 12.8.93, PASS; 14801d8a8 has the identical tree 0dac92194e). Not repeated.

### Other checks
- `python3 utils/jargon.py --port oneflow`: **clean**. The three 68718f038 hits are gone with the squash.
- Documentation: README's ROCm block and `build_rocm.sh` reproduced this build; no tracked file has an LD_PRELOAD or libLLVM instruction.
- `git -C projects/oneflow/src status --porcelain`: no modified tracked file; HEAD = origin `moat-port` = 14801d8a8. Nothing was pushed to the fork.
- Telemetry: four `test` phases of 1-2 s (exit 123; the suites wrote to a relative path that did not exist after `cd`) were removed from stats.jsonl before committing; the recorded phases are the real runs. The 25 s compile phase with exit 1 (the `--user` wrapper) is real and stays.
- Core dumps: `ulimit -c 0` on every run. bash printed "core dumped" for the deliberate exit-crash probes and the known aborting files, and no core file landed in the job directory (the host's `core_pattern` pipe handler takes them regardless).

Records not pushed from this job at first: commit 008b1462 stayed local because MOAT's pre-push checks failed. The host's installed command wrapper came from a newer trunk copy of `utils/gh_guard.py` (73d6ed50) than this branch had, and the orient merge of trunk stopped on a skill-file conflict. I did not reinstall the host's hooks or skip the checks. jeff resolved the conflict on origin (253f5f25); this job merged that and pushed.

## Investigation 2026-11-13 (linux-gfx942): test_id_shuffle GatherNd fault rate

jeff's request: measure the rate of the `CudaGatherNd<int, int>` memory fault that `expensive/test_id_shuffle.py` hit once in 90 runs in "Validation 2026-11-13 (linux-gfx942) at 14801d8a8", and find out whether it predates the submitted head. Investigation only: no fork edit or push, no moatlib state change (this is evidence, not a revalidation), no gh write. Host MI300X, `HIP_VISIBLE_DEVICES=4,5` pinned by the dispatcher, `ulimit -c 0` on every GPU run, same venv recipe as the validation above (`"$FLEET_VENV"/bin/python -m pip install -r dev-requirements.txt fsspec pytest-timeout pytest-repeat opencv-python-headless==4.10.0.84 portalocker protobuf==3.20.3 numpy==1.26.4`). Scripts in this job's `agent_space/` (gitignored): `env.sh`, `build.sh`, `build_old.sh`, `run1.sh`, `rate.sh`, `chk.sh`, `lvl3.sh`, `run1p.sh`, `ratep.sh`, `t/{ids_check,grad_check}.py`, `t/prefix/test_id_shuffle_prefix.py`.

### Verdict: **a defect in upstream's test, not in the port, and it predates the submitted head.**

The test indexes the whole static-shape `cur_rank_inverse_indices` output (13312 entries), but `id_shuffle` writes only the first `num_unique` entries. The rest is whatever the allocator's memory held before. When that stale tail holds large values, the test's eager `cur_rank_unique_ids[cur_rank_inverse_indices]` (the first `CudaGatherNd<int, int>`) reads far out of bounds and faults. **Rate: 14801d8a8 7/300 (2.3%), d9e41753e 15/300 (5.0%), same signature every time; Fisher exact p = 0.13, so the two rates are not distinguishable.** A GPU-side check caught the bad indices without a crash: they always sit in the unwritten tail, and every written entry is in range. A copy of the test that indexes only the written prefix (`[:np_num_unique]`) ran **0 faults in 200 runs** (100 per build); at the measured rates about 7 were expected. The port changes none of the code involved (below). The 2026-11-13 gfx942 `validation-failed` was recorded for this fault. With it explained, the remaining question is jeff's: whether to re-record gfx942 at 14801d8a8. This job changes no state.

### Builds (two clean builds in this job's directory)
- Trees: one fresh clone of `AMD-Ecosystem/oneflow` (`agent_space/w/head`, detached at 14801d8a83043e852b72a3671bc88e5896aec904) plus `git worktree add --detach ../old d9e41753e07b814fe0cea985444397aef97709cc`. d9e41753e is unreachable from any branch after the squash, but `git fetch origin d9e41753e07b814fe0cea985444397aef97709cc` still served it. `git status --porcelain --untracked-files=no` was empty in both before and after.
- 14801d8a8: `build.sh head 56` = `rm -rf build && bash build_rocm.sh all` with `CMAKE_PREFIX_PATH=$R`, `ROCM_CLANG(_C)` from the SDK wheel, `CMAKE_HIP_ARCHITECTURES=gfx942`, `JOBS=56`, job-local cold ccache, `pywrap/python3` first on PATH (drops pip's `--user`). rc 0, 835 s (run at the same time as the first d9e41753e build), `OF_GPU_WARP_SIZE 64`. Import without any preload: `1.0.0+cu00.git.14801d8a8`, `LD_PRELOAD` None.
- d9e41753e, first try (same recipe): rc 0, 838 s, but **`import oneflow` failed: `liboneflow.so: undefined symbol: unw_getcontext`.** That configure found the host's GNU libunwind (`Found Unwind: /usr/include/x86_64-linux-gnu (found version "1.6.2")`), while ROCm clang compiled against LLVM's `<libunwind.h>`. This is the mismatch 14801d8a8's `cmake/cuda.cmake` fixes with `set(CMAKE_DISABLE_FIND_PACKAGE_Unwind ON)` (de15305bb). d9e41753e's own validation host evidently had no libunwind installed.
- d9e41753e, used: `build_old.sh 112` = `rm -rf build`, then build_rocm.sh's configure verbatim (the two trees' `configure()` are identical) plus `-DCMAKE_DISABLE_FIND_PACKAGE_Unwind=ON` on the command line, the same setting 14801d8a8 makes in source, with no fork edit. rc 0, 210 s (the ccache was warm from the first try). No `unw_` symbol left in liboneflow.so.
- **d9e41753e cannot run without the libLLVM preload.** Without it, `import oneflow` aborts with `realloc(): invalid pointer` (`therock-libllvm-symbol-interposition`). The version-script fix in `oneflow/ir/install-llvm.cmake` landed after d9e41753e, and it cannot be supplied from the command line. So **the d9e41753e arm ran with `LD_PRELOAD=$R/lib/llvm/lib/libLLVM.so.24.0git`**, its own validated recipe ("Validation 2026-11-01 (linux-gfx942 ...)"). The 14801d8a8 arm ran with no preload. This is the only intended difference between the arms besides the code. The preload changes host-side symbol binding, not device memory contents. Smoke: 64 x 1000 GPU softmax sums to 64.0; the test file 4 passed.

### Rate (`rate.sh 300 50`: one fresh `pytest -v` process per run of `expensive/test_id_shuffle.py`, `--timeout=600` plus `timeout -k 30 900`, `PYTHONUNBUFFERED=1` so the running test id reaches the log before an abort; two lanes at once, GPU 4 and GPU 5, one build per lane, lanes swap builds every 50 runs, 17:10-18:08)
| build | GPU 4 | GPU 5 | total |
|---|---|---|---|
| 14801d8a8 (no preload) | 4/150 | 3/150 | **7/300 (2.3%)** |
| d9e41753e (libLLVM preload) | 9/150 | 6/150 | **15/300 (5.0%)** |
- Every failure is rc 134: `HSA_STATUS_ERROR_MEMORY_FAULT`, `grid=[13312, 1, 1], workgroup=[256, 1, 1]`, faulting kernel `oneflow::(anonymous namespace)::CudaGatherNd<int, int>(oneflow::NdIndexSliceArgs, int const*, int const*, int*)` in 22 of 22. Then `cuda_device.cpp:122 Check failed: cudaSetDevice(device_index_) : an illegal memory access was encountered (700)` in 20; the other 2 (d9e41753e GPU 4 run 60, GPU 5 run 130) abort through a thrown `Error: an illegal memory access was encountered` in the same thread instead. No other rc occurred. Mean run time: 9.6 s for 14801d8a8, 10.0 s for d9e41753e.
- Failing test (the last `::test_` id the log printed): `test_embedding_gradient_shuffle` in 21 of 22, `test_id_shuffle` in 1 (14801d8a8 GPU 5 run 199). Both index `cur_rank_inverse_indices` the same way.
- Failing runs (full stderr kept in `agent_space/rate/keep/<build>_gpu<g>_run<i>.log`; faulting address in parentheses): 14801d8a8 GPU 4 runs 11 (0x71459945c000), 45 (0x786508c2f000), 112 (0x7ee11f42f000), 125 (0x778b91cf9000); GPU 5 runs 172 (0x7e8071cf9000), 199 (0x70b608a8e000), 284 (0x7cbc0dcf9000). d9e41753e GPU 4 runs 60, 90, 98, 170, 184, 194, 269, 293, 296; GPU 5 runs 4, 6, 39, 113, 130, 213.
- Against the validation's 1 in 90: that is 1.1%, within the noise of 2.3% (7/300 vs 1/90, Fisher p = 0.69). The higher rate here may come from back-to-back fresh processes on both GPUs, but it is not significant either way.

### Which launch, and whether its indices are in range
- `AMD_LOG_LEVEL=3` (`lvl3.sh`, until the first fault, one lane per build): **d9e41753e faulted on run 10 of 10** (GPU 5, `agent_space/lvl3/old_l3/run10.log`, 5 MB). 14801d8a8 ran 13 clean runs before I stopped the lane, because one capture was enough. Level 3 left the run time unchanged (mean 9.7 s over the 23 runs), and the d9e41753e fault came at about its base rate, so logging did not change the behaviour. HIP_LAUNCH_BLOCKING was not needed. The log names the launch. After the graph's last `hipMemcpyAsync` and `hipEventRecord`, the same stream gets two back-to-back `hipLaunchKernel(..., {52,1,1}, {256,1,1}, ...)` of `CudaGatherNd<int, int>`, 52 x 256 = 13312 threads. These are the test's `cur_rank_unique_ids[cur_rank_inverse_indices]` and then `[inverse_unique_partition_indices]`. Both have dense shape (13312) and 13312 indices, so the grid cannot tell them apart, and level 3 does not print kernarg contents. The hang dump has the queue at `rptr=137, wptr=140`. The index checks below say it is the first gather.
- `t/grad_check.py`: `_test_embedding_gradient_shuffle`'s graph, all 8 cases. Before each eager gather, the eager stream counts out-of-range indices on the GPU with no host sync in between, then clamps them so nothing faults. After `flow.cuda.synchronize()` it counts again and checks positions against `cur_rank_num_unique`. In one process for 100 iterations: 800 cases per build, **0 bad**. That matches the validation's in-process probe (6000 iterations, clean). In steady state the pool hands back the same buffers, whose tail holds earlier in-range values.
- **In fresh processes** (`chk.sh grad_check.py 1 120`: one iteration of the 8 cases per process, head on GPU 4 and old on GPU 5, swapped halfway; plus 12 processes of a first version without the position output): bad indices caught in **4 of 120 14801d8a8 processes and 2 of 120 d9e41753e processes** (plus 1 of 12 in the first version, d9e41753e). Every catch is the same case, `quant=0 fp16=1 emb=128`, the first unquantized case after the four quantized ones. Every catch reads, e.g. `pre_inverse=4 pre_partition=0 post_inverse=4 post_partition=0 values_ok=True cri_max=46407680 iupi_max=999 n0=13312 num_unique=1000 bad_positions_min=1284 bad_below_num_unique=0 first_num_unique_in_range=True`. In words: 4 entries of `cur_rank_inverse_indices` are out of range (up to 757034504) and still out of range after a full sync, so this is not a stream race. They all sit at positions 1156 to 8580, past `num_unique = 1000`. All 1000 written entries are in range, `inverse_unique_partition_indices` is always in range, and the reconstructed ids are exact. No process crashed.
- The code agrees. The op gives `cur_rank_inverse_indices` the static shape `num_ids * parallel_num` (`oneflow/user/ops/data_shuffle_op.cpp:82`). `IdShuffle`'s second `UniqueAndPartition` writes `reverse_index[i]` only for `i < received_elem_cnt` (`one_embedding_data_shuffle.cuh`, `HashTableUniqueAndPartitionPairs`), and nothing clears the rest. The embedding-shuffle ops take the count from `num_unique_matrix` and never read the tail; only the test's reference check does. The fault happens only when a fresh allocation's stale tail holds a large value, which is why it is first-test-in-process heavy (21 of 22 in the first test function to run).
- Confirmation (`ratep.sh 100 25`, the same harness on `t/prefix/test_id_shuffle_prefix.py`, a copy of the test outside the fork whose four reference gathers index `cur_rank_inverse_indices[:np_num_unique]`; nothing else changed): **0 memory faults in 200 runs.**
| build | GPU 4 | GPU 5 |
|---|---|---|
| 14801d8a8 (no preload) | 50 pass of 50 | 50 pass of 50 |
| d9e41753e (libLLVM preload) | 49 pass, 1 assertion failure, of 50 | 50 pass of 50 |

At the measured rates, about 2.3 faults were expected on 14801d8a8 and 5 on d9e41753e. Seeing none of an expected 7.3 has Poisson probability of about 0.0007.
- **A separate, rare, unexplained assertion (open):** d9e41753e GPU 4 run 39 (`agent_space/ratep/keep/old_gpu4_run39.log`) failed `test_embedding_gradient_shuffle`'s final gradient `np.allclose` (test line 290 of the copy: `AssertionError: False is not true`). Nothing faulted, and the other three test functions passed. In the original test the faulting gather comes before this assertion, so a fault can hide it. I reran only that test function, on a copy that prints the case, the tolerance and the worst element when the comparison fails (`ratep2.sh 60 30`, `t/prefix2/test_id_shuffle_prefix2.py`): **0 of 60 runs per build failed**. So it is 1 in 320 runs overall, on d9e41753e only, with the case unknown. I stopped there because of the budget. It may be the quantized case's int8 round-trip against the numpy reference (tolerance 0.5) or the fp16 sum order (1e-2), but that is not established. The next step is the `prefix2` loop for a few hundred runs, to name the case.

### Does the port touch any of this?
`git diff origin/master 14801d8a8` over `one_embedding_data_shuffle.cuh`, `nd_index_slice_kernels.cu`, `nd_index_slice_util.h`, `ops/data_shuffle_op.cpp` and the test file: no change. `data_shuffle_kernel.cu`: +8 -1, `kWarpSize` from `OF_GPU_WARP_SIZE` and `OF_SHFL_FULL_MASK`, in the quantize warp reduction only. Between d9e41753e and 14801d8a8, 34 files changed, none of them id_shuffle, gather_nd or the test. The one VM change is ebd0b3938 (`EnqueueNNGraph` before publishing the job), which cannot explain stale values that are still there after a sync.

### Conclusion
- Rate at the submitted head 14801d8a8: 7/300 runs (2.3%; 4/150 on GPU 4, 3/150 on GPU 5). At d9e41753e: 15/300 (5.0%; 9/150, 6/150). It predates the submitted head and was already present at d9e41753e, the head gfx942 last validated. Its earlier gfx942 records only missed it (5 rounds there, 89 of 90 here).
- Cause: upstream's test reads the uninitialized tail of `cur_rank_inverse_indices`. It is not a GPU-kernel or HIP-port defect, and id_shuffle's outputs are correct. On CUDA the same tail is equally uninitialized, so the test is latent there too; whether it faults depends on stale contents and the address map. Not checked on NVIDIA.
- Open, separate from the fault: one gradient `allclose` failure in 320 runs of the prefix copies (d9e41753e only, case unknown; details above).
- For jeff's ruling, not acted on here. (1) Whether this explanation turns the 2026-11-13 gfx942 `validation-failed` at 14801d8a8 into `completed`; this job recorded no state. (2) Whether to fix the test (index `cur_rank_inverse_indices[:np_num_unique]` in `_test_id_shuffle` and `_test_embedding_gradient_shuffle`): on the port branch with a regression note, or as a separate upstream fix. The `moat-port` branch is behind an open upstream PR, so a fix there would be a `moat-fix-<pr#>` round.

### Other checks
- No fork edit and no push: both trees clean (`--untracked-files=no`) at the end; nothing pushed anywhere except this notes commit on `port/oneflow`.
- Telemetry: the three compile phases (835 s, 838 s, 210 s) are recorded. The test loops were not wrapped in timeit.
- Host load: 1-min load average 10 to 110 during the rate run (other jobs on the host); both builds saw the same load because they ran at the same time.

## Ruling 2026-11-13 (jeffdaily): linux-gfx942 completed at 14801d8a8

jeffdaily ruled that the 2026-11-13 gfx942 validation at 14801d8a8 counts as completed. Its one failure, the `CudaGatherNd<int, int>` memory fault in `expensive/test_id_shuffle.py`, is a defect in upstream's test: `id_shuffle` writes only the first `num_unique` entries of `cur_rank_inverse_indices` and the test gathers through all 13312, so the unwritten tail occasionally holds stale out-of-range values in a fresh process (see "Investigation 2026-11-13 (linux-gfx942): test_id_shuffle GatherNd fault rate": 7/300 at 14801d8a8, 15/300 at d9e41753e, 0/200 with the gather limited to the written part). Everything else in that run passed or was explained. Whether and where to fix the test is a separate decision.
## Investigation 2026-11-13 (linux-gfx90a): test_id_shuffle GatherNd fault rate

jeff's request: measure the rate of the `expensive/test_id_shuffle.py` abort that the 2026-11-13 gfx942 validation saw once in 90 runs (`HSA_STATUS_ERROR_MEMORY_FAULT` in `CudaGatherNd<int, int>`, then 700), and find out whether gfx90a shows it too. This records evidence only. No platform state was changed, nothing was pushed to the fork, and nothing was posted to GitHub.

Host: AMD Instinct MI250X / MI250 (`gfx90a`, wave64), `HIP_VISIBLE_DEVICES=0,1` pinned by the dispatcher, the two GCDs of one MI250X. Each run set `HIP_VISIBLE_DEVICES` to 0 or 1, inside the pin. EPYC 7702 x2 (128 cores), ROCm 10.1.0 SDK wheels (`$FLEET_VENV-base-links/_rocm_sdk_devel`), Python 3.11.17, numpy 1.26.4, protobuf 3.20.3. `ulimit -c 0` and `LD_PRELOAD` unset (`env -u LD_PRELOAD`) on every GPU run, `OMP_NUM_THREADS=4`, `ONEFLOW_TEST_DEVICE_NUM=1`. amd-smi showed no other load on GPUs 0/1 (GPUs 2/3 idle, 0 GB used). Scripts in `agent_space/` (gitignored): `env.sh`, `build.sh`, `loop.sh`, `loopv.sh`, `loopv3.sh`, `gpu0.sh`, `gpu1.sh`, `t/{gs_probe,fp16_oracle_sim,fp16_oracle_sim_atomic}.py`.

### Build (documented procedure, clean, this job's directory)
Fresh fork clone (`--reference` the c741328b7 gfx90a job's clone, `--dissociate`), `git checkout 14801d8a83043e852b72a3671bc88e5896aec904` = `git ls-remote origin moat-port`, `protect-fork` installed. `utils/timeit.sh oneflow compile -- bash agent_space/build.sh all` = `bash build_rocm.sh all` with `CMAKE_PREFIX_PATH=$R`, `ROCM_CLANG(_C)` from the SDK wheel, `CMAKE_HIP_ARCHITECTURES=gfx90a`, `JOBS=$MAX_JOBS` (128), job-local cold ccache, `pywrap/python3` first on PATH (drops `--user`). **rc 0, 647 s**, `CMAKE_BUILD_TYPE: Release`, `OF_GPU_WARP_SIZE: 64`, no preload. A smoke run passed 4 of 4 in 8.8 s. `git status --porcelain` showed no modified tracked file after the runs.

### Runs (one fresh pytest process per run: `python3 -m pytest -p no:cacheprovider -p no:randomly --timeout=600 <-q | -s -v> -rfE --junitxml=... python/oneflow/test/expensive/test_id_shuffle.py` under `timeout -k 30 900`, stdout and stderr per run to `j/<batch>/g<gpu>/r<i>.{out,err}`; 8.9-11.6 s per run, median 9.6 s)
| batch | config | GPU 0 | GPU 1 |
|---|---|---|---|
| A (`loop.sh`, 17:02-17:09) | `-q` (pytest fd capture on) | 48 runs: 0 abort, 0 fail | 46 runs: **1 abort (run 194, rc 134)**, 0 fail |
| B (`loopv.sh`, 17:09-17:34) | `-s -v` | 150 runs: 0 abort, 0 fail | 150 runs: 0 abort, **3 fail** (runs 192, 219, 291) |
| C (`loopv.sh` via `gpu0.sh`, 18:40-19:12) | `-s -v` | 200 runs: 0 abort, 0 fail | -- |
| L3 (`loopv3.sh` via `gpu1.sh`, 18:40-19:20) | `-s -v`, `AMD_LOG_LEVEL=3` | -- | 250 runs: 0 abort, **1 fail** (run 2116) |
| **total** | | **398 runs: 0 abort, 0 fail** | **446 runs: 1 abort, 4 fail** |

**Memory-fault-class aborts: 1 in 844 runs on this build (GPU 0: 0 of 398, GPU 1: 1 of 446).** Wrong-value failures: 4 of 844 (GPU 0: 0 of 398, GPU 1: 4 of 446). Batch A was meant to be the first 300 runs. I stopped it at 94 completed runs, because its `-q` config lets pytest's fd capture swallow an abort's stderr. Run 194's `.err` holds only `timeout: the monitored command dumped core`, and there is no junit file and no oneflow glog. The 94 completed runs are counted; the run in flight when it was stopped is not. Batch B is the full 300 with stderr kept. C and L3 add runs; L3 also tried to capture a fault with launch logging. 18:41-18:51 the gradient probe below ran on the same GPU as C runs 1001-1014 (GPU 0) and L3 runs 2001-2065 (GPU 1), so those runs were not alone on the GPU.

### The abort (run A194)
rc 134 (SIGABRT, `dumped core` with `ulimit -c 0`) is the gfx942 failure's exit code. It came during the file's tests on GPU 1, 12.7 s against the usual 9 s. Its error text was lost (above), so **I cannot confirm it was the `CudaGatherNd<int, int>` memory fault. It is the only abort, and it is consistent with that fault.** No other run in 844 aborted, and no log contains `MEMORY_FAULT`, `illegal memory` or `(700)`. Measured rate on gfx90a: at most 1 in 844 (0.12%), against 1 in 90 on gfx942. Together that is at most 2 in 934 runs.

### Launch logging (`AMD_LOG_LEVEL=3`)
The fault did not recur in 250 logged runs, so no faulting launch was captured. Level 3 does not change the run time (9.6 s, about 3 MB of stderr per run) and did not change the observed rate (0 in 250, against 0 in 550 unlogged `-s -v` runs). What level 3 does show: each `CudaGatherNd<int, int>` call is logged as `hipLaunchKernel(fn, {52,1,1}, {256,1,1}, kernarg, 0, stream)`. That is 52 x 256 = 13312 threads, the `grid=[13312,1,1]` of the gfx942 hang dump (the dump counts work-items), with the stream handle. The kernarg *values* (index and dense pointers, `NdIndexSliceArgs`) are not printed, so a capture at level 3 would name the launch and its stream order but not show whether an index was out of range. Release build: the device-side `assert` in `OffsetInSliceToOffsetInDense` (nd_index_slice_util.h:101) compiles out, so an out-of-range index reads out of bounds instead of trapping. HIP_LAUNCH_BLOCKING was not tried; with no reproduction there was no rate to compare.

### The wrong-value failures (new, not in the gfx942 record)
All 4 are `test_embedding_gradient_shuffle` at test_id_shuffle.py:292, the gradient `np.allclose`. The id reconstruction assert at :287 passed first in each, so the ids and indices were right and the gradient values were not. Logs: `j/B/g1/r{192,219,291}.{out,err}`, `j/L3/g1/r2116.{out,err}` (full level-3 trace).
- `t/gs_probe.py <iters>`: runs `_test_embedding_gradient_shuffle` for all 8 combinations, in process, and logs the mismatch when the test's `allclose` fails. **GPU 0, 40 iterations (320 calls): 0 mismatches. GPU 1, 150 iterations (1200 calls): 1 mismatch**, at iteration 149, `enable_quantize=False, fp16=True, embedding_size=128`. 16 of 1703936 elements failed, all column 48, at rows that reconstruct one unique id appearing 16 times. So one segment sum was off: device 0.00537 against reference -0.00684, diff 0.0122, tolerance atol = rtol = 1e-2. No non-finite values.
- For fp16 with `embedding_size=128` (no padding), the sum is upstream's `UnsortedSegmentRowSumGpu<__half2, ...>` with `cuda::atomic::Add` (named in r2116's trace), so the device accumulates in fp16 in no fixed order. The test's reference is a python `sum` over fp16 rows, which also rounds to fp16 at every add. The port does not change unsorted_segment_sum_kernel_util.cu, the half atomics, or this test.
- CPU simulation of the reference-precision explanation, with the test's id distribution and 128 columns: fp32 accumulation on the device side gives 0 failing draws of 40 (`t/fp16_oracle_sim.py`), and fp16 accumulation in random order gives 0 of 40 (`t/fp16_oracle_sim_atomic.py`). 40 draws cannot rule out a rate near 1%, so this neither confirms nor excludes reference precision. A 0.0122 deviation is far smaller than one dropped or doubled term (terms are uniform in [-1, 1]), so it looks like rounding, not a lost update.
- GPU split: 4 of 446 runs and 1 of 1200 probe calls on GPU 1, against 0 of 398 runs and 0 of 320 probe calls on GPU 0. With 4 to 5 events in all, the odds that every one lands on the same GCD by chance are roughly 1 in 8 to 1 in 16: suggestive, not established.

### Conclusion
- **The GatherNd memory fault is rare on gfx90a too: at most 1 abort in 844 runs at 14801d8a8, on GPU 1, with its error text lost.** I cannot say whether it is the same fault, so this does not establish that the fault happens on gfx90a, and so does not establish that it is not gfx942-specific. It also does not show gfx90a to be clean: one rc-134 abort in a file that otherwise never aborts on this host fits the same fault at a similar order of rate (gfx942 1/90; gfx90a at most 1/844). Even 1/90 on gfx942 rests on a single event, so these rates are not significantly different.
- Next step if jeff wants the fault named: a few thousand fresh-process runs with `-s` (never `-q`) split over both arches, with kernarg logging that prints values (for example `AMD_LOG_LEVEL=4`, or rocprofv3 kernel tracing). Separately, the gradient value miss on GPU 1 should be checked with more simulated draws, and with the reference summed in float64, before calling it reference rounding.
- Time: about an hour (17:34-18:40) went idle because my wait loop's `pgrep -f` pattern matched itself; batch B had already finished at 17:34. That idle hour is why there are 844 runs and not more.
- The 14801d8a8 gfx90a validation record stands on its own runs; this investigation changes no state.

### Against the gfx942 root cause (read after this section was written)
The gfx942 investigation above and jeff's ruling, which landed on this branch while these runs were going, trace the fault to upstream's test: it gathers through the unwritten tail of the static-shape `cur_rank_inverse_indices`, and the tail holds whatever the allocator's memory held before. That makes the rate a property of stale device memory contents, not of a kernel, so a much lower rate on gfx90a (at most 1 in 844, against 7 in 300 on gfx942 at the same head) fits the same mechanism. This host's fresh allocations happen to hold an out-of-range value in that tail far less often. The A194 abort is consistent with it but remains unconfirmed, because its text was lost. The gradient wrong-value failures above are a different symptom: the id reconstruction passed in each, so they do not involve the tail, and they remain open.

## Porter round 2026-11-13 (fix 10697)

jeff's request (2026-11-13): a small fix round on `moat-fix-10697`, because upstream is inactive and fixes ride on the port. Two items, both in upstream's tests, neither in the port's code: (1) `expensive/test_id_shuffle.py` gathers through the unwritten tail of `cur_rank_inverse_indices` (the GatherNd fault of the two investigations above); (2) `test_embedding_gradient_shuffle`'s intermittent fp16 gradient-value failure (4/446 on gfx90a, 1/320 on gfx942). Host MI300X (linux-gfx942), `HIP_VISIBLE_DEVICES=4,5` pinned by the dispatcher, each run inside the pin; `ulimit -c 0` and `LD_PRELOAD` unset on every GPU run. ROCm 10.1.0 SDK wheels, Python 3.11.17, numpy 1.26.4, protobuf 3.20.3. Scripts in this job's `agent_space/` (gitignored): `env.sh`, `build.sh`, `run1.sh`, `rate.sh`, `chain5.sh`, `final.sh`, `edit2.py`, `t/{sim,grad_probe,recheck,bound_stats}.py`.

### Result
- Staging tip `moat-fix-10697` = **50353bcb0**, two commits on 14801d8a8 (published `moat-port`, untouched):
  - d1eec6e13 `[ROCm] Index only the written id_shuffle inverse indices in tests`: `cur_rank_inverse_indices[:np_num_unique]` in the four host-side gathers of `test_id_shuffle.py` (`_test_id_shuffle` x2, `_test_embedding_gradient_shuffle` x2) and the two of `test_dynamic_allocation_gradient_shuffle.py`, which has the same gradient function. The other files in `expensive/` that use id_shuffle (`test_id_shuffle_global.py`, `test_id_shuffle_p2p_global.py`, `test_one_embedding_p2p_shuffle_global.py`) only pass `cur_rank_inverse_indices` to ops or slice outputs by `num_unique` already; `_test_unique_key_value`'s `inverse_indices` has one entry per input id, all written.
  - 50353bcb0 `[ROCm] Bound the fp16 gradient shuffle check by the summation error`: for `fp16 and not enable_quantized_comm` the test compares the op with the exact float64 sum of the float16 rows, per element, within gamma(n-1) * sum|x| (below). Quantized and float32 cases unchanged. Same change in `test_dynamic_allocation_gradient_shuffle.py`.
- Item 2 verdict: **the tolerance is too tight; the op is correct.** No C++/HIP/CUDA source changed in this round, so there is no CUDA path for nvcc to compile; the mi355 nvcc check was not needed and not run.
- Formatting: `black==19.10b0` (dev-requirements.txt) on both files. `python3 utils/jargon.py --port oneflow` clean (after `git fetch origin master:master`; without a local `master` it refuses to resolve the range), and `jargon.py -C projects/oneflow/src --commits 14801d8a8..moat-fix-10697 --diff 14801d8a8..moat-fix-10697` clean (2 commits). `utils/prose.py` clean on both bodies. ASCII only.
- **Commit-body error, not amended:** 50353bcb0's body says "In every failing case captured on an MI300X, the op was within 0.009 of the exact float64 sum and the reference within 0.003". That was true of the first three captures; across all 18 the maxima are 0.0151 (op) and 0.0090 (reference), still far inside the per-element bound. The no-amend rule stands, so it is registered as deferred `oneflow-50353bcb0-body-wording` for jeff: the fix round merges by fast-forward, so the body ships as written unless he rules otherwise.

### Build (documented procedure, clean, this job's directory)
Fresh clone of `AMD-Ecosystem/oneflow` at `projects/oneflow/src` (`--reference` the 084609 job's clone, `--dissociate`), `protect-fork` installed, `git checkout -b moat-fix-10697 14801d8a83043e852b72a3671bc88e5896aec904`. venv: `$FLEET_VENV/bin/python -m pip install -r dev-requirements.txt fsspec pytest-timeout pytest-repeat opencv-python-headless==4.10.0.84 portalocker protobuf==3.20.3 numpy==1.26.4`. `utils/timeit.sh oneflow compile -- bash agent_space/build.sh 56` = `rm -rf build && bash build_rocm.sh all` with `CMAKE_PREFIX_PATH=$R`, `ROCM_CLANG(_C)` from the SDK wheel, `CMAKE_HIP_ARCHITECTURES=gfx942`, `JOBS=56`, job-local cold ccache, `pywrap/python3` first on PATH (drops pip's `--user`): **rc 0, 746 s**, `OF_GPU_WARP_SIZE: 64`. Import with no preload: `1.0.0+cu00.git.14801d8a8`, `LD_PRELOAD` None; 64 x 1000 softmax sums to 64.0. Both edits are Python-only, so the one build serves every commit of the round.

### Item 1: fresh-process runs (`rate.sh`: one `pytest -v --capture=sys --timeout=600` process per run under `timeout -k 30 900`, `PYTHONUNBUFFERED=1`, failing logs kept in `rate/<lane>/keep/`)
| lane | file | GPU | runs | GatherNd faults (rc 134) | other failures |
|---|---|---|---|---|---|
| fixed (20:09-21:05) | d1eec6e13 `test_id_shuffle.py` | 4 | 320 | **0** | 2 (`test_embedding_gradient_shuffle` :292, item 2) |
| control (20:09-20:27) | 14801d8a8 `test_id_shuffle.py` (copy in `agent_space/t/old/`) | 5 | 100 | **3** (runs 9, 16, 19: `HSA_STATUS_ERROR_MEMORY_FAULT`, `CudaGatherNd<int, int>`, then `cudaSetDevice ... (700)`; all during `test_embedding_gradient_shuffle`) | 1 (run 90, :292, item 2) |
| final (20:52-21:22) | 50353bcb0 `test_id_shuffle.py` (`t/new/`, blob f3fbae9b = the tip's) | 5 | 160 | **0** | 0 |
| final (21:05-21:35) | same | 4 | 160 | **0** | 0 |
- **GatherNd faults with the fix: 0 in 640 fresh runs** (320 at d1eec6e13 + 320 at 50353bcb0, both GPUs). 0/320 at d1eec6e13 against the control's 3/100 on the same build and day: one-sided Fisher p = 0.013; against the 7/300 of the gfx942 investigation, p = 0.006. The GatherNd fault is gone with the fix.

### Item 2: cause of the fp16 gradient mismatch
- **The op's accumulation.** In one process (1n1d), `EmbeddingGradientShuffleKernel` sums in `UniquePartitionEmbeddingGrad` -> `UnsortedSegmentSum` (`one_embedding_data_shuffle.cuh`): memset 0, then `UnsortedSegmentSumKernelUtil<half>`, which for even `embedding_size` (128) takes the `half2` path, one thread per half2 element doing `cuda::atomic::Add(half2*)`; for 17 it pads to 18 and uses `UnsortedSegmentHalfGpu`, also half2 atomics. Each id's rows (up to ~32; mean 13.3) are added in whatever order the atomics land, every add rounded to fp16. On HIP, `AddImpl(half2*)` is the generic CAS loop (`AtomicCASBinaryImpl` with `a + b` on half2, 32-bit `atomicCAS`), so no update can be lost; CUDA sm_60+ uses native `atomicAdd(half2*)`. The second stage (`UniqueCurRankEmbeddingGrad`) adds one received row into a zeroed buffer, which is exact. The port changes none of this (`git diff origin/master 14801d8a8` over `one_embedding_data_shuffle.cuh` and `unsorted_segment_sum_kernel_util.cu`: empty; `atomic.cuh` only drops the `__trap` for `USE_HIP`).
- **The test's reference** is `sum(rows)` over float16 numpy rows in index order: also an fp16 add per row, in a different order. Tolerance `atol = rtol = 1e-2`.
- **GPU capture** (`t/grad_probe.py`: the test's fp16/128/unquantized graph and data distribution, per call compares the op's per-id sums with the test's reference and with a float64 sum of the same fp16 rows, and saves every case the test's `isclose` rejects to `probe/*/fail_*.npz`; GPU 5): in one process, 4000 calls: **17 fail the old check (0.43%)**; 150 fresh processes x 4 calls: 1 fail (call 0 of process f21). Each failure is exactly one element. All 18:

| id | col | rows | op | reference (fp16, index order) | exact (float64) | \|op-ref\| | old tol | \|op-exact\| | \|ref-exact\| | bound |
|---|---|---|---|---|---|---|---|---|---|---|
| 190 | 48 | 20 | -0.054688 | -0.065918 | -0.062836 | 0.011230 | 0.010659 | 0.008148 | 0.003082 | 0.119 |
| 321 | 18 | 22 | 0.084473 | 0.095703 | 0.093185 | 0.011230 | 0.010957 | 0.008713 | 0.002518 | 0.133 |
| 189 | 34 | 30 | -0.040527 | -0.051270 | -0.049484 | 0.010742 | 0.010513 | 0.008957 | 0.001785 | 0.211 |
| 922 | 93 | 21 | 0.005371 | -0.004944 | 0.002567 | 0.010315 | 0.010049 | 0.002804 | 0.007511 | 0.099 |
| 45 | 85 | 29 | -0.164062 | -0.151367 | -0.160065 | 0.012695 | 0.011514 | 0.003998 | 0.008698 | 0.207 |
| 905 | 65 | 22 | 0.049805 | 0.039062 | 0.041595 | 0.010742 | 0.010391 | 0.008209 | 0.002533 | 0.126 |
| 664 | 115 | 28 | -0.405762 | -0.391113 | -0.391754 | 0.014648 | 0.013911 | 0.014008 | 0.000641 | 0.218 |
| 778 | 77 | 22 | -0.064453 | -0.053711 | -0.060791 | 0.010742 | 0.010537 | 0.003662 | 0.007080 | 0.115 |
| 793 | 116 | 23 | 0.330566 | 0.346680 | 0.345673 | 0.016113 | 0.013467 | 0.015106 | 0.001007 | 0.138 |
| 885 | 105 | 24 | -0.022461 | -0.034668 | -0.031677 | 0.012207 | 0.010347 | 0.009216 | 0.002991 | 0.164 |
| 249 | 15 | 26 | 0.745117 | 0.727539 | 0.731201 | 0.017578 | 0.017275 | 0.013916 | 0.003662 | 0.169 |
| 138 | 16 | 21 | 0.069824 | 0.059082 | 0.067383 | 0.010742 | 0.010591 | 0.002441 | 0.008301 | 0.133 |
| 328 | 26 | 26 | 0.007080 | -0.003418 | 0.001099 | 0.010498 | 0.010034 | 0.005981 | 0.004517 | 0.142 |
| 121 | 54 | 25 | 0.040527 | 0.029053 | 0.037231 | 0.011475 | 0.010291 | 0.003296 | 0.008179 | 0.162 |
| 730 | 97 | 24 | 0.070557 | 0.059326 | 0.060944 | 0.011230 | 0.010593 | 0.009613 | 0.001617 | 0.153 |
| 202 | 8 | 20 | -0.065674 | -0.052734 | -0.061768 | 0.012939 | 0.010527 | 0.003906 | 0.009033 | 0.127 |
| 220 | 36 | 17 | 0.056641 | 0.067383 | 0.062744 | 0.010742 | 0.010674 | 0.006104 | 0.004639 | 0.091 |
| 77 | 25 | 22 | -0.038086 | -0.026855 | -0.032959 | 0.011230 | 0.010269 | 0.005127 | 0.006104 | 0.149 |

  (Last row from the fresh-process run, `probe/fresh/fail_f21_0.npz`.) The input terms of each in-process case are in `probe/inproc.log`. Pattern: long segments (17 to 30 rows), small exact sums, the op and the reference on opposite sides of the exact value, each off by a few fp16 ulps of the partial sums (ulp is 2^-9 to 2^-8 for partial sums of 2 to 8). Where the sum is near zero, `rtol` adds nothing, so two correct fp16 sums 0.0105 to 0.018 apart fail.
- **Against float64 over all 4000 + 600 calls (512M + 77M elements): max |op - exact| = 0.0283, max |reference - exact| = 0.0325, and max |op - exact| / bound = 0.999 (never above 1).** The op's error is no larger than the reference's own fp16 error.
- **CPU model** (`t/sim.py`, 8000 draws: the test's ids and uniform[-1, 1] fp16 gradients, reference = fp16 sum in index order, "device" = fp16 sum in a random order, either independent per half2 column pair or one order per id): the old check fails in **32 of 8000 draws (0.40%)** (pair 15/4000, row 17/4000); max |device - exact| 0.022 to 0.030 per 1000 draws. The GPU's 0.43% and 0.028 match a correct order-independent fp16 sum. A lost or doubled update would move a sum by one term (mean |x| = 0.5), not by 0.01.
- **Rates seen in full test runs:** this round 2/320 (fixed lane) + 1/100 (control) on gfx942 (0/320 at the tip, where the check is the new one); gfx90a 4/446 and gfx942 1/320 earlier: 8 of 1186 runs (0.67%) with the old check, against 0.4% per call predicted (the test makes one such call per run; Poisson P(8 or more) at 4.7 expected is 0.11, so consistent).

### Item 2: the new tolerance and how it was derived
- Recursive summation of n floating-point numbers in any order, each add rounded to nearest with unit roundoff u, satisfies |computed - exact| <= gamma(n-1) * sum|x_i|, gamma(m) = m*u / (1 - m*u) (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., section 4.2). For float16, u = 2^-11. The fp16 inputs are the same on both sides (numpy `astype(np.float16)` and the device cast are both round-to-nearest; the n = 1 elements matched exactly in every probe call), the float64 sum of at most ~40 fp16 values is exact, and the op's second stage is exact, so a correct op can never exceed this bound, in any atomic order. The test computes it per element from that element's rows.
- Measured against it: max |op - exact| / bound = 0.999 over 589M elements (n = 2 elements reach it), and the 18 old-check failures sit at 0.48 to 0.96 of it (`t/recheck.py`, all pass the new check).
- Strength (`t/bound_stats.py`, 50 draws of the test's data): the bound's median is 0.039, 99th percentile 0.124, max 0.30 (ids with up to 32 rows); the old effective tolerance was 0.01 + 0.01*|expected|, so the median is about the old tolerance at |expected| = 3. One dropped or doubled row would exceed its element's bound in 95.3% of elements; over the 1.7M elements a call compares, any systematic summation fault fails it.
- Comparing against the exact sum instead of the fp16 sequential sum removes the reference's own rounding (up to 0.0325 above), which is why the bound covers the op's error alone rather than the sum of two errors.

### Final runs at the staging tip
Both test files at the tip, run from blob-identical copies in `agent_space/t/new/` (`git ls-tree` blobs f3fbae9b `test_id_shuffle.py`, 9bb763a5 `test_dynamic_allocation_gradient_shuffle.py`) so the item-1 lane's file was never edited under it; `final.sh`, one fresh pytest process per run:
- `expensive/test_id_shuffle.py`: **320 of 320 passed** (160 on GPU 4, 160 on GPU 5), 0 memory faults, 0 gradient failures. At the old check's 0.4% per call about 1.3 gradient failures were expected; the new check cannot fail on a correct op.
- `expensive/test_dynamic_allocation_gradient_shuffle.py`: **50 of 50 passed** (25 per GPU).
- Pushed `moat-fix-10697` (plain push, new branch, protect-fork hook passed): `git ls-remote` shows `moat-fix-10697` 50353bcb0c935d019f69e590fa7d59dc6907a509, `moat-port` 14801d8a83043e852b72a3671bc88e5896aec904 (unchanged). `git status --porcelain --untracked-files=no` empty.

### Records
- `set-state --agent porter oneflow linux-gfx942 porting` (lock, 15c03b91), then `advance-head oneflow 50353bcb0c935d019f69e590fa7d59dc6907a509` and `set-state --agent porter oneflow linux-gfx942 ported` (stage `ported`, lock released), so a reviewer covers 14801d8a8..50353bcb0. The gfx942, gfx90a and gfx1100 records stay at `validated_sha` 14801d8a8 and read `revalidate` against the new head. Deferred `oneflow-50353bcb0-body-wording` registered (above). `moat-port` still at 14801d8a8; nothing else pushed to the fork. No gh writes.
- Telemetry: compile 746 s recorded by timeit; the test loops were not wrapped in timeit.

### Fix 10697: tip commit reworded (2026-11-13)

jeff authorized amending the message of 50353bcb0 and force-pushing `moat-fix-10697`. The one wrong figure ("the op was within 0.009 ... the reference within 0.003", true only of the first 3 of 18 captured cases) now reads 0.016 and 0.009. The new tip e806d8cc8 has the same tree and parent (d1eec6e13) as 50353bcb0; it was built with `git commit-tree` and pushed with `--force-with-lease` on the old sha. `moat-port` is untouched at 14801d8a8. head_sha advanced to e806d8cc8; deferral `oneflow-50353bcb0-body-wording` done. No review or validation had run at 50353bcb0, so none is lost.

## Review 2026-11-13 (fix 10697)

Reviewer, linux-gfx942 (MI300X, `HIP_VISIBLE_DEVICES=5` pinned by the dispatcher, `ulimit -c 0`, `LD_PRELOAD` unset). /pr-review local-branch `moat-fix-10697`, delta 14801d8a8...e806d8cc8 (d1eec6e13, e806d8cc8). Fresh fork clone in this job; `git ls-remote`: `moat-fix-10697` e806d8cc80b6, `moat-port` 14801d8a8 (unchanged). e806d8cc8 and 50353bcb0 share tree d3efbbc5e7 and parent d1eec6e13. Nothing pushed to the fork, no gh writes.

**Verdict: changes-requested, on the e806d8cc8 commit message only. The code in both commits is correct, and no further code change is needed.**

### Problems (both in e806d8cc8's body, upstream-visible)
1. "In every failing case captured on an MI300X, the op was within 0.016 of the exact float64 sum and the reference within 0.009." The reference figure is false: across the 18 captures, max |reference - exact| = **0.009033** (id 202 col 8, `probe/inproc/fail_inproc_3522.npz` in the porter job). I recomputed every capture from its saved `ids` and `grad`, not from the table (`agent_space/recheck.py`). The op figure rounds 0.015106 up to 0.016, so by the same convention the reference figure should read 0.010. Fix: "within 0.016 ... and the reference within 0.010" (or "0.0152 ... 0.0091").
2. "in 8 of 1186 runs on MI300X and MI250X GPUs: one gradient element off by 0.010 to 0.018". The 8 count is right (2/320 + 1/100 this round, 4/446 gfx90a, 1/320 gfx942 earlier). But no run log records which element failed or by how much. The kept logs (`rate/fixed/keep/run{65,202}.log`, `rate/old/keep/run90.log`) show only `AssertionError: False is not true`, and the earlier gfx942 one is recorded as "case unknown". "One element, 0.010 to 0.018" comes from the 18 probe captures plus the gfx90a probe's 0.0122, not from those runs. That the 8 were the fp16 unquantized case does hold by analysis: the quantized tolerance (0.5) and the float32 tolerance (1e-4 in test_id_shuffle.py, 1e-3 in the dynamic-allocation file) cannot be crossed by rounding alone. Fix: attribute the magnitude to the captures, e.g. "...in 8 of 1186 runs on MI300X and MI250X GPUs. In 18 failing calls captured on an MI300X, one gradient element was off by 0.010 to 0.018 against a tolerance of ...".

Either fix means rewording e806d8cc8 again, which needs jeff's authorization for another `--force-with-lease` on `moat-fix-10697`, as last time. Alternatively jeff may rule the wording acceptable as written, and then this review's code verdict stands as passed.

### Verified, no finding
- **Diff scope:** Python only, two files (`expensive/test_id_shuffle.py` +40 -13, `expensive/test_dynamic_allocation_gradient_shuffle.py` +36 -9). No C++/HIP/CUDA/CMake change, so no nvcc check is owed.
- **Item 1, completeness:** every host-side gather through `cur_rank_inverse_indices` in `python/oneflow/test/` is in these two files, and all six are sliced `[:np_num_unique]` (test_id_shuffle.py:77, :83, :292, :297; dynamic:145, :150). Both test classes are `skip_unless_1n1d`, so the written length is `received_elem_cnt` = first-stage num_unique = `np_num_unique`, which the test asserts against `cur_rank_num_unique`. `HashTableUniqueAndPartitionPairs` writes `reverse_index[i]` only for `i < num_keys` (one_embedding_data_shuffle.cuh:50-98), and the output's static shape is `num_ids * parallel_num` (data_shuffle_op.cpp:82). `_test_unique_key_value`'s `inverse_indices` has `keys_shape` (data_shuffle_op.cpp:39), so every entry is written. The global/p2p files only pass `cur_rank_inverse_indices` to ops or slice by `num_unique`. Their `flow._C.gather(data, cur_rank_unique_ids)` reads the unwritten tail of `cur_rank_unique_ids`, but that gather is bounds-checked (gather_kernel_util.cu:37: out of range yields 0) and its consumer reads only `num_unique` rows, so it is safe.
- **Item 1, coverage:** `[inverse_unique_partition_indices]` indexes the first gather with values < num_unique, so slicing changes no value the test compares. A wrong written entry still fails `array_equal`. Nothing that checked the full output lost coverage.
- **Item 2, derivation:** the op memsets to 0 and then adds n rows (one_embedding_data_shuffle.cuh:403-405), so it makes n-1 rounding adds; the 1n1d second stage adds one row into a zeroed buffer, which is exact. That makes gamma(n-1) * sum|x| (Higham 4.2, any order) the right bound. `n = np_rows.shape[0]` is taken per unique id, and every column of an id has the same n. u = 2**-11 is fp16's. The float64 sum of at most ~35 fp16 values in [-1, 1] is exact, and my independent recompute matched the saved float64 sums bit for bit in all 18 captures (0 mismatches). It also gives max |op - exact| / bound 0.96 within them. The comparison indexes the per-id arrays with `np_inverse`, matching the op output's row order (the id check just before passes).
- **Item 2, catches faults (GPU, `agent_space/inject.py` against the tip's test functions, porter job's gfx942 build run read-only: checkout 50353bcb0, same tree, clean).** The fault goes into the graph's `embedding_grad` input while the reference keeps the original. 20 trials per cell, with fp16 unquantized at sizes 128 and 17, in both files:

  | fault | test_id_shuffle.py | dynamic_allocation |
  |---|---|---|
  | one row dropped | 40/40 caught | 40/40 |
  | one row doubled | 40/40 | 40/40 |
  | one element dropped | 37/40 | 39/40 |
  | one element doubled | 38/40 | 35/40 |

  Rows: 160/160. Single elements: 149/160 = 93%, consistent with the commit's "about 95% of elements" (95.3% simulated). With no fault: 3/3 pass.
- **Float32 and quantized:** the code is unchanged (they take the `else` branch, `check_fp16_sum_bound` False). Full files, one pytest process per file: test_id_shuffle.py 4 passed x3, test_dynamic_allocation_gradient_shuffle.py 1 passed x3. Running both files in one pytest process fails `test_embedding_shuffle` (test_id_shuffle.py:163, `cur_rank_embeddings elem_cnt should be 1`), because the dynamic file sets `ONEFLOW_ONE_EMBEDDING_USE_DYNAMIC_MEMORY_ALLOCATION=1` at module level (line 20). That predates this round, and upstream runs the files separately.
- **d1eec6e13 message:** accurate ("about 2%" = 7/300 and 3/100; mechanism matches the code). Titles are 66 and 68 characters. `python3 utils/jargon.py --port oneflow` clean, `jargon.py --commits/--diff 14801d8a8..moat-fix-10697` clean, `utils/prose.py` clean on both bodies, ASCII-only bodies and added lines, "Authored with Claude.", no Co-Authored-By. The other e806d8cc8 figures check out: 17/4000 (0.43%), CPU model 0.40%, up to about 32 rows, and no element over the bound.
- The fleet guard refused installing the test requirements into this job's venv, so I built nothing here and did the GPU checks on the porter job's existing build, which I did not modify.

### Fix 10697: tip commit reworded again (2026-11-13)

The review at e806d8cc8 found two overstatements in its message: the reference's largest miss across the 18 captures is 0.009033, not "within 0.009", and the "one element off by 0.010 to 0.018" range comes from the 18 probe captures, not the 8 failing runs (whose logs show only `AssertionError`). jeff approved the corrected text and a second `--force-with-lease` on `moat-fix-10697`. The paragraph now reads "... in 8 of 1186 runs on MI300X and MI250X GPUs. In 18 failing calls captured on an MI300X, one gradient element was off by 0.010 to 0.018 against ..." and "In every one of those 18 cases, the op was within 0.0152 of the exact float64 sum and the reference within 0.0091." The trailing blank line the first reword added is gone. New tip 44d5f04cd: same tree and parent (d1eec6e13) as 50353bcb0 and e806d8cc8; it differs from 50353bcb0 only in that paragraph. `moat-port` untouched at 14801d8a8. head_sha advanced to 44d5f04cd and the stage set back to `ported` for a wording-only re-review; the code verdict of the e806d8cc8 review (same tree) stands.

## Re-review 2026-11-13 (fix 10697)

Reviewer, linux-gfx942, wording-only re-review of the reworded tip 44d5f04cd. Fresh fork clone in this job; `git ls-remote`: `moat-fix-10697` 44d5f04cdda6, `moat-port` 14801d8a8 (unchanged). e806d8cc8 is no longer on the fork, so it was fetched from the previous reviewer job's clone. No build, no GPU run, nothing pushed to the fork, nothing written to GitHub.

**Verdict: review-passed.** `git diff e806d8cc8 44d5f04cd` is empty (both tree d3efbbc5e7), and 44d5f04cd's parent is d1eec6e13, so the code verdict of "Review 2026-11-13 (fix 10697)" carries over unchanged. The message differs from e806d8cc8 only in the one paragraph and the dropped trailing blank line.

### Verified, no finding
- Both previous problems are fixed: the magnitude "0.010 to 0.018" is now attributed to the 18 captured failing calls, not the 8 runs, and the extremes read "within 0.0152 ... and the reference within 0.0091".
- Every figure recomputed from the porter job's 18 saved captures (`agent_space/probe/{inproc,fresh}/fail_*.npz`, from `ids` and `grad`, not from the table; `agent_space/rr.py` in this job): one failing element per call in all 18; |op - reference| 0.010315 to 0.017578 (message: 0.010 to 0.018); max |op - exact| 0.015106 (0.0152) and max |reference - exact| 0.009033 (0.0091); old tolerance `isclose(rtol=atol=1e-2)` = 0.01 + 0.01 * |expected|. The other figures match the notes: 8 of 1186 runs (2/320 + 1/100 + 4/446 + 1/320; MI300X = gfx942, MI250X = gfx90a), 17/4000 = 0.43%, CPU model 32/8000 = 0.40%, up to about 32 rows, about 95% (95.3%), u = 2**-11, no element over the bound (max ratio 0.999 over 589M elements, the porter's figure; not re-derivable from the captures).
- Title 67 characters, `[ROCm]` prefix. `python3 utils/jargon.py --port oneflow` clean (local `master` fetched), `jargon.py -C projects/oneflow/src --commits/--diff 14801d8a8..44d5f04cd` clean, `utils/prose.py` clean on the body, ASCII only, "Authored with Claude.", no agent trailer, message ends with exactly one newline.
- Records note, not a finding on the fork: the porter round's "the 18 old-check failures sit at 0.48 to 0.96 of it" is the largest |op - exact| / bound over all elements of each failing call; the failing elements themselves sit at 0.018 to 0.110 of their bound (e.g. id 793: 0.015106 / 0.138). The commit message does not cite this ratio.

## Validation 2026-11-13 (linux-gfx1100) at 44d5f04cd (validator, NOT RUN: the fleet guard refused the test-requirements install; no state change)

jeff's request (2026-11-13): validate the fix round `moat-fix-10697` at 44d5f04cdda699579b9f0825301b92317499f138 on linux-gfx1100 (dispatcher pin `HIP_VISIBLE_DEVICES=2,3`), with the full recipe plus 200 fresh runs of `expensive/test_id_shuffle.py` and 50 of `expensive/test_dynamic_allocation_gradient_shuffle.py`. jeff's instructions said that if a guard refuses anything, the job stops and records it, without routing around it, even if it looks like a false match. That happened before the build, so this job has no GPU evidence. linux-gfx1100 keeps its record (`completed`, `validated_sha` 14801d8a8). The selector still reads it `revalidate` at 44d5f04cd.

### What ran
- `bash utils/orient.sh` on `port/oneflow`: `next: projects/oneflow state=revalidate -> dispatch: validator`, platform linux-gfx1100.
- Fresh fork clone at `projects/oneflow/src` (`--reference` the c741328 gfx1100 job's clone, `--dissociate`), `protect-fork` installed. **Head confirmed:** `HEAD` = `git ls-remote origin moat-fix-10697` = 44d5f04cdda699579b9f0825301b92317499f138, `moat-port` = 14801d8a83043e852b72a3671bc88e5896aec904 (unchanged). `git diff --stat 14801d8a8 HEAD`: the two test files only (+74 -24). `git status --porcelain`: empty.
- `python3 utils/jargon.py --port oneflow`: clean.

### Where it stopped
- The job venv (`$FLEET_VENV`, `venv-base-links/_rocm_sdk_devel` = ROCm 10.1.0 SDK, Python 3.11.17) has numpy 1.26.4 but protobuf 6.33.5, and lacks pytest-timeout and opencv. Every earlier gfx1100 recipe installs them first. I sent this command: `<job>/venv/bin/python -m pip install -q -r dev-requirements.txt fsspec pytest-timeout pytest-repeat opencv-python-headless==4.10.0.84`, with the venv written as `$J/venv` (`J` = the job directory). The PreToolUse fleet guard blocked it: "pip changing the Python environment every job on this host shares (a job dispatched with --build has its own in $FLEET_VENV) ... Do not look for another way to do it". The whole Bash call was blocked, so nothing in it ran.
- Why it matched: `guard.py` `verdict()` refuses any `pip install` whose command text contains neither the literal `FLEET_VENV` nor the expanded `$FLEET_VENV` path. `$J/venv` is that same directory, but the text is neither string, so the guard refused it. The target was the job's own venv, not a shared environment, so the refusal looks like a false match. Per jeff's instruction I did not retry with `"$FLEET_VENV"` written out. I also did not use another job's already provisioned venv, and I did not let `build_rocm.sh`'s `pip install` step run instead.
- The same guard refusal stopped the reviewer's install in "Review 2026-11-13 (fix 10697)".

### To unblock (jeff's decision)
Either allow this job to run the install written as `"$FLEET_VENV"/bin/python -m pip install -r dev-requirements.txt fsspec pytest-timeout pytest-repeat opencv-python-headless==4.10.0.84`, the form every earlier recipe in this file uses, or make the guard accept the job venv however its path is written. Then re-dispatch. The scripts from the c741328 gfx1100 job (`rx7900-20261111-194315-oneflow-reval-c741328-1100/moat/agent_space`: `env.sh`, `build.sh`, `suite.sh`, `cap.sh`, `reg.sh`, `run1.sh`, `cmp.py`, `t/{imp,cg_grad,longk}.py`) and its junit (`j/new_{graph,graph_cg,tensor,modules,cap,reg}`) still exist on this host, so the per-id comparison against c741328b7 is still possible. The last two-GPU one_embedding/P2P and fused-harness record on this platform is at d9e41753e ("Validation 2026-11-01 (linux-gfx1100, two W7800 ...)"), since the 8d470dcc1 and c741328b7 rounds had one GPU.

## Validation 2026-11-13 (linux-gfx1100) at 44d5f04cd (validator, two W7800, `HIP_VISIBLE_DEVICES=2,3` pinned; jeff's request: validate the fix round for Oneflow-Inc/oneflow#10697, NO LD_PRELOAD of libLLVM)

Host: two Radeon Pro W7800 48GB (gfx1100, wave32), ROCm 10.1.0 SDK wheels, Python 3.11.17, numpy 1.26.4, protobuf 3.20.3, torch 2.16.0a0+git6c3af02 symlinked read-only into `$FLEET_VENV-base-links`. venv: `"$FLEET_VENV"/bin/python -m pip install -q -r dev-requirements.txt fsspec pytest-timeout pytest-repeat opencv-python-headless==4.10.0.84 portalocker protobuf==3.20.3 numpy==1.26.4` (the guard allowed this spelling). `ulimit -c 0` and `LD_PRELOAD` unset on every run, `OMP_NUM_THREADS=4`, `ONEFLOW_TEST_DEVICE_NUM=1` (one GPU, `HIP_VISIBLE_DEVICES` 2 or 3) or `=2` (two ranks). Fresh fork clone, `protect-fork` installed. **Head: `HEAD` = `git ls-remote origin moat-fix-10697` = 44d5f04cdda699579b9f0825301b92317499f138, `moat-port` 14801d8a8 unchanged, `git diff --stat 14801d8a8 HEAD` = the two test files; `git status --porcelain --untracked-files=no` empty before and after.** Scripts (gitignored `agent_space/`): the c741328 gfx1100 job's `env.sh build.sh run1.sh suite.sh cap.sh reg.sh cmp.py t/{imp,longk,cg_grad}.py`, plus `emb.sh fh10.sh loop.sh ctl.sh t/{fused_harness,fhcmp}.py`.

### Verdict: **completed**. Every check passes or matches this platform's c741328b7 record, except two modules ids, both random-draw autotests explained below. `expensive/test_id_shuffle.py` **200 of 200** and `test_dynamic_allocation_gradient_shuffle.py` **50 of 50**: 0 GatherNd faults, 0 fp16 gradient-check failures, as predicted. But a pre-fix control on this host was also 200 of 200 clean, so gfx1100 cannot show the fault going away. It shows the round breaks nothing and that the new bound does not fail here. The removal itself rests on gfx942 (0/640 against 3/100).

### Build and import
`build_rocm.sh all` (`CMAKE_HIP_ARCHITECTURES=gfx1100`, `JOBS=32`, cold ccache, pywrap drops `--user`): **rc 0, 1257 s**, `OF_GPU_WARP_SIZE: 32`; `readelf --dyn-syms`: libLLVMSupport 3002 / libLLVMDemangle 455 `@@ONEFLOW_LLVM`, liboneflow.so 4, as everywhere. `import oneflow` without preload, both orders: rc 0, `1.0.0+cu00.git.44d5f04cd`, softmax sums to 64.0, fp32 mm 8.2e-6 / 6.3e-6 (torch identical), round trip bitwise; no `realloc(): invalid pointer`.

### Suites (one pytest per file, `--timeout=900`; per id against the c741328b7 gfx1100 junit still on this host, `rx7900-20261111-194315-oneflow-reval-c741328-1100/moat/agent_space/j/new_*`)
| suite | 44d5f04cd | per id vs c741328b7 | per-file rc |
|---|---|---|---|
| graph (GPU 2) | 144 s, 152 / 2 / 3 / 102 | **259 of 259** | identical |
| graph, `ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1` | 142 s, the same | **259 of 259** | identical, slowest file 25 s |
| tensor | 81 s, 368 / 1 / 0 / 22 | **391 of 391** | identical |
| modules (GPU 3, 6 at a time) | 942 s, 1614 / 53 / 5 / 316 | **1986 of 1988** | identical |
- `test_special_ops::test_flow_digamma_with_random_data` pass -> fail: `op_kernel_not_found_error`, digamma, `kCPU`, `kFloat16`, the CPU float16 draw recorded for its sibling psi at c741328b7. Reruns: 1 of 3 failed the same way.
- `test_higher_derivative_loss::test_smooth_l1_loss_grad_grad` fail -> pass: the randomized autotest recorded as flipping since fde793b9f. The three reruns aborted earlier, in torch's own `binary_cross_entropy_out_cuda` (0x1016, the BCE oracle trap from "Validation 2026-11-11 (linux-gfx90a) at c741328b7"). So they add no evidence on smooth_l1.
- Every other failure has the recorded id and class. Aborts as before: test_dataset and test_loss (134), the three `/dataset` graph files.

### Capture, regression, long-k, cold capture (GPU 2)
- Capture (fused_mlp_cuda_graph, cuda_graph_stream_creation, cuda_graph_fft_plan), `ROCBLAS_USE_HIPBLASLT` unset and `=1`: **30 of 30 pass**. No `(906)`, `(901)` or `PARSE_ERROR` in any log.
- Regression (c741328b7 list + gfx942's six added files): 96 / 1 / 5 skip; the 54 shared ids are the same; the one failure is `test_cublas_fused_mlp`, the known float32 tolerance item.
- `t/longk.py`: **PASS**; every fp16 value identical to c741328b7 (same seed), e.g. 1 x 2^20 x 64 oneflow 3.75e-4 vs torch 3.9e-3.
- `t/cg_grad.py`, k = 64 / 1024 / 2^20, hipBLASLt unset and `=1`: **16 of 16 bitwise in all 6**.

### Two GPUs (`emb.sh`, 2 rounds; `oneflow.distributed.launch --nproc_per_node 2`, random `--master_port`; results read from both ranks)
**35 of 35 launches rc 0.** test_id_shuffle_global with embedding + gradient P2P 0 and 1: 3 passed. test_oneembedding (2 ranks), test_id_shuffle_p2p_global, test_one_embedding_p2p_shuffle_global: 1 passed. test_comm_ops -k bfloat16: 2. test_graph_nccl_logical_bfloat16: 3. dynamic_allocation_..._shuffle_global: 1 skipped (upstream). One-GPU files: all pass, adam/ftrl skipped (upstream). That is the d9e41753e two-GPU record. Not run: id_shuffle P2P with test_id_shuffle_global, upstream's IPC key collision (gfx942).

### Fused harness (`t/fused_harness.py`, rebuilt from this file's description; my first version's loss and pass detection were wrong and were fixed before any counted run)
**`fh10.sh`: 10 of 10 rc 0 with FUSE=1 and embedding, gradient and id_shuffle P2P on, 32 s each, no hang (150 s limit)**; plus 2 all-P2P runs from `emb.sh`. Every fused run has 2 `sparse_feature` inputs and `skip_last_gather` true. Against unfused: identical losses at all 50 steps in 17 runs, top-weight relL2 6.0e-5 to 8.8e-5, the unfused rerun's own noise (7.4e-5).

### The two changed files (one fresh `pytest -v` process per run, one file per process; failing logs kept)
| file | GPU 2 | GPU 3 | GatherNd faults | other failures |
|---|---|---|---|---|
| expensive/test_id_shuffle.py at 44d5f04cd | 100 / 100 | 100 / 100 | **0** | **0** |
| expensive/test_dynamic_allocation_gradient_shuffle.py at 44d5f04cd | 25 / 25 | 25 / 25 | **0** | **0** |
| control: 14801d8a8's test_id_shuffle.py, a copy outside the fork | 100 / 100 | 100 / 100 | 0 | 0 |
No log is kept: all 450 runs rc 0, no `FAILED`, `AssertionError` or `MEMORY_FAULT`. The fault has never been seen on gfx1100. Stale-tail rates differ by arch (gfx942 7/300, gfx90a at most 1/844), and this one is at the low end. The old fp16 check's expected 1.3 failures in 200 control runs did not show (Poisson P(0) = 0.26).

### Other
- CUDA gate: Python-only round; "CUDA compile check at c741328b7" (same tree as 14801d8a8) stands.
- `jargon.py --port oneflow`: clean (needs `master` and a local `moat-fix-10697` branch in the clone). Docs: README/`build_rocm.sh` reproduced the build; no LD_PRELOAD instruction. Nothing pushed to the fork, no gh writes, no core file in the job directory.
- Records: notes.md was 1004 KB with the first draft of this section, and check.py's `blobs` gate (1000 KB) refused the push, so the section was shortened. The file is close to the cap. The next long section will hit it, and splitting or archiving old sections is jeff's call. While diagnosing, the fleet guard refused a plain `git push origin port/oneflow 2>&1`, reading `2>&1` as the branch. I did not retry it in another form.
## Validation 2026-11-13 (linux-gfx90a) at 44d5f04cd

MI250X, pin `HIP_VISIBLE_DEVICES=2,3` kept, `ulimit -c 0`, no `LD_PRELOAD`. Fresh clone, head = `ls-remote moat-fix-10697` = 44d5f04cd; tracked tree clean throughout; nothing pushed to the fork. The full record (tables, commands, scripts) is in this section as committed in 814604de; it was shortened here for the notes size gate.

**Verdict: validation-failed, for one unexplained hang.** The fused harness with every P2P switch on (`FUSE_EMBEDDING_INTERACTION`, embedding, gradient and id_shuffle P2P) hung once in 73 runs (fh10 run 9: both ranks stopped after step 0, killed at 240 s, no error in either log, no stacks taken). The other 72 completed runs (60 of them with gdb hang capture) and every other configuration were bitwise equal to unfused. The round changes only two Python files the harness never loads, so the code that hung is that of 14801d8a8. Earlier records in this configuration: gfx90a d9e41753e 0/36, gfx942 14801d8a8 0/37. Whether it can still count as non-port is jeff's call. Next step: a few hundred `fhh.sh` runs to catch it with stacks.

- Changed files, one fresh pytest process per run: test_id_shuffle.py **200/200 pass** (100 per GPU) and test_dynamic_allocation_gradient_shuffle.py **50/50**. **0 GatherNd or memory faults, 0 other failures**, as the fix predicts.
- Build: `build_rocm.sh all`, clean, rc 0, 672 s, warp 64. `import oneflow` works in both orders without preload.
- Per id vs the c741328b7 junit: graph 259/259, graph with CUDA graphs 259/259 and tensor 391/391 the same. Modules: 1981/1983 the same; the two differences are smooth_l1 grad_grad and digamma random-data, which flip on reruns here (1/3 and 2/3 fail), as recorded before.
- Capture tests: 3 tests x 5 runs, hipBLASLt unset and =1, 30/30 pass, plus 3/3 with an outer CUDA graph. cg_grad: 6/6 bitwise. Long-k: PASS, equal to torch (fp16 numbers identical to c741328b7). Regression tests: all pass, except the known test_cublas_fused_mlp float32 item.
- Two GPUs, 3 rounds: test_id_shuffle_global P2P 0/1, test_id_shuffle_p2p_global, test_one_embedding_p2p_shuffle_global, comm_ops bf16, nccl_logical bf16 and 2-rank test_oneembedding all pass on both ranks (the dynamic-allocation global test skips upstream). The one-GPU one_embedding files all pass.
- jargon clean; docs unchanged; CUDA gate not owed (Python-only round).

## Validation 2026-11-13 (linux-gfx942) at 44d5f04cd: completed

Validator job on an MI300X host (gfx942, wave64), `HIP_VISIBLE_DEVICES=2,3` pinned, `ulimit -c 0`, no LD_PRELOAD on any run. HEAD and `origin/moat-fix-10697` were 44d5f04cd, `moat-port` 14801d8a8, tracked tree clean before and after; nothing pushed to the fork, no GitHub writes. The worker could not write this section: the fleet guard refused its notes heredoc as a GitHub write, a false match on the prose (the guard's `gh <group> <verb>` pattern matched words inside the heredoc text). It stopped as instructed. jeff ruled on 2026-11-14 that the result be recorded from the console host from the worker's report; this section is that record.

- Build: clean, rc 0, 708 s, gfx942 warp 64, libLLVM symbol counts as at 14801d8a8. `import oneflow` works in both orders.
- Graph suite: 152 pass / 2 fail / 3 error / 102 skip, identical per test id to 14801d8a8, with and without `ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1`. Tensor: 368 pass / 1 fail (`test_safetensors`, module not installed), identical per id.
- Modules: 1614 pass / 53 fail / 5 error / 316 skip; 1980 of 1983 shared ids match. Differences: three known random flakes passed (tf32 matmul, `psi`, global `smooth_l1` grad_grad); `test_higher_derivative_loss` missed torch's BCE device trap this run, so its five ids were collected, three failing in the classes recorded for their global twins.
- Capture tests 5/5 each with and without `ROCBLAS_USE_HIPBLASLT=1`; fused_mlp under outer CUDA graphs 3/3. Long-k passes, `test_matmul_long_k` 4/4. Regression tests pass except the known `test_cublas_fused_mlp` float32 tolerance item.
- Two GPUs, 2 rounds: every test matches the record; `test_id_shuffle_global` with id_shuffle P2P on still aborts rc 1 on the recorded upstream IPC key collision (`kv_.emplace` CHECK, `IdShuffleP2PKernel` -> `RpcClient::PushKV`). Fused harness 10/10 with P2P on, losses equal to unfused at all 50 steps; embedding rounds 8/8.
- Changed files, one fresh process per run: `test_id_shuffle.py` 200/200 (100 per GPU), 0 GatherNd faults, 0 other failures (the unfixed file's rate predicts about 5; P(0) about 1%); `test_dynamic_allocation_gradient_shuffle.py` 50/50.
- `jargon.py --port oneflow` clean. No C++/HIP/CMake change, so no nvcc check owed.

## Ruling 2026-11-14 (jeffdaily): linux-gfx90a completed at 44d5f04cd

The one fused-harness hang in 73 runs (all P2P on, run 9, no error in either log, 60 clean reruns with stack capture) is not to be investigated: "I don't care about this hang. oneflow is not maintained. Let's just be done with it." The round changes only two Python test files the harness never loads, so the compiled code is the 14801d8a8 the open PR shows. Recorded completed on that ruling; no deferral.
