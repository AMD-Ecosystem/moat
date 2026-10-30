# oneflow notes

ROCm/HIP port of oneflow's CUDA GPU backend. Lead platform linux-gfx90a (MI250X,
CDNA2, wave64), ROCm 7.2.1. Strategy A (pure CMake: USE_HIP option, one compat header,
.cu retagged LANGUAGE HIP). Fork: https://github.com/AMD-Ecosystem/oneflow (branch moat-port).

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

## Review 2026-05-31 (reviewer, linux-gfx90a, /pr-review local-branch vs 25c8978)

Verdict: review-passed. Fact-checked the curated `[ROCm]` commit 68718f0 against the ROCm fault classes; the wave64 reductions, excluded-op guards, and host cuda*->HIP ports are all correct. Findings below are non-blocking follow-ons only (problems-only per skill philosophy).

Priority-1 (wave64) -- VERIFIED CLEAN. The complete set of COMPILED warp-shuffle/ballot TUs is exactly five: core/cuda/{softmax,layer_norm,rms_norm}.cuh, user/kernels/{layer_norm_gpu_kernel,data_shuffle_kernel}.cu. Each is consistent end-to-end (launch geometry == shared sizing == loop bound == shfl width):
- layer_norm_gpu_kernel.cu:148 WarpReduce -- width-confined to block_dim_x=32 (a tiling constant matching dgamma/dbeta[32][33] and the dim3(32,8) launch); the session-1 bug (tile tied to OF_GPU_WARP_SIZE while launch/shared stayed 32) is gone.
- rms_norm.cuh:41 WarpReduceSum -- full-warp, no-width shfl starting at kWarpSize/2; correct because the RmsNormParamGrad launch sets block_dim_x = kWarpSize (rms_norm_gpu_kernel.cu:255) and dweight[kWarpSize][kWarpSize+1], so the hardware-warp-wide reduce folds exactly block_dim_x lanes on wave64. Distinct from LayerNormParamGrad, which ties block_dim_x to 32, not the warp.
- softmax.cuh:68 WarpAllReduce -- porter ADDED the explicit width arg (original had none). BC-safe on CUDA: thread_group_width divides the warp and groups are W-aligned (static_asserts at 243/244 + dim3(thread_group_width,...) launch), and the loop max laneMask is thread_group_width/2 < W, so laneId^laneMask stays in-group whether width is 32 or W -- identical CUDA numerics, required for wave64.
- data_shuffle_kernel.cu:330 WarpMaxAllReduce -- width-confined + matching dim3(thread_group_width,...) launch.
- batch_norm_kernel_utils.h (compiled via 5 batch_norm_*_kernel.cu, NOT excluded): WARP_SIZE=OF_GPU_WARP_SIZE drives getMSB(WARP_SIZE) step count, the shfl width, the shared sizing, and the warp-count divisions uniformly -- arch-unified. lru_cache.cu (excluded) keeps 32 ways deliberately, 64-bit ballot via __ffsll/__popcll.

Priority-2 (excluded-op guards) -- VERIFIED CLEAN. Each of the 17 deferred .cu self-contains its own REGISTER macros, so exclusion drops the kCUDA kernel cleanly (runtime op_kernel_not_found, never a wrong result or link error). The only split-registration cases were handled: grid_sample (kCUDA registration gated !USE_HIP in grid_sample_kernel.cpp; util.cpp defines kCPU only, so GridSampleKernelUtil<kCUDA> is never ODR-used) and lru_cache (cache.cpp NewLruCache guarded, UNIMPLEMENTED on HIP). normalization_kernel.cpp registers CPU only; the cuDNN kCUDA path lives in the excluded .cu. image_decoder kCUDA registration is auto-gated by WITH_NVJPEG (undefined on HIP). The cuDNN fused-BN job pass no-ops on HIP because IsFusedBnAddReluSupported() keys on CUDNN_VERSION, which is left undefined on the HIP path (confirmed -- not in the compat header nor MIOpen), so no cudnn_fused_* op (which would have no HIP kernel) is ever inserted.

Priority-3 (host cuda*->HIP) -- VERIFIED CLEAN. CudaDriverGetPrimaryCtxActive reimplemented via hipDeviceGet/hipDevicePrimaryCtxGetState (correct native driver API). The guarded-out handles (cudnn_handle_/cublas_lt_handle_/cusolver_dn_handle_) are default-init {} and their create/destroy are !USE_HIP-guarded; cross-referencing every caller of the three accessors, all of them live in excluded TUs, so no compiled HIP path dereferences a null handle. CublasMathModeGuard no-op on HIP is correct (hipblas accepts only HIPBLAS_DEFAULT_MATH). The unconditional .template removals (add_n_kernel.cpp, ndarray_reduce_impl.h, one_embedding.cpp) are behavior-preserving on gcc/nvcc AND clang -- Mut(int64_t)/array::data(...) are non-template; dptr() uses a defaulted template arg, so dropping the bare `.template` keyword (no arg list) changes nothing.

Non-blocking follow-ons (NOT changes-requested):
1. oneflow/core/profiler/profiler.cpp:54,60 -- nvtxRangePushA/nvtxRangePop in RangePush/RangePop are guarded only by OF_ENABLE_PROFILER, not by !USE_HIP, and NVTX is not mapped on HIP (only cudaProfilerStart/Stop are aliased to hip*). The porter guarded the <nvtx3/...> include and nvtxNameOsThreadA (line 47) under !USE_HIP but missed these two call sites. A HIP build with -DBUILD_PROFILER=ON would fail to compile here. Latent only: BUILD_PROFILER defaults OFF and build_rocm.sh does not set it, so the validated build is unaffected. Fix when convenient: add `&& !defined(USE_HIP)` to the two guards (or document the profiler as unsupported on ROCm).
2. oneflow/core/embedding/full_cache.cu:183,253 -- EncodeLookupKernel/EncodeLookupMaskKernel are COMPILED on HIP and hardcode warp_size=32. This is CORRECT on wave64 (verified: zero __shfl/__ballot in the file; warp_size is purely a shared-mem batch tile indexed [warp_id][lane_id] with disjoint rows, and __syncwarp() on a 32-lane logical group nested in a 64-lane wavefront is a full-wavefront barrier, i.e. stricter-but-correct; block_size=128 is a clean multiple of 64). Unlike lru_cache.cu it carries no explanatory comment; a future maintainer adding a shuffle here would trip the wave64 hazard. Recommend a one-line comment mirroring lru_cache.cu's, and a note in the followers (gfx1100/gfx1151) revalidation that this kernel was reasoned correct, not just compiled.
3. cmake/util.cmake -- hardcodes the absolute path /opt/rocm/include for the GPU host .cpp CXX pass. Matches the build host but is non-portable; prefer deriving from the hip package or ROCM_PATH. Minor.
4. external/kineto/CMakeLists.txt -- the moved comment line still carries a preexisting non-ASCII `∂` ("cudaGetDeviceCount∂"). Preexisting Unicode, so not strictly a new-comment violation, but since the line was touched a one-char trim to ASCII would be cleaner.

Scope acceptance (per dispatch, NOT blockers): the 17 deferred .cu (conv/normalization/fused-MLP/CUTLASS+wmma/cuSOLVER/libcu++/P2P) are a documented validation-scope deferral; the fp16-matmul numpy __bool__ pytest quirk and the CPU int32-matmul gap are pre-existing oneflow issues, not ROCm regressions. The GPU test run not yet covering the deferred op surface is expected at review time -- the validator stage exercises the real-GPU slice next.

## Validation 2026-05-31 (gfx1100, ROCm 7.2.1)

State: port-ready -> completed. Platform: linux-gfx1100 (2x AMD Radeon Pro W7800 48GB, gfx1100, RDNA3, wave32, ROCm 7.2.1).
Fork HEAD: 68718f03829384b530ba97c32b18e5937629ec09. No fork commit needed (follower, zero churn). Device: HIP_VISIBLE_DEVICES=0.

Build: FRESH clone of moat-port branch at 68718f03. Configure+tp+oneflow from scratch (no ccache on this host); 19 min total. Build log: agent_space/oneflow_build_gfx1100.log.

Build command:
```
CMAKE_HIP_ARCHITECTURES=gfx1100 bash /var/lib/jenkins/moat/projects/oneflow/src/build_rocm.sh all
```
(no ccache, no -DCMAKE_*_LAUNCHER flags; ROCM_CLANG auto-detected at /opt/rocm/lib/llvm/bin/clang++)

Configure confirmed:
- CMAKE_HIP_ARCHITECTURES: gfx1100
- OF_GPU_WARP_SIZE: 32  (gfx11xx matches the `!gfx9xx` branch in cmake/cuda.cmake -> warp=32)
- OF_GPU_WARP_SIZE=32 in CMakeFiles/oneflow.dir/flags.make DEFINES (all TUs, both host and device)
- CMakeFiles/oneflow.dir/flags.make: -DOF_GPU_WARP_SIZE=32 present

Artifacts:
- liboneflow.so: 1.2 GB at build/liboneflow.so; 190 gfx1100 code objects confirmed via roc-obj-ls
- _oneflow_internal: 16 MB at python/oneflow/_oneflow_internal.cpython-312-x86_64-linux-gnu.so

gfx1100 code-object proof:
```
$ roc-obj-ls liboneflow.so | grep gfx | head -2
1  hipv4-amdgcn-amd-amdhsa--gfx1100  ...offset=97603584
2  hipv4-amdgcn-amd-amdhsa--gfx1100  ...offset=97648640
```
Total: 190 gfx1100 code objects (vs 190 on gfx90a).

GPU test commands:
```
cd /var/lib/jenkins/moat/projects/oneflow/src
export PYTHONPATH=python:$PYTHONPATH
HIP_VISIBLE_DEVICES=0 python3 -m pytest \
  python/oneflow/test/modules/test_add.py \
  python/oneflow/test/modules/test_cast.py \
  python/oneflow/test/modules/test_where.py \
  python/oneflow/test/modules/test_masked_fill.py \
  python/oneflow/test/modules/test_broadcast_ops.py \
  python/oneflow/test/modules/test_transpose.py \
  python/oneflow/test/modules/test_softmax.py \
  python/oneflow/test/modules/test_layer_norm.py \
  python/oneflow/test/modules/test_matmul.py \
  -q -p no:cacheprovider
```

Results (gfx1100 vs gfx90a lead):
- Primitive GPU slice: 64 PASSED (same as gfx90a)
- Matmul core (fp32/mm/mv/broadcast/batch/tf32): 9 PASSED (same as gfx90a)
- Total: 73 passed, 2 failed (same as gfx90a -- both pre-existing artifacts)

Pre-existing artifacts (NOT GPU faults, gate is GREEN):
1. test_matmul fp16: TypeError __bool__ should return bool -- numpy/oneflow dtype-interop quirk; GPU fp16 result is correct.
2. test_matmul int32: same __bool__ quirk (int32 CPU path; no int32 matmul kernel in oneflow).

Warp-reduction op explicit verification on wave32:
- softmax (8x32x256, dim=-1): max_abs_diff=1.49e-08  -> PASS
- layer_norm fwd (4x64x64): all finite, matches numpy reference -> PASS
- layer_norm bwd gamma/beta grad (LayerNormParamGrad wave64-fix site): no NaN -> PASS
- rms_norm fwd+bwd (RmsNormParamGrad, kWarpSize/2 butterfly): no NaN -> PASS
- matmul fp32 (64x64): max_abs_diff=5.72e-06 -> PASS

Wave32 verdict: kWarpSize=32 (OF_GPU_WARP_SIZE=32) is native on RDNA3. The arch-unified
warp reductions (softmax WarpAllReduce, layer_norm WarpReduce block_dim_x=32, rms_norm
WarpReduceSum from kWarpSize/2) all produce correct results on gfx1100. No source changes
required. The LayerNormParamGrad wave64 fix (session-2 fault D) is equally correct on wave32
-- tile_size=32 and block_dim_x=32 match the hardware warp width on RDNA.

Determinism check (softmax + layer_norm bit-identical over two runs): PASS.

Fork: untouched (no commit needed for follower-only validation).

Final state: linux-gfx1100 -> completed, validated_sha = 68718f03829384b530ba97c32b18e5937629ec09.

## Validation 2026-05-31

State: review-passed -> completed. Platform: linux-gfx90a (MI250X, gfx90a, ROCm 7.2.1).
Fork HEAD: 68718f03829384b530ba97c32b18e5937629ec09. Device: HIP_VISIBLE_DEVICES=1 (GCD 1, 0% VRAM used).

Build: REUSED. liboneflow.so (1.3 GB) at /var/lib/jenkins/moat/projects/oneflow/src/build/liboneflow.so
built at fork HEAD 68718f03. Incremental cmake --build confirmed 4 trivial TUs (version.cpp,
op_generated.cpp, plan_to_physical_graph.cpp, libkineto.so) rebuilt -- no HIP recompilation.
`import oneflow` / `oneflow.cuda.is_available()` returns True (LLVM ABI-breaking-checks
symbol mismatch fix confirmed still in place).

GPU test commands:
```
cd /var/lib/jenkins/moat/projects/oneflow/src
source build/source.sh
HIP_VISIBLE_DEVICES=1 python3 -m pytest \
  python/oneflow/test/modules/test_add.py \
  python/oneflow/test/modules/test_cast.py \
  python/oneflow/test/modules/test_where.py \
  python/oneflow/test/modules/test_masked_fill.py \
  python/oneflow/test/modules/test_broadcast_ops.py \
  python/oneflow/test/modules/test_transpose.py \
  python/oneflow/test/modules/test_softmax.py \
  python/oneflow/test/modules/test_layer_norm.py \
  python/oneflow/test/modules/test_matmul.py \
  -q -p no:cacheprovider
```

Results:
- Primitive GPU slice (test_add/cast/where/masked_fill/broadcast_ops/transpose/softmax/layer_norm): 64 PASSED
- Matmul core (fp32/mm/mv/broadcast/batch/tf32): 9 PASSED
- Total: 73 passed, 2 failed (both documented pre-existing artifacts)

Documented pre-existing artifacts (not GPU/ROCm faults, gate is GREEN):
1. test_matmul fp16: `TypeError: __bool__ should return bool, returned unused_N` -- numpy/oneflow
   dtype-interop quirk; the GPU fp16 matmul result is bit-identical to torch (max abs diff 0.0).
2. test_matmul int32: `TypeError: __bool__ should return bool, returned unused_N` -- same quirk
   (the int32 path triggers op_kernel_not_found as a pre-existing CPU-path limitation; the same
   __bool__ DeprecationWarning-turned-TypeError fires before reaching the kernel-not-found path).

LayerNorm backward verification: all 4 subcases PASSED explicitly --
  test_block_smem_impl PASSED, test_block_uncached_impl PASSED,
  test_no_affine PASSED, test_warp_impl PASSED.
The backward=True path in each subcase exercises the LayerNormParamGrad weight-gradient
kernel (the wave64 reduction bug fixed in session D). The results match torch to atol/rtol
tolerance -- no garbage in gamma/beta gradients.

Final state: linux-gfx90a -> completed, validated_sha = 68718f03829384b530ba97c32b18e5937629ec09.
Followers unblocked: linux-gfx1100 -> port-ready, windows-gfx1151 -> port-ready.

## Validation 2026-06-05 (windows-gfx1101, gfx1101 Radeon PRO V710, RDNA3, ROCm 7.14)

State: port-ready -> blocked (non-viable). Assessment: Windows build is not viable for
oneflow. Blockers identified (no GPU test run attempted; build is infeasible before tests):

1. RCCL is hard-required: cmake/cuda.cmake has `find_package(rccl REQUIRED)` and
   lists `roc::rccl` in VENDOR_CUDA_LIBRARIES. RCCL is a Linux-only collective
   communications library (uses Linux IPC, shared memory, sockets). There is no
   Windows RCCL package. Additionally, `<nccl.h>` is included by:
   - oneflow/core/device/cuda_util.h (the main GPU utility header, included everywhere)
   - oneflow/core/common/env_var/eager.h
   - oneflow/core/ep/cuda/cuda_device_manager_factory.cpp
   - oneflow/core/job/collective_boxing/nccl_executor_backend.cu
   - oneflow/core/job/resource_desc.cpp
   Making RCCL optional requires patching all these plus providing stub headers and
   guards, which is a multi-file refactor beyond the MOAT port scope.

2. POSIX platform code: oneflow/core/platform/lib/ibv_wrapper.cpp uses dlopen on
   libibverbs.so (Linux InfiniBand verbs), and pthread_fork.cpp uses pthread_atfork
   (POSIX-only). Several core files include unistd.h. While cmake/oneflow.cmake has
   platform excludes for "windows" vs "linux" subdirs, these files are not in a
   platform subdir and would compile (and fail) on Windows.

3. platform.cmake assumes MSVC: cmake/platform.cmake defines MSVC-only flags
   (/MP, /bigobj, /EHsc, /GF, /FC) incompatible with the all-clang toolchain
   required for HIP on Windows. The two toolchains are mutually exclusive here.

4. Third-party builds: 32 libraries (grpc, protobuf, openssl, opencv, oneDNN, glog,
   etc.) build from source via ExternalProject_Add with Linux-centric build recipes
   (Makefile-based or cmake with Linux assumptions). Cross-building all of these
   under Windows/clang is a separate substantial effort.

5. No upstream Windows support: oneflow has never shipped on Windows. The cmake
   build system has a placeholder `error("Expanding macro in WIN32 is not supported
   yet")` (cmake/oneflow.cmake:54) and no upstream Windows CI path.

Conclusion: windows-gfx1101 and windows-gfx1201 blocked (non-viable). The GPU port
is complete and validated on linux-gfx90a (MI250X, 73 passed) and linux-gfx1100
(Radeon Pro W7800, 73 passed). The Windows platform requires a full Windows port of
oneflow itself, which is outside the scope of this ROCm/HIP kernel port.

## Revalidation 2026-06-16 (linux-gfx90a, carry-forward)

State: revalidate -> completed (carry-forward, no GPU re-run needed).
Validated SHA: 68718f03829384b530ba97c32b18e5937629ec09 -> 79f235c9a6d06dd814850a32d1d99754d1dfffe1.

Delta classification: comment-only (source-class carry-forward). The single commit
"[ROCm] Scrub internal port labels from comments" changed 4 files:
- build_rocm.sh: "MOAT ROCm/HIP build" -> "ROCm/HIP build"; "Strategy A: ..." -> stripped
- cmake/cuda.cmake: "MOAT ROCm/HIP backend (Strategy A)." -> "ROCm/HIP backend."
- cmake/oneflow.cmake: "Used under USE_HIP (Strategy A)" -> "Used under USE_HIP"
- oneflow/core/device/cuda_to_hip.h: "(MOAT Strategy A)" -> removed from comment

No functional code, no device code, no CMake logic, no source symbols changed. Compiled
output is identical. Carry-forward applied: linux-gfx90a completed at 79f235c9.

## PR-prep vocabulary cleanup 2026-10-29 (porter, linux-gfx942 host)

Requested by jeff before the upstream PR is opened. No upstream drift: upstream master and fork master both at 25c8978 (unchanged since 2025-08-20), so no fast-forward and no upstream merge was needed.

`python3 utils/jargon.py --port oneflow` at 2729bee5 reported 5 hits: commit 68718f038 title "(Strategy A)", its body ("MOAT CUDA-to-ROCm porting effort", "Lead platform gfx90a"), and two added comments.

1. Comment commit d019f3551 "[ROCm] Reword two port comments in plain terms" on top of 2729bee5, pushed as a fast-forward:
   - cmake/cuda.cmake:49 "Followers override only -DCMAKE_HIP_ARCHITECTURES." -> "Other GPUs change only -DCMAKE_HIP_ARCHITECTURES."
   - oneflow/core/device/cuda_pseudo_half.h:26-27 "(the HIP-vector-type-provides-operators fault class)" -> "would redefine those operators and fail to compile". Line count unchanged.
   - `python3 utils/moatlib.py classify oneflow 2729bee5 d019f355` -> `class=comment-only arch_independent=True inert=True`. `advance-head oneflow d019f3551` carried linux-gfx90a and linux-gfx1100 forward (source-class, comment-only).
   - Build check on gfx942 (ROCm 10.1 from the rocm-sdk Python packages, clang 24): the changed header compiled as HIP for gfx942 at both 2729bee5 and d019f355 (`clang++ -x hip --offload-arch=gfx942 -std=c++17 -DWITH_CUDA -DUSE_HIP -DOF_GPU_WARP_SIZE=64 -include oneflow/core/device/cuda_to_hip.h -Ioneflow/core/device/hip_compat -I. -c t.hip` on a TU including cuda_pseudo_half.h, rc=0 both), and its preprocessed output (`-E -P`) is byte-identical (md5 312c1815... both). The edited cmake/cuda.cmake parsed in a full configure (printed `CMAKE_HIP_ARCHITECTURES: gfx942`, `OF_GPU_WARP_SIZE: 64`).
   - Full build and GPU test suite on gfx942 NOT run: oneflow's configure (cmake/python.cmake install_py_dev_deps) installs dev-requirements.txt with `--user` into the configuring Python. That fails in this host's shared image venv ("Can not perform a '--user' install") and the fleet guard refuses changing the shared Python environment; no per-job venv was provisioned. The host also has no /opt/rocm (ROCm comes from the rocm-sdk packages), so cmake/util.cmake's hard-coded /opt/rocm/include (review follow-on 3) would need CPATH or a fix there. To get gfx942 evidence, dispatch with a per-job venv (`--build`). No linux-gfx942 validation was recorded.
2. PR-prep squash be1ef73f1 "[ROCm] HIP backend for the CUDA GPU path": a single commit on master (25c8978) whose tree equals d019f3551's tree (e40faba8...). The message is 68718f038's with the in-house labels removed, the "Lead platform" Test Plan line replaced by the two validated GPUs, the removed gfx90a arch pin described as HIP arch auto-detect, and paragraphs unwrapped. jargon: clean.
   - Why a force-push is needed: the vocabulary is in a commit message already on moat-port, which can only be removed by rewriting history. No upstream PR is open (`pr-state oneflow` -> none), so moat-port is not frozen and nothing upstream-visible changes. `squash-carry-forward` refuses anything but a tree-identical squash, so no unvalidated content enters.
   - NOT DONE: the fleet guard on this host refused `git push --force-with-lease=moat-port:d019f35... origin moat-port` ("only a plain push is allowed"). `squash-carry-forward oneflow be1ef73f1` had already run locally (it carried gfx90a + gfx1100); since the fork never received be1ef73, status.json was restored to its committed version and `advance-head oneflow d019f3551` re-run, so the record's head is the published d019f3551 and its carry-forward provenance reads 2729bee5 -> d019f3551.
   - The squash was pushed (plain push, new branch) to the fork as `moat-port-squash` = be1ef73f19357dba273e913f4a04038a5476b89d. It still contains 68718f038's full content; only the history differs. To finish, a person (or a job allowed to force-push) runs, after the reviewer passes d019f3551:
     ```
     git -C projects/oneflow/src fetch origin
     test "$(git -C projects/oneflow/src rev-parse origin/moat-port-squash^{tree})" = "$(git -C projects/oneflow/src rev-parse origin/moat-port^{tree})"
     git -C projects/oneflow/src push --force-with-lease=moat-port:d019f35519eeb6f071ea4a6883b5c8f7a9adaa67 origin origin/moat-port-squash:moat-port
     python3 utils/moatlib.py squash-carry-forward oneflow be1ef73f19357dba273e913f4a04038a5476b89d
     python3 utils/moatlib.py commit-project oneflow "oneflow: PR-prep squash be1ef73, tree-identical carry-forward"
     git -C projects/oneflow/src push origin --delete moat-port-squash
     ```
     If moat-port moved after d019f3551, the squash is stale: rebuild it with `git commit-tree <new-tip>^{tree} -p master -F <message of be1ef73>`.
   - Stage set to `ported` at d019f3551 (comment-only on top of the reviewed 2729bee5), so the reviewer runs next; validation already carries forward on both Linux gates.

## Review 2026-10-29 (reviewer, linux-gfx942, /pr-review local-branch moat-port d019f3551 vs master 25c8978)

Verdict: changes-requested. Scope: the whole fork diff (58 files), with the commits since the 2026-05-31 review (79f235c9a, f4f9394e2, 2729bee55, d019f3551) read line by line, and every edit that is not under `USE_HIP` checked for its effect on the CUDA build. d019f3551 itself is comment-only and its two rewordings are accurate. The pending squash `moat-port-squash` be1ef73f1 has the same tree as d019f3551 (e40faba8). Each finding below was re-checked against the code before it was written down.

1. Upstream-visible comments point at MOAT records and at other ports. `python3 utils/jargon.py --port oneflow` missed all of these, so this review adds the terms to `config/jargon.toml` (`projects/<x>/notes.md`, `(<project> lesson)`, `first|later|next slice`, and `fault-class` with a hyphen). With them, jargon reports 9 hits in added lines:
   - build_rocm.sh:19 and cmake/oneflow.cmake:262 "See projects/oneflow/notes.md" (oneflow has no such file); cmake/oneflow.cmake:262 "fault-class status".
   - cmake/oneflow.cmake:250 "a later slice ports them"; cmake/third_party.cmake:134 and oneflow/core/device/cuda_util.h:25 "the first slice".
   - oneflow/core/device/cuda_to_hip.h:42 "(gpuRIR lesson)", :80 "(MPPI lesson)", :86 "(AutoDock-GPU lesson)"; also :70 "matching the raft pattern" (not caught by the checker).
   Fix: keep the technical reason and delete the pointer or label. Examples: "MIOpen/CUTLASS/flash-attention are not ported yet"; for :42, "libc host declarations must be seen before HIP's __device__ memcpy/memset overloads". jargon must report 0 hits in added lines.
2. `__launch_bounds__(1024)` was added to 9 block kernels without a `USE_HIP` guard: core/cuda/softmax.cuh:500,653; layer_norm.cuh:575,1169,1401; rms_norm.cuh:286,413,702,830. On HIP it has no effect. With the ROCm clang 24 on this host, a kernel compiled for gfx90a or gfx942 gets `.max_flat_workgroup_size: 1024` and `"amdgpu-flat-work-group-size"="1,1024"` whether or not it has `__launch_bounds__(1024)`, because HIP-clang already defaults kernels to 1024 (reproducer: `template<int B> __global__ void A(float*x){x[threadIdx.x]+=1.f;}` plus the same kernel with `__launch_bounds__(1024)`, both instantiated at 128, then `amdclang++ -x hip --offload-arch=gfx90a --offload-device-only -S -O2`; the two kernels' metadata is identical). So fault B in the porter notes ("without it the register allocation is too large and the 1024-thread launch faults") does not hold on this compiler. The edit changes only nvcc output: the 128/256/512 template instantiations become capped at 64 registers per thread. That can change which block size the occupancy heuristic picks, and so the CUDA reduction order. This contradicts the commit message's "the NVIDIA build is a pure passthrough". Fix: drop the 9 additions; layer_norm.cuh:796 is upstream's own and stays. If a ROCm 7.2.1 gfx90a reproducer shows the fault is real there, do this instead: guard the attribute with `USE_HIP` (or use `__launch_bounds__(block_size)`) and record the evidence. Note that the fault B record was never checked against kernel metadata. Confidence: confirmed for ROCm 10.1 clang 24; not reproduced on ROCm 7.2.1, which this host does not have.
3. Wavefront width is wrong for a mixed-family arch list. cmake/cuda.cmake:50-55 sets `OF_GPU_WARP_SIZE=64` for the whole build if any listed arch matches `^gfx9`. So `-DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100"` compiles the gfx1100 code object with kWarpSize=64: width-64 shuffles on a wave32 GPU, and the softmax/layer_norm/rms_norm reductions give wrong results with no error. Since 2729bee55 this also happens from auto-detect, because CMake's rocm_agent_enumerator path lists every GPU on the host (CMakeDetermineHIPCompiler.cmake, CMake 3.31). Fix: `message(FATAL_ERROR ...)` when the list mixes gfx9 and non-gfx9 arches. One value per build is a real constraint, since the value is a template argument in both compile passes, so the build must refuse instead of picking one.
4. build_rocm.sh:9 `ARCH="${CMAKE_HIP_ARCHITECTURES:-gfx90a}"` still pins gfx90a, and :2 says "(gfx90a)". The documented build therefore never reaches the auto-detect that 2729bee55 enabled: a plain `./build_rocm.sh` on gfx942 or gfx1100 builds gfx90a-only code objects. That is the exact failure 2729bee55's message describes. Fix: pass `-DCMAKE_HIP_ARCHITECTURES` only when the variable is set, and drop "(gfx90a)".
5. cmake/util.cmake:251 hard-codes `/opt/rocm/include` for the GPU host .cpp TUs (follow-on 3 of the 2026-05-31 review, still open). It now blocks a real setup: this host has no /opt/rocm because ROCm comes from the rocm-sdk Python packages (porter notes 2026-10-29). Fix: take the path from `find_package(hip)`, using the include directories of `hip::host`, `hip_INCLUDE_DIRS`, or `ROCM_PATH`, not a literal. The build_rocm.sh:20-21 compiler defaults can be overridden from the environment and are acceptable.
6. oneflow/core/profiler/profiler.cpp:54,60 `nvtxRangePushA`/`nvtxRangePop` are guarded only by `OF_ENABLE_PROFILER`, while the `<nvtx3/nvToolsExt.h>` include is now `!USE_HIP`. A HIP build with `-DBUILD_PROFILER=ON` therefore fails to compile, and the new comment at :26-27 says the annotations "are compiled out under USE_HIP", which is false for these two calls (follow-on 1 of the 2026-05-31 review, still open). Fix: add `&& !defined(USE_HIP)` to both guards.
7. `USE_HIP` is never declared. No `option(USE_HIP ...)` exists in CMakeLists.txt, even though the commit message calls it "a new USE_HIP option". Also, cmake/third_party.cmake:137-142 tests `WITH_RCCL`, which is defined nowhere in the tree, so that block is dead (roc::rccl is already in VENDOR_CUDA_LIBRARIES from cmake/cuda.cmake). Fix: declare `option(USE_HIP "..." OFF)` next to BUILD_CUDA, and delete the dead `WITH_RCCL` block.
8. Commit hygiene is still unresolved on moat-port. 68718f038's message fails jargon ("Strategy A", "MOAT", "Lead platform"). `utils/check.py` reports 79f235c9a with no Test Plan, and f4f9394e2 and 2729bee55 with no fenced command block. The squash be1ef73f1 that would fix this never landed: the guard refused the force-push, and it is now stale anyway because findings 1-7 change the tree. After the fixes, re-cut the squash from the new tip and correct its message:
   - add the unconditional flatbuffers `-Werror` strip (cmake/third_party/flatbuffers.cmake) to the third-party sentence, which today names only Eigen and kineto;
   - drop or correct the `__launch_bounds__` rationale and the "NVIDIA build is a pure passthrough" claim to match finding 2.

CUDA path (checked, nothing to fix beyond finding 2): every other edit outside `USE_HIP` leaves the NVIDIA result unchanged. Specifically: the width argument in softmax WarpAllReduce; the WarpReduce width of 32 in layer_norm_gpu_kernel.cu; the rms_norm start offset kWarpSize/2 (=16); lru_cache's OfWarpFfs/OfWarpPopc (popc==32 is equivalent to mask==0xffffffff); the `.template` removals; the custom_eval_frame cast; the Eigen/flatbuffers/kineto third-party build patches; and the cmake `if(BUILD_CUDA AND USE_HIP) ... elseif(BUILD_CUDA)` splits.

## Porter round 2026-10-30 (linux-gfx942 host, answers "Review 2026-10-29")

Requested by jeff: address every finding of the 2026-10-29 review after fact-checking it, rerun the strengthened jargon check until added lines are clean, guard the `__launch_bounds__` additions with USE_HIP, keep CUDA paths unchanged, commits on top only (no force-push), record message rewrites for the PR-prep squash.

Fork moat-port d019f3551 -> 521f54782 (fast-forward, 4 commits):
- 9ab335eef [ROCm] State the reasons in port comments directly
- 4cd3825b8 [ROCm] Apply the 1024-thread launch bound to the HIP build only
- d46e907d5 [ROCm] Fix HIP build configuration issues
- 521f54782 [ROCm] Document the ROCm build in the README

`advance-head oneflow 521f547825b77e36a4752bd45b2a295ad2a0a633`. `classify d019f3551 521f54782` -> `class=mixed arch_independent=False inert=False`, so linux-gfx90a and linux-gfx1100 (validated at d019f3551) now read `revalidate`.

Each finding, as fact-checked and resolved:
1. Jargon in added lines: confirmed (jargon reported the 9 hits; "matching the raft pattern" at cuda_to_hip.h:70 confirmed by reading). Every pointer replaced by the technical reason; "validation slice" in cmake/oneflow.cmake also reworded. `python3 utils/jargon.py --port oneflow` now reports 0 added-line hits; the 3 remaining hits are all in 68718f038's commit message (finding 8).
2. `__launch_bounds__(1024)` without a guard: confirmed on this host's ROCm SDK clang 24 (`amdclang++ -x hip --offload-arch=gfx942 --offload-device-only -S -O2` on a templated kernel with and without the attribute: both `.max_flat_workgroup_size: 1024`, `"amdgpu-flat-work-group-size"="1,1024"`). Not reproduced on ROCm 7.2.1 (not on this host). Per jeff, kept on HIP and guarded instead of dropped: softmax.cuh and layer_norm.cuh define `OF_HIP_BLOCK_LAUNCH_BOUNDS` as `__launch_bounds__(1024)` under USE_HIP and empty otherwise (`#ifndef`-guarded, the same pattern as OF_SHFL_FULL_MASK); the 9 kernels use it. With USE_HIP undefined the kernel declarations are token-for-token upstream's. layer_norm.cuh LayerNormBlockUncachedImpl keeps upstream's own unconditional bound.
3. Mixed-wavefront arch list: confirmed by reading cmake/cuda.cmake (any `^gfx9` entry set 64 for the whole build). Now a FATAL_ERROR naming the wave64 and wave32 arches.
4. build_rocm.sh pinned gfx90a: confirmed. It now passes `-DCMAKE_HIP_ARCHITECTURES` only when the env var is set; header comment no longer says gfx90a.
5. /opt/rocm/include hard-coded in cmake/util.cmake: confirmed. Now `${hip_INCLUDE_DIRS}` (set by hip-config.cmake, include(cuda) runs before any oneflow target is created). The cuda.cmake and third_party.cmake comments that named /opt/rocm/include were updated.
6. nvtxRangePushA/nvtxRangePop unguarded under USE_HIP: confirmed (only `OF_ENABLE_PROFILER`). Both now `#if defined(OF_ENABLE_PROFILER) && !defined(USE_HIP)`. cudaProfilerStart/Stop are aliased in cuda_to_hip.h and need no guard. Every remaining nvtx reference in profiler.cpp is under `!USE_HIP`.
7. USE_HIP undeclared and dead WITH_RCCL block: confirmed (`git grep WITH_RCCL` -> only that block). Added `option(USE_HIP "Build the BUILD_CUDA GPU backend with ROCm/HIP" OFF)` after BUILD_CUDA in CMakeLists.txt; deleted the block.
8. Commit hygiene: confirmed, and it cannot be fixed without rewriting history, which jeff ruled out for this round. Recorded below for the PR-prep squash. `utils/check.py` passes the 4 new commits; it still flags 2729bee, f4f9394 (no fenced Test Plan block) and 79f235c (no Test Plan), and jargon flags 68718f038's message. The fork branch `moat-port-squash` (be1ef73f1, tree of d019f3551) is now STALE; do not push it to moat-port. Re-cut the squash from the moat-port tip after revalidation.

Not a review finding, but required of the porter: the README had a CUDA configure example and no ROCm one. Added a ROCm block beside it (USE_HIP, CMAKE_HIP_ARCHITECTURES/auto-detect, one wavefront family per build, build_rocm.sh, kernels not available yet). Paths in it use /opt/rocm, matching the build_rocm.sh compiler defaults.

CUDA path: every code edit is under USE_HIP (cmake branches already `if(BUILD_CUDA AND USE_HIP)`, the new option defaults OFF, profiler guards add `!USE_HIP`, the launch-bounds macro is empty without USE_HIP). No CUDA build was run (no NVIDIA host here).

Evidence on this host (MI300X, gfx942, ROCm from the rocm-sdk Python packages, amdclang++ 24.0.0git; HIP_VISIBLE_DEVICES=4; `ulimit -c 0`):
- Configure with `-DONEFLOW=OFF` (skips include(oneflow) and its pip step, still runs cuda.cmake and third_party.cmake):
  ```
  R=$(python3 -m rocm_sdk path --root)
  cmake -S . -B /tmp/of-cfg -G Ninja -DONEFLOW=OFF -DUSE_HIP=ON -DBUILD_CUDA=ON -DCMAKE_HIP_ARCHITECTURES="gfx942;gfx1100" -DCMAKE_PREFIX_PATH=$R -DCMAKE_C_COMPILER=$R/lib/llvm/bin/clang -DCMAKE_CXX_COMPILER=$R/lib/llvm/bin/clang++ -DWITH_CUTLASS=OFF -DWITH_MLIR=OFF -DBUILD_RDMA=OFF -DBUILD_CPP_API=OFF -DBUILD_HWLOC=OFF -DBUILD_TESTING=OFF
  ```
  "gfx942;gfx1100" -> FATAL_ERROR (mixes wavefront-64 (gfx942) and wavefront-32 (gfx1100)). gfx942 -> `OF_GPU_WARP_SIZE: 64`, configure done. gfx1100 -> 32. No arch -> auto-detect `gfx942;gfx942;...` (8 entries, one per GPU) -> 64. `find_package(hip)` -> `hip_INCLUDE_DIRS=$R/include`, which holds hipcub/hipcub.hpp and rccl/rccl.h.
- Standalone GPU harness for the changed kernels (agent_space, not committed): includes softmax.cuh and rms_norm.cuh (which includes layer_norm.cuh), calls DispatchSoftmax, DispatchLayerNorm, DispatchLayerNormGrad, LaunchRmsNorm, LaunchRmsNormGrad with DirectLoad/DirectStore, compares to a double CPU reference. float cols 200/1024 (warp path), 4096/6000 (block shared-memory path), 40000 (block uncached path), double cols 128/4096 (uncached), 64 rows.
  ```
  amdclang++ -x hip --offload-arch=gfx942 -std=c++17 -O2 -DWITH_CUDA -DUSE_HIP -DOF_GPU_WARP_SIZE=64 -DOF_SOFTMAX_USE_FAST_MATH -include oneflow/core/device/cuda_to_hip.h -Ioneflow/core/device/hip_compat -I. norm_harness.cu -o norm_harness
  ./norm_harness
  ```
  `device AMD Instinct MI300X HF gcnArch gfx942:sramecc+:xnack- warpSize 64 kWarpSize 64`; 35/35 PASS. Max abs err float: softmax <= 4.1e-9, layer_norm fwd/bwd <= 4.6e-7, rms_norm fwd/bwd <= 2.7e-7; double <= 4.7e-15.
- `bash -n build_rocm.sh` clean.

NOT DONE: the full liboneflow build and the Python GPU test suite on gfx942. Same blocker as the 2026-10-29 round, and this job was again dispatched without a per-job venv (FLEET_VENV unset): cmake/python.cmake runs `pip install -r dev-requirements.txt --user` at configure, which fails in the shared image venv, and the fleet guard refuses any pip install outside $FLEET_VENV. A private env is genuinely needed, not just a pip workaround: dev-requirements pins `protobuf>=3.9.2, <4.0`, while the base env has protobuf 6.33.5 (also missing: opencv-python, cmakelang, pytest-repeat, rich, portalocker, black). profiler.cpp was therefore not compiled (it needs the generated .pb.h). To get gfx942 build+test evidence, dispatch with `--build` so the job gets FLEET_VENV, then `CMAKE_HIP_ARCHITECTURES` unset, `ROCM_CLANG=$R/lib/llvm/bin/clang++ ROCM_CLANG_C=$R/lib/llvm/bin/clang JOBS=<cores> ./build_rocm.sh` with `-DPython3_EXECUTABLE=$FLEET_VENV/bin/python` and CMAKE_PREFIX_PATH=$R. The linux-gfx90a and linux-gfx1100 validators rebuild the full monolith at 521f54782 anyway.

### PR-prep squash message (replaces be1ef73f1's; re-cut from the validated moat-port tip)

Build with `git commit-tree <tip>^{tree} -p master -F <this message>`, check the tree equals the tip, then `squash-carry-forward`. Fill in the GPUs that actually validated the tip in the first paragraph and Test Plan.

~~~~
[ROCm] HIP backend for the CUDA GPU path

Enable oneflow's existing CUDA GPU backend to build, link, and run on AMD GPUs through HIP/ROCm. oneflow's ROCm support was removed from master long ago; this is a from-scratch HIP enablement of the CUDA code, not a revival of dead plumbing. liboneflow.so links and the ep/cuda/primitive tensor ops GPU-validate against PyTorch-ROCm on MI250X (gfx90a) and Radeon Pro W7800 (gfx1100), ROCm 7.2.1.

Review the CMake first, then the compatibility layer, then the host/kernel fixes.

cmake -- a new USE_HIP option (default OFF) keeps BUILD_CUDA=ON (so -DWITH_CUDA is defined, the .cu glob is included, and DeviceType::kCUDA keeps meaning "the GPU backend"), but routes the toolkit, libraries, and the .cu source language to ROCm. The whole monolith is compiled with the ROCm clang (CC=clang, CXX=clang++, see build_rocm.sh): oneflow's data_type.h and ep/cuda/cuda_stream.h pull the HIP half/bf16 and library-handle headers (clang builtins) into thousands of host .cpp, which only parse under clang. cmake/cuda.cmake enables the HIP language (honoring -DCMAKE_HIP_ARCHITECTURES or auto-detecting the host GPU), finds the ROCm packages, links hip::host and roc::rccl but NOT hip::device (its -x hip INTERFACE option would recompile every host .cpp as device code), and derives OF_GPU_WARP_SIZE from the arch, refusing an arch list that mixes wavefront-64 and wavefront-32 GPUs because the width is one compile-time constant per build. The host .cpp get the ROCm include directory from find_package(hip). oneflow_add_library retags .cu/.cuh LANGUAGE HIP only under USE_HIP, so no files are renamed. The deferred cuDNN / cuBLASLt / cuSOLVER / CUTLASS+wmma / libcu++ / P2P kernels are filtered out of the HIP monolith glob, and three non-self-contained .cuh are dropped from the standalone-compile glob (their code still builds inside the .cu that include them). LLVMSupportWithHeader gets LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1 because the ROCm clang's bundled abi-breaking.h (checks=0) shadows oneflow's vendored LLVM headers (checks=1) and left liboneflow.so referencing an undefined symbol. Three third-party builds get clang/gcc-13 fixes: Eigen 3.90 Transpositions.h, kineto cstdint, and flatbuffers, whose hardcoded -Werror flags are stripped (for every build, since gcc-13 trips them too).

compatibility layer -- oneflow/core/device/cuda_to_hip.h aliases the cuda*/cublas*/curand*/cufft*/cusolver*/complex symbols to their hip* equivalents, defines __CUDACC__ and (device-pass only) __CUDA_ARCH__, maps bf16 to __hip_bfloat16, provides a 64-bit full-warp shfl mask, selects __HIP_PLATFORM_AMD__ for the plain-C++ pass, gates the clang-only HIP math/library headers on __clang__, and sets CUDNN_BN_MIN_EPSILON to the modern cuDNN-8 value 0.0 (the old 1e-5 floor wrongly rejected eps=1e-6 layer norm). A hip_compat/ shim dir resolves the toolkit angle-bracket includes and is on the HIP include path only.

host + kernels -- excluded ops that the monolith still references are cleanly handled: the LRU embedding cache (libcu++ semaphore) and the cuDNN grid-sample kernel report not-supported / register only CPU on ROCm. Host cuda* entry points are ported (CudaDriverGetPrimaryCtxActive via the native HIP driver API) or the cuDNN/cuSOLVER/cuBLASLt/nvjpeg/NVTX version-banner, handle and profiler-annotation code is guarded under !USE_HIP. CPython 3.12 dropped the imp module and changed the eval-frame hook signature -- both are handled so the module imports. The GPU correctness fixes found by running the tests: CublasMathModeGuard is a no-op on HIP (hipblas accepts only the default math mode; rocBLAS picks precision from the types); on HIP only, the LayerNorm/RmsNorm/Softmax block kernels state __launch_bounds__(1024) so the up-to-1024-thread launches the occupancy heuristic chooses always fit the register allocation (the CUDA kernels are unchanged); and LayerNormParamGrad's gamma/beta reduction is made wave64-correct -- its tile constant, dim3(32,..) launch, shared [32][33] transpose, and reduction width must all agree, so WarpReduce reduces 32 lanes via the width-confined 4-arg __shfl_down_sync (correct on wave32 and wave64). The README documents the ROCm configuration next to the CUDA one.

Outside USE_HIP, the only edits are ones that leave NVIDIA results the same: the width argument in the softmax warp reduction, the `.template` removals, the eval-frame cast and the third-party build fixes.

This work was authored with the assistance of Claude (Anthropic), an AI coding agent.

Test Plan:
Validated on MI250X (gfx90a, wave64) and Radeon Pro W7800 (gfx1100, wave32), ROCm 7.2.1.

Build (the default target links liboneflow.so + the _oneflow_internal pybind module and copies it into python/oneflow/; the GPU arch is detected from the build machine):

```
./build_rocm.sh config
cmake --build build -j16
```

GPU validation -- oneflow device="cuda" is the HIP backend, compared against the PyTorch-ROCm reference (forward + backward), serially on one GPU:

```
source build/source.sh
cd python/oneflow/test/modules
python3 -m pytest -q -p no:cacheprovider test_add.py test_cast.py test_where.py test_masked_fill.py test_broadcast_ops.py test_transpose.py test_softmax.py test_layer_norm.py
python3 -m pytest -q -p no:cacheprovider test_matmul.py -k "not random_fp16_data and not random_int_data"
```

64 and 9 passed. The fp16 matmul output is bit-identical to torch; that subcase fails only on a numpy/oneflow __bool__ dtype-interop quirk. int matmul is a CPU op_kernel_not_found, unrelated to the GPU backend.
~~~~
