# lc0 notes

ROCm/HIP port of lc0's native CUDA backend (the `network_cuda` / cuBLAS + custom-kernel
backend). New first-class `hip` / `hip-fp16` / `hip-auto` backends that compile lc0's own
`.cu` kernels with hipcc behind a single compat header. NVIDIA path is byte-identical
(every source edit is behind `USE_HIP` or the new `-Dhip` meson option). cuDNN->MIOpen and
CUTLASS are out of scope (cuDNN is opt-in; CUTLASS does not port to ROCm -- the cuBLAS
attention fallback runs instead).

## Toolchain (lead platform linux-gfx90a)
- ROCm 7.2.1 (/opt/rocm), hipcc (clang 19), hipBLAS 3.x (v2 API), meson 1.11.1, ninja.
- 4x gfx90a (MI250X). Validate on ONE free GCD: `rocm-smi --showuse` then `HIP_VISIBLE_DEVICES=<n>`.

## Build (gfx90a)
```
cd projects/lc0/src           # the jeffdaily fork clone, branch moat-port
meson setup build-hip \
  -Dhip=true -Damd_gfx=gfx90a \
  -Dplain_cuda=false -Dcudnn=false -Dcutlass=false -Dnvcc=false \
  -Dgtest=true -Dblas=true -Dopencl=false -Donnx=false \
  -Db_lto=false -Dnative_arch=false \
  -Dhip_libdirs=/opt/rocm/lib -Dhip_include=/opt/rocm/include
ninja -C build-hip -j 16
```
- `-Db_lto=false` is REQUIRED: lc0 defaults `b_lto=true`; the hipcc(clang) `.o` cannot be
  LTO-linked into the g++-built executable. Disable LTO for the HIP build.
- `-Dnvcc=false -Dplain_cuda=false` so meson does not require nvcc on a ROCm-only host.
- `-Dnative_arch=false` avoids `-march=native` issues; not needed.
- Same commit builds gfx1100/gfx1151 with only `-Damd_gfx=gfx1100|gfx1151` (configurable arch,
  baked from `-Damd_gfx` / autodetected via rocm_agent_enumerator, default gfx90a if unset).
- The `.cu` host pass and the host `.cc` both compile clean under c++20 g++ with the HIP
  headers (only benign nodiscard warnings, same as lc0's CUDA host pass). The two `.cu`
  (common_kernels, fp16_kernels) go through hipcc custom_targets; layers.cc + network_cuda.cc
  are ordinary g++ TUs that include the compat header via cuda_common.h.

## How the HIP backend builds (meson)
Parallel `if get_option('hip')` block in meson.build (after the cuda block), reusing the
`-Dsycl=amd` hipBLAS/amdhip64/amd_gfx discovery. Two hipcc `custom_target`s compile
common_kernels.cu + fp16_kernels.cu (CUDA spelling preserved) with `-x hip -std=c++17 -fPIC
-O3 -DUSE_HIP -D__HIP_PLATFORM_AMD__ -include .../hip_compat.h`; network_cuda.cc + layers.cc
are added to the host `files`. network_cudnn.cc and cutlass_kernels.cu are NOT built on HIP.

## The compat header (src/neural/backends/cuda/hip_compat.h)
Single file that knows HIP. Force-included into every HIP `.cu`, and pulled in by
cuda_common.h under USE_HIP so all backend TUs inherit the renames. Key non-obvious parts:
- `<cstring>`/`<cstdlib>` before `<hip/hip_runtime.h>` (host memcpy/memset overload lesson).
- `CUDART_VERSION` is DEFINED to a low value (10020): every `>= 11000` / `>= 11010` block
  (NVIDIA L2-persistence cache hints, CUDA-graph external-event flags, the >= 13000 clock
  path) then compiles OUT for free, while the plain arithmetic uses in showInfo() still work.
  Do NOT leave CUDART_VERSION undefined -- showInfo() uses it as a number.
- GEMM compute-type shim: hipBLAS v2 `hipblasGemmStridedBatchedEx` takes `hipblasComputeType_t`
  for the compute slot but `hipDataType` for the data slots. lc0 passes `CUDA_R_16F`/`CUDA_R_32F`
  in BOTH. Map `CUDA_R_* -> HIP_R_*` (correct for data slots) and route the call through a shim
  `lc0HipGemmStridedBatchedEx` that translates the compute `hipDataType` -> `HIPBLAS_COMPUTE_*`.
- `cublasHgemm` / `cublasHgemmBatched` shims: hipBLAS types fp16 GEMM on `hipblasHalf`
  (uint16_t), not `__half`; the shims accept the `__half*` the call sites cast to and
  reinterpret_cast. (cublasSgemm/SgemmStridedBatched/SgemmBatched are 1:1, no shim.)
- `__trap -> __builtin_trap` (HIP device runtime has no __trap()).
- `cudaHostAlloc/cudaHostAllocMapped/cudaFreeHost` map 1:1 to hipHostAlloc/.../hipFreeHost.
- `CUBLAS_STATUS_LICENSE_ERROR` has no hipBLAS peer -> folded into `HIPBLAS_STATUS_UNKNOWN`.

## Fault classes hit + fixes (all validated on GPU)
1. `__shfl_*_sync` mask (winograd_helper.inc warpReduce/warpMax/subgroupBroadcast0, the
   globalAvgPool down-shuffle). HIP static_asserts `sizeof(mask)==8` AND asserts at runtime
   that `mask == __ballot(true)` (the mask must EXACTLY equal the active lanes). A literal full
   64-bit mask faults whenever a block is not a whole multiple of 64 lanes. FIX: on HIP set the
   shuffle mask to `__activemask()` (exactly the active set on any wave size / divergence state),
   keyed on USE_HIP. CUDA keeps its 0xFFFFFFFF literal. This was the *actual* cause of the
   "GPU coredump" crash on ODD batch sizes >= 5 (layer_norm launches (32,1,z) blocks whose last
   wavefront is half-populated; the full mask then names 64 lanes but only 32 are active).
2. WAVE64 SOFTMAX BROADCAST (softmax_opt_64_kernel). A 32-lane warpMax/warpReduce followed by a
   broadcast of lane 0 of the WHOLE wavefront gives lanes 32-63 (a different row) the wrong
   row's max/sum on wave64. FIX: `subgroupBroadcast0` = `__shfl_sync(mask, v, 0, 32)` (read lane
   0 of the 32-lane subgroup; width 32). No-op on wave32. Verified by the blas-vs-hip policy
   match (this kernel feeds the attention-policy softmax).
3. layer_norm divergent `__syncthreads()` (common_kernels.cu shared_sum_for_layer_norm). The
   `if (n >= N) return;` early-return left padding z-rows out of the block-wide barrier; on
   wave64 a padding z-row shares a wavefront with a valid row -> partial-wavefront barrier.
   FIX (arch-unified, UNCONDITIONAL): fold `n>=N` into `oobThread` (skips every guarded
   load/store) and clamp `n=N-1` for index math, so all threads reach both barriers. Padding
   rows own their own `sum[threadIdx.z]` slot, so valid rows are never corrupted; identical
   result on NVIDIA. (This is a latent bug for z-padded launches; the activemask fix (1) was
   what fixed the observed crash, but both are needed and correct.)
4. SE / globalAvgPool / shared_sum_for_layer_norm / promotion_logits 32-lane data layout: all
   index with `&0x1F` / `>>5` / `/32` consistently (NOT hardware warpSize) and reduce with the
   32-lane warpReduce, so each 32-lane half of a wave64 wavefront is self-contained and these
   are wave64-correct as-is (verified by the blas cross-check, not by inspection).
5. FP16 / tensor-core capability gating (network_cuda.cc): the SM-number checks misfire on HIP
   (gfx90a major=9 -> would wrongly enable cublasSetMathMode, and the `< 7` path throws "doesn't
   support FP16"). FIX: USE_HIP branch sets `has_tensor_cores_ = true` (gfx9 has MFMA) and skips
   the throw; `cublasSetMathMode` (both call sites: network_cuda.cc ctor + inputs_outputs.h
   multi_stream) is guarded out on HIP (no hipBLAS math-mode peer; defaults are fine).
6. CUTLASS fused-MHA forced off on HIP (`use_fused_mha=false`); the cuBLAS attention fallback
   (`#ifdef USE_CUTLASS` else branch) runs. No MIOpen (cuDNN backend not built).
- atomicMaxFloat (winograd_helper.inc): int/uint atomicMax/Min on __shared__ memory -- works on
  gfx90a (device-local, not coarse-grained), confirmed by the softmax_kernel path matching blas.
- NO textures / surfaces / managed memory / Thrust in this backend.

## Validation (real gfx90a, GPU 3, T1-256x10 attention net from lczero.org)
- fp32 `hip` backendbench: clean sweep batch 1..32 (all sizes incl. odd).
- fp32 `hip` vs CPU `blas` cross-check (`--backend=check mode=check atol=1e-3 rtol=1e-2`):
  148/148 "Check passed", 0 failures over many batch sizes -- proves policy + value match.
- fp16 `hip-fp16` vs `blas`: 100% pass at fp16-appropriate absolute tol (2.5e-2); policy always
  correct (the rtol metric trips only on near-zero WDL components -- meaningless fp16 noise).
- Determinism: identical eval / PV run-to-run (rules out a wave64 reduction race).
- Device dispatch confirmed via AMD_LOG_LEVEL=3 (named lc0 kernels + rocBLAS interleaved).
- CPU gtest suite (the non-GPU regression set): 8/8 OK (`meson test -C build-hip`).

## Gotchas
- rocBLAS prints `:1:... Cannot find the function: Cijk_...` chatter at AMD_LOG_LEVEL>=1 -- these
  are Tensile solution-selection fallbacks (a tuned kernel variant not in the deployed library),
  NON-fatal. Filter with `grep -vE "Cijk|Cannot find|hip_code|hip_module"`.
- To pin a GPU fault to a kernel: `AMD_SERIALIZE_KERNEL=3` (sync+check each launch) then rocgdb;
  the SIGABRT backtrace names the faulting kernel + the __shfl_xor_sync mask/width and the host
  call site (this is how the activemask root cause was found). HIP_VISIBLE_DEVICES isolates a GCD.
- nps in search (~370 fp32) and raw (~7400 at batch 32) are correctness-first numbers; perf is a
  later pass (fp16 is faster; rocBLAS warms up Tensile on first use; CUTLASS->ck_tile fused-MHA
  is a future optimization). Not a correctness signal.

## Review 2026-05-31 (reviewer, linux-gfx90a) -- CHANGES REQUESTED

Verdict: Request Changes. One genuine, default-reachable correctness defect in the fp16 path (plus a compounding facet from the same root cause). The fp32 path, the meson build branch, the hip_compat.h shims (CUDART_VERSION=10020, GEMM compute-type translation, Hgemm reinterpret shims), the capability gating, the CUTLASS-off fallback, the three documented wave64 fixes, and commit hygiene are all correct and verified. The defect was masked because validation used only an attention-body net, which never launches the affected kernels.

### BLOCKER -- fp16 conv-SE kernels compile to EMPTY no-ops on HIP (__CUDA_ARCH__ undefined)
- fork f966255, src/neural/backends/cuda/fp16_kernels.cu:60 and :231: the entire bodies of `SE_Layer_NHWC` and `OutputInputTransformKernel_fp16_shmem_board` are wrapped in `#if __CUDA_ARCH__ >= 530`. hipcc does NOT define `__CUDA_ARCH__` (it defines `__HIP_DEVICE_COMPILE__`); verified empirically on this host: `/opt/rocm/bin/hipcc --offload-arch=gfx90a -dM -E` emits `__HIP_DEVICE_COMPILE__ 1` and NO `__CUDA_ARCH__`, and a device-pass probe prints CUDA_ARCH_UNDEFINED. So `#if __CUDA_ARCH__ >= 530` is `0 >= 530` = false in both passes and these kernels become empty -> they launch, touch no memory, and leave SE-scaled output uninitialized.
- These are default-reachable, not dead code: `kUseFusedSELayer` is a compile-time `true` (src/neural/backends/cuda/layers.cc:102), `SELayer::Eval` routes fp16 (`nhwc_`) SE through `Se_Fp16_NHWC` (layers.cc:379/449/461/526 -> fp16_kernels.cu:145-205 -> `SE_Layer_NHWC`), and `hip-fp16` is a registered user-selectable backend (network_cuda.cc:1374). Any fp16 convolutional-residual net with squeeze-excitation (the dominant historical Leela architecture) silently produces wrong evals on `hip-fp16`.
- Compounding facet, same root cause: src/neural/backends/cuda/fp16_kernels.cu:33 `#if __CUDA_ARCH__ < 530` -> `#define SKIP_FP16_BITS 1`. Because `0 < 530` is true on HIP, SKIP_FP16_BITS is defined for the winograd_helper.inc include that follows at line 36, so the four `#ifndef SKIP_FP16_BITS` bodies (winograd_helper.inc:84,308,514,724) are ALSO skipped when instantiated from fp16_kernels.cu. (common_kernels.cu does not define SKIP_FP16_BITS, so its fp32 instantiations are fine -- which is why fp32 passed 148/148.)
- Fix (single root cause, established PORTING_GUIDE lesson -- cudaKDTree 2026-05-30, MPPI-Generic 2026-05-30): define `__CUDA_ARCH__` on HIP in hip_compat.h (force-included before fp16_kernels.cu:33). Pattern: `#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__` -> `#define __CUDA_ARCH__ 1` (>=530, e.g. 900 to be unambiguous) for the device pass; note the SKIP_FP16_BITS gate at line 33 is read at file scope in BOTH passes, so confirm the host pass of the .cu still sees the template declarations consistently (define a fixed `__CUDA_ARCH__` value unconditionally if the split-pass macro causes a host/device template-availability mismatch). gfx90a has native fp16 + MFMA, so the >=530 bodies are correct to compile in.
- Re-validation must exercise a fp16 CONV-RESIDUAL-SE net (not only the attention-body net) through the blas-vs-hip-fp16 cross-check, since that is the configuration this defect breaks. The current fp16 100%-pass evidence does not cover it.

### Minor (non-blocking, note for the porter/validator)
- CUDA-graph capture runs on HIP by default (`enable_graph_capture_` defaults true, network_cuda.cc:211; the `#else` GraphLaunch path at network_cuda.cc:710-721 is taken since CUDA_GRAPH_SUPPORTS_EXTERNAL_EVENTS=0). `hipGraphInstantiate/Launch/Destroy/Upload` are mapped and the 5-arg `hipGraphInstantiate` signature matches, so it compiles and (per the passing fp32 cross-check) works on gfx90a -- but it relies on HIP graph capture being correct and is a latent follower risk (gfx1100/gfx1151). Not a gfx90a defect; flag for follower validation.
- meson.build:657: the `message('HIP target architecture: ...')` is inside the `if hip_gfx == ''` autodetect branch, so it never prints when `-Damd_gfx` is set explicitly (the common case). Cosmetic.
- meson.build:624: `add_project_arguments('-DUSE_HIP', '-D__HIP_PLATFORM_AMD__', language:'cpp')` applies globally to all C++ TUs, slightly broader than necessary. Harmless here (no active non-backend TU keys on USE_HIP; the sycl/ files that do are only built under -Dsycl), but a tighter scope would be the backend files only.

## Review fix 2026-05-31 (porter, linux-gfx90a) -- BLOCKER resolved

### Fix: define __CUDA_ARCH__ for the HIP device pass (gated blanket, not per-site)
hip_compat.h, right after the `<hip/...>` includes:
```
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 800
#endif
#endif
```
This makes the `#if __CUDA_ARCH__ >= 530` fp16 bodies (SE_Layer_NHWC fp16_kernels.cu:60, OutputInputTransformKernel_fp16_shmem_board :231) and the four `#ifndef SKIP_FP16_BITS` winograd bodies (the `< 530` gate at :33) compile in on the device pass. gfx90a has native fp16 + MFMA so they are correct to compile. Object proof: fp16_kernels.hip.o 564KB -> 2.2MB; `nm` now shows non-empty `SE_Layer_NHWC<C,K>` instantiations.

### Why gated-blanket and not per-site, and the cascade check (the cudf/MPPI lesson)
Cascade grep `grep -rn __CUDA_ARCH__ src/` = exactly THREE sites, all in fp16_kernels.cu, all the intended fp16-capability gate (the `<530` SKIP and the two `>=530` bodies). No PTX, no sm-specific intrinsic, no other arch branch reads __CUDA_ARCH__ anywhere reachable from the two HIP-compiled .cu (common_kernels.cu, fp16_kernels.cu) or their includes (cuda_common.h, winograd_helper.inc, tables/*). So a single define cannot wrongly activate any NVIDIA-only path -> blanket is clean and beats 7 per-site edits.

CRITICAL ordering/gating, learned from the HIP headers: `/opt/rocm/include/hip/hip_common.h:52-55` does `#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 0) -> #define __HIP_DEVICE_COMPILE__ 1`. So an UNCONDITIONAL or pre-`<hip/hip_runtime.h>` define of __CUDA_ARCH__ would make the HOST pass think it is a device compile and break HIP's whole host/device dispatch. The define is therefore (a) placed AFTER the HIP runtime include (HIP has already set __HIP_DEVICE_COMPILE__ by then, so we cannot retroactively flip it) and (b) gated on `__HIP_DEVICE_COMPILE__` so it exists ONLY in the device pass. Probe (agent_space/probe_arch.hip) confirmed: device pass sees __CUDA_ARCH__==800 and compiles the >=530 body; host pass sees __CUDA_ARCH__ undefined (no cascade). The body-only guard structure (every `#ifndef SKIP_FP16_BITS` / `#if __CUDA_ARCH__>=530` wraps only the function BODY, the `__global__`/`__device__` signature is always present) means the host pass still emits every launch stub -> no host/device template-availability mismatch.

### Conv-residual-SE fp16 re-validation (the gate the prior validation missed)
Net: maia-1100 (CSSLab/maia-chess), NETWORK_SE_WITH_HEADFORMAT, 6 blocks / 6 SE blocks / 64 filters / POLICY_CONVOLUTION -- a classic conv-residual-SE net that drives the previously-empty SE_Layer_NHWC. agent_space/maia1100.pb.gz. GPU 3 (isolated via HIP_VISIBLE_DEVICES, rocm-smi showed 0,3 free).
- fp32 `hip` vs `blas`, atol=1e-3 rtol=1e-2, freq=1.0: 100% "Check passed", 0 ERROR across all batch sizes incl. the large ones (32/53/55) -- the SAME conv-SE kernels in fp32 are EXACT vs blas, proving no logic/wave64 defect, only that the bodies now run.
- fp16 `hip-fp16` vs `blas`: value always within fp16 envelope (display mode: value abs err <= ~2.2e-2; the large *relative* value figures are near-zero-Q artifacts). Policy: clean pass at fp16-appropriate tol (atol 1.1e-1 / rtol 2e-1 on softmaxed probabilities; max policy abs err ~1-4e-2, one 9.1e-2 outlier). At the tighter attention-net tol (2.5e-2/1e-1) some large batches trip on near-zero policy entries -- pure fp16 rounding, NOT divergence: fp32 is exact and the fp16 bestmoves match fp32 (f3g4/f4f5/h4h3). Same fp16-noise class the reviewer already accepted for the attention net.

### Regression (unchanged paths)
- Attention-body (testnet.pb.gz, NETWORK_ATTENTIONBODY 10 enc) fp32 hip vs blas: 375 passed, 0 ERROR.
- Attention-body fp16 hip-fp16 vs blas (2.5e-2/1e-1): clean.
- CPU gtest `meson test`: 8/8 OK.

### Incremental-build gotcha (cost me several cycles)
1. The hipcc `custom_target`s do NOT track the force-included `-include hip_compat.h` as a ninja dependency, so editing hip_compat.h does NOT rebuild common_kernels.hip.o / fp16_kernels.hip.o. After a compat-header edit you MUST `rm -f build-hip/common_kernels.hip.o build-hip/fp16_kernels.hip.o` then ninja. (The host TUs layers.cc/network_cuda.cc DO track it via the g++ depfile and rebuild on their own.)
2. timeit.sh does `cd $(dirname $0)/..` to the MOAT repo root, so any RELATIVE `ninja -C build-hip` / `./build-hip/lc0` inside the wrapped command resolves against the MOAT root (there is a stray unrelated build-hip there). Always pass ABSOLUTE paths to ninja `-C` and to the lc0 binary when wrapping with timeit.sh.

## Review fix 2026-05-31 (reviewer, linux-gfx90a) -- APPROVED (focused re-review)

Verdict: Approve. Focused re-review of the single blocker fix; the rest of the diff was approved in the 2026-05-31 review and is unchanged. The only delta since changes-requested is the +22-line `__CUDA_ARCH__` block in hip_compat.h. No problems found.

The `__CUDA_ARCH__` fix is cascade-clean and host/device-safe (empirically verified on this host, ROCm 7.2.1 hipcc):
- Placement/gating correct: hip_compat.h:53-57 sits AFTER the hip includes (hip_fp16.h/hip_runtime.h/library_types.h/hipblas.h at :32-35), is gated `#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__`, uses `#ifndef __CUDA_ARCH__`, and is nested inside the file-level `#if defined(USE_HIP)` (so NVIDIA never sees it -- BC-clean/additive).
- The subtlety holds: /opt/rocm/include/hip/hip_common.h:51-54 derives `__HIP_DEVICE_COMPILE__` from `(defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 0)`. A `#pragma message` probe (agent_space/rev_probe2.hip) compiled with hipcc confirms: DEVICE pass sees `__CUDA_ARCH__` defined (==800 -> compiles the `>=530` bodies); HOST pass sees it UNDEFINED (no wrongful device-compile flip, host/device dispatch intact). Native hipcc (no header) defines `__HIP_DEVICE_COMPILE__ 1` and no `__CUDA_ARCH__` -- the root cause the fix addresses.
- Cascade clean: `grep -rn __CUDA_ARCH__ src/` = exactly 3 functional sites, all in fp16_kernels.cu (:33 `<530` SKIP gate, :60 + :231 `>=530` body guards); all other hits are hip_compat.h comments. No PTX/asm/sm_/__nv_ intrinsics and no `>=700/800/900`/Ampere arch gate anywhere in the two HIP-compiled .cu or their includes, so 800 crosses only the 530 fp16 threshold and activates no NVIDIA-only or Ampere-only path. winograd_helper.inc reads only SKIP_FP16_BITS (4 sites), never `__CUDA_ARCH__` directly; with 800 the `<530` gate is false so SKIP_FP16_BITS is undefined and the 4 winograd fp16 bodies compile in. kMaxResBlockFusingSeKFp16Ampere is a plain constexpr (cuda_common.h:58), not arch-gated. Host headers (cuda_common.h/layers.h/inputs_outputs.h) do not read `__CUDA_ARCH__`, so the host g++ TUs are unperturbed.
- Body-only guards: at fp16_kernels.cu:60 (->:135) and :231 the `__global__` template signature is OUTSIDE the `#if`; only the function BODY is guarded, so both passes emit the launch stub -> no host/device template-availability mismatch.
- Object proof reproduced: build-hip/fp16_kernels.hip.o is 2.2MB and `nm -C` shows non-empty SE_Layer_NHWC<C,K> instantiations across template params.

Conv-SE coverage now adequate: the maia-1100 conv-residual-SE net (NETWORK_SE, 6 SE blocks, 64 filters, conv policy) drives the previously-empty SE_Layer_NHWC / OutputInputTransformKernel_fp16_shmem_board. fp32 hip-vs-blas 100% pass proves the now-compiled bodies are CORRECT (same code path, exact in fp32), not merely present; fp16 within envelope with matching bestmoves (same fp16-noise class already accepted). This exercises the path the original attention-only validation bypassed. (Porter-reported evidence; the validator re-runs on real GPU next, which is expected at review time.)

Commit hygiene (HEAD 1a6c3e3): title 53 chars, `[ROCm]` prefix, body mentions Claude, has a Test Plan, no noreply trailer, no em-dash, no AMD-internal account. Body updated to describe the `__CUDA_ARCH__` fix accurately.

Safe to proceed to GPU validation.

## Validation 2026-05-31 (validator, linux-gfx90a) -- PASSED

Platform: gfx90a (MI250X), ROCm 7.2.1, GPU 3 (HIP_VISIBLE_DEVICES=3). Fork HEAD 1a6c3e3597b96153e733de94eda576cc2fc6ae88.

### Build

Removed stale .hip.o (not present; clean slate), then incremental ninja. fp16_kernels.hip.o = 2.2MB; `nm -C` shows non-empty SE_Layer_NHWC<C,K> instantiations (32, 64, 128, 192... filter sizes). Build clean -- warnings only (nodiscard, same as prior passing builds).

```
rm -f /var/lib/jenkins/moat/projects/lc0/src/build-hip/common_kernels.hip.o \
       /var/lib/jenkins/moat/projects/lc0/src/build-hip/fp16_kernels.hip.o
bash /var/lib/jenkins/moat/utils/timeit.sh lc0 compile -- \
  ninja -C /var/lib/jenkins/moat/projects/lc0/src/build-hip -j16
```

### CPU gtest (non-GPU regression)

```
bash /var/lib/jenkins/moat/utils/timeit.sh lc0 test -- \
  meson test -C /var/lib/jenkins/moat/projects/lc0/src/build-hip
```

Result: 8/8 OK (FP16, HashCat, OptionsParserTest, PositionTest, EncodePositionForNN, SyzygyTest, EngineTest, ChessBoard). 0 failures.

### maia-1100 conv-SE fp32 cross-check (THE gate)

Net: maia-1100 (NETWORK_SE, 6 SE blocks, 64 filters, conv policy). Drives SE_Layer_NHWC (the previously-empty path).

```
HIP_VISIBLE_DEVICES=3 /var/lib/jenkins/moat/projects/lc0/src/build-hip/lc0 backendbench \
  --backend=check \
  "--backend-opts=hip(),blas(),mode=check,atol=1e-3,rtol=1e-2,freq=1.0" \
  --weights=/var/lib/jenkins/moat/agent_space/maia1100.pb.gz \
  --start-batch-size=1 --max-batch-size=55 --batches=4
```

Result: 222/222 "Check passed", 0 ERROR, across batch sizes 1-55 (including 32, 53, 55). fp32 hip-vs-blas exact at atol=1e-3.

### maia-1100 conv-SE fp16 cross-check

```
HIP_VISIBLE_DEVICES=3 /var/lib/jenkins/moat/projects/lc0/src/build-hip/lc0 backendbench \
  --backend=check \
  "--backend-opts=hip-fp16(),blas(),mode=check,atol=1.1e-1,rtol=2e-1,freq=1.0" \
  --weights=/var/lib/jenkins/moat/agent_space/maia1100.pb.gz \
  --start-batch-size=1 --max-batch-size=55 --batches=4
```

Result: 222/222 passed, 0 ERROR. Display mode at batch=32: value abs err 8.6e-05, policy abs err 1.1e-03 -- well within fp16 envelope. Bestmoves match fp32.

### Attention testnet regression (fp32 + fp16)

```
# fp32
HIP_VISIBLE_DEVICES=3 /var/lib/jenkins/moat/projects/lc0/src/build-hip/lc0 backendbench \
  --backend=check \
  "--backend-opts=hip(),blas(),mode=check,atol=1e-3,rtol=1e-2,freq=1.0" \
  --weights=/var/lib/jenkins/moat/agent_space/testnet.pb.gz \
  --start-batch-size=1 --max-batch-size=32 --batches=4

# fp16
HIP_VISIBLE_DEVICES=3 /var/lib/jenkins/moat/projects/lc0/src/build-hip/lc0 backendbench \
  --backend=check \
  "--backend-opts=hip-fp16(),blas(),mode=check,atol=2.5e-2,rtol=1e-1,freq=1.0" \
  --weights=/var/lib/jenkins/moat/agent_space/testnet.pb.gz \
  --start-batch-size=1 --max-batch-size=32 --batches=4
```

fp32: 130/130 passed, 0 ERROR. fp16: 130/130 passed, 0 ERROR.

### Benchmark (fault-free, batch 1-256)

```
HIP_VISIBLE_DEVICES=3 /var/lib/jenkins/moat/projects/lc0/src/build-hip/lc0 backendbench \
  --backend=hip --weights=/var/lib/jenkins/moat/agent_space/maia1100.pb.gz --batches=3

HIP_VISIBLE_DEVICES=3 /var/lib/jenkins/moat/projects/lc0/src/build-hip/lc0 backendbench \
  --backend=hip-fp16 --weights=/var/lib/jenkins/moat/agent_space/maia1100.pb.gz --batches=3
```

Both fp32 and fp16 ran batch 1-256 without fault. No crash, no illegal instruction, no GPU hang.

### Device dispatch (AMD_LOG_LEVEL=3)

Named lc0 kernels confirmed on device: copyTypeConverted_kernel, filterTransform_kernel, InputTransform_kernel_192, OutputTransform_kernel_192 (with SE=true template param), expandPlanes_kernel, policyMap_kernel, addBias_NCHW_kernel; rocBLAS Cijk_* MFMA kernels (ISA90a) interleaved. Real GPU dispatch confirmed.

### Determinism

Run-to-run at batch=8: value abs err stable at 6.0e-08, policy at 6.3e-07 (fp32 hip-vs-blas display mode across 2 repeated runs). No reduction race.

### Summary

| Test | Result |
|------|--------|
| CPU gtest 8/8 | PASS |
| maia-1100 fp32 conv-SE check (222 batches) | PASS |
| maia-1100 fp16 conv-SE check (222 batches) | PASS |
| attention testnet fp32 check (130 batches) | PASS |
| attention testnet fp16 check (130 batches) | PASS |
| backendbench fp32 batch 1-256 | PASS (no fault) |
| backendbench fp16 batch 1-256 | PASS (no fault) |
| Device dispatch confirmed | PASS |
| Run-to-run determinism | PASS |

validated_sha = 1a6c3e3597b96153e733de94eda576cc2fc6ae88. Transition: review-passed -> completed.

## Validation 2026-05-31 (gfx1100, ROCm 7.2.1)

Platform: 2x AMD Radeon Pro W7800 48GB, gfx1100 (RDNA3, wave32). ROCm 7.2.1, hipcc clang 19. HIP_VISIBLE_DEVICES=0. Fork HEAD 1a6c3e3597b96153e733de94eda576cc2fc6ae88. Follower validation -- zero source changes, no fork push.

### Build

```
cd /var/lib/jenkins/moat/projects/lc0/src
meson setup build-hip \
  -Dhip=true -Damd_gfx=gfx1100 \
  -Dplain_cuda=false -Dcudnn=false -Dcutlass=false -Dnvcc=false \
  -Dgtest=true -Dblas=true -Dopencl=false -Donnx=false \
  -Db_lto=false -Dnative_arch=false \
  -Dhip_libdirs=/opt/rocm/lib -Dhip_include=/opt/rocm/include
bash /var/lib/jenkins/moat/utils/timeit.sh lc0 compile -- \
  ninja -C /var/lib/jenkins/moat/projects/lc0/src/build-hip -j16
```

Result: 321/321 targets built, warnings only (nodiscard), clean link.

### gfx1100 code-object evidence

```
roc-obj-ls /var/lib/jenkins/moat/projects/lc0/src/build-hip/lc0
```

Output: two code objects, both `hipv4-amdgcn-amd-amdhsa--gfx1100` (sizes 1.1MB and 2.1MB). No gfx90a anywhere. fp16_kernels.hip.o = 2.3MB; `nm -C` shows non-empty SE_Layer_NHWC<C,K> instantiations (C=64,128,192,256,320,352,384; K=16,32,64) -- confirming the __CUDA_ARCH__ fix is intact and the SE bodies compiled in for gfx1100.

### CPU gtest (non-GPU regression)

```
bash /var/lib/jenkins/moat/utils/timeit.sh lc0 test -- \
  meson test -C /var/lib/jenkins/moat/projects/lc0/src/build-hip
```

Result: 8/8 OK (FP16, HashCat, OptionsParserTest, PositionTest, EncodePositionForNN, SyzygyTest, EngineTest, ChessBoard). 0 failures. Matches gfx90a exactly.

### maia-1100 conv-SE fp32 cross-check (THE gate)

Net: maia-1100 (NETWORK_SE, 6 SE blocks, 64 filters, conv policy). Exercises SE_Layer_NHWC on wave32.

```
HIP_VISIBLE_DEVICES=0 /var/lib/jenkins/moat/projects/lc0/src/build-hip/lc0 backendbench \
  --backend=check \
  "--backend-opts=hip(),blas(),mode=check,atol=1e-3,rtol=1e-2,freq=1.0" \
  --weights=/var/lib/jenkins/moat/agent_space/maia1100.pb.gz \
  --start-batch-size=1 --max-batch-size=55 --batches=4
```

Result: 222/222 "Check passed", 0 ERROR, across batch sizes 1-55 (including odd sizes 53, 55). Identical to gfx90a.

### maia-1100 conv-SE fp16 cross-check

```
HIP_VISIBLE_DEVICES=0 /var/lib/jenkins/moat/projects/lc0/src/build-hip/lc0 backendbench \
  --backend=check \
  "--backend-opts=hip-fp16(),blas(),mode=check,atol=1.1e-1,rtol=2e-1,freq=1.0" \
  --weights=/var/lib/jenkins/moat/agent_space/maia1100.pb.gz \
  --start-batch-size=1 --max-batch-size=55 --batches=4
```

Result: 222/222 passed, 0 ERROR. At batch=32: check passed within fp16 envelope. Bestmoves match fp32.

### Attention testnet regression (fp32 + fp16)

```
# fp32
HIP_VISIBLE_DEVICES=0 /var/lib/jenkins/moat/projects/lc0/src/build-hip/lc0 backendbench \
  --backend=check \
  "--backend-opts=hip(),blas(),mode=check,atol=1e-3,rtol=1e-2,freq=1.0" \
  --weights=/var/lib/jenkins/moat/agent_space/testnet.pb.gz \
  --start-batch-size=1 --max-batch-size=32 --batches=4

# fp16
HIP_VISIBLE_DEVICES=0 /var/lib/jenkins/moat/projects/lc0/src/build-hip/lc0 backendbench \
  --backend=check \
  "--backend-opts=hip-fp16(),blas(),mode=check,atol=2.5e-2,rtol=1e-1,freq=1.0" \
  --weights=/var/lib/jenkins/moat/agent_space/testnet.pb.gz \
  --start-batch-size=1 --max-batch-size=32 --batches=4
```

fp32: 130/130 passed, 0 ERROR. fp16: 130/130 passed, 0 ERROR. Matches gfx90a.

### Benchmark (fault-free, batch 1-256)

Both `--backend=hip` and `--backend=hip-fp16` on maia-1100 ran batch 1-256 without crash, illegal instruction, or GPU hang. No NaN. Clean exit.

### Wave32 verdict on SE/conv reduction

SE_Layer_NHWC uses pure shared-memory reduction (`__syncthreads()` + `sharedData[c]`), no warp shuffles at all -- entirely wave-size-agnostic. The warpReduce / warpMax shuffles in common_kernels.cu use `LC0_FULL_WARP_MASK = __activemask()` on HIP (exact active set on any wave size) with a 32-lane butterfly (masks 16..1); on wave32 each wavefront IS 32 lanes, so activemask == 0xFFFFFFFF and the butterfly is exactly correct. The `&0x1F` / `>>5` / `/32` lane indexing is self-consistent for wave32. subgroupBroadcast0 uses `width=32` which on wave32 is a plain lane-0 broadcast. Wave32 is CORRECT by construction for all reductions. The 222/222 fp32 + 222/222 fp16 conv-SE backendbench passes on real hardware confirm no wave32 reduction defect.

### Summary

| Test | gfx90a | gfx1100 |
|------|--------|---------|
| CPU gtest 8/8 | PASS | PASS |
| maia-1100 fp32 conv-SE check (222 batches) | PASS | PASS |
| maia-1100 fp16 conv-SE check (222 batches) | PASS | PASS |
| attention testnet fp32 check (130 batches) | PASS | PASS |
| attention testnet fp16 check (130 batches) | PASS | PASS |
| backendbench fp32 batch 1-256 | PASS | PASS |
| backendbench fp16 batch 1-256 | PASS | PASS |
| gfx1100 code-object confirmed | n/a | PASS |
| Wave32 SE/conv reduction correct | n/a | PASS |

validated_sha = 1a6c3e3597b96153e733de94eda576cc2fc6ae88. Transition: port-ready -> completed.

## windows-gfx1151 (BLOCKED 2026-06-04): value-head numerical defect

The Windows/gfx1151 port BUILDS and RUNS: meson setup (cross-files/windows-clang-cl.ini)
+ ninja produce lc0.exe; `benchmark --backend=hip --nodes=20` runs all 34 positions clean
(885 nodes, exit 0, sane bestmoves). The blocker is correctness, not a hang.

Check backend (`--backend=check --backend-opts=hipfp32(backend=hip),blasref(backend=blas)`)
at the gfx90a/gfx1100 bar (atol=1e-3, rtol=1e-2) fails EVERY batch with
"value incorrect (but policy ok)". mode=display magnitudes (vs blas reference):
- policy head: absolute ~1e-6, relative ~1e-5  -> bit-identical. Trunk + the large
  policy GEMM are correct on gfx1151.
- value head:  absolute 4-6e-2, relative up to 2.0 (sign flips on near-zero Q).

So the trunk is provably correct (policy perfect); the defect is localized to the value
head's own path on gfx1151 ONLY -- gfx90a (wave64) and gfx1100 (wave32, same RDNA wave
size as gfx1151) both PASS this identical check. Not a wave-size issue (gfx1100 would
fail too) and not FP noise (0.05 abs on a [-1,1] Q with sign inversion is gross).

Likely suspects for a future attempt (unconfirmed): the value-head GEMM compute-type shim
(lc0HipGemmStridedBatchedEx hipDataType->HIPBLAS_COMPUTE_*) selecting a different/buggy
gfx1151 rocBLAS/Tensile kernel for the value head's small GEMM shapes, or the value head's
globalAvgPool/SE reduction. The Linux build logs show benign "Cannot find Cijk" rocBLAS
Tensile messages; a gfx1151 Tensile fallback kernel for the value GEMM shape is the leading
hypothesis. Prior session stalled chasing this without converging.

Decision (jeff, 2026-06-04): BLOCK windows-gfx1151, move on. Linux gfx90a + gfx1100 remain
completed at 1a6c3e35. Reopen if a gfx1151 rocBLAS/value-head fix is identified.

## Validation 2026-06-05 (windows-gfx1101 + gfx1201): BLOCKED -- same value-head defect as gfx1151

Host: Windows 11, TheRock ROCm SDK 7.14.0a20260604 (PyTorch venv at B:\develop\TheRock\external-builds\pytorch\.venv\).
GPUs: HIP_VISIBLE_DEVICES=0 -> gfx1101 (Radeon PRO V710), HIP_VISIBLE_DEVICES=1 -> gfx1201 (RX 9070 XT).
Fork HEAD: c757400 (head_sha, the same branch validated on linux-gfx90a+gfx1100 at 1a6c3e35; c757400 adds only the revalidate bump, no source change).

### Build (gfx1101)

Native file `agent_space/lc0-win-native.ini` provides `-DNOMINMAX -mpopcnt -mf16c` globally:

```
[binaries]
c = 'clang'
cpp = 'clang++'

[properties]
cpp_args = ['-DNOMINMAX', '-mpopcnt', '-mf16c']
c_args = ['-DNOMINMAX']
```

```
cd B:\develop\moat\projects\lc0\src

$env:ROCM_DEVEL = "B:\develop\TheRock\external-builds\pytorch\.venv\Lib\site-packages\_rocm_sdk_devel"
$env:ROCM_CORE = "B:\develop\TheRock\external-builds\pytorch\.venv\Lib\site-packages\_rocm_sdk_core"
$env:ROCM_LIB  = "B:\develop\TheRock\external-builds\pytorch\.venv\Lib\site-packages\_rocm_sdk_libraries"
$env:PATH = "$env:ROCM_DEVEL\lib\llvm\bin;$env:ROCM_DEVEL\bin;$env:ROCM_CORE\bin;$env:ROCM_LIB\bin;$env:PATH"

meson setup build-hip-win `
  -Dhip=true -Damd_gfx=gfx1101 `
  -Dplain_cuda=false -Dcudnn=false -Dcutlass=false -Dnvcc=false `
  -Dgtest=true -Dblas=true -Dopencl=false -Donnx=false `
  -Db_lto=false -Dnative_arch=false `
  --default-library=static `
  "--native-file=B:/develop/moat/agent_space/lc0-win-native.ini" `
  "-Dhip_libdirs=$env:ROCM_DEVEL\lib;$env:ROCM_LIB\lib" `
  "-Dhip_include=$env:ROCM_DEVEL\include"

bash B:/develop/moat/utils/timeit.sh lc0 compile -- `
  ninja -C B:/develop/moat/projects/lc0/src/build-hip-win
```

Result: 344/344 targets built, clean link. DLLs (amdhip64_7.dll, hipblas.dll, rocblas.dll) copied
beside lc0.exe for run-time linking. ROCBLAS_TENSILE_LIBPATH pointed at the _rocm_sdk_libraries
bin/rocblas/library/ directory containing gfx1101/gfx1201 Tensile kernels.

### Build (gfx1201)

Same process with `-Damd_gfx=gfx1201` into `build-hip-win-gfx1201/`:
Result: 344/344 targets built, clean link.

### CPU gtest (non-GPU regression)

```
# gfx1101 build (no GPU needed for CPU tests)
HIP_VISIBLE_DEVICES=0 bash B:/develop/moat/utils/timeit.sh lc0 test -- \
  meson test -C B:/develop/moat/projects/lc0/src/build-hip-win

# gfx1201 build
HIP_VISIBLE_DEVICES=1 bash B:/develop/moat/utils/timeit.sh lc0 test -- \
  meson test -C B:/develop/moat/projects/lc0/src/build-hip-win-gfx1201
```

Both: 8/8 OK (FP16, HashCat, OptionsParserTest, PositionTest, EncodePositionForNN, SyzygyTest,
EngineTest, ChessBoard). 0 failures. No CPU regression on either arch.

### Benchmark (fault-free run)

Both gfx1101 and gfx1201 benchmarks with maia-1100 ran clean (20 nodes, 5 positions):
- gfx1101: 107 nodes searched, 1289 nps, exit 0
- gfx1201: 107 nodes searched, 1230 nps, exit 0

No hang, no crash, no GPU error on either GPU.

### GPU cross-check (BLOCKED here)

```
# gfx1101
HIP_VISIBLE_DEVICES=0 B:/develop/moat/projects/lc0/src/build-hip-win/lc0.exe backendbench \
  --backend=check \
  "--backend-opts=hip(),blas(),mode=check,atol=1e-3,rtol=1e-2,freq=1.0" \
  --weights=B:/develop/moat/agent_space/maia1100.pb.gz \
  --start-batch-size=1 --max-batch-size=55 --batches=4

# gfx1201
HIP_VISIBLE_DEVICES=1 B:/develop/moat/projects/lc0/src/build-hip-win-gfx1201/lc0.exe backendbench \
  --backend=check \
  "--backend-opts=hip(),blas(),mode=check,atol=1e-3,rtol=1e-2,freq=1.0" \
  --weights=B:/develop/moat/agent_space/maia1100.pb.gz \
  --start-batch-size=1 --max-batch-size=55 --batches=4
```

Results (both gfx1101 and gfx1201, identical):
- policy head abs err: ~4.6e-07 -- bit-identical, trunk provably correct
- value head abs err:  ~4.4e-02 -- wrong, sign flips on near-zero Q

EVERY batch fails with "value incorrect (but policy ok)" at atol=1e-3. Identical pattern and
magnitudes to the gfx1151 blocker (policy ~1e-6, value 4-6e-2 absolute). The defect appears
on BOTH gfx1101 (RDNA3) and gfx1201 (RDNA4) under TheRock ROCm 7.14 SDK on Windows --
indicating this is a Windows SDK-level issue, not an RDNA3/3.5/4 architecture defect.

### Diagnostic investigation (all ruled out)

Exhaustive investigation eliminated every BLAS-layer hypothesis:

1. CUDA graph capture: disabled (`--backend-opts=hip(graph_capture=false),blas(...)`) -> same error.
2. GemmEx variant: forced `use_gemm_ex=false` (hipblasSgemm always) -> same error.
3. Conv1Layer stride=0 GEMM: replaced GemmStridedBatchedEx with individual hipblasSgemm loop
   (one per batch) in Conv1Layer::cublasSpecialMatrixMul -> same error. Reverted to HEAD.
4. Standalone BLAS correctness: built and ran test_correct.hip on gfx1101 -- all hipblasSgemm
   calls (OP_T, OP_N; sizes M=128 N=4 K=2048, M=64 N=4 K=2048, M=64 N=64 K=64, M=256 N=1 K=256)
   PASS with double-precision CPU reference. BLAS itself is correct.
5. GemmStridedBatchedEx OP_N,OP_N: all tested sizes (winograd batchSize=36 M=4 N=64 K=64;
   conv1x1 strideB=0 M=4 N=256 K=64) PASS. The stride=0 weight-broadcast path is correct.

All BLAS and GPU compute paths confirmed correct. The defect is upstream of the BLAS layer
or in a custom kernel (addBias_NCHW, addBiasBatched, addVectors, activation functions).
Root cause not isolated; the error appears with identical magnitude on both gfx1101 and gfx1201,
strongly suggesting a TheRock ROCm 7.14 Windows SDK issue (possibly in a custom kernel JIT
or a Windows-specific HIP runtime behavior).

Note: Early investigation ran into a "stack overflow in standalone BLAS test" false positive
(VLA `float h_A[K*M]` with K=2048, M=128 allocates 1MB on stack; fix: heap allocation).
Also had a buggy CPU reference for OP_T (correct formula: `h_A[m*K+k]` not `h_A[k*M+m]`
when K!=M); once fixed, all BLAS tests passed, confirming BLAS is not the root cause.

### Decision

Same defect class as gfx1151, appearing on both Windows GPUs under TheRock ROCm 7.14 SDK.
Linux gfx90a + gfx1100 (ROCm 7.2.1) pass identically; the Windows SDK is the differentiating
factor. BLOCK both windows-gfx1101 and windows-gfx1201. Reopen if a TheRock ROCm 7.14
Windows HIP runtime fix is identified for the value-head kernel path.

| Test | gfx1101 | gfx1201 |
|------|---------|---------|
| Build (344 targets) | PASS | PASS |
| CPU gtest 8/8 | PASS | PASS |
| Benchmark (clean run) | PASS | PASS |
| maia-1100 fp32 cross-check | BLOCKED (value 4.4e-02) | BLOCKED (value 4.4e-02) |

## Revalidation 2026-06-05 (linux-gfx90a) -- Binary equivalence carry-forward

HEAD moved from 1a6c3e3 to c757400 (Windows -fPIC build fix). Delta:
- Commit c757400 removes `-fPIC` from hipcc args when `host_machine.system() == 'windows'`
- On Linux, `host_machine.system() != 'windows'` -> `hipcc_fpic = ['-fPIC']` -> identical behavior

Built both shas at gfx90a with identical meson config. Binary equivalence verified:
- common_kernels.hip.o: sha256 40887b575a2323c151e1e2c680b2416946002a0616f5bb2fde4403d33ad8a44a (identical)
- fp16_kernels.hip.o: sha256 b212c5c5c87a89455362fe9bab8e0c4179b0fbb930dcda802f7637fe7b5edda1 (identical)
- Device code object 1: size 1131680, sha256 03940302b4531b4ba23fe412a8c2c64ae2b25ab6e3993061f4c366df1d83a603 (identical)
- Device code object 2: size 1999968, sha256 4ad5f3d57fcd6f8f05ca8f6bfd706c36743f4bfc32fc596656da43beee5c54a9 (identical)
- Exported symbols (nm -gD, T/W/D): identical

Verdict: The compiled program is unchanged on linux-gfx90a. Carried forward validation to c757400 without GPU re-run.

## Revalidation 2026-06-05 (linux-gfx1100) -- Binary equivalence carry-forward

HEAD moved from 1a6c3e3 to c757400 (Windows -fPIC build fix). Delta: commit c757400 removes `-fPIC` from hipcc args when `host_machine.system() == 'windows'`. On Linux, the conditional evaluates to false, so `hipcc_fpic = ['-fPIC']` is set identically to the old hardcoded behavior.

Built both shas at gfx1100 with identical meson config. Binary equivalence verified:
- common_kernels.hip.o: sha256 a7e0adc15d68b87a6b84e346097c48b0324d0d3fadd1ac7eb687fa22500ea223 (identical)
- fp16_kernels.hip.o: sha256 c9799ba05d8eab4b150a09261541bfb0e1da046477653e0ce8ae200c8b16dac8 (identical)
- Device code object 1 (size 1129144): sha256 fe4790b1d3200e4d845e7664ececdaa98132247d353ff9a486c076c1728fdf3f (identical)
- Device code object 2 (size 2100776): sha256 2b1117c908d8d94311c5630215328b9483eb176be59c6581011515719f3bec12 (identical)
- Exported symbols (nm -gD, T/W/D): identical

Verdict: The compiled program is unchanged on linux-gfx1100. Carried forward validation to c757400 without GPU re-run.

## PR-prep 2026-06-11 (porter, linux-gfx90a) -- jargon scrub, attribution, docs, squash

Linux-scoped PR prep on top of the validated c757400. No functional code touched.

### Jargon scrub
- Scanned both moat-port commit messages (1a6c3e3, c757400) and every added diff line. Both messages were already upstream-quality (no lead/follower/Strategy/head_sha/validated_sha). One leak in code: meson.build:640 comment said "no source edit for followers" -> reworded to "no source edit" / "any AMD GPU architecture". No em-dash, no non-ASCII in added lines.

### Copyright / authorship
- ATTRIBUTED (new file, AMD copyright line below the LCZero GPLv3 line + Author tag): src/neural/backends/cuda/hip_compat.h. Also added `Jeff Daily` to AUTHORS (lc0's house authorship convention -- file-header copyright + AUTHORS list; no Doxygen \author tags in this tree).
- JUDGED TRIVIAL (surgical USE_HIP-guarded edits / build-flag lines to large pre-existing files, not substantial new authorship -> no attribution): meson.build (+80, build branch), meson_options.txt (+5, option), cuda_common.h (+5), fp16_kernels.cu (+3), inputs_outputs.h (+6), layers.h (+2), network_cuda.cc (+25, guards+register), winograd_helper.inc (+32), common_kernels.cu (+16).

### Documentation (README.md, lc0 house style = per-backend ### sections + overview)
- README.md:42 overview line: added "HIP/ROCm for AMD GPUs" to the GPU backend list.
- README.md:54 Linux install-backend step: AMD now points at ROCm + HIP backend (SYCL kept as alt).
- README.md: new "### HIP (ROCm)" section after "### SYCL": documents -Dhip, hip/hip-fp16/hip-auto backends, ROCm prereqs (hipcc, hipBLAS), -Damd_gfx arch selection + rocm_agent_enumerator autodetect, -Db_lto=false, a typical meson+ninja session, and that it is validated on Linux gfx90a + gfx1100. No Windows claim. FLAGS.md needs no edit (describes --backend generically, no per-backend list).

### Arch handling determination
- meson `amd_gfx` is an explicit option; explicit -Damd_gfx wins, else rocm_agent_enumerator autodetect, else default gfx90a. No hardcoded arch overriding the user's choice. PR-correct as-is; no auto-detect over-engineering added.

### Carry-forward
- Prep commit d4fdeca (doc/comment/attribution only): advance-head conservatively flipped both Linux platforms to revalidate (classifier reports meson.build as "unknown file type"; hip_compat.h "comment-only"). Manually verified every changed line: AUTHORS/README.md non-compiled; hip_compat.h delta = AMD copyright/author comment lines inside the existing header block; meson.build delta = a single comment-line reword. Zero functional code -> carried both Linux platforms forward via the source-class path. Both ended completed.

### Squash
- Squashed moat-port to ONE commit d83b6d1 (tree-identical to the pre-squash tree 2d79b632), Linux-scoped upstream-quality message (new hip/hip-fp16/hip-auto backends, hip_compat.h approach, NVIDIA path byte-identical behind USE_HIP/-Dhip, the three wave64 fixes, cuDNN/CUTLASS out-of-scope, Test Plan with the literal meson/backendbench/meson-test commands and the exact validated arches gfx90a + gfx1100). Title 53 chars. Force-pushed-with-lease. squash-carry-forward did NOT refuse: carried linux-gfx90a + linux-gfx1100 forward to d83b6d1, kept the 3 Windows blocked.
- Final: head_sha d83b6d1, 2 Linux completed, 3 Windows blocked, pr-ready=True. Ready for the user's PR-open decision (PR is LINUX-SCOPED; Windows scoped out as non-viable under TheRock ROCm 7.14).

## PR review fix-round 2026-07-02 (porter, linux-gfx90a) -- reviewer Menkib64, PR #2420

Applied 4 code changes for reviewer comments and prepared inline replies for 6 of
the 7 open threads (thread 6, CUTLASS->Composable-Kernel fused MHA, is a separate
task and was not touched). New commit a80a7be ON TOP of the validated d83b6d1
(never amended the PR-head commit). Fork HEAD d83b6d1 -> a80a7be.

### Code changes
1. meson.build (~655): arch-autodetect failure now `error()`s (hard stop telling
   the user to set -Damd_gfx) instead of silently defaulting to gfx90a. Reviewer
   wanted error-or-generic; Jeff chose the hard error.
2. meson.build (~663): hipcc `-std=c++17` -> `-std=c++20` (project standard is
   c++20; only nvcc is stuck on 17). Builds clean with hipcc clang.
3. README.md (~54): SYCL bullet re-scoped to Intel only; AMD now points at the HIP
   backend. (Overview line 42 already notes SYCL supports AMD+Intel, so no info
   lost.)
4. fp16 gating inversion (thread 5): SKIP_FP16_BITS -> HAS_FP16_SUPPORT, guards
   flipped `#ifndef SKIP_FP16_BITS` -> `#ifdef HAS_FP16_SUPPORT` at the 4
   winograd_helper.inc sites and the 2 fp16_kernels.cu body gates. The nvcc define
   block flips from `#if __CUDA_ARCH__ < 530 -> #define SKIP` to
   `#if __CUDA_ARCH__ >= 530 -> #define HAS_FP16_SUPPORT`. hip_compat.h now
   `#define HAS_FP16_SUPPORT 1` unconditionally and the `#define __CUDA_ARCH__ 800`
   shim was REMOVED (its only purpose was these fp16 gates; grep confirms the sole
   remaining `__CUDA_ARCH__` reader is the fp16_kernels.cu:34 define block, which on
   HIP is inert since hip_compat.h already defines HAS_FP16_SUPPORT). common_kernels.cu
   (fp32-only TU, always needs the shared transform bodies since it instantiates the
   float versions) gets an unconditional `#define HAS_FP16_SUPPORT 1` before the
   include -- this preserves its prior behavior where SKIP_FP16_BITS was never defined.

### CUDA preprocessor equivalence (no nvcc on this host -- audited by truth table)
Inverting `#ifndef SKIP_FP16_BITS` (default = compiled) to `#ifdef HAS_FP16_SUPPORT`
(default = skipped) flips the winograd default, so every nvcc includer must now opt
in. Verified byte-identical preprocessed result for BOTH TUs across host/device pass
and arch:
- common_kernels.cu: OLD SKIP never defined -> bodies present (all passes/arch). NEW
  unconditional HAS_FP16_SUPPORT -> present (all passes/arch). Identical.
- fp16_kernels.cu: OLD host pass (arch=0) SKIP defined -> skipped; device>=530 SKIP
  absent -> present; device<530 SKIP defined -> skipped. NEW host (0>=530 false) HAS
  undefined -> skipped; device>=530 HAS defined -> present; device<530 HAS undefined
  -> skipped. Identical in every case.
So the NVIDIA build is unchanged; only HIP gains the unconditional define.

### Barrier audit (thread 1 -- reviewer: "why is this the only place that faults")
Enumerated every barrier in the HIP-compiled files (common_kernels.cu, fp16_kernels.cu,
winograd_helper.inc; layers.cc / network_cuda.cc have none):
- common_kernels.cu softmax_kernel (867,873,883): no early return, all threads reach.
- common_kernels.cu shared_sum_for_layer_norm (919,926) called by layer_norm_kernel:
  THIS is the fixed site. The `if (n>=N) return` was folded into oobThread so all
  threads reach both barriers.
- common_kernels.cu promotion_logits_kernel (1168,1183): `if(threadInGroup<32)` guards
  work not barriers; all threads reach. No early return.
- fp16_kernels.cu SE_Layer_NHWC (89,106) and OutputInputTransformKernel_fp16_shmem_board
  (284,299,309): `if(c<K)`/`if(k<se_K)` guard work; all threads reach barriers. No early return.
- winograd OutputTransform_kernel (356,367,371) and OutputTransform_SE_relu_InputTransform_kernel
  (565,580,590): barriers inside `if(use_se)` which is a TEMPLATE param (block-uniform);
  `if(k<se_K)` guards work only. No divergent early return.
Every conditional `return;` in these files (winograd:728 `if(k>=C) return`, the various
`if(n>=N) return` in the barrier-free kernels, etc.) is in a kernel with NO __syncthreads
AFTER it, so an early-returning lane never abandons a barrier its wavefront-mates still run.
The ONLY barrier downstream of a per-row early return where the row granularity (32 lanes,
one threadIdx.z row = blockDim.x*blockDim.y with blockDim=(32,1,z)) is SMALLER than the AMD
wave (64) was layer_norm_kernel -- two z-rows share one wave64, so one row returning is a
HALF-wave divergence around S_BARRIER. On NVIDIA each z-row is a whole 32-lane warp, so it is
whole-warp granularity, matching the ISA's S_ENDPGM clause -> benign. That is exactly why it
is the only site that faults, and it answers the reviewer's S_BARRIER/S_ENDPGM quote: his text
covers WHOLE-wave early termination (all lanes gone -> S_ENDPGM -> dropped from the barrier
wait set); layer_norm's case is INTRA-wave lane divergence (the wave stays live, never executes
S_ENDPGM), which that clause does not cover.

### Validation (real gfx90a MI250X, ROCm 7.2.1, GPU 3, fork HEAD a80a7be)
Fresh clone + meson setup + `ninja` (330/330, benign nodiscard warnings only).
fp16_kernels.hip.o = 2.2MB, `nm -C` shows 28 non-empty SE_Layer_NHWC instantiations;
roc-obj-ls shows two gfx90a code objects. (Note: OpenBLAS was not preinstalled -- installed
libopenblas-dev and `meson setup --reconfigure` so the `blas` CPU reference backend registers.)
- CPU gtest `meson test`: 8/8 OK.
- maia-1100 conv-residual-SE net (2020, the T60-era SE-conv architecture the reviewer flagged
  for fp16 garbage) fp32 hip vs blas, atol=1e-3 rtol=1e-2: 222/222 Check passed, 0 error.
- maia-1100 fp16 hip-fp16 vs blas, atol=1.1e-1 rtol=2e-1: 222/222 passed, 0 error. Display mode
  batch 32: value abs 8.6e-05, policy abs 1.1e-03 -- sane fp16 output, NOT garbage. This is the
  reviewer's exact concern (old conv-SE net in fp16) and it passes.
- attention t1-256x10 net fp32: 130/130; fp16: 130/130.
- backendbench hip + hip-fp16 batch 1-256: clean, exit 0, no fault/crash.
gfx90a is thus fully re-validated on real GPU at a80a7be (state stays pr-open; advance-head
does not touch pr-open leads). linux-gfx1100 flipped to revalidate (functional change) for its
own host.

### Notes
- Could not fetch a distinct larger real T60 net without excessive effort (no reliable
  storage.lczero.org sha URL); maia-1100/1900 ARE the 2020 conv-residual-SE (T60-era)
  architecture, which is the fp16 path the reviewer worried about. Used maia-1100.
- Incremental-build gotcha still applies: after editing hip_compat.h you must
  `rm build-hip/*.hip.o` before ninja (the force-include is not tracked). This was a fresh
  build so N/A here.

## CUDA compile-check 2026-07-02 (linux-gfx90a, nvcc 12.6)

Verifies the fp16-gating refactor (PR review round: SKIP_FP16_BITS -> HAS_FP16_SUPPORT,
guard inversion, removal of the `#define __CUDA_ARCH__ 800` shim) did not regress the
CUDA build path. Fork HEAD a80a7be. nvcc 12.6 (CUDA 12.6) from /opt/conda/envs/cuda/bin/nvcc;
no NVIDIA GPU (compile-only).

### What was checked

The refactor touched fp16_kernels.cu, common_kernels.cu, and hip_compat.h:
- OLD: `#if __CUDA_ARCH__ < 530 -> #define SKIP_FP16_BITS 1`, guards `#ifndef SKIP_FP16_BITS`
- NEW: `#if __CUDA_ARCH__ >= 530 -> #define HAS_FP16_SUPPORT 1`, guards `#ifdef HAS_FP16_SUPPORT`
- `#define __CUDA_ARCH__ 800` shim REMOVED from hip_compat.h (shim is inside `#if defined(USE_HIP)`
  so NVIDIA never saw it anyway; no CUDA-path impact)

### Compile commands

```
NVCC=/opt/conda/envs/cuda/bin/nvcc
CUDA_INCDIR=/opt/conda/envs/cuda/targets/x86_64-linux/include
LC0_SRC=/var/lib/jenkins/moat/projects/lc0/src/src

# fp16-touched TU 1
bash utils/timeit.sh lc0 cuda-compile -- \
  $NVCC -arch=sm_70 -std=c++17 \
  -I"$LC0_SRC" -I"$CUDA_INCDIR" \
  -c "$LC0_SRC/neural/backends/cuda/fp16_kernels.cu" \
  -o /tmp/fp16_kernels.o

# fp16-touched TU 2
bash utils/timeit.sh lc0 cuda-compile -- \
  $NVCC -arch=sm_70 -std=c++17 \
  -I"$LC0_SRC" -I"$CUDA_INCDIR" \
  -c "$LC0_SRC/neural/backends/cuda/common_kernels.cu" \
  -o /tmp/common_kernels.o
```

Both exit 0, no errors or warnings from the refactored code.

### HAS_FP16_SUPPORT gate verification

```
# sm_70 (700 >= 530 -> HAS_FP16_SUPPORT defined): 11MB object,
#   28+ SE_Layer_NHWC<C,K> instantiations with real bodies (W weak symbols)
$NVCC -arch=sm_70 ... -c fp16_kernels.cu -o fp16_sm70.o
nm fp16_sm70.o | grep SE_Layer_NHWC   # shows W _ZN6lczero...SE_Layer_NHWC<C,K> symbols

# sm_50 (500 < 530 -> HAS_FP16_SUPPORT NOT defined): 598KB object,
#   only __device_stub__ launch wrappers (no kernel bodies)
$NVCC -arch=sm_50 ... -c fp16_kernels.cu -o fp16_sm50.o
nm fp16_sm50.o | grep SE_Layer_NHWC   # shows only __device_stub__ stubs, no W symbols
```

Size contrast: 11MB (sm_70, bodies compiled in) vs 598KB (sm_50, bodies absent) --
exactly what the truth-table analysis predicted.

### Verdict

CUDA path OK. The HAS_FP16_SUPPORT gate is logically identical to the old SKIP_FP16_BITS gate
for all host/device pass and arch combinations:
- CUDA host pass: `__CUDA_ARCH__` undefined (=0), 0 >= 530 false -> HAS_FP16_SUPPORT absent -> bodies skipped
- CUDA device >= 530 (e.g. sm_70): 700 >= 530 true -> HAS_FP16_SUPPORT defined -> bodies compiled in
- CUDA device < 530 (e.g. sm_50): 500 >= 530 false -> HAS_FP16_SUPPORT absent -> bodies skipped
- HIP: hip_compat.h defines HAS_FP16_SUPPORT 1 unconditionally (under USE_HIP) -- replaces the
  removed __CUDA_ARCH__ shim; NVIDIA build never includes hip_compat.h so is unaffected

The upstream PR's "CUDA preprocessed result is unchanged" claim is confirmed by actual nvcc compilation.

## Revalidation 2026-07-02 (linux-gfx1100) -- Binary equivalence carry-forward

HEAD moved from d83b6d1 (validated_sha) to a80a7be (PR review fix-round). Delta (one commit):
- meson.build: hipcc -std=c++17 -> -std=c++20; arch-autodetect silently-default -> error()
- README.md: SYCL bullet scoped to Intel only
- fp16_kernels.cu, common_kernels.cu, winograd_helper.inc, hip_compat.h: SKIP_FP16_BITS -> HAS_FP16_SUPPORT refactor (logically equivalent guard inversion)

Classifier verdict: class=mixed arch_independent=False inert=False.

Built a80a7be for gfx1100 into build-hip-new (using identical meson options with gfx1100), then ran codeobj_diff.py against the existing build-hip (d83b6d1, gfx1100, c++17):

```
python3 utils/codeobj_diff.py \
  projects/lc0/src/build-hip/lc0 \
  projects/lc0/src/build-hip-new/lc0
```

Result: verdict=identical (exported symbols + device ISA identical, 213 exports). The c++20 flag change and the fp16 guard rename produce the same GPU device code objects for gfx1100.

Carried forward validation to a80a7be without GPU re-run. Transition: revalidate -> completed.

## PR review round 2 2026-07-06 (porter, linux-gfx90a) -- co-build + barrier guard + CK triage

Reviewer Menkib64 approved PR #2420 and contributed real work. Three sub-tasks this round.
Fork HEAD a80a7be -> 223ee639 (5 new commits, none amending a80a7be).

### Item 1: compile HIP and CUDA backends together (thread PRRT_kwDOCBNonM6JUjs8)
Menkib64 solved this on his branch https://github.com/Menkib64/lc0/tree/hip_shared_backend
(4 commits built directly on our a80a7be). Cherry-picked all 4 onto moat-port preserving
HIS authorship (strongest credit; shows as author in git log and the PR):
- d0c4eab8 "Allow compiling hip and cuda together"
- f94a8a1a "Print HIP instead of CUDA when using hip backend"
- 03d8bff0 "Use HIP_VERSION for build version check"
- 72ef79f6 "Don't build pseudo libraries by default"

His approach (clean, incorporated unchanged):
- NS_BACKEND macro: cuda_common.h defines it `cudnn_backend`, hip_compat.h defines it
  `hip_backend`. Every shared TU (common_kernels.cu, fp16_kernels.cu, winograd_helper.inc,
  layers.cc, network_cuda.cc, cutlass_kernels.cu, kernels.h, inputs_outputs.h, layers.h)
  opens `namespace lczero { namespace NS_BACKEND {`, so the two backends' symbols are
  distinct and never collide in one binary.
- meson: instead of `add_project_arguments('-DUSE_HIP'...)` globally and dumping cuda/hip
  files into the shared `files`/`deps`/`includes`, each backend collects its own
  {files,deps,cxxargs,includes}; each is built as a `static_library` with its own cpp_args
  (isolating -DUSE_HIP / -DUSE_CUDNN / -DUSE_CUTLASS), then `extract_objects()` into the
  final executable. build_by_default:false so the pseudo libs are not built standalone.
- BACKEND_NAME / BACKEND_NAME_LC macros drive user-facing strings and the REGISTER_NETWORK
  names (cuda* vs hip*), replacing the `#if defined(USE_HIP)` registration block.
- network_cuda.cc wraps the CudaNetwork classes in an anonymous namespace.

Adaptation: none needed to his code; it applied and built clean. One observation for
upstream: `extract_objects()` on generated sources (the .o custom_targets) needs meson
>= 0.61.0; lc0's project() declares `>=0.60`, so meson prints a feature-version warning
(not an error) on 0.60. A follow-up could bump the min meson version; left as-is here.

Co-build PROVEN on this host (nvcc 12.6 + hipcc gfx90a into ONE binary):
```
meson setup build-cobuild-both -Dhip=true -Damd_gfx=gfx90a \
  -Dplain_cuda=true -Dnvcc=true -Dcc_cuda=70 -Dcudnn=false -Dcutlass=false \
  -Dcudnn_libdirs=/opt/conda/envs/cuda/lib \
  -Dcudnn_include=/opt/conda/envs/cuda/targets/x86_64-linux/include \
  -Dgtest=false -Dblas=true -Dopencl=false -Donnx=false -Db_lto=false \
  -Dnative_arch=false -Dhip_libdirs=/opt/rocm/lib -Dhip_include=/opt/rocm/include
ninja -C build-cobuild-both        # 270/270, clean link
build-cobuild-both/lc0 --help      # Backend VALUES: hip-auto,cuda-auto,hip,cuda,
                                   #   hip-fp16,cuda-fp16,blas,... -- BOTH registered
```
(cudnn_libdirs=/opt/conda/envs/cuda/lib both supplies cublas/cudart AND derives the nvcc
path via meson's fs.parent(libdir)+'/bin/nvcc', so no PATH edit is needed and ROCm tools
are not shadowed.) Single-backend HIP build (build-hip-cobuild) and the nvcc compile-check
of the shared TUs both stay green, so existing single-backend builds are preserved.

### Item 2: debug divergent-barrier guard (commit 223ee639; barrier thread suggestion)
winograd_helper.inc now defines `__device__ __forceinline__ void lc0SyncThreads()`; all 19
block-barrier sites in common_kernels.cu / fp16_kernels.cu / winograd_helper.inc call it
instead of __syncthreads(). In a HIP debug build (USE_HIP && !NDEBUG && device pass) it
computes the expected active lane mask from the block layout (linear tid, blockDim,
warpSize) and asserts `__ballot(1) == expected` before the real __syncthreads(). This
catches exactly the layer_norm failure class (intra-wave divergence: some lanes early-exit
before a block barrier) and does NOT false-positive on a legitimately partial final wave
(expected mask accounts for it). No-op wrapper on CUDA and in HIP release. Because the
hipcc custom_targets build with an explicit command line, meson.build now mirrors meson's
b_ndebug/buildtype logic to add -DNDEBUG in release so the device asserts are truly off.
fp16_kernels.cu gained `#include <cassert>`.

### Item 3: CUTLASS -> Composable Kernel fused-MHA (thread PRRT_kwDOCBNonM6JUnBY) -- SCOPED OUT
Genuine feasibility check on ROCm 7.2.1, not a hand-wave:
- ck_tile fmha headers ARE present (/opt/rocm/include/ck_tile/ops/fmha_fwd.hpp) and
  fmha_fwd_args exposes an elementwise bias_ptr with nhead/batch bias strides, so the API
  CAN express lc0's smolgen additive attention bias and arbitrary seqlen/hdim.
- BLOCKER: `float fmha_fwd(fmha_fwd_traits, fmha_fwd_args, const stream_config&)` is
  DECLARED-ONLY in the header; there is NO prebuilt fmha instance library in /opt/rocm/lib
  (checked -- none). CK fmha's definitions come from its example codegen (generate.py emits
  hundreds of per-shape instance .cpp, long compile) or you hand-instantiate a ck_tile fmha
  pipeline. Integrating either into lc0 means vendoring composable_kernel + its codegen (a
  large build/dependency addition, the CUDA CUTLASS path is a git subproject used only on
  NVIDIA) or a deep hand-written pipeline.
- lc0's attention is a fixed seqlen_q=seqlen_k=64 (8x8 board) with head_dim usually 32 or
  64 -- off CK fmha's tuned profile (LLM-scale seqlens; hdim 64/128/256, hdim=32 often
  unsupported/padded), so the fused kernel's HBM-traffic win over the correct-and-validated
  3-step cuBLAS fallback is small at this shape. Benefit is PERFORMANCE-ONLY; PR is approved.
Decision: keep the cuBLAS attention fallback (validated correct: 222/222 + 130/130 vs blas),
scope CK fmha out. Deferral already registered: deferred.py id `lc0-ck-fused-mha`.

## CK fmha perf measurement 2026-07-06 (gfx90a, ROCm 7.2.1)

Jeff requested measured numbers before finalizing the PR reply on the CK fused MHA question.
Shape: S=64 (8x8 board), head_dim={32,64}, fp16, nhead=8, smolgen additive bias (batch,nhead,S,S).
Platform: MI250X GPU 3 (HIP_VISIBLE_DEVICES=3), ROCm 7.2.1.

### Unfused hipBLAS 3-step baseline (lc0's actual path)

Standalone HIP C++ microbench (agent_space/fmha_bench/bench_unfused), hipBLAS GemmStridedBatched
fp16, warmup=200, iters=1000. GPU event timing, no Python overhead.

```
batch  hdim  unfused_us
1      32    25.9
8      32    26.2
32     32    46.0
128    32    157.7
256    32    302.6
1      64    25.9
8      64    25.6
32     64    42.7
128    64    155.2
256    64    302.2
```

### CK flash fmha (PyTorch flash_attn CK backend, WITHOUT additive bias)

PyTorch F.scaled_dot_product_attention with SDPBackend.FLASH_ATTENTION (no attn_mask).
GPU event timing. These are the BEST CASE numbers CK could achieve -- with bias they would be
slightly higher (one extra HBM read). See below for why bias blocks the CK path entirely.

```
batch  hdim  flash_nobias_us
1      32    48.4
8      32    50.3
32     32    48.9
128    32    67.5
256    32    129.0
1      64    49.2
8      64    50.2
32     64    50.6
128    64    92.5
256    64    179.4
```

### Critical finding: CK flash does NOT support additive bias

PyTorch ROCm's flash_attn (which drives the CK fmha instances in libck_sdpa.a) explicitly rejects
non-null attn_mask: warning "Flash Attention does not support non-null attn_mask." Falls through
to the math backend (fp32-upcast, 305-836 us -- MUCH worse than lc0's hipBLAS path). The low-level
ck_tile fmha API in /opt/rocm headers HAS a bias_ptr field, but using it requires the full CK
codegen (declared-only in headers, no prebuilt library) plus a custom wrapper for lc0. This is
exactly the "deep hand-written pipeline" path scoped out.

### Speed comparison (CK flash best case vs hipBLAS unfused, hdim=32)

| batch | hipBLAS_us | CK_flash_us (no bias) | ratio | verdict        |
|-------|------------|------------------------|-------|----------------|
|     1 |      25.9  |         48.4           | 0.54x | hipBLAS faster |
|     8 |      26.2  |         50.3           | 0.52x | hipBLAS faster |
|    32 |      46.0  |         48.9           | 0.94x | roughly equal  |
|   128 |     157.7  |         67.5           | 2.34x | CK would win   |
|   256 |     302.6  |        129.0           | 2.35x | CK would win   |

hdim=64 is similar: CK slower at batch<=8, equal at batch=32, ~1.7-1.7x faster at batch>=128.

### Verdict

The scope-out is confirmed with measured data:

1. CK flash does not support lc0's smolgen additive bias at all via the standard interface.
   Supporting it requires vendoring CK's codegen infrastructure -- a large dependency addition.

2. Even without the bias blocker: CK is 1.9x SLOWER than hipBLAS at batch=1-8 (dominant
   lc0 use case for chess engine inference -- the search tree generates many short batches).
   Equal at batch=32. Only faster at batch>=128, where it would give ~2.3x (hdim=32).

3. The ~2.3x win at large batches comes from 4 kernel launches -> 1 (HBM traffic matters
   less at S=64 than kernel launch overhead per head). This is a real but shape-limited win.

4. Conclusion: at lc0's S=64 with smolgen additive bias, CK fused MHA is architecturally
   blocked and, even hypothetically, is a net negative or neutral at the inference batch sizes
   most relevant to chess engine use. The cuBLAS/hipBLAS 3-step fallback remains the correct
   and appropriate path. Worth noting in the PR reply but does not change the scope decision.

### Validation (real gfx90a MI250X, ROCm 7.2.1, GPU 3, HEAD 223ee639, DEBUG build)
Because buildtype defaults to debug and the hipcc kernels are not compiled with -DNDEBUG,
the barrier guard is ACTIVE on-GPU in this build -- these runs also stress-test the guard,
which never false-fired.
- meson test (CPU gtest): 8/8 OK.
- maia-1100 conv-SE fp32 hip vs blas (atol=1e-3 rtol=1e-2): 222/222 Check passed, 0 ERROR.
- maia-1100 fp16 hip-fp16 vs blas (atol=1.1e-1 rtol=2e-1): 222/222 passed, 0 ERROR.
- attention testnet fp32 (atol=1e-3): 130/130. fp16 (atol=2.5e-2): 130/130, 0 ERROR.
- backendbench hip + hip-fp16 batch 1-256: exit 0, no fault/abort/assert.
- co-build binary registers hip+cuda+hip-fp16+cuda-fp16+hip-auto+cuda-auto.
- nvcc 12.6 compile-check of common_kernels.cu + fp16_kernels.cu (CUDA path, NS_BACKEND=
  cudnn_backend, barrier wrapper no-op): both exit 0. CUDA path preserved.

advance-head 223ee639 classified the change functional: linux-gfx1100 -> revalidate (its
own host), linux-gfx90a stays pr-open (validated here at 223ee639, PR update pending Jeff),
Windows stays blocked. gfx90a is re-validated on real GPU at the new head.

## Validation 2026-08-08 (validator, linux-gfx1100) -- VALIDATION-FAILED (documentation stale)

Platform: 4x AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), ROCm 7.2.1, hipcc clang 19,
meson 1.11.1, ninja. GPU 0 (all 4 free, `rocm-smi --showuse` 0% everywhere). Fresh clone of
`AMD-Ecosystem/lc0` @ moat-port, HEAD 223ee63914f3c7c1da020d63072e133523b2df91 (matches
`head_sha`). `libopenblas-dev` installed for the CPU `blas` reference backend (was not present
on this host). Fetched `maia1100.pb.gz` (CSSLab/maia-chess) and `testnet.pb.gz`
(t1-256x10-distilled-swa-2432500.pb.gz from storage.lczero.org) into `agent_space/` (gitignored
per-host, not present from a prior session on this host).

### Delta classification (a80a7be -> 223ee639, the last real-GPU pass recorded for gfx1100)

```
python3 utils/moatlib.py classify lc0 a80a7be 223ee639
```

Verdict: `class=mixed arch_independent=False`. Real functional/device-code changes for
this arch: the Menkib64 co-build cherry-picks wrap every shared TU in a `namespace
NS_BACKEND` (renames all mangled symbols: `mixed`/`rename-only` per file), and the new
`lc0SyncThreads()` debug-barrier-assert wrapper is a genuinely new function called at all
19 barrier sites (active in the default `debug` buildtype). Not a candidate for the
binary-equivalence carry-forward shortcut -- proceeded straight to a full real-GPU
revalidation rather than spending a build-twice-and-diff cycle on a delta already known
to change the compiled output.

### Build (gfx1100, debug buildtype -- so the new barrier-guard asserts are ACTIVE)

```
bash utils/timeit.sh lc0 compile -- \
  meson setup /var/lib/jenkins/moat/projects/lc0/src/build-hip /var/lib/jenkins/moat/projects/lc0/src \
  -Dhip=true -Damd_gfx=gfx1100 \
  -Dplain_cuda=false -Dcudnn=false -Dcutlass=false -Dnvcc=false \
  -Dgtest=true -Dblas=true -Dopencl=false -Donnx=false \
  -Db_lto=false -Dnative_arch=false \
  -Dhip_libdirs=/opt/rocm/lib -Dhip_include=/opt/rocm/include
bash utils/timeit.sh lc0 compile -- \
  ninja -C /var/lib/jenkins/moat/projects/lc0/src/build-hip -j16
```

Result: 331/331 targets, clean link, warnings only (benign nodiscard). `roc-obj-ls
build-hip/lc0`: two code objects, both `hipv4-amdgcn-amd-amdhsa--gfx1100` (1163256 and
2205056 bytes) -- no gfx90a anywhere. `nm -C fp16_kernels.hip.o` shows 28 non-empty
`SE_Layer_NHWC` instantiations (the fp16-gating refactor still compiles the conv-SE bodies
in on this arch).

### CPU gtest (non-GPU regression)

```
bash utils/timeit.sh lc0 test -- meson test -C /var/lib/jenkins/moat/projects/lc0/src/build-hip
```

Result: 8/8 OK (FP16, HashCat, PositionTest, OptionsParserTest, SyzygyTest,
EncodePositionForNN, EngineTest, ChessBoard). 0 failures. Matches every prior run.

### maia-1100 conv-SE cross-check (THE gate; also exercises all 19 barrier-guard sites)

```
HIP_VISIBLE_DEVICES=0 bash utils/timeit.sh lc0 test -- \
  build-hip/lc0 backendbench --backend=check \
  "--backend-opts=hip(),blas(),mode=check,atol=1e-3,rtol=1e-2,freq=1.0" \
  --weights=agent_space/maia1100.pb.gz --start-batch-size=1 --max-batch-size=55 --batches=4
```

fp32: 222/222 "Check passed", 0 ERROR (identical count to every prior gfx1100/gfx90a run).

```
HIP_VISIBLE_DEVICES=0 bash utils/timeit.sh lc0 test -- \
  build-hip/lc0 backendbench --backend=check \
  "--backend-opts=hip-fp16(),blas(),mode=check,atol=1.1e-1,rtol=2e-1,freq=1.0" \
  --weights=agent_space/maia1100.pb.gz --start-batch-size=1 --max-batch-size=55 --batches=4
```

fp16: 222/222 passed, 0 ERROR.

### Attention testnet regression (fp32 + fp16)

Same commands as prior sessions (see above), swapped to `testnet.pb.gz`, atol=1e-3/rtol=1e-2
(fp32) and atol=2.5e-2/rtol=1e-1 (fp16), batch 1-32. fp32: 130/130 passed, 0 ERROR. fp16:
130/130 passed, 0 ERROR. Both match every prior run exactly.

### Benchmark (fault-free, batch 1-256, debug build so the barrier-guard asserts are live)

```
HIP_VISIBLE_DEVICES=0 bash utils/timeit.sh lc0 test -- \
  build-hip/lc0 backendbench --backend=hip --weights=agent_space/maia1100.pb.gz --batches=3
HIP_VISIBLE_DEVICES=0 bash utils/timeit.sh lc0 test -- \
  build-hip/lc0 backendbench --backend=hip-fp16 --weights=agent_space/maia1100.pb.gz --batches=3
```

Both exit 0, batch 1-256, no crash/SIGABRT/hang. The `lc0SyncThreads()` debug assert (new
this round, checks the active-lane mask at all 19 barrier sites before every `__syncthreads`)
never fired across any of the above runs -- real stress evidence that the barrier-guard
addition is not itself introducing a wave32 regression on this arch.

### CUDA no-regression gate

Already recorded at this exact head_sha: see "PR review round 2 2026-07-06 (porter,
linux-gfx90a)" above -- "nvcc 12.6 compile-check of common_kernels.cu + fp16_kernels.cu
(CUDA path, NS_BACKEND=cudnn_backend, barrier wrapper no-op): both exit 0. CUDA path
preserved." Per validator.md this gate runs once per head_sha; skipped here.

### Jargon scrub

```
python3 utils/jargon.py --commits d8ce482..223ee639 -C projects/lc0/src
python3 utils/jargon.py --diff d8ce482...223ee639 -C projects/lc0/src
```

Both: `jargon: clean`.

### Documentation check -- FOUND A REAL STALENESS (this is why this is validation-failed, not completed)

README.md:165 (the "### HIP (ROCm)" section, added in the PR-prep round) reads: "The
target GPU architecture is taken from `-Damd_gfx` (e.g. `-Damd_gfx=gfx90a`); if it is
omitted it is autodetected with `rocm_agent_enumerator`, **defaulting to `gfx90a`**."

That claim is no longer true. The 2026-07-02 PR-review-round-2 commit (item 1, "arch
autodetect failure now error()s") REMOVED the silent gfx90a fallback: meson.build:645-661
now does `error('Could not autodetect an AMD GPU architecture. Set -Damd_gfx explicitly,
e.g. -Damd_gfx=gfx90a.')` when `rocm_agent_enumerator` is missing or finds nothing, and
`meson_options.txt:203` defaults the `amd_gfx` option to `''` (no default value anywhere).
README.md was never updated to match, so it now describes the OLD (silently-defaulting)
behavior that this fork specifically replaced with a hard, user-facing error. Confirmed by
reading meson.build:642-661 and meson_options.txt:201-204 directly -- not a hypothesis.

This is a real, previously-uncaught doc/code drift: the 2026-07-02/2026-07-06 rounds were
porter self-reports responding to the upstream PR reviewer's code-review comments and were
never run back through our own reviewer agent (no `## Review` entry exists between
2026-05-31 and this validation), so this is the first pass that actually checked the
documentation against the current meson.build. The recipe commands themselves are still
correct (they always pass `-Damd_gfx=` explicitly), so the build is reproducible; only the
prose description of the omitted-flag fallback path is wrong.

Per validator.md: "Neither is yours to fix quietly: send it back with validation-failed
and say which." Not editing README.md here. Fix needed (for the porter): reword README.md:165
to say autodetection failure now errors out and asks the user to set -Damd_gfx explicitly,
instead of claiming a silent gfx90a default.

### Summary

| Check | Result |
|-------|--------|
| Build (331/331 targets, gfx1100 code objects) | PASS |
| CPU gtest 8/8 | PASS |
| maia-1100 fp32 conv-SE check (222 batches) | PASS |
| maia-1100 fp16 conv-SE check (222 batches) | PASS |
| attention testnet fp32 check (130 batches) | PASS |
| attention testnet fp16 check (130 batches) | PASS |
| backendbench fp32 + fp16 batch 1-256, barrier-guard live | PASS (no fault, no assert) |
| CUDA no-regression gate | already recorded at 223ee639 (gfx90a porter session) |
| jargon scrub (commits + diff, base d8ce482) | clean |
| ROCm build documentation | STALE (README.md:165 fallback-arch claim) |

All GPU technical checks pass cleanly on real gfx1100 hardware, matching every prior
session's magnitudes exactly. The sole reason this is not `completed` is the README
staleness above. Transition: `review-passed -> validation-failed` (project stage, not a
per-arch fact -- every arch's existing `completed` record is left untouched; a doc-only
porter fix should classify as arch-independent and auto-carry-forward every already-passed
arch with no GPU rerun needed).
