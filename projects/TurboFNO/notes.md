# TurboFNO notes

## Port summary (linux-gfx90a lead)
- Strategy A: one `utils/cuda_to_hip.h` compat shim + a `USE_HIP` CMake path.
  No AMD-native rewrite; the GEMM and FFT are hand-written portable HIP/CUDA C++.
- Fork: https://github.com/AMD-Ecosystem/TurboFNO branch `moat-port`.
- Submodule TurboFFT: upstream https://github.com/shixun404/TurboFFT pinned at
  sha e28570417284b8e66c124880e8805b677075076b. NO source edits needed
  in the submodule -- its generated float2 headers are pure device code (no CUDA
  includes, no shuffles); the `TurboFFT/Common` NVIDIA CUDA-samples bundle is on
  the include path but no built TU references it. `.gitmodules` URL repointed to
  the jeffdaily fork for reproducibility.

## What changed
- NEW `utils/cuda_to_hip.h`: on ROCm includes hip_runtime + hipfft + hipblas and
  aliases the exact symbol set used (cudaMalloc/Memcpy/DeviceSynchronize/Event*,
  cudaDeviceProp, cufftCreate/Plan1d/PlanMany/ExecC2C/Destroy, CUFFT_C2C/FORWARD,
  cublasCreate/Cgemm/Destroy, CUBLAS_OP_N, cuFloatComplex). On NVIDIA: plain CUDA
  includes. cuFloatComplex -> hipFloatComplex (both float2), so the `(cuFloatComplex*)`
  casts at the hipblas/hipfft call sites are correct with no buffer change.
- `utils/utils.cuh`: include the shim; DROP `helper_functions.h`/`helper_cuda.h`
  (NVIDIA CUDA-samples, not on ROCm, unused).
- All built `.cu`: CUDA includes routed through the shim. All kernel `.cuh`:
  dead `#include <mma.h>` guarded out under HIP.
- NEW `cmake/turbofno_targets.cmake`: `turbofno_configure_target()` selects the
  CUDA-vs-HIP toolchain. All 10 `fusion_variants/*/CMakeLists.txt` rewritten to a
  uniform shape: `option(USE_HIP)`, `project(... LANGUAGES CXX HIP|CUDA)`,
  `set_source_files_properties(... LANGUAGE HIP)`, link `hip::hipfft roc::hipblas`
  (HIP) or `CUDA::cublas CUDA::cufft` (CUDA), `-ffp-contract=on` on HIP.
  `CMAKE_HIP_ARCHITECTURES` left to the caller (no hardcoded gfx / warp width).
- `install.sh`: `USE_HIP=1` switch (optional `CMAKE_HIP_ARCHITECTURES`).

## Build (gfx90a)
```
cd projects/TurboFNO/src && export PROJECT_ROOT=$(pwd)
USE_HIP=1 ./install.sh                 # all 10 variants
# or per variant:
cmake -S fusion_variants/1D_D_exp_fused_fft_cgemm_ifft -B build \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ && cmake --build build -j16
```
Followers: `-DCMAKE_HIP_ARCHITECTURES=gfx1100` (or gfx1151). A
`gfx90a;gfx1100` fat binary emits both code objects (`llvm-objdump --offloading`).
All 10 variants configure + build clean on gfx90a, ROCm 7.2.1.

## Validation method + results (gfx90a, wavefront 64, ROCm 7.2.1)
The shipped `main()` in every variant is TIMING-ONLY (verify_vector/verify_matrix
exist in utils.cu but call sites are commented out). Added a standalone harness
(agent_space/turbofno_validate.cu) that exercises the two hand-written device
primitives the fused kernels compose, against an independent ROCm-library gold:
- cgemm (the logical-32 register-tiled complex GEMM) vs hipblasCgemm:
  max_rel_diff ~9.3e-6, 0 outliers -> PASS.
- fft_8 (hand-rolled radix-2 256-point FFT) vs hipfftExecC2C FORWARD, batch 1024:
  max_rel_diff ~1.4e-5, 0 outliers -> PASS.
Runtime smoke: TurboFNO_1D_D (fused) and TurboFNO_1D_E (hipFFT/hipBLAS baseline)
both run a trimmed config sweep to completion with no CHECK_CUDA_KERNEL errors.

Note on the shipped full default config sweep: bs_list goes up to 32768 while the
buffers are sized to the initial bs=128 and reused -- the upstream main reads OOB
for the large bs entries (a pre-existing upstream issue, not a port concern). Use
a small config for runtime smoke.

## Logical-32 GEMM tiling on wave64: CORRECT, unchanged
The only wave-coupled idiom is the GEMM register tiling (utils/TurboFNO.h:
WARP_M 32, WID=threadIdx.x/32; cgemm/fused index with TID%32). It is a logical
32-lane subgroup fenced by __syncthreads, NOT a warp-collective op. There are no
__shfl/__ballot/warpSize/__syncwarp/__activemask in the COMPILED surface (the
shuffle-using code is only in TurboFFT's fft_codegen.py generator script, which
emits a non-float2 path that this build does not include). The GEMM result
matches hipBLAS to ~1e-5 on wave64 -> the tiling is correct as-is. NO wave64
hardcode introduced; the device width is never baked into a build constant.

## Fault classes hit
- Library swap: cuFFT->hipFFT, cuBLAS->hipBLAS (hipBLAS v3 here: hipblasCgemm
  takes `hipComplex*` == float2; hip::hipfft + roc::hipblas CMake targets).
- Dead-header removal: `<mma.h>` (no tensor-core code) and the two CUDA-samples
  headers, both unresolvable / unused on ROCm.
- fp fast-math drift handled by pinning `-ffp-contract=on`; tolerances are
  relative and fast-math aware.
- NOT triggered: textures/surfaces, rule-of-five, OOB neighbor reads, 256B pitch,
  __smid, layered arrays, managed memory, streams (default only).

## Follower delta (gfx1100 / gfx1151)
Reuse this branch, rebuild with the target arch. Expected no source change: the
only wave dependency is the logical-32 tiling, which is wave-agnostic; RDNA is
wave32 so each logical warp is exactly one wavefront (the safer width). Validate
by the same cgemm-vs-hipBLAS and fft_8-vs-hipFFT harness on the follower host,
and/or a cross-arch output diff against the gfx90a result (deterministic).
gfx1151: confirm hipFFT/hipBLAS presence in the Windows HIP SDK.

## Validation 2026-06-20 (windows-gfx1101, RDNA3 wave32)

GPU: AMD Radeon PRO V710, gfx1101, warpSize=32, 27 CUs. ROCm 7.14.0a20260604 (TheRock nightly).
Fork: AMD-Ecosystem/TurboFNO moat-port @ e100b3d (submodule shixun404/TurboFFT @ e285704).

Build: all 10 fusion variants configured and compiled clean for gfx1101 via Ninja + all-clang
(clang++ 23.0.0 from _rocm_sdk_devel). Warnings only, no errors. Build dir per variant:
`build_win_gfx1101/`.

Commands:
```
# Build all 10 variants + numerical harness for gfx1101
python3 agent_space/turbofno_build_win_gfx1101.py --arch gfx1101 --jobs 64

# Run numerical harness (DLLs already in agent_space/ from prior runs)
# HIP_VISIBLE_DEVICES=1 = gfx1101 (V710) on this host; gfx1201 = device 0
cd agent_space && HIP_VISIBLE_DEVICES=1 ./turbofno_validate_win_gfx1101.exe
# PATH must include: _rocm_sdk_core/bin, _rocm_sdk_devel/bin, _rocm_sdk_libraries/bin
# ROCBLAS_TENSILE_LIBPATH=_rocm_sdk_libraries/bin/rocblas/library

# Runtime smoke (both ran to completion, RC=0)
# TurboFNO_1D_D.exe (fused FFT-GEMM-iFFT): fusion_variants/1D_D_.../build_win_gfx1101/
# TurboFNO_1D_E.exe (hipFFT+hipBLAS baseline): fusion_variants/1D_E_baseline/build_win_gfx1101/
```

Numerical results (agent_space/turbofno_validate_win_gfx1101.exe on HIP_VISIBLE_DEVICES=1):
- Device: AMD Radeon PRO V710, warpSize=32, CUs=27
- GEMM: cgemm (logical-32 tiling, wave32) vs hipblasCgemm M=256 N=256 K=128 (col-maj):
  outlier_cnt=0, outlier_perct=0.000000%, max_rel_diff=3.958292e-05 -> PASS
- FFT: fft_8 (hand-rolled radix-2 256pt) vs hipfftExecC2C FORWARD, batch=1024:
  outlier_cnt=0, outlier_perct=0.000000%, max_rel_diff=3.576712e-04 -> PASS

Runtime smoke (RC=0, no CHECK_CUDA_KERNEL errors):
- TurboFNO_1D_D (fused FFT-GEMM-iFFT): ran to completion, RC=0.
- TurboFNO_1D_E (hipFFT+hipBLAS baseline): ran to completion, RC=0.

Wave32 note: gfx1101 is wave32 identical to gfx1100 and gfx1201. The cgemm logical-32
tiling (WID=threadIdx.x/32, __syncthreads fences) maps one-to-one to a single wavefront
on RDNA3, same as gfx1100. max_rel_diff for GEMM (3.96e-5) and FFT (3.58e-4) match
gfx1201 exactly (identical FP behavior on RDNA3/4 for these ops).

Result: windows-gfx1101 COMPLETED at e100b3d.

## Validation 2026-06-07 (windows-gfx1201, RDNA4 wave32)

GPU: AMD Radeon RX 9070 XT, gfx1201, warpSize=32, 32 CUs. ROCm 7.14.0a20260604 (TheRock nightly).
Fork: AMD-Ecosystem/TurboFNO moat-port @ e100b3d (submodule shixun404/TurboFFT @ e285704).

Build: all 10 fusion variants configured and compiled clean for gfx1201 via Ninja + all-clang
(clang++ 23.0.0 from _rocm_sdk_devel). Warnings only, no errors.

Commands:
```
# Set up environment
export PROJECT_ROOT=B:\develop\moat\projects\TurboFNO\src
export HIP_VISIBLE_DEVICES=0   # gfx1201 is device 0 (V710 offline this session)
ROCM=B:\develop\TheRock\external-builds\pytorch\.venv\Lib\site-packages\_rocm_sdk_devel
CLANG=$ROCM\lib\llvm\bin\clang++.exe

# Build all 10 variants (Ninja, -j32 shared with another concurrent build)
# python3 agent_space/turbofno_build_win_gfx1201.py --arch gfx1201 --jobs 32

# Build numerical harness directly
$CLANG -x hip --offload-arch=gfx1201 -DUSE_HIP -std=c++17 -O2 -ffp-contract=on \
    -I$PROJECT_ROOT/utils \
    -I$PROJECT_ROOT/fusion_variants/1D_A_exp_fft+cgemm+ifft \
    -I$PROJECT_ROOT/TurboFFT/TurboFFT/include/code_gen/generated/float2 \
    -I$ROCM/include \
    agent_space/turbofno_validate_win.cu \
    $PROJECT_ROOT/utils/utils.cu \
    -L$ROCM/lib -lhipfft -lhipblas \
    -o agent_space/turbofno_validate_win.exe

# Run (DLLs from _rocm_sdk_core/bin and _rocm_sdk_devel/bin on PATH;
#      ROCBLAS_TENSILE_LIBPATH for rocBLAS Tensile kernels)
# HIP_VISIBLE_DEVICES=0 agent_space/turbofno_validate_win.exe
```

Windows-specific note: harness uses utils.cuh include before the generated FFT headers
(turboFFT_ZADD/ZSUB/ZMUL macros must be visible when fft_radix_2_logN_8_upload_0.cuh is
included). Harness compilation is agent_space/turbofno_validate_win.cu.
Runtime: DLLs (amdhip64_7.dll, hipfft.dll, hipblas.dll, rocfft.dll) copied next to exe
or on PATH from _rocm_sdk_core/bin + _rocm_sdk_devel/bin. ROCBLAS_TENSILE_LIBPATH=
_rocm_sdk_libraries/bin/rocblas/library. The rocblaslt stderr errors (cannot read
TensileLibrary_lazy_gfx1201.dat) are non-fatal; they do not affect hipblasCgemm path.

Numerical results (agent_space/turbofno_validate_win.cu on GPU 0):
- Device: AMD Radeon RX 9070 XT, warpSize=32, CUs=32
- GEMM: cgemm (logical-32 tiling, wave32) vs hipblasCgemm M=256 N=256 K=128 (col-maj):
  outlier_cnt=0, outlier_perct=0.000000%, max_rel_diff=3.958292e-05 -> PASS
- FFT: fft_8 (hand-rolled radix-2 256pt) vs hipfftExecC2C FORWARD, batch=1024:
  outlier_cnt=0, outlier_perct=0.000000%, max_rel_diff=3.576712e-04 -> PASS

Runtime smoke (RC=0, no CHECK_CUDA_KERNEL errors):
- TurboFNO_1D_D (fused FFT-GEMM-iFFT): ran to completion, RC=0.
- TurboFNO_1D_E (hipFFT+hipBLAS baseline): ran to completion, RC=0.

Result: windows-gfx1201 COMPLETED at e100b3d.

## Validation 2026-06-04 (linux-gfx1100, RDNA3 wave32)

GPU: AMD Radeon Pro W7800 48GB, gfx1100, warpSize=32, 35 CUs. ROCm 7.2.1.
Fork: AMD-Ecosystem/TurboFNO moat-port @ e100b3d (submodule shixun404/TurboFFT @ e285704).

Note on host GPU accessibility: this host has 4 W7800 GPUs (gfx1100). GPUs 0, 1, 3
were clock-gated at 0 MHz and the ROCm 7.2.1 runtime hung during queue creation on
them (acquireQueue spin-wait; a known ROCm/driver issue with deep GFXOFF on RDNA3
when orphaned KFD contexts are present). GPU 2 was clocked up (2948 MHz, active
from another session) and was responsive. All validation ran on GPU 2 (HIP_VISIBLE_DEVICES=2).

Build: all 10 fusion variants configured and compiled clean for gfx1100 (~45s, warnings only, no errors).
gfx1100 code objects confirmed via llvm-objdump --offloading for 1D_A, 1D_D, 1D_E, 2D_D, 2D_E
(all show `hipv4-amdgcn-amd-amdhsa--gfx1100`).

Commands:
```
# Build all 10 variants
cd projects/TurboFNO/src && export PROJECT_ROOT=$(pwd)
USE_HIP=1 CMAKE_HIP_ARCHITECTURES=gfx1100 bash install.sh

# Build + run numerical harness (column-major matrices; cgemm.cuh is col-major throughout)
/opt/rocm/llvm/bin/clang++ -x hip --offload-arch=gfx1100 -DUSE_HIP -std=c++17 -O2 -ffp-contract=on \
    -I${PROJECT_ROOT}/utils -I${PROJECT_ROOT}/fusion_variants/1D_A_exp_fft+cgemm+ifft \
    -I${PROJECT_ROOT}/TurboFFT/TurboFFT/include/code_gen/generated/float2 \
    ${PROJECT_ROOT}/utils/utils.cu agent_space/turbofno_validate.cu \
    -L/opt/rocm/lib -lhipfft -lhipblas -o /tmp/turbofno_validate_gfx1100
HIP_VISIBLE_DEVICES=2 /tmp/turbofno_validate_gfx1100

# Runtime smoke
HIP_VISIBLE_DEVICES=2 fusion_variants/1D_D_exp_fused_fft_cgemm_ifft/build/TurboFNO_1D_D | head -30
HIP_VISIBLE_DEVICES=2 fusion_variants/1D_E_baseline/build/TurboFNO_1D_E | head -30
```

Numerical results (agent_space/turbofno_validate.cu on GPU 2):
- Device: AMD Radeon Pro W7800 48GB, warpSize=32, CUs=35
- GEMM: cgemm (logical-32 tiling, wave32) vs hipblasCgemm M=256 N=256 K=128 (col-maj):
  outlier_cnt=0, outlier_perct=0.000000%, max_rel_diff=3.487317e-05 -> PASS
- FFT: fft_8 (hand-rolled radix-2 256pt) vs hipfftExecC2C FORWARD, batch=1024:
  outlier_cnt=0, outlier_perct=0.000000%, max_rel_diff=8.255286e-05 -> PASS

Wave32 GEMM tiling confirmation: the cgemm kernel uses WID=threadIdx.x/32, WARP_M=32,
and __syncthreads for synchronization -- a block-level fence that is wave-size-agnostic.
On wave32 each logical tile (32 threads) is exactly one wavefront. No warp-collective
ops (no __shfl/__ballot/__syncwarp). The numerical PASS (max_rel_diff ~3.5e-5, 0 outliers)
on wave32 confirms the tiling is correct as-is.

Runtime smoke (first 30 lines, no CHECK_CUDA_KERNEL errors):
- TurboFNO_1D_D (fused FFT-GEMM-iFFT): bs=1, dimX=1, DY=128, N=64..128, K=8..112 timing lines printed cleanly.
- TurboFNO_1D_E (hipFFT+hipBLAS baseline): bs=1, dimX=1, DY=128, N=64..128, K=8..112 timing lines printed cleanly.

Result: linux-gfx1100 COMPLETED at e100b3d.

## Validation 2026-06-04 (linux-gfx90a)

GPU: AMD Instinct MI250X / MI250, gfx90a, warpSize=64, 104 SMs. ROCm 7.2.1.
Fork: AMD-Ecosystem/TurboFNO moat-port @ e100b3d (submodule shixun404/TurboFFT @ e285704).

Build: all 10 fusion variants configured and compiled clean for gfx90a (warnings only, no errors).
gfx90a code objects confirmed via llvm-objdump --offloading for 1D_A, 1D_D, 1D_E, 2D_D, 2D_E.

Commands:
```
# Build all 10 variants
cd projects/TurboFNO/src && export PROJECT_ROOT=$(pwd) && USE_HIP=1 CMAKE_HIP_ARCHITECTURES=gfx90a bash install.sh

# Build + run numerical harness
/opt/rocm/llvm/bin/clang++ -x hip --offload-arch=gfx90a -DUSE_HIP -std=c++17 -O2 -ffp-contract=on \
    -I${PROJECT_ROOT}/utils -I${PROJECT_ROOT}/fusion_variants/1D_A_exp_fft+cgemm+ifft \
    -I${PROJECT_ROOT}/TurboFFT/TurboFFT/include/code_gen/generated/float2 \
    ${PROJECT_ROOT}/utils/utils.cu agent_space/turbofno_validate.cu \
    -L/opt/rocm/lib -lhipfft -lhipblas -o /tmp/turbofno_validate
HIP_VISIBLE_DEVICES=0 /tmp/turbofno_validate

# Runtime smoke
HIP_VISIBLE_DEVICES=0 fusion_variants/1D_D_exp_fused_fft_cgemm_ifft/build/TurboFNO_1D_D | head -30
HIP_VISIBLE_DEVICES=0 fusion_variants/1D_E_baseline/build/TurboFNO_1D_E | head -30
```

Numerical results (agent_space/turbofno_validate.cu on GPU 0):
- Device: AMD Instinct MI250X / MI250, warpSize=64, SMs=104
- GEMM: cgemm (logical-32 tiling, wave64) vs hipblasCgemm, M=256 N=256 K=128:
  outlier_cnt=0, outlier_perct=0.000000%, max_rel_diff=9.313226e-06 -> PASS
- FFT: fft_8 (hand-rolled radix-2 256pt) vs hipfftExecC2C FORWARD, batch=1024:
  outlier_cnt=0, outlier_perct=0.000000%, max_rel_diff=1.415610e-05 -> PASS

Runtime smoke (first 30 lines, no CHECK_CUDA_KERNEL errors):
- TurboFNO_1D_D (fused FFT-GEMM-iFFT): bs=1..128 timing lines printed cleanly.
- TurboFNO_1D_E (hipFFT+hipBLAS baseline): bs=1..2 timing lines printed cleanly.
  (Large bs values, e.g. 32768, trigger the pre-existing upstream OOB allocation;
  not a port regression -- documented in notes above.)

Result: linux-gfx90a COMPLETED at e100b3d.

## Review 2026-06-03 (reviewer, linux-gfx90a)
Verdict: review-passed. Diff upstream c83a74b..e100b3d on AMD-Ecosystem/TurboFNO
moat-port. No blocking findings. Confirmations and minor (non-blocking) items below.

Confirmed sound:
- Logical-32 GEMM tiling on wave64 is CORRECT and unchanged (the key item).
  cgemm.cuh:58-114 reads operands only from __shared__ (sA/sB) indexed by WID/
  WARP_M/(TID%32), shared via __syncthreads (cgemm.cuh:58), with explicit
  per-thread float2 FMA (cgemm.cuh:102-103). No __shfl/__ballot/warpSize/
  __syncwarp anywhere in the compiled surface (repo + the actually-included
  float2 generated FFT headers). So WID=threadIdx.x/32 / WARP_M 32 (TurboFNO.h:
  5,13) is a logical 32-lane subgroup, not a warp-collective; on wave64 a
  wavefront holds two such tiles and correctness is width-independent. No wave64
  hardcode introduced. Followers (wave32) need only a cross-arch output diff.
- hipblasCgemm signature: this host's hipBLAS takes hipComplex* (hipblas.h:14694),
  and hipComplex==hipFloatComplex==float2 (amd_hip_complex.h:46,135). The shim
  alias cuFloatComplex->hipFloatComplex makes the (cuFloatComplex*) call-site
  casts type-correct. Validated numerically (~9.3e-6 vs hipblasCgemm).
- cmake targets hip::hipfft and roc::hipblas both exist and match the ROCm config
  packages (/opt/rocm/lib/cmake/{hipfft,hipblas}); CMAKE_HIP_ARCHITECTURES left to
  the caller (turbofno_targets.cmake:12-14), no hardcoded gfx/warp width; CUDA
  path preserved under else() with CUDA::cublas CUDA::cufft.
- Dead headers: <mma.h> guarded out on HIP in all kernel .cuh that had it; the two
  CUDA-samples headers dropped from utils.cuh; nothing built references them.
- Commit message: [ROCm] prefix, <=72 char title, Claude disclosure, Test Plan,
  no noreply trailer, no MOAT jargon, ASCII-only. CUDA build path intact.
- Submodule TurboFFT clean at e285704, no source edits, .gitmodules repointed.

Minor (non-blocking) cleanup, optional before upstream PR:
- The 10 .cu drivers each carry three redundant `#include "cuda_to_hip.h"` lines
  (e.g. 1D_A/fused.cu:2-3 and :20) -- the porter replaced three distinct CUDA
  headers with the same shim include rather than collapsing to one. Harmless
  (#pragma once) but untidy; collapse to a single include.
- cuda_to_hip.h aliases two symbols never used anywhere: cublasDestroy (line 64)
  and CUFFT_INVERSE (line 51). Plan said "alias the exact set used"; drop them or
  leave -- harmless dead #defines, no effect on the CUDA path.
- Pre-existing upstream (NOT port concerns, do not fix here): E baseline calls
  cublasCreate without a matching cublasDestroy (handle leak); the iFFT exec uses
  CUFFT_FORWARD on iplan (fused.cu:190,212); the default config sweep reads OOB
  for large bs (notes already record this).

## 2026-06-24 -- nvcc CUDA-path build check (pre-PR, after license cleared)
The port had never been compiled with nvcc (HIP builds + numerical validation only). Verified the CUDA/NVIDIA path is intact before opening the upstream PR: nvcc 12.6 (conda popsift-cuda126, host gcc 12.4), CMAKE_CUDA_ARCHITECTURES=86, USE_HIP=OFF. All 10 fusion variants (1D_A..E, 2D_A..E) configure, compile, AND LINK clean against CUDA::cublas/CUDA::cufft -- PASS=10 FAIL=0, reaching the link stage (catches the undefined-reference class, not just compile). Confirms the cuda_to_hip.h NVIDIA #else path, the include-routing through the shim, the dropped CUDA-samples headers, and the turbofno_targets.cmake CUDA branch leave the NVIDIA build working. Throwaway build dirs (fusion_variants/*/build-cuda-nvcccheck) removed.

## 2026-06-24 -- shim-header refactor (smaller PR diff) + arch auto-detect

Reworked the port at jeff's request to shrink the upstream diff. OLD approach (9ae7066): edited 38 kernel/driver source files to add `#include "cuda_to_hip.h"` (some 3x) and `#if`-guard out `<mma.h>`. NEW approach (ec49bcf): the shim-HEADER method -- a new `hip_compat/` dir holds shadow headers (cuda_runtime.h, cublas_v2.h, cufftXt.h, cufft.h, mma.h, helper_functions.h, helper_cuda.h) added to the include path on the HIP build ONLY (turbofno_targets.cmake `target_include_directories BEFORE`). The runtime/blas/fft shadows pull in HIP + alias the cuda*/cufft*/cublas* spellings; mma.h + the two CUDA-samples headers are empty stubs (no wmma/mma_sync or helper code is compiled). Result: ALL 38 source files reverted to upstream byte-identical; `utils/cuda_to_hip.h` removed.

Diff: 53 files (40 source edits) -> 21 files, 0 source (.cu/.cuh) edits. Remaining diff = hip_compat/ (8 stubs) + cmake/turbofno_targets.cmake + 10 variant CMakeLists (USE_HIP switch; the per-variant compile-options/link block moved into the shared cmake; min raised 3.18->3.21 for first-class HIP language) + install.sh + README.

Arch: removed the install.sh `CMAKE_HIP_ARCHITECTURES:-gfx90a` default; `enable_language(HIP)` now auto-detects the host GPU (confirmed: cache shows gfx90a auto-detected with no pin). Overridable via CMAKE_HIP_ARCHITECTURES env. README updated.

Validation of the refactor (behavior-preserving proof):
- linux-gfx90a: REAL GPU numerical harness at ec49bcf -- GEMM max_rel 9.313226e-06, FFT 1.415610e-05, 0 outliers, BIT-IDENTICAL to the original 9ae7066/e100b3d run. PLUS all 10 variant device objects byte-identical (codeobj_diff old vs new) on gfx90a. -> carried forward (binary-equiv).
- linux-gfx1100: all 10 variant device objects + exported symbols byte-identical old(9ae7066) vs new(ec49bcf) via codeobj_diff (built both here, ROCm 7.2.1). -> carried forward (binary-equiv).
- CUDA path: all 10 variants nvcc 12.6 (USE_HIP=OFF, sm_86) configure+compile+LINK clean with the reverted (upstream-identical) sources; hip_compat does NOT leak onto the CUDA include path.
- windows-gfx1201 / windows-gfx1101: flipped to revalidate by the source refactor (cannot build the Windows ROCm SDK here). Device code is provably identical on both Linux arches (wave64 + wave32), so a quick binary-equiv or numerical re-run on the gfx1201 workstation will carry the Windows tier. PR-ready blocks on ONE Windows arch until then.

State: head_sha 9ae7066 -> ec49bcf. gfx90a, gfx1100 completed at ec49bcf. gfx1201/gfx1101 revalidate. gfx1151 port-ready (optional).

## Revalidation 2026-06-24 (windows-gfx1201, binary-equiv)

Revalidate trigger: head_sha moved 9ae7066 -> ec49bcf (shim-header refactor). validated_sha was 9ae70665
(carried from e100b3d via comment-only). The old 9ae70665 SHA is no longer in git history (force-push);
used e100b3d (the real GPU-validated build, build_win/) as the "old" reference.

Method: built ec49bcf5 for gfx1201 into build_win_ec49/ dirs (all 10 variants, clang++ 23.0.0 from
_rocm_sdk_devel, -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201). All 10 compiled clean (warnings only).

Compared device ISA via:
1. llvm-objcopy --dump-section=.hip_fatbin from COFF .cu.obj
2. clang-offload-bundler --unbundle --type=bc --targets=hipv4-amdgcn-amd-amdhsa--gfx1201
3. llvm-objcopy --dump-section=.text from extracted AMD GPU ELF
4. sha256 compare

All 10 kernel object .text sections IDENTICAL (byte-for-byte):
- 1D_A fused.cu.obj: 5c0e56acd45f9c24 (43904 B)
- 1D_B fused.cu.obj: ecd31dc1a355606b (91136 B)
- 1D_C fused.cu.obj: 5b50fcc92b0203bf (181376 B)
- 1D_D fused.cu.obj: 84f9e5d6f4c372ed (188800 B)
- 1D_E fused.cu.obj: b8cc76ab9ea88bbf (1408 B)
- 2D_A fused_trunc_2D.cu.obj: 04a7c05f16f84d32 (89984 B)
- 2D_B fused_trunc_2D.cu.obj: ca4687aec3f8af4c (138240 B)
- 2D_C fused_trunc_2D.cu.obj: 60bfa1a786bd54d8 (228352 B)
- 2D_D fused_trunc_2D.cu.obj: 0d666f32e638eb04 (235520 B)
- 2D_E fused_trunc_2D.cu.obj: c670a04600f9ce61 (1920 B)

Exported kernel symbols identical; only __hip_cuid_ differs (build-path hash, not functional).
utils.cu.obj has no .hip_fatbin section (no device kernels, as expected).

Verdict: binary-equiv confirmed on gfx1201. windows-gfx1201 carried forward to ec49bcf5.

## Fix round 2026-08-24 (linux-gfx1100, porter): sample-derived error macros replaced

Dispatch: deferred.json item `turbofno-nvidia-proprietary-rescan`. Upstream's own
`utils/utils.cuh` embedded two host-side error-check macros copied from the NVIDIA
CUDA library samples, under the samples' notice comment block. They compile on both
the CUDA and the HIP path, so the port carried them. Required outcome: the file
carries no notice block and no sample-derived macro text, with the macro NAMES
kept so no call site moves. Commit 03141cf on top of ec49bcf (fast-forward push,
nothing rewritten -- ec49bcf was validated by all four platforms).

### What the file actually contained (both copies)

The copied block appeared TWICE in the 267-line header:
- lines 42-78: an edited copy (upstream added `fflush(stdout)` and, in CUFFT_CALL,
  `return 1`). Because it comes first, this is the copy that actually expands.
- lines 156-203: the notice comment block, then lines 205-240 a second, verbatim
  copy of the same two macros, dead because the `#ifndef` guards are already
  satisfied by the first copy, then lines 242-257 a commented-out
  `traits<CUFFT_C2C>` template from the same sample file.
Removing only the notice-covered block would have left the derived macro text that
actually compiles. Both copies had to go.

### What changed

- `CUDA_RT_CALL` / `CUFFT_CALL` rewritten independently. Independence bar (plvs
  2026-08-13 precedent): own control-flow shape (`do { ... } while (0)`, matching
  the file's own `CUDA_CALLER` idiom, not the sample's bare `{ ... }` block), own
  internal naming (`turbofno_rt_status` / `turbofno_fft_status`, not `status`, and
  a direct declaration rather than the sample's `static_cast<cudaError_t>(call)`),
  and an own message format (`"%s:%d: %s -> %s (%d)"`, compiler-style
  file:line first) rather than a paraphrase of the sample's
  `"ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d)."`.
- Observable behaviour deliberately preserved so the round stays a text
  replacement: CUDA_RT_CALL reports and CONTINUES (it never aborted upstream, and
  making it abort would turn the known large-`bs` upstream OOB into a crash);
  CUFFT_CALL reports and `return 1`s from the enclosing function. Both silent on
  success.
- Notice block, dead duplicate macros, and the commented-out traits template
  deleted together as the code the notice covered. No standalone licence or
  NOTICE file was touched anywhere.
- `git diff --stat ec49bcf..03141cf` = `utils/utils.cuh | 26 insertions, 137
  deletions`, one file. `python3 utils/licenses.py scan-nvidia TurboFNO` now
  reports "no NVIDIA proprietary licence text" (it reported utils.cuh before).

### Call sites: 316 total, all untouched

Paren-matched scan of the whole repo (excluding the submodule and utils.cuh
itself), skipping commented-out lines: 284 `CUDA_RT_CALL` and 32 `CUFFT_CALL`
active invocations, of which 142 + 16 are in the ten built `fusion_variants/`.
Zero source files other than utils.cuh appear in the diff. The same scan confirmed
every invocation is followed by `;`, which is what makes the switch from a bare
`{ ... }` block to `do { ... } while (0)` safe (the new form is strictly better --
it also survives an unbraced `if`/`else`).

Left as is, flagged for a decision: `CUBLAS_CALL` (2 active call sites) is
upstream's own macro, sits outside the notice-covered block and has no counterpart
in it, but its message string
`"ERROR: cuBLAS call \"%s\" failed in line %d of file %s with error code (%d)."`
is a word-reordering of the sample's sentence. Out of scope for this round; worth
a person's call on whether it should get the same treatment.

### VERIFIED on this host (linux-gfx1100, ROCm 7.2.3, W7800, gfx1100)

Device-code binary equivalence (the load-bearing proof that a host-only header
change did not move numerics). Built all ten variants at ec49bcf and at 03141cf,
extracted `.hip_fatbin` from each variant's kernel object with
`llvm-objcopy --dump-section`, and compared sha256:
```
export PROJECT_ROOT=$(pwd)
USE_HIP=1 CMAKE_HIP_ARCHITECTURES=gfx1100 bash install.sh   # 43s, 10/10, 0 errors
/opt/rocm/llvm/bin/llvm-objcopy --dump-section=.hip_fatbin=out.bin <variant>.cu.o /dev/null
```
All 10 SAME, byte for byte (1D_A 561568 B, 1D_B 654176, 1D_C 682832, 1D_D 639192,
1D_E 53456, 2D_A 1042328, 2D_B 1152712, 2D_C 1181496, 2D_D 1139088, 2D_E 55960).
Even `__hip_cuid_` matched. `utils.cu.o` is byte-identical as a whole object.
Only the ten `fused*.cu.o` host halves differ, which is the change itself.

Runtime smoke on GPU 0 (this session GPU 0 was healthy; see the 2026-06-04 note
about clock-gated GPUs on this host, which did not recur):
- `TurboFNO_1D_D` (fused): sweep prints cleanly, no macro error output.
- `TurboFNO_1D_E` (hipFFT/hipBLAS baseline): same, exercising the CUFFT_CALL
  success path around hipfftCreate/hipfftPlan1d/hipfftExecC2C.

Failure-path probe for BOTH macros (`agent_space/turbofno_macro_failpath.cu`,
built with `clang++ -x hip --offload-arch=gfx1100 -DUSE_HIP -Ihip_compat -Iutils
... -lhipfft -lhipblas`):
```
[probe] CUFFT_CALL success path: silent, control flow continued
.../turbofno_macro_failpath.cu:14: cufftExecC2C(plan, nullptr, nullptr, CUFFT_FORWARD) -> FFT status 6
[probe] fft_probe returned 1 (expected 1)
[probe] CUDA_RT_CALL success path: silent, q=non-null
[probe] .../turbofno_macro_failpath.cu:29: cudaMalloc(&p, (size_t)1 << 48) -> out of memory (2)
[probe] CUDA_RT_CALL failure path: control flow continued, p=(nil)
```
Both failure paths fire with the intended message and the intended control flow;
both are silent on success.

CUDA path intact, nvcc 12.8 (`/opt/conda/envs/cuda-12.8`), `USE_HIP=OFF`,
`CMAKE_CUDA_ARCHITECTURES=86`: all ten variants configure, compile AND LINK.
PASS=10 FAIL=0. (The CUDA-samples headers utils.cuh includes on that path come
from the submodule's `TurboFFT/Common`, which is on the include path and was not
touched.) Throwaway `fusion_variants/*/build-cuda-check` dirs removed afterwards.

### Gotcha: hipfftPlan1d with length 0 HANGS (ROCm 7.2.3)

Chasing a forced CUFFT_CALL error, `hipfftPlan1d(&plan, 0, HIPFFT_C2C, 1)` did not
return an error -- it never returned at all. Reproduced with a 12-line program that
does not include any TurboFNO header, so it is a rocFFT robustness issue, not a
port issue. cuFFT returns `CUFFT_INVALID_SIZE` for the same call. Working error
triggers that DO return a status: `hipfftExecC2C(plan, nullptr, nullptr, dir)`
-> 6, on either a planned or an unplanned handle. Note `hipfftPlan1d(..., batch=0)`
returns SUCCESS. Promoted to the skill (fault-classes) since any project porting
cuFFT error-handling tests can hit it.

## Review 2026-08-24 (reviewer, linux-gfx1100): CHANGES REQUESTED

Scope: `git diff ec49bcf..03141cf` on `moat-port` (one file, `utils/utils.cuh`,
+26 -137) plus the skill lesson promoted with it (MOAT `8018ccf`). Verdict is
changes-requested on the promoted lesson and one notes number ONLY. The fork
branch needs no change: every claim made for 03141cf was re-verified on this
host and held.

### 1. Promoted lesson: the binary-equivalence recipe is unsound as written

`.claude/skills/cuda-to-rocm/references/strategy-a-cmake.md:133-137` tells the
reader to dump `.hip_fatbin` per object and sha256-compare, then asserts "if it
is not identical, something other than the helper changed". That inference only
holds when both builds happen in the SAME directory. `__hip_cuid_<hash>` is
derived from the source path, so two checkouts built side by side in different
directories differ with no source change at all.

Measured here: ec49bcf built at two different absolute paths, same compiler and
flags, gives all ten `.hip_fatbin` sections DIFFERENT at identical sizes -- 54
differing bytes in 1D_E's 53456-byte section, and `llvm-nm` shows
`__hip_cuid_647db2785858a339` vs `__hip_cuid_842064d23045946`. Following the
paragraph literally produces a 10/10 false alarm; it produced one here before
the cause was found.

Required edits to that paragraph:
- say the two builds must occur in the same directory (build sha A, capture the
  objects, `git checkout` sha B in place, rebuild), or that `__hip_cuid_` must be
  excluded from the compare;
- point at `utils/codeobj_diff.py`, MOAT's canonical instrument for exactly this
  question. It compares normalized device ISA plus exported symbols, so it is
  immune to the path artifact: run across DIFFERENT paths and across both shas it
  reported `identical (exported symbols + device ISA identical (3 exports))` for
  all ten `TurboFNO_*` executables. The hand-rolled sha256 compare should be the
  fallback, not the headline;
- note that `codeobj_diff.py`'s OVERALL verdict degrades to `indeterminate` when
  the build tree contains CMake's own probe binaries
  (`CMakeFiles/<ver>/CompilerId{CXX,HIP}/a.out`,
  `CMakeDetermineCompilerABI_{CXX,HIP}.bin`), which have no device code. The
  per-binary lines are the answer; the next validator carrying this project
  forward will hit this.

The same caveat is missing from this file's own record of the method at
`projects/TurboFNO/notes.md:440-448` ("Built all ten variants at ec49bcf and at
03141cf"): add that both builds were in one directory, so a later reader does not
reproduce the false alarm.

### 2. Notes: CUBLAS_CALL call-site count is half the tree count

`projects/TurboFNO/notes.md:430` says `CUBLAS_CALL` has "2 active call sites".
The tree has 4: `fusion_variants/{1D_E_baseline/fused.cu:186,
2D_E_baseline/fused_trunc_2D.cu:219}` and
`fusion_variants_benchmark/{1D_E_baseline/fused.cu:203,
2D_E_baseline/fused_trunc_2D.cu:236}`. Two of the four are in the built
variants. This is inconsistent with the same section's own tree-wide accounting
(the 284 + 32 figures span `fusion_variants` AND `fusion_variants_benchmark`:
146+146 raw lines minus 4+4 commented for CUDA_RT_CALL, 16+16 for CUFFT_CALL --
both re-derived here and correct). Since this number is what a person weighs when
ruling on the escalated CUBLAS_CALL question, state 4 (2 built).

### Reviewer recommendation on the escalated CUBLAS_CALL question (person's call)

Recommend giving it the same treatment, in one further round, and rewriting the
message string only. Reasons: it is the last string in the header whose sentence
tracks the sample's, it now sits directly below two macros that were rewritten
precisely because their sentences did, and anyone comparing the header to the
sample lands on it immediately -- which is the question this round was meant to
close. Cost is bounded: keep the name `CUBLAS_CALL` and the
report-then-`exit(EXIT_FAILURE)` behaviour, so none of the 4 call sites move, the
diff stays one file, and the same host-only binary-equivalence proof applies.
The argument for leaving it: it is upstream's OWN macro, sits outside the removed
notice block, and rewriting it edits upstream code for no functional gain, which
cuts against "smallest complete port". On balance the tidier provenance story is
worth one more one-line commit, but this is a person's decision and is NOT
blocking either this round or validation.

### Claims re-verified independently on this host (all CONFIRMED)

ROCm 7.2.3, AMD Radeon Pro W7800 (gfx1100, wave32); CUDA 12.8
(`/opt/conda/envs/cuda-12.8`), no NVIDIA GPU.

1. No NVIDIA proprietary text. `licenses.py scan-nvidia TurboFNO` clean;
   independent `git grep -i` for "NOTICE TO LICENSEE", "Licensed Deliverables",
   "PROPRIETARY and", "NVIDIA software license agreement" over the tracked tree =
   0 hits; on-disk grep including the submodule checkout = 0 hits. The three
   locations named by the porter (former 42-78, 156-240, 242-257) are all gone in
   03141cf. `hip_compat/helper_{cuda,functions}.h` are 2-line inert stubs.
   The submodule's `TurboFFT/Common/helper_cuda.h` carries a BSD-3-Clause header,
   not the proprietary notice, and the gitlink is unchanged at e285704.
2. Independence bar met. do/while(0) matches the file's OWN `CUDA_CALLER` idiom
   at `utils/utils.cuh:28-37`; internals are `turbofno_rt_status` /
   `turbofno_fft_status` by direct declaration, not `auto status =
   static_cast<...>( call )`; message is `"%s:%d: %s -> %s (%d)"`, compiler-style,
   not a reordering of the deleted sentence (compared against the ec49bcf text).
3. 316 call sites untouched. `git diff --name-only ec49bcf..03141cf` =
   `utils/utils.cuh` alone; a whole-tree `diff -rq` of the two checkouts also
   reports only that file. Re-derived counts: 284 active CUDA_RT_CALL (292 lines
   - 8 commented) and 32 CUFFT_CALL, matching the notes exactly. Every wrapped
   expression is `cudaMalloc` (152) or `cudaMemcpy` (140) for CUDA_RT_CALL and
   `cufftCreate/Plan1d/PlanMany/ExecC2C/Destroy` for CUFFT_CALL, so dropping the
   sample's `static_cast` for a direct declaration is type-safe at every site --
   there is no enum-to-enum initialisation anywhere.
4. Behaviour preserved. Probe built with the port's own header
   (`clang++ -x hip --offload-arch=gfx1100 -DUSE_HIP -Ihip_compat -Iutils`):
   CUDA_RT_CALL silent on success, reports and CONTINUES on
   `cudaMalloc(1<<48)`; CUFFT_CALL silent on success, reports and RETURNS 1 from
   its enclosing function on `cufftExecC2C(plan, nullptr, nullptr, FORWARD)`
   (status 6). The probe also exercised `if (r) CUDA_RT_CALL(...); else ...`,
   which compiles and takes the else branch -- the do/while(0) dangling-else
   improvement is real. No repo file or script consumes the old message text
   (grep for "CUDA RT call" outside the submodule = 0 hits).
5. Device-binary equivalence. Both shas rebuilt in ONE directory, all 10
   `.hip_fatbin` sections byte-identical, sizes matching the notes exactly
   (1D_A 561568, 1D_B 654176, 1D_C 682832, 1D_D 639192, 1D_E 53456,
   2D_A 1042328, 2D_B 1152712, 2D_C 1181496, 2D_D 1139088, 2D_E 55960).
   Independently, `utils/codeobj_diff.py` over the two builds:
   `identical (exported symbols + device ISA identical (3 exports))` on all ten
   executables. `utils/utils.cu` references none of the three macros, so the
   byte-identical `utils.cu.o` claim follows.
6. Both build paths healthy. HIP `USE_HIP=1 CMAKE_HIP_ARCHITECTURES=gfx1100 bash
   install.sh` at BOTH shas: 10/10 executables, 0 `error:`, 436 warnings each
   (unchanged). CUDA path at 03141cf with nvcc 12.8 and
   `-DCMAKE_CUDA_ARCHITECTURES=86`: 10/10 configure + compile + LINK, 10
   executables. The literal Test Plan command
   (`cmake -S fusion_variants/1D_E_baseline -B build-cuda -DUSE_HIP=OFF
   -DCMAKE_CUDA_ARCHITECTURES=86` + `cmake --build`) works as written with nvcc on
   PATH, and `install.sh` is mode 755 so `./install.sh` works too. Runtime smoke
   on gfx1100: 1D_E completes its whole sweep (rc=0), 1D_D emits 732 clean sweep
   lines before my own timeout; neither printed macro error output.
7. Submodule and notice FILES untouched. `TurboFFT` gitlink identical;
   `git diff ec49bcf..03141cf -- LICENSE NOTICE` empty. `NOTICE` makes no
   reference to the removed sample code, so nothing dangles.
8. Commit hygiene. Title `[ROCm] Replace sample-derived error-check macros in
   utils.cuh` = 61 chars, correct prefix; no `Co-Authored-By` trailer; AI
   assistance disclosed; Test Plan present with literal fenced commands;
   `python3 utils/jargon.py --port TurboFNO` = clean; no AMD-internal account
   reference. The hard-wrapped body is correct for a commit message
   (`utils/prose.py` scope is PR and issue bodies).

The rocFFT gotcha promoted at `references/fault-classes.md:319-326` also
reproduces here exactly as written: `hipfftPlan1d(&plan, 0, HIPFFT_C2C, 1)` never
returns (killed at 60 s) on ROCm 7.2.3 / gfx1100 from a program including no
project header, while `hipfftPlan1d(&plan, 16, HIPFFT_C2C, 0)` returns 0 and
`hipfftExecC2C(plan, nullptr, nullptr, dir)` returns status 6. That entry stands
as accurate; `deferred.json:rocfft-plan1d-size0-hang` is the right home for the
report and still awaits a person's ruling.

### Not applicable to this delta

No wavefront-size, texture/rule-of-five, OOB-neighbour, texture-pitch, per-arch
or library-substitution surface: the change is host-side preprocessor text in one
header, and the device code is proven unchanged. Strategy A remains correct.
Platform validation state is a separate matter -- all four platforms sit at
`validated_sha` ec49bcf while `head_sha` is 03141cf, so revalidation (or a
`codeobj_diff` carry-forward, which the evidence above already supports) is the
validator's next step regardless of this verdict.
