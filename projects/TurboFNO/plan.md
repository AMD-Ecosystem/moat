# TurboFNO port plan (linux-gfx90a lead)

## Project
- Name: TurboFNO
- Upstream: https://github.com/shixun404/TurboFNO
- Default branch: main
- Base sha: c83a74b860248d4a9fb9626179263f3761f5add4
- Submodule: TurboFFT @ e28570417284b8e66c124880e8805b677075076b (branch TurboFNO_dev), required for all FFT/iFFT kernels and include paths.
- SC25-paper research repo: "first fully fused FFT-GEMM-iFFT GPU kernel" for the Fourier Neural Operator.

## DISPOSITION
clean-port, Strategy A (pure CMake + single cuda_to_hip compat header). Effort class: medium (mechanical, but 10 CMake builds, a CUDA-samples-header dependency to remove, cuFFT/cuBLAS -> hipFFT/hipBLAS in the baseline, and a wave64 correctness verification to add). NOT a CUTLASS problem and NOT already-AMD-supported. Dispatch a porter.

## Existing AMD support
None. No HIP/ROCm path, no OpenCL/Vulkan/SYCL path, no stale port branch or fork. Pure CUDA. A HIP port targeting ROCm adds value. -> proceed.

## Build classification: cmake (Strategy A)
Evidence: each variant has its own `CMakeLists.txt`, e.g.
- `fusion_variants/1D_A_exp_fft+cgemm+ifft/CMakeLists.txt:2` `project(... LANGUAGES CXX CUDA)`
- line 9 `find_package(CUDAToolkit REQUIRED)`
- lines 26-31 `file(GLOB VARIANT_SOURCES "*.cu" "${PROJECT_ROOT}/utils/*.cu")` + `add_executable`
- line 49 `target_link_libraries(... CUDA::cublas CUDA::cufft)`
No `find_package(Torch)`, no `torch.utils.cpp_extension`, no setup.py/pyproject. -> pure CMake, Strategy A.
`install.sh` loops 10 variant dirs (1D/2D x {A,B,C,D,E}), each a standalone CMake build emitting one `TurboFNO_<dim>_<variant>` benchmark executable. Builds require `export PROJECT_ROOT=$(pwd)` (CMake FATAL_ERROR otherwise) and the TurboFFT submodule on the include path.

## Port strategy: A (compat header), rationale
The compiled device surface is entirely hand-written CUDA C++ (SIMT FMA, no vendor templates). The only CUDA-library use is cuFFT/cuBLAS in the two E baselines (reference/comparison path). So: add one `cuda_to_hip.h` aliasing the small `cudaXxx`/`cufftXxx`/`cublasXxx` set the code uses (-> `hip*`/`hipfft`/`hipblas`), gate `enable_language(HIP)` behind `option(USE_HIP)` in each CMakeLists (or a shared `.cmake` include), mark the `.cu` `LANGUAGE HIP`, and read `${CMAKE_HIP_ARCHITECTURES}` (default gfx90a only when unset; never hardcode the literal so gfx1100/gfx1151 build with only `-DCMAKE_HIP_ARCHITECTURES=`). Link `hip::hipfft hip::hipblas` (or roc equivalents) on the HIP path.

No AMD-native rewrite is needed for correctness. The GEMM is a plain `float2` FMA kernel (below); MFMA/CK is a future perf option, explicitly out of scope for the correctness-first port.

## CUDA surface inventory
Device kernels (all in `.cu`/`.cuh`, no vendor headers):
- Complex GEMM `cgemm` (`fusion_variants/*/cgemm.cuh`): shared-memory-tiled, double-buffered, register-blocked SIMT GEMM on `float2`. Tile 64x64x8, 256 threads/block, per-thread 4x4. The complex multiply is explicit FMA (`cgemm.cuh:100-101`). NO tensor cores.
- Hand-rolled radix-2 FFT/iFFT kernels (`fft_*_stride_DY`, `ifft_*_stride_DY`, plus `_fused`/`_fused_output`/`_trunc` variants) auto-generated in the TurboFFT submodule under `TurboFFT/TurboFFT/include/code_gen/generated/float2/`. Pure register-butterfly + `__shared__` + `__syncthreads()`; complex ops via `turboFFT_ZADD/ZSUB/ZMUL` macros (`utils/utils.cuh:265-267`). Twiddles via `__cosf`/`__sinf` (device fast-math intrinsics; HIP-supported).
- Fused kernels `fused_fft_cgemm_ifft_{7..10}` etc.: the GEMM tiling above with FFT/iFFT inlined; same primitives.
- Data-staging kernels in the drivers: `direct_copy_colmajor_float4_truncation` / `_zero_padding` (`fused.cu:29,55`) -- `float4` reinterpret copies, fully portable.
- Runtime/API: `cudaMalloc/Memcpy/DeviceSynchronize`, `cudaEvent*` timing, `cudaGetLastError`, `cudaDeviceProp` (`utils/utils.cu:30`). All 1:1 to `hip*`.
- Library calls (E baselines only): `cufftCreate/Plan1d/PlanMany/ExecC2C/Destroy` and `cublasCgemm` (`fusion_variants/1D_E_baseline/fused.cu:174-190`, `2D_E_baseline/fused_trunc_2D.cu:196-208`). -> hipFFT + hipBLAS.

GEMM finding (decisive): hand-written, NOT CuTe/CUTLASS. `grep` for `cutlass|cute::|cute/` across repo+submodule = zero hits. `#include <mma.h>` appears in every `cgemm.cuh`/fused header but `nvcuda::`/`wmma::`/`fragment`/`mma_sync`/`fill_fragment` = zero hits: the include is DEAD (drop it or leave it; `mma.h` may not exist for HIP, so the compat header should `#if defined(USE_HIP)` skip it or the porter removes the include).

FFT-library finding (decisive): the FUSED/experimental kernels use the project's OWN hand-rolled FFT (portable). cuFFT is used ONLY in the E baselines as the timing/reference comparator. So cuFFT->hipFFT is a small, isolated substitution confined to two files; the headline contribution does not depend on it.

PTX finding: NONE. `grep` for `asm volatile|asm(|__asm|wgmma|cp.async|ldmatrix` = zero hits. No memory-access PTX, no tensor-core PTX. Nothing to convert.

## Risk list
1. wave64 logical-warp GEMM tiling (PRIMARY). `utils/TurboFNO.h` hardcodes `WARP_M 32`, `WARP_N 16`, `WID (threadIdx.x/32)`, and the kernels index with `TID % 32` (`cgemm.cuh:60,64,...`; same in fused D `fused_fft_cgemm_ifft_8.cuh:57,61,...`). This is a *logical 32-thread* register-tiling partition, not a warp-collective op -- per PORTING_GUIDE this idiom is arch-agnostic (a 32-lane subgroup regardless of physical wavefront) and cross-warp sharing is fenced by `__syncthreads()`. So it SHOULD be correct on wave64 with no change. BUT it must be VERIFIED on gfx90a, because: (a) on wave64 a 64-thread half-block is one wavefront holding two logical warps -- any place relying on independent-warp scheduling would change; there are none here (no `__syncwarp`, no shuffle), but confirm; (b) confirm `threadIdx.x/32` partitioning still maps to the intended shared-memory tiles. There are NO `__shfl`/`__ballot`/`warpSize`/`__activemask`/`__syncwarp` anywhere (repo+submodule grep = zero), so the classic wave-collective fault class does not apply. Do NOT introduce a wave64 hardcode; leave the logical-32 math as-is and prove it numerically.
2. cuFFT/cuBLAS substitution (E baselines): hipFFT plan/exec enum and handle parity (`hipfftPlan1d`, `hipfftPlanMany`, `HIPFFT_C2C`, `HIPFFT_FORWARD`), hipBLAS `hipblasCgemm` with `HIPBLAS_OP_N` and `hipblasComplex*` casts. Watch the hipBLAS v2 enum/type names.
3. CUDA-samples headers. `utils/utils.cuh:5-6` includes `helper_functions.h` and `helper_cuda.h` (NVIDIA CUDA-samples, not shipped, not on ROCm). They are effectively UNUSED in TurboFNO's own TUs (`checkCudaErrors`/`sdk*` appear only in the submodule's standalone `main.cu`, which TurboFNO does not build). Remove the two includes (preferred) or shim them; without this the HIP build will not find the headers. Confirm nothing in the built `.cu` references their symbols (grep already says no).
4. No built-in correctness gate. The shipped `main()` in every variant is TIMING-ONLY; `verify_vector`/`verify_matrix` are implemented in `utils/utils.cu:114,166` but the call sites are commented out. The validator must ADD a correctness comparison (the E baseline already computes a cuFFT/cuBLAS `dC_ref`/`diFFT_output_ref`; wire `verify_vector` against it, or diff a fused variant against the E reference). This is the GPU validation gate -- see Test plan.
5. `--expt-relaxed-constexpr` (CUDA flag, `CMakeLists.txt:34`): harmless under HIP but should be gated to the CUDA path or dropped for HIP (clang-HIP does not need it).
6. fp fast-math drift: `__cosf/__sinf/__fsqrt` and the default `-ffp-contract=fast` on clang-HIP can drift ~1 ULP vs CUDA (PORTING_GUIDE fault class). The verification tolerance is relative (`verify_vector` reports MAE/RMSE/rel-diff, threshold ~1e-1 rel for outliers), so a generous-but-meaningful tolerance is appropriate; do not demand bit-exactness against a CUDA-produced gold. Pin `-ffp-contract=on` only if drift exceeds tolerance.
7. Non-power-of-handle resource cleanup: drivers create/destroy cufft/cublas handles per outer iteration in E; ensure hip handle lifetimes are clean (rule-of-five class is not triggered -- no texture/surface objects anywhere).

Not applicable: textures/surfaces (none), managed/pinned memory (none -- plain malloc + cudaMemcpy), streams (default stream only), Thrust/CUB/cuRAND/cuSPARSE (none), layered arrays (none), `__smid` (none).

## File-by-file change list
- NEW `utils/cuda_to_hip.h`: `#if defined(USE_HIP)||defined(__HIP_PLATFORM_AMD__)` include `<hip/hip_runtime.h>`, `<hipfft/hipfft.h>`, `<hipblas/hipblas.h>`; alias the exact `cuda*/cufft*/cublas*/CUFFT_*/CUBLAS_*` symbols the code uses; `#else` include the CUDA runtime + cufft/cublas. Avoid pulling `mma.h` on HIP.
- `utils/utils.cuh`: include `cuda_to_hip.h`; remove the two CUDA-samples includes (lines 5-6). Keep the `CUFFT_CALL`/`CUBLAS_CALL`/`CUDA_RT_CALL` macros (they work post-alias).
- Each of the 10 `fusion_variants/*/CMakeLists.txt`: add `option(USE_HIP OFF)` + `enable_language(HIP)` branch; `set_source_files_properties(${VARIANT_SOURCES} PROPERTIES LANGUAGE HIP)`; default `CMAKE_HIP_ARCHITECTURES` to gfx90a only when unset; link `hip::hipfft hip::hipblas` on HIP else `CUDA::cublas CUDA::cufft`; gate `--expt-relaxed-constexpr` to CUDA. Prefer factoring this into one shared `cmake/turbofno_hip.cmake` `include()`d by each, to keep the diff DRY across 10 files.
- `fusion_variants/*/*.cu` and `*.cuh`, submodule generated `.cuh`: NO source edits expected beyond what the compat header covers; drop the dead `#include <mma.h>` if it fails to resolve on HIP. The logical-32 GEMM math stays unchanged.
- (validator, not porter) add a correctness check to at least one 1D and one 2D variant (uncomment/extend `verify_vector` against the E reference path).

## Build commands (gfx90a)
Per variant (example 1D_D):
```
export PROJECT_ROOT=/var/lib/jenkins/moat/projects/TurboFNO/src
cd $PROJECT_ROOT/fusion_variants/1D_D_exp_fused_fft_cgemm_ifft
cmake -S . -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build build -j16
```
Multi-arch correctness build for the warp-size proof:
`-DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100"` then confirm both code objects with `llvm-objdump --offloading build/TurboFNO_1D_D`.
Full sweep: adapt `install.sh` to pass `-DUSE_HIP=ON` (the porter should add a HIP path to install.sh or document the per-variant invocation in notes.md).

## Test plan
- GPU validation gate (must add; none ships): build the E baseline (cuFFT/cuBLAS -> hipFFT/hipBLAS reference) AND a fused variant (A/D) for the SAME problem sizes from `benchmark_config/problem_size_{1d,2d}.txt`; compare the fused FFT-GEMM-iFFT output against the hipFFT+hipBLAS reference using `verify_vector` (relative-error metrics already printed). Pass = outlier_perct ~0 and max_rel_diff within tolerance (fast-math/contraction-aware, e.g. rel < 1e-2..1e-1 as the code's own outlier threshold). Cover at least one 1D and one 2D problem-size sweep.
- Independent FFT round-trip check: FFT followed by iFFT of random input should recover the input (scaled); a cheap correctness probe for the hand-rolled FFT independent of the GEMM.
- Cross-arch consistency (followers): the algorithm is deterministic; gfx1100/gfx1151 must diff their fused output against the gfx90a output for identical input (PORTING_GUIDE: a reference-less "non-zero/plausible" gate would mask a wave32 logical-warp bug).
- Non-GPU regression: all 10 variants must still configure + build under both `-DUSE_HIP=ON` (ROCm) and the default CUDA path (no USE_HIP) so the NVIDIA build is intact.
- Runtime smoke: each built `TurboFNO_*` runs to completion over its config sweep with no `CHECK_CUDA_KERNEL`/`CUDA_RT_CALL` errors.

## Open questions
- hipBLAS major version on the host: pick `hipblasCgemm` signature/enum spelling accordingly (v2 enums). Verify at build time.
- Does any fused variant need a tolerance looser than the E baseline due to the hand-rolled FFT's fast-math twiddles? Determine empirically during validation; set the gate tolerance from the cuFFT-vs-TurboFFT difference observed on gfx90a, not an arbitrary constant.
- install.sh currently hardcodes the CUDA path; decide whether to add a HIP branch to it or leave it CUDA-only and document HIP invocation in notes.md (lean: add a USE_HIP env switch to install.sh).

## Delta plan: linux-gfx1100 / windows-gfx1151 (followers, on demand)
Reuse the gfx90a fork branch; rebuild with `-DCMAKE_HIP_ARCHITECTURES=gfx1100` (or gfx1151) -- no source change expected because the only wave dependency is the logical-32 GEMM tiling, which is wave-agnostic, and there are no wave-collective ops. RDNA is wave32, so each 32-thread logical warp is exactly one wavefront (the wave64 "two-logical-warps-per-wavefront" case disappears) -- if anything this is the SAFER width for this code. Validate by cross-arch output diff against the gfx90a result (deterministic). gfx1151: confirm hipFFT/hipBLAS availability in the Windows HIP SDK; if a library is missing, the E baseline reference may need a CPU FFT/GEMM fallback for the gate, but the fused kernels (no library deps) should still build and run.
