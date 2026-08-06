# 3P-ADMM-PC2 notes

Privacy-preserving distributed ADMM for LASSO. CPU does the optimization
(Paillier homomorphic encryption + quantized ADMM, gmpy2). The GPU's only job
is to accelerate the Paillier modular exponentiation `g^m mod n` over a batch
of big integers, via cuFFT-based large-integer polynomial multiplication +
Barrett reduction.

## Build classification
ext_type `nvcc-shared`: ad-hoc nvcc commands in README.md sec.1 (no CMake, no
Makefile, no setup.py). The `.cu` is compiled + a generated C wrapper, linked
with `g++ -shared` into `/tmp/lib_cufft.so`. The Python side
(`crypto/paillier_gpu.py`, `protocol/edge_crt_helper.py`) loads that .so by
absolute path via `ctypes.CDLL` and calls `init_gpu(int)` / `run_modexp(...)`.

## Port scope (LIVE GPU path only)
Ported ONLY `gpu/cufft_modexp.cu` (cuFFT Z2Z + scalar bigint kernels). This is
the only GPU code wired into the protocol.

Deliberately EXCLUDED:
- Dead register/warp-NTT files: `gpu/reg_ntt_modexp*.cu`, `gpu/ntt_modexp*.cu`,
  `gpu/ntt_funcs.cu`, `gpu/ntt_kernel2.cu`, `gpu/ntt_wrapper*.cu`,
  `gpu/modexp_final.cu`. Referenced by nothing in the repo (no Python import,
  no launch site, not built by the README). The `reg_ntt_*` and `ntt_funcs`
  files DO use `__shfl_xor_sync(0xffffffff, ..., delta)` warp-NTT butterflies
  with blockDim=32 (one warp/task) -- classic wave64 hazards -- but they are
  not on any live path, so porting them would expose the wave64 fault class for
  no validatable benefit. A future contributor reviving the register-NTT path
  on AMD would need the `__GFX9__`->64 / RDNA->32 + width-aware ballot
  treatment.
- Standalone PyCUDA self-test: `gpu/gpu_modexp.py` (JITs `gpu/modexp.cu` /
  `gpu/modexp_v2.cu`). PyCUDA is not in requirements.txt and this is a
  `__main__` self-test, not called by the protocol. Out of scope.

## What changed
- ADD `gpu/cuda_to_hip.h`: single compat shim. Under `__HIP__`/
  `__HIP_PLATFORM_AMD__`/`USE_HIP` it includes `<hip/hip_runtime.h>` +
  `<hipfft/hipfft.h>` and aliases the exact CUDA symbols the kernel uses
  (`cudaMalloc/Free/Memcpy` + dirs; `cufftHandle`/`cufftDoubleComplex`,
  `cufftPlanMany`/`cufftExecZ2Z`/`cufftDestroy`, `CUFFT_Z2Z`/`CUFFT_FORWARD`,
  `CUFFT_INVERSE`->`HIPFFT_BACKWARD`); else it includes the original CUDA
  headers. NVIDIA build stays byte-identical.
  - Gotcha: hipcc compiling `-x hip` defines `__HIP__`, NOT
    `__HIP_PLATFORM_AMD__`, at preprocess time. Key the `#if` off `__HIP__`
    too, or the shim falls through to `<cuda_runtime.h>` and fails to find it.
  - libc headers (`<cstdlib><cstring><cmath>`) are included BEFORE the HIP
    runtime (gpuRIR lesson): host `calloc`/`free`/`round` in the .cu would
    otherwise risk binding to HIP `__device__` overloads.
- EDIT `gpu/cufft_modexp.cu`: replaced `<cuda_runtime.h>`/`<cufft.h>`/`<math.h>`
  with `#include "cuda_to_hip.h"`. No body changes; all cuda*/cufft* spellings
  preserved.
- ADD `gpu/build_hip.sh`: hipcc recipe mirroring the README nvcc steps,
  producing `/tmp/lib_cufft.so` with the SAME `init_gpu`/`run_modexp` C ABI
  (Python untouched), linking `-lhipfft`. Arch from `HIP_ARCH` (default
  `gfx90a`, never a literal). hipcc does host+device link in one step (no
  separate `-dlink`/`-fgpu-rdc` needed for this non-RDC build).

## Build (gfx90a)
```
HIP_VISIBLE_DEVICES=1 \
BUILD=<scratch> OUT=/tmp/lib_cufft.so \
  bash gpu/build_hip.sh
```
Followers: `HIP_ARCH=gfx1100` / `gfx1151`. Multi-arch fat binary:
`HIP_ARCH="gfx90a,gfx1100"` builds clean (verified). The only warnings are
benign `-Wunused-value` on the cuda*Memcpy nodiscard return (present in the
NVIDIA build too). hipcc/ROCm 7.2.1, hipfft present at /opt/rocm.

## hipFFT rounding note (the one real risk -- PASSED)
The bigint multiply is a double-precision Z2Z FFT then `round()` to recover
exact integer limbs (`norm_round` + `carry_prop`), relying on a < 0.5 rounding
margin. hipFFT twiddles differ bit-for-bit from cuFFT, so a thin margin could
in principle round a limb the wrong way. Validated statistically: 6144 random
modexp cases across moduli up to 2046-bit n^2 -- all exact-match gmpy2. The
double-FFT margin holds on gfx90a (full-rate fp64). No code change was needed.

## Validation (real gfx90a, GCD 1, HIP_VISIBLE_DEVICES=1)
- modexp-vs-gmpy2 gold: 6144/6144 exact match. `gpu_batch_modexp(g,m,n)` vs
  `gmpy2.powmod(g,m,n)`, n at 255/510/511/1022/768/1535/1023/2046 bits, varied
  m_bits, 256/batch. Harness: agent_space/3p-admm/validate.py (not committed).
- ALGORITHM PRECONDITION: the kernel assumes base g < modulus n (no initial
  reduction). A test with g >= n mismatches gmpy2 -- this is expected (contract
  violation), NOT a HIP bug. Real Paillier always has g < modulus. Reducing
  g %= n before the call makes it match exactly.
- encrypt round-trip: `crypto/paillier_gpu.encrypt_batch_gpu` (GPU g^m mod n^2)
  -> CPU `crypto/paillier.decrypt` = 64/64 plaintexts recovered.
- CPU regression (no GPU touched): `crypto/test_paillier.py`,
  `test_quantization.py`, `test_full_chain.py` all pass.
- gfx90a dispatch confirmed via AMD_LOG_LEVEL=3 (hipFFT `twiddle_gen_radices_dp`
  + our kernels on device; rocminfo Name gfx90a).

## Wave size
Not exposed in the live path: no `__shfl`/`__ballot`/`warpSize`/`__syncwarp`,
no cub/thrust/curand, no textures/atomics. The `<<<N,32>>>` launches
(`cond_copy`, `load_complex` grid.x) are independent-thread strided copies, not
warp collectives. Multi-arch is trivially fine; gfx1100/gfx1151 deltas are
expected to be no-op revalidations (same .so rebuilt for the arch). hipFFT is
arch-agnostic.

## Upstream warts (NOT touched -- out of port scope)
- Hardcoded `/mnt/3p-admm-pc2/...` absolute paths in Python (lib + kernel source
  paths) and `sys.path.append('/mnt/3p-admm-pc2')` in the test files. For our
  build+validate we point the loader at the real /tmp/lib_cufft.so and set
  PYTHONPATH. We do not refactor upstream's path scheme.
- `/tmp` is non-persistent (README notes "recompile each boot").
- Distributed end-to-end tests (`experiments/*.py`, `protocol/master_node.py`)
  need multi-node SSH / matpool cloud; not runnable here. The modexp gold gate
  + encrypt round-trip are the representative validatable proxy for the GPU work
  those drive.

## Validation 2026-06-02 (validator, linux-gfx90a, GCD 1)

validated_sha: 6ef301f3204579779bbaa1f32a466934f720903a

GPU arch: gfx90a (AMD Instinct MI250X, ROCm 7.2.1, HIP_VISIBLE_DEVICES=1)

Commands run:
```
# gfx90a build
HIP_VISIBLE_DEVICES=1 BUILD=/var/lib/jenkins/moat/agent_space/3p-admm OUT=/tmp/lib_cufft.so \
  bash utils/timeit.sh 3P-ADMM-PC2 compile -- bash projects/3P-ADMM-PC2/src/gpu/build_hip.sh

# fat binary (gfx90a + gfx1100)
HIP_VISIBLE_DEVICES=1 HIP_ARCH="gfx90a,gfx1100" \
  BUILD=/var/lib/jenkins/moat/agent_space/3p-admm OUT=/tmp/lib_cufft_multi.so \
  bash utils/timeit.sh 3P-ADMM-PC2 compile -- bash projects/3P-ADMM-PC2/src/gpu/build_hip.sh

# modexp gold match
HIP_VISIBLE_DEVICES=1 AMD_LOG_LEVEL=0 \
  bash utils/timeit.sh 3P-ADMM-PC2 test -- python3 agent_space/3p-admm/validate.py

# encrypt round-trip, CPU regression tests
HIP_VISIBLE_DEVICES=1 PYTHONPATH=<src> python3 crypto/test_paillier.py
HIP_VISIBLE_DEVICES=1 PYTHONPATH=<src> python3 crypto/test_quantization.py
HIP_VISIBLE_DEVICES=1 PYTHONPATH=<src> python3 crypto/test_full_chain.py
```

Results:
- Build gfx90a: PASS (only benign -Wunused-value on nodiscard hipMemcpy/hipFree, matches CUDA build)
- Fat binary gfx90a+gfx1100: PASS; roc-obj-ls confirms both code objects embedded
  (hipv4-amdgcn-amd-amdhsa--gfx1100 @ offset=12288, hipv4-amdgcn-amd-amdhsa--gfx90a @ offset=40960)
- modexp vs gmpy2 gold: 6144/6144 exact match
  n at 255/510/511/768/1022/1023/1535/2046 bits, 256/batch, 3 reps each, varied m_bits
- Encrypt round-trip: 64/64 plaintexts recovered (GPU g^m mod n^2 -> CPU decrypt)
- CPU regression: test_paillier PASS, test_quantization PASS, test_full_chain PASS (max error 1.33e-09)
- Native gfx90a dispatch confirmed (AMD_LOG_LEVEL=3):
  "Using native code object for device: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-"
  hipFFT twiddle_gen_radices_dp dispatched; all our kernels dispatched via hipLaunchKernel
- hipFFT double-FFT rounding margin holds: 0 mismatches across all 6144 crypto-significant cases

VERDICT: PASS -- review-passed -> completed

## Review 2026-06-02 (reviewer, gfx90a)
review-passed. No problems found; no changes requested.

Verified independently on real gfx90a (HIP_VISIBLE_DEVICES=1, GCD 1, ROCm 7.2.1):
- Scope correct: diff touches only gpu/cuda_to_hip.h (new), gpu/build_hip.sh (new),
  gpu/cufft_modexp.cu (one include-line swap). No Python, no dead reg_ntt_*/ntt_funcs
  warp-NTT files, no PyCUDA self-test touched -- so the live path genuinely has no
  warp/wave64 surface (grep confirms no __shfl/__ballot/warpSize/__syncwarp in the .cu;
  the <<<N,32>>> launches are independent-thread strided copies). Multi-arch trivially fine.
- Shim symbol coverage complete: every cuda*/cufft*/CUFFT_* token used by cufft_modexp.cu
  (cudaMalloc/Free/Memcpy + H2D/D2H, cufftHandle/DoubleComplex/PlanMany/ExecZ2Z/Destroy,
  CUFFT_Z2Z/FORWARD/INVERSE->HIPFFT_BACKWARD) is aliased. libc (cstdlib/cstring/cmath)
  included before <hip/hip_runtime.h>. #if keys off __HIP__ -- independently confirmed
  `hipcc -x hip -dM -E` defines __HIP__ but NOT __HIP_PLATFORM_AMD__ at preprocess, so the
  __HIP__ key is load-bearing and correct. NVIDIA #else path falls through to original CUDA
  headers (math.h -> cmath is a superset; round() still resolves), byte-identical build.
- build_hip.sh wrapper (init_gpu/run_modexp -> cufft_init/cufft_modexp) is identical to the
  README's wr_cufft.cu, preserving the ctypes C ABI; arch from HIP_ARCH (no literal); no
  -fgpu-rdc needed (no cross-TU __device__ funcs).
- Built the gfx90a,gfx1100 fat binary clean (only benign -Wunused-value on nodiscard
  hipMemcpy, present on CUDA too); roc-obj-ls confirms both code objects embedded.
- Reproduced the correctness gate: 6144/6144 exact vs gmpy2.powmod across n at
  255/510/511/768/1022/1023/1535/2046 bits, varied m_bits, 256/batch. hipFFT double-FFT
  rounding margin holds.
- g>=n: confirmed the kernel has NO initial g%=n reduction (identical to upstream CUDA), so
  the precondition is pre-existing, not a port defect; after g%=n the batch matches gmpy2.
- Commit hygiene: title "[ROCm] Port live cuFFT Paillier modexp to hipFFT" (47 chars),
  mentions Claude, no noreply trailer, no ghstack, no em-dash. Fork features == upstream
  features @ dd96d5d (clean mirror); single port commit on moat-port. Actions disabled.

Non-blocking: the __HIP__-vs-__HIP_PLATFORM_AMD__ compat-shim-keying trap (an #include'd shim
keyed only on __HIP_PLATFORM_AMD__ silently falls through to <cuda_runtime.h> because hipcc
defines __HIP__, not __HIP_PLATFORM_AMD__, at preprocess) is a generalizable lesson and was
appended to PORTING_GUIDE.md.

## Validation 2026-06-02 (gfx1100, linux-gfx1100, HIP_VISIBLE_DEVICES=0)

validated_sha: 6ef301f3204579779bbaa1f32a466934f720903a

GPU arch: gfx1100 (AMD Radeon Pro W7800 48GB, ROCm 7.2.1, HIP_VISIBLE_DEVICES=0, wave32)

No code change from gfx90a lead -- validate-first follower, fork untouched at 6ef301f.

Commands run:
```
# gfx1100 build (HIP_ARCH=gfx1100, no literal)
HIP_VISIBLE_DEVICES=0 HIP_ARCH=gfx1100 \
  BUILD=/var/lib/jenkins/moat/agent_space/3p-admm-gfx1100 OUT=/tmp/lib_cufft_gfx1100.so \
  bash utils/timeit.sh 3P-ADMM-PC2 compile -- bash projects/3P-ADMM-PC2/src/gpu/build_hip.sh

# modexp gold match (run twice for determinism)
HIP_VISIBLE_DEVICES=0 AMD_LOG_LEVEL=0 LIB_CUFFT=/tmp/lib_cufft_gfx1100.so \
  bash utils/timeit.sh 3P-ADMM-PC2 test -- python3 agent_space/3p-admm-gfx1100/validate.py

# gfx1100 dispatch confirmation
HIP_VISIBLE_DEVICES=0 AMD_LOG_LEVEL=3 LIB_CUFFT=/tmp/lib_cufft_gfx1100.so \
  python3 agent_space/3p-admm-gfx1100/validate.py 2>&1 | grep "native code\|gfx1100\|0x1016"

# CPU regression
PYTHONPATH=projects/3P-ADMM-PC2/src python3 crypto/test_paillier.py
PYTHONPATH=projects/3P-ADMM-PC2/src python3 crypto/test_quantization.py
PYTHONPATH=projects/3P-ADMM-PC2/src python3 crypto/test_full_chain.py
```

Results:
- Build gfx1100: PASS -- only benign -Wunused-value on nodiscard hipMemcpy/hipFree (identical to gfx90a)
- roc-obj-ls confirms gfx1100 code object: hipv4-amdgcn-amd-amdhsa--gfx1100 @ offset=12288
  (single-arch gfx1100 only .so; no gfx90a object present in this build)
- Native gfx1100 dispatch confirmed (AMD_LOG_LEVEL=3):
  "Using native code object for device: amdgcn-amd-amdhsa--gfx1100"
  hipFFT twiddle_gen_radices_dp dispatched; all kernels hipLaunchKernel hipSuccess; no HSA 0x1016
- modexp vs gmpy2 gold (run 1): 6144/6144 exact match
  n at 255/510/511/768/1022/1023/1535/2046 bits, 256/batch, 3 reps each, varied m_bits
- modexp vs gmpy2 gold (run 2, determinism): 6144/6144 exact match
- hipFFT double-FFT rounding margin holds on wave32 (gfx1100 fp64 full-rate): 0 mismatches
- CPU regression: test_paillier PASS, test_quantization PASS, test_full_chain PASS (max error 1.33e-09)
- Fork working tree: clean (only __pycache__ .pyc touched by running tests; no source change)
- No fork push (zero-churn follower validation; no code change needed)

Wave32 note: live path has no warp intrinsics (__shfl/__ballot/warpSize/__syncwarp absent),
no cub/thrust/curand. The <<<N,32>>> launches are independent-thread strided copies -- wave32
behavior is identical to wave64. Dead warp-NTT files remain excluded (no live path, wave64 hazard).

VERDICT: PASS -- port-ready -> completed
gfx1100 hipFFT Z2Z + Barrett reduction correct at wave32; matches gfx90a@6ef301f exactly.

## Validation 2026-06-04 (windows-gfx1151, gfx1151, AMD Radeon 8060S, TheRock 7.13)

validated_sha: 6ef301f3204579779bbaa1f32a466934f720903a

GPU arch: gfx1151 (AMD Radeon 8060S, RDNA3.5, wave32, TheRock ROCm 7.13 pip wheels)

No code change from gfx90a lead -- zero-churn follower validation, fork untouched at 6ef301f.

### Windows-specific build notes

On Windows the build_hip.sh Linux recipe needs two adaptations:
1. Remove `-fPIC` (unsupported on MSVC target; position-independent code is implicit for Windows DLLs).
2. Add a `.def` file to the linker so `init_gpu` and `run_modexp` are exported. On Windows with
   clang/MSVC-ABI, `extern "C"` functions in a DLL are NOT auto-exported (unlike Linux .so);
   `__declspec(dllexport)` is silently ignored in device-code TUs compiled with hipcc, so a
   `/DEF:lib_cufft.def` passed via `-Wl,/DEF:...` is the clean solution.
3. DLL runtime env: copy TheRock DLLs (amdhip64_7.dll, amd_comgr0713.dll, rocm_kpack.dll,
   hipfft.dll, rocfft.dll, hiprtc07013.dll, hiprtc-builtins07013.dll) from
   `_rocm_sdk_devel/bin/` + `_rocm_sdk_libraries_gfx1151/bin/` to the same directory as
   lib_cufft.dll. Windows loader searches the DLL's own directory first, so this ensures
   TheRock's runtime loads instead of the broken System32 Adrenalin amdhip64_7.dll. In the
   Python ctypes loader, call `os.add_dll_directory(dll_dir)` before loading.

The hipfft.dll and rocfft.dll from `_rocm_sdk_libraries_gfx1151/bin/` (the gfx1151-specific
package) contain the precompiled gfx1151 device kernels; no separate .kpack file needed.

Commands run:
```
# Build (agent_space/3p-admm-win/build_win.sh)
HIP_ARCH=gfx1151 bash utils/timeit.sh 3P-ADMM-PC2 compile -- \
  bash agent_space/3p-admm-win/build_win.sh
# -> lib_cufft.dll (150 KB, exports: init_gpu, run_modexp)

# modexp gold match
bash utils/timeit.sh 3P-ADMM-PC2 test -- \
  C:/Users/jdaily/AppData/Local/Programs/Python/Python313/python.exe \
  agent_space/3p-admm-win/validate_win.py

# CPU regression (PYTHONUTF8=1 works around upstream Chinese print() on cp1252 terminal)
PYTHONPATH=projects/3P-ADMM-PC2/src PYTHONUTF8=1 python crypto/test_paillier.py
PYTHONPATH=projects/3P-ADMM-PC2/src PYTHONUTF8=1 python crypto/test_quantization.py
PYTHONPATH=projects/3P-ADMM-PC2/src PYTHONUTF8=1 python crypto/test_full_chain.py
```

Results:
- Build gfx1151: PASS -- 1 benign warning (`--ld-path` unused during linking); DLL exports
  confirmed (dumpbin: init_gpu @ 0x28A0, run_modexp @ 0x28B0)
- modexp vs gmpy2 gold: 6144/6144 exact match
  n at 255/510/511/768/1022/1023/1535/2046 bits, 256/batch, 3 reps, varied m_bits
- hipFFT double-FFT rounding margin holds on gfx1151 (RDNA3.5, wave32, fp64 via f64 ALU):
  0 mismatches across all 6144 crypto-significant cases
- CPU regression: test_paillier PASS, test_quantization PASS, test_full_chain PASS
  (max error 1.33e-09, identical to gfx90a/gfx1100)
- No fork code change needed (zero-delta follower; same commit as gfx90a)

LOW-NUMERIC-RISK target confirmed: the gfx1151 RDNA3.5 FP-divergence class does NOT apply here
because the correctness gate is BIT-EXACT integer modexp against gmpy2 (not FP tolerance), and
the Z2Z double-FFT rounding margin holds at < 0.5 limb error even on gfx1151.

VERDICT: PASS -- port-ready -> completed

## Validation 2026-06-04 (windows-gfx1101, AMD Radeon PRO V710, RDNA3, ROCm 7.14 / TheRock)

validated_sha: 6ef301f3204579779bbaa1f32a466934f720903a (zero-churn follower; fork untouched)

GPU arch: gfx1101 (AMD Radeon PRO V710, RDNA3, wave32). Host = the gfx1101+gfx1201 Windows
workstation (memory windows-gfx1101-gfx1201-host); HIP_VISIBLE_DEVICES=0 pins gfx1101 (the
only visible device this process). TheRock pip ROCm SDK 7.14 (rocm-sdk-devel @ 60850-d34cbb64).

### Windows gfx1101 build recipe (REUSABLE; scripts in agent_space/3p-admm-win/, not committed)
Single ROCm root = the pip rocm-sdk-devel tree
`.../site-packages/_rocm_sdk_devel` (clang++, hipcc, hip cmake, hipfft.lib all present).
build_win.sh adapts gpu/build_hip.sh for Windows, same two adaptations the gfx1151 note found:
1. NO -fPIC (clang rejects it for the x86_64-pc-windows-msvc target; PIC is implicit on Windows).
2. Export init_gpu/run_modexp via a `.def` (`-Wl,/DEF:lib_cufft.def`) -- extern "C" funcs are
   not auto-exported from a Windows DLL.
Link `-L$ROCM/lib -lhipfft` (resolves hipfft.lib). hipcc all-clang, `--offload-arch=gfx1101`.
Output lib_cufft.dll (143 KB; only the benign `--ld-path unused` warning). PATH must include
`$ROCM/bin:$ROCM/lib/llvm/bin` so hipcc finds clang + lld-link.

### Windows DLL runtime (no per-arch library package needed)
There is NO _rocm_sdk_libraries_gfx1101 package on this host; the multi-arch `_rocm_sdk_libraries`
plus rocFFT runtime kernel generation cover gfx1101 (AMD_LOG_LEVEL=3 shows rocFFT RTC-compiling
its FFT kernels at run time). The validate harness calls `os.add_dll_directory()` for
`_rocm_sdk_core/bin`, `_rocm_sdk_devel/bin`, and `_rocm_sdk_libraries/bin` before loading
lib_cufft.dll, so TheRock's amdhip64_7.dll / amd_comgr.dll / rocm_kpack.dll / hiprtc0714.dll /
hipfft.dll / rocfft.dll load instead of the broken System32 Adrenalin amdhip64. gmpy2 was
pip-installed into the TheRock venv. The repo's crypto/paillier_gpu.py hardcodes
`/tmp/lib_cufft.so`; the harness monkeypatches `paillier_gpu._lib` to the built DLL.

### Result
```
# build (HIP_ARCH=gfx1101)
bash utils/timeit.sh 3P-ADMM-PC2 compile -- bash agent_space/3p-admm-win/build_win.sh
# modexp gold (HIP_VISIBLE_DEVICES=0 = gfx1101)
HIP_VISIBLE_DEVICES=0 .../venv/Scripts/python.exe agent_space/3p-admm-win/validate_win.py
```
- Build gfx1101: PASS (lib_cufft.dll, exports init_gpu/run_modexp).
- modexp vs gmpy2 gold: 6144/6144 EXACT match -- n at 255/510/511/768/1022/1023/1535/2046 bits,
  256/batch, 3 reps/size, varied m_bits (64 / nb/2 / nb-1), g < n precondition honored.
- hipFFT double-FFT rounding margin holds on gfx1101 (RDNA3, wave32, fp64): 0 mismatches.
- Native gfx1101 dispatch: only gfx1101 visible (HIP_VISIBLE_DEVICES=0), hipGetDevice success,
  rocFFT RTC-compiled device kernels; bit-exact GPU results (the .cu kernels are GPU-only, so a
  silent CPU fallback is impossible). No fork code change (zero-churn follower; same 6ef301f).

VERDICT: PASS -- port-ready -> completed. gfx1101 matches gfx90a/gfx1100/gfx1151 exactly.
This is the first real GPU validation on the new Windows host -- it establishes the reusable
Windows ROCm build + DLL-runtime recipe for the rest of the gfx1101/gfx1201 sweep.

## Validation 2026-06-04 (windows-gfx1201, AMD Radeon RX 9070 XT, RDNA4, ROCm 7.14 / TheRock)

validated_sha: 6ef301f3204579779bbaa1f32a466934f720903a (zero-churn follower; fork untouched)

GPU arch: gfx1201 (AMD Radeon RX 9070 XT, RDNA4, wave32). HIP_VISIBLE_DEVICES=1 pins gfx1201
(detect_arch confirms device 1 = gfx1201). Same host/recipe as the gfx1101 run above; only
`--offload-arch=gfx1201` and the device pin change.
```
HIP_ARCH=gfx1201 OUT=.../lib_cufft_gfx1201.dll bash agent_space/3p-admm-win/build_win.sh
HIP_VISIBLE_DEVICES=1 LIB_CUFFT=.../lib_cufft_gfx1201.dll .../python.exe agent_space/3p-admm-win/validate_win.py
```
- Build gfx1201: PASS (lib_cufft_gfx1201.dll, exports init_gpu/run_modexp).
- modexp vs gmpy2 gold: 6144/6144 EXACT match (same 8 moduli x 3 reps x 256/batch as gfx1101).
- hipFFT double-FFT rounding margin holds on gfx1201 (RDNA4, wave32, fp64): 0 mismatches.
- No fork code change (zero-churn follower; same 6ef301f). RDNA4 FP behavior irrelevant -- the
  gate is bit-exact integer modexp, no wave-size surface in the live path.

VERDICT: PASS -- port-ready -> completed. Both Windows archs (gfx1101 + gfx1201) now match
gfx90a/gfx1100/gfx1151 exactly; all five platforms terminal -> PR-ready.
