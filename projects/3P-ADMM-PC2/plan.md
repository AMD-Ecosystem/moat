# Port plan: 3P-ADMM-PC2 (linux-gfx90a lead)

## Project
- Name: 3P-ADMM-PC2
- Upstream: https://github.com/Samarvivian/3P-ADMM-PC2
- Default branch: `features` (per upstream.json; README also lists `releases`, `master`)
- What it is: a privacy-preserving distributed ADMM framework for LASSO. CPU does the
  optimization (Paillier homomorphic encryption + quantized ADMM, gmpy2). The GPU's ONLY
  job is to accelerate the Paillier modular exponentiation `g^m mod n` over a batch of
  big integers, via large-integer polynomial multiplication. The ADMM math itself
  (`admm/*.py`, `protocol/*.py`) is pure Python/numpy/gmpy2 -- there are NO GPU
  linear-algebra kernels, no sparse/dense matvec, no cuBLAS/cuSPARSE/cuSOLVER. The GPU
  surface is crypto big-integer arithmetic.

## Existing AMD support
- None. NVIDIA-only: README targets CUDA 12.1, `-arch=sm_86` (RTX A4000/A2000), links
  `-lcudart -lcufft`. No HIP path, no OpenCL/Vulkan/SYCL path. A ROCm/HIP port of the
  CUDA ModExp kernel adds clear value.
- Decision: PROCEED with a mechanical CUDA->HIP port of the live GPU path. This is NOT a
  CUTLASS/Hopper/perf-rewrite case (confirmed: no CUTLASS, no CuTe, no wgmma, no warp
  specialization anywhere). Correctness-first mechanical port is the right and sufficient
  first step.

## Build classification: ad-hoc nvcc (NOT CMake, NOT a torch extension, NOT a Makefile)
- Evidence: `find` shows no CMakeLists.txt, no Makefile, no setup.py, no pyproject.toml.
  `requirements.txt` is `fastapi`, `uvicorn`, `numpy` only -- no torch, no pycuda listed.
- The build is hand-rolled `nvcc` commands embedded in README.md sec. "1. 每次开机重新编译GPU库"
  (lines 161-198): compile `gpu/cufft_modexp.cu` with a generated C wrapper `wr_cufft.cu`
  (exposing `init_gpu(int)` / `run_modexp(...)`), device-link, then `g++ -shared` into
  `/tmp/lib_cufft.so` linking `-lcuda -lcudart -lcufft`.
- The Python side (`crypto/paillier_gpu.py`, `protocol/edge_crt_helper.py`) loads
  `/tmp/lib_cufft.so` by absolute path via `ctypes.CDLL` and calls `run_modexp`/`init_gpu`.
- ext_type recorded as `nvcc-shared` (custom; closest to Strategy A, "only the `.cu` sees
  the GPU toolchain"). Set in upstream.json + status.json.

## Port strategy: A-flavored compat header + a HIP build recipe (no CMake to gate)
Rationale: there is no CMake `enable_language(HIP)` to flip and no torch hipify to lean on.
The minimal-footprint analogue of Strategy A here is:
1. Add ONE compat header `gpu/cuda_to_hip.h` that, under `__HIP_PLATFORM_AMD__`/`USE_HIP`,
   `#include <hip/hip_runtime.h>` + `<hipfft/hipfft.h>` and aliases the exact CUDA symbols
   the live kernel uses to their HIP spellings; else `#include <cuda_runtime.h>`+`<cufft.h>`.
   Include `<cstring>`/`<cstdlib>` BEFORE the HIP runtime (PORTING_GUIDE gpuRIR lesson:
   inside a .cu compiled as HIP, host memcpy/memset/free/calloc can otherwise resolve to
   HIP `__device__` overloads). `cufft_modexp.cu` uses `calloc`/`free`/`round` on the host.
2. `#include "cuda_to_hip.h"` at the top of `gpu/cufft_modexp.cu` in place of the bare
   `cuda_runtime.h`/`cufft.h` includes. Keep all `cudaXxx`/`cufftXxx` spellings in the body;
   the header aliases them. This keeps the NVIDIA build byte-identical.
3. Provide a ROCm build recipe alongside the README's nvcc one: `hipcc` compiling the same
   `cufft_modexp.cu` + wrapper, linking `-lhipfft`, producing `/tmp/lib_cufft.so` with the
   SAME `init_gpu`/`run_modexp` C ABI so the Python ctypes side is UNCHANGED. Add it as a
   small `gpu/build_hip.sh` (and document in notes.md). Default arch from an env/arg, never
   a literal (so gfx1100/gfx1151 followers reuse it with only an arch change).
4. The hardcoded `/mnt/3p-admm-pc2/...` absolute paths in the Python (lib path, kernel
   source paths) are an upstream portability wart, not a HIP issue; for our build+validate
   we point the loader at our actual `/tmp/lib_cufft.so` path. Do not refactor upstream's
   path scheme as part of the port (out of scope; flag in notes).

## CUDA surface inventory
LIVE GPU code (what actually runs in the protocol), `gpu/cufft_modexp.cu` (204 lines):
- Library: cuFFT. `cufftHandle`, `cufftPlanMany(...CUFFT_Z2Z...)`, `cufftExecZ2Z` (FORWARD/
  INVERSE), `cufftDestroy`, `cufftDoubleComplex`. -> hipFFT: `hipfftHandle`,
  `hipfftPlanMany`, `hipfftExecZ2Z` (`HIPFFT_FORWARD`/`HIPFFT_BACKWARD`), `hipfftDestroy`,
  `hipfftDoubleComplex`. APIs are ~1:1; watch the FORWARD/INVERSE enum spelling
  (CUFFT_INVERSE -> HIPFFT_BACKWARD) and that hipFFT complex type aliases `double2`.
- Runtime: `cudaMalloc`, `cudaFree`, `cudaMemcpy` (H2D/D2H), `cudaMemcpyHostToDevice`,
  `cudaMemcpyDeviceToHost`. -> `hipMalloc`/`hipFree`/`hipMemcpy` 1:1.
- Kernels (all plain, no intrinsics): `cmul`, `norm_round`, `load_complex`, `carry_prop`,
  `shr_kernel`, `sub_correct`, `extract_bit`, `cond_copy`. Launches use `dim3`/blockDim
  32 or 256; `cond_copy<<<N,32>>>` is a strided memcpy (`for i=j;i<LEN;i+=blockDim.x`),
  NOT a warp-collective. NO `__shfl*`, NO `__ballot`, NO `warpSize`, NO `__syncwarp`, NO
  cooperative groups, NO textures/surfaces, NO atomics, NO cub/thrust/curand. Confirmed by
  grep (none).
- Host math: `round`, `calloc`, `free` (cstdlib/cmath; see include-order note above).

SECONDARY GPU path, `gpu/gpu_modexp.py` -> PyCUDA JITs `gpu/modexp.cu` / `gpu/modexp_v2.cu`:
- `import pycuda.autoinit / pycuda.driver / pycuda.compiler.SourceModule`. PyCUDA is NOT in
  requirements.txt and this module is a standalone `__main__` self-test, NOT called by the
  protocol/experiments. `modexp.cu`: one-thread-per-task scalar bigint, no intrinsics.
  `modexp_v2.cu`: 128-thread block, all `__syncthreads()` (block barrier, wave-size
  agnostic), no warp intrinsics. Hardcodes `/mnt/3p-admm-pc2/gpu/*.cu` source paths.
  DECISION: out of scope for the lead port. PyCUDA-on-ROCm is non-standard and this path is
  not exercised by any validatable workload. Note it; do not port unless validation needs it.

DEAD/experimental files (in `gpu/`, referenced by NOTHING in the repo -- no Python import,
no `<<<...>>>` launch site, not built by the README): `reg_ntt_modexp.cu`,
`reg_ntt_modexp_v2.cu`, `reg_ntt_modexp_v3.cu`, `ntt_modexp.cu`, `ntt_modexp_v2.cu`,
`ntt_modexp_fast.cu`, `ntt_funcs.cu`, `ntt_kernel2.cu`, `ntt_wrapper.cu`,
`ntt_wrapper2.cu`, `modexp_final.cu`, `cufft_modexp` siblings. The `reg_ntt_*` and
`ntt_funcs` files DO contain `__shfl_xor_sync(0xffffffff, ..., delta)` warp-NTT butterflies
launched (conceptually) with blockDim=32 (one warp/task) and `for(i=lane;i<...;i+=32)`
strides -- i.e. classic wave64 hazards. But they are NOT in any live path. DECISION: DO NOT
port the dead files. Porting them would be effort against unused code AND would expose the
wave64 fault class for no validatable benefit. If a future maintainer wants the register-NTT
path on AMD, that is a separate task (see Open questions).

## Risk list
- LOW overall: the live kernel is the easy class (cuFFT + scalar bigint, no warp ops).
- hipFFT enum/type drift: `CUFFT_INVERSE` -> `HIPFFT_BACKWARD`, `CUFFT_FORWARD` ->
  `HIPFFT_FORWARD`, `cufftDoubleComplex` -> `hipfftDoubleComplex`. Verify `hipfftPlanMany`
  batched (the `N` batch arg) semantics match cuFFT's (they do in ROCm 7.2.x). Alias these
  in the compat header so the body is untouched.
- hipFFT precision: this is a NUMERIC-CORRECTNESS-VIA-FFT bigint multiply. The kernel does
  Z2Z (double-precision) FFT then `round()` to recover exact integer coefficients
  (`norm_round` + `carry_prop`). gfx90a has full-rate fp64, so double FFT round-off should
  stay within the < 0.5 rounding margin the algorithm relies on, same as CUDA. BUT hipFFT's
  twiddle/rounding differs from cuFFT's bit-for-bit, so the recovered coefficients could
  occasionally round the wrong way if the design margin is thin. This is the one real risk
  to watch in validation -- check `g^m mod n` against gmpy2 over many random inputs, not
  just one. (No bit-exactness is required vs CUDA; only that the final integer matches the
  gmpy2 gold.)
- Wave size: NOT exposed in the live path (no warp intrinsics, no hardcoded-32 warp logic;
  `<<<N,32>>>` blocks are independent-thread strided copies). The MULTI-ARCH per-arch
  warpSize standard therefore needs NO code change for the lead port: there is no device
  warp constant to set and no host warpSize query to add. The follower deltas (gfx1100/
  gfx1151, wave32) are expected to be no-op revalidations (same .so rebuilt for the arch).
  Flag the dead `reg_ntt_*` shfl files only so a later contributor knows they would need the
  `__GFX9__`->64 / RDNA->32 + width-aware ballot treatment if ever revived.
- Build wart (not a HIP bug): hardcoded `/mnt/3p-admm-pc2` absolute paths in Python; `/tmp`
  non-persistent .so. We control these for our build; do not refactor upstream's scheme.
- Include order: host `calloc`/`free`/`round` in `cufft_modexp.cu` -- put `<cstring>`/
  `<cstdlib>`/`<cmath>` before `<hip/hip_runtime.h>` in the compat header (gpuRIR lesson).
- No rule-of-five/texture/OOB/pitch/atomicMin classes apply (no such constructs in the live
  kernel). `sub_correct`/`carry_prop`/`shr_kernel` index within per-task strided buffers;
  bounds are guarded by `if(t>=N) return` and `src<xR_len` style checks -- no edge OOB.

## File-by-file change list
- ADD `gpu/cuda_to_hip.h` -- the single compat header (HIP includes + cuda*/cufft* aliases),
  cstdlib/cstring/cmath included first. Only the symbols cufft_modexp.cu uses.
- EDIT `gpu/cufft_modexp.cu` -- replace the two includes (`<cuda_runtime.h>`, `<cufft.h>`)
  with `#include "cuda_to_hip.h"`. No body changes expected (aliases cover it). If hipFFT
  needs `HIPFFT_BACKWARD` where the body has `CUFFT_INVERSE`, alias `CUFFT_INVERSE` ->
  `HIPFFT_BACKWARD` in the header rather than editing the body.
- ADD `gpu/build_hip.sh` -- hipcc recipe mirroring the README nvcc steps, arch from
  `${HIP_ARCH:-gfx90a}`, links `-lhipfft`, emits `/tmp/lib_cufft.so` with the same C ABI.
- (notes.md) record the build + the `/mnt` path caveat + the dead-file decision.
- DO NOT touch: any Python, the dead `gpu/*.cu` files, `admm/`, `protocol/`, `crypto/`
  (except possibly a thin validate harness in agent_space, not committed to the fork).

## Build commands (gfx90a)
ROCm build (analogue of README sec.1, via build_hip.sh):
```
hipcc -O2 --offload-arch=gfx90a -fPIC -c gpu/cufft_modexp.cu -o /tmp/cufft_modexp.o
# generated wrapper wr_cufft.cu exposing init_gpu/run_modexp (same as README)
hipcc -O2 --offload-arch=gfx90a -fPIC -c /tmp/wr_cufft.cu -o /tmp/wr_cufft.o
hipcc -shared -fPIC /tmp/cufft_modexp.o /tmp/wr_cufft.o -lhipfft -o /tmp/lib_cufft.so
```
(hipcc does host+device link in one step; no separate `-dlink` needed for a non-RDC build.
If RDC is wanted, add `-fgpu-rdc` to compiles and a `--hip-link` device-link step.)
Followers: same script with `HIP_ARCH=gfx1100` / `gfx1151`.

## Test plan
GPU-validatable slice (real GPU, the load-bearing correctness gate):
- The GPU's contract is `out[i] == (g^m mod n)` over a batch. Validate the ported
  `/tmp/lib_cufft.so` by calling `run_modexp` (through `crypto/paillier_gpu.py:gpu_batch_modexp`
  or `protocol/edge_crt_helper.py:gpu_modexp_diff_g`) on many random (g, m, n=p*q) triples
  and comparing against `gmpy2.powmod(g, m, n)` -- the exact CPU gold the repo already uses
  in `gpu/gpu_modexp.py:__main__` and throughout paillier_gpu.py. Pass = all batches match
  gmpy2 (with the final `% n`). Use realistic Paillier sizes (n^2 up to ~2048-bit, LEN=128
  base-65536 limbs) AND small sizes; vary m_bits. A single-input check is insufficient
  given the hipFFT rounding-margin risk -- run a few thousand.
- Higher-level GPU functional check: `crypto/paillier_gpu.py:encrypt_batch_gpu` /
  `encrypt_batch_gpu_fast` produce ciphertexts; decrypt with the CPU `crypto/paillier.py`
  and confirm round-trip equals the plaintext batch. This exercises the GPU g^m path inside
  the real encryption routine. Buildable as a small local harness (agent_space), not
  committed to the fork.
Non-GPU regression set (must not regress; pure CPU, runnable as-is up to the /mnt path):
- `crypto/test_paillier.py` (keygen/enc/dec/homomorphic add+mul), `crypto/test_quantization.py`
  (gamma1/gamma2 quantization), `crypto/test_full_chain.py`. These are CPU/gmpy2 only and
  the port does not touch them; run them to confirm no collateral breakage.
- NOT runnable in CI here (need multi-node SSH / matpool cloud, out of scope): the
  `experiments/test_distributed_pc2.py`, `test_large_scale.py`, `test_cen_vs_dis.py`,
  `monitor_gpu.py`, and `protocol/master_node.py` end-to-end protocol. The GPU-slice +
  encrypt round-trip above is the representative validatable proxy for the GPU work these
  drive.

## Open questions
- Does upstream want the dead `reg_ntt_*` register-NTT variants ported too? They are the
  faster-claimed path in the README perf table but are not wired into the live protocol and
  carry the wave64 warp-NTT hazard. Lead port deliberately excludes them; revisit only if
  the user/upstream asks (would be a separate AMD-native or width-aware effort).
- hipFFT batched-plan numeric margin: confirm in validation that double-FFT bigint multiply
  recovers exact limbs across the full input range on gfx90a; if a thin-margin miss appears,
  options are a larger FFT length or a wider integer accumulation -- but expect it to pass
  given fp64 parity.
- PyInstaller `dist/3P-ADMM-PC2` binary and the FastAPI web app are NVIDIA/x86 packaging
  artifacts; not part of the GPU port.
