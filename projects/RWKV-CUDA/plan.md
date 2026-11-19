# RWKV-CUDA -- ROCm/HIP port plan (lead: linux-gfx90a, MI250X, ROCm 7.2.1)

## Project
- Name: RWKV-CUDA
- Upstream: https://github.com/BlinkDL/RWKV-CUDA
- Default branch: main
- HEAD at clone: 9b17d5d ("+ rwkv7 state-passing (infctx) kernel")
- What it is: BlinkDL's reference/optimization playground for the RWKV time-mixing
  recurrence (WKV) CUDA kernels -- a set of standalone subprojects, each a `.cu`
  kernel + a tiny torch extension wrapper (`*_op.cpp`) + a `run.py` harness that
  JIT-builds the extension and checks correctness/gradients against a PyTorch
  reference and benchmarks speed. Covers RWKV-4 (wkv, depthwise_conv1d), RWKV-5
  (wkv5, wkv5_bf16, wkv5a), RWKV-6 (wkv6, wkv6_state), RWKV-7 (rwkv7_fast_fused).

## Existing AMD support
None. No HIP/ROCm/gfx references anywhere in the tree (grep clean). This is a
fresh CUDA-to-HIP port and it adds value: there is no HIP path today.
Decision: PROCEED with a mechanical Strategy-B port (correctness-first).

These are small, custom, hand-written WKV recurrence kernels -- NOT CUTLASS/CuTe,
no wgmma, no warp specialization, no tensor-core MMA. The compute is an
elementwise linear-attention scan with per-thread/per-block state in shared
memory or registers. A mechanical HIP translation is the right call; there is no
AMD-native rewrite warranted for correctness validation. (A later MFMA/CK perf
pass is out of scope for the MOAT correctness gate and not needed here.)

## Build classification: torch-extension (Strategy B)
Evidence:
- No CMakeLists.txt / Makefile anywhere in the tree.
- Every subproject builds its kernel through `torch.utils.cpp_extension.load(...)`
  (JIT). Examples: wkv5/run.py:418-420 and :466-467; wkv5_bf16/run.py:42;
  wkv6/run.py:70,126; wkv6_state/run.py:78; wkv5a/run.py:98; wkv/run.py:87;
  depthwise_conv1d/run.py:72; rwkv7_fast_fused/rwkv7_cuda_benchmark*.py:75,104.
- Wrappers `#include <torch/extension.h>` and register via `PYBIND11_MODULE` and/or
  `TORCH_LIBRARY` (e.g. wkv5/cuda/wkv5_op.cpp; rwkv7_fast_fused/cuda/*.cpp).
Conclusion: pytorch extension -> Strategy B. ext_type = "torch-extension".

## Port strategy: B (torch hipify), with the real work in the run.py build flags
On a ROCm torch, `cpp_extension.load` auto-runs `torch.utils.hipify` on the `.cu`/
`.cpp` sources and links the HIP runtime -- the kernel SOURCES need essentially no
hand edits (no compat header, no manual symbol renames; see "CUDA surface" -- there
are no warp intrinsics, no textures, no CUDA libraries). The actual blocker is the
nvcc-only build flags baked into each `run.py` harness, which hipcc rejects.

The minimal, low-churn fix: make the `extra_cuda_cflags` lists ROCm-aware in each
harness (detect `torch.version.hip`), dropping the nvcc/PTX-only and MSVC-only
flags on HIP and keeping the portable ones. Source `.cu`/`.cpp` stay in CUDA
spelling so the NVIDIA path is byte-for-byte unchanged and hipify owns the
translation. Confirmed available on this box: torch 2.13 + HIP 7.2.53211.

## CUDA surface inventory
- Kernels: per-subproject `kernel_forward` / `kernel_backward` (and rwkv7
  `forward_kernel`/`backward_kernel`). Launch shape is the WKV idiom:
  `<<<dim3(B*H), dim3(N)>>>` (N = head size), one thread per channel-within-head,
  `N x N` (or `N`) recurrence state in `__shared__` or registers, advanced over T
  with `__syncthreads()` between timesteps. rwkv7 uses `<<<dim3(H,B),dim3(N)>>>`
  with `__launch_bounds__(N,2)`.
- `__global__`/`__device__`: standard; templated on element type `F`. Hipify-clean.
- Warp intrinsics: NONE. No `__shfl*`, `__ballot`, `__any/__all`, `__activemask`,
  `__syncwarp`, no warp-synchronous reductions. (grep confirmed across all .cu.)
- Hardcoded 32: the only `32`s are host-side block-size caps for tiny test configs
  (`dim3 threadsPerBlock(min(SIZE,32))` in wkv5 v1/v1a/v1b/v1c/v2/v3/ref, wkv v2) --
  a launch-size clamp, NOT a warp-width assumption; correct on wave64. The
  production configs launch `dim3(N)` with N=head size (64), independent of warp.
- Atomics: `atomicAdd` on float (gradient accumulation into gw/gu/y in several
  v1/v2/ref variants). 1:1 in HIP; non-deterministic accumulation order is expected
  and already CUDA behavior -- validate by err-ratio, not bitwise (see Test plan).
- fp16/bf16: wkv5_bf16, wkv5a, wkv6, wkv6_state use `at::BFloat16` (ATen type, HIP
  torch supplies it) and compute internally in `float` -- portable. rwkv7 uses raw
  `#include <cuda_bf16.h>`, `__nv_bfloat16`, `__float2bfloat16_rn`,
  `__bfloat162float` under `#ifndef _FP32_`; hipify maps the header and the type,
  and HIP provides the conversion intrinsics by the same spelling
  (amd_detail/amd_hip_bf16.h). rwkv7 also has a `_FP32_` build path that drops bf16
  entirely -- the clean first-bringup slice.
- `__expf`: used in wkv6_state and rwkv7 (`__expf(-__expf(...))`). HIP provides
  `__expf`; not a rename. With `--use_fast_math`/default HIP contraction this is a
  fast-math path on both vendors (the kernels are explicitly fast-math).
- Textures/surfaces: NONE. cuBLAS/cuFFT/cuRAND/cuSPARSE/Thrust/CUB: NONE.
  pinned/managed memory: NONE.
- Streams/events/explicit allocs: a few non-default variants only --
  wkv6/cuda/wkv6_cuda_v1a.cu uses `cudaMalloc`; wkv5_bf16/cuda/wkv5_cuda_v3.cu uses
  `cudaEvent*` for internal timing. Both hipify 1:1; neither is the default kernel
  version selected by its run.py, so neither is on the primary validation path.

## Risk list
- BUILD FLAGS (primary, certain): every harness passes nvcc-only flags hipcc does
  not accept: `-res-usage`, `-Xptxas -O3`, `--extra-device-vectorization`, and
  (wkv/run.py, depthwise_conv1d/run.py) the MSVC-only `extra_cflags=['/wd4624']`.
  `--use_fast_math` and `-O3` are accepted by hipcc. Fix per harness: build the
  cuda-cflag list conditionally on `torch.version.hip` (keep `-O3`, `--use_fast_math`,
  and the `-D...` macro defines; drop `-res-usage`, `-Xptxas -O3`,
  `--extra-device-vectorization`; drop the `/wd...` MSVC cflag on non-Windows).
- bf16 intrinsics in rwkv7 (low): `__nv_bfloat16` + `__float2bfloat16_rn` /
  `__bfloat162float` via `cuda_bf16.h`. hipify maps the header and type; HIP supplies
  the conversion intrinsics. Validate the `_FP32_` path first to decouple bf16 from
  the recurrence correctness, then the bf16 path.
- warpSize 32-vs-64: LOW. No warp-level code exists, so wave64 vs wave32 does not
  affect correctness here -- the kernels are `__syncthreads()`-synchronized over a
  block of N threads with no implicit warp-lockstep assumption. (This also means the
  gfx1100/gfx1151 followers should pass with no delta; see follower note.)
- Empty/partial kernels in some default versions: wkv5 default `CUDA_KERNEL_VERSION
  ='1d'` has an EMPTY `kernel_backward` (wkv5_cuda_v1d.cu:68-74) and empty
  `cuda_backward`. So for wkv5 the `correctness`/`correctness_more` (forward-only)
  jobs are the v1d validatable slice; the `backward` job must select a version that
  implements backward (e.g. v1/v1b/v1e have real backward kernels). Pick the slice
  accordingly rather than assuming every job works for every default version.
- atomicAdd ordering / fast-math: results are NOT bitwise-reproducible vs the torch
  reference even on CUDA; the harnesses already judge by `get_err_ratio` and
  `torch.allclose`. Use the same err-ratio bar on AMD; do not impose bitwise equality.
- No CUTLASS / no library swaps / no textures: nothing from those fault classes
  applies. The 256B-pitch, texture-linear-filter, layered-array, rule-of-five, and
  OOB-neighbor classes are all N/A (no textures, no resource handles, no stencils).

## File-by-file change list (minimal; sources stay CUDA-spelled)
Source kernels/wrappers (`*.cu`, `*_op.cpp`): expected ZERO hand edits -- hipify
translates them. (If the rwkv7 bf16 intrinsics surprise the build, the only possible
touch is a `USE_ROCM`-guarded include/typedef in rwkv7_clampw.{cu,cpp}; treat as
contingency, not planned.)

Harness build flags (the planned edits), make `extra_cuda_cflags` HIP-aware:
- wkv5/run.py (lines ~420 and ~467)
- wkv5_bf16/run.py (~42)
- wkv5a/run.py (~98)
- wkv6/run.py (~70 and ~126)
- wkv6_state/run.py (~78)
- wkv/run.py (~87; also drop the MSVC `/wd4624` extra_cflags on non-Windows)
- depthwise_conv1d/run.py (~72; same MSVC cflag note)
- rwkv7_fast_fused/rwkv7_cuda_benchmark.py (~74/75 bf16, ~103/104 fp32)
- rwkv7_fast_fused/rwkv7_cuda_benchmark_state.py (~74, ~106)
- rwkv7_fast_fused/rwkv7_cuda_benchmark_state_passing.py (~74, ~107)
Implementation note: factor the flag selection (e.g. a small `is_hip =
torch.version.hip is not None` branch building the cflag list) rather than editing
each string in place, to keep each harness readable. Keep all `-D...` defines.

## Build commands (gfx90a)
No configure step -- JIT build via the harness. From each subproject dir, with a
ROCm torch active and the assigned GPU visible:
  export PYTORCH_ROCM_ARCH=gfx90a            # so hipcc targets the lead arch
  export HIP_VISIBLE_DEVICES=<one GCD>        # hold a single GCD only during a GPU run
  cd projects/RWKV-CUDA/src/wkv5 && python run.py correctness
The `load(... verbose=True)` build line is printed; it must show hipcc/`-x hip`
and link `amdhip64`/`torch_hip`. A stale JIT cache after editing flags: clear
~/.cache/torch_extensions (or use a fresh TORCH_EXTENSIONS_DIR) before rebuilding.

## Test plan (real GPU gate)
The repo IS its own test suite -- each run.py compares the CUDA kernel against a
PyTorch/numpy reference (forward result and backward gradients) and reports
`torch.allclose` + `get_err_ratio`. GPU-validatable slice (run on gfx90a, judged by
err-ratio within the harness's own allclose tolerance, NOT bitwise):
- wkv5:  `python run.py correctness` (fwd, tiny), then `python run.py correctness_more`
  (fwd vs cuda-ref at B8/T4096/C4096/H64). For backward, set CUDA_KERNEL_VERSION to a
  version with a real backward (v1/v1b/v1e) and run `python run.py backward`.
- wkv5_bf16: `python run.py correctness` then `correctness_more`; this is the bf16 +
  full fwd/bwd kernel (v1b has both) -- the key bf16 correctness gate.
- wkv6 / wkv6_state: `python run.py correctness` (+ backward where the default version
  implements it).
- rwkv7_fast_fused: run `rwkv7_cuda_benchmark.py` first with DTYPE=float (the `_FP32_`
  path) to validate the chunked recurrence fwd+bwd decoupled from bf16, then with
  DTYPE=bfloat16. Same for `_state` and `_statepassing` harnesses.
- wkv (RWKV-4) and depthwise_conv1d: `python run.py` -- their built-in correctness +
  gradient checks vs the pytorch reference.
Pass bar: each harness prints `correct = True` (allclose) and a small err ratio
(consistent with the CUDA baseline) for forward AND the gradients it checks.
Non-GPU regression set: none -- there are no CPU-only unit tests in this repo; the
only "non-GPU" code is the python reference inside each run.py, which is exercised as
the gold for the GPU comparison. So there is nothing to regress separately; the gate
is the GPU correctness/gradient comparison above.

## Follower platforms (preview, not a delta-plan yet)
Because there is no warp-level code, the wave32 RDNA followers (gfx1100, gfx1151)
are expected to pass with the SAME source and the same HIP-aware flags -- only
`PYTORCH_ROCM_ARCH` changes. Validate first, delta-plan only on failure. (A
delta-plan section will be appended on demand if a follower regresses.)

## Open questions
- None blocking. The single decision is cosmetic: implement the per-harness flag
  selection as a small `is_hip` branch (preferred for readability) vs. a shared
  helper. Leave to the porter.
