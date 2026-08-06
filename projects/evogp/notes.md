# evogp notes

## Port summary (linux-gfx90a)
Strategy B (torch CUDAExtension + build-time hipify). Single `evogp.evogp_cuda`
op library over four .cu files. Fork: https://github.com/AMD-Ecosystem/evogp,
branch `moat-port`, HEAD ba4fa7e. Validated on gfx90a (MI250X, wave64), ROCm
7.2, torch 2.13a (hip 7.2.53211).

## Build (gfx90a)
```
export HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx90a
cd src
rm -rf build src/evogp.egg-info src/evogp/*.so src/evogp/hip   # clean stale hipify mirror after any source edit
pip install -e . --no-build-isolation
```
Requires a ROCm PyTorch active (confirm `torch.version.hip` is non-None).

## Validation (real GPU gate)
```
export HIP_VISIBLE_DEVICES=0
python -m evogp.sr_test     # 1000-pop SR, 100 gens, manual_seed(0)
```
PASS: completes with no HIP fault; best fitness (negated MSE) converges
monotonically from -0.25 (gen 0) to -0.018 (gen 99), finite throughout. Two
fixed-seed runs gave the identical final best (-0.0179). AMD_LOG_LEVEL=3 shows
native `amdgcn-amd-amdhsa--gfx90a` code objects dispatching.

## Changes made (4 fault classes hit; 2 build-only, 1 correctness)
1. setup.py: gate the device-compiler ("nvcc" key) flag list on
   `torch.version.hip`. The nvcc/ptxas-only flags (--ptxas-options, -Xptxas,
   -lineinfo, -maxrregcount, -lcudart, --expt-relaxed-constexpr, -use_fast_math)
   reach hipcc verbatim (hipify rewrites source, not compile args) and fail/are
   meaningless. Drop them on ROCm, keep -O3.
2. kernel.h: `#include "device_launch_parameters.h"` is a CUDA-toolkit-only
   header. It is quoted, so hipify treats it as a local header and does NOT
   strip/translate it -> `fatal error: file not found` on hipcc. Guarded behind
   `#ifndef USE_ROCM`; hip_runtime.h (from hipified `cuda_runtime.h`) already
   supplies blockIdx/threadIdx/etc.
3. Correctness (the real bug): per-thread `alloca` scratch corruption on
   gfx90a. The tree-interpreter kernels (treeGPEvalKernel,
   treeGPRegressionFitnessKernel x3 variants, treeGPGenerate, _gpTreeReplace)
   allocated ~8 KB per-thread working stacks via `alloca` (dynamic device
   scratch). On gfx90a these dynamic allocations alias across concurrently
   resident wavefronts: results were CORRECT up to ~128 threads in flight but
   wrong at/above ~200 (tree_evaluate returned garbage; tree_SR_fitness tripped
   the interpreter's `assert(top == 1)` and aborted with
   HSA_STATUS_ERROR_EXCEPTION). Fix: replace every `alloca(MAX_STACK * ...)`
   with a fixed-size local array (`float stack[MAX_STACK]` etc.), same size as
   before. clang gives this a stable per-lane scratch frame. Arch-unified: no
   wave-width dependence, identical on CUDA / wave32 / wave64.

## Gotchas / for followers (gfx1100, gfx1151)
- The alloca->local-array fix is the load-bearing change. If a follower
  regresses, it is NOT wave width (there are zero warp intrinsics, zero
  hardcoded 32, and the fitness reduction is fully `__syncthreads()`-guarded
  with the warp-synchronous tail commented out). Suspect scratch sizing instead.
- The `top == 1` device assert in _treeGPEvalByStack* is the canary: if it
  fires, the tree data feeding eval is corrupt (almost certainly scratch
  aliasing again), not a logic bug -- the generate kernel produces structurally
  valid trees (verified: 0 malformed over 1000).
- Diagnostic trick that localized this: eval the SAME forest at increasing pop
  sizes and compare tree[0] against a CPU reference. Correct below the
  occupancy threshold, wrong above it -> scratch corruption, not a data bug.
- Stale hipify mirror: `src/evogp/hip/` is regenerated each build. It is now in
  .gitignore. After ANY source edit, `rm -rf build src/evogp/hip` before
  rebuilding or you compile stale .hip files.
- `thrust::random` (taus88) works as a drop-in under rocThrust; no source edit
  needed. No cub/curand/cuBLAS/textures/CUTLASS anywhere.

## Pre-existing upstream test bug (NOT a port issue)
`test/test_bind_success.py` passes a 24-element `roulette_funcs` tensor, but the
binding (torch_wrapper.cu) correctly requires `Function::END` = 29 entries
(the Function enum grew). The op rejects it with "roulette_funcs must have
shape [29], but got shape [24]" -- identical on CUDA. Did not patch upstream's
test; the authoritative GPU gate is sr_test, which passes. If you want the
per-op probe to run, the test's roulette tensor needs padding to 29.

## Actions disabled on fork
`gh api -X PUT repos/AMD-Ecosystem/evogp/actions/permissions -F enabled=false` run
after fork.

## Validation 2026-06-02 (linux-gfx90a, validator)

GPU: AMD Instinct MI250X (gfx90a, GCD 0), ROCm 7.2, torch 2.13a (hip 7.2.53211).
Fork: AMD-Ecosystem/evogp @ moat-port, SHA ba4fa7e6656fbaaf0a42eacca215e543b1c9a2e0.

Build (clean from committed source):
```
export HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx90a
rm -rf build src/evogp.egg-info src/evogp/*.so src/evogp/hip
pip install -e /var/lib/jenkins/moat/projects/evogp/src --no-build-isolation
# Completed in ~53s; .so contains amdgcn-amd-amdhsa--gfx90a code objects (confirmed via strings)
```

GPU gate (run twice, fixed seed):
```
export HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx90a
python -m evogp.sr_test
```

Run 1: Gen 0 best -0.2511 -> Gen 99 best -0.0179; exit 0; no HIP fault; no assert(top==1).
Run 2: Gen 0 best -0.2511 -> Gen 99 best -0.0179; exit 0; identical -- deterministic PASS.
Total test time ~5.3s per run (fast after gen0 GPU warmup at ~394ms).
AMD_LOG_LEVEL=3 confirms: "Using native code object for device: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-".
alloca->fixed-array fix holds at 1000-pop / 1024-thread blocks; assert(top==1) canary silent.
test/test_bind_success.py pre-existing failure confirmed unchanged (backend-independent, not gated).

Result: PASS -> linux-gfx90a completed (validated_sha ba4fa7e6656fbaaf0a42eacca215e543b1c9a2e0).

## Validation 2026-06-02 (linux-gfx1100, validator)

GPU: 2x AMD Radeon Pro W7800 48GB (gfx1100, RDNA3 wave32), HIP_VISIBLE_DEVICES=0, ROCm 7.2, torch 2.13.0a0+gitb5e90ff (hip 7.2.53211).
Fork: AMD-Ecosystem/evogp @ moat-port, SHA ba4fa7e6656fbaaf0a42eacca215e543b1c9a2e0 (no code change; follower validate-first).

Build (clean from committed source):
```
rm -rf build src/evogp.egg-info src/evogp/*.so src/evogp/hip
HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1100 pip install -e /var/lib/jenkins/moat/projects/evogp/src --no-build-isolation
# Completed in ~39s; Strategy B (torch hipify) generated src/evogp/hip/*.hip
```

gfx1100 code-object evidence:
- roc-obj-ls on evogp_cuda.cpython-312-x86_64-linux-gnu.so shows 3x "hipv4-amdgcn-amd-amdhsa--gfx1100" code objects; no gfx90a.
- AMD_LOG_LEVEL=3 confirms: "Using native code object for device: amdgcn-amd-amdhsa--gfx1100 co: amdgcn-amd-amdhsa--gfx1100" on first kernel dispatch.

GPU gate (run twice, fixed seed):
```
HIP_VISIBLE_DEVICES=0 python -m evogp.sr_test
```

Run 1: Gen 0 best -0.2511 -> Gen 99 best -0.0179; exit 0; ~3.7s total (300ms gen0 warmup, ~5ms/gen thereafter); no HIP/HSA fault; no assert(top==1) abort.
Run 2: Gen 0 best -0.2511 -> Gen 99 best -0.0179; exit 0; identical output -- deterministic PASS.

Verdict: PASS. Fixed-array scratch (alloca->float stack[MAX_STACK]) holds at gfx1100 (RDNA3) occupancy. No HSA_STATUS_ERROR_EXCEPTION, no assert(top==1) canary, fitness converges monotonically from -0.2511 (gen 0) to -0.0179 (gen 99), finite throughout, two seeded runs identical. Results match gfx90a baseline exactly (-0.0179 final best, same trajectory). No fork code change needed -- arch-unified fix validated on wave32.

Result: PASS -> linux-gfx1100 completed (validated_sha ba4fa7e6656fbaaf0a42eacca215e543b1c9a2e0).

## Review 2026-06-02 (linux-gfx90a, reviewer)
Reviewed moat-port @ ba4fa7e vs base ee11f1e via /pr-review. Verdict: review-passed (no problems found). Re-ran the GPU gate on real gfx90a (GCD 0).

Verified on hardware:
- Clean rebuild from committed source (rm -rf build src/evogp/hip; pip install -e . --no-build-isolation) succeeds; hipify mirror src/evogp/hip/*.hip generated (Strategy B active); torch HIP build defines -DUSE_ROCM=1 so the #ifndef USE_ROCM guard correctly drops device_launch_parameters.h.
- sr_test: 100 generations, two fixed-seed runs both converge gen0 best -0.2511 -> gen99 best -0.0179 (identical final), finite throughout, exit 0, no HIP fault / no assert(top==1).
- alloca fully converted at all 6 sites (forward.cu x4, generate.cu x2 arrays, mutation.cu x3); fixed-array sizes byte-match the original alloca bounds (float[MAX_STACK], int16_t[2*MAX_STACK], GPNode[MAX_STACK], NchildDepth[MAX_STACK]); arrays decay to the same pointers at eval call sites; assert(top==1) canary did not fire at pop 1000 / 1024-thread blocks. Arch-unified: zero warp intrinsics, zero hardcoded 32, fitness reduction fully __syncthreads()-guarded (warp-tail commented out at forward.cu:464,639).
- hipified output confirms __constant__ preserved, cudaMemcpyToSymbolAsync->hipMemcpyToSymbolAsync (hipMemcpyDeviceToDevice), float atomicAdd preserved (HIP-correct on gfx90a), thrust::random resolved against rocThrust.
- setup.py: ROCm branch keeps only -O3; CUDA else-branch byte-identical to upstream's original nvcc flag list (CUDA path unchanged).
- test/test_bind_success.py failure confirmed pre-existing and backend-independent: host-side check_tensor(roulette_funcs, {Function::END=29}) rejects the test's 24-element tensor; identical on CUDA, unrelated to any HIP change.
- Commit hygiene clean: title 50 chars [ROCm]; body mentions Claude + Test Plan; no noreply/ghstack trailer; ASCII-clean msg and diff; fork main == origin main (clean mirror); Actions disabled on fork.

## Validation 2026-06-05 (windows-gfx1101, validator)

GPU: AMD Radeon PRO V710 (gfx1101, RDNA3 wave32), HIP_VISIBLE_DEVICES=0, ROCm 7.14.0a20260604, torch 2.9.1+rocm7.14.0a20260604.
Fork: AMD-Ecosystem/evogp @ moat-port, SHA ba4fa7e6656fbaaf0a42eacca215e543b1c9a2e0 (no code change; follower validate-first).

Build environment notes (Windows-specific):
- ROCM_HOME must be set explicitly to _rocm_sdk_devel (torch's _find_rocm_home shutil.which('hipcc') fails when venv Scripts/ is not on PATH in bash context).
- DISTUTILS_USE_SDK=1 required; VC environment is already activated on this host.
- MSVC link.exe must precede Git's /usr/bin/link.exe on PATH (Git link is a Unix tool that fails on /LTCG flags); prepend C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64 to PATH.

Build command:
```
cd B:/develop/moat/projects/evogp/src
PATH="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64:$PATH" \
  HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1101 \
  ROCM_HOME="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel" \
  DISTUTILS_USE_SDK=1 \
  pip install -e . --no-build-isolation
# Completed in ~29s; hipify generated src/evogp/hip/*.hip; .pyd contains hipv4-amdgcn-amd-amdhsa--gfx1101 code objects (confirmed via strings)
```

GPU gate (run twice, fixed seed):
```
HIP_VISIBLE_DEVICES=0 python -m evogp.sr_test
```

Run 1: Gen 0 best -0.2511 -> Gen 99 best -0.0179; exit 0; ~5s total (~270ms gen0 warmup, ~16ms/gen thereafter); no HIP/HSA fault; no assert(top==1) abort.
Run 2: Gen 0 best -0.2511 -> Gen 99 best -0.0179; exit 0; identical output -- deterministic PASS.

gfx1101 code-object evidence: strings evogp_cuda.cp312-win_amd64.pyd | grep gfx -> "hipv4-amdgcn-amd-amdhsa--gfx1101".

Verdict: PASS. Fixed-array scratch (alloca->float stack[MAX_STACK]) holds at gfx1101 (RDNA3 wave32) occupancy. No HIP fault, no assert(top==1) canary, fitness converges monotonically from -0.2511 (gen 0) to -0.0179 (gen 99), finite throughout, two seeded runs identical. Results match gfx90a and gfx1100 baseline exactly. No fork code change needed -- arch-unified fix validated on Windows RDNA3.

Result: PASS -> windows-gfx1101 completed (validated_sha ba4fa7e6656fbaaf0a42eacca215e543b1c9a2e0).

## Validation 2026-06-06 (windows-gfx1201, validator)

GPU: AMD Radeon RX 9070 XT (gfx1201, RDNA4 wave32), HIP_VISIBLE_DEVICES=0, ROCm 7.14.0a20260604, torch 2.9.1+rocm7.14.0a20260604.
Fork: AMD-Ecosystem/evogp @ moat-port, SHA ba4fa7e6656fbaaf0a42eacca215e543b1c9a2e0 (no code change; follower validate-first).

Note: gfx1101 (Radeon PRO V710) was absent from the bus during this validation; gfx1201 (RX 9070 XT) was the only GPU, enumerated at HIP_VISIBLE_DEVICES=0.

Build command:
```
rm -rf B:/develop/moat/projects/evogp/src/src/evogp/*.pyd B:/develop/moat/projects/evogp/src/src/evogp/hip B:/develop/moat/projects/evogp/src/src/evogp.egg-info
PATH="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64:$PATH" \
  HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1201 \
  ROCM_HOME="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel" \
  DISTUTILS_USE_SDK=1 \
  pip install -e B:/develop/moat/projects/evogp/src --no-build-isolation
# Completed in ~29s; hipify generated src/evogp/hip/*.hip; .pyd contains hipv4-amdgcn-amd-amdhsa--gfx1201 code objects (confirmed via strings)
```

GPU gate (run twice, fixed seed):
```
HIP_VISIBLE_DEVICES=0 python -m evogp.sr_test
```

Run 1: Gen 0 best -0.2511 -> Gen 99 best -0.0179; exit 0; ~3.5s total (~280ms gen0 warmup, ~4ms/gen thereafter); no HIP/HSA fault; no assert(top==1) abort.
Run 2: Gen 0 best -0.2511 -> Gen 99 best -0.0179; exit 0; identical output -- deterministic PASS.

gfx1201 code-object evidence: strings evogp_cuda.cp312-win_amd64.pyd | grep gfx -> "hipv4-amdgcn-amd-amdhsa--gfx1201".

Verdict: PASS. Fixed-array scratch (alloca->float stack[MAX_STACK]) holds at gfx1201 (RDNA4 wave32) occupancy. No HIP fault, no assert(top==1) canary, fitness converges monotonically from -0.2511 (gen 0) to -0.0179 (gen 99), finite throughout, two seeded runs identical. Results match gfx90a/gfx1100/gfx1101 baseline exactly. No fork code change needed -- arch-unified fix validated on Windows RDNA4.

Result: PASS -> windows-gfx1201 completed (validated_sha ba4fa7e6656fbaaf0a42eacca215e543b1c9a2e0).

## PR prep 2026-06-17 (orchestrator)

PR-prep pass on moat-port. Prior state: README ROCm/HIP build section already
added in 6447ebc (carried forward doc-only to all platforms). This pass added
one doc-only commit on top (6bbf08f) extending that section to mention
PYTORCH_ROCM_ARCH for AMD arch selection.

- Scrub: clean. No MOAT vocabulary or banned phrasing in the tracked tree
  (checked MOAT/Strategy A,B/head_sha/validated_sha/moat-port/lead/follower/
  curated commit/byte-for-byte/byte-identical/bit-identical/bit-for-bit).
- Attribution: no copyright/author header added. All 5 changed source/build
  files (setup.py, forward.cu, generate.cu, kernel.h, mutation.cu) are
  pre-existing upstream files with NO copyright header, and none were authored
  by us from scratch; a lone AMD copyright would falsely imply sole authorship,
  so per the copyright rule none was added.
- Arch: setup.py hardcodes no AMD arch; torch selects via PYTORCH_ROCM_ARCH.
  CUDA else-branch nvcc flag list is identical to upstream's original, so a
  build without ROCm is an unchanged CUDA configuration.
- advance-head 6bbf08f: doc-only, all four completed platforms carried forward
  (validated_sha -> 6bbf08f). No platform flipped to revalidate; no GPU re-run.
- pr-ready: True (linux-gfx90a, linux-gfx1100, windows-gfx1201, windows-gfx1101
  completed; windows-gfx1151 blocked/non-viable, scoped out of the PR claim).
- Standalone validation: `python -m evogp.sr_test` is fully self-contained
  (synthetic SR data, no external dataset, no gated model), the gate already
  used by all completed platforms.
