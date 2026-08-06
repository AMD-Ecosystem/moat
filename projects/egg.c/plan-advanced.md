# egg.c -- Advanced trainers + d-eggs HIP/ROCm port plan (lead: linux-gfx90a)

> STACKED PR on PR #8. This is the SECOND egg.c PR. PR #8 (the integer-only
> `full_cuda_train_egg.cu` port, fork tip `5081bbf`) is the base. This plan's
> work lives on branch `moat-port-advanced` (based at `5081bbf`); its upstream PR
> base branch is `moat-port` (the PR #8 branch), NOT upstream `master`. All
> compat-header changes here are ADDITIVE on top of the integer-only port's
> `egg_hip_compat.cuh` / `egg_warp_compat.cuh`; nothing in PR #8 is reverted.
> Do NOT rebase this branch onto upstream master -- it must sit on PR #8 so the
> diff is only the advanced delta.

Project: egg.c (d0rc/egg.c, default branch master). Raw-nvcc/Makefile build (no CMake).

## Existing AMD support
None for these components. `grep -rniE 'amd|rocm|hip|gfx'` over README.md and
d-eggs/README.md returns only the AMD build block MOAT itself added in PR #8
(README.md:40-43). No upstream rocm/hip branch, no separate AMD project, no
community fork. Decision: genuine from-scratch HIP port, same compat-header
pattern proven in PR #8. Upstream merges contributor PRs (d0rc/egg.c has merged
PRs #5/#6), so an upstream PR is the right vehicle -- but stacked as a second PR
after #8 lands, or against #8's branch while #8 is open.

## Build classification
Makefile / raw-nvcc, ext_type=make. Strategy A (compat-header), exactly as PR #8.
Evidence: top-level files compiled directly with `nvcc full_cuda_train_*.cu`;
d-eggs/Makefile line 1-3 `NVCC = nvcc`, `NVCCFLAGS = -O3 -Iinclude -arch=sm_86
-lcublas -rdc=true`. No CMake, no torch.

## Port strategy
Strategy A. Reuse and EXTEND the two PR #8 compat headers; add cuBLAS->hipBLAS to
`egg_hip_compat.cuh`; reintroduce the warp shuffle helpers in
`egg_warp_compat.cuh` (PR #8's PR-prep removed `eggShflDownSync`/`eggShflXorSync`/
`egg_lane_mask_t` as unused -- they are NEEDED here for the transformer head
reductions and RoPE). Keep every HIP body behind `#if defined(__HIP__)` so the
nvcc build is byte-identical, the invariant PR #8 already validated.

## The load-bearing wave-width fact (carried from PR #8 notes)
`WARP_SIZE`/`EGG_WARP_SIZE` = 32 is a DATA-LAYOUT/tiling stride, NOT the physical
wavefront. It MUST stay 32 on wave32 AND wave64. On wave64 (gfx90a) two logical
32-lane warps share one 64-lane wavefront, so every shuffle/reduce must run at
EXPLICIT width 32 with a 64-bit all-ones mask. The transformers add a NEW twist
the integer-only trainer did not have: the attention score reduction uses a
32-lane `__shfl_down_sync(...,off)` (off from 16) to reduce ONE 64-element head,
then `(tid % 32)==0` lane-0 does an `atomicAdd` -- i.e. TWO logical 32-warps each
reduce half the head and the atomicAdd combines them. On wave64 those two logical
warps are one wavefront; width-32 confinement keeps them independent (identical to
the EnvGS/icicle cross-logical-warp lesson). RoPE uses `__shfl_xor_sync(...,1)`
(neighbor pair swap) which is a width-1 partner exchange -- correct at any width
as long as the mask covers the wavefront; pin width 32 + 64-bit mask for safety.

---

## Component 1: full_cuda_train_egg_transformer.cu (844 lines) -- single-GPU transformer
CUDA surface:
- `#include <cuda_runtime.h>`, `<cub/cub.cuh>`, thrust headers (line 6,14-19).
- `#define WARP_SIZE 32` (line 35) -- KEEP 32 (logical tiling).
- `cub::BlockReduce<AccumType, BLOCK_THREADS>` (line 157) and BlockReduce
  TempStorage (189,478) -- BLOCK-wide collective, explicit BLOCK_THREADS, NOT
  warp-width-dependent. No change beyond `namespace cub = hipcub` (already in
  PR #8 compat header).
- `__shfl_down_sync(0xFFFFFFFF, df, off)` head reduction, off=16 (lines 272, 541)
  -> route through `eggShflDownSync` (width-32, 64-bit mask). The `(tid%32)==0`
  lane-0 atomicAdd that follows (line 277) is already 32-correct since width-32
  leaves lane 0 of each logical warp as the partial-sum holder.
- `extern __shared__ ActType s_mem[]` dynamic shared (187,476) + `__shared__`
  scalars/logits/counts -- host launch passes `sm_size`; no warp-count sizing, so
  arch-independent. No change.
- thrust device_ptr/reduce/transform_reduce -- rocThrust exposes thrust:: at the
  same paths (PR #8 note); no remap.

HIP work: include `egg_hip_compat.cuh` + `egg_warp_compat.cuh` under `__HIP__`,
guard the `cuda_runtime.h`/`cub/cub.cuh` includes `#if !defined(__HIP__)`, swap
the two `__shfl_down_sync` sites to `eggShflDownSync`. No cuBLAS, no multi-GPU.

## Component 2: full_cuda_train_egg_transformer_adam.cu (1171 lines) -- transformer + Adam
Component-1 surface PLUS:
- RoPE `__shfl_xor_sync(0xFFFFFFFF, val, 1)` (line 423) -> `eggShflXorSync`
  (width-32, 64-bit mask).
- Two more `__shfl_down_sync(...,off)` head reductions (lines 564, 579) ->
  `eggShflDownSync`.
- `atomicMax` on `__shared__ int32_t` (567, 714, 946) -- int32 shared atomics are
  fully supported on AMD; no change. (NOTE the managed-atomic audit only applies
  to d-eggs cudaMallocManaged buffers, not these __shared__ ones.)
- `#include "egg_adaptive_normalize.h"` (line 22) -- SHARED helper, see below.
- Adam optimizer kernels are elementwise (`update_*_adam_kernel`): no warp ops,
  no cuBLAS. No change.

egg_adaptive_normalize.h (top-level, 2 functions): both have the SAME logical-32
max-reduction `__shfl_down_sync(mask, abs_v, offset)` off=16 with `warp_id=tid/32`,
`lane_id=tid%32` (lines 19-21, 68-70). Route both through `eggShflDownSync`. This
header is included by Components 2 AND 3, so fix it once.

HIP work: includes guard + the shfl swaps above. No cuBLAS, no multi-GPU.

## Component 3: full_cuda_train_transformer_adam_mgpu.cu (1767 lines) -- Int8NativeFormer, multi-GPU + Muon
Component-2 surface PLUS the cuBLAS + multi-GPU + NTT surface:
- `#include "egg_ntt.cuh"` (line 48), `#include "muon_internal.cuh"` (line 541),
  `#include "egg_adaptive_normalize.h"` (24).
- `#define WARP_SIZE 32` (guarded `#ifndef`, 68) -- keep.
- `__shfl_xor_sync(...,1)` RoPE (634) -> eggShflXorSync; `__shfl_down_sync(...,off)`
  head reductions (797, 812) -> eggShflDownSync.
- `cub::BlockReduce<long long, BLOCK_THREADS> BlockReduce64` (1007,1239),
  `cub::BlockScan<long long, BLOCK_THREADS> BlockScan64` (1254) -- BLOCK-wide,
  arch-independent. No change beyond `namespace cub = hipcub`.

cuBLAS surface (the only cuBLAS in the whole project, shared with muon_internal.cuh):
- `CHECK_CUBLAS` macro + `cublasStatus_t`/`CUBLAS_STATUS_SUCCESS` (273).
- `cublasHandle_t`, `cublasCreate`, `cublasSetStream` (1299,1476,1477).
- In muon_internal.cuh: `cublasSetPointerMode`, `CUBLAS_POINTER_MODE_DEVICE/HOST`,
  `cublasSnrm2`, `cublasSgemm`, `cublasOperation_t`, `CUBLAS_OP_T/N`.
  ALL host-side cuBLAS calls (no device-side cublas, no -rdc need from cublas).
  -> hipBLAS 1:1: hipblasHandle_t, hipblasCreate, hipblasSetStream,
     hipblasSetPointerMode (HIPBLAS_POINTER_MODE_DEVICE/HOST), hipblasSnrm2,
     hipblasSgemm, hipblasOperation_t (HIPBLAS_OP_T/N), hipblasStatus_t
     (HIPBLAS_STATUS_SUCCESS). Add these to egg_hip_compat.cuh under __HIP__ and
     include <hipblas/hipblas.h>; link `-lhipblas`. hipBLAS-on-AMD is rocBLAS.
  RISK (numerics): Newton-Schulz is an iterative orthogonalization; the column-
  major/row-major gymnastics in muon_internal.cuh (the long comment block at
  195-330) must produce the same Gram-matrix orientation under hipBLAS. hipBLAS
  matches cuBLAS column-major convention, so the OP_T/OP_N logic transfers 1:1;
  validate numerically (below). hipblasSnrm2 in DEVICE pointer mode writes the
  norm to device memory async -- supported on hipBLAS/rocBLAS.

Multi-GPU surface (NO NCCL/RCCL, NO peer access -- confirmed by grep):
- Replicated data-parallel: each GPU gets a FULL model copy + a population shard.
  `cudaSetDevice(i)` loop (1443,1527,1551,1568), one `cudaStreamCreate` per device
  (1453), one cuBLAS handle per device (1476). Host aggregates fitness in PINNED
  memory (`cudaMallocHost`, 1492-1493) and broadcasts back via
  `cudaMemcpyAsync` H2D per device (1571). `cudaMemcpyToSymbol` for LUTs per
  device (1447-1449); `cudaGetSymbolAddress` (1473).
  -> ALL 1:1 HIP: hipSetDevice, hipStreamCreate, hipStreamSynchronize,
     hipMallocHost (or hipHostMalloc), hipMemcpyAsync, hipMemsetAsync,
     hipMemcpyToSymbol, hipGetSymbolAddress. Add the missing ones to the compat
     header (PR #8 already has MallocHost? NO -- add hipMallocHost/hipHostMalloc,
     hipStreamCreate, hipStreamSynchronize, hipMemcpyAsync, hipMemsetAsync,
     hipGetSymbolAddress, cudaStream_t typedef, cudaMemcpyHostToDevice already
     present). LOW RISK: this is the same data-parallel pattern with no device-to-
     device traffic; the 4 gfx1100 GPUs on this host exercise it directly.

egg_ntt.cuh (587 lines, NTT/WHT): all `__device__`, uses `__syncthreads()` only,
NO `__shfl`/`__ballot`/warpSize. WHT/NTT butterflies operate over `__shared__`
arrays indexed by `tid`/`blockDim.x` with `__syncthreads()` barriers (lines
106-414). Arch-independent; compiles under hipcc unchanged once includes are
guarded. WATCH (PR #8 / icicle lesson): an in-place shared NTT that relies on
implicit wave64 lockstep between two logical warps would need a barrier on wave32
-- but this NTT already fences every stage with `__syncthreads()` (block-wide),
so it is safe on wave32. No change beyond the include guard.

HIP work: includes guard, shfl swaps (shared with comp 1/2), cuBLAS->hipBLAS via
the compat header, multi-GPU runtime symbols via the compat header. -fgpu-rdc NOT
needed (worker/mgpu are single-TU: kernels are in-file or #included, not separately
compiled; cuBLAS links host-side).

## Component 4: d-eggs/ -- Distributed EGGROLL (23 files, its own Makefile)
This is the largest and highest-risk component. Makefile:
`NVCCFLAGS = -O3 -Iinclude -arch=sm_86 -lcublas -rdc=true`, single .cu TU
`src/worker.cu` (which `#include`s `kernels.cu` at line 22, so it is ONE TU --
`-rdc=true` is the upstream default, not a true cross-TU device-link need).
coordinator.cpp is pure host C++ (sockets/pthread), no CUDA.

CUDA surface (worker.cu + included headers):
- cuBLAS via optimizer/muon.cuh (181-240): SAME cublasSnrm2/Sgemm/SetPointerMode/
  Sgemm as top-level muon_internal.cuh. -> hipBLAS, same swap.
- `__shfl_xor_sync(...,1)` RoPE in model/layers.cuh (139); `__shfl_down_sync(...,off)`
  head reductions in layers.cuh (369,382); `__shfl_down_sync(mask,...,offset)`
  off=16 in math/adaptive_norm.cuh (21,70). -> eggShflXorSync/eggShflDownSync.
- `#ifndef WARP_SIZE / #define WARP_SIZE 32` in include/config.h (30-31) -- keep 32
  (logical). `ALIGNED_DIM = (HIDDEN_DIM+31)&~31` (config) is a 32-alignment, fine.
- MANAGED MEMORY: worker.cu:215-218 `cudaMallocManaged` + `cudaMemAdvise`
  (SetPreferredLocation, SetAccessedBy) + `cudaMemPrefetchAsync`. -> hipMallocManaged,
  hipMemAdvise (hipMemAdviseSetPreferredLocation/SetAccessedBy), hipMemPrefetchAsync.
  RISK (managed-memory atomic audit, per task): confirm whether any kernel does
  atomicMin/Max/Add into a managed buffer that the HOST also touches between
  launches -- HIP managed coherence on gfx90a is page-migration based; a host read
  of a managed counter without a sync can see stale data. The mgpu/transformer
  __shared__ atomics are NOT managed and are fine; the audit is specifically the
  worker.cu managed buffers. Add `hipDeviceSynchronize`/prefetch-back before host
  reads if found lacking (most likely already synced via the graph launch).
- hipGraph: worker.cu uses `cudaGraph_t`/`cudaGraphExec_t` (355-356),
  `cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)` (439),
  `cudaGraphInstantiate` (522), `cudaGraphGetNodes` (529), `cudaGraphLaunch` (533).
  -> hipGraph_t, hipGraphExec_t, hipStreamBeginCapture(hipStreamCaptureModeGlobal),
  hipGraphInstantiate, hipGraphGetNodes, hipGraphLaunch. All 1:1.
  RISK (hipGraph): stream-capture of a sequence that includes cuBLAS calls -- the
  Newton-Schulz update is captured into the graph. hipBLAS/rocBLAS must be
  capture-safe (no host sync inside the captured region). cublasSnrm2 in DEVICE
  pointer mode is chosen precisely to avoid implicit sync, which is capture-
  friendly. VALIDATE the captured graph replays correctly; if rocBLAS breaks
  capture on this ROCm, fall back to eager launch under `__HIP__` (a documented
  scope-out, not a blocker) -- but try capture first.
- Distributed: worker.cu opens TCP sockets to a coordinator (host C++); the
  GPU work is single-node per worker. Multi-NODE is orchestrated by the
  coordinator over sockets, NOT GPU-to-GPU. So d-eggs needs NO RCCL: each worker
  is a single-GPU process; the 4 gfx1100 GPUs run 4 worker processes coordinated
  by one coordinator. This is the natural validation topology.

Build (d-eggs): add a `USE_HIP` arm to d-eggs/Makefile (do NOT replace the nvcc
arm -- gate it):
```
ifeq ($(USE_HIP),1)
  GPUCC   = hipcc
  GPUFLAGS = -O3 -Iinclude -x hip $(addprefix --offload-arch=,$(GPU_ARCH)) -lhipblas
  # GPU_ARCH defaults e.g. gfx90a; caller overrides. No -rdc / -fgpu-rdc (single TU).
else
  GPUCC   = nvcc
  GPUFLAGS = -O3 -Iinclude -arch=sm_86 -lcublas -rdc=true
endif
worker: src/worker.cu
	$(GPUCC) $(GPUFLAGS) -o worker src/worker.cu
```
Do NOT hardcode the arch; default `GPU_ARCH ?= gfx90a` and let followers pass
`GPU_ARCH="gfx90a gfx1100"` for a fat binary. coordinator/print_arch stay g++.

---

## Compat-header extensions (additive on PR #8)
egg_hip_compat.cuh -- add under `#if defined(__HIP__)`:
- `#include <hipblas/hipblas.h>`; alias cublas* -> hipblas* (Handle, Create,
  SetStream, SetPointerMode, Snrm2, Sgemm, Status_t, Operation_t, the
  POINTER_MODE_/OP_/STATUS_SUCCESS enums).
- runtime symbols the advanced files add that PR #8 lacks: cudaStream_t,
  cudaStreamCreate, cudaStreamSynchronize, cudaMemcpyAsync, cudaMemsetAsync,
  cudaMallocHost (hipHostMalloc), cudaGetSymbolAddress, cudaSetDevice,
  cudaGetDeviceCount, cudaMallocManaged, cudaMemAdvise (+ the advise enums),
  cudaMemPrefetchAsync, the hipGraph family (cudaGraph_t, cudaGraphExec_t,
  cudaStreamBeginCapture + cudaStreamCaptureModeGlobal, cudaGraphInstantiate,
  cudaGraphGetNodes, cudaGraphLaunch).
  Use hipify's cuda_to_hip_mappings.py as the authoritative name source.

egg_warp_compat.cuh -- REINTRODUCE (PR #8 prep removed them as unused):
- `eggShflDownSync(T v, int off)` -> `__shfl_down_sync(EGG_FULL_MASK, v, off, EGG_WARP_SIZE)`
- `eggShflXorSync(T v, int lane_mask)` -> `__shfl_xor_sync(EGG_FULL_MASK, v, lane_mask, EGG_WARP_SIZE)`
  Templated on the value type (AccumType is int32/int64 in different sites; the
  head-reduce df is AccumType, RoPE val is AccumType). EGG_FULL_MASK is already
  `~0ull` under __HIP__. Keep them `__device__ __forceinline__`. (egg_lane_mask_t
  was unused even here -- do NOT reintroduce it; only the two shfl helpers.)

Both headers keep every body behind `__HIP__`, so the nvcc build is unchanged and
the explicit width-32 args are no-ops on NVIDIA (32-wide native).

## Build recipe (lead gfx90a)
Single-GPU transformers:
```
/opt/rocm/bin/hipcc -O3 --offload-arch=gfx90a -x hip \
  projects/egg.c/src/full_cuda_train_egg_transformer.cu -o egg_xf
/opt/rocm/bin/hipcc -O3 --offload-arch=gfx90a -x hip \
  projects/egg.c/src/full_cuda_train_egg_transformer_adam.cu -o egg_xf_adam
```
mgpu + muon (needs hipBLAS):
```
/opt/rocm/bin/hipcc -O3 --offload-arch=gfx90a -x hip \
  projects/egg.c/src/full_cuda_train_transformer_adam_mgpu.cu -lhipblas -o egg_mgpu
```
Multi-arch gate (one fat binary; confirm BOTH code objects with
`llvm-objdump --offloading | grep -io 'gfx[0-9a-f]*'` -> gfx90a gfx1100):
```
/opt/rocm/bin/hipcc -O3 --offload-arch=gfx90a --offload-arch=gfx1100 -x hip \
  full_cuda_train_transformer_adam_mgpu.cu -lhipblas -o egg_mgpu_multi
```
d-eggs: `cd d-eggs && make USE_HIP=1 GPU_ARCH=gfx90a` (worker), `make coordinator`.
Use `utils/timeit.sh egg.c compile -- ...` and absolute source paths (PR #8 gotcha).

## Test plan / validation gate
Non-GPU regression (must not regress): `d-eggs/test_ternary.cpp`
(`g++ -O3 -Id-eggs/include`), already green in PR #8 -- re-run.

Per component GPU gate (mirror PR #8's loss-decrease + fixed-seed determinism):
1. transformer / transformer_adam (single GPU, gfx90a then gfx1100):
   - Build fat binary, confirm both code objects.
   - Run with the 18000-byte repeating-text corpus; PASS = Loss decreases
     monotonically over the first ~16 steps and the generated sample becomes
     text-like (not garbage). Add/confirm the `EGG_FIXED_SEED` determinism hook
     (PR #8 added it to full_cuda_train_egg.cu; check whether the transformers
     share that seed path -- if not, add the same env override) and require TWO
     pinned-seed runs BIT-IDENTICAL on Loss + Up+/Up- -- the decisive wave64
     width-32 fingerprint (a wrong 64-lane head reduction diverges run-to-run).
   - Cross-arch: gfx1100 step-0 Loss must match gfx90a step-0 Loss bit-for-bit
     (same EGG_FIXED_SEED), as PR #8 proved for the integer trainer.
2. mgpu + muon (multi-GPU + cuBLAS, this host has 4 gfx1100):
   - Run on N=2 and N=4 GPUs (the loop enumerates devices); PASS = loss decreases
     and N-GPU result is deterministic with EGG_FIXED_SEED. The per-GPU model
     copies + host fitness aggregation must give a stable trajectory.
   - cuBLAS/Muon NUMERICAL check: dump the Newton-Schulz output (the orthogonalized
     momentum buffer) for a fixed input and compare hipBLAS vs a CPU float
     reference (X X^T / norm Newton-Schulz, a few iterations) to a tight rtol
     (e.g. 1e-4). This catches a row/col-major or OP_T/OP_N transpose error in the
     hipBLAS swap, which a loss-decrease alone could mask. Build `USE_MUON=1`.
3. d-eggs distributed (4 gfx1100 workers + 1 coordinator):
   - `make USE_HIP=1`, launch the coordinator + 4 worker processes (one per
     `HIP_VISIBLE_DEVICES=0..3`) via run-workers.sh; PASS = workers connect,
     training loss decreases, hipGraph capture/replay succeeds (no capture error),
     checkpoint round-trips. Managed-memory atomics: confirm host-side counters
     read correct values (no stale managed page) across steps.
   - If hipGraph capture fails on rocBLAS-in-capture, fall back to eager launch
     under __HIP__ and document; still a PASS for correctness.

full_trained_egg.c (583 lines, .c): CPU-only, NOT CUDA. It is ARM-NEON
(`#include <arm_neon.h>`, `#include <dispatch/dispatch.h>` -- Apple GCD). It was
never x86-buildable upstream and has no GPU code. NO port needed; leave byte-
identical (PR #8 already confirmed it untouched / not a regression). Just note it
in the PR as out of scope.

## Recommended PORT ORDER (simplest-first; share code where noted)
1. egg_warp_compat.cuh + egg_hip_compat.cuh extensions FIRST (shfl helpers +
   runtime symbols; hipBLAS aliases can land with step 3). They unblock all.
2. full_cuda_train_egg_transformer.cu -- smallest, no cuBLAS, no multi-GPU. Proves
   the head-reduction width-32 fix end to end. Validate (gfx90a + gfx1100).
3. full_cuda_train_egg_transformer_adam.cu -- adds RoPE shfl_xor +
   egg_adaptive_normalize.h (shared shfl fix). Validate.
   (Steps 2-3 share egg_adaptive_normalize.h and all the shfl helpers; port together.)
4. full_cuda_train_transformer_adam_mgpu.cu + muon_internal.cuh + egg_ntt.cuh --
   adds hipBLAS (Newton-Schulz numerical gate) + replicated multi-GPU. Validate on
   2 and 4 gfx1100. (muon_internal.cuh and d-eggs/optimizer/muon.cuh are NEAR-
   IDENTICAL cuBLAS code -- port one, mirror to the other.)
5. d-eggs/ -- the distributed subsystem: hipGraph capture, managed memory +
   atomic audit, the Makefile USE_HIP arm, multi-process/multi-node coordinator.
   Highest risk; do last. Reuses the muon hipBLAS swap and all shfl helpers from
   steps 1-4.

Shared-code map: the shfl helpers (warp_compat) and hipBLAS aliases (hip_compat)
are written once and consumed by 1-5. egg_adaptive_normalize.h is shared by 3,4.
muon Newton-Schulz is duplicated in muon_internal.cuh (top-level) and
d-eggs/optimizer/muon.cuh -- one fix, applied to both.

## Risk / uncertainty flags
- RCCL/NCCL: NOT used anywhere (confirmed by grep). No RCCL dependency. Multi-GPU
  is replicated data-parallel via host pinned-memory aggregation; distributed is
  TCP sockets between processes. LOW risk.
- hipBLAS Newton-Schulz numerics: row/col-major + OP_T/OP_N must transpose
  identically; hipBLAS is column-major like cuBLAS so 1:1, but VERIFY with the CPU
  reference gate (the muon comment block at muon_internal.cuh:195-330 is fiddly).
- hipGraph + rocBLAS in capture (d-eggs): capture-safety of cuBLAS-in-graph is the
  single biggest unknown. Device-pointer-mode Snrm2 is chosen to avoid sync (good
  for capture). Fallback: eager launch under __HIP__, documented scope-out.
- managed memory atomic coherence (d-eggs worker.cu): audit host reads of managed
  counters; add a sync/prefetch-back if a stale-page read is possible on gfx90a.
- -fgpu-rdc: the upstream `-rdc=true` is the nvcc default for the single-TU worker
  build; NOT actually needed (kernels.cu is #included, cuBLAS links host-side). The
  USE_HIP arm OMITS -fgpu-rdc. If a link error reveals a genuine cross-TU device
  symbol, add `-fgpu-rdc` to both compile and a `hipcc -fgpu-rdc` device-link step.
- EGG_FIXED_SEED hook: PR #8 added it only to full_cuda_train_egg.cu. The advanced
  trainers may use the same `time(NULL)^...` seed without the override -- if so,
  add the identical env override so the determinism gate is reproducible. (Behavior
  preserved when unset; same as PR #8.)
- Windows followers (gfx1201): hipBLAS availability on TheRock + hipGraph on
  Windows ROCm are the follower unknowns -- defer to the follower delta-plan, not
  the lead. d-eggs uses POSIX sockets (sys/socket.h, arpa/inet.h) so the
  coordinator/worker distributed path is Linux-only without a Winsock shim; scope
  the Windows follower to the single-GPU transformers (steps 2-3), not d-eggs.

## Open questions
- Does the upstream prefer one stacked PR (advanced) on top of #8, or fold both
  into #8? Default: a separate second PR with base = the #8 branch.
- hipGraph-capture-with-rocBLAS on ROCm 7.2.1: confirm capture-safe at port time;
  if not, eager fallback (documented).
- Whether the transformer trainers already carry the EGG_FIXED_SEED hook or need
  it added for the determinism gate.
