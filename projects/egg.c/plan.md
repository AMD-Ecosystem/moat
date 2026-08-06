# egg.c (EGGROLL in C) -- ROCm/HIP port plan

## Project
- Name: egg.c
- Upstream: https://github.com/d0rc/egg.c
- Default branch: master
- What it is: a dependency-free, integer-only ("int8 weights / int64 accum") language-model trainer using EGGROLL (Evolution Guided General Optimization via Low-rank Learning) -- gradient-free Evolution Strategies with low-rank perturbations, no backprop, no PyTorch/JAX. Ships a CPU/Apple-Silicon (NEON+GCD) path and several standalone CUDA `.cu` trainers, plus a distributed coordinator/worker subsystem (`d-eggs/`).

## Existing AMD support
None. The only non-NVIDIA backend is CPU/Apple-Silicon (ARM NEON, Grand Central Dispatch) in `full_trained_egg.c`. No HIP/ROCm/OpenCL/Vulkan path, no stale port branch, no AMD mention anywhere (the README "Bare Metal" line is unrelated). A fresh CUDA-to-HIP port adds clear value.
Decision: PROCEED with a correctness-first mechanical CUDA-to-HIP port.

## Build classification
cmake | torch-extension -> NEITHER. This is a raw-nvcc / Makefile project.
- No CMake, no `find_package(Torch)`, no `setup.py`/`pyproject.toml`, no `torch.utils.cpp_extension`. Confirmed: only build file in tree is `d-eggs/Makefile`.
- Top-level `.cu` files are compiled one-shot by hand, per README:
  - `nvcc -O3 full_cuda_train_egg.cu -o egg_cuda`
  - `nvcc -O3 -arch=native full_cuda_train_transformer_adam_mgpu.cu -o egg_transformer_mgpu`
- `d-eggs/Makefile`: `NVCC = nvcc`, `NVCCFLAGS = -O3 -Iinclude -arch=sm_86 -lcublas -rdc=true`; targets `worker` (`worker.cu` which `#include`s `kernels.cu`), `coordinator` (pure C++), `print_arch`.
Treated as a pure-compiler project. ext_type = "make".

## Port strategy
Strategy A spirit (minimal-footprint compat header), adapted to a no-CMake project:
- There is no CMake `enable_language(HIP)` lever and no `.cu` retag mechanism; the build is direct `nvcc` invocation. Port = compile the existing `.cu` with `hipcc` instead of `nvcc`. hipcc accepts `.cu` directly and maps `<cuda_runtime.h>`, `cub::`, `thrust::`, `<cublas_v2.h>` etc. The CUDA spellings in these files (`cudaMalloc`, `cudaMemcpyAsync`, `cudaGraph*`, `cublas*`, `cub::`, `thrust::`) already resolve under hipcc via HIP's CUDA-compat layer + hipCUB/rocThrust/hipBLAS, so most files need ZERO symbol edits.
- The only source changes are the genuine semantic fault classes (warp size + masks), guarded by `#if defined(__HIP_PLATFORM_AMD__)` / `__HIP__` so the NVIDIA nvcc build is byte-for-byte unchanged.
- Add a small compat shim header (`egg_warp_compat.cuh`) holding the per-arch warp-width constant and the wave-agnostic warp-reduce/broadcast helpers; include it where WARP_SIZE / cub::WarpReduce / __shfl* are used. Keep the diff small.
- Add `build_hip.sh` (and a HIP arm in `d-eggs/Makefile` behind `USE_HIP`) recording the exact hipcc command for gfx90a (and a `gfx90a;gfx1100` multi-arch line for the warp-size test). Do NOT hardcode arch in a way a follower must edit; pass `--offload-arch` from a variable defaulting to gfx90a.
- Reason this is not Strategy B: no torch anywhere. Reason it is not literal Strategy A: no CMake to add `enable_language(HIP)` to; the compiler swap is the build flip.

Port-vs-rewrite: this is hand-written integer CUDA (int8/int32/int64 MACs, custom warp reductions, a LUT-free NTT/WHT, a Muon Newton-Schulz via cuBLAS GEMM). NO CUTLASS / CuTe / wgmma / Hopper / tensor-core paths. A mechanical HIP port is correct and sufficient; no AMD-native (CK/rocWMMA/MFMA) rewrite is warranted for correctness. (Perf tuning of the int8 MACs for CDNA could be a later pass but is out of scope for the port.)

## CUDA surface inventory
Files (lines): full_cuda_train_egg.cu (1145), full_cuda_train_egg_transformer.cu (844), full_cuda_train_egg_transformer_adam.cu (1171), full_cuda_train_transformer_adam_mgpu.cu (1767), egg_ntt.cuh (587), muon_internal.cuh (346); d-eggs/src/{kernels.cu (490), worker.cu (721)}, d-eggs/include/**.{cuh,h}.

- Kernels: `__global__` training/forward kernels (`train_sequence_kernel`, `compute_fitness_kernel`, `update_matrix_kernel`, `update_vector_kernel`, `generate_sequence_kernel`, adam/muon update kernels, NTT). Many `__device__ __forceinline__` helpers. Standard launch syntax `<<<grid,block,smem[,stream]>>>`.
- Warp intrinsics:
  - `__shfl_sync(0xFFFFFFFF, ...)` -- 64-bit broadcast helper `warpBroadcast` (egg.c lines 188-191).
  - `__shfl_down_sync(0xFFFFFFFF, df, off)` with off in {16,8,4,2,1} -- a LOGICAL-32 tree reduction (transformer, transformer_adam, mgpu, layers.cuh, adaptive_norm.cuh).
  - `__shfl_xor_sync(0xFFFFFFFF, val, 1)` -- RoPE pairwise (lane^1) neighbor swap (transformer_adam, mgpu, layers.cuh).
  - `__popc(i & coeff_idx)` -- NTT bit-reversal (egg_ntt.cuh, ntt.cuh); arch-agnostic popcount, fine.
  - `#define WARP_SIZE 32` in egg.c / transformer / transformer_adam; used as warp_id = tid/WARP_SIZE, lane_id = tid%WARP_SIZE, warps_per_block = BLOCK_THREADS/WARP_SIZE, AND as a DATA-LAYOUT STRIDE (loops `i += WARP_SIZE`, per-lane arrays `h_local[N_LAYERS][MAX_STRIDE=8]` indexed `[i/WARP_SIZE]`).
- CUB / hipCUB:
  - `cub::WarpReduce<long long>` per warp (egg.c train_sequence_kernel) + `cub::WarpReduce::TempStorage temp_storage[BLOCK_THREADS/WARP_SIZE]`.
  - `cub::BlockReduce<...,BLOCK_THREADS>`, `cub::BlockScan<...,BLOCK_THREADS>`, `cub::Max()` (egg.c block-per-perturbation variant; d-eggs/kernels.cu softmax/argmax/scan).
- Thrust / rocThrust: `thrust::reduce`, `thrust::transform_reduce`, `thrust::device_ptr`, `thrust::plus`, `thrust::device` (egg.c, transformer variants). rocThrust is a drop-in (same `thrust::` API + header paths under /opt/rocm/include).
- cuBLAS -> hipBLAS: ONLY in full_cuda_train_transformer_adam_mgpu.cu + d-eggs (Muon Newton-Schulz orthogonalization GEMMs; `cublasCreate/SetStream/<gemm>`). `perform_newton_schulz` + muon_internal.cuh.
- CUDA Graphs: d-eggs/src/worker.cu (`cudaStreamBeginCapture`/`EndCapture`, `cudaGraphInstantiate`, `cudaGraphLaunch`, `cudaGraphGetNodes`). HIP supports the hipGraph API 1:1; capture-around-update path needs a GPU smoke after port.
- Streams/events + pinned/managed memory: `cudaStreamCreate/Synchronize/Destroy`, `cudaMallocHost` (pinned, mgpu), `cudaMallocManaged` (d-eggs/worker.cu:215). All have direct HIP equivalents.
- Multi-GPU: mgpu file `cudaGetDeviceCount`/`cudaSetDevice` per-GPU contexts, no NCCL (sync is fitness-score broadcast only -- bandwidth-light, easy to port).
- Textures/surfaces: NONE. cuFFT/cuRAND/cuSPARSE: NONE (RNG is a hand-rolled hash `noise_from_hash`).
- NTT/WHT (egg_ntt.cuh, ntt.cuh): block-wide, `__syncthreads()`-synchronized, default `NTT_MODE=0` (disabled). Wave-agnostic.

## Risk list
1. WARP_SIZE-as-data-layout (PRIMARY, wave64 fault class). egg.c maps ONE perturbation to ONE warp and partitions HIDDEN_DIM across the 32 lanes (`i += WARP_SIZE`, per-lane array stride `MAX_STRIDE=8 = max HIDDEN_DIM 256 / 32`). The `cub::WarpReduce<long long>` summing per warp, the `warpBroadcast` (lane 0 -> all via `__shfl_sync`), `warps_per_block = BLOCK_THREADS/WARP_SIZE`, and `blocks = POPULATION_SIZE/warps_per_block` ALL assume a 32-lane warp. On gfx90a the physical wavefront is 64.
   - Decision per MULTI-ARCH standard: keep the LOGICAL warp at 32 lanes (a width-32 subgroup) on ALL arches. Width-32 logical-warp ops are arch-agnostic (guide: "width-32 LOGICAL-warp ops ... operate within a 32-lane subgroup regardless of the physical wavefront"). This keeps the per-lane data partition, MAX_STRIDE=8, and the host launch geometry IDENTICAL to CUDA and avoids re-architecting the kernel.
   - Concretely: define `EGG_WARP_SIZE = 32` for the LOGICAL warp (NOT the physical wavefront) on every arch -- this is correct because the algorithm's data tiling is fixed at 32, not derived from hardware width. Pin `cub::WarpReduce<long long, 32>` (explicit LOGICAL_WARP_THREADS=32) so it reduces the right 32-lane scope on a wave64 device (default would use 64). Make `__shfl*` use explicit width 32 (`__shfl_sync(mask,v,lane,32)`, `__shfl_down_sync(mask,v,off,32)`, `__shfl_xor_sync(mask,v,1,32)`). On wave64 two logical warps share a wavefront -- width-32 shuffles keep them independent.
   - Host warps_per_block/blocks stay computed from EGG_WARP_SIZE=32; do NOT switch them to runtime hipDeviceAttributeWarpSize (that would be 64 and break the data layout). This is the "logical warp" exception, not the "physical warp" host-query rule. Document why in notes.md.
2. 32-bit shuffle masks on wave64. `0xFFFFFFFF` covers only lanes 0-31. With explicit width=32 the mask semantics are per-subgroup and correct; verify no path relies on a full-64 mask. The `off=16` halving reduction is already a logical-32 reduction and is correct once width=32 is explicit.
3. RoPE `__shfl_xor_sync(...,1)` (lane^1 pairwise). With width=32 the XOR-1 partner stays within the 32-lane subgroup; safe. Without an explicit width on wave64 it is still lane^1 (partner in same physical wavefront half) but mask coverage is the concern; pin width=32.
4. cub::WarpReduce default width on HIP. hipCUB `WarpReduce<T>` default LOGICAL_WARP_THREADS = physical warp size (64 on gfx90a). MUST template it `<long long, 32>` to match the 32-lane data partition, else it sums across 64 lanes (32 of which are a different perturbation) -> wrong fitness/loss.
5. cub/hipCUB BlockReduce/BlockScan TempStorage reuse on a wave64 block (guide CV-CUDA class). d-eggs/kernels.cu and egg.c block variant reuse the same `TempStorage` union across back-to-back BlockReduce/BlockScan calls; a 64-thread sub-step lowers to one wavefront with no syncing epilogue. Add `__syncthreads()` between reused-TempStorage collectives. Audit each reuse site.
6. cuBLAS -> hipBLAS Newton-Schulz (mgpu + d-eggs muon). Watch hipBLAS v2 enum/handle differences and row/col-major + transpose flags in the GEMM chain; verify orthogonalization output, not just compile.
7. CUDA Graph capture (d-eggs/worker.cu). hipGraph is 1:1 but capture-mode + instantiate-once-then-relaunch needs a GPU smoke; capture of hipBLAS calls is the fragile part.
8. cudaMallocManaged on gfx90a (d-eggs/worker.cu:215) -- coarse-grained managed memory drops int atomicMin/atomicMax silently (guide cudaKDTree class). Audit for atomicMin/Max on the managed buffer; the update kernels use atomicAdd/atomicCAS (unaffected) -- confirm no atomicMin/Max RMW on managed memory.
9. `cudaMemcpyAsync` from pageable host buffer freed too soon (guide CV-CUDA class). mgpu copies fitness from `cudaMallocHost` pinned buffers (safe), but audit any async copy from a transient std::vector/local.
10. -ffp-contract: the integer MAC core is exact (no float drift), but the ES-AdamW optimizer (mgpu/adam) and adaptive-norm use float32. If any bit-exact float check is added to validation, pin `-ffp-contract=on`. Not expected to matter for loss-curve validation.
11. Determinism for validation: noise is a deterministic hash of (seed, index); the per-step `seed = time(NULL) ^ ...` makes runs differ by wall-clock. To prove correctness reproducibly, pin the seed in a small harness (or set step seed deterministic) and compare loss across two runs.
12. atomicAdd on `unsigned long long` / `long long` counters (d-eggs total_updates, fitness) -- supported on HIP, but confirm 64-bit atomic support path compiles.

## File-by-file change list
- NEW `egg_warp_compat.cuh`: define `EGG_WARP_SIZE 32` (logical, all arches); wave-agnostic helpers `eggWarpReduceSum` (wraps `cub::WarpReduce<T,32>`), `eggWarpBroadcast` (uses `__shfl_sync(mask,v,lane,32)`), and width-32 wrappers for `__shfl_down_sync`/`__shfl_xor_sync`. Guard HIP-specific bits on `#if defined(__HIP__)` (per task: key the shim on `__HIP__`, not `__HIP_PLATFORM_AMD__`). On CUDA it expands to the existing spellings so nvcc output is unchanged. Include `<cstring>`/`<cstdlib>` before any hip header here (guide gpuRIR class) -- though these files do not currently mix host memcpy in .cu device scope, keep the ordering safe.
- `full_cuda_train_egg.cu`: replace `#define WARP_SIZE 32` use of the warp-reduce with `cub::WarpReduce<long long,32>`; route `warpBroadcast` and the `__shfl_sync` through the width-32 wrappers; keep WARP_SIZE=32 for the data-layout math (correct on all arches). Audit the block-variant BlockReduce/BlockScan TempStorage reuse for `__syncthreads()`.
- `full_cuda_train_egg_transformer.cu`, `full_cuda_train_egg_transformer_adam.cu`: width-32 on the `off=16` `__shfl_down_sync` reductions and the RoPE `__shfl_xor_sync(...,1)`; `cub::BlockReduce<...,BLOCK_THREADS>` -- audit TempStorage reuse. `ALIGNED_DIM = (HIDDEN_DIM+31)&~31` is a 32-alignment for the logical warp -- keep.
- `full_cuda_train_transformer_adam_mgpu.cu`: above + cuBLAS->hipBLAS (compiles under hipcc as-is via compat, but verify GEMM transpose/layout and Newton-Schulz numerics); multi-GPU device-loop unchanged; `cudaMallocHost` pinned copies unchanged.
- `muon_internal.cuh`: cuBLAS Newton-Schulz GEMM chain -- verify under hipBLAS.
- `egg_ntt.cuh`: no change (block-wide `__syncthreads`, popcount, NTT_MODE=0 default). Verify it compiles under hipcc when a non-zero NTT_MODE is selected.
- `d-eggs/src/kernels.cu`: BlockReduce/BlockScan TempStorage `__syncthreads()` audit; width on any shuffles in included headers (layers.cuh, adaptive_norm.cuh).
- `d-eggs/src/worker.cu`: hipGraph smoke (capture/instantiate/launch); `cudaMallocManaged` atomicMin/Max audit.
- `d-eggs/include/model/layers.cuh`, `d-eggs/include/math/adaptive_norm.cuh`: width-32 on `__shfl_down_sync`/`__shfl_xor_sync`.
- NEW `build_hip.sh`: hipcc build lines for each top-level trainer; arch from `${EGG_HIP_ARCH:-gfx90a}` via `--offload-arch`, plus a `gfx90a;gfx1100` multi-arch line as the warp-size test.
- `d-eggs/Makefile`: add a `USE_HIP=1` branch swapping NVCC->hipcc, `-lcublas`->`-lhipblas`, `-arch=sm_86`->`--offload-arch=$(EGG_HIP_ARCH)`, keep `-rdc=true`. Leave the CUDA default path untouched.

## Build commands (gfx90a)
Lead trainer (simplest, no cuBLAS -- primary bring-up):
```
cd projects/egg.c/src
/opt/rocm/bin/hipcc -O3 --offload-arch=gfx90a -x hip full_cuda_train_egg.cu -o egg_hip
```
Multi-arch warp-size test (one fat binary; confirm BOTH code objects with `llvm-objdump --offloading egg_hip`):
```
/opt/rocm/bin/hipcc -O3 --offload-arch=gfx90a --offload-arch=gfx1100 -x hip full_cuda_train_egg.cu -o egg_hip_multi
```
Transformer/adam and mgpu (cuBLAS path):
```
/opt/rocm/bin/hipcc -O3 --offload-arch=gfx90a -x hip full_cuda_train_egg_transformer_adam.cu -lhipblas -o egg_transformer_adam_hip
/opt/rocm/bin/hipcc -O3 --offload-arch=gfx90a -x hip -rdc=true full_cuda_train_transformer_adam_mgpu.cu -lhipblas -o egg_transformer_mgpu_hip
```
d-eggs:
```
cd projects/egg.c/src/d-eggs && make USE_HIP=1 EGG_HIP_ARCH=gfx90a
```
(If hipcc does not auto-include the HIP runtime for `.cu`, add `-x hip`; verified hipcc accepts `.cu`.)

## Test plan
The project ships NO automated GPU test suite (only `d-eggs/test_ternary.cpp`, a CPU unit test for ternary packing, and two Python tokenizer scripts). Validation is therefore a real-GPU functional/behavioral run.

GPU-validatable slice (lead): `full_cuda_train_egg.cu`.
- Provide a small `input.txt` (tiny text corpus) in the run cwd.
- Run N training steps; the program prints `Step S | Loss: X.XXXX | ...` each step and periodically emits a colored generated-text sample.
- PASS criteria (correctness-first, behavioral):
  1. Loss decreases over steps (bits/byte trends down from ~8 toward the documented ~1.4-2 range on a small corpus) -- proves the int8 forward + ES fitness + update loop is numerically correct on AMD.
  2. The generated sample becomes increasingly text-like (not garbage/`.`-only), matching the README screenshot behavior.
  3. Determinism check: with a pinned seed, two runs produce a bit-identical loss sequence -- this is the decisive warp-reduce/shuffle-mask correctness fingerprint (a wave64 mask/width bug shows as wrong AND non-deterministic loss, per popsift/MPPI classes).
  4. Compare the AMD loss trajectory against an NVIDIA (or CPU `full_trained_egg.c`) reference trajectory on the same corpus/seed; they should track within ES noise.
- Secondary GPU slices: `full_cuda_train_egg_transformer_adam.cu` (RoPE shuffle + BlockReduce), then `full_cuda_train_transformer_adam_mgpu.cu` (hipBLAS Newton-Schulz + multi-GPU + behavioral loss), then `d-eggs` worker+coordinator (hipGraph + managed memory; run a short distributed session loopback if feasible, else a single-worker smoke).
- Multi-arch build gate: `egg_hip_multi` must emit gfx90a AND gfx1100 code objects (`llvm-objdump --offloading`), and the single-arch gfx90a run must pass the loss/determinism checks above. (gfx1100 runtime validation is the follower's job.)

Non-GPU regression set (must not regress):
- `g++ -O3 -Iinclude d-eggs/test_ternary.cpp` builds and PASSES (CPU ternary pack/unpack roundtrip) -- untouched by the port.
- CPU build `clang -O3 full_trained_egg.c -o egg` still compiles and runs (we touch only `.cu`/`.cuh`; this file is pure C and must be left byte-identical).
- NVIDIA nvcc build of every ported `.cu` must remain unchanged (all HIP-specific code behind `#if defined(__HIP__)`).

## Open questions
1. Does upstream `full_cuda_train_egg.cu` (block-per-perturbation `train_sequence_kernel` at line ~208 vs warp-per-perturbation at line ~439) actually launch the warp variant (confirmed: line 965 launches the warp variant)? Confirm the block variant is dead/unused so the BlockReduce/BlockScan TempStorage audit can be scoped or skipped.
2. cuBLAS Newton-Schulz: do the GEMMs assume column-major (cuBLAS native) layout that needs explicit handling under hipBLAS? Verify orthogonalization numerically, not just at compile.
3. d-eggs distributed run: can it be validated single-host (coordinator + one worker over loopback) on the assigned GPU, or is a single-worker direct-kernel smoke the practical ceiling for MOAT validation?
4. What corpus/step-count gives a quick-but-meaningful loss-drop signal for the validator (keep wall time modest on a shared 4-GCD host)?
