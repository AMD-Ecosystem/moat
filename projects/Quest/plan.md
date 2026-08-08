# Quest -- ROCm/HIP port plan (lead: linux-gfx90a / MI250X, ROCm 7.2.1)

## Project
- Name: Quest
- Upstream: https://github.com/mit-han-lab/Quest (mit-han-lab/Quest)
- Default branch: main
- Identity: this is **mit-han-lab Quest** -- "Query-Aware Sparsity for Efficient Long-Context LLM Inference" (arXiv 2406.10774), a PyTorch LLM-inference framework that accelerates long-context self-attention by loading only the top-K critical KV-cache pages. It is NOT the QuEST quantum state-vector simulator (the task brief's tentative guess was wrong; confirmed via README.md, pyproject.toml `name = "quest"`, and the kernel surface: paged-KV attention, page metadata min/max estimation, top-K page selection).

## Existing AMD support
None. Exhaustive grep for hip/rocm/amd/gfx/USE_HIP/wavefront across all .cu/.cuh/.h/CMake/py/md/toml found zero hits. There is no `GPUACCEL`/`USE_HIP`/`ENABLE_HIP` option and no `hip`/`gpu` backend dir. The build is CUDA-only (`project(... LANGUAGES CUDA CXX)`, `CMAKE_CUDA_ARCHITECTURES native`). Decision: this is a fresh CUDA-to-HIP port, NOT an enablement task. (Contrast with the QuEST simulator, which does ship a multi-backend CPU/CUDA/HIP build; Quest does not.)

## Build classification: torch-extension (CMake + Torch), NOT pure CMake
Two distinct build surfaces in the repo:

1. **`quest/ops/` -- the end-to-end operators (the real product).** `quest/ops/CMakeLists.txt` does `find_package(Torch REQUIRED)`, `find_library(TORCH_PYTHON_LIBRARY ...)`, builds a `pybind11_add_module(_kernels ...)` over `csrc/*.cu`, and links `${TORCH_LIBRARIES}` + `raft::raft` + `Torch`. Built via `quest/ops/setup.sh` (cmake + ninja, `-DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')`). This is the surface exercised by `quest/tests/*.py` through `import quest.utils`.
   Evidence: `quest/ops/CMakeLists.txt` lines `find_package(Torch REQUIRED)`, `pybind11_add_module(_kernels MODULE ... ${PYTORCH_SOURCES})`, `target_link_libraries(_kernels PRIVATE ${TORCH_LIBRARIES} raft::raft ...)`.
2. **`kernels/` -- standalone nvbench benchmarks + googletest unit tests.** `kernels/CMakeLists.txt` `project(bsk LANGUAGES CUDA CXX)`, builds `bench_*`/`test_*` executables linking `raft::raft`, `nvbench::main`, `gtest`. This is dev tooling, not shipped, and pulls in **nvbench (NVIDIA-only)**.

Per PORTING_GUIDE "Build classification": a CMake build that finds Torch and uses its extension machinery is a **pytorch extension -> Strategy B**. The deciding lines are `find_package(Torch REQUIRED)` + `pybind11_add_module` + `${TORCH_LIBRARIES}` in `quest/ops/CMakeLists.txt`. However the usual Strategy-B "torch hipifies the sources at build time" does NOT apply automatically here, because the module is built through **CMake**, not `torch.utils.cpp_extension.CUDAExtension` (which is what triggers torch's auto-hipify). So this is a hybrid: a Torch extension whose `.cu` are compiled by a CMake `CUDA`/`HIP` language target. The port must hipify by hand under a `USE_HIP` CMake option (Strategy-A mechanics) while linking ROCm Torch (Strategy-B environment).

Recommended ext_type: `torch-extension` (set in upstream.json + status.json).

## The wall: flashinfer PTX FlashAttention-2 kernels + raft + nvbench
The Quest-specific C++ is THIN (csrc wrappers + one estimate kernel + one rms_norm). The heavy lifting is a vendored hard-fork of **flashinfer's** CUDA attention kernels under `kernels/include/{decode,prefill}/*.cuh`, which `#include "flashinfer/..."` from the flashinfer submodule pinned at `9f49803b` (a 2024 CUDA/PTX-only commit, long before flashinfer's later partial ROCm work). These headers are built on NVIDIA-specific primitives that have NO HIP spelling:
- `flashinfer/mma.cuh` -- `mma::mma_sync_m16n16k16_row_col_f16f16f32` / `...f16f16f16` (Tensor Core `mma.sync` PTX) used throughout `prefill.cuh` (lines ~309-320, ~495-498).
- `flashinfer/permuted_smem.cuh` -- `ldmatrix_m8n8x4` / `ldmatrix_m8n8x4_trans` (PTX `ldmatrix`), `prefill.cuh` ~292, ~302, ~491.
- `flashinfer/cp_async.cuh` -- `cp_async::pred_load` / `commit_group` / `wait_group` (PTX `cp.async`), pervasive in both `decode_attn.cuh` and `prefill.cuh`.
- `flashinfer/math.cuh` -- `math::ptx_exp2` (PTX `ex2.approx`), `cooperative_groups`, `cuda::pipeline`.
- `prefill.cuh:52` hardcodes `constexpr uint32_t warp_size = 32;` and the entire fragment/tiling math is built around a 32-lane warp + 16x16x16 m16n16k16 MMA fragment layout.

There is NO mechanical cuda-to-hip alias for `mma.sync`/`ldmatrix`/`cp.async`. A correctness-first HIP port of these kernels requires REWRITING them against AMD-native primitives: `__builtin_amdgcn_mfma_*` (or rocWMMA) for the 16x16x16 matmul, `__builtin_amdgcn_ds_read_b128`/plain shared-memory loads for the ldmatrix transposes, and direct global->shared copies (CDNA has no `cp.async`; ROCm 6+/7 has limited async-copy builtins on gfx94x, not gfx90a) for the pipelined loads. The 32-lane fragment tiling must be reworked for wave64 on gfx90a. This is a per-PORTING_GUIDE "perf-critical attention kernel, NVIDIA-tuned (no straight HIP translation)" case.

Two further hard dependencies:
- **raft (RAPIDS) `raft/matrix/detail/select_k-inl.cuh`** -- `topk.cu`'s `decode_select_k` uses raft's radix select_k directly (`raft::matrix::detail::select::radix::impl`, `SelectAlgo::kRadix8bits`). raft is pinned via rapids-cmake to `branch-24.02` and is a CUTLASS-consuming RAPIDS library. raft IS a MOAT project (and rmm/raft ROCm work is in flight), so the topk path should ride on a ported raft rather than the NVIDIA raft 24.02. This adds a MOAT dependency: `depends_on = [raft]` (and transitively rmm).
- **nvbench (NVIDIA-only)** -- only used by the `kernels/` bench/test executables. nvbench has no ROCm port; the `bench_*` targets are out of scope. The `test_*` gtests in `kernels/` do not need nvbench (`gtest` only) but they `#include` the same flashinfer kernels, so they share the wall.

cuQuantum/custatevec: N/A (this is not the quantum simulator; no cuQuantum dependency exists).

## Port strategy
**Strategy B environment (ROCm Torch) + Strategy-A `USE_HIP` CMake option, staged, correctness-first.** Because the FA2 kernels are an AMD-native rewrite (not a translation), do NOT attempt to port everything at once. Stage by what is GPU-validatable and how much rewrite each needs:

- **Stage 1 (mechanical, small, real GPU value): rms_norm + topk + page-append + the estimate score kernel.**
  - `rms_norm.cu` is a self-contained CUTLASS-derived warpReduce/blockReduce RMSNorm with `__shfl_xor_sync(0xffffffff, v, mask, 32)` and `wid = tid>>5` / `lane = tid&0x1f` -- a textbook wave64 fault (warp-size 32 hardcode; `__GFX9__` constant + 64-bit-aware reduce; the `>>5`/`&0x1f` and `blockDim.x/32` must become `/warpSize`). Mechanical HIP port.
  - `topk.cu` -> route through a ported raft's select_k (dependency build).
  - `page.cu` (AppendPagedKVCacheDecode/Prefill) and `estimate.cu` (page min/max metadata + estimate-score) -- check whether the flashinfer helpers they pull are pure data-movement (`paged_kv_t`, append) vs MMA. The append/estimate paths look free of MMA (they are gather/scatter + a small score), so they may port mechanically; confirm at porter time.
- **Stage 2 (AMD-native rewrite, large): decode_attn.cuh + prefill.cuh.** Rewrite the FA2 inner loop on MFMA/rocWMMA + LDS staging + wave64 tiling. This is the bulk of the effort and the genuine perf-critical work. A mechanical translation is not possible (no `mma.sync`/`ldmatrix`/`cp.async` HIP spelling).

Mechanics for both stages: add one `cuda_to_hip.h` compat header for the toolkit symbols the csrc use (`cuda_fp16.h`/`cuda_bf16.h` -> `hip/hip_fp16.h`/`hip_bf16.h`, stream/error/launch), add a `USE_HIP` option to `quest/ops/CMakeLists.txt` that `enable_language(HIP)`, marks `csrc/*.cu` `LANGUAGE HIP`, sets `HIP_ARCHITECTURES` from `${CMAKE_HIP_ARCHITECTURES}` (default gfx90a only when unset -- never a literal), and links ROCm Torch + a ported raft. Keep the NVIDIA path byte-for-byte under `else()`.

## CUDA surface inventory
- Kernels (Quest-owned): `rms_norm.cu` (RMSNorm e8/general, warp+block reduce), `estimate.cu` (page min/max metadata + estimate-score), `page.cu` (paged-KV append decode/prefill), `bsk_ops.cu` (dispatch glue).
- Kernels (vendored flashinfer, the wall): `decode_attn.cuh` (single-query paged decode FA), `prefill.cuh` (FA2 prefill with MMA), `decode_page.cuh`, `decode_handler.cuh`, all consuming `flashinfer/{mma,cp_async,permuted_smem,math,vec_dtypes,rope,state,cascade,layout}.cuh`.
- Warp intrinsics / wave64 locus: `rms_norm.cu` `__shfl_xor_sync(0xffffffff,...,32)`, `tid>>5`, `tid&0x1f`, `shared[NUM][33]`, `blockDim.x/32` -- the probability/normalization reduction the brief anticipated. `prefill.cuh` `constexpr warp_size = 32` + MMA fragment layout. decode uses `static_assert(bdx <= 32)` (block-dim-x tied to warp width).
- Tensor Core MMA: `mma_sync_m16n16k16_*` (PTX) -- rewrite to MFMA/rocWMMA.
- Async copy: `cp_async::*` (PTX `cp.async`) -- rewrite to direct LDS staging on gfx90a.
- ldmatrix: `ldmatrix_m8n8x4[_trans]` (PTX) -- rewrite to LDS loads.
- Libraries: raft select_k (-> a ported raft), Torch (-> ROCm Torch), pybind11/googletest (arch-agnostic), nvbench (NVIDIA-only, out of scope).
- Thrust/CUB: none directly in Quest csrc; only transitively inside raft (handled by the raft port). cuBLAS/cuFFT/cuRAND/cuSPARSE: none. Textures/surfaces: none. Pinned/managed memory: none in Quest csrc. Streams/events: standard, via Torch's current stream.

## Risk list
- **Attention-kernel rewrite (highest risk/effort):** `mma.sync`+`ldmatrix`+`cp.async` have no HIP spelling; the FA2 decode/prefill kernels need an AMD-native (MFMA/rocWMMA + LDS) rewrite, not a translation. Plan to land Stage 1 first and treat Stage 2 as its own deliverable.
- **wave64 in rms_norm:** hardcoded warp=32 (`__shfl_xor_sync(...,32)`, `>>5`, `&0x1f`, `blockDim.x/32`, `shared[NUM][33]`). Use the PORTING_GUIDE `kWarpSize` pattern (`__GFX9__` -> 64) and a 64-entry warp-shared bound; the classic unrolled warp-synchronous reduction tail (if present) races on wave64 (PORTING_GUIDE MPPI lesson) -- prefer a `__syncthreads`-tree reduction.
- **raft dependency (build-gating):** topk uses raft 24.02 radix select_k; must build against a ported raft. `depends_on=[raft]` (raft itself depends on rmm). Selector will not pick Quest until raft's lead platform is `completed`. Also raft's CUTLASS-based primitives are why raft is its own MOAT port. The radix select itself may carry wave64 / `begin_bit` hazards already catalogued under cudaKDTree -- inherit raft's fixes.
- **nvbench out of scope:** the `kernels/` bench targets cannot build on ROCm; restrict the ROCm build to the `_kernels` pybind module (+ optionally the gtest `test_*` once the kernels port).
- **flashinfer submodule pin:** 9f49803 is CUDA/PTX-only; do not expect to swap in upstream flashinfer's later ROCm path (different API, different commit). The rewrite stays in Quest's vendored copy under `USE_HIP` guards.
- **fp16/bf16 headers:** `cuda_fp16.h`/`cuda_bf16.h`/`nv_half`/`nv_bfloat16` -> HIP equivalents in the compat header.
- **`-ffp-contract`, `__fsqrt_rn`, NaN min/max:** the attention/rmsnorm reductions compare against a torch CPU/GPU reference at rtol/atol 5e-3 (fp16) / 2e-2 (bf16) -- loose enough that the 1-ULP fault classes are unlikely to fail, but keep `-ffp-contract=on` in `CMAKE_HIP_FLAGS` per guide.
- **Configurable arch:** never hardcode gfx90a literal; drive `HIP_ARCHITECTURES` from `${CMAKE_HIP_ARCHITECTURES}` so gfx1100/gfx1151 followers need no source edit (and note: any MFMA rewrite is wave64/CDNA-specific and will need a wave32 path for RDNA followers -- flag for the delta-plan).

## File-by-file change list (Stage 1 first)
- `quest/ops/CMakeLists.txt`: add `option(USE_HIP ...)`, `enable_language(HIP)` branch, `set_source_files_properties(csrc/*.cu LANGUAGE HIP)`, `HIP_ARCHITECTURES` from cache var, link ROCm Torch + jeffdaily raft; keep CUDA path under `else()`. Add `--offload-compress` + `-ffp-contract=on` to HIP compile opts.
- `quest/ops/cmake/get_raft.cmake`: point at a ported raft (or `CPM_raft_SOURCE` to the locally-built `_deps/raft`); align RAPIDS_VERSION to the ported raft.
- NEW `quest/ops/csrc/cuda_to_hip.h`: alias the toolkit symbols csrc use (fp16/bf16 headers and types, stream/error). Force-include on the HIP target.
- `quest/ops/csrc/rms_norm.cu`: wave-size abstraction; replace `__shfl_xor_sync(0xffffffff,...,32)`/`>>5`/`&0x1f`/`/32`/`[33]` with `kWarpSize`-driven forms; verify reduction has no unsynced wave-synchronous tail.
- `quest/ops/csrc/{topk,page,estimate,approx_attn,batch_prefill,bsk_ops}.cu`: build under HIP; fix only what the compiler flags. topk rides jeffdaily raft.
- `kernels/include/{decode/decode_attn.cuh, prefill/prefill.cuh}` + the vendored `flashinfer/*` headers: Stage-2 AMD-native rewrite of MMA/cp.async/ldmatrix, guarded by `USE_HIP`.
- `quest/ops/setup.sh`: optional `USE_HIP` passthrough so the existing build path works on ROCm.

## Build commands (gfx90a)
Prereq: ROCm 7.2.1 Torch in the env; a ported raft built+installed into `_deps/raft` (see DEPENDENCIES.md), pointed at via `CPM_raft_SOURCE` or `CMAKE_PREFIX_PATH`.
```
# from quest/ops, ROCm Torch active
mkdir -p build && cd build
cmake -GNinja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" \
  -DCPM_raft_SOURCE=/var/lib/jenkins/moat/_deps/raft \
  ..
ninja
# setup.sh symlinks the built _kernels*.so up to quest/ops/ for import
```
Do NOT trial-compile from the MOAT repo root (a prior trial leaked `ccaff89d*-{amplitude,phase,basis,iqp}.o` into the root); use `agent_space/` if a no-GPU trial compile is wanted.

## Test plan
Real GPU gate (the project's own pytest suite under `quest/tests/`, which imports the built `_kernels` and compares against a torch CPU/analytic reference to rtol/atol 5e-3 fp16 / 2e-2 bf16):
- `pytest quest/tests/test_rope.py` -- pure-torch RoPE reference; closest thing to a non-GPU regression baseline and the lowest-risk first green.
- `pytest quest/tests/test_topk.py` -- exercises raft select_k via the ported raft (compares collected attention scores, not exact indices).
- `pytest quest/tests/test_estimate.py` -- page min/max metadata + estimate-score vs CPU reference.
- `pytest quest/tests/test_decode_attention.py`, `test_prefill_attention.py`, `test_approx_attention.py` -- the FA2 decode/prefill/approx paths; these are the Stage-2 gate and will not pass until the attention rewrite lands.
Staged success criteria: Stage 1 = rope + topk + estimate (+ rms_norm if a test covers it) PASS on gfx90a; Stage 2 = decode/prefill/approx_attention PASS. The `kernels/` gtests (`test_batch_decode`, `test_page`, `test_prefill`, `test_max_possible`) are a secondary GPU gate once the kernels port (no nvbench needed for `test_*`); the `bench_*` nvbench targets are out of scope.
Non-GPU regression set: the pure-Python evaluation harness under `evaluation/` and `quest/models|utils` import paths must still import; `test_rope.py` doubles as the CPU-reference sanity.

## Disposition
Proceed with the port (fresh CUDA-to-HIP, Strategy B/A hybrid), staged. Set linux-gfx90a -> planned. Record `depends_on=[raft]`. NOT a skip (no existing AMD support), NOT blocked (no proprietary wall: raft is portable via the MOAT raft port; the attention kernels are an AMD-native rewrite, which is in-scope MOAT work per the perf-critical-kernel policy; no cuQuantum/custatevec).

## Open questions
- Does `estimate.cu`/`page.cu` pull any MMA/cp.async transitively from the flashinfer headers, or only data-movement helpers? If MMA-free they port mechanically in Stage 1; confirm at porter time by compiling them in isolation.
- raft 24.02 vs the ROCm raft branch jeffdaily ports: API drift in `select_k-inl.cuh` between 24.02 and the ported branch may need a thin shim. Confirm the ported raft exposes the same `raft::matrix::detail::select::radix::impl` entry points.
- Stage-2 scope: is a wave64 MFMA FA2 rewrite within one porter pass, or should it be split decode-first then prefill? Likely split; decode (no prefill MMA tiling, simpler) is the better first AMD-native target.

## Delta plan: gfx1100 / gfx1151 (followers, on demand)
RDNA is wave32. Stage 1 rms_norm with the `kWarpSize` abstraction is wave-agnostic and should need no change. The Stage-2 attention rewrite is the risk: if it is built on gfx90a MFMA (CDNA `__builtin_amdgcn_mfma_*` / wave64 fragment tiling), RDNA has no MFMA -- the follower needs a WMMA (`__builtin_amdgcn_wmma_*`, RDNA3 16x16x16) wave32 path or a scalar fallback. Plan the Stage-2 kernel with a wave-size-parameterized fragment layout from the start to minimize follower divergence. Validate followers on their own hardware; do not re-plan.
