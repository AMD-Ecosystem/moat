# SpargeAttn -- ROCm/HIP port plan (lead: linux-gfx90a)

## Project
- Name: SpargeAttn
- Upstream: https://github.com/thu-ml/SpargeAttn
- Default branch: main
- Clone analyzed at: projects/SpargeAttn/src (read-only, depth=1)
- What it is: training-free sparse + quantized (INT8 / FP8) attention for diffusion/video transformers (CogVideo, Flux, Hunyuan, Wan). Built on the SageAttention2 kernels, themselves derived from FlashInfer.

## Existing AMD support
- None. README advertises Ampere (>=12.0), Ada FP8 (>=12.4), Hopper FP8 (>=12.3), Blackwell (>=12.8) only; setup.py SUPPORTED_ARCHS = {8.0, 8.6, 8.7, 8.9, 9.0} and hard-requires CUDA_HOME and nvcc >= 12.0. No HIP path, no OpenCL/Vulkan/SYCL path, no AMD fork among the 91 upstream forks (checked).
- Decision: this is NOT a mechanical CUDA->HIP port and NOT an "already supported" skip. The hot kernels are an AMD-native rewrite (reimplement, not port). Lead platform set to BLOCKED pending that rewrite -- see Port strategy and Risk list for why a hipify pass cannot produce a correct, let alone competitive, kernel.

## Build classification: torch-extension (Strategy B)
Evidence (setup.py):
- `from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME`
- two `CUDAExtension` modules: `spas_sage_attn._qattn` (csrc/qattn/*) and `spas_sage_attn._fused` (csrc/fused/*)
- `cmdclass={"build_ext": BuildExtension}`
- Python side (`spas_sage_attn/core.py`) does `import spas_sage_attn._qattn as qattn` / `import spas_sage_attn._fused as fused` and calls into the compiled ops directly.
ext_type set to `torch-extension` in upstream.json / status.json.

On a ROCm torch, `BuildExtension` would auto-hipify the `.cu`/`.cuh` and link amdhip64/c10_hip/torch_hip. That machinery is fine; the problem is the kernel content, not the build classifier.

## Port strategy: B hipify in principle, but BLOCKED -- requires an AMD-native kernel reimplementation
A torch-extension port normally means "build against ROCm torch, fix the wave-size fault class, done." That does NOT apply here because every compute kernel is written directly against NVIDIA tensor-core PTX that has no HIP/hipify translation:

- csrc/mma.cuh (39 inline-asm blocks): `mma.sync.aligned.m16n8k16 .f16/.f16/.f32`, `.m16n8k32 .s8.s8.s32`, `.m16n8k64 .s4.s4.s32`, `ldmatrix.sync.aligned.m8n8.x2/x4[.trans]`. These PTX ops do not exist on AMD; hipcc/hipify pass inline PTX through unchanged and it fails to assemble for gfx90a. The AMD analogue is the MFMA family (`__builtin_amdgcn_mfma_*` / rocWMMA / Composable Kernel), which has a completely different instruction set, tile shapes, and -- critically -- a different per-lane register-fragment layout.
- csrc/wgmma.cuh (11 asm blocks, SM90 path): `wgmma.mma_async` + smem matrix descriptors (`make_smem_desc`), `SM90_ENABLED` gated on `__CUDA_ARCH__ >= 900`. Hopper-only; no AMD equivalent at all.
- csrc/cp_async.cuh (8 asm blocks): `cp.async.cg/ca.shared.global`, `cp.async.commit_group`, `cp.async.wait_group`. Used by BOTH the qattn kernels and the _fused TransposePadPermute/MeanScale kernels (csrc/fused/fused.cu). On AMD async global->LDS copy is `__builtin_amdgcn_global_load_lds` / buffer intrinsics with different semantics; not a textual swap.
- csrc/numeric_conversion.cuh is only "Inspired by CUTLASS" (a comment) -- no actual cutlass/cute include anywhere in the tree. So this is NOT a CUTLASS/CuTe template port (which the guide says is unportable); it is a hand-rolled inline-PTX kernel, which is just as unportable mechanically but is at least a finite, readable surface to reimplement.

The decisive structural blocker is the fragment layout, not just the instruction names. The kernels hardwire the 32-lane MMA register distribution into shared-memory addressing and reductions:
- csrc/qattn/qk_int_sv_f16_cuda_sm80.cuh:197-199 compute smem offsets as `lane_id % 16`, `lane_id / 16`, `lane_id % 8 + (lane_id/16)*8`, `(lane_id/8) % 2` -- this is exactly the m16n8k16 / ldmatrix lane->element mapping for a 32-thread warp.
- The online-softmax cross-lane reductions use `__shfl_xor_sync(0xffffffff, x, 0x1/0x2/0x4/0x8/0x10)` with strides chosen to match where the MMA accumulator distributes a row across the 32 lanes (csrc/qattn/attn_utils.cuh:442-443,495-496,772-773; qk_int_sv_*_sm8x.cuh; csrc/reduction_utils.cuh:30,42,124,173 also pass an explicit width of 32).
On gfx90a a wavefront is 64 lanes and MFMA distributes a tile across 64 lanes in a layout unrelated to the m16n8kK 32-lane layout, so the smem permutation, the ldmatrix-equivalent load, the MMA, and the shuffle-reduction strides ALL have to change together and consistently. There is no per-line fix; the kernel body is rebuilt around the MFMA fragment.

Total kernel surface to reimplement: csrc/*.cuh + csrc/qattn + csrc/fused ~= 7150 LOC of .cu/.cuh, with per-arch instantiation generators (instantiations_sm80/sm89/sm90 autogen.py, ~65 lines each) producing the launch matrix.

Recommended path when unblocked (correctness-first):
1. Target a single AMD-native variant first: INT8 QK x FP16/BF16 PV (the sm80 path, `qk_int_sv_f16_cuda_sm80`), since CDNA has strong INT8 and f16 MFMA and no FP8 dependency. Reimplement the QK^T and PV matmuls with rocWMMA (or CK / raw `__builtin_amdgcn_mfma_i32_16x16x16_i8` and `mfma_f32_16x16x16_f16`), keeping the FlashAttention online-softmax math but rederiving the smem layout and the cross-lane softmax reductions for the MFMA accumulator layout on 64 lanes.
2. Reimplement the _fused helpers (TransposePadPermute, MeanScale) -- these are simpler (no tensor cores) but still use cp.async; swap the async LDS prefetch for the AMD equivalent or a plain global->LDS load + __syncthreads, and re-check the permute, which exists to feed the FP8 MMA layout (re-derive for the AMD layout or drop if the AMD kernel does not need the permute).
3. FP8 (sm89/sm90) is a second phase: gfx90a has no FP8 MFMA (FP8 matrix is gfx94x/CDNA3+). On gfx90a either dequantize FP8->f16 and run the f16 MFMA path, or scope FP8 out for the lead platform. Keep the Python autotune/quant logic (spas_sage_attn/*.py) intact; it is arch-agnostic except where it assumes the C++ op names.
4. The block-sparse mask machinery and per-block/per-warp INT8 quant live in Python (spas_sage_attn/quant_per_block.py, quant_per_warp_cuda.py, autotune.py, core.py) and a Triton example (Triton_SpargeAttn/triton_kernel_example.py is a standalone demo, NOT wired into the package). These are portable and do not gate the decision.

This is the same class as the CUTLASS/Hopper-gated raft kernels deferred in MEMORY/PORTING_GUIDE: a real AMD-native rewrite, not a hipify pass. It is a multi-day kernel-engineering effort, so the lead is blocked and we move on rather than attempt a translation that cannot compile.

## CUDA surface inventory
- Kernels (`__global__`): qattn attention kernels (sm80 INT8+f16, sm89 INT8+FP8, sm90 INT8+FP8 wgmma) instantiated via the three autogen.py generators; _fused TransposePadPermuteKernel + MeanScaleKernel (csrc/fused/fused.cu).
- Tensor-core PTX: mma.sync (m16n8k16 f16/f16/f32, f16/f16/f16; m16n8k32 s8/s8/s32; m16n8k64 s4/s4/s32), ldmatrix.sync m8n8.x2/x4[.trans] (csrc/mma.cuh); wgmma.mma_async + smem descriptors (csrc/wgmma.cuh, SM90). -> AMD: MFMA via rocWMMA / CK / __builtin_amdgcn_mfma_* ; NO wgmma equivalent. REWRITE.
- Async copy PTX: cp.async.cg/ca + commit_group/wait_group (csrc/cp_async.cuh), used by qattn and _fused. -> AMD global_load_lds / buffer intrinsics, different semantics. REWRITE.
- Warp intrinsics: `__shfl_xor_sync(0xffffffff, ...)` with width 32 and MMA-layout-specific strides; `__syncwarp()`; static `__shared__ T shared[32]` warp-reduce scratch sized to 32 warps (csrc/reduction_utils.cuh). -> 32-bit full masks are wrong on wave64; widths and strides are coupled to the (to-be-replaced) fragment layout. REWRITE WITH the kernel.
- Numeric: hand-rolled fp16/bf16/fp8/int8/int4 conversions (csrc/numeric_conversion.cuh, "inspired by CUTLASS", no cutlass include); math.cuh fast-math helpers. -> mostly portable; fp8 type (`__nv_fp8`) needs the HIP fp8 type and only has matrix HW on CDNA3+.
- Libraries: none. No cuBLAS/cuFFT/cuRAND/cuSPARSE/Thrust/CUB. Pure torch + hand kernels. `extra_link_args=['-lcuda']` (driver API link) on the qattn module -> drop / replace with the HIP runtime on AMD.
- Memory/streams: standard torch tensor pointers + default-stream `<<<grid,block>>>` launches; no explicit pinned/managed/streams/events surface beyond torch's.

## Risk list
- (BLOCKER) NVIDIA tensor-core + async PTX with no HIP translation: mma.sync, ldmatrix, wgmma, cp.async. Not hipify-able; AMD MFMA has different shapes and lane layout.
- wave64 vs warp32: every `__shfl_xor_sync` mask is `0xffffffff` (32-bit) and the shuffle strides + smem permutation encode the 32-lane MMA fragment. On gfx90a (64-lane) these are all wrong and must be re-derived for the MFMA accumulator layout. reduction_utils.cuh `shared[32]` warp-scratch and `blockDim.x / 32` warp counts also assume 32.
- FP8 on gfx90a: no FP8 matrix instruction on CDNA2; the sm89/sm90 FP8 kernels have no direct gfx90a target (need dequant-to-f16 or scope out). FP8 matrix is CDNA3+ (gfx94x).
- Hopper-only wgmma path (csrc/wgmma.cuh, qk_int_sv_f8_cuda_sm90): no AMD analogue; this variant is simply not portable and should be dropped for AMD, not translated.
- Layout-coupled correctness: smem swizzle (permuted_smem.cuh get_permuted_offset), the ldmatrix loads, the MMA, and the softmax shuffle reductions are one interlocked design; a partial port silently corrupts the online-softmax accumulation. Must be reimplemented and validated as a unit.
- Numeric tolerance: this is approximate (quantized + sparse) attention; validation is tolerance-based (cosine sim / relative error vs a dense reference), not bit-exact -- pick the tolerance from the upstream accuracy claims, not from a bitwise diff.

## File-by-file change list (for the eventual AMD-native reimplementation; lead is blocked now)
- setup.py: add a ROCm branch -- detect `torch.version.hip`, skip the nvcc/CUDA_HOME/SUPPORTED_ARCHS gate, set HIP arch (gfx90a), drop `-lcuda` and the nvcc-only flags (`-Xptxas`, `-gencode`, `--use_fast_math` -> `-ffast-math`), select the AMD source set, keep ENABLE_BF16.
- csrc/mma.cuh: REWRITE -- new MFMA wrappers (rocWMMA or __builtin_amdgcn_mfma_*) under `#ifdef __HIP_PLATFORM_AMD__`, keep the CUDA PTX path under #else.
- csrc/wgmma.cuh: exclude from the AMD build (no equivalent).
- csrc/cp_async.cuh: AMD async/blocking global->LDS path under a HIP guard.
- csrc/qattn/qk_int_sv_f16_cuda_sm80.cuh (+ _f8_sm89/_sm90): rederive smem offsets, loads, MMA calls, and softmax shuffle reductions for the MFMA layout; sm90 dropped on AMD, sm89 FP8 -> dequant-to-f16 or deferred.
- csrc/qattn/attn_utils.cuh, reduction_utils.cuh, permuted_smem.cuh: warp-size and fragment-layout abstraction (kWarpSize per arch; 64-bit shuffle masks; layout-driven strides).
- csrc/fused/fused.cu: cp.async swap; re-check the FP8-feed permute.
- csrc/qattn/instantiations_sm80/sm89/sm90/autogen.py: regenerate the AMD launch matrix (drop sm90, gate FP8).
- numeric_conversion.cuh / math.cuh: HIP fp8 type + fast-math intrinsic equivalents; mostly mechanical.
- Python (spas_sage_attn/*.py, core.py): keep; only adjust if op signatures/names change in the AMD build.

## Build commands (gfx90a) -- to be used once unblocked
- Use a ROCm PyTorch (image rocm/dev-ubuntu-24.04:7.2.x-complete or a ROCm torch venv).
- `PYTORCH_ROCM_ARCH=gfx90a python setup.py develop` (or `pip install -e .`) after the setup.py ROCm branch lands; BuildExtension hipifies the selected sources.
- Do not build under agent_space and do not hold a GPU during planning; this is a no-GPU planning task.

## Test plan
- No CI / no unit-test suite in-repo. The exercisable surface is:
  - `evaluate/*.py` (cogvideo_example.py, flux_example.py, hunyuan_example.py, wan_example.py) -- end-to-end diffusion generation; heavy (model weights), use for a final sanity gen, not the unit gate.
  - A direct op-level harness (to be written for validation): call `spas_sage_attn.core` block-sparse attention on random Q/K/V, compare against a dense reference (torch SDPA / manual softmax(QK^T/sqrt(d))V) by cosine similarity and relative L2, at the accuracy bar the paper/README claims. This is the real GPU correctness gate.
  - Determinism: fixed-seed, two runs bit-identical for the deterministic (non-atomic) reductions; tolerance-based for the quantized path.
- Non-GPU regression set: none in-repo to protect.

## Open questions
1. Scope FP8 (sm89/sm90) out of the gfx90a lead entirely (gfx90a has no FP8 matrix HW), making the lead deliverable the INT8-QK x f16-PV sparse path, and pick FP8 up on a CDNA3 (gfx94x) follower? Recommended: yes.
2. rocWMMA vs Composable Kernel (ck_tile) vs raw MFMA builtins for the matmuls? Guide prefers ck_tile for new AMD-native work; rocWMMA is the lowest-friction drop-in for a fragment-style kernel. Decide at reimplementation start.
3. Is an AMD-native sparse-attention rewrite the best use of effort vs adopting an existing AMD attention kernel (e.g. CK fused attention / a ROCm flash-attention) and layering SpargeAttn's block-sparse mask + INT8 quant on top? Likely the latter is faster to correctness; evaluate before writing MFMA from scratch.

## Disposition
Lead platform linux-gfx90a: BLOCKED (blocked=true). Reason: all compute kernels are hand-written NVIDIA tensor-core + async PTX (mma.sync / ldmatrix / wgmma / cp.async) with the 32-lane fragment layout baked into smem addressing and softmax shuffle reductions; there is no portable kernel path and no hipify translation. A correct gfx90a port requires an AMD-native MFMA (rocWMMA/CK) reimplementation of the attention kernels (~7k LOC kernel surface), with FP8 paths having no gfx90a HW. This is a reimplement-not-port effort deferred per the perf-critical-kernel guidance; moving on.
