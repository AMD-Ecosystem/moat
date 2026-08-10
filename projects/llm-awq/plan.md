# Port plan: llm-awq

## Project
- Name: llm-awq
- Upstream: https://github.com/mit-han-lab/llm-awq (MLSys 2024; AWQ -- Activation-aware Weight Quantization, INT4/INT3 W4A16 + W8A8 LLM inference)
- Default branch: main
- Pinned base commit (depth-1 clone): d6e797a "[Major] set VILA deps as optional (#294)"
- The native extension is `awq_inference_engine`, built from `awq/kernels/` (a PyTorch CUDAExtension).

## Existing AMD support -- assessment and decision
Decision: PROCEED with a from-scratch HIP port of this repo's `awq_inference_engine`. No existing port to adopt or improve.

Evidence gathered (the required existing-support checks):
- README/docs grep: the ONLY AMD mention is README line 60, a 2024/05 AMD community blog "AMD adopts AWQ to improve LLM serving efficiency." That is adoption of the AWQ *method/algorithm*, not a HIP port of THIS repo's kernels. (URL now 301-redirects to amd.com/community-updates.)
- Web search ("llm-awq ROCm/AMD/HIP", "MI300 gfx9", "awq_inference_engine port"): AMD/ROCm run AWQ via vLLM and via HuggingFace's Exllama path (AWQ weights converted to Exllama/GPTQ format at load and run by ExllamaV2 ROCm kernels) -- i.e. SEPARATE kernel implementations in downstream frameworks. None is a port of `mit-han-lab/llm-awq`'s `awq_inference_engine`.
- `gh api repos/mit-han-lab/llm-awq/forks`: no fork under the ROCm/AMD/GPUOpen orgs and none with rocm/hip/amd in the name (first page).
- `gh pr list --repo mit-han-lab/llm-awq --state all --search "ROCm OR HIP OR AMD"`: no ROCm/HIP/AMD enablement PR (the one hit is an unrelated 2023 README PR).
- No ROCm/* repo named "awq" (rocm.docs.amd.com has no separately-named llm-awq port; AWQ-on-ROCm docs all point at vLLM).

Authoritativeness judgment: there is no AMD-authored or community HIP fork of this engine at all -- nothing to reuse, improve, or skip behind. The value MOAT adds is a clean ROCm/HIP port of the original reference engine itself. AMD's *method-level* support via vLLM/Exllama does NOT supersede porting this codebase (a different deliverable, different kernels).

## Build classification: torch-extension (Strategy B)
Evidence:
- `awq/kernels/setup.py` line 2: `from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension`; lines 28-49 build a single `CUDAExtension(name="awq_inference_engine", ...)` with `cmdclass={"build_ext": BuildExtension}`, `install_requires=["torch"]`.
- Root `pyproject.toml` pins `torch==2.3.0`, `torchvision==0.18.0`, `transformers==4.46.0` (CUDA-default wheels).
- ext_type confirmed `torch-extension` in upstream.json/status.json (already set; re-affirmed).

Strategy B consequence: torch's `BuildExtension` runs `torch.utils.hipify` over the extension `.cu`/`.cuh` at build time when built against a ROCm torch, and links `amdhip64`/`c10_hip`/`torch_hip`. Keep sources in CUDA spelling; do NOT add a `cuda_to_hip.h` and do NOT hand-rename runtime symbols. Fix only what hipify cannot translate (the fault classes below), guarded by `USE_ROCM`/`__HIP_PLATFORM_AMD__`.

## Port strategy: Strategy B (torch-hipify) + targeted PTX rewrites
This is the perf-critical-quantization case the guide flags (line 41 / 2026-05-30 changelog): the GEMM hot paths are hand-written NVIDIA tensor-core PTX. hipify does NOT translate inline PTX. So this is NOT a purely mechanical port -- it is Strategy B for everything hipify CAN do, plus a correctness-first manual rewrite of every inline-PTX/tensor-core kernel to a portable HIP equivalent.

Decision per kernel (port-vs-rewrite):
- GEMV kernels (`quantization/gemv_cuda.cu`, `quantization_new/gemv/gemv_cuda.cu`): SIMT, only `__shfl_*_sync` warp reductions, no PTX. MECHANICAL HIP port + warp-size fixes. These are the decode-stage path (the default `WQLinear.forward` uses `gemv_forward_cuda_new` when tokens < 8) -- so a working GEMV alone already enables single-token generation.
- GEMM kernels (`quantization/gemm_cuda_gen.cu`, `quantization_new/gemm/gemm_cuda.cu`, `w8a8/w8a8_gemm_cuda.cu`): hand-written `ldmatrix.sync` + `mma.sync.m16n8k16`/`m16n8k32` + `cp.async`/`cvta` + `lop3.b32` PTX. NONE of this compiles under clang/HIP. REWRITE the hot loop. Correctness-first options, in order of preference:
  1. Portable dequant + SIMT/`half2` FMA accumulation (drop the tensor-core path on AMD; keep CUDA's MMA path under `#if !defined(__HIP_PLATFORM_AMD__)`). Simplest, guaranteed-correct, slower. Valid first deliverable per guide (correctness-first mechanical port is acceptable even if a later AMD-native pass is wanted).
  2. AMD-native MFMA via rocWMMA (gfx90a has f16->f32 MFMA; the s8->s32 w8a8 path maps to int8 MFMA) for the performance-relevant rewrite. This is the eventual high-value target but is a larger effort; scope as a follow-on if option 1 validates first.
  Recommendation: land option 1 (portable correctness) as the lead port to unblock validation and the multi-arch PR, and register the rocWMMA/MFMA rewrite as deferred perf work. RDNA (gfx1100/gfx1201, wave32) has NO f32-input WMMA intrinsic (changelog 2026-06-09) and a different (16x16) MMA fragment shape than CDNA MFMA, so the option-2 rewrite would itself be arch-split -- another reason to ship the portable SIMT path first and treat wave32 uniformly.
- `lop3.b32` dequant (`quantization*/dequantize.cuh`): the INT4->FP16x2 bit trick. `lop3` is NVIDIA PTX (a 3-input LUT op). On HIP replace with the equivalent boolean expression the immLut encodes (`(a & BOTTOM_MASK) | I4s_TO_F16s_MAGIC_NUM` for the immLut 0xea used here = `(a & b) | c`), guarded by `__HIP_PLATFORM_AMD__`. Same bit result, no PTX. The follow-on `prmt`/`__byte_perm` uses in the same header also need a HIP expansion or `__builtin`-based replacement.
- The PTX semaphore (`quantization_new/gemm/semaphore.h`: `ld.global.acquire.gpu`/`st.global.release.gpu`): used for split-K reduction locks. Replace with `__atomic_load_n`/`__atomic_store_n` with `__ATOMIC_ACQUIRE`/`__ATOMIC_RELEASE` (HIP supports these) under a HIP guard; CUDA keeps the PTX.
- Attention (`attention/decoder_masked_multihead_attention*.{cu,h,hpp,...}` + `ft_attention.cpp`): FasterTransformer single-query masked MHA. Heavy half2/bf16 + warp ops + some PTX in the utils. It is in the MAIN setup.py sources (the `single_query_attention` binding). Assess whether the inference path under test actually invokes it (the qmodule path does not; it may be reached only by the TinyChat generation path / cache-attention). If the validation entrypoint exercises it, it needs the same warp-size + bf16-fallback (`__CUDA_ARCH__ < 800`) treatment; if not, it can compile-only as long as hipify + the fault-class fixes let it build. Confirm during porting which symbols the chosen validation model actually calls.

## CUDA surface inventory
- Native module: `awq_inference_engine` (single CUDAExtension). 13 `.cu`/`.cpp` TUs in `awq/kernels/setup.py`.
- Inline PTX (will NOT hipify; manual rewrite required):
  - Tensor-core MMA: `mma.sync.aligned.m16n8k16.f32.f16.f16.f32` (gemm_cuda_gen.cu, quantization_new gemm), `mma.sync.aligned.m16n8k32.s32.s8.s8.s32` (w8a8).
  - `ldmatrix.sync.aligned.m8n8.x4[.trans].shared.b16` (all three GEMMs).
  - `cvta.to.shared.u64` shared-pointer conversion (all three GEMMs).
  - `cp.async.cg.shared.global` async-copy (quantization_new gemm, w8a8); `#include <cuda_pipeline_primitives.h>` (NVIDIA-only header) at w8a8/w8a8_gemm_cuda.cu:14 and quantization_new/gemm/gemm_cuda.cu.
  - `lop3.b32` INT4 dequant LUT (quantization/dequantize.cuh, quantization_new/dequantize.cuh).
  - `ld.global.acquire.gpu` / `st.global.release.gpu` split-K semaphore (quantization_new/gemm/semaphore.h).
  - Misc PTX in attention utils (`decoder_masked_multihead_attention_utils.h`, `_template.hpp`).
- Warp intrinsics (hipify-translatable, but warp-size-sensitive): `__shfl_down_sync(0xffffffff,...)` (gemv_cuda.cu, full-warp tree reduce 16..1), `__shfl_xor_sync(FINAL_MASK,...,32)` width-32 (layernorm/reduction.cuh, w8a8/reduction_utils.cuh, quantization_new gemv). No `__ballot`/`__activemask`.
- Half/bf16: heavy `half2`/`__nv_bfloat16` use; `-DENABLE_BF16` set in both cxx and nvcc flags. `__CUDA_ARCH__ >= 800` guards select Ampere bf16 intrinsics vs software fallbacks (cuda_bf16_fallbacks.cuh, w8a8/utils.cuh) -- on HIP `__CUDA_ARCH__` is UNDEFINED so the `< 800` software-fallback branch is taken, which is the correct path (no NVIDIA bf16 intrinsics emitted). Verify no host-only throw hides behind a `__CUDA_ARCH__` guard taken wrong on HIP (use `defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)` where a device/host branch matters -- gsplat changelog 2026-05-30).
- No CUB/Thrust, no real CUTLASS dependency (the "cutlass" hits are commented-out namespace markers and a header copied from FasterTransformer). No textures/surfaces. No cuBLAS/cuFFT/etc. No streams/events beyond `c10/cuda/CUDAGuard.h` (hipified to the HIP guard).
- `--use_fast_math` in nvcc flags -> on HIP clang this is `-ffast-math` with `-ffp-contract=fast`; numeric drift is expected and acceptable for LLM perplexity (no bit-exact gate), but note it if a tolerance check is added.
- Separate optional extension `awq/kernels/csrc/attention/setup.py` (standalone `ft_attention`) hardcodes `arch=compute_70/80/90,code=sm_*` -- NOT built by the main setup.py; ignore unless a flow needs it, in which case its arch flags must be replaced by `PYTORCH_ROCM_ARCH`.

## Risk list
1. (Highest) Inline PTX in all three GEMM kernels does not exist on HIP -- the port's central work is rewriting them (portable SIMT/half2 first; rocWMMA/MFMA later). The whole GEMM (prefill) inference path depends on this.
2. INT4->FP16 `lop3`/`prmt` dequant: must reproduce the exact bit math with portable ops so dequantized weights are byte-identical to CUDA (a wrong LUT silently corrupts every weight). Microtest the dequant against a CPU reference before any GEMM run.
3. Warp size 32 vs 64. `quantization/gemv_cuda.cu` `warp_reduce_sum` reduces offsets 16..1 over a full `0xffffffff` mask with `#define WARP_SIZE 32`, and the launch geometry assumes 32 lanes/warp; on gfx90a (wave64) a full-mask 16..1 reduce only sums 32 of the 64 lanes -> half the partial products dropped. Make the reduction warpSize-generic (offsets from `warpSize/2` down, 64-bit mask) AND fix the host launch's lanes-per-output assumption, OR keep it an explicit width-32 logical-warp reduce only if the data layout truly packs 32 lanes per output (verify the indexing). The layernorm/w8a8 block-reduces use width-32 logical `__shfl_xor_sync(...,32)` (arch-agnostic within a 32-subgroup) BUT then do a `shared[32]` block reduce keyed on `blockDim.x/32` -- on wave64 the warps-per-block count and `lane = threadIdx.x % 32` / `wid = threadIdx.x / 32` partition is wrong; size the shared array to a wave-width upper bound and derive warps-per-block from the real `warpSize`.
4. 64-bit shuffle mask: HIP `__shfl_*_sync` static_assert `sizeof(mask)==8` (changelog 2026-05-30, AutoDock-GPU). The literal `0xffffffff` / `~0` masks fail to COMPILE on HIP. Provide a full-mask compat constant `0xffffffffffffffffULL` under `USE_ROCM` (hipify may or may not rewrite `~0`; verify). This is a compile error before any wave64-correctness question.
5. Divergent-return-then-shuffle: confirm no GEMV/reduction `return`s out-of-range lanes before a `__shfl_*_sync` full-mask collective (HIP `__hip_check_mask` SIGABRT, raft changelog 2026-05-31). If found, use `__activemask()` on HIP.
6. `cuda_pipeline_primitives.h` include: NVIDIA-only; the `cp.async` path must be `#if !defined(__HIP_PLATFORM_AMD__)`-guarded and the HIP path uses a plain blocking shared-load (or `__builtin_nontemporal`/direct copy), folded into the GEMM rewrite.
7. PTX semaphore acquire/release -> HIP `__atomic_*` with explicit memory order; a wrong fence on split-K silently races the partial sums.
8. RDNA wave32 followers: any wave64-specific GEMM rewrite (MFMA) must NOT hardcode 64-lane geometry (changelog 2026-06-02 / 2026-05-30 popsift). The portable SIMT GEMM avoids this entirely, which is the argument for shipping it first; an MFMA path would need a separate wave32 WMMA branch (no f32-input WMMA on RDNA -- 2026-06-09).
9. Dependency environment (Strategy B subsection / 2026-06-10): `pyproject.toml` pins CUDA torch 2.3.0 etc. An unfiltered `pip install -e .` will clobber the ROCm torch. Strip the torch/torchvision/flash-attn/triton pins and install ROCm builds first; build the kernel extension with `--no-build-isolation` against the ROCm torch (`cd awq/kernels && python setup.py install`). `flash-attn` (README step) is optional for the quant kernels themselves -- skip or use the ROCm flash-attn; it is not needed to exercise the GEMM/GEMV.
10. Import success != backend: confirm the run actually calls `awq_inference_engine.gemm_forward_cuda_new`/`gemv_forward_cuda_new` on the GPU (not a silent fallback) -- the qmodule path has no fallback, so an import or a kernel launch failure will be a hard error, which is good for validation honesty.

## File-by-file change list (lead, gfx90a)
- `awq/kernels/csrc/quantization/dequantize.cuh` -- replace `lop3.b32` (4x) and any `prmt`/`__byte_perm` with `__HIP_PLATFORM_AMD__`-guarded portable bit expressions; CUDA keeps PTX.
- `awq/kernels/csrc/quantization_new/dequantize.cuh` -- same dequant treatment (INT4 and any INT3 path).
- `awq/kernels/csrc/quantization/gemm_cuda_gen.cu` -- guard the `cvta`/`ldmatrix`/`mma.sync` block `#if !defined(__HIP_PLATFORM_AMD__)`; add a portable HIP GEMM (dequant + half2 FMA over the same shared tiles) `#else` branch.
- `awq/kernels/csrc/quantization_new/gemm/gemm_cuda.cu` -- same; also guard the `cp.async`/`cuda_pipeline_primitives.h` include and async-copy path.
- `awq/kernels/csrc/quantization_new/gemm/semaphore.h` -- HIP `__atomic_*` acquire/release branch under guard.
- `awq/kernels/csrc/w8a8/w8a8_gemm_cuda.cu` -- guard `cvta`/`ldmatrix`/`mma.sync.s8.s8.s32`/`cp.async`; portable int8 dequant+accumulate HIP branch (or route through int8 MFMA later).
- `awq/kernels/csrc/quantization/gemv_cuda.cu` -- warpSize-generic reduction + 64-bit mask; fix `WARP_SIZE`/launch lanes-per-output for wave64.
- `awq/kernels/csrc/quantization_new/gemv/gemv_cuda.cu` -- 64-bit mask on `__shfl_xor_sync`; verify width-16/full-reduce semantics on wave64.
- `awq/kernels/csrc/layernorm/reduction.cuh`, `awq/kernels/csrc/w8a8/reduction_utils.cuh` -- block-reduce shared array sized to wave-width upper bound; warps-per-block from real warpSize; 64-bit mask on full-warp shuffles (width-32 logical ones are fine).
- `awq/kernels/csrc/w8a8/utils.cuh`, `attention/cuda_bf16_fallbacks.cuh` -- verify the `__CUDA_ARCH__ < 800` software-fallback branches are taken on HIP (likely no edit needed; `__CUDA_ARCH__` undefined -> correct branch). Add `|| defined(__HIP_DEVICE_COMPILE__)` only where a device/host split is actually wrong on HIP.
- `awq/kernels/csrc/attention/*` -- only if the validation model invokes `single_query_attention`; otherwise ensure it merely builds. Decide during porting.
- `awq/kernels/setup.py` -- no source-list change; rely on torch hipify. Drop NVIDIA-only nvcc flags that clang rejects on the ROCm build (`--threads=8`, `--use_fast_math` becomes hipcc's own; the `-U__CUDA_NO_*` defines are harmless/translated) -- minimal, ROCm-gated edits only. Document the ROCm build in PR-prep (README install block parallel to the CUDA one).
- New-file copyright headers per MOAT rule on any substantially rewritten kernel (parallel `Copyright (c) 2026 Advanced Micro Devices, Inc.` line + author `Jeff Daily`), in the porter/prep phase.

## Build commands (gfx90a)
Build against a ROCm PyTorch (do NOT let the CUDA pins overwrite it):
```
# in a ROCm torch env (torch.version.hip set)
cd projects/llm-awq/src
# strip CUDA-default pins, install the rest against ROCm torch:
grep -vEi '^(torch|torchvision|torchaudio|xformers|triton|flash.attn)' <(python - <<'PY'
import tomllib,sys
d=tomllib.load(open("pyproject.toml","rb"))
print("\n".join(d["project"]["dependencies"]))
PY
) | pip install -r /dev/stdin
# build the native engine against the installed ROCm torch:
cd awq/kernels
PYTORCH_ROCM_ARCH=gfx90a python setup.py install   # or: pip install . --no-build-isolation
```
Multi-arch sanity (followers): `PYTORCH_ROCM_ARCH="gfx90a;gfx1100"` then confirm both code objects with `llvm-objdump --offloading <built .so>`. Note: torch hipify + the inline-PTX guards mean the GEMM PTX must be fully #else-branched for the build to even compile on HIP -- a clean compile is the first gate (necessary, not sufficient).

## Test plan
There is NO unit-test suite (`find -iname '*test*'` is empty). Validation is end-to-end real-quantized-model inference (the kernels under test are exercised only through a quantized LLM forward pass).

GPU validation (lead, gfx90a):
1. Microtest the INT4 dequant in isolation: dequantize a known packed weight on GPU vs a CPU reference; assert exact (catches the lop3/prmt rewrite). Author this fresh (no upstream test exists).
2. Real quantized inference (smallest viable model to keep download/memory bounded, e.g. a 4-bit AWQ Llama/OPT-125M-class or a small Llama3-8B AWQ if memory allows on MI300):
   - Decode path (GEMV): generate tokens (batch/seq giving <8 tokens/forward) -> exercises `gemv_forward_cuda_new`.
   - Prefill path (GEMM): a prompt of >=8 tokens -> exercises `gemm_forward_cuda_new`.
   - Use `python -m awq.entry --model_path <model> --w_bit 4 --q_group_size 128 --load_quant <awq-int4-ckpt> --tasks wikitext` (perplexity) as the end-to-end gate; a finite, CUDA-comparable perplexity on a known model is the correctness signal (no bit-exact reference; LLM ppl tolerant of fast-math drift). Confirm the kernels actually ran on GPU (the qmodule path has no CPU fallback, so a launch fault is a hard error).
   - If a real INT4 checkpoint is heavy to obtain, fall back to: quantize a small HF model with `--q_backend real --dump_quant`, then reload and eval -- self-contained.
3. w8a8 path (if a W8A8 model/flow is available): exercise `w8a8_gemm_forward_cuda` + `invoke_quant` + `rms_norm_general`; otherwise scope w8a8 as build-only on the lead and note it.
Non-GPU regression set: the AWQ search (`run_awq`, fake-quant) is pure PyTorch and must not regress; a `--q_backend fake` run is a CPU/torch-only sanity that should behave identically.

## Open questions
- Does the chosen validation model's generation path call `single_query_attention` (the FT masked-MHA `.cu`), or only the GEMM/GEMV linears? Determines whether the attention kernels need full porting or merely a clean compile. Resolve by tracing the model forward during porting.
- W8A8: is there an accessible W8A8 checkpoint/flow to GPU-validate `w8a8_gemm_cuda` (int8 MMA), or should it be lead-build-only with int8-MFMA validation deferred? (rocBLAS/MFMA int8->int32 is supported on CDNA -- arrayfire changelog 2026-05-31 -- so an int8 MFMA rewrite is viable later.)
- Portable-SIMT GEMM vs immediate rocWMMA/MFMA: recommend portable-first to unblock validation + the multi-arch PR, with the MFMA perf rewrite registered as deferred work. Confirm this is the desired correctness-first scope.
