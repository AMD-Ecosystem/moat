# slime port plan

Planner run 2026-11-20 on linux-gfx942 (fleet job, no GPU used). Source examined: fork `AMD-Ecosystem/slime` default branch `main` at `79aa41a` (an unmodified mirror of THUDM/slime main). No `moat-port` branch exists yet.

**Three rulings from Jeff are needed before a porter starts.** They are listed under "Open questions" and at the top of the planner section of notes.md. This plan gives the recommended answers; the porter must not start until they are recorded.

## Project

- Name: slime
- Upstream: https://github.com/THUDM/slime (Apache-2.0)
- Fork: https://github.com/AMD-Ecosystem/slime, `fork_default_branch` `main`
- What it is: an RL post-training framework. Megatron-LM does training and SGLang does rollout, with Ray orchestrating. It is almost entirely Python. Its own GPU code is two small kernels; everything heavy belongs to dependencies.

## Existing AMD support

Verdict: **improvable**. The ROCm path is upstream and authored by AMD, but it is stale on `main`, and the refresh is open as PRs that have not merged.

- On `main`: `docs/en/platform_support/amd_tutorial.md`, `scripts/run-qwen3-4B-amd.sh`, and `docker/Dockerfile.rocm`, which is pinned to ROCm 6.3.4 and SGLang 0.4.9/0.5.0rc0 and still says "Still need to update amd_patch". Also `docker/Dockerfile.rocm_MI350-5` (SGLang v0.5.6, gfx950 only). `docker/amd_patch/latest` is byte-identical to `docker/amd_patch/sglv0.5.0rc0`; neither has been touched since 2026-08-28. The CUDA path meanwhile tracks SGLang v0.5.21 and Megatron core 0.20 (`9b1209b`). The accelerator abstraction branches on `torch.version.hip` (`slime/utils/accelerator/cuda.py:32,39`), and there is `slime/utils/rocm_checkpoint_writer.py`.
- THUDM/slime#2274 (merged 2026-08-15) made the INT4 QAT extension build under ROCm.
- THUDM/slime#2506, "[ROCm] Support ROCm 7.15 on MI300/MI325" by jun-amd (AMD), is OPEN and BLOCKED on review, last updated 2026-11-12, +250/-635 over 4 files. It rebuilds `Dockerfile.rocm` on `rocm/primus:v26.5`, tracks the CUDA image's SGLang v0.5.21 with the CUDA `docker/patch/v0.5.21` stack, and pins Megatron `core_v0.19.2`, gfx942 default. It rewrites both tutorials and the Qwen3-4B AMD script. **Authoritative**: AMD-authored, current, and the same stack the CUDA image uses.
- THUDM/slime#2369, "[ROCm] Update and unify ROCm support for gfx942 and gfx950" by LZ-QWQ (also the author of the merged #2274), is OPEN and DIRTY (conflicts), last updated 2026-09-20, +6135/-2028. It uses ROCm 7.2 and SGLang v0.5.15.post1, consolidates into one arch-selectable Dockerfile, and adds ROCm stack patches (aiter rmsnorm, ROCr IPC/VMM fixes). It is partly superseded by #2506 on gfx942.
- Dependency dispositions already on record: sgl-project/sglang `already-supported`; Megatron-LM, TransformerEngine, DeepEP and DeepGEMM `ported-elsewhere` (ROCm/Megatron-LM, ROCm/TransformerEngine, ROCm/DeepEP, AITER for DeepGEMM-class GEMMs); flash-attention `already-supported`. None is a MOAT `depends_on` entry. They are consumed as environment.

Decision: **do not re-port the ROCm path.** Validate the upstream ROCm path at the fork head on top of #2506's stack, fix what the validation finds plus the two kernel-level gaps below, and keep the delta small and disjoint from #2506 and #2369. A MOAT upstream PR that touches the Dockerfile, the AMD tutorials or the AMD script would compete with an open AMD PR (the gsplat lesson). See Open question 3 for the delivery vehicle.

Port versus rewrite: neither kernel is a CUTLASS/CuTe or wgmma kernel, so a mechanical HIP port is the right shape. The DeepGEMM calls in the GLM-5 alignment path have no HIP translation target in DeepGEMM itself; on AMD they would need a reimplementation against AITER or hipBLASLt (Open question 1).

## Build classification

**torch-extension** (Strategy B), plus one SGLang runtime-JIT kernel.

- `slime/backends/megatron_utils/kernels/int4_qat/setup.py`: `from torch.utils.cpp_extension import BuildExtension, CUDAExtension`, a single source `fake_int4_quant_cuda.cu`. Under a ROCm torch the build hipifies and calls hipcc (#2274 made this work).
- `slime/backends/sglang_utils/jit_kernels/csrc/gemm/glm5_router_gemm.cuh`: not built by slime. SGLang's `load_jit` compiles it at run time when `SGLANG_JIT_KERNEL_EXTRA_PATH` points at `slime/backends/sglang_utils/jit_kernels` (`slime/ray/utils.py` `RAY_DEFAULT_ENV_VARS`, `slime/backends/megatron_utils/alignment/env.py`). The SGLang-side wrapper `sglang/kernels/jit/glm5_router_gemm.py` and its caller `router_gemm_batch_invariant` come from slime's own `docker/patch/{v0.5.15.post1,v0.5.21,latest}/sglang-deterministic.patch`. In SGLang v0.5.21, `load_jit` supports ROCm: `kernels/jit/utils/compile/toolchain.py` selects `$ROCM_HOME/bin/hipcc` with `-D__HIP_PLATFORM_AMD__=1 -fno-gpu-rdc`. It does **not** hipify. SGLang's own ROCm JIT kernels are hand-written HIP sources (`*_hip.cuh`, `dsa_gfx950/*`). So this file must compile as HIP as written.
- Everything else is Python, so `set-ext-type slime torch-extension`.

## Port strategy

Strategy B for the extension (torch's build-time hipify; fix only what it cannot). For the JIT kernel, use an in-source `__HIP_PLATFORM_AMD__` branch, because nothing hipifies it. The CUDA spelling stays the default, so the CUDA path is byte-identical in behaviour.

## CUDA surface inventory

`surface.py` floor: 2 files, 4 kernels, 4 launches, 3 warp-intrinsic sites, no CUDA libraries.

| file | content | HIP status | action |
|---|---|---|---|
| `kernels/int4_qat/fake_int4_quant_cuda.cu` (12.8 KB) | 3 kernels (`int4_quant_1x32`, `_32x1`, `_common`), XOR-shuffle reductions over 32-lane groups, `WARP_SIZE = 32` constants, 256-thread blocks | already HIP-clean upstream (#2274): `WARP_XOR` uses maskless `__shfl_xor(v, m, 32)` under HIP, so the 32-lane logical groups are correct on wave64 and wave32 | none to the kernel; validate on both widths |
| `kernels/int4_qat/setup.py` | `os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx950")` under ROCm | **defect**: a plain `python setup.py install` on an MI300 (gfx942) or RDNA host builds gfx950-only code objects, and the import fails with "no kernel image" on that GPU. It also overrides torch's own default, which is the arch list torch was built for | drop the forced default (let torch pick, or derive it from `torch.cuda.get_device_properties(i).gcnArchName` as the CUDA branch derives `arch_list`) |
| `jit_kernels/csrc/gemm/glm5_router_gemm.cuh` (5.9 KB) | 1 templated kernel (16 instantiations), `__nv_bfloat16`, `<cuda_bf16.h>`, `uint4` 16-byte loads, `__shfl_xor_sync(0xffffffff, ...)` over `kWarpSize = 32`, `__launch_bounds__(128, 1)`, `cudaStream_t` | **does not compile under hipcc**: no `cuda_bf16.h`/`cuda_runtime.h` on the HIP include path, and a 32-bit mask on `__shfl_xor_sync` is a HIP compile error | HIP branch: `<hip/hip_bf16.h>` + `<hip/hip_runtime.h>`, alias `__nv_bfloat16` to `__hip_bfloat16`, `cudaStream_t` to `hipStream_t`; reduction via maskless `__shfl_xor(sum, offset, 32)`. With width 32 the logical warp stays 32 lanes on both wavefront widths, so `kNumWarps`, `warp_sums` and lane-0 writes are unchanged. The three `device_type == kDLCUDA` checks are likely to reject ROCm tensors, because a ROCm torch exports DLPack device `kDLROCM`. Verify what `TensorView::device().device_type` reports and, under the HIP branch, accept `kDLROCM` |

Python-level CUDA assumptions (not in the census, added to surface.json):
- `slime/backends/megatron_utils/alignment/` (4.9k lines): imports `deep_gemm` (bf16 NN/NT/TN and grouped GEMMs), relies on DeepEP low-latency dispatch, and sets `NCCL_*`/`CUBLAS_WORKSPACE_CONFIG`/`NVTE_*` determinism env. It only runs when GLM-5 train/rollout alignment is switched on (`--megatron-deepgemm-*-layers`, `MEGATRON_USE_SGLANG_*`). It is not reachable on ROCm: DeepGEMM has no ROCm build.
- The accelerator abstraction and `rocm_checkpoint_writer.py` are already ROCm-aware; the validation exercises them.

## Risk list

- **Environment, the largest risk.** A slime GPU test needs SGLang v0.5.21 with slime's patch stack, sgl-kernel built through `setup_rocm.py`, Megatron core 0.19/0.20, ROCm TransformerEngine, aiter, flash-attn and torch_memory_saver for HIP. #2506 assembles exactly this on `rocm/primus:v26.5`. A fleet worker runs inside a container and probably cannot `docker build`. Either the validator assembles the same stack in its own job venv (authorized: dependency builds go in the job directory), or Jeff provides an `amdagi/slime:rocm7.15_torch2.12_py3.12`-equivalent environment. Budget at least one full round for this alone.
- **Shared multi-GPU host.** The e2e tests want 4 to 8 GPUs (`pr-test.yml.j2`), and other jobs share the fleet hosts. RCCL collectives need the whole host at once (`cuda-to-rocm` libraries.md). E2E runs need a dispatch that reserves the host.
- **Wavefront.** Both kernels use 32-lane logical groups expressed with width-32 shuffles. That is correct on wave64 and wave32, but proven only by running it on both.
- **DLPack device type** for the JIT kernel's `RuntimeCheck(... == kDLCUDA)`; see the inventory.
- **bf16 numerics.** The router GEMM accumulates fp32 with `-ffp-contract=fast` under clang, against nvcc's default contraction. The kernel test compares with a tolerance against a torch fp32 matmul, not bit-exactly. The batch-invariance property (the reason the kernel exists) holds per kernel, because the reduction order is fixed by construction. Test it by running M=1..16 and checking each row equals the M=1 result for that row.
- **Megatron/TE patch drift.** `docker/amd_patch/latest` is stale; #2506 instead applies the CUDA `docker/patch/v0.5.21` stack on ROCm. If that stack does not apply or run on ROCm, the fix belongs in #2506's area. Record it and feed it back (Open question 3); do not fork the patch stack.
- No textures, no managed memory, no cuBLAS/cuDNN calls, no 64 KB+ shared memory (the router kernel uses 16x4 floats).

## File-by-file change list

Subject to the rulings below. This is the recommended scope (Q1 option B).

1. `slime/backends/megatron_utils/kernels/int4_qat/setup.py`: remove the forced `PYTORCH_ROCM_ARCH=gfx950` default. When unset and a GPU is visible, derive the arch from `gcnArchName` (strip the `:feature` suffix), mirroring the CUDA branch's `get_device_capability` loop. Otherwise leave torch's default. CUDA branch untouched.
2. `slime/backends/sglang_utils/jit_kernels/csrc/gemm/glm5_router_gemm.cuh`: the `__HIP_PLATFORM_AMD__` branch described in the inventory (only if Q1 keeps the router kernel in scope).
3. New kernel tests, shipped with the port, that skip cleanly when no GPU is present:
   - `tests/test_int4_qat_kernel.py`: build or import `fake_int4_quant_cuda`. Compare `(1,32)`, `(32,1)` and a common block size, sym and asym, bf16/fp16/fp32, against a pure-torch reference of the same fake-quant (the reference can follow `quantizer_compressed_tensors.py`'s semantics). Include N not divisible by 32 and M not divisible by 32.
   - `tests/test_glm5_router_gemm_kernel.py` (if in scope): needs SGLang with `load_jit`. Compare with `x.float() @ w.float().t()` for M=1..16, and check batch invariance.
4. No changes to `docker/`, `docs/en|zh/platform_support/amd_tutorial.md` or `scripts/run-qwen3-4B-amd.sh`; #2506 owns them. If validation finds a defect there, record the exact fix in notes.md for Jeff to route to jun-amd, not as a fork commit (Q3).
5. Commit titles `[ROCm] ...`; jargon check before every push.

## Build commands

Environment: the #2506 stack (ROCm 7.15-class Primus base: torch 2.12 ROCm, SGLang v0.5.21 `e00930c` with `docker/patch/v0.5.21/*.patch` applied, sgl-kernel `python setup_rocm.py install` with `AMDGPU_TARGET=<gfx>`, Megatron-LM `core_v0.19.2` `4b4acac9`, ROCm TransformerEngine, torch_memory_saver HIP build). Read the exact steps from #2506's `docker/Dockerfile.rocm` at the time of the round. On a fleet worker, build into `$FLEET_VENV` with capped `MAX_JOBS`; never into the image's shared Python.

```
# slime itself
cd <fork checkout>
$FLEET_VENV/bin/python -m pip install -e . --no-deps
# INT4 QAT extension (after fix 1, no PYTORCH_ROCM_ARCH needed; set it to build for another arch)
cd slime/backends/megatron_utils/kernels/int4_qat
utils/timeit.sh slime compile -- $FLEET_VENV/bin/python setup.py build_ext --inplace
```

The router kernel has no build step: SGLang JIT-compiles it on first call with `SGLANG_JIT_KERNEL_EXTRA_PATH=<fork>/slime/backends/sglang_utils/jit_kernels`.

CUDA no-regression: build the INT4 extension with nvcc against a CUDA torch on the mi355 fleet host (`TORCH_CUDA_ARCH_LIST="9.0"`). Compile the router `.cuh` through SGLang's CUDA `load_jit`, or with a direct `nvcc -std=c++20 -c` against SGLang's `kernels/jit/include` and tvm-ffi headers. Record the command and the result, or `cuda-not-validated: <exact error>` when the host torch is ROCm-only.

## Test plan

GPU tests, per platform:

- wave64 (gfx942 / gfx950), full set:
  1. `tests/test_int4_qat_kernel.py` (new): must pass.
  2. `tests/test_glm5_router_gemm_kernel.py` (new, if in scope): must pass.
  3. Single and dual GPU upstream tests from `pr-test.yml.j2`: `test_ppo_logprob_entropy_gpu.py` (2 GPUs), `test_sglang_top_p_copy.py` (1), `test_glm5_next_gpu.py` (1).
  4. The AMD recipe as the e2e gate: `scripts/run-qwen3-4B-amd.sh` (as #2506 has it) for a bounded number of rollouts (for example `--num-rollout 3`) on 8 GPUs. Record reward, loss and log-prob mismatch for each step.
  5. Small e2e tests on 4 GPUs, if the host can be reserved: `test_qwen2.5_0.5B_fully_async_short.py` and `test_qwen2.5_0.5B_pipeline_rl.py --transport nccl`.
- wave32 (RDNA): items 1 and 2 only (kernel-level). SGLang's ROCm sgl-kernel and the Megatron/TE ROCm stack target CDNA (gfx942/gfx950), so the RL stack itself has no RDNA environment. See Q2.
- windows: none possible (Ray + Megatron + SGLang are Linux-only). See Q2.

Non-GPU regression set (every platform, must not regress against the same commit on the mirror branch): `pytest tests -m "not system"` restricted to the `num_gpus: 0` set in `pr-test.yml.j2`. At minimum: `tests/utils/test_sglang_arguments.py`, `tests/utils/test_sglang_config.py`, `tests/test_sglang_weight_update_compat.py`, `tests/test_glm5_indexer_q_norm.py`, plus the pure-logic unit files (`test_policy_loss.py`, `test_cispo_loss.py`, `test_discounted_returns.py`, `test_rm_*.py`, `test_rollout_*.py`, `test_accelerator.py`). Record pass/fail counts before and after.

## Open questions (need Jeff before porting)

1. **Scope of the GLM-5 alignment path (a partial-port cut).** The router kernel is reachable only inside slime's GLM-5 deterministic train/rollout alignment. That path also needs DeepGEMM (no ROCm build), DeepEP low-latency and slime's patched batch-invariant SGLang, and its test (`test_glm52_6layer_deterministic_e2e.py`) needs 8 GPUs and GLM-5 weights.
   - A: port the router kernel to HIP with a kernel-level test, and defer the rest of the alignment path. This is a small delta, but on ROCm the kernel stays unreachable from slime until DeepGEMM is replaced.
   - B (recommended): A, and register the alignment path (DeepGEMM to AITER/hipBLASLt, DeepEP-LL to ROCm/DeepEP) as a deferral for the AMD slime team.
   - C: port the whole alignment path to ROCm. This is a multi-week reimplementation of batch-invariant GEMMs on AITER/hipBLASLt, needs multi-GPU GLM-5 validation, and overlaps the AMD team's area.
   - D: scope the router kernel out too, and validate only the existing ROCm path.
2. **Gates.** `wave32` is not waivable in `config/arches.toml`, and the slime RL stack has no RDNA environment. Recommended: satisfy wave32 with the kernel-level tests only, and state that limit in notes and in any PR. The alternative is to treat wave32 as unsatisfiable for an e2e claim, which would need a config change only a person can make. `windows` is waivable; recommend a waiver (Ray/Megatron/SGLang have no Windows path), which needs maintainer approval.
3. **Delivery vehicle.** AMD's own #2506 (jun-amd, current) and #2369 are open on the same surface. Options:
   - a: one small upstream PR from the fork with only fixes 1 to 3 (setup.py arch default, router kernel HIP branch, kernel tests). It is disjoint from #2506's files.
   - b: no MOAT upstream PR. The fork's `moat-port` is validation evidence, and Jeff hands the findings to jun-amd.
   - c: a, and also post validation feedback on #2506. That is a GitHub-visible write, so it needs Jeff's approval of the exact text.

   Recommended: a, timed after #2506 merges so that the validation is on the stack upstream ships.
