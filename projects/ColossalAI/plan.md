# ColossalAI port plan

## Rulings (Jeff Daily, 2026-11-23)

The planner's questions Q1 to Q5 (`notes.md`, Planning 2026-11-20) are ruled; the full text is in `notes.md` under "Rulings on the planner's questions". Q1: the `windows` gate is waived. Q2: fa2 is enabled on ROCm when the ROCm build of `flash_attn` is installed, behind a ROCm capability check, and fa3 is `scoped_out` (Hopper-only). Q3: option b, both wavefront widths run the full 1-, 2- and 4-GPU batches (see the test plan below). Q4: no internal AMD effort known. Q5: no ROCm Docker image; the docs give the install steps.

## Project

- Name: ColossalAI
- Upstream: https://github.com/hpcaitech/ColossalAI, default branch `main`
- Fork: https://github.com/AMD-Ecosystem/ColossalAI, port branch `moat-port`
- Planned at upstream `14d1c15147564603d4564198f38af1ea3c42be92` (2026-11-18, version 0.5.0). Intake screened `f4ef8b4`; upstream moved by one merge (#6477, MoE EP layer, Python only) and the CUDA surface is unchanged.
- Licence Apache-2.0 (intake).

## Existing AMD support

None. `colossalai/accelerator` has cuda, npu and cpu only. The one ROCm reference in the tree is a refusal: `colossalai/kernel/kernel_loader.py:138` returns "requires NVIDIA CUDA in this adapter; ROCm is not supported" for the FlashAttention adapters, with the comment "ROCm needs its own capability policy". Upstream issues #3401, #3451 and #5669 ask for AMD support; #5669 (2024-04) has a contributor saying "some collaboration with AMD is underway", and nothing public followed (intake searched PRs, branches, ROCm and AMD-Ecosystem orgs; the `amd` and `AMD-AGI` orgs were unreachable through SSO). Judgment: no authoritative AMD port exists; the remark is the only hint of an internal effort (Q4).

Delivery vehicle: an upstream PR. Upstream is active, takes contributions, and its own code anticipates a ROCm policy.

Port versus rewrite: no kernel is a CUTLASS/CuTe or Hopper-specific performance kernel. They are apex/Megatron/vLLM-derived elementwise, reduction and decoding kernels. A mechanical HIP port through torch's hipify is the right first step; no AMD-native rewrite is planned.

## Build classification

torch extension (`set-ext-type torch-extension` recorded). Evidence:

- `setup.py:96-117`: with `BUILD_EXT=1`, every `ALL_EXTENSIONS` entry that is available builds through `ext.build_aot()` into `torch.utils.cpp_extension.CUDAExtension`, with `BuildExtension` as `build_ext`.
- `extensions/cuda_extension.py`: `build_aot` returns `CUDAExtension(...)`, and `build_jit` calls `torch.utils.cpp_extension.load(...)` (the default when `BUILD_EXT` is unset: kernels compile on first use into `~/.cache/colossalai/torch_extensions`).
- No CMake; no other native build.

## Port strategy

Strategy B: rely on torch's build-time hipify (it runs on ROCm torch for both `CUDAExtension` and `load`) and fix only what it cannot handle:

1. the extension loader, which refuses ROCm torch before any compile;
2. the nvcc-only flags;
3. inline PTX and the NVIDIA fp8 conversion intrinsics;
4. wavefront-width assumptions and 32-bit lane masks;
5. `__CUDA_ARCH__` gates that drop the bf16 paths under HIP;
6. the Python layers that assume NVIDIA (attention backend policy, fp8 dtypes, Triton launch widths, the accelerator version string, the install checker).

Every change keeps the CUDA build byte-for-byte equivalent: ROCm branches key on `torch.version.hip` in Python and on `USE_ROCM` (which torch defines for HIP extension builds) in C++. Shared device code changes must be correct on both wave32 and wave64.

## CUDA surface inventory

`surface.json` lists 24 CUDA files (regex floor) and 26 components. The detail:

| extension (module) | sources | used by | hazards |
|---|---|---|---|
| `cpu_adam_x86` | `kernel/x86/cpu_adam.cpp` | `CPUAdam`, `HybridAdam` (Gemini/ZeRO CPU offload) | a `CUDAExtension` though CPU-only; header includes `cublas_v2.h`, `cuda.h`, `cuda_fp16.h`, `cuda_runtime_api.h`; `-lcudart -lcublas` in cxx flags; uses `__half` on the host |
| `fused_optim_cuda` | `multi_tensor_{sgd,scale,adam,l2norm,lamb}_kernel.cu`, `multi_tensor_apply.cuh` | `FusedAdam`, `FusedLAMB`, `FusedSGD`, `HybridAdam` | `multi_tensor_l2norm_kernel.cu:20-100` block reduce: `tid < 32`, `__shfl_down_sync(0xffffffff, ...)` |
| `layernorm_cuda` | `layer_norm_kernel.cu` | `colossalai/nn/layer/layernorm.py` (MixedFusedLayerNorm) | launches `threads(32,4)` (`:588,626,639,651`) and reduces with `srcLaneB = (threadIdx.x + (1<<l)) & 31` through `WARP_SHFL` at its default width (`:76,166`): on wave64 two 32-thread rows share one wavefront and the upper row reads the lower row's lanes. `-maxrregcount=50` nvcc-only |
| `moe_cuda` | `moe_kernel.cu` | `colossalai/moe/_operation.py` (MoeDispatch/MoeCombine/cumsum) | `cub::BlockLoad`/`BlockStore` (hipify to hipCUB); 32/64/128-thread blocks, i.e. blocks narrower than a wave64 wavefront; `__syncthreads` in the cumsum kernel |
| `inference_ops_cuda` | `decode_kv_cache_memcpy`, `context_kv_cache_memcpy`, `fused_rotary_emb_and_cache`, `activation`, `rms_layernorm`, `get_cos_and_sin`, `flash_decoding_attention`, `convert_fp8` | `colossalai/inference` (nopadding llama/baichuan, attention backends) | inline PTX in `funcs/cast_functor.h:285,311,357,376-379,460` (`mov.b32` pack/unpack, `cvt.f32.f16`); `<cuda_fp8.h>` `__nv_cvt_*` E5M2 conversions (`:174-441`); bf16 functors under `#if __CUDA_ARCH__ >= 800` (`cast_functor.h:121-170`, `binary_functor.h`, `:432`); `attention/attention_utils.h:36-41` `WARP_SIZE 32` with `__shfl_xor_sync(uint32_t(-1), ...)` and `__shfl_sync(uint32_t(-1), var, src_lane)` at default width; `flash_decoding_attention_kernel.cu` `WARP_SIZE 32`, dynamic shared memory sized from `max_num_blocks_per_seq` plus logits, `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` |
| `scaled_masked_softmax_cuda` | `scaled_masked_softmax_kernel.cu` | `colossalai/nn/layer/scaled_softmax.py` | host launchers (`:262,290,390`) use `constexpr int WARP_SIZE = ... C10_WARP_SIZE`; on ROCm torch `C10_WARP_SIZE` is `at::cuda::warp_size()` in host code, a runtime call, so this does not compile. Device side (`:41,164`) is fine. `-DTHRUST_IGNORE_CUB_VERSION_CHECK`, half-operator flags |
| `scaled_upper_triangle_masked_softmax_cuda` | `scaled_upper_triang_masked_softmax_kernel.cu` | same | same pattern (`:274,399`) |

Shared headers: `funcs/reduce_function.h` (`kWarpReduceMask = 0xffffffff`; logical width 32 already explicit through the `width = 32` template default and `lane_id & 0x1f`, `shm[LANES][32]`), `funcs/{binary,unary,ternary,cast}_functor.h`, `common/{data_type,vec_type_traits,micros}.h`, `kernel/cuda/utils/{vec_copy,micros,gpu_launch_config,nvgpu_dev_info}.h` (`cudaGetDeviceProperties`; hipify handles it, and the only call site is commented out in `activation_kernel.cu:58`).

No textures, no graphics interop, no NVRTC or driver API, no CUTLASS, no NCCL calls from C++ (collectives go through `torch.distributed`, which is RCCL on ROCm), no cuRAND. cuBLAS appears only as an include in the CPU Adam header.

Non-C++ surfaces the census cannot see:

- `extensions/cuda_extension.py`, `extensions/utils.py`: `assert_compatible` raises when `torch.version.cuda is None` (always on ROCm); `assert_build_compatible` requires `CUDA_HOME` and an `nvcc` whose major version matches torch's CUDA; `set_cuda_arch_list` derives `TORCH_CUDA_ARCH_LIST` from `nvcc -V`; `append_nvcc_threads` runs `nvcc -V`; `get_cuda_home_include` adds `$CUDA_HOME/include`.
- torch's `BuildExtension._hipify_compile_flags` rewrites the first `CUDA` in every `-D` flag under `nvcc` to `HIP`, so the AOT build receives `-DCOLOSSAL_WITH_HIP`, and every header gated on `#if defined(COLOSSAL_WITH_CUDA)` compiles to nothing. The JIT path (`load`) does not take that route. The two paths must give the same result.
- `colossalai/kernel/kernel_loader.py:136-158`: the fa2/fa3 adapters refuse ROCm; capability checks use NVIDIA SM numbers. The sdpa backend has no such check and is what ROCm gets today.
- `colossalai/quantization/fp8.py`: `cuda_arch = int("".join(get_device_capability()))` gates `torch.compile` at `< 89`; ROCm reports gfx942 as (9,4) so it reads 94, and gfx1100 as 110. The dtypes are hard-coded `float8_e4m3fn`/`float8_e5m2`; `torch._scaled_mm` on gfx942 needs `float8_e4m3fnuz`/`float8_e5m2fnuz` (OCP types need gfx950 or gfx12).
- `colossalai/kernel/triton/*`: `num_warps` up to 32 (`rms_layernorm.py:107`, `no_pad_rotary_embedding.py:16`); at 64 lanes that is 2048 threads, over the 1024-thread block limit, on CDNA.
- `colossalai/accelerator/cuda_accelerator.py:28`: `get_version` returns `torch.version.cuda` (None on ROCm).
- `colossalai/cli/check/check_installation.py`: reports through `CUDA_HOME`/`nvcc` and `torch.version.cuda`.
- `setup.py:19-20`: raises on `win32` (Q1).

No CUDA in `applications/` or `examples/` (no `.cu`, no `cpp_extension` use); they are Python on top of the library and inherit the port.

## Risk list

1. **AOT define rewrite** (above). Silent failure: the extension builds and imports, but no kernel code is in it. The porter must check that the built `.so` exports the kernel symbols (or that a kernel test runs) for both `BUILD_EXT=1` and the JIT path. The fix stays out of the CUDA path: on ROCm, guard the headers on `COLOSSAL_WITH_CUDA || COLOSSAL_WITH_HIP` (the shared spelling torch produces), or pass a define with no `CUDA` substring. The porter chooses one and records it.
2. **Wavefront width.** Keep a logical 32-lane warp wherever the code already uses one (attention utils, reduce_function, l2norm, layernorm, flash decoding). Every shuffle then needs an explicit width of 32 and a full-width mask type (a 64-bit mask on HIP, built so it is valid on wave32 and wave64), because HIP's `*_sync` builtins (`HIP_ENABLE_WARP_SYNC_BUILTINS`, set by torch) reject or mis-handle a 32-bit `uint32_t(-1)` mask. An xor shuffle with an offset below 32 never crosses the 32-lane half, but `__shfl_sync(mask, v, src_lane)` at default width does, so `SHFL_SYNC` and the layernorm `WARP_SHFL` calls must pass width 32. Softmax: replace the host `constexpr` `C10_WARP_SIZE` either with a logical 32 on both sides (recommended: one geometry on every arch, CUDA unchanged since `C10_WARP_SIZE` is 32 there) or with `at::cuda::warp_size()` on the host matched to `C10_WARP_SIZE` on the device; host and device must agree, or rows are skipped.
3. **Inline PTX** in `cast_functor.h`: replace it under `USE_ROCM` with `__half2float`/`__half_as_ushort` and shift/or packing (or a union). Keep the PTX on CUDA.
4. **fp8 conversions**: map `__nv_cvt_*` E5M2 to `hip_fp8.h` (`__hip_cvt_float_to_fp8`, `__hip_cvt_halfraw_to_fp8`, `__hip_cvt_bfloat16raw_to_fp8`, the `x2` forms, `__HIP_SATFINITE`, `__HIP_E5M2`, the OCP encoding). Hipify does not translate these names. Check each exists in the target ROCm's `hip_fp8.h` before relying on it. On gfx942 the OCP conversion is done in software; the kv cache is an internal byte format, so OCP E5M2 on every arch keeps it the same bits as CUDA. `test_convert_fp8.py` and `test_kv_cache_memcpy.py` must pass on both widths.
5. **bf16 under `__CUDA_ARCH__ >= 800`**: under HIP `__CUDA_ARCH__` is undefined, so these specializations vanish and the bf16 instantiations in the dispatch macros either fail to compile or pick a generic path. Extend each guard with `|| defined(USE_ROCM)`; do not re-key it. HIP's bf16 header lacks some CUDA spellings (`__floats2bfloat162_rn`, `make_bfloat162`, `__hmax2` on `half2`); see the build-headers reference.
6. **LDS budget** in flash decoding: dynamic shared memory grows with `max_num_blocks_per_seq` and the logits buffer. AMD gives 64 KB per workgroup (160 KB on gfx950), against up to 227 KB opt-in on Hopper. The test matrix (batch 4e) must pass on every validated arch, and a size the device cannot hold must fail with a clear error, not a launch failure.
7. **Blocks narrower than a wavefront** in `moe_kernel.cu` (32-thread blocks on wave64). `BlockLoad`/`BlockStore` with VECTORIZE need no intra-block exchange, so this should be safe; check the cumsum kernel's reductions for `blockDim.x / warpSize` arithmetic.
8. **nvcc-only flags** under hipcc: `--use_fast_math`, `-lineinfo`, `-maxrregcount=50`, `--expt-relaxed-constexpr`, `--expt-extended-lambda`, `--threads 4`, `-U__CUDA_NO_HALF*` (rewritten to `-U__HIP_NO_HALF*` by torch, which is wanted). On ROCm give each extension a HIP flag list: drop what clang rejects, and do not replace `--use_fast_math` with `-ffast-math` (it implies finite-math-only; see the numerics reference). `-lcudart -lcublas` in `cpu_adam_x86` cxx flags must not reach the ROCm link line.
9. **fp8 Python layer on gfx942**: needs the fnuz dtypes for `_scaled_mm` and for anything compared against `torch.finfo(float8_e4m3fn)`; the casts' max values differ (240 against 448 for e4m3). Choose the dtype in one helper (the way pytorch tests select `e4m3_type`), keyed on a capability probe, not on the arch name. The `cuda_arch < 89` compile gate needs a ROCm meaning (fp8-capable: gfx942, gfx950, gfx12).
10. **Triton `num_warps`** above 16 on wave64: cap it from the device warp size (`torch.cuda.get_device_properties(dev).warp_size`). On CUDA the cap is a no-op.
11. **Dependencies on ROCm**: `requirements.txt` pins `torch>=2.5,<2.14` and requires `bitsandbytes>=0.39` (its main wheels carry ROCm support from 0.45 or so; it is imported lazily) and `galore_torch` (pure PyTorch). Tests need `flash_attn` (ROCm build: CK backend for gfx942/gfx950, Triton backend on RDNA) and `triton` (ROCm triton). The validator builds in a job venv with a ROCm torch wheel inside the pin (2.13.x), not the host's dev torch 2.16, which is outside it. No requirements change is planned.
12. **Multi-GPU tests** use `torch.distributed` with backend `nccl`, which ROCm torch maps to RCCL; 4-GPU and 8-GPU batches need the whole host (see the libraries reference: RCCL collectives need the whole multi-GPU host). Q3.
13. **Numerics**: tolerance changes for AMD numerics go behind `torch.version.hip`, never loosening CUDA.

## File-by-file change list

Python build and loader:

- `extensions/cuda_extension.py`: `assert_compatible` accepts `torch.version.hip`; `assert_build_compatible` on ROCm checks `ROCM_HOME` instead of `CUDA_HOME`/nvcc; `get_cuda_home_include` returns the ROCm include dir (or nothing; hipcc adds it); `build_jit`/`build_aot` skip `set_cuda_arch_list` on ROCm (torch reads `PYTORCH_ROCM_ARCH` or detects); the `COLOSSAL_WITH_CUDA` define handled per risk 1.
- `extensions/utils.py`: `set_cuda_arch_list` and `append_nvcc_threads` return early on ROCm.
- `extensions/pybind/*/*_cuda.py` and `cpu_adam_x86.py`: a ROCm flag list per extension (risk 8). Factor this into one helper in `cuda_extension.py` rather than seven copies.
- `colossalai/kernel/kernel_loader.py`: ROCm policy for fa2 per the Q2 ruling: allow fa2 when `flash_attn` is installed, with the same dtype, mask and head-dimension rules as on CUDA, and in place of the SM test a check that the GPU is one the ROCm build of flash-attn supports. fa3 keeps refusing ROCm.
- `colossalai/quantization/fp8.py`: fp8 dtype helper and compile gate (risk 9). Any test that hard-codes `float8_e4m3fn` uses the same helper.
- `colossalai/kernel/triton/rms_layernorm.py`, `no_pad_rotary_embedding.py` (and any other launch above 16 warps): cap `num_warps` (risk 10).
- `colossalai/accelerator/cuda_accelerator.py`: `get_version` returns `torch.version.hip` when set.
- `colossalai/cli/check/check_installation.py`: a ROCm branch that reports the ROCm/HIP versions instead of trying `nvcc`.

C++/HIP (each change guarded or width-agnostic, CUDA unchanged):

- `extensions/csrc/funcs/cast_functor.h`: PTX replacements, fp8 conversion mapping, bf16 guard extension.
- `extensions/csrc/funcs/binary_functor.h`, `ternary_functor.h`, `unary_functor.h`: bf16 guard extension and any missing HIP bf16 spellings.
- `extensions/csrc/funcs/reduce_function.h`: full-width mask type on HIP.
- `extensions/csrc/kernel/cuda/attention/attention_utils.h`: mask type, `SHFL_SYNC` width 32.
- `extensions/csrc/kernel/cuda/flash_decoding_attention_kernel.cu`: check that the logical-warp constants still hold; LDS limit check (risk 6).
- `extensions/csrc/kernel/cuda/multi_tensor_l2norm_kernel.cu`: mask type, width 32.
- `extensions/csrc/kernel/cuda/layer_norm_kernel.cu`: `WARP_SHFL(..., srcLaneB, 32)` at the two reduction loops and the two broadcasts (`:116-117` and the half-precision twin).
- `extensions/csrc/kernel/cuda/scaled_masked_softmax_kernel.cu`, `scaled_upper_triang_masked_softmax_kernel.cu`: the host warp-size fix (risk 2).
- `extensions/csrc/kernel/cuda/moe_kernel.cu`: review only, expected unchanged.
- `extensions/csrc/kernel/x86/cpu_adam.h`: CUDA includes need either hipify (automatic) or replacing with what the file actually uses (`cuda_fp16.h` for `__half`); drop `cublas_v2.h` on ROCm if unused.

Tests (added, upstream style, pytest, skipped without a GPU):

- `tests/test_kernels/` or alongside the existing kernel tests: `layernorm_cuda` forward/backward against `torch.nn.functional.layer_norm`; `scaled_masked_softmax` and `scaled_upper_triang_masked_softmax` forward/backward against a torch reference over sequence lengths that cover every `WARP_ITERATIONS` case (16 to 2048 or the kernel's maximum, both widths); `moe_cuda` dispatch/combine forward/backward against an einsum reference. These kernels have no upstream tests, and wave64 faults in them would otherwise go unseen.

Docs:

- `README.md` install section and `docs/` (where the CUDA build is documented): a ROCm subsection covering a ROCm torch wheel, `BUILD_EXT=1`, `PYTORCH_ROCM_ARCH`, and where to get flash-attn and bitsandbytes for ROCm. No device-selection env vars in example commands.

Not changed: `docker/` (CUDA images; no ROCm image per the Q5 ruling), `applications/`, `examples/`, `.github/` (fork Actions stay disabled).

## Build commands

In the job venv (`$FLEET_VENV`), never the host's shared Python:

```
# ROCm torch inside the upstream pin (2.5 <= torch < 2.14), matching the host ROCm major
$FLEET_VENV/bin/python -m pip install "torch==2.13.*" torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm<ver>
$FLEET_VENV/bin/python -m pip install -r requirements/requirements.txt -r requirements/requirements-test.txt
#   flash_attn from its ROCm build (pip wheel for ROCm or a source build of ROCm/flash-attention); triton comes with the ROCm torch wheel
cd projects/ColossalAI/src
BUILD_EXT=1 PYTORCH_ROCM_ARCH=<gfx> MAX_JOBS=<physical cores> $FLEET_VENV/bin/python -m pip install --no-build-isolation -v -e .
$FLEET_VENV/bin/python -c "import colossalai, colossalai._C.fused_optim_cuda, colossalai._C.inference_ops_cuda, colossalai._C.layernorm_cuda, colossalai._C.moe_cuda, colossalai._C.scaled_masked_softmax_cuda, colossalai._C.scaled_upper_triangle_masked_softmax_cuda, colossalai._C.cpu_adam_x86"
```

(The import names follow `prebuilt_import_path`; the porter confirms them from `base_extension.py`.) The JIT path is checked too: in a clean checkout without `BUILD_EXT`, the first kernel test compiles into `~/.cache/colossalai/torch_extensions`. Use a per-job cache dir, and never JIT-build with the shared checkout as a source path.

Wrap builds and tests in `utils/timeit.sh ColossalAI compile|test -- ...`.

## Test plan

GPU tests, by upstream CI batch (`.github/scripts/ci/run_colossalai_batch.sh` defines them; run the same file lists with `python -m pytest -v -ra --tb=short`, since the script itself expects `CUDA_HOME` for some batches):

Validation breadth follows the Q3 ruling (option b): both wavefront widths run every 1-, 2- and 4-GPU batch, each on a host whose GPUs are all free for the run (RCCL collectives need the whole set); the 8-GPU batch runs on wave64 when a whole 8-GPU host is free. The wave32 run needs four working RDNA GPUs on one host.

| batch | GPUs | content | wave64 | wave32 |
|---|---|---|---|---|
| 3 | 1 | `tests/test_optimizer/test_adam_kernel.py`, `test_adam_optim.py` (FusedAdam, CPUAdam, HybridAdam) | required | required |
| 4b | 1 | `tests/test_infer/test_kernels/cuda/` except flash decoding | required | required |
| 4e | 1 | `tests/test_infer/test_kernels/cuda/test_flash_decoding_attention.py` | required | required |
| 4c | 1 | `tests/test_infer/test_kernels/triton` | required | required |
| 4a | 1 | `tests/test_fp8/test_fp8_cast.py`, `test_fp8_hook.py`, `test_fp8_linear.py` | required | required; the GEMM tests skip where the GPU has no fp8 GEMM (gfx11), recorded |
| new | 1 | `tests/test_kernels` (layernorm, softmax, MoE kernels added by the port) | required | required |
| attn | 1 | `tests/test_shardformer/test_layer/test_attention_backends.py -k gpu` with `COLOSSALAI_TEST_ATTENTION_BACKEND=fa2`, then `=sdpa` | required | required (fa2 through flash-attn's Triton backend, `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`) |
| 2 | 1 | single-GPU core (accelerator, fp16, lazy, inference structures) | required | required |
| 4d | 1 | `tests/test_smoothquant` | required (all four skip at this head on every platform: they import modules upstream removed) | same |
| 1 | 0 | CPU-only configuration, pipeline and tensor logic, GPUs hidden | required | required |
| 5 | 2 | booster/DDP/FSDP/LowLevelZeRO/LoRA/numerics | required | required |
| 6, 7, 8, 9, 9q | 4 | distributed tensor, fp8 collectives, Booster/Gemini/ZeRO/MoE/pipeline/checkpoint, ShardFormer/inference engine | required | required |
| 10 | 8 | `-m largedist` | when 8 GPUs are free, otherwise recorded as not run | not required |
| 11 | 2 | apex-dependent Gemini tests | only if ROCm apex is installed; otherwise recorded as skipped | same |
| 4f | 1 | FA3 strict attention | not applicable (FA3 is Hopper-only, `scoped_out`) | not applicable |

The exact commands for the 2-, 4- and 8-GPU batches are in `notes.md` ("Multi-GPU batch commands for the validators"), prepared by the porter and not yet run.

Record pass/fail/skip counts per batch, and the reason for each skip, in `notes.md`. A batch whose tests skip because `flash_attn` is missing does not count as passing the fa2 path.

Non-GPU regression set: batch 1 (`cpu`: CPU-only configuration, pipeline and tensor logic) and `python -B -m unittest discover .github/scripts/ci` must pass unchanged.

CUDA no-regression: compile the extensions with nvcc against a CUDA torch (mi355 fleet host, `BUILD_EXT=1`) before each presentation. If the host has only ROCm torch, record `cuda-not-validated:` with the exact error, per the moat-porting rule. Every C++ change is under `USE_ROCM` or width-agnostic, and every Python change under `torch.version.hip`, so the CUDA diff should reduce to the added branches.

## Open questions

Q1 to Q5 are ruled (see "Rulings" above). Questions raised in porting round 1 are at the top of that round's section in `notes.md`.
