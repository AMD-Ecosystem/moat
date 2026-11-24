# ColossalAI notes

## Intake screen (2026-11-05, windows-gfx1151 host, intake role)

Screened at upstream `hpcaitech/ColossalAI` main `f4ef8b405cbd444539cc257ba268b327ec00272d` (2026-11-05, version 0.5.0). Source: third-party "AMD AI Software Gap Matrix" seed (verdict not-yet, AMD path none), treated as a lead only.

### Licence

- Top level: Apache-2.0 (`LICENSE`, tier 1). Recorded with `set-license-spdx`, evidence https://github.com/hpcaitech/ColossalAI/blob/main/LICENSE.
- `LICENSE` appends third-party notices; all are permissive (Apache-2.0, MIT, BSD-3-Clause).
- `python3 utils/licenses.py scan-nvidia agent_space/ColossalAI-screen`: no NVIDIA proprietary licence text.
- Kernel provenance read by hand in `extensions/csrc`: NVIDIA/apex fused adam / multi_tensor_apply (BSD-3), vLLM and FasterTransformer attention utils (Apache-2.0, "Copyright (c) 2020-2023, NVIDIA CORPORATION" under Apache), Microsoft DeepSpeed cpu_adam (MIT). All clean.
- One submodule: `examples/tutorial/fastfold/FastFold` -> hpcaitech/FastFold, Apache-2.0, not archived. Tutorial only, not part of the package build.

### Duplicate effort

- Upstream tree: `grep -rniE 'amd|rocm|hip|gfx'` over docs, Python, requirements, Dockerfiles finds nothing ROCm-related (only "AMD 5700xt" in a ColossalChat VRAM table). `colossalai/accelerator/` has cuda, npu, cpu only.
- Upstream PRs (`gh pr list --state all --search "rocm OR hip OR amd"`): none relevant. Branches matching hip/rocm/amd: none. Commit search for rocm/hip: none.
- Upstream issues: #3401 (2023, "Compativility for amd") and #3451 (2023, AMD GPU feature request), maintainer reply "will consider in follow-up"; #5669 (2024-04, "Does colossalai support rocm?"), contributor Edenzzzz replied "Not currently, but some collaboration with AMD is underway". Nothing visible came of it: no PR, branch, doc or docker change since.
- ROCm and AMD-Ecosystem orgs: no ColossalAI repo or fork (`gh search repos`; `ROCm/ColossalAI` 404; code search in ROCm for colossalai empty). The `amd` and `AMD-AGI` orgs are behind SAML SSO for this token and could not be searched.
- Web search (ColossalAI ROCm / MI300 / AMD collaboration / docker): nothing.
- Classification: no AMD path found. The 2024 "collaboration underway" remark is the only signal of a possible internal AMD effort; see question below.

### Viability

- Actively maintained: upstream pushed 2026-11-05, 41k stars, not archived; requirements track torch>=2.5,<2.14 and transformers 5.17.
- CUDA surface: 18 files in `extensions/csrc/kernel/cuda` (~255 KB `.cu`), built AOT with `BUILD_EXT=1` or JIT via `extensions/cuda_extension.py` (torch cpp_extension). Kernels: fused optimizers (adam/lamb/sgd/l2norm/scale, apex-derived), layernorm/rmsnorm, scaled (upper-triang) masked softmax, MoE, inference kernels (flash decoding attention, kv-cache memcpy, fused rotary, fp8 convert). Also Triton kernels (`extensions/triton_extension.py`, `colossalai/kernel/triton`).
- Hazards seen in a quick look (planner to confirm): inline PTX in `extensions/csrc/funcs/cast_functor.h` (mov.b32 / cvt.f32.f16, needs HIP intrinsics); warp-size-32 assumptions (`WARP_SIZE`, `0xffffffff` masks) in reduce_function.h, attention_utils.h, flash_decoding_attention_kernel.cu, layer_norm, l2norm, both softmax kernels; `nvgpu_dev_info.h`. No CUTLASS/CuTe.
- Packaging: `_CudaExtension.assert_compatible` raises when `torch.version.cuda is None`, which is always true on ROCm torch, and the build path requires `CUDA_HOME` and `check_system_pytorch_cuda_match`. So even the pure-PyTorch/Triton parts that should already run on ROCm refuse the kernel path; extension build needs a ROCm branch (torch hipify path, `-DCOLOSSAL_WITH_CUDA` equivalents).
- GPU dependencies (none are MOAT projects, so no `set-deps`): torch (ROCm wheels), triton (ROCm triton), flash_attn (test requirement; ROCm has the CK-based ROCm/flash-attention and upstream Dao-AILab ROCm build), bitsandbytes>=0.39 (upstream ROCm support exists), galore_torch (pure PyTorch). Distributed tests need multi-GPU (RCCL) on the wave64 hosts.
- Tests: large pytest suite under `tests/` (booster, zero, shardformer, moe, fp8, infer, optimizer, ...), many multi-GPU via `colossalai.testing.spawn`. Kernel-level tests in `tests/test_infer` and `tests/test_optimizer`.
- Windows: ColossalAI is Linux/NCCL-oriented (fabric, ray, torch.distributed); the windows gate is likely a waiver candidate, for a person to rule on later.

### Recommendation: fork (take up)

Work: (1) make the extension loader accept ROCm torch (HIP build via torch's hipify, ROCm arch list instead of CUDA_HOME checks); (2) port the 18 kernels: replace inline PTX casts with HIP intrinsics, make warp reductions wavefront-size portable (wave64 and wave32); (3) packaging/docs (ROCm install section, requirements notes for flash_attn/bitsandbytes on ROCm); (4) run the kernel and distributed GPU test suites on wave64 and wave32. Upstream is active and has open AMD-support requests, so a PR has a plausible destination.

### Question for jeff

The only sign of AMD work is a 2024-04 comment on hpcaitech/ColossalAI#5669 ("some collaboration with AMD is underway"); nothing public followed, and the `amd` / `AMD-AGI` GitHub orgs are behind SAML SSO for the token used here, so I could not search them. Do you know of an internal AMD ColossalAI effort that MOAT should coordinate with before forking?

## Planning (2026-11-20, linux-gfx942 fleet host, planner role)

### Questions for Jeff (a ruling is needed before a porter starts)

The plan in `plan.md` assumes the recommended option for each question. A different ruling changes the sections named.

**Q1. Windows gate.** `setup.py:19-20` raises `RuntimeError("Windows is not supported yet. Please try again within the Windows Subsystem for Linux (WSL).")`, and the stack is Linux/NCCL-only (fabric, ray, torch.distributed). The port cannot meet the `windows` gate without adding Windows support upstream never had.
- (a) Recommended: waive the `windows` gate for ColossalAI. Agents may only suggest this; a maintainer approves the waiver.
- (b) Keep the gate and make Windows work too: a project-wide change that goes beyond a ROCm port, with very little chance upstream takes it.

**Q2. Attention backends: is this a library-layer scope cut?** Upstream's fa2/fa3 adapters refuse ROCm (`colossalai/kernel/kernel_loader.py:136-140`, "ROCm needs its own capability policy"). Without a change, ROCm uses the sdpa backend, which already works.
- fa2 option (a), recommended: enable fa2 on ROCm in this port when the ROCm build of `flash_attn` is installed (CK backend on gfx942/gfx950, Triton backend on RDNA), with a ROCm capability check. This is small Python work plus validating the attention tests with flash_attn present.
- fa2 option (b): defer fa2 on ROCm to a follow-up and register a deferral, so ROCm runs sdpa only. This is a partial port of the attention layer.
- fa3: the adapter and its test batch 4f are Hopper/SM90-only by design (`flash_attn_3`), and the package has no AMD counterpart.
  - (a) Recommended: record it `scoped_out` as NVIDIA-architecture-specific.
  - (b) Record it `owed`, which blocks `pr-ready` until something covers it.

**Q3. Validation breadth for the multi-GPU suite.** Upstream CI splits GPU tests into batches by GPU count: 1 GPU (2, 3, 4a-4e), 2 GPUs (5, 11), 4 GPUs (6, 7, 8, 9, 9q), 8 GPUs (10, `largedist`).
- (a) Recommended: wave64 runs every 1-, 2- and 4-GPU batch (it needs a whole 8-GPU MI300-class host, exclusive while it runs); batch 10 runs when 8 GPUs are free, otherwise it is recorded as not run. wave32 runs every 1-GPU batch plus batch 5 if the host has 2 GPUs. The distributed layer is Python on RCCL and identical across arches.
- (b) Both widths run the full 4-GPU set; this needs a 4-GPU RDNA host.
- (c) wave64 runs single-GPU batches only, which leaves distributed training on ROCm untested. Not recommended.

**Q4. Carried over from intake, still unanswered.** Do you know of an internal AMD ColossalAI effort (hpcaitech/ColossalAI#5669, 2024-04: "some collaboration with AMD is underway") that MOAT should coordinate with? The `amd` and `AMD-AGI` orgs were unreachable through SSO.

**Q5. ROCm Docker image.** Upstream ships CUDA images (`docker/Dockerfile`, `docker/torch213`).
- (a) Recommended: no ROCm image in this port; the docs give the ROCm install steps.
- (b) Add a ROCm Dockerfile as an extra deliverable.

### What was analysed

- Upstream `14d1c15147564603d4564198f38af1ea3c42be92` (2026-11-18). Intake saw `f4ef8b4`; the delta is one Python-only merge (#6477).
- Build type recorded: `set-ext-type ColossalAI torch-extension`. Strategy B (torch hipify), with AOT (`BUILD_EXT=1`) and JIT (`cpp_extension.load`) both in scope.
- 7 CUDA-built extensions, 24 CUDA files. `surface.json` has 26 components; `surface.py check` passes. Only the Arm CPU Adam and the Ascend NPU flash attention are `scoped_out` (no CUDA). FA3 awaits Q2.
- Host torch read (not modified): `/opt/python-3.11-venv` torch `2.16.0a0+git6c3af02`, HIP `7.16.26385`, triton 3.9.0. That torch is outside upstream's pin (`torch>=2.5,<2.14`), so a validator must use a job venv with a 2.13.x ROCm wheel.

### Findings worth carrying to the porter

- **AOT define rewrite**: `torch.utils.cpp_extension.BuildExtension._hipify_compile_flags` replaces the first `CUDA` in each `-D` flag under `nvcc` with `HIP`, so on ROCm `-DCOLOSSAL_WITH_CUDA` reaches hipcc as `-DCOLOSSAL_WITH_HIP`. Every header that gates on `#if defined(COLOSSAL_WITH_CUDA)` (`cast_functor.h`, `reduce_function.h`, `common/micros.h`, ...) then compiles empty. JIT `load()` does not go through that method. Check the built module, not only the build log.
- **`C10_WARP_SIZE` on ROCm torch 2.16** (`torch/headeronly/macros/Macros.h`): host code gets `at::cuda::warp_size()` (runtime); device code gets a `constexpr` 64 on `__GFX9__` and 32 otherwise; `C10_WARP_SIZE_{LOWER,UPPER}_BOUND` exist for array sizing. The softmax host launchers' `constexpr int WARP_SIZE = ... C10_WARP_SIZE` therefore fails to compile on ROCm.
- torch adds `-DHIP_ENABLE_WARP_SYNC_BUILTINS=1`, `-D__HIP_NO_HALF_OPERATORS__=1` and `-D__HIP_NO_HALF_CONVERSIONS__=1` to every HIP extension. The extensions' `-U__CUDA_NO_HALF_*` flags are rewritten to `-U__HIP_NO_HALF_*`, which is what they need.
- There are no upstream tests for `layernorm_cuda`, either softmax extension or `moe_cuda`. The plan adds small GPU tests for them.
- The upstream GPU CI batch definitions in `.github/scripts/ci/run_colossalai_batch.sh` (batches 1-11 with GPU counts) are the validator's test list. Running that script asks for `CUDA_HOME` on some batches, so use pytest directly with the same file lists.

### Next

After Jeff rules on Q1-Q5, a porter starts from `plan.md`. If the rulings differ from the recommendations, update the Q-dependent rows in `plan.md` (the test plan table, the `kernel_loader.py` line, `surface.json` for FA3 and Docker) in the first porting round.

## Rulings on the planner's questions (Jeff Daily, 2026-11-23)

- Q1 (windows gate): **waived**; recorded with `approve-waiver ColossalAI windows --by "Jeff Daily"`.
- Q2 (attention backends): fa2 **a** (enable fa2 on ROCm in this port when the ROCm build of `flash_attn` is installed, with a ROCm capability check); fa3 **a** (recorded `scoped_out` as NVIDIA-architecture-specific).
- Q3 (multi-GPU validation breadth): **b**. Both wavefront widths run the full 4-GPU set (1-, 2- and 4-GPU batches on wave64 and on wave32; the 8-GPU batch on wave64 when a whole host is free). The wave32 run needs four working RDNA GPUs on one host, exclusively; the Radeon fleet host's fourth GPU had its driver reset on 2026-11-23 for this.
- Q4 (internal AMD effort): Jeff is not aware of any. The port proceeds without coordination.
- Q5 (ROCm Docker image): **a**, none in this port; the docs give the ROCm install steps.

## Porting round 1 (2026-11-23, linux-gfx942 fleet job, porter role)

### Questions for Jeff (raised in this round; nobody could answer during the job)

1. **Chinese docs.** The CUDA build is also documented in `docs/README-zh-Hans.md` and `docs/source/zh-Hans/get_started/installation.md`. The ROCm section went into `README.md` and `docs/source/en/get_started/installation.md` only, because a Chinese section conflicts with the ASCII-only rule for new upstream prose. Options: (a) leave the zh-Hans docs to upstream's translators (current state); (b) add the same section in English to the zh-Hans files; (c) add a Chinese translation (non-ASCII). Evidence: the zh-Hans install guide mirrors the English one line for line.
2. **Upstream defects found while testing, none caused by the port.** Each was registered with `utils/deferred.py` (ids below) and needs your defer/now ruling; reporting any of them upstream is a GitHub write and yours to approve.
   - `colossalai-upper-triang-softmax-never-loads`: `colossalai/nn/layer/scaled_softmax.py` `ScaledUpperTriangMaskedSoftmax.forward` loads its kernel under `if scaled_upper_triang_masked_softmax:` (should be `is None`), so the fused causal softmax always fails with `AttributeError: 'NoneType' object has no attribute 'forward'` on every platform.
   - `colossalai-softmax-missing-sk-multiple-of-4`: `FusedScaleMaskSoftmax.is_kernel_available` dropped Megatron's `sk % 4 == 0` condition. Above 64 keys the kernels load four elements at a time, so a ragged key length (127, 129, 511, 2047) gives wrong probabilities (measured on gfx942: 13 to 173 mismatched elements per case, up to 0.36 absolute); on NVIDIA the misaligned vector load would fault instead.
   - `colossalai-cpu-adam-fp16-test-unseeded`: `tests/test_optimizer/test_adam_kernel.py::test_cpu_adam_kernel` (fp16 params and grads) draws unseeded data and misses its `rtol=atol=1e-3` on 27 of 400 seeds. A plain-torch emulation of the same fp16 update misses on exactly the same 27 seeds, so it fails on any platform, depending on test order (it failed here only when run after `tests/test_kernels`).
3. **ROCm compiler crash** (`rocm-10.2-llvm-isel-v4bf16-scalar-to-vector`, registered as a rocm-bug-report). With ROCm 10.2.0a20261022 (AMD clang 24.0.0git 2792f8ceeb5a), compiling `flash_decoding_attention_kernel_v1<__hip_bfloat16, uint8_t, 128, 32, 128>` aborts in instruction selection: `LLVM ERROR: Cannot select: v4bf16 = scalar_to_vector` (in the device LTO link, `ld.lld -plugin-opt=mcpu=gfx942 -plugin-opt=O3`). The trigger was an fp8-to-bf16 conversion written as `__float2bfloat16(__half2float(...))`; the port now builds the bf16 bits directly (exact for every E5M2 value), which avoids it. Reproduced again with the hipified tree and the exact hipcc command (recipe under "ROCm compiler crash" below); a 20-line kernel with the same conversion does not reproduce it, and the SDK has no `llc`/`llvm-reduce`, so there is no reduced reproducer yet. Should it be reduced and filed (ROCm/llvm-project), and by whom?

### What changed on the fork (`moat-port`, base `main` = upstream `da3df39e`)

The fork's `main` mirror was 34 commits behind upstream; `moatlib.py mirror-branch ColossalAI` fast-forwarded it to upstream `da3df39e66ed` (2026-11-23) and created `moat-port` from it. The CUDA surface did not change between the planned `14d1c151` and `da3df39e` (`git diff` of `extensions/`, `colossalai/kernel`, `colossalai/quantization`, `colossalai/accelerator`, `setup.py`, `requirements`, the kernel tests: empty).

Commits (head `e2b767e85f8b`):

1. `a2c1b5c8` `[ROCm] Build and run the C++/CUDA kernels with a ROCm PyTorch`: loader accepts ROCm torch, checks ROCM_HOME, skips nvcc probing; nvcc flags translated for hipcc (`to_hipcc_flags`: drop `--use_fast_math`, `-lineinfo`, `--expt-*`, `-maxrregcount`; rename CUDA to HIP in `-D`/`-U` like torch's AOT path, so `-DCOLOSSAL_WITH_HIP` in both AOT and JIT; headers accept either macro); no ROCm include dir in `include_dirs`; `.gitignore` for the hipified copies (`extensions/csrc/kernel/hip/`, `*_hip.h`, `*_hip.cpp`, `*.hip`). Kernels: `COLOSSAL_FULL_WARP_MASK` (`0xffffffffull << (__lane_id() & 32)` on ROCm, `0xffffffffu` on CUDA) in `reduce_function.h`, `attention_utils.h` (also width 32), l2norm; layernorm `WARP_SHFL(..., 32)`; softmax `SOFTMAX_WARP_SIZE` 32 on ROCm; software E5M2 conversions and PTX replacements in `cast_functor.h`; HIP-missing bf16 builders in `data_type.h`; `CastFunctor<at::BFloat16, float>` for HIP bf16 lanes; `namespace cub = hipcub` in `moe_kernel.cu`; `general_hipLaunchKernelGGL` forwarder in `rms_layernorm_kernel.cu`; host `rsqrt` templates excluded on ROCm; flash-decoding shared-memory check; fp32 flash-decoding test reference in float64 on ROCm.
2. `c416f9e3` `[ROCm] Handle fp8 GEMMs, Triton warp limits and ROCm versions`: `scaled_mm_fp8_formats()` (probe, ROCm only; FNUZ on gfx942), `cast_to_fp8` accepts the FNUZ names, fp8 GEMM tests skip by the probe on ROCm; Triton RMSNorm `num_warps` capped at `1024 // warp_size`; `get_version` and `colossalai check -i` report HIP; ten test modules' `version.parse(torch.version.cuda)` gates accept ROCm.
3. `de7ef000` `[ROCm] Enable the FlashAttention-2 backend on AMD GPUs` (Q2 ruling): fa2 on ROCm with the CUDA dtype/mask/head-dim rules plus an AMD GPU check (gfx90a, gfx942, gfx950 for the CK backend; also gfx11, gfx12 with `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`); fa3 keeps refusing ROCm; NVIDIA-emulating CPU tests pin `torch.version.hip = None`; new CPU tests for the ROCm policy.
4. `df45dd5f` `Add GPU tests for the layernorm, softmax and MoE kernels` (`tests/test_kernels`, 155 tests).
5. `180c8741` `[ROCm] Hipify the CPU Adam sources in the runtime build too` (`with_cuda=True` on ROCm in `build_jit`).
6. `e2b767e8` `[ROCm] Document installing on AMD GPUs` (README, `docs/source/en/get_started/installation.md`).

`python3 utils/jargon.py --port ColossalAI`: clean. No non-ASCII in added lines or commit messages; titles 38 to 61 characters.

### Build recipe (linux-gfx942, job venv, no system ROCm)

TheRock nightly index `https://nightly.repo.amd.com/rocm/whl-next/` (the old `rocm.nightlies.amd.com/v2` and `whl-multi-arch` indexes stop before 10.2):

```
D=10.2.0a20261022
$FLEET_VENV/bin/python -m pip install --index-url https://nightly.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple \
  "rocm[libraries,devel]==$D" "rocm-sdk-device-gfx942==$D" "torch==2.13.0+rocm$D" "amd-torch-device-gfx942==2.13.0+rocm$D" "torchvision==0.28.0+rocm$D"
export ROCM_HOME=$($FLEET_VENV/bin/rocm-sdk path --root) ROCM_PATH=$ROCM_HOME HIP_PATH=$ROCM_HOME
export PATH=$FLEET_VENV/bin:$ROCM_HOME/bin:/usr/local/bin:/usr/bin:/bin PYTORCH_ROCM_ARCH=gfx942
# requirements without torch/torchvision/torchaudio/triton/flash_attn, constrained to the installed torch/triton/rocm pins
$FLEET_VENV/bin/python -m pip install -c constraints.txt -r reqs.txt
cd projects/ColossalAI/src && BUILD_EXT=1 PYTORCH_ROCM_ARCH=gfx942 MAX_JOBS=48 $FLEET_VENV/bin/python -m pip install --no-build-isolation --no-deps -v -e .
```

Versions: torch 2.13.0+rocm10.2.0a20261022, HIP 7.17.26422, triton 3.8.0+git66b3f9ff.rocm10.2.0a20261022, AMD clang 24.0.0git, Python 3.11.17, transformers 5.17.0, bitsandbytes 0.50.2 (lazy import; it loads its ROCm 7.15 binary). Full extension build: about 2 minutes at MAX_JOBS=48.

flash-attn for ROCm, built from source in the job directory (CK backend, about 23 minutes):

```
git clone --depth 1 --branch v2.8.3.post2 https://github.com/Dao-AILab/flash-attention fa2 && cd fa2 && git submodule update --init --depth 1 csrc/composable_kernel
BUILD_TARGET=rocm GPU_ARCHS=gfx942 MAX_JOBS=48 FLASH_ATTENTION_FORCE_BUILD=TRUE $FLEET_VENV/bin/python -m pip wheel --no-build-isolation --no-deps -v . -w wheels
$FLEET_VENV/bin/python -m pip install --no-deps wheels/flash_attn-2.8.3.post2-cp311-cp311-linux_x86_64.whl
```

Host gotchas (this fleet image):

- The job venv sees the base env's ROCm 10.1 `amdsmi` through `venv-base-links`; with the 10.2 wheels, `torch.cuda.device_count()` then returns 0 while kernels run (`torch._C._cuda_getDeviceCount()` is 1). Fix: a `.pth` in the venv putting `$ROCM_HOME/share/amd_smi` (the 10.2 copy) first.
- `torch.utils.cpp_extension.ROCM_HOME` resolves from `hipcc` on PATH; with `/opt/python-3.11-venv/bin` first it picks the base env's 10.1 SDK. Put the venv's bin and `$ROCM_HOME/bin` first.
- The GPU this job got (HIP_VISIBLE_DEVICES=2) is a 24 GiB partition (`total: 25767706624`), so the flash-decoding test's own memory guard skips 114 large cases a whole MI300X would run.
- The fleet guard refuses any shell command whose text contains a pip-install phrase unless it names `$FLEET_VENV/bin/python` literally, including prose inside a heredoc; write such text with the editor tool.

Runtime (JIT) path: a source checkout without `BUILD_EXT`, `HOME` pointed at a job-local directory; all seven extensions build at first use (see results).

Documented install: `BUILD_EXT=1 PYTORCH_ROCM_ARCH=gfx942 pip install --no-build-isolation --no-deps .` from a fresh worktree at `df45dd5f` (non-editable; dependencies already present), then the kernel tests run from outside the tree against site-packages (`colossalai._C/*.so` from the wheel).

### Results at `e2b767e8` (porter evidence, one MI300X partition, gfx942, wave64)

All runs: `ulimit -c 0`, `timeout`, `HIP_VISIBLE_DEVICES=2`, pytest flags as upstream's batch script (`-v -s -ra --tb=short --durations=20`, `FAST_TEST=1`), file lists from `.github/scripts/ci/run_colossalai_batch.sh`.

| batch | result | skips |
|---|---|---|
| 3 adam kernels | 62 passed | none |
| 4b inference CUDA kernels | 60 passed, 108 skipped | 108: upstream skips `test_convert_fp8.py` unconditionally ("FP8 conversion still needs improvement") |
| 4e flash decoding | 1326 passed, 258 skipped, 0 failed | 114 memory guard (24 GiB partition); 144 "requires vllm" (comparison against vLLM, not installed) |
| 4c Triton kernels | 372 passed, 65 skipped | 64 "Alibi kernel does not support new kcache layout yet"; 1 unconditional `skip(reason="cuda error")` |
| 4a fp8 | 22 passed | none |
| new `tests/test_kernels` | 155 passed | none |
| attention, `COLOSSALAI_TEST_ATTENTION_BACKEND=fa2 -k gpu` | 185 passed, 383 skipped | the 383 are the fa3/sdpa variants this strict mode deselects |
| attention, `=sdpa -k gpu` | 198 passed, 370 skipped | fa2/fa3 variants deselected |
| attention CPU, `-k 'not gpu'` (GPUs hidden) | 87 passed | none |
| 2 single-GPU core | 23 passed, 2 skipped | upstream HF5 compatibility guards (`test_runtime_compatibility.py:46,100`) |
| 4d smoothquant | 4 skipped | all four import modules upstream removed (`op_builder.smoothquant`, `inference.quant`, `int8_rotary_embedding_fwd`); they skip on CUDA too |
| 1 CPU core (`CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=`) | 358 passed, 599 skipped | GPU-only cases (568 attention GPU cases, plugins needing a device); 875 s against upstream's 15-minute budget |
| `python -B -m unittest discover .github/scripts/ci` | 84 OK | |
| JIT path (no prebuilt kernels): `tests/test_kernels`, adam kernels, rms_layernorm, silu_and_mul, kv_cache_memcpy | 240 passed, 1 failed | the failure is `test_cpu_adam_kernel[fp16-fp16]`, the unseeded fragility above |
| documented non-editable install, tests from outside the tree: `tests/test_kernels` + adam kernels | 186 passed, 1 failed | same CPU Adam fp16 case; it passes when run alone (32/32 twice) |

Targeted checks (scripts kept in the job's scratch area, described here):

- E5M2 conversions against `torch.float8_e5m2` with saturation: every half and bf16 bit pattern (65,536 each) and 16.7M floats (random, small, random bit patterns, edge values) encode bit-identically; all 256 codes decode identically to half, bf16 and float.
- The new layernorm test catches the wave64 fault: with the `WARP_SHFL(..., 32)` change reverted, 42 of 45 cases fail; with it, 45 of 45 pass.
- fp32 flash-decoding cases that missed `atol=1e-7` against the GPU fp32 reference: against a float64 CPU reference the kernel meets the test's tolerance on all three (0 violations), and so does the GPU fp32 reference; they miss only against each other (by up to 1.9e-8 beyond tolerance), because hipBLASLt's fp32 GEMM rounds differently from cuBLAS. `__expf` versus `expf` made no difference.
- `scaled_mm_fp8_formats()` on gfx942 returns `('e4m3fnuz', 'e5m2fnuz')`; the compiled OCP fp8 casts (`torch.compile`, max-autotune) match eager bit for bit, so the `cuda_arch < 89` compile gate (94 on gfx942) is left as is.
- Triton RMSNorm at N=16384: 32 warps fail on gfx942 with `OutOfResources ... threads, Required: 2048, Hardware limit: 1024`; the capped launch matches the reference.

### CUDA no-regression compile

Done on this host, which has no NVIDIA GPU, at `e2b767e8` in a scratch worktree. Toolkit: nvcc 12.9.86 assembled per cuda-to-rocm's validation.md (no conda): `cuda_nvcc`, `cuda_cudart`, `cuda_cccl`, `cuda_profiler_api` from NVIDIA's `redistrib_12.9.1.json` (sha256 checked), extracted with `--strip-components=1`, plus `lib64 -> lib` and the cuBLAS/cuSPARSE/cuSOLVER/cuRAND/cuFFT headers linked from the pip wheels. torch 2.13.0+cu129 and its NVIDIA libraries (`cuda-toolkit[...]==12.9.1`, cudnn, cusparselt, nccl, nvshmem) installed with the job venv's pip into a job-local `--target` directory put first on `PYTHONPATH`. Command:

```
CUDA_HOME=<toolkit> PATH=<toolkit>/bin:$PATH PYTHONPATH=<target> FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.5;8.0;9.0" BUILD_EXT=1 \
  $FLEET_VENV/bin/python setup.py build_ext --inplace
```

Result: exit 0; all seven extensions compiled and linked (six carry an `.nv_fatbin` section; `cpu_adam_x86` is host code). The only warning in project code is upstream's missing return after `assert(false)` in the sm_75 branch of the bf16-to-fp8 functor (`cast_functor.h`, CUDA branch unchanged). Compile-checked with nvcc, not run.

### Multi-GPU batch commands for the validators (prepared, not run)

Not run in this job (one GPU).

These mirror `.github/scripts/ci/run_colossalai_batch.sh` at the port head (same file lists, `PYTEST_EXTRA`, `FAST_TEST=1`, per-batch timeouts and pytest flags). Upstream's script itself cannot run on ROCm because batches 5, 8, 9, 9q, 10 and 11 demand `CUDA_HOME/bin/nvcc`, so call pytest directly. Each batch needs exactly the listed number of GPUs, all on one host and not shared with other jobs while it runs (RCCL collectives). Run from the fork checkout at `head_sha`, built as in "Build recipe" above (`BUILD_EXT=1`, job venv with the ROCm torch 2.13 wheel and the ROCm flash-attn build). On a wave32 host use that host's GPU indices and its `PYTORCH_ROCM_ARCH`; on RDNA the fa2 backend also needs `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` and a flash-attn built with it.

```bash
# from projects/ColossalAI/src, with the job venv's python first on PATH and the env of the 1-GPU runs
ulimit -c 0
export PYTHONPATH=$PWD FAST_TEST=1 PYTHONDONTWRITEBYTECODE=1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=<job dir>/cache/triton-dist TORCHINDUCTOR_CACHE_DIR=<job dir>/cache/inductor-dist
R=<job dir>/results; mkdir -p $R
run() {  # run <label> <timeout minutes> <comma-separated GPU indices> <pytest args...>
  local label=$1 tmin=$2 gpus=$3; shift 3
  HIP_VISIBLE_DEVICES=$gpus timeout --kill-after=60 $((tmin*60)) python -m pytest -v -s -ra --tb=short \
    --maxfail=${MAXFAIL:-10} --durations=20 --junitxml=$R/$label.xml "$@" > $R/$label.log 2>&1
  echo "$label exit=$?"; tail -1 $R/$label.log
}

# 2 GPUs
run 05-booster-lora-2gpu 90 0,1 \
  tests/test_benchmark \
  tests/test_booster/test_plugin/test_dp_plugin_base.py \
  tests/test_booster/test_plugin/test_torch_ddp_plugin.py \
  tests/test_booster/test_plugin/test_torch_fsdp_plugin.py \
  tests/test_booster/test_plugin/test_low_level_zero_plugin.py \
  tests/test_booster/test_plugin/test_gemma4.py \
  tests/test_booster/test_plugin/test_gemma4_moe.py \
  tests/test_booster/test_plugin/test_gemma4_alignment.py \
  tests/test_booster/test_mixed_precision/test_bf16.py \
  tests/test_checkpoint_io/test_low_level_zero_master_resume.py \
  tests/test_checkpoint_io/test_gemini_master_resume.py \
  tests/test_zero/test_gemini/test_grad_accum_reduction.py \
  tests/test_zero/test_gemini/test_tied_accumulation_resume.py \
  tests/test_lora/test_lora.py \
  tests/test_numerics
# batch 11 needs apex; without `import apex.amp` working, record it as skipped (upstream treats pytest exit 5 as a skip)
run 11-apex-gemini-2gpu 150 0,1 tests/test_zero/test_gemini/test_grad_accum.py tests/test_zero/test_gemini/test_grad_clip.py

# 4 GPUs
run 06-distributed-tensor-4gpu 60 0,1,2,3 \
  tests/test_cluster/test_device_mesh_manager.py tests/test_cluster/test_process_group_mesh.py \
  tests/test_device/test_alpha_beta.py tests/test_device/test_device_mesh.py tests/test_device/test_extract_alpha_beta.py \
  tests/test_device/test_init_logical_pg.py tests/test_device/test_search_logical_device_mesh.py \
  tests/test_tensor/test_comm_spec_apply.py tests/test_tensor/test_dtensor/test_comm_spec.py \
  tests/test_tensor/test_dtensor/test_dtensor.py tests/test_tensor/test_dtensor/test_layout_converter.py \
  tests/test_tensor/test_mix_gather.py tests/test_tensor/test_padded_tensor.py tests/test_tensor/test_shape_consistency_apply.py
run 07-fp8-collectives-4gpu 45 0,1,2,3 \
  tests/test_fp8/test_all_to_all_single.py tests/test_fp8/test_fp8_all_to_all.py tests/test_fp8/test_fp8_all_to_all_single.py \
  tests/test_fp8/test_fp8_allgather.py tests/test_fp8/test_fp8_allreduce.py tests/test_fp8/test_fp8_fsdp_comm_hook.py \
  tests/test_fp8/test_fp8_reduce_scatter.py
# batch 7 supplement: the DDP hook file defines no test_ functions at this head, so upstream runs it as a script
HIP_VISIBLE_DEVICES=0,1,2,3 timeout --kill-after=60 2700 python tests/test_fp8/test_fp8_ddp_comm_hook.py > $R/07-fp8-ddp-hook-4gpu.log 2>&1; echo "07-fp8-ddp-hook-4gpu exit=$?"
run 08-distributed-training-4gpu 240 0,1,2,3 \
  tests/test_booster tests/test_checkpoint_io tests/test_pipeline tests/test_optimizer tests/test_zero tests/test_moe \
  -m "not largedist" \
  --ignore=tests/test_booster/test_accelerator.py \
  --ignore=tests/test_booster/test_mixed_precision/test_fp16_torch.py \
  --ignore=tests/test_booster/test_plugin/test_dp_plugin_base.py \
  --ignore=tests/test_booster/test_plugin/test_torch_ddp_plugin.py \
  --ignore=tests/test_booster/test_plugin/test_torch_fsdp_plugin.py \
  --ignore=tests/test_booster/test_plugin/test_low_level_zero_plugin.py \
  --ignore=tests/test_booster/test_plugin/test_gemma4.py \
  --ignore=tests/test_booster/test_plugin/test_gemma4_moe.py \
  --ignore=tests/test_booster/test_plugin/test_gemma4_alignment.py \
  --ignore=tests/test_booster/test_mixed_precision/test_bf16.py \
  --ignore=tests/test_checkpoint_io/test_low_level_zero_master_resume.py \
  --ignore=tests/test_checkpoint_io/test_gemini_master_resume.py \
  --ignore=tests/test_zero/test_gemini/test_grad_accum_reduction.py \
  --ignore=tests/test_zero/test_gemini/test_tied_accumulation_resume.py \
  --ignore=tests/test_optimizer/test_lr_scheduler.py \
  --ignore=tests/test_optimizer/test_adam_kernel.py \
  --ignore=tests/test_optimizer/test_adam_optim.py \
  --ignore=tests/test_pipeline/test_pipeline_utils/test_t5_pipeline_utils.py \
  --ignore=tests/test_pipeline/test_pipeline_utils/test_whisper_pipeline_utils.py \
  --ignore=tests/test_pipeline/test_schedule/test_pipeline_schedule_utils.py \
  --ignore=tests/test_zero/test_gemini/test_grad_accum.py \
  --ignore=tests/test_zero/test_gemini/test_grad_clip.py
# batches 9 and 10 run offline; skip the DeepSeek remote-code files when no local Hub snapshot exists (as upstream does)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
DS=()
python -c 'from transformers import AutoConfig; AutoConfig.from_pretrained("deepseek-ai/deepseek-moe-16b-base", trust_remote_code=True, local_files_only=True)' >/dev/null 2>&1 \
  || DS+=(--ignore=tests/test_shardformer/test_model/test_shard_deepseek.py)
python -c 'from transformers import AutoConfig; AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V3", trust_remote_code=True, local_files_only=True)' >/dev/null 2>&1 \
  || DS+=(--ignore=tests/test_shardformer/test_model/test_shard_deepseek_v3.py)
run 09-model-and-inference-4gpu 120 0,1,2,3 tests/test_infer tests/test_lazy tests/test_shardformer \
  -m "not largedist" "${DS[@]}" \
  --ignore=tests/test_infer/test_async_engine/test_request_tracer.py \
  --ignore=tests/test_infer/test_batch_bucket.py \
  --ignore=tests/test_infer/test_config_and_struct.py \
  --ignore=tests/test_infer/test_kvcache_manager.py \
  --ignore=tests/test_infer/test_request_handler.py \
  --ignore=tests/test_infer/test_kernels/cuda \
  --ignore=tests/test_infer/test_kernels/triton \
  --ignore=tests/test_lazy/test_models.py \
  --ignore=tests/test_lazy/test_ops.py \
  --ignore=tests/test_shardformer/test_model/test_shard_chatglm2.py \
  --ignore=tests/test_shardformer/test_model/test_shard_qwen2.py
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
run 09q-qwen2-4gpu 240 0,1,2,3 tests/test_shardformer/test_model/test_shard_qwen2.py -m "not largedist"

# 8 GPUs (wave64 only, when a whole 8-GPU host is free)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
run 10-largedist-8gpu 360 0,1,2,3,4,5,6,7 tests/test_infer tests/test_shardformer -m largedist "${DS[@]}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
```

Record per batch: pass/fail/skip counts, each skip reason, and for a failure the first error. A batch whose fa2 cases skip because flash_attn is missing does not count as passing the fa2 path.

### For the validators

- Wave32: fa2 on RDNA needs flash-attn built with its Triton backend (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` at build and run time); without it the fa2 acceptance must be recorded as not run, not passed. gfx11 has no fp8 GEMM, so the `scaled_mm` tests skip by the probe (recorded skip); gfx12 takes the OCP types. The fp8 collectives (batch 7) compile OCP fp8 casts with Triton (`cuda_arch` reads 110/120 on RDNA); if that fails to compile on gfx11, record the error.
- Build with `PYTORCH_ROCM_ARCH` set to the host's GPU; the extensions need about 2 minutes, the CK flash-attn build about 23 minutes at MAX_JOBS=48.

### ROCm compiler crash: reproduction recipe

At `a2c1b5c8`, replace the ROCm branch of `CastFunctor<uint8_t, __nv_bfloat16>` in `extensions/csrc/funcs/cast_functor.h` with `return __float2bfloat16(detail::halfraw_to_float(detail::e5m2_to_halfraw(val)));`, run the `BUILD_EXT=1` build once (it hipifies the tree, then fails), and rerun the `hipcc ... flash_decoding_attention_kernel.hip` command from the build log: `LLVM ERROR: Cannot select: ... v4bf16 = scalar_to_vector`, `In function: _Z34flash_decoding_attention_kernel_v1I14__hip_bfloat16hLi128ELi32ELi128EEvPT_PKS1_PKT0_S7_PKiS9_PKfiifiiii`, stack through `SelectionDAGISel::CannotYetSelect` in `ld.lld` (libLLVM.so.24.0git). Compiling the same TU's `--offload-device-only -emit-llvm` bitcode with `clang -O3 -c` or `ld.lld -plugin-opt=O3` does not crash, so the trigger depends on the full offload pipeline.

## Lesson for cuda-to-rocm: torch's runtime extension build hipifies only extensions with CUDA sources

Belongs in strategy-b-torch.md. `torch.utils.cpp_extension.load()` decides `with_cuda` from the source list (any `.cu`/`.cuh`), and hipifies only when it is true, while an ahead-of-time `CUDAExtension` always hipifies. A `CUDAExtension` whose sources are all `.cpp` but include CUDA headers (a CPU optimizer including `cuda_fp16.h`, `cublas_v2.h`) therefore builds with `BUILD_EXT=1` and fails at runtime build with "cublas_v2.h: No such file or directory". Pass `with_cuda=True` to `load()` on ROCm only. Check the runtime build path with nothing prebuilt (a source checkout and a scratch `HOME`), not only the ahead-of-time one. (ColossalAI, torch 2.13)

## Lesson for cuda-to-rocm: torch renames CUDA to HIP in -D flags only in the ahead-of-time build

Belongs in strategy-b-torch.md. `BuildExtension._hipify_compile_flags` rewrites the first `CUDA` in every `nvcc` `-D`/`-U` flag (`-DMYLIB_WITH_CUDA` becomes `-DMYLIB_WITH_HIP`, `-U__CUDA_NO_HALF_OPERATORS__` becomes `-U__HIP_NO_HALF_OPERATORS__`), but `load()` passes `extra_cuda_cflags` through untouched. Headers guarded on the CUDA-named macro then compile empty in the ahead-of-time build (the module imports, the kernels are missing), and the runtime build keeps the half-operator undefines ineffective. Translate the flags yourself on ROCm the same way for both paths, and make the guards accept the HIP spelling. (ColossalAI, found by its planner)

## Lesson for cuda-to-rocm: HIP's OCP fp8 conversions are unusable on gfx942; E5M2 is the top byte of a half

Belongs in faults-numerics.md (fp8). `hip_fp8.h`'s `__hip_cvt_*` functions assert "type is unsupported by current target device" for `__HIP_E5M2`/`__HIP_E4M3` when compiled for gfx942, which supports only the FNUZ encodings, so a CUDA kv cache stored as OCP E5M2 cannot use them there. E5M2 is exactly the upper byte of an IEEE half: decode is `half_bits = byte << 8`; encode from half is round-to-nearest-even on the low byte (`(mag + 0x7f + ((mag >> 8) & 1)) >> 8`) with saturation to 0x7b and NaN to 0x7f; from float, rebias the exponent and round the mantissa to 2 bits, with a separate subnormal branch below 2^-14. Bit-identical to `torch.float8_e5m2` over every half and bf16 pattern. An E5M2 value is also exact in bf16 (the upper half of its float bits), which avoids a rounding conversion entirely. (ColossalAI, ROCm 10.2)

## Lesson for cuda-to-rocm: torch._scaled_mm fp8 types differ by AMD GPU; probe once

Belongs in libraries.md (hipBLASLt). `torch._scaled_mm` on MI300-series (gfx942) accepts only `float8_e4m3fnuz`/`float8_e5m2fnuz` and fails `HIPBLAS_STATUS_NOT_SUPPORTED` for the OCP types; gfx950 and gfx12 take OCP; gfx90a and gfx11 have none. Instead of an architecture list, try a 16x16 `_scaled_mm` with the OCP pair, then the FNUZ pair, once per device (cached, excluded from `torch.compile`), and cast to whatever works; a test of the fp8 GEMM should skip on "no pair works", not on the compute-capability number, which on ROCm is an AMD architecture number. Keep fp8 data exchanged between ranks in the OCP formats so it does not depend on the GPU. (ColossalAI)

## Lesson for cuda-to-rocm: c10::BFloat16 has no conversion from __hip_bfloat16 in an extension

Belongs in strategy-b-torch.md, next to "A torch API used by a ROCm-only branch is a version floor nobody sees". torch 2.13's `torch/headeronly/util/BFloat16.h` declares `BFloat16(const __hip_bfloat16&)` only under `TORCH_HIP_VERSION >= 702`, and no installed header or `cpp_extension` flag defines `TORCH_HIP_VERSION`, so in an extension the guard is always false. CUDA code that passes a `__nv_bfloat16` (hipified to `__hip_bfloat16`) where an `at::BFloat16` is expected compiles on CUDA and fails on ROCm with "no matching function for call". Add an overload or specialization taking the HIP type under `USE_ROCM`. (ColossalAI, torch 2.13)

## Lesson for cuda-to-rocm: hipify breaks token pasting onto a kernel launch, and leaves CUB enumerators unrenamed

Belongs in strategy-b-torch.md ("Headers and functions hipify leaves broken"). (1) A dispatch macro that pastes a prefix onto a launch expression (`general_##__VA_ARGS__` with `kernel<T><<<g, b, 0, s>>>(...)`, choosing `general_kernel`) breaks once hipify rewrites the launch into `hipLaunchKernelGGL((kernel<T>), ...)`: the paste produces `general_hipLaunchKernelGGL`. Define that name under `USE_ROCM` as a forwarder, `#define general_hipLaunchKernelGGL(K, ...) hipLaunchKernelGGL((PASTE K), __VA_ARGS__)` with `#define PASTE(...) general_##__VA_ARGS__`, keeping the parentheses so template commas survive. (2) hipify renames `cub::BlockLoad` to `hipcub::BlockLoad` but not `cub::BLOCK_LOAD_VECTORIZE`; `namespace cub = hipcub;` under `USE_ROCM` fixes all of them. (ColossalAI)

## Lesson for cuda-to-rocm: Triton num_warps above 16 cannot launch on a 64-wide wavefront

Belongs in faults-wavefront.md. Triton's `num_warps` counts wavefronts, so 32 warps on CDNA is 2048 threads; the launch fails with `OutOfResources: out of resource: threads, Required: 2048, Hardware limit: 1024`. Heuristics written for NVIDIA (`min(max(N // 256, 8), 32)`) hit it at large N; cap with `1024 // torch.cuda.get_device_properties(dev).warp_size`, which is unchanged on NVIDIA. Tests often never reach the large-N case, so check it directly. (ColossalAI)

## Lesson for cuda-to-rocm: an LLVM instruction-selection abort on bf16 vector assembly (ROCm 10.2 nightly)

Belongs in diagnosis.md (compiler crashes). `LLVM ERROR: Cannot select: v4bf16 = scalar_to_vector` during the device LTO link for gfx942 (AMD clang 24.0.0git, ROCm 10.2.0a20261022), triggered by bf16 values produced through `__float2bfloat16(__half2float(x))` and stored as bf16x2 pairs in a large kernel. Rewriting the conversion so it does not go through the rounding `__float2bfloat16` (here it was exact, so the bits could be built directly) avoided it. When a ROCm build aborts in ISel, look for the function name in the crash dump, change how that value is produced, and keep the TU and command for a report. (ColossalAI)

## Lesson for moat-porting: a job venv with newer TheRock wheels needs its own amdsmi

Belongs in the "Fleet hosts" section. Fleet job venvs reach the base env's `amdsmi` through `venv-base-links`. After installing newer TheRock wheels into `$FLEET_VENV` (10.2 over a 10.1 base), `torch.cuda.device_count()` returned 0 while kernels ran, because torch counts devices through the old `amdsmi`; tests that check the device count then skip or fail. A `.pth` in the venv's site-packages putting `$(rocm-sdk path --root)/share/amd_smi` first fixes it. Also put the venv's bin and the SDK's bin first on PATH, or `cpp_extension.ROCM_HOME` resolves to the base env's SDK. (ColossalAI)

## Review 2026-11-23 (linux-gfx942 fleet job, reviewer role, fork `moat-port` at `e2b767e8`, base `main` = upstream `da3df39e`)

Verdict: **review-passed**. No blocking findings. The review read all 44 changed files, including the unchanged code around them that they must stay consistent with. Each point below was checked against the code and the installed HIP and torch headers. This host has no NVIDIA GPU. The porter's nvcc compile (notes above) is the only CUDA compile evidence; the mi355 fleet host should repeat it with real nvcc before the port goes to Jeff.

### Verification record (the areas Jeff asked to look at hardest)

- **Warp reductions (wave64, wave32, CUDA).** On ROCm, `COLOSSAL_FULL_WARP_MASK` is `0xffffffffull << (__lane_id() & 32)` (`extensions/csrc/common/micros.h:239`). HIP's `__shfl_*_sync` (`amd_warp_sync_functions.h`) requires a 64-bit mask. It waterfalls over the distinct mask values and asserts that each mask equals the ballot of the lanes passing it. On wave64, each 32-lane half therefore names exactly itself. On wave32, `& 32` is 0 and `__hip_adjust_mask_for_wave32` truncates the mask to `0xffffffff`. On CUDA the macro is `0xffffffffu`, the same value as the removed `kWarpReduceMask` and `uint32_t(-1)`. Every user keeps the shuffles inside 32-lane segments:
  - `warp_reduce` passes width 32 (or the softmax `WARP_SIZE <= 32`). HIP's `__shfl_xor` clamps the source lane to the caller's `width` segment of hardware lanes, and blocks are 1-D or have `blockDim.x` equal to that power of two, so logical warps sit on aligned segments.
  - `attention_utils.h:41-43` now passes `WARP_SIZE` (32). Before the port, the default width on wave64 was 64, so `SHFL_SYNC(x, 0)` in lanes 32-63 read lane 0 of the other warp: that was the wave64 bug. On CUDA, 32 is the default width, so nothing changes.
  - layernorm `WARP_SHFL(..., 32)` is ATen's `WARP_SHFL`: `__shfl(v, lane, width)` on ROCm, with the mask ignored. With `blockDim = (32, k)`, row `y=1` shares a wave64 wavefront with row `y=0`, so width 32 is required there. On CUDA it equals the default `warpSize`.
  - The unchanged `WARP_SHFL_XOR(..., mask)` at `layer_norm_kernel.cu:527` takes xor offsets below 32, which never cross bit 5, so it is correct at any width.
  - The l2norm `__shfl_down_sync` (`multi_tensor_l2norm_kernel.cu:51,95`) runs inside `if (tid < 32)`, with the low-half mask equal to the active set. Its default width is 64 on wave64, so lanes `j + i >= 32` read inactive lanes. Their values never reach the lanes `< lanes` that are used: the lane indices those lanes reach are partial sums of distinct offsets `16..lanes`, always at most 31. Correct as written. Width 32 would be clearer but changes no result.
  - `block_reduce`, `block_max` and `block_sum` index warps as `threadIdx.x >> 5`. That matches the 32-lane logical warps on every platform.
  - Flash-decoding's only early `return` (`flash_decoding_attention_kernel.cu:501`) depends on `blockIdx` alone, so it is block-uniform.
- **Softmax `SOFTMAX_WARP_SIZE`.** It is 32 on ROCm in both the kernels and the host launchers (`get_batch_per_block` and both dispatchers), and `C10_WARP_SIZE` (32) on CUDA. Kernel and launcher geometry agree on every platform. On wave64, two logical warps share a wavefront through width-32 shuffles.
- **Inline PTX and fp8 replacements.** Every change in `cast_functor.h` is `#if defined(USE_ROCM) <new> #else <original> #endif`, and `cuda_fp8.h` is still included on CUDA. The CUDA bodies are byte-identical to upstream (checked in the diff). The packing replacements match PTX `mov.b32 {lo, hi}`: low half first, so `(hi << 16) | lo` and `(hi << 8) | lo`. I re-derived the E5M2 encoders:
  - Half: RNE on the low byte, saturation at `0x7b`, NaN to `0x7f`.
  - Float: rebias by 112, RNE at bit 21, subnormal branch below 2^-14 counting 2^-16 units, saturation at 57344.
  - Both match CUDA's `__NV_SATFINITE` semantics.
  - The bf16 decode builds the bits exactly, which is also the compiler-crash workaround.
  - `data_type.h`'s bf16 helpers and the `CastFunctor<at::BFloat16, float>` specialization are ROCm-only, and the ROCm 10.1 headers define none of those names. The `TORCH_HIP_VERSION >= 702` guard behind that lesson is confirmed in torch's `headeronly/util/BFloat16.h`, and no installed header or `cpp_extension` flag defines the macro.
- **fa2 enablement.** CUDA selection is unchanged. With `torch.version.hip is None`, the first test is false, the new ROCm branch is skipped, and the SM checks run exactly as before. On ROCm, fa3 (`hopper=True`) still refuses. fa2 takes the CUDA dtype, mask, head-dim and alignment rules, then the arch check on `gcnArchName.split(":")[0]`: gfx90a, gfx942 and gfx950 for CK, the same list as flash-attn 2.8.3's `setup.py` `allowed_archs`, plus `gfx11` and `gfx12` when `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, the exact string flash-attn tests.
  - Non-blocking: the `gfx11` prefix also admits gfx1150/1151/1152. flash-attn's Triton backend lists only gfx1100-1102 and gfx1200/1201 in `is_rdna()`. That list only picks autotune configs (`fwd_prefill.py:208-213`, used only under `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE`), and the default config is arch-agnostic. A wave32 validator on a gfx115x GPU should therefore treat fa2 there as needing its own evidence.
- **Triton RMSNorm.** The cap is `1024 // warp_size` (`colossalai/kernel/triton/rms_layernorm.py:109`). On CUDA (32) it is 32, so no change. On wave32 it is 32, 1024 threads. On wave64 it is 16, 1024 threads. `warp_size` exists in torch device properties since 2024-07 (pytorch #128449, torch 2.5), which matches upstream's `torch>=2.5` floor. The other Triton kernels' fixed `num_warps` are at most 16. The one 32-warp autotune config (`no_pad_rotary_embedding.py:16`) is pruned by Triton's autotuner as `OutOfResources` on wave64, so it needs no change.
- **Build path.** `to_hipcc_flags` renames CUDA to HIP in `-D`/`-U` flags, so torch's own AOT rename finds nothing left to change, and AOT and JIT compile the same macros. No extension passes `-gencode` or `-arch`. ROCm targets come from `PYTORCH_ROCM_ARCH`, else from the torch build's arch list, as the docs say.
- **Commits.** All six titles are 72 characters or fewer. `[ROCm]` is on the five ROCm-specific commits and absent from the shared-test commit. Each body has a Test Plan and "Authored with Claude." There is no `Co-Authored-By` and no AMD-internal reference. `utils/jargon.py --port ColossalAI` is clean.
- **Lessons.** The eight `## Lesson for ...` sections above match the code and headers this review checked. None is overstated.

### Non-blocking (no change requested)

- `_LinearFp8.forward` unpacks `scaled_mm_fp8_formats()`. That returns `None` on AMD GPUs with no fp8 GEMM (gfx90a, gfx11), so `linear_fp8` fails there with a `TypeError` instead of a clear message. Without the port it failed in hipBLASLt anyway, and the tests skip by the same probe, so this is a usability nit, not a defect.

## Validation 2026-11-24 (linux-gfx942 fleet job, validator role, fork `moat-port` at `e2b767e8`)

One MI300X partition (gfx942, wave64, 24 GiB, `HIP_VISIBLE_DEVICES=2`). This job used only that GPU. Environment: a fresh job venv with TheRock `10.2.0a20261022` wheels (`rocm[libraries,devel]`, `rocm-sdk-device-gfx942`, `torch==2.13.0+rocm10.2.0a20261022`, `amd-torch-device-gfx942`, `torchvision==0.28.0+rocm...`), installed exactly as in the porter's "Build recipe". No system ROCm. The amdsmi `.pth` fix was applied, and `torch.cuda.device_count()` returned 1. Versions: HIP 7.17.26422, triton 3.8.0+git66b3f9ff.rocm10.2.0a20261022, Python 3.11.17. Requirements were installed with the torch/triton/rocm pins as constraints. All runs used `ulimit -c 0` and `timeout`, the pytest flags of upstream's batch script, `FAST_TEST=1`, and the file lists from `.github/scripts/ci/run_colossalai_batch.sh`, called with pytest directly.

Build: `BUILD_EXT=1 PYTORCH_ROCM_ARCH=gfx942 MAX_JOBS=64 $FLEET_VENV/bin/python -m pip install --no-build-isolation --no-deps -v -e .` built all seven extensions. flash-attn `v2.8.3.post2` was built from source in the job directory with the CK backend (`BUILD_TARGET=rocm GPU_ARCHS=gfx942 FLASH_ATTENTION_FORCE_BUILD=TRUE ... pip wheel --no-build-isolation --no-deps .`) and installed with `--no-deps`.

| batch | result | skips / notes |
|---|---|---|
| new `tests/test_kernels` (layernorm, both softmaxes, MoE) | 155 passed | none |
| 3 adam kernels | 62 passed | none |
| 4a fp8 | 22 passed | none |
| 4b inference CUDA kernels | 60 passed, 108 skipped | 108: upstream's unconditional `test_convert_fp8.py` skip |
| 4c Triton kernels | 372 passed, 65 skipped | upstream's own skips (Alibi kcache layout, one `skip("cuda error")`) |
| 4d smoothquant | 4 skipped | the tests import modules upstream removed; they skip on CUDA too |
| 4e flash decoding | 1326 passed, 258 skipped, 0 failed | 114 memory guard (24 GiB partition), 144 "requires vllm" |
| 2 single-GPU core | 23 passed, 2 skipped | upstream HF5 guards (`test_runtime_compatibility.py:46,100`) |
| attention `COLOSSALAI_TEST_ATTENTION_BACKEND=fa2 -k gpu` | 185 passed, 383 skipped (other backends' variants), 87 deselected | the real ROCm flash-attn 2.8.3.post2 build; strict mode forbids any SDPA fallback |
| attention `=sdpa -k gpu` | 198 passed, 370 skipped, 87 deselected | |
| 4f strict fa3 (`=fa3 -k gpu`) | 10 failed, 19 skipped (maxfail) | all `ModuleNotFoundError: No module named 'flash_attn_3'`. Strict acceptance fails by design without the Hopper-only package, as on any non-Hopper CUDA host. fa3 is `scoped_out` (Q2 ruling), so this is not a port failure |
| 1 CPU core (GPUs hidden) | 358 passed, 599 skipped | device-required guards only. 1048 s while sharing the host with the flash-attn build and GPU batches (upstream budget 15 min; the porter measured 875 s) |
| runtime (JIT) path: a worktree at the head, no prebuilt kernels, scratch `HOME`; `tests/test_kernels`, adam kernels, rms_layernorm, silu_and_mul, kv_cache_memcpy | 240 passed, 1 failed | all seven extensions built at first use into the scratch cache. The failure is `test_cpu_adam_kernel[fp16-fp16]` (one near-zero element, -1.7e-4 vs -1.3e-3), the registered unseeded fragility `colossalai-cpu-adam-fp16-test-unseeded`. Run alone: 12 passed |

These results match the porter's evidence count for count. Integrity: `git -C projects/ColossalAI/src status --porcelain` is empty at `e2b767e8`. `utils/jargon.py --port ColossalAI` is clean. The ROCm build is documented next to the CUDA one in `README.md` and `docs/source/en/get_started/installation.md`. The zh-Hans docs are the porter's open question 1 for Jeff.

CUDA no-regression gate: not run in this job, because this host has no nvcc in the job environment. The porter's nvcc 12.9 compile at this same head (notes above) is the recorded CUDA evidence. The real-nvcc compile on the mi355 fleet host runs elsewhere and is still owed before the port goes to Jeff.

### Multi-GPU batches still owed for linux-gfx942 (Q3 ruling: both wavefront widths)

Not run here: this job had one GPU and must not take others. These batches need a whole host, scheduled separately. `surface.json` keeps them `owed` (`tests:distributed`, `tests:apex-gemini`, `tests:largedist`), which blocks `pr-ready` until they are recorded:

- 2 GPUs: batch 5 (booster/LoRA/numerics), batch 11 (apex Gemini; recorded as skipped if `import apex.amp` fails, as upstream does).
- 4 GPUs: batches 6 (distributed tensor), 7 (fp8 collectives plus the DDP-hook script supplement), 8 (distributed training), 9 (model and inference, offline, DeepSeek files ignored without a local snapshot), 9q (Qwen2).
- 8 GPUs (wave64 only): batch 10 (`largedist`), when a whole 8-GPU host is free.

wave32 owes the same 2- and 4-GPU batches on a 4-GPU RDNA host.

I checked the porter's prepared commands ("Multi-GPU batch commands for the validators" above) mechanically against `run_colossalai_batch.sh` at this head, with a script that compares each batch's test paths, `--ignore` list, `-m` marker and timeout. All eight batches match: 5, 11, 6, 7, 8, 9, 9q and 10. The GPU counts are right, HF offline and the DeepSeek ignores apply to 9 and 10 only, and `tests/test_fp8/test_fp8_ddp_comm_hook.py` defines no `test_` functions, so the script-mode supplement is what upstream runs. Two additions for whoever runs them:

- Upstream's script asserts `torch.cuda.device_count() == <expected>` before each batch; the prepared `run()` does not. Run `python -c 'import torch; print(torch.cuda.device_count())'` with the batch's `HIP_VISIBLE_DEVICES` first. On these fleet images a job venv with newer TheRock wheels reports 0 devices without the amdsmi `.pth` fix, and the distributed tests then skip instead of running.
- Use the venv's `python` first on PATH (`ROCM_HOME` resolves from `hipcc` on PATH), and a flash-attn build in the venv, so the fa2 cases in batches 8 and 9 run instead of skipping.

## CUDA compile check 2026-11-24 (linux-gfx950 fleet job with NVIDIA nvcc, no NVIDIA GPU, fork `moat-port` at `e2b767e8`, base `main` = upstream `da3df39e`)

Verdict: **pass, no CUDA regression.** Both checkouts build all seven extensions with nvcc 12.8. The object set, the compile and link command lines and the warnings are identical. The kernel machine code of 10 of the 17 device objects is byte-identical at every architecture. In the other 7, every difference was traced to one of two causes: an `assert`/`TORCH_CHECK` line-number immediate that moved because the port added lines above it, or the layernorm `WARP_SHFL(..., 32)` width, which equals CUDA's default. Compile-checked only; nothing ran (no NVIDIA GPU, and no AMD GPU was used). No fix is needed. Nothing in the fork was changed or pushed.

### Setup

- Toolkit: the host's `/usr/local/cuda-12.8`, `nvcc` "Cuda compilation tools, release 12.8, V12.8.93". Host compiler g++ 13.3.0 (Ubuntu 24.04), ninja 1.13.0, Python 3.11.17.
- PyTorch: `torch==2.11.0+cu128` (CUDA 12.8) installed into the job venv with `pip install "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128`. The upstream pin allows `<2.14`, but the cu128 index stops at 2.11.0, so 2.11 is the newest cu128 build inside the pin. torch's CUDA major/minor matches nvcc, so `check_system_pytorch_cuda_match` passes.
- The toolkit on this host has no `nvdisasm`. SASS was read with the `nvdisasm` 12.8.55 that the venv's Triton wheel ships (`NVDISASM_PATH`), and section bytes were compared with pyelftools.
- Clone: `git clone https://github.com/AMD-Ecosystem/ColossalAI`, branch `moat-port` = `e2b767e85f8b`, `git merge-base origin/main moat-port` = `da3df39e66ed` = the fork's `main`. Two detached worktrees (`base` at `da3df39e`, `head` at `e2b767e8`), so both builds have the same path length and layout. Both worktrees show a clean `git status --porcelain` after building.

Build, run identically in each worktree (upstream's AOT build; the arch list is upstream's own default for "no visible GPU, CUDA >= 12.8, torch >= 2.7" from `extensions/utils.py:set_cuda_arch_list`, which printed `TORCH_CUDA_ARCH_LIST=6.0;6.1;6.2;7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0`):

```
export PATH=$FLEET_VENV/bin:/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8
export BUILD_EXT=1 FORCE_CUDA=1 MAX_JOBS=21 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
unset TORCH_CUDA_ARCH_LIST
python setup.py build_ext --inplace > build.log 2>&1
```

Base: exit 0, 524 s. Head: exit 0, 525 s. Both builds: `[extension] Building extensions cpu_adam_x86, layernorm_cuda, moe_cuda, fused_optim_cuda, inference_ops_cuda, scaled_masked_softmax_cuda, scaled_upper_triangle_masked_softmax_cuda`. Seven `.so` files each; the six CUDA ones carry `.nv_fatbin` and `cpu_adam_x86` is host code.

Runtime (JIT) path, both checkouts: `LayerNormCudaExtension().build_jit()` with the cache folder redirected to a job directory. Exit 0, the module loads, and it exports `forward_affine` and `backward_affine`. The generated `build.ninja` is identical between base and head once paths are normalized. This exercises `device_flags()` and the `with_cuda=` argument on CUDA.

### Comparison method and results

1. Object set: 24 objects in each tree, same names (17 nvcc device objects, 6 pybind host objects, `cpu_adam.o`).
2. Command lines: all 31 compile and link commands are identical once paths are normalized and ninja's `[n/m]` prefix is dropped. The same flags (`-DCOLOSSAL_WITH_CUDA`, `--use_fast_math`, `-lineinfo`, `--expt-relaxed-constexpr`, `-maxrregcount=50` for layernorm, the 11 `-gencode` pairs) reach nvcc. This confirms that `device_flags()` returns `nvcc_flags()` unchanged on CUDA.
3. Warnings: after normalizing paths and line numbers, the warning lines (783 per build) are the same set with the same counts. All are pre-existing upstream warnings: `std::unary_function`/`std::binary_function` deprecation in the functor headers, `#940-D missing return` in `CastFunctor<__nv_bfloat16, uint8_t>` (the sm_75 `assert(false)` branch, unchanged CUDA code), the pre-sm_75 offline-compilation deprecation, a pybind11 visibility warning and `#pragma unroll` in `cpu_adam.cpp`. No new warning anywhere, and none in the changed files. No errors.
4. Device code: `cuobjdump -xelf all` on each device object (11 cubins each, 187 per tree), then each `.text.<kernel>` section compared byte by byte, with `nvdisasm -c` diffs where they differ:

| object | kernel code base vs head | cause of difference |
|---|---|---|
| activation, get_cos_and_sin, flash_decoding_attention, rms_layernorm, multi_tensor_{adam,l2norm,lamb,scale,sgd}, scaled_masked_softmax, scaled_upper_triang_masked_softmax | identical at all 11 archs | none |
| context_kv_cache_memcpy, decode_kv_cache_memcpy, convert_fp8, fused_rotary_emb_and_cache | identical at sm_80..sm_120; differ at sm_60..sm_75 | only `R8 = 430` becomes `R8 = 550`: the `__assertfail` line argument of the sm_75 `assert(false)` in `CastFunctor<__nv_bfloat16, uint8_t>`. Base `cast_functor.h:430` and head `cast_functor.h:550` are the same line; the ROCm helpers inserted above it moved it. Same instruction count. |
| moe_kernel | differs at all archs | only the `__assertfail` line argument in R8, +5 at every site (lines 14, 35, 57, 80, 109, 137, 172, 203, 369, the `assert(cols % pack_size == 0)` checks), because of the 5-line `USE_ROCM` block added at the top. sm_80..sm_120 encode some of these immediates as `HFMA2 R8, -RZ, RZ, 0, <n * 2^-24>` and sm_120 switches some between `MOV` and `HFMA2`; the decoded values all shift by exactly 5. Same instruction count. |
| layer_norm_kernel | identical at sm_100, sm_120; differs at sm_60..sm_90 (e.g. sm_80: 12519 -> 12495 SASS lines, 44 -> 47 registers in one kernel) | `WARP_SHFL(v, lane)` became `WARP_SHFL(v, lane, 32)`. ATen's default width is `warpSize`, so base computes the `shfl.sync.idx` clamp operand at run time as `((32 - WARP_SZ) << 8) \| 31`, and head passes the constant `31`. These are the same value on every NVIDIA GPU (`WARP_SZ` = 32), and ptxas folds both to identical code for sm_100/sm_120. PTX at sm_80: 95 `shfl.sync` in both, `WARP_SZ` reads 10 -> 5 (the five remaining are the unchanged `WARP_SHFL_XOR` default-width calls). The `rsqrt` host-template guard is `#if !defined(USE_ROCM)`, a no-op on CUDA. Same semantics; the code differs only in register allocation and scheduling. |

5. Host code: `objdump -d -j .text` of each object, with addresses removed. Every pybind object and `cpu_adam.o`, plus 11 of the 17 device objects, are identical. In flash_decoding, rms_layernorm, both softmax kernels, layer_norm and moe, every changed instruction is `mov $<line>,%edx`, the `__LINE__` argument of `TORCH_CHECK`. The shifts equal the number of lines the port added above each check (flash_decoding +21, rms_layernorm +9, softmax +9, upper-triangle softmax +9, layer_norm +4, moe +5). Defined symbols (`nm --defined-only`) are identical for all 24 objects.

### Per-file result (port-changed files in the CUDA build)

All pass. Each header is covered by every object that includes it.

- Build scripts: `extensions/cuda_extension.py`, `extensions/utils.py`: pass. On CUDA, `is_rocm_pytorch()` is False, so `device_flags()` returns `nvcc_flags()` as is, `with_cuda=None` is `load()`'s default, `set_cuda_arch_list` and `append_nvcc_threads` take the original path (the new early return needs `torch.version.hip`), and `get_cuda_home_include` and `assert_build_compatible` keep the CUDA_HOME branch. Evidence: identical command lines (AOT) and an identical `build.ninja` (JIT). `setup.py` is unchanged.
- `common/data_type.h`, `common/micros.h`, `common/mp_type_traits.h`, `common/vec_type_traits.h`, `funcs/binary_functor.h`, `funcs/ternary_functor.h`, `funcs/unary_functor.h`, `funcs/reduce_function.h`: pass. The guard widening to `|| defined(COLOSSAL_WITH_HIP)` leaves CUDA (`-DCOLOSSAL_WITH_CUDA`) unchanged. `COLOSSAL_FULL_WARP_MASK` is `0xffffffffu` on CUDA, the same value as the removed `kWarpReduceMask`. The new bf16 helpers are `USE_ROCM`-only.
- `funcs/cast_functor.h`: pass. The CUDA bodies are unchanged; the only code effect is the assert line number above.
- `kernel/cuda/attention/attention_utils.h`: pass. The added width argument `WARP_SIZE` (32) equals CUDA's default; flash_decoding's kernel code is byte-identical at all archs.
- `flash_decoding_attention_kernel.cu`, `rms_layernorm_kernel.cu`, `multi_tensor_l2norm_kernel.cu`, `scaled_masked_softmax_kernel.cu`, `scaled_upper_triang_masked_softmax_kernel.cu`: pass, kernel code byte-identical (host side: `TORCH_CHECK` line numbers only).
- `moe_kernel.cu`: pass (assert line numbers only).
- `layer_norm_kernel.cu`: pass (equivalent shuffle width; see the table).
- Not part of the CUDA build: `.gitignore`, `README.md`, `docs/.../installation.md`, and the tests.

### Pure-Python changes: the CUDA branch stays reachable

Each new branch keys on `torch.version.hip`, which is `None` on a CUDA PyTorch build. A CUDA install therefore takes the original code everywhere:

- `colossalai/accelerator/cuda_accelerator.py:get_version`: `hip is not None` is False, so it returns `torch.version.cuda` as before.
- `colossalai/cli/check/check_installation.py`: the extra HIP line prints only on ROCm; the CUDA output is unchanged.
- `colossalai/kernel/kernel_loader.py:_flash_attention_support`: upstream refused whenever `hip is not None`, and the port refuses when `hip is not None and hopper`. Both are False on CUDA. The new `if torch.version.hip is not None: return _rocm_flash_attention_support(q)` is also skipped, so CUDA reaches the original SM checks (`(9, 0)` for fa3, `major >= 8` and so on). `import os` is used only by the ROCm helper.
- `colossalai/kernel/triton/rms_layernorm.py`: `max_num_warps = 1024 // get_device_properties(x.device).warp_size` is `1024 // 32 = 32` on every NVIDIA GPU, so `num_warps = min(max(next_pow2(N) // 256, 8), 32)`, the same expression as before. This change is unguarded. `_CudaDeviceProperties.warp_size` first appears in torch v2.5.0 (pytorch `f6a0be5023d`, "Add warpSize to Device properties (#128449)"; absent in v2.4.0), which is exactly the floor of the `torch>=2.5` pin, so every supported CUDA torch has it. It is present in 2.11.0+cu128 here.
- `colossalai/quantization/fp8.py`: `scaled_mm_fp8_formats()` returns `("e4m3", "e5m2")` immediately when `hip is None`; the probe (`_rocm_scaled_mm_fp8_formats`) is never called on CUDA. `_LinearFp8.forward` then casts with `"e4m3"` and stores `ctx.e5m2 = "e5m2"`, which `backward` uses, as the old hard-coded strings did. `cast_to_fp8` now uses `_FP8_TYPES.get(fp8_format, torch.float8_e5m2)`. For `"e4m3"` that is `float8_e4m3fn` and for anything else `float8_e5m2`, the same mapping as upstream's `e4m3fn if fp8_format == "e4m3" else e5m2` for every string except the two new FNUZ names. The `cuda_arch < 89` compile gate is unchanged.
- Tests: the fp8 skips keep `get_device_capability()[0] < 9` on CUDA, `TRITON_CUDA_SUPPORT` keeps `version.parse(torch.version.cuda) > 11.4` after a short-circuited `hip is not None`, and the flash-decoding reference keeps `ref_dtype = dtype` on CUDA. The new `tests/test_kernels` files run on any GPU (`torch.cuda.is_available()`).

Proposed fix: none needed. Optional, not a regression: `assert(false)` without a return in `CastFunctor<__nv_bfloat16, uint8_t>` (sm_75 path) gives the pre-existing `#940-D` warning in base and head alike; it is upstream code and out of the port's scope.
## Validation 2026-11-24 (linux-gfx1100 fleet job, validator role, fork `moat-port` at `e2b767e8`)

Verdict: **validation-failed** for linux-gfx1100 (wave32). Every 1-GPU batch was run. The kernels, Triton kernels, optimizers, fp8 and SDPA attention all pass. The failure is in the fa2 path that the port enables on gfx11: with the only released flash-attn (2.8.3.post2, Triton AMD backend), 2 of 185 strict fa2 acceptance tests fail. Both causes are bugs in flash-attn. With them fixed in a scratch copy of flash-attn, 185 of 185 pass, so the port's adapter code is correct. A source change is still needed before gfx11 fa2 can ship: see "Proposed change for the porter" below. No source was changed and nothing was pushed to the fork.

### Environment

- One Radeon RX 7900-class GPU (gfx1100, wave32, 44.98 GiB visible, `HIP_VISIBLE_DEVICES=0`, pinned by the dispatcher and kept). Host CPU: AMD Ryzen Threadripper PRO 5975WX (Zen 3, AVX2, no AVX-512).
- TheRock `10.2.0a20261022` wheels in the job venv (no system ROCm), as in the porter's "Build recipe" with gfx1100 device packages: `rocm[libraries,devel]`, `rocm-sdk-device-gfx1100`, `torch==2.13.0+rocm10.2.0a20261022`, `amd-torch-device-gfx1100`, `torchvision==0.28.0+rocm...`, **plus `amd-torch-device-gfx110x==2.13.0+rocm10.2.0a20261022`** (see the host gotcha below). HIP 7.17.26422, triton 3.8.0+git66b3f9ff.rocm10.2.0a20261022, Python 3.11.17. The amdsmi `.pth` fix was applied, and `torch.cuda.device_count()` returned 1. Requirements were installed with the torch/triton/rocm pins as constraints (transformers 5.17.0, peft 0.21.2, bitsandbytes 0.50.2).
- Build: `BUILD_EXT=1 PYTORCH_ROCM_ARCH=gfx1100 MAX_JOBS=32 $FLEET_VENV/bin/python -m pip install --no-build-isolation --no-deps -v -e .` built all seven extensions (17 `--offload-arch=gfx1100` compiles, 201 s). `git -C projects/ColossalAI/src status --porcelain` stays empty.
- flash-attn `v2.8.3.post2` built in the job directory with the Triton backend: `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE BUILD_TARGET=rocm FLASH_ATTENTION_FORCE_BUILD=TRUE $FLEET_VENV/bin/python -m pip wheel --no-build-isolation --no-deps -v . -w wheels` (pure Python, under a minute), installed with `--no-deps`. It imports only with `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` set at run time; without it, `import diffusers` (pulled in by `colossalai.inference`) fails with `No module named 'flash_attn_2_cuda'`. Every GPU and CPU run below sets it.
- All runs: `ulimit -c 0`, `timeout`, upstream's pytest flags (`-v -s -ra --tb=short --maxfail=10 --durations=20`), `FAST_TEST=1`, and the file lists of `.github/scripts/ci/run_colossalai_batch.sh`, called with pytest directly. The CPU batch ran with `CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=` (device count 0) and upstream's `cpu_tracebacks` path, concurrently with the GPU batches.

Host gotcha (TheRock torch wheels on RDNA3): `amd-torch-device-gfx1100` carries only `torch/.kpack/torch_gfx1100.kpack`. aotriton's kernel images (`torch/lib/aotriton.images/amd-gfx110x/...`) ship in the family wheel `amd-torch-device-gfx110x`. Without it, SDPA's flash and memory-efficient backends fail with `hipErrorInvalidImage` ("device kernel image is invalid") while the math backend works. Installing the family wheel fixed it (max error against the math backend: 9.8e-4 fp16).

### Results at `e2b767e8` (one gfx1100 GPU)

| batch | result | skips / notes |
|---|---|---|
| new `tests/test_kernels` (layernorm, both softmaxes, MoE) | 155 passed | none |
| 2 single-GPU core | 23 passed, 2 skipped | upstream HF5 guards (`test_runtime_compatibility.py:46,100`) |
| 3 adam kernels | 62 passed | none |
| 4a fp8 | 1 passed, 21 skipped | 21: "Test requires an fp8 GEMM", the port's probe. `scaled_mm_fp8_formats()` returns `None` on gfx1100, and torch itself refuses `_scaled_mm` for both the OCP and FNUZ pairs ("only supported on ... ROCm MI300+"). Expected for gfx11 (porter's note). `test_fp8_cast` passes |
| 4b inference CUDA kernels | 60 passed, 108 skipped | 108: upstream's unconditional `test_convert_fp8.py` skip |
| 4c Triton kernels | 372 passed, 65 skipped | 64 "Alibi kernel does not support new kcache layout yet", 1 unconditional `skip("cuda error")` (both upstream) |
| 4d smoothquant | 4 skipped | the tests import modules upstream removed; they skip on CUDA too |
| 4e flash decoding | 1391 passed, 193 skipped, 0 failed | 144 "requires vllm" (vLLM comparison, not installed); 49 "Required GPU memory is larger than capacity" (measured below) |
| attention `COLOSSALAI_TEST_ATTENTION_BACKEND=sdpa -k gpu` | 198 passed, 370 skipped, 87 deselected | skips are the other backends' variants |
| attention `=fa2 -k gpu` | **2 failed**, 183 passed, 383 skipped, 87 deselected | failures analysed below |
| 4f strict fa3 (`=fa3 -k gpu`) | 10 failed, 19 skipped (maxfail) | all `ModuleNotFoundError: No module named 'flash_attn_3'`, as on gfx942; fa3 is `scoped_out` (Q2 ruling) |
| 1 CPU core (GPUs hidden) | 357 passed, 1 failed, 599 skipped | skips are device-required guards; the failure is pre-existing upstream (below). 832 s, inside upstream's 15-minute budget |
| runtime (JIT) path: worktree at the head, no prebuilt kernels, scratch `HOME`; `tests/test_kernels`, adam kernels, rms_layernorm, silu_and_mul, kv_cache_memcpy | 270 passed, 1 failed | all seven extensions built at first use. The failure is `test_cpu_adam_kernel[fp16-fp16]`, the registered unseeded fragility `colossalai-cpu-adam-fp16-test-unseeded`; run alone it passes (12 passed, twice) |

Not run here (this job has one GPU and must not take more; a later job with four GPUs on this host runs them): batch 5 (2 GPUs), batch 11 (2 GPUs, apex), batches 6, 7, 8, 9, 9q (4 GPUs). Batch 10 (8 GPUs) is wave64-only. `tests:distributed` and `tests:apex-gemini` stay `owed`. Note for that job: TheRock's index also publishes `apex` (`https://nightly.repo.amd.com/rocm/pytorch/whl-next/apex/`), which batch 11 needs, and the job also needs the `amd-torch-device-gfx110x` wheel.

### fa2 failures: two flash-attn 2.8.3.post2 Triton-backend bugs, fixed on flash-attn main

1. `test_attention_gpu_actual_kernel_and_error[False-fa2]` (dense): `out` and `grad/v` align, but `grad/q` (2.34e-1, 155x baseline) and `grad/k` (2.16e-1, 139x) fail. Reproduced without ColossalAI: `flash_attn_func` with a contiguous `dout` gives dq/dk/dv errors of 1.3e-2/9.9e-3/1.3e-2 against float64 SDPA, and with the same `dout` as a transposed view (strides `(2048, 128, 1024, 1)`) dq and dk are off by 1.63 while dv stays 1.3e-2. Cause: `flash_attn_triton_amd/bwd_prefill_split.py` (the default `BWD_MODE=split`; `bwd_prefill_onekernel.py` has the same code) `_bwd_preprocess` addresses `DO` with `O`'s strides (`DO += bid * stride_ob + hid * stride_oh + q_start * stride_om` and a shared `offs_do`), so delta = rowsum(dO * O) is computed from the wrong elements whenever dout's layout differs from out's. `maybe_contiguous` passes such a tensor through, because its last stride is 1. ColoAttention's fa2 path returns `out.transpose(1, 2)` (`colossalai/shardformer/layer/attn.py:879-881`). Any caller that consumes that `(B, H, S, D)` view directly feeds a transposed dout. The shardformer models (`_llama_tf5.py`, `_vit_tf4.py`, ...) call `.permute(...).contiguous()` first, so their dout reaches flash-attn contiguous and they do not hit this.
2. `test_attention_gpu_compile_forward_backward[fa2]`: `TypeError: expected Tensor() for op: torch.ops.flash_attn._flash_attn_forward.default` at `assert_size_stride(buf4, (2,), (1,), ...)`. The Triton backend's `fwd`/`varlen_fwd` return `rng_state = None` when dropout is 0 (`interface_fa.py:81,430`), while the registered fake op returns a `(2,)` int64 tensor, so any `torch.compile(fullgraph=True)` graph over fa2 fails on the Triton backend.

Proof that the port is otherwise correct: a scratch copy of the installed flash-attn with `_bwd_preprocess` given its own `dout` strides and `rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)` in both places, put first on `PYTHONPATH`: the fa2 acceptance batch gives **185 passed**, 383 skipped, 87 deselected. The direct repro gives matching dq/dk for both dout layouts.

Upstream status: flash-attn main (`b038b497`, 2026-11-22) has moved the Triton AMD backend into ROCm/aiter (submodule `third_party/aiter` at `f0966b0a`). At that aiter commit, `_bwd_preprocess` takes `stride_dob, stride_doh, stride_dom, stride_dod`, and `interface_v2.py` always returns a tensor `rng_state`. Both bugs are fixed there. No flash-attn release carries the fix: PyPI and the tags stop at 2.8.3.post2. No upstream report is needed.

### Proposed change for the porter (not made here)

The fa2 enablement on gfx11/gfx12 (`colossalai/kernel/kernel_loader.py:137-144`, `_ROCM_FA2_TRITON_ARCHS`) is correct code, but with flash-attn 2.8.3.post2 it yields wrong dq/dk for non-contiguous upstream gradients and breaks `torch.compile`. Options, smallest first:
- (a) Admit the Triton backend (gfx11, gfx12) only when the installed flash-attn is newer than 2.8.3.post2, that is, a build of flash-attn main with the aiter backend. With 2.8.3.post2, gfx11/gfx12 fall back to SDPA (which passes, 198/198). Document in the ROCm install section that RDNA needs flash-attn built from main with `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` at build and run time. Then a wave32 revalidation runs fa2 against such a build.
- (b) Keep 2.8.3.post2 and work around (1) in ColoAttention's ROCm Triton path, for example by making the incoming gradient contiguous (a small autograd identity around the flash output). This does not fix (2), so `torch.compile` over fa2 on RDNA would stay broken. Not recommended.
- (c) Drop gfx11/gfx12 from the fa2 enablement entirely (SDPA only on RDNA). This narrows Jeff's Q2 ruling ("Triton backend on RDNA"), so it needs his decision.

(a) keeps the Q2 scope and needs no workaround code. The CK path on gfx942 is unaffected (fa2 185/185 there).

### Batch 1 failure: pre-existing upstream, host-dependent

`tests/test_model_zoo/test_gemma4.py::test_gemma4_cpu_ddp_equivalence[1-e2b]` (CPU only, world size 1): `149/150 tensors aligned`, `model.layers.1.mlp.up_proj.weight baseline 3.75e-08 test_err 6.11e-07 ratio 16.30 FAIL (> 2x baseline error)`. Deterministic (3 of 3 reruns with identical numbers). Upstream base `da3df39e` in a separate worktree, same venv: identical failure and numbers, so it is not the port. It passed on the gfx942 host. The likely difference is the host CPU (this one is Zen 3 with AVX2 only), which changes the CPU kernels torch dispatches. That is a hypothesis, not verified. Registered as `colossalai-gemma4-cpu-ddp-equivalence-host-dependent`.

### Flash decoding: measured memory need of the 49 skipped cases

The test catches `torch.cuda.OutOfMemoryError` and skips. Each skipped case was rerun unchanged with `pytest.skip` replaced to print the caught error (script in the job's scratch area). Allocation the reference requested, by case (`KV_GROUP_NUM` 1/2/16 and `HEAD_SIZE` 64/128 share each row):

| cases | alibi | dtype | kv length | batch | single allocation tried | already allocated |
|---|---|---|---|---|---|---|
| 12 | yes | fp16 | 16384 | 32 | 490.36 GiB | 15.6-34.8 GiB |
| 24 | yes | fp16/fp32 | 8192 | 32 | 126.41 GiB | 4.1-14.2 GiB |
| 12 | yes | fp16/fp32 | 16384 | 7 | 50.50 GiB | 7.3-10.7 GiB |
| 1 | no | fp32 | 16384 | 32 | 7.83 GiB | 35.47 GiB (about 43.3 GiB total) |

The alibi cases' single allocation is the test's float alibi mask (batch x heads x S x S), larger than this card's 44.98 GiB. The 490 GiB cases exceed any single GPU, so a whole MI300X skips them too. The one non-alibi case needs about 43.3 GiB because on ROCm its reference is float64 (the port's change, so the tolerance measures the kernel); it does not fit beside the reference's other buffers on 44.98 GiB. No test was shrunk.

### Gates

- CUDA no-regression: not rerun (no nvcc in this job); the porter's nvcc 12.9 compile at this head is the recorded evidence, and the real-nvcc mi355 compile is still owed elsewhere.
- `python3 utils/jargon.py --port ColossalAI`: clean. Integrity: the fork checkout is clean at `e2b767e8`.

## Lesson for cuda-to-rocm: TheRock torch device wheels split kernels between a chip wheel and a family wheel

Belongs in linux-host-setup.md (pip-installed ROCm). With TheRock's per-architecture torch wheels, `amd-torch-device-gfx1100` holds only the torch kernel pack. aotriton's images, which back SDPA's flash and memory-efficient backends, ship in the family wheel `amd-torch-device-gfx110x`. With only the chip wheel, `scaled_dot_product_attention` fails with `hipErrorInvalidImage` ("device kernel image is invalid") on any input that picks those backends, while the math backend works. Install both wheels, or check `torch/lib/aotriton.images/` exists. (ColossalAI, torch 2.13.0+rocm10.2.0a20261022)

## Lesson for cuda-to-rocm: flash-attn 2.8.3's Triton AMD backward reads dout with out's strides

Belongs in libraries.md (FlashAttention on ROCm). In flash-attn 2.8.3.post2's Triton AMD backend (the RDNA path, `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`), the backward preprocess indexes `dout` with `out`'s strides. A non-contiguous `dout` with last stride 1 (for example, the gradient of `out.transpose(1, 2)`) passes `maybe_contiguous` and gives wrong dq and dk while dv stays correct. The same backend returns `rng_state=None` without dropout, which breaks `torch.compile` against the op's fake implementation. Both are fixed on flash-attn main, where the backend comes from ROCm/aiter; no release had the fix as of 2026-11. Test an fa2 integration on RDNA with a transposed upstream gradient and under `torch.compile`, not only with a contiguous `dout`. (ColossalAI)

## Porting round 2 (2026-11-24, linux-gfx1100 fleet job, porter role, fork `moat-port` `e2b767e8` -> `f780f7ea`)

Implements the validator's option (a) from "Validation 2026-11-24 (linux-gfx1100 ...)": the Triton backend stays admitted on gfx11/gfx12 (Jeff's Q2 ruling), but only with a flash-attn that has both fixes; otherwise fa2 is not selected and attention falls back to SDPA. No workaround code for the 2.8.3.post2 bugs (option b) and RDNA is not dropped (option c). One commit on top of `moat-port`, pushed plainly: `f780f7ea [ROCm] Require flash-attn 2.8.4 for its Triton backend`.

### The check and why a version check is justified here

`colossalai/kernel/kernel_loader.py`: new `_ROCM_FA2_TRITON_MIN_VERSION = "2.8.4"` and `_rocm_flash_attention_triton_support()` (`lru_cache`d, so it runs and warns once per process). `_rocm_flash_attention_support` calls it only when the Triton list was chosen (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`) and the arch passed. It reads `importlib.metadata.version("flash-attn")`; below 2.8.4 it emits one `UserWarning` ("Attention backend 'fa2' is not used: it requires flash-attn >= 2.8.4 with the Triton backend (flash-attn X computes wrong gradients for a non-contiguous output gradient and fails under torch.compile); build flash-attn from its main branch with FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE.") and returns that reason, so `auto` moves on to sdpa and an explicit `fa2` raises `NotImplementedError` with it. flash-attn not installed: returns None and leaves it to the existing availability check (unchanged behaviour: auto skips, strict raises ModuleNotFoundError).

Main does bump the version, and the bump postdates both fixes, so the version string identifies the fixed code. Evidence from flash-attn's history (clone in the job directory):
- `701ebe05` (2026-01-28, "[AMD] Triton Backend for ROCm #3") rewrote the in-tree backend: `_bwd_preprocess` takes `stride_dob, stride_doh, stride_dom, stride_dod`, and `interface_v2.py` returns `rng_state = torch.as_tensor([philox_seed, philox_offset])` in `fwd` and `varlen_fwd`. `flash_attn/__init__.py` still said `2.8.3` then.
- `4219765d` (2026-03-11) set `__version__ = "2.8.4"`; `3f94643f` (2026-03-12) moved the backend to ROCm/aiter. setup.py's `get_package_version()` takes the metadata version from `__version__` (plus an optional `+FLASH_ATTN_LOCAL_VERSION`), so a main build installs as `flash_attn 2.8.4`.
- aiter's `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/` was added in aiter `e0d86ba5b` (2025-11-13) already with `stride_dob` in the preprocess and a tensor `rng_state`, so the system-aiter option (`FLASH_ATTENTION_USE_SYSTEM_AITER=TRUE`) cannot pair 2.8.4 with an unfixed backend either.

How it behaves:
- flash-attn 2.8.3.post2 (the release; `__version__ = "2.8.3.post2"`): rejected. Verified on this GPU: auto selects `sdpa` twice with exactly one warning; strict `fa2` raises with the reason.
- flash-attn main `b038b497` (`flash_attn 2.8.4`, backend module `aiter.ops.triton._triton_kernels.flash_attn_triton_amd.interface_v2`): admitted, auto selects `fa2`, no warning.
- A main build from 2026-01-28 to 2026-03-10 (fixed, but reporting `2.8.3`): rejected. Conservative; it falls back to SDPA, never to the bugs.
- Any version, Triton backend not enabled: the check is not reached.

Behaviour on Instinct (stated plainly):
- gfx942/gfx90a/gfx950 with the CK backend (`FLASH_ATTENTION_TRITON_AMD_ENABLE` unset, the documented and validated path): one new line executes, `if supported is _ROCM_FA2_TRITON_ARCHS:`, which is False, then `return None` as before. The selection result is identical; nothing else on that path changed. Import time additionally defines the new constant and function and imports `PackageNotFoundError` (every platform, CUDA included).
- gfx942 etc. with `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` (Triton backend on CDNA, not the validated path): now also gated, because the two bugs are the Triton backend's, not RDNA's. With 2.8.3.post2 this changes fa2 to an SDPA fallback there.
- CUDA: `_rocm_flash_attention_support` is never called; no executed line changes.

Tests (`tests/test_shardformer/test_layer/test_attention_backends.py`): `test_attention_rocm_fa2_architectures` now pins the version to 2.8.4 and clears the cache, so its six cases do not depend on the installed flash-attn. New `test_attention_rocm_fa2_triton_version` (5 cases: gfx1100/2.8.3.post2, gfx1201/2.8.3, gfx942+Triton/2.8.3.post2 -> sdpa with the warning; gfx1100/2.8.4 and 2.8.4+rocm -> fa2), which also checks that a second selection does not warn again (warnings as errors) and that strict fa2 raises the reason.

Docs: README.md and docs/source/en/get_started/installation.md (the ROCm install section, after the existing flash-attn paragraph, both identical): CK on gfx90a/gfx942/gfx950; on gfx11/gfx12 flash-attn >= 2.8.4, built from main, with the backend enabled at build and run time:

```
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE AITER_USE_SYSTEM_TRITON=1 pip install --no-build-isolation .
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
```

Verified here exactly (with `$FLEET_VENV/bin/python -m pip`) at flash-attn `b038b497`, aiter submodule `f0966b0a`: installs `flash_attn-2.8.4` and `amd-aiter` (plus `flydsl 0.3.4.1`); `pip freeze` before/after shows torch, triton 3.8.0+git66b3f9ff.rocm10.2.0a20261022 and the rocm wheels unchanged. `AITER_USE_SYSTEM_TRITON=1` is flash-attn's own documented switch to keep the PyTorch-matched Triton (without it aiter's setup installs its own Triton).

### Environment (this job)

Fresh job venv, TheRock `10.2.0a20261022`: `rocm[libraries,devel]`, `rocm-sdk-device-gfx1100`, `torch==2.13.0+rocm10.2.0a20261022`, `amd-torch-device-gfx1100`, `amd-torch-device-gfx110x`, `torchvision==0.28.0+rocm10.2.0a20261022`, the amdsmi `.pth`, requirements + requirements-test constrained to the installed torch/triton/rocm pins. GPU: AMD Radeon PRO W7800 48GB (gfx1100), `HIP_VISIBLE_DEVICES=0` kept, device count 1. Extensions: `BUILD_EXT=1 PYTORCH_ROCM_ARCH=gfx1100 MAX_JOBS=32 pip install --no-build-isolation --no-deps -v -e projects/ColossalAI/src` (17 gfx1100 compiles). flash-attn 2.8.3.post2: wheel as in the validator's recipe. Lint: black 120 / isort profile black clean on the two Python files. Gotcha: `utils/timeit.sh` runs its command from the MOAT root, so give `pip install -e` an absolute path.

### Results at `f780f7ea` (one gfx1100 GPU, `ulimit -c 0`, `timeout`, `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, `FAST_TEST=1`)

| flash-attn | run | result |
|---|---|---|
| any | `test_attention_backends.py -k 'not gpu'` | 92 passed, 568 deselected |
| 2.8.3.post2 | selection probe | auto -> sdpa, sdpa; 1 warning; strict fa2 -> NotImplementedError with the reason |
| 2.8.3.post2 | `COLOSSALAI_TEST_ATTENTION_BACKEND=sdpa -k gpu` | 198 passed, 370 skipped, 92 deselected |
| 2.8.3.post2 | `=fa2 -k gpu` (`--maxfail=3`) | 3 failed, all the intended `NotImplementedError: Attention backend 'fa2' requires flash-attn >= 2.8.4 ...` (strict mode refuses instead of falling back) |
| main 2.8.4 | selection probe | auto -> fa2, no warning |
| main 2.8.4 | `=fa2 -k gpu`, no scratch patch, `PYTHONPATH` = the fork checkout only | **185 passed**, 383 skipped, 92 deselected (349 s) |
| main 2.8.4 | `=sdpa -k gpu` | 198 passed, 370 skipped, 92 deselected |

The 5 more deselected tests than in the validator's run are the new CPU test's cases (`-k gpu` deselects them).

### For the CUDA compile recheck owed elsewhere

The delta `e2b767e8..f780f7ea` touches four files, none compiled: `colossalai/kernel/kernel_loader.py`, `tests/test_shardformer/test_layer/test_attention_backends.py`, `README.md`, `docs/source/en/get_started/installation.md`. No C++/CUDA source, header, setup.py or extension build file changed, so the nvcc compile of the extensions is unaffected; the recheck needs only `python -c "import colossalai.kernel.kernel_loader"` and `pytest tests/test_shardformer/test_layer/test_attention_backends.py -k 'not gpu'` under a CUDA torch.

## Review 2026-11-24 (linux-gfx1100 fleet job, reviewer role, delta `e2b767e8...f780f7ea`, pr-review skill)

Verdict: **changes-requested**. Scope is the one-commit delta; the full port passed review at `e2b767e8`. The review ran as an independent pr-review pass, and every finding was rechecked here before recording.

1. **Blocker: the SDPA fallback breaks `torch.compile(fullgraph=True)`.** `colossalai/kernel/kernel_loader.py:159`: `warnings.warn` sits inside the `lru_cache`d `_rocm_flash_attention_triton_support()`. Dynamo traces through `lru_cache` into the body on every compile, and it cannot trace `_warnings.warn`. Rechecked on this GPU (real gfx1100, only `kernel_loader.version` patched): `torch.compile(f, backend="eager", fullgraph=True)` over a function doing `ColoAttention._select_backend(x, x, "auto", ...)` compiles and gives fa2 with version 2.8.4, and raises `Unsupported: Dynamo does not know how to trace the Python builtin _warnings.warn` with 2.8.3.post2, after an eager warm-up. At `e2b767e8` there was no warning, so this is a regression from the delta: a compiled model on RDNA with the released flash-attn fails instead of falling back. (A first CPU recheck with a `SimpleNamespace` device-properties stub failed for both versions on the stub itself, so it says nothing; the GPU recheck is the evidence.) Fix: do not warn while compiling (`torch.compiler.is_compiling()`), and add a CPU test of a fullgraph-compiled auto selection under the 2.8.3.post2 stub, built from objects Dynamo can trace.
2. **Minor: the docs paragraph misstates flash-attn.** `README.md:554`, `docs/source/en/get_started/installation.md:82` say that on gfx11/gfx12 flash-attn "needs its Triton backend". flash-attn main (the documented build) added RDNA 3/4 to its composable_kernel backend in `5301a359` (2026-03-27, "[AMD ROCm] Update CK and add RDNA 3/4 support"; `setup.py` `validate_and_update_archs` allows gfx1100-gfx1201, README lists "RDNA 3/4 GPUs" under CK). ColossalAI still admits gfx11/gfx12 only with the Triton backend (`_ROCM_FA2_CK_ARCHS`, unchanged, per Jeff's Q2 ruling "Triton backend on RDNA"), so the sentence should say ColossalAI uses flash-attn's Triton backend there, not that flash-attn needs it. "Radeon RX 7000 and 9000 series" also leaves out Radeon PRO parts (this host's W7800). Fix: "On Radeon GPUs (gfx11, gfx12), Colossal-AI uses flash-attn's Triton backend, which needs flash-attn 2.8.4 or later ...". Whether to also admit CK on RDNA with flash-attn >= 2.8.4 is a scope question for Jeff, not this round (see "Open for Jeff" below).
3. **Minor: a test leaves the gate cache filled with a mocked result.** `tests/test_shardformer/test_layer/test_attention_backends.py:514-516`: `test_attention_rocm_fa2_architectures` clears the cache before selecting but not after, so the TRUE cases leave `None` (supported, from the patched "2.8.4") cached after monkeypatch restores `version`. In file order the next test clears it; with another order or `-k` selection, on a gfx11 host with flash-attn 2.8.3.post2, later fa2 tests would select the buggy backend. Fix: an autouse-style fixture or `try/finally` that clears before and after, for both tests.

Checked with no finding: version parsing of `2.8.3.post2` and `2.8.4+rocm`; a broken or unidentifiable install fails the same way the existing `is_available` check does; CUDA never reaches the gate; the CK path (Triton not enabled) selects identically; eager warn-once works; the notes' flash-attn history (`701ebe05`, `4219765d`, `3f94643f`, aiter `e0d86ba5b`) and setup.py version derivation hold; setup.py inits `third_party/aiter` itself, so the documented plain clone works; commit title 55 chars, `[ROCm]` correct, prose disclosure, ASCII, no internal vocabulary; `-k 'not gpu'` 92 passed.

Open for Jeff (not a finding against this delta): flash-attn >= 2.8.4 also supports gfx11/gfx12 with its default CK backend, which ColossalAI still refuses on RDNA (falls back to SDPA without a message when `FLASH_ATTENTION_TRITON_AMD_ENABLE` is unset). Admitting CK on RDNA for >= 2.8.4 would widen the Q2 ruling, so it needs his decision.

## Porting round 3 (2026-11-24, linux-gfx1100 fleet job, porter role, `f780f7ea` -> `632c6026`)

Answers the three findings of the review above in one commit: `632c6026 [ROCm] Keep the flash-attn version fallback traceable by torch.compile`.

1. Blocker: `colossalai/kernel/kernel_loader.py` wraps the warning in `if not torch.compiler.is_compiling():`. Eager selection still warns once; under compile the warning is skipped and the reason is still returned, so auto falls back to sdpa and strict fa2 raises. If a process only ever selects inside compiled graphs, no warning is printed (tracing does not fill the eager `lru_cache`, so the first eager selection warns). `torch.compiler.is_compiling` exists from torch 2.3; the line runs only with flash-attn's Triton backend enabled and flash-attn < 2.8.4, and 2.8.3's Triton backend asks for Triton 3.2 (torch >= 2.6), and the package itself requires `torch>=2.5` (`requirements/requirements.txt:10`; the README's "2.2 <= PyTorch" is older upstream text), so it is available wherever the line runs. Executed-line accounting is otherwise as in round 2: the gfx942 CK path and CUDA execute nothing new from this commit.
2. Docs (both files): "On Instinct GPUs (gfx90a, gfx942, gfx950), Colossal-AI uses the default composable_kernel backend of `flash-attn`. On Radeon GPUs (gfx11, gfx12), it uses the Triton backend of `flash-attn` 2.8.4 or later, ..." The build command block is unchanged.
3. Tests: a `clear_rocm_fa2_triton_cache` fixture clears the cache before and after both version-dependent tests. New `test_attention_rocm_fa2_triton_version_compiled` (2 cases) compiles `x + (1 if _rocm_flash_attention_triton_support() else 2)` with `fullgraph=True, backend="eager"`. It traces the gate itself and stubs no device: a first version stubbed `torch.cuda.get_device_properties` to compile a full `_select_backend`, and on a GPU host Dynamo's own Triton check (`torch/utils/_triton.py` `cuda_extra_check`, `.major`) read the stub and failed, also leaving `test_attention_compile_dense_forward_backward` failing after it. Negative check: with the guard replaced by `if True:`, the 2.8.3.post2 case fails.

Results at `632c6026` (gfx1100, flash-attn main 2.8.4 installed, `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`):

| run | result |
|---|---|
| `test_attention_backends.py -k 'not gpu'`, GPU visible | 94 passed, 568 deselected |
| same, `CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=` | 94 passed, 568 deselected |
| real-GPU fullgraph compile of `_select_backend(x, x, "auto", ...)`, only `kernel_loader.version` patched | 2.8.4: fa2 (eager and compiled); 2.8.3.post2: sdpa (eager and compiled; before this commit, Unsupported `_warnings.warn`) |
| `COLOSSALAI_TEST_ATTENTION_BACKEND=fa2 -k gpu` | 185 passed, 383 skipped, 94 deselected |
| `=sdpa -k gpu` | 198 passed, 370 skipped, 94 deselected |

Files the delta `e2b767e8..632c6026` touches (for the CUDA recheck): still only `colossalai/kernel/kernel_loader.py`, `tests/test_shardformer/test_layer/test_attention_backends.py`, `README.md`, `docs/source/en/get_started/installation.md`. Nothing compiled changed.

## Review 2026-11-24 (linux-gfx1100 fleet job, reviewer role, delta `f780f7ea...632c6026` and the whole `e2b767e8...632c6026`, pr-review skill)

Verdict: **review-passed** at `632c6026`. All three findings of the previous review are closed: the warning is guarded by `torch.compiler.is_compiling()` (`colossalai/kernel/kernel_loader.py:159`), available on every torch the package allows (`torch>=2.5`, the API exists from 2.3); `test_attention_rocm_fa2_triton_version_compiled` fails with the guard disabled (`Unsupported` at 2.8.3.post2) and passes with it, and tracing leaves the eager cache empty; the `clear_rocm_fa2_triton_cache` fixture covers all three version-dependent tests; the docs paragraph (`README.md:554`, `docs/source/en/get_started/installation.md:82`, identical) now states the backend Colossal-AI uses per GPU family, consistent with `_ROCM_FA2_CK_ARCHS`/`_ROCM_FA2_TRITON_ARCHS`. Whole delta: ASCII, titles 55 and 70 characters with a correct `[ROCm]` prefix, prose AI disclosure, no internal vocabulary (`jargon.py --port ColossalAI` clean), fork checkout clean at `632c6026`. CPU: `-k 'not gpu'` 94 passed. One nit in these notes (a wrong source for the torch floor in "Porting round 3") was corrected in place. The open question for Jeff about CK on RDNA with flash-attn >= 2.8.4 stands.

## Validation 2026-11-24 (linux-gfx1100 fleet job, validator role, fork `moat-port` at `632c6026`)

Verdict: **completed** for linux-gfx1100 (wave32). Every 1-GPU batch ran as in the previous gfx1100 validation; all pass except the two registered upstream fragilities, which are not this port's and do not gate. Strict fa2 against flash-attn main: **185 of 185**, no scratch patch.

### Environment

- One AMD Radeon PRO W7800 48GB (gfx1100, wave32), `HIP_VISIBLE_DEVICES=0` pinned by the dispatcher and kept; `torch.cuda.device_count()` 1.
- Job venv as in "Porting round 2": TheRock `10.2.0a20261022`, torch `2.13.0+rocm10.2.0a20261022` with **both** `amd-torch-device-gfx1100` and `amd-torch-device-gfx110x`, triton `3.8.0+git66b3f9ff.rocm10.2.0a20261022`, Python 3.11, amdsmi `.pth`. Extensions built (editable install, so Python changes are live) during porter round 2 in this same checkout; no compiled source changed between `e2b767e8` and `632c6026`, so they are the head's kernels: `BUILD_EXT=1 PYTORCH_ROCM_ARCH=gfx1100 MAX_JOBS=32 pip install --no-build-isolation --no-deps -v -e <abs path to src>` (17 gfx1100 compiles).
- flash-attn for the GPU batches: main `b038b497` (aiter submodule `f0966b0a`), installed with the documented command (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE AITER_USE_SYSTEM_TRITON=1 pip install --no-build-isolation .`); reports `flash_attn 2.8.4`. `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` set for every run.
- All runs: `ulimit -c 0`, `timeout --kill-after=60`, upstream's pytest flags (`-v -s -ra --tb=short --maxfail=10 --durations=20`), `FAST_TEST=1`, file lists from `.github/scripts/ci/run_colossalai_batch.sh`, pytest called directly, job-local `HOME`, Triton and inductor caches.

### Results at `632c6026` (one gfx1100 GPU)

| batch | result | skips / notes |
|---|---|---|
| new `tests/test_kernels` | 155 passed | none |
| 2 single-GPU core | 23 passed, 2 skipped | upstream HF5 guards |
| 3 adam kernels | 61 passed, **1 failed** | `test_cpu_adam_kernel[p_dtype2-g_dtype2-0.0-True]` (fp16/fp16, CPU kernel): the registered unseeded fragility `colossalai-cpu-adam-fp16-test-unseeded`. One element 0.5900 vs 0.5884 at rtol/atol 1e-3. Run alone twice (`-k cpu_adam`): 12 passed, then 1 failed/11 passed. Not gated |
| 4a fp8 | 1 passed, 21 skipped | "Test requires an fp8 GEMM": gfx11 has none (expected) |
| 4b inference CUDA kernels | 60 passed, 108 skipped | upstream's unconditional `test_convert_fp8.py` skip |
| 4c Triton kernels | 372 passed, 65 skipped | 64 Alibi kcache layout, 1 unconditional "cuda error" (upstream) |
| 4d smoothquant | 4 skipped | modules upstream removed |
| 4e flash decoding | 1391 passed, 193 skipped, 0 failed | 144 "requires vllm", 49 "Required GPU memory is larger than capacity" (measured in the previous gfx1100 validation) |
| attention `COLOSSALAI_TEST_ATTENTION_BACKEND=sdpa -k gpu` | 198 passed, 370 skipped, 94 deselected | other backends' variants |
| attention `=fa2 -k gpu`, flash-attn main 2.8.4 | **185 passed**, 383 skipped, 94 deselected | `PYTHONPATH` = the fork checkout only, no scratch flash-attn |
| 4f strict fa3 (`=fa3 -k gpu`) | 10 failed, 19 skipped (maxfail) | all `ModuleNotFoundError: No module named 'flash_attn_3'`; fa3 is `scoped_out` (Q2 ruling) |
| 1 CPU core, `CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=` (device count 0), flash-attn 2.8.3.post2 (see below) | 364 passed, **1 failed**, 599 skipped | the failure is `test_gemma4_cpu_ddp_equivalence[1-e2b]`, registered `colossalai-gemma4-cpu-ddp-equivalence-host-dependent`: identical numbers to the previous run (149/150 aligned, `model.layers.1.mlp.up_proj.weight` 3.75e-08 vs 6.11e-07, ratio 16.30). +7 passed versus the previous run are the new attention CPU tests. 706 s |
| runtime (JIT) path: worktree at `632c6026`, `PYTHONPATH` = worktree (imports confirmed from it), fresh `HOME`; `tests/test_kernels`, adam kernels, rms_layernorm, silu_and_mul, kv_cache_memcpy | 270 passed, 1 failed | all seven extensions built at first use into `$HOME/.cache/colossalai/torch_extensions`; the failure is the same cpu_adam fp16 fragility (`[p_dtype2-g_dtype2-0.0-False]`) |

Selection checks (from the porter rounds, same tree): with flash-attn 2.8.3.post2, auto selects sdpa with one warning and strict fa2 raises the reason; the sdpa batch passes 198/198 with 2.8.3.post2.

Not run here (one GPU in this job): batch 5 (2 GPUs), batch 11 (2 GPUs, apex), batches 6, 7, 8, 9, 9q (4 GPUs); a later four-GPU job on this host runs them. Batch 10 (8 GPUs) is wave64-only. `tests:distributed` and `tests:apex-gemini` stay `owed`.

### CPU batch and flash-attn main: aiter needs a GPU at import

With flash-attn main installed, the CPU batch (GPUs hidden) failed at collection (3 errors: `test_request_tracer.py`, `test_gemma4.py`, `test_gemma4_moe.py`): `colossalai.inference` and `tests/kit/model_zoo/diffusers` import diffusers, which imports flash-attn, which imports `aiter.ops.triton._triton_kernels.flash_attn_triton_amd`; at aiter `f0966b0a`, `aiter/ops/triton/utils/_triton/arch_info.py` runs `triton.runtime.driver.active.get_current_target()` at import, which raises `RuntimeError: 0 active drivers` with no GPU and then falls back to `from jax._src.lib import gpu_triton` (`No module named 'jax'`). Unsetting `FLASH_ATTENTION_TRITON_AMD_ENABLE` does not help, since flash-attn main falls back to the Triton backend when `flash_attn_2_cuda` is missing. aiter fixed it in `b1cdc19eb` (2026-10-08, the probe moved to first use, with a comment naming exactly this case); flash-attn main still pins the older `f0966b0a` (2026-09-19). Not this port's code and no report needed (fixed upstream, awaiting flash-attn's submodule bump). The CPU batch was therefore run with flash-attn 2.8.3.post2 installed, as in the previous gfx1100 validation; it exercises no fa2 kernel.

### Gates

- Integrity: `git -C projects/ColossalAI/src status --porcelain` empty at `632c6026`; the JIT worktree was removed.
- `python3 utils/jargon.py --port ColossalAI`: clean. Docs: the ROCm install section in README.md and docs/source/en/get_started/installation.md carries the flash-attn build for RDNA.
- CUDA no-regression: not run here (no nvcc in this job). **Owed on another host at `632c6026`.** The delta since the last CUDA compile (`e2b767e8`) touches only `colossalai/kernel/kernel_loader.py`, `tests/test_shardformer/test_layer/test_attention_backends.py`, `README.md`, `docs/source/en/get_started/installation.md`: no compiled file, so that job needs only an import of `colossalai.kernel.kernel_loader` and `pytest tests/test_shardformer/test_layer/test_attention_backends.py -k 'not gpu'` under a CUDA torch (the extension compile is unchanged from `e2b767e8`).
- linux-gfx942 was validated at `e2b767e8` and now reads `revalidate`; the delta changes no line the CK path executes beyond one False condition (see "Porting round 2"), so a carry-forward over this Python-only delta is that host's call.

### Open for Jeff

flash-attn >= 2.8.4 also supports gfx11/gfx12 in its default composable_kernel backend (`5301a359`). ColossalAI still refuses CK on RDNA (silent SDPA when `FLASH_ATTENTION_TRITON_AMD_ENABLE` is unset), per the Q2 ruling. Whether to admit CK on RDNA for flash-attn >= 2.8.4 is his decision; nothing was changed for it.

## Lesson for cuda-to-rocm: flash-attn main's Triton backend cannot be imported without a GPU (aiter before b1cdc19eb)

Belongs in libraries.md (FlashAttention on ROCm). flash-attn's main branch takes its Triton AMD backend from ROCm/aiter (submodule `third_party/aiter`). With the aiter it pinned as of 2026-11 (`f0966b0a`), importing flash-attn in a process that sees no GPU fails: aiter's `arch_info.py` queries Triton's active driver at import (`RuntimeError: 0 active drivers`) and falls back to importing jax. Anything that imports flash-attn indirectly breaks too, for example diffusers, so a CPU-only test job of a project that depends on diffusers fails at collection. Unsetting `FLASH_ATTENTION_TRITON_AMD_ENABLE` does not avoid it, because flash-attn main falls back to the Triton backend when its compiled CK module is missing. aiter fixed it in `b1cdc19eb` (2026-10-08). Until flash-attn bumps its submodule, run GPU-less jobs without flash-attn main installed, or install a newer aiter with `FLASH_ATTENTION_USE_SYSTEM_AITER=TRUE`. (ColossalAI)
