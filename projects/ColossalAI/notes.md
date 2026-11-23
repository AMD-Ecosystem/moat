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
