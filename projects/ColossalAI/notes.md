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
