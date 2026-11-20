# espnet notes

## Intake screen (2026-11-05, windows-gfx1151 host, gap-matrix seed)

Source of lead: third-party "AMD AI Software Gap Matrix" seed (verdict: unknown, AMD path unknown). Screened at upstream master, shallow clone 2026-11-05.

### Licence
- `licenses.py check espnet/espnet`: Apache-2.0, tier 1 (LICENSE read; GitHub field agrees).
- `licenses.py scan-nvidia agent_space/espnet-screen`: no NVIDIA proprietary licence text. `.gitmodules` exists but is empty; nothing vendored with CUDA.

### CUDA surface
- Only CUDA source in tree: `espnet2/asr_transducer/decoder/modules/rwkv/cuda/wkv_cuda.cu` (4.8 KB, RWKV WKV kernel; same kernel FunASR vendors). JIT-built via `torch.utils.cpp_extension.load` in `rwkv/attention.py`, only for the RWKV transducer decoder; optional.
- The loader passes nvcc-only flags (`-t 4`, `-res-usage`, `--maxrregcount 60`, `-Xptxas -O3`, `--extra-device-vectorization`), so the RWKV decoder fails under ROCm torch as written; kernel has no warp intrinsics (`blockDim = min(C, 32)`), so a `torch.version.hip` flag branch is likely the whole fix.
- Rest of espnet/espnet2/espnet3 is plain PyTorch.

### GPU-specific dependencies
- k2 (MOAT project, upstream ROCm support merged via k2-fsa/k2#1353): OPTIONAL. `espnet2/asr/ctc.py` and the transducer model guard `import k2`; only `espnet2/asr/bayes_risk_ctc.py` imports it unguarded (BRCTC loss only). Installed by `tools/installers/install_k2.sh`, part of `make extra`, not the default. Not recorded with set-deps because it is not a hard build dependency; validating BRCTC / k2 pruned-RNNT on ROCm is a good use of the k2 port.
- warp-transducer (`warprnnt_pytorch`, HawkAaron/warp-transducer, CUDA, not a MOAT project): hard requirement of the `espnet2/asr_transducer` training loss (exits if missing); installed by `install_warp-transducer.sh` in `make extra`. espnet2's other transducer path uses torchaudio RNNT loss. Candidate follow-up: check whether warp-transducer builds for ROCm, or point the docs at the torchaudio path.
- cauchy_mult (S4 kernel from HazyResearch/state-spaces, external `.cu`): optional, with torch and pykeops fallbacks in `espnet2/asr/state_spaces/cauchy.py`.
- flash-attn (`install_flash_attn.sh`, in default `make python`): ROCm path exists in ROCm/flash-attention; installer is CUDA-wheel oriented.
- `speechlm` extra, Linux only: torchtitan, liger-kernel, flash-linear-attention, tilelang (all Triton/TileLang; each has some ROCm support upstream, unverified with espnet).
- `tools/installers/install_torch.sh` takes only a CUDA version or cpu: no ROCm option. This is the main packaging gap (ROCm wheel index, `make` with ROCm torch).
- No onnxruntime-gpu / TensorRT deployment path in tree.

### Existing AMD effort
- No ROCm/HIP/AMD PR or branch upstream (the only search hit, #6645, is unrelated). No ROCm issues. Only in-tree mention is a `ROCM_` env-prefix row in `espnet3/utils/logging_utils.py`.
- No ROCm/ or AMD-Ecosystem repo. Web search: no ESPnet-on-ROCm write-up. amd / AMD-AGI orgs not searchable with this token (SAML SSO); that part of the duplicate check stays open.

### Recommendation: fork (validate lane)
Work: add a ROCm option to `install_torch.sh` / docs, run the GPU unit tests (`test/`) and one ASR recipe stage on ROCm torch, fix the RWKV WKV JIT flags under HIP, validate the optional k2 paths against the merged k2 ROCm support, and decide warp-transducer (ROCm build vs docs pointing at torchaudio RNNT). Upstream is very active (pushed 2026-11-05, 10k stars).

## Planning (2026-11-20, linux-gfx1100 host, planner; no GPU used)

### Decisions needed from Jeff before a porter starts

Full text and evidence in `plan.md`, "Decisions needed from Jeff". Summary:

- **Q1 warp-transducer GPU.** espnet2's default RNNT loss (`warprnnt_pytorch.RNNTLoss`; ESPnetASRModel transducer, st, slu, asr_transducer task) comes from an external CUDA repo (ljn7/warp-transducer) that builds CPU-only on a ROCm host, so GPU transducer training fails on AMD. Options: (a) scope out, CPU loss only; (b) adopt warp-transducer as its own project (needs a fork) and depend on it; (c) on a ROCm torch use the in-tree Numba `RNNTLossNumba` at those call sites (reuses the Numba HIP work this port needs anyway; upstream behaviour change gated on `torch.version.hip`). **Recommended (c), with (b) as a deferral.**
- **Q2 `espnet2/speechlm`.** Its training needs torchtitan, liger-kernel, flash-linear-attention and tilelang (all third-party Triton/TileLang). Options: (a) validate it on ROCm in this port; (b) leave it out as a deferral. **Recommended (b).** This would be a partial port that leaves out an espnet module.
- **Q3 confirm scope-outs.** cauchy_mult and pykeops (external optional S4 accelerators with a torch fallback in the tree); a ROCm Docker image or CI variant; on Windows the Numba RNNT and k2 tests skip with a reason (Numba HIP is Linux-only, and k2 publishes no Windows wheels).

### What changed since intake

- **CUDA surface intake missed:** `espnet2/asr/transducer/rnnt_multi_blank/` is about 3.7k lines of **Numba CUDA** kernels (NeMo-derived RNNT and multi-blank RNNT loss: 20 `@cuda.jit`, `cuda.shfl_down_sync(0xFFFFFFFF, ...)`, shared arrays, `external_stream`, `_WARP_SIZE = 32`). The multi-blank transducer (`transducer_multi_blank_durations`) is GPU-only and needs it. Target: Numba HIP (`pip install numba-hip`, `numba.hip.pose_as_cuda()`). numba-hip's sources have no `shfl_down_sync` (`device_init.py` lists the `*_sync` intrinsics as TODO); a portable fallback (warp size 1, so the shared-memory fold runs to the end) is in plan.md R3. numba-hip is tested with numba 0.58 to 0.61.2, while espnet's Makefile installs numba unpinned.
- Intake's "espnet2's other transducer path uses torchaudio RNNT loss" no longer holds at `30a4a1e2`: there is no torchaudio RNNT call left in the tree. That is why Q1 matters.
- `get_flash_attn_compatability.is_flash_attn_supported()` is an NVIDIA device-name allow list, so flash-attn is never used on AMD even when installed. Planned fix: a ROCm branch.
- The RWKV WKV JIT flags also force `-std=c++17`, which torch's C++20 headers reject (espnet's own warp-transducer installer documents this for torch 2.14).
- The RWKV GPU tests (`test_rwkv_decoder` and the `is_rwkv` training cases) pass CPU tensors into the GPU kernel. Upstream CI never runs them. On gfx1100 that would be a memory-access fault that kills pytest. Check on CUDA first (plan.md R10).
- ROCm wheels checked 2026-11-20: `torch-2.13.0+rocm7.1/7.2` and `torch-2.14.0+rocm7.2/7.14`, `torchaudio-2.11.0+rocm7.1/7.2` (cp312, download.pytorch.org); k2 `1.24.4.dev20260710+rocm7.{1,2}.torch2.13.0` on https://k2-fsa.github.io/k2/rocm.html (no torch 2.14 build, matching the CUDA side's `k2_missing_for`).

### Records

- `set-ext-type espnet torch-extension`: pure-Python package. The one compiled piece is the JIT `cpp_extension.load` of the RWKV kernel (Strategy B), alongside Numba JIT kernels and installer changes.
- `surface.json`: generated floor (1 CUDA file) plus 16 planner-added components (JIT extension, Numba kernels, flash-attn gate, four installer items, docs, three test components, and five items pending Jeff's answers to Q1 to Q3, marked PENDING JEFF in their notes). Nothing is recorded scoped_out until Jeff rules.
- No `depends_on`. k2 is optional and published as ROCm wheels; numba-hip is AMD's own package, used as environment. Under Q1 (b), warp-transducer would become one.
