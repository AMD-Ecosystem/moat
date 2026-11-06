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
