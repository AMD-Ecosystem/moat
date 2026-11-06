# rl notes

## Intake screen (2026-11-05, intake agent, gap-matrix seed)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: unknown / AMD path: unknown). Screened at upstream main, shallow clone in `agent_space/rl-screen`.

Context: TorchRL is part of the PyTorch org (pytorch/rl, ~3.6k stars, active: pushed 2026-11-06). jeff is a PyTorch maintainer, so an upstream PR has a normal path in.

### Licence
- `licenses.py check pytorch/rl`: MIT, tier 1. LICENSE file is MIT (Meta Platforms). No submodules, no vendored GPU code.
- `licenses.py scan-nvidia agent_space/rl-screen`: no NVIDIA proprietary licence text.

### CUDA surface
- One file: `torchrl/csrc/cuda_segment_tree.cu` (9.3 KB), added 2026-04-28 in #3677 "[Feature] Add CUDA support for prioritized replay sampling". GPU sum/min segment trees (`CudaSumSegmentTreeFp32/64`, `CudaMinSegmentTreeFp32/64`) used by `torchrl/data/replay_buffers/samplers/prioritized.py`.
- OPTIONAL: the import is in a `try/except ImportError`; without it the prioritized sampler uses the CPU segment tree from `pybind.cpp`/`segment_tree.h`. The rest of TorchRL is pure Python on torch.
- Kernel is simple (single-thread/blocked tree updates, no warp intrinsics, no cub/thrust, no hard-coded 32). torch's `CUDAExtension` hipifies it automatically under ROCm torch; expected to port mechanically.

### Why ROCm torch does not get the GPU tree today
- `setup.py::_should_build_cuda()` returns False unless `torch.version.cuda` is set AND `nvcc` exists and its version matches; on ROCm torch `torch.version.cuda` is None, so the extension is built CPU-only (only `FORCE_CUDA=1` overrides, untested on ROCm).
- `.github/workflows/build-wheels-linux.yml` and `release.yml` carry a ROCm wheel lane, `build_rocm` default false. #3603 (2026-04-08) disabled CUDA and ROCm wheels as "torchrl is CPU-only"; CUDA wheels came back after #3677, ROCm stayed off.
- Existing ROCm awareness: `torchrl/modules/tensordict_module/_rnn_precision.py` already treats `torch.version.hip` as "no TF32, use ieee" for Triton RNN kernels (tested in `test/modules/test_rnn.py`). So upstream knows about ROCm but has never built or tested the `.cu` there.

### Duplicate effort
- `gh pr list --repo pytorch/rl --state all --search "rocm OR hip OR amd"`: no ROCm/HIP enablement PR (only dependabot vllm bumps and #3603 above). Issues: none relevant.
- No ROCm/ or AMD-Ecosystem fork or separately named AMD project found (`torchrl org:ROCm` search empty). `amd`/`AMD-AGI` orgs not searchable with this token (SAML SSO), so that check is open, but nothing in upstream docs links one.

### GPU dependencies (none are MOAT projects; not set-deps)
- Optional LLM extras: `llm-vllm` (vllm), `llm-sglang` (sglang). Both have ROCm upstream (MOAT dispositions already-supported). Core needs only torch + tensordict (pure Python).

### Recommendation: take up (fork)
Validate-lane work, small: (1) teach `setup.py` to build the extension with `.cu` under ROCm torch (`torch.version.hip` + ROCM_HOME, no nvcc check); (2) build and run `test/rb/test_prioritized.py`, `test/rb/test_storages.py` and the GPU-marked test suite on ROCm torch across wave64/wave32/windows (Windows wheels are CPU-only upstream, so the windows gate is a CPU-extension plus GPU torch run); (3) upstream PR for the setup.py gate and docs, and propose turning on the ROCm wheel lane. Viable: yes.
