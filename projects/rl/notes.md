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

## Planning (2026-11-20, planner, linux-gfx942 host, no GPU used)

**Decision needed from jeff before the porter starts: should the upstream PR also turn on TorchRL's ROCm wheel lane?** `build-wheels-linux.yml` `with-rocm` and `release.yml` `build_rocm` default to disabled (off since pytorch/rl#3603). Options:
- (a) flip both defaults in the same PR;
- (b) PR changes `setup.py`, docs and tests only, and its body proposes the lane flip (recommended);
- (c) a separate CI PR after the source PR merges.

The porter must not touch `.github/` until ruled. This is not a partial port: the whole CUDA surface (one `.cu`) is in scope under every option, Windows included.

- Planned at upstream `1d94cfa5c35f` (2026-11-19). `ext_type` = torch-extension. Strategy: B, torch build-time hipify.
- Core change: `setup.py::_should_build_cuda()` returns False on ROCm torch (it needs `torch.version.cuda` and a matching nvcc). Add a `torch.version.hip` + `ROCM_HOME` branch; add the hip version to the stale-build info; drop `-G` for the HIP device compiler in `DEBUG=1`. Add unit tests in `test/test_setup.py` and a docs line.
- Kernel sweep: 4 kernels, 256-thread blocks, no warp intrinsics, shared memory, atomics, libraries or uninitialised allocations (`torch::full`). No source change expected in `torchrl/csrc`.
- Validation trap: every GPU segment-tree test `pytest.skip`s when the binding is absent. Assert `hasattr(torchrl._torchrl, "CudaSumSegmentTreeFp32")` and record `-rs` skips.
- `packaging/pkg_helpers.bash` already exports `FORCE_CUDA=1` for `rocm*`. So the FORCE_CUDA route on ROCm exists but is untested; the auto-detect makes it unnecessary.
- surface.json: 11 components; `ci:rocm_wheel_lane` stays unaccounted pending the ruling.

## Porter round (2026-11-20, linux-gfx942, fleet job, MI300X)

### Open questions for jeff (none blocks the reviewer)
0. **Scope addition found by the porter: TorchRL's Triton kernels fail on ROCm.** The plan treated the Triton RNN path as pure Python outside the port, but `torchrl/modules/tensordict_module/_rnn_triton.py` (6 sites) and `torchrl/modules/models/_dreamer_v3_block_gru_triton.py` (3 sites) call `tl.extra.cuda.libdevice.tanh`, which ROCm Triton 3.9 rejects at compile time: `RuntimeError: Implicit conversion of CUDA __nv_tanhf device function has been dropped; please, update your source program to use triton.language.extra.<op> ...`. Evidence on this MI300X (torch 2.15.0+rocm10.2.0, triton 3.9.0+rocm10.2.0): `pytest -m gpu test/modules/test_rnn.py` is 38 failed / 10 passed as upstream; with the backend-neutral spelling `tl.extra.libdevice.tanh` (a 9-line sed, tried in the working tree and reverted, not committed) it is 46 passed / 2 failed; `pytest test/modules/test_dreamer_components.py -k "triton or Triton or gru or GRU"` goes from 8 failed / 19 passed to 27 passed. The 2 remaining failures are `TestRecurrentMatmulPrecision.test_lstm_triton_precision_grad_parity[tf32x3]` and its GRU twin: the test forces `tf32x3`, and ROCm Triton's `tl.dot` allows only `bf16x3, bf16x6, ieee, tf32` (`AssertionError: input_precision must be one of ...`); `_rnn_precision.py` already maps `auto` to `ieee` on AMD, so only an explicit `tf32x3` request hits it (needs a skip or an error message on ROCm). Options: (a) add the tanh change plus a ROCm handling of explicit `tf32x3` to this port (one more commit, same PR); (b) keep this PR to the segment tree and send the Triton fix as a second PR; (c) leave the Triton path alone. Check before choosing: does `tl.extra.libdevice` exist on Triton 3.0 through 3.5, the versions torch 2.11 to 2.14 pin? TorchRL's own floor comment says 2.2, but torch >= 2.11 never ships that old a Triton. `surface.json` lists `triton:rnn_libdevice_tanh` as owed until ruled. Raw logs from this job are container-local; the commands above reproduce them.
1. **ROCm wheel lane (the planner's question, still unruled).** Should the upstream PR also flip `with-rocm` in `.github/workflows/build-wheels-linux.yml` and `build_rocm` in `release.yml` to enabled? Options as the planner wrote them: (a) flip both in the same PR; (b) PR changes `setup.py`, docs and tests only and its body proposes the flip (planner's recommendation); (c) a separate CI PR after the source PR merges. Nothing under `.github/` was touched; `surface.json` lists `ci:rocm_wheel_lane` as owed pending the ruling. Evidence relevant to the ruling: with this port, a wheel built on a ROCm torch now carries the GPU segment trees with no workflow change other than the flag (the gate auto-detects ROCm; `packaging/pkg_helpers.bash` already exports `FORCE_CUDA=1` for `rocm*`).
2. **Fork default branch.** Your standing rule is that every AMD-Ecosystem fork has `moat-port` as its GitHub default branch. That is a GitHub settings write, which a fleet worker may not make. To apply it: `gh repo edit AMD-Ecosystem/rl --default-branch moat-port`. The recorded `fork_default_branch` stays `main`.
3. **Fork mirror is behind upstream.** Fork `main` is at `a711ef99d`; upstream `main` (and the planned base) is `1d94cfa5c`, a fast-forward. `moat-port` was cut from upstream `1d94cfa5c`, so a fork review PR `moat-port -> main` would show the upstream commits in between until the mirror is fast-forwarded (`git push origin upstream/main:main` from the fork clone is the plain fast-forward; I did not push to the mirror).
4. **PR title tag (for publication, not now).** TorchRL's agent rules (`CLAUDE.md` section 13) want a `[Tag]` from their canonical set (`[Feature]`, `[BugFix]`, ...) on the PR title; the fork commit title carries `[ROCm]` per our convention. The upstream PR title probably wants `[Feature]` (for example `[Feature] Build the GPU segment tree extension with ROCm PyTorch`). Their rules also say the PR body must not carry test commands or pass counts.

### Environment (job-local, nothing shared modified)
- TheRock ROCm 10.2.0 release wheels from `https://stable.repo.amd.com/rocm/whl-next/` (the release index; `nightly.repo.amd.com/rocm/whl-next/` has only 10.2.0a nightlies, `repo.amd.com/rocm/whl/<family>/` stops at 7.13, `whl-multi-arch` at 7.14.1), installed into `$FLEET_VENV`:
  ```
  pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple \
      "torch[device-gfx942]==2.15.0+rocm10.2.0" "rocm[libraries,devel]==10.2.0" "rocm-sdk-device-gfx942==10.2.0"
  rocm-sdk init        # expands _rocm_sdk_devel; torch's ROCM_HOME then points at it
  pip install "pybind11[global]" "pyvers>=0.2.3" "hoptorch>=0.1.4" "tensordict>=0.15.2,<0.16.0" pyyaml scipy psutil \
      pytest-instafail pytest-mock pytest-asyncio pytest-benchmark pytest-timeout gymnasium
  ```
  torch 2.15.0+rocm10.2.0, `torch.version.hip` 7.17.26434, triton 3.9.0+rocm10.2.0, tensordict 0.15.6, Python 3.11.17. The job venv's `zz-base-links.pth` (base-image ROCm 10.1 links) was moved aside to `$FLEET_VENV/zz-base-links.pth.disabled` so only 10.2 is visible.
- Without the `[device-gfx942]` extra (`amd-torch-device-gfx942`), torch imports and sees the GPU but every kernel fails with `hipErrorInvalidKernelFile` ("CUDA error: invalid kernel file").
- GPU: `HIP_VISIBLE_DEVICES=3` (dispatcher pin), MI300X gfx942; `ulimit -c 0` for every GPU run.

### Fork change (`moat-port` at `c76657b03bc3`, one commit on upstream `1d94cfa5c`)
`[ROCm] Build the GPU segment tree extension with ROCm PyTorch` -- 6 files, +130/-2:
- `setup.py`: imports `ROCM_HOME`; `_should_build_cuda()` keeps `FORCE_CUDA=1` first, then for `torch.version.hip` returns `ROCM_HOME is not None` (mirrors torch's own `IS_HIP_EXTENSION`); the CUDA branch is byte-identical. `_get_build_cuda_version()` needed no change (no nvcc and `torch.version.cuda` None give None). Stale-build info gains `hip_version` (compared, logged, recorded). New `_device_debug_flag()` gives `-g` under HIP, `-G` otherwise, used in both `DEBUG=1` branches.
- `.gitignore`: `torchrl/csrc/*.hip` (torch writes `hip_segment_tree.hip` next to the `.cu` in the source tree).
- `test/test_setup.py`: 3 tests (9 cases): the gate over rocm / rocm-without-ROCM_HOME / rocm+FORCE_CUDA / cuda / cuda-nvcc-mismatch / cpu; the `DEBUG=1` device flag for rocm and cuda (Unix); a HIP version change removes the stale `.so` and records the new version. 8 of the 9 new cases fail against the unmodified `setup.py` (the `cuda` debug-flag case passes on both, as it should).
- Docs: `CONTRIBUTING.md` (after `python setup.py develop`), `docs/source/index.rst` (Installation, after the clone-and-develop block), `README.md` (after the CUDA-wheel paragraph: AMD GPUs build from source against ROCm PyTorch).
- `torchrl/csrc/*`: no change. Hipify output differs from the `.cu` only in the includes (`ATen/hip/HIPContext.h`, `c10/hip/HIPGuard.h`) and `hipLaunchKernelGGL` launches. Hipify also rewrites the device-compile `-DWITH_CUDA` to `-DWITH_HIP`; harmless, because only `pybind.cpp` (host compiler, keeps `-DWITH_CUDA`) reads it. hipcc got `-std=c++20`, the same as the host compiler (plan risk 5).
- Upstream pre-commit (ufmt, flake8, pyupgrade, codespell, autoflake, ...) clean on the changed files; `jargon.py --port rl` clean; `prose.py` clean on the commit message.

### Build
```
PYTORCH_ROCM_ARCH=gfx942 MAX_JOBS=96 pip install -v --no-build-isolation --no-deps -e projects/rl/src     # 44 s
python -c "import torchrl._torchrl as e; assert hasattr(e, 'CudaSumSegmentTreeFp32')"                      # Cuda{Sum,Min}SegmentTreeFp{32,64} present
```
- `DEBUG=1 PYTORCH_ROCM_ARCH=gfx942 python setup.py build_ext --inplace` (scratch copy of the tree): builds, hipcc gets `-O0 -g`, binding present. With upstream's `-G`, hipcc fails: `clang++: error: unsupported option '-G' for target 'x86_64-unknown-linux-gnu'` (reproduced on a one-line kernel), so every HIP debug build would have broken (plan risk 2 confirmed).
- CUDA compile check (no NVIDIA GPU here, compile only): a separate job-local venv with `torch==2.14.1+cu130` and pip `nvidia-cuda-nvcc==13.0.*`, `nvidia-cuda-crt==13.0.*`, `nvidia-cuda-cccl==13.0.*`, `nvidia-nvvm==13.0.*` (the default nvvm 13.5 emits PTX 9.5 that ptxas 13.0 rejects), `CUDA_HOME=<site-packages>/nvidia/cu13` (plus `lib64 -> lib` and `libcudart.so -> libcudart.so.13` symlinks, environment only), `TORCH_CUDA_ARCH_LIST=9.0 python setup.py build_ext --inplace` on `git archive HEAD`: the gate picked nvcc 13.0.88 (matches `torch.version.cuda` 13.0), `cuda_segment_tree.cu` compiled and linked, `.so` carries the `CudaSegmentTree*` symbols. Not the mi355 host's system nvcc; the reviewer or checkup may still want that run before presenting.
- CPU-only fallback: same CUDA torch venv with no `CUDA_HOME` and `FORCE_CUDA=0`: `CppExtension` build, no nvcc invoked, no `CudaSegmentTree` symbols.

### GPU tests (MI300X, at `c76657b0`'s tree)
| command | result |
|---|---|
| `pytest -v test/rb/test_prioritized.py -rs` | 36 passed, 0 skipped |
| `pytest -v test/rb/test_storages.py -k cuda -rs` | 8 passed, 162 deselected |
| `TORCHRL_SMOKE_REQUIRE_CUDA_EXT=1 pytest -v test/smoke_test.py` | 2 passed |
| `pytest -v test/test_setup.py -k "not install_no_deps"` | 13 passed |
| `pytest test/rb -m gpu -rs` | 53 passed, 1 skipped (ray) |
| `pytest test/rb -q -rs --timeout 900` | 4554 passed, 1008 skipped, 0 failed (265 s) |
| `HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= pytest test/rb -q -rs` (CPU) | 4500 passed, 1062 skipped, 0 failed |

The five pass-criteria tests all PASSED: `test_cuda_segment_tree_parity`, `test_cuda_prioritized_replay_buffer_samples_on_cuda`, `test_tensordict_prioritized_replay_buffer_memmap_storage_cuda_sampler`, `test_cuda_prioritized_replay_buffer_weight_matches_cpu_formula`, `TestSharedStorageInit::test_prioritized_memmap_cuda_sampler_after_multiprocess_writes`. No skip anywhere says "not built with CUDA segment tree support"; the full-suite skips are optional deps (zstandard, ray, h5py, torchvision, gym), MPS, and unsupported-combination skips.

Benchmark (evidence only): `pytest benchmarks/test_replaybuffer_benchmark.py -k 'sampler_sample_scale or sample_mixed_devices' --benchmark-only`, mean us: sample at 1M entries GPU 337 vs CPU 10688 (32x); at 10M GPU 305 vs CPU 19926 (65x); cuda storage + cuda sampler 391, memmap CPU storage + cuda sampler 2422, cuda storage + CPU sampler 12564.

### Not done here
- Windows gate: the planner's risk 6 (TheRock `hipcc.exe`, setuptools traps) is untested; no Windows-specific change was needed or made. `platform:windows` stays owed in `surface.json`.
- wave32 gate: no wavefront-dependent code (planner sweep), not run here.

## Lesson for cuda-to-rocm: nvcc's -G breaks a torch extension's HIP debug build
Belongs in `strategy-b-torch.md`, beside the entry on the shared `"nvcc"` flag list. Many `setup.py` files add `-G` (nvcc device debug info) to `extra_compile_args["nvcc"]` under a `DEBUG=1` switch. Under ROCm torch that list goes to hipcc, and clang rejects it: `clang++: error: unsupported option '-G' for target 'x86_64-unknown-linux-gnu'`. Release builds never take the path, so the break stays hidden until someone debugs. Use `-g` when `torch.version.hip` is set and keep `-G` for CUDA. (rl)

## Lesson for cuda-to-rocm: TheRock torch release wheels need the device extra
Belongs in `linux-host-setup.md`. The ROCm 10.x release wheels (10.0 to 10.2 at the time of writing) are on `https://stable.repo.amd.com/rocm/whl-next/`; `nightly.repo.amd.com/rocm/whl-next/` has only the nightlies and `repo.amd.com/rocm/whl/` stops at 7.x. There, `torch==2.15.0+rocm10.2.0` ships its GPU code objects in separate `amd-torch-device-<gfx>` packages. Install `torch[device-<gfx>]` (for example `device-gfx942`); without the extra, torch imports and enumerates the GPU, but the first kernel fails with `hipErrorInvalidKernelFile` ("invalid kernel file"). Run `rocm-sdk init` afterwards so torch's `ROCM_HOME` resolves to the expanded `_rocm_sdk_devel` (math-library headers) and not to `_rocm_sdk_core`. (rl)
