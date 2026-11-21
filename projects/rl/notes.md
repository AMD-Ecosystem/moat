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

## Review 2026-11-21 (reviewer, linux-gfx950 fleet job, MI355X)

Reviewed `moat-port` at `c76657b03bc3` (one commit on upstream `1d94cfa5c`), `git diff 1d94cfa5c...HEAD`. Verdict: **changes requested** (one finding).

### Port Correctness
1. **`setup.py:82-83` turns a working CPU-only install into a hard build failure on a default TheRock ROCm PyTorch install.** The ROCm branch returns `ROCM_HOME is not None` and checks nothing else, while the CUDA branch it sits beside checks that a usable compiler exists (`_has_nvcc()` plus the version match, and a warning plus CPU fallback when that fails). TheRock torch wheels depend on `rocm` -> `rocm-sdk-core` only. Until the user also installs `rocm[devel]` and runs `rocm-sdk init`, torch's `_find_rocm_home()` resolves `ROCM_HOME` to `_rocm_sdk_core`. That directory has a `hipcc` that cannot compile device code: `clang++: error: cannot find ROCm device library; provide its path via '--rocm-path' or '--rocm-device-lib-path' ...`. Reproduced on this host (torch 2.15.0+rocm10.2.0) by hiding the expanded `_rocm_sdk_devel` in the job venv: `ROCM_HOME` printed `.../_rocm_sdk_core`, then `PYTORCH_ROCM_ARCH=gfx950 python setup.py build_ext --inplace` at the port head **failed** (`RuntimeError: Error compiling objects for extension`), and the same command at upstream `1d94cfa5c` **succeeded** (CPU `CppExtension`, no hipcc call). The new README line sends AMD users down exactly this path ("install TorchRL from source ... compiled whenever ROCm is found"). The same thing happens with a runtime-only `/opt/rocm` (torch's Guess #4 accepts the directory even when it has no hipcc). Fix: in the HIP branch, check that the toolchain can actually compile before returning True. If it cannot, log a warning in the style of the nvcc-mismatch one ("... Set FORCE_CUDA=1 to build anyway") and fall back to the CPU build; `FORCE_CUDA=1` keeps forcing. Checking that hipcc exists is not enough, because the core `hipcc` exists. A trivial compile probe does tell the cases apart: `<ROCM_HOME>/bin/hipcc --offload-arch=<arch> -c t.hip` returned 1 with `_rocm_sdk_core` and 0 with `_rocm_sdk_devel`. setup.py already probes the host compiler with a subprocess (`_get_cxx20_compatibility_flag`). `_should_build_cuda()` runs three times per build (stale-build check, `_get_build_cuda_version`, `get_extensions`), so cache the probe. Extend `test_should_build_gpu_extension` with a case where the probe fails. Have the docs line say that TheRock wheels need `rocm[devel]` and `rocm-sdk init` for the GPU kernels.

### Points checked on request (no finding; evidence, not praise)
- CUDA branch unchanged: the diff leaves every CUDA line of `_should_build_cuda()` intact, and the new `hip` test is skipped whenever `torch.version.hip` is None (every CUDA and CPU torch). `ROCM_HOME` and `CUDA_HOME` are mutually exclusive in torch 2.11 and 2.15 (`cpp_extension.py`: each one is computed only when `torch.version.hip`, respectively `torch.version.cuda`, is set), so `build_cuda_version` stays None on ROCm. The CUDA release build of upstream `1d94cfa5c` and of the port head gives **identical** nvcc, c++ and link command lines (`diff` of the extracted commands, paths normalized; see the CUDA compile check below).
- `FORCE_CUDA` ordering: still first, as upstream had it. On ROCm torch with no `ROCM_HOME`, `FORCE_CUDA=1` builds `CUDAExtension` and fails loudly, which is what forcing means; covered by the `rocm-force-cuda` case. `packaging/pkg_helpers.bash` exports `FORCE_CUDA=1` for `rocm*`, so the wheel lane would not depend on the new branch.
- Stale-build info: an old build-info file without `hip_version` reads as None, so CUDA and CPU users see no new spurious rebuild. ROCm users get one rebuild, which is what they want, because the old build was CPU-only. Recorded file at the port head on gfx950: `"hip_version": "7.17.26434", "build_cuda_version": null, "build_cuda": true`.
- `-g` versus `-G`: correct. hipcc 10.2 rejects `-G` on gfx950 as well (`clang++: error: unsupported option '-G' for target 'x86_64-unknown-linux-gnu'`). CUDA `DEBUG=1` still passes `-O0 -G -DWITH_CUDA` to nvcc 12.8 and builds. The lesson "nvcc's -G breaks a torch extension's HIP debug build" is accurate.
- GPU segment-tree tests really run: the five tests guarded by `hasattr(ext, "CudaSumSegmentTreeFp32")` (`test/rb/test_prioritized.py:104,148,209,317`, `test/rb/test_storages.py:1242`) all PASSED on gfx950, with no skip. The other skip conditions are only `torch.cuda.is_available()`, which ROCm satisfies.

### GPU evidence at the port head (gfx950, MI355X, `HIP_VISIBLE_DEVICES=3`, `ulimit -c 0`)
Environment: the job venv with the porter's recipe, using gfx950 (`"torch[device-gfx950]==2.15.0+rocm10.2.0" "rocm[libraries,devel]==10.2.0" "rocm-sdk-device-gfx950==10.2.0"` from `https://stable.repo.amd.com/rocm/whl-next/`, then `rocm-sdk init`); the venv's `zz-base-links.pth` (ROCm 10.1) moved aside to `$FLEET_VENV/zz-base-links.pth.disabled`. Built from `git archive HEAD` with `PYTORCH_ROCM_ARCH=gfx950 MAX_JOBS=64 pip install -v --no-build-isolation --no-deps -e .` (31 s); binding exposes `Cuda{Sum,Min}SegmentTreeFp{32,64}`.

| command | result |
|---|---|
| `pytest -q -rs test/rb/test_prioritized.py` | 36 passed, 0 skipped |
| `pytest -v -rs test/rb/test_storages.py -k cuda` | 8 passed |
| `TORCHRL_SMOKE_REQUIRE_CUDA_EXT=1 pytest -q test/smoke_test.py` | 2 passed |
| `pytest -q test/test_setup.py -k "not install_no_deps"` | 13 passed |
| `pytest -q -rs test/rb -m gpu` | 53 passed, 1 skipped (ray) |
| ad-hoc CPU vs GPU parity, Sum/Min x Fp32/Fp64, sizes 16/1000/131072 (query and scan_lower_bound) | OK. No upstream test touches the Fp64 GPU trees; the sampler uses them for `dtype=torch.float64`. |

### CUDA compile check (this host's real CUDA 12.8 toolkit, no NVIDIA GPU, compile and link only)
- Separate job-local env: `$FLEET_VENV/bin/python -m venv --without-pip venv-cuda`, then `$FLEET_VENV/bin/python -m pip --python venv-cuda/bin/python install --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple "torch==2.11.0+cu128" "pybind11[global]" setuptools wheel ninja setuptools_scm numpy cmake`. That is torch 2.11.0+cu128 (`torch.version.cuda` 12.8, the TorchRL floor), and `/usr/local/cuda-12.8` (nvcc 12.8.93) matches it.
- Same environment as upstream's cu128 wheel job (`packaging/pkg_helpers.bash`), but on the auto-detect path (no `FORCE_CUDA`) so the gate's CUDA branch decides:
  ```
  export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
  export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;10.0+PTX" MAX_JOBS=64
  python -m pip wheel -v --no-build-isolation --no-deps -w cuda-wheel .     # on git archive c76657b0
  ```
  Result: **PASS** (74 s). `cuda_segment_tree.cu` compiled by `/usr/local/cuda-12.8/bin/nvcc` for sm_70 through sm_100 plus compute_100 PTX, and linked against `-lcudart -lc10_cuda -ltorch_cuda`. The `.so` in the wheel has 64 dynamic `CudaSegmentTree` symbols (float and double, Sum and Min) and NEEDS `libcudart.so.12`; `cuobjdump --list-elf` shows the 7 cubins. Build info: `"cuda_version": "12.8", "hip_version": null, "build_cuda_version": "12.8", "build_cuda": true`. The only warnings are upstream's existing ones (`-Warray-bounds` in libstdc++, `-Wcomment` in `segment_tree.h`), the same at `1d94cfa5c`.
- `python setup.py build_ext --inplace` at upstream `1d94cfa5c` and at the port head with the same environment: compile and link command lines identical.
- `DEBUG=1 TORCH_CUDA_ARCH_LIST=9.0 python setup.py build_ext --inplace`: PASS, nvcc got `-O0 -G -DWITH_CUDA`.
- `pytest -q test/test_setup.py -k "not install_no_deps"` on the CUDA torch: 13 passed.

### Input for jeff on question 0 (Triton `tl.extra.cuda.libdevice.tanh`); a person's decision, not ruled here
- **Triton versions actually pinned.** The porter guessed 3.0 to 3.5; the pins are newer. From PyPI `requires_dist`: torch 2.11.0 -> `triton==3.6.0`, 2.12.0 -> `3.7.0`, 2.13.0 -> `3.7.1`, 2.14.0/2.14.1 -> `~=3.8.0`, 2.15.0 -> `~=3.9.0`. TorchRL's floor is `torch>=2.11.0a0` (pyproject.toml); its `_TRITON_MIN_VERSION = "2.2"` comment is stale but harmless.
- **The neutral spelling exists and lowers identically on CUDA in every one of those versions.** Each Linux x86_64 cp311 wheel (3.6.0, 3.7.0, 3.7.1, 3.8.0, 3.9.0) ships `triton/language/extra/libdevice.py` with `def tanh` (a stub). Its `triton/backends/nvidia/compiler.py` has `return {"triton.language.extra.libdevice": libdevice}` with `from triton.language.extra.cuda import libdevice`, so on NVIDIA the stub is mapped to the exact CUDA libdevice module at compile time. Compiled without a GPU, `triton.compile(ASTSource(...), target=GPUTarget("cuda", 90, 32))` on a kernel calling `tl.extra.libdevice.tanh` gives PTX identical to the same kernel calling `tl.extra.cuda.libdevice.tanh` once `.loc` debug lines are removed (0 diff lines), and the `from triton.language.extra import libdevice` / `libdevice.tanh` form also compiles, in all five versions. Triton 3.0.0 also has the file, which covers anything older that someone might run.
- **One caveat for whoever writes the change.** `tl.extra.libdevice` is not a module attribute at Python level (`AttributeError` outside `@triton.jit`; `triton/language/extra/__init__.py` imports only backend subpackages). It resolves only inside jitted code. All 9 TorchRL sites are inside `@triton.jit` functions, so the sed is safe. Triton's documented spelling is the explicit `from triton.language.extra import libdevice`.
- **gfx950 reproduction.** Same scratch copy, sed applied to the 9 sites, not committed. Upstream, `pytest -m gpu test/modules/test_rnn.py` gives 38 failed / 10 passed, the same count as gfx942. With the neutral spelling it is **44 passed / 4 failed**; `test_dreamer_components.py -k "triton or Triton or gru or GRU"` gives 27 passed. On gfx950, ROCm Triton's `tl.dot` allows only `('ieee', 'bf16x3', 'bf16x6')` (CDNA4 has no xf32), so both explicit `tf32` and `tf32x3` fail in `TestRecurrentMatmulPrecision::test_{lstm,gru}_triton_precision_grad_parity`. On gfx942 only `tf32x3` failed. Any ROCm handling of explicit precision requests under option (a) has to follow the allowed set per architecture (or map explicit tf32 modes the way `auto` already maps to `ieee` on AMD), not special-case `tf32x3` alone.

## Porter round 2 (2026-11-21, linux-gfx942, fleet job, MI300X): review finding 1

Fork `moat-port` at `123797f73094` = `c76657b0` + one commit, `[ROCm] Fall back to the CPU build when hipcc cannot compile` (5 files, +122/-15).

### What the core-only layout lacks (gfx942, ROCm 10.2.0 SDK wheels)
- `_rocm_sdk_core` has `bin/hipcc`, `include/hip` and the device bitcode at `lib/llvm/amdgcn/bitcode/*.bc`, but no `amdgcn` at its root. The expanded `_rocm_sdk_devel` has `amdgcn -> lib/llvm/amdgcn` (symlink). Neither has `lib/llvm/lib/clang/24/lib/amdgcn/bitcode` (only `amdgcn-amd-amdhsa`, `x86_64-*`).
- clang finds the device libraries at `--rocm-device-lib-path`, `$HIP_DEVICE_LIB_PATH`, its resource dir `lib/clang/<N>/lib/amdgcn/bitcode`, then `<rocm candidate>/amdgcn/bitcode`. `hipcc -v` on core shows `Found HIP installation: .../_rocm_sdk_core` then `cannot find ROCm device library`. So the missing piece in core-only is `<ROCM_HOME>/amdgcn/bitcode`.
- Chosen check (no trial compile): `_has_hipcc()` = `<ROCM_HOME>/bin/hipcc` (`hipcc.exe`/`hipcc.bat` on Windows, the names torch's `_get_hipcc_path` uses) exists, and either `HIP_DEVICE_LIB_PATH` is set or a `*.bc` exists in `<ROCM_HOME>/amdgcn/bitcode` or `<ROCM_HOME>/lib/llvm/lib/clang/*/lib/amdgcn/bitcode`. Real dirs: `_rocm_sdk_devel` True, `_rocm_sdk_core` False, 0.2 ms each, so no caching needed. HIP branch: `ROCM_HOME is None` -> False silently (as `CUDA_HOME is None`); toolchain unusable -> `logger.warning(... Set FORCE_CUDA=1 to build anyway.)` and False. `FORCE_CUDA=1` still returns first. CUDA branch byte-identical (`git diff -U0 setup.py` touches only the HIP branch and adds `_has_hipcc`).
- The warning prints 3 times per `setup.py` run (`_should_build_cuda()` is called 3 times), the same as the existing nvcc-mismatch warning; left as is so the CUDA branch stays untouched.
- Windows not checked here: whether TheRock's Windows devel layout has `<ROCM_HOME>/amdgcn/bitcode` (a symlink on Linux) is unverified; if the Windows validator sees the warning with a working hipcc, the check needs the Windows path added.

### Tests (`test/test_setup.py`)
`test_should_build_gpu_extension` now builds fake ROCm trees in `tmp_path` (`_make_rocm_home`): `rocm` (`amdgcn/bitcode`), `rocm-clang-resource-dir`, `rocm-no-device-libs` (bitcode only under `lib/llvm/amdgcn`, the core layout), `rocm-no-hipcc`, `rocm-no-device-libs-force-cuda`, plus the earlier cases. New `test_rocm_without_device_libs_warns` checks the warning text and that `HIP_DEVICE_LIB_PATH` turns it into a build. The stale-build HIP test uses the `rocm` layout. `pytest -q test/test_setup.py -k "not install_no_deps"`: 18 passed; against `c76657b0`'s `setup.py`: 3 failed (`rocm-no-device-libs`, `rocm-no-hipcc`, the warning test), 15 passed. Upstream pre-commit (ufmt, flake8, pyupgrade, codespell, autoflake, ...) clean on all 5 files; `jargon.py --port rl` clean; `prose.py` clean.

### Docs
README, CONTRIBUTING.md and `docs/source/index.rst` now say the kernels are built when ROCm with `hipcc` is found, and that the ROCm SDK Python packages need `rocm[devel]` and `rocm-sdk init` first (CONTRIBUTING also names the CPU fallback and warning).

### The three cases on this host (MI300X, `HIP_VISIBLE_DEVICES=3`, `ulimit -c 0`)
Environment: fresh job venv; the porter recipe from round 1 (`torch[device-gfx942]==2.15.0+rocm10.2.0`, `rocm[libraries,devel]==10.2.0`, `rocm-sdk-device-gfx942==10.2.0` from `https://stable.repo.amd.com/rocm/whl-next/`, test deps), `zz-base-links.pth` moved to `$FLEET_VENV/zz-base-links.pth.disabled`. Before `rocm-sdk init`, torch warned that devel was not expanded and `ROCM_HOME` was `_rocm_sdk_core`.
1. **Full devel SDK** (after `rocm-sdk init`, `ROCM_HOME=_rocm_sdk_devel`): `PYTORCH_ROCM_ARCH=gfx942 pip install -v --no-build-isolation --no-deps -e projects/rl/src` succeeded, hipcc from `_rocm_sdk_devel/bin` compiled `hip_segment_tree.hip`, binding has `Cuda{Sum,Min}SegmentTreeFp{32,64}`.

   | command | result |
   |---|---|
   | `pytest -q -rs test/rb/test_prioritized.py` | 36 passed, 0 skipped |
   | `pytest -q -rs test/rb/test_storages.py -k cuda` | 8 passed |
   | `pytest -q -rs test/rb/test_prioritized.py test/rb/test_storages.py -k "cuda or prioritized"` | 44 passed |
   | `TORCHRL_SMOKE_REQUIRE_CUDA_EXT=1 pytest -q test/smoke_test.py` | 2 passed |
   | `pytest -q test/test_setup.py -k "not install_no_deps"` | 18 passed |
   | `pytest -q -rs test/rb -m gpu` | 53 passed, 1 skipped (ray) |
2. **Core-only, reproduced as the reviewer did** (`_rocm_sdk_devel` moved out of site-packages; torch's `ROCM_HOME` printed `.../_rocm_sdk_core`): `PYTORCH_ROCM_ARCH=gfx942 python setup.py build_ext --inplace` on a copy of the working tree: **rc=0**, warning `Skipping TorchRL GPU extension build because the ROCm installation at .../_rocm_sdk_core has no hipcc or no ROCm device libraries. Set FORCE_CUDA=1 to build anyway.` (x3), no hipcc call. The same command on `git archive c76657b0`: **rc=1**, `clang++: error: cannot find ROCm device library ...`, `RuntimeError: Error compiling objects for extension` (the reviewer's reproduction, on gfx942). `pip install -v --no-build-isolation --no-deps <copy>`: installed, `torchrl._torchrl` has only the CPU `{Sum,Min}SegmentTreeFp{32,64}`.
3. **`FORCE_CUDA=1` on core-only**: `FORCE_CUDA=1 PYTORCH_ROCM_ARCH=gfx942 python setup.py build_ext --inplace`: **rc=1**, `cannot find ROCm device library`, `Error compiling objects for extension`.

Devel restored and the editable install rebuilt from the fork afterwards (binding present).

### Not done
- CUDA compile check with real nvcc: this host has none. The CUDA branch and every CUDA command line are unchanged by this commit (only the HIP branch and a new helper); the reviewer's CUDA 12.8 build at `c76657b0` stands for the CUDA lines, and the `cuda`/`cuda-nvcc-mismatch`/`cpu` gate cases pass. The checkup should still run the mi355 nvcc build before presenting.
- Triton files untouched (question 0 unruled); nothing under `.github/`.
