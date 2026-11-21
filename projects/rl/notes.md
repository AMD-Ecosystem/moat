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

## Review 2026-11-21 round 2 (reviewer, linux-gfx942 fleet job, MI300X)

Reviewed `moat-port` at `123797f73094` (`c76657b0` + `123797f7` on upstream `1d94cfa5c`), `git diff 1d94cfa5c...HEAD`. Verdict: **changes requested** (two findings, both in the new toolchain check). Round-1 finding 1 is fixed for the case it named (Linux core-only SDK without `HIP_DEVICE_LIB_PATH`), but the check models only the device libraries, which is not the whole of what the core SDK lacks, and it misses the Windows SDK layout.

### Port Correctness
1. **`setup.py:62-75` (`_has_hipcc`) treats "hipcc plus device libraries" as buildable, but a torch extension also needs the ROCm math-library headers; the `HIP_DEVICE_LIB_PATH` shortcut at `setup.py:71-72` reproduces the round-1 hard failure.** The hipified `cuda_segment_tree.cu` includes `ATen/hip/HIPContext.h`, whose `HIPContextLight.h` (torch 2.15.0+rocm10.2.0, lines 11-17) includes `hipsparse/hipsparse.h`, `hipblas/hipblas.h`, `hipblaslt/hipblaslt.h` and `hipsolver/hipsolver.h`. `_rocm_sdk_core/include` has none of them (only `CL amd_comgr aqlprofile-sdk hip hsa hsakmt rocprofiler-register roctracer`); they come with the devel tree. The porter's note "the missing piece in core-only is `<ROCM_HOME>/amdgcn/bitcode`" is therefore incomplete: the check passes on devel only because the devel tree happens to carry both the `amdgcn` symlink and the headers. Reproduced on this host with `_rocm_sdk_devel` moved out of site-packages (torch `ROCM_HOME` = `_rocm_sdk_core`), on `git archive 123797f7`:
   - `PYTORCH_ROCM_ARCH=gfx942 python setup.py build_ext --inplace`: rc=0, the warning x3, CPU build (round-1 case fixed, porter's claim confirmed).
   - `FORCE_CUDA=1 ...`: rc=1, `cannot find ROCm device library` (forcing still fails loudly, confirmed).
   - `HIP_DEVICE_LIB_PATH=<core>/lib/llvm/amdgcn/bitcode PYTORCH_ROCM_ARCH=gfx942 python setup.py build_ext --inplace` (no `FORCE_CUDA`): **rc=1**, the gate returned True with no warning, then `HIPContextLight.h:11:10: fatal error: 'hipsparse/hipsparse.h' file not found`. (A one-line kernel does compile with core hipcc plus that variable, rc=0, so the variable does fix the device-library half; the headers are the other half.)
   The same false positive applies to a system `/opt/rocm` with `hipcc` and `rocm-device-libs` but without the hip* library dev packages (the `rocm-hip-runtime-dev` set without `rocm-hip-sdk`), and to Windows users, who commonly export `HIP_DEVICE_LIB_PATH` for TheRock (GPUMD notes), on a core-only install. `test/test_setup.py:273-275` asserts the shortcut and so locks in the false positive. Fix: require the library headers as well (for example `<ROCM_HOME>/include/hipsparse/hipsparse.h` and `hipblas/hipblas.h`, the ones torch's HIP context header pulls in), so `HIP_DEVICE_LIB_PATH` substitutes only for the device-library half; extend the fake layouts (the `rocm-no-device-libs` and the env case without headers must give False, warning included), and have the warning name what is missing. The commit message's description of the check changes accordingly.

2. **On Windows the documented recipe (`rocm[devel]`, `rocm-sdk init`) will always take the CPU fallback: TheRock's Windows devel tree has neither location the check (and clang) looks in.** Listing of `rocm_sdk_devel-10.2.0-py3-none-win_amd64.whl` (`rocm_sdk_devel/_devel.tar`, from `https://stable.repo.amd.com/rocm/whl-next/`): the top level of `_rocm_sdk_devel/` is only `.info bin cmake include lib libexec share __init__.py`, with no `amdgcn` and no `llvm` symlink (the Linux devel tree has `amdgcn -> lib/llvm/amdgcn`; `rocm_sdk/_devel.py` extracts what the tarball holds and makes no such link on Windows, where directory symlinks need admin rights); `lib/llvm/lib/clang/24/lib/` holds only `windows`, no `amdgcn/bitcode`; the bitcode is only at `_rocm_sdk_devel/lib/llvm/amdgcn/bitcode/` (`hipcc.exe`/`hipcc.bat` are present). So `_has_hipcc()` returns False on Windows. That also matches what hipcc itself does: `hipcc -v` here shows clang taking its HIP installation from its own location (`lib/llvm/bin/../../..`, i.e. the SDK root) and looking for `<root>/amdgcn/bitcode`, and the Windows TheRock builds recorded in the GPUMD, Gpufit and CuRast notes needed `--rocm-device-lib-path` or `HIP_DEVICE_LIB_PATH=<rocm>/lib/llvm/amdgcn/bitcode`. The porter's round-2 note (the check "needs the Windows path added" only if the validator sees the warning with a working hipcc) undersells it: the warning is certain, and hipcc would not work there unaided either, so the Windows gate gets a CPU-only extension and every GPU segment-tree test skips. Under the standing rule that the documented build must work on every supported platform with workarounds in the build files, the fix belongs in `setup.py`: when `<ROCM_HOME>/amdgcn/bitcode` is absent but `<ROCM_HOME>/lib/llvm/amdgcn/bitcode/*.bc` exists (with the headers from finding 1), add `--rocm-device-lib-path=<that dir>` to `extra_compile_args["nvcc"]` under HIP and let the gate pass. On Linux core-only the header check still keeps it on CPU. A fake-layout test case covers the gate; the build itself is the Windows validator's to prove. Not a request to run Windows now.

### Points checked (no finding; evidence)
- CUDA branch of `_should_build_cuda()` byte-identical: the HIP block is inserted after the `FORCE_CUDA` early return and before `torch_cuda_version = ...`; no CUDA line is touched, and `_device_debug_flag()` keeps `-G` when `torch.version.hip` is None. No nvcc on this host, so no CUDA build was run in this review; the round-2 commit changes no CUDA command line, and the round-1 reviewer's CUDA 12.8 build at `c76657b0` still stands for them. The checkup's mi355 nvcc build before presenting is still owed.
- Full devel SDK layout: `<ROCM_HOME>/amdgcn -> lib/llvm/amdgcn` present (10.2.0 here and 10.1.0 in the base image venv), gate True; the editable install from the fork has `CudaSumSegmentTreeFp32`.
- `pytest -q test/test_setup.py -k "not install_no_deps"` at the head: 18 passed.
- `jargon.py --port rl` clean; `prose.py` clean on the round-2 commit message; ASCII only; no Co-Authored-By or noreply trailer; `[ROCm]` titles apply (the added lines are HIP-specific). Triton files and `.github/` untouched, as required.
- Earlier lessons in this file ("nvcc's -G breaks a torch extension's HIP debug build", "TheRock torch release wheels need the device extra") still read as accurate.

Environment restored: `_rocm_sdk_devel` moved back into site-packages, torch `ROCM_HOME` = `_rocm_sdk_devel`, `torchrl._torchrl` has `CudaSumSegmentTreeFp32`. Scratch builds were in `agent_space/rev2/` (`git archive 123797f7`); the fork working tree was not modified.

## Porter round 3 (2026-11-21, linux-gfx942, fleet job, MI300X): review round 2 findings 1 and 2

Fork `moat-port` at `845ee50d3feb` = `123797f7` + one commit, `[ROCm] Check that hipcc can preprocess the HIP headers` (5 files, +171/-96). No amend, no rebase, plain push.

### Design: why a preprocess probe and not a third path list
Two rounds of path checks were each incomplete for the same reason: they listed the files the porter thought a build needs, and the build needs whatever torch's HIP headers happen to include. Round 2 modeled the device libraries; the review showed the math-library headers (`HIPContextLight.h` includes hipsparse, hipblas, hipblaslt, hipsolver); a future torch can add another include, and system, SDK-wheel and Windows layouts each put things in different places. A list can only be kept correct by re-deriving it every time torch or the SDK changes.

Options weighed:
- (a) Extend the list with the four headers. Smallest diff, but it is the third list, it copies torch's include set into TorchRL, and it goes stale silently when torch changes.
- (b) Full compile probe (`hipcc -c` of a file including `ATen/hip/HIPContext.h`), the round-1 reviewer's suggestion. Correct by construction, but measured 5.1 s on devel per call, and `_should_build_cuda()` runs 3 times per setup.py process and setup.py runs several times per pip install.
- (c) **Preprocess probe (chosen)**: `hipcc -E` of `#include <ATen/hip/HIPContext.h>` + `#include <c10/hip/HIPGuard.h>` (the two ROCm-specific includes of the hipified kernel; `torch/extension.h` is the shared host requirement the CPU build already has), with the hipcc under `ROCM_HOME`, `include_paths(device_type="cuda")`, `COMMON_HIP_FLAGS` + `COMMON_HIPCC_FLAGS` and the `PYTORCH_ROCM_ARCH` targets, i.e. what the build will actually include and invoke. Measured on this host: devel rc=0 in 0.36-0.41 s; core rc=1 in 0.04 s with `clang++: error: cannot find ROCm device library` (the driver checks device libraries even for `-E`); core plus a device-lib path rc=1 in 0.2 s with `HIPContextLight.h:11:10: fatal error: 'hipsparse/hipsparse.h' file not found`. So preprocessing sees both halves at about a tenth of the compile cost, and its first error line is exactly "what is missing" for the warning. Not cached: 3 calls cost about 1.2 s per setup.py run, and caching complicates the tests for little.

The one layout fact that stays is finding 2's workaround, because it is a flag the build needs, not a check: `_hip_device_lib_args()` returns `--rocm-device-lib-path=<ROCM_HOME>/lib/llvm/amdgcn/bitcode` when torch is HIP, `HIP_DEVICE_LIB_PATH` is unset (a user's explicit setting wins; clang ranks the flag above the variable), `<ROCM_HOME>/amdgcn/bitcode` is absent and that directory has `*.bc`. It is appended to `extra_compile_args["nvcc"]` (after the platform block, `[]` on CUDA, so CUDA command lines do not change) and to the probe, so probe and build agree.

Torch floor check: `torch==2.11.0` (CPU wheel from download.pytorch.org, `torch/utils/cpp_extension.py`) defines `COMMON_HIP_FLAGS` (l.285), `COMMON_HIPCC_FLAGS` (l.294) and `include_paths(device_type: str = "cpu", torch_include_dirs=True)` (l.1612), so the new module-top imports cannot break a CUDA or CPU build on the floor.

### Real layouts (ROCm 10.2.0 SDK Python packages) and the gate's answer
| layout | hipcc | device bitcode | root `amdgcn` | math headers (`include/hipsparse`, `hipblas`, `hipblaslt`, `hipsolver`) | device-lib flag added | gate |
|---|---|---|---|---|---|---|
| Linux devel (`_rocm_sdk_devel`, after `rocm-sdk init`) | `bin/hipcc` | `lib/llvm/amdgcn/bitcode` | `amdgcn -> lib/llvm/amdgcn` (and `llvm -> lib/llvm`) | all four | no | **build** (probe rc=0; GPU extension built and tested below) |
| Linux core (`_rocm_sdk_core`; include is only `CL amd_comgr aqlprofile-sdk hip hsa hsakmt rocprofiler-register roctracer`) | `bin/hipcc` | `lib/llvm/amdgcn/bitcode` | absent | none | yes | **CPU + warning** (`'hipsparse/hipsparse.h' file not found`) |
| Linux core + `HIP_DEVICE_LIB_PATH=<core>/lib/llvm/amdgcn/bitcode` | same | same | absent | none | no (variable set) | **CPU + warning** (same header error) |
| Windows devel (`rocm_sdk_devel-10.2.0-py3-none-win_amd64.whl`, `rocm_sdk_devel/_devel.tar`, 9986 entries, 714 file links to `_rocm_sdk_core`) | `bin/hipcc.exe`, `bin/hipcc.bat` (links to core) | only `lib/llvm/amdgcn/bitcode` (75 `.bc`, links to core); `lib/llvm/lib/clang/24/lib/` has only `windows` | absent; top level is `.info __init__.py bin cmake include lib libexec share` | all four | yes | **build expected**; proven on Linux by the proxy below, the Windows build itself is the Windows validator's |
| Windows core (`rocm_sdk_core-10.2.0-py3-none-win_amd64.whl`, 667 entries) | `bin/hipcc.exe`, `hipcc.bat` | `lib/llvm/amdgcn/bitcode` | absent | none (`include/` is `CL amd_comgr hip ...`) | yes | CPU + warning (header error; inferred, same headers as Linux core) |
| system `/opt/rocm` | `bin/hipcc` | `amdgcn/bitcode` | itself | depends on packages | no | probe decides; a runtime-dev-only install without the hip* library dev packages gets the warning |

Listings: `pip download --no-deps --only-binary=:all: --platform win_amd64 --python-version 3.11 --index-url https://stable.repo.amd.com/rocm/whl-next/ rocm-sdk-devel==10.2.0 rocm-sdk-core==10.2.0`, then the wheels' zip and inner tar members listed with Python `zipfile`/`tarfile` (nothing extracted or run). The unit test's fake layouts (`_ROCM_LAYOUTS` in `test/test_setup.py`: `linux-devel`, `linux-core`, `windows-devel`, `system`) carry exactly the hipcc and bitcode entries above; the headers are not faked because the probe, not a file check, decides on them.

### Tests (`test/test_setup.py`)
- `test_should_build_gpu_extension`: the gate over `rocm`, `rocm-hipcc-cannot-compile`, `rocm-hipcc-cannot-compile-force-cuda`, `rocm-no-rocm-home`, `rocm-force-cuda`, `cuda`, `cuda-nvcc-mismatch`, `cpu`, with `_hipcc_error` set to `None` or the real hipsparse error line. The warning case checks that the text names `ROCM_HOME`, the missing header and `FORCE_CUDA=1`.
- `test_rocm_device_lib_path`: `get_extensions()` over the four real layouts plus `linux-core-hip-device-lib-path`; the flag is passed for `linux-core` and `windows-devel` only. Replaces the round-2 assertion that `HIP_DEVICE_LIB_PATH` alone makes the gate pass (the false positive the review named).
- `test_hipcc_probe_agrees_with_gpu_build`: with ROCm torch and an installed `torchrl._torchrl` that has `CudaSumSegmentTreeFp32`, the real probe must return None (guards against probe flags drifting from the build's). Ran (not skipped) here.
- `test_debug_build_device_debug_flag` pins `ROCM_HOME=None` so the nvcc list stays exact; the stale-build test sets the probe to succeed.
- `pytest -q test/test_setup.py -k "not install_no_deps"`: **21 passed**. Against `123797f7`'s `setup.py`: 12 failed / 9 passed (the device-lib cases `linux-core` and `windows-devel` fail on behaviour; the rest fail because `_hipcc_error` does not exist there).
- Upstream pre-commit (check-docstring-first, mixed-line-ending, end-of-file-fixer, ufmt, flake8, pyupgrade, codespell, autoflake, sphinx underline, forbid-version-file) passed on all 5 files; `jargon.py --port rl` clean; `prose.py` clean on the commit message.

### Evidence on this host (MI300X, `HIP_VISIBLE_DEVICES=3`, `ulimit -c 0`)
Environment: fresh job venv with the round-1 recipe (`torch[device-gfx942]==2.15.0+rocm10.2.0`, `rocm[libraries,devel]==10.2.0`, `rocm-sdk-device-gfx942==10.2.0` from `https://stable.repo.amd.com/rocm/whl-next/`, `rocm-sdk init`, test deps), `zz-base-links.pth` moved to `$FLEET_VENV/zz-base-links.pth.disabled`. `torch.version.hip` 7.17.26434.

1. **Full devel SDK**: `PYTORCH_ROCM_ARCH=gfx942 MAX_JOBS=96 pip install -v --no-build-isolation --no-deps -e projects/rl/src` (wrapped in `timeit.sh rl compile`) rc=0; hipcc from `_rocm_sdk_devel/bin` compiled `hip_segment_tree.hip`, no `--rocm-device-lib-path` in the log; binding has `Cuda{Sum,Min}SegmentTreeFp{32,64}`; build info `"hip_version": "7.17.26434", "build_cuda": true`.

   | command | result |
   |---|---|
   | `pytest -q -rs test/rb/test_prioritized.py` | 36 passed, 0 skipped |
   | `pytest -q -rs test/rb/test_storages.py -k cuda` | 8 passed |
   | `TORCHRL_SMOKE_REQUIRE_CUDA_EXT=1 pytest -q test/smoke_test.py` | 2 passed |
   | `pytest -q -rs test/rb -m gpu` | 53 passed, 1 skipped (ray) |
   | `pytest -q test/test_setup.py -k "not install_no_deps"` | 21 passed |
2. **Core-only** (`_rocm_sdk_devel` moved out of site-packages; torch `ROCM_HOME` printed `.../_rocm_sdk_core`), `PYTORCH_ROCM_ARCH=gfx942 python setup.py build_ext --inplace` on a copy of the working tree: **rc=0**, warning x3 `Skipping TorchRL GPU extension build because hipcc from the ROCm installation at .../_rocm_sdk_core cannot compile it: .../torch/include/ATen/hip/HIPContextLight.h:11:10: fatal error: 'hipsparse/hipsparse.h' file not found. Set FORCE_CUDA=1 to build anyway.`, CPU `CppExtension` only (0 `CudaSumSegmentTree` symbols).
3. **Core-only + `HIP_DEVICE_LIB_PATH=<core>/lib/llvm/amdgcn/bitcode`** (the case round 2 got wrong): **rc=0**, the same warning x3, CPU extension only.
4. **`FORCE_CUDA=1` on core-only**: **rc=1**, hipcc (given `--rocm-device-lib-path=<core>/lib/llvm/amdgcn/bitcode`) stops at `HIPContextLight.h:11:10: fatal error: 'hipsparse/hipsparse.h' file not found`, `1 error generated when compiling for gfx942`. Loud, as forcing should be.
5. **Windows-layout proxy** (devel restored, its `amdgcn` link moved aside, which leaves exactly the Windows devel arrangement): a plain `hipcc -c tiny.hip --offload-arch=gfx942` fails `cannot find ROCm device library`; `PYTORCH_ROCM_ARCH=gfx942 python setup.py build_ext --inplace` rc=0, no warning, hipcc got `--rocm-device-lib-path=<devel>/lib/llvm/amdgcn/bitcode`, `.so` has 28 `CudaSumSegmentTree` dynamic symbols. Link restored afterwards.

Environment restored: devel back in site-packages with its `amdgcn` link, editable install from the fork at `845ee50d` has the GPU binding.

### Not done
- No nvcc on this host. The CUDA branch of `_should_build_cuda()` is byte-identical, the only shared-path change is `extra_compile_args["nvcc"] += _hip_device_lib_args()`, which is `[]` without `torch.version.hip` (the `cuda` debug-flag test asserts the exact nvcc list). The checkup's mi355 nvcc build before presenting is still owed.
- The Windows build itself (MSVC environment, `hipcc.exe` through setuptools) is the Windows validator's; this round makes the documented recipe take the GPU path there instead of a certain CPU fallback.
- Triton files and `.github/` untouched (questions 0 and 1 still unruled).

### Correction for the final history
At PR-prep squash time the commit message must describe the end state; these statements in earlier commit bodies are now wrong:
- `c76657b0`: "the gate now builds the extension if PyTorch finds a ROCm installation (`ROCM_HOME`)". It builds only when hipcc under `ROCM_HOME` can preprocess the HIP headers (otherwise CPU plus a warning).
- `123797f7`: "`hipcc` must exist under `ROCM_HOME`, and the device libraries must be where clang looks for them (`<ROCm>/amdgcn/bitcode` or clang's resource directory), unless `HIP_DEVICE_LIB_PATH` names them". Replaced by the preprocess probe; `HIP_DEVICE_LIB_PATH` alone no longer passes, the resource-directory check is gone, and `<ROCm>/lib/llvm/amdgcn/bitcode` is passed to hipcc when `<ROCm>/amdgcn` is absent.
- `123797f7`'s Test Plan line "With `FORCE_CUDA=1` it fails in hipcc as before": it still fails, but now on the missing hipSPARSE header rather than the device libraries, because the build passes the device-lib path.
- `123797f7` title "Fall back to the CPU build when hipcc cannot compile" stays accurate.
- `845ee50d` (added by porter round 4, windows-gfx1151): "so the Windows SDK packages build the GPU extension without extra setup" was false at that commit; the probe had no `-std=` and failed on rocPRIM's C++17 `#error` on Windows. True from `d3b67c35`. Its "PyTorch's include paths and HIP flags" also omits the language standard, which the probe passes from `d3b67c35` on; the squashed message should say the probe uses BuildExtension's standard.

## Lesson for cuda-to-rocm: ask hipcc whether a torch extension can build; do not list ROCm paths
Belongs in `strategy-b-torch.md`, near "Keep compiler flags in the right command-line namespace". A `setup.py` that decides between a GPU and a CPU build under ROCm PyTorch is tempted to check files: `hipcc` exists, the device bitcode is in `<ROCm>/amdgcn/bitcode`. That misses what the build really needs, which is whatever torch's HIP headers include: `ATen/hip/HIPContext.h` pulls in the hipSPARSE, hipBLAS, hipBLASLt and hipSOLVER headers, which a core-only ROCm (the ROCm SDK `core` Python package, or a runtime-dev install without the hip* library dev packages) lacks. Ask the compiler instead: `hipcc -E` a two-line file (`#include <ATen/hip/HIPContext.h>`, `#include <c10/hip/HIPGuard.h>`) with `torch.utils.cpp_extension.include_paths(device_type="cuda")`, `COMMON_HIP_FLAGS` and `COMMON_HIPCC_FLAGS`. The driver refuses to start without the device libraries even for `-E`, and a missing header stops the preprocessor, so one call covers both in under half a second (a full `-c` is about 5 s), and its first error line tells the user what is missing. (rl)

## Lesson for cuda-to-rocm: on Windows, the ROCm SDK packages' device libraries are only in lib/llvm/amdgcn/bitcode
Belongs in `windows.md`. The Linux `rocm[devel]` tree links `<ROCm>/amdgcn -> lib/llvm/amdgcn`, and clang finds the device libraries at `<ROCm>/amdgcn/bitcode`. The Windows devel tree (`rocm_sdk_devel-*-win_amd64.whl`, 10.2.0 checked) has no such link and nothing under clang's resource directory (`lib/llvm/lib/clang/<N>/lib/` holds only `windows`), so plain hipcc fails with "cannot find ROCm device library". A build file can make the documented install work without user setup: when `<ROCm>/amdgcn/bitcode` is absent, `HIP_DEVICE_LIB_PATH` is unset and `<ROCm>/lib/llvm/amdgcn/bitcode/*.bc` exists, pass `--rocm-device-lib-path=<that directory>` to hipcc. The same layout is what the Linux core-only package has, so the flag is harmless there. (rl)

## Review 2026-11-21 round 3 (reviewer, linux-gfx942 fleet job, MI300X)

Reviewed `moat-port` at `845ee50d3feb` (`c76657b0` + `123797f7` + `845ee50d` on upstream `1d94cfa5c`), `git diff 1d94cfa5c...HEAD`. Verdict: **review passed** (no findings). Both round-2 findings are resolved, reproduced on this host below.

### Round-2 findings, re-checked
- Finding 1 (`HIP_DEVICE_LIB_PATH` false positive; headers not modelled): resolved. The gate no longer lists files; `_hipcc_error()` (`setup.py:80-119`) preprocesses `ATen/hip/HIPContext.h` and `c10/hip/HIPGuard.h` with the build's hipcc, include paths and HIP flags. Core-only (`_rocm_sdk_devel` moved out of site-packages, torch `ROCM_HOME` = `_rocm_sdk_core`), on `git archive 845ee50d`, `PYTORCH_ROCM_ARCH=gfx942 MAX_JOBS=96`:
  - `python setup.py build_ext --inplace`: rc=0, warning x3 quoting `HIPContextLight.h:11:10: fatal error: 'hipsparse/hipsparse.h' file not found`, `.so` has 0 `CudaSumSegmentTree` symbols.
  - `HIP_DEVICE_LIB_PATH=<core>/lib/llvm/amdgcn/bitcode python setup.py build_ext --inplace` (the round-2 failing case): rc=0, same warning x3, CPU extension only.
  - `FORCE_CUDA=1 ...`: rc=1 in hipcc on the same hipsparse header (loud, as forcing should be).
  - Probe directly: 0.17 s, same first error line; `hipcc -E` of a bare `hip_runtime.h` include on core without the flag gives `clang++: error: cannot find ROCm device library`, so the claim that `-E` checks the device libraries holds.
- Finding 2 (Windows SDK layout always falls back to CPU): resolved for the layout as listed in round 2. `_hip_device_lib_args()` (`setup.py:66-77`) adds `--rocm-device-lib-path=<ROCM_HOME>/lib/llvm/amdgcn/bitcode` to both the probe and `extra_compile_args["nvcc"]` (`setup.py:406-407`). Windows-layout proxy (devel with its `amdgcn` link moved aside, restored afterwards): plain `hipcc -c tiny.hip --offload-arch=gfx942` fails `cannot find ROCm device library`; `python setup.py build_ext --inplace` rc=0, no warning, the hipcc line carries the flag once, `.so` has 28 `CudaSumSegmentTree` dynamic symbols. The Windows build itself stays the Windows validator's.

### Points checked (no finding; evidence)
- Probe on the full devel SDK: `None` for `PYTORCH_ROCM_ARCH` = `gfx942` (0.39 s), `gfx942;gfx90a` (0.54 s), `gfx942 gfx1100` (0.56 s) and unset (0.42 s); `-E -o` with several offload targets works. Device-lib flag not added on devel (`[]`).
- Probe flags against the real hipcc line (from the proxy build log): the probe passes the same hipcc, `COMMON_HIP_FLAGS`, `COMMON_HIPCC_FLAGS`, the `PYTORCH_ROCM_ARCH` targets, `include_paths(device_type="cuda")` and the device-lib flag. It omits what only affects the extension's own sources (`-I.`, the Python include, `-DTORCH_EXTENSION_NAME`, `-DWITH_HIP`, `-O3`, `-fno-gpu-rdc`) and `-std=c++20`. The last does not matter with torch 2.15: the only C++20 `#error` in the torch headers is in `torch/csrc/api/include/torch/all.h`, which the two probe headers do not reach, and the probe passes at clang's default `201703L`.
- CUDA and CPU paths: the CUDA branch of `_should_build_cuda()` is byte-identical; `_hip_device_lib_args()` returns `[]` when `torch.version.hip` is None, so CUDA nvcc lists are unchanged (the `cuda` case of `test_debug_build_device_debug_flag` pins the exact list). The new module-top imports (`COMMON_HIP_FLAGS`, `COMMON_HIPCC_FLAGS`, `include_paths(device_type=...)`, `ROCM_HOME`) exist in the floor `torch==2.11.0+cpu` (`cpp_extension.py` l.251, 285, 294, 1612; wheel downloaded and read, not installed). No nvcc on this host; the mi355 nvcc build before presenting is still owed.
- Tests: `pytest -q test/test_setup.py -k "not install_no_deps"` on the head copy: 21 passed. `test/rb/test_prioritized.py` against the porter's editable install at `845ee50d` (`HIP_VISIBLE_DEVICES=3`, `ulimit -c 0`): 36 passed. The new tests follow the file's existing pattern (fresh `exec_module` of setup.py, monkeypatched private helpers, as the cxx20 tests already do); every parameter reaches an assertion. `test_hipcc_probe_agrees_with_gpu_build` is skipped on non-ROCm torch, so it never runs in upstream CI, but needs no GPU device and therefore no `gpu` marker under upstream's CLAUDE.md 7b.
- Upstream pre-commit on the six changed files (in a throwaway clone): all hooks passed. `jargon.py --port rl` clean, `prose.py` clean on the `845ee50d` message, ASCII only, no Co-Authored-By or noreply trailer, titles 54-61 characters with `[ROCm]` justified (HIP-specific added lines).
- Commit message `845ee50d`: every claim matches what I reproduced (devel needs no flag, core logs the hipsparse error with and without `HIP_DEVICE_LIB_PATH`, `FORCE_CUDA=1` fails on the same header, proxy builds with the flag, probe well under a second). The earlier bodies of `c76657b0` and `123797f7` describe superseded checks; the porter's "Correction for the final history" lists exactly those sentences and must be applied when the PR history is prepared.
- Docs: README, CONTRIBUTING.md and `docs/source/index.rst` describe the end state (full ROCm install with hipcc, `rocm[devel]` plus `rocm-sdk init`, CPU fallback with a warning, `FORCE_CUDA=1`).
- Lessons: "ask hipcc whether a torch extension can build" and "on Windows ... device libraries are only in lib/llvm/amdgcn/bitcode" are accurate as written (timings and the `-E` device-library behaviour reproduced; Windows listing taken from round 2's wheel inspection). The anchor `strategy-b-torch.md` "Keep compiler flags in the right command-line namespace" exists.

### For the Windows validator (not a finding; unverifiable here)
`windows.md` records that `hipcc.exe` does not quote what it forwards to clang. The probe writes `probe.hip` and `probe.i` under `tempfile.TemporaryDirectory()` (`%TEMP%`, inside the user profile) and passes `-I` paths unescaped. If the Windows host's profile path contains a space while the checkout and venv do not, the probe would fail on a path split and the build would take the CPU fallback with a confusing warning (`FORCE_CUDA=1` still builds). With the venv inside such a profile the real build would break too, and the fallback is then the better outcome. If the validator sees a path-split warning, the fix is to put the probe file under the build directory (or feed it on stdin) rather than `%TEMP%`.

Environment restored: `_rocm_sdk_devel` back in site-packages with its `amdgcn -> lib/llvm/amdgcn` link, torch `ROCM_HOME` = `_rocm_sdk_devel`, `torchrl._torchrl` has `CudaSumSegmentTreeFp32`. Scratch builds in `agent_space/rev3*`; the fork working tree was not modified (`git status --porcelain` empty).

## Validation 2026-11-21 (validator, linux-gfx942 fleet job, MI300X)

Head `845ee50d3feb81e21555cfa61dfa871df7eb935e` = reviewed_sha; `git -C projects/rl/src rev-parse HEAD` matches and `git status --porcelain` is empty (integrity gate clean). GPU: MI300X (gfx942), `HIP_VISIBLE_DEVICES=3`, `ulimit -c 0`, job venv torch 2.15.0+rocm10.2.0, `ROCM_HOME` = `_rocm_sdk_devel`.

Build: `PYTORCH_ROCM_ARCH=gfx942 MAX_JOBS=96 $FLEET_VENV/bin/python -m pip install -v --no-build-isolation --no-deps -e projects/rl/src` (via `timeit.sh rl compile`), rc=0, "Successfully installed torchrl-0.15.0+g845ee50d3". Log shows `_rocm_sdk_devel/bin/hipcc ... -c projects/rl/src/torchrl/csrc/hip_segment_tree.hip --offload-arch=gfx942 -DWITH_HIP -std=c++20` (the hipify step reports `cuda_segment_tree.cu -> hip_segment_tree.hip [skipped, already hipified]`, an untracked copy from earlier builds that is regenerated by every build). `torchrl._torchrl` exposes `CudaMinSegmentTreeFp32/Fp64` and `CudaSumSegmentTreeFp32/Fp64`.

| command (all via `timeit.sh rl test`) | result |
|---|---|
| `pytest -q -rs test/rb/test_prioritized.py` | 36 passed, 0 skipped |
| `pytest -q -rs test/rb/test_storages.py -k cuda` | 8 passed, 162 deselected, 0 skipped |
| `TORCHRL_SMOKE_REQUIRE_CUDA_EXT=1 pytest -q test/smoke_test.py` | 2 passed |
| `pytest -q test/test_setup.py -k "not install_no_deps"` | 21 passed, 2 deselected |
| `pytest -q -rs test/rb -m gpu` | 53 passed, 1 skipped (`test_rb_distributed.py:413` ray required) |
| CPU regression: `HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= pytest -q test/rb` | 4500 passed, 1062 skipped (CUDA-needed and optional-dependency skips), 0 failed |

No skip in the GPU runs says the extension was not built. Triton RNN tests were not run (outside the gate).

CUDA gate: `cuda-not-validated: no nvcc on this host (no /opt/conda/envs/cuda-12.8) and the job venv carries ROCm torch only, which blocks a torch CUDAExtension compile`. The CUDA branch of `_should_build_cuda()` is unchanged and `_hip_device_lib_args()` is `[]` without `torch.version.hip` (see round 3 and review notes); the mi355 nvcc build before presenting remains owed.

Finish checks: `python3 utils/jargon.py --port rl` clean; ROCm build documented in README.md, CONTRIBUTING.md and docs/source/index.rst (in the branch diff).

## CUDA compile check (nvcc) at 845ee50d

Validator role's CUDA compile check only (2026-11-21, linux-gfx950 fleet job, mi355 host: real CUDA 12.8 toolkit `/usr/local/cuda-12.8`, nvcc 12.8.93, no NVIDIA GPU, so compile and link only). Fork `moat-port` cloned fresh into the job directory, `HEAD` = `845ee50d3feb81e21555cfa61dfa871df7eb935e`; upstream base `1d94cfa5c` from the same clone. Each build ran on its own `git archive <rev>` tree. No state, platform record or fork source changed. Method as in the first review's check at `c76657b0`. Result: **PASS**, and the CUDA command lines are identical to upstream's.

Environment (job-local, nothing shared modified):
```
$FLEET_VENV/bin/python -m venv --without-pip venv-cuda
$FLEET_VENV/bin/python -m pip --python venv-cuda/bin/python install --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple "torch==2.11.0+cu128" "pybind11[global]" setuptools wheel ninja setuptools_scm numpy cmake
$FLEET_VENV/bin/python -m pip --python venv-cuda/bin/python install pip pytest pytest-timeout pytest-instafail pytest-mock "pyvers>=0.2.3" "hoptorch>=0.1.4" "tensordict>=0.15.2,<0.16.0" pyyaml scipy psutil pytest-asyncio pytest-benchmark
export PATH=$PWD/venv-cuda/bin:/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8
unset FORCE_CUDA ROCM_HOME ROCM_PATH HIP_PATH; ulimit -c 0
```
torch 2.11.0+cu128 (`torch.version.cuda` 12.8, `torch.version.hip` None; the TorchRL floor), tensordict 0.15.6, Python 3.11.17. The fleet guard refused `venv-cuda/bin/python -m pip install ...` as a shared-environment write (venv-cuda is inside the job directory), so the test dependencies went in through the same `$FLEET_VENV/bin/python -m pip --python venv-cuda/bin/python` form as the torch install.

1. **Wheel, as upstream's cu128 wheel job builds it but on the auto-detect path (no `FORCE_CUDA`), so the gate's CUDA branch decides**, on `git archive 845ee50d`:
   ```
   export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;10.0+PTX" MAX_JOBS=64
   python -m pip wheel -v --no-build-isolation --no-deps -w ../cuda-wheel .
   ```
   rc=0, 76 s. `cuda_segment_tree.cu` compiled by `/usr/local/cuda-12.8/bin/nvcc` with `-gencode` for sm_70, 75, 80, 86, 89, 90, 100 and compute_100, `-O3 -DWITH_CUDA`; the link line carries `-lcudart -lc10_cuda -ltorch_cuda`. No "Skipping TorchRL" warning. The wheel's `_torchrl.cpython-311-x86_64-linux-gnu.so` has 64 dynamic `CudaSegmentTree` symbols and NEEDS `libcudart.so.12` and `libc10_cuda.so`; `cuobjdump --list-elf` lists the 7 cubins and `--list-ptx` the sm_100 PTX. Build info: `{"torch_version": "2.11.0+cu128", "python_version": "3.11", "cuda_version": "12.8", "hip_version": null, "build_cuda_version": "12.8", "force_cuda": false, "build_cuda": true}`. Warnings are upstream's existing ones only (`-Wcomment` in `segment_tree.h`, libstdc++ `-Warray-bounds` on `__builtin_memmove`, setuptools_scm and `version.py` packaging notes).
2. **Command lines against upstream**: `python setup.py build_ext --inplace` with the same environment and arch list on `git archive 1d94cfa5c` and on `git archive 845ee50d`, both rc=0 (72 s each). The nvcc line, both `c++` compile lines (`pybind.cpp`, `utils.cpp`) and the `x86_64-linux-gnu-g++ -shared` link line, extracted from the logs with the tree path normalized: **identical** (`diff` empty). The generated `build/temp.linux-x86_64-cpython-311/build.ninja` (normalized) is identical too, and both logs have the same 8 warning lines. The only difference in the build info is the new `"hip_version": null` key. So `_hip_device_lib_args()` adds nothing to the nvcc list and the `_hipcc_error()` probe never runs on CUDA torch, as the round-3 review argued from the source.
3. **`DEBUG=1`**: `DEBUG=1 TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=64 python setup.py build_ext --inplace` on both trees: rc=0 each, nvcc got `-O0 -G -DWITH_CUDA`, 112 `CudaSegmentTree` symbols in each (unstripped), compile and link command lines identical between upstream and head.
4. **`test/test_setup.py` on the CUDA torch**:
   - `python -m pytest -q -rs test/test_setup.py -k "not install_no_deps"` on `git archive 845ee50d`: **20 passed, 1 skipped** (`test_hipcc_probe_agrees_with_gpu_build`, "needs ROCm PyTorch", by design), 2 deselected. (The ROCm runs report 21 passed because that test runs there.)
   - Whole file from a git checkout (`git clone` of the fork clone, detached at `845ee50d`, so `test_install_no_deps_has_nonzero_version` does not skip as "not a git checkout"), `TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=64 python -m pytest -q -rs test/test_setup.py`: **22 passed, 1 skipped** (same ROCm-only test), 56 s. Both `install_no_deps` cases (editable and wheel) do a `pip install --no-deps --no-build-isolation` into `venv-cuda` and build the CUDA extension with nvcc: the installed `torchrl 0.15.0+g845ee50d` has 64 `CudaSegmentTree` symbols, build info `"build_cuda": true, "build_cuda_version": "12.8"`. The test uninstalls `setuptools_scm` from `venv-cuda` (its own behaviour, job-local env).

Owed item from the round-3 review and the validation ("the mi355 nvcc build before presenting") is satisfied at `845ee50d`. Scratch trees and logs are in the job directory (`nvcc/`, container-local).
## Validation 2026-11-21 (validator, linux-gfx1100 fleet job, Radeon Pro W7800, wave32)

Head `845ee50d3feb81e21555cfa61dfa871df7eb935e` = reviewed_sha; fresh clone of `moat-port`, `git -C projects/rl/src rev-parse HEAD` matches and `git status --porcelain` is empty (integrity gate clean). GPU: AMD Radeon Pro W7800 48GB, gfx1100, `warp_size` 32, dispatcher pin `HIP_VISIBLE_DEVICES=2` only, `ulimit -c 0`, `timeout` around every GPU command.

Environment: job venv, `zz-base-links.pth` (base-image ROCm 10.1) moved to `$FLEET_VENV/zz-base-links.pth.disabled`; then
```
pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple \
    "torch[device-gfx1100]==2.15.0+rocm10.2.0" "rocm[libraries,devel]==10.2.0" "rocm-sdk-device-gfx1100==10.2.0"
rocm-sdk init
pip install "pybind11[global]" "pyvers>=0.2.3" "hoptorch>=0.1.4" "tensordict>=0.15.2,<0.16.0" pyyaml scipy psutil \
    pytest-instafail pytest-mock pytest-asyncio pytest-benchmark pytest-timeout gymnasium setuptools_scm ninja
```
torch 2.15.0+rocm10.2.0 (pulls `amd-torch-device-gfx1100` and `amd-torch-device-gfx110x`), `torch.version.hip` 7.17.26434, triton 3.9.0+rocm10.2.0, `ROCM_HOME` = `_rocm_sdk_devel`.

Build: `PYTORCH_ROCM_ARCH=gfx1100 MAX_JOBS=16 $FLEET_VENV/bin/python -m pip install -v --no-build-isolation --no-deps -e projects/rl/src` (via `timeit.sh rl compile`), rc=0, "Successfully installed torchrl-0.15.0+g845ee50d". Log shows `_rocm_sdk_devel/bin/hipcc ... -c .../torchrl/csrc/hip_segment_tree.hip ... -DWITH_HIP ... --offload-arch=gfx1100 -fno-gpu-rdc -std=c++20`. `torchrl._torchrl` exposes `CudaMinSegmentTreeFp32/Fp64` and `CudaSumSegmentTreeFp32/Fp64` (plus the CPU `{Min,Sum}SegmentTreeFp{32,64}`).

| command (from the fork root, all via `timeit.sh rl test`) | result |
|---|---|
| `pytest -q -rs test/rb/test_prioritized.py` | 36 passed, 0 skipped |
| `pytest -q -rs test/rb/test_storages.py -k cuda` | 8 passed, 162 deselected, 0 skipped |
| `TORCHRL_SMOKE_REQUIRE_CUDA_EXT=1 pytest -q test/smoke_test.py` | 2 passed |
| `pytest -q test/test_setup.py -k "not install_no_deps"` | 21 passed, 2 deselected |
| `pytest -q -rs test/rb -m gpu` | 53 passed, 1 skipped (`test_rb_distributed.py:413` ray required) |
| CPU regression: `HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= pytest -q test/rb` | 4500 passed, 1062 skipped, 0 failed |

Identical counts to the gfx942 validation; no skip in the GPU runs says the extension was not built.

Triton RNN tests (outside this port; upstream behaviour awaiting jeff's ruling, see porter round 1 question 0), reported separately:
- `pytest -q -m gpu test/modules/test_rnn.py`: 38 failed, 10 passed, 225 deselected. Errors: `RuntimeError: Implicit conversion of CUDA __nv_tanhf device function has been dropped` (`tl.extra.cuda.libdevice.tanh`) and, for the two `test_{lstm,gru}_triton_precision_grad_parity[tf32x3]` cases, `AssertionError: input_precision must be one of ...`.
- `pytest -q test/modules/test_dreamer_components.py -k "triton or Triton or gru or GRU"`: 8 failed, 19 passed, 230 deselected; all 8 the `__nv_tanhf` error.
Same counts as the porter's MI300X baseline, so wave32 adds no failure.

CUDA gate: `cuda-not-validated: no nvcc on this host (no /opt/conda/envs, no /usr/local/cuda) and the job venv carries ROCm torch only`; unchanged from the gfx942 validation, the mi355 nvcc build remains owed.

Finish checks: `python3 utils/jargon.py --port rl` clean (after `git -C projects/rl/src fetch origin main:main` in the fresh clone); ROCm build documented in README.md (lines 470-472), CONTRIBUTING.md and docs/source/index.rst. Nothing pushed to the fork.

## Validation 2026-11-20 (validator, windows-gfx1151, Radeon 8060S APU, wave32): FAILED, GPU build gate falls back to CPU on the documented install

Head `845ee50d3feb81e21555cfa61dfa871df7eb935e` (fresh clone of `moat-port` in the worktree, `git status --porcelain` empty). Environment: existing `agent_space/torch_venv` torch 2.12.0+rocm7.14.0a20260519 (hip 7.13), rocm-sdk-devel 7.14.0a20260519, reached through a small overlay venv `agent_space/rl_venv` (a `.pth` to torch_venv's site-packages, plus pybind11, pyvers, hoptorch, tensordict 0.15.6, pytest plugins, gymnasium, setuptools_scm; 31 MB, pip cache `agent_space/pip_cache_rl`). Env script `agent_space/rl_env.sh` = `win_torch_env.sh` with `HIP_DEVICE_LIB_PATH` unset (verified empty); it keeps `ROCM_PATH`/`HIP_PATH` (torch on Windows only finds ROCM_HOME through them), `CC=CXX=clang-cl`, `DISTUTILS_USE_SDK=1`, `PYTORCH_ROCM_ARCH=gfx1151`. Commands via `timeit.sh rl compile`: `python -m pip install -v --no-build-isolation --no-deps -e projects/rl/src`.

**Build 1, `HIP_DEVICE_LIB_PATH` unset, no FORCE_CUDA (the documented path): rc=0 but CPU-only.** The log prints, for every build_ext call:
`Skipping TorchRL GPU extension build because hipcc from the ROCm installation at D:/Develop/moat/agent_space/torch_venv/Lib/site-packages/_rocm_sdk_devel cannot compile it: ...\_rocm_sdk_devel\include\rocprim\config.hpp:69:6: error: "rocPRIM requires at least C++17". Set FORCE_CUDA=1 to build anyway.`
**Build 2, `HIP_DEVICE_LIB_PATH` set by hand: identical message and CPU-only result.** So the device-library handling is not the cause.

Cause: `_hipcc_error()` in setup.py runs `hipcc -E probe.hip` with COMMON_HIP_FLAGS + COMMON_HIPCC_FLAGS but no `-std=`. On Linux hipcc defaults to a C++ standard rocPRIM accepts; on Windows the clang driver default is below C++17 and rocPRIM's `#error` fires (it is included through ATen/hip/HIPContext.h). The real compile line (build 3) carries `-std=c++20` from BuildExtension. Checked by hand with the same flags: `hipcc -E probe.hip ... --rocm-device-lib-path=<bitcode>` rc=1 without `-std`, rc=0 with `-std=c++17` and with `-std=c++20`. The port's own test `test/test_setup.py::test_hipcc_probe_agrees_with_gpu_build` fails here with the same assertion (`_hipcc_error()` is not None), so the check designed to catch this does.

**Build 3, `FORCE_CUDA=1` and `HIP_DEVICE_LIB_PATH` unset: GPU build OK.** hipcc line: `hipcc.exe -std=c++20 ... -c torchrl/csrc/hip_segment_tree.hip --offload-arch=gfx1151 --rocm-device-lib-path=...\_rocm_sdk_devel\lib\llvm\amdgcn\bitcode`, so `_hip_device_lib_args()` works on Windows as claimed and nothing else in the build fails. `torchrl._torchrl` exposes `CudaMinSegmentTreeFp32/Fp64` and `CudaSumSegmentTreeFp32/Fp64`.

Tests on that forced GPU build (`ulimit -c 0`, single runs, via `timeit.sh rl test`, `-rs`):

| command | result |
|---|---|
| `pytest -q -rs test/rb/test_prioritized.py` | 36 passed, 0 skipped |
| `pytest -q -rs test/rb/test_storages.py -k cuda` | 8 passed, 162 deselected, 0 skipped |
| `TORCHRL_SMOKE_REQUIRE_CUDA_EXT=1 pytest -q test/smoke_test.py` | 2 passed |
| `pytest -q -rs test/test_setup.py -k "not install_no_deps"` | 18 passed, 1 failed (`test_hipcc_probe_agrees_with_gpu_build`, the probe bug above), 2 skipped (`test_setup.py:303` "checks the Unix compiler flags", Windows-irrelevant), 2 deselected |
| `pytest -q -rs test/rb -m gpu` | 53 passed, 1 skipped (`test_rb_distributed.py:413` ray required) |

Not run: Triton RNN tests (out of scope), full CPU `test/rb` regression (verdict already decided; Linux CPU regression stands at 4500 passed).

What a porter must change: make the probe in `_hipcc_error()` pass the same C++ standard the build uses (at least on win32: `-std=c++20`, or derive it from the BuildExtension flag), so the documented Windows install builds the GPU extension without FORCE_CUDA. After that, rebuild on Windows with no `FORCE_CUDA` and no `HIP_DEVICE_LIB_PATH`; the tests above already pass on this APU. Linux validators did not see this because the Linux hipcc default standard satisfies rocPRIM.

CUDA gate: not rerun (host has no nvcc; passed at this head on linux-gfx950). Any fix touches setup.py, so the nvcc compile check owes a rerun at the new head.

## Lesson for cuda-to-rocm: a hipcc preprocess probe must carry the build's -std on Windows
Belongs in `strategy-b-torch.md` next to the "ask hipcc" probe lesson. A `hipcc -E` capability probe that omits the language standard works on Linux (hipcc defaults to a modern standard) and fails on Windows with rocPRIM's `#error "rocPRIM requires at least C++17"`, which sends a documented install down the CPU-only path with a plausible-looking reason. Pass the same `-std=` the real compile gets. Derive it the way `torch.utils.cpp_extension.BuildExtension` does for the installed torch rather than hard-coding one: torch 2.11 adds `-std=c++17` (`cpp_flag = cpp_flag_prefix + 'c++17'`, and `cuda_cflags = ['-std=c++17']` on the Windows ninja path), torch 2.12 and later `-std=c++20`. (rl)

## Porter round 4 (2026-11-20, windows-gfx1151, Radeon 8060S APU): Windows validation failure, probe C++ standard

Fork `moat-port` at `d3b67c350ca2` = `845ee50d` + one commit, `[ROCm] Preprocess the hipcc check with the build's C++ standard` (`setup.py` +6/-3, `test/test_setup.py` +36). No amend, no rebase, plain push. CUDA branch of `_should_build_cuda()` byte-identical (`_hipcc_error()` is reached only from the ROCm branch); `.github/` and Triton files untouched.

### Design
`_hipcc_error()` now passes `"-std=c++20" if _torch_extension_requires_cxx20() else "-std=c++17"` to `hipcc -E`. How BuildExtension picks the standard, read from the sources:
- torch 2.11.0 (`v2.11.0` `torch/utils/cpp_extension.py`, fetched read-only with `gh api ... contents/...?ref=v2.11.0`): `append_std17_if_no_std_present` uses `cpp_flag = cpp_flag_prefix + 'c++17'` (l.770, Unix non-ninja path: applied to the hipcc flags too); Windows ninja HIP `cuda_cflags = ['-std=c++17']` (l.1101); Unix ninja HIP `cuda_flags + ['-std=c++17']` (l.2880).
- torch 2.12.0 (installed here) and `main`: the same three places with `c++20`.
- Windows non-ninja HIP path (`win_hip_flags`) passes no `-std` at all in either version; that is torch's own gap and would make the build fail loudly, not silently fall back; the default (ninja) path is what pip uses.
So the standard depends only on the torch release, and setup.py already detects it for the host flags with `_torch_extension_requires_cxx20()` (regex on `BuildExtension.build_extensions` source for `cpp_flag = ... 'c++20'`, fallback `torch.__config__.show()`). Reusing it keeps one detection in the file. Not added to `extra_compile_args["nvcc"]`, so the build lines and `test_build_extension_owns_cxx_standard_or_uses_alias` are unchanged. Other build-only flags (`-O3`, `-DWITH_HIP`, `-DTORCH_EXTENSION_NAME`, `-fms-runtime-lib=dll` which torch adds on Windows when it writes the ninja file) do not decide whether the two probe headers preprocess.

Why Linux passed: Linux hipcc's default standard is C++17 (clang's GNU default), which rocPRIM accepts, so probe and build agreed by accident (round-3 review measured `__cplusplus` 201703L in the probe).

### Test
`test_hipcc_probe_uses_build_cxx_standard[True--std=c++20|False--std=c++17]`: fake `ROCM_HOME/bin/hipcc{,.exe}`, `subprocess.run` replaced by a hipcc stand-in that returns rocPRIM's `#error` unless a `-std=` is present (the Windows behaviour), asserts `_hipcc_error() is None` and that exactly the build's standard was passed. Runs on any platform, no GPU, no ROCm. Against `845ee50d`'s setup.py (`git stash push setup.py`): 2 failed; with the fix: 2 passed.

### Evidence on this host
Env: `source agent_space/rl_env.sh` (torch_venv torch 2.12.0+rocm7.14.0a20260519, hip 7.13.26190, rocm-sdk-devel 7.14.0a20260519, `PYTORCH_ROCM_ARCH=gfx1151`), then `unset FORCE_CUDA HIP_DEVICE_LIB_PATH` (verified empty). `ulimit -c 0`, single runs.

- Probe directly (setup.py loaded with `importlib`): `_hipcc_error()` -> None in 0.8 s, `_torch_extension_requires_cxx20()` True, `_should_build_cuda()` True.
- Documented install: `utils/timeit.sh rl compile -- $TORCH_PY -m pip install -v --no-build-isolation --no-deps -e projects/rl/src` rc=0 (log `agent_space/rllogs/build4.log`), no "Skipping TorchRL GPU extension build" line; `build/.torchrl_build_info.json` `"build_cuda": true, "force_cuda": false`. hipcc line: `hipcc.exe -std=c++20 -Xcompiler -D__HIP_PLATFORM_AMD__=1 ... -c ...\torchrl\csrc\hip_segment_tree.hip -o ...hip_segment_tree.obj ... -O3 -DWITH_HIP --rocm-device-lib-path=...\_rocm_sdk_devel\lib\llvm\amdgcn\bitcode -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_torchrl --offload-arch=gfx1151 -fno-gpu-rdc`.
- `torchrl._torchrl`: `CudaMinSegmentTreeFp32`, `CudaMinSegmentTreeFp64`, `CudaSumSegmentTreeFp32`, `CudaSumSegmentTreeFp64`.

| command (from the worktree root via `utils/timeit.sh rl test --`, `pytest -q -rs --rootdir=projects/rl/src`) | result |
|---|---|
| `test/rb/test_prioritized.py` | 36 passed, 0 skipped |
| `test/rb/test_storages.py -k cuda` | 8 passed, 162 deselected, 0 skipped |
| `TORCHRL_SMOKE_REQUIRE_CUDA_EXT=1 ... test/smoke_test.py` | 2 passed |
| `test/test_setup.py -k "not install_no_deps"` | 21 passed, 2 skipped (`test_setup.py:339` "checks the Unix compiler flags"), 2 deselected; `test_hipcc_probe_agrees_with_gpu_build` now passes |

Negative case, without a new environment: the torch_venv already carries the core-only package `_rocm_sdk_core` (include = `CL amd_comgr hip hsa hsakmt`, `bin/hipcc.exe`, `lib/llvm/amdgcn/bitcode`, no root `amdgcn`). With `ROCM_PATH=HIP_PATH=<site-packages>/_rocm_sdk_core` torch's `ROCM_HOME` is core, `_hip_device_lib_args()` returns the core bitcode flag, and `_should_build_cuda()` returns False with `Skipping TorchRL GPU extension build because hipcc from the ROCm installation at .../_rocm_sdk_core cannot compile it: ...\torch\include\ATen/hip\HIPContextLight.h:11:10: fatal error: 'hipsparse/hipsparse.h' file not found. Set FORCE_CUDA=1 to build anyway.` Gate decision only; the CPU-fallback build itself was shown on Linux in round 3 and is the Linux revalidators' to repeat.

Lint: black 22.3.0 and flake8 6.1.0 (the pre-commit pins, installed with `pip --target agent_space/tmp/lint`) clean on both files; `jargon.py --port rl` clean (needed a local `main` branch in the fork clone: `git branch main origin/main`); `prose.py` clean on the message.

### Owed
- nvcc compile check at `d3b67c35` (the change touches setup.py only in the ROCm-only probe, but the rule is to rerun it before presenting).
- Re-review of the delta (head moved at `review-passed`), then revalidation on linux-gfx942, linux-gfx1100 and this host.

## Review 2026-11-21 round 4 (reviewer, linux-gfx950 fleet job, MI355X)

Reviewed the delta `845ee50d..d3b67c35` (`git diff 845ee50d...d3b67c35`: `setup.py` +6/-3, `test/test_setup.py` +36), and the port where the delta touches it (`_hipcc_error()`, `_should_build_cuda()`, `get_extensions()`). Verdict: **review passed**, no finding.

### Points checked (no finding; evidence)
- **The standard depends only on the torch release, on every BuildExtension path that reaches hipcc.** Read `torch/utils/cpp_extension.py` at `v2.11.0`, `v2.12.0`, `v2.13.0`, `v2.14.0`, `v2.15.0` and `origin/main` (pytorch checkout in the job worktree, `git show <tag>:...`). Unix non-ninja (`unix_wrap_single_compile`) and Unix ninja (`unix_wrap_ninja_compile`) both run `append_std17_if_no_std_present` on the hipcc flags, which appends `cpp_flag = cpp_flag_prefix + 'c++17'` in 2.11 and `'c++20'` in 2.12 through main, and only when no `-std=` is present. Windows ninja (`win_wrap_ninja_compile`) puts a literal `cuda_cflags = ['-std=c++17']` (2.11) / `['-std=c++20']` (2.12+) first on the hipcc line, which is what the Windows validator's log shows (`hipcc.exe -std=c++20 ...`). Windows non-ninja passes none (below). TorchRL's `extra_compile_args["nvcc"]` never carries a `-std=` (`setup.py:341-410`; `_get_cxx20_compatibility_flag()` goes only to `cxx`, and `test_build_extension_owns_cxx_standard_or_uses_alias` pins that), so torch's flag is never suppressed. Env-borne hipcc flags (`HIPCC_COMPILE_FLAGS_APPEND`) reach probe and build alike, because the probe runs the same hipcc in the same environment.
- **Detection.** `_torch_extension_requires_cxx20()` is upstream's own function (present at `1d94cfa5c`); its regex matches the `cpp_flag = ... 'c++20'` line in 2.12 through main and not 2.11's `'c++17'`. On this host (torch 2.15.0+rocm10.2.0) it returns True. Its fallback, `"-std=c++20" in torch.__config__.show()`, is False on torch 2.15, so a torch installed without `.py` sources would make the probe use `-std=c++17` while the build uses `-std=c++20`. Harmless today: torch 2.15's HIP headers preprocess under `-std=c++17` here (rc 0). Not this delta's concern.
- **Linux behaviour measured.** Probe headers plus `long v = __cplusplus;` through `_rocm_sdk_devel/bin/hipcc -E` with the probe's flags: no `-std` -> rc 0, `201703L` (the Linux default the porter cited); `-std=c++14` -> rc 1, `rocprim/config.hpp:78:6: error: "rocPRIM requires at least C++17"` (the Windows failure, reproduced by the standard alone); `-std=c++17` -> `201703L`; `-std=c++20` -> `202002L`.
- **The flag goes only to the probe.** `setup.py build_ext --inplace` with `PYTORCH_ROCM_ARCH=gfx950` on `git archive 845ee50d` and `git archive d3b67c35`: both rc=0, no "Skipping" warning; the hipcc, both `c++` and the link command lines, paths normalized, are **identical** (`diff` empty), each carrying `-std=c++20` from torch.
- **The new test would have caught the fault on Linux.** `pytest test/test_setup.py -k probe_uses_build` on this host: 2 passed at `d3b67c35`; with `845ee50d`'s `setup.py` swapped in, 2 failed. It needs no GPU and no ROCm, so CPU and CUDA CI run it too. `pytest -q -rs test/test_setup.py -k "not install_no_deps"` on ROCm torch: 22 passed, 1 skipped (`test_hipcc_probe_agrees_with_gpu_build`, extension not built yet in this tree; the validator runs it).
- **CUDA branch byte-identical.** The delta's only code line is inside `_hipcc_error()`, which is reached only from the `torch.version.hip` branch of `_should_build_cuda()`; no CUDA line changed. (The nvcc compile check at this head is recorded separately below.)
- **Commit.** Title 63 characters, `[ROCm]` prefix fits (the change is ROCm-only), body has rationale, "Authored with Claude.", Test Plan in fenced blocks, no `Co-Authored-By`. `jargon.py --port rl` clean.
- **Lesson "a hipcc preprocess probe must carry the build's -std on Windows"** is accurate as written. One correction to the porter round 4 Design paragraph (notes, not upstream text): the 2.11 line 2880 it cites (`cuda_flags + ['-std=c++17']`) is in `_write_ninja_file_to_build_library`, the JIT `load()` path, not BuildExtension's Unix ninja path; BuildExtension's Unix ninja path gets the standard from `append_std17_if_no_std_present` on `cuda_post_cflags` instead. Same value, so the conclusion stands.

### Input for jeff: torch's Windows non-ninja HIP path (a view, not a finding against this delta)
The porter's observation is correct. In every version read (2.11 through main) `win_wrap_single_compile` builds the hipcc line from `win_hip_flags(cflags)` = `COMMON_HIPCC_FLAGS + COMMON_HIP_FLAGS + cflags + _get_rocm_arch_flags(cflags)` with no `-std=`, while the CUDA branch beside it appends `-std=c++17`/`c++20` explicitly and the ninja path adds it for HIP too. BuildExtension takes that path whenever ninja is not found (`is_ninja_available()` false -> "Falling back to using the slow distutils backend"). On Windows hipcc's default standard is below C++17, so any torch HIP extension that includes ATen's HIP headers fails to compile there; this is a torch bug worth reporting upstream (a one-line `-std=` in the HIP branch of `win_wrap_single_compile`, mirroring the CUDA branch).
What it means for this port: the ninja path is what TorchRL's `pyproject.toml` build requirements give (`ninja` is listed), but the documented `python setup.py develop` and `pip install --no-build-isolation` use whatever environment the user has, and torch does not depend on ninja. On Windows with the devel SDK and no ninja, the probe (now passing `-std`) says hipcc can build, and the compile then fails with rocPRIM's `#error`: a loud install failure where upstream builds CPU-only. Before this delta the same case fell back to CPU, but only through the same accident that broke the ninja case. I judge a loud failure with a clear compiler message acceptable for that corner and not worth blocking on; the alternatives are (a) have the probe omit `-std` when `sys.platform == "win32"` and ninja is unavailable, so probe and build agree and the user gets the CPU fallback plus the rocPRIM warning (two lines, keeps the build lines untouched), or (b) add the standard to `extra_compile_args["nvcc"]` on win32 HIP, which changes the build lines and the `owns_cxx_standard` test. Either is jeff's call; neither is verifiable on this Linux host.
