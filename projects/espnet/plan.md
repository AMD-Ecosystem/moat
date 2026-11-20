# espnet port plan

## Decisions needed from Jeff before a porter starts

Three questions. The plan below is written for the recommended answer to each; a different answer changes only the items named.

**Q1. warp-transducer (espnet2's default RNNT loss) has no ROCm GPU build.** `warprnnt_pytorch.RNNTLoss` is imported unguarded as THE transducer loss by `ESPnetASRModel` (when `transducer_multi_blank_durations` is empty), `espnet2/st/espnet_model.py`, `espnet2/slu/espnet_model.py`, and the standalone `espnet2/asr_transducer` task (which `exit(1)`s without it). It lives outside the espnet tree (`ljn7/warp-transducer`, a fork of `HawkAaron/warp-transducer`, Apache-2.0) and is built by `tools/installers/install_warp-transducer.sh` in `make extra`. Its CMake finds no CUDA on a ROCm host and silently builds CPU-only, so on an AMD GPU every transducer training run fails at the loss on GPU tensors. Intake's note that "espnet2's other transducer path uses torchaudio RNNT loss" no longer holds: there is no torchaudio RNNT call left in the tree at this sha.
- (a) Scope out GPU transducer training through warp-transducer; document that on ROCm the transducer loss must run on CPU. Smallest, but leaves the most common transducer recipe broken on AMD GPUs.
- (b) Adopt warp-transducer as its own project (a person creates the `AMD-Ecosystem` fork; a separate intake), port its CMake/CUDA (`src/rnnt_entrypoint.cu`, `include/detail/{gpu_rnnt,gpu_rnnt_kernel,reduce}.h`: warp-ctc-style warp reductions with a hard-coded warp size of 32), and point `install_warp-transducer.sh` at a ROCm build. Most complete; the delivery vehicle is a fork upstream of espnet that espnet itself only clones.
- (c) In espnet, when running on a ROCm torch, use the in-tree Numba RNNT loss (`espnet2/asr/transducer/rnnt_multi_blank/rnnt_multi_blank.py:RNNTLossNumba`, the NeMo-derived standard RNNT with the same `blank`/`fastemit_lambda`/`reduction` knobs) in place of `warprnnt_pytorch` for those four call sites. This port has to make the Numba kernels run on ROCm anyway (multi-blank transducer, item 2 below), so (c) reuses that work; it is an upstream behaviour change gated on `torch.version.hip`, which maintainers must accept.
- **Recommended: (c), with (b) recorded as a deferral** for anyone who needs bit-compatibility with warp-transducer on AMD. This plan's items 2 and 6 assume (c); under (a) item 6 disappears; under (b) item 6 becomes `depends_on: [warp-transducer]` plus an installer change.

**Q2. `espnet2/speechlm` training depends on a Triton/TileLang stack outside the tree.** The `speechlm` extra pulls torchtitan 0.2.2, liger-kernel, flash-linear-attention and tilelang (Linux only); `espnet2/speechlm/model/speechlm/lm/loss.py` imports liger-kernel at module level. Each has some upstream ROCm support, none verified with espnet. espnet's own speechlm code is plain torch.
- (a) In scope: install the ROCm builds of those four packages and run `test/espnet2/speechlm` (including the `-k cuda` FSDP test) on the GPU. Fixes, if any, land in those packages, not espnet.
- (b) Out of scope for this port, recorded as a deferral: the port validates every espnet module except speechlm, and the PR says so.
- **Recommended: (b).** The work is in four third-party packages, and a speechlm failure would hold the whole port while the core toolkit is ready. The deferral keeps it visible.

**Q3. Confirm the proposed scope-outs** (each also in `surface.json` with its reason; a "no" moves it back in):
- `cauchy_mult` (HazyResearch state-spaces CUDA extension, cloned and built by `install_cauchy_mult.sh`) and `pykeops`: optional S4 accelerators outside the tree; `espnet2/asr/state_spaces/s4.py` falls back to plain torch. Not ported.
- A ROCm Docker image (`docker/build.sh` builds CUDA images only) and a ROCm CI image variant: publishing images and adding GPU CI is a maintainer's choice; not offered in this PR.
- Windows: Numba HIP is Linux-only and k2 publishes no Windows wheels (and `install_k2.sh` exits on non-Linux), so on the windows gate the Numba RNNT paths (multi-blank, and the Q1 (c) fallback) and k2 paths cannot run. The plan makes those tests skip with a stated reason on a ROCm torch without Numba HIP, and validates everything else on Windows.

## Project

- name: espnet
- upstream: https://github.com/espnet/espnet (default branch `master`); MOAT fork `AMD-Ecosystem/espnet`, port branch `moat-port`
- planned against upstream `30a4a1e27f4dbd3bb2f9256a86d80b795990bcfb` (2026-11-20)
- licence Apache-2.0 (intake)

## Existing AMD support

None upstream or in AMD orgs (intake: no ROCm/HIP PR, branch or issue; no ROCm/ or AMD-Ecosystem repo). Decision: **improvable, port here.** The relevant authoritative AMD efforts are dependencies, not competing ports:
- k2: MOAT ported it; ROCm support merged upstream (k2-fsa/k2#1353) and k2 now publishes ROCm wheels at https://k2-fsa.github.io/k2/rocm.html (`1.24.4.dev20260710+rocm7.{1,2}.torch2.13.0`, also torch 2.10 to 2.12.1). Authoritative.
- Numba HIP (`numba.hip`, AMD's ROCm target for Numba, `ROCm/hip-python/packages/numba-hip`, `pip install numba-hip`): authoritative, experimental. It offers `numba.hip.pose_as_cuda()` so unchanged `from numba import cuda` code runs on it. MOAT recorded NVIDIA/numba-cuda `ported-elsewhere` on this basis.
- flash-attention: ROCm support upstream (CK backend for Instinct, Triton backend for Radeon); MOAT has a `flash-attention` project.

No performance-kernel rewrite question: the only compiled kernel is a 120-line RWKV recurrence; a mechanical port is right.

Delivery vehicle: an ordinary upstream PR to espnet/espnet. Upstream merges platform work into the tree (it carries macOS and Windows branches in its installers); nothing suggests it links forks instead.

## Build classification

**torch-extension** (recorded with `set-ext-type`), but most of the port is not a build at all:
- espnet is a pure-Python package (`pyproject.toml`, setuptools; no `setup.py` extension, no CMake). Nearly all GPU work is plain PyTorch, which runs on a ROCm torch as is.
- The one compiled extension is JIT-built at run time: `espnet2/asr_transducer/decoder/modules/rwkv/attention.py:132 load_wkv_kernel()` calls `torch.utils.cpp_extension.load(sources=[cuda/wkv_op.cpp, cuda/wkv_cuda.cu], extra_cuda_cflags=[...])`. That is Strategy B: torch hipifies at load time.
- A second GPU path is JIT Python: Numba CUDA kernels in `espnet2/asr/transducer/rnnt_multi_blank/` (about 3.7k lines). The surface scan cannot see these (no `.cu` files); they are added to `surface.json` by hand.
- The installer (`tools/Makefile` + `tools/installers/*.sh`) is where espnet chooses CUDA, and it has no ROCm path at all (below).

## Port strategy

Strategy B for the RWKV extension (keep CUDA spelling; fix only what torch's hipify and hipcc cannot take, under `torch.version.hip`). Numba HIP in `pose_as_cuda` mode for the Numba kernels, with source edits only where Numba HIP lacks a `numba.cuda` API or where a 32-lane assumption breaks. Installer and docs changes so `make` on a ROCm host installs a ROCm torch and ROCm k2 instead of silently falling back to CPU torch. Every change is gated on ROCm (`torch.version.hip` in Python, a ROCm version variable in shell); the CUDA and CPU paths stay byte-for-byte the same in behaviour.

## CUDA surface inventory

| # | surface | where | ROCm mapping / risk |
|---|---|---|---|
| 1 | RWKV WKV kernel: `kernel_forward`, `kernel_backward` (`__global__`, 2 `<<<>>>` launches on the legacy default stream), plain float math (`exp`, `max`), per-thread arrays `q[Tmax]`, `r[Tmax]` | `espnet2/asr_transducer/decoder/modules/rwkv/cuda/wkv_cuda.cu`, `wkv_op.cpp` (pybind + `TORCH_LIBRARY`) | hipify handles all of it. No warp intrinsics; `blockDim = min(C, 32)` is a block size, not a warp assumption, and is valid on both wavefront widths. Breakers are the nvcc-only flags (risk R1) and `-std=c++17` (R2). |
| 2 | Numba CUDA kernels: 20 `@cuda.jit` (kernels and device functions), `cuda.shared.array`, `cuda.syncthreads` (10), `cuda.shfl_down_sync(0xFFFFFFFF, ...)` (1), `cuda.external_stream(torch stream)`, `cuda.select_device`, `cuda.as_cuda_array` (torch tensors via `__cuda_array_interface__`), `_WARP_SIZE = 32`, `_THREADS_PER_BLOCK = 32` | `espnet2/asr/transducer/rnnt_multi_blank/{rnnt.py, utils/global_constants.py, utils/rnnt_helper.py, utils/cuda_utils/{reduce,gpu_rnnt,gpu_rnnt_kernel}.py}` | Numba HIP via `numba.hip.pose_as_cuda()`. `external_stream`, `select_device`, `as_cuda_array` exist in numba-hip's `api.py`. `shfl_down_sync` is NOT found in numba-hip's sources (its `device_init.py` lists the `*_sync` intrinsics as a TODO): risk R3. Workspace is `torch.zeros`, so no uninitialised-memory read. |
| 3 | `torch.cuda.*`, `.cuda()`, `device="cuda"`, cuDNN flags, `nccl` backend, `autocast("cuda")` (about 80 call sites) | espnet2, espnet3 | Work unchanged on a ROCm torch (cuDNN flags reach MIOpen; `nccl` is RCCL). |
| 4 | flash-attn gate by NVIDIA device name | `espnet2/torch_utils/get_flash_attn_compatability.py` (called by the transformer, conformer, branchformer, e-branchformer encoders and transformer decoder) | Returns `None` on every AMD GPU, so flash-attn is never used on ROCm even when installed. Add a ROCm branch (item 4). |
| 5 | Installer CUDA selection | `tools/Makefile` (CUDA version from `nvcc`; no nvcc means `CPU_ONLY=0` and a CPU torch), `tools/installers/install_torch.sh` (cuXXX or cpu index only), `install_k2.sh` (CUDA/CPU wheel index only), `install_warp-transducer.sh` + Makefile `warp-transducer.done` (requires `CUDA_HOME` when not CPU-only), `tools/check_install.py` (prints `torch.version.cuda`, which is `None` on ROCm) | Add a ROCm path: torch `+rocmX.Y` from https://download.pytorch.org/whl/rocmX.Y (checked: `torch-2.13.0+rocm7.1/7.2`, `torch-2.14.0+rocm7.2/7.14`, `torchaudio-2.11.0+rocm7.1/7.2` exist for cp312), k2 from `rocm.html`, numba-hip. |
| 6 | Optional external CUDA packages | warp-transducer (Q1), cauchy_mult + pykeops (Q3), speechlm stack (Q2), flash-attn (`install_flash_attn.sh`: `pip install flash-attn`, whose setup.py builds the ROCm backend on a ROCm torch) | per the questions above |

No textures, no CUDA libraries (cuBLAS etc.) called directly, no driver API, no NVRTC/PTX (Numba HIP JITs through the ROCm compiler itself).

## Risk list

- **R1 nvcc-only JIT flags.** `-t 4`, `-res-usage`, `--maxrregcount 60`, `-Xptxas -O3`, `--extra-device-vectorization` and `--use_fast_math` are nvcc options; hipcc/clang rejects them. Under `torch.version.hip` pass a HIP list (`-O3`, `-ffast-math` if fast math is kept, `-DTmax=...`); keep the CUDA list untouched.
- **R2 `-std=c++17` against torch's C++20 headers.** The same list forces C++17; espnet's own `install_warp-transducer.sh` documents that torch 2.14 headers need C++20 (`requires`). On the HIP branch omit `-std` and let torch choose. Do not change the CUDA branch unless it is shown broken there too (if it is, that is a shared fix without the `[ROCm]` prefix, recorded separately).
- **R3 `shfl_down_sync` under Numba HIP.** `reduce.py:106` calls `cuda.shfl_down_sync(0xFFFFFFFF, x, offset)`. If Numba HIP has no such intrinsic (likely, see inventory row 2), or applies HIP's 64-bit mask rule, the kernel fails to compile. A fix that holds on both wavefront widths without any shuffle: on ROCm, set the reduction's warp size to 1 so the existing shared-memory folding loop (`while offset >= warp_size`) runs down to offset 1 and the shuffle loop never executes. The result in thread 0 is the same; the cost is a few more `syncthreads` on a 128-wide block. If Numba HIP does provide a shuffle, the 32-wide reduction stays correct on wave64 too (the shared-memory stage folds to 32 elements and the shuffles only read lanes 0..31), but the mask must then be the full 64-bit one. Prove whichever is chosen on wave32 and wave64.
- **R4 Numba / NumPy version.** `tools/Makefile` installs `numba` unpinned (`pip install -U numba`); numba-hip is tested with numba 0.58 to 0.61.2. The ROCm path must pin a compatible numba (and check espnet's NumPy floor against it). `pose_as_cuda()` must run before any `from numba import cuda` in the package: put it in `espnet2/asr/transducer/rnnt_multi_blank/__init__.py` (or a tiny helper imported first) under `torch.version.hip`.
- **R5 Numba HIP is experimental.** It documents no device printf, no fp16 and no cooperative groups; none are used here. Treat any other `numba.cuda` API gap as a port item, not an environment failure, and record it.
- **R6 hipify mirror location for a JIT build.** `cpp_extension.load` hipifies the sources at load time and can write the mirror beside the sources (here a sibling `rwkv/hip/` directory, because the source directory is named `cuda`) or into the build directory, depending on the torch version's `_jit_compile`. The porter must check on the real ROCm torch where it lands, add the mirror path to `.gitignore` if it is in the tree (strategy-b `.gitignore` rule; `git status --porcelain` empty after a run), and note that a read-only site-packages install would fail if it writes beside the sources.
- **R7 per-thread arrays.** `kernel_backward` keeps `float q[Tmax], r[Tmax]` per thread (`Tmax` = context size, 512 by default): 4 KiB of scratch per thread. Correct on AMD; slow, and the same on CUDA. Not changed.
- **R8 test gating uses `torch.cuda.is_available()`.** On a ROCm torch it is True, so GPU-only tests (WKV, multi-blank) run, which is what validation needs. On Windows, or any ROCm torch without numba-hip, the Numba tests must skip with a reason rather than fail (Q3).
- **R9 ROCm wheel matrix.** k2 ROCm wheels exist for torch 2.13.0 but not 2.14.0 (the CUDA side has the same gap; `k2_missing_for="2.14.0"`). Validate on torch 2.13.0+rocm7.2 (the Makefile default `TH_VERSION`). torchaudio 2.11.0+rocm is used with torch 2.13 exactly as upstream does on CUDA.
- **R10 the WKV tests hand CPU tensors to the GPU kernel.** `test_decoder.py::test_rwkv_decoder` and the `is_rwkv` cases of `test_espnet_transducer_model.py::test_model_training` build the model and inputs on the CPU and only skip when no GPU exists; `WKVLinearAttention.forward` passes `data_ptr()` of whatever it gets to `cuda_forward`, which launches with host pointers. Upstream CI never runs these (no GPU). On a gfx1100 with XNACK off, a kernel dereferencing pageable host memory is a memory-access fault that aborts the whole pytest process, not one failed test. Check the same tests on the CUDA host first: if they fail there too, the fix (move model and inputs to the GPU in the test, and optionally a device check in `WKVLinearAttention.forward`) is a shared test fix with no `[ROCm]` prefix, in its own commit. Run these tests in their own pytest process until settled.
- **R11 JIT build time against `execution_timeout`.** `test_rwkv_decoder` carries `@pytest.mark.execution_timeout(20)`, and the first call builds the extension. If hipcc's first build exceeds that, warm the torch extensions cache (`TORCH_EXTENSIONS_DIR`) in a separate step before the run; do not raise the timeout.

## File-by-file change list

1. `espnet2/asr_transducer/decoder/modules/rwkv/attention.py`: in `load_wkv_kernel`, pick the HIP flag list when `torch.version.hip` (R1, R2). The ROCm error message stays "CUDA is currently a requirement..." only for genuinely GPU-less hosts.
2. `espnet2/asr/transducer/rnnt_multi_blank/`: call `numba.hip.pose_as_cuda()` on a ROCm torch before `numba.cuda` is imported (R4); fix `reduce.py`'s shuffle per R3; fix any further Numba HIP API gap found at run time. Gate everything on ROCm.
3. `.gitignore`: the WKV hipify mirror, if it lands in the tree (R6).
4. `espnet2/torch_utils/get_flash_attn_compatability.py`: under `torch.version.hip`, return True when `flash_attn` imports and the device arch is one ROCm flash-attention supports (CK: gfx90a, gfx942, gfx950; Triton backend: gfx11, gfx12). Keep the NVIDIA name list as is.
5. Installer:
   - `tools/Makefile`: detect ROCm when there is no nvcc (`hipconfig --version` or `$ROCM_PATH/.info/version`; accept `ROCM_VERSION=X.Y` from the command line as `CUDA_VERSION` is), pass it to `install_torch.sh`, install numba-hip with a pinned compatible numba, and drop the `CUDA_HOME` requirement from `warp-transducer.done` on ROCm.
   - `tools/installers/install_torch.sh`: accept `rocmX.Y` in the `<cuda_version>` slot (spelling to match the script's style) and install from `https://download.pytorch.org/whl/rocmX.Y` with the existing per-torch-version table, falling back to the newest ROCm the torch version publishes.
   - `tools/installers/install_k2.sh`: when `torch.version.hip` is set, install `k2==<ver>+rocm<X.Y>.torch<ver>` from `https://k2-fsa.github.io/k2/rocm.html`.
   - `tools/check_install.py`: report `torch.version.hip` on ROCm instead of an empty CUDA line.
6. Q1 (c) only: `espnet2/asr/espnet_model.py`, `espnet2/st/espnet_model.py`, `espnet2/slu/espnet_model.py`, `espnet2/asr_transducer/espnet_transducer_model.py`: on a ROCm torch, construct `RNNTLossNumba(blank=..., reduction="mean", fastemit_lambda=...)` instead of `warprnnt_pytorch.RNNTLoss`; check the reduction and FastEmit semantics match warp-transducer's (both derive from the same algorithm; compare losses and gradients on CPU, where both run, before relying on it).
7. Tests: skip the Numba-backed tests with a reason when the Numba GPU target is unavailable on a ROCm torch (R8). Add a small GPU test comparing `RNNTLossNumba` on GPU against its CPU path (and against warp-transducer's CPU build where installed) if none exists.
8. `doc/installation.md`: a ROCm subsection next to the CUDA one: `make TH_VERSION=2.13.0 ROCM_VERSION=7.2` (or whatever spelling item 5 settles on), what the optional extras do on ROCm (k2 wheel, numba-hip, flash-attn build, what is not available).

No change to `espnet3/`, recipes, or `docker/`.

## Build commands

No compile step for espnet itself. On the executing platform:

```
python3 -m venv <venv> && . <venv>/bin/activate
pip install torch==2.13.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -e '.[asr,test]'            # from the fork checkout
pip install numba-hip 'numba==0.61.2'   # pin per R4
pip install k2==1.24.4.dev20260710+rocm7.2.torch2.13.0 --no-deps -f https://k2-fsa.github.io/k2/rocm.html
pip install ninja                        # WKV JIT
```

Then the same through the changed installer, which is the deliverable: `cd tools && ./setup_venv.sh $(command -v python3) && make TH_VERSION=2.13.0 <rocm-version-arg>` followed by `make k2.done` and `python3 check_install.py`. The WKV extension builds on first use (`load_wkv_kernel`), for the GPU torch sees. A host whose system ROCm differs from the wheel's uses the wheel's bundled runtime; record both versions.

## Test plan

GPU tests (each must run, not skip, on every Linux validator; record pass/skip/fail counts):
- WKV: `pytest test/espnet2/asr_transducer/test_decoder.py test/espnet2/asr_transducer/test_espnet_transducer_model.py test/espnet2/asr_transducer/test_beam_search_transducer.py -k rwkv` (the `is_rwkv` training cases move the model to GPU through the kernel). Also a forward and backward check of `WKVLinearAttention` on GPU against a float64 CPU reference of the same recurrence (an agent_space script, or a test if upstream wants one), at the default context size and at `C < 32`.
- Numba RNNT: `pytest test/espnet2/asr/test_espnet_model.py -k transducer` (the `multi_blank_durations=[2]` case runs on `cuda`), `test/espnet2/asr/transducer/test_transducer_beam_search.py`, plus the item 7 GPU-vs-CPU loss/gradient comparison.
- k2 on GPU: `pytest test/espnet2/bin/test_asr_inference_k2.py test/espnet2/asr/test_ctc.py test/espnet2/asr_transducer/test_espnet_transducer_model.py` with k2 installed (these `importorskip` k2; confirm they did not skip).
- device tests: `pytest test/espnet2/torch_utils/test_device_funcs.py test/espnet2/torch_utils/test_resolve_device.py`.
- flash-attn gate (item 4): where ROCm flash-attention installs, build a `ConformerEncoder(use_flash_attn=True)` on GPU and confirm the flash path is taken and matches the SDPA path within tolerance; where it does not install, record that.
- Integration, the GPU recipe upstream already runs: `ci/test_integration_espnet2.sh`'s mini_an4 multi-blank transducer block (`egs2/mini_an4/asr1/run.sh ... --ngpu 1 ... transducer_multi_blank_durations=[2]`), plus one ordinary ASR training stage with `--ngpu 1` (stage 10 to 13) and, under Q1 (c), the plain transducer block with `--ngpu 1`.
- Full suite on the GPU host: `pytest -q test/espnet2 test/espnet3` (upstream's `ci/test_python_espnet2.sh` / `test_python_espnet3.sh` without the lint steps). Compare against a CPU-torch run of the same checkout on the same host to separate ROCm failures from environment ones (network-fetching tests, missing optional packages).

Non-GPU regression set: the CPU-torch `pytest test/espnet2` run above must not change between upstream and the port; `ci/test_shell_espnet2.sh` installer bats tests if any touch the edited scripts; `pycodestyle`/`flake8`/`black`/`isort` as configured in `setup.cfg` and `ci/test_flake8.sh` on every edited Python file.

CUDA path: the edits are Python and shell with ROCm-only branches, so there is nothing for nvcc to compile except the unchanged WKV sources. Before presenting, still run the WKV JIT build and test on the mi355 fleet host under a CUDA torch to show the CUDA flag list is untouched, and a CPU-torch `make` to show the installer's CPU path is unchanged.

Windows (gfx1151 or other windows-* gate): torch from the ROCm Windows wheels; plain-torch tests, WKV JIT build and test, device tests; Numba and k2 tests skip with the R8 reason (Q3).

## Open questions

Q1 to Q3 at the top. None of them blocks planning; each needs Jeff's answer before the porter starts.
