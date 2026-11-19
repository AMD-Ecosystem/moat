# tinygrad notes

## Intake screen (2026-11-05)

Source: "AMD AI Software Gap Matrix" seed (verdict partial, AMD path upstream). Screened at upstream master `24a9c3a2f50d05d41a5e95d170ae4d9f3ec8e2b0` (shallow clone in `agent_space/tinygrad-screen`).

Recommendation: **decline, SKIP_REASON `already-supported`.** No gap a MOAT port would fill.

### Licence

- Top level: MIT (`python3 utils/licenses.py check tinygrad/tinygrad` -> tier 1, cleared to contribute). Evidence: https://github.com/tinygrad/tinygrad/blob/master/LICENSE
- `python3 utils/licenses.py scan-nvidia agent_space/tinygrad-screen` -> one hit: `extra/nvJitLink.h`, an NVIDIA proprietary-licence CUDA header vendored only as input to the ctypes autogen for the NV backend (`tinygrad/runtime/autogen/__init__.py:70`). Irrelevant to any AMD work; it would only matter if a port touched that file, which none would.

### What the `.cu` files are

All 256 `.cu`/`.cuh` files (about 2.2 MB, which is the linguist figure) live under `extra/`, none under the `tinygrad/` package:
- `extra/thunder/cuda/` (246 files): a vendored copy of ThunderKittens' CUDA headers, research/experiment code. Upstream already carries the AMD counterpart side by side in `extra/thunder/amd/` (HipKittens-style `cdna4` and `udna1` headers plus HIP `.cpp` kernels: flash attention fwd/bwd, bf16/mxfp8 GEMM, MoE router), and CI tests them (`DEV=NULL:HIP:gfx950 pytest test/testextra/test_tk.py`).
- `extra/gemm/max_kernels/` (10 files): hand-written GEMM experiments.

tinygrad itself generates kernels at run time; no CUDA source is compiled for the library.

### Existing AMD support upstream (mature, first-party)

- Backends: `tinygrad/runtime/ops_amd.py` (HCQ, renderers `AMD:LLVM` and `AMD:HIP` via COMGR; interfaces KFD over amdgpu, PCI via tinygrad's own "AM" userspace driver, USB for asm24xx eGPU) and `ops_hip.py`. `docs/runtime.md` lists CDNA3, CDNA4, RDNA3, RDNA4.
- CI on real AMD hardware: `.github/workflows/benchmark.yml` runs on self-hosted `7900xtx`, `4x7900xtx`, `9060xt` and `usb3-9060` runners; `test.yml` runs AMD renderer/emulator tests for gfx942, gfx950, gfx1100, gfx1201. tinycorp ships AMD tinyboxes and an MLPerf submission on 8x MI350X (`examples/mlperf/training_submission_v6.0/tinycorp/.../tinybox_8xMI350X`).
- Open upstream PRs matching "rocm OR hip": none.
- `ROCm/tinygrad`: a plain fork, last pushed 2024-05-07, stale; not an active AMD effort and not needed given upstream support.

### Why no residual work

The policy line for little-CUDA candidates (build/run on ROCm, packaging, GPU tests, docs) does not find a gap here either: upstream is pure Python with no compiled extension, documents AMD use, and runs AMD GPUs in its own CI on both wave64 (CDNA, emulated plus MI350X benchmarks) and wave32 (RDNA3/RDNA4 hardware). Windows is the one gate MOAT tracks that upstream does not cover, but tinygrad's AMD backend talks to the Linux kernel driver (KFD) or its own PCI driver, so a Windows path would be a new backend, not a port, and nothing in the matrix lead asks for it.

No question left for jeff.
