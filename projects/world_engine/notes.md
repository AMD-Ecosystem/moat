# world_engine notes

## Intake screen (2026-11-12)

Origin: Jeff Daily's 2026-11-12 ruling on a third-party "AMD AI software gap" list -- validation project (PyTorch + Triton).

- Upstream: https://github.com/Overworldai/world_engine, default branch `main`, pushed 2026-09-30, not archived, small (41 files: `src/` world model, autoencoder, KV cache, quantize; `examples/benchmark.py`, `gen_sample.py`, `prof.py`).
- Licence: **GPL-3.0 (LICENSE), tier 3** -- contributing upstream needs Jeff's licence clearance (`record-license-clearance`) before anything is offered. `scan-nvidia` on a depth-1 clone: clean. No submodules. Git-URL dependencies: madebyollin/taehv (MIT), dropbox/gemlite (Apache-2.0).
- CUDA surface: no `.cu`/`.cuh`. Pure PyTorch (`torch==2.11.0` pinned) plus Triton (`triton==3.6.0` on Linux, `triton-windows==3.6.0.post26` on win32) and gemlite (Triton low-bit GEMM kernels).
- README claims "Optimized backends for Nvidia, AMD, Apple Silicon, etc." but the source has no ROCm/HIP/AMD code path: a grep of every `.py` finds only `torch.version.cuda` in the benchmark and FlashInfer NVFP4 in `src/quantize.py`. So the AMD claim is untested by anything in the repo.
- Duplicate effort: no ROCm/AMD/HIP issues, PRs or branches upstream; no AMD-Ecosystem or ROCm fork; no community ROCm repo found.

## Validation scope

- Delivery route: upstream PR of fixes (Triton/gemlite on ROCm, quant-scheme gating) if Jeff clears the GPL-3.0 licence; fork-only otherwise.
- GPU gate: `pytest examples/benchmark.py` (upstream's benchmark harness, pytest-benchmark) plus `examples/gen_sample.py` end to end with `Overworld/Waypoint-1.5-1B`, bf16 first, then `intw8a8` and `fp8w8a8` quant.
- CUDA-only dependencies: FlashInfer NVFP4 (`nvfp4` quant, Blackwell only -- gap, document as unsupported on AMD); `fp8w8a8` uses `torch._scaled_mm` e4m3 (works on gfx942/gfx950 with fnuz/ocp fp8 selection; check, the README lists only NVIDIA); `intw8a8` via gemlite Triton kernels (gemlite's AMD support is partial -- main risk). The `torch==2.11.0` and `triton==3.6.0` exact pins resolve to CUDA wheels on PyPI; a ROCm install needs the ROCm torch index and its bundled triton.
- Windows: upstream has a Windows story (`triton-windows` pin), so Windows is in scope -- no waiver; triton-windows is CUDA-only, so on ROCm Windows the Triton path depends on TheRock's Windows Triton (risk).
- Existing AMD work to back: none.

## Declined (2026-11-12)

Jeff Daily declined world_engine as license-blocked on 2026-11-12: GPL-3.0 (tier 3) needs a licence clearance that is not his to give, the README's AMD claim has no ROCm code behind it, and the GPU stack leans on NVIDIA-specific pieces (gemlite low-bit Triton GEMMs, NVFP4 via FlashInfer). Intake is re-recorded as decline and the project is held. Do not request a fork.
