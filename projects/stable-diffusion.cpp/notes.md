# stable-diffusion.cpp notes

## Intake screen (2026-11-05, gap-matrix seed)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: ready, upstream).

- Licence: MIT (`utils/licenses.py check leejet/stable-diffusion.cpp` -> tier 1), https://github.com/leejet/stable-diffusion.cpp/blob/master/LICENSE. Vendored `src/core/cuda/sageattention/LICENSE` is Apache-2.0 (tier 1). Not NVIDIA-owned, so no per-file NVIDIA scan was needed.
- Upstream active (pushed 2026-11-05), not archived.
- Existing AMD support, upstream and maintained by the project itself:
  - `-DSD_HIPBLAS=ON` (sets `GGML_HIP`), documented in `docs/build.md` "## Build with HipBLAS" plus a dedicated `docs/hipBLAS_on_Windows.md`.
  - CI: `.github/workflows/build.yml` job `windows-latest-rocm` (ROCm 10.1.0); releases ship ROCm binaries for Linux and Windows, e.g. master-1052-f5df5f8: `sd-master-f5df5f8-bin-Linux-Ubuntu-24.04-x86_64-rocm-10.1.0.zip`, `sd-master-f5df5f8-bin-win-rocm-10.1.0-x64.zip`.
  - sd.cpp's own CUDA kernels (`src/core/cuda/`, 24 `.cu`/`.cuh`) are built for HIP too: `src/core/cuda/CMakeLists.txt` `sd_cuda_sources(hip ...)` compiles rope, int8, w4, convrot, affine, modulate, bf16, rms_affine and registration for HIP, with an AMD-specific W4A8 WMMA path (`-mwavefrontsize64` on GFX11/GFX12). Upstream already writes AMD-native code.
- Residual CUDA-only surface (optional acceleration, not a correctness gap): `--sage-attn` SageAttention (`sage_attention.cu`, `sageattention/*.cuh`; SM80+/SM89 INT8/FP8 mma, cp.async), `sol_attention.cu`, and the CUTLASS/CuTe kernels behind `SD_CUDA_CUTLASS_PATH` (`cute_attention.cu`, `i8_*.cu`, `bf16_gemm.cu`). None has a hipify translation; covering them would be an AMD-native rewrite (rocWMMA/CK/MFMA), not a port.
- Open upstream PRs touching AMD: only #2110 "docs: update hipblas build step". No competing port.

## Recommendation

Decline, reason `already-supported`: upstream ships a maintained HIP build with ROCm 10.1 CI and Linux/Windows ROCm release binaries, and compiles its own core kernels for HIP with AMD-specific WMMA paths. The CUDA-only remainder is optional NVIDIA-tuned attention/GEMM acceleration.

## Question for jeff

Is an AMD-native SageAttention / CUTLASS-attention equivalent for sd.cpp (the `--sage-attn` and `SD_CUDA_CUTLASS_PATH` kernels, CUDA-only today) worth tracking as separate deferred work, or does the decline cover it?
