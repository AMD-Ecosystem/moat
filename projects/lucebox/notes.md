# lucebox notes

Intake screen 2026-08-06 (agent: intake, platform linux-gfx90a). Upstream: https://github.com/Luce-Org/lucebox (2720 stars, 257 forks, pushed same day, not archived, not a fork). Screened against a shallow clone of `main`; no fork exists and none should be created.

## Disposition: DECLINE -- already-supported (and a duplicate of an already-skipped repo)

Two independent reasons, either one sufficient.

**1. Upstream ships a first-class, actively maintained ROCm/HIP backend.** This is not a "someone once got it to compile" claim; it is upstream's own supported build path, benchmarked per-arch, with published ROCm container images and a dedicated perf-engineering doc.

**2. MOAT already dispositioned this exact codebase.** `data/dispositions.json` carries `luce-org/lucebox-hub` -> skip / `already-supported` / "real/current HIP backend (research-confirmed)" decided 2026-05-30. `Luce-Org/lucebox` is the same project under a renamed repo: its own README still instructs `git clone --recurse-submodules https://github.com/Luce-Org/lucebox-hub`, the published images are `ghcr.io/luce-org/lucebox-hub:{cuda12,rocm}`, and the HIP perf doc references `Luce-Org/lucebox-hub` PRs #119/#122/#156. So the candidate row is the same target arriving under a second name, and the earlier finding stands unchanged and has only strengthened since.

## Licence (recorded fact, tier 1)

`Apache-2.0`. GitHub's field, the repo badge and the LICENSE file itself agree; the file was read in full (201 lines, unmodified Apache-2.0 text, "Copyright 2026 Lucebox", no appended non-commercial or field-of-use rider).

Vendored/submodule sweep -- all permissive, none unlicensed:

- `server/deps/llama.cpp` -- MIT ("Copyright (c) 2023-2026 The ggml authors"), vendored in-tree.
- `server/deps/llama.cpp/gguf-py` -- MIT.
- `server/deps/Block-Sparse-Attention` -- the only git submodule (mit-han-lab/Block-Sparse-Attention, pinned 49d6c39), BSD-3-Clause per its upstream repo.

NVIDIA proprietary licence check: all four TEXT markers from `config/licenses.toml` `tier3.nvidia_proprietary.text_markers` were grepped across the whole checkout ("NVIDIA Source Code License", "NVIDIA Software License Agreement", "NVIDIA CORPORATION and its licensors retain all intellectual property", "NVIDIA End User License Agreement"). Zero hits. Nothing here forces tier 3.

Nothing ambiguous, so nothing to escalate: a permissive top level over permissively licensed vendored parts.

## Existing AMD support -- the evidence

Upstream build system, not a downstream patch:

- `server/CMakeLists.txt` line 2: `set(DFLASH27B_GPU_BACKEND "cuda" CACHE STRING "GPU backend to build: cuda or hip")`, with `DFLASH27B_HIP_ARCHITECTURES` (documented examples `gfx906;gfx1100`), `AMDGPU_TARGETS` override precedence, and a mixed CUDA+HIP configuration path.
- HIP-specific sources maintained beside the CUDA ones: `server/src/rms_norm_hip.cu`, `server/src/flashprefill_kernels.hip.cu`, `server/src/bsa_launcher_hip.cu`, plus a `server/src/hip_compat/` shim (`cuda_fp16.h`, `cuda_bf16.h`) and a `server/src/device_runtime.h` that already reasons about the 64-lane wavefront ballot difference.
- rocWMMA is a real dependency of the fast path: `DFLASH27B_HIP_SM80_EQUIV=ON` selects the rocWMMA "Phase 2" flashprefill kernels, and the build emits `test_flashprefill_kernels` as a HIP numerical check. Documented apt line: `hipblas-dev hipcub-dev rocblas-dev rocprim-dev rocwmma-dev`.
- `server/docs/HIP_PERF_PLAN.md` is an ongoing ROCm optimization program with rocprofv3 kernel traces, arch-aware tuning, a falsified hypothesis written up so contributors do not repeat it, and shipped wins (RDNA MMQ tile override: gfx1201 54.65 -> 59.37 tok/s, gfx1100 56.78 -> 60.18, gfx1151 11.53 -> 12.00). `server/docs/MIXED_BACKEND.md` covers mixed CUDA+HIP runs.
- Benchmarked AMD targets in the top-level README support matrix: gfx1151 (Strix Halo, 37 tok/s), gfx1100 (RX 7900 XTX, 50 tok/s), gfx1201 (Radeon AI PRO R9700, 55 tok/s, "first-class RDNA4 target"), with per-arch DDTree budget tuning and a note that gfx1200 and gfx1201 are not code-object compatible.
- `Dockerfile.rocm` and a published `ghcr.io/luce-org/lucebox-hub:rocm` image; repo topics include `rocm` and `strix-halo`; the README carries a HIP 7+ badge next to the CUDA 12+ one.
- Even CDNA is present at the backend-precision level: `server/src/common/backend_precision.cpp` branches on `gfx90a`, with unit tests asserting BF16 activation selection for `gfx90a` and `gfx942`.

Per `assess-existing-support.md`, this is the authoritative case: the support is in the upstream tree, maintained by upstream, and validated by upstream on real AMD hardware. There is no "validate and improve a stale third-party port" opening -- the port is theirs and it is ahead of us.

## Viability facts (recorded even though we decline)

- It genuinely uses CUDA: 300 `.cu`/`.cuh` files total, but that count is dominated by the vendored `server/deps/llama.cpp/ggml/src/ggml-cuda/` tree (which itself has upstream ROCm support, plus a `ggml/rocmfp4/` addition). Hand-written first-party device code is about 12.8k lines across `server/src` and `optimizations/megakernel`.
- Build type: CMake (`ext_type: cmake`) for `server/`, which vendors ggml and needs no PyTorch. `optimizations/megakernel/` is the one PyTorch CUDAExtension component.
- No MOAT project dependencies (`depends_on` stays empty). It vendors llama.cpp/ggml rather than consuming any MOAT-ported library.
- Upstream is very much alive: pushed the day of this screen, 83 open issues, active PR-numbered engineering docs.

## The one CUDA-only component, and why it does not change the answer

`optimizations/megakernel/` (7712 lines across `kernel.cu`, `prefill.cu`, `prefill_bw.cu`, `prefill_megakernel.cu`, `kernel_gb10_nvfp4.cu`) has no HIP path: zero `hip`/`rocm`/`__HIP` hits, and every file uses inline PTX `asm volatile` / tensor-core `mma` intrinsics with cooperative grid launch, targeting specific NVIDIA arches (Ampere sm_86 reference, sm_121a GB10 NVFP4). Its own README pitches "tensor cores, shared memory, cooperative grid launches, register-resident state".

That is the reimplement-not-port class MOAT has repeatedly declined -- the same shape as mirage (skipped `cant-port`: "no tractable correctness-first HIP target -- AMD support would be a ground-up CK/ck_tile+HIP backend, not a mechanical port"), SpargeAttn and FlashRT. An NVFP4 GB10 megakernel has no mechanical AMD translation; it would be a fresh MFMA/WMMA implementation. And it is the demo/optimization component, not the server: the shippable product path (`server/`, DFlash + PFlash + KVFlash) is exactly the part that already runs on ROCm.

The other conceivable gap is CDNA (gfx90a/gfx942), since every benchmarked AMD target is RDNA. But the project's stated purpose is "consumer hardware & heterogeneous computing" -- data-center Instinct parts are deliberately out of its scope, and upstream has still put gfx90a/gfx942 handling in its precision selection. Chasing CDNA here would be pushing a feature upstream does not want into a codebase whose maintainers are already better ROCm engineers on this stack than a port would make us.

## What to do instead of porting

Nothing to plan, nothing to fork. If lucebox is interesting to AMD it is as a collaboration or a consumer of ROCm (their `HIP_PERF_PLAN.md` tl;dr names a concrete ROCm-side gap -- `mul_mat_q` for q4_K/q4_0/q5_0 in `ggml-cuda/mmq.cuh`+`mmvq.cuh` leaving a ~4x throughput gap versus an RDNA-native engine on gfx1100), not as a MOAT port target. Note also that everything above concerns CONTRIBUTING; using or shipping lucebox in AMD software is a separate question this screen does not answer.

## Lesson

Nothing to promote to the `cuda-to-rocm` skill: no new fault class, strategy or diagnostic came out of this. The one reusable observation is already MOAT policy -- a renamed upstream repo re-enters `data/candidates.json` under a new `full_name` and is not matched against an existing disposition, so intake should check dispositions for the ORG and for sibling repo names, not just the exact `owner/repo`. Recorded here, in the second write-up it would have saved.
