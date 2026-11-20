# ktransformers notes

## Intake screen (2026-11-05, windows-gfx1151 host, upstream main c9873312)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict partial, AMD path upstream). Lead, not finding.

### Licence

- `utils/licenses.py check kvcache-ai/ktransformers`: Apache-2.0, tier 1. Top-level LICENSE read: Apache-2.0.
- `utils/licenses.py scan-nvidia agent_space/ktransformers-screen` (shallow main tree, submodules not initialised): no NVIDIA proprietary licence text.
- Vendored Marlin (`kt-kernel/cuda/gptq_marlin`) carries Apache-2.0 headers (Neural Magic / Elias Frantar).
- Submodules: llama.cpp (MIT), pybind11 (BSD-3), kvcache-ai/custom_flashinfer (FlashInfer fork, Apache-2.0), kvcache-ai/sglang (SGLang fork, Apache-2.0). Not scanned file by file; only matters if adopted.

### What the project is now

The repository was restructured in 2025-10 (PR #1523 "add kt-kernel"). The live product is `kt-kernel/`: a CPU MoE expert kernel library (Intel AMX, AVX512, AVX2, AMD BLIS) that plugs into the kvcache-ai SGLang fork (`sglang-kt`). Hot experts run on GPU inside SGLang, cold experts run on CPU in kt-kernel. The old ktransformers inference/SFT tree with its GPU operators lives under `archive/`.

CUDA surface, measured on the shallow clone:

- `archive/` holds ~457 KB of `.cu/.cuh` (legacy ktransformers_ext, custom_marlin, kt-sft). Archived, not the live path.
- `kt-kernel/cuda/` holds ~143 KB: `gptq_marlin`, `custom_gguf/dequant.cu`, `moe/moe_topk_softmax_kernels.cu`, built by a standalone `cuda/setup.py` as the `KTransformersOps` extension. Nothing outside `archive/` imports `KTransformersOps`; the live kt-kernel Python package and CMake build do not build or use it.
- The kt-kernel CMake build compiles no CUDA kernels on any backend; its GPU dependency is only the host stream bridge (`cudaLaunchHostFunc` to order CPUInfer work on the GPU stream).

### Existing AMD support (upstream, maintainer-owned)

- `kt-kernel/CMakeLists.txt`: `KTRANSFORMERS_USE_ROCM` option, `find_package(HIP)`, links `libamdhip64`; `setup.py`/`install.sh` expose it as `CPUINFER_USE_ROCM=1`.
- `kt-kernel/cpu_backend/vendors/hip.h`: maintained HIP/hipBLAS vendor shim; `cpuinfer.h` enables `submit_with_cuda_stream` / `sync_with_cuda_stream` under `KTRANSFORMERS_USE_ROCM` (PR #1918 merged 2026-04-08, reverted #1925, and the ROCm path is present again on current main).
- `doc/en/ROCm.md` (Beta, since 2025-03, tested on EPYC 9274F + Radeon 7900 XTX), with a Hygon DCU (gfx936, DTK) section added after issue #2065 (2026-07, user reports the kt-kernel ROCm build works).
- AMD CPU side is also actively supported upstream (AMD BLIS backend, PRs #1600, #1702, #1744).
- GPU side: SGLang. Upstream `sgl-project/sglang` is dispositioned `already-supported` (ROCm upstream, AMD-maintained). The kvcache-ai fork is 221 commits ahead of upstream SGLang; the compare's first 300 changed files touch no `sgl-kernel`, `.cu`, or HIP/ROCm file, so its delta is Python-level kt integration on top of upstream SGLang's ROCm kernels.
- `kt-kernel/test/per_commit/test_amd_placeholder.py`: maintainers have registered an AMD CI suite, still a placeholder.

### Duplicate-effort check

- No ktransformers repository in ROCm or AMD-Ecosystem orgs (`gh search repos ktransformers --owner ...`: empty). `amd` and `AMD-AGI` orgs are behind SAML SSO for this token and were not searched; that check is open.
- Fork `liminfei-amd/ktransformers` (name suggests an AMD employee): 0 commits ahead of upstream main, 63 behind; carries only upstream branch names. No independent AMD effort visible.
- Upstream PRs mentioning ROCm/HIP/AMD: #1918/#1925 (ROCm stream bridge), #2070 (closed docs note for ROCm/vendor torch builds), AMD BLIS CPU PRs. No open ROCm porting PR.

### Gaps that remain

- Marlin/GPTQ, GGUF dequant and MoE top-k softmax kernels have no HIP build, but they belong to the archived ktransformers path and the unused `KTransformersOps` extension, not the live kt-kernel + SGLang path. Porting them would serve archived code only.
- `doc/en/ROCm.md` is stale for the main path: its run steps use `ktransformers/local_chat.py` and `optimize_rules/rocm/*.yaml`, which now live under `archive/`. Users hit this (issue #2115, 2026-07: 2x Radeon Pro V620 gfx1030, could not get the kt-kernel + sglang-kt stack running; issue #1178: nv_bfloat build errors on ROCm). The friction is SGLang-on-ROCm packaging (sgl-kernel ROCm build, consumer RDNA arches), which is SGLang's, plus stale docs.
- No real AMD GPU tests (placeholder only).

### Recommendation

Decline, reason `already-supported`. The ROCm path for the live product is upstream and maintainer-owned (kt-kernel `CPUINFER_USE_ROCM`, HIP vendor shim, ROCm docs, registered AMD CI slot), and the GPU compute comes from SGLang, already dispositioned `already-supported`. The ~600 KB of `.cu` the matrix counts is almost entirely archived or unused code, so there is no live CUDA kernel for MOAT to port.

The residual value is a docs refresh of `doc/en/ROCm.md` for the kt-kernel + sglang-kt flow and a real AMD test replacing the placeholder. That is a small, packaging/docs-shaped contribution under the 2026-11-05 "little CUDA source still in scope" policy; if a person prefers it to a decline, the narrow scope is: build kt-kernel with `CPUINFER_USE_ROCM=1` plus sglang-kt on wave64 (MI300-class) and wave32 (gfx1100), run kt-kernel's tests and one SGLang hybrid smoke, and send upstream the ROCm.md rewrite. Windows is not a plausible gate (kt-kernel and SGLang are Linux-only). It would depend on SGLang's ROCm build working for the sglang-kt fork, which MOAT does not own.

### Question for jeff

Decline as `already-supported`, or take the narrow docs-and-validation scope above (kt-kernel ROCm build + sglang-kt smoke on wave64/wave32, ROCm.md rewrite upstream, Windows gate waived as Linux-only)?
