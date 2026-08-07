# pegainfer notes

## 2026-08-07 -- Intake screen (linux-gfx90a)

Upstream renamed mid-screen: `openinfer-project/pegainfer` -> `pegainfer-project/pegainfer` (GitHub redirect confirmed via `port_request.py check`). Scaffolded and recorded under the current name; nothing was previously adopted, dispositioned or requested for either name.

### Licence

Apache-2.0, tier 1, cleared to contribute (`python3 utils/licenses.py check pegainfer-project/pegainfer`). GitHub's licence field matched cleanly; no per-file NVIDIA-proprietary markers found in a scan of the top-level tree. Not independently re-checked inside the four vendored `third_party/` submodules (DeepEP, DeepGEMM, FlashMLA, flashinfer) since the viability finding below makes that moot; if this project is ever revisited each submodule's own licence needs the same per-file check before anything from it is touched.

### Duplicate effort

No mature or partial AMD port of pegainfer itself: no `pegainfer` (or `openinfer`) repo in AMD-Ecosystem or ROCm, no fork of `pegainfer-project/pegainfer` in either org (checked both current and pre-rename names), and pegainfer's own README/docs carry zero mentions of amd/rocm/hip/gfx (the only hits were `cudaExecutionCtxStreamCreate`/`cudaStreamDefault` substring false-positives in a CUDA green-contexts doc). GitHub lists 30 forks of pegainfer; all are personal/community forks, none AMD-flavored.

One relevant partial fact: pegainfer vendors `flashinfer` as a submodule (`pegainfer-kernels/third_party/flashinfer`, pinned to `flashinfer-ai/flashinfer` upstream), and **AMD-Ecosystem already carries `AMD-Ecosystem/flashinfer` ("FlashInfer+ROCm: ROCm port of FlashInfer")**. That's a coordination point for one of pegainfer's four vendored kernel libraries, not for pegainfer itself -- and note flashinfer-ai/flashinfer was itself dispositioned `cant-port` in `data/dispositions.json` on 2026-06-03 ("no existing AMD path... AMD may be porting independently") -- the fork that appeared four days ago confirms that guess was right: someone ported it directly, not through MOAT's mechanical hipify pipeline. That is the shape any AMD support for pegainfer's other three vendored libraries would likely take too, if it happens at all.

### Viability -- CUDA surface

Genuinely and heavily CUDA, through two layers, both blocking:

**1. Hand-written kernels.** `pegainfer-kernels/csrc/` has ~30 `.cu`/`.cuh` files across model backends (GLM5.2, Kimi-K2, DeepSeek-V2-Lite, Qwen3.5), compiled via a `cc` build-dependency (nvcc). Several are explicitly Blackwell/SM100-only, e.g. `glm52/glm52_deepgemm_grouped_sm100.cu`, which is an AOT instantiation of DeepGEMM's `tcgen05` (5th-gen tensor-core) masked grouped GEMM, hand-tuned per SM count for B200/GB300. This is the same fault class the project already carries seven `cant-port` dispositions for (flashinfer, FlashKDA, mirage, cuLA, qutlass, TileFusion, h100_gemm, ozIMMU): CUTLASS/CuTe-style warp-specialized PTX (wgmma/TMA/tcgen05/mbarrier) with no CK/ck_tile equivalent, reimplementation rather than translation.

**2. The `cudarc` crate.** Every model backend, plus `pegainfer-core` and `pegainfer-server`, depend on `cudarc` (Rust CUDA driver-API bindings: context/stream/memory management, NVRTC JIT compilation, cuBLASLt, NCCL). `cudarc` has **no ROCm/HIP backend at all** -- confirmed by search; it is CUDA-only, and the Rust ecosystem's HIP-capable crates (`hip-sys`, `rocm-rs`, `oxicuda-rocm`, `cubecl-hip`) are separate, unrelated, and far less mature. This is the more decisive finding: it sits underneath every model the engine serves, including the simplest ones with no exotic kernels, not just the Blackwell-specific paths. Nothing in pegainfer can reach a ROCm device today regardless of which kernel path is exercised.

Also vendors three more CUDA-only third-party kernel libraries as git submodules: `DeepEP`, `DeepGEMM` (pegainfer's own fork), `FlashMLA` (all deepseek-ai, Hopper/Blackwell-specific, NVSHMEM- and PTX-heavy) -- none has a known ROCm port. (`flashinfer`, the fourth, does -- see Duplicate effort above.)

Not archived: actively developed, pushed today, 629 stars.

### Recommendation

**Decline**, reason `cant-port`. Two independent, compounding blockers, both cheaply verified rather than estimated: the shared GPU binding layer (`cudarc`) has zero ROCm path today, and the model-specific kernels that would remain even if it did are dominated by Blackwell-only CUTLASS/CuTe-style PTX matching the project's existing `cant-port` pattern. Taking this up would mean first authoring a HIP backend for `cudarc` itself (a separate, foundational Rust-crate project, not a pegainfer port) and then a ground-up CK/ck_tile reimplementation of the GLM5.2/Kimi-K2 MoE kernels -- multi-week work with no mechanical-port slice to start from, not a hipify pass. If a person disagrees, the highest-leverage narrower option would be scoping to the plain Qwen3 backend only (skips the SM100-specific kernels) -- but that still needs `cudarc` to speak HIP first, which nothing in the current Rust ecosystem provides.
