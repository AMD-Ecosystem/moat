# mistral.rs notes

## Intake screen (2026-11-05, windows-gfx1151 host, intake role)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict unknown, AMD path unknown). Upstream screened at `52b3c49b2021e77bbd80bcbbbcfae534a89decd0` (default branch `master`, pushed 2026-11-03, 7.8k stars, not archived).

### Licence

- `python3 utils/licenses.py check EricLBuehler/mistral.rs` -> MIT, tier 1, cleared to contribute. `LICENSE` is the plain MIT text, "Copyright (c) 2024 Eric Buehler".
- No submodules. Vendored third-party trees, each with its own licence file: `mistralrs-core/third_party/flashinfer_gdn` (Apache-2.0), `mistralrs-core/third_party/ggml_gdn` (MIT, ggml), `mistralrs-paged-attn/third_party/flash-attention` (BSD-3), `mistralrs-quant/third_party/ggml_mmvq` (MIT), `mistralrs-quant/third_party/deepgemm_sm90` (Apache-2.0 / MIT / BSD-3).
- `python3 utils/licenses.py scan-nvidia agent_space/mistral.rs-screen` -> 1 hit: `mistralrs-quant/third_party/deepgemm_sm90/LICENSE-BSD-3-CLAUSE`. Read it: it is the CUTLASS BSD-3 licence, whose trailing paragraph says files under CUTLASS's `python/CuTeDSL` directory are under the NVIDIA EULA. No CuTeDSL files are vendored here (`find -iname '*cutedsl*'` is empty; CUTLASS itself is fetched at build time, not vendored). Judgement: clean, the hit is the EULA carve-out sentence in an otherwise BSD-3 file. Also moot for a port: deepgemm_sm90 is Hopper WGMMA code that a HIP build would not compile at all.

### CUDA surface outside candle

About 3.5 MB of `.cu`/`.cuh` (the matrix's 2.45 MB undercounts vendored headers). By crate:

- `mistralrs-quant/kernels` (~1.37 MB): GGUF MMQ/MMVQ (from llama.cpp), GPTQ, Marlin, AFQ, HQQ, bitsandbytes, blockwise/scalar FP8, MXFP4/MXFP8, NVFP4 (decode, w4a16, and a CUTLASS path), MoE / grouped MoE / indexed MoE / routed GEMV, LoRA, rotary. CUTLASS-dependent: `blockwise_fp8`, `cutlass_moe`, `marlin`, `nvfp4_cutlass`.
- `mistralrs-quant/third_party/deepgemm_sm90` (~0.98 MB): Hopper-only, out of scope for HIP.
- `mistralrs-core/src/cuda` (~0.6 MB): ~20 model-specific kernels (GDN / gated delta net incl. `gdn_mma.cu`, SSM, MoE GEMM incl. `moe_gemm_wmma.cu`, rope, top-k sort, speculative rejection, CUDA graph helpers). `third_party/flashinfer_gdn_sm90` is Hopper-only.
- `mistralrs-paged-attn` (~0.57 MB): vLLM-style paged attention v1/v2, KV cache ops, MLA, FlashInfer decode, and an FA3 (Hopper) path; `third_party/flash-attention/hopper`.
- `mistralrs-flash-attn` (~30 KB of its own, 54 files): FA2 sm80 instantiations on CUTLASS/CuTe. The ROCm answer is CK / aotriton flash-attention, not hipify (same question as candle's flash-attn crates).
- Build: each of `mistralrs-core`, `-quant`, `-paged-attn`, `-flash-attn` has a `build.rs` that drives nvcc through `cudaforge`. A HIP port needs a hipcc path in each.
- Libraries: cuBLASLt (`mistralrs-quant/src/cublaslt/`, via `cudarc::cublaslt`) -> hipBLASLt; NCCL (`mistralrs-quant/src/distributed`, `nccl` feature, `cudarc::nccl`) -> RCCL; optional cuDNN via candle -> MIOpen.
- The Rust side is the bigger cost: 272 `.rs` files and ~2070 `feature = "cuda"` gates, and ~320 call sites reach CUDA through **candle's re-export of cudarc** (`candle_core::cuda::cudarc::driver`, `candle_core::cuda_backend::cudarc::driver`). mistral.rs does not depend on cudarc directly; it reaches into candle's CUDA device/storage types.

### Dependency on candle (hard)

`Cargo.toml` pins crates.io `candle-core` / `candle-nn` / `candle-flash-attn-v3` / `candle-metal-kernels` 0.11.1. Every GPU path is a candle backend, so a candle ROCm backend is a prerequisite: recorded with `moatlib.py set-deps mistral.rs candle`. Candle analysis lives in `projects/candle/notes.md` on `port/candle` (open community PR huggingface/candle#3801, rocm-rs based, no maintainer review).

The candle backend's **design** decides the shape of the mistral.rs work, so it is worth settling on the candle side before planning here:
- #3801 adds a separate `Device::Rocm` / `Storage::Rocm` on `rocm-rs`. mistral.rs's ~320 cudarc call sites through candle would then each need a parallel ROCm arm (or a mistral.rs-side device abstraction): a large Rust change across the whole engine.
- The alternative seen in the wild (pstreet's fork, below) is a `cudarc-hip` crate, a cudarc API over HIP, so the existing `cudarc::driver` call sites and the `cuda` feature compile against HIP with far fewer Rust edits. That is a different candle backend from #3801.

### Existing AMD effort

- Upstream: no ROCm/HIP/AMD mention in README or docs. No open or merged ROCm/HIP PR (`gh pr list --state all --search "rocm OR hip OR amd"`: only unrelated hits). No hip/rocm/amd branch.
- Issues: #431 "Feature request: Rocm support" (open, 2024-06-14) and #1345 "ROCm support" (open, 2025-05-17); also #631 SYCL request, #2004 WGPU/CubeCL (closed).
- **Maintainer stance (EricLBuehler, owner, also a candle merger)**: on #431 (2024) "ROCm support is definitely on the radar ... I think we can use ZLUDA to quickly get started with ROCm support." On #1345, when a contributor argued platform support should be delegated to candle/Burn, he replied switching to Burn is attractive but its quantization support is lacking and it would need a type rework; a commenter notes he looked at `RustNSparks/rocm-rs` (rocm-rs issue #1). Net: positive in principle, no work started, and he points at the dependency layer. A contributor offered to donate an AMD GPU. No refusal anywhere.
- **Community fork `pstreet/mistral.rs`** (branch `pstreet-rocm`, not a GitHub fork, 4 stars, pushed 2026-11-05, ~4100 commits) plus `pstreet/candle` (created 2026-08-31, pushed 2026-11-03) carrying `cudarc-hip` and `cudarc-nvidia` crates. README calls it a "Deployment fork" serving ROCm `gfx1151` (Strix Halo) plus the Ryzen AI NPU; uses CK flash attention; points mistral.rs at a path-dependency candle checkout. Author said on #431 he has not focused on merging back but would like to work with the maintainer. Judgement per `assess-existing-support.md`: **non-authoritative** (personal deployment fork, single consumer arch, mixed with NPU and unrelated config/tuning work, not structured for upstream). Use as a hint only (proof the cudarc-hip approach carries mistral.rs on gfx1151; CK FA works), not as a base.
- AMD orgs: `ROCm/mistral.rs`, `AMD-Ecosystem/mistral.rs` do not exist. `amd/mistral.rs` and `AMD-AGI/mistral.rs` return 404 but those orgs are behind SAML SSO for this token, so that duplicate-effort check stays open.
- `gh search repos "mistral.rs rocm"`: no results.

### Viability

Viable but large and sequenced. Real, extensive CUDA surface (quant kernels, paged attention, model kernels, cuBLASLt, NCCL) on top of a candle backend that does not exist yet. Hopper-only pieces (deepgemm_sm90, flashinfer_gdn_sm90, FA3) and CUTLASS-based kernels (Marlin, cutlass_moe, nvfp4_cutlass, blockwise_fp8, FA2) are out of hipify scope: they get excluded or replaced by CK/hipBLASLt paths, a planner decision. The GGUF MMQ/MMVQ kernels come from llama.cpp, which has a mature HIP port to borrow from. Windows: upstream has a Windows MSVC + CUDA issue (#2310), so the Windows gate is plausible but untried.

### Recommendation

**Take up, sequenced behind candle** (the dependency is recorded, so the selector holds it until candle is done). The work: once candle has a ROCm backend, add a hipcc path to the four `cudaforge` build scripts, port mistral.rs's own kernels (quant, paged-attn, core model kernels) with wave64/wave32 care, map cuBLASLt -> hipBLASLt and NCCL -> RCCL, route flash attention to CK, and exclude Hopper/CUTLASS-only paths. Upstream is receptive in principle (owner said ROCm "is on the radar") and the owner also merges candle, which argues for designing the candle backend with mistral.rs's cudarc-through-candle usage in mind. Do not adopt pstreet's fork as a base.

### Question for jeff

mistral.rs (MIT, owner EricLBuehler says ROCm is "on the radar", no upstream PR) hard-depends on a candle ROCm backend, and its ~320 call sites use cudarc through candle's re-export. The candle backend in huggingface/candle#3801 is a separate rocm-rs `Device::Rocm`, which would force a parallel ROCm arm at each of those sites; a community fork (pstreet/candle + pstreet/mistral.rs, gfx1151 only, non-authoritative) instead uses a `cudarc-hip` shim that keeps them. Do you want MOAT to (a) adopt mistral.rs now as a dependent of candle and let the candle work decide the backend shape, (b) adopt it but first raise the cudarc-compatibility question in the candle plan, or (c) decline for now and re-screen once a candle ROCm backend lands? Also: can someone with `amd` / `AMD-AGI` SSO confirm neither org has a mistral.rs fork?
