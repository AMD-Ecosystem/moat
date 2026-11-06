# candle notes

## Intake screen (2026-11-05, windows-gfx1151 host, intake role)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict partial, AMD path community-fork). Upstream screened at `cfcbd7186d13271843ce7eb41d25659bd0835850` (2026-11-01).

### Licence

- `python3 utils/licenses.py check huggingface/candle` -> Apache-2.0, tier 1, cleared to contribute. Repo is dual MIT / Apache-2.0 (`LICENSE-MIT`, `LICENSE-APACHE`).
- No submodules (`.gitmodules` absent). `candle-flash-attn` and `candle-flash-attn-v3` vendor flash-attention / CUTLASS sources (BSD-3 / BSD), which the scan covers.
- `python3 utils/licenses.py scan-nvidia agent_space/candle-screen` -> "no NVIDIA proprietary licence text".

### What it is and how the CUDA path is built

- Rust ML framework. CUDA is a backend (`candle-core/src/cuda_backend`, ~4.3k lines of Rust) behind the `cuda` feature, using `cudarc` 0.19.10 for driver/runtime/cuBLAS/cuRAND/cuDNN/NCCL bindings.
- Kernels: `candle-kernels` (15 `.cu` in `src`, plus `src/mmq_gguf` and `src/moe`). `build.rs` uses `cudaforge::KernelBuilder` to compile most modules to PTX at build time (loaded at runtime through cudarc), and statically compiles the MoE / MMVQ / MMQ GGUF kernels. Optional `cutile` feature.
- Separate crates `candle-flash-attn` (FA2, 54 files) and `candle-flash-attn-v3` (FA3 Hopper, 67 files) are CUTLASS/CuTe and have no hipify path; the ROCm answer there is CK / aotriton flash-attention, a separate question (`cryptomilk/candle-flash-attn-rocm`, a community repo created for exactly this, exists).
- So a HIP port is a **backend implementation** (Rust: device, storage, allocator, BLAS/RNG bindings, kernel loading, op dispatch) plus kernel reuse, not a `.cu` kernel port. It needs a cudarc equivalent for HIP: `RustNSparks/rocm-rs` (107 stars, pushed 2026-10-30) is what both community PRs use.

### Existing AMD effort (the deciding finding)

- No ROCm/HIP mention anywhere in the upstream tree or README.
- **Open upstream PR huggingface/candle#3801 "Feat/rocm backend" (xmiksay)**: +18491/-523, 123 files, 52 commits, last commit 2026-09-11, mergeable=CONFLICTING. Continues #3424 (n1gh7e0us, opened 2026-03-28, still open, author endorsed the takeover). Design: `Device::Rocm`/`Storage::Rocm` behind `#[cfg(feature = "rocm")]`, `rocm-rs` bindings, rocBLAS matmul, optional MIOpen conv, and it compiles the **same** `candle-kernels/src/*.cu` with hipcc through a ~360-line header shim (shadows `cuda_fp16.h` etc.), JIT-compiled with hipcc at runtime and disk-cached. Touches CUDA sources only in `quantized.cu` (+90/-43, `MMQ_ARCH` RDNA tile arms) and `unary.cu` (+65, new transpose kernels); author states these were never nvcc-compiled. Reports the full candle test matrix passing on gfx1101 (RX 7800 XT, ROCm 7.2.4) with quantized results bit-identical to CUDA; third parties report it working on gfx1201 (ROCm 7.14, ROCm 10) and #3424 on gfx1151 (Linux, TheRock 7.12). Includes a GPU-free CI job.
- Judgement: community, not AMD-official, but **sound and following good practice** (shared kernel sources rather than a hand-written fork, real test-suite evidence, non-invasive feature gating). Per `assess-existing-support.md` this is the "authoritative but incomplete unmerged PR" case: value is validating and improving it, not a parallel port.
- Gaps visible from the PR: tested only on RDNA wave32, Linux. Nothing on CDNA wave64 (gfx90a/gfx942/gfx950) and nothing on Windows. The kernels inherited from llama.cpp (`quantized.cu`, MMQ/MMVQ) carry warp-size-32 assumptions, so wave64 is the likely differentiator (unverified: a planner check). A reviewer (zachelnet) reports an f32 im2col conv memory cliff at large spatial sizes on gfx1201.
- Other community efforts, all smaller and non-authoritative: closed #3186 (AI-generated, untested), `vberthet/candle` (stale 2024-02), `Nu11ified/candle-rocm` (2026-03, not a fork, small), `vuongthai91/candle-rocm`.
- AMD-org check: `ROCm/candle` and `AMD-Ecosystem/candle` do not exist. `amd/candle` and `AMD-AGI/candle` return 404, but those orgs are behind SAML SSO for this token, so a 404 there is **not** proof of absence; that duplicate-effort check stays open.

### Maintainer stance

- No maintainer has commented on or reviewed #3801 or #3424 (only users and contributors). A contributor wrote on issue #3985 (2026-09-17): "candle is not really moving forward ... There is a PR for adding ROCm support, but it doesn't look like someone cares about it".
- Only explicit maintainer statement found: Narsil on issue #346 (2023-08-09): not in the near future, but start with ROCm bindings and "aim for the same thing as cuda". Neither acceptance nor refusal of a HIP backend.
- Upstream is active: ivarflakstad (version bumps, CI) and EricLBuehler (cudarc bump, cutile, quant kernels) merge regularly through 2026-11-01. EricLBuehler is both a candle merger and the author of `EricLBuehler/mistral.rs`.

### Relation to mistral.rs

`EricLBuehler/mistral.rs` (another seeded candidate) depends on crates.io `candle-core`/`candle-nn` 0.11.1 and `candle-flash-attn-v3`, not on a fork. Any mistral.rs ROCm path needs a candle ROCm backend first, so candle would be a hard dependency (`moatlib.py set-deps mistral.rs candle`) once mistral.rs is scaffolded; it is not scaffolded yet, so no dependency is recorded here.

### Viability

Viable: real CUDA backend with large GPU surface, permissive licence, active upstream, a working community HIP backend already exists to build on. Main costs: Rust backend work rather than kernel hipify; runtime hipcc JIT on Windows is an untried path; flash-attn crates are a separate CK question.

### Recommendation

**Take up, as validate-and-improve on #3801, not a from-scratch port.** Work: fork candle, base the port on #3801's branch (rebased or merged to current main since it conflicts), validate on wave64 (CDNA) and Windows which the PR never covered, fix what breaks (warp-size assumptions in the shared kernels are the first suspect), nvcc-compile the two touched `.cu` files on the mi355 host (the PR's own stated gap), and offer the fixes to the PR author. Do not open a competing upstream PR.

### Question for jeff

Upstream has an open, sound third-party ROCm backend (huggingface/candle#3801, xmiksay, building on #3424) with no maintainer review in three months. Do you want MOAT to (a) adopt candle and validate/improve #3801's branch on wave64 + Windows, offering fixes to its author (no competing PR), (b) only post validation evidence on #3801 without a fork, or (c) decline as duplicate effort and revisit if #3801 stalls or is closed? Also: can someone with `amd` / `AMD-AGI` SSO confirm neither org has a candle fork?
