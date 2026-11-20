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

## Planning (2026-11-20, linux-gfx942 host, planner role)

### Questions for jeff (the porter waits for these rulings)

Intake's question (adopt as validate-and-improve, post evidence only, or decline) was answered by creating the fork. What follows is what the plan still needs from you. The plan (`plan.md`) marks each step that depends on one of these questions and shows my recommendation; a recommendation is not a ruling.

**Q1. What does `moat-port` start from?** #3801 is 21 upstream commits stale and conflicting. cryptomilk (Andreas Schneider) keeps a rebased copy of it, `cryptomilk/candle:asn-feat-rocm-backend` (5 behind main), with 12 more commits. One of them (`76c28f73`, variadic shuffle shims) is required: without it, upstream's newer `reduce.cu` does not compile under the #3801 shim, and an omitted width would be wrong on wave64.
- (a) #3801 head `fca7197d` (xmiksay's history), merge upstream main, then cherry-pick cryptomilk's 12 commits with his authorship. The trial merge conflicts in 3 files (`Cargo.toml`, `candle-core/src/quantized/mod.rs` 3 hunks, `candle-kernels/src/unary.cu`). Our commits stay applicable to #3801's own branch. **Recommended.**
- (b) cryptomilk's branch, merge upstream main (2 conflicting files). This is the simplest merge, but its commits are rebased copies that #3801's branch does not share, so offering fixes back to #3801 means cherry-picking ours across.
- (c) #3801 head plus upstream main only, leaving cryptomilk's commits out (we redo the shuffle fix ourselves and drop the RDNA WMMA MMQ). This is the smallest base but duplicates work that already exists.

**Q2. How is the work delivered?** MOAT's pipeline ends in `upstream.py --publish`, which would open our own PR on huggingface/candle. Intake recommended no competing PR. No maintainer has reviewed #3801 in 2.5 months, and the last activity is cryptomilk's 2026-09-28 ping.
- (a) No MOAT upstream PR. Offer our commits to xmiksay (a PR into `xmiksay/candle:feat/rocm-backend`, which is an external write needing your approval of the draft) and post wave64/Windows/CUDA-compile evidence on #3801 (also your approval). The fork's `moat-port` stays our deliverable until #3801 lands. You would also need to rule how the MOAT record closes: for example `set-maintained` on the evidence comment, or a hold until #3801 merges or closes. **Recommended**, with a review date to revisit (b) if #3801 stays silent.
- (b) Open MOAT's own upstream PR, superseding #3801 with xmiksay's and cryptomilk's commits and authorship kept and both credited, and ask them first whether they agree. This is the normal pipeline, but it is the competing-PR pattern intake flagged (gsplat #970).
- (c) Evidence only on #3801, with no fork branch work beyond what validation needs.

**Q3. Is the attention layer in scope?** `candle-flash-attn` (FA2, CUTLASS sm80) and `candle-flash-attn-v3` (FA3, Hopper TMA/wgmma) have no hipify path; #3801 does not touch them; upstream keeps both outside the workspace as opt-in crates behind the `flash-attn` feature. An AMD-native target exists: CK FMHA (or aotriton). cryptomilk's `candle-flash-attn-rocm` does this, but it has **no licence**, so it is reference only.
- (a) Scope it out of this port. Record a deferral for a ROCm flash-attention crate on CK FMHA, to be ruled separately once the backend lands. Without it, models with `--use-flash-attn` fall back to the non-flash attention path on ROCm, which every example already supports. **Recommended**, because the backend is the prerequisite and the delivery question (Q2) is unsettled.
- (b) Include it now: a new FA2-forward path on CK FMHA built from candle's build, for gfx942/gfx950 and RDNA, with forward-only parity tests against `flash_attn_tests.rs`. Your "pursue feasible opt-in surfaces" principle points here. It roughly doubles the port and adds a CK build dependency to a Rust crate.

**Q4. Confirm the smaller cuts and inclusions.**
- `cutile` (NVIDIA cuTile): scope out, no AMD-native target exists.
- `nccl` feature and `llama_multiprocess` example: scope out and record a deferral. RCCL would need new bindings that `rocm-rs` does not have.
- Include (small, in the plan): a ROCm arm for `qwen3_moe`'s FusedMoe selection, a `rocm_fwd` for the `custom-ops` example, and the routed-MoE fallback after upstream #4068.
- Still open from intake: someone with `amd`/`AMD-AGI` SSO should confirm neither org has a candle fork.

### What the analysis found (evidence for the plan)

- State of the art, re-checked today:
  - #3801 OPEN, DIRTY, head `fca7197d` (2026-09-11), 52 commits, all Martin Miksanik.
  - #3424 still open.
  - `gh pr list --state all --search "ROCm OR HIP OR AMD"` shows nothing new on the ROCm side; Vulkan #4110 and SYCL #3974 are unrelated backends.
  - #4075 (cryptomilk) is an open CUDA mmvq over-read fix.
  - cryptomilk's branches: `asn-feat-rocm-backend` is newest (2026-10-17; 64 ahead, 5 behind upstream main). `-wip` variants are older.
- Trial merges (local scratch worktrees, nothing pushed):
  - `fca7197d` + `origin/main@1a49dc27`: conflicts `Cargo.toml` (1 hunk), `candle-core/src/quantized/mod.rs` (3), `candle-kernels/src/unary.cu` (1).
  - `asn-feat-rocm-backend` + main: `Cargo.toml`, `candle-core/src/quantized/mod.rs`.
- Compile break found by reading:
  - Upstream #3600 (`804d361e`) uses 3-argument `__shfl_xor_sync` at `candle-kernels/src/reduce.cu:165,177,969,979`.
  - #3801's `hip_shim/hip_compat.h` defines `__shfl_xor_sync(mask, var, lane_mask, width)` as a 4-parameter macro, so the merged tree's `reduce.cu` fails to preprocess under HIP.
  - cryptomilk's `76c28f73` replaces these with variadic macros that default the width to 32.
  - `moe/moe_wmma_gguf.cu:69` has a 3-argument `__shfl_sync` too, but that file is not in the ROCm module list.
- Wave64 sweep of the 11 runtime-compiled modules:
  - No `__ballot`, `__activemask`, `__any_sync`/`__all_sync` or `warpSize`.
  - Warp operations appear only in `reduce.cu` and `quantized.cu`, all at logical width 32 (`WARP_SIZE 32`).
  - Safe on wave64 once the shim pins the width; to be proven on gfx942/gfx90a.
- `candle-rocm-kernels` compiles `affine binary cast conv fill indexing quantized reduce sort ternary unary`. The `mmq_gguf`/`mmvq_gguf`/`moe` static CUDA kernels are not compiled for ROCm in #3801 (equivalents in `quantized.cu`). cryptomilk adds an RDNA3/4-only WMMA build of `mmq_gguf` (`has_wmma`: gfx11/gfx12), loaded with `libloading`.
- `rocm-rs` 0.5 (`RustNSparks/rocm-rs`, MIT, pushed 2026-10-30): `build.rs` runs bindgen and sets `rustc-link-search={ROCM_PATH:-/opt/rocm}/lib`. This host has no `/opt/rocm` (pip SDK, `hipcc` HIP 7.16.26385), so `ROCM_PATH` must be set. There is no RCCL module.
- CUDA-visible edits that need the nvcc check:
  - #3801: `conv.cu` +106/-14, `quantized.cu` +90/-43, `unary.cu` +65/-0, and `candle-nn/src/moe.rs` +47/-300 (moved to `moe/cuda.rs`).
  - cryptomilk: `quantized/cuda.rs` +65/-10, `mmq_gguf/mmq_common.cuh`, and the ten `mmq_instance_*.cu`.
- Toolchain on this host: cargo 1.97.1, hipcc from the pip SDK. No GPU was used and nothing was built (planner role; the host is shared).
- `surface.json`: `surface.py generate` found 183 CUDA files and 0 components, because it does not read Cargo workspaces. I added 33 components by hand: backend, kernels crate, quantized, nn, transformers, pyo3, examples, features (`miopen`, `nccl`, `cutile`, `flash-attn`), the two flash-attn crates, test suites, benches, CPU regression, the CUDA no-regression build, the three gates and the pip-SDK ROCm root. None is covered or scoped out yet; the Q3/Q4 items get `scoped_out` reasons (or become work) once ruled.
