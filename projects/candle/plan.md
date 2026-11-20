# candle port plan

**Porter: do not start until jeff has ruled on the four questions in `notes.md`, section "Planning (2026-11-20)".** Questions 1 to 3 change what the porter builds. The plan below marks every step that depends on a ruling and gives the recommended answer, but the recommendation is not the ruling.

## Project

- Upstream: https://github.com/huggingface/candle, default branch `main`, analysed at `1a49dc27c1da7671e0ee12279ddf4ba619412856` (2026-11-19).
- Fork: https://github.com/AMD-Ecosystem/candle (only `main`, an unmodified mirror; no `moat-port` yet).
- Licence: MIT OR Apache-2.0 (intake).
- Planned on linux-gfx942 (MI300X, ROCm pip SDK, HIP 7.16.26385). No GPU was used and nothing was built.

## Existing AMD support and decision

**Improvable: a sound, unmerged third-party HIP backend exists. Validate and improve it; do not port from scratch.**

- https://github.com/huggingface/candle/pull/3801 "Feat/rocm backend" (xmiksay = Martin Miksanik; 52 commits, all his; +18491/-523, 123 files). Still OPEN, `mergeStateStatus` DIRTY, last commit 2026-09-11, last comment 2026-09-28 (a ping to the maintainers). No maintainer has reviewed it. It continues #3424 (n1gh7e0us, still open). Judgment: community and not AMD-official, but authoritative in the sense of `assess-existing-support.md`. It follows good practice: it shares the CUDA kernel sources through a header shim, gates everything behind `feature = "rocm"`, and has real test-suite evidence on RDNA.
- https://github.com/cryptomilk/candle/tree/asn-feat-rocm-backend (Andreas Schneider = cryptomilk, last push 2026-10-17). This is #3801's 52 commits **rebased** onto upstream `12bb5d1f` (5 commits behind today's main), plus 12 follow-up commits. cryptomilk asked on #3801 (2026-09-27) for these to be added to the PR; they have not been. The follow-ups that matter here are:
  - `76c28f73` makes the `__shfl*_sync` shims variadic, so an omitted width defaults to 32. Without it, upstream's #3600 `reduce.cu` (3-argument `__shfl_xor_sync`, lines 165, 177, 969 and 979) does not even preprocess under the #3801 shim, and a width falling back to HIP's `warpSize` would be wrong on wave64.
  - `addfbed2` (fast_sum_small), `dcc8c800` (`__floats2bfloat162_rn`), `f1ab8c04` (indexed_moe nwarps from k), `f30912fd` (missing `_rocm` test idents).
  - `e9a24ca0` adds an async host-to-device path. `7e59e82e` splits RDNA3 from RDNA4 detection.
  - `879c87ee`, `f40c946f`, `f426a423`, `63f0cdbb` and `148115b7` add RDNA3/4 WMMA MMQ through `candle-kernels/src/mmq_gguf`, compiled to a shared library at run time and loaded with `libloading`. They gate on `has_wmma()` (gfx11/gfx12 only); CDNA keeps the `quantized.cu` MMQ. These commits also edit CUDA-visible code: `candle-core/src/quantized/cuda.rs` (+65/-10), `mmq_gguf/mmq_common.cuh` and all ten `mmq_instance_*.cu` launchers.
- https://github.com/huggingface/candle/pull/4075 (cryptomilk, open) fixes an mmvq_gguf over-read on CUDA. #3801 carries the equivalent guard in `quantized.cu`.
- https://github.com/cryptomilk/candle-flash-attn-rocm is a CK-FMHA FlashAttention-2 forward crate for candle on ROCm, last pushed 2026-11-01. **It has no licence file**, so it is a hint only, never code to inherit.
- Smaller community efforts are non-authoritative and ignored: closed #3186 (AI-generated), vberthet/candle, Nu11ified/candle-rocm, vuongthai91/candle-rocm.
- Not checked: `amd/` and `AMD-AGI/` behind SAML SSO (intake's open item).
- Other open backend PRs (Vulkan #4110, SYCL #3974) do not touch the CUDA or ROCm path.
- Delivery shape: upstream merges backends; it does not link platform forks. A normal upstream PR is a valid vehicle in principle, but a sound competing PR already exists, which is why delivery is question 2.
- Performance kernels: the plan is a mechanical, correctness-first port. It reuses the CUDA sources via the shim for everything except attention. An AMD-native rewrite (an MFMA MMQ for CDNA like llama.cpp's, or wiring `__builtin_amdgcn_sdot4` for CDNA `__dp4a`) is optional later work, recorded as a deferral candidate, not part of this port. Flash attention is CUTLASS/CuTe and has no hipify path; its AMD-native target is CK FMHA (question 3).

## Build classification

**`cargo`: a Rust workspace. Neither Strategy A nor B.** This is a third build class, a backend implementation plus runtime-compiled kernels:

- `Cargo.toml` `[workspace] members` are candle-core, -datasets, -examples, -nn, -pyo3, -transformers, wasm and tensor-tools. candle-kernels, candle-flash-attn, candle-flash-attn-v3, candle-metal-kernels and candle-onnx are `exclude`d crates pulled in by path.
- CUDA today: `candle-core/Cargo.toml` feature `cuda` -> `cudarc` 0.19.10 (driver, cuBLAS, cuRAND, NVRTC-free PTX loading). `candle-kernels/build.rs` uses `cudaforge::KernelBuilder` to compile `src/*.cu` to PTX loaded at run time, and statically links the `mmq_gguf/`, `mmvq_gguf.cu` and `moe/` kernels.
- ROCm in #3801: `rocm` feature -> `rocm-rs` 0.5 (RustNSparks, MIT; bindgen over the ROCm headers, links `amdhip64`, `rocblas` and `rocrand`, plus `MIOpen` under `miopen`). It adds a new crate `candle-rocm-kernels` that `include_str!`s the shared `candle-kernels/src/*.cu`. At first use it compiles them with `hipcc --genco -O3 -std=c++17 -D__CUDA_ARCH__=890 [-DRDNA2|-DRDNA3]` for the detected `gfx*`, with `-include hip_shim/hip_compat.h` and shadow `cuda_fp16.h`/`cuda_bf16.h`/`cuda_fp8.h`/`cuda.h`/`cuda/std/limits`. The result is cached under `~/.cache/candle-rocm/<arch>-<rocm-version>/` (`CANDLE_ROCM_CACHE_DIR`). The `-D__CUDA_ARCH__=890` define is deliberate and documented in `candle-rocm-kernels/src/compile/cache.rs`.
- Recorded with `moatlib.py set-ext-type candle cargo`.

## Port strategy

The work order. Steps marked [Q1] or [Q2] depend on that ruling, and the recommended answer is shown.

1. **Base the fork branch [Q1, recommended (a)].** Create `moat-port` on the fork from #3801's head `fca7197d2853451172df7bfbce95bc3cc6f0dcc1`, keeping xmiksay's commits and authorship. **Merge** (do not rebase) upstream `main` (`1a49dc27` or newer). A trial merge in a scratch worktree on this host conflicts in only three files: `Cargo.toml` (0.11.0 -> 0.11.1 version bump vs the new `candle-rocm-kernels` line), `candle-core/src/quantized/mod.rs` (three hunks: #3697 aarch64 repack, #4068 Metal routed MoE vs the ROCm `QStorage` arms) and `candle-kernels/src/unary.cu` (#3600's `UNARY_OP_BF16_VEC` vs #3801's `ucopy_bf16`/`transpose2d_bf16`; keep both). Then cherry-pick cryptomilk's 12 follow-up commits (`76c28f73` .. `148115b7`), keeping his authorship. Each needs re-resolution only where it touches the same three files.
2. **Upstream drift since the PR base (`ddf1b879`, 21 commits).** After the clean auto-merge, check every surface #3801 multiplexes (`cuda-to-rocm` upstream-sync.md: a clean merge lands only the NVIDIA half):
   - #3600 `804d361e` added vectorised bf16 unary, cast and binary macros and about 300 lines of new reduction kernels (`reduce.cu`), with new dispatch in `cuda_backend/mod.rs`. The shared `.cu` changes must compile under the shim (3-argument shuffles: `76c28f73`). Decide whether the ROCm backend should dispatch the new fast paths (`fast_sum_small` is wired by `addfbed2`; check the rest against `cuda_backend/mod.rs`).
   - #3909 `e68b659f` gates the `__hmax_nan`/`__hmin_nan` shim in `compatibility.cuh` on `__CUDA_ARCH__`. Under `-D__CUDA_ARCH__=890`, make sure it neither redefines nor drops what `hip_shim/cuda_bf16.h` supplies.
   - #3944 `73e25aac` fixed the CUDA `copy_strided_src` over-copy. The ROCm `ops_copy.rs` already copies `elem_count`; the extended `slice_scatter` test proves it.
   - #4068 `cfcbd718` added Metal routed quantized MoE (`quantized/mod.rs`, `candle-transformers/src/fused_moe.rs`). Give the ROCm arm the same fallback CUDA has.
   - #4019 (scatter_set aliasing bail) and #3997 (index_select backward) are backend-neutral; their tests run on rocm.
3. **Wave64 (CDNA) correctness, the main differentiator: #3801 never ran on wave64.** Every shared kernel uses logical width-32 warp operations (`WARP_SIZE 32` in `reduce.cu` and `quantized.cu`; `__shfl_xor_sync(0xffffffff, v, m, 32)`). With the variadic shim, these are arch-agnostic (`faults-wavefront.md`, "Width-32 LOGICAL warp ops are arch-agnostic"). There is no `__ballot`, `__activemask` or `warpSize` use in the 11 runtime modules. Verify on gfx942 and gfx90a: reductions, softmax/rms_norm/layer_norm, MMVQ/MMQ/MoE, sort. Things to look for:
   - `launch.rs` occupancy constants written for wave32 (perf only).
   - `quantized/rocm/q8_1.rs` "one per wave32" comments that might hide a geometry assumption.
   - The `hip_shim/shim_test.hip` comment that contradicts `hip_compat.h` on `__syncwarp` (it says the shim does not define it; it does).
4. **fp8 on CDNA3.** `hip_shim/cuda_fp8.h` maps `__nv_fp8_e4m3` to HIP's OCP `__hip_fp8_e4m3`. gfx942's native fp8 is FNUZ, so OCP conversion must be the software path there. Run `tests_f8e4m3.rs` on gfx942 and gfx90a, and on gfx950 if a host offers it (OCP hardware).
5. **ROCm root without `/opt/rocm`.** `rocm-rs`'s `build.rs` defaults `ROCM_PATH=/opt/rocm` for bindgen and linking. The runtime compiler wants `hipcc` on PATH and `clang-offload-bundler` under `$ROCM_PATH`. This host (pip SDK) and the Windows TheRock SDK have no `/opt/rocm`. First try the documented route, `ROCM_PATH=$(rocm-sdk path --root)`. If it does not work, the fix goes in `candle-rocm-kernels` (resolve the bundler next to `hipcc`). A fix needed in `rocm-rs` itself is a third-party change: record it and bring it to jeff (`feedback_rocm_build_just_works`: no README-only workarounds).
6. **CUDA no-regression [required before presenting].** #3801 edits shared CUDA sources, and its author says they were never nvcc-compiled: `quantized.cu` (+90/-43), `unary.cu` (+65), `conv.cu` (+106/-14, chunked im2col). cryptomilk's commits edit `mmq_gguf/*` and `quantized/cuda.rs`. `candle-nn/src/moe.rs` moved its CUDA code to `moe/cuda.rs`. On the mi355 host (nvcc):
   - `CUDA_COMPUTE_CAP=80 cargo build -p candle-kernels`, then `CUDA_COMPUTE_CAP=90`.
   - `cargo check -p candle-core -p candle-nn -p candle-transformers -p candle-examples --features cuda`.
   - `cargo check ... --features cudnn`.
   - Diff the PTX of every module the port did not touch against upstream main (`validation.md`, "Diff the PTX").
   Fix any break in the same round.
7. **Small ROCm arms the PR missed [include; Q4 asks only to confirm].**
   - `candle-transformers/src/models/qwen3_moe.rs:225`: `cfg!(feature = "cuda")` picks FusedMoe. Add the rocm arm if `candle-nn` moe/rocm covers `FusedMoe`, and test with `quantized-qwen3-moe`.
   - `examples/custom-ops`: add a `rocm_fwd` using `get_or_load_custom_func`.
   - `quantized`/`mistral`/`gemma4`/`flux` examples: `set_force_dmmv` has a ROCm counterpart only if `quantized::rocm` exposes one. Do not invent one.
8. **Windows (windows gate, last per `feedback_linux_before_windows_validation`).** There is no evidence at all: #3424's gfx1151 run was Linux. Expected work:
   - `rocm-rs` bindgen needs libclang and the TheRock include/lib layout (`amdhip64.lib`, `rocblas.lib`, `rocrand.lib`).
   - Runtime compile through `hipcc.exe` (see `windows.md`, "hipcc.exe and hiprtc traps").
   - `clang-offload-bundler.exe` path, and the cache path via `dirs`/`fslock`.
   - The `mmq_gguf` shared library (`.dll` rather than `.so`, loaded by `libloading`).
   Put workarounds in the build files, guarded to Windows.
9. **Out of scope unless jeff rules otherwise [Q3, Q4].** These are candle-flash-attn (FA2), candle-flash-attn-v3 (FA3), the `flash-attn` features, `cutile`, `nccl`/RCCL and `llama_multiprocess`. They stay uncovered in `surface.json` until the ruling, and are then recorded as `scoped_out` (with reason) or as work.
10. Do not add GitHub Actions. #3801 itself carries `.github/workflows/ci_rocm.yaml`, a GPU-free check job. It is the PR author's and stays as is; fork Actions remain disabled.

## CUDA surface inventory

`surface.py` census at `1a49dc27`: 183 CUDA files, 215 `__global__`, 699 device functions, 40 warp-intrinsic sites, 99 launches; libraries `cutlass` 519 (all in the flash-attn crates), `cub` 1, `thrust` 4.

| area | files | ROCm path | status |
|---|---|---|---|
| shared kernels | `candle-kernels/src/{affine,binary,cast,conv,fill,indexing,quantized,reduce,sort,ternary,unary}.cu` + `cuda_utils.cuh`, `compatibility.cuh`, `binary_op_macros.cuh` | same sources via hipcc + `hip_shim` (#3801) | covered by #3801; #3600 drift and the 3-argument shuffles need cryptomilk's `76c28f73` |
| GGUF MMQ | `mmq_gguf/*.cu(h)` (MMA PTX for Turing+, plus llama.cpp RDNA WMMA arms) | RDNA3/4: shared library (cryptomilk); others: `quantized.cu` MMQ with the AMPERE tiles | WMMA only on wave32, so check that CDNA takes the fallback |
| MMVQ / MoE static kernels | `mmvq_gguf.cu`, `moe/{moe_align,moe_gguf,moe_wmma,moe_wmma_gguf}.cu` (`nvcuda::wmma`) | `quantized.cu` MMVQ + `indexed_moe_forward` (#3801 `quantized/rocm/{mmvq,moe}.rs`) | not hipified; parity via tests |
| backend (Rust) | `candle-core/src/cuda_backend/*` (cudarc) | `candle-core/src/rocm_backend/*` (rocm-rs) | #3801 |
| BLAS | cuBLAS / cuBLASLt (cudarc) | rocBLAS (`rocm_backend/gemm.rs`, precision knobs) | #3801 |
| RNG | cuRAND | rocRAND (`rocm_backend/rng.rs`) | #3801 |
| conv | cuDNN (`cudnn`) | im2col+GEMM default (256 MB chunk cap), MIOpen under `miopen` | #3801; the im2col memory cliff was reported on gfx1201 and mitigated by the chunked path |
| sort | `sort.cu` bitonic | same source | #3801 |
| collectives | NCCL (`nccl` feature, cudarc) | none (rocm-rs has no RCCL) | Q4 |
| cuTile | `cutile` feature | no AMD target | Q4 |
| attention | `candle-flash-attn` (FA2, CUTLASS sm80, 54 files), `candle-flash-attn-v3` (FA3 sm90 TMA/wgmma, 67 files) | CK FMHA / aotriton reimplementation | Q3 |
| examples | `custom-ops` kernel (`layernorm_kernels.cu`) | `rocm_fwd` via the custom-kernel API | step 7 |

## Risk list

- **wave64**: logical width-32 operations are safe only with the variadic shim. Any 3-argument `*_sync` reaching HIP's own overload uses width 64, giving wrong reductions on CDNA. `76c28f73` covers it; the risk returns whenever upstream adds a new call form.
- **Upstream drift**: kernels upstream adds later reach the ROCm build automatically (shared sources), but their Rust dispatch does not. Re-check `cuda_backend/mod.rs` against `rocm_backend` on every sync.
- **fp8 encoding on gfx942** (OCP vs FNUZ): see step 4.
- **Runtime compile**: the first run compiles about 11 modules (about 70 s for `quantized`). Validators must count that time and keep a persistent `CANDLE_ROCM_CACHE_DIR`. A stale cache is keyed by source, headers, flags, arch and toolchain (`2b9abd98`, `248848f4`), so clearing it is never the fix for a wrong result.
- **`__CUDA_ARCH__=890` define** feeds every numeric gate in the shared sources (`build-headers-macros.md`, "Defining `__CUDA_ARCH__` for the HIP device pass also feeds every numeric gate"). The PR audited it for fp8 entry points; re-audit after #3600 and #3909.
- **`rocm-rs` third-party dependency**: build and link paths (`/opt/rocm` default, unconditional bindgen) and Windows support are outside candle. Fixes there need a separate upstream contribution.
- **Uninitialised memory**: ROCm does not zero fresh allocations. Check `alloc.rs` caching, and the first use of every accumulator kernel (`atomicAdd` in `reduce.cu` sum and `index_add`, scatter_add).
- **CUDA regressions** in the shared `.cu` and `quantized/cuda.rs`: step 6.
- **Delivery conflict**: shipping a parallel PR against #3801 repeats the gsplat mistake. Q2.

## File-by-file change list

The porter's own edits on top of the base, beyond the conflict resolutions in step 1:

- `candle-rocm-kernels/src/hip_shim/hip_compat.h`, `cuda_bf16.h`: whatever #3600/#3909 still need after `76c28f73`/`dcc8c800` (expect little).
- `candle-rocm-kernels/src/compile/{hipcc,detect,cache}.rs`: ROCm-root resolution (step 5); Windows tool names and paths (step 8).
- `candle-core/src/rocm_backend/{ops_reduce,ops_elementwise,ops_cast,mod}.rs`: dispatch parity with #3600 where it matters for correctness.
- `candle-core/src/quantized/mod.rs`, `candle-transformers/src/fused_moe.rs`: ROCm arm for routed MoE after #4068.
- `candle-transformers/src/models/qwen3_moe.rs`: rocm arm for FusedMoe (step 7).
- `candle-examples/examples/custom-ops/main.rs` (+ `build.rs` only if the kernel source must be embedded): `rocm_fwd`.
- Any wave64 fix found in validation: in the shared `.cu` only if it is correct on both widths and leaves nvcc output unchanged; otherwise in the shim.
- `README.md` / `candle-rocm-kernels/README.md`: the pip-SDK `ROCM_PATH` line and Windows, once they work.

## Build commands

Linux (any gfx; pip SDK shown, `/opt/rocm` hosts drop the export):
```
export ROCM_PATH=$(rocm-sdk path --root)     # pip SDK hosts only
export CANDLE_ROCM_CACHE_DIR=$PWD/agent_space/candle-rocm-cache
cd projects/candle/src
cargo build --release -p candle-core -p candle-nn -p candle-transformers --features rocm
cargo build --release -p candle-core --features miopen
make rocm-shim-test                          # every shared module through hipcc for this arch + shim_test.hip
```
CUDA no-regression (mi355 host): step 6. Windows: to be derived by the Windows round; record the exact commands in notes.

## Test plan

GPU (each platform, with a persistent kernel cache; `--test-threads=1` as the PR's Makefile does, since first-use compiles serialise):
```
cargo test -p candle-core --features rocm -- rocm --test-threads=1              # in-crate rocm_backend + quantized/rocm tests and every test_device! _rocm variant
cargo test -p candle-core --features rocm --tests -- rocm --test-threads=1      # integration suites: tensor, conv, grad, matmul, pool, quantized, layout, bilinear, indexing, custom_op, gguf
cargo test -p candle-nn --features rocm -- --test-threads=1
cargo test -p candle-transformers --features rocm
cargo test -p candle-rocm-kernels
cargo test -p candle-core --features miopen -- rocm --test-threads=1            # MIOpen conv path
make rocm-shim-test
```
Model smoke tests (real kernels end to end; HF hub downloads):
```
cargo run --release --features rocm --example quantized-qwen3 -- --which 0.6b --prompt "The capital of France is" --sample-len 32
cargo run --release --features rocm --example quantized-qwen3-moe -- <smallest option>   # MoE path, after step 7
cargo run --release --features rocm --example bert -- --prompt "hello world"
```
Compare each generation against the CPU build of the same command with greedy sampling (`--temperature 0` where the example has it). Token streams should match for the first tokens; record any divergence point.

Counting: record pass/fail/ignored per suite. A green run with `0 passed` for the `rocm` filter means the filter matched nothing, which is not a pass. Count `_rocm` tests run vs the number of `test_device!` instantiations.

Non-GPU regression set (must stay identical to upstream main on the same host):
```
cargo test -p candle-core -p candle-nn -p candle-transformers
cargo test -p candle-core --features rocm -- --skip rocm     # CPU tests in a rocm-featured build
```
Coverage: wave64 (gfx942 and/or gfx90a; gfx950 extra), wave32 (gfx1100/gfx1201), windows (gfx1151), each at the same `head_sha`.

## Open questions

The four questions for jeff are in `notes.md`, "Planning (2026-11-20)". Planner-level unknowns the porter resolves:
- Does `rocm-sdk path --root` satisfy `rocm-rs` bindgen and the runtime bundler lookup on this host class? (step 5)
- Does `candle-nn` moe/rocm cover `FusedMoe` for `qwen3_moe`?
- Which #3600 fast paths are correctness-relevant for the ROCm dispatch, and which are only performance?
