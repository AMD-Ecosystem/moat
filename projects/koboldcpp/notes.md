# koboldcpp notes

## Intake screen (2026-11-05, windows-gfx1151 host)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict partial, AMD path upstream). Screened at upstream `concedo` @ 6d2fedf8624e5f05926af3191f9efdcb13c4133c (shallow clone in `agent_space/koboldcpp-screen`).

### Licence

- `python3 utils/licenses.py check LostRuins/koboldcpp` -> AGPL-3.0, tier 3 (strong copyleft; contributing needs a person's approval). LICENSE.md is the AGPL-3.0 text. Recorded with `set-license-spdx`.
- `python3 utils/licenses.py scan-nvidia agent_space/koboldcpp-screen` -> no NVIDIA proprietary licence text.
- Vendored ggml/llama.cpp and stable-diffusion.cpp are MIT upstream; not a blocker for a decline.

### CUDA surface

- All CUDA lives in the vendored ggml CUDA backend (`ggml/src/ggml-cuda/`, the llama.cpp backend with its mature HIP path) plus three koboldcpp-specific legacy files: `otherarch/ggml_v2-cuda.cu`, `otherarch/ggml_v2-cuda-legacy.cu`, `otherarch/ggml_v3-cuda.cu` (old GGML v2/v3 model formats).
- No other `.cu`/`.cuh` anywhere in the tree.

### Existing AMD support (upstream, first party)

- Makefile: `make LLAMA_HIPBLAS=1` builds `koboldcpp_hipblas` with `-DGGML_USE_HIP -DGGML_USE_CUDA -DGGML_HIP_NO_VMM`, auto-detects targets via `amdgpu-arch` (default list gfx803..gfx1201), links hipBLAS/rocBLAS, and has both `.so` and `.dll` build rules.
- CMakeLists.txt: `LLAMA_HIPBLAS` option builds ggml-rocm plus `ggml-v2-rocm`, `ggml-v3-rocm`, `ggml-v2-legacy-rocm` -- i.e. the koboldcpp-specific legacy `.cu` files are also compiled for HIP.
- README build table: "AMD ROCm | `make LLAMA_HIPBLAS=1`"; also links the AUR `koboldcpp-hipblas` package.
- CI: `.github/workflows/kcpp-build-release-linux-rocm.yaml` builds in `rocm/dev-ubuntu-22.04:7.14.0-full` and uploads `koboldcpp-linux-x64-rocm` to the `rocm-rolling` release (last asset update 2026-10-30, matching v1.124.2 of 2026-10-31).
- Merged upstream PRs maintaining it: #1823 "ROCm 7.1 CI" (2025-11), #2349 "AMD ROCm version bump for the binary" (2026-07). No open HIP/ROCm PRs.

### Community fork

- YellowRoseCx/koboldcpp-rocm (780 stars): last release v1.104.yr0-ROCm 2025-12-30, last push 2026-07-29; about 20 releases behind upstream v1.124.2. README itself says "check its maintenance status". Non-authoritative and now largely superseded by upstream's own ROCm build. Its historical value was Windows ROCm binaries.
- No ROCm or AMD-Ecosystem repo for koboldcpp (`gh search repos koboldcpp --owner ROCm --owner AMD-Ecosystem` -> none). The `amd` / `AMD-AGI` orgs were not searched (SAML SSO for this token), so a duplicate check there is still open, though irrelevant to a decline.

### Gap

Only packaging: upstream ships no Windows ROCm binary (Windows AMD users are pointed at Vulkan) and the README's GPU acceleration line still says `--usecuda` is "Nvidia Only" without mentioning the Linux ROCm rolling binary. The Makefile already has the Windows `.dll` HIP rule, and llama.cpp itself ships Windows HIP builds, so this is release-engineering for the maintainer rather than a CUDA-to-HIP port. There is no unported CUDA code.

### Recommendation

Decline, `SKIP_REASON` `already-supported`: upstream koboldcpp builds its whole CUDA surface (vendored ggml plus its own legacy v2/v3 `.cu` files) with HIP via `LLAMA_HIPBLAS`, and publishes a CI-built Linux ROCm binary at each release. The residual Windows-binary/docs gap is packaging, not porting, and on a tier-3 (AGPL-3.0) project.

## Ruling: validation project (Jeff Daily, 2026-11-10)

Jeff Daily overrode the intake decline: koboldcpp becomes a validation project. The port is not stale upstream, but the only ROCm fork is a stale community one (YellowRoseCx/koboldcpp-rocm) and AMD has no fork. Validate upstream's own HIP build (LLAMA_HIPBLAS) on every required gate (wave64, wave32, Windows) and fix what breaks. The fixes go on `moat-port` of an AMD-Ecosystem fork. A person creates that fork, never an agent; the project waits in `awaiting-fork` until it exists, then `release-forks` moves it to `screened` for the planner.

Licence: AGPL-3.0 is tier 3. Validation and fork-side fixes need no clearance. Anything offered upstream needs a licence clearance recorded by a person (`record-license-clearance`) first.

## Planning (2026-11-20, linux-gfx950 fleet worker)

### Decisions Jeff must make before a porter starts

The planner could not get answers during this headless job, so nothing below is assumed. The plan states a recommendation for each; the porter must not start until Jeff has ruled on all three.

**Q1. Which Windows route meets the `windows` gate?** Upstream documents no Windows ROCm build and points Windows AMD users at Vulkan. Two undocumented routes exist in the tree, and both look broken as written (plan.md R1, R2).
- (a) **CMake route (recommended).** Fix the existing `LLAMA_HIPBLAS` MSVC branch of `CMakeLists.txt`: `ggml-rocm` is built from `GGML_SOURCES_CUDA` instead of `GGML_SOURCES_ROCM` (line 217); ROCm is found by globbing `C:/Program Files/AMD/ROCm/` (the legacy HIP SDK, not TheRock); the clang toolchain setup is missing. Add TheRock discovery and Windows DLL loading in `koboldcpp.py`, and document it. Cost: edits to a file whose header calls it "ONLY for CUBLAS builds on Windows". But it is the route upstream's Windows CUDA build already uses.
- (b) **Makefile route.** Use `make LLAMA_HIPBLAS=1` under w64devkit with TheRock hipcc. Expected to fail at link: MinGW g++ objects meet MSVC-ABI hipcc objects, and lld-link rejects `-Wl,-rpath`. Making it work means changing how the whole library is compiled on Windows.
- (c) **Waive the `windows` gate** and validate Linux only. This needs a maintainer-approved waiver, and Windows AMD users stay on Vulkan as upstream intends. It departs from the 2026-11-10 ruling, which named Windows.

**Q2. Which branch is the PR base, and which branch does `moat-port` start from?** `status.json` records `fork_default_branch: "main"`, which exists neither on the fork nor upstream. The fork carries only `concedo`. Upstream merges PRs into `concedo_experimental`: all 30 most recent merged PRs target it, including the two ROCm CI PRs #1823 and #2349. It is 634 commits ahead of `concedo`. The HIP blocks of the Makefile and CMakeLists are identical on both branches.
- (a) **`concedo_experimental` (recommended).** Mirror upstream's `concedo_experimental` to the fork, run `python3 utils/moatlib.py set-base-branch koboldcpp concedo_experimental --by <who>`, and cut `moat-port` from it.
- (b) **`concedo`.** Run `set-base-branch koboldcpp concedo --by <who>`; `moat-port` starts from the release branch. Any upstream PR would then need retargeting.
- Either way the wrong `main` record needs a person's `set-base-branch` first. Until then `jargon.py --port`, review-PR and classify base resolution have nothing to resolve.
- Separately, per standing practice, Jeff may set `moat-port` as the fork's GitHub default branch once it exists.

**Q3. How deep must each modality be validated?** Nothing is cut from the build: text, image (sdcpp), whisper, TTS, music (ACE-Step), embeddings and vision projectors all run on the one ggml HIP backend inside `koboldcpp_hipblas`. The question is the depth of evidence each platform owes.
- (a) **Recommended.** Gate on T1 (llama.cpp `test-backend-ops` against koboldcpp's ggml) and T2 (greedy HIP-vs-CPU text generation: mmq, nommq, BF16, FlashAttention with `--quantkv`, legacy GGML v2/v3). Run one T3 smoke per platform for image, whisper, TTS, embeddings and vision. Record music and the two-GPU split as `owed` with a reason where the hardware or model is missing.
- (b) Gate every modality with a full differential run on every platform.
- (c) Gate text only, and record the other modalities as owed. They would then block `pr-ready` until done.

Not blocking a porter: the licence is AGPL-3.0 (tier 3), so a person must record a licence clearance before anything is offered upstream.

### What planning found

- Build type is `make` (recorded with `set-ext-type`). The vendored ggml CUDA backend matches llama.cpp aa588a3463357ee990b314f41dda242869130704 (2026-10-30, the second parent of 4f78d791) on `concedo` 6d2fedf8. koboldcpp's GPU-relevant patches to it are in plan.md, "CUDA surface inventory".
- The Linux HIP build is upstream-maintained and released (rocm-rolling, ROCm 7.14 container). The likely Linux finding (R4): the `LLAMA_PORTABLE` default `GPU_TARGETS` list lacks gfx950, gfx1150 and gfx1151. Because the CI container has no GPU, the published binary should have no code object for those GPUs. The porter must check this first on the rolling binary with `roc-obj-ls` or `llvm-objdump --offloading`.
- Shared-code oddity to exercise, not to "fix for HIP" (R6): the koboldcpp pool allocator subtracts the requested `size` instead of the freed `b.size` from `pool_size` on the nommq largest-buffer free path in `ggml-cuda.cu`.
- No GPU test suite upstream. The T1/T2 harnesses are validator-side only and are never committed to the fork. T1's overlay recipe is in plan.md.
- `surface.json`: the generated floor (CMake-derived, from a file upstream marks unsupported) plus 27 planner-added components: the Makefile target per OS, GPU target list, ROCm discovery, packaging, the Python loader, the four CUDA areas, every modality, the tests and the README. CPU-ISA options are scoped out with reasons. The CMake `ggml-*-rocm` library entries stay unaccounted until Q1 is ruled.

## Rulings on the planner's questions (Jeff Daily, 2026-11-23)

- Q1 (Windows route): **a** (ruled later the same day). Fix the CMake `LLAMA_HIPBLAS` MSVC branch: build `ggml-rocm` from the ROCm source list, discover the current ROCm SDK (TheRock) instead of only the legacy `C:/Program Files/AMD/ROCm/` path, add the clang toolchain setup, add Windows DLL loading in `koboldcpp.py`, and document it. The first porter job was dispatched before this ruling and does Linux only; the Windows route is a follow-up porter round. No waiver.
- Q2 (PR base): **a**, `concedo_experimental`. Upstream's branch was mirrored unmodified to the fork at 0382471c8b72474b45537280404606d29d5f2cae and `set-base-branch koboldcpp concedo_experimental --by jeffdaily` was run on 2026-11-23. `moat-port` is cut from it.
- Q3 (validation depth): **a**. Gate on T1 and T2; one T3 smoke per platform for image, whisper, TTS, embeddings and vision; music and the two-GPU split recorded `owed` with a reason where the model or hardware is missing.
- Standing ruling of the same day: Windows laptop GPU runs go last across all ports; they are queued, not started.
- Still open for a person: the AGPL-3.0 licence clearance before anything is offered upstream.

## Porter round 1 (2026-11-23, linux-gfx942 fleet worker)

### For a person (open questions; nothing below was assumed)

1. **Q1 (Windows route) was ruled (a) while this round ran** (Jeff, the 2026-11-23 rulings above; the branch commit landed mid-job). As that ruling says, this round did Linux only: no Windows route was built or edited, no waiver was assumed, and every Windows/CMake surface entry stays `owed` for the follow-up Windows porter round. What this round learned for that round:
   - R1 still holds at fork head `0fff6ace6` (base `0382471c`): `CMakeLists.txt:217` is `add_library(ggml-rocm ${GGML_SOURCES_CUDA})`, and lines 219-227 append sources to `GGML_SOURCES_ROCM` after the target already exists, so even the ROCm list is never used. `option(LLAMA_CUBLAS ... ON)` (line 39) still defaults on, and ROCm is still found with the PowerShell glob of `C:/Program Files/AMD/ROCm/` (line 181). Option (a) means rewriting this block; it is not a one-line fix.
   - The three fixes on `moat-port` are route-independent: the hipBLAS compute-type fix (legacy GGML v2) applies to any HIP build, including Windows; the rocminfo regex fix matters on Windows too, because TheRock ships `rocminfo` for Windows since rocm-systems #9598 (2026-08-07), and its output uses the same `N (0xH) KB` format; the gfx950 target only affects `LLAMA_PORTABLE` Makefile builds.
   - The Makefile's ROCm discovery already handles a TheRock pip SDK on Linux: with `ROCM_PATH=$(rocm-sdk path --root)` it finds `bin/hipcc`, `bin/hipconfig`, `lib/llvm/bin/amdgpu-arch` and `lib/`, with no other setting. Option (b) (w64devkit + Makefile) inherits that discovery, but R2 (MinGW g++ objects linked by an MSVC-ABI hipcc, `-Wl,-rpath=` rejected by lld-link) was not tested here, since that needs Windows.
   - R3 is unchanged: `koboldcpp.py:1091` adds only `%HIP_PATH%\bin` to the DLL search path, and a TheRock pip SDK does not set `HIP_PATH`.
2. **CUDA no-regression compile is owed on the mi355 host (real nvcc).** This host has no nvcc. The only CUDA-compiled files changed are `otherarch/ggml_v2-cuda.cu` and `otherarch/ggml_v2-cuda-legacy.cu`, and only inside their `#if defined(GGML_USE_HIP)` blocks. With `#include` lines stripped, `cpp -P -x c++` (no `GGML_USE_HIP`) gives byte-identical output for base and port (sha256 prefix `b305bd90fc51e312` and `4cfd311705246476` for both), while the HIP preprocessed output differs, as intended. The Makefile change is inside `ifdef LLAMA_HIPBLAS`, and `koboldcpp.py` changes only the rocminfo (AMD) branch. This is supporting evidence, not the nvcc compile the standing rule asks for before presenting.
3. **Shared-code finding R6 (not fixed, per plan): the legacy pool's size counter underflows, and its destructor asserts on it.** `ggml_cuda_pool_leg::alloc` (`ggml-cuda.cu:607-616`, a koboldcpp patch) zeroes `b.size` and then does `pool_size -= size` with the *requested* size, not the freed buffer's size. A temporary trace (reverted, never committed) in a nommq `--benchmark` run hit the path 4 times; the first was `requested=8716288 freed=1686016 pool_size_before=1748224`, so the `size_t` counter wraps below zero and stays off by `requested - freed`. `~ggml_cuda_pool_leg()` asserts `pool_size == 0`, so destroying the backend context after this path has run should abort. The destructor did not run at process exit here (no trace line), so this was not observed as a crash. On HIP this pool is the only pool (`GGML_HIP_NO_VMM`); on CUDA it is used only without VMM. The fix would be `pool_size -= b.size` before `b.size = 0`. That is a koboldcpp shared-code bug, so telling upstream is a person's call (AGPL clearance still open).
4. Licence clearance (AGPL-3.0) is still open before anything is offered upstream (unchanged).

### What R4 found (upstream's published binary)

`koboldcpp-linux-x64-rocm` from the `rocm-rolling` release (asset updated 2026-11-23T14:33Z, 577,818,120 bytes, sha256 `4049d047...24a7fd`) is a PyInstaller one-file build. Extracted with PyInstaller 6.22.3's `CArchiveReader` (no execution), then `llvm-objdump --offloading koboldcpp_hipblas.so`. ggml code objects are present for gfx803, gfx900, gfx906, gfx908, gfx90a, gfx942, gfx1100, gfx1101, gfx1102, gfx1151, gfx10-1-generic, gfx10-3-generic and gfx12-generic (144 each). **There are none for gfx950 (MI350/MI355), gfx1103, gfx1150, gfx1152 or gfx1153**, while the bundled rocBLAS `library/` ships Tensile files for gfx908, gfx90a, gfx942, gfx950, gfx1010-1012, gfx1030-1036, gfx1100-1103, gfx1150-1153, gfx1200, gfx1201 and gfx1250. (Also noticed, not a port item: gfx803/gfx900/gfx906 have ggml code objects but no rocBLAS kernels in the bundle.) Upstream added gfx1151 and the generic targets today (`ee4727c64`, "try to optimize rocm", 2026-11-23), after the planner read `concedo`. The port adds gfx950 only (CDNA4; `vendors/hip.h` already maps it). gfx1103 and gfx1150/1152/1153 are RDNA3/3.5 iGPUs that upstream's own comment chose to keep specialized and limited to gfx1151. Adding them costs compile time and binary size, which is the maintainer's trade-off, so they are noted here and not added.

### Environment

- Host: AMD Instinct MI300X in a CPX partition (24,574 MiB per device, gfx942:sramecc+:xnack-), `HIP_VISIBLE_DEVICES=1` as pinned by the dispatcher, `ulimit -c 0` and `timeout` on every GPU run.
- ROCm: TheRock 10.2.0 SDK (HIP 7.17.26434) installed into the job venv: `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[libraries,devel]==10.2.0" "rocm-sdk-device-gfx942==10.2.0"`. `R=$($FLEET_VENV/bin/rocm-sdk path --root)` is `.../site-packages/_rocm_sdk_devel`. No system ROCm (`/opt/rocm` absent).
- gcc 13.3.0 (Ubuntu 24.04), `make -j96`.

### Build (fork `moat-port` at 0fff6ace6, base `concedo_experimental` 0382471c)

```
make -j96 LLAMA_HIPBLAS=1 ROCM_PATH=$R          # GPU_TARGETS auto-detected: gfx942; 48 s; koboldcpp_hipblas.so 117 MB
make -j96 LLAMA_HIPBLAS=1 LLAMA_PORTABLE=1 LLAMA_HIP_GENERIC_TARGETS=1 LLAMA_HIP_OFFLOAD_COMPRESS=1 ROCM_PATH=$R koboldcpp_hipblas
                                                 # release recipe flags, in a separate worktree: 581 s, 137 MB, 14 code objects incl. gfx950
```

Warnings only: `-s` unused for the hipcc compile steps (149), `loop not unrolled` (device), `-Winvalid-noreturn`, and pre-existing host format warnings. The RUNPATH points at the SDK's `lib/`, and `ldd` resolves every ROCm library from the SDK.

### Fixes on `moat-port` (4 commits)

| commit | change | evidence |
|---|---|---|
| 92a47669c | `otherarch/ggml_v2-cuda.cu`, `ggml_v2-cuda-legacy.cu`: in the HIP >= 6.5 branch map `CUBLAS_COMPUTE_32F_FAST_16F` to `HIPBLAS_COMPUTE_32F` (comment on the same line, so CUDA line numbers stay put) | Before: a GGML v2 model's first >=32-token prompt aborted the server with `cuBLAS error 7 at otherarch/ggml_v2-cuda.cu:672`. A standalone reproducer (hipblasGemmEx FP16 x FP16 -> FP32, OP_T/OP_N, 64x48x32) gives `HIPBLAS_COMPUTE_32F status=0`, `HIPBLAS_COMPUTE_32F_FAST_16F status=7 (HIPBLAS_STATUS_NOT_SUPPORTED)`. After: both v2 layouts are 64/64 identical to the CPU |
| c128de20e | `Makefile`: gfx950 in the `LLAMA_PORTABLE` list | R4 above; the portable build carries gfx950 |
| aa98b0c58 | `koboldcpp.py:2167`: rocminfo size regex accepts `N (0xH) KB` | rocminfo from ROCm 10.1 (on PATH) and 10.2 both print `25163776 (0x17ff800) KB` (rocm-systems `rocminfo.cc` `int_to_string` writes `dec << " "` then `"(0x"`). Before: `Unable to detect VRAM.`, no auto backend, no layer estimate. After: `Detected AMD GPU VRAM from rocminfo: [('AMD Instinct MI300X', '24574'), ...]`, `Auto Recommended GPU Layers: 25` |
| 0fff6ace6 | README: GPU_TARGETS / LLAMA_PORTABLE / ROCM_PATH (pip SDK) sentence under the build table; `--usecuda` "(Nvidia, or AMD with a ROCm build)" | documented commands were the ones used here |

`python3 utils/jargon.py --port koboldcpp`: clean. `utils/prose.py` on all four messages: clean.

### T1: llama.cpp `test-backend-ops` against koboldcpp's ggml

- llama.cpp `a8f8294c` (second parent of `8b5f1208`, "Merge branch 'upstream'", 2026-11-23) is in the fork's own history; two worktrees of it. Overlay: koboldcpp's `ggml/include/`, `ggml/src/ggml-cuda/` (llama.cpp's `ggml-cuda/CMakeLists.txt` restored), `ggml/src/*.c *.cpp *.h`, except `ggml-backend-reg.cpp`, which was kept pristine because koboldcpp's copy inlines `dl_error()` and collides with llama.cpp's `ggml-backend-dl.cpp`. koboldcpp's `ggml.c` needs `-I ggml/src/ggml-cpu`. The FA instances must match koboldcpp's five pairs: `-DGGML_CUDA_FA_QUANTS=q4_0-q4_0,q5_1-q5_1,q8_0-q8_0,bf16-bf16` (otherwise the q5_1 vec instances are missing at link). The CPU backend is llama.cpp's.
- `cmake -G Ninja -DGGML_HIP=ON -DGPU_TARGETS=gfx942 -DGGML_HIP_NO_VMM=ON -DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$R -DROCM_PATH=$R ...; cmake --build . --target test-backend-ops`; run `test-backend-ops test -b ROCm0`.
- **Overlay: 20806/20806 passed**, 7209 cases not supported (OUT_PROD 1472, FLASH_ATTN_EXT 1405, CONV_2D 1138, IM2COL_3D 1024, SET_ROWS 607, ...), 603 s. **Pristine: 20806/20806 passed**, the same 7209 not supported, 610 s. FA with K=V types f16/q8_0/q4_0/q5_1/bf16 passed 1925/374/349/336/343 cases (hsk=64 included).

### T2: greedy HIP vs CPU (Qwen2.5-0.5B-Instruct GGUFs from Qwen/Qwen2.5-0.5B-Instruct-GGUF; BF16 made with `quantize_gguf ... BF16` from the f16 file)

Harness: a koboldcpp server per config, `/api/v1/generate` with `top_k 1, rep_pen 1, seed 42, max_length 64`, two prompts (5 tokens; 120-token paragraph), `--gpulayers 99`, compared with the same flags under `--usecpu`. Same results on two runs (GPU deterministic). Rerun at the head (0fff6ace6) after a clean rebuild:

| config | p0 | p1 |
|---|---|---|
| Q4_K_M mmq / nommq | diverge@19 / @28 | diverge@21 / identical |
| Q8_0 mmq / nommq | identical / identical | identical / diverge@0 |
| BF16 mmq, BF16 nommq, F16 nommq | identical | identical |
| Q4_K_M FA off | diverge@19 | identical |
| Q4_K_M `--quantkv` q8_0 / bf16 | diverge@19 / @19 | identical / identical |
| Q4_K_M `--quantkv` q5_1 / q4_0 | diverge@5 / @1 | diverge@7 / @1 |
| pythia-160m GGML v3 Q4_0 / Q5_1 | identical / diverge@13 | diverge@7 / diverge@21 |
| pythia-160m GGML v2 quant-v1 / quant-v0 Q4_0 | identical / identical | identical / identical |

Every divergence was checked for a near-tie: the top-3 at the divergence point from both backends, and for legacy v3 at the actual decode step (temperature 0.01, top_k 5, log-prob gaps rescaled). The f16/q8_0/bf16-KV and mmq/nommq cases have top-2 gaps of 0.01-0.11 nats on both sides. The v3 Q4_0 decode step has a 0.026-nat gap (' who' vs ' then'). The q5_1/q4_0 KV cases are not ties at the token level (CPU and GPU disagree by up to 1.3 nats), so perplexity decided them (`llama-perplexity`, koboldcpp ggml overlay, `-c 512 -fa on`, LICENSE.md+README.md as corpus, `-ngl 99` vs `--device none`): f16 5.8775/5.8995, q8_0 6.1851/6.1954, q5_1 26.11/27.16, q4_0 108.64/108.34 (GPU/CPU). The backends agree; q4_0/q5_1 KV wrecks this 2-KV-head model on either one. No HIP defect.

`--benchmark` (Q8_0, ctx 4096): nommq 27,184 T/s prompt, 331 T/s generation; mmq 22,834 / 327 T/s. No `--quantkv` value aborted with FA on (R7).

R5 (BF16 compute-type revert) is not exercised on gfx942, which has fast BF16; the BF16 rows above cover the plain path. A gfx90a or gfx1030 run is needed for the revert path.

Legacy models: `rustformers/pythia-ggml` `pythia-160m-q4_0.bin` and `-q5_1.bin` (quantization V2, GGML v3 backend); the two GGML v2 files were converted from `pythia-160m-f16.bin` by a validator-side script (all 2-D weights to Q4_0 with float scale; quant-version-1 = ftype 1002 with low/high-half nibbles, NEOX_4 -> `ggml_v2-cuda.cu`; quant-version-0 = header without par_res, ftype 2, interleaved nibbles, NEOX_2 -> `ggml_v2-cuda-legacy.cu`). On the GPU the v2 path offloads only batched (>=32-token) prompt matmuls, so p1 is the test.

### T3 modality smoke (one server, all modalities; GPU vs CPU)

| modality | model | GPU result | CPU result | GPU evidence |
|---|---|---|---|---|
| image | koboldcpp/imgmodel sdxs-512-tinySDdistilled_Q8_0 (1 step, 512x512) | PNG of a red apple on a table, 0.5 s | PNG, 8.4 s | diffusion model 488 MB + VAE in ROCm0 VRAM, `unet compute buffer ... on ROCm0` (text encoder on CPU by default) |
| vision | ggml-org SmolVLM-256M-Instruct Q8_0 + mmproj | "In the foreground of the picture there is an apple and a apple on the table." | "A red apple and an apple on a wooden table." | `mtmd_backends: using ROCm0 backend` |
| whisper | koboldcpp/whisper tiny.en q5_1, whisper.cpp jfk.wav | exact JFK quote | same | `whisper_backend_init: using backend ROCm0` |
| TTS | OuteTTS-0.2-500M Q4_0 + WavTokenizer-Large-75 Q4_0, `--ttsgpu` | 3.09 s WAV, RMS 2986 | 2.98 s, RMS 2806 | both models offloaded to ROCm0 |
| embeddings | second-state all-MiniLM-L6-v2 Q8_0, `--embeddingsgpu` | dim 384, cos 0.9386 / 0.0599 | 0.9374 / 0.0586 | 7/7 layers on ROCm0 |

Music (ACE-Step) and the two-GPU split: `owed` (model set not downloaded; one GPU pinned).

### Other observations (no change made)

- `resolve_fused_ops: ... Flash Attention over K/V rows is assigned to device CPU ... set to disabled` appears on every HIP load. This is llama.cpp's own probe: `ggml_cuda_get_best_fattn_kernel` gives the `kv_rows` variant only to NVIDIA Turing+ MMA, the probe turns the fused op off, and ordinary FlashAttention stays on ROCm0 (`flash_attn = enabled`). Inherited, the same as pristine llama.cpp.
- rocminfo ignores `HIP_VISIBLE_DEVICES` and lists all 63 partitions; the launcher uses the smallest of the first four only for its VRAM estimate. Harmless here.
- Non-GPU tests: upstream deleted `tests/` (`36253efd1`, "fix lookup race", 2026-11-20), so the plan's four Python tests do not exist at the base; recorded as scoped out with that reason. `python -m py_compile koboldcpp.py` passes.

### Validator-side material (not in the fork)

T1 overlay recipe above; T2/T3 harness scripts, the legacy-model converter and the hipBLAS reproducer were in this job's `agent_space/` (wiped with the job). Their behaviour is described above in enough detail to rebuild them.

## Lesson for cuda-to-rocm: hipblasGemmEx rejects HIPBLAS_COMPUTE_32F_FAST_16F for FP16 inputs

For `references/libraries.md` (cuBLAS section). Symptom: a cuBLAS-to-hipBLAS shim that maps `CUBLAS_COMPUTE_32F_FAST_16F` to `HIPBLAS_COMPUTE_32F_FAST_16F` (the natural spelling once HIP 6.5 introduced `hipblasComputeType_t`) fails at run time with `HIPBLAS_STATUS_NOT_SUPPORTED` (7) on an FP16 x FP16 -> FP32 `hipblasGemmEx`, while the CUDA build of the same code runs (not re-verified against current cuBLAS here). Cause: hipBLAS's header documents FAST_16F as "32-bit input can use 16-bit compute" (`hipblas-common.h`), and the rocBLAS backend and does not take it with 16-bit inputs. Fix: map it to `HIPBLAS_COMPUTE_32F` for those GEMMs (before HIP 6.5 the shims used `HIPBLAS_R_32F`, which is why older builds worked). Seen with ROCm 10.2 (HIP 7.17) on gfx942 (koboldcpp legacy GGML v2 path). Check every compute-type enum a shim maps, not just the ones the main path uses.

## Lesson for cuda-to-rocm: rocminfo prints pool sizes as "N (0xH) KB"

For `references/platform-limits.md` (device queries). Symptom: a launcher that scrapes `rocminfo` for VRAM finds nothing on ROCm 10.1/10.2 (koboldcpp: "Unable to detect VRAM", no GPU auto-selection). Cause: rocminfo writes `Size: 25163776 (0x17ff800) KB` with a space before the parenthesis; parsers written against `25163776(0x17ff800) KB` (regex `(\d+)\(0x`) miss it. Fix: allow optional whitespace (`(\d+)\s*\(0x[0-9a-fA-F]+\)\s*KB`), or query the HIP runtime (`hipMemGetInfo`) instead of scraping text. rocminfo also ignores `HIP_VISIBLE_DEVICES` (it lists every agent), so it is a poor source for the devices a process will see.

## Lesson for moat-porting: TheRock ROCm 10.x release wheels live on stable.repo.amd.com

For the "Fleet hosts" section. `repo.amd.com/rocm/whl/<family>/` stops at 7.13 and `repo.amd.com/rocm/whl-multi-arch/` at 7.14.1; the ROCm 10.x release wheels are on `https://stable.repo.amd.com/rocm/whl-next/` (nightlies on `nightly.repo.amd.com/rocm/whl-next/`). A job dispatched with `FLEET_VENV` may install them into that venv (`$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[libraries,devel]==10.2.0" "rocm-sdk-device-<gfx>==10.2.0"`); the job guard refuses an install outside it, and also refuses any shell command whose text merely contains the two words of a pip install, even inside a grep pattern.

## Review 2026-11-23 (linux-gfx942 fleet worker, fork `moat-port` 0fff6ace6 vs `concedo_experimental` 0382471c8)

Verdict: review-passed for the Linux round. The Windows/CMake surface entries are `owed` by Jeff's ruling (Q1 = a, later Windows round) and are not a finding. No defect found in the four fork commits; each point below was checked against the code, not taken from the porter's notes.

Checks made (so the next round need not redo them):
- CUDA behaviour of the GGML v2 compute-type fix: the edited macro sits inside `#if defined(GGML_USE_HIP)` (`otherarch/ggml_v2-cuda.cu:7-74`, `ggml_v2-cuda-legacy.cu:7-73`); the CUDA branch includes `cublas_v2.h` and uses cuBLAS's own enum. `GGML_USE_HIP` comes only from `HIPFLAGS` under `ifdef LLAMA_HIPBLAS` (`Makefile:361`) and the CMake `LLAMA_HIPBLAS` branch (`CMakeLists.txt:216`), and `Makefile:21-22` refuses `LLAMA_CUBLAS` together with `LLAMA_HIPBLAS`. Reproduced the preprocessor check at both revisions with `#include` lines stripped: `cpp -P -x c++` without `GGML_USE_HIP` gives sha256 prefix `e3325a2bdc89fac4` (v2) and `7557c1cc3c39dba8` (v2-legacy) for base and head alike; with `-DGGML_USE_HIP -DHIP_VERSION=70000000` the outputs differ, as intended. The fix does not touch the vendored `ggml/src/ggml-cuda/` backend; the macro is used only at `ggml_v2-cuda.cu:671` and `ggml_v2-cuda-legacy.cu:649` (FP16 x FP16 -> FP32 `cublasGemmEx`). `ggml_v3-cuda.cu:27` and `ggml/src/ggml-cuda/vendors/hip.h:169` keep the FAST_16F mapping but never use the macro (v3's GemmEx at `ggml_v3-cuda.cu:8315` uses `CUBLAS_COMPUTE_16F`), so nothing else needs the same change. The real nvcc compile on the mi355 host is still owed before presenting (standing rule), as the porter recorded.
- rocminfo regex: the TheRock 10.2 `rocminfo` on this host prints `Size:                    25163776 (0x17ff800) KB` for every GPU pool (189 lines), so the old `(\d+)\(0x` pattern misses it and the new `\s*` pattern accepts both spellings. The change is inside the rocminfo branch only (`koboldcpp.py:2137-2178`), reached only when `nvidia-smi` found no device.
- gfx950 portable target: inside `ifdef LLAMA_PORTABLE` under `ifeq ($(origin GPU_TARGETS),undefined)` (`Makefile:336-358`); a user-supplied `GPU_TARGETS` or a non-portable build is unaffected. CDNA4 is handled by the vendored backend.
- README sentence matches the Makefile (`amdgpu-arch` detection, `/opt/rocm` then `/usr`, `ROCM_PATH`).
- `python3 utils/jargon.py --port koboldcpp`: clean. `python3 utils/check.py`: `commits: ok`. All four titles carry `[ROCm]`, are ROCm-specific, are under 72 characters, disclose the agent in prose, and have no trailer.

### Lesson corrections (apply when promoting; not a fork change)

- "hipblasGemmEx rejects HIPBLAS_COMPUTE_32F_FAST_16F for FP16 inputs": the cuBLAS documentation's `cublasGemmEx` type table (docs.nvidia.com/cuda/cublas, cublasComputeType_t and the GemmEx table, fetched 2026-11-23) lists `CUBLAS_COMPUTE_32F_FAST_16F` (and FAST_16BF, FAST_TF32) only with `CUDA_R_32F` A/B/C; `CUBLAS_COMPUTE_32F` is the documented compute type for FP16 inputs with FP32 output. So the koboldcpp CUDA code passes an undocumented combination that cuBLAS tolerates, and hipBLAS enforces the documented one: this is the CUDA-lenient/AMD-strict pattern, and the lesson should say so rather than "the CUDA build of the same code runs (not re-verified)". The fix advice then generalizes: map FAST_* compute types to `HIPBLAS_COMPUTE_32F` whenever the inputs are not FP32. Also repair the garbled clause "and the rocBLAS backend and does not take it with 16-bit inputs". The header quote is accurate (`hipblas-common.h:79`, `HIPBLAS_COMPUTE_32F_FAST_16F = 4, /**< 32-bit input can use 16-bit compute */`, ROCm 10.2 SDK).
- The rocminfo lesson checks out as written.
