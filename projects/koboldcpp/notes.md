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
