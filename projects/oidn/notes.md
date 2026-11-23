# oidn notes

## Intake screen (2026-11-05)

Source of the lead: third-party "AMD AI Software Gap Matrix", entry "Intel Open Image Denoise", verdict partial, AMD path upstream. Screened against `RenderKit/oidn` default branch (shallow clone with submodules, latest release v2.5.1 of 2026-08-18, last push 2026-10-05; active, not archived, 2.1k stars).

### Licence

`python3 utils/licenses.py check RenderKit/oidn` -> Apache-2.0, tier 1 (LICENSE.txt). `scan-nvidia agent_space/oidn-screen` (with submodules): 15 hits, all under `external/cutlass/python/CuTeDSL/` (the NVIDIA CuTe DSL Python package inside the vendored NVIDIA/cutlass submodule). OIDN's CUDA device uses only CUTLASS C++ headers; the HIP device uses Composable Kernel (`external/composable_kernel`, MIT) and never touches cutlass. No hit in OIDN's own tree.

### Existing AMD support: upstream, maintained, with prebuilt binaries

- `devices/hip/` is a first-class upstream HIP backend (`OIDN_DEVICE_HIP`), maintained by the Intel OIDN team (atafra), using Composable Kernel convolutions: `ck_conv_dl.cpp` (RDNA2 dot path) and `ck_conv_wmma.cpp` (RDNA3/3.5/4 WMMA path).
- Supported archs (`hip_device.cpp` `getArch`): DL = gfx1030/1031/1032/1034/1035/1036; WMMA = gfx1100/1101/1102/1103/1150/1151/1152/1200/1201. README: "AMD GPUs with RDNA 2, RDNA 3, RDNA 3.5, and RDNA 4 architectures". gfx1033 is compiled but deliberately not enabled (CK lacks it, per atafra on #267).
- ROCm requirement: HIP SDK / ROCm 6.4.2 or newer; driver Adrenalin 25.30.02.01+ (Windows) or Radeon Software for Linux 25.30.1+.
- Prebuilt binaries: `scripts/build.py` full/package builds pass `-D OIDN_DEVICE_HIP=ON` for every non-macOS x86_64 package, so the release `oidn-<ver>.x86_64.linux.tar.gz` and `oidn-<ver>.x64.windows.zip` ship the HIP device for both Linux and Windows.
- AMD already works upstream directly: PR #267 (Richard Geslot, AMD) added gfx1031-1036 and gfx1150/1151; atafra merged it as 5a962dc (Aug 2025). The CHANGELOG then records RDNA 3.5 gfx1152 and RDNA4 support. PR #288 (external semaphore for HIP, merged May 2026) shows upstream takes HIP contributions.
- No AMD-Ecosystem, ROCm or GPUOpen fork or separately named project found (`gh search repos oidn --owner ...`). No open upstream PR or issue for CDNA / Instinct (searched cdna, mi300, mi250, gfx90a, gfx942, gfx908, instinct). Open HIP issues (#218, #262, #268) are build/user questions on unsupported or misconfigured GPUs, not missing ports.

### What "partial" means: CDNA (Instinct) only

The one AMD family not covered is CDNA (gfx908/gfx90a/gfx942/gfx950). That is not a missing port of CUDA code; it would be new feature work in an upstream-owned backend:
- The HIP backend is wave32-only by design: `devices/hip/CMakeLists.txt` compiles with `-mcumode -mno-wavefrontsize64`, and shared kernels in `devices/gpu/` (e.g. `gpu_input_process.h`) assume subgroup size relates to channel count. CDNA is wave64-only.
- CDNA has neither WMMA nor the RDNA2 dot path; it needs a new Composable Kernel XDL/MFMA convolution instantiation (a third `ck_conv_*.cpp`), plus tuning.
- OIDN's audience is rendering (Blender Cycles, renderers on workstation/consumer GPUs); the README targets "laptops to workstations and compute nodes". Instinct render-farm demand exists but no upstream user has asked for it.

### Recommendation

Decline, SKIP_REASON `already-supported`. OIDN ships a maintained upstream HIP backend covering every RDNA2/3/3.5/4 GPU, with prebuilt Linux and Windows binaries, and AMD contributes to it directly. There is no CUDA-to-HIP port to do. The CDNA gap is a feature addition (wave64 + MFMA conv path) to Intel's backend, not a narrow arch-list addition, and MOAT's wave64 gate could only be met by building that feature.

### Question for jeff

Should MOAT take up CDNA (gfx90a/gfx942/gfx950) support for OIDN's upstream HIP backend as a feature project? It needs a wave64 build of the HIP device, a Composable Kernel XDL/MFMA convolution path alongside the existing DL and WMMA ones, and fixes to the shared `devices/gpu` kernels' subgroup-size assumptions; upstream (Intel, atafra) accepts HIP PRs but has never been asked for Instinct support, and AMD's rendering team (Richard Geslot) already contributes RDNA support there. My default is decline `already-supported`.

## Ruling: validation project (Jeff Daily, 2026-11-10)

Jeff Daily overrode the intake decline: oidn becomes a validation project. Validate upstream's HIP device (devices/hip). Validate on every required gate (wave64, wave32, Windows) and fix what breaks; the fixes go on `moat-port` of a new AMD-Ecosystem fork. A person creates that fork, never an agent; the project waits in `awaiting-fork` until it exists, then `release-forks` moves it to `screened` for the planner.


## Planner (linux-gfx90a, 2026-11-20)

### Decisions needed from Jeff before a porter starts

**Q1 (blocking): scope and delivery, given upstream PR #302.** On 2026-11-06, the day after intake, `wensongc` (GitHub profile: AMD) opened https://github.com/RenderKit/oidn/pull/302, "HIP: add CDNA4 gfx950 (AMD Instinct MI350) support". It is open and mergeable, with no reviews. It has one commit, 75bc1d936d, on top of `master` 6602ee2 (= our fork's `master`). It adds a CK XDL/MFMA convolution TU for gfx950 only, which the author tested on MI350X; RDNA was only compiled, and Windows was not built. Plain HIP-on-master cannot meet the wave64 gate at all (no CDNA target is enabled). The `wave64` waiver criterion in `config/arches.toml` (GPU work inseparable from a graphics pipeline) does not fit OIDN, which is pure compute. Options:
- **A (recommended):** base `moat-port` on #302's head, generalise its gfx950-only CDNA path to gfx90a (MI200) and gfx942 (MI300) with a tile valid there, and validate on gfx90a, gfx942, gfx950, RDNA and Windows, fixing what breaks. Deliver by coordinating with wensongc (our commits onto his PR branch, or a follow-up PR after #302 merges), never a competing PR. You or someone at AMD would need to contact him; any comment on #302 is upstream-visible and is yours to approve.
- **B (smallest):** base `moat-port` on #302's head with no extension. wave64 is met on gfx950 (mi355 host), wave32 and Windows on RDNA. Fix only what breaks. gfx90a/gfx942 stay unsupported by design and get `set-blocked`. Delivery is the same coordination, limited to fixes.
- **C (not recommended):** an independent CDNA port from `master` that ignores #302. That would be a second, competing AMD PR.
- If you would rather wait for #302 to merge first, the project needs a `set-hold` until then.

**Q2: gfx908 (MI100).** The pinned CK supports XDL there, but there is no fleet hardware to validate it. Default: do not enable it.

**Q3: `fork_default_branch` is wrong.** The record says `main`. Both the fork and upstream use `master` (`main` does not exist on AMD-Ecosystem/oidn). Per `moatlib.set_base_branch` this is a person's ruling: `python3 utils/moatlib.py set-base-branch oidn master --by <you>`. Until then, `jargon.py --port`, review-PR base and `pr-ready` all resolve against a branch that does not exist.

**Q4: weights submodule.** `.gitmodules` has `url = ../oidn-weights.git`. On the fork that resolves to AMD-Ecosystem/oidn-weights, which does not exist, and the real repository stores `*.tza` in Git LFS. Default: validators override locally (`git config submodule.weights.url https://github.com/RenderKit/oidn-weights.git`, `git lfs install`) and `.gitmodules` stays untouched. The alternative is for you to create an AMD-Ecosystem/oidn-weights mirror.

**Q5: training scripts scoped out.** `training/` is PyTorch only (`'cuda'` device strings, `torch.distributed` NCCL backend), so a ROCm PyTorch runs it unchanged and there is nothing to port. Your 2026-11-10 ruling scopes the work to `devices/hip`. surface.json records `python:training` as scoped out on that basis. Say if you want a ROCm training smoke run as well.

### Findings

- Build: `cmake` (recorded). The HIP device is a native HIP backend (no hipify and no compat header), built as an ExternalProject with ROCm `clang++` (`devices/CMakeLists.txt:91-185`, `devices/hip/CMakeLists.txt`). `GPU_TARGETS` is a hard-coded `CACHE INTERNAL` list of 19 RDNA/GCN-APU targets. #302 adds `OIDN_HIP_TARGETS` to override it.
- Correction to the intake screen: only `ck_conv_wmma.cpp` is compiled `-mcumode -mno-wavefrontsize64`, not the whole backend. The shared `devices/gpu` kernels use the runtime `warpSize` and need only `warpSize % dstPaddedC == 0`, which holds for 64 with `tensorBlockC` 8 or 32. Reading the code, nothing in `devices/gpu`, `core/kernel.h` or `common/platform.h` is wave32-only, and #302's gfx950 `oidnTest` pass agrees.
- The CK submodule pin (3e6d21adeb33, 2025-04-09) already has `device_grouped_conv_fwd_multiple_abd_xdl_cshuffle{,_v3}.hpp` and defines `__gfx9__` for gfx908/90a/942/950, so CDNA needs no CK bump.
- #302 design points the reviewer should check (plan.md risks 2-4): `#define warpSize 64` in the MFMA TU is safe only while that TU builds solely for wave64 targets. `DL_KERNELS;CK_USE_WMMA` became per-source, so CK headers see different macros in different TUs. Its tile uses AK1=BK1=16 (gfx950 double-rate MFMA), and `newHIPConv` picks tiles by cost without calling `isSupported()`.
- Upstream `scripts/test.py` image regression needs Intel's private baseline repository (`OIDN_VALIDATION_REPO`). The plan substitutes a CPU-device differential through `oidnDenoise -r ref.pfm`.
- Upstream re-check 2026-11-20: no other CDNA/Instinct PR or issue. `devel` is 3 commits ahead of `master` and touches no HIP/GPU file.
- surface.json: generated floor (12 CUDA files, 27 components), plus 6 added components the scan cannot see: the three apps built through the `oidn_add_app` macro (oidnTest, oidnDenoise, oidnBenchmark), `python:training`, `feature:hip-cdna-conv`, and `platform:windows-hip`. The other vendors' backends (CUDA, curtn, SYCL, Metal, BNNS) and training are scoped out with reasons. The CUDA files in the census are the NVIDIA backend, not port surface.

## Rulings on the planner's questions (Jeff Daily, 2026-11-23)

- Q1 (scope and delivery, given RenderKit/oidn PR 302): **A**. Base `moat-port` on PR 302's head, generalise its gfx950-only CDNA path to gfx90a and gfx942, validate on gfx90a, gfx942, gfx950, RDNA and Windows, fix what breaks. Delivery is by coordinating with the PR's author, never a competing PR; Jeff contacts the author himself. Nothing is posted on PR 302 by an agent.
- Q2 (gfx908): not enabled. Jeff has no MI100 fleet host; he is asking for one but the port does not wait for it.
- Q3 (base branch): yes. `set-base-branch oidn master --by jeffdaily` was run on 2026-11-23.
- Q4 (weights submodule): no AMD-Ecosystem/oidn-weights mirror will be created. Validators override the submodule URL locally; `.gitmodules` stays untouched.
- Q5 (training): Jeff wants a ROCm training smoke run. `python:training` is therefore not scoped out: it needs no source port unless the smoke run finds a fault, but a short training run on ROCm PyTorch (and inference with the result) is part of the evidence on the Linux gates.
- Standing ruling of the same day: Windows laptop GPU runs go last across all ports. The Windows validation is queued, not started.

## Porter (linux-gfx950, 2026-11-23)

### Questions for a person (nothing below waits on them)

**P1: CMake option naming for the CDNA targets (coordination with the PR 302 author).** I kept the author's `OIDN_HIP_GFX950` option with its meaning unchanged, and gfx90a/gfx942 are probed and built whenever the compiler accepts them, with no option of their own (opt-out only by setting `OIDN_HIP_TARGETS`). Options: (a) keep as is (implemented; smallest delta on his commit); (b) one `OIDN_HIP_CDNA` option for all three, renaming his; (c) one option per target (`OIDN_HIP_GFX90A`, `OIDN_HIP_GFX942`). This is a naming choice for whoever talks to wensongc; a change is a small follow-up commit.

**P2: fork GitHub default branch.** Jeff's standing rule says every AMD-Ecosystem fork of a MOAT project has `moat-port` as its GitHub default branch. `moat-port` now exists on AMD-Ecosystem/oidn (pushed in this round); the GitHub default is still `master`. Changing it is a GitHub settings write that a fleet worker may not make. The recorded `fork_default_branch` stays `master`.

**P3: the commit-message gate flags PR 302's commit, which the ruling keeps verbatim.** A full `python3 utils/check.py` (and `moatlib.py audit-commits`) exits 1 with four findings on 75bc1d9 (wensongc): no `[ROCm]` prefix although the diff is ROCm-specific, no AI disclosure, no Test Plan, a Co-Authored-By trailer. Our commit 06bdc4c passes. The pre-push hook runs `check.py --fast`, which skips this gate, so nothing was blocked in this round, but every host with a fork clone will see the gate fail until a person rules. Options: (a) a trunk tooling change that exempts commits authored outside our team when a ruling keeps them (e.g. a per-project list of foreign shas in the record); (b) rewrite or squash his commit (contradicts the 2026-11-23 ruling and breaks coordination with the author); (c) leave the gate red for oidn, since delivery goes through his PR and not through our publish path. I did not change any tooling (trunk work, and the choice is a person's).

**P4 (information for the reviewer): PR 302's commit carries a `Co-authored-by: Cursor <cursoragent@cursor.com>` trailer.** Kept verbatim per the ruling (his commit stays as he wrote it). Our commit has no trailer and discloses the agent in prose.

### Toolchain and build (exact)

- ROCm 10.2.0 TheRock SDK and ROCm PyTorch installed into the job venv from the stable index: `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel,device-gfx90a,device-gfx942,device-gfx950]==10.2.0" "torch[device-gfx950]==2.15.0+rocm10.2.0"`. `rocm-sdk version` 10.2.0, `hipconfig --version` 7.17.26434. `ROCM_PATH=$(rocm-sdk path --root)` (`.../site-packages/_rocm_sdk_devel`), `PATH` and `LD_LIBRARY_PATH` prepended with its `bin`/`lib`. (The older indices stop early: `repo.amd.com/rocm/whl/gfx950-dcgpu` at 7.13, `repo.amd.com/rocm/whl-multi-arch` at 7.14.1, `nightly.repo.amd.com/rocm/whl-next` has 10.2/10.3 nightlies only; the 10.2.0 release is on `stable.repo.amd.com/rocm/whl-next/`.)
- ISPC: OIDN's CPU device needs ISPC >= 1.30 (1.28 is refused by `cmake/oidn_ispc.cmake`). Used the ispc v1.32.0 Linux release tarball under the job directory. oneTBB 2021.11 from the image.
- Fork clone: `git clone https://github.com/AMD-Ecosystem/oidn src`, `git fetch https://github.com/RenderKit/oidn pull/302/head:pr302`, `git checkout -b moat-port pr302`, `moatlib.py protect-fork oidn`. Weights (Q4 ruling): `git config submodule.weights.url https://github.com/RenderKit/oidn-weights.git`, `GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive external/composable_kernel weights`, then in `weights/`: `git lfs install --local --skip-repo && git lfs pull` (real `.tza`, e.g. rt_alb.tza 1830792 bytes). `.gitmodules` untouched. `git lfs install --local` in the fork root refuses because the fork's pre-push hook already exists; `--skip-repo` installs the filters without touching hooks.
- Configure and build (default target list, i.e. what ships):
  ```
  cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DOIDN_DEVICE_HIP=ON -DOIDN_DEVICE_CPU=ON \
        -DROCM_PATH=$ROCM_PATH -DHIP_CLANG_NUM_PARALLEL_JOBS=4 -DISPC_EXECUTABLE=<ispc>
  cmake --build build -j 16
  ```
  68 s wall on this host (132 steps), 0 warnings. PR 302 as written (75bc1d9) also builds clean with ROCm 10.2 (69 s); no CK-vs-new-clang breakage (plan risk 6 did not fire).

### What changed (fork `moat-port` 06bdc4c, one commit on top of PR 302's 75bc1d9)

`[ROCm] HIP: add CDNA2/CDNA3 gfx90a and gfx942 support`:
- `devices/hip/CMakeLists.txt`: `OIDN_HIP_CDNA_TARGETS gfx90a gfx942 gfx950`; each is probed (gfx950 only if `OIDN_HIP_GFX950`), accepted ones are appended to `OIDN_HIP_TARGETS`; `ck_conv_mfma.cpp` builds for exactly the enabled CDNA targets; every CDNA target gets `--no-offload-arch` on the DL and WMMA sources. Defines: `OIDN_HIP_MFMA` plus `OIDN_HIP_GFX90A`/`GFX942`/`GFX950` per enabled target.
- `hip_device.cpp` `getArch`: gfx90a, gfx942, gfx950 -> `HIPArch::MFMA`, each under its own define, so a GPU is only reported supported when its kernels were compiled. `hip_conv.cpp` dispatch under `OIDN_HIP_MFMA`. `tensorBlockC` stays 32.
- `ck_conv_mfma.cpp`: `#error` in the device pass for any target other than gfx90a/gfx942/gfx950 (guards the `#define warpSize 64`); comments generalised.
- No new tile. PR 302's tile (256 threads, 128x64x64, AK1=BK1=16, 32x32 XDL) is reused on all three targets; see risk 4.
- Docs: README.md and doc/overview.md list CDNA 2/3/4 (Instinct MI200/MI300/MI350 series); README.md and doc/compilation.md describe `OIDN_HIP_GFX950` and the gfx90a/gfx942 behaviour (PR 302 had added the option to README.md only, which is generated from doc/; doc/compilation.md now has it too); CHANGELOG entry. The HIP build itself was already documented upstream (README "HIP device for AMD GPUs"), so no new build block.
- CUDA, SYCL, Metal, CPU backends and `devices/gpu`, `core`, `common`: untouched (diff is confined to `devices/hip/`, README.md, doc/, CHANGELOG.md).

### PR 302 design risks (plan risks 1-4), checked

1. **Wavefront width across TUs** -- verified on the built objects (`objcopy --only-section=.hip_fatbin` + `clang-offload-bundler --list`): `ck_conv_mfma.cpp.o` = gfx90a gfx942 gfx950 only; `ck_conv_wmma.cpp.o` and `ck_conv_dl.cpp.o` = the 19 RDNA/APU targets, no CDNA target; the shared kernels (`hip_engine.cpp.o`) carry all 22. Same check on PR 302's own build: MFMA gfx950 only, DL/WMMA no gfx950.
2. **`#define warpSize 64`** -- `ck::get_warp_size()` at the CK pin returns HIP's `warpSize`, which ROCm 7+ no longer makes constexpr, so the define is needed and is correct for any wave64-only TU. All three MFMA targets report `.wavefront_size: 64` in their code-object notes. The new `#error` fires when the TU is compiled for gfx1100 (`error: "MFMA convolutions must be compiled only for wave64 CDNA targets"`) and gfx942 still compiles. No TU emits an out-of-line host copy of `get_warp_size` (`nm -C` on all four CK objects), so the define cannot leak across TUs at link time.
3. **CK macro split across TUs** -- no ODR hazard. In the CK pin, `DL_KERNELS`/`CK_USE_WMMA` are read only by the prebuilt-instance declaration headers under `library/include/ck/library/tensor_operation_instance/` and by `config.h.in`; no OIDN TU includes those instance headers (each includes only `ck/ck.hpp`, `ck/utility/data_type.hpp` and one `device/impl/*.hpp`), and the generated `config.h` leaves `CK_USE_WMMA`/`CK_ENABLE_DL_KERNELS` undefined. The RDNA (wave32) run is still owed by the wave32 validator as runtime evidence.
4. **Tile validity on gfx90a/gfx942** -- valid by construction at the CK pin: `xdlops_gemm.hpp` `GetMfma<half_t,32,32>` selects `mfma_f32_32x32x16f16` under `__gfx950__` and `mfma_f32_32x32x8f16` otherwise, KPack 16 satisfies `KPack % k_per_blk == 0`, and `IsSupportedArgument` checks nothing arch-specific beyond `is_xdl_supported()` (gfx908/90a/942/950) and C % 4 (C is padded to 32). Disassembly of the built code objects: gfx90a 96x `v_mfma_f32_32x32x8f16`, gfx942 96x `v_mfma_f32_32x32x8_f16`, gfx950 48x `v_mfma_f32_32x32x16_f16`. Resources on all three: LDS 24768 bytes (limit 64 KiB on gfx90a/gfx942), scratch 0, VGPR spills 0, VGPRs <= 126, AGPRs 0, max workgroup 256. So no K1=8 tile was added; the single-rate path may be slower than a K1=8 tile from CK's instance list, which only a gfx90a/gfx942 benchmark can tell.

### Evidence on gfx950 (AMD Instinct MI355X, HIP_VISIBLE_DEVICES=3, ROCm 10.2.0)

GPU 3 was shared during the whole round: another container's process (50 GB VRAM) held it at 100% GFX activity. Correctness runs are unaffected; timings are not reference numbers.
- `build/oidnDenoise --ld`: Device 0 "AMD Instinct MI355X", Type HIP.
- `timeout 900 build/oidnTest --device hip`: All tests passed, 20 of 20 test cases (assertion counts 2472 and 2496 in two runs; the count varies run to run, the PR 302 build gave 2379 and 2463).
- `timeout 1800 build/oidnTest --device cpu`: All tests passed (1820 assertions in 20 test cases).
- CPU-reference image differential (`oidnDenoise -d cpu ... -o ref.pfm`, then `oidnDenoise -d hip ... -r ref.pfm`, default maxerror), deterministic synthetic PFM inputs at 1920x1080, 1023x577, 257x129: 19 option sets x 3 sizes = **57/57 with errors=0**. Sets: RT hdr / hdr+alb+nrm / ldr+srgb+alb+nrm at -q high, balanced, fast; hdr+alb; hdr+alb+nrm --clean_aux; ldr+alb; RTLightmap hdr and dir; and on HIP only -t half, --maxmem 256, --inplace, --buffer device, --buffer managed. avgerror 1.75e-4 to 6.9e-4 (RT/RTLightmap hdr), 1.3e-3 (RTLightmap dir). Scripts were in agent_space (gen_inputs.py, diff_run.py); they write PFM with numpy and parse `values=..., errors=..., avgerror=...`.
- GPU dispatch proof: `rocprofv3 --kernel-trace --stats` on one 1023x577 RT hdr+alb+nrm run shows 20 `kernel_grouped_conv_fwd_multiple_abd_xdl_cshuffle` dispatches (16 + 4) plus the OIDN input/output/pool/upsample/autoexposure kernels.
- `build/oidnBenchmark -d hip` (informational, shared GPU): RT hdr_alb_nrm 1920x1080 2.85 ms, 3840x2160 11.98 ms, 1280x720 1.41 ms; RTLightmap 2048x2048 5.48 ms.
- Extra configurations: `-DOIDN_HIP_TARGETS=gfx90a,gfx942,gfx950` builds and `oidnTest --device hip` passes (2352 assertions); the DL/WMMA TUs then get no target and clang falls back to its default gfx906 (dead code; PR 302 has the same behaviour with a gfx950-only list). `-DOIDN_HIP_GFX950=OFF`: MFMA TU gfx90a+gfx942 only, the MI355X is not listed by `--ld` and `oidnDenoise -d hip` fails cleanly with "unsupported device type: HIP" (the unsupported-arch contract, plan risk 11). `-DOIDN_WARN_AS_ERRORS=ON`: HIP device builds clean.
- Integrity: `git -C projects/oidn/src status --porcelain` clean at 06bdc4c (build trees are ignored; CK submodule clean because PR 302 generates `ck/config.h` in the build tree).

### What remains unproven for gfx90a and gfx942 (for their validators)

Built and statically inspected here only (code objects present, correct MFMA instructions, wave64, resources within limits). Never executed. Owed on each: `oidnTest --device hip` 20/20; the CPU-reference differential (same 57 cases, regenerate inputs with any deterministic generator); a kernel trace showing the XDL conv kernels dispatch; and the benchmark for information (no K1=8 tile exists, so a slow result is a tuning question, not a failure). Specific runtime risks only those GPUs can retire: CK `IsSupportedArgument` at run time on that device (`newHIPConv` never calls `isSupported()`, so a refusal would surface as a failed launch, not a fallback); `getArch` name matching on the real `gcnArchName` (gfx90a reports e.g. `gfx90a:sramecc+:xnack-`, stripped by `getArchName`); and the single-rate MFMA numerics against CPU. Build for the default list with the recipe above; `-DOIDN_HIP_TARGETS=<gfx>` is a fast inner loop only.

### Training smoke run (Q5 ruling)

ROCm PyTorch 2.15.0+rocm10.2.0 in the job venv, OpenImageIO 3.2.2.0 and tensorboard from PyPI. Synthetic dataset (16 train, 4 valid scenes, 384x384, `_0004spp`/`_0016spp`/`_ref` x hdr/alb/nrm EXR written with OpenImageIO). From the training dir, run with cwd at a scratch directory:
```
python preprocess.py hdr alb nrm --filter RT -D data -P preproc -t train -v valid
python train.py hdr alb nrm --filter RT -P preproc -t train -v valid -R results -r smoke --epochs 20 --valid_epochs 5 --save_epochs 10
python infer.py -D data -R results -r smoke -i valid -O infer --metric psnr ssim
python export.py -R results -r smoke -o smoke.tza
oidnDenoise -d hip --hdr ... --alb ... --nrm ... -w smoke.tza -r <cpu output with the same weights>
```
- Defaults on `cuda` are mixed precision and `torch.compile(mode='reduce-overhead')`. All five steps exit 0. Training loss 0.631 -> 0.466 over 20 epochs (unseeded); epoch 1 took 15m39s (Inductor/Triton compile and MIOpen first-use tuning on a shared GPU), later epochs about 400-500 images/s. Validation never ran: 8 validation images is less than one batch of 16, so the validation loader is empty (dataset size, not a ROCm issue).
- GPU vs CPU inference with the same checkpoint (`infer.py -d cpu`): PSNR 3.8178 vs 3.8175 dB, per-pixel max abs difference <= 7.2e-5 on outputs of mean magnitude about 4.5e-3. The low PSNR is the undertrained model (40 optimizer steps, LR still in warm-up), the same on CPU.
- The weights trained on ROCm load into the HIP device: `oidnDenoise -d hip -w smoke.tza` vs `-d cpu -w smoke.tza`, errors=0, avgerror 7.9e-5 (1023x577 and 257x129).
- Seeded isolation (`--seed 1`, same data, 20 epochs), final training loss:
  - CPU fp32 (`-d cpu`): 0.588277 -> 0.516252
  - GPU fp32 eager (`-p fp32 --compile disable`): 0.588277 -> 0.516252, identical to six decimals in every epoch except epoch 11 (0.523057 vs 0.523058)
  - GPU mixed eager (`-p mixed --compile disable`): 0.588382 -> 0.515506 (within 1e-3, fp16 autocast)
  - GPU defaults (mixed + `torch.compile` reduce-overhead, warm caches): 0.588264 -> 0.516244, exit 0
  An unseeded CPU run had started at 0.475, which is what made the first GPU-vs-CPU comparison look different: the initialization dominates these short runs. The eager GPU run's first epoch also took about 10 minutes (MIOpen first-use find and kernel compile on the shared GPU; the user find-db under `~/.config/miopen` was being written), so a slow first epoch is tuning, not a hang.
- Nothing in `training/` needed changing; no fork commit for training.
- Environment trap (not the port): `torch.cuda.device_count()` returned 0 with `HIP_VISIBLE_DEVICES=3` because the venv picked up the base image's ROCm 10.1 `amdsmi` Python package through the venv's base-env link file; matmuls still ran. `PYTHONPATH=$ROCM_PATH/share/amd_smi` (the 10.2 SDK's own amdsmi) gives 1. Keep it out of the environment of MOAT's own tools (`timeit.sh` then cannot find a usable Python).
- Harness trap: the training progress bar redraws one line with `\r`, so `tail` of the log shows the start of epoch 1 long after the run has moved on; read it with `tr '\r' '\n'`.

## Lesson for cuda-to-rocm: one Composable Kernel XDL tile can serve every CDNA generation

Belongs in `references/libraries.md`, "Generic GEMM on CK device operations". Symptom: a CK XDL instance taken from a gfx950 (double-rate MFMA) example looks gfx950-only (K1=16), and porters reach for a second K1=8 tile for gfx90a/gfx942. Cause: CK picks the MFMA instruction per offload target at compile time (`xdlops_gemm.hpp` `GetMfma<half_t,32,32>`: `mfma_f32_32x32x16f16` under `__gfx950__`, `mfma_f32_32x32x8f16` otherwise; KPack only has to be a multiple of `k_per_blk`), and `IsSupportedArgument` checks only `is_xdl_supported()` plus layout/vector conditions. Fix: compile the same instance for all CDNA targets and prove it from the code objects (disassemble per target, check `.wavefront_size`, LDS, scratch) before adding tiles; a K1=8 tile is a tuning choice, not a correctness need. Seen on oidn, CK pin 3e6d21ad (2025-04), ROCm 10.2.

## Lesson for moat-porting: `git lfs install --local` in a fork clone collides with the fork pre-push hook

Belongs in "Fleet hosts" or "Traps in MOAT's own tools". After `protect-fork`, `git lfs install --local` stops with "Hook already exists: pre-push" (and a submodule update that needs LFS then leaves pointer files). Use `GIT_LFS_SKIP_SMUDGE=1 git submodule update --init ...`, then inside the LFS submodule `git lfs install --local --skip-repo && git lfs pull`; never `git lfs update --force`, which would replace the freeze hook. (oidn)

## Review 2026-11-23 (linux-gfx90a, fork moat-port 06bdc4c vs master 6602ee2)

### Port Review: oidn (moat-port vs master)

#### Summary
Upstream PR 302 (75bc1d9, wensongc) adds a Composable Kernel XDL/MFMA convolution TU for gfx950; our commit 06bdc4c builds the same TU for gfx90a and gfx942, maps those archs to `HIPArch::MFMA` under per-target defines, and guards the TU's `#define warpSize 64` with an `#error` for any non-CDNA device pass. Verdict: **Approve** (review-passed). No defect found in 06bdc4c. The real-GPU runs on gfx90a are the validator's next step.

#### What was checked (by reading the code, not the porter's summary)
- Wavefront: the only host-side subgroup value is `subgroupSize = prop.warpSize` (`devices/hip/hip_device.cpp:179`); `gpu_input_process.h:148-165` and `:234-254` use it generically and need only `subgroupSize % dstPaddedC == 0` (64 % 32 holds). `suggestWorkGroupSize` 32x32 is width-independent. No hard-coded 32 in the diff or the code around it.
- `#define warpSize 64` (`ck_conv_mfma.cpp:16`) comes after `hip_runtime.h` and `hip_conv.h`, and the TU is fed only `--offload-arch` for gfx90a/gfx942/gfx950 (CMake `_oidn_mfma_flags`); the `#error` at `:12-15` makes any other device pass fail.
- CK at the pin 3e6d21ad: `xdlops_gemm.hpp:964-971` selects `mfma_f32_32x32x16f16` under `__gfx950__` and `mfma_f32_32x32x8f16` otherwise; `IsSupportedArgument` (`device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp:1193-1213`) has no gfx90a/gfx942-specific refusal beyond `is_xdl_supported()` (`device_prop.hpp:55-59`, lists gfx908/90a/942/950); the kernel body is compiled under `__gfx9__`, which `ck.hpp:54` defines for these targets, and `CK_BUFFER_RESOURCE_3RD_DWORD` is 0x00020000 for `__gfx9__` (`ck.hpp:79-80`). The lesson "one CK XDL tile can serve every CDNA generation" is accurate against these lines (the only other device check, a gfx908 fp64 refusal, does not apply to fp16/fp32-acc).
- `getArchName` strips `:sramecc+:xnack-` (`hip_device.cpp:85-94`), so the new `gfx90a`/`gfx942` comparisons match a real `gcnArchName`.
- `hip_conv.cpp` dispatch under `OIDN_HIP_MFMA`; `tensorBlockC` 32 is one of the two `newInputProcess` instantiations (`hip_engine.cpp:183`).
- CMake: default list = 19 RDNA/APU targets + probed CDNA targets; CDNA targets get `--no-offload-arch` on DL and WMMA, non-CDNA on MFMA. CUDA, SYCL, CPU and `devices/gpu`/`core`/`common` untouched; additive and opt-out via `OIDN_HIP_TARGETS`.
- Branch currency: `rev-list --count moat-port..origin/master` = 0; upstream `master` is still 6602ee2 and PR 302's head is still 75bc1d9 (`git ls-remote`, 2026-11-23). `jargon.py --port oidn` clean. Diff is ASCII. 06bdc4c: `[ROCm]` title (54 chars), AI disclosure in prose, Test Plan, no trailer.

#### Observations on PR 302's own commit (75bc1d9; kept as written per the ruling, not ours to fix)
- Commit hygiene against our gate: no `[ROCm]` prefix, no Test Plan, a `Co-authored-by: Cursor <cursoragent@cursor.com>` trailer (`audit-commits` reports these four). Already raised as P3/P4 for a person; nothing new.
- `OIDN_HIP_TARGETS` is a working user override (top-level pass-through in `devices/CMakeLists.txt`) but has no documentation entry of its own and no `option`/cache declaration at the top level; it is described only inside the `OIDN_HIP_GFX950` entry (README and, since 06bdc4c, `doc/compilation.md`). Worth mentioning to the author if upstream wants it documented; not a correctness issue.
- With an `OIDN_HIP_TARGETS` that names only CDNA targets, the DL/WMMA TUs get no offload arch and clang falls back to its default target (dead code). The porter recorded this; same behaviour in PR 302 alone.

## Validation 2026-11-23 (linux-gfx90a, AMD Instinct MI250X / MI250, fork moat-port 06bdc4c)

First run of the MFMA path on gfx90a hardware. One GCD, `HIP_VISIBLE_DEVICES=0` (dispatcher pin), idle at start (0% GFX, 0 GB VRAM). All GPU runs under `ulimit -c 0` and `timeout`.

### Toolchain (job directory only)
- ROCm 10.2.0 TheRock SDK pip-installed into the job venv over the base image's 10.1 links: `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel,device-gfx90a,device-gfx942,device-gfx950]==10.2.0"`. `rocm-sdk version` 10.2.0, `hipconfig --version` 7.17.26434. `ROCM_PATH=$(rocm-sdk path --root)`, `PATH` += `$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin`, `LD_LIBRARY_PATH` += `$ROCM_PATH/lib`. No system ROCm used.
- ISPC v1.32.0 Linux release tarball extracted under the job directory (`deps/ispc`); oneTBB 2021.11 from the image. Torch was not installed (no training run, below).
- Weights (Q4 ruling): `git config submodule.weights.url https://github.com/RenderKit/oidn-weights.git`, `GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive external/composable_kernel weights`, then in `weights/` `git lfs install --local --skip-repo && git lfs pull` (rt_alb.tza 1830792 bytes). `.gitmodules` untouched.

### Build (default target list, all targets)
```
cmake -G Ninja -S src -B src/build -DCMAKE_BUILD_TYPE=Release -DOIDN_DEVICE_HIP=ON -DOIDN_DEVICE_CPU=ON \
      -DROCM_PATH=$ROCM_PATH -DHIP_CLANG_NUM_PARALLEL_JOBS=4 -DISPC_EXECUTABLE=<ispc>
cmake --build src/build -j 32
```
168 s, 132 steps, 0 warnings. Code objects (`objcopy --only-section=.hip_fatbin` + `clang-offload-bundler --list`): `ck_conv_mfma.cpp.o` = gfx90a, gfx942, gfx950 only; `ck_conv_wmma.cpp.o` and `ck_conv_dl.cpp.o` = the 19 RDNA/APU targets, no CDNA; `hip_engine.cpp.o` = all 22.

### Results
- `oidnDenoise --ld`: Device 0 "AMD Instinct MI250X / MI250", Type HIP (so `getArch` matched the real `gcnArchName` for gfx90a).
- `timeout 1800 oidnTest --device hip`: **All tests passed (2499 assertions in 20 test cases)**.
- `timeout 3600 oidnTest --device cpu`: **All tests passed (1820 assertions in 20 test cases)**.
- CPU-reference image differential (`oidnDenoise -d cpu ... -o ref.pfm`, then `oidnDenoise -d hip ... -r ref.pfm`, default maxerror), deterministic numpy-generated PFM inputs at 1920x1080, 1023x577, 257x129 (seeds 1234-1236), the same 19 option sets as the gfx950 run (RT hdr, hdr+alb+nrm, ldr+srgb+alb+nrm at -q high/balanced/fast; hdr+alb; hdr+alb+nrm --clean_aux; ldr+alb; RTLightmap hdr and dir; HIP-only -t half, --maxmem 256, --inplace, --buffer device, --buffer managed): **57/57, errors=0 in every case**, avgerror 1.57e-4 to 6.30e-4.
- MFMA path really runs: `rocprofv3 --kernel-trace --stats` on one 1023x577 RT hdr+alb+nrm run shows 20 `ck::...::kernel_grouped_conv_fwd_multiple_abd_xdl_cshuffle` dispatches (16 + 4, 3.57 ms + 1.34 ms) on agent gfx90a "AMD Instinct MI250X / MI250", plus the OIDN input/output/pool/upsample/autoexposure kernels. Disassembly of the gfx90a code object from `ck_conv_mfma.cpp.o`: 96x `v_mfma_f32_32x32x8f16`, `.wavefront_size: 64`, LDS 24768 bytes, scratch 0, VGPR/SGPR spills 0, VGPRs 126/60, AGPRs 0, max workgroup 256. So CK's `IsSupportedArgument`-less launch works on gfx90a and the single-rate MFMA numerics agree with CPU.
- `oidnBenchmark -d hip` (information only, idle GCD): RT hdr_alb_nrm 1920x1080 17.16 ms, 3840x2160 70.70 ms, 1280x720 7.91 ms; RT hdr_calb_cnrm 1920x1080 30.41 ms; RTLightmap 2048x2048 33.86 ms. About 6x the shared-MI355X numbers; one MI250X GCD has far lower fp16 MFMA throughput, so this is not evidence of a problem. Whether a K1=8 tile is faster on gfx90a remains a tuning question (porter's note).
- Integrity: `git -C projects/oidn/src status --porcelain` empty at 06bdc4c; `jargon.py --port oidn` clean.

### Not run here, and why
- CUDA no-regression compile: this host has no CUDA toolkit (no nvcc, no conda). Not recorded at this head anywhere yet; it is owed by the mi355 host before the port is presented. The diff touches no CUDA source (one `option()` line inside `if(NOT APPLE)` and pass-through lines inside `if(OIDN_DEVICE_HIP)`).
- ROCm training smoke run (Q5 ruling): not repeated. The porter ran it on gfx950 (wave64 gate) and the porter's list of what gfx90a still owes does not include it; the request was to run it here only if the notes said it was still needed. Nothing in `training/` is arch-specific (it runs through ROCm PyTorch, not this library's kernels). If Jeff wants it on gfx90a as well, it is the porter's recipe above with `torch[device-gfx90a]==2.15.0+rocm10.2.0`.
