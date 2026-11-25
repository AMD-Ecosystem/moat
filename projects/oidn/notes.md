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

## Validation 2026-11-23 (linux-gfx950, AMD Instinct MI355X, fork moat-port 06bdc4c)

Validator run on gfx950, additive to the wave64 gate that linux-gfx90a already satisfies (orient offered nothing for this platform; Jeff Daily asked for it). One GPU, `HIP_VISIBLE_DEVICES=4` (dispatcher pin), shared: another process held it at 100% GFX and 69 GB VRAM throughout, so the timings below are not reference numbers. All GPU runs under `ulimit -c 0` and `timeout`.

### Toolchain (job directory only)
- ROCm 10.2.0 TheRock SDK and ROCm PyTorch pip-installed into the job venv over the base image's 10.1 links: `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel,device-gfx90a,device-gfx942,device-gfx950]==10.2.0" "torch[device-gfx950]==2.15.0+rocm10.2.0" OpenImageIO` (OpenImageIO 3.2.2.0). `rocm-sdk version` 10.2.0, `hipconfig --version` 7.17.26434. `ROCM_PATH=$(rocm-sdk path --root)`, `PATH` += `$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin`, `LD_LIBRARY_PATH` += `$ROCM_PATH/lib`. No system ROCm used. (The fleet guard refuses a pip command that does not spell `$FLEET_VENV`; a relative path to the same venv is refused.)
- ISPC v1.32.0 Linux release tarball under the job directory; oneTBB 2021.11 from the image.
- Fork clone at 06bdc4c, `protect-fork` installed. Weights (Q4 ruling): `git config submodule.weights.url https://github.com/RenderKit/oidn-weights.git`, `GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive external/composable_kernel weights`, then in `weights/` `git lfs install --local --skip-repo && git lfs pull` (rt_alb.tza 1830792 bytes). `.gitmodules` untouched; `git status --porcelain` empty.

### Build (default target list, all targets)
```
cmake -G Ninja -S src -B src/build -DCMAKE_BUILD_TYPE=Release -DOIDN_DEVICE_HIP=ON -DOIDN_DEVICE_CPU=ON \
      -DROCM_PATH=$ROCM_PATH -DHIP_CLANG_NUM_PARALLEL_JOBS=4 -DISPC_EXECUTABLE=<ispc>
cmake --build src/build -j 21
```
73 s, 132 steps, 0 warnings. Code objects (`objcopy --dump-section .hip_fatbin` + `clang-offload-bundler --list`): `ck_conv_mfma.cpp.o` = gfx90a, gfx942, gfx950 only; `ck_conv_wmma.cpp.o` and `ck_conv_dl.cpp.o` = the 19 RDNA/APU targets, no CDNA; `hip_engine.cpp.o` = all 22.

### Results
- `oidnDenoise --ld`: Device 0 "AMD Instinct MI355X", Type HIP; Device 1 the EPYC 9575F CPU.
- `timeout 1800 oidnTest --device hip`: **All tests passed (2553 assertions in 20 test cases)**.
- `timeout 3600 oidnTest --device cpu`: **All tests passed (1820 assertions in 20 test cases)**.
- CPU-reference image differential (`oidnDenoise -d cpu ... -o ref.pfm`, then `oidnDenoise -d hip ... -r ref.pfm`, default maxerror), freshly written deterministic numpy PFM inputs at 1920x1080, 1023x577, 257x129 (seeds 1234-1236; not the same images as the earlier runs), the same 19 option sets (RT hdr, hdr+alb+nrm, ldr+srgb+alb+nrm at -q high/balanced/fast; hdr+alb; hdr+alb+nrm --clean_aux; ldr+alb; RTLightmap hdr and dir; HIP-only -t half, --maxmem 256, --inplace, --buffer device, --buffer managed): **57/57, errors=0 in every case**. avgerror 1.74e-4 to 6.66e-4 for every set except RTLightmap dir, 1.0e-2 to 1.05e-2 (this run's directional input is unit-variance-scale Gaussian noise, a harder input than before; still errors=0).
- MFMA path really runs: `rocprofv3 --kernel-trace --stats --output-format csv` on one 1023x577 RT hdr+alb+nrm run: 49 dispatches on the gfx950 agent (wavefront 64), of which **20 are `ck::...::kernel_grouped_conv_fwd_multiple_abd_xdl_cshuffle`** (16 + 4, 4.15 ms + 1.41 ms), plus the OIDN input/output/pool/upsample/autoexposure kernels and 16 copyBuffer. The traced XDL kernel: VGPR 112, AGPR 0, SGPR 112, LDS 25088, scratch 0, workgroup 256. Disassembly of the gfx950 code object from `ck_conv_mfma.cpp.o`: 48x `v_mfma_f32_32x32x16_f16` (the double-rate instruction), `.wavefront_size: 64` on all four kernels, LDS 24768 bytes, private segment 0, VGPR spills 0, AGPRs 0, max workgroup 256, VGPRs 111/64. Two of the four kernels report `.sgpr_spill_count: 16` (spilled to VGPR lanes; private segment is 0, so no scratch traffic); the gfx90a validator reported 0 SGPR spills there. Performance note only.
- `oidnBenchmark -d hip` (information only, GPU at 100% from another job): RT hdr_alb_nrm 1920x1080 9.78 ms, 3840x2160 20.44 ms, 1280x720 11.69 ms; RT hdr_calb_cnrm 1920x1080 8.91 ms; RTLightmap 2048x2048 8.44 ms. Noisy (1280x720 slower than 1920x1080) because of the co-tenant; the porter's numbers on a less loaded GPU were 2.85 / 11.98 / 1.41 ms.
- Integrity: `git -C projects/oidn/src status --porcelain` empty at 06bdc4c; `jargon.py --port oidn` clean. ROCm build documentation is upstream's existing README "HIP device" section plus the port's `OIDN_HIP_GFX950`/CDNA text (checked by the review).

### Training smoke run (Q5 ruling), repeated by the validator on gfx950

ROCm PyTorch 2.15.0+rocm10.2.0, OpenImageIO 3.2.2.0, tensorboard 2.13.0 (already in the venv). Synthetic dataset written with OpenImageIO: 24 train and 8 valid scenes, 384x384, `<scene>_0004spp`/`_0016spp`/`_ref` x `.hdr`/`.alb`/`.nrm` `.exr` (half), smooth procedural albedo, normals and lighting with gamma-distributed per-spp noise. 8 validation scenes give 16 validation images, one full batch, so **validation ran this time** (the porter's 4 valid scenes left the validation loader empty). cwd a scratch directory under the job dir, `PYTHONPATH=$ROCM_PATH/share/amd_smi` for the torch processes only (same amdsmi trap as the porter recorded: without it `torch.cuda.device_count()` is 0; with it 1, `gfx950:sramecc+:xnack-`).

```
python training/preprocess.py hdr alb nrm --filter RT -D data -P preproc -t train -v valid
python training/train.py hdr alb nrm --filter RT -P preproc -t train -v valid -R results -r smoke --seed 1 --epochs 20 --valid_epochs 5 --save_epochs 10
python training/train.py ... -r cpu_fp32 -d cpu -p fp32 --seed 1 ...      # CPU reference, OMP_NUM_THREADS=21
python training/infer.py -D data -R results -r smoke -i valid -O infer_gpu --metric psnr ssim
python training/infer.py -D data -R results -r smoke -i valid -O infer_cpu --metric psnr ssim -d cpu
python training/export.py -R results -r smoke -o smoke.tza
oidnDenoise -d cpu --hdr .. --alb .. --nrm .. -w smoke.tza -o ref.pfm; oidnDenoise -d hip ... -w smoke.tza -r ref.pfm
```
- Every step exits 0. GPU run with the defaults (cuda, mixed precision, `torch.compile` reduce-overhead): finished in 1m 4s; epoch 1 56 s (Inductor/Triton compile; a gfx950 MIOpen user find-db from earlier work already existed under `~/.config/miopen`, which is likely why this first epoch took about a minute rather than the porter's 15), later epochs 290-485 images/s.
- Seeded GPU (mixed + compile) vs CPU fp32, same data, `--seed 1`: training loss 0.550259 -> 0.378758 (GPU) vs 0.550258 -> 0.378677 (CPU), every epoch within 1.3e-4; validation loss at epochs 5/10/15/20 0.531636/0.497990/0.409037/0.388879 (GPU) vs 0.531611/0.497995/0.408897/0.388785 (CPU).
- Inference on the 16 validation images with the GPU-trained checkpoint: GPU psnr_avg 3.1431, ssim_avg 0.3156; CPU 3.1428, 0.3156. Per-pixel max abs difference 4.5e-3 on outputs up to 0.81 (max relative 2.5%): `infer.py` runs the mixed-precision model in fp16 on the GPU and casts input to fp16 only off-CPU, so this is an fp16-vs-fp32 inference difference, not a GPU fault. Low PSNR is the 60-step undertrained model, identical on CPU.
- Weights trained on ROCm load into the HIP device: `oidnDenoise -d hip -w smoke.tza` vs `-d cpu -w smoke.tza`, errors=0 at 1920x1080, 1023x577, 257x129, avgerror 1.92e-4, 1.93e-4, 1.78e-4. smoke.tza 7698439 bytes.
- Nothing in `training/` needed changing.

### Not run here, and why
- CUDA no-regression compile: **cuda-not-validated: no CUDA toolkit on this host** (no `nvcc`, no `/usr/local/cuda`, no conda; `/opt/conda/envs/cuda-12.8` is absent). Fetching a package manager (micromamba) into the job directory to install `cuda-toolkit=12.8` was refused by this session's permission classifier, so the gate is still not recorded at 06bdc4c on any host. The diff touches no CUDA source (one `option()` line inside `if(NOT APPLE)` and pass-through lines inside `if(OIDN_DEVICE_HIP)`); a person or a session allowed to install the toolkit owes it before the port is presented.
## Validation 2026-11-23 (linux-gfx942, AMD Instinct MI300X, fork moat-port 06bdc4c)

First run of the MFMA path on gfx942 hardware. One GPU, `HIP_VISIBLE_DEVICES=3` (dispatcher pin), idle at start (0% GFX, 0.3 GB VRAM); other jobs were using other GPUs on the host. All GPU runs under `ulimit -c 0` and `timeout`. Additive evidence: the wave64 gate was already satisfied by linux-gfx90a at this head.

### Toolchain (job directory only)
- ROCm 10.2.0 TheRock SDK and ROCm PyTorch pip-installed into the job venv: `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel,device-gfx90a,device-gfx942,device-gfx950]==10.2.0" "torch[device-gfx942]==2.15.0+rocm10.2.0" OpenImageIO` (pip prints resolver conflicts against the base image's 10.1 `rocm-sdk-device-*` links; harmless). `rocm-sdk version` 10.2.0, `hipconfig --version` 7.17.26434, OpenImageIO 3.2.2.0, tensorboard 2.13.0 from the image. `ROCM_PATH=$(rocm-sdk path --root)`, `PATH` += `$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin`, `LD_LIBRARY_PATH` += `$ROCM_PATH/lib`. No system ROCm used.
- ISPC v1.32.0 and git-lfs v3.6.1 Linux release tarballs extracted under the job directory (the image has no git-lfs); oneTBB 2021.11 from the image.
- Fork clone: `git clone https://github.com/AMD-Ecosystem/oidn projects/oidn/src`, `git checkout moat-port` (06bdc4c), `moatlib.py protect-fork oidn`. Weights (Q4 ruling): `git config submodule.weights.url https://github.com/RenderKit/oidn-weights.git`, `GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive external/composable_kernel weights`, then in `weights/` `git lfs install --local --skip-repo && git lfs pull` (rt_alb.tza 1830792 bytes). `.gitmodules` untouched.

### Build (default target list, all targets)
```
cmake -G Ninja -S projects/oidn/src -B projects/oidn/src/build -DCMAKE_BUILD_TYPE=Release -DOIDN_DEVICE_HIP=ON -DOIDN_DEVICE_CPU=ON \
      -DROCM_PATH=$ROCM_PATH -DHIP_CLANG_NUM_PARALLEL_JOBS=4 -DISPC_EXECUTABLE=<ispc>
cmake --build projects/oidn/src/build -j 64
```
114 s (configure + build), 132 steps, 0 warnings. Code objects (`objcopy --dump-section=.hip_fatbin` + `clang-offload-bundler --list`): `ck_conv_mfma.cpp.o` = gfx90a, gfx942, gfx950 only; `ck_conv_wmma.cpp.o` and `ck_conv_dl.cpp.o` = the 19 RDNA/APU targets, no CDNA; `hip_engine.cpp.o` = all 22.

### Results
- `oidnDenoise --ld`: Device 0 "AMD Instinct MI300X", Type HIP (`getArch` matched the real `gcnArchName` for gfx942); Device 1 the EPYC 9654 CPU.
- `timeout 1800 oidnTest --device hip`: **All tests passed (2361 assertions in 20 test cases)**, 29 s.
- `timeout 3600 oidnTest --device cpu`: **All tests passed (1820 assertions in 20 test cases)**, 11 s.
- CPU-reference image differential (`oidnDenoise -d cpu ... -o ref.pfm`, then `oidnDenoise -d hip ... -r ref.pfm`, default maxerror), deterministic numpy-generated PFM inputs at 1920x1080, 1023x577, 257x129 (seeds 1234-1236; hdr = albedo x shading x exponential noise, ldr clipped, normals normalised, directional in [-1,1]), the same 19 option sets as the gfx950 and gfx90a runs (RT hdr, hdr+alb+nrm, ldr+srgb+alb+nrm at -q high/balanced/fast; hdr+alb; hdr+alb+nrm --clean_aux; ldr+alb; RTLightmap hdr and dir; HIP-only -t half, --maxmem 256, --inplace, --buffer device, --buffer managed, compared with the CPU float output of the same inputs): **57/57, errors=0 in every case**, avgerror 1.86e-4 to 8.60e-4 (RTLightmap dir highest, as on gfx950).
- MFMA path really runs: `rocprofv3 --kernel-trace --stats --output-format csv` on one 1023x577 RT hdr+alb+nrm run shows 20 `ck::...::kernel_grouped_conv_fwd_multiple_abd_xdl_cshuffle` dispatches (16 with AddRelu 0.99 ms total + 4 with Add 0.375 ms; tile 256 threads, 128x64x64, 32x32 XDL) on Agent 5 = gfx942 "AMD Instinct MI300X", wavefront 64, plus the OIDN input/output/pool/upsample/autoexposure kernels. Dispatch resources from the trace: LDS 25088 (allocation granule), scratch 0, workgroup 256. Disassembly of the gfx942 code object unbundled from `ck_conv_mfma.cpp.o`: 96x `v_mfma_f32_32x32x8_f16`, `.wavefront_size: 64`, `.group_segment_fixed_size` 24768, private segment 0, VGPR/SGPR spills 0, VGPRs max 124, SGPRs max 104, AGPRs 0, max workgroup 256. So CK's launch without `IsSupportedArgument` works on gfx942 and the single-rate MFMA numerics agree with CPU.
- `oidnBenchmark -d hip` (information only, idle GPU): RT hdr_alb_nrm 1920x1080 4.94 ms, 3840x2160 19.52 ms, 1280x720 2.23 ms; RT hdr_calb_cnrm 1920x1080 8.59 ms; RTLightmap 2048x2048 9.39 ms. Between the MI250X GCD (17.16 ms at 1080p) and the shared MI355X (2.85 ms); whether a K1=8 tile is faster on gfx942 remains a tuning question.
- Integrity: `git -C projects/oidn/src status --porcelain` empty at 06bdc4c; `jargon.py --port oidn` clean; `moatlib.py audit-clean` OK.

### Training smoke run (Q5 ruling) on gfx942
The porter's recipe (gfx950 section above) with `torch[device-gfx942]==2.15.0+rocm10.2.0`, `PYTHONPATH=$ROCM_PATH/share/amd_smi` on the training commands only (`torch.cuda.device_count()` 1, "AMD Instinct MI300X"). Synthetic dataset written with OpenImageIO (half EXR, 384x384): 16 train and 8 valid scenes, each `sceneNN_0004spp`/`_0016spp`/`_ref` x hdr/alb/nrm (procedural albedo/normals, ref = albedo x shading, noisy = ref x Gamma(spp, 1/spp) noise). Eight valid scenes give 16 validation images, one full batch, so validation ran this time (it never ran in the gfx950 smoke run). Run from a scratch directory with the training scripts by path:
```
python preprocess.py hdr alb nrm --filter RT -D data -P preproc -t train -v valid
python train.py hdr alb nrm --filter RT -P preproc -t train -v valid -R results --epochs 20 --valid_epochs 5 --save_epochs 10 --seed 1 -r gpu_default
python train.py ... --seed 1 -r gpu_fp32 -p fp32 --compile disable
python train.py ... --seed 1 -r cpu_fp32 -d cpu
python infer.py -D data -R results -r gpu_default -i valid -O infer_gpu --metric psnr ssim   (and -d cpu -O infer_cpu)
python export.py -R results -r gpu_default -o smoke.tza
oidnDenoise -d cpu --hdr ... --alb ... --nrm ... -w smoke.tza -o w.pfm; oidnDenoise -d hip ... -w smoke.tza -r w.pfm
```
- Every step exits 0. Nothing in `training/` needed changing; no fork commit.
- Seeded loss, 20 epochs (train / valid at epoch 20): CPU fp32 0.567769 -> 0.493250 / 0.502625; GPU fp32 eager 0.567769 -> 0.493250 / 0.502625, max per-epoch difference from CPU 1e-6 over all 24 train and valid values; GPU defaults (mixed precision + `torch.compile(mode='reduce-overhead')`) 0.567756 -> 0.493245 / 0.502624, max difference from CPU 1.5e-5.
- Epoch 1 took 22 min (defaults) and 21 min (fp32 eager) on the GPU: Inductor/Triton compile and MIOpen first-use find with a cold user find-db (`~/.config/miopen` was being written, GPU at 87-95% busy); later epochs 180-350 images/s. CPU epochs about 6 images/s. A slow first epoch is tuning, not a hang.
- GPU vs CPU inference with the same checkpoint: PSNR 3.5656 vs 3.5654 dB, SSIM 0.2251 both (16 images); per-pixel max abs difference 4.3e-4 on outputs of mean magnitude 2.6e-2. Low PSNR = 40-step undertrained model, identical on CPU.
- The weights trained on ROCm load into the HIP device: `oidnDenoise -d hip -w smoke.tza` vs `-d cpu -w smoke.tza`, errors=0, avgerror 7.98e-5 (1023x577) and 7.82e-5 (257x129). `smoke.tza` 7698439 bytes.

### Not run here, and why
- CUDA no-regression compile: no CUDA toolkit on this host; still owed by the mi355 host at this head (see the gfx90a section).

## CUDA no-regression compile at 06bdc4c (linux mi355 host, nvcc 12.8, 2026-11-23)

Compile and link only; no GPU used. Run-time behaviour of the CUDA device: **cuda-not-validated**.

Toolchain: nvcc 12.8 (V12.8.93, `/usr/local/cuda-12.8`), host gcc 13.3.0 (Ubuntu 24.04), CMake 3.31.6 and Ninja 1.13.0 from the job venv. Fresh clone of the fork; submodules `external/cutlass` d55f6be and `weights` 28883d1 (the `weights` URL pointed at `RenderKit/oidn-weights`, since the relative `../oidn-weights.git` does not exist under the fork's owner); same submodule SHAs at both commits. The CPU device was left off (needs ISPC; the diff does not touch it). Same source directory for both builds, so paths match.
```
git checkout 06bdc4c   # then again with: git checkout master  (6602ee2, merge-base of the port)
cmake -G Ninja -B ../oidn-build-head -DCMAKE_BUILD_TYPE=Release -DOIDN_DEVICE_CUDA=ON -DOIDN_DEVICE_CPU=OFF -DCUDAToolkit_ROOT=/usr/local/cuda-12.8
cmake --build ../oidn-build-head -j 25 -v
```
- Both configure and build exit 0 (head 116 build lines, 26 s). Outputs at both: `libOpenImageDenoise.so`, `libOpenImageDenoise_core.so`, `libOpenImageDenoise_device_cuda.so.2.5.1` (CUDA Driver API, links `libcuda.so.1`, absent on this host as expected), `libcurtn.a`, `oidnTest`, `oidnDenoise`, `oidnBenchmark`. 4 CUDA objects (`cuda_engine.cu`, `cuda_conv.cu`, `cutlass_conv_sm75.cu`, `cutlass_conv_sm80.cu`), gencode sm_75, sm_80, sm_90, sm_100, sm_120 (the CUDA device's own list; `CUDA_ARCHITECTURES OFF`).
- Warnings: 0 at head, 0 at master (`grep -ci warning` over the full verbose logs).
- CUDA sub-build command lines (`ninja -t commands` in `devices/cuda/build`, build-dir path normalised): **identical** at head and master, all 12 commands.
- CUDA sub-build CMake cache: identical. Top-level cache differs only by the new `OIDN_HIP_GFX950:BOOL=ON` entry (declared unconditionally inside `if(NOT APPLE)`, consumed only under `if(OIDN_DEVICE_HIP)`).
- Device code: the 15 cubins extracted with `cuobjdump -xelf all` from the 4 objects. `cuda_conv` and `cuda_engine` cubins are byte-identical (md5) for all five SMs. The two CUTLASS objects' cubins differ in bytes only through nvcc's per-invocation internal-linkage names (`_INTERNAL_<hash>_..._cu_<random>_<pid>`): recompiling the master object a second time with the identical command also changes the md5. With those names normalised, symbol tables (name, value, size) and section sizes are identical at head and master for all 15 cubins. (`nvdisasm` is not in this toolkit install, so no SASS diff.)

**Verdict: no CUDA regression at 06bdc4c.** The CUDA backend configures, compiles and links the same way as at master, with the same flags, no new warnings and equivalent device code. The port's diff reaches CUDA builds only as one unused cache option.

## Validation 2026-11-23 (linux-gfx1100, AMD Radeon Pro W7800 48GB, fork moat-port 06bdc4c): FAILED, pre-existing upstream

Wave32 gate, RDNA3 WMMA convolution path. One GPU, `HIP_VISIBLE_DEVICES=1` (dispatcher pin). All GPU runs under `ulimit -c 0` and `timeout`. Host note: GPU 1 showed 100% GFX activity, ~116 W and 95-97 C junction from before this job's first GPU command until the end, with 0 GB VRAM and no process visible from this container (another container's tenant, presumably). Correctness should not depend on that, but it is recorded because the failure below is nondeterministic.

### Toolchain (job directory only)
- ROCm 10.2.0 TheRock SDK and ROCm PyTorch pip-installed into the job venv: `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel,device-gfx1100]==10.2.0" "torch[device-gfx1100]==2.15.0+rocm10.2.0" OpenImageIO`. `rocm-sdk version` 10.2.0, `hipconfig --version` 7.17.26434. `ROCM_PATH=$(rocm-sdk path --root)`, `PATH` += `$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin`, `LD_LIBRARY_PATH` += `$ROCM_PATH/lib`. No system ROCm.
- ISPC v1.32.0 Linux release tarball under the job directory; oneTBB 2021.11 from the image.
- Fork clone at 06bdc4c, `protect-fork` installed; weights per the Q4 ruling (`git config submodule.weights.url https://github.com/RenderKit/oidn-weights.git`, `GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive external/composable_kernel weights`, then in `weights/` `git lfs install --local --skip-repo && git lfs pull`; rt_alb.tza 1830792 bytes). `.gitmodules` untouched.

### Build (default target list, all targets)
```
cmake -G Ninja -S src -B src/build -DCMAKE_BUILD_TYPE=Release -DOIDN_DEVICE_HIP=ON -DOIDN_DEVICE_CPU=ON \
      -DROCM_PATH=$ROCM_PATH -DHIP_CLANG_NUM_PARALLEL_JOBS=4 -DISPC_EXECUTABLE=<ispc>
cmake --build src/build -j 10
```
153 s, 132 steps, 0 warnings. Code objects: `ck_conv_mfma.cpp.o` = gfx90a, gfx942, gfx950 only; `ck_conv_wmma.cpp.o` and `ck_conv_dl.cpp.o` = the 19 RDNA/APU targets, no CDNA; `hip_engine.cpp.o` = all 22. The gfx1100 WMMA code object: 192x `v_wmma_f32_16x16x16_f16`, 8 kernels, LDS 12384/18528 bytes, private segment 0, VGPR/SGPR spills 0, VGPRs 117-171.

### Results
- `oidnDenoise --ld`: Device 0 "AMD Radeon Pro W7800 48GB", Type HIP (`getArch` -> WMMA).
- `timeout 1800 oidnTest --device hip`: **FAILED, 6 of 20 test cases, 21 of 2475 assertions**. All failures are the output sanity check `isBetween(output, 0.1f, 1.0f)` or `compareImage(...)`: oidnTest.cpp:750 (single filter: 3 frames), :879 x2, :905 x4 (multiple filters), :974 (multiple devices), :1008/:1018 (lazy/eager release), :1126/:1144 (inplace filter: same buffer, aliased shared buffers), :1239 x9 (filter update). The first execute in "single filter" (:741) passes; repeated executes fail.
- `timeout 3600 oidnTest --device cpu`: **All tests passed (1820 assertions in 20 test cases)**.
- CPU-reference image differential (same 19 option sets x 3 sizes as the gfx90a/gfx950 runs, fresh numpy PFM inputs at 1920x1080, 1023x577, 257x129, seeds 1234-1236): **57/57 FAIL, every value of every image wrong** (errors == values), avgerror 0.013 to 0.47 (RT hdr+alb+nrm balanced 1920x1080: 0.059).
- Nondeterministic: three runs of `oidnDenoise -d hip --hdr hdr_257x129.pfm` on identical input give three different images (md5 differ, max abs difference between two runs 6.17, avgerror vs CPU 0.066 / 0.008 / 0.056; output mean 1.52/1.51/1.48 vs CPU 1.43). `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3` does not fix it (avgerror 0.063/0.062/0.061), so it is not a cross-kernel or copy ordering race; it is inside a kernel or deterministic-plus-intra-kernel race. `--buffer device` is the same. LDR inputs (no autoexposure) fail too.
- WMMA path really runs: `rocprofv3 --kernel-trace --stats` on one 1023x577 RT hdr+alb+nrm run shows 20 `ck::kernel_grouped_conv_multiple_d_wmma_cshuffle` dispatches (9 + 7 + 3 + 1 across four instantiations) plus the OIDN input/output/pool/upsample/autoexposure kernels and 18 copyBuffer. No DL or XDL kernel dispatches.
- The GPU itself computes correctly through other WMMA users: with ROCm PyTorch on the same pin, a 4096^2 fp16 matmul (hipBLASLt) has max relative error 3.5e-4 vs CPU and 20 runs are bit-identical; a 64->64 3x3 fp16 conv on 1x64x577x1023 (MIOpen) has max relative error 1.5e-3 and 20 runs are bit-identical.

### Bisection: not caused by the port
Same toolchain (ROCm 10.2.0), same GPU, same default target list, separate worktrees of the fork clone:
- upstream `master` 6602ee2: `oidnTest --device hip` **6 of 20 test cases, 21 of 2242 assertions failed** (same test cases); differential 1023x577 errors=1770813/1770813, avgerror 0.058.
- PR 302 head 75bc1d9: **6 of 20 test cases, 21 of 2463 assertions failed**; differential 257x129 errors=99459/99459, avgerror 0.060.
- port 06bdc4c: as above.
- upstream `master` 6602ee2 built with an older TheRock ROCm, 7.11.0 (`rocm-sdk-devel`/`rocm-sdk-core` 7.11.0 from `https://repo.amd.com/rocm/whl/gfx110X-all/`, `pip install --no-deps --target <jobdir>/deps/rocm711`, `_devel.tar` extracted by hand, AMD clang 22.0.0git, run against its own 7.11 `libamdhip64.so.7`; `HIP_PLATFORM=amd` is needed or `hip-config.cmake` stops with "Unexpected HIP_PLATFORM"): still wrong, 257x129 avgerror 0.034, 1023x577 avgerror 0.050, every value.
So the wrong output is in upstream's own RDNA3 WMMA path (or the shared kernels on wave32) as built by both current TheRock compilers, before PR 302 and our commit. Our commit and PR 302 add nothing that the gfx1100 build executes differently (the WMMA TU's flags only gained `--no-offload-arch` for CDNA targets).

### Not decided here: toolchain vs upstream code vs this host
Upstream validates with ROCm 6.4.2+ (README) and ships prebuilt Linux binaries built with the HIP device. Running the v2.5.1 release binaries (`oidn-2.5.1.x86_64.linux.tar.gz`, extracted under the job directory) on this GPU is the decisive check: if they pass, the fault is a compiler change between upstream's toolchain and ROCm 7.11/10.2 against the CK pin 3e6d21ad (a toolchain bug report plus a block); if they fail too, it is upstream code or this GPU. This session's permission classifier refused running downloaded prebuilt binaries, so it was not done. Also worth running on a second gfx1100/gfx1101 card (this job was confined to GPU 1, which ran hot and busy throughout).

### Training smoke run (Q5 ruling) on gfx1100: passes
ROCm PyTorch 2.15.0+rocm10.2.0, OpenImageIO 3.2.2.0, tensorboard 2.13.0. Synthetic dataset written with OpenImageIO: 24 train and 8 valid scenes, 384x384, `<scene>_0004spp`/`_0016spp`/`_ref` x `.hdr`/`.alb`/`.nrm` `.exr` (half), procedural albedo/normals/lighting with gamma-distributed per-spp noise (48 training, 16 validation images, so validation runs). cwd a scratch directory; `PYTHONPATH=$ROCM_PATH/share/amd_smi` only on the torch processes (`torch.cuda.device_count()` 1, `gfx1100`).
```
python training/preprocess.py hdr alb nrm --filter RT -D data -P preproc -t train -v valid
python training/train.py hdr alb nrm --filter RT -P preproc -t train -v valid -R results -r smoke --seed 1 --epochs 20 --valid_epochs 5 --save_epochs 10
python training/train.py ... -r cpu_fp32 -d cpu -p fp32 --seed 1 ...      # CPU reference, OMP_NUM_THREADS=16
python training/infer.py -D data -R results -r smoke -i valid -O infer_gpu --metric psnr ssim
python training/infer.py -D data -R results -r smoke -i valid -O infer_cpu --metric psnr ssim -d cpu
python training/export.py -R results -r smoke -o smoke.tza
oidnDenoise -d cpu --hdr .. --alb .. --nrm .. -w smoke.tza -o ref.pfm; oidnDenoise -d hip ... -w smoke.tza -r ref.pfm
```
- Every training step exits 0. GPU run with the defaults (cuda, mixed precision, `torch.compile` reduce-overhead): 38m48s total; epoch 1 took 35m49s (Inductor/Triton compile and MIOpen first-use tuning on a loaded host, load average ~50), later epochs 3-7 s (6-14 images/s). Training loss 0.529486 -> 0.446761, validation loss at epochs 5/10/15/20 0.504455/0.475821/0.459313/0.456841.
- Seeded CPU fp32 reference, same data, `--seed 1`: epoch 1/2/3 loss 0.529474/0.510556/0.510113 vs GPU 0.529486/0.510546/0.510114 (within 1.2e-5). Stopped after 3 epochs (about 5 min per epoch on this host; the full 20 would have taken 1.5 h more).
- Inference on the 16 validation images with the GPU-trained checkpoint: GPU psnr_avg 3.7943, ssim_avg 0.3099; CPU identical to four decimals. Per-pixel max abs difference 5.2e-3 on outputs up to 0.78 (fp16 inference on GPU vs fp32 on CPU, as on gfx950). Low PSNR is the 60-step undertrained model.
- Export: smoke.tza 7698439 bytes. Denoising with it: CPU device fine; the HIP device on gfx1100 is close but not within the default tolerance: 1920x1080 errors=17319/6220800 avgerror 1.63e-3, 1023x577 5738/1770813 avgerror 1.60e-3, 257x129 513/99459 avgerror 2.75e-3. That is the same WMMA-path defect as above, much smaller with these near-initial weights than with the shipped ones; gfx950 gave errors=0 with its smoke weights.
- Nothing in `training/` needed changing: ROCm PyTorch training, inference and export work on gfx1100.

### Other
- CUDA no-regression compile: cuda-not-validated: no CUDA toolkit on this host. Still owed at 06bdc4c (mi355 host).
- Integrity: `git -C projects/oidn/src status --porcelain` empty at 06bdc4c; `jargon.py --port oidn` clean. Bisect worktrees live outside the fork clone's tracked tree.

### Verdict
`validation-failed` on linux-gfx1100 at 06bdc4c: the HIP device produces wrong, nondeterministic output on RDNA3 with ROCm 10.2 (and 7.11). The fault predates the port (upstream master and PR 302 fail identically), so the porter's job is to find where it lives (start with the release-binary check above, then CK's WMMA grouped-conv kernel vs a newer CK pin), not to revert anything of ours. If the release binaries pass, this becomes a toolchain bug report and a `blocked` flag rather than a code fix.

## Validation follow-up 2026-11-24 (linux-gfx1100, second W7800 card, fork moat-port 06bdc4c): still FAILED; root cause is a Composable Kernel LDS-barrier bug, fixed in CK rocm-7.2.0

Requested by Jeff Daily: was the 2026-11-23 failure the card? Different physical GPU this time: `HIP_VISIBLE_DEVICES=2` (dispatcher pin) is the W7800 at PCI 0000:23:00.0 (UUID ...e10d0963767a506a); the 2026-11-23 run's `HIP_VISIBLE_DEVICES=1` maps to PCI 0000:46:00.0 (HIP order on this host follows rocminfo: 0x43, 0x46, 0x23, 0x26). The card was idle and cool at start (0% GFX, 34 C edge, 28 MB VRAM, ECC counters 0). All GPU runs under `ulimit -c 0` and `timeout`.

### Toolchain (job directory only)
Same recipe as the 2026-11-23 run: `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel,device-gfx1100]==10.2.0" "torch[device-gfx1100]==2.15.0+rocm10.2.0"` (rocm-sdk 10.2.0, hipconfig 7.17.26434), `ROCM_PATH=$(rocm-sdk path --root)`, ISPC v1.32.0 tarball under the job directory, `$FLEET_VENV/bin` first on PATH. Weights per the Q4 ruling (`git config submodule.weights.url https://github.com/RenderKit/oidn-weights.git`, `GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive external/composable_kernel weights`, `git lfs install --local --skip-repo && git lfs pull` in `weights/`). Build: the default-target-list command from the 2026-11-23 section, `-DHIP_CLANG_NUM_PARALLEL_JOBS=2`, `cmake --build -j $MAX_JOBS` (10); 0 warnings everywhere.

### (1) GPU health: the card is fine
ROCm PyTorch on the pinned GPU, 20 repetitions each against a CPU fp64 reference: 4096^2 fp16 matmul max error 4.9e-4 of max|ref|, 20/20 bit-identical; 64->64 3x3 fp16 conv on 1x64x577x1023 max error 1.5e-3 of max|ref|, 20/20 bit-identical.

### (2) Reproduces on this card, nondeterministic
| build | `oidnTest --device hip` x3 | CPU-reference differential x3 (19 option sets x 3 sizes, same sets as before) |
|---|---|---|
| port 06bdc4c | 6/20 cases failed each run (21, 22, 22 of ~2480 assertions) | 0/57, 0/57, 0/57 (every value wrong) |
| upstream master 6602ee2 | 6/20 cases failed each run (22, 22, 22 assertions) | 0/57, 0/57, 0/57 |

Same failing assertions as the 2026-11-23 run (oidnTest.cpp:741 (sometimes), 750, 879 x2, 905 x4, 974, 1008, 1018, 1126, 1144, 1239 x9). avgerror 0.012 to 0.33 across the 342 HIP runs; RT hdr+alb+nrm balanced 1920x1080 avgerror 0.057/0.058/0.060 (port), 0.064/0.066/0.059 (master). Three runs of `oidnDenoise -d hip --hdr --alb --nrm` on identical 257x129 and 1023x577 inputs give 3 distinct outputs each. `oidnTest --device cpu`: all passed (1820 assertions in 20 test cases). So it is not the card: two different W7800s, one idle and cool, fail identically.

### (3) Narrowing: Composable Kernel at the pin, `block_sync_lds()` on gfx11
- **Smallest reproducer, no OIDN code**: a 150-line HIP program that instantiates exactly OIDN's two WMMA instances (`DeviceGroupedConvFwdMultipleD_Wmma_CShuffle`, fp16 NHWC, 3x3, pad 1, fp32 accumulate, bias AddRelu, MNKPadding, 128 threads, 256x32x32 and 128x64x32 tiles, K1 8) against CK 3e6d21ad, compares to a CPU reference and repeats. Built with `clang++ -x hip --offload-arch=gfx1100 -mcumode -mno-wavefrontsize64 -O3` (OIDN's flags). At 129x257 with (C,K) in {16..160}x{16..112}: 20 of 22 cases wrong and nondeterministic (0.003% to 0.5% of elements wrong per run, rel. error up to 6.7); at 64x64 mostly right but 2 cases still fail. So it does not depend on image size in principle (a bigger grid only makes the race more likely), nor on OIDN.
- **Compiler flags**: device `-O0` gives 0/22 failures; `-O1` 15/22, `-O3` 20/22; `-mno-cumode` (WGP mode) 22/22; `NumGemmKPrefetchStage=2` 8/22. Timing-dependent, as expected of a race.
- **CK pin**: the same program against CK develop head (1a8b8b860) passes 22/22, deterministic. Bisecting ROCm/composable_kernel between 3e6d21ad and 1e6bbed1f (both ends tested) gives the first good commit as **331273b47 "Fix multiple test failures with staging compiler. (#3103)"** (Illia Silin, 2025-10-28). It changes only `include/ck/utility/synchronization.hpp`: `block_sync_lds()` used `__builtin_amdgcn_s_waitcnt(0xc07f)` on every non-gfx12 target. That immediate is the gfx9/gfx10 encoding of `s_waitcnt lgkmcnt(0)`. gfx11 moved the counter fields, so the same immediate decodes as `vmcnt(48) lgkmcnt(7)`: the barrier no longer waits for the wave's own LDS reads and writes. The fix adds a gfx11 branch with `0xfc07` (= `lgkmcnt(0)` on gfx11). First CK release tag containing it: `rocm-7.2.0`; OIDN pins `rocm-6.4.3-132-g3e6d21ade`.
- **ISA proof**: in the ROCm 10.2 build's gfx1100 code object for `ck_conv_wmma.cpp`, 139 of 144 `s_barrier` are preceded by `s_waitcnt vmcnt(48) lgkmcnt(7)`, several directly after in-flight `ds_store_b128`/`ds_load_b128`. The gfx1030 (DL) path is unaffected (gfx10 uses the gfx9 encoding), as are gfx12 (own branch) and CDNA. The upstream v2.5.1 release library (built with AMD clang roc-7.2.4, per its embedded version string), disassembled statically (not executed: the session's permission classifier refused running the downloaded binaries, as in the previous run), has the same `s_waitcnt vmcnt(48) lgkmcnt(7)` (plus `s_waitcnt_depctr 0xffe3`) before all 144 barriers of its gfx1100 WMMA kernels. So the released binaries carry the same race. Whether it shows on their schedule was not tested.
- **Fix verified in the full library**: (a) the one-line CK change applied to the pin in a scratch worktree (gfx11 branch `__builtin_amdgcn_s_waitcnt(0xfc07)`): reproducer 0/22 failures in 3 runs; OIDN 06bdc4c with it: `oidnTest --device hip` all passed 3/3 (2517, 2538, 2529 assertions), differential 57/57 x3 (avgerror 1.9e-4 to 2.4e-3), 3 identical outputs out of 3 at both sizes. (b) The CK submodule moved to tag `rocm-7.2.4` (a7ed94f71), no OIDN source change: builds clean (0 warnings, all 22 targets incl. MFMA), `oidnTest --device hip` all passed 3/3 (2514, 2535, 2532), differential 57/57 x3, deterministic. Neither was committed or pushed; the fork clone is clean at 06bdc4c.
- Not done: forcing the DL path on gfx1100 (not needed once the race was located), and an older-compiler build (the defect is in CK source, so every compiler emits it; the 2026-11-23 ROCm 7.11 build failed too).

**Component**: Composable Kernel at OIDN's pinned commit (fixed upstream in CK since rocm-7.2.0). Not OIDN's HIP backend code, not the compiler, not the runtime, not the card. It is OIDN upstream's bug (stale CK pin), present before PR 302 and our commit.

### State
Left `validation-failed` on linux-gfx1100 at 06bdc4c (already recorded by the 2026-11-23 run; the evidence does not support `completed`). Integrity: `git -C projects/oidn/src status --porcelain` empty.

### For Jeff: options (a fleet worker cannot decide these)
1. **Porter fix in the fork: bump `external/composable_kernel` to CK `rocm-7.2.0` or later** (tested `rocm-7.2.4` here: builds unchanged, gfx1100 passes). It is a one-line gitlink change, and the CK fix is in a tagged release, so no CK fork is needed. Cost: it changes the CK under every HIP path, so gfx90a/gfx942/gfx950 (MFMA), and gfx1030 if anyone has one, need revalidation at the new head, plus the nvcc compile (unaffected: CUDA uses cutlass). It also widens the diff on top of PR 302 into upstream's submodule pin, which is coordination with the PR 302 author and atafra. A ROCm 6.4.2 build (upstream's stated minimum) against CK 7.2.x was not checked.
2. **Block linux-gfx1100 with the reason and ship the port as is**: `set-blocked oidn linux-gfx1100` with "upstream bug: CK pin 3e6d21ad block_sync_lds uses gfx9 s_waitcnt encoding on gfx11 (fixed CK #3103, rocm-7.2.0); master fails identically". That leaves the wave32 gate unsatisfied unless gfx12 (RDNA4) or another wave32 platform validates it (gfx12 takes the separate, correct branch, so a gfx1200/gfx1201 run would probably pass), or a person waives the gate.
3. **Report it upstream to RenderKit/oidn** (works with either option above). No ROCm or CK bug report is needed: CK already fixed it. Read-only search found no existing oidn issue or PR on this. Draft, not posted:

====
Title: HIP: wrong, nondeterministic output on RDNA3 (stale Composable Kernel pin)

My agent came up with this analysis, but I've read and approved it.

On RDNA3 (gfx1100, Radeon Pro W7800, two different cards) the HIP device gives wrong and run-to-run different images: `oidnTest --device hip` fails 6 of 20 test cases, and every pixel differs from the CPU device beyond the default tolerance. Reproduced on master (6602ee2) built with ROCm 10.2 and 7.11.

Cause: the pinned composable_kernel (3e6d21ad, rocm-6.4.3-132) implements `block_sync_lds()` with `__builtin_amdgcn_s_waitcnt(0xc07f)`, which is the gfx9/gfx10 encoding of `lgkmcnt(0)`. On gfx11 the same immediate decodes to `vmcnt(48) lgkmcnt(7)`, so the WMMA convolution kernels hit `s_barrier` with LDS reads and writes still in flight. The gfx1100 code object has `s_waitcnt vmcnt(48) lgkmcnt(7)` before its barriers; the v2.5.1 release library's gfx1100 kernels contain the same sequence. CK fixed this in ROCm/composable_kernel#3103 (commit 331273b, in tag rocm-7.2.0 and later), adding a gfx11 branch with `0xfc07`.

Moving the submodule to CK `rocm-7.2.4` builds without source changes and fixes it on gfx1100: `oidnTest --device hip` passes 3/3, `oidnDenoise -d hip` matches the CPU device on 57/57 configurations, and the output is deterministic. A standalone CK grouped-conv WMMA test using the same two instances reproduces the bug at the old pin and passes with the fix. gfx10 (DL path) and gfx12 are not affected. gfx1150/1151/1152 are gfx11 too and use the same kernels.
====

## Lesson for cuda-to-rocm: Composable Kernel before rocm-7.2.0 races on gfx11 LDS barriers
A project that vendors or pins Composable Kernel older than tag rocm-7.2.0 and runs its WMMA (or any LDS-staged) kernels on RDNA3/RDNA3.5 (gfx11xx) can get wrong, nondeterministic results, while the same code is correct on gfx10, gfx12 and CDNA. `ck::block_sync_lds()` there emits `__builtin_amdgcn_s_waitcnt(0xc07f)`: the gfx9/gfx10 encoding of `lgkmcnt(0)`, which on gfx11 decodes to `vmcnt(48) lgkmcnt(7)`, so `s_barrier` does not wait for LDS operations. Diagnose by disassembling the gfx11 code object and looking for `s_waitcnt vmcnt(48) lgkmcnt(7)` right before `s_barrier`. Symptoms also go away at device `-O0`. Fixed in CK #3103 (331273b, rocm-7.2.0); moving the pin to a 7.2.x tag is enough for OIDN (no source change). Hand-written `s_waitcnt` immediates are architecture-specific in general: gfx9/10, gfx11 and gfx12 each encode the counters differently.

## Correspondence 2026-11-25: PR 302 author agrees to our commits on their branch

Jeff contacted the author of RenderKit/oidn#302 (`wensongc`, AMD) on Teams. Their reply (2026-11-25, relayed by Jeff): we may push to their PR branch; nobody at Intel has responded to them and they do not know the timeline; the oidn owner does not seem very active. PR state checked read-only the same day: open, head `wensongc:hip-gfx950`, `maintainerCanModify` true, last updated 2026-11-06, no reviews. Delivery route per the Q1 ruling: our commits go onto their branch (a push Jeff approves and runs himself), not a competing PR. Nothing pushed yet; the port's gates (wave32 at gfx1100 failing on the stale Composable Kernel pin, Windows) are not met.

## Ruling (Jeff Daily, 2026-11-25): CK bump inside PR 302

Option 1 of "For Jeff: options": bump `external/composable_kernel` in the fork to CK tag `rocm-7.2.4`. No separate upstream issue (the draft above is not to be filed); the fix ships inside PR 302 as one more commit on top of `moat-port`, with PR 302's own commit left as its author wrote it.

## Porter (linux-gfx1100, 2026-11-25): CK submodule to rocm-7.2.4, fork moat-port 130cb6d

One new commit on `moat-port`, no amend, rebase or force-push: **130cb6d** `[ROCm] HIP: update Composable Kernel to rocm-7.2.4 to fix RDNA3 output` on top of 06bdc4c. It changes only the `external/composable_kernel` gitlink, 3e6d21ad -> **a7ed94f71cb412178ad034c98c05724c37083efa** (the commit the annotated tag `rocm-7.2.4`, tag object 3fdb9b30, names). `git merge-base --is-ancestor 331273b47 a7ed94f71` true (the `block_sync_lds` fix is in); `git merge-base --is-ancestor 3e6d21ad a7ed94f71` also true (the old pin is an ancestor, 799 commits; `include/` diff 677 files). No OIDN source change was needed: the build is clean as is. CK's `include/ck/config.h.in` gained only `CK_USE_WMMA_FP8` (left undefined by OIDN's `configure_file`, like `CK_USE_WMMA`). Message: gfx11 cause in three sentences with CK commit 331273b and ROCm/composable_kernel#3103, AI disclosure in prose, Test Plan with literal commands, no trailer; `prose.py` clean, ASCII; `jargon.py --port oidn` clean; `audit-commits` reports only the four known findings on 75bc1d9 (P3), none on 130cb6d. Pushed `06bdc4c..130cb6d moat-port`; `advance-head oidn 130cb6d`.

### Toolchain (job directory only, one GPU `HIP_VISIBLE_DEVICES=2` = W7800 PCI 0000:23:00.0, idle at start)
- ROCm 10.2.0 TheRock wheels into the job venv: `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "rocm[libraries,devel,device-gfx1100,device-gfx1030,device-gfx1200,device-gfx1201,device-gfx90a,device-gfx942,device-gfx950]==10.2.0"`. `rocm-sdk path --root` = `$FLEET_VENV/lib/python3.11/site-packages/_rocm_sdk_devel`; `rocm-sdk version` 10.2.0; `hipconfig --version` 7.17.26434. `$FLEET_VENV/bin` first on PATH, then `$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin`; `LD_LIBRARY_PATH=$ROCM_PATH/lib`.
- **Compiler the builds actually invoke** (from the caches): `OIDN_DEVICE_HIP_COMPILER` = HIP sub-build `CMAKE_CXX_COMPILER` = `$ROCM_PATH/llvm/bin/clang++` (symlink to `lib/llvm/bin/clang++`), **AMD clang version 24.0.0git (ROCm/llvm-project bdc35841c97f)**. Host parts `/usr/bin/c++`.
- ISPC v1.32.0 Linux release tarball under the job directory; oneTBB from the image. Weights per the Q4 ruling (URL override, `GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive external/composable_kernel weights`, `git lfs install --local --skip-repo && git lfs pull` in `weights/`; rt_alb.tza 1830792 bytes); `.gitmodules` untouched.

### Build with ROCm 10.2 (default target list, i.e. every RDNA/APU target plus gfx90a/gfx942/gfx950)
```
cmake -G Ninja -S projects/oidn/src -B projects/oidn/src/build -DCMAKE_BUILD_TYPE=Release -DOIDN_DEVICE_HIP=ON -DOIDN_DEVICE_CPU=ON \
      -DROCM_PATH=$ROCM_PATH -DHIP_CLANG_NUM_PARALLEL_JOBS=4 -DISPC_EXECUTABLE=<ispc>
cmake --build projects/oidn/src/build -j 16
```
112 s, 132 steps, **0 warnings**. Code objects (`objcopy --dump-section .hip_fatbin` + `clang-offload-bundler --list`): `ck_conv_mfma.cpp.o` = gfx90a, gfx942, gfx950 only; `ck_conv_wmma.cpp.o` and `ck_conv_dl.cpp.o` = the 19 RDNA/APU targets (gfx1030-1036, gfx1100-1103, gfx1150-1152, gfx1200, gfx1201, gfx902/909/90c). So the **RDNA2 (gfx1030), RDNA4 (gfx1200/gfx1201) and CDNA (gfx90a/gfx942/gfx950) code paths all compile** against CK rocm-7.2.4 in this one build.
- ISA: gfx1100 WMMA code object, 144 `s_barrier`, every one preceded (after its last `ds_*` op) by `s_waitcnt lgkmcnt(0)`; **0** `s_waitcnt vmcnt(48) lgkmcnt(7)` (was 139/144 at the old pin). gfx1201: barriers preceded by `s_wait_dscnt 0x0` (gfx12 branch, unchanged). gfx1030 DL code object: 20 `s_barrier`, all after `s_waitcnt lgkmcnt(0)`.

### Results on gfx1100 (ROCm 10.2 build)
- `oidnDenoise --ld`: Device 0 "AMD Radeon Pro W7800 48GB", Type HIP, PCI 0000:23:00.0.
- `timeout 1800 oidnTest --device hip` x3: **All tests passed** each time (2517, 2532, 2535 assertions in 20 test cases; the count varies run to run on every platform, as recorded before).
- `timeout 3600 oidnTest --device cpu`: All tests passed (1820 assertions in 20 test cases).
- CPU-reference differential, the same 19 option sets x 3 sizes as every earlier validation (RT hdr, hdr+alb+nrm, ldr+srgb+alb+nrm at -q high/balanced/fast; hdr+alb; hdr+alb+nrm --clean_aux; ldr+alb; RTLightmap hdr and dir; HIP-only -t half, --maxmem 256, --inplace, --buffer device, --buffer managed against the CPU float output of hdr+alb+nrm), fresh deterministic numpy PFM inputs (1920x1080, 1023x577, 257x129, seeds 1234-1236; hdr = albedo x shading x exponential noise, ldr clipped, normals normalized, directional N(0,0.5) clipped to [-1,1]), `oidnDenoise -d cpu -o ref.pfm` then `oidnDenoise -d hip -r ref.pfm`, default maxerror: **57/57 errors=0 in each of three runs**, avgerror 1.59e-4 to 7.8e-3 (RTLightmap dir highest, 6.9e-3 to 7.8e-3; RT hdr+alb+nrm balanced 1920x1080 6.58e-4). **The avgerror of every one of the 57 cases is identical across the three runs.**
- Determinism: three `oidnDenoise -d hip --hdr --alb --nrm` runs on identical input give bit-identical output (md5) at 257x129, 1023x577 and 1920x1080 (1 distinct of 3 at each size; the old pin gave 3 distinct of 3).

### ROCm 6.4.2 (upstream's stated minimum) against CK rocm-7.2.4: configures, compiles, and passes
- Packages: `comgr`, `hip-dev`, `hip-runtime-amd`, `hipcc`, `hsa-rocr`, `hsa-rocr-dev`, `rocm-core`, `rocm-device-libs`, `rocm-llvm`, `rocminfo`, `rocprofiler-register`, all `*.60402-120~24.04` from `https://repo.radeon.com/rocm/apt/6.4.2/` (noble), SHA256 of each checked against that repository's `Packages` index, unpacked with `dpkg -x` into `<jobdir>/deps/rocm642` (nothing installed into the image).
- Build in a clean environment (`env -i`, PATH = venv bin + `<rocm642>/opt/rocm-6.4.2/bin` + `/usr/bin:/bin`, no ROCm 10.2 on any path; `build.ninja` has no `_rocm_sdk` reference):
  ```
  cmake -G Ninja -S projects/oidn/src -B <jobdir>/build642 -DCMAKE_BUILD_TYPE=Release -DOIDN_DEVICE_HIP=ON -DOIDN_DEVICE_CPU=OFF \
        -DROCM_PATH=<rocm642>/opt/rocm-6.4.2 -DHIP_CLANG_NUM_PARALLEL_JOBS=4
  cmake --build <jobdir>/build642 -j 16 -- -k 0
  ```
  Compiler `<rocm642>/opt/rocm-6.4.2/llvm/bin/clang++`, **AMD clang 19.0.0git (roc-6.4.2 25224)**. 103 steps, **0 errors, 0 warnings**; the CDNA probe accepted gfx90a, gfx942 and gfx950. Code objects: MFMA gfx90a/gfx942/gfx950, DL and WMMA the 19 RDNA/APU targets; gfx1100 WMMA 144 barriers, 0 `vmcnt(48) lgkmcnt(7)`.
- Beyond the compile check, the 6.4.2-built library was run against its own 6.4.2 runtime (`LD_LIBRARY_PATH=<rocm642>/lib`, `ldd` shows its `libamdhip64.so.6` and `libhsa-runtime64.so.1`) on the W7800: `oidnDenoise --ld` lists it, `oidnTest --device hip` **All tests passed (2475 assertions in 20 test cases)**, and the CPU-reference differential (references from the 10.2 build's CPU device; the 6.4.2 build has no CPU device) **57/57 errors=0**.
- So the bump implies **no new minimum**: ROCm 6.4.2 stays valid; the documented minimum is unchanged.

### Not changed, and why
- No CHANGELOG line for the CK update (the request was a gitlink-only commit). If upstream wants one, it is a one-line follow-up for whoever talks to the PR 302 author.
