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
