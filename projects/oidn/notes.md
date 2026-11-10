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

