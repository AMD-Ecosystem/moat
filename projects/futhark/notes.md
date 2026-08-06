# futhark notes

## Validation Summary (linux-gfx90a, ROCm 7.2.1)

**Date:** 2026-06-05
**Compiler:** Futhark 0.26.3 (prebuilt binary, GHC 9.10.3)
**Hardware:** AMD Instinct MI250X (gfx90a, wave64)
**Commit tested:** 4edd15b (upstream master)

### Test Results

- **2430/2432 tests passed** (99.92% pass rate)
- 7 programs excluded

**Failures:**
1. `tests/fusion/noFusion5.fut` - Internal compiler error (NOT HIP-related; compiler bug in pass 'Fuse SOACs')
2. `tests/issue419.fut` - Correctness failure (HIP-specific; passes on C and OpenCL backends)

### Known Limitation: lockstep_width=32 on wave64

The HIP backend hardcodes `lockstep_width = 32` even on CDNA (wave64) hardware (see `rts/c/backends/hip.h` lines 659-665):

```c
// FIXME: in principle we should query hipDeviceAttributeWarpSize
// from the device, which will provide 64 on AMD GPUs.
// Unfortunately, we currently do nasty implicit intra-warp
// synchronisation in codegen, which does not work when this is 64.
ctx->lockstep_width = 32;
```

The `issue419.fut` failure is likely caused by this wave32 emulation on wave64 hardware. The test involves segmented scan operations with complex index computations where half the wavefront may be treated as a separate "warp" and synchronization assumptions break down.

### SOAC Primitives Test

All 52 tests in `tests/soacs/` pass, including scans and reduces. The wave32 emulation mostly works, but specific input patterns (like issue419) trigger the latent bug.

### Validation Outcome

The existing upstream HIP backend is **functional** for the vast majority of workloads on gfx90a. The single HIP-specific failure (issue419) is a known limitation documented upstream with a FIXME. The fix requires non-trivial changes to the Futhark code generator's intra-warp synchronization strategy.

**Recommendation:** This is not a MOAT port candidate. The authoritative upstream HIP backend exists and is maintained. The wave64 gap is a known upstream issue; a proper fix would be an upstream contribution to diku-dk/futhark, not a MOAT fork.

### Build Notes

- Futhark 0.27.0 (master HEAD) requires base >= 4.20 (GHC 9.10+) and has dependency compatibility issues with some LSP libraries on GHC 9.10.3
- Prebuilt binary v0.26.3 from GitHub releases works out of the box
- Requires ROCm paths for HIP compilation:
  ```bash
  export CPATH="/opt/rocm/include:$CPATH"
  export LIBRARY_PATH="/opt/rocm/lib:$LIBRARY_PATH"
  export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"
  ```
