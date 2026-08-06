# Futhark MOAT Port Plan

## Project

- **Name:** futhark
- **Upstream:** https://github.com/diku-dk/futhark
- **Default branch:** main
- **Type:** Functional array programming language compiler with existing HIP backend

## Existing AMD support

**Status: MATURE AUTHORITATIVE HIP SUPPORT**

Futhark has an official HIP backend (`futhark hip` command) that has been part of the project since 2022 (issue #2003, PR #2008). The HIP backend is maintained by the upstream Futhark team at DIKU (University of Copenhagen).

The HIP backend:
- Lives in `rts/c/backends/hip.h` (951 lines)
- Uses HIPRTC for runtime kernel compilation
- Supports unified/managed memory via `hipMallocManaged`
- Has official documentation at `docs/man/futhark-hip.rst`
- Is actively maintained with recent additions (f16 hardware support in v0.27.0)

**KNOWN LIMITATION (wave64 vs wave32):**

The HIP backend hardcodes `lockstep_width = 32` even on CDNA (wave64) hardware (lines 659-665 of hip.h):

```c
// FIXME: in principle we should query hipDeviceAttributeWarpSize
// from the device, which will provide 64 on AMD GPUs.
// Unfortunately, we currently do nasty implicit intra-warp
// synchronisation in codegen, which does not work when this is 64.
// Once our codegen properly synchronises intra-warp operations, we
// can use the actual hardware lockstep width instead.
ctx->lockstep_width = 32;
```

This is a deliberate design choice with a documented FIXME. The scan algorithm (in `ImpGen/GPU/Base.hs`) also hardcodes `chunk_size = 32`. The Futhark developers are aware that true wave64 support requires fixing implicit intra-warp synchronization assumptions in the code generator -- a non-trivial compiler change, not a simple constant flip.

**Decision: VALIDATE-AND-ASSESS (not skip, not from-scratch port)**

This is a validation-and-improvement scenario per PORTING_GUIDE: the existing HIP backend is authoritative (upstream-maintained) but potentially below best practices due to the wave64 gap. MOAT's value is:

1. **Validate** the existing HIP backend on real gfx90a/MI250X with ROCm 7.x
2. **Assess** whether the wave32 emulation on wave64 hardware causes correctness issues
3. **Document** any test failures or performance concerns
4. **Contribute fixes** if practical improvements are identified

We do NOT re-implement the HIP backend from scratch; we validate what exists.

## Build classification

**Not applicable** -- Futhark is a Haskell compiler, not a CUDA/CMake project.

The Futhark compiler itself is built with GHC/Cabal:
- `futhark.cabal` defines the build
- Runtime support is in C (rts/c/) including the HIP backend
- The compiler generates HIP/C code that is compiled via HIPRTC or an external C compiler

## Port strategy

**Strategy: VALIDATE EXISTING AUTHORITATIVE HIP BACKEND**

No code changes expected unless validation reveals issues.

Steps:
1. Build the Futhark compiler (requires GHC)
2. Run `make test-hip` on gfx90a
3. Assess test results with focus on:
   - Any wave64-sensitive failures (scan, reduce operations)
   - Correctness vs reference (sequential C backend)
   - ROCm 7.x compatibility
4. If tests pass, document as "validated" with notes on the wave32 limitation
5. If tests fail, diagnose root causes and assess fix complexity

## CUDA surface inventory

**Not applicable** -- Futhark generates code, it does not contain CUDA kernels to port.

The HIP backend (`rts/c/backends/hip.h`) uses:
- `hipInit`, `hipGetDeviceCount`, `hipDeviceGet`, `hipGetDeviceProperties`
- `hipStreamCreate`, `hipStreamSynchronize`, `hipStreamDestroy`
- `hipMalloc`, `hipMallocManaged`, `hipFree`
- `hipMemcpyHtoD`, `hipMemcpyDtoH`, `hipMemcpyWithStream`
- `hipModuleLoadData`, `hipModuleGetFunction`, `hipModuleLaunchKernel`
- `hipEventCreate`, `hipEventRecord`, `hipEventElapsedTime`, `hipEventDestroy`
- HIPRTC: `hiprtcCreateProgram`, `hiprtcCompileProgram`, `hiprtcGetCode`

All of these are standard HIP runtime API calls -- no CUDA code exists.

## Risk list

1. **Wave64 emulation (KNOWN):** The HIP backend treats wave64 hardware as wave32, which may cause suboptimal scan/reduce performance but should not affect correctness (barriers are inserted when `lockstep_width <= skip_threads`).

2. **ROCm 7.x compatibility:** The HIP backend was developed against earlier ROCm versions. Need to verify HIPRTC API compatibility with ROCm 7.2.1.

3. **Test infrastructure:** The test suite requires a working Futhark compiler. Building from source requires GHC (Haskell toolchain), which may need installation.

4. **gfx90a-specific issues:** The FIXME mentions "on AMD GPUs" generically. Need to confirm behavior specifically on gfx90a/MI250X with wave64.

## File-by-file change list

**Likely no changes needed** -- this is a validation exercise.

If issues are found:
- `rts/c/backends/hip.h` -- potential HIP runtime fixes
- `src/Futhark/CodeGen/ImpGen/GPU/Base.hs` -- wave64 chunk_size changes (complex)

## Build commands

### Install Futhark compiler

Option A: Install via package manager (if available):
```bash
# Check distribution packages or use ghcup
ghcup install ghc
ghcup install cabal
```

Option B: Build from source:
```bash
cd projects/futhark/src
make configure
make build
```

### Run HIP backend tests

```bash
cd projects/futhark/src
make test-hip
```

Or manually:
```bash
cabal run -- futhark test tests -c --backend=hip --tuning=tuning_gpu
```

## Test plan

### GPU tests

1. **Full HIP test suite:**
   ```bash
   make test-hip
   ```
   This runs `futhark test tests -c --backend=hip --tuning=tuning_gpu`

2. **Scan-specific tests:** Focus on scan/reduce operations which are most sensitive to warp-size assumptions:
   ```bash
   cabal run -- futhark test tests/scan* tests/reduce* -c --backend=hip
   ```

3. **Cross-validate against sequential backend:**
   ```bash
   cabal run -- futhark test tests -c --backend=c  # Reference
   cabal run -- futhark test tests -c --backend=hip  # Compare
   ```

### Non-GPU tests (must not regress)

- Unit tests: `make unittest`
- Type checking: `make test-t`
- Interpreter: `make test-interpreter`

### Validation criteria

- All tests that pass on CUDA should pass on HIP
- No incorrect results vs sequential C backend
- No HIPRTC compilation failures on gfx90a
- Reasonable performance (no extreme slowdowns from wave32 emulation)

## Open questions

1. **GHC availability:** Is GHC/Cabal installed on the validation host? If not, what is the installation path?

2. **Wave64 performance:** Even if tests pass, is there measurable performance degradation from treating wave64 as wave32? Would require benchmarking.

3. **Upstream contribution path:** If improvements are identified, the correct path is an upstream PR to diku-dk/futhark, not a MOAT fork. The Futhark team is the authority on their code generator.

4. **ROCm 7.x testing:** Has the Futhark team tested on ROCm 7.x? The recent Windows HIP bug (issue #2204) was on ROCm 6.1.

## Outcome scenarios

### Scenario A: Tests pass, no issues found
- Mark as "validated (existing support)"
- Document the wave32 emulation limitation
- No upstream PR needed
- Optionally file an issue with validation confirmation

### Scenario B: Minor ROCm 7.x compatibility issues
- Fix in upstream PR to diku-dk/futhark
- Not a MOAT fork port

### Scenario C: Correctness failures on wave64
- This would indicate the wave32 emulation is insufficient
- Root cause analysis needed
- Upstream bug report with minimal repro
- Potentially complex compiler fix required (out of MOAT scope)

### Scenario D: Build infrastructure issues
- Document blocker (GHC unavailable, etc.)
- Mark as blocked until resolved
