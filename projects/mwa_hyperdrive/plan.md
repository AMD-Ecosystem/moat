# mwa_hyperdrive Port Plan

## Project

- **Name**: mwa_hyperdrive
- **Upstream**: https://github.com/MWATelescope/mwa_hyperdrive
- **Default branch**: main

## Existing AMD Support

**Status: ALREADY SUPPORTED (mature, authoritative upstream HIP support)**

mwa_hyperdrive has mature, authoritative HIP support directly in the upstream repository. This is not a community fork or an external port -- it is first-party AMD/ROCm support maintained by the MWA Telescope team.

### Evidence

1. **Cargo.toml** (lines 35-36): Dedicated `hip` feature that parallels the `cuda` feature:
   ```toml
   hip = ["mwa_hyperbeam/hip", "hip-sys", "cc"]
   ```

2. **CHANGELOG.md**:
   - v0.3.0 (2023-09-27): "Support for HIP, which allows AMD GPUs to be used instead of only NVIDIA GPUs via CUDA."
   - v0.4.0 (2024-06-19): "rocm6 support"
   - v0.4.1 (2024-07-31): "fix a compile error when specifying env `HIP_FLAGS` with `--features=hip`"

3. **build.rs** (lines 135-224): Complete HIP build configuration including:
   - HIP_PATH detection via `hip_sys::hiprt::get_hip_path()`
   - hipcc compiler selection
   - HYPERDRIVE_HIP_ARCH / HYPERBEAM_HIP_ARCH environment variable support for arch targeting
   - ROCm version compatibility handling

4. **Dockerfile** (lines 19-27): Production Docker builds for Pawsey Setonix MI250 (gfx90a):
   ```dockerfile
   # -> rocm, setonix MI250
   # export ROCM_VER=6.3.3
   # docker build . \
   #   --build-arg=BASE_IMAGE=quay.io/pawsey/rocm-mpich-base:rocm${ROCM_VER}-mpich3.4.3-ubuntu24.04 \
   #   --build-arg=FEATURES=hip \
   #   --build-arg=HIP_ARCH=gfx90a \
   ```

5. **GPU kernel sources** (src/gpu/common.cuh): Full HIP/CUDA abstraction layer with `__HIPCC__` guards for HIP-specific types (hipFloatComplex, hipDoubleComplex) and runtime API mappings (hipMalloc, hipMemcpy, etc.).

6. **Rust bindings** (src/gpu/mod.rs lines 57-64): HIP runtime bindings via hip_sys crate:
   ```rust
   #[cfg(feature = "hip")]
   use hip_sys::hiprt::{
       hipDeviceSynchronize as gpuDeviceSynchronize, ...
   };
   ```

7. **GPU tests** (src/gpu/tests.rs): HIP-specific error message assertions, confirming HIP path is tested.

### Deployment Context

This software is actively used on Pawsey Setonix (AMD MI250X GPUs) for MWA radio telescope calibration. The HIP support is production-grade and actively maintained.

### Warp Size Handling

The code handles warp size correctly:
- common.cuh line 28: `#define __syncwarp __syncthreads` for HIP (conservative, safe for wave64)
- common.cuh line 69: `#define warpSize 32` only under `__CUDACC__` (CUDA section)
- HIP path uses the native `warpSize` from the HIP runtime, which returns 64 on CDNA and 32 on RDNA

The kernels in model.cu and peel.cu are thread-parallel over baselines/frequencies without warp-collective primitives that require explicit warp-width handling. The `warp_reduce` function in peel.cu uses `__syncwarp()` (mapped to `__syncthreads()` on HIP) which is safe.

## Decision

**SKIP (already-supported)**

This project has mature, first-party HIP support for gfx90a. A MOAT port would duplicate the upstream maintainers' work. No port is warranted.

### Build Commands (for reference)

Build with HIP for gfx90a:
```bash
HYPERDRIVE_HIP_ARCH=gfx90a cargo build --release --features=hip
```

Run tests:
```bash
HYPERDRIVE_HIP_ARCH=gfx90a cargo test --release --features=hip
```

## Disposition

The project will be marked `already-supported` via triage.py.
