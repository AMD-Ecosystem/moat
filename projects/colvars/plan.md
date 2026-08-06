# Colvars -- plan.md

## Project

- Name: colvars
- Upstream: https://github.com/Colvars/colvars
- Default branch: main

## Existing AMD support

**Status: Mature ROCm/HIP support already upstream -- skip (already-supported).**

The Colvars project has comprehensive, production-ready HIP support merged directly into the upstream repository:

1. **PR #810** (merged 2025-06-26): "Support the NAMD CudaGlobalMaster interface on HIP"
2. **PR #888** (merged): "build: guard the CUDAGM interface by NAMD_CUDA and NAMD_HIP macros"
3. **PR #890** (merged): "Copy Colvars CUDA source code to all supported platforms"
4. **PR #894** (merged 2026-04-22): "Add HIP support for the unit tests" -- validated on MI250X (gfx90a) and Radeon 610M
5. **PR #933** (merged 2026-05-12): "refactor: update the implementation of COLVARS_SYNC_WARP on HIP"

### Evidence of HIP support

- `src/colvar_gpu_support.h`: Full CUDA-to-HIP compat header (~100 mappings) with proper `COLVARS_HIP` guards
- `src/cuda/*.cu`: All kernel files include hipcub headers and alias `cub` to `hipcub` under COLVARS_HIP
- `tests/functional_gpu/CMakeLists.txt`: Full HIP build path with `-DGPU_TYPE=HIP`
- `namd/cudaglobalmaster/CMakeLists.txt`: USE_HIP option for NAMD integration
- NAMD 2.15 alpha HIP/ROCm builds available at https://www.ks.uiuc.edu/Research/namd/alpha/2.15_amdhip/

The project correctly handles wave-size differences using the HIP `warpSize` builtin and logical-warp CUB/hipCUB reductions.

## Decision

No MOAT port is needed. The upstream already supports AMD GPUs via HIP with proper CMake integration, and the implementation follows best practices. MOAT would duplicate AMD's existing work.

## Disposition

- Set to `blocked` with reason: already-supported
- All platforms blocked (no need for lead or follower validation)
