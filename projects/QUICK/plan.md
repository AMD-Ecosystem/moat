# QUICK Porting Plan

## Project

- Name: QUICK
- Upstream: https://github.com/merzlab/QUICK
- Default branch: main (at bc80f98)

## Existing AMD Support Assessment

**Verdict: AUTHORITATIVE but INCOMPLETE/DISABLED -- requires validation and enablement work, not from-scratch port**

### Background

QUICK is a quantum chemistry package with explicit HIP support developed by the Merz/Goetz labs (UCSD/MSU). The AMD GPU support was published in:

> Manathunga, M.; Aktulga, H. M.; Goetz, A. W.; Merz, K. M. "Quantum Mechanics/Molecular Mechanics Simulations on NVIDIA and AMD Graphics Processing Units." J. Chem. Inf. Model. 63, 711-717 (2023).

The publication demonstrated comparable performance between NVIDIA V100 and AMD MI100.

### Current State

1. **Configure script BLOCKS HIP builds** (line 1171-1173):
   ```bash
   echo  "Error: HIP support is currently unavailable in this QUICK release. "
   echo  "       Support will be added back in a future release. "
   exit 1
   ```
   This was added due to ROCm compiler bugs in intermediate versions.

2. **CMake HIP path EXISTS and is MAINTAINED**:
   - `quick-cmake/QUICKCudaConfig.cmake` has full HIP detection and configuration
   - Supports gfx908, gfx90a, gfx942 architectures
   - ROCm version guard: blocks 5.4.3 <= version < 6.2.1 due to scalar register fill/spill bug
   - ROCm 6.2.1+ and ROCm <= 5.4.2 are explicitly supported

3. **Dedicated HIP source tree**: `src/gpu/hip/` with:
   - Complete CMakeLists.txt for HIP kernel compilation
   - rocBLAS/rocSOLVER Fortran bindings
   - HIP-specific versions of all GPU kernels

4. **GitHub Issue #433**: Recent user report (May 2026) confirms:
   - HIP/ROCm 6.2.1 builds require CMake workarounds
   - ROCm 6.x changed rocBLAS module path
   - HIP version detection fails with certain ROCm install layouts
   - User reports 3x performance regression vs AmberTools23 QUICK

### Classification

This is **authoritative upstream HIP support** from the original developers that:
- Was disabled due to ROCm compiler bugs (now fixed in 6.2.1+)
- Has bit-rotted against ROCm 6.x API changes
- Needs validation and minor fixes, NOT a from-scratch port

## Build Classification

**cmake** (Strategy A equivalent, but build system already exists)

Evidence:
- CMakeLists.txt at root with `project(quick)`
- `quick-cmake/QUICKCudaConfig.cmake` handles CUDA/HIP configuration
- `src/gpu/hip/CMakeLists.txt` for HIP kernels
- Fortran + C++ + CUDA/HIP mixed-language build
- No PyTorch/torch dependency

## Port Strategy

**VALIDATE-AND-FIX existing upstream HIP support** (not Strategy A from-scratch)

The MOAT value is:
1. Remove the configure script block on HIP
2. Fix CMake for ROCm 6.2.1+ (rocBLAS module path, version detection)
3. Build and validate on gfx90a with ROCm 7.2.x
4. Run the test suite to confirm correctness
5. Investigate the 3x performance regression reported in issue #433
6. Contribute fixes upstream (this is the maintainers' own code)

## CUDA/HIP Surface Inventory

### GPU Kernels (all in src/gpu/ and src/gpu/hip/):
- `gpu.cu` - Main GPU module (146KB)
- `gpu_get2e.cu` - Two-electron integrals (37KB)
- `gpu_getxc.cu` - Exchange-correlation (116KB)
- `gpu_oei.cu` - One-electron integrals (10KB)
- `gpu_oeprop.cu` - Electrostatic properties (10KB)
- `gpu_lri.cu` - Long-range interactions (13KB)
- `gpu_MP2.cu` - MP2 correlation (610KB)
- `gpu_get2e_grad_ffff.cu` - ERI gradients with f-functions (34KB)

### Warp Primitives
- **No `__shfl*`, `__ballot`, `__activemask`, `warpSize` found** in HIP sources
- Code appears warp-size agnostic (no wave64 fault class)

### Atomics
- Uses `atomicAdd` for Fock matrix accumulation
- Architecture flags: `-munsafe-fp-atomics` for gfx90a, `-DUSE_LEGACY_ATOMICS` for older (gfx906/908)
- Float atomicAdd on gfx90a is hardware-supported with unsafe-fp-atomics

### Library Dependencies
- rocBLAS (via Fortran module bindings in `src/gpu/hip/rocblas/`)
- rocSOLVER (via Fortran module bindings in `src/gpu/hip/rocsolver/`)
- MAGMA optional (for eigensolvers)

### Runtime Features
- hipMalloc/hipFree, hipMemcpy, hipDeviceSynchronize
- hipStream_t for async operations
- Texture/surface NOT used (unlike many image-processing CUDA codes)

## Risk List

1. **ROCm version detection** - Current regex assumes ROCm path like `/opt/rocm-X.Y.Z/`; system paths like `/software/.../6.2.1` break it (issue #433)

2. **rocBLAS module path changed** - ROCm 6.x moved `rocblas_module.f90` from `${ROCM_PATH}/rocblas/include/` to `${ROCM_PATH}/include/rocblas/` - code already handles this with version check but user reports it still fails

3. **Performance regression** - User reports 3x slowdown vs previous version; root cause unknown (could be compiler regression, missing optimization flags, or changed defaults)

4. **No RDNA support** - Only gfx908/gfx90a/gfx942 architectures defined; no gfx1100/gfx1151 support currently

5. **Configure script bypass needed** - The shell configure script must have the HIP block removed/commented

6. **AmberTools integration complexity** - QUICK is often built as part of AmberTools; standalone build should work but integration build has additional issues

## File-by-file Change List (Minimal for Validation)

1. **configure** (line 1171-1173): Comment out or remove the `exit 1` block for HIP
2. **quick-cmake/QUICKCudaConfig.cmake**: 
   - May need to update ROCm version regex for non-standard paths
   - Verify rocBLAS module path logic for ROCm 7.2.x
3. No source code changes expected for basic validation

## Build Commands

### Configure (gfx90a, ROCm 7.2.x)
```bash
export ROCM_PATH=/opt/rocm
mkdir build && cd build
cmake .. \
    -DHIP=ON \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_Fortran_COMPILER=gfortran \
    -DQUICK_USER_ARCH=gfx90a \
    -DCMAKE_BUILD_TYPE=Release
```

### Build
```bash
make -j$(nproc)
```

### Alternative: configure script (after removing HIP block)
```bash
./configure --hip --prefix=$PWD/install
make hipinstall
```

## Test Plan

### GPU Tests (primary validation)
```bash
cd test
../tools/runtest --hip --ene --grad
```

Test categories:
- `--ene`: Energy calculations (HF/DFT with various basis sets)
- `--grad`: Gradient calculations
- `--opt`: Geometry optimization
- `--api`: API tests
- `--esp`: Electrostatic potential

### Non-GPU Regression (serial build)
```bash
./configure --serial --prefix=$PWD/install
make serialinstall
./tools/runtest --serial --ene --grad
```

### Determinism Check
Run same calculation twice, compare outputs for bit-identical results.

## Open Questions

1. **Root cause of 3x performance regression** in issue #433 - is this a ROCm 6.x compiler issue, missing optimization flags, or something else?

2. **RDNA (wave32) support** - Should gfx1100/gfx1151 be added? The code appears warp-agnostic but is untested.

3. **ROCm 7.2.x compatibility** - The code is validated for 6.2.1; need to verify 7.2.x works

4. **Upstream merge policy** - This is the maintainers' own code; they will likely accept fixes via PR

## Decision

**VALIDATE-AND-IMPROVE, not from-scratch port**

MOAT value: Validate the existing authoritative HIP support on gfx90a/ROCm 7.2.x, fix any bit-rot, and contribute fixes upstream. The performance regression investigation is secondary to correctness validation.

## Delta Plan: gfx1100/gfx1151

Not currently supported by upstream. After gfx90a validation:
1. Add `gfx1100` to `QUICK_USER_ARCH` options in QUICKCudaConfig.cmake
2. Verify no wave64-specific assumptions (preliminary grep shows none)
3. Build and test on RDNA hardware

