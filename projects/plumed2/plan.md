# PLUMED2 cudaCoord Plugin Porting Plan

## Project

- **Name**: plumed2
- **Upstream**: https://github.com/plumed/plumed2
- **Default branch**: main
- **Scope**: The cudaCoord plugin (`plugins/cudaCoord/`) -- a GPU-accelerated coordination number calculator

## Existing AMD Support

**None.** Thorough assessment found no AMD/ROCm/HIP support:
- `grep -rniE 'amd|rocm|hip|gfx' README* docs/` returns only unrelated mentions (e.g. "Chip" in bibliography)
- No forks under ROCm/AMD/GPUOpen orgs or with rocm/hip/amd in name
- No upstream PRs or issues mentioning ROCm/HIP
- WebSearch for "plumed ROCm" / "plumed AMD GPU" / "plumed HIP" / "plumed MI300" returns no HIP port

PLUMED itself is a plugin framework for MD codes (GROMACS, LAMMPS, NAMD). Those MD engines have AMD GPU support via SYCL/HIP, but PLUMED's own CUDA plugin has no HIP path. The cudaCoord plugin is currently NVIDIA-only.

**Decision**: Proceed with a fresh CUDA-to-HIP port. This adds native HIP acceleration for coordination number calculations on AMD GPUs.

## Build Classification

**Strategy A: Pure (non-CMake) Make build**

Evidence:
- `plugins/cudaCoord/Makefile` drives the build with nvcc
- `plugins/cudaCoord/configure.sh` extracts config from the host plumed install (`plumed config makefile_conf`)
- No `setup.py`, `find_package(Torch)`, or PyTorch dependency -- not a pytorch extension
- Not pure CMake; it is a traditional Makefile with nvcc as the compiler

The plugin builds as a standalone `.so` that plumed loads at runtime via its plugin mechanism.

## Port Strategy

**Strategy A (compat-header)** adapted to a Makefile project:

1. Add a `cuda_to_hip.h` compat header that aliases CUDA symbols to HIP equivalents
2. Create a `Makefile.hip` (or modify Makefile with `USE_HIP` conditionals) that:
   - Uses hipcc instead of nvcc
   - Compiles `.cu` as HIP (hipcc recognizes `.cu` and compiles it as HIP code)
   - Links against amdhip64 instead of CUDA libraries
3. Update `configure.sh` to detect ROCm and emit appropriate `Makefile.conf`

Key point: hipcc can compile `.cu` files directly (they are syntactically compatible after the compat header aliases CUDA to HIP), so no file renaming is needed.

## CUDA Surface Inventory

### Runtime API Symbols
| CUDA | HIP Equivalent |
|------|----------------|
| `cudaMalloc` | `hipMalloc` |
| `cudaFree` | `hipFree` |
| `cudaMemcpy` | `hipMemcpy` |
| `cudaMemcpyAsync` | `hipMemcpyAsync` |
| `cudaMemcpyHostToDevice` | `hipMemcpyHostToDevice` |
| `cudaMemcpyDeviceToHost` | `hipMemcpyDeviceToHost` |
| `cudaStream_t` | `hipStream_t` |
| `cudaStreamCreate` | `hipStreamCreate` |
| `cudaStreamDestroy` | `hipStreamDestroy` |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` |
| `cudaFuncAttributes` | `hipFuncAttributes` |
| `cudaFuncGetAttributes` | `hipFuncGetAttributes` |

### Libraries
| CUDA | HIP Equivalent |
|------|----------------|
| `thrust::device_vector` | rocThrust `thrust::device_vector` (same header path) |
| `thrust::host_vector` | rocThrust `thrust::host_vector` (same header path) |
| `thrust::raw_pointer_cast` | rocThrust `thrust::raw_pointer_cast` |
| `cub::BlockReduce` | hipCUB `hipcub::BlockReduce` |
| `cub::LoadDirectBlocked` | hipCUB `hipcub::LoadDirectBlocked` |

### Headers
| CUDA | HIP Equivalent |
|------|----------------|
| `<cuda_runtime.h>` | `<hip/hip_runtime.h>` |
| `<device_launch_parameters.h>` | `<hip/hip_runtime.h>` (defines launch params) |
| `<thrust/device_vector.h>` | same path (rocThrust) |
| `<thrust/host_vector.h>` | same path (rocThrust) |
| `<cub/block/block_load.cuh>` | `<hipcub/hipcub.hpp>` or `<hipcub/block/block_load.hpp>` |
| `<cub/block/block_reduce.cuh>` | `<hipcub/hipcub.hpp>` or `<hipcub/block/block_reduce.hpp>` |

## Risk List

1. **CUB -> hipCUB namespace**: CUB uses `cub::`, hipCUB can use `hipcub::` or `cub::` with the HIPCUB_NAMESPACE_ALIAS flag. The compat header should `#define cub hipcub` on HIP.

2. **wave64 block reduction (cub/hipCUB TempStorage reuse)**: The reduction kernel uses `cub::BlockReduce` with `TempStorage`. On wave64 (gfx90a), a 64-thread block is one wavefront and BlockReduce may lower without syncing epilogue. The code already has `__syncthreads()` between tile loads (line 258, 277 in Coordination.cu), but verify that TempStorage is not reused across multiple collective calls without sync. LOW RISK: the reduction kernels use separate invocations.

3. **rocThrust device_vector**: rocThrust is a drop-in for Thrust with identical API and header paths. No code changes needed. MINIMAL RISK.

4. **Makefile adaptation**: Need to handle:
   - nvcc flags (`-dc`, `-dlto`) -> hipcc equivalents (`-fgpu-rdc`, `--offload-compress` if needed)
   - SM architecture (`--gpu-architecture=sm_XX`) -> `--offload-arch=gfx90a`
   - Link flags (`-shared`, `-dlto`) -> hipcc shared library linking

5. **No warp primitives**: The code has NO `__shfl*`, `__ballot`, `__syncwarp`, `__activemask`, or hardcoded warpSize/32. This is a SIGNIFICANT LOW-RISK factor -- no wave64-vs-wave32 issues.

6. **`nearbyint`/`nearbyintf`**: Used in `pbcClamp()` device functions. HIP supports these device math functions. NO RISK.

7. **`__forceinline__`**: HIP recognizes this. NO RISK.

8. **CCCL dependency**: The README mentions CCCL (NVIDIA container library). For HIP, we use rocThrust/hipCUB instead. Need to verify no CCCL-specific headers are used beyond Thrust/CUB.

9. **LTO (`-dlto`)**: nvcc's device-LTO is NVIDIA-specific. hipcc does not have `-dlto`. Use standard hipcc compilation. The plugin is small (2 source files), so LTO is not critical. LOW RISK.

10. **Rule-of-five on streams**: The code creates/destroys cudaStreams in constructor/destructor. HIP streams work identically. NO RISK.

## File-by-File Change List

### New Files

1. **`src/cuda_to_hip.h`** (NEW)
   - CUDA-to-HIP symbol aliases
   - Include guards for `USE_HIP` / `__HIP_PLATFORM_AMD__`
   - Thrust/CUB namespace aliasing

### Modified Files

2. **`Makefile`** (MODIFY)
   - Add `USE_HIP` option
   - When `USE_HIP=1`:
     - Replace `NVCC=nvcc` with `HIPCC=hipcc`
     - Replace nvcc flags with hipcc equivalents
     - Replace CUDA libs with HIP libs
   - Keep CUDA path unchanged for `USE_HIP=0` (or unset)

3. **`configure.sh`** (MODIFY)
   - Detect `USE_HIP` environment variable or ROCm installation
   - Emit HIP-appropriate compiler flags to Makefile.conf
   - Alternative: create a separate `configure-hip.sh`

4. **`src/cudaHelpers.cuh`** (MODIFY)
   - Add `#include "cuda_to_hip.h"` at top
   - Change `<cub/block/block_load.cuh>` -> conditional include for hipCUB
   - Change `<cub/block/block_reduce.cuh>` -> conditional include for hipCUB

5. **`src/Coordination.cu`** (MODIFY)
   - Add `#include "cuda_to_hip.h"` at top (before cuda_runtime.h)
   - Update CUB includes to conditional hipCUB

6. **`src/Coordination.cuh`** (MODIFY)
   - Add `#include "cuda_to_hip.h"` at top

7. **`src/cudaHelpers.cu`** (MODIFY, if exists or needed)
   - Add compat header include

### Minimal Diff Approach

The compat header defines all the CUDA->HIP symbol mappings, so the actual `.cu` files only need one include line added. The main work is in the Makefile and configure.sh.

## Build Commands

### Configure and Build for gfx90a (Linux)

```bash
# Prerequisites: ROCm 7.x installed, plumed2 built and installed

# Set up environment
export PLUMED_KERNEL=/path/to/libplumed.so  # or wherever plumed kernel is
export USE_HIP=1

# In the plugin directory
cd plugins/cudaCoord

# Configure (generates Makefile.conf)
./configure.sh

# Build
make USE_HIP=1 HIP_ARCHITECTURES=gfx90a
# or equivalently:
# hipcc will be used with --offload-arch=gfx90a

# Output: CudaCoordination.so (the plugin)
```

### Multi-arch Build

For follower platforms, change `HIP_ARCHITECTURES`:
- gfx1100: `make USE_HIP=1 HIP_ARCHITECTURES=gfx1100`
- gfx1101/gfx1201 (Windows): needs adaptation for Windows build

## Test Plan

### Regression Tests (GPU)

The plugin has a built-in regtest suite:

```bash
cd regtest
ln -s ../../../regtest/scripts .   # as per README
make check
```

Tests include:
- `cudatest/rt-double-*`: Double precision tests
- `cudatest/rt-float-*`: Single precision tests
- `cudatestPair/`: Pair mode tests
- `cudatestWB/`: Tests with different configurations

These run plumed with the cudaCoord plugin against test trajectories and compare outputs.

### Validation Criteria

1. All existing regtests pass (comparing HIP output vs CUDA reference values)
2. Determinism: Two runs produce bit-identical output
3. Correctness: Coordination numbers match the CPU reference (`COORDINATION` action in plumed)

### Non-GPU Tests

The plugin's regtests are GPU-focused. The main plumed regtest suite should still pass (plugin loading should not break other plumed functionality).

## Open Questions

1. **plumed installation location**: The configure.sh expects `plumed` in PATH. Need to verify plumed can be built on ROCm host (plumed itself is CPU-only, so this should be fine).

2. **Dynamic loading**: The plugin is loaded via `dlopen`. Verify the HIP-built `.so` can be loaded by plumed.

3. **MPI compatibility**: The plugin uses `comm.Bcast` (MPI). Verify MPI+HIP works (should be fine with standard ROCm MPI setup).

4. **Double precision performance**: gfx90a has good FP64. The plugin supports both float and double; both should work.

5. **Memory management**: Thrust device_vector manages GPU memory. rocThrust should handle this identically.

6. **configure.sh modifications**: Decide whether to:
   - Modify existing configure.sh with USE_HIP detection
   - Create a new configure-hip.sh
   - Use CMake for the plugin (more invasive but cleaner)
