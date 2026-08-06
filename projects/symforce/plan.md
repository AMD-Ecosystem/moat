# symforce Port Plan

## Project
- **Name**: symforce
- **Upstream**: https://github.com/symforce-org/symforce
- **Default branch**: main

## Existing AMD Support
**None found.** Searches conducted:
- Grep of upstream docs (`grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/`): no references
- Web search for "symforce ROCm", "symforce AMD GPU", "symforce HIP": no results
- GitHub fork scan (`gh api repos/symforce-org/symforce/forks`): no ROCm/AMD/HIP named forks
- No ROCm PRs/issues in upstream

**Decision**: Port proceeds.

## Build Classification
**cmake** (Strategy A with HIP toolchain)

The CUDA code lives in the **Caspar** module (`symforce/caspar/`) -- an experimental CUDA code generation and execution backend for symbolic computation. Evidence:

- `symforce/caspar/source/templates/buildfiles/CMakeLists.txt.jinja` (lines 7-18): Pure CMake project with `project(caspar_library LANGUAGES CXX CUDA)`, `find_package(CUDAToolkit REQUIRED)`, and explicit `CMAKE_CUDA_ARCHITECTURES`
- Runtime `.cu` files: `shared_indices.cu`, `solver_tools.cu`, `sort_indices.cu`
- Runtime `.cuh` header: `memops.cuh` (main CUDA kernel utilities)
- No PyTorch extension machinery (`torch.utils.cpp_extension` not used)
- The main symforce CMakeLists.txt has no CUDA content; Caspar is a separate generated library

The CUDA backend in `symforce/codegen/backends/cuda/` is pure codegen (generates CUDA source text) with no `.cu` to compile.

**ext_type**: `cmake`

## Port Strategy
**Strategy A: Pure CMake with compat header**

1. Add a `cuda_to_hip.h` compat header in `symforce/caspar/source/runtime/`
2. Modify the Jinja CMakeLists.txt template to support HIP via `enable_language(HIP)` + `LANGUAGE HIP` on `.cu` sources
3. Runtime sources (`memops.cuh`, `shared_indices.cu`, `solver_tools.cu`, `sort_indices.cu`) include the compat header and stay in CUDA spelling

## CUDA Surface Inventory

### Kernels and Device Functions
| File | Kernels/Functions |
|------|-------------------|
| `shared_indices.cu` | `SharedIndicesKernel` (`__global__`), `Sort`, `MakeUnique`, `GetNUnique`, `GetOrd`, `Swap` (`__device__`) |
| `solver_tools.cu` | `AlphaFromNumDenomKernel`, `BetaFromNumDenomKernel` (`__global__`) |
| `sort_indices.cu` | `Arange`, `TargetFromArgsort`, `SelectIndexKernel` (`__global__`) |
| `memops.cuh` | Many `__forceinline__ __device__` template utilities: `ReadIdx*`, `WriteIdx*`, `AddIdx*`, `WriteSum*`, `FlushSum*`, `SumStore`, etc. |

### Warp Intrinsics
| Intrinsic | Location | Risk |
|-----------|----------|------|
| `__syncwarp()` | `shared_indices.cu:55,67` | Low -- used inside Sort for stride <= 32; wave64 safe (syncs full wavefront) |
| `__syncthreads()` | Throughout | None -- standard block sync |
| `cg::tiled_partition<32>` | `memops.cuh:333` | **Medium** -- hardcoded 32-lane tile; works on wave64 (HIP CG supports width-32 tiles) but NOT optimal |
| `cg::reduce(group, ...)` | `memops.cuh:270,314,336,342,375` | **HIGH RISK** -- HIP CG has NO `cg::reduce`; must be replaced with manual butterfly reduction |
| `cg::labeled_partition` | `memops.cuh:263` | **HIGH RISK** -- HIP CG has NO `cg::labeled_partition`; must be replaced with `match_any` + manual group reduce |
| `cg::binary_partition` | `memops.cuh:306,367` | Low -- supported in HIP CG |
| `cg::coalesced_threads()` | `memops.cuh:263,306,367` | Low -- supported in HIP CG |
| `cg::memcpy_async` | `shared_indices.cu:122` | **HIGH RISK** -- HIP CG has NO `memcpy_async`; must fall back to synchronous copy |
| `cg::wait` | `shared_indices.cu:123` | Paired with `memcpy_async` -- remove |

### CUDA Runtime API
| Symbol | Count | HIP Equivalent |
|--------|-------|----------------|
| `cudaMalloc` | - | `hipMalloc` |
| `cudaMemcpy` | 2 | `hipMemcpy` |
| `cudaMemset` | 2 | `hipMemset` |
| `cudaMemcpyDeviceToDevice` | 1 | `hipMemcpyDeviceToDevice` |
| `cudaMemcpyDeviceToHost` | 1 | `hipMemcpyDeviceToHost` |
| `cuda_runtime.h` | 2 | `hip/hip_runtime.h` |
| `cooperative_groups.h` | 2 | `hip/hip_cooperative_groups.h` |
| `cooperative_groups/reduce.h` | 2 | **N/A -- no HIP equivalent** |
| `cooperative_groups/memcpy_async.h` | 2 | **N/A -- no HIP equivalent** |

### Library Dependencies
| CUDA Library | Usage | HIP Equivalent |
|--------------|-------|----------------|
| CUB (`cub/cub.cuh`) | `cub::DeviceReduce::Sum`, `cub::DeviceRadixSort::SortPairs`, `cub::DeviceRadixSort::SortKeys` | hipCUB (`hipcub/hipcub.hpp`) -- 1:1 API |
| CUDAToolkit (`CUDA::cudart`) | CMake link | `hip::host` |

### Textures/Surfaces
None.

### Streams/Events
None explicit in runtime code.

### Pinned/Managed Memory
None.

## Risk List

1. **cg::reduce (HIGH)**: HIP Cooperative Groups (ROCm 7.2.1) has no `cg::reduce`. Must implement a manual butterfly reduction: `for(o=group.size()/2;o>0;o>>=1) v = op(v, group.shfl_xor(v,o))`. Used in 5 locations in `memops.cuh`.

2. **cg::labeled_partition (HIGH)**: HIP CG has no `labeled_partition`. Must replace with `cg::match_any(warp, label)` to get a mask of same-label lanes, then do a masked reduction over that mask. Used in 1 location (`memops.cuh:263`).

3. **cg::memcpy_async / cg::wait (HIGH)**: HIP CG has no async memcpy. Fall back to synchronous block-strided copy + `__syncthreads()`. Used in `shared_indices.cu:122-123`.

4. **Warp size 32 in tiled_partition<32>**: The code uses `tiled_partition<32>` which creates 32-lane tiles. On wave64 (gfx90a), this works correctly (HIP CG supports 32-lane tiles within a 64-lane wavefront) but the code structure assumes 32-lane operations. The `SumStore` function at line 333 does a two-level reduction assuming 32-wide groups -- this is portable but should be verified.

5. **__syncwarp in Sort**: The `__syncwarp` calls in the odd-even merge sort are used when `size <= 32`. On wave64, `__syncwarp` syncs the full wavefront, which is MORE lanes than needed but still correct. No issue.

6. **CUB -> hipCUB**: DeviceRadixSort and DeviceReduce::Sum are 1:1 with hipCUB. The namespace changes from `cub::` to `hipcub::`.

## File-by-File Change List

### New Files
| File | Purpose |
|------|---------|
| `symforce/caspar/source/runtime/cuda_to_hip.h` | Compat header with CUDA->HIP symbol mappings |

### Modified Files
| File | Changes |
|------|---------|
| `symforce/caspar/source/runtime/memops.cuh` | Include compat header; replace `cg::reduce` with butterfly shfl_xor reduction; replace `cg::labeled_partition` with `match_any`+masked-reduce pattern |
| `symforce/caspar/source/runtime/shared_indices.cu` | Include compat header; replace `cg::memcpy_async`/`cg::wait` with synchronous block-strided copy |
| `symforce/caspar/source/runtime/solver_tools.cu` | Include compat header (cudaMem* -> hipMem*, cub -> hipcub) |
| `symforce/caspar/source/runtime/sort_indices.cu` | Include compat header (cub -> hipcub) |
| `symforce/caspar/source/templates/buildfiles/CMakeLists.txt.jinja` | Add `USE_HIP` option; gate `enable_language(HIP)` vs CUDA; mark sources `LANGUAGE HIP`; link `hip::host` instead of `CUDA::cudart` |

### Unchanged Files
- `symforce/codegen/backends/cuda/` -- pure code generation, no .cu compilation
- All non-caspar symforce code -- no CUDA content

## Build Commands

### Configure (gfx90a)
```bash
cd projects/symforce/src
# Generate a caspar library (this creates the CMakeLists.txt from the jinja template)
python -c "
from symforce.caspar import CasparLibrary
import symforce.symbolic as sf
from symforce import typing as T
from symforce.caspar import memory as mem

caslib = CasparLibrary('test_caspar')

@caslib.add_kernel
def simple_kernel(
    a: T.Annotated[sf.V3, mem.ReadSequential],
) -> T.Annotated[sf.V3, mem.WriteSequential]:
    return a * 2

caslib.generate('/tmp/caspar_gen')
"

# Build with HIP
cd /tmp/caspar_gen
cmake -B build \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCASPAR_BUILD_PYTHON_BINDINGS=OFF
cmake --build build -j$(nproc)
```

### Alternative: Build runtime library directly
```bash
cd projects/symforce/src/symforce/caspar/source/runtime
# After porting, the runtime can be built standalone:
cmake -B build \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build build -j$(nproc)
```

## Test Plan

### GPU Tests
The caspar module has example scripts that serve as integration tests:

1. **kernel_example**: `symforce/caspar/examples/kernel_example/gen_and_run.py`
   - Generates a simple kernel, compiles, and runs on GPU
   - Validates output against PyTorch reference
   - Requires: PyTorch with ROCm support

2. **multiple_factors**: `symforce/caspar/examples/multiple_factors/gen_and_run.py`
   - More complex factor graph example
   - Full GPU validation

3. **bal example**: `symforce/caspar/examples/bal/gen_and_run.py`
   - Bundle adjustment on GPU

### Validation Commands
```bash
# Set up PyTorch ROCm environment
# Run kernel example
cd projects/symforce/src
HIP_VISIBLE_DEVICES=0 python symforce/caspar/examples/kernel_example/gen_and_run.py

# Run multiple_factors example
HIP_VISIBLE_DEVICES=0 python symforce/caspar/examples/multiple_factors/gen_and_run.py
```

### Non-GPU Tests (Must Not Regress)
The main symforce test suite (unit tests for symbolic computation, codegen, optimization) is CPU-only and should pass unchanged:

```bash
cd projects/symforce/src
pip install -e ".[dev]"
pytest test/ -v --ignore=test/symforce_cuda_codegen_test.py
```

The `symforce_cuda_codegen_test.py` tests codegen output text (no GPU execution) and should also pass unchanged since it tests the code generator, not compilation.

## Open Questions

1. **PyTorch ROCm integration**: The caspar examples use PyTorch with `device="cuda"`. With ROCm PyTorch, this should work transparently (ROCm PyTorch maps "cuda" device to HIP), but needs verification.

2. **cg::labeled_partition complexity**: The replacement for `labeled_partition` is non-trivial. The `FlushSharedSum` function uses it for coalesced writes to shared output indices. Need to verify the `match_any` + prefix-scan pattern correctly handles the use case.

3. **sm_75 minimum**: The CUDA CMake template sets `CMAKE_CUDA_ARCHITECTURES 75 80 86 89 75-virtual` with comment "sm_75 minimum required by cooperative_groups::reduce". Since we are replacing `cg::reduce` with manual shuffles, this constraint does not apply to HIP. The HIP equivalent is `gfx90a;gfx1100` etc.

4. **Generated kernel templates**: The Jinja templates (`kernel.cu.jinja`, `caspar_mappings.cu.jinja`, etc.) generate CUDA code. These may need HIP variants or USE_HIP guards for generated code if they contain CUDA-specific constructs beyond what the compat header covers.
