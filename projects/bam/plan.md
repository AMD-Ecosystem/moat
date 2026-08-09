# BAM Port Plan

## Project

- **Name**: bam (Big Accelerator Memory)
- **Upstream**: https://github.com/ZaidQureshi/bam
- **Default branch**: master
- **Description**: GPU-initiated on-demand high-throughput storage access system (ASPLOS'23) -- enables GPUs to directly orchestrate NVMe SSD access without CPU mediation, featuring a software cache and high-throughput concurrent queues.

## Existing AMD support

**None found**. Searched:
- Upstream docs: `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/` found only a BIOS IOMMU reference ("In AMD Systems, this requires disabling `IOMMU` in the BIOS") -- no AMD GPU support.
- GitHub forks via `gh api repos/ZaidQureshi/bam/forks`: 30 forks, none with rocm/hip/amd in name or description.
- Web search: no ROCm/HIP port found.
- No upstream rocm/hip branches or related PRs/issues.

**Decision**: Proceed with a fresh port. No existing AMD work to build on.

## Build classification

**Pure CMake project (Strategy A)**.

Evidence:
- `CMakeLists.txt` line 2: `project (libnvm LANGUAGES CUDA C CXX)`
- Uses legacy FindCUDA: `find_package(CUDA 8.0)` and `CUDA_ADD_LIBRARY`, `cuda_add_executable`
- No PyTorch or torch.utils.cpp_extension involvement
- Lines 253-266: `make_benchmark` macro uses `cuda_add_executable(... OPTIONS ${GPU_ARCHS} -D__CUDA__)`

## Port strategy

**Strategy A (compat header + CMake HIP enablement)** with legacy-FindCUDA shim.

Rationale:
1. Pure CMake build with no PyTorch dependency.
2. The project uses legacy FindCUDA (`cuda_add_library`, `cuda_add_executable`) rather than modern `enable_language(CUDA)`, so the port requires a shim macro approach (per PORTING_GUIDE "Strategy-A on LEGACY FindCUDA module" entry).
3. Under `USE_HIP`, define `cuda_add_library` and `cuda_add_executable` shims that call `add_library`/`add_executable` then mark sources `LANGUAGE HIP`.
4. A single `cuda_to_hip.h` compat header aliases CUDA runtime symbols to HIP.

## CUDA surface inventory

### Header includes
- `<cuda.h>` (driver API, nvm_util.h line 12)
- No `<cuda_runtime.h>` explicit includes found (the runtime is pulled via cuda.h or CUDA_LIBRARIES)

### Runtime calls
| CUDA symbol | Count | Files |
|-------------|-------|-------|
| `cudaMalloc` | ~15 | benchmarks/*, include/ctrl.h, include/queue.h |
| `cudaMemcpy` | ~20 | benchmarks/*, include/ctrl.h, include/queue.h |
| `cudaMemset` | ~5 | benchmarks/* |
| `cudaFree` | ~5 | benchmarks/* |
| `cudaSetDevice` | ~10 | benchmarks/* |
| `cudaGetDeviceProperties` | ~5 | benchmarks/*settings.h |
| `cudaDeviceSynchronize` | ~10 | benchmarks/* |
| `cudaDeviceGetPCIBusId` | ~3 | benchmarks/* |
| `cudaEvent*` | 5 | include/event.h |

### Device/kernel annotations
- `__global__` kernels: minimal inline kernels in benchmarks (bfs, cc, pagerank, etc.)
- `__device__` functions: ~50 in include/page_cache.h, include/nvm_queue.h, include/nvm_util.h, include/util.h
- `__host__ __device__`: ~30 across include/*.h
- `__forceinline__`: ~20 across include/*.h

### Warp intrinsics
| Intrinsic | Count | Files |
|-----------|-------|-------|
| `__activemask()` | ~10 | include/page_cache.h, include/util.h |
| `__shfl_sync()` | ~15 | include/page_cache.h |
| `__syncwarp()` | ~12 | include/page_cache.h, benchmarks/reduction/main.cu |
| `__match_any_sync()` | 5 | include/page_cache.h |
| `lane_id()` | 3 | include/nvm_util.h (PTX asm), include/util.h |
| `warp_id()` | 1 | include/nvm_util.h (PTX asm) |
| `get_smid()` | 1 | include/nvm_util.h (PTX asm) |
| `__popc()` | 5 | include/util.h, include/host_util.h |
| `__ffs()` | 3 | include/page_cache.h, include/host_util.h |

### PTX inline assembly (nvm_util.h lines 276-295)
```c
__forceinline__ __device__ uint32_t lane_id() {
    uint32_t ret;
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}
__forceinline__ __device__ unsigned warp_id() {
    unsigned ret;
    asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}
__forceinline__ __device__ uint32_t get_smid() {
    uint32_t ret;
    asm  ("mov.u32 %0, %smid;" : "=r"(ret) );
    return ret;
}
```

### MMIO inline assembly (nvm_parallel_queue.h)
```c
asm volatile ("st.mmio.relaxed.sys.global.u32 [%0], %1;" :: "l"(sq->db),"r"(new_db) : "memory");
```

### Hardcoded warp size 32
| Pattern | Files |
|---------|-------|
| `#define WARP_SIZE 32` | benchmarks/bfs/main.cu, benchmarks/pagerank/main.cu, benchmarks/reduction/main.cu, benchmarks/vectoradd/scan.cu |
| `tid / 32` | ~10 occurrences in benchmarks/cache, pattern, vectoradd, readwrite |
| `>> (32 - lane)` | include/util.h line 112 (`prior_mask`) |
| `512/32` | include/page_cache.h line 1306 (`warp_memcpy`) |

### simt/atomic usage (CRITICAL DEPENDENCY)
- **Submodule**: `include/freestanding` -> github.com/ogiroux/freestanding (commit 6360203)
- Uses `simt::atomic<T, simt::thread_scope_device>` and `simt::thread_scope_system` extensively
- `#include <simt/atomic>` in include/nvm_types.h line 10, include/ctrl.h line 31, include/page_cache.h (via nvm_types.h)
- The `simt::atomic` provides C++ standard-library-style atomics with thread-scope memory ordering semantics

This is the **biggest risk**: the freestanding library is CUDA-specific (NVCC/Clang+CUDA) and has no HIP/ROCm equivalent. The ROCm/libhipcxx project provides `cuda::std::atomic` but NOT the `simt::` namespace with its thread-scope variants.

### Library usage
- `cub::DeviceScan::InclusiveSum` (benchmarks/scan/main.cu only, commented out in main CMakeLists.txt)
- No cuBLAS, cuFFT, cuRAND, cuSPARSE, cuDNN, Thrust usage in active code

### Textures/surfaces
None found.

### Pinned/managed memory
None found (uses standard cudaMalloc).

### Streams/events
- `cudaEvent_t`, `cudaEventCreate`, `cudaEventRecord`, `cudaEventDestroy`, `cudaEventElapsedTime` (include/event.h)
- `cudaStream_t` usage minimal

## Risk list

### CRITICAL: simt::atomic (ogiroux/freestanding) has no HIP equivalent

The `simt::atomic` header-only library provides CUDA-specific atomics with thread scope (`thread_scope_device`, `thread_scope_system`) and memory ordering. This is fundamental to the page cache and queue implementation.

**Options**:
1. **Port freestanding to HIP**: Significant work -- requires understanding the underlying PTX and mapping to AMD GCN/CDNA.
2. **Use hip::std::atomic from libhipcxx**: ROCm/libhipcxx (github.com/ROCm/libhipcxx) provides `cuda::std::atomic` and `hip::std::atomic`. However, the thread_scope variants may need adaptation.
3. **Replace with HIP native atomics**: Convert `simt::atomic<T, scope>::fetch_add/store/load` to explicit `atomicAdd`/`atomicExch` + `__threadfence_block()`/`__threadfence()` calls.

**Recommendation**: Option 3 is most tractable -- replace `simt::atomic` usage with direct HIP atomic intrinsics plus appropriate fences. The usage is localized to nvm_types.h and page_cache.h structures.

### HIGH: PTX inline assembly

Three PTX instructions need HIP equivalents:
- `mov.u32 %0, %laneid` -> `__lane_id()` HIP builtin
- `mov.u32 %0, %warpid` -> `threadIdx.x / warpSize` or `__builtin_amdgcn_workitem_id_x() / __builtin_amdgcn_wavefrontsize()`
- `mov.u32 %0, %smid` -> `__smid()` HIP builtin (but note the smid upper-bound issue from PORTING_GUIDE)
- `st.mmio.relaxed.sys.global.u32` -> This is for MMIO doorbell writes to NVMe controller registers. HIP has `__atomic_store_n` with `__ATOMIC_RELAXED` or may need platform-specific volatile store.

### HIGH: Warp size hardcoding

Multiple `#define WARP_SIZE 32` and `/32` patterns. CDNA (gfx90a) has wavefront 64, RDNA (gfx1100/gfx1151) has wavefront 32. Must abstract:
- Compile-time: `#ifdef __GFX9__ ... 64 ... #else ... 32`
- Runtime host: `hipGetDeviceProperties().warpSize`
- The `prior_mask = mask >> (32 - lane)` in util.h is particularly tricky -- assumes 32-bit mask.

### MEDIUM: 64-bit ballot masks

HIP `__shfl_sync`, `__ballot_sync`, etc. require 64-bit masks (ROCm 7.x static_asserts sizeof(mask)==8). The code uses `uint32_t mask = __activemask()` which is 32-bit. Per PORTING_GUIDE, define a compat full-warp mask (`0xffffffffffffffffULL` for HIP).

### MEDIUM: __match_any_sync

Used in page_cache.h for coalescing threads accessing the same page. HIP has `__match_any_sync` but need to verify behavior with wave64.

### LOW: Legacy FindCUDA CMake

The build uses `find_package(CUDA 8.0)` + `cuda_add_library`/`cuda_add_executable`. Need shim macros under USE_HIP.

### LOW: Kernel module

The project includes a Linux kernel module (`module/`) for NVMe access. This is pure C and independent of CUDA/HIP, but validation requires the module to load.

### PLATFORM: PCIe P2P requirements

BaM requires PCIe peer-to-peer between GPU and NVMe. NVIDIA Tesla GPUs expose full BAR for P2P. AMD MI200 (gfx90a) also supports PCIe P2P but the driver/runtime support paths differ. The kernel module may need adaptation for AMD GPU memory mapping.

## File-by-file change list

### New files
- `include/cuda_to_hip.h` -- compat header with CUDA->HIP aliases
- `cmake/hip_compat.cmake` -- HIP build macros (cuda_add_library/cuda_add_executable shims)

### Modified files

**CMakeLists.txt**:
- Add `option(USE_HIP "Build with HIP for AMD GPUs" OFF)`
- Add `if(USE_HIP) include(cmake/hip_compat.cmake) ... endif()` before the cuda_add_library calls
- HIP architecture default: `if(NOT DEFINED CMAKE_HIP_ARCHITECTURES) set(CMAKE_HIP_ARCHITECTURES "gfx90a") endif()`

**include/nvm_util.h**:
- Replace PTX `%laneid`/`%warpid`/`%smid` with HIP-compatible implementations
- Guard with `#if defined(__HIP_PLATFORM_AMD__)`

**include/nvm_types.h**:
- Replace `#include <simt/atomic>` with HIP-compatible atomic abstractions
- Replace `simt::atomic<T, scope>` with direct atomic intrinsics + fences

**include/page_cache.h**:
- Replace `simt::atomic` usage
- Fix 64-bit ballot mask requirement for `__shfl_sync`, `__activemask`
- Abstract warp size in warp_memcpy and related

**include/util.h**:
- Replace PTX `__activemask()` guard
- Fix `prior_mask = mask >> (32 - lane)` for wave64

**include/host_util.h**:
- Add HIP equivalents for host-side fallbacks

**include/ctrl.h**:
- Replace `simt::atomic` usage

**include/event.h**:
- Map cudaEvent* to hipEvent*

**benchmarks/**/main.cu**:
- Replace `#define WARP_SIZE 32` with arch-aware constant
- Fix hardcoded `/32` warp calculations

**include/nvm_parallel_queue.h**:
- Replace PTX MMIO store with HIP-compatible volatile store or atomic

## Build commands

### Configure (gfx90a)
```bash
git submodule update --init --recursive
mkdir -p build && cd build
cmake .. \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -Dno_cuda=ON
```

### Build
```bash
make libnvm -j$(nproc)
make benchmarks -j$(nproc)
cd module && make -j$(nproc)
```

## Test plan

### Kernel module (no GPU required)
```bash
cd build/module
sudo make reload
ls /dev/libnvm*  # should show device nodes
```

### GPU benchmarks (requires NVMe SSD unbound from nvme driver)

**Block benchmark (I/O stack)**:
```bash
sudo ./bin/nvm-block-bench --threads=262144 --blk_size=64 --reqs=1 \
  --pages=262144 --queue_depth=1024 --page_size=512 \
  --num_blks=2097152 --gpu=0 --n_ctrls=1 --num_queues=128 --random=true
```

**Array benchmark (cache + I/O)**:
```bash
sudo ./bin/nvm-array-bench --threads=$((1024*1024)) --blk_size=64 --reqs=1 \
  --pages=$((1024*1024)) --queue_depth=1024 --page_size=512 \
  --gpu=0 --n_ctrls=1 --num_queues=128 --random=false
```

### Validation criteria
1. Kernel module loads without error
2. Block benchmark completes and reports throughput
3. Array benchmark completes and reports cache hit/miss stats
4. Run-to-run determinism (same input produces same output)

### Non-GPU regression
- Kernel module compile (pure C)
- Library host code compile

## Open questions

1. **simt::atomic replacement strategy**: Should we port freestanding to HIP, use libhipcxx, or replace with raw atomics? The raw atomics approach is most tractable but requires careful memory ordering analysis.

2. **PCIe P2P on AMD MI200**: Does the AMD driver expose GPU BAR memory for NVMe P2P access the same way NVIDIA Tesla does? The kernel module may need AMD-specific GPU memory mapping code.

3. **MMIO doorbell writes**: The `st.mmio.relaxed.sys.global.u32` PTX is used for writing NVMe doorbell registers. What is the correct HIP equivalent for MMIO stores with relaxed ordering?

4. **Validation hardware**: GPU validation requires a system with:
   - AMD MI200/MI300 (gfx90a/gfx94x) with full BAR exposed
   - NVMe SSD that can be unbound from the kernel nvme driver
   - IOMMU disabled
   - PCIe switch or direct connection (no IOMMU path)

   If such hardware is not available in the MOAT validation environment, this port may need to be blocked pending hardware access.

5. **Wave64 warp coalescing**: The page cache's `__match_any_sync` based coalescing assumes 32 threads per warp. With wave64, the coalescing granularity changes. Need to verify correctness.
