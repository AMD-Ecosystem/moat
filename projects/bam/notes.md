# bam notes

Keep: jeff notes AMD may have a GPUDirect-storage equivalent (double-check later). The NVMe/GPUDirect path is the hard part; the GPU compute is portable.

AMD reportedly has a GPUDirect-storage equivalent; double-check before porting the NVMe/GPUDirect path.

## Port status (2026-06-04)

**Partial port completed, blocked on hardware requirements and warp-size issues.**

### What works
- libnvm library compiles with HIP for gfx90a
- simt::atomic replaced with hip_atomic.h (HIP-compatible atomics with thread scope)
- PTX inline assembly replaced (lane_id, warp_id, smid, MMIO doorbell writes)
- CMake shim for legacy FindCUDA macros (cuda_add_library/cuda_add_executable)
- cuda_to_hip.h compat header with CUDA->HIP symbol mappings

### Remaining issues
1. **Warp size (HIGH)**: page_cache.h uses uint32_t masks throughout for __activemask(), __match_any_sync(), __shfl_sync(). HIP requires 64-bit masks. Fixing requires changing all mask types to uint64_t and updating ballot/popc operations.

2. **Benchmark warp size (MEDIUM)**: Several benchmarks have #define WARP_SIZE 32 hardcoded. Need wave64-aware abstraction.

3. **switch statement (LOW)**: page_cache.h has a switch that jumps over variable initialization. clang rejects this.

### Hardware requirements (BLOCKING)
BaM requires:
- Custom libnvm kernel module loaded
- NVMe SSD unbound from kernel nvme driver
- IOMMU disabled in BIOS
- PCIe peer-to-peer support between GPU and NVMe
- MI200 (gfx90a) full BAR exposure for P2P

This validation system does not have the required hardware configuration. The kernel module needs to be built and an NVMe drive dedicated to the project.

### To resume
1. Fix warp mask types in page_cache.h (uint32_t -> uint64_t for all __activemask/ballot paths)
2. Add warpSize-aware macros in benchmarks
3. Fix switch statement in page_cache.h
4. Set up hardware: build kernel module, unbind NVMe, disable IOMMU
