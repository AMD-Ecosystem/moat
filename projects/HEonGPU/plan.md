# HEonGPU Porting Plan

## Project
- **Name:** HEonGPU
- **Upstream:** https://github.com/Alisah-Ozcan/HEonGPU
- **Default branch:** main
- **Description:** High-performance Fully Homomorphic Encryption (FHE) library optimized for GPUs, supporting BFV, CKKS, and TFHE schemes

## Existing AMD support

**None found.** Assessment performed 2026-06-05:
- Upstream docs grep (`grep -rniE 'amd|rocm|hip|gfx' README* docs/`): No AMD/ROCm references
- Web search ("HEonGPU ROCm", "HEonGPU AMD GPU", "Alisah-Ozcan HEonGPU AMD"): No AMD port found
- GitHub forks: No forks under ROCm/AMD/GPUOpen orgs or with rocm/hip/amd in name
- Upstream branches: No rocm/hip branches
- Upstream issues/PRs: None mentioning AMD/ROCm/HIP

**Decision:** Proceed with a fresh HIP port. The project is CUDA-only; adding ROCm/HIP support enables it on AMD GPUs.

## Build classification

**Pure CMake (Strategy A)** -- NOT a PyTorch extension.

Evidence:
- Root CMakeLists.txt line 7: `project(HEonGPU VERSION 1.1 LANGUAGES C CXX CUDA ASM)`
- Line 34: `find_package(CUDAToolkit REQUIRED)`
- Line 35: `find_package(Thrust REQUIRED)`
- No `find_package(Torch)`, no `setup.py`, no `torch.utils.cpp_extension`

Set `ext_type: cmake` in upstream.json and status.json.

## Port strategy

**Strategy A** -- single `cuda_to_hip.h` compat header, mark `.cu` sources `LANGUAGE HIP`, minimal diff.

Rationale:
- Pure CMake build with `.cu` kernel sources
- No PyTorch dependency
- Standard CUDA runtime API usage (`cudaMalloc`, `cudaMemcpy`, `cudaStream_t`, etc.)
- cuRAND device API (`curand_init`, `curand`, `curandState_t`)
- Minimal Thrust usage (only `thrust/host_vector.h` in one header)

The port adds:
1. `src/include/heongpu/cuda_to_hip.h` compat header
2. CMake option `USE_HIP` with `enable_language(HIP)`, `set_source_files_properties(... LANGUAGE HIP)`
3. Fix warp-size assumptions (the main risk class)
4. Library substitutions: cuRAND -> hipRAND, RMM -> hipMM (or RMM branch-25.08 which has nascent HIP support)

## CUDA surface inventory

### Main library (98 .cu/.cuh files in src/)

**CUDA runtime API:**
- `cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemcpyDeviceToDevice` -- map to `hipMemcpy*`
- `cudaStream_t`, `cudaStreamDefault`, `cudaStreamCreate` -- map to `hipStream_t`
- `cudaEvent_t` -- map to `hipEvent_t`
- `cuda_runtime.h` includes (4 files) -- redirect via compat header

**cuRAND device API:**
- `<curand_kernel.h>` (6 includes)
- `curandState_t`, `curand_init`, `curand` -- map to `hiprand_kernel.h`, `hiprandState_t`, `hiprand_init`, `hiprand`

**Thrust:**
- `#include <thrust/host_vector.h>` (1 location, memorypool.cuh) -- rocThrust drop-in

**Warp intrinsics (RISK):**
- `warpSize` used at runtime (3 locations: util.cuh:309, decryption.cu:449, encryption.cu:291, keygeneration.cu:1087)
- `__shfl_down_sync(0xFFFFFFFF, ...)` in `warp_reduce()` (util.cuh:312)
- Hardcoded `>> 5` (divide by 32) and `& 31` in warp-id/lane calculations

**No usage of:**
- Textures/surfaces
- cudaMallocManaged / unified memory
- cuBLAS / cuFFT / cuSPARSE
- Cooperative groups
- PTX inline assembly

### Thirdparty submodules (76 .cu/.cuh files)

GPU-NTT, GPU-FFT, RNGonGPU also need porting:
- 181 references to warp-size-32 patterns (`& 31`, `>> 5`, `<< 5`)
- `warpThreadIndex = threadIdx.x & 31` in RNGonGPU/aes.cu
- Block-tiled NTT kernels with implicit 32-thread tile assumptions

These are from the same author (Alisah-Ozcan), maintained in separate repos. The HEonGPU port may need to carry patches for them, or upstream HIP support PRs to those repos.

### External dependencies

- **RMM (RAPIDS Memory Manager):** Fetched via FetchContent from rapidsai/rmm branch-25.08. AMD has hipMM (RAPIDS RMM port for HIP). Options: (a) use AMD's hipMM, (b) use RMM's nascent HIP backend (if any), (c) abstract the allocator interface.
- **NTL (Number Theory Library):** CPU-only, no porting needed
- **GMP:** CPU-only, no porting needed
- **OpenSSL, ZLIB:** CPU-only, no porting needed

## Risk list

1. **Warp size (wave64 vs wave32):** The `warp_reduce()` function and warp-id/lane calculations use `warpSize` at runtime, which is correct, but also use hardcoded `>> 5` which assumes 32. On gfx90a (wave64), `warpSize=64` but `>> 5` gives wrong warp-id. The 32-bit mask `0xFFFFFFFF` in `__shfl_down_sync` is wrong for 64-lane wavefronts -- HIP requires a 64-bit mask. Fixes:
   - Replace `>> 5` with `/ warpSize` or a device-compile-time constant (`kWarpSize`)
   - Use `0xFFFFFFFFFFFFFFFFULL` mask on HIP
   - Verify shared memory sizing (`smem = (THREADS / 32 + 1) * sizeof(uint32_t)`) uses warpSize

2. **Thirdparty warp assumptions:** GPU-NTT, GPU-FFT, RNGonGPU have pervasive hardcoded 32-thread assumptions. These may need to stay at width-32 tiling (algorithmically safe if they use width-32 shuffles which are wave-agnostic) or require per-arch branching.

3. **RMM dependency:** RMM is a CUDA-native memory pool. Need to evaluate:
   - Does RMM branch-25.08 have any HIP backend?
   - If not, swap for hipMM or stub out with basic hipMalloc

4. **hipRAND device API coverage:** Verify `hiprand_init`, `hiprand`, `hiprandState_t` are 1:1 drop-ins for `curand_init`, `curand`, `curandState_t`.

5. **Rule-of-five on CUDA handles:** Not observed -- the project appears to use RAII via RMM/devicevector, but validate stream/event handle lifetimes.

6. **`CUDA_SEPARABLE_COMPILATION ON`:** Device-linking with HIP is supported but needs `CUDA_SEPARABLE_COMPILATION` -> `HIP_SEPARABLE_COMPILATION` or equivalent CMake handling.

## File-by-file change list

### New files
- `src/include/heongpu/cuda_to_hip.h` -- compat header with CUDA->HIP symbol aliases

### Modified files

**CMakeLists.txt (root):**
- Add `option(USE_HIP "Build with HIP for AMD GPUs" OFF)`
- Gate `enable_language(CUDA)` vs `enable_language(HIP)`
- When `USE_HIP`: skip `find_package(CUDAToolkit)`, `find_package(Thrust)` (use rocThrust)
- Set `CMAKE_HIP_ARCHITECTURES` default (gfx90a when unset)

**src/CMakeLists.txt:**
- Mark all `.cu` sources `LANGUAGE HIP` when `USE_HIP`
- Replace `CUDA::curand` with `hip::host` + `hiprand`
- Handle `CUDA_SEPARABLE_COMPILATION` -> HIP equivalent

**src/include/heongpu/util/util.cuh:**
- Include `cuda_to_hip.h`
- Fix `warp_reduce()`: 64-bit mask on HIP, verify warpSize loop

**src/lib/kernel/{decryption,encryption,keygeneration}.cu:**
- Fix `wid = idx >> 5` -> `wid = idx / kWarpSize` or use `warpSize`
- Fix `n_warps = (blockDim.x + warpSize - 1) >> 5` -> `/ warpSize`

**src/lib/host/tfhe/{encryptor,keygenerator,decryptor}.cu:**
- Fix shared memory sizing `THREADS / 32` -> `THREADS / warpSize` equivalent

**src/include/heongpu/kernel/*.cuh:**
- Replace `#include "cuda_runtime.h"` -> `#include "cuda_to_hip.h"`
- Replace `#include <curand_kernel.h>` -> via compat header

**thirdparty/ (GPU-NTT, GPU-FFT, RNGonGPU):**
- Either patch locally or upstream HIP support PRs
- The `& 31`, `>> 5`, `<< 5` patterns need analysis -- many may be width-32 logical tile ops that are wave-agnostic

## Build commands

Configure (gfx90a):
```bash
cmake -B build \
    -DUSE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx90a \
    -DHEonGPU_BUILD_TESTS=ON \
    -DHEonGPU_BUILD_EXAMPLES=ON \
    -DCMAKE_BUILD_TYPE=Release
```

Build:
```bash
cmake --build build -j$(nproc)
```

## Test plan

### GPU tests (validation gate)
15 gtest executables covering BFV, CKKS, TFHE:
- `bfv_{addition,encoding,encryption,multiplication,relinearization,rotation_method_1,rotation_method_2}_testcases`
- `ckks_{addition,encoding,encryption,multiplication,relinearization,rotation_method_1,rotation_method_2}_testcases`
- `tfhe_gate_boot_testcases`

Run:
```bash
cd build
ctest --output-on-failure
```

Each test exercises GPU kernels (key generation, encryption, FHE operations) and verifies correctness against expected decrypted outputs.

### Determinism check
FHE operations with fixed seeds should be deterministic. Run each test twice and confirm identical output.

### Non-GPU tests
None -- this is a GPU-only library. All tests require GPU execution.

## Open questions

1. **RMM on HIP:** Does RMM branch-25.08 work with hipcc, or do we need hipMM / a custom allocator? Investigate at build time.

2. **Thirdparty submodule strategy:** Should we carry local patches for GPU-NTT/GPU-FFT/RNGonGPU, or open upstream PRs to those repos (same author)? The latter is cleaner if responsive.

3. **Width-32 tile correctness on wave64:** The NTT kernels use `<< 5` for 32-element tile indexing. If these are pure index arithmetic (no warp shuffles across the tile boundary), they work unchanged. Need to trace the kernel logic to confirm.

4. **TFHE curandState per-thread state:** The TFHE encryptor allocates 512*32 curandState entries and initializes them per-thread. Verify hiprand_kernel has the same state size and initialization semantics.
