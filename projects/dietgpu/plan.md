# dietgpu -- ROCm/HIP port plan (lead: linux-gfx90a)

## Project
- Name: dietgpu
- Upstream: https://github.com/facebookresearch/dietgpu (facebookresearch/dietgpu)
- Default branch: main
- Base sha analyzed: a4d70a14066d2c3e5fe1849b3723e4cd423eee7e
- What it is: GPU lossless/lossy compression for numerical data. Two layers: a generalized byte-oriented range-ANS (rANS) entropy codec (`dietgpu/ans/`), and a floating-point split codec built on it (`dietgpu/float/`) for fp16/bf16/fp32. C++ raw-pointer API plus an optional PyTorch tensor binding.

## Existing AMD support
None. No HIP/ROCm/gfx/amdgcn references anywhere in the tree (the only "hip" substring matches are inside the word "ownership"). The Dockerfile bases on `nvidia/cuda:11.3.1`. This is a fresh CUDA-to-HIP port; AMD value is clear (no OpenCL/Vulkan path either).
Decision: PORT (mechanical, correctness-first). No CUTLASS/CuTe, no wgmma/MMA, no warp-specialized GEMM -- so no AMD-native rewrite is warranted. The hot kernels are bandwidth-bound entropy/byte-split kernels; a correct HIP translation is the right deliverable. A later wave64-tuning pass (occupancy, vectorization) is optional and out of scope for the lead bringup.

## Build classification: pure CMake (Strategy A) for the testable core
Top-level `CMakeLists.txt`: `project(dietgpu LANGUAGES CUDA CXX)`, `find_package(CUDA REQUIRED)`, `enable_testing()` via googletest. Evidence that the GPU-validatable core is NOT torch-coupled:
- `dietgpu/utils/CMakeLists.txt`, `dietgpu/ans/CMakeLists.txt`, `dietgpu/float/CMakeLists.txt` build `dietgpu_utils`, `gpu_ans`, `gpu_float_compress` as plain CUDA SHARED libs (only `glog` + `${CUDA_LIBRARIES}`); they do NOT find or link Torch.
- The four gtest executables -- `ans_test`, `ans_statistics_test`, `batch_prefix_sum_test` (ans/CMakeLists.txt), `float_test` (float/CMakeLists.txt) -- link only `gpu_ans`/`gpu_float_compress`/`dietgpu_utils` + `gtest_main`. No Torch. These are the real GPU correctness gate and need no PyTorch.
- Only the top-level `dietgpu/CMakeLists.txt` target `dietgpu` (DietGpu.cpp, the Python/torch binding) does `find_package(Torch REQUIRED)` (`#include <ATen/...>`, `torch/types.h`). That is a thin optional host wrapper over the same device libs.

So this is "torch-coupled" only at the optional Python-binding layer; the codec libraries and their tests are standalone CUDA/CMake. Strategy A (one compat header + `enable_language(HIP)` + `.cu` marked `LANGUAGE HIP`) ports the whole core and validates fully via gtest WITHOUT a ROCm PyTorch. The torch binding can be built later against a ROCm torch (Strategy-B-style hipify is unnecessary because the binding host file `DietGpu.cpp` has no `.cu`; it just calls the already-HIP device libs), but it is NOT part of the lead GPU gate.

ext_type recorded as `cmake`.

## Port strategy (Strategy A)
1. Add one compat header `dietgpu/utils/cuda_to_hip.h`: on `USE_HIP`/`__HIP_PLATFORM_AMD__` include `<hip/hip_runtime.h>` (and `<cstring>`/`<cstdlib>` BEFORE it per gpuRIR lesson) and alias the cuda* runtime symbols this project actually uses (full list in inventory). On NVIDIA it is a no-op include of `<cuda_runtime.h>`. Force-include it on every HIP TU via `CMAKE_HIP_FLAGS -include .../cuda_to_hip.h` so it precedes each file's own `#include <cuda.h>`/`<cuda_runtime.h>` (those toolkit headers are absent on ROCm; either shim them by name in a HIP-only include dir per MPPI-Generic, or have the compat header satisfy them -- prefer name-shims `cuda.h`/`cuda_runtime.h` in a `hip_compat/` dir added to the include path only under USE_HIP, leaving the CUDA build byte-for-byte unchanged).
2. CMake: add `option(USE_HIP)`; when ON, `enable_language(HIP)`, default `CMAKE_HIP_ARCHITECTURES` to gfx90a ONLY when unset (never a literal -- followers pass `-DCMAKE_HIP_ARCHITECTURES=gfx1100` with no source edit), set the `.cu` sources to `LANGUAGE HIP`, set `HIP_ARCHITECTURES` from `${CMAKE_HIP_ARCHITECTURES}`. Replace `find_package(CUDA)` use with a USE_HIP branch; the `--generate-line-info` nvcc flag must be guarded to the CUDA language only.
3. cub -> hipCUB: the sources `#include <cub/cub.cuh>` and use `cub::BlockScan/BlockReduce/BlockRadixSort`. Alias in the compat header (`#define cub hipcub` + include `<hipcub/hipcub.hpp>`), the cudaKDTree-proven pattern. rocThrust/hipCUB require C++17; the project sets C++14 -- bump the HIP build to `-std=c++17` (GPUMD lesson; rocPRIM hard-errors on <C++17).
4. Replace the PTX inline asm and warp intrinsics (see risk list) behind the existing helper names in `PtxUtils.cuh` and `DeviceDefs.cuh`, guarded by `#if defined(__HIP_PLATFORM_AMD__)`, so call sites are untouched.

## CUDA surface inventory
Kernels (all warp/block cooperative, no textures, no surfaces, no managed/pinned mem, no streams-with-events in kernels):
- ANS encode: `ansEncodeBatchFull/Partial`, `ansEncodeCoalesceBatch` (GpuANSEncode.cuh); device cores `encodeOneWarp`, `encodeOnePartialWarp`, `ansEncodeWarpBlock`.
- ANS decode: `ansDecodeKernel` (`__launch_bounds__(128)`), `ansDecodeTable` (GpuANSDecode.cuh); cores `decodeOneWarp`, `decodeOnePartialWarp`, `ansDecodeWarpBlock`.
- ANS statistics: histogram + weight normalization (GpuANSStatistics.cuh) using `cub::BlockRadixSort` (full 32-bit key) and `cub::BlockScan`, plus `blockSum` built on `warpReduceAllSum`.
- Checksum: `cub::BlockReduce` (GpuChecksum.cuh).
- Batched exclusive prefix sum: `cub::BlockScan` two-pass (BatchPrefixSum.cuh).
- Float split/join: `splitFloat`/join kernels (GpuFloatCompress.cuh/GpuFloatDecompress.cuh) -- per-warp histograms `histogram[kWarps][kNumSymbols+1]`, byte-plane split using `rotateLeft/rotateRight` (PTX `shf`), then feed the ANS codec.

Warp intrinsics / warp-width-pinned constructs:
- `kWarpSize = 32` hardcoded in `utils/DeviceDefs.cuh` -- the single source of the warp constant, threaded through ~40 sites.
- `__ballot_sync(0xffffffff, ...)` + `__popc(vote & getLaneMaskLt()/Ge())` -- the rANS read/write lane-prefix logic in encode/decode (32-bit masks).
- `__shfl_sync(0xffffffff, ..., 0)` warp-uniform broadcast of block id (encode/decode kernels).
- `__shfl_xor_sync(0xffffffff, val, mask, kWarpSize)` tree reductions in `warpReduceAll{Min,Max,Sum}`; an sm_80 `__reduce_*_sync` fast path guarded by `__CUDA_ARCH__ >= 800`.
- PTX inline asm in `utils/PtxUtils.cuh`: `bfe.u32/u64` (getBitfield), `bfi.b32` (setBitfield), `shf.l/r.clamp.b32` (rotateLeft/Right), `mov %%laneid` (getLaneId), `mov %%lanemask_lt/le/gt/ge` (getLaneMask*). Also one `shf.r.clamp.b32` directly in `float/GpuFloatUtils.cuh` bf16 join.
- `__umulhi` (encode division-by-mul), `__clz` -- both natively in HIP.

Libraries: cub/CUB only -> hipCUB/rocPRIM. No cuBLAS/cuFFT/cuRAND/cuSPARSE/Thrust device calls (a stray unused `CURAND_VERIFY` macro in DeviceUtils.h references CURAND but no curand symbol is called; leave or drop it). No cooperative groups. No CUTLASS/CuTe (confirmed).

Runtime API used (host, StackDeviceMemory.cpp / DeviceUtils.cpp -- all 1:1 hip*): cudaMalloc/Free, cudaMemcpy(Async), cudaGetDevice/SetDevice/GetDeviceCount, cudaGetDeviceProperties (cudaDeviceProp), cudaPointerGetAttributes (cudaPointerAttributes, cudaMemoryTypeDevice/Host), cudaStreamCreateWithFlags/Destroy/WaitEvent (cudaStreamNonBlocking/Default), cudaEventCreateWithFlags/Record/Destroy/Synchronize/ElapsedTime (cudaEventDisableTiming/Default), cudaDeviceSynchronize, cudaGetLastError, cudaGetErrorName/String, cudaProfilerStart/Stop, cudaOccupancyMaxActiveBlocksPerMultiprocessor. `RAII CudaStream`/`CudaEvent` already move-only with explicit handles -- watch rule-of-five but they look fine.

## Risk list

### R1 (CENTRAL) -- wave64 vs the warp-pinned rANS state layout
This is the dominant fault locus. The rANS coder runs one warp per block with ONE independent rANS state PER LANE. `struct ANSWarpState { ANSStateT warpState[kWarpSize]; }` (GpuANSUtils.cuh) is written into the compressed archive header (`out->warpState[laneId] = state` in encode; read at `headerIn->getWarpStates()[block].warpState[laneId]` in decode). The input is striped across lanes by `kWarpSize` (`in[inOffset + j*kWarpSize]`, `inOffset = laneId`), and the number of interleaved states defines the entire bitstream geometry.
Implications:
- Setting `kWarpSize = 64` on gfx90a (correct CDNA value) makes the codec use 64 lanes/states. ENCODE and DECODE on the same arch stay self-consistent, so the round-trip gtests (the gate) remain valid -- but the on-disk archive format is now wave-width-dependent: a gfx90a (64-lane) archive is NOT byte-compatible with a CUDA/RDNA (32-lane) archive, and vice-versa. This is acceptable for MOAT correctness (we validate round-trip on one arch) but MUST be called out: dietgpu archives are not portable across warp widths, and gfx1100/gfx1151 (wave32) will produce 32-lane archives. Document in notes.md; do not attempt cross-arch archive interop.
- `ANSWarpState` is `static_assert`-free on size but its array length feeds `getCompressedOverhead` and header offsets (`getWarpStates`, `getBlockWords`). With kWarpSize=64 these recompute correctly as long as kWarpSize is a true compile-time constant -- KEEP it `constexpr`, do not switch to a runtime `warpSize`. Use the PORTING_GUIDE per-arch constant form: `#if __HIP_PLATFORM_AMD__: __GFX9__ ? 64 : 32; else 32`. Verify the `static_assert(sizeof(ANSCoalescedHeader)==32)` and uint4-divisibility asserts still hold (they are independent of kWarpSize; the warp-state array is variable-length after the header).

### R2 -- 32-bit ballot masks on wave64
`__ballot_sync(0xffffffff, pred)` returns a 32-bit value on CUDA but the active mask on gfx90a is 64 lanes. On HIP `__ballot` returns `unsigned long long`. The literal `0xffffffff` membermask and `__popc(vote & getLaneMaskLt())` will silently drop lanes 32-63. Fix: on HIP use 64-bit ballot (`unsigned long long vote = __ballot(pred)` -- HIP `__ballot` takes no mask), `__popcll`, and 64-bit lane masks. This is the rANS prefix-count that decides how many words each lane reads/writes -- a wrong mask corrupts the stream (not just perf). Route through helpers so encode/decode call sites are unchanged.

### R3 -- lane-mask PTX intrinsics (getLaneMaskLt/Le/Gt/Ge, getLaneId)
`mov %%lanemask_*` and `mov %%laneid` PTX do not exist on AMD. Replace with HIP equivalents producing 64-bit masks: `getLaneId()` = `__lane_id()` (or `threadIdx.x & 63` within a 1D warp-aligned block) ; lane masks computed from laneId as `(1ULL<<laneId)-1` (lt), `((1ULL<<laneId)<<1)-1` (le), `~((2ULL<<laneId)-1)` (gt), `~((1ULL<<laneId)-1)` (ge), all `unsigned long long`. Must pair with R2's 64-bit ballot.

### R4 -- bitfield/rotate PTX (bfe, bfi, shf.l/r.clamp.b32)
`getBitfield`/`setBitfield`/`rotateLeft`/`rotateRight` use PTX. Replace on HIP with portable C++ (`(val>>pos)&((1u<<len)-1)`; `__bitinsert_u32`/mask-merge; for shf.l.clamp use a funnel-shift: `(uint32_t)((((uint64_t)hi<<32)|lo) << shift >> 32)` clamped, or HIP `__builtin_amdgcn` / a plain rotate when hi==lo). The `rotateLeft(v,1)`/`rotateRight(v,1)` calls in the float split/join are correctness-critical (they round-trip the float bytes) -- unit-test the replacement against the PTX semantics. The direct `shf.r.clamp.b32` in GpuFloatUtils bf16 `join` likewise.

### R5 -- warpReduce fast path on sm_80 (__reduce_*_sync)
`warpReduceAll{Min,Max,Sum}` take `__reduce_*_sync` under `#if __CUDA_ARCH__ >= 800`. `__CUDA_ARCH__` is undefined in HIP device compilation, so HIP correctly takes the `__shfl_xor_sync` fallback -- but that fallback hardcodes `0xffffffff` and `kWarpSize` width 32 in the mask/width args; on HIP map `__shfl_xor_sync(mask,v,delta,width)` to `__shfl_xor(v,delta,width)` with width=kWarpSize(=64) so the tree covers all 64 lanes. The loop bound `mask = Width/2` already uses kWarpSize, so it scales; just ensure the shfl width and the dropped mask are correct on HIP.

### R6 -- cub block-collective TempStorage reuse race on wave64 (PORTING_GUIDE class)
A 64-thread block is a single wavefront on gfx90a, so `cub::BlockReduce`/`BlockRadixSort`/`BlockScan` lower without a syncing epilogue; back-to-back reuse of the same `TempStorage` union races (masked on CUDA by the 2-warp split). dietgpu uses BlockScan in the prefix-sum two-pass, BlockRadixSort+BlockScan sequentially in statistics (`SortDescending` then `ExclusiveSum`, with `smemPdf` reuse and an existing `__syncthreads`), BlockReduce in checksum. Threads counts are 128/256/512 (>64) so they are multi-wavefront, which lessens but does not remove the risk where the same TempStorage is reused across two collective calls. Audit each reuse and add `__syncthreads()` between reused-TempStorage collectives. Also note BlockRadixSort sorts the FULL 32-bit packed (prob<<16|sym) key (begin_bit 0), so the hipCUB nonzero-begin_bit bug (cudaKDTree) does NOT apply.

### R7 -- hipCUB requires C++17; project is C++14
rocPRIM/hipCUB `#error` on <C++17. Bump the HIP build to `-std=c++17` (CUDA build can stay 14). GPUMD-proven.

### R8 -- cuda.h / cuda_runtime.h toolkit includes
Many .cuh `#include <cuda.h>`; DeviceUtils.h includes `<cuda_runtime.h>`. ROCm ships no such named headers. Provide name-shim headers in a USE_HIP-only include dir (MPPI-Generic pattern) or have the compat header define them, so source include lines are untouched and the CUDA path is unchanged.

### R9 -- `__launch_bounds__(128)` on the decode kernel
On gfx90a 128 threads = 2 wavefronts; launch bounds are fine but occupancy/registers differ. Not a correctness risk; flag only if the decode kernel fails to launch (resources). The decode grid uses `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (hip equivalent 1:1) so it self-sizes.

### R10 -- non-portable archive / determinism check
The round-trip gate must encode and decode on the SAME arch (it does). Add a fixed-seed determinism check (two encode runs bit-identical) per the MPPI-Generic/popsift practice to catch any unsynced wave64 reduction or leftover 32-lane mask. A wrong R2/R3 mask typically shows as a decode mismatch on inputs whose blocks exceed 32 lanes of compressed words.

## File-by-file change list
- `dietgpu/utils/cuda_to_hip.h` (NEW): HIP runtime include + cuda* aliases (runtime API list above) + `cub`->`hipcub` alias; cstring/cstdlib before hip_runtime.
- `hip_compat/cuda.h`, `hip_compat/cuda_runtime.h` (NEW, HIP-only include dir): each `#include "dietgpu/utils/cuda_to_hip.h"`.
- `dietgpu/utils/DeviceDefs.cuh`: `kWarpSize` -> per-arch constexpr (64 on __GFX9__, else 32).
- `dietgpu/utils/PtxUtils.cuh`: USE_HIP branches for getBitfield/setBitfield/rotateLeft/rotateRight/getLaneId/getLaneMask*/warpReduceAll* (64-bit ballot/masks, hip shfl widths).
- `dietgpu/float/GpuFloatUtils.cuh`: replace the direct `shf.r.clamp.b32` asm in bf16 join under USE_HIP.
- `dietgpu/ans/GpuANSEncode.cuh`, `GpuANSDecode.cuh`: switch `__ballot_sync`/`__popc`+lanemask to the 64-bit helpers (only via the PtxUtils helpers if possible; otherwise local USE_HIP guards). Replace `__shfl_sync(...,0)` broadcast with `__shfl(...,0,kWarpSize)`.
- `dietgpu/ans/GpuANSStatistics.cuh`, `GpuChecksum.cuh`, `BatchPrefixSum.cuh`: cub->hipcub via alias; add `__syncthreads()` between reused-TempStorage collectives (R6).
- Top `CMakeLists.txt` + `dietgpu/{,utils,ans,float}/CMakeLists.txt`: `option(USE_HIP)`, `enable_language(HIP)`, configurable `CMAKE_HIP_ARCHITECTURES`, `.cu` `LANGUAGE HIP`, guard `--generate-line-info` to CUDA, `-std=c++17` for HIP, `-include` the compat header, USE_HIP-only include dir for hip_compat. Leave the torch-binding `dietgpu` target gated so the core builds without Torch.
- The Python/torch binding (`DietGpu.cpp`) needs no device changes; build it later against a ROCm torch -- NOT part of the lead gate.

## Build commands (gfx90a)
Core (no Torch needed -- this is the gate):
```
cd projects/dietgpu/src
git submodule update --init --recursive   # glog, googletest (needed; --depth=1 clone omitted them)
cmake -S . -B build-hip -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-hip --target ans_test ans_statistics_test batch_prefix_sum_test float_test gpu_ans gpu_float_compress dietgpu_utils
```
Followers reuse the same commit: `-DCMAKE_HIP_ARCHITECTURES=gfx1100` (wave32) with no source edit.
A CPU-only docker compile check (`rocm/dev-ubuntu-24.04:7.2.4-complete`) is a manual smoketest only, never the gate.

## Test plan (real GPU gate)
GPU-validatable gtest executables (round-trip compress->decompress equality on generated data is the core correctness check):
- `float_test`: FloatTest.Batch, LargeBatch, BatchSize1 -- fp16/bf16/fp32 split-codec round trip.
- `ans_test`: ANSTest.ZeroSized, BatchPointer, BatchPointerLarge, BatchStride -- rANS round trip across batch shapes/strides (exercises full + partial warp paths, the R1/R2/R3 locus).
- `ans_statistics_test`: Histogram, Normalization{,_NonZero,_EqualWeight} -- exercises BlockRadixSort/BlockScan/blockSum (R5/R6).
- `batch_prefix_sum_test`: BatchPrefixSum.OneLevel, TwoLevel -- BlockScan (R6).
Run: `cd build-hip && ctest --output-on-failure` (gtest_discover_tests registers all). Gate = all of the above PASS on gfx90a.
Add a fixed-seed determinism probe (encode twice, compare bytes) for the wave64 mask/reduction classes (R10).
Non-GPU regression set: none beyond these (no separate host-only unit suite). The Python tests (`ans_test.py`, `float_test.py`, `benchmark.py`) require the torch binding + a ROCm PyTorch and are a stretch goal, NOT the lead gate.

## Disposition
PORT, lead linux-gfx90a. Mechanical Strategy-A HIP port; central work is the wave64 rANS warp-layout correctness (R1-R5) plus cub->hipCUB (R6/R7). No skip (no existing AMD support), no AMD-native rewrite (no CUTLASS/perf-tuned MMA). Advancing linux-gfx90a to `planned`.

## Open questions
- Torch binding: build it in this port or defer? Recommended: defer the `dietgpu` (Python) target to a follow-up; the GPU gate is fully covered by the gtest core, which needs no ROCm PyTorch. Revisit if upstream-PR scope wants the .so.
- Cross-warp-width archive portability is intentionally dropped (wave64 archives differ from wave32/CUDA). Confirm this is acceptable for the eventual upstream PR framing (it is a documented consequence of wave64, not a bug).
