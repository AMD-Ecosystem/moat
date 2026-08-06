# CudaSift port plan (Strategy A, gfx90a lead)

## Project
- Upstream: Celebrandil/CudaSift (branch `AdaLovelace`, sm_89-tuned), https://github.com/Celebrandil/CudaSift
- A standalone CUDA implementation of SIFT feature extraction + matching. Depends on OpenCV for image IO.

## Existing AMD support
None in-repo (no hip/rocm/__GFX refs). Closely-related prior art: `AMD-Ecosystem/colmap` branch `rocm-sift-gpu` ports a SIFT-GPU that is plausibly descended from this code; reuse its AMD-fault fixes (texture rule-of-five, OOB index clamps, 256-byte texture pitch). Decision: port (genuine target, no HIP path exists).

## Build classification
Pure CMake CUDA project -> Strategy A. Evidence: `CMakeLists.txt` `project(CudaSift LANGUAGES CXX CUDA)`, `set(CMAKE_CUDA_ARCHITECTURES 89)`, no torch/`find_package(Torch)`. Note: several `.cpp` (mainSift, examples, tests) are compiled as CUDA via `set_source_files_properties(... LANGUAGE CUDA)`.

## Port strategy: Strategy A
1. Add `option(USE_HIP "Build with HIP for AMD GPUs" OFF)`. When ON: `enable_language(HIP)`, set the existing CUDA sources (and the `LANGUAGE CUDA` cpp files) to `LANGUAGE HIP`, and `set_target_properties(... HIP_ARCHITECTURES "gfx90a")`. When OFF: keep the current CUDA path unchanged (BC: NVIDIA build must still work).
2. Add one compat header `cuda_to_hip.h` that, on `__HIP_PLATFORM_AMD__`, includes `<hip/hip_runtime.h>` and aliases the CUDA symbols this repo uses to their hip equivalents; no-op (include `<cuda_runtime.h>`) on NVIDIA. Force-include it via CMake (`-include cuda_to_hip.h` for HIP) so no per-file edits are needed, keeping the diff minimal. Symbols to alias (all have hip equivalents): cudaError(_t)/cudaSuccess/cudaGetErrorString/cudaGetLastError/cudaDeviceSynchronize; cudaGetDeviceCount/cudaGetDeviceProperties/cudaSetDevice/cudaDeviceProp; cudaEvent_t + event calls; cudaStream_t; cudaMalloc/Free/MallocPitch/Memcpy/Memcpy2D; cudaMallocArray/FreeArray/cudaArray/cudaChannelFormatDesc/cudaCreateChannelDesc; cudaTextureObject_t/cudaResourceDesc/cudaTextureDesc/cudaCreateTextureObject/cudaDestroyTextureObject/tex2D; cudaMemcpyToSymbol (constant mem in cudaSiftD.cu); cudaMemcpyKinds.
3. Use hipify's `cuda_to_hip_mappings.py` as the authoritative name source when filling the header.

## CUDA surface inventory
- Textures (primary surface): `cudaTextureObject_t` created in cudaSiftH.cu (`cudaCreateTextureObject`), sampled with `tex2D<float>` in the descriptor/orientation kernels (cudaSiftD.cu, the `ExtractSiftDescriptors*` family) at rotated neighbor offsets `tex2D(texObj, xpos+-cosa, ypos+-sina)`. Backed by `cudaMallocArray` + `cudaMemcpyToArray` (cudaImage.cu).
- Pitched memory: `cudaMallocPitch` + `cudaMemcpy2D` (cudaImage.cu, cudaSiftH.cu).
- Constant memory: `__constant__` device arrays in cudaSiftD.cu (CONST kernels) via `cudaMemcpyToSymbol`.
- Warp shuffles: cudautils.h `ShiftDown/ShiftUp/Shuffle` use `__shfl_*_sync(0xffffffff, var, delta, width=32)`.
- No cuBLAS/cuFFT/cuRAND/cuSPARSE/Thrust/CUB. Clean.

## Risk list
- warpSize 64 vs 32: the `__shfl_*_sync(0xffffffff, ..., width=32)` helpers assume a 32-lane warp. On gfx90a (wave64) HIP largely ignores the 32-bit mask; `width=32` shuffles within 32-lane subgroups. This MAY preserve the intended 32-lane semantics, but it MUST be validated on-device (numeric parity of descriptors). The `idx<32 ... +32` and `idx/32` patterns in cudaSiftD.cu are descriptor-layout constants (128-dim / subregions), not warp size; leave them.
- Texture handle rule-of-five: verify the `cudaCreateTextureObject`/`cudaDestroyTextureObject` lifecycle in cudaSiftH.cu and the `t_data` cudaArray in cudaImage.cu (default-init handles to 0, no double-free) -- AMD faults where CUDA tolerates. This is the colmap CuTexObj class.
- Out-of-bounds texture reads at image edges: `tex2D` fetches at `+-cosa/+-sina` rely on texture addressMode clamping; confirm the `cudaTextureDesc` uses clamp (it should be clamped by the sampler, unlike colmap's linear-memory reads, so likely OK -- verify).
- `cudaMemcpyToArray` is deprecated and may be removed/limited in current HIP; if `hipMemcpyToArray` is unavailable, switch the array upload to `hipMemcpy2DToArray`/`hipMemcpyParam2D`. Likely the single most invasive change.
- 256-byte texture pitch alignment on AMD: cudaArray-backed textures manage their own layout (less risky than colmap's pitched 2D binds), but the `cudaMallocPitch` paths should be checked.
- Arch: replace the hardcoded `CMAKE_CUDA_ARCHITECTURES 89` intent with `HIP_ARCHITECTURES gfx90a` under USE_HIP.

## File-by-file change list
- `CMakeLists.txt`: add `USE_HIP` option + `enable_language(HIP)` + mark sources `LANGUAGE HIP` + `HIP_ARCHITECTURES` + force-include the compat header for HIP. Keep the CUDA path intact.
- `cuda_to_hip.h` (new): the alias header above.
- `cudaImage.cu`: the `cudaMallocArray`/`cudaMemcpyToArray` path may need the modern hip array-copy API (guarded).
- `cudautils.h`: only if validation shows the warp-shuffle width=32 path is wrong on wave64 (use a warp_size abstraction); otherwise leave.
- `cudaSiftH.cu`: texture-object lifecycle hardening if the rule-of-five check flags it.
All other files: unchanged (plain CUDA spelling, aliased by the header).

## Build commands (gfx90a)
Prerequisite: OpenCV dev headers + ROCm. Configure and build out-of-source:
```
cmake -S projects/CudaSift/src -B projects/CudaSift/src/build-hip \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build projects/CudaSift/src/build-hip -j
```

## Test plan (real GPU, gfx90a)
- Build `test_extract`, `test_match`, `test_homography`, `benchmark` (BUILD_TESTS=ON) and `cudasift` on the data/ images.
- GPU gate: run the tests; confirm SIFT keypoint counts and match/homography results are correct (compare against the CUDA-path expectation / the repo's own pass criteria), not just that it runs.
- No non-GPU regression set here (the project is GPU-centric); ensure the NVIDIA/CUDA build still configures with USE_HIP=OFF.

## Open questions
- Is OpenCV (dev) installed on the gfx90a host? If not, this blocks the build (resolve or `set-blocked`).
- Does the installed ROCm provide a working `hipMemcpyToArray` (or do we move to `hipMemcpy2DToArray`)? Determine at first build.
