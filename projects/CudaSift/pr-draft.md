# PR draft (NOT yet opened)

- Base: `Celebrandil/CudaSift:AdaLovelace`
- Head: `AMD-Ecosystem/CudaSift:moat-port` (squashed commit `3e4df2b`)
- Title: `[ROCm] Add HIP build for AMD GPUs`

---

## Body

Adds an opt-in ROCm/HIP build so CudaSift runs on AMD GPUs, while leaving the existing NVIDIA build unchanged. Set `USE_HIP=ON` to compile the same sources through the HIP toolchain and select the target architecture with `CMAKE_HIP_ARCHITECTURES` (defaults to `gfx90a`). `USE_HIP=OFF` keeps the original CUDA path byte-for-byte.

```bash
mkdir build-hip && cd build-hip
cmake .. -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### How it works

The starting point for review is `cuda_to_hip.h`, the single force-included compatibility header. On AMD it includes the HIP runtime and maps the CUDA runtime, memory, event, and texture symbols the project uses to their HIP equivalents; on NVIDIA it is a no-op that pulls in `<cuda_runtime.h>`. The `.cu` and CUDA-language `.cpp` translation units keep their plain CUDA spelling and are compiled as HIP through `set_source_files_properties`, so no per-file include edits are needed. The `hip_compat/` directory holds small shims so the few sources that include `<cuda_runtime.h>`/`<cuda.h>` directly resolve on ROCm, which ships no CUDA headers; it is added to the include path for the HIP build only.

A few shared-code edits generalize call sites without changing CUDA behavior: `cudaMemcpyToSymbolAsync` is given its explicit offset and kind arguments (HIP's overload defaults only the stream), and two array subscripts in `MatchSiftPoints2` are cast to `int` (HIP's `min` overload resolution otherwise yields a non-integer index). `cudaMemcpyToArray`, which is removed from current HIP, is rewritten to `hipMemcpy2DToArray` under a HIP guard in `CudaImage::CopyToTexture`, reproducing the original byte count.

### Texture and wavefront handling

The live texture in `ExtractSift` is a pitched 2D resource bound directly to the image buffer. Every image fed to it has a pitch that is a multiple of 512 bytes from a base that is at least 256-byte aligned, which satisfies the AMD 256-byte texture pitch requirement; the sampler clamp modes keep edge fetches in bounds. The descriptor and orientation reductions use width-32 shuffles, which split a 64-lane wavefront into exact 32-lane subgroups and preserve the CUDA partial-sum layout; the `__any_sync` 32-bit full mask is mapped to the mask-free `__any`, which polls the launched lanes.

### Validation

Built and run on gfx90a (MI250X, CDNA2), gfx1100 (RDNA3), and gfx1201 (RDNA4) with ROCm. The project's test suite passes on each:

```
./test_extract      # 10 passed, 0 failed
./test_match        # 11 passed, 0 failed
./test_homography   #  8 passed, 0 failed
```

Key numbers are consistent across architectures: 1910 keypoints at the default threshold, 3104 with scale-up, a 1910x1910 self-match, and the RANSAC homography recovered within sub-pixel error on the stereo pair.

This work was authored with assistance from Claude.
