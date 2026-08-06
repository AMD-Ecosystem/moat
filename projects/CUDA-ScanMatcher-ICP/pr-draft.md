# PR draft: CUDA-ScanMatcher-ICP

- Upstream: botforge/CUDA-ScanMatcher-ICP
- Base branch: master
- Head: AMD-Ecosystem:moat-port @ ce20c461b35aaa5781724cc0fd2742f8604c4b89
- Status: DRAFT -- do not open until approved

## Title

[ROCm] Add HIP/ROCm support for AMD GPUs

## Body

This adds AMD GPU support to the CUDA scan-matcher (ICP point-cloud registration) through HIP, so the project builds and runs on ROCm in addition to CUDA.

A single `src/cuda_to_hip.h` compatibility header maps the CUDA runtime and OpenGL-interop symbols the project uses to their HIP equivalents. The same sources compile for NVIDIA (the default) and AMD (configure with `-DUSE_HIP=ON`); the CUDA build path is unchanged. The GL interop is updated to the modern graphics-resource API (`hipGraphicsGLRegisterBuffer` and friends), which HIP provides and which supersedes the old `cudaGL*` entry points.

GLM device-function support is enabled through `src/glm_device.h`, which defines the qualifier macros GLM expects so its math functions get `__host__ __device__` under hipcc. rocThrust is a drop-in for CUDA Thrust; it requires C++17, so the host language standard is raised to match. Include order matters: Thrust headers must precede GLM so rocThrust selects the HIP backend.

### Build

```bash
mkdir build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

`CMAKE_HIP_ARCHITECTURES` selects the target GPU (for example `gfx90a` for CDNA2, `gfx1100` for RDNA3, `gfx1201` for RDNA4); it defaults to `gfx90a` if unset. The README documents the AMD build alongside the existing Windows and Linux CUDA instructions.

### Validation

Built and run on gfx90a (MI250X, CDNA2), gfx1100 (RDNA3), and gfx1201 (RDNA4). The headless ICP run (`VISUALIZE 0`) completes 10 iterations on each, with stable nearest-neighbor timing per step and no errors.

### Notes

This work was authored with assistance from Claude, an AI assistant by Anthropic.

## To open (after approval)

```bash
gh pr create --repo botforge/CUDA-ScanMatcher-ICP --base master \
  --head AMD-Ecosystem:moat-port \
  --title "[ROCm] Add HIP/ROCm support for AMD GPUs" \
  --body-file projects/CUDA-ScanMatcher-ICP/pr-draft-body.md
```

(Extract the body section above into a separate body file at open time to avoid the draft scaffolding, or pass `--body-file` pointing at the prose only.)
