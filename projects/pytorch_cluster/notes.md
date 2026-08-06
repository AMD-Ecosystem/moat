# pytorch_cluster notes

## Disposition: SKIP (already-supported) -- decided 2026-06-11

Planner assessed rusty1s/pytorch_cluster on linux-gfx90a (gfx90a, MI250X, ROCm 7.2.1)
and recorded `triage.py skip --reason already-supported`. No port, no fork, no plan.md.

### Why skip
- Upstream ALREADY MERGED ROCm build support: PR #149 "Enable ROCm build"
  (MERGED 2022-10-17). It added the `torch.version.hip` -> `USE_ROCM` + build-time
  hipify path and changed ONLY setup.py (+23/-2) and csrc/version.cpp (+8); it touched
  no `.cu`. That plumbing is in the tree today:
  - setup.py L18-19 (`WITH_CUDA = CUDA_HOME is not None or torch.version.hip`),
    L80-84 (define `USE_ROCM`, undef `__HIP_NO_HALF_CONVERSIONS__`),
    L123-126 (hipify abs-path `include_package_data` workaround).
  - csrc/version.cpp L9-13, L29-31 (`#ifdef USE_ROCM` -> `hip/hip_version.h`, HIP_VERSION).
- Build classification: torch CUDAExtension (Strategy B). setup.py uses
  `torch.utils.cpp_extension` CUDAExtension/BuildExtension; ROCm torch hipifies the
  `.cu` at build time. There is no standalone CMake CUDA path that matters here.
- CUDA surface (csrc/cuda/*.cu: fps, graclus, grid, knn, nearest, radius, rw) carries
  ZERO fault classes:
  - No warpSize / __shfl / __ballot / __activemask. The only intra-block reductions
    (fps_cuda.cu, nearest_cuda.cu) are __syncthreads() tree reductions over a
    THREADS-sized shared array -- arch-agnostic, correct on wave64 and wave32.
  - No cub / thrust / cuBLAS / cuFFT / cuSPARSE.
  - No textures / surfaces / layered arrays / inline PTX / managed memory.
  - graclus_cuda.cu uses a `__device__ bool` + `cudaMemcpyFromSymbol`
    (-> hipMemcpyFromSymbol, clean).
  - The ONLY library dependency is device-API cuRAND in rw_cuda.cu
    (`curand.h`, `curand_kernel.h`, `curandState_t`, `curand_init`, `curand`,
    `curand_uniform`); torch hipify maps these 1:1 to hipRAND/rocRAND.
  Net: the kernels build and run unmodified through torch hipify -- which is exactly
  why PR #149 needed no `.cu` change. There is no source delta for MOAT to contribute.
- Maintained AMD distribution exists (community, not AMD-official, but actively
  maintained): Looong01/pyg-rocm-build and torch-cluster-rocm on PyPI build the
  UNMODIFIED upstream source against ROCm torch; they track ROCm 7.2.2 / PyTorch 2.11
  (April 2026).
- Upstream is DEPRECATED in favor of pyg-lib>=0.7.0 (latest commit #271, README CAUTION
  banner), and AMD states PyG is NOT on its ROCm roadmap (ROCm/ROCm discussion #3655,
  AMD collaborator: "PyG support is currently not on our roadmap"). An upstream PR would
  be a no-op into a deprecated repo whose maintainer steers users to pyg-lib.

### If revisited later
The actionable PyG-family ROCm work, if any, is in pyg-lib (the successor), not here.
torch-cluster's own ROCm story is closed: merged build glue + clean-hipify kernels +
maintained community wheels.

### Env facts (this host)
- linux-gfx90a, AMD Instinct MI250X (gfx90a, CDNA2, wave64), ROCm 7.2.1.
