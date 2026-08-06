# [ROCm] Add AMD GPU support via HIP

**Base:** `ingowald/cudaKDTree:master`
**Head:** `AMD-Ecosystem/cudaKDTree:moat-port` (`93a838d`)

---

## Title

```
[ROCm] Add AMD GPU support via HIP
```

## Body

This adds an opt-in HIP build of the sample and test programs so cudaKDTree runs on AMD GPUs through ROCm. The CUDA build path and the header-only library are unchanged for existing users: AMD support is enabled with `-DUSE_HIP=ON` and is otherwise inert.

The port introduces one new header, `cukd/cuda_to_hip.h`, that aliases the CUDA runtime spellings this project uses to their HIP equivalents under `USE_HIP` and is a plain include of the CUDA runtime headers otherwise. The `.cu` sources are marked `LANGUAGE HIP` via a small `cukd_gpu_sources()` CMake helper gated on `option(USE_HIP)`. Thrust maps to rocThrust and CUB to hipCUB with no source change. The target architecture is read from `CMAKE_HIP_ARCHITECTURES` (default `gfx90a`), so building for other AMD GPUs needs no further edits.

Host/device guards keyed on `__CUDA_ARCH__` become a backend-agnostic `CUKD_DEVICE_CODE` (which is `__CUDA_ARCH__` on CUDA and `__HIP_DEVICE_COMPILE__` on HIP), so the `__host__ __device__` helpers stay available on both backends. There are also a few clang two-phase-lookup (`this->`) and explicit-specialization attribute fixes that nvcc/MSVC accept implicitly, and `cudaMallocAsync` is kept on the HIP path.

Two ROCm-specific issues surfaced during GPU validation and are worked around behind `USE_HIP` guards, with the CUDA path keeping its native behavior:

1. Integer `atomicMin`/`atomicMax` silently no-op on coarse-grained managed memory on gfx90a/CDNA2, which left the spatial builder's bounding box empty and produced a degenerate tree. These are emulated with an `atomicCAS` loop (signed and unsigned variants; the unsigned one is required by the `(uint32_t)-1` leaf sentinel).
2. hipCUB `DeviceRadixSort::SortKeys` mis-sorts with a nonzero `begin_bit`, which corrupted the leaf primitive ranges. Sorting the full 64-bit key restores correct ordering; the high nodeID bits still dominate and the low primID bits just make the per-leaf order deterministic.

The README gains a short AMD/ROCm support note describing the `USE_HIP` option and `CMAKE_HIP_ARCHITECTURES`.

### Build

```
cmake -S . -B build-hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DBUILD_ALL_TESTS=ON -DCUKD_ENABLE_STATS=ON
cmake --build build-hip -j
ctest --test-dir build-hip
```

### Validation

Built and run on gfx90a (MI250X, CDNA2), gfx1100 (RDNA3), and gfx1201 (RDNA4) under ROCm; the same source also builds and runs on gfx1101 and gfx1151.

The regular k-d tree's in-tree brute-force verifier (`-v`) passes both FCP and kNN across dimensions {2, 3, 4, 8} x build modes {stackBased, stackFree, cct} x k in {4, 8, 50}, for uniform and clustered inputs, with maximum relative error around 1e-7 and deterministic checksums across repeated runs. The spatial tree matches the regular tree and CPU brute force with 0 mismatches.

CTest is 14/15. The single failing case, `cukdTestBuildersSameResult`, is a pre-existing host-vs-device tie-break in widest-split-dim selection (the three device builders agree; only the host builder hash differs, producing a different but equally valid balanced tree). It fails identically on CUDA and is not a port regression.

This work was authored with the assistance of Claude.
