# CuMesh ROCm Port Plan

## Project

- **Name**: CuMesh
- **Upstream**: https://github.com/JeffreyXiang/CuMesh
- **Default branch**: main
- **Description**: GPU-accelerated 3D geometry processing library for PyTorch (mesh cleaning, decimation, remeshing, UV unwrapping)

## Existing AMD support

**Status**: IMPROVABLE -- the upstream setup.py already has partial ROCm/HIP infrastructure, but the CUDA code itself does not compile on HIP.

**Upstream code**: setup.py imports `IS_HIP_EXTENSION` from torch and has a `BUILD_TARGET="rocm"` path. It passes `--offload-arch` to hipcc when HIP is detected. However, the actual `.cu` sources use `::cuda::std::tuple` (CCCL) and device-only `Vec3f`/`QEM` constructors that fail hipCUB's `DeviceSegmentedReduce` template instantiation on ROCm.

**Open PRs**:
- PR #30 (ZJLi2013, ROCm 6.4): 4 files changed, replaces `cuda::std::plus()` with `cub::Sum()`, adds `rocprim::tuple` for `int3_decomposer`, adds `__host__` qualifier to `Vec3f`/`QEM` methods. Tested on MI300X (gfx942), import + basic ops confirmed.
- PR #31 (andyluo7, ROCm 7.0.2): 30 files changed, broader `#ifdef __HIP_PLATFORM_AMD__` guards, `cudaMalloc -> hipMalloc`, `cub -> hipcub` explicit mapping, updates cubvh submodule. Tested as part of TRELLIS.2 enablement.

**Community fork**: ATLAS-0321/CuMesh-rocm (rocm-port branch, pushed 2026-05-02). Not AMD-official, one-off personal fork. Commits suggest ROCm kernel ports + cubvh-rocm submodule. Non-authoritative -- reference only.

**Issue #18**: User build error on AMD RX 9060 XT -- `::cuda::std::tuple` and device-only constructors.

**Decision**: Port from scratch following best practices; use PR #30 as a HINT (its fixes are minimal and clean) but validate independently. PR #30 is small (4 files) and addresses the exact compile failures. PR #31 is larger and duplicates symbol mapping that torch hipify handles automatically. The minimal PR #30 approach is preferred.

**Merge policy**: Upstream accepts PRs (not a reference-only repo). The goal is a clean upstream PR.

## Build classification

**Classification**: PyTorch extension (Strategy B)

**Evidence**:
- `setup.py` line 2: `from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION`
- Uses `CUDAExtension` for three modules: `cumesh._C`, `cumesh._cubvh`, `cumesh._cumesh_xatlas`
- Build command: `pip install . --no-build-isolation`

## Port strategy

**Strategy B: PyTorch extension**

Torch hipifies extension sources at build time. Do not add a compat header for basic runtime symbols; let torch's hipify translate `cudaMalloc`/`cudaStream_t`/etc. Fix only what hipify cannot:

1. `::cuda::std::tuple` in `int3_decomposer` -- CCCL header not available on HIP; use `rocprim::tuple` under `#ifdef __HIP_PLATFORM_AMD__`
2. `::cuda::std::plus()` -- not available on HIP; use portable `cub::Sum()` unconditionally (works on CUDA too)
3. `Vec3f`/`QEM` device-only constructors -- hipCUB's `DeviceSegmentedReduce` template requires host-callable default constructors; add `__host__` qualifier
4. cubvh submodule -- needs to be fetched and may need ATen/cuda -> ATen/hip header mappings; evaluate after main module builds
5. setup.py nvcc-only flags -- `--extended-lambda`, `-U__CUDA_NO_HALF_*` are NVIDIA-only; gate behind `IS_HIP` check

## CUDA surface inventory

### Kernels (`__global__`)

| File | Count | Notes |
|------|-------|-------|
| src/shared.cu | 2 | hook_edges, compress_components |
| src/geometry.cu | 3 | face_areas, face_normals, vertex_normals |
| src/connectivity.cu | ~15 | edge extraction, boundary, components |
| src/clean_up.cu | ~10 | face sorting, duplicate removal, hole filling |
| src/simplify.cu | ~5 | QEM simplification |
| src/atlas.cu | ~15 | chart clustering, UV unwrap |
| src/remesh/*.cu | ~10 | dual contouring, svox2vert |
| src/hash/hash.cu | ~2 | spatial hashing |
| third_party/cubvh | TBD | BVH construction |

### Device functions

- `Vec3f`, `QEM` structs with `__device__ __forceinline__` methods (dtypes.cuh)
- Various `__device__` helpers for packing/unpacking 64-bit keys

### Warp intrinsics

- **None found.** No `__shfl*`, `__ballot`, `__activemask`, or hardcoded warpSize. The `32` literals are bit-shift amounts for 64-bit key packing, not warp-size-related.

### Textures/surfaces

- **None.** No texture objects or cudaArray usage in main code.

### CUB usage (heavy)

| API | Files |
|-----|-------|
| `cub::DeviceRadixSort::SortPairs` | shared.h, connectivity.cu, atlas.cu |
| `cub::DeviceRadixSort::SortKeys` | connectivity.cu, clean_up.cu |
| `cub::DeviceScan::ExclusiveSum` | shared.h, connectivity.cu, atlas.cu, remesh |
| `cub::DeviceSelect::Flagged/If` | shared.h, connectivity.cu, clean_up.cu |
| `cub::DeviceReduce::Max/Sum/ReduceByKey` | shared.h, connectivity.cu, atlas.cu |
| `cub::DeviceSegmentedReduce::Sum/Reduce/Max` | atlas.cu, clean_up.cu |
| `cub::DeviceRunLengthEncode::Encode` | connectivity.cu |

All CUB device algorithms are auto-converted to hipCUB by torch's hipify.

### CCCL usage

| Symbol | File | Notes |
|--------|------|-------|
| `::cuda::std::tuple` | clean_up.cu:242 | `int3_decomposer` for RadixSort |
| `::cuda::std::plus` | atlas.cu:327 | ReduceByKey (already version-guarded) |

### Other CUDA APIs

- `cudaMalloc`/`cudaFree`/`cudaMemcpy*` -- auto-converted
- `cudaStream_t` via `at::cuda::getCurrentCUDAStream()` -- works with HIP torch
- `c10/cuda/CUDAStream.h` -- torch provides the HIP equivalent

### Thrust

- **None found.** No thrust usage in main code.

### cuBLAS/cuFFT/cuRAND/cuSPARSE

- **None.** Pure compute kernels.

### Pinned/managed memory

- **None.** Uses `cudaMalloc` device memory only.

### Streams/events

- Uses `cudaStream_t` via PyTorch's stream management. No explicit event usage.

## Risk list

1. **`::cuda::std::tuple` / `rocprim::tuple` (MEDIUM)**: The `int3_decomposer` functor for `DeviceRadixSort::SortPairs` with a decomposer uses `cuda::std::tuple` from CCCL, unavailable on HIP. Fix: `#ifdef __HIP_PLATFORM_AMD__` with `rocprim::tuple`.

2. **`Vec3f`/`QEM` device-only constructors (MEDIUM)**: hipCUB's `DeviceSegmentedReduce` template instantiation requires host-callable constructors for the value type. Add `__host__` qualifier to all `Vec3f` and `QEM` methods in `dtypes.cuh`.

3. **cubvh submodule (LOW-MEDIUM)**: The `third_party/cubvh` submodule was empty in the shallow clone. It needs:
   - Recursive clone to fetch
   - Potential ATen/cuda header mappings (PR #31 mentions this)
   - Eigen (included in cubvh/third_party/eigen)

4. **NVCC-only flags in setup.py (LOW)**: `--extended-lambda`, `-U__CUDA_NO_HALF_*` cause errors with hipcc. Already partially handled by upstream's IS_HIP check; verify coverage.

5. **No warp-size issues**: No warp intrinsics or hardcoded warpSize found. The port should work on both wave64 (gfx90a) and wave32 (gfx1100) without wave-size fixes.

6. **xatlas (CPU-only, no risk)**: The xatlas module compiles as CPU-only C++, unaffected by HIP porting.

## File-by-file change list

### src/dtypes.cuh
- Add `__host__` qualifier to all `Vec3f` and `QEM` method declarations and definitions
- Reason: hipCUB `DeviceSegmentedReduce` requires host-callable constructors

### src/clean_up.cu
- Replace `::cuda::std::tuple` with `rocprim::tuple` under `#ifdef __HIP_PLATFORM_AMD__`
- Add `#include <rocprim/types/tuple.hpp>` for HIP path

### src/atlas.cu
- Simplify `::cuda::std::plus()` to portable `cub::Sum()` unconditionally
- Remove the `CUDART_VERSION >= 12090` conditional

### setup.py
- Gate NVCC-only flags (`--extended-lambda`, `-U__CUDA_NO_HALF_*`) behind `not IS_HIP`
- Verify `GPU_ARCHS` default handling for HIP path

### third_party/cubvh (if needed)
- Ensure submodule is recursively cloned
- May need ATen/cuda -> ATen/hip header guards in cubvh source

## Build commands

### Configure + Build (gfx90a)

```bash
# Clone with submodules
git clone --recursive https://github.com/JeffreyXiang/CuMesh.git
cd CuMesh

# Install with ROCm PyTorch
GPU_ARCHS=gfx90a pip install . --no-build-isolation -v
```

### For gfx1100 (follower)

```bash
GPU_ARCHS=gfx1100 pip install . --no-build-isolation -v
```

### For multi-arch build

```bash
GPU_ARCHS="gfx90a;gfx1100" pip install . --no-build-isolation -v
```

## Test plan

### GPU tests (validation)

The project has no formal test suite. Use the example scripts in `examples/` directory:

```bash
cd examples

# Test mesh simplification
python simplify.py

# Test remeshing
python remesh.py

# Test UV unwrapping
python uv_unwrap.py

# Test hole filling
python fill_holes.py

# Test duplicate face removal
python remove_duplicate_faces.py

# Test face orientation unification
python unify_orientations.py
```

Expected: Each script should run without errors and produce output `.ply` files.

### Additional functional tests

```python
import torch
import cumesh

# Basic init/read roundtrip
vertices = torch.randn(100, 3).float().cuda()
faces = torch.randint(0, 100, (200, 3)).int().cuda()

mesh = cumesh.CuMesh()
mesh.init(vertices, faces)
v, f = mesh.read()
assert v.shape == (100, 3)
print("Init/read: PASS")

# Compute normals
mesh.compute_face_normals()
mesh.compute_vertex_normals()
print("Normals: PASS")

# cuBVH test
from cumesh import cuBVH
bvh = cuBVH(vertices, faces)
print("cuBVH: PASS")

# xatlas (CPU) test
from cumesh import Atlas
atlas = Atlas()
atlas.add_mesh(vertices.cpu(), faces.cpu())
atlas.compute_charts()
atlas.pack_charts()
print("Atlas: PASS")
```

### Non-GPU tests

- xatlas module is CPU-only -- should work unchanged
- Python package import test: `python -c "import cumesh"`

## Open questions

1. **cubvh submodule ROCm readiness**: The cubvh submodule (JeffreyXiang/cubvh, trellis.2 branch) may need its own HIP fixes. PR #31 mentions updating it to track a rocm-port branch. Assess after main module builds.

2. **Upstream PR strategy**: Two PRs (#30, #31) are already open. Options:
   - Comment on PR #30 offering validation results
   - Open a new cleaner PR if the existing ones stall
   - Coordinate with existing contributors

3. **TRELLIS.2 dependency**: CuMesh is used by TRELLIS.2. Porting CuMesh enables TRELLIS.2 ROCm support. This increases the value of the port.
