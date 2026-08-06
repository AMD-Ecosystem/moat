# Port Plan: visionaray

## Project

- Name: visionaray
- Upstream: https://github.com/szellmann/visionaray
- Default branch: main
- Description: Header-only ray tracing library for CPU/GPU

## Existing AMD support

**Status: EXPERIMENTAL HIP support exists upstream -- validate and improve**

The CHANGELOG (v0.4.2, 2024-06-30) states: "Experimental support for AMD's HIP GPGPU language. Tested with the anari-visionaray ANARI device."

The project ships parallel implementations in `include/visionaray/cuda/` and `include/visionaray/hip/`:
- Both directories have: array.h, cast.h, device_vector.h, fill.h, pitch2d.h, safe_call.h, texture_object.h, util.h
- CUDA-only: managed_allocator.h, managed_vector.h (missing from HIP side)
- Both cuda_texture*.inl and hip_texture*.inl exist for 1D/2D/3D textures

**Key gaps in upstream HIP support:**
1. No `VSNRAY_ENABLE_HIP` CMake option (build system lacks HIP path)
2. No HIP scheduler (cuda_sched.h/inl exists, no hip_sched equivalent)
3. LBVH GPU builder (detail/bvh/lbvh.h) is `__CUDACC__`-only, uses CUB -- HIP builds get no GPU BVH builder
4. `VSNRAY_GPU_MODE` macro checks only `__CUDA_ARCH__`, not `__HIP_DEVICE_COMPILE__`
5. Missing hip/managed_allocator.h and hip/managed_vector.h
6. No HIP CI in GitHub Actions (only CUDA ON/OFF tested)

**Authoritative vs community:** Upstream experimental HIP support maintained by the author -- AUTHORITATIVE but incomplete. The MOAT value is to validate and complete it on gfx90a, not to port from scratch.

**Decision:** Validate and improve the existing upstream HIP support. The headers already exist but the build system and some components (scheduler, LBVH builder, macros) need gaps filled.

## Build classification

**cmake** -- Pure CMake project (Strategy A applies)

Evidence:
- `CMakeLists.txt` at root: `cmake_minimum_required(VERSION 3.22)`, `project(visionaray ...)`
- No setup.py, no torch dependency, no `find_package(Torch)`
- Library is header-only: `add_library(visionaray INTERFACE)` in src/visionaray/CMakeLists.txt
- CUDA enabled via `VSNRAY_ENABLE_CUDA` option and `enable_language(CUDA)`

## Port strategy

**Strategy A variant -- Validate and extend existing HIP path**

This is NOT a from-scratch port. The project already has a dual CUDA/HIP header structure. The work is:

1. **Add CMake HIP support** -- Add `VSNRAY_ENABLE_HIP` option, `enable_language(HIP)`, and HIP library linking mirroring the CUDA path
2. **Fix macros** -- Update `VSNRAY_GPU_MODE` to check `__HIP_DEVICE_COMPILE__` in addition to `__CUDA_ARCH__`
3. **Add HIP scheduler** -- Create hip_sched.h/inl mirroring cuda_sched.h/inl with HIP API calls
4. **Enable HIP LBVH builder** -- Add `__HIPCC__` guards to lbvh.h, swap CUB calls for hipCUB
5. **Add missing HIP headers** -- hip/managed_allocator.h and hip/managed_vector.h mirroring the CUDA versions
6. **GPU test validation** -- Enable/write HIP unit tests

The existing `include/visionaray/hip/*` headers already use native HIP API (hipMalloc, hipMemcpy, hipArray_t, hipTextureObject_t, etc.) -- no translation needed for those.

## CUDA surface inventory

### Kernels / Device Code
| Location | Type | HIP Status |
|----------|------|------------|
| detail/bvh/lbvh.h | GPU BVH builder kernels | CUDA-only (`__CUDACC__`), needs `__HIPCC__` |
| detail/cuda_sched.inl | GPU render scheduler | CUDA-only, needs hip_sched equivalent |
| test/unittests/array.cu | Array unit tests | Uses thrust, should work with rocThrust |
| test/unittests/texture.cu | Texture unit tests | CUDA-only, needs HIP version |
| examples/*.cu | CUDA examples | Optional, lower priority |

### Warp intrinsics
**None found.** `grep -rniE '__shfl|__ballot|warpSize|__activemask|warp'` returned no matches. The library appears to avoid warp-level primitives, reducing wave64/wave32 risk.

### CUDA libraries
- **Thrust** -- Used in random_generator.h, array.inl, examples. rocThrust is a drop-in, same `<thrust/*>` headers.
- **CUB** -- Used in detail/bvh/lbvh.h for `cub::DeviceMergeSort::StableSortKeys`. Needs hipCUB swap.

### Textures / Surfaces
- 1D/2D/3D texture objects via `hipTextureObject_t` / `cudaTextureObject_t`
- Full hip_texture*.inl implementations already exist
- Uses `hipCreateTextureObject`, `hipCreateChannelDesc`, `hipMallocArray`, etc.
- No surface objects, no layered arrays

### Memory
- Device allocation: `hipMalloc`/`hipFree` (already in hip/array.h, hip/device_vector.inl)
- Managed memory: CUDA has managed_allocator.h/managed_vector.h, HIP side missing (minor)
- Streams: cuda_sched uses `cudaStream_t`, hip_sched will need `hipStream_t`

## Risk list

1. **LBVH CUB -> hipCUB** -- `cub::DeviceMergeSort::StableSortKeys` maps to `hipcub::DeviceMergeSort::StableSortKeys`. Low risk, API is 1:1.

2. **VSNRAY_GPU_MODE macro** -- Currently `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0`. Must add `|| (defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__)` for HIP device code detection.

3. **cuda_sched -> hip_sched** -- Scheduler uses `cudaStreamCreate`, `cudaStreamDestroy`, kernel launch `<<<>>>`. The `<<<>>>` syntax works in hipcc. Need to swap cuda* API -> hip* API.

4. **No warp intrinsics** -- No wave64/wave32 risk (none found).

5. **Thrust usage** -- rocThrust provides identical `<thrust/*>` headers on ROCm, no source change needed.

6. **Test infrastructure** -- Unit tests in test/unittests/ use gtest + CUDA. The .cu files should compile as HIP with `LANGUAGE HIP`.

7. **Texture linear filter** -- The hip_texture*.inl files use `hipFilterModeLinear` / `hipFilterModePoint`. Check if any float textures request linear filtering (see PORTING_GUIDE fault class). A quick grep shows `detail::map_filter_mode(filter_mode_)` which maps to `hipFilterModeLinear` or `hipFilterModePoint` -- if a user creates a float texture with linear filter, it may fail on AMD per the known fault class.

## File-by-file change list

### CMake (new HIP option)
- **CMakeLists.txt** (root): Add `VSNRAY_ENABLE_HIP` option
- **src/visionaray/CMakeLists.txt**: Add `if(VSNRAY_ENABLE_HIP) enable_language(HIP) ... endif()` mirroring CUDA path

### Macros
- **include/visionaray/detail/macros.h**: Fix `VSNRAY_GPU_MODE` to detect HIP device compilation

### New HIP headers (mirror CUDA)
- **include/visionaray/hip/managed_allocator.h** (new): Copy from cuda/, replace cuda -> hip
- **include/visionaray/hip/managed_vector.h** (new): Copy from cuda/, replace cuda -> hip

### HIP scheduler (new)
- **include/visionaray/detail/hip_sched.h** (new): Copy cuda_sched.h, adapt for HIP
- **include/visionaray/detail/hip_sched.inl** (new): Copy cuda_sched.inl, swap CUDA -> HIP API

### LBVH builder
- **include/visionaray/detail/bvh/lbvh.h**: Add `#ifdef __HIPCC__` blocks parallel to `__CUDACC__`, include hipCUB, swap CUDA_SAFE_CALL -> HIP_SAFE_CALL

### BVH header
- **include/visionaray/bvh.h**: Already has both `__CUDACC__` and `__HIPCC__` guards -- verify LBVH include path works under HIP

### Tests
- **test/unittests/CMakeLists.txt**: Add HIP path mirroring CUDA, compile .cu as HIP
- **test/unittests/array.cu**: Should compile as HIP (uses thrust, which is drop-in)
- **test/unittests/texture.cu**: May need HIP version or verification

## Build commands

### Configure (gfx90a)
```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DVSNRAY_ENABLE_HIP=ON \
  -DVSNRAY_ENABLE_CUDA=OFF \
  -DVSNRAY_ENABLE_UNITTESTS=ON \
  -DVSNRAY_ENABLE_COMMON=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -S projects/visionaray/src
```

### Build
```bash
cmake --build build -j$(nproc)
```

### Test
```bash
cd build && ctest --output-on-failure
```

## Test plan

### GPU tests (require gfx90a)
1. Build with `VSNRAY_ENABLE_HIP=ON VSNRAY_ENABLE_UNITTESTS=ON`
2. Run `ctest` -- exercises array.cu, texture.cu tests on GPU
3. Build and run examples (cuda_unified_memory, smallpt, etc.) if ported to HIP

### Non-GPU tests (must not regress)
1. CPU-only build with `VSNRAY_ENABLE_HIP=OFF VSNRAY_ENABLE_CUDA=OFF VSNRAY_ENABLE_UNITTESTS=ON`
2. Run `ctest` -- exercises CPU-side math/simd/bvh tests
3. Verify all CPU tests pass unchanged

### Validation approach
Since the upstream HIP support is experimental and "seldom tested":
1. First verify the CURRENT HIP headers compile at all under hipcc
2. Fix any compilation errors
3. Add the CMake HIP path
4. Validate GPU correctness via unit tests
5. Optionally validate via anari-visionaray if time permits

## Open questions

1. **anari-visionaray validation**: The CHANGELOG says HIP was "tested with the anari-visionaray ANARI device." Should we validate against that too, or are visionaray's own unit tests sufficient?

2. **Examples priority**: Should HIP versions of the CUDA examples (raytracinginoneweekend_cuda, opengl_interop) be created, or is library validation sufficient?

3. **Linear filter fault class**: If float textures with linear filtering are used (filterMode=cudaFilterModeLinear on float), they will fail on AMD. Need to audit usage patterns and potentially add software bilinear fallback. May be out of scope for initial validation.

4. **Upstream PR strategy**: The project already has experimental HIP support. Our changes improve and validate it. An upstream PR should be straightforward since we are extending existing work, not adding a new backend.
