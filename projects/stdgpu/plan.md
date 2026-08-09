# stdgpu Port Plan

## Project

- **Name:** stdgpu
- **Upstream:** https://github.com/stotko/stdgpu
- **Default branch:** main
- **Description:** Efficient STL-like Data Structures on the GPU (unordered_map, unordered_set, bitset, vector, deque, queue, stack, mutex, atomic)

## Existing AMD Support

**Status:** EXPERIMENTAL HIP backend exists upstream -- needs validation and improvement.

stdgpu already ships an experimental HIP backend (`STDGPU_BACKEND_HIP`) since PR #121/#143. The upstream docs (README.md line 49, docs/index.md) list HIP as an "experimental" backend alongside CUDA and OpenMP. The HIP backend has a complete directory structure (`src/stdgpu/hip/`, `cmake/hip/`, `tests/stdgpu/hip/`).

**PORTING_GUIDE documented bug (line 324):** The concurrent hashtable deduplication (`unordered_base::try_insert`) is NOT wave64-correct. On a 64-lane wavefront, two threads inserting the SAME key can both report success (the mutex/occupied bitset memory ordering does not serialize same-key lanes the way CUDA SIMT implicitly does). Symptom: insert mask popcount is 1 too high for a batch with one duplicate (1024 vs 1023). Basic Insert/Find/Erase/Clear/IO on distinct keys PASS; the bug manifests only on concurrent duplicate-key insertions.

**Authoritative assessment:** The HIP backend is UPSTREAM-AUTHORITATIVE (maintained by the stdgpu author, Patrick Stotko, not a community fork). MOAT value: validate on real GPU, identify/fix the wave64 dedup bug, contribute fixes upstream.

**Decision:** Validate and improve the existing HIP backend, not a from-scratch port.

## Build Classification

**Type:** Pure CMake project (Strategy A compatible, but HIP backend already exists)

**Evidence:**
- `CMakeLists.txt` line 68: `enable_language(HIP)` when `STDGPU_BACKEND == STDGPU_BACKEND_HIP`
- No `setup.py`, no `torch.utils.cpp_extension`, no PyTorch dependency
- Backend selection via `-DSTDGPU_BACKEND=STDGPU_BACKEND_HIP`
- HIP sources in `src/stdgpu/hip/` compiled directly with `LANGUAGE HIP`

**ext_type:** cmake

## Port Strategy

Since the HIP backend already exists upstream, this is NOT a Strategy A/B port. The strategy is:

1. **Build and run the existing HIP backend** on gfx90a (ROCm 7.2.x)
2. **Validate the test suite** -- identify which tests pass/fail
3. **Root-cause the wave64 dedup bug** in `unordered_base::try_insert`
4. **Fix the bug** (likely requires proper wave-collective serialization or a different locking strategy)
5. **Contribute fix upstream** since the backend is authoritative

## CUDA Surface Inventory

stdgpu's HIP backend is already ported (no CUDA surface to translate). The GPU primitives used:

- **Atomics:** `atomicExch`, `atomicCAS`, `atomicAdd`, `atomicSub`, `atomicAnd`, `atomicOr`, `atomicXor`, `atomicMin`, `atomicMax`, `atomicInc`, `atomicDec` -- all wrapped in `stdgpu::hip::atomic_*` (see `src/stdgpu/hip/impl/atomic_detail.h`)
- **Memory fences:** `__threadfence()` via `atomic_thread_fence()`
- **Thrust/rocThrust:** used for parallel algorithms (`for_each_index`, etc.)
- **No warp intrinsics:** No `__shfl*`, `__ballot`, `__activemask` -- the concurrent containers rely on atomics and mutexes, not warp-collective ops
- **No textures/surfaces**
- **No cuBLAS/cuFFT/cuRAND**

## Risk List

1. **Wave64 dedup bug (HIGH):** The `unordered_base::try_insert` concurrent duplicate-key deduplication fails on wave64. The pattern is:
   - Thread acquires per-bucket mutex via `try_lock()` (atomic exchange on bitset)
   - Thread rechecks `!contains(key)` and `!occupied(bucket_index)`
   - If both pass, thread inserts and sets occupied bit
   
   On wave64, two threads with the SAME key can BOTH acquire locks (for different buckets if keys hash the same, or even for the same bucket if the atomic exchange returns stale values across the wavefront). The re-check of `contains()` races because both threads see it as false before either commits. This is NOT a simple warp-size constant issue -- it's a concurrent-algorithm correctness issue that CUDA's 32-lane wavefront happens to mask (smaller race window).

2. **Test coverage for the bug:** The `insert_double` test inserts the same key TWICE from the HOST (serial), which doesn't trigger the concurrent race. Need a GPU-side concurrent duplicate insertion test.

3. **Build compatibility:** Upstream requires ROCm 7.1+ (per docs). Should work on ROCm 7.2.x.

4. **rocThrust dependency:** The HIP backend uses rocThrust. Already shipped with ROCm.

## File-by-File Change List

No source changes required for build bringup. For the wave64 dedup fix, changes will likely be in:

- `src/stdgpu/impl/unordered_base_detail.cuh` -- `try_insert` logic
- Possibly `src/stdgpu/impl/mutex_detail.cuh` -- if the lock mechanism needs wave64-aware serialization
- A new test case for concurrent duplicate insertion on GPU

## Build Commands

```bash
# Configure for HIP backend, gfx90a
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTDGPU_BACKEND=STDGPU_BACKEND_HIP \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a

# Build
cmake --build build --config Release --parallel $(nproc)

# Run tests
cd build && ctest -V -C Release
```

For multi-arch:
```bash
-DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100"
```

## Test Plan

**Real GPU tests (HIP backend):**
- `teststdgpu` -- the main test binary covering:
  - `atomic.hip` -- atomic operations
  - `bitset.hip` -- bitset operations
  - `deque.hip` -- deque operations
  - `memory.hip` -- memory management
  - `mutex.hip` -- mutex operations (important for the bug)
  - `unordered_map.hip` -- hash map (the buggy container)
  - `unordered_set.hip` -- hash set (also uses unordered_base)
  - `vector.hip` -- vector operations

**Expected pass/fail:**
- Basic operations (single-threaded semantics from host) should PASS
- Concurrent duplicate-key insertion should FAIL or produce incorrect counts on wave64

**Non-GPU tests:**
- CPU-only tests (`algorithm.cpp`, `bit.cpp`, `contract.cpp`, `functional.cpp`, `iterator.cpp`, `limits.cpp`, `numeric.cpp`, `ranges.cpp`) must not regress

**Wave64 dedup bug reproduction:**
1. Create a device array with duplicate keys (e.g., 1024 elements where key 0 appears twice)
2. Call bulk `insert(begin, end)` on an `unordered_set` or `unordered_map`
3. Check `size()` -- on wave64 it may report 1024 instead of 1023
4. Alternative: call `insert_unique_parallel` test with crafted duplicates

## Open Questions

1. **Fix strategy for wave64 dedup:** Options include:
   - Add warp-level coordination before the CAS (e.g., elect one thread per unique key via ballot/match)
   - Use a different lock-free insert algorithm that doesn't rely on the SIMT execution model
   - Document as a known limitation for wave64 (not recommended if Open3D depends on this)

2. **Upstream PR vs fork-only:** Since the HIP backend is upstream-maintained, fixes should go upstream. Will need to coordinate with Patrick Stotko.

3. **Open3D dependency:** Open3D uses stdgpu for its hashmap. If the wave64 bug affects Open3D's use case, prioritize the fix. If Open3D only inserts unique keys, the bug is latent.
