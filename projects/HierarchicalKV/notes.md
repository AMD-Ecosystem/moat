# HierarchicalKV notes

## Intake screen (2026-11-11, windows-gfx1151 host, intake role)

Requested by Jeff Daily on 2026-11-11: HugeCTR (NVIDIA-Merlin/HugeCTR) was ruled a take-up needing a canonical AMD-Ecosystem home, and HierarchicalKV is its hard build dependency with no AMD port. Upstream: https://github.com/NVIDIA-Merlin/HierarchicalKV, default branch `master` (not `main`: scaffold recorded `fork_default_branch: main`; whoever creates the fork must check the mirror branch name before `set-base-branch` or the first sync). Screened at the shallow-clone tip c79175830d5110b0d76868dc465ee0415ba080e4 (2026-10-30). Not archived; pushed 2026-11-11; 210 stars; actively developed (dual-bucket and research benchmarks added in 2026).

### Licence

- `python3 utils/licenses.py check NVIDIA-Merlin/HierarchicalKV`: Apache-2.0, tier 1, cleared to contribute. Evidence: https://github.com/NVIDIA-Merlin/HierarchicalKV/blob/master/LICENSE
- `python3 utils/licenses.py scan-nvidia agent_space/HierarchicalKV-screen` (shallow clone with submodules): no NVIDIA proprietary licence text.
- Every header under `include/` carries the Apache-2.0 notice (`grep -rL Apache include` is empty).
- Submodules: only `tests/googletest` (BSD-3-Clause, Google). There is NO vendored libcu++: `<cuda/atomic>`, `<cuda/std/semaphore>`, `<cuda/barrier>` come from the CUDA toolkit's CCCL.
- `third_party/recsys-examples/legacy-250aaa.patch` is a 27-file patch against NVIDIA/recsys-examples for an optional benchmark (`benchmark/native_tbe`); its files are SPDX Apache-2.0. Not part of the library or tests.
- `baselines/` FetchContent-downloads warpcore, BGHT and cuCollections for comparison benchmarks only (not built by default; out of port scope).

### Duplicate effort

- No AMD/ROCm port found. Searched: `gh search repos HierarchicalKV` (results: upstream, Ascend/HierarchicalKV-ascend, yangzhen92/HierarchicalKV-ascend, xiaobingling93-pixel/Ascend-HierarchicalKV-ascend, HYGON-AI/HierarchicalKV-das); ROCm and AMD-Ecosystem orgs (none); upstream forks list filtered for rocm/amd/hip (only `gamdwk/HierarchicalKV`, a username false positive); upstream PRs and issues searched for "hip OR rocm OR amd" (only false-positive matches #200, #50, #48, all merged, unrelated); upstream branches (none matching); README/docs grep for amd/rocm/hip/gfx (none).
- HYGON-AI/HierarchicalKV-das is a downstream adaptation for Hygon DCU ("hcu") hardware. Per Jeff's ruling it is NOT a base for this port and was not used in this analysis.
- The Ascend repos target Huawei NPUs, not relevant.

### What HugeCTR consumes

- HugeCTR `.gitmodules` pins `third_party/HierarchicalKV` at 28046ba088d9fea6e719afbeb12930b63d7c9fff (`gh api repos/NVIDIA-Merlin/HugeCTR/contents/third_party/HierarchicalKV`). The port should target master; HugeCTR's port will bump its submodule to our fork.
- HugeCTR's top-level CMakeLists.txt adds `third_party/HierarchicalKV/include` to the include path; the code that includes `merlin_hashtable.cuh` is `sparse_operation_kit/kit_src/variable/impl/hkv_variable.{h,cu}` (SOK, the TensorFlow plugin) per GitHub code search. HKV is header-only, so the consumer compiles the kernels itself: the port's headers must be HIP-clean when included from another project's hipcc/clang build.

### Viability and CUDA surface

Header-only C++17/CUDA library, about 22k lines under `include/` (`merlin_hashtable.cuh` 4062 lines plus `merlin/core_kernels/*.cuh`). Genuinely CUDA. No CUDA math libraries; only CUB and Thrust (hipCUB/rocThrust).

- libcu++: `cuda::atomic<T, cuda::thread_scope_device>` (30 uses; on K, S, T, bool, int), `cuda::std::memory_order_*` (about 390 uses), `<cuda/std/semaphore>`, `<cuda/barrier>`, `<cuda/std/functional>` (`cuda::std::plus`), `thrust::cuda::par(_nosync).on(stream)`. ROCm equivalent is libhipcxx (TheRock ships it under math-libs/libhipcxx with `cuda/atomic`, `cuda/barrier`, `cuda/semaphore`, `cuda/std/*`). Precedent: HEonGPU builds against libhipcxx through hipMM (notes on origin/port/HEonGPU). Keys and bucket locks are lock-free CAS on 64-bit `cuda::atomic` with acquire/release ordering; correctness depends on libhipcxx's scoped atomics lowering correctly on AMD (memory-model fault class).
- Cooperative groups: `cg::this_thread_block` (44), `cg::tiled_partition<N>` with N in {4, 8, 16, 32}, `thread_block_tile` methods `.shfl` (242), `.ballot` (78), `.sync` (182), `.any` (14), `cg::reduce(g, x, cg::less<S>())` (10). HIP provides tiled_partition, reduce and less (amd_hip_cooperative_groups_reduce.h).
- Raw warp intrinsics with 32-bit masks: `__ballot_sync(0xffffffff, ...)`, `__any_sync`, `__syncwarp(mask)` in update.cuh, update_score.cuh, update_values.cuh; masks held in `unsigned int`. HIP's `*_sync` builtins static_assert a 64-bit mask, so these do not compile as written and need a wavefront-width mask on wave64 and wave32.
- `__vcmpeq4` (62 uses): CUDA SIMD-in-word byte compare at the core of digest probing. HIP has no SIMD video intrinsics; a portable bitwise shim is needed (CUDA path untouched).
- `__pipeline_memcpy_async` / `__pipeline_commit` / `__pipeline_wait_prior` from `<cuda_pipeline.h>` (about 290 uses across 13 kernel files, plus an `LDGSTS` wrapper in kernel_utils.cuh). HIP has none of the primitives; a synchronous-copy shim (commit and wait as no-ops, plus a barrier where the code relies on wait for visibility) is the likely route.
- `__stcs` / `__ldcs` on `uint4`/`uint2`/integers (HIP only defines them for half types); `__ldg` is fine.
- Inline PTX: one site, `mov.u64 %0,%%globaltimer` in `utils.cuh` `device_nano()` (maps to `wall_clock64()`); plus one each in tests/test_util.cuh and benchmark/benchmark_util.cuh.
- Host API: cudaHostAlloc / cudaMallocHost / cudaHostGetDevicePointer (values can live in pinned host memory read by kernels over the bus), cudaMallocAsync / cudaFreeAsync, cudaMemcpyToSymbol. All have HIP equivalents; host-resident values on AMD need coherent (fine-grained) pinned memory.

### Warp-size risk

Probing is written for 32-lane groups: `GROUP_SIZE = 32` tiles in the lookup, contains, dual-bucket and find_or_insert kernels (with `BLOCK_SIZE = 128`), `DIGEST_SPAN` digest loads per lane, `__ffs` (116) and `__popc` (19) on 32-bit vote results, `((block_size + 31) / 32) * 32` launch rounding, and `n % 32` / `dim >= 32` dispatch in merlin_hashtable.cuh. The cooperative-groups path is mostly expressed as `tiled_partition<32>`, which is a sub-wavefront tile on wave64 and therefore keeps 32-lane semantics if HIP's tile ballot/shfl are tile-relative. The raw `__ballot_sync`/`__any_sync` retry loops in update*.cuh are the explicit warp-width code: on wave64 (Instinct) a 32-bit mask silently drops lanes 32-63; on wave32 (Radeon) the native width matches, so wave32 is the lower-risk gate and wave64 is the one to test hardest. The retry loops spin (`while (__any_sync(mask, !completed))`) on keys locked by other threads; under AMD's lock-step execution a lock holder in the same wavefront releases within the same iteration, so no intra-wave deadlock is evident at screen depth, but this is the spin-under-contention class the planner should check (see gfx1151 low-CU starvation lesson from stdgpu).

### Build system and tests

- CMake (`project(merlin-hkvs LANGUAGES CXX CUDA)`, `-Dsm=80` selects `-gencode`), builds about 20 gtest executables and two benchmarks; googletest via submodule. `tests/accum_or_assign_test.cc` is named without its `.cu` suffix and relies on CMake's pre-CMP0115 extension guessing (cmake_minimum_required 3.10).
- Bazel (`WORKSPACE`, `include/BUILD` `cuda_cc_library`, `build_deps/gpus` TF-style cuda_configure) for TensorFlow Recommenders-Addons style consumers. No TF ops in this repo. Bazel ROCm support would be a separate, deferrable item; HugeCTR consumes the headers through CMake.
- Tests: 20 `tests/*_test.cc.cu` files, 208 TEST/TEST_F cases, all GPU; run with `bash run_all_tests.sh` (executes every `build/*_test`). Some tests size tables at 64M keys and `max_hbm_for_vectors` up to GB(16)/GB(100) (hybrid HBM+host), so a run on the gfx1151 laptop must exclude or shrink the largest cases (no big-tensor tests on the APU).
- Upstream supports CUDA >= 11.2, compute capability 8.0/8.6/8.7/9.0.

### Dependencies

No MOAT project dependency (CUB/Thrust/libcu++ come from ROCm: hipCUB, rocThrust, libhipcxx). HugeCTR depends on this project (recorded on origin/port/HugeCTR).

### Recommendation

Fork. Apache-2.0 throughout, no AMD port exists, it is actively maintained, header-only with no CUDA math libraries, and it unblocks HugeCTR, which Jeff has already ruled a take-up. Scope: library headers plus the CMake gtest suite on wave64 and wave32; the Bazel build and the research benchmarks/baselines stay out of the first port. Main risks: 32-bit warp masks and 32-lane digest probing on wave64, the `__pipeline_*` async-copy and `__vcmpeq4` shims across the hot kernels, and lock-free CAS correctness through libhipcxx scoped atomics.
