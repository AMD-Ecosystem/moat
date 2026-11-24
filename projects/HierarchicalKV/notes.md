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

## Planner (2026-11-20, linux-gfx942 host, planner role)

### Decisions Jeff must make before a porter starts

Nobody could answer these during the planning job, so the plan assumes nothing beyond what is stated. Each needs Jeff's ruling.

**1. Scope cut (this is a partial port).** Intake proposed a library-plus-CMake-tests first port. These parts of the repository would be left out:
- the Bazel build (`WORKSPACE`, `include/BUILD`, `include/merlin/BUILD`, `build_deps/gpus` TF-style `cuda_configure`; header targets for TensorFlow Recommenders-Addons style consumers);
- the opt-in research benchmarks (`-DHKV_BUILD_REPRO_BENCHMARKS=ON`, 21 targets in `benchmark/CMakeLists.txt`, OFF by default);
- `benchmark/native_tbe` (a Python suite over FBGEMM-GPU TBE and a patched NVIDIA recsys-examples checkout on CUDA torch);
- `baselines/` (FetchContent builds of warpcore, BGHT and cuCollections for comparison).

Options:
- **A (planner's recommendation):** port the header library and every target the default CMake build makes (20 gtest executables, 214 cases, plus the 3 default benchmarks), and defer the four items above. HugeCTR, the reason for this port, consumes the headers through its CMake include path only.
- **B:** A plus the research benchmarks. The marginal cost is low: they include the same headers. Their own extra surface is `benchmark_util.cuh` (`__shfl*`, `__syncwarp`, `warpSize`) and `tbe_like_cache_baseline.cc.cu`.
- **C:** B plus a ROCm Bazel path (a `rocm_configure` equivalent). This is a large, separate piece of work with no consumer in view.

`native_tbe` and `baselines/` are outside every option: one is a third-party Python stack, the other benchmarks other hash-table libraries. Under A or B the porter records the deferred items as `scoped_out` in `surface.json` with your ruling as the reason. Deferred work goes in `deferred.json` via `utils/deferred.py add`.

**2. Windows gate.** The required gates are wave64, wave32 and windows. Upstream documents Linux only (CUDA >= 11.2, sm80+). Its only known consumer, HugeCTR, is Linux-only. The code uses no Linux-only API (`<thread>`, `<mutex>`, `<shared_mutex>`, `std::getenv`), so a Windows build with TheRock is plausible. However, seven test files size tables at 64M keys and/or `max_hbm_for_vectors` of 16-100 GB (`accum_or_assign`, `uint32_score`, `find_or_insert_ptr`, `export_batch_if`, `find_or_insert`, `merlin_hashtable`, `find_or_insert_ptr_lock`). The standing rule forbids those on the gfx1151 laptop.

Options:
- **(a)** waive the windows gate for this project (an agent may only suggest it: `moatlib.py suggest-waiver HierarchicalKV windows --reason ...`);
- **(b)** validate on Windows with a named gtest filter that excludes the large cases (the porter would derive the exact filter);
- **(c)** validate on Windows on a discrete Radeon host, if one exists, with the full suite.

**3. Base branch name.** `status.json` has `fork_default_branch: main`. Upstream's default branch is `master`, and the AMD-Ecosystem fork's GitHub default and only branch is also `master` (`gh api repos/AMD-Ecosystem/HierarchicalKV`, 2026-11-20). `moatlib.py set-base-branch HierarchicalKV master --by <you>` fixes it. The tool defines this as a person's ruling, so the planner did not run it. The porter cuts `moat-port` from this branch.

### Planning summary

- Upstream tip analysed: `451f240d185968d29ae022408e0bc3a8e987a4bb` (2026-11-15). Intake used `c79175830d51`; the counts moved slightly (208 -> 214 test cases).
- Build type: `cmake`, Strategy A (compat header, `enable_language(HIP)` behind an option). The library is header-only, so the compat header has to be included from the library headers themselves, not force-included by CMake: consumers compile it.
- Host toolchain facts (ROCm 10.1 pip SDK here):
  - libhipcxx is CCCL 3.0.2-based. Its include root is `<sdk>/include/hipccl` and it is found with `find_package(libhipcxx)`.
  - Its `<cuda/barrier>` and `<cuda/std/semaphore>` `#error` on AMD. HKV includes both but uses neither, so the includes get guards.
  - rocThrust is `THRUST_VERSION 200805` and has `thrust::hip::par_nosync`; `thrust::cuda` exists there only for `memory_resource`.
  - hipCUB 4.7.
- HIP's `*_sync` functions run `__hip_check_mask`, which asserts the mask equals the arriving lanes. The upstream CMake ends up Debug (`set(CMAKE_BUILD_TYPE "Debug")` at line 121, after the Release default), so `NDEBUG` is undefined and that assertion traps. The early-return TLP kernels in `update_values.cuh`/`update_score.cuh` must build their masks from the lanes actually present, not from `0xffffffff` widened.
- The CMake HIP Debug flags (`-O0`) would compile kernels unoptimised where nvcc's Debug (`-g`) does not, so the HIP branch sets `CMAKE_HIP_FLAGS_DEBUG` optimised.
- Full inventory, risks, change list, build and test commands are in `plan.md`. `surface.json` was generated (76 CUDA files, 34 components), and 6 planner-added components cover what the scan cannot see: the 19 foreach-built repro benchmarks, `native_tbe`, Bazel, the inline-PTX timer, hybrid host-memory values and the libcu++ atomics.
- No GPU was used and nothing was built in this run.

## Rulings on the planner's questions (Jeff Daily, 2026-11-23)

- Decision 1 (scope): **C**. Port the header library, every default CMake target (the gtest executables and the 3 default benchmarks), the opt-in research benchmarks (`-DHKV_BUILD_REPRO_BENCHMARKS=ON`), and a ROCm Bazel path (a `rocm_configure` equivalent beside `cuda_configure`). `benchmark/native_tbe` and `baselines/` stay out (outside every option the planner offered).
- Decision 2 (windows gate): **b** (ruled later the same day). Validate on the Windows laptop with a named gtest filter that excludes the large cases (64M-key tables, 16-100 GB `max_hbm_for_vectors`). No waiver. The laptop run comes last, after wave64 and wave32 pass at the same head, and the validator must read each remaining test for repeated CAS/spin loops and reduce repetition counts in a driver before running (a repeated atomicCAS test hard-froze this laptop on 2026-11-20).
- Decision 3 (base branch): **yes**. `set-base-branch HierarchicalKV master --by jeffdaily` was run on 2026-11-23.

## Porter (2026-11-23, linux-gfx942 MI300X, porter role, scope C)

### Questions and gaps for a person (read first)

1. **Bazel "tests".** Upstream's Bazel build defines no test targets: `//include/...` are header libraries and `//benchmark:benchmark_util` is the only binary (it compiles `merlin_hashtable_benchmark.cc.cu`). The `cuda_cc_test` macro exists but nothing uses it. I took the ruling's "Bazel header targets and tests build for ROCm" to mean: every Bazel target builds with `--config=rocm` and `benchmark_util` runs on the GPU. Options: (a) accept that as the Bazel evidence (what this round delivers); (b) add Bazel gtest targets for ROCm. That would be new upstream surface for both toolkits, beyond a minimal port, and needs your call.
2. **CUDA Bazel config not exercised.** `bazel build --config=cuda` needs the full TF-style toolkit (CUPTI, cuBLAS, cuDNN; `cuda_configure` stops at "Could not find any cupti.h"), which the minimal nvcc toolkit assembled here does not have. The shared Bazel edits are two header-list entries (`multi_vector.hpp`, which upstream's BUILD was missing, and `hip_compat.cuh`) plus a `cuda_configure` early return taken only when `NEED_ROCM=1`. A CUDA Bazel build on an NVIDIA host would close this.
3. **Windows.** The windows gate was ruled (b) while this job ran. Nothing Windows-specific was done here: no Windows build attempt, no filter (that is the validator's, per the ruling). The HIP CMake path has not been tried on Windows.
4. **Three research-benchmark failures that look upstream, registered as deferrals.** (a) `hkv-tlpv2-eviction-skips-inflight-slot`: `hkv_adversarial_performance_benchmark` fails its exact-admission oracle in `bucket_skew_1over16` (12 to 556 errors, timing dependent) inside the unmodified upstream `tlp_v2_upsert_kernel_with_io`, whose victim scan skips in-flight slots. The same code runs on CUDA, but no NVIDIA run was possible here. Options: run it on a CUDA host and file upstream if it reproduces; or decide the port should make TLPv2 exact on HIP (a behaviour change beyond a port). (b) `hkv-lookup-modes-lf075-bucket-overflow`: `hkv_lookup_modes_benchmark` with its default parameters overflows 779 buckets at LF 0.75 (CPU-computed from the hash), so its all-found check cannot pass on any GPU. (c) `hkv-dual-bucket-hardcodes-bucket-128`: `e10_dual_bucket_analysis` faults (`hipErrorIllegalAddress` in `dual_bucket_pipeline_upsert_kernel_with_io`) on a 64-slot dual-bucket table, because the dual-bucket upsert path hard-codes 128-slot buckets and `init()` does not reject other sizes; out of bounds on any GPU. Whether and how to report these upstream, or whether the port should guard (c), is a person's call.

### Fork commits (`moat-port`, base `master` = upstream 451f240)

- `2e1f3fc` Name the accum_or_assign_test source with its .cu suffix (shared, no prefix)
- `ffdd97b` Declare multi_vector.hpp in the Bazel merlin target (shared upstream Bazel bug: `//benchmark:benchmark_util` fails "'merlin/multi_vector.hpp' file not found" under any toolkit)
- `8740206` [ROCm] Build the library, tests and benchmarks with HIP
- `89198a1` [ROCm] Retry locked slots through the grid-stride loop on HIP
- `fe6cdd2` [ROCm] Add a Bazel configuration for ROCm
- `8f10a46` [ROCm] Skip held slots in the TLPv1 eviction scans on HIP (superseded by the next commit)
- `815cfdc` [ROCm] Publish TLPv1 inserts inside the eviction loop on HIP

### What the port does

- `include/merlin/hip_compat.cuh` (new) is included first by every library header that touches a CUDA API (`debug.hpp` heads almost every chain). It detects HIP with `__HIP__ || __HIPCC__ || __HIP_PLATFORM_AMD__` and defines `MERLIN_HIP`. Under HIP it includes the HIP runtime, `hip_fp16`, cooperative groups plus `hip/cooperative_groups/hip_reduce.h`, and the three hipCUB device headers (with `namespace cub = hipcub`), and maps the ~70 CUDA runtime names the repo uses. `cudaMallocHost` maps to `hipHostMalloc`, whose C++ template takes the 3-arg form, and `cudaMemcpyTo/FromSymbol` wrap `HIP_SYMBOL`. It also supplies a SWAR `__vcmpeq4`, synchronous `__pipeline_memcpy_async` with no-op commit/wait, `__ldcs`/`__stcs` templates (nontemporal builtins for arithmetic types), `umax`, `global_timer_ns` (`wall_clock64()*10`, fixed 100 MHz), and global `__shfl` overloads for pointers and enums. The latter must be declared before `hip_cooperative_groups.h` so the tile `shfl` template finds them (CUDA tile shfl takes any type; HIP's `__shfl` only arithmetic ones). Lane-mask helpers `warp_ballot`/`warp_any`/`warp_sync`, `warp_mask_t` and `kFullWarpMask` are the CUDA intrinsics on CUDA and the maskless forms on HIP. They exist in both builds, but the functions only under `__CUDACC__ || __HIPCC__`, so host-only includes of `debug.hpp` still compile.
- CUDA-only includes are guarded with `#ifndef MERLIN_HIP`: `<cuda/barrier>` and `<cuda/std/semaphore>` (libhipcxx `#error`s on AMD and neither is used), `<cuda_pipeline.h>`, CG, `cub/cub.cuh`, `cuda_runtime*.h`, `cuda_fp16.h`. libhipcxx serves `<cuda/atomic>` and `<cuda/std/functional>`.
- `CUDART_VERSION >= 11030` guards gain `|| defined(MERLIN_HIP)`, so the TLPv2 kernels and the instrumented upsert-outcomes path (`insert_or_assign(..., outcomes)`, used by `hkv_hash_sensitivity_benchmark`) are available on HIP. `CUDART_VERSION >= 10000` in `utils.cuh` gains the same, for `attr.type`. Thrust uses `thrust::hip::par_nosync`. HIP already defines the `uint64_t` overloads of atomicCAS/Exch/Add (as `unsigned long`), so `utils.cuh`'s copies are CUDA-only. Two `const auto found_vote = g.ballot(...)` became `unsigned int`; a HIP tile ballot returns 64 bits, which made `__ffs` ambiguous.
- CMake: `option(HKV_USE_HIP)`, `project(... CXX HIP)`, `find_package(hip hipcub rocthrust libhipcxx REQUIRED)`, `link_libraries(hip::hipcub roc::rocthrust libhipcxx::libhipcxx)`, test and benchmark `*.cc.cu` marked `LANGUAGE HIP` (in `benchmark/CMakeLists.txt` too, since source properties are directory-scoped), and `tests/hip_compat/` (one-line `cuda_runtime.h`, `cuda_runtime_api.h`, `cooperative_groups.h` that include `merlin/hip_compat.cuh`) prepended to the include path for the HIP build only, so the test sources stay untouched. `CMAKE_HIP_FLAGS_DEBUG` gets `-O3` appended: the root file's mid-file `set(CMAKE_BUILD_TYPE "Debug")` makes every target Debug, and HIP Debug is `-O0`. Assertions stay live, so HIP's `__hip_check_mask` traps on any bad warp mask in the test run. The benchmark `-O3` genex became `$<$<COMPILE_LANGUAGE:CUDA,HIP>:-O3>`. `CMAKE_HIP_ARCHITECTURES` is not pinned.
- `tbe_like_cache_baseline.cc.cu`: three `__shfl_sync(0xffffffff, ...)` use `nv::merlin::kFullWarpMask`. The kernel is warp-per-miss on `warpSize` throughout with 256-thread blocks; its `grid_for(..., block / 32)` only sizes a grid-stride launch.
- Bazel: `build_deps/gpus/rocm_configure.bzl` (`rocm_configure` -> `@local_config_rocm`: a `unix_cc_toolchain_config` clang toolchain on the ROCm LLVM, built-in include dirs from `clang -E -v` plus `$ROCM/include` and the resource dir, a `rocm` target linking `-lamdhip64` with RUNPATH, and `build_defs.bzl` with HIP versions of the `cuda_*` macros: `-x hip --rocm-path --offload-arch=<each> -isystem <rocm>/include/hipccl -DUSE_HIP=1`). With `NEED_ROCM=1`, `cuda_configure` writes a forwarding `@local_config_cuda` (alias `cuda` -> `@local_config_rocm//rocm`, `build_defs.bzl` re-exporting the ROCm macros), so the BUILD files are unchanged and no CUDA install is needed. `.bazelrc` `build:rocm` sets `--crosstool_top=@local_config_rocm//crosstool:toolchain`, `NEED_ROCM=1`, and passes `ROCM_PATH` and `ROCM_AMDGPU_TARGETS` through (default: `offload-arch`).
- Docs: README "with cmake for AMD GPUs (ROCm)" and "with bazel for AMD GPUs (ROCm)" sections plus a requirements line, and one line in `benchmark/README.md`. The Sphinx site renders the README, so there is no other build doc to update.

### Faults found and fixed (SIMT forward progress)

All the tile and thread-per-key insert and update kernels hung on gfx942. The cause: an AMD wavefront runs its tiles and threads in lockstep, and lanes that leave a loop wait at its exit for the rest of the wavefront. Upstream code spins in a loop while a slot reads `LOCKED_KEY`, while the holder, another tile or thread of the same wavefront, does its copy and unlock after that loop. CUDA's independent thread scheduling lets the holder proceed. Three shapes:
1. Tile helpers `find_and_lock_when_vacant`/`_when_full`/`find_and_lock_for_update` spin in place on LOCKED (`if (vote) continue;`, `while (load == LOCKED) {}`), and their callers retry CONTINUE in a do-while, with the critical section after the loop. Seen: `AccumOrAssignTest.test_export_batch_if` hung in `accum_or_assign_kernel_with_io<...,4>` (GPU 100% for minutes, found by `AMD_LOG_LEVEL=3`, last ShaderName).
2. The update kernels already retry at the caller with lock, copy and unlock inside the do-while (upstream's `try_find_and_lock_for_update` comment says spinning "can prevent another tile in the warp from publishing a held key"). Still, `update_score_kernel<...,4>` hung alone under `AMD_SERIALIZE_KERNEL=3` (`CorrectnessStress.DeferredDuplicateWritersApplyEveryFrequencyIncrement`). Once a tile owns the slot the do-while condition is statically false, so the compiler can route the critical section straight to the loop exit, where it runs after all lanes leave. Fix for 1 and 2: `kRetryLockedAtCaller` (true only under HIP). The helpers return CONTINUE instead of spinning, and every caller retries the key from the top of its grid-stride loop (`t -= blockDim.x * gridDim.x; continue;`). The holder's critical section and the retrier both reach the common latch, whose condition (`t < N`) the compiler cannot know. Used by 8 insert/update callers plus `prepare_cpu_update_kernel` (stride `/TILE_SIZE`).
3. The thread-per-key TLPv1 kernels (`tlp_v1_{upsert,upsert_and_evict,find_or_insert}`) pick the lowest-score slot and retry while it is LOCKED, and the holder's copy and unlock come after the eviction loop. Seen: `FindOrInsertTest.test_basic_when_full` hung in `tlp_v1_find_or_insert_kernel_with_io` (1M-slot table filled with 1M keys). A first fix (8f10a46) skipped held slots when choosing the victim, as the TLPv2 and pointer kernels upstream already do. Skipping is inexact under same-bucket contention: a thread that skips an in-flight slot holding the bucket's lowest score evicts the second-lowest key instead, so a bucket no longer keeps exactly its highest scores. That is what `hkv_adversarial_performance_benchmark`'s exact-admission oracle catches in the upstream TLPv2 kernel (finding 1 under Benchmarks). 815cfdc reverts the skip so that TLPv1 stays exact. Under HIP, a thread that took a slot in the digest scan publishes before the eviction loop, a thread that takes one in the loop publishes in that same pass, and the loop condition is `warp_any(kFullWarpMask, occupy_result == INITIAL)`, so the wavefront leaves together and no publish can be moved past the exit. Waiting threads re-read the bucket after the holder has published and evict the true minimum, as on CUDA. `tlp_v1_upsert_and_evict` (whose retry loop upstream already reworked to publish inside it) only gains the wavefront-wide exit and a `published` flag. The post-loop code moved into a `publish` lambda that CUDA calls once after the loop, as before.
CUDA: the flag is `constexpr false`, and sm_80 PTX of the four largest test TUs is identical to upstream (see below).

## Lesson for cuda-to-rocm: a lock-retry loop deadlocks when the holder is in the same wavefront

Belongs in `references/faults-wavefront.md` (next to "Intra-wave barrier divergence").

Symptom: a hash-table or lock-based kernel that passes on CUDA hangs on AMD with the GPU at 100%, often only once there is contention (a full bucket, duplicate keys in a batch). `AMD_LOG_LEVEL=3` shows the last `ShaderName` launched; `AMD_SERIALIZE_KERNEL=3` pins down which of several concurrent kernels never returns.

Cause: a lane that leaves a loop waits at the loop's exit until every lane of the wavefront has left. CUDA's independent thread scheduling (Volta+) lets a lock holder run ahead; AMD does not. Any retry loop that spins while a slot is locked deadlocks when the holder is another tile or thread of the same wavefront that already left the loop and does its copy and unlock after it. Three shapes were seen:
- a helper spins in place (`while (load == LOCKED) {}`, or a ballot-and-`continue` loop) with the critical section after the call;
- the caller retries in a `do { ... } while (result == CONTINUE)` with the critical section after the loop;
- the critical section is inside the do-while, but once the lock is taken the loop condition is statically false, so the compiler may route the critical section to the loop exit, where it runs after the other lanes. Putting the unlock "inside the loop" is not enough on its own.

Fix: retry through a loop whose latch every lane reaches after finishing its critical section, and whose condition the compiler cannot decide on that path. In a grid-stride kernel, re-run the same index from the top (`t -= stride; continue;`). A uniform exit (`while (__any(!done))`) also works. Alternatively never wait on a held slot: when choosing an eviction victim, skip locked slots and retry if none is left. Gate the change so the CUDA path stays identical, and check with a PTX diff. (HierarchicalKV, ROCm 10.2, gfx942)

## Lesson for cuda-to-rocm: tile shuffles of pointers and enums need `__shfl` overloads, declared before the CG header

Belongs in `references/build-headers-macros.md` (near "Count cooperative-groups call sites").

CUDA's `thread_block_tile::shfl` takes any trivially copyable type. HIP's tile `shfl` calls `__shfl(var, src, width)`, which exists only for arithmetic types, so `g.shfl(ptr, lane)` or `g.shfl(enum_value, lane)` fails with "no matching function for call to '__shfl'" deep inside `amd_hip_cooperative_groups.h`. Add template overloads (pointer: round-trip through `unsigned long long`; enum: through `std::underlying_type`). Declare them after `hip/hip_runtime.h` and before `hip/hip_cooperative_groups.h`: for a pointer to a fundamental type there is no associated namespace for argument-dependent lookup, so only declarations visible at the template's definition are found. A HIP tile `ballot()` also returns `unsigned long long` (the tile's own bits, compacted), so `const auto v = g.ballot(...); __ffs(v)` becomes ambiguous; store it in `unsigned int` as the CUDA type was. (HierarchicalKV, ROCm 10.2)

## Lesson for moat-porting: `pkill -f <pattern>` on a fleet worker kills the worker's own shell

Belongs in the "Fleet hosts" section.

The Bash tool runs each command in a shell whose command line contains the whole command text, so `pkill -f build-hip/foo_test` (or `pgrep -f ... | kill`) matches that shell too and the call dies with exit 144 before reporting anything. Collect PIDs with `ps -eo pid,args | grep <pattern> | grep -v grep` into a file in one call and kill them in the next. (HierarchicalKV)

### Build recipe (linux-gfx942, this round)

Toolchain: TheRock ROCm 10.2.0 wheels in the job venv: `python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[libraries,devel,device-gfx942]==10.2.0"`. HIP 7.17.26434, AMD clang 24.0.0git, hipCUB/rocThrust/libhipcxx from the same SDK (libhipcxx root `<sdk>/include/hipccl`). CMake 3.31.6 and Ninja. No system ROCm. The older indexes (`repo.amd.com/rocm/whl/<gfx>`, `rocm.nightlies.amd.com/v2`) stop at 7.x; 10.x releases are on `stable.repo.amd.com/rocm/whl-next/` and nightlies on `nightly.repo.amd.com/rocm/whl-next/`.

```
R=$(rocm-sdk path --root); export ROCM_PATH=$R PATH=$R/bin:$PATH
git submodule update --init tests/googletest
cmake -S . -B build-hip -G Ninja -DHKV_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_PREFIX_PATH=$R
cmake --build build-hip -j $MAX_JOBS          # 20 tests + 3 benchmarks, about 3 min at -j192
cmake -S . -B build-hip-repro -G Ninja -DHKV_USE_HIP=ON -DHKV_BUILD_REPRO_BENCHMARKS=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_PREFIX_PATH=$R
cmake --build build-hip-repro -j $MAX_JOBS    # + 21 research benchmarks
```
Run the tests with `ulimit -c 0` and a generous per-binary timeout (the HIP-linked executables get a RUNPATH to `<sdk>/lib`, so `LD_LIBRARY_PATH` is not needed): several upstream tests take 10-18 minutes here (see the timings below).

Bazel: bazelisk 1.25.0 (GitHub release binary) fetches Bazel 5.0.0 per `.bazeliskrc`, all under the job directory (`BAZELISK_HOME`, `--output_user_root`):
```
ROCM_PATH=$R ROCM_AMDGPU_TARGETS=gfx942 bazel build --config=rocm //...
```
As upstream's README says, do not clone with `--recursive` for Bazel: with `tests/googletest` checked out, `//...` also loads googletest's own Bazel packages, which need `@rules_python` (fails the same way for CUDA).

### GPU test results at 815cfdc (final head; MI300X gfx942, ROCm 10.2, CMake default build type = Debug with HIP -O3, assertions live)

All 20 binaries pass: 232 gtest cases in 19 binaries, plus `export_batch_if_test` (a plain `main`, rc 0, prints `done`). Per binary at 8f10a46 with the GPU to itself (cases, seconds): accum_or_assign 14 (255), assign_score 8 (295), assign_values 3 (74), correctness_stress 62 (10), dual_bucket 33 (139), dynamic_max_capacity 1 (4), export_batch_if main (71), find_or_insert_ptr_lock 13 (268), find_or_insert_ptr 18 (1066), find_or_insert 19 (1005), find_with_missed_keys 7 (20), group_lock 4 (21), insert_and_evict 9 (608), lock_unlock 1 (1), merlin_hashtable 24 (685), merlin_memory_pool 6 (0), reserved_keys 1 (1), save_and_load 5 (16), uint32_score 4 (40). The same 20/20 and 232 cases at 815cfdc, run while benchmark lanes shared the GPU (accum_or_assign 799 s, assign_score 722 s, find_or_insert_ptr 1161 s, find_or_insert 1026 s, merlin_hashtable 684 s). Nothing was filtered or resized. `IS_BLOSSOM_CI` is unset, so the hybrid HBM+host cases (`max_hbm_for_vectors = 0`) ran. The concurrency set (find_or_insert_ptr_lock, group_lock, lock_unlock, correctness_stress, dual_bucket) passed in 3 of 3 runs at 8f10a46 and 3 of 3 at 815cfdc. Kernels ran on the GPU: `AMD_LOG_LEVEL=3 save_and_load_test` logged 680,413 dispatches, all `nv::merlin` kernels (allocate_bucket_vectors/others and get_bucket_others_address 163,840 each, then upsert_kernel_lock_key_hybrid, write_kernel_unlock_key, dump_kernel, group_lock).

Where the time goes: table init launches three single-thread kernels per bucket (upstream design), so 64M-128M-slot tests spend minutes in launch overhead. The 128M-slot `*_customized_correct_rate` cases take 5-7 minutes each, and `test_find_or_insert_multi_threads` 265 s. No test needs more memory than the MI300X has; the 64M-key and 16-100 GB-allowance cases ran as written.

Before the fixes (at 89198a1/fe6cdd2): `AccumOrAssignTest.test_export_batch_if`, `CorrectnessStress.DeferredDuplicateWritersApplyEveryFrequencyIncrement` and `FindOrInsertTest.test_basic_when_full` hung (GPU 100%, killed at the timeout); each passes after its fix (0.4 s, 0.5 s, 2.2 s alone).

### CUDA no-regression (real nvcc)

No nvcc on this host, so a CUDA 12.8 toolkit was assembled from NVIDIA's redistributable archives (cuda_nvcc 12.8.93, cuda_cudart 12.8.90, cuda_cccl 12.8.90; sha256-checked against `redistrib_12.8.1.json`; `lib64 -> lib` symlink needed) under the job directory. At 8f10a46 and again at the final head 815cfdc:
- `cmake -S . -B build-cuda -G Ninja -Dsm=80 -DHKV_BUILD_REPRO_BENCHMARKS=ON -DCMAKE_CUDA_COMPILER=<toolkit>/bin/nvcc -DCUDAToolkit_ROOT=<toolkit>` then `ninja`: 94/94 steps, rc 0 (all 20 tests, 3 default and 21 research benchmarks).
- PTX: `nvcc -std=c++17 -arch=sm_80 -ptx` of `merlin_hashtable_test`, `find_or_insert_test`, `accum_or_assign_test` and `correctness_stress_test` (`-DHKV_ENABLE_P1_PARTITION_TESTS=1`) from upstream 451f240 and from the port, with `.loc`/`.file`/comment lines dropped and nvcc's per-file `_INTERNAL_<hash>_` and `__N__<hash>_` normalized (they follow the source path): identical at 8f10a46 (0.5M-2.7M lines each). At 815cfdc, after also canonicalizing virtual-register numbers per function, the only differences are swapped operands of commutative `and.pred` instructions (60-240 lines per TU, from the TLPv1 `publish` lambda; `diff | grep -v and.pred` is empty). So the CUDA device code is semantically the upstream code.
- The CUDA Bazel config was not run (question 2 above).

### Benchmarks

Smoke runs at 815cfdc: each binary once, default arguments, a 10-minute cap, in three parallel lanes beside the test suite. Throughput numbers are therefore informational and gate nothing. The only arguments supplied are the required ones: `concurrency_benchmark triple_group`, `concurrency_rw_lock rw_lock`, and `--source-id <sha256 of git ls-tree -r HEAD>` for both digest-ablation binaries (a provenance label printed in their output).
- Ran to completion, no error: `dual_bucket_benchmark` (4 s), `e9_eviction_comparison` (204 s), `hash_sensitivity_benchmark`, `hkv_correctness_stress_benchmark` (`cases=15;failed_cases=0`), `hkv_hash_sensitivity_benchmark`, `latency_benchmark` (478 s), `p25_admission_control`, `tbe_like_cache_baseline` (its warp-per-miss kernel with `kFullWarpMask` shuffles runs in the Debug build with HIP's mask checks live).
- Producing measurements at the 10-minute cap, no device fault or error line: `batch_size`, `bucket_size`, `concurrency_benchmark`, `concurrency_rw_lock`, `e17_single_vs_dual`, `e19_cache_quality_sweep`, `e8_hkv_baseline`, `error_bars`, `score_strategy`, `merlin_hashtable_benchmark`, `find_with_missed_keys_benchmark`. Also the two digest-ablation binaries, which spend minutes in table init: an `AMD_LOG_LEVEL=3` trace shows 1.76M init launches in 90 s for a 128M-slot table, so they are slow, not hung.
- Failing, all traced to upstream (questions 4a-c, registered as deferrals): `hkv_adversarial_performance_benchmark` (rc 2, exact-admission oracle in `bucket_skew_1over16`, inside the unmodified TLPv2 upsert); `hkv_lookup_modes_benchmark` (rc 1, LF 0.75 all-found check that 779 overflowing buckets make impossible); `e10_dual_bucket_analysis` (rc 134, memory fault on a 64-slot dual-bucket table in the 128-hard-coded dual-bucket upsert).
- Bazel `//benchmark:benchmark_util` (built with `--config=rocm` in a clean clone at 815cfdc) prints its full HBM tables on the MI300X (insert_or_assign about 0.55 B-KV/s at LF 0.5 for dim 8, with the GPU shared) until the cap.
