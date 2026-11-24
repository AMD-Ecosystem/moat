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
- The CMake HIP Debug flags (`-g -O`, corrected 2026-11-24; this line first said `-O0`) would compile kernels at clang's minimal `-O` level where nvcc's Debug (`-g`) still optimizes device code, so the HIP branch sets `CMAKE_HIP_FLAGS_DEBUG` optimised.
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
- CMake: `option(HKV_USE_HIP)`, `project(... CXX HIP)`, `find_package(hip hipcub rocthrust libhipcxx REQUIRED)`, `link_libraries(hip::hipcub roc::rocthrust libhipcxx::libhipcxx)`, test and benchmark `*.cc.cu` marked `LANGUAGE HIP` (in `benchmark/CMakeLists.txt` too, since source properties are directory-scoped), and `tests/hip_compat/` (one-line `cuda_runtime.h`, `cuda_runtime_api.h`, `cooperative_groups.h` that include `merlin/hip_compat.cuh`) prepended to the include path for the HIP build only, so the test sources stay untouched. `CMAKE_HIP_FLAGS_DEBUG` gets `-O3` appended: the root file's mid-file `set(CMAKE_BUILD_TYPE "Debug")` makes every target Debug, and CMake's HIP Debug flags are `-g -O` (corrected 2026-11-24; earlier text said `-O0`; the compile line reads `-g -O -O3`). Assertions stay live, so HIP's `__hip_check_mask` traps on any bad warp mask in the test run. The benchmark `-O3` genex became `$<$<COMPILE_LANGUAGE:CUDA,HIP>:-O3>`. `CMAKE_HIP_ARCHITECTURES` is not pinned.
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

## Review 2026-11-24 (linux-gfx942 MI300X, reviewer role, fork head 815cfdc)

### Summary

The port builds the header library, tests, benchmarks and a Bazel path with HIP through a compat header. Its forward-progress fixes keep the insert and update kernels it changed from hanging. The pipeline update kernels still hang on HIP when a batch repeats a key inside one wavefront, which CUDA survives, so the verdict is **Request Changes**. Review ran inline plus two read-only audit subagents (lock-wait loops across the library; CMake/Bazel/commits). Every finding below was re-read in the source, and finding 1 was reproduced on the GPU.

### Port Correctness / Fault Classes

1. **Same-wavefront lock wait left in the retry stage of the three pipeline update kernels (reproduced hang).**
   - Where: `include/merlin/core_kernels/update.cuh:576-583`, `update_score.cuh:561-568`, `update_values.cuh:581-588`.
   - The loop: `do { if (rank == 0) acquired = CAS(key -> LOCKED); acquired = g.shfl(acquired, 0); } while (acquired == 0);`, with the copy and the release store after the loop.
   - How it hangs: groups are 16 lanes, so 4 per wave64 wavefront and 2 per wave32. When two groups of one wavefront retry the same key at the same step, the winner leaves the loop and waits at its exit, and the loser spins on a slot that is released only after the loop.
   - Reached through `assign()`, `assign_scores()` and `assign_values()` with the default `unique_key=true` on a batch that repeats a key. `KernelSelector_Update*::callable()` picks the TLP or pipeline kernels only for `unique_key`. The TLP update kernels already exit wavefront-uniformly (`warp_any`); the pipeline ones do not.
   - Reproduced on gfx942 with ROCm 10.2, headers at 815cfdc. Probe setup:
     - `HashTable<uint64_t, float, uint64_t, kLru>` with capacity 128Ki, 128-slot buckets and dim 64, which selects `pipeline_update_kernel_with_io`;
     - 32Ki keys inserted, then a 1024-key `assign()` batch of distinct present keys, except that key 7 sits at index `16*g` (g < copies) of every 128-key block.
   - Probe results (each under `timeout 60`/`120`):

     | Batch | Result |
     |---|---|
     | copies=2 | hang (rc 124) |
     | copies=3 | hang (rc 124) |
     | copies=4 | hang (rc 124) |
     | copies=1 (duplicates only across wavefronts) | rc 0 |
     | unique control | rc 0 |

     `AMD_LOG_LEVEL=3`: the last kernel dispatched before the queued `group_lock::lock_update_kernel` is `pipeline_update_kernel_with_io<unsigned long, float, unsigned long, ...>`.
   - CUDA completes, because independent thread scheduling lets the winner release. Strictly this is outside the documented hint ("unique_key: if all keys in the same batch are unique"), but a GPU hang where CUDA returns is a behaviour divergence in the exact fault class this port fixes. HugeCTR, the consumer, is not known to guarantee unique batches.
   - The fix belongs under `kRetryLockedAtCaller`, as in `tlp_update_*_kernel`: do the CAS, copy and release in one pass of a loop with a wavefront-uniform exit (`while (warp_any(kFullWarpMask, !done))`), or take the retry list through a loop whose latch the holder reaches after its release. CUDA must stay unchanged.
   - Re-run the probe with `assign_scores`/`assign_values` too (their retry loop is the same code; only `assign()` was run here).
2. **`CUDA_ERROR_CHECK` is now CUDA-only.**
   - `CMakeLists.txt:85-88` moved `add_compile_definitions(CUDA_ERROR_CHECK)` (and `-lineinfo`) inside `if(NOT HKV_USE_HIP)`. A user who configures `-DCMAKE_BUILD_TYPE=Debug` gets the post-launch synchronizing error checks of `include/merlin/utils.cuh:101` on CUDA but not on HIP, and `tests/find_with_missed_keys_test.cc.cu:245` branches on the macro.
   - Fix: keep the define outside the guard and leave only `-lineinfo` CUDA-only.
3. **Dead define in the Bazel ROCm copts.** `build_deps/gpus/rocm/build_defs.bzl.tpl:16` passes `-DUSE_HIP=1`, but nothing reads `USE_HIP` (detection is `__HIP__`/`__HIPCC__` in `hip_compat.cuh`). Remove it.

### Records

4. `notes.md` (planner line 97, porter line 131) and the body of fork commit 8740206 say CMake's HIP Debug flags are `-O0`. They are `-g -O`: `Modules/Compiler/Clang-HIP.cmake:12` appends ` -O`, and `build-hip/CMakeCache.txt` shows `CMAKE_HIP_FLAGS_DEBUG:STRING=-g -O`. The `-O3` append is still justified (it matches nvcc optimizing device code in Debug). Correct the notes, and see the decision below for the commit body.

### Checked without findings

- The kernels the port changed:
  - The tile helpers return CONTINUE only on tile-uniform values (ballot, `g.any`, shuffled result).
  - Work before each `t -= blockDim.x * gridDim.x; continue;` is idempotent: same-value stores (`src_offset`, `keys_index`, `dst[i]=nullptr`) and reads. Counters, size increments and evicted outputs come after the check.
  - The `size_t` wrap is restored by the loop increment.
  - Each holder's critical section precedes the common latch, so no slot stays locked across an iteration. No double insert: a retried key re-runs the duplicate scan, so a key published meanwhile becomes DUPLICATE.
  - TLPv1 upsert, find_or_insert and upsert_and_evict publish in the pass that takes the slot, and REFUSED lanes hold no lock. The exit is `__any` over active lanes, so it is wavefront-wide on wave64 and on wave32, and the victim is the true minimum, as on CUDA.
- CUDA paths, by source reading:
  - `kRetryLockedAtCaller` is constexpr false.
  - Each added branch is `!k && <old cond>`, `k && ...`, or a CONTINUE test after a loop that exits only on non-CONTINUE.
  - `find_and_lock_for_update` never returns CONTINUE on CUDA, so the new check in `update_values_kernel` is dead there.
  - The `published`/`publish` refactor is order-preserving.
  - `warp_*` on CUDA are the original intrinsics with a 32-bit mask.
  - CUDA-only includes and the `uint64_t` atomics are kept under `#ifndef MERLIN_HIP`.

  **The CUDA compile check was not run here (this host has no nvcc); it runs elsewhere.**
- Other loops audited: TLPv2, hybrid lock-key, dual-bucket, pipeline upsert, lookup, erase and group_lock kernels. They skip LOCKED slots, are bounded (`remaining_candidates`), or never wait on a same-wavefront holder. The remaining corners (`score == MAX_SCORE` with a full locked bucket, `keys[-1]`) predate the port and behave the same on CUDA.
- `hip_compat.cuh`:
  - The SWAR `__vcmpeq4` is exact: no inter-byte carry, and `0x01010101*0xff` does not overflow.
  - The synchronous `__pipeline_memcpy_async` is safe, because every consumer already waits and syncs as CUDA requires.
  - The maskless `__ballot`/`__any`/`__syncwarp` act on the lanes present, which is what the masks encoded.
  - The `__shfl` pointer and enum overloads are fine.
  - The `wall_clock64()*10` timer assumption holds here (`hipDeviceAttributeWallClockRate` = 100000 kHz on gfx942).
- The `-O3` append to `CMAKE_HIP_FLAGS_DEBUG` is HIP-only and does not accumulate across reconfigures. `rocm_configure` leaves `cuda_configure` untouched when `NEED_ROCM` is unset. `utils/jargon.py --port HierarchicalKV` is clean.
- HIP build at 815cfdc with TheRock ROCm 10.2.0 wheels (job venv, no system ROCm), gfx942, default build: 52/52 steps, rc 0.

### Decision for a person (history)

8f10a46 introduces the skip-held-slot TLPv1 eviction that 815cfdc reverts. A bisect can land on an inexact-eviction commit, and 815cfdc's body refers to "the previous commit". 815cfdc's body also says "The CUDA code is unchanged", although the CUDA source moved into a `publish` lambda; "CUDA behaviour is unchanged" is accurate, given the PTX diff in the porter notes. 8740206's body has the `-O0` error (item 4). Agents may not force-push `moat-port`. Options:
- (a) Leave history as is and fix the wording in the upstream PR body (squash-merge upstream removes the intermediate state).
- (b) Authorize a one-time rewrite of `moat-port` before the review PR (squash 8f10a46 into 815cfdc, correct both bodies).
- (c) Leave everything.

The reviewer suggests (a) unless upstream merges with merge commits.

## Porter 2026-11-24 (linux-gfx942 MI300X, porter role, answers "Review 2026-11-24")

Requested by Jeff Daily (fleet job): address every finding of the 2026-11-24 review with new commits on `moat-port` (no amend, rebase or force-push), fix the pipeline update hang, and hunt the library for the same shape.

### Fork commits (on top of 815cfdc; pushed, head 592f800)

- `37516a1` [ROCm] Release slots in the pipeline update retry pass on HIP (finding 1)
- `02ff9c1` Add a repeated-key test for the pipeline update kernels (finding 1 regression test; shared, no prefix)
- `b4b770f` Build the repeated-key test with the other tests (its CMake target; split out because `CMakeLists.txt` also carried the finding-2 change and no commit was rewritten)
- `d44146c` [ROCm] Define CUDA_ERROR_CHECK in HIP Debug builds (finding 2)
- `592f800` [ROCm] Drop an unused define from the Bazel HIP options (finding 3)

Finding 4 (records): the `-O0` text in the planner and porter sections is corrected in place (CMake's HIP Debug flags are `-g -O`; the ninja compile line here reads `-g -O -O3`). The same error in fork commit 8740206's body is part of the history decision below, still open.

### Finding 1: the fix

The retry stage of `pipeline_update_kernel_with_io` (update.cuh), `pipeline_update_score_kernel` (update_score.cuh) and `pipeline_update_values_kernel_with_io` (update_values.cuh) now, under `kRetryLockedAtCaller` (HIP only):
- loops `while (warp_any(active_warp_mask, retry_mask != 0))`, and a group with nothing left `continue`s to that latch;
- scans the bucket once per pass (the `has_locked_keys` do-while becomes one pass) and keeps the key for the next pass when the key is not found but the bucket has a LOCKED slot;
- tries the compare-and-swap once per pass; a group that fails keeps the key and its position (`pending_pos`, a register, so the scan is not repeated, matching the CUDA loop that keeps `retry_pos`) for the next pass;
- a group that takes the slot copies the value, updates the score and releases the key in that same pass, then clears the bit.
No group ever spins inside a pass, and every holder releases before the common latch, whose condition (a wavefront vote) the compiler cannot decide. It is the shape upstream's `tlp_update_score_kernel` / `tlp_update_values_kernel` already use. On CUDA the flag is constexpr false: the ternary picks the original `retry_mask != 0`, the inner loops keep their original conditions, and the added `if (acquired == 0)` / `pending_pos` paths are unreachable.

Reproduction and regression test (`tests/repeated_key_test.cc.cu`, 9 gtest cases): `HashTable<uint64_t, float, uint64_t, kCustomized>`, 1024 buckets of 128, dim 64 (256-byte values select the pipeline kernels for assign/assign_values), 75% prefill (load factor > 0.6 selects the pipeline kernel for assign_scores); a 1024-key batch of present keys with one present key at index `16*g` (g < copies, copies = 2, 3, 4) of every 128-key block, i.e. the first key of 2-4 consecutive 16-thread groups (one wave64 wavefront; 2 per wave32 wavefront). Every copy carries the same value/score, so the result is checked exactly. The API call and a stream sync run on a worker thread under a 120 s watchdog that `_Exit(1)`s with "FAILED: ... did not finish", so a hang fails the binary instead of hanging it.
- Headers at 815cfdc (before the fix): `assign_2_copies`, `assign_4_copies`, `assign_scores_2_copies`, `assign_values_2_copies` each "did not finish within 120 s", rc 1.
- After the fix: 9/9 pass in about 1 s total. `AMD_LOG_LEVEL=3` shows `pipeline_update_kernel_with_io`, `pipeline_update_score_kernel` and `pipeline_update_values_kernel_with_io` dispatched 3 times each (once per copy count).
- The test is shared code and passes on CUDA by construction (independent thread scheduling); it was not compiled with nvcc here.

### The hunt: every lock-taking kernel (wave64 and wave32)

An inline reading plus a read-only audit subagent covered every kernel and helper under `include/` that CASes a key to LOCKED_KEY or takes another lock word. Verdicts, each re-checked in the source:
- **Tile helpers and their grid-stride callers** (insert_or_assign / upsert io and hybrid, find_or_insert, find_ptr_or_insert, insert_and_evict, accum_or_assign, the `*_with_io` update kernels, `prepare_cpu_update_kernel`): SAFE. On HIP `find_and_lock_when_vacant`, `find_and_lock_when_full`, `find_and_lock_for_update` and `try_find_and_lock_for_update` return CONTINUE on a LOCKED slot instead of spinning; each caller holds at most one slot and releases it on every path (success, REFUSED, accum mismatch, evict restore) inside the same grid-stride iteration, before the latch (`t -= stride; continue;`). The argument does not depend on the tile size (4, 8, 16, 32) or the wavefront width.
- **TLPv1 eviction** (upsert, find_or_insert, upsert_and_evict): SAFE, fixed in the previous round (publish in the pass, `warp_any` exit).
- **TLPv2, filter and hybrid eviction loops, pipeline upsert/find_or_insert/upsert_and_evict, dual-bucket upsert**: SAFE. They skip LOCKED slots when choosing a victim, the digest CAS loops are bounded by the candidate mask, and the pipeline eviction loops mark candidates visited, so no lane waits on a holder. One upstream edge case: see the deferral below.
- **`tlp_update_*` retry loops, `tlp_update_kernel_with_io`**: SAFE (release in the CAS pass; the latter never waits).
- **Lookups, contains, find (ptr), erase, export/dump, size, clear**: take no lock and never wait on LOCKED (a locked key reads as missing).
- **Rehash bucket mutex**: each tile locks only its own bucket.
- **Locks held across kernels** (`*_lock_key_hybrid` -> `*_unlock_key`, `find_or_insert_ptr_kernel_lock_key` -> unlock kernel or `unlock_keys`, `lock_kernel_with_filter` -> `unlock_keys`, group_lock kernels): no wavefront hazard. The locking kernels do not wait on a repeated key in the batch.
- **`gpu_select_kvm_kernel` (array_kernels.cuh:90)**: a thread spins until a lower thread empties a slot, and that thread empties its own slot only after its own spin, so mask `[0,1,1]` would hang on HIP. It has no caller (`gpu_boolean_mask` is unused in include/, tests and benchmarks), so it is left unchanged; it must be fixed if anything revives it.
- **Host-held locks** (`find_or_insert(..., locked_key_ptrs)`, `lock_keys`): writers that run before `unlock_keys` can spin on CUDA and HIP alike; that is the documented API contract, not a port fault.

Only the three pipeline update retry stages needed a fix, so they are the only kernels with a new test.

Evidence for the SAFE verdicts (local probe, not committed: `agent_space/tile_probe.cc.cu`, built with the same flags as the tests; each API call plus a later `find` under a 60 s watchdog): 19 API variants (`insert_or_assign`, `find_or_insert`, pointer `find_or_insert`, each with `unique_key` false and true; pointer `find_or_insert` with `locked_key_ptrs` + `unlock_keys`; `accum_or_assign`; `assign`, `assign_scores`, `assign_values` with `unique_key` false and true; `insert_and_evict` false and true; `lock_keys` + `unlock_keys`; `erase`; `export_batch`) x prefill load factors 0, 40, 70 and 95% (tile sizes 4, 8 and 32 and the TLP/pipeline selectors) x 2, 3 and 4 copies x spacing 1, 4 and 16 x dims 16 and 64 on 1024x128 tables, plus a full single 128-slot bucket with 2 and 4 copies. Each 128-key block repeats an existing key (even blocks) or a new key (odd blocks). Result on gfx942 at the fixed head: **1406 of 1406 runs completed, no hang, no error**. Only the wave64 hardware was available; wave32 rests on the argument above (nothing in it depends on the width, and the `warp_any` exits are `__any` over the active lanes of whatever width).

Deferral registered: `hkv-tlpv2-max-score-locked-bucket-spin`. In `tlp_v2_upsert_kernel_with_io`, a thread whose desired score is MAX_SCORE spins at `if (min_pos < 0) continue;` while every slot of its full bucket is LOCKED by threads of its own block, which publish only after the loop and a `__syncthreads`. That deadlocks on CUDA too, so it is upstream code and not a port fault; it needs a CUDA reproduction before an upstream report.

### Finding 2: verification

`cmake -DHKV_USE_HIP=ON -DCMAKE_BUILD_TYPE=Debug ...`: the HIP compile lines now carry `-DCUDA_ERROR_CHECK` (the default configure does not: `grep -c CUDA_ERROR_CHECK build-hip/build.ninja` = 0). In that Debug build `repeated_key_test` passes 9/9 and `find_with_missed_keys_test` passes 3 cases and fails the 4 `find_and_update_rr_*` cases with upstream's own `FAIL() << "R-role overlap requires CUDA_ERROR_CHECK disabled."`, the same result as a CUDA Debug build. The default build is unaffected.

### Build and tests at 592f800 (TheRock ROCm 10.2.0 wheels in the job venv, no system ROCm, gfx942)

- `cmake -S . -B build-hip -G Ninja -DHKV_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_PREFIX_PATH=$R && cmake --build build-hip -j 48`: 54/54, rc 0 (21 tests + 3 benchmarks).
- The same with `-DHKV_BUILD_REPRO_BENCHMARKS=ON` in `build-hip-repro`: 96/96, rc 0.
- Clean clone at 592f800, bazelisk 1.25.0 (Bazel 5.0.0): `ROCM_PATH=$R ROCM_AMDGPU_TARGETS=gfx942 bazel build --config=rocm //...`, "Build completed successfully", rc 0.
- GPU, `ulimit -c 0`, timeout per binary: `repeated_key_test` 9/9, `correctness_stress_test` 62/62, `assign_values_test` 3/3, `uint32_score_test` 4/4. The full suite is the validator's.
- `python3 utils/jargon.py --port HierarchicalKV`: clean.
- **CUDA compile check: not run. This host has no nvcc; it runs elsewhere.** It should cover the three changed update headers and the new test.

### Open for a person

- The history decision under "Review 2026-11-24 / Decision for a person (history)" is still open. This round added commits only, so 8740206's `-O0` body and the 8f10a46/815cfdc pair stand as described there.

## Review 2026-11-24 (round 2, linux-gfx942 MI300X, reviewer role, fork head 592f800)

### Summary

Delta `815cfdc..592f800` (5 commits) reviewed in the context of the full port (`451f240..592f800`). The three pipeline update retry stages now take, copy and release a slot in one pass, and the loop exit is wavefront-uniform. The hang the round-1 review reproduced is gone for `assign`, `assign_scores` and `assign_values`. The new test fails rather than hangs on the old headers. Findings 2 and 3 are fixed. Finding 4 is fixed in `notes.md`, with one leftover in `plan.md` (N2). No blocking problem, so the verdict is **review-passed**. The non-blocking items are below. Review done inline: this harness cannot dispatch child agents. Each item was re-read in the source before it was recorded.

**The CUDA compile check was not run here (this host has no nvcc); it runs elsewhere.** It should compile the three changed update headers and `tests/repeated_key_test.cc.cu`. A PTX diff of the update kernels against 815cfdc would confirm the source reading below.

### Non-blocking findings

- **N1. A lesson overstates one of its fixes (`notes.md:155`, "Lesson for cuda-to-rocm: a lock-retry loop deadlocks ...").** The lesson offers "Alternatively never wait on a held slot: when choosing an eviction victim, skip locked slots" as an equal fix. It gives no caveat, but this port reverted exactly that fix (8f10a46 -> 815cfdc): skipping changes which key is evicted under same-bucket contention, so a bucket no longer keeps exactly its top scores (Porter 2026-11-23, "Faults found and fixed", item 3). Before promotion, add that skipping is acceptable only where the kernel already tolerates inexact eviction, and otherwise prefer the publish-in-pass / uniform-exit fix. Severity: low. The lesson is promoted after this review, so the trunk session should apply this.
- **N2. `plan.md:51` still says CMake's HIP Debug flags are `-O0 -g`.** Round-1 finding 4 named only the notes lines, and those are corrected. The plan has the same error. The real flags are `-g -O` (`build-hip/CMakeCache.txt`). Severity: records only.
- **N3. Bisect wording in two commit bodies.** The Test Plan of `37516a1` says the repeated-key test is "added in the next commit". The source arrives in `02ff9c1`, but its CMake target arrives only in `b4b770f`. `02ff9c1`'s own Test Plan (`cmake --build build --target repeated_key_test`) fails at that commit, because the target does not exist yet. This affects history only and belongs with the open history decision (option (b) would fold it in). Severity: low.

### Verification of round-1 findings

1. **Pipeline update retry hang: fixed.**
   - Source: `include/merlin/core_kernels/update.cuh:546-611`, `update_score.cuh:531-590`, `update_values.cuh:551-613`.
   - Under `kRetryLockedAtCaller`:
     - the latch is `warp_any(active_warp_mask, retry_mask != 0)`, and finished groups `continue` to it;
     - the bucket scan is one pass, and a key not found in a bucket that has a LOCKED slot is kept for the next pass;
     - the CAS is tried once, and on failure the position is kept in `pending_pos`, as CUDA's spin keeps `retry_pos`;
     - the copy, score update and release-store run in the pass that took the slot, before the latch.
   - Why it cannot hang: no group spins inside a pass. Groups at `key_idx_base >= n` return before the loop and hold no lock. The pipeline stage before the loop releases every slot it took (CAS in step 3, release in step 4 or the emptying steps) and never waits. Two groups racing for one slot make progress, because one CAS wins each pass.
   - Duplicate handling: every copy sees the same key at the same position, so all copies are applied in sequence, as on CUDA. The argument does not depend on the wavefront width.
   - CUDA path (source reading):
     - `kRetryLockedAtCaller` is `constexpr false` (`kernel_utils.cuh:49-53`), so the ternary reduces to the original `retry_mask != 0`.
     - The added `if (k && ...) continue;` lines are dead.
     - Both inner do-whiles keep their original conditions (`!k && ...`).
     - `if (acquired == 0)` follows a loop that exits only when `acquired != 0`, so it is unreachable, and `pending_pos` stays -1. `retry_pos` is therefore `sm_position[...]` as before.
     - The CUDA semantics are unchanged.
2. **`CUDA_ERROR_CHECK`: fixed.** `CMakeLists.txt:87-92` (define at :88, CUDA-only `-lineinfo` at :90): the define is set for both toolkits in Debug, and only `-lineinfo` stays CUDA-only. Against upstream, the CUDA lines are unchanged apart from the added `if(NOT HKV_USE_HIP)`/`endif()` brackets. The porter's Debug-build evidence (`find_with_missed_keys_test` gives the same 4 R-role `FAIL`s as a CUDA Debug build) is consistent with `tests/find_with_missed_keys_test.cc.cu` branching on the macro.
3. **Dead Bazel define: fixed.** `-DUSE_HIP=1` is gone from `build_deps/gpus/rocm/build_defs.bzl.tpl`. `git grep 'USE_HIP\b'`, excluding `HKV_USE_HIP`, finds nothing at 592f800.
4. **Records: fixed in `notes.md`.** The leftover is in `plan.md` (N2). The `8740206` body stays with the history decision.

### GPU evidence (this review; gfx942, TheRock ROCm 10.2 SDK from the job venv, `ulimit -c 0`, HIP_VISIBLE_DEVICES as dispatched)

**Probe.**
- Source: `agent_space/review2/probe.cc.cu`, not committed.
- Compiled with the prescribed clang++ line, against the 592f800 headers and, for a negative control, against `git archive 815cfdc include tests/hip_compat`.
- Setup:
  - `HashTable<uint64_t, float, uint64_t, kLru>` with capacity 128Ki, 128-slot buckets and dim 64;
  - prefill keys `1..P`, with P = 32Ki (the round-1 setup) and P = 96Ki (75%, which selects the pipeline kernel for `assign_scores`);
  - a 1024-key batch of distinct present keys, with key 7 at index `16*g` (g < copies) of every 128-key block, and copies 0 meaning the unique control.
- After the call, `find` checks that every batch key is present and holds the expected value.
- Each run was under `timeout 120`.
- `AMD_LOG_LEVEL=3` confirmed which kernel each case dispatches:
  - `assign` -> `pipeline_update_kernel_with_io`;
  - `assign_values` -> `pipeline_update_values_kernel_with_io`;
  - `assign_scores` at 96Ki -> `pipeline_update_score_kernel`;
  - `assign_scores` at 32Ki -> `tlp_update_score_kernel`.

| Headers | Op | Prefill | copies 0 (control) | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|---|
| 592f800 | assign | 32Ki | ok | ok | ok | ok | ok |
| 592f800 | assign_scores (TLP) | 32Ki | ok | ok | ok | ok | ok |
| 592f800 | assign_values | 32Ki | ok | ok | ok | ok | ok |
| 592f800 | assign | 96Ki | ok | ok | ok | ok | ok |
| 592f800 | assign_scores (pipeline) | 96Ki | ok | ok | ok | ok | ok |
| 592f800 | assign_values | 96Ki | ok | ok | ok | ok | ok |
| 815cfdc | assign | 32Ki | rc 0 | - | hang (rc 124) | - | - |
| 815cfdc | assign_scores (pipeline) | 96Ki | - | - | hang (rc 124) | - | - |
| 815cfdc | assign_values | 32Ki | - | - | hang (rc 124) | - | - |

"ok" means the run completed with `bad=0` (every batch key found, every value as expected) and `size` equal to the prefill.

**`build-hip/repeated_key_test`.**
- The binary is current at 592f800: `ninja -C build-hip -n` reports "no work to do", and `git status` shows no modified tracked files.
- `timeout 600`: 9/9 passed in 932 ms, rc 0.
- Regression check: the same test source was built against the 815cfdc headers (`agent_space/review2/rk_test_815cfdc`, linked with `build-hip/lib/libgtest{_main,}.a`). It printed `FAILED: assign with 2 copies of a key did not finish within 120 s` and exited rc 1. The watchdog therefore turns a regression into a failure, not a hang.

### Checked without findings

- **Test `tests/repeated_key_test.cc.cu`.**
  - The batch is built from keys a `find` showed present, and the repeated key is distinct from the rest of the batch.
  - Every copy writes the same value and score, so the exact check is deterministic.
  - Both expected cases are correct: an untouched score stays 100 (prefill) for `assign_values`, and untouched values stay 0 for `assign_scores`.
  - The watchdog joins the worker before reading `err`.
  - The comment's warp arithmetic is right: indices 0/16/32/48 are in one wave64 wavefront, and on wave32 2 copies share one wavefront.
  - CMake target: added beside the other tests, and the HIP `*.cc.cu` glob marks it `LANGUAGE HIP`.
- **Commit messages of the 5 new commits.**
  - Titles are 48-61 characters.
  - `[ROCm]` appears on the three HIP-specific commits. The test and CMake-target commits are shared and carry no prefix.
  - Each body discloses the AI agent and has a fenced Test Plan.
  - No Co-Authored-By or noreply trailer.
  - `python3 utils/jargon.py --port HierarchicalKV`: clean.
- **Hunt verdicts, spot-checked in the source.**
  - `gpu_select_kvm_kernel`: its spin is at `array_kernels.cuh:90`, and its only caller `gpu_boolean_mask` has no caller anywhere in the tree.
  - `rehash_kernel_for_fast_mode`: each tile locks only `locks[t / TILE_SIZE]`, its own bucket, and `move_key_to_new_bucket` is a bounded loop (`core_kernels.cuh:537-597`).
  - TLPv2 digest CAS loops: bounded by `cmp_result`.
  - Pipeline upsert eviction: bounded by `remaining_candidates` (`upsert.cuh:799-850`).
  - The `hkv-tlpv2-max-score-locked-bucket-spin` deferral is accurate. With `score == MAX_SCORE` and every slot LOCKED, `min_pos` stays -1 and `score < min_score` is false, so the thread `continue`s indefinitely (`upsert.cuh:427-432`). The holders release only after the `__syncthreads` at `upsert.cuh:470`, so the same block deadlocks on CUDA as well.

### Decisions for a person

- The round-1 history decision (squash 8f10a46 into 815cfdc and fix the `8740206` body, or fix the wording only in the upstream PR body) is still open. N3 adds the `37516a1`/`02ff9c1` Test Plan wording to the same question. Reviewer suggestion unchanged: (a), unless upstream merges with merge commits.

## Validation 2026-11-24 (linux-gfx942 MI300X, validator role, fork head 592f800): FAILED, intermittent hang

Requested by Jeff Daily (same fleet job, after the round-2 review passed). TheRock ROCm 10.2.0 wheels in the job venv (HIP 7.17.26434, AMD clang 24.0.0git), no system ROCm, one pinned GPU, `ulimit -c 0`, a timeout around every GPU run. Builds at 592f800 are the porter-round builds recorded above (default 54/54, research benchmarks 96/96, both under `timeit.sh compile`).

**Verdict: validation-failed.** `find_or_insert_ptr_test` (`FindOrInsertPtrTest.test_find_or_insert_multi_threads`) hangs intermittently at 592f800; the job's instructions count any hang as a failure. Everything else passed.

### Full suite (run 1), per binary under a 3600 s timeout

All 21 binaries pass (241 gtest cases in 20 binaries; the earlier 232 plus the 9 new repeated-key cases), with `export_batch_if_test` as written (plain `main`, rc 0, prints `done`, 116 s). Cases and seconds: repeated_key 9 (1), correctness_stress 62 (10), dynamic_max_capacity 1 (3), lock_unlock 1 (1), reserved_keys 1 (0), merlin_memory_pool 6 (1), save_and_load 5 (22), group_lock 4 (21), find_with_missed_keys 7 (46), uint32_score 4 (46), assign_values 3 (76), dual_bucket 33 (165), find_or_insert_ptr_lock 13 (314), accum_or_assign 14 (533), assign_score 8 (314), insert_and_evict 9 (549), merlin_hashtable 24 (898), find_or_insert 19 (1145), find_or_insert_ptr 18 (1293). Nothing filtered or resized; `IS_BLOSSOM_CI` unset. Two other lanes shared the GPU.

### Insert/update binaries, runs 2 and 3 (repeated_key, correctness_stress, find_or_insert_ptr_lock, group_lock, lock_unlock, dual_bucket, assign_values, uint32_score, accum_or_assign, assign_score, insert_and_evict, merlin_hashtable, find_or_insert, find_or_insert_ptr)

- Run 2: 14/14 pass.
- Run 3: 13/13 pass up to `find_or_insert_ptr_test`. That lane was stopped by the session's 2-hour background limit while `test_find_or_insert_multi_threads` was running (not a test timeout), so that attempt has no verdict. The binary was rerun alone under `timeout 3600`: **rc 124, stuck in `test_find_or_insert_multi_threads`** (normally 250-330 s).

### The hang (`FindOrInsertPtrTest.test_find_or_insert_multi_threads`)

The case runs 64 host threads on one table (kLru, 32M -> 128M capacity, max_load_factor 0.5, dim 16), each with its own stream: `assign` of 1M random keys (dim 16 = 64-byte values selects `pipeline_update_kernel_with_io`, which this round changed), pointer `find`, pointer `find_or_insert` + `read_or_write_ptr`, then `cudaStreamDestroy`.

Repetition of the case alone (`--gtest_filter`, stall threshold 900 s):

| Head | Runs | Hangs |
|---|---|---|
| 592f800 (this round) | 11 with a verdict (full-binary runs 1 and 2, the rerun, 4 plain runs, 5 under rocgdb) | 3 (the rerun; plain run 2; rocgdb run 1) |
| 815cfdc (previous head; same toolchain, built from a clean clone into `agent_space/hkv815/build`) | 10 | 0 |

3/11 against 0/10 is suggestive but not conclusive (Fisher exact p about 0.1).

State at a hang (rocgdb run 1, the test launched under the SDK's `rocgdb`, SIGINT after 900 s):
- `info dispatches`: **"No dispatches are currently active."** No kernel is spinning; the GPU work is finished.
- Host: only one worker thread is left. It is blocked in `pthread_mutex_lock` inside `libamdhip64.so.7`, called from `hipEventQuery`, called from `nv::merlin::MemoryPool<DeviceAllocator<char>>::collect_pending_unsafe` (`include/merlin/memory_pool.cuh:451-453`). The main thread waits to join it; the other two threads are the ROCr async-event threads in `hsaKmtWaitOnMultipleEvents`. `/proc/<pid>/task/*/wchan`: 2 x `futex_wait_queue`, 2 x `kfd_wait_on_events`.
- The memory pool records its ready events on the caller's stream (`memory_pool.cuh:569`) and queries them later from any thread; every worker destroys its stream at the end. Querying an event whose recording stream was destroyed is legal on CUDA.

A minimal HIP program with exactly that pattern (64 threads, 16 pooled events recorded on per-iteration streams that are destroyed right away, queried under one mutex from other threads; `agent_space/evq/evq.hip`, not committed) completed 4 of 4 runs (12,800 iterations each), so the pattern alone did not reproduce it.

What is known and what is not:
- The hang is host-side, inside the HIP runtime's `hipEventQuery`, with no GPU work outstanding. The update kernels do not touch the memory pool, and no kernel is running at the hang, so a direct kernel deadlock from this round's change is ruled out.
- It was seen only at 592f800, 3 times in 11 runs, and never in 10 runs at 815cfdc. The only compiled difference between the two heads in this test is the three pipeline update headers, so the round's timing change remains a suspect until a larger baseline says otherwise.

Next steps for the porter (not done here):
1. Run 815cfdc's `find_or_insert_ptr_test --gtest_filter=*multi_threads` 20 or more times to see whether it ever hangs (pre-existing runtime race versus regression).
2. At a hang, find the owner of the HIP mutex: `AMD_LOG_LEVEL=4` or a debug-symbol `libamdhip64` and `rocgdb` `info threads` plus `print` of the mutex owner field. The capture did not identify the owner: the worker waits on HIP mutex `0x7fbe58017cd0` (futex value 2, contended), and no other application thread was alive, but the mutex's owner field was not read.
3. If it reproduces at 815cfdc or in a smaller program, file it against the HIP runtime (CLR) with this stack and register a `rocm-bug-report` deferral; a workaround in the port would be to synchronize before `cudaStreamDestroy` in the pool's users or to record pool events only on streams that outlive the query, which is a behaviour change to discuss first.

Logs (container-local, not committed): `agent_space/val/` (`A-*.log`, `B2-*.log`, `B3-*.log`, `B3r-find_or_insert_ptr_test.log`, `diag-*.log`, `diag2-rg592-1.{log,smi,wchan}`).

### Other validator items (all pass)

- Repeated-key tests: 9/9 in each of runs 1-3 (1-4 s).
- Default benchmarks (10-minute cap each): `dual_bucket_benchmark` rc 0 in 2 s; `merlin_hashtable_benchmark` and `find_with_missed_keys_benchmark` were still producing measurements at the cap (rc 124 from `timeout`; result tables printed, no error, fault or abort line), as in the previous round.
- Research benchmarks: build only, as requested: `build-hip-repro` 96/96, rc 0.
- `bazel build --config=rocm //...` in a clean clone at 592f800 with a fresh output root: "Build completed successfully, 13 total actions", rc 0. The Bazel `//benchmark:benchmark_util` ran on the GPU for 300 s, printing its HBM tables (insert_or_assign 0.556 B-KV/s at LF 0.5, dim 8), no error lines, stopped at the cap.
- Integrity: `git -C projects/HierarchicalKV/src status --porcelain` shows no modified tracked files. `python3 utils/jargon.py --port HierarchicalKV`: clean. The README documents the HIP CMake and Bazel builds.
- **CUDA compile gate: not run here. This host has no nvcc (and no conda CUDA env); the CUDA compile check runs elsewhere.** It should cover `update.cuh`, `update_score.cuh`, `update_values.cuh` and `tests/repeated_key_test.cc.cu` at 592f800.
- Records: the review's N2 (`plan.md:51` said HIP Debug flags `-O0 -g`) is corrected in place.
