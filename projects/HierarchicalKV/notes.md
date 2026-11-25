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

Promoted to cuda-to-rocm/references/faults-wavefront.md in 9d8549ee.

Belongs in `references/faults-wavefront.md` (next to "Intra-wave barrier divergence").

Symptom: a hash-table or lock-based kernel that passes on CUDA hangs on AMD with the GPU at 100%, often only once there is contention (a full bucket, duplicate keys in a batch). `AMD_LOG_LEVEL=3` shows the last `ShaderName` launched; `AMD_SERIALIZE_KERNEL=3` pins down which of several concurrent kernels never returns.

Cause: a lane that leaves a loop waits at the loop's exit until every lane of the wavefront has left. CUDA's independent thread scheduling (Volta+) lets a lock holder run ahead; AMD does not. Any retry loop that spins while a slot is locked deadlocks when the holder is another tile or thread of the same wavefront that already left the loop and does its copy and unlock after it. Three shapes were seen:
- a helper spins in place (`while (load == LOCKED) {}`, or a ballot-and-`continue` loop) with the critical section after the call;
- the caller retries in a `do { ... } while (result == CONTINUE)` with the critical section after the loop;
- the critical section is inside the do-while, but once the lock is taken the loop condition is statically false, so the compiler may route the critical section to the loop exit, where it runs after the other lanes. Putting the unlock "inside the loop" is not enough on its own.

Fix: retry through a loop whose latch every lane reaches after finishing its critical section, and whose condition the compiler cannot decide on that path. In a grid-stride kernel, re-run the same index from the top (`t -= stride; continue;`). A uniform exit (`while (__any(!done))`) also works. Alternatively never wait on a held slot: when choosing an eviction victim, skip locked slots and retry if none is left. Gate the change so the CUDA path stays identical, and check with a PTX diff. (HierarchicalKV, ROCm 10.2, gfx942)

## Lesson for cuda-to-rocm: tile shuffles of pointers and enums need `__shfl` overloads, declared before the CG header

Promoted to cuda-to-rocm/references/build-headers-macros.md in 9d8549ee.

Belongs in `references/build-headers-macros.md` (near "Count cooperative-groups call sites").

CUDA's `thread_block_tile::shfl` takes any trivially copyable type. HIP's tile `shfl` calls `__shfl(var, src, width)`, which exists only for arithmetic types, so `g.shfl(ptr, lane)` or `g.shfl(enum_value, lane)` fails with "no matching function for call to '__shfl'" deep inside `amd_hip_cooperative_groups.h`. Add template overloads (pointer: round-trip through `unsigned long long`; enum: through `std::underlying_type`). Declare them after `hip/hip_runtime.h` and before `hip/hip_cooperative_groups.h`: for a pointer to a fundamental type there is no associated namespace for argument-dependent lookup, so only declarations visible at the template's definition are found. A HIP tile `ballot()` also returns `unsigned long long` (the tile's own bits, compacted), so `const auto v = g.ballot(...); __ffs(v)` becomes ambiguous; store it in `unsigned int` as the CUDA type was. (HierarchicalKV, ROCm 10.2)

## Lesson for moat-porting: `pkill -f <pattern>` on a fleet worker kills the worker's own shell

Promoted to moat-porting/SKILL.md in 9d8549ee.

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

## CUDA compile check 2026-11-24 (mi355 fleet host, real nvcc, no GPU used)

Requested by Jeff Daily (fleet job): CUDA no-regression compile of fork commit `592f800` against the upstream base `451f240` (merge-base of `592f800` and the fork's `master`), before he sees the port. No GPU was used or visible to the job. Nothing changed in the fork, nothing was pushed, and the stage and work lock (linux-gfx942) were not touched. **Verdict: no CUDA regression.** Every target builds at both commits. There are no new errors or warnings, and the sm_80 device code of every common binary is byte-identical, including encodings.

Toolchain: NVIDIA CUDA 12.8 (`nvcc` V12.8.93, cuobjdump 12.8) at `/usr/local/cuda-12.8`, host gcc 13.3.0, CMake 3.31.6, Ninja 1.13.0. The toolkit has no `nvdisasm`, so `cuobjdump -sass` used `NVDISASM_PATH=<venv>/site-packages/triton/backends/nvidia/bin` (nvdisasm 13.1.80). On 12.8 cubins it prints "switching to Old format", which only selects the listing layout. Clones (with `--recurse-submodules`, googletest `0e0d9fe`) and all scratch live under the job directory.

### CMake (upstream's documented configure, plus the research benchmarks)

Configured the same way at both commits. The only differences from the README's `cmake -DCMAKE_BUILD_TYPE=Release -Dsm=80 .. && make -j` are Ninja, used so that warnings are not interleaved, and `-DHKV_BUILD_REPRO_BENCHMARKS=ON`, used so that every test and benchmark target is built:
```
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -Dsm=80 -DHKV_BUILD_REPRO_BENCHMARKS=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
cmake --build build-cuda -j 32 -- -k 0
```
- Base `451f240`: 94/94 steps, rc 0, 275 s. Port `592f800`: 96/96 steps, rc 0, 266 s.
- Targets (`ninja -t targets`): base has 43 executables (19 `*_test` gtest binaries and 24 benchmarks: the 3 default ones plus 21 research ones). Port has the same 43 plus `repeated_key_test`. Nothing was lost or renamed. (Earlier sections count "20 tests"; this CMakeLists has 19 test `add_executable`s at the base.)
- Errors: none at either commit.
- Default-flag warnings: 34 lines at each commit. They are the same multiset once line numbers are dropped, and each one points at the same source statement:
  - five `#69-D integer conversion resulted in truncation`, at `S min_score = MAX_SCORE` and similar lines in `find_or_insert.cuh` and `upsert_and_evict.cuh`. Their line numbers moved because the port added lines above them, for example `find_or_insert.cuh` 914 -> 933 and `upsert_and_evict.cuh` 156 -> 161;
  - three `#177-D` unused variables, in `dual_bucket_test`, `e8_hkv_baseline` and `e9_eviction_comparison`;
  - one gcc `-Wstringop-overflow` in `hkv_hash_sensitivity_benchmark`.
  None is in `update.cuh`, `update_score.cuh`, `update_values.cuh`, `kernel_utils.cuh`, `hip_compat.cuh` or `repeated_key_test.cc.cu`.
- Stricter warnings: every test TU was recompiled at both commits with its exact ninja command plus `-Xcompiler=-Wall,-Wextra` (39 compiles, all rc 0).
  - The 19 common TUs give identical per-TU warning multisets: 1074 lines at each commit, from upstream code such as `-Wsign-compare` in `core_kernels.cuh` and `test_util.cuh`, unused `file`/`line` in `utils.cuh`, and `-Wswitch` in `allocator.cuh`.
  - `repeated_key_test.cc.cu` adds 11 lines. All of them are those same pre-existing upstream header warnings, instantiated once more. None is in the test file or in any `update*.cuh`.

### Device code (cuobjdump SASS, sm_80)

- Full `cuobjdump -sass` text, including the instruction encodings and control words, for all 43 common executables, base against port. Only path-derived symbol hashes were normalized (`_INTERNAL_<hash>`, `__N__<hash>`, the anonymous-namespace `_cc_cu_<hash>_<n>`, which follow the worktree path). Result: **43/43 identical**, 58.1M SASS lines in total, 0 kernels added or removed.
- These binaries contain the kernels the port changed, so they were compared. In `merlin_hashtable_test`, for example, there are 60 `pipeline_update_kernel_with_io`, 12 `pipeline_update_score_kernel` and 40 `pipeline_update_values_kernel_with_io` instantiations. The same binary also has the TLPv1 `upsert`/`find_or_insert`/`upsert_and_evict` kernels (60 each), 144 `update_kernel_with_io`, 288 `find_or_insert_kernel`, and the `accum_or_assign`, `upsert_kernel_with_io_core`, `find_ptr_or_insert` and `prepare_cpu_update_kernel` kernels.
- There is therefore no difference to explain. On CUDA, `kRetryLockedAtCaller` is `constexpr false`, and nvcc folds away every added branch: the `pending_pos` path, the `warp_any` ternaries, the `if (acquired == 0)` after a loop that exits only on `acquired != 0`, and the grid-stride `t -= stride; continue;` retries. The TLPv1 `publish` lambda inlines to the original instruction sequence. The swapped commutative `and.pred` operands that the 2026-11-23 porter saw in PTX at 815cfdc do not reach SASS. `warp_ballot`/`warp_sync` are the original intrinsics with the 32-bit full mask.
- `repeated_key_test` has no base counterpart. Each of its 334 kernels was matched against every base binary: **334/334 equal, instruction for instruction, to an instantiation that a base binary already contains.** These include the 5 `pipeline_update_kernel_with_io`, the `pipeline_update_score_kernel` and the 5 `pipeline_update_values_kernel_with_io` instantiations for `<uint64_t, float, uint64_t>`. Some kernels exist in two variants across base binaries, because per-TU context changes codegen. Four of those (`accum_or_assign_kernel*<...,4,4|32>` and `rehash_kernel_for_fast_mode<...,4>`) differ from the `merlin_hashtable_test` copy but equal the `accum_or_assign_test`/`assign_*_test` copy.

### Bazel CUDA build (closes porter question 2 of 2026-11-23)

It could be set up inside the job directory. bazelisk 1.25.0 (GitHub release binary) fetched Bazel 5.0.0 per `.bazeliskrc` (`BAZELISK_HOME` and `--output_user_root` under the job directory). The CUPTI headers and cuDNN that `cuda_configure` requires came from pip wheels installed with `pip install --no-deps --target <job>/bzdeps "nvidia-cuda-cupti-cu12==12.8.*" "nvidia-cudnn-cu12==9.*"` (12.8.90 and 9.28.1.4). Trees were checked out without googletest, as the README requires for Bazel. Command:
```
bazel --output_user_root=<job>/bazel/root build --config=cuda \
  --action_env CUDA_PATHS=/usr/local/cuda-12.8,<bzdeps>/nvidia/cuda_cupti,<bzdeps>/nvidia/cudnn \
  --action_env CUDA_TOOLKIT_PATH=/usr/local/cuda-12.8 --action_env CUDA_VERSION=12 --action_env CUDNN_VERSION=9 \
  --action_env CUDNN_INSTALL_PATH=<bzdeps>/nvidia/cudnn --action_env CUDA_COMPUTE_CAPABILITIES=8.0 --jobs=32 //...
```
- Base `451f240`: `cuda_configure` succeeds, then the build fails: `include/merlin_hashtable.cuh:38:10: fatal error: merlin/multi_vector.hpp: No such file or directory` while compiling `benchmark/merlin_hashtable_benchmark.cc.cu`. This is the upstream BUILD bug that fork commit `ffdd97b` fixes, now confirmed under CUDA.
- Port `592f800`: 7 targets, "Build completed successfully, 1822 total actions", rc 0, no compiler warnings. The `NEED_ROCM` early return in `cuda_configure` leaves the CUDA path working.
- To compare like with like, the base was rebuilt in a scratch worktree with only `ffdd97b` applied (uncommitted, local): rc 0. The SASS of `//benchmark:benchmark_util` and `libbenchmark_lib.so` is identical to the port's, 379/379 kernels each.

### Nothing to fix

No CUDA break, so there is no fix to propose. The CUDA path of `592f800` is upstream's device code, with the shared Bazel fix on top.
## Porter 2026-11-24 round 3 (linux-gfx942 MI300X, porter role): diagnosis of the `test_find_or_insert_multi_threads` hang

Requested by Jeff Daily (fleet job): diagnose the intermittent hang from "Validation 2026-11-24" before any fix. TheRock ROCm 10.2.0 wheels installed into the job venv (`rocm[libraries,devel,device-gfx942]==10.2.0`, HIP 7.17.26434, rocm-systems pin 43fa692 per `share/therock/therock_manifest.json`), one pinned GPU, `ulimit -c 0`, a timeout around every GPU run. No fork commit this round.

### Verdict

**HIP runtime defect, already filed and already fixed upstream; not caused by this round's update-kernel change.** It is the use-after-free the LichtFeld-Studio port found (deferral `hip-event-query-destroyed-stream-uaf`, filed by Jeff as https://github.com/ROCm/rocm-systems/issues/14755, open). It is fixed on rocm-systems `develop` by https://github.com/ROCm/rocm-systems/pull/15025 (99ad350, merged 2026-11-18). That fix is not in 10.2.0: the 10.2.0 pin 43fa692 is on a branch that left `develop` at 2026-10-14, and the GitHub compare of 99ad350 against 43fa692 reads "diverged". It hangs at the previous head 815cfdc as well, and a 40-line program reproduces it deterministically. Registered `hkv-hip-event-query-destroyed-stream-uaf` (`rocm-bug-report`, status `filed` against #14755). No new issue to draft (see "Issue" below). A port-side workaround is drafted, not committed, pending Jeff's ruling. **The stage stays `porting`:** its only exits are `ported`/`delta-ported` and nothing was ported. The next porter resumes after the ruling (re-stamping the lock; this job has stopped).

### Mechanism (CLR source at the 10.2.0 pin, matched to the shipped binary)

- `hipEventRecord_common` (`projects/clr/hipamd/src/hip_event.cpp`) runs `e->SetCaptureStream(stream)` on every record, under capture or not. Nothing clears it when that stream is destroyed (`Stream::ResetCaptureStateLocked` clears only events recorded during a capture).
- `hipEventQuery` -> `ihipEventQuery` -> `checkEventCaptureRestrictions(event)` reads `captureStream_`, then `s->GetCaptureStatus() == hipStreamCaptureStatusActive && s->IsEventCaptured(event)`; `IsEventCaptured` takes the stream's `std::mutex lock_` and hashes into `captureEvents_`. There is no `hip::isValid(stream)` check there, while `hipEventRecord`, `hipEventDestroy` and `hipStreamWaitEvent` all have one. `hipEventSynchronize` takes the same path.
- Disassembly of `libamdhip64.so.7` (10.2.0): the internal `hipEventQuery` at +0x1883eb calls +0x1878b0 (`checkEventCaptureRestrictions`), which loads `*(event+8)` into r14, compares `0x234(%r14)` with 1 and calls `pthread_mutex_lock(r14+0x1d0)` at +0x1878ed (return +0x1878f2). The bucket walk at +0x18797b (`mov (%rax),%r12`) follows. The exported `hipEventQuery` reaches it through the dispatch table at +0x4a9335 (return `hipEventQuery+0x1b`).
- HierarchicalKV: `MemoryPool::put_raw` records each ready event on the caller's stream (`memory_pool.cuh:569`), and `collect_pending_unsafe` queries pending events later from whichever thread next needs a buffer (`:453`). Every worker in the test destroys its stream at the end, so later queries run on events whose stored stream is freed. Whether a query hangs, crashes or passes depends on what the heap has put in the freed 776-byte `hip::Stream` chunk. This is the "who holds the mutex" answer: **nobody**. The "mutex" is reused heap memory, so its owner field is garbage, and no thread exited holding a lock.

### Step 1: baseline, 30 runs per head, alternating

Conditions, matched to the validator's diagnostic runs (its `diag.sh`/`diag2.sh` ran one 592f800 and one 815cfdc instance at the same time, stall threshold 900 s, the later ones under rocgdb): two lanes ran concurrently on the one GPU. Lane 1 ran 592, 815, 592, ...; lane 2 ran 815, 592, 815, ... So at every moment one instance of each head was running. 15 iterations per lane gives 30 runs per head. Every run was launched under gdb, because `kernel.yama.ptrace_scope=1` forbids attaching later (the validator's `gdb -p` capture failed the same way). Each run: `timeout --kill-after=30 1500 gdb -batch ... -ex run ... --args <bin> --gtest_filter=FindOrInsertPtrTest.test_find_or_insert_multi_threads`, with a capture at 900 s (wchan, maps, SIGINT, then gdb prints every thread's stack and runs a Python script that reads the waiter's mutex, its `__owner`, and the event and stream registers in the caller frame). Builds: 592f800 is the fork clone `projects/HierarchicalKV/src`; 815cfdc is a clean clone in `agent_space/hkv815`. Both use the recipe above (`cmake -DHKV_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942`, `ninja find_or_insert_ptr_test`). Normal run time: 255-300 s.

| Head | Runs | Pass | Hang | Crash in the same HIP frames | Other failure |
|---|---|---|---|---|---|
| 592f800 (this round) | 30 | 29 | 0 | 1 (SIGSEGV at +0x18797b, `L2-592-14`) | 0 |
| 815cfdc (previous head) | 30 | 27 | 1 (`L2-815-13`) | 1 (glibc abort, `L1-815-14`) | 1 (capacity assertion, `L2-815-6`, below) |

Plus 4 plain runs (not under gdb), started before the switch to gdb, 2 per head, all passed. With the validator's runs, the hang-or-crash count is 592f800: 4 in 41, and 815cfdc: 2 in 40. **The defect is present at both heads**, and a difference in the counts would not be significant.

- `L2-815-13`, the hang at 815cfdc, at 900 s: four threads (main joining, two ROCr `kfd_wait_on_events`, one worker in `futex_wait_queue`), the validator's picture exactly. Worker stack: `futex_wait` <- `___pthread_mutex_lock` (pthread_mutex_lock.c:93) <- libamdhip64 `0x7ffff7a538f2` <- `0x7ffff7a543f0` <- `hipEventQuery` `0x7ffff7d7533b` <- `MemoryPool<DeviceAllocator<char>>::collect_pending_unsafe` <- `get_raw` <- `StaticWorkspace` <- `HashTable::find_or_insert` (batch 1). These are the same return addresses as the validator's capture. Mutex `0x7fbe8201aab0`: `__lock` 2, `__count` 32767, `__owner` -341710336, `__nusers` 32767, `__kind` 0. The words are `0x00007fff00000002 0x00007fffeba1ea00 ... 0x000055555750f3a0 0x45 0x00007fffecd385c0`: pointers from another object, not a lock record. The owner is not a live TID. Caller frame: event `0x5555626f54d0`, whose second word (`captureStream_`) is `0x7fbe8201a8e0`, and `0x7fbe8201a8e0 + 0x1d0` is the mutex. The status word at `+0x234` reads 1. The worker's live stream (the `collect_pending_unsafe` argument) is `0x7fbe82024760`, so the event names an earlier, destroyed stream.
- `L1-815-14` (815cfdc): `Fatal glibc error: pthread_mutex_lock.c:450 (__pthread_mutex_lock_full): assertion failed: e != ESRCH || !robust`, SIGABRT, from `libamdhip64 0x7ffff7a538f2` <- `hipEventQuery` <- `collect_pending_unsafe`. The garbage `__kind` sent glibc down the robust-mutex path.
- `L2-592-14` (592f800): SIGSEGV at `libamdhip64 0x7ffff7a5397b` (+0x18797b, the `captureEvents_` bucket walk) <- `hipEventQuery` <- `collect_pending_unsafe` <- `find_or_insert` (batch 1).
- (The loop's `rc` column is gdb's exit status, so outcomes were classified from the logs.)

**Separate, open: `L2-815-6`.** `find_or_insert_ptr_test.cc.cu:3231: Failure`: `table->capacity()` was 67108864 where `MAX_CAPACITY` 134217728 was expected, in the third configuration (16/8/40 threads), 237 s. Not a hang, and not the frames above. It happened once in 30 runs at 815cfdc, never at 592f800, and is absent from the validator's logs. Upstream's auto-rehash (`while (!reach_max_capacity_ && fast_load_factor(n) > max_load_factor) reserve(capacity() * 2)`; `reserve` reads `capacity()` and writes `reach_max_capacity_` before taking its lock) has unlocked check-then-act steps, so one candidate is an upstream race. The other is heap corruption from this defect: when the freed memory reads "capturing" and the bogus event lookup succeeds, `InvalidateCapture` writes into the freed stream. Not diagnosed this round. Recorded so that a recurrence is recognised.

### Step 2: who holds the mutex

Nobody (see Mechanism and `L2-815-13`). `AMD_LOG_LEVEL=4` was not needed: the gdb capture names the lock (the freed stream's `lock_` at `+0x1d0`) and shows its owner field is garbage. The capture was checked against the reproducer first. There, the same script prints `owner=0 lock=2 kind=0`, `event+8 = stream`, `mutex = stream+0x1d0`, `status word 1`.

### Step 3: a small program, run outside the project

`agent_space/uaf/repro.cpp` (container-local; the full source is below). It creates an event, records it on a non-blocking stream, synchronizes, destroys the stream, and mallocs chunks of the stream's `malloc_usable_size` until it gets the freed address back. It fills the chunk with 32-bit 1s (`control`: zeros), then calls `hipEventQuery`. The validator's attempt (record, destroy, query from other threads) ran clean because the next `hipStreamCreate` takes the chunk back with a clean status, which is also what the skill says. The fill makes it deterministic and independent of the object layout.

| Runtime | fill 1s | control (zeros) |
|---|---|---|
| 10.2.0 (job venv) | hang 3/3 (rc 124 under `timeout 30`/`60`) | `hipSuccess` 3/3 |
| nightly 10.3.0a20261123 `rocm-sdk-core` (rocm-systems pin 51e17db, which contains 99ad350; the same object linked with RUNPATH to the extracted wheel, checked with `ldd`) | `hipSuccess` 3/3 | `hipSuccess` 3/3 |

A gdb stack of the 10.2.0 hang has frames identical to the test's: `pthread_mutex_lock` <- `0x7ffff7a538f2` <- `0x7ffff7a543f0` <- `hipEventQuery 0x7ffff7d7533b` <- `main`. The nightly's `hip::Stream` is a different size (792 bytes) and the code no longer has `cmpl $0x1,0x234(...)`, which is why the reproducer fills the whole chunk instead of two words.

```
#include <hip/hip_runtime.h>
#include <malloc.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("%s -> %s\n", #x, hipGetErrorString(e_)); return 1; } } while (0)
int main(int argc, char** argv) {
  const bool control = argc > 1;
  hipEvent_t ev; hipStream_t s;
  CK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
  CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  const size_t chunk = malloc_usable_size((void*)s);
  CK(hipEventRecord(ev, s));
  CK(hipStreamSynchronize(s));
  CK(hipStreamDestroy(s));  // legal: the event outlives its stream
  std::vector<void*> held; void* reused = nullptr;
  for (int i = 0; i < 64 && !reused; ++i) { void* p = malloc(chunk); if (p == (void*)s) reused = p; else held.push_back(p); }
  printf("destroyed stream %p (%zu bytes) reused: %s\n", (void*)s, chunk, reused ? "yes" : "no");
  if (!reused) return 2;
  int* w = static_cast<int*>(reused);
  for (size_t i = 0; i < chunk / sizeof(int); ++i) w[i] = control ? 0 : 1;
  printf("hipEventQuery...\n"); fflush(stdout);
  const hipError_t q = hipEventQuery(ev);  // completed long ago
  printf("hipEventQuery returned %s\n", hipGetErrorString(q));
  return q == hipSuccess ? 0 : 1;
}
```
Build and run: `hipcc -O1 -g repro.cpp -o repro && timeout 60 ./repro; timeout 60 ./repro control`.

### Issue

Nothing new to file. #14755 (Jeff, 2026-10-31, "[HIP] hipEventQuery and hipEventSynchronize read the destroyed recording stream (use-after-free)") is this defect, and #15025 fixes it on `develop`. What is missing is a release that carries the fix (10.2.0 does not). If Jeff wants the second affected project on record, a draft comment for #14755 (not posted; it is his to post):

====
A second project hits this on TheRock 10.2.0 (HIP 7.17.26434, gfx942): NVIDIA-Merlin/HierarchicalKV's memory pool records its ready events on the caller's stream and queries them later from other threads after the caller destroys the stream. Its multi-thread find_or_insert test hangs in `pthread_mutex_lock` <- `hipEventQuery`, or crashes in the same frames, in about 3 of 60 runs. A standalone reproducer (reuse the freed stream chunk, fill it with 32-bit 1s, query) hangs 3/3 on 10.2.0 and returns `hipSuccess` 3/3 with the 10.3.0a20261123 nightly runtime, which contains #15025. Is #15025 going into a 10.2.x patch release?
====

### Proposed port-side workaround (drafted, NOT committed; needs Jeff's ruling)

The pool is the only place in the library that records or queries an event (`grep cudaEventRecord\|cudaEventQuery\|cudaEventSynchronize\|cudaStreamWaitEvent include/`: one record at `memory_pool.cuh:569`, one query at `:453`). So the smallest complete change is at that one record site, HIP build only: after recording the ready event on the caller's stream, make a pool-owned non-blocking stream wait on it and re-record the event there. The event's stored stream is then one that lives as long as the pool, and the event still completes only after the caller's work. Diff (18 added lines, `include/merlin/memory_pool.cuh`; saved as `agent_space/memory_pool_reanchor.diff`, built in the worktree `agent_space/hkvW`):

```
@@ MemoryPool(const MemoryPoolOptions& options, BaseAllocator* allocator)
     pending_.reserve(options_.max_pending);
+
+#ifdef MERLIN_HIP
+    CUDA_CHECK(cudaStreamCreateWithFlags(&event_stream_, cudaStreamNonBlocking));
+#endif
   }
@@ ~MemoryPool()
     deplete_stock();
+
+#ifdef MERLIN_HIP
+    CUDA_CHECK(cudaStreamDestroy(event_stream_));
+#endif
   }
@@ put_raw
         CUDA_CHECK(cudaEventRecord(ready_event, stream));
+#ifdef MERLIN_HIP
+        // HIP's event query reads the stream an event was last recorded on,
+        // even after that stream is destroyed. Move the event onto a stream
+        // the pool owns, ordered after the caller's work.
+        CUDA_CHECK(cudaStreamWaitEvent(event_stream_, ready_event, 0));
+        CUDA_CHECK(cudaEventRecord(ready_event, event_stream_));
+#endif
         pending_.emplace_back(*first, allocation_size, ready_event);
@@ members
   BaseAllocator* allocator_;
+#ifdef MERLIN_HIP
+  cudaStream_t event_stream_{nullptr};
+#endif
```

What it changes and costs:
- **When the pool records events (the behaviour change to rule on):** each returned workspace buffer costs one stream-wait marker and one extra event record on the pool's stream (no kernel), so a buffer is ready a few microseconds later than now. No result changes: buffers go back to stock only after the caller's work, as before. Both records happen under the pool mutex, so no query sees the in-between state. `hipStreamWaitEvent` checks `hip::isValid` on the old stream itself.
- **Resources:** one extra HIP stream per pool. A table has two pools (device and host), so two streams per table, created on the device that is current at construction (the same device as the pool's events, which `hipEventRecord` requires). TheRock 10.1 measured about 8 MiB of host memory per used stream (cuda-to-rocm `faults-memory-streams.md`).
- **CUDA:** unchanged (`#ifdef MERLIN_HIP`).
- **Scope:** covers every event the library holds. Events a user records on streams they destroy are the user's own exposure to the HIP bug. The skill's other fix (a compatibility-header stream pool that parks destroyed streams) does not suit a header-only library: the header is included by users, so it would rewrite their own `cudaStreamDestroy` calls.
- **Evidence so far:** the same re-anchor in the standalone program (`agent_space/uaf/reanchor.cpp`, `REANCHOR=1`: wait plus re-record on a second stream before the destroy) returns `hipSuccess` 3/3 against 3/3 hangs without it on 10.2.0. Built into HierarchicalKV at 592f800 (uncommitted worktree): `merlin_memory_pool_test` 6/6, and `test_find_or_insert_multi_threads` 2/2 run concurrently (268.9 s, 266.8 s; same as unpatched). Two runs cannot show the hang is gone (its rate here is about 1 in 20); a 30-run loop would be the check after a ruling.

Options for Jeff:
- (a) Commit the workaround (one `[ROCm]` commit on `moat-port`, then review and revalidate) and keep it until a ROCm release with #15025 is the baseline.
- (b) No port change. Validate against a runtime that has the fix (a 10.3 nightly now, or the first release carrying #15025). The test hits the defect in about 1 in 20 runs on 10.2.0, so a 10.2.0 validation can pass or fail by luck.
- (c) Both: commit (a) and revisit it when the fix ships.

Porter suggestion: (a). Users on 10.2.0 run this pool in every multi-stream application that destroys streams, and the cost is two API calls per workspace release.

### Lesson for cuda-to-rocm: in a library, re-anchor the event instead of parking streams

Belongs in `references/faults-memory-streams.md`, appended to "Querying an event after its recording stream was destroyed is a use-after-free on HIP".

The stream-pool fix described there fits an application that owns every stream. A header-only or shared library cannot route its users' `cudaStreamDestroy`, and its users include its headers, so a compatibility-header macro would rewrite their calls. When the library's own event holder is a single record site (a workspace or buffer pool that records "ready" events on the caller's stream), fix it at that site instead. Under the HIP build, after `hipEventRecord(ev, callerStream)`, do `hipStreamWaitEvent(ownStream, ev, 0); hipEventRecord(ev, ownStream);` with `ownStream` a non-blocking stream that the holder creates and destroys after its events. The event still completes only after the caller's work, but the stream HIP stores in it now outlives every query. Cost: two API calls per record and one stream per holder. A standalone test that reuses the freed stream chunk (fill with 32-bit 1s) hangs on the query without the re-anchor and returns success with it. The defect is fixed in rocm-systems `develop` by ROCm/rocm-systems#15025 (2026-11-18), after 10.2.0. (HierarchicalKV `MemoryPool`, TheRock 10.2.0, gfx942)

Logs (container-local): `agent_space/base/` (`L<lane>-<head>-<n>.log`, `lane1.out`, `lane2.out`, `L2-815-13.{wchan,maps,smi}`, `plain/`), `agent_space/uaf/`, `agent_space/clr/` (sparse rocm-systems at 43fa692).

## Ruling on the round-3 workaround (Jeff Daily, 2026-11-24)

Option (a). Commit the drafted re-anchor workaround in `MemoryPool` as one `[ROCm]` commit on `moat-port`, then review and revalidate on linux-gfx942, including a 30-run loop of `FindOrInsertPtrTest.test_find_or_insert_multi_threads`. Keep the workaround until the minimum supported ROCm includes ROCm/rocm-systems#15025 (the fix for ROCm/rocm-systems#14755).

## Porter 2026-11-24 round 4 (linux-gfx942 MI300X, porter role): event re-anchor workaround committed

Requested by Jeff Daily (fleet job), implementing his ruling (a) above. One fork commit on `moat-port`: **`4033ed6` "[ROCm] Record memory pool events on a pool-owned stream"** (parent 592f800; no amend, rebase or force; pushed plainly, `git ls-remote` shows 4033ed6). It is the round-3 draft exactly, plus a removal note in the code comment and one line wrapped for the 80-column Google style:
- `MemoryPool` constructor: `cudaStreamCreateWithFlags(&event_stream_, cudaStreamNonBlocking)` after the events are created (same current device as the events).
- `~MemoryPool`: `cudaStreamDestroy(event_stream_)` after `await_pending`, the event destroys and `deplete_stock`.
- `put_raw`, after `cudaEventRecord(ready_event, stream)`, still under `mutex_`: `cudaStreamWaitEvent(event_stream_, ready_event, 0)` and `cudaEventRecord(ready_event, event_stream_)`, with a comment naming ROCm/rocm-systems#14755 and saying it goes once the minimum supported ROCm contains #15025.
- member `cudaStream_t event_stream_{nullptr}`.
All four blocks are under `#ifdef MERLIN_HIP` (defined by `merlin/hip_compat.cuh`, which `memory_pool.cuh` reaches through `debug.hpp`). CUDA is unchanged.

**Files the delta touches (for the CUDA compile recheck): `include/merlin/memory_pool.cuh` only** (20 added lines, nothing removed).

Removal condition registered as deferral `hkv-remove-memory-pool-event-reanchor` (kind `other`).

Toolchain: TheRock ROCm 10.2.0 wheels installed into the job venv (`$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[libraries,devel,device-gfx942]==10.2.0"`); `rocm-sdk path --root` = `<job>/venv/lib/python3.11/site-packages/_rocm_sdk_devel`. Compiler the build invokes (`CMAKE_HIP_COMPILER`, and the first word of every ninja HIP command): `<sdk>/lib/llvm/bin/clang++`, AMD clang 24.0.0git (llvm-project bdc35841c97f), HIP 7.17.26434. Host C++ `/usr/bin/c++`. CMake 3.31.6, Ninja 1.13.0.

Build and checks:
- `cmake -S . -B build-hip -G Ninja -DHKV_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_PREFIX_PATH=$R` then `utils/timeit.sh HierarchicalKV compile -- cmake --build build-hip -j $MAX_JOBS`: 54/54, rc 0. No warning in `memory_pool.cuh`.
- `merlin_memory_pool_test`: 6/6 (0.5 s).
- `find_or_insert_ptr_test --gtest_filter=FindOrInsertPtrTest.test_find_or_insert_multi_threads`: 1/1 pass, 270 s (unpatched 255-300 s).
- `utils/jargon.py --port HierarchicalKV`: clean. Fork tree clean apart from untracked `build-hip/`.

## Review 2026-11-24 (round 3, linux-gfx942 MI300X, reviewer role, delta 592f800...4033ed6)

### Summary
The delta is the HIP-only event re-anchor in `MemoryPool` (`include/merlin/memory_pool.cuh`, +20 lines, one commit). **Verdict: Approve; no findings.** Investigation done inline (the delta is one file), each point checked against the code at 4033ed6:

- **Stream and event lifetime, destruction order.** The stream is created in the constructor after the `max_pending` events (`memory_pool.cuh:350-353`) and destroyed last in the destructor (`:368-370`), after `await_pending()` (which queries every pending event, all of them now recorded on `event_stream_`), the event destroys, and `deplete_stock()`. No event outlives the stream it is recorded on. Every `cudaEventRecord` in the library is `put_raw`'s (`grep` of `include/`), so no event can still name a caller stream once `put_raw` returns. The wait and the re-record (`:585-586`) run under `mutex_` (`:539`), and every `cudaEventQuery` path (`collect_pending_unsafe`, called from `get_raw`, `put_raw` and `await_pending`) also holds `mutex_`, so no query sees the event between the two records. The wait is issued while the caller's stream is still alive (same call), and HIP stores the stream of the last record, so after `:586` the stored stream is `event_stream_`, which the round-3 standalone test confirmed (3/3 success with the re-anchor against 3/3 hangs without).
- **Device for multi-GPU tables.** `HashTable::init` calls `cudaSetDevice(options_.device_id)` (`merlin_hashtable.cuh:1062-1066`) before it constructs both pools (`:1117-1120`), so `event_stream_` is created on the table's device, the same device as the pool's events (created in the same constructor). The re-record therefore pairs an event and a stream of one device, whatever device is current at `put_raw`. A caller stream on another device fails at the pre-existing `cudaEventRecord(ready_event, stream)` first, as before; a cross-device `cudaStreamWaitEvent` would be legal anyway.
- **Host pool.** `MemoryPool<HostAllocator<char>>` is the same template: it gets its own stream on the same device, and the change is identical for it. Two streams per table, as the round-3 cost note says.
- **Copy and move.** `MemoryPool` declares no copy or move members and holds a `std::mutex` and a `const MemoryPoolOptions`, so its implicit copy and move constructors and assignments are deleted; the raw `event_stream_` handle cannot be duplicated or double-destroyed. The table holds the pools through `std::unique_ptr` (`merlin_hashtable.cuh:1117`). `Workspace` objects hold a pool pointer, not the stream.
- **CUDA build.** Every added non-blank line is inside one of four `#ifdef MERLIN_HIP` blocks, and `MERLIN_HIP` is defined only under `__HIP__`/`__HIPCC__`/`__HIP_PLATFORM_AMD__` (`hip_compat.cuh:21-23`). Removing those blocks from the 4033ed6 file gives the 592f800 file plus two blank lines (checked by script). The CUDA token stream is therefore identical and there is no device code in the file, so CUDA device code cannot change. It is not byte-for-byte identical in host code: the two blank lines and the `#ifdef` lines shift the line numbers of the `CUDA_CHECK`s below line 348 (`CUDA_CHECK` passes `__LINE__` to `cuda_check_`, `debug.hpp:48-51`), so CUDA host objects that instantiate the pool differ in those integer constants only. The remote CUDA compile recheck should expect exactly that (SASS identical; host differences limited to `__LINE__` constants).
- **Behaviour note (not a defect, part of the ruled design).** One stream per pool serializes readiness: a buffer returned by a short job is not ready until every earlier-returned buffer's caller work is also done, because each re-record follows the earlier waits on `event_stream_`. It cannot deadlock (no user stream waits on `event_stream_`, and the waited work is already enqueued), and it only delays buffer reuse; the multi-thread case ran in 270 s, inside the unpatched 255-300 s.
- **Commit hygiene.** Title 55 characters with `[ROCm]`, AI assistance disclosed in prose, no trailer, Test Plan with literal commands. `utils/jargon.py --port HierarchicalKV`: clean.

### Recommendation
**Approve.** GPU evidence (the full suite and the 30-run loop) is the validator's next step.

## Validation 2026-11-24 (round 2, linux-gfx942 MI300X, validator role, fork head 4033ed6): PASSED

Requested by Jeff Daily (same fleet job, after the round-3 review passed). Built and run at `4033ed6` (`git -C projects/HierarchicalKV/src rev-parse HEAD`), the build from "Porter 2026-11-24 round 4" (`build-hip`, 54/54 under `timeit.sh compile`). Toolchain as recorded there: TheRock ROCm 10.2.0 wheels in the job venv, builds invoke `<sdk>/lib/llvm/bin/clang++` (AMD clang 24.0.0git, HIP 7.17.26434). `ulimit -c 0`, a timeout around every GPU run, `IS_BLOSSOM_CI` unset.

**Device: one whole MI300X, not a partition.** `HIP_VISIBLE_DEVICES=3` resolves to PCI 0000:65:00 (`hipGetDeviceProperties`: AMD Instinct MI300X, gfx942:sramecc+:xnack-, 304 CUs, 192 GiB). `amd-smi static --partition` reports that GPU (amd-smi GPU 3, BDF 0000:65:00.0) as ACCELERATOR_PARTITION SPX, MEMORY_PARTITION NPS1, and `amd-smi process` showed no other process on it.

### `FindOrInsertPtrTest.test_find_or_insert_multi_threads`, 30 runs: 30 pass, 0 hang, 0 crash, 0 abort

Conditions match the round-3 diagnosis baseline: two lanes running at the same time on the one GPU, 15 runs each, every run launched under gdb (`timeout --kill-after=30 1500 gdb -q -batch -nx -ex run -ex 'info threads' -ex 'thread apply all bt' --args build-hip/find_or_insert_ptr_test --gtest_filter=FindOrInsertPtrTest.test_find_or_insert_multi_threads`), with a stall watchdog at 900 s that would save `/proc/<pid>/task/*/wchan` and maps and SIGINT the test so gdb prints every stack. Nothing else used the GPU during the loop (the suite, benchmarks and Bazel build ran afterwards). Result: all 30 logs show `[  PASSED  ] 1 test.` and gdb's `exited normally`. The watchdog never fired, gdb caught no signal, and no log has `Failure`, `hipError` or `Aborted`. Run time 270-286 s (mean 278.7 s). Baseline for comparison: unpatched, 6 hangs or crashes in 81 runs across both heads (round-3 table and validator runs); here 0 in 30.

### Full suite, per binary under a 3600 s timeout: 20/20 binaries pass, 241 gtest cases

All 20 test executables CMake builds (19 gtest binaries plus `export_batch_if_test`), as built, nothing filtered or resized. Cases and seconds: repeated_key 9 (2), correctness_stress 62 (11), dynamic_max_capacity 1 (3), lock_unlock 1 (1), reserved_keys 1 (1), merlin_memory_pool 6 (0), save_and_load 5 (24), group_lock 4 (21), find_with_missed_keys 7 (19), uint32_score 4 (47), assign_values 3 (79), export_batch_if main (rc 0, prints `done`, 76), dual_bucket 33 (151), find_or_insert_ptr_lock 13 (315), accum_or_assign 14 (313), assign_score 8 (317), insert_and_evict 9 (645), merlin_hashtable 24 (747), find_or_insert 19 (977), find_or_insert_ptr 18 (1097). No `[  FAILED  ]` line. The default benchmarks ran on the GPU during the first part of the suite.

Count correction: the first validation's "all 21 binaries" is one too many. The CMakeLists has 23 executables, 20 tests and 3 benchmarks, and that validation's own list names the same 20 tests (19 with case counts plus `export_batch_if_test`) and the same 241 cases. Nothing is missing.

### Repeated-key tests

`repeated_key_test`: 9/9 in the suite run and again in runs 2 and 3 (2 s, 1 s).

### Default benchmarks (10-minute cap each)

- `dual_bucket_benchmark`: rc 0 in 2 s, "Benchmark Complete".
- `find_with_missed_keys_benchmark`: still printing measurements at the cap (rc 124 from `timeout`), no error, fault or abort line, as in earlier rounds.
- `merlin_hashtable_benchmark`: still producing measurements at the cap. It did not exit within the 30 s grace after SIGTERM, so `timeout` sent SIGKILL (rc 137; earlier rounds got rc 124). It was then in the `dim = 64, capacity = 512 Million-KV, HBM = 32 GB, HMEM = 96 GB` configuration, which sets up 96 GB of host memory; a process inside a large host-memory pin is slow to act on SIGTERM. The log has the earlier tables and no error, fault or abort line.

### ROCm Bazel build

A clean clone at 4033ed6 without googletest, bazelisk 1.25.0 with `BAZELISK_HOME` under the job directory (Bazel 5.0.0 per `.bazeliskrc`), fresh output root: `ROCM_PATH=$R ROCM_AMDGPU_TARGETS=gfx942 bazelisk --output_user_root=<job>/bazel/root build --config=rocm --jobs=$MAX_JOBS //...` under `timeit.sh compile`: "Build completed successfully, 13 total actions", rc 0, 76 s. The ROCm crosstool calls `<sdk>/lib/llvm/bin/clang`. `bazel-bin/benchmark/benchmark_util` ran on the GPU for 300 s (stopped by the cap, rc 124), printing its HBM tables (insert_or_assign 0.438 B-KV/s at LF 0.5, dim 8) with no error lines.

### Other items

- Integrity: `git -C projects/HierarchicalKV/src status --porcelain` shows no modified tracked file (only the untracked `build-hip/`).
- `python3 utils/jargon.py --port HierarchicalKV`: clean. The README's HIP CMake and Bazel sections still apply (the delta changes no build step).
- **CUDA compile gate: not run here; this host has no nvcc. The recheck on another host covers the one file the delta touches: `include/merlin/memory_pool.cuh`.** The review predicts identical SASS, with CUDA host objects differing only in the `__LINE__` constants of the `CUDA_CHECK`s below line 348.
- Host hygiene: my first bazelisk call, made before I set `BAZELISK_HOME`, downloaded Bazel 9.3.3 into the default bazelisk cache in the shared home directory (`~/.cache/bazelisk`). Nothing used it, and it was left in place because that cache may hold other jobs' downloads.

Logs (container-local): `agent_space/val4/` (`L<lane>-<n>.log`, `lane1.out`, `lane2.out`, `suite/`, `bench/`, `bazel.log`, `bazel-bench.log`).

## CUDA compile check at 4033ed6 2026-11-24 (mi355 fleet host, real nvcc, no GPU used)

Requested by Jeff Daily (fleet job). This is a CUDA no-regression compile of fork commit `4033ed6` ("[ROCm] Record memory pool events on a pool-owned stream") against the upstream base `451f240`, which is the merge-base with the fork's `master`. It reuses the method of "CUDA compile check 2026-11-24" (at 592f800) and adds a preprocessor and host-object check of every `memory_pool.cuh` user. No GPU was used or visible to the job. Nothing changed in the fork and nothing was pushed (`moat-port` is still `4033ed6` per `ls-remote`). The stage and the work lock were not touched. **Verdict: no CUDA regression.** All targets build at both commits, with no new errors or warnings. The sm_80 SASS of every binary is identical to the base. The delta changes the CUDA build only through the `__LINE__` values of four `CUDA_CHECK`s in host code, which the linux-gfx942 review predicted.

Toolchain as before: CUDA 12.8 (`nvcc` V12.8.93, cuobjdump 12.8), gcc 13.3.0, CMake 3.31.6, Ninja 1.13.0. `NVDISASM_PATH` pointed at the image venv's triton wheel (`/opt/python-3.11-venv/.../triton/backends/nvidia/bin`, nvdisasm 13.1.80), because the toolkit has no `nvdisasm`. One clone with `--recurse-submodules` (googletest `0e0d9fe`) and detached worktrees at `451f240`, `592f800` and `4033ed6`, all under the job directory. `git diff --stat 592f800 4033ed6`: `include/merlin/memory_pool.cuh` only, +21 lines (four `#ifdef MERLIN_HIP` blocks).

### Preprocessed output of every TU (592f800 vs 4033ed6)

nvcc's own two preprocessing passes were run for each of the 44 CMake TUs. The commands were taken from `nvcc --dryrun --keep` on each TU's exact ninja command: the device pass (`gcc -D__CUDA_ARCH__=800 ... -E`, `.cpp1.ii`) and the host pass (`.cpp4.ii`). Before comparing, line markers and blank lines were dropped and the worktree path was replaced by a constant.
- 41 TUs reach `memory_pool.cuh`: every TU that includes `merlin_hashtable.cuh`, plus `memory_pool_test`. In each of them, both the device and the host output differ in exactly 4 lines, the same 4 lines in all 82 files. After integer literals are masked the output is identical. The 4 lines are the `__LINE__` argument of `CUDA_CHECK` (`cuda_check_(..., __FILE__, __LINE__)`, `debug.hpp:48-51`) below the first inserted block: `cudaEventDestroy` 357->362, `state` in `collect_pending_unsafe` 470->479, `cudaEventRecord(ready_event, stream)` 569->578, `cudaDeviceSynchronize` 574->592. No `event_stream_`, `cudaStreamCreateWithFlags`, `cudaStreamWaitEvent` or new token reaches either CUDA pass.
- The other 3 TUs (`group_lock_test`, `hash_sensitivity_benchmark`, `tbe_like_cache_baseline`) do not include the pool, and their output is byte-identical.

### Compiled output of memory_pool.cuh users (host side, 592f800 vs 4033ed6)

To keep `__FILE__` equal, `memory_pool_test`, `dynamic_max_capacity_test` and `reserved_keys_test` were compiled at each commit through one symlinked path (their exact ninja commands). Every byte difference between the objects is one of the following:
- `objdump -d` of the `MemoryPool` member functions (`~MemoryPool`, `put_raw`, the `collect_pending_unsafe` lambda): only `mov $imm,%edx` immediates, 0x165->0x16a, 0x1d6->0x1df, 0x239->0x242 and 0x23e->0x250. These are the four line numbers above. `reserved_keys_test` instantiates none of them, so its disassembly is identical.
- `.nv_fatbin` and `.rodata` (208, 360 and 315 bytes): these are equal once the 8-hex-digit TU hash in the `_INTERNAL_<hash>_..._cc_cu_<pathhash>` names is swapped (for example `9c53edd7` -> `b680ac92` for memory_pool_test). cudafe derives that hash from the TU content, so the `__LINE__` change moves it. The same hash appears in section and symbol names and in the debug info, which also carries the shifted line table.

### CMake (same configure as the 592f800 check)

```
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -Dsm=80 -DHKV_BUILD_REPRO_BENCHMARKS=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
cmake --build build-cuda -j 25 -- -k 0
```
- Base `451f240`: 94/94 steps, rc 0, 280 s. `4033ed6`: 96/96, rc 0, 265 s. `592f800` (rebuilt for the host comparison): 96/96, rc 0, 281 s.
- Targets: the base has 43 executables, and 4033ed6 has the same 43 plus `repeated_key_test` (`ninja -t targets`). Nothing was lost or renamed.
- Errors: none. Default-flag warnings: 34 lines at each commit, and base and 4033ed6 have the same multiset once line numbers are dropped (the same five `#69-D`, three `#177-D` and one `-Wstringop-overflow` as before). 592f800 and 4033ed6 match exactly, line numbers included. None is in `memory_pool.cuh`, any `update*.cuh`, `kernel_utils.cuh`, `hip_compat.cuh` or `repeated_key_test.cc.cu`.
- Stricter warnings: each of the 20 test TUs was recompiled with `-Xcompiler=-Wall,-Wextra` at 592f800 and at 4033ed6. All 40 compiles returned rc 0, with 1085 warning lines at each commit, and each TU's warnings are identical to the line. The only `memory_pool.cuh` warnings are the pre-existing upstream `-Wunused-parameter 'stream'` at lines 77, 84, 99 and 106, above the change.

### Device code (cuobjdump SASS, sm_80)

- Full `cuobjdump -sass` text (encodings and control words included, only path-derived symbol hashes normalized) for all 43 common executables, base against 4033ed6: **43/43 identical**, 58.1M lines, 22142 kernel bodies, no kernel added or removed. This covers the kernels the port changed (the `pipeline_update*`, TLPv1 `upsert`/`find_or_insert`/`upsert_and_evict` and `update_kernel_with_io` families listed in the 592f800 check). The result is the same as at 592f800 and agrees with the preprocessor result: the delta has no device code.
- `repeated_key_test`, 4033ed6 against 592f800: identical, 334 kernels, 783631 lines. The 592f800 check matched each of those 334 kernels to an instantiation that already exists in a base binary, so this still holds.
- No difference to explain.

### Bazel CUDA build

Same setup as the 592f800 check, rebuilt inside this job directory: bazelisk 1.25.0 fetched Bazel 5.0.0, with `BAZELISK_HOME` and `--output_user_root` under the job directory. The CUPTI and cuDNN wheels were installed with `pip install --no-deps --target <job>/bzdeps "nvidia-cuda-cupti-cu12==12.8.*" "nvidia-cudnn-cu12==9.*"`. Trees were checked out without googletest. The command is the one recorded at 592f800, with `--jobs=25`.
- Base `451f240`: fails with the known upstream BUILD bug, `include/merlin_hashtable.cuh:38:10: fatal error: merlin/multi_vector.hpp: No such file or directory`, which fork commit `ffdd97b` fixes.
- Base with only `ffdd97b` applied (local, uncommitted scratch tree): 7 targets, "Build completed successfully, 1822 total actions", rc 0.
- `4033ed6`: 7 targets, 1822 actions, rc 0, no compiler warnings.
- SASS of `//benchmark:benchmark_util`, `libbenchmark_lib.so`, `libbenchmark_lib.a` and `.pic.a`: identical between base+ffdd97b and 4033ed6, 379 kernels each. For the archives the only difference is cuobjdump's `member <path>` header line.

### Nothing to fix

There is no CUDA break, so no fix is proposed. On CUDA, 4033ed6 is 592f800 with four host-side `__LINE__` constants moved.

Logs and scripts (container-local, job directory `cc/`): `logs/` (`*-build.log`, `*-bazel.log`, `ppcmp.txt`, `sass-base-p4033.txt`), `ppn/*.diff`, `host/`, `scripts/` (`pp.sh`, `ppcmp.sh`, `rawcmp.sh`, `hostsame.sh`, `strict.sh`, `bazel_cuda.sh`).
## Validation 2026-11-25 (linux-gfx1201 Radeon AI PRO R9700, wave32, validator role, fork head 4033ed6): FAILED, two-process hang

Requested by Jeff Daily (fleet job): the port's first wave32 validation, at 4033ed6, following the gfx942 round-2 recipe so the two can be compared. Run 2026-11-24 21:10 to 2026-11-25 03:15.

**Verdict: validation-failed.** Every test binary passes when it has the GPU to itself, but the gfx942 loop condition (two instances of `FindOrInsertPtrTest.test_find_or_insert_multi_threads` at the same time on the one GPU) hangs on this card in 4 of 4 pairs, and the job counts any hang as a failure. The stall is in upstream's group-lock spin kernels, which wait on a kernel in another stream (details below). With `GPU_MAX_HW_QUEUES=2` one pair passed. A proposed change is below; nothing was pushed to the fork.

### Host, device, toolchain

- Device: one AMD Radeon AI PRO R9700 (`HIP_VISIBLE_DEVICES=0` as pinned; `hipGetDeviceProperties`: gfx1201, PCI e3:00 = amd-smi GPU 3, 32 CUs, warpSize 32, 64 KiB LDS per block, 31.86 GiB, `hipDeviceAttributeWallClockRate` 100000 kHz, so `global_timer_ns`'s fixed 100 MHz holds here too). `amd-smi process` showed no other process on any GPU at the start.
- TheRock ROCm 10.2.0 wheels in the job venv: `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[libraries,devel,device-gfx1201]==10.2.0"` (the venv also carries older 10.1.0 `rocm-sdk-device-*` packages for other targets from its base; the gfx1201 one is 10.2.0). `rocm-sdk path --root` = `<job>/venv/lib/python3.11/site-packages/_rocm_sdk_devel`.
- Compiler the CMake build invokes (`CMAKE_HIP_COMPILER`, and the compiler in every ninja HIP rule): `<sdk>/lib/llvm/bin/clang++`, AMD clang 24.0.0git (llvm-project bdc35841c97f), HIP 7.17.26434. Host C++ `/usr/bin/c++`. CMake 3.31.6, Ninja 1.13.0 (venv). The Bazel ROCm crosstool calls `<sdk>/lib/llvm/bin/clang`.

### Build (4033ed6, documented commands)

```
R=$(rocm-sdk path --root); export ROCM_PATH=$R PATH=$R/bin:$PATH
git submodule update --init tests/googletest
cmake -S . -B build-hip -G Ninja -DHKV_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_PREFIX_PATH=$R
bash utils/timeit.sh HierarchicalKV compile -- cmake --build build-hip -j 24
```
54/54, rc 0, 7 min at -j24 (20 tests + 3 benchmarks). Effective flags `-g -O -O3 -std=gnu++17 --offload-arch=gfx1201` (CMake default build type, i.e. Debug with assertions live). Warnings are all upstream shapes: 718 `-Wunused-value` (nodiscard `hipError_t`), 88 `-Wswitch`, 50 `-Wconstant-conversion` (the `S min_score = MAX_SCORE` lines that give nvcc's `#69-D`).

### Full suite, one binary at a time, under gdb with a 3600 s stall capture: 20/20 binaries pass, 241 gtest cases

All 20 test executables as built, nothing filtered or resized, `IS_BLOSSOM_CI` unset, nothing else on the GPU. Seconds (cases): repeated_key 5 (9), correctness_stress 20 (62), dynamic_max_capacity 5 (1), lock_unlock 5 (1), reserved_keys 5 (1), merlin_memory_pool 5 (6), save_and_load 20 (5), group_lock 30 (4), find_with_missed_keys 25 (7), uint32_score 60 (4), assign_values 75 (3), export_batch_if 95 (plain `main`, exit 0, prints `done`), dual_bucket 55 (33), find_or_insert_ptr_lock 415 (13), accum_or_assign 340 (14), assign_score 331 (8), insert_and_evict 430 (9), merlin_hashtable 735 (24), find_or_insert 1081 (19), find_or_insert_ptr 1336 (18). Every process `exited normally`; no `Failure` or `hipError` line. No case needed more memory than the card has: the 64M-key tables and the 16 GB `max_hbm_for_vectors` cases ran as written (the multi-thread case peaks at 16.3-18.3 GB of VRAM).

### Repeated-key tests (same-wavefront contention)

`repeated_key_test` 20 consecutive runs (plus the suite run and three others): 9/9 every time, about 5 s each. On wave32 two copies of the key share one 32-lane wavefront, the case the round-2 fix targets. `AMD_LOG_LEVEL=3` shows `pipeline_update_kernel_with_io`, `pipeline_update_score_kernel` and `pipeline_update_values_kernel_with_io` dispatched 3 times each, as on gfx942.

### `FindOrInsertPtrTest.test_find_or_insert_multi_threads`

| Condition | Runs | Result |
|---|---|---|
| one instance at a time (sizing run, the suite's run, 10 loop runs under gdb with a 1200 s stall capture) | 12 | 12 pass, 335-395 s (loop mean 371 s) |
| two instances at once, the gfx942 recipe (second started 20 s after the first) | 4 pairs | **4 of 4 pairs hang, both processes each time** |
| two instances at once, `GPU_MAX_HW_QUEUES=2` in both, under gdb | 1 pair | both pass, 361 s each |

The hanging pairs: (1) the first loop pair under gdb with the gfx942 900 s watchdog; (2) a pair under gdb with a 3000 s threshold, to rule out "slow" (one instance takes 335-395 s); (3) a plain pair with no debugger, killed by `timeout 1500`; (4) a pair under `AMD_LOG_LEVEL=3`, killed at 1500 s. The 30-run loop was stopped after the first pair, because every pair hangs.

State at the hang (pair 2 at 3000 s, gdb `thread apply all bt`; pair 1 the same):
- GPU `GFX_ACTIVITY` 100 % the whole time, no host progress.
- Process B (43 threads): 20 workers spinning in `group_shared_mutex::lock_read` (host loop on `h_update_count_`), 5 in `update_read_lock` (host CAS on `h_unique_flag_`), one in `lock_update_read`, and the rest waiting on GPU completion: `hipStreamSynchronize` (after a lock kernel), and `hipMemcpyAsync`/`hipStreamSynchronize` from `HashTable::fast_load_factor` inside `assign` (`rocr::core::BusyWaitSignal::WaitRelaxed`).
- Process A: only the main thread left, in `HashTable` destruction (`destroy_table` -> `hipFree` at 900 s, `hipMemcpy` at 3000 s), busy-waiting on a GPU signal. It is starved, not deadlocked on its own.
- VRAM at the hang: 29.6 GB in pair 2 but 19.7 GB in pair 3, so the card's memory is not what runs out.

Which kernels (pair 4, `AMD_LOG_LEVEL=3` dispatch lines with the per-bucket init kernels filtered out): each process stopped dispatching within about 35 s of the second one starting, and its last kernel is one of upstream's single-thread group-lock spin kernels, launched right after the unlock kernel it waits for went to another stream:
- A: `unlock_read_kernel`, `unlock_read_kernel`, then `lock_update_read_kernel` (spins on the device until `read_count <= 1` and `update_count == 0`), then three `__amd_rocclr_copyBuffer`, then nothing.
- B: `pipeline_update_kernel_with_io` + `unlock_update_kernel` on the worker's stream, then `lock_read_kernel` (spins until `update_count == 0`), then nothing.

Mechanism (my reading; the port does not change these files): `group_shared_mutex` (`include/merlin/group_lock.cuh`) admits a holder with host-side counters, then launches a one-thread lock kernel on a per-thread lock stream (`get_lock_stream()`) and synchronizes it. The previous holder released by enqueuing an `unlock_*_kernel` on its own stream, behind its work, and decremented the host counter at once. So the device-side spin in `lock_*_kernel` (`core_kernels/group_lock_kernels.cuh`) waits for a kernel queued on another stream. This only completes if the two streams' hardware queues both make progress. HIP maps a process's streams onto `GPU_MAX_HW_QUEUES` (default 4) hardware queues. With two such processes on this card the queues are oversubscribed, and the queue holding the unlock kernel never runs again while the spinner occupies the GPU (GPU at 100 %, no progress for 50 minutes). One process alone, or two processes with 2 queues each, passed. MI300X ran the same two-lane loop 30/30 at this head; it schedules more hardware queues. CUDA does not guarantee concurrent progress of independent streams either, so this is an upstream forward-progress assumption that this card's queue scheduling exposes. Whether that scheduling is itself a defect (a spinning queue starving a runnable queue indefinitely instead of time-slicing) is a question for a person, below.

Side effect that contaminated two diagnostic runs: after a hung pair is killed, the GPU stays wedged for a while. New processes stall in their first cache-invalidate packet at code-object load (`rocr::AMD::AqlQueue::ExecutePM4` <- `GpuAgent::InvalidateCodeCaches` <- `BlitKernel::Initialize` <- `Executable::Freeze`), with no test thread started. This happened to a pair launched under rocgdb right after pair 2 (so its `info dispatches` = "No dispatches are currently active" says nothing about the hang) and to a second `GPU_MAX_HW_QUEUES=2` pair right after pair 4. It cleared within about 20 minutes the first time. After the last kill (03:11) it was still present at 03:15: `repeated_key_test` stalled in its first case under `timeout 120` twice, and `amd-smi process -g 3` listed one process holding about 682 MB of VRAM under a PID that does not exist on the host (3227120, then 3236139 at the next check). A host or driver owner should check GPU 3 (PCI e3:00) before the next job uses it.

### Proposed change (not made; needs a person's ruling)

The port must not hang where CUDA returns. Options:
- (a) HIP-only, in `group_shared_mutex`: make each `unlock_read`/`unlock_update`/`unlock_update_read` wait for its unlock kernel (`cudaStreamSynchronize(stream)`, or record an event the next lock waits on) before decrementing the host counters. The host then admits the next holder only after the device counters are released, so the lock kernels never wait on a kernel in another queue. Cost: an unlock blocks the caller's thread until its own stream's preceding work finishes, which serializes holders more than CUDA does. CUDA unchanged under `#ifdef MERLIN_HIP`. It should be re-run on this card with the two-lane loop and on gfx942.
- (b) Treat it as a ROCm queue-scheduling defect and file it with the evidence above (one process passes, two hang, `GPU_MAX_HW_QUEUES=2` passes), plus the post-kill wedge. This leaves users on this card exposed until a fix ships.
- (c) Both.
Setting `GPU_MAX_HW_QUEUES` in the tests is not a fix: users run the library in their own processes.

### Other items

- ROCm Bazel build: clean clone at 4033ed6 without googletest, bazelisk 1.25.0 (release binary, sha256 fd8fdff4...), `BAZELISK_HOME` and `--output_user_root` under the job directory, Bazel 5.0.0 per `.bazeliskrc`: `ROCM_PATH=$R ROCM_AMDGPU_TARGETS=gfx1201 bazelisk --output_user_root=<job>/bazel/root build --config=rocm --jobs=24 //...` under `timeit.sh compile`: "Build completed successfully, 13 total actions", rc 0, 55 s. `benchmark_util` embeds only `amdgcn-amd-amdhsa--gfx1201`. It ran 300 s on the GPU (cap, rc 124), printing its tables ("GPU: 1 x AMD Radeon AI PRO R9700"; dim 8 at LF 0.50: insert_or_assign 0.156, find 0.378, find_or_insert 0.155, assign 0.306 B-KV/s), no error lines. (My first bazelisk call ran from the wrong directory through `timeit.sh` and downloaded Bazel 9.3.3 into the job's `BAZELISK_HOME`; nothing used it.)
- Default benchmarks, 10-minute cap each, alone on the GPU: `dual_bucket_benchmark` rc 0 in 2 s ("Benchmark Complete", final LF 0.9973); `find_with_missed_keys_benchmark` and `merlin_hashtable_benchmark` still printing tables at the cap (rc 124), no error, fault or abort line.
- Integrity: `git -C projects/HierarchicalKV/src status --porcelain` shows only untracked `build-hip/`; HEAD 4033ed6f9d60. `python3 utils/jargon.py --port HierarchicalKV`: clean (after fetching `master` into the clone). README documents the HIP CMake and Bazel builds.
- **CUDA compile gate: not run here; this host has no CUDA toolkit.** No CUDA compile at 4033ed6 is recorded yet; it covers `include/merlin/memory_pool.cuh` (the round-3 review predicts identical SASS and host objects differing only in `__LINE__` constants).

Logs (container-local, not committed): `agent_space/val/` (`build.log`, `bazel.log`, `suite/`, `rk/`, `single/`, `loop900/` (pair 1 and the VRAM samples `mem.txt`), `conc/` (pair 2), `diag/` (pair 3 `plain-*`, `q2-*`), `lp/` (pair 4 dispatch lines), `rg/` (the wedged rocgdb pair), `bench/`).

## Ruling on the two-process hang (Jeff Daily, 2026-11-25)

Option (c) of "Validation 2026-11-25 (linux-gfx1201 ...)": both a HIP-only workaround in the library (in `group_shared_mutex`, so a lock kernel never spins on an unlock kernel queued in another queue) and a ROCm issue draft for Jeff. Jeff expects the workaround may cost a lot of performance: measure it, do not assume; the measurement decides whether it ships. Porter round on linux-gfx1201 (Radeon AI PRO R9700), confirming the cause first with more than one run.

## Porter 2026-11-25 (linux-gfx1201 Radeon AI PRO R9700, porter role): two-process hang; GPU wedged after the 5th unmodified pair, round INCOMPLETE

Requested by Jeff Daily (fleet job), implementing his ruling (c) above. **Stopped early: the pinned GPU stayed wedged for more than two hours after the fifth hung pair was stopped, so all GPU work stopped, as the job instructed.** Done: 5 unmodified pairs (5 of 5 hang), queue counts per process, two standalone reproducers (no hang), the workaround (written, built 54/54, committed in the job's fork clone as `2fc61db`, **not pushed** because it has no GPU run behind it), the HIP runtime source check, the ROCm issue draft. Not done (each needs a working GPU): the `GPU_MAX_HW_QUEUES` pairs, the single-process runs, the diagnostic dispatch pair, the workaround's 30 pairs and full test set, and every cost measurement. The stage stays `porting` with the lock stamped to linux-gfx1201; this job has stopped, so the next porter re-stamps it.

The prompt calls the hanging loop "the two-process `repeated_key_test` loop". The loop the gfx1201 validation ran, and this round ran, is two concurrent `find_or_insert_ptr_test --gtest_filter=FindOrInsertPtrTest.test_find_or_insert_multi_threads`; `repeated_key_test` passes (validation section above).

### Host, device, toolchain

- `HIP_VISIBLE_DEVICES=2` as pinned resolves to PCI 03:00 (`hipGetDeviceProperties`: AMD Radeon AI PRO R9700, gfx1201, 32 CUs, warpSize 32) = amd-smi GPU 0 = DRM card3 = KFD gpu_id 7727. **This is a different card from the gfx1201 validation's (PCI e3:00).** No other process on any GPU at the start.
- TheRock ROCm 10.2.0 wheels installed into the job venv: `$FLEET_VENV/bin/python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[libraries,devel,device-gfx1201]==10.2.0"` (pip warns that the venv's older 10.1.0 `rocm-sdk-device-*` packages for other targets require 10.1.0 libraries; nothing here uses them). `rocm-sdk path --root` = `<job>/venv/lib/python3.11/site-packages/_rocm_sdk_devel`. rocm-systems pin 43fa692 (`share/therock/therock_manifest.json`).
- Compiler the builds invoke: `CMAKE_HIP_COMPILER` = `<sdk>/lib/llvm/bin/clang++`, AMD clang 24.0.0git (llvm-project bdc35841c97f), HIP 7.17.26434. CMake 3.31.6 and Ninja 1.13.0 from the venv. CMake default build type (empty, so `-O3 -g` with assertions live, as the validation).
- Kernel 6.11.0-26-generic, amdgpu DKMS 6.19.14.31400100, `sched_policy=0`, `cwsr_enable=1`, `hws_max_conc_proc=-1`, `queue_preemption_timeout_ms=9000`; MES firmware 0x91, MEC 0xd3e; KFD topology for this GPU: `num_cp_queues 4`, `num_sdma_engines 2`, `num_sdma_queues_per_engine 6`. No debugfs and no `dmesg` access in the container.
- Builds (each `timeit.sh ... compile`, rc 0, 54/54): the unmodified 4033ed6 in a detached worktree `agent_space/hkv-base` (459 s), and 2fc61db in `projects/HierarchicalKV/src/build-hip` (482 s at nice 19). Configure: `cmake -S . -B build-hip -G Ninja -DHKV_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_PREFIX_PATH=$R`. `concurrency_benchmark` (an opt-in research benchmark, `-DHKV_BUILD_REPRO_BENCHMARKS=ON`, built as the single target into `build-repro`) for both trees, for the cost measurement that did not run.

### Step 1: is the cause confirmed (partially done)

Each run: `timeout -s TERM -k 60 1200 <build>/find_or_insert_ptr_test --gtest_filter=FindOrInsertPtrTest.test_find_or_insert_multi_threads`, the second process started 20 s after the first, exactly two processes on the card, `ulimit -c 0`, and after every pair a trivial-kernel check (`devq`: device properties, one `hipLaunchKernelGGL`, read back) under `timeout -s TERM -k 30 60`, retried every 2 minutes.

| Condition | Runs | Result |
|---|---|---|
| unmodified 4033ed6, two processes | 5 pairs | **5 of 5 pairs hang, both processes every time**: no gtest result, each process stopped by `timeout` at 1200-1203 s (rc 124, SIGTERM was enough). One instance alone takes 335-395 s (validation). The check after pairs 1-4 passed at the first try; **after pair 5 the GPU stayed wedged (below)** |
| `GPU_MAX_HW_QUEUES=2`, two processes | 0 of 5 pairs | not run (wedge) |
| `GPU_MAX_HW_QUEUES=3`, two processes | 0 of 5 pairs | not run (wedge) |
| one process alone | 0 of 5 runs | not run (wedge); the validation has 12 of 12 passes on the other card |
| unmodified, `AMD_LOG_LEVEL=3`, kernel dispatch lines kept | 0 of 1 pair | not run (wedge); the validation's one traced pair is the only dispatch evidence |

With the validation's runs: unmodified two-process pairs hang 9 of 9 on two R9700 cards. Whether the hang is the lock kernels (the validation's dispatch trace) was not re-traced here, and the workaround, which would test the mechanism, never ran.

**Queues per process.** Sampled 60 s after the first process started from `/sys/class/kfd/kfd/proc/<host pid>/queues/*/{type,gpuid}` (host PIDs; the container has its own PID namespace): every test process holds **6 queues on this GPU, 5 compute (type 0) and 1 SDMA (type 1)**, in 9 of the 10 samples (pair 4's second process had 2 compute queues at that moment, still starting up). Five compute queues is HIP's `GPU_MAX_HW_QUEUES` default of 4 plus ROCr's utility queue. **Two processes therefore hold 10 compute queues against the 4 that `num_cp_queues` reports for this GPU; one process alone already holds 5.** The standalone programs below hold 5 compute queues each. `stats_7727/evicted_ms` (a constant that differs by process start time) and `counters_7727/page_in` did not change during a run, so they say nothing about KFD evictions here.

**Standalone reproducers: no hang yet.** Two programs, built with `<sdk>/bin/hipcc -O2 --offload-arch=gfx1201`, each run as two processes started 1 s apart (container-local `agent_space/hang/`):
- `spinrepro <iters> <pairs> <work_us> <blocks>`: per pair of streams, a busy kernel (wall-clock spin, 256 blocks of 64) then a one-thread release kernel on stream B, then a one-thread kernel on stream A that spins until the flag is set, then `hipStreamSynchronize(A)`. 1 stream pair: 20000 iterations in 15.35 s and 15.28 s. 4 stream pairs (8 streams): 20000 iterations in 86.12 s and 85.96 s. Both pass.
- `lockrepro <threads> <iters> <work_us> <blocks>`: HierarchicalKV's lock pattern. Each host thread owns a work stream and a lock stream (`hipStreamCreate`, as `get_lock_stream()`), takes a host flag, launches a one-thread CAS-spin `acquire` kernel on its lock stream and synchronizes it, queues a busy kernel plus a one-thread `release` kernel on its work stream, and clears the host flag without waiting. 16 threads x 3000: 48000 acquisitions in 41.43 s and 41.20 s. 64 threads x 600: 38400 in 57.62 s and 57.98 s. Both pass.
- Variants that add what the test does beside the lock, an allocate/copy/free cycle (`CHURN=<n>`: `hipMalloc`, `hipHostMalloc`, two synchronous `hipMemcpy`, two frees every n iterations) and a small device-to-pageable copy synchronized inside the critical section (`PAGEABLE=<n>`, as `HashTable::fast_load_factor` does), are built and were not run (wedge). The next things to try, in that order.

### The wedge after pair 5

Pair 5 was stopped at 20:43 UTC (rc 124, both processes exited). From then on, every new HIP process on this GPU stalled; device properties still read normally:
- the check program stalled in its first packet: gdb (launched under gdb because `ptrace_scope=1`) shows the main thread in `rocr::core::InterruptSignal::WaitRelaxed` <- `hsa_signal_wait_scacquire` <- `AqlQueue::ExecutePM4` <- `GpuAgent::InvalidateCodeCaches` <- `BlitKernel::Initialize` <- `GpuAgent::DmaCopy` <- `RegionMemory::Freeze` <- `ExecutableImpl::Freeze` <- ... <- `hipLaunchKernel` <- `main`, and the two ROCr event threads in `hsaKmtWaitOnMultipleEvents`. The validation's wedge had the same frames.
- `amd-smi` at 20:55, with only that stalled check process on the GPU (1 queue, 171 MB of VRAM): `GFX_ACTIVITY` 84 %, socket power 138 W. A minute later 3 %, 15 W, and from 21:00 on 0-3 %, 1-14 W.
- Checks at 21:06, 21:08 to 21:35 every 3 minutes, one with a 540 s cap (21:44, never got past the first packet), 21:45 to 22:45 every 5 minutes, 22:47 and 22:51: every one timed out. **Still wedged 2 h 08 min after the kill; GPU work stopped.** The other four hung pairs on this card did not wedge it (each check passed within seconds). On the validation's card (e3:00) a wedge cleared within about 20 minutes once, and was still present when that job ended the other time.
- **A host or driver owner needs to check GPU PCI 0000:03:00.0 (amd-smi GPU 0, card3, KFD gpu_id 7727) before the next job uses it.** Nothing here can reset it (no root).

Log of the probes: `agent_space/hang/real/wedge-base5.txt`, gdb capture `wedge-gdb.txt` (container-local).

### Existing ROCm issues (read-only search)

`gh search issues` on ROCm/rocm-systems for oversubscription, `GPU_MAX_HW_QUEUES`, spin, starvation and cross-stream terms found no report of this hang. The nearest is **ROCm/rocm-systems#15850** (open, unlabeled, 2026-11-14): on an R9700 with `num_cp_queues 4` under MES, a HIP process with 5 compute queues stalls for good when a KFD evict/restore leaves a queue with pending packets unmapped and no further doorbell comes; `GPU_MAX_HW_QUEUES=3` (4 queues on 4 slots) cured it there. Same card, same queue oversubscription, a different trigger (a full queue and evictions). Also seen: #14856 (a 4th hardware queue starting late on MI300A/MI250X) and #11678 (a host wait inside the execution lock).

### Step 2: the workaround (written and built; NOT GPU-tested, NOT pushed)

Fork commit **`2fc61db` "[ROCm] Wait for released group locks on the host before locking"**, parent 4033ed6, in this job's clone only (`git ls-remote` still shows `moat-port` at 4033ed6). It was not pushed because nothing has run it on a GPU (validate before pushing), and because Jeff ruled that the cost measurement decides whether it ships. The diff below is the durable copy; the clone is wiped with the job.

The narrower form, chosen over "each unlock synchronizes its stream before decrementing the host counters":
- Each `unlock_*` still queues its unlock kernel on the caller's stream and returns without blocking. Under the HIP guard it then also records one mutex-owned event on that stream and makes a mutex-owned non-blocking **release stream** wait on it, under a `std::mutex`: `read_release_stream_` for `unlock_read`, `update_release_stream_` for `unlock_update`, both for `unlock_update_read`. This happens before the host counter is decremented.
- Each `lock_*`, once admitted by the host counters and before it launches its lock kernel, synchronizes on the host the release stream for the counts its kernel checks: `lock_read` the update stream (its kernel waits for `update_count == 0`), `lock_update` the read stream (`read_count == 0`), `lock_update_read` both. The lock kernel then finds the counters already released and never spins on a kernel in another queue.
- Why it is still correct: the lock kernel used to wait for exactly the unlock kernels of the conflicting holders that released before this locker was admitted. Admission needs the host decrement, which comes after that unlock's event and stream wait (a release decrement, read with acquire or seq_cst), so every such unlock is already ordered on the release stream when the locker synchronizes it. The device kernels and counters are unchanged, so mutual exclusion is still enforced on the device exactly as before. The only change is that the host waits for the release first.
- Why it should cost little (to be measured): the locker already blocked on the host until its lock kernel finished, and that kernel could not finish before those same unlock kernels had run. So the locker waits about as long as before. Unlocks stay asynchronous for the caller, which the stricter form would not. The extra work is one event record plus one or two stream waits per unlock, under a mutex shared by all unlockers of one table, and one or two (usually already complete) stream synchronizations per lock. A single release stream per count serializes the release markers, but a locker only waits on the count it conflicts with, and the lock already orders those holders.
- No exposure to ROCm/rocm-systems#14755: the event is only recorded and waited on while the caller's stream is alive, and is never queried or synchronized; only the mutex's own streams are synchronized. In the HIP source at the 10.2.0 pin (rocm-systems 43fa692, `projects/clr/hipamd/src`), `hipEventRecord_common` checks `hip::isValid` on the event's previous stream before touching it (`hip_event.cpp:452-456`), and `hipStreamWaitEvent_common` returns early when the event's stream is no longer valid (`hip_stream.cpp:570-575`). A stream-wait goes through `MemoryPool`'s `AddSafeStream` bookkeeping when both streams are on one device (`hip_stream.cpp:634-637`); it is O(allocations in the stream-ordered pool's free heap), and the library uses stream-ordered allocation only in `allocator.cuh` (`cudaMallocAsync`/`cudaFreeAsync`) and one `cudaFreeAsync` in `utils.cuh`. The measurement will show whether that matters.
- What it relies on instead: an event wait on another stream completes once the recorded work has run. That is the documented stream and event contract, which upstream's spin kernel does not rely on: it assumes that another stream keeps running while the spinning kernel occupies its own queue, which neither CUDA nor HIP promises. If MES can also leave the unlocking queue unmapped behind a queue whose head is a cross-queue barrier, this form could still stall, which only the 30-pair loop can show. The stricter form (`cudaStreamSynchronize(stream)` in each unlock before the host decrement) needs no cross-queue wait at all and is the fallback.
- Resources per table (one `group_shared_mutex`): two non-blocking streams and one event, created in the constructor on the device current then (the device of the mutex's counters), destroyed in the destructor after its `cudaDeviceSynchronize`.

**Files the commit touches that a CUDA build reads: `include/merlin/group_lock.cuh` only** (+55 lines, nothing removed). Every added line is inside a `#ifdef MERLIN_HIP` block (`MERLIN_HIP` comes from `merlin/hip_compat.cuh` through `core_kernels/group_lock_kernels.cuh`), and there is no device code in the additions. Expected on nvcc: identical SASS. Host objects should differ only in the `__LINE__` immediates of the `CUDA_CHECK`s in `group_lock.cuh` below the first inserted block (`debug.hpp:48-51` passes `__LINE__`), in every TU that includes `merlin_hashtable.cuh` or `group_lock.cuh` (`group_lock_test` too).

```
diff --git a/include/merlin/group_lock.cuh b/include/merlin/group_lock.cuh
@@ group_shared_mutex() noexcept
     CUDA_CHECK(cudaDeviceSynchronize());
+#ifdef MERLIN_HIP
+    CUDA_CHECK(
+        cudaEventCreateWithFlags(&release_event_, cudaEventDisableTiming));
+    CUDA_CHECK(cudaStreamCreateWithFlags(&read_release_stream_,
+                                         cudaStreamNonBlocking));
+    CUDA_CHECK(cudaStreamCreateWithFlags(&update_release_stream_,
+                                         cudaStreamNonBlocking));
+#endif
   }
@@ ~group_shared_mutex() noexcept
     CUDA_CHECK(cudaDeviceSynchronize());
+#ifdef MERLIN_HIP
+    CUDA_CHECK(cudaStreamDestroy(update_release_stream_));
+    CUDA_CHECK(cudaStreamDestroy(read_release_stream_));
+    CUDA_CHECK(cudaEventDestroy(release_event_));
+#endif
     CUDA_CHECK(cudaFree(d_update_count_));
@@ lock_read()
       if (h_update_count_.load(std::memory_order_seq_cst) == 0) {
         {
+#ifdef MERLIN_HIP
+          CUDA_CHECK(cudaStreamSynchronize(update_release_stream_));
+#endif
           cudaStream_t stream = get_lock_stream();
@@ unlock_read(cudaStream_t stream)
     { group_lock::unlock_read_kernel<<<1, 1, 0, stream>>>(d_read_count_); }
+#ifdef MERLIN_HIP
+    mark_released(stream, read_release_stream_);
+#endif
     h_read_count_.fetch_sub(1, std::memory_order_release);
@@ lock_update()
       if (h_read_count_.load(std::memory_order_seq_cst) == 0) {
         {
+#ifdef MERLIN_HIP
+          CUDA_CHECK(cudaStreamSynchronize(read_release_stream_));
+#endif
           cudaStream_t stream = get_lock_stream();
@@ unlock_update(cudaStream_t stream)
     { group_lock::unlock_update_kernel<<<1, 1, 0, stream>>>(d_update_count_); }
+#ifdef MERLIN_HIP
+    mark_released(stream, update_release_stream_);
+#endif
     h_update_count_.fetch_sub(1, std::memory_order_release);
@@ lock_update_read(), before the lock kernel
     {
+#ifdef MERLIN_HIP
+      CUDA_CHECK(cudaStreamSynchronize(read_release_stream_));
+      CUDA_CHECK(cudaStreamSynchronize(update_release_stream_));
+#endif
       cudaStream_t stream = get_lock_stream();
@@ unlock_update_read(cudaStream_t stream), after the unlock kernel
+#ifdef MERLIN_HIP
+    mark_released(stream, read_release_stream_, update_release_stream_);
+#endif
     h_read_count_.fetch_sub(1, std::memory_order_release);
@@ private:
+#ifdef MERLIN_HIP
+  // A lock kernel spins until earlier unlock kernels, queued on the
+  // holders' streams, have run. HIP gives no guarantee that another
+  // stream's queue runs while the spinning kernel occupies its own, and
+  // with several processes on one GPU it may not. So each unlock is also
+  // ordered onto a stream owned here, and the next conflicting locker
+  // waits for that stream on the host before it launches its kernel,
+  // which then finds the counters already released.
+  void mark_released(cudaStream_t stream, cudaStream_t release_stream,
+                     cudaStream_t other_release_stream = nullptr) {
+    std::lock_guard<std::mutex> guard(release_mutex_);
+    CUDA_CHECK(cudaEventRecord(release_event_, stream));
+    CUDA_CHECK(cudaStreamWaitEvent(release_stream, release_event_, 0));
+    if (other_release_stream != nullptr) {
+      CUDA_CHECK(cudaStreamWaitEvent(other_release_stream, release_event_, 0));
+    }
+  }
+
+  std::mutex release_mutex_;
+  cudaEvent_t release_event_;
+  cudaStream_t read_release_stream_;
+  cudaStream_t update_release_stream_;
+#endif
   std::atomic<int> h_update_count_;
```

The local commit's message (draft; its second paragraph's mechanism sentence and its Test Plan describe runs that have not happened, and must be rewritten from the real results before any push): title as above; body: the forward-progress assumption of the lock kernels, the 5-of-5 two-process hang on the R9700 with 5 compute queues per process against 4, the release-stream design and why the wait covers exactly what the lock kernel waited for, "CUDA is unchanged: every addition is under the HIP guard", the AI-assistance sentence, and a Test Plan (configure, build, the 30-pair loop, every test binary, the benchmarks before and after). `python3 utils/jargon.py --port HierarchicalKV` on the local branch: clean.

### Step 3: cost measurement (not run); what the gfx1201 rerun and the gfx942 follow-up must measure

Unmodified 4033ed6 against the workaround, same build flags, one process at a time on the GPU, nothing else on it, **at least 3 runs per build, interleaved (base, workaround, base, ...)**, per benchmark the change in percent of each reported throughput with the run-to-run spread (min-max or standard deviation of the 3):
1. `dual_bucket_benchmark` (default CMake target), to completion (about 2 s), cap 300 s.
2. `find_with_missed_keys_benchmark` (default target), fixed cap 420 s; compare only the table rows that every run finished.
3. `merlin_hashtable_benchmark` (default target; the same source as the Bazel `//benchmark:benchmark_util` earlier validations ran, `benchmark/BUILD`), fixed cap 420 s, rows finished in every run. It calls the locked APIs from one thread, so it shows the per-call overhead.
4. `concurrency_benchmark triple_group` (research target, `-DHKV_BUILD_REPRO_BENCHMARKS=ON`, `cmake --build <dir> --target concurrency_benchmark`), cap 900 s: multi-threaded find/assign/insert mixes and an assign-only thread sweep (1, 2, 5, 10 threads) on one table, the most lock-sensitive workload; compare `throughput_bkvs` per workload line of its CSV.
5. Test-suite runtime: every one of the 20 test binaries once per build, one at a time, nothing filtered, per-binary seconds and the total; the multi-thread binaries (`find_or_insert_ptr_lock_test`, `find_or_insert_ptr_test`, `group_lock_test`, `merlin_hashtable_test`) matter most.
Scripts for all of this are written (`agent_space/bench.sh <tree> <out>`, `agent_space/suite.sh <build dir> <out>`, container-local; the commands above are complete without them). On gfx942 the same list applies, plus its 30-run two-lane loop of `test_find_or_insert_multi_threads` (the 4033ed6 baseline there is 270-286 s per run), to show the workaround does not slow the case it does not need.

### Steps 4 and 5

- Step 4 (30 clean pairs and the full test set with the workaround): not run (wedge). The chain for it is `agent_space/hang/chain2.sh` (5 single-process runs, then 30 pairs of the workaround build, stopping on a failed check).
- Step 5: the issue draft is below.

## Issue draft for Jeff

Target **ROCm/rocm-systems**, labels **`bug`** and **`project: clr`** (looked up with `gh label list`; the queue count per process is HIP's `GPU_MAX_HW_QUEUES` default plus ROCr's utility queue, so `project: rocr-runtime` is the other candidate; the scheduling itself is MES firmware and amdgpu, which have no label in this repository). Searched first (above): no existing report; #15850 is the nearest and is linked. **Not posted.** Two things for Jeff to decide: whether to post before the standalone program reproduces it (so far only the HierarchicalKV test does; the `CHURN`/`PAGEABLE` variants are the next try), and whether it belongs as a comment on #15850 instead of a new issue.

Title: `[clr] gfx1201: two HIP processes hang on a cross-stream spin wait`

====
### Problem

On a Radeon AI PRO R9700 (gfx1201, MES), two HIP processes on the same GPU hang for good when each runs a single-thread kernel that spins on a device counter until a kernel queued on another stream of the same process decrements it. Each process alone completes. After the hung processes are killed, the GPU sometimes stays unusable: a new process stalls in the first packet it submits.

### System

- Radeon AI PRO R9700 32 GB (gfx1201), `num_cp_queues 4` in the KFD topology, amdgpu `sched_policy=0`, `cwsr_enable=1`, MES firmware 0x91
- Linux 6.11.0-26-generic, amdgpu DKMS 6.19.14.31400100
- TheRock ROCm 10.2.0 wheels (HIP 7.17.26434, rocm-systems 43fa692)

### Reproduction

NVIDIA-Merlin/HierarchicalKV, HIP build (`cmake -DHKV_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201`), two instances of one test on one GPU, the second started 20 s after the first:

```
./find_or_insert_ptr_test --gtest_filter=FindOrInsertPtrTest.test_find_or_insert_multi_threads
```

The test runs 64 host threads on one table. Its reader/writer lock admits a holder on the host, then launches a one-thread kernel on a per-thread stream that spins until the device counters allow it, and synchronizes that stream. The previous holder releases by queuing a one-thread unlock kernel on its own stream, behind its work.

- Two processes: 9 of 9 pairs hung, both processes every time, on two R9700 cards (left for 900 to 3000 s before being stopped; one instance alone takes 335 to 395 s). The GPU reads 100 % busy with no host progress.
- One process at a time: 12 of 12 runs passed.
- `GPU_MAX_HW_QUEUES=2` in both processes: 1 of 1 pair passed.
- `AMD_LOG_LEVEL=3`, one traced pair: in each process the last kernel dispatched is one of the spinning lock kernels, launched right after the unlock kernel it waits for was queued on another stream.
- Each process holds 6 KFD queues on the GPU (5 compute, 1 SDMA, from `/sys/class/kfd/kfd/proc/<pid>/queues`), so two processes hold 10 compute queues against 4 slots.

A standalone program with the same pattern (two processes; in each, host threads queue a busy kernel and a one-thread release kernel on one stream, then launch a one-thread kernel on a second stream that spins until the release kernel has run; 2 to 128 streams per process) did not hang in 4 pairs of runs of 15 to 86 s.

### After killing the hung processes

After one of the hung pairs was stopped (SIGTERM from `timeout`; both processes exited), the GPU read 84 % activity at 138 W twelve minutes later, with only a stalled new process (one queue) open on it, then 0 to 3 % at 1 to 14 W. For the next two hours every new process stalled in its first packet: `AqlQueue::ExecutePM4` <- `GpuAgent::InvalidateCodeCaches` <- `BlitKernel::Initialize`, during code object load, while device properties still read normally. On this card the other four hung pairs did not leave the GPU in this state; on the other card it happened twice and once cleared within about 20 minutes.

### Related

ROCm/rocm-systems#15850 reports a stall on the same card with more compute queues than `num_cp_queues`, where a queue with pending packets is left unmapped after a KFD evict/restore and never mapped again.

### Expected

A queue with pending packets runs again while another queue spins, or the runtime does not create more queues than the device has slots, and killing processes returns the GPU to a usable state.
====

`utils/prose.py` and `utils/jargon.py` on the body: clean.

## Lesson for cuda-to-rocm: a kernel that spins on another stream's kernel can hang once several processes share an RDNA4 GPU

Belongs in `references/diagnosis.md` (hangs), next to the SIMT forward-progress material. A lock or flag that a one-thread kernel spins on, released by a kernel queued on a different stream, assumes the other stream's queue keeps running while the spinner holds its own. Neither CUDA nor HIP promises that. On a Radeon AI PRO R9700 (gfx1201, MES, KFD `num_cp_queues 4`) every HIP process holds 5 compute queues by default (`GPU_MAX_HW_QUEUES` 4 plus ROCr's utility queue; count them in `/sys/class/kfd/kfd/proc/<host pid>/queues/*/type`), and HierarchicalKV's group lock hung in 9 of 9 two-process runs while one process always passed. Test multi-stream code with two processes on one GPU, not only one. Prefer host-side waits on events or on library-owned streams to device spins across streams. After stopping a hung run, check the GPU with a trivial kernel before the next run: it can stay unusable for hours (new processes stall in their first packet, `InvalidateCodeCaches`) while device queries still work. (HierarchicalKV, TheRock 10.2.0, gfx1201)

### Next steps (for whoever resumes)

1. A host owner checks or resets GPU PCI 03:00 on this host (and e3:00 if it recurs).
2. Re-dispatch this porter round: re-stamp the lock, rebuild 2fc61db from the diff above (or cherry-pick it if the clone still exists), then in this order, runs that should not hang first: 5 single-process runs, the 30 pairs with the workaround, the full test set with the workaround, the benchmarks and suite before and after (3 interleaved runs each), and last the runs that hang (the `GPU_MAX_HW_QUEUES=2` and `=3` pairs, the traced pair, the `CHURN`/`PAGEABLE` reproducers), since each hang can wedge the GPU.
3. Push 2fc61db only after the 30 pairs and the full set pass and Jeff has the cost numbers; then `advance-head`, review, the gfx942 revalidation with the list above, and the nvcc check of `include/merlin/group_lock.cuh`.

Logs (container-local, not committed): `agent_space/hang/real/` (`base-<n>-{A,B}.log`, `base-<n>-queues.txt`, `base-<n>-health.log`, `wedge-base5.txt`, `wedge-gdb.txt`, `summary.txt`), `agent_space/hang/` (`spinrepro.cpp`, `lockrepro.cpp`, `pairtest.sh`, `health.sh`, `chain2.sh`), `agent_space/{bench,suite}.sh`, `agent_space/clr/rs` (sparse rocm-systems at 43fa692), `agent_space/build-{base,wa}.log`.
