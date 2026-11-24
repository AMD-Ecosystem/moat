# HierarchicalKV port plan

Planner run 2026-11-20 on linux-gfx942 (MI300X host, ROCm 10.1 pip SDK, no GPU used, nothing built). Analysis is against the upstream shallow clone at `451f240d185968d29ae022408e0bc3a8e987a4bb` (2026-11-15). Intake screened `c79175830d51`; the census below was re-taken at the newer tip and matches intake's counts to within a few call sites.

Jeff ruled the planner's three questions on 2026-11-23 (scope C, windows not yet ruled, base branch `master`); see "Scope C" below, which supersedes the open questions at the end.

## Scope C (Jeff Daily's ruling, 2026-11-23)

The rulings in notes.md ("Rulings on the planner's questions") supersede the open questions at the end of this plan. Scope is **C**, wider than the option A this plan was first written for:

1. The header library and every default CMake target (20 gtest executables, 3 default benchmarks). Unchanged from the plan below.
2. The opt-in research benchmarks (`-DHKV_BUILD_REPRO_BENCHMARKS=ON`: the 19 `HKV_REPRO_SOURCES` targets plus `digest_ablation_no_digest` and `concurrency_rw_lock`). Extra surface beyond the library: `benchmark/benchmark_util.cuh` (`__shfl*`, `__syncwarp`, `warpSize`, `%globaltimer`), `tbe_like_cache_baseline.cc.cu` (raw warp intrinsics, `warpSize`), and the `$<$<COMPILE_LANGUAGE:CUDA>:-O3>` option in `benchmark/CMakeLists.txt`, which needs a HIP twin. Built under the same `HKV_USE_HIP` branch; each binary is run once as a smoke test (they are benchmarks, not tests; they gate nothing beyond running to completion without a device fault).
3. A ROCm Bazel path beside the existing `cuda_configure`: a `rocm_configure` repository rule under `build_deps/gpus/` that locates a ROCm install (`ROCM_PATH`, default the pip SDK or `/opt/rocm`), generates `@local_config_rocm` with a `rocm` header/library target, a HIP clang toolchain, and a `build_defs.bzl` exposing the same `cuda_cc_library` macro name (so `include/BUILD`, `include/merlin/BUILD`, `include/merlin/core_kernels/BUILD` and `benchmark/BUILD` need no per-backend fork beyond selecting the config), plus a `build:rocm` config in `.bazelrc`. The Bazel targets in the repository are the header libraries and `//benchmark:benchmark_util` (it compiles `merlin_hashtable_benchmark.cc.cu`); there are no Bazel test targets upstream, so "the Bazel tests" means building every Bazel target (`bazel build --config=rocm //...`) and running `//benchmark:benchmark_util`. Bazel is built or fetched (bazelisk, version 5.0.0 per `.bazeliskrc`) inside the job directory.

Order of work: (1) library and default targets with their GPU tests passing, (2) research benchmarks, (3) Bazel. If the Bazel path cannot be finished in one round, its surface entries stay `owed` with what is done and what remains, never `scoped_out`.

Still out of scope: `benchmark/native_tbe` and `baselines/` (outside every option; `scoped_out` in `surface.json` citing the ruling).

The windows gate is **not ruled**. The port neither assumes a waiver nor writes a Windows-specific test filter; a Windows run is a validator question after the ruling.

Base branch: `master` (`set-base-branch` recorded by jeffdaily, 2026-11-23). Build toolchain for this round: TheRock ROCm 10.2.0 wheels (`rocm[libraries,devel,device-gfx942]==10.2.0` from `https://stable.repo.amd.com/rocm/whl-next/`).

## Project

- Name: HierarchicalKV (HKV), NVIDIA-Merlin.
- Upstream: https://github.com/NVIDIA-Merlin/HierarchicalKV, default branch `master`.
- Fork: https://github.com/AMD-Ecosystem/HierarchicalKV. Its GitHub default branch is `master` (checked read-only with `gh api` on 2026-11-20), and it has no other branch yet. `status.json` records `fork_default_branch: main`, which is wrong (see open question 3).
- Licence Apache-2.0 (intake). No MOAT dependency. HugeCTR depends on it: its SOK plugin includes `merlin_hashtable.cuh` and compiles the kernels itself.

## Existing AMD support

None is authoritative. Intake found no ROCm or AMD port, either upstream or in the ROCm and AMD-Ecosystem orgs. HYGON-AI/HierarchicalKV-das adapts HKV for Hygon DCU. Jeff ruled that it is not a base for this port, so it was not consulted. The Ascend repositories target Huawei NPUs. Upstream has not said it would refuse platform ports, so an upstream PR remains the delivery vehicle.

**Port vs rewrite.** These are hand-written hash-table kernels: no CUTLASS, no tensor cores, no wgmma. A mechanical, correctness-first HIP port is the right first step. Performance will lag in one known place. CUDA's `__pipeline_memcpy_async` overlaps global-to-shared copies with compute, and the HIP shim makes those copies synchronous. An AMD-native async path (`global_load_lds` on gfx94x/gfx950) is possible later work, not part of this port.

## Build classification

Strategy A applies: this is a pure CMake build, and the library is header-only.

- `CMakeLists.txt:17` has `project(merlin-hkvs LANGUAGES CXX CUDA)` and `find_package(CUDAToolkit)`. No torch or setup.py is involved in the library or the default targets.
- There is no library target. `add_library(hierarchical_kv ...)` is a commented-out TODO, and the deliverable is `include/`. Consumers (HugeCTR) put `include/` on their include path and compile `merlin_hashtable.cuh` with their own device compiler.
- The default build produces 20 gtest executables plus 3 benchmarks (`merlin_hashtable_benchmark`, `find_with_missed_keys_benchmark`, `dual_bucket_benchmark`). GoogleTest comes from the `tests/googletest` submodule, pinned at 0e0d9fe.
- `HKV_BUILD_REPRO_BENCHMARKS` (OFF) adds `benchmark/CMakeLists.txt`, which has 19 research benchmark targets plus `digest_ablation_no_digest` and `concurrency_rw_lock`.
- `baselines/CMakeLists.txt` is a separate opt-in CMake tree. It FetchContents warpcore, BGHT and cuCollections to compare against third-party hash tables.
- A Bazel path also exists: `WORKSPACE`, `include/BUILD` and `include/merlin/BUILD` use `cuda_cc_library`, and `build_deps/gpus` is TF-style `cuda_configure`. It serves TensorFlow Recommenders-Addons style consumers. Nothing in this repo builds TF ops.
- `benchmark/native_tbe` is a Python suite driving FBGEMM-GPU TBE through a patched NVIDIA recsys-examples checkout (CUDA torch). It is not compiled by CMake.

`set-ext-type HierarchicalKV cmake` is recorded.

**Two CMake facts that change the HIP build:**
- `set(CMAKE_BUILD_TYPE "Debug")` at `CMakeLists.txt:121` comes after the Release default. The directory's final value wins at generate time, so every target in the default build is configured Debug. `NDEBUG` is therefore undefined, and HIP's `__hip_assert` is a live `__builtin_trap()`. nvcc's Debug flags are `-g` only, which leaves device code optimised. CMake's HIP Debug flags are `-g -O` (corrected 2026-11-24; this line first said `-O0 -g`), clang's minimal optimisation level. The HIP branch must set `CMAKE_HIP_FLAGS_DEBUG` to `-O2 -g` (or `-O3 -g`) so the device code is optimised as on CUDA. It must not remove the line: the debug assertions are useful (warp-mask risk below) and the CUDA build must stay byte-identical.
- `add_executable(accum_or_assign_test tests/accum_or_assign_test.cc)` names a file that does not exist. It works only through CMake's pre-CMP0115 extension guessing (`cmake_minimum_required(VERSION 3.10)`), which appends `.cu`. A `set_source_files_properties(... LANGUAGE HIP)` keyed on the real file name needs the `.cc.cu` spelling. Fix the `add_executable` line to name the real file. That is a shared, harmless fix.

## Port strategy

Strategy A: one compat header, plus `enable_language(HIP)` behind an option. The CUDA path stays a pure passthrough.

1. **CMake.**
   - Add `option(HKV_USE_HIP "Build for AMD GPUs with ROCm" OFF)`; the porter may choose the name, but it must not be MOAT-named.
   - Under the option, `project(... LANGUAGES CXX)` plus `enable_language(HIP)` replaces CUDA, and `find_package(hip)`, `find_package(hipcub)`, `find_package(rocthrust)` and `find_package(libhipcxx)` provide the dependencies.
   - Mark every `*.cc.cu` that is built `LANGUAGE HIP`.
   - Take the architecture list from `CMAKE_HIP_ARCHITECTURES` instead of `-Dsm` and `-gencode`. Drop `CUDA_ARCHITECTURES OFF` and `CUDA_SEPARABLE_COMPILATION` on that path. Nothing needs RDC because it is header-only, so no `-fgpu-rdc`.
   - Keep the CUDA branch textually as it is.
   - libhipcxx's include root in the ROCm 10.1 pip SDK is `<sdk>/include/hipccl` (its `libhipcxx-header-search.cmake` says so). Use the imported target from `find_package(libhipcxx)` rather than hard-coding that path.
2. **Compat header** `include/merlin/hip_compat.cuh` (name at the porter's discretion). It is an empty include on CUDA. Every public header that names a CUDA header or API includes it first. The header must be self-contained for a consumer that only adds `include/` to its path, which is why shadow headers on a HIP-only include path are not the right shape here: HugeCTR would have to learn a second include directory. It contains:
   - Detection on `__HIP__` (clang in HIP mode) or `__HIPCC__`. Do not key on `__HIP_PLATFORM_AMD__` before `hip/hip_runtime.h` is included (build-headers-macros.md).
   - `hip/hip_runtime.h`, `hip/hip_fp16.h` and `hip/hip_cooperative_groups.h`. The hipCUB algorithm headers actually used are `device_scan`, `device_radix_sort` and `device_reduce` (prefer them to the umbrella header), with `namespace cub = hipcub;`.
   - The `cuda*` to `hip*` runtime aliases used by `include/`, `tests/` and the default benchmarks. Take the list from a hipify-perl dry run over those trees, then review it. Note that `cudaMallocHost(ptr, size, flags)` (3-arg, `allocator.cuh:72`) maps to `hipHostMalloc(ptr, size, flags)`, `cudaHostAllocDefault` and `cudaHostAllocMapped` map to `hipHostMallocDefault` and `hipHostMallocMapped`, and the 2-arg form also goes to `hipHostMalloc`.
   - `__vcmpeq4(a, b)`, used 62 times across 10 kernels: a portable SWAR byte compare returning exactly `0xff`/`0x00` per byte (`contains.cuh` and the dual-bucket kernels consume the full value, not only `& 0x01010101`). HIP has no SIMD-in-word intrinsics.
   - `__pipeline_memcpy_async(dst, src, size)`, `__pipeline_commit()` and `__pipeline_wait_prior(n)`. All 47 copy sites are the 3-argument form with sizes from `sizeof` of K, S, a pointer, `byte2`, `byte8` or `byte16`. The shim does a synchronous 4, 8 or 16-byte copy (plus the existing `byte`/`byte2` scalar specialisations in `LDGSTS`), and commit and wait become no-ops. This is safe: each copy is complete when issued, so `wait_prior(N)` holds trivially, and every consumer already has a `g.sync()` or `__syncthreads()` after its wait. A synchronous copy can only narrow a race window, never open one. `<cuda_pipeline.h>` is not included on HIP.
   - `__stcs` and `__ldcs` overloads for `uint4`, `uint2`, `uint32_t`, `uint64_t` and the scalar K/S types. HIP only has `__half`/`__half2` overloads. Use `__builtin_nontemporal_store/load` for scalars and plain accesses for vector structs. Do not redeclare HIP's half overloads.
   - A project-owned lane-mask type and full-mask constant (`uint64_t` and `~0ull` on HIP; `unsigned` and `0xffffffffu` on CUDA). Do not name macros after HIP's warp functions.
   - The thrust policy: `thrust::cuda::par(_nosync)` is used in `merlin_hashtable.cuh:190-194` and `array_kernels.cuh:111-115`. rocThrust 2.8 (`THRUST_VERSION 200805`) has `thrust::hip::par_nosync`, but `thrust::cuda` exists in rocThrust only for `memory_resource`, so a namespace alias would collide. Guard those two sites to pick `thrust::hip::par_nosync`.
3. **Guarded includes in the sources.** libhipcxx in ROCm 10.1 deliberately `#error`s on AMD for `<cuda/barrier>` and `<cuda/std/semaphore>`. HKV includes both: `kernel_utils.cuh:22`, `types.cuh:22` and `group_lock_kernels.cuh:19`. It uses neither; grep finds no `barrier`/`semaphore` symbol outside the include lines. Guard the three includes with `#ifndef` on the HIP detection. `<cuda/atomic>` and `<cuda/std/functional>` work as they are.
4. **Device timer.** `device_nano()` (`utils.cuh:76`), `host_nano_kernel` (`tests/test_util.cuh:51`) and `benchmark_util.cuh:247` read `%globaltimer` via inline PTX. On HIP, use `wall_clock64()` converted to nanoseconds; the counter runs at `hipDeviceAttributeWallClockRate`, which is 100 MHz on current CDNA and RDNA, but read it rather than assume it. Keep one helper so library and test agree. The units matter: `RSHIFT_ON_NANO = 20` turns nanoseconds into about 1 ms epochs for `EpochLru`, and an unscaled 100 MHz counter would make epochs 10x longer.
5. **Warp masks.** These are the only explicit warp-width code. Details are under the risks below.

## CUDA surface inventory

Counts are for `include/` (I), the 20 default tests (T) and `benchmark/` (B), at 451f240.

| feature | I / T / B | HIP mapping | risk |
|---|---|---|---|
| `__global__` kernels | 97 / 11 / 19 | same | none |
| `cuda::atomic<T, thread_scope_device>` (K, S, int, bool, uint32 digest words) | 30 / 0 / 2 | libhipcxx `cuda::atomic` | memory model; `reinterpret_cast<AtomicKey<K>*>` over raw bucket memory assumes `sizeof(cuda::atomic<T>) == sizeof(T)` (add a `static_assert` on HIP) |
| `cuda::std::memory_order_*` | 426 / 4 / 3 | libhipcxx | acquire/release CAS locking correctness |
| `atomicCAS` / `atomicAdd` / `atomicExch` | 10 / 83 / 8 (I) | native HIP | none (device memory) |
| cooperative groups `tiled_partition<N>`, N in {1, 4, 8, 16, 32} | 48 | HIP CG | tile-relative `ballot` compacts to bits 0..N-1 (faults-wavefront.md), so `__ffs`/`__popc` on 32-bit results are correct on wave64 |
| tile `.shfl` / `.ballot` / `.any` / `.sync`, `cg::reduce(g, x, cg::less<S>())` | 242 / 81 / 14 / ~180 / 10 | HIP CG (`amd_hip_cooperative_groups_reduce.h`) | tile size 1 for `remove_kernel_v2<...,1>`: confirm it compiles |
| raw `__ballot_sync` / `__any_sync` / `__syncwarp(mask)` | 7 / 2 / 5 (I), 3 `__syncwarp` (B) | HIP `*_sync` with a 64-bit mask | **high**: see risks 1-2 |
| `__shfl_sync` in `__shfl_sync_ptr` (`kernel_utils.cuh:87`, defined but unused) | 1 | needs 64-bit mask | non-dependent call in a template, so HIP's `static_assert` fires at definition: must change even though unused |
| `__shfl*`, `warpSize` in benchmarks (`benchmark_util.cuh`, `tbe_like_cache_baseline.cc.cu`) | 0 / 0 / 3+4 | width-32 forms | only if the repro benchmarks are in scope |
| `__vcmpeq4` | 62 / 0 / 0 | compat SWAR shim | correctness of the shim is the digest probing's correctness |
| `__pipeline_memcpy_async` / commit / wait_prior, `<cuda_pipeline.h>` | 47 / 120 / 120 | synchronous shim | performance only |
| `<cuda/barrier>`, `<cuda/std/semaphore>` | 3 includes, 0 uses | guard out | none |
| `__stcs` / `__ldcs` (uint4, uint2, K, S) | 8 / 1 | compat overloads | none |
| `__ldg` | 1 | native | none |
| inline PTX `%globaltimer` | 1 / 1 / 1 | `wall_clock64()` scaled | units (see 4) |
| `__ffs` / `__popc` on 32-bit tile ballots and digest compares | 117 / 19 | native | none |
| `__constant__` + `cudaMemcpyToSymbol` (reserved-key constants, `types.cuh:59-107`) | 7 | `hipMemcpyToSymbol(HIP_SYMBOL(x))` or plain symbol | header-defined `__constant__` in a header-only lib: each TU gets its own copy, same as CUDA without RDC; check that every TU's copy is initialised by the init call it makes |
| `cudaHostAlloc(Mapped)` + `cudaHostGetDevicePointer` (hybrid HBM + host values, `core_kernels.cuh:214-228`) | 3 | `hipHostMalloc(..., hipHostMallocMapped)` + `hipHostGetDevicePointer` | kernels do plain loads and stores on host-resident values (no atomics there; the only atomics hit device `buckets_size`), so default coherence suffices |
| `cudaMallocAsync` / `cudaFreeAsync` | 1+2 / 57 | `hipMallocAsync` / `hipFreeAsync` | mempool support on gfx942/gfx1100: yes |
| `cudaGetDeviceProperties` (`sharedMemPerBlock`, `multiProcessorCount`, `maxThreadsPerBlock`) | 2 | same | AMD reports 64 KB LDS per block (gfx950 160 KB) vs 48 KB on sm80; kernel selection by shared-memory size will pick the pipeline kernels for larger `dim` on AMD. Correct, but a different path from CUDA |
| CUB `DeviceScan::ExclusiveSum`, `DeviceRadixSort::SortPairs`, `DeviceReduce::Reduce(..., cuda::std::plus<>())` | 6 | hipCUB | radix sort is integer keys with `begin_bit = 0`, `end_bit = 8*sizeof(K)`, so neither the 64-bit bit-range nor the float-NaN rocPRIM defects apply; rocPRIM's 4-byte minimum temp storage is harmless (sizes come from the query call) |
| Thrust `device_malloc_allocator`, `exclusive_scan`, `sort` | 9 | rocThrust | policy namespace (step 2) |
| streams, events | 4+9 / 151+11 / 18+83 | native | none |
| `cudaMallocManaged` (one test) | 0 / 1 | `hipMallocManaged` | gfx942 supports it; on the APU it is system memory |

No cuBLAS, cuFFT, cuRAND, cuSPARSE, cuDNN, NCCL, textures, driver API, NVRTC or graph capture.

## Risk list

1. **Raw warp masks, wave64 and Debug.** `update_values.cuh` and `update_score.cuh` each have a one-thread-per-key kernel. The kernel returns early (reserved key, `kv_idx >= n`), then computes `active_warp_mask = __ballot_sync(0xffffffff, true)`, later `if (!retry_needed) return; __syncwarp(active_warp_mask); retry_warp_mask = __ballot_sync(active_warp_mask, true);` and spins on `__any_sync(retry_warp_mask, !completed)`. CUDA allows exited threads in the mask. HIP's `__hip_check_mask` asserts `mask == __ballot(true)` for the lanes that arrive, and that is a trap whenever `NDEBUG` is undefined, which it is in this CMake's effective Debug build. So the port must not just widen the literal to 64 bits. On HIP the masks must be the lanes actually present at each call, i.e. `__activemask()` (64-bit) at each point, or mask-free `__ballot`/`__any`. The loop logic is wave-wide and nothing indexes the mask, so whole-wavefront semantics are correct on wave64 and wave32 alike.
   - The pipelined kernels (`update.cuh:181`, `update_values.cuh:207`, `update_score.cuh:206`) compute `__ballot_sync(0xffffffff, key_idx_base < n)` with all lanes present, then `__syncwarp(active_warp_mask)` from exactly those lanes. A 64-bit full mask is right there.
   - Every `unsigned int` holding one of these masks becomes the lane-mask type.
2. **`retry_mask |= 1U << (index within group)`** in the pipelined kernels: the index is below `GROUP_SIZE` (16 or 32), so it is not a wave-width mask. Leave it.
3. **libhipcxx atomics.** All bucket locking is CAS of a 64-bit key to `LOCKED_KEY` with acquire, then a release store, through `cuda::atomic<K, thread_scope_device>`, sometimes `reinterpret_cast` onto raw memory. Correctness depends on libhipcxx lowering device-scope acquire/release to agent-scope HIP atomics. The concurrency tests (`find_or_insert_ptr_lock_test`, `group_lock_test`, `lock_unlock_test`, `correctness_stress_test`) are the evidence. Run them several times on wave64.
4. **Spin loops under lock-step execution.** The retry loops above, and the `group_lock` reader/writer spin, wait on keys locked by other threads. A lock holder in the same wavefront releases within the same pass (CAS, copy, store-release in one branch body), so no intra-wave deadlock is evident. Cross-wave spinning needs forward progress between wavefronts. That is fine on discrete GPUs; it is a starvation risk on a low-CU APU (the gfx1151 lesson from stdgpu). If any test hangs, classify it with this in mind before changing code.
5. **`__vcmpeq4` shim.** It is the core of digest probing; a wrong byte mask silently loses keys. Unit-check the shim against a host reference over byte patterns (0x00, 0x80, 0xff edges) before running the suite, as a porter self-check rather than a shipped test.
6. **Kernel selection keyed on `sharedMemPerBlock`.** The AMD value differs, so AMD runs different kernels than sm80 for some `dim`. The tests cover every `dim` they use, so the pipeline and TLP kernels both run, but the paths taken are not CUDA's.
7. **Test memory size.** Seven test files size tables at 64M keys (`INIT_CAPACITY = 64 * 1024 * 1024UL`) and/or `max_hbm_for_vectors` of `GB(16)`/`GB(100)` (an allowance, not an allocation). Fine on MI300X (192 GB). It needs checking on a 16-24 GB Radeon. It is not allowed on the gfx1151 laptop (standing rule) without shrinking or filtering (open question 2).
8. **Header-defined `__constant__` reserved keys** (`types.cuh`). Per-TU copies initialised by `init_reserved_keys` through `cudaMemcpyToSymbol`. This is the same model as CUDA without RDC; just keep it non-RDC.
9. **Fresh device memory is not zeroed on ROCm.** `buckets_size`, the main read-modify-write counter, is zeroed by a synchronous `cudaMemset` at init (`core_kernels.cuh:373`), and `realloc` zeroes the grown tail (`core_kernels.cuh:138`, `:161`). Check any other counter the porter touches the same way.

## File-by-file change list

- `CMakeLists.txt`: HIP option branch, `CMAKE_HIP_FLAGS_DEBUG` optimised, real `.cc.cu` name for `accum_or_assign_test`, HIP language on sources, libhipcxx/hipCUB/rocThrust targets. The CUDA branch is unchanged.
- `include/merlin/hip_compat.cuh` (new): everything in step 2.
- `include/merlin/types.cuh`, `include/merlin/core_kernels/kernel_utils.cuh`, `include/merlin/core_kernels/group_lock_kernels.cuh`: compat include, guard `<cuda/std/semaphore>`/`<cuda/barrier>`/`<cuda_pipeline.h>`.
- `include/merlin/utils.cuh`, `include/merlin/debug.hpp`, `include/merlin/optimizers.cuh`, `include/merlin/array_kernels.cuh`, `include/merlin/allocator.cuh`, `include/merlin/core_kernels.cuh`, `include/merlin/flexible_buffer.cuh`, `include/merlin/memory_pool.cuh`, `include/merlin/group_lock.cuh`, `include/merlin_hashtable.cuh`: compat include in place of, or before, the CUDA headers. `utils.cuh` gets the timer branch. `array_kernels.cuh` and `merlin_hashtable.cuh` get the thrust policy guard.
- `include/merlin/core_kernels/update.cuh`, `update_values.cuh`, `update_score.cuh`: lane-mask type and HIP-correct masks (risk 1).
- `include/merlin/core_kernels/kernel_utils.cuh`: `__shfl_sync_ptr` mask type.
- `tests/test_util.cuh`: timer branch. Tests include merlin headers, so the runtime aliases reach them. Expect a handful of test-only spellings that hipify flags.
- `benchmark/benchmark_util.cuh`: timer branch, needed because the 3 default benchmarks include it.
- `README.md`: a "Building for AMD GPUs (ROCm)" section with the configure line and the consumer note (add `include/`, compile as HIP, link libhipcxx/hipCUB/rocThrust include paths). No device-selection environment variables in example commands.
- `benchmark/CMakeLists.txt`, `benchmark/*.cc.cu` as needed (scope C), Bazel files under `build_deps/gpus/`, `WORKSPACE`, `.bazelrc` and the `BUILD` files (scope C).
- Not touched: `baselines/`, `benchmark/native_tbe`, `third_party/`.

## Build commands

On a Linux host with the ROCm pip SDK (or `/opt/rocm`). Set `ROCM_PATH` to the SDK root (`rocm-sdk path --root`), and `MAX_JOBS` to physical cores at most.

```
git submodule update --init tests/googletest
cmake -S . -B build-hip -G Ninja -DHKV_USE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_PREFIX_PATH="$ROCM_PATH"
cmake --build build-hip -j "$MAX_JOBS"
```

Use the executing arch: gfx942 or gfx90a for wave64, gfx1100 or gfx1201 for wave32, gfx1151 on Windows. Use `bash utils/timeit.sh HierarchicalKV compile -- ...` for telemetry.

The CUDA no-regression compile runs on an nvcc host (mi355 fleet host, or `/opt/conda/envs/cuda-12.8`). Configure the unchanged CUDA branch with `cmake -S . -B build-cuda -Dsm=80` and build every default target. Building is enough to show the CUDA path is untouched. Record the command and result in notes.md.

## Test plan

- GPU tests: all 20 default gtest executables, 214 `TEST`/`TEST_F` cases (208 at intake's tip) (export_batch_if_test is a plain `main()` with no gtest cases). Upstream runs them with `bash run_all_tests.sh`, which executes every `build/*_test`. For the HIP tree, run each `build-hip/*_test` and record per-binary pass/fail counts plus the gtest summary line. `IS_BLOSSOM_CI` must stay unset so the hybrid-memory correctness checks run.
- Repeat the concurrency set three times on wave64: `find_or_insert_ptr_lock_test`, `group_lock_test`, `lock_unlock_test`, `correctness_stress_test`, `dual_bucket_test`. Intermittent lock bugs show up there.
- The run must use the CMake default (effective Debug, `NDEBUG` undefined) so HIP's warp-mask assertions are live. A Release rerun is optional extra evidence.
- Benchmarks: build the 3 default benchmarks and run `merlin_hashtable_benchmark` once with default arguments as a smoke test. Record throughput for information only; it gates nothing.
- Hybrid HBM + host-memory cases (values in mapped pinned memory) are inside `merlin_hashtable_test`, `find_or_insert_test` and the others via `max_hbm_for_vectors`. They must run, not be filtered.
- Prove kernels run on the GPU (count dispatches with `AMD_LOG_LEVEL=3` on one binary, or rocprofv3 `--kernel-trace`).
- There are no non-GPU tests to protect: every test needs a device. The non-regression bar is the nvcc compile of the unchanged CUDA branch.
- Consumer check (recommended): compile a one-file consumer that includes `merlin_hashtable.cuh` with only `include/` plus the ROCm package include paths. That is HugeCTR's contract.

## Open questions (Jeff must rule before the porter starts)

1. **Scope cut.** Intake recommended leaving these out of the first port: the Bazel build, the opt-in research benchmarks (`HKV_BUILD_REPRO_BENCHMARKS`, 21 targets), `benchmark/native_tbe` (FBGEMM/recsys-examples Python suite) and `baselines/` (third-party hash-table comparisons). This plan follows that, but it is a partial port and needs your answer. Options are in notes.md.
2. **Windows gate.** Upstream documents Linux only, and its consumer HugeCTR is Linux-only. The code has no Linux-only API, so a Windows build is plausible, but the large-table tests cannot run on the gfx1151 laptop as written. The options are to waive the gate or to validate on Windows with named large cases filtered.
3. **Base branch.** `status.json` says `fork_default_branch: main`, while upstream and the fork both use `master`. `moatlib.py set-base-branch HierarchicalKV master --by <you>` records it. The tool treats it as a person's ruling, so the planner did not run it.
