# knowhere port plan

Planner run 2026-11-20 on linux-gfx1100 (fleet job, no GPU used, nothing built). Upstream analysed at zilliztech/knowhere `main` `ee6b2d37359c` (2026-11-20). Builds on the intake screen in `notes.md`; the licence and duplicate-effort calls there are not repeated.

**Two questions for Jeff are open at the top of the `## Planning` section of `notes.md`** (may a porter start before the ROCm-DS conversation; does Milvus-level validation gate the PR). The plan below does not depend on how they are answered, except for when the porter starts and what the validator owes.

## Project

- Name: knowhere
- Upstream: https://github.com/zilliztech/knowhere (Apache-2.0), default branch `main`
- Fork: https://github.com/AMD-Ecosystem/knowhere, `fork_default_branch` `main` (3 commits behind upstream on 2026-11-20; the porter fast-forwards it with `moatlib.py mirror-branch` and cuts `moat-port` from it)
- Required gates: `wave64`, `wave32`. `windows` is waived (Jeff Daily, 2026-11-12: knowhere, Milvus and hipVS are Linux-only).
- Hard dependencies: none that are MOAT projects. hipVS and hipRaft (AMD-Ecosystem, ROCm-DS) are environment, not deliverables; `depends_on` stays empty.

## Existing AMD support and decision

- **hipVS / hipRaft (authoritative, AMD ROCm-DS).** https://github.com/AMD-Ecosystem/hipVS and https://github.com/AMD-Ecosystem/hipRaft are AMD's ports of cuVS and RAFT, keeping the `cuvs::` / `raft::` namespaces, header layout and CMake package names. They are the library knowhere's GPU path calls into, not a knowhere port. This port builds on them; it does not reimplement anything they provide.
- **What changed since intake.** Intake measured hipVS 1.0.0 (`release/rocmds-26.03`, cuVS 25.10 base) against knowhere's cuVS 26.02 pin and found two gaps: the CAGRA ACE build mode missing, and an unknown 25.10->26.02 drift; it recommended holding until ROCm-DS said whether hipVS would move to a 26.x base. It has: hipVS **2.0.0** is now the default branch `release/26.10` (head `82fe184a6ee88d1d1fa9596a30bea3e90a9b3abe`, 2026-11-03), based on **cuVS 26.06** (with 26.04), targeting ROCm 10.0.0, with hipRaft 2.0.0 on RAFT 26.06 and hipMM 5.0.0. hipVS is now *ahead* of knowhere's pin. `graph_build_params::ace_params` exists in hipVS 2.0 (and knowhere only names it in a `std::variant`; no knowhere config selects it). wave32 (RDNA) is listed as officially supported, with a list of wave32 fixes across CAGRA, IVF-PQ and IVF-Flat.
- **Every cuVS/RAFT symbol knowhere uses exists in hipVS/hipRaft 2.0** (checked by header grep: brute_force/ivf_flat/ivf_pq/cagra build/extend/search/serialize/deserialize, `serialize_to_hnswlib`, `hash_mode`, `search_algo`, `iterative_search_params`, `codebook_gen::PER_CLUSTER`, `filtering::bitset_filter`, `core::bitset`, `refine`, `vamana::build`, `kmeans::fit/predict` with `balanced_params`, and the `device_resources_manager` setters). Signature-level compatibility is unknown until the first compile (26.02 vs 26.06; see Risks).
- **Community port (not authoritative).** `konkolchin/knowhere-hip` (now served as `krainov/knowhere-hip`, branch `2.5`, last pushed 2026-08-12) targets the old knowhere 2.5 line on hipVS 25.10. A hint only; no code is inherited. Its two runtime claims are used as risk inputs: IVF-Flat and IVF-PQ worked on gfx1100; filtered (bitset) CAGRA search did not on hipVS 25.10.
- **No upstream ROCm work.** No ROCm/HIP/AMD PRs or issues on zilliztech/knowhere (rechecked 2026-11-20).
- **Decision: port.** A mechanical HIP build of knowhere's own cuVS glue against hipVS 2.0. knowhere has no kernels of its own beyond two element-wise thrust functors, so there is no port-versus-rewrite question; performance lives in hipVS.
- **Delivery vehicle:** an upstream PR to zilliztech/knowhere adding an opt-in ROCm build of the cuVS backend, CUDA path unchanged. Upstream does not link platform forks. Notes for PR preparation (moat-checkup, not the porter): knowhere requires a DCO `Signed-off-by` on every commit, uses `feat:`-style PR titles, and its mergify rules want an `issue: #<n>` reference for `kind/feature`.

## Build classification

**cmake** (recorded with `set-ext-type knowhere cmake`), driven by Conan 2 through `make`:

- `Makefile` composes Conan settings from `WITH_*` flags; `make build` runs `conan install` then `conan build`; `conanfile.py` forwards options as CMake cache variables (`WITH_GPU`, `WITH_CUVS`, ...).
- `CMakeLists.txt:28-60`: `WITH_GPU` without Cardinal sets `WITH_CUVS`; that calls `knowhere_prepare_cuda()` (`cmake/utils/cuda.cmake`, which runs `scripts/install_cuda.sh` and forces `/usr/local/cuda-13.2`), sets `CMAKE_CUDA_ARCHITECTURES RAPIDS`, includes `cmake/libs/librapids.cmake` (downloads rapids-cmake 26.02, `rapids_cuda_init_architectures`), re-declares `project(knowhere CXX C CUDA)`, includes `cmake/libs/libcuvs.cmake` (`rapids_cpm_find(cuvs 26.02.00 ... COMPONENTS cuvs_static)` from `rapidsai/cuvs` `v26.02.00`), and requires `find_package(CUDAToolkit 13.2 EXACT)`.
- `CMakeLists.txt` globs `src/common/*.cu` into the `knowhere` shared library and links `raft::raft cuvs::cuvs CUDA::cublas CUDA::cusparse CUDA::cusolver`; `cmake/libs/libdiskann.cmake:36-64` adds `thirdparty/DiskANN/src/diskann_gpu.cpp` and links `cuvs::cuvs` into the static `diskann`.
- The Faiss GPU tree (`thirdparty/faiss/faiss/gpu`, 75 `.cu`) is never compiled (`cmake/libs/libfaiss.cmake` globs CPU sources only), and `src/index/gpu/{flat_gpu,ivf_gpu}` is removed from the build unconditionally (`CMakeLists.txt`, `KNOWHERE_GPU_SRCS`).

## Port strategy

**Strategy A, CMake HIP language, compat through hipRaft's own CUDA-name headers.**

- A new opt-in CMake option `WITH_ROCM` (default `OFF`), forwarded as `with_rocm` by `conanfile.py` and `WITH_ROCM=True` by the `Makefile`, selects the HIP build of the cuVS backend whenever `WITH_CUVS` is in effect. With it `OFF` every CUDA line runs exactly as today.
- hipRaft ships `raft/cuda_runtime.h`, `raft/library_types.h`, `raft/cublas_v2.h` and friends that map CUDA runtime names onto HIP when `__HIP_PLATFORM_AMD__` is defined, and its `raft/util/cuda_rt_essentials.hpp` already dispatches (`raft/cuda_runtime.h` on HIP, `cuda_runtime.h` on CUDA). Every CUDA runtime name knowhere and its DiskANN GPU file use is covered there (`cudaGetDeviceCount`, `cudaGetErrorString`, `cudaMemGetInfo`, `cudaGetDevice`, `cudaSetDevice`, `cudaMallocHost`, `cudaFreeHost`, `cudaStreamCreate`, `cudaStream_t`, `cudaSuccess`, `cudaDataType_t`, all thirteen `CUDA_R_*`/`CUDA_C_*` values). So knowhere needs **no hipify run and no compat header of its own**: replacing its three direct `cuda_runtime_api.h` includes with `raft/util/cuda_rt_essentials.hpp` is platform-neutral, and the one `cuda_fp16.h` include gets a guarded `hip/hip_fp16.h`.
- The four `.cu` translation units are compiled as HIP (`set_source_files_properties(... LANGUAGE HIP)`); the `.cc` files that include RAFT headers stay on the host C++ compiler, which is how they build on CUDA too. They need `__HIP_PLATFORM_AMD__` and the HIP include path, which arrive through hipVS/hipRaft's exported targets (`hip::host`); the porter verifies this and adds `hip::host` explicitly only if it does not.
- **hipVS is consumed as an installed package first, source fetch second.** `rapids_cpm_find` already tries `find_package` before CPM. The HIP branch asks for hipVS's own version (`2.0`, its `cuvs-config-version.cmake` says `02.0.0`; asking for `26.02.00` would reject it, the same trap as hipMM's version numbers in the `cuda-to-rocm` libraries reference), does not request the `cuvs_static` component (the prebuilt package ships only the shared library), and falls back to CPM-fetching `AMD-Ecosystem/hipVS` pinned by commit (hipVS has no tags).

## CUDA surface inventory

| Item | Where | HIP/ROCm mapping | Risk |
|---|---|---|---|
| 4 `.cu` index instantiations (brute_force, ivf_flat, ivf_pq, cagra), ~20 lines each | `src/common/cuvs/integration/*_index.cu` | compile as HIP | low |
| cuVS glue template, 823 lines | `src/common/cuvs/integration/cuvs_knowhere_index.cuh` | unchanged source; hipVS headers | API drift 26.02->26.06 |
| `__device__` functors: `check_valid_entry`, `normalize_int8_cosine_op`, `rescale_int8_cosine_distance_op` (no `__global__`, no `<<<>>>`, no warp intrinsics, no shared memory, no hard-coded 32) | same `.cuh` | unchanged | none |
| thrust: `device_ptr`, `transform`, `make_zip_iterator`, `make_tuple`, `tuple`, `fill`, `for_each`, `partition`, `counting_iterator` with `raft::resource::get_thrust_policy` | same `.cuh` | rocThrust via hipRaft | low |
| raft linalg/matrix: `row_normalize`, `reduce`, `matrixVectorOp`, `matrix::fill`, `copy` | same `.cuh` | hipRaft | low |
| `cudaDataType_t` LUT / internal-distance dtype strings (`CUDA_R_16F`, ...) | `cuvs_knowhere_index.cuh:288` | `raft/library_types.h` macros -> `HIP_R_*` | low |
| `#include <cuda_fp16.h>`, `half` | `src/common/cuvs/integration/type_mappers.hpp:18,80` | `hip/hip_fp16.h` (`half`) | low |
| `#include <cuda_runtime_api.h>` + runtime calls | `src/index/index_factory.cc:18`, `src/common/comp/knowhere_config.cc:30`, `src/common/cuvs/integration/raft_initialization.cc` | `raft/util/cuda_rt_essentials.hpp` | low |
| runtime calls through RAFT includes | `src/index/gpu_cuvs/gpu_cuvs_{ivf_flat,ivf_pq,cagra}.cc`, `thirdparty/DiskANN/src/diskann_gpu.cpp` (`cudaMallocHost`/`cudaFreeHost` pinned staging, `cudaMemGetInfo`, device get/set) | already include `raft/util/cuda_rt_essentials.hpp` or RAFT headers; `diskann_gpu.cpp` gets the include explicitly if the transitive one is missing | low |
| RAFT resource manager: stream pools, mem pool, workspace pool (`default_pool_resource`) | `raft_initialization.cc` | hipRaft (hipMM 5.0 / CCCL memory resources) | drift: memory-resource API moved in RAFT 26.04/26.06 |
| DiskANN GPU build: `vamana::build`, `kmeans::fit/predict`, `balanced_params`, `brute_force` | `thirdparty/DiskANN/src/diskann_gpu.cpp` (+ `partition_and_pq.cpp`, `aux_utils.cpp` `#ifdef KNOWHERE_WITH_CUVS` call sites) | hipVS | API drift |
| Link: `CUDA::cublas`, `CUDA::cusparse`, `CUDA::cusolver` | `CMakeLists.txt` | `roc::hipblas`, `roc::hipsparse`, `roc::hipsolver` (or drop if hipVS's exported link interface already carries them) | low |
| Build: CUDA 13.2 install/force, rapids-cmake 26.02, `RAPIDS` archs, `CUDAToolkit 13.2 EXACT` | `CMakeLists.txt`, `cmake/utils/cuda.cmake`, `cmake/libs/librapids.cmake`, `cmake/libs/libcuvs.cmake`, `conanfile.py`, `Makefile` | HIP branch (below) | main work |
| Dead code: RMM includes, Faiss-GPU wrappers | `src/index/gpu/gpu_res_mgr.h`, `src/index/gpu/{flat_gpu,ivf_gpu}` | not compiled upstream; untouched | none |
| Vendored Faiss GPU (already carries upstream Faiss's ROCm support) | `thirdparty/faiss/faiss/gpu` | not compiled upstream; untouched | none |
| Cardinal (`WITH_CARDINAL`) | `cmake/libs/cardinal/*`, fetched from private `zilliztech/cardinal` | closed source, unobtainable; out of scope by necessity | none |

No driver API, NVRTC, PTX, textures, surfaces, cuFFT, cuRAND, cuDNN, NCCL, CUTLASS or managed memory in knowhere itself. hipVS 2.0 uses hipRTC at run time for its JIT-LTO search kernels; that is inside hipVS (see Risks).

## Risk list

1. **hipVS prebuilt wheels carry no wave32 device code.** `amd-libhipvs 2.0.0` (`https://pypi.amd.com/rocm-10.0.0/simple`, with `amd-libraft 2.0.0`, `amd-libhipmm 5.0.0`, `amd-rocmds-logger`) has code objects for `gfx90a`, `gfx942`, `gfx950` only (`strings libcuvs.so | grep amdgcn-amd-amdhsa--`; the gfx1xxx names in the binary are tables, not bundles). The wave64 gate can use the wheels; the wave32 gate (gfx1100, gfx1201, gfx1151) needs hipVS built from source for that arch. Building a dependency from source in the job's own directory is authorized. Expect a long build (the three-arch `libcuvs.so` is 608 MB); restrict `CMAKE_HIP_ARCHITECTURES` to the host arch, cap `MAX_JOBS`, and build hipVS once into a prefix under the job directory, then point knowhere at it through `CMAKE_PREFIX_PATH`.
2. **ROCmDS-cmake fetches six dependencies from the wrong GitHub org.** On `AMD-Ecosystem/ROCmDS-cmake` `release/26.10`, libhipcxx, hipCollections, hipcomp-core, hipBench, rocThrust and jitify-hip default to `AMD-Ecosystem/...`, which does not exist; a from-source hipVS/hipMM configure fails with `repository 'https://github.com/AMD-Ecosystem/libhipcxx/' not found`. Workaround, reproduced 2026-11-17 on a gfx942 host: `export RAPIDS_CMAKE_ROCM_ORG=ROCm` before configuring. Applies to the hipVS source build in risk 1 and to knowhere's own CPM fallback. Do not put it in knowhere's build files; it is a dependency defect (Jeff is deciding whether to file it).
3. **cuVS 26.02 -> 26.06 API drift.** All symbols exist; signatures may not match. Candidates: RAFT's move to CCCL memory resources (affects `workspace_resource_factory::default_pool_resource` and `set_workspace_memory_resource` in `raft_initialization.cc`), CAGRA `index_params`/`graph_build_params` constructors, IVF-PQ `search_params` fields. Fix any break in the HIP path only, guarded on `__HIP_PLATFORM_AMD__` (not on `CUVS_VERSION_*`: hipVS writes its own version, 2.0, into `cuvs/version_config.h`, so a cuVS version test reads wrong). Keep each guard minimal and note it in `notes.md`; they disappear when upstream knowhere bumps cuVS past 26.06.
4. **Filtered CAGRA search.** The community port found bitset-filtered CAGRA unsupported at run time on gfx1100 with hipVS 25.10. hipVS 2.0 lists CAGRA filtering and wave32 fixes. The GPU UT covers it ("Test Gpu Index Search With Bitset", "Test Gpu Index Search Simple Bitset", "Test GPU cuVS indexes pad filtered results"). If it still fails, it is a hipVS defect: record the minimal reproducer and the platform, draft the hipVS issue for Jeff, and do not paper over it in knowhere.
5. **hipVS JIT-LTO at run time.** hipVS 2.0 compiles CAGRA/IVF search fragments with hipRTC per exact gfx arch. The runtime needs hipRTC/comgr from the SDK and bitcode for the running arch (a source build must include it). First searches are slow; do not mistake JIT latency for a hang. hipVS also needs `libomp.so` from the SDK's `lib/llvm/lib` on the loader path (`cuda-to-rocm` libraries reference, RAPIDS section).
6. **ROCm version.** hipVS 2.0 targets ROCm 10.0.0; fleet hosts run the ROCm 10.1 pip SDK (no `/opt/rocm`). The 2.0 wheels have already been used against 10.1 (TornadoVM). hipVS's IVF-PQ has a `static_assert` that fires on ROCm major > 10; 10.1 is fine. If hipVS's Composable Kernel distance backend fails to build for gfx11/gfx12 on 10.1, configure hipVS with `BUILD_CK_BACKEND=OFF` and record it.
7. **int8 data on IVF-PQ.** hipVS rejects `coarse_search_dtype = CUDA_R_8I` on AMD (no i8xi8->f32 hipBLASLt matmul). knowhere does not set `coarse_search_dtype` (default `CUDA_R_32F`), so int8 IVF-PQ in the UT (`test_gpu_search.cc:298`) should pass; if it errors, this is why.
8. **Conan + HIP toolchain.** Conan builds the CPU dependencies with the host gcc and libstdc++; the HIP sources are compiled by the SDK's `amdclang++` with the same libstdc++. Watch for a second `fmt`/`spdlog` from hipRaft's logger colliding with Conan's `fmt/12.1.0`; rapids_logger hides spdlog, so a clash is unlikely but would show at link time.
9. **Multi-GPU hosts.** `initialize_raft` and `deserialize` loop over every visible device. Pin `HIP_VISIBLE_DEVICES` to one GPU of the target arch for every run (the DiskANN GPU test also wants >= 2 GB free).
10. **wave32/wave64.** knowhere has no warp-level code; the fault class lives in hipVS, which is why both gates are run, not one.

## File-by-file change list

Fork branch `moat-port`. Every change is behind `WITH_ROCM` or `__HIP_PLATFORM_AMD__`, or is platform-neutral.

- `CMakeLists.txt`: add `knowhere_option(WITH_ROCM "Build the cuVS GPU backend for ROCm (hipVS)" OFF)`. When `WITH_CUVS AND WITH_ROCM`: skip `knowhere_prepare_cuda()` and the `CMAKE_CUDA_ARCHITECTURES` default; default `CMAKE_HIP_ARCHITECTURES` from `HIPARCHS`/`GPU_TARGETS` when unset (else the hipVS/ROCmDS-cmake default list); `set(CMAKE_HIP_STANDARD 20)`; `project(knowhere CXX C HIP)` instead of `CUDA`; skip `find_package(CUDAToolkit 13.2 EXACT)`; mark the cuVS `.cu` sources `LANGUAGE HIP`; link `roc::hipblas roc::hipsparse roc::hipsolver` instead of `CUDA::cublas CUDA::cusparse CUDA::cusolver` (or nothing extra if hipVS's interface already links them; the porter checks with `cmake --graphviz` or the link line). `WITH_ROCM` with Cardinal is an error (`message(FATAL_ERROR ...)`): Cardinal is CUDA-only.
- `cmake/libs/librapids.cmake`: HIP branch downloads `RAPIDS.cmake` from `AMD-Ecosystem/ROCmDS-cmake` (`release/26.10`, the line hipVS 2.0 uses) and calls `rapids_hip_init_architectures(knowhere)` (module `rapids-hip`) instead of the CUDA pair; the CUDA-only `CMAKE_CUDA_FLAGS` block stays CUDA-only. If ROCmDS-cmake turns out to be more trouble than value for a consumer that mostly `find_package`s, the porter may drop rapids-cmake from the HIP branch and use plain `find_package(cuvs 2.0 REQUIRED)` + `FetchContent` fallback, and record why.
- `cmake/libs/libcuvs.cmake`: HIP branch sets the version to hipVS's (`2.0`), the repository to `https://github.com/AMD-Ecosystem/hipVS.git`, the pin to a commit on `release/26.10` (`82fe184a6ee88d1d1fa9596a30bea3e90a9b3abe` as of 2026-11-20), drops `COMPONENTS cuvs_static` and `rapids_find_package(CUDAToolkit ...)`, and keeps `BUILD_C_LIBRARY OFF`, `BUILD_TESTS OFF`, `BUILD_MG_ALGOS OFF`, `CUVS_NVTX OFF` for the fallback build.
- `cmake/utils/cuda.cmake`: unchanged (not called under `WITH_ROCM`).
- `cmake/libs/libdiskann.cmake`: unchanged unless the HIP build needs `hip::host` on `diskann` explicitly.
- `conanfile.py`: add option `with_rocm` (default `False`) and `tc.cache_variables["WITH_ROCM"]`.
- `Makefile`: add `WITH_ROCM ?=` with the same True/False parsing as `WITH_GPU`, forwarded as `-o &:with_rocm=...`, added to the `unexport` list and the `build` help text.
- `src/index/index_factory.cc`, `src/common/comp/knowhere_config.cc`, `src/common/cuvs/integration/raft_initialization.cc`: replace the direct `cuda_runtime_api.h` include with `#include <raft/util/cuda_rt_essentials.hpp>` (platform-neutral; it is how `src/index/gpu_cuvs/*.cc` already get the runtime).
- `src/common/cuvs/integration/type_mappers.hpp`: `#ifdef __HIP_PLATFORM_AMD__` -> `#include <hip/hip_fp16.h>`, else `#include <cuda_fp16.h>`.
- `thirdparty/DiskANN/src/diskann_gpu.cpp`: add `#include <raft/util/cuda_rt_essentials.hpp>` only if the HIP compile cannot see `cudaMallocHost`/`cudaFreeHost` transitively.
- `src/common/cuvs/integration/cuvs_knowhere_index.cuh`, `raft_initialization.cc`, `diskann_gpu.cpp`: minimal `__HIP_PLATFORM_AMD__` guards only where 26.06 drift breaks the compile (risk 3).
- Docs: one short ROCm subsection in `.agents/skills/building/SKILL.md` (the upstream build guide its `AGENTS.md` maintains) and the matching row in its build table: `make WITH_GPU=True WITH_ROCM=True [WITH_UT=True]`, hipVS from the AMD package index or `CMAKE_PREFIX_PATH`, `HIPARCHS`. Optionally a line in `README.md` if it lists GPU builds. No device-selection variables in front of example commands.
- Not touched: `scripts/install_cuda.sh`, the CUDA CI (`ci/*.groovy`, `ci/docker/builder/gpu/*`), `thirdparty/faiss/faiss/gpu`, `src/index/gpu/*`, Cardinal. No fork GitHub Actions.

## Build commands

Everything in the job's own directory and venv; nothing into the image's shared Python or the SDK.

```bash
# Conan and the knowhere build prerequisites (the upstream script's pinned versions)
$FLEET_VENV/bin/pip install conan==2.28.1
conan profile detect --force
conan remote add default-conan-local2 https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local2
# system packages as in scripts/install_deps.sh (libaio for DiskANN, ccache, ...)

# CPU baseline first (same host, same Conan cache): the non-GPU regression reference
make BUILD_DIR=$PWD/build-cpu WITH_UT=True

# hipVS for this arch
#  wave64 (gfx90a/gfx942/gfx950): prebuilt packages
$FLEET_VENV/bin/pip install --target $JOB/deps/hipvs --index-url https://pypi.amd.com/rocm-10.0.0/simple \
    amd-libhipvs==2.0.0 amd-libraft==2.0.0 amd-libhipmm==5.0.0 amd-rocmds-logger
#  wave32 (gfx1100/gfx1201/gfx1151): from source, host arch only
git clone -b release/26.10 https://github.com/AMD-Ecosystem/hipVS $JOB/deps/hipVS-src
git -C $JOB/deps/hipVS-src checkout 82fe184a6ee88d1d1fa9596a30bea3e90a9b3abe
export RAPIDS_CMAKE_ROCM_ORG=ROCm
cmake -S $JOB/deps/hipVS-src/cpp -B $JOB/deps/hipVS-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGPU_TARGETS=gfx1100 -DAMDGPU_TARGETS=gfx1100 \
    -DBUILD_TESTS=OFF -DBUILD_C_LIBRARY=OFF -DBUILD_MG_ALGOS=OFF -DCMAKE_INSTALL_PREFIX=$JOB/deps/hipvs-prefix
cmake --build $JOB/deps/hipVS-build -j $MAX_JOBS && cmake --install $JOB/deps/hipVS-build

# knowhere, HIP
export CMAKE_PREFIX_PATH=<hipVS/hipRaft/hipMM/logger lib64/cmake dirs>:$ROCM_SDK/lib/cmake
export HIPARCHS=gfx1100     # the host arch
make BUILD_DIR=$PWD/build-hip WITH_GPU=True WITH_ROCM=True WITH_UT=True WITH_C_API_TESTS=True
make BUILD_DIR=$PWD/build-hip WITH_GPU=True WITH_ROCM=True WITH_BENCHMARK=True   # benchmark compile coverage
```

`WITH_DISKANN` defaults to `True` in the `Makefile`, so the GPU build includes the DiskANN GPU path. If the wheels must come before the SDK's own CMake packages (the wheel hipRaft provides `CCCL::CCCL`, the SDK's may not), order `CMAKE_PREFIX_PATH` accordingly. The porter records the exact working command lines in `notes.md` under a build-recipe heading for the validator to reuse.

## Test plan

Gate (each of `wave64` and `wave32`, one pinned GPU, `ulimit -c 0`):

```bash
build-hip/Release/tests/ut/knowhere_tests "Test All GPU Index"
build-hip/Release/tests/ut/knowhere_tests "Test GPU cuVS indexes pad filtered results"
build-hip/Release/tests/ut/knowhere_tests "[cagra]"
build-hip/Release/tests/ut/knowhere_tests "[diskann]"     # GPU build path when a GPU is present
build-hip/Release/tests/ut/knowhere_tests                 # full suite, HIP build
ctest --test-dir build-hip/Release -R knowhere_c_api --output-on-failure
```

- "Test All GPU Index" (`tests/ut/test_gpu_search.cc:199`) iterates all four GPU indexes over search, bitset search, top-k, serialize/deserialize, CAGRA adapt-for-CPU (with and without ef), simple bitset, cosine, int8 cosine score range and Hamming. Record the per-index, per-section pass/fail and the recall numbers it prints.
- The non-GPU regression set is the rest of `knowhere_tests` plus the C API tests: compare the HIP build's full-suite result with the CPU baseline from the same host (`build-cpu`), test for test. A test that fails identically on the CPU build is pre-existing, not a regression; list it.
- `test_index_check.cc`, `test_config.cc` (`check cagra index config`), `test_knowhere_init.cc` (`SetRaftMemPool`), `test_nullable_id_map.cc` (GPU rows) carry `KNOWHERE_WITH_CUVS` paths and run in the full suite.
- Benchmarks: build `benchmark_float`, `benchmark_float_bitset`, `benchmark_float_qps`, `benchmark_float_range_bitset` (the ones with `KNOWHERE_WITH_CUVS` sections) under `WITH_BENCHMARK=True`. Running them needs HDF5 ANN datasets; one smoke run of `benchmark_float --gtest_filter=*CUVS*` on a small generated file (`gen_hdf5_file`) is useful evidence but not the gate.
- `tests/faiss` (`WITH_FAISS_TESTS`): CPU-only; run once on the HIP build as extra regression evidence if time allows.
- Milvus-level validation: see open question 2 in `notes.md`. Not part of the gate unless Jeff rules otherwise.
- CUDA path: before presenting the port, the CUDA build (`make WITH_GPU=True WITH_UT=True` with CUDA 13.2) must still compile on the mi355 fleet host's nvcc (MOAT's standing rule), proving the `WITH_ROCM` changes left it intact.

## Open questions

1. (Jeff) May a porter start now, or does porting wait for the ROCm-DS conversation intake recorded? Details and options in `notes.md`.
2. (Jeff) Is a Milvus GPU build plus its GPU index tests required before the upstream PR, a follow-on, or not wanted? Details and options in `notes.md`.
3. (porter, at first compile) Which of the 26.02->26.06 drift candidates actually break, and the minimal guard for each.
4. (porter/validator) Does bitset-filtered CAGRA pass on wave32 and wave64 with hipVS 2.0?
5. (FYI for Jeff's ROCm-DS conversation, not blocking) hipVS 2.0 wheels have no gfx11/gfx12/gfx1151 code objects; RDNA users must build hipVS from source. Also the community `konkolchin`/`krainov` knowhere-hip and milvus-hip repositories, which may be contractor work for AMD.
