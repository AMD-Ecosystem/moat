# cuBQL -- ROCm/HIP port plan (Linux gfx90a lead)

## Project
- Name: cuBQL
- Upstream: https://github.com/NVIDIA/cuBQL (NVIDIA, maintainer Ingo Wald)
- Default branch: main
- Clone HEAD analyzed: e82f1dc ("Merge pull request #34 ... HIP and CPU use of cuBQL", 2026-06-14)
- License: Apache-2.0 (cleared)
- Build type: CMake (pure, not a torch extension)
- Ported as a dependency of `barney` (its `rtcore/cuda` software ray-tracer is built on cuBQL).

## Existing AMD support (IMPORTANT -- changes the framing)

The upstream is NOT a pristine CUDA-only codebase. Its maintainer (Ingo Wald, NVIDIA) merged
**PR #34 on 2026-06-14, titled "Various fixes and cleanups for HIP and CPU use of cuBQL"**, whose
body says the changes "came as a result of using cuBQL once in a HIP environment". So the device/
host SOURCE already carries first-party HIP scaffolding:
- `cuBQL/math/common.h`: `#if defined(__HIPCC__)` includes `<hip/hip_runtime.h>` / `<hip/driver_types.h>`.
- `cuBQL/builder/cuda.h`: `#ifdef __HIPCC__ include <hip/hip_runtime.h>`.
- `cuBQL/builder/cuda/builder_common.h`: includes `<hipcub/hipcub.hpp>` and does `namespace cub { using namespace hipcub; }` under `__HIPCC__`.
- `cuBQL/math/math.h`, `vec.h`: device guards already widened to `__HIP_DEVICE_COMPILE__` / `__HIPCC__`.

Authoritative-vs-community judgment: **AUTHORITATIVE but INCOMPLETE.** This is the upstream
author's own in-progress HIP enablement, not a community fork. Per PORTING_GUIDE ("AUTHORITATIVE
but incomplete -> validate and improve it"), the deliverable is to FINISH and VALIDATE his HIP path
on real gfx90a and contribute the missing pieces upstream -- NOT a from-scratch re-port and NOT a
parallel compat-header that fights his scheme. No separately-named AMD project (ROCm-DS style) and
no ROCm-org fork exists; upstream forks are all personal. No HIP/ROCm doc.

### What is actually missing (the real delta to deliver)

1. **CMake has ZERO HIP path.** Root `CMakeLists.txt` only does `enable_language(CUDA)` gated by
   `CUBQL_DISABLE_CUDA`/`CUBQL_OMP`; there is no `USE_HIP`/`enable_language(HIP)`, no `LANGUAGE HIP`
   on the `.cu`, and `CUBQL_HAVE_CUDA` is the only switch that builds the GPU libs/samples. So today
   nothing ever invokes hipcc -- the HIP source path is dead code. This is the bulk of the work.

2. **The `cudaXxx` RUNTIME symbols are UNMAPPED under HIP.** The scaffolding aliases only `cub->hipcub`
   and pulls `hip_runtime.h`; it never maps `cudaMalloc`/`cudaMemcpy`/`cudaStream_t`/... to `hipXxx`.
   All ~12 distinct runtime symbols flow through one token-paste macro
   `#define CUBQL_CUDA_CALL(call) CUBQL_CUDA_CHECK(cuda##call)` (math/common.h:320), e.g.
   `CUBQL_CUDA_CALL(MallocAsync(...))` -> literal `cudaMallocAsync(...)`. HIP does not auto-provide
   `cudaXxx` spellings, so under hipcc these are undefined. ONE small compat shim fixes the entire
   runtime surface in a single place.

3. **`CUDART_VERSION >= 11020` guard (cuda.h:57) is FALSE on HIP** (CUDART_VERSION undefined == 0).
   So the `AsyncGpuMemoryResource` (hipMallocAsync path) is compiled OUT and it silently falls back
   to `ManagedMemMemoryResource` (cudaMallocManaged). That fallback is ALSO unmapped, so it would not
   compile either. Need a HIP branch that selects the async resource (hipMallocAsync + hipMemPool are
   supported on ROCm; see version note below).

## Build classification: CMake (Strategy A)

Evidence: root `CMakeLists.txt:14` `project(cuBQL ... LANGUAGES C CXX)`, `:30` `enable_language(CUDA)`,
GPU libs built via `add_specific_instantiation(cuda cu ...)` compiling `builder/cuda/instantiate_builders.cu`
(cuBQL/CMakeLists.txt:60-137). No `find_package(Torch)`, no setup.py/pyproject, no `CUDAExtension`.
=> pure CMake. ext_type = cmake.

## Port strategy: Strategy A, aligned to the upstream's existing `__HIPCC__` scheme

Do NOT introduce a separate `cuda_to_hip.h` that races the upstream guards. The upstream already
chose `#ifdef __HIPCC__` + `<hip/hip_runtime.h>` + `cub=hipcub` as its mechanism. Extend THAT:

1. **Runtime symbol shim (small, central).** In the existing `#if defined(__HIPCC__)` block of
   `cuBQL/math/common.h` (right after `<hip/hip_runtime.h>` is included, before the macros that
   token-paste `cuda##call`), add the `cudaXxx -> hipXxx` aliases for exactly the symbols cuBQL uses.
   The complete set (from the inventory below):
   `cudaError_t, cudaSuccess, cudaStream_t, cudaEvent_t, cudaGetErrorString, cudaGetLastError,
    cudaDeviceSynchronize, cudaStreamSynchronize, cudaGetDevice, cudaSetDevice, cudaGetDeviceCount,
    cudaMalloc, cudaFree, cudaMallocManaged, cudaMallocAsync, cudaFreeAsync, cudaMemcpy, cudaMemset,
    cudaMemcpyDeviceToHost, cudaMemcpyHostToDevice, cudaMemcpyDefault, cudaMemPool_t,
    cudaDeviceGetDefaultMemPool, cudaMemPoolSetAttribute, cudaMemPoolAttrReleaseThreshold`.
   Each maps 1:1 to the `hip*` name. Because every call goes through `CUBQL_CUDA_CALL(cuda##X)` and the
   raw `cudaStream_t`/`cudaMemcpyDeviceToHost` spellings, aliasing the names is sufficient; call sites
   are untouched. Keep these aliases under `__HIPCC__` only so the CUDA build is byte-identical.
   (Prefer this minimal alias list over `#include <hip/hip_runtime.h>`-with-deprecated tricks.)

2. **Async-memory version guard.** Replace the bare `#if CUDART_VERSION >= 11020` (cuda.h:57) selection
   so the HIP toolchain takes the async path: `#if defined(__HIPCC__) || CUDART_VERSION >= 11020`.
   hipMallocAsync / hipFreeAsync and the hipMemPool API (hipDeviceGetDefaultMemPool,
   hipMemPoolSetAttribute, hipMemPoolAttrReleaseThreshold) are available on ROCm 5.2+; our stack is
   ROCm 7.2.1, so fine. If a runtime probe shows the default pool unsupported on some arch, fall back
   to the (now-mapped) plain hipMalloc `DeviceMemoryResource`; keep the CUDA `CUDART_VERSION` arm intact.

3. **CMake `USE_HIP` path mirroring the CUDA gating.** In root `CMakeLists.txt`, add a parallel option
   that does NOT rip out CUDA:
       option(CUBQL_USE_HIP "Build cuBQL GPU code with HIP for AMD GPUs" OFF)
   When `CUBQL_USE_HIP` is ON: set `CUBQL_DISABLE_CUDA`/skip the CUDA detection, `enable_language(HIP)`,
   set `CUBQL_HAVE_CUDA ON` (the existing switch that gates all GPU libs/samples -- reuse it so the
   per-type instantiations and the s02-s07 samples build unchanged), and do NOT hardcode the arch:
       if (NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
         set(CMAKE_HIP_ARCHITECTURES "gfx90a")   # default lead only when unset
       endif()
   In `cuBQL/CMakeLists.txt` and `samples/CMakeLists.txt`, mark the `.cu` sources `LANGUAGE HIP` when
   `CUBQL_USE_HIP` (`set_source_files_properties(... PROPERTIES LANGUAGE HIP)`), so hipcc sees them and
   `__HIPCC__` is defined. The `add_specific_instantiation(cuda cu ...)` targets and every
   `if (CUBQL_HAVE_CUDA)` sample stanza then work with no per-target edits. Gate the CUDA-only target
   properties (`CUDA_SEPARABLE_COMPILATION`, `CUDA_RESOLVE_DEVICE_SYMBOLS`, `CUDA_USE_STATIC_CUDA_RUNTIME`)
   so they are not applied on the HIP path (harmless but tidy: wrap or leave -- they are ignored for HIP
   targets). Keep `-DCUBQL_HAVE_CUDA=1` compile def on the cuda instantiation targets (the source uses it).
   Subproject mode: the `CMAKE_CUDA_ARCHITECTURES`-not-set FATAL_ERROR (CMakeLists.txt:70) must be guarded
   so a HIP subproject build (barney) is not forced to set a CUDA arch -- bypass it when `CUBQL_USE_HIP`.

4. **`<<<...,32>>>` are launch geometry, NOT warp ops** -- block size 32, arch-agnostic; leave as-is.
   No `__shfl`/`__ballot`/`__popc`/`warpSize`/`__activemask`/cooperative-groups anywhere in cuBQL
   (verified). So the warp-size fault class is LOW risk here (see Risks).

## CUDA surface inventory

- Kernels / device code: `__global__` kernels + `__device__`/`__host__` helpers across
  `builder/cuda/{radix,rebinMortonBuilder,sm_builder,sah_builder,elh_builder,wide_gpu_builder,
   refit,refit_aggregate,builder_common}.h` and the samples. `atomicMin/atomicMax/atomicAdd`,
  `__threadfence()` (refit_aggregate.h:54), `__syncthreads`, `__shared__`. All compile directly under hipcc.
- Device intrinsics: `__int_as_float`/`__float_as_int` (have HIP equivalents; guarded already),
  `CUDART_INF_F` (HIP provides it). No warp-shuffle/ballot/popc/activemask. No textures/surfaces.
  No cudaArray. No managed-memory layered anything.
- Library: CUB `DeviceRadixSort::SortKeys`/`SortPairs` in radix/rebinMorton/sm/elh builders ->
  hipCUB/rocPRIM. Upstream already aliases `cub=hipcub`, so SortKeys/SortPairs resolve. Watch the
  temp-storage two-call pattern (size query then sort) -- standard, supported by hipCUB. NO thrust in
  the builder (only CUB).
- Runtime API (~20 call sites, all via `CUBQL_CUDA_CALL(cuda##X)` or raw spellings): the symbol set
  listed in Strategy step 1. Pinpoints: `cuda.h` (Malloc/MallocManaged/MallocAsync/FreeAsync/
  GetDeviceCount/SetDevice/DeviceGetDefaultMemPool/MemPoolSetAttribute/cudaMemPoolAttrReleaseThreshold/
  cudaMemPool_t), `gpu_builder.h` (GetDevice/StreamSynchronize), `radix.h`/`rebinMortonBuilder.h`/
  `sm_builder.h`/`sah_builder.h`/`elh_builder.h`/`wide_gpu_builder.h` (Memcpy{D2H,H2D,Default}/Memset/
  events/StreamSynchronize), `math/common.h` (DeviceSynchronize/GetLastError/StreamSynchronize/
  GetErrorString and the cudaError_t/cudaSuccess in the check macros).
- Streams/events: `cudaStream_t` plumbed through every builder signature (the instantiation TU's
  template signatures, cuda.h, all builder headers). `cudaEvent_t` in sm_builder.h:453, radix.h:618,
  rebinMortonBuilder.h:1194. 1:1 to hipStream_t/hipEvent_t.

## Risk list (fault classes)

- **Warp size (LOW).** No `__shfl*`/`__ballot`/`__popc`/`warpSize`/cooperative-groups in cuBQL; the
  `,32>>>` launches are block dims, and CUB radix sort is block/device-collective (arch-internal).
  No host buffer is sized by warp count. So wave64-vs-wave32 should not bite. STILL: the cross-arch
  consistency gate is the real test -- on the gfx1100/gfx1151 follower, diff sample output against the
  gfx90a lead for the same seed (the builders are deterministic given a fixed input), don't accept
  "plausible".
- **hipCUB block-collective TempStorage reuse race on wave64 (MEDIUM).** PORTING_GUIDE: a block-collective
  reusing the same `TempStorage` union across back-to-back calls races on a 64-thread (single-wavefront)
  block without an explicit `__syncthreads()` (CUDA's 32-wide warps masked it). cuBQL uses
  `DeviceRadixSort` (device-wide, allocates its own temp), which is lower risk than Block* collectives,
  but audit any builder that reuses a temp buffer across two sort calls (rebinMortonBuilder.h does several
  back-to-back `SortPairs`). If a builder miscomputes only on gfx90a, add `__syncthreads()` / fresh temp.
- **hipMallocAsync / hipMemPool availability + semantics (MEDIUM).** The async pool path is the default
  on HIP after the version-guard fix. Confirm hipDeviceGetDefaultMemPool + hipMemPoolSetAttribute(
  hipMemPoolAttrReleaseThreshold) succeed on gfx90a/ROCm 7.2.1 at runtime; the upstream comment warns
  async allocs land on the stream's device, not the cudaSetDevice device -- behavior is the same on HIP,
  so single-GPU validation is clean. Fallback to plain hipMalloc resource if the pool API errors.
- **CUDART_VERSION == 0 on HIP (LOW, already in inventory).** Beyond cuda.h:57, grep confirms only that
  one gate; fix as above. (PORTING_GUIDE host-side analog of the `__CUDA_ARCH__` collapse trap.)
- **Fresh device memory not zeroed on ROCm (LOW-MEDIUM).** Builders that write node/prim arrays partially
  and rely on zeroed allocations could read garbage in untouched bytes (passes in isolation, fails after
  allocator reuse). Audit `clearBuildState`/`initNodes`/`writeFinalNodes` paths; if a sample's result is
  nondeterministic in-suite only, add an explicit hipMemset before the partial-write kernel (HIP-gated).
- **-ffp-contract=fast default on hipcc (LOW).** cuBQL is integer-Morton + box math; a 1-ULP FMA drift
  could only matter if a sample does a 0-tolerance float compare (the samples print/inside-outside-count,
  not bit-exact gold). If a numeric sample diverges, pin `-ffp-contract=on` in the HIP flags. Not expected
  to gate.
- **Rule-of-five / OOB-neighbor / texture-pitch fault classes: N/A** (no RAII GPU handles held across
  frames, no textures/surfaces, no stencil neighbor reads in cuBQL).
- **CUDA-path byte-identity / no-regression:** every change is under `__HIPCC__` (source) or
  `CUBQL_USE_HIP`/`LANGUAGE HIP` (CMake). With `CUBQL_USE_HIP=OFF` the preprocessor output and the CUDA
  CMake graph are unchanged, so the NVIDIA build is byte-identical. Verify by configuring once WITHOUT
  USE_HIP and confirming the cuda targets/flags are untouched (a no-regression gate; full nvcc compile-only
  if a CUDA box is reachable, else preprocessor/AST diff of the touched headers with `__HIPCC__` undefined).

## File-by-file change list (all additive / HIP-guarded)

- `CMakeLists.txt` (root): add `option(CUBQL_USE_HIP ...)`; when ON, skip CUDA detect, `enable_language(HIP)`,
  set `CUBQL_HAVE_CUDA ON`, default `CMAKE_HIP_ARCHITECTURES` to gfx90a only when unset; guard the
  subproject `CMAKE_CUDA_ARCHITECTURES` FATAL_ERROR for the HIP case.
- `cuBQL/CMakeLists.txt`: in `add_specific_instantiation`, when `CUBQL_USE_HIP`, set the `.cu` source
  `LANGUAGE HIP` and skip the CUDA-only `set_target_properties`. Keep `-DCUBQL_HAVE_CUDA=1`.
- `samples/CMakeLists.txt` + `samples/s01_closestPoint_points_gpu/CMakeLists.txt`: mark the sample `.cu`
  `LANGUAGE HIP` under `CUBQL_USE_HIP` (the `if (CUBQL_HAVE_CUDA)` stanzas otherwise unchanged).
- `cuBQL/math/common.h`: inside the existing `#if defined(__HIPCC__)` block, add the `cudaXxx -> hipXxx`
  alias set (the runtime symbol shim). New file content => add the AMD copyright parallel line + author.
- `cuBQL/builder/cuda.h`: widen the async-resource guard to `defined(__HIPCC__) || CUDART_VERSION >= 11020`.
  (cudaMemPool_t / DeviceGetDefaultMemPool / MemPoolSetAttribute resolve via the shim.)
- Possibly `builder/cuda/builder_common.h`: only if hipCUB needs a `__syncthreads()` after a reused-temp
  sort (add ONLY if validation shows a wave64 miscompute; do not pre-emptively churn).

Attribution: add `Copyright (c) 2026 Advanced Micro Devices, Inc.` parallel line + `Jeff Daily` author tag (project house style: SPDX header at top) to any file we substantively
extend (common.h shim, cuda.h). Trivial CMake flag edits need neither.

## Build commands (gfx90a)

    cmake -S projects/cuBQL/src -B projects/cuBQL/src/build-hip \
      -DCUBQL_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++   # if CMake does not auto-find hipcc
    cmake --build projects/cuBQL/src/build-hip -j --target \
      cuBQL_cuda_float3 cuBQL_sample01_points_closestPoint_cuda sample02_distanceToTriangleMesh \
      sample03_insideOutside

Multi-arch correctness build (do at lead bringup so followers need no source change):
    cmake ... -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100"
    llvm-objdump --offloading <built .so/exe>   # expect BOTH gfx90a and gfx1100 code objects

CUDA no-regression configure (must stay byte-identical):
    cmake -S projects/cuBQL/src -B build-cuda   # USE_HIP defaults OFF; confirm cuda targets unchanged

## Test plan

cuBQL ships **samples, not a gtest/ctest suite** (root CMake `#add_subdirectory(testing)` is commented out).
The samples ARE the validation harness; they self-generate data, so no dataset download is needed (good,
given host egress limits).

Real-GPU validation on gfx90a (each must run and produce correct, deterministic output):
- `cuBQL_sample01_points_closestPoint_cuda` (BinaryBVH, radix builder, points::findClosest) -- exercises
  build + traverse end-to-end; prints closest-point per query. Cross-check the printed closest IDs are the
  true nearest by brute force for the 20 queries (sample is small/deterministic).
- `cuBQL_sample01_points_closestPoint_wideBVH_cuda` -- WideBVH (collapse) path.
- `sample02_distanceToTriangleMesh`, `sample03_insideOutside`, `sample04_boxOverlapsOrInsideSurfaceMesh`
  -- triangle queries; need a mesh input (use a generated/bundled OBJ; s03/s04 produce a numeric
  inside/outside count that is a strong correctness signal).
- Exercise multiple builders explicitly (radix is default; force SAH and rebinRadix and ELH via BuildConfig
  in s02/s07) so each `instantiate_builders.cu`-generated builder is actually run, plus
  `sample07_aggregateNBody` (CUDA_SEPARABLE_COMPILATION path).
Determinism/cross-arch: same seed must give identical results lead vs follower; a divergence is the
warp/temp-race or zero-init bug, not noise.

Non-GPU regression set (must not break with USE_HIP=OFF AND on the CPU builders):
- CPU sample `samples/s01_closestPoint_points_cpu` (closestPoint.cpp, cpu/spatialMedian builder) builds and
  runs host-only -- unaffected by the HIP path; confirm still builds. The OMP builder (`CUBQL_OMP`) path is
  likewise untouched.

## "## Install as a dependency" plan (for barney to consume)

cuBQL is consumed as a CMake subproject (its CMake explicitly supports `CUBQL_IS_SUBPROJECT`). The porter
must fill notes.md `## Install as a dependency` with BOTH consumption modes:
1. **add_subdirectory (submodule pin)** -- barney sets, BEFORE `add_subdirectory(cuBQL)`:
   `set(CUBQL_USE_HIP ON)` and `set(CMAKE_HIP_ARCHITECTURES <arch>)`, links `cuBQL` (interface, headers)
   plus the needed `cuBQL_cuda_<type><dim>` instantiation target (e.g. `cuBQL_cuda_float3`). The subproject
   FATAL_ERROR guard fix (above) is REQUIRED so a HIP consumer is not forced to set a CUDA arch.
2. **installed-tree mode** -- since most of cuBQL is header-only + one instantiation TU, barney can also be
   pointed at an installed copy via `-DCMAKE_PREFIX_PATH=.../_deps/cuBQL/install`. cuBQL currently has no
   `install()` rules; if barney needs installed-tree consumption, that is a small additive follow-up (export
   the `cuBQL` INTERFACE target + headers). Default plan: barney consumes via `add_subdirectory` against
   AMD-Ecosystem/cuBQL @ moat-port into `_deps/cuBQL/` (gitignored), matching MOAT's deps-first workflow.
Record the exact barney CMake lines + the `cuBQL_cuda_*` target names actually linked once built.

## Open questions / unknowns

1. Does upstream want the CMake `USE_HIP` wiring as a PR, or is Wald already drafting it? He merged the
   SOURCE-side HIP cleanup (PR #34) one day before our scaffold and is clearly active. Before opening any
   upstream PR, re-run `gh pr list --repo NVIDIA/cuBQL --search "HIP OR ROCm OR AMD"` to avoid duplicating
   an in-flight CMake PR; coordinate the delta (the CMake path + runtime shim) rather than racing it.
2. Does hipCUB `DeviceRadixSort` over cuBQL's key types (uint32/uint64 morton keys, int/longlong) match
   CUB result ordering bit-for-bit? Expected yes (radix sort is total-order deterministic); confirm via the
   closest-point cross-check.
3. hipMemPool default-pool support on every target arch (gfx90a confirmed-path; verify gfx1100/gfx1151 at
   follower time -- RDNA/Windows HIP SDK pool support may differ; fallback to hipMalloc resource is ready).
4. Whether barney needs an `install()` export or only `add_subdirectory` (resolve when porting barney).
