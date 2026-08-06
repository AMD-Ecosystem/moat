# cudaKDTree port plan (linux-gfx90a lead)

Upstream: https://github.com/ingowald/cudaKDTree (branch `master`, HEAD 0e174fe).
A header-only CUDA k-d tree library (build + fcp/knn queries) plus sample/test
`.cu` executables. Apache-2.0.

## Existing AMD support assessment

None. No `hip`, `rocm`, `__GFX`, `amdgcn`, `USE_HIP`, or HIP runtime anywhere in
the tree; CI (`.github/workflows/{Ubuntu,Windows}.yml`) is CUDA-only (nvcc, CUDA
12.x). This is a genuine first-time CUDA->HIP port, not an already-supported repo
and not an abandoned-port finish. Disposition: PORT.

## Build classification

Pure CMake (`project(... LANGUAGES C CXX)` + `enable_language(CUDA)`), not a
pytorch extension (no `find_package(Torch)`, no `torch.utils.cpp_extension`).
=> Strategy A: `enable_language(HIP)`, one `cuda_to_hip.h` compat
header, mark `.cu` sources `LANGUAGE HIP`, `HIP_ARCHITECTURES gfx90a`. Keep the
CUDA build byte-for-byte unchanged; guard every divergence with `USE_HIP`.

## CUDA surface (the whole thing)

- Host runtime API (18 distinct symbols): `cudaMalloc/Free`, `cudaMallocManaged`,
  `cudaMallocAsync/cudaFreeAsync`, `cudaMemcpy` + `cudaMemcpy{Default,HostToDevice,
  DeviceToHost}`, `cudaStream_t`, `cudaStreamSynchronize`, `cudaDeviceSynchronize`,
  `cudaError_t`, `cudaSuccess`, `cudaGetLastError`, `cudaGetErrorString`,
  `cudaGetSymbolAddress` (stats path only). All have 1:1 hip spellings.
- Headers: `<cuda_runtime.h>`, `<cuda.h>`, `<math_constants.h>` (in common.h),
  `<cuda_runtime.h>` (in cubit.h).
- Thrust: `builder_thrust.h` (the DEFAULT builder) uses `thrust::{sort, device_vector,
  zip_iterator, fill, make_tuple, raw_pointer_cast, device.on(stream)}`. ROCm ships
  rocThrust at `/opt/rocm/include/thrust` with the identical `thrust::` API and the
  same header paths, so these compile unchanged on HIP -- no source swap needed.
- CUB: none directly (cubit is a hand-rolled bitonic sorter, not cub).
- Device intrinsics: `__clzll` (helpers.h levelOf, HIP-supported), atomics
  `atomicMin/atomicMax/atomicAdd` (spatial-kdtree.h, HIP-supported). No `__shfl*`,
  no `__ballot`, no `__popc/__any/__all`, no `__activemask`, no `__syncwarp`
  anywhere in the tree.
- Textures / surfaces: NONE (the two "texture"/"surface" string hits are a comment
  and a doc-string). So the texture fault classes (pitch, linear-filter, coherency)
  do NOT apply here.
- `__constant__ g_traversalStats` pointer + `cudaGetSymbolAddress` only under
  `CUKD_ENABLE_STATS`; HIP supports both.
- Version guard: `helpers.h:52` `#if CUDART_VERSION >= 11020` selects the
  cudaMallocAsync allocator; `CUDART_VERSION` is undefined on HIP. hip DOES provide
  `hipMallocAsync/hipFreeAsync`, so extend the guard to keep the async path on HIP.

## Warp-size analysis (the MOAT high-risk class) -- NO hazard here

gfx90a is wave64. The only fixed-32 code is `cukd/cubit/*` (a shared-memory bitonic
sorter, used only by `builder_bitonic`, but always compiled because `builder.h`
includes it). Its `threadIdx.x & -32`, `^ (32-1)`, `l + 32` are bitonic-merge index
arithmetic over a `__shared__` array sized `2*1024` with explicit `__syncthreads()`
at every stage boundary >= 32 -- they are NOT lane masks and use no warp intrinsics.
The sub-32 down-sweeps (seq sizes 4/8/16) omit `__syncthreads()` and rely on
warp-synchronous execution of a <=32-thread sub-range; a 64-wide wavefront executes
all 64 lanes in lockstep, so any <=32-lane sub-group is a fortiori lockstep -- the
assumption strengthens, it does not break (the dangerous direction is assuming MORE
than warpSize threads run in lockstep, which never happens here). Query kernels
(`fcp.h`, `knn.h`) are per-thread traversals with per-thread candidate lists and no
cross-lane communication. Conclusion: correctness is wave-size-independent; no
ballot/shuffle/popsift-style fix required. We will still PROVE this on-GPU with the
built-in brute-force verifier across builders.

## Port steps (minimal footprint)

1. Add `cukd/cuda_to_hip.h`: on `USE_HIP`/`__HIP_PLATFORM_AMD__`, include `<hip/hip_runtime.h>`
   and alias only the symbols above; otherwise no-op include the CUDA runtime
   headers. Single file that knows about HIP.
2. Replace the direct `<cuda_runtime.h>`/`<cuda.h>`/`<math_constants.h>` includes in
   `cukd/common.h` and `cukd/cubit/cubit.h` (and the bare `cudaMallocManaged` call
   sites in the sample/test `.cu`) with `#include "cukd/cuda_to_hip.h"`.
3. `helpers.h:52`: `#if CUDART_VERSION >= 11020 || defined(__HIP_PLATFORM_AMD__)` so
   HIP keeps the async allocator (`cudaMallocAsync` -> `hipMallocAsync` via header).
4. `CMakeLists.txt` (+ `samples/CMakeLists.txt`, `testing/CMakeLists.txt`):
   `option(USE_HIP)`; when ON, `enable_language(HIP)`, set every `.cu` target's
   sources `LANGUAGE HIP`, set `HIP_ARCHITECTURES`, `target_compile_definitions(... USE_HIP)`,
   and skip the nvcc-only `-Xcompiler` OpenMP flags. When OFF, the existing CUDA path
   is untouched.
5. Record quirks in notes.md; append generalizable lessons (rocThrust drop-in;
   shared-memory bitonic `& -32` is wave-agnostic) to PORTING_GUIDE Changelog.

## Build & validation

- Configure: `cmake -S . -B build-hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DBUILD_ALL_TESTS=ON`.
- Validate on a free gfx90a GPU (pick via rocm-smi, export HIP_VISIBLE_DEVICES):
  - `testing/floatN-knn-and-fcp.cu` has a built-in `-v` CPU brute-force verifier
    (`verifyFCP` = brute-force nearest; `verifyKNN` = std::priority_queue k-NN; rel
    err <= 1e-6) plus `checkTree` recursive k-d-tree-invariant check. `BUILD_ALL_TESTS`
    builds the full matrix {dim 2,3,4,8} x {stackBased,stackFree,cct} x {fcp,knn} x
    {regular,explicit-dim,spatial}.
  - Run `-v` across several k (4,8,20,50,64) and point counts (e.g. 1k, 100k, 1M),
    both uniform and `--clustered`; require "verification succeeded".
  - Run the CTest builder unit tests (empty/simple input, same-result-across-builders).
  - Repeat a query run to confirm deterministic CHECKSUM across runs.
- Marking `ported` requires a genuine correct GPU run, per CLAUDE.md; lint/compile is
  not validation.
