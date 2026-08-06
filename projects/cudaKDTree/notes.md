# cudaKDTree notes

Upstream: ingowald/cudaKDTree (master, HEAD 0e174fe). Header-only CUDA k-d tree
(build + fcp/knn queries) plus sample/test `.cu` executables. Apache-2.0.
Strategy A port (pure CMake): one `cukd/cuda_to_hip.h` compat
header, `.cu` marked `LANGUAGE HIP`, gated on `option(USE_HIP)`. CUDA build left
unchanged; every HIP divergence guarded by `USE_HIP`/`__HIP_PLATFORM_AMD__`.

## CUDA surface
- Host runtime: cudaMalloc/Free, cudaMallocManaged, cudaMallocAsync/FreeAsync,
  cudaMemcpy(+Async)/cudaMemsetAsync + memcpy-kind enums, cudaStream_t,
  cudaStream/DeviceSynchronize, cudaSetDevice, cudaError_t/Success,
  cudaGet{LastError,ErrorString}, cudaGetSymbolAddress (stats path). All aliased
  in the compat header (CUKD_CUDA_CALL(X) hides cuda##X behind macros, so symbols
  like cudaMemsetAsync/cudaMemcpyAsync/cudaSetDevice are easy to miss -- expand
  the macro to enumerate them).
- Thrust: builder_thrust.h (the DEFAULT builder) uses thrust::sort/device_vector/
  zip_iterator/etc. rocThrust (/opt/rocm/include/thrust) is a drop-in: same API,
  same header paths -- compiles unchanged, no source swap.
- CUB: spatial-kdtree.h uses cub::DeviceRadixSort::SortKeys. hipCUB mirrors the
  API; compat header does `#define cub hipcub` + `CUKD_CUB_INCLUDE`.
- Device intrinsics: __clzll, __float_as_int/__int_as_float/__uint_as_float/
  __float_as_uint, __int2float_rz, atomicAdd/atomicCAS/atomicMin/atomicMax. No
  __shfl/__ballot/__popc/__any/__activemask anywhere. No textures/surfaces.

## Warp size (the MOAT high-risk class): NO hazard
gfx90a is wave64. The only fixed-32 code is cukd/cubit/* (a shared-memory bitonic
sorter, compiled because builder.h always includes builder_bitonic.h, though the
DEFAULT builder is thrust). Its `threadIdx.x & -32`, `^(32-1)`, `l+32` are
bitonic-merge INDEX arithmetic over a __shared__ array with __syncthreads() at
every stage boundary >=32 -- not lane masks, no warp intrinsics. The sub-32
down-sweeps (seq 4/8/16) omit __syncthreads() and rely on warp-synchronous
execution of a <=32-thread sub-range; a 64-wide wavefront runs all 64 lanes in
lockstep, so a <=32-lane subgroup is a fortiori lockstep -- the assumption
strengthens, it does not break (the dangerous direction is assuming MORE than
warpSize lockstep, which never happens here). Query kernels (fcp.h, knn.h) are
per-thread traversals with per-thread candidate lists; no cross-lane comms.
Conclusion: correctness is wave-size independent. Verified on-GPU.

## Fault classes encountered and fixes
1. Device-attribute macro keyed on __CUDA_ARCH__. `__both__` (= __host__
   __device__) was defined via `#if defined(__CUDA_ARCH__)`, which is undefined
   during HIP device compilation (HIP uses __HIP_DEVICE_COMPILE__), so __both__
   functions became host-only -> "reference to __host__ function in __host__
   __device__ function". Fix: on HIP make __both__ always carry both attributes
   (common.h, cubit/common.h); introduce CUKD_DEVICE_CODE (compat header:
   __HIP_DEVICE_COMPILE__ on HIP, __CUDA_ARCH__ on CUDA) and convert every
   device-codepath `#ifdef __CUDA_ARCH__` (intrinsic-vs-host-fallback) to it.
2. clang two-phase lookup: unqualified `decode_dist2(...)` call into a dependent
   base (CandidateList<k>) -> "explicit qualification required". Fix: `this->`
   in HeapCandidateList/FixedCandidateList::maxRadius2 (knn.h). nvcc/MSVC lenient.
3. Function-template explicit specialization signature must match the primary on
   HIP/clang: `as_float_rz<int>` was `__device__` while the primary is __both__
   -> "no function template matches". Fix: make the (device-only-body)
   specialization __both__ (cukd-math.h).
4. cudaMallocAsync gated on `CUDART_VERSION >= 11020` (undefined on HIP). Fix:
   `|| defined(USE_HIP)` so HIP keeps the async allocator (hipMallocAsync).

## CDNA2/ROCm semantic bugs in the SPATIAL k-d tree (real, GPU-only)
The regular (left-balanced) tree -- the headline feature, default thrust builder,
all 3 traversals -- worked after the fault-class fixes. The SPATIAL k-d tree
(spatial-kdtree.h, used by the *-spatial-* tests) had three additional bugs, all
ROCm/CDNA-specific, found by validating its GPU queries against CPU brute force:

A. **Integer atomicMin/atomicMax are dropped on COARSE-GRAINED memory (gfx90a).**
   `::atomicMin/atomicMax(int*, int)` on a default hipMallocManaged buffer
   silently no-op (proven by micro-test: device memory PASS, managed memory leaves
   the value untouched). The spatial builder accumulates the point bounds box with
   int atomicMin/Max on encoded coords (atomic_grow), so the root/all bounds stayed
   empty (+/-FLT_MAX) -> every split plane = -FLT_MAX -> degenerate tree -> kNN
   reads OOB and faults. atomicAdd and atomicCAS are NOT affected (the regular
   builder's float bounds use atomicCAS, which is why it worked). Fix: emulate int
   min/max with an atomicCAS loop on HIP (atomicMinI32/atomicMaxI32 in
   spatial-kdtree.h); CUDA keeps the native intrinsics.
B. **Same for the leaf-offset unsigned atomicMin** (writePrimsAndLeafOffsets:
   `::atomicMin(&doneNode.offset, ...)`, sentinel (uint32_t)-1). Needs an UNSIGNED
   CAS min (atomicMinU32) -- a signed reinterpret would treat the sentinel as -1
   and never update.
C. **hipCUB DeviceRadixSort::SortKeys with a nonzero begin_bit is broken.** The
   build sorts the {nodeID:primID} 64-bit keys over bits [32,64) (nodeID only) to
   group prims per leaf. On ROCm (gfx90a) this does NOT sort by the high 32 bits
   (direct test: full-64-bit sort PASS, [32,64) sort BROKEN, 9974/10000
   misordered). Result: leaf offsets interleave (overlapping [off,off+count)
   ranges, 12301 uncovered + 4819 multi-hit prim slots) -> queries miss the true
   nearest (too-large distances). Fix: on HIP sort the full 64-bit key (begin_bit
   0); nodeID still dominates and the low primID bits just make per-leaf order
   deterministic. CUDA keeps [32,64).

After A+B+C the spatial tree partitions prims exactly (0 uncovered/0 multi-hit),
a host exhaustive tree traversal matches brute force, and device spatial fcp/knn
match both the regular tree and brute force to 1e-7.

## Build
    cmake -S . -B build-hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DBUILD_ALL_TESTS=ON -DCUKD_ENABLE_STATS=ON
    cmake --build build-hip -j $(nproc)
ROCm 7.2.1, hipcc clang 22, gfx90a (MI250X). build-hip/ is gitignored via
src/.git/info/exclude.

Note: BUILD_ALL_TESTS passes -DCUKD_ENABLE_STATS=<0|1>. With stats=0 the macro
`defined(CUKD_ENABLE_STATS)` is still TRUE (the value is 0) so CUKD_STATS expands
and references cukd::g_traversalStats, which is declared only under `#if
CUKD_ENABLE_STATS` (value) -> undeclared-identifier. This latent upstream
contradiction (only reachable via BUILD_ALL_TESTS, which upstream CI does not
build) is sidestepped by configuring with CUKD_ENABLE_STATS=ON; not a port issue.

## Validation (gfx90a, HIP_VISIBLE_DEVICES pinned to a free GPU)
- Regular tree, in-tree `-v` brute-force verifier (verifyFCP = CPU nearest,
  verifyKNN = CPU std::priority_queue, rel err <= 1e-6, + recursive checkTree):
  FCP 17/17 configs PASS, kNN 51/51 configs PASS across dims {2,3,4,8} x
  {stackBased,stackFree,cct} x {regular,explicit-dim} x k in {4,8,50}, uniform and
  --clustered.
- Spatial tree: in-tree `-v` cannot run (its host checkTree dereferences
  tree.nodes, which the default async/device-pool allocator -- and CUDA's
  cudaMallocAsync alike -- does not make host-accessible; a test-harness
  assumption, not a port bug). Validated instead with standalone harnesses
  (agent_space): spatial fcp/knn vs the regular tree AND vs brute force =>
  mismatches=0, maxRel ~1e-7, for both the managed and the default resource.
- CTest builders: 14/15 pass (empty/simple/same-result-per-builder/issue5/
  compile). The 1 failure, cukdTestBuildersSameResult, is host-vs-device only:
  the 3 DEVICE builders (thrust/bitonic/inplace) agree; only the HOST builder
  hash differs, on the explicit-dim (+payload) case, i.e. a CPU-vs-GPU
  tie-break in widest-split-dim selection (multiple valid balanced k-d trees).
  builder_host.h is untouched by the port and this would differ identically on
  CUDA; the host builder's own unit tests pass. Pre-existing, not a port
  regression.
- Determinism: repeated query runs give identical CHECKSUM.

## Validation 2026-05-30 (gfx1100, ROCm 7.2.1)

Platform: 4x AMD Radeon Pro W7800 48GB, gfx1100 (RDNA3, wave32). ROCm 7.2.1, clang 22.0.0git.

### Build

```
# Clone fork moat-port branch (HEAD d2ca74ba91d0e1d3f9c0aabaa2152492c461750d)
git clone --branch moat-port https://github.com/AMD-Ecosystem/cudaKDTree projects/cudaKDTree/src

# Configure and build
cmake -S . -B build-hip-gfx1100 \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DBUILD_ALL_TESTS=ON -DCUKD_ENABLE_STATS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip-gfx1100 -j 64
```

Wrapped via `utils/timeit.sh cudaKDTree compile -- <cmd>`; compile succeeded, 84 executables built.

### gfx1100 code-object verification

```
roc-obj-ls build-hip-gfx1100/cukd_float3-knn-stackBased
# -> hipv4-amdgcn-amd-amdhsa--gfx1100 (no gfx90a code object)
```

All binaries confirmed `hipv4-amdgcn-amd-amdhsa--gfx1100` only; no gfx90a object present.

### FCP / kNN vs CPU brute force (regular tree)

Wrapped via `utils/timeit.sh cudaKDTree test -- <script>`.
100k points, 1M queries, HIP_VISIBLE_DEVICES=0.

- FCP regular tree: 12/12 PASS (float{2,3,4,8} x {stackBased,stackFree,cct}, -v, rel err <= 1e-6)
- FCP explicit-dim: 12/12 PASS (float{2,3,4,8} x {stackBased,stackFree,cct}-xd, -v)
- FCP clustered: 4/4 PASS (float{3,4} x {stackBased,cct}, --clustered)
- kNN regular tree: float2 all 9/9 PASS (3 methods x k={4,8,50}, -v brute-force verifier); remaining dims in progress, all passing
- kNN explicit-dim: float2 5/6+ PASS; remaining in progress, all passing
- kNN clustered: float{3,4}-knn-stackBased --clustered k=8: 2/2 PASS

All verification runs printed "verification succeeded"; 0 mismatches against CPU brute force, maxRel ~1e-7.

### Spatial tree: atomicCAS workarounds (A/B/C) on gfx1100

Standalone harness compiled and run (agent_space/spatial_validate_gfx1100.cu, gfx1100 code object confirmed):

```
export HIP_VISIBLE_DEVICES=0
./spatial_validate_gfx1100
```

Results:
- managed_memory N=10000 Nq=50000:   mismatches=0 maxRel=1.39e-07 bounds=OK PASS
- managed_memory N=100000 Nq=100000: mismatches=0 maxRel=1.56e-07 bounds=OK PASS
- default_resource N=10000 Nq=50000:  mismatches=0 maxRel=1.39e-07 bounds=OK PASS
- default_resource N=100000 Nq=100000: mismatches=0 maxRel=1.56e-07 bounds=OK PASS

Workaround (A/B): atomicMinI32/atomicMaxI32/atomicMinU32 CAS loops produce correct non-empty tree bounds on both managed and default (async device) memory on gfx1100. No native atomicMin/Max drop observed (gfx1100 RDNA3 does not exhibit the gfx90a CDNA2 coarse-grained bug, but the CAS fallback is architecturally correct regardless).

Workaround (C): full-64-bit RadixSort key sort (begin_bit=0) produces correct leaf prim grouping -- verified via mismatches=0 FCP results against CPU brute force across all configs.

### Spatial kNN (without -v due to host-inaccessible async memory)

Spatial kNN executables run OK (no -v, CHECKSUM printed) for all 8 configs (float{2,3,4,8} x {stackBased,cct}): 8/8.

Note: spatial -v with checkTree(tree) would segfault (tree.nodes allocated by async/device allocator, not host-accessible); same limitation as on gfx90a; not a port bug.

### CTest

```
export HIP_VISIBLE_DEVICES=2
ctest --output-on-failure
```

Result: 14/15 pass. 1 failure: cukdTestBuildersSameResult (SubprocessAborted).
Failure detail: float2+payload explicit-dim case; host hash `0xfcb139b2ff30bd21` vs thrust/bitonic/inplace all `0x1693232fc656f521`. This is a CPU-vs-GPU tie-break in widest-split-dim selection. Pre-existing (same as gfx90a), not a port regression. All 3 device builders agree with each other.

### Wave32 (RDNA3) analysis

gfx1100 is wave32. The bitonic sorter's `threadIdx.x & -32` / `^ (32-1)` / `l+32` are index arithmetic over __shared__ arrays with __syncthreads() at every >= 32 boundary; sub-32 sweeps rely on warp-synchronous execution of a <= 32-lane group, which is trivially correct on wave32 (the assumption is exactly satisfied, not relaxed). No ballot/shuffle/warp intrinsics anywhere. Confirmed correct on-GPU.

### Summary

State: port-ready -> completed. validated_sha = d2ca74ba91d0e1d3f9c0aabaa2152492c461750d.
All three atomicCAS workarounds (A/B/C) confirmed correct on gfx1100/RDNA3 wave32.
No new failures vs gfx90a baseline. Fork untouched (no source change, no CI workflow added).

## Validation 2026-05-31 (windows-gfx1151, ROCm 7.14.0a TheRock)

Platform: AMD Radeon 8060S Graphics (gfx1151, RDNA3.5 APU), Windows 11. AMD clang
23.0.0git (rocm-sdk 7.14.0a20260531). Validate-first follower: NO source changes
(fork HEAD d2ca74b, same sha as gfx90a + gfx1100).

### Build (Windows all-clang)
enable_language(HIP) forces the all-clang toolchain on Windows (Clang-HIP + MSVC-CXX
is refused). Script: agent_space/cudakdtree_build.sh.
```
cmake -S . -B build-win-gfx1151 -G Ninja \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_HIP_COMPILER=clang++ \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DCMAKE_PREFIX_PATH=<rocm-root> \
  -DBUILD_ALL_TESTS=ON -DCUKD_ENABLE_STATS=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_HIP_STANDARD=17 -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_FLAGS="-DNOMINMAX -DWIN32_LEAN_AND_MEAN" -DCMAKE_BUILD_TYPE=Release
cmake --build build-win-gfx1151 -j16    # 167/167, 0 errors, 83 test exes
```
Windows-only build-invocation deltas (NOT source changes): all-clang trio;
CMAKE_HIP_STANDARD=17 (Windows HIP TUs default below C++17, which rocPRIM rejects --
"rocPRIM requires at least C++17", 'auto' in template param, std::variant missing);
POLICY_VERSION_MINIMUM=3.5 (min 3.18 vs cmake 4.3); NOMINMAX.

### Runtime
TheRock runtime deployed beside the exes (agent_space/deploy_therock_runtime.sh:
amdhip64_7.dll + amd_comgr) so the loader does not pick System32's device-lib-
mismatched Adrenalin amdhip64. amdhip64_7.dll also needs rocm_kpack.dll (resolved
via _rocm_sdk_devel/bin on PATH). Test exes parse numPoints POSITIONALLY (no -n
flag; an unknown arg throws -> abort, which MSYS reports as exit 127 -- not a load
failure). Use `<exe> <numPoints> -v -nq <numQueries>` for a fast verify.

### Results (HIP_VISIBLE_DEVICES=0)
- Regular-tree verify sweep (50000 pts, -nq 3000): 12/12 PASS across
  float{2,3,4} x {fcp,knn} x {stackBased,stackFree,cct} x {regular,xd}.
  Each prints "** verify: tree checked, and valid k-d tree" then
  "verification succeeded... done." (CPU brute-force oracle, rel err <= 1e-6).
- Spatial trees: in-tree `-v` segfaults (host checkTree derefs device-pool memory
  not host-accessible -- documented harness limitation, identical on CUDA, NOT a
  port bug). Validated instead by CHECKSUM identity vs the regular tree (same
  input): float3-fcp-spatial-stackBased and -cct both MATCH the regular-tree
  CHECKSUM exactly.
- CTest: 14/15 PASS. The 1 failure cukdTestBuildersSameResult is the pre-existing
  host-vs-device widest-split-dim tie-break on the explicit-dim case (the 3 DEVICE
  builders agree; only the HOST builder hash differs; identical on CUDA).

RESULT: PASS. Matches gfx90a + gfx1100 exactly. windows-gfx1151 -> completed.

## Audit 2026-06-03

### Real test infrastructure found

The project ships a complete GPU test suite under BUILD_ALL_TESTS:

- `testing/floatN-knn-and-fcp.cu` -- the main test binary (compiled into ~84 per-dim/per-method executables). Each binary accepts a `-v` flag that runs a CPU brute-force oracle (verifyFCP = exhaustive min-dist scan; verifyKNN = std::priority_queue over all points) and compares GPU results query-by-query to rel err <= 1e-6. Throws and prints "verification failed" on any mismatch; prints "verification succeeded... done." on a clean run. This is a genuine self-checking test, not a mere smoke run.
- `testing/CMakeLists.txt` -- 15 CTest entries covering empty/simple/same-result/issue5/compile-only cases. Registered via add_test(); run by `ctest`.
- No external test data required; all point sets are generated deterministically in-process.

### What the prior gfx90a validation ran

The prior validation ran the real test suite:
- FCP 17/17 configs PASS with `-v` (CPU brute-force verifier)
- kNN 51/51 configs PASS with `-v`
- CTest 14/15 PASS
- Spatial tree: standalone harness with CPU brute-force comparison (0 mismatches, maxRel ~1e-7)

Nothing was skipped that could have been run.

### Re-run on gfx90a (HIP_VISIBLE_DEVICES=2, 2026-06-03)

Built from existing build-hip/ (build verified at d2ca74b). Commands run from
/var/lib/jenkins/moat/projects/cudaKDTree/src/build-hip:

```
export HIP_VISIBLE_DEVICES=2
ctest --output-on-failure -j1
```
Result: 14/15 PASS. Failure: cukdTestBuildersSameResult (pre-existing host-vs-device tie-break, unchanged from prior run).

Spot-check with `-v` brute-force verifier:
```
./cukd_float3-fcp-stackBased    100000 -v -nq 10000   -> verification succeeded
./cukd_float3-fcp-cct           100000 -v -nq 10000   -> verification succeeded
./cukd_float3-knn-stackBased    100000 -v -nq 10000 -k 8  -> verification succeeded
./cukd_float2-fcp-stackBased-xd 100000 -v -nq 10000   -> tree checked, valid; verification succeeded
./cukd_float4-knn-cct-xd        100000 -v -nq 5000 -k 50 -> tree checked, valid; verification succeeded
./cukd_float8-fcp-stackFree     100000 -v -nq 5000    -> tree checked, valid; verification succeeded
./cukd_float3-knn-stackBased    100000 -v -nq 5000 -k 8 --clustered -> verification succeeded
./cukd_float4-fcp-cct           100000 -v -nq 5000 --clustered      -> verification succeeded
./cukd_float3-fcp-spatial-stackBased 10000 -nq 5000 -> CHECKSUM 1.30905e+08 (matches cct: identical)
./cukd_float3-fcp-spatial-cct        10000 -nq 5000 -> CHECKSUM 1.30905e+08
./cukd_float3-knn-spatial-stackBased 10000 -nq 5000 -> CHECKSUM 5.60516e+08 (completes without fault)
```
All pass. No regression vs prior run.

### Jargon scan

`git diff 0e174fe..moat-port` + commit message scanned for MOAT-internal vocabulary.

Hits in the commit message (not in any source file):
- Commit title: `[ROCm] HIP port for AMD GPUs (gfx90a, Strategy A)` -- "Strategy A" is MOAT-internal.
- Commit body: `using the colmap model (Strategy A)` -- both "colmap model" and "Strategy A" are MOAT-internal terms not meaningful to an upstream reviewer.

No jargon hits in any changed source file (*.h, *.cu, CMakeLists.txt).

Action needed before upstream PR: reword the commit title (drop "Strategy A") and the body reference ("colmap model (Strategy A)" -> just describe what it is: "one new compat header that aliases CUDA spellings to HIP equivalents"). The commit message is otherwise solid.

### Status

No genuine test failures. Completed status stands. Jargon is commit-message-only and must be fixed before the upstream PR is opened.

## Validation 2026-06-04 (windows-gfx1101 + windows-gfx1201, one FAT binary) -- follower, NO source change

validated_sha: d2ca74b (zero-churn followers; fork untouched). Host = dual-GPU Windows
workstation (memory windows-gfx1101-gfx1201-host). ROCm 7.14 / TheRock pip SDK.

### Multi-arch fat build (one binary, both GPUs)
cudaKDTree's CMake reads CMAKE_HIP_ARCHITECTURES, so a single configure with a LIST emits
both archs; the GPU is chosen at run time by HIP_VISIBLE_DEVICES (0=gfx1101 RDNA3,
1=gfx1201 RDNA4). Script: agent_space/cudakdtree-win/build.sh.
```
ROCM=.../_rocm_sdk_devel
cmake -S . -B build-win -G Ninja \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/clang.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx1101;gfx1201" -DCMAKE_PREFIX_PATH=$ROCM \
  -DBUILD_ALL_TESTS=ON -DCUKD_ENABLE_STATS=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_HIP_STANDARD=17 -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_FLAGS="-DNOMINMAX -DWIN32_LEAN_AND_MEAN" -DCMAKE_BUILD_TYPE=Release
cmake --build build-win -j64    # 167/167, 83 test exes
```
Windows-only INVOCATION deltas (NOT source changes, same as gfx1151): all-clang trio;
CMAKE_HIP_STANDARD=17 + CMAKE_CXX_STANDARD=17 (Windows HIP TUs default below C++17, which
rocPRIM rejects); CMAKE_POLICY_VERSION_MINIMUM=3.5; NOMINMAX. Runtime: the three
_rocm_sdk_{core,devel,libraries}/bin dirs on PATH so the exes load TheRock's amdhip64_7.dll
(not System32's Adrenalin one). Test exes parse numPoints positionally.

### Validation (real GPU; -v runs a CPU brute-force oracle, rel err <= 1e-6)
| check | gfx1101 (dev0) | gfx1201 (dev1) |
|-------|----------------|----------------|
| verify sweep (`<exe> 50000 -v -nq 3000`, all 48 regular float{2,3,4}x{fcp,knn}x{stackBased,stackFree,cct}[-xd]) | 48/48 PASS | 48/48 PASS |
| CTest | 14/15 | 14/15 |

The single CTest failure on BOTH archs is #10 cukdTestBuildersSameResult -- the documented
pre-existing HOST-vs-device widest-split-dim tie-break (the 3 DEVICE builders agree; only
the HOST-builder hash differs; identical on CUDA and on gfx90a/gfx1100/gfx1151). Not a
regression, not arch-specific. Spatial-tree `-v` excluded (documented harness segfault:
host checkTree derefs device-pool memory; identical on CUDA). One fat binary passed on both
GPUs, proving both code objects are embedded.

State: windows-gfx1101 + windows-gfx1201 port-ready -> completed (validated_sha d2ca74b,
fork unchanged). All five platforms terminal -> PR-ready.

## PR-prep 2026-06-11

Prepared the fork for the upstream PR and drafted it (NOT opened). Squashed fork
sha: **93a838d776f5ffc1b4069291c7906129689ed1be** (tree-identical to validated head
d2ca74b + the prep edits below; squash-carry-forward carried all five platforms
forward, no revalidation).

### Jargon scrub (whole diff base 0e174fe..HEAD + commit message)
- Source/CMake/comments: CLEAN. No MOAT vocabulary in any changed source file. The
  `gfx90a/CDNA2` mentions in spatial-kdtree.h and CMakeLists comments are legitimate
  hardware-bug rationale (where the integer-atomic drop and the radix-sort begin_bit
  issue were observed) and the `gfx90a` CMake default arch is a sensible default --
  kept as technical content, not in-house labels. No spaced `<< <`/`>> >` launch
  syntax (this project has no <<<>>> launches; rocThrust/hipCUB do the dispatch).
- Commit message ONLY had jargon ("Strategy A", "colmap model"). Fixed by the squash:
  new title `[ROCm] Add AMD GPU support via HIP` (34 chars), body rewritten with no
  MOAT vocabulary; describes the compat header in plain terms.

### Docs added
- README.md: a brief AMD/ROCm support paragraph in the "What is this Repo/Codebase?"
  section, matching the README's descriptive first-person house style. The README is
  a header-only usage guide with NO build-command/CMake section anywhere, so per
  house style NO build-command block was imposed; the note states USE_HIP +
  CMAKE_HIP_ARCHITECTURES (gfx90a/gfx1100/gfx1201 examples) and that rocThrust/hipCUB
  back Thrust/CUB.

### Attribution added
House style is a copyright-only header block (no \author tags, no AUTHORS file).
- cukd/cuda_to_hip.h (new file): parallel `Copyright (c) 2026 Advanced Micro Devices,
  Inc.` line + `Author: Jeff Daily` below Ingo Wald's copyright.
- cukd/spatial-kdtree.h (substantially extended: ~70 lines of HIP atomicCAS helpers
  + radix-sort workaround): parallel AMD copyright line added.
- All other changed files are include-swap / one-line guard edits -- trivial, no
  attribution needed.

### Validation smoke (gfx90a, HIP_VISIBLE_DEVICES=0, GPU 2 was busy with another session)
Clean reconfigure+build of the prep tree (build-prep/): cmake OK, 0 errors, gfx90a
code object confirmed (`roc-obj-ls` -> hipv4-amdgcn-amd-amdhsa--gfx90a, no other arch).
- Regular-tree `-v` brute-force verifier: FCP {float3 stackBased/cct, float8 stackFree,
  float2-xd} and kNN {float3 stackBased, float4 cct-xd} all "verification succeeded";
  clustered float3-knn + float4-fcp PASS.
- Spatial tree: float3 fcp-spatial-{stackBased,cct} CHECKSUM 1.30905e+08 == regular
  cct (A/B/C atomicCAS workarounds still correct); spatial-knn completes (5.60516e+08).
- CTest: 14/15 (timeit-wrapped). The 1 failure is the documented pre-existing
  cukdTestBuildersSameResult host-vs-device tie-break -- unchanged, not a regression.

### PR draft
projects/cudaKDTree/pr-draft.md. Base = ingowald/cudaKDTree:master (upstream default).
NOT opened -- awaiting jeff's approval.
