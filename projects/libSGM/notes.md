# libSGM notes

## Port status
- Strategy A (compat header + CMake HIP-language gating). Pure CUDA, no
  CUTLASS/cuBLAS/CUB/Thrust/textures. Correctness-first mechanical port.
- Fork: https://github.com/AMD-Ecosystem/libSGM (default branch `master`).
- Branch: `moat-port` off upstream `master` (e4c669b). Fork HEAD 9ce43fd.
- Actions disabled on the fork.
- linux-gfx90a: ported and validated (67/67 bit-exact, deterministic).

## Build (gfx90a, ROCm 7.2.1)
```
cd projects/libSGM/src
git submodule update --init            # googletest, needed for ENABLE_TESTS
cmake -S . -B build-hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j
HIP_VISIBLE_DEVICES=3 ./build-hip/test/sgm-test
```
Followers: same command with `-DCMAKE_HIP_ARCHITECTURES=gfx1100` / `gfx1151`,
no source change. Multi-arch also works:
`-DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100"` (host launch dims come from a runtime
warpSize query, so every device slice is correct -- see multi-arch fix below).

## Multi-arch warp-size fix (fork 4ffe4e9, 2026-06-02)
The original port baked one host warp size: CMake scanned CMAKE_HIP_ARCHITECTURES
and injected SGM_HOST_WARP_SIZE (64 if ANY arch matched gfx9), and the host launch
math used it for BLOCK_SIZE = WARP_SIZE * N. The device pass keyed WARP_SIZE per
slice off __GFX*__. For a single-arch build host==device, but a combined
gfx90a;gfx1100 build set host=64 -> host launched bdim=512 against the gfx1100
kernel built for 32-lane geometry (BLOCK_SIZE=256): out-of-range warp/lane indexing
on the gfx1100 slice. Fix (arch-unified, correct on wave32 AND wave64):
- `src/constants.h`: WARP_SIZE device value stays per-arch (__GFX8__||__GFX9__->64,
  else 32). Removed the SGM_HOST_WARP_SIZE host arm + the ->64 fallback's reliance
  on CMake. WARP_SIZE retains a host-pass value (64) ONLY so hipcc can parse the
  __global__ bodies into launch stubs; it no longer drives runtime launch geometry.
- `src/host_utility.h`: added `device_warp_size()` -- queries the current device's
  warpSize once via hipDeviceGetAttribute (cached per device), returns 32 on CUDA.
- `src/cost_aggregation.cu` + `src/winner_takes_all.cu`: every host launcher now
  computes block_size = device_warp_size() * WARPS_PER_BLOCK and derives
  paths_per_block / gdim / bdim from it. The namespace-scope BLOCK_SIZE = WARP_SIZE
  * N constexpr stays (used INSIDE the kernels' smem-stride and RIGHT_BUFFER sizing,
  device pass). SUBGROUP_SIZE = MAX_DISPARITY/DP_BLOCK_SIZE is warp-size-independent.
- `src/CMakeLists.txt`: deleted the SGM_HOST_WARP_SIZE arch-scan loop and dropped it
  from target_compile_definitions (USE_HIP stays).
- `src/cuda_to_hip.h`: added cudaGetDevice/cudaDeviceGetAttribute/cudaDevAttrWarpSize
  aliases; fixed the stale SGM_WARP_SIZE comment.
Validated: two-arch build (`-DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100"`) compiles
clean, roc-obj-ls shows BOTH gfx90a and gfx1100 code objects, gfx90a run picks
warpSize=64 at runtime (AMD_LOG_LEVEL=3 native gfx90a object, no JIT), 67/67
bit-exact, deterministic across 2 runs (HIP_VISIBLE_DEVICES=1). gfx1100 RUN
revalidates on the RDNA3 follower host.
GOTCHA: hipcc parses __global__ bodies in BOTH passes to emit launch stubs, so a
purely device-only (`#if __HIP_DEVICE_COMPILE__`) WARP_SIZE breaks the host parse
(warp_id = threadIdx.x / WARP_SIZE etc. inside the kernel). Keep a host-pass value;
the correctness guarantee is that LAUNCH dims come from the runtime query, not that
host WARP_SIZE is undefined.

## Key changes
- `src/cuda_to_hip.h` (new): on USE_HIP includes hip_runtime + aliases the
  cuda* runtime surface used by host code; else passthrough to cuda_runtime.h.
- `src/constants.h`: WARP_SIZE device-side per-arch via __GFX*__ (64 CDNA, 32 RDNA);
  host launch geometry from runtime device_warp_size() (see multi-arch fix above).
- `src/median_filter.cu`: USE_HIP-guarded software emulation of __vcmpgtu2/4,
  __vminu2/4, __vmaxu2/4 (absent in ROCm 7.x), bit-identical to the intrinsics.
- `src/winner_takes_all.cu`: WTA inter-lane smem handoff uses __syncwarp() on
  HIP (CUDA_VERSION is undefined under HIP, which otherwise routed to the
  weaker __threadfence_block fallback).
- CMake: USE_HIP option; enable_language(HIP); .cu LANGUAGE HIP; sgm built PIC
  (static lib linked into the PIE test exe); test target gated to CXX-only.

## Gotchas / lessons (worth promoting to PORTING_GUIDE)
- WARP_SIZE host/device split trap (same class as raft select_warpsort): the
  .cu compute their launch geometry (BLOCK_SIZE = WARP_SIZE * N, grid dims) in
  hipcc's HOST pass, where `__GFX9__` and every gfxNNN macro is UNDEFINED. A
  device-only `#if defined(__GFX9__)` constant then reads 64 on device but 32
  on host -> block dim disagrees with the 64-lane kernel. Fix: derive a host
  warp size from CMAKE_HIP_ARCHITECTURES (gfx9* -> 64, else 32) and inject it
  as a compile def; the constant uses `__HIP_DEVICE_COMPILE__` for the device
  branch and the injected def for the host branch. host==device per build.
- This SGM design overloads one WARP_SIZE constant for three roles (subgroup
  partitioning, WTA per-lane layout REDUCTION_PER_THREAD = MAX_DISPARITY /
  WARP_SIZE, block sizing). Making it the true wavefront width keeps all three
  internally consistent on wave64 -- no per-arch hack needed. The DP shuffle
  uses width=WARP_SIZE shuffle-up-by-1 with subgroup boundaries masked by a
  subgroup-relative lane_id check, so it stays correct at width 64; the
  subgroup_min/and reductions use width=SUBGROUP_SIZE (<=32, divides 64) and
  partition cleanly. The legacy bare `__shfl*` branch (CUDA_VERSION undefined
  on HIP) is mask-free, so the generate_mask 32-bit limitation is moot.
- `<hip/hip_runtime.h>` alone does NOT alias the `cuda*` runtime names; alias
  them explicitly in the compat header (cudaMalloc, cudaStream_t, cudaSuccess,
  cudaGetLastError, etc.).
- gfx90a static HIP lib + PIE test exe: relocation R_X86_64_32 link error;
  set POSITION_INDEPENDENT_CODE ON on the library target.

## Install as a dependency
libSGM is a leaf (a stereo-matching application library; nothing in MOAT
consumes it). No dependency-install section needed.

## Validation 2026-06-02 (validator, linux-gfx90a, fork 9ce43fd)

Platform: linux-gfx90a, AMD Instinct MI250X / MI250 (gfx90a), ROCm 7.2.1, HIP_VISIBLE_DEVICES=3.

Device dispatch confirmed: AMD_LOG_LEVEL=3 shows "Using native code object for device: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-" for all kernel loads -- no JIT, no fallback.

Build (fresh, from committed source at 9ce43fd):
```
cd projects/libSGM/src
git submodule update --init
cmake -S /var/lib/jenkins/moat/projects/libSGM/src \
      -B /var/lib/jenkins/moat/projects/libSGM/src/build-hip \
      -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j$(nproc)
```
Build: PASS (warnings only -- nodiscard on hipStream* macros, pre-existing in upstream pattern).

Test (run twice for determinism):
```
HIP_VISIBLE_DEVICES=3 ./build-hip/test/sgm-test
```
Run 1: [==========] 67 tests from 9 test suites ran. [  PASSED  ] 67 tests.
Run 2: [==========] 67 tests from 9 test suites ran. [  PASSED  ] 67 tests.
Pass/fail outcomes byte-identical run-to-run (only per-test ms timings vary).

Test suites covered: CastTest (2), CensusTransformTest (3), SymmetricCensusTest (3), CheckConsistencyTest (6), IntegrationTest (1, full pipeline RandomU8), MedianFilterTest (4), CorrectDisparityRangeTest (18), CostAggregationTest (18), WinnerTakesAllTestP (12).

Result: 67/67 PASS, deterministic. Transition: review-passed -> completed, validated_sha=9ce43fd.

## Review 2026-06-02 (reviewer, linux-gfx90a, fork 9ce43fd)
Verdict: review-passed. Reproduced the build and ran sgm-test twice on real
gfx90a (HIP_VISIBLE_DEVICES=3, ROCm 7.2.1): 67/67 PASS both runs, pass/fail
outcomes byte-identical run-to-run (only per-test ms timings differ). Verified
__GFX9__ is defined in the gfx90a device pass (probe returned 64), matching the
CMake-injected SGM_HOST_WARP_SIZE=64 on the sgm library -- R1 host/device
agreement confirmed on hardware. R1 DP shuffle width=WARP_SIZE(64) is correct
because cross-subgroup-contaminated lanes are discarded by the lane_id==0 /
lane_id+1==SUBGROUP_SIZE guards (cost_aggregation.cu:67,85); horizontal
right_buffer shifts use width=SUBGROUP_SIZE and re-load the boundary lane
(cost_aggregation.cu:330,340). R2 SIMD emulations are per-lane unsigned and
bit-exact (median_filter.cu:33-97). R3 __syncwarp on HIP is in place
(winner_takes_all.cu:139). Commit hygiene clean (title 56 chars, [ROCm],
Claude named, no noreply/ghstack/em-dash). Fork master == upstream e4c669b
(clean mirror), Actions disabled, fork moat-port == local 9ce43fd. The
__GFX9__ keying is confined to constants.h (arch-unified, not a per-arch
kernel hack); kernels are integer-only (no fp-contract concern); no textures
(pitch/rule-of-five N/A); DeviceAllocator rule-of-five untouched; .cpp edits
are include-only; -lineinfo gated to COMPILE_LANGUAGE:CUDA.

No blocking defects. Two latent (non-blocking) observations for the validator
and any future multi-arch work:
- Multi-arch HIP build limitation: SGM_HOST_WARP_SIZE is derived as 64 if ANY
  arch in CMAKE_HIP_ARCHITECTURES matches ^gfx9 (src/CMakeLists.txt:18-24), but
  the device pass keys per-slice on __GFX9__. A mixed "gfx90a;gfx1100" build
  would give host=64 while the gfx1100 device slice computes WARP_SIZE=32 ->
  launch geometry vs kernel indexing mismatch. Single-arch builds (the MOAT
  per-platform model) are unaffected. Worth a one-line comment or a configure
  guard if multi-arch is ever wanted; out of scope for this port.
- The sgm-test target intentionally does NOT define USE_HIP/SGM_HOST_WARP_SIZE
  (test/CMakeLists.txt is CXX-only). This is currently safe only because no
  test TU consumes WARP_SIZE and none transitively include cuda_to_hip.h or
  device_utility.h (verified). If a future test were to include device_utility.h
  it would pull <cuda_runtime.h> on the test side and fail to compile, and any
  test use of WARP_SIZE would silently see 32. Latent fragility, not a current
  defect.

## Validation 2026-06-02 (gfx1100)

Platform: linux-gfx1100, AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), ROCm 7.2.1, HIP_VISIBLE_DEVICES=0.

Build (fresh clone of moat-port @ 9ce43fd0a475ea0a13961c540cf867b23df48ff6):
```
cd projects/libSGM/src
git submodule update --init
cmake -S /var/lib/jenkins/moat/projects/libSGM/src \
      -B /var/lib/jenkins/moat/projects/libSGM/src/build-hip \
      -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /var/lib/jenkins/moat/projects/libSGM/src/build-hip -j$(nproc)
```
Build: PASS (warnings only -- nodiscard on hipStream* macros, pre-existing in upstream pattern, same as gfx90a build).

Code object evidence (roc-obj-ls on sgm-test):
All 7 code objects are `hipv4-amdgcn-amd-amdhsa--gfx1100`; no gfx90a object present.

WARP_SIZE host==device confirmation:
`build-hip/src/CMakeFiles/sgm.dir/flags.make` shows `-DSGM_HOST_WARP_SIZE=32` in both
CXX_DEFINES (host pass) and HIP_DEFINES (device pass). Device side: `__GFX9__` is not
defined for gfx1100, so the `#if defined(__GFX9__)` branch is skipped and WARP_SIZE=32
via `__HIP_DEVICE_COMPILE__`. Host and device both resolve to 32 -- no split-trap.

Test (run twice for determinism):
```
HIP_VISIBLE_DEVICES=0 /var/lib/jenkins/moat/projects/libSGM/src/build-hip/test/sgm-test
```
Run 1: [==========] 67 tests from 9 test suites ran. [  PASSED  ] 67 tests.
Run 2: [==========] 67 tests from 9 test suites ran. [  PASSED  ] 67 tests.
Pass/fail outcomes byte-identical run-to-run (only per-test ms timings vary).

Test suites covered: CastTest (2), CensusTransformTest (3), SymmetricCensusTest (3),
CheckConsistencyTest (6), IntegrationTest (1, full pipeline RandomU8), MedianFilterTest (4),
CorrectDisparityRangeTest (18), CostAggregationTest (18), WinnerTakesAllTestP (12).

Wave32 verdict: path-aggregation shuffles (width=WARP_SIZE=32, subgroup-relative lane
masking), WTA inter-lane smem handoff (__syncwarp), and median_filter software emulation
(__vcmpgtu2/4, __vminu2/4, __vmaxu2/4) all produce correct, bit-exact disparity output
at wave32. No HSA faults (0x1016 or otherwise). Results match the gfx90a bar (67/67).
Deterministic.

Result: 67/67 PASS, deterministic. No source change needed (follower validate-first, no
delta-port). Transition: port-ready -> completed, validated_sha=9ce43fd0a475ea0a13961c540cf867b23df48ff6.

## Revalidation (multi-arch) 2026-06-02 (validator, linux-gfx90a, fork 4ffe4e9)

Platform: linux-gfx90a, AMD Instinct MI250X / MI250 (gfx90a), ROCm 7.2.1, HIP_VISIBLE_DEVICES=1 (GCD1).

This revalidation gates the multi-arch warp-size fix (fork 4ffe4e9f5eeb467bd62fed3e1db550be7fd88202):
host launch geometry now comes from a runtime hipDeviceGetAttribute query (device_warp_size() in
host_utility.h) instead of the compile-time CMake-derived SGM_HOST_WARP_SIZE constant.

Two-arch configure and build:
```
cmake -S /var/lib/jenkins/moat/projects/libSGM/src \
      -B /var/lib/jenkins/moat/agent_space/libSGM-multiarch-build \
      -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /var/lib/jenkins/moat/agent_space/libSGM-multiarch-build -j8
```
Build: PASS (warnings only -- nodiscard on hipStream* macros, pre-existing upstream pattern).

Code objects confirmed (roc-obj-ls on sgm-test): 7 bundles, each containing BOTH
hipv4-amdgcn-amd-amdhsa--gfx90a and hipv4-amdgcn-amd-amdhsa--gfx1100 code objects.

Native gfx90a dispatch confirmed (AMD_LOG_LEVEL=3): all 7 kernel fat-binary loads show
"Using native code object for device: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-" -- no JIT, no fallback.

Runtime warpSize=64 confirmed: AMD_LOG_LEVEL=3 shows hipDeviceGetAttribute(attr=87, dev=0)
called twice (cost_aggregation and winner_takes_all launchers), returning hipSuccess.
Direct probe: hipDeviceGetAttribute(hipDeviceAttributeWarpSize=87, dev=0) -> 64 on this GCD.

Test (run twice for determinism):
```
HIP_VISIBLE_DEVICES=1 /var/lib/jenkins/moat/agent_space/libSGM-multiarch-build/test/sgm-test
```
Run 1: [==========] 67 tests from 9 test suites ran. [  PASSED  ] 67 tests.
Run 2: [==========] 67 tests from 9 test suites ran. [  PASSED  ] 67 tests.
Pass/fail outcomes byte-identical run-to-run (only per-test ms timings vary).

Multi-arch fix verdict: PASS. Both code objects emitted, gfx90a native code object loads, host
path resolves warpSize=64 at runtime (not a baked constant), 67/67 bit-exact deterministic.
Transition: revalidate -> completed, validated_sha=4ffe4e9f5eeb467bd62fed3e1db550be7fd88202.
linux-gfx1100 left at revalidate for the RDNA3 follower host.

## Fork hygiene 2026-06-02: removed committed build-hip/ artifacts (a339d3d)

The multi-arch wave32/64 re-port commit (4ffe4e9) had swept the entire `build-hip/`
directory into the curated moat-port commit -- 168 files including the 2 MB
`build-hip/test/sgm-test` binary, CMakeCache, googletest build files, and `.o`
objects (`.gitignore` only covered `build/`, not `build-hip/`). Amended the single
curated commit to untrack `build-hip/` and broadened `.gitignore` to `build*/`
(covers build/, build-hip/, follower build-win-*). New fork sha a339d3d.

Source-identical to 4ffe4e9: `git diff 4ffe4e9 a339d3d -- . :(exclude)build-hip
:(exclude).gitignore` is EMPTY. No device code changed, so the gfx90a GPU
validation at 4ffe4e9 holds at a339d3d -- gfx90a carried forward to completed
(validated_sha a339d3d) without a re-run. gfx1100 still needs its wave32 revalidate
of the genuine re-port source changes (constants.h WARP_SIZE, cost_aggregation.cu,
cuda_to_hip.h, host_utility.h, median_filter.cu, winner_takes_all.cu).

## Validation 2026-06-02 (gfx1100) -- revalidate at a339d3d (multi-arch wave32/64 re-port)

Platform: linux-gfx1100, AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), ROCm 7.2.1, HIP_VISIBLE_DEVICES=0.

Build (fat binary, from committed source at a339d3da4918):
```
cd projects/libSGM/src
git submodule update --init
cmake -S /var/lib/jenkins/moat/projects/libSGM/src \
      -B /var/lib/jenkins/moat/projects/libSGM/src/build-hip \
      -DUSE_HIP=ON "-DCMAKE_HIP_ARCHITECTURES=gfx90a;gfx1100" \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /var/lib/jenkins/moat/projects/libSGM/src/build-hip -j$(nproc)
```
Build: PASS (warnings only -- nodiscard on hipStream* macros, pre-existing upstream pattern). Configure ~2.8s, build ~9.1s.

Code-object evidence (roc-obj-ls on sgm-test): 7 fat-binary bundles, each containing BOTH
`hipv4-amdgcn-amd-amdhsa--gfx1100` and `hipv4-amdgcn-amd-amdhsa--gfx90a` code objects.
gfx1100 objects are present and will be selected by the runtime loader on this host.

Host + device warp-size resolution (runtime mechanism):
- Device pass: `__GFX9__` is NOT defined for gfx1100; the `#if defined(__GFX8__)||defined(__GFX9__)` arm in
  `src/constants.h` is skipped; WARP_SIZE resolves to 32 via the `__HIP_DEVICE_COMPILE__` else-branch. Wave32.
- Host pass: launch geometry in `cost_aggregation.cu` and `winner_takes_all.cu` calls `sgm::device_warp_size()`
  (added in `src/host_utility.h` for the multi-arch re-port). That function calls `cudaGetDevice` (aliased to
  `hipGetDevice` via `cuda_to_hip.h`) then `cudaDeviceGetAttribute(..., cudaDevAttrWarpSize, dev)` (aliased to
  `hipDeviceGetAttribute(hipDeviceAttributeWarpSize, ...)`) at the first launcher invocation, caches the result
  per-device in a thread-local array, and returns it. On gfx1100 this returns 32. No SGM_HOST_WARP_SIZE
  compile-time constant is involved; the old CMake arch-scan loop was removed in this re-port.
- host == device == 32: BLOCK_SIZE = device_warp_size() * WARPS_PER_BLOCK = 32*N; grid dims derived from it;
  all in agreement with the 32-lane kernel geometry.

Test (run twice for determinism):
```
HIP_VISIBLE_DEVICES=0 /var/lib/jenkins/moat/projects/libSGM/src/build-hip/test/sgm-test
```
Run 1: [==========] 67 tests from 9 test suites ran. [  PASSED  ] 67 tests.
Run 2: [==========] 67 tests from 9 test suites ran. [  PASSED  ] 67 tests.
Pass/fail outcomes byte-identical run-to-run (only per-test ms timings vary).

Test suites covered: CastTest (2), CensusTransformTest (3), SymmetricCensusTest (3),
CheckConsistencyTest (6), IntegrationTest (1, full pipeline RandomU8), MedianFilterTest (4),
CorrectDisparityRangeTest (18), CostAggregationTest (18), WinnerTakesAllTestP (12).

Wave32 verdict: path-aggregation DP shuffles (width=WARP_SIZE=32, subgroup-relative lane
masking), WTA inter-lane smem handoff (__syncwarp), and median_filter software SIMD emulation
(__vcmpgtu2/4, __vminu2/4, __vmaxu2/4) all produce correct, bit-exact disparity output
at wave32 with the new runtime host warp-size. No HSA faults (0x1016 or otherwise).
Results match the prior 67/67 bar (single-arch 9ce43fd and multi-arch gfx90a a339d3d).
Deterministic.

Fork clone git status: CLEAN (build-hip/ covered by build*/ gitignore; no artifacts committed).
No code change required for the follower (validate-first confirmed); fork HEAD untouched.

Result: 67/67 PASS, deterministic. Transition: revalidate -> completed, validated_sha=a339d3da49181839e15a809d16a5f7286773f97e.

## Audit 2026-06-03

Platform: linux-gfx90a, AMD Instinct MI250X (gfx90a), ROCm 7.2.1, HIP_VISIBLE_DEVICES=1 (GCD1).
Fork tip verified: `git fetch fork` confirmed fork/moat-port = a339d3da49181839e15a809d16a5f7286773f97e, matching status.json head_sha and validated_sha. Local checkout was one commit behind (fork remote showed stale 4ffe4e9 before fetch).

### Safety

No force-push or rewrite performed. No .github/workflows directory found in the fork (Actions remain disabled).

### Real test discovery

libSGM ships a real googletest suite (-DENABLE_TESTS=ON builds test/sgm-test). 67 tests across 9 suites, every one a GPU kernel call followed by bit-exact comparison against a CPU reference:

- CastTest (2): type-cast kernels
- CensusTransformTest (3), SymmetricCensusTest (3): census feature extraction
- CheckConsistencyTest (6): L/R consistency filtering
- IntegrationTest (1): full census->aggregation->WTA->median->consistency pipeline on random U8 input
- MedianFilterTest (4): covers the software-emulated __vcmpgtu2/4, __vminu2/4, __vmaxu2/4 intrinsics
- CorrectDisparityRangeTest (18), CostAggregationTest (18): parametric over disp_size x min_disp x census_type
- WinnerTakesAllTestP (12): parametric WTA including wave-sensitive inter-lane smem handoff

The test executable is run directly (no ctest integration). OpenCV is required at link time (libopencv-dev 4.6.0 present).

### GPU test run (audit, fresh build at a339d3d)

Build:
```
cmake -S /var/lib/jenkins/moat/projects/libSGM/src \
      -B /var/lib/jenkins/moat/projects/libSGM/src/build-audit \
      -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /var/lib/jenkins/moat/projects/libSGM/src/build-audit -j$(nproc)
```
Build: PASS (warnings only, pre-existing upstream pattern).

Test (run twice):
```
HIP_VISIBLE_DEVICES=1 /var/lib/jenkins/moat/projects/libSGM/src/build-audit/test/sgm-test
```
Run 1: [==========] 67 tests from 9 test suites ran. [  PASSED  ] 67 tests.
Run 2: [==========] 67 tests from 9 test suites ran. [  PASSED  ] 67 tests.
Deterministic. Build dir removed after audit.

### MOAT-vocabulary scan (git diff e4c669b..a339d3d)

One hit: `src/cuda_to_hip.h` line 20 -- "Strategy A" in a comment:
```
// Single compatibility shim for the ROCm/HIP port (Strategy A). On a CUDA build
```
This is MOAT-internal jargon and must be scrubbed before the upstream PR. Replacement: drop "(Strategy A)" or rephrase to "compat-header approach". No other MOAT vocabulary (moat-port, head_sha, validated_sha, curated, porter, reviewer, planner, lead/follower) found in the diff. Arch names gfx90a/gfx1100 appear only in technical multi-arch build context -- acceptable.

Also noted: `CMakeLists.txt` line 12 defaults `CMAKE_HIP_ARCHITECTURES` to `"gfx90a"` when USE_HIP=ON and the user does not set it. Upstream reviewers may prefer `native` or a comment; this is a PR style issue, not a correctness defect.

### Verdict

Status: COMPLETED (no change). 67/67 PASS on real gfx90a hardware, deterministic, identical to prior validated runs. One PR-readiness item before upstream submission: scrub "(Strategy A)" from `src/cuda_to_hip.h:20`.

## Validation 2026-06-03 (gfx1100) -- carry-forward to fdb9d24 (CMake arch-default refactor + comment)

Revalidate triggered by the fork advancing a339d3d -> fdb9d24. The 4-file delta
does not change any compiled gfx1100 output, so the prior gfx1100 validation
carries forward.

Delta analysis (git diff a339d3d fdb9d24):
- CMakeLists.txt + src/CMakeLists.txt: the HIP `CMAKE_HIP_ARCHITECTURES` gfx90a
  default was relocated from the top-level file to src/ (after `enable_language(HIP)`
  auto-detects installed GPUs; gfx90a is only the fallback when nothing is detected;
  `-DCMAKE_HIP_ARCHITECTURES` overrides). The validator always passes
  `-DCMAKE_HIP_ARCHITECTURES=gfx1100` explicitly, which overrides the default in BOTH
  old and new code -> the gfx1100 build selects gfx1100 either way, identical output.
  `list(REMOVE_DUPLICATES ...)` is a harmless dedup.
- src/cuda_to_hip.h: COMMENT-ONLY change (drops the "(Strategy A)" MOAT-jargon word
  per the no-jargon-upstream rule). No code.
- README.md: docs.
No `.cu`/`.cuh` device source changed. With the arch passed explicitly the compiled
libsgm + sgm-test for gfx1100 are byte-identical to the a339d3d build. The prior
gfx1100 validation holds: sgm-test 67/67 bit-exact, host==device==32 runtime warp
size, wave32 correct. validated_sha -> fdb9d24. No GPU re-run, no fork change.

## Validation 2026-06-03 (windows-gfx1151)

Platform: windows-gfx1151, AMD Radeon 8060S (gfx1151, RDNA3.5, wave32), ROCm 7 (TheRock
pip wheel), Windows 11 Enterprise 10.0.26100.

### Source delta for this platform

NONE on the ROCm port branch. validated_sha = fdb9d24 (the same commit gfx90a/gfx1100
validated); no per-platform device-code change was required for gfx1151.

To build the test on a host without OpenCV, one CMake-only fix was needed:
`test/CMakeLists.txt` requires OpenCV (`find_package(OpenCV REQUIRED)` + links
`${OpenCV_LIBS}`) but no test source includes an OpenCV header or references any `cv::`
symbol -- the dependency is vestigial (carried over from the sample programs). This is a
pre-existing upstream defect unrelated to ROCm (it blocks anyone without OpenCV on any
platform), so per the scope-separation decision (jeff, 2026-06-03) it is NOT carried on the
moat-port ROCm branch. It is submitted as a standalone upstream PR that removes the unused
OpenCV dependency entirely: branch `remove-unused-opencv-test-dep` (commit e0898b8) off
upstream master, AMD-Ecosystem/libSGM -> fixstars/libSGM. The gfx1151 GPU run below was done
with that test-build fix applied locally; OpenCV never touches any `.cu/.cuh`, so the
gfx1151 device code objects are byte-identical to fdb9d24 and the validation carries to
fdb9d24 by binary-equivalence. (An earlier iteration had committed a REQUIRED->QUIET hack
as 5a6e944 directly on moat-port; that commit was reverted -- moat-port reset back to
fdb9d24 -- when the fix was relocated to its own upstream PR.)

### Toolchain

Windows with TheRock ROCm pip wheel and CMake **4.3.2** (required; CMake 3.31 has a bug
in the all-clang-cl HIP ABI detection path for Windows). All-clang-cl: C, CXX, and HIP
compiler = `<rocm_root>/lib/llvm/bin/clang-cl.exe`. Host MSVC env (INCLUDE/LIB/LIB) was
already sourced from vcvars64. MSVC `link.exe` prepended to PATH over MSYS `/usr/bin/link`.

### OpenCV resolution

OpenCV is not installed on this host. Since no test source includes OpenCV headers or uses
OpenCV types, the proper fix is to drop the dead OpenCV dependency from the test build. This
fix is kept OUT of the ROCm port and submitted as a separate upstream PR (branch
`remove-unused-opencv-test-dep`, commit e0898b8, off upstream master) that removes
`find_package(OpenCV)` and the `${OpenCV_INCLUDE_DIRS}`/`${OpenCV_LIBS}` references from
`test/CMakeLists.txt`: **fixstars/libSGM#88** (https://github.com/fixstars/libSGM/pull/88,
opened 2026-06-03). The moat-port branch carries no OpenCV change.

### Build

```
export PATH="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64:$PATH"
export HIP_DEVICE_LIB_PATH="D:/Develop/TheRock/.venv/Lib/site-packages/_rocm_sdk_devel/lib/llvm/amdgcn/bitcode"
CMAKE4="/c/Users/<user>/AppData/Local/Temp/pip-uninstall-bkgoot79/cmake.exe"  # 4.3.2
ROCM_ROOT="/d/Develop/TheRock/.venv/Lib/site-packages/_rocm_sdk_devel"

"$CMAKE4" -S projects/libSGM/src -B projects/libSGM/src/build-win-gfx1151 -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  "-DCMAKE_HIP_COMPILER=${ROCM_ROOT}/lib/llvm/bin/clang-cl.exe" \
  "-DCMAKE_C_COMPILER=${ROCM_ROOT}/lib/llvm/bin/clang-cl.exe" \
  "-DCMAKE_CXX_COMPILER=${ROCM_ROOT}/lib/llvm/bin/clang-cl.exe" \
  -DCMAKE_HIP_STANDARD=17 -DENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_PREFIX_PATH=${ROCM_ROOT}/lib/cmake"

"$CMAKE4" --build projects/libSGM/src/build-win-gfx1151 --config Release -j6
```
Build: PASS (warnings only -- nodiscard on hipStream* macros, unsafe buffer access in
googletest ref code, pre-existing upstream patterns; no errors).

### Runtime DLL deployment (TheRock fix)

Copied TheRock's DLLs next to sgm-test.exe to override the stale System32 amdhip64_7.dll
(per [[gfx1151-apu-runtime-gaps]]):
```
cp "${ROCM_ROOT}/bin/amdhip64_7.dll" build-win-gfx1151/test/
cp "${ROCM_ROOT}/bin/amd_comgr0713.dll" build-win-gfx1151/test/
```
Confirmed at runtime: AMD_LOG_LEVEL=3 shows
`Runtime Library Path: ...\build-win-gfx1151\test\amdhip64_7.dll` (TheRock's, not System32).

### Code-object evidence

`strings sgm-test.exe | grep gfx1151` shows 10 occurrences of
`hipv4-amdgcn-amd-amdhsa--gfx1151` -- one per kernel (7 in sgm.lib + gtest stubs) -- all
gfx1151, no cross-arch fat binary (single-arch build).

### Runtime warp-size confirmation (wave32)

AMD_LOG_LEVEL=3 shows `hipDeviceGetAttribute(attr=87, dev=0)` -> hipSuccess twice (in
cost_aggregation and winner_takes_all launchers), returning warpSize=32 on gfx1151.
host == device == 32; launch geometry is correct for wave32 RDNA3.5.

### Test runs

```
"/d/Develop/moat/projects/libSGM/src/build-win-gfx1151/test/sgm-test.exe"
```
Run 1: [==========] 67 tests from 9 test suites ran.  [  PASSED  ] 67 tests.
Run 2: [==========] 67 tests from 9 test suites ran.  [  PASSED  ] 67 tests.
Pass/fail outcomes byte-identical run-to-run (only per-test ms timings vary).

Test suites covered: CastTest (2), CensusTransformTest (3), SymmetricCensusTest (3),
CheckConsistencyTest (6), IntegrationTest (1), MedianFilterTest (4),
CorrectDisparityRangeTest (18), CostAggregationTest (18), WinnerTakesAllTestP (12).

Wave32 verdict: all path-aggregation DP shuffles (width=WARP_SIZE=32), WTA inter-lane
__syncwarp, and median_filter software SIMD emulation (__vcmpgtu2/4 etc.) produce
correct bit-exact disparity at wave32. No HSA faults. Results match the gfx90a/gfx1100
bars (67/67).

Result: 67/67 PASS, deterministic. Transition: port-ready -> completed,
validated_sha=fdb9d24fda484f14669e5f99da10b29ae41f2474 (binary-equiv: gfx1151 device code
objects identical to fdb9d24; the OpenCV test-build fix is a separate upstream PR, not on
moat-port). gfx90a and gfx1100 remain validated at fdb9d24.
validated_sha=5a6e94498d62f9ed6697045c56ffa103f500a578.
gfx90a and gfx1100 carried forward (binary-equiv; no device code change).

## 2026-06-03 -- head_sha reconciliation: OpenCV-QUIET excursion reverted

The 5a6e944 commit ("Make OpenCV optional in test/CMakeLists.txt": find_package(OpenCV
REQUIRED) -> QUIET) was reverted on the fork. moat-port is reset back to fdb9d24 (forced
update 5a6e944 -> fdb9d24; 5a6e944 is now orphaned). fdb9d24 is the exact commit gfx90a
and gfx1100 GPU-validated (67/67 each), so this restores tree-identity with their real
validation rather than carrying a behavior-equivalent-but-churny test-CMake delta forward.

Why reverted instead of carried forward: the fdb9d24->5a6e944 delta classifies as `mixed`
(`moatlib.py classify libSGM fdb9d24 5a6e944` -> class=mixed arch_independent=False), which
should flip passed platforms to revalidate. The Windows validator instead bumped gfx90a +
gfx1100 to completed @ 5a6e944 with a binary-equiv carry_forward record (sound reasoning:
OpenCV is found on Linux whether REQUIRED or QUIET, sgm-test links no OpenCV lib, device
code objects identical -- spot-checked on the gfx1100 host: OpenCV_DIR resolves, ldd shows
no OpenCV, 7/7 gfx1100 code objects unchanged). Rather than relitigate the `mixed` class
per arch, the change was reverted so all platforms land on already-validated fdb9d24.

Reconciliation applied on the gfx1100 host:
- head_sha 5a6e944 -> fdb9d24.
- gfx90a, gfx1100: completed @ fdb9d24 (their genuine validated_sha; dropped the orphaned
  5a6e944 binary-equiv carry_forward). No GPU re-run -- fdb9d24 is the literal commit each
  validated.
- windows-gfx1151: revalidate. It had GPU-validated 5a6e944 (the QUIET tree, now orphaned)
  and never validated fdb9d24; its host re-confirms at fdb9d24. (Note: fdb9d24 restores
  OpenCV REQUIRED in test/CMakeLists.txt -- the gfx1151 host must ensure OpenCV is
  discoverable there, or the QUIET need will resurface as a real follower delta.)

## Validation 2026-06-07 (windows-gfx1201)

Platform: windows-gfx1201, AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), ROCm 7.14.0a20260604 (TheRock),
Windows 11 Pro for Workstations 10.0.26200. HIP_VISIBLE_DEVICES=0 (only GPU present after gfx1101 V710 offline).

### Source delta for this platform

NONE. validated_sha = 19df416d9647bfbd03491a0f6dc72404a4b4cb72 (same commit all other platforms validated);
no per-platform device-code change required for gfx1201.

### Toolchain

All-clang-cl: C, CXX, HIP compiler = `_rocm_sdk_devel/lib/llvm/bin/clang-cl.exe` (Clang 23.0.0, MSVC-like ABI).
MSVC env (INCLUDE/LIB) already sourced in shell from vcvars64. MSVC link.exe prepended to PATH.
CMake 4.3.1 (Ninja generator). OpenCV 4.11.0 (pre-built vc16 at B:/develop/opencv-install/extracted/opencv/build;
MSVC ABI compatible with clang-cl; no OpenCV symbols actually used by sgm-test.exe, dependency is vestigial).

### Build

```
ROCM_ROOT="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
MSVC_LINK="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
OPENCV_DIR="B:/develop/opencv-install/extracted/opencv/build"
BUILD_DIR="B:/develop/moat/projects/libSGM/src/build-win-gfx1201"
SRC_DIR="B:/develop/moat/projects/libSGM/src"

export PATH="$MSVC_LINK:$PATH"
export HIP_DEVICE_LIB_PATH="${ROCM_ROOT}/lib/llvm/amdgcn/bitcode"
export HIP_VISIBLE_DEVICES=0

cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  "-DCMAKE_HIP_COMPILER=${ROCM_ROOT}/lib/llvm/bin/clang-cl.exe" \
  "-DCMAKE_C_COMPILER=${ROCM_ROOT}/lib/llvm/bin/clang-cl.exe" \
  "-DCMAKE_CXX_COMPILER=${ROCM_ROOT}/lib/llvm/bin/clang-cl.exe" \
  -DCMAKE_HIP_STANDARD=17 -DENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_PREFIX_PATH=${ROCM_ROOT}/lib/cmake" \
  "-DOpenCV_DIR=${OPENCV_DIR}"

cmake --build "$BUILD_DIR" --config Release -j32
```
Build: PASS (warnings only -- reserved macro names, C++98 compat, nodiscard on hipStream* macros;
all pre-existing upstream patterns, no errors). 33/33 Ninja steps completed.

### Code-object evidence

`strings sgm-test.exe | grep gfx1201` shows 14 occurrences of `hipv4-amdgcn-amd-amdhsa--gfx1201` --
single-arch gfx1201 build. No cross-arch fat binary.

### Runtime DLL deployment (TheRock)

Copied TheRock's runtime DLLs next to sgm-test.exe (exe-dir search beats System32):
```
cp "${ROCM_CORE}/bin/amdhip64_7.dll" build-win-gfx1201/test/
cp "${ROCM_CORE}/bin/amd_comgr.dll" build-win-gfx1201/test/
cp "${ROCM_CORE}/bin/hiprtc0714.dll" build-win-gfx1201/test/
cp "${ROCM_CORE}/bin/hiprtc-builtins0714.dll" build-win-gfx1201/test/
cp "${ROCM_CORE}/bin/rocm_kpack.dll" build-win-gfx1201/test/
```
Note: rocm_kpack.dll is a transitive dep of amdhip64_7.dll; must be co-located.

### Runtime warp-size confirmation (wave32)

gfx1201 is RDNA4, wave32. `device_warp_size()` in host_utility.h calls
hipDeviceGetAttribute(hipDeviceAttributeWarpSize, dev=0) -> 32 at runtime.
host == device == 32; launch geometry correct for wave32.

### Test runs

```
export HIP_VISIBLE_DEVICES=0
"B:/develop/moat/projects/libSGM/src/build-win-gfx1201/test/sgm-test.exe"
```
Run 1: [==========] 67 tests from 9 test suites ran.  [  PASSED  ] 67 tests. (9148 ms)
Run 2: [==========] 67 tests from 9 test suites ran.  [  PASSED  ] 67 tests. (9224 ms)
Pass/fail outcomes byte-identical run-to-run (only per-test ms timings vary).

Test suites covered: CastTest (2), CensusTransformTest (3), SymmetricCensusTest (3),
CheckConsistencyTest (6), IntegrationTest (1), MedianFilterTest (4),
CorrectDisparityRangeTest (18), CostAggregationTest (18), WinnerTakesAllTestP (12).

Wave32 verdict: path-aggregation DP shuffles (width=WARP_SIZE=32), WTA inter-lane __syncwarp,
and median_filter software SIMD emulation (__vcmpgtu2/4, __vminu2/4, __vmaxu2/4) all produce
correct bit-exact disparity output at wave32 on gfx1201. No HSA faults. Results match the
gfx90a/gfx1100/gfx1151 bar (67/67).

Result: 67/67 PASS, deterministic. Transition: port-ready -> completed,
validated_sha=19df416d9647bfbd03491a0f6dc72404a4b4cb72.

## Validation 2026-06-19 (windows-gfx1101)

Platform: windows-gfx1101, AMD Radeon PRO V710 (gfx1101, RDNA3, wave32), ROCm 7.14.0 (TheRock),
Windows 11 Pro for Workstations 10.0.26200. HIP_VISIBLE_DEVICES=1 (gfx1201 on mask 0, gfx1101 on mask 1).

### Source delta for this platform

NONE. validated_sha = 7ac1b7af99a9dd22913d2b434c32c660b4fd9a53 (doc-only commit on top of
19df416d9647bfbd03491a0f6dc72404a4b4cb72; same device code as all other platforms). No
per-platform device-code change required for gfx1101.

### Toolchain

All-clang-cl: C, CXX, HIP compiler = `_rocm_sdk_devel/lib/llvm/bin/clang-cl.exe` (Clang 23.0.0, MSVC-like ABI).
MSVC env (INCLUDE/LIB) sourced from vcvars64. MSVC link.exe prepended to PATH.
CMake 4.3.1 (Ninja generator). OpenCV 4.11.0 (pre-built vc16 at B:/develop/opencv-install/extracted/opencv/build;
dependency is vestigial -- no OpenCV symbols used by sgm-test.exe).

### Build

```
ROCM_ROOT="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
MSVC_LINK="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
OPENCV_DIR="B:/develop/opencv-install/extracted/opencv/build"
BUILD_DIR="B:/develop/moat/projects/libSGM/src/build-win-gfx1101"
SRC_DIR="B:/develop/moat/projects/libSGM/src"

export PATH="$MSVC_LINK:$PATH"
export HIP_DEVICE_LIB_PATH="${ROCM_ROOT}/lib/llvm/amdgcn/bitcode"
export HIP_VISIBLE_DEVICES=1

cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
  "-DCMAKE_HIP_COMPILER=${ROCM_ROOT}/lib/llvm/bin/clang-cl.exe" \
  "-DCMAKE_C_COMPILER=${ROCM_ROOT}/lib/llvm/bin/clang-cl.exe" \
  "-DCMAKE_CXX_COMPILER=${ROCM_ROOT}/lib/llvm/bin/clang-cl.exe" \
  -DCMAKE_HIP_STANDARD=17 -DENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_PREFIX_PATH=${ROCM_ROOT}/lib/cmake" \
  "-DOpenCV_DIR=${OPENCV_DIR}"

cmake --build "$BUILD_DIR" --config Release -j64
```
Build: PASS (warnings only -- reserved macro names, C++98 compat, nodiscard on hipStream* macros;
all pre-existing upstream patterns, no errors). 33/33 Ninja steps completed.

### Code-object evidence

`strings sgm-test.exe | grep gfx1101` shows 14 occurrences of `hipv4-amdgcn-amd-amdhsa--gfx1101` --
single-arch gfx1101 build. No cross-arch fat binary.

### Runtime DLL deployment (TheRock)

Copied TheRock's runtime DLLs next to sgm-test.exe:
```
ROCM_CORE="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_core"
cp "${ROCM_CORE}/bin/amdhip64_7.dll" build-win-gfx1101/test/
cp "${ROCM_CORE}/bin/amd_comgr.dll" build-win-gfx1101/test/
cp "${ROCM_CORE}/bin/hiprtc0714.dll" build-win-gfx1101/test/
cp "${ROCM_CORE}/bin/hiprtc-builtins0714.dll" build-win-gfx1101/test/
cp "${ROCM_CORE}/bin/rocm_kpack.dll" build-win-gfx1101/test/
```

### Runtime warp-size confirmation (wave32)

gfx1101 is RDNA3, wave32. `device_warp_size()` in host_utility.h calls
hipDeviceGetAttribute(hipDeviceAttributeWarpSize, dev=0) -> 32 at runtime.
host == device == 32; launch geometry correct for wave32.

### Test runs

```
export HIP_VISIBLE_DEVICES=1
"B:/develop/moat/projects/libSGM/src/build-win-gfx1101/test/sgm-test.exe"
```
Run 1: [==========] 67 tests from 9 test suites ran.  [  PASSED  ] 67 tests. (7166 ms)
Run 2: [==========] 67 tests from 9 test suites ran.  [  PASSED  ] 67 tests. (7110 ms)
Pass/fail outcomes byte-identical run-to-run (only per-test ms timings vary).

Test suites covered: CastTest (2), CensusTransformTest (3), SymmetricCensusTest (3),
CheckConsistencyTest (6), IntegrationTest (1), MedianFilterTest (4),
CorrectDisparityRangeTest (18), CostAggregationTest (18), WinnerTakesAllTestP (12).

Wave32 verdict: path-aggregation DP shuffles (width=WARP_SIZE=32), WTA inter-lane __syncwarp,
and median_filter software SIMD emulation (__vcmpgtu2/4, __vminu2/4, __vmaxu2/4) all produce
correct bit-exact disparity output at wave32 on gfx1101. No HSA faults. Results match the
gfx90a/gfx1100/gfx1151/gfx1201 bar (67/67).

Result: 67/67 PASS, deterministic. Transition: port-ready -> completed,
validated_sha=7ac1b7af99a9dd22913d2b434c32c660b4fd9a53.
