# rmagine notes

The ported project is **uos/rmagine** and the port lives on the
**AMD-Ecosystem/rmagine** fork, branched off upstream main (v2.4.2).

History note for anyone reading the older entries below: this record was
scaffolded under the name `rmcl` and renamed to `rmagine` once it was clear the
substantive port is rmagine's, not rmcl's. rmcl is a downstream ROS 2 consumer
whose portable GPU compute IS rmagine's CUDA backend; it wants rmagine
2.4-2.5.0, so this fork is in range for it. Entries written before the rename
say "rmcl" where they mean this project; `projects/rmcl/` no longer exists and
its paths have been rewritten to `projects/rmagine/src`.

## Fork / dependency

- Fork: https://github.com/AMD-Ecosystem/rmagine  (branch `moat-port`, off upstream main 6b93e86)
- Validated rmagine commit (Stage 1): a downstream that pins rmagine (rmcl does,
  through source_dependencies.yaml) should pin the moat-port HEAD recorded in
  status.json `head_sha`.
- Actions disabled on the fork.
- rmcl itself was NOT modified (its rmcl_ros .cu + ROS 2 layer is a separate
  milestone, see "Deferred").

## Stage 1 delivered: rmagine::cuda HIP compute backend (gfx90a, validated)

Strategy A (compat header + LANGUAGE HIP) behind a new top-level `USE_HIP`
option. NVIDIA build byte-identical when USE_HIP=OFF.

Key pieces:
- NEW `src/rmagine_cuda/include/rmagine/util/cuda/cuda_to_hip.h` -- single compat
  shim mapping the ~30 cuda runtime + cuCtx driver + curand device symbols used
  to hip/hiprand spellings. Included in place of `<cuda_runtime.h>`/`<cuda.h>`/
  `<curand*.h>` at every rmagine_cuda TU (11 redirected include lines + the
  math_svd test). Force-included on HIP TUs via `CMAKE_HIP_FLAGS -include`.
- `src/rmagine_cuda/CMakeLists.txt` -- USE_HIP branch: enable_language(HIP),
  LANGUAGE HIP on RMAGINE_CUDA_SRCS, `-fgpu-rdc` + `--hip-link` for the cross-TU
  `__device__` svd/umeyama_transform calls (CUDA used CUDA_SEPARABLE_COMPILATION),
  link hip::host + hip::hiprand (hip::device PRIVATE so the --offload-arch flag
  does not leak to plain-C++ consumers), drop the cublas/cusolver link refs (no
  call sites), default CMAKE_HIP_ARCHITECTURES=gfx90a only when unset.

### Wave-size hardening: warp-synchronous reduction tail (USE_HIP-guarded)
`statistics.cu` sum_kernel, the four `math_batched.cu` chunk_sums kernels, and
`memory_math.cu` cov_kernel<1024> / sum_kernel<1024> ran the `__syncthreads`
tree only to s>32, then a 32-lane `volatile` warpReduce tail with no
`__syncwarp`. That tail assumes a 32-lane lockstep wavefront. On gfx90a a
64-lane wavefront executes the low 32 lanes in lockstep in practice, so this is
NOT observed to miscompute on this hardware today (the reviewer's covtest and
this run's new asserting test both show ~1e-9..1e-4 rel match to a CPU
reference and bit-identical results run-to-run). The fix is wave-size
hardening, not a reproduced-corruption fix: the unsynchronized tail is not
guaranteed correct on a 64-lane wave. Fix (USE_HIP-guarded, CUDA byte-identical):
drop the warp tail, run the full block-wide `__syncthreads` tree to s>0. Applied
consistently to every warp-tail reduction in the exported reduction API
(rm::sum / rm::mean / rm::cov / sumBatched). The statistics_p2p/p2l/
objectwise_p2l kernels already ran the full tree (s>0), so they needed no change.

DEAD CODE: `statistics.cu:17 sum_kernel<blockSize,T>` has zero callers and is
not declared in any header. The load-bearing reductions are `memory_math.cu`
sum_kernel<1024> (backs rm::sum/rm::mean) and cov_kernel<1024> (backs rm::cov),
plus the math_batched chunk_sums set (backs sumBatched). The statistics.cu
sum_kernel was annotated as unused in-source and hardened only so all warp-tail
reductions in that TU read consistently; its fix is not functionally
load-bearing. `sum_kernel_test` (memory_math.cu) already ran the full s>0 tree
with no warp tail, so it needed no change.

### THE actually-decisive AMD fix: NaN-seed in the reduction kernels
The reduction kernels seeded shared memory with `sdata[tid] *= 0.0`, which reads
uninitialized LDS first. On AMD that garbage is routinely NaN/Inf and survives
the multiply (nan*0 = nan), poisoning the sum: rm::sum / rm::mean over Vector
and sumBatched returned NaN on gfx90a. (This is the hazard the plan flagged; the
reviewer's covtest got lucky LDS and did not hit it because cov_kernel uses
setZeros(), and rm::sum<int> cannot NaN.) Fix (unconditional -- UB on CUDA too):
seed each lane with a true typed zero `data[0] - data[0]` before accumulating.
The new asserting test (below) is what surfaced this.

### Two more AMD-surfaced bugs (also UB on CUDA, fixed unconditionally)
- `memory_math.cu` multNx1(Quaternion,Quaternion) called multNxN_kernel, which
  reads `b[id]` for id<N off a size-1 buffer; the Transform/Matrix3x3 siblings
  already use multNx1_kernel (reads `b[0]`). CUDA tolerated the OOB read; AMD
  faults (cuda_math memory access fault). Switched to multNx1_kernel.
- `shared_functions.h` keyed RMAGINE_FUNCTION on `__CUDA_ARCH__` (device pass
  only). HIP/clang then saw the shared math/memory helpers as host-only when
  device code called them and rejected the decl/def attribute mismatch. Re-keyed
  on `__CUDACC__`/`__HIPCC__` (defined in BOTH passes on each toolchain); a plain
  g++ build defines neither, so the CPU path is unchanged. `MemoryView::raw`
  definitions in Memory.tcc got the matching RMAGINE_FUNCTION.

### CudaContext
`cuCtxSetSharedMemConfig`/`cuCtxGetSharedMemConfig`/`CU_SHARED_MEM_CONFIG_*`
USE_HIP-guarded out (CDNA has no configurable LDS bank width). The rest of the
driver context API maps 1:1 to hipCtx*; `hipCtxCreate(&ctx, 0, dev)` matches the
pre-CUDA-13 `cuCtxCreate` form taken when CUDA_VERSION is undefined under HIP.

## Build recipe (gfx90a; standalone, no ROS/Embree/Vulkan/OptiX)

```
cd projects/rmagine/src   # AMD-Ecosystem/rmagine @ moat-port
export HIP_VISIBLE_DEVICES=1   # this host: GCD 1 only (others busy)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DRMAGINE_EMBREE_DISABLE=ON -DRMAGINE_OPTIX_DISABLE=ON \
  -DRMAGINE_VULKAN_DISABLE=ON -DRMAGINE_VULKAN_CUDA_INTEROP_DISABLE=ON \
  -DRMAGINE_OUSTER_DISABLE=ON -DRMAGINE_BUILD_TESTS=ON -DRMAGINE_BUILD_TOOLS=OFF
cmake --build build -j
```
Followers: same command, only change `-DCMAKE_HIP_ARCHITECTURES=gfx1100` (or
gfx1151). No source change -- the wave64 fix (full __syncthreads tree) is
wave-agnostic and correct on wave32 too. Deps: ROCm 7.2, Eigen3, TBB, Boost,
assimp (apt: libboost-all-dev libassimp-dev).

For Stage 2 HIPRT testing, also set:
```
export HIPRT_PATH=/path/to/HIPRT   # HIPRT SDK root (contains hiprt/, contrib/)
export LD_LIBRARY_PATH=$HIPRT_PATH/dist/bin/Release:$LD_LIBRARY_PATH
```
HIPRT_PATH is required for Orochi JIT kernel source discovery during BVH builds.

## Validation (real gfx90a / MI250X, GCD 1, HIP_VISIBLE_DEVICES=1) -- PASS

```
cd build && ctest --output-on-failure -R '^cuda_'   # 7/7 PASS
ctest --output-on-failure -R '^core_'               # 12/12 PASS (host unchanged)
```
- cuda_math, cuda_memory, cuda_memory_slicing, cuda_math_svd,
  cuda_math_statistics, cuda_math_reduction, cuda_math_reduction_correctness
  all PASS.
- NEW asserting gate `cuda_math_reduction_correctness`
  (tests/cuda/math_reduction_correctness.cpp): computes rm::sum / rm::mean /
  rm::cov over 4099 (non-power-of-two) Vector pairs, compares each component to
  a double-precision CPU reference to ~1e-4 rel (throws on mismatch/NaN), and
  asserts the GPU result is bit-identical across two runs. This is the real
  correctness/determinism gate for the reduction hardening + NaN-seed fix; the
  pre-existing cuda_math* tests only PRINT their reduction outputs (no assert),
  so they did not gate the reductions. Confirmed via AMD_LOG_LEVEL=3 that it
  dispatches the fixed sum_kernel<1024> and cov_kernel<1024> on MI250X.
- Before the NaN-seed fix this test FAILED with `sum = -nan` (and cuda_math
  sumBatched printed nan), proving the `*= 0.0` LDS-seed bug was real on AMD.
- hipRAND is not bitwise-identical to cuRAND (expected); the noise/random paths
  are validated statistically, not bitwise.
- Build dir for this run: agent_space/rmcl_build (scratch, gitignored). GCD 1
  only (HIP_VISIBLE_DEVICES=1).

### Gotchas (for followers / future passes)
- The compat header comment must not contain a literal end-of-comment marker
  inside a block comment (`cuda*` then `/curand*` closed the comment early ->
  bogus parse errors). Spell symbol globs out in prose.
- `-fgpu-rdc` is required: svd/umeyama_transform are cross-TU `__device__`
  functions; without it the device link fails with undefined hidden symbols.
- hip::device must be PRIVATE on the lib or `--offload-arch=gfx90a` leaks onto
  the plain-C++ test compiles (g++ rejects it).
- hiprand cmake target is `hip::hiprand`, not bare `hiprand` (-> `-lhiprand` not
  found).
- `timeit.sh` cd's to the MOAT repo root, so pass an ABSOLUTE build dir to the
  wrapped cmake/ctest command.

## Deferred (this host / scope)

- rmcl-layer milestone (rmcl_ros particle_motion.cu + resampling.cu + the MICP
  CUDA sensors + ROS 2 nodes): needs ROS 2 jazzy + Embree, not on this host. The
  two .cu are trivial (curand->hiprand; resampling's reduction already runs the
  full __syncthreads tree, wave-safe). Build them through rmagine's HIP toolchain
  in a colcon workspace once ROS 2 + Embree are provisioned. Not a blocker for
  the rmagine_cuda deliverable.

## Stage 2 (OptiX->HIPRT MCL backend) -- COMPLETE

The Monte-Carlo / global-localization GPU path is NVIDIA-OptiX-gated and is a
separate HIPRT reimplementation stage. All four sensor types are now implemented.

### Stage 2 All Sensors COMPLETED (2026-06-05)

Implemented and validated rmagine_hiprt ray-tracing backend for all sensor types:
- PinholeSimulatorHiprt (depth camera)
- SphericalSimulatorHiprt (lidar - phi/theta spherical coordinates)
- O1DnSimulatorHiprt (one origin, N directions - planar lidar)
- OnDnSimulatorHiprt (N origins, N directions - depth camera array)

Fork HEAD 4223818 @ moat-port.

Structure:
- `include/rmagine/util/hiprt/HiprtContext.hpp` -- HIPRT context wrapper
- `include/rmagine/map/hiprt/HiprtMesh.hpp` -- triangle mesh for HIPRT
- `include/rmagine/map/hiprt/HiprtScene.hpp` -- scene/BVH management
- `include/rmagine/simulation/hiprt/sim_program_data.h` -- kernel data structs
- `include/rmagine/simulation/{Pinhole,Spherical,O1Dn,OnDn}SimulatorHiprt.hpp`
- Implementation files in `src/util/`, `src/map/`, `src/simulation/`
- CMakeLists.txt wiring (USE_HIP-gated, requires HIPRT_PATH)

Key mapping (OptiX -> HIPRT):
- optixAccelBuild (triangle GAS) -> hiprtCreateGeometry + hiprtBuildGeometry
- raygen/closesthit/miss programs -> single HIP kernel (embedded source, JIT)
- optixTrace -> hiprtGeomTraversalClosest + getNextHit()
- SBT records -> HiprtMeshData structs passed as kernel args
- launch params -> kernel args (PinholeTraceParams, etc.)
- optixModuleCreate -> hiprtBuildTraceKernels (JIT, cached)
- optixLaunch -> hipModuleLaunchKernel (HIP driver API)

Implementation notes:
- Kernel source is embedded as a C++ raw string literal (not a separate file)
  because hiprtBuildTraceKernels takes source, not a file path
- BVH build requires temp buffer from hiprtGetGeometryBuildTemporaryBufferSize
- HIPRT module lifecycle managed by HIPRT's JIT cache (not manually unloaded)
- Device model structs must match rmagine types exactly (layout-sensitive)
- For O1Dn/OnDn, HiprtO1DnModelDevice/HiprtOnDnModelDevice hold device pointers
  to the directions/origins arrays copied to VRAM
- Transform3f uses Quaternionf{x,y,z,w} + Vector3f{x,y,z} + stamp (matches rmagine)

Validated on gfx90a (MI250X, ROCm 7.2.1):
- test_all_simulators traces rays against a 2x2 quad at z=2:
  - Pinhole: 25/64 hits, center range=2.0 (exact)
  - Spherical: 25/25 hits at phi=pi/2, center range=2.0 (exact)
  - O1Dn: 4/4 hits, avg range=2.08 (correct given ray angles)
  - OnDn: 4/4 hits, avg range=2.0 (exact, all rays point +Z)
- Stage 1 rmagine_cuda tests 7/7 PASS (no regression)
- Stage 1 core tests 12/12 PASS (no regression)

Remaining work (future):
- Multi-mesh scene support (current impl merges meshes into one geometry)
- Face normals computation (currently not used)
- Integration with rmcl's MCL localization path (requires ROS 2 + Embree)

Original scope:
- rmagine ships an OptiX ray-mesh-intersection backend (rmagine_optix:
  optixAccelBuild / optixTrace / module+SBT+pipeline, ~43 OptiX symbols across
  ~60 files). rmcl's RCCOptix + rmcl_ros optix BeamEvaluateProgram
  (__raygen__/__closesthit__/__miss__ PTX) + PCDSensorUpdaterOptix +
  rmcl_localization build only `if(RMCL_OPTIX AND RMCL_EMBREE)`.
- Plan: reimplement rmagine's OptiX ray-mesh backend against AMD HIPRT (the
  EnvGS Stage 2 path + agent_space/hiprt_probe/ are the reference), so the
  rmcl_localization Monte-Carlo node gets an AMD GPU ray-casting backend. Build
  the BVH with HIPRT, port the per-beam likelihood raygen/closesthit/miss to a
  HIPRT trace kernel, and feed the result into the existing rmcl correspondence /
  statistics compute (already HIP-validated in Stage 1).
- HIPRT is NOT in ROCm 7.2 (only hiprtc); needs the HIPRT SDK. Greenlit by jeff
  (OptiX->HIPRT) but explicitly NOT in scope this run.
- The Vulkan RT backend (cross-vendor) is a separate deferral: no Vulkan SDK on
  this host.

## Review 2026-06-01 (reviewer, linux-gfx90a) -- CHANGES REQUESTED

Reviewed moat-port HEAD 100e713 vs upstream merge-base 6b93e86 in
projects/rmagine/src. Stage 1 (rmagine::cuda compute port) only; Stage 2
(OptiX->HIPRT MCL backend) is a separate future stage and was not in scope.
Built fresh in agent_space and ran the suites on real gfx90a (GCD 1,
HIP_VISIBLE_DEVICES=1): ctest -R '^cuda_' = 6/6 PASS, ctest -R '^core_' = 12/12
PASS, reproducing the porter's result. The build, the Strategy-A compat-header
approach, the CMake gating, the NVIDIA-path guarding, and commit hygiene are all
sound. The problems below are about the central wave64 correctness narrative and
the inconsistency / dead-targeting of the reduction fix, plus inaccurate
test-evidence claims. They are fixable by either completing the fix or correcting
the analysis.

### 1. The wave64 warp-tail fix is applied to 5 kernels but the identical pattern is left, unguarded, in three publicly-exported kernels

The same `s > 32` tree + unsynced `volatile warpReduce<...>` tail that was
USE_HIP-guarded out of statistics.cu sum_kernel and the four math_batched.cu
chunk_sums kernels still runs verbatim -- on BOTH the HIP and CUDA paths -- in:
- src/rmagine_cuda/src/math/memory_math.cu:447-459 `cov_kernel<blockSize>`,
  launched as `cov_kernel<1024>` by the public `rm::cov()` (memory_math.cu:1552)
  and reached by `rm::mean`-adjacent covariance use.
- src/rmagine_cuda/src/math/memory_math.cu:1450-1464 `sum_kernel<nMemElems>`,
  launched as `sum_kernel<1024>` by the public `rm::sum()` (memory_math.cu:1496,
  1515) and therefore by `rm::mean()` (memory_math.cu:1528-1533).
- src/rmagine_cuda/src/math/memory_math.cu:1745 `sum_kernel_test` (the
  reduction-test kernel `rm::sum_reduce_test_t4` dispatches).

If the porter's stated root cause is correct (the tail "races and yields
wrong/nondeterministic sums" on wave64), then leaving it in the exported
covariance/sum/mean API is a real defect: rmcl's correspondence path consumes
exactly these statistics. Fix these three the same way (full `__syncthreads`
tree to s>0, USE_HIP-guarded), or explain why they are exempt.

### 2. The stated severity of the wave64 bug is not reproducible on gfx90a -- the analysis is overstated

I built two focused harnesses (agent_space/covtest/) against the as-shipped
libraries and ran them on gfx90a (GCD 1):
- `rm::cov()` over 4000 non-trivial Vector pairs (cov_kernel<1024>, the UNFIXED
  warp tail): GPU result matches a CPU double reference to ~1e-9 relative error
  and is bit-identical across 8 runs (0 run-to-run mismatches).
- `rm::sumBatched()` (the FIXED chunk_sums path): correct to ~7e-10 rel and
  bit-identical across 8 runs.
- `rm::sum()` over 20.7M ints (sum_kernel<1024>, UNFIXED): correct
  (20736000) and identical across 5 runs.

So on this hardware the warp-synchronous tail produces correct, deterministic
results (a wave64 executes lockstep and the `volatile` accesses order the LDS
reads/writes). The "WRONG / NaN / garbage covariance / run-to-run
nondeterministic" framing in the commit body and in notes.md "Root cause" is not
empirically supported here. The fix is defensible as spec-compliance / future
wave-size safety, but the notes and commit message should be re-scoped to say
that rather than asserting an observed miscompute/nondeterminism that the tests
do not actually exhibit. (If the porter did observe a failure, attach the
reproducer; I could not reproduce one.)

### 3. The headline-fixed kernel (statistics.cu sum_kernel) is dead code

src/rmagine_cuda/src/math/statistics.cu:17 `sum_kernel<blockSize,T>` has zero
callers anywhere in src/ or tests/ and is not declared in statistics.cuh /
statistics.h. The USE_HIP fix at statistics.cu:38-67 therefore validates nothing
and the "decisive fix" lands on an unused symbol. The real statistics path is
`statistics_p2p/p2l/objectwise_p2l`, which already ran the full tree (s>0) and
were correctly left unchanged. Either drop the dead kernel or note that the
load-bearing reduction fix is actually the math_batched chunk_sums set, not
statistics.cu sum_kernel.

### 4. Test-evidence claims in the commit body and notes are inaccurate

The commit body and notes say "cuda_math_reduction and cuda_math_statistics
dispatch the fixed sum_kernel / statistics kernels and assert no NaN ... two runs
of cuda_math_statistics are bit-identical." In fact:
- tests/cuda/math_statistics.cpp main() (line 181-199) runs only `test_sum()`,
  which calls `rm::sum` (the UNFIXED memory_math.cu sum_kernel), prints
  `total == size`, and does NOT call `checkStats` -- no assertion at all; the
  test passes regardless of the value. Its "determinism" is trivial (one fixed
  scalar sum).
- tests/cuda/math_reduction.cpp main() (line 206) runs only `test_sum_2()`,
  which calls `sum_reduce_test_t4` (sum_kernel_test) and only prints; the
  `checkStats` / `statistics_p2p` calls are in commented-out test1/test2.
- The fixed chunk_sums kernels are actually exercised by `cuda_math`
  (tests/cuda/math.cpp:402-499 sumBatched scalar/vector/matrix), but that block
  also only prints -- no EXPECT/ASSERT on the summed values.
So no running test asserts numeric correctness of any fixed reduction. The one
genuine assertion-by-crash is the multNx1 fault fix (see below), exercised by
cuda_math multNx1(Q,Q1,Q) at math.cpp:160. Re-scope the test-plan prose to what
the tests actually check, or add an assertion (EXPECT-style) on a sumBatched /
cov result against a CPU reference so the reduction fix has a real gate.

### Verified correct (no action)
- multNx1(Quaternion) OOB fix (memory_math.cu:619): switching multNxN_kernel ->
  multNx1_kernel is correct (multNx1_kernel reads b[0], memory_math.cu:54),
  matches the Transform/Matrix3x3 siblings, and is genuinely validated -- the
  pre-fix OOB faulted cuda_math, which now passes on gfx90a.
- shared_functions.h __CUDA_ARCH__ -> __CUDACC__/__HIPCC__ re-key
  (shared_functions.h:50) and the matching Memory.tcc raw() RMAGINE_FUNCTION
  (Memory.tcc:58,63): plain g++ defines neither macro so the CPU path is
  unchanged; core_ tests 12/12 PASS confirm the host path is intact.
- cuda_to_hip.h symbol surface, the CudaContext cuCtx* mapping with
  cuCtxSetSharedMemConfig/GetSharedMemConfig USE_HIP-guarded out, and the
  hipCtxCreate(&ctx,0,dev) form (CUDA_VERSION undefined under HIP -> pre-13
  branch at CudaContext.cpp:44) are correct. cuRAND->hipRAND validated
  statistically, not bitwise (correct bar).
- CrossStatistics zero-seed `sdata[tid] *= 0.0` is a true zero (statistics
  reductions produce no NaN; core/cuda statistics tests pass).
- Strategy A, USE_HIP default OFF, enable_language(HIP), -fgpu-rdc/--hip-link,
  hip::device PRIVATE, default gfx90a only when unset: all correct and minimal.
- Commit hygiene: [ROCm] title 65 chars, Claude disclosed, no noreply/coauthor/
  ghstack/em-dash; fork origin/main == upstream 6b93e86 (clean mirror); fork
  Actions disabled. No AMD-internal account references.

### Porter response 2026-06-01 (re-review, HEAD 3d098d5) -- all 4 items addressed
1. Wave-tail hardening extended to the three exported kernels: cov_kernel<1024>
   and sum_kernel<1024> (memory_math.cu) now USE_HIP-guarded full-tree like the
   other five. sum_kernel_test was ALREADY full-tree (s>0, no warpReduce) so it
   needed no change (reviewer line-number was approximate).
2. Re-scoped all prose (commit body + notes "Wave-size hardening" + the in-source
   comments in statistics.cu/math_batched.cu/memory_math.cu) from
   "races/WRONG/NaN/nondeterministic" to "wave-size hardening: removes an
   unsynchronized-warp-tail assumption not guaranteed on a 64-lane wave; not
   observed to miscompute on gfx90a today." Honest.
3. statistics.cu sum_kernel annotated in-source as unused/dead (zero callers, no
   header decl); notes "DEAD CODE" paragraph added. Its fix is not load-bearing.
4. Added tests/cuda/math_reduction_correctness.cpp -- an ASSERTING gate on
   rm::sum/mean/cov vs a CPU double reference + 2-run determinism. Wired into
   ctest (cuda_math_reduction_correctness). Running it surfaced a REAL
   AMD-specific NaN-seed bug (`sdata[tid] *= 0.0` on uninitialized LDS) that the
   reviewer's covtest missed; fixed unconditionally with a true typed zero
   (data[0]-data[0]) in sum_kernel + the four chunk_sums kernels. multNx1 OOB
   fix and shared_functions.h macro fix left as-is (reviewer confirmed correct).
Re-validated on gfx90a (GCD 1): cuda_ 7/7 PASS, core_ 12/12 PASS.

### Required before re-review
Address items 1-4: either (a) extend the USE_HIP-guarded full-tree fix to
cov_kernel<1024>, sum_kernel<1024> (int+Vector), and sum_kernel_test for
consistency across the exported reduction API, and add at least one
correctness-asserting GPU check (sumBatched or cov vs CPU reference); or (b) if
the warp tail is intentionally left as benign-on-CDNA, correct the commit
message and notes to state that the fix is spec/wave-size hardening (not an
observed miscompute), justify why the unguarded exported kernels are acceptable,
and remove or annotate the dead statistics.cu sum_kernel. The functional port is
otherwise GPU-clean on gfx90a.

## Review 2026-06-01 (reviewer DELTA re-review, linux-gfx90a) -- REVIEW PASSED

Focused delta re-review of 100e713 -> 3d098d5 (5 files: math_batched.cu,
memory_math.cu, statistics.cu, tests/cuda/CMakeLists.txt + the new
tests/cuda/math_reduction_correctness.cpp). The compat header, multNx1 OOB fix,
__CUDACC__/__HIPCC__ re-key, and commit hygiene were cleared in the prior pass
and are unchanged in the delta. Rebuilt fresh in agent_space and ran on real
gfx90a (GCD 1, HIP_VISIBLE_DEVICES=1): ctest -R '^cuda_' = 7/7 PASS (incl. the
new cuda_math_reduction_correctness), ctest -R '^core_' = 12/12 PASS.

All four prior findings resolved; the NaN-seed fix it surfaced is correct.

1. Wave-tail hardening now consistent across the exported reduction API. The
   USE_HIP-guarded full-`__syncthreads`-tree-to-s>0 form is applied to
   cov_kernel<1024> (memory_math.cu:447-474, backs rm::cov) and sum_kernel<1024>
   (memory_math.cu:1470-1501, backs rm::sum/rm::mean) with the CUDA s>32 +
   warpReduce tail preserved in the #else. No exported warp-tail reduction
   remains unguarded. sum_kernel_test was already a full s>0 tree (no warp tail),
   confirmed -- the prior review's line number was approximate; nothing to change
   there. CUDA path is byte-identical (the new code is strictly inside the
   USE_HIP/`__HIP_PLATFORM_AMD__` branch).

2. Prose re-scoped honestly. The commit body, the notes "Wave-size hardening"
   paragraph, and the in-source comments (statistics.cu:42-49,
   memory_math.cu:448-451 / 1471-1474, math_batched.cu x4) now describe an
   unsynchronized-warp-tail assumption not guaranteed on a 64-lane wave, "not
   observed to miscompute on gfx90a today" -- no false claim of reproduced
   corruption/nondeterminism.

3. Dead code annotated. statistics.cu:16-20 marks sum_kernel as unused (no
   callers, no header decl) and points to the load-bearing memory_math.cu
   kernels.

4. Real asserting test. math_reduction_correctness.cpp builds 4099 (non-power-of-
   two) Vector pairs, runs rm::sum/rm::mean/rm::cov, and check_rel() THROWS
   (uncaught -> non-zero exit -> ctest fail) on >1e-4 rel error or on NaN
   (`got != got`); the determinism block uses exact `==` across two runs. It
   genuinely gates the fixed sum_kernel<1024>/cov_kernel<1024> (porter confirmed
   dispatch via AMD_LOG_LEVEL=3, and the test failed with -nan before the seed
   fix). This is a real assert, not print-only.

5. NaN-seed fix verified correct. `sdata[tid] *= 0.0` (read of uninitialized LDS,
   NaN/Inf-prone on AMD) replaced by a true typed zero `sdata[tid] = data[0];
   sdata[tid] -= data[0];` at every prior `*= 0.0` site: sum_kernel
   (memory_math.cu:1458-1459) and all four chunk_sums kernels (math_batched.cu).
   Vector is Vector3_<float> with operator-= (Vector3.hpp:214), int trivially, so
   data[0]-data[0] is a correct component-wise zero for every reduced type.
   cov_kernel never had the bug (it seeds via setZeros(), memory_math.cu:427) and
   correctly was not touched by the seed change. The fix is unconditional (the
   `*= 0.0` was UB on CUDA too) and does not alter CUDA numerics (a true zero is
   what `*= 0.0` was always intended to produce).

Minor (non-blocking, no action): the new seed reads `data[0]` unconditionally on
every lane, so an empty reduction (N==0) would read OOB where the old `*= 0.0`
did not dereference data. This path was already meaningless under the old code
(the accumulate loop ran zero times and returned uninitialized LDS), no
in-tree/rmcl caller passes an empty buffer, and the in-source comment scopes the
guarantee to "non-empty reduction input." Acceptable; flagging only for a future
hardening pass.

Verdict: clean. Stage 2 (OptiX->HIPRT) remains out of scope. Handing to the
validator.

## Validation 2026-06-01 (linux-gfx90a, GCD 1, HIP_VISIBLE_DEVICES=1) -- PASS

Fork: AMD-Ecosystem/rmagine moat-port HEAD 3d098d5. Clean build from source; build dir
agent_space/rmcl_valclean_build (scratch, gitignored). GPU: AMD Instinct MI250X /
MI250, gfx90a:sramecc+:xnack-.

### Configure

```
cmake -S projects/rmagine/src \
      -B agent_space/rmcl_valclean_build \
      -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DRMAGINE_EMBREE_DISABLE=ON -DRMAGINE_OPTIX_DISABLE=ON \
      -DRMAGINE_VULKAN_DISABLE=ON -DRMAGINE_VULKAN_CUDA_INTEROP_DISABLE=ON \
      -DRMAGINE_OUSTER_DISABLE=ON -DRMAGINE_BUILD_TESTS=ON \
      -DRMAGINE_BUILD_TOOLS=OFF
```

### Build

```
cmake --build agent_space/rmcl_valclean_build -j$(nproc)
```

74/74 targets built cleanly (HIP compiler: clang++ 22.0.0 / ROCm 7.2). No errors;
only pre-existing nodiscard warnings on hipMemset macro expansions (unchanged from
upstream behavior).

### Test results

```
export HIP_VISIBLE_DEVICES=1
ctest --test-dir agent_space/rmcl_valclean_build --output-on-failure -R '^cuda_'
# Run 1: 7/7 PASS (2.22 s)
# Run 2: 7/7 PASS (2.23 s)  -- determinism confirmed
ctest --test-dir agent_space/rmcl_valclean_build --output-on-failure -R '^core_'
# 12/12 PASS (3.14 s)  -- host path no regression
```

Tests passing:
- cuda_math, cuda_memory, cuda_memory_slicing, cuda_math_svd,
  cuda_math_statistics, cuda_math_reduction (pre-existing suite)
- cuda_math_reduction_correctness (new asserting gate added by porter)
- core_math, core_memory, core_memory_slicing, core_quaternion, core_math_svd,
  core_math_statistics, core_math_cov_transform, core_math_gaussians,
  core_math_matrix_slicing, core_math_reduction, core_math_cholesky, core_math_lie

### GPU dispatch confirmed (AMD_LOG_LEVEL=3)

```
AMD_LOG_LEVEL=3 ./bin/rmagine_tests_cuda_math_reduction_correctness
```

ShaderName lines confirm dispatch of:
- `void rmagine::cuda::sum_kernel<1024u, rmagine::Vector3_<float>>(...)`
- `void rmagine::cuda::cov_kernel<1024u>(...)`
on device `amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-`. Test exits with:
`PASS: rm::sum/mean/cov match CPU reference and are deterministic`

### Remaining work

Stage 2 (OptiX->HIPRT MCL backend) is a separate future stage; scope and plan
documented in "Stage 2 (OptiX->HIPRT MCL backend)" section above. Not in scope
for this validation.

## Validation 2026-06-01 (linux-gfx1100, HIP_VISIBLE_DEVICES=0) -- PASS

Fork: AMD-Ecosystem/rmagine moat-port HEAD 3d098d5 (identical to gfx90a validated SHA 3d098d58eb59). No source change. GPU: 4x AMD Radeon Pro W7800 48GB, gfx1100 (RDNA3, wave32), ROCm 7.2.1.

### Configure

```
cmake -S /var/lib/jenkins/moat/projects/rmagine/src \
      -B /var/lib/jenkins/moat/agent_space/rmcl_gfx1100_build \
      -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DRMAGINE_EMBREE_DISABLE=ON -DRMAGINE_OPTIX_DISABLE=ON \
      -DRMAGINE_VULKAN_DISABLE=ON -DRMAGINE_VULKAN_CUDA_INTEROP_DISABLE=ON \
      -DRMAGINE_OUSTER_DISABLE=ON -DRMAGINE_BUILD_TESTS=ON \
      -DRMAGINE_BUILD_TOOLS=OFF
```

### Build

```
cmake --build /var/lib/jenkins/moat/agent_space/rmcl_gfx1100_build -j$(nproc)
```

74/74 targets built cleanly (HIP compiler: clang++ 22.0.0 / ROCm 7.2.1). No errors; only pre-existing nodiscard warnings on hipMemset macro expansions.

### Code-object arch evidence

```
roc-obj-ls lib/librmagine-cuda.so.2.4.2
```

Output: `hipv4-amdgcn-amd-amdhsa--gfx1100` (833688 bytes). No gfx90a code object present.

### Test results

```
export HIP_VISIBLE_DEVICES=0
# Run 1
ctest --test-dir /var/lib/jenkins/moat/agent_space/rmcl_gfx1100_build --output-on-failure -R '^cuda_'
# 7/7 PASS (1.85 s)
# Run 2 (determinism)
ctest --test-dir /var/lib/jenkins/moat/agent_space/rmcl_gfx1100_build --output-on-failure -R '^cuda_'
# 7/7 PASS (1.87 s)
# Host regression check
ctest --test-dir /var/lib/jenkins/moat/agent_space/rmcl_gfx1100_build --output-on-failure -R '^core_'
# 12/12 PASS (2.60 s)
```

Tests passing (same set as gfx90a):
- cuda_math, cuda_memory, cuda_memory_slicing, cuda_math_svd,
  cuda_math_statistics, cuda_math_reduction (pre-existing suite)
- cuda_math_reduction_correctness (asserting gate for rm::sum/mean/cov vs CPU reference)
- core_math, core_memory, core_memory_slicing, core_quaternion, core_math_svd,
  core_math_statistics, core_math_cov_transform, core_math_gaussians,
  core_math_matrix_slicing, core_math_reduction, core_math_cholesky, core_math_lie

### GPU dispatch confirmed (AMD_LOG_LEVEL=3)

```
AMD_LOG_LEVEL=3 ./bin/rmagine_tests_cuda_math_reduction_correctness
```

ShaderName lines confirm dispatch on `amdgcn-amd-amdhsa--gfx1100` of:
- `void rmagine::cuda::sum_kernel<1024u, rmagine::Vector3_<float>>(...)`
- `void rmagine::cuda::cov_kernel<1024u>(...)`
Test exits with: `PASS: rm::sum/mean/cov match CPU reference and are deterministic`

### Wave32 verdict

The full-`__syncthreads`-tree-to-s>0 reduction (USE_HIP-guarded) is correct on the
32-lane wavefront of gfx1100. With wave32 the original warp-tail concern is even
sharper (only 32 lanes execute in lockstep, not 64), but the fix runs the
complete block-wide tree so no warp-synchronous tail is exercised at all. No
leftover unsynchronized warp tail remains in the HIP path. No HSA 0x1016, no HIP
error, no NaN, no hang. Results match the CPU reference within documented
tolerance (~1e-4 rel) and are bit-identical run-to-run. Matches gfx90a@3d098d5.

No source change from the gfx90a-validated HEAD (follower validate-first; no
delta port needed).

## Validation 2026-06-05 (linux-gfx1100 REVALIDATE, HIP_VISIBLE_DEVICES=0) -- PASS

Revalidation after HEAD moved from 3d098d5 (Stage 1 only) to db7f064 (Stage 1 + Stage 2 HIPRT).
Fork: AMD-Ecosystem/rmagine moat-port HEAD db7f064. GPU: AMD Radeon Pro W7800 48GB, gfx1100 (RDNA3, wave32), ROCm 7.2.1.

### Delta classification

Changes 3d098d5 -> db7f064 add entirely new rmagine_hiprt component (Stage 2 HIPRT ray-tracing, 1742 insertions across 14 new files). Stage 1 (rmagine_cuda compute backend) source is unchanged.

### HIPRT availability

HIPRT SDK not present on this gfx1100 host (/var/lib/jenkins/moat/third_party/HIPRT does not exist). CMake correctly skipped rmagine_hiprt component with warning message. Stage 1 components (rmagine-core + rmagine-cuda) built successfully without HIPRT.

### Build

```
cmake -S /var/lib/jenkins/moat/projects/rmagine/src \
      -B /var/lib/jenkins/moat/agent_space/rmcl_gfx1100_revalidate \
      -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DRMAGINE_EMBREE_DISABLE=ON -DRMAGINE_OPTIX_DISABLE=ON \
      -DRMAGINE_VULKAN_DISABLE=ON -DRMAGINE_VULKAN_CUDA_INTEROP_DISABLE=ON \
      -DRMAGINE_OUSTER_DISABLE=ON -DRMAGINE_BUILD_TESTS=ON \
      -DRMAGINE_BUILD_TOOLS=OFF
cmake --build /var/lib/jenkins/moat/agent_space/rmcl_gfx1100_revalidate -j
```

74/74 targets built cleanly (HIP compiler: clang++ 22.0.0 / ROCm 7.2.1). Only pre-existing nodiscard warnings on hipMemset macro expansions. rmagine_hiprt was skipped (HIPRT SDK not found, expected).

### Code-object arch evidence

```
roc-obj-ls lib/librmagine-cuda.so.2.4.2
```

Output: `hipv4-amdgcn-amd-amdhsa--gfx1100` (833688 bytes). No gfx90a code object present.

### Test results (Stage 1 rmagine_cuda -- the validated scope)

```
export HIP_VISIBLE_DEVICES=0
ctest --test-dir /var/lib/jenkins/moat/agent_space/rmcl_gfx1100_revalidate --output-on-failure -R '^cuda_'
# 7/7 PASS (1.81 s)
ctest --test-dir /var/lib/jenkins/moat/agent_space/rmcl_gfx1100_revalidate --output-on-failure -R '^core_'
# 12/12 PASS (2.67 s)
```

Tests passing (identical set to prior gfx1100 validation at 3d098d5):
- cuda_math, cuda_memory, cuda_memory_slicing, cuda_math_svd, cuda_math_statistics, cuda_math_reduction, cuda_math_reduction_correctness
- core_math, core_memory, core_memory_slicing, core_quaternion, core_math_svd, core_math_statistics, core_math_cov_transform, core_math_gaussians, core_math_matrix_slicing, core_math_reduction, core_math_cholesky, core_math_lie

### GPU dispatch confirmed (AMD_LOG_LEVEL=3)

```
AMD_LOG_LEVEL=3 ./bin/rmagine_tests_cuda_math_reduction_correctness
```

ShaderName lines confirm dispatch on `amdgcn-amd-amdhsa--gfx1100` of:
- `void rmagine::cuda::sum_kernel<1024u, rmagine::Vector3_<float>>(...)`
- `void rmagine::cuda::cov_kernel<1024u>(...)`

Test exits with: `PASS: rm::sum/mean/cov match CPU reference and are deterministic`

### Stage 2 HIPRT verdict

Stage 2 HIPRT component was not built or tested on this platform (HIPRT SDK unavailable). This is consistent with the gfx90a experience: Stage 2 code compiles but HIPRT SDK runtime environment is not functional for JIT BVH builds on any platform yet. Stage 1 remains the validated deliverable on all platforms.

### Revalidation verdict

Stage 1 (rmagine_cuda HIP compute backend) -- NO REGRESSION. 7/7 cuda_ tests + 12/12 core_ tests PASS on gfx1100@db7f064, matching prior validation at 3d098d5. The addition of the rmagine_hiprt component (Stage 2) does not affect Stage 1 functionality when HIPRT is absent (correctly skipped by CMake). Wave32 reduction correctness confirmed (full __syncthreads tree, no warp tail).

## Validation 2026-06-05 (linux-gfx1100 REVALIDATE at 4223818, HIP_VISIBLE_DEVICES=0) -- PASS

Revalidation after HEAD moved from db7f064 (Stage 2 Pinhole HIPRT) to 4223818 (Stage 2 all simulators).
Fork: AMD-Ecosystem/rmagine moat-port HEAD 4223818. GPU: AMD Radeon Pro W7800 48GB, gfx1100 (RDNA3, wave32), ROCm 7.2.1.

### Delta classification

Changes db7f064 -> 4223818 add SphericalSimulatorHiprt, O1DnSimulatorHiprt, OnDnSimulatorHiprt (1624 insertions across 8 files in rmagine_hiprt). Stage 1 (rmagine_cuda compute backend) source is unchanged.

### HIPRT availability

HIPRT SDK not present on this gfx1100 host (/var/lib/jenkins/moat/third_party/HIPRT does not exist). CMake correctly skipped rmagine_hiprt component with warning message. Stage 1 components (rmagine-core + rmagine-cuda) built successfully without HIPRT.

### Build

```
cmake -S /var/lib/jenkins/moat/projects/rmagine/src \
      -B /var/lib/jenkins/moat/agent_space/rmcl_gfx1100_revalidate_4223818 \
      -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DRMAGINE_EMBREE_DISABLE=ON -DRMAGINE_OPTIX_DISABLE=ON \
      -DRMAGINE_VULKAN_DISABLE=ON -DRMAGINE_VULKAN_CUDA_INTEROP_DISABLE=ON \
      -DRMAGINE_OUSTER_DISABLE=ON -DRMAGINE_BUILD_TESTS=ON \
      -DRMAGINE_BUILD_TOOLS=OFF
cmake --build /var/lib/jenkins/moat/agent_space/rmcl_gfx1100_revalidate_4223818 -j
```

74/74 targets built cleanly (HIP compiler: clang++ 22.0.0 / ROCm 7.2.1). Only pre-existing nodiscard warnings on hipMemset macro expansions. rmagine_hiprt was skipped (HIPRT SDK not found, expected).

### Code-object arch evidence

```
roc-obj-ls lib/librmagine-cuda.so.2.4.2
```

Output: `hipv4-amdgcn-amd-amdhsa--gfx1100` (833688 bytes). No gfx90a code object present.

### Test results (Stage 1 rmagine_cuda -- the validated scope)

```
export HIP_VISIBLE_DEVICES=0
ctest --test-dir /var/lib/jenkins/moat/agent_space/rmcl_gfx1100_revalidate_4223818 --output-on-failure -R '^cuda_'
# 7/7 PASS (1.98 s)
ctest --test-dir /var/lib/jenkins/moat/agent_space/rmcl_gfx1100_revalidate_4223818 --output-on-failure -R '^core_'
# 12/12 PASS (2.64 s)
```

Tests passing (identical set to prior gfx1100 validation at db7f064):
- cuda_math, cuda_memory, cuda_memory_slicing, cuda_math_svd, cuda_math_statistics, cuda_math_reduction, cuda_math_reduction_correctness
- core_math, core_memory, core_memory_slicing, core_quaternion, core_math_svd, core_math_statistics, core_math_cov_transform, core_math_gaussians, core_math_matrix_slicing, core_math_reduction, core_math_cholesky, core_math_lie

### GPU dispatch confirmed (AMD_LOG_LEVEL=3)

```
AMD_LOG_LEVEL=3 ./bin/rmagine_tests_cuda_math_reduction_correctness
```

ShaderName lines confirm dispatch on `amdgcn-amd-amdhsa--gfx1100` of:
- `void rmagine::cuda::sum_kernel<1024u, rmagine::Vector3_<float>>(...)`
- `void rmagine::cuda::cov_kernel<1024u>(...)`

Test exits with: `PASS: rm::sum/mean/cov match CPU reference and are deterministic`

### Stage 2 HIPRT verdict

Stage 2 HIPRT component (all four simulators: Pinhole, Spherical, O1Dn, OnDn) was not built or tested on this platform (HIPRT SDK unavailable). CMake correctly skipped the component. Stage 1 remains the validated deliverable on all platforms.

### Revalidation verdict

Stage 1 (rmagine_cuda HIP compute backend) -- NO REGRESSION. 7/7 cuda_ tests + 12/12 core_ tests PASS on gfx1100@4223818, matching prior validations at db7f064 and 3d098d5. The addition of three more HIPRT simulators (Stage 2) does not affect Stage 1 functionality when HIPRT is absent (correctly skipped by CMake). Wave32 reduction correctness confirmed (full __syncthreads tree, no warp tail).

## Review 2026-06-05 (reviewer, linux-gfx90a, Stage 2 HIPRT) -- REVIEW PASSED

Re-review of moat-port HEAD db7f064 (fix commit for Transform3f struct layout).
Previous review found that the JIT kernel's Transform3f was missing the `stamp`
field that rmagine::Transform_ has (28 vs 32 bytes), which would corrupt pose
data when indexing Tbm arrays in multi-pose simulations.

### Fix verified correct

The fix at db7f064 adds `unsigned int stamp` to Transform3f in both locations:
- `src/rmagine_hiprt/include/rmagine/simulation/hiprt/pinhole_trace_kernel.h:33`
- `src/rmagine_hiprt/src/simulation/PinholeSimulatorHiprt.cpp:254` (embedded JIT source)

This matches the upstream `rmagine::Transform_<float>` layout:
- `Quaternion_<DataT> R` (4 floats)
- `Vector3_<DataT> t` (3 floats)
- `uint32_t stamp` (1 uint32)

Total: 32 bytes, matching Transform_ exactly. Fix is correct.

### Non-blocking note: dead header has stale PinholeModelDev layout

The header file `pinhole_trace_kernel.h` defines `PinholeModelDev` with field
order `{width, height, f, c, range}`, while the embedded JIT source in
`PinholeSimulatorHiprt.cpp` (lines 259-268) has the correct order `{width,
height, range, f, c}` matching upstream `rmagine::PinholeModel`. The header is
never included anywhere (dead code -- the actual kernel is the embedded raw
string literal), so this inconsistency does not affect functionality. A future
cleanup pass could either fix the header or delete it.

### ROCm fault-class check

- No hardcoded warpSize/32 assumptions in HIPRT code
- No warp-synchronous primitives (__syncwarp, __shfl, etc.)
- No texture objects requiring 256B pitch alignment
- Memory management: HiprtMesh destructor frees pre_transform; HiprtScene
  destructor frees m_geom and m_mesh_data_device; merged vertex/index buffers
  are intentionally kept alive (documented as POC limitation)
- No AMD-internal account references

### Commit hygiene

- `[ROCm]` prefix, 57 chars title
- Root cause explained (4-byte size mismatch -> Tbm array corruption)
- Claude disclosure present
- No Co-Authored-By: noreply trailer

Verdict: clean. Handing to validator for Stage 2 HIPRT validation.

## Validation 2026-06-05 (linux-gfx90a, GCD 1, HIP_VISIBLE_DEVICES=1)

Fork: AMD-Ecosystem/rmagine moat-port HEAD db7f064 (Stage 2 HIPRT Transform3f fix).
Build: agent_space/rmcl_hiprt_stage2_build (fresh). GPU: AMD Instinct MI250X /
MI250, gfx90a:sramecc+:xnack-, ROCm 7.2.1.

### Build (PASSED)

```
export HIP_VISIBLE_DEVICES=1
export HIPRT_PATH=/var/lib/jenkins/moat/third_party/HIPRT
cmake -S /var/lib/jenkins/moat/projects/rmagine/src \
      -B /var/lib/jenkins/moat/agent_space/rmcl_hiprt_stage2_build \
      -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DRMAGINE_EMBREE_DISABLE=ON -DRMAGINE_OPTIX_DISABLE=ON \
      -DRMAGINE_VULKAN_DISABLE=ON -DRMAGINE_VULKAN_CUDA_INTEROP_DISABLE=ON \
      -DRMAGINE_OUSTER_DISABLE=ON -DRMAGINE_BUILD_TESTS=ON \
      -DRMAGINE_BUILD_TOOLS=OFF
cmake --build /var/lib/jenkins/moat/agent_space/rmcl_hiprt_stage2_build -j
```

81/81 targets built cleanly (includes rmagine_hiprt library). Only pre-existing
nodiscard warnings on hipFree/hipMemcpy/hipMemset (cosmetic, unchanged from
Stage 1). rmagine_hiprt library built: `lib/librmagine_hiprt.so`.

### Stage 1 rmagine_cuda tests (PASSED -- NO REGRESSION)

```
ctest --test-dir /var/lib/jenkins/moat/agent_space/rmcl_hiprt_stage2_build \
      --output-on-failure -R '^cuda_'
```

7/7 PASS (2.23 s):
- cuda_math, cuda_memory, cuda_memory_slicing, cuda_math_svd,
  cuda_math_statistics, cuda_math_reduction, cuda_math_reduction_correctness

### Core tests (PASSED -- NO REGRESSION)

```
ctest --test-dir /var/lib/jenkins/moat/agent_space/rmcl_hiprt_stage2_build \
      --output-on-failure -R '^core_'
```

12/12 PASS (3.09 s):
- core_math, core_memory, core_memory_slicing, core_quaternion, core_math_svd,
  core_math_statistics, core_math_cov_transform, core_math_gaussians,
  core_math_matrix_slicing, core_math_reduction, core_math_cholesky,
  core_math_lie

### Stage 2 HIPRT Pinhole test (PASSED -- HIPRT_PATH fix)

Test harness built cleanly (agent_space/rmcl_hiprt_test/build/test_pinhole),
links against the new rmagine_hiprt library.

**RESOLVED (2026-06-05)**: The prior `hiprtBuildGeometry` error 2 was caused by
missing `HIPRT_PATH` environment variable. HIPRT's Orochi JIT subsystem uses
`HIPRT_PATH` to locate device kernel sources for BVH builder compilation. With
`HIPRT_PATH` set correctly, the test passes:

```
export HIP_VISIBLE_DEVICES=1
export HIPRT_PATH=/var/lib/jenkins/moat/third_party/HIPRT
export LD_LIBRARY_PATH=$HIPRT_PATH/dist/bin/Release:$LD_LIBRARY_PATH
cd $HIPRT_PATH/dist/bin/Release
/var/lib/jenkins/moat/agent_space/rmcl_hiprt_test/test_pinhole
```

Output:
```
[RMagine - CudaContext] CUDA Driver Version / Runtime Version: 70253.21.1 / 70253.21.1
[RMagine - CudaContext] Construct context on device 0 - AMD Instinct MI250X / MI250
=== rmagine_hiprt Pinhole Test ===
Creating HiprtContext...
[HiprtContext] Created on device 0
  Context on device 0
Creating mesh...
  4 vertices, 2 faces
Creating scene...
[HiprtScene] Creating geometry: 4 verts, 2 tris, stride=12
[HiprtScene] Geometry created, getting temp buffer size...
[HiprtScene] Temp buffer size: 512 bytes
[HiprtScene::commit] Built BVH with 4 vertices, 2 faces
  Scene BVH built
Creating PinholeSimulatorHiprt...
Simulating ranges...
[PinholeSimulatorHiprt] JIT kernel compiled successfully

Results (8x8 = 64 rays):
  Hits: 25, Misses: 39
  Range: [2, 2.44949]

  Center pixel range: 2
  Expected (ray through center hitting z=2): 2.0

[PASS] Center pixel range is correct
[PASS] Got 25 hits

=== Test PASSED ===
```

Results validated:
- Center pixel range = 2.0 (exact hit at z=2 quad)
- 25/64 rays hit the quad (correct given FOV and quad geometry)
- BVH build succeeded (hiprtBuildGeometry no longer errors)
- JIT kernel compilation succeeded (hiprtBuildTraceKernels)
- Trace results are geometrically correct

### Verdict

Stage 1 (rmagine_cuda HIP port): VALIDATED. 7/7 cuda_ tests + 12/12 core_ tests
PASS on gfx90a. No regression. Matches prior gfx90a validation at 3d098d5.

Stage 2 (HIPRT Pinhole backend): VALIDATED on gfx90a (db7f064). The HIPRT_PATH
environment variable must be set to the HIPRT SDK root for JIT kernel source
discovery. With this set, BVH build and ray tracing work correctly.

## Validation 2026-06-05 (linux-gfx90a Stage 2 HIPRT All Simulators) -- PASS

Fork: AMD-Ecosystem/rmagine moat-port HEAD 4223818 (Stage 2 complete: all 4 HIPRT
sensor simulators). Build: agent_space/rmcl_stage2_validation (fresh). GPU: AMD
Instinct MI250X / MI250, gfx90a:sramecc+:xnack-, ROCm 7.2.1, GCD 1
(HIP_VISIBLE_DEVICES=1).

### Build (PASSED)

```
export HIP_VISIBLE_DEVICES=1
export HIPRT_PATH=/var/lib/jenkins/moat/third_party/HIPRT
export LD_LIBRARY_PATH=$HIPRT_PATH/dist/bin/Release:$LD_LIBRARY_PATH
cmake -S /var/lib/jenkins/moat/projects/rmagine/src \
      -B /var/lib/jenkins/moat/agent_space/rmcl_stage2_validation \
      -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DRMAGINE_EMBREE_DISABLE=ON -DRMAGINE_OPTIX_DISABLE=ON \
      -DRMAGINE_VULKAN_DISABLE=ON -DRMAGINE_VULKAN_CUDA_INTEROP_DISABLE=ON \
      -DRMAGINE_OUSTER_DISABLE=ON -DRMAGINE_BUILD_TESTS=ON \
      -DRMAGINE_BUILD_TOOLS=OFF
cmake --build /var/lib/jenkins/moat/agent_space/rmcl_stage2_validation -j
```

84/84 targets built cleanly (includes rmagine_hiprt library). Only pre-existing
nodiscard warnings on hipFree/hipMemcpy/hipMemset (cosmetic).

### Stage 1 rmagine_cuda tests (PASSED -- NO REGRESSION)

```
ctest --test-dir /var/lib/jenkins/moat/agent_space/rmcl_stage2_validation \
      --output-on-failure -R '^cuda_'
```

7/7 PASS (2.21 s):
- cuda_math, cuda_memory, cuda_memory_slicing, cuda_math_svd,
  cuda_math_statistics, cuda_math_reduction, cuda_math_reduction_correctness

### Stage 1 core tests (PASSED -- NO REGRESSION)

```
ctest --test-dir /var/lib/jenkins/moat/agent_space/rmcl_stage2_validation \
      --output-on-failure -R '^core_'
```

12/12 PASS (3.06 s):
- core_math, core_memory, core_memory_slicing, core_quaternion, core_math_svd,
  core_math_statistics, core_math_cov_transform, core_math_gaussians,
  core_math_matrix_slicing, core_math_reduction, core_math_cholesky,
  core_math_lie

### Stage 2 HIPRT all simulators test (PASSED)

Test harness: agent_space/rmcl_hiprt_test/test_all_simulators (built against
rmagine_stage2_validation).

```
export HIP_VISIBLE_DEVICES=1
export HIPRT_PATH=/var/lib/jenkins/moat/third_party/HIPRT
export LD_LIBRARY_PATH=$HIPRT_PATH/dist/bin/Release:$LD_LIBRARY_PATH
/var/lib/jenkins/moat/agent_space/rmcl_hiprt_test/build_stage2/test_all_simulators
```

All 4 HIPRT simulators PASSED (5/5 test cases):
1. **PinholeSimulatorHiprt**: 25/64 hits, center range=2.0 (exact)
2. **SphericalSimulatorHiprt**: 0/25 hits (rays point +X, not at quad, expected)
3. **SphericalSimulatorHiprt (rotated pose)**: 25/25 hits at phi=pi/2, center
   range=2.0 (exact)
4. **O1DnSimulatorHiprt**: 4/4 hits, avg range=2.08 (correct given ray angles)
5. **OnDnSimulatorHiprt**: 4/4 hits, avg range=2.0 (exact, all rays point +Z)

### GPU dispatch confirmed (AMD_LOG_LEVEL=3)

Trace kernels confirmed dispatched on `amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-`:
- `pinhole_trace_kernel`
- `spherical_trace_kernel`
- `o1dn_trace_kernel`
- `ondn_trace_kernel`

Plus BVH build kernels:
- `InitGeomData`, `ComputeCentroidBox_TriangleMesh`,
  `ComputeMortonCodes_TriangleMesh`, `EmitTopologyAndFitBounds_TriangleMesh`,
  `Collapse_TrianglePairNode_ScratchNode`, `CompactTasks`,
  `PackLeaves_TriangleMesh_TrianglePairNode`

### Verdict

Stage 1 (rmagine_cuda HIP compute backend): VALIDATED. 7/7 cuda_ tests + 12/12
core_ tests PASS on gfx90a. No regression from prior validations.

Stage 2 (rmagine_hiprt HIPRT ray-tracing backend): VALIDATED on gfx90a (4223818).
All four sensor simulators (Pinhole, Spherical, O1Dn, OnDn) trace rays correctly
against test geometry. BVH build and ray tracing kernels execute successfully on
AMD Instinct MI250X gfx90a. HIPRT_PATH environment variable required for JIT
kernel source discovery (Orochi subsystem).

## Validation 2026-06-07 (windows-gfx1201, HIP_VISIBLE_DEVICES=0) -- PASS

Fork: AMD-Ecosystem/rmagine moat-port HEAD 4223818. GPU: AMD Radeon RX 9070 XT,
gfx1201 (RDNA4, wave32), ROCm 7.14.0a20260604 (TheRock nightly).

### Windows build prerequisites (first run)

Built from source since no package manager provided these; installed once:
- assimp v5.3.1: cloned from github/assimp/assimp, built with amdclang/amdclang++,
  installed to agent_space/assimp_install.
- Eigen3: reused agent_space/eigen_install (from prior gtsam/amgcl builds).
- TBB: reused agent_space/tbb_install (pre-built).
- Boost 1.87.0: header-only usage (boost/algorithm/string/join and
  boost/current_function); created a minimal BoostConfig.cmake pointing to the
  source tree headers (agent_space/boost_cmake_config/).

### Windows-specific build fixes (not source changes; CMake/env only)

1. `-D_USE_MATH_DEFINES -DNOMINMAX` via CMAKE_CXX_FLAGS: rmagine_core uses M_PI /
   M_PI_2 which are not defined by default on Windows (MSVC/clang targeting MSVC
   ABI). Standard Windows fix.

2. `-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE`: rmagine has no __declspec(dllexport)
   decorators; CMake auto-exports all symbols when building SHARED libs.

3. `-fuse-ld=lld-link` stripping: CMake 4.3's Windows-Clang platform module injects
   `-fuse-ld=lld-link` into LINK_FLAGS for all Clang languages including HIP.
   amdclang++ in --hip-link (device-link) mode rejects it. Post-processed
   build.ninja to remove all occurrences. This is a build-env issue, not a source
   issue; the fix is done once after cmake configure before building.

### Configure

```
ROCM="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"

cmake -S /b/develop/moat/projects/rmagine/src \
      -B /b/develop/moat/agent_space/rmcl_gfx1201_build \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="$ROCM/lib/llvm/bin/amdclang.exe" \
      -DCMAKE_CXX_COMPILER="$ROCM/lib/llvm/bin/amdclang++.exe" \
      -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/amdclang++.exe" \
      -DUSE_HIP=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
      -DCMAKE_PREFIX_PATH="$ROCM;$ROCM_CORE" \
      -DBoost_DIR="/b/develop/moat/agent_space/boost_cmake_config" \
      -DEigen3_DIR="/b/develop/agent_space/eigen_install/share/eigen3/cmake" \
      -Dassimp_DIR="/b/develop/moat/agent_space/assimp_install/lib/cmake/assimp-5.3" \
      -DTBB_DIR="/b/develop/moat/agent_space/tbb_install/lib/cmake/TBB" \
      -DRMAGINE_EMBREE_DISABLE=ON -DRMAGINE_OPTIX_DISABLE=ON \
      -DRMAGINE_VULKAN_DISABLE=ON -DRMAGINE_VULKAN_CUDA_INTEROP_DISABLE=ON \
      -DRMAGINE_OUSTER_DISABLE=ON -DRMAGINE_BUILD_TESTS=ON \
      -DRMAGINE_BUILD_TOOLS=OFF \
      -DCMAKE_CXX_FLAGS="-D_USE_MATH_DEFINES -DNOMINMAX" \
      -DCMAKE_C_FLAGS="-D_USE_MATH_DEFINES" \
      -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE

# Post-process to strip -fuse-ld=lld-link from build.ninja (HIP device-link compat)
sed -i 's/-fuse-ld=lld-link//g' /b/develop/moat/agent_space/rmcl_gfx1201_build/build.ninja
```

### Build

```
cmake --build /b/develop/moat/agent_space/rmcl_gfx1201_build -j64
```

38/38 targets built cleanly (rmagine-core.dll, rmagine-cuda.dll, 7 cuda test
exes, 12 core test exes). Only pre-existing nodiscard warnings on hipMemset
macro expansions (unchanged). rmagine_hiprt was skipped (HIPRT SDK absent, as
expected). HIP compiler: amdclang++ 23.0.0 (ROCm 7.14).

### Runtime DLL setup

Copied to bin/ dir (exe-dir search takes priority over System32):
- amdhip64_7.dll, amd_comgr.dll (TheRock runtime)
- hiprand.dll, rocrand.dll, hiprtc0714.dll, hiprtc-builtins0714.dll
- assimp.dll, tbb12.dll (dependency)

### Test results

```
export HIP_VISIBLE_DEVICES=0  # RX 9070 XT (gfx1201)
# Run 1
ctest --test-dir /b/develop/moat/agent_space/rmcl_gfx1201_build --output-on-failure -R "^cuda_"
# 7/7 PASS (1.99 s)
# Run 2 (determinism)
ctest --test-dir /b/develop/moat/agent_space/rmcl_gfx1201_build --output-on-failure -R "^cuda_"
# 7/7 PASS
ctest --test-dir /b/develop/moat/agent_space/rmcl_gfx1201_build --output-on-failure -R "^core_"
# 12/12 PASS (1.84 s)
```

Tests passing:
- cuda_math, cuda_memory, cuda_memory_slicing, cuda_math_svd,
  cuda_math_statistics, cuda_math_reduction (pre-existing suite)
- cuda_math_reduction_correctness (asserting gate for rm::sum/mean/cov vs CPU reference)
- core_math, core_memory, core_memory_slicing, core_quaternion, core_math_svd,
  core_math_statistics, core_math_cov_transform, core_math_gaussians,
  core_math_matrix_slicing, core_math_reduction, core_math_cholesky, core_math_lie

### GPU dispatch confirmed

```
ctest --test-dir rmcl_gfx1201_build --output-on-failure --verbose -R "cuda_math_reduction_correctness"
```

Output on device 0 (AMD Radeon RX 9070 XT):
```
[RMagine - CudaContext] CUDA Driver Version / Runtime Version: 71460.85.0 / 71460.85.0
[RMagine - CudaContext] Construct context on device 0 - AMD Radeon RX 9070 XT
sum = 143.779, 2048, -1571.54
cov(0,0) = -0.188745 (ref -0.188745)
PASS: rm::sum/mean/cov match CPU reference and are deterministic
```

Run 2 bit-identical (same sum/cov values). gfx1201 wave32 reduction correctness
confirmed: full __syncthreads tree (USE_HIP-guarded, no warp tail) produces
correct rm::sum / rm::mean / rm::cov on RDNA4 wave32. Matches gfx90a@4223818 and
gfx1100@4223818.

### Stage 2 HIPRT verdict

HIPRT SDK not present on this host. CMake correctly skipped rmagine_hiprt with a
warning. Stage 1 (rmagine_cuda HIP compute backend) is the validated deliverable.

### Verdict

Stage 1 (rmagine_cuda HIP compute backend): VALIDATED on windows-gfx1201. 7/7
cuda_ tests + 12/12 core_ tests PASS on gfx1201@4223818. No regression. Matches
prior gfx90a and gfx1100 validations at the same SHA.

## Validation 2026-06-16 (windows-gfx1101, HIP_VISIBLE_DEVICES=0) -- PASS

Fork: AMD-Ecosystem/rmagine moat-port HEAD 4223818. GPU: AMD Radeon PRO V710,
gfx1101 (RDNA3, wave32), ROCm 7.14.0a20260604 (TheRock nightly).

### Configure

```
ROCM="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"

cmake -S /b/develop/moat/projects/rmagine/src \
      -B /b/develop/moat/agent_space/rmcl_gfx1101_build \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="$ROCM/lib/llvm/bin/amdclang.exe" \
      -DCMAKE_CXX_COMPILER="$ROCM/lib/llvm/bin/amdclang++.exe" \
      -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/amdclang++.exe" \
      -DUSE_HIP=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
      -DCMAKE_PREFIX_PATH="$ROCM" \
      -DBoost_DIR="/b/develop/moat/agent_space/boost_cmake_config" \
      -DEigen3_DIR="/b/develop/agent_space/eigen_install/share/eigen3/cmake" \
      -Dassimp_DIR="/b/develop/moat/agent_space/assimp_install/lib/cmake/assimp-5.3" \
      -DTBB_DIR="/b/develop/moat/agent_space/tbb_install/lib/cmake/TBB" \
      -DRMAGINE_EMBREE_DISABLE=ON -DRMAGINE_OPTIX_DISABLE=ON \
      -DRMAGINE_VULKAN_DISABLE=ON -DRMAGINE_VULKAN_CUDA_INTEROP_DISABLE=ON \
      -DRMAGINE_OUSTER_DISABLE=ON -DRMAGINE_BUILD_TESTS=ON \
      -DRMAGINE_BUILD_TOOLS=OFF \
      -DCMAKE_CXX_FLAGS="-D_USE_MATH_DEFINES -DNOMINMAX" \
      -DCMAKE_C_FLAGS="-D_USE_MATH_DEFINES" \
      -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE

# Post-process to strip -fuse-ld=lld-link from build.ninja (HIP device-link compat)
sed -i 's/-fuse-ld=lld-link//g' /b/develop/moat/agent_space/rmcl_gfx1101_build/build.ninja
```

rmagine_hiprt skipped (HIPRT SDK absent, expected).

### Build

```
cmake --build /b/develop/moat/agent_space/rmcl_gfx1101_build -j64
```

72/72 targets built cleanly (rmagine-core.dll, rmagine-cuda.dll, 7 cuda test
exes, 12 core test exes). Only pre-existing nodiscard warnings on hipMemset
macro expansions (unchanged). rmagine_hiprt was skipped (HIPRT SDK absent, as
expected). HIP compiler: amdclang++ 23.0.0 (ROCm 7.14).

### Runtime DLL setup

Copied to bin/ dir (exe-dir search takes priority over System32):
- amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll,
  hiprtc-builtins0714.dll (from _rocm_sdk_core/bin)
- hiprand.dll, rocrand.dll (from _rocm_sdk_libraries/bin)
- assimp.dll, tbb12.dll (from gfx1201 build, same source)

### Test results

```
export HIP_VISIBLE_DEVICES=0  # Radeon PRO V710 (gfx1101)
# Run 1
ctest --test-dir /b/develop/moat/agent_space/rmcl_gfx1101_build --output-on-failure -R "^cuda_"
# 7/7 PASS (2.42 s)
# Run 2 (determinism)
ctest --test-dir /b/develop/moat/agent_space/rmcl_gfx1101_build --output-on-failure -R "^cuda_"
# 7/7 PASS
ctest --test-dir /b/develop/moat/agent_space/rmcl_gfx1101_build --output-on-failure -R "^core_"
# 12/12 PASS (1.88 s)
```

Tests passing:
- cuda_math, cuda_memory, cuda_memory_slicing, cuda_math_svd,
  cuda_math_statistics, cuda_math_reduction (pre-existing suite)
- cuda_math_reduction_correctness (asserting gate for rm::sum/mean/cov vs CPU reference)
- core_math, core_memory, core_memory_slicing, core_quaternion, core_math_svd,
  core_math_statistics, core_math_cov_transform, core_math_gaussians,
  core_math_matrix_slicing, core_math_reduction, core_math_cholesky, core_math_lie

### GPU dispatch confirmed

```
ctest --test-dir rmcl_gfx1101_build --output-on-failure --verbose -R "cuda_math_reduction_correctness"
```

Output on device 0 (AMD Radeon PRO V710):
```
[RMagine - CudaContext] CUDA Driver Version / Runtime Version: 71460.85.0 / 71460.85.0
[RMagine - CudaContext] Construct context on device 0 - AMD Radeon PRO V710
sum = 143.779, 2048, -1571.54
cov(0,0) = -0.188745 (ref -0.188745)
PASS: rm::sum/mean/cov match CPU reference and are deterministic
```

Run 2 bit-identical (same sum/cov values). gfx1101 wave32 reduction correctness
confirmed: full __syncthreads tree (USE_HIP-guarded, no warp tail) produces
correct rm::sum / rm::mean / rm::cov on RDNA3 wave32 (Radeon PRO V710).
Matches gfx90a@4223818, gfx1100@4223818, and gfx1201@4223818.

### Stage 2 HIPRT verdict

HIPRT SDK not present on this host. CMake correctly skipped rmagine_hiprt with a
warning. Stage 1 (rmagine_cuda HIP compute backend) is the validated deliverable.

### Verdict

Stage 1 (rmagine_cuda HIP compute backend): VALIDATED on windows-gfx1101. 7/7
cuda_ tests + 12/12 core_ tests PASS on gfx1101@4223818. No regression. Matches
prior gfx90a, gfx1100, and gfx1201 validations at the same SHA.

## Install as a dependency

rmagine installs a normal CMake package, so a dependent consumes it with
`find_package(rmagine ...)` against a staging prefix. Verified end to end on
linux-gfx1100 at moat-port 0aea7af (see the two "Delta round 2026-08-12"
sections below -- at 4223818 and earlier this did NOT work; the exported config
resolved CUDA at consumer time and failed the whole package on a CUDA-free
host, and the public headers were not includable from a plain C++ consumer).

Host deps (Ubuntu 24.04): `sudo apt install -y libtbb-dev libboost-dev
libeigen3-dev libassimp-dev cmake`.

### Build and install the dependency

```
git clone -b moat-port https://github.com/AMD-Ecosystem/rmagine _deps/rmagine/src
cmake -S _deps/rmagine/src -B _deps/rmagine/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DRMAGINE_EMBREE_DISABLE=ON -DRMAGINE_OPTIX_DISABLE=ON \
  -DRMAGINE_VULKAN_DISABLE=ON -DRMAGINE_VULKAN_CUDA_INTEROP_DISABLE=ON \
  -DRMAGINE_OUSTER_DISABLE=ON \
  -DRMAGINE_BUILD_TESTS=OFF -DRMAGINE_BUILD_TOOLS=OFF \
  -DCMAKE_INSTALL_PREFIX=$PWD/_deps/rmagine/install
cmake --build _deps/rmagine/build -j
cmake --install _deps/rmagine/build
```

Set `-DCMAKE_HIP_ARCHITECTURES` to the consuming host's arch (gfx90a, gfx1100,
gfx1101, gfx1201 are all validated). Leave `RMAGINE_BUILD_TESTS=ON` if you want
the `cuda_`/`core_` ctest suites in the dependency's build tree as a smoke check.

### Install layout

```
<prefix>/include/rmagine-2.4.2/rmagine/...        headers (core + cuda)
<prefix>/lib/librmagine-core.so.2.4.2             CPU library
<prefix>/lib/librmagine-cuda.so.2.4.2             GPU compute library (HIP)
<prefix>/lib/cmake/rmagine-2.4.2/rmagine-config.cmake
<prefix>/lib/cmake/rmagine-2.4.2/rmagine-core-config.cmake
<prefix>/lib/cmake/rmagine-2.4.2/rmagine-cuda-config.cmake
<prefix>/lib/cmake/rmagine-2.4.2/rmagine-{core,cuda}-targets.cmake
<prefix>/share/rmagine-2.4.2/package.xml
```

### What a dependent does

```cmake
find_package(rmagine 2.4 REQUIRED
  COMPONENTS core
  OPTIONAL_COMPONENTS embree cuda optix vulkan vulkan-cuda-interop)
target_link_libraries(<target> PRIVATE rmagine::core rmagine::cuda)
```

Configure the dependent with

```
-DCMAKE_PREFIX_PATH="/abs/path/to/_deps/rmagine/install;/opt/rocm"
```

(append, do not replace, if the dependent already needs a prefix path; use
`$ROCM_PATH` in place of `/opt/rocm` if ROCm is installed elsewhere). The ROCm
root is REQUIRED, not optional: `rmagine-cuda-config.cmake` runs
`find_dependency(hip)` and `find_dependency(hiprand)`, and if CMake cannot find
them the whole `find_package(rmagine)` call fails hard and takes `rmagine::core`
down with it:

```
CMake Error at .../CMakeFindDependencyMacro.cmake:76 (find_package):
  Could not find a package configuration file provided by "hip" ...
Call Stack (most recent call first):
  .../rmagine-cuda-config.cmake:46 (find_dependency)
  .../rmagine-config.cmake:55 (include)
  CMakeLists.txt:4 (find_package)
```

It is easy to miss this because a host with `/opt/rocm/bin` on `PATH` gets the
ROCm prefix for free (CMake derives search prefixes from `PATH`), so the recipe
appears to work there and fails on a host that installs ROCm without touching
`PATH`. Verified both ways on linux-gfx1100: with `/opt/rocm/bin` stripped from
`PATH` and `ROCM_PATH`/`HIP_PATH`/`ROCM_HOME` unset, the prefix-path-only form
fails with the error above and the two-entry form configures, builds and runs.

At runtime the dependent needs `<prefix>/lib` on `LD_LIBRARY_PATH` unless it
sets an RPATH.

Targets that exist in a ROCm build: `rmagine::core`, `rmagine::cuda`. Targets
that do NOT exist: `rmagine::embree` (needs Embree), `rmagine::optix`,
`rmagine::vulkan`, `rmagine::vulkan-cuda-interop`, `rmagine::ouster`. Ask for
them only through `OPTIONAL_COMPONENTS`. `rmagine_COMPONENTS_FOUND` reports
what was actually found (`core;cuda` for the configuration above), and
`rmagine_cuda_USE_HIP` is `ON` in a ROCm build, so a dependent that has its own
`.cu` can switch its own toolchain on it instead of guessing.

The HIPRT ray-tracing backend (`rmagine_hiprt`) has no install rules and is not
part of the package; it also requires the HIPRT SDK via `HIPRT_PATH`. A
dependent that needs GPU ray casting on AMD cannot get it through
`find_package` today.

A consumer does NOT need the HIP language enabled to link `rmagine::cuda`:
`hip::device` is PRIVATE on the library, so plain g++/C++ consumers work. Only
`hip::host` and `hip::hiprand` are in the public interface, and the package
config finds them for you (given the ROCm root on the prefix path, above).

All 16 installed `rmagine_cuda` headers compile TOGETHER at the top of a plain
C++ translation unit with nothing above them. That much is a gate, not a hope:
`tests/cuda/public_headers.cpp` (ctest `cuda_public_headers`) includes all of
them ahead of any standard header and is compiled by the host C++ compiler.
Until 0aea7af it was false -- see the delta round below.

What that gate does NOT cover is per-header independence. Everything below the
first include in that TU is standing on the includes of the headers above it,
so a single TU cannot show that any one header is includable on its own. Two
of the 16 are not, as the very first include of a TU:

- `rmagine/math/linalg.cuh` -- includes only `rmagine/math/types.h`, then
  declares `__device__` functions at lines 20 and 23 with nothing in scope that
  defines `__device__`. Fails with "`__device__` does not name a type".
- `rmagine/util/cuda/CudaHelper.hpp` -- throws `std::runtime_error` at lines 51
  and 64 without `<stdexcept>`. Fails with "'runtime_error' is not a member of
  'std'".

Both are PRE-EXISTING upstream defects, not port regressions, and both fail the
same way on the NVIDIA side: checked standalone at upstream 6b93e86 against the
CUDA 12.8 headers with g++ 13.3.0, same two errors at the same lines. The port
left `linalg.cuh` untouched and changed only `CudaHelper.hpp`'s include of
`cuda_runtime.h` to the compat header. The other 14 do pass standalone on both
backends (checked one TU per header, both include paths).

So a dependent that reaches for either of those two as its first include needs
`<stdexcept>` (or any header pulling it) above `CudaHelper.hpp`, and a device
compiler or the runtime header above `linalg.cuh` -- exactly as it would
against upstream CUDA. The honest gate for per-header independence is one TU
per header; that is not built here, and building it would want the two
two-line upstream fixes with it.

## Delta round 2026-08-12 (porter, linux-gfx1100) -- exported CMake config fix

Scope: make an installed ROCm build of rmagine consumable. Install-side only;
no device code touched. Fork moat-port 4223818 -> e9cbdbf (2 commits).

### The bug (reproduced here before fixing)

`src/rmagine_cuda/cmake/rmagine-cuda-config.cmake.in` resolved a CUDA toolkit at
consumer time regardless of backend:

```cmake
if(@CUDAToolkit_FOUND@)
    find_dependency(CUDAToolkit)
else(@CUDAToolkit_FOUND@)
    find_dependency(CUDA)
endif(@CUDAToolkit_FOUND@)
```

Under `USE_HIP` the CUDA branch of `rmagine_cuda/CMakeLists.txt` never runs, so
`CUDAToolkit_FOUND` is undefined and the INSTALLED file literally reads `if()`
(valid CMake, evaluates false) and falls through to `find_dependency(CUDA)`.
Observed on this CUDA-free gfx1100 host with CMake 3.31.6, from the throwaway
consumer described above:

```
CMake Error at .../Modules/FindCUDA.cmake:883 (message):
  Specify CUDA_TOOLKIT_ROOT_DIR
Call Stack (most recent call first):
  .../rmagine-2.4.2/rmagine-cuda-config.cmake:38 (find_dependency)
  .../rmagine-2.4.2/rmagine-config.cmake:55 (include)
  CMakeLists.txt:4 (find_package)
```

`find_dependency` failing returns out of `rmagine-config.cmake` entirely, so the
whole package is unusable -- `rmagine::core` included -- and
`OPTIONAL_COMPONENTS` does not rescue it (rmagine-config.cmake.in includes every
component config that exists on disk, with no optional/required distinction).
The rmcl planner predicted `rmagine_FOUND=0`; on this CMake the module raises a
hard error first. Same outcome: the dependent never reaches its own code.

Second half, also real: `rmagine-cuda-targets.cmake` records `hip::host` and
`hip::hiprand` in `INTERFACE_LINK_LIBRARIES`, so the config must
`find_dependency(hip)` and `find_dependency(hiprand)` or the consumer fails at
GENERATE time (after a successful configure) on a link to a nonexistent target.

### The fix (cfc475d)

One file, `rmagine-cuda-config.cmake.in`: substitute the backend into
`rmagine_cuda_USE_HIP` and branch on it; on HIP find `hip` + `hiprand` and look
for no CUDA at all; otherwise run the original CUDAToolkit/CUDA/check_language
block unchanged.

Tested as a VARIABLE, not as `if("@USE_HIP@")`. `if("ON")` is a quoted constant
and depends on the consumer's CMP0012: with the policy unset (e.g. a bare
`cmake -P` script) `if("ON")` is FALSE and the consumer silently takes the CUDA
branch. Confirmed here with a 4-case truth table before choosing the form.

e9cbdbf is a comment-only follow-up removing in-house shorthand from two
comments (`utils/jargon.py` flagged them; they were pre-existing at 4223818).

### Evidence (linux-gfx1100, Radeon Pro W7800, ROCm 7.2.3, HIP_VISIBLE_DEVICES=0)

Build + install of the fork at e9cbdbf into a staging prefix, then a throwaway
consumer project (`agent_space/consumer/`, scratch) that makes the exact
`find_package` call rmcl makes and links `rmagine::core` + `rmagine::cuda`:

```
-- rmagine_FOUND=1
-- rmagine_COMPONENTS_FOUND=core;cuda
-- Generating done
[RMagine - CudaContext] Construct context on device 0 - AMD Radeon Pro W7800 48GB
sum = 4096.000000 8192.000000 12288.000000 (expected 4096.000000 8192.000000 12288.000000)
PASS: consumed rmagine::core + rmagine::cuda from the install prefix
```

The consumer runs `rm::sum` over a 4096-element device buffer, so the pass is a
real GPU dispatch through the installed library, not just a link check.

In-tree suites at e9cbdbf, same host: `ctest -R '^cuda_'` 7/7 PASS,
`ctest -R '^core_'` 12/12 PASS. Unchanged from 4223818, as expected for an
install-side change.

### Consequence for the other platforms

Advancing head_sha to e9cbdbf puts linux-gfx90a, windows-gfx1101 and
windows-gfx1201 (and this platform's own record) behind. The delta is
`classify` = mixed only because it does not recognise `.cmake.in`; in substance
it is one generated-package-config file plus two comments, with no device code
and no change to any compiled artifact. A carry-forward is the appropriate
disposition if a maintainer agrees; the one thing a revalidation SHOULD add
that the old evidence does not cover is an install + `find_package` consume on
that platform, since that is the path this round fixed.

### Gotchas

- An installed CMake package config is invisible to every in-tree gate. rmagine
  had four platform validations, a review pass, and 19 passing ctests while its
  installed package was unusable by anyone, because every gate ran from the
  build tree. If a project installs a package config, consume it from a
  separate project as part of the port. Promoted to the `cuda-to-rocm` skill
  (`references/strategy-a-cmake.md`, "The installed package config is part of
  the port").
- `libassimp-dev` was missing on this host and is a hard `find_package(assimp)`
  in the top-level CMakeLists; `sudo apt install -y libassimp-dev` fixes it.
- The ROCm build is still undocumented in rmagine's own README (no `USE_HIP`
  anywhere in it). Deliberately NOT fixed in this round: the round was scoped to
  the install bug, and the README defers build detail to an external wiki
  (uos.github.io/rmagine_docs) that is a separate repository. It needs a
  decision on house style before the upstream PR.

## Review 2026-08-12 (reviewer, linux-gfx1100, delta round) -- CHANGES REQUESTED

Scope: `git diff 4223818...HEAD` on AMD-Ecosystem/rmagine `moat-port` (e9cbdbf), three
files, plus the record and commit-message state of the whole branch. Working tree at
`projects/rmagine/src` is clean. The CMake fix itself is correct and I verified it
independently (see "Verified, no action" at the end); the findings below are the
consume-path defect the round's own verification did not reach, two upstream-visible
text problems, and record hygiene.

### 1. The compat shim pulls a device-side rocRAND header into four public headers, and a plain-C++ consumer of the installed package fails to compile

`src/rmagine_cuda/include/rmagine/util/cuda/cuda_to_hip.h:23` includes
`<hiprand/hiprand_kernel.h>` unconditionally inside the HIP branch
(`cuda_to_hip.h:19`). That shim replaced the CUDA includes in six installed public
headers, but upstream only four of them pulled the runtime/driver headers and never
touched cuRAND:

- `util/cuda/CudaContext.hpp:38` -- upstream `<cuda_runtime.h>` + `<cuda.h>`
- `util/cuda/CudaHelper.hpp:38` -- upstream `<cuda_runtime.h>`
- `util/cuda/CudaDebug.hpp:41` -- upstream `<cuda_runtime.h>`
- `util/cuda/CudaStream.hpp:4` -- upstream `<cuda_runtime.h>` + `<cuda.h>`

(`random.cuh:5` and `noise/NoiseCuda.hpp:45` genuinely need the device header --
they name `curandState`/`hiprandState` in host-visible declarations -- so they are
not part of this finding.)

Consequence, reproduced here against the install prefix built at e9cbdbf
(`agent_space/install_final`), consumer compiled with the default `/usr/bin/c++`,
linking `rmagine::core rmagine::cuda`:

```
#include <rmagine/util/cuda/CudaContext.hpp>
#include <cstdio>
int main(){ printf("headers ok\n"); return 0; }
```

```
In file included from /opt/rocm/include/rocrand/rocrand_kernel.h:28,
                 from /opt/rocm/include/hiprand/hiprand_kernel_rocm.h:37,
                 from /opt/rocm/include/hiprand/hiprand_kernel.h:110,
                 from .../rmagine/util/cuda/cuda_to_hip.h:23,
                 from .../rmagine/util/cuda/CudaContext.hpp:38,
/opt/rocm/include/rocrand/rocrand_mtgp32.h:443:9: error: 'printf' was not declared in this scope
```

Moving `#include <cstdio>` above the rmagine include makes the same TU compile, so
the breakage is include-order dependent -- the worst shape for a downstream to
diagnose. This is the documented "compat header must be host-includable" fault class
in the `cuda-to-rocm` skill: a device-side header leaking into host TUs.

Nothing in tree covers it. No test under `tests/` includes any of the four headers
(checked), and `agent_space/consumer/main.cpp` includes only `Memory.hpp`,
`MemoryCuda.hpp`, `memory_math.cuh` and `math/types.h`, whose include chain reaches
`util/cuda/cuda_definitions.h` and never `cuda_to_hip.h` -- which is why the round's
end-to-end consume check passed while this path was broken. It also makes
notes.md:1311 ("plain g++/C++ consumers work") true only for the header subset that
was actually tried.

Fix: stop the shim from dragging hipRAND into headers that never needed it -- have
the two headers that really use `hiprandState` pull the device header themselves and
drop the unconditional include from `cuda_to_hip.h`, restoring upstream's per-header
include footprint. A bare `#include <cstdio>` ahead of the hipRAND include papers
over this one rocRAND bug and leaves the footprint regression in place. Extend the
throwaway consumer to include the four headers so the fix has a gate.

### 2. Two jargon hits on the branch will refuse the review PR; the "publish-time squash" plan does not exist

`python3 utils/jargon.py --port rmagine` reports:

```
commit 3d098d58e:11: 'Strategy A'
commit 3d098d58e:24: 'followers'
```

`utils/upstream.py:560-569` (`open_review_pr`) scans `jargon.port_range(...)` --
every commit message on the branch -- and returns a `jargon` refusal before it will
even print the `--review` preview, let alone open the PR; `utils/upstream.py:654-677`
re-scans the PR's commit messages at publish time. There is no squash anywhere in
that path: the PR is opened `--head <branch> --base <base>` with all commits, and
the comment at `upstream.py:553-559` says in as many words that this is the last
cheap point and that fixing it after the review PR costs every architecture its
validation. So the reasoning that the publish path absorbs these is wrong; they are
a hard gate.

I do not accept "rewriting validated history is forbidden" either -- AGENTS.md
prices a `head_sha` advance, it does not forbid one. This round has already moved
`head_sha` 4223818 -> e9cbdbf and already put all four platform records behind, and
rewording a commit message changes SHAs without changing a single tree, so the
carry-forward argument for the other three platforms is exactly as strong after the
reword as before it. Doing it in this round is strictly cheaper than any later
moment. Reword both lines of 3d098d58e (name the technique, e.g. "a compatibility
header"; name the GPUs instead of "followers") and re-run `jargon.py --port`.

### 3. cfc475d's commit body describes a failure mode its own pasted evidence contradicts

The body says the CUDA fall-through means "find_package reports the whole rmagine
package as not found, rmagine::core included ... A downstream project then dies at
its first target_link_libraries(... rmagine::core ...), before any of its own code is
compiled." I reproduced the pre-fix config against the same install prefix (old
`if()`/`find_dependency(CUDA)` block restored by hand, CMake 3.31.6, no CUDA on the
host) and what happens is a hard `message(FATAL_ERROR)` inside `FindCUDA.cmake:883`
during `find_package(rmagine)`: configure aborts there and never reaches any
`target_link_libraries`. That is the same error text the commit pastes four
paragraphs later, so the narrative and the evidence in one commit disagree.
notes.md:1353 already records the correct version ("the module raises a hard error
first"). Reword the body to the mechanism that was actually observed. This is
upstream-visible text on a project whose review history already turned once on
commit prose overstating what was seen.

### 4. The "Install as a dependency" recipe omits ROCm from the consumer's prefix path, and reproduces the failure this round fixed

notes.md:1293-1296 tells a dependent to configure with
`-DCMAKE_PREFIX_PATH=/abs/path/to/_deps/rmagine/install` and nothing else. That
worked on this host only because `/opt/rocm/bin` is on `PATH`, which CMake turns
into a search prefix. With ROCm off `PATH` and the ROCM_* environment unset,
following the recipe literally gives:

```
CMake Error at .../CMakeFindDependencyMacro.cmake:76 (find_package):
  Could not find a package configuration file provided by "hip" ...
Call Stack (most recent call first):
  .../rmagine-cuda-config.cmake:46 (find_dependency)
  .../rmagine-config.cmake:55 (include)
  CMakeLists.txt:4 (find_package)
```

-- the same shape of hard failure out of `rmagine-config.cmake`, taking
`rmagine::core` down with it, that the round exists to remove. Reproduced here.
State in the recipe that the consumer's prefix path must also carry the ROCm root
(`-DCMAKE_PREFIX_PATH="<install>;$ROCM_PATH"`, or `/opt/rocm`), since rmcl's porter
will follow the recipe literally on a host that is not this one.

Everything else in that section I checked against the real install tree at
`agent_space/install_final` and it matches: layout, the `-2.4.2` suffixes,
`share/rmagine-2.4.2/package.xml`, `rmagine::core`/`rmagine::cuda` as the only
targets a ROCm build exports, `rmagine_COMPONENTS_FOUND=core;cuda`,
`rmagine_cuda_USE_HIP=ON` in the generated config, and `rmagine_hiprt` having zero
`install()` rules.

### 5. Record hygiene: this project's records are still rmcl's

- `projects/rmagine/notes.md:1` is titled `# rmcl notes` and opens by explaining
  that rmcl is the named MOAT project. It is not; `projects/rmcl` no longer exists.
- `projects/rmagine/plan.md:1` is rmcl's plan ("rmcl -- ROCm/HIP porting plan"),
  not rmagine's. The project has no plan of its own.
- notes.md contains 10 references to `projects/rmcl/rmagine_src` and
  `projects/rmcl/src`, paths that no longer exist, inside build and validation
  recipes a reader is meant to re-run.

Retitle notes.md, replace or retarget plan.md, and rewrite the stale paths to
`projects/rmagine/src`.

### Open item ruled on: README does not mention USE_HIP

Confirmed: `USE_HIP`, `rocm` and `hip` appear nowhere in `README.md` (255 lines).
The README's "Backends" table is about ray-tracing backends (Embree/OptiX/Vulkan)
and does not list the CUDA compute backend at all, and "Installation and Usage"
defers advanced options to the external wiki. My view: this does not block review or
validation, and I agree it was out of scope for a round scoped to the install bug.
It does need to be settled before the review PR is opened, and the cheap resolution
is not a README edit at all -- the upstream PR body must state the option, its
default (`OFF`), and that the NVIDIA path is unchanged, and offer the maintainer a
README or wiki sentence rather than pushing one unasked into a file whose house
style sends build detail elsewhere. Record that decision in the PR body draft, do
not leave it implicit.

### Verified, no action

Fact-checked independently rather than accepted from the round's account:

- `set(rmagine_cuda_USE_HIP @USE_HIP@)` + `if(rmagine_cuda_USE_HIP)` is genuinely
  policy-independent, and the rejected `if("@USE_HIP@")` genuinely is not. With
  CMP0012 unset (`cmake -P`, CMake 3.31.6): `if("ON")` is FALSE with a CMP0012
  warning, `if(<var>)` where the var is `ON` is TRUE, `OFF` is FALSE, and an
  undefined var is FALSE. `USE_HIP` is `option(... OFF)` at CMakeLists.txt:15 so
  the substitution is always `ON`/`OFF`, and an empty substitution would unset the
  variable and correctly fall to the CUDA branch. The load-bearing line holds.
- The CUDA branch is behaviour-identical to upstream: only the dropped `else(...)`
  /`endif(...)` arguments, the trailing whitespace after `check_language(CUDA)`, and
  indentation changed. The file was untouched by the port before this commit.
- `find_dependency(hip)`/`find_dependency(hiprand)` are required, not defensive:
  the installed `rmagine-cuda-targets.cmake:63` carries
  `INTERFACE_LINK_LIBRARIES "rmagine::core;hip::host;hip::hiprand"`. Include order
  in the config is safe -- the exported missing-target check
  (`rmagine-cuda-targets.cmake:104-119`) only covers `rmagine::core`, and imported
  link interfaces resolve at generate time. `hip::host` is also what carries
  `__HIP_PLATFORM_AMD__=1` (`/opt/rocm/lib/cmake/hip/hip-config-amd.cmake:141`) into
  a consumer, which is what makes `cuda_to_hip.h` take the HIP branch downstream.
- No device code in the delta: the only non-CMake file is a comment inside the
  shim's leading block comment, with no `*/` reintroduced and no preprocessor
  effect. `src/rmagine_cuda/CMakeLists.txt` changed a comment only. Carrying the
  other three platforms forward is defensible on the code, subject to finding 1
  changing that.
- Commit hygiene on the two new commits: `[ROCm]` titles at 55 and 47 characters,
  AI-assistance disclosed, Test Plan in fenced blocks, no `Co-Authored-By`, no
  non-ASCII in messages or added lines, no AMD-internal account references.

## Delta round 2026-08-12b (porter, linux-gfx1100) -- review findings 1, 3, 4, 5

Answers the review above. Findings 1/3/4/5 only; finding 2 (the two jargon hits in
3d098d58e) was explicitly held back, because rewording that commit rewrites 4223818
and all four platforms' `validated_sha` point at it -- destroying that evidence is a
person's call and it is with jeff. `jargon.py --port rmagine` therefore still reports
exactly those two hits and nothing else.

Fork moat-port e9cbdbf -> 0aea7af. The two commits above 4223818 were rewritten
(message-only) and one new commit added:

- 7382086 = cfc475d with a corrected body (finding 3), same tree
- 74ba640 = e9cbdbf replayed, same tree
- 0aea7af = new, the header fix (finding 1)

### Finding 1: the compat header dragged hipRAND into four public headers

Reproduced first, exactly as the reviewer described, against the round-2 install
prefix with the default host compiler:

```
/usr/bin/c++ -std=c++17 -c hdrs.cpp -o hdrs.o \
  -I install_final/include/rmagine-2.4.2 -I/opt/rocm/include -D__HIP_PLATFORM_AMD__=1
# hdrs.cpp: #include <rmagine/util/cuda/CudaContext.hpp> then #include <cstdio>
/opt/rocm/include/rocrand/rocrand_mtgp32.h:443:9: error: 'printf' was not declared in this scope
```

Fix: `cuda_to_hip.h` keeps only the runtime/driver mapping; the cuRAND mapping and
`<hiprand/hiprand_kernel.h>` move to a new sibling
`include/rmagine/util/cuda/curand_to_hiprand.h`, included by exactly the two headers
that name `curandState` in a declaration (`util/cuda/random.cuh`,
`noise/NoiseCuda.hpp`) and that pulled cuRAND upstream too. Every other header is
back to its upstream include footprint. All curand macro users reach one of those
two headers (checked by grep over the whole tree), so nothing else needed an edit.

The new header also does `#include <cstdio>` ahead of the hipRAND include. That is
on top of the footprint fix, not instead of it: it makes the two RNG headers
host-includable on the ROCm path, which they are not without it. This repairs a
ROCm-only gap rather than restoring parity with a broken NVIDIA path -- cuRAND's
device header IS host-includable (`#include <curand.h>` then `<curand_kernel.h>`
in a plain g++ `-fsyntax-only` TU against the CUDA 12.8 headers exits 0), while
hipRAND's is not, because `rocrand_mtgp32.h:443` calls `printf` with only
`<stdlib.h>` and `<string.h>` above it. Without the `<cstdio>` the port's HIP path
would have been BEHIND upstream here, not level with it. Confirmed a plain-c++ TU
including `NoiseCuda.hpp` compiles with it.

The gate (this is the part the round was really about): new
`tests/cuda/public_headers.cpp` / ctest `cuda_public_headers` includes all 16
installed rmagine_cuda headers at the very top of a plain C++ TU with nothing above
them, then queries the device count through the mapped runtime. It is compiled by
`CMAKE_CXX_COMPILER` (/usr/bin/c++ here), NOT by the HIP compiler, so it is a real
downstream simulation. Verified it bites: putting `<hiprand/hiprand_kernel.h>` back
into `cuda_to_hip.h` and rebuilding gives

```
FAILED: tests/cuda/CMakeFiles/rmagine_tests_cuda_public_headers.dir/public_headers.cpp.o
/opt/rocm/include/rocrand/rocrand_mtgp32.h:443:9: error: 'printf' was not declared in this scope
```

Also covered from the install side: `agent_space/consumer3` gained a second
executable whose TU includes CudaContext/CudaDebug/CudaHelper/CudaStream and
NoiseCuda.hpp before `<cstdio>`.

### Findings 3 and 4

3: cfc475d's body claimed the consumer "dies at its first target_link_libraries";
the real mechanism, which its own pasted evidence shows, is `message(FATAL_ERROR)`
in FindCUDA.cmake aborting inside `find_package(rmagine)`. Reworded (7382086). While
in there, its Test Plan's `-DCMAKE_PREFIX_PATH=/path/to/install` got the ROCm root
too, since that commit is what introduced the `find_dependency(hip)` requirement.

4: reproduced and fixed in "Install as a dependency" above. See that section for the
error and the verification; short version, the consumer's prefix path needs the ROCm
root as a second entry and the old recipe only worked because `/opt/rocm/bin` was on
`PATH`.

### Finding 5

notes.md retitled `# rmagine notes` with a short rename history at the top; the 10
`projects/rmcl/...` paths in build and validation recipes rewritten to
`projects/rmagine/src` (the reviewer's own quotes of those paths are left as
written); plan.md rewritten to be rmagine's plan, with rmcl demoted to the
downstream consumer it is and the Stage 2 HIPRT work it never covered recorded.

### Evidence (linux-gfx1100, AMD Radeon Pro W7800, ROCm 7.2.3, HIP_VISIBLE_DEVICES=0)

Build tree `agent_space/build_final` reconfigured to install into
`agent_space/install_round3`; everything else as in the recipe above.

```
ctest --test-dir agent_space/build_final -R '^cuda_'   # 8/8 PASS (7 old + cuda_public_headers)
ctest --test-dir agent_space/build_final -R '^core_'   # 12/12 PASS
```

Installed-package consume, with ROCm deliberately off `PATH` and ROCM_PATH/HIP_PATH/
ROCM_HOME unset:

```
env -u ROCM_PATH -u HIP_PATH -u ROCM_HOME PATH="$PATH_without_rocm" \
  cmake -S consumer3 -B consumer3_build -G Ninja \
  -DCMAKE_PREFIX_PATH="$PWD/install_round3;/opt/rocm"
cmake --build consumer3_build
LD_LIBRARY_PATH=$PWD/install_round3/lib ./consumer3_build/consumer_headers
LD_LIBRARY_PATH=$PWD/install_round3/lib ./consumer3_build/consumer
```

```
-- rmagine_FOUND=1
-- rmagine_COMPONENTS_FOUND=core;cuda
PASS: installed public headers include cleanly from plain C++
[RMagine - CudaContext] Construct context on device 0 - AMD Radeon Pro W7800 48GB
sum = 4096.000000 8192.000000 12288.000000 (expected 4096.000000 8192.000000 12288.000000)
PASS: consumed rmagine::core + rmagine::cuda from the install prefix
```

### Consequence for the other platforms

0aea7af touches headers and adds a test, so unlike the round-2 delta this is not a
pure install-side change: a revalidation actually recompiles something. The four
platform records were already behind at e9cbdbf and stay behind. The new ctest is
plain C++ and should pass anywhere the rest of the suite builds; the one platform
where it is worth watching is Windows, where the host compiler is MSVC rather than
gcc and the rocRAND header's `printf` habit may behave differently.

### Gotchas

- A compat header is included by everything, so anything it includes is in the public
  include footprint of every header that uses it. Adding the device RNG header there
  was invisible in-tree because every in-tree TU reaches those headers through some
  other include first. Promoted to the `cuda-to-rocm` skill
  (`references/strategy-a-cmake.md`) as a header-hygiene rule with a test recipe,
  since it generalises to any project using the single-compat-header approach.
- `/opt/rocm/bin` on `PATH` silently supplies the ROCm CMake prefix. Any "consume the
  installed package" recipe verified on a host with ROCm on `PATH` is under-tested;
  strip it and re-run before writing the recipe down.

## History rewrite 2026-08-12 (jargon gate, review finding 2)

`utils/jargon.py --port rmagine` had two hits, both in the *message* of
`3d098d58e` ("Strategy A", "followers"). The round-3 porter left them, reasoning
that a publish-time squash would absorb them. That is wrong, and it is the kind of
wrong that only shows up at the worst moment: `utils/upstream.py:634-645` pulls
every commit message on the branch (`gh api repos/<slug>/pulls/<n>/commits`) and
scans it with `jargon.scan_text`, and `:560-569` refuses even the `--review`
preview. There is no squash anywhere in that path. Unfixed, this port could never
be published.

Fixing it meant rewording a commit that four platforms' `validated_sha` depended on
(`4223818c9` is a descendant of `3d098d58e`), so a person ruled it: jeff, 2026-08-12,
"reword now, re-record the evidence" -- rewriting is preferable to letting the gate
sit until publication, and the evidence is repointed rather than orphaned.

### What was done

The message change is two phrases; no file in any tree changed:

- `Strategy A:` -> `The existing .cu sources ...` (describes the approach instead of
  naming the in-house strategy)
- `so followers pass their own arch` -> `so a build targeting another GPU passes its
  own arch`

Rebase: amend `3d098d58e`, then `git rebase --onto <new> 3d098d58e moat-port`.
Force-push with `--force-with-lease=moat-port:0aea7af33` so a concurrent push would
have refused rather than been clobbered. `git diff <old-tip> <new-tip>` is empty.

### Commit map (old -> new, every tree verified IDENTICAL)

```
3d098d58e -> 2b6824fd2  [ROCm] Port rmagine::cuda compute backend to HIP   (reworded)
9088c8e15 -> c99f2fcc3  [ROCm] WIP: rmagine_hiprt Pinhole sensor skeleton
4d2cd269c -> 2c1122c9e  [ROCm] Stage 2: Pinhole HIPRT ray-tracing proof-of-concept
db7f06475 -> a5755c066  [ROCm] Fix Transform3f struct layout to match rmagine::Transform_
4223818c9 -> 9e642a6a6  [ROCm] Stage 2: Complete HIPRT sensor simulators   <- the validated one
73820862e -> ee33fc4aa  [ROCm] Resolve HIP in the installed rmagine::cuda config
74ba64005 -> 27a88badb  [ROCm] Reword two comments in the HIP build path
0aea7af33 -> e7a7b279f  [ROCm] Keep the compatibility header host-includable
```

Nothing at or below the upstream base `6b93e861` was touched.

### Evidence, repointed not re-run

All four platforms had `validated_sha = 4223818c9`, a commit that no longer exists.
Each was repointed to `9e642a6a6` -- the same tree under a new name -- with
`moatlib.py carry-forward ... source-class`. No GPU re-ran and none needed to: a
commit-message reword cannot change compiled output. `carry-forward` is the only
API that writes `validated_sha`, and its `source-class` method (doc/comment-only)
is the closest honest fit for "zero files changed"; the `detail` field records what
actually happened so a later reader is not misled into thinking evidence was
extrapolated across a code change.

This matters most for `windows-gfx1101` and `windows-gfx1201`: those runs happened
on machines this host cannot re-run on demand, so orphaning their references would
have destroyed evidence that is expensive to reproduce.

All four still read `revalidate`, which is correct and unrelated: `head_sha` is
`e7a7b279f` and the round-3 header fix genuinely recompiles.

`python3 utils/jargon.py --port rmagine`: **clean**.

### Left for a person

`c99f2fcc3` is still titled `[ROCm] WIP: rmagine_hiprt Pinhole sensor skeleton`, and
two commits are titled `Stage 2:`. If this branch is published as-is, upstream
maintainers see a WIP commit in the series. Deciding whether the branch should be
reshaped into a clean series is a scope call for a person, and a bigger rewrite
touches more evidence than this one did -- this rewrite deliberately changed the
minimum needed to clear the publication gate.

## Review 2026-08-13 (reviewer, linux-gfx942, delta round) -- CHANGES REQUESTED

Scope: `git diff 9e642a6...e7a7b27` on AMD-Ecosystem/rmagine `moat-port` -- 8 files,
three commits (the round-2 install-config fix and its comment follow-up under their
post-rewrite SHAs, plus the round-3 header-footprint fix) -- together with the history
rewrite that produced those SHAs and the two `cuda-to-rocm` lesson edits riding on
`port/rmagine`. No upstream PR is open (`moatlib.py pr-state` = none), so the working
branch is `moat-port`. No clone existed on this host; the fork was cloned fresh and the
worktree is clean. ROCm here is the conda SDK at
`/opt/conda/envs/py_3.12/lib/python3.12/site-packages/_rocm_sdk_devel`; a CUDA 12.8
header tree exists at `/opt/conda/envs/cuda-12.8`, which let me check the NVIDIA side of
this delta for the first time in this project.

The code in the delta is correct. Both findings are claims that outrun their evidence,
in the same class this project has twice been sent back for. Neither requires a fork
commit to fix.

### 1. "Includable from a plain C++ TU in any order" is false for two of the sixteen installed headers, and the new gate cannot see it

notes.md:1351-1355 states it as established fact -- "Every installed `rmagine_cuda`
header is includable from a plain C++ translation unit in any order, including as the
very first include. That is a gate, not a hope" -- and names `cuda_public_headers` as
the gate. `tests/cuda/public_headers.cpp:1-8` states the same "in any order"
requirement.

The test includes all 16 headers in ONE translation unit in ONE fixed order, with
`cuda_to_hip.h` first (public_headers.cpp:10). It therefore proves the headers coexist
headers-first; it cannot prove any one of them stands alone, because every header after
line 10 is already standing on the includes of the ones above it. Two of the sixteen
fail as the first include, on both backends:

```
# HIP path (-D__HIP_PLATFORM_AMD__=1, ROCm 7.x conda SDK, g++ 13.3.0)
printf '#include <rmagine/math/linalg.cuh>\nint main(){return 0;}\n' > x.cpp
g++ -std=c++17 -fsyntax-only x.cpp -Isrc/rmagine_core/include -Isrc/rmagine_cuda/include \
  -I$ROCM/include -D__HIP_PLATFORM_AMD__=1 -I/usr/include/eigen3
src/rmagine_cuda/include/rmagine/math/linalg.cuh:20:1: error: '__device__' does not name a type

printf '#include <rmagine/util/cuda/CudaHelper.hpp>\nint main(){return 0;}\n' > y.cpp
# same flags
src/rmagine_cuda/include/rmagine/util/cuda/CudaHelper.hpp:51:20: error: 'runtime_error' is not a member of 'std'
```

`linalg.cuh:4` includes only `rmagine/math/types.h` and then declares `__device__`
functions at linalg.cuh:20,23 with nothing that defines `__device__`;
`CudaHelper.hpp:38` includes the runtime shim but throws `std::runtime_error` at
CudaHelper.hpp:51,64 without `<stdexcept>`.

Both are PRE-EXISTING upstream defects, not port regressions -- I ran the identical
check against the upstream base 6b93e86 with the CUDA 12.8 headers and got the same two
errors at the same lines, and `CudaStream.hpp` passes standalone on both. So this is not
a request to re-port anything. It is that the record now tells the next consumer (rmcl's
porter is the named audience) something the gate does not establish and that is false
for two installed headers they may reach for.

Fix, cheapest first: correct notes.md:1351-1355 to what the test proves -- all sixteen
installed headers compile together at the top of a plain C++ TU with nothing above them,
which is the regression that was actually fixed -- and say that per-header independence
is not covered, naming `linalg.cuh` and `CudaHelper.hpp` as the two that fail it
upstream. That needs no fork commit and does not touch `head_sha`. If you would rather
make the claim true, the gate has to be one TU per header (a loop over the installed
headers, or one small object per header) and the two upstream defects need the obvious
two-line fixes (`<stdexcept>` in CudaHelper.hpp, the runtime shim include in
linalg.cuh); weigh that against a fifth revalidation round for four platforms, and note
that `tests/cuda/public_headers.cpp:1-3` should get the same tightening if a fork commit
happens for any other reason.

Same correction is needed in the promoted lesson, which is the part that outlives this
project: `.claude/skills/cuda-to-rocm/references/fault-classes.md:303-306` prescribes
"a test that includes every installed public header at the top of a plain C++ TU, with
nothing above them", and `references/strategy-a-cmake.md:136-138` points at it. As
written, every future porter builds the same single-TU gate and inherits the same blind
spot. Say that one TU per header is what catches a header that only compiles because an
earlier include supplied its dependencies, and that the single-TU form only catches a
shim that poisons the whole set -- with rmagine's own `linalg.cuh` as the example that
slipped through.

### 2. notes.md:1701 justifies the `<cstdio>` workaround with a claim about cuRAND that is not true

notes.md:1699-1702: "it makes even the two RNG headers host-includable, which they are
not upstream either (curand_kernel.h needs nvcc)". Checked directly against CUDA 12.8
headers with the plain host compiler:

```
printf '#include <curand.h>\n#include <curand_kernel.h>\nint main(){return 0;}\n' > c.cpp
g++ -std=c++17 -fsyntax-only c.cpp -I/opt/conda/envs/cuda-12.8/targets/x86_64-linux/include
# exit 0
```

cuRAND's device header is host-includable; hipRAND's is not, for the specific reason the
commit body gives (`rocrand_mtgp32.h:443` calls `printf` with only `<stdlib.h>` and
`<string.h>` above it -- confirmed here). So the `<cstdio>` include in
`curand_to_hiprand.h:35` is not restoring parity with a broken NVIDIA path, it is
repairing a ROCm-only gap, and the port's HIP path is the one that would have been
BEHIND upstream without it. Reword that parenthetical; the fix itself is right and
stays. This matters because a future reader could take "curand_kernel.h needs nvcc" as
licence to skip host-includability on the CUDA side of some other project.

### Verified, no action (fact-checked here, not taken from the round's account)

- The footprint fix is correct and complete. `cuda_to_hip.h` no longer includes
  `<hiprand/hiprand_kernel.h>` and its CUDA branch no longer includes `<curand*.h>`;
  `curand_to_hiprand.h` carries both, and the only two headers that name `curandState`
  in a declaration include it (`random.cuh:5`, `NoiseCuda.hpp:45`) -- exactly the two
  that pulled cuRAND upstream (checked against 6b93e86). All five `.cu` that use
  `curandState`/`curand_*` reach it through `random.cuh` (NoiseCuda.cu:2,
  GaussianNoiseCuda.cu:3, RelGaussianNoiseCuda.cu:3, UniformDustNoiseCuda.cu:3,
  random.cu:1), so removing the include from the force-included shim leaves no
  translation unit short. Every other header is back to its upstream include footprint.
- The reported bug and the fix both reproduce on this host's ROCm: `CudaContext.hpp`
  before `<cstdio>` compiles now under plain g++ with `-D__HIP_PLATFORM_AMD__=1`, as does
  `NoiseCuda.hpp`, and `rocrand_mtgp32.h:443` is the `printf` in question.
- The test does cover all 16 installed headers: `install(DIRECTORY include/rmagine ...)`
  at `src/rmagine_cuda/CMakeLists.txt:225-229` installs the whole tree, which is 16
  files, and public_headers.cpp lists all 16.
- The new target does NOT break the NVIDIA build. `tests/cuda` is added whenever the
  `rmagine-cuda` target exists (tests/CMakeLists.txt), so `public_headers.cpp` is
  compiled by the host C++ compiler on a `USE_HIP=OFF` build too, where it pulls
  `<curand_kernel.h>` through `random.cuh`. I compiled the file as-is against the CUDA
  12.8 headers with g++ 13.3.0 and it succeeds, so the unconditional target is safe and
  correctly placed where both backends run it.
- `rmagine-cuda-config.cmake.in`: the CUDA branch is behaviour-identical to upstream and
  the HIP branch is reached through a variable, not a quoted constant. Re-ran the policy
  check on this host (CMake 3.31.6, `cmake -P`, CMP0012 unset): `if("ON")` is FALSE with
  a CMP0012 warning, `if(<var>)` with the var set to `ON` is TRUE. The load-bearing line
  holds.
- The history rewrite is exactly what it claims. Every old SHA's tree, fetched from the
  fork through the GitHub API, is byte-identical to its replacement:
  3d098d58e/2b6824f, 9088c8e15/c99f2fc, 4d2cd269c/2c1122c, db7f06475/a5755c0,
  4223818c9/9e642a6 (tree 94af630a90625e35afa915e7bce34d116a8a0d0a -- the one four
  platforms' `validated_sha` depends on), 73820862e/ee33fc4, 74ba64005/27a88ba,
  0aea7af33/e7a7b27. The message diff for 2b6824f is the two phrases described and
  nothing else. The carry-forward of the four platform records is sound.
- `python3 utils/jargon.py --port rmagine`: clean. Prior finding 2 closed.
- Prior findings 3, 4 and 5 closed: ee33fc4's body now describes the FindCUDA
  `message(FATAL_ERROR)` aborting inside `find_package(rmagine)`; the "Install as a
  dependency" recipe carries the ROCm root as a second prefix-path entry
  (notes.md:1301-1328); notes.md and plan.md are rmagine's, and the only surviving
  `projects/rmcl/...` strings are the reviewer quotes and the rename history.
- Commit hygiene on the whole branch: `[ROCm]` titles at 47-64 characters, AI assistance
  disclosed, Test Plan in fenced blocks, no `Co-Authored-By`/noreply trailer, no
  non-ASCII in any message or added line, single author, no AMD-internal account
  references. Fork `origin/main` is exactly upstream `uos/rmagine` 6b93e861 (verified via
  API), so the base is a clean mirror.
- No device code in this delta, so the wave-size, OOB, RAII and library-swap classes are
  as validated at 9e642a6; nothing here changes kernel numerics on either backend.

Still open for a person, unchanged by this round: `c99f2fc` is titled
"[ROCm] WIP: rmagine_hiprt Pinhole sensor skeleton" and two commits are titled
"Stage 2:", and the README says nothing about `USE_HIP`. Both belong in the decision
about how the branch and PR body are shaped, not in a porter round.

## Delta round 2026-08-13 (porter, linux-gfx942) -- record corrections only

Answers the review above. Both findings were claims in the record that outran their
evidence; the reviewer stated no fork commit is needed, and none was made. Fork
`moat-port` stays at e7a7b27 and `head_sha` did not move, so the four completed
platforms keep their `validated_sha` and no revalidation is triggered. The clone at
`projects/rmagine/src` is clean at e7a7b27.

### Finding 1: reworded the header-includability claim (notes.md, "Install as a dependency")

Reproduced both halves on this host before rewriting anything. ROCm is the conda SDK at
`/opt/conda/envs/py_3.12/lib/python3.12/site-packages/_rocm_sdk_devel`, CUDA 12.8 headers
at `/opt/conda/envs/cuda-12.8/targets/x86_64-linux/include`, g++ 13.3.0. One TU per
header, all 16 from `tests/cuda/public_headers.cpp`:

```
printf '#include <%s>\nint main(){return 0;}\n' "$h" > t.cpp
g++ -std=c++17 -fsyntax-only t.cpp -Isrc/rmagine_core/include -Isrc/rmagine_cuda/include \
  -I$ROCM/include -D__HIP_PLATFORM_AMD__=1 -I/usr/include/eigen3
```

14 of 16 pass standalone. The two that fail, on the ROCm path and on the CUDA path
(same command with `-I$CUDA_INC` and no `__HIP_PLATFORM_AMD__`):

```
src/rmagine_cuda/include/rmagine/math/linalg.cuh:20:1: error: '__device__' does not name a type
src/rmagine_cuda/include/rmagine/util/cuda/CudaHelper.hpp:51:20: error: 'runtime_error' is not a member of 'std'
```

Both are pre-existing upstream defects: the same two errors at the same lines against a
`git worktree` of upstream 6b93e86 with the CUDA 12.8 headers. `linalg.cuh` is untouched
by the port (`git log 6b93e86..HEAD --` on it is empty) and the port's only change to
`CudaHelper.hpp` is `<cuda_runtime.h>` -> the compat header, which is not what breaks it.

The notes claim now says what the test actually proves -- the 16 compile together at the
top of a plain C++ TU with nothing above them -- and names the two that fail standalone
plus what a dependent needs above them. Source is unchanged, so
`tests/cuda/public_headers.cpp:1-8` still says "in any order"; tightening that comment
(and, if wanted, a per-header TU gate with the two two-line upstream fixes) is registered
in `projects/rmagine/deferred.json` as `rmagine-per-header-include-gate`, because it costs
a fork commit and a fifth revalidation round across four completed platforms.

### Finding 2: corrected the `<cstdio>` rationale (notes.md, delta round 2026-08-12b)

"curand_kernel.h needs nvcc" was false. Confirmed here:

```
printf '#include <curand.h>\n#include <curand_kernel.h>\nint main(){return 0;}\n' > c.cpp
g++ -std=c++17 -fsyntax-only c.cpp -I/opt/conda/envs/cuda-12.8/targets/x86_64-linux/include
# exit 0
```

The parenthetical now states the true reason: cuRAND's device header is host-includable,
hipRAND's is not (`rocrand_mtgp32.h:443` calls `printf` with only `<stdlib.h>` and
`<string.h>` above it), so the `<cstdio>` in `curand_to_hiprand.h:35` repairs a ROCm-only
gap rather than restoring parity with a broken NVIDIA path. The workaround itself stands.

### Lesson corrections (they outlive this project)

`.claude/skills/cuda-to-rocm/references/fault-classes.md` gained "Know what the single-TU
header test proves, and what it does not" directly under the shim-footprint entry that
prescribed the single-TU gate: one TU catches a shim that poisons the whole set, one TU
per header is what catches a header standing on an earlier include, and the per-header
form should be expected to surface pre-existing upstream defects rather than port
regressions. `references/strategy-a-cmake.md` now carries the same distinction where it
points at that entry. rmagine's `linalg.cuh` is the named example that slipped through.

## Review 2026-08-13b (reviewer, linux-gfx942, record-correction round) -- REVIEW PASSED

Scope: the delta since the review above -- MOAT commit `25e477e` on `port/rmagine`
(notes.md, `projects/rmagine/deferred.json`, and the two `cuda-to-rocm` lesson files) plus
confirmation that the fork did not move. No problems found, so this section is evidence
rather than findings.

The fork is unchanged and the four platform records are untouched by this round:
`projects/rmagine/src` is clean at `e7a7b279f`, `git ls-remote origin moat-port` on
AMD-Ecosystem/rmagine returns the same SHA, and `status.json.head_sha` still reads it. The
code at `e7a7b279f` was reviewed in full in the round above; nothing in this delta touches
source, so the fault classes stand as cleared there.

Every corrected claim was re-derived here rather than read from the porter's account, on
this host's conda ROCm SDK, the CUDA 12.8 header tree, and g++ 13.3.0:

- One TU per header over all 16 installed `rmagine_cuda` headers: 14 pass standalone, and
  the same two fail on BOTH include paths -- `math/linalg.cuh:20` ("`__device__` does not
  name a type") and `util/cuda/CudaHelper.hpp:51` ("'runtime_error' is not a member of
  'std'"). Ran the ported tree twice, once with `-D__HIP_PLATFORM_AMD__=1` against the ROCm
  includes and once against the CUDA 12.8 includes; identical result, which is what
  notes.md:1373-1374 now claims.
- Pre-existing, not port regressions: a `git worktree` at upstream `6b93e86` compiled
  against the CUDA 12.8 headers gives the same two errors at the same two lines, and
  `CudaStream.hpp` passes there. `git log 6b93e86..HEAD -- .../linalg.cuh` is empty and the
  only `CudaHelper.hpp` hunk is `<cuda_runtime.h>` -> the compat header, as stated.
- The dependent workarounds notes.md:1376-1379 offers actually work: `<stdexcept>` above
  `CudaHelper.hpp` compiles, and the runtime header above `linalg.cuh` compiles on both
  backends (`cuda_to_hip.h` on the HIP path, `<cuda_runtime.h>` on the CUDA path).
- Finding 2's replacement rationale holds in both directions: `<curand.h>` +
  `<curand_kernel.h>` under plain g++ with the CUDA 12.8 headers exits 0, while
  `<hiprand/hiprand_kernel.h>` alone fails through `rocrand/rocrand_mtgp32.h` and compiles
  once `<cstdio>` precedes it. `rocrand_mtgp32.h:443` is the `printf`, and that header's
  only includes are `rocrand/rocrand.h`, `hip/hip_runtime.h`, `<stdlib.h>`, `<string.h>`.
- The lesson edits match the evidence. `fault-classes.md:308-320` states the single-TU
  limitation and names rmagine's two headers correctly, and `strategy-a-cmake.md:136-140`
  points at both entry titles as they are actually spelled.
- Prior finding 1 is closed: notes.md:1351-1355 now claims only what the gate proves (the
  16 compile together, nothing above them), and the residual "in any order" wording in
  `tests/cuda/public_headers.cpp:1-8` -- which cannot be fixed without a fork commit and a
  fifth revalidation round -- is registered as `rmagine-per-header-include-gate` in
  `projects/rmagine/deferred.json`, unruled, which is where that call belongs. Prior finding
  2 is closed at notes.md:1724-1733.
- Nothing else regressed: `python3 utils/jargon.py --port rmagine` clean,
  `python3 utils/check.py` clean, and the only other commit in the delta (`6e98240`) is
  session telemetry.

Unchanged and still for a person, not a porter: the `WIP` and `Stage 2:` commit titles on
the branch, and the README saying nothing about `USE_HIP`.

## Validation 2026-08-13 (linux-gfx942, MI300X) -- PASS

Fork: AMD-Ecosystem/rmagine `moat-port` HEAD `e7a7b279f` (first arch validated at this
exact SHA; the four already-`completed` platforms carry-forward from `9e642a6a6`, an
ancestor tree-identical only up to the round-2/round-3 delta -- this run covers the
round-2 install-config fix and the round-3 header-footprint fix for real on GPU for the
first time). GPU: 8x AMD Instinct MI300X HF, gfx942 (CDNA3, wave64), ROCm: conda SDK at
`/opt/conda/envs/py_3.12/lib/python3.12/site-packages/_rocm_sdk_devel` (HIP runtime/
clang 23.0.0git). No `/opt/rocm` on this host. `libassimp-dev` was missing and installed
via `sudo apt-get install -y libassimp-dev` (5.3.1+ds-2build1, matches the version other
platforms built against). No HIPRT SDK present on this host
(`/var/lib/jenkins/moat/third_party/HIPRT` does not exist) -- Stage 2 rmagine_hiprt
correctly skipped by CMake, consistent with every non-gfx90a platform to date.

### Configure

```
ROCM=/opt/conda/envs/py_3.12/lib/python3.12/site-packages/_rocm_sdk_devel
utils/timeit.sh rmagine compile -- cmake -S /var/lib/jenkins/moat/projects/rmagine/src \
  -B /var/lib/jenkins/moat/agent_space/rmagine_gfx942_build \
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=$ROCM \
  -DRMAGINE_EMBREE_DISABLE=ON -DRMAGINE_OPTIX_DISABLE=ON \
  -DRMAGINE_VULKAN_DISABLE=ON -DRMAGINE_VULKAN_CUDA_INTEROP_DISABLE=ON \
  -DRMAGINE_OUSTER_DISABLE=ON -DRMAGINE_BUILD_TESTS=ON -DRMAGINE_BUILD_TOOLS=OFF
```

### Build

```
utils/timeit.sh rmagine compile -- cmake --build /var/lib/jenkins/moat/agent_space/rmagine_gfx942_build -j
```

76/76 targets built cleanly (HIP compiler clang++ 23.0.0git). Only pre-existing
nodiscard warnings on hipMemset/hipCtxSetCurrent/hipStreamDestroy/hipDeviceSynchronize
(unchanged from other platforms). `rmagine_tests_cuda_public_headers` (the round-3 gate)
built and linked as target 73/76.

### Test results

```
export HIP_VISIBLE_DEVICES=0
utils/timeit.sh rmagine test -- ctest --test-dir /var/lib/jenkins/moat/agent_space/rmagine_gfx942_build --output-on-failure -R '^cuda_'
# Run 1: 8/8 PASS (2.99 s)
# Run 2 (determinism): 8/8 PASS (3.25 s)
utils/timeit.sh rmagine test -- ctest --test-dir /var/lib/jenkins/moat/agent_space/rmagine_gfx942_build --output-on-failure -R '^core_'
# 12/12 PASS (3.74 s)
```

Tests passing (8, one more than the gfx90a-era 7 -- `cuda_public_headers` is new since
0aea7af): cuda_math, cuda_memory, cuda_memory_slicing, cuda_math_svd,
cuda_math_statistics, cuda_math_reduction, cuda_math_reduction_correctness,
cuda_public_headers.
core_ (12): core_math, core_memory, core_memory_slicing, core_quaternion, core_math_svd,
core_math_statistics, core_math_cov_transform, core_math_gaussians,
core_math_matrix_slicing, core_math_reduction, core_math_cholesky, core_math_lie.

### GPU dispatch confirmed (AMD_LOG_LEVEL=3)

```
AMD_LOG_LEVEL=3 ./bin/rmagine_tests_cuda_math_reduction_correctness
```

ShaderName lines confirm dispatch of `sum_kernel<1024u, Vector3_<float>>` and
`cov_kernel<1024u>` on this device; `rocminfo` names it `amdgcn-amd-amdhsa--gfx942:
sramecc+:xnack-`. Exit: `PASS: rm::sum/mean/cov match CPU reference and are
deterministic`. `strings librmagine-cuda.so.2.4.2 | grep -o gfx[0-9]*` shows `gfx942`
code object present.

### CUDA no-regression gate (runs once per head_sha; first Linux arch at e7a7b279f)

```
CUDA=/opt/conda/envs/cuda-12.8
utils/timeit.sh rmagine cuda-compile -- cmake -S /var/lib/jenkins/moat/projects/rmagine/src \
  -B /var/lib/jenkins/moat/agent_space/rmagine_cuda_gate_build \
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=OFF \
  -DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_C_COMPILER=/usr/bin/gcc-13 \
  -DRMAGINE_EMBREE_DISABLE=ON -DRMAGINE_OPTIX_DISABLE=ON \
  -DRMAGINE_VULKAN_DISABLE=ON -DRMAGINE_VULKAN_CUDA_INTEROP_DISABLE=ON \
  -DRMAGINE_OUSTER_DISABLE=ON -DRMAGINE_BUILD_TESTS=ON -DRMAGINE_BUILD_TOOLS=OFF
utils/timeit.sh rmagine cuda-compile -- cmake --build /var/lib/jenkins/moat/agent_space/rmagine_cuda_gate_build -j
```

77/77 targets built cleanly with NVIDIA nvcc 12.8.93 / g++-13.3.0, arch pinned
`sm_80` via `-DCMAKE_CUDA_ARCHITECTURES=80` (the `set_target_properties(rmagine-cuda
PROPERTIES CUDA_ARCHITECTURES all)` fallback in `rmagine_cuda/CMakeLists.txt:181` is
guarded by `NOT DEFINED CMAKE_CUDA_ARCHITECTURES`, so the pin took and was not
silently overridden). Only pre-existing `cuCtxSetSharedMemConfig`/
`cuCtxGetSharedMemConfig` deprecation warnings (CUDA_VERSION >= 13 style deprecation,
unrelated to the port; the USE_HIP=OFF path is untouched by any port commit). No
NVIDIA-only regression: this is a pure passthrough compile, no code differs on the
CUDA side. `cuda_public_headers` (the round-3 gate) also compiles cleanly against the
CUDA 12.8 headers with `-DUSE_HIP=OFF`, confirming the new test is backend-neutral.

### Integrity gate

`git -C projects/rmagine/src status --porcelain` empty; HEAD = `e7a7b279f8f21cd5fb339282d7fb1b15beda64ae`, matching `status.json.head_sha` exactly.

### Jargon / documentation gate

`python3 utils/jargon.py --port rmagine` -> clean. Documentation: the two open review
items (WIP/Stage-2 commit titles, README silent on USE_HIP) are a person's branch-
shaping/PR-body call, explicitly ruled non-blocking for review and validation by the
reviewer in "Review 2026-08-13" ("Open item ruled on: README does not mention
USE_HIP" / not repeated as a blocking finding in the delta re-review that passed).
Not re-litigated here; consistent with the four already-`completed` platforms.

### Verdict

Stage 1 (rmagine_cuda HIP compute backend) VALIDATED on gfx942/MI300X at `e7a7b279f`:
8/8 cuda_ (2 runs, bit-identical) + 12/12 core_ PASS, no regression. CUDA
no-regression gate PASS (pure passthrough, once per head_sha). Integrity clean.
`wave64` gate satisfied (already satisfied by gfx90a/gfx1100; this is additional
evidence on a different wave64 card, MI300X vs MI250X). Recorded `validated_sha =
e7a7b279f8f21cd5fb339282d7fb1b15beda64ae`, state `completed`.

Verdict: clean. Handing to the validator for the linux-gfx942 GPU run at `e7a7b279f`.
