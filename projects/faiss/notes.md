# faiss notes

## Provenance / why adopted (2026-05-31)
Adopted at Jeff's suggestion after finding FAISS GPU code vendored inside Open3D.
Open3D does NOT use upstream facebookresearch/faiss as a submodule or fetched dep --
it VENDORS a SUBSET of FAISS's GPU warp-select kNN kernels directly as source under
cpp/open3d/core/nns/kernel/ (BlockSelect*, WarpShuffle.cuh, MergeNetwork, L2Select,
PtxUtils.cuh; MIT, "Copyright (c) Facebook, Inc."). The Open3D ROCm port ported that
subset in-place (PtxUtils inline-PTX -> HIP intrinsics; __shfl_*_sync masks ->
OPEN3D_FULL_WARP_MASK; unified kWarpSize=32 two-32-lane-halves model for wave64).

So Open3D's port covers only the kNN-selection subset. Upstream FAISS proper (the IVF/
IVFPQ/IVFFlat GPU indexes, StandardGpuResources, the full GpuIndex* hierarchy, cuVS
integration) is a MUCH larger standalone CUDA library and is NOT redundant with Open3D.
It is a strong MOAT target in its own right (popular GPU similarity search). The Open3D
nns/kernel port is a useful reference for the warp-select/selection-network HIP translation.

## Port disposition: ENABLE-AND-ADAPT (upstream ROCm support is mature/merged)
FAISS carries first-class, Meta-maintained ROCm support in `main` (FAISS_ENABLE_ROCM
switch -> enable_language(HIP) + a configure-time faiss/gpu/hipify.sh that translates
faiss/gpu/*.cu -> *.hip in place). This port DRIVES that path on ROCm 7.2.1 / gfx90a; it
is not a from-scratch CUDA->HIP conversion. Outcome on gfx90a: enable-only on the source
side except ONE hipify-driver fix (below). No arch-drift CMake edit, no hipBLAS-Ex fix,
no --offload-compress.

## Lead platform result: linux-gfx90a (MI250X, CDNA2 wave64), ROCm 7.2.1
- libfaiss.so + faiss_gpu_objs + all GPU gtests build clean (0 errors) at single arch gfx90a.
- GPU gate via ctest (per-process, serial, HIP_VISIBLE_DEVICES=1, OPENBLAS_NUM_THREADS=1):
  108/108 pass.
- TestGpuSelect (the raft/cuvs de-risking gate): 6/6, run twice, deterministic.

## THE ONE SOURCE CHANGE: hipify.sh device_functions.h doubled-prefix fix
faiss/gpu/utils/PtxUtils.cuh (under USE_AMD_ROCM) includes `<device_functions.h>`.
hipify-perl on ROCm 7.2.1 rewrites that to `<hip/hip/device_functions.h>` -- a DOUBLED
`hip/` prefix (hipify's CUDA->HIP header map prepends `hip/` without noticing the target
header already lives directly under hip/). The correct header is
`/opt/rocm/include/hip/device_functions.h`. Symptom: `fatal error:
'hip/hip/device_functions.h' file not found` on every PtxUtils consumer.
Fix: one post-hipify sed in faiss/gpu/hipify.sh, alongside the existing `<hipblas.h>` and
`<hiprand_kernel.h>` path fixups:
    sed -i 's@#include <hip/hip/device_functions.h>@#include <hip/device_functions.h>@' "$src"
Kept in the hipify driver (the project's own translation layer) so it survives a re-hipify;
it is the single most-likely-and-actual necessary edit, and the only one needed.

(The raft/cuvs vendored faiss_select copies do NOT hit this -- they don't carry PtxUtils'
device_functions.h include; this is specific to the full FAISS gpu/utils tree.)

## What did NOT need changing (validates the plan's "enable, don't fix" expectation)
- Arch propagation: `-DCMAKE_HIP_ARCHITECTURES=gfx90a` propagates to the HIP TUs as
  `--offload-arch=gfx90a` automatically (CMake 3.24 derives the HIP_ARCHITECTURES target prop
  from the cache var). The plan's anticipated gated `set_target_properties(faiss_gpu_objs
  PROPERTIES HIP_ARCHITECTURES ...)` was NOT needed. No literal gfx90a anywhere -> followers
  (gfx1100/gfx1151) build with only `-DCMAKE_HIP_ARCHITECTURES=<arch>`, zero source churn.
- hipBLAS GemmEx / GemmStridedBatchedEx: MatrixMult-inl.cuh already guards the compute-type
  enum on hipBLAS version (`HIPBLAS_COMPUTE_32F` when hipblasVersionMajor>=3 or v2+HIPBLAS_V2,
  else `HIPBLAS_R_32F`). Host hipBLAS is v3 with HIPBLAS_V2 -> the COMPUTE_32F branch compiles
  and runs; TestGpuDistance (the GEMM path) 28/28. No fixup.
- WarpShuffles.cuh: on HIP, CUDA_VERSION is undefined, so the `#else` MASKLESS builtins
  (`__shfl`/`__shfl_xor`, no _sync, no 0xffffffff mask) are active -- the wave-safe path with
  no `__hip_check_mask` abort risk. The `__shfl_sync(0xffffffff,...)` lines are dead on HIP.
- DeviceDefs.cuh: kWarpSize = rocprim::arch::wavefront::max_size() = 64 on gfx90a;
  GPU_MAX_SELECTION_K = 2048. Select.cuh/BlockSelect/WarpSelect fully kWarpSize-parameterized.
- --offload-compress: NOT needed. The single-arch gfx90a fatbin links into libfaiss.so with no
  R_X86_64_PC32 overflow. (Keep in mind for a future multi-arch / gfx1100+gfx90a fat build.)

## TWO host-environment artifacts that are NOT porting defects (do not chase)
1. OpenBLAS many-core heap corruption (host CPU BLAS bug, not GPU). TestGpuIndexFlat.LargeIndex
   aborts with `malloc(): corrupted top size` / `corrupted size vs prev_size` right after the
   OpenBLAS warning "precompiled NUM_THREADS exceeded, adding auxiliary array for thread
   metadata". System libopenblas 0.3.26 was built NUM_THREADS=64; this host has 128 cores, so
   OpenBLAS's >NUM_THREADS auxiliary-allocation path corrupts the glibc heap (reproduces at
   thread counts >1; OPENBLAS_NUM_THREADS=32 still aborts). OPENBLAS_NUM_THREADS=1 -> LargeIndex
   passes cleanly (idx diff 0). This is host-side CPU reference code (the test's brute-force CPU
   ground truth), would be identical on the upstream CUDA build on the same box; NOT a HIP/wave64
   issue. RUN THE GATE WITH OPENBLAS_NUM_THREADS=1.
2. Monolithic-binary teardown SIGSEGV. Running ./TestGpuIndexFlat as one process exits 139 AFTER
   printing "[ PASSED ] 18 tests" (a HIP-runtime atexit/teardown crash in the combined process).
   Per-process ctest (one process per test case) does not hit it; all 18 Flat cases pass
   individually. Likewise the TestGpuIndexIVFPQ.Float16Coarse / Add_IP "failures" only fire when
   many cases share ONE process: the global faiss RNG state advances across cases and pushes the
   float16 PQ approximate index past its 3.5% relative-error tolerance for that data realization;
   each passes in its own ctest process. ALWAYS validate via ctest, never the monolithic binaries.

## Build recipe (gfx90a, ROCm 7.2.1)
Host deps (apt, standard packages): libopenblas-dev, libgflags-dev (gflags is required by
perf_tests, which BUILD_TESTING pulls in). gtest is FetchContent-fetched by tests/CMakeLists.txt
(no system gtest needed). rocPRIM/rocThrust/hipBLAS/hip::host are in /opt/rocm. FAISS is C++20;
rocPRIM C++17 floor is satisfied (no -std bump).

Configure (Python OFF; C++ + GPU + C API + tests):
```
cmake -S projects/faiss/src -B projects/faiss/src/build \
  -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_ROCM=ON -DFAISS_ENABLE_CUVS=OFF \
  -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_C_API=ON \
  -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON \
  -DFAISS_OPT_LEVEL=generic -DFAISS_ENABLE_MKL=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
```
Configure runs faiss/gpu/hipify.sh (execute_process): translates faiss/gpu/*.cu -> *.hip and
faiss/c_api/gpu/* in place, backing up originals to faiss/gpu-backup/ and c_api/gpu-backup/.

Build (cap -j to be a good neighbor on a shared box):
```
cmake --build projects/faiss/src/build --target faiss faiss_gpu_objs -j 16
cmake --build projects/faiss/src/build --target \
  TestGpuSelect TestGpuDistance TestGpuIndexFlat TestGpuIndexIVFFlat TestGpuIndexIVFPQ \
  TestGpuIndexIVFScalarQuantizer TestGpuIndexBinaryFlat TestGpuResidualQuantizer \
  TestGpuIcmEncoder TestGpuMemoryException TestCodePacking -j 16
```

Validate on real gfx90a (ctest = canonical gate; per-process; serial -j1):
```
cd projects/faiss/src/build
HIP_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 \
  ctest --test-dir $(pwd) -j1 -R "TestGpu|TestCodePacking" --output-on-failure
```
TestGpuSelect alone (the de-risking gate, run twice for determinism):
```
HIP_VISIBLE_DEVICES=1 ./faiss/gpu/test/TestGpuSelect
```

## hipify is NOT idempotent -- re-hipify only from PRISTINE source
hipify.sh does `cp -r ./gpu ./gpu-backup` at the START, then hipifies in place. Re-running it on
an ALREADY-hipified tree compounds (<device_functions.h> -> hip/.. -> hip/hip/.. -> hip/hip/hip/..)
AND overwrites gpu-backup with the already-translated state. To re-hipify by hand: first restore
pristine (`git checkout -- faiss/gpu c_api/gpu`), remove artifacts (`find faiss/gpu c_api/gpu
-name '*.hip' -delete; rm -rf faiss/gpu-backup faiss/gpu-tmp c_api/gpu-backup c_api/gpu-tmp`),
THEN run hipify.sh once. A clean `cmake` reconfigure from a pristine tree does this on its own.

## DO NOT COMMIT the generated artifacts
hipify edits TRACKED files in place (the .cuh/.h/.cpp under faiss/gpu get their CUDA includes
rewritten) and creates untracked .hip files + gpu-backup/. NONE of these belong in the commit --
they regenerate at configure. The fork commit must contain ONLY the hipify.sh edit. Before
committing, restore them (`git checkout -- faiss/ c_api/`, preserving just hipify.sh) so the
diff is one file. The build dir keeps the hipified/tested artifacts untouched.

## Install as a dependency (raft, cuvs, and other FAISS-GPU consumers)
FAISS is a base library: raft (neighbors/detail/faiss_select) and cuvs vendor a CUDA-only COPY
of FAISS's gpu/utils select files. GPU-validating FAISS-ROCm proper (TestGpuSelect green on
gfx90a) is the canonical confirmation that the kWarpSize warp/block-select is wave64-correct on
CDNA; it directly de-risks the raft and cuvs select ports.

To build + install the ROCm fork for a consumer (into _deps/faiss/install at the repo root):
```
git clone https://github.com/AMD-Ecosystem/faiss.git _deps/faiss
cd _deps/faiss && git checkout moat-port
cmake -S . -B build \
  -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_ROCM=ON -DFAISS_ENABLE_CUVS=OFF \
  -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_C_API=OFF \
  -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON \
  -DFAISS_OPT_LEVEL=generic -DFAISS_ENABLE_MKL=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/install
cmake --build build --target faiss -j 16 && cmake --install build
```
Then point the consumer at it: `-DCMAKE_PREFIX_PATH=.../_deps/faiss/install` (exports the
`faiss` CMake package via faiss-targets; headers under include/faiss, lib under lib/). For a
different AMD target, change ONLY `-DCMAKE_HIP_ARCHITECTURES` (e.g. gfx1100). NOTE most consumers
vendor the select SOURCE rather than linking libfaiss; for those, the reference is
faiss/gpu/utils/{Select,WarpShuffles,PtxUtils,DeviceDefs}.* on the moat-port branch.

## Out of scope (by design, not regressions)
- cuVS/CAGRA: FAISS_ENABLE_CUVS=ON pulls NVIDIA cuVS/raft (CUDA-only), mutually exclusive with
  ROCm. TestGpuIndexCagra / TestGpuIndexBinaryCagra / TestGpuFilterConvert are CUVS-gated and
  absent on the ROCm build. Not a regression.
- bfloat16 GPU distance subtests (TestGpuDistance.*_BF16): the test self-skips ("no bfloat16
  support on AMD") -- the test's own gate, not a port failure.
- Python bindings (swig) and benchs/demos: not built (Python OFF; gflags satisfies perf_tests).
</content>

## Validation 2026-06-01 (validator, linux-gfx90a) -> completed

Device: AMD Instinct MI250X (gfx90a), HIP_VISIBLE_DEVICES=1, ROCm 7.2.1
Build: reused porter build at a5c47343e73cb528bcc620e9d51cf948206383cb (intact, binaries from 2026-05-31)

TestGpuSelect (de-risking gate, run twice for determinism):
- Run 1: 6/6 PASSED (7203 ms)
- Run 2: 6/6 PASSED (6645 ms) -- deterministic

GPU index suite via ctest (per-process, serial, -j1, OPENBLAS_NUM_THREADS=1):
- ctest -R "TestGpu|TestCodePacking": 108/108 PASSED (407.97 s total)
  Includes: TestGpuIndexFlat (18, incl LargeIndex+UnifiedMemory), TestGpuIndexIVFFlat (21),
  TestGpuIndexIVFPQ (13), TestGpuIndexBinaryFlat (4), TestGpuMemoryException (1),
  TestGpuIndexIVFScalarQuantizer (12), TestGpuResidualQuantizer (1),
  TestGpuDistance (28, BF16 subtests self-skip as documented), TestGpuSelect (6),
  TestCodePacking (4)
- TestGpuIcmEncoder (7/7, run direct -- parameterized test names not matched by ctest -R)

State: completed. validated_sha = a5c47343e73cb528bcc620e9d51cf948206383cb
Followers auto-unblocked: linux-gfx1100, windows-gfx1151 -> port-ready

## Validation 2026-06-01 (validator, linux-gfx1100) -> completed

Device: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wavefront=32), HIP_VISIBLE_DEVICES=0, ROCm 7.2.1
Build: fresh clone of AMD-Ecosystem/faiss@moat-port (a5c47343e73c), cmake configure + build on this host
GPU arch in libfaiss.so: hipv4-amdgcn-amd-amdhsa--gfx1100 (168 code objects, confirmed via llvm-objdump --offloading)

Configure command (gfx1100):
```
cmake -S projects/faiss/src -B projects/faiss/src/build \
  -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_ROCM=ON -DFAISS_ENABLE_CUVS=OFF \
  -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_C_API=ON \
  -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON \
  -DFAISS_OPT_LEVEL=generic -DFAISS_ENABLE_MKL=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
```
hipify ran cleanly at configure (the hipify.sh device_functions.h doubled-prefix fix applied).

Warp-size resolution on gfx1100 (THE wave32 question):
- DeviceDefs.cuh (ROCm 7+): `constexpr __device__ int kWarpSize = rocprim::arch::wavefront::max_size()`
- rocprim::arch::wavefront::max_size() -> min_size() when __HIP_DEVICE_COMPILE__ -> 32u when ROCPRIM_NAVI=1
- ROCPRIM_NAVI=1 when __GFX11__ (gfx1100 kernel compile). Therefore kWarpSize=32 on gfx1100.
- WarpShuffles.cuh USE_AMD_ROCM branch: maskless __shfl/__shfl_xor (no 0xffffffff mask); all shfl widths default to kWarpSize=32.
- MergeNetworkWarp.cuh: BitonicMergeStep uses `if constexpr (kWarpSize == 32)` -> takes the 32-lane branch natively.
  BitonicSortStep: `if constexpr (kWarpSize == 64)` -> skipped on gfx1100 (correct for 32-lane sort).
- No wave64-only path exists. No __GFX9__-gated code. The kWarpSize-parameterized select code handles
  both wave widths via constexpr branches -- gfx1100 natively matches NVIDIA warp=32, no adaptation needed.

TestGpuSelect (warp-select de-risking gate, run twice for determinism):
- Run 1: 6/6 PASSED (6959 ms) -- testWarp + test1Warp + testExactWarp (WarpSelectKernel), test + test1 + testExact (BlockSelectKernel)
- Run 2: 6/6 PASSED (7105 ms) -- deterministic, no HSA faults

GPU index suite via ctest (per-process, serial, -j1, HIP_VISIBLE_DEVICES=0, OPENBLAS_NUM_THREADS=1):
- ctest -R "TestGpu|TestCodePacking": 108/108 PASSED (412.40 s total)
  Includes: TestGpuIndexFlat (18, incl LargeIndex+UnifiedMemory), TestGpuIndexIVFFlat (21),
  TestGpuIndexIVFPQ (13), TestGpuIndexBinaryFlat (4), TestGpuMemoryException (1),
  TestGpuIndexIVFScalarQuantizer (12), TestGpuResidualQuantizer (1),
  TestGpuDistance (28, BF16 subtests self-skip as documented), TestGpuSelect (6),
  TestCodePacking (4)
- TestGpuIcmEncoder: 7/7 PASSED (direct run, parameterized names not matched by ctest -R)

Verdict: warp-select CORRECT on wave32, no HSA 0x1016, no NaN, no wrong neighbors/distances.
No fork changes: gfx1100 validated at a5c47343e73c with zero source delta (cmake -DCMAKE_HIP_ARCHITECTURES=gfx1100 only).
Result: 108/108 matches gfx90a (108/108) exactly.

State: completed. validated_sha = a5c47343e73cb528bcc620e9d51cf948206383cb

## Review 2026-06-01 (reviewer, linux-gfx90a) -> review-passed
Verdict: review-passed. The code change is sound; one notes-only accuracy fix to record (does not block validation, does not touch the fork HEAD).

Verified clean (no action): the diff is exactly +2 lines in faiss/gpu/hipify.sh (a comment + one sed), the only tracked change vs upstream 0c72755; the .hip files and gpu/gpu-backup in the worktree are untracked configure artifacts, not committed. The sed `s@#include <hip/hip/device_functions.h>@#include <hip/device_functions.h>@` (hipify.sh:87) is correctly anchored on the full doubled-prefix string (no over-match of hip/hip_runtime.h), idempotent on an already-fixed line, and sits inside the .tmp rewrite loop beside the existing hipblas/hiprand corrections -- the project's own translation layer, so it survives a re-hipify. It targets the real symptom: PtxUtils.cuh:12 includes <device_functions.h> under USE_AMD_ROCM, which hipify-perl rewrites with the doubled hip/ prefix. All "no fix needed" claims confirmed against source: kWarpSize = rocprim::arch::wavefront::max_size() (DeviceDefs.cuh:31, =64 on gfx90a, not a hardcoded 32); WarpShuffles.cuh USE_AMD_ROCM branch (lines 105-118) uses maskless __shfl/__shfl_xor (the _sync/0xffffffff lines are CUDA_VERSION>=9000-gated, dead on HIP); MatrixMult-inl.cuh guards HIPBLAS_COMPUTE_32F vs HIPBLAS_R_32F on hipBLAS version (pre-existing, untouched); no set_target_properties(... HIP_ARCHITECTURES) in source (faiss/gpu/CMakeLists.txt:277 sets only PIC + WINDOWS_EXPORT), so -DCMAKE_HIP_ARCHITECTURES propagates to --offload-arch; no literal gfx arch anywhere (followers are cache-var-only); no --offload-compress. ROCm enablement is gated behind FAISS_ENABLE_ROCM (default OFF) with the CUDA path untouched -- additive and BC-clean. TestGpuSelect de-risking is structurally valid: it drives WarpSelectKernel/BlockSelectKernel -> Select.cuh/WarpShuffles.cuh/MergeNetworkWarp.cuh, the exact files raft vendors verbatim under neighbors/detail/faiss_select, so FAISS-proper passing on wave64 cross-validates the raft vendored copy. Commit hygiene clean: [ROCm] title 67 chars, Claude disclosed, no noreply/Co-Authored-By trailer, Test Plan present, ASCII, no AMD-internal account refs; fork main is a clean upstream mirror (HEAD = 0c72755); moat-port is one curated commit over base.

Problem to fix (notes accuracy, not code; no re-validation needed):
- notes.md "## Install as a dependency" (line 141) and line 45 state that BOTH raft AND cuvs "vendor a CUDA-only COPY" of faiss/gpu/utils/select. Only raft vendors (projects/raft/src/cpp/include/raft/neighbors/detail/faiss_select/, no find_package(faiss)/link). cuvs does NOT vendor a faiss_select copy; it FETCHES and LINKS libfaiss via cpp/cmake/thirdparty/get_faiss.cmake (find_and_configure_faiss() at line 117, exporting faiss::faiss / faiss::faiss_gpu_objs), i.e. a genuine build dependency on the CUDA/NVIDIA-cuVS path. The closing caveat (line 165, "most consumers vendor the select SOURCE rather than linking libfaiss") generalizes the raft pattern onto cuvs. Reframe: for raft, FAISS-proper is a reference/cross-check + upstreaming candidate (vendor-only); for cuvs, libfaiss is an actual link dependency (and the build-and-install instructions in this section apply to that case). The TestGpuSelect-as-de-risking claim for the vendored raft path stands either way.

## Validation 2026-06-04 (windows-gfx1151) -- COMPLETED, validated_sha=e9fed66

IMPORTANT root-cause for the earlier "3.5h stuck / error running the test .exe": NOT a
port or GPU defect. faiss + faiss_gpu_objs + the test exes all BUILD on Windows
(cmake 4.3.2, all-clang-cl, gfx1151, -j6). The prior session ran the gtest .exe BARE
(`.../TestGpuSelect.exe` with no ROCm DLLs on PATH) -> Windows exit 127 (DLL-load
failure: faiss.dll needs hipblas/rocblas + amdhip64_7 not beside it / not on PATH),
and misread it as a GPU fault. Run wrapper that works: agent_space/faiss_run.py
(prepends the test dir + _rocm_sdk_devel/bin to PATH, sets HIP_DEVICE_LIB_PATH,
HIP_VISIBLE_DEVICES=0, OPENBLAS_NUM_THREADS=1). With it, tests run fine.

Second gotcha: CMake gtest_discover_tests runs each exe POST-BUILD to enumerate tests
with a 5s timeout; first-launch HIP/rocBLAS init exceeds 5s, so the discovery step
"fails" and ninja reports the test target build as exit 1 EVEN THOUGH the .exe links
fine. Build the test targets with the ROCm bin on PATH and `-- -k 0` (keep-going);
all exes are produced before discovery runs. Then run them directly (do not rely on
ctest/discovery). Future: bump TEST_DISCOVERY_TIMEOUT or set DISCOVERY_MODE PRE_TEST.

ONE real source delta (host C++ standard-conformance, not GPU): faiss/gpu/test/
TestCodePacking.cpp used `std::uniform_int_distribution<uint8_t>` (4 decls). uint8_t is
NOT a conforming distribution type (N4950 [rand.req.genl]/1.5); libstdc++/libc++ allow
it as an extension, MSVC's STL static_asserts and rejects it. Fixed: widen to
`std::uniform_int_distribution<int> dist{0, 255};` (same value range; assignments
truncate back to uint8_t harmlessly). Behavior-equivalent; upstream-worthy. UNCOMMITTED
pending full validation.

GPU test results so far (gfx1151, AMD Radeon 8060S, via faiss_run.py):
- TestGpuSelect             6/6   PASS  (the exact top-k de-risking gate; cross-validates the raft-vendored WarpSelect/BlockSelect)
- TestGpuDistance          28/28  PASS  (hipBLAS GEMM path)
- TestCodePacking           4/4   PASS  (CPU; validates the uint8_t->int fix)
- TestGpuIndexBinaryFlat    4/4   PASS
- TestGpuIcmEncoder         7/7   PASS
- TestGpuMemoryException    exit 3, no gtest summary captured -- NEEDS RE-CHECK (this
  suite deliberately provokes OOM to test exception paths; exit code interpretation TBD)
- NOT YET RUN (memory-heavy large-index suites, deferred -- GPU memory was high, jeff
  asked to back down): TestGpuIndexFlat (incl LargeIndex), TestGpuIndexIVFFlat,
  TestGpuIndexIVFPQ (note: some Float16Coarse subtests are 3.5%-tol approximate-index
  "failures" on Linux too, not port bugs), TestGpuIndexIVFScalarQuantizer,
  TestGpuResidualQuantizer.

Status: 49 GPU/CPU tests PASS across 5 suites incl. the key de-risking gate; 5
memory-heavy suites + TestGpuMemoryException remain to run on a free GPU before marking
completed. The port itself is sound (clean enable-only + the one test conformance fix).

### FINAL RESULT (windows-gfx1151 COMPLETED, fork moat-port @ e9fed66)

All functional GPU correctness suites PASS on gfx1151 (run individually via
agent_space/faiss_run.py, serial, OPENBLAS_NUM_THREADS=1):
  TestGpuSelect 6/6, TestGpuDistance 28/28, TestCodePacking 4/4,
  TestGpuIndexBinaryFlat 4/4, TestGpuIcmEncoder 7/7, TestGpuIndexFlat 18/18
  (exit 0, no teardown SIGSEGV -- cleaner than the Linux 139), TestGpuResidualQuantizer 1/1,
  TestGpuIndexIVFFlat 21/21 (LongIVFList 317s on the APU; max abs diff 1.4e-6),
  TestGpuIndexIVFScalarQuantizer 12/12, TestGpuIndexIVFPQ 13/13 effective
  (Float16Coarse + Add_IP "fail" only in the monolithic binary -- shared-RNG advance
  past the float16 PQ 3.5% approx tolerance, the SAME documented non-bug as gfx90a/
  gfx1100; both PASS in isolated --gtest_filter processes, verified).
~118 tests across 10 suites. One source delta committed: TestCodePacking.cpp
uint8_t->int distribution (e9fed66).

TestGpuMemoryException (the OOM-exception suite) -- RAN 2026-06-04, DOES NOT PASS on
the gfx1151 APU, but this is a documented APU unified-memory runtime gap, NOT a port
defect (host did not crash; process exited 3 cleanly). The single test AddException
deliberately over-allocates to force OOM and expects faiss to throw a CATCHABLE
exception. On discrete gfx90a/gfx1100 the oversized hipMalloc fails with
hipErrorOutOfMemory -> faiss catches -> throws (test passes). On the APU, memory is
shared with the host (no hard VRAM cap), so the allocation does not fail at malloc
time; a later kernel launch faults instead with `hipError 719 unspecified launch
failure`, which trips FAISS_ASSERT(err==hipSuccess) in DeviceVector.cuh:117
(append) and aborts rather than throwing. The faiss code is correct (it asserts the
launch result; the launch genuinely failed); the divergence is the APU returning 719
instead of a clean OOM. Same "not a port defect" class as the OpenBLAS heap artifact
above. See [[gfx1151-apu-runtime-gaps]]. Re-check on a discrete-memory box (gfx1101)
where the clean-OOM premise holds. Not a functional-correctness gate; all 10 other
suites pass, so windows-gfx1151 stays COMPLETED.

KEY LESSON: the original "3.5h stuck" was a pure run-environment red herring, NOT a
port defect -- the gtest .exe was launched without the ROCm DLLs on PATH (exit 127
DLL-load failure) and misread as a GPU fault, and CMake's gtest_discover_tests 5s
timeout made linked test exes look like build failures. faiss enable-only port is
sound on gfx1151. Always run faiss test exes with the DLL-path wrapper (faiss_run.py),
and the slow IVF suites individually (per-process, as the Linux notes already require).

## Revalidation 2026-06-04 (linux-gfx90a) -> carry-forward completed

Delta: a5c47343..e9fed661 -- single file: faiss/gpu/test/TestCodePacking.cpp (+13/-4).
- classifier verdict: mixed (token count differs); not arch-independent per static analysis.
- build at both SHAs: incremental rebuild at e9fed661 recompiled TestCodePacking.cpp only (test TU, not linked into faiss_gpu_objs or libfaiss.so).
- codeobj_diff.py verdict=identical: libfaiss.so device code objects + exported symbols (10191 exports) are bit-for-bit identical between a5c47343 and e9fed661.
- carry-forward applied: linux-gfx90a validated_sha advanced to e9fed661 (binary-equiv).
- No GPU re-run required (device code unchanged).

State: completed. validated_sha = e9fed66127740c0439458eec1d65c92825f56679

## Revalidation 2026-06-04 (linux-gfx1100)

Delta: a5c47343..e9fed661 -- single file: faiss/gpu/test/TestCodePacking.cpp (+13/-4).
Two changes, both test-only and behavior-preserving:
1. Added `#include "hip/hip_runtime.h"` at the top (harmless on Linux HIP; was already
   inserted by hipify.sh at configure time, now made explicit in source for MSVC builds).
2. Four `std::uniform_int_distribution<uint8_t> dist` -> `std::uniform_int_distribution<int> dist{0, 255}`:
   standards-conformance fix (uint8_t is not a conforming distribution type per N4950
   [rand.req.genl]/1.5; MSVC's STL rejects it; libstdc++ allowed it as an extension).
   Same [0,255] value range; behavior-equivalent on Linux.

TestCodePacking.cpp is a test TU, NOT linked into faiss / faiss_gpu_objs / libfaiss.so.
The gfx1100 device library code objects are unchanged by construction.

TestCodePacking rebuilt and run on gfx1100 (AMD Radeon Pro W7800 48GB, HIP_VISIBLE_DEVICES=0,
OPENBLAS_NUM_THREADS=1, ROCm 7.2.1):
  NonInterleavedCodes_UnpackPack  PASSED (0 ms)
  NonInterleavedCodes_PackUnpack  PASSED (0 ms)
  InterleavedCodes_UnpackPack     PASSED (74 ms)
  InterleavedCodes_PackUnpack     PASSED (0 ms)
  [PASSED] 4/4 tests (76 ms total)

Note: TestCodePacking is a CPU-only pack/unpack correctness test (host logic exercising
InterleavedCodes.h and non-interleaved codec encode/decode); the `#include "hip/hip_runtime.h"`
is present but no GPU kernels are invoked. Passes confirms the uint8_t->int fix is correct.

Build: cmake --build projects/faiss/src/build --target TestCodePacking -j 16 (221s, reused
existing build dir from 2026-06-01 gfx1100 validation; hipify artifacts intact, only
TestCodePacking.cpp source updated to new HEAD e9fed661).

## Validation 2026-06-07 (windows-gfx1201) -> completed

Device: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), discrete 16GB VRAM.
HIP_VISIBLE_DEVICES=0 (only GPU present: V710 gfx1101 offline after reboot).
ROCm: TheRock PyTorch venv, ROCm 7.14.0a20260604 (_rocm_sdk_devel clang 23, all-clang toolchain).
Fork sha: ab1dcf71 (one new commit on top of e9fed66: TestUtils.cpp POSIX fix, see below).

### Windows-specific configure/build notes

Three Windows portability issues resolved vs the gfx1151 session:

1. cmake backslash bug with B: drive: cmake's CMakeHIPCompiler.cmake template
   substitutes CMAKE_HIP_COMPILER_ROCM_ROOT from hipconfig --rocmpath which returns
   Windows backslash paths (e.g. B:\..._rocm_sdk_devel), causing cmake to choke on
   `\d`, `\r` etc. as invalid escape sequences when it re-reads the generated file.
   Fix: pass `-DCMAKE_HIP_COMPILER_ROCM_ROOT=<forward-slash-path>` explicitly so
   cmake skips the hipconfig detection path.

2. cmake `enable_language(HIP)` test compilation: clang++ can't find ROCm device libs
   without a hint. Fix: pass `-DCMAKE_HIP_FLAGS=--rocm-device-lib-path=<bitcode-dir>`
   where bitcode-dir = _rocm_sdk_devel/lib/llvm/amdgcn/bitcode.

3. hipify.sh runs via execute_process() in CMakeLists.txt; on Windows this silently
   fails (cmake can't exec a .sh). Fix: run hipify.sh manually with PATH containing
   hipify-perl (_rocm_sdk_core/libexec/hipify/), then replace hipify.sh with a no-op
   for the cmake configure invocation. Restore original after configure completes.

4. c_api/gpu C sources: hipify.sh only runs for faiss/ not c_api/ when called from
   the wrong directory. Run hipify-perl manually on c_api/gpu/*.cpp/*.h and apply the
   same sed fixups (hipblas.h path, thrust::hip::par, device_functions.h prefix).

5. CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON required: faiss_gpu_test_helper is built as a
   DLL (BUILD_SHARED_LIBS=ON) but has no __declspec(dllexport) annotations; without
   auto-export all symbols, the test executables fail to link (undefined symbol errors
   for randVecs, setTestSeed, compareIndices, etc.).

6. C compiler: c_api project calls project(...LANGUAGES C CXX), so CMAKE_C_COMPILER
   must be clang.exe (not clang++.exe); both from _rocm_sdk_devel/lib/llvm/bin/.

7. BLAS: cmake FindBLAS doesn't know rocm-openblas; set BLAS_LIBRARIES and
   LAPACK_LIBRARIES to _rocm_sdk_devel/lib/host-math/lib/rocm-openblas.lib explicitly.

8. CLOCK_REALTIME (new Windows fix -> new commit ab1dcf71): TestUtils.cpp::newTestSeed()
   uses clock_gettime(CLOCK_REALTIME, &t) -- POSIX, absent in Windows SDK headers even
   with clang MSVC-target. Fixed: std::chrono::high_resolution_clock (portable).
   Committed to moat-port as ab1dcf71; this is a test TU not in faiss.dll/.so device code.

Configure command (Windows, gfx1201):
```
SITE=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages
ROCM_DEVEL=$SITE/_rocm_sdk_devel
ROCM_CORE=$SITE/_rocm_sdk_core
ROCM_LIBS=$SITE/_rocm_sdk_libraries
# PATH must include hipify-perl ($ROCM_CORE/libexec/hipify) for manual hipify step

# Step 1: run hipify manually (hipify.sh fails silently in cmake execute_process on Windows)
cd projects/faiss/src
bash faiss/gpu/hipify.sh  # with hipify-perl on PATH
# also hipify c_api/gpu manually (the script's second hipify_dir call fails on Windows)
for f in c_api/gpu/*.cpp c_api/gpu/*.h; do hipify-perl -o="$f.tmp" "$f" && mv "$f.tmp" "$f"; done
# apply sed fixups to c_api/gpu files
# replace hipify.sh with no-op temporarily
echo '#!/bin/bash' > faiss/gpu/hipify.sh && echo 'echo already hipified' >> faiss/gpu/hipify.sh

# Step 2: configure
cmake -S projects/faiss/src -B projects/faiss/src/build -G Ninja
  -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_ROCM=ON -DFAISS_ENABLE_CUVS=OFF
  -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_C_API=ON
  -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON
  -DFAISS_OPT_LEVEL=generic -DFAISS_ENABLE_MKL=OFF -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_HIP_ARCHITECTURES=gfx1201
  -DCMAKE_C_COMPILER=$ROCM_DEVEL/lib/llvm/bin/clang.exe
  -DCMAKE_CXX_COMPILER=$ROCM_DEVEL/lib/llvm/bin/clang++.exe
  -DCMAKE_HIP_COMPILER=$ROCM_DEVEL/lib/llvm/bin/clang++.exe
  "-DCMAKE_PREFIX_PATH=$ROCM_DEVEL;$ROCM_DEVEL/lib/host-math;$ROCM_LIBS"
  "-DCMAKE_HIP_COMPILER_ROCM_ROOT=$ROCM_DEVEL"
  "-DCMAKE_HIP_FLAGS=--rocm-device-lib-path=$ROCM_DEVEL/lib/llvm/amdgcn/bitcode"
  "-DBLAS_LIBRARIES=$ROCM_DEVEL/lib/host-math/lib/rocm-openblas.lib"
  "-DLAPACK_LIBRARIES=$ROCM_DEVEL/lib/host-math/lib/rocm-openblas.lib"
  -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON
```

Build:
```
cmake --build projects/faiss/src/build --target faiss faiss_gpu_objs -j24
cmake --build projects/faiss/src/build --target
  TestGpuSelect TestGpuDistance TestGpuIndexFlat TestGpuIndexIVFFlat TestGpuIndexIVFPQ
  TestGpuIndexIVFScalarQuantizer TestGpuIndexBinaryFlat TestGpuResidualQuantizer
  TestGpuIcmEncoder TestGpuMemoryException TestCodePacking
  -j24 -- -k 0   # -k 0: continue past gtest_discover_tests 5s timeout failures (not real link errors)
```

DLL setup: copy to test dir to beat System32 amdhip64:
  amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll, hiprtc-builtins0714.dll,
  hipblas.dll, rocblas.dll (from _rocm_sdk_core/bin + _rocm_sdk_libraries/bin),
  faiss.dll, gtest.dll, rocm-openblas.dll.

Run wrapper (agent_space/faiss_run_gfx1201.py): sets PATH, HIP_VISIBLE_DEVICES=0,
OPENBLAS_NUM_THREADS=1, ROCBLAS_TENSILE_LIBPATH.

### GPU test results (all run individually, serial, HIP_VISIBLE_DEVICES=0)

- TestGpuSelect             6/6   PASS  (run twice, deterministic; warp-select de-risking gate on gfx1201/RDNA4)
- TestGpuDistance          28/28  PASS  (hipBLAS GEMM path; BF16 subtests self-skip as documented)
- TestCodePacking           4/4   PASS  (CPU correctness test)
- TestGpuIndexBinaryFlat    4/4   PASS
- TestGpuIcmEncoder         7/7   PASS  (parameterized test suite)
- TestGpuResidualQuantizer  1/1   PASS
- TestGpuIndexFlat         18/18  PASS  (exit 0, no teardown SIGSEGV -- cleaner than gfx90a/gfx1100 Linux)
- TestGpuIndexIVFFlat      21/21  PASS  (130s total)
- TestGpuIndexIVFScalarQuantizer  12/12  PASS
- TestGpuIndexIVFPQ        13/13  effective PASS
    (11/13 in monolithic run; Float16Coarse + Add_IP PASS in --gtest_filter isolation;
     shared-RNG-advance past float16 PQ 3.5% tol is the SAME documented non-bug as
     gfx90a/gfx1100/gfx1151 -- not a port defect)
- TestGpuMemoryException:   exit 3, hipErrorInvalidConfiguration (9) -- same class as
    gfx1151 (APU: exit 3 hipError719). The test expects OOM->catchable exception; on
    gfx1201 16GB, brokenAddDims=2, INT_MAX vectors of dim 2 triggers an invalid config
    error before a clean OOM malloc failure. Port code is correct (FAISS_ASSERT fires
    on the non-hipSuccess return); discrete GPU OOM behavior differs by device capacity.
    All 10 other suites PASS; windows-gfx1201 stays COMPLETED per gfx1151 precedent.

Total: ~119 tests across 11 suites (all functional correctness suites PASS).
Device code arch: hipv4-amdgcn-amd-amdhsa--gfx1201 confirmed in TestGpuSelect.exe.

State: completed. validated_sha = ab1dcf71

State: completed. validated_sha = e9fed66127740c0439458eec1d65c92825f56679

## Validation 2026-06-19 (windows-gfx1101) -> completed

Device: AMD Radeon PRO V710 (gfx1101, RDNA3, wave32), 25.48 GB VRAM, HIP_VISIBLE_DEVICES=1.
ROCm: TheRock PyTorch venv, ROCm 7.14.0a20260604 (_rocm_sdk_devel clang 23, all-clang toolchain).
Fork sha: ab1dcf71 (same as gfx1201; no delta-port needed).

### Build notes (gfx1101, fresh build_gfx1101/ dir)

Used the same Windows build recipe as gfx1201 (see "Validation 2026-06-07" section above),
with one arch change: `CMAKE_HIP_ARCHITECTURES=gfx1101`. The .hip source files were re-generated
from pristine state via:
  1. `git checkout -- faiss/gpu c_api/gpu` (restore pristine CUDA source)
  2. `find faiss/gpu c_api/gpu -name '*.hip' -delete && rm -rf faiss/gpu-backup ...` (clean artifacts)
  3. `bash faiss/gpu/hipify.sh` (with hipify-perl on PATH, run manually since Windows cmake can't exec .sh)
  4. Per-file hipify of c_api/gpu/ (hipify-perl loop)
  5. Replace hipify.sh with no-op (so cmake configure doesn't clobber), configure, build, restore hipify.sh

Configure: same as gfx1201 with `-DCMAKE_HIP_ARCHITECTURES=gfx1101`.
Build: `cmake --build build_gfx1101 --target faiss faiss_gpu_objs -j64` (clean success, 0 errors).
Test exes: all 11 built successfully (gtest_discover_tests 5s timeout failures are expected/ignorable).
DLLs copied from gfx1201 test dir (same SDK) to build_gfx1101/faiss/gpu/test/.
Run wrapper: agent_space/faiss_run_gfx1101.py (HIP_VISIBLE_DEVICES=1, OPENBLAS_NUM_THREADS=1).

### GPU test results (all run individually via faiss_run_gfx1101.py)

- TestGpuSelect             6/6   PASS  (run twice, deterministic; warp-select de-risking gate on gfx1101)
- TestGpuDistance          28/28  PASS  (hipBLAS GEMM path; BF16 subtests self-skip as documented)
- TestCodePacking           4/4   PASS  (CPU correctness test; uint8_t->int fix validated)
- TestGpuIndexBinaryFlat    4/4   PASS
- TestGpuIcmEncoder         7/7   PASS  (hipRAND path; parameterized suite)
- TestGpuResidualQuantizer  1/1   PASS
- TestGpuIndexFlat         18/18  PASS  (exit 0, no teardown crash; LargeIndex max abs diff 0)
- TestGpuIndexIVFFlat      21/21  PASS  (336s total; LongIVFList 171s)
- TestGpuIndexIVFScalarQuantizer  12/12  PASS
- TestGpuIndexIVFPQ        13/13  effective PASS
    (11/13 in monolithic run; Float16Coarse + Add_IP PASS in --gtest_filter isolation;
     shared-RNG-advance past float16 PQ 3.5% tol is the SAME documented non-bug as all other platforms)
- TestGpuMemoryException:   exit 3, hipError 9 (invalid configuration argument) -- same class as
    gfx1201 (exit 3, hipErrorInvalidConfiguration). Test expects clean OOM->catchable exception;
    on 25GB discrete VRAM the brokenAddDims allocation fails with invalid config rather than a clean
    OOM. Port code is correct; discrete GPU OOM behavior differs by device capacity. Not a port defect.

Total: ~119 tests across 11 suites (all functional correctness suites PASS).
GPU arch in faiss.dll: --offload-arch=gfx1101 confirmed in build_gfx1101 CMakeCache.
No delta-port needed: zero source changes vs ab1dcf71 (cmake -DCMAKE_HIP_ARCHITECTURES=gfx1101 only).
Result matches gfx1201 (119 tests) exactly.

State: completed. validated_sha = ab1dcf71

## Port round 2026-08-20 (porter, linux-gfx90a) -- defect fix + history rewrite -> ported

Branch rewritten: ab1dcf71 (3 commits) -> 514cb457 (3 commits). Force-pushed with
--force-with-lease; no upstream PR existed (pr_state none), so moat-port was not frozen.

### THE DEFECT: a leaked hipify artifact, not a fix -- REMOVED

Commit e9fed66 carried `#include "hip/hip_runtime.h"` as LINE 1 of
faiss/gpu/test/TestCodePacking.cpp, ABOVE the copyright header and unguarded.
That file also compiles in NVIDIA/CUDA builds, where no HIP headers exist, so
upstream CUDA CI would have failed on it. The earlier note (Revalidation
2026-06-04, linux-gfx1100) rationalized it as "made explicit in source for MSVC
builds". That rationalization was wrong. Evidence gathered this round:

1. hipify-perl (ROCm 7.14) on the PRISTINE upstream TestCodePacking.cpp makes
   exactly ONE change: it inserts `#include "hip/hip_runtime.h"` at line 1. That
   is the whole diff. So the committed line is byte-identical to what the
   translator emits.
2. hipify.sh's `find ./gpu-tmp -name "*.cpp"` is RECURSIVE, so it processes
   faiss/gpu/test/*.cpp. Confirmed empirically: this round's cmake configure
   (which runs hipify.sh via execute_process) modified 154 tracked files in
   place, and inserted that exact include into BOTH TestCodePacking.cpp AND
   TestUtils.cpp -- the latter never having carried it in any commit. The include
   is generated, on every ROCm configure, for free.
3. The Windows flow does NOT bypass this. The recorded gfx1201/gfx1151 recipe
   runs `bash faiss/gpu/hipify.sh` manually with hipify-perl on PATH; only the
   SECOND hipify_dir call (c_api) fails on Windows. The faiss/gpu call succeeds,
   so Windows gets the include inserted exactly like Linux.
4. TestCodePacking.cpp uses ZERO hip*/cuda* API, no kernels, no launches. It
   needs no HIP header of its own on any platform.
5. Keeping it in source is actively harmful beyond the CUDA break: hipify-perl is
   NOT idempotent for this insertion. Re-hipifying a file that already contains
   the include ADDS A SECOND copy (verified). The committed line made every
   manual re-hipify stack another duplicate -- the same non-idempotency class
   already documented above for device_functions.h.

CONCLUSION: the include was collateral from editing TestCodePacking.cpp in a
hipified worktree -- precisely the failure the "## DO NOT COMMIT the generated
artifacts" section above warns about. Chose REMOVAL over a `#if defined(USE_AMD_ROCM)`
guard: a guard would fix the CUDA break but leave a redundant hunk that a Meta
reviewer would rightly question, would still not stop hipify from inserting its own
copy above it, and would not restore idempotency. Removal returns the file to
upstream form plus only the uint8_t fix.

### Also fixed this round

- Commit 1 message said "gfx1100/gfx1151 followers" (in-house vocabulary that would
  have shipped upstream). Reworded to name the GPUs. `jargon.py --port faiss` clean.
- Commit 1 message claimed the doubled hip/ prefix affects "ROCm 7.x" generally.
  Verified FALSE on 7.14: hipify-perl there emits the correct <hip/device_functions.h>.
  The sed is still needed for 7.2.x and is a harmless idempotent no-op on 7.14.
  Message now scopes the claim ("observed on ROCm 7.2.1; ROCm 7.14 emits the correct
  path already").
- Commit 1 Test Plan used literal /opt/rocm paths; changed to $ROCM_PATH so the recipe
  reads correctly on installs that are not at /opt/rocm (this host is one).
- Commit 2 title was 74 chars (over the 72 limit) and carried a redundant "faiss:"
  prefix. Retitled to 58 chars.
- Commit 3 (TestUtils.cpp) left `<time.h>` as an orphan include -- clock_gettime and
  timespec were its only consumers and it removed both. Dropped it; `<chrono>` takes
  its slot, keeping the include block in the file's existing sorted order.
- Commit 3 message referenced commit "e9fed66" by sha, which the rewrite invalidated.
  Now refers to it descriptively.

### Doc gap closed (INSTALL.md)

INSTALL.md already documented `-DFAISS_ENABLE_ROCM=ON` (Meta maintains the ROCm path),
but its GPU option list documented `-DCMAKE_CUDA_ARCHITECTURES` with no AMD parallel.
Our own porting experience is that a ROCm install shipping no default target list
(no bin/target.lst) gives the HIP compiler no arch at all, so
`-DCMAKE_HIP_ARCHITECTURES` is REQUIRED there, not optional. Added one bullet in the
same list, same house style, immediately after the ROCm bullet.

### Scan of the other two commits (requested)

Only three added lines on the whole branch mention hip/rocm/amd/gfx: the two in
hipify.sh (which executes only when FAISS_ENABLE_ROCM=ON) and the defective include.
TestUtils.cpp's change is pure standard C++ (`<chrono>`), portable to nvcc. Nothing
else is unguarded or CUDA-hostile. Branch is clean.

### Build + test recipe on ROCm 7.14 (TheRock wheel, NO /opt/rocm)

Prior gfx90a validation used ROCm 7.2.1 at /opt/rocm; that path no longer exists on
this host. Adaptations, all environmental (zero source impact):

```
SDK=/opt/conda/envs/py_3.12/lib/python3.12/site-packages/_rocm_sdk_devel
export ROCM_PATH=$SDK
export PATH=$SDK/bin:$SDK/libexec/hipify:$PATH   # hipify-perl must be on PATH:
                                                 # CMake runs hipify.sh at configure
cmake -S . -B build -G Ninja \
  -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_ROCM=ON -DFAISS_ENABLE_CUVS=OFF \
  -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_C_API=OFF -DFAISS_ENABLE_EXTRAS=OFF \
  -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON \
  -DFAISS_OPT_LEVEL=generic -DFAISS_ENABLE_MKL=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_C_COMPILER=$SDK/bin/amdclang \
  -DCMAKE_CXX_COMPILER=$SDK/bin/amdclang++ \
  -DCMAKE_HIP_COMPILER=$SDK/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=$SDK \
  -DBLAS_LIBRARIES=$SDK/lib/host-math/lib/librocm-openblas.so \
  -DLAPACK_LIBRARIES=$SDK/lib/host-math/lib/librocm-openblas.so
```

Deltas vs the ROCm 7.2.1 recipe recorded above:
- No system libopenblas-dev on this host. The SDK BUNDLES OpenBLAS as
  lib/host-math/librocm-openblas.so; pointing BLAS_LIBRARIES/LAPACK_LIBRARIES at it
  avoids an apt install entirely (the Windows sessions used the same library).
  Keep OPENBLAS_NUM_THREADS=1 -- it is still OpenBLAS, so the many-core heap
  artifact documented above still applies.
- CMAKE_PREFIX_PATH must be set explicitly. The top CMakeLists prepends /opt/rocm,
  which does not exist here, so find_package(HIP)/find_package(hipBLAS) need the hint.
- Ninja instead of Make (faster; no behavioral difference observed).
- C_API and EXTRAS OFF to save wall clock. Neither is a GPU gate and neither is
  touched by any commit on this branch.
- CMAKE_HIP_ARCHITECTURES=gfx90a still propagates as --offload-arch automatically;
  no target property needed, same as 7.2.1.

### Results at 514cb457 (MI250X gfx90a, ROCm 7.14, HIP_VISIBLE_DEVICES=1)

- libfaiss.so + faiss_gpu_objs: 303/303 targets, 0 errors.
- 11 GPU test targets: 27/27, 0 errors.
- `ctest -j1 -R "TestGpu|TestCodePacking"`: **108/108 PASSED, 0 failed** (626s).
  Identical suite size and result to the ROCm 7.2.1 baseline (108/108).
- TestCodePacking (the touched file): 4/4 PASSED.
- TestGpuSelect (raft/cuvs de-risking gate): 6/6 PASSED.
- TestGpuIcmEncoder (direct; not matched by ctest -R): 7/7 PASSED.

No regression. The port builds and passes identically on ROCm 7.14 and 7.2.1.

### Platform effect

head_sha ab1dcf71 -> 514cb457 orphans all five platforms' validated_sha, so every
arch is now stale and needs re-testing. This was unavoidable: the jargon in commit 1's
message and the defective include both live in already-validated commits, and neither
is fixable by appending. The gfx90a evidence above was gathered AT the new head, so
that arch's re-test is already satisfied in substance. The delta for the other four is
tiny and test-only: one removed include line (which their own hipify step re-inserts
at configure), one removed `<time.h>`, and an INSTALL.md bullet. libfaiss device code
is untouched by all of it -- a binary-equivalence carry-forward should apply cleanly.

## Review 2026-08-20 (reviewer, linux-gfx90a) -> changes-requested

Scope: full review of all three commits on moat-port, `git diff 0c72755ec...HEAD`
(the 2026-06-01 review predates the two Windows commits, so nothing here is a delta
review). Fork clone clean: `git status --porcelain` shows zero modified tracked files;
every `.hip` / `gpu-backup` entry is untracked configure output. Integrity gate OK
(`moatlib.py audit-clean faiss`).

Two problems, both in upstream-visible text, both fixable in one porter round.

### 1. faiss/gpu/hipify.sh:79 -- the comment still carries the claim this round proved false

    # undo the doubled hip/ prefix hipify-perl (ROCm 7.x) emits for device_functions.h

"ROCm 7.x" is exactly the over-broad claim the porter caught and corrected in commit
a509e6a8b's message ("Some hipify-perl releases map ... Observed on ROCm 7.2.1; ROCm
7.14 emits the correct path already, so the fix is a no-op there"). The in-source
comment was not updated with it, so the branch now asserts two different things about
the same sed. Re-confirmed empirically this session on this host's ROCm 7.14
(`_rocm_sdk_devel/libexec/hipify/hipify-perl` on the pristine
`faiss/gpu/utils/PtxUtils.cuh`): the output is `#include <hip/device_functions.h>`,
correct, no doubled prefix. Drop the version parenthetical or scope it the way the
commit message does -- e.g. "undo the doubled hip/ prefix some hipify-perl releases
emit for device_functions.h (observed on ROCm 7.2.1)". A Meta reviewer reading only
the script would otherwise be told the wrong affected range.

### 2. Commit bd1809f32 -- Test Plan has no commands and no fenced block

AGENTS.md requires a Test Plan "with literal commands in fenced blocks". bd1809f32's
Test Plan is a prose lead-in plus an indented results table; there is no ``` fence and
no command anywhere in the body. The other two commits (a509e6a8b, 514cb4577) both do
this correctly. The repository's own gate agrees:

```
$ python3 utils/moatlib.py audit-commits faiss
faiss bd1809f: Test Plan has no fenced command block
```

Note that `pr_ready` does NOT call `commit_message_problems`, so nothing downstream
catches this before publication -- only `check.py`'s slow `commits` gate and this
review. Add the fenced build/run commands the message already describes in prose (the
Windows gfx1151 recipe, or at minimum the Linux gfx90a `TestCodePacking` re-check it
cites).

### Control-plane defect riding on this branch (not the fork)

utils/check.py:605, from c3e02db (`check: gate the commit-message rules on fork
branches`), duplicates its own trailing comment and strips the one it moved:

    "forks": (gate_forks, True),
    "commits": (gate_commits, True),   # slow: shells out per fork clone      # slow: shells out per fork clone

The `forks` entry lost its explanatory comment and the `commits` entry carries two
copies of it. Cosmetic, but it merges to `main` with this port branch.

### Checked and clean (no action -- recorded so the next round does not re-derive it)

- **No unguarded HIP in CUDA-compiled files.** Of the whole branch diff, only three
  added lines mention hip/rocm/amd/gfx: the two in `faiss/gpu/hipify.sh` (a script that
  runs only from the `FAISS_ENABLE_ROCM` arm of CMakeLists.txt:87) and the INSTALL.md
  bullet. The leaked `#include "hip/hip_runtime.h"` is gone from
  `faiss/gpu/test/TestCodePacking.cpp:1`; the file now opens on the Meta copyright
  header. The only HIP includes left in tracked `faiss/`/`c_api/` source are
  `faiss/gpu/utils/Float16.cuh:28-29`, inside the pre-existing
  `#if !defined(USE_AMD_ROCM) ... #else` arm, untouched by this branch.
- **Both test-file changes are nvcc-portable standard C++.** TestCodePacking.cpp uses
  only `<random>`; TestUtils.cpp swaps `<time.h>` for `<chrono>`.
- **The hipify sed is correct, scoped and idempotent.** Anchored on the full literal
  `#include <hip/hip/device_functions.h>`, so it cannot over-match `hip/hip_runtime.h`;
  a no-op once applied, and a no-op on ROCm 7.14 (verified above). It sits in the
  `hipify_dir()` post-hipify loop beside the existing hipblas/hiprand corrections, and
  because that loop is relative to the cd'd directory it covers both `faiss/gpu` and
  `c_api/gpu` (hipify.sh:118-124). The `cuh` extension is in the loop's list, so
  PtxUtils.cuh is reached.
- **uint8_t -> int distribution is behavior-preserving.** `uniform_int_distribution<uint8_t>`
  default-constructs to [0, 255]; the replacement is explicitly `{0, 255}`. libstdc++
  computes its range in `common_type<mt19937::result_type, make_unsigned<T>::type>` =
  `uint32_t` for both `uint8_t` and `int`, so `__urange` is 255 either way and the
  generated sequence is unchanged. All eight `dist(gen)` sites assign into `uint8_t` or
  mask with `& mask`, so the widened type truncates identically. The tests seed from
  `std::random_device` anyway, so no sequence identity is relied upon.
- **INSTALL.md bullet is accurate and in house style.** Two-space indent, backticked
  option, parenthetical doc link, trailing comma -- parallel to the
  `-DCMAKE_CUDA_ARCHITECTURES` entry three bullets above, placed immediately after the
  `-DFAISS_ENABLE_ROCM` entry. Multi-arch example `"gfx90a;gfx942"` pins nothing.
- **ROCm fault classes have nothing to say about this delta.** No device code is
  touched: the four files are one shell script, one Markdown doc, and two host-only
  test TUs. No warpSize/32 literal, no texture or resource handle, no neighbor read, no
  pitch, no library swap, no per-arch branch anywhere in the diff. Strategy is
  unchanged and correct (faiss's own configure-time hipify driver, ROCm gated behind
  `FAISS_ENABLE_ROCM` default OFF; the CUDA arm of CMakeLists.txt:88-92 is untouched).
- **Message hygiene otherwise clean.** `jargon.py --port faiss`: clean. Titles 67 / 58 /
  61 chars, all `[ROCm]`-prefixed. All three bodies disclose AI assistance. No
  `Co-Authored-By`, no noreply trailer, ASCII throughout, no AMD-internal account
  reference. a509e6a8b's Test Plan uses `$ROCM_PATH`, not a literal `/opt/rocm`.
- **The promoted lesson is correct.** `.claude/skills/cuda-to-rocm/references/strategy-a-cmake.md:114-130`
  (on this branch, from d18d46c) claims hipify-perl prepends
  `#include "hip/hip_runtime.h"` and is non-idempotent for that insertion. Both
  reproduced here on ROCm 7.14: one pass over the pristine TestCodePacking.cpp inserts
  exactly that line at line 1 and changes nothing else; a second pass adds a SECOND
  copy. The entry's advice (delete, do not guard) follows from that.

### gfx90a evidence at 514cb457 -- substantively sound, with one caveat for the validator

The porter's 108/108 run is real: `build/Testing/Temporary/LastTest.log` ends
"Aug 20 05:15 UTC" with 108 `Passed` and zero failures. But the three commits were
re-created at 05:17 (committer date), i.e. AFTER that run, so the run was not against
the 514cb457 commit object. The reflog is
`ab1dcf710 -> 8c98cf861 ("rebuild") -> 514cb4577 ("rebuild2")`, and
`git diff 8c98cf861 514cb4577` is INSTALL.md alone (+5 lines). So the compiled source
tree at test time is identical to 514cb457's except for a Markdown file. The evidence
carries; the validator should record it as such rather than as a run at the commit.

### Platforms needing validation

Nothing is validated at 514cb457. All five platforms still record
`validated_sha = ab1dcf71`: linux-gfx90a, linux-gfx1100, windows-gfx1101,
windows-gfx1151, windows-gfx1201. Fixing the two findings above advances head again,
so the porter should land both before any validator starts. The resulting delta stays
test/doc/build-glue only, so a binary-equivalence carry-forward remains the expected
route for the four non-gfx90a arches.

## Porter round 2026-08-20 (linux-gfx90a) -- answers the 2026-08-20 review

Both findings fixed by rebuilding the three-commit branch (no upstream PR, so a
rewrite is allowed). New head `378d793e` (was `514cb457`); old tip kept locally as
`backup-514cb457` in the fork clone.

**Finding 1 (hipify.sh comment scope).** The comment now says what the commit message
says. Folded into the commit that owns the file (`b4cf8268`, was `a509e6a8`), not a
follow-up commit, so the branch never asserts the over-broad claim:

```
-    # undo the doubled hip/ prefix hipify-perl (ROCm 7.x) emits for device_functions.h
+    # ROCm 7.2.x hipify-perl emits a doubled hip/hip/ prefix for device_functions.h;
+    # undo it (harmless no-op on later versions, which emit the correct path)
```

`bash -n faiss/gpu/hipify.sh` passes; the sed itself is byte-identical.

**Finding 2 (bd1809f Test Plan had no fenced command block).** Reworded in place
(`111ccaa8`, was `bd1809f3`). The Test Plan now carries the literal Windows gfx1151
hipify/configure/build/run commands and the Linux gfx90a `TestCodePacking` re-check in
two fenced blocks, keeps the 10-suite results table, and explains `-- -k 0` (the
`gtest_discover_tests` 5s timeout, not a link failure). Gate now clean:

```
$ python3 utils/moatlib.py audit-commits faiss
OK: fork commit messages conform (1 local clone(s) judged)
$ python3 utils/jargon.py --port faiss
jargon: clean
```

**No rebuild was needed, and this is verifiable rather than asserted.** The whole tree
delta versus the reviewed tip is the one comment line above, in a script that runs only
from the `FAISS_ENABLE_ROCM` arm of CMakeLists.txt and is not compiled on either path:

```
$ git diff 514cb4577 378d793e6 --stat
 faiss/gpu/hipify.sh | 3 ++-
$ git diff 514cb4577 378d793e6 -- . ':!faiss/gpu/hipify.sh'
(empty)
```

So the porter's 108/108 gfx90a ctest run recorded above still describes this tree
exactly, minus a comment. No `timeit.sh compile` entry for this round: nothing was
compiled, and inventing one would be false telemetry.

**Classification / platform state.** `advance-head` moved head to `378d793e`; all five
platforms carry `validated_sha = ab1dcf71` and therefore read as needing revalidation,
same as they did at `514cb457`. Note for whoever carries them forward:

```
$ python3 utils/moatlib.py classify faiss 514cb45770310d75400c096bcbd5570cea907b54 378d793e6...
class=mixed arch_independent=False inert=False
faiss/gpu/hipify.sh: unknown file type
```

`mixed` here is only the classifier having no rule for `.sh`; the single hunk is a
comment. Against `ab1dcf71` the delta is still hipify.sh plus the two host-only test
TUs (TestCodePacking.cpp, TestUtils.cpp), neither of which enters `faiss_gpu_objs`
device code, so binary-equivalence carry-forward remains the expected route for the
four non-gfx90a arches, exactly as the review predicted.

**Control-plane defect from the review also fixed** (rides on this branch, not the
fork): `utils/check.py` GATES table had the `# slow: shells out per fork clone`
comment duplicated on `commits` and missing from `forks`.
