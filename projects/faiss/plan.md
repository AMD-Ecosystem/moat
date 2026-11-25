# FAISS -- ROCm/HIP port plan (lead platform: Linux gfx90a, ROCm 7.2.1)

## Project
- Name: faiss
- Upstream: https://github.com/facebookresearch/faiss
- Default branch: main
- Analyzed at: 0c72755 (shallow clone in projects/faiss/src, version 1.14.2)
- Lead platform: Linux gfx90a (MI250X, CDNA2 wave64), ROCm 7.2.1. Host has 4x gfx90a GCDs.

## Existing AMD support: MATURE, MERGED upstream -> decision: ENABLE-AND-ADAPT (not a fresh port)

FAISS already carries first-class, Meta-maintained ROCm support in `main`. This is NOT a from-scratch
CUDA->HIP conversion and NOT a finish-the-stale-branch job. The disposition is "enable the existing
upstream ROCm path, adapt it to our exact ROCm 7.2.1 / gfx90a, and fix whatever does not build or
validate on gfx90a." Evidence (all in the analyzed tree):

- Top CMakeLists.txt:
  - line 66-69: `option(FAISS_ENABLE_GPU ON)`, `option(FAISS_ENABLE_CUVS OFF)`, `option(FAISS_ENABLE_ROCM OFF)`.
  - line 80-93: when `FAISS_ENABLE_ROCM` is ON -> `enable_language(HIP)`, `add_definitions(-DUSE_AMD_ROCM)`,
    `find_package(HIP REQUIRED)`, `find_package(hipBLAS REQUIRED)`, `set(GPU_EXT_PREFIX "hip")`, and
    `execute_process(COMMAND .../faiss/gpu/hipify.sh)`. The CUDA branch sets `GPU_EXT_PREFIX "cu"`.
- faiss/gpu/hipify.sh: a Meta-authored hipify driver that runs `hipify-perl` over `faiss/gpu/*.{cu,cuh,h,cpp,c}`
  IN PLACE (backing up the originals to `faiss/gpu-backup/`), renames `*.cu`->`*.hip`, and applies a few
  post-fixups (`__nv_bfloat16`->`__hip_bfloat16`, `thrust::cuda::par`->`thrust::hip::par`,
  `<hipblas.h>`->`<hipblas/hipblas.h>`, `<hiprand_kernel.h>`->`<hiprand/hiprand_kernel.h>`).
- faiss/gpu/CMakeLists.txt: `if(FAISS_ENABLE_ROCM) list(TRANSFORM FAISS_GPU_SRC REPLACE cu$ hip)` (lines 210-212,
  320-322) so the source list points at the hipified `.hip` files; `target_link_libraries(faiss_gpu_objs
  PRIVATE hip::host roc::hipblas)` (line 343); the IVF-interleaved codegen reads the `.${GPU_EXT_PREFIX}`
  template (line 230) so generated TUs are `.hip` on ROCm.
- faiss/gpu/utils/DeviceDefs.cuh: ROCm-version-aware warp size. On `ROCM_VERSION_MAJOR >= 7` it uses
  `constexpr __device__ int kWarpSize = rocprim::arch::wavefront::max_size();` (includes
  `<rocprim/intrinsics/arch.hpp>`), with a `__AMDGCN_WAVEFRONT_SIZE` 32/64 fallback for ROCm < 7.
  `GPU_MAX_SELECTION_K = 2048` on ROCm.
- The select path is wave-agnostic and fully kWarpSize-parameterized: WarpShuffles.cuh uses `__shfl*`
  (no `_sync`, no 0xffffffff mask) on ROCm with `width = kWarpSize` defaults; PtxUtils.cuh replaces the
  inline-PTX `bfe`/`laneid`/`bar.sync` with HIP `getBitfield`/`getLaneId` implementations under
  `#ifdef USE_AMD_ROCM`. Select.cuh / the BlockSelect/WarpSelect/MergeNetwork machinery is templated on
  `NumWarpQ`/`NumThreadQ` with `kWarpSize`, not a literal 32.

an earlier 2023 ROCm/faiss fork (branches incl. `rocm-warp32-only`) is STALE and
SUPERSEDED by this upstream support -- do NOT use it.

Host preflight (all confirmed present on this gfx90a box, ROCm 7.2.1):
- /opt/rocm/.info/version = 7.2.1.
- /opt/rocm/include/rocm-core/rocm_version.h defines ROCM_VERSION_MAJOR=7, MINOR=2, PATCH=1 (so the
  ROCm-7 `rocprim::arch::wavefront` path in DeviceDefs.cuh is the one we compile).
- /opt/rocm/include/rocprim/intrinsics/arch.hpp defines `rocprim::arch::wavefront::max_size()` (line 98),
  `ROCPRIM_HOST_DEVICE constexpr`; on the device pass it returns the compile-target wave size (64 for
  gfx90a), so the exact API DeviceDefs.cuh calls EXISTS in 7.2.1.
- hipBLAS cmake package present (/opt/rocm/lib/cmake/hipblas/), so `find_package(hipBLAS)` and the
  `roc::hipblas` target resolve.
- 4x gfx90a GCDs via rocm_agent_enumerator. No /opt/rocm/bin/target.lst -> CMake HIP has no default arch,
  so the build MUST pass `-DCMAKE_HIP_ARCHITECTURES=gfx90a` explicitly (see Build commands / Risks).

## Build classification: pure CMake with a hipify pre-step (NOT a pytorch extension) -> Strategy A-variant

- Decider: top CMakeLists.txt is a standalone CMake project (`project(faiss ... LANGUAGES ...)`, line 53-57);
  GPU sources are `.cu`; no `find_package(Torch)`, no `torch.utils.cpp_extension`/`CUDAExtension`. Torch only
  appears in optional Python contrib *tests* (`torch_test_contrib_gpu.py`), not in the build. So this is the
  pure-CMake class (PORTING_GUIDE "Build classification").
- It is NOT the colmap "mark .cu as LANGUAGE HIP + one compat header" Strategy A, and NOT setup.py
  Strategy B. FAISS uses a third, already-implemented pattern: a configure-time **hipify.sh** that
  translates `faiss/gpu/*.cu` -> `*.hip` in place and a CMake `FAISS_ENABLE_ROCM` switch that
  `enable_language(HIP)` and points the source list at the `.hip` files. Because this machinery is already
  upstream and Meta-maintained, our job is to drive it on gfx90a/7.2.1 and fix gaps -- we add as little as
  possible (ideally only an arch default; see File-by-file). Set ext_type = "cmake".

## CUDA surface inventory (faiss/gpu)
- ~74 `.cu` + ~65 `.cuh` translation units. Kernels: IVFFlat/IVFPQ/IVFScalarQuantizer scan
  (impl/scan/*, IVFInterleaved codegen over a Codec x metric x (threads|warpQ|threadQ) matrix), flat L2/IP
  distance (GpuDistance, L2Norm, Distance), PQ code load/scan (PQCodeLoad.cuh, PQScanMultiPass*),
  brute-force + warp/block top-k selection (Select.cuh, BlockSelectKernel, WarpSelectKernel, MergeNetwork),
  vector residual, transpose, ICM encoder.
- Warp intrinsics: `__shfl*` via WarpShuffles.cuh (ROCm path already drops `_sync`/mask, uses width=kWarpSize);
  bitfield/laneid/bar.sync inline PTX in PtxUtils.cuh (already HIP-ported under USE_AMD_ROCM). No raw
  hardcoded `32` warp literal in the select code -- it is kWarpSize-driven. No `__ballot`/`__activemask`
  reliance with a 32-bit mask on the hot path (the select uses shuffles, not ballots).
- Libraries:
  - cuBLAS (16 `cublasHandle`, GemmEx/SgemmEx/GemmStridedBatchedEx, SetMathMode) in
    utils/BlasHandle, utils/MatrixMult-inl.cuh, GpuResources -> hipBLAS. hipify maps the symbols; CMake links
    `roc::hipblas`; hipify.sh fixes the `<hipblas/hipblas.h>` include path. Watch hipBLAS v2 enum/signature
    diffs (PORTING_GUIDE library-swaps) -- the GemmEx compute-type/algo enums are the usual friction point.
  - cuRAND: only impl/IcmEncoder.cu (`curand_kernel.h`, `curandState_t`, `curand_init`, `curand_uniform`)
    -> hipRAND device API; hipify.sh fixes the `<hiprand/hiprand_kernel.h>` include. Small, isolated.
  - Thrust: ~10 TUs use `thrust::` -> rocThrust (drop-in; same headers under /opt/rocm/include). hipify.sh
    rewrites `thrust::cuda::par`->`thrust::hip::par`. rocPRIM/rocThrust hard-`#error` below C++17, but FAISS is
    `CMAKE_CXX_STANDARD 20` so the floor is satisfied (no `-std` bump needed, unlike GPUMD).
  - No cuFFT, cuSPARSE, cuDNN. No CUB direct (uses faiss's own select, not cub).
- half / bfloat16: ~15 TUs (Float16.cuh, MathOperators.cuh). hipify.sh pre-rewrites `__nv_bfloat16(2)`
  -> `__hip_bfloat16(2)` before hipify to avoid the inaccurate auto-conversion.
- Memory/streams: StandardGpuResources.cpp uses `cudaHostAlloc` (pinned, line 361) and `cudaMallocManaged`
  (Unified space ONLY, line 593) -> hipHostMalloc / hipMallocManaged; default device allocation is plain
  `cudaMalloc` (Unified is opt-in MemorySpace, NOT the default). Streams/events via
  `cudaStreamCreate`/`cudaEventCreate` -> hip equivalents. All within hipify's 1:1 mapping table.
- NO textures, NO surfaces, NO cudaArray, NO jitify/nvrtc, NO CUDA driver API (cuModule/cuLaunch). This
  removes a whole bank of PORTING_GUIDE fault classes (texture linear-filter, 256B texture pitch,
  layered-array coherency, rule-of-five on texture handles) from scope.

## cuVS (optional, OUT OF SCOPE for this port)
`FAISS_ENABLE_CUVS=ON` pulls NVIDIA cuVS/raft (CuvsCagra/CuvsFlatIndex/CuvsIVF*) and `USE_NVIDIA_CUVS=1`.
cuVS is CUDA/raft-only and is mutually exclusive with ROCm here. Build with `FAISS_ENABLE_CUVS=OFF` (the
default). CAGRA-on-cuVS tests (TestGpuIndexCagra, TestGpuFilterConvert, test_cuvs.py) are CUVS-gated and do
not compile/run on the ROCm build -- that is expected, not a regression.

## Risk list (keyed to PORTING_GUIDE fault classes)
1. ARCH DRIFT (primary, low-severity-but-must-handle): upstream ROCm CI targets **gfx942** (MI300; it writes
   `gfx942` to /opt/rocm/bin/target.lst on the runner -- build_cmake/action.yml ~line 150) on an MI325 box.
   We target **gfx90a**. Both are CDNA wave64 so the wave-size logic is identical, but our host has NO
   target.lst, so HIP has no default arch. MUST pass `-DCMAKE_HIP_ARCHITECTURES=gfx90a`. The ROCm
   `faiss_gpu_objs` target currently sets no `HIP_ARCHITECTURES` and no `--offload-arch`
   (faiss/gpu/CMakeLists.txt line 343-344 is empty options), so it relies entirely on the cache var. If the
   cache var does not propagate to the HIP TUs, fall back to `CMAKE_HIP_FLAGS=--offload-arch=gfx90a`.
   This is exactly the PORTING_GUIDE "configurable HIP arch, never a literal gfx90a" lesson: do NOT hardcode
   gfx90a in the fork's CMake -- keep it cache-var-driven so gfx1100/gfx1151 followers build with only a
   `-DCMAKE_HIP_ARCHITECTURES=<arch>` change and no source/commit churn.
2. VERSION DRIFT (low): upstream ROCm CI is ROCm 7.2 (build_cmake/action.yml `ROCM_VERSION="7.2"`); we are
   7.2.1 -- same minor, so the `ROCM_VERSION_MAJOR>=7` rocprim-wavefront path matches. Confirmed the
   `rocprim::arch::wavefront::max_size()` symbol and `rocm-core/rocm_version.h` exist on our box. Low risk;
   watch for any 7.2.1 hipBLAS GemmEx enum tightening vs 7.2.0.
3. WARP SIZE on FOLLOWERS (deferred to delta-plan): gfx90a is wave64 and the select is kWarpSize-driven, so
   the lead should be clean. gfx1100/gfx1151 are RDNA wave32; `max_size()` returns 32 there, exercising the
   other kWarpSize value. Static shared-mem arrays must be sized to a compile-time bound, not the runtime
   wave -- audit BlockSelect shared storage when the follower delta-plan is requested. No action for gfx90a.
4. hipBLAS GemmEx/SgemmEx/GemmStridedBatchedEx enum + signature diffs (PORTING_GUIDE library swaps): the most
   likely real compile/runtime friction. hipBLAS v2 compute-type/algo enums differ from cuBLAS; the
   `cublasSetMathMode`/tensor-op path may map differently. Localize to utils/MatrixMult-inl.cuh and
   utils/BlasHandle if GEMM tests misbehave.
5. Managed-memory int atomicMin/Max silent-drop on gfx90a (cudaKDTree lesson): NOT triggered here -- grep
   found ZERO `atomicMin`/`atomicMax` in faiss/gpu, and managed memory is opt-in (`MemorySpace::Unified`),
   not the default allocator. No action; noted so the validator does not chase a phantom.
6. OOB neighbor reads / warp-synchronous unsynced reductions (colmap, MPPI lessons): the select uses
   explicit shuffles and barriers, not a `volatile sdata` warp-synchronous tail, and is the most-exercised
   ROCm path upstream. Low risk, but TestGpuSelect determinism is the direct check.
7. hipify mutates the read-only reference tree: `hipify.sh` runs at CMAKE configure (execute_process) and
   edits faiss/gpu IN PLACE, creating faiss/gpu-backup/ and renaming .cu->.hip. In the porter's FORK clone
   this is expected (it is how the upstream build works). For THIS planner's read-only src clone we did not
   configure, so it is untouched. The porter should be aware that a clean reconfigure re-hipifies from
   gpu-backup/; if a rebuild ever looks stale after editing a `.hip`, re-run hipify.sh (the Strategy-B
   "re-hipify before rebuild" gotcha applies to this hybrid too). Recommend the porter NOT commit the
   generated `.hip`/gpu-backup/ artifacts (they regenerate at configure).
8. Large-image link reach (cudf `--offload-compress` lesson): FAISS GPU is heavily templated (IVF codegen
   matrix, select on the Codec x metric grid). Single-arch (gfx90a only) keeps the fatbin small and should
   link fine, but if `faiss_gpu_objs` -> libfaiss.so link hits `R_X86_64_PC32` overflow, add
   `--offload-compress` to the HIP compile options before any split. Watch only; likely unneeded at one arch.

## File-by-file change list (expected MINIMAL -- this is enable-and-adapt)
The realistic expectation is ZERO-to-FEW source edits; the upstream ROCm path is meant to build as-is on
CDNA. Concretely:
- NO change anticipated to: DeviceDefs.cuh, WarpShuffles.cuh, PtxUtils.cuh, Select.cuh and the
  BlockSelect/WarpSelect machinery (already ROCm-correct and wave-agnostic), hipify.sh.
- faiss/gpu/CMakeLists.txt (line 342-344, ROCm branch): if `-DCMAKE_HIP_ARCHITECTURES` does not reach the
  HIP TUs, add `set_target_properties(faiss_gpu_objs PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")`
  GATED so it defaults the lead arch only when unset (PORTING_GUIDE: never a literal gfx90a; keep followers
  zero-churn). This is the single most-likely necessary edit.
- POSSIBLE, only if they fail to compile/run on 7.2.1: hipBLAS GemmEx enum fixups in
  utils/MatrixMult-inl.cuh / utils/BlasHandle (guard with USE_AMD_ROCM); a 7.2.1-specific hipRAND or
  hipBLAS include/signature nudge. Treat each as a discovered gap, fix minimally under USE_AMD_ROCM.
- Do NOT reimplement the select against CK/rocWMMA. The select is correctness-first kWarpSize CUDA->HIP and
  is precisely the code raft/cuvs vendor; a mechanical (already-done) port that PASSES on gfx90a is the goal.
  (CUTLASS-rewrite guidance does not apply: FAISS GPU is not CUTLASS-based.)

## Build commands (gfx90a)
Dependencies to install first (apt/conda, no need to ask): a CPU BLAS (OpenBLAS:
`apt-get install -y libopenblas-dev`), OpenMP (libgomp, present with gcc), gtest
(`apt-get install -y libgtest-dev` or let CMake fetch), gflags (for perf_tests/benchs, optional). ROCm
7.2.1 + rocm-hip-sdk (hip::host), hipBLAS (roc::hipblas), rocPRIM/rocThrust (header-only, in /opt/rocm)
are already on the host. SWIG + Python bindings are OPTIONAL -- build the C++ + GPU libs and gtests FIRST;
add `-DFAISS_ENABLE_PYTHON=ON` (needs swig, numpy) only after the C++ GPU gates pass.

Recommended C++/GPU-only configure (mirrors upstream ROCm CI minus Python):
```
cmake -S projects/faiss/src -B projects/faiss/src/build \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_ROCM=ON \
  -DFAISS_ENABLE_CUVS=OFF \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DFAISS_ENABLE_C_API=ON \
  -DBUILD_TESTING=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DFAISS_OPT_LEVEL=generic \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
```
(If find_package(HIP) does not auto-locate the compiler, the explicit CMAKE_HIP_COMPILER above is the
PORTING_GUIDE-recommended fallback. CMAKE_PREFIX_PATH already gets /opt/rocm prepended by the top CMake.)

Build:
```
cmake --build projects/faiss/src/build --target faiss faiss_gpu_objs -j$(nproc)
cmake --build projects/faiss/src/build --target \
  TestGpuSelect TestGpuDistance TestGpuIndexFlat TestGpuIndexIVFFlat TestGpuIndexIVFPQ -j$(nproc)
```
Optional manual CPU-only compile smoketest (never a gate): docker image rocm/dev-ubuntu-24.04:7.2.4-complete.

## Test plan
"Validated on gfx90a" = the faiss/gpu C++ gtests build and PASS on a real gfx90a GCD with no non-GPU
regression, with TestGpuSelect as the de-risking gate.

Primary GPU gates (faiss/gpu/test, run on one assigned gfx90a GCD):
- **TestGpuSelect** (TestGpuSelect.hip; tests: test, test1, testExact, testWarp, test1Warp, testExactWarp)
  -- directly exercises runBlockSelect / runWarpSelect over k in [1, GPU_MAX_SELECTION_K=2048], i.e. the
  EXACT kWarpSize warp/block top-k code that raft and cuvs vendor as CUDA-only copies. THIS IS THE
  RAFT/CUVS DE-RISKING GATE: green here proves the wave64 kWarpSize select is correct on real gfx90a.
- TestGpuDistance (flat L2/IP, the hipBLAS GEMM path), TestGpuIndexFlat, TestGpuIndexIVFFlat,
  TestGpuIndexIVFPQ -- the core index kernels + the hipBLAS-backed distance and IVF scan.
- Secondary: TestGpuIndexIVFScalarQuantizer, TestGpuIndexBinaryFlat, TestGpuResidualQuantizer,
  TestGpuIcmEncoder (the only hipRAND consumer), TestGpuMemoryException, TestCodePacking.

Run (serially on ONE GPU per the pacing memory; -jN against a single GPU flaps):
```
cd projects/faiss/src/build && ctest --output-on-failure
# or per-binary, e.g.:  ./faiss/gpu/test/TestGpuSelect
```
The upstream C++-test command is `make -C build test` (== ctest). CUVS-gated tests (TestGpuIndexCagra,
TestGpuIndexBinaryCagra, TestGpuFilterConvert) are absent on the ROCm build by design -- not a regression.

Non-GPU regression set (must not regress): the CPU `tests/` gtests (TestCodePacking and the whole CPU
faiss test suite under top-level `tests/`) and, only if Python is built, the CPU pytest
(`tests/test_*.py`). These are unaffected by the GPU port; run a subset to confirm the host C++ build
still links and passes.

## Staged strategy + likely walls
1. Configure with FAISS_ENABLE_ROCM=ON + gfx90a; confirm hipify.sh runs and the .hip tree + IVF codegen
   appear. WALL: arch not propagating to HIP TUs -> add the gated HIP_ARCHITECTURES set or
   `--offload-arch=gfx90a` in CMAKE_HIP_FLAGS (Risk 1).
2. Build libfaiss + faiss_gpu_objs. WALLS: (a) hipBLAS GemmEx enum/signature mismatch on 7.2.1
   (Risk 4) -> minimal USE_AMD_ROCM fixup in MatrixMult-inl.cuh/BlasHandle; (b) link reach if templated
   fatbin is huge (Risk 8) -> `--offload-compress`. Both are PORTING_GUIDE-known.
3. Build + run TestGpuSelect FIRST (the de-risking gate). Determinism: a select bug from a wave64 issue
   would show as wrong/non-deterministic top-k -- run twice and diff if suspicious (MPPI determinism check).
4. Run the remaining index gtests (Distance/Flat/IVFFlat/IVFPQ). WALL: an IVF/PQ scan kernel with its own
   wave assumption -- unlikely (the scan is templated on SUB_THREADS/warpQ, and the warp ops route through
   WarpShuffles.cuh's kWarpSize path), but if a kernel hardcodes 32 outside WarpShuffles, fix per the
   PORTING_GUIDE warp-size class.
5. (Optional) FAISS_ENABLE_PYTHON=ON + swig, run the GPU pytest, only after C++ gates are green.
6. Mark gfx90a validated when the primary + secondary gtests pass with no non-GPU regression.

Effort estimate: LOW-to-MODERATE. Because the ROCm path is merged and CI-exercised on CDNA (gfx942), the
likeliest outcome is "configure with the right arch, build, fix at most a hipBLAS-enum nit, tests pass."
The realistic risk band is a hipBLAS GemmEx adaptation and the arch-propagation CMake one-liner. This is an
enable-and-adapt, not a fix-large-gaps, port.

## Strategic value (capture in PR/notes)
FAISS is the canonical upstream of the `faiss_select/` warp/block-select code that BOTH raft and cuvs
vendor as CUDA-only copies (raft is being patched from upstream FAISS-ROCm; cuvs carries its own copy).
GPU-validating FAISS-ROCm proper -- specifically TestGpuSelect passing on real gfx90a -- proves the
kWarpSize select works on CDNA wave64 and DE-RISKS the raft and cuvs select ports directly.

## Open questions
- Does `-DCMAKE_HIP_ARCHITECTURES=gfx90a` propagate to faiss_gpu_objs without an explicit
  HIP_ARCHITECTURES set on the target? (faiss/gpu/CMakeLists.txt line 343-344 sets no arch.) Resolve at
  configure step 1; if not, add the gated set (File-by-file).
- Does hipBLAS 7.2.1 accept the exact GemmEx compute-type/algo enums the hipified MatrixMult uses, or is a
  v2-enum fixup needed? Resolve at build step 2 / TestGpuDistance.
- Confirm the C API (`-DFAISS_ENABLE_C_API=ON`) hipifies cleanly (hipify.sh also processes c_api/); drop it
  if it costs time -- it is not a GPU gate.
