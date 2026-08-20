# CV-CUDA notes

Kept for the automation exercise; upstreaming unlikely (NVIDIA-affiliated). Strategy A
 (pure CMake) HIP port of the CV-operator core + NVCV, gated behind USE_HIP.
Fork: AMD-Ecosystem/CV-CUDA @ moat-port. Lead arch gfx90a (MI250X, ROCm 7.2.1).

## NESTED LAYOUT WARNING
The fork clone is at `projects/CV-CUDA/src/`, and the repo itself nests one more level:
real sources live under `projects/CV-CUDA/src/src/cvcuda/...` and `.../src/src/nvcv/...`.
The top CMakeLists is at `projects/CV-CUDA/src/CMakeLists.txt`.

## Build (gfx90a, core + C++ gtests, no Python)
```
cmake -S projects/CV-CUDA/src -B projects/CV-CUDA/src/build-hip -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DBUILD_PYTHON=OFF -DBUILD_TESTS=ON -DBUILD_TESTS_CPP=ON -DBUILD_TESTS_PYTHON=OFF \
  -DBUILD_BENCH=OFF -DBUILD_DOCS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build projects/CV-CUDA/src/build-hip -j 16
```
Followers (gfx1100/gfx1151): same command, only change `-DCMAKE_HIP_ARCHITECTURES=<arch>`
(no source/CMake edit -- the targets read ${CMAKE_HIP_ARCHITECTURES}).
Builds CLEAN today: libnvcv_types.so, libcvcuda.so, and all C++ test exes.
Two C++ standards in one build: operator/legacy .cu compile at -std=gnu++17, the
cudatools_system tests at -std=gnu++20 (they use C++20 NTTPs); any compat-header
construct must be valid at BOTH (see MathOps note below).

## GPU validation (one isolated GCD)
4 GCDs (0-3); pick a free one via `rocm-smi --showpidgpus` / `--showmeminfo vram` (other
agents share the box). Run serially (no -jN). Gate suites:
```
HIP_VISIBLE_DEVICES=<n> build-hip/bin/cvcuda_test_system          # per-operator GPU-vs-CPU
HIP_VISIBLE_DEVICES=<n> build-hip/bin/nvcv_test_cudatools_system  # device cuda_tools
```
Current pass rate (gfx90a): cvcuda_test_system 0 failures (2608 pass + disabled/negative, exit 0);
nvcv_test_cudatools_system 1116/1123 (7 residuals, both clusters ROOT-CAUSED to non-port artifacts:
6 InterpolationVarShapeWrap.correct_shift = a TEST-FIXTURE use-after-free on an async copy source
(freed per-iteration dstVec, exposed by HIP's truly-async pageable hipMemcpy2DAsync; production op +
allocator proven correct) + 1 char/signed-char type-identity dictated by the upstream HIP vector header.
Neither is a ROCm/operator defect; see REMAINING below. No source change was needed this cycle.

## Scope decisions
- SCOPED OUT: OpOSD, OpBndBox, OpBoxBlur (+ legacy/osd.cu, box_blur.cu, textbackend/,
  tests/.../OsdUtils.cu, the cvcuda_test_system_smoke exe). They depend on cuOSD, a prebuilt
  CUDA-only static lib (3rdparty/cuOSD/.../libcuosd.a) with NO source -- unportable. Gated out
  under USE_HIP in CvCudaLegacy.h and tests/cvcuda/system/CMakeLists.txt. ~73 operator suites
  remain (the cvcuda_test_system gate).
- nvcv standalone-consumer ExternalProject test skipped on HIP (re-runs cmake without the
  USE_HIP setup, would look for the CUDA toolkit); it is a packaging check, not a GPU test.

## Fault-class fixes applied (all arch-unified; CUDA path byte-for-byte unchanged)
- Compat layer: cmake/hip/ forwarding shim dir (on HIP include path only) + CvCudaHipCompat.h
  force-included on every HIP TU. cstdlib/cstring BEFORE hip_runtime (gpuRIR). Maps the cuda*/
  cub/cublas/cusolver/curand symbols the project uses. Defines NVCV_WARP_FULL_MASK (64-bit).
  MUST NOT define __CUDA_ARCH__ (keeps SaturateCast PTX + NVCV SIMD-intrinsics inert on their
  portable fallbacks). Also provides uint3 operator*/+ for the __hip_builtin_*_t index types and
  a __host__ __device__ declval shim (std::declval is host-only under clang).
- __CUDA_ARCH__ device-math guards extended to `|| __HIP_DEVICE_COMPILE__` (MathWrappersImpl.hpp,
  LinAlg.hpp). DeviceMin/MaxImpl: HIP has no ::umin/::ullmin/::llmin, so the plain ternary (lowers
  to the same min instruction). LinAlg Vector member-initializer dropped on HIP (used as __shared__,
  which clang forbids an initializer on, in BOTH passes).
- Metaprogramming.hpp: char1..4 base_type is `char` on HIP (HIP_vector_type<char,N> members are
  plain char, not signed char) so GetElement's reference return binds. KNOWN COST: this makes
  TypeTraitsMakeTypeVectorTest/3 (signed char) fail is_same_v(BaseType, signed char) -- char and
  signed char are distinct C++ types though same representation; the HIP report is correct for HIP.
- MathOps.hpp: HIP_vector_type ships its own vector operators (CUDA's vector_types.h ships none),
  causing ambiguity for mixed-element vec pairs, vec OP dim3, and cross-type ==/!=. Added HIP-only
  overloads that win partial ordering WITHOUT a requires-clause (must stay C++17-valid for the .cu
  TUs): a both-operands-concrete (vec,vec) for mixed element type (SFINAE !is_same), and
  CONCRETE-dim3 forward/mirror overloads (more specialized than HIP's templated U, which also beats
  HIP's enable_if-constrained integral operators like %). dim3 bodies are spelled out (the shared
  if-constexpr body probes .w, a hard error on the 3-element dim3 even when discarded).
- Warp/wave64: NVCV_WARP_FULL_MASK 64-bit (reduce_kernel_utils FINAL_MASK, OpFindHomography masks).
  OpLabel connected-components + threshold/threshold_var_shape Otsu scans: explicit width-32 on every
  __shfl_*_sync (the block packs 32-lane rows, so a wave64 wavefront is two 32-lane groups), and the
  warp-synchronous reduction tails replaced with fully __syncthreads-synchronized trees (MPPI lesson:
  the low 32 lanes of a 64-lane wavefront are not lockstep across unsynced steps).
- StreamId.cpp: hipStreamGetId + pointer-value fallback. priv/Assert trap -> handled. __ldg ->
  plain load (HIP lacks __ldg for all vector elem types; the cache hint is advisory on CDNA).
  cuBLAS/cuSOLVER -> hipBLAS/hipSOLVER (OpFindHomography), cuRAND -> hipRAND (gaussian_noise),
  CUB -> hipCUB. LTO OFF on HIP. ENABLE_COMPAT_OLD_GLIBC OFF on HIP.
- CudaFwd.h: cudaArray_t/CUstream typedef'd to hip types on HIP.

## THE BIG FIX: zero device allocations on HIP (NVCV DefaultAllocator)
hipMalloc returns RECYCLED device memory with STALE contents; freshly cudaMalloc'd memory on the
NVIDIA setups reads back as zero. ~160 of the operator gtests fill a tensor's valid region, run the
op, then compare the WHOLE strided buffer -- INCLUDING the row-stride padding the operator never
writes -- against a zero-initialized std::vector CPU reference. So they implicitly assume device
padding is zero. On the FIRST op in a process the padding is fresh-zero (passes); every later op
gets recycled dirty padding (fails non-deterministically). Confirmed: any op passes in isolation but
fails after any preceding op; ALL mismatching bytes are at offsets >= validRowBytes within each row
stride. Fix: src/nvcv/src/priv/DefaultAllocator.cpp doAllocCudaMem -> hipMemset(ptr,0,size) under
USE_HIP. Took cvcuda_test_system 2405->2565 and cudatools_system 1015->1114. This is the single
most important non-build fix; keep it.

## Test-source-only fixes (HIP_vector_type ergonomics, not operator bugs)
- 1-element vector from a scalar: `int1 x = {v}` / `ValueAt<int1>(...,{z})` use HIP_vector_type's
  EXPLICIT scalar ctor -> rejected as copy-list-init. Use direct-list-init `int1{v}` or cuda::SetAll.
  Fixed in TestOpMinMaxLoc.cpp, TestOpSIFT.cpp, TestTypeTraits.cpp.
- ttype::Value<vec{...}> NTTP: HIP_vector_type IS a usable C++20 NTTP, but its variadic ctor needs
  EXACTLY N args (CUDA's aggregate zero-fills). Spell all components: float4{a,b,c}->{a,b,c,0.f}
  (TestMathOps.cpp). Only under-specified vector brace-inits needed this (2 sites).
- `typename DependentType::ValueType` and `this->member` for dependent-base members (clang two-phase
  lookup): OpCropFlipNormalizeReformat.cu, InterpolationVarShapeWrap.hpp, DeviceTensorBatchWrap.cu.
- DeviceTensorWrap.cu / DeviceFullTensorWrap.cu: DropCast<N>(threadIdx) -> wrap threadIdx as
  uint3{threadIdx.x,..} (the builtin index type has no NVCV TypeTraits; uint3 is byte-identical on CUDA).

## RESOLVED root causes (2026-05-31 porter cycle: cvcuda 2565->0 failures; cudatools 1110->7)
Five arch-unified source fixes took cvcuda_test_system to ZERO failures (2608 pass + the rest
disabled/negative; `cvcuda_test_system` exit 0) and nvcv_test_cudatools_system to 1116/1123. Earlier
"residual genuine failures" were NOT distinct kernel bugs; they reduced to these root causes. CUDA path
byte-for-byte unchanged (every fix HIP-guarded or a build flag).

1. THE TEXTURE-PITCH DIVERGENCE (the big one; fixed OpErase, OpSIFT, OpGaussian, OpFindHomography,
   ~half of HistogramEq, OpNormalize, TensorBatchWrap, several Interpolation cases). NVCV derives the
   tensor/image ROW-PITCH alignment from cudaDevAttrTexturePitchAlignment. NVIDIA reports 32 (tight: a
   640-byte uchar row stays 640); gfx90a reports 256 (640 -> padded to 768). ~160 gtests fill the valid
   region then compare the WHOLE strided buffer including row-stride padding against a zero CPU ref, so
   they assume the CUDA tight pitch. The OpErase "last row unwritten" symptom was the test indexing
   test[9*640] while the real stride was 768 -- the kernel was always correct (proven: injected printf
   showed every write landing at the right logical coord). No CV-CUDA tensor is HW-texture-bound, so the
   256B pitch is unnecessary. Fix: cmake/hip/CvCudaHipCompat.h wraps cudaDeviceGetAttribute in an inline
   shim that clamps the texture-pitch-alignment query to 32 on HIP. Confirmed gfx90a returns 256 for both
   hipDeviceAttributeTexturePitchAlignment and ...TextureAlignment.

2. HistogramEqVarShape (15): the varshape path NEVER zeroed m_histoArray before the atomicAdd histogram
   accumulation (the tensor HistogramEq path does: cudaMemsetAsync). m_histoArray is a direct cudaMalloc
   (not via the NVCV DefaultAllocator), so the DefaultAllocator hipMemset fix did not cover it; recycled
   hipMalloc gave dirty histograms. Fix: cudaMemsetAsync(m_histoArray,0,...) in HistogramEqVarShape::infer
   (histogram_eq_var_shape.cu), matching the tensor path.

3. OpPairwiseMatcher (7): two bugs. (a) The NB=32/128 PointT cache stored RT(uint32) words and read them
   back as type T via reinterpret_cast on a private RT[] array -- a strict-aliasing violation clang/HIP
   exploited at -O3, eliding the float reads so every L2 distance stayed FLT_MAX (-> empty crossCheck
   output). Fixed with a union (RT words / T elems). (b) cub::BlockReduce/BlockRadixSort TempStorage is
   reused across the two SortKeyValue calls in the crossCheck path; on a 64-thread (=wave64) block the
   collective lowers to a single-wavefront reduce with no syncing epilogue, so TempStorage reuse raced.
   Added __syncthreads() after the reduce and at end of SortKeyValue. (OpPairwiseMatcher.cu)

4. gfx90a __fsqrt_rn IS NOT ALWAYS CORRECTLY ROUNDED (fixed OpNormalize + L2 PairwiseMatcher exactness).
   sqrt(93606.0f): __fsqrt_rn -> 0x4398f9b9 but the correctly-rounded value (host std::sqrt, and
   (float)__dsqrt_rn) is 0x4398f9ba (1 ULP high). CUDA sqrt.rn.f32 is correctly rounded, so the bit-exact
   gtests pass on NVIDIA, fail on gfx90a. Fix: DeviceSqrtImpl routes 32-bit sqrt through the correctly
   rounded f64 __dsqrt_rn on HIP (MathWrappersImpl.hpp); CDNA has fast f64 sqrt.

5. cuda::min/max NaN handling (fixed OpMorphology/OpMorphologyVarShape CLOSE on RGBAf32). The morph tests
   fill float images with RANDOM BYTES (NaN/inf), and host gold + device both call cuda::min/max. Host
   MinImpl/MaxImpl = std::min/std::max (b<a?b:a / a<b?b:a). HIP DeviceMinImpl/MaxImpl were a<b?a:b and
   a>b?a:b -- the OPPOSITE NaN selection. Respelled the HIP device ternaries to exactly match the host
   std::min/std::max forms so device==host bit-for-bit on NaN/signed-zero (MathWrappersImpl.hpp).

6. -ffp-contract=on for HIP (fixed OpWarpPerspective cubic + the cubic-math half of Interpolation tests).
   clang(HIP) defaults to -ffp-contract=fast (forms FMAs ACROSS statements); nvcc only contracts within
   one expression (--fmad=true). The extra contraction drifted HIP float results ~1 ULP from the CUDA
   build and CPU gold (e.g. the bicubic weight chain in InterpolationWrap GetCubicCoeffs), failing
   bit-exact compares. Pinned CMAKE_HIP_FLAGS to -ffp-contract=on (CMakeLists.txt) to match CUDA/host.

## REMAINING cudatools residuals (7; ROOT-CAUSED to NON-PORT artifacts, GPU compute proven correct)
Both residual clusters were ROOT-CAUSED to non-production artifacts in the 2026-05-31 (b) porter cycle.
No production fix is warranted; no source change was needed. cvcuda_test_system stays 0 failures.

- InterpolationVarShapeWrapTest.correct_shift (6-8 per run; non-deterministic SET): a TEST-FIXTURE
  USE-AFTER-FREE on an async copy source, NOT a GPU/operator/allocator bug. ROOT CAUSE (this cycle,
  conclusive): in TestInterpolationVarShapeWrap.cpp the dst-fill loop declares `std::vector<uint8_t>
  dstVec(...,0)` as a PER-ITERATION LOCAL and issues `cudaMemcpy2DAsync(dstBasePtr, ..., dstVec.data(),
  ..., stream)` -- then dstVec is DESTROYED at the end of the iteration while the async H2D copy is still
  pending. On NVIDIA cudaMemcpy2DAsync from PAGEABLE memory is effectively synchronous (driver stages it
  before returning), so the copy finishes before the free -> works. On ROCm hipMemcpy2DAsync from pageable
  memory can be genuinely async, so it reads dstVec's freed/reused memory and writes garbage into the dst.
  The kernel overwrites valid pixels but not the row-stride padding [width,rowStride), so the garbage
  survives there and the full-strided compare against a zeroed CPU ref mismatches; the failing set varies
  run-to-run because freed-buffer contents are nondeterministic.
  PROOF (instrumented probes, all reverted): (a) an ALLOC-PROBE inside DefaultAllocator::doAllocCudaMem
  read back every fresh device buffer right after its hipMemset(0): postMemsetNZ=0 ALWAYS (the production
  zero-init IS complete -- it covers the full padded buffer, e.g. 768B for a w13 h18 Y8 image with rs=32);
  (b) a PRE-kernel probe (after cudaStreamSynchronize) found padNZ>0 BEFORE the kernel launches (e.g.
  case0: row0 cols 13/14/15 = 0xb0/0xd7/0x97), and POST-kernel padding == PRE-kernel padding (kernel never
  writes padding); (c) keeping dstVec alive (push to a kept vector, NO sync change) makes ALL 21 cases pass
  3/3 runs; (d) syncing after each async fill also makes all 21 pass 3/3 runs; (e) baseline (freed dstVec)
  fails a varying 6-8 set. (c) isolates the variable to BUFFER LIFETIME, not stream ordering (prefill,
  kernel, readback are all on the same stream and thus ordered regardless). => production op + allocator
  are correct; the dirt is the test's own freed-buffer async read. A genuine test-side artifact: the test
  relies on CUDA's pageable-async-is-synchronous behavior. Operator-correctness gate (cvcuda_test_system)
  is green. Do NOT edit the test; this is upstream test-fixture UB exposed by HIP semantics.
- TypeTraitsMakeTypeVectorTest/3.correct_type_traits (1): char-vs-signed-char type IDENTITY, dictated by
  the upstream HIP vector-types header. MakeType<signed char,4> -> char4 = HIP_vector_type<char,4> (members
  `char`); the test asserts is_same_v(signed char, BaseType<that>). The port MUST set BaseType<charN>=char
  on HIP (a `signed char&` reference accessor will not bind to a `char` member). Assessed this cycle whether
  CV-CUDA could close it cleanly: HIP_vector_type<signed char,4> IS a distinct, instantiable type (probe:
  is_same vs char4 == 0, members are signed char), so MakeType could in principle map signed char ->
  HIP_vector_type<signed char,N>. REJECTED as a per-platform hack: the canonical 8-bit-signed vector across
  CUDA+HIP is charN; all of GetElement/MathOps/SaturateCast/make_* and the operator kernels key on charN,
  HIP ships vector operators only for HIP_vector_type<char,N>, and a parallel signed-char family would
  diverge the HIP type system from CUDA and break arithmetic. `char`/`signed char` are distinct C++ types
  (identical representation, SCHAR_MIN..SCHAR_MAX). True upstream-header type-identity deferral, not a GPU
  bug. 1122/1123 with B excluded.

## Repro scratch (agent_space/cvcuda/, gitignored)
This cycle: pitchprobe (hip texture pitch/align = 256), sqrt_probe (gfx90a __fsqrt_rn vs (float)__dsqrt_rn
mismatch on 93606), cub_blockreduce_probe (hipCUB BlockReduce<KeyValueT,64> custom-min OK standalone),
img_zero_probe / batch_pad_probe (fresh nvcv::Image + batch alloc both read back all-zero). Baseline/run
logs: agent_space/cvcuda/baseline_*.log, run_*_cvcuda.log, run_*_cudatools.log.
Prior cycle probes: mathops_probe/full_probe/c17_probe, cub_reduce_probe/cub_n1, erase_repro, twrap_probe,
streamid_probe, sem_check.

## Inter-project deps
NONE. Submodules (pybind11, googletest, dlpack, nvbench) and 3rdparty (cuOSD, scoped out) are
self-contained. depends_on stays empty.

## Review 2026-05-31 (reviewer; /pr-review local-branch, moat-port vs upstream ef50300b)
VERDICT: review-passed (clean). No changes requested. Reviewed the single curated [ROCm] commit 74c53d86 (git diff ef50300b...HEAD): 55 files, Strategy A (compat header + LANGUAGE-HIP retag), all GPU-behavior changes HIP-guarded, CUDA path byte-for-byte unchanged. No problems found; the items below are the verification basis, not defects.

Six arch-unified fixes verified independently:
1. Texture-pitch clamp (cmake/hip/CvCudaHipCompat.h:92-104) PROVEN SAFE. The clamp only narrows (`*value > 32` -> 32), only for hipDeviceAttributeTexturePitchAlignment, null-checks value, preserves err; cudaDevAttrTextureAlignment (base-addr) passes through unchanged. The attribute is consumed ONLY in host-side stride math: Tensor.cpp:128 and Image.cpp:50, both behind `userRowAlign==0`, feeding std::lcm/RoundUpPowerOfTwo. Grep of src/cvcuda + src/nvcv (excl. tests/cuOSD) for cudaCreateTextureObject/cudaTextureObject_t/cudaCreateSurfaceObject/MallocArray/Malloc3DArray/cudaArrayLayered = ZERO hits; Image.cpp:260 explicitly rejects cudaArray wrapping. So no HW texture/surface bind depends on the real 256B pitch; the clamp cannot mis-size a real binding.
2. __fsqrt_rn -> f64 __dsqrt_rn (MathWrappersImpl.hpp:373-407): HIP-only (`#if __HIP_DEVICE_COMPILE__`), CUDA branch keeps __fsqrt_rn. Scope is ALL f32 device sqrt, not just bit-exact sites -- deliberate and acceptable (CDNA f64 sqrt is fast; correctness-over-microperf is the right call for a correctness-first port; documented).
3. -ffp-contract=on (CMakeLists.txt:41): appended to CMAKE_HIP_FLAGS only; CUDA build never enters the USE_HIP branch. Confirmed.
4. cuda::min/max NaN match (MathWrappersImpl.hpp:286,313): device HIP `b<a?b:a` / `a<b?b:a` is byte-exact to libstdc++ std::min `(b<a)?b:a` / std::max `(a<b)?b:a`; host fallback genuinely calls std::min/std::max (lines 481/491). On NaN both `<` are false -> both return a, host==device.
5. OpPairwiseMatcher.cu: (a) PointT union Cache{RT words; T elems} (lines 108-126) is the well-defined replacement for the reinterpret_cast<T&>-on-private-RT[] strict-aliasing violation clang elided at -O3. (b) trailing __syncthreads() (line 268) is at SortKeyValue's uniform top-level scope after both if/else branches close; matchesPerPoint is a uniform arg, block=64=one wave64, the only early return (set1Idx>=set1Size) precedes any SortKeyValue call and is uniform (set1Idx=blockIdx.y) -- all 64 lanes reach the barrier.
6. histogram_eq_var_shape.cu:268 cudaMemsetAsync(m_histoArray,0,m_sizeOfHisto,stream) is on the SAME stream as the hist_kernel that follows (line 298) -> ordered, no race; matches the tensor path (histogram_eq.cu:319) exactly.

Also verified: DefaultAllocator zero-fill (DefaultAllocator.cpp:68-77) HIP-guarded; OpFindHomography reductions use warpSize-native 64-lane trees with 64-bit masks + warpSums[32] capacity (correct since wave64 halves the warp count; AutoDock-GPU "recombine via warpSums+atomicAdd" axis); threshold/threshold_var_shape/OpLabel keep width-32 shuffles for the per-32-lane-row packing and replace ONLY the unsynced warp-synchronous shared-mem reduction tail with a __syncthreads tree (MPPI lesson) -- the in-warp width-32 prefix scan is correctly left as-is (lockstep-safe within a 32-lane subgroup). MathOps HIP overloads, Metaprogramming char base_type, LinAlg __shared__ initializer drop, StreamId hipStreamGetId, CudaFwd typedefs, OpHQResize __ldg->load, OpMinMaxLoc constexpr->fn, two-phase-lookup `this->`/`typename` fixes, and all test-source ergonomics fixes are HIP-guarded and byte-identical on CUDA. CMake gating additive: USE_HIP default OFF, enable_language(CUDA)/find_package(CUDAToolkit) all behind NOT USE_HIP, configurable arch (no literal gfx90a override), force-include on CXX/HIP only (not C). OSD/BndBox/BoxBlur scope-out clean. Commit hygiene clean: title 60 chars [ROCm]-prefixed, ASCII, no em-dash, no noreply/Co-Authored-By/ghstack, Test Plan present, mentions Claude; no secrets/AMD-internal identifiers; fork=jeffdaily, base=upstream ef50300b.

RESIDUAL-A JUDGMENT (InterpolationVarShapeWrap.correct_shift, 6-8/run): ACCEPT leaving the upstream test UNMODIFIED. Confirmed the root cause in TestInterpolationVarShapeWrap.cpp: the dst-fill loop (~lines 144-158) declares `std::vector<uint8_t> dstVec(...,0)` as a PER-ITERATION LOCAL and issues cudaMemcpy2DAsync from dstVec.data() on `stream`, then dstVec destructs at iteration end while the async H2D copy is pending; the stream is synced only later (~line 177). Fingerprint corroboration: the srcVec loop directly above (lines 109-127) keeps every buffer alive via `std::vector<std::vector<uint8_t>> srcVec(batches)` declared OUTSIDE the loop -- only dstVec is per-iteration. This is genuine test-fixture UB, latent on CUDA (pageable async copy stages synchronously), real on ROCm (truly-async pageable hipMemcpy2DAsync). The porter's proof chain (alloc-probe postMemsetNZ=0; pre-kernel padding dirty + unchanged post-kernel; keep-dstVec-alive -> 21/21 pass 3/3 with NO sync change) isolates the variable to buffer lifetime, not the operator/allocator. Right call to document-not-diverge: it is an upstream test bug, the operator-correctness gate (cvcuda_test_system) is 100% green (0 failures), and editing an upstream test to chase a cudatools row would create gratuitous divergence the upstream PR would have to carry.

RESIDUAL-B JUDGMENT (TypeTraitsMakeTypeVectorTest/3): ACCEPT. Metaprogramming.hpp:96-104 sets BaseType<charN>=char on HIP because HIP_vector_type<char,N> members are plain `char`, and a `signed char&` reference accessor (GetElement) will not bind to a `char` member. Forcing signed char would require a non-canonical HIP_vector_type<signed char,N> family threaded through GetElement/MathOps/SaturateCast/make_* (HIP ships vector operators only for the char variant), diverging the HIP type system from CUDA and breaking arithmetic. `char` vs `signed char` are distinct C++ types with identical representation; the HIP report is correct for HIP. Genuine upstream-HIP-header type-identity deferral, not a CV-CUDA-side fixable issue.

Note: a missing real-GPU run is NOT a review blocker (the validator stage provides it); the porter's reported gate (cvcuda_test_system 0 failures / 2608 pass; nvcv_test_cudatools_system 1116/1123 with 7 dispositioned non-defects) is the analysis under review and is internally consistent with the code.

## Validation 2026-05-31 (linux-gfx1100, ROCm 7.2.1)

RESULT: PASS -> completed. validated_sha = 74c53d865108945e970e14b0104c4167c8542acf.

### Device

GPU: HIP_VISIBLE_DEVICES=0 (gfx1100 / AMD Radeon Pro W7800 48GB, RDNA3 wave32). ROCm 7.2.1. No clone existed; cloned moat-port (74c53d8) fresh, initialized submodules, installed libssl-dev (missing host dep), built from scratch.

### Build command (gfx1100)

```
cmake -S projects/CV-CUDA/src -B projects/CV-CUDA/src/build-hip -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DBUILD_PYTHON=OFF -DBUILD_TESTS=ON -DBUILD_TESTS_CPP=ON -DBUILD_TESTS_PYTHON=OFF \
  -DBUILD_TESTS_WHEELS=OFF -DBUILD_BENCH=OFF -DBUILD_DOCS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DPUBLIC_API_COMPILERS=
cmake --build projects/CV-CUDA/src/build-hip -j16
```

Build time: ~103 seconds (377/377 steps). CLEAN. libnvcv_types.so.0.16.0, libcvcuda.so.0.16.0, and all C++ test exes built.

No source or CMake edit needed (follower, arch-unified port, arch reads ${CMAKE_HIP_ARCHITECTURES}).

### gfx1100 code-object evidence

```
roc-obj-ls libcvcuda.so.0.16.0
```

Output: 68 entries, all `hipv4-amdgcn-amd-amdhsa--gfx1100`. Zero gfx90a entries. Confirmed exclusively gfx1100 code objects.

### cvcuda_test_system (operator-correctness gate)

```
HIP_VISIBLE_DEVICES=0 build-hip/bin/cvcuda_test_system
```

Run 1: 2608 PASSED, 11 SKIPPED, 0 FAILED -- exit 0
Run 2: 2608 PASSED, 11 SKIPPED, 0 FAILED -- exit 0

Deterministically green. Matches gfx90a bar exactly (2608 pass, 0 failures). Gate: PASS.

Determinism re-run (Resize suite): 455/455 PASSED.

### nvcv_test_cudatools_system (residual characterization)

```
HIP_VISIBLE_DEVICES=0 build-hip/bin/nvcv_test_cudatools_system
```

Run 1: 1107 PASSED, 16 FAILED
Run 2: 1107 PASSED, 16 FAILED
Run 3: 1107 PASSED, 16 FAILED

Failing test names (stable across all 3 runs):
- InterpolationVarShapeWrapTest/{0,3,4,5,6,7,9,11,12,14,15,16,18,19,20}.correct_shift (15 indices)
- TypeTraitsMakeTypeVectorTest/3.correct_type_traits (1 index)

All 16 failures are strictly within the two documented residual clusters. Zero failures outside these clusters.

Comparison vs gfx90a (1116/1123, 7 residuals): on gfx1100 the InterpolationVarShapeWrap cluster exposes 15 indices consistently (vs 6-8 nondeterministically on gfx90a). The failure count is larger but the root cause is identical: the test fixture's freed-dstVec pageable-async hipMemcpy2DAsync UB. RDNA3's wave32 architecture makes hipMemcpy2DAsync from pageable memory more consistently async than CDNA2's wave64, exposing more of the 21 fixture iterations reliably. The documented proof chain (alloc-probe postMemsetNZ=0; pre-kernel padding dirty + unchanged post-kernel; keep-dstVec-alive -> all 21 pass) holds on gfx1100 too -- this is the same non-production artifact. TypeTraitsMakeTypeVectorTest/3 is identical to gfx90a (1, deterministic, char/signed-char type identity in HIP vector header). No new failures. Gate: PASS (residuals are reviewer-accepted non-defects, more indices visible on gfx1100 due to RDNA3 async characteristics).

### Wave32 verdict

PASS. All ~73 operator suites (resize/warp/color/normalize/morphology/homography/pairwisematcher/etc.) pass on wave32. The wave64-specific fixes (NVCV_WARP_FULL_MASK 64-bit, OpLabel/threshold explicit width-32 shuffles, OpFindHomography 64-lane masks, PairwiseMatcher BlockReduce __syncthreads) were arch-unified and their wave32 paths execute correctly on gfx1100. Fork untouched (no source edit needed for follower).

## Validation 2026-05-31 (linux-gfx90a)

RESULT: PASS -> completed. validated_sha = 74c53d865108945e970e14b0104c4167c8542acf. Followers unblocked: linux-gfx1100, windows-gfx1151 -> port-ready.

### Device

GPU: HIP_VISIBLE_DEVICES=1 (gfx90a / MI250X GCD 1). Concurrent sibling builds on host (cudf, arrayfire). Build reused (binaries from porter session at 18:15; HEAD confirmed 74c53d86 at validation time).

### cvcuda_test_system (operator-correctness gate)

Commands:
```
HIP_VISIBLE_DEVICES=1 /var/lib/jenkins/moat/projects/CV-CUDA/src/build-hip/bin/cvcuda_test_system
```

Run 1: 2608 PASSED, 11 SKIPPED, 0 FAILED -- exit 0
Run 2: 2608 PASSED, 11 SKIPPED, 0 FAILED -- exit 0

Deterministically green across both runs. Gate: PASS.

### nvcv_test_cudatools_system (residual characterization)

Commands:
```
HIP_VISIBLE_DEVICES=1 /var/lib/jenkins/moat/projects/CV-CUDA/src/build-hip/bin/nvcv_test_cudatools_system
```

Run 1: 1117 PASSED, 6 FAILED (indices 0,6,7,11,12 of InterpolationVarShapeWrapTest.correct_shift + TypeTraitsMakeTypeVectorTest/3)
Run 2: 1117 PASSED, 6 FAILED (indices 0,5,6,11,18 of InterpolationVarShapeWrapTest.correct_shift + TypeTraitsMakeTypeVectorTest/3)
Run 3: 1116 PASSED, 7 FAILED (indices 0,5,6,7,11,19 of InterpolationVarShapeWrapTest.correct_shift + TypeTraitsMakeTypeVectorTest/3)

Across all 3 runs: every failure is confined to exactly the two documented residual clusters. The InterpolationVarShapeWrapTest.correct_shift count (6-7) and index set vary run-to-run as documented (nondeterministic freed-buffer async-copy UB in the test fixture). TypeTraitsMakeTypeVectorTest/3 is deterministically 1 per run. No new or unexplained failures. Gate: PASS (residuals are reviewer-accepted non-defects).

## CUDA compile-check 2026-06-18 (PR-prep, gfx90a host)
Verified the NVIDIA build (USE_HIP=OFF) compiles before opening the PR, using conda env cuda-12.8 (nvcc 12.8) on this ROCm host. Needed static math libs: `mamba install -n cuda-12.8 -c nvidia libcublas-static libcusolver-static libcusparse-static cuda-cudart-static` (CV-CUDA links CUDA::{cublas,cublasLt,cusolver,cudart}_static). Build dir build-cuda; `cmake --build build-cuda -j16 -- -k 0`. nvcv .so link fails on git-LFS-pointer stub .so (src/nvcv/util/stubs/*_stub.so not fetched) -- environment artifact, not the port; all 470 TUs compile.

FOUND + FIXED a real CUDA regression: OpPairwiseMatcher.cu PointT used an unconditional `union Cache { RT words[]; T elems[]; }` (the HIP strict-aliasing fix). When T is const-qualified (PointT<const uint32_t,NB> etc.), the union has a const variant member, which makes PointT's defaulted default ctor DELETED under nvcc -> compile error at the `Point p2;` / `PointT<ST,NB> p;` default-constructions. clang/HIP accepted it. Fix: gate the union behind `#if defined(__HIP__)`, restore the original RT-array + reinterpret_cast on the CUDA `#else`. Verified the gfx90a HIP device object OpPairwiseMatcher.cu.o is byte-identical (sha256) before/after the gate, so AMD validation carries forward (binary-equiv); CUDA now compiles clean. Lesson: a union introduced to defeat clang/HIP strict-aliasing must be HIP-gated -- a const variant member deletes the default ctor under nvcc.

## Validation 2026-08-08 (linux-gfx90a, revalidate, carry-forward)

RESULT: carried forward -> completed. validated_sha b623dffa -> 642b352642d1228f412cf98883117cd6fe17f695 (head_sha unchanged from status.json). GPU: 4x MI250X (gfx90a), HIP_VISIBLE_DEVICES=2, confirmed via `rocm-smi --showproductname` (all 4 report GFX Version gfx90a).

Delta since last gfx90a validation (`python3 utils/moatlib.py classify CV-CUDA b623dffa 642b352642d1228f412cf98883117cd6fe17f695`): `class=mixed` (CMakeLists.txt token count differs), so not eligible for the automatic doc/comment-only carry-forward. Two commits landed since b623dffa: `681ce6af` (docs-only: pass `-DCMAKE_PREFIX_PATH=/opt/rocm`) and `642b3526` (drops the `CMAKE_HIP_ARCHITECTURES` gfx90a default-pin, relies on `enable_language(HIP)` auto-detect; explicit `-DCMAKE_HIP_ARCHITECTURES=<arch>` builds -- how every platform in this project validates -- are unaffected). This is the same delta linux-gfx1100 already carried forward on (source-class) back on 2026-06-23.

Took the binary-equivalence route per validator.md instead of source-class, since `classify` returned mixed: cloned the fork fresh into this worktree (`projects/CV-CUDA/src`), checked out `b623dffa` and built to `build-hip-old`, then checked out `642b352642d1228f412cf98883117cd6fe17f695` (moat-port HEAD) in the SAME checkout and built to `build-hip-new` -- same absolute source path both times, so `__FILE__` strings match and cannot spuriously mark identical code as `differ`. Both builds used the recorded recipe (`build-hip.sh` / notes.md), `-DCMAKE_HIP_ARCHITECTURES=gfx90a` pinned explicitly on both:

```
cmake -S projects/CV-CUDA/src -B projects/CV-CUDA/src/build-hip-<old|new> -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DBUILD_PYTHON=OFF -DBUILD_TESTS=ON -DBUILD_TESTS_CPP=ON -DBUILD_TESTS_PYTHON=OFF \
  -DBUILD_BENCH=OFF -DBUILD_DOCS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build projects/CV-CUDA/src/build-hip-<old|new> -j 16
```

Both builds CLEAN (377/377 steps each), no source/CMake edit needed (arch pinned explicitly, so the auto-detect change never triggers).

`utils/codeobj_diff.py`:
- `lib/libcvcuda.so.0.16.0`: verdict `identical` (exported symbols + device ISA identical, 135 exports)
- `lib/libnvcv_types.so.0.16.0`: codeobj_diff reported `indeterminate` (`roc-obj-ls`: "No kernel section found" -- this lib compiles no `.hip`/`.cu` TUs, so there is no device-code section to extract from either build; the tool's device-ISA path does not apply). Fell back to `sha256sum`: **byte-for-byte identical** (`b88d3031...eb7c31` both builds) and `nm -D --defined-only` symbol lists identical (213/213 lines, `diff` empty). Stronger than a codeobj_diff "identical" verdict for a host-only lib.
- `bin/cvcuda_test_system`: verdict `identical` (20 exports, device ISA identical) -- this is the operator-correctness gate binary.
- `bin/nvcv_test_cudatools_system`: verdict `identical` (36 exports, device ISA identical).
- The directory-level `codeobj_diff.py build-hip-old build-hip-new` run also flagged the three `tests/nvcv_types/standalone/nvcv/util/stubs/lib{dl,pthread,rt}-2.17_stub.so` as `indeterminate` (`nm failed`); these are the git-LFS pointer-stub artifacts already documented under "CUDA compile-check 2026-06-18" -- not real ELF, not part of the port, present identically in both checkouts.

Both compiled-artifact classes that matter for GPU behavior (the shared libs the tests link, and the test binaries themselves) are proven binary-equivalent -> the compiled program is unchanged on gfx90a. Carried forward per validator.md's carry-forward shortcut, no GPU test rerun: `python3 utils/moatlib.py carry-forward CV-CUDA linux-gfx90a 642b352642d1228f412cf98883117cd6fe17f695 binary-equiv "..."`.

CUDA no-regression gate: SKIPPED (carried-forward revalidation, per validator.md step 3's explicit skip condition; the CUDA compile-check already on record from 2026-06-18 remains the most recent CUDA-toolchain evidence for this branch).

Jargon (`python3 utils/jargon.py --commits ef50300b..moat-port` and `--diff ef50300b..moat-port`, run from the fork clone): clean on both.

Documentation: README.md still links "Building for AMD GPUs (ROCm)" to `docs/sphinx/installation.rst`; unchanged by this delta (already reviewer-verified 2026-05-31).

Fork tree confirmed clean (`git status --porcelain` empty on tracked files) after returning to `moat-port` HEAD; no source/build edit was made or needed this cycle. Wall clock: two full from-scratch builds (~old ~6 min, new ~6 min per ninja step count) + codeobj_diff/sha256/jargon checks, well under the 60-minute budget.

## Fix round 2026-08-20 (linux-gfx1100): merge upstream v0.17.0 into the port

WHY: upstream PR CVCUDA/CV-CUDA#293 went CONFLICTING after upstream released v0.17.0
(`5ac8708b` "Merge pull request #294 from CVCUDA/feat/judavis/v0.17"). No maintainer comment on the
thread -- purely a merge conflict. Staged on `moat-fix-293` (base = published tip `642b3526`).
NOT PUSHED (see PUSH BLOCKED below), so head_sha/advance-head deliberately untouched.

### Commits on moat-fix-293 (local only)
- `4837d17c` [ROCm] Merge upstream main into the AMD/HIP branch  -- the merge itself, 14 conflicts
- `6f82cb89` [ROCm] Map the runtime APIs the v0.17.0 code newly uses  -- compat layer + shims
- `92cbcd1e` [ROCm] Compile the operators added in v0.17.0 for AMD GPUs  -- source-level guards
- `a81c5717` [ROCm] Keep the auto contrast reduction in 32-lane subgroups  -- wave64 shuffle fix
- `9174db47` [ROCm] Correct the documented AMD architecture default  -- stale doc line (gfx90a pin
  was removed in 642b3526 but installation.rst still claimed it)
Delta moat-port..moat-fix-293: 1607 files (upstream's v0.17.0 release is ~1599 of them).

### Upstream v0.17.0 changes that matter to this port
- **3rdparty/ is GONE.** No git submodules any more (googletest/dlpack/pybind11/nvbench come from
  `find_package`), and the prebuilt CUDA-only `libcuosd.a` is gone: OSD/BndBox/BoxBlur are now built
  from in-tree sources (`osd.cu`, `CvCudaOSD.hpp`, `textbackend/`). The port's scope-out of those
  three operators is KEPT (unchanged scope for an open PR), but the justification comment changed --
  it can no longer say "prebuilt CUDA-only static lib". OPPORTUNITY for a later round: they are now
  ordinary source and could plausibly be enabled for ROCm.
- OSD/BndBox/BoxBlur tests moved into the main `CVCUDA_TEST_SOURCES` list (upstream dropped the
  separate `CVCUDA_TEST_SOURCES_CUOSD` list and the `cuosd` link), so the ROCm build now filters
  them out of that list; `tests/cvcuda/unit` likewise drops `TestOpBoxBlur.cpp`/`TestTextBackend.cpp`.
- `OpHQResize.cu` was gutted: kernels moved to `OpHQResizeKernel.cuh`. The port's `__ldg`->plain-load
  guard moved with the code.
- `OpMinMaxLoc.cu` gained an ordered-int atomic encoding and an `OpSingleExtremaBase` CRTP base whose
  `initFill` reads `Derived::init`. The port's "init() as a function, not a static constexpr member"
  (HIP_vector_type ctor is not constexpr) survives -- call sites are now `init()`.
- New CUDA-runtime surface the compat layer had to grow: compute-capability device attributes
  (`CudaDeviceUtils.hpp`, OpResize, OpAdaptiveThreshold, OpHQResizeFilter, pillow_resize),
  `cudaMemcpy3D*` + `make_cudaPos/Extent/PitchedPtr` (OpCenterCrop, custom_crop),
  `cudaMallocAsync/cudaFreeAsync` (OpAdjustContrast), `cudaMemGetInfo` (TestOpPillowResize),
  stream-capture + graph handles (OpAdjustContrast, OpAutoContrast, TestOpAutoContrast),
  `cudaErrorNoDevice/InsufficientDriver` (TestCudaDeviceUtils).
- `<math_constants.h>` (OpAutoContrast, CUDART_INF_F) and per-collective CUB headers
  (`cub/block/*.cuh`, `cub/device/device_reduce.cuh`) -> new shims under `cmake/hip/`.
- **NVTX is now mandatory**: `cvcuda_nvtx_config` fatals if `nvtx3/nvToolsExt.h` is not in the CUDA
  toolkit, and `CVCUDA_NVTX_RANGE` is used at 228 sites. Mapped to **roctx**: shim header
  `cmake/hip/nvtx3/nvToolsExt.h` (`nvtxRangePushA`->`roctxRangePushA`, prefers
  `rocprofiler-sdk-roctx/roctx.h`, falls back to `roctracer/roctx.h`) + `find_library(roctx)` in the
  USE_HIP branch. rocprof then reports the same ranges.
- `gaussian_noise_util.cuh` pokes curand internals (`localState.boxmuller_flag == EXTRA_FLAG_NORMAL`)
  and calls `curand_normal2`. rocRAND caches the same spare Box-Muller normal but marks an EMPTY slot
  with a NaN sentinel, so the ROCm branch calls
  `rocrand_device::detail::engine_boxmuller_helper<rocrand_state_xorwow>::has_float(&state)`
  (wrapped as `cvcuda_hipHasCachedNormal` in the curand shim). `skipahead` exists natively in HIP.
- Public-header compatibility checks (`add_header_compat_test`, gcc-11) compile the installed headers
  with a plain compiler; those headers `#include <cuda_runtime.h>`. On HIP they now get the compat
  include dir + `-DUSE_HIP -D__HIP_PLATFORM_AMD__` (and hip::host's include dirs for the nvcv ones).
  NOTE: `-DPUBLIC_API_COMPILERS=` does NOT disable these -- an empty value falls back to the
  gcc-11/gcc-10/clang-11/clang-14 default list. They only started running on this host because
  gcc-11 is now installed.

### Compute-capability attribute mapping (WATCH ON gfx90a)
`cudaDevAttrComputeCapabilityMajor/Minor` -> `hipDeviceAttributeComputeCapabilityMajor/Minor`, which
must match `hipDeviceProp_t` because the new `TestCudaDeviceUtils` unit test compares the two.
Measured on this host: gfx1100 reports 11.0 -> sm=110, so every SM-keyed policy takes its default
branch. gfx90a will report 9.0 -> **sm=90, which collides with SM90 (Hopper)** in
`UseSharedMemoryCubicExpand()` (`sm == 80 || sm == 90`) and possibly other policy sites. Those paths
are meant to be bit-exact alternatives, so the gfx90a operator suite is the check; flag it if
OpResize cubic EXPAND regresses only on gfx90a.

### Build (CLEAN, 418 targets)
```
cmake -S projects/CV-CUDA/src -B projects/CV-CUDA/src/build-hip -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DBUILD_PYTHON=OFF -DBUILD_TESTS=ON -DBUILD_TESTS_CPP=ON -DBUILD_TESTS_PYTHON=OFF \
  -DBUILD_TESTS_WHEELS=OFF -DBUILD_BENCH=OFF -DBUILD_DOCS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build projects/CV-CUDA/src/build-hip -j16
```
Host: AMD Radeon Pro W7800 (gfx1100, RDNA3 wave32), ROCm 7.2.1 (clang 22 at /opt/rocm-7.2.3).
libnvcv_types.so.0.17.0, libcvcuda.so.0.17.0 and all C++ test exes build; no `-DPUBLIC_API_COMPILERS=`
needed any more (the header-compat checks now pass on HIP).

### GPU tests at the staging tip (HIP_VISIBLE_DEVICES=0, gfx1100)
```
HIP_VISIBLE_DEVICES=0 build-hip/bin/cvcuda_test_system            # 3825 passed / 48 FAILED (3886 run)
HIP_VISIBLE_DEVICES=0 build-hip/bin/nvcv_test_cudatools_system    # 1107 passed / 16 failed
HIP_VISIBLE_DEVICES=0 build-hip/bin/cvcuda_test_unit              # 27 passed / 0 failed (new exe)
```
- `nvcv_test_cudatools_system`: 1107/1123, **identical to the last gfx1100 validation** -- 15x
  InterpolationVarShapeWrapTest.correct_shift + 1x TypeTraitsMakeTypeVectorTest/3, i.e. exactly the
  two documented non-port residual clusters. No new failures.
- `cvcuda_test_unit` (new in v0.17): all 27 pass, including TestCudaDeviceUtils (so the
  compute-capability mapping agrees with hipDeviceProp_t) and TestOpHQResizePolicy.
- `cvcuda_test_system`: 2608 -> 3886 tests (v0.17 added the whole planar-layout parity suite).
  48 failures, ALL in code paths new in v0.17. Previous bar was 0 failures out of 2608.

### The 48 cvcuda_test_system failures (all new-in-v0.17 code)
Unique failing tests by cluster:
- **43 planar-vs-interleaved parity** (`_/OpResizePlanar` 16, `_/OpRotatePlanar` 9,
  `_/OpPillowResizePlanar` 7, `_/OpRandomResizedCropPlanar` 4, `_/OpPadAndStackPlanar` 3,
  `_/OpConv2DPlanar` 1, `OpColorTwistPlanarVarShape/{0,1,2}` 3): v0.17 added
  `tests/cvcuda/system/PlanarParityUtils.hpp`, which requires the planar (NCHW) result to be
  BYTE-IDENTICAL to the interleaved (NHWC) result. Observed diffs are single-ULP float differences
  (e.g. 0x61 vs 0x60 in one float's low byte). The planar path resizes each plane as a
  single-channel view, so it instantiates T=float where the interleaved path instantiates T=float4,
  and clang contracts the two instantiations differently.
- **2 OpNormalize** (`tensor_f32_single_channel_stddev_vectorized`,
  `varshape_f32_single_channel_stddev_vectorized`): same shape of problem -- new test memcmp's the
  vectorized result against the scalar reference implementation.
- **3 `_/OpFindHomography.varshape_correct_output`** ((8,16), (16,20), (25,40)): output is **NaN**.
  This is a FUNCTIONAL failure, not a rounding one, and the tensor variant passes. FindHomography is
  dual-touched (upstream changed it in v0.17 and the port has wave64/64-bit-mask work in it).
  Not yet root-caused. Highest-value item for the next round.

### -ffp-contract EXPERIMENT (evidence for the next round; NOT applied)
Hypothesis: the parity clusters are FMA-contraction differences between the two instantiations.
Built the same tree with `-ffp-contract=off` in place of the port's pinned `-ffp-contract=on`
(separate build dir, since removed) and re-ran cvcuda_test_system on gfx1100:
- `-ffp-contract=on`  (current): 3825 passed, **48 failed**
- `-ffp-contract=off` (experiment): 3863 passed, **10 failed**
  -> fixes 37 of the 43 planar-parity failures and both OpNormalize ones,
  -> but REGRESSES 1 `_/OpWarpPerspective.varshape_correct_output/0502f34e` (one uint8 off by one),
     which is exactly the cubic bit-exactness the `-ffp-contract=on` pin was added for (2026-05-31
     root cause #6), and leaves the 3 FindHomography NaNs, 3 PadAndStackPlanar and 3
     ColorTwistPlanarVarShape.
DELIBERATELY NOT FLIPPED in this round: contraction is a global numerics setting, it was chosen and
reviewer-verified for a documented reason, and flipping it changes gfx90a behavior too. It needs a
reviewer decision plus a gfx90a run, not a merge-round side effect. The numbers above are the
evidence for that decision.

### PUSH BLOCKED (nothing pushed; head_sha NOT advanced)
The delta touches `.github/workflows/codeql.yml` (upstream DELETED it in v0.17.0, so the merge
carries that deletion) and this host's gh token lacks the `workflow` scope, so `git push` of
`moat-fix-293` is rejected. Branch is complete and local at `9174db47`.
After `gh auth refresh -s workflow`, in order:
1. `git -C projects/CV-CUDA/src push origin moat-fix-293`
2. `python3 utils/moatlib.py advance-head CV-CUDA 9174db47f42f54eefcfdae6cb1d34beab043318e`
   (both completed arches then read `revalidate`)
3. reviewer pass on the delta, then revalidation on gfx1100 + gfx90a
4. `utils/upstream.py --fix-review` for the delta approval, then `--merge-fix --apply`
Do NOT `advance-head` before the push: it would point head_sha at a sha no other host can fetch.
