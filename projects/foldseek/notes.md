# foldseek notes

Fork: https://github.com/AMD-Ecosystem/foldseek (branch moat-port)
Default branch: master (NOT main).
Port head: e7471b4164e38cbac58b4f2c6c1b592e9bfac330

Scope: the GPU structure-search aligner (`--gpu`), which is the vendored
libmarv (CUDASW++4.0) at `lib/mmseqs/lib/libmarv`. Strategy A (pure CMake +
single compat header). ProstT5's ggml-CUDA LLM backend is OUT OF SCOPE and
stays off (USE_HIP does not turn on GGML_CUDA).

libmarv is the SAME aligner MMseqs2 vendors -- this port and the MMseqs2 port
must converge. See "Cross-reference: MMseqs2" below.

## Build recipe (gfx90a, ROCm 7.2.1)
```
cmake -S projects/foldseek/src -B projects/foldseek/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build projects/foldseek/build --target foldseek -j16
```
- `USE_HIP` is a new cache option threaded top-level -> lib/mmseqs -> libmarv.
  It drives the libmarv subdir (parallel to `ENABLE_CUDA`) and the
  `HAVE_CUDA=1` define + marv link in mmseqs/foldseek, WITHOUT enabling the
  ProstT5 ggml-CUDA path (that one is still keyed off `ENABLE_CUDA`).
- libmarv builds as a SHARED library on HIP so the `-fgpu-rdc` device link
  runs inside the HIP toolchain (clang `--hip-link`); the rest of the project
  then links `libmarv.so` as an ordinary host shared object. A static libmarv
  left the device code unlinked and g++ (the project's host linker) cannot do
  the amdgcn device link -> relocation/`R_X86_64_32` PIE failure. SHARED +
  POSITION_INDEPENDENT_CODE fixes it.
- Force-include `cuda_to_hip.h` and `-I hip_compat` are added only on the HIP
  compile via `target_compile_options($<COMPILE_LANGUAGE:HIP>...)`.

## Compat surface (all under lib/mmseqs/lib/libmarv/src)
- `cuda_to_hip.h` (NEW, force-included on HIP): cudaXxx->hipXxx runtime/event/
  stream/mempool/IPC aliases; `__grid_constant__`->nothing;
  `cudaFuncSetAttribute(func,...)`->`hipFuncSetAttribute((const void*)func,...)`;
  `THRUST_DEVICE_PAR_NOSYNC`->`thrust::hip::par_nosync`; a `cub::SwitchDevice`
  RAII shim (hipCUB has none); `WARP_FULL_MASK`=64-bit + `WarpMaskT`; short2/
  int2/float2 `__shfl*` overloads (HIP covers scalars and __half2 only);
  `__hmax2`/`__hmin2` pairwise emulation; and scalar emulations of the DPX
  intrinsics (`__vadd2`, `__vmaxs2`, `__vibmax_u16x2`, `__vimax3_s16x2[_relu]`,
  `__viaddmax_s16x2[_relu]`, `__vibmax_s32`, `__vimax3_s32`,
  `__viaddmax_s32[_relu]`).
- `hip_compat/cooperative_groups.h` + `cooperative_groups/reduce.h` (NEW):
  HIP cg has tiled_partition/thread_block_tile with shfl_* but no free
  `reduce()` and no `greater/less/plus`. Shim adds them; `reduce()` is a
  butterfly all-reduce (`shfl_xor`) that works for any commutative+associative
  op (the int3 endpoint-pack lambda and `greater<half2/float/int>`), and a
  `tile_shfl_xor` that relays arbitrary trivially-copyable T as 32-bit words.
- `hip_compat/cuda_fp16.h` (NEW): `<cuda_fp16.h>` -> `<hip/hip_fp16.h>`.

## DPX strategy (THE key item -- matches the plan)
- libmarv selects DPX vs portable at runtime from a per-device kernel config.
  TRAP: on ROCm gfx90a `hipDeviceAttributeComputeCapabilityMajor` returns
  **9** (== Hopper sm90), so the unmodified selector picks the sm90 config
  whose entries set `dpx=1` -> it would launch the NVIDIA-only DPX short2/int
  kernels. FIX: under `__HIPCC__`, `getOptimalKernelConfigs_gapless/_SW` and
  the benchmark `supportsDPX` are forced to the portable default
  (sm89 config, `dpx=0`). So AMD always runs the half2 gapless + float SW
  SIMT path. (The user-supplied tuning-config-file override that can force
  dpx=1 is an edge case not on the default foldseek path; the runtime
  `if(!config.dpx)` dispatch still selects half2/float there.)
- The DPX `*_instantiation_dpx*.cu` and `*_smithwaterman_instantiation_dpx.cu`
  TUs are KEPT in the HIP build (their kernel templates are referenced via
  `extern template` from the dispatcher, so excluding them would leave
  unresolved device symbols at link). They compile against the scalar DPX
  emulations in cuda_to_hip.h and are simply never launched.
- `__hmax2`/`__hmin2`: ROCm 7.2.x has scalar `__hmax`/`__hmin` and `__hadd2`
  but not the packed max/min; emulated pairwise in cuda_to_hip.h for the
  portable half2 path.

## Wave-size handling (gfx90a wave64)
- PSSM gapless + Smith-Waterman kernels reduce only within
  `cg::tiled_partition<N>` tiles, N in {4,8,16,32} <= 32: wave-agnostic.
- `getPaddedQueryLength` pads the query layout by `sizeof(char4)*32`. This is
  a host+device-SHARED serialized/in-memory FORMAT constant (PORTING_GUIDE
  warp-size-dependent-format class). PINNED at the literal 32 (NOT warpSize),
  so the host-built layout matches device reads on wave64. Left untouched.
- `kernelhelpers.cuh::warp_max_reduce_broadcast` (legacy `kernels.cuh` path,
  not the foldseek `--gpu` path) made warp-width-generic: `__reduce_max_sync`
  guard tightened to `defined(__CUDA_ARCH__) && >= 800`, shuffle reduction
  start changed from literal 16 to `warpSize/2`. Its `0xFFFFFFFF` masks (and
  the kernels.cuh shfl masks) widened to 64-bit `WARP_FULL_MASK`/`WarpMaskT`
  because HIP `__shfl_*_sync` static_assert require a 64-bit mask on wave64.
- Inline PTX `%laneid` (`cuda_helpers.cuh::lane_id`) given a `__HIPCC__`
  branch using `__lane_id()`. (lane_id is actually unused by libmarv.)

## Other guards
- `#ifdef __NVCC__`/`#ifdef __CUDACC__` whole-file/qualifier guards across
  hpc_helpers (cuda_raiiwrappers, simple_allocation, peer_access,
  utility_kernels, custom_thrust_allocators, nvtx_markers, timers), blosum.cu,
  blosum.hpp `extern __constant__`, and convert.cuh qualifiers were widened to
  `defined(__CUDACC__) || defined(__HIPCC__)`. EXCEPTION: the uint64
  atomicCAS/Add/Exch/Min/Max/And/Or + ffs overloads in cuda_helpers.cuh stay
  NVCC-only -- HIP provides those natively and the library's versions collided.
- `convert.cuh::ClampToInvalid::operator()(uint)` device path: ROCm has no
  `__vminu4`; added a scalar per-byte clamp under `__HIP_DEVICE_COMPILE__`
  (the `#else` host path used `std::memcpy`, not callable on device).
- NVTX is compiled out via the pre-existing `NO_NVTOOLSEXT` target define;
  `nvtx::ScopedRange` becomes a no-op stub on HIP.
- `kernels.cuh`/`pssm*.cuh` are headers: marked `HEADER_FILE_ONLY` on HIP so
  they are NOT compiled standalone (doing so duplicated the `__constant__
  constantQuery4` device global and broke the device link). Only the `.cu`
  TUs are LANGUAGE HIP.

## Validation (real gfx90a, HIP_VISIBLE_DEVICES=1, ZERO downloads)
Inputs: bundled `example/` (28 SCOP structures). CPU mode is the oracle.
```
foldseek createdb example/ exDB
foldseek search exDB exDB aln_cpu tmp_cpu --gpu 0 -e 10      # CPU oracle
foldseek makepaddedseqdb exDB exDB_pad                       # GPU needs padded DB
foldseek search exDB exDB_pad aln_gpu tmp_gpu --gpu 1 -e 10  # libmarv half2/float path
foldseek convertalis exDB exDB     aln_cpu cpu.m8
foldseek convertalis exDB exDB_pad aln_gpu gpu.m8
```
RESULT: all 834 CPU hits reproduced on GPU with byte-identical pident /
alnlen / query+target coords / e-value / bitscore (0 differing rows on the 834
common (query,target) pairs, 0 CPU-only pairs). GPU additionally returns 38
borderline low-bitscore pairs (mostly bits 1-10) -- the known GPU-vs-CPU
ungapped-prefilter sensitivity difference, NOT a numeric/wave bug. GPU result
is a strict superset; every shared alignment is exact. Confirms the wave64
half2/float SIMT path is correct.

The GPU search log shows `ungappedprefilter ... --gpu 1` (the libmarv GPU
prefilter) ran, feeding `structurealign`.

The foldseek/MMseqs2 `regression/` submodules pull external data (egress-bound)
and are deferred; the bundled example cross-check is the gate. The `--gpu 0`
CPU run doubles as the non-GPU regression check (unchanged behavior).

## Cross-reference: MMseqs2 (sibling port, same libmarv)
At the time of this port the MMseqs2 sibling notes were still empty (planner
only). This foldseek port establishes the libmarv HIP approach; MMseqs2 should
reuse `cuda_to_hip.h` + `hip_compat/` verbatim and the same DPX-forced-off /
__hmax2-emulation / wave-mask widening / getPaddedQueryLength-pinned-at-32 /
shared-lib-device-link decisions. The only foldseek-specific wiring is the
top-level/foldseek CMake gating; the libmarv changes are identical and should
land identically in both forks.

## Follower platforms (gfx1100 / gfx1151, wave32)
Reuse this moat-port branch and validate first. Wave32 is the natural fit for
libmarv's 32-lane logic. The fixes here are wave-agnostic: getPaddedQueryLength
stays at 32 on every arch; WARP_FULL_MASK/WarpMaskT auto-narrow to 32-bit on
CUDA but on HIP wave32 the 64-bit mask still works (HIP narrows it). Cross-arch
result diff: gfx1100 GPU hits/scores must match gfx90a GPU and CPU on example/.

## Validation 2026-06-04 (linux-gfx90a, HIP_VISIBLE_DEVICES=2)

Build: incremental cmake --build --target foldseek -j16. libmarv.so confirmed to contain gfx90a
device code objects (hipv4-amdgcn-amd-amdhsa--gfx90a, 45 MB). Fork HEAD e7471b4 verified.

GPU arch: gfx90a (MI250X GCD, 104 CUs). ROCm 7.2.1.

GPU vs CPU validation (bundled example/, 28 SCOP structures):
```
foldseek createdb example/ exDB
foldseek search exDB exDB aln_cpu tmp_cpu --gpu 0 -e 10
foldseek makepaddedseqdb exDB exDB_pad
HIP_VISIBLE_DEVICES=2 foldseek search exDB exDB_pad aln_gpu tmp_gpu --gpu 1 -e 10
foldseek convertalis exDB exDB aln_cpu cpu.m8
foldseek convertalis exDB exDB_pad aln_gpu gpu.m8
```

Results:
- CPU hits: 834, GPU hits: 872
- CPU-only pairs: 0 (all 834 CPU hits reproduced on GPU)
- GPU-only pairs: 38 (borderline low-bitscore, bits 1-113, expected ungapped-prefilter sensitivity)
- Common pairs with identical fields: 834/834 (0 mismatches, byte-identical pident/alnlen/coords/evalue/bitscore)

GPU search log confirmed `ungappedprefilter ... --gpu 1` (libmarv GPU prefilter) ran.
All 834 CPU hits byte-identical on GPU. GPU result is strict superset. PASS.

State: linux-gfx90a -> completed, validated_sha = e7471b4164e38cbac58b4f2c6c1b592e9bfac330.

## Review 2026-06-04 (reviewer, /pr-review local-branch mode)
Verdict: review-passed. Reviewed `git diff 718d4217...e7471b41` on moat-port
(single commit). No correctness defects, no blocking findings. The GPU-vs-CPU
score-exact result is documented; the validator runs the real GPU gate next.

Verified sound (no action):
- DPX CC==9 forcing. gfx90a reports hipDeviceAttributeComputeCapabilityMajor==9
  (collides with NVIDIA sm90). The `__HIPCC__` branch in
  getOptimalKernelConfigs_gapless (gapless_kernel_config.cuh:345) and
  getOptimalKernelConfigs_SW (smithwaterman_kernel_config.cuh:301) returns the
  `_default()` config == `_sm89()`, whose every entry has the dpx field = 0
  (confirmed: gapless sm89 4th column all 0; SW sm89 4th column all 0). The
  runtime dispatch `if(!config.dpx)` (cudasw4.cuh:1932/1945/1962/1975) then
  selects the half2/float SIMT kernels. benchmarking.cuh:1007 supportsDPX is
  forced false under __HIPCC__. So NO DPX kernel is reachable at runtime on AMD.
  The DPX *_instantiation_dpx*.cu TUs are KEPT as LANGUAGE HIP (referenced by the
  dispatcher's extern templates; excluding them breaks the device link) and
  compile against the scalar DPX emulations in cuda_to_hip.h -- dead but valid
  device code. The score-exact GPU-vs-CPU result confirms the portable path
  executes. CONFIRMED correct; merits PORTING_GUIDE promotion (see below).
- __hmax2/__hmin2 pairwise emulation (cuda_to_hip.h) via scalar __hmax/__hmin on
  low/high halves. Correct for finite integer-valued alignment scores; the
  executed gapless reduce (mathops.cuh:123 reduce_max) uses it. Byte-identical.
- getPaddedQueryLength (kernelhelpers.cuh:9) PINNED at literal `sizeof(char4)*32`
  (NOT warpSize). Host+device-shared serialized query-layout constant; pinning
  it keeps the host-built layout matching device reads on wave64. Correct per the
  PORTING_GUIDE warp-size-dependent-FORMAT class (dietgpu precedent).
- wave64 lane masks: WARP_FULL_MASK = 0xFFFFFFFFFFFFFFFFull / WarpMaskT =
  unsigned long long on HIP (cuda_to_hip.h), defaulted to 32-bit on CUDA
  (hpc_helpers.h:36). warp_max_reduce_broadcast signature widened to WarpMaskT,
  reduction start changed from literal 16 -> warpSize/2, __reduce_max_sync guard
  tightened to `defined(__CUDA_ARCH__) && >= 800` (kernelhelpers.cuh:25). All
  kernels.cuh shuffle masks widened to WARP_FULL_MASK. Executed half2/float
  reductions use width=group_size (<=32) sub-warp shuffles; the cg::reduce tiles
  are tiled_partition<groupsize> with groupsize in {4,8,16,32} (confirmed from
  the FOR_EACH_VALID_CONFIG macros) -- wave-agnostic, tile-relative on wave64.
- cooperative_groups shim (hip_compat/): reduce() butterfly all-reduce over the
  tile via shfl_xor, greater/less/plus operators, tile_shfl_xor relaying T as
  32-bit words (int3=12B=3 words exact, half2=4B=1 word). Matches CUDA cg::reduce
  for the commutative+associative max/add ops used. reduce.h forwards to it.
- __CUDACC__/__NVCC__ whole-file guards widened to also accept __HIPCC__ across
  hpc_helpers, blosum.cu/.hpp, convert.cuh -- correct. The uint64 atomic + ffs
  overloads in cuda_helpers.cuh kept NVCC-only (HIP provides them natively;
  widening would collide). lane_id() __HIPCC__ branch uses __lane_id(). ClampToInvalid
  uint path: __vminu4 -> scalar per-byte clamp under __HIP_DEVICE_COMPILE__.
- Build: USE_HIP gate (default OFF) threaded top-level -> mmseqs -> libmarv;
  enable_language(HIP), .cu marked LANGUAGE HIP, headers HEADER_FILE_ONLY,
  -fgpu-rdc + --hip-link device link inside a SHARED libmarv, force-include
  cuda_to_hip.h + -Ihip_compat only on HIP. CUDA path byte-identical (else
  branches unchanged). find_package(CUDAToolkit) stays inside NOT LIBRARY_ONLY
  (executables not built on the foldseek path) so no CUDA dep leaks to the HIP
  build. rocThrust/hipCUB resolved via the HIP toolchain include path.
- Commit hygiene: title `[ROCm] Add HIP/AMD GPU support for the libmarv structure
  aligner` (<=72), mentions Claude, Test Plan present, no noreply trailer, no
  jargon/em-dash/internal-account refs in the diff. RAII wrappers already
  move-only with guarded dtor (unchanged).

PORTING_GUIDE lesson to promote (CC-major collision): on ROCm,
hipDeviceAttributeComputeCapabilityMajor returns 9 on gfx90a (and other CDNA),
COLLIDING with NVIDIA Hopper sm90. Any CUDA per-device selector that switches on
compute-capability major to pick a Hopper-only codepath (DPX, wgmma, TMA, etc.)
will wrongly select it on AMD. Force the portable/default path under __HIPCC__
rather than relying on the unrecognized-device fallthrough (the collision means
there is NO fallthrough -- it matches sm90 exactly). Keep the Hopper-only TUs
compiling (extern-template device-link reachability) against scalar emulations
but never launch them.

Cross-reference MMseqs2 (same vendored libmarv): this port establishes the
approach. MMseqs2 should reuse cuda_to_hip.h + hip_compat/ verbatim and the same
DPX-forced-off / __hmax2-emulation / wave-mask-widening / pad-32 /
shared-lib-device-link decisions. No divergence flagged here; the only
foldseek-specific wiring is the top-level/src CMake gating. Reviewer to re-check
parity when the MMseqs2 fork lands.

## Validation 2026-06-04 (linux-gfx1100, RDNA3 wave32)

Build: cmake configure + cmake --build --target foldseek -j16 for gfx1100.
libmarv.so confirmed to contain gfx1100 code objects
(hipv4-amdgcn-amd-amdhsa--gfx1100, via llvm-objdump --offloading). Fork HEAD
e7471b4 verified on branch moat-port.

GPU arch: gfx1100 (AMD Radeon Pro W7800 48GB). warpSize=32 confirmed via
hipDeviceProp_t. ROCm 7.2.1. GPU[3] used (GPU[0] had orphaned KFD contexts
blocking queue creation; GPU[3] was responsive).

Wave32 confirmation: getPaddedQueryLength pins the layout at sizeof(char4)*32
(literal 32, not warpSize), so host-built layout matches device reads on wave32.
WARP_FULL_MASK/WarpMaskT are 64-bit on HIP; HIP narrows the mask for wave32
automatically. The PSSM/SW kernels reduce within cg::tiled_partition<N> tiles
(N<=32), wave-agnostic. All confirmed wave32-safe.

GPU vs CPU validation (bundled example/, 28 SCOP structures):
```
foldseek createdb example/ exDB
foldseek search exDB exDB aln_cpu tmp_cpu --gpu 0 -e 10
foldseek makepaddedseqdb exDB exDB_pad
CUDA_VISIBLE_DEVICES=3 HIP_VISIBLE_DEVICES=3 foldseek search exDB exDB_pad aln_gpu tmp_gpu --gpu 1 -e 10
foldseek convertalis exDB exDB aln_cpu cpu.m8
foldseek convertalis exDB exDB_pad aln_gpu gpu.m8
```

Results:
- CPU hits: 834, GPU hits: 872
- CPU-only pairs: 0 (all 834 CPU hits reproduced on GPU)
- GPU-only pairs: 38 (borderline low-bitscore, expected ungapped-prefilter sensitivity difference)
- Common pairs with identical fields: 834/834 (0 mismatches, byte-identical pident/alnlen/coords/evalue/bitscore)

GPU search log confirmed `ungappedprefilter ... --gpu 1` (libmarv GPU prefilter on gfx1100) ran.
All 834 CPU hits byte-identical on GPU. GPU result is strict superset. PASS.

Cross-arch check: gfx1100 results (834 CPU hits, 38 GPU-only) match gfx90a results exactly.

State: linux-gfx1100 -> completed, validated_sha = e7471b4164e38cbac58b4f2c6c1b592e9bfac330.

## Validation 2026-06-08 (windows-gfx1201, RDNA4 RX 9070 XT)

Platform: Windows 11, AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32).
ROCm 7.14.0a20260604 (TheRock nightly pip SDK). HIP_VISIBLE_DEVICES=0.
Fork head: 1a50788 (Windows fixes commit on top of e7471b4).

### Windows delta (required build fixes)

The full foldseek binary has deep POSIX mmap/shm_open dependencies
(DBReader.cpp, FileUtil.cpp, etc.) that are impractical to port within
a validation pass. The standalone libmarv (LIBRARY_ONLY=1) is built as a
DLL, and the existing Marv API harness validates GPU alignment correctness.

Fixes applied to foldseek fork (paralleling MMseqs2 sibling port d34d42d3+398b7c00):

1. cuda_to_hip.h: NOMINMAX/WIN32_LEAN_AND_MEAN before hip_runtime.h;
   HIP_DISABLE_WARP_SYNC_BUILTINS keyed on HIP version (7.14+) not OS
   (suppresses bfloat16 warp-sync overload redefinition on ROCm 7.14);
   __shfl_*_sync macros mapping to maskless HIP equivalents (needed when
   warp-sync builtins are disabled on ROCm 7.14).
2. mapped_file.hpp: Win32 CreateFileMapping/MapViewOfFile implementation
   behind #ifdef _WIN32 (POSIX mmap used in LIBRARY_ONLY=1 by marv.cu
   indirectly via dbdata.cpp).
3. marv.cu: strtok_r -> strtok_s under _WIN32.
4. marv.h: MARV_API __declspec(dllexport/dllimport) for DLL visibility.
5. CMakeLists.txt (libmarv): CMAKE_HIP_USING_LINKER_DEFAULT "" on WIN32
   (clang rejects -fuse-ld=lld-link when --hip-link is present);
   MARV_BUILDING_DLL define for the DLL build.
6. tinyexpr/CMakeLists.txt: guard -fPIC behind if(NOT WIN32).

### Build

```
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel

cmake -S projects/foldseek/src/lib/mmseqs/lib/libmarv/src \
      -B projects/foldseek/build-marv \
      -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
      -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
      -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
      -DCMAKE_PREFIX_PATH=$ROCM -DLIBRARY_ONLY=1 \
      -DCMAKE_BUILD_TYPE=Release

HIP_VISIBLE_DEVICES=0 utils/timeit.sh foldseek compile -- \
  cmake --build projects/foldseek/build-marv -j24 --target marv
```

Build succeeded. marv.dll is 62 MB (62603264 bytes), 14 Marv:: symbols exported
(confirmed via llvm-objdump -p). gfx1201 device code confirmed (COFF DLL, 62MB,
device ISA embedded).

### GPU validation (Marv API harness)

Used the existing marv_validate_gfx1201.cpp harness (same Marv::scan() API
as MMseqs2 validation). Compiled against foldseek's marv.dll:

```
$ROCM/lib/llvm/bin/clang++.exe -std=c++17 -O2 \
  -Iprojects/foldseek/src/lib/mmseqs/lib/libmarv/src \
  -Iprojects/foldseek/src/lib/mmseqs/lib/libmarv/src/hip_compat \
  -Lprojects/foldseek/build-marv -lmarv \
  -o agent_space/foldseek_val_gfx1201.exe \
  agent_space/marv_validate_gfx1201.cpp

HIP_VISIBLE_DEVICES=0 utils/timeit.sh foldseek test -- \
  agent_space/foldseek_val_gfx1201.exe
```

Results:
- Test 1: 20-residue query (all 20 standard amino acids). GPU returns
  top hit id=2, score=116 (expected BLOSUM62 self-score). PASS.
- Test 2: 16xAla query. Top hit id=3, score=64 (16 * BLOSUM62[A][A]=4).
  PASS.

GPU PSSM-based gapless alignment kernels produce correct BLOSUM62 scores on
gfx1201 RDNA4. The Marv::scan() path (same as foldseek ungappedprefilter --gpu)
is exercised.

VERDICT: PASS. State -> completed (validated_sha = 1a507881a2d5086e2a29b6a98a374fb841ba7ffe).

Note: advance-head e7471b4 -> 1a50788 flipped linux-gfx90a and linux-gfx1100 to
revalidate (moatlib classified delta as mixed/arch_independent=False due to the
__shfl_*_sync macro additions in cuda_to_hip.h). On Linux ROCm 7.2.x the Windows
guards are dead code and __shfl_*_sync macros have identical semantics to the
native HIP builtins; full Linux revalidation is expected to be a formality.
linux-gfx1101 is deferred (GPU offline this session).

## Validation 2026-06-08 (linux-gfx90a revalidate, HIP_VISIBLE_DEVICES=3)

Platform: Linux, AMD Instinct MI250X GCD 3 (gfx90a, 104 CUs). ROCm 7.2.1.
Fork head: 1a507881 (Windows fixes commit on top of e7471b4).

Change class: classify reported mixed/arch_independent=False (due to unconditional
__shfl_*_sync macro additions in cuda_to_hip.h). codeobj_diff verdict=differ (device
code objects differ between e7471b41 and 1a507881 builds). Full GPU revalidation required.

Build: cmake configure + cmake --build --target foldseek -j16 for gfx90a at 1a507881
into build-1a507881/. Build succeeded. libmarv.so confirmed at 1a507881 (gfx90a device code).

GPU vs CPU validation (bundled example/, 28 SCOP structures):
```
FOLDSEEK=projects/foldseek/build-1a507881/src/foldseek
HIP_VISIBLE_DEVICES=3

$FOLDSEEK createdb example/ exDB
$FOLDSEEK search exDB exDB aln_cpu tmp_cpu --gpu 0 -e 10
$FOLDSEEK makepaddedseqdb exDB exDB_pad
HIP_VISIBLE_DEVICES=3 $FOLDSEEK search exDB exDB_pad aln_gpu tmp_gpu --gpu 1 -e 10
$FOLDSEEK convertalis exDB exDB aln_cpu cpu.m8
$FOLDSEEK convertalis exDB exDB_pad aln_gpu gpu.m8
```

Results:
- CPU hits: 834, GPU hits: 872
- CPU-only pairs: 0 (all 834 CPU hits reproduced on GPU)
- GPU-only pairs: 38 (borderline low-bitscore, expected ungapped-prefilter sensitivity difference)
- Common pairs with identical fields: 834/834 (0 mismatches, byte-identical pident/alnlen/coords/evalue/bitscore)

GPU search log confirmed `ungappedprefilter ... --gpu 1` (libmarv GPU prefilter on gfx90a) ran.
All 834 CPU hits byte-identical on GPU. GPU result is strict superset. PASS.

Cross-arch check: results (834 CPU hits, 38 GPU-only) match previous gfx90a and gfx1100 results exactly.

State: linux-gfx90a -> completed, validated_sha = 1a507881a2d5086e2a29b6a98a374fb841ba7ffe.

## Validation 2026-06-08 (linux-gfx1100 revalidate, HIP_VISIBLE_DEVICES=3)

Platform: Linux, AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32). ROCm 7.2.1.
Fork head: 1a507881 (Windows fixes commit on top of e7471b4). GPU[3] used.

Change class: classify reported mixed/arch_independent=False. codeobj_diff verdict=differ
(libmarv.so device code objects differ between e7471b41 and 1a507881 builds for gfx1100:
the unconditional __shfl_*_sync macro additions in cuda_to_hip.h changed device ISA).
Full GPU revalidation required.

Build: cmake configure + cmake --build --target foldseek -j16 for gfx1100 at 1a507881
into agent_space/foldseek-gfx1100-gpu3/build-new/. Build succeeded.

```
cmake -S agent_space/foldseek-gfx1100-gpu3/src-new \
      -B agent_space/foldseek-gfx1100-gpu3/build-new \
      -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
HIP_VISIBLE_DEVICES=3 cmake --build agent_space/foldseek-gfx1100-gpu3/build-new \
      --target foldseek -j16
```

GPU vs CPU validation (bundled example/, 28 SCOP structures):
```
FOLDSEEK=agent_space/foldseek-gfx1100-gpu3/build-new/src/foldseek
TESTDIR=agent_space/foldseek-gfx1100-gpu3/testdir

$FOLDSEEK createdb <example/> $TESTDIR/exDB
$FOLDSEEK search $TESTDIR/exDB $TESTDIR/exDB $TESTDIR/aln_cpu $TESTDIR/tmp_cpu --gpu 0 -e 10
$FOLDSEEK makepaddedseqdb $TESTDIR/exDB $TESTDIR/exDB_pad
HIP_VISIBLE_DEVICES=3 $FOLDSEEK search $TESTDIR/exDB $TESTDIR/exDB_pad $TESTDIR/aln_gpu $TESTDIR/tmp_gpu --gpu 1 -e 10
$FOLDSEEK convertalis $TESTDIR/exDB $TESTDIR/exDB $TESTDIR/aln_cpu $TESTDIR/cpu.m8
$FOLDSEEK convertalis $TESTDIR/exDB $TESTDIR/exDB_pad $TESTDIR/aln_gpu $TESTDIR/gpu.m8
```

Results:
- CPU hits: 834, GPU hits: 872
- CPU-only pairs: 0 (all 834 CPU hits reproduced on GPU)
- GPU-only pairs: 38 (borderline low-bitscore, expected ungapped-prefilter sensitivity difference)
- Common pairs with identical fields: 834/834 (0 mismatches, byte-identical pident/alnlen/coords/evalue/bitscore)

GPU search log confirmed `ungappedprefilter ... --gpu 1` (libmarv GPU prefilter on gfx1100) ran.
All 834 CPU hits byte-identical on GPU. GPU result is strict superset. PASS.

Cross-arch check: gfx1100 results (834 CPU hits, 38 GPU-only) match gfx90a and prior gfx1100 results exactly.

State: linux-gfx1100 -> completed, validated_sha = 1a507881a2d5086e2a29b6a98a374fb841ba7ffe.

## Validation 2026-06-19 (windows-gfx1101, RDNA3 Radeon PRO V710)

Platform: Windows 11, AMD Radeon PRO V710 (gfx1101, RDNA3, warpSize=32, HIP_VISIBLE_DEVICES=1).
ROCm 7.14.0a20260604 (TheRock nightly pip SDK). Fork head: e296e43 (comment-only on top of 1a50788).

### Build

libmarv built as LIBRARY_ONLY=1 DLL for gfx1101, mirroring the gfx1201 recipe:

```
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel

cmake -S projects/foldseek/src/lib/mmseqs/lib/libmarv/src \
      -B projects/foldseek/build-marv-gfx1101 \
      -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
      -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
      -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
      -DCMAKE_PREFIX_PATH=$ROCM -DLIBRARY_ONLY=1 \
      -DCMAKE_BUILD_TYPE=Release

HIP_VISIBLE_DEVICES=1 utils/timeit.sh foldseek compile -- \
  cmake --build projects/foldseek/build-marv-gfx1101 -j64 --target marv
```

Build succeeded. marv.dll is 56.8 MB (56821248 bytes). `.hipFatB` + `.hip_fat` sections confirmed present (gfx1101 device ISA embedded).

### GPU validation (Marv API harness)

Compiled the existing marv_validate_gfx1201.cpp harness against foldseek's gfx1101 marv.dll:

```
HIP_VISIBLE_DEVICES=1 utils/timeit.sh foldseek test -- \
  agent_space/foldseek-gfx1101-val/foldseek_val_gfx1101.exe
```

Results:
- Test 1: 20-residue query (all 20 standard amino acids). GPU returns top hit id=2, score=116
  (expected BLOSUM62 self-score). PASS.
- Test 2: 16xAla query. Top hit id=3, score=64 (16 * BLOSUM62[A][A]=4). PASS.

GPU PSSM-based gapless alignment kernels produce correct BLOSUM62 scores on gfx1101 RDNA3.
The Marv::scan() path (same as foldseek ungappedprefilter --gpu) is exercised.
Results match gfx1201 and Linux gfx1100/gfx90a exactly.

VERDICT: PASS. State -> completed (validated_sha = e296e43ec267ec824e5056d7f99c3661f2551643).

## Upstream HIP support validation on linux-gfx90a (2026-10-29)

Purpose: upstream foldseek gained HIP through its `lib/mmseqs` subtree (5b3bc29, 2026-08-20, then f53fe627 "Update lib/mmseqs to MMseqs2 22c2fff"). This validates that path as if it were our port and lists the gaps. No MOAT state or fork branch was changed. The libmarv-level analysis is in the MMseqs2 notes (`port/MMseqs2`, same date); foldseek's vendored `lib/mmseqs/lib/libmarv` is byte-identical to MMseqs2 master 50374396 (`diff -rq` clean), so every libmarv verdict there applies here.

Host: linux-gfx90a, MI250X GCD, HIP_VISIBLE_DEVICES=0, CUDA_VISIBLE_DEVICES=0. ROCm 7.16.26385 (TheRock pip SDK), AMD clang 24.0.0git, gcc 13.3.0, cmake 3.31.6.

Upstream sha: `463739e0014a1549a527de589102cde98f802f37` (master, 2026-09-09).

### Build

foldseek has no HIP option of its own, no ROCm Dockerfile and no ROCm CI; `ENABLE_HIP` is declared as a cache option in `lib/mmseqs/CMakeLists.txt` and works from the top level:

```
R=$(hipconfig --rocmpath)
cmake -S foldseek -B foldseek/build-hip -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=1 \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=$R -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++
utils/timeit.sh foldseek compile -- cmake --build foldseek/build-hip -j32 --target foldseek
```

Result: PASS, no HIP warnings; `foldseek` carries `amdgcn-amd-amdhsa--gfx90a`. ProstT5's ggml is built CPU-only (top-level `ENABLE_CUDA` stays 0, so `GGML_CUDA` is off and nothing sets `GGML_HIP`).

### GPU tests (`ulimit -c 0`)

1. Our port's oracle on the bundled `example/` (28 SCOP structures):
   ```
   foldseek createdb example/ exDB; foldseek makepaddedseqdb exDB exDB_pad
   foldseek search exDB exDB     aln_cpu tmp_cpu --gpu 0 -e 10
   foldseek search exDB exDB_pad aln_gpu tmp_gpu --gpu 1 -e 10
   foldseek convertalis ...      # cpu.m8 / gpu.m8
   ```
   CPU 834 hits, GPU 872; all 834 CPU pairs present on GPU, 0 CPU-only, 38 GPU-only; 0 of 834 common rows differ in ANY column. Log shows `ungappedprefilter ... --gpu 1`. Identical to every validation of our port (gfx90a and gfx1100).
2. Upstream regression (`regression` submodule, foldseek-regression 4acf0dd5), CPU paths in the HIP binary: `SKIP_PROSTT5=1 regression/run_regression.sh build-hip/src/foldseek <scratch>`: 18/18 TEST SUCCESS. ProstT5 skipped because its weight download (`prostt5-f16-gguf.tar.gz`) stalled at 0 bytes on this host. The suite has no GPU test.

### Comparison with our port

All libmarv items: see the MMseqs2 table (upstream better or equivalent on the cc 9.0 collision, DPX, `__hmax2`, `cg::reduce`, legacy mask widths, `__syncwarp`, static RDC link; same `getPaddedQueryLength` literal 32; latent `ClampToInvalid` divergence; no Windows; RDNA unvalidated). foldseek-specific:

| Our port | Upstream | Verdict |
|---|---|---|
| New `USE_HIP` threaded top-level -> mmseqs -> libmarv, keeping `ENABLE_CUDA` for ggml | `ENABLE_HIP` lives in `lib/mmseqs` and sets `ENABLE_CUDA` only in that scope, so ggml stays CPU | Equivalent outcome. Caveat: `-DENABLE_HIP=1 -DENABLE_CUDA=1` together would turn on `GGML_CUDA` and fail without a CUDA toolkit; not a supported combination, just untidy. |
| Windows libmarv DLL (gfx1201/gfx1101 harness) | none | Platform our port covered, upstream does not (same as MMseqs2; foldseek binary is POSIX-only). |
| ProstT5 ggml GPU: out of scope | CPU-only on ROCm | Feature gap both share. ggml has a HIP backend (`GGML_HIP`), so `foldseek createdb --prostt5-model ... --gpu 1` could use the GPU on ROCm. |

### Gaps

Defects on gfx90a: none found.

1. Docs/packaging: foldseek README and wiki say nothing about ROCm, and there is no ROCm Dockerfile or CI job (MMseqs2 has both). A foldseek user cannot tell `-DENABLE_HIP=1` exists.
2. ProstT5 on ROCm: ggml built without `GGML_HIP`, so the ProstT5 path is CPU-only on AMD. A full port would map `ENABLE_HIP` to `GGML_HIP=ON` (+ `GPU_TARGETS`/`CMAKE_HIP_ARCHITECTURES`) in the top-level `if(ENABLE_PROSTT5)` block and validate `createdb --prostt5-model` CPU vs GPU. Needs the weights download, which failed here.
3. RDNA/wave32 runtime: unvalidated (same as MMseqs2; needs a gfx1100 host run of the example oracle, expect 834/872/0 diff).
4. Windows: not offered upstream; low value.

### Draft (not posted)

====
Title: Build ProstT5 with the HIP backend when ENABLE_HIP is set

`-DENABLE_HIP=1` builds the GPU prefilter for ROCm through lib/mmseqs, but the ProstT5 block in the top-level CMakeLists only turns on `GGML_CUDA` for `ENABLE_CUDA`, so on AMD GPUs ProstT5 inference stays on the CPU. This sets `GGML_HIP` (and passes `CMAKE_HIP_ARCHITECTURES` through as ggml's GPU targets) when `ENABLE_HIP` is on.

Test plan: build with `-DENABLE_HIP=1 -DCMAKE_HIP_ARCHITECTURES=gfx90a`, then compare `foldseek createdb <fasta> db --prostt5-model weights --gpu 1` against `--gpu 0` output.
====
(Draft only: the change has not been written or run. It needs a porter round before anyone offers it.)

## Upstream HIP support validation on linux-gfx1100 (2026-10-29)

jeff's request: validate upstream's own ROCm/HIP support as if porting foldseek ourselves, find every gap, and seed follow-up work from what this port learned. No MOAT state or fork branch was changed. See also the MMseqs2 notes section of the same date: the GPU library is the same code.

### What was tested

- Upstream: https://github.com/steineggerlab/foldseek `master` at `463739e0014a1549a527de589102cde98f802f37` (2026-09-09). HIP arrives only through the `lib/mmseqs` subtree update 5b3bc29 (2026-08-20, MMseqs2 dfef51d74, which includes "Add HIP CI" and "Add ROCm to readme" on the MMseqs2 side). `lib/mmseqs/lib/libmarv` is byte-identical to MMseqs2 master 50374396's libmarv (`diff -rq` clean), so every libmarv finding in the MMseqs2 section applies here.
- Host and toolchain as in the MMseqs2 section: Radeon Pro W7800 gfx1100 wave32, TheRock `rocm-sdk-devel` 10.1.0 (HIP 7.16.26385), gcc 13.3, CMake 3.31.6.

### Build

foldseek's top-level CMake has no HIP option of its own. `ENABLE_HIP` is a cache option of `lib/mmseqs` that it forwards to libmarv, so passing it at the top level works:
```
R=/opt/python-3.11-venv/lib/python3.11/site-packages/_rocm_sdk_devel; export ROCM_PATH=$R HIP_PATH=$R
cmake -S foldseek -B foldseek/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=1 \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH=$R
utils/timeit.sh foldseek upstream-hip-compile -- cmake --build foldseek/build -j24
```
PASS: clean build in 58 s. The `foldseek` binary carries gfx1100 code objects (`llvm-objdump --offloading`) and links `libamdhip64.so.7`.

### GPU tests (`ulimit -c 0`, `CUDA_VISIBLE_DEVICES=0`)

1. This port's gate (bundled `example/`, `createdb`, `makepaddedseqdb`, `search --gpu 1 -e 10` vs `--gpu 0`): GPU 872 pairs, CPU 834, common 834, 0 CPU-only, 38 GPU-only, 0 pairs differ in bitscore or pident. Identical to every gfx90a/gfx1100 validation of our port.
2. Stronger oracle, larger set: the 284 SCOP domains from the foldseek-regression submodule (`regression/data/scop`). The GPU result against `search db db_pad --prefilter-mode 1` (CPU ungapped, same algorithm) is byte-identical when sorted (19440 = 19440 rows). Against the default k-mer CPU search: 16180 common, 0 differing, 3260 GPU-only, 0 CPU-only.
3. foldseek's regression suite (`regression` submodule 4acf0dd5, `run_regression.sh build/src/foldseek <scratch>`): 18/19 GOOD. The one failure, `run_easy_search_prostt5`, is environmental: it downloads `prostt5-f16-gguf.tar.gz` from foldseek.steineggerlab.workers.dev, and from this host DNS resolves and TCP connects but no bytes arrive (curl and wget, IPv4 and IPv6). It was stopped after several minutes. The suite has no `--gpu` test at all, which is why item 2 was added.

### Fault-class comparison with our port

Every libmarv item is as recorded in the MMseqs2 section: DPX left out, not emulated; cc 9.0 collision handled by AMD config tables; `hmax2` via `v_pk_max_f16`; `cg::reduce` replaced by a butterfly; legacy full-warp kernels compiled out. foldseek-specific items:

| Our port | Upstream | Verdict |
|---|---|---|
| `getPaddedQueryLength` kept at the literal `sizeof(char4)*32` (host/device format constant) | Unchanged literal 32, not `warpSize` | Same. Correct on wave32 (and on wave64 per our gfx90a runs). |
| `USE_HIP` top-level option threaded to libmarv, deliberately not enabling ggml | `ENABLE_HIP` exists only inside `lib/mmseqs`. The top-level `ENABLE_CUDA` stays 0, so ProstT5 ggml stays CPU-only. | Same scope as ours. See gap 1. |
| Windows libmarv DLL plus harness (gfx1201, gfx1101) | None | Platform gap, as for MMseqs2 |

### Gaps found

1. ProstT5 (`createdb --prostt5-model ... --gpu 1`) is still CUDA-only. This is a GPU module neither upstream's HIP nor our port covered, and it has four layered blockers, each confirmed here:
   - a. `CMakeLists.txt:87-92` turns on `GGML_CUDA` only when `ENABLE_CUDA` is set. Nothing turns on `GGML_HIP`.
   - b. `CMakeLists.txt` forces `GGML_STATIC ON`, and the vendored `lib/prostt5/ggml/src/ggml-hip/CMakeLists.txt:101` stops with `message(FATAL_ERROR "Static linking not supported for HIP/ROCm")` (configure with `-DENABLE_HIP=1 -DGGML_HIP=ON` fails there).
   - c. With that guard removed in a scratch copy (reverted afterwards), the vendored ggml does not compile against ROCm 7 hipBLAS: `ggml-cuda.cu:1112:9: error: unknown type name 'hipblasDatatype_t'` (also at :1614, :1615, plus a `hipblasGemmEx` overload mismatch). `hipblasDatatype_t` was removed in hipBLAS 3; current ggml maps to `hipDataType`/`hipblasComputeType_t`. So the ggml snapshot itself needs updating.
   - d. `src/strucclustutils/structcreatedb.cpp:556` keeps only devices whose name contains "CUDA". A HIP ggml names its devices from `GGML_CUDA_NAME`, which is "ROCm" (`lib/prostt5/ggml/include/ggml-cuda.h:11`). So even a working ggml-hip build would end in "No GPU devices found". Fix: also accept "ROCm" (or test the backend type instead of the name).

   Not runtime-tested, because the weights are unreachable from this host (above). This is the largest real HIP gap in foldseek and a natural follow-up PR: (a) and (d) are small foldseek changes, while (b) and (c) need a ggml update or a shared-library ggml-hip.
2. Build and docs surface. foldseek's README GPU text is NVIDIA-only ("requires an NVIDIA GPU of the Ampere generation or newer", `CUDA_VISIBLE_DEVICES`). There is no top-level `ENABLE_HIP`, no ROCm Dockerfile and no ROCm CI, while MMseqs2 has all three. Draft README addition (not posted): build with `cmake -DENABLE_HIP=1 -DCMAKE_HIP_ARCHITECTURES=<gfx>`; `--gpu 1` structure search works on ROCm (validated on gfx1100 with GPU output byte-identical to `--prefilter-mode 1`); ProstT5 GPU inference is CUDA-only for now.
3. Windows: as for MMseqs2 (our port covered only the libmarv DLL).

### Verdict

Upstream's HIP support is complete and correct for foldseek's `--gpu` structure search on gfx1100, through the shared libmarv. The real gap is ProstT5 GPU inference (gap 1, four blockers), followed by docs, Dockerfile and CI parity with MMseqs2 (gap 2).

### Cross-reference: linux-gfx90a and linux-gfx1100 sections (2026-10-29)

- gfx90a gap 3 ("RDNA/wave32 runtime unvalidated, expect 834/872/0 diff"): validated on gfx1100 with exactly 834/872/0, plus the SCOP 284-domain run, byte-identical to CPU ungapped.
- The gfx90a draft PR "Build ProstT5 with the HIP backend when ENABLE_HIP is set" is not sufficient as written. Setting `GGML_HIP=ON` (blocker a) runs into the other three blockers found on gfx1100: `GGML_STATIC` vs ggml-hip's static guard (b), the vendored ggml's removed `hipblasDatatype_t` (c), and `structcreatedb.cpp:556` dropping devices not named "CUDA" while ggml-hip names them "ROCm" (d). The PR needs a ggml update (or a shared ggml-hip) plus the device-name fix, and must be validated with the weights on a host that can download them.
- The gfx90a caveat that `-DENABLE_HIP=1 -DENABLE_CUDA=1` together would turn on `GGML_CUDA` stands. It is another reason to key ggml's backend on `ENABLE_HIP` explicitly.
