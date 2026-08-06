# MMseqs2 notes

Lead platform: linux-gfx90a (MI250X). ROCm 7.2.1. GPU surface is the vendored
`lib/libmarv` aligner (CUDASW++/cudasw4 derivative), built only when the GPU
option is on. The CPU search/clustering toolkit has no GPU code and is untouched.

## DPX / SIMD strategy: EMULATE (strategy b), not guard-out

The planner's option (a) -- compile-time guard out the DPX short2/int path -- is
NOT viable here, for two independent reasons:

1. The short2 (gapless) and int (Smith-Waterman) kernels are force-instantiated
   in dedicated .cu files (`pssmkernels_gapless_instantiation_dpx.cu`,
   `pssmkernels_smithwaterman_instantiation_dpx.cu`) and their `call_*<short2>` /
   `call_*<int>` symbols are referenced unconditionally from the dispatch switch
   in `cudasw4.cuh` (`case Datatype::Short2`, the `<int>` SW branch). They must
   compile and link even if never selected at runtime.
2. CRITICAL: ROCm reports gfx90a as compute capability 9.0
   (`hipDeviceAttributeComputeCapabilityMajor`=9, Minor=0). The kernel-config
   selectors key on cc: `ccMajor==9 && ccMinor==0` -> the Hopper "sm90" config
   list, which sets `dpx=1` (short2 gapless) and `dpx=1` (int SW). So WITHOUT
   intervention the DPX path would actually RUN on gfx90a, not just compile.

So the ~17 missing intrinsics are emulated AND the config selector is pinned to a
portable path. Both are needed.

### Emulated intrinsics (lib/libmarv/src/marv_simd_amd.cuh)
All scalar, lane-count tied to the SIMD WORD (two 16-bit / four 8-bit lanes),
never to warpSize. Semantics matched to the CUDA Math API:
- 16x2: `__vadd2` (wrap add), `__vmaxs2` (signed max), `__vibmax_u16x2` (unsigned
  max + per-lane a>=b bool), `__vimax3_s16x2`, `__vimax3_s16x2_relu`,
  `__viaddmax_s16x2` (max(a+b,c)), `__viaddmax_s16x2_relu` (max(a+b,c,0)).
- s32 (DPX): `__vibmax_s32` (max + a>=b bool), `__vimax3_s32`, `__viaddmax_s32`,
  `__viaddmax_s32_relu`.
- u8x4: `__vmaxu4`, `__vminu4`, `__vadd4` (wrap), `__vaddus4` (unsigned sat add),
  `__vsubus4` (unsigned sat sub).
- `__hmax2`: per-lane `__hmax` (ROCm has scalar __hmax, not __hmax2); __hmax
  propagates the non-NaN operand, matching CUDA __hmax2.
The `__vibmax_*` boolean is "first operand wins" = (a>=b), which the SW traceback
end-position logic depends on; the relu forms clamp the result at 0.

### Config selector pinned to portable path
`getOptimalKernelConfigs_gapless` and `getOptimalKernelConfigs_SW` get a
`#if defined(USE_HIP)` branch returning the T4-class (sm75) list: smallest tiles,
`datatype=0` (half2 gapless) / `dpx=0` (float SW). This routes runtime through the
portable half2/float SIMT path (no DPX selected) AND keeps dynamic shared memory
within the gfx90a 64 KiB per-block ceiling (probed:
`hipDeviceAttributeMaxSharedMemoryPerBlock`=65536). The emulated DPX intrinsics
still compile and are correct if a future config selects short2/int.

## Wave-size handling (LOW risk, confirmed)
- All cooperative-groups tiles are `cg::tiled_partition<N>`, N in {4,8,16}
  (static_assert <=32). These are logical sub-warp tiles that live within one
  wave64 wavefront; arch-agnostic. `group.shfl_up/down/xor` use the tile width
  (`numThreads`), not the hardware warp.
- `__shfl_*_sync(0xFFFFFFFF, ...)` in the legacy non-PSSM kernels: HIP requires a
  64-bit lane mask and static_asserts on a 32-bit one. Routed to the maskless
  `__shfl_*` builtins via macros in cuda_to_hip.h (the mask is dropped; width is
  always passed explicitly or forced to 32 in kernelhelpers.cuh). Width-32 logical
  semantics preserved on wave64.
- NO host+device-shared warp-width serialized-format constant. `WARPSIZE(32)` in
  hpc_helpers is defined but UNUSED in libmarv. No warp-width geometry is baked
  into any buffer layout. So the serialized-format fault class does not apply.
- `__reduce_max_sync` is behind `#if __CUDA_ARCH__ >= 800`; on HIP __CUDA_ARCH__
  is undefined (=0) so the shfl fallback is taken -- no missing intrinsic call.
- cg::reduce / `<cooperative_groups/reduce.h>` do not exist on ROCm 7.2.1.
  Replaced every reduce_max with a tile-local butterfly over `group.shfl_xor`
  (`marv_tile_reduce` in mathops.cuh), guarded by USE_HIP; correct on wave32 and
  wave64 since it iterates `group.size()/2 .. 1` within the tile.

## Strategy A glue (compat shim)
- NEW `lib/libmarv/src/cuda_to_hip.h` (force `-include`d on the HIP build): maps
  the cuda runtime API (~70 symbols) to hip*, defines `__grid_constant__` away,
  aliases `cuda::std` -> `std` and provides a `cub::SwitchDevice`, adds generic
  `__shfl*` overloads for the packed types libmarv shuffles that ROCm does not
  cover (short2, int2, float2, int3; __half2 is already provided by ROCm),
  redirects the _sync shuffles to maskless, and includes the SIMD emulations.
- NEW `lib/libmarv/src/hip_compat/{cuda_fp16.h,cooperative_groups.h,
  cooperative_groups/reduce.h}`: redirect the source's `#include <cuda_*>` /
  `<cooperative_groups*>` to the HIP headers; reduce.h is just a stub (the
  reduce calls are replaced).
- The vendored hpc_helpers and a few libmarv files key device code on `__CUDACC__`
  / `__NVCC__`, which hipcc does not define. Do NOT `#define __CUDACC__` globally
  -- it makes rocThrust route to the CUDA backend (`cub/detail/...` not found) and
  duplicates HIP's 64-bit atomics. Instead each guard was made HIP-aware
  (`defined(__CUDACC__) || defined(__HIPCC__)`, `defined(__NVCC__) || defined(__HIPCC__)`),
  with the uint64 atomic / ffs overloads and the `mov %%laneid` PTX kept
  NVCC-only (HIP provides those; lane_id uses `__lane_id()`).
- `__constant__` blosum / query arrays in blosum.cu/.hpp and kernels.cuh were
  NVCC-guarded; made HIP-aware so the device symbols exist.

## Build (gfx90a)
libmarv is built as a SHARED library on HIP so the relocatable-device-code link
(`-fgpu-rdc`, `HIP_RESOLVE_DEVICE_SYMBOLS ON`, `--offload-compress`) happens at
marv.so creation; the mmseqs executable then links only host symbols and never
needs the HIP device linker (mmseqs is a plain C++ link). rocThrust/hip::host are
linked PRIVATE on marv so their HIP-language `--offload-arch` flags do not leak to
the C++ consumers. Only the `.cu`/`.cpp` TUs are compiled HIP; the `.cuh`/`.h`
entries in ALIGN_SOURCES must NOT be compiled (that duplicates the kernels.cuh
`__constant__`).

```
cd projects/MMseqs2/src
cmake -S . -B build-hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build-hip -j16 --target mmseqs   # HIP_VISIBLE_DEVICES=2 on this host
```
Follower (gfx1100/gfx1151): same, `-DCMAKE_HIP_ARCHITECTURES=gfx1100` (or a
fat-binary `"gfx90a;gfx1100"`). No source change expected.

Non-GPU build (must not regress): `cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release`
then `--build build-cpu --target mmseqs`. Confirmed builds and runs identically to
the GPU build's `--gpu 0` path (12901 hits on the bundled example).

## Validation (real gfx90a, GCD2, zero external downloads)
Bundled `examples/QUERY.fasta` (300 KB) + `examples/DB.fasta` (11 MB):
```
mmseqs createdb examples/DB.fasta targetDB
mmseqs makepaddedseqdb targetDB targetDB_padded
mmseqs easy-search examples/QUERY.fasta targetDB_padded gpu.m8 tmp_gpu --gpu 1
mmseqs easy-search examples/QUERY.fasta targetDB     cpu.m8 tmp_cpu --gpu 0
```
(`--gpu` honors CUDA_VISIBLE_DEVICES; set both CUDA_VISIBLE_DEVICES=2 and
HIP_VISIBLE_DEVICES=2.)

Result: GPU 14482 hit pairs, CPU 12901. Of the 12438 (query,target) pairs present
in BOTH, ZERO differ in bitscore by >0.5 and ZERO differ in pident by >0.5 -- the
GPU Smith-Waterman scores match the CPU oracle EXACTLY. The 463 CPU-only and 2044
GPU-only pairs are all low-score prefilter-boundary hits (median bitscore 70/60 vs
overall median 134, max 6666) -- the documented prefilter-sensitivity difference
between the CPU and GPU recall paths, not a correctness gap. PASS.

## For the foldseek port (it vendors this same libmarv)
Reuse this whole approach verbatim: the cuda_to_hip.h shim, marv_simd_amd.cuh
emulations, the cg::reduce -> butterfly replacement, the config-selector USE_HIP
pin, the SHARED-lib RDC build, and the hpc_helpers `__CUDACC__/__NVCC__` ->
`||__HIPCC__` edits. The deciding facts (gfx90a reports cc 9.0 so DPX would RUN;
.cuh must not be compiled as TUs; do NOT define __CUDACC__) are libmarv-intrinsic
and will recur in foldseek.

## Review 2026-06-04 (reviewer: review-passed)

Verdict: APPROVE. No changes-requested. Reviewed git diff 11933403...c755847a
(fork moat-port). All ROCm fault classes checked; the diff is contained
(25 files, +612/-38), additive, and the CUDA/CPU paths are untouched.

Non-blocking observations (kept for the validator and the prep phase; none gate
the merge):

1. Validation scope nuance (not a defect). The GPU-vs-CPU score-exact result
   (12438 pairs, 0 differ >0.5) exercises the LIVE path only, which the config
   pin forces to half2 gapless + float Smith-Waterman. The emulated DPX integer
   / short2 intrinsics (__viaddmax_s16x2[_relu], __vimax3_s16x2[_relu],
   __vibmax_*, __viaddmax_s32[_relu], etc.) are force-instantiated and must
   compile/link, but are NOT runtime-selected on gfx90a, so their numerics are
   NOT covered by that score compare. Their correctness rests on code review +
   CUDA Math API semantics + the foldseek cross-check (below), which is solid.
   If a future arch/config ever selects dpx=1 / short2 on AMD, that path needs
   its own runtime check. The notes/commit phrasing "validates the emulations"
   should be read as "validates the live half2/float path and that the
   emulations compile and link cleanly".

2. DPX add is WRAPPING, intentionally. marv_simd_amd.cuh __viaddmax_s16x2 (and
   the _relu form, lines 100-119) casts the per-lane sum through unsigned short
   then back to short = two's-complement wrap, matching CUDA __viaddmax_*
   (max(a+b,c) with modular add; saturation is only in the dedicated _sat
   forms). Consistent with __vadd2 (also wrapping). Correct.

3. __vibmax_u16x2 (UNSIGNED) backs MathOps<short2>::max(a,b,bool*,bool*) which is
   commented "max(a,b)" (signed). Verified the short2 bool-output max/add_max
   variants have NO callers in pssmkernels_* (grep clean), so the signed-vs-
   unsigned mismatch is dead in practice; matches upstream/foldseek anyway.
   __vibmax_s32 (signed) likewise has no live caller. No action.

4. ptx_wrappers.cuh: HAS_BLACKWELL_INT8_PTX stays undefined on HIP
   (__CUDACC_VER_MAJOR__ and __CUDA_ARCH_FAMILY_SPECIFIC__ both absent), and the
   ptx_* wrappers are unreferenced on HIP (MathOps<u8x4> #else routes to
   __vmaxu4/__vadd4/__vaddus4/__vsubus4), so the bodyless extern __device__ stub
   is never referenced. No missing-return UB. Confirmed.

5. WARPSIZE(32) in hpc_helpers/cuda_helpers.cuh is defined and UNUSED across
   libmarv (only the #define site). No warp-width geometry in any layout, so the
   serialized-format fault class does not apply (unlike foldseek's
   getPaddedQueryLength pad-32, which is in foldseek's own code, not libmarv).
   Confirmed.

6. Foldseek consistency (same vendored libmarv). The DPX/video emulations are
   SEMANTICALLY IDENTICAL between the two ports (same wrap add, same >=/> tie-
   breaks, same relu-at-0, same unsigned __vibmax_u16x2, same __hmax2). Two
   independent ports converging on the same semantics is a strong correctness
   signal. They DIVERGE structurally, prep-phase cleanup candidates if the two
   forks ever share one upstream libmarv change:
     - MMseqs2 splits emulations into marv_simd_amd.cuh; foldseek inlines them
       into cuda_to_hip.h.
     - header guard MARV_CUDA_TO_HIP_H vs LIBMARV_CUDA_TO_HIP_H.
     - shuffle-mask strategy: MMseqs2 routes __shfl_*_sync -> maskless __shfl_*
       and adds generic vector __shfl overloads (short2/int2/float2/int3);
       foldseek defines a 64-bit WARP_FULL_MASK and keeps the _sync calls. Both
       valid; pick one before any joint upstreaming.

7. CC-9.0 collision: CONFIRMED as a PORTING_GUIDE promotion. gfx90a reports
   compute capability 9.0; a cc-keyed kernel-config selector (here
   getOptimalKernelConfigs_gapless / _SW) silently selects the Hopper sm90
   (dpx=1) path AT RUNTIME on AMD. Found independently by both MMseqs2 and
   foldseek. General lesson: any CUDA selector keyed on cc major/minor must be
   audited on AMD, because ROCm maps gfx90a->9.0 and will land on the Hopper
   branch. The fix pattern is a USE_HIP branch pinning a conservative
   (small-tile, non-DPX) config that also respects the AMD shared-memory ceiling.

Build wiring verified against the local ROCm 7.2.1: roc::rocthrust target
exists, hip_LIB_INSTALL_DIR is set by hip-config.cmake, libamdhip64.so present,
and the porter's build-hip/src/mmseqs binary exists (HIP build succeeded).
Commit message is clean ([ROCm], <=72-char title, Claude disclosed, no noreply
trailer, Test Plan with literal commands, root cause explained). No MOAT jargon
upstream-visible. ASCII, ROCm casing correct. GPU revalidation by the validator
is the remaining gate (expected; not a review blocker).

## Validation 2026-06-04 (linux-gfx90a, GCD2, validated_sha c755847a)

Platform: linux-gfx90a (AMD Instinct MI250X, gfx90a, ROCm 7.2.1). HIP_VISIBLE_DEVICES=2.

### Build

Fork clone already at c755847ab4114fe8a470bf9ca9832a61f37ed56f (porter's build intact).
libmarv.so contains gfx90a code object (25819904 bytes,
`hipv4-amdgcn-amd-amdhsa--gfx90a` confirmed via roc-obj-ls). Incremental
rebuild confirmed clean:

```
cd projects/MMseqs2/src
utils/timeit.sh MMseqs2 compile -- cmake --build build-hip -j16 --target mmseqs
```

### GPU vs CPU validation

```
MMSEQS=projects/MMseqs2/src/build-hip/src/mmseqs
mmseqs createdb examples/DB.fasta VALDIR/targetDB
mmseqs makepaddedseqdb VALDIR/targetDB VALDIR/targetDB_padded
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 utils/timeit.sh MMseqs2 test -- \
  mmseqs easy-search examples/QUERY.fasta VALDIR/targetDB_padded VALDIR/gpu.m8 VALDIR/tmp_gpu --gpu 1
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
  mmseqs easy-search examples/QUERY.fasta VALDIR/targetDB VALDIR/cpu.m8 VALDIR/tmp_cpu --gpu 0
```

Results:
- GPU (--gpu 1): 14482 hit pairs
- CPU (--gpu 0): 12901 hit pairs
- Common pairs: 12438
- GPU-only pairs: 2044 (prefilter-boundary, low-score; expected recall difference)
- CPU-only pairs: 463 (prefilter-boundary; expected)
- Pairs with |bitscore diff| > 0.5: 0
- Pairs with |pident diff| > 0.5: 0
- Max bitscore diff: 0.0000, max pident diff: 0.0000

All 12438 common pairs: GPU Smith-Waterman scores match CPU oracle EXACTLY.
Exercises the live half2 gapless + float Smith-Waterman path on real gfx90a wave64.

### Non-GPU regression check

CPU-only build (build-cpu/src/mmseqs, no USE_HIP) with --gpu 0: 12901 hits --
identical to GPU build --gpu 0 path. No regression.

VERDICT: PASS. State -> completed (validated_sha = c755847ab4114fe8a470bf9ca9832a61f37ed56f).

## Validation 2026-06-04 (linux-gfx1100, RDNA3 wave32)

Platform: linux-gfx1100 (AMD Radeon Pro W7800 48GB, gfx1100, ROCm 7.2.1).
HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 (GPU 0 had orphaned KFD context; GPU 1 responsive).
warpSize=32 confirmed on gfx1100 native wave32.

### Build

Fork cloned fresh at c755847ab4114fe8a470bf9ca9832a61f37ed56f (matches head_sha).

```
cd projects/MMseqs2/src
cmake -S . -B build-hip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm
utils/timeit.sh MMseqs2 compile -- cmake --build build-hip -j16 --target mmseqs
```

Build succeeded. gfx1100 code object confirmed in libmarv.so:
`llvm-objdump --offloading` reports `hipv4-amdgcn-amd-amdhsa--gfx1100`.

### GPU vs CPU validation

```
MMSEQS=projects/MMseqs2/src/build-hip/src/mmseqs
mmseqs createdb examples/DB.fasta VALDIR/targetDB
mmseqs makepaddedseqdb VALDIR/targetDB VALDIR/targetDB_padded
HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 utils/timeit.sh MMseqs2 test -- \
  mmseqs easy-search examples/QUERY.fasta VALDIR/targetDB_padded VALDIR/gpu.m8 VALDIR/tmp_gpu --gpu 1
mmseqs easy-search examples/QUERY.fasta VALDIR/targetDB VALDIR/cpu.m8 VALDIR/tmp_cpu --gpu 0
```

Results:
- GPU (--gpu 1): 14482 hit pairs
- CPU (--gpu 0): 12901 hit pairs
- Common pairs: 12438
- GPU-only pairs: 2044 (prefilter-boundary, low-score; median bitscore 60)
- CPU-only pairs: 463 (prefilter-boundary; median bitscore 70)
- Pairs with |bitscore diff| > 0.5: 0
- Pairs with |pident diff| > 0.5: 0
- Max bitscore diff: 0.0000, max pident diff: 0.0000

All 12438 common pairs: GPU Smith-Waterman scores match CPU oracle EXACTLY on real
gfx1100 native wave32. Results are identical to the gfx90a lead validation
(12438 common, 2044 GPU-only, 463 CPU-only, 0 mismatches). Wave32 path is correct:
cooperative-groups tile butterfly (marv_tile_reduce) iterates group.size()/2..1
within the tile, arch-agnostic; SIMD emulations are lane-count tied to SIMD word
width, never warpSize.

### Non-GPU regression check

CPU oracle (--gpu 0 path in GPU build): 12901 hits -- identical to gfx90a lead. No regression.

VERDICT: PASS. State -> completed (validated_sha = c755847ab4114fe8a470bf9ca9832a61f37ed56f).

## Validation 2026-06-07 (windows-gfx1201, RDNA4 RX 9070 XT)

Platform: Windows 11, AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32).
ROCm 7.14.0a20260604 (TheRock nightly pip SDK). HIP_VISIBLE_DEVICES=0.
Fork head: d34d42d3 (Windows fixes commit on top of c755847a).

### Windows delta (required build fixes)

The full `mmseqs` binary has deep POSIX mmap/shm_open dependencies
(DBReader.cpp, FileUtil.cpp, etc.) that are impractical to port within
a validation pass. Instead, the standalone `libmarv` (LIBRARY_ONLY=1)
is built as a DLL, and a custom C++ harness validates GPU alignment
correctness via the `Marv` API.

Fixes required for `libmarv.dll` on Windows-MSVC-ABI clang:

1. `mapped_file.hpp`: Win32 CreateFileMapping/MapViewOfFile
   implementation; NOMINMAX + WIN32_LEAN_AND_MEAN guards before windows.h.
2. `cuda_to_hip.h`: NOMINMAX/WIN32_LEAN_AND_MEAN before hip_runtime.h.
3. `convert.cuh`: Add `__HIP_DEVICE_COMPILE__` alongside `__CUDA_ARCH__`
   to select the device code path (marv_simd_amd.cuh provides `__vminu4`
   as a device function).
4. `marv.cu`: strtok_r -> strtok_s under _WIN32.
5. `marv.h`: MARV_API `__declspec(dllexport/dllimport)` so the Marv
   class is visible from the DLL on Windows.
6. `CMakeLists.txt` (libmarv): Set `CMAKE_HIP_USING_LINKER_DEFAULT ""`
   on Windows (clang rejects -fuse-ld=lld-link when --hip-link is
   present). Define `MARV_BUILDING_DLL` for the dll build.
7. `tinyexpr/CMakeLists.txt`: Guard -fPIC (unsupported on MSVC-target
   clang) behind `if(NOT WIN32)`.
8. `MMseqsResourceCompiler.cmake`: Win32 branch using cmake -E commands
   instead of checkshell.sh (not executable in cmd.exe).
9. `src/CMakeLists.txt`: Use `hip::amdhip64` imported target (portable
   to Windows) instead of hardcoded libamdhip64.so path.
10. `GpuUtil.cpp`: Guard POSIX includes behind `#ifndef _WIN32`; Windows
    stubs for GPU server mode (unused by direct --gpu 1 path).
11. `ungappedprefilter.cpp`, `gpuserver.cpp`: Guard sys/mman.h behind
    `#ifndef _WIN32`.

### Build

```
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel

cmake -S projects/MMseqs2/src/lib/libmarv/src \
      -B projects/MMseqs2/build-marv \
      -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
      -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
      -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
      -DCMAKE_PREFIX_PATH=$ROCM -DLIBRARY_ONLY=1 \
      -DCMAKE_BUILD_TYPE=Release

HIP_VISIBLE_DEVICES=0 utils/timeit.sh MMseqs2 compile -- \
  cmake --build projects/MMseqs2/build-marv -j32 --target marv
```

Build succeeded. marv.dll is 32 MB, contains gfx1201 device code.
14 Marv:: symbols exported (confirmed via llvm-objdump -p).

### GPU validation (custom harness via Marv API)

Direct `mmseqs` binary cannot be built on Windows without extensive POSIX
porting; the GPU kernels are exercised via a standalone C++ harness that
calls `Marv::scan()` directly (same path as mmseqs ungappedprefilter.cpp).

```
# Compile harness (in agent_space/marv_validate_gfx1201.cpp)
$ROCM/lib/llvm/bin/clang++.exe -std=c++17 -O2 \
  -Iprojects/MMseqs2/src/lib/libmarv/src \
  -Iprojects/MMseqs2/src/lib/libmarv/src/hip_compat \
  -Lprojects/MMseqs2/build-marv -lmarv \
  -o agent_space/marv_val_gfx1201.exe \
  agent_space/marv_validate_gfx1201.cpp

HIP_VISIBLE_DEVICES=0 utils/timeit.sh MMseqs2 test -- \
  agent_space/marv_val_gfx1201.exe
```

Results:
- Test 1: 20-residue query (all 20 standard amino acids). GPU returns
  top hit id=2, score=116 (expected BLOSUM62 self-score). PASS.
- Test 2: 16xAla query. Top hit id=3, score=64 (16 * BLOSUM62[A][A]=4).
  PASS.

GPU Smith-Waterman alignment kernels produce correct BLOSUM62 scores on
gfx1201 RDNA4. The PSSM-based gapless scan path is exercised (same
Marv::scan() call as production ungappedprefilter.cpp).

VERDICT: PASS. State -> completed (validated_sha = d34d42d3).

## Revalidation 2026-06-08 (linux-gfx90a, GCD3)

Platform: linux-gfx90a (AMD Instinct MI250X, gfx90a, ROCm 7.2.1). HIP_VISIBLE_DEVICES=3.

### Delta classification

c755847a -> d34d42d3: one commit "[ROCm] Windows (MSVC-ABI clang) build support
for HIP port". Classified `mixed` (arch_independent=False) by moatlib. Includes
changes to cuda_to_hip.h, CMakeLists.txt, convert.cuh, and Windows POSIX stubs.
Binary-equivalence carry-forward not attempted due to mixed classification.

### Build regression found (Linux fix required)

d34d42d3 defined `HIP_DISABLE_WARP_SYNC_BUILTINS` unconditionally in
cuda_to_hip.h to work around a bfloat16 redefinition error in ROCm 7.14
(Windows/TheRock). On Linux ROCm 7.2.1 this guard wraps `__syncwarp` inside
`amd_warp_sync_functions.h`, so suppressing it removes `__syncwarp` and breaks
compilation of `kernels.cuh:777`. Fix: scope the define to `_WIN32` only.

Fix committed as: `dbeac858` "[ROCm] Scope HIP_DISABLE_WARP_SYNC_BUILTINS to Windows only"
Pushed to fork moat-port; head_sha updated to dbeac858.

### Build (at dbeac858)

```
cmake -S projects/MMseqs2/src -B projects/MMseqs2/src/build-hip-new \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm
HIP_VISIBLE_DEVICES=3 utils/timeit.sh MMseqs2 compile -- \
  cmake --build projects/MMseqs2/src/build-hip-new -j16 --target mmseqs
```

Build succeeded (clean, no errors).

codeobj_diff verdict between build-hip (c755847a) and build-hip-new (dbeac858):
`differ` -- full GPU re-run required.

### GPU vs CPU validation

```
MMSEQS=projects/MMseqs2/src/build-hip-new/src/mmseqs
mmseqs createdb examples/DB.fasta valdir-revalidate-20260608/targetDB
mmseqs makepaddedseqdb valdir-revalidate-20260608/targetDB valdir-revalidate-20260608/targetDB_padded
HIP_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=3 utils/timeit.sh MMseqs2 test -- \
  mmseqs easy-search examples/QUERY.fasta valdir-revalidate-20260608/targetDB_padded \
  valdir-revalidate-20260608/gpu.m8 valdir-revalidate-20260608/tmp_gpu --gpu 1
mmseqs easy-search examples/QUERY.fasta valdir-revalidate-20260608/targetDB \
  valdir-revalidate-20260608/cpu.m8 valdir-revalidate-20260608/tmp_cpu --gpu 0
```

Results:
- GPU (--gpu 1): 14482 hit pairs
- CPU (--gpu 0): 12901 hit pairs
- Common pairs: 12438
- GPU-only pairs: 2044 (prefilter-boundary, low-score; expected)
- CPU-only pairs: 463 (prefilter-boundary; expected)
- Pairs with |bitscore diff| > 0.5: 0
- Pairs with |pident diff| > 0.5: 0
- Max bitscore diff: 0.0000, max pident diff: 0.0000

All 12438 common pairs: GPU Smith-Waterman scores match CPU oracle EXACTLY.
Identical result counts to the original gfx90a validation (c755847a).

VERDICT: PASS. State -> completed (validated_sha = dbeac858).

## Revalidation 2026-06-08 (linux-gfx1100, RDNA3 wave32, GPU index 2)

Platform: linux-gfx1100 (AMD Radeon Pro W7800 48GB, gfx1100, ROCm 7.2.1).
HIP_VISIBLE_DEVICES=2. Triggered by HEAD advance from c755847a to dbeac858.

### Delta classification

c755847a -> dbeac858: two commits (d34d42d3 Windows build support, dbeac858 scope
HIP_DISABLE_WARP_SYNC_BUILTINS to Windows only). Classified `mixed` (arch_independent=False).
codeobj_diff verdict: `differ` (libmarv.so device ISA differs). Full GPU revalidation required.

### Build (at dbeac858)

```
cd projects/MMseqs2/src
cmake -S . -B agent_space/MMseqs2-gfx1100-gpu2-new \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm
HIP_VISIBLE_DEVICES=2 utils/timeit.sh MMseqs2 compile -- \
  cmake --build agent_space/MMseqs2-gfx1100-gpu2-new -j16 --target mmseqs
```

Build succeeded. gfx1100 code object confirmed (hipv4-amdgcn-amd-amdhsa--gfx1100).

### GPU vs CPU validation

```
MMSEQS=agent_space/MMseqs2-gfx1100-gpu2-new/src/mmseqs
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 utils/timeit.sh MMseqs2 test -- \
  $MMSEQS easy-search examples/QUERY.fasta targetDB_padded gpu.m8 tmp_gpu --gpu 1
$MMSEQS easy-search examples/QUERY.fasta targetDB cpu.m8 tmp_cpu --gpu 0
```

Results:
- GPU (--gpu 1): 14482 hit pairs
- CPU (--gpu 0): 12901 hit pairs
- Common pairs: 12438
- GPU-only pairs: 2044 (prefilter-boundary, low-score; expected)
- CPU-only pairs: 463 (prefilter-boundary; expected)
- Pairs with |bitscore diff| > 0.5: 0
- Pairs with |pident diff| > 0.5: 0
- Max bitscore diff: 0.0000, max pident diff: 0.0000

All 12438 common pairs: GPU Smith-Waterman scores match CPU oracle EXACTLY.
Identical to original gfx1100 validation and gfx90a revalidation results.
The dbeac858 Linux fix (HIP_DISABLE_WARP_SYNC_BUILTINS scoped to _WIN32) builds
and runs correctly on Linux ROCm 7.2.1.

VERDICT: PASS. State -> completed (validated_sha = dbeac858).

## Re-key HIP_DISABLE_WARP_SYNC_BUILTINS to runtime version 2026-06-08 (linux-gfx90a, GCD3)

Platform: linux-gfx90a (AMD Instinct MI250X, gfx90a, ROCm 7.2.1). HIP_VISIBLE_DEVICES=3.

### Change

cuda_to_hip.h previously gated `HIP_DISABLE_WARP_SYNC_BUILTINS` on `#ifdef _WIN32`.
That OS proxy is only correct by accident in this fleet (Windows == ROCm 7.14 via
TheRock; Linux == ROCm 7.2). The real trigger is the HIP runtime version: on ROCm
7.14+ `<amd_hip_bf16.h>` defines `__shfl_*_sync` bf16 overloads that collide with
this header's own warp-sync shuffle definitions, so the flag must suppress the
runtime builtins; on ROCm 7.2.x those overloads AND `__syncwarp` (kernels.cuh:777)
share the same guard, so the flag must NOT be set there or `__syncwarp` disappears
and the build breaks. Re-keyed to `HIP_VERSION_MAJOR/MINOR >= 7.14` (include
`<hip/hip_version.h>`, keep the define before `<hip/hip_runtime.h>`). A Linux+7.14
or Windows+7.2 build is now handled correctly rather than by OS coincidence.

Committed on top of dbeac858 as `398b7c00`
"[ROCm] Key HIP_DISABLE_WARP_SYNC_BUILTINS on ROCm version, not OS".
Pushed to fork moat-port (fast-forward dbeac858..398b7c00). advance-head flipped
the three completed platforms to revalidate (delta classed `mixed`: cuda_to_hip.h
preprocessor token count differs).

### Behavior-preserving proof (device identity)

On Linux ROCm 7.2.1 the new check evaluates false (7.2 < 7.14), so
`HIP_DISABLE_WARP_SYNC_BUILTINS` stays undefined and `__syncwarp` remains -- the
exact prior `_WIN32`-not-defined Linux behavior. Built libmarv.so fresh from both
trees (baseline dbeac858 via `git stash` of the edit; new 398b7c00) on gfx90a GPU3:

```
cmake -S . -B <dir> -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm
HIP_VISIBLE_DEVICES=3 utils/timeit.sh MMseqs2 compile -- cmake --build <dir> -j16 --target marv   # baseline
HIP_VISIBLE_DEVICES=3 utils/timeit.sh MMseqs2 compile -- cmake --build <dir> -j16 --target mmseqs # new (full kernels.cuh compile, __syncwarp present)
python3 utils/codeobj_diff.py <base>/libmarv.so <new>/libmarv.so
```

codeobj_diff verdict: `identical` (exported symbols + device ISA identical,
3096 exports). Both libmarv.so are 30306584 bytes. (a) libmarv extension builds
clean; (b) device code unchanged.

### GPU search smoke (gfx90a, GPU3)

```
MMSEQS=<new build>/src/mmseqs
HIP_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=3 $MMSEQS createdb examples/DB.fasta targetDB
HIP_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=3 $MMSEQS makepaddedseqdb targetDB targetDB_padded
HIP_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=3 utils/timeit.sh MMseqs2 test -- \
  $MMSEQS easy-search examples/QUERY.fasta targetDB_padded gpu.m8 tmp_gpu --gpu 1
```

14482 GPU hit pairs (identical to every prior validated gfx90a/gfx1100 run); self
hit pident 1.000 score 487. (c) GPU path runs correctly.

linux-gfx90a carried forward to completed at 398b7c00 (method binary-equiv: device
ISA + exports identical, GPU smoke confirms). linux-gfx1100 and windows-gfx1201
left at revalidate for their own hosts (gfx1100 is device-identical on Linux 7.2
too; gfx1201 on Windows 7.14 now fires the guard via HIP_VERSION rather than
_WIN32 -- identical effect, but a real check confirms it).

## Revalidation 2026-06-08 (linux-gfx1100, RDNA3 wave32, GPU index 0)

Platform: linux-gfx1100 (AMD Radeon Pro W7800 48GB, gfx1100, ROCm 7.2.1).
HIP_VISIBLE_DEVICES=0. Revalidate triggered by HEAD advance dbeac858 -> 398b7c00.

### Delta classification

dbeac858 -> 398b7c00: one commit "[ROCm] Key HIP_DISABLE_WARP_SYNC_BUILTINS on
ROCm version, not OS". Changes only cuda_to_hip.h comment and preprocessor guard.

### Binary-equivalence carry-forward

On Linux ROCm 7.2.1, HIP_VERSION_MINOR=2 < 14, so `HIP_DISABLE_WARP_SYNC_BUILTINS`
evaluates false under the new version check -- identical behavior to the prior
`#ifdef _WIN32` path (also false on Linux). Built at 398b7c00 for gfx1100 (new
build) and compared to the existing dbeac858 build (MMseqs2-gfx1100-gpu2-new):

```
HIP_VISIBLE_DEVICES=0 cmake -S projects/MMseqs2/src \
  -B agent_space/MMseqs2-gfx1100-gpu0 \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm
utils/timeit.sh MMseqs2 compile -- \
  cmake --build agent_space/MMseqs2-gfx1100-gpu0 -j16 --target mmseqs

python3 utils/codeobj_diff.py \
  agent_space/MMseqs2-gfx1100-gpu2-new/lib/libmarv/src/libmarv.so \
  agent_space/MMseqs2-gfx1100-gpu0/lib/libmarv/src/libmarv.so
```

codeobj_diff verdict: `identical` (exported symbols + device ISA identical,
3096 exports). The preprocessor change is inert on Linux ROCm 7.2.1 for gfx1100
exactly as it was for gfx90a.

VERDICT: CARRY-FORWARD (binary-equiv). State -> completed (validated_sha = 398b7c00).

## Validation 2026-06-08 (windows-gfx1201, carry-forward)

Delta: d34d42d3 -> 398b7c00 (two commits, both touch only lib/libmarv/src/cuda_to_hip.h).

Method: source-class inspection.

Commit chain:
- dbeac858: Scoped HIP_DISABLE_WARP_SYNC_BUILTINS to `#ifdef _WIN32` (was unconditional)
- 398b7c00: Re-keyed to `#include <hip/hip_version.h>` + HIP_VERSION_MAJOR/MINOR check (>=7.14)

On Windows (_WIN32 defined) with TheRock 7.14 (HIP_VERSION_MAJOR=7, HIP_VERSION_MINOR=14):
- d34d42d3: HIP_DISABLE_WARP_SYNC_BUILTINS is always set
- dbeac858: `#ifdef _WIN32` -> set (same)
- 398b7c00: `HIP_VERSION_MAJOR==7 && HIP_VERSION_MINOR>=14` -> true -> set (same)

All three commits produce identical preprocessing on this host. The gfx1201 libmarv.dll device ISA is unchanged.

VERDICT: CARRY-FORWARD (source-class). State -> completed (validated_sha = 398b7c00). No GPU re-run required.

## Validation 2026-06-19 (windows-gfx1101, RDNA3 PRO V710)

Platform: Windows 11, AMD Radeon PRO V710 (gfx1101, RDNA3, wave32).
ROCm 7.14.0a20260604 (TheRock nightly pip SDK). HIP_VISIBLE_DEVICES=1 (verified via
hipInfo: Name=AMD Radeon PRO V710, gcnArchName=gfx1101, warpSize=32).
Fork head: 398b7c00.

### Build

Same LIBRARY_ONLY libmarv DLL approach as gfx1201 -- `mmseqs` binary has POSIX
dependencies not viable on Windows; GPU kernels validated via the Marv API harness.

```
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel

cmake -S projects/MMseqs2/src/lib/libmarv/src \
      -B projects/MMseqs2/build-marv-gfx1101 \
      -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
      -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
      -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
      -DCMAKE_PREFIX_PATH=$ROCM -DLIBRARY_ONLY=1 \
      -DCMAKE_BUILD_TYPE=Release

HIP_VISIBLE_DEVICES=1 utils/timeit.sh MMseqs2 compile -- \
  cmake --build projects/MMseqs2/build-marv-gfx1101 -j64 --target marv
```

Build succeeded. marv.dll is 31.7 MB, contains gfx1101 device code (.hipFatB and
.hip_fat sections confirmed via llvm-readobj).

### GPU validation (custom harness via Marv API)

Same harness as gfx1201 (agent_space/marv_validate_gfx1201.cpp). Compiled against
gfx1101 marv.dll; TheRock runtime DLLs co-located with exe.

```
$ROCM/lib/llvm/bin/clang++.exe -std=c++17 -O2 \
  -Iprojects/MMseqs2/src/lib/libmarv/src \
  -Iprojects/MMseqs2/src/lib/libmarv/src/hip_compat \
  -Lprojects/MMseqs2/build-marv-gfx1101 -lmarv \
  -o agent_space/marv_val_gfx1101.exe \
  agent_space/marv_validate_gfx1201.cpp

HIP_VISIBLE_DEVICES=1 utils/timeit.sh MMseqs2 test -- \
  agent_space/marv_val_gfx1101.exe
```

Results:
- Test 1: 20-residue query (all 20 standard amino acids). GPU returns
  top hit id=2, score=116 (expected BLOSUM62 self-score). PASS.
- Test 2: 16xAla query. Top hit id=3, score=64 (16 * BLOSUM62[A][A]=4).
  PASS.

GPU Smith-Waterman alignment kernels produce correct BLOSUM62 scores on
gfx1101 RDNA3. The PSSM-based gapless scan path is exercised (same
Marv::scan() call as production ungappedprefilter.cpp). Results are
identical to the gfx1201 validation (same expected scores, same hit ids).
Post-run hipInfo confirms gfx1101 remained healthy (no TDR event).

VERDICT: PASS. State -> completed (validated_sha = 398b7c00).
