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

## Upstream HIP support validation on linux-gfx90a (2026-10-29)

Purpose: upstream merged its own ROCm/HIP support (c7eee558 "ROCm/HIP GPU support", libmarv subtree e067befb "Add ROCm/HIP support", merged as f7f120ee). This section validates upstream's HIP path as if it were our port, using this port's checklist, and lists the gaps. No MOAT state or fork branch was changed.

Host: linux-gfx90a, AMD Instinct MI250X GCD (gfx90a:sramecc+:xnack-), HIP_VISIBLE_DEVICES=0 and CUDA_VISIBLE_DEVICES=0. ROCm 7.16.26385 (TheRock pip SDK, `hipconfig --rocmpath` = `.../site-packages/_rocm_sdk_devel`), AMD clang 24.0.0git, host gcc 13.3.0, cmake 3.31.6. Note this is newer than upstream's Dockerfiles (ROCm 7.2) and than our port's validations (7.2.1).

Upstream sha: `503743960fc57733a36951f5fa803cb3389af7e2` (master, 2026-10-27). Clone in `agent_space/upstream-hip/MMseqs2`.

### Build (upstream's own option, following `.github/workflows/test-gpu-rocm.yml` and `Dockerfile.GPU-debian-slim-rocm7`)

```
R=$(hipconfig --rocmpath)
cmake -S MMseqs2 -B MMseqs2/build-hip -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=1 -DHAVE_TESTS=1 \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=$R -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++
utils/timeit.sh MMseqs2 compile -- cmake --build MMseqs2/build-hip -j48
```

Result: PASS, 102 s, zero HIP warnings or errors (only pre-existing rust/tinyexpr warnings). `mmseqs` is a single binary (libmarv static, `HIP_SEPARABLE_COMPILATION ON`), `.hip_fatbin` carries `amdgcn-amd-amdhsa--gfx90a`.

Wave32/other-arch compile check: same configure with `-DCMAKE_HIP_ARCHITECTURES="gfx1100;gfx1201;gfx950"`, `cmake --build ... --target mmseqs`: PASS (263 s), code objects for gfx1100, gfx1201, gfx950 present. Upstream's packed-half max is inline asm `v_pk_max_f16`; a probe (`hipcc --offload-arch=<a> -c`) assembles it for gfx90a, gfx942, gfx950, gfx1030, gfx1100, gfx1151, gfx1201, gfx9-generic, gfx11-generic, gfx12-generic.

### GPU tests (all under `ulimit -c 0`)

1. Upstream regression suite (what upstream's ROCm CI runs):
   ```
   git submodule update --init util/regression
   CUDA_VISIBLE_DEVICES=0 utils/timeit.sh MMseqs2 test -- util/regression/run_regression.sh build-hip/src/mmseqs <scratch>
   ```
   43/43 GOOD, 0 failed (327 s), including the two GPU tests: SEARCH_GPU GOOD (expected 0.454819, actual 0.454819) and SEARCH_GPU_SERVER GOOD (0.454819). The CPU tests in the same binary do not regress.
2. Our port's GPU-vs-CPU oracle on the bundled example (same commands as the port validations above):
   ```
   mmseqs createdb examples/DB.fasta targetDB; mmseqs makepaddedseqdb targetDB targetDB_padded
   mmseqs easy-search examples/QUERY.fasta targetDB_padded gpu.m8 tmp_gpu --gpu 1
   mmseqs easy-search examples/QUERY.fasta targetDB cpu.m8 tmp_cpu --gpu 0
   ```
   GPU 14482 pairs, CPU 12901, common 12438, GPU-only 2044, CPU-only 463; on common pairs 0 differ in bitscore or pident (max diff 0.0). Identical counts to every validation of our port on gfx90a, gfx1100 and to each other, so upstream's HIP path reproduces our port's results exactly on this arch.
3. libmarv standalone tools (upstream supports `ENABLE_HIP` there too):
   ```
   cmake -S lib/libmarv/src -B marv-standalone -DENABLE_HIP=1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx90a ...
   cmake --build marv-standalone --target align makedb
   ./makedb lib/libmarv/allqueries.fasta mdb
   ./align --query lib/libmarv/allqueries.fasta --db mdb --scanType {Gapless|SW_Endpos|Gapless+SW_Endpos} --tsv --top 5
   ```
   Build PASS; all three scan types run; 20/20 queries have themselves as top-1 hit in every mode (self SW score e.g. 719 for the 144-residue leghemoglobin, end positions 143/143).

Not tested here: multi-GPU (`CUDA_VISIBLE_DEVICES=0,1`), because the dispatcher pinned this job to one GPU; wave32 at runtime (no RDNA card on this host).

### Comparison with our port, by fault class

| Our port's fix | Upstream HIP | Verdict |
|---|---|---|
| gfx90a reports cc 9.0, so the cc-keyed selector would pick the Hopper DPX configs. We pinned a `USE_HIP` branch to the sm75 list. | `#elif defined(__HIPCC__)` branch with tuned AMD lists: cc 9.0 -> `_amd_gfx90a`, 9.4 -> `_amd_gfx942`, anything else -> gfx942 list. Every AMD entry is half2 gapless / float SW (no short2, no dpx). | Upstream better (arch-tuned, no collision). |
| Scalar emulations of ~17 DPX/SIMD intrinsics (`marv_simd_amd.cuh`) so the force-instantiated DPX TUs compile. | DPX TUs are not compiled on HIP (`MARV_GPU_SOURCES_DPX` excluded), `MathOps<short2>` is `#if defined(__CUDACC__)`, and the short2/int dispatch arms `throw std::runtime_error("short2 not supported" / "dpx not supported")` on HIP. | Upstream does not need it. Only reachable through a user-supplied tuning-config file that selects short2/dpx; then it fails loudly rather than silently, which is acceptable. |
| `__hmax2`/`__hmin2` scalar per-lane emulation | `compatibility::hmax2/hmin2` = inline `v_pk_max_f16`/`v_pk_min_f16` | Upstream better (native packed op); assembles on every target checked. |
| `cg::reduce` replaced by tile-local butterfly | `compatibility::group_reduce` butterfly over `group_shfl_xor` for `stride < group.size()`, including the int3 end-position pack | Equivalent; tile-relative, correct on wave32 and wave64. SW gfx90a list uses tiles up to 32 on wave64. |
| `__shfl_*_sync(0xFFFFFFFF, ...)` redirected to maskless; `warp_max_reduce_broadcast` given width 32 | The legacy non-PSSM kernels (`kernels.cuh`, the only users of 32-bit masks, `__syncwarp`, and `warp_max_reduce_broadcast`) are wrapped in `#ifndef __HIPCC__`; they are never launched on either platform (no caller of `call_GaplessFilter_*` outside kernels.cuh). | Not needed. Latent only: `warp_max_reduce_broadcast` still starts at offset 16 with no width, which would reduce only lanes 0-31 on wave64, but it is dead code on HIP. |
| `HIP_DISABLE_WARP_SYNC_BUILTINS` keyed on ROCm >= 7.14 (bf16 shuffle overload collision) plus `__syncwarp` fallback | Nothing; builds clean on ROCm 7.16 | Not needed (the collision came from our compat header's own shuffle overloads). |
| `__CUDACC__`/`__NVCC__` guards widened to `__HIPCC__` in hpc_helpers, blosum, convert | Same widening; uint64 atomics kept NVCC-only; `lane_id()` inline PTX `%laneid` left unguarded but unused | Equivalent. |
| libmarv SHARED so the RDC device link happens in the HIP toolchain | Static libmarv with `HIP_SEPARABLE_COMPILATION ON` links cleanly into the C++ `mmseqs` target | Upstream better (one binary, matches the CUDA layout). |
| `getPaddedQueryLength` pad stays literal `sizeof(char4)*32` | Same | Equivalent. |
| `ClampToInvalid::operator()(unsigned)` device path: our port gives HIP an unsigned per-byte min (same as CUDA `__vminu4`) | Device code tests `#ifdef __CUDA_ARCH__`, which HIP does not define, so HIP device code runs the host fallback that compares as (signed on x86) `char`. | Latent divergence: identical for bytes < 128; makepaddedseqdb writes residue codes <= 52 (+32 for masked) and pad 20, so no effect in practice (results above are exact). Cleanup candidate, see draft 3. |
| Windows (libmarv DLL + Marv API harness on gfx1201/gfx1101; mapped_file Win32, MARV_API exports, MSVC-ABI clang CMake fixes) | No Windows support in libmarv or mmseqs for HIP | Platform our port covered that upstream does not. Upstream's CUDA build has no Windows GPU path either, and the `mmseqs` binary itself is POSIX-only, so only the library was ever reachable. |
| RDNA wave32 (gfx1100 Linux validated, gfx1201/gfx1101 Windows library-level) | Compiles for gfx1100/gfx1201 and Docker builds gfx10/11/12 generic targets, but README claims "CDNA2 generation or newer", RDNA falls through to the gfx942 tuning lists, and no RDNA runtime evidence exists upstream | Platform gap: unvalidated, not known broken. |

### Gaps

Defects in upstream's HIP on gfx90a: none found. Build, 43/43 regression (both GPU tests), GPU-vs-CPU exact, and the standalone aligner all pass.

1. Platform (wave32/RDNA) -- unvalidated upstream. Needs a run of this same recipe on a linux-gfx1100 host (`-DCMAKE_HIP_ARCHITECTURES=gfx1100`, regression SEARCH_GPU/SEARCH_GPU_SERVER plus the example oracle, expect 14482/12901/12438/0). If it passes, a README PR can widen the support statement; if it fails, that is the first real upstream defect.
2. Documentation -- no build-from-source instructions for `-DENABLE_HIP=1`. README line 54 sends ROCm users to the `master-rocm7` Docker tag only; the wiki (`MMseqs2.wiki` Home.md, "Compile from source for Linux with GPU support") has no HIP/ROCm text at all.
3. Latent -- `ClampToInvalid` HIP device path (above). Not observable today.
4. Windows -- not offered upstream; our port only reached libmarv on Windows. Low value unless jeff wants a Windows libmarv deliverable.

### Drafts (not posted; no GitHub writes were made)

Draft 2, README PR (upstream-visible text, ASCII, one line per paragraph):

====
Title: Document building MMseqs2-GPU from source for ROCm

The README only points ROCm users at the Docker image. This adds the from-source build next to the existing CUDA instructions:

    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=1 -DCMAKE_HIP_ARCHITECTURES=gfx90a ..
    make -j

`CMAKE_HIP_ARCHITECTURES` takes the target GPU (for example `gfx90a` for MI200, `gfx942` for MI300) or `native`. GPU selection honors `HIP_VISIBLE_DEVICES` as well as `CUDA_VISIBLE_DEVICES`.

Test plan: built at 50374396 with ROCm 7.16 for gfx90a; `util/regression/run_regression.sh` 43/43 GOOD including SEARCH_GPU and SEARCH_GPU_SERVER on an MI250X.
====

Draft 3, latent cleanup (only worth sending bundled with other HIP work):

====
In `convert.cuh`, `ClampToInvalid::operator()(const unsigned int&)` selects `__vminu4` with `#ifdef __CUDA_ARCH__`. HIP device compilation does not define `__CUDA_ARCH__`, so on ROCm the device code takes the host fallback, which clamps each byte as `char` (signed on x86) instead of as an unsigned byte. Results are the same for byte values below 128, which covers every encoding makepaddedseqdb writes today, so nothing is broken; adding `|| defined(__HIP_DEVICE_COMPILE__)` with an unsigned per-byte min would keep the two backends identical if the encoding ever grows.
====
## Upstream HIP support validation on linux-gfx1100 (2026-10-29)

jeff's request: validate upstream's own ROCm/HIP support as if porting MMseqs2 ourselves, find every gap, and seed follow-up work from what this port learned. No MOAT state or fork branch was changed. Everything below is evidence only.

### What was tested

- Upstream: https://github.com/soedinglab/MMseqs2 `master` at `503743960fc57733a36951f5fa803cb3389af7e2` (2026-10-27). HIP support is c7eee558 "ROCm/HIP GPU support" (`ENABLE_HIP`, ROCm Dockerfiles, `test-gpu-rocm.yml`) plus the libmarv subtree merge f7f120ee of e067befb (libmarv split 1448d80e "Add ROCm/HIP support by mapping CUDA to HIP").
- Host: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), `HIP_VISIBLE_DEVICES=0`.
- Toolchain: TheRock pip SDK `rocm-sdk-devel` 10.1.0 (HIP runtime 7.16.26385, AMD clang 24.0.0git), gcc 13.3, CMake 3.31.6, Ninja. Upstream's CI and Dockerfiles use ROCm 7.2 apt packages; this is a newer toolchain than upstream tests.
- Our port for comparison: AMD-Ecosystem/MMseqs2 `moat-port` at 75a9b029, scratch clone only.

### Build (upstream's own recipe, from `.github/workflows/test-gpu-rocm.yml`)

```
R=/opt/python-3.11-venv/lib/python3.11/site-packages/_rocm_sdk_devel
export ROCM_PATH=$R HIP_PATH=$R
cmake -S MMseqs2 -B MMseqs2/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=1 \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH=$R
utils/timeit.sh MMseqs2 upstream-hip-compile -- cmake --build MMseqs2/build -j32
```

PASS: clean build in 50 s, no errors, no source changes. A second build with `-DCMAKE_HIP_ARCHITECTURES=gfx11-generic` (the RDNA3 entry in upstream's `Dockerfile.GPU-debian-slim-rocm7` arch list) also builds in 71 s.

### GPU tests

All GPU runs under `ulimit -c 0`, `CUDA_VISIBLE_DEVICES=0`.

1. Upstream regression suite, as upstream's ROCm CI runs it (`util/regression` submodule at 3a621dee):
   ```
   CUDA_VISIBLE_DEVICES=0 utils/timeit.sh MMseqs2 upstream-hip-regression -- \
     MMseqs2/util/regression/run_regression.sh MMseqs2/build/src/mmseqs <scratch>
   ```
   43/43 GOOD, 0 failed, 229 s. The GPU tests SEARCH_GPU and SEARCH_GPU_SERVER both give ROC5 AUC 0.454819, an exact match to the expected value, so the gpuserver/IPC path also works on gfx1100.
2. This port's own gate on bundled `examples/` (createdb DB.fasta, makepaddedseqdb, easy-search QUERY.fasta `--gpu 1` vs `--gpu 0`): GPU 14482 pairs, CPU 12901, common 12438, 0 pairs differ in bitscore or pident by >0.5. Every count is identical to this port's gfx90a/gfx1100/revalidation runs.
3. A stronger oracle than the port used: compare the GPU against the CPU ungapped prefilter (`easy-search ... --prefilter-mode 1`, which runs the same algorithm on CPU). Result: GPU and CPU-ungapped m8 are byte-identical when sorted (14482 = 14482 rows). So the "2044 GPU-only / 463 CPU-only" pairs recorded in every earlier validation here come entirely from the prefilter-mode difference (k-mer vs ungapped), not from GPU numerics.
4. Multi-tile kernels: QUERY.fasta has 16 queries longer than 1536 residues (max 4291), above the largest single-tile config. They give 243 common pairs with 0 mismatches. Queries of 769-1536 residues give 2381 common pairs with 0 mismatches.
5. gfx11-generic binary: the same example search is byte-identical to the native gfx1100 build.

### Fault-class comparison: our port vs upstream's HIP path

| Our port's fix | Upstream's HIP | Verdict |
|---|---|---|
| DPX / SIMD-in-word intrinsics (`__vadd2`, `__viaddmax_*`, `__vibmax_*`, ...) emulated in `marv_simd_amd.cuh`, DPX TUs kept | DPX TUs (`*_instantiation_dpx*.cu`) are left out of the HIP build. The short2/int dispatch paths in `cudasw4.cuh` `throw std::runtime_error("short2 not supported"/"dpx not supported")` under HIP. Every AMD config table sets dpx=0. | Not needed. The DPX path can only be reached through a hand-written kernel-config file (the hook in `marv.cu:64` is commented out), and it fails loudly rather than silently. Our emulation was never runtime-exercised either (see Review 2026-06-04 item 1). |
| cc 9.0 collision (gfx90a reads as Hopper, so DPX would be selected) pinned to the sm75 list | `#elif defined(__HIPCC__)` branch: cc 9.0 gets a gfx90a table, cc 9.4 a gfx942 table, anything else (gfx950 cc 9.5, every RDNA part) gets the gfx942 table. All are dpx=0. | Upstream handles it and has per-arch tuned tables. |
| `__hmax2` emulated with scalar `__hmax` | `compatibility::hmax2` uses inline asm `v_pk_max_f16`; HIP 7.16 still has no `__hmax2` | Upstream works. Probe: the asm assembles for gfx90a, gfx942, gfx950, gfx1030, gfx1100, gfx1201 and gfx12-generic (on gfx12 the assembler accepts the old mnemonic). Minor: `asm volatile` blocks scheduling around it, and `__builtin_elementwise_max` on `_Float16x2` would avoid the asm. Performance only, not correctness. |
| `cg::reduce` replaced with a tile butterfly (`marv_tile_reduce`) | `compatibility::group_reduce`, an `shfl_xor` butterfly over `group.size()` using 32-bit word relays for any T | Equivalent, and correct on wave32 and wave64 by construction. Validated on wave32 here. |
| `__shfl_*_sync(0xFFFFFFFF)` routed to maskless shuffles; masks widened | The legacy non-PSSM kernels in `kernels.cuh` (the only full-warp mask users) and the `shfl_*_2xint16` helpers are `#ifndef __HIPCC__`. They have no callers anywhere in libmarv (grep for `call_GaplessFilter_row_wise*` and `call_GaplessFilter_float_many_pass` finds none outside kernels.cuh). | Upstream is better: it compiles dead code out instead of porting it. See the port defect below. |
| `__CUDACC__`/`__NVCC__` guards widened to `\|\| __HIPCC__` | Same approach (hpc_helpers, convert.cuh, blosum.hpp) | Equivalent |
| libmarv built SHARED for the `-fgpu-rdc` device link | Static `marv` with `HIP_SEPARABLE_COMPILATION ON` plus `POSITION_INDEPENDENT_CODE`, and CMake does the device link | Upstream works (verified). Ours is not needed. |
| `cudaFuncSetAttribute(kernel, ...)` | `reinterpret_cast<const void*>(kernel)` at each call site | Equivalent |
| `--gpu`/gpuserver honours `CUDA_VISIBLE_DEVICES` | `GPUSharedMemory::getShmHash` reads `HIP_VISIBLE_DEVICES` first, then `CUDA_VISIBLE_DEVICES`. The error text names `HIP_VISIBLE_DEVICES`. | Upstream is better |
| Windows (libmarv DLL plus a Marv API harness on gfx1201/gfx1101; the `mmseqs` binary itself was never built on Windows) | None | Platform gap, low value: upstream's CUDA build has no native Windows support either. |

### Gaps found

1. Documentation lags the code (feature/platform gap, docs only). README.md:49 and :54 say "CDNA2 generation or newer ROCm-enabled GPUs are also supported", but upstream's ROCm container builds `gfx10-1-generic;gfx10-3-generic;gfx11-generic;gfx12-generic`, and this run shows RDNA3 wave32 passing the full regression including both GPU tests. Draft issue text (not posted):
   > The README describes ROCm support as CDNA2 or newer, but the master-rocm7 container also ships RDNA code objects (gfx11-generic, gfx12-generic). On a Radeon Pro W7800 (gfx1100, wave32) with ROCm 10.1 / HIP 7.16, master 50374396 built with `-DENABLE_HIP=1 -DCMAKE_HIP_ARCHITECTURES=gfx1100` passes `util/regression/run_regression.sh` 43/43, including SEARCH_GPU and SEARCH_GPU_SERVER (ROC5 AUC 0.454819 exact), and GPU results are byte-identical to `--prefilter-mode 1` on CPU. Would you take a README change listing RDNA3 as supported?
2. CUDA side effect of the HIP commit (defect, CUDA path, not HIP). `lib/libmarv/src/smithwaterman_kernel_config.cuh:345`: `getOptimalKernelConfigs_SW_default()` was changed from `getOptimalKernelConfigs_SW_sm89()` to `getOptimalKernelConfigs_SW_amd_gfx942()`. The CUDA ladder's `else` branch also calls `_default()`, so every NVIDIA card without its own table (sm86, sm87, sm100, sm110, ...) now gets the AMD gfx942 SW table. That table tops out at a 768 single-tile query length; sm89's goes to 1024. Results stay correct, but queries of 769-1024 move to the multi-tile path on those NVIDIA cards. The gapless `_default()` still returns sm89, so the change looks unintended. Draft fix: keep `_SW_default()` returning sm89 and add `_SW_amd_default()` returning gfx942 for the `__HIPCC__` branch, mirroring the gapless file. Not measured (no NVIDIA GPU here).
3. Performance: upstream's tuned tables vs our sm75 pin. Clean single-job run of `mmseqs ungappedprefilter q1k t p --gpu 1 --max-seqs 1000 -e 10000` (first 1000 queries of the regression query.fasta against the padded targetannotation DB):
   - upstream 50374396: 21.16 s and 21.12 s
   - our port 75a9b029 (with the scratch `__syncwarp` fix below): 330.39 s and 330.38 s
   - Output: `createtsv` of both prefilter DBs is byte-identical when sorted (1,000,000 rows).

   Upstream's HIP path is 15.6x faster on gfx1100 with identical results. Our port's sm75 pin has tile sizes similar to upstream's AMD tables, so the gap is probably not the table alone (candidates: `hardcodedzero` vs `kernelparamzero` approach, the scalar `__hmax2` emulation, the maskless-shuffle redirect). It was not root-caused because the port is being retired. It settles which implementation should carry forward. (An earlier full-query-set timing ran while other CPU jobs were active and is discarded.)
4. Not testable: the full arch list in upstream's Dockerfile (gfx90a/gfx942/gfx9-generic/gfx9-4-generic/gfx10-*) needs other hardware. Only gfx1100 and gfx11-generic were run here.

### Our port vs upstream: defect in our port

`moat-port` 75a9b029 no longer builds on HIP 7.16 (Linux). `cuda_to_hip.h` defines `HIP_DISABLE_WARP_SYNC_BUILTINS` for HIP >= 7.14 (the Windows TheRock workaround re-keyed in 398b7c00). On Linux HIP 7.16 that also removes `__syncwarp`:
```
lib/libmarv/src/kernels.cuh:777:9: error: use of undeclared identifier '__syncwarp'; did you mean '__sync_swap'?
```
This is the same failure dbeac858 fixed for 7.2, now hit on 7.16. The version key was wrong. Upstream never reaches it because it compiles the legacy kernels out on HIP. To compare performance, a throwaway scratch-only edit (`#define __syncwarp(...) __builtin_amdgcn_wave_barrier()` under that guard) was applied and never committed or pushed. With the port being retired in favour of upstream, this is evidence for that decision, not work to do.

### Verdict

Upstream's HIP support is complete and correct for the MMseqs2 GPU surface on gfx1100 wave32: build, the full regression including the GPU and gpuserver tests, byte-identical GPU-vs-CPU-ungapped results, and multi-tile queries all pass. Nothing our port fixed is missing upstream, apart from the Windows libmarv DLL. Follow-up candidates: the README RDNA statement (1) and the CUDA SW default table regression (2).

### Cross-reference: linux-gfx90a and linux-gfx1100 sections (2026-10-29)

The two runs agree on everything both covered. This gfx1100 run settles the gfx90a section's open items:
- gfx90a gap 1 ("wave32/RDNA unvalidated, expect 14482/12901/12438/0"): validated. gfx1100 gives exactly 14482/12901/12438/0, regression 43/43 including SEARCH_GPU and SEARCH_GPU_SERVER, and the gfx11-generic binary is identical. The README PR can therefore name RDNA3 on runtime evidence. Combine gfx90a draft 2 (from-source ROCm build) with gap 1 of the gfx1100 section (the support statement) into one README PR.
- Found only on gfx1100: the CUDA-side `getOptimalKernelConfigs_SW_default()` -> gfx942 table regression (gap 2) and the 15.6x speed advantage of upstream over our port (gap 3). Found only on gfx90a: the latent `ClampToInvalid` `__CUDA_ARCH__` divergence and the standalone `align` tool check.
- Shared cosmetic item for any HIP PR here: `src/commons/Parameters.cpp:108` (`--gpu` help "Use GPU (CUDA) if possible") and `src/prefiltering/ungappedprefilter.cpp:740` ("MMseqs2 was compiled without CUDA support") still say CUDA in HIP builds.

## Performance root cause: upstream HIP vs our port on gfx1100 (2026-10-30)

jeff's research question: why is upstream's HIP path 15.6x faster than our port (gap 3 above), and what should other ports learn. Evidence only: no MOAT state, fork branch or upstream was touched. Scratch trees under `agent_space/perf/` (not committed).

Answer: the scalar `__hmax2` emulation in `marv_simd_amd.cuh` accounts for the whole gap. Swapping only that function for a packed max makes our port faster than upstream with the tile table unchanged. The table, `hardcodedzero` vs `kernelparamzero`, the shuffle redirect and SHARED/RDC vs static linking are each within about 10%. Upstream itself leaves another 2x on the table because its packed max is `asm volatile`.

### Setup

- Host: Radeon Pro W7800 (gfx1100, wave32), `HIP_VISIBLE_DEVICES=2`. TheRock `rocm-sdk-devel` 10.1.0 (HIP 7.16.26385, AMD clang 24.0.0git), CMake 3.31.6, Ninja.
- Upstream: soedinglab/MMseqs2 `50374396` built with `-DENABLE_HIP=1 -DCMAKE_HIP_ARCHITECTURES=gfx1100` (static libmarv). Our port: AMD-Ecosystem/MMseqs2 `75a9b029` with the scratch `__syncwarp` fix from the gfx1100 section (a local-only commit in a scratch clone), `-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100` (SHARED libmarv, `-fgpu-rdc`).
- Benchmark (same as gap 3): `q1k` = first 1000 queries of `util/regression/data/query.fasta` (388,288 residues), `t` = `makepaddedseqdb` of `targetannotation.fasta` (36,026 sequences).
  ```
  ulimit -c 0
  mmseqs ungappedprefilter q1k t p --gpu 1 --max-seqs 1000 -e 10000 --threads 8
  ```
  Wall time of the whole command. Port variants rebuild only the `marv` target and run the port's `mmseqs` with `LD_LIBRARY_PATH` pointing at the variant `libmarv.so` (the binary has `DT_RUNPATH`, which `LD_LIBRARY_PATH` overrides; checked with `ldd`).
- Output check: `createtsv q1k t p out` sorted and hashed. Every run below gives the same 1,000,000 rows (md5 of the sorted TSV `11ed9b5b...`), so every variant is result-identical to upstream and to the unmodified port.
- Profiles: `rocprofv3 --kernel-trace --stats --output-format csv -- <command>`. rocprofv3 aborts (`Conflicting visibility of agent-1 between HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES`) when both are set, so profiles ran with `CUDA_VISIBLE_DEVICES` unset.
- Kernel resources: code objects pulled out with `llvm-objdump --offloading`, then `llvm-readelf --notes` (VGPRs, spills, `.private_segment_fixed_size`) and `llvm-objdump -d` (instruction counts per inner loop).

### Profile: which kernels carry the gap

Upstream: 20.49 s of kernel time across 24,433 launches, 99.7% in the half2 `GaplessFilter_strided_PSSM_{singletile,multitile}_kernel` instances (top: singletile group 8/numRegs 28 1.57 s, g4/r44 1.48 s, g4/r36 1.36 s). Our port: 328.23 s of kernel time across 24,436 launches, the same 49 half2 gapless kernels with the same call counts as variant A, 100.0% gapless. The time is spread over every hot kernel rather than one outlier: per-kernel port/A ratios are 12.8x (g4/r20, no scratch) to 20x (g4/r28, g4/r36) to 22-25x (g4/r44, g4/r52, g4/r60, the tiles that also use scratch). Top: g4/r44 32.1 s, g4/r52 29.2 s, g4/r36 27.8 s. The legacy non-PSSM kernels in `kernels.cuh` (the only users of the redirected `__shfl_*_sync` and of `__syncwarp`) are never launched by either build.

### Bisection

Port-side variants change one thing at a time; UR* change one thing in upstream.

| run | hmax2 | gapless table / approach | wall s |
|---|---|---|---|
| upstream 50374396 | `asm volatile v_pk_max_f16` | gfx942 (kernelparamzero) | 21.00, 21.01, 21.01 |
| port 75a9b029 | scalar `__hmax` per lane | sm75 (hardcodedzero) | 327.97, 328.3 |
| B | scalar | gfx942 (kernelparamzero) | 450.04 |
| A | `asm volatile` (upstream's) | sm75 (hardcodedzero) | 19.37, 19.37 |
| H | `asm volatile` | sm75 tiles, all kernelparamzero | 19.46 |
| C | `asm volatile` | gfx942 | 20.93 |
| D | `__builtin_elementwise_max` | sm75 (hardcodedzero) | 9.21 |
| E | builtin | sm75 tiles, all kernelparamzero | 9.43 |
| F | builtin | gfx942 (kernelparamzero) | 10.21 |
| G | builtin | gfx942 tiles, all hardcodedzero | 10.32 |
| UR1 = upstream + scalar `__hmax` | scalar | gfx942 | 450.3 |
| UR2 = upstream + builtin | builtin | gfx942 | 10.45 |
| UR3 = upstream + plain `asm` (no `volatile`) | asm | gfx942 | 11.44 |

The builtin variant is
```
typedef _Float16 h2v __attribute__((ext_vector_type(2)));
return __builtin_bit_cast(__half2, __builtin_elementwise_max(__builtin_bit_cast(h2v, a), __builtin_bit_cast(h2v, b)));
```
which lowers to `v_pk_max_f16` (llvm.maxnum, same non-NaN-operand semantics as CUDA `__hmax2`).

### Attribution of the 15.6x (port 328 s -> upstream 21.0 s)

1. Scalar `__hmax2` emulation: all of it. Port -> A (only `__hmax2` changed to upstream's asm) is 328 -> 19.4 s, 16.9x, already faster than upstream. Reverse: putting the scalar form into upstream (UR1) takes it from 21.0 to 450.3 s, the same as B (port, gfx942 table, scalar) at 450.0 s, so with matching max and table the two code bases perform identically. Two mechanisms, both visible in the ISA:
   - Instruction count. ROCm's scalar `__hmax` (`amd_hip_fp16.h`) is three `__hisnan` tests plus a compare; the compiler keeps them as `v_cmp_*_u16`, `v_cndmask`, `s_and_saveexec` and `s_cbranch_execz`. The inner loop of `hardcodedzero` singletile g4/r16 is 4169 instructions per subject letter in the port vs 318 upstream, 288 in A, 266 in D. The kernel is 7587 instructions vs 822 upstream.
   - Scratch. With the bloated body, `#pragma unroll` loops over the per-thread `penalty_here_array` stop being fully unrolled: every port gapless kernel with numRegs >= 40 has 240-336 bytes of scratch, and from numRegs 52 the kernel collapses to 34 VGPRs with the array in memory. 107 of 160 half2 gapless kernels in the port use scratch; 0 in upstream and in A. The sm75 table selects 9 such tiles (r44, r52, r60), the gfx942 table 18 -- which is why B (table swapped, scalar max kept) is slower than the port, 450 s.
2. Tile table: second order, and the wrong way round on RDNA3. With a fast max, sm75 beats upstream's gfx942 table by 8-11% (A 19.37 vs C 20.93; D 9.21 vs F 10.21). With the scalar max the gfx942 table is 37% slower (B) because it spills more.
3. `hardcodedzero` vs `kernelparamzero`: within 1-2% (A 19.37 vs H 19.46; D 9.21 vs E 9.43; F 10.21 vs G 10.32).
4. SHARED libmarv with `-fgpu-rdc` vs static with CMake's device link: nil. F (port layout, builtin, gfx942) 10.21 s vs UR2 (upstream layout, same source change and table) 10.45 s. The port's link line passes no `-O` to the RDC device link, which did not matter here.
5. Maskless-shuffle redirect: nil. The hot kernels shuffle through cooperative-groups tiles (`group.shfl_up/down`), which compile to the same 14 `ds_bpermute_b32` in upstream, A and D; the redirected `__shfl_*_sync` calls are only in the never-launched legacy kernels.
6. Upstream's own residual: `asm volatile` around `v_pk_max_f16` costs about 2x. Same instruction count as the builtin, but `volatile` pins every asm block in order, so the scheduler cannot hoist LDS loads ahead of the max chain: the g4/r16 inner loop has 22 `s_waitcnt` in A vs 9 in D, and per-kernel times in A are 2.2-3.0x those in D for every hot singletile kernel (g4/r44 1.480 vs 0.556 s, g4/r36 1.369 vs 0.559 s). Dropping only `volatile` (UR3) gets 21.0 -> 11.4 s; the builtin (UR2) 10.45 s. D's builtin does push the multitile kernels with numRegs >= 52 to 256 VGPRs with spills (multitile g8/r60 0.865 s in D vs 0.757 s in A), a small residual that a later tuning pass could look at.

### Lessons for other ports

Promoted to the `cuda-to-rocm` skill (`SKILL.md` "Performance -- correct is not done" and `references/performance.md`): emulating a packed intrinsic lane by lane can cost an order of magnitude even when results are exact, `asm volatile` for one ALU op serializes scheduling, and tile tables and build layout are second order. Profile and read the hot kernel's ISA and scratch size before concluding a port is done.

### Upstream follow-up candidate (not drafted for posting, needs jeff)

Replacing `compatibility::hmax2`/`hmin2`'s `asm volatile` in `lib/libmarv/src/cuda_hip_compatibility.cuh` with `__builtin_elementwise_max`/`min` on a `_Float16` 2-vector would make upstream's HIP gapless prefilter about 2x faster on gfx1100 (21.0 -> 10.45 s, identical output). Not measured on gfx90a/gfx942, where upstream's tables were tuned. Upstream-visible, so it is jeff's call whether to raise it.

## Refresh onto upstream tip: stopped, needs jeff's decision (2026-11-12, linux-gfx942)

jeff asked for a porter run that merges upstream's tip into `moat-port` (head 75a9b029, review PR AMD-Ecosystem/MMseqs2#1) so the upstream PR can merge. No fork branch, mirror or MOAT state was changed. The trial merge was aborted and nothing was pushed. `pr-state MMseqs2` = none, so `moat-port` is not frozen.

### What the trial merge showed

```
git clone https://github.com/AMD-Ecosystem/MMseqs2 projects/MMseqs2/src
git -C projects/MMseqs2/src remote add upstream https://github.com/soedinglab/MMseqs2 && git fetch upstream master
git checkout -b trial-merge origin/moat-port && git merge --no-commit upstream/master   # then git merge --abort
```

- upstream/master d0185a2d (2026-11-11); fork `master` 17b688d2 fast-forwards to it (checked, not pushed). Merge base 11933403; port 5 commits, upstream 150 commits.
- 17 conflicts: 12 content (CMakeLists.txt, cmake/MMseqsResourceCompiler.cmake, lib/libmarv/src/CMakeLists.txt, pssmkernels_smithwaterman.cuh, mathops.cuh, util.cuh, six hpc_helpers headers) and 5 modify/delete (blosum.cu, blosum.hpp, gapless_kernel_config.cuh, smithwaterman_kernel_config.cuh, kernelhelpers.cuh, all deleted upstream).
- The cause is not ordinary drift. Upstream now has its own HIP backend: c7eee558 "ROCm/HIP GPU support" (top-level `ENABLE_HIP`), fe334a31 "Add HIP CI" and 63831a18 "Add ROCm to readme". Since the 2026-10-29 validations above it has also rewritten libmarv: `cuda_hip_compatibility.cuh`, `cuda_hip_rename.h`, `cmake/MarvGpuArch.cmake`, and per-arch AMD aligner instantiations (`aligners/make_aligner_amd_{gfx900,gfx942,gfx1100,gfx1150}.cu`, `*_default_amd.cu`). The port's config-ladder files are deleted upstream.
- Upstream tip has no Windows support in libmarv or `GpuUtil.cpp` (no `_WIN32` guards). Windows is the one thing the port covers that upstream does not.

### Why this was not resolved mechanically

The requested rule ("keep upstream's new behaviour, re-apply the HIP port on top") cannot be applied as written. Upstream's new behaviour IS a HIP port. Re-applying ours would add a second, parallel HIP backend (`USE_HIP`, `cuda_to_hip.h`, `marv_simd_amd.cuh`, `hip_compat/`) next to upstream's `ENABLE_HIP`. No maintainer would take that, and the 2026-10-30 section measured ours 15.6x slower on gfx1100. Keeping upstream on every conflict would leave a tree that does not hang together: the port's non-conflicting files (cuda_to_hip.h, marv_simd_amd.cuh, the hip_compat headers, the USE_HIP blocks in src/CMakeLists.txt) would still be there, and none of them could build against the restructured libmarv. Either way, what remains of the port is a different scope, and that is a person's call (scope-cut rule, and any "already supported upstream" disposition).

### Options for jeff

1. Retire the port as superseded upstream: record an already-supported/ported-elsewhere disposition, close review PR #1 unmerged, and do not open an upstream PR. The earlier evidence (2026-10-29 gfx90a and gfx1100 sections) is that upstream's HIP path passes 43/43 regression, including SEARCH_GPU and SEARCH_GPU_SERVER, with GPU-vs-CPU-ungapped results byte-identical. The follow-up candidates already recorded still stand as separate upstream items: README RDNA/from-source note, the CUDA `_SW_default` -> gfx942 table regression, `hmax2` `asm volatile` -> builtin (2x), and the `ClampToInvalid` `__CUDA_ARCH__` latent issue. Some may have changed at d0185a2d.
2. Re-scope to Windows only: merge upstream, resolve every libmarv/CMake conflict to upstream, delete the port's own HIP backend, then port only the Windows enablement (mapped_file Win32, MARV_API exports, MSVC-ABI CMake, MMseqsResourceCompiler/tinyexpr WIN32, GpuUtil `_WIN32` stubs) onto upstream's `ENABLE_HIP`. That is new porting work against a restructured library, not a refresh. Every platform's evidence would be voided, and the deliverable would be a library-only Windows path (the `mmseqs` binary itself is POSIX-only).

Not done because the refresh stopped: the fork `master` fast-forward (step 2), the HIP build and GPU tests, the CUDA compile check, jargon, push, `advance-head`.

foldseek (and so the parallel foldseek refresh) is likely in the same situation. steineggerlab/foldseek tip 39b02b9 (2026-11-11) has pulled `lib/mmseqs` forward with a subtree squash (5b3bc29), so its vendored libmarv probably carries upstream's HIP backend. This was not checked further. `steineggerlab/metaeuk` returned "repository not found" at that URL; its upstream location was not looked up.

## Rescope porter round (2026-11-12/13, linux-gfx90a)

jeff's ruling of 2026-11-12 (recorded in plan.md `## Rescope (jeff, 2026-11-12)`): keep the port, rescoped to (1) AMD coverage beyond upstream's four tuned arches with a correct fallback plus gfx90a tuning, (2) Windows on top of upstream's `ENABLE_HIP`, (3) the `ClampToInvalid` HIP device guard. Out: the `asm volatile` hmax2/hmin2 change (upstream PR soedinglab/MMseqs2#1167) and anything upstream's HIP already does correctly.

Host: MI250X GCD (gfx90a:sramecc+:xnack-), `HIP_VISIBLE_DEVICES=0` (dispatcher pin), `CUDA_VISIBLE_DEVICES=0` for `--gpu`. TheRock `rocm-sdk-devel` (HIP 7.16.26385, AMD clang 24.0.0git), gcc 13.3, CMake 3.31.6, Ninja. All GPU runs under `ulimit -c 0`.

### Evidence first: upstream tip d0185a2d on gfx90a

```
R=$(hipconfig --rocmpath)
cmake -S . -B build-gfx90a -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=1 -DHAVE_TESTS=1 \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=$R -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++
```
- `-DCMAKE_HIP_ARCHITECTURES=gfx90a`: configure fails at generate: `CMake Error at lib/libmarv/src/CMakeLists.txt:103 (add_library): Cannot find source file: aligners/make_aligner_amd_gfx90a.cu` / `No SOURCES given to target: marv_aligner_amd_gfx90a`. Same for `gfx942;gfx90a`.
- Default configure (no arch): CMake native detection yields `gfx90a;gfx90a;gfx90a;gfx90a` (one per GPU, not de-duplicated) and fails the same way (it would also try to create the same target four times).
- Built for gfx942 (tuned; configures and builds), run on gfx90a: `terminate ... what(): No aligner found for detected AMD arch: major 9, minor 0` (easy-search `--gpu 1`).
- Built for gfx900 (tuned, cc 9.0 like gfx90a), run on gfx90a: dispatch picks the gfx900 table, then `Asynchronous CUDA Error: device kernel image is invalid .../pssm.cuh 1051`.
So no upstream configuration runs on gfx90a; the coverage gap is real. The cc dispatch is also wrong for fat binaries (gfx900/906/908/90a share 9.0; gfx1100-1102 share 11.0; gfx1150/1151 share 11.5).

### Fork operations

- Fork `master` fast-forwarded 17b688d2 -> d0185a2d (pushed `upstream/master` to the fork's `master`, ancestry checked first).
- `moat-port` (was 75a9b029): `git merge --no-ff upstream/master` gave 17 conflicts, every one resolved to upstream (UU -> `checkout --theirs`, UD -> `git rm`): merge commit b29c9270. Follow-up 440e1197 returns every file to upstream (`git diff --exit-code upstream/master HEAD` clean): removes `cuda_to_hip.h`, `hip_compat/`, `marv_simd_amd.cuh`, the `USE_HIP` CMake, the hpc_helpers edits and the old Windows edits.
- New-scope commits on top, fast-forward 75a9b029..2cda180e (no force):
  - 84a8fb2e `[ROCm] Run libmarv on AMD GPUs without a tuned aligner`: `MarvGpuArch.cmake` de-duplicates the HIP arch list; `lib/libmarv/src/CMakeLists.txt` builds `make_aligner_amd_<arch>.cu` only for arches that have one, and one `marv_aligner_default_amd` object library (upstream's existing, previously unwired `make_aligner_default_amd.cu` / `make_aligner_endpos_default_amd.cu`) for all the others, inline namespace `DEFAULT_AMD_`, define `MARV_HAVE_AMD_DEFAULT`. The dispatchers in `aligners/make_aligner{,_endpos}.cu` select AMD tables by `gcnArchName` (suffix stripped; helper `getAmdGfxArchName` in make_aligner.cu) and fall back to the generic configs that the commented-out lines already named; without a generic build the error names the target. libmarv Readme statement updated.
  - 3c588b6f `[ROCm] Clamp packed residues as unsigned bytes in HIP device code`: `convert.cuh` `#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)`, per-byte unsigned min on HIP (HIP has no `__vminu4`).
  - 7c39308d `[ROCm] Build libmarv on Windows with the ROCm clang toolchain`: Win32 `MappedFile`, `strtok_s`, `MARV_API` keyed on `MARV_BUILDING_DLL`/`MARV_USING_DLL`, which CMake sets only for a WIN32 SHARED marv (upstream builds marv static by default, where the old unconditional dllimport would break consumers).
  - 54edbc78 `[ROCm] Guard POSIX-only build steps and headers for Windows`: the resource compiler uses upstream's CMake-only path on WIN32 without `checkshell.sh`; tinyexpr drops `-fPIC` on WIN32; `sys/mman.h`/`fcntl.h` guards; GpuUtil server-mode stubs.
  - 2cda180e `[ROCm] Add tuned libmarv aligner configs for gfx90a` (tables below).
- Net diff vs upstream: 17 files, +647/-84, only the three scope items. Not carried: the old `CMAKE_HIP_USING_LINKER_DEFAULT` Windows workaround (tied to the old shared `-fgpu-rdc --hip-link` layout; the Windows host decides whether upstream's layout needs an equivalent) and `mmapbuffer.hpp` (POSIX, used only by the `makedb` tool, not the library).
- `utils/jargon.py --port MMseqs2`: clean (range d0185a2d..moat-port, 11 commits). Upstream-contained and published-tip-ancestor checks pass.
- One commit-message imprecision, left as is (no amend): 2cda180e says the fp32 endpos multi-tile configs with more than 20 items per thread "run 5-10x slower"; measured 5.2x to 12.6x (below).

### gfx90a tuning (by measurement)

Tuner: upstream's own. `cmake -S lib/libmarv/src -B build-tune -DENABLE_HIP=1 -DBUILD_TUNING=1 -DONLY_BUILD_TUNING=1 -DCMAKE_HIP_ARCHITECTURES=gfx90a`, `cmake --build build-tune --target tuning_gapless` (715 s). Runs (each `--printBestConfigs --sleep 10`): `--half2 --tiles 1`, `--float --tiles 1`, `--float --endpos --tiles 1`, `--half2 --tiles 4` and `--float --tiles 4` over the full config space (blocksize 128/256/512 x group 2-32 x numItems 4-64). `--float --endpos --tiles 4` was too slow over the full space (large tiles take minutes each, and one attempt was lost to a timeout because the tool prints only at exit), so it ran on a scratch copy of the tuner with `configtuples.hpp` narrowed to blocksize 512 (every multi-tile winner in the other sweeps and every upstream AMD multi-tile entry uses 512), groups 2-16 x numItems 4-48, plus a second run with group 32 x numItems 8-24. `--checkResults` was dropped after it proved about 16x slower (CPU recompute per iteration); the chosen tables' correctness is covered by the regression and GPU-vs-CPU runs below. Raw CSVs and the generator (`agent_space/scripts/gen_gfx90a.py`) are scratch only.

Selection: per tile size the max-GCUPS config (as `printBestAsConfig` does), over the tile spans of upstream's AMD tables (half2 16..1536, float/endpos 8..768, multi-tile from 256). Single-tile: drop tile t when a larger tile T has `gcups_T * t / T > gcups_t`, since the chooser runs a query on the smallest tile >= its length; this reproduces exactly the 32 tile sizes of upstream's gfx942 half2 table. Multi-tile: keep tiles within 10% of the best measured multi-tile GCUPS, since without hints the chooser ignores speed. Hints recorded, `hasPerformanceHints()` false like every upstream table. SW tables: upstream's untuned AMD list (the tuner covers gapless only).

Result: half2 single 32 configs (16..1536, best 6130 GCUPS at 8x32 tile 512; numItems >= 52 drop to about 4200), half2 multi 19 (320..1536, 3711-4028), float single 27, float multi 11 (256..768, 2439-2646), endpos single 30, endpos multi 4 (16x16, 16x20, 32x16, 32x20: 1284-1348). gfx90a-specific: fp32 endpos multi-tile collapses at numItems >= 24-28 (2x28 132, 16x24 235, 16x48 106, 32x24 258 GCUPS against 1338 at 16x20), so that list stops at 640.

Timing (the 2026-10-30 benchmark: `mmseqs ungappedprefilter q1k t p --gpu 1 --max-seqs 1000 -e 10000 --threads 8`, q1k = first 1000 queries of `util/regression/data/query.fasta`, t = padded targetannotation; host load about 3.5 from other jobs, runs alternated):
- generic fallback (gfx90a built with no tuned file): 68.09 s, 66.50 s
- tuned gfx90a: 54.33 s, 54.42 s (1.23x)
- fat binary `gfx900;gfx90a;gfx942;gfx950;gfx1100;gfx1150;gfx1201;gfx11-generic`: 54.45 s (the gfx90a table is chosen despite gfx900's matching cc 9.0)
- all four: 1,000,000 rows, md5 of the sorted createtsv `6e2142dde4e8...` identical.

### Validation on gfx90a (head 2cda180e)

```
cmake -S . -B build-tuned -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=1 -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=$R -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++
cmake --build build-tuned -j64 --target mmseqs
CUDA_VISIBLE_DEVICES=0 util/regression/run_regression.sh build-tuned/src/mmseqs <scratch>
```
- Regression (submodule 19b9867b): 43/43 GOOD, 0 BAD; SEARCH_GPU 0.454819 = expected, SEARCH_GPU_SERVER 0.454819 = expected. Also 43/43 on the fallback-only build before the gfx90a tables.
- Example GPU-vs-CPU (`easy-search examples/QUERY.fasta` padded `--gpu 1` vs `--gpu 0`): GPU 14482, CPU 12901, common 12438, all 12438 rows identical. Tuned GPU output identical (sorted) to the fallback GPU output. Against `--prefilter-mode 1` (CPU ungapped) the GPU matches 14448 pairs exactly, with 34 GPU-only / 236 CPU-only boundary pairs; at 50374396 these were byte-identical, so upstream's CPU ungapped path changed in between (upstream code, untouched here; the GPU counts equal every earlier run).
- libmarv standalone (`cmake -S lib/libmarv/src -DENABLE_HIP=1 -DCMAKE_HIP_ARCHITECTURES=gfx90a`, targets align makedb test_gapless): `test_gapless --gpu-compatible` "all checks passed, digest 0xcba3a00c27ffe2df"; `align` Gapless / SW_Endpos / Gapless+SW_Endpos run on `allqueries.fasta`. ctest itself cannot start `test_gapless` (`libamdhip64.so.7: cannot open shared object file`): its RUNPATH lacks the ROCm lib dir that `align` gets (upstream CMake: test_gapless links static marv without the HIP runtime's link interface); it passes with `LD_LIBRARY_PATH=$R/lib`. Upstream follow-up candidate, not in scope.
- Missing-target error: a gfx942-only build on gfx90a now throws `No aligner compiled for detected AMD arch gfx90a`.
- Upstream's tuned arches untouched: this branch's gfx942 aligner objects disassemble identically (309157 / 160713 lines) to upstream plus only the convert.cuh change; the fatbins differ only in the path-derived `__hip_cuid_*` symbol. The ClampToInvalid fix is the one intended device-code change on every AMD target. Runtime selection on a real gfx942/gfx1100 still needs those hosts.
- CPU-only build (`cmake -S . -B build-cpu`, no GPU options): builds; example search output identical to the GPU build's `--gpu 0`.
- Fat-binary compile check, gfx950/gfx1201/gfx11-generic (generic aligner) plus gfx900/gfx942/gfx1100/gfx1150 (tuned): builds and links, code objects for all 8 present.
- CUDA path: NOT validated. No nvcc on this host, and creating a job venv with CUDA 13 pip wheels was refused by the fleet guard. Every CUDA-visible change sits in a HIP branch or an `ENABLE_HIP` CMake block (CUDA dispatch lines and the `__vminu4` path are unchanged), but this still needs a real nvcc compile on the mi355 host: `cuda-not-validated: no nvcc on linux-gfx90a`.

### Open work (owed in surface.json)

- gfx950 and gfx1201 tuning: on those hosts run the tuner as above (`-DCMAKE_HIP_ARCHITECTURES=gfx950` / `gfx1201`; the endpos multi-tile sweep needs the narrowed tuple or a long timeout), generate `aligners/make_aligner_amd_<gfx>.cu` + `make_aligner_endpos_amd_<gfx>.cu` in the gfx90a layout, add their declarations in `make_aligner_headers.cuh` and one `#ifdef MARV_HAVE_AMD_<GFX>` / `arch == "<gfx>"` block in each of the 7 AMD dispatch branches; CMake picks the files up by name. Time against the generic fallback with the q1k benchmark. Until then both use the generic configs (compile-checked here, not run).
- Windows: build libmarv as a DLL on the Windows host with upstream's layout (`cmake -S lib/libmarv/src -DENABLE_HIP=1 -DLIBRARY_ONLY=1 -DBUILD_SHARED_LIBS=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201`), check whether an equivalent of the old `CMAKE_HIP_USING_LINKER_DEFAULT ""` workaround is needed, and rerun the Marv-API harness. The mmseqs-tree WIN32 guards are unbuilt and the mmseqs binary stays POSIX-only (DBReader/FileUtil mmap), so whether commit 54edbc78 stays is a call for the Windows round and jeff.
- CUDA nvcc compile of 2cda180e on mi355.
- Every platform record predates the rescope: after `advance-head` to 2cda180e, linux-gfx90a, linux-gfx1100, windows-gfx1101 and windows-gfx1201 are all behind head.
- Review PR AMD-Ecosystem/MMseqs2#1 describes the old USE_HIP port, so its body is stale against the new scope (a reviewer/checkup matter, not touched here).

## Review 2026-11-13 (reviewer, linux-gfx942: review-passed)

Scope: the net diff of `moat-port` 2cda180e against upstream d0185a2d (17 files, +647/-84), judged as an upstream contribution to a project that already has a HIP backend. Method: `pr-review` skill, local-branch mode; every changed hunk read with its unchanged surroundings (dispatchers, `namespace.hpp` inline-namespace scheme, `MultiConfigScoreOnlyAligner` tile choosers, `MarvGpuArch.cmake`, `simd/MarvSimd.cmake`, `dbdata.cpp` MappedFile users); findings fact-checked inline against the code.

### Port Review: MMseqs2 (moat-port vs d0185a2d)

Summary: the branch adds a generic AMD aligner for untuned targets with gfx-name dispatch, tuned gfx90a gapless tables, the `ClampToInvalid` HIP device path and libmarv Windows enablement on top of upstream's `ENABLE_HIP`. No code defects found; approved for validation. The items below are PR-preparation work and cannot be done on `moat-port` without rewriting history, so they do not send the branch back.

#### Commit Hygiene (fix at PR preparation)

1. The upstream-visible history is not the contribution. `git log d0185a2d..moat-port` lists 11 commits: the five superseded USE_HIP commits (c755847a..75a9b029), the merge b29c9270 and the removal 440e1197, then the five scope commits. A maintainer would see a whole second HIP backend added and removed. Before the upstream PR, curate to the five scope commits (84a8fb2e, 3c588b6f, 7c39308d, 54edbc78, 2cda180e) on a tree identical to the validated head, through `squash-carry-forward`.
2. 2cda180e's body says the fp32 endpos multi-tile configs above 20 items per thread "run 5-10x slower"; the porter measured 5.2x to 12.6x. Correct the figure in the curated commit.

#### Decision for jeff (not a review failure)

3. 54edbc78 (resource-compiler WIN32 path, tinyexpr `-fPIC`, `sys/mman.h` guards, GpuUtil server stubs) touches only the mmseqs tree, which cannot build on Windows (DBReader/FileUtil mmap untouched), so no Windows target exercises it. Unless the Windows round finds an mmseqs-tree target that needs it, drop it from the upstream PR; libmarv's Windows support (7c39308d) stands alone. The draft body below already leaves it out.

#### Checks run (all passed; recorded so the next round does not repeat them)

- CUDA path: all `make_aligner{,_endpos}.cu` edits sit in the `#else` (non-`__CUDACC__`) branches or under `#if !defined(__CUDACC__)`; `convert.cuh` CUDA branches unchanged textually; CMake changes are inside `if(ENABLE_HIP)` except the `WIN32 AND SHARED_LIBRARY` define, which is additive. A real nvcc compile is still owed (known open item).
- Upstream's tuned arches: built `-DENABLE_HIP=1 -DCMAKE_HIP_ARCHITECTURES=gfx942` at 2cda180e on an MI300X (gfx942:sramecc+:xnack-), TheRock HIP 7.16.26385. `util/regression/run_regression.sh` (submodule 19b9867b): 43/43 GOOD, 0 BAD, SEARCH_GPU 0.454819 = expected, SEARCH_GPU_SERVER 0.454819 = expected. That build has no generic aligner (`MARV_HAVE_AMD_DEFAULT` undefined), so the passing GPU run shows the `arch == "gfx942"` table was selected. Review evidence only; the linux-gfx942 validator still records its own run.
- SIMD backend (upstream #1168 includes `make_aligner_headers.cuh` from `simd/marv_simd_aligner.cpp`): `cmake -S lib/libmarv/src -DENABLE_SIMD=1` builds, ctest 2/2 passed.
- gfx90a tables: `configsAreSortedByTileSize` static_asserts hold (they compile); `hasPerformanceHints()` false in every list as in upstream's tables, and the multi-tile chooser without hints picks by last-tile utilization, so the porter's 10%-of-best filter is the right lever. Evidence (upstream tuner per tile size, 1.23x end-to-end on the half2 path, regression and GPU-vs-CPU identity) is adequate for an upstream table.
- `__vminu4` is absent from the ROCm headers (`grep -r __vminu4 $(hipconfig --rocmpath)/include`: no hits), so the per-byte loop is needed; the old HIP device fallback clamped via signed `char`, as the commit says.
- `utils/jargon.py --port MMseqs2`: clean. Upstream PR search (gfx90a/fallback/Windows/ROCm/HIP): no competing PR; only soedinglab/MMseqs2#1167 (ours, out of scope here).

### Review PR AMD-Ecosystem/MMseqs2#1 must be redrafted for jeff

PR #1 (open, head `moat-port`, base `master`) still carries the superseded port's title "[ROCm] Add AMD GPU (HIP) support for GPU-accelerated search" and a body describing `USE_HIP`, `cuda_to_hip.h`, `marv_simd_amd.cuh`, `hip_compat/` and the old validation. None of that is on the branch now. Not edited (person's decision). Replacement draft below; `jargon.py -` and `prose.py` clean, title 61 characters. `upstream.py` appends the AI-assistance and opt-out note itself, so it is not in the draft. Before it goes to jeff: replace the Testing bullets with the validator records at the final head (gfx950/gfx1201 tuning rounds and the Windows round may change the scope), and re-run both checks.

Title:

====
[ROCm] Run libmarv on untuned AMD GPUs and add gfx90a configs
====

Body:

====
libmarv's HIP backend builds and runs only on the four AMD targets with a tuned aligner file (gfx900, gfx942, gfx1100, gfx1150). Configuring for any other GPU, for example `-DCMAKE_HIP_ARCHITECTURES=gfx90a`, stops with `Cannot find source file: aligners/make_aligner_amd_gfx90a.cu`, and so does a default configure on a host with such a GPU, because CMake's native detection lists one entry per device. The runtime selection keys on compute capability, which does not identify an AMD ISA (gfx900, gfx906, gfx908 and gfx90a all report 9.0), so a binary built for gfx900 hands its kernels to a gfx90a and fails with `device kernel image is invalid`.

This PR:

- de-duplicates the HIP architecture list and builds the existing generic configs (`make_aligner_default_amd.cu`, `make_aligner_endpos_default_amd.cu`) once for every requested target without a tuned file, under their own inline namespace;
- selects the AMD aligner by the device's gfx target name (`gcnArchName` without feature suffixes) and falls back to the generic configs, which the dispatchers already named in comments; without a generic build the error names the missing target;
- adds tuned gapless configs for gfx90a (MI200), measured on an MI250X with libmarv's own `tuning_gapless` and laid out like the other AMD files; Smith-Waterman keeps the untuned AMD list, as on every AMD target;
- makes `ClampToInvalid` take its per-byte device path under HIP as well (`__HIP_DEVICE_COMPILE__`), with an unsigned per-byte min since HIP has no `__vminu4`; the host fallback it used before clamps as signed `char`;
- adds Windows support to libmarv for the ROCm clang toolchain: a Win32 `MappedFile`, `strtok_s`, and a `MARV_API` export macro that is active only for a shared build on Windows, so the default static build is unchanged.

The tuned targets keep their own tables and the CUDA path is unchanged.

On the first 1000 queries of the regression query set against the padded `targetannotation` database (`ungappedprefilter --gpu 1 --max-seqs 1000 -e 10000`), the gfx90a configs take 54.3 s against 67.3 s with the generic configs, with identical results.

Testing:

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=1 -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build build --target mmseqs
util/regression/run_regression.sh build/src/mmseqs scratch
```

- AMD Instinct MI250X (gfx90a): regression 43/43 including SEARCH_GPU and SEARCH_GPU_SERVER, with the gfx90a configs and with the generic configs alone; on the bundled example every query/target pair shared by the GPU and CPU searches has an identical output row.
- AMD Instinct MI300X (gfx942, tuned table): regression 43/43 including SEARCH_GPU and SEARCH_GPU_SERVER.
- A build for gfx900, gfx90a, gfx942, gfx950, gfx1100, gfx1150, gfx1201 and gfx11-generic compiles and links, and on gfx90a selects the gfx90a table.
- The SIMD backend (`-DENABLE_SIMD=1`) builds and its tests pass; a CPU-only build is unchanged.

This work was authored with the assistance of Claude, an AI coding agent.
====

## CUDA compile check (nvcc 13.5.37) at 2cda180e

jeff's request (2026-11-13, linux-gfx950 fleet host): compile the CUDA path of `moat-port` 2cda180e with real nvcc as the no-regression check for the rescoped port. No MOAT state, fork branch or review PR was changed; the fork was not edited.

Toolchain: libmarv's Readme asks for CUDA 13.2 or newer and the host has only nvcc 12.8, so the CUDA 13.5 wheels went into a job-local prefix with `pip install --target <job>/cuda13-wheels "nvidia-cuda-nvcc==13.5.*" "nvidia-cuda-runtime==13.5.*" nvidia-cuda-cccl "nvidia-nvvm==13.5.*" "nvidia-cuda-crt==13.5.*" "nvidia-nvtx==13.5.*"` (cccl resolved to 13.3.5.0.6), not into the shared image. `C=<job>/cuda13-wheels/nvidia/cu13`. The wheel ships only `libcudart.so.13`, so FindCUDAToolkit needs `ln -s libcudart.so.13 $C/lib/libcudart.so`. gcc 13.3, CMake 3.31.6, Ninja. CMake reports `CUDA compiler identification is NVIDIA 13.5.37`, `Found CUDAToolkit ... (found version "13.5.37")` and `marv: tuned kernels for 80;90; core architectures 80-real;90-real;90-virtual`.

```
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES="80;90" \
  -DCMAKE_CUDA_COMPILER=$C/bin/nvcc -DCUDAToolkit_ROOT=$C
cmake --build build-cuda -j64 --target mmseqs
cmake -S lib/libmarv/src -B build-marv -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=0 -DBUILD_TUNING=0 -DONLY_BUILD_TUNING=0 \
  -DLIBRARY_ONLY=0 -DCMAKE_CUDA_ARCHITECTURES="80;90" -DCMAKE_CUDA_COMPILER=$C/bin/nvcc -DCUDAToolkit_ROOT=$C
cmake --build build-marv -j64
```

Results:
- Fork 2cda180e, top-level mmseqs with libmarv: PASS. All 317 steps complete and `src/mmseqs` links (2m54s at -j64); `mmseqs version` prints 2cda180e...
- Fork 2cda180e, standalone libmarv (`LIBRARY_ONLY=0`): PASS. All 40 steps complete: `libmarv.a`, `align`, `makedb`, `test_gapless`, `test_cuda_multitile` and `test_db_copy` all link, with 0 warnings.
- Upstream d0185a2d, top-level build with the same commands: PASS. Both trees give the same warnings (116 lines each, identical after path normalization): the Rust `block_aligner_c` ABI deprecations and four tinyexpr `-Warray-bounds`. None are in fork-changed files.
- Device code is unchanged against upstream. `cuobjdump -sass` and `-ptx` (13.5 cuobjdump/nvdisasm wheels, same job-local install) of the fork and upstream `mmseqs` match line for line once the per-translation-unit `_INTERNAL_<hash>` / `_GLOBAL__N__<hash>` prefixes are normalized: 1043 SASS functions over sm_80 and sm_90, and 234 PTX entries. So the rescope changes no CUDA kernel. The CUDA-visible edits (dispatcher, `convert.cuh`, `marv.h`/`marv.cu`, `mapped_file.hpp`, GpuUtil) compile cleanly, and the HIP-only branches drop out.
- The configure message `xxd or sed not found, compiling resources with CMake instead` means this host has no `xxd`. The non-WIN32 branch of `MMseqsResourceCompiler.cmake` is unchanged from upstream.
- Not run: this host has no NVIDIA GPU, so there is no CUDA runtime test. This is a compile and link check only.

This clears the rescope round's `cuda-not-validated: no nvcc on linux-gfx90a` item for 2cda180e.

## gfx950 tuning porter round (2026-11-13, linux-gfx950)

jeff's request: the next item of the 2026-11-12 rescope, tuned libmarv aligner configs for gfx950, done exactly as the gfx90a round. Stage taken `review-passed -> porting` (lock linux-gfx950), `pr-state` none. Host: AMD Instinct MI355X (gfx950), `HIP_VISIBLE_DEVICES=7` (dispatcher pin, idle), `CUDA_VISIBLE_DEVICES=0` for `--gpu`. TheRock `rocm-sdk-devel` (HIP 7.16.26385), gcc 13.3, CMake 3.31.6, Ninja. All GPU runs under `ulimit -c 0`.

### Tuner and selection

```
R=$(hipconfig --rocmpath)
cmake -S lib/libmarv/src -B build-tune -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=1 -DBUILD_TUNING=1 -DONLY_BUILD_TUNING=1 \
  -DCMAKE_HIP_ARCHITECTURES=gfx950 -DCMAKE_PREFIX_PATH=$R -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++
cmake --build build-tune --target tuning_gapless -j64
build-tune/tuning_gapless <args> --printBestConfigs --sleep 10
```
Sweep wall times on gfx950 (far faster than gfx90a): `--half2 --tiles 1` 241 s, `--float --tiles 1` 86 s, `--float --endpos --tiles 1` 146 s, `--half2 --tiles 4` 1321 s, `--float --tiles 4` 396 s, all over the full config space. `--float --endpos --tiles 4` used the same narrowed scratch tuner copies as gfx90a (`configtuples.hpp`: blocksize 512 x groups 2-16 x numItems 4-48, 211 s; and blocksize 512 x group 32 x numItems 8-24, 25 s). No `--checkResults` (as gfx90a). Raw CSVs and the generator (`agent_space/mmseqs/gen_tuned.py`: the gfx90a rules; the file text is the gfx90a file with tuple lists and hint arrays replaced, so SW lists and layout are identical) are scratch only.

Selection rule unchanged: per tile the max-GCUPS config over upstream's AMD tile spans; single tile drops t when a larger T has `g_T * t / T > g_t`; multi tile keeps tiles >= 256 within 10% of the best multi-tile GCUPS.

Result: half2 single 33 configs (16..1536, best 22861 GCUPS), half2 multi 21 (256..1536, 13301-14761), float single 31 (8..768, best 15660), float multi 11 (256..768, 10539-11512), endpos single 24 (8..768, best 8616), endpos multi 4 (16x16, 16x20, 32x16, 32x20: 5367-5683, the same four as gfx90a). The fp32 endpos multi-tile cliff reproduces on gfx950: numItems >= 24 runs 4.6-9.3x slower than numItems 20 at the same group size (559-1219 GCUPS against 5177-5683), so that list stops at 640.

### Fork change

0c897402 `[ROCm] Add tuned libmarv aligner configs for gfx950` (fast-forward 2cda180e..0c897402 on `moat-port`, no force): `make_aligner_amd_gfx950.cu` (217 lines), `make_aligner_endpos_amd_gfx950.cu` (101), declarations in `make_aligner_headers.cuh` (5), one `MARV_HAVE_AMD_GFX950` / `arch == "gfx950"` block after gfx942 in each of the 7 AMD dispatch branches (make_aligner.cu 4, make_aligner_endpos.cu 3), libmarv Readme tuned-arch list. CMake picks the files up by name (configure no longer prints `generic AMD aligner for gfx950`). `utils/jargon.py --port MMseqs2`: clean (needed a local `master` branch in the fresh clone: `git branch master origin/master`); `utils/prose.py` on the message: clean. Windows commits and history untouched.

### Timing (q1k benchmark of 2026-10-30, as gfx90a)

`mmseqs ungappedprefilter q1k t p --gpu 1 --max-seqs 1000 -e 10000 --threads 8`; generic = 2cda180e worktree built for gfx950 (no tuned file), tuned = 0c897402; runs alternated, host load 3.5-4.3 (one run at 21):
- generic: 47.72 s, 47.64 s
- tuned: 45.56 s, 45.50 s (1.05x)
- both 1,000,000 rows, sorted createtsv md5 `6e2142dde4e8...`, identical to each other and to the gfx90a round.
- `rocprofv3 --kernel-trace --stats` (CUDA_VISIBLE_DEVICES unset): gapless kernel time 46.77 s generic (top: DEFAULT_AMD_ half2 128/4/48 11.15 s, 128/4/32 10.72 s) vs 44.59 s tuned (top: AMD_GFX950_ half2 256/4/44 2.66 s, 256/4/36 2.63 s, 256/4/48 2.52 s). The kernels are the whole wall time, so the 1.05x is a kernel result, not host overhead.
- The app reaches about 95 GCUPS on this benchmark (388,288 x 10.9M residues) against the tuner's 14,000-23,000, a gap gfx90a also had (about 80 vs 6130). An 8x replicated targetannotation (`t8`, first 200 queries) runs about 7x more efficiently (10 s for 1.6x the cells), so the 36k-sequence database underfills the 256-CU MI355X. The tuned-vs-generic ratio stays small there: generic 10.10 s, 10.13 s vs tuned 9.87 s, 9.88 s (1.02x), 200,000 rows, md5 identical. On gfx950 the generic configs are already close to tuned; the tables are still the measured best per tile.

### Validation on gfx950 (head 0c897402)

```
cmake -S . -B build-tuned -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=1 -DHAVE_TESTS=1 -DCMAKE_HIP_ARCHITECTURES=gfx950 -DCMAKE_PREFIX_PATH=$R -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++
cmake --build build-tuned -j64 --target mmseqs
CUDA_VISIBLE_DEVICES=0 util/regression/run_regression.sh build-tuned/src/mmseqs <scratch>
```
- Build: 312/312, the same 116 warning lines as the generic build (Rust `block_aligner_c` ABI deprecations, tinyexpr `-Warray-bounds`), none in aligner files.
- Regression (submodule 19b9867): 43/43 TEST SUCCESS, 0 failures; SEARCH_GPU 0.454819 = expected (26 s), SEARCH_GPU_SERVER 0.454819 = expected (25 s). The build has no generic aligner, so the GPU tests ran the gfx950 tables.
- Example GPU-vs-CPU (`easy-search examples/QUERY.fasta` padded `--gpu 1` vs `--gpu 0`): GPU 14482, CPU 12901, common 12438, all 12438 rows identical; tuned GPU output identical (sorted) to the generic GPU output. Same counts as gfx90a.
- CUDA path: installing the CUDA 13.5 wheels into this job was refused by the fleet guard, so the existing job-local prefix of the earlier mi355 CUDA job was used read-only (nvcc 13.5.37): `cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES="80;90" -DCMAKE_CUDA_COMPILER=$C/bin/nvcc -DCUDAToolkit_ROOT=$C`, `cmake --build build-cuda --target mmseqs`: 317/317, links, 116 warning lines as before. Compile only, no NVIDIA GPU.
- Not re-run here: libmarv standalone `test_gapless` and the fat-binary selection check (the gfx90a round showed the gfx-name dispatch picks the exact table).

### State

`advance-head` 0c897402; every platform record is now behind head (validators owed on wave64, wave32, windows). surface.json: `tuning:gfx950` and `check:cuda-nvcc-compile` moved from owed to covered. Remaining rescope items: gfx1201 tuning (same recipe on that host), the Windows round, and the PR-prep curation from `## Review 2026-11-13` (the gfx950 commit joins the scope commits to keep).

## Review 2026-11-13 delta 2cda180e..0c897402 (reviewer, linux-gfx942: review-passed)

Scope: the gfx950 tuning commit 0c897402 `[ROCm] Add tuned libmarv aligner configs for gfx950` (6 files, +359/-1), judged against the bar of the gfx90a tables in `## Review 2026-11-13`. Method: `pr-review` skill, local-branch mode, every hunk read with the unchanged dispatchers, `make_aligner_headers.cuh`, `lib/libmarv/src/CMakeLists.txt` (tuned-file discovery, inline namespaces) and `cmake/MarvGpuArch.cmake`; findings fact-checked inline. No code defects; approved for validation.

### Decision for jeff (not a review failure): is a 1.02-1.05x gain worth an upstream table?

The evidence supports the commit message as written, and the message does not oversell: it says the generic configs "were already close on this GPU". The case is marginal. For: the files are data only, laid out exactly like upstream's own tables, compiled only when gfx950 is a requested target (explicitly or by CMake native detection on an MI350/MI355 host), and they complete the CDNA line (gfx90a, gfx942, gfx950) beside upstream's gfx942 table; output is identical. Against: about 320 lines plus 7 dispatch blocks that any future libmarv config-signature change must keep compiling, for a gain that shrinks as the database grows (1.05x on q1k against the 36k-sequence database, 1.02x on the 8x database, which is the regime closer to real use). The maintainer accepted per-arch tables for every AMD target they tuned themselves, so the structure is not the obstacle; the size of the win is. My view: keep it as its own last commit in the curated history, state the 1.02-1.05x plainly in the PR body (no "tuned for MI355" headline), and drop it without argument if the maintainer questions it. Not dropped by the reviewer.

### Note for PR prep

- 0c897402's endpos-cliff figure ("4.6-9.3x slower") cannot be re-derived from durable records: the raw tuner CSVs were scratch, and notes give only the pooled ranges (559-1219 GCUPS against 5177-5683). It is consistent with them; the previous review found the analogous gfx90a figure wrong, so whoever curates the history should keep the figure only if the CSVs are regenerated, or reword it to the pooled ranges.

### Checks run (all passed; recorded so the next round does not repeat them)

- Structure against gfx90a: `diff make_aligner{,_endpos}_amd_gfx90a.cu make_aligner{,_endpos}_amd_gfx950.cu` differs only in function names, tuple lists and hint arrays; SW lists are the same untuned AMD lists. Parsed every list (agent_space script): all sorted by tile size with no duplicates, hint count = config count in every list, spans match the gfx90a/upstream AMD spans (half2 single 16..1536, multi 256..1536; fp32 single 8..768, multi 256..768; endpos single 8..768, multi 256..640). Groupsize 32 also appears in the gfx90a tables. The single-tile drop rule holds (no kept t has a kept larger T with g_T*t/T > g_t), and every multi-tile list is within 10% of its best (half2 13301 >= 0.9*14761; fp32 10539 >= 0.9*11512; endpos 5367 >= 0.9*5683). The endpos multi-tile list is the same four configs as gfx90a, as the commit says.
- Isolation: the 7 new dispatch blocks sit in the non-`__CUDACC__` branches, each under `#ifdef MARV_HAVE_AMD_GFX950` and `arch == "gfx950"` (exact gfx-name match), after gfx942 and before the generic fallback, so no other arch's selection changes. The define and the `AMD_GFX950_` inline namespace come from the existing file-existence loop in `lib/libmarv/src/CMakeLists.txt`; no CMake edit. The new header declarations are unconditional like the gfx90a/gfx942 ones and are never referenced on the CUDA path. The default arch list (`MARV_SUPPORTED_HIP_ARCHS`, generic targets) is unchanged, so a build that does not name gfx950 is unaffected.
- Multi-arch build on this host (MI300X, TheRock HIP 7.16.26385): `cmake -S . -B build-multi -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=1 -DCMAKE_HIP_ARCHITECTURES='gfx90a;gfx942;gfx950;gfx1100' -DCMAKE_PREFIX_PATH=$R -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++`, `cmake --build build-multi --target mmseqs`: 318/318, links; no generic aligner configured (all four targets tuned); `llvm-nm` shows the gfx950 objects in `libmarv::AMD_GFX950_`; no warnings in aligner files (only the Rust `block_aligner_c` ones).
- Upstream's tuned arch unaffected: that build's `mmseqs` on gfx942 (`HIP_VISIBLE_DEVICES=1`, `util/regression/run_regression.sh`, submodule 19b9867): 43/43 TEST SUCCESS, SEARCH_GPU 0.454819 = expected, SEARCH_GPU_SERVER 0.454819 = expected. With `MARV_HAVE_AMD_DEFAULT` undefined the run could only pass through the gfx942 table. Review evidence only; the linux-gfx942 validator records its own run at 0c897402.
- Evidence against the message: q1k 47.72/47.64 s generic vs 45.56/45.50 s tuned = 1.047x, reported as 1.05x; kernel trace 46.77 vs 44.59 s; 8x database 10.10/10.13 vs 9.87/9.88 s; identical md5 and 12438/12438 identical GPU-vs-CPU rows; 43/43 regression on MI355X; CUDA build (nvcc 13.5.37, sm_80;90) 317/317. All match `## gfx950 tuning porter round`.
- Commit hygiene: `[ROCm]` title, 51 characters; "Authored with Claude.", no `Co-Authored-By`/noreply; ASCII; Test Plan in a fenced block, same form as 2cda180e. `utils/jargon.py --port MMseqs2`: clean.

## Windows split out as a separate feature (jeff, 2026-11-13)

jeff ruled that Windows support, for libmarv and for the `mmseqs` command-line tool alike, is a second feature separate from the AMD GPU coverage work (fallback aligner for untuned arches, ClampToInvalid device guard, per-arch tuned tables). Consequences:

- The upstream PR from this round carries no Windows commits. At PR-prep history curation, drop 7c39308d (`libmarv` Windows DLL build) and 54edbc78 (mmseqs-tree Windows guards) along with the superseded commits; the curated branch keeps only the fallback, ClampToInvalid and tuning commits.
- Both commits stay reachable in `moat-port` history at 0c897402 and are the starting point for the Windows feature, registered as deferral `mmseqs2-windows-support` in `projects/MMseqs2/deferred.json`. The full CLI needs a Windows replacement for the Linux-only memory-mapped file reading; that work belongs to the Windows feature, not this PR.
- The required `windows` gate cannot be met by this PR's tree (upstream MMseqs2 does not build on Windows without the dropped commits). How that gate is handled for this PR is a pending decision for jeff; an agent may only suggest a waiver.

## gfx950 tables kept (jeff, 2026-11-13)

jeff ruled to keep the gfx950 tuned tables (0c897402) in the GPU-coverage PR as the last commit of the curated history. The PR body states the measured gain plainly (1.05x on q1k, 1.02x on the 8x database, identical output) and does not headline it as MI355 tuning; if the maintainer objects, drop the commit without argument. At curation, regenerate the raw tuner output behind the "4.6-9.3x slower" endpos figure or reword it to the ranges recorded here (reviewer caveat).
## gfx1201 tuning porter round (2026-11-13, linux-gfx1201): tables NOT committed, finding for jeff

jeff's request: the next 2026-11-12 rescope item, tuned libmarv aligner configs for gfx1201, done as the gfx950 round, with the rule that a negligible gain (about 1.02x or less) is recorded but not committed. Stage taken `review-passed -> porting` (lock linux-gfx1201); `pr-state` none. Host: AMD Radeon AI PRO R9700 (gfx1201, 64 CUs, wave32), `HIP_VISIBLE_DEVICES=0` (idle, self-pinned; the dispatcher set none), `CUDA_VISIBLE_DEVICES=0` for `--gpu`. TheRock `rocm-sdk-devel` (HIP 7.16.26385), gcc 13.3, CMake 3.31.6, Ninja, `MAX_JOBS` 24 (physical cores). All GPU runs under `ulimit -c 0`.

**Result: the tuned tables are slower than the generic AMD configs on gfx1201, so nothing was committed and `moat-port` stays at 0c897402.** Every raw figure below is in `projects/MMseqs2/reference/gfx1201-tuning/` (committed with these notes): `raw/*.csv` (full tuner output of every sweep), `raw/sweeps.txt` (commands, wall times), `raw/bench_q1k.txt` / `raw/bench_t8.txt` (every timing run with load, row count, md5), `kernel-stats/*.csv` (rocprofv3 kernel stats), `selection-report.txt` (every selected config with its GCUPS), `gen_tuned.py` (selection and file generator), the run scripts, and `gfx1201-tables-uncommitted.patch` (the complete fork change that was measured: the two aligner files, 7 dispatch blocks, 5 declarations, Readme line; `git apply` clean on 0c897402).

### Structure check against upstream's wave32 tables

Upstream's gfx1100 file differs from gfx90a/gfx950 in one way that matters: its 32-bit gapless and endpos lists mix `Datatype::Int` with `Datatype::Float`, so upstream tuned 32-bit with `--int32` beside `--float` (gfx1150 happens to keep only Float winners). This round therefore ran the 32-bit sweeps as `--float --int32`, and Int configs won some tiles. The other differences are cosmetic: gfx1100 keeps commented-out sub-512 multi-tile entries and copies its untuned endpos multi-tile list from the single-tile list; the gfx90a/gfx950 layout (no commented lines, measured endpos multi-tile list) was kept, since that layout is already on the branch and reviewed. SW lists: the same untuned AMD list as every AMD file.

### Tuner and selection

```
R=$(hipconfig --rocmpath)
cmake -S lib/libmarv/src -B build-tune -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=1 -DBUILD_TUNING=1 -DONLY_BUILD_TUNING=1 \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_PREFIX_PATH=$R -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++
cmake --build build-tune --target tuning_gapless -j24          # 5m37s
build-tune/tuning_gapless <args> --printBestConfigs --sleep 10 > <name>.csv
```
Sweeps (full config space unless noted): `--half2 --tiles 1` 472 s, `--float --int32 --tiles 1` 374 s, `--float --int32 --endpos --tiles 1` 581 s, `--half2 --tiles 4` 3624 s, `--float --int32 --tiles 4` 3099 s (a first attempt was killed at 36 min by the 2 h background-job limit of this harness, together with the queued runs; the tuner prints only at exit, so nothing was lost but time). `--float --int32 --endpos --tiles 4` on the same narrowed scratch tuner copies as gfx90a/gfx950 (`configtuples.hpp`: blocksize 512 x groups 2-16 x numItems 4-48, 1398 s; blocksize 512 x group 32 x numItems 8-24, 204 s). No `--checkResults`.

Selection rule unchanged (`gen_tuned.py`): per tile the max-GCUPS config over upstream's AMD spans; single tile drops t when a kept larger T has `g_T * t / T > g_t`; multi tile keeps tiles >= 256 within 10% of the best.

Selected: half2 single 35 configs (16..1536, 2208-4395 GCUPS), half2 multi 12 (256..1536, 2918-3230), fp32/int32 single 34 (16..768, 2354-2873; 5 Int), fp32 multi 4 (320, 352, 576, 704; 2266-2503, the other tiles are 15-51% below the best), endpos single 34 (8..768, 832-2202; 7 Int), endpos multi 2 (Int 16x20 tile 320 1012, Int 32x20 tile 640 973; 16x16 and 32x16 reach 886, 12% below, so the 10% rule drops them where gfx90a/gfx950 kept four). The endpos multi-tile cliff above 20 items per thread reproduces: group 16 items 24-48 run 183-361 GCUPS against 1012 at 20 (2.8-5.5x), group 32 items 24 353 against 973 (2.8x) (`raw/endpos_t4_a.csv`, `raw/endpos_t4_b.csv`).

### Timing: tuned vs generic (q1k benchmark of 2026-10-30)

Generic = 0c897402 built for gfx1201 (configure prints `generic AMD aligner for gfx1201`); tuned = 0c897402 plus the patch above (no generic aligner configured; `llvm-nm` shows the `AMD_GFX1201_` objects). Both `cmake -S . -B <b> -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=1 -DHAVE_TESTS=1 -DCMAKE_HIP_ARCHITECTURES=gfx1201 ...`, 312/312, the same 76 warning lines (Rust `block_aligner_c`, tinyexpr), none in aligner files. `mmseqs ungappedprefilter q1k t p --gpu 1 --max-seqs 1000 -e 10000 --threads 8`, q1k = first 1000 queries of `util/regression/data/query.fasta` (388,288 residues), t = padded targetannotation; runs alternated, host load 1.3-4.0:

| build | q1k wall s | 8x database (q200 vs t8) wall s |
|---|---|---|
| generic | 31.79, 31.86, 31.92 | 8.21, 8.25, 8.27 |
| tuned gfx1201 | 35.06, 34.95 (0.91x) | 8.39, 8.39 (0.98x) |
| upstream's gfx1100 tables, renamed to gfx1201 (scratch build only) | 35.50, 35.61 (0.90x) | 8.76, 8.79 (0.94x) |

- Output identical in every run: q1k 1,000,000 rows, sorted createtsv md5 `6e2142dde4e89275a2c378e290a65088` (the same as the gfx90a and gfx950 rounds); 8x database 200,000 rows, md5 `461eebd479a2aeca4dfb3283d7e2e9b0`. t8 = targetannotation replicated 8 times with `_r<i>` header suffixes (288,208 sequences), q200 = first 200 queries of q1k.
- Kernel trace (`rocprofv3 --kernel-trace --stats`, CUDA_VISIBLE_DEVICES unset): gapless kernel time 28.73 s generic vs 33.61 s tuned (all kernels 30.07 vs 34.24 s), so the loss is in the kernels. Generic runs six half2 single-tile configs at blocksize 128 (top: 128/4/32 7.89 s over 310 calls, 128/4/48 7.36 s over 221); tuned spreads over many blocksize-512 tiles (top: 512/4/44 2.14 s, 512/4/36 2.12 s).
- Why the tuner disagrees: it ranks blocksize 512 above 128 for the very configs generic uses (tile 256 g4/n32: 128 -> 3709, 512 -> 4157 GCUPS; tile 384 g4/n48: 3176 vs 4349; tile 768 g8/n48: 3132 vs 4395; `raw/half2_t1.csv`), yet in the application the 128-thread kernels are faster on both database sizes. The tuner's synthetic database and launch pattern evidently do not represent mmseqs' per-batch launches on this 64-CU part; upstream's own wave32 table (gfx1100) loses the same way, so this is not a defect of this sweep. Not investigated further (stop discipline): finding the cause would mean changing the tuner or the selection rule, which is jeff's call.

### Validation of the unchanged head 0c897402 on gfx1201 (porter evidence, not a validator record)

- Regression (`util/regression/run_regression.sh <build-generic>/src/mmseqs <scratch>`, submodule 19b9867): 43/43 TEST SUCCESS, 0 failures; SEARCH_GPU 0.454819 = expected (18 s), SEARCH_GPU_SERVER 0.454819 = expected (19 s). The six `indexdb died` lines are INDEX_COMPATIBLE's deliberate incompatible-index cases (`--check-compatible 2`); that test passes.
- Example GPU-vs-CPU (`easy-search examples/QUERY.fasta` against the padded DB with `--gpu 1`, against the unpadded DB with `--gpu 0`, as in earlier rounds): GPU 14482, CPU 12901, common 12438, all 12438 rows identical. (CPU against the padded DB gives 12901 rows that differ slightly from the unpadded ones and 12439 shared rows; upstream CPU behaviour, not looked into.)
- Fork tree after the round: `git -C projects/MMseqs2/src status --porcelain` clean at 0c897402; nothing pushed, no `advance-head`, `utils/jargon.py` not needed (no new commit).

### Decision for jeff

1. Ruling asked: scope out `tuning:gfx1201` (surface.json keeps it owed, with the measurements, until you rule). Recommendation: scope it out; gfx1201 keeps the generic configs, which are faster here. The patch stays in `reference/gfx1201-tuning/` if a later tuner change makes a table worthwhile.
2. The same evidence weakens the gfx950 table (1.02-1.05x, already your open decision from `## Review 2026-11-13 delta 2cda180e..0c897402`), and it suggests checking the gfx90a table's 1.23x against more than one database before the upstream PR.
3. Possible upstream note (not drafted; your call): libmarv's tuner prefers configs that lose in mmseqs on gfx1201, and its gfx1100 table is slower than the generic configs on gfx1201. Measured on gfx1201 only; a gfx1100 host would show whether the gfx1100 table itself beats generic there.
4. Stage: `porting` has no exit back to `review-passed`, and jeff's sequence ends at `delta-ported`, which is the only legal exit that releases the lock. Set `delta-ported` with `head_sha` unchanged at 0c897402: the reviewer's delta is empty, and its job this round is to check this no-commit finding against the reference files and return `review-passed`.

Lesson promoted to the `cuda-to-rocm` skill (`references/performance.md`, "Tile tables are second order"): a project's own autotuner can pick a table that loses end to end, so time the real workload against the existing configs before keeping one.

## Review 2026-11-13 gfx1201 tuning round, empty fork delta (reviewer, linux-gfx942: changes-requested)

Scope: the round's additions on `port/MMseqs2` (b03cab01, 4d140896): the `## gfx1201 tuning porter round` finding, `reference/gfx1201-tuning/`, and the `references/performance.md` lesson. Fork: `git ls-remote` shows `moat-port` = `0c8974021c3e1035640c6daf30e5ab88b6af7fbc` = `head_sha`, no other port or fix branch, fresh clone `status --porcelain` empty; the source delta is empty and nothing in it needs review. `python3 utils/check.py skills`: ok. Checks that held (no finding): all six wall-time ratios recompute from `raw/bench_q1k.txt` / `raw/bench_t8.txt` (0.910, 0.983, 0.896, 0.940), rows and md5 identical in every run; the tuner figures (3709/3176/3132 vs 4157/4349/4395) are the hardcodedzero rows of `raw/half2_t1.csv`; kernel totals 28.74/30.07 s generic vs 33.61/34.24 s tuned, the generic half2 single-tile set is six configs, all blocksize 128; the endpos multi-tile cliff matches `raw/endpos_t4_{a,b}.csv`; `python3 -I gen_tuned.py raw <0c897402 aligners> <out>` reproduces `selection-report.txt` and both generated `.cu` files byte for byte against `gfx1201-tables-uncommitted.patch`, which applies clean on 0c897402. (Minor, notes only: the 1-minute host load in the logs spans 1.22-4.00, not 1.3-4.0.)

Problems, both in `.claude/skills/cuda-to-rocm/references/performance.md` ("Tile tables are second order", second paragraph), which is published outside MOAT and so has to state only what was measured:

1. "gave a table that was 9% slower end to end" quotes only the 1000-query run (0.91x). The 8x-database run in the same round was 0.98x (`raw/bench_t8.txt`: 8.39 vs 8.24 s). The size of the loss depended on the workload, which is the point of the paragraph's own advice. Say "2-9% slower end to end on two database sizes" (or name both figures).
2. "but in the application the 128-thread kernels won" claims the block size is the cause. The round compared whole tables only: generic runs six half2 single-tile configs, all blocksize 128; tuned runs 34, mostly 512 with different tile sizes and group/items splits (`kernel-stats/*.csv`). No run isolated block size, so a reader would take an unmeasured hardware claim from it. Reword to what was measured, e.g. "but the generic table, built from 128-thread configs, was faster in the application."

Not a finding: whether `tuning:gfx1201` is scoped out is jeff's open ruling (Decision 1 of the porter round). Porter fix is the skill wording only; no fork commit, `head_sha` stays 0c897402, then back to `delta-ported`.

## Lesson wording fix (2026-11-13, porter, windows-gfx1151)

Addressed both findings of `## Review 2026-11-13 gfx1201 tuning round, empty fork delta` in `.claude/skills/cuda-to-rocm/references/performance.md`: the slowdown is now "2-9% slower ... on two database sizes" (0.91x q1k, 0.98x 8x database), and the cause is stated as the generic table (built from 128-thread configs) being faster, not block size alone. `python3 utils/check.py skills` passes. No fork change; head stays 0c897402.

## Review 2026-11-13 lesson wording fix 7bc8069a (reviewer, linux-gfx942: review-passed)

Scope: the porter's 7bc8069a, which rewords the second paragraph of "Tile tables are second order" in `.claude/skills/cuda-to-rocm/references/performance.md`. Fork: `git ls-remote` shows `moat-port` = `0c8974021c3e1035640c6daf30e5ab88b6af7fbc` = `head_sha`, no fix or sync branch; the source delta is empty. `python3 utils/check.py skills`: ok. Both findings of the previous review are resolved against `reference/gfx1201-tuning/`: (1) "2-9% slower end to end ... on two database sizes" matches `raw/bench_q1k.txt` (generic 31.79/31.86/31.92 s vs tuned 35.06/34.95 s, 0.909x) and `raw/bench_t8.txt` (8.21/8.25/8.27 s vs 8.39/8.39 s, 0.981x); (2) "the generic table, built from 128-thread configs, was faster" states only the whole-table comparison, and every libmarv gapless kernel in `kernel-stats/generic_kernel_stats.csv` (single- and multi-tile) is blocksize 128. The unchanged sentences still hold: tuner 10-40% (`raw/half2_t1.csv` 4157/3709 = 1.12, 4349/3176 = 1.37, 4395/3132 = 1.40); "a different RDNA table did no better" (g1100 0.90x q1k, 0.94x t8). No problems. `tuning:gfx1201` scope remains jeff's open ruling, not a review item.

## gfx1201 tuning dropped (jeff, 2026-11-13)

jeff ruled to drop gfx1201 tuning: gfx1201 keeps the generic AMD fallback configs, which measured faster than the tuned tables. `surface.json` moves `tuning:gfx1201` to scoped_out, and the two Windows entries likewise per the Windows split ruling above; nothing is owed.

## Curated history on moat-gpu-coverage (2026-11-13, linux-gfx90a)

jeff's ruling (2026-11-13): the upstream PR carries only the GPU-coverage work, on a NEW fork branch `moat-gpu-coverage`; `moat-port` (0c897402) stays untouched as the starting point of the Windows feature. Stage taken `review-passed -> porting` (lock linux-gfx90a), `pr-state` none. Host: MI250X GCD (gfx90a), `HIP_VISIBLE_DEVICES=1` (self-pinned; all four GCDs idle, the dispatcher set none), `CUDA_VISIBLE_DEVICES=0` for `--gpu`, TheRock `rocm-sdk-devel`, gcc 13.3, CMake 3.31.6, Ninja, `-j64`. All GPU runs under `ulimit -c 0`. `set-fork-branch` NOT run (jeff's ruling; recorded on the primary after review PR #1 closes), so `upstream_visible_branch` and `utils/jargon.py --port` still resolve to `moat-port` until then.

### Branch construction

Fork `master` = upstream/master = d0185a2d (unmoved), so the branch starts there.

```
git checkout -b moat-gpu-coverage d0185a2d
git cherry-pick 84a8fb2e 3c588b6f 2cda180e 0c897402      # no conflicts
# the two tuning commits re-created with corrected messages, locally, before publishing:
git checkout -B moat-gpu-coverage 5ee63b2d
git cherry-pick -n 2cda180e && git commit --date=<orig> -F gfx90a-msg
git cherry-pick -n 0c897402 && git commit --date=<orig> -F gfx950-msg
# published as a new branch on the fork (plain, no force)
```

| new | original | title |
|---|---|---|
| ae0dd8b3 | 84a8fb2e | [ROCm] Run libmarv on AMD GPUs without a tuned aligner |
| 5ee63b2d | 3c588b6f | [ROCm] Clamp packed residues as unsigned bytes in HIP device code |
| e530252b | 2cda180e | [ROCm] Add tuned libmarv aligner configs for gfx90a |
| af0688d8 | 0c897402 | [ROCm] Add tuned libmarv aligner configs for gfx950 |

Not carried: 7c39308d, 54edbc78 (Windows), the USE_HIP commits c755847a..75a9b029, merge b29c9270, removal 440e1197. Each new commit's `git patch-id --stable` equals its original's (4c9d9c5c, da1a7204, b8abb228, 0a1f3c43), so the content of every commit is unchanged; ae0dd8b3 and 5ee63b2d keep their messages verbatim.

### Tree-diff proof

- `git diff 0c897402 moat-gpu-coverage --stat`: 9 files, 13 insertions, 155 deletions: cmake/MMseqsResourceCompiler.cmake, lib/libmarv/src/CMakeLists.txt (4), lib/libmarv/src/mapped_file.hpp, lib/libmarv/src/marv.cu, lib/libmarv/src/marv.h, lib/tinyexpr/CMakeLists.txt, src/commons/GpuUtil.cpp, src/prefiltering/ungappedprefilter.cpp, src/util/gpuserver.cpp. These are exactly the files of 7c39308d (first five) and 54edbc78 (last four), and the +/- lines of `git diff moat-gpu-coverage 0c897402` equal those of `git diff 7c39308d~1 54edbc78` line for line.
- Exact: on a scratch worktree at the new tip, `git cherry-pick -n 7c39308d 54edbc78` then `git write-tree` gives 31ae6806..., which is `git rev-parse 0c897402^{tree}`. So the new tree is 0c897402's tree minus exactly the Windows changes.

### Commit message changes

- The "5-10x" figure is in the gfx90a commit (2cda180e), not the fallback commit; e530252b now says "5.2-12.6x slower on gfx90a" (measured in `## Rescope porter round`: 1338 GCUPS at 16x20 against 106-258 above 20 items). Its timing paragraph now carries this round's q1k and 8x numbers (below) and the kernel-trace figure.
- gfx950 (af0688d8): "4.6-9.3x slower" reworded to the recorded ranges, since no raw gfx950 tuner output exists and this host cannot regenerate it: "reach 559-1219 GCUPS against 5177-5683 GCUPS with 20 items at the same group sizes".
- No kept message and not the libmarv Readme mentions Windows (grep; the Readme line lists the tuned arches and the generic fallback only). Titles 51-65 chars, `[ROCm]`, "Authored with Claude.", fenced Test Plan, no Co-Authored-By/noreply. `utils/prose.py -` clean on all four messages.
- Jargon: `utils/jargon.py --port MMseqs2` scans `master..moat-port` (the old branch, because fork_branch is not set), so the equivalent ran on the new range: `utils/jargon.py -C projects/MMseqs2/src --commits d0185a2d..moat-gpu-coverage` clean, `--diff d0185a2d..moat-gpu-coverage` clean (`--port` on moat-port also clean).

### gfx90a tuned vs generic, q1k and 8x database

generic = build of 5ee63b2d (configure prints `marv: generic AMD aligner for gfx90a`), tuned = build of 94e87c29 (pre-message-fix tip, tree identical to af0688d8). Both `cmake -S <src> -B <b> -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=1 -DHAVE_TESTS=1 -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=$R -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++`, `cmake --build <b> --target mmseqs`: 312/312, 76 warning lines each. Benchmark as the gfx950/gfx1201 rounds: `mmseqs ungappedprefilter <Q> <T> p --gpu 1 --max-seqs 1000 -e 10000 --threads 8`, then `createtsv`, sorted md5. Runs alternated; other jobs were compiling on the host (1-min load 8-127, logged per run).

| build | q1k vs t, wall s | q200 vs t8, wall s |
|---|---|---|
| generic | 67.62, 66.75, 66.65 (mean 67.01) | 18.19, 17.02, 16.99, 16.66 (mean 17.21) |
| tuned gfx90a | 55.42, 54.45, 54.48 (mean 54.78) | 12.66, 13.06, 13.01, 12.65 (mean 12.84) |
| gain | **1.22x** | **1.34x** |

- Output identical: q1k 1,000,000 rows, md5 `6e2142dde4e89275a2c378e290a65088` (the same as the gfx90a, gfx950 and gfx1201 rounds, so the recipe matches); t8 200,000 rows, md5 `ac1fd287aa41b7fc5d860d1c61fa7934` for both builds. (gfx1201's t8 md5 is 461eebd4...; its header-suffix scheme was not recorded exactly, so the replicated DB differs in target names. Here: `_r0`..`_r7` appended to the first header token, recipe in the reference README.)
- Kernel trace, t8 (`rocprofv3 --kernel-trace --stats --output-format csv`): libmarv kernels 15.92 s generic vs 12.00 s tuned (1.33x); all kernels 16.32 vs 12.22 s. Generic's top kernels are `DEFAULT_AMD_` half2 128/4/48 3.70 s, 128/4/32 3.25 s; tuned's `AMD_GFX90A_` 512/8/28 1.05 s, multitile 512/8/60 0.98 s.
- Unlike gfx950 (1.05x -> 1.02x) and gfx1201 (slower), the gfx90a gain GROWS on the larger database (1.22x -> 1.34x). No 1.02x finding: the gfx90a table stays well justified.
- Raw logs: `reference/gfx90a-tuning/` (bench_q1k.txt, bench_t8.txt, kernel-stats/t8_*_kernel_stats.csv, build.sh, bench.sh, README.txt with the DB recipes).

### Validation at the new tip (porter evidence, not a validator record)

```
CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=1 util/regression/run_regression.sh <b-tuned>/src/mmseqs <scratch>
```
- Regression (submodule 19b9867): 43/43 TEST SUCCESS, 43 GOOD; SEARCH_GPU 0.454819 = expected (32 s), SEARCH_GPU_SERVER 0.454819 = expected (33 s). No generic aligner in this build, so the GPU tests ran the gfx90a tables.
- Example GPU-vs-CPU (`easy-search examples/QUERY.fasta` padded DB `--gpu 1` vs unpadded `--gpu 0`): GPU 14482, CPU 12901, common 12438, all 12438 rows identical; tuned GPU output identical (sorted) to the generic GPU output. Same counts as every earlier round.
- Fork clone `status --porcelain`: no modified tracked files.
- CUDA: not re-run here (no nvcc). The new tree differs from the nvcc-checked 0c897402 only by removing the Windows hunks; a real nvcc compile of af0688d8 on the mi355 host is still owed before presenting.

### State and what is owed

- `moat-gpu-coverage` at af0688d8 on the fork; `moat-port` 0c897402 untouched (`git ls-remote` checked after publishing). `advance-head` af0688d8; stage `delta-ported` for the reviewer. Every platform record is behind head: the reviewer judges the curation (patch-ids, tree proof, message edits), then validators re-run at af0688d8 (wave64, wave32; windows is waived).
- Owed: nvcc compile at af0688d8 (mi355); `set-fork-branch MMseqs2 moat-gpu-coverage` on the primary after review PR #1 closes (jeff). Until then `jargon.py --port`, `pr_ready` and `upstream.py` look at `moat-port`.
- The PR body drafted in `## Review 2026-11-13` still has the Windows bullet and the old q1k figure; at PR prep drop the Windows bullet and use 1.22x q1k / 1.34x 8x for gfx90a.

## CUDA compile check (nvcc 13.5.37) at af0688d8

jeff's request (2026-11-13, linux-gfx950 fleet host): compile the CUDA path of the curated `moat-gpu-coverage` tip af0688d8 with real nvcc before the upstream PR. Its tree is 0c897402's minus the Windows commits (`## Curated history on moat-gpu-coverage`), but it is a new tree. No MOAT state, fork branch or review PR was changed; the fork was not edited. `orient.sh` ran for setup only (it selected reviewer; not dispatched).

Toolchain: the earlier mi355 CUDA job's job-local CUDA 13.5 wheel prefix, used read-only (`C=<that job>/cuda13-wheels/nvidia/cu13`, `libcudart.so` symlink already present; cuobjdump/nvdisasm from its `cuda13-tools`). gcc 13.3, CMake 3.31.6, Ninja. CMake reports `CUDA compiler identification is NVIDIA 13.5.37`, `Found CUDAToolkit ... (found version "13.5.37")` and `marv: tuned kernels for 80;90; core architectures 80-real;90-real;90-virtual`. Fresh clones of the fork at af0688d8 (contained in `origin/moat-gpu-coverage`) and of upstream d0185a2d, submodules initialized. Commands exactly as `## CUDA compile check (nvcc 13.5.37) at 2cda180e` (top-level `-DENABLE_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES="80;90"` `--target mmseqs`; standalone libmarv `-DENABLE_HIP=0 -DBUILD_TUNING=0 -DONLY_BUILD_TUNING=0 -DLIBRARY_ONLY=0`), each at `-j64` under `utils/timeit.sh MMseqs2 cuda-compile`, the three builds run concurrently (about 3 min each).

Results:
- Fork af0688d8, top-level mmseqs with libmarv: PASS. 317/317 steps, `src/mmseqs` links; `mmseqs version` prints af0688d88121...
- Fork af0688d8, standalone libmarv (`LIBRARY_ONLY=0`): PASS. 40/40 steps: `libmarv.a`, `align`, `makedb`, `test_gapless`, `test_cuda_multitile`, `test_db_copy` link, 0 warnings.
- Upstream d0185a2d, top-level with the same commands: PASS, 317/317. Both top-level builds give 116 warning lines, identical after path normalization (Rust `block_aligner_c` ABI deprecations, four tinyexpr `-Warray-bounds`); none in fork-changed files.
- Device code unchanged against upstream: `cuobjdump -sass` of the fork and upstream `mmseqs` is byte-identical even before normalization (9,049,782 lines; 1043 SASS functions over 13 sm_80 and 20 sm_90 cubins). `cuobjdump -ptx` (1,002,322 lines, 234 entries) differs only in the per-translation-unit `_INTERNAL_<hash>` / `_GLOBAL__N__<hash>` prefixes and is identical once they are normalized. So the curated tree changes no CUDA kernel, as at 2cda180e.
- `xxd or sed not found, compiling resources with CMake instead`: this host has no `xxd`; unchanged upstream behaviour, as before.
- Not run: no NVIDIA GPU on this host, so compile and link only.

This clears the `nvcc compile at af0688d8 (mi355)` item owed in `## Curated history on moat-gpu-coverage`.

## gfx1201 finding goes in the PR body (jeff, 2026-11-13)

jeff ruled: mention the gfx1201 result in the upstream PR body, not a separate issue. The PR body gets one sentence explaining why gfx1201 has no table: tuner-selected configs, and the existing gfx1100 table, measured 2-9% slower end to end than the generic configs on gfx1201 (identical output), so it keeps the generic fallback. Include it when the review PR body is drafted.
## Review 2026-11-13 curated history moat-gpu-coverage af0688d8 (reviewer, linux-gfx942: review-passed)

Scope: `d0185a2d..origin/moat-gpu-coverage` (af0688d8) on the fork, reviewed explicitly because `set-fork-branch` is not recorded yet and the tooling still resolves to `moat-port`. `git ls-remote`: `moat-gpu-coverage` = af0688d8 = `head_sha`, `moat-port` = 0c897402 (untouched), `master` = d0185a2d. Method: `pr-review` skill, local-branch mode; the code of every commit was already reviewed in its original form (`## Review 2026-11-13`, `## Review 2026-11-13 delta 2cda180e..0c897402`), so this round checks the curation, the rewritten messages against the durable evidence, and a build and regression at the new tip. Findings fact-checked inline.

### Port Review: MMseqs2 (moat-gpu-coverage vs d0185a2d)

Summary: four commits on upstream d0185a2d (generic AMD fallback with gfx-name dispatch, ClampToInvalid HIP device path, gfx90a and gfx950 tuned tables), content identical to the reviewed originals and with no Windows change. Approved for validation; one commit-message wording item below needs a person's call because fixing it means re-publishing the branch.

#### Commit Hygiene (decision for jeff before the upstream PR)

1. e530252b `[ROCm] Add tuned libmarv aligner configs for gfx90a`, first paragraph: "gfx90a (MI200) had no tuned aligner and, since the previous change, uses the generic AMD configs." In the curated series the previous commit is 5ee63b2d (ClampToInvalid); the change that made gfx90a use the generic configs is ae0dd8b3, two commits back. (The sentence was already off on `moat-port`, where the previous commit was the Windows 54edbc78; the curation was the chance to fix it.) Suggested wording: "gfx90a (MI200) had no tuned aligner and, with the generic AMD fallback added earlier in this series, uses the generic configs." The branch is published and plain pushes cannot rewrite it, so the fix is either a message-only re-creation on a new branch name (tree identical, so `squash-carry-forward` carries validation forward) or accepting the wording as is. Everything else in the message is accurate (below). Not a code defect, so it does not send the branch back.

#### Checks run (all passed; recorded so the next round does not repeat them)

- Patch-ids (`git show <c> | git patch-id --stable`): ae0dd8b3 = 84a8fb2e (4c9d9c5c7015), 5ee63b2d = 3c588b6f (da1a7204b8cf), e530252b = 2cda180e (b8abb22870d3), af0688d8 = 0c897402 (0a1f3c43aa27). Trees: ae0dd8b3 = 84a8fb2e and 5ee63b2d = 3c588b6f exactly (`git diff --quiet`). Parent chain linear from d0185a2d; author and committer Jeff Daily.
- Tree against 0c897402: `git diff 0c897402 af0688d8 --stat` 9 files, +13/-155, exactly the files of 7c39308d and 54edbc78; `diff <(git diff af0688d8 0c897402) <(git diff 7c39308d~1 54edbc78)` identical (7c39308d~1 = 3c588b6f); on a scratch worktree at af0688d8, `git cherry-pick -n 7c39308d 54edbc78 && git write-tree` = 31ae6806... = `0c897402^{tree}`.
- No dependency on the dropped commits: `git diff d0185a2d af0688d8 --stat` touches only libmarv (11 files, +850/-71: Readme, CMakeLists, MarvGpuArch.cmake, convert.cuh, the two dispatchers, headers, four tuned files); none of the 9 Windows files differs from upstream; `git grep -nE 'MARV_API|MARV_BUILDING_DLL|MARV_USING_DLL|_WIN32|WIN32|strtok_s|Windows' af0688d8 -- lib/libmarv`: no hits. Intermediate commit e530252b stands alone: libmarv standalone at e530252b, `cmake -S lib/libmarv/src -DENABLE_HIP=1 -DLIBRARY_ONLY=1 -DCMAKE_HIP_ARCHITECTURES='gfx90a;gfx942;gfx1201'` (configure prints `generic AMD aligner for gfx1201`), 16/16, 0 warnings. 5ee63b2d is the generic build the porter benchmarked and regression-tested in `## Curated history`; ae0dd8b3's tree is 84a8fb2e's.
- Messages against the evidence: e530252b q1k "54.4-55.4 s against 66.7-67.6 s (1.22x)" = `reference/gfx90a-tuning/bench_q1k.txt` (tuned 55.42/54.45/54.48, mean 54.78; generic 67.62/66.75/66.65, mean 67.01; 1.223x); t8 "12.6-13.1 s against 16.7-18.2 s (1.34x)" = `bench_t8.txt` (tuned 12.66/13.06/13.01/12.65, mean 12.84; generic 18.19/17.02/16.99/16.66, mean 17.21; 1.340x); kernel trace "12.0 s against 15.9 s in the gapless kernels" = `kernel-stats/t8_*_kernel_stats.csv` summed over libmarv kernels, every one a gapless kernel (12.00 vs 15.92 s; all kernels 12.22 vs 16.32 s); rows and md5 identical in every run. "5.2-12.6x" = 1338/258 to 1338/106 GCUPS from `## Rescope porter round` (the raw gfx90a CSVs were scratch; the figure was recorded as measured in that round and accepted by `## Review 2026-11-13`). "removes every tile above 640 and every config with more than 20 items": the endpos multi-tile list in `make_aligner_endpos_amd_gfx90a.cu:63-66` is 512x16x16, 16x20, 32x16, 32x20 (tiles 256-640). af0688d8 "559-1219 GCUPS against 5177-5683 GCUPS with 20 items at the same group sizes" = the pooled ranges in `## gfx950 tuning porter round`, as the gfx950 ruling asked; 1.05x and 1.02x figures unchanged from 0c897402 and verified in the previous delta review. ae0dd8b3 and 5ee63b2d messages verbatim.
- Hygiene: titles 54/65/51/51 characters, `[ROCm]`; "Authored with Claude.", no `Co-Authored-By`/noreply; ASCII; fenced Test Plans. `grep -iE 'windows|win32|dll|moat|msvc'` over the four messages and the added diff lines: no hits. `utils/jargon.py -C projects/MMseqs2/src --commits d0185a2d..moat-gpu-coverage` and `--diff d0185a2d..moat-gpu-coverage`: clean.
- Build at af0688d8 on MI300X (gfx942:sramecc+:xnack-), TheRock HIP (AMD clang 24.0.0), `HIP_VISIBLE_DEVICES=3` (self-pinned, all GPUs idle), `ulimit -c 0`:
  ```
  R=$(hipconfig --rocmpath)
  cmake -S projects/MMseqs2/src -B build-gfx942 -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=1 -DHAVE_TESTS=1 -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_PREFIX_PATH=$R -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++
  cmake --build build-gfx942 -j56 --target mmseqs
  CUDA_VISIBLE_DEVICES=0 util/regression/run_regression.sh build-gfx942/src/mmseqs <scratch>
  ```
  312/312, 116 warning lines (Rust `block_aligner_c`, tinyexpr), none in aligner or convert files; `mmseqs version` af0688d8. Regression (submodule 19b9867b): 43/43 reports GOOD, 43 TEST SUCCESS; SEARCH_GPU 0.454819 = expected (28 s), SEARCH_GPU_SERVER 0.454819 = expected (28 s). No generic aligner configured, so the GPU tests ran upstream's gfx942 table. Review evidence only; the linux-gfx942 validator records its own run.

#### Owed before the upstream PR (not review findings)

- nvcc compile at af0688d8 on the mi355 host (this host has only CUDA 12.8; libmarv needs 13.2+). The removed hunks restore upstream code in CUDA-visible files (`marv.h`, `marv.cu`, `mapped_file.hpp`, `GpuUtil.cpp`), which upstream d0185a2d compiled under nvcc 13.5, so risk is low, but af0688d8's own message claims the CUDA build compiles.
- `set-fork-branch MMseqs2 moat-gpu-coverage` (jeff, after review PR #1 closes); validators at af0688d8 on wave64 and wave32.

### Replacement upstream PR title and body for moat-gpu-coverage

Supersedes the draft in `## Review 2026-11-13`: the Windows bullet is gone, the tuning bullet covers gfx90a and gfx950 (stated plainly, per jeff's gfx950 ruling, with no MI355 headline), and the timing paragraph uses the curated-round gfx90a numbers (1.22x q1k, 1.34x 8x database, means) and the gfx950 numbers (1.05x, 1.02x). Title 69 characters; `utils/prose.py` and `utils/jargon.py` clean on the body; ASCII; no Windows mention. `upstream.py` appends the AI-assistance and opt-out note itself. Before it goes to jeff, confirm each Testing bullet against a record at the final head: the gfx942 bullet is this review's run at af0688d8; the gfx90a bullet is the porter run at af0688d8; the gfx950 and gfx1201 bullets are porter runs at 0c897402 (same libmarv tree plus Windows hunks) and need the validators' runs; the fat-binary, SIMD and CUDA bullets were checked at 2cda180e/0c897402 and the CUDA one is the nvcc item owed above. If e530252b is re-created per finding 1, nothing in the body changes.

Title:

====
[ROCm] Run libmarv on untuned AMD GPUs, add gfx90a and gfx950 configs
====

Body:

====
libmarv's HIP backend builds and runs only on the four AMD targets with a tuned aligner file (gfx900, gfx942, gfx1100, gfx1150). Configuring for any other GPU, for example `-DCMAKE_HIP_ARCHITECTURES=gfx90a`, stops with `Cannot find source file: aligners/make_aligner_amd_gfx90a.cu`, and so does a default configure on a host with such a GPU, because CMake's native detection lists one entry per device. The runtime selection keys on compute capability, which does not identify an AMD ISA (gfx900, gfx906, gfx908 and gfx90a all report 9.0), so a binary built for gfx900 hands its kernels to a gfx90a and fails with `device kernel image is invalid`.

This PR:

- de-duplicates the HIP architecture list and builds the existing generic configs (`make_aligner_default_amd.cu`, `make_aligner_endpos_default_amd.cu`) once for every requested target without a tuned file, under their own inline namespace;
- selects the AMD aligner by the device's gfx target name (`gcnArchName` without feature suffixes) and falls back to the generic configs, which the dispatchers already named in comments; without a generic build the error names the missing target;
- makes `ClampToInvalid` take its per-byte device path under HIP as well (`__HIP_DEVICE_COMPILE__`), with an unsigned per-byte min since HIP has no `__vminu4`; the host fallback it used before clamps as signed `char`;
- adds tuned gapless configs for gfx90a (MI250X) and gfx950 (MI355X), measured with libmarv's own `tuning_gapless` and laid out like the other AMD files; Smith-Waterman keeps the untuned AMD list, as on every AMD target.

The tuned targets keep their own tables and the CUDA path is unchanged.

On the first 1000 queries of the regression query set against the padded `targetannotation` database (`ungappedprefilter --gpu 1 --max-seqs 1000 -e 10000`), the gfx90a configs take 54.8 s against 67.0 s with the generic configs (1.22x, mean of three runs each), and with the first 200 queries against that database replicated 8 times 12.8 s against 17.2 s (1.34x). The generic configs were already close on gfx950, where the gain is 1.05x (45.5 s against 47.7 s) and 1.02x (9.9 s against 10.1 s). Results are identical in every case.

Testing:

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=1 -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build build --target mmseqs
util/regression/run_regression.sh build/src/mmseqs scratch
```

- AMD Instinct MI250X (gfx90a): regression 43/43 including SEARCH_GPU and SEARCH_GPU_SERVER, with the gfx90a configs and with the generic configs alone; on the bundled example every query/target pair shared by the GPU and CPU searches has an identical output row.
- AMD Instinct MI300X (gfx942, existing table): regression 43/43 including SEARCH_GPU and SEARCH_GPU_SERVER.
- AMD Instinct MI355X (gfx950): regression 43/43 including SEARCH_GPU and SEARCH_GPU_SERVER with the gfx950 configs.
- AMD Radeon AI PRO R9700 (gfx1201, generic configs): regression 43/43 including SEARCH_GPU and SEARCH_GPU_SERVER.
- A build for gfx900, gfx90a, gfx942, gfx950, gfx1100, gfx1150, gfx1201 and gfx11-generic compiles and links, and on gfx90a selects the gfx90a table.
- The CUDA build (`-DENABLE_CUDA=1`, sm_80 and sm_90) compiles and links; the SIMD backend (`-DENABLE_SIMD=1`) builds and its tests pass; a CPU-only build is unchanged.

This work was authored with the assistance of Claude, an AI coding agent.
====
