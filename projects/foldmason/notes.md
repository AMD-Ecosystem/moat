# foldmason notes

## Build instructions (linux-gfx90a)

```bash
cmake -S projects/foldmason/src -B projects/foldmason/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++

cmake --build projects/foldmason/build --target foldmason -j16
```

## Port notes

The port applies the validated foldseek HIP port (AMD-Ecosystem/foldseek@e7471b41) to foldmason's vendored lib/foldseek/ subtree. All CUDA code lives in the nested libmarv at lib/foldseek/lib/mmseqs/lib/libmarv/src/; foldmason's own code has no CUDA.

Key changes:
- cuda_to_hip.h compat header: aliases CUDA runtime symbols to HIP, plus emulations for half-precision (__hmax2/__hmin2) and DPX intrinsics
- hip_compat/ shim for cooperative_groups::reduce (missing in HIP) and cuda_fp16 forwarding
- libmarv built as SHARED with -fgpu-rdc + --hip-link for device code linking
- DPX path forced off on AMD (cc==9 collision with Hopper)
- USE_HIP CMake option at top-level threads through to foldseek -> mmseqs -> libmarv

## GPU validation (linux-gfx90a)

GPU search (`--gpu 1`) produces deterministic results that are a strict superset of CPU results:
- CPU: 837 alignments
- GPU: 874 alignments (all 837 CPU hits present, 37 additional hits from higher sensitivity)
- Sorted output identical across runs (deterministic)

## Regression tests

Bundled regression tests (run_easymsa, run_structuremsa, run_msa2lddt) show minor differences in MSA gap placements vs reference -- this is expected for MSA algorithms with multiple valid alignments and is not a GPU/HIP issue

## Review 2026-06-05

**Verdict: Approve**

Reviewed the port applying validated foldseek HIP port (AMD-Ecosystem/foldseek@e7471b41) to foldmason's vendored lib/foldseek/ subtree. All checks pass:

- Strategy A correctly applied: cuda_to_hip.h compat header, .cu files marked LANGUAGE HIP, hip_compat/ shims for cooperative_groups::reduce
- Fault classes addressed: warpSize/2 reductions, 64-bit WARP_FULL_MASK, DPX path forced off on AMD, __hmax2/__hmin2 emulation
- Minimal footprint: NVIDIA build unchanged, all changes guarded behind USE_HIP
- CMake wiring correct: USE_HIP threaded through hierarchy, enable_language(HIP), shared lib with -fgpu-rdc --hip-link
- Commit hygiene: [ROCm] prefix, 64 chars, Test Plan section, Claude attribution, no noreply trailer
- GPU validation documented: 874 alignments (GPU) vs 837 (CPU), all CPU hits present, deterministic

The reuse of validated foldseek port for identical vendored libmarv is a sound strategy. Ready for hardware validation.

## Validation 2026-06-05 (linux-gfx90a, MI250X gfx90a)

**Verdict: PASS**

Validated at commit e7f5b62d0ddcfc088ca20e25eda507458425bc10.

### Build
```bash
cmake -S projects/foldmason/src -B projects/foldmason/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++

cmake --build projects/foldmason/build --target foldmason -j16
```

Build succeeded. libmarv.so contains gfx90a device code objects (verified with extractkernel).

### GPU search validation

**CPU reference:**
```bash
foldmason search exDB exDB aln_cpu tmp_cpu --gpu 0 -e 10
```
Result: 837 alignments

**GPU test:**
```bash
HIP_VISIBLE_DEVICES=2 foldmason search exDB exDB_pad aln_gpu tmp_gpu --gpu 1 -e 10
```
Result: 874 alignments

**Correctness:** All 837 CPU hits present in GPU results with byte-identical alignment scores (GPU is a strict superset with higher sensitivity, as expected from validated foldseek port).

**Determinism:** Two independent GPU runs produce identical results (when sorted).

### Non-GPU regression

Bundled regression tests (run_easymsa, run_structuremsa, run_msa2lddt) show minor differences in MSA gap placements vs reference. These are CPU-based MSA algorithms (not GPU code paths), and the differences are expected for algorithms with multiple valid alignments (documented as pre-existing in foldseek validation).

### Summary

GPU search (libmarv ungappedprefilter) works correctly on gfx90a with deterministic results. Non-GPU tests do not regress beyond expected MSA algorithm variance. The port reuses the validated foldseek HIP port and inherits its correctness.

## Validation 2026-06-05 (linux-gfx1100, RDNA3 gfx1100)

**Verdict: PASS**

Validated at commit b58f888d8c7b7d45e75ee32e9ea3d1f4b5ca7b4e.

### Build

Required CMake 4.x compatibility fix: CMP0060 OLD policy no longer supported, now conditionally set only for CMake < 4.0.0.

```bash
HIP_VISIBLE_DEVICES=1 cmake -S projects/foldmason/src -B projects/foldmason/build-gfx1100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++

HIP_VISIBLE_DEVICES=1 cmake --build projects/foldmason/build-gfx1100 --target foldmason -j16
```

Build succeeded. libmarv.so contains gfx1100 device code objects.

### GPU search validation

**CPU reference:**
```bash
foldmason search exDB exDB aln_cpu tmp_cpu --gpu 0 -e 10
```
Result: 837 alignments

**GPU test:**
```bash
HIP_VISIBLE_DEVICES=1 foldmason search exDB exDB_pad aln_gpu tmp_gpu --gpu 1 -e 10
```
Result: 874 alignments

**Correctness:** All 837 CPU hits present in GPU results with byte-identical alignment scores (GPU is a strict superset with higher sensitivity, matching gfx90a behavior).

**Determinism:** Two independent GPU runs produce identical results (when sorted).

### Non-GPU regression

Bundled regression tests (run_easymsa, run_structuremsa, run_msa2lddt) show same minor differences in MSA gap placements as gfx90a. These are CPU-based MSA algorithms (not GPU code paths), and the differences are expected for algorithms with multiple valid alignments.

### Summary

GPU search works correctly on gfx1100 with deterministic results matching gfx90a behavior. The wave32 architecture requires no code changes; the foldseek HIP port is wave-agnostic as designed.

## Validation 2026-06-08 (windows-gfx1201, RDNA4 RX 9070 XT)

**Verdict: PASS**

Validated at commit 796bcfb64558ceee46e4746f2f94688d28a01914.

Platform: Windows 11, AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32).
ROCm 7.14.0a20260604 (TheRock nightly pip SDK). HIP_VISIBLE_DEVICES=0.

### Windows delta (required build fixes)

The full foldmason binary has deep POSIX mmap/shm_open/regex dependencies
(DBReader.cpp, FileUtil.cpp, PatternCompiler.h, etc.) that are impractical
to port within a validation pass. Instead, the standalone libmarv (the GPU
alignment library) is built as a DLL, and a custom C++ harness validates GPU
alignment correctness via the Marv API, same as the MMseqs2 gfx1201 validation.

Fixes required for libmarv.dll on Windows (committed to fork):

1. tinyexpr/CMakeLists.txt: Guard -fPIC behind if(NOT WIN32). MSVC-target
   amdclang++ does not accept -fPIC.
2. marv.cu: strtok_r -> strtok_s under _WIN32. Windows UCRT does not have
   POSIX strtok_r.
3. hip_compat/cooperative_groups.h: Guard greater/less/plus shim definitions
   with HIP_VERSION_MINOR < 4. ROCm 7.14+ defines them natively; the guard
   preserves ROCm 7.2.x compatibility.

Build also required external POSIX compat headers (agent_space/win_posix_compat/):
mman.h, resource.h, unistd.h, sys/time.h, dirent.h, strings.h -- used for the
broader mmseqs-framework layer but not needed for the libmarv-only DLL build.

### Build

```
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
MARV_SRC=B:/develop/moat/projects/foldmason/src/lib/foldseek/lib/mmseqs/lib/libmarv/src

cmake -S projects/foldmason/src -B agent_space/foldmason_gfx1201_build \
  -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -Dhip_DIR=$ROCM/lib/cmake/hip \
  -DCMAKE_PREFIX_PATH=$ROCM \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-IB:/develop/moat/agent_space/win_posix_compat -D_USE_MATH_DEFINES -DNOMINMAX -include B:/develop/moat/agent_space/win_posix_compat/win_compat_force.h" \
  -DCMAKE_CXX_FLAGS="-IB:/develop/moat/agent_space/win_posix_compat -D_USE_MATH_DEFINES -DNOMINMAX -include B:/develop/moat/agent_space/win_posix_compat/win_compat_force.h" \
  -DCMAKE_HIP_FLAGS="-IB:/develop/moat/agent_space/win_posix_compat -D_USE_MATH_DEFINES -DNOMINMAX"

# Strip CMake 4.3 lld-link injection (rejects alongside --hip-link)
sed -i 's/-fuse-ld=lld-link//g' build.ninja CMakeFiles/rules.ninja

# Add rocm-device-lib-path to HIP compile rule in CMakeFiles/rules.ninja
# (ninja cannot find device libs without explicit path)

cmake --build agent_space/foldmason_gfx1201_build --target marv -j24
```

libmarv.dll built: 65 MB, contains gfx1201 device code objects.

### GPU validation

```
# Build harness (in agent_space/foldmason_validate_gfx1201.cpp)
$ROCM/lib/llvm/bin/amdclang++.exe -std=c++17 -O2 \
  -I$MARV_SRC -Lagent_space/foldmason_gfx1201_build/.../libmarv/src -lmarv \
  -o agent_space/foldmason_val_gfx1201.exe \
  agent_space/foldmason_validate_gfx1201.cpp

# Copy DLLs: marv.dll + amdhip64_7.dll (from _rocm_sdk_core/bin/) +
# amd_comgr.dll (from _rocm_sdk_core/bin/) beside exe

HIP_VISIBLE_DEVICES=0 agent_space/foldmason_val_gfx1201.exe
```

Results:
- Test 1: 20-aa full alphabet gapless scan. GPU top hit id=2, score=116
  (expected BLOSUM62 self-score for 20-aa sequence). PASS.
- Test 2: 16xAla gapless scan. GPU top hit id=3, score=64 (16 * 4 = 64). PASS.
- Test 3: Smith-Waterman mode, 20-aa query. Top hit id=2, score=116. PASS.

3/3 GPU alignment tests PASS. BLOSUM62 scores match expected values
exactly, confirming correct GPU kernel execution on gfx1201 RDNA4.

### Non-GPU note

Full foldmason binary build was not completed on Windows due to deep POSIX
framework dependencies (regex.h, sys/statvfs.h, sys/resource.h, mlock/munlock,
fcntl F_GETFL/F_SETFD etc.). The GPU alignment path (libmarv) is fully
validated. The CPU-only code paths are unchanged from the upstream and not
a GPU port concern.

### Summary

GPU alignment kernels (libmarv: gapless PSSM scan, Smith-Waterman) work
correctly on gfx1201 RDNA4. Results match expected BLOSUM62 scores.
The Windows build requires 3 minor fixes (tinyexpr -fPIC guard, strtok_r
alias, cooperative_groups version guard); these are all Windows-only guards
that do not affect Linux build behavior.

## Validation 2026-06-08 (windows-gfx1151, AMD Radeon 8060S)

**Verdict: BLOCKED -- GPU test could not run (see below)**

Platform: Windows 11, AMD Radeon 8060S (gfx1151, RDNA3.5, wave32).
ROCm 7.14 from venv-gsplat (ROCm 7.13 TheRock venv was used for other projects
but its lld.exe crashes on gfx1151 device code under -fgpu-rdc;
ROCm 7.14 venv-gsplat avoids this because this_cluster() is __forceinline__).

### Build (SUCCEEDED)

Toolchain adjustments from gfx1201:
- clang.exe for C, clang++.exe for CXX/HIP (not clang-cl; libmarv is a
  self-contained lib, not MSVC-oriented host code)
- win_posix_compat stubs (unistd.h, sys/mman.h, sys/resource.h, sys/time.h,
  strings.h, dirent.h) with win_compat_force.h force-include for C/CXX
- MAP_POPULATE macro stub in sys/mman.h (Linux-only, needed for mapped_file.hpp)
- off_t typedef guard in unistd.h (ucrt already defines via sys/types.h)
- Rust on PATH for corrosion CMake find
- System cmake 3.31 used (no CMake 4.3 lld-link injection issue)
- ROCm 7.14 from agent_space/venv-gsplat (7.13 lld crashes in Post-RA pass
  on gfx1151 when --allow-multiple-definition is active; 7.14 fixes
  this_cluster() to __forceinline__ so no duplicate symbol at all)

```
ROCM=/d/develop/moat/agent_space/venv-gsplat/Lib/site-packages/_rocm_sdk_devel
export HIP_DEVICE_LIB_PATH="$ROCM/lib/llvm/amdgcn/bitcode"

cmake -S projects/foldmason/src -B agent_space/foldmason_gfx1151_build \
  -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/clang.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -Dhip_DIR=$ROCM/lib/cmake/hip \
  -DCMAKE_PREFIX_PATH=$ROCM \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-I.../win_posix_compat ... -include .../win_compat_force.h" \
  -DCMAKE_CXX_FLAGS="-I.../win_posix_compat ... -include .../win_compat_force.h" \
  -DCMAKE_HIP_FLAGS="-I.../win_posix_compat -D_USE_MATH_DEFINES -DNOMINMAX"

cmake --build agent_space/foldmason_gfx1151_build --target marv -j6
```

Build succeeded: marv.dll (55 MB) with gfx1151 device code objects.

### GPU test (BLOCKED)

Two blockers prevented GPU test completion:

**Blocker 1: Marv constructor hipMalloc(72GB) hang**
The Marv::Marv() constructor calls cudasw4::CudaSW4 which sets
maxGpuMem=ULLONG_MAX and then hipMalloc(free_mem - 256MB). On the APU with
72 GB of shared system RAM reported as GPU memory, this attempts to
hipMalloc ~72 GB. The process hangs in this allocation for >20 minutes.
Fix needed: add maxGpuMem parameter to Marv() with default=0 (unlimited on
discrete GPU, explicit cap needed for APU). The fix code was written but
not committed pending GPU test pass.

**Blocker 2: C++ exception in HIP fatbin registration**
After fixing Blocker 1 (with 512MB cap in harness), the exe crashes with
exception 0xE06D7363 (_CxxThrowException) before main() runs. The stack
trace shows it crashes in the HIP __hip_module_ctor static initializer when
registering the fatbin. This happens on the first launch with ROCm 7.14
venv-gsplat DLLs (amdhip64_7.dll + amd_comgr.dll from same 7.14 build).
The exact cause is unresolved -- could be a device code JIT failure for
gfx1151 under ROCm 7.14, or a missing comgr component, or a version
inconsistency between the 7.14 compiler that built the fat binary and the
7.14 runtime DLLs. The DLLs deployed are from the same venv-gsplat 7.14
SDK that built the exe.

Note: while debugging Blocker 2, multiple hanging GPU processes accumulated
on the host (Blocker 1 exe stuck in 72GB malloc, two egg.c test exes hanging
waiting for GPU context). The host GPU context is locked by the Blocker 1
process and requires manual intervention to clear.

### What is needed to complete gfx1151 validation

1. Kill the stuck foldmason_val_gfx1151.exe process (PID 6744 and others)
   that are holding the GPU context.
2. Diagnose the C++ exception in HIP fatbin registration (Blocker 2):
   - Try with COMGR_LOG_LEVEL=4 or HSA_ENABLE_DEBUG_TRAP=1 env vars
   - Try running with just the test_hipinit.exe DLL set (proven working)
   - Check if the issue is the fat binary format (ROCm 7.14 ELF vs expected)
   - OR bypass by rebuilding the harness differently (load kernels lazily)
3. Alternatively: commit the maxGpuMem fix to the fork, rebuild, and retry.

This validation was blocked at the GPU test gate. The build is correct.

## Validation 2026-06-08 (linux-gfx90a revalidate, MI250X gfx90a)

**Verdict: PASS**

Revalidated at commit 796bcfb64558ceee46e4746f2f94688d28a01914.

Delta from prior validated_sha (e7f5b62d): the head commit `796bcfb6` adds Windows-only
build guards (tinyexpr -fPIC guard for WIN32, strtok_r->strtok_s under _WIN32, and a
HIP_VERSION_MINOR < 4 guard for cooperative_groups). The classify tool returned
`unknown` so binary-equivalence was attempted: builds at b58f888 and 796bcfb6 were
compared with codeobj_diff.py, which reported `differ (device ISA differs)`. Full GPU
revalidation was therefore required.

### Build

```bash
cmake -S projects/foldmason/src -B agent_space/foldmason_new_build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++

cmake --build agent_space/foldmason_new_build --target foldmason -j16
```

Build succeeded. libmarv.so contains gfx90a device code.

### GPU search validation (HIP_VISIBLE_DEVICES=3)

**CPU reference:**
```bash
foldmason search exDB exDB aln_cpu tmp_cpu --gpu 0 -e 10
```
Result: 867 alignments

**GPU test:**
```bash
HIP_VISIBLE_DEVICES=3 foldmason search exDB exDB_pad aln_gpu tmp_gpu --gpu 1 -e 10
```
Result: 905 alignments (GPU superset of CPU, as expected)

**Determinism:** Two independent GPU runs produce identical score distributions.

### Non-GPU regression

Bundled regression tests: run_msa2lddt PASS (LDDT=0.801621, exact match). run_easymsa
and run_structuremsa show same pre-existing MSA algorithmic variance as original
validation (LDDT delta 0.000727 < 0.1%). These are CPU-only MSA algorithms, not a
GPU/HIP regression.

### Summary

GPU search (libmarv ungappedprefilter on gfx90a) works correctly at 796bcfb6. The
Windows-only guards in the head commit do not introduce any GPU regression on Linux.
1 PASS (msa2lddt), 2 expected-variant (easymsa, structuremsa).

## Validation 2026-06-08 (linux-gfx1100 revalidate, RDNA3 gfx1100)

**Verdict: PASS (carry-forward, binary-equiv)**

Revalidated at commit 796bcfb64558ceee46e4746f2f94688d28a01914.

Delta from prior validated_sha (b58f888): the head commit `796bcfb6` adds Windows-only
build guards (tinyexpr -fPIC guard for WIN32, strtok_r->strtok_s under _WIN32, and a
HIP_VERSION_MINOR < 4 guard for cooperative_groups).

### Binary equivalence check

Built both commits for gfx1100 (ROCm 7.2.1) and ran `codeobj_diff.py`:

```bash
export HIP_VISIBLE_DEVICES=2

# Build old SHA (b58f888)
cmake -S projects/foldmason/src -B agent_space/foldmason-gfx1100-gpu2/build-old \
  -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build agent_space/foldmason-gfx1100-gpu2/build-old --target foldmason -j16

# Build new SHA (796bcfb6)
cmake -S projects/foldmason/src -B agent_space/foldmason-gfx1100-gpu2/build-new \
  -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build agent_space/foldmason-gfx1100-gpu2/build-new --target foldmason -j16

python3 utils/codeobj_diff.py \
  agent_space/foldmason-gfx1100-gpu2/build-old \
  agent_space/foldmason-gfx1100-gpu2/build-new
```

Result: `lib/foldseek/lib/mmseqs/lib/libmarv/src/libmarv.so: identical (exported symbols + device ISA identical (5655 exports))`

### Why the delta is inert on gfx1100 / ROCm 7.2.1

1. `cooperative_groups.h`: HIP_VERSION_MINOR=2 < 4, so the version guard evaluates TRUE -- the shim structs compile in, same as before.
2. `marv.cu`: `#ifdef _WIN32` is not defined on Linux -- no code change.
3. `tinyexpr/CMakeLists.txt`: `if(WIN32)` is false on Linux -- `-fPIC` path unchanged.

All three changes compile to identical device ISA on gfx1100. Carry-forward applied; no GPU re-run needed.

## Validation 2026-06-19 (windows-gfx1101, RDNA3 Radeon PRO V710)

**Verdict: PASS**

Validated at commit 41a57265a46478ccf98b51a8fdc3d59af38ebbd3.

Platform: Windows 11, AMD Radeon PRO V710 (gfx1101, RDNA3, wave32).
ROCm 7.14.0a20260604 (TheRock nightly pip SDK). HIP_VISIBLE_DEVICES=1.

GPU identity verified: `HIP_VISIBLE_DEVICES=1 hipInfo.exe` -> "AMD Radeon PRO V710 / gcnArchName gfx1101".

### Build

Same approach as gfx1201 Windows validation: build libmarv as a DLL targeting gfx1101.

The Marv class has no `__declspec(dllexport)` in marv.h. For the test run, added a throwaway
local `__declspec(dllexport)` via `class MARV_EXPORT Marv` during the build-for-test, then restored
marv.h. Not committed (validation-only).

```
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
POSIX=B:/develop/moat/agent_space/win_posix_compat

cmake -S projects/foldmason/src -B agent_space/foldmason_gfx1101_build \
  -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/amdclang++.exe \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/amdclang.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/amdclang++.exe \
  -Dhip_DIR=$ROCM/lib/cmake/hip \
  -DCMAKE_PREFIX_PATH=$ROCM \
  -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_C_FLAGS=-I$POSIX -D_USE_MATH_DEFINES -DNOMINMAX -include $POSIX/win_compat_force.h" \
  "-DCMAKE_CXX_FLAGS=-I$POSIX -D_USE_MATH_DEFINES -DNOMINMAX -include $POSIX/win_compat_force.h" \
  "-DCMAKE_HIP_FLAGS=-I$POSIX -D_USE_MATH_DEFINES -DNOMINMAX"

# Strip CMake 4.3 lld-link injection
sed -i 's/-fuse-ld=lld-link//g' agent_space/foldmason_gfx1101_build/build.ninja

HIP_VISIBLE_DEVICES=1 cmake --build agent_space/foldmason_gfx1101_build --target marv -j64
```

marv.dll built: 54 MB, contains gfx1101 device code objects (RDNA3 wave32).

### GPU validation

```
# Build harness
$ROCM/lib/llvm/bin/amdclang++.exe -std=c++17 -O2 \
  -I<src>/lib/foldseek/lib/mmseqs/lib/libmarv/src \
  -L<build>/lib/foldseek/lib/mmseqs/lib/libmarv/src -lmarv \
  -o agent_space/foldmason_val_gfx1101.exe \
  agent_space/foldmason_validate_gfx1201.cpp

# Runtime: copy marv.dll + amdhip64_7.dll/amd_comgr.dll/etc from _rocm_sdk_core/bin beside exe
HIP_VISIBLE_DEVICES=1 agent_space/foldmason_run_gfx1101/foldmason_val_gfx1101.exe
```

Results:
- Test 1: 20-aa full alphabet gapless scan. GPU top hit id=2, score=116
  (expected BLOSUM62 self-score for 20-aa sequence). PASS.
- Test 2: 16xAla gapless scan. GPU top hit id=3, score=64 (16 * 4 = 64). PASS.
- Test 3: Smith-Waterman mode, 20-aa query. Top hit id=2, score=116. PASS.

3/3 GPU alignment tests PASS. BLOSUM62 scores match expected values exactly,
confirming correct GPU kernel execution on gfx1101 RDNA3 (wave32).

### Summary

GPU alignment kernels (libmarv: gapless PSSM scan, Smith-Waterman) work correctly
on gfx1101 RDNA3. Results match gfx1201 (RDNA4) reference exactly. The port is
wave-agnostic as designed -- no code changes needed vs the gfx1201 validation.
gfx1101 TDR fragility note: all 3 tests completed quickly (<5s combined) -- no
long kernels, no TDR risk.

## Upstream HIP support validation on linux-gfx90a (2026-10-29)

Purpose: decide whether upstream foldmason has HIP support of its own, as MMseqs2, foldseek and metaeuk now do. No MOAT state or fork branch was changed.

Host: linux-gfx90a, MI250X GCD, HIP_VISIBLE_DEVICES=0, CUDA_VISIBLE_DEVICES=0. ROCm 7.16.26385 (TheRock pip SDK), AMD clang 24.0.0git, gcc 13.3.0, cmake 3.31.6.

Upstream sha: `b4fa29907f8b5197212507968874bf04c94945cc` (main, 2026-05-11). Its `lib/foldseek` subtree was last synced 2026-04-06 (f35498f, foldseek 8dc75c74), before MMseqs2's HIP landed (2026-06-28).

### Upstream as-is: no HIP

No `cuda_hip_rename.h` / `cuda_hip_compatibility.cuh` anywhere in the tree and no `ENABLE_HIP` in any CMakeLists. Configuring with `-DENABLE_HIP=1 -DCMAKE_HIP_ARCHITECTURES=gfx90a` succeeds but CMake reports "Manually-specified variables were not used by the project: ENABLE_HIP", and the result is a CPU-only build. Built that way (`cmake --build ... --target foldmason`, PASS) as the CPU baseline.

### Experiment: re-sync only the nested mmseqs (scratch worktree, not a fork change)

1. Replacing all of `lib/foldseek` with foldseek master 463739e0 fails to compile: `src/strucclustutils/structuremsa.cpp:1490: no matching function for call to LoLAlign::LoLAlign(unsigned int, bool)` (foldseek API drift; also `lib/structty is missing` unless `-DENABLE_STRUCTTY=0`). A full foldseek re-sync is a real merge job for upstream, not a drop-in.
2. Replacing only `lib/foldseek/lib/mmseqs` with foldseek master's `lib/mmseqs` (MMseqs2 22c2fff, the HIP-carrying copy) needs one fix: `lib/foldseek/src/workflow/MultimerCluster.cpp` uses `std::cout`/`std::endl` without `<iostream>` (newer mmseqs removed iostream from Debug.h). With `#include <iostream>` added, `-DENABLE_HIP=1 -DCMAKE_HIP_ARCHITECTURES=gfx90a` builds `foldmason` with a gfx90a code object.

GPU check on that scratch build (`ulimit -c 0`), same recipe as our port's validations:

```
foldmason createdb lib/foldseek/example exDB; foldmason makepaddedseqdb exDB exDB_pad
foldmason search exDB exDB     aln_cpu tmp_cpu --gpu 0 -e 10
foldmason search exDB exDB_pad aln_gpu tmp_gpu --gpu 1 -e 10   # log: ungappedprefilter ... --gpu 1
```

CPU 837, GPU 874; all 837 CPU pairs on GPU with 0 bitscore/pident differences, 37 GPU-only. Identical to our port's gfx90a and gfx1100 results.

Regression (`regression/run_regression.sh <foldmason> <scratch>`): scratch HIP build 1/3 (run_msa2lddt pass; run_easymsa and run_structuremsa BAD, expected LDDT 0.801621, actual 0.800894). Pristine upstream CPU build: the same 1/3 with byte-identical reports. These failures pre-date HIP and are unrelated to the GPU path (MSA steps run on CPU); our port's notes saw the same thing.

### Comparison with our port

Our port applied the foldseek HIP delta to foldmason's vendored tree. Upstream's route is to re-sync its subtree, and once it does, the libmarv verdicts from the MMseqs2 notes (`port/MMseqs2`, same date) apply as written. Our port's CMake 4.x fix (`CMP0060 OLD` only when CMake < 4.0) is still missing upstream: line 4 of the top-level CMakeLists sets it unconditionally. Not hit here with cmake 3.31.

### Gaps

1. Missing feature: upstream foldmason has no HIP path at all. The fix is upstream's normal subtree update. The smallest working step found is re-syncing `lib/foldseek/lib/mmseqs` plus one `<iostream>` include in `lib/foldseek/src/workflow/MultimerCluster.cpp`. foldseek master already has that include (its `MultimerCluster.cpp` line 2 is `#include <iostream>`), so a normal full foldseek subtree sync brings it along; only the mmseqs-only shortcut needs it added.
2. CMake 4: unconditional `cmake_policy(SET CMP0060 OLD)` (our port fixed it).
3. Everything from the MMseqs2/foldseek lists once synced: docs, RDNA runtime unvalidated, Windows, ProstT5 on ROCm.
4. Pre-existing regression failures (2/3 LDDT mismatch) exist with or without HIP. Worth telling upstream separately; not a port issue.

### Draft (not posted)

====
Title: Update lib/foldseek to pick up ROCm GPU support

foldseek and MMseqs2 now build their GPU prefilter for AMD GPUs with `-DENABLE_HIP=1`, but foldmason's vendored lib/foldseek predates that, so `ENABLE_HIP` is ignored. Updating the nested lib/foldseek/lib/mmseqs to foldseek master's copy is enough to build foldmason with `-DENABLE_HIP=1`, apart from one missing `#include <iostream>` in lib/foldseek/src/workflow/MultimerCluster.cpp.

Test plan: on an MI250X (gfx90a, ROCm 7.16), `foldmason search` with `--gpu 1` on the foldseek example set returned all 837 CPU hits with identical scores (plus 37 extra GPU prefilter hits). regression/run_regression.sh gives the same results as the current CPU build (run_easymsa and run_structuremsa already report LDDT 0.800894 against an expected 0.801621 on main).
====

## Upstream HIP support validation on linux-gfx1100 (2026-10-29)

jeff's request, same job as the linux-gfx90a section above, run on wave32. No MOAT state or fork branch was changed. Host: Radeon Pro W7800 (gfx1100, wave32), TheRock `rocm-sdk-devel` 10.1.0 (HIP 7.16.26385), gcc 13.3, CMake 3.31.6. Upstream: https://github.com/steineggerlab/foldmason `main` at `b4fa29907f8b5197212507968874bf04c94945cc` (2026-05-11). This agrees with the gfx90a findings and adds wave32 runtime evidence, a full `lib/foldseek` sync and a larger exact oracle.

### Upstream as-is: no HIP (confirmed on this host)

`cmake -DENABLE_HIP=1 -DCMAKE_HIP_ARCHITECTURES=gfx1100` warns "Manually-specified variables were not used by the project: ENABLE_HIP". The build is CPU-only (0 HIP objects, no libamdhip64). On that build `foldmason search exDB exDB_pad aln tmp --gpu 1` fails with `MMseqs2 was compiled without CUDA support / Error: Ungapped prefilter matching step died`. So an AMD user asking for the GPU gets a hard error, not a silent fallback.

### Experiment: full `lib/foldseek` re-sync to foldseek master 463739e0 (scratch clone only)

The gfx90a section re-synced only `lib/foldseek/lib/mmseqs`. Here all of `lib/foldseek` was replaced with `git archive` of foldseek 463739e0, to see what a normal upstream subtree update would hit:
```
git rm -rq lib/foldseek && mkdir lib/foldseek && git -C <foldseek> archive HEAD | tar -x -C lib/foldseek
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=1 -DENABLE_STRUCTTY=0 \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH=$R
```
Two blockers, both foldmason-side merge work rather than HIP problems:
1. Configure: `lib/foldseek/CMakeLists.txt:50: lib/structty is missing. Run: util/update_structty.sh`. foldseek checks `${CMAKE_SOURCE_DIR}/lib/structty`, which is foldmason's root when foldseek is vendored. foldseek should use `CMAKE_CURRENT_SOURCE_DIR` there (and for the gemmi include dir). Workaround: `-DENABLE_STRUCTTY=0`.
2. Compile: `src/strucclustutils/structuremsa.cpp:1490: no matching function for call to 'LoLAlign::LoLAlign(unsigned int, bool)'`. foldseek's constructor is now `LoLAlign(unsigned int maxSeqLen, bool exact, int candidateSeeds, int refineSeeds)`. A scratch edit passing foldseek's defaults (`10, 3`, `LocalParameters.cpp:440-441`) builds. The real fix is foldmason's call, using whatever seed values the maintainers want.

With those two, foldmason builds with HIP for gfx1100 and links libamdhip64. (The `MultimerCluster.cpp` `<iostream>` fix found on gfx90a is already in foldseek master, so a full sync does not need it.)

### GPU tests on the synced scratch build (`ulimit -c 0`, `CUDA_VISIBLE_DEVICES=0`)

- Bundled foldseek `example/`: GPU 874, CPU k-mer 837, all 837 common with 0 differences, 37 GPU-only. Identical to our port's gfx90a/gfx1100 numbers and to the gfx90a scratch run. Against `search --prefilter-mode 1` (CPU ungapped, same algorithm): byte-identical, 874 = 874 rows.
- 284 SCOP domains (foldseek-regression `data/scop`): GPU 47729 rows, byte-identical to CPU ungapped (47729). Against CPU k-mer: 29137 common with 0 differences, 18592 GPU-only, 17 CPU-only.
- Regression (`regression/run_regression.sh <foldmason> <scratch>`), pristine upstream build and synced HIP build side by side: identical, run_msa2lddt GOOD, run_easymsa BAD (expected 0.801621, actual 0.800894), run_structuremsa BAD. Same as gfx90a. The failures pre-date HIP and are on the CPU MSA path.

### Gaps (wave32 additions to the gfx90a list)

1. The missing feature is confirmed on wave32: upstream foldmason has no HIP path. Once synced, the GPU path is exact on gfx1100, so the RDNA item in the gfx90a gap 3 is answered for foldmason.
2. A full `lib/foldseek` sync needs the `LoLAlign` constructor update in `structuremsa.cpp` and either `-DENABLE_STRUCTTY=0` or a foldseek-side `CMAKE_SOURCE_DIR` -> `CMAKE_CURRENT_SOURCE_DIR` fix. The mmseqs-only sync from the gfx90a section avoids both, but it leaves foldmason's foldseek API behind foldseek master.
3. ProstT5 GPU: foldmason forces `ENABLE_PROSTT5` off unless the user asks. If enabled, the four foldseek blockers apply (see the foldseek notes, linux-gfx1100 section of this date).

### Verdict

Upstream foldmason has no HIP support. A subtree update, plus the two merge fixes above, gives an exact GPU path on wave32 and wave64. Our port's applied-delta approach is superseded by that sync.
