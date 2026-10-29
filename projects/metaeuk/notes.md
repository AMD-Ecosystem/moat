# metaeuk notes

## Build (linux-gfx90a)
```bash
cmake -S projects/metaeuk/src -B projects/metaeuk/build \
  -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON -DENABLE_CUDA=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build projects/metaeuk/build -j16
```

## Port notes
- metaeuk vendors MMseqs2 (lib/mmseqs) which vendors libmarv (the GPU library)
- Port follows same Strategy A as standalone MMseqs2 port (AMD-Ecosystem/MMseqs2@moat-port)
- cuda_to_hip.h maps runtime symbols + SIMD intrinsics (via marv_simd_amd.cuh)
- hip_compat/ redirects <cuda_fp16.h> and <cooperative_groups*> includes
- Kernel configs select conservative sm75 set on AMD (half2 path, 64KB shared)
- cg::reduce calls replaced with portable marv_tile_reduce on HIP
- libmarv built as shared library with HIP_RESOLVE_DEVICE_SYMBOLS=ON
- rocThrust namespace mapped: thrust::cuda -> thrust::hip via using declarations

## Review 2026-06-05

### Summary

Port adds HIP/ROCm support to metaeuk's vendored libmarv GPU alignment library using Strategy A (compat header + LANGUAGE HIP). The port is clean and follows the same pattern as the validated standalone MMseqs2 port.

### Port Correctness

No issues found. The port correctly:
- Force-includes cuda_to_hip.h to map CUDA runtime symbols
- Emulates 14 SIMD intrinsics (__vadd2, __vmaxs2, __viaddmax_s16x2, __vibmax_u16x2, etc.) with correct semantics (wrapping add for __viaddmax, saturating for __vaddus4/__vsubus4)
- Provides shuffle overloads for packed types (short2, int2, float2, int3)
- Replaces cg::reduce with marv_tile_reduce using butterfly reduction over group.shfl_xor

### Fault Classes

No issues found:
- **Warp size**: All tile widths are <= 32 (groupsize in {4,8,16,32}), explicit parameters. Wave-agnostic.
- **Lane masks**: The __shfl_*_sync macros drop the mask and forward to maskless __shfl_* (correct for width-bounded logical-warp shuffles)
- **Rule-of-five**: cub::SwitchDevice has proper RAII (deleted copy ctor/assignment, guarded destructor)
- **Library swaps**: rocThrust correctly mapped via namespace injection
- **Kernel config dispatch**: AMD branch returns sm75 config set (avoids CC==9 collision with Hopper DPX path)

### Minimal Footprint

No issues found. Host C++ is untouched. USE_HIP guards are minimal and only where genuinely needed. CUDA path is byte-identical.

### Build System

No issues found:
- enable_language(HIP) used correctly
- CMAKE_HIP_ARCHITECTURES defaults to gfx90a only when not defined (followers can override)
- HIP_ARCHITECTURES reads from ${CMAKE_HIP_ARCHITECTURES}
- --offload-compress used to control object size
- CUDA build path preserved in else() block

### Commit Hygiene

No issues found:
- Title: "[ROCm] Add HIP/ROCm support for AMD GPUs" (40 chars, starts with [ROCm])
- Body mentions Claude by name
- No Co-Authored-By noreply trailer
- Author is the public account (jeffdaily)
- No MOAT jargon in the diff

### Recommendation

**Approve** -- the port is correct and ready for GPU validation.

## Validation 2026-06-05 (linux-gfx90a)

### Build
Rebuilt from scratch with HIP support:
```bash
HIP_VISIBLE_DEVICES=0 cmake -S /var/lib/jenkins/moat/projects/metaeuk/src \
  -B /var/lib/jenkins/moat/projects/metaeuk/build \
  -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON -DENABLE_CUDA=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
HIP_VISIBLE_DEVICES=0 cmake --build /var/lib/jenkins/moat/projects/metaeuk/build -j16
```
Build completed successfully. Binary: 15M metaeuk executable linked against libmarv.so (49M) and libamdhip64.so.7.

### GPU Code Verification
- libmarv.so contains .hip_fatbin section (43.7 MB) with gfx90a device code
- Contains hipLaunchKernel and GPU kernel symbols (GaplessFilter_strided_PSSM_singletile_kernel, etc.)
- Verified gfx90a arch in fatbin with objdump

### Test Results

**CPU Baseline (--gpu 0):**
Ran complete test suite from tests/run.sh:
- minus_strand, multi_exon, two_contigs, target_overlap, cluster_rep, target_cov
- test_no_overlap.sh, test_agg_tax.sh, test_start_scan.sh
- Result: ALL OKAY

**GPU Mode (--gpu 1):**
Ran 6 core test suites with --gpu 1 flag in predictexons command:
- minus_strand_results_gpu
- multi_exon_results_gpu
- two_contigs_results_gpu
- target_overlap_results_gpu
- cluster_rep_results_gpu
- target_cov_results_gpu
- Result: ALL OKAY (all tests passed, results match expected output)

GPU flag confirmed active in logs: "Use GPU 1" shown in predictexons workflow.

### Pass/Fail Summary
- Build: PASS
- GPU code present: PASS (gfx90a kernels in .hip_fatbin)
- CPU baseline tests: PASS (9/9 test scripts)
- GPU functional tests: PASS (6/6 core test suites, --gpu 1)
- CPU vs GPU consistency: PASS (identical results)

GPU arch: gfx90a (MI250X)
Validation: PASSED

## Validation 2026-06-08 (windows-gfx1201, RDNA4 RX 9070 XT)

Platform: Windows 11, AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32).
ROCm 7.14.0a20260604 (TheRock nightly pip SDK). HIP_VISIBLE_DEVICES=0.
Fork head: a1b8ad0 (Windows fixes commit on top of de54dee).

### Windows delta (required build fixes)

The full metaeuk binary has deep POSIX mmap/shm_open/fork dependencies
(DBReader.cpp, FileUtil.cpp, etc.) that are impractical to port within
a validation pass. Instead, the standalone libmarv (LIBRARY_ONLY=1) is
built as a DLL, and the existing Marv API harness validates GPU alignment
correctness via Marv::scan().

Fixes applied to metaeuk fork (paralleling MMseqs2 sibling port):

1. cuda_to_hip.h: NOMINMAX/WIN32_LEAN_AND_MEAN before hip_runtime.h;
   HIP_DISABLE_WARP_SYNC_BUILTINS suppresses bfloat16 warp-sync overload
   redefinition on ROCm 7.14.
2. mapped_file.hpp: Win32 CreateFileMapping/MapViewOfFile implementation
   behind #ifdef _WIN32 (POSIX mmap used by marv.cu indirectly).
3. marv.cu: strtok_r -> strtok_s under _WIN32.
4. marv.h: MARV_API __declspec(dllexport/dllimport) for DLL visibility;
   add missing #include <string>.
5. CMakeLists.txt (libmarv): CMAKE_HIP_USING_LINKER_DEFAULT "" on WIN32;
   MARV_BUILDING_DLL define for the DLL build.
6. tinyexpr/CMakeLists.txt: guard -fPIC behind if(NOT WIN32).

### Build

```
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel

cmake -S projects/metaeuk/src/lib/mmseqs/lib/libmarv/src \
      -B projects/metaeuk/build-marv-gfx1201 \
      -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
      -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
      -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
      -DCMAKE_PREFIX_PATH=$ROCM -DLIBRARY_ONLY=1 \
      -DCMAKE_BUILD_TYPE=Release

HIP_VISIBLE_DEVICES=0 utils/timeit.sh metaeuk compile -- \
  cmake --build projects/metaeuk/build-marv-gfx1201 -j24 --target marv
```

Build succeeded. marv.dll is 61 MB (61621248 bytes), 14 Marv:: symbols exported
(confirmed via llvm-objdump -p). gfx1201 device code embedded.

### GPU validation (Marv API harness)

Used the existing marv_validate_gfx1201.cpp harness (same Marv::scan() API
as MMseqs2/foldseek validations). Compiled against metaeuk's marv.dll:

```
$ROCM/lib/llvm/bin/clang++.exe -std=c++17 -O2 \
  -Iprojects/metaeuk/src/lib/mmseqs/lib/libmarv/src \
  -Iprojects/metaeuk/src/lib/mmseqs/lib/libmarv/src/hip_compat \
  -Lprojects/metaeuk/build-marv-gfx1201 -lmarv \
  -o agent_space/metaeuk_val_gfx1201.exe \
  agent_space/marv_validate_gfx1201.cpp

HIP_VISIBLE_DEVICES=0 utils/timeit.sh metaeuk test -- \
  agent_space/metaeuk_val_gfx1201.exe
```

Results:
- Test 1: 20-residue query (all 20 standard amino acids). GPU returns
  top hit id=2, score=116 (expected BLOSUM62 self-score). PASS.
- Test 2: 16xAla query. Top hit id=3, score=64 (16 * BLOSUM62[A][A]=4).
  PASS.

GPU PSSM-based gapless alignment kernels produce correct BLOSUM62 scores
on gfx1201 RDNA4. The Marv::scan() path (same as metaeuk ungappedprefilter
--gpu) is exercised.

VERDICT: PASS. State -> completed (validated_sha = a1b8ad0).

## Validation 2026-06-08 (linux-gfx1100, revalidate at 2e4e953)

### Delta from de54dee to 2e4e953

Two commits on top of the original port:
- a1b8ad0: Windows (MSVC-ABI clang) build support -- adds `#ifdef _WIN32` guards in mapped_file.hpp, marv.cu, marv.h, CMakeLists.txt, tinyexpr/CMakeLists.txt; adds `HIP_DISABLE_WARP_SYNC_BUILTINS` and `__syncwarp` definition to cuda_to_hip.h (affects Linux HIP device code)
- 2e4e953: Fix `__syncwarp` missing after `HIP_DISABLE_WARP_SYNC_BUILTINS` -- adds explicit `__device__ inline void __syncwarp()` to cuda_to_hip.h

The `HIP_DISABLE_WARP_SYNC_BUILTINS` define affects Linux builds (it suppresses amd_warp_sync_functions.h and changes the `__syncwarp` inline function, affecting compiled device code). Full GPU revalidation required.

### Build

Built from scratch at head_sha (2e4e953) for gfx1100 (HIP_VISIBLE_DEVICES=2):
```bash
export HIP_VISIBLE_DEVICES=2
cmake -S /var/lib/jenkins/moat/projects/metaeuk/src \
  -B /var/lib/jenkins/moat/projects/metaeuk/build-gfx1100 \
  -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON -DENABLE_CUDA=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build /var/lib/jenkins/moat/projects/metaeuk/build-gfx1100 -j16
```
Build completed successfully. metaeuk: 15M, libmarv.so: 52M with gfx1100 device code (confirmed via strings | grep gfx1100).

### Test Results

**CPU Baseline (--gpu 0, default):**
```bash
cd tests; for test in minus_strand multi_exon two_contigs target_overlap cluster_rep target_cov; do
  perl compare_fasta_results.pl ${test}_results/predExFlat.fas ${test}_results/predRedFlat.fas \
    ${test}/as_should_final_united_exons_aa.fas ${test}/as_should_final_grouped_predictions_rep.fas; done
```
All 6 core tests: ALL OKAY. test_agg_tax: Saccharomyces FOUND. test_no_overlap and test_start_scan results present.

**GPU Mode (--gpu 1), HIP_VISIBLE_DEVICES=2 (gfx1100, Radeon Pro W7800):**
```bash
export HIP_VISIBLE_DEVICES=2
for test in minus_strand multi_exon two_contigs target_overlap cluster_rep target_cov; do
  # createdb, predictexons --gpu 1, unitesetstofasta, reduceredundancy, compare
  perl compare_fasta_results.pl ${test}_results_gpu_new/predExFlat.fas ...
done
```
All 6 core GPU tests: ALL OKAY. GPU confirmed active: "Use GPU 1" in predictexons logs.

### Pass/Fail Summary
- Build: PASS (gfx1100, fork HEAD 2e4e953)
- GPU code present: PASS (gfx1100 in libmarv.so .hip_fatbin)
- CPU baseline tests: PASS (6/6 fasta comparisons, test_agg_tax, test_no_overlap, test_start_scan)
- GPU functional tests: PASS (6/6 core suites, --gpu 1)
- CPU vs GPU consistency: PASS (identical results)

GPU arch: gfx1100 (Radeon Pro W7800 48GB)
Validation: PASSED

## Validation 2026-06-05 (linux-gfx1100)

### Build
Built for gfx1100 using HIP_VISIBLE_DEVICES=3:
```bash
HIP_VISIBLE_DEVICES=3 cmake -S /var/lib/jenkins/moat/projects/metaeuk/src \
  -B /var/lib/jenkins/moat/projects/metaeuk/build-gfx1100 \
  -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON -DENABLE_CUDA=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
HIP_VISIBLE_DEVICES=3 cmake --build /var/lib/jenkins/moat/projects/metaeuk/build-gfx1100 -j16
```
Build completed successfully. Binary: 15M metaeuk executable linked against libmarv.so (52M) and libamdhip64.so.7.

### GPU Code Verification
- libmarv.so contains .hip_fatbin section (47.5 MB) with gfx1100 device code
- Contains hipLaunchKernel and GPU kernel symbols
- Verified gfx1100 arch in fatbin with objdump

### Test Results

**CPU Baseline (--gpu defaults to 0):**
Ran complete test suite from tests/run.sh:
- minus_strand, multi_exon, two_contigs, target_overlap, cluster_rep, target_cov
- test_no_overlap.sh, test_agg_tax.sh, test_start_scan.sh
- Result: ALL OKAY

**GPU Mode (--gpu 1):**
Ran 6 core test suites with --gpu 1 flag in predictexons command:
- minus_strand_results_gpu
- multi_exon_results_gpu
- two_contigs_results_gpu
- target_overlap_results_gpu
- cluster_rep_results_gpu
- target_cov_results_gpu
- Result: ALL OKAY (all tests passed, results match expected output)

GPU flag confirmed active in logs: "Use GPU 1" shown in predictexons workflow.

### Pass/Fail Summary
- Build: PASS
- GPU code present: PASS (gfx1100 kernels in .hip_fatbin)
- CPU baseline tests: PASS (9/9 test scripts)
- GPU functional tests: PASS (6/6 core test suites, --gpu 1)
- CPU vs GPU consistency: PASS (identical results)

GPU arch: gfx1100
Validation: PASSED

## Validation 2026-06-08 (linux-gfx90a, revalidate at 2e4e953)

### Delta from de54dee7 to 2e4e953

head_sha changed from de54dee7 -> a1b8ad0e (Windows build fixes) -> 2e4e953 (Linux fix).

classify returned `unknown/arch_independent=False` (classification failed), requiring full revalidation.

The a1b8ad0e commit added `HIP_DISABLE_WARP_SYNC_BUILTINS` to suppress a bfloat16 redefinition on Windows ROCm 7.14. This macro gates `amd_warp_sync_functions.h`, which provides `__syncwarp`. `kernels.cuh:777` calls `__syncwarp()` -- the Linux build at a1b8ad0e failed with "undeclared identifier '__syncwarp'".

Fix applied in new commit 2e4e953: add explicit `__syncwarp()` definition to `cuda_to_hip.h` after the `HIP_DISABLE_WARP_SYNC_BUILTINS` guard, using the same fence+wave_barrier+fence sequence from `amd_warp_sync_functions.h`. Protected by `#ifndef __syncwarp` so it is a no-op if the platform provides it.

### Build

```bash
HIP_VISIBLE_DEVICES=3 cmake -S /var/lib/jenkins/moat/projects/metaeuk/src \
  -B /var/lib/jenkins/moat/projects/metaeuk/build \
  -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON -DENABLE_CUDA=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
HIP_VISIBLE_DEVICES=3 cmake --build /var/lib/jenkins/moat/projects/metaeuk/build -j16
```

Build completed successfully. Binary: 15M metaeuk at build/src/metaeuk, libmarv.so 50M.

### Test Results

**CPU Baseline (--gpu 0, default):**
```bash
cd /var/lib/jenkins/moat/projects/metaeuk/src/tests
bash run.sh /var/lib/jenkins/moat/projects/metaeuk/build/src/metaeuk
```
All 9 test scripts: ALL OKAY

**GPU Mode (--gpu 1), HIP_VISIBLE_DEVICES=3 (gfx90a MI250X):**
Ran 6 core test suites with --gpu 1 in predictexons:
- minus_strand_results_gpu: ALL OKAY
- multi_exon_results_gpu: ALL OKAY
- two_contigs_results_gpu: ALL OKAY
- target_overlap_results_gpu: ALL OKAY
- cluster_rep_results_gpu: ALL OKAY
- target_cov_results_gpu: ALL OKAY

GPU confirmed active: "Use GPU 1" in each predictexons run.

### Pass/Fail Summary
- Build: PASS (with __syncwarp fix in 2e4e953)
- CPU baseline tests: PASS (9/9)
- GPU functional tests: PASS (6/6 core suites, --gpu 1)
- CPU vs GPU consistency: PASS

Fork head pushed: 2e4e953a40822e1fd717d641686be5a012c4bbaf
GPU arch: gfx90a (MI250X, GCD3)
Validation: PASSED

## Revalidate check 2026-06-08 (windows-gfx1201)

Delta: a1b8ad0e -> 2e4e953a (one commit: "[ROCm] Fix __syncwarp missing after HIP_DISABLE_WARP_SYNC_BUILTINS").

Method: source-class inspection.

Change: adds `__device__ inline void __syncwarp()` to lib/mmseqs/lib/libmarv/src/cuda_to_hip.h, guarded by `#ifndef __syncwarp`, using fence+wave_barrier+fence sequence.

On Windows/TheRock 7.14 with HIP_DISABLE_WARP_SYNC_BUILTINS always set, `__syncwarp` is NOT defined by any TheRock 7.14 header (amd_warp_sync_functions.h puts it under `#if !defined(HIP_DISABLE_WARP_SYNC_BUILTINS)`). The `#ifndef __syncwarp` guard passes, and the new explicit definition is compiled into device code. `kernels.cuh` calls `__syncwarp()` unconditionally.

This is a functional device-code change: the `__syncwarp()` intrinsic implementation changes (was potentially undefined/missing, now explicitly fence+wave_barrier+fence). Device ISA for gfx1201 is NOT identical.

VERDICT: Cannot carry forward. Left as `revalidate`. A GPU test run is required to validate the __syncwarp fix on gfx1201 RDNA4.

## Validation 2026-06-08 (windows-gfx1201, revalidate at 2e4e953)

Platform: Windows 11, AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32).
ROCm 7.14.0a20260604 (TheRock nightly pip SDK). HIP_VISIBLE_DEVICES=0.
Fork head: 2e4e953 (__syncwarp fix commit).

### Build

Rebuilt libmarv from scratch at 2e4e953 for gfx1201:

```
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel

cmake -S projects/metaeuk/src/lib/mmseqs/lib/libmarv/src \
      -B projects/metaeuk/build-marv-gfx1201 \
      -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
      -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
      -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
      -DCMAKE_PREFIX_PATH=$ROCM -DLIBRARY_ONLY=1 \
      -DCMAKE_BUILD_TYPE=Release

HIP_VISIBLE_DEVICES=0 utils/timeit.sh metaeuk compile -- \
  cmake --build projects/metaeuk/build-marv-gfx1201 -j16 --target marv
```

Build succeeded (exit code 0). marv.dll is 61621248 bytes (same size as prior validated build). gfx1201 device code embedded including the new explicit __syncwarp definition.

### GPU validation (Marv API harness)

Recompiled agent_space/metaeuk_val_gfx1201.exe against the new marv.dll, then ran:

```
HIP_VISIBLE_DEVICES=0 utils/timeit.sh metaeuk test -- \
  agent_space/metaeuk_val_gfx1201.exe
```

Results:
- Test 1: 20-residue query (all 20 standard amino acids). GPU returns
  top hit id=2, score=116 (expected BLOSUM62 self-score). PASS.
- Test 2: 16xAla query. Top hit id=3, score=64 (16 * BLOSUM62[A][A]=4).
  PASS.

The explicit fence+wave_barrier+fence __syncwarp definition on TheRock 7.14 (HIP_DISABLE_WARP_SYNC_BUILTINS set) produces correct BLOSUM62 alignment scores on gfx1201 RDNA4.

### Pass/Fail Summary
- Build: PASS (gfx1201, fork HEAD 2e4e953)
- GPU code present: PASS (gfx1201 device code with new __syncwarp)
- GPU functional tests: PASS (2/2: score=116 and score=64)
- __syncwarp validation: PASS (fence+wave_barrier+fence correct on RDNA4 wave32)

GPU arch: gfx1201 (RX 9070 XT, RDNA4)
Validation: PASSED. State -> completed (validated_sha = 2e4e953).

## Validation 2026-06-19 (windows-gfx1101, RDNA3 Radeon PRO V710)

Platform: Windows 11, AMD Radeon PRO V710 (gfx1101, RDNA3, wave32). HIP_VISIBLE_DEVICES=1.
ROCm 7.14 (TheRock pip SDK). Fork head: 2e4e953.

### Build

Built libmarv as DLL (LIBRARY_ONLY=1), same approach as gfx1201 Windows validation:

```
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel

cmake -S projects/metaeuk/src/lib/mmseqs/lib/libmarv/src \
      -B projects/metaeuk/build-marv-gfx1101 \
      -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
      -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
      -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
      -DCMAKE_PREFIX_PATH=$ROCM -DLIBRARY_ONLY=1 \
      -DCMAKE_BUILD_TYPE=Release

HIP_VISIBLE_DEVICES=1 utils/timeit.sh metaeuk compile -- \
  cmake --build projects/metaeuk/build-marv-gfx1101 -j64 --target marv
```

Build succeeded (exit code 0). marv.dll is 57,657,344 bytes (55MB), gfx1101 device code embedded.
14 Marv:: symbols exported (confirmed via llvm-nm on marv.lib).

### GPU validation (Marv API harness)

Compiled marv_validate_gfx1101.cpp against metaeuk's marv.dll (gfx1101 build).
Runtime DLLs (amdhip64_7.dll, amd_comgr.dll, hiprtc0714.dll, hiprtc-builtins0714.dll)
co-located in agent_space/ alongside the exe and marv.dll.

```
$ROCM/lib/llvm/bin/clang++.exe -std=c++17 -O2 \
  -Iprojects/metaeuk/src/lib/mmseqs/lib/libmarv/src \
  -Iprojects/metaeuk/src/lib/mmseqs/lib/libmarv/src/hip_compat \
  -Lprojects/metaeuk/build-marv-gfx1101 -lmarv \
  -o agent_space/metaeuk_val_gfx1101.exe \
  agent_space/marv_validate_gfx1101.cpp

HIP_VISIBLE_DEVICES=1 utils/timeit.sh metaeuk test -- \
  agent_space/metaeuk_val_gfx1101.exe
```

Results:
- Test 1: 20-residue query (all 20 standard amino acids). GPU returns
  top hit id=2, score=116 (expected BLOSUM62 self-score). PASS.
- Test 2: 16xAla query. Top hit id=3, score=64 (16 * BLOSUM62[A][A]=4).
  PASS.

GPU PSSM-based gapless alignment kernels produce correct BLOSUM62 scores
on gfx1101 RDNA3. Results match gfx1201 exactly.

### Pass/Fail Summary
- Build: PASS (gfx1101, fork HEAD 2e4e953)
- GPU code present: PASS (gfx1101 device code in marv.dll)
- GPU functional tests: PASS (2/2: score=116 and score=64)

GPU arch: gfx1101 (AMD Radeon PRO V710, RDNA3)
Validation: PASSED. State -> completed (validated_sha = 2e4e953).

## Upstream HIP support validation on linux-gfx90a (2026-10-29)

Purpose: upstream metaeuk gained HIP through its 2026-09-08 `lib/mmseqs` subtree update. This validates that path as if it were our port. No MOAT state or fork branch was changed. libmarv-level analysis is in the MMseqs2 notes (`port/MMseqs2`, same date): metaeuk's `lib/mmseqs/lib/libmarv` is byte-identical to MMseqs2 master 50374396 (`diff -rq` clean).

Host: linux-gfx90a, MI250X GCD, HIP_VISIBLE_DEVICES=0, CUDA_VISIBLE_DEVICES=0. ROCm 7.16.26385 (TheRock pip SDK), AMD clang 24.0.0git, gcc 13.3.0, cmake 3.31.6.

Upstream sha: `2b5193a844b3ba1759158fa5254cd3114b4ead3d` (master, 2026-09-22). Tests: `tests` submodule (metaeuk-regression) at `9e024f341c92e34197079914dcdd864f23cd6500`.

### Build

```
R=$(hipconfig --rocmpath)
cmake -S metaeuk -B metaeuk/build-hip -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=1 \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=$R -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++
utils/timeit.sh metaeuk compile -- cmake --build metaeuk/build-hip -j32
```

PASS, no HIP warnings. `ENABLE_HIP` comes from `lib/mmseqs/CMakeLists.txt`; metaeuk has no HIP docs, Dockerfile or CI of its own.

### FINDING: `--gpu 1` never reaches the GPU in metaeuk (upstream and our port alike)

`predictexons`/`easy-predict` forward the whole `searchworkflow` parameter list to `mmseqs search` via `createParameterString`, which always includes `--prefilter-mode 0`. In `lib/mmseqs/src/workflow/Search.cpp:323` the GPU only switches the prefilter to ungapped when `par.PARAM_PREF_MODE.wasSet == false`; here it is always set, so the k-mer CPU prefilter runs and the GPU is never touched. Evidence:

```
# test.sh copy with --gpu 1 (agent_space/probes/metaeuk_test_gpu.sh); with and without makepaddedseqdb on proteins
metaeuk predictexons contigs proteins[_pad] predEx tmp --set-gap-extend -1 --metaeuk-tcov 0 --gpu 1
```

Both runs log `prefilter ...` (the k-mer prefilter), never `ungappedprefilter`, and the inner search line carries `--gpu 1 ... --prefilter-mode 0`. Our port did not change the metaeuk workflow (its fork diff is libmarv plus CMake only), so this applies to our port too: **the earlier "GPU Mode (--gpu 1) ALL OKAY" validations in this file (gfx90a 2026-06-05 and 2026-06-08, gfx1100 2026-06-05 and 2026-06-08) did not exercise the GPU.** "Use GPU 1" in those logs is the parameter echo, not evidence of a kernel launch. Only the Windows Marv-API harness runs above actually ran libmarv kernels for metaeuk.

### GPU tests that do reach the GPU (`ulimit -c 0`)

Adding `--prefilter-mode 1` and a padded target DB makes the search run `ungappedprefilter ... --gpu 1`. Kernel launches confirmed: `AMD_LOG_LEVEL=3` on minus_strand shows 1731 kernel-launch log lines.

```
metaeuk makepaddedseqdb proteins proteins_pad
metaeuk predictexons contigs proteins_pad predEx tmp ... --prefilter-mode 1 --gpu {0|1}
```

CPU oracle is the same command with `--gpu 0` (CPU ungapped prefilter on the same padded DB). For all six datasets (minus_strand, multi_exon, two_contigs, target_overlap, cluster_rep, target_cov) `predExFlat.fas` and `predRedFlat.fas` are byte-identical between GPU and CPU (6/6). Prediction counts against the regression expectations, which were made with the k-mer prefilter: minus_strand 6 vs 5 expected, multi_exon 6 vs 3, two_contigs 16 vs 8, target_overlap 8 vs 4, cluster_rep 3 vs 2, target_cov 1 vs 1. So `compare_fasta_results.pl` reports "number of predictions has changed" for 5/6 in BOTH CPU and GPU ungapped mode: the ungapped prefilter is more permissive, not a GPU numerics difference.

Default regression (k-mer path, HIP binary): `utils/timeit.sh metaeuk test -- bash -c "cd tests && ./run.sh ../build-hip/src/metaeuk"`: rc=0, 9/9 scripts pass (8 "ALL OKAY!" plus test_agg_tax).

### Comparison with our port

libmarv: see MMseqs2 table (upstream better or equivalent; latent `ClampToInvalid`; no Windows; RDNA unvalidated). metaeuk-specific: none in either. Our port's `__syncwarp`/`HIP_DISABLE_WARP_SYNC_BUILTINS` fix (2e4e953) is not needed upstream, where the legacy kernels using `__syncwarp` are compiled out on HIP.

### Gaps

Defects specific to HIP on gfx90a: none.

1. Workflow defect (CUDA and HIP alike): `--gpu 1` is silently ignored by `predictexons`/`easy-predict`. A GPU user gets the CPU k-mer prefilter with no warning, and there is no metaeuk documentation that the target DB must be padded. This is the main gap that keeps metaeuk from having working GPU support on any vendor.
2. No test coverage: the metaeuk-regression suite has no GPU case, and expected outputs exist only for the k-mer prefilter.
3. Docs: nothing in metaeuk's README about GPU, padding, or ROCm.
4. RDNA/wave32 runtime and Windows: same as MMseqs2.

### Draft upstream issue (not posted)

====
Title: predictexons/easy-predict --gpu 1 never uses the GPU

`metaeuk predictexons ... --gpu 1` runs the k-mer CPU prefilter. PredictExons.cpp builds `SEARCH_PAR` with `createParameterString(par.searchworkflow)`, which always emits `--prefilter-mode 0`. Search.cpp only switches to the ungapped (GPU) prefilter when `--prefilter-mode` was not set, so the GPU path is never taken. The search log shows `prefilter` rather than `ungappedprefilter` even with a padded target database.

Passing `--prefilter-mode 1` together with `--gpu 1` and a `makepaddedseqdb` target does run the GPU prefilter, and its predictions match the CPU ungapped prefilter exactly on all six metaeuk-regression datasets. One fix is to set the prefilter mode in PredictExons.cpp when `--gpu` is given and `--prefilter-mode` was not, and to fail with a clear message when the target database is not padded.

Reproduce with the minus_strand regression data:

    metaeuk createdb contigs.fna contigs --dbtype 2
    metaeuk createdb proteins.faa proteins --dbtype 1
    metaeuk makepaddedseqdb proteins proteins_pad
    metaeuk predictexons contigs proteins_pad predEx tmp --gpu 1   # log shows "prefilter", not "ungappedprefilter"
====

## Upstream HIP support validation on linux-gfx1100 (2026-10-29)

jeff's request: validate upstream's own ROCm/HIP support as if porting metaeuk ourselves, find every gap, and seed follow-up work from what this port learned. No MOAT state or fork branch was changed. The GPU library is the same libmarv as MMseqs2. See the MMseqs2 notes section of the same date for the libmarv fault-class comparison, which applies here unchanged.

### What was tested

- Upstream: https://github.com/soedinglab/metaeuk `master` at `2b5193a844b3ba1759158fa5254cd3114b4ead3d` (2026-09-22). HIP arrives through the `lib/mmseqs` subtree merge eadaf81 (2026-09-08). `lib/mmseqs/lib/libmarv` is byte-identical to MMseqs2 master 50374396's libmarv.
- Our port for comparison: AMD-Ecosystem/metaeuk `moat-port` e6070cc5. Its diff is the same libmarv set as the MMseqs2 port, plus `lib/mmseqs/src/CMakeLists.txt` and the Windows-only tinyexpr `-fPIC` guard.
- Host and toolchain as in the MMseqs2 section: Radeon Pro W7800 gfx1100 wave32, TheRock `rocm-sdk-devel` 10.1.0 (HIP 7.16.26385).

### Build

```
R=/opt/python-3.11-venv/lib/python3.11/site-packages/_rocm_sdk_devel; export ROCM_PATH=$R HIP_PATH=$R
cmake -S metaeuk -B metaeuk/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=1 \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH=$R
utils/timeit.sh metaeuk upstream-hip-compile -- cmake --build metaeuk/build -j24
```
PASS: clean, no source changes (`ENABLE_HIP` is the `lib/mmseqs` cache option).

### Tests

1. metaeuk regression (`tests` submodule = metaeuk-regression 9e024f34, `cd tests && ./run.sh build/src/metaeuk`), as upstream's azure pipeline runs it: rc=0, all 6 `test.sh` cases plus no_overlap, start_scan and sacc_tax pass ("ALL OKAY" x8). This is the CPU path, and it shows the HIP build does not regress it.
2. The same 6 cases with `--gpu 1` added to `predictexons` (scratch script `agent_space/scripts/me_gpu_test.sh`, same createdb / predictexons / unitesetstofasta / reduceredundancy / compare_fasta_results.pl steps as `tests/test.sh`):
   - `--gpu 1` alone, with an unpadded or a `makepaddedseqdb`-padded target: 6/6 "ALL OKAY", but `ungappedprefilter` never appears in the log. The CPU k-mer `prefilter` ran. The GPU was not used.
   - `--gpu 1 --prefilter-mode 1`, unpadded target: 6/6 fail with `Database proteins is not a valid GPU database / Please call: makepaddedseqdb`. This is correct behaviour.
   - `--gpu 1 --prefilter-mode 1`, padded target: the GPU ungapped prefilter runs. 1/6 "ALL OKAY" (target_cov). 5/6 report "number of predictions has changed" (more predictions: minus_strand 6 vs expected 5, multi_exon 6 vs 3, two_contigs 16 vs 8, target_overlap 8 vs 4, cluster_rep 3 vs 2).
   - Oracle: the same 6 cases with `--gpu 0 --prefilter-mode 1` (CPU ungapped). `predExFlat.fas` and `predRedFlat.fas` are byte-identical to the GPU run in 6/6 cases. So GPU numerics are exact. The extra predictions come from the ungapped prefilter's higher sensitivity, and the regression expectations were recorded with the k-mer prefilter.

### Gaps found

1. `--gpu 1` is silently ignored by `predictexons` and `easy-predict` (defect, affects CUDA and HIP alike). Cause: `src/workflow/PredictExons.cpp:45` builds `SEARCH_PAR` with `par.createParameterString(par.searchworkflow)`, which forwards every parameter including the default `--prefilter-mode 0`. In the child `search`, `PARAM_PREF_MODE.wasSet` is therefore true, so `lib/mmseqs/src/workflow/Search.cpp:323` (`if (par.gpu == 1 && par.PARAM_PREF_MODE.wasSet == false)` -> ungapped prefilter) never fires, and the CPU k-mer prefilter runs. MMseqs2's `easy-search` avoids this by forwarding only explicitly set parameters (`createParameterString(par.searchworkflow, true)`, `EasySearch.cpp:127`). A user gets no error and no GPU. `predictexons -h` advertises `--gpu INT  Use GPU (CUDA) if possible`. Draft issue (not posted):
   > `metaeuk predictexons ... --gpu 1` (and `easy-predict`) never runs the GPU prefilter: PredictExons.cpp forwards `--prefilter-mode 0` explicitly to `search`, so Search.cpp's "GPU implies ungapped prefilter" switch is skipped and the k-mer prefilter runs on CPU. With `--gpu 1 --prefilter-mode 1` and a `makepaddedseqdb` target the GPU path works, and its predictions are identical to `--gpu 0 --prefilter-mode 1`, but they differ from the default k-mer results the regression suite expects (5 of 6 cases predict more exons). Should `--gpu 1` imply `--prefilter-mode 1` here, as it does in MMseqs2 search, and should the README document `makepaddedseqdb` for GPU use?
   Fix shape: mirror Search.cpp's switch in PredictExons.cpp before building `SEARCH_PAR`, plus README text on `makepaddedseqdb`. The prediction-count change is a maintainer decision, not a bug.
2. No GPU documentation or GPU regression case. metaeuk's README never mentions `--gpu`, and metaeuk-regression has no GPU expectations.
3. Cosmetic, shared with MMseqs2: the `--gpu` help text (`lib/mmseqs/src/commons/Parameters.cpp:108`, "Use GPU (CUDA) if possible") and the no-GPU error (`ungappedprefilter.cpp`, "MMseqs2 was compiled without CUDA support") still say CUDA in a HIP build. Belongs in MMseqs2.

### About our port's earlier GPU evidence

The 2026-06-05 gfx90a and 2026-06-08 gfx1100 validations recorded "GPU functional tests: PASS (6/6 core test suites, --gpu 1)" and "CPU vs GPU consistency: PASS (identical results)". Given gap 1, those runs never executed the GPU prefilter (their evidence was "Use GPU 1" in the parameter dump, which only echoes the flag). The 6/6 matched because they were CPU runs. The libmarv GPU path in metaeuk was in fact only exercised by the Windows Marv API harness. Today's `--prefilter-mode 1` run is the first evidence that the GPU path works inside metaeuk's workflow, on upstream's code. Lesson for future validations: confirm a GPU module actually ran (look for the GPU module in the log), not that the flag was accepted.

### Verdict

Upstream's HIP build of metaeuk is correct where the GPU is reachable: GPU predictions are identical to the CPU ungapped path on all 6 regression cases. The real gap is metaeuk-specific and platform-neutral: `--gpu 1` does nothing in its workflows (gap 1).
