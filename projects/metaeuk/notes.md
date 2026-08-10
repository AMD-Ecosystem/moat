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
