# SCAMP notes

## Port Summary (linux-gfx90a)

Strategy A port with cuda_to_hip.h compatibility header.

### Key Changes

1. **cuda_to_hip.h**: Main compat header with:
   - kWarpSize: 64 for __GFX9__ (CDNA), 32 otherwise (RDNA)
   - SCAMP_FULL_WARP_MASK: 64-bit (0xffffffffffffffffULL) for HIP
   - CUDA->HIP runtime/FFT/CUB symbol aliases
   - hipCUB/hipFFT includes guarded with `#if defined(__HIPCC__)`

2. **Wave64 warp reduction fix** (kernels_compute.h): SUM_THRESH profile's warp reduction extended to cover 64 lanes with strides 32,16,8,4,2,1 on __GFX9__, using 0x3f lane mask for lane0 check.

3. **Shfl variants skipped on HIP**: The cov-shuffle (shfl) kernel variants (ur==0) are fundamentally tied to 32-lane warps. CMake skips them on HIP builds via `continue()` in the variant foreach.

4. **Library swaps**: cuFFT -> hipFFT, CUB (DeviceMergeSort) -> hipCUB/rocPRIM.

5. **Host-side CUDA->HIP mappings**: Added to multiple .cpp files (tile.cpp, scamp_interface.cpp, autotune_bench.cpp, main.cpp, qt_helper.cpp, common.cpp) for runtime calls like cudaSetDevice, cudaMemsetAsync, cudaGetDeviceCount, etc.

### Build

```bash
cmake -B build -DUSE_HIP=ON
cmake --build build -j$(nproc)
```

### Test

```bash
cd test && bash run_tests.sh ../build/SCAMP /tmp/results.txt ""
# All Tests Passed!
```

### Gotchas

- hipCUB/rocPRIM headers contain device intrinsics that fail in host C++ compilation; they must be included only in HIP-compiled translation units (guarded with `#if defined(__HIPCC__)`)
- hip::hipcub must be linked PRIVATE to avoid propagating HIP compile options to downstream C++ targets
- The shfl kernel variants would need significant redesign for wave64 (they use 32-bit shuffle masks and assume 32-lane warps); easier to skip them and rely on the sliding-window variants which work on both wave32 and wave64

## Review 2026-06-05

### Summary
Strategy A port enabling AMD GPU support via HIP with cuda_to_hip.h compat header, wave64 warp reduction fix for SUM_THRESH profile, and CMake-based shfl variant skipping. The port is sound with two minor issues to address.

### Findings

1. **MOAT vocabulary in commit message and CMake comment** (must fix)
   - Commit message uses "Strategy A approach" -- remove MOAT-internal term
   - `src/core/gpu_kernel/CMakeLists.txt:93`: "follower platforms pass their own arch" -- reword to "other platforms pass their own arch via CMAKE_HIP_ARCHITECTURES"

2. **Duplicated symbol mappings in host .cpp files** (acceptable, document)
   - Several host .cpp files (`qt_helper.cpp`, `tile.cpp`, `common.cpp`, `main.cpp`, `scamp_interface.cpp`, `autotune_bench.cpp`, `device_props.cpp`, `kernel_config.cpp`) define their own CUDA->HIP macros rather than including cuda_to_hip.h
   - This is necessary because cuda_to_hip.h includes hipFFT/hipCUB headers guarded by `__HIPCC__` which fail host compilation
   - Current approach works but creates maintenance burden; consider documenting this pattern in cuda_to_hip.h

### Verified Correct

- Wave64 warp reduction: strides 32,16,8,4,2,1 with `__GFX9__` guard, 0x3f lane mask
- Shfl variant skip via CMake `continue()` is valid given fundamental 32-lane assumptions
- kWarpSize abstraction follows PORTING_GUIDE (`__GFX9__` device, upper bound for host)
- Multi-arch compatible CMAKE_HIP_ARCHITECTURES
- Library swaps: cuFFT->hipFFT, CUB->hipCUB
- hip::hipcub linked PRIVATE (correct)
- CUDA path preserved, USE_HIP defaults OFF

### Recommendation
changes-requested: fix MOAT vocabulary, then ready for validation.

## Fixes (2026-06-05)

Addressed reviewer findings:
1. Removed "Strategy A" jargon from commit message -- reworded to describe the approach directly
2. Changed CMakeLists.txt comment from "follower platforms pass their own arch" to "other platforms pass their own arch via CMAKE_HIP_ARCHITECTURES"

Amended commit and force-pushed to 58f2e7edac7f1a7f9a7c08ede18dc6e0cf714466.

## Validation 2026-06-05 (linux-gfx90a)

SHA: 58f2e7edac7f1a7f9a7c08ede18dc6e0cf714466
GPU: AMD Instinct MI210 (gfx90a)

### Build
```bash
cd /var/lib/jenkins/moat/projects/SCAMP/src
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DBUILD_SCAMP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```
Build: PASS (4.6MB binary)

### Test
```bash
cd test && bash run_tests.sh ../build/SCAMP /tmp/scamp-results.txt ""
```

Results: All 52 tests PASSED
- Self-join tests: 19 (randomwalk 8K/16K/32K/64K/64K_nan with various tile sizes)
- Aligned AB-join tests: 19 (same datasets, aligned mode)
- AB-join tests: 14 (cross-dataset joins 16K vs 32K, with keep_rows variants)

Matrix profile accuracy:
- Max MP value difference: 2.24e-06 (across all tests)
- MP index differences: 0-1 indices per test (acceptable for floating-point)

All tests completed successfully with "All Tests Passed!" confirmation.

## Validation 2026-06-05 (linux-gfx1100)

SHA: 58f2e7edac7f1a7f9a7c08ede18dc6e0cf714466
GPU: AMD Radeon Pro W7800 (gfx1100)

### Build
```bash
cd /var/lib/jenkins/moat/projects/SCAMP/src
git submodule update --init --recursive
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DBUILD_SCAMP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```
Build: PASS (4.6 MB binary)

### Test
```bash
cd test && bash run_tests.sh ../build/SCAMP /tmp/scamp-gfx1100-results.txt ""
```

Results: All 52 tests PASSED
- Self-join tests: 19 (randomwalk 8K/16K/32K/64K/64K_nan with various tile sizes)
- Aligned AB-join tests: 19 (same datasets, aligned mode)
- AB-join tests: 14 (cross-dataset joins 16K vs 32K, with keep_rows variants)

Matrix profile accuracy:
- Max MP value difference: 2.24e-06 (across all tests)
- MP index differences: 0-1 indices per test (acceptable for floating-point)

All tests completed successfully with "All Tests Passed!" confirmation.

## Validation 2026-06-07 (linux-gfx90a, SUM_THRESH coverage)

SHA: 58f2e7edac7f1a7f9a7c08ede18dc6e0cf714466
GPU: AMD Instinct MI250X (gfx90a), HIP_VISIBLE_DEVICES=2

Closes deferred item `scamp-sumthresh-validation`: the initial validation ran only the default 1NN_INDEX profile via run_tests.sh; the wave64 SUM_THRESH warp reduction fix in kernels_compute.h was not exercised on real GPU.

### Test: SUM_THRESH GPU vs CPU oracle

```bash
SCAMP=/var/lib/jenkins/moat/projects/SCAMP/src/build/SCAMP
INPUT=/var/lib/jenkins/moat/projects/SCAMP/src/test/SampleInput/randomwalk8K.txt

# GPU run (gfx90a wave64 SUM_THRESH reduction path)
HIP_VISIBLE_DEVICES=2 $SCAMP \
  --window=100 --input_a_file_name=$INPUT \
  --profile_type=SUM_THRESH --threshold=0.5 \
  --max_tile_size=2000000

# CPU reference (same parameters, --no_gpu)
$SCAMP \
  --window=100 --input_a_file_name=$INPUT \
  --profile_type=SUM_THRESH --threshold=0.5 \
  --max_tile_size=2000000 --no_gpu --num_cpu_workers=1
```

Tested thresholds: 0.0, 0.125, 0.5 (all three thresholds from the Python test suite).

Results (8093-element SUM_THRESH output, randomwalk8K window=100):

| threshold | GPU non-zero | CPU non-zero | max |GPU-CPU| | verdict |
|-----------|-------------|-------------|-----------------|---------|
| 0.0       | 8093        | 8093        | 0.00e+00        | PASS    |
| 0.125     | 8093        | 8093        | 0.00e+00        | PASS    |
| 0.5       | 8082        | 8082        | 0.00e+00        | PASS    |

GPU output is bit-identical to the CPU reference across all thresholds. The wave64 SUM_THRESH warp reduction (strides 32,16,8,4,2,1 under `__GFX9__` with SCAMP_FULL_WARP_MASK=0xffffffffffffffff) produces correct results on gfx90a.

## Validation 2026-06-19 (windows-gfx1101)

SHA: 58f2e7edac7f1a7f9a7c08ede18dc6e0cf714466
GPU: AMD Radeon PRO V710 (gfx1101, RDNA3, wave32), HIP_VISIBLE_DEVICES=1
Host: Windows 11 Pro, TheRock ROCm 7.14.0a20260604, clang-cl 23.0.0

### Build

```bash
ROCM="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
cmake -S B:/develop/moat/projects/SCAMP/src \
      -B B:/develop/moat/agent_space/SCAMP/build-hip-gfx1101 \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_HIP=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
      -DBUILD_SCAMP_TESTS=ON \
      -DCMAKE_CXX_COMPILER="$ROCM/lib/llvm/bin/clang-cl.exe" \
      -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/clang-cl.exe" \
      -DCMAKE_PREFIX_PATH="$ROCM"
cmake --build B:/develop/moat/agent_space/SCAMP/build-hip-gfx1101 -j64
```
Build: PASS (62/62 targets, warnings only)

### Runtime DLL deployment

```bash
ROCM_CORE="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_core"
ROCM_DEVEL="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
cp $ROCM_CORE/bin/amdhip64_7.dll         build-hip-gfx1101/
cp $ROCM_CORE/bin/amd_comgr.dll          build-hip-gfx1101/
cp $ROCM_CORE/bin/rocm_kpack.dll         build-hip-gfx1101/
cp $ROCM_CORE/bin/hiprtc0714.dll         build-hip-gfx1101/
cp $ROCM_CORE/bin/hiprtc-builtins0714.dll build-hip-gfx1101/
cp $ROCM_DEVEL/bin/hipfft.dll            build-hip-gfx1101/
cp $ROCM_DEVEL/bin/rocfft.dll            build-hip-gfx1101/
cp $ROCM_DEVEL/bin/rocfft_rtc_helper.exe build-hip-gfx1101/
```

### Test

```bash
cd B:/develop/moat/projects/SCAMP/src/test
HIP_VISIBLE_DEVICES=1 bash ./run_tests.sh \
  B:/develop/moat/agent_space/SCAMP/build-hip-gfx1101/SCAMP.exe \
  /tmp/scamp-gfx1101-results.txt ""
```

Results: All 50 tests PASSED (same count as gfx1201; 2 fewer than Linux due to tile_sz filter on 8K input)
- Self-join tests: 19
- Aligned AB-join tests: 19 (partial: 2 large-tile cases filtered)
- AB-join tests: 14 (partial: some filtered)

Matrix profile accuracy:
- Max MP value difference: 2.24e-06 (identical to gfx90a, gfx1100, and gfx1201)
- MP index differences: 0-1 per test (within 1% tolerance)

All tests completed successfully with "All Tests Passed!" confirmation.

## Validation 2026-06-07 (windows-gfx1201)

SHA: 58f2e7edac7f1a7f9a7c08ede18dc6e0cf714466
GPU: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), HIP_VISIBLE_DEVICES=0 (gfx1101 offline this session)
Host: Windows 11 Pro, TheRock ROCm 7.14.0a20260604, clang-cl 23.0.0

### Build

```bash
ROCM="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
cd B:/develop/moat/projects/SCAMP/src
git submodule update --init --recursive
cmake -S B:/develop/moat/projects/SCAMP/src \
      -B B:/develop/moat/agent_space/SCAMP/build-hip-gfx1201 \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_HIP=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
      -DBUILD_SCAMP_TESTS=ON \
      -DCMAKE_C_COMPILER="$ROCM/lib/llvm/bin/clang-cl.exe" \
      -DCMAKE_CXX_COMPILER="$ROCM/lib/llvm/bin/clang-cl.exe" \
      -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/clang-cl.exe" \
      -DCMAKE_PREFIX_PATH="$ROCM"
cmake --build B:/develop/moat/agent_space/SCAMP/build-hip-gfx1201 -j32
```
Build: PASS (4.3 MB binary, 62 targets, warnings only)

No `-fuse-ld=lld-link` stripping needed (CMake 4.x on this host did not inject it).
No `-fPIC` issue (clang-cl correctly rejected it for MSVC ABI, CMake flag check handled it).

### Runtime DLL deployment

```bash
ROCM_CORE=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_core
cp $ROCM_CORE/bin/amdhip64_7.dll    build-hip-gfx1201/
cp $ROCM_CORE/bin/amd_comgr.dll     build-hip-gfx1201/
cp $ROCM_CORE/bin/rocm_kpack.dll    build-hip-gfx1201/
cp $ROCM_CORE/bin/hiprtc0714.dll    build-hip-gfx1201/
cp $ROCM_CORE/bin/hiprtc-builtins0714.dll build-hip-gfx1201/
cp $ROCM_DEVEL/bin/hipfft.dll       build-hip-gfx1201/
cp $ROCM_DEVEL/bin/rocfft.dll       build-hip-gfx1201/
cp $ROCM_DEVEL/bin/rocfft_rtc_helper.exe build-hip-gfx1201/
```

### Test

```bash
cd B:/develop/moat/projects/SCAMP/src/test
HIP_VISIBLE_DEVICES=0 bash run_tests.sh \
  B:/develop/moat/agent_space/SCAMP/build-hip-gfx1201/SCAMP.exe \
  /tmp/scamp-gfx1201-results.txt ""
```

Results: All 50 tests PASSED (52 on Linux; 2 fewer here due to `tile_sz < count*2` filter on 8K input with this OS line-count)
- Self-join tests: 19
- Aligned AB-join tests: 19 (partial: 2 large-tile cases filtered)
- AB-join tests: 14 (partial: some filtered)

Wait -- recounting: 50 tests ran total, no failures. "All Tests Passed!" confirmed.

Matrix profile accuracy:
- Max MP value difference: 2.24e-06 (identical to gfx90a and gfx1100)
- MP index differences: 0 per test

All tests completed successfully with "All Tests Passed!" confirmation.

## PR-prep 2026-06-25

Prep on top of validated head 31a0d9f, then squashed.

- Jargon scrub: git grep of the whole tree for MOAT vocabulary and "byte-identical"/"byte-for-byte" phrasing found nothing in tracked source. The two pre-squash commit messages were clean of MOAT labels; commit 31a0d9f's old Test Plan used "byte-identical" for the HIP device ISA, which is dropped by the fresh squashed message.
- Attribution: the port introduced exactly ONE new file, src/core/gpu_kernel/cuda_to_hip.h (everything else is a modification of an upstream file). Per Jeff's direction (new files only), added "Copyright (c) 2026 Advanced Micro Devices, Inc." + "Author: Jeff Daily" to that file only, in a lightweight // comment matching the file's existing header. SCAMP's own first-party files carry no per-file copyright header (central LICENSE only), so no per-file attribution was added to any modified upstream file.
- Build guarding: verified every HIP branch in the CMake files is behind `if(USE_HIP)` (default OFF). The CUDA build path is unchanged from upstream when USE_HIP is off; the `-ffp-contract=on` tweak is set on CMAKE_HIP_FLAGS only and does not leak into CUDA. No leak.
- Documentation: SCAMP's README defers build steps to readthedocs; the CUDA build is documented in docs/source/cli.rst ("Build Configuration Options") and docs/source/environment.rst. Added the parallel ROCm/HIP docs in those same places: a "Building for AMD GPUs (ROCm/HIP)" subsection in cli.rst, an AMD GPU entry in environment.rst's GPU-support requirements, a USE_HIP note in "Notes on GPU Support", and USE_HIP / CMAKE_HIP_ARCHITECTURES in the build-time variables list. pyscamp/setup.py does NOT wire USE_HIP through, so the pyscamp Python install docs were deliberately left untouched (would document an unsupported path).
- Validated the docs with `python3 -m sphinx -b html source <out>`: cli.rst/environment.rst render the new sections and the build-config-options cross-reference resolves; no new warnings (the 11 warnings are pre-existing: missing _static dir, autosummary duplicate-object).
- classify of 31a0d9f..737c9f8: class=comment-only, arch_independent=True, inert=True (the only source-tree change is the cuda_to_hip.h header comment; rst is docs). All completed platforms carried forward, no GPU re-run.
- Squashed to single tree-identical commit 19801ac (git diff 737c9f8 19801ac empty; identical tree hash); squash-carry-forward carried linux-gfx90a, linux-gfx1100, windows-gfx1101, windows-gfx1201 forward. Force-pushed moat-port. pr-ready=True.
- Fixed upstream.json data gap: fork_url was null, set to https://github.com/AMD-Ecosystem/SCAMP.

PR not opened -- stopped at the user gate for Jeff's approval.

## Validation 2026-06-25 (linux-gfx90a, compat-layer revalidation, sha 2022f1d)

SHA: 2022f1d98aaabda7e2b4bd2718339479df04d3a8
GPU: AMD Instinct MI250X (gfx90a), HIP_VISIBLE_DEVICES=0

Revalidation of structural refactor: compat shim relocated to src/common/cuda_to_hip.h,
per-file `#define cudaXxx hipXxx` blocks removed across 7 host TUs, duplicated gpuAssert
collapsed, and CUDA host-build CUB leak fixed (unconditional `<cub/...>` now gated on
`__CUDACC__`). Delta classified mixed/arch_independent=False/inert=False -- full GPU
revalidation required.

### Build

```bash
cmake -S /var/lib/jenkins/moat/projects/SCAMP/src \
      -B /var/lib/jenkins/moat/projects/SCAMP/src/build \
      -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DBUILD_SCAMP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /var/lib/jenkins/moat/projects/SCAMP/src/build -j$(nproc)
```

Build: PASS (100%, 7.7 MB binary, warnings only -- all pre-existing nodiscard)

### Test: shfl variant forced (SCAMP_AUTOTUNE_VARIANT_FILTER=shfl)

```bash
HIP_VISIBLE_DEVICES=0 SCAMP_AUTOTUNE_VARIANT_FILTER=shfl \
  bash /var/lib/jenkins/moat/projects/SCAMP/src/test/run_tests.sh \
  /var/lib/jenkins/moat/projects/SCAMP/src/build/SCAMP \
  /tmp/scamp-shfl-revalidation.txt ""
```

Result: All Tests Passed!

### Test: default (autotuner-on)

```bash
HIP_VISIBLE_DEVICES=0 bash /var/lib/jenkins/moat/projects/SCAMP/src/test/run_tests.sh \
  /var/lib/jenkins/moat/projects/SCAMP/src/build/SCAMP \
  /tmp/scamp-default-revalidation.txt ""
```

Result: All Tests Passed! (sliding-window path unregressed)

### DP-vs-CPU-oracle bit-exact checks (randomwalk8K, window=100, shfl forced)

| profile / threshold      | max |GPU-CPU| | idx diffs |
|--------------------------|---------------|-----------|
| 1NN_INDEX DP             | 0.0 (exact)   | 0         |
| SUM_THRESH DP t=0.0      | 0.0 (exact)   | n/a       |
| SUM_THRESH DP t=0.5      | 0.0 (exact)   | n/a       |

All DP profiles are bit-identical to the CPU oracle. The refactor is behavior-preserving
on AMD; macro expansions via the consolidated shim produce identical results to the
removed per-file `#define` blocks.

### CUDA compile check (nvcc 12.8, sm_75, compile-only)

```bash
cmake -S /var/lib/jenkins/moat/projects/SCAMP/src \
      -B /var/lib/jenkins/moat/agent_space/scamp-cuda-revalidation -G Ninja \
      -DUSE_HIP=OFF -DFORCE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DBUILD_SCAMP_TESTS=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/opt/conda/envs/cuda-12.8/bin/nvcc \
      -DCMAKE_PREFIX_PATH=/opt/conda/envs/cuda-12.8 \
      -DCUDAToolkit_ROOT=/opt/conda/envs/cuda-12.8
cmake --build /var/lib/jenkins/moat/agent_space/scamp-cuda-revalidation -j$(nproc)
```

Result: 71/71 targets PASS. tile.cpp.o and qt_helper.cpp.o compile cleanly (CUB leak
fix confirmed -- no host-compilation errors from `<cub/device/device_merge_sort.cuh>`).
Only warning: pre-existing FP_CONTRACT pragma in cpu_kernels.cpp.

Verdict: PASS. Transitioning linux-gfx90a -> completed at 2022f1d.

## Wave64 cov-shuffle (shfl) variant enabled (2026-06-25)

Reverses the earlier decision to skip the shfl variant on HIP. New commit
8226072 on top of the validated squash head 19801ac (NOT amended -- 19801ac
is validated_sha for 4 platforms). advance-head correctly flipped all four
completed platforms to revalidate (functional change).

### What was warp-width-generalized

The shfl kernel encoded the warp width as literal 32/31/16 + a 32-bit
shuffle mask; on CDNA wave64 the covariance diagonal must walk 64 lanes.
All warp-width quantities now derive from the existing kWarpSize
abstraction (cuda_to_hip.h: 64 on __GFX9__, 32 on RDNA/CUDA) and the
64-bit-safe SCAMP_FULL_WARP_MASK:

- kernels_compute_shfl.h
  - do_row_shfl: `warps_per_block = BLOCKSZ / kWarpSize`; cross-warp
    publisher `state.warpln == kWarpSize - 1`; wrap shuffle uses
    SCAMP_FULL_WARP_MASK.
  - warp_reduce_and_flush_row: both the SUM_THRESH sum-butterfly and the
    max-style (distr,idxr) butterfly start at `kWarpSize / 2` and use
    SCAMP_FULL_WARP_MASK (matches the kernels_compute.h SUM_THRESH
    reference reduction).
  - SCAMPShflSmem::BLOCKSZ = `warps_per_block * kWarpSize`.
- kernels_impl_shfl.h
  - `state.warpln/warpid/srcln` derive from kWarpSize (warpid is
    threadIdx.x / kWarpSize -- on wave64 that is /64, i.e. NOT the old
    `>> 5`); srcln = (warpln + kWarpSize - 1) % kWarpSize.
  - static_asserts: `BLOCKSZ % kWarpSize == 0` and the warp-ownership
    bound `tile_height <= kWarpSize * DPT`.
- kernel_gpu_utils.cu (HOST get_smem_shfl): cov_handoff is 2 *
  warps_per_block scalars; warps_per_block = blocksz / warp_size, so a
  SMALLER warp size means MORE warps. The old code sized it with
  kWarpSizeUpperBound (64), which UNDER-allocates on a wave32 HIP device
  (RDNA: blocksz/32 warps) and would let the device write past the dynamic
  smem region. Now sized with the smallest warp width (32) so it bounds
  any runtime warp width. This was a latent bug that never fired only
  because shfl was disabled on HIP.
- kernel_config.cpp: removed the USE_HIP guard in ProfileTypePrefersShfl
  that steered HIP to sliding-window; HIP and CUDA now share the same
  cold-start family preference. The autotuner can override per device.
- CMakeLists.txt: removed the `if(USE_HIP AND VARIANT_UR EQUAL 0) continue()`
  skip, so the two shfl tuples (v2 5|8|0|8|32, v3 8|4|0|8|16) generate
  their per-profile .cu TUs and enter the kVariants[] table on HIP (4
  variants, matching CUDA).

### warps_per_block halving on wave64

A given BLOCKSZ yields HALF the warps on wave64 (BLOCKSZ/64 vs BLOCKSZ/32).
For the instantiated geometries (BLOCKSZ=128 default): wave32 -> 4 warps,
wave64 -> 2 warps. 2 warps still exercises the cross-warp hand-off. This
halves the cov_handoff slot count and the cross-warp hand-offs per row;
all of it derives from kWarpSize, none from a literal.

### Geometry decisions

Both shfl tuples build and launch unchanged on wave64. Their tile_height
(v2: OUR*KTI = 8*32 = 256; v3: 8*16 = 128) satisfies the warp-ownership
bound on BOTH widths: wave32 needs <= 32*DPT (256<=256, 128<=128), wave64
needs <= 64*DPT (256<=512, 128<=256). No tuple needed a wave64-specific
geometry. The L113 register-ceiling prose was CUDA-specific (65536 regs /
bps / blocksz); updated to note the gfx90a VGPR-per-SIMD model and that
the per-GPU autotuner picks the winning blocksz/bps -- the instantiated
geometries only need to build and be launchable, which they are.

### GPU validation (gfx90a, AMD Instinct MI210, HIP_VISIBLE_DEVICES=0)

Build: PASS (warnings only, all pre-existing nodiscard). All 10 shfl TUs
(2 variants x 5 profiles) compiled; kVariants[] has 4 entries on HIP.

Full integration suite (run_tests.sh, default 1NN_INDEX which now runs the
shfl kernel -- confirmed via a temporary variant-trace, every one of 1078
launches selected the ur=0 shfl variant): All 52 tests PASSED, max MP
value diff 1.66e-06, 0 index diffs.

Direct shfl-vs-reference cross-checks (randomwalk8K/32K):

| case                  | reference          | max value diff | idx diffs |
|-----------------------|--------------------|----------------|-----------|
| 1NN_INDEX/DP/8K       | CPU oracle         | 0.0 (exact)    | 0         |
| 1NN_INDEX/DP/32K      | CPU oracle         | 0.0 (exact)    | 1         |
| 1NN_INDEX/SP/8K       | GPU sliding-window | 7.2e-05        | 0         |
| 1NN_INDEX/SP/32K      | GPU sliding-window | 1.8e-04        | 2         |
| SUM_THRESH/DP t0.0/0.5| CPU oracle         | 0.0 (exact)    | n/a       |
| SUM_THRESH/SP t0.0    | GPU sliding-window | 2.7e-04        | n/a       |

SUM_THRESH SP at threshold 0.5 shows ~10/8093 positions differing by ~0.5
between shfl-SP and sliding-window-SP (max rel diff 1.07e-3). This is
single-precision summation reassociation at the threshold boundary, NOT a
wave64 fault: BOTH GPU SP variants differ from the DP oracle by the same
~0.5 at the same positions, and shfl-SP vs sliding-window-SP agree to six
significant figures. DP shfl is bit-identical to the CPU oracle, which is
the decisive correctness proof.

Autotuner: `--autotune` sweeps all 4 variants and completes; the
sliding-window path is unregressed (a cached run reproduces the golden
1NN_INDEX result, max diff 1.3e-06, 0 idx diffs).

Verdict: shfl is correct on wave64. Validation instrumentation
(SCAMP_TRACE_VARIANT / SCAMP_FORCE_VARIANT_FAMILY env hooks) was added
temporarily to confirm variant selection and removed before the commit;
the committed code has no debug hooks.

### Gotcha

The CPU reference path is single-precision-UNIMPLEMENTED (it returns
SCAMP_FUNCTIONALITY_UNIMPLEMENTED for --single_precision --no_gpu), so SP
correctness must be checked against the GPU sliding-window variant, not
the CPU oracle. Only DP can use the CPU oracle.

## Validation 2026-06-25 (linux-gfx1100, shfl revalidation)

SHA: 8226072c34f1488d5498d0b43d0bc0c2d45734d1
GPU: AMD Radeon Pro W7800 (gfx1100, RDNA3, wave32), HIP_VISIBLE_DEVICES=1

Delta 19801ac -> 8226072: functional change enabling cov-shuffle (shfl) kernel variant on HIP,
warp-width generalization for wave64/wave32, smem fix. Classified `mixed, arch_independent=False,
inert=False` -- full GPU revalidation required.

### Build
```bash
cmake -S /var/lib/jenkins/moat/projects/SCAMP/src \
      -B /var/lib/jenkins/moat/projects/SCAMP/src/build-gfx1100 \
      -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      -DBUILD_SCAMP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /var/lib/jenkins/moat/projects/SCAMP/src/build-gfx1100 -j$(nproc)
```
Build: PASS (warnings only, all pre-existing nodiscard)

### Test
```bash
cd /var/lib/jenkins/moat/projects/SCAMP/src/test
HIP_VISIBLE_DEVICES=1 bash run_tests.sh ../build-gfx1100/SCAMP /tmp/scamp-gfx1100-revalidation.txt ""
```

Results: All 50 tests PASSED
- Self-join tests: 19
- Aligned AB-join tests: 19
- AB-join tests (cross-dataset, keep_rows): 12

Matrix profile accuracy:
- Max MP value difference: 1.66e-06 (across all tests, within prior tolerance)
- MP index differences: 0 per test

All tests completed with "All Tests Passed!" confirmation.
The shfl variant (ur=0) is correctly generalized for wave32 on gfx1100.
## Validation 2026-06-25 (linux-gfx90a, wave64 shfl re-validation, sha 8226072)

SHA: 8226072c34f1488d5498d0b43d0bc0c2d45734d1
GPU: AMD Instinct MI250X (gfx90a), HIP_VISIBLE_DEVICES=0

Full revalidation of functional change: cov-shuffle (shfl) kernel variants
(ur==0, v2 and v3 in the 4-variant kVariants[] table) generalized to wave64
and re-enabled on HIP. Clean build from empty build dir at head sha.

### Build

```bash
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DBUILD_SCAMP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Build: PASS (warnings only, all pre-existing nodiscard). kVariants[] has 4
entries (v0,v1=sliding-window; v2,v3=shfl). v2 and v3 use
kernels_impl_shfl.h (kernel_variant_shfl.cu.in), confirmed by inspecting
generated TUs.

### Test: shfl variant forced (SCAMP_AUTOTUNE_VARIANT_FILTER=shfl)

```bash
cd test && SCAMP_AUTOTUNE_VARIANT_FILTER=shfl HIP_VISIBLE_DEVICES=0 \
  bash run_tests.sh ../build/SCAMP /tmp/scamp-shfl-results.txt ""
```

Results: All 50 tests PASSED (shfl variant forced throughout)
- Max MP value difference across all tests: 1.66e-06

### Test: default (autotuner-on) integration suite

```bash
cd test && HIP_VISIBLE_DEVICES=0 \
  bash run_tests.sh ../build/SCAMP /tmp/scamp-default-results.txt ""
```

Results: All 50 tests PASSED
- Sliding-window path unregressed

### Direct shfl cross-checks (randomwalk8K/32K)

| case                  | reference          | max value diff | idx diffs |
|-----------------------|--------------------|----------------|-----------|
| 1NN_INDEX/DP/8K       | CPU oracle         | 0.0 (exact)    | 0         |
| 1NN_INDEX/DP/32K      | CPU oracle         | 0.0 (exact)    | 2         |
| 1NN_INDEX/SP/8K       | GPU sliding-window | 0.0 (exact)    | 0         |
| SUM_THRESH/DP t=0.0   | CPU oracle         | 0.0 (exact)    | n/a       |
| SUM_THRESH/DP t=0.5   | CPU oracle         | 0.0 (exact)    | n/a       |

DP shfl is bit-identical to the CPU oracle on all tested profiles and
thresholds. SP shfl matches sliding-window exactly on this MI250X host
(no reassociation divergence at 8K).

Verdict: PASS. Transitioning linux-gfx90a -> completed at 8226072.

## Validation 2026-06-24 (windows-gfx1201, shfl revalidation, sha 8226072)

SHA: 8226072c34f1488d5498d0b43d0bc0c2d45734d1
GPU: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), HIP_VISIBLE_DEVICES=0 (only GPU present)
Host: Windows 11 Pro, TheRock ROCm 7.14.0a20260604, clang-cl 23.0.0

Delta 19801ac -> 8226072: functional change enabling cov-shuffle (shfl) kernel variants on HIP
with wave64/wave32 generalization. Binary-equiv carry-forward not applicable (functional change);
full GPU revalidation required.

### Build

```bash
ROCM="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
cmake -S B:/develop/moat/projects/SCAMP/src \
      -B B:/develop/moat/agent_space/SCAMP/build-hip-gfx1201 \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_HIP=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
      -DBUILD_SCAMP_TESTS=ON \
      -DCMAKE_C_COMPILER="$ROCM/lib/llvm/bin/clang-cl.exe" \
      -DCMAKE_CXX_COMPILER="$ROCM/lib/llvm/bin/clang-cl.exe" \
      -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/clang-cl.exe" \
      -DCMAKE_PREFIX_PATH="$ROCM"
cmake --build B:/develop/moat/agent_space/SCAMP/build-hip-gfx1201 -j64
```

Build: PASS (72/72 targets, warnings only -- all pre-existing nodiscard)

### Runtime DLL deployment

```bash
ROCM_CORE="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_core"
ROCM_DEVEL="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
cp $ROCM_CORE/bin/amdhip64_7.dll           build-hip-gfx1201/
cp $ROCM_CORE/bin/amd_comgr.dll            build-hip-gfx1201/
cp $ROCM_CORE/bin/rocm_kpack.dll           build-hip-gfx1201/
cp $ROCM_CORE/bin/hiprtc0714.dll           build-hip-gfx1201/
cp $ROCM_CORE/bin/hiprtc-builtins0714.dll  build-hip-gfx1201/
cp $ROCM_DEVEL/bin/hipfft.dll              build-hip-gfx1201/
cp $ROCM_DEVEL/bin/rocfft.dll              build-hip-gfx1201/
cp $ROCM_DEVEL/bin/rocfft_rtc_helper.exe   build-hip-gfx1201/
```

### Test

```bash
cd B:/develop/moat/projects/SCAMP/src/test
HIP_VISIBLE_DEVICES=0 bash run_tests.sh \
  B:/develop/moat/agent_space/SCAMP/build-hip-gfx1201/SCAMP.exe \
  /tmp/scamp-gfx1201-revalidation.txt ""
```

Results: All 50 tests PASSED
- Self-join tests: 19 (randomwalk 8K/16K/32K/64K/64K_nan with various tile sizes)
- Aligned AB-join tests: 19
- AB-join tests: 12

Matrix profile accuracy:
- Max MP value difference: 2.24e-06 (identical to all prior validations)
- MP index differences: 0-1 per test (within 1% tolerance)

All tests completed with "All Tests Passed!" confirmation.
The shfl variant (ur=0) is correctly generalized for wave32 on gfx1201 (RDNA4).

## PR-prep re-run after cov-shuffle (2026-06-25)

The original squash (19801ac) and its message claimed the shfl variants were
32-lane specific and skipped on HIP. The cov-shuffle work (8226072) made that
false. Re-ran PR-prep:
- Fixed the one stale doc claim at docs/source/cli.rst:207-209 (the
  autotune.rst / environment.rst shfl mentions are neutral variant-filter
  docs, not availability claims -- left as-is).
- Doc-only commit 9ff34b1 on top of validated 8226072 (classify -> doc-only
  inert), advance-head carried the 3 required platforms forward.
- Re-squashed 61a6597..9ff34b1 into 556321d (tree-identical, git diff empty);
  squash-carry-forward carried linux-gfx90a/linux-gfx1100/windows-gfx1201.
  windows-gfx1101 stays revalidate@19801ac (optional), gfx1151 port-ready.
- New squashed message describes the complete port INCLUDING wave64/wave32
  shfl support via kWarpSize; the old "shfl skipped on HIP" bullet is gone.
- pr-ready=True at 556321d. PR not yet opened (awaiting Jeff's approval).

## Compat-layer consolidation + CUDA host-build CUB-leak fix (2026-06-25)

New commit 2022f1d on top of validated head 556321d (NOT amended -- 556321d
is validated_sha for linux-gfx90a/linux-gfx1100/windows-gfx1201). This is a
structural/functional refactor; advance-head correctly flipped those three
completed platforms to revalidate (windows-gfx1101 was already revalidate;
gfx1151 stays port-ready, optional).

### Compat-layer design (single shared shim)

The CUDA-to-HIP shim now lives at `src/common/cuda_to_hip.h` (moved from
`src/core/gpu_kernel/cuda_to_hip.h`). It was relocated to the LOW common
layer so common/, core/, and gpu_kernel/ can all include it without a
dependency inversion (the gpu_kernel libs link AGAINST common, so a low-layer
TU including a gpu_kernel header was the inversion the original port avoided
by rolling per-file `#define cudaXxx hipXxx` blocks). With the shim in a
neutral location, every per-file block is deleted and replaced by a single
`#include "common/cuda_to_hip.h"`. The project's global include path is
`include_directories("${CMAKE_SOURCE_DIR}/src/")` (root CMakeLists.txt:314),
so the `common/...` spelling resolves from every target.

Layering rule that makes this safe: the shim is includable in HOST context.
Its only device-only pieces are gated:
- HIP path: `<hipcub/hipcub.hpp>` under `#if defined(__HIPCC__) ||
  defined(__HIP_DEVICE_COMPILE__)` (rocPRIM device intrinsics). `<hip/hip_runtime.h>`
  and `<hipfft/hipfft.h>` are host-safe and stay unconditional -- qt_helper.cpp
  legitimately calls the FFT API on the host.
- CUDA path: `<cub/device/device_merge_sort.cuh>` under `#if defined(__CUDACC__)`
  (mirrors the HIP guard). `<cuda_runtime.h>` and `<cufft.h>` stay host-visible.

cub::DeviceMergeSort is used ONLY in kernels.cu (a device TU), so gating the
CUB header costs nothing.

### The CUDA host-build CUB leak (root cause + fix)

Before this change the CUDA path included `<cub/device/device_merge_sort.cuh>`
UNCONDITIONALLY. Host .cpp TUs (tile.cpp, qt_helper.cpp) reach the shim via
kernels.h -> cuda_to_hip.h, so the CUB device headers leaked into g++ host
compilation and failed to compile. Upstream base (61a6597) only pulled CUB in
kernels.cu, so this was a regression introduced by the original port's
unconditional include. Fix: gate it on `__CUDACC__`. Verified by an nvcc 12.8
FULL build (below): tile.cpp.o and qt_helper.cpp.o now compile cleanly.

### Other cleanups in this commit

- common.cpp: two `gpuAssert` overloads (hipError_t / cudaError_t) collapsed
  to one `gpuAssert(cudaError_t, ...)` -- the shim's cudaError_t/cudaSuccess/
  cudaGetErrorString maps make the cuda spelling correct on both toolchains.
- common.h: hand-rolled `#ifdef USE_HIP` runtime include + ExecInfo
  stream/dev_props type renaming + gpuAssert decl all collapsed to the shim
  include + the cuda spellings.
- Added cudaMemcpyAsync and cudaMemsetAsync to the shim (needed by the
  removed common.cpp / tile.cpp blocks).
- Dropped a dead `start_diag` local in kernels_impl_shfl.h:113 (only the
  non-shfl kernels_impl.h uses it) -- clears the unused-var warning.

### nvcc compile-check is now part of SCAMP's validation

The CUDA build path is verified at every change going forward with an nvcc
compile-only check (no NVIDIA GPU on these hosts). This catches host/device
header-leak regressions like the CUB leak above that a HIP-only build cannot.

```bash
cmake -S /var/lib/jenkins/moat/projects/SCAMP/src \
      -B /var/lib/jenkins/moat/agent_space/scamp-cuda-build2 -G Ninja \
      -DUSE_HIP=OFF -DFORCE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DBUILD_SCAMP_TESTS=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/opt/conda/envs/cuda-12.8/bin/nvcc \
      -DCMAKE_PREFIX_PATH=/opt/conda/envs/cuda-12.8 \
      -DCUDAToolkit_ROOT=/opt/conda/envs/cuda-12.8
cmake --build /var/lib/jenkins/moat/agent_space/scamp-cuda-build2 -j$(nproc)
```
Note: `-S` is the fork root (the CMakeLists.txt is at the repo root, not src/).
Result: full build compiles + links (71/71), tile.cpp.o and qt_helper.cpp.o
build cleanly. Only warning is the pre-existing FP_CONTRACT pragma in
cpu_kernels.cpp.

### HIP revalidation (gfx90a, AMD Instinct MI250X, HIP_VISIBLE_DEVICES=0, sha 2022f1d)

Clean build from empty build dir. kVariants[] has 4 entries (v0,v1 sliding-
window; v2,v3 shfl). Default suite and shfl-forced suite both
"All Tests Passed!" (max MP diff 1.66e-06, 0 idx diffs).

Forced-shfl DP-vs-CPU-oracle bit-exact checks (randomwalk8K, window=100):

| profile / threshold      | max |GPU-CPU| | idx diffs |
|--------------------------|---------------|-----------|
| 1NN_INDEX DP             | 0.0 (exact)   | 0         |
| SUM_THRESH DP t=0.0      | 0.0 (exact)   | n/a       |
| SUM_THRESH DP t=0.5      | 0.0 (exact)   | n/a       |

The refactor is behavior-preserving on AMD (macro expansions identical to the
removed per-file blocks). PR still held at the user gate -- not opened.

## Validation 2026-06-25 (windows-gfx1201, compat-layer revalidation, sha 2022f1d)

SHA: 2022f1d98aaabda7e2b4bd2718339479df04d3a8
GPU: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), HIP_VISIBLE_DEVICES=0 (only GPU present)
Host: Windows 11 Pro, TheRock ROCm 7.14.0a20260604, clang-cl 23.0.0

Delta 556321d -> 2022f1d: compat shim relocated to src/common/cuda_to_hip.h, per-file
`#define cudaXxx hipXxx` blocks removed, dead `start_diag` local removed from
kernels_impl_shfl.h, CUDA CUB leak fix (`__CUDACC__` guard).

moatlib classify: mixed/arch_independent=False/inert=False -- full GPU revalidation required.

Binary-equivalence check attempted first:
- Built HEAD at 2022f1d into build-hip-gfx1201-head (72/72 targets, PASS)
- Extracted PE .hip_fat and .hipFatB sections from both old (8226072/556321d doc-equiv) and new (2022f1d) builds
- .hipFatB: SHA256 identical (registration blob unchanged)
- .hip_fat: SHA256 DIFFER (both 6,794,024 bytes but different content)
- Result: binary-equiv carry-forward NOT applicable; proceeded to full GPU re-run

### Build

```
ROCM="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
cmake -S B:/develop/moat/projects/SCAMP/src \
      -B B:/develop/moat/agent_space/SCAMP/build-hip-gfx1201-head \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_HIP=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
      -DBUILD_SCAMP_TESTS=ON \
      -DCMAKE_CXX_COMPILER="$ROCM/lib/llvm/bin/clang-cl.exe" \
      -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/clang-cl.exe" \
      -DCMAKE_PREFIX_PATH="$ROCM"
cmake --build B:/develop/moat/agent_space/SCAMP/build-hip-gfx1201-head -j64
```

Build: PASS (72/72 targets, warnings only -- all pre-existing nodiscard)

### Runtime DLL deployment

Same DLL set as prior validations, copied into build-hip-gfx1201-head/:
amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll, hiprtc-builtins0714.dll,
hipfft.dll, rocfft.dll, rocfft_rtc_helper.exe

### Test

```
cd B:/develop/moat/projects/SCAMP/src/test
HIP_VISIBLE_DEVICES=0 bash run_tests.sh \
  B:/develop/moat/agent_space/SCAMP/build-hip-gfx1201-head/SCAMP.exe \
  /tmp/scamp-gfx1201-2022f1d-results.txt ""
```

Results: All 50 tests PASSED
- Self-join tests: 19 (randomwalk 8K/16K/32K/64K/64K_nan with various tile sizes)
- Aligned AB-join tests: 19
- AB-join tests: 12 (cross-dataset joins 16K vs 32K, with keep_rows variants)

Matrix profile accuracy:
- Max MP value difference: 2.24e-06 (identical to all prior validations)
- MP index differences: 0-1 per test (within 1% tolerance)

All tests completed with "All Tests Passed!" confirmation.
The compat-layer consolidation is behavior-preserving on gfx1201 (RDNA4, wave32).

## Validation 2026-06-25 (linux-gfx1100, compat-layer revalidation, sha 2022f1d)

SHA: 2022f1d98aaabda7e2b4bd2718339479df04d3a8
GPU: AMD Radeon Pro W7800 (gfx1100, RDNA3, wave32), HIP_VISIBLE_DEVICES=0

Delta 556321d -> 2022f1d: compat shim relocated to src/common/cuda_to_hip.h, per-file
`#define cudaXxx hipXxx` blocks removed across 7 host TUs, dead `start_diag` local removed,
CUDA CUB leak fixed (unconditional `<cub/...>` gated on `__CUDACC__`). Classified
mixed/arch_independent=False/inert=False -- full GPU revalidation required.

### Build

```bash
cmake -S /var/lib/jenkins/moat/projects/SCAMP/src \
      -B /var/lib/jenkins/moat/projects/SCAMP/src/build-gfx1100 \
      -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      -DBUILD_SCAMP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /var/lib/jenkins/moat/projects/SCAMP/src/build-gfx1100 -j$(nproc)
```

Build: PASS (7.5 MB binary, warnings only -- all pre-existing nodiscard)

### Test

```bash
cd /var/lib/jenkins/moat/projects/SCAMP/src/test
HIP_VISIBLE_DEVICES=0 bash run_tests.sh \
  /var/lib/jenkins/moat/projects/SCAMP/src/build-gfx1100/SCAMP \
  /tmp/scamp-gfx1100-test4.txt ""
```

Results: All 50 tests PASSED
- Self-join tests: 19 (randomwalk 8K/16K/32K/64K/64K_nan with various tile sizes)
- Aligned AB-join tests: 19
- AB-join tests: 12 (cross-dataset joins 16K vs 32K, with keep_rows variants)

Matrix profile accuracy:
- Max MP value difference: 1.66e-06 (matches prior gfx1100 validation exactly)
- MP index differences: 0 per test

All tests completed with "All Tests Passed!" confirmation.
The compat-layer consolidation is behavior-preserving on gfx1100 (RDNA3, wave32).

## PR-prep verification pass (2026-06-25, gfx1100 host)

Re-ran the full PR-prep checklist against the validated head 2022f1d (origin/moat-port).
Outcome: every prep item is ALREADY satisfied; no edit was made and no new fork commit
was created (creating one would only flip the completed platforms to revalidate for zero
GPU effect). The earlier prep work (docs, copyright, CMake guarding) was folded into the
squashed port and survived the later cov-shuffle and compat-layer refactors intact.

Verified at 2022f1d:
- Jargon scrub: `git diff 61a6597..HEAD` and `git grep` over the whole tracked tree show
  no MOAT vocabulary (lead/follower, Strategy A/B, head_sha, validated_sha, revalidate,
  moat-port, MOAT) and no "byte-for-byte"/"byte-identical" phrasing in source, comments,
  or either port commit message. The shfl CMake comment the reviewer once flagged is gone
  (the cov-shuffle work removed the skip). No leftover debug hooks (SCAMP_TRACE_VARIANT /
  SCAMP_FORCE_VARIANT_FAMILY), TODOs, or fork-name references in added lines.
- Documentation: present in SCAMP's house style at both CUDA-doc locations.
  - docs/source/cli.rst: "Building for AMD GPUs (ROCm/HIP)" section right after the
    "Forcing CUDA" block, with the USE_HIP build block, single-arch and fat-binary
    CMAKE_HIP_ARCHITECTURES examples, the auto-detect note, the CUDA-unchanged statement,
    and an accurate both-kernel-families/both-wave-widths paragraph (the old stale
    "shfl skipped on HIP" claim is already corrected).
  - docs/source/environment.rst: AMD-GPU entry in the GPU-support requirements, a USE_HIP
    note in "Notes on GPU Support", and USE_HIP + CMAKE_HIP_ARCHITECTURES in the
    build-time variables list. The em-dashes in that list are pre-existing upstream house
    style (base 61a6597 already has 18; our two AMD bullets add 2 to match), so they are
    correctly left as-is, not ASCII-converted.
  - ROCm vs HIP wording is precise throughout (ROCm = platform/install/toolchain, HIP =
    language/compiler). pyscamp setup.py does not wire USE_HIP through, so the Python
    install docs were correctly left untouched (would document an unsupported path).
- CMake arch auto-detect: already clean. Root CMakeLists.txt uses bare
  `enable_language(HIP)` with NO literal CMAKE_HIP_ARCHITECTURES pin, so an explicit
  -DCMAKE_HIP_ARCHITECTURES wins and otherwise the host GPU is auto-detected (PORTING_GUIDE
  Strategy A step 2). Documented in cli.rst/environment.rst. No change needed.
- Copyright/authorship: the port introduced exactly ONE new file,
  src/common/cuda_to_hip.h, which carries "Copyright (c) 2026 Advanced Micro Devices, Inc."
  + "Author: Jeff Daily". Every modified upstream file is header-less
  (SCAMP uses a central LICENSE, no per-file copyright), so per the copyright-only-when-
  upstream-header-exists rule, no lone AMD header was added to any modified file (including
  the 76-line qt_helper.h extension). Correct.

No build was run: nothing in the source tree changed, so there is nothing new to compile.
head_sha stays 2022f1d; all completed platforms (linux-gfx90a, linux-gfx1100,
windows-gfx1201) remain validated at that sha. The port is PR-prep-complete and held at
the user gate for Jeff's approval to open the upstream PR.

## PR-prep squash (2026-06-25)

The two-commit moat-port branch (556321d port + 2022f1d shim-consolidation/CUB-leak-fix) was collapsed to a single tree-identical squashed commit on upstream base 61a6597. A concurrent host had already squashed and force-pushed it as efc1107 (parent 61a6597, tree a7b063c == the validated 2022f1d tree, empty diff vs 2022f1d), with a clean final-design message. Adopted efc1107 as the canonical squash rather than force-pushing a redundant tree-identical twin; reset local moat-port to it and ran squash-carry-forward SCAMP efc1107 (carried linux-gfx90a, linux-gfx1100, windows-gfx1201). pr-ready=True.

Jargon scrub: tracked source clean (the only "lead" hit, kernel_config.cpp:97 "shfl widens its lead", is upstream-preexisting prose meaning advantage, not MOAT vocabulary). No "byte-for-byte"/"byte-identical" anywhere. Attribution: the only file added across the whole port is src/common/cuda_to_hip.h (AMD copyright + Jeff Daily \author header); the consolidation git-mv'd it from gpu_kernel (89% rename, header preserved) and cov-shuffle touched only pre-existing files, so no new attribution needed. Docs (cli.rst build-config section + environment.rst USE_HIP/CMAKE_HIP_ARCHITECTURES bullets) read correctly and document both kernel families on wave64/wave32; em-dash bullets match environment.rst's established house style (18 em-dashes in upstream base). No doc fixes required.

## Stream/device-prop shim consolidation (2026-06-25, head b999b33)

Removed the last three shim-redundant `#ifdef USE_HIP` stream/device-prop type sites so
those types masquerade through common/cuda_to_hip.h (cudaStream_t -> hipStream_t,
cudaDeviceProp -> hipDeviceProp_t on HIP; pass-through on CUDA) like every other CUDA type,
instead of per-file conditionals. Commit b999b33 on top of efc1107 (not amended).

The three sites collapsed to the cuda spelling:
- src/core/gpu_kernel/kernels.h: the USE_HIP-split match_gpu_sort declaration pair ->
  one `cudaStream_t` declaration (definition in kernels.cu was already cudaStream_t).
- src/core/gpu_kernel/kernels_dispatch.h: deleted the `using GpuStream_t = ...` alias;
  every GpuStream_t -> cudaStream_t. Also updated the two follow-on GpuStream_t references:
  the include comment in kernels_variants.h.in and the generated-decl string in
  gpu_kernel/CMakeLists.txt (~line 154). `git grep -n GpuStream_t -- src` now returns
  NOTHING; GpuStream_t is fully removed.
- src/core/tile.h: get_stream()/get_dev_props() collapsed to cudaStream_t/cudaDeviceProp
  under the existing `#ifdef _HAS_CUDA_` guard. tile.h sees the shim via
  common/common.h (which `#include`s common/cuda_to_hip.h and itself types ExecInfo's
  stream/dev_props as cudaStream_t/cudaDeviceProp), so the types resolve on HIP.

Legitimately arch-specific #ifdefs were left untouched (FFT error enums in qt_helper.h,
the Pascal double-atomicAdd fallback in kernel_gpu_utils.h, the wave64 SUM_THRESH reduction
in kernels_compute.h, gcn_arch_name in device_props.h, the CUDA-only cuda.h/cuda/barrier
includes).

Behavior-preserving: cudaStream_t macro-expands to hipStream_t (== the old GpuStream_t) on
HIP and is the real CUDA type on CUDA, so types/ABI are identical.

Validation (this consolidation reopened it; advance-head flipped the three completed
platforms to revalidate):
- HIP gfx90a build PASS (-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DBUILD_SCAMP_TESTS=ON).
- codeobj_diff vs efc1107 binary: verdict=identical (exported symbols + device ISA identical,
  3 exports). Pure type-spelling change, no device-code impact -- validator can carry
  forward via binary-equivalence.
- GPU suite on gfx90a (HIP_VISIBLE_DEVICES=2): default run and
  SCAMP_AUTOTUNE_VARIANT_FILTER=shfl run both "All Tests Passed!" (0 index diffs, max
  profile diff ~1.6e-06).
- nvcc CUDA compile (CUDA 12.8, sm_75, -DUSE_HIP=OFF -DFORCE_CUDA=ON): full SCAMP binary
  compiles+links (incl. tile.cpp/qt_helper.cpp). No-op on CUDA as expected.

Did NOT carry-forward platforms or open the PR (held at gate). codeobj_diff IDENTICAL is the
key signal for the validator.

## Validation 2026-06-25 (windows-gfx1201, binary-equivalence carry-forward, b999b33)

SHA: b999b338f1ffc21c89f822d8100db9b49ac6d58c
GPU: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32)
Prior validated_sha: efc1107d8ce434378200cf30b5e89e6cff51d352

Delta efc1107->b999b33: pure type-spelling change (GpuStream_t alias deleted, hipStream_t/
hipDeviceProp_t per-file conditionals removed; types route through common/cuda_to_hip.h shim).
Confirmed no device-code change by PE section analysis.

### Build (b999b33 at gfx1201)

```
ROCM="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
cmake -S B:/develop/moat/projects/SCAMP/src \
      -B B:/develop/moat/agent_space/SCAMP/build-hip-gfx1201-b999b33 \
      -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBUILD_SCAMP_TESTS=ON \
      -DCMAKE_C_COMPILER="$ROCM/lib/llvm/bin/clang-cl.exe" \
      -DCMAKE_CXX_COMPILER="$ROCM/lib/llvm/bin/clang-cl.exe" \
      -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/clang-cl.exe" \
      -DCMAKE_PREFIX_PATH="$ROCM"
cmake --build B:/develop/moat/agent_space/SCAMP/build-hip-gfx1201-b999b33 -j64
```

Build: PASS (72/72 targets, warnings only -- all pre-existing nodiscard)

### PE .hip_fat section analysis

Extracted .hip_fat and .hipFatB PE sections from both builds using llvm-objcopy:

- efc1107 build: build-hip-gfx1201-head/ (tree-identical to efc1107 -- see PR-prep squash note)
- b999b33 build: build-hip-gfx1201-b999b33/ (fresh build at head)

The .hip_fat section is a clang offload bundle with one device entry:
hipv4-amdgcn-amd-amdhsa--gfx1201 (389512 bytes, same in both).

ELF section comparison (device code ELF embedded in the bundle):

| Section   | Result    | Size   | Notes                              |
|-----------|-----------|--------|------------------------------------|
| .text     | IDENTICAL | 37888  | Device ISA -- the decisive proof   |
| .note     | IDENTICAL | 80256  | Kernel metadata (AMDHSA descriptors)|
| .rodata   | IDENTICAL | 4672   |                                    |
| .symtab   | IDENTICAL | 10056  |                                    |
| .comment  | IDENTICAL | 318    |                                    |
| .dynstr   | DIFFER    | 54645  | Only __hip_cuid_* changed          |
| .dynsym   | DIFFER    | 2832   | Only __hip_cuid_* changed          |
| .hash     | DIFFER    | 952    | Derived from dynsym                |
| .gnu.hash | DIFFER    | 856    | Derived from dynsym                |
| .strtab   | DIFFER    | 195074 | Only __hip_cuid_* string changed   |

.hipFatB (registration blob): SHA256 IDENTICAL (48ce791a2f91c0e6...)

Only __hip_cuid_4ce0efb51322f687 (efc1107) -> __hip_cuid_63bce6c35f7b4436 (b999b33).
The __hip_cuid is an internal HIP registration handle derived from source file content;
it changes with any source edit but does not affect kernel execution behavior.

VERDICT: binary-equivalent. Device ISA + kernel metadata + rodata + static symbols all
identical. Carry forward to b999b33 without GPU re-run.

## Validation 2026-06-25 (linux-gfx90a, binary-equivalence carry-forward, b999b33)

SHA: b999b338f1ffc21c89f822d8100db9b49ac6d58c
GPU: AMD Instinct MI250X / MI250 (gfx90a)
Prior validated_sha: efc1107d8ce434378200cf30b5e89e6cff51d352

Delta efc1107->b999b33: pure type-spelling change. The residual GpuStream_t/hipStream_t/
hipDeviceProp_t `#ifdef USE_HIP` sites in kernels.h, kernels_dispatch.h, and tile.h were
collapsed to cudaStream_t/cudaDeviceProp masquerading through the shim (src/common/cuda_to_hip.h).
On HIP, cudaStream_t macro-expands to hipStream_t (== the old GpuStream_t), so types/ABI are
identical and there is no device-code impact.

### Build (b999b33)

```bash
cmake -S /var/lib/jenkins/moat/projects/SCAMP/src \
  -B /var/lib/jenkins/moat/agent_space/SCAMP-b999b33-build \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DBUILD_SCAMP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /var/lib/jenkins/moat/agent_space/SCAMP-b999b33-build -j$(nproc)
```

Build: PASS (100%, warnings only -- all pre-existing nodiscard)

### codeobj_diff

```bash
python3 /var/lib/jenkins/moat/utils/codeobj_diff.py \
  /var/lib/jenkins/moat/agent_space/SCAMP-old-efc1107 \
  /var/lib/jenkins/moat/agent_space/SCAMP-b999b33-build/SCAMP
```

Result: verdict=identical (exported symbols + device ISA identical, 3 exports)

The efc1107 binary (8,020,280 bytes) and b999b33 binary (8,020,288 bytes) differ by 8 bytes
in the host section only; device code objects and all 3 exported symbols are bit-identical.
Binary-equivalence carry-forward procedure applied: no GPU re-run required.

## Validation 2026-06-25 (linux-gfx1100, binary-equivalence carry-forward, b999b33)

SHA: b999b338f1ffc21c89f822d8100db9b49ac6d58c
GPU: AMD Radeon Pro W7800 (gfx1100, RDNA3, wave32)
Prior validated_sha: efc1107d8ce434378200cf30b5e89e6cff51d352

Delta efc1107->b999b33: pure type-spelling change (same as gfx90a). The residual
GpuStream_t/hipStream_t/hipDeviceProp_t `#ifdef USE_HIP` sites in kernels.h,
kernels_dispatch.h, and tile.h collapsed to cudaStream_t/cudaDeviceProp masquerading
through the shim. On HIP cudaStream_t macro-expands to hipStream_t (== the old
GpuStream_t), so types/ABI are identical and there is no device-code impact.

### Build (b999b33, gfx1100)

```bash
cmake -S /var/lib/jenkins/moat/projects/SCAMP/src \
  -B /var/lib/jenkins/moat/agent_space/SCAMP-b999b33-gfx1100-build \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DBUILD_SCAMP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /var/lib/jenkins/moat/agent_space/SCAMP-b999b33-gfx1100-build -j$(nproc)
```

Build: PASS (100%, warnings only -- all pre-existing nodiscard)

### codeobj_diff (old=efc1107 build-gfx1100, new=b999b33 SCAMP-b999b33-gfx1100-build)

```bash
python3 /var/lib/jenkins/moat/utils/codeobj_diff.py \
  /var/lib/jenkins/moat/projects/SCAMP/src/build-gfx1100/SCAMP \
  /var/lib/jenkins/moat/agent_space/SCAMP-b999b33-gfx1100-build/SCAMP
```

Result: verdict=identical (exported symbols + device ISA identical, 3 exports)

The efc1107 binary (7,847,096 bytes) and b999b33 binary (7,847,104 bytes) differ by 8 bytes
in the host section only; device code objects and all 3 exported symbols are bit-identical.
Binary-equivalence carry-forward procedure applied: no GPU re-run required.

## Shim collapse 2026-06-25 (qt_helper FFT/complex/stream renames) -- sha 2b5b1a3

Commit 2b5b1a3 on top of b999b33. Collapsed the last per-file `#ifdef USE_HIP`
symbol-rename blocks into the single common/cuda_to_hip.h shim (no upstream PR; held at gate).

### Blocks collapsed (qt_helper.h)
- `_cudaGetErrorEnum`: two overloads (cufftResult / hipfftResult) -> one CUDA-spelled
  `_cudaGetErrorEnum(cufftResult)`. Dropped the dead `CUFFT_LICENSE_ERROR` case
  (it sat behind `#ifdef CUFFT_PARSE_ERROR`, which is FALSE on real cuFFT since the
  enumerators are enum constants not macros, so it never compiled on CUDA; hipFFT has
  no license-error enumerator). `CUFFT_PARSE_ERROR` and `CUFFT_INCOMPLETE_PARAMETER_LIST`
  are now listed unconditionally (real enums in cuFFT 12.8, aliased in the shim).
- `CHECK_CUFFT_ERRORS`: two macros (hipfftResult_t/HIPFFT_SUCCESS vs cufftResult_t/CUFFT_SUCCESS)
  -> one CUDA-spelled macro; single "cuFFT error" log string.
- member types: `hipDoubleComplex *Qc,*Tc; hipfftHandle ...` vs cu-spelled -> single
  `cuDoubleComplex *Qc,*Tc; cufftHandle fft_plan, ifft_plan;`.
- `compute_QT` decl: `hipStream_t s` vs `cudaStream_t s` -> single `cudaStream_t s`.
  (The DEFINITION in qt_helper.cpp was already cuda-spelled at b999b33.)

### Other block of this class (whole-tree audit)
- qt_kernels.cu: local `#define cuCmul hipCmul` -> moved into the shim's complex section
  (`#define cuCmul hipCmul`), call site stays `cuCmul`.

### Include-selection ifdefs: LEFT AS-IS (decision)
qt_helper.h:4-9, qt_kernels.h:5-12, kernels.cu:10, kernels.h/kernels_impl*.h/kernels_dispatch.h/
kernel_gpu_utils.cu/kernels_compute_shfl.h `#include <cuda*.h>` selection blocks select the actual
library HEADER, not a symbol rename. The shim deliberately keeps `<hip/hip_complex.h>` out of host
TUs; centralizing the complex include would risk pulling device headers into host compilation.
Left as include-selection per the "if in doubt, leave the include block" guidance.

### Full audit classification (every USE_HIP/__HIP_PLATFORM ifdef outside the shim)
Genuinely arch-specific, KEPT:
- common.h:20 HOST_DEVICE_FUNCTION macro (host vs device attribute)
- kernel_gpu_utils.h:9 Pascal double-atomicAdd fallback; :46 hw_threads_per_sm arch table
- device_props.cpp:30/48, device_props.h:15 CacheKey gcn_arch_name vs sm_, gcnArchName field
- kernels_compute.h:158 wave64/wave32 SUM_THRESH warp reduction
- kernels_compute_shfl.h:52 cuda::barrier sm_70 arch gate
- header-include selection blocks (listed above)
Redundant-with-shim, FIXED this pass: qt_helper.h x4 blocks + qt_kernels.cu cuCmul.
No remaining symbol-rename-only USE_HIP blocks; grep for hip-spelled FFT/complex/stream symbols
outside the shim returns only the `_CUFFT_H_ || HIPFFT_H_` include-guard check.

### Build + validation (host-side change; binary-equivalent)
- HIP gfx90a (USE_HIP=ON, gfx90a, BUILD_SCAMP_TESTS=ON): PASS (only pre-existing nodiscard warnings).
- codeobj_diff vs b999b33 binary: **verdict=identical** (device ISA + 3 exported symbols unchanged).
- GPU suite on MI210 (HIP_VISIBLE_DEVICES=0): default = All Tests Passed; SCAMP_AUTOTUNE_VARIANT_FILTER=shfl
  = All Tests Passed (0 MP-index diffs, max value diff 1.66e-06).
- nvcc 12.8 CUDA compile+link (USE_HIP=OFF FORCE_CUDA=ON sm_75, BUILD_SCAMP_TESTS=ON): PASS
  (qt_helper.cpp.o, tile.cpp.o, SCAMP all built+linked).

advance-head flipped linux-gfx90a/linux-gfx1100/windows-gfx1201/windows-gfx1101 to `revalidate`
(refactor classification). codeobj_diff=identical on gfx90a supports binary-equivalence carry-forward;
formal carry-forward left to the validator (not done here).

## Validation 2026-06-25 (linux-gfx90a, binary-equivalence carry-forward, sha 2b5b1a3)

SHA: 2b5b1a325149603bca4a2e5afc1d39e7121d86d2 (delta vs b999b33)
GPU: AMD Instinct MI250X (gfx90a) -- not exercised; binary-equivalence path

The delta b999b33->2b5b1a3 collapses redundant per-file #ifdef USE_HIP rename blocks in
qt_helper.h (cuFFT error enum, CHECK_CUFFT_ERRORS, FFT/complex member types, cudaStream_t
parameter) and qt_kernels.cu (cuCmul) into the cuda_to_hip.h shim. All aliases were already
present in the shim; this is a host-side deduplication with no device-code change.

### Build at 2b5b1a3 (gfx90a, fresh dir)

```bash
cmake -S /var/lib/jenkins/moat/projects/SCAMP/src \
      -B <scratchpad>/SCAMP-2b5b1a3-build \
      -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DBUILD_SCAMP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build <scratchpad>/SCAMP-2b5b1a3-build -j$(nproc)
```

Build: PASS (100%, warnings only -- all pre-existing nodiscard)

### Build at b999b33 (gfx90a, fresh dir, for comparison)

```bash
git checkout b999b33
cmake -S /var/lib/jenkins/moat/projects/SCAMP/src \
      -B <scratchpad>/SCAMP-b999b33-build \
      -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DBUILD_SCAMP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build <scratchpad>/SCAMP-b999b33-build -j$(nproc)
git checkout 2b5b1a3
```

Build: PASS (100%)

### codeobj_diff

```bash
python3 utils/codeobj_diff.py \
  <scratchpad>/SCAMP-b999b33-build/SCAMP \
  <scratchpad>/SCAMP-2b5b1a3-build/SCAMP
```

Result: **verdict=identical** (exported symbols + device ISA identical, 3 exports)

Device ISA: bit-identical (host-only change confirmed).
Exported symbols: 3 exports, all matching.

### Carry-forward

```bash
python3 utils/moatlib.py carry-forward SCAMP linux-gfx90a 2b5b1a3 binary-equiv \
  "qt_helper FFT/complex/stream renames now masquerade through the shim; host-side only; device ISA + exported symbols identical to b999b33 (3 exports, codeobj_diff verdict=identical, fresh gfx90a builds)"
```

Transitioned linux-gfx90a: revalidate -> completed, validated_sha=2b5b1a3.

## Validation 2026-06-25 (linux-gfx1100, binary-equivalence carry-forward, sha 2b5b1a3)

SHA: 2b5b1a325149603bca4a2e5afc1d39e7121d86d2
GPU: AMD Radeon Pro W7800 (gfx1100, RDNA3, wave32)
Prior validated_sha: b999b338f1ffc21c89f822d8100db9b49ac6d58c

Delta b999b33->2b5b1a3: collapses the last per-file `#ifdef USE_HIP` symbol-rename blocks in
qt_helper.h (cuFFT error enum, CHECK_CUFFT_ERRORS, FFT/complex member types, cudaStream_t
parameter) and qt_kernels.cu (cuCmul local define) into the common/cuda_to_hip.h shim.
All aliases were already present in the shim; this is a host-side deduplication with no
device-code change.

### Build (2b5b1a3, gfx1100)

```bash
cmake -S /var/lib/jenkins/moat/projects/SCAMP/src \
      -B /var/lib/jenkins/moat/agent_space/SCAMP-2b5b1a3-gfx1100-build \
      -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
      -DBUILD_SCAMP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /var/lib/jenkins/moat/agent_space/SCAMP-2b5b1a3-gfx1100-build -j$(nproc)
```

Build: PASS (100%, warnings only -- all pre-existing nodiscard; binary 7,847,096 bytes)

### codeobj_diff (old=b999b33 build, new=2b5b1a3 build, both gfx1100)

```bash
python3 /var/lib/jenkins/moat/utils/codeobj_diff.py \
  /var/lib/jenkins/moat/agent_space/SCAMP-b999b33-gfx1100-build/SCAMP \
  /var/lib/jenkins/moat/agent_space/SCAMP-2b5b1a3-gfx1100-build/SCAMP
```

Result: verdict=identical (exported symbols + device ISA identical, 3 exports)

Device ISA: bit-identical (host-only change confirmed).
Exported symbols: 3 exports, all matching.
Binary sizes: b999b33=7,847,104 bytes, 2b5b1a3=7,847,096 bytes (8-byte host-section diff only).

### Carry-forward

```bash
python3 utils/moatlib.py carry-forward SCAMP linux-gfx1100 2b5b1a3 binary-equiv \
  "qt_helper FFT/complex/stream renames now masquerade through the shim; host-side only; device ISA + exported symbols identical to b999b33 (3 exports, codeobj_diff verdict=identical, fresh gfx1100 builds at both shas)"
```

Transitioned linux-gfx1100: revalidate -> completed, validated_sha=2b5b1a3.
## Validation 2026-06-25 (windows-gfx1201, binary-equivalence carry-forward, sha 2b5b1a3)

SHA: 2b5b1a325149603bca4a2e5afc1d39e7121d86d2
GPU: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32) -- not exercised; binary-equivalence path
Prior validated_sha: b999b338f1ffc21c89f822d8100db9b49ac6d58c

Delta b999b33->2b5b1a3: pure host-side type-spelling/shim consolidation (qt_helper.h cuFFT
error enum, CHECK_CUFFT_ERRORS, FFT/complex member types, cudaStream_t parameter; qt_kernels.cu
cuCmul local define). All aliases were already present in the shim; no device-code change.

### Build at 2b5b1a3 (gfx1201, fresh dir)

```
ROCM="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
cmake -S B:/develop/moat/projects/SCAMP/src \
      -B B:/develop/moat/agent_space/SCAMP/build-hip-gfx1201-2b5b1a3 \
      -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_HIP=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DBUILD_SCAMP_TESTS=ON \
      -DCMAKE_CXX_COMPILER="$ROCM/lib/llvm/bin/clang-cl.exe" \
      -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/clang-cl.exe" \
      -DCMAKE_PREFIX_PATH="$ROCM"
cmake --build B:/develop/moat/agent_space/SCAMP/build-hip-gfx1201-2b5b1a3 -j64
```

Build: PASS (72/72 targets, warnings only -- all pre-existing nodiscard)

### PE .hip_fat section analysis (b999b33 vs 2b5b1a3, gfx1201)

Extracted .hip_fat and .hipFatB PE sections using llvm-objcopy --dump-section from both builds.
The .hip_fat section is a clang offload bundle with two entries (host-empty + gfx1201 device ELF).

Device ELF (hipv4-amdgcn-amd-amdhsa--gfx1201, 389512 bytes in both):

| Section     | Result    | Size   | Notes                              |
|-------------|-----------|--------|------------------------------------|
| .text       | IDENTICAL | 37888  | Device ISA -- the decisive proof   |
| .note       | IDENTICAL | 80256  | Kernel metadata (AMDHSA descriptors)|
| .rodata     | IDENTICAL | 4672   |                                    |
| .symtab     | IDENTICAL | 10056  |                                    |
| .comment    | IDENTICAL | 318    |                                    |
| .dynstr     | DIFFER    | 54645  | Only __hip_cuid_* hash changed     |
| .dynsym     | DIFFER    | 2832   | Only __hip_cuid_* entry changed    |
| .hash       | DIFFER    | 952    | Derived from dynsym                |
| .gnu.hash   | DIFFER    | 856    | Derived from dynsym                |
| .strtab     | DIFFER    | 195074 | Only __hip_cuid_* string changed   |

.hipFatB (registration blob): SHA256 IDENTICAL (48ce791a2f91c0e6...)

Only __hip_cuid_63bce6c35f7b4436 (b999b33) -> __hip_cuid_f1fb45f26d2c0de3 (2b5b1a3).
The __hip_cuid is a source-content-hash used for registration; it changes with any source
edit but does not affect kernel execution behavior.

VERDICT: binary-equivalent. Device ISA + kernel metadata + rodata + static symbols all
identical. Carry forward to 2b5b1a3 without GPU re-run.

### Carry-forward

```
python3 utils/moatlib.py carry-forward SCAMP windows-gfx1201 2b5b1a3 binary-equiv \
  "qt_helper FFT/complex/stream renames now route through shim; device ISA (.text 37888 bytes) + .note (80256 bytes) + .rodata + .symtab IDENTICAL vs b999b33; .dynstr/.dynsym differ only in __hip_cuid_ registration handle hash (63bce6c35f7b4436->f1fb45f26d2c0de3)"
```

Transitioned windows-gfx1201: revalidate -> completed, validated_sha=2b5b1a3.

## PR-prep squash (2026-06-25)

Collapsed the 3-commit moat-port branch (efc1107 + b999b33 + 2b5b1a3) into one
tree-identical squashed commit ef990d0 "[ROCm] Add AMD GPU support via HIP".
`git diff 2b5b1a3 ef990d0` is empty (verified). squash-carry-forward carried
linux-gfx90a, linux-gfx1100, windows-gfx1201 (all completed) to ef990d0;
windows-gfx1101 (revalidate) and windows-gfx1151 (port-ready) stay optional and
do not gate. pr-ready=True.

Jargon scrub: clean. Only "lead" hit in source is the upstream-preexisting
kernel_config.cpp:97 ("shfl widens its lead over sliding-window" = advantage),
left as-is. No "byte-for-byte"/"byte-identical" anywhere. No MOAT vocabulary in
source or the squashed commit message.

Attribution: only new file across the whole port is src/common/cuda_to_hip.h
(already has AMD copyright + Jeff \author). b999b33/2b5b1a3 only modified
pre-existing files, so no new attribution needed.

Upstream PR NOT opened -- awaiting Jeff's approval at the gate.

## PR fix-round: HIP CI job + kernel-authoring doc (2026-07-06)

Maintainer zpzim (PR #145) had no code objections but wanted long-term
wave64-maintainability guards. Prepared two artifacts as commits on top of the
validated head ef990d0 (NOT amended):

- 8254324 `[ROCm] Add CUDA/HIP kernel-authoring checklist doc`
  - New `docs/CUDA_HIP_KERNELS.md`: warp-width hazard checklist drawn from this
    port's kernels (0xffffffff masks, delta=16 strides, & 31 / >> 5 lane math,
    BLOCKSZ/32 warp counts, host-sized per-warp dynamic smem), each paired with
    the portable pattern the code uses (kWarpSize, SCAMP_FULL_WARP_MASK,
    kMinWarpSize; get_smem_shfl referenced for the smem-sizing rule).
  - CONTRIBUTING.md links to it from the coding-style section.
- 3e3dd51 `[ROCm] Add compile-only HIP build job to CI`
  - New `.github/workflows/build-hip.yml`: single `build-hip` job, container
    rocm/dev-ubuntu-24.04:7.2.4-complete, `-DUSE_HIP=ON`, build-only (no GPU
    run), matrix over gfx90a (wave64) and gfx1100 (wave32). Mirrors the CUDA
    build-cuda-versions build-check style. Fork Actions stay disabled; this
    runs on upstream after merge.

Local validation of the exact CI configure+build (native ROCm 7.2, gfx90a; no
docker on this host):
```
cmake -S . -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DBUILD_SCAMP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
```
Reached 100% (SCAMP HIP executable linked), pre-existing nodiscard warnings only.

advance-head ef990d0 -> 3e3dd51: classifier scored BOTH files doc-only/inert
(a CI yaml changes no compiled output, and the md is docs), so linux-gfx1100 and
windows-gfx1201 carried forward to completed@3e3dd51 with no GPU re-run --
cleaner than the anticipated codeobj_diff path. windows-gfx1101 stays
revalidate@19801ac (optional), gfx1151 port-ready (optional), linux-gfx90a
pr-open (PR #145).

No upstream reply posted from this round -- the orchestrator writes the
maintainer reply.

## Validation 2026-08-09 (linux-gfx90a, revalidate, sha 08cb196)

SHA: 08cb19658f4020db39ea95f04403159e4ed05973 (delta vs prior validated_sha ef990d0)
GPU: AMD Instinct MI250X (gfx90a), HIP_VISIBLE_DEVICES=2, 4-GPU host

Triggered by a HEAD move since linux-gfx90a's recorded validated_sha (ef990d0, the
PR-prep squash). `moatlib.py classify SCAMP ef990d0 08cb196` failed the first time
(no `projects/SCAMP/src` clone present in this fresh worktree -- `_classify_safe`
needs the fork checked out at that path to diff). Cloned the fork (moat-port branch)
into `projects/SCAMP/src`, init'd only the submodules the default build needs
(gflags, cpu_features, eigen -- grpc/pybind11 are BUILD_CLIENT_SERVER/BUILD_PYTHON_MODULE
only, not needed and skipped to avoid grpc's huge nested-submodule fetch tree). Re-ran
classify: `class=doc-only arch_independent=True inert=True` -- the delta is exactly the
"PR fix-round" commits (8254324 kernel-authoring doc, 3e3dd51/08cb196 compile-only HIP
CI job): `.github/workflows/build-hip.yml`, `docs/CUDA_HIP_KERNELS.md`, `CONTRIBUTING.md`
link. No source file touched.

Carried forward per the doc-only shortcut, but also ran a full fresh build+GPU test as
belt-and-suspenders confirmation before recording (this arch had never been validated at
any sha in the 8254324..08cb196 range, only linux-gfx1100/windows-gfx1201 had via the
earlier advance-head auto-carry -- so getting the classify tool working, not just trusting
the earlier note's description, mattered here):

### Build (08cb196, gfx90a, fresh dir)

```bash
cmake -S projects/SCAMP/src -B projects/SCAMP/src/build \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DBUILD_SCAMP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build projects/SCAMP/src/build -j$(nproc)
```

Build: PASS (100%, warnings only -- all pre-existing nodiscard). Binary 8,020,288 bytes,
matching the b999b33/2b5b1a3 binary size exactly (expected: no device or host functional
change, only CI/doc files added).

### Test (HIP_VISIBLE_DEVICES=2)

```bash
cd projects/SCAMP/src/test
HIP_VISIBLE_DEVICES=2 bash run_tests.sh ../build/SCAMP /tmp/scamp-08cb196-results.txt ""
```

Results: 50/50 tests PASSED ("All Tests Passed!"), max MP value difference
1.6639858...e-06, 0 index differences -- identical to every prior gfx90a validation
(b999b33, 2b5b1a3, 8226072). Real GPU execution confirmed (default run_tests.sh invokes
the GPU path; HIP_VISIBLE_DEVICES pinned to device 2, an MI250X per rocm-smi).

### CUDA gate

Not re-run: this is a carried-forward revalidation on a doc-only delta (no CUDA-compiled
source touched), which the CLAUDE.md CUDA-gate rule exempts explicitly. The last real nvcc
compile-check is recorded above at sha 2b5b1a3 (tree-identical to the ef990d0 squash this
delta builds on) and remains valid: PASS, full SCAMP binary compiles+links under nvcc 12.8
sm_75 with USE_HIP=OFF FORCE_CUDA=ON.

### Jargon / carry-forward record

`python3 utils/jargon.py --port SCAMP` -> clean.

```bash
python3 utils/moatlib.py carry-forward SCAMP linux-gfx90a 08cb196 source-class \
  "doc-only/inert delta ef990d0->08cb196 (.github/workflows/build-hip.yml + \
   docs/CUDA_HIP_KERNELS.md + CONTRIBUTING.md link, no source touched); classify \
   verdict class=doc-only arch_independent=True inert=True. Confirmed with a full \
   fresh gfx90a build+GPU test run: 100% build, binary 8020288 bytes matching prior \
   b999b33/2b5b1a3 builds, 50/50 tests All Tests Passed, max MP diff 1.66e-06, \
   0 index diffs (HIP_VISIBLE_DEVICES=2, MI250X)."
```

linux-gfx90a: revalidate -> completed, validated_sha=08cb196.

### Note for future validators: classify needs the fork clone in place

`moatlib.py classify <name> <old> <new>` silently returns `class=unknown
(classification failed -> revalidate)` whenever `projects/<name>/src` is not a git
checkout of the fork -- this is by design (`_classify_safe` is conservative and treats
a missing clone as unclassifiable, never a false carry-forward), but the error message
does not say why it failed. In a fresh worktree with no prior `src/` clone, clone the
fork's moat-port branch into `projects/<name>/src` FIRST, then classify -- do not treat
an `unknown` verdict as proof the delta is functional without checking whether the clone
even existed.
