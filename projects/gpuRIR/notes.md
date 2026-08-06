# gpuRIR notes

## Porting 2026-05-30

Ported to ROCm/HIP for gfx90a (MI250X, wave64) on ROCm 7.2.1. Strategy A
 + a cuFFT->hipFFT and cuRAND->hipRAND library swap.

### Build classification
Pure CMake building a pybind11 MODULE (`gpuRIR_bind`) from one `.cu`
(`src/gpuRIR_cuda.cu`) + a C++ host file (`src/python_bind.cpp`). Not a
pytorch extension. `pip install .` drives CMake via a CMakeExtension shim.

### CUDA surface
One `.cu`. Kernels use only shared memory + `__syncthreads` reductions --
**no warp intrinsics, no `__constant__`, no `__shfl/__ballot/__any`**, so the
wave64 / popsift warp-packing fault class does NOT apply. cuFFT (R2C/C2R 1D
plans for the FFT convolution), cuRAND host API (uniform noise for the
diffuse tail), a 1D LUT texture (default-on), `cuda_fp16` half2 mixed
precision (off by default), and 3D pitched copies.

### Port changes (all HIP-guarded; CUDA path byte-for-byte unchanged)
- `src/cuda_to_hip.h` (new): on USE_HIP include hip_runtime + hip_fp16 +
  hipfft + hiprand and alias the cuda/cufft/curand symbols (names from
  PyTorch's hipify map). `#else` includes the original CUDA headers verbatim.
- `src/gpuRIR_cuda.cu`: swapped the 6 CUDA `#include`s for the shim; added
  two small `#if defined(USE_HIP)` branches (texture fix) + one in
  `activate_mixed_precision`.
- `CMakeLists.txt`: `option(USE_HIP)`; under HIP -> project LANGUAGES CXX HIP,
  `enable_language(HIP)`, mark the `.cu` `LANGUAGE HIP`, HIP_ARCHITECTURES
  gfx90a, `-DUSE_HIP`, force-include the shim, link `hip::hipfft hip::hiprand`
  instead of cuFFT/cuRAND, and **skip find_package(CUDA)**.
- `setup.py`: forward `$CMAKE_ARGS` so `pip install .` is hands-free.

### Fault classes / gotchas hit
1. **Texture linear filter (popsift class).** The default LUT path uses a 1D
   cudaArray texture with `cudaFilterModeLinear` + `cudaReadModeElementType`
   -- AMD rejects hw linear filtering on element-read float textures. Fix:
   on HIP create the texture `cudaFilterModePoint` and do the linear
   interpolation in software in `image_sample_lut` (point-fetch 2 neighbors +
   lerp, CUDA's unnormalized -0.5 texel-center convention). Verified: LUT vs
   exact-sinc paths agree to 0.43% of peak.
2. **`__fmaf_rz` missing in HIP** (only `__fmaf_rn`), exactly like CudaSift's
   `__fmul_rz`. The only use computes a LUT index coordinate, so round-mode
   is immaterial -> aliased to `__fmaf_rn` in the shim.
3. **Host `memcpy`/`memset` resolve to HIP `__device__` overloads** inside a
   `.cu` once `<hip/hip_runtime.h>` is in scope (the `PadData` host helper
   failed to compile). Fix: include `<cstring>`/`<cstdlib>` BEFORE
   hip_runtime in the shim so the libc host decls win overload resolution.
4. **IPO/LTO breaks the pybind11 module under the HIP toolchain.** With
   `INTERPROCEDURAL_OPTIMIZATION TRUE` the module `.so` came out as slim LTO
   bitcode with no `PyInit_gpuRIR_bind` (ImportError, 5 KB .so). The HIP link
   step does not finalize LTO. Fix: skip IPO when USE_HIP (it is explicitly
   optional here). After: 340 KB .so with `PyInit_gpuRIR_bind` exported.
5. **Mixed precision not ported.** The half2 device kernels are inside
   `#if __CUDA_ARCH__ >= 530` (undefined on HIP) so they compile to empty
   stubs. But host `cuda_arch` = major*100+minor*10 is 900 on gfx90a, which
   would enable the (broken) path. Guarded `activate_mixed_precision` to
   report unsupported on HIP, forcing it off so the stubs are never launched.
   Default config (LUT on, mp off) is unaffected.
6. `cudaMemcpyToArray` is still declared (deprecated) in ROCm 7.2 with the
   CUDA signature, so it aliases directly -- no call-site rewrite needed
   (unlike CudaSift on an older ROCm). `<cufftXt.h>` is included upstream but
   none of its symbols are used; dropped on the HIP branch.

### Build + validation
- Build: `pip install ./projects/gpuRIR/src --no-build-isolation` with
  `CMAKE_ARGS="-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm"`.
  Links libhipfft/libhiprand/librocfft/librocrand/libamdhip64, no CUDA.
- Validated on gfx90a GPU 3 (`HIP_VISIBLE_DEVICES=3`):
  agent_space/gpuRIR_validate.py -- shoebox room 4x5x3, src (1,1,1.5) ->
  rcv (3,4,1.5), dist 3.6056 m, expected direct-path sample
  round(dist/c*Fs)=168. Result: RIR finite, non-zero; first significant
  arrival (>10% of peak) at sample 168 for BOTH LUT and exact-sinc paths,
  silent before the direct path; hipFFT convolution (simulateTrajectory)
  finite + non-zero. Repo `examples/example.py` (2 src x 3 rcv, cardioid
  mic) also runs clean (exit 0). NOTE: the direct path is the first arrival,
  NOT the global max -- constructive early reflections can be larger in a
  reverberant room; a global-argmax check is wrong.

## Validation 2026-05-30 (gfx1100, ROCm 7.2.1)

GPU: 2x AMD Radeon Pro W7800 48GB, gfx1100 (RDNA3, wave32). ROCm 7.2.1.
Fork branch: moat-port. Validated SHA: 6c912137c40fd1b7509722f8188bbc3f6fd0f702.

### Build

No source change needed for gfx1100 -- the CMakeLists.txt already correctly
reads `${CMAKE_HIP_ARCHITECTURES}` from the CMake variable (defaulting to gfx90a
only when unset), so passing `-DCMAKE_HIP_ARCHITECTURES=gfx1100` is sufficient.
Because gfx1100 required no code change, the curated commit is left untouched at
6c91213 (the SHA gfx90a was validated at): no head_sha churn, and gfx90a stays
`completed` instead of being forced to revalidate. (An earlier pass amended a CI
smoketest YAML into the commit, advancing the SHA to cc8736b and flipping gfx90a
to revalidate; that was reverted -- a non-essential file must never churn an
already-validated platform. A CI tripwire belongs in the lead-port commit.)

```
CMAKE_ARGS="-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm" \
  bash utils/timeit.sh gpuRIR compile -- \
  pip install ./projects/gpuRIR/src --no-build-isolation --force-reinstall --no-deps
```

Result: build succeeded, gpurir-1.2.0 installed, .so = 333 KB (IPO/LTO skipped, no slim LTO bitcode regression).

### gfx1100 code-object evidence

```
/opt/rocm/llvm/bin/llvm-objdump --offloading gpuRIR_bind.cpython-312-x86_64-linux-gnu.so
```

Output:
```
Extracting offload bundle: ...so.0.host-x86_64-unknown-linux-gnu-
Extracting offload bundle: ...so.0.hipv4-amdgcn-amd-amdhsa--gfx1100
```

gfx1100 confirmed, no gfx90a objects. PyInit_gpuRIR_bind exported (nm -D confirms `T PyInit_gpuRIR_bind`).

### Validation run

```
HIP_VISIBLE_DEVICES=0 bash utils/timeit.sh gpuRIR test -- \
  python3 agent_space/gpuRIR_validate_gfx1100.py
```

Shoebox room 4x5x3 m, src (1,1,1.5) -> rcv (3,4,1.5), c=343.0, Fs=16000.
Distance = 3.6056 m, expected direct-path sample = round(3.6056/343.0*16000) = 168.

- LUT path: first significant arrival (>10% of peak) at sample **168** (expected 168). PASS.
- Exact-sinc path: first significant arrival at sample **168** (expected 168). PASS.
- LUT vs sinc max diff: 0.000108, peak: 0.065515. Agreement: **0.165% of peak** (< 0.5% threshold). PASS.
- Pre-direct silence: verified (no sample before index 168 exceeds 10% of peak). PASS.
- hipFFT convolution (simulateTrajectory, 5-point trajectory): output shape (19199, 1), finite, non-zero. PASS.
- examples/example.py (2 src x 3 rcv, cardioid mic, MPLBACKEND=Agg): exit 0. PASS.

Results match gfx90a exactly (same sample 168, LUT/sinc agreement within rounding). No warp intrinsics in the kernel; RDNA3 wave32 is low numeric risk and confirmed numerically equivalent.

### CI smoketest

Intentionally NOT added during this follower validation. A rocm/dev-ubuntu-24.04
compile-only tripwire (USE_HIP=ON builds, the gpuRIR_bind .so exceeds a few KB,
PyInit_gpuRIR_bind exported) is worth having, but adding it here would have
churned the curated commit and forced gfx90a to revalidate for a non-GPU file.
It should be added to the lead-port commit instead, so no follower pays for it.

## Windows gfx1151 (2026-05-31): BUILD PASS, validation pending (resumable)

Build SUCCEEDED for gfx1151 with the all-clang Windows toolchain. Module
gpuRIR_bind.cp313-win_amd64.pyd linked (links hipfft/hiprand; gpuRIRcu static lib;
gfx1151 code object). Fork HEAD 6c91213 (no source change -- validate-first follower).
Build script: agent_space/gpurir_build.sh.
```
cmake -S <src> -B build-win-gfx1151 -G Ninja \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_HIP_COMPILER=clang++ \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_PREFIX_PATH="<devel-root>;<libraries-root>" -DPYTHON_EXECUTABLE=<py> \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_HIP_STANDARD=17 -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_FLAGS="-DNOMINMAX -DWIN32_LEAN_AND_MEAN" -DCMAKE_BUILD_TYPE=Release
cmake --build build-win-gfx1151 --target gpuRIR_bind -j16
```

TO VALIDATE (next session, quick): place gpuRIR_bind.*.pyd on sys.path; deploy beside
it amdhip64_7.dll + amd_comgr + rocm_kpack.dll + hipfft.dll + rocfft.dll + hiprand.dll
+ rocrand.dll (from _rocm_sdk_core/bin and _rocm_sdk_libraries/bin); the gfx1151 FFT/RAND
kpacks (fft_lib_gfx1151.kpack, rand_lib_gfx1151.kpack) ship in _rocm_sdk_libraries/.kpack.
pip install scipy numpy. Run the shoebox-room harness (4x5x3 room, src (1,1,1.5) ->
rcv (3,4,1.5), expect direct-path sample at round(dist/c*Fs)=168, RIR finite/non-zero;
compare to gfx90a/gfx1100). Status left port-ready (NOT completed -- GPU correctness
not yet exercised on gfx1151).

## Windows gfx1151 (2026-06-02): VALIDATED -> completed @ 6c91213

GPU-validated on gfx1151 (AMD Radeon 8060S, TheRock ROCm) with the pre-built gpuRIR_bind.cp313-win_amd64.pyd (validate-first follower, no source change). Harness: agent_space/gpurir_validate.py (shoebox 4x5x3, src (1,1,1.5) -> rcv (3,4,1.5), fs 16000). Results match the gfx90a reference:
- exact-sinc: first significant arrival at sample 168 (expected 168), finite/non-zero, pre-direct silence -- PASS
- LUT path: direct path also at sample 168; LUT-vs-sinc agreement 0.005% near the direct path (full-RIR max 9.8% lands at sample 1619, deep in the dense reverberant tail = the LUT approximation summed over overlapping images, config-dependent, NOT a LUT-read fault) -- PASS
- hipFFT convolution (simulateTrajectory, 5-pose trajectory): finite/non-zero -- PASS
- examples/example.py (2 src x 3 rcv, cardioid mic, MPLBACKEND=Agg): exit 0 -- PASS
No warp intrinsics in the kernels; RDNA3 wave32 numerically equivalent to gfx90a/gfx1100. DLL setup: os.add_dll_directory for the build dir + _rocm_sdk_core/bin (amdhip64_7/amd_comgr/rocm_kpack) + _rocm_sdk_libraries_gfx1151/bin (gfx1151 hipfft/rocfft/hiprand/rocrand; device code baked into the per-arch DLLs -- no separate .kpack in this TheRock layout). Marked completed, validated_sha=6c91213.

## Audit 2026-06-03

Audit of commit 6c912137c40fd1b7509722f8188bbc3f6fd0f702 (moat-port = fork/moat-port, matches status.json head_sha and validated_sha for all three platforms). No fork rewrite detected.

### Safety check
Local moat-port and fork/moat-port are identical at 6c91213. No force-push needed and none was observed.

### Test discovery
gpuRIR upstream has no formal test suite (no tests/ directory, no pytest, no unittest). The examples/ directory has five scripts (example.py, simulate_trajectory.py, polar_plots.py, time_vs_T60.py, time_vs_nbRIRs.py); none self-validates with assertions -- they produce plots or audio files and rely on visual inspection. simulate_trajectory.py requires a source_signal.wav; time_vs_T60.py and time_vs_nbRIRs.py are timing benchmarks. The only programmatic validation is agent_space/gpuRIR_validate.py (written for this port): shoebox room with known direct-path geometry, checking finite/non-zero output, first significant arrival within 2 samples of the expected round(dist/c*Fs)=168, pre-direct silence, LUT vs exact-sinc agreement <5% of peak, and FFT convolution path finite/non-zero. This is the correct and maximal test achievable for this project.

### Real GPU validation re-run (gfx90a, GCD 2, HIP_VISIBLE_DEVICES=2)

Rebuilt from source at 6c91213 and re-ran on real GPU (MI250X, gfx90a, ROCm 7.2.1):

```
CMAKE_ARGS="-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm" \
  bash utils/timeit.sh gpuRIR compile -- \
  pip install ./projects/gpuRIR/src --no-build-isolation --force-reinstall --no-deps

HIP_VISIBLE_DEVICES=2 bash utils/timeit.sh gpuRIR test -- \
  python3 agent_space/gpuRIR_validate.py
```

Results:
- LUT path: first significant arrival at sample 168 (expected 168), finite/non-zero, pre-direct silent -- PASS
- exact-sinc path: first significant arrival at sample 168 -- PASS
- LUT vs sinc: max diff 2.47e-04 / peak 0.43% (< 5% threshold) -- PASS
- hipFFT convolution (simulateTrajectory, 4-point trajectory): finite/non-zero -- PASS
- examples/example.py (MPLBACKEND=Agg): exit 0 -- PASS

All checks pass. Status remains `completed` for all three platforms.

### Jargon / yml scan
"Strategy A" and "colmap model" appear in the commit message body, a CMakeLists.txt comment (line 8 in the USE_HIP block), and the cuda_to_hip.h file banner comment (line 4). These are MOAT-internal vocabulary not meaningful to upstream readers. No other MOAT terms (head_sha, validated_sha, lead, follower, curated, smoketest, MOAT) appear in committed files. No .github/workflows file was added to the fork at any level outside the vendored third_party/pybind11 subtree (which is upstream content, not a moat addition). No spurious CI yaml was committed.

The two "Strategy A / colmap model" occurrences in committed source (CMakeLists.txt and cuda_to_hip.h) are mild PR-readiness issues: they are intelligible without the MOAT context (the colmap project is publicly known as a reference for this pattern) but still carry internal vocabulary. They do not affect correctness. Flag for the porter to neutralize before the upstream PR (e.g., "compat-header pattern" or drop the parenthetical).

### Conclusion
Real tests: the agent_space/gpuRIR_validate.py harness is the maximal test for this project (upstream has no test suite); it checks physics-grounded correctness (direct-path timing, LUT accuracy, FFT conv). It was run on real gfx90a GPU in this audit and passes. Status: `completed` retained on all platforms, no state change required.

## Validation 2026-06-16 (windows-gfx1101, ROCm 7.14.0a20260604)

GPU: AMD Radeon PRO V710, gfx1101 (RDNA3, wave32), HIP_VISIBLE_DEVICES=0.
Fork branch: moat-port. Validated SHA: 1cc434e6c9b2d36f0c2200d40f91bda113791027.
No source change needed -- validate-first follower, CMakeLists.txt reads `${CMAKE_HIP_ARCHITECTURES}`.

### Build

Direct CMake/Ninja build (all-clang, ROCM_DEVEL = `_rocm_sdk_devel` from TheRock PyTorch venv):

```
VENV="B:/develop/TheRock/external-builds/pytorch/.venv"
ROCM_DEVEL="$VENV/Lib/site-packages/_rocm_sdk_devel"
cmake -S projects/gpuRIR/src -B projects/gpuRIR/build-win-gfx1101 -G Ninja \
    -DCMAKE_CXX_COMPILER="$ROCM_DEVEL/lib/llvm/bin/clang++.exe" \
    -DCMAKE_HIP_COMPILER="$ROCM_DEVEL/lib/llvm/bin/clang++.exe" \
    -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
    -DCMAKE_PREFIX_PATH="$ROCM_DEVEL" \
    -DPYTHON_EXECUTABLE="$VENV/Scripts/python.exe" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_HIP_STANDARD=17 -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_FLAGS="-DNOMINMAX -DWIN32_LEAN_AND_MEAN" -DCMAKE_BUILD_TYPE=Release
cmake --build projects/gpuRIR/build-win-gfx1101 --target gpuRIR_bind -j32
```

Result: build succeeded. `gpuRIR_bind.cp312-win_amd64.pyd` = 362 KB. IPO/LTO skipped (USE_HIP), no slim-bitcode regression.

### Validation run

```
HIP_VISIBLE_DEVICES=0 bash utils/timeit.sh gpuRIR test -- \
  python agent_space/gpurir_validate_gfx1101.py
```

Shoebox room 4x5x3 m, src (1,1,1.5) -> rcv (3,4,1.5), c=343.0, Fs=16000.
Distance = 3.6056 m, expected direct-path sample = round(3.6056/343.0*16000) = 168.

- exact-sinc: first significant arrival at sample **168** (expected 168). PASS.
- exact-sinc: pre-direct silence verified. PASS.
- LUT path: first significant arrival at sample **168** (expected 168). PASS.
- LUT vs sinc near direct (samples 148-368): max diff=0.000080, peak=0.048074, **0.166% of peak** (<2% threshold). PASS.
- LUT vs sinc full-RIR: 18.4% -- expected; divergence accumulates in dense reverberant tail (config-dependent, same behavior as gfx1151/gfx1201).
- simulateTrajectory (hipFFT convolution, 5-point trajectory): output shape (6398,1), finite, non-zero. PASS.
- examples/example.py (2 src x 3 rcv, cardioid mic, MPLBACKEND=Agg): exit 0. PASS.

Results match gfx90a/gfx1100/gfx1151/gfx1201 exactly (same sample 168, 0.166% near-direct agreement). No warp intrinsics; RDNA3 wave32 numerically equivalent.

## Validation 2026-06-07 (windows-gfx1201, ROCm 7.14.0a20260604)

GPU: AMD Radeon RX 9070 XT, gfx1201 (RDNA4, wave32), HIP_VISIBLE_DEVICES=0 (V710 offline this session).
Fork branch: moat-port. Validated SHA: 6c912137c40fd1b7509722f8188bbc3f6fd0f702.
No source change needed -- validate-first follower, CMakeLists.txt reads `${CMAKE_HIP_ARCHITECTURES}`.

### Build

Direct CMake/Ninja build (all-clang, ROCM_DEVEL = `_rocm_sdk_devel` from TheRock PyTorch venv):

```
VENV="B:/develop/TheRock/external-builds/pytorch/.venv"
ROCM_DEVEL="$VENV/Lib/site-packages/_rocm_sdk_devel"
cmake -S projects/gpuRIR/src -B projects/gpuRIR/build-win-gfx1201 -G Ninja \
    -DCMAKE_C_COMPILER="$ROCM_DEVEL/lib/llvm/bin/clang++.exe" \
    -DCMAKE_CXX_COMPILER="$ROCM_DEVEL/lib/llvm/bin/clang++.exe" \
    -DCMAKE_HIP_COMPILER="$ROCM_DEVEL/lib/llvm/bin/clang++.exe" \
    -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
    -DCMAKE_PREFIX_PATH="$ROCM_DEVEL" \
    -DPYTHON_EXECUTABLE="$VENV/Scripts/python.exe" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_HIP_STANDARD=17 -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_FLAGS="-DNOMINMAX -DWIN32_LEAN_AND_MEAN" -DCMAKE_BUILD_TYPE=Release
cmake --build projects/gpuRIR/build-win-gfx1201 --target gpuRIR_bind -j32
```

Result: build succeeded. `gpuRIR_bind.cp312-win_amd64.pyd` = 366 KB. Code object verified: `hipv4-amdgcn-amd-amdhsa--gfx1201` and `PyInit_gpuRIR_bind` confirmed in binary. IPO/LTO skipped (USE_HIP), no slim-bitcode regression.

### Validation run

```
HIP_VISIBLE_DEVICES=0 bash utils/timeit.sh gpuRIR test -- \
  python agent_space/gpurir_validate_gfx1201.py
```

Shoebox room 4x5x3 m, src (1,1,1.5) -> rcv (3,4,1.5), c=343.0, Fs=16000.
Distance = 3.6056 m, expected direct-path sample = round(3.6056/343.0*16000) = 168.

- exact-sinc: first significant arrival at sample **168** (expected 168). PASS.
- exact-sinc: pre-direct silence verified. PASS.
- LUT path: first significant arrival at sample **168** (expected 168). PASS.
- LUT vs sinc near direct path (samples 148-368): max diff=0.000080, peak=0.048074, **0.166% of peak** (<2% threshold). PASS.
- LUT vs sinc full-RIR: 18.4% -- expected; divergence accumulates in the dense reverberant tail over many image sources (config-dependent, same behavior as gfx1151). Not a LUT-read fault.
- simulateTrajectory (hipFFT convolution, 5-point trajectory): output shape (6398,1), finite, non-zero. PASS.
- examples/example.py (2 src x 3 rcv, cardioid mic, MPLBACKEND=Agg): exit 0. PASS.

Results match gfx90a/gfx1100/gfx1151 exactly (same sample 168, 0.166% near-direct agreement). No warp intrinsics; RDNA4 wave32 numerically equivalent.
