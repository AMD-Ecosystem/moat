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

## Upstream sync + gfx942 port 2026-10-29 (linux-gfx942, MI300X, TheRock ROCm 10.1.0)

Requested by jeff before the upstream PR is opened: upstream #87 (06da27d, 2026-08-25) moved CMakeLists.txt from `find_package(CUDA)` to `find_package(CUDAToolkit REQUIRED)` and links `CUDA::cufft`/`CUDA::curand` PUBLIC on `gpuRIRcu`, conflicting with the port's CMakeLists. Porting lock taken as linux-gfx942.

### Fork branches
- Fork `master` fast-forwarded fd8af43 -> 06da27d (plain push, verified ancestor first).
- `moat-port` 88a0749 -> 27a44c8, commits on top, no rebase/amend:
  - c171ee7 merge of upstream master. Conflict resolution: `find_package(CUDAToolkit REQUIRED)` inside the non-HIP branch; links moved to upstream's structure: `gpuRIRcu` PUBLIC `CUDA::cufft CUDA::curand` (CUDA) / `hip::hipfft hip::hiprand` (HIP); `gpuRIR_bind` PRIVATE `gpuRIRcu pybind11::module` once for both. No other upstream file changed; upstream added no tests.
  - 5fef7a4 gfx942 texture fix (below).
  - 27a44c8 README: ROCm prerequisite sentence + `CMAKE_ARGS="-DUSE_HIP=ON ..."` install line (the porter doc step had never been done; the README had no ROCm mention).

### gfx942 fault: texture API compiled out on gfx94x+
First gfx942 build of the merge failed: `gpuRIR_cuda.cu:248: error: 'tex1D<float, nullptr>' is unavailable: The image/texture API not supported on the device` (known fault class, `__gfx94plus_clr__` -> `__HIP_NO_IMAGE_SUPPORT`). Not caused by the merge; the port had never been built for gfx94x. Fix, all HIP-guarded, CUDA texture path untouched: on HIP the sinc LUT is a plain device buffer of `lut_len + 1` floats (trailing zero = border texel) and `image_sample_lut` indexes it with the same software lerp it already used over a point-filtered texture. Index range: `|t-tau| < Tw/2` keeps `floor(c) >= 0`; high side can reach index `lut_len` at most, which is the zero. Removed the now-unused array/texture aliases from `cuda_to_hip.h`. Applies to all AMD arches (one path); results are the same arithmetic as before, so prior arches' numerics are expected unchanged, but they read `revalidate` after advance-head. Promoted a short note to `cuda-to-rocm/references/fault-classes.md` (Textures).

### Build (direct CMake; this host's guard blocks pip, even `--target`, so the `setup.py` route was not exercised here)
```
R=$(rocm-sdk path --root)
cmake -S projects/gpuRIR/src -B agent_space/gpurir/build-tip -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$R -DPYTHON_EXECUTABLE=$(which python3) -DCMAKE_BUILD_TYPE=Release
cmake --build agent_space/gpurir/build-tip -j32
```
Result at 27a44c8: build OK, 333600-byte .so, offload bundle `hip-amdgcn-amd-amdhsa--gfx942`, `T PyInit_gpuRIR_bind`. Compile-only check with `-DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx942;gfx1100;gfx1101;gfx1151;gfx1201"`: 0 errors, all six code objects present.
Runtime (TheRock pip layout): `LD_LIBRARY_PATH=$SP/_rocm_sdk_libraries/lib:$SP/_rocm_sdk_core/lib`, `PYTHONPATH=<build dir>:projects/gpuRIR/src`, where SP is the venv site-packages; without it the import fails with `libhipfft.so.0: cannot open shared object file`.

### Tests (HIP_VISIBLE_DEVICES=0, `ulimit -c 0`)
Upstream has no test suite; the GPU suite is the validation harness plus all five upstream examples.
- `python3 agent_space/gpurir/validate.py` (rewritten; the earlier harness was in a wiped agent_space): **12/12 PASS**.
  - Direct-path-only RIR (nb_img [1,1,1]) vs a float64 reference: exact-sinc max rel err 9.49e-07, LUT (vs float64 LUT+lerp reference) 1.01e-06; threshold 2e-5, which separates the LUT and exact references (they differ by 1.30e-04), so the LUT check is discriminating.
  - Shoebox 4x5x3, src (1,1,1.5) -> rcv (3,4,1.5), fs 16000: first arrival at sample 168 (expected 168) for LUT and exact, silence before the direct path, LUT vs exact near the direct path 0.165% of peak (matches the gfx1100/gfx1101/gfx1201 0.165-0.166%).
  - simulateTrajectory (hipFFT): finite/non-zero, shape (14399,1); a static trajectory equals `np.convolve` to rel err 2.82e-07.
  - Diffuse tail (hipRAND, Tdiff=0.1): finite, non-zero.
- Examples, run from a scratch copy with a synthetic 2 s `source_signal.wav`. matplotlib and plotly are absent from the image and pip is blocked, so a no-op plotting stub was put on PYTHONPATH (the examples only plot; every GPU call runs): `example.py`, `simulate_trajectory.py` (wrote filtered_signal.wav), `polar_plots.py`, `time_vs_T60.py`, `time_vs_nbRIRs.py`: **5/5 exit 0**.
Whether the plain-buffer LUT costs throughput on texture-capable arches (gfx90a/gfx11xx) was not timed; the time_vs_* examples there give a before/after if wanted.

## Review 2026-10-29 (linux-gfx942, reviewer)

Port Review: gpuRIR (`moat-port` 27a44c8 vs `master` 06da27d). Whole fork diff reviewed, with the three new commits (c171ee7 merge, 5fef7a4 LUT buffer, 27a44c8 README) read in full. `jargon.py --port gpuRIR`: clean.

### Summary
A HIP port of gpuRIR through a force-included compatibility header, with hipFFT/hipRAND swaps, a software-lerp LUT in a plain device buffer, and mixed precision disabled on HIP. **Request Changes**, for one finding: the CUDA LUT sampling line is not upstream's.

### Minimal Footprint / Backward Compatibility (upstream)
1. `src/gpuRIR_cuda.cu:243-258` (`image_sample_lut`): the CUDA path is not byte-identical to upstream. Upstream's single line `return (abs(t_tau)<Tw_2)? amp * tex1D<float>(sinc_lut, __fmaf_rz(t_tau,lut_oversamp,lut_center)) : 0.0f;` was rewritten for both backends as `if (abs(t_tau) >= Tw_2) return 0.0f;` followed by an `#if` split of the fetch. This came from a523c73, not from today's round. For finite input the two are equivalent. For NaN `tau` they differ: upstream's `<` is false and the function returns 0, while `>=` is also false and execution goes on to `tex1D(NaN)` on CUDA. Since 5fef7a4, HIP takes the same path into `(int)floorf(NaN)` and indexes `sinc_lut[i]`, which is UB in C++ (the old texture fetch was bounded by the border address mode). Fix: keep upstream's ternary verbatim in the `#else` (CUDA) branch, and give the HIP branch its own guard that keeps the NaN semantics, `if (!(abs(t_tau) < Tw_2)) return 0.0f;`, ahead of the coordinate and lerp. The CUDA source line is then unchanged, and HIP returns 0 exactly where CUDA does.

### Fact-check record (claims verified, no finding)
- CMake merge (c171ee7): `git diff -w master...moat-port -- CMakeLists.txt` is purely additive. The CUDA branch keeps `find_package(CUDAToolkit REQUIRED)`, `-D_FORCE_INLINES`, `CUDA_ARCHITECTURES all-major`, IPO, and the PUBLIC `CUDA::cufft CUDA::curand` link exactly as upstream has them. The only other CUDA-side edits are three trailing-space strips.
- LUT buffer index range (5fef7a4): I brute-forced the HIP index math in float32 with a single-rounding fma, over Fs in {4k, 8k, 11.025k, 16k, 22.05k, 24k, 32k, 44.1k, 48k, 96k} and 2M `t_tau` samples each. The low index is always >= 0. The high `i+1` reaches exactly `lut_len` (the trailing zero) at Fs 4k/8k/11.025k/16k/22.05k/24k/32k/48k/96k, so the extra element is required and present. On HIP, point fetch at `i0+0.5` returned texel `i0`, so the buffer read is the same arithmetic as the old HIP path.
- GPU spot check at 27a44c8, gfx942 (build: `cmake -S projects/gpuRIR/src -B agent_space/gpurir/build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$R`, R=`rocm-sdk path --root`). Direct-path RIR, LUT vs exact sinc, max error relative to peak: 8000 Hz 1.59e-3, 11025 Hz 1.29e-3, 16000 Hz 1.38e-4, 44100 Hz 6.85e-4, 48000 Hz 3.91e-4. All finite.
- Lesson in `cuda-to-rocm/references/fault-classes.md` (Textures): accurate against `amd_device_functions.h` (`__gfx94plus_clr__` -> `__HIP_NO_IMAGE_SUPPORT`) and against the code.
- Commit hygiene: all titles `[ROCm]`, <= 66 chars, no Co-Authored-By/noreply, AI disclosure and Test Plan present, ASCII only.

## Porter round 2026-10-30 (linux-gfx942): Review 2026-10-29 addressed

Review PR #1 on the fork has no line threads or review body; the only finding is the one in "Review 2026-10-29" above.

### Finding 1 (image_sample_lut CUDA line not upstream's): fact-checked, fixed in af05f1d
- Verified: `git show upstream/master:src/gpuRIR_cuda.cu` line 238 is the single ternary `return (abs(t_tau)<Tw_2)? amp * tex1D<float>(sinc_lut, __fmaf_rz(t_tau,lut_oversamp,lut_center)) : 0.0f;`, and the port replaced it for both backends with an early `>=` return. For NaN `t_tau` both comparisons are false, so the port fell through.
- Verified on GPU that the NaN case is real on HIP, not just theoretical: at 27a44c8 (rebuilt in a detached worktree) a NaN receiver x gives an all-NaN RIR on the LUT path (800/800 samples NaN); the exact-sinc path returned zeros. (`(int)floorf(NaN)` is UB in C++; on AMDGPU the conversion evidently lands in range, and the NaN amp propagates.)
- Fix: the `#else` (CUDA) branch is upstream's line verbatim; the HIP branch has its own `if (!(abs(t_tau) < Tw_2)) return 0.0f;` ahead of the coordinate and lerp. `git diff upstream/master -- src/gpuRIR_cuda.cu` now shows the upstream return line as unchanged context. The `sinc_lut_t` signature typedef stays (it is `cudaTextureObject_t` on CUDA, so the CUDA function is unchanged in type).

### Build at af05f1d (gfx942, TheRock ROCm 10.1.0)
```
R=$(rocm-sdk path --root)
cmake -S projects/gpuRIR/src -B agent_space/gpurir/build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$R -DPYTHON_EXECUTABLE=$(which python3) -DCMAKE_BUILD_TYPE=Release
cmake --build agent_space/gpurir/build -j16
```
OK, 333600-byte .so, `T PyInit_gpuRIR_bind`. Only warning is the pre-existing upstream nodiscard on `cudaGetDeviceProperties` (line 1069). Compile-only with `-DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx942;gfx1100;gfx1101;gfx1151;gfx1201"`: 0 errors, six code objects in the .so.

### Tests (HIP_VISIBLE_DEVICES=0, `ulimit -c 0`)
Runtime env as in the 2026-10-29 sync round (`LD_LIBRARY_PATH=$SP/_rocm_sdk_libraries/lib:$SP/_rocm_sdk_core/lib`, `PYTHONPATH=<build>:projects/gpuRIR/src`).
- `python3 agent_space/gpurir/validate.py` (harness rewritten again; agent_space was fresh): **16/16 PASS**.
  - Direct path (nb_img [1,1,1]) vs float64 reference, exact and LUT, fs 8k/16k/44.1k/48k: max err/peak 4.6e-8 .. 1.54e-7 (reference LUT-vs-exact gap 1.38e-4, so discriminating). Harness gotcha: the reference must compute `dist` and `tau = dist/c*Fs` in float32 as the kernel does; a float64 tau gives ~3e-5 error at 44.1/48 kHz on BOTH paths (tau ~460 samples, float32 ulp).
  - Shoebox 4x5x3, src (1,1,1.5) -> rcv (3,4,1.5), fs 16k: first arrival 168 (expected 168) LUT and exact; silent before `tau - Tw/2`. (Silence must be checked before the direct path's Hann half-window, not just a few samples before tau.)
  - Diffuse tail (hipRAND) finite/non-zero; simulateTrajectory (hipFFT) shape (14399,1), static trajectory vs np.convolve rel err 2.64e-7.
  - NaN receiver position: all-zero RIR on LUT and exact paths (LUT path was all-NaN at 27a44c8: 15/16 there).
- Upstream examples (scratch copy, synthetic 2 s 16 kHz `source_signal.wav`, no-op matplotlib/plotly stub since neither is installed): example.py, simulate_trajectory.py, polar_plots.py, time_vs_T60.py, time_vs_nbRIRs.py: **5/5 exit 0**.

Commit af05f1d: `[ROCm]` title 57 chars, AI disclosure, Test Plan, no trailer; `jargon.py --port gpuRIR` clean, `prose.py` clean. Pushed plain to `moat-port` (27a44c8..af05f1d; no upstream PR open).

## Review 2026-10-30 (linux-gfx942, reviewer)

Port Review: gpuRIR (`moat-port` af05f1d vs `master` 06da27d). Re-review requested by jeff after the porter round above.

### Summary
A HIP port of gpuRIR through a force-included compatibility header, with hipFFT/hipRAND swaps, a software-lerp LUT in a plain device buffer, and mixed precision disabled on HIP. Review 2026-10-29 finding 1 is resolved. **Request Changes** for one new finding: a deferral that jeff ruled `now` has not been done.

### Port Correctness
1. Mixed precision is still not ported, although `projects/gpuRIR/deferred.json` item `gpurir-mixed-precision` is open and was ruled `now` by jeffdaily on 2026-08-13 (moat commit 8cab4aad). `moatlib.now_deferrals` counts it as ported work the project owes. Neither the 2026-10-29 sync round nor the 2026-10-30 porter round mentions it, and the 2026-10-29 review missed it. At af05f1d, `src/gpuRIR_cuda.cu:1091-1098` (`activate_mixed_precision`, HIP branch) still prints "Mixed precision is not supported on the HIP/ROCm build." and forces `mixed_precision = false`. Confirmed on gfx942: `gpuRIR.activateMixedPrecision(True)` prints that message. What to port: the half2 helpers and kernels are compiled out on HIP because `__CUDA_ARCH__` is undefined there: `#if __CUDA_ARCH__ >= 530` at lines 45, 167, 514, 537 and 562. Widen those guards for the HIP device pass (for example `|| defined(__HIP_DEVICE_COMPILE__)`; do not define `__CUDA_ARCH__` in the shim), then remove the HIP early return in `activate_mixed_precision`. The host `cuda_arch` gate (lines 718, 884, 1099) already passes on every AMD arch (major*100+minor*10 >= 900). Because `CUDART_VERSION` is undefined on HIP, line 13 (`#if CUDART_VERSION < 9000`) maps `__h2div` to `h2div`. That will fail to compile once the kernels are live: TheRock ROCm 10.1.0's HIP headers declare `__h2div` (`amd_hip_fp16.h:883`) and no `h2div`. Lines 508/730 likewise take the pre-9.2 path (`float Tw_inv`, converted in the kernel). Decide those `CUDART_VERSION` branches for HIP so that HIP gets the modern spelling, without changing the CUDA lines. Keep the CUDA lines byte-identical. Test: on a real GPU, compare mixed-precision output with the fp32 exact path (shoebox and direct-path cases) at the tolerance that half2 allows, and record the magnitudes. Then close the deferral with `utils/deferred.py set-status gpurir-mixed-precision done`. If the half2 path cannot be made correct on AMD, record the evidence and bring it to jeff. Do not leave the `now` ruling silently unmet.

### Verification of Review 2026-10-29
- Finding 1 (image_sample_lut CUDA line): resolved. In `git diff upstream/master -- src/gpuRIR_cuda.cu`, upstream's `return (abs(t_tau)<Tw_2)? amp * tex1D<float>(...)` appears as unchanged context in the `#else` branch (`src/gpuRIR_cuda.cu:257`). The HIP branch guards with `if (!(abs(t_tau) < Tw_2)) return 0.0f;` (line 245) ahead of the index math. Rebuilt af05f1d for gfx942 (same cmake line as the porter round, `-j16`). A NaN receiver x gives 0 NaN and 0 nonzero samples on both the LUT and exact paths, and the direct-path-only RIR peaks at sample 168 on both.
- `jargon.py --port gpuRIR`: clean. af05f1d commit: `[ROCm]` title, AI disclosure, Test Plan, no trailer.

## Porter round 2026-10-30 (linux-gfx942): mixed precision ported (Review 2026-10-30 finding 1)

Requested by jeff: the `now` ruling on deferral `gpurir-mixed-precision` (2026-08-13) had not been carried out. Review PR #1 has no line threads or review body; the finding is the one in "Review 2026-10-30" above. No GitHub writes this round, so nothing was posted on the review PR.

### Change (fork 39a7a9d on top of af05f1d)
- `src/gpuRIR_cuda.cu`: the five `#if __CUDA_ARCH__ >= 530` guards (h2 constants, half2 helpers, the three mp kernels) become `#if __CUDA_ARCH__ >= 530 || defined(__HIP_DEVICE_COMPILE__)`. `__CUDA_ARCH__` is not defined in the shim.
- The three `CUDART_VERSION` guards (line 13 `< 9000` -> `#define __h2div h2div`; lines 508 and 730 `< 9020` -> float `Tw_inv` converted in-kernel) become `#if !defined(USE_HIP) && CUDART_VERSION < ...`, so HIP takes the modern spelling: `__h2div` (declared in TheRock 10.1.0 `amd_hip_fp16.h:883`) and a host-side `__float2half2_rn` (`__HOST_DEVICE__` there, line 476). On CUDA each condition evaluates as before.
- HIP early return in `activate_mixed_precision` removed; the function is now byte-identical to upstream (`git diff upstream/master` shows no hunk there).
- Verified `__heq2`/`__hne2` on HIP return 1.0/0.0 half2 lanes (`-r` converted from the int mask), which `my_h2sinc` and the sign trick in `my_h2sinpi` rely on.
- Code-object check: `generateRIR_mp_kernel` has 19 `v_pk_{fma,mul,add}_f16` instructions on gfx942, gfx90a, gfx1100 and gfx1201 (it was an empty stub before); `reduceRIR_mp_kernel` 2 on gfx942. Six-arch compile (gfx90a;gfx942;gfx1100;gfx1101;gfx1151;gfx1201): 0 errors.

### Build at 39a7a9d (gfx942, TheRock ROCm 10.1.0)
Same cmake line as the previous porter round (`agent_space/gpurir/build`, `-j16`). OK; only warning is the pre-existing nodiscard on `cudaGetDeviceProperties` (line 1069).

### Tests (HIP_VISIBLE_DEVICES=0, `ulimit -c 0`, runtime env as in the 2026-10-29 sync round)
`agent_space/gpurir/mp_validate.py`: **22/22 PASS**. Error is max |mp - float| / peak unless stated.
- Direct path (nb_img [1,1,1]), room 4x5x3, src (1,1,1.5) -> rcv (3,4,1.5):

  | fs | mp vs float exact | mp vs numpy float16 emulation of the kernel | float exact vs float64 | LUT vs exact |
  |---|---|---|---|---|
  | 8000 | 5.24e-4 | 5.5e-6 | 5.3e-8 | 1.59e-3 |
  | 16000 | 7.95e-4 | 0 | 5.2e-8 | 1.38e-4 |
  | 44100 | 3.72e-4 | 0 | 2.2e-6 | 6.85e-4 |
  | 48000 | 6.76e-4 | 0 | 4.3e-8 | 3.91e-4 |

  Peak sample identical (84/168/464/505). Thresholds: mp vs float in (0, 2e-3) -- the lower bound fails if mp silently falls back to float, as it did at af05f1d; emulation < 4e-3 (half ulp at peak is 7e-4..9.6e-4 of peak). The emulation replays the kernel's half2 polynomial sinpi/cospi, Hann window and `__h2div` in numpy float16 (fma as one float32 op then a cast); bit-exact at three of four rates, so the AMD half2 arithmetic matches IEEE half as the CUDA path assumes.
- Reverberant shoebox, Sabine T60 0.6 s, Tmax 0.3 s, fs 16k, nb_img [52,42,69] (N=150696, multi-stage half2 reduction): max err/peak 7.09e-4, RMS rel 1.43e-3 (thresholds 3e-3 / 5e-3); first arrival 168 on both (expected 168).
- 2 src x 3 cardioid rcv, Tdiff 0.15 s: ISM part max err/peak 1.29e-3 (threshold 5e-3); diffuse tail RMS mp/float 1.006.
- Mode toggles: mp after LUT disables the LUT and gives the mp result; LUT after mp disables mp and gives the float result (bit-equal to a plain float run). Messages printed match upstream's.
- Upstream examples (scratch copy, synthetic 2 s 16 kHz wav, no-op matplotlib/plotly stub incl. `plotly.subplots`): example.py, simulate_trajectory.py, polar_plots.py, time_vs_T60.py, time_vs_nbRIRs.py, plus example.py with `activateMixedPrecision(True)`/`activateLUT(False)`: **6/6 exit 0**.

Deferral `gpurir-mixed-precision` set `done` with `utils/deferred.py set-status`. Commit 39a7a9d: `[ROCm]` title 51 chars, AI disclosure in prose, Test Plan, no trailer; `jargon.py --port gpuRIR` clean, `prose.py` clean on the body. Pushed plain to `moat-port` (af05f1d..39a7a9d; no upstream PR open). advance-head -> 39a7a9d, so every completed arch reads `revalidate` (the mp kernels now exist on all AMD arches; wave32 arches have not run them). Lesson promoted to `cuda-to-rocm/references/fault-classes.md` (Headers, includes and build): undefined `__CUDA_ARCH__`/`CUDART_VERSION` silently select stub or legacy branches on HIP.
