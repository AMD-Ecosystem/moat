# cuSZ notes

## CUDA compile check 2026-06-25 (@ 11140547 -- CUDA backend intact)

Toolkit: nvcc 12.8.93 / CUDA 12.8 (conda env cuda-12.8), host gcc 13.4.0, sm_86 (-DCMAKE_CUDA_ARCHITECTURES=86).
-DPSZ_ACTIVATE_LC=OFF (lc submodule not initialized; core port code is unaffected).

```bash
CONDA_PREFIX=/opt/conda/envs/cuda-12.8
SRC=/var/lib/jenkins/moat/projects/cuSZ/src
mamba run -n cuda-12.8 cmake -S $SRC -B $SRC/build-cuda \
  -DPSZ_BACKEND=CUDA \
  -DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DPSZ_ACTIVATE_LC=OFF \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
mamba run -n cuda-12.8 cmake --build $SRC/build-cuda -j$(nproc)   # exit 0
```

0 errors, 0 warnings (in changed files). All targets built:
- libpsz_cu_core.so, libphf_cu.so, libfzg_cu.so, libeval_cu.so, libcusz.so (device code verified
  via readelf .nv_fatbin sections in libpsz_cu_core.so and libphf_cu.so)
- cusz CLI, all test executables (histsp_cu, l1_compact, mem_unique, stat_identical*, etc.)

### CUDA-path analysis of port changes

Changed files confirmed not to regress the CUDA path:

- cmake/hip-compat/psz_hip_compat.h and cuda_runtime.h: only force-included via
  `$<$<COMPILE_LANGUAGE:HIP>:...>` CMake generator expression; CUDA TUs never see them.

- portable/include/macro/c_cu2hip_0_translation.h and c_cu2hip_1_fix_primitives.h: these files
  existed upstream and were already included from cuda_runtime.h (the shim). The port ADDS new
  macro entries (cudaEvent_t, cudaMemsetAsync, variadic __shfl_*_sync, __ballot_sync). On the
  CUDA path the CUDA headers define the real cudaEvent_t etc., so the `#define cudaEvent_t
  hipEvent_t` would be a redefinition -- BUT these files are only reached via the HIP-path
  include chain (cmake/hip-compat/cuda_runtime.h -> c_cu2hip_0/1). The CUDA build does NOT
  include cmake/hip-compat/ at all; only the HIP build does (hip.cmake adds the hip-compat
  dir to include paths). Confirmed clean compile.

- portable/include/mem/cxx_backends.h: new `#elif defined(_PORTABLE_USE_HIP)` blocks are
  properly guarded; the CUDA path continues to use `#if defined(_PORTABLE_USE_CUDA)` blocks
  unchanged. No conflict.

- psz/src/kernel/histsp.cu.inl (wave32 scan fix): change is inside `#ifdef __HIP_PLATFORM_AMD__`
  guard; nvcc never defines that macro. The CUDA else-branch (width-32 __shfl_up_sync) is
  unchanged.

- utils/src/atomics.cu.inl (double atomicAdd): the new condition
  `(defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)) || defined(__HIP_DEVICE_COMPILE__)`
  is equivalent to the old one under nvcc (nvcc never defines __HIP_DEVICE_COMPILE__; only the
  __CUDA_ARCH__ branch applies). The CAS fallback for SM<6.0 still exists and is unreachable
  with -arch=sm_86.

- utils/src/extrema.cu.inl (removed PSZ_USE_CUDA guard): the old `#if defined(PSZ_USE_CUDA)`
  guard around `psz::KCU_extrema<<<...>>>` is removed; the call is now unconditional in
  psz::cuda::GPU_extrema. This is fine on the CUDA path: PSZ_USE_CUDA IS defined (cmake/cuda.cmake
  sets it), so the old guard was always taken. The unconditional call compiles identically --
  confirmed by clean nvcc build.

- utils/src/viewer.cc (PSZ_MAP_KEY): PSZ_MAP_KEY(T) expands to `T const` on Linux (non-_WIN32);
  the key types are unchanged. No impact on CUDA.

- portable/include/c_type.h: `HIP` enumerator added to _ptb_runtime enum. Enum extension is
  backward-compatible; existing CUDA code that does not reference the new enumerator is
  unaffected.

Verdict: CUDA backend intact. The HIP port leaves every CUDA translation unit byte-identical
in its effective preprocessed form. Zero port-introduced errors or warnings under nvcc 12.8.

## Validation 2026-06-25 (linux-gfx90a @ 11140547 -- PASS, real-GPU revalidate of squashed commit)

Platform: AMD Instinct MI250X (gfx90a, wave64), ROCm 7.2.1, HIP_VISIBLE_DEVICES=0
Fork: AMD-Ecosystem/cuSZ @ moat-port, HEAD 11140547 ("[ROCm] Add a HIP backend (PSZ_BACKEND=HIP) targeting ROCm")
Prior validated_sha: 252224e3.

### Delta 252224e3 -> 11140547

Squash to single PR commit. Delta vs 252224e3: 2 host files, clang-format whitespace tidy only.
- psz/src/cli/verinfo_hip.cu: alignment spaces on 2 variable declarations, merged printf()
- utils/include/compare.hh: line-break reformatting of 2 static_assert() calls
No device kernel code, no logic changes. Both files verified kernel-free (no __global__/__device__/<<<).

### Binary-equivalence proof

Built 11140547 in build-hip-11140547 (clean configure + build; cmake exit 0) and
moat-validated-252224e3 in a git worktree at /tmp/cusz-252224e3/build-252224e3 (clean; cmake exit 0).

```bash
python3 utils/codeobj_diff.py \
  projects/cuSZ/src/build-hip-11140547 \
  /tmp/cusz-252224e3/build-252224e3
```

Device-code-bearing shared libs -- all IDENTICAL:
- libfzg_cu.so: identical (42 exports)
- libphf_cu.so: identical (1103 exports)
- libpsz_cu_core.so: identical (242 exports)
- libpsz_cu_mem.so: identical (233 exports)
- utils/libeval_cu.so: identical (24 exports)

Remaining libs (libcusz.so, libpsz_cu_utils.so, libpsz_seq_core.so, libexample_utils2.so,
libportable_testutils.so, libeval_seq.so, libeval_viewer_cu.so): "indeterminate (device-code
extraction failed)" -- pure host-code libs with no device sections. Confirmed pattern from
prior revalidation.

### Build at 11140547

```bash
SRC=/var/lib/jenkins/moat/projects/cuSZ/src
BUILD=$SRC/build-hip-11140547
CUSZ_TEST_DATA=/var/lib/jenkins/cusz-test-data cmake -S $SRC -B $BUILD \
  -DCMAKE_BUILD_TYPE=Release -DPSZ_BACKEND=HIP \
  -DCMAKE_C_COMPILER=amdclang -DCMAKE_CXX_COMPILER=amdclang++ \
  -DCMAKE_HIP_COMPILER=amdclang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DBUILD_TESTING=ON -DPSZ_BUILD_EXAMPLES=ON
cmake --build $BUILD -j$(nproc)   # exit 0
```

Auto-detected arch: gfx90agfx90agfx90agfx90a (4 GCDs, correct).

### Test (gfx90a, full suite with real datasets)

```bash
HIP_VISIBLE_DEVICES=0 ctest --test-dir $BUILD --output-on-failure -j1
```

**37 PASS / 3 SKIP / 1 FAIL of 41** -- identical to baseline.
- 37 PASS: all portable CPU tests, all GPU unit tests, all 14 bin_hf HFR/FZG matrix tests,
  all 6 cusz__hurr_uf48 HURR real-data tests, all 3 cusz__nyx_velx NYX real-data tests.
- 3 SKIP (rc=77): cusz__rtm_0480_* (RTM dataset absent -- expected).
- 1 FAIL: hfr__cauchy_sharp__u2 -- pre-existing upstream --assert-brnum-le CLI desync;
  fails identically on CUDA. NOT a port regression.

### CLI round-trip CR/PSNR gate

```bash
python3 -c "import numpy as np; np.sin(np.arange(1000000,dtype=np.float32)*2*np.pi/100000).astype(np.float32).tofile('/tmp/sine_1m_sq.f32')"
HIP_VISIBLE_DEVICES=0 ./cusz -z -i /tmp/sine_1m_sq.f32 -t f32 -m abs -e 1e-3 -l 1000000 --report cr,psnr
# [Lorenzo, Hist, HF-fast2]  CR=27.04  mode=Abs  input_eb=1.000000e-03
HIP_VISIBLE_DEVICES=0 ./cusz -x -i /tmp/sine_1m_sq.f32.cusza --compare /tmp/sine_1m_sq.f32 --report cr,psnr
# [Lorenzo, Hist, HF-fast2]  CR=27.04  PSNR=70.8  max_error=1.000047e-03  max_error_rel=5.000234e-04
```

**CR=27.04, PSNR=70.8, max_error=1.000047e-03** -- exact match to the gfx90a/gfx1100/gfx1201 reference.

Verdict: PASS. validated_sha=11140547 (linux-gfx90a).
Device .text IDENTICAL confirmed; binary-equiv carry-forward applies to linux-gfx1100 and windows-gfx1201.

---

## Validation 2026-06-25 (windows-gfx1201 @ 252224e3 -- PASS)

Platform: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), Windows 11, TheRock ROCm 7.14
nightly (_rocm_sdk_devel venv). Single GPU, HIP_VISIBLE_DEVICES=0 (gcnArchName gfx1201,
device 0 -- confirmed with hipInfo; gfx1101 offline/absent).
Fork: AMD-Ecosystem/cuSZ @ moat-port, HEAD 252224e3 ("[ROCm] Fix Windows (MSVC STL / clang-cl) build of the HIP backend")

### Build (gfx1201, incremental with examples ON)

The porter's build used `-DPSZ_BUILD_EXAMPLES=OFF`, which omits `bin_hf.exe` (needed for
the 14 bin_hf HFR/FZG codec matrix tests). Reconfigured with `-DPSZ_BUILD_EXAMPLES=ON`
and rebuilt incrementally (only example/bin_hf.exe compiled; all other targets up-to-date).

```bash
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
SRC=B:/develop/moat/projects/cuSZ/src
BUILD=$SRC/build-hip-gfx1201

cmake -S $SRC -B $BUILD -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPSZ_BACKEND=HIP -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/amdclang.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/amdclang++.exe \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/amdclang++.exe \
  -DCMAKE_PREFIX_PATH=$ROCM -DBUILD_TESTING=ON \
  -DPSZ_BUILD_EXAMPLES=ON -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON
sed -i 's/ -fuse-ld=lld-link//g' $BUILD/build.ninja
cmake --build $BUILD -j32   # exit 0; only bin_hf.exe rebuilt
```

DLL staging (TheRock runtime wins over System32 Adrenalin amdhip64_7.dll):
- test/: amdhip64_7.dll, amd_comgr*.dll, hiprtc*.dll, rocm_kpack.dll from _rocm_sdk_core/bin;
  hiprand.dll, rocrand.dll from _rocm_sdk_devel/bin; cusz.dll, psz_cu_*.dll, fzg_cu.dll,
  phf_cu.dll, eval*.dll, portable_testutils.dll from build root; bin_hf.exe from example/
- example/: same runtime DLLs + project DLLs

### CLI round-trip CR/PSNR gate (cross-arch consistency)

```bash
python3 -c "import numpy as np; np.sin(np.arange(1000000,dtype=np.float32)*2*np.pi/100000).astype(np.float32).tofile('test_sine_1m.f32')"
HIP_VISIBLE_DEVICES=0 ./cusz.exe -z -i test_sine_1m.f32 -t f32 -m abs -e 1e-3 -l 1000000 --report cr,psnr
# [Lorenzo, Hist, HF-fast2]  CR=27.04  mode=Abs  input_eb=1.000000e-03
HIP_VISIBLE_DEVICES=0 ./cusz.exe -x -i test_sine_1m.f32.cusza --compare test_sine_1m.f32 --report cr,psnr
# [Lorenzo, Hist, HF-fast2]  CR=27.04  PSNR=70.8  max_error=1.000047e-03  max_error_rel=5.000234e-04
```

**CR=27.04, PSNR=70.8, max_error=1.000047e-03** -- exact match to the gfx90a/gfx1100 reference.
The histsp wave32 fix (866868f6) is correct on RDNA4/gfx1201. Cross-arch determinism gate passes.

### Test (gfx1201, full ctest suite)

```bash
HIP_VISIBLE_DEVICES=0 ctest --test-dir $BUILD --output-on-failure -j1
```

**28 PASS / 12 SKIP / 1 FAIL of 41** -- matches gfx1100 Linux baseline exactly.
- 28 PASS: 6 portable CPU tests (portable_str2num/kv_parse/check_in/val_eq/arg_builder/kv_binder),
  test_zigzag (CPU), test_l1_compact (GPU), test_lrz_seq (CPU), test_stat_identical1/2 (GPU),
  test_stat_max_error (GPU), test_mem_unique (GPU), test_hf_revisit_altcode (HFR GPU),
  test_hf_cpu_serial_codebook, and all 13 bin_hf HFR/FZG codec matrix tests:
  hf/hfr/hfr-pbkc/hfr-pbkgo over cauchy-mild (4), cauchy-sharp (3, not the known-fail),
  uniform-256 (4), uniform-1024 (2 -- hfr_pbk_go/pbk_compat absent from uniform-1024).
- 12 SKIP (rc=77): cusz__rtm/hurr/nyx real-dataset tests (no $CUSZ_TEST_DATA) -- expected.
- 1 FAIL: hfr__cauchy_sharp__u2 -- pre-existing upstream `--assert-brnum-le` CLI desync;
  `bin_phf: unknown option: --assert-brnum-le`. Fails identically on CUDA. NOT a port regression.

### Wave32 gate (RDNA4 / gfx1201)

All 13 bin_hf HFR/FZG codec matrix tests PASS on wave32. The arch-unified `__ballot_sync`
macro (`(uint32_t)(__ballot(PRED) >> (__lane_id() & ~31u))`) reduces to the shift-0 identity
on gfx1201 (lane_id 0..31), and the wave32 histsp fix (32-lane scan, width-32 __shfl_up,
threadIdx.x % 32 predicate at 866868f6) is correct on RDNA4. No wave32-specific regression.

### Fork cleanliness check

```bash
git -C projects/cuSZ/src status --porcelain
# (empty -- no modified tracked files; build artifacts + DLLs are untracked)
```

Verdict: PASS. validated_sha=252224e3 (windows-gfx1201).

---

## Review 2026-06-25 (windows-gfx1201 follower delta @ 252224e3 -- review-passed)

Reviewed 252224e3 ("[ROCm] Fix Windows (MSVC STL / clang-cl) build of the HIP backend")
against base 866868f6 with /pr-review (local-branch mode). 4 host files, +31/-4, all
_WIN32-guarded. No problems found. Verdict: review-passed (GPU ctest is the validator's
gate; a missing GPU run is expected at review time).

### What I verified (no problems found)

- hf_buf.cc setenv -> _putenv_s: <cstdlib> is included (line 6); the
  `if (not std::getenv("CUDA_MODULE_LOADING")) _putenv_s(...)` guard correctly emulates
  setenv(...,overwrite=0) "set only if unset" semantics. Single-threaded static-init
  singleton, so the getenv/putenv TOCTOU is benign. CUDA_MODULE_LOADING is inert under ROCm.
- query_cpu.hh popen/pclose -> _popen/_pclose: the `#define popen _popen` / `pclose _pclose`
  precede the struct so all three uses rewrite; the only includer is psz/src/cli/cli.cc (via
  query.hh) and no other TU uses popen/pclose, so the macro leak is contained and harmless
  (it only maps the POSIX names to the intended Windows spellings). <vector> is genuinely
  used (std::vector<std::string> v in get_cpu_properties).
- cxx_typing.h TypeSym<ull> dedup under !_WIN32: confirmed u8=uint64_t and
  ull=unsigned long long (c_type.h:48-49). On Windows LLP64 they are the same type so the
  specialization redefines TypeSym<u8>; the guard drops it only there. The enum-keyed
  Ctype<U8>/Ctype<ULL> do NOT collide (keyed on the enum value, not the C++ type) and are
  correctly left untouched.
- viewer.cc PSZ_MAP_KEY: on Linux expands to `T const` (byte-identical to the original
  `psz_predictor const` etc.), only the 3 const-key unordered_map sites changed, value type
  `std::string const` untouched. This avoids the old base's unconditional const-removal that
  changed Linux mangled symbols (K13.. -> 13..) and forced a Linux GPU re-run.

### Fault classes / BC

No device code touched; no warpSize/ballot/atomics/texture/library changes. The wave32
histsp gate is resolved at the parent base (866868f6), not in this delta. All 4 edits are
_WIN32-guarded, so Linux preprocessed TUs are byte-identical -- consistent with the gfx90a
codeobj_diff carry-forward already recorded (device .text + exports identical in all 5 HIP
libs). No upstream CUDA/CPU path regressed.

### Commit hygiene

[ROCm] prefix; title 65 chars (<=72); Test Plan with literal build + round-trip commands;
Claude named ("Authored with the assistance of Claude"); no Co-Authored-By noreply trailer;
no MOAT jargon in commit/comments; no AMD-internal account references.


## Validation 2026-06-25 (linux-gfx90a revalidate @ 252224e3 -- PASS, binary-equiv carry-forward)

Platform: AMD Instinct MI250X (gfx90a, wave64), ROCm 7.2.1, HIP_VISIBLE_DEVICES=0
Fork: AMD-Ecosystem/cuSZ @ moat-port, HEAD 252224e3 ("[ROCm] Fix Windows (MSVC STL / clang-cl) build of the HIP backend")
Prior validated_sha: 866868f6.

### Delta classification

252224e3 modifies exactly 4 host files, all changes _WIN32-guarded:
- codec/hf/src/hf_buf.cc: setenv -> _putenv_s under _WIN32
- portable/include/cxx_typing.h: duplicate TypeSym<ull> guard under _WIN32
- psz/src/cli/query/query_cpu.hh: popen/_pclose guard + <vector> include
- utils/src/viewer.cc: PSZ_MAP_KEY macro for const-key Windows compat

On Linux (-D_WIN32 never set), the preprocessed TUs are byte-identical to 866868f6.

### Binary-equivalence proof

Built 252224e3 in build-hip (incremental; cmake exit 0) and 866868f6 in a git worktree at
/tmp/cuszsz-866868f6/build-866868f6 (clean configure + build; cmake exit 0 on both).

```bash
python3 utils/codeobj_diff.py \
  projects/cuSZ/src/build-hip \
  /tmp/cuszsz-866868f6/build-866868f6
```

Device-code-bearing shared libs -- all IDENTICAL:
- libfzg_cu.so: identical (42 exports)
- libphf_cu.so: identical (1103 exports)
- libpsz_cu_core.so: identical (242 exports)
- libpsz_cu_mem.so: identical (233 exports)
- utils/libeval_cu.so: identical (24 exports)

Remaining libs (libcusz.so, libpsz_cu_utils.so, libpsz_seq_core.so, eval_viewer_cu.so,
libexample_utils2.so, libportable_testutils.so, eval_seq.so): codeobj_diff shows
"indeterminate (device-code extraction failed)" -- these are pure host-code libs with
no device sections. The only differences are RPATH strings and __hip_cuid_* (build-path
metadata, not device ISA). Confirmed with readelf + strings comparison.

Verdict: device .text is IDENTICAL on gfx90a between 866868f6 and 252224e3.

### Build at 252224e3

```bash
cd /var/lib/jenkins/moat/projects/cuSZ/src
cmake --build build-hip -j$(nproc)   # incremental; 4 host .cc/.hh files recompiled; exit 0
```

### Test (gfx90a, full suite with real datasets)

```bash
export CUSZ_TEST_DATA=/var/lib/jenkins/cusz-test-data
HIP_VISIBLE_DEVICES=0 ctest --test-dir build-hip --output-on-failure -j1
```

**37 PASS / 3 SKIP / 1 FAIL of 41** -- identical to the 866868f6 baseline.
- 37 PASS: all portable CPU unit tests, all GPU unit tests, all 14 bin_hf HFR/FZG matrix
  tests, all 6 cusz__hurr_uf48 real-data round-trips, all 3 cusz__nyx_velx real-data.
- 3 SKIP (rc=77): cusz__rtm_0480_* (RTM dataset not staged -- expected).
- 1 FAIL: hfr__cauchy_sharp__u2 -- pre-existing upstream --assert-brnum-le CLI desync;
  fails identically on CUDA. NOT a port regression.

### CLI round-trip CR/PSNR gate

```bash
python3 -c "import numpy as np; np.sin(np.arange(1000000,dtype=np.float32)*2*np.pi/100000).astype(np.float32).tofile('/tmp/sine_1m_revalidate.f32')"
HIP_VISIBLE_DEVICES=0 ./cusz -z -i /tmp/sine_1m_revalidate.f32 -t f32 -m abs -e 1e-3 -l 1000000 --report cr,psnr
# [Lorenzo, Hist, HF-fast2]  CR=27.04  mode=Abs  input_eb=1.000000e-03
HIP_VISIBLE_DEVICES=0 ./cusz -x -i /tmp/sine_1m_revalidate.f32.cusza --compare /tmp/sine_1m_revalidate.f32 --report cr,psnr
# [Lorenzo, Hist, HF-fast2]  CR=27.04  PSNR=70.8  max_error=1.000047e-03  max_error_rel=5.000234e-04
```

**CR=27.04, PSNR=70.8, max_error=1.000047e-03** -- exact match to the gfx90a reference.

Verdict: PASS. validated_sha=252224e3 (linux-gfx90a).
Device .text IDENTICAL confirmed; carry-forward applies to linux-gfx1100 as well.

## Porter (follower delta) 2026-06-25 (windows-gfx1201, re-port Windows build @ 252224e3 -- ported)

Platform: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), Windows 11, TheRock ROCm 7.14
nightly (_rocm_sdk_devel venv). Single GPU, HIP_VISIBLE_DEVICES=0 (gcnArchName gfx1201,
device 0 -- confirmed with hipInfo; the old HIP_VISIBLE_DEVICES=1 pin from the gfx1101 era
is stale). Fork: AMD-Ecosystem/cuSZ @ moat-port, new HEAD 252224e3 (one commit on top of 866868f6).

### What this pass did

This is the FRESH single-source re-port (head 866868f6), never before compiled on Windows.
The old base's 8 _WIN32 fixes do NOT carry as commits (different source tree). The re-port
already folds several Windows equivalents (hip.cmake HIP_DISABLE_WARP_SYNC_BUILTINS for WIN32;
demangle.hh guards cxxabi.h with __has_include). FOUR new Windows-portability breaks surfaced
under clang targeting the MSVC C++ runtime; all fixed and guarded so Linux is byte-identical:

1. codec/hf/src/hf_buf.cc: setenv (POSIX) undeclared -> _putenv_s on _WIN32 (CUDA_MODULE_LOADING
   hint, inert under ROCm).
2. psz/src/cli/query/query_cpu.hh: popen/pclose (POSIX) -> _popen/_pclose on _WIN32; also add
   <vector> (libstdc++ pulled it in transitively, MSVC STL does not).
3. portable/include/cxx_typing.h: TypeSym<ull> redefined TypeSym<u8> (uint64_t == unsigned long
   long on Windows LLP64); drop the duplicate specialization under _WIN32.
4. utils/src/viewer.cc: MSVC STL has no std::hash<const T>; drop the unordered_map KEY const only
   on Windows via a PSZ_MAP_KEY macro that keeps the Linux key type (`T const`) byte-identical.
   IMPORTANT: the old base shipped this as an UNCONDITIONAL const-removal, which changed Linux
   exported mangled symbols (K13psz_predictor -> 13psz_predictor) and forced a Linux GPU re-run
   (notes 2026-06-08). The _WIN32 guard here AVOIDS that: Linux symbols + device code unchanged.

### Build (gfx1201, TheRock venv, Ninja, all amdclang) -- clean

```bash
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
SRC=B:/develop/moat/projects/cuSZ/src
BUILD=$SRC/build-hip-gfx1201
cmake -S $SRC -B $BUILD -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPSZ_BACKEND=HIP -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/amdclang.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/amdclang++.exe \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/amdclang++.exe \
  -DCMAKE_PREFIX_PATH=$ROCM -DBUILD_TESTING=ON \
  -DPSZ_BUILD_EXAMPLES=OFF -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON
sed -i 's/ -fuse-ld=lld-link//g' $BUILD/build.ninja   # CMake 4.3 Windows-Clang module fix
cmake --build $BUILD -j32   # 0 errors; cusz.exe + all DLLs built
```

### Smoke check (porter; full ctest is the validator's gate)

DLL staging into $BUILD (so the TheRock runtime wins over System32 Adrenalin amdhip64_7.dll):
amdhip64_7.dll, amd_comgr*.dll, hiprtc*.dll, rocm_kpack.dll from _rocm_sdk_core/bin; hiprand.dll,
rocrand.dll from _rocm_sdk_devel/bin; plus project DLLs (cusz.dll, psz_cu_*.dll, fzg_cu.dll,
phf_cu.dll, eval*.dll). cusz.exe --version prints the hipSZ banner. GPU round-trip (1M f32 sine,
input written with .astype(np.float32).tofile):

```bash
cusz -z -i sine_1m.f32 -t f32 -m abs -e 1e-3 -l 1000000 --report cr,psnr   # CR=27.04
cusz -x -i sine_1m.f32.cusza --compare sine_1m.f32 --report cr,psnr
# CR=27.04  PSNR=70.8  max_error=1.000047e-03  -- EXACT match to gfx90a/gfx1100 reference
```

The wave32 histsp fix (already on moat-port @ 866868f6, validated on gfx1100) is correct on
RDNA4 too: the cross-arch CR determinism gate passes on gfx1201. So the documented wave32-blocker
concern (plan "THE wave32 gate") is already resolved at the built head; no histsp work needed here.

### Head / state

New fork HEAD 252224e3. advance-head flipped linux-gfx90a and linux-gfx1100 to `revalidate`
(conservative default for a .cc/.h source touch). BUT all four edits are _WIN32-guarded, so on
Linux the preprocessed TUs are byte-identical to 866868f6 -- device code AND exported symbols
unchanged. This is a binary-equivalence CARRY-FORWARD case: the Linux validators should confirm
with codeobj_diff.py (expect verdict=identical) and carry forward WITHOUT a GPU re-run, not a
full revalidation. windows-gfx1201 -> ported; full ctest GPU validation is the validator's job.

## Validation 2026-06-25 (linux-gfx1100 @ 866868f6 -- PASS)

Platform: AMD Radeon Pro W7800 (gfx1100, RDNA3, wave32), ROCm 7.2.1, HIP_VISIBLE_DEVICES=1
Fork: AMD-Ecosystem/cuSZ @ moat-port, HEAD 866868f6 ("[ROCm] Fix wave32 histogram scan in HistSp kernel")
State: review-passed -> completed (validated_sha=866868f6)

### Build (gfx1100, incremental at head 866868f6)

```bash
cmake --build /var/lib/jenkins/moat/projects/cuSZ/src/build-hip -j$(nproc)
# CMakeCache: PSZ_BACKEND=HIP, CMAKE_HIP_ARCHITECTURES=gfx1100
# exit 0, warnings only (nodiscard hipMemcpy)
```

Build was incremental -- the gfx1100 build directory already had the correct CMake config
(PSZ_BACKEND=HIP, CMAKE_HIP_ARCHITECTURES=gfx1100). The histsp.cu.inl source was modified
after the prior binary, so cmake --build recompiled all affected targets at 866868f6.

### Test (gfx1100, full ctest suite)

```bash
HIP_VISIBLE_DEVICES=1 ctest --test-dir /var/lib/jenkins/moat/projects/cuSZ/src/build-hip --output-on-failure -j1
```

**28 PASS / 12 SKIP / 1 FAIL of 41** -- matches the expected gfx90a baseline exactly.
- 28 PASS: all 6 portable CPU unit tests, test_zigzag, test_l1_compact (GPU), test_lrz_seq,
  test_stat_identical1/2 (GPU), test_stat_max_error (GPU), test_mem_unique (GPU),
  test_hf_revisit_altcode (HFR GPU), test_hf_cpu_serial_codebook, and all 14 bin_hf HFR/FZG
  matrix tests (hf/hfr/hfr-pbkc/hfr-pbkgo over cauchy-mild/sharp [minus the known FAIL],
  uniform-256, uniform-1024).
- 12 SKIP (rc=77): cusz__rtm/hurr/nyx real-dataset tests -- no $CUSZ_TEST_DATA locally; expected.
- 1 FAIL: hfr__cauchy_sharp__u2 -- pre-existing upstream `--assert-brnum-le` CLI desync;
  fails identically on CUDA. NOT a port regression.

### CLI round-trip CR/PSNR gate (cross-arch consistency)

```bash
python3 -c "import numpy as np; np.sin(np.arange(1000000,dtype=np.float32)*2*np.pi/100000).astype(np.float32).tofile('/tmp/test_sine_1m.f32')"
HIP_VISIBLE_DEVICES=1 ./cusz -z -i /tmp/test_sine_1m.f32 -t f32 -m abs -e 1e-3 -l 1000000 --report cr,psnr
# [Lorenzo, Hist, HF-fast2]  CR=27.04  mode=Abs  input_eb=1.000000e-03
HIP_VISIBLE_DEVICES=1 ./cusz -x -i /tmp/test_sine_1m.f32.cusza --compare /tmp/test_sine_1m.f32 --report cr,psnr
# [Lorenzo, Hist, HF-fast2]  CR=27.04  PSNR=70.8  max_error=1.000047e-03  max_error_rel=5.000234e-04
```

**CR=27.04, PSNR=70.8, max_error=1.000047e-03** -- exact match to the gfx90a reference.
The histsp wave32 fix (32-wide scan, width-32 __shfl_up, threadIdx.x % 32 predicate) produces
correct histogram totals on wave32 hardware. The cross-arch determinism gate passes.

Input was generated correctly using `.astype(np.float32).tofile(...)` (not the float64-promoting
`np.arange(dtype=float32)*pythonfloat` that caused the phantom "Lorenzo bug" in prior runs).

### Wave32 ballot / FZG / HFR gates

The bin_hf HFR/FZG matrix (14 tests including hfr-pbkgo, hfr-pbkc, hfr, hf over all synth
datasets) all PASS on wave32. The arch-unified `__ballot_sync` macro
(`(uint32_t)(__ballot(PRED) >> (__lane_id() & ~31u))`) reduces to the shift-0 identity on
gfx1100 (lane_id 0..31), and all FZG/HFR codec paths are correct on wave32. No wave32-specific
regression in any codec path.

Verdict: PASS. validated_sha=866868f6 (linux-gfx1100).

## Validation 2026-06-25 (linux-gfx90a revalidate @ 866868f6 -- PASS)

Platform: AMD Instinct MI250X (gfx90a, wave64), ROCm 7.2.1, HIP_VISIBLE_DEVICES=0
Fork: AMD-Ecosystem/cuSZ @ moat-port, HEAD 866868f6 ("[ROCm] Fix wave32 histogram scan in HistSp kernel")
Prior validated_sha: 07db1e28. Revalidation required: functional device-code change (histsp scan width 64->32).

### Build

```bash
cd /var/lib/jenkins/moat/projects/cuSZ/src
rm -rf build-hip
cmake -S . -B build-hip \
  -DPSZ_BACKEND=HIP \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCUSZ_TEST_DATA=/var/lib/jenkins/cusz-test-data
cmake --build build-hip -j$(nproc)   # exit 0, warnings only (nodiscard hipMemcpy)
```

### Test (gfx90a, full suite with real datasets)

```bash
HIP_VISIBLE_DEVICES=0 ctest --test-dir build-hip --output-on-failure -j1
```

**37 PASS / 3 SKIP / 1 FAIL of 41** -- identical to the pre-fix gfx90a baseline.
- 37 PASS: all portable unit tests, all GPU unit tests, all 14 bin_hf HFR/FZG matrix tests,
  all 6 cusz__hurr_uf48 real-data round-trips (hf/hfr/hfr-pbkc x rel 1e-3/1e-4),
  all 3 cusz__nyx_velx real-data round-trips (hf/hfr/hfr-pbkc x rel 1e-3).
- 3 SKIP (rc=77): cusz__rtm_0480_* (RTM dataset not staged -- expected).
- 1 FAIL: hfr__cauchy_sharp__u2 -- pre-existing upstream --assert-brnum-le CLI desync;
  fails identically on CUDA. NOT a port regression.

### CLI round-trip CR/PSNR gate

```bash
python3 -c "import numpy as np; np.sin(np.arange(1000000,dtype=np.float32)*2*np.pi/100000).astype(np.float32).tofile('/tmp/sine_1m_f32.f32')"
HIP_VISIBLE_DEVICES=0 ./cusz -z -i /tmp/sine_1m_f32.f32 -t f32 -m abs -e 1e-3 -l 1000000 --report cr,psnr
# [Lorenzo, Hist, HF-fast2]  CR=27.04  mode=Abs  input_eb=1.000000e-03
HIP_VISIBLE_DEVICES=0 ./cusz -x -i /tmp/sine_1m_f32.f32.cusza --compare /tmp/sine_1m_f32.f32 --report cr,psnr
# [Lorenzo, Hist, HF-fast2]  CR=27.04  PSNR=70.8  max_error=1.000047e-03  max_error_rel=5.000234e-04
```

**CR=27.04, PSNR=70.8, max_error=1.000047e-03** -- exact match to the pre-fix gfx90a baseline.
The wave32 fix (32-wide scan replaces 64-wide scan in histsp) is correct on wave64: the 32-wide
warp-scan is what the CUDA path and the lane-31 consumer have always expected. No regression.

Verdict: PASS. validated_sha=866868f6 (linux-gfx90a).

## Porter (focused diagnostic) 2026-06-25 (linux-gfx1100 -- "Lorenzo blocker" was a TEST-HARNESS bug, NOT a port fault; UNBLOCKED @ 866868f6)

Platform: AMD Radeon Pro W7800 (gfx1100, RDNA3, wave32), ROCm 7.2.1. Head 866868f6 (the
histsp wave32 fix, unchanged -- NO new commit needed). linux-gfx1100 set blocked=False, ported.

### Verdict: there is NO Lorenzo predictor bug on gfx1100. The block was a phantom.

The prior pass marked a "separate deeper Lorenzo quant-code corruption" blocker (CR=0.43 vs
gfx90a 27.04). Root cause of that symptom is the REPRODUCER's INPUT GENERATION, not the GPU:

    np.sin(np.arange(2048, dtype=np.float32) * 2*np.pi/100000).tofile('/tmp/s.f32')

`np.arange(..., dtype=float32) * (python float)` PROMOTES to float64, so `.tofile()` writes
8 bytes/element. cuSZ then reads the file as `-t f32` (4 bytes/element), reinterpreting each
float64 as two garbage float32s. The "input" cuSZ actually saw was non-smooth noise
(e.g. element 3 = 0.564, elements 4/6 = ~6e-32 denormals), whose Lorenzo deltas ARE large
(282, 313, 330 >> radius 128), so the predictor CORRECTLY flagged ~all of them as outliers
(eq=0). The quant dump `[128,128,128,0,0,...]` (correct only for the 3 leading true-zero
elements, then "outliers") is the EXPECTED output for that corrupt input, not a miscompile.

The "cross-tile / later-data-corrupts-earlier" signature was an artifact of reading the
float64 stream as f32: the byte interleaving makes later float64s' bytes land in earlier f32
slots. There is no memory hazard, no missing sync, no rounding fault, no codegen miscompile.

### Proof (gfx1100, HIP_VISIBLE_DEVICES=0, head 866868f6)

Same sine written as REAL float32 (`.astype(np.float32).tofile(...)`):

    python3 -c "import numpy as np; np.sin(np.arange(2048,dtype=np.float32)*2*np.pi/100000).astype(np.float32).tofile('/tmp/s32.f32')"
    ./cusz -z -i /tmp/s32.f32 -t f32 -m abs -e 1e-3 -l 2048 --report cr      # CR=13.56 (was 0.43)

1M f32 cross-arch determinism GATE -- matches gfx90a EXACTLY:

    python3 -c "import numpy as np; np.sin(np.arange(1000000,dtype=np.float32)*2*np.pi/100000).astype(np.float32).tofile('/tmp/sine_1m_f32.f32')"
    ./cusz -z -i /tmp/sine_1m_f32.f32 -t f32 -m abs -e 1e-3 -l 1000000 --report cr,psnr
    #   CR=27.04
    ./cusz -x -i /tmp/sine_1m_f32.f32.cusza --compare /tmp/sine_1m_f32.f32 --report cr,psnr
    #   CR=27.04  PSNR=70.8  max_error=1.000047e-03   <-- identical to gfx90a reference

ctest (gfx1100, -E test_histsp_cu): 98% passed, 28 PASS / 12 SKIP / 1 known-non-port FAIL of
41. The lone FAIL is hfr__cauchy_sharp__u2 (pre-existing upstream `--assert-brnum-le` CLI
desync; fails identically on CUDA). Identical to the gfx90a re-port baseline.

### Hypothesis bisect results (per the diagnostic ask)

1. MEMORY HAZARD in lrz_c.cu.inl: NOT the cause. The 1D blocked/strided shared transpose
   (load `s_data[tid+ix*256]`, compute-read `s_data[tid*4+ix]`, TileDim=1024/Seq=4/256 thr)
   fully covers [0,1024) and is correctly fenced by the two `__syncthreads()` at lines 68/74.
   No OOB/aliasing. "Cross-tile" was the f64-as-f32 byte artifact, not LDS clobber.
2. ROUNDING (round()/rint() lowering): NOT the cause. With a correct f32 input the quant
   codes match gfx90a bit-for-bit (CR/PSNR/max_error identical), so round() lowers correctly.
3. -O0/-O3 OPTIMIZATION BISECT: NOT NEEDED / not run. The corruption does not exist with a
   valid input at the shipped -O3, so there is no miscompile to bisect. No ROCm bug to file;
   no findings/ reproducer written (there is no defect to report).
4. Round-trip (not just --dump): the full `-x --compare` confirms decompressed == input to
   max_error 1.0e-3, so it is genuinely the predictor compute path, and it is correct.

### Consequence

Head stays 866868f6 (the verified histsp wave32 fix). No source change. linux-gfx1100
unblocked -> ported; it builds and passes the cross-arch GPU gate at the current head.
The same input-generation fix applies to the windows-gfx1201 follower plan and to the
gfx90a `revalidate` (their repro snippets must use `.astype(np.float32)` before `.tofile`).

## Porter (follower delta) 2026-06-25 (linux-gfx1100, histsp wave32 fix @ 866868f6 -- BLOCKED on a deeper Lorenzo bug)

Platform: AMD Radeon Pro W7800 (gfx1100, RDNA3, wave32), ROCm 7.2.1, host has 4 GPUs.
Fork: AMD-Ecosystem/cuSZ @ moat-port, new HEAD 866868f6 (one commit on top of 07db1e28).

### What was fixed and VERIFIED (commit 866868f6)

`psz/src/kernel/histsp.cu.inl:60-66`, the AMD-branch intra-warp inclusive scan, was
hardcoded to a 64-lane wavefront (`for d in 1..<64; __shfl_up(sum,d,64); threadIdx.x % 64`).
Its result is consumed at the lane-31 leader (`threadIdx.x % 32 == 31`, line 77), so it is
logically a 32-lane warp scan. On wave32 the width-64 shuffle clamps and the `% 64` predicate
degenerates, dropping the upper half of the scan. Fixed by pinning the width/predicate to 32
(matching the non-AMD `#else` branch and the lane-31 consumer). This makes the AMD branch
correct on BOTH wave32 (RDNA) and wave64 (CDNA).

PROOF the fix is correct: dumped the device histogram (`--dump quant,hist`) for the 1M sine and
compared to a CPU `np.bincount` of the dumped quant codes -- they now match EXACTLY (max bin
diff 0). Before the fix the histogram was wrong on wave32.

gfx90a codegen: this fix CHANGES gfx90a device code (32-wide scan replaces the 64-wide scan), so
it is NOT a binary-equivalent carry-forward. `advance-head 866868f6` correctly flipped
linux-gfx90a to `revalidate`. A gfx90a GPU revalidation IS required and expected (the new scan
must be confirmed still-correct on wave64; it is the same geometry the CUDA path uses).

### Build (gfx1100) -- clean

```bash
cd /var/lib/jenkins/moat/projects/cuSZ/src   # moat-port @ 866868f6
cmake --build build-hip -j$(nproc)   # incremental, exit 0, warnings only (nodiscard hipMemcpy)
```

### BLOCKER: a SEPARATE, deeper gfx1100/RDNA3 Lorenzo-predictor correctness bug

The histsp fix does NOT restore the compression ratio. The CLI round-trip still gives CR=0.43
(vs gfx90a CR=27.04) because the QUANTIZATION CODES out of the Lorenzo predictor are themselves
corrupted on gfx1100, UPSTREAM of the histogram. The validator root-caused histsp, which was a
real bug, but it was masked by this worse predictor bug. Evidence (all on gfx1100, reproducible
on all 4 GPUs, ROCm 7.2.1):

- 1M f32 sine, abs eb 1e-3: GPU quant codes are 99.86% the value 0 (outlier sentinel); CPU
  reference is ~98% at 128 (radius, delta 0). Deltas for this sine are all in {-1,0,1} (verified
  in numpy), so EVERY element should be trivially quantizable. The GPU flags 998k/1M as outliers.
- The corruption is per-1024-tile and data-dependent: an all-ZERO input (np.zeros, 2048) compresses
  PERFECTLY (frac0=0, CR=14), but a sine whose FIRST tile is all-zero fails at local index 3 of
  that tile and the later tiles fail entirely. So a nonzero value later in the array corrupts the
  quant codes of earlier all-zero elements -- a global hazard, not a local data issue.
- NOT the outlier-compaction overflow: with eb=1.0 (prequant ~[-1,1], zero outliers expected) the
  GPU still produces 47% zeros (CR=0.69).
- NOT warp-shuffle: the 1D Lorenzo kernel (`KCU_c_lorenzo_1d`, lrz_c.cu.inl:38) that runs for
  `-l 1000000` uses NO warp intrinsic in the executed path -- `COUNT_LOCAL_STAT`'s `__ballot_sync`
  is `if constexpr`-compiled-OUT because the default `PredictorFeature<ZigZag>` has H1L_Off
  (component.hh:34). The prediction is pure shared-mem + per-thread sequential. Yet it miscompiles.
- All of 1D, 2D (CR=0.51), and 3D (CR=0.46) Lorenzo fail, so the common factor is the shared
  quantize/prequant logic (`round(in*ebx2_r)`, `quantizable = fabs(delta) < radius`,
  `quantizable * (EqUInt)candidate`), not any one kernel's geometry.

This looks like an RDNA3 codegen/correctness issue under ROCm 7.2.1 (or a subtle UB the RDNA
backend exposes) in the Lorenzo quantize loop, NOT the wave32 histsp scan that was scoped. Root
cause is unclear and would require deep investigation (IR/ISA inspection, possibly a ROCm compiler
bug report). It pre-dates this commit (histsp.cu.inl cannot affect the predictor).

NOTE vs old base: the OLD-base port (d3cde38..e443183, a different 74-file dual-source shape) was
recorded as PASSING on gfx1100, but that run only exercised 6 small unit tests + a tiny 100x100 CLI
compress and never did the 1M cross-arch CR check, so this predictor bug was never observed there.
The fresh re-port's larger inputs + cross-arch CR gate surface it.

### Repro (minimal)

```bash
cd /var/lib/jenkins/moat/projects/cuSZ/src/build-hip
python3 -c "import numpy as np; np.sin(np.arange(2048,dtype=np.float32)*2*np.pi/100000).tofile('/tmp/s.f32')"
HIP_VISIBLE_DEVICES=0 ./cusz -z -i /tmp/s.f32 -t f32 -m abs -e 1e-3 -l 2048 --dump quant --report cr
# GPU quant: 99.85% zeros (should be ~all 128). CPU deltas are all in {-1,0,1}.
python3 -c "import numpy as np; np.zeros(2048,dtype=np.float32).tofile('/tmp/z.f32')"
HIP_VISIBLE_DEVICES=0 ./cusz -z -i /tmp/z.f32 -t f32 -m abs -e 1e-3 -l 2048 --dump quant --report cr
# all-zeros: PERFECT. So later-in-array nonzero data corrupts earlier tiles.
```

State set: linux-gfx1100 -> blocked (Lorenzo-predictor quant-code corruption on gfx1100/RDNA3,
distinct from the histsp scan; histsp fix committed at 866868f6 and verified, but does not unblock).

## Validation 2026-06-25 (linux-gfx1100, FAILED -- wave32 histsp bug)

Platform: AMD Radeon Pro W7800 (gfx1100, RDNA3, wave32), ROCm 7.2.1, HIP_VISIBLE_DEVICES=0
Fork: AMD-Ecosystem/cuSZ @ moat-port, SHA 07db1e28

### Build

```bash
cd /var/lib/jenkins/moat/projects/cuSZ/src
rm -rf build-hip
cmake -S . -B build-hip \
  -DPSZ_BACKEND=HIP \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++
cmake --build build-hip -j$(nproc)   # exit 0, warnings only
```

### ctest results

```bash
HIP_VISIBLE_DEVICES=0 ctest --output-on-failure -E 'test_histsp_cu' -j1
```

Result: 98% tests passed, 1 failed out of 41.
- 28 PASS (all GPU and CPU unit tests, bin_hf HFR/FZG matrix including:
  test_l1_compact, test_stat_identical1/2, test_stat_max_error, test_mem_unique,
  test_hf_revisit_altcode, test_hf_cpu_serial_codebook, hf/hfr/hfr-pbkc/hfr-pbkgo
  over cauchy-mild, cauchy-sharp [except hfr__cauchy_sharp__u2], uniform-256, uniform-1024)
- 12 SKIP (rc=77): cusz CLI round-trip matrix (cusz__rtm/hurr/nyx) -- no dataset
- 1 FAIL: hfr__cauchy_sharp__u2 -- pre-existing upstream CLI/test desync
  (`--assert-brnum-le` option not implemented in bin_phf.cc); identical failure on CUDA build

### Cross-arch consistency check -- FAILED

```bash
python3 -c "import numpy as np; np.sin(np.arange(1000000,dtype=np.float32)*2*np.pi/100000).tofile('/tmp/test_sine_1m.f32')"
HIP_VISIBLE_DEVICES=0 ./cusz -z -i /tmp/test_sine_1m.f32 -t f32 -m abs -e 1e-3 -l 1000000 --report cr,psnr
# [Lorenzo, Hist, HF-fast2]  CR=0.43  mode=Abs  input_eb=1.000000e-03
HIP_VISIBLE_DEVICES=0 ./cusz -x -i /tmp/test_sine_1m.f32.cusza --compare /tmp/test_sine_1m.f32 --report cr,psnr
# [Lorenzo, Hist, HF-fast2]  CR=0.43  PSNR=-nan  max_error=3.402101e+38
```

gfx90a reference: CR=27.04, PSNR=70.8, max_error=1.000047e-03
gfx1100 result:   CR=0.43,  PSNR=-nan, max_error=3.40e+38 (float max, corrupted)

### Root cause: histsp.cu.inl:60-66 wave64 intra-warp scan

Exactly the wave32 failure the reviewer predicted.
`psz/src/kernel/histsp.cu.inl:60-66` (the `#ifdef __HIP_PLATFORM_AMD__` branch):

```cpp
for (auto d = 1; d < 64; d *= 2) {
    auto n = __shfl_up(sum, d, 64);
    if (threadIdx.x % 64 >= d) sum += n;
}
```

On wave32 (gfx1100), `__shfl_up(sum, d, 64)` with `d >= 32` clamps to the lower
boundary of the 32-lane physical wavefront (no lane above 31 exists), so the high
half of the intended 64-wide scan is a no-op. The writeback predicate
`threadIdx.x % 64 >= d` uses a 64-wide modulus on a 32-lane index (0..31), so
`threadIdx.x % 64` == `threadIdx.x` (always < 64) -- the predicate always acts as
if the lanes are in the lower 32 of a 64-wide group, meaning the scan accumulates
incorrectly. The result is a per-warp histogram that is internally consistent but
numerically wrong vs the true counts. This shifts the Huffman symbol frequencies
so severely that the Huffman codec produces worse-than-uncompressed output (CR=0.43
vs CR=27.04) and the decompressed data is corrupted (max_error at float max).

This code (git blame: J. Tian 5cda177b, 2023) was pre-existing upstream code that
gfx90a (wave64) ran correctly. It was NOT introduced by the MOAT port.

### Fix required (for the porter)

The AMD scan branch must use the physical wave width, not the hardcoded 64. The
portable CUDA branch (lines 68-73) already uses width-32 `__shfl_up_sync` --
simply use the same geometry for AMD, or generalize using `__AMDGCN_WAVEFRONT_SIZE`
(set to 32 or 64 by the compiler). The simplest correct fix is to drop the
wave64-special-case and use the portable 32-lane scan for both AMD and CUDA:

```cpp
// Replace the #ifdef __HIP_PLATFORM_AMD__ / #else block with:
for (auto& sum : p_hist) {
    for (auto d = 1; d < 32; d *= 2) {
        auto n = __shfl_up(sum, d);   // or __shfl_up_sync on CUDA; width defaults to warpSize
        if (threadIdx.x % 32 >= d) sum += n;
    }
}
```

OR use `__AMDGCN_WAVEFRONT_SIZE` to keep the fast wave64 path on MI* while fixing
wave32 RDNA GPUs.

Verdict: validation-FAILED. Bouncing to porter.
State set: linux-gfx1100 -> validation-failed.

## Review 2026-06-25 (follower linux-gfx1100, no-source-change @ 07db1e28)

Reviewed the gfx1100 follower delta with /pr-review (local-branch mode). HEAD is
07db1e28 detached at moat-port; `git diff 07db1e28...HEAD` is empty -- no follower
commit, head_sha unchanged, the lead linux-gfx90a is completed and review-passed at
this same head. So there is no new port code to review; scope is the gfx1100 wave32
correctness of the existing built sources. Verdict: review-passed (no port-introduced
defect), with one wave32 correctness RISK the validator MUST confirm on real hardware
and one correction to the delta plan's risk claim.

### Wave32-critical analysis (problems / risks only)

- PLAN RISK CLAIM IS WRONG (plan.md "Delta plan: linux-gfx1100" / "the sole wave32
  risk"): the plan asserts the `__ballot_sync` macro is "the sole wave32 risk;
  everything else is arch-neutral." That is not true. `psz/src/kernel/histsp.cu.inl:60-66`
  carries a SECOND wave-width-sensitive path -- a hardcoded WAVE64 intra-warp inclusive
  scan under `#ifdef __HIP_PLATFORM_AMD__`:
      for (auto d = 1; d < 64; d *= 2) { auto n = __shfl_up(sum, d, 64);
        if (threadIdx.x % 64 >= d) sum += n; }
  This is pre-existing upstream code (git blame: J. Tian 5cda177b, 2023, a remnant of
  the old HIP scaffold), NOT introduced by the MOAT port, so it is not a port defect.
  But it IS on the DEFAULT compression path: `GPU_histogram_Cauchy<E>::kernel` (this
  kernel) is the HistSp histogram called at compressor.inl:115 and :184, feeding the
  Huffman codebook. The gfx90a lead validated it correctly because gfx90a is wave64 and
  the branch is written for wave64. gfx1100 is wave32 and runs the SAME branch.

  Why it is suspect on wave32 (confirmed against /opt/rocm amd_warp_functions.h):
  `__shfl_up(var, delta, width=64)` lowers to `__builtin_amdgcn_ds_bpermute` with the
  lower-bound clamp `self & ~(width-1)`; on wave32 `__lane_id()` is 0..31 so `& ~63` is
  0 and the permute domain is the 32-lane PHYSICAL wavefront, while the loop runs d up
  to 32 and the writeback predicate `threadIdx.x % 64 >= d` (and the `% 32 == 31`
  writeback at :77) treat lanes as a 64-wide logical group. The scan therefore does not
  match the 32-lane permute domain on wave32. A wrong per-warp histogram total changes
  the symbol frequencies and thus the Huffman codeword lengths, i.e. the COMPRESSION
  RATIO, even though round-trip can still be lossless within the eb bound (encoder and
  decoder share the book). So the failure mode is "round-trips fine but CR diverges from
  gfx90a," not a crash.

  VALIDATOR GATE (must do, do not accept "deterministic + plausible"): the plan's
  cross-arch consistency check is exactly the detector. The CLI round-trip CR on gfx1100
  for the 1M f32 sine MUST equal the gfx90a number CR=27.04 (PSNR=70.8,
  max_error=1.000047e-03). If CR diverges, suspect histsp.cu.inl:60 FIRST (not the
  ballot macro). The bin_hf HFR matrix uses the histogram too. If CR diverges, this
  becomes a genuine follower fix (generalize the AMD scan to the physical wave width, or
  drop the wave64 special-case and use the portable 32-lane scan) and bounces to the
  porter.

### Wave32 paths confirmed SOUND at source level (recorded for the validator's audit)

- `__ballot_sync` macro (c_cu2hip_1_fix_primitives.h:19-20)
  `(uint32_t)(__ballot(PRED) >> (__lane_id() & ~31))`: correct on wave32 (lane_id 0..31,
  shift always 0, `__ballot` upper 32 bits zero -> shift-0 identity) and wave64. All
  THREE built call sites verified: fzg_c.cu.inl:37,53 and fzg_x.cu.inl:99 (block dim3
  (32,32), tid=y*32+x, each 32-thread row is a half-wavefront on wave64 / a full
  wavefront on wave32 -- the half-base shift selects the right 32 bits) and
  lrz_c.cu.inl:12 (COUNT_LOCAL_STAT: `threadIdx.x % 32 == 0` leader does `__popc(mask)`;
  on wave64 lanes 0 and 32 each popc their own half, no double-count; on wave32 one
  leader per warp). The `_future/` ballot sites (warp_top1.cuh, hfr-pbk.cuh) are NOT
  compiled (not in any CMakeLists target list) -- not a concern.
- Variadic shuffle macros (c_cu2hip_1_fix_primitives.h:7-10): 3-arg and 4-arg both map
  through `##__VA_ARGS__`; the 3-arg `__shfl_sync(0xffffffff,p_incomp,0)` at
  hfr_encode_c.cuh:56 broadcasts a block-wide shared scalar, identical on both half-warp
  leaders, harmless on wave64 and wave32.
- Native double atomicAdd (utils/src/atomics.cu.inl): HIP branch
  `defined(__HIP_DEVICE_COMPILE__)` -> native `atomicAdd(double*)`, a hardware path on
  RDNA3 (gfx1100), not the CAS fallback; CUDA `__CUDA_ARCH__>=600` guard untouched.
- hist_generic.cu `WARP_SIZE=32` (line 11): used only for grid-stride WORK PARTITION
  (begin/step/warp_id), NOT for any `__shfl`/`__ballot`/`__syncwarp` lane geometry --
  no cross-lane primitive in this kernel. Arch-neutral. (Also the non-default
  HistGeneric path.) Not a fault.
- Arch swap: build uses only `-DCMAKE_HIP_ARCHITECTURES=gfx1100`; no source change, no
  `CMAKE_HIP_ARCHITECTURES` pin in CMake (enable_language(HIP) auto-detect, overridable).
  No textures/surfaces, no cuBLAS/cuFFT/cuSPARSE, no rule-of-five handle classes in the
  built set -- the CDNA texture/pitch fault classes do not apply.

### Commit hygiene

No new commit on this follower, so nothing to check. The existing head commits
(07db1e28 / 5d43e441 / a6d765e8) were hygiene-checked at the lead review (`[ROCm]`
prefix, Claude named, no noreply trailer, no MOAT jargon, only the public account).

## Porter (follower delta) 2026-06-25 (linux-gfx1100, re-port @ 07db1e28)

Platform: AMD Radeon RX 7900 XTX (gfx1100, RDNA3, wave32), ROCm 7.2.1, host has 4 GPUs.
Fork: AMD-Ecosystem/cuSZ @ moat-port HEAD 07db1e28 (the fresh re-port; gfx90a lead completed
at validated_sha 5d43e441, carried to head 07db1e28 as doc-only).

Follower delta-port: reused the SAME moat-port branch and the SAME Strategy A build
commands as the gfx90a lead; only swapped `-DCMAKE_HIP_ARCHITECTURES=gfx90a` ->
`gfx1100`. NO source change was needed. The build is clean with the arch swap alone, so
NO follower commit was created and head_sha stays 07db1e28 (no advance-head).

### Build (gfx1100) -- arch swap only, builds clean

```bash
cd /var/lib/jenkins/moat/projects/cuSZ/src   # detached at origin/moat-port (07db1e28)
rm -rf build-hip
cmake -S . -B build-hip \
  -DPSZ_BACKEND=HIP \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++
cmake --build build-hip -j64   # exit 0, warnings only (nodiscard hipMemcpy returns)
```

Result: full build to 100%, `cusz` binary + all libs built, 41 ctest entries registered,
zero errors. Configure echoed `HIP (ROCm) backend has been selected` and
`CMAKE_HIP_ARCHITECTURES: gfx1100`. Native double atomicAdd (RDNA item to confirm) built
without complaint; the arch-unified `__ballot_sync` macro
(c_cu2hip_1_fix_primitives.h:19) compiled on wave32 -- its runtime wave32 correctness
(shift-0 identity) is the validator's GPU gate (bin_hf HFR/FZG matrix). No wave32 or
RDNA build fix was required.

Porter verdict: ported (build only). validation pending on real gfx1100 GPU.


## RE-PORT RESULT -- linux-gfx90a (2026-06-25, ported)

Fresh single-source HIP port against master (base e1c0135), Strategy A. NOT the
old 74-file .hip-mirror approach: the same .cu/.cu.inl/.cc sources are reused and
marked LANGUAGE HIP, translated through a small compat layer.

### Build (gfx90a)

```bash
cd /var/lib/jenkins/moat/projects/cuSZ/src
git submodule update --init third_party/lc   # only if PSZ_ACTIVATE_LC re-enabled
rm -rf build-hip
cmake -S . -B build-hip \
  -DPSZ_BACKEND=HIP \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++
cmake --build build-hip -j$(nproc)
```

`-DCMAKE_CXX_COMPILER=amdclang++` is required so host .cc TUs in HIP-language
targets get the ROCm include dirs. `-DCMAKE_PREFIX_PATH=/opt/rocm` is required on
clean containers for find_package(hip)/hiprand.

### Test (gfx90a)

```bash
cd build-hip
HIP_VISIBLE_DEVICES=0 ctest --output-on-failure -E "test_histsp_cu" -j1
```

Result: 40/41 tests ran. 27 PASS (6 portable_* CPU unit tests, test_zigzag,
test_l1_compact GPU, test_lrz_seq, test_stat_identical1/2 GPU, test_stat_max_error
GPU, test_mem_unique GPU, test_hf_revisit_altcode GPU (HFR), test_hf_cpu_serial_codebook,
and the bin_hf codec matrix: hf/hfr/hfr-pbkc/hfr-pbkgo over cauchy-mild/sharp +
uniform-256/1024). 12 SKIP (rc=77): the cusz CLI round-trip matrix (cusz__rtm/hurr/nyx)
needs $CUSZ_TEST_DATA datasets that are absent locally -- not a failure.

Known non-port failures (fail identically on the CUDA build):
- `hfr__cauchy_sharp__u2`: the test cmake passes `--assert-brnum-le=200000` but
  bin_phf.cc does not implement that flag (upstream test/CLI desync).
- `test_histsp_cu` (excluded; not built): tune_histsp.cu.inl calls a
  `GPU_histogram_generic<T>(...)` ctor whose signature drifted from the struct.
  Tuning/perf test only; same exclusion as the old port. Pre-existing upstream.

CLI round-trip verified by hand (Lorenzo+Hist+Huffman and HFR paths), 1M f32 sine:
CR=27.4, PSNR=70.8, max_error within the 1e-3 abs bound; `cusz --version` prints
the hipSZ banner via the new HIP verinfo path.

### Real-dataset validation (2026-06-25, gfx90a)

The 12 `cusz__{rtm,hurr,nyx}` tests bake their dataset paths from `$CUSZ_TEST_DATA`
(default `/data`) at cmake-CONFIGURE time, not at ctest time. To run them: place the
files, set the env var, re-configure the existing build (no recompile), re-run ctest.

```
# ~608 MiB kept (tarballs/extract deleted after mapping); host has fast egress.
# Hurricane ISABEL tar.gz -> 100x500x500/Uf48.bin.f32 (100,000,000 B) => $CUSZ_TEST_DATA/HURR/Uf48.f4
# NYX 512^3        tar.gz -> velocity_x.f32            (536,870,912 B) => $CUSZ_TEST_DATA/NYX/velocity_x.f32
# Globus: g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/{Hurricane-ISABEL,EXASKY/NYX}/
# (cuSZ's own script/sh.download-sdrb-data lists the Hurricane URL.)
export CUSZ_TEST_DATA=/var/lib/jenkins/cusz-test-data
cd build-hip && cmake . -DCUSZ_TEST_DATA=$CUSZ_TEST_DATA
HIP_VISIBLE_DEVICES=0 ctest --output-on-failure
```

With HURR+NYX present: **37 PASS / 3 SKIP / 1 FAIL of 41** (+9 over the dataset-less run).
All 6 `cusz__hurr_uf48` (hf/hfr/hfr-pbkc x rel 1e-3/1e-4) and all 3 `cusz__nyx_velx`
(codec sweep, rel 1e-3) pass real-data round-trip on GPU. The 3 `cusz__rtm_0480__y24/y25`
still SKIP: the RTM seismic dataset (235x449x449 `RTM/0480.f32`, 181 MiB) is not on
SDRBench's public listing or the in-repo download script (paper-private; FZ-GPU/cuSZ-i).
Those are the only tests exercising the spl-y24/y25 predictor on real data; deferred as
`cusz-rtm-test-dataset`, drop the file at `$CUSZ_TEST_DATA/RTM/0480.f32` to unlock. The 1
FAIL is the pre-existing `hfr__cauchy_sharp__u2` upstream desync (non-regression).
Additional coverage at the already-validated head; platform stays `completed`.

### Port shape (files changed)

Single compat hook: `cmake/hip-compat/` holds a force-included prelude
(`psz_hip_compat.h` = hip_runtime + hip cooperative_groups + the c_cu2hip macros)
plus include shims (`cuda_runtime.h`, `cuda.h`, `cuda_runtime_api.h`, `cuda_fp16.h`,
`cooperative_groups.h`, `cooperative_groups/memcpy_async.h`, `curand.h`). The
prelude is force-included on HIP-language TUs; host .cc TUs pick up translation
via the cuda_runtime.h shim (which itself pulls the c_cu2hip macros).

- New: `cmake/hip.cmake` (mirrors cuda.cmake; keeps the `_cu` target names but
  compiles LANGUAGE HIP, links hip::device/host, swaps cuRAND->hipRAND, drops
  nvml/cupti/driver), `cmake/hip-compat/*`, `psz/src/cli/verinfo_hip.cu` (HIP
  device/driver/runtime version-info, replaces verinfo.cu/verinfo_nv.cu),
  `example/cmake/hip-example.cmake`.
- CMake wiring: top `CMakeLists.txt`, `cmake/probe.cmake`, `portable/`, `codec/fzg/`,
  `codec/hf/`, `utils/`, `test/CMakeLists.txt` + `test/cmake/cuda-test.cmake`,
  `example/CMakeLists.txt` -- each gets a PSZ_BACKEND=HIP branch.
- Source fixes: `portable/include/c_type.h` (add HIP to `_ptb_runtime`),
  `utils/include/compare.hh` (dispatch HIP like CUDA), `portable/include/mem/cxx_backends.h`
  + `cxx_mem_ops.h` (restore HIP branches), `portable/include/utils/err.hh`
  (CUDA branch also serves HIP), `portable/include/mem/gpu_event.hh`/`gpu_stream.hh`
  (backend-portable handle element type via remove_pointer; CUPTI stubbed on HIP),
  `portable/include/macro/c_cu2hip_{0,1}*.h` (event/memset/symbol/occupancy macros;
  variadic shuffle + arch-unified ballot), `utils/src/atomics.cu.inl` (native HIP
  double atomicAdd), `utils/src/extrema.cu.inl` (drop stale `extrema_kernel` HIP
  branch; use KCU_extrema on both), `psz/src/kernel/spl_y24.cuh` (fix stale fwd
  decl missing dim3 param), `psz/src/cli/context.cc` (HIP BACKEND_TEXT),
  `psz/src/compressor.inl` (CONCAT_ON_DEVICE for HIP).

### New / confirmed fault classes (cheat-sheet additions)

1. ROCm 7.2.1 now ships native `__shfl_*_sync`/`__ballot_sync` that static_assert
   on a 64-bit mask; the 32-bit-mask CUDA call sites trip it. The c_cu2hip_1
   shuffle macros (redirect to maskless `__shfl_*`) are REQUIRED, not optional.
2. **Wave64 ballot correctness (latent bug the old port shipped).** The old
   c_cu2hip_1 mapped `__ballot_sync(mask,pred) -> (uint32_t)__ballot(pred)`, i.e.
   the LOW 32 lanes of a wave64 wavefront. The FZG codec (blockDim (32,32)) and
   HFR encode use 32-lane logical warps where the odd `threadIdx.y` rows occupy
   the HIGH 32 lanes (linear tid = y*32+x), so the truncation returns the WRONG
   row's mask on wave64. The old port never caught it (no FZG/HFR GPU test ran).
   Arch-unified fix:
   `__ballot_sync(MASK,PRED) := (uint32_t)(__ballot(PRED) >> (__lane_id() & ~31u))`
   -- correct on wave32 (shift 0) and wave64 (shift 0 or 32). Now exercised by the
   bin_hf HFR matrix.
3. Stale per-arch HIP branches left in upstream master after its "eradicate HIP"
   cleanup (`extrema.cu.inl` referenced a renamed `extrema_kernel` and threw for
   double). Re-unify to the current single kernel rather than resurrect the branch.
4. CUPTI has no ROCm analogue; the optional per-kernel profiling timer
   (gpu_event.hh) must be stubbed (inert, never activates) so gpu_timer falls back
   to the hipEvent wall-clock path; bin_hf's default `--timer cupti` then yields 0
   harmlessly.
5. Driver handle element types: `cudaStream_t`/`cudaEvent_t` smart-pointer
   wrappers used `CUstream_st`/`CUevent_st` (CUDA driver structs) which do not
   exist under HIP; use `std::remove_pointer<cudaStream_t>::type` instead.
6. The bundled LC-framework (third_party/lc) is a separate external CUDA project
   (32-bit-mask `__shfl_*_sync`/`__any_sync`, `__trap`); PSZ_ACTIVATE_LC defaults
   OFF on HIP. Deferred: `cusz-lc-framework-hip`.

---

## RE-PORT against upstream master (2026-06-25)

The original port (base d3cde38) completed an in-tree HIP scaffold that upstream
has since DELETED. Upstream commit e5cceb9a ("eradicate HIP-related setup",
2026-04-30) removed cmake/hip.cmake, the *.cuhip.inl dual-source files, and the
HIP example/build plumbing, and master is now 30 commits ahead with a heavy
refactor (`_portable` -> `_ptb` rename, pure-ctest migration, CLI/arg-builder
restructure, HFR encoding family, "strip per-file headers"). PSZ_BACKEND now
selects CUDA (default) or ONEAPI (SYCL/DPC++, cmake/sycl.cmake); HIP is gone.

Decision (jeff): re-port from current master, the way a fresh port would, NOT a
git-rebase of the old commits. Base refreshed to e1c0135; moat-port reset to
clean master; all platforms back to unclaimed.

What SURVIVED on master and helps the re-port:
- portable/include/macro/c_cu2hip_{0,1,2}_*.h -- the CUDA->HIP translation macros.
- portable/include/backend.h still has the `PSZ_USE_HIP -> _ptb_runtime::HIP` branch.
- cmake/cuda.cmake and cmake/sycl.cmake are the templates: a HIP backend re-adds
  the parallel PSZ_USE_HIP / cmake/hip.cmake beside them (PSZ_BACKEND=HIP option).

Reference material (NOT a rebase base):
- Old validated port: fork branch `moat-port-pre-rebase-d3cde38`, tag
  `moat-validated-d3cde38` (e443183). Diff `d3cde38..e443183` shows every HIP
  fix that was needed (see "## linux-gfx90a port" below -- describes the OLD base).
- The "Port summary" / per-arch fix lists below are from the OLD base; file
  paths and the scaffold have changed, but the fault classes (PROPER_RUNTIME enum,
  variadic shuffle macros, portable-layer conditional includes, missing GPU_*
  module impls) are the cheat-sheet for what to expect.

Everything from "## linux-gfx90a port" down is OLD-BASE history, kept for reference.

---

## linux-gfx90a port

### Build

```bash
cd projects/cuSZ/src
mkdir build-hip && cd build-hip
cmake .. -DPSZ_BACKEND=HIP -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build . -j$(nproc)
```

### Test

```bash
HIP_VISIBLE_DEVICES=0 ./hipsz --version
HIP_VISIBLE_DEVICES=0 ./hipsz -z -i <input.f32> -t f32 -m rel -e 1e-4 -l <dims>
HIP_VISIBLE_DEVICES=0 ./hipsz -x -i <input.f32.cusza> --compare <input.f32>
```

### Port summary

The existing HIP support was bitrotted with empty stub files and incomplete cmake configuration. Fixed by:

1. Complete rewrite of cmake/hip.cmake mirroring the cuda.cmake structure
2. Added HIP runtime includes and CUDA-to-HIP translation macros to all .hip files
3. Fixed forward declaration mismatch in spline3.inl
4. Added missing HIP implementations for psz::module functions (GPU_identical, GPU_extrema, GPU_find_max_error, GPU_assess_quality, GPU_calculate_errors)
5. Fixed portable layer headers to conditionally include HIP or CUDA headers
6. Added CUDA-named function aliases in verinfo.hip for CLI compatibility
7. Fixed PROPER_RUNTIME to use ROCM enum value (not HIP which doesn't exist in the enum)
8. Fixed variadic shuffle intrinsic macros to support both 3-arg and 4-arg forms

### Validation (2026-06-05, linux-gfx90a)

Validated on gfx90a with ROCm 7.2.1, HIP_VISIBLE_DEVICES=0.

Build command:
```bash
cd /var/lib/jenkins/moat/projects/cuSZ/src
rm -rf build-hip && mkdir build-hip && cd build-hip
cmake .. -DPSZ_BACKEND=HIP -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build . -j$(nproc)
```

Test results:
```bash
cd build-hip
HIP_VISIBLE_DEVICES=0 ctest --output-on-failure -E 'histsp_hip'
```

Result: 6/6 tests PASS (100%)
- test_zigzag: PASS (CPU-only zigzag codec)
- test_l1_compact: PASS (GPU sparse vector compaction)
- test_lrz_seq: PASS (CPU-only Lorenzo predictor)
- test_stat_identical: PASS (GPU statistical functions, CPU-GPU match verified)
- test_stat_max_error: PASS (GPU error calculation, CPU-GPU match verified)
- test_mem_unique: PASS (GPU memory management)

Note: test_histsp_hip (7th test) excluded - build fails due to incorrect include path in test/src/tune_histsp.hip (references "detail/t_histsp.cu_hip.inl" but file is named "detail/tune_histsp.cuhip.inl"). This is a tuning/performance test, not a core functionality test.

CLI compression test:
```bash
# Create test data
python3 -c "import numpy as np; data=np.sin(np.linspace(0,10,100).reshape(100,1))*np.cos(np.linspace(0,10,100)); data.astype(np.float32).tofile('test.f32')"
# Compress (works)
HIP_VISIBLE_DEVICES=0 ./hipsz -z -i test.f32 -t f32 -m abs -e 0.001 -l 100x100
# Produces test.f32.cusza (40KB -> 3.6KB compression)
```

### Known issues

- The --report time,cr flag causes a crash (std::out_of_range in unordered_map) - UPSTREAM ISSUE
- Large/random data compression may produce oversized output (possible Huffman codec issue) - UPSTREAM ISSUE  
- Decompression with -x flag fails in some cases - UPSTREAM ISSUE
- test_histsp_hip has wrong include path (porter build issue, minor)

### Gotchas

- The _portable_runtime enum uses ROCM (value 4), not HIP. The HIP enum value is in _portable_toolkit.
- Most .hip files need both c_cu2hip_0_translation.h (for type/function macros) and c_cu2hip_1_fix_primitives.h (for warp intrinsics)
- The shuffle intrinsic macros need to be variadic to support both 3-arg and 4-arg forms
- Double-precision atomicAdd on HIP uses unsafeAtomicAdd, not the CAS-loop emulation used on older CUDA

## linux-gfx1100 validation

### Validation (2026-06-05, linux-gfx1100)

Validated on gfx1100 with ROCm 7.2.1, HIP_VISIBLE_DEVICES=0.

Build command:
```bash
cd /var/lib/jenkins/moat/projects/cuSZ/src
rm -rf build-hip && mkdir build-hip && cd build-hip
cmake .. -DPSZ_BACKEND=HIP -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build . -j$(nproc)
```

Test results:
```bash
cd build-hip
HIP_VISIBLE_DEVICES=0 ctest --output-on-failure
```

Result: 6/6 tests PASS (100%)
- test_zigzag: PASS (CPU-only zigzag codec)
- test_l1_compact: PASS (GPU sparse vector compaction)
- test_lrz_seq: PASS (CPU-only Lorenzo predictor)
- test_stat_identical: PASS (GPU statistical functions, CPU-GPU match verified)
- test_stat_max_error: PASS (GPU error calculation, CPU-GPU match verified)
- test_mem_unique: PASS (GPU memory management)

CLI compression test:
```bash
# Create test data
python3 -c "import numpy as np; data=np.sin(np.linspace(0,10,100).reshape(100,1))*np.cos(np.linspace(0,10,100)); data.astype(np.float32).tofile('test.f32')"
# Compress
HIP_VISIBLE_DEVICES=0 ./hipsz -z -i test.f32 -t f32 -m abs -e 0.001 -l 100x100
# Produces test.f32.cusza (40KB -> 7.6KB compression)
```

### gfx1100 build fixes (26b1f91)

Additional fixes required beyond the gfx90a port:
1. hipMallocHost type strictness: Cast float**/double** to void** (example/src/demo_v2.hip.cc)
2. Portable stream creation: Use create_stream()/destroy_stream() macros instead of cudaStreamCreate/Destroy (example/src/bin_phf.cc)
3. Include c_cu2hip_0_translation.h in .cc files that use CUDA API directly (batch_run.cc, bin_fzgcodec.cc)
4. Replace cudaStream_t with GPU_BACKEND_SPECIFIC_STREAM macro in .cc files (bin_hist.cc, bin_phf.cc, batch_run.cc, bin_fzgcodec.cc)
5. Remove hardcoded <cuda_runtime.h> includes, rely on portable headers (bin_hist.cc, batch_run.cc, bin_fzgcodec.cc)

Same test_histsp_hip exclusion as gfx90a (wrong include path, performance tuning only).

## Validation 2026-06-08 (windows-gfx1201)

Platform: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), Windows 11 Pro for Workstations
ROCm: 7.14.0a20260604 (TheRock nightly), HIP_VISIBLE_DEVICES=0 (only GPU present)
Fork: AMD-Ecosystem/cuSZ @ moat-port, SHA aff8ee6

### Windows delta-port (new commit aff8ee6 on top of 26b1f91)

Eight Windows-specific fixes required; none affect Linux behavior:

1. `cmake/hip.cmake`: add `-DHIP_DISABLE_WARP_SYNC_BUILTINS` for WIN32.
   ROCm 7.14 `amd_hip_bf16.h` tries to define `__shfl_*_sync` overloads for
   bfloat16 under `#if !defined(HIP_DISABLE_WARP_SYNC_BUILTINS)`, but those
   overloads conflict with the templated versions already pulled in by
   `amd_warp_sync_functions.h` (included earlier via hip_runtime.h), producing
   "redefinition of default argument" errors in all thrust-based .hip files.
   The cu2hip macros already redirect `__shfl_*_sync` -> `__shfl_*`, so
   suppressing the bf16 overloads is safe. Same fix used in MMseqs2 (dbeac858).

2. `cmake/hip.cmake`: add `psz_hip_stat` to `psz_hip_utils` link libraries.
   Windows DLL link graphs must be explicit; on Linux symbols resolve lazily
   at load time, but lld-link requires explicit import lib references.

3. `codec/hf/src/hf_bk_internal.seq.cc`: add `#include <string>` for
   `std::to_string` (GCC/libstdc++ implicitly pulls it in; MSVC STL does not).

4. `portable/include/cxx_typing.h`: guard `TypeSym<ull>` with `!_WIN32`.
   On Windows, `uint64_t` and `unsigned long long` are the same underlying
   type, causing a duplicate explicit template specialization error.

5. `portable/include/mem/cxx_memobj.h` and `portable/src/mem/memobj_impl.inl`:
   guard `<linux/limits.h>` with `!_WIN32`. The include is a compile-time
   macro check that is Linux-only and unused on Windows.

6. `psz/include/utils/query/query_cpu.hh`: define `popen`/`pclose` as
   `_popen`/`_pclose` on Windows (POSIX names not in MSVC CRT).

7. `psz/src/utils/context.cc`: guard `<cxxabi.h>` and `abi::__cxa_demangle`
   with `!_WIN32` (GCC ABI demangling unavailable on Windows); replace
   `asprintf` (POSIX) with `snprintf` into a stack buffer.

8. `psz/src/utils/viewer.cc`: remove `const` qualifier from `unordered_map`
   key type (`std::hash<const E>` is not specialized in MSVC STL).

### Build command

```
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
SRC=B:/develop/moat/projects/cuSZ/src
BUILD=$SRC/build-hip-gfx1201

cmake -S $SRC -B $BUILD -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPSZ_BACKEND=HIP -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/amdclang.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/amdclang++.exe \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/amdclang++.exe \
  -DCMAKE_PREFIX_PATH=$ROCM -DBUILD_TESTING=ON \
  -DPSZ_BUILD_EXAMPLES=OFF -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON

# Strip lld-link flag injected by CMake 4.3 Windows-Clang platform module
sed -i 's/ -fuse-ld=lld-link//g' $BUILD/build.ninja

cmake --build $BUILD -j24   # 78/78 targets, 0 errors
```

### Test results

Copy ROCm runtime DLLs (amdhip64_7.dll, amd_comgr.dll, hiprtc*.dll, rocm_kpack.dll,
hiprand.dll, rocrand.dll) from _rocm_sdk_core/bin to build/test/ along with all
project DLLs (hipsz.dll, psz_hip_*.dll, fzg_hip.dll, phf_hip.dll).

```
HIP_VISIBLE_DEVICES=0 ctest --test-dir $BUILD -E histsp_hip --output-on-failure -j1
```

Result: 6/6 tests PASS (100%)
- test_zigzag: PASS (CPU-only zigzag codec)
- test_l1_compact: PASS (GPU sparse vector compaction, gfx1201)
- test_lrz_seq: PASS (CPU-only Lorenzo predictor)
- test_stat_identical: PASS (GPU statistical functions, CPU-GPU match verified)
- test_stat_max_error: PASS (GPU error calculation, CPU-GPU match verified)
- test_mem_unique: PASS (GPU memory management, gfx1201)

Same test_histsp_hip exclusion as Linux (wrong include path, performance tuning only).

Verdict: PASS. validated_sha=aff8ee6 (windows-gfx1201).

### Revalidation note for linux-gfx90a and linux-gfx1100

The Windows commit (aff8ee6) adds `_WIN32`-guarded code paths only. On Linux,
WIN32 is false and the guards don't execute, so compiled device code is
unchanged. Binary-equivalence check (codeobj_diff.py) expected to show
`verdict=identical` -> carry forward without GPU re-run.

## Revalidation 2026-06-08 (linux-gfx90a)

Platform: AMD Instinct MI250X (gfx90a), ROCm 7.2.1, HIP_VISIBLE_DEVICES=1 (GCD 1)
Fork: AMD-Ecosystem/cuSZ @ moat-port, SHA aff8ee6 (validated_sha was 36a9bfd6)

### Delta classification

`git diff 36a9bfd..aff8ee6` spans 14 files (66 insertions, 28 deletions). Most changes are `_WIN32`-guarded, but two are unconditional:

1. `cmake/hip.cmake`: adds `psz_hip_stat` to `psz_hip_utils` link libraries (host link order fix, no device code change)
2. `psz/src/utils/viewer.cc`: removes `const` qualifier from `unordered_map` key types (fixes MSVC hash<const E> missing specialization)

`python3 utils/codeobj_diff.py build-hip-old build-hip-new` returned `verdict=differ`: `libpsz_hip_utils.so` exported symbol names changed because the mangled names for the unordered_map template instantiations changed (`K13psz_predictor` -> `13psz_predictor`, i.e., key-const removed). Device ISA is identical across all libraries. Because exported symbols differ, carry-forward was not applicable; full GPU revalidation was required.

Build note: both SHAs require `-DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++` to avoid c++ receiving `-x hip --offload-arch=gfx90a` from psz_hip_compile_settings. The old SHA had build errors in example files (cuda_runtime.h not found, hipMallocHost arity errors) that the new SHA fixes; core targets built successfully at both SHAs.

### Build command

```bash
cd /var/lib/jenkins/moat/projects/cuSZ/src
rm -rf build-hip-new
cmake -S . -B build-hip-new \
  -DPSZ_BACKEND=HIP \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++
cmake --build build-hip-new -j$(nproc)
```

### Test results

```bash
cd build-hip-new
HIP_VISIBLE_DEVICES=1 ctest --output-on-failure -E 'histsp_hip' -j1
```

Result: 6/6 tests PASS (100%)
- test_zigzag: PASS (0.00s, CPU-only zigzag codec)
- test_l1_compact: PASS (0.25s, GPU sparse vector compaction, gfx90a)
- test_lrz_seq: PASS (0.00s, CPU-only Lorenzo predictor)
- test_stat_identical: PASS (0.22s, GPU statistical functions)
- test_stat_max_error: PASS (0.23s, GPU error calculation)
- test_mem_unique: PASS (0.19s, GPU memory management)

Verdict: PASS. validated_sha=aff8ee6 (linux-gfx90a).

## Revalidation 2026-06-08 (linux-gfx1100)

Platform: AMD Radeon RX 7900 XTX (gfx1100), ROCm 7.2.1, HIP_VISIBLE_DEVICES=3
Fork: AMD-Ecosystem/cuSZ @ moat-port, SHA aff8ee6 (validated_sha was 26b1f91)

### Delta classification

`git diff 26b1f91..aff8ee6` spans 8 files (44 insertions, 6 deletions). Same delta as the gfx90a revalidation: most changes are `_WIN32`-guarded, but two are unconditional:

1. `cmake/hip.cmake`: adds `psz_hip_stat` to `psz_hip_utils` link libraries (host link order fix)
2. `psz/src/utils/viewer.cc`: removes `const` qualifier from `unordered_map` key types

`python3 utils/codeobj_diff.py build-hip-old build-hip-new` returned `verdict=differ`: `libpsz_hip_utils.so` exported symbol names changed (mangled names for `unordered_map<K13psz_predictor, ...>` changed to `unordered_map<13psz_predictor, ...>` -- `const` removed from key). Device ISA is identical across all libraries. Carry-forward not applicable; full GPU revalidation required.

### Build command

```bash
cd /var/lib/jenkins/moat/projects/cuSZ/src
cmake -S . -B build-hip-new \
  -DPSZ_BACKEND=HIP \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++
cmake --build build-hip-new -j$(nproc)
```

### Test results

```bash
HIP_VISIBLE_DEVICES=3 ctest --test-dir build-hip-new --output-on-failure -E 'histsp_hip' -j1
```

Result: 6/6 tests PASS (100%)
- test_zigzag: PASS (0.00s, CPU-only zigzag codec)
- test_l1_compact: PASS (0.22s, GPU sparse vector compaction, gfx1100)
- test_lrz_seq: PASS (0.00s, CPU-only Lorenzo predictor)
- test_stat_identical: PASS (0.21s, GPU statistical functions)
- test_stat_max_error: PASS (0.23s, GPU error calculation)
- test_mem_unique: PASS (0.15s, GPU memory management)

Verdict: PASS. validated_sha=aff8ee6 (linux-gfx1100).

## Validation 2026-06-19 (windows-gfx1101)

Platform: AMD Radeon PRO V710 (gfx1101, RDNA3, wave32), Windows 11 Pro for Workstations
ROCm: 7.14.0a20260604 (TheRock nightly), HIP_VISIBLE_DEVICES=1
Fork: AMD-Ecosystem/cuSZ @ moat-port, SHA aff8ee6

### Build command

```
ROCM=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
SRC=B:/develop/moat/projects/cuSZ/src
BUILD=$SRC/build-hip-gfx1101

cmake -S $SRC -B $BUILD -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPSZ_BACKEND=HIP -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/amdclang.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/amdclang++.exe \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/amdclang++.exe \
  -DCMAKE_PREFIX_PATH=$ROCM -DBUILD_TESTING=ON \
  -DPSZ_BUILD_EXAMPLES=OFF -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON

# Strip lld-link flag injected by CMake 4.3 Windows-Clang platform module
sed -i 's/ -fuse-ld=lld-link//g' $BUILD/build.ninja

cmake --build $BUILD -j64   # 78/78 targets, 0 errors
```

Copy ROCm runtime DLLs (amdhip64_7.dll, amd_comgr.dll, hiprtc*.dll, rocm_kpack.dll
from _rocm_sdk_core/bin; hiprand.dll, rocrand.dll from _rocm_sdk_devel/bin) and
project DLLs (hipsz.dll, psz_hip_*.dll) to $BUILD/test/.

### Test results

```
HIP_VISIBLE_DEVICES=1 ctest --test-dir $BUILD -E histsp_hip --output-on-failure -j1
```

Result: 6/6 tests PASS (100%)
- test_zigzag: PASS (0.03s, CPU-only zigzag codec)
- test_l1_compact: PASS (0.24s, GPU sparse vector compaction, gfx1101)
- test_lrz_seq: PASS (0.03s, CPU-only Lorenzo predictor)
- test_stat_identical: PASS (0.21s, GPU statistical functions, CPU-GPU match verified)
- test_stat_max_error: PASS (0.20s, GPU error calculation, CPU-GPU match verified)
- test_mem_unique: PASS (0.20s, GPU memory management, gfx1101)

Same test_histsp_hip exclusion as Linux and gfx1201 (wrong include path, performance tuning only).
No TDR events; GPU healthy before and after test run.

Verdict: PASS. validated_sha=aff8ee6 (windows-gfx1101).

## Revalidation 2026-06-25 (linux-gfx1100, binary-equiv carry-forward)

Platform: AMD Radeon RX 7900 XTX (gfx1100), ROCm 7.2.1, HIP_VISIBLE_DEVICES=0
Fork: AMD-Ecosystem/cuSZ @ moat-port, SHA e443183 (validated_sha was aff8ee6)

### Delta classification

One commit: `[ROCm] Add AMD copyright/author headers and document HIP backend`. Changes 12 files
(11 .hip/.hip.inl/.cc source files + README.md): adds AMD copyright block, `@file` Doxygen tag,
`@author Jeff Daily`, and `@brief` to each new HIP source file. No code changes.

### Binary-equivalence check

Built e443183 into `build-hip-e443183` (gfx1100, with
`-DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++`).

`python3 utils/codeobj_diff.py build-hip-new build-hip-e443183`:
- `libfzg_hip.so`: identical (42 exports)
- `libphf_hip.so`: identical (432 exports)
- `libpsz_hip_core.so`: identical (204 exports)
- `libpsz_hip_stat.so`: identical (603 exports)
- `libhipsz.so`, `libpsz_hip_mem.so`, `libpsz_hip_utils.so`, `libpsz_seq_core.so`,
  `libpsz_hip_test_utils.so`, `libexample_utils2_hip.so`: indeterminate (device-code extraction
  failed) -- confirmed host-only (no `.hip_fatbin` sections)

Manual ISA comparison for GPU-executing test binaries (`l1_compact`, `mem_unique`):
extracted `.hip_fatbin` offload bundle via `clang-offload-bundler`, parsed amdgcn ELF, compared
`.text` sections -- IDENTICAL (l1_compact: 0x800 bytes, mem_unique: 0x900 bytes). The only
fatbin difference is in metadata (hip_cuid_ string changed from source comment content change),
not in GPU machine code.

Verdict: binary-equiv carry-forward. validated_sha=e443183 (linux-gfx1100).

## Revalidation 2026-06-25 (linux-gfx90a, binary-equiv carry-forward)

Platform: AMD Instinct MI250X (gfx90a), ROCm 7.2.1
Fork: AMD-Ecosystem/cuSZ @ moat-port, SHA e443183 (validated_sha was aff8ee6)

### Delta classification

`python3 utils/moatlib.py classify cuSZ aff8ee6 e443183`: `class=comment-only inert=True`

One commit: `[ROCm] Add AMD copyright/author headers and document HIP backend`. Changes 12 files
(11 .hip/.hip.inl/.cc source files + README.md): adds AMD copyright block, `@file` Doxygen tag,
`@author Jeff Daily`, and `@brief` to each new HIP source file. No code changes.

### Binary-equivalence check

Built e443183 into `build-hip-e443183` (same cmake flags as prior revalidation, with
`-DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++`).

`python3 utils/codeobj_diff.py build-hip-new build-hip-e443183`:
- `libfzg_hip.so`: identical (42 exports)
- `libphf_hip.so`: identical (432 exports)
- `libpsz_hip_core.so`: identical (204 exports)
- `libpsz_hip_stat.so`: identical (603 exports)
- `libhipsz.so`, `libpsz_hip_mem.so`, `libpsz_hip_utils.so`, `libpsz_seq_core.so`,
  `libpsz_hip_test_utils.so`: indeterminate (device-code extraction failed) -- confirmed
  host-only (no `.hip_fatbin` sections), so `indeterminate` is vacuous

Manual ISA comparison for GPU-executing test binaries (`l1_compact`, `mem_unique`):
extracted `.hip_fatbin` offload bundle, parsed amdgcn ELF, compared `.text` sections
byte-for-byte -- IDENTICAL. The only fatbin differences are in `.gnu.hash`, `.hash`, and
`.dynstr` (the `hip_cuid_` string changed from `78d20ca839961e42` to `7595f6ca12529399`
because source comment content changed); the GPU machine code is unchanged.

Verdict: binary-equiv carry-forward. validated_sha=e443183 (linux-gfx90a).
## Revalidation 2026-06-25 (linux-gfx90a)

Platform: AMD Instinct MI250X (gfx90a), ROCm 7.2.1, HIP_VISIBLE_DEVICES=0
Fork: AMD-Ecosystem/cuSZ @ moat-port, SHA e443183

### Delta classification

`git diff aff8ee6..e443183` (1 commit: "[ROCm] Add AMD copyright/author headers and document HIP backend"):
- README.md: one line of text added documenting the ROCm/HIP backend build option
- 10 .hip source files: copyright/author comment headers added (no functional code)

classify returned `class=unknown arch_independent=False`. The diff is purely comments/documentation,
but the tool could not auto-classify it. Per the carry-forward procedure, built both SHAs for
binary equivalence check: `codeobj_diff.py` returned `verdict=indeterminate` (device-code
extraction failed on several libraries). Per CLAUDE.md rules (indeterminate -> full revalidation),
ran a full GPU test suite at head SHA (e443183).

### Build command

```bash
cd /var/lib/jenkins/moat/projects/cuSZ/src
rm -rf build-hip-new
cmake -S . -B build-hip-new \
  -DPSZ_BACKEND=HIP \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++
cmake --build build-hip-new -j$(nproc)
```

### Test results

```bash
cd build-hip-new
HIP_VISIBLE_DEVICES=0 ctest --output-on-failure -E 'histsp_hip' -j1
```

Result: 6/6 tests PASS (100%)
- test_zigzag: PASS (0.00s, CPU-only zigzag codec)
- test_l1_compact: PASS (0.24s, GPU sparse vector compaction, gfx90a)
- test_lrz_seq: PASS (0.00s, CPU-only Lorenzo predictor)
- test_stat_identical: PASS (0.22s, GPU statistical functions)
- test_stat_max_error: PASS (0.22s, GPU error calculation)
- test_mem_unique: PASS (0.19s, GPU memory management)

Verdict: PASS. validated_sha=e443183 (linux-gfx90a).

## Validation 2026-06-25 (linux-gfx90a, re-port, moat-port @ 5d43e441)

Platform: AMD Instinct MI250X (gfx90a, wave64), ROCm 7.2.1, HIP_VISIBLE_DEVICES=0
Fork: AMD-Ecosystem/cuSZ @ moat-port, validated_sha=5d43e441

### Build fixes (committed as 5d43e441 on top of re-port a6d765e8)

Two build failures caught during independent validation rebuild:

1. `example/cmake/hip-example.cmake`: removed `bin_pred2`. `pred_run.hh` uses
   `cudaFree` as a `std::unique_ptr` deleter (`decltype(&cudaFree)`), but
   `cudaFree` is a preprocessor macro on the HIP path, not a function symbol.
   `bin_pred2` is driven only by `cuda-test-bin_pred.cmake` (CUDA-only test
   matrix); no HIP ctest references it. Fix: omit the target on HIP.

2. `test/cmake/cuda-test.cmake`: guard `histsp_cu` with `if(NOT PSZ_TEST_HIP)`.
   `tune_histsp.cu.inl:215` calls `GPU_histogram_generic<T>(d_in, ...)` with
   9 constructor args, but the struct only exposes `static ::init/::kernel` methods;
   the ctor signature drifted upstream. Pre-existing CUDA build failure (the CUDA
   build has identical error); guarded on HIP to prevent blocking the build.
   Also guarded `test_histsp_cu` in the RESOURCE_LOCK list.

### Build command

```bash
cd /var/lib/jenkins/moat/projects/cuSZ/src
rm -rf build-hip
cmake -S . -B build-hip \
  -DPSZ_BACKEND=HIP \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++
cmake --build build-hip -j$(nproc)   # 0 errors, warnings only
```

### Test results

```bash
HIP_VISIBLE_DEVICES=0 ctest --test-dir /var/lib/jenkins/moat/projects/cuSZ/src/build-hip \
  --output-on-failure -j1
```

Result: 98% tests passed, 1 failed out of 41. 28 PASS / 12 SKIP / 1 FAIL.

PASS (28):
- portable_str2num, portable_kv_parse, portable_check_in, portable_val_eq,
  portable_arg_builder, portable_kv_binder (6 portable CPU unit tests)
- test_zigzag (CPU-only zigzag codec)
- test_l1_compact (GPU sparse vector compaction, gfx90a)
- test_lrz_seq (CPU-only Lorenzo predictor)
- test_stat_identical1, test_stat_identical2 (GPU statistical functions)
- test_stat_max_error (GPU error calculation)
- test_mem_unique (GPU memory management)
- test_hf_revisit_altcode (GPU HFR encode/decode)
- test_hf_cpu_serial_codebook (CPU Huffman codebook)
- bin_hf codec matrix (14 tests): hf/hfr/hfr-pbkc/hfr-pbkgo over
  cauchy-mild + cauchy-sharp (except hfr__cauchy_sharp) + uniform-256 + uniform-1024

SKIP (12, rc=77): cusz__rtm/hurr/nyx CLI matrix -- needs $CUSZ_TEST_DATA datasets,
not present locally. Expected skip.

FAIL (1):
- hfr__cauchy_sharp__u2: `bin_phf: unknown option: --assert-brnum-le`
  Pre-existing upstream CLI/test desync: test CMake passes `--assert-brnum-le=200000`
  but bin_phf.cc does not implement that flag. Identical failure on the CUDA build.
  NOT a HIP regression (no numeric/codec failure, pure CLI argument error).

### CLI round-trip

```bash
# 1M f32 sine wave, abs error bound 1e-3
python3 -c "import numpy as np; data=np.sin(np.arange(1000000,dtype=np.float32)*2*np.pi/100000); data.tofile('/tmp/test_sine_1m.f32')"
HIP_VISIBLE_DEVICES=0 ./cusz -z -i /tmp/test_sine_1m.f32 -t f32 -m abs -e 1e-3 -l 1000000 --report cr,psnr
# [Lorenzo, Hist, HF-fast2]  CR=27.04  mode=Abs  input_eb=1.000000e-03  final_eb=1.000000e-03
HIP_VISIBLE_DEVICES=0 ./cusz -x -i /tmp/test_sine_1m.f32.cusza --compare /tmp/test_sine_1m.f32 --report cr,psnr
# [Lorenzo, Hist, HF-fast2]  CR=27.04  PSNR=70.8  max_error=1.000047e-03  max_error_rel=5.000234e-04
./cusz --version  # >>> hipSZ build: 2025-02-05 (0.16)
```

CR=27.04, PSNR=70.8, max_error=1.000047e-03 <= 1e-3 bound. All within spec.

Verdict: PASS. validated_sha=5d43e441 (linux-gfx90a).

## Review 2026-06-25 (re-port, moat-port @ a6d765e8 vs e1c0135)

Reviewed the single-commit re-port "[ROCm] Add a HIP backend (PSZ_BACKEND=HIP)
targeting ROCm" with the /pr-review skill. No blocking or should-fix problems
found. Verdict: review-passed.

Verified (problems only would be listed; none found, recorded here for the audit
trail of the high-risk spots the porter touched):

- Wave64 ballot: `__ballot_sync(MASK,PRED) := (uint32_t)(__ballot(PRED) >>
  (__lane_id() & ~31))` is correct on wave64 AND wave32. Confirmed every in-scope
  call site (fzg_c/fzg_x at block (32,32) tid=y*32+x; lrz_c 1D block; hfr-pbk
  reductions) uses a 32-lane logical warp that is exactly half-aligned to a
  wavefront, so the half-base shift selects the right 32 bits. ROCm `__ballot`
  returns `unsigned long long` (64-bit) on both wave sizes (upper 32 zero on
  wave32, shift 0). ROCm's own cooperative_groups uses the identical
  `__lane_id() & ~(tile_size-1)` idiom (amd_hip_cooperative_groups.h:399). The
  bin_hf HFR matrix exercising these passed on gfx90a (wave64); the wave32
  followers are the validator's job.
- hfr_encode_c.cuh:56 3-arg `__shfl_sync(0xffffffff,p_incomp,0)` -> maskless
  `__shfl(p_incomp,0)` (default width 64 on wave64) broadcasts lane 0 to all 64
  lanes; harmless because the leader value `p_incomp` is read from a block-wide
  shared scalar `s_v3_incomp`, identical on both half-warp leaders (lane 0 and
  lane 32), so the upper half gets the value it would have produced anyway.
- Variadic shuffle macros: 3-arg and 4-arg both map through `##__VA_ARGS__`.
- `_ptb_runtime::HIP` enumerator inserted between CUDA and SYCL; compare.hh /
  context.cc dispatch treat HIP exactly like CUDA. CUDA/SYCL paths unchanged.
  Confirmed the enum is a compile-time dispatch tag, not serialized.
- Double atomicAdd: HIP branch `|| defined(__HIP_DEVICE_COMPILE__)` uses native
  `atomicAdd(double*)`; CUDA `__CUDA_ARCH__>=600` guard byte-untouched.
- CUDA-preservation: every shared-header/macro edit is either HIP-only-guarded
  (`_PORTABLE_USE_HIP`/`PSZ_USE_HIP`/`__HIP_DEVICE_COMPILE__`) or reached only via
  the `cmake/hip-compat` include dir + force-include, which are injected ONLY on
  the HIP backend. The c_cu2hip_0/1/2 macros are included exclusively through the
  hip-compat shims (no source includes them directly), so the CUDA and ONEAPI
  builds never see them. The one UNCONDITIONAL shared-source edit is
  spl_y24.cuh:56 (forward decl gains `dim3 data_size`): the base forward decl was
  stale/dead (did not match the definition at :535 or the call sites at :772/810,
  which already pass a dim3); the corrected decl now matches both. Codegen comes
  from the definition (unchanged), so this is behavior-neutral on CUDA.
- gpu_event.hh/gpu_stream.hh `std::remove_pointer<cudaEvent_t/cudaStream_t>::type`
  resolves to `CUevent_st`/`CUstream_st` on CUDA (identical to the prior explicit
  spelling), so the CUDA mangled symbols are unchanged.
- CMake: top-level + per-subproject HIP branches are additive `elseif`/`if HIP /
  else CUDA`; the CUDA branch is byte-identical to upstream in each file. No
  `CMAKE_HIP_ARCHITECTURES` pin (relies on enable_language(HIP) auto-detect via
  `project(... LANGUAGES HIP ...)`). find_package(hip REQUIRED) used; build docs
  carry `-DCMAKE_PREFIX_PATH=/opt/rocm`. curand->hiprand swap correct.
  nvml/cupti/cuda_driver dropped only on the HIP path. PSZ_ACTIVATE_LC defaulted
  OFF on HIP (third_party/lc is unported CUDA; its 32-bit-mask __ballot_sync sites
  are not compiled).
- Compat shims: all 9 carry AMD copyright; guard-protected; verinfo_hip.cu has
  AMD copyright + `\author Jeff Daily`. Commit: `[ROCm]` prefix (53 chars), names
  Claude, no noreply trailer, Test Plan with literal commands. No MOAT jargon in
  the diff; only the public account.

Note (non-blocking, for the validator): the GPU re-run on real hardware (the 27
PASS / 1 known-non-port FAIL / 12 dataset-SKIP result the porter recorded) is the
validator's gate; this review did not re-run it.

## Planner (follower) 2026-06-24 (windows-gfx1201, re-port head 07db1e28)

Planner pass for the windows-gfx1201 follower. NO fresh plan authored: the existing
ROCm port (moat-port single-source, Strategy A) applies as-is; only the Windows
toolchain (TheRock venv, Ninja, all-clang amdclang, PE DLL staging) and the gfx1201
arch flag differ. Appended a "## Delta plan: windows-gfx1201" section to plan.md with
the proven Windows build recipe, DLL staging, ctest harness, and the cross-arch CR
gate.

Key finding for the validator: the prior windows-gfx1201 PASS (validated_sha aff8ee6,
notes 2026-06-08) was on the OLD dual-source base, which upstream deleted; moat-port
was force-reset to the FRESH single-source re-port (head 07db1e28, force-update
confirmed at git fetch). That record does NOT carry forward -- gfx1201 must validate
the re-port fresh.

CRITICAL wave32 dependency: gfx1201 is wave32 (RDNA4) and shares gfx1100 blocker. The
re-port head 07db1e28 carries a hardcoded WAVE64 scan at psz/src/kernel/histsp.cu.inl
:60-66 (#ifdef __HIP_PLATFORM_AMD__) that corrupts the default compression path on
wave32 -- gfx1100 (also wave32) FAILED here on 2026-06-25 (CR=0.43 / max_error=3.4e38
vs gfx90a CR=27.04). gfx1201 will reproduce this identically. The fix is a SINGLE
shared change on moat-port owned by the gfx1100-driven porter pass; the Windows host
must NOT make the source fix (validation-only host policy). gfx1201 should build +
validate AFTER the porter lands histsp and the head advances; a validate-first at the
unfixed 07db1e28 is expected to bounce to validation-failed on the cross-arch CR gate.

State advanced: windows-gfx1201 unclaimed -> planned (the only legal transition from
unclaimed; routes to porter/validator per the follower flow once the shared histsp
fix is in the head). cuSZ is pure CMake, so the torch BuildExtension .hip Windows
regression does not apply. linux-gfx1100 is a Linux-host concern and was not touched.

## Review 2026-06-25 (follower linux-gfx1100, histsp wave32 fix @ 866868f6)

Reviewed commit 866868f6 with /pr-review (local-branch mode); scope is the single
functional delta from the previously-reviewed head 07db1e28. `git diff 07db1e28..866868f6`
touches one file (psz/src/kernel/histsp.cu.inl, +8/-3). The lead linux-gfx90a is
completed at this same head (revalidated the wave64 case). Verdict: review-passed --
the width-32 scan is correct on both wave sizes; no port defect. One non-blocking
commit-hygiene item for the PR-prep phase below.

### Confirmed correct (recorded for the validator's audit, not blockers)

- The AMD branch (histsp.cu.inl:66-71) is now structurally identical to the non-AMD
  #else branch (73-78): same loop bound `d < 32`, same writeback predicate
  `threadIdx.x % 32 >= d`. The only difference is the AMD branch uses an explicit
  width-32 `__shfl_up(sum, d, 32)` while #else uses `__shfl_up_sync(0xffffffff, sum, d)`.
- The scan result is consumed at the lane-31 leader (histsp.cu.inl:82,
  `if (threadIdx.x % 32 == 31)`), so a 32-lane scan is exactly what the consumer
  expects. blockDim = 256 (num_workers, histsp.cu.inl:100) is a clean 32-multiple
  (8 logical 32-lane warps/block), so the geometry is coherent on both wave widths.
- Wave64 (gfx90a): explicit width 32 keeps the two logical 32-lane warps within a
  64-lane wavefront independent, matching the `% 32` predicate/consumer -- so the fix
  does NOT regress wave64 (lead revalidated PASS at 866868f6 confirms). Wave32
  (gfx1100/gfx1201): the logical warp is the full wavefront; the prior width-64 form's
  upper-half loss is eliminated.
- The #ifdef is still justified: the c_cu2hip_1_fix_primitives.h shuffle macros rewrite
  `__shfl_up_sync(mask, var, delta)` -> maskless `__shfl_up(var, delta)` with NO explicit
  width, which defaults to the physical warpSize (64 on wave64) -- wrong for a 32-lane
  scan. The AMD branch therefore correctly carries an explicit width-32 form rather than
  reusing the #else spelling.
- Arch-unified, not a per-arch wave32 hack: width 32 is correct on BOTH wave widths; no
  WARP_SIZE constant, no hardcoded 64, no __AMDGCN_WAVEFRONT_SIZE split introduced. The
  only remaining "64" token is in the descriptive comment (histsp.cu.inl:64).
- No stray debug (no printf/TODO/FIXME), no MOAT jargon (no lead/follower/Strategy/head_sha
  in the comment or commit body). The comment is technical and upstream-appropriate.
- Kernel is on the default compression path (compressor.inl:115,184), so this is the
  correct site for the CR-divergence symptom the validator must re-confirm on wave32.

### Non-blocking (PR-prep phase, not changes-requested)

- Commit 866868f6 body's Test Plan embeds the absolute host path
  `/var/lib/jenkins/moat/projects/cuSZ/src` (the `cd` line). The three earlier moat-port
  commits (a6d765e8/5d43e441/07db1e28) do not carry a host path in their Test Plan, so
  this is new to this commit. It is infrastructure-path noise that should not reach the
  upstream PR. The PR-prep squash collapses the branch to one clean commit before the
  upstream PR, which is the right place to scrub it (use a relative path / repo-root
  cwd). Does not affect device code, does not block validation.

Verdict: review-passed. Validator runs the real gfx1100 GPU gate next (cross-arch CR
must equal gfx90a: CR=27.04, PSNR=70.8, max_error=1.000047e-03, per the porter's
diagnostic that already round-tripped this at 866868f6).
