# ntransformer notes

## Port summary (linux-gfx90a, lead)
- Fork: https://github.com/AMD-Ecosystem/ntransformer, branch `moat-port`.
- Strategy A (pure CMake): `option(USE_HIP)` selects `project(... HIP)` vs `project(... CUDA)`; `set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE HIP)` compiles the `.cu` files as HIP without renaming them. NVIDIA path is byte-for-byte unchanged (USE_HIP default OFF).
- Compat header `src/core/cuda_to_hip.h` aliases the exact runtime + fp16 surface (1:1 hip*). A HIP-only shim dir `src/core/hip_compat/` holds `cuda_runtime.h` and `cuda_fp16.h` that just `#include "../cuda_to_hip.h"`, so the project's literal `#include <cuda_runtime.h>`/`<cuda_fp16.h>` resolve to HIP aliases. The shim dir is added with `target_include_directories(... BEFORE PUBLIC ...)` only under USE_HIP -- ROCm ships no cuda_*.h shims, hence this dir.
- Links `hip::host` (vs `CUDA::cudart`). HIP std = C++20.

## Build (gfx90a)
```
cmake -S . -B build-hip -DUSE_HIP=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build build-hip -j
ctest --test-dir build-hip --output-on-failure
```
Multi-arch correctness build (must emit both code objects):
```
cmake -S . -B build-multi -DUSE_HIP=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build build-multi -j
roc-obj-ls build-multi/test_gemm   # -> gfx90a AND gfx1100
```
USE_GPUNVME stays OFF (needs external gpu-nvme-direct lib + VFIO/root; not GPU-validatable here).

## Validation (real gfx90a, GCD 1, ROCm 7.2.1, MI250X)
- Multi-arch build clean; `roc-obj-ls` shows both `gfx90a` and `gfx1100` native code objects in test_gemm and the lib.
- `ctest`: test_tensor PASS, test_gemm PASS (gemv f32/q4_0/q6_k smem + q6_k_large no-smem, silu_mul, rmsnorm -- exact/tolerance matches).
- Deterministic across 3 runs (byte-identical output). AMD_LOG_LEVEL=3 confirms `Using native code object for device: gfx90a`.
- End-to-end GGUF decode intentionally NOT run (no model downloaded; host egress is slow). Unit tests are the agreed gate.

## Gotchas (the two real build-breakers, both the __HIPCC__-vs-__CUDACC__ trap)
1. `src/core/types.h` guarded BOTH the fp16 `half` typedef (line ~12) AND the `NT_CUDA_CHECK` macro (line ~223) on `#ifdef __CUDACC__`. hipcc defines `__HIPCC__`, not `__CUDACC__`, so every device TU fell through (uint16_t typedef / undefined NT_CUDA_CHECK) and failed to compile. Fix: `#if defined(__CUDACC__) || defined(__HIPCC__)` on both. (This is the MPPI-Generic trap; note there were TWO such guards in this file, not one -- grep the whole file for `__CUDACC__`.)
2. ROCm 7.2.1 `__shfl_xor_sync` requires a 64-bit lane mask and statically rejects an implicitly promoted 32-bit literal (`static_assert ... sizeof(MaskT)==8`). The kernels passed `0xFFFFFFFF`. Fix: a backend-portable `NT_WARP_MASK` in types.h (`0xFFFFFFFFFFFFFFFFULL` on `__HIPCC__`, else `0xFFFFFFFFu`), substituted at all 28 `__shfl_xor_sync` sites in gemm.cu/attention.cu/rmsnorm.cu/softmax.cu. Arch-unified: full-wavefront constant is correct on wave64 (all 64 lanes) and wave32 (high bits ignored). On CUDA it is the original 32-bit literal -> NVIDIA path semantically unchanged.

## Warp-width analysis (confirmed during port)
- rmsnorm/softmax/attention reductions use the `warpSize` builtin (offset starts at `warpSize/2`) and a `__shared__ float[32]` cross-warp buffer (32 = wave32 upper bound at 1024 threads; safe on wave64 at 16 warps). Wave-agnostic, no change.
- gemm.cu GEMV is a logical-warp-of-32 butterfly: `block(32, GEMV_WARPS=8)`, `flat_id = warp_id*32+tid`, `b += 32`, reduction `offset` starts at 16. Every shfl offset <= 16 so the XOR butterfly never crosses the 32-lane boundary; on wave64 two 32-rows pack into one wavefront but each reduces independently and correctly. Correct on wave64 AND wave32, no change. test_gemm's exact dot-product checks (e.g. y[0]=256 q6_k, 32768 q6_k_large) confirm this on gfx90a.
- `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, 64KB)` -> `hipFuncSetAttribute`; accepted on gfx90a, tests pass (q6_k_large exercises the no-smem fallback anyway).
- nodiscard warnings on `cudaFree`/`cudaEventDestroy` etc. in streamer.cu/gemm.cu are benign (HIP marks these [[nodiscard]]; upstream ignores the return). Not errors.

## Followers (gfx1100 / gfx1151)
- Multi-arch build already proves gfx1100 compiles and emits a code object. Build is `-DCMAKE_HIP_ARCHITECTURES=gfx1100` (no source edit). Validate a wave32 run of test_gemm/test_tensor.
- Watch points on wave32: the `warpSize`-based reductions and the logical-32 GEMV butterfly are wave-agnostic by construction (see analysis above); NT_WARP_MASK high bits are ignored on wave32. Expect a clean pass; delta-port only if a numeric test fails.

## Validation 2026-06-02 (validator, linux-gfx90a)

Platform: MI250X gfx90a, ROCm 7.2.1, HIP_VISIBLE_DEVICES=1 (GCD 1).
Fork: AMD-Ecosystem/ntransformer moat-port @ 144ab937bbaa7aad3440106358006dc014d776b6.

### Multi-arch build (gfx90a + gfx1100)
```
cmake -S projects/ntransformer/src -B agent_space/ntransformer/build-multi \
  -DUSE_HIP=ON -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_HIP_ARCHITECTURES=gfx90a;gfx1100" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build agent_space/ntransformer/build-multi -j$(nproc)
```
Result: BUILD CLEAN (nodiscard warnings only, documented benign).

`roc-obj-ls agent_space/ntransformer/build-multi/test_gemm` confirms:
- `hipv4-amdgcn-amd-amdhsa--gfx1100` code object present
- `hipv4-amdgcn-amd-amdhsa--gfx90a` code object present

### ctest (2 deterministic runs)
```
HIP_VISIBLE_DEVICES=1 AMD_LOG_LEVEL=3 ctest --test-dir agent_space/ntransformer/build-multi \
  --output-on-failure -R "test_tensor|test_gemm"
```
Run 1: 2/2 PASS (0.45s total). Run 2: 2/2 PASS (0.46s total). Deterministic.

AMD_LOG_LEVEL=3 confirms: `Using native code object for device: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-`

Exact/tolerance results:
- gemv_f32: y[0]=32.0 (expected 32.0), y[1]=-16.0 (expected -16.0) PASS
- gemv_q4_0 (smem): y[0]=256.0 (expected 256.0), y[1]=-256.0 (expected -256.0) PASS
- gemv_q6_k_large (no-smem): y[0]=32768.0 (expected 32768.0), y[1]=-32768.0 PASS
- silu_mul: [0.000, 0.731, -0.269, 1.762] within tolerance PASS
- rmsnorm: [0.365148, 0.730296, 1.095444, 1.460593] exact match PASS

### Verdict
PASS. State: linux-gfx90a review-passed -> completed. validated_sha=144ab937bbaa7aad3440106358006dc014d776b6. linux-gfx1100 unblocked to port-ready.

## Review 2026-06-02 (reviewer, linux-gfx90a)
Reviewed moat-port 144ab93 vs upstream f2237be via /pr-review. No problems found; recommendation Approve -> review-passed.

Verified on real gfx90a (MI250X, ROCm 7.2.1, HIP_VISIBLE_DEVICES=1):
- USE_HIP=ON gfx90a build clean; host .cpp compiled by /usr/bin/c++ (confirms the host path never sees hipcc, so the float16_t=uint16_t storage branch holds on both backends).
- ctest: test_tensor PASS, test_gemm PASS. Exact dot-product checks match (q6_k smem y[0]=256, q6_k_large no-smem y[0]=32768, rmsnorm/silu_mul within tolerance) -- confirms the logical-32 GEMV butterfly and the warpSize cross-warp reductions are correct on wave64.
- Multi-arch CMAKE_HIP_ARCHITECTURES="gfx90a;gfx1100" builds; roc-obj-ls test_gemm shows both gfx90a and gfx1100 native code objects.

Fault-class review (all clear):
- Strategy A applied correctly: option(USE_HIP) with project(... HIP) vs project(... CUDA); set_source_files_properties(LANGUAGE HIP) (no file renames); compat header + hip_compat shim dir added BEFORE only under USE_HIP. NVIDIA path byte-identical when OFF.
- Both __CUDACC__ traps in src/core/types.h fixed (half typedef line 16, NT_CUDA_CHECK line 238) -> `#if defined(__CUDACC__) || defined(__HIPCC__)`. The third __CUDACC__ guard at src/core/tensor.cpp:7 is correctly LEFT UNCHANGED: tensor.cpp is a host .cpp (never nvcc/hipcc), so it always takes the extern-C stub #else branch on both backends -- not a missed trap.
- All 28 __shfl_xor_sync sites substituted with NT_WARP_MASK (0xFFFFFFFFFFFFFFFFULL on HIP / 0xFFFFFFFFu on CUDA); zero raw 0xFFFFFFFF literals remain. warpSize-based reductions are wave-agnostic; gemm.cu GEMV offset starts at 16 and the XOR butterfly never crosses the 32-lane boundary (verified: warp_id*32+tid layout, each logical 32-row writes its own tid==0). Correct on wave64 and wave32.
- No hardcoded warpSize=32; the `__shared__ float[32]` cross-warp buffers are sized at the wave32 upper bound (1024/32), safe at wave64 (16 warps). attention.cu:50 `acc[HEAD_DIM/32]` is dead/unused (cosmetic, harmless).
- No textures/resource handles -> no rule-of-five or pitch concerns. No library swaps (no cuBLAS/cuFFT/Thrust/CUB). hipFuncSetAttribute(64KB) accepted on gfx90a (q6_k smem test passes). nodiscard warnings on hipFree/hipFuncSetAttribute are benign (upstream ignores return; CUDA-identical behavior).
- Commit hygiene: title `[ROCm] Add HIP backend for AMD GPUs (gfx90a/gfx1100)` 52 chars, mentions Claude, no noreply trailer, no ghstack, no em-dash, no AMD-internal account refs. Fork main is a clean upstream mirror (f2237be).

## Validation 2026-06-02 (linux-gfx1100)

Platform: AMD Radeon Pro W7800 48GB x4, gfx1100 (RDNA3, wave32), ROCm 7.2.1, HIP_VISIBLE_DEVICES=0.
Fork: AMD-Ecosystem/ntransformer moat-port @ 144ab937bbaa7aad3440106358006dc014d776b6 (no fork push; follower validate-first, no source changes needed).

### Build (gfx1100 single-arch)
```
cmake -S projects/ntransformer/src -B agent_space/ntransformer/build-hip \
  -DUSE_HIP=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build agent_space/ntransformer/build-hip -j$(nproc)
```
Result: BUILD CLEAN (nodiscard warnings only, documented benign).

gfx1100 code-object evidence (`roc-obj-ls agent_space/ntransformer/build-hip/test_gemm`):
- `hipv4-amdgcn-amd-amdhsa--gfx1100` present (no gfx90a; single-arch build correct)

### ctest (3 deterministic runs)
```
HIP_VISIBLE_DEVICES=0 AMD_LOG_LEVEL=3 ctest --test-dir agent_space/ntransformer/build-hip \
  --output-on-failure -R "test_tensor|test_gemm"
```
Run 1: 2/2 PASS (0.58s). Run 2: 2/2 PASS (0.43s). Run 3: 2/2 PASS (0.45s). Deterministic.

AMD_LOG_LEVEL=3 confirms: `Using native code object for device: amdgcn-amd-amdhsa--gfx1100`

Exact dot-product results (vs gfx90a@144ab93 -- byte-identical):
- gemv_f32: y[0]=32.0 (expected 32.0), y[1]=-16.0 (expected -16.0) PASS
- gemv_q4_0 (smem): y[0]=256.0 (expected 256.0), y[1]=-256.0 (expected -256.0) PASS
- gemv_q6_k_large (no-smem): y[0]=32768.0 (expected 32768.0), y[1]=-32768.0 PASS
- silu_mul: [0.000, 0.731, -0.269, 1.762] within tolerance PASS
- rmsnorm: [0.365148, 0.730296, 1.095444, 1.460593] exact match PASS

### Wave32 verdict
- NT_WARP_MASK = 0xFFFFFFFFFFFFFFFFULL: high 32 bits ignored on wave32 -- all 28 __shfl_xor_sync sites correct.
- warpSize-based reductions (offset starts warpSize/2=16, __shared__ float[32] cross-warp buffer): correct at wave32 (32 lanes/warp, buffer holds up to 32 warps -- well within 1024/32=32 max).
- gemm.cu GEMV logical-32 butterfly (block(32, GEMV_WARPS=8), offsets <= 16): native at wave32, XOR butterfly never crosses lane boundary. Exact dot-products (y[0]=32768 q6_k_large) prove correct.
- Zero HSA 0x1016 faults. No NaN, no hang, no wrong output.
- Fork clone: stays clean (build in agent_space, not in src/).

### Verdict
PASS. State: linux-gfx1100 port-ready -> completed. validated_sha=144ab937bbaa7aad3440106358006dc014d776b6.

## Validation 2026-06-03 (windows-gfx1151)

Platform: AMD Radeon 8060S gfx1151 (RDNA3.5, wave32), Windows 11, TheRock ROCm pip wheels (clang-cl 23.0.0git), HIP_VISIBLE_DEVICES=0.
Fork: AMD-Ecosystem/ntransformer moat-port @ 1249659 (new commit on top of 144ab937; gfx90a/gfx1100 validated_sha=144ab93 is a reachable ancestor, unaffected).

### Windows-specific fixes (new commit 1249659, NOT an amend of 144ab93)

The Linux port at 144ab93 compiles cleanly with clang++ on POSIX but fails on Windows because:
1. `src/core/types.h`: `aligned_alloc` is POSIX/C11 -- absent from MSVC/UCRT.
2. `src/model/loader.cpp`: uses `open`/`fstat`/`mmap`/`munmap`/`close`/`madvise` (POSIX mmap).
3. `src/memory/streamer.cu` + `streamer.h`: uses `sys/sysinfo.h`, `sys/mman.h`, `fcntl.h`, `unistd.h`; `sysinfo()`, `/proc/meminfo`, and `open`/`mmap`/`munmap` in `init_delta`.
4. `CMakeLists.txt`: `-march=native` is a gcc-driver flag; clang-cl (MSVC frontend) rejects it.

Fix: added `src/core/platform.h` with `#ifdef _WIN32` shims:
- `NT_ALIGNED_ALLOC`/`NT_ALIGNED_FREE` -> `_aligned_malloc`/`_aligned_free`
- `NtMmap` struct + `nt_mmap_open`/`nt_mmap_close` -> `CreateFileMapping`/`MapViewOfFile`/`CloseHandle`
- `nt_available_ram`/`nt_total_ram` -> `GlobalMemoryStatusEx`

All three POSIX-heavy files guarded with `#ifndef _WIN32` / `#ifdef _WIN32` blocks; Linux path byte-for-byte unchanged.

### Build (gfx1151 single-arch, all-clang-cl)
```
ROCM_DEVEL=D:/Develop/TheRock/.venv/Lib/site-packages/_rocm_sdk_devel
cmake -S projects/ntransformer/src -B agent_space/ntransformer/build-hip \
  -G Ninja -DUSE_HIP=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DCMAKE_HIP_STANDARD=20 \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_CXX_COMPILER=$ROCM_DEVEL/lib/llvm/bin/clang-cl.exe \
  -DCMAKE_HIP_COMPILER=$ROCM_DEVEL/lib/llvm/bin/clang-cl.exe \
  -DCMAKE_CXX_FLAGS="-DNOMINMAX -DWIN32_LEAN_AND_MEAN" \
  -DCMAKE_PREFIX_PATH=$ROCM_DEVEL
cmake --build agent_space/ntransformer/build-hip -j4
```
Result: BUILD CLEAN. nodiscard warnings on hipFree/hipHostFree/hipMemcpyAsync only -- benign (same as Linux, documented). All 27 build objects compiled; `test_gemm.exe`, `test_tensor.exe`, `ntransformer.exe` produced.

### Deploy runtime DLLs (TheRock dlls, not System32 Adrenalin)
```
cp $ROCM_DEVEL/../_rocm_sdk_core/bin/amdhip64_7.dll build-hip/
cp $ROCM_DEVEL/../_rocm_sdk_core/bin/amd_comgr0713.dll build-hip/
cp $ROCM_DEVEL/../_rocm_sdk_core/bin/rocm_kpack.dll build-hip/
cp $ROCM_DEVEL/../_rocm_sdk_core/bin/hiprtc*.dll build-hip/
```
Note: System32 `amdhip64_7.dll` (Adrenalin) returns `invalid argument` for device library -- must use TheRock DLLs beside the exe.

### ctest (2 deterministic runs)
```
HIP_VISIBLE_DEVICES=0 ctest --test-dir agent_space/ntransformer/build-hip \
  --output-on-failure -R "test_tensor|test_gemm"
```
Run 1: 2/2 PASS (10.11s total). Run 2: 2/2 PASS (10.12s total). Deterministic.

Exact dot-product results (vs gfx90a/gfx1100 @ 144ab93 -- byte-identical):
- gemv_f32: y[0]=32.0, y[1]=-16.0 PASS
- gemv_q4_0 (smem): y[0]=32.0, y[1]=-16.0 PASS
- gemv_q6_k (smem): y[0]=256.0, y[1]=-256.0 PASS
- gemv_q6_k_large (no-smem): y[0]=32768.0, y[1]=-32768.0 PASS
- silu_mul: [0.000, 0.731, -0.269, 1.762] within tolerance PASS
- rmsnorm: [0.365148, 0.730296, 1.095444, 1.460593] exact match PASS

### Wave32 verdict (gfx1151)
- NT_WARP_MASK = 0xFFFFFFFFFFFFFFFFULL: high 32 bits ignored on wave32 -- all 28 shfl sites correct.
- warpSize-based reductions (wave32: offset starts at 16, shared[32] = 32 max warps): correct.
- gemm.cu GEMV logical-32 butterfly native on wave32. Exact q6_k_large y[0]=32768 proves correctness.
- Zero HSA faults, no NaN, no hang, no wrong output.

### Verdict
PASS. State: windows-gfx1151 port-ready -> completed. validated_sha=1249659 (new commit on top of 144ab93).

## Windows gfx1151 (2026-06-03): VALIDATED -> completed @ 1249659

GPU-validated on gfx1151 (AMD Radeon 8060S, TheRock ROCm). 2/2 ctest PASS, byte-identical to the gfx90a reference: gemv_f32 / gemv_q4_0 / gemv_q6_k / gemv_q6_k_large (no-smem) / silu_mul / rmsnorm all exact or within tol. Built all-clang-cl (CMAKE_CXX_COMPILER=CMAKE_HIP_COMPILER=clang-cl, -DCMAKE_HIP_ARCHITECTURES=gfx1151, HIP/CXX std 20, -DNOMINMAX -DWIN32_LEAN_AND_MEAN). Ran with amdhip64_7+amd_comgr0713+rocm_kpack+hiprtc deployed beside the test exe (System32 Adrenalin amdhip64 is device-lib-mismatched).

WINDOWS PORT (new commit 1249659 ON TOP of the gfx90a 144ab93 -- not an amend; 144ab93 stays a reachable ancestor so gfx90a/gfx1100 carry-forward). Five POSIX->Win32 host fixes, all #ifdef _WIN32 (Linux paths byte-identical): a new src/core/platform.h shimming aligned_alloc (_aligned_malloc), mmap (CreateFileMapping/MapViewOfFile), and RAM query (GlobalMemoryStatusEx); types.h includes platform.h before namespace nt; loader.cpp/.h and streamer.cu/.h swap POSIX open/fstat/mmap/munmap/madvise + /proc/meminfo+sysinfo for the Win32 shims under _WIN32; CMakeLists uses /O2 /DNDEBUG instead of -O3 -march=native under WIN32 (clang-cl rejects -march). No warp intrinsics -> wave32 numerically equivalent. gfx90a/gfx1100 -> revalidate at 1249659 (carry-forward expected: the delta is entirely _WIN32-guarded, Linux codegen unchanged).

## Validation 2026-06-03 (linux-gfx90a, revalidate -> carry-forward)

Platform: MI250X gfx90a, ROCm 7.2.1, HIP_VISIBLE_DEVICES=2 (GCD 2).
Fork: AMD-Ecosystem/ntransformer moat-port @ 124965909f8a1746c7d717dc32eba419d3757462.
Prior validated_sha: 144ab937bbaa7aad3440106358006dc014d776b6.

### Carry-forward check

Delta classify: `class=mixed arch_independent=False` -- full binary-equivalence check required.

Built both SHAs for gfx90a into separate build dirs, then ran `codeobj_diff.py`:

```
cmake -S projects/ntransformer/src -B agent_space/ntransformer/build-old \
  -DUSE_HIP=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build agent_space/ntransformer/build-old -j$(nproc)
# (src at 144ab937)

cmake -S projects/ntransformer/src -B agent_space/ntransformer/build-new \
  -DUSE_HIP=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build agent_space/ntransformer/build-new -j$(nproc)
# (src at 124965909)

python3 utils/codeobj_diff.py build-old/test_gemm build-new/test_gemm
python3 utils/codeobj_diff.py build-old/ntransformer build-new/ntransformer
```

Results:
- `test_gemm`: `verdict=identical` (exported symbols + device ISA identical, 2 exports)
- `ntransformer`: `verdict=identical` (exported symbols + device ISA identical, 7 exports)
- `test_tensor`: no device code objects (pure CPU test; roc-obj-ls "No kernel section found") -- nothing to diff

### Verdict
CARRY-FORWARD (binary-equiv). New commit 1249659 adds only `#ifdef _WIN32` host-side shims; device ISA on gfx90a is byte-identical to 144ab937. State: linux-gfx90a revalidate -> completed. validated_sha=124965909f8a1746c7d717dc32eba419d3757462.

## Validation 2026-06-03 (gfx1100) -- carry-forward to 1249659 (Windows-only delta)

Revalidate triggered by the fork advancing 144ab93 -> 1249659 (windows-gfx1151
enablement). The delta is Windows-only; the gfx1100 Linux compiled code is
functionally identical, so the prior gfx1100 validation carries forward.

Delta analysis (git diff 144ab93 1249659, 7 files):
- CMakeLists.txt: wraps the MSVC `/O2` flags in `if(WIN32)`; the Linux `else()`
  branch keeps the original `-O3 -DNDEBUG -march=native` -- Linux flags unchanged.
- src/core/platform.h (NEW): Win32 `_aligned_malloc`/`HANDLE`-mmap/`nt_*_ram`
  helpers under `#ifdef _WIN32`; the POSIX `#else` branch defines
  `NT_ALIGNED_ALLOC(a,s) = aligned_alloc(a,s)` and `NT_ALIGNED_FREE(p) = ::free(p)`
  -- exactly the original behavior (verified in platform.h:92-93).
- src/core/types.h: `nt_aligned_alloc/free` route through the `NT_ALIGNED_ALLOC/FREE`
  macros, which expand to the original `aligned_alloc`/`::free` on Linux -- no-op.
- src/memory/streamer.cu/.h, src/model/loader.cpp/.h: every added block is
  `#ifdef _WIN32` (Win32 mmap / `nt_total_ram`/`nt_available_ram`); the `#else`
  Linux branches are the original `/proc/meminfo`+`sysinfo`+POSIX-`mmap` code
  (only comment-wording tweaks). No device-kernel change.

Net effect on gfx1100: the device kernels (gemv/attention/rmsnorm) are untouched
and the Linux host path is functionally identical (macro indirection resolves to
the original aligned_alloc/free). The prior gfx1100 validation holds: test_gemm +
test_tensor 2/2, exact dot-products (gemv_q6_k_large y[0]=32768, rmsnorm exact),
wave32 correct. validated_sha -> 1249659. No GPU re-run, no fork change.

## Validation 2026-06-06 (windows-gfx1201)

Platform: AMD RX 9070 XT gfx1201 (RDNA4, wave32), Windows 11, TheRock ROCm 7.14.0a20260604 pip wheels (clang-cl 23.0.0git), HIP_VISIBLE_DEVICES=0 (gfx1101 absent).
Fork: AMD-Ecosystem/ntransformer moat-port @ 124965909f8a1746c7d717dc32eba419d3757462 (same Windows commit as gfx1151).

### Build (gfx1201 single-arch, all-clang-cl)
```
ROCM_DEVEL=/b/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
cmake -S projects/ntransformer/src -B agent_space/ntransformer/build-hip-gfx1201 \
  -G Ninja -DUSE_HIP=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_CXX_STANDARD=20 -DCMAKE_HIP_STANDARD=20 \
  -DCMAKE_CXX_COMPILER=$ROCM_DEVEL/lib/llvm/bin/clang-cl.exe \
  -DCMAKE_HIP_COMPILER=$ROCM_DEVEL/lib/llvm/bin/clang-cl.exe \
  "-DCMAKE_CXX_FLAGS=-DNOMINMAX -DWIN32_LEAN_AND_MEAN" \
  -DCMAKE_PREFIX_PATH=$ROCM_DEVEL
cmake --build agent_space/ntransformer/build-hip-gfx1201 -j64
```
Result: BUILD CLEAN. nodiscard warnings on hipFree/hipHostFree/hipEventDestroy only -- benign (same as Linux and gfx1151, documented). All 27 build objects compiled; test_gemm.exe, test_tensor.exe, ntransformer.exe produced.

### Deploy runtime DLLs (TheRock dlls, not System32 Adrenalin)
```
ROCM_CORE=.venv/Lib/site-packages/_rocm_sdk_core
cp $ROCM_CORE/bin/amdhip64_7.dll build-hip-gfx1201/
cp $ROCM_CORE/bin/amd_comgr.dll build-hip-gfx1201/
cp $ROCM_CORE/bin/rocm_kpack.dll build-hip-gfx1201/
cp $ROCM_CORE/bin/hiprtc0714.dll build-hip-gfx1201/
cp $ROCM_CORE/bin/hiprtc-builtins0714.dll build-hip-gfx1201/
```
Note: DLL name is `amd_comgr.dll` on this TheRock build (not `amd_comgr0713.dll` as on gfx1151 host).

### ctest (2 deterministic runs)
```
HIP_VISIBLE_DEVICES=0 ctest --test-dir agent_space/ntransformer/build-hip-gfx1201 \
  --output-on-failure -R "test_tensor|test_gemm"
```
Run 1: 2/2 PASS (8.52s total). Run 2: 2/2 PASS (8.41s total). Deterministic.

Exact dot-product results (vs gfx90a/gfx1100/gfx1151 -- byte-identical):
- gemv_f32: y[0]=32.0, y[1]=-16.0 PASS
- gemv_q4_0 (smem): y[0]=32.0, y[1]=-16.0 PASS
- gemv_q6_k (smem): y[0]=256.0, y[1]=-256.0 PASS
- gemv_q6_k_large (no-smem): y[0]=32768.0, y[1]=-32768.0 PASS
- silu_mul: [0.000, 0.731, -0.269, 1.762] within tolerance PASS
- rmsnorm: [0.365148, 0.730296, 1.095444, 1.460593] exact match PASS

### Wave32 verdict (gfx1201, RDNA4)
- NT_WARP_MASK = 0xFFFFFFFFFFFFFFFFULL: high 32 bits ignored on wave32 -- all 28 shfl sites correct.
- warpSize-based reductions (wave32: offset starts at 16, shared[32] = 32 max warps): correct.
- gemm.cu GEMV logical-32 butterfly native on wave32. Exact q6_k_large y[0]=32768 proves correctness.
- Zero HSA faults, no NaN, no hang, no wrong output.

### Verdict
PASS. State: windows-gfx1201 port-ready -> completed. validated_sha=124965909f8a1746c7d717dc32eba419d3757462.

## Validation 2026-06-16 (windows-gfx1101)

Platform: AMD Radeon PRO V710 gfx1101 (RDNA3, wave32), Windows 11, TheRock ROCm 7.14.0a20260604 pip wheels (clang-cl 23.0.0git), HIP_VISIBLE_DEVICES=0 (only GPU visible).
Fork: AMD-Ecosystem/ntransformer moat-port @ 0ddaad4db896592f6c695572b9f5137dbdbe5242.

### Build (gfx1101 single-arch, all-clang-cl)
```
ROCM_DEVEL=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
cmake -S projects/ntransformer/src -B agent_space/ntransformer/build-hip-gfx1101 \
  -G Ninja -DUSE_HIP=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1101 -DCMAKE_CXX_STANDARD=20 -DCMAKE_HIP_STANDARD=20 \
  -DCMAKE_CXX_COMPILER=$ROCM_DEVEL/lib/llvm/bin/clang-cl.exe \
  -DCMAKE_HIP_COMPILER=$ROCM_DEVEL/lib/llvm/bin/clang-cl.exe \
  "-DCMAKE_CXX_FLAGS=-DNOMINMAX -DWIN32_LEAN_AND_MEAN" \
  -DCMAKE_PREFIX_PATH=$ROCM_DEVEL
cmake --build agent_space/ntransformer/build-hip-gfx1101 -j64
```
Result: BUILD CLEAN (27/27 objects). nodiscard warnings on hipFree/hipHostFree/hipEventDestroy only -- benign (same as all prior platforms, documented).

### Deploy runtime DLLs (TheRock dlls, not System32 Adrenalin)
```
ROCM_CORE=.venv/Lib/site-packages/_rocm_sdk_core
cp $ROCM_CORE/bin/amdhip64_7.dll build-hip-gfx1101/
cp $ROCM_CORE/bin/amd_comgr.dll build-hip-gfx1101/
cp $ROCM_CORE/bin/rocm_kpack.dll build-hip-gfx1101/
cp $ROCM_CORE/bin/hiprtc0714.dll build-hip-gfx1101/
cp $ROCM_CORE/bin/hiprtc-builtins0714.dll build-hip-gfx1101/
```

### ctest (3 deterministic runs)
```
HIP_VISIBLE_DEVICES=0 ctest --test-dir agent_space/ntransformer/build-hip-gfx1101 \
  --output-on-failure -R "test_tensor|test_gemm"
```
Run 1: 2/2 PASS (8.50s total). Run 2: 2/2 PASS (8.37s total). Run 3 (verbose): 2/2 PASS (8.39s total). Deterministic.

Exact dot-product results (vs gfx90a/gfx1100/gfx1151/gfx1201 -- byte-identical):
- gemv_f32: y[0]=32.0, y[1]=-16.0 PASS
- gemv_q4_0 (smem): y[0]=32.0, y[1]=-16.0 PASS
- gemv_q6_k (smem): y[0]=256.0, y[1]=-256.0 PASS
- gemv_q6_k_large (no-smem): y[0]=32768.0, y[1]=-32768.0 PASS
- silu_mul: [0.000, 0.731, -0.269, 1.762] within tolerance PASS
- rmsnorm: [0.365148, 0.730296, 1.095444, 1.460593] exact match PASS

### Wave32 verdict (gfx1101)
- NT_WARP_MASK = 0xFFFFFFFFFFFFFFFFULL: high 32 bits ignored on wave32 -- all 28 shfl sites correct.
- warpSize-based reductions (wave32: offset starts at 16, shared[32] = 32 max warps): correct.
- gemm.cu GEMV logical-32 butterfly native on wave32. Exact q6_k_large y[0]=32768 proves correctness.
- Zero HSA faults, no NaN, no hang, no wrong output. Post-test hipInfo confirmed GPU still healthy.

### Verdict
PASS. State: windows-gfx1101 port-ready -> completed. validated_sha=0ddaad4db896592f6c695572b9f5137dbdbe5242.

## Doc cleanup 2026-06-16 (porter, PR-prep): remove out-of-scope research notes
Removed docs/research/ (5 files: R1_delta_streaming, R2_sparse_ffn, R3_predictive_skip, README, literature_survey) -- speculative "novel optimization" design notes introduced by upstream-side commits e23e5dd/1fa1484, unrelated to the ROCm/HIP port and not for the upstream PR. The dir held ONLY these speculative files, so the whole tree went. No external references (grepped README + other docs: none). Also scrubbed the in-house "Strategy A:" porting-strategy label from src/core/cuda_to_hip.h:6 (kept the technical explanation). New commit 387b41a ON TOP of the validated 1249659 (reachable ancestor preserved -- no amend). advance-head: inert carry-forward, all 4 completed platforms (gfx90a/gfx1100/gfx1201/gfx1151) carried to 387b41a with NO revalidation. Pushed AMD-Ecosystem/ntransformer moat-port (fast-forward).
