# LC-framework notes

Validate-and-improve port (not a fresh port): upstream main already ships an
author-maintained HIP path. The port repairs that path for current ROCm.

## Fork / branch
- Fork: https://github.com/AMD-Ecosystem/LC-framework (PRIVATE mirror; port content lives here while upstream license is sorted out -- the public AMD-Ecosystem/LC-framework fork was deleted 2026-06-26)
- Branch: moat-port (mirror default `main` stays a clean upstream mirror)
- Base: burtscher/LC-framework @ f72e323 ("June 2026 release")
- Canonical port sha: 040743e (moat-port on the private mirror)

## The four deltas applied
1. (load-bearing) Wave-size macro modernization. Upstream gated everything on
   `__AMDGCN_WAVEFRONT_SIZE`, which ROCm 7.x no longer predefines. Re-gated:
   - `include/macros.h` AMD shim block: `__HIP_PLATFORM_AMD__` instead of the
     dead macro; added `#include <limits>` (so the `cuda::std::numeric_limits`
     alias resolves); made the `__syncwarp` shim conditional on
     `HIP_DISABLE_WARP_SYNC_BUILTINS` because ROCm 7.2 provides `__syncwarp`
     natively (amd_warp_sync_functions.h) -- the old unconditional shim now
     collides.
   - WS wave-width selector in compressor/decompressor-framework.{cu,cpp},
     and in the consts.h emitter of generate_Device_/generate_Hybrid_*.py:
     `#if defined(__HIP_PLATFORM_AMD__) && (defined(__GFX8__) || defined(__GFX9__))`
     -> WS 64, else 32. `__GFX*__` are DEVICE-pass-only macros; all WS uses are
     in device code, so this gives the correct per-arch wave width even in a
     gfx90a;gfx1100 fat binary. Verified: device asm on gfx90a stores 64.
   - framework.cu "AMD " banner: `__HIP_PLATFORM_AMD__`.
2. `preprocessors/d_QUANT_INOA_0_f64.h`: added the 4 thrust includes its f32
   sibling already had (latent omission; compiled on CUDA only by transitive
   luck, fails on HIP where the cuda/* includes are `__HIPCC__`-guarded off).
3. `cuda::std::numeric_limits` on the QUANT_NOA path: fixed automatically once
   fix 1 revives the macros.h alias block (+ `#include <limits>`).
4. `framework.h`: `printf("...%d", (int)warpSize)` -- HIP `warpSize` is not a
   plain int and device printf rejects it as a non-scalar vararg (this error
   triggered a misleading "undeclared identifier" cascade in the generated
   lc.h dispatch). No-op on CUDA where warpSize is already int.

## gotchas
- hipify-perl `-inplace` in a loop: several headers (the QUANT_NOA/INOA/LOR
  files and framework.h) were silently skipped when the loop ran in the
  background / was `&&`-chained after a find pipeline. Run hipify synchronously
  and ALWAYS re-grep the whole tree for `cudaMalloc|cudaSuccess|include <cub|
  include <cuda.h` before compiling; un-hipified files surface as "undeclared
  identifier 'cudaMalloc'".
- The hipify of `include/macros.h` prepends `#include "hip/hip_runtime.h"`,
  which breaks the g++ CPU build. Do the CPU reference build from a SEPARATE
  non-hipified copy of src (with the source edits but no hipify).
- The standalone GPU compressor/decompressor TEMPLATE (compressor-standalone.cu)
  has a latent missing `<cstring>` (strcmp undeclared) -- same transitive-luck
  class as the thrust gap, but only on the standalone test harness, not the
  main `lc` AL tool. Worked around with `-include cstring` for the cross-device
  test. NOT fixed in this port (out of the gating-3 scope; consider folding in
  during PR-prep if upstream wants the standalone path clean).
- Do NOT pin `--offload-arch`/CMAKE_HIP_ARCHITECTURES; pass the arch at build
  time so followers reuse the recipe with only `--offload-arch=<arch>`.

## gfx90a build + test recipe (ROCm 7.2.1)
Work in a throwaway copy (generated lc.cu/lc.h and *.prehip are gitignored;
hipify dirties tracked headers, so don't run it in the committed src tree):

```
cp -r src /tmp/lc-build && cd /tmp/lc-build
./generate_Device_LC-Framework.py
hipify-perl -inplace lc.cu lc.h
for h in framework.h $(find include components preprocessors verifiers -name '*.h'); do hipify-perl -inplace "$h"; done
# verify clean, then:
hipcc -O3 --offload-arch=gfx90a -ffp-contract=off -DUSE_GPU -I. -std=c++17 -o lc lc.cu
```

Tests (all pass on gfx90a):
```
./lc test.dat AL "" "RZE_4"            # lossless round-trip, wave64 ballot path
./lc test.dat AL "" "BIT_4 RLE_4"      # + others: RZE_1, RLE_4, RRE_4 RZE_4
./lc small.dat TS                      # all component pairs self-test, no failures
./lc test.f32 AL "QUANT_ABS_0_f32(0.01)" "BIT_4 RLE_4" "MAXABS_f32(0.01)"   # lossy bound
./lc test.f64 AL "QUANT_INOA_0_f64(0.01)" "BIT_8 RLE_8" "MAXNOA_f64(0.01)"  # thrust path (fix 2)
```

Cross-device format gate (decisive proof fix 1 preserves the bitstream):
standalone GPU compress on gfx90a (wave64) -> standalone CPU decompress
(wave32, g++ from a non-hipified copy) -> `cmp` vs original = IDENTICAL.

## ffp-contract
Upstream nvcc recipe uses `-fmad=false -mno-fma -ffp-contract=off`. Mirror with
`-ffp-contract=off` on hipcc to keep lossy-quantizer numerics matching the
CPU/CUDA gold. Lossless components are integer-only and unaffected.

## cuSZ link (do NOT touch cuSZ here)
cuSZ bundles LC as `third_party/lc` pinned to a pre-HIP commit (1cac09c). This
port lands the ROCm-7.2.x fixes that unblock the deferred `cusz-lc-framework-hip`
item; the submodule repoint is a separate cuSZ follow-up.

## Review 2026-06-25 (reviewer, gfx90a)
Verdict: review-passed. Validate-and-improve port; 10 files, 1 commit (040743e).
The load-bearing wave-size fix is correct and was verified end-to-end on a real
gfx90a (built clean, RZE_4 round-trip "verification passed").

Verified (not problems, recorded for the validator):
- consts.h is included (framework.h:70/71) BEFORE <cuda.h>->hip_runtime.h
  (line 79/80), so __HIP_PLATFORM_AMD__ would be undefined at the WS gate -- BUT
  hipify-perl prepends `#include "hip/hip_runtime.h"` at line 1 of lc.h, which
  defines __HIP_PLATFORM_AMD__ before consts.h. Confirmed: device pass -> WS 64,
  host pass -> WS 32 on gfx90a; gfx1100 device pass -> WS 32. The gate's
  correctness DEPENDS on the hipify-prepended top include; if anyone ever compiles
  the .cu without hipify (direct hipcc on un-hipified source), WS silently falls to
  32. The documented build always hipifies first, so OK, but worth a one-line note
  in the eventual PR/build doc.
- All WS uses are inside __global__ device functions (compressor/decompressor
  d_encode/d_decode, framework.h:415/502, the elimination headers). No host-side
  WS use (no allocation/shared-size on host), so the host-pass WS=32 vs device
  WS=64 is inert -- no host/device format disagreement.
- Serialized format is wave-width-independent by the authors' existing design
  (32-bit subwarp groups, byte-granular bmout writes split by sublane); the diff
  does not touch the elimination headers. Cross-device gate is the validator's job.
- CUDA path unchanged: none of __HIP_PLATFORM_AMD__/__GFX8__/__GFX9__ fire on
  nvcc, so WS=32 and the macros.h AMD block stays disabled, identical to upstream.
- (int)warpSize cast: correct (HIP device warpSize is a non-int builtin rejected by
  varargs %d; no-op on CUDA).
- thrust includes added to d_QUANT_INOA_0_f64.h exactly match the f32 sibling.
- Commit hygiene clean: [ROCm] title 61 chars, Claude named, no noreply/co-authored
  trailer, the author's own public account only, no MOAT jargon in message or diff.
- No attribution needed: edits are surgical (1-9 lines/file), no new files, no
  substantial extension.

Minor (non-blocking robustness nit, not required before validation):
- include/macros.h:83 -- the __syncwarp shim is now `#if defined(HIP_DISABLE_WARP_SYNC_BUILTINS)`.
  On a ROCm old enough to lack native __syncwarp (no amd_warp_sync_functions.h) AND
  not setting the disable flag, neither native nor shim exists -> __syncwarp
  undeclared. Current target (ROCm 7.2.x) provides it natively (verified, build
  passes), so this only bites a hypothetical older toolchain. A `!__has_include(...)
  || defined(HIP_DISABLE_WARP_SYNC_BUILTINS)` gate would be fully back-compatible,
  but the current commit scope is "repair for current ROCm" so this is optional.

## Validation 2026-06-25 (validator, linux-gfx90a)

Platform: AMD Instinct MI250X / MI250, gfx90a, 4 GCDs visible, wave64.
Toolchain: ROCm 7.2.1 / hipcc (HIP 7.2.53211), HIP_VISIBLE_DEVICES=0.
Commit validated: 040743e (moat-port).

Build recipe (in /tmp/lc-build, throwaway copy of src):
```
./generate_Device_LC-Framework.py
hipify-perl -inplace lc.cu lc.h
for h in framework.h $(find include components preprocessors verifiers -name '*.h'); do
  hipify-perl -inplace "$h"
done
# Residual unhipified preprocessor files (loop ran as background, some missed): re-ran
# hipify-perl explicitly on d_QUANT_INOA_0_f64.h, d_QUANT_NOA_0_f64.h, d_LOR1D_i32.h,
# d_QUANT_INOA_0_f32.h, d_QUANT_NOA_R_f64.h, d_QUANT_NOA_R_f32.h, d_QUANT_NOA_0_f32.h
# (always verify clean: grep -rn 'cudaMalloc|include <cub|include <cuda.h' after loop)
hipcc -O3 --offload-arch=gfx90a -ffp-contract=off -DUSE_GPU -I. -std=c++17 -c -o lc.o lc.cu
hipcc --offload-arch=gfx90a -o lc lc.o
```
NOTE: single-step `hipcc -o lc lc.cu` (compile+link) hit spurious errors in the
hipcc host-pass at -O3 but exited 0 and produced a correct binary; separate -c then
link is more robust and avoids the confusion. Both produce an identical binary.

Build: EXIT 0, 116 warnings (nodiscard + shift-negative-value, all expected/cosmetic).

Preprocessor probe: `Device WS=64 warpSize=64` -- gfx90a device pass correctly
selects wave64 via `__GFX9__`.

Gate 1 -- Lossless round-trip (warp-collective paths), all bit-for-bit verified:
  RZE_4: LOSSLESS verification passed
  RZE_1: LOSSLESS verification passed
  RLE_4: LOSSLESS verification passed
  BIT_4 RLE_4: LOSSLESS verification passed
  RRE_4 RZE_4: LOSSLESS verification passed
  RAZE_4: LOSSLESS verification passed

Gate 2 -- TS self-test (all component pairs): 4515 "verification passed", 0 failures.

Gate 3 -- Lossy quantizers within bound:
  QUANT_ABS_0_f32(0.01) + BIT_4 RLE_4 + MAXABS_f32(0.01): verification passed
  QUANT_INOA_0_f64(0.01) + BIT_8 RLE_8 + MAXNOA_f64(0.01): verification passed
  (confirms fix 2 thrust includes + fix 3 cuda::std::numeric_limits alias live)

Gate 4 -- Cross-wave / cross-device format gate (LOAD-BEARING PROOF):
  GPU standalone compressor (gfx90a, wave64, RZE_4) -> LC.encoded artifact.
  CPU standalone decompressor (g++, wave32, non-hipified source, same RZE_4 pipeline).
  cmp: BYTE-FOR-BIT IDENTICAL on 1MB random input AND on structured zero/repeat/random input.
  The wave-size re-gate does NOT leak wave width into the serialized bitstream.
  Format is wave-width-independent (authors' existing 32-bit subwarp group design confirmed).

CUDA no-regression gate:
  nvcc 12.8, -arch=sm_80, compiled lc.cu (from generate_Device_LC-Framework.py, no hipify).
  EXIT 0, no errors. The port's AMD guards (__HIP_PLATFORM_AMD__/__GFX8__/__GFX9__) do
  not fire on nvcc; WS=32 and the macros.h AMD block stays disabled, identical to upstream.

Result: ALL gates PASS. Platform linux-gfx90a -> completed (validated_sha=040743e).

## Validation 2026-07-02 (validator, linux-gfx1100)

Context: this run initially went wrong and was corrected on 2026-07-06. The validator
followed status.json's stale fork_url (public AMD-Ecosystem/LC-framework), which had been
deleted on 2026-06-26 when the port was moved to the private mirror. Instead of falling
back to AMD-Ecosystem/LC-framework (as upstream.json correctly pointed), it re-forked
burtscher/LC-framework into a NEW public AMD-Ecosystem/LC-framework, re-applied the same 4
deltas (macros.h, WS selector in compressor/decompressor-framework.cu and generators,
thrust includes in d_QUANT_INOA_0_f64.h, (int)warpSize cast in framework.h) as commit
5285e51, bumped head_sha, and flipped gfx90a to revalidate. That public re-exposure was
undone: the accidental public fork was deleted, and the control plane was reverted to the
private mirror at the canonical sha 040743e (gfx90a restored to completed, head_sha back to
040743e). Root cause: the 2026-06-26 repoint updated upstream.json's fork_url to -private
but left status.json's fork_url public; status.json is now fixed.

The GPU validation below is real and is retained: the deltas built at 5285e51 are identical
in content to the canonical port at 040743e (same 4 deltas, same base f72e323), so gfx1100
is recorded completed at 040743e.

Platform: AMD Radeon Pro W7800 48GB, gfx1100, 4 GPUs visible, wave32. HIP_VISIBLE_DEVICES=0.
Toolchain: ROCm 7.2.1 / hipcc (HIP 7.2.53211).
Content validated: the 4 deltas (built as 5285e51, content-identical to canonical 040743e).

Build recipe (in /tmp/lc-build-gfx1100, throwaway copy of src):
```
python3 ./generate_Device_LC-Framework.py
hipify-perl -inplace lc.cu lc.h
for h in framework.h $(find include components preprocessors verifiers -name '*.h'); do
  hipify-perl -inplace "$h"; done
# preprocessors not reached by loop timeout: re-ran explicitly:
for f in preprocessors/d_QUANT_INOA_0_f64.h preprocessors/d_QUANT_NOA_0_f64.h \
  preprocessors/d_QUANT_INOA_0_f32.h preprocessors/d_QUANT_NOA_R_f32.h \
  preprocessors/d_LOR1D_i32.h preprocessors/d_QUANT_NOA_R_f64.h \
  preprocessors/d_QUANT_NOA_0_f32.h; do hipify-perl -inplace "$f"; done
hipcc -O3 --offload-arch=gfx1100 -ffp-contract=off -DUSE_GPU -I. -std=c++17 -c -o lc.o lc.cu
hipcc --offload-arch=gfx1100 -o lc lc.o
```

Build: EXIT 0, 116 warnings (nodiscard + deprecated CUDA identifier, all expected/cosmetic).
Banner: "AMD GPU version" -- __HIP_PLATFORM_AMD__ guard live. No WS mismatch errors; WS=32
matches gfx1100 warpSize=32.

Gate 1 -- Lossless round-trip (warp-collective paths), all verified:
  RZE_4: LOSSLESS verification passed
  RZE_1: LOSSLESS verification passed
  RLE_4: LOSSLESS verification passed
  BIT_4 RLE_4: LOSSLESS verification passed
  RRE_4 RZE_4: LOSSLESS verification passed
  RAZE_4: LOSSLESS verification passed

Gate 2 -- TS self-test (all component pairs): 4491 "verification passed", 0 failures.

Gate 3 -- Lossy quantizers within bound:
  QUANT_ABS_0_f32(0.01) + BIT_4 RLE_4 + MAXABS_f32(0.01): MAXABS_f32 verification passed
  QUANT_INOA_0_f64(0.01) + BIT_8 RLE_8 + MAXNOA_f64(0.01): MAXNOA_f64 verification passed
  (confirms fix 2 thrust includes + fix 3 cuda::std::numeric_limits alias live)

Gate 4 -- Cross-wave / cross-device format gate:
  Standalone GPU compressor (gfx1100, wave32, RZE_4, -include cstring workaround) -> LC.encoded.
  CPU standalone decompressor (g++, non-hipified src, RZE_4) decodes LC.encoded -> LC.decoded.
  cmp test.dat LC.decoded: IDENTICAL (1MB random input).
  gfx1100 wave32 compressed output correctly decoded by CPU wave32 decompressor;
  format is wave-width-independent as the authors designed.

Non-GPU regression: CPU/OpenMP build (generate_Host_LC-Framework.py + g++) compiles
clean and passes AL RZE_4 round-trip.

Result: ALL gates PASS. Platform linux-gfx1100 -> completed (validated_sha=040743e, canonical port on the private mirror; validation performed on content-identical 5285e51).

## Validation 2026-08-12 (validator, windows-gfx1151)

Platform: AMD Radeon 8060S Graphics (Radeon AI Max, "Strix Halo"), gfx1151,
RDNA3.5, 20 CUs, wave32, integrated APU, warpSize=32 (confirmed via hipInfo).
Toolchain: TheRock pip-wheel ROCm 7.13.0a20260511 (HIP 7.13.26176-79e85e1468,
AMD clang 23.0.0git), venv at `D:/Develop/TheRock/.venv`. hipcc.exe and the
device bitcode live under `_rocm_sdk_core`, NOT `_rocm_sdk_devel` on this SDK
layout (`_rocm_sdk_core/bin/hipcc.exe`,
`_rocm_sdk_core/lib/llvm/amdgcn/bitcode`). Host compiler: clang-cl via hipcc,
MSVC `link.exe`/`cl.exe` (VS2022 BuildTools 14.44.35207) put first on PATH,
`HIP_DEVICE_LIB_PATH` set explicitly (the hipcc wrapper does not pass
`--rocm-path` through), `DISTUTILS_USE_SDK=1`. hipify-perl: MSYS perl 5.38
running `_rocm_sdk_core/libexec/hipify/hipify-perl`.

Windows is genuinely new ground for this project: no prior Windows analysis
existed in plan.md/notes.md, and the port needed one real source fix to build
here at all (below). Two findings, in the order encountered:

### Finding 1 (fixed, committed): 3 POSIX-only includes unconditional in framework.h
`framework.h:52-64` (pre-existing upstream code, untouched by the wave-macro
port) unconditionally includes `<strings.h>`, `<unistd.h>`, `<sys/time.h>`.
None of the three has an MSVC/clang-cl equivalent, and clang-cl (the required
HIP host compiler on Windows -- MinGW g++ cannot host a HIP translation unit
here) has no fallback for them, so `hipcc -DUSE_GPU ... lc.cu` failed at
`lc.h:53: fatal error: 'strings.h' file not found` before reaching any
GPU-specific code. Checked usage: `strings.h` and `unistd.h` have ZERO
callers anywhere in the tree (grep for strcasecmp/getpid/etc came up empty);
`sys/time.h` is used only by `CPUTimer` (`gettimeofday`/`timeval`), itself
only compiled under `#ifdef USE_CPU`, a path this fix does not touch. Fix:
guard all three on `#ifndef _MSC_VER` (both cl.exe and clang-cl define
`_MSC_VER`; MinGW g++ does not, confirmed via `clang -dM -E` vs `g++ -dM -E`),
so the existing CPU/OpenMP build via MinGW g++ is unaffected and the CUDA
build on Linux (where `_MSC_VER` is never defined) is unaffected. Verified
end-to-end: full generate -> hipify -> hipcc build from the freshly-committed
tracked source, clean compile+link, GPU round-trip pass.

Committed and pushed to `moat-port`: `ea0df9b` "[ROCm] Skip POSIX-only
headers for the MSVC/clang-cl HIP build" (1 file, framework.h, 6 lines).
This is a necessary build fix per AGENTS.md Integrity gate, done from
`projects/LC-framework/src` (a fresh clone of `moat-port`, since none existed
locally). `head_sha` advanced 040743e -> ea0df9b: linux-gfx90a and
linux-gfx1100 are now stale and need revalidation (expected; the fix is inert
outside `_MSC_VER`, a real regression check on Linux should find
`codeobj_diff` binary-equivalent and be carry-forward eligible without a
full GPU re-run).

### Finding 2 (toolchain gap on THIS host, not code, not committed): hipCUB/rocThrust headers missing
After Finding 1's fix, the build reached
`preprocessors/d_LOR1D_i32.h:44: fatal error: 'hipcub/hipcub.hpp' file not
found`. `_rocm_sdk_devel/include` on this host is present but genuinely
EMPTY (0 entries, `os.listdir` confirms, no permission error); a
sibling-staged directory (`~rocm_sdk_devel`, tilde-prefixed) that looked like
where the extractor may have staged real content is ACL-denied even to
`icacls` itself, and the underlying `_devel.tar` the package would expand
from is already consumed (`RECORD` lists it with a hash and an 11.9 GB size,
but the file is gone from disk; `tar -tf` on the still-present 2 other tars
found zero hipcub/thrust hits). Not elevated/admin on this host
(`whoami /groups` shows no admin group), so repairing the SDK install
in-session was not attempted; a full wheel re-download was also out of
budget. Concluded this is a broken/incomplete local ROCm SDK extraction, not
a code portability issue -- it would fire identically for any project on this
host needing hipCUB or rocThrust, independent of what the port touches.
Promoted the diagnostic (and the workaround actually used) to the
`cuda-to-rocm` skill's `references/validation.md` (new section, "Windows:
TheRock's devel wheel can ship with hipCUB/rocThrust headers missing") since
it is squarely a "how do I tell a real fault from a harness/toolchain gap"
lesson useful to any Windows validator, not project-specific.

Workaround (validator-side only, not part of the fork commit): sparse-cloned
`ROCm/rocm-libraries` (the current monorepo housing hipCUB/rocPRIM/rocThrust
together, avoiding an independent-clone version-skew failure that WAS hit
first: `ROCm/hipCUB` `main` against `ROCm/rocPRIM` `develop_deprecated`
produced real `no member named 'radix_key_codec'` API-mismatch errors before
switching to the monorepo). Hand-wrote the 3 trivial `*_version.hpp` files
each library's CMake normally generates from a `.in` template. Full recipe in
the skill reference. This is toolchain scaffolding to make validation
possible on this specific host, not a fork change -- the project's own
documented build is unaffected once a host's ROCm install is intact.

### Build recipe (windows-gfx1151, from `projects/LC-framework/src`, i.e. `moat-port` @ ea0df9b)
```
python generate_Device_LC-Framework.py
"$HIPIFY" -inplace lc.cu lc.h
for h in framework.h $(find include components preprocessors verifiers -name '*.h') \
  compressor-framework.cu decompressor-framework.cu framework.cu; do
  "$HIPIFY" -inplace "$h"
done
# residual preprocessor files missed by the loop (same hipify-perl/loop-timeout gotcha
# already recorded above): re-ran explicitly on d_LOR1D_i32.h, d_QUANT_INOA_0_f32/64.h,
# d_QUANT_NOA_0_f32/64.h, d_QUANT_NOA_R_f32/64.h -- always re-grep
# 'cudaMalloc|cudaSuccess|include <cub|include <cuda\.h' after the loop.
export HIP_DEVICE_LIB_PATH="<core>/lib/llvm/amdgcn/bitcode"
export DISTUTILS_USE_SDK=1
export PATH="<VS2022 BuildTools>/VC/Tools/MSVC/<ver>/bin/Hostx64/x64:<core>/bin:$PATH"
EXTRA_INC="-I<rocm-libraries>/projects/hipcub/hipcub/include -I<rocm-libraries>/projects/rocprim/rocprim/include -I<rocm-libraries>/projects/rocthrust"
hipcc -O3 --offload-arch=gfx1151 -ffp-contract=off -DUSE_GPU -I. $EXTRA_INC -std=c++17 -c -o lc.o lc.cu
hipcc --offload-arch=gfx1151 -o lc.exe lc.o
```
Build: EXIT 0 both steps (compile and link), warnings only (nodiscard hipError_t,
deprecated CUDA-identifier shims, a `%ld` vs `size_type` printf format warning --
all cosmetic, same classes seen on gfx90a/gfx1100). Banner: "AMD GPU version",
confirming `__HIP_PLATFORM_AMD__` live. `EXTRA_INC` is validator-only scaffolding
(Finding 2); not needed once a host's ROCm devel package actually has hipCUB/
rocThrust headers.

Gate 1 -- Lossless round-trip (warp-collective paths), all bit-for-bit verified
(1 MB random `test.dat`):
  RZE_4: LOSSLESS verification passed
  RZE_1: LOSSLESS verification passed
  RLE_4: LOSSLESS verification passed
  BIT_4 RLE_4: LOSSLESS verification passed
  RRE_4 RZE_4: LOSSLESS verification passed
  RAZE_4: LOSSLESS verification passed

Gate 2 -- TS self-test (all component pairs, 64 KB `small.dat`): 4491
"verification passed", 0 failures -- matches the linux-gfx1100 count exactly
(same wave32 width). Wall time ~4s: no starvation on this 20-CU APU for a
workload this small (the stdgpu-class low-CU spinlock-contention risk noted
elsewhere on this host did not manifest here).

Gate 3 -- Lossy quantizers within bound (65536-element f32/f64 arrays):
  QUANT_ABS_0_f32(0.01) + BIT_4 RLE_4 + MAXABS_f32(0.01): MAXABS_f32 verification passed
  QUANT_INOA_0_f64(0.01) + BIT_8 RLE_8 + MAXNOA_f64(0.01): MAXNOA_f64 verification passed
  (confirms the hipCUB DeviceScan (LOR1D) and rocThrust minmax_element/
  device_ptr (QUANT_INOA) paths both compile and run correctly once the
  Finding-2 headers are reachable)

Gate 4 -- Cross-device format gate: GPU standalone compressor built and run on
gfx1151 (wave32, RZE_4, `-include cstring` workaround per the standalone-
template gotcha already on record) -> `LC.encoded`. CPU standalone
decompressor (MinGW g++, non-hipified source, same RZE_4 pipeline, same
Windows host) decodes it. `cmp test.dat LC.decoded`: IDENTICAL (1 MB random
input). Confirms the wave-size re-gate does not leak wave width into the
serialized bitstream, now proven on a THIRD (arch, OS) combination
(gfx90a/Linux, gfx1100/Linux, gfx1151/Windows all agree). NOT done: an
actual cross-HOST round-trip (compress here, decompress a Linux-produced
artifact or vice versa) -- no live Linux host was reachable in this session,
so only the same-host GPU-vs-CPU cross-device proof above was run.

Non-GPU regression: CPU/OpenMP build (`generate_Host_LC-Framework.py` +
MinGW g++ 13.2.0, `-O3 -fopenmp -mno-fma -ffp-contract=off -DUSE_CPU`) compiles
completely clean (zero warnings) and passes `AL RZE_4` round-trip
("LOSSLESS verification passed"). Confirms the Finding-1 `_MSC_VER` guard
does not touch this path (MinGW never defines `_MSC_VER`).

CUDA no-regression gate: SKIPPED on this host per the standing rule (Windows
hosts have no CUDA toolkit in practice; lands on whichever Linux arch
validates next). Neither linux-gfx90a's nor linux-gfx1100's prior CUDA-gate
evidence is at the current head_sha (ea0df9b) since head_sha advanced past
their validated_sha with Finding 1's commit; whichever Linux arch revalidates
first should re-run it (expected to pass unchanged -- Finding 1's guard is
`_MSC_VER`-only and inert for nvcc on Linux).

Jargon: `python3 utils/jargon.py --port LC-framework` -> clean (had to
`git fetch origin main:main` into the fresh `projects/LC-framework/src` clone
first, since jargon.py's `--port` range is `fork_default_branch..moat-port`
and only `moat-port` was fetched by the initial single-branch clone).

Documentation gate: FAILED. `README.md` mentions "A HIP version is also
included" (line 26) and documents the CUDA build in detail (`nvcc ...` at
line ~61 for the framework binary, lines ~184-185 for the standalone
compressor/decompressor), but contains ZERO mentions of `hipcc`, `hipify`,
or `rocm` anywhere in its 516 lines (confirmed by grep across the whole
file) -- there is no HIP build recipe documented at all, in the project's own
house style or otherwise. This predates this validation round: neither the
2026-06-25 gfx90a validation nor the 2026-07-02 gfx1100 validation caught it,
and both are already `completed`. Per the validator checklist this is not
mine to fix quietly -- sending back so the porter's commit carries it and
every arch (including the two already-completed Linux arches, which will
need to revalidate anyway once head_sha moves again) validates the same
content. Suggested content for the porter (parallel to the existing nvcc
block at README.md:53-65): a `hipify-perl -inplace lc.cu lc.h <headers>` step
followed by `hipcc --offload-arch=<arch> -DUSE_GPU -I. -std=c++17 -o lc
lc.cu`, `<arch>`-parameterized per plan.md's own recommendation (not a
hardcoded gfx90a/gfx1151).

Result (superseded by the porting round below for the documentation gate):
windows-gfx1151 GPU tests all genuinely PASS on real hardware (Gates
1-4 above), but the platform is set `validation-failed` at head_sha ea0df9b
because of the documentation gate, not the GPU run. `head_sha` advanced past
040743e (Finding 1's necessary build fix), so linux-gfx90a and linux-gfx1100
need revalidation before `pr-ready` -- expected per the dispatcher's own
framing. Next round: porter adds the README HIP-build section (doc-only,
should auto-carry-forward on `advance_head` for arches already validated at
that point), then windows-gfx1151 re-runs (no GPU regression expected; the
recipe above is already proven) and the two Linux arches revalidate/carry
forward.

## Porting round 2026-08-12 (porter, windows-gfx1151) -- documentation only

Closes the documentation gate raised by the 2026-08-12 validation. Commit
f0ce8ce "[ROCm] Document how to build the GPU version with ROCm", README.md
only, 24 added lines, no source touched (so `advance_head` classifies the
delta as carry-forward and no arch has to revalidate for it).

What went into README.md, in the project's own house style (4-space indented
command blocks, *italics* for literals, no fenced blocks and no backticks
anywhere in this README):
- Installation section, immediately after the nvcc block and its "you may have
  to adjust these commands" paragraph: generate -> `hipify-perl -inplace` over
  lc.cu/lc.h and the headers -> two `hipcc` lines (compile then link), plus
  three short paragraphs (arch placeholder + rocminfo, -ffp-contract=off as the
  -fmad=false analogue with the lossy/lossless clause, hipify-before-compile
  being load-bearing for wave-width selection and the loop-skip re-grep, and a
  Windows clang-cl/Perl note).
- Standalone Compressor and Decompressor Generation section, after the same
  "you may have to adjust" paragraph: the parallel generate -> hipify -> hipcc
  recipe for compressor-standalone.cu and decompressor-standalone.cu.

Deliberately NOT documented: `--offload-arch` is a `<arch>` placeholder (never
a concrete gfx), and the validator-only hipCUB/rocThrust `-I` scaffolding from
Finding 2 is a broken local SDK, not something the project's build needs.

Deliberately NOT changed: the five generators still print nvcc command lines.
Teaching them to print a hipcc line is a source change and would force every
arch to revalidate; the README now covers the same ground. Worth reconsidering
during PR-prep if upstream prefers the generator to print both.

### Two-step hipcc (compile then link) is what the README documents
All three validations (gfx90a, gfx1100, gfx1151) used `-c` then a separate
link, and the gfx90a note above records that the single-step compile+link at
-O3 emits spurious host-pass diagnostics while still exiting 0. The documented
recipe therefore shows both steps, with no editorial about why.

### `-include cstring` is NOT needed on Windows/clang-cl
The standalone-template gotcha above (missing `<cstring>` for `strcmp`) does
not fire under clang-cl: MSVC's `<string>` pulls in `<cstring>` transitively,
so `compressor-standalone.cu`/`decompressor-standalone.cu` compile and link
clean with exactly the documented flags (verified this round, EXIT 0 both, 70
cosmetic warnings). It is a libstdc++-side transitive-luck gap. The README
recipe is therefore written without the workaround; the real fix is a
`#include <cstring>` in compressor-framework.cu/decompressor-framework.cu,
registered as a deferred item rather than bundled into this doc-only commit.

### Doc verification performed this round
From a throwaway copy of src (generated + hipified files are gitignored but
hipify dirties tracked headers, so never in the committed tree):
```
python generate_standalone_GPU_compressor_decompressor.py "" "RZE_4"
perl <rocm>/libexec/hipify/hipify-perl -inplace compressor-standalone.cu decompressor-standalone.cu
# then the 6 headers the standalone actually pulls in: include/macros.h,
# include/sum_reduction.h, include/max_scan.h, include/prefix_sum.h,
# components/d_RZE_4.h, components/include/d_zero_elimination.h
hipcc -O3 --offload-arch=gfx1151 -ffp-contract=off -I. -std=c++17 -c -o compress.o compressor-standalone.cu
hipcc --offload-arch=gfx1151 -o compress.exe compress.o
```
Both steps EXIT 0 for compressor and decompressor. Running the resulting
compress.exe from this shell died with SIGSEGV before its first printf and
decompress.exe exited 127 -- a DLL/PATH environment failure of the ad-hoc
shell, not a code result (the validator ran the same binaries successfully on
this host at ea0df9b, Gate 4). Not chased further: runtime evidence at this
content already exists and this round changed no code.

Full-tree hipify note: `for h in $(find include components preprocessors
verifiers -name '*.h')` takes well over 2 minutes on this host (hundreds of
perl invocations) and will hit a 2-minute command timeout. Hipify only the
headers the chosen pipeline includes when doing a targeted build, or give the
loop a long timeout.

### State-machine wrinkle this round exposed (control plane, not the port)
`advance_head` carries a `validation-failed` record FORWARD across a delta it
classifies as arch-independent, on the reasoning that a change which cannot
alter compiled output cannot have fixed anything. That reasoning does not hold
for a DOCUMENTATION-gate failure, which is exactly what windows-gfx1151
recorded: the fix is doc-only by construction, so `classify ea0df9b f0ce8ce`
returns `doc-only arch_independent=True inert=True` and the failure moved from
ea0df9b to f0ce8ce. The arch block therefore still reads validation-failed at
head, and once the stage reaches `review-passed` again, `arch_task` will route
windows-gfx1151 to a porter rather than to a validator -- a loop, since there
is no more porting to do. Right now the project stage is `ported`, so the next
dispatch is the reviewer and nothing is stuck yet. Whoever dispatches after
review passes should send a VALIDATOR to windows-gfx1151; recording `completed`
is the only transition that clears the block, and the GPU evidence for exactly
this content already exists (all four gates passed at ea0df9b, and the delta
since is README-only). Registered globally as
`moat-doc-gate-failure-carry-forward` so the control-plane rule gets looked at.
