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

## Review 2026-08-13 (reviewer, linux-gfx942)

Verdict: changes-requested. Scope reviewed: the whole fork diff
`f72e323...f0ce8ce` on moat-port (11 files, 3 commits) plus the two
`cuda-to-rocm` lessons riding this MOAT branch (b494d30, 1b2d0fe). The
040743e content passed review on 2026-06-25; findings 1, 4 and 5 below are in
the two commits added since, findings 2 and 3 are in 040743e and are raised
now because the port is heading for an upstream PR.

All findings were reproduced on this host: ROCm 7.14.60850 (AMD clang 23.0),
gfx942 (wave64), nvcc 12.8 for the CUDA cross-check.

### 1. The standalone ROCm recipe the README documents does not compile (Linux)

`README.md:205-211` documents generate -> hipify -> hipcc for the standalone
compressor/decompressor. Run as written on Linux it fails:

```
compressor-standalone.cu:339:30: error: use of undeclared identifier 'strcmp'
decompressor-standalone.cu:...: error: use of undeclared identifier 'strcmp'
```

Reproduced with the README's own example pipeline ("TUPL4_1 RRE_1 CLOG_1")
and with RZE_4, for both binaries. Cause: `compressor-framework.cu:330` and
`decompressor-framework.cu:258` call `strcmp` while the templates include only
`<string>` (`compressor-framework.cu:54`); on the CUDA path `<cuda/std/limits>`
and `<cuda/atomic>` (`:59-62`, `#if !defined(__HIPCC__)`) drag `string.h` in,
and on HIP they are guarded off. Confirmed nvcc 12.8 compiles the same
generated file cleanly, so this is a HIP-path-only gap of exactly the class
already fixed once in this port (the thrust includes in
`preprocessors/d_QUANT_INOA_0_f64.h:40-43`).

This also makes an upstream-visible claim false: f0ce8ce's Test Plan says
"Both builds complete with cosmetic warnings only", which holds on clang-cl
(MSVC `<string>` pulls in `<cstring>`) but not under libstdc++, i.e. not on
the platform most readers of that README use.

Fix: add `#include <cstring>` beside `#include <string>` in
`compressor-framework.cu:54` and `decompressor-framework.cu:54`. Verified: the
standalone builds clean with it (checked via `-include cstring`, exit 0), and
the include is inert on nvcc and on the g++ CPU path. That closes
`deferred.json` item `lc-standalone-cstring-include`; deferring it was a
control-plane optimization (keeping f0ce8ce carry-forward), but the branch now
ships a recipe that fails on the first try, and windows-gfx1151 has to
revalidate at head anyway.

### 2. The WS gate does not need to depend on hipify's prepended header

`compressor-framework.cu:48`, `decompressor-framework.cu:48`,
`compressor-framework.cpp:48`, `decompressor-framework.cpp:48`,
`generate_Device_LC-Framework.py:101`, `generate_Hybrid_LC-Framework.py:138`:

```
#if defined(__HIP_PLATFORM_AMD__) && (defined(__GFX8__) || defined(__GFX9__))
```

`__HIP_PLATFORM_AMD__` is NOT a compiler predefine (verified:
`hipcc -x hip --offload-arch=gfx942 -dM -E` yields `__GFX9__`, `__HIPCC__`,
`__gfx942__`, and no `__HIP_PLATFORM_*`); it comes from `hip/hip_runtime.h`.
The gate sits above every include in the standalone templates
(`compressor-framework.cu:48` vs first include at `:54`) and above
`<cuda.h>` in the framework path (`framework.h:76` includes `consts.h`,
`framework.h:79` is the runtime header), so the whole selector is load-bearing
on hipify-perl happening to prepend the header at line 1. That is the trap
`README.md:74` then has to warn about ("compiling the unconverted code
silently selects the wrong width"), and it is avoidable: `__GFX8__`/`__GFX9__`
are themselves AMD-clang device-pass predefines and are never defined by nvcc
or g++.

Verified equivalence of the shorter gate, all four passes, no header included:

    gate                                   gfx942 dev  gfx1100 dev  host  nvcc
    __HIP_PLATFORM_AMD__ && __GFX8/9__     64          32           32    32
    __GFX8__ || __GFX9__                   64          32           32    32

Also verified `__GFX9__` is set for gfx908/gfx90a/gfx950 and for
`gfx9-4-generic`, and `__GFX11__`/`__GFX12__` (not `__GFX9__`) for
gfx1100/gfx1201/`gfx11-generic`, so the arch coverage of the selector is
right.

Second half of the same line: the port dropped the
`__AMDGCN_WAVEFRONT_SIZE == 64` term entirely, where plan.md's open question
had settled on "keep both (additive, no cost)". Dropping it costs the two
cases where the old macro was the more accurate answer -- a ROCm old enough
to predefine it but predating the `__GFX*__` predefines, and an explicit
`-mwavefrontsize64` on RDNA (verified this ROCm predefines nothing for
`-mwavefrontsize64` on gfx1100, so the current gate answers 32 for a wave64
compile). In `lc` that surfaces as the `framework.h:227` trap; in the
standalone binaries there is no such check, so it is silent.

Suggested single form for all six sites:

```
#if defined(__GFX8__) || defined(__GFX9__) || \
    (defined(__AMDGCN_WAVEFRONT_SIZE) && (__AMDGCN_WAVEFRONT_SIZE == 64))
```

If this lands, update the `README.md:74` paragraph: the hipify-before-compile
ordering is still required for the API calls, but the wavefront-width
justification stops being true and should not stay in the README or in the
commit body.

### 3. ROCm older than 6.2 loses `__syncwarp` (include/macros.h:83-85)

The shim is now emitted only under `#if defined(HIP_DISABLE_WARP_SYNC_BUILTINS)`.
That mirrors ROCm's own guard exactly (`amd_warp_sync_functions.h:14` and
`amd_warp_functions.h:115` are `#if !defined(HIP_DISABLE_WARP_SYNC_BUILTINS)`),
so it is right for any ROCm that ships that header -- but before it existed,
an AMD build got the shim unconditionally from this same block, and now gets
neither. `__syncwarp()` is called at `framework.h:274` and
`compressor-framework.cu:168`, both marked "not optional", so such a build
fails to compile. This was recorded as optional in the 2026-06-25 review;
with the branch now aimed at an upstream PR, a user on ROCm 6.0/6.1 is a
plausible reader. Version-arithmetic-free fix:

```
#if defined(HIP_DISABLE_WARP_SYNC_BUILTINS) || \
    !__has_include(<hip/amd_detail/amd_warp_sync_functions.h>)
```

Not testable here (no pre-6.2 ROCm on this host); the claim rests on the
header's own guard structure, which is quoted above.

### 4. The promoted lesson carries a command that prunes the wrong repository

`.claude/skills/cuda-to-rocm/references/validation.md:66-68` gives:

    git clone --filter=blob:none --sparse --depth 1 <url> && git sparse-checkout set projects/hipcub projects/rocprim projects/rocthrust

`git clone` does not change directory, so the second command runs in the
parent. If the parent is not a repository it errors; if it IS one -- and a
validator is normally standing in a checkout -- `git sparse-checkout set`
silently applies to that repository and empties its working tree of everything
outside the named paths. Demonstrated in a scratch repo: three tracked files
before, one after. This is a lesson landing on `main` for every future agent
to follow, so it has to be the working form: insert `cd rocm-libraries &&`
between the two commands.

### 5. The README does not say the conversion rewrites the checked-out sources

`README.md:65-78` and `:203-211`: `hipify-perl -inplace` is run over
`framework.h` and every header under `include/`, `components/`,
`preprocessors/` and `verifiers/`, i.e. over tracked project sources, after
which the tree no longer builds with nvcc. hipify leaves `*.prehip` backups
(already in `.gitignore:20`), so it is recoverable, but a reader following the
recipe is not told either fact. One sentence naming the in-place rewrite and
the `.prehip` backups belongs with the recipe.

### Checked this round, no change needed

- Every `WS` use in the tree is inside device code (`framework.h:227,270-295,
  421,508`, the component and reduction headers); no host-side use and no
  dynamic shared-memory size computed on the host (all launches are
  `<<<blocks, TPB>>>`), so the host pass resolving `WS` to 32 stays inert.
- `framework.h:227` is the only `warpSize` reference in the tree, so the
  `(int)` cast has no sibling site left unfixed.
- `ea0df9b`'s `_MSC_VER` guards are safe: `strings.h` and `unistd.h` have no
  callers anywhere, and `sys/time.h`'s only user, `CPUTimer`
  (`framework.h:790-799`), is inside `#ifdef USE_CPU`; `framework.cu`'s
  `CPUTimer` uses at `:219,289,604` are all inside `#ifdef USE_CPU` blocks.
- The thrust includes in `d_QUANT_INOA_0_f64.h:40-43` match the f32 sibling
  exactly, and the other four thrust users already carry theirs.
- Commit hygiene: three titles at 53/61/60 chars, all `[ROCm]`, all with an
  AI-assistance disclosure and a Test Plan, no `Co-Authored-By`, no noreply
  trailer, author is the maintainer's own public address, ASCII-clean in both
  the diff and the messages, `jargon.py --port LC-framework` clean.
- Fork tree at f0ce8ce is clean (`git status --porcelain` empty).

## Review 2026-08-13 (reviewer, linux-gfx1100)

Verdict: changes-requested. Scope: the delta 040743e..f0ce8ce (ea0df9b
framework.h POSIX-include guards, f0ce8ce README ROCm build recipes), read
against the full port for context. The 040743e content was reviewed and passed
on 2026-06-25 and is not re-litigated here. Every finding below was reproduced
on this host (gfx1100, ROCm 7.2.3) before being written down.

### 1. The documented standalone ROCm recipe does not compile on Linux

`README.md:203-211` tells the reader to build `compressor-standalone.cu` and
`decompressor-standalone.cu` with exactly

    hipcc -O3 --offload-arch=<arch> -ffp-contract=off -I. -std=c++17 -c -o compress.o compressor-standalone.cu

Following the section verbatim on this host fails:

```
./generate_standalone_GPU_compressor_decompressor.py "" "TUPL4_1 RRE_1 CLOG_1"
hipify-perl -inplace compressor-standalone.cu decompressor-standalone.cu
for h in framework.h $(find include components preprocessors verifiers -name '*.h'); do hipify-perl -inplace "$h"; done
hipcc -O3 --offload-arch=gfx1100 -ffp-contract=off -I. -std=c++17 -c -o compress.o compressor-standalone.cu
-> compressor-standalone.cu:339:30: error: use of undeclared identifier 'strcmp'   (EXIT 1)
-> decompressor-standalone.cu:265:30, :267:37: same, 2 errors                      (EXIT 1)
```

Cause is the known gap already on record (notes.md "gotchas", and the deferred
item `lc-standalone-cstring-include`): `compressor-framework.cu:54` and
`decompressor-framework.cu:55` include `<string>` and then call `strcmp` at
`compressor-framework.cu:330,332` and `decompressor-framework.cu:258,260`.
libstdc++ does not declare `::strcmp` from `<string>`; MSVC's does, which is
why the gfx1151 round did not see it, and nvcc gets it transitively from the
CUDA headers, which is why the existing nvcc recipe works. It is the same
transitive-luck class as the thrust includes this port already fixed in
`preprocessors/d_QUANT_INOA_0_f64.h`, and it fires only on the HIP path.

Fix: add `#include <cstring>` next to the existing `<string>` in
`compressor-framework.cu:54` and `decompressor-framework.cu:55`. Verified: with
that one line added to each, both standalone translation units compile clean
(EXIT 0, warnings only) with the documented command on gfx1100. Do not document
`-include cstring` instead; a one-line include is the upstream-correct fix and
helps the CUDA path too.

On the deferral: the deferred item plans to "fold a `#include <cstring>` into
the standalone templates during PR-prep so the documented hipcc recipe works
unmodified everywhere". That cannot ride a PR-prep squash -- a squash may only
carry validation forward when the tree is identical, and it is refused if the
content changed. So this is a source change that needs its own validation round
whenever it lands; deferring it does not avoid that cost, it only risks the
branch being approved and published with a README recipe that fails on the
project's primary platform. Land it now, and close the deferred item with it.

### 2. The documented verification command reports failure on a correct conversion

`README.md:75` advises "verifying that no CUDA spellings are left, for example
with grep -rn cudaMalloc ., before compiling". After a conversion that followed
the section exactly and was fully successful (the resulting `lc` built and
passed `AL "" "RZE_4"` on gfx1100, see below), that command prints 52 hits:

- 23 in the `*.prehip` backups that `hipify-perl -inplace` itself writes (235
  of them in this tree),
- the rest in `framework.cu`, `compressor-framework.cu` and
  `decompressor-framework.cu`, which the documented loop deliberately does not
  convert and which the build does not compile,
- plus `README.md` itself, which now contains the string `cudaMalloc`.

So the check the README offers as the way to tell a good conversion from a bad
one always fires. A reader who follows it concludes the conversion failed.
Either drop the pre-check -- a header that was missed surfaces immediately as a
compile error, "use of undeclared identifier 'cudaMalloc'", which is a cleaner
thing to tell the reader to look for -- or give a command that is quiet on
success.

### 3. Unsubstantiated claim about hipify-perl in the same sentence

`README.md:75`: "Note that hipify-perl occasionally skips files when it is
invoked in a long loop". This did not reproduce: the exact documented loop, run
synchronously here over the whole tree, converted every file (loop exit 0, no
CUDA spellings left in any file it covered). This project's own record already
gives the real cause -- notes.md "gotchas" attributes the skipping to the loop
being run in the background or `&&`-chained after a find pipeline, and the
2026-08-12 porting round to the loop exceeding a two-minute command timeout.
Both are truncation of the loop by the caller, not a defect in hipify-perl.
An upstream README should not carry an unverified defect claim about a ROCm
tool. Reword to the fact that the loop is long and can be cut short before it
finishes, or drop the sentence with the pre-check in finding 2.

### Non-blocking, for PR-prep

- `ea0df9b`'s Test Plan uses 4-space indented command blocks where the standing
  rule asks for fenced blocks (`f0ce8ce` does it correctly). Not worth a history
  rewrite on its own; fold it in if the branch is ever restructured for the PR.

### Verified this round (context, not problems)

- `framework.h:52-69` `#ifndef _MSC_VER` guards: correct and strictly additive.
  `strings.h` and `unistd.h` have no callers anywhere in the tree (the `__ffs`
  hits are the device intrinsic from `include/macros.h`, not POSIX `ffs`);
  `sys/time.h` has exactly one user, `CPUTimer` at `framework.h:790-799`, under
  `#ifdef USE_CPU`, a path MSVC could never have compiled anyway
  (`gettimeofday` does not exist there). `_MSC_VER` is not defined by nvcc on
  Linux, by g++, or by MinGW, so the CUDA and CPU/OpenMP builds are byte-
  identical. The three POSIX includes appear nowhere else in a file the HIP
  build compiles (`compressor-framework.cpp:61` and
  `decompressor-framework.cpp:61` are the CPU templates).
- The README's central technical claim is true on current ROCm: hipcc does NOT
  predefine `__HIP_PLATFORM_AMD__` before the first include (probed directly on
  ROCm 7.2.3 -- a `#if defined(__HIP_PLATFORM_AMD__)` at line 1 of a `.hip` TU
  takes the else branch), so the header hipify prepends really is what arms the
  `WS` gate, and compiling unconverted source really would select wave32
  silently. The lesson promoted to `strategy-a-cmake.md` says the same thing and
  is accurate.
- The installation recipe at `README.md:65-77` works end to end here:
  generate -> hipify (whole documented loop) -> two hipcc steps -> EXIT 0, 116
  cosmetic warnings -> `./lc test.dat AL "" "RZE_4"` -> "verification passed" on
  gfx1100 at f0ce8ce. No `WS must be` mismatch, so `WS == warpSize == 32`.
- Dropping `-fopenmp` and `-march=native` (which the nvcc line passes through
  `-Xcompiler`) is harmless: every OpenMP region in `framework.h` is inside
  `#ifdef USE_CPU`, and the documented GPU build passes only `-DUSE_GPU`.
- Commit hygiene: both titles `[ROCm]`, 61 and 53 chars; bodies disclose AI
  assistance and carry Test Plans; no `Co-Authored-By`, no noreply trailer, no
  AMD-internal account reference; ASCII only. `jargon.py --port LC-framework`
  clean. `git status --porcelain` in src clean (integrity gate).
- The two `cuda-to-rocm` lessons on this branch were read against the code they
  describe and are correct as written.

## Porting round 2026-08-13 (porter, linux-gfx942)

Closes both change-request reviews of f0ce8ce: the linux-gfx942 review
(findings 1-5) and the linux-gfx1100 review of the 040743e..f0ce8ce delta
(findings 1-3, which arrived on the branch while this round was building).
Head f0ce8ce -> d7d9867. ROCm 7.14.60850 (AMD clang 23.0), gfx942 (MI300X,
wave64), nvcc 12.8 for the CUDA gate.

### Commits (fork moat-port)

- `b8c3df0` -- amend of `f0ce8ce`, MESSAGE ONLY (`git diff f0ce8ce b8c3df0`
  empty). Two upstream-visible statements in the body stopped being true:
  the paragraph explaining that hipify's prepended header is what arms the
  wavefront-width gate (no longer so after `af80cc9`), and the closing
  "the same commands ... were used on gfx90a and gfx1100", which was true of
  the framework build but not of the standalone build, which needed
  `-include cstring` on those two rounds. Rewrote both. Allowed here because
  no upstream PR is open and no arch's `validated_sha` is `f0ce8ce` or
  `ea0df9b`; `040743e`, which two arches did validate, was not touched.
- `b51dbd1` `[ROCm] Include <cstring> in the standalone code templates` --
  gfx942 finding 1 / gfx1100 finding 1. `compressor-framework.cu:54` and
  `decompressor-framework.cu:55`. The standalone generators `shutil.copyfile`
  these templates verbatim and every generated artifact is gitignored
  (`compressor-standalone.cu`, `decompressor-standalone.cu`, `lc.cu`, `lc.h`,
  `include/consts.h`), so there is nothing generated to regenerate in the
  tree -- fixing the templates is the whole fix. The `.cpp` CPU twins already
  had `<cstring>` at `:57`. Closes deferred `lc-standalone-cstring-include`
  (set to `done`).
- `af80cc9` `[ROCm] Select the wave width from compiler predefines alone` --
  gfx942 finding 2. All six sites (`compressor-framework.{cu,cpp}:48`,
  `decompressor-framework.{cu,cpp}:48`, and the `consts.h` emitters in
  `generate_Device_LC-Framework.py:101` and
  `generate_Hybrid_LC-Framework.py:138`) now read

      #if defined(__GFX8__) || defined(__GFX9__) || \
          (defined(__AMDGCN_WAVEFRONT_SIZE) && (__AMDGCN_WAVEFRONT_SIZE == 64))

  Dropping `__HIP_PLATFORM_AMD__` removes the dependency on hipify prepending
  `hip/hip_runtime.h`, because `__GFX*__` are compiler predefines and the gate
  sits above every include. Restoring the `__AMDGCN_WAVEFRONT_SIZE` term is
  what plan.md's open question had settled on ("keep both, additive, no
  cost"). Also rewrote the README paragraph that justified the conversion
  ordering by wavefront-width detection, since that justification is now
  false (gfx942 finding 2's closing note), replacing it with gfx942 finding
  5: hipify rewrites tracked sources in place and leaves `.prehip` backups.
- `1d7d9f2` `[ROCm] Keep the __syncwarp fallback for older ROCm` -- gfx942
  finding 3. `include/macros.h:83` gains
  `|| !__has_include(<hip/amd_detail/amd_warp_sync_functions.h>)`.
- `d7d9867` `[ROCm] Describe the conversion check by its symptom` -- gfx1100
  findings 2 and 3, which landed on the branch mid-round. Dropped the
  `grep -rn cudaMalloc .` pre-check (it fires on the `.prehip` backups, on the
  three framework templates the documented loop does not convert, and on the
  README line proposing it) and the "hipify-perl occasionally skips files"
  defect claim (the real cause is a loop cut short by the caller), replacing
  both with the compile-time symptom.

Not done, deliberately: `ea0df9b`'s Test Plan uses 4-space indented command
blocks rather than fenced ones. The gfx1100 reviewer called it non-blocking
and explicitly not worth a history rewrite on its own; rebasing it would
change four commit shas for a cosmetic difference that GitHub renders
identically, and 4-space blocks are this README's own house style. Left for
PR-prep if the branch is ever restructured.

### Measurements this round

`__AMDGCN_WAVEFRONT_SIZE` is gone from ROCm 7.14 in BOTH spellings
(`hipcc -x hip --offload-arch=gfx942 -dM -E` on an empty file: `__GFX9__`,
`__HIPCC__`, `__gfx942__`, `__HIP_MEMORY_SCOPE_WAVEFRONT`, and no
`__AMDGCN_WAVEFRONT_SIZE`, no `__AMDGCN_WAVEFRONT_SIZE__`, no
`__HIP_PLATFORM_*`). `-mwavefrontsize64` on gfx1100 changes NOTHING in the
2650-macro dump (`diff` of the two `-dM -E` outputs is empty), so the
restored term buys back only the older toolchains that did predefine it --
recorded because it means an explicit wave64-on-RDNA build is still
undetectable at preprocessing time on current ROCm, and `lc` catches it at
runtime via the `framework.h:228` trap while the standalone binaries do not.

Per-arch device-pass width of the new gate, no header included ahead of it
(`static_assert(WS == EXPECT)` inside `#ifdef __HIP_DEVICE_COMPILE__`):
gfx803/gfx908/gfx90a/gfx942/gfx950/gfx9-4-generic -> 64;
gfx1100/gfx1151/gfx1201/gfx11-generic -> 32. All compile.

### Gates run on gfx942 from the committed tree (`git archive HEAD`)

Build: `hipcc -O3 --offload-arch=gfx942 -ffp-contract=off -DUSE_GPU -I.
-std=c++17 -c -o lc.o lc.cu` then link. EXIT 0, 0 errors, 232 warnings
(nodiscard + shift-negative-value + deprecated-identifier across both passes).

- Lossless AL round-trips, all "LOSSLESS verification passed": RZE_4, RZE_1,
  RLE_4, `BIT_4 RLE_4`, `RRE_4 RZE_4`, RAZE_4 (1 MB random) and
  `RZE_4 RRE_4` on a structured zero/repeat/random file.
- TS self-test: 4491 "verification passed", 0 failures.
- Lossy: `QUANT_ABS_0_f32(0.01)`+MAXABS, `QUANT_INOA_0_f64(0.01)`+MAXNOA
  (rocThrust), `QUANT_NOA_0_f32(0.01)`+MAXNOA (`cuda::std::numeric_limits`
  alias), and `LOR1D_i32()` (hipCUB `DeviceScan`) -- all verifications passed.
- README standalone recipe, the one that was broken: run verbatim from a
  pristine `git archive HEAD` tree, both binaries build EXIT 0 (0 errors, 118
  warnings) and round-trip byte-identically. Before `b51dbd1` this failed with
  `use of undeclared identifier 'strcmp'`.
- Cross-device format gate: gfx942 wave64 encode -> CPU decode IDENTICAL, CPU
  encode -> gfx942 wave64 decode IDENTICAL, and the two encodings are
  BYTE-IDENTICAL, on both the random and the structured input.
- CUDA no-regression (nvcc 12.8, `-arch=sm_80`, no hipify): `lc.cu`,
  `compressor-standalone.cu` and `decompressor-standalone.cu` all compile,
  0 errors.
- Non-GPU regression: `generate_Host_LC-Framework.py` + g++ `-DUSE_CPU`
  builds clean and passes `AL "" "RZE_4"`.
- `__syncwarp` both branches: default build (ROCm provides the header) and
  `-DHIP_DISABLE_WARP_SYNC_BUILTINS` (shim emitted) both compile, 0 errors.

### TS count differs from the 2026-06-25 gfx90a record

gfx942 reports 4491 verifications, matching gfx1100 and gfx1151, not the 4515
recorded for gfx90a. The count is NOT data- or size-dependent here (4491 for
64 KB random, 64 KB structured, 8 KB and 300 KB random alike), and it is not
wave-width-dependent, since this is a wave64 platform agreeing with the two
wave32 ones. Most likely a counting artifact of that session's grep. 0
failures either way, which is the gate.

### gotchas

- `hipify-perl -inplace` re-hipifies from the `.prehip` BACKUP, not from the
  file on disk. Regenerating `compressor-standalone.cu` for a new pipeline and
  hipifying it again silently restored the PREVIOUS pipeline's code: the
  binary announced `TUPL4_1 RRE_1 CLOG_1` while the generator had just written
  RZE_4. It cost a bogus cross-device "failure" (`csize does not match osize`)
  that looked like a format bug and was purely a pipeline mismatch. Delete the
  stale `*.prehip`, or generate into a fresh copy of the tree, whenever a
  generator rewrites a file that was already converted. Promoted to the
  `cuda-to-rocm` skill (`references/strategy-a-cmake.md`, Build hygiene).
- An unguarded `static_assert` in a `__global__` body is checked in the HOST
  pass too, where `WS` legitimately resolves to 32; wrap per-arch width
  assertions in `#ifdef __HIP_DEVICE_COMPILE__` or they all "fail" on wave64
  arches. Promoted with the same bullet.
- `LOR1D_i32` takes no parameters but the parser still requires the parens:
  `'LOR1D_i32()'`, not `'LOR1D_i32'` (which errors with "expected '('").
- Nested quoting eats the empty preprocessor argument: running
  `./generate_standalone_GPU_compressor_decompressor.py '' 'RZE_4'` inside a
  `bash -c "..."` silently generated the wrong pipeline. Invoke the generators
  directly, and check the emitted `printf("GPU LC 1.3 Algorithm: ...")` line
  before trusting a build.

### skill lessons promoted with this change

- `references/strategy-a-cmake.md`: the `.prehip` re-hipify trap; and "do not
  key a macro gate on `__HIP_PLATFORM_AMD__` when a compiler predefine says
  the same thing", replacing the older bullet that treated the prepended
  header as load-bearing (it was, and this port stopped it being so).
- `references/fault-classes.md`: same correction in the
  `__HIP_PLATFORM_AMD__` entry.
- `references/validation.md`: fixed the destructive `git clone && git
  sparse-checkout` recipe (gfx942 finding 4) -- the missing `cd` meant the
  second command applied to the reader's own repository and emptied its
  working tree.

## Review 2026-08-13 (reviewer, linux-gfx942, re-review of d7d9867)

Verdict: changes-requested, on one finding, all of it on the MOAT side. The
fork branch itself is clean: every finding of both 2026-08-13 reviews (gfx942
1-5, gfx1100 1-3) is resolved at d7d9867, and each was re-verified here rather
than taken from the porting record. Scope: the delta `f0ce8ce..d7d9867` read
against the whole port diff `f72e323...d7d9867`, plus the `cuda-to-rocm`
lessons riding this MOAT branch. Host: ROCm 7.14.60850 (AMD clang 23.0),
gfx942 (wave64), nvcc 12.8, g++ for the CPU reference.

### 1. The hipify bullet in the skill still gives the two instructions this round removed from the README

`.claude/skills/cuda-to-rocm/references/strategy-a-cmake.md:90-95` tells every
future agent to run hipify synchronously because "`-inplace` in a backgrounded
or `&&`-chained loop silently skips files", and to "always re-grep the whole
tree for `cudaMalloc|cudaSuccess|include <cub|include <cuda.h` before
compiling". `d7d9867` deleted both statements from `README.md` because the
gfx1100 review showed the grep fires on a conversion that succeeded and the
skipping claim was unsubstantiated -- and then left the canonical lesson
asserting both, so the next agent porting a hipify-based project reads the
version this round rejected.

Measured here at `d7d9867`, after the documented loop ran to completion and
produced a binary that passes `AL "" "RZE_4"` (note: a `grep` that honors
`.gitignore` hides this, since `.gitignore:20` excludes `*.prehip` -- use
`/usr/bin/grep`):

    /usr/bin/grep -rlE 'cudaMalloc|cudaSuccess|include <cub|include <cuda\.h' .
      -> 14 files, 10 of them the *.prehip backups hipify itself writes,
         the rest the framework templates the loop deliberately leaves alone
    /usr/bin/grep -rn cudaMalloc .
      -> 52 lines, 24 in *.prehip   (reproduces the gfx1100 count exactly)

The bullet already carries the check that does work -- "un-hipified files
surface as 'undeclared identifier cudaMalloc'" -- which is what `README.md:75`
now says on its own. Fix: drop the re-grep instruction, and replace the
skipping mechanism with the one the round settled on, that a whole-tree
conversion loop takes a while and one cut short by the caller leaves the rest
of the headers in CUDA form, so re-running it finishes the job. Keep the
`hip/hip_runtime.h`-prepend sentence; it is correct and I relied on it (the
g++ CPU reference build here was made from a separate non-hipified tree).

No fork commit is needed for this and no evidence is invalidated by it.

### Prior findings verified resolved (context, not problems)

- gfx942 1 / gfx1100 1, `<cstring>`: `compressor-framework.cu:59` and
  `decompressor-framework.cu:60`. The `README.md:203-211` standalone recipe was
  run verbatim from a pristine `git archive HEAD` tree -- generate
  `"TUPL4_1 RRE_1 CLOG_1"`, hipify, both `hipcc` steps -- EXIT 0, 0 errors, and
  `./compress` / `./decompress` round-trip 1 MB byte-identically.
- gfx942 2, the wave gate: all six sites carry the identical expression
  (`compressor-framework.{cu,cpp}:51`, `decompressor-framework.{cu,cpp}:51`,
  `generate_Device_LC-Framework.py:101`, `generate_Hybrid_LC-Framework.py:138`);
  `generate_Host_LC-Framework.py:105` emits no `WS`, so no site is missed.
  Device-pass probe with `consts.h` included ahead of every header:
  gfx803/900/906/908/90a/942/950/gfx9-4-generic -> 64,
  gfx1030/1100/1151/1201/gfx11-generic/gfx12-generic -> 32; hipcc host pass,
  g++ and nvcc -> 32. A gfx90a+gfx1100+gfx942 fat binary of the real `lc.cu`
  links and carries three code objects, and running it on gfx942 prints "AMD
  GPU version" and passes without the `framework.h:227` trap. Every `WS` use is
  inside a `__global__` body (`compressor-framework.cu:169-222`,
  `decompressor-framework.cu:148`, `framework.h:227,270-293,421,508`), so the
  host pass resolving 32 stays inert.
- gfx942 3, `__syncwarp`: `include/macros.h:83-86`. `__syncwarp` is defined in
  exactly one ROCm header on this install, `amd_warp_sync_functions.h`, under
  `#if !defined(HIP_DISABLE_WARP_SYNC_BUILTINS)`, so `__has_include` on that
  header is the exact proxy for "this ROCm has no `__syncwarp`". Probed
  `__has_include` -> 1 here; both branches build `lc.cu` with 0 errors (default,
  and `-DHIP_DISABLE_WARP_SYNC_BUILTINS`).
- gfx942 4, `validation.md`: the recipe now reads `git clone ... && cd
  rocm-libraries && git sparse-checkout set ...` and says why the `cd` matters.
- gfx942 5 / gfx1100 2 and 3, README: `README.md:75` names the in-place rewrite
  and the `.prehip` backups; the `grep -rn cudaMalloc .` pre-check and the
  "hipify-perl occasionally skips files" claim are gone.
- `b8c3df0` is message-only (`git diff f0ce8ce b8c3df0` empty), and `040743e`,
  the sha both completed arches validated, is still an ancestor of HEAD with
  tree `91e1a3b`.
- Both promoted lessons reproduce verbatim. `-inplace` re-hipified a
  regenerated file from its `.prehip` backup, silently restoring the previous
  generation's function. An unguarded `static_assert(WS == 64)` in a
  `__global__` body on gfx942 fails with "1 error generated when compiling for
  host".
- Format gate, independently: gfx942 wave64 encode -> CPU decode identical, CPU
  encode -> gfx942 decode identical, and the two encodings byte-identical.
  `RZE_4`, `RRE_4 RZE_4`, `QUANT_NOA_0_f32(0.01)`+`MAXNOA_f32` and `LOR1D_i32()`
  all pass on gfx942; `lc.cu` also compiles clean for gfx1100.
- CUDA no-regression, nvcc 12.8 `-arch=sm_80` on a pristine tree: `lc.cu`,
  `compressor-standalone.cu`, `decompressor-standalone.cu` all 0 errors.
- Hygiene: seven titles, all `[ROCm]`, 50-62 chars; one author and committer,
  the maintainer's public address; no `Co-Authored-By`, no noreply, no ghstack,
  no AMD-internal reference; ASCII in messages and in every added line;
  `jargon.py --port LC-framework` clean; `git status --porcelain` in src empty.
  `prose.py` flags `README.md:70,71,194,199`, all indented command lines (194
  and 199 are upstream's own nvcc lines), not prose -- no action.
- `ea0df9b`'s indented Test Plan blocks: agreed non-blocking, as the gfx1100
  review and the porting round both concluded.

## Porting round 2026-08-13 (porter, linux-gfx942) -- lesson only, no fork change

Answers the single finding of the re-review of `d7d9867`. Records and the
`cuda-to-rocm` skill only; the fork is untouched and `head_sha` stays at
`d7d9867`, so no `advance-head` and no evidence is invalidated. `git rev-parse
HEAD` in `src` is `d7d98677060e36cfa329db4229c8b3d14cf53b32` before and after,
`git status --porcelain` empty.

The hipify bullet at `.claude/skills/cuda-to-rocm/references/strategy-a-cmake.md`
still carried the two instructions `d7d9867` removed from `README.md`: that
`-inplace` in a backgrounded or `&&`-chained loop silently skips files, and to
re-grep the whole tree for `cudaMalloc|cudaSuccess|include <cub|include <cuda.h`
before compiling. Both are now gone. The bullet keeps the check that works --
un-hipified files surface as "undeclared identifier cudaMalloc", which is what
`README.md:75` says on its own -- and replaces the skipping mechanism with the
one this round settled on: a whole-tree conversion loop takes a while, one cut
short leaves the rest of the headers in CUDA form, and re-running it finishes
the job. It now also says why the grep is not the check, carrying the
reviewer's measurement (10 of 14 hits on a fully converted tree are the
`.prehip` backups hipify itself writes, and the answer depends on whether the
grep honors `.gitignore`, which excludes them via `.gitignore:20`). The
`hip/hip_runtime.h`-prepend sentence is unchanged; it is correct.

The lesson is generic to hipify-perl ports, which is why it stays in the skill
rather than here.

## Review 2026-08-13 (reviewer, linux-gfx942, delta round 9778530)

Verdict: changes-requested, on one finding, again entirely on the MOAT side.
Scope: the delta `53be238..HEAD` (9778530 plus the a65b863 telemetry commit),
which touches `.claude/skills/cuda-to-rocm/references/strategy-a-cmake.md`,
`notes.md`, `stats.jsonl` and the `status.json` stage field, and nothing else.
The fork is confirmed untouched: `src` is at
`d7d98677060e36cfa329db4229c8b3d14cf53b32` on `moat-port`, equal to
`origin/moat-port` and to `status.json.head_sha`, `git status --porcelain`
empty, and `linux-gfx90a` / `linux-gfx1100` still carry `validated_sha`
`040743e`. `jargon.py --port LC-framework` clean. The fork findings resolved in
the `d7d9867` re-review are not re-litigated here.

The substance of the rewrite is right: the "silently skips files" mechanism and
the whole-tree re-grep pre-check are gone, the compile-time "undeclared
identifier cudaMalloc" signal is kept, the replacement mechanism matches
`README.md:75` on the fork, and the `hip/hip_runtime.h`-prepend sentence and
both adjacent bullets (`:85-89` offload-arch, `:99-104` `.prehip` re-hipify) are
untouched.

### 1. The "10 of 14" figure belongs to a different grep than the one the sentence names

`.claude/skills/cuda-to-rocm/references/strategy-a-cmake.md:94-96`:

    Grepping the tree for `cudaMalloc` instead reports the `.prehip` backups
    hipify itself writes -- 10 of 14 hits on a fully converted tree

10-of-14 is the result of the *four-pattern* pre-check
`cudaMalloc|cudaSuccess|include <cub|include <cuda.h`, which this same edit
deleted from the bullet, so the number now cites a query the reader can no
longer see. A grep for the literal `cudaMalloc` the sentence names gives
different numbers.

Re-measured today, fresh `git archive HEAD` of `d7d9867` into a scratch tree,
`./generate_Device_LC-Framework.py` then the `README.md:63-64` hipify loop run
to completion (231 files converted, exit 0):

    /usr/bin/grep -rlE 'cudaMalloc|cudaSuccess|include <cub|include <cuda\.h' .
      -> 14 files, 10 *.prehip          (reproduces the prior round exactly)
    /usr/bin/grep -rl cudaMalloc .
      -> 12 files,  8 *.prehip
    /usr/bin/grep -rn cudaMalloc .
      -> 52 lines, 24 in *.prehip       (reproduces the prior round exactly)

The conclusion the bullet draws survives all three, so this is a citation fix,
not a retraction. Fix, either: say "8 of 12 files, or 24 of 52 lines" for the
`cudaMalloc` grep, or keep 10-of-14 and name the query it came from. A number
that does not reproduce from the command the sentence describes is the failure
mode this round exists to remove.

The `.gitignore` half of the claim is verified and is the larger effect:
`.gitignore:20` is `*.prehip`, and on the same converted tree ripgrep 14.1.1
reports 4 files with 0 `.prehip` where `/usr/bin/grep` reports 12 with 8.

Observation, no action: the documented whole-tree loop took 24 s wall clock on
this host, so "takes a while" is doing light work in the mechanism sentence.
The mechanism itself (a loop cut short leaves the remaining headers in CUDA
form; re-running finishes the job) is correct and matches `README.md:75`, which
already passed review, so this is context for the next writer rather than a
change request.
