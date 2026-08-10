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
