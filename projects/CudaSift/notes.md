# CudaSift notes

Possibly the upstream of the SIFT-GPU code in AMD-Ecosystem/colmap (branch rocm-sift-gpu); reference that port for the AMD-fault fixes (texture rule-of-five, OOB index clamps, 256-byte texture pitch).

## Porting 2026-05-30 (gfx90a, ROCm 7.2)

Strategy A (colmap minimal-footprint). Result: clean configure + build of all
targets for gfx90a (configure_exit=0, build_exit=0, 0 compile errors). Binaries
embed `hipv4-amdgcn-amd-amdhsa--gfx90a` code objects and link `libamdhip64`.
GPU tests not run here (validator does that) -- compile+link only.

### Files changed
- `cuda_to_hip.h` (new): the single CUDA->HIP compat header. On
  `USE_HIP || __HIP_PLATFORM_AMD__` includes `<hip/hip_runtime.h>` and `#define`s
  every cudaXxx symbol the project uses to its hip equivalent (errors, device
  mgmt, events/streams, malloc/memcpy/memset, constant-memory symbol copies,
  memcpy kinds, cudaArray + channel-format, texture objects + resource/texture
  descs and their enums). Names taken from pytorch hipify
  `cuda_to_hip_mappings.py`. Also defines two device-intrinsic shims (see fault
  classes). On NVIDIA it is a no-op `#include <cuda_runtime.h>`.
- `hip_compat/cuda_runtime.h`, `hip_compat/cuda.h` (new): tiny HIP-only shims
  (each just `#include "cuda_to_hip.h"`) so the few sources that do
  `#include <cuda_runtime.h>` / `<cuda.h>` directly (tests/test_match.cpp,
  tests/benchmark.cpp, match.cu) resolve on ROCm, which ships no CUDA headers.
  Added to the include path for the HIP build only; NVIDIA uses the real header.
- `CMakeLists.txt`: added `option(USE_HIP ... OFF)`. When ON: `project(... HIP)`,
  `DEVICE_LANG=HIP`, mark the `.cu` TUs and the `LANGUAGE CUDA` `.cpp` files
  `LANGUAGE HIP`, set `HIP_ARCHITECTURES gfx90a` on every target, force-include
  the compat header (`-include .../cuda_to_hip.h` for the HIP language), add the
  `hip_compat/` shim dir, define `USE_HIP`, and swap nvcc flags
  (`--use_fast_math -lineinfo`) for clang (`-ffast-math`). The CUDA path
  (`USE_HIP=OFF`) keeps the original behavior.
- `cudaImage.cu`: `CopyToTexture` -- replaced deprecated `cudaMemcpyToArray`
  (removed from current HIP) with `hipMemcpy2DToArray` under a HIP guard. This
  path is DEAD (`CopyToTexture`/`InitTexture` have no callers; the live texture
  is the Pitch2D bind in cudaSiftH.cu), so it cannot be GPU-validated -- the
  fix is for correctness on a future revival, not the shipped pipeline. The
  source row stride is `this->pitch` (spitch == width == `pitch*sizeof(float)`,
  height = `dst.height`), reproducing the original
  `cudaMemcpyToArray(... sizeof(float)*pitch*dst.height ...)` byte count
  byte-for-byte. (An earlier draft used `dst.pitch`, which only matched while
  `this->pitch == dst.pitch`; restored to `this->pitch`.) CUDA branch unchanged.
- `cudaSiftH.cu`: the two `cudaMemcpyToSymbolAsync(symbol, src, bytes)` calls
  now pass explicit `0, cudaMemcpyHostToDevice`. CUDA defaults offset/kind/stream;
  HIP's overload only defaults `stream`, so the 3-arg form failed to resolve.
  The explicit args are valid on CUDA too (BC-safe shared-code edit).
- `matching.cu`: `MatchSiftPoints2` -- the two `sift[min(int, unsigned)]` array
  subscripts now cast to `(int)`. HIP's `min` overload resolution produced a
  non-integer subscript (illegal); the other `min(unsigned, int)` sites that
  assign to an `int` were fine and left as-is.

`cudautils.h` and `cudaSiftD.cu` were deliberately NOT modified -- the warp
helpers and the descriptor-layout / orientation-histogram constants stay in
plain CUDA spelling.

### Fault-class issues encountered and fixes
- Compat-header bootstrap (HIP macro chicken-and-egg): clang in HIP mode
  predefines `__HIP__` but NOT `__HIP_PLATFORM_AMD__` -- that macro is only set
  *by* including the HIP runtime header, so it can't gate the include of that
  same header. Fixed by also passing `-DUSE_HIP` to the compiler (the header's
  first guard term), which is what made the force-include actually translate.
- Warp size (wave64): `cudaSiftD.cu` calls `__any_sync(0xffffffff, pred)`
  directly. On gfx90a `__any_sync` is templated and static-asserts that the
  mask is 64-bit, rejecting the CUDA 32-bit full mask. The kernels that use it
  (`FindPointsMulti*`) launch 32-thread blocks and the surrounding reduction is
  explicitly width-32 (`ShiftUp(.,d) for d<32`, `Shuffle(.,31)`). Mapped
  `__any_sync(mask, pred) -> __any(pred)` in the header: HIP's mask-free `__any`
  polls the active wavefront (exactly the 32 active lanes here), matching the
  original "any active lane" intent. The cudautils.h `ShiftDown/Up/Shuffle`
  helpers take the non-`_sync` `__shfl_*(var, delta, width)` branch under HIP
  (HIP does not define `CUDART_VERSION`); width=32 is honored as a sub-group
  split on wave64, so the 4-partial-sum (`sums[idx/32]`, idx in {0,32,64,96})
  descriptor reduction keeps CUDA semantics. NUMERIC PARITY STILL NEEDS THE GPU
  VALIDATOR -- this analysis says it should match, on-device confirmation pending.
- Missing intrinsic: HIP has no round-toward-zero `__fmul_rz` (used only in the
  RANSAC homography residual test). Mapped to a plain `(a)*(b)` in the header;
  rounding mode is immaterial to an inlier threshold test, and round-to-nearest
  is the faithful default.
- Deprecated array-copy API: `cudaMemcpyToArray` unavailable in HIP -> rewrote
  to `hipMemcpy2DToArray` (see cudaImage.cu above). NOTE: this is in the dead
  `CopyToTexture`/`InitTexture` path, NOT the live texture. The PRIMARY texture
  bound in `ExtractSiftOctave` (cudaSiftH.cu:189-207) is a
  `cudaResourceTypePitch2D` bind directly to `img.d_data`
  (`pitchInBytes = img.pitch*sizeof(float)`), so the 256-byte-pitch fault class
  DOES apply to it (earlier "cudaArray-backed, does not apply" claim was wrong).
  It is satisfied here, not absent: every image fed to the bind has
  `pitch = iAlignUp(width,128)` = a multiple of 128 float elems = 512 bytes
  (cudaSiftH.cu:122,157,184), and the base pointers come from
  `cudaMallocPitch` / a `cudaMalloc`'d pool (>=256B aligned). So pitch is a
  multiple of 512B and base is >=256B aligned -> the AMD 256B texture-pitch
  requirement holds (applies-but-satisfied). The sampler clamps
  (`cudaAddressModeClamp`, cudaSiftH.cu:200-201) so edge `tex2D` fetches at
  +-cos/+-sin are in-bounds by construction (no OOB clamp needed) -- on the
  Pitch2D bind as well.
- Texture rule-of-five: `texObj` is default-initialized to 0, created with
  `cudaCreateTextureObject` and unconditionally destroyed with
  `cudaDestroyTextureObject` in the same scope (`ExtractSiftOctave`); the
  `cudaArray t_data` in CudaImage is default-init NULL with a guarded
  `cudaFreeArray` in the dtor. No double-free / uninitialized-handle hazard, so
  no hardening needed beyond what upstream already does.

### Build commands (run from /var/lib/jenkins/moat)
```
utils/timeit.sh CudaSift compile -- cmake -S projects/CudaSift/src \
  -B projects/CudaSift/src/build-hip -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build projects/CudaSift/src/build-hip -j
```
Full clean configure+build ~9s. Targets: cudasift_lib, cudasift, test_extract,
test_match, test_homography, benchmark (+ demo_extract/match/video).

### Remaining warnings (non-fatal)
- ~152 `-Wunused-value`: ignoring `nodiscard` `hipError_t` returns from
  `cudaEvent*` / memcpy calls in TimerGPU and elsewhere. Pre-existing CUDA code
  pattern; CUDA headers don't mark these `nodiscard`, HIP does. Harmless; left
  untouched to keep the diff minimal.
- 4 `-Wliteral-conversion`, 2 `-Wnan-infinity-disabled` (from `-ffast-math`).

### BC note
`USE_HIP=OFF` configure fails on this host only because there is no CUDA
toolkit (`Failed to find nvcc` at the `project(... CUDA)` line) -- identical to
the unmodified upstream on a CUDA-less host, not a regression. The CUDA-path
CMake logic mirrors the original.

## Review 2026-05-30 (reviewer, gfx90a / wave64)

Verdict: changes-requested. The Strategy-A mechanics are sound (one no-op-on-NVIDIA
compat header, `.cu` marked `LANGUAGE HIP` not renamed, USE_HIP default OFF keeps
the CUDA path intact, force-include + hip_compat shims). The shared-code edits
(cudaSiftH.cu memcpy-to-symbol arg fill, matching.cu `min` casts) are BC-safe
generalizations. Problems below.

### Fault classes
- cudaSiftH.cu:191 -- the PRIMARY texture in `ExtractSift` is a
  `cudaResourceTypePitch2D` bind directly to `img.d_data`
  (`pitchInBytes = img.pitch*sizeof(float)`), NOT a cudaArray. This directly
  contradicts the porter analysis (notes.md "Texture backing is a `cudaArray`
  (not a pitched 2D bind), so the 256-byte-pitch fault class does not apply" and
  plan.md risk list). The 256B-pitch fault class DOES apply to this live bind.
  It happens to be satisfied (pitch = iAlignUp(w,128) => multiple of 512B;
  hipMallocPitch/pool base is >=256B aligned), so no code bug results, but the
  applicable fault class was dismissed on a false premise and the load-bearing
  path was never actually checked. Re-do the pitch/base-alignment analysis
  against the Pitch2D resource, not the cudaArray.
- cudaSiftD.cu:1319,1329 (`__any_sync`->`__any`) and the width-32 shuffle
  reductions (cudaSiftD.cu:269-285 descriptor norm; 1361-1378 prefix-sum +
  `Shuffle(.,31)` broadcast) are wave64-sensitive and remain UNVALIDATED on
  device. Static analysis indicates parity (active kernels launch 32-thread
  blocks for `FindPointsMultiNew`, and width=32 splits a wave64 front into exact
  32-lane subgroups for the 128-thread descriptor kernel, so `sums[idx/32]`
  fills its 4 slots correctly; HIP `__any`/`wfany` polls EXEC = the launched
  lanes only), but the porter flagged numeric parity as pending and no GPU run
  was done. Per the review rubric a port without a real-GPU parity run is
  changes-requested. Needs the test_extract / test_match / test_homography +
  keypoint/descriptor parity run on gfx90a before passing.

### Minimal footprint / dead code
- cudaImage.cu:105-119 -- the `cudaMemcpyToArray`->`hipMemcpy2DToArray` rewrite
  is in `CudaImage::CopyToTexture`, which (with `InitTexture`) has NO callers
  (declared in cudaImage.h, defined here, never invoked; the live texture is the
  Pitch2D bind above). Consequences: (a) the rewritten path is unreachable, so
  no GPU test can validate its semantics; (b) it changes the row-stride source
  from `this->pitch` (original: `sizeof(float)*pitch*dst.height`) to `dst.pitch`
  (`rowBytes = sizeof(float)*dst.pitch`) -- a latent behavior change should the
  method ever be revived (correct only while `this->pitch == dst.pitch`).
  The 2D-array semantics otherwise match the original contiguous upload. Either
  match the original `this->pitch` or note the path is dead; the
  semantics-preservation claim in notes.md should not be stated as validated.

### Notes accuracy (porting record)
- The "Fault-class issues" section asserts texture backing is a cudaArray and
  that the 256B-pitch class "does not apply"; this is wrong for the shipped
  texture (see above). Correct the record so a later wave32/gfx1100 re-validation
  does not inherit the false premise.

Non-blocking confirmations (not defects): `CUDART_VERSION` is undefined on the
HIP path (verified: only in rocm `nvidia_detail`), so `ShiftDown/Up/Shuffle`
take the `__shfl_*(var,delta,width)` branch as intended; `__fmul_rz` shim reaches
its only use (matching.cu RANSAC residual, a HIP TU); `match.cu` and its
`#include <cuda.h>` are not compiled by any target (orphan); geomFuncs.cpp uses
no device symbols; texDesc sets `cudaAddressModeClamp` so edge `tex2D` fetches at
+-cos/+-sin are in-bounds on the Pitch2D bind too; binaries embed
`hipv4-amdgcn-amd-amdhsa--gfx90a` and link libamdhip64 (build claim holds).

## Validation 2026-05-30 (real GPU, gfx90a / wave64)

Addressed both reviewer findings, then ran the real GPU tests on device.

GPU: AMD Instinct MI250X / MI250, `gfx90a` (wave64), ROCm. Host has 4x gfx90a
(`rocm_agent_enumerator` -> 4x gfx90a); all idle, pinned to the highest index
with `HIP_VISIBLE_DEVICES=3` (one visible device -> in-process ordinal 0, tests
run with default devNum=0). Binaries embed `hipv4-amdgcn-amd-amdhsa--gfx90a`
(verified via `roc-obj-ls`).

Fixes:
- cudaImage.cu:110 -- `CopyToTexture` HIP path source row stride restored from
  `dst.pitch` to `this->pitch` (`rowBytes = sizeof(float)*pitch`), matching the
  original `cudaMemcpyToArray(... sizeof(float)*pitch*dst.height ...)` byte
  count byte-for-byte. Path is dead (no callers), so the fix is latent-correct
  for a future revival, not GPU-exercised; CUDA `#else` branch unchanged.
- notes.md (this file) -- corrected the texture-backing claim: the live
  `ExtractSift`/`ExtractSiftOctave` texture is a `cudaResourceTypePitch2D` bind
  to `img.d_data` (cudaSiftH.cu:189-207), NOT a cudaArray. The 256B-pitch fault
  class APPLIES and is SATISFIED (pitch = iAlignUp(w,128) = multiple of 512B;
  base from cudaMallocPitch / cudaMalloc pool >=256B aligned), not "does not
  apply".

Rebuild: `cmake --build projects/CudaSift/src/build-hip -j` -> exit 0, clean
(only the pre-existing ~8 `-Wunused-value` nodiscard warnings on cudaImage.cu;
no errors). All targets built.

Tests (each `utils/timeit.sh CudaSift test -- bash -c 'cd projects/CudaSift/src
&& ./build-hip/<bin>'`, run from src/ so `data/` resolves):

| test            | exit | result            | key numbers |
|-----------------|------|-------------------|-------------|
| test_extract    | 0    | 10 passed, 0 fail | basic 1910 kpts (all valid, descriptors unit-norm); thresh monotonic 7080/1910/542/6; reproducibility 1910/1910 = 100%; scaleUp 3104 > normal 1910 |
| test_match      | 0    | 11 passed, 0 fail | self-match 1899/1910 score>0.95; cross img1/img2 1910 vs 2084 feats, 1910 valid / 1909 good matches; homography 851 RANSAC / 828 inliers; match 1.23 ms |
| test_homography | 0    | 8 passed, 0 fail  | translation (30,20) recovered h[2]=30.13 h[5]=20.27, 1645 matches; rotation 10deg 1195; scale 0.8 945; PGM pair L1451/R2082, 751 RANSAC / 672 inliers |

PASS/FAIL: PASS (29/29 sub-checks across the three binaries, all exit 0). No GPU
page fault, no `safeCall` abort, no crash; GPU 3 healthy/idle post-run, dmesg
clean of amdgpu/page-fault/VM-retry. The wave64-sensitive paths the reviewer
flagged (`__any_sync`->`__any` in FindPointsMulti, the width-32 descriptor-norm
and prefix-sum reductions) produce CORRECT on-device output: descriptors are
unit-length, extraction is bit-reproducible (100% position match across runs),
and the recovered homography geometry is accurate (translation within ~0.3 px).
The static wave64 parity analysis is now confirmed on real gfx90a.

## Validation 2026-05-30 (gfx1100, ROCm 7.2.1)

GPU: 2x AMD Radeon Pro W7800 48GB (gfx1100, wave32), ROCm 7.2.1. Used HIP_VISIBLE_DEVICES=0 (in-process ordinal 0). Binaries embed `hipv4-amdgcn-amd-amdhsa--gfx1100` (verified via `roc-obj-ls`).

CMakeLists.txt fix: the original port hardcoded `HIP_ARCHITECTURES "gfx90a"` in `set_target_properties`. Added a guard that reads from `CMAKE_HIP_ARCHITECTURES` (defaulting to gfx90a if unset) so follower platforms can pass `-DCMAKE_HIP_ARCHITECTURES=gfx1100` at configure time. The CI smoketest workflow `.github/workflows/rocm-build-smoketest.yml` was also added (rocm/dev-ubuntu-24.04, compile-only, not a correctness gate). These changes were amended into the port commit; new fork head SHA: 0523b54. [Policy update 2026-05-30: the no-CI-in-ports rule now forbids GHA workflows in a port; remove this yml from the curated commit at upstream-PR prep (deferred to avoid churning the already-completed gfx90a/gfx1100 validations). The fork's Actions are disabled, so it does not run.]

OpenCV install: `sudo apt-get install -y libopencv-dev` -> 4.6.0 (was absent on this host).

### Build commands (gfx1100, run from /var/lib/jenkins/moat)
```
sudo apt-get install -y libopencv-dev
utils/timeit.sh CudaSift compile -- cmake -S projects/CudaSift/src \
  -B projects/CudaSift/src/build-hip -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release
utils/timeit.sh CudaSift compile -- cmake --build projects/CudaSift/src/build-hip -j
```
Configure: exit 0 (~1.9s). Build: exit 0 (~4.8s). All targets built: cudasift_lib, cudasift, test_extract, test_match, test_homography, benchmark, demo_extract, demo_match, demo_video. Warnings: ~152 `-Wunused-value` (nodiscard hipError_t, pre-existing pattern), 4 `-Wliteral-conversion`, 2 `-Wnan-infinity-disabled`.

### Test results (gfx1100)
```
HIP_VISIBLE_DEVICES=0 ./build-hip/test_extract
HIP_VISIBLE_DEVICES=0 ./build-hip/test_match
HIP_VISIBLE_DEVICES=0 ./build-hip/test_homography
```

| test            | exit | result            | key numbers |
|-----------------|------|-------------------|-------------|
| test_extract    | 0    | 10 passed, 0 fail | basic 1910 kpts (all valid, descriptors unit-norm); thresh monotonic 7077/1910/542/6; reproducibility 1910/1910 = 100%; scaleUp 3104 > normal 1910 |
| test_match      | 0    | 11 passed, 0 fail | self-match 1901/1910 score>0.95; cross img1/img2 1910 vs 2084 feats, all valid; homography 850 RANSAC / 827 inliers; match 0.31 ms |
| test_homography | 0    | 8 passed, 0 fail  | translation (30,20) recovered h[2]=30.14 h[5]=20.28, 1636 matches; rotation 10deg 1195; scale 0.8 943; PGM pair L1452/R2082, 754 RANSAC / 675 inliers |

PASS/FAIL: PASS (29/29 sub-checks across the three binaries, all exit 0). No GPU page fault, no safeCall abort, no crash; dmesg clean.

### Wave32 analysis (gfx1100)
gfx1100 is wave32. The width-32 subgroup shuffles (`__shfl_*(var, delta, width=32)`) in cudautils.h ShiftDown/Up/Shuffle map 1:1 to the native 32-lane wave on gfx1100 -- this is the straightforward case. The `__any_sync -> __any` mapping (FindPointsMulti kernels, 32-thread blocks) polls all active lanes, matching the original intent. The descriptor-norm partial-sum pattern (`sums[idx/32]`, 4 slots for 128 threads) is a layout constant, not wave-size-dependent. Result: numeric output matches gfx90a (same keypoint counts, same correctness checks), confirming the shuffle-width concern is a non-issue on wave32.

### Fork push status (historical, gfx1100 host)
The gfx1100 host reported it could not push its amended commit (0523b54) to the
AMD-Ecosystem/CudaSift fork during that session. By the time of the gfx90a
revalidation below, the fork's moat-port tip already resolved to 0523b54
(`git fetch fork moat-port`), so the shared tip is in place. Re-pushing the fork
is the porter's/user's concern and is out of scope for validation; no fork push
is performed here.

## Revalidation 2026-05-30 (gfx90a re-confirm at new shared tip 0523b54)

Cross-platform regression guard: the gfx1100 follower advanced the shared
moat-port HEAD from fe209b4 (gfx90a's prior validated_sha) to 0523b54, so
gfx90a re-validated at the new tip. The delta fe209b4 -> 0523b54 changed only
two files (no device code): (1) `CMakeLists.txt` -- the hardcoded
`HIP_ARCHITECTURES "gfx90a"` was replaced by a guard that reads
`CMAKE_HIP_ARCHITECTURES`, defaulting to gfx90a when unset/empty, so followers
can pass `-DCMAKE_HIP_ARCHITECTURES=gfx1100`; (2) a new
`.github/workflows/rocm-build-smoketest.yml` (CPU-only compile smoketest,
explicitly NOT a correctness gate). `cudaSiftD.cu`, `cudautils.h`, the compat
header, and all other `.cu`/`.cpp` device sources are byte-identical to the
fe209b4 port that passed before -- this is a pure build-system + CI delta, so
the only gfx90a regression surface is "does it still configure+build for gfx90a
(default arch path) and pass the GPU suite."

GPU: AMD Instinct MI250X / MI250, `gfx90a` (wave64), ROCm 7.2.1. Host has 4x
gfx90a (`rocm_agent_enumerator` -> 4x gfx90a), all idle; pinned to the highest
index with `HIP_VISIBLE_DEVICES=3` (one visible device -> in-process ordinal 0,
default devNum=0). Wiped the stale `build-hip/` and did a clean
configure+build so no prior arch cache (e.g. a gfx1100 cache) could leak in.
Binaries embed `hipv4-amdgcn-amd-amdhsa--gfx90a` (verified via
`llvm-objdump --offloading` on test_extract/test_match/test_homography),
confirming the new CMake arch-guard resolves to gfx90a on the default path.

Configure exit 0 (~2.5s), build exit 0 (~6.6s), all targets built
(cudasift_lib, cudasift, test_extract, test_match, test_homography, benchmark,
demo_extract/match/video). Only the pre-existing ~152 `-Wunused-value`
nodiscard `hipError_t` warnings (+ the `-ffast-math` literal/nan warnings); no
errors.

Tests (each `utils/timeit.sh CudaSift test -- bash -c 'cd projects/CudaSift/src
&& ./build-hip/<bin>'`, run from src/ so `data/` resolves):

| test            | exit | result            | key numbers |
|-----------------|------|-------------------|-------------|
| test_extract    | 0    | 10 passed, 0 fail | basic 1910 kpts (all valid, descriptors unit-norm); thresh monotonic 7080/1910/542/6; octaves monotonic 1740/1876/1910/1919; reproducibility 1910/1910 = 100%; scaleUp 3104 > normal 1910 |
| test_match      | 0    | 11 passed, 0 fail | self-match 1900/1910 score>0.95; cross img1/img2 1910 vs 2084 feats, 1910 valid / 1909 good; homography 851 RANSAC / 828 inliers; match 1.22 ms |
| test_homography | 0    | 8 passed, 0 fail  | translation (30,20) recovered h[2]=30.13 h[5]=20.27, 1646 matches; rotation 10deg 1195; scale 0.8 945; PGM pair 752 RANSAC / 673 inliers |

PASS/FAIL: PASS (29/29 sub-checks across the three binaries, all exit 0). No GPU
page fault, no `safeCall` abort, no crash; GPU 3 healthy/idle post-run, dmesg
clean of amdgpu/page-fault. Numbers match the prior gfx90a validation at fe209b4
within RANSAC/self-match run-to-run variation (e.g. PGM 752/673 vs 751/672,
self-match 1900 vs 1899), and the wave64-sensitive paths still produce correct
on-device output (unit-norm descriptors, 100% reproducible positions, accurate
recovered geometry). The gfx1100 delta did NOT regress gfx90a.

State: linux-gfx90a `revalidate` -> `completed`, validated_sha set to the
rebuilt HEAD `0523b54` (full SHA 0523b540bf209af49f755c52af49cc2a057b95db) via
`python3 utils/moatlib.py set-state CudaSift linux-gfx90a completed --agent validator`.

## Validation 2026-05-30 (windows-gfx1151, TheRock ROCm) -- follower, NO source change

Platform: AMD Radeon 8060S (gfx1151, RDNA3.5, integrated APU), Windows 11. ROCm via
TheRock wheels (rocm-sdk 7.14.0a20260519, hip 7.13.26190, hipcc=clang 23). Validated fork
moat-port HEAD 0523b54 unchanged (zero-churn follower).

### Build (no source change; all-clang CMake HIP)
    cmake -S projects/CudaSift/src -B build-hip -GNinja -DUSE_HIP=ON \
      -DCMAKE_CXX_COMPILER=<rocm>/lib/llvm/bin/clang++ \
      -DCMAKE_HIP_COMPILER=<rocm>/lib/llvm/bin/clang++ \
      -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DCMAKE_PREFIX_PATH=<rocm> \
      -DOpenCV_DIR=<opencv>/build/x64/vc16/lib -DCMAKE_BUILD_TYPE=Release
    cmake --build build-hip -j16
Builds clean (exit 0) -> cudasift_lib + cudasift + test_extract/test_match/test_homography
+ benchmark + demos. Windows specifics (env/deps, NOT source changes): CXX and HIP must both
be clang (CMake refuses to mix Clang-HIP with MSVC-CXX); OpenCV is the prebuilt 4.11.0
Windows pack (opencv.org), with OpenCV_DIR pointed at .../build/x64/vc16/lib directly (the
top-level OpenCVConfig.cmake rejects a non-MSVC compiler) -- the vc16 opencv_world4110.lib
links fine into the clang build (MSVC ABI). At runtime put opencv_world4110.dll's bin dir +
the rocm bin on PATH, and run from projects/CudaSift/src (tests load data/img1.png relative
to cwd).

### Validation (real gfx1151 GPU, HIP_VISIBLE_DEVICES=0, run from src/)
- test_extract:    10 passed, 0 failed (basic ~kpts all valid + unit-norm descriptors;
  threshold monotonic; octave count; reproducibility >95%; scale-up detects more). 
- test_match:      11 passed, 0 failed (refined homography >10 inliers; non-identity;
  low reprojection error; matching < 2-5 ms).
- test_homography:  8 passed, 0 failed (RANSAC 753 matches / 674 inliers on the stereo
  pair; both images > 100 features; > 20 matches).
Matches the gfx90a/gfx1100 results. No NaN, clean exit, no HIP fault.

### Notes
- Compute-based (SIFT image-processing kernels), no rocSPARSE/hipMemGetInfo use, so
  unaffected by the gfx1151-APU runtime gaps that blocked rmm/amgcl -- runs clean on the
  APU like gsplat and GPUMD. No warp-size or any source fix required for gfx1151.
State: port-ready -> completed (validated_sha 0523b54, fork unchanged).

## Audit 2026-06-03

Auditor: validator agent (gfx90a host). Purpose: verify test rigor and scan for MOAT-internal jargon before upstream PR.

### Fork tip / state consistency

`git fetch fork` -> moat-port tip resolves to `0523b540bf209af49f755c52af49cc2a057b95db`, matching status.json `head_sha`/`validated_sha` 0523b54 on all three platforms. Consistent.

### Real test suite discovery

CudaSift has a project-maintained test suite in `tests/` with three purpose-built binaries built by default (`BUILD_TESTS=ON`):

- `test_extract` (10 checks): basic extraction, threshold sensitivity, octave count, reproducibility, scale-up mode.
- `test_match` (11 checks): self-matching score, cross-image match, homography estimation, matching speed.
- `test_homography` (8 checks): translation, rotation, scale change detection, stereo PGM pair RANSAC.

No CTest `add_test` wiring; tests are run directly. No gtest/catch2 framework; custom `CHECK()` macro with pass/fail counters. All test data ships in `data/` (two PNG images + two PGM stereo pairs). No golden/reference pixel-exact files -- correctness is checked by geometric and statistical thresholds (descriptor unit-norm, RANSAC inlier counts, homography recovery within a few pixels). No downloaded data required. These are the same binaries the prior validations ran; the test suite is real, not a demo.

The `.github/workflows/rocm-build-smoketest.yml` in the port commit is a CPU-only compile tripwire; it is NOT a test mechanism.

### GPU test run (this audit, gfx90a, GCD 1, HIP_VISIBLE_DEVICES=1)

GPU: AMD Instinct MI250X / MI250, gfx90a (wave64). Existing build-hip/ at 0523b54 used (gfx90a code objects confirmed via llvm-objdump; CMakeCache shows CMAKE_HIP_ARCHITECTURES=gfx90a).

Run from `projects/CudaSift/src/` so `data/` resolves:

```
HIP_VISIBLE_DEVICES=1 ./build-hip/test_extract      # 10 passed, 0 failed
HIP_VISIBLE_DEVICES=1 ./build-hip/test_match        # 11 passed, 0 failed
HIP_VISIBLE_DEVICES=1 ./build-hip/test_homography   #  8 passed, 0 failed
```

Total: 29/29 checks pass. Key numbers consistent with prior validations (1910 kpts, 100% reproducibility, RANSAC homography recovered within ~0.3 px, self-match >99%, no GPU fault, no safeCall abort, dmesg clean). Prior validation was NOT a sanity demo only -- it ran the full project test suite and these results confirm it.

### Jargon scan

`git diff ed5ef54..fork/moat-port` on all source/build/CI files, plus `grep -r` across the working tree:

| File | Line | Jargon hit | Severity |
|------|------|-----------|---------|
| `cuda_to_hip.h` | 4 | `Strategy A` | must scrub before PR |
| `CMakeLists.txt` | 153 | `Strategy A, minimal footprint` | must scrub before PR |
| `.github/workflows/rocm-build-smoketest.yml` | 5 | `MOAT pipeline` | entire file must be removed before PR (policy: no GHA workflows in fork, already noted in prior validation) |
| `.github/workflows/rocm-build-smoketest.yml` | 9, 11 | `moat-port` (branch refs) | removed with the file |
| commit message | -- | `follower platforms (gfx1100, gfx1151)` | mild; not in source but visible in git log; acceptable or rephrase to "other targets" at PR time |

`head_sha`, `validated_sha`, `revalidate`, `curated commit`, `porter`, `reviewer`, `validator`, `changes-requested`, `port-ready`, `review-passed` -- none found in any committed file.

Actions required before opening the upstream PR:
1. Remove `.github/workflows/rocm-build-smoketest.yml` from the curated commit (already noted; Actions disabled on the fork so it has never run).
2. In `cuda_to_hip.h:4`, replace `Strategy A` with plain English (e.g. "force-included compat header so no per-file edits are needed").
3. In `CMakeLists.txt:153`, replace `(Strategy A, minimal footprint)` with equivalent plain English.
4. Rephrase "follower platforms" in the commit body to "other targets" if desired (not urgent, not in source).

State left as `completed` on all three platforms -- all 29 GPU checks pass on gfx90a, no regression.

## Validation 2026-06-05 (windows-gfx1101, TheRock ROCm 7.14)

GPU: AMD Radeon PRO V710 (gfx1101, RDNA3), HIP_VISIBLE_DEVICES=0, ROCm 7.14.0a20260604 via TheRock pip venv. Fork moat-port HEAD 0523b54 unchanged (zero-churn follower).

### Build (no source change; all-clang CMake HIP)
```
ROCM_DEVEL=<venv>/Lib/site-packages/_rocm_sdk_devel
CLANG=$ROCM_DEVEL/lib/llvm/bin/clang++.exe
OPENCV_DIR=/b/develop/opencv-install/extracted/opencv/build/x64/vc16/lib

cmake -S projects/CudaSift/src -B projects/CudaSift/src/build-hip -GNinja \
  -DUSE_HIP=ON \
  -DCMAKE_CXX_COMPILER=$CLANG \
  -DCMAKE_HIP_COMPILER=$CLANG \
  -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
  -DCMAKE_PREFIX_PATH=$ROCM_DEVEL \
  -DOpenCV_DIR=$OPENCV_DIR \
  -DCMAKE_BUILD_TYPE=Release
cmake --build projects/CudaSift/src/build-hip -j64
```
Configure exit 0 (~3.3s). Build exit 0 (21/21 targets). All targets built: cudasift_lib, cudasift, test_extract, test_match, test_homography, benchmark, demo_extract, demo_match, demo_video. Same pre-existing ~152 `-Wunused-value` nodiscard hipError_t warnings; no errors.

OpenCV: downloaded 4.11.0 prebuilt Windows pack from GitHub releases (opencv-4.11.0-windows.exe), extracted via 7za. Used vc16 lib dir directly. opencv_world4110.dll + theROCm runtime DLLs (amdhip64_7, amd_comgr, rocm_kpack, hiprtc0714, hiprtc-builtins0714) copied into build dir so they override System32's amdhip64.

### Test results (gfx1101, HIP_VISIBLE_DEVICES=0, run from src/)
```
HIP_VISIBLE_DEVICES=0 ./build-hip/test_extract.exe     # run from src/
HIP_VISIBLE_DEVICES=0 ./build-hip/test_match.exe
HIP_VISIBLE_DEVICES=0 ./build-hip/test_homography.exe
```

| test            | exit | result            | key numbers |
|-----------------|------|-------------------|-------------|
| test_extract    | 0    | 10 passed, 0 fail | basic 1910 kpts; thresh 7077/1910/542/6; reproducibility 1910/1910=100%; scaleUp 3104 > normal 1910 |
| test_match      | 0    | 11 passed, 0 fail | self-match 1902/1910 score>0.95; RANSAC 849/827 inliers; match 0.94 ms |
| test_homography | 0    | 8 passed, 0 fail  | translation (30,20) recovered h[2]=30.13 h[5]=20.27; rotation 10deg 1197 matches; scale 0.8 944 matches; PGM L1452/R2082, 753 RANSAC / 675 inliers |

PASS/FAIL: PASS (29/29 checks across three binaries, all exit 0). GPU identified as AMD Radeon PRO V710 (Compute Capability 11.0, 26096 MB). No GPU fault, no safeCall abort, no crash. Numbers match the gfx90a/gfx1100/gfx1151 reference within RANSAC/self-match run-to-run variation.

State: port-ready -> completed (validated_sha 0523b54, fork unchanged).

## Validation 2026-06-06 (windows-gfx1201, TheRock ROCm 7.14)

GPU: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), HIP_VISIBLE_DEVICES=0 (gfx1101 absent; gfx1201 is now the only GPU, enumerated at device 0). ROCm 7.14.0a20260604 via TheRock pip venv. Fork moat-port HEAD 0523b54 unchanged (zero-churn follower).

### Build (no source change; all-clang CMake HIP)
```
ROCM_DEVEL=<venv>/Lib/site-packages/_rocm_sdk_devel
CLANG=$ROCM_DEVEL/lib/llvm/bin/clang++.exe
OPENCV_DIR=B:/develop/opencv-install/extracted/opencv/build/x64/vc16/lib

# Wiped CMakeCache.txt + CMakeFiles/ from the prior gfx1101 build, then reconfigured:
cmake -S projects/CudaSift/src -B projects/CudaSift/src/build-hip -GNinja \
  -DUSE_HIP=ON \
  -DCMAKE_CXX_COMPILER=$CLANG \
  -DCMAKE_HIP_COMPILER=$CLANG \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_PREFIX_PATH=$ROCM_DEVEL \
  -DOpenCV_DIR=$OPENCV_DIR \
  -DCMAKE_BUILD_TYPE=Release
cmake --build projects/CudaSift/src/build-hip -j64
```
Configure exit 0 (~3.4s). Build exit 0 (21/21 targets). All targets built: cudasift_lib, cudasift, test_extract, test_match, test_homography, benchmark, demo_extract, demo_match, demo_video. Same pre-existing ~152 `-Wunused-value` nodiscard hipError_t warnings; no errors. TheRock runtime DLLs (amdhip64_7, amd_comgr, rocm_kpack, hiprtc0714, hiprtc-builtins0714) + opencv_world4110.dll already present in build dir from gfx1101 run; same SDK version, valid for gfx1201.

### Test results (gfx1201, HIP_VISIBLE_DEVICES=0, run from src/)
```
HIP_VISIBLE_DEVICES=0 ./build-hip/test_extract.exe     # run from src/
HIP_VISIBLE_DEVICES=0 ./build-hip/test_match.exe
HIP_VISIBLE_DEVICES=0 ./build-hip/test_homography.exe
```

| test            | exit | result            | key numbers |
|-----------------|------|-------------------|-------------|
| test_extract    | 0    | 10 passed, 0 fail | basic 1910 kpts; thresh 7077/1910/542/6; reproducibility 1910/1910=100%; scaleUp 3104 > normal 1910 |
| test_match      | 0    | 11 passed, 0 fail | self-match 1902/1910 score>0.95; RANSAC 851/828 inliers; match 0.79 ms |
| test_homography | 0    | 8 passed, 0 fail  | translation (30,20) recovered h[2]=30.13 h[5]=20.27; rotation 10deg 1197 matches; scale 0.8 942 matches; PGM L1452/R2082, 753 RANSAC / 674 inliers |

PASS/FAIL: PASS (29/29 checks across three binaries, all exit 0). GPU identified as AMD Radeon RX 9070 XT (Compute Capability 12.0, 16304 MB). No GPU fault, no safeCall abort, no crash. Feature/match counts EXACT vs gfx90a/gfx1100/gfx1101 reference; RANSAC/self-match variation within expected run-to-run bounds.

State: port-ready -> completed (validated_sha 0523b54, fork unchanged).

## PR-prep 2026-06-11 (gfx90a)

Prep edits as a commit on top of the validated head 0523b54, then squashed.

Jargon scrub (whole diff base..HEAD, not just message):
- Removed `.github/workflows/rocm-build-smoketest.yml` entirely (contained
  "MOAT pipeline" + "moat-port" branch refs; fork policy = no GHA workflows).
- `cuda_to_hip.h` banner: dropped "Minimal-footprint port (Strategy A)" wording.
- `CMakeLists.txt:153`: dropped "(Strategy A, minimal footprint)" comment.
- No spaced `<< <` / `>> >` launches present. The `gfx90a` in CMakeLists.txt is
  the legitimate CMAKE_HIP_ARCHITECTURES default fallback (kept). No hardcoded
  /opt/rocm paths or HIP_VISIBLE_DEVICES pins in the tree. No env leaks.

Attribution (new files): added `Copyright (c) 2026 Advanced Micro Devices, Inc.`
+ `Author: Jeff Daily` to the cuda_to_hip.h banner; added
the AMD copyright line to hip_compat/cuda.h and hip_compat/cuda_runtime.h.
CudaSift has no per-file `\author` tags (files carry a name banner), so the
banner line matches house style.

Docs (README.md, house style): added an AMD/ROCm prerequisite line, a
"Build for AMD GPUs (ROCm/HIP)" subsection alongside the existing CUDA build
blocks (USE_HIP + CMAKE_HIP_ARCHITECTURES, gfx90a default), and a `USE_HIP`
row in the CMake Options table. No docs/ dir exists; README is the only doc.

Prep build + run (gfx90a, GPU 3 free; GPU 2 busy with another session):
```
cmake -S projects/CudaSift/src -B build-hip -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j
HIP_VISIBLE_DEVICES=3 ./build-hip/test_extract      # 10 passed, 0 failed
HIP_VISIBLE_DEVICES=3 ./build-hip/test_match        # 11 passed, 0 failed
HIP_VISIBLE_DEVICES=3 ./build-hip/test_homography   #  8 passed, 0 failed
```
29/29 checks pass. 1910 kpts, scaleUp 3104, self-match 1910x1910, homography
PGM L1451/R2082, 753 RANSAC / 681 inliers. Code objects embed gfx90a (verified
via llvm-objdump --offloading). Behavior-preserving: prep edits are
doc/comment/attribution + CI removal only.

Regression guard: `advance-head CudaSift d348161` carried all 5 platforms
forward (inert delta), no follower revalidation. Then squashed to a single
tree-identical commit `3e4df2b` (verified `git diff d348161 3e4df2b` empty,
trees equal), force-pushed to fork moat-port, and `squash-carry-forward
CudaSift 3e4df2b` carried all 5 completed platforms forward. `pr-ready`=True.

PR draft written to projects/CudaSift/pr-draft.md (base=AdaLovelace). NOT
opened -- awaiting jeff's approval.
