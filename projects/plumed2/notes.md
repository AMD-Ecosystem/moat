# plumed2 notes

## Build instructions

### Prerequisites

Build and install PLUMED first from the main source tree:

```bash
cd projects/plumed2/src
./configure --prefix=/path/to/plumed/install --enable-modules=all
make -j$(nproc)
make install
```

### Building the cudaCoord plugin for HIP

```bash
export PATH="/path/to/plumed/install/bin:$PATH"
export PLUMED_KERNEL="/path/to/plumed/install/lib/libplumedKernel.so"
export LD_LIBRARY_PATH="/path/to/plumed/install/lib:$LD_LIBRARY_PATH"
export USE_HIP=1

cd plugins/cudaCoord
./configure.sh
make USE_HIP=1 HIP_ARCHITECTURES=gfx90a
```

For other architectures, change `HIP_ARCHITECTURES` (e.g., `gfx1100`).

### Running tests

```bash
cd regtest
ln -s ../../../regtest/scripts .
cd ..
make USE_HIP=1 check
```

## Port notes

### hipFuncGetAttributes workaround

HIP's `hipFuncGetAttributes` doesn't support templated kernel function pointers
the same way CUDA does. The code uses `cudaFuncGetAttributes` to query
`maxDynamicSharedSizeBytes` and `maxThreadsPerBlock` for optimal kernel launch
parameters.

Workaround: Use `hipDeviceGetAttribute` to query the max shared memory per block
from the device directly, rather than per-kernel attributes. This provides
reasonable defaults that work correctly.

### Tested on linux-gfx90a

- All 32 regression tests pass
- Double and single precision modes work
- Both ortho PBC and no-PBC configurations work

## Validation 2026-06-05

### Platform: linux-gfx90a

GPU: AMD Instinct MI250X (gfx90a), HIP_VISIBLE_DEVICES=0

Build command:
```bash
export PATH="/var/lib/jenkins/moat/_deps/plumed2/install/bin:$PATH"
export PLUMED_KERNEL="/var/lib/jenkins/moat/_deps/plumed2/install/lib/libplumedKernel.so"
export LD_LIBRARY_PATH="/var/lib/jenkins/moat/_deps/plumed2/install/lib:$LD_LIBRARY_PATH"
export USE_HIP=1
export HIP_VISIBLE_DEVICES=0

cd /var/lib/jenkins/moat/projects/plumed2/src/plugins/cudaCoord
./configure.sh
make USE_HIP=1 HIP_ARCHITECTURES=gfx90a
```

Test command:
```bash
make USE_HIP=1 check
```

Results:
- 32/32 regression tests PASS (0 errors)
- Test suites: cudatest (double/float), cudatestPair, cudatestWB
- Configurations tested: ortho PBC, no PBC, MPI variants
- Plugin verified linked against libamdhip64.so.7
- Code objects verified compiled for gfx90a architecture

## Review 2026-06-05

### Problems found

1. **Commit message contains MOAT jargon** (`plugins/cudaCoord/src/cuda_to_hip.h` commit 9b67b5f9d)
   - Line: "The port follows the Strategy A pattern: a single cuda_to_hip.h compatibility"
   - Per CLAUDE.md: "Upstream-visible text (commit messages, code comments, PR bodies) carries no MOAT-internal vocabulary -- no 'Strategy A/B'..."
   - Fix: Amend the commit message to remove "Strategy A pattern" and describe it in neutral terms, e.g. "The port uses a compatibility header approach: a single cuda_to_hip.h..."

### Verified fault classes (no issues found)

- **Warp size**: No hardcoded 32, no `__shfl*`, `__ballot`, `__syncwarp`, `__activemask`, or warpSize-dependent code. The plan correctly noted this is LOW RISK.
- **Rule-of-five on resource handles**: The stream handles are created/destroyed in constructor/destructor with `mpiActive` guards. The class is derived from Colvar (PLUMED base) and is not copied by the framework. This is an existing upstream pattern.
- **OOB neighbor reads**: Not applicable; the code uses tile-based iteration with explicit bounds checking (`activeThread`, `activeCandidates`).
- **Texture pitch alignment (256B)**: Not applicable; no texture objects used.
- **BlockReduce TempStorage reuse**: Each kernel uses TempStorage once, no reuse races.
- **Library swaps**: CUB -> hipCUB correctly mapped via `#define cub hipcub`.

### Build system review

- **HIP_ARCHITECTURES**: Uses `?=` to default to gfx90a only when unset, allowing follower platforms to override without source changes. Correct per PORTING_GUIDE.
- **Makefile conditionals**: Clean separation of CUDA vs HIP paths via `ifdef USE_HIP`.
- **configure.sh**: Properly handles flag differences (removes `-Xcompiler` prefixes for hipcc/clang).

### hipFuncGetAttributes workaround

The workaround queries device attributes (`hipDeviceAttributeMaxSharedMemoryPerBlock`) instead of per-kernel attributes. This is acceptable because the code uses the value only for dynamic shared memory sizing defaults, not for precise per-kernel tuning. The workaround correctly documents the limitation.

### Commit hygiene

- Title: `[ROCm]` prefix, under 72 chars. OK.
- Body: Discloses Claude assistance. OK.
- No `Co-Authored-By: noreply` trailer. OK.
- **PROBLEM**: Contains "Strategy A" MOAT jargon (see above).

### Validation status

Porter reports 32/32 regression tests pass on gfx90a. This is the correct validation path (real GPU run with the project's test suite). Reviewer does not block for missing GPU run; the validator stage provides that.

## Review fix 2026-06-05

Amended commit message to remove MOAT jargon "Strategy A pattern" -- now reads "uses a compatibility header approach". Pushed as 2ba581c3b.

## Review 2026-06-05 (re-review after jargon fix)

Re-reviewed commit 2ba581c3b after porter amended the commit message.

### Jargon fix verified

The previous review requested removal of "Strategy A pattern" MOAT jargon from the commit message. The porter amended it to "uses a compatibility header approach". This is now correct -- no MOAT-internal vocabulary remains.

### All ROCm fault classes verified (no issues)

- Warp size: No hardcoded warpSize, no warp-level intrinsics
- Rule-of-five: Stream handles managed in ctor/dtor, class not copied
- OOB neighbor reads: Tile-based iteration with explicit bounds
- 256B texture pitch: Not applicable, no textures
- BlockReduce TempStorage: Single use per kernel, no reuse races
- Library swaps: CUB -> hipCUB via `#define cub hipcub`
- hipFuncGetAttributes workaround: Queries device attributes, appropriate fallback

### Commit hygiene

- Title: `[ROCm]` prefix, 37 chars. OK.
- Body: Discloses Claude assistance. OK.
- No Co-Authored-By noreply trailer. OK.
- No MOAT jargon. OK.

### Result

Approved for validation. State: review-passed.

## Validation 2026-06-05 (linux-gfx1100)

### Platform: linux-gfx1100

GPU: AMD Radeon Pro W7800 48GB (gfx1100), host has 4x gfx1100 GPUs

Build command:
```bash
export PATH="/var/lib/jenkins/moat/_deps/plumed2/install/bin:$PATH"
export PLUMED_KERNEL="/var/lib/jenkins/moat/_deps/plumed2/install/lib/libplumedKernel.so"
export LD_LIBRARY_PATH="/var/lib/jenkins/moat/_deps/plumed2/install/lib:$LD_LIBRARY_PATH"
export USE_HIP=1
export HIP_VISIBLE_DEVICES=2

cd /var/lib/jenkins/moat/projects/plumed2/src/plugins/cudaCoord
./configure.sh
make USE_HIP=1 HIP_ARCHITECTURES=gfx1100
```

Build result: SUCCESS
- Plugin compiled for gfx1100 (verified with roc-obj-ls)
- Linked against libamdhip64.so.7

Test attempt:
```bash
make USE_HIP=1 check
```

Result: FAILURE - all 32 tests segfault

### Error details

All tests crash with segmentation fault (exit code 139) during plugin initialization, before any GPU kernel execution. The crash occurs in `libamdhip64.so.7` during `thrust::device_vector<double>` construction in the `CudaCoordination` constructor.

Stack trace:
```
[1] /opt/rocm/lib/libamdhip64.so.7(+0x2b09df)
[2] /opt/rocm/lib/libamdhip64.so.7(+0x2b1234)
[3] /opt/rocm/lib/libamdhip64.so.7(+0x306ed5)
[4] /opt/rocm/lib/libamdhip64.so.7(+0x2b1d9e)
[5] /opt/rocm/lib/libamdhip64.so.7(+0x2c9c4e)
[6] CudaCoordination.so(_ZZN6thrust...parallel_for...uninitialized_fill...)
[7] CudaCoordination.so(_ZN6thrust...vector_baseIdNS0_16device_allocatorIdEEEC2Em)
[8] CudaCoordination.so(_ZN4PLMD6colvar16CudaCoordinationIdEC1E...)
```

The crash is inside the HIP runtime during the first GPU memory allocation (thrust device_vector resize in `setUpPermanentGPUMemory()`).

Tested all 4 gfx1100 GPUs (HIP_VISIBLE_DEVICES=0,1,2,3): all crash identically.

Simple rocThrust test program works correctly on the same GPUs, suggesting this is specific to the dynamically-loaded plugin context or the plumed initialization sequence.

### Diagnosis

This is a gfx1100-specific failure. The same commit (2ba581c3b) passed all 32 tests on linux-gfx90a but crashes on linux-gfx1100 during HIP runtime initialization. The failure is not in the ported code itself but in the HIP runtime layer during device_vector allocation in a dynamically loaded shared library.

Possible causes:
1. gfx1100-specific ROCm 7.2.1 runtime bug with late GPU initialization in dynamically loaded plugins
2. Incompatibility between the plumed libplumedKernel.so and the HIP-enabled plugin loading sequence
3. Host-specific ROCm configuration issue (though simple HIP/rocThrust programs work)

This is validation-failed, not blocked -- the port architecture is sound (gfx90a proof), but gfx1100 encounters a runtime-layer crash that needs porter investigation.

## Delta fix 2026-06-05 (linux-gfx1100)

### Root cause

The crash occurred because the HIP runtime was not fully initialized when thrust::device_vector was first constructed in a dlopen'd plugin context. On RDNA3 (gfx1100) specifically, the runtime initialization that happens implicitly on CDNA (gfx90a) does not occur in a dynamically loaded shared library until explicitly triggered.

### Fix

Added explicit HIP runtime initialization before the first GPU memory allocation:

1. In `cuda_to_hip.h`: added `ensureHipRuntimeInitialized()` helper function that calls `hipGetDevice()`/`hipSetDevice()` to trigger runtime initialization
2. In `Coordination.cu`: call `ensureHipRuntimeInitialized()` before the first `thrust::device_vector` construction in the CudaCoordination constructor

The fix is HIP-only (guarded by `#if defined(USE_HIP)`) and does not affect CUDA builds.

### Validation

Built and tested on gfx1100 (AMD Radeon Pro W7800 48GB):

```bash
cd plugins/cudaCoord
make clean
make USE_HIP=1 HIP_ARCHITECTURES=gfx1100 check
```

Result: 32/32 regression tests pass.

Commit: cb90687f8

## Review 2026-06-05 (linux-gfx1100 delta fix)

Reviewed commit cb90687f8 (delta fix for gfx1100 HIP runtime initialization crash).

### Verified

- Fix placement is correct: `ensureHipRuntimeInitialized()` is called before the first GPU allocation (`thrust::device_vector<calculateFloat> inputZeroMax = inputs;` at line 1536)
- Member device_vectors (lines 905-912) are default-constructed without allocation and only resized later via `setUpPermanentGPUMemory()` which is also guarded by `mpiActive`
- Helper function uses C++11 thread-safe static local initialization
- Fix is arch-unified (applies to all HIP builds via `USE_HIP || __HIP_PLATFORM_AMD__` guard)
- CUDA build is byte-identical (all changes inside preprocessor guards)
- Commit message follows conventions: `[ROCm]` prefix, no noreply trailer, no MOAT jargon

### Fault classes checked (no issues)

- No warpSize/32 concerns (fix does not touch kernel code)
- No rule-of-five concerns (no new resource handles)
- No OOB, texture, or library swap concerns

### Result

Approved for validation. State: review-passed.

## Validation 2026-06-05 (linux-gfx1100, second attempt)

### Platform: linux-gfx1100

GPU: AMD Radeon Pro W7800 48GB (gfx1100), HIP_VISIBLE_DEVICES=2

Build command:
```bash
export PATH="/var/lib/jenkins/moat/_deps/plumed2/install/bin:$PATH"
export PLUMED_KERNEL="/var/lib/jenkins/moat/_deps/plumed2/install/lib/libplumedKernel.so"
export LD_LIBRARY_PATH="/var/lib/jenkins/moat/_deps/plumed2/install/lib:$LD_LIBRARY_PATH"
export USE_HIP=1
export HIP_VISIBLE_DEVICES=2

cd /var/lib/jenkins/moat/projects/plumed2/src/plugins/cudaCoord
./configure.sh
make USE_HIP=1 HIP_ARCHITECTURES=gfx1100
```

Test command:
```bash
make USE_HIP=1 HIP_ARCHITECTURES=gfx1100 check
```

Results:
- 32/32 regression tests PASS (0 errors)
- Test suites: cudatest (double/float), cudatestPair, cudatestWB
- Configurations tested: ortho PBC, no PBC, MPI variants, multiple thread counts (512/256/128/64)
- Plugin verified compiled for gfx1100 architecture (verified with roc-obj-ls)
- Commit cb90687f8 (delta fix for HIP runtime initialization) works correctly on gfx1100

Note: Build requires passing `HIP_ARCHITECTURES=gfx1100` as a make argument (not just environment variable) to ensure correct architecture is compiled.

## Revalidation 2026-06-05 (linux-gfx90a)

The linux-gfx90a platform was revalidated after commit cb90687f8 which added explicit HIP runtime initialization for gfx1100. This was a functional change (adding `ensureHipRuntimeInitialized()` call before first `thrust::device_vector` construction).

GPU: AMD Instinct MI250X (gfx90a), HIP_VISIBLE_DEVICES=0

Build command:
```bash
export PATH="/var/lib/jenkins/moat/_deps/plumed2/install/bin:$PATH"
export PLUMED_KERNEL="/var/lib/jenkins/moat/_deps/plumed2/install/lib/libplumedKernel.so"
export LD_LIBRARY_PATH="/var/lib/jenkins/moat/_deps/plumed2/install/lib:$LD_LIBRARY_PATH"
export USE_HIP=1
export HIP_VISIBLE_DEVICES=0

cd /var/lib/jenkins/moat/projects/plumed2/src/plugins/cudaCoord
make clean
make USE_HIP=1 HIP_ARCHITECTURES=gfx90a -j$(nproc)
```

Test command:
```bash
make USE_HIP=1 check
```

Result: PASS
- 32/32 regression tests pass (0 errors)
- Test suites: cudatest (double/float), cudatestPair, cudatestWB
- Configurations tested: ortho PBC, no PBC, MPI variants
- The gfx1100 runtime initialization fix does not affect gfx90a behavior

Validated commit: cb90687f8706326ed2ce02f4a98c18bd05bdc582

## Validation 2026-06-08

### Platform: windows-gfx1201

GPU: AMD Radeon RX 9070 XT (gfx1201), HIP_VISIBLE_DEVICES=0 (confirmed via hipInfo.exe -- V710 offline, RX 9070 XT now at device 0).

### Toolchain assessment: BLOCKED

PLUMED is an autoconf/autotools project. Its build requires:

1. `./configure` (a GNU Autoconf 2.69-generated POSIX shell script) to build the main PLUMED library -- this requires autoconf, automake, and standard POSIX compiler detection that does not work on Windows without MSYS2/MinGW toolchains targeting native Windows.
2. The cudaCoord plugin's `configure.sh` calls `plumed config makefile_conf` and `plumed info --include-dir` -- these require a working `plumed` binary in PATH from a completed PLUMED build.
3. The Makefile produces a `.so` shared library (SOEXT=so). PLUMED's plugin mechanism uses `dlopen` -- a POSIX-only API with no `.dll` equivalent in the current build system.
4. No conda available on this host; no pre-built `plumed.exe` in PATH; `autoconf`/`automake`/`autoreconf` are absent.

hipcc is available at `B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel/bin/hipcc.exe` and `gmake` at `/c/Strawberry/c/bin/gmake`, so the GPU compiler toolchain is present. But building PLUMED itself (the prerequisite for the plugin) requires autotools + a POSIX shell environment that this Windows host does not have configured for native compilation.

Specifically: no `autoconf`, no `automake`, no `autoreconf` in PATH. The pre-generated `configure` script is a POSIX sh script that will fail on Windows because it tries to detect POSIX tools (gcc, ld, ranlib, etc.) and generate POSIX Makefiles using Linux conventions (`-fPIC`, `-ldl`, SONAME, `dlopen`) incompatible with Windows native builds.

**Resolution**: windows-gfx1201 is blocked with reason: plumed2 requires autotools + POSIX build toolchain for the main PLUMED library (prerequisite for the cudaCoord plugin). No autoconf/automake present on this Windows host, and PLUMED's build system has no Windows-native (MSVC/clang-cl) build path; the plugin output is a .so using dlopen which is POSIX-only.
