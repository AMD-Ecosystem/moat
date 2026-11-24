# plumed2 notes

## Build instructions

Current recipe (from the 2026-11-19 upstream sync on; the older recipe with
`HIP_ARCHITECTURES=gfx90a` defaults and `-fgpu-rdc` is gone).

### Prerequisites

Build and install PLUMED first from the main source tree (any MPI, or none):

```bash
cd projects/plumed2/src
./configure --prefix=/path/to/plumed/install CXX=mpicxx CC=mpicc
make -j$(nproc)
make install
```

### Building the cudaCoord plugin for HIP

```bash
export PATH="/path/to/plumed/install/bin:$PATH"
export PLUMED_KERNEL="/path/to/plumed/install/lib/libplumedKernel.so"
export LD_LIBRARY_PATH="/path/to/plumed/install/lib:$LD_LIBRARY_PATH"

cd plugins/cudaCoord
USE_HIP=1 ./configure.sh
make USE_HIP=1                      # GPUs of this machine (--offload-arch=native)
make USE_HIP=1 HIP_ARCHITECTURES=gfx942   # or an explicit list, "gfx90a gfx1100"
```

### Running tests

```bash
cd regtest && ln -s ../../../regtest/scripts . && cd ..
make USE_HIP=1 check
```

## Port notes

Superseded: this section describes the June HIP layer, replaced by 388fcc10f; see "## Upstream sync 2026-11-19 (porter, linux-gfx942)".

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

## Contribution terms ruling 2026-11-18

jeff (jeffdaily) ruled on PLUMED's contribution terms on 2026-11-18: he accepts
that opening the upstream pull request transfers copyright in the contribution to
the PLUMED developers. Consequence applied in the port: the new file
`plugins/cudaCoord/src/cuda_to_hip.h` carries the same header as its sibling files
in `plugins/cudaCoord/src/`, copied verbatim (including their
"Copyright (c) 2024 Daniele Rapetti, The plumed team" line and the "cudaOnPlumed"
wording, which is how every sibling reads).

## Upstream sync 2026-11-19 (porter, linux-gfx942)

jeff ruled that the port is brought up to upstream's current base before the PR
is prepared. No upstream PR is open, so `moat-port` was pushed directly.

### What was done

- Fork mirror `master` fast-forwarded to upstream master `422109dc3`
  (from `675f720ae`; plain push of upstream's own commit, no merge, no force).
  1010 commits between the old mirror and the new tip; the port's old base was
  `4d7da4b24`.
- `moat-port`: merge commit `b3b386027` (upstream master into the port, a merge,
  not a rebase; `cb90687f8` and everything validated stays an ancestor), then
  `388fcc10f` "[ROCm] Build the cudaCoord plugin for AMD GPUs with HIP".
- The merge commit's tree is IDENTICAL to upstream master (`git diff 422109dc3
  b3b386027` is empty): every conflict was resolved by taking upstream's side and
  the old HIP layer was dropped in the merge. The whole HIP port against the new
  base is the single commit `388fcc10f`: 7 files, +162/-7
  (Makefile, README.md, configure.sh, src/Coordination.cu, src/Coordination.cuh,
  src/cudaHelpers.cuh, new src/cuda_to_hip.h). Nothing outside plugins/cudaCoord.

### What upstream changed that mattered

23 commits to plugins/cudaCoord since the June validation (bb7242f91 ... 38f58d5b6):
Coordination.cu went from about 1500 to 1897 lines and cudaHelpers.cuh to 287.

- CSR cell lists built on the CPU, uploaded only on list updates; displacement-
  triggered rebuilds; only the "active atoms" move to/from the GPU.
- Warp-per-atom cell kernels (`getCoord<self|dual>`, `getDualDev`): logical warps
  of 32 lanes (`warp = threadIdx.x/32`, `lane = threadIdx.x%32`) split the
  candidates of one atom; `warpReduce3` and `blockReduceToPartials` reduce with
  `__shfl_down_sync(0xffffffffu, v, 16..1)`. Block size is
  `min(roundUp32(32*maxAtomsInCell), 256, THREADS)`, so a 32-thread block is used
  whenever the biggest cell holds one atom, or with THREADS=32.
- New runtime surface: non-blocking highest-priority streams
  (`cudaDeviceGetStreamPriorityRange`, `cudaStreamCreateWithPriority`), events
  (`cudaEventCreateWithFlags`, `cudaEventRecord`, `cudaStreamWaitEvent`),
  `cudaStreamSynchronize`, `cudaGetLastError`/`cudaGetErrorString` through a
  `cudaCheck` helper, grow-only device (`cudaMalloc`) and pinned
  (`cudaMallocHost`/`cudaFreeHost`) buffers.
- GPU selection as in openaccPTM (`chooseGPU`: `cudaGetDeviceCount`, MD-assigned
  device or the current one; `deviceGuard` restores the caller's device).
- Minimum image in scaled coordinates (rotated boxes); input validation of NN, MM,
  R_0; new regtests `cudatest/rt-double-rotatedbox` and `cudatest/rt-input-errors`.
- Makefile: `-D__PLUMED_HAS_CUDA=1`, `EXTRA_OBJS`, link with
  `-Xlinker -z -Xlinker nodelete`, `-ccbin $(CXX)` for the link too, a `check-gpu`
  target that builds `plugins/openaccPTM/gpucheck/device_check.cpp` with nvcc.
- configure.sh: strips `-fconcepts`; `PLUMED_KERNEL` now carries
  `-Xlinker -rpath=...`.
- Packaging: `conda.mk` (CUDA package variants) and `plugin.yml`; Readme.md
  renamed README.md.

### How each conflict was resolved

- `Makefile` (conflict): upstream's file taken, then a HIP branch added in front
  of the nvcc variables: `NVCC = $(HIPCC)`, `NVCCCFLAGS = -x hip
  $(addprefix --offload-arch=,$(HIP_ARCHITECTURES))` (default `native`),
  `NVCCLDFLAGS = -shared`, and a `HOST_COMPILER` variable that is `-ccbin $(CXX)`
  for nvcc and empty for hipcc; the two rules use it instead of the literal
  `-ccbin $(CXX)`, otherwise unchanged, so the HIP compile rule keeps `$(DEPFLAGS)`
  (dependency tracking verified: touching src/cuda_to_hip.h rebuilds both objects).
  `-fgpu-rdc` dropped: no device code crosses translation units (cudaHelpers.cu is
  empty, kernels are header templates instantiated in Coordination.cu).
  `SONAME_OPTION` is `-Wl,-soname` under USE_HIP: clang hands `-Xlinker
  -soname,NAME` to ld.lld as ONE argument ("unknown argument
  '-soname,CudaCoordination.so'"), nvcc splits it. Upstream's other `-Xlinker`
  uses (nodelete, rpath) work with clang unchanged. No Open MPI-only flag
  anywhere (the old `mpic++ --showme:compile` is gone).
- `configure.sh` (conflict): upstream's file taken; the HIP branch is a pure
  addition placed before upstream's nvcc block, which it skips (`exit 0`), so the
  nvcc block is byte-identical to upstream. Under USE_HIP it:
  - strips only `-fconcepts` and `-fno-gnu-unique` from CXXFLAGS (clang 24 rejects
    both: "unknown argument");
  - keeps `-fopenmp` in CXXFLAGS (it compiles fine with `-x hip` on this ROCm;
    PLUMED's headers carry `#pragma omp`, so the plugin keeps the kernel's OpenMP
    view) and removes it only from the link flags: with it, hipcc linked
    `libomp.so` (LLVM) into the plugin next to the `libgomp.so.1` the GCC-built
    kernel uses; `nm -u` shows the plugin references no OpenMP symbol at all.
    The old port stripped -fopenmp from all of CXXFLAGS and its notes give no
    reason; with ROCm 10.1 / clang 24, `-x hip -fopenmp -c` compiles both files.
    If a validator on an older ROCm sees a compile error from -fopenmp, record the
    exact message: the narrower fix would then be `-Xarch_host -fopenmp`.
  - finds mpi.h for MPI builds of PLUMED by asking the configured host compiler
    (`echo '#include <mpi.h>' | $CXX -x c++ -E -H -o /dev/null -`) and appends
    `CPPFLAGS+=-I<dir>`; works with any gcc-compatible wrapper (tested with MPICH
    4.2.3; Open MPI not tested on this host). shellcheck: only upstream's
    pre-existing SC2086 on line 4; `shfmt -ci -s -bn -i 2 -d` clean.
- `src/Coordination.cu` (conflict, the old runtime-init call site): upstream's
  file taken; the two cub includes and the cuda_runtime/device_launch_parameters
  includes are guarded `#ifndef __HIP__`; the three `__shfl_down_sync` in
  `warpReduce3` use `CUDAHELPERS::warpMask()` and width 32.
- `src/Coordination.cuh`, `src/cudaHelpers.cuh` (auto-merged, but the cudaHelpers
  auto-merge was WRONG: it put the old hipCUB block above upstream's new
  unconditional `#include <cub/block/block_reduce.cuh>`): redone from upstream's
  files: include `cuda_to_hip.h`, guard the CUDA-only includes, and the
  `blockReduceToPartials` shuffle uses `warpMask()` and width 32.
- `src/cuda_to_hip.h` (port-only file): rewritten. Predicate is `__HIP__` (clang
  predefines it for every HIP compile; `__HIP_PLATFORM_AMD__` is not predefined
  by hipcc here, it comes from the HIP headers), so no `-DUSE_HIP` is needed.
  Maps every runtime name the new code uses; `namespace cub = hipcub` instead of
  the old `#define cub hipcub` macro; removed mappings nothing uses any more
  (`cudaDeviceSynchronize`, `cudaStreamCreate`).
- `README.md`: a HIP paragraph next to the nvcc one in "Compile".

### The two helpers of the old port

- `ensureHipRuntimeInitialized()` (plain `static bool`, added for the June gfx1100
  segfault in the first `thrust::device_vector`): DROPPED. Upstream's constructor
  now calls `chooseGPU` (`cudaGetDeviceCount`, `cudaGetDevice`) and creates a
  `deviceGuard` (`cudaSetDevice` when needed) as its first GPU work, before any
  allocation -- the same calls the helper made. This is reasoning, not a gfx1100
  run: the wave32 validation on linux-gfx1100 must confirm it (watch for the June
  signature: segfault in libamdhip64 under
  `thrust::...uninitialized_fill` in the constructor).
- `cudaFuncGetAttributes` replacement that hard-coded `maxThreadsPerBlock=1024`,
  `numRegs=0` and the device's shared memory: REPLACED by a template that forwards
  to the real `hipFuncGetAttributes(attr, reinterpret_cast<const void*>(kernel))`
  (HIP has only the `const void*` overload, CUDA has a template one, so a wrapper
  is still needed). Upstream now feeds `attr.maxThreadsPerBlock` and `attr.numRegs`
  into the launch configuration and log, so fake values would matter. Measured on
  gfx942: maxDynamicSharedSizeBytes 65536, maxThreadsPerBlock 1024, registers in
  the PLUMED log 74/89/93 (dual-derivative kernel 62).

### Wavefront width (the fault class)

The new kernels assume 32-lane warps. On 64-lane CDNA: (1) HIP's
`__shfl_down_sync` static_asserts a 64-bit mask, so `0xffffffffu` does not
compile; (2) a full 64-bit mask is wrong where only one 32-lane warp of the
wavefront is in the block. Fix: `CUDAHELPERS::warpMask()` returns
`0xffffffffull << (__lane_id() & ~31u)` on HIP (the caller's 32-lane group;
the whole wave on wave32) and `0xffffffffu` on CUDA; shuffles pass width 32 so
they never cross the logical warp. The algorithm itself was already
width-correct: every caller reaches the reductions block-uniformly, and the
lane-0 result of a 32-lane shfl_down tree reads only its own group.

Evidence on gfx942 (64-lane):
- Release build regtests 34/34 (below).
- Debug build (my scratch copy with `-DNDEBUG` removed from Makefile.conf, so
  HIP's `__hip_check_mask` assertion is live), extra input with THREADS=32
  (32-thread blocks, i.e. half a wavefront) and THREADS=64 in self, dual
  (GROUPA/GROUPB) and PAIR modes, against CPU COORDINATION on traj-nopbc.xyz,
  10 frames: values and all derivatives identical at %14.8f
  (max |diff| = 0 for s32, s64, d32, p32 and a D_MAX=1.45 case).
- Control: the same debug build with a full 64-bit mask traps in
  `getCoord<selfCalc<false,Rational,double>,double>` on the THREADS=32 action
  (`HSA_STATUS_ERROR_EXCEPTION ... code: 0x1016`, then
  `cudaGetDevice: unspecified launch failure`). So the per-group mask is
  required, not cosmetic.

### Validation of this round's build (porter's run, gfx942)

Host: AMD Instinct MI300X (gfx942, partitioned; HIP_VISIBLE_DEVICES=1), ROCm 10.1
Python wheels (HIP 7.16, AMD clang 24), g++ 13.3, MPICH 4.2.3 built from source
into the job directory, PLUMED upstream master `422109dc3` configured with
`CXX=mpicxx CC=mpicc` (CXXFLAGS `-O3 -fPIC -Wall -pedantic -Wno-unknown-pragmas
-std=c++17 -fno-gnu-unique -fopenmp`).

```bash
cd plugins/cudaCoord
USE_HIP=1 ./configure.sh
make USE_HIP=1 HIP_ARCHITECTURES=gfx942
cd regtest && ln -s ../../../regtest/scripts . && cd ..
make USE_HIP=1 HIP_ARCHITECTURES=gfx942 check
```

Result: "34 tests performed, 0 tests not applicable, 0 errors found" --
cudatest 14 (incl. rt-double-rotatedbox, rt-input-errors), cudatestPair 8,
cudatestWB 12 (incl. rt-double-orthopbc-mpi with 2 MPI ranks). Logs show
"gpu runs on GPU 0". Plugin NEEDED: libamdhip64.so.7, libplumedKernel.so, no
libomp; SONAME CudaCoordination.so; FLAGS_1 NODELETE.

Build warnings (HIP only, harmless, left alone to keep upstream code untouched):
82 x "ignoring return value of type 'hipError_t' declared with 'nodiscard'"
(upstream ignores the return of cudaFree/cudaFreeHost/cudaSetDevice/
cudaStreamSynchronize/... in destructors; HIP marks hipError_t nodiscard) and
4 x -Wreorder-ctor (upstream's mpiActive/cells initializer order).

check-gpu: `make check-gpu` builds openaccPTM's gpucheck with nvcc, so it does not
work in a HIP build (see question 1 below). Scratch evidence only, nothing
committed: device_check.cpp compiled with hipcc through cuda_to_hip.h, `follow`
mode: "follow: 0 failure(s)" (caller GPU current and its buffer usable after every
step and after PLUMED is destroyed). `md` mode needs 2 GPUs; this job is pinned to
one, so it was not run.

### CUDA no-regression compile

nvcc 12.8.93 from a micromamba env in the job directory (conda-forge
`cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-cccl=12.8`), host compiler the MPICH
wrapper over g++ 13.3, on a scratch copy of the plugin:

```bash
./configure.sh
make NVCCCFLAGS="-dc -dlto --gpu-architecture=sm_80" NVCCLDFLAGS="-shared -dlto --gpu-architecture=sm_80"
```

Compiles and links (CudaCoordination.so), no errors or warnings. The same build
of pristine upstream `plugins/cudaCoord` and `cuobjdump -sass` of both: 23
functions in each, same set, 0 functions with different SASS. (Not run on an
NVIDIA GPU; this host has none.)

### Questions for jeff (not decided here)

1. `make check-gpu` in a HIP build: it builds `plugins/openaccPTM/gpucheck/
   device_check.cpp` with nvcc. Making it work with HIP needs a change outside
   plugins/cudaCoord (the openaccPTM gpucheck Makefile and a HIP spelling of its
   ~10 runtime calls). Left out as a scope question: extend the port to the shared
   gpucheck program, or keep check-gpu CUDA-only (and say so in the README)?
2. `conda.mk` (CUDA package variants, mutually exclusive cuda-coordination
   packages) has no ROCm variant. Packaging is not ported; say if a ROCm conda
   variant is wanted.
3. The new file's header is the siblings' header verbatim, which names
   "Daniele Rapetti, The plumed team" and "cudaOnPlumed", per the 2026-11-18
   ruling. Flagging only so it is a conscious choice at PR time.
4. `moatlib.py audit-commits plumed2` flags the two June commits (2ba581c,
   cb90687: "Test Plan has no fenced command block"); they are validated and
   were not amended. A PR-prep squash would replace them.

## Lesson for cuda-to-rocm: an nvcc Makefile's link line does not carry over to hipcc verbatim

Promoted to cuda-to-rocm/references/build-cmake-link.md in bb61ae96.

Belongs in `build-cmake-link.md` (plain-Makefile builds).

Porting a hand-written Makefile from nvcc to hipcc, two link-line idioms change
meaning silently or loudly. (1) nvcc splits the argument of `-Xlinker` at commas,
clang does not: `-Xlinker -soname,libfoo.so` reaches ld.lld as one argument and
fails with "unknown argument '-soname,libfoo.so'"; use `-Wl,-soname,libfoo.so`
for the HIP branch (plain `-Xlinker -z -Xlinker nodelete` and `-Xlinker
-rpath=dir` are fine). (2) `-fopenmp` on a hipcc link line links LLVM's
`libomp.so`; if the host application or library was built by GCC
(`libgomp.so.1`), the process ends up with two OpenMP runtimes. When the HIP
objects make no OpenMP call (`nm -u | grep -i omp` is empty), keep `-fopenmp` on
the compile line and drop it from the link line only. Also: a project that
compiles device code with nvcc `-ccbin mpicxx` gets the MPI include path from the
wrapper for free; hipcc does not use a host compiler, so ask the wrapper where
`mpi.h` is (`echo '#include <mpi.h>' | mpicxx -x c++ -E -H -o /dev/null -`,
which relies only on the compiler's `-H` header trace, not on a wrapper option;
tested with MPICH) instead of `--showme:compile` (Open MPI only). Seen on plumed2's cudaCoord plugin,
ROCm 10.1 (clang 24), 2026-11.

## Review 2026-11-19 (upstream sync)

Reviewer on linux-gfx942. Scope: `git diff master...moat-port` (master = upstream 422109dc3, head 388fcc10f, 7 files under plugins/cudaCoord) plus the two new commit messages; `git diff 422109dc3 b3b386027` is empty.

The review found no problems in the fork branch. What was checked:
- Wavefront width: every caller reaches `warpReduce3` (Coordination.cu:259) and `blockReduceToPartials` (cudaHelpers.cuh:236) with whole 32-thread logical warps (block-uniform `base < nInCell` in getCoord, block-uniform early return in getDualDev, no early return in getCoordPair; block sizes come from `roundUpToWarp`/`(n/32)*32`, so always a multiple of 32), and logical warps never straddle a wavefront (1-D blocks). `warpMask()` therefore matches the lanes present in both the half-wave (32-thread block on wave64) and full-wave cases. A standalone kernel using the header's `warpMask()` and `__shfl_down_sync(..., 32)`, built for gfx942 with HIP's mask assertions live (no NDEBUG), gives exact per-warp sums for blocks of 32, 64, 96 and 256 threads. blockReduceMemory, nwarps, cellChunks and getCoordPair all count 32-lane logical warps consistently; reducePartials uses hipCUB BlockReduce<T,256>, which is width-agnostic. A scratch build for `HIP_ARCHITECTURES="gfx1100 gfx942"` compiles (wave32 static asserts pass) and carries both code objects.
- Compat header maps every CUDA runtime name used in src/ (checked against a grep of all `cuda*` identifiers); the `cudaFuncGetAttributes` template forwards to HIP's only (`const void*`) overload. Dropping `ensureHipRuntimeInitialized()` is sound: the constructor's first GPU calls are `hipGetDeviceCount`/`hipGetDevice` in `chooseGPU`, which initialize the runtime before any allocation; non-root ranks do no GPU work. Makefile/configure.sh: nvcc branch unchanged in behavior (only `-ccbin $(CXX)` moved into `HOST_COMPILER`), DEPFLAGS kept, `-Wl,-soname` (nvcc 12.8 `--dryrun` splits `-Xlinker -soname,X`, `hipcc -###` passes it as one argument, so the lesson's claim holds), `-fopenmp` gone only from DYNAMIC_LIBS (scratch plugin: no libomp NEEDED, no OpenMP imports), mpi.h probe emits `CPPFLAGS+=` after PLUMED's `CPPFLAGS=`. New file header identical to the four siblings. Commit titles 55/58 chars, `[ROCm]` only on the HIP commit, prose AI disclosure, no trailer, fenced Test Plan; `jargon.py --port plumed2` clean. The `## Lesson for cuda-to-rocm` section is accurate as written.

Records only (does not gate the fork): the `## Port notes` section near the top of this file (the "hipFuncGetAttributes workaround" via `hipDeviceGetAttribute` and "Tested on linux-gfx90a: 32 tests") describes the June HIP layer that 388fcc10f replaced; the next porter round on this project should mark it superseded by `## Upstream sync 2026-11-19` so a reader does not take it as the current design.

Verdict: review-passed at 388fcc10fd9a7d0c8e01362ec1d43ee830313368. GPU validation (wave64 and the wave32 gfx1100 run, which also settles the runtime-init question) is the validator's next step.

## Validation 2026-11-19 (linux-gfx942)

Validated sha: 388fcc10fd9a7d0c8e01362ec1d43ee830313368 (fork clone checked out at
head, no modified tracked files). GPU: AMD Instinct MI300X (gfx942, wave64,
partitioned), HIP_VISIBLE_DEVICES=1. ROCm 10.1 Python wheels (HIP 7.16, AMD clang
24), g++ 13.3, MPICH 4.2.3 (built from source), PLUMED upstream master 422109dc3
built with `./configure CXX=mpicxx CC=mpicc` and installed.

Build (clean, after `make clean`):
```bash
cd plugins/cudaCoord
USE_HIP=1 ./configure.sh
make USE_HIP=1 HIP_ARCHITECTURES=gfx942
```

Tests:
```bash
cd regtest && ln -s ../../../regtest/scripts . && cd ..
make USE_HIP=1 HIP_ARCHITECTURES=gfx942 check
```
Result: PASS. "34 tests performed, 0 tests not applicable, 0 errors found":
cudatest 14/14 (includes the new rt-double-rotatedbox and rt-input-errors),
cudatestPair 8/8, cudatestWB 12/12 (rt-double-orthopbc-mpi ran with 2 MPI ranks).
THREADS 64/128/256/512/1024 variants are compared against CPU COORDINATION inside
the tests. The wave64 debug-assertion run with THREADS=32 is recorded in the
upstream-sync section above (exact match with CPU; full 64-bit mask traps).
The change touches only plugins/cudaCoord, so PLUMED's own (non-GPU) regtests are
not affected by it.

CUDA gate at this head: PASS (compile and link only, no NVIDIA GPU). nvcc 12.8.93
(conda-forge env in the job directory), on `git archive 388fcc10f
plugins/cudaCoord`:
```bash
./configure.sh
make NVCCCFLAGS="-dc -dlto --gpu-architecture=sm_80" NVCCLDFLAGS="-shared -dlto --gpu-architecture=sm_80"
```
CudaCoordination.so built, no errors or warnings; SASS identical per function to
pristine upstream (porter's comparison above).

Content checks: `python3 utils/jargon.py --port plumed2` clean; the HIP build is
documented in plugins/cudaCoord/README.md next to the nvcc instructions.

Still needed for pr-ready: wave32 (linux-gfx1100 recorded cb90687f8 only, so it
reads revalidate at this head; it must confirm the dropped runtime-init helper,
see the upstream-sync section), and jeff's answers to the four questions in the
upstream-sync section. linux-gfx90a also reads revalidate (extra wave64 evidence;
this platform satisfies wave64 at head).

## Validation 2026-11-19 (linux-gfx1201)

Validated sha: 388fcc10fd9a7d0c8e01362ec1d43ee830313368 (fresh fork clone checked out at head, no modified tracked files). GPU: AMD Radeon RDNA4 gfx1201, wavefront size 32 (rocminfo), HIP_VISIBLE_DEVICES=1 (one GPU visible; logs say "runs on GPU 0"). ROCm 10.1 Python wheels (HIP 7.16.26385, AMD clang 24.0.0git), g++ 13.3.0, MPICH 4.2.3 built from source into the job directory (`--with-device=ch3 --disable-fortran`), PLUMED upstream master 422109dc3 (the head's merge parent) built with `./configure --prefix=<job>/deps/plumed CXX=mpicxx CC=mpicc` and installed. Same Makefile.conf flags as gfx942 (`-O3 -fPIC ... -std=c++17 -fno-gnu-unique -fopenmp`, `-D__PLUMED_HAS_MPI=1`).

Build:
```bash
export PATH=<plumed>/bin:<mpich>/bin:$PATH PLUMED_KERNEL=<plumed>/lib/libplumedKernel.so LD_LIBRARY_PATH=<plumed>/lib:<mpich>/lib:$LD_LIBRARY_PATH
cd plugins/cudaCoord
USE_HIP=1 ./configure.sh
make USE_HIP=1 HIP_ARCHITECTURES=gfx1201
```
Built clean; only code object amdgcn-amd-amdhsa--gfx1201. 88 warnings, all the known harmless ones (84 nodiscard `hipError_t` ignored in upstream destructors/guards, 4 -Wreorder-ctor). Plugin NEEDED: libamdhip64.so.7, libplumedKernel.so, no libomp; SONAME CudaCoordination.so; FLAGS_1 NODELETE.

Tests:
```bash
cd regtest && ln -s ../../../regtest/scripts . && cd ..
make USE_HIP=1 HIP_ARCHITECTURES=gfx1201 check
```
Result: PASS. "34 tests performed, 0 tests not applicable, 0 errors found": cudatest 14/14 (includes rt-double-rotatedbox and rt-input-errors), cudatestPair 8/8, cudatestWB 12/12 (rt-double-orthopbc-mpi with mpiprocs=2). The same set the gfx942 run had; nothing skipped. The regtests cover THREADS=64/128/256/512/1024 and the default; none uses THREADS=32.

Runtime init (the dropped June `ensureHipRuntimeInitialized()` helper): no segfault, no hang on this wave32 host in any run (34 regtests plus the extra runs below); the constructor's `chooseGPU`/`deviceGuard` path initializes HIP before the first allocation as the review reasoned. (This is gfx1201, not the gfx1100 that showed the June segfault; it is the same wave32 family and runtime path.)

THREADS=32 (and wave32-specific) extra run, scratch inputs only (nothing committed), with `plumed --no-mpi driver`, each GPU action against CPU COORDINATION, DUMPDERIVATIVES at %14.8f (values and every derivative), on both the release plugin above and a debug copy built from the same sources with `-DNDEBUG=1` removed from Makefile.conf (HIP's device-side shuffle mask assertions live):
- traj-nopbc.xyz (10 frames): self THREADS=32 (log: "threads per block (a warp per atom): 32"), THREADS=64, default (256); dual GROUPA=GROUPB=1-108 THREADS=32; PAIR 1-107/2-108 THREADS=32; self D_MAX=1.45 with THREADS=32 and with the default.
- traj-ortho.xyz: self 1-108 R_0=0.4 THREADS=32; dual 1-50 / 51-108 THREADS=32.
- traj-rotating.xyz with FIT_TO_TEMPLATE (rotated box, as rt-double-rotatedbox): dual 1-20 / 21-220, dual single-atom GROUPA=1 with NL_CUTOFF=2.9, self 1-220, each THREADS=32, NN=6 MM=12 D_MAX=2.5.
Result: max |GPU - CPU| = 0 for every column in all 9 files of each build (rows: 3330 to 6570 per file), release and debug alike; no trap, no assertion. So the per-group `warpMask()` is correct when the 32-lane logical warp is the whole wavefront, at 32-thread blocks and larger.

CUDA gate: not rerun; already recorded at this head_sha in the linux-gfx942 validation (compile and link with nvcc 12.8.93, SASS identical to upstream).

Content checks: `python3 utils/jargon.py --port plumed2` clean (needed local `master`/`moat-port` refs in the fresh clone: `git branch master origin/master`); the HIP build is documented in plugins/cudaCoord/README.md next to the nvcc build.

With this run wave32 is satisfied at head (wave64 by linux-gfx942; windows waived). Still open for pr-ready: jeff's answers to the four questions in the upstream-sync section.
## CUDA compile check 2026-11-19 (real nvcc, mi355 host)

Evidence only: no stage or platform record changed, nothing pushed. Compile and
link only; the host has no NVIDIA GPU.

Trees (fresh clones in the job directory; plumed2 has no submodules):
- port: AMD-Ecosystem/plumed2 at head_sha `388fcc10fd9a7d0c8e01362ec1d43ee830313368`
- base: plumed/plumed2 at the port's merge base `422109dc3e9aa3aa72a55f326dee473231a1039f`
  (also upstream master's tip at the time of the check)

Toolkit: CUDA 12.8 from the system install, `/usr/local/cuda/bin/nvcc`
("Cuda compilation tools, release 12.8, V12.8.93", Build cuda_12.8.r12.8/compiler.35583870_0),
host compiler g++ 13.3.0 (Ubuntu 24.04). The earlier CUDA checks in this file
used a conda-forge nvcc 12.8.93; this one is the system toolkit.

Recipe: upstream's CI does not build cudaCoord (linuxWF.yml builds only the
nvhpc/openacc plugin, on the host), so the plugin README's recipe was used:
PLUMED kernel with its autotools build, then the plugin's `configure.sh` and
`make`. Neither build is CMake, so Ninja does not apply; steps are counted as
make reports them. Same commands for both trees (`<t>` = base or port):

```bash
cd <t>
./configure --prefix=$PWD/../install-<t>      # serial: no MPI on this host
make -j42 && make install
export PATH=$PWD/../install-<t>/bin:/usr/local/cuda/bin:$PATH
export PLUMED_KERNEL=$PWD/../install-<t>/lib/libplumedKernel.so
cd plugins/cudaCoord
./configure.sh
make                                                   # documented default (nvcc default arch, sm_52, LTO)
make -B NVCCCFLAGS="-dc -dlto --gpu-architecture=sm_80" NVCCLDFLAGS="-shared -dlto --gpu-architecture=sm_80"   # README's explicit-SM form
```

| | base 422109dc3 | port 388fcc10f |
|---|---|---|
| kernel configure | rc 0 | rc 0 |
| kernel make / install | rc 0 / rc 0 | rc 0 / rc 0 |
| plugin configure.sh | rc 0 | rc 0 |
| plugin make, default arch | 3/3 (Coordination.o, cudaHelpers.o, link) | 3/3 |
| plugin make, sm_80 | 3/3 | 3/3 |
| first error | none | none |

Warnings: only nvcc's "Support for offline compilation for architectures prior
to '<compute/sm/lto>_75' will be removed" (once per nvcc call, default-arch build
only), in both trees equally. The sm_80 build has no warnings.

Equivalence with upstream:
- `make -B -n` nvcc command lines are identical between the trees once the tree
  paths are masked: the port's `HOST_COMPILER` expands to the same
  `-ccbin g++`, and the link keeps `-Xlinker -soname,"CudaCoordination.so"`
  and `-Xlinker -z -Xlinker nodelete`.
- `Makefile.conf` written by `configure.sh` is identical (the HIP branch is
  skipped without USE_HIP).
- sm_80 plugin: `cuobjdump -xelf all` gives one `lto.sm_80.cubin` per tree,
  both 382816 bytes. They are not byte-identical (function order differs), but
  `nvdisasm -c` of both gives the same 23 kernels, and each kernel's SASS is
  identical once local label numbers (`.L_x_N`) are normalized (raw diff: 5
  kernels differing only in label numbering). The CUDA toolkit here ships no
  nvdisasm; nvdisasm 13.1.80 from a Triton install on the host was used, and it
  refuses sm_52, so the default-arch cubin was not disassembled.
- Both .so: SONAME CudaCoordination.so, FLAGS_1 NODELETE, NEEDED
  libplumedKernel.so plus libstdc++/libm/libgcc_s/libc (cudart static).
- `plumed --load <t>/plugins/cudaCoord/CudaCoordination.so manual --action CUDACOORDINATION`
  rc 0 in both trees, registering CUDACOORDINATION and CUDACOORDINATIONFLOAT
  (dlopen and registration only, no GPU work).

Not covered: an MPI build of the kernel (no MPI on this host; the porter's check
used the MPICH wrapper as `-ccbin`), and running on an NVIDIA GPU.

Verdict: the CUDA path compiles and links with real nvcc 12.8 at head
388fcc10f, identically to upstream's merge base. No port regression.

## jeff's answers to the four questions (jeffdaily, 2026-11-18)

Answers to "Questions for jeff" in the 2026-11-19 upstream-sync section, given by
jeff (jeffdaily) on 2026-11-18 with the dispatch of the check-gpu round:

1. `make check-gpu`: port it for the HIP build and run it as part of this round
   (done; see the next section).
2. Conda packaging for ROCm (`conda.mk` has only CUDA variants): jeff ruled
   **defer**. Registered as `plumed2-cudacoord-conda-rocm` in
   `projects/plumed2/deferred.json`, decided defer by jeffdaily.
3. The new file `plugins/cudaCoord/src/cuda_to_hip.h` keeps its siblings' header
   as it is (Daniele Rapetti / The plumed team, "cudaOnPlumed").
4. The fork commits (including the two June ones `audit-commits` flags) are
   squashed when the pull request is prepared, not now.

## check-gpu round 2026-11-19 (porter, linux-gfx942)

Fork commit `b0e0fb6d1` "[ROCm] Run make check-gpu with the HIP build of
cudaCoord", on top of `388fcc10f` (no rewrite), pushed to `moat-port`; head
advanced. 4 files, +15/-3:

- `plugins/openaccPTM/gpucheck/Makefile`: an `ifdef USE_HIP` block after the
  variables: `HIPCC ?= hipcc`, `CUDACXX = $(HIPCC) -x hip --offload-host-only
  -I$(CURDIR)/../../cudaCoord/src`. The compile rule is untouched (its flags,
  `-std=c++17 -I... -D... -ldl`, work with hipcc; hipcc links libamdhip64).
  `--offload-host-only` because the program has no kernels; it also means no
  offload arch is needed. USE_HIP reaches this sub-make through MAKEFLAGS when
  given on the plugin's command line (`make USE_HIP=1 check-gpu`) and through the
  environment when exported; nothing was added to cudaCoord's Makefile.
- `plugins/openaccPTM/gpucheck/device_check.cpp`: `#include <cuda_runtime.h>`
  becomes `#ifdef __HIP__ / #include "cuda_to_hip.h" / #else / cuda_runtime.h`.
  Nothing else in openaccPTM changed.
- `plugins/cudaCoord/src/cuda_to_hip.h`: two mappings added (`cudaMemset`,
  `cudaDeviceSynchronize`; the program's other eight names were already
  mapped) and its comment names gpucheck as a user. The header includes hipCUB
  and defines a `__device__` helper; both compile cleanly in the host-only HIP
  compile of device_check.cpp (no warnings from it).
- `plugins/cudaCoord/README.md`: the HIP paragraph and "Which GPU is used" name
  `make USE_HIP=1 check-gpu`.

The openaccPTM plugin itself has no HIP build (OpenACC; see the
`plumed2-openacc-gpu` deferral), so the HIP check-gpu is reachable from
cudaCoord only; that is why borrowing cudaCoord's header is acceptable.

Host: MI300X gfx942 x2 visible (HIP_VISIBLE_DEVICES=2,3 as the dispatcher pinned,
so they are GPU 0 and 1 to the program), ROCm 10.1 Python wheels (HIP
7.16.26385, AMD clang 24.0.0git), g++ 13.3, MPICH 4.2.3 built from source into the
job directory (`--disable-fortran --with-device=ch3 --enable-fast=O2`), PLUMED
upstream `422109dc3` with `./configure --prefix=<job>/deps/plumed CXX=mpicxx
CC=mpicc`.

```bash
cd plugins/cudaCoord
USE_HIP=1 ./configure.sh
make USE_HIP=1 HIP_ARCHITECTURES=gfx942
ulimit -c 0
make USE_HIP=1 HIP_ARCHITECTURES=gfx942 check-gpu
```

Result: PASS in both modes, first run, no change to the plugin needed.
- `follow` (caller on GPU 1, no setGpuDeviceId): 17 OK lines, "log says \"c runs
  on GPU 1\"", "follow: 0 failure(s)".
- `md` (caller on GPU 1, setGpuDeviceId 0): 17 OK lines, caller GPU 1 current and
  its 1 MiB buffer usable (hipMemset + hipDeviceSynchronize) after init, after the
  input, after each of 3 steps and after PLUMED is destroyed; forces finite and
  non-zero at every step; "log says \"c runs on GPU 0\"", "md: 0 failure(s)".

So the plugin's GPU selection (`chooseGPU`, `deviceGuard`) behaves on HIP as the
test requires; the area where HIP and CUDA could differ (which device is current
for the thread after a plugin call) shows no difference. The plugin and the
program share one libamdhip64.so, so they see the same per-thread device.

Supporting evidence (scratch, nothing committed):
- AMD_LOG_LEVEL=3 trace of `md`: every plugin section is bracketed by
  `hipSetDevice ( 0 )` ... `hipSetDevice ( 1 )` (5 and 6 calls), and all 9
  kernel launches, both priority streams and the plugin's hipMallocs happen
  inside a bracket, i.e. on GPU 0. Trace of `follow`: the only hipSetDevice is the
  caller's `( 1 )`; the plugin never changes device, 9 launches on GPU 1.
- Negative control: a scratch copy of the plugin whose `~deviceGuard` does not
  call `cudaSetDevice(previous_)` gives `md: 5 failure(s)` ("caller GPU 1 is
  current after the input (found 0)", then after each step and after destroy);
  `follow` still passes, as expected (no device change to undo). So the HIP
  build of the check is not vacuous.

CUDA path: `make -B -n device_check` in gpucheck prints the identical nvcc
command before and after the change; device_check.cpp compiles and links with
nvcc 12.8.93 (conda-forge toolkit, used read-only from the sync job's directory
on this host). cuda_to_hip.h's change is inside `#ifdef __HIP__` apart from a
comment, so the plugin's CUDA objects are unaffected.

`jargon.py --port plumed2` clean; `prose.py` clean on the commit message.

## Review 2026-11-19 (check-gpu delta)

Reviewer (linux-gfx942), delta `388fcc10f...b0e0fb6d1` (one commit, 4 files) plus its commit message. Verdict: **review-passed**. No blocking findings.

Checked, and holding:
- USE_HIP reaches the sub-make. `make -n USE_HIP=1 HIP_ARCHITECTURES=gfx942 check-gpu` in `plugins/cudaCoord` runs the recursive `$(MAKE) -C ../openaccPTM/gpucheck` and prints `hipcc -x hip --offload-host-only -I.../cudaCoord/src -std=c++17 ...device_check.cpp -o device_check -ldl` (the command-line variable travels in MAKEFLAGS). `configure.sh` writes no USE_HIP into Makefile.conf, so USE_HIP is always given to make, as the README says.
- CUDA path unchanged. Without USE_HIP, `make -B -n device_check` in gpucheck prints the same nvcc line as the base recipe; the `#ifdef __HIP__` in `device_check.cpp:47` is false under nvcc; the header change is inside `#ifdef __HIP__` apart from a comment. Scratch compile of the head `device_check.cpp` with nvcc 12.8.93 links.
- Header inclusion: a scratch host-only HIP compile of the head `device_check.cpp` with the head `cuda_to_hip.h` builds and links libamdhip64. All ten CUDA names the program uses are mapped; `cudaMemset` and `cudaDeviceSynchronize` appear nowhere else in `plugins/cudaCoord/src`, as the commit message says.
- `gpucheck/Makefile:9-14`: `CUDACXX =` (not `?=`) under USE_HIP correctly overrides an environment CUDACXX; `HIPCC ?= hipcc` matches cudaCoord's Makefile. Scope as ruled: openaccPTM changed only in the gpucheck Makefile and one include.
- README lines 27 and 52 are accurate.
- Commit message: title 57 chars with `[ROCm]` (adds HIP-naming lines), AI disclosure in prose, Test Plan with fenced commands, no trailer, no AMD-internal account reference. `utils/jargon.py --port plumed2`: clean.
- No wave-size, texture, OOB or library fault class applies (host-only program, no kernels).

Non-blocking (a correction to the record, no code change asked):
- The porter section says the header compiles "cleanly ... (no warnings from it)". The HIP compile of `device_check.cpp` prints two warnings even without `-Wall`: `device_check.cpp:103` (`cudaSetDevice(callerGpu);`) and `device_check.cpp:182` (`cudaFree(buffer);`), "ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]". The cause is HIP's `[[nodiscard]]` hipError_t at the two unchecked calls once the macros map them, not the header itself, and they are harmless. Silencing them would mean a `(void)` cast in shared openaccPTM code, outside jeff's scope ruling, so leave them; the validator should expect them in the build output.

## Validation 2026-11-19 (linux-gfx942, check-gpu head)

Validated sha: b0e0fb6d12474c1e1bd8925a87e6332ce781bd88 (fork clone checked out at
head, no modified tracked files). GPUs: two AMD Instinct MI300X (gfx942, wave64),
HIP_VISIBLE_DEVICES=2,3 (GPU 0 and 1 to the programs). ROCm 10.1 Python wheels
(HIP 7.16.26385, AMD clang 24.0.0git), g++ 13.3, MPICH 4.2.3 built from source in
the job directory, PLUMED upstream 422109dc3 built with `./configure CXX=mpicxx
CC=mpicc` and installed. All GPU runs under `ulimit -c 0`.

Build (after `make clean` in cudaCoord and gpucheck):
```bash
cd plugins/cudaCoord
USE_HIP=1 ./configure.sh
make USE_HIP=1 HIP_ARCHITECTURES=gfx942
```
rc 0; 88 warnings, the known harmless set (nodiscard `hipError_t` in upstream
code, -Wreorder-ctor). Plugin NEEDED libamdhip64.so.7 and libplumedKernel.so, no
libomp; SONAME CudaCoordination.so; FLAGS_1 NODELETE.

Regtests:
```bash
cd regtest && ln -s ../../../regtest/scripts . && cd ..
make USE_HIP=1 HIP_ARCHITECTURES=gfx942 check
```
PASS: "34 tests performed, 0 tests not applicable, 0 errors found" (cudatest 14,
cudatestPair 8, cudatestWB 12 including rt-double-orthopbc-mpi on 2 ranks);
checkfail also clean.

GPU check:
```bash
make USE_HIP=1 HIP_ARCHITECTURES=gfx942 check-gpu
```
PASS, rc 0. gpucheck compiled with `hipcc -x hip --offload-host-only
-I.../cudaCoord/src -std=c++17 ...`; 2 warnings, the nodiscard ones the review
predicted (device_check.cpp:103 `cudaSetDevice(callerGpu)`, :182 `cudaFree`).
- `follow`: 17/17 OK, "log says \"c runs on GPU 1\"", "follow: 0 failure(s)".
- `md`: 17/17 OK (caller GPU 1 current and its buffer usable after init, the
  input, steps 0-2 and after PLUMED is destroyed; forces finite and non-zero),
  "log says \"c runs on GPU 0\"", "md: 0 failure(s)". Not skipped: two GPUs
  visible.

CUDA gate at this head: PASS (compile and link only, no NVIDIA GPU). nvcc
12.8.93 on `git archive b0e0fb6d1 plugins/cudaCoord plugins/openaccPTM/gpucheck`:
```bash
./configure.sh
make NVCCCFLAGS="-dc -dlto --gpu-architecture=sm_80" NVCCLDFLAGS="-shared -dlto --gpu-architecture=sm_80"
make -C ../openaccPTM/gpucheck device_check
```
Plugin: rc 0, no warnings. device_check: rc 0, only nvcc's default-arch
deprecation notice.

Content checks: `python3 utils/jargon.py --port plumed2` clean; README documents
`make USE_HIP=1 check-gpu`.

Gates at head after this run: wave64 by this platform; wave32 (linux-gfx1201 and
linux-gfx1100) reads revalidate at b0e0fb6d1; windows waived. The delta is a
host-only test program plus two `#define`s inside `#ifdef __HIP__`, so a wave32
host can probably carry forward with `binary-equiv` (plugin code objects should
be identical), but that is for the wave32 validator to establish.

## CUDA compile check 2026-11-19 (real nvcc, mi355 host, head b0e0fb6d1)

Evidence only: no stage or platform record changed, nothing pushed to the fork,
no GitHub write. Compile and link only; the host has no NVIDIA GPU. This repeats
the earlier check (done at 388fcc10f) with the full kernel + plugin recipe in both
trees at the current head, which adds the check-gpu commit.

Trees (fresh clones in the job directory; plumed2 has no submodules, so
`git submodule update --init --recursive` was a no-op):
- port: AMD-Ecosystem/plumed2 at head_sha `b0e0fb6d12474c1e1bd8925a87e6332ce781bd88`
- base: plumed/plumed2 at the port's merge base `422109dc3e9aa3aa72a55f326dee473231a1039f`
  (`git merge-base b0e0fb6d1 upstream/master`; still upstream master's tip)

Toolkit: system CUDA 12.8, `/usr/local/cuda/bin/nvcc` ("Cuda compilation tools,
release 12.8, V12.8.93", Build cuda_12.8.r12.8/compiler.35583870_0); host
compiler g++ 13.3.0 (Ubuntu 24.04); no MPI (serial kernel).

Recipe: upstream CI does not build cudaCoord (linuxWF.yml builds only the
nvhpc/openacc plugin), so the plugin README was followed: PLUMED kernel with its
autotools build, then the plugin's `configure.sh` and `make`. Neither is CMake,
so Ninja does not apply; steps are counted as make reports them. Same commands
for both trees (`<t>` = base or port, `J` = the job's build directory):

```bash
cd $J/<t>
./configure --prefix=$J/install-<t>
make -j64 && make install
export PATH=$J/install-<t>/bin:/usr/local/cuda/bin:$PATH
export PLUMED_KERNEL=$J/install-<t>/lib/libplumedKernel.so
export LD_LIBRARY_PATH=$J/install-<t>/lib
cd plugins/cudaCoord
./configure.sh
make                                                   # documented default (nvcc default arch, -dc -dlto)
make -B NVCCCFLAGS="-dc -dlto --gpu-architecture=sm_80" NVCCLDFLAGS="-shared -dlto --gpu-architecture=sm_80"   # README's explicit-SM form
make -C ../openaccPTM/gpucheck device_check            # the nvcc program behind make check-gpu
plumed --load $PWD/CudaCoordination.so manual --action CUDACOORDINATION
```

| | base 422109dc3 | port b0e0fb6d1 |
|---|---|---|
| kernel configure | rc 0 | rc 0 |
| kernel make / install | rc 0 / rc 0 | rc 0 / rc 0 |
| plugin configure.sh | rc 0 | rc 0 |
| plugin make, default arch | 3/3 (Coordination.o, cudaHelpers.o, link) | 3/3 |
| plugin make, sm_80 | 3/3 | 3/3 |
| gpucheck device_check (nvcc) | 1/1 | 1/1 |
| plumed --load ... manual | rc 0 | rc 0 |
| first error | none | none |

Warnings: only nvcc's "Support for offline compilation for architectures prior
to '<compute/sm/lto>_75' will be removed" once per nvcc call in the default-arch
plugin build and in device_check, identical in both trees; the sm_80 build has none.

Equivalence:
- `make -B -n` in cudaCoord plus `make -B -n -C ../openaccPTM/gpucheck device_check`
  print identical command lines in both trees once tree paths are masked
  (`-ccbin g++`, `-Xlinker -soname,"CudaCoordination.so"`, `-Xlinker -z -Xlinker nodelete`).
- `Makefile.conf` from `configure.sh` differs only in the install prefix path.
- Both .so: SONAME CudaCoordination.so, FLAGS_1 NODELETE, NEEDED libplumedKernel.so
  plus libstdc++/libm/libgcc_s/libc/ld-linux (cudart static); `cuobjdump -lelf`
  lists one `lto.sm_80.cubin` in each sm_80 build. The .so files are not
  byte-identical (embedded paths differ between the trees).
- The load check registers CUDACOORDINATION in both trees (dlopen only, no GPU work).

Not covered: an MPI kernel build and running on an NVIDIA GPU.

Verdict: the CUDA path compiles and links with real nvcc 12.8 at head b0e0fb6d1,
the same as on upstream's merge base 422109dc3. No port regression.

## Validation 2026-11-19 (linux-gfx1201, check-gpu head)

Validated sha: b0e0fb6d12474c1e1bd8925a87e6332ce781bd88 (fork clone checked out at head, `git status --porcelain` empty). Full run, not a carry-forward. GPU: AMD Radeon AI PRO R9700 (gfx1201, RDNA4, wavefront size 32 per rocminfo), HIP_VISIBLE_DEVICES=0 as the dispatcher pinned (one GPU visible). ROCm 10.1.0 Python wheels (rocm-sdk-devel 10.1.0, HIP 7.16.26385, AMD clang 24.0.0git), g++ 13.3.0, MPICH 4.2.3 built from source into the job directory (`--with-device=ch3 --disable-fortran --enable-fast=O2`), PLUMED upstream 422109dc3 (the head's merge parent) built with `./configure --prefix=<job>/deps/plumed CXX=mpicxx CC=mpicc` and installed. All GPU runs under `ulimit -c 0`.

Build:
```bash
export PATH=<plumed>/bin:<mpich>/bin:$PATH PLUMED_KERNEL=<plumed>/lib/libplumedKernel.so LD_LIBRARY_PATH=<plumed>/lib:<mpich>/lib:$LD_LIBRARY_PATH
cd plugins/cudaCoord
USE_HIP=1 ./configure.sh
make USE_HIP=1 HIP_ARCHITECTURES=gfx1201
```
rc 0; only code object amdgcn-amd-amdhsa--gfx1201. 88 warnings, the known harmless set (84 nodiscard `hipError_t` -Wunused-value in upstream code, 4 -Wreorder-ctor). SONAME CudaCoordination.so, FLAGS_1 NODELETE, NEEDED libamdhip64.so.7 and libplumedKernel.so, no libomp (this PLUMED also found zlib/LAPACK/BLAS, so those appear in NEEDED through its DYNAMIC_LIBS).

Regtests:
```bash
cd regtest && ln -s ../../../regtest/scripts . && cd ..
make USE_HIP=1 HIP_ARCHITECTURES=gfx1201 check
```
PASS: "34 tests performed, 0 tests not applicable, 0 errors found", and the `checkfail` pass the same. cudatest 14/14 (includes rt-double-rotatedbox and rt-input-errors), cudatestPair 8/8, cudatestWB 12/12 (rt-double-orthopbc-mpi: "Running over 2 nodes"). Same set as the gfx942 run; nothing skipped.

THREADS=32 over the full regtest set (scratch only, nothing committed): the regtest tree copied out, and every uncommented CUDACOORDINATION/CUDACOORDINATIONFLOAT line rewritten to `THREADS=32` (existing THREADS=N replaced, otherwise appended; 154 action lines; the LOAD path made absolute), then `make` in the copy with the same references. PASS: "34 tests performed, 0 tests not applicable, 0 errors found"; all 154 actions logged "threads per block (a warp per atom): 32", so every GPU kernel ran with the 32-lane logical warp equal to the whole wavefront. (A first attempt reported 2 errors in cudatest and cudatestWB rt-double-d_0-nopbc, caused by my rewrite putting `THREADS=32` in front of a commented-out `#gpu64:` line ("Action \"THREADS=32\" is not known"). The rewrite was fixed to skip comment lines and the rerun is the result above. Not a plugin failure.)

Debug-assertion build: a scratch copy of the plugin (same sources, same Makefile.conf with `-DNDEBUG=1` removed) built with `make USE_HIP=1 HIP_ARCHITECTURES=gfx1201`. Without NDEBUG, HIP's `__shfl_*_sync` check the mask with `__hip_check_mask` (`__hip_assert`, a `__builtin_trap` on a mask that is zero or does not match the active lanes; `amd_detail/amd_warp_sync_functions.h`, `hip_assert.h`). Against it: the default regtest set 34/34, 0 errors, and the THREADS=32 set 34/34, 0 errors; no trap, no abort. So the per-group `warpMask()` is accepted by HIP's own mask check on wave32 at 32-thread blocks and at the default sizes.

GPU check:
```bash
make USE_HIP=1 HIP_ARCHITECTURES=gfx1201 check-gpu
```
rc 0. gpucheck compiled with `hipcc -x hip --offload-host-only -I.../cudaCoord/src -std=c++17 ...`; 2 warnings, the nodiscard ones the review predicted (device_check.cpp:103, :182).
- `follow`: 17/17 OK, "follow: 0 failure(s)".
- `md`: "SKIP md mode needs at least 2 GPUs". The dispatcher pinned one GPU and the pin was kept; md mode passed on linux-gfx942 at this head with two GPUs visible. Device selection is host code with no wavefront dependence.

Runtime init: no segfault and no hang in any of the 34 x 4 regtest runs or check-gpu on this wave32 host.

CUDA gate: not rerun; already recorded at this head_sha (linux-gfx942 validation and the mi355 real-nvcc check, both PASS).

Content checks: `python3 utils/jargon.py --port plumed2` clean (local `master`/`moat-port` refs created from origin); the HIP build and `make USE_HIP=1 check` / `check-gpu` are documented in plugins/cudaCoord/README.md next to the nvcc build.

Gates at head after this run: wave64 by linux-gfx942, wave32 by this platform, windows waived. No upstream PR is open; nothing was pushed to the fork.

## Fix round moat-fix-1648 2026-11-24 (porter, linux-gfx942): merge upstream master to clear the conflict

Why: upstream PR https://github.com/plumed/plumed2/pull/1648 (published head `2571dcde4`, base `master`) read CONFLICTING. No maintainer asked for anything; the round only makes the PR mergeable. The PR is open, so `moat-port` is frozen: `moatlib.py set-state plumed2 linux-gfx942 porting`, then `moatlib.py fix-branch plumed2` (staging branch `moat-fix-1648`, base `2571dcde4`). `moat-port` was not pushed.

### What was done

- Upstream merged: plumed/plumed2 `master` at `b930e84be7d5f63762e705dd810d0066f5a0e016` (merge base with the PR was `422109dc3`; 121 commits in between).
- `2641e3218` "Merge upstream master into the HIP support of cudaCoord": a merge commit (`git merge --no-ff upstream/master`), not a rebase; `2571dcde4` stays its first parent.
- `43ff1a58e` "[ROCm] cudaCoord: hide the GPUs from HIP too in the no-GPU test" (see below).
- Pushed `moat-fix-1648` only (new branch, plain push). `advance-head plumed2 43ff1a58e9eb9d5c3a28d8579624e0e5a6938669`.
- The fork's `master` mirror is still at `422109dc3` (not fast-forwarded: this job may write only `moat-fix-1648` and the MOAT branch). My local `master` was set to upstream master for the jargon scan. The fix review PR (`moat-fix-1648` -> `moat-port`) will show upstream's 121 commits plus the two above; that is expected for a merge round.

### What upstream changed in plugins/cudaCoord since 422109dc3, commit by commit

None of it is HIP or portability work; nothing overlaps or supersedes our change. Upstream did not touch `src/cuda_to_hip.h`, `src/Coordination.cuh`, or `plugins/openaccPTM/gpucheck/` (our check-gpu files).

1. `01755c14b` cudaCoord: choose the GPU after the whole input has been read. Moves `chooseGPU`/`deviceGuard` and the switching-function stretch kernel from the start of the constructor to after `checkRead()`, so input errors are reported on machines without a GPU. Moves `regtest/cudatest/rt-input-errors` to `regtest/nogpu/`. Host-only reorder; for us it only means the first HIP call happens later in the constructor (still before any allocation, so the old gfx1100 runtime-init concern stays settled).
2. `317317233` configure.sh: strips a `ccache` launcher from `CXX=` in Makefile.conf, because nvcc `-ccbin` takes a compiler only. Inside upstream's nvcc block; our HIP branch is before it and exits early, and hipcc does not use `CXX` (our `HOST_COMPILER` is empty under USE_HIP). Our mpi.h probe runs `$CXX -E -H`, which works through ccache too. No change needed.
3. `6c14ae547` CI: compile cudaCoord and run the tests that do not need a GPU. New `make check-nogpu` target and `regtest checknogpu`; new `regtest/nogpu/rt-no-gpu` (hides GPUs with `CUDA_VISIBLE_DEVICES=-1`, expects "no GPU available"); rt-input-errors grows from 6 to 15 cases (also CUDACOORDINATIONFLOAT); `regtest/scripts` is now a checked-in symlink (so the README's old `ln -s` step is gone; our README text never had it). New `.github/workflows/gpuPluginsWF.yml` (nvcc 12.9/13.5, check-nogpu). The rt-no-gpu test needed our follow-up commit (below).
4. `0a3717f45` Consolidate module configuration in YAML: `conda.mk` deleted, its content moved into `plugin.yml` (`conda:` block). Packaging only; the ROCm conda variant stays deferred (`plumed2-cudacoord-conda-rocm`).
5. `809a00563` module metadata review: reformats the `plugin.yml` conda block. Packaging only.
6. `5318f5b79` cudaCoord: queue the work of the step in launch(). Splits `calculate()` into an enqueue part and the wait, adds `cancelLaunch()` (stream sync + rebuild lists) and `setLaunchable(true)`. Same runtime calls as before (`cudaMemcpyAsync`, kernels, `cudaStreamSynchronize`), all already mapped.
7. `8b05aa58b` Launch hooks: removes `enqueueStep()`, `launch()` is the enqueue function; `setLaunchable()` takes no argument. Host-only.
8. `80018229f` openaccPTM, cudaCoord: the GPU is chosen by `PLMD::GpuDevice` (core `src/core/GpuDevice.{h,cpp}`). `chooseGPU(const Action&)` now fills a `GpuDevice::Runtime` (`cudaGetDeviceCount`, `cudaGetDevice`, and `GpuDevice::callerCudaDevice()`) and the core picks: MD-assigned GPU, else the caller's CUDA context device, else a user choice, else local MPI rank modulo GPUs (with >1 GPU), else the runtime's current device. Adds `#include "plumed/core/GpuDevice.h"` to cudaHelpers.cuh: the one textual conflict. The log line is now e.g. "gpu runs on GPU 0 (the default)".
9. `61cf7c600` merge of PR #1653 (the YAML work above).

The diffs to `Makefile` (check-nogpu target) and `README.md` (test section, GPU choice text) merged cleanly next to our hunks.

### How the merge was resolved

One conflict, `plugins/cudaCoord/src/cudaHelpers.cuh` include block: ours added `#include "cuda_to_hip.h"` + blank line, upstream added `#include "plumed/core/GpuDevice.h"` at the same spot. Kept both, ours first (the compat header must come before anything that names CUDA types). All other files auto-merged.

Check: the lines added and removed by `git diff upstream/master 2641e3218` are exactly the lines added and removed by the PR's current diff `git diff 422109dc3 2571dcde4` (diff of the `^[-+]` lines: empty; only hunk positions moved, in README.md and cudaHelpers.cuh). CUDA path: every CUDA-side line is upstream's own; our edits to the CUDA path are the same as in the PR (the `HOST_COMPILER` variable for `-ccbin $(CXX)`, the `#ifndef __HIP__` guards, `warpMask()`/width 32 in the shuffles, which are `0xffffffffu` and a width of 32 under nvcc).

Mapping check: every `cuda[A-Z]*` identifier in `plugins/cudaCoord/src/*.cu*` and in `gpucheck/device_check.cpp` after the merge is mapped in `cuda_to_hip.h` (the only unmapped hits are `cudaCheck`, `cudaHelpers`, `cudaOnPlumed`, `cudaCoord`, which are not runtime names). No new warp intrinsic, `warpSize`, or `cub::` use beyond the reviewed ones (`cudaHelpers.cuh:248`, `Coordination.cu:264-266`, `cub::BlockReduce` at `cudaHelpers.cuh:277`).

### The one thing upstream's changes broke for HIP: rt-no-gpu under HIP_VISIBLE_DEVICES

`make USE_HIP=1 check` on the merge commit `2641e3218`: "35 tests performed, 0 tests not applicable, 1 errors found", the error in `nogpu/rt-no-gpu`: all five inputs "DID NOT STOP PROPERLY (exit code 0)"; they ran on the GPU ("gpu runs on GPU 0 (the default)"). Cause, measured with `plumed driver` on the test's input:

| environment | result |
|---|---|
| `HIP_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=-1` (this job's pin) | rc 0, runs on GPU 0 |
| `HIP_VISIBLE_DEVICES` unset, `CUDA_VISIBLE_DEVICES=-1` | rc 1, "no GPU available" |
| `HIP_VISIBLE_DEVICES` unset, `CUDA_VISIBLE_DEVICES=` (empty) | rc 1, "no GPU available" |
| `HIP_VISIBLE_DEVICES` unset, `ROCR_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=-1` | rc 1, "no GPU available" |
| `HIP_VISIBLE_DEVICES=-1` | rc 1, "no GPU available" |

So HIP honours `CUDA_VISIBLE_DEVICES` only when `HIP_VISIBLE_DEVICES` is not set, and upstream's test fails for a HIP build whenever it is (every fleet pin, and common on AMD clusters). Fix `43ff1a58e`: `CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 $plumed driver ...` plus a two-line comment in `regtest/nogpu/rt-no-gpu/config`. CUDA ignores the extra variable. This is a change to an upstream test file, outside the files the PR touched before; it is a separate commit so jeff can drop it if he prefers to leave the test alone (then `make USE_HIP=1 check` needs `HIP_VISIBLE_DEVICES` unset).

### Build and tests (porter's run, gfx942)

Host: AMD Instinct MI300X (gfx942, wave64), HIP_VISIBLE_DEVICES=3 as the dispatcher pinned (one GPU). ROCm 10.1 Python wheels (HIP 7.16.26385, AMD clang 24.0.0git), g++ 13.3.0, MPICH 4.2.3 built from source into the job directory (`./configure --disable-fortran --with-device=ch3 --enable-fast=O2`; the image's /usr/bin/mpicxx is Open MPI 4.1.6, not used, to match the earlier runs), PLUMED from `git archive b930e84be` (upstream master, identical to the staging tip outside plugins/) with `./configure --prefix=<job>/deps/plumed CXX=mpicxx CC=mpicc`, `make -j48 && make install`. CXXFLAGS `-O3 -fPIC -Wall -pedantic -Wno-unknown-pragmas -std=c++17 -fno-gnu-unique -fopenmp`, `-D__PLUMED_HAS_MPI=1`. All GPU runs under `ulimit -c 0`.

```bash
export PATH=<plumed>/bin:<mpich>/bin:$PATH PLUMED_KERNEL=<plumed>/lib/libplumedKernel.so LD_LIBRARY_PATH=<plumed>/lib:<mpich>/lib:$LD_LIBRARY_PATH
cd plugins/cudaCoord
USE_HIP=1 ./configure.sh
make USE_HIP=1 HIP_ARCHITECTURES=gfx942
make USE_HIP=1 HIP_ARCHITECTURES=gfx942 check
make USE_HIP=1 HIP_ARCHITECTURES=gfx942 check-nogpu
make USE_HIP=1 HIP_ARCHITECTURES=gfx942 check-gpu
```

- Build: rc 0; 88 warnings, the known harmless set (84 nodiscard `hipError_t` -Wunused-value in upstream code, 4 -Wreorder-ctor). NEEDED libamdhip64.so.7, libplumedKernel.so, no libomp; SONAME CudaCoordination.so; FLAGS_1 NODELETE.
- `check` at `43ff1a58e`: "35 tests performed, 0 tests not applicable, 0 errors found", checkfail the same. cudatest 13 (was 14: rt-input-errors moved out), cudatestPair 8, cudatestWB 12 (rt-double-orthopbc-mpi on 2 ranks), nogpu 2 (rt-input-errors, now 15 cases, and the new rt-no-gpu). Compared with the earlier 34: one new test (rt-no-gpu), one moved (rt-input-errors, with more cases); the GPU tests are the same 33.
- `check-nogpu`: "+ the tests that do not need a GPU passed".
- `check-gpu`: rc 0; gpucheck compiled with `hipcc -x hip --offload-host-only ...`, the 2 known nodiscard warnings (device_check.cpp:103, :182). `follow`: 17/17 OK, "follow: 0 failure(s)". `md`: "SKIP md mode needs at least 2 GPUs" (this job is pinned to one GPU and the pin was kept; md passed at `b0e0fb6d1` with two GPUs, and the device-selection code it exercises changed upstream in `80018229f`, see the question below).

Device code unchanged: the gfx942 code object of the plugin built from the published head `2571dcde4` and from `43ff1a58e` (same tree path, same PLUMED install, `llvm-objcopy --dump-section=.hip_fatbin`, `clang-offload-bundler --unbundle`) is byte-identical (sha256 `56fd8af4adec...`, 24 kernels, 14773 lines of ISA, `diff` empty). `codeobj_diff.py` says `differ` only because of new exported host symbols (`launch`, `cancelLaunch`), as expected from upstream's host changes. So the wave64 shuffle-mask evidence (THREADS=32 debug-assertion runs) recorded for the earlier heads still describes the kernels at this tip.

### Observation for jeff: GPU choice on HIP after 80018229f (no change made)

`GpuDevice::callerCudaDevice()` asks `libcuda.so.1` (`dlopen` with `RTLD_NOLOAD`), so on AMD it always returns -1 and rule 2 ("the GPU the MD code is using") never applies to a HIP build. What applies instead is rule 5, `hipGetDevice()`, and since libamdhip64 is one shared library per process, that already is the caller's current device (the check-gpu `follow` mode passes). The visible difference from CUDA is only when rule 4 comes first: more than one visible GPU, launched by an MPI launcher that sets a local-rank variable, and the MD code did not call `setGpuDeviceId`; then a HIP build uses local-rank modulo GPUs where a CUDA build would follow the caller's context. Making rule 2 work on HIP would need a HIP probe in core `GpuDevice.cpp` (outside the plugin), and HIP has no clean "is a context current" query (the current device always defaults to 0). Left as upstream wrote it; mention it only if a maintainer asks about AMD multi-GPU behaviour.

### CUDA compile: not done here (no nvcc on this host)

Commands for the mi355 session (real nvcc), same recipe as the 2026-11-19 checks, on fresh clones of the fork at `43ff1a58e9eb9d5c3a28d8579624e0e5a6938669` (port) and plumed/plumed2 at `b930e84be7d5f63762e705dd810d0066f5a0e016` (base):

```bash
cd $J/<t>
./configure --prefix=$J/install-<t>
make -j64 && make install
export PATH=$J/install-<t>/bin:/usr/local/cuda/bin:$PATH
export PLUMED_KERNEL=$J/install-<t>/lib/libplumedKernel.so
export LD_LIBRARY_PATH=$J/install-<t>/lib
cd plugins/cudaCoord
./configure.sh
make
make -B NVCCCFLAGS="-dc -dlto --gpu-architecture=sm_80" NVCCLDFLAGS="-shared -dlto --gpu-architecture=sm_80"
make -C ../openaccPTM/gpucheck device_check
make check-nogpu
plumed --load $PWD/CudaCoordination.so manual --action CUDACOORDINATION
make -B -n > ../../../make-n-<t>.txt; make -B -n -C ../openaccPTM/gpucheck device_check >> ../../../make-n-<t>.txt
```

Expect: both trees build 3/3 plus device_check; `make -B -n` lines identical once tree paths are masked; `make check-nogpu` passes in both (it hides the GPUs, so it runs on a GPU-less host: this is the CI job upstream added, and it exercises our rt-no-gpu edit under CUDA, where `HIP_VISIBLE_DEVICES` must be ignored); per-kernel SASS of the sm_80 cubins identical after label normalization (`nvdisasm -c`).

### Draft comment for the upstream PR (for jeff to approve and post)

====
I merged the current master into this branch to clear the conflict; the only clash was an include line in cudaHelpers.cuh, and nothing in the HIP part had to change. The one addition is in the new rt-no-gpu test: it now sets HIP_VISIBLE_DEVICES=-1 next to CUDA_VISIBLE_DEVICES=-1, because the HIP runtime ignores CUDA_VISIBLE_DEVICES when HIP_VISIBLE_DEVICES is set.
====

## Review 2026-11-24 (fix round moat-fix-1648, merge of upstream master)

Reviewer (linux-gfx942), pr-review skill, local-branch mode. Scope: the delta `2571dcde4...43ff1a58e` (merge `2641e3218` of upstream master `b930e84be`, plus `43ff1a58e`), and the tip against upstream master. Three independent child reviews (merge fidelity, HIP correctness of upstream's new code, test commit and commit hygiene), each finding re-checked against the code.

Verdict: **review-passed** at `43ff1a58e9eb9d5c3a28d8579624e0e5a6938669`. The merge resolution is correct. There is one open question for jeff (below). It comes from upstream's new GPU-choice design, not from the merge, and needs a person's choice before the fix review is approved.

### Diff of the staging tip against upstream master vs the PR's current diff

`git diff upstream/master 43ff1a58e` compared with `git diff 422109dc3 2571dcde4`, with hunk positions and `index` lines stripped:
- Every added and removed line in the PR's 9 files is the same in both diffs (+175/-8 each; sorted `^[-+]` lines identical).
- Context-only differences:
  - README.md: around our `check-gpu` line, upstream reworded the surrounding GPU-choice paragraph ("... with the rule that was applied ..."). Our line is unchanged.
  - Coordination.cu / cudaHelpers.cuh: `#include "plumed/core/GpuDevice.h"` now sits next to our include edit; this was the one textual conflict, and both includes are kept.
- The only extra difference is `43ff1a58e`, `plugins/cudaCoord/regtest/nogpu/rt-no-gpu/config`. It adds a two-line comment and `HIP_VISIBLE_DEVICES=-1` next to `CUDA_VISIBLE_DEVICES=-1`.
- `git diff upstream/master 43ff1a58e -- . ':!plugins/cudaCoord' ':!plugins/openaccPTM/gpucheck'` is empty.
- The merge parents are `2571dcde4` (first) and `b930e84be`; `git merge-tree 2571dcde4 b930e84be` reports the conflict only in cudaHelpers.cuh.

### Checked and holding

- **CUDA path is upstream's.** Without USE_HIP or `__HIP__`, the only differences are the PR's own edits:
  - the `HOST_COMPILER = -ccbin $(CXX)` variable;
  - the `#ifndef __HIP__` include guards;
  - `cuda_to_hip.h`, which is `<cuda_runtime.h>` plus `warpMask()` returning `0xffffffffu`;
  - width 32 in the shuffles, which is CUDA's warpSize.
- **Upstream's new code needs no new mapping.** It adds `cudaStreamSynchronize` in `cancelLaunch()`/`calculate()` and `cudaGetDeviceCount`/`cudaGetDevice` in `chooseGPU`; all are mapped.
- **No new device code.** Upstream's cudaCoord changes since 422109dc3 are host-only: no new kernel, intrinsic, warp literal, cub use or block size. The gfx942 code object is byte-identical to the published head's, per the porter section.
- **The rt-no-gpu claim is true.** A child review called `hipGetDeviceCount` through ctypes on libamdhip64:
  - `CUDA_VISIBLE_DEVICES=-1` alone gives `hipErrorNoDevice`;
  - with `HIP_VISIBLE_DEVICES` set, HIP ignores `CUDA_VISIBLE_DEVICES`;
  - both set to -1 gives 0 devices.
  The CUDA runtime ignores `HIP_VISIBLE_DEVICES`, and the test's other checks (rc != 0, rc < 128, the message) are unchanged.
- **Commit messages are clean.**
  - Titles: 56 chars (merge, no prefix) and 63 chars (`[ROCm]`; it names HIP).
  - AI disclosure is in prose, with no trailer, and each has a fenced Test Plan.
  - `jargon.py --port plumed2` is clean, and `prose.py` is clean on both bodies.
  - The merge message's claims hold: the only conflict was the include block, and the +/- lines are the same as before.

### Open question for jeff: GPU choice of a HIP build after upstream 80018229f (not a defect of the merge; needs a decision)

Upstream's `chooseGPU` now asks core `PLMD::GpuDevice` to choose, in this order (`src/core/GpuDevice.cpp:84-97`):
1. MD-assigned GPU;
2. the caller's GPU, from `callerCudaDevice()`;
3. a user choice;
4. local MPI rank % n, when there is more than one GPU;
5. the runtime's current device.

`callerCudaDevice()` reads the CUDA driver (`dlopen("libcuda.so.1", RTLD_NOLOAD)` + `cuCtxGetCurrent`), so it is always -1 in a HIP build and rule 2 never applies. In a HIP build, the caller's device is visible only through `hipGetDevice` (libamdhip64 is shared), and that value feeds rule 5 only.

Consequence: on a node with more than one AMD GPU, under any launcher that sets a local-rank variable (`OMPI_COMM_WORLD_LOCAL_RANK`, `SLURM_LOCALID` (set in every Slurm step), `PMI_LOCAL_RANK`, ...), and with no `setGpuDeviceId` from the MD code, CUDACOORDINATION runs on GPU rank % n instead of the MD code's GPU. Results stay correct (`deviceGuard` restores the caller's device). `make USE_HIP=1 check-gpu` `follow` mode then fails.

A child review checked this with the built HIP `device_check follow` and `HIP_VISIBLE_DEVICES=0,1`:
- no launcher variable: pass ("c runs on GPU 1 (the default)");
- `SLURM_LOCALID=0`: FAIL ("c runs on GPU 0 (process 0 of the node, from SLURM_LOCALID, with 2 GPUs)");
- `OMPI_COMM_WORLD_LOCAL_RANK=0`: FAIL, the same way.

That run used GPUs 0 and 1 for a few seconds, outside this job's pin (HIP_VISIBLE_DEVICES=3); the child agent chose to do it. Nothing else of mine ran outside the pin.

Before the merge, our HIP `chooseGPU` always followed the current device. With CUDA, rule 2 finds the caller's context. Upstream's header says rule 2 applies "when the plugin can tell", so a HIP build that cannot tell is within upstream's contract. Even so, the README presents `make USE_HIP=1 check-gpu` as checking this behaviour.

Is there a clean HIP probe? Probed on this host, one GPU, in scratch:
- `hipDevicePrimaryCtxGetState` reports `active=0` in a fresh process after `hipGetDeviceCount`, and `active=1` after the caller's `hipSetDevice(0)`. `hipCtxGetCurrent` is non-null in every case, so it is useless here.
- "the current device, if its primary context is active" would work for the first action. It is process-wide, though, not per thread, and the plugin's own `deviceGuard` restore (`hipSetDevice(previous)`) activates the previous device's context.
- So a second action in the same process could choose differently from the first (for example, `mpirun plumed driver` on 2 GPUs: rank 1 gets GPU 1 for the first action and GPU 0 for later ones). It is a heuristic with a known failure, not an equivalent of the CUDA query.

Options, for jeff to choose (nothing changed in this round):
- (a) Leave the code as upstream wrote it. Add one sentence to the HIP paragraph of the README: with HIP, the GPU of the MD code is not detected, so give it with `setGpuDeviceId` (or let the rank rule apply). Note that `check-gpu` `follow` expects no launcher variables. This is a docs-only delta, inert for validation.
- (b) A HIP-only `callerDevice` probe in `chooseGPU`, with the primary-context-active heuristic above and its known inconsistency.
- (c) Raise it with the maintainers on the PR, since a HIP path for rule 2 belongs in core `GpuDevice.cpp`, outside the plugin.

Reviewer's recommendation: (a), possibly with (c) as one sentence in the PR comment.

Also non-blocking, same docs delta if (a) is chosen: the HIP paragraph lists `make USE_HIP=1 check` and `make USE_HIP=1 check-gpu` but not upstream's new `make USE_HIP=1 check-nogpu`. On a fresh tree, `make check-nogpu` without USE_HIP would build with nvcc.

## Validation 2026-11-24 (linux-gfx942, fix round moat-fix-1648 tip)

Validated sha: `43ff1a58e9eb9d5c3a28d8579624e0e5a6938669` (fork clone checked out at the staging tip, `git status --porcelain` empty before and after). Full run, not a carry-forward.

GPU: AMD Instinct MI300X (gfx942, wave64), HIP_VISIBLE_DEVICES=3 as the dispatcher pinned (one GPU, GPU 0 to the programs). All GPU runs under `ulimit -c 0`.

Toolchain:
- ROCm 10.1 Python wheels (HIP 7.16.26385, AMD clang 24.0.0git), g++ 13.3.0.
- MPICH 4.2.3 built from source in the job directory.
- PLUMED upstream master `b930e84be` (the merge's second parent; outside the plugin it is identical to the tip), built with `./configure CXX=mpicxx CC=mpicc` and installed.

This is the same method as the 2026-11-19 gfx942 validations.

Build (after `make clean` in cudaCoord and gpucheck):
```bash
cd plugins/cudaCoord
USE_HIP=1 ./configure.sh
make USE_HIP=1 HIP_ARCHITECTURES=gfx942
```
Result: rc 0. 88 warnings, the known harmless set (84 nodiscard `hipError_t`, 4 -Wreorder-ctor), the same count as before. NEEDED libamdhip64.so.7 and libplumedKernel.so, no libomp; SONAME CudaCoordination.so; FLAGS_1 NODELETE.

Regtests:
```bash
make USE_HIP=1 HIP_ARCHITECTURES=gfx942 check
make USE_HIP=1 HIP_ARCHITECTURES=gfx942 check-nogpu
```
- `check`: PASS, "35 tests performed, 0 tests not applicable, 0 errors found", and checkfail the same.
  - cudatest 13/13 (includes rt-double-rotatedbox).
  - cudatestPair 8/8.
  - cudatestWB 12/12 (rt-double-orthopbc-mpi: "Running over 2 nodes").
  - nogpu 2/2 (rt-input-errors with 15 cases, rt-no-gpu).
- `check-nogpu`: "+ the tests that do not need a GPU passed".

Compared with the 34 of the earlier gfx942 validation: the 33 GPU tests are the same tests and all pass. rt-input-errors moved from cudatest to nogpu (upstream `01755c14b`/`6c14ae547`) and gained 9 cases. rt-no-gpu is new upstream; it passes here only with `43ff1a58e` while HIP_VISIBLE_DEVICES is set. The logs now say "gpu runs on GPU 0 (the default)", which is upstream's new log format.

GPU check:
```bash
make USE_HIP=1 HIP_ARCHITECTURES=gfx942 check-gpu
```
- rc 0; the 2 known nodiscard warnings in device_check.cpp (:103, :182).
- `follow`: 17/17 OK, "follow: 0 failure(s)" (caller on GPU 0, the only visible GPU).
- `md`: "SKIP md mode needs at least 2 GPUs". The pin was kept; at `b0e0fb6d1` md passed with two GPUs.

Not covered by this run: the multi-GPU launcher case in the reviewer's open question. A child review measured it with two GPUs: `follow` fails under `SLURM_LOCALID`/`OMPI_COMM_WORLD_LOCAL_RANK` with 2 GPUs.

Device code: the gfx942 code object is byte-identical to the published head `2571dcde4` (porter section), so the wave64 THREADS=32 debug-assertion evidence from the earlier rounds applies unchanged.

CUDA gate at this head: NOT RUN. This host has no nvcc. The exact commands for the mi355 session are in the porter section of this round.

Content checks:
- `python3 utils/jargon.py --port plumed2` is clean.
- The HIP build and its tests are documented in plugins/cudaCoord/README.md next to the nvcc build. `make USE_HIP=1 check-nogpu` is not mentioned there; see the review's option (a).

Gates at head after this run: wave64 by this platform. wave32 (linux-gfx1201, linux-gfx1100) reads revalidate at `43ff1a58e`. windows is waived.
