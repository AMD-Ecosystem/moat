# AutoDock-GPU notes

Already runs on AMD via OpenCL (DEVICE=OCLGPU); this port adds a native HIP/ROCm
backend (DEVICE=HIP) as an improvement, not to fill a gap. The HIP backend ports
the maintained CUDA kernels (cuda/) rather than the parallel OpenCL copy
(device/), so AMD users get the same kernels NVIDIA users get.

## Backend selection (top-level Makefile)

- DEVICE=CUDA / DEVICE=GPU (when nvcc + a CUDA test compile pass) -> Makefile.Cuda
  (nvcc compiles cuda/kernels.cu; g++ links the host .cpp + -lcurand -lcudart).
- DEVICE=OCLGPU / DEVICE=OPENCL / DEVICE=CPU -> Makefile.OpenCL (device/ kernels).
- DEVICE=HIP (NEW) -> Makefile.Hip: hipcc compiles cuda/kernels.cu with
  --offload-arch, g++ links the host .cpp + -lamdhip64. Bypasses the nvcc probe.
cuda/kernels.cu is one TU that #includes the other cuda/*.cu.

## Port = Strategy A (compat header + .cu compiled by hipcc)

- cuda/cuda_to_hip.h: the only file that knows HIP names. Under USE_HIP it
  #includes <cstring>/<cstdlib> BEFORE <hip/hip_runtime.h> (so host memcpy binds
  to libc, not HIP device overloads) and aliases the cuda* runtime symbols the
  project uses to hip*; on NVIDIA it is a plain <cuda_runtime.h> include, so the
  CUDA build is byte-for-byte unchanged. Included from cuda/kernels.cu,
  cuda/GpuData.h, and host/inc/performdocking.h.
- The HIP build defines USE_CUDA too (performdocking.h) so the ~30 shared
  `#ifdef USE_CUDA` GPU-host blocks compile unchanged; the NVIDIA-only includes
  (cuda.h/curand.h/cuda_runtime_api.h) are gated to NOT fire under USE_HIP.
  curand is never actually called (PRNG seeds are made on the host CPU; the
  device PRNG is a hand-rolled LCG in auxiliary_genetic.cu), so no hipRAND link
  is needed.
- Makefile.Hip kernel TU needs -fPIE: g++ links the host as PIE by default, so
  the hipcc-built kernels.o must be position-independent or ld rejects the
  R_X86_64_32 relocation.

## Fault classes hit

### Warp size / wave64 (the real one)
gfx90a is wave64. AutoDock-GPU is partly warp-size parameterized: kernels read
cData.warpmask / cData.warpbits, but the HOST hardcoded them to 32-lane values
and a few device spots assumed 32 lanes. Ported NATIVELY to wave64 (warpmask=63,
warpbits=6), not as two 32-lane halves, because the reductions are genuinely
warp-wide and combine per-warp partials via a block-wide atomicAdd, so a true
64-lane warp is the natural mapping and matches the upstream parameterization.
Fixes:
- host/src/performdocking.cpp ~785: derive warpmask/warpbits from the device's
  props.warpSize at runtime (cudaGetDeviceProperties) instead of the literals
  31/5. Yields 31/5 on CUDA (warpSize 32) and 63/6 on gfx90a (64); zero behavior
  change for NVIDIA.
- cuda/kernels.cu: device-side WARP_SIZE constant (64 for __GFX9__, else 32, else
  CUDA 32). WARPMINIMUM2 gains a stride-32 exchange under wave64; REDUCEFLOATSUM /
  REDUCEINTEGERSUM gain a stride-32 shuffle-add under wave64.
- cuda/kernel4.cu:93 `blockDim.x / 32` -> `blockDim.x / WARP_SIZE` (warp count for
  the final cross-warp reduction).
- HIP mask width: ROCm 7.2 static_asserts that __shfl_sync/__any_sync/__ballot_sync
  take a 64-BIT mask (sizeof==8) regardless of active wave width. WARP_FULL_MASK is
  0xffffffffffffffffULL under USE_HIP, 0xffffffffU on CUDA. (Keyed on USE_HIP, NOT
  on WARP_SIZE>32 -- a 32-bit literal fails to compile even before any wave64
  concern.)

### Inline PTX
cuda/kernels.cu llitoulli/ullitolli use asm("mov.b64"). NVIDIA-only; replaced with
a memcpy bit-cast under USE_HIP (they are pure 64-bit reinterprets; in fact unused
in the codebase, but kept and guarded to preserve the CUDA path).

### Not hit
Atomics: only atomicAdd on int/float (no int atomicMin/Max -> no CAS-emulation
needed). No textures/surfaces -> none of those fault classes apply. One
cudaMallocManaged buffer (pMem_gpu_evals_of_runs) but it is only touched by
atomicAdd and a single-thread +=, so the managed-memory int-atomicMin/Max drop
class does not bite.

## Build (gfx90a)

    cd projects/AutoDock-GPU/src
    make DEVICE=HIP NUMWI=64 HIP_ARCH=gfx90a       # one wavefront/block
    make DEVICE=HIP NUMWI=128 HIP_ARCH=gfx90a      # two wavefronts/block

Arch is configurable (HIP_ARCH ?= gfx90a, default only when unset); followers
build with HIP_ARCH=gfx1100 / gfx1151 and no source/Makefile edit. ROCm 7.2.1,
HIP 7.2, on a 4x MI250X (gfx90a) host. Build artifacts (bin/, kernels.o) are
gitignored upstream.

## Validation -- REAL GPU, gfx90a (HIP_VISIBLE_DEVICES=3, MI250X)

Input: shipped input/1stp (streptavidin-biotin, the canonical AutoDock case),
--nrun 10, fixed --seed. Best binding energy + best-pose reference RMSD (RMSD to
the crystal ligand) from the .dlg RMSD table:

| build / seed     | best binding energy | best-pose reference RMSD | cluster      |
|------------------|---------------------|--------------------------|--------------|
| 64wi  / seed 42  | -8.28 kcal/mol      | 0.48 A                   | 1, 10/10 runs|
| 64wi  / seed 7   | -8.37 kcal/mol      | 0.37 A                   | top cluster  |
| 128wi / seed 42  | -8.33 kcal/mol      | 0.39 A                   | top cluster  |

All runs recover the native biotin pose (sub-0.5 A reference RMSD), best energy
clustered tightly around -8.3 kcal/mol, all 10 runs converging into a single
cluster -- the known-good 1stp result. The GA min-reduction (kernel4 elitism)
and the energy/gradient sum-reductions (calcenergy / calcMergeEneGra) are thus
numerically correct on wave64. The 128wi case (2 wavefronts/block) additionally
exercises the cross-warp final reduction in kernel4 and is also correct.

Cross-check against the OpenCL backend (DEVICE=OCLGPU) was attempted as a
same-machine reference but the upstream OpenCL kernels fail to compile under the
ROCm 7.2 OpenCL runtime (a pre-existing argument-type error in gradient_inter_z,
unrelated to this port) -- which only reinforces the value of the native HIP
backend. The HIP result stands on its own against the established 1stp reference.

## Scope / deliverables

Source edits live as working-tree changes under projects/AutoDock-GPU/src (parent
delivers; not forked/pushed here). 6 files modified + 2 new (cuda/cuda_to_hip.h,
Makefile.Hip); CUDA and OpenCL/CPU paths untouched and behaviorally unchanged. No
GitHub Actions added, README not regenerated.

## Validation 2026-05-30 (gfx1100, ROCm 7.2.1)

Platform: 4x AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), ROCm 7.2.1,
HIP 7.2.53211. No fork changes; the curated commit (4a860c8) is untouched.

### Build commands

```
make -C projects/AutoDock-GPU/src DEVICE=HIP NUMWI=64  HIP_ARCH=gfx1100
make -C projects/AutoDock-GPU/src DEVICE=HIP NUMWI=128 HIP_ARCH=gfx1100
```

Both built cleanly (warnings only: nodiscard on hipDeviceReset in RTERROR macro,
and INFINITY undefined-behavior under -ffast-math -- both pre-existing in
upstream, no errors). HIP_ARCH is not hardcoded in Makefile.Hip (defaults gfx90a
when unset); gfx1100 requires no source or Makefile edit -- exactly as designed.

### GFX1100 code-object evidence

```
$ roc-obj-ls bin/autodock_gpu_64wi
1  host-x86_64-unknown-linux-gnu-                     ...autodock_gpu_64wi#offset=331776&size=0
1  hipv4-amdgcn-amd-amdhsa--gfx1100                   ...autodock_gpu_64wi#offset=331776&size=85048

$ roc-obj-ls bin/autodock_gpu_128wi
1  host-x86_64-unknown-linux-gnu-                     ...autodock_gpu_128wi#offset=331776&size=0
1  hipv4-amdgcn-amd-amdhsa--gfx1100                   ...autodock_gpu_128wi#offset=331776&size=85048
```

No gfx90a object present in either binary.

### Wave32 path analysis

gfx1100 is NOT __GFX9__, so at device-compile time:
- WARP_SIZE = 32 (the __GFX9__ branch is not taken)
- WARPMINIMUM_TOP expands to nothing (guard: WARP_SIZE > 32 is false)
- WARPSUM_TOP expands to nothing (same guard)
- The stride-32 shuffle-add and exchange steps are compiled out

At runtime, cudaGetDeviceProperties returns warpSize=32 for gfx1100, so:
- cData.warpmask = 31, cData.warpbits = 5

The kernel4 cross-warp reduction uses blockDim.x / WARP_SIZE (=32), yielding
2 warps/block for 64wi and 4 warps/block for 128wi -- correct on wave32.

WARP_FULL_MASK = 0xffffffffffffffffULL (64-bit, required by the HIP 7.2 API
regardless of actual wave width; upper bits address no lanes on wave32 so
all-ones is still correct as "all active lanes").

### Docking results (1stp streptavidin-biotin, --nrun 10, HIP_VISIBLE_DEVICES=0)

```
HIP_VISIBLE_DEVICES=0 bin/autodock_gpu_64wi  -ffile input/1stp/derived/1stp_protein.maps.fld \
  -lfile input/1stp/derived/1stp_ligand.pdbqt -nrun 10 -seed 42 -resnam <out>
HIP_VISIBLE_DEVICES=0 bin/autodock_gpu_64wi  -ffile input/1stp/derived/1stp_protein.maps.fld \
  -lfile input/1stp/derived/1stp_ligand.pdbqt -nrun 10 -seed 7  -resnam <out>
HIP_VISIBLE_DEVICES=0 bin/autodock_gpu_128wi -ffile input/1stp/derived/1stp_protein.maps.fld \
  -lfile input/1stp/derived/1stp_ligand.pdbqt -nrun 10 -seed 42 -resnam <out>
```

| build / seed     | best binding energy | best-pose reference RMSD | cluster       |
|------------------|---------------------|--------------------------|---------------|
| 64wi  / seed 42  | -8.28 kcal/mol      | 0.40 A                   | 1, 10/10 runs |
| 64wi  / seed 7   | -8.37 kcal/mol      | 0.38 A                   | 1, 10/10 runs |
| 128wi / seed 42  | -8.35 kcal/mol      | 0.41 A                   | 1, 10/10 runs |

Reference (wave64, gfx90a lead):

| build / seed     | best binding energy | best-pose reference RMSD |
|------------------|---------------------|--------------------------|
| 64wi  / seed 42  | -8.28 kcal/mol      | 0.48 A                   |
| 64wi  / seed 7   | -8.37 kcal/mol      | 0.37 A                   |
| 128wi / seed 42  | -8.33 kcal/mol      | 0.39 A                   |

All three cases: all 10 runs converge to a single cluster; best energies within
0.02 kcal/mol of the wave64 lead; best-pose reference RMSD sub-0.5 A in all
cases. No NaN in energies, no HIP fault, clean exit. The 128wi case exercises
the cross-warp final reduction in kernel4 on wave32 (4 warps/block, WARP_SIZE=32
path) and is also correct.

PASS: wave32 warp reductions (WARPMINIMUM2, REDUCEINTEGERSUM, REDUCEFLOATSUM)
are numerically correct on gfx1100. Fork is untouched; head_sha = 4a860c8.

## Validation 2026-06-05 (windows-gfx1101, ROCm 7.14)

Platform: Radeon PRO V710 (gfx1101, RDNA3, wave32), ROCm 7.14 (TheRock PyTorch
venv), Windows 11 Pro for Workstations 10.0.26200. HIP_VISIBLE_DEVICES=0.

### Windows-specific source fix required

The HIP device compiler on Windows defines both `_WIN32` and
`__HIP_DEVICE_COMPILE__` simultaneously, but the Windows SDK headers pulled in
by `miscellaneous.h` (`processthreadsapi.h -> winnt.h`) require that the x86
architecture macro (`_AMD64_`) be set up by the MSVC host toolchain. The HIP
device pass targets an AMD GPU, so this macro is absent in that pass, causing
`winnt.h` to emit `#error "No Target Architecture"`. Fix: guard the
`processthreadsapi.h` include with `!__HIP_DEVICE_COMPILE__` and add a stub
`processid()` for the device-compile path (LocalRNG references it in
header-inlined code that the device TU sees). Committed as b623ccc on top of
the port; this is a Windows HIP device-pass issue, fully inert on Linux (the
`_WIN32` branch never compiles on Linux hosts).

### Build commands (Windows, no Make -- direct compiler invocation)

```
SITE=/path/to/venv/Lib/site-packages
HIPCC=$SITE/_rocm_sdk_core/bin/hipcc.exe
CXX=$SITE/_rocm_sdk_devel/lib/llvm/bin/clang++.exe
ROCM_INC=$SITE/_rocm_sdk_devel/include
ROCM_LIB=$SITE/_rocm_sdk_devel/lib
SRC=projects/AutoDock-GPU/src
IFLAGS="-I$SRC/common -I$SRC/host/inc -I$SRC/cuda -I$ROCM_INC"
HOST_DEFS="-DUSE_HIP -D__HIP_PLATFORM_AMD__ -D_AMD64_ -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS -DVERSION=b623ccc"

# Kernel TU (64wi)
$HIPCC -DN64WI --offload-arch=gfx1101 -x hip -std=c++17 -O3 -ffast-math \
  -D_AMD64_ -DUSE_HIP -DVERSION=b623ccc $IFLAGS \
  -c $SRC/cuda/kernels.cu -o $SRC/kernels_64wi.obj

# Host .cpp files
for cpp in $SRC/host/src/*.cpp; do
    $CXX -std=c++17 $IFLAGS $HOST_DEFS -O3 -DN64WI -DGPU_DEVICE \
        -c $cpp -o $SRC/$(basename $cpp .cpp).obj
done

# Link
$CXX $SRC/*.obj $SRC/kernels_64wi.obj -L$ROCM_LIB -lamdhip64 \
  -o $SRC/bin/autodock_gpu_64wi.exe

# Repeat with -DN128WI for 128wi variant
```

Key Windows differences vs Linux:
- No Makefile.Hip; drive compiler directly (hipcc.exe / clang++.exe)
- No -fPIE (clang rejects for Windows/MSVC target)
- Add -D_AMD64_ so Windows SDK headers resolve architecture macros
- Add -DNOMINMAX to prevent min/max macro pollution from windows.h
- amdhip64 loaded from C:\Windows\System32 (driver ABI, works correctly)

### Docking results (1stp streptavidin-biotin, --nrun 10, HIP_VISIBLE_DEVICES=0)

```
HIP_VISIBLE_DEVICES=0 bin/autodock_gpu_64wi.exe \
  -ffile input/1stp/derived/1stp_protein.maps.fld \
  -lfile input/1stp/derived/1stp_ligand.pdbqt -nrun 10 -seed 42 -resnam <out>
HIP_VISIBLE_DEVICES=0 bin/autodock_gpu_64wi.exe ... -seed 7 ...
HIP_VISIBLE_DEVICES=0 bin/autodock_gpu_128wi.exe ... -seed 42 ...
```

| build / seed     | best binding energy | best-pose reference RMSD | cluster       |
|------------------|---------------------|--------------------------|---------------|
| 64wi  / seed 42  | -8.28 kcal/mol      | 0.59 A                   | 1, 10/10 runs |
| 64wi  / seed 7   | -8.37 kcal/mol      | 0.39 A                   | 1, 10/10 runs |
| 128wi / seed 42  | -8.28 kcal/mol      | 0.59 A                   | 1, 10/10 runs |

Reference (wave64, gfx90a lead):

| build / seed     | best binding energy | best-pose reference RMSD |
|------------------|---------------------|--------------------------|
| 64wi  / seed 42  | -8.28 kcal/mol      | 0.48 A                   |
| 64wi  / seed 7   | -8.37 kcal/mol      | 0.37 A                   |
| 128wi / seed 42  | -8.33 kcal/mol      | 0.39 A                   |

All three cases: all 10 runs converge to a single cluster; best energies match
the gfx90a lead within 0.05 kcal/mol; best-pose reference RMSD sub-0.6 A in
all cases. No NaN, no HIP fault, clean exit. The 128wi case exercises the
cross-warp final reduction in kernel4 on wave32 (4 warps/block, WARP_SIZE=32)
and is also correct.

PASS: gfx1101 (RDNA3, wave32) docking results are numerically correct.
head_sha = b623ccc.

## Validation 2026-06-06 (windows-gfx1201, ROCm 7.14)

Platform: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), ROCm 7.14 (TheRock PyTorch
venv), Windows 11 Pro for Workstations 10.0.26200. HIP_VISIBLE_DEVICES=0. warpSize=32.

### Build commands (Windows, direct compiler invocation, gfx1201)

Same toolchain as gfx1101. No source changes; only --offload-arch changes from gfx1101 to gfx1201.

```
SITE=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages
HIPCC=$SITE/_rocm_sdk_devel/bin/hipcc.exe
CXX=$SITE/_rocm_sdk_devel/lib/llvm/bin/clang++.exe
ROCM_INC=$SITE/_rocm_sdk_devel/include
ROCM_LIB=$SITE/_rocm_sdk_devel/lib
SRC=B:/develop/moat/projects/AutoDock-GPU/src
IFLAGS="-I$SRC/common -I$SRC/host/inc -I$SRC/cuda -I$ROCM_INC"

# Kernel TU (64wi)
$HIPCC -DN64WI --offload-arch=gfx1201 -x hip -std=c++17 -O3 -ffast-math \
  -D_AMD64_ -DUSE_HIP $IFLAGS \
  -c $SRC/cuda/kernels.cu -o $SRC/kernels_64wi_gfx1201.obj

# Kernel TU (128wi)
$HIPCC -DN128WI --offload-arch=gfx1201 -x hip -std=c++17 -O3 -ffast-math \
  -D_AMD64_ -DUSE_HIP $IFLAGS \
  -c $SRC/cuda/kernels.cu -o $SRC/kernels_128wi_gfx1201.obj

# Link (64wi) -- host .obj reused from gfx1101 build (host is x86_64, arch-independent)
$CXX $SRC/{calcenergy,getparameters,main,miscellaneous,performdocking,processgrid,processligand,processresult,setup}.obj \
  $SRC/kernels_64wi_gfx1201.obj -L$ROCM_LIB -lamdhip64 \
  -o $SRC/bin/autodock_gpu_64wi_gfx1201.exe

# Link (128wi)
$CXX $SRC/{calcenergy,getparameters,main,miscellaneous,performdocking,processgrid,processligand,processresult,setup}.obj \
  $SRC/kernels_128wi_gfx1201.obj -L$ROCM_LIB -lamdhip64 \
  -o $SRC/bin/autodock_gpu_128wi_gfx1201.exe
```

Both built cleanly (warnings only: nodiscard on hipDeviceReset, strncpy deprecation, stricmp deprecation -- all pre-existing upstream). gfx1201 is wave32 (same as gfx1101); no source changes from gfx1101 build.

### Docking results (1stp streptavidin-biotin, --nrun 10, HIP_VISIBLE_DEVICES=0)

```
HIP_VISIBLE_DEVICES=0 bin/autodock_gpu_64wi_gfx1201.exe \
  -ffile input/1stp/derived/1stp_protein.maps.fld \
  -lfile input/1stp/derived/1stp_ligand.pdbqt -nrun 10 -seed 42 -resnam <out>
HIP_VISIBLE_DEVICES=0 bin/autodock_gpu_64wi_gfx1201.exe ... -seed 7 ...
HIP_VISIBLE_DEVICES=0 bin/autodock_gpu_128wi_gfx1201.exe ... -seed 42 ...
```

| build / seed     | best binding energy | best-pose reference RMSD | cluster       |
|------------------|---------------------|--------------------------|---------------|
| 64wi  / seed 42  | -8.28 kcal/mol      | 0.59 A                   | 1, 10/10 runs |
| 64wi  / seed 7   | -8.37 kcal/mol      | 0.39 A                   | 1, 10/10 runs |
| 128wi / seed 42  | -8.28 kcal/mol      | 0.59 A                   | 1, 10/10 runs |

Reference (wave64, gfx90a lead):

| build / seed     | best binding energy | best-pose reference RMSD |
|------------------|---------------------|--------------------------|
| 64wi  / seed 42  | -8.28 kcal/mol      | 0.48 A                   |
| 64wi  / seed 7   | -8.37 kcal/mol      | 0.37 A                   |
| 128wi / seed 42  | -8.33 kcal/mol      | 0.39 A                   |

All three cases: all 10 runs converge to a single cluster; best energies match
the gfx90a lead within 0.05 kcal/mol; best-pose reference RMSD sub-0.6 A in
all cases. No NaN, no HIP fault, clean exit. The 128wi case exercises the
cross-warp final reduction in kernel4 on wave32 (4 warps/block, WARP_SIZE=32)
and is also correct. GPU: AMD Radeon RX 9070 XT (gfx1201, RDNA4).

PASS: gfx1201 (RDNA4, wave32) docking results are numerically correct.
head_sha = b623ccc.

## Revalidation 2026-06-05 (linux-gfx90a, binary-equivalence carry-forward)

Platform: 4x AMD Instinct MI250X (gfx90a), ROCm 7.2.1, HIP 7.2.

### Delta analysis: 4a860c8 -> b623ccc

Single commit (b623ccc): "[ROCm] Guard processthreadsapi.h from HIP device compilation on Windows"

Change: `host/inc/miscellaneous.h` adds `!__HIP_DEVICE_COMPILE__` guard to the `_WIN32` branch that includes `processthreadsapi.h`, plus a stub `processid()` for the device-compile path on Windows.

Impact on Linux/gfx90a: NONE. On Linux, `_WIN32` is not defined, so the modified `#if defined(_WIN32) && !defined(__HIP_DEVICE_COMPILE__)` condition remains false (same as before), and the `#elif !defined(_WIN32)` Linux branch is taken (unchanged). The new `#else` stub is never compiled on Linux. Preprocessor output confirmed identical except for line numbers.

### Binary-equivalence verification

Built at both commits (4a860c8 and b623ccc) with `make DEVICE=HIP NUMWI=64 HIP_ARCH=gfx90a`:

- GPU code object (gfx90a, 79424 bytes at offset 331776): SHA256 `eae182d2f4bac7f82be07f65a60a3e4015128c830dfee33f9acba2c47aee1f33` (IDENTICAL)
- Host .text section: SHA256 `9382fc7797554108d5fd30704488f2d34327d51f076a168eebef28bff493b4dd` (IDENTICAL)
- Exported symbols: IDENTICAL (nm -D diff empty)

Only differences: embedded VERSION string and debug line-number metadata (expected, non-functional).

VERDICT: Binary-equivalent on linux-gfx90a. Validation carried forward to b623ccc without GPU re-run.

## PR-prep 2026-06-08 (lead) -- jargon scrub + docs + squash; carry-forward, no GPU re-run

Pre-PR cleanup. Fork moat-port squashed to ONE commit on the upstream base (clean
PR diff): 43041f8 [ROCm] Add native HIP backend for AMD GPUs, parent 28a6139
(an ancestor of the current develop tip 6b150b3; the PR's merge-base diff is the
port only). 10 files, +379/-24. base develop.

NOTE on entry state: the local clone was STALE (4a860c8); the fork tip and
status head_sha agreed at b623ccc (the validated commit). Synced local to b623ccc
before prep. upstream.json had fork_url/base_sha null -- populated them.

Edits (both behavior-preserving, no compiled source touched):
- Makefile.Hip: reworded the HIP_ARCH default comment to drop the in-house "lead
  arch" wording (verified comment-only: the file's non-comment content is
  byte-identical across the change).
- README.md: documented DEVICE=HIP alongside DEVICE=CUDA -- added HIP to the
  <TYPE> table and an override note ("For AMD GPUs, use DEVICE=HIP with
  HIP_ARCH=<gfx-target> ..."). House style (markdown table + prose).
- Commit message scrub: dropped "the colmap model (Strategy A)" (cross-project
  ref + in-house label) and unified the AI disclosure; title dropped "(gfx90a)".

Carry-forward (NO GPU re-run): the classifier cannot tokenize Makefiles, so the
delta classified `mixed`. But the change is provably comment-only (Makefile) +
doc-only (README) touching no compiled source, so all FOUR platforms
(linux-gfx90a, linux-gfx1100, windows-gfx1101, windows-gfx1201) were carried
forward as source-class with that justification; squash-carry-forward then
advanced them to 43041f8. gfx1151 kept blocked (retired). pr-ready=True.

DID NOT add a Makefile classifier to changeclass.py (GNU make recipe-line `#`
and escaping rules make a safe general tokenizer non-trivial; a wrong one risks a
false carry-forward). The per-change manual comment-only verification is the safe
path here.

NEXT: upstream-PR gate (lead-only, jeff approval). No existing jeffdaily PR on
ccsb-scripps/AutoDock-GPU. PR base = develop; scope out retired gfx1151.

## PR-prep addendum 2026-06-08 -- TENSOR=ON scope disclosed

jeff flagged the `$(error TENSOR=ON ...)` guard in Makefile.Hip. Verified: TENSOR=ON
(USE_NVTENSOR) is an OPTIONAL, off-by-default NVIDIA tensor-core (WMMA) acceleration
of ONLY the energy/gradient sum-reduction in cuda/calcMergeEneGra.cu, gated
MIN_COMPUTE 8.0 (Ampere+). The code is #ifdef USE_NVTENSOR / [tensor path] / #else
[standard reduction] / #endif, so the default build (which the HIP port builds) uses
the standard reduction with identical docking results -- it is a perf path, not core
functionality. The HIP backend fail-fasts on TENSOR=ON (clean, not a silent
mis-build). So the port is functionally complete for the default docking pipeline;
the one optional tensor-core optimization is scoped out.

Actions: (1) registered deferred work autodock-gpu-tensor-core-reduction
(feature-port; rocWMMA/MFMA mapping); (2) added an explicit "Out of scope" paragraph
to the commit message and a "## Scope" section to the PR body so the upstream PR does
not overclaim parity. Commit message amended (tree-identical) 43041f8 -> 221e2d4f;
all 4 platforms carried forward (no GPU re-run). Fork moat-port @ 221e2d4f.

## TENSOR=ON tensor-core reduction on AMD via rocWMMA 2026-06-08 (gfx90a)

Implemented the optional TENSOR=ON sum-reduction (previously the scoped-out
NVIDIA-only feature) on AMD using rocWMMA, as a NEW commit on top of 221e2d4f
(221e2d4f stays the validated port; this is additive). 3 files: cuda/kernels.cu,
host/src/performdocking.cpp, Makefile.Hip. CUDA (nvcuda/tcec), OpenCL, and CPU
paths are byte-for-byte unchanged (git diff shows ZERO removed lines in
kernels.cu and performdocking.cpp; only the AMD `#if defined(__HIP_PLATFORM_AMD__)`
branches and the `#else` wrappers were added around the original code).

### What the reduction does (so the wave64 rework is checkable)

data_to_be_reduced is 4*NUM_OF_THREADS_PER_BLOCK floats; thread t writes 4 values
(torque_rot.xyz, energy) at data[4t..4t+3]. The reduction sums each of the 4
interleaved components across all threads, leaving the 4 sums in data[0..3]. As a
16x16 col-major tile (ld=16), offset o -> (row=o%16, col=o/16), so component c
lands in rows {c, c+4, c+8, c+12}. Stage 1 (V = sum of A*ones over tiles) row-sums
each tile -> V[r] = partial sum of component r%4 over a quarter of the threads.
Stage 2 (C = Q*W, Q = a 4x4-tiled 4x4 identity, i.e. Q[i][k]=1 iff i%4==k%4)
sums the four row-blocks -> C[c][0] = full sum of component c. data[0..3] = first
column of C.

### rocWMMA mapping (1:1 with nvcuda::wmma)

rocwmma::fragment / fill_fragment / load_matrix_sync / mma_sync /
store_matrix_sync / matrix_a,matrix_b,accumulator,col_major,mem_col_major map
directly onto the NVIDIA names (aliased `namespace adwmma = rocwmma`). Data type
is plain `float`: CDNA gfx9 has native fp32 MFMA, so the reduction is EXACT fp32
and the tf32 split-and-correct error compensation (mtk::wmma::tcec) is dropped
entirely. The 16x16x16 fragment shape composes fine -- rocWMMA builds it from the
native K=4 MFMA internally (verified: the device code object carries 32
v_mfma_f32_16x16x4f32 instructions; 4 per mma_sync = K=16/K=4).

### PoC (agent_space/adgpu_poc/) -- ran on gfx90a before touching the source

- poc_k16.cpp: confirmed fragment<...,16,16,16,float,...> compiles AND runs
  correctly on gfx90a (1 tile / NUMWI=64). The K=16 shape does compose; no need to
  hand-loop K=4 with rocWMMA.
- poc_k16_2tile.cpp: confirmed the 2-tile accumulation loop (NUMWI=128) is correct.
Both reproduced the exact reference column/row sums.

### CRITICAL wave64 adaptations (get these wrong -> wrong result or hang)

1. Wave width: nvcuda WMMA is a 32-lane warp collective; CDNA MFMA is a 64-lane
   wavefront collective. The `if (threadIdx.x <= 31)` guard becomes
   `if (threadIdx.x < 64)`.
2. Identity-tile fill: the NVIDIA per-lane index math (k = threadIdx.x<<3,
   kk = 16-(threadIdx.x>>1), 8 entries/lane) assumes 32 lanes filling 256 entries.
   On wave64 that math does not cover the tile. Reworked to derive each entry from
   its column-major offset (wave-size independent):
   `for (o = threadIdx.x; o < 256; o += 64) Q[o] = ((o%16)&3)==((o/16)&3) ? 1 : 0`.
3. LDS round-trip barriers (the subtle one; caused a NUMWI=128 SIGSEGV at first):
   rocWMMA load/store_matrix_sync are per-lane LDS ops with NO implicit barrier
   (unlike nvcuda's `.sync` collectives -- rocWMMA's synchronize_workgroup is a
   separate explicit call). The V->W round-trip through LDS and the identity-fill
   both need a barrier. But `__syncthreads()` is WORKGROUP-scoped and DEADLOCKS
   when the block has >1 wavefront (NUMWI=128: only wavefront 0 enters the `if`,
   the others never reach the barrier -> EXIT 139). Fix: a wavefront-scoped
   barrier `wavefront_lds_barrier()` = LDS release-fence + __builtin_amdgcn_wave_barrier()
   + acquire-fence, at all three LDS hazards (V-store->W-load, W-load->Q-fill
   clobber, Q-fill->Q-load). NUMWI=64 happened to work with __syncthreads (block ==
   1 wavefront, non-divergent) which is why the bug only showed at 128.

### Makefile.Hip

Removed `$(error TENSOR=ON ...)`. TENSOR=ON now sets `NVTENSOR = -DUSE_NVTENSOR`
(reaches both the hipcc kernel TU and the g++ host TUs, activating the
calcMergeEneGra.cu call sites and the kernels.cu code) and enforces NUMWI in
{64,128,256} (the original NUMWI>32 requirement) with a clean `$(error)`. rocWMMA
is header-only, already on `-I$(ROCM_PATH)/include`; no extra link. HIP_ARCH stays
configurable. Gotcha: directive lines inside the `ifeq` must NOT be tab-indented
(a leading tab makes `$(error)` "recipe commences before first target"); use no
indentation.

### Build (gfx90a, ROCm 7.2.1)

    make DEVICE=HIP NUMWI=64  TENSOR=ON HIP_ARCH=gfx90a
    make DEVICE=HIP NUMWI=128 TENSOR=ON HIP_ARCH=gfx90a
    make DEVICE=HIP NUMWI=64            HIP_ARCH=gfx90a   # baseline
NB: `make clean` with TARGET=autodock_gpu_64wi wipes bin/autodock_gpu_64wi* (so a
saved *_tensoron next to it is deleted on the next clean) -- copy benchmark
binaries OUTSIDE bin/. All four (64/128 x on/off) built clean (pre-existing
nodiscard warnings only).

### Correctness -- TENSOR=ON vs OFF parity (1stp, -nrun 10, GPU 3 = MI250X)

| config            | best energy | best-pose ref RMSD | cluster       |
|-------------------|-------------|--------------------|---------------|
| 64wi  OFF seed42  | -8.28       | 0.48 A             | 1, 10/10 runs |
| 64wi  ON  seed42  | -8.27       | 0.31 A             | 1, 10/10 runs |
| 64wi  OFF seed7   | -8.37       | 0.37 A             | 1, 10/10 runs |
| 64wi  ON  seed7   | -8.37       | 0.40 A             | 1, 10/10 runs |
| 128wi OFF seed42  | -8.33       | 0.39 A             | 1, 10/10 runs |
| 128wi ON  seed42  | -8.25       | 0.59 A             | 1, 10/10 runs |

TENSOR=ON matches TENSOR=OFF to docking tolerance in every case: native biotin
pose recovered, best energy ~-8.28 +/- 0.08 kcal/mol, all 10 runs in one cluster,
ref RMSD sub-0.6 A. Per-run scatter differs slightly (the tensor reduction sums in
a different float order, so the GA trajectory diverges) -- expected and benign; the
best pose converges identically. No NaN, no HIP fault, clean exit.

### Benchmark -- honest numbers (1stp, nrun 100, ngen 80000, "Job took" docking-only, best-of-5/median-of-8, GPU 3)

| config       | docking time | TENSOR=ON vs OFF |
|--------------|--------------|------------------|
| 64wi  OFF    | 0.634 s      | --               |
| 64wi  ON     | 0.701 s      | 1.11x SLOWER     |
| 128wi OFF    | 0.809 s (median, +/-0.002) | --  |
| 128wi ON     | 0.514 s (median, +/-0.002) | 1.57x FASTER |

The reduction is small, so the result is configuration-dependent and honest: at
NUMWI=64 the MFMA setup is pure overhead vs an already-cheap single-wavefront
shuffle (tensor LOSES ~11%); at NUMWI=128 the standard reduction pays cross-
wavefront shuffle + atomicAdd costs that the single-wavefront MFMA collective
avoids (tensor WINS 1.57x, rock-stable across 8 reps). So tensor cores help here
only when the block spans multiple wavefronts.

### ck_tile feasibility spike (PROVEN in-kernel; rocWMMA wins on fit)

ck_tile CAN implement this as a __device__ sub-reduction called mid-kernel -- it is
NOT host-launch-only. Evidence (agent_space/adgpu_poc/poc_cktile_init.cpp):
ck_tile::WarpGemmMfmaF32F32F32M16N16K4 invoked via its CK_TILE_DEVICE operator()
on register-resident static_distributed_tensor fragments inside a kernel
(one wavefront) compiled and ran correctly (C=4.0 for A=B=ones, K=4) on gfx90a.
The warp/ MFMA primitives (ops/gemm/warp/warp_gemm.hpp) are device building blocks,
exactly the in-kernel op we need.

BUT the fit is worse than rocWMMA, with concrete reasons found while spiking:
1. In ROCm 7.2.1 the convenient f32 M16N16K16 alias is NOT exposed (only ...K4);
   I'd manually loop K=4 four times. rocWMMA composes K=16 for me.
2. The fragment fill/extract API drifts in this version: sweep_tile fails to
   instantiate over a WarpGemm fragment (reverse_slice_sequence_impl<sequence<>...>
   / get_y_unpacks_from_x_unpacks), and static_distributed_tensor::initialize /
   thread_buffer::initialize do not exist. I had to poke the raw thread buffer with
   static_for, which means manually reimplementing the lane->(row,col) data
   distribution to load the LDS tile and to build the identity in the warp-gemm's
   own register layout (rocWMMA's load_matrix_sync/store_matrix_sync handle that).
3. ck_tile WarpGemm fragments are designed to be fed by tile_window + load_tile
   pipelines; using them for a tiny ad-hoc 16x16 mid-kernel reduction fights the
   framework's grain.

Decision: kept rocWMMA. Its fragment API is a near-mechanical 1:1 swap for the
existing nvcuda::wmma code (matches house style, reads like the original), composes
K=16, and manages the LDS<->fragment distribution. ck_tile would be materially more
boilerplate for no correctness or (for this small reduction) clear perf gain. Did
NOT force a ck_tile implementation, per the spike's "rocWMMA wins on fit" outcome.

### Scope / state

This is additive work on top of the validated port; status.json platform states are
the lead's to manage (TENSOR=OFF default build is unchanged and stays validated).
No upstream PR action taken here.

## TENSOR=ON committed into state machine; followers to verify 2026-06-09

Per jeff: commit the rocWMMA TENSOR=ON support and let the followers verify as usual.
advance_head 221e2d4f -> 3d466a4 classified the delta `mixed` (real device-code
branches in kernels.cu/performdocking.cpp + Makefile.Hip) -> all platforms revalidate.
gfx90a marked completed (porter GPU-validated TENSOR=ON: docking matches the non-tensor
path; 1.57x faster at NUMWI=128). linux-gfx1100 + windows-gfx1101/gfx1201 left in
revalidate to verify on their own hosts -- the DEFAULT (TENSOR=OFF) build is unchanged
(tensor code is #ifdef USE_NVTENSOR-gated) so it carries forward binary-equiv; TENSOR=ON
itself is CDNA-only for now (RDNA needs bf16 error-correction -- see deferred
autodock-gpu-tensor-core-reduction, rescoped). gfx1151 blocked (retired).

NEXT: once followers verify at 3d466a4, squash 221e2d4f+3d466a4 -> one commit and re-prep
the PR body (document CDNA TENSOR=ON via rocWMMA + the benchmark; RDNA as a follow-up;
drop the earlier "out of scope" note), then open the upstream PR (gate).

## Revalidation 2026-06-09 (linux-gfx1100, binary-equivalence carry-forward)

Platform: 4x AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), ROCm 7.2.1, HIP 7.2.

### Delta analysis: 221e2d4 -> 3d466a4

Single commit (3d466a4): "[ROCm] Add rocWMMA tensor-core sum-reduction for TENSOR=ON on AMD"

Change: cuda/kernels.cu, host/src/performdocking.cpp, Makefile.Hip all add #ifdef USE_NVTENSOR
branches for the rocWMMA MFMA reduction. The DEFAULT build (TENSOR=OFF, no -DUSE_NVTENSOR) leaves
those branches uncompiled.

Impact on gfx1100 TENSOR=OFF build: NONE. Built both SHAs with identical flags:

```
make -C .../src-old DEVICE=HIP NUMWI=64 HIP_ARCH=gfx1100   # at 221e2d4
make -C .../src-new DEVICE=HIP NUMWI=64 HIP_ARCH=gfx1100   # at 3d466a4
python3 utils/codeobj_diff.py src-old/bin/autodock_gpu_64wi src-new/bin/autodock_gpu_64wi
```

```
verdict=identical
  autodock_gpu_64wi vs autodock_gpu_64wi: identical (exported symbols + device ISA identical (14 exports))
```

TENSOR=ON itself is CDNA-only for now (RDNA needs bf16 error-correction per deferred task
autodock-gpu-tensor-core-reduction); gfx1100 is not expected to use TENSOR=ON.

VERDICT: binary-equiv carry-forward. linux-gfx1100 -> completed at 3d466a4. No GPU re-run needed.

## TENSOR=ON RDNA (gfx11+, wave32) via bf16 error correction 2026-06-09 (gfx1100)

Implements the deferred task autodock-gpu-tensor-core-reduction. New commit on top of
3d466a4 (the validated port stays untouched; this is additive). Single source file:
cuda/kernels.cu. Makefile.Hip unchanged.

Committed and pushed: fork moat-port HEAD = f01306b (parent 3d466a4, the validated
port; NOT an amend). status.json platform states left untouched (lead's to manage);
no upstream PR action.

### Root cause this fixes

Commit 3d466a4 added one AMD TENSOR=ON branch coded for CDNA wave64 native fp32 MFMA.
On gfx11/gfx12 (RDNA, wave32) it COMPILES but is SILENTLY WRONG: rocWMMA has no
fp32->fp32 WMMA intrinsic on RDNA (WMMA accepts only f16/bf16 inputs), so the fp32
mma_sync falls through to an unsupported no-op (zero v_wmma_* instructions emitted),
the reduction returns 0, and docking produces ~+1e8 kcal/mol garbage instead of ~-8.3.

### Fix structure (CDNA and NVIDIA paths BYTE-IDENTICAL)

The AMD branch in reduce_via_tensor_units is now split by arch:
- `#if defined(__GFX9__)`: the original wave64 native-fp32 MFMA code, verbatim (only
  reindented under the new #if and the comment reworded). Proven byte-identical: built
  gfx90a TENSOR=ON NUMWI=64 before and after; `utils/codeobj_diff.py` -> verdict=identical
  (exported symbols + device ISA identical, 14 exports).
- `#else` (RDNA/gfx11+): NEW bf16 error-correction branch (below).
The NVIDIA `#else` (mtk::wmma::tcec) is untouched -- git diff shows the only removed lines
in kernels.cu are comment lines; no nvcuda/tcec/mtk:: CODE line is removed.

### RDNA branch math (mirrors the NVIDIA tcec scheme)

bfloat16_t inputs, float32_t accumulator, 16x16x16. fp32 operand v is split
hi=(bfloat16)v, lo=(bfloat16)(v-(float)hi); cross products accumulate in the fp32
accumulator. The all-ones (stage 1 frag_P) and the 4x4-tiled identity (stage 2 frag_Q)
are exact in bf16 (lo==0), so only the data tiles (stage 1 A) and the partial-sum tile
W=V (stage 2) are split:
- Stage 1: V += A_hi*P + A_lo*P  (per tile)
- Stage 2: C  = Q*W_hi + Q*W_lo
This drops the negligible lo*lo term, exactly as the CUDA tcec path does.

### wave32 correctness (the current code was wave64-coded)

- wavefront gate `threadIdx.x < 64` -> `threadIdx.x < WARP_SIZE` (=32 on RDNA).
- identity-fill stride `o += 64` -> `o += WARP_SIZE`.
- LDS barrier unchanged (wavefront-scoped __builtin_amdgcn_wave_barrier, already
  wave-size-correct; it synchronizes whatever lanes are in the single wavefront).
All keyed on WARP_SIZE so the same source is correct on wave32 and wave64.

### Makefile.Hip: RDNA re-enabled (no change needed)

The existing `ifeq ($(TENSOR),ON)` block already sets -DUSE_NVTENSOR for any AMD arch
with NUMWI in {64,128,256}; there was no CDNA-only fail-fast to remove. NUMWI>32
requirement kept.

### Build (gfx1100, ROCm 7.2.1, HIP_VISIBLE_DEVICES=0)

    make DEVICE=HIP NUMWI=64  TENSOR=ON HIP_ARCH=gfx1100
    make DEVICE=HIP NUMWI=128 TENSOR=ON HIP_ARCH=gfx1100
    make DEVICE=HIP NUMWI=64             HIP_ARCH=gfx1100   # TENSOR=OFF sanity

Both TENSOR=ON builds clean (pre-existing nodiscard warnings only).

### v_wmma_* confirmation (the reduction is no longer a no-op)

```
llvm-objdump -d <gfx1100 code object> | grep v_wmma
  64wi:  4 v_wmma_f32_16x16x16_bf16   (1 stage-1 tile x2 hi/lo + 2 stage-2 hi/lo)
  128wi: 6 v_wmma_f32_16x16x16_bf16   (2 stage-1 tiles x2 + 2 stage-2)
```

### Docking results (1stp streptavidin-biotin, --nrun 10, HIP_VISIBLE_DEVICES=0)

Best binding energy / best-pose reference RMSD; all cases 10/10 runs in ONE cluster:

| config            | TENSOR=ON       | TENSOR=OFF ref (same host) |
|-------------------|-----------------|----------------------------|
| 64wi  seed 42     | -8.26 / 0.62 A  | -8.28 / 0.40 A             |
| 64wi  seed 7      | -8.38 / 0.39 A  | -8.37 / 0.38 A             |
| 128wi seed 42     | -7.94 / 0.47 A  | -8.35 / 0.42 A             |
| 128wi seed 7      | -7.91 / 0.73 A  | --                         |
| 128wi seed 1/100/999 | -8.10 / -8.30 / -8.28 | --                  |

All TENSOR=ON cases recover the native biotin pose (sub-0.75 A reference RMSD), best
energy in the -8.3 kcal/mol class (NOT 1e8 garbage), all 10 runs in a single cluster.
64wi matches TENSOR=OFF within 0.02 kcal/mol. 128wi shows more run-to-run scatter
(across seeds 1/42/7/100/999 the best ranges -7.91..-8.30): the bf16 emulation sums in
a different float order, so the GA trajectory diverges -- the same benign effect the
gfx90a CDNA TENSOR=ON path already documents (gfx90a 128wi ON was -8.25 vs OFF -8.33).
No NaN, no HIP fault, clean exit in every run.

The gfx1100 TENSOR=OFF default build is unaffected (references above match the prior
gfx1100 validation), as the RDNA tensor code is USE_NVTENSOR-gated.

VERDICT: PASS on gfx1100 (RDNA3, wave32). NUMWI=64 and NUMWI=128 both correct. The
silent-zero TENSOR=ON bug on RDNA is fixed; fp32 is emulated with bf16 error correction.

### Reproducible-build confirmation at the committed sha (f01306b)

Rebuilt from clean source trees (git archive of the parent 3d466a4 = baseline,
working tree = the fix) and re-ran on gfx1100 (HIP_VISIBLE_DEVICES=0), wrapped in
utils/timeit.sh, to certify the commit that was actually pushed:

- v_wmma_f32_16x16x16_bf16 in the gfx1100 TENSOR=ON code object: 64wi=4, 128wi=6.
- CDNA gfx90a TENSOR=ON NUMWI=64, baseline(3d466a4) vs new: codeobj_diff
  verdict=identical (exported symbols + device ISA identical, 14 exports).
- gfx1100 TENSOR=OFF default build, baseline vs new: codeobj_diff verdict=identical
  (the entire tensor path is USE_NVTENSOR-gated, so the default is bit-unchanged).
- NVIDIA path: git diff shows the only removed lines in kernels.cu are comments; no
  nvcuda::/mtk::/tcec code line and no CDNA code line is removed.
- Docking (1stp, --nrun 10): 64on/42 -8.26 (ref 0.62 A), 64off/42 -8.28; 64on/7
  -8.38 (0.39 A), 64off/7 -8.37; 128on/42 -7.94 (0.47 A), 128off/42 -8.33; 128on/7
  -7.91 (0.73 A). All -8.3-class, all 10/10 runs in one cluster, no NaN/fault.

Fork moat-port HEAD after push: f01306b.

## Revalidation 2026-06-09 (windows-gfx1201, ROCm 7.14) -- f01306b -> c2f2f6f

Platform: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), ROCm 7.14 (TheRock PyTorch
venv), Windows 11 Pro for Workstations 10.0.26200. HIP_VISIBLE_DEVICES=0. warpSize=32.

### Delta from prior validated sha (221e2d4 -> f01306b)

3 files changed: cuda/kernels.cu (+164 lines), Makefile.Hip (+13), host/src/performdocking.cpp (+8).
The change adds: (1) the rocWMMA CDNA MFMA tensor reduction branch (3d466a4) and
(2) the RDNA bf16 error correction tensor reduction branch for gfx11+/wave32 (f01306b).
Both are #ifdef USE_NVTENSOR gated; the default TENSOR=OFF build is unaffected.

### Windows source fix required for TENSOR=ON (committed as c2f2f6f)

rocWMMA's `float8.hpp` defines `numeric_limits<hip_fp8_e4m3>::min()` and `::max()`
static methods. On Windows, the HIP device compiler includes Windows SDK headers as
part of startup (triggered by `-D_AMD64_`), which defines `min` and `max` as macros
via `windows.h`. These macros break the `float8.hpp` method definitions with
"too few arguments provided to function-like macro invocation" errors.

Fix: add `#ifdef min / #undef min / #endif` and `#ifdef max / #undef max / #endif`
guards immediately before the `<rocwmma/rocwmma.hpp>` include in kernels.cu. The
undefs are inside `#if defined(__HIP_PLATFORM_AMD__)` (already gated) and are no-ops
on Linux. Committed as c2f2f6f on top of f01306b; pushed to fork moat-port.

The TENSOR=OFF default build is unaffected (the entire block is inside #ifdef USE_NVTENSOR).

### Build commands (Windows, direct compiler invocation, gfx1201)

```
SITE=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages
HIPCC=$SITE/_rocm_sdk_devel/bin/hipcc.exe
CXX=$SITE/_rocm_sdk_devel/lib/llvm/bin/clang++.exe
ROCM_INC=$SITE/_rocm_sdk_devel/include
ROCM_LIB=$SITE/_rocm_sdk_devel/lib
SRC=B:/develop/moat/projects/AutoDock-GPU/src
IFLAGS="-I$SRC/common -I$SRC/host/inc -I$SRC/cuda -I$ROCM_INC"
HOST_DEFS="-DUSE_HIP -D__HIP_PLATFORM_AMD__ -D_AMD64_ -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS"
VERSION=$(git -C "$SRC" rev-parse --short HEAD)
VERSION_DEF="-DVERSION=\"$VERSION\""

# Kernel TU (TENSOR=OFF)
$HIPCC -DN64WI --offload-arch=gfx1201 -x hip -std=c++17 -O3 -ffast-math \
  -D_AMD64_ -DUSE_HIP $VERSION_DEF $IFLAGS \
  -c $SRC/cuda/kernels.cu -o $SRC/kernels_64wi_off_gfx1201.obj

# Kernel TU (TENSOR=ON) -- rocWMMA RDNA bf16 path
$HIPCC -DN64WI --offload-arch=gfx1201 -x hip -std=c++17 -O3 -ffast-math \
  -D_AMD64_ -DUSE_HIP -DUSE_NVTENSOR $VERSION_DEF $IFLAGS \
  -c $SRC/cuda/kernels.cu -o $SRC/kernels_64wi_on_gfx1201.obj
# (repeat with -DN128WI for 128wi variants)

# Host .cpp files
for cpp in $SRC/host/src/*.cpp; do
    base=$(basename $cpp .cpp)
    $CXX -std=c++17 $IFLAGS $HOST_DEFS -O3 -DN64WI -DGPU_DEVICE $VERSION_DEF \
        -c $cpp -o $SRC/${base}_gfx1201.obj
done

# Link
$CXX $SRC/{calcenergy,getparameters,main,miscellaneous,performdocking,processgrid,processligand,processresult,setup}_gfx1201.obj \
  $SRC/kernels_64wi_off_gfx1201.obj -L$ROCM_LIB -lamdhip64 \
  -o $SRC/bin/autodock_gpu_64wi_gfx1201.exe
```

All 4 variants (64wi/128wi x TENSOR=OFF/ON) built cleanly at c2f2f6f (pre-existing
warnings only: nodiscard on hipDeviceReset, strncpy/stricmp deprecation,
INFINITY undefined-behavior under -ffast-math, amdgpu occupancy target).

### Docking results (1stp streptavidin-biotin, --nrun 10, HIP_VISIBLE_DEVICES=0)

TENSOR=OFF (default build):

| build / seed     | best binding energy | best-pose reference RMSD | cluster       |
|------------------|---------------------|--------------------------|---------------|
| 64wi  / seed 42  | -8.28 kcal/mol      | 0.59 A                   | 1, 10/10 runs |
| 64wi  / seed 7   | -8.37 kcal/mol      | 0.39 A                   | 1, 10/10 runs |
| 128wi / seed 42  | -8.32 kcal/mol      | 0.43 A                   | 1, 10/10 runs |

TENSOR=ON (new RDNA bf16 error-correction path, rocWMMA WMMA v_wmma_f32_16x16x16_bf16):

| build / seed     | best binding energy | best-pose reference RMSD | cluster       |
|------------------|---------------------|--------------------------|---------------|
| 64wi  / seed 42  | -8.29 kcal/mol      | 0.38 A                   | 1, 10/10 runs |
| 64wi  / seed 7   | -8.37 kcal/mol      | 0.38 A                   | 1, 10/10 runs |
| 128wi / seed 42  | -8.29 kcal/mol      | 0.39 A                   | 1, 10/10 runs |

Reference (wave64, gfx90a lead):

| build / seed     | best binding energy | best-pose reference RMSD |
|------------------|---------------------|--------------------------|
| 64wi  / seed 42  | -8.28 kcal/mol      | 0.48 A                   |
| 64wi  / seed 7   | -8.37 kcal/mol      | 0.37 A                   |
| 128wi / seed 42  | -8.33 kcal/mol      | 0.39 A                   |

All 6 cases: all 10 runs converge to a single cluster; best energies match the
gfx90a lead within 0.05 kcal/mol; best-pose reference RMSD sub-0.6 A in all cases.
No NaN, no HIP fault, clean exit. TENSOR=ON and TENSOR=OFF results agree within
0.01 kcal/mol (64wi) and 0.03 kcal/mol (128wi) -- the new RDNA bf16 path is
numerically correct; it does not produce the ~+1e8 garbage that the broken CDNA-only
path produced on RDNA before this fix.

PASS: gfx1201 (RDNA4, wave32) TENSOR=OFF and TENSOR=ON both correct at c2f2f6f.
Fork moat-port HEAD after push: c2f2f6f.

## Revalidation 2026-06-09 (linux-gfx90a, binary-equivalence carry-forward)

Platform: 4x AMD Instinct MI250X (gfx90a), ROCm 7.2.1, HIP 7.2.

### Delta analysis: f01306b -> c2f2f6f

Single commit (c2f2f6f): "[ROCm] Undefine windows.h min/max macros before rocWMMA include on Windows"

Change: `cuda/kernels.cu` adds `#ifdef min / #undef min / #endif` and `#ifdef max / #undef max / #endif`
guards immediately before `#include <rocwmma/rocwmma.hpp>`, inside the existing
`#if defined(__HIP_PLATFORM_AMD__)` block.

Impact on Linux/gfx90a: NONE. On Linux, `min` and `max` are NOT defined as macros (they
are std:: inline functions); the `#ifdef min` and `#ifdef max` conditions are false, so
the undef lines are preprocessed away entirely. Compilation output is bit-identical.

### Binary-equivalence verification

Built at both commits with `make DEVICE=HIP NUMWI=64 HIP_ARCH=gfx90a` from clean git checkouts
in agent_space/adgpu-reval-old (f01306b) and agent_space/adgpu-reval-new (c2f2f6f):

```
python3 utils/codeobj_diff.py \
  agent_space/adgpu-reval-old/bin/autodock_gpu_64wi \
  agent_space/adgpu-reval-new/bin/autodock_gpu_64wi
```

```
verdict=identical
  autodock_gpu_64wi vs autodock_gpu_64wi: identical (exported symbols + device ISA identical (14 exports))
```

VERDICT: Binary-equivalent on linux-gfx90a. Validation carried forward to c2f2f6f without GPU re-run.

## Revalidation 2026-06-09 (linux-gfx1100, binary-equivalence carry-forward)

Platform: 4x AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), ROCm 7.2.1, HIP 7.2.

### Delta analysis: f01306b -> c2f2f6f

Single commit (c2f2f6f): "[ROCm] Undefine windows.h min/max macros before rocWMMA include on Windows"

Change: `cuda/kernels.cu` adds `#ifdef min / #undef min / #endif` and `#ifdef max / #undef max / #endif`
guards immediately before `#include <rocwmma/rocwmma.hpp>`, inside the existing
`#if defined(__HIP_PLATFORM_AMD__)` block.

Impact on Linux/gfx1100: NONE. On Linux, `min` and `max` are NOT defined as macros; the `#ifdef min`
and `#ifdef max` conditions are false and those lines are preprocessed away entirely.

### Binary-equivalence verification (gfx1100)

Built at both commits from clean git worktrees in agent_space/adgpu-cf-old (f01306b) and
agent_space/adgpu-cf-new (c2f2f6f), for both TENSOR=OFF and TENSOR=ON, HIP_ARCH=gfx1100:

```
make -C agent_space/adgpu-cf-old DEVICE=HIP NUMWI=64 HIP_ARCH=gfx1100             # TENSOR=OFF
make -C agent_space/adgpu-cf-new DEVICE=HIP NUMWI=64 HIP_ARCH=gfx1100             # TENSOR=OFF
make -C agent_space/adgpu-cf-old DEVICE=HIP NUMWI=64 TENSOR=ON HIP_ARCH=gfx1100   # TENSOR=ON
make -C agent_space/adgpu-cf-new DEVICE=HIP NUMWI=64 TENSOR=ON HIP_ARCH=gfx1100   # TENSOR=ON
python3 utils/codeobj_diff.py agent_space/adgpu-cf-old/bin/autodock_gpu_64wi \
                               agent_space/adgpu-cf-new/bin/autodock_gpu_64wi
```

Both TENSOR=OFF and TENSOR=ON:
```
verdict=identical
  autodock_gpu_64wi vs autodock_gpu_64wi: identical (exported symbols + device ISA identical (14 exports))
```

VERDICT: Binary-equivalent on linux-gfx1100 for both default (TENSOR=OFF) and TENSOR=ON builds.
Validation carried forward to c2f2f6f without GPU re-run. Full prior gfx1100 scope (TENSOR=OFF
docking + TENSOR=ON bf16 rocWMMA RDNA path validated at f01306b) stands unchanged.

## Re-prep: dropped TENSOR scope-out, re-squashed 2026-06-10

Per jeff, re-prepped the PR with tensor support INCLUDED (the earlier scope-out is gone).
Prep edits on top of the validated c2f2f6f: scrubbed the 1 jargon comment (Makefile.Hip
"followers (gfx1100, gfx1151)" -> neutral), documented TENSOR=ON in README.md (matrix-core
rocWMMA, NUMWI>=64), added AMD copyright lines to the 2 substantial port artifacts
(cuda_to_hip.h new, kernels.cu +254). Re-squashed the 4 fork commits -> ONE clean commit
d1b5e7b5 (full port, 10 files +569/-24). Delta vs c2f2f6f is doc/comment/copyright-only, so
gfx90a/gfx1100/gfx1201 carried forward (classify false-positived "mixed" on Makefile.Hip
unknown file type; verified inert by full diff). gfx1101 revalidate (not required; gfx1201
satisfies the Windows tier), gfx1151 port-ready (host retired). pr-ready=True, 0 jargon.
Commit message (autodock_pr_msg.txt): scope-out paragraph replaced with rocWMMA tensor
support; Test Plan includes TENSOR=ON. AWAITING jeff's go to open the upstream PR.

## Revalidation 2026-06-11 (windows-gfx1101) -- full GPU re-run at bb2f062

Platform: AMD Radeon PRO V710 (gfx1101, RDNA3, wave32), ROCm 7.14 (TheRock PyTorch
venv), Windows 11 Pro for Workstations 10.0.26200. HIP_VISIBLE_DEVICES=0.

### Delta classification: 221e2d4 -> bb2f062

`classify` returned `class=mixed arch_independent=False`. The delta (a squash of all
work from 221e2d4 to bb2f062, including the rocWMMA TENSOR=ON CDNA+RDNA branches,
the Windows min/max guard, and the re-prep doc/comment/copyright edits) contains real
device code in cuda/kernels.cu and host/src/performdocking.cpp, so carry-forward is not
appropriate -- full GPU revalidation was performed.

### Binary-equivalence check: TENSOR=OFF default build

Built at both 221e2d4 and bb2f062 for gfx1101 (--offload-arch=gfx1101, same flags) and
compared ELF .text sections embedded in the PE .hip_fat section:

- 64wi TENSOR=OFF: .text sha256 ee3eb094d3c7e89f... (IDENTICAL old vs new)
- 128wi TENSOR=OFF: .text sha256 ee3eb094d3c7e89f... (IDENTICAL old vs new)
- .symtab: IDENTICAL for both variants
- Only delta: __hip_cuid_ compilation-unit ID string (non-functional metadata);
  .gnu.hash/.hash entries derived from it also differ (expected, non-functional)

The TENSOR=OFF default build is binary-equivalent on gfx1101: no ISA change.

### Build commands (Windows, direct compiler invocation, gfx1101)

Same toolchain as the original 2026-06-05 gfx1101 validation:

```
SITE=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages
HIPCC=$SITE/_rocm_sdk_devel/bin/hipcc.exe
CXX=$SITE/_rocm_sdk_devel/lib/llvm/bin/clang++.exe
ROCM_INC=$SITE/_rocm_sdk_devel/include
ROCM_LIB=$SITE/_rocm_sdk_devel/lib
SRC=B:/develop/moat/projects/AutoDock-GPU/src
IFLAGS="-I$SRC/common -I$SRC/host/inc -I$SRC/cuda -I$ROCM_INC"
HOST_DEFS="-DUSE_HIP -D__HIP_PLATFORM_AMD__ -D_AMD64_ -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS"

# Kernel TU (64wi, TENSOR=OFF)
HIP_VISIBLE_DEVICES=0 $HIPCC -DN64WI --offload-arch=gfx1101 -x hip -std=c++17 -O3 -ffast-math \
  -D_AMD64_ -DUSE_HIP $IFLAGS -c $SRC/cuda/kernels.cu -o kernels_64wi_off.obj

# Kernel TU (64wi, TENSOR=ON)
HIP_VISIBLE_DEVICES=0 $HIPCC -DN64WI --offload-arch=gfx1101 -x hip -std=c++17 -O3 -ffast-math \
  -D_AMD64_ -DUSE_HIP -DUSE_NVTENSOR $IFLAGS -c $SRC/cuda/kernels.cu -o kernels_64wi_on.obj

# (repeat with -DN128WI for 128wi variants)
# Link each variant with host .objs + -lamdhip64
```

All 4 variants (64wi/128wi x TENSOR=OFF/ON) built cleanly at bb2f062 (pre-existing
warnings only: nodiscard on hipDeviceReset, strncpy/stricmp deprecation,
INFINITY undefined-behavior under -ffast-math).

### Docking results (1stp streptavidin-biotin, --nrun 10, HIP_VISIBLE_DEVICES=0)

TENSOR=OFF (default build, ISA-identical to prior validated sha 221e2d4):

| build / seed     | best binding energy | best-pose reference RMSD | cluster       |
|------------------|---------------------|--------------------------|---------------|
| 64wi  / seed 42  | -8.28 kcal/mol      | 0.59 A                   | 1, 10/10 runs |
| 64wi  / seed 7   | -8.37 kcal/mol      | 0.39 A                   | 1, 10/10 runs |
| 128wi / seed 42  | -8.30 kcal/mol      | 0.35 A                   | 1, 10/10 runs |

TENSOR=ON (new RDNA bf16 error-correction path, rocWMMA v_wmma_f32_16x16x16_bf16):

| build / seed     | best binding energy | best-pose reference RMSD | cluster       |
|------------------|---------------------|--------------------------|---------------|
| 64wi  / seed 42  | -8.35 kcal/mol      | 0.43 A                   | 1, 10/10 runs |
| 64wi  / seed 7   | -8.37 kcal/mol      | 0.38 A                   | 1, 10/10 runs |
| 128wi / seed 42  | -8.30 kcal/mol      | 0.45 A                   | 1, 10/10 runs |

Reference (wave64, gfx90a lead):

| build / seed     | best binding energy | best-pose reference RMSD |
|------------------|---------------------|--------------------------|
| 64wi  / seed 42  | -8.28 kcal/mol      | 0.48 A                   |
| 64wi  / seed 7   | -8.37 kcal/mol      | 0.37 A                   |
| 128wi / seed 42  | -8.33 kcal/mol      | 0.39 A                   |

All 6 cases: all 10 runs converge to a single cluster; best energies match the
gfx90a lead within 0.07 kcal/mol; best-pose reference RMSD sub-0.6 A in all cases.
No NaN, no HIP fault, clean exit. TENSOR=ON and TENSOR=OFF results agree within
0.07 kcal/mol -- the new RDNA bf16 path is numerically correct; it does not produce
the ~+1e8 garbage that the broken CDNA-only path would produce on RDNA. No DLL
issues -- amdhip64 loaded correctly from System32 (driver ABI path).

PASS: gfx1101 (RDNA3, wave32) TENSOR=OFF and TENSOR=ON both correct at bb2f062.
head_sha = bb2f062.

## PR fix-round 2026-07-02 (doc-only) -- NUMWI=64 guidance + community validation

Young-TW posted an independent validation on PR #320 (gfx1201 / RDNA4, RX 9070 XT,
Linux, ROCm 7.13.0): builds clean, docking results correct (3ce3 ~-14.58, 7cpa ~-21
kcal/mol), WARP_SIZE resolves to 32 on gfx1201 as designed. His NUMWI sweep found
NUMWI=64 fastest across 3ce3 and 7cpa (32 +16-18%, 128 +7-17%, 256 +55-64% slower).
rocprofv3 shows gpu_gradient_minAD_kernel (ADADELTA local search) is 99.6% of GPU
time and VGPR-bound: 152 VGPR, 0 spill, 7680 B LDS/block -> 9/16 waves/SIMD. His
amdgpu_waves_per_eu attempt did not move VGPR (already at __launch_bounds__ max
occupancy); 152 VGPR is intrinsic to the kernel live set.

Action (this round):
- README.md Hints line: added "On AMD GPUs, `NUMWI=64` was fastest in our tests on
  both consumer (RDNA) and datacenter cards." Doc-only, house style (prose in the
  existing Hints line). Commit 2966dcd on top of bb2f062. classify bb2f062->2966dcd
  = doc-only, arch_independent, inert. advance-head carried linux-gfx1100,
  windows-gfx1101, windows-gfx1201 forward to 2966dcd; linux-gfx90a stays pr-open;
  gfx1151 stays blocked (retired). No GPU re-run.
- Posted one PR comment (issuecomment-4869331270): thanks + NUMWI=64 now in docs
  (2966dcd) + agreed the VGPR profile matches ours and register pressure is intrinsic
  so occupancy gains need source-level changes. Only approved upstream action.

7cpa input issue (Young-TW's minor note): confirmed pre-existing upstream and
unrelated to this PR. input/7cpa/derived/7cpa_protein.maps.fld references
`variable N file=rec.A.map` etc., but the shipped map files are named
`7cpa_rec.A.map`. The port does not touch input/ (git diff 28a6139..moat-port --
input/ is empty), so this is a data-file naming mismatch in upstream's bundled
input, not a port fault. Noted as such in the PR reply; a separate fix if pursued.
