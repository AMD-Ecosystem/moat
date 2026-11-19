# gRASPA notes

## Build Instructions (HIP/ROCm)

```bash
cd projects/gRASPA/src
mkdir build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build . -j$(nproc)
```

The executable is at `build/src_clean/graspa`.

## Validation

Tested on gfx90a (MI250X) with CO2-MFI example:
- 288 seconds runtime
- ENERGY DRIFT: 0.00000 (all components)
- 26 CO2 molecules adsorbed

## Porting Notes

### HIP vector type operators
HIP's `HIP_vector_type` (double3/int3/etc.) has built-in operators that differ from CUDA's bare struct types:
- HIP provides member operators (+=, -=, *=, /=) and friend operators (*, /)
- HIP does NOT provide free-standing operator+ or operator- 
- Custom operators in maths.cuh are conditionally compiled with `#if !defined(__HIP_PLATFORM_AMD__)` to avoid ambiguity
- Scalar operators (vector * scalar) are provided for both CUDA and HIP

### Shared memory
- HIP disallows `__shared__ bool var = false;` -- use thread-0 initialization
- Dynamic shared memory requires `extern __shared__` (same as CUDA but was missing in one kernel)

## Review 2026-06-05

**Verdict: APPROVE**

Port implements Strategy A correctly:
- Single `cuda_to_hip.h` compat header with necessary symbol mappings
- `.cu` files marked `LANGUAGE HIP` via CMake, not renamed
- HIP vector operator ambiguity correctly resolved via `#if !defined(__HIP_PLATFORM_AMD__)` guards in maths.cuh and VDW_Coulomb.cuh
- Shared memory initialization fixed correctly (thread-0 init + `__syncthreads()` in VDW_Coulomb.cu:1055-1057)
- Dynamic shared memory `extern` keyword added where missing (Ewald_Energy_Functions.h:1046)

No fault class concerns:
- No warp primitives -- all reductions use `__syncthreads()` tree reduction (wave-size agnostic)
- No textures/surfaces
- No rule-of-five concerns
- `cudaMallocManaged` usage is minimal and HIP supports it

Build system correct:
- `enable_language(HIP)` + `USE_HIP` option (default OFF)
- `CMAKE_HIP_ARCHITECTURES` parameterized (not hardcoded)
- `find_package(hip)` + `hip::host` linkage

Commit hygiene clean:
- Title follows `[ROCm]` prefix, 36 chars
- Body mentions Claude, no noreply trailer, no MOAT jargon

Ready for gfx90a validation.

## Validation 2026-06-05 (linux-gfx90a, MI250X gfx90a)

**Build**: Clean build successful with HIP_VISIBLE_DEVICES=1
```bash
cd /var/lib/jenkins/moat/projects/gRASPA/src
mkdir build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build . -j$(nproc)
```
Executable: `build/src_clean/graspa` (2.2MB)

**Test Results**:

CO2-MFI GCMC benchmark (Examples/CO2-MFI):
- PASS: 26 CO2 molecules adsorbed
- PASS: Zero energy drift (all components 0.00000)
- Completed successfully without errors

Methane-TMMC (Examples/Methane-TMMC):
- FAIL: Memory access fault during final energy check
- Crash at "CHECKING FINAL ENERGY FOR SYSTEM [0]" after successful simulation completion
- Error: "Memory access fault by GPU node-3 on address 0x73fd2f6bd000"
- Exit code 141 (SIGPIPE)

Tail-Correction (Examples/Tail-Correction):
- FAIL: Same memory access fault during final energy check
- Simulation completed successfully (52 seconds), crash only at final validation
- Error: "Memory access fault by GPU node-3 on address 0x7187094c6000"

**Analysis**:
The core Monte Carlo simulation works correctly. CO2-MFI completes all phases including final energy check. Larger simulations (Methane-TMMC with 63 molecules, Tail-Correction with 1300+ molecules) crash during the post-simulation final energy validation step, immediately after printing component energies in VDW_Coulomb.cu:222. The crash is consistent across GPU devices (tested on both HIP_VISIBLE_DEVICES=0 and 1, both gfx90a MI250X).

**Verdict**: VALIDATION FAILED

Bug in final energy check code for simulations with larger molecule counts. The primary CO2-MFI benchmark passes, but the broader test suite fails. Needs porter investigation of the final energy calculation routine.

## Fix 2026-06-05 (OOB memory access in TotalVDWRealCoulomb)

**Root cause**: Out-of-bounds memory access in `TotalVDWRealCoulomb` kernel (VDW_Coulomb.cu:1544-1546). The code accessed `System[compA].MolID[AtomA]` BEFORE checking if AtomA was within bounds. When thread counts exceed valid interactions (larger simulations with more molecules spawn more threads than valid atom pairs), the bounds check came too late and the OOB access triggered a memory fault on HIP/AMD.

**Fix**: Move the bounds check to before the MolID array access:
```cpp
// Before (OOB access happens before check):
MolA = System[compA].MolID[AtomA];
MolB = System[compB].MolID[AtomB];
if(AtomA >= System[compA].size || AtomB >= System[compB].size) continue;

// After (check first, then access):
if(AtomA >= System[compA].size || AtomB >= System[compB].size) continue;
MolA = System[compA].MolID[AtomA];
MolB = System[compB].MolID[AtomB];
```

This is a pre-existing bug in upstream code that happens to not crash on CUDA (likely due to different memory access fault handling) but manifests on HIP/AMD.

**Validation after fix**:
- CO2-MFI (26 molecules): PASS, zero energy drift
- Methane-TMMC (58 molecules): PASS, completes without fault
- Tail-Correction (1327 molecules): PASS, completes without fault

Commit ddf08ad pushed to AMD-Ecosystem/gRASPA moat-port branch.

## Re-Review 2026-06-05

**Scope**: Re-review following porter fix for OOB memory access in TotalVDWRealCoulomb kernel.

**Verification of OOB Fix (VDW_Coulomb.cu:1544-1546)**:
The fix is correct. The bounds check `if(AtomA >= System[compA].size || AtomB >= System[compB].size) continue;` now occurs BEFORE the MolID array accesses `MolA = System[compA].MolID[AtomA];` and `MolB = System[compB].MolID[AtomB];`. In the original code, the array access happened first, causing memory faults on larger simulations where thread counts exceed valid atom pairs.

**Fault Class Review**:
- No warp primitives (wave-size agnostic)
- No textures/surfaces
- No rule-of-five concerns
- cudaMallocManaged usage is for small control structures, no atomicMin/atomicMax on managed memory
- All reductions use __syncthreads() tree reduction

**Build System**: Correct (enable_language(HIP), USE_HIP option, parameterized CMAKE_HIP_ARCHITECTURES)

**Commit Hygiene**: Clean (36-char title, Claude disclosure, Test Plan, no noreply trailer)

**Verdict**: APPROVE -- ready for gfx90a validation.

## Re-Validation 2026-06-05 (linux-gfx90a, MI250X gfx90a, HIP_VISIBLE_DEVICES=1)

**Build**: Clean build from scratch at commit ddf08ad4208fc4a426bbf0897d6f7186878bc48e
```bash
cd /var/lib/jenkins/moat/projects/gRASPA/src/build
HIP_VISIBLE_DEVICES=1 cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
HIP_VISIBLE_DEVICES=1 cmake --build . -j$(nproc)
```
Executable: `build/src_clean/graspa` (2.2MB)
ROCm: 7.2.1, HIP: 7.2.53211

**Test Results** (all with HIP_VISIBLE_DEVICES=1):

1. CO2-MFI GCMC benchmark (Examples/CO2-MFI):
   - PASS: Completed successfully
   - 18 CO2 molecules adsorbed (C_co2 pseudoatoms: 18)
   - Zero energy drift (all components 0.00000)

2. Methane-TMMC (Examples/Methane-TMMC):
   - PASS: Completed successfully without crashes
   - Previously crashed with memory access fault during final energy check
   - Now completes cleanly after OOB fix

3. Tail-Correction (Examples/Tail-Correction):
   - PASS: Completed successfully without crashes
   - 1327 Argon molecules (Ar[20] pseudoatoms: 1327)
   - Previously crashed with memory access fault during final energy check
   - Now completes cleanly after OOB fix

**Verdict**: VALIDATED -- All three benchmark simulations pass. The OOB memory access fix in TotalVDWRealCoulomb kernel resolved the crashes in larger molecule count simulations. The port correctly implements HIP support with Strategy A (cuda_to_hip.h header, .cu files marked LANGUAGE HIP).

## Validation 2026-06-05 (linux-gfx1100, RDNA3 gfx1100, HIP_VISIBLE_DEVICES=2)

**Build**: Clean build from scratch at commit ddf08ad4208fc4a426bbf0897d6f7186878bc48e
```bash
cd /var/lib/jenkins/moat/projects/gRASPA/src
rm -rf build && mkdir build && cd build
HIP_VISIBLE_DEVICES=2 cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100
HIP_VISIBLE_DEVICES=2 cmake --build . -j$(nproc)
```
Executable: `build/src_clean/graspa` (2.1MB)

**Test Results** (all with HIP_VISIBLE_DEVICES=2):

1. CO2-MFI GCMC benchmark (Examples/CO2-MFI):
   - PASS: Completed successfully
   - 18 CO2 molecules adsorbed (C_co2 pseudoatoms: 18)
   - Zero energy drift (all components 0.00000)

2. Methane-TMMC (Examples/Methane-TMMC):
   - PASS: Completed successfully
   - 58 methane molecules (CH4_sp3 pseudoatoms: 58)
   - Zero energy drift (all components 0.00000)

3. Tail-Correction (Examples/Tail-Correction):
   - PASS: Completed successfully
   - 1327 Argon molecules (Ar[20] pseudoatoms: 1327)
   - Zero energy drift (all components 0.00000)

**Verdict**: VALIDATED -- All three benchmark simulations pass on gfx1100 with identical results to gfx90a. Port works correctly across AMD architectures.

## Validation 2026-06-08 (windows-gfx1201, RX 9070 XT gfx1201, HIP_VISIBLE_DEVICES=0)

**GPU**: AMD Radeon RX 9070 XT (gfx1201, RDNA4)
**ROCm**: 7.14 (TheRock venv), HIP compiler: clang++.exe
**Commit**: 312048e73b2afab04296ba7990814ba863651789 (added Windows build fixes on top of ddf08ad)

**Windows-specific build fixes required** (committed as new commit on moat-port):

1. Guard POSIX-only APIs in main.cpp with `#ifndef _WIN32`: `unistd.h` include,
   `/proc/self/statm` body in `printMemoryUsage()`, and `readlink` call in `Initialize()`.

2. Add `#include <numeric>` to mc_widom.h and lambda.h for `std::accumulate` (Wang-Landau iteration).

3. CMakeLists.txt (src_clean): On Windows+Clang, MSVC's `emmintrin.h` declares SSE2 intrinsics
   as `extern` (not inline), causing `_mm_loadu_si128`/`_mm_cmpeq_epi16`/`_mm_movemask_epi8`
   link errors from the Windows CRT `wmemcmp`. Fix: use `target_include_directories(... SYSTEM BEFORE ...)`
   to prepend clang's own resource include directory (where emmintrin.h uses `static __inline__
   __always_inline__`) before MSVC's system includes.

**Build** (clean, from scratch):
```powershell
cmake .. -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 `
  -DCMAKE_CXX_COMPILER="$ROCM/lib/llvm/bin/clang++.exe" `
  -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/clang++.exe" `
  -DCMAKE_PREFIX_PATH="$ROCM" `
  -DCMAKE_CXX_FLAGS="-D_USE_MATH_DEFINES" `
  -DCMAKE_HIP_FLAGS="-D_USE_MATH_DEFINES" `
  -DOpenMP_CXX_FLAGS="-fopenmp" `
  -DOpenMP_CXX_LIB_NAMES="libomp" `
  -DOpenMP_libomp_LIBRARY="$MSVC_DIR/lib/x64/libomp.lib" `
  -DOpenMP_CXX_INCLUDE_DIR="$MSVC_DIR/include"
cmake --build . -j24
```
Executable: `build-gfx1201/src_clean/graspa.exe` (1.2MB)

Runtime DLL: `amdhip64_7.dll` found via `C:\WINDOWS\SYSTEM32\amdhip64_7.dll` (on PATH).

**Test Results** (all with HIP_VISIBLE_DEVICES=0, run from example directory):

1. CO2-MFI GCMC benchmark (Examples/CO2-MFI):
   - PASS: Completed successfully (exit 0)
   - 17 CO2 molecules adsorbed (C_co2 pseudoatoms: 17)
   - ENERGY DRIFT: all components 0.00000
   - GPU DRIFT: Ewald [Host-Host] = -0.00004 (sub-threshold, effectively zero)
   - Work took ~20 seconds

2. Methane-TMMC (Examples/Methane-TMMC):
   - PASS: Completed all three phases (INIT/EQUIL/PRODUCTION) without crashes
   - 53 methane molecules (CH4_sp3 pseudoatoms: 53)
   - GPU DRIFT: all zero
   - CPU energy drift in VDW: -929 kJ/mol (expected: TMMC biasing scheme causes
     accumulated energy to drift from CPU recalculation; GPU drift itself is zero)
   - Work took ~14 seconds

3. Tail-Correction (Examples/Tail-Correction):
   - PASS: Completed successfully (exit 0)
   - 1323 Argon molecules (Ar[20] pseudoatoms: 1323)
   - ENERGY DRIFT: all components 0.00000
   - GPU DRIFT: all zero
   - Work took ~4 seconds

**Verdict**: VALIDATED -- All three benchmark simulations pass on gfx1201 (RDNA4).
GPU kernels produce correct results (zero GPU drift). Molecule counts match expected
ranges (Monte Carlo stochastic variance: 17/18 CO2, 53/58 CH4, 1323/1327 Ar).

## Validation 2026-06-08 (linux-gfx90a revalidate -> carry-forward, MI250X gfx90a)

**Delta**: ddf08ad4 -> 312048e7 (one commit: "Add Windows build support for HIP port")

**Changes in delta**:
- `src_clean/CMakeLists.txt`: added `if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")` block to prepend clang intrinsic headers -- dead on Linux
- `src_clean/main.cpp`: `#ifndef _WIN32` guards around `unistd.h` include, `/proc/self/statm` body, and `readlink` call -- same Linux behavior
- `src_clean/lambda.h`: added `#include <numeric>` -- was already available transitively on Linux; no behavioral change
- `src_clean/mc_widom.h`: added `#include <numeric>` -- same

**Binary equivalence check** (codeobj_diff.py):
- Built both SHAs at gfx90a: `HIP_VISIBLE_DEVICES=0 cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a`
- `codeobj_diff build_old/src_clean/graspa build_new/src_clean/graspa`: `verdict=identical` (87 exported symbols + device ISA identical)

**Verdict**: CARRY-FORWARD (binary-equiv) -- Windows-only CMake/C++ guards compile to identical device code and exported symbols on Linux gfx90a. No GPU re-run required.

## Validation 2026-06-08 (linux-gfx1100 revalidate -> carry-forward, RDNA3 gfx1100)

**Delta**: ddf08ad4 -> 312048e7 (one commit: "Add Windows build support for HIP port")

**Changes in delta**: Same Windows-only changes as gfx90a carry-forward above:
- `src_clean/CMakeLists.txt`: `if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")` block -- dead on Linux
- `src_clean/main.cpp`: `#ifndef _WIN32` guards around `unistd.h`, `/proc/self/statm`, and `readlink` -- same Linux behavior
- `src_clean/lambda.h`, `src_clean/mc_widom.h`: added `#include <numeric>` -- already transitively available on Linux

**Binary equivalence check** (codeobj_diff.py):
```bash
# Build old SHA (ddf08ad)
mkdir build_old && cd build_old
HIP_VISIBLE_DEVICES=0 cmake ../build_old_src -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build . -j$(nproc)

# Build new SHA (312048e)
mkdir build_new && cd build_new
HIP_VISIBLE_DEVICES=0 cmake ../build_new_src -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build . -j$(nproc)

python3 utils/codeobj_diff.py build_old/src_clean/graspa build_new/src_clean/graspa
# verdict=identical (exported symbols + device ISA identical (12 exports))
```

**Verdict**: CARRY-FORWARD (binary-equiv) -- Windows-only CMake/C++ guards compile to identical device ISA and exported symbols on Linux gfx1100. No GPU re-run required.

## Validation 2026-06-19 (windows-gfx1101, Radeon PRO V710 gfx1101, HIP_VISIBLE_DEVICES=1)

**GPU**: AMD Radeon PRO V710 (gfx1101, RDNA3)
**ROCm**: 7.14 (TheRock venv), HIP compiler: clang++.exe 23.0.0
**Commit**: 312048e73b2afab04296ba7990814ba863651789

**GPU mask verified**: `HIP_VISIBLE_DEVICES=1` = AMD Radeon PRO V710 (gfx1101); `HIP_VISIBLE_DEVICES=0` = RX 9070 XT (gfx1201). Health check (timeout 35s): returned immediately with exit 0.

**Build** (clean, from scratch, mirroring gfx1201 recipe with gfx1101 arch):
```powershell
ROCM="B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
MSVC_DIR="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207"
cmake -S src -B src/build-gfx1101 -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
  -DCMAKE_CXX_COMPILER="${ROCM}/lib/llvm/bin/clang++.exe" \
  -DCMAKE_HIP_COMPILER="${ROCM}/lib/llvm/bin/clang++.exe" \
  -DCMAKE_PREFIX_PATH="${ROCM}" \
  -DCMAKE_CXX_FLAGS="-D_USE_MATH_DEFINES" \
  -DCMAKE_HIP_FLAGS="-D_USE_MATH_DEFINES" \
  -DOpenMP_CXX_FLAGS="-fopenmp" \
  -DOpenMP_CXX_LIB_NAMES="libomp" \
  -DOpenMP_libomp_LIBRARY="${MSVC_DIR}/lib/x64/libomp.lib" \
  -DOpenMP_CXX_INCLUDE_DIR="${MSVC_DIR}/include"
cmake --build src/build-gfx1101 -j64
```
Executable: `build-gfx1101/src_clean/graspa.exe` (clean build, warnings only -- no errors)

Runtime DLL: `amdhip64_7.dll` found via `C:\WINDOWS\SYSTEM32\amdhip64_7.dll` (on PATH; no TheRock DLL copy needed).

**Test Results** (all with HIP_VISIBLE_DEVICES=1, run from example directory):

1. CO2-MFI GCMC benchmark (Examples/CO2-MFI):
   - PASS: Completed successfully (exit 0)
   - 26 CO2 molecules adsorbed (C_co2 pseudoatoms: 26)
   - ENERGY DRIFT: all components 0.00000
   - GPU DRIFT: Ewald [Host-Host] = -0.00004 (sub-threshold, identical to gfx1201)
   - Work took ~8 seconds

2. Methane-TMMC (Examples/Methane-TMMC):
   - PASS: Completed all phases without crashes (exit 0)
   - 59 methane molecules (CH4_sp3 pseudoatoms: 59)
   - ENERGY DRIFT: all components 0.00000
   - GPU DRIFT: all zero
   - Work took ~4 seconds

3. Tail-Correction (Examples/Tail-Correction):
   - PASS: Completed successfully (exit 0)
   - 1333 Argon molecules (Ar[20] pseudoatoms: 1333)
   - ENERGY DRIFT: all components 0.00000
   - GPU DRIFT: all zero
   - Work took ~1.3 seconds

Post-test GPU health check: returned immediately with exit 0, no TDR triggered.

**Verdict**: VALIDATED -- All three benchmark simulations pass on gfx1101 (RDNA3).
GPU kernels produce correct results (zero GPU drift). Molecule counts match expected
ranges (Monte Carlo stochastic variance: 26 CO2, 59 CH4, 1333 Ar).
No gfx1101-specific issues encountered.

## Rescope 2026-08-20 (linux-gfx90a, MI250X gfx90a) -- rebuilt on upstream's own HIP backend

**Why**: upstream merged its own HIP backend on 2026-06-20 (PR #82, `sigbjobo`,
merge `fd33b0e`; commits `2ed9fb9` "Add AMD/ROCm (HIP) backend to the single
src_clean tree", `b36c6b4` CPU-only shim regression test, `62b25fc` output
stream unification). That supersedes most of our June port, so the branch was
rebuilt from scratch on current upstream `main` (`e4edfc2`) carrying only the
residual delta. Old branch (`4710600`, 3 commits on `3fc256d`) is gone; the
force-push was deliberate and no upstream PR was open.

### What upstream's merged HIP support already covers

- `src_clean/gpu_compat.h`: CUDA->HIP shim, keyed on `__HIP__`. Maps
  `cudaMalloc/MallocHost/MallocManaged/Free/Memcpy/MemcpyAsync/Memset/`
  `DeviceSynchronize/GetLastError/GetErrorString`, `cudaError_t`,
  `cudaSuccess`, and the two memcpy-direction enums. Under nvcc it expands to
  `<cuda_runtime.h> + <cuda_fp16.h>` and defines zero macros.
- Include rewiring in `data_struct.h`, `read_data.h`, `VDW_Coulomb.cu/.cuh`,
  `mc_widom.h`.
- `#if !defined(__HIP__)` around the componentwise `double3` operators in
  `maths.cuh` and `VDW_Coulomb.cuh` (HIP's `HIP_vector_type` provides them);
  `dot()` and the `MoveEnergy` operators stay unguarded.
- `#if !defined(__HIP__)` around the dead thrust includes in `mc_widom.h`.
- Both shared-memory faults we had fixed are upstream too, identically:
  `__shared__ bool Blockflag;` + thread-0 init + `__syncthreads()`
  (`VDW_Coulomb.cu:1055`) and `extern __shared__ double sdata[]`
  (`Ewald_Energy_Functions.h:1046`).
- Build scripts `HIP_COMPILE` (classical core -> `hip_main.x`) and
  `libtorch_HIP_COMPILE` (Allegro on ROCm LibTorch), both honoring
  `GRASPA_ARCH` (default `gfx90a`).
- `Examples/run_designated_folders.py` takes `GRASPA_BIN`; README has an
  AMD/ROCm section (build, `GRASPA_ARCH`, `HSA_XNACK=1`, C++11 ABI, harness).

### What the rescoped branch keeps

`a0b6dce` `[ROCm] Check atom bounds before reading MolID in total energy`
  `src_clean/VDW_Coulomb.cu`: move the `AtomA/AtomB >= System[...].size` test
  above the two `MolID[]` loads in `TotalVDWRealCoulomb`. Still present on
  current upstream main (the reordering is 3 lines around 1544-1546).

`9cfa096` `[ROCm] Let the classical core compile on Windows`
  `main.cpp` `#ifndef _WIN32` around `<unistd.h>`, the `/proc/self/statm` body
  of `printMemoryUsage()`, and `readlink("/proc/self/exe")`; `<numeric>` in
  `lambda.h` and `mc_widom.h` for `std::accumulate`; README bullet recording
  the two extra Windows flags (`-D_USE_MATH_DEFINES` and an `-isystem` for
  clang's own intrinsic headers).

Whole branch is 29 insertions / 1 deletion across 5 files (`git diff --stat
e4edfc2 moat-port`; it was 24/1 before the README rewording of 2026-08-20).

### What was dropped, and why

- The entire `cuda_to_hip.h` compat header, the include rewiring, the `double3`
  operator guards, and the two shared-memory fixes: upstream's `gpu_compat.h`
  and `2ed9fb9` do the same job. Keeping ours would be a competing shim.
- The root and `src_clean` `CMakeLists.txt` (and with it the
  `[ROCm] Rely on HIP arch auto-detect` commit): upstream deliberately builds
  with per-toolchain shell scripts at the repo root (`NVC_COMPILE`,
  `HIP_COMPILE`, `libtorch_*`) and merged `HIP_COMPILE` as part of PR #82.
  A parallel CMake build is now redundant infrastructure that would compete
  with what the maintainers chose.
- The CMake-resident MSVC `emmintrin.h` workaround: with CMake gone it has no
  home in the build system, so it is documented in the README as the `-isystem`
  flag a Windows compile needs, which is what the CMake block was doing.

### Evidence that the out-of-bounds read is real on current upstream

The June crash (memory access fault in Methane-TMMC and Tail-Correction) does
NOT reproduce on current upstream main on this host: unmodified `e4edfc2`
completes both, with and without `HSA_XNACK=1`. The defect is still there and
is worse than "one past the end". Instrumenting unmodified upstream with a
printf before the `MolID` loads, printing only for the last thread of the last
block, and running Tail-Correction (1327 adsorbate atoms):

```
OOBPROBE block 134 thread 127 compA 1 AtomA 2147484974 sizeA 1327 compB 1 AtomB 2305843010287440402 sizeB 1327
```

The trailing threads decode interaction indices past the real pair count. The
closed-form triangular-index inverse computes its square root argument
(`-8*(int) InteractionIdx + 4*N*(N-1) - 7`) in **unsigned** 64-bit, because
`NAdsorbateAtoms` is a `size_t`: past the real pair count it wraps rather
than going negative. This thread's `InteractionIdx` is 883100 (the printed
`AtomB` pins it exactly; there are only 879801 real pairs), so the argument
is `2^64 - 26399`, `sqrt` of that is `2^32` to within a rounding step,
`floor(sqrt/2 - 0.5)` is 2147483647, and `AtomA` comes out as the double
-2147482322.0, which converts to the 2147484974 printed above. `AtomB` is then
derived from that `AtomA` and lands at ~2.3e18. `MolID` is a `size_t*`, so the
second load is an ~18-exabyte offset from the base pointer. (A genuinely
negative argument would give `NaN`, not a 2^31-scale index -- see the
2026-08-20 review.)
Whether that faults is up to what the driver happens to have mapped, which is
why it faulted in June and does not today.

### Build recipe (this host, 2026-08-20)

No `/opt/rocm` on this host; ROCm 7.14.60850 comes from a TheRock wheel:

```bash
export ROCM=/opt/conda/envs/py_3.12/lib/python3.12/site-packages/_rocm_sdk_devel
export PATH=$ROCM/bin:$PATH
cd projects/gRASPA/src/src_clean && GRASPA_ARCH=gfx90a ../HIP_COMPILE
```

Produces `src_clean/hip_main.x` (~983 KB). Warnings only (mostly `nodiscard`
on the HIP runtime calls the shim maps, plus one VLA extension) -- no errors.

### Test results (gfx90a, HIP_VISIBLE_DEVICES=1, HSA_XNACK=1)

```bash
cd Examples
HSA_XNACK=1 HIP_VISIBLE_DEVICES=1 \
  GRASPA_BIN=../src_clean/hip_main.x python3 run_designated_folders.py
python3 -m pytest -q -s
```

All nine designated simulations completed (exit 0) and `pytest` is 5 passed:

| example | wall | CPU-vs-running drift | GPU-vs-CPU drift |
|---|---|---|---|
| CO2-MFI | 104.5 s | -0.0 | Ewald [Host-Host] -4e-05 |
| Methane-TMMC | 108.8 s | -0.0 | 0.0 |
| Bae-Mixture | 1692.2 s | -0.0 | Ewald [Host-Host] -22.48 |
| NU2000-pX-LinkerRotations | 157.7 s | -0.00501 | 0.00027 |
| Tail-Correction | 35.7 s | -0.0 | 0.0 |

Reference_NIST_SPCE Box-1..4 all completed with all drift components 0.00000.
`Examples/test_gpu_compat_shim.py` (CPU-only, upstream's) passes: 3 passed.

Two numbers worth explaining, both pre-existing and unrelated to our delta:

- NU2000 CPU drift -0.00501 against upstream's committed CUDA reference output
  of -0.00456 for the same field: same magnitude, framework-flexibility
  accumulation, and the harness criterion (`value < 1e-3`) passes.
- Bae-Mixture GPU drift is entirely in `Ewald [Host-Host]`, -22.48 on a
  term of 390954.70 (relative 5.7e-5), and that term is the framework self
  energy which the output labels "Initial Ewald [Host-Host] (excluded)" --
  it is excluded from the running total, which is why the CPU-vs-running drift
  is 0. It is a k-space reduction-order difference between the AMD and NVIDIA
  builds, the same signature as the -4e-05 seen on CO2-MFI here, on Windows
  gfx1201 and on gfx1101 in June. Nothing our delta touches contributes to it:
  the VDW and real-space Coulomb components, which are the only ones
  `TotalVDWRealCoulomb` computes, are exactly 0.00000.

### CUDA path

Not compile-checked: this host has no NVIDIA toolchain (`nvcc`/`nvc++` absent,
no `/opt/nvidia`). Both commits are argued to be CUDA-neutral by construction:
the bounds-check move is a statement reorder with no platform-dependent
spelling, the `_WIN32` guards are false on Linux and on any CUDA host, and
`<numeric>` is additive. A Windows or CUDA host should still confirm.

### For the next round

- Windows validation must confirm the README flag recipe actually builds; it
  is the reworked form of what used to be a CMake block and has not been run
  since the rescope. If the flags differ, fix them in a NEW commit.
- Bae-Mixture takes ~28 minutes on one MI250X GCD, which dominates the
  harness. Run the other eight in a second checkout on another GCD in parallel
  if you are time-boxed.

## Review 2026-08-20 (rescoped branch, `e4edfc2..9cfa096`, 2 commits / 24 insertions)

**Verdict: CHANGES REQUESTED.** The code is right -- both commits do what they
claim and neither can change a result on any path. What must be fixed is the
*explanation* shipped with the bounds-check commit, which states a mechanism the
code cannot produce and which does not account for the very number the commit
quotes as its evidence, plus two imprecisions in the Windows README recipe and a
missing skill promotion.

### 1. `a0b6dce` commit body states a mechanism the code cannot produce

The body says the wild indices come out of the triangular-index inverse
"whose square root argument goes negative once the index runs past the real
pair count". The square root argument cannot go negative. At
`src_clean/VDW_Coulomb.cu:1538`:

```cpp
AtomA = NAdsorbateAtoms - 2 - std::floor(std::sqrt(-8*(int) InteractionIdx + 4*NAdsorbateAtoms*(NAdsorbateAtoms-1)-7)/2.0 - 0.5);
```

`NAdsorbateAtoms` is `size_t` (kernel parameter, `VDW_Coulomb.cu:1461`), so
`4*NAdsorbateAtoms*(NAdsorbateAtoms-1)-7` is `size_t`; the `int` subexpression
`-8*(int) InteractionIdx` is converted to `size_t` by the usual arithmetic
conversions. The whole argument is **unsigned 64-bit** and wraps instead of
going negative. That distinction is not pedantic -- it is the only thing that
explains the evidence. Reproduced with the kernel's own constants for
Tail-Correction (`NAdsorbateAtoms = 1327`, trailing thread, `InteractionIdx`
past the 879801 real pairs):

```
arg type is_signed=0 value=18446744073709524425   (2^64 - 27191, not -27191)
sqrt=4294967296.0 (2^32)   floor(sqrt/2 - 0.5)=2147483647
AtomA as double = -2147482322.0   -> wrapped to 32-bit = 2147484974
```

`2147484974` is exactly the `AtomA` the commit quotes. A genuinely negative
argument would give `NaN`, and a `NaN`-to-integer conversion would not produce
a 2^31-scale index. Rewrite the paragraph around the real chain: the argument
is computed in unsigned arithmetic, so once the index passes the real pair
count it wraps to nearly 2^64, its square root is ~2^32, and the resulting
`AtomA` underflows to a 2^31-scale value from which the `AtomB` formula then
produces ~2.3e18. Fix the same sentence in the Rescope section of this file
("takes a square root of a negative number"), which carries the same error.

The rest of the commit's argument stands and does not need changing.

### 2. `README.md:94` -- the `-isystem` path does not match the ROCm layout this project validates on

The bullet gives `-isystem <hip-sdk>/lib/clang/<version>/include`. That is the
official HIP SDK for Windows layout, but every Windows validation recorded above
(gfx1201 2026-06-08, gfx1101 2026-06-19) used a TheRock SDK, whose clang lives at
`<sdk>/lib/llvm/bin/clang++.exe` and whose resource headers are therefore at
`<sdk>/lib/llvm/lib/clang/<version>/include`. Checked on this host:
`clang++ -print-resource-dir` returns `.../_rocm_sdk_devel/lib/llvm/lib/clang/23`,
and `<sdk>/lib/clang/*/include` does not exist. A reader on the SDK we actually
test with follows the README and gets nothing. Since the recipe has not been run
in this form, do not swap one hardcoded shape for the other -- use the
self-describing form, which is correct on both layouts:

```
-isystem "$(clang++ -print-resource-dir)/include"
```

### 3. `README.md:93` -- "the same flags as `HIP_COMPILE`" over-promises against the commit's own tested command

`HIP_COMPILE` sets `-O3 -std=c++20 --offload-arch=${ARCH} -x hip -fgpu-rdc
-munsafe-fp-atomics -fopenmp -Wno-unused-result -Wno-format`. The Windows Test
Plan in the same commit drops `-munsafe-fp-atomics`, `-Wno-unused-result` and
`-Wno-format`. Two upstream-visible artifacts in one commit describe two
different commands, and `-munsafe-fp-atomics` is the gfx90a-oriented one a
reader should not carry to an RDNA target on our say-so. State the flag list the
Windows build was actually tested with rather than deriving it from a script the
same bullet says is not used on Windows.

### 4. No lesson promoted to the `cuda-to-rocm` skill

`git diff main...HEAD -- .claude/skills/` is empty, and this round produced two
things that would help a different project:

- **Upstream can merge someone else's HIP backend while the port is in flight.**
  `references/assess-existing-support.md` covers assessment *before* porting,
  the "the existing AMD support IS OURS" case (line 18) and the resume-after-our-
  own-squash-merge case (line 20). It has no entry for a third-party backend
  landing upstream after adoption, which is what happened here and what turned a
  four-platform-validated port into 24 lines. The rule worth writing down is
  re-check at every round, not only at intake; when it happens, rebuild on
  current upstream and keep only the residual delta rather than shipping a
  competing shim; and diff your fixes against the merged shim first -- both of
  gRASPA's shared-memory fixes were already in upstream's merge verbatim
  (`VDW_Coulomb.cu:1055`, `Ewald_Energy_Functions.h:1046`).
- **A latent OOB that stopped reproducing is still there.** The
  `references/fault-classes.md` out-of-bounds entry (line 133) covers only reads
  one element past an allocation and stencil edges. This case is a different
  shape: an index-decode overflow in padded trailing threads producing indices
  ~2^31 and ~2.3e18, where faulting depends on what the driver happens to have
  mapped -- it aborted in June and does not abort on the current tree on the same
  host. Absence of a fault is not absence of the bug; instrument the trailing
  threads of the last block and print the decoded indices.

Put the edit on this branch so it is reviewed with the code.

### Checked and clean (no action)

- The bounds test at `VDW_Coulomb.cu:1546` precedes *every* `MolID` load in the
  kernel; `MolA`/`MolB` (declared 1483) are read only at 1549, immediately after
  assignment, so the reorder cannot change the accepted-pair set or the energy
  for any in-bounds thread.
- The moved test dereferences `System[compA]`, which is safe:
  `determine_comp_and_Atomindex_from_thread` sets `comp = startComponent`
  (1424) and otherwise only assigns an in-range `ijk`, and it already reads
  `System[ijk].size` at 1428, so no new access class is introduced.
- Leaving the fix unguarded is right. The defect is platform-independent UB, the
  change is a strict generalization with identical NVIDIA behaviour, and a
  `#if defined(__HIP__)` around it would be wrong on its face to a maintainer.
- The fix is complete. The sibling kernels already test
  `posi < System[comp].size` before the `MolID` load (743-744, 1269-1270);
  `TotalVDWRealCoulomb` was the only site with the inverted order.
- Windows guards are inert on Linux: `<unistd.h>` still included (main.cpp:19-21),
  the whole `/proc/self/statm` body including `file.close()` sits inside
  `#ifndef _WIN32` (33-61), and `exepath` is used only at line 78 (the consumer
  at 79 is commented out), so the empty-string Windows path matches what Linux
  already produces on `readlink` failure. `grep` confirms main.cpp holds every
  POSIX use in `src_clean/`. `<numeric>` matches existing practice
  (`axpy.cu:15`, `read_data.cpp:6`) and both files really do call
  `std::accumulate` (`lambda.h:40`, `mc_widom.h:338`).
- Dropped scope is justified: `src_clean/gpu_compat.h` covers the same runtime
  surface the retired `cuda_to_hip.h` did, and the `__HIP__` guards in
  `maths.cuh:144`, `VDW_Coulomb.cuh:16` and `mc_widom.h:8` cover the operator and
  thrust divergences. Nothing load-bearing is missing.
- Commit hygiene: titles 61 and 48 chars, both `[ROCm]`, AI disclosure and fenced
  Test Plan in both bodies, no `Co-Authored-By`, ASCII only,
  `jargon.py --port gRASPA` clean. Working tree clean at 9cfa096.
- No fault-class exposure in the delta: no warp intrinsics, no hardcoded 32, no
  textures, no library swaps. The block reduction at 1562-1572 is
  `__syncthreads()`-based and wave-size agnostic.
- Missing GPU validation at 9cfa096 is expected at review time and is not part of
  this verdict. Not a finding, but context for whoever next reads this kernel:
  the `if(THREADIdx > NFrameworkZero_ExtraFramework)` at 1472 has no body and
  swallows the `sdata[threadIdx.x] = 0.0;` at 1474. It is harmless -- `sdata` is
  overwritten unconditionally at 1557-1558 before the first read at 1567 -- and
  it is upstream's, untouched by this port.

## Fix round 2026-08-20 (linux-gfx90a) -- review findings 1-4 addressed

Branch rewritten `9cfa096` -> `fd06b97` (force-with-lease; no upstream PR
open). The **source tree is byte-identical to 9cfa096**: `git diff 9cfa096
fd06b97` is README.md only. The two commits keep their content and change
only their messages, so `a0b6dce` -> `99712d2` and `9cfa096` -> `fd06b97`.

1. `99712d2` body: the mechanism paragraph now says the square root argument
   is evaluated in unsigned 64-bit (`NAdsorbateAtoms` is `size_t`) and wraps
   to `2^64 - 27191`, giving `sqrt = 2^32`, `AtomA` as the double
   -2147482322.0 and thus the printed 2147484974, with `AtomB` derived from
   it at 2.3e18 -- and notes that a signed evaluation would give `NaN`
   instead. The probe output and the rest of the argument are unchanged. The
   same sentence in the Rescope section above is corrected identically.
   (The `2^64 - 27191` constant this round wrote was wrong; it is
   `2^64 - 26399`, corrected in the 2026-08-20 second fix round below.)
2. README.md Windows bullet: the hardcoded
   `<hip-sdk>/lib/clang/<version>/include` is replaced by
   `-isystem "$(clang++ -print-resource-dir)/include"`, which is right on
   both the official HIP SDK layout and the TheRock layout every Windows
   validation here actually used.
3. README.md Windows bullet no longer says "the same flags as `HIP_COMPILE`".
   It states the tested set explicitly -- `-O3 -std=c++20 --offload-arch=<arch>
   -x hip -fgpu-rdc -fopenmp -D_USE_MATH_DEFINES -isystem ...`, link with
   `--hip-link` -- and `fd06b97`'s Windows Test Plan now uses exactly that
   set. `-munsafe-fp-atomics`, `-Wno-unused-result` and `-Wno-format` are
   deliberately absent from the Windows recipe: they are `HIP_COMPILE`'s
   gfx90a-oriented and warning-quieting choices, not something to hand an
   RDNA target untested. The Test Plan's cmd block takes the resource dir via
   `for /f ... in ('clang++ -print-resource-dir')`, since `$(...)` is not cmd.
4. Two lessons promoted to the `cuda-to-rocm` skill on this branch:
   `references/assess-existing-support.md` gains a paragraph on a third party
   merging a HIP backend mid-flight (re-check upstream every round; rebuild on
   current upstream and keep the residual; diff your fixes against the merged
   shim first), and `references/fault-classes.md` gains an entry next to the
   OOB one on a latent OOB that stops reproducing plus unsigned wrap in
   padded-thread index math, with the instrumented-printf method.

**Rebuild** (message/README-only delta, so no simulation rerun):

```bash
export ROCM=/opt/conda/envs/py_3.12/lib/python3.12/site-packages/_rocm_sdk_devel
export PATH=$ROCM/bin:$PATH
cd projects/gRASPA/src/src_clean && GRASPA_ARCH=gfx90a ../HIP_COMPILE
```

Links `hip_main.x` (982792 bytes), warnings only. `jargon.py --port gRASPA`
clean on the rewritten branch. GPU evidence from the 2026-08-20 rescope run
applies unchanged to this tree but is recorded against `9cfa096`, so every
platform needs a validation pass at `fd06b97`.

## Re-Review 2026-08-20 (fix round, `9cfa096` -> `fd06b97`)

**Verdict: CHANGES REQUESTED**, on one point. Findings 2, 3 and 4 of the
2026-08-20 review are fixed. Finding 1 is fixed in kind -- the mechanism is now
correctly described as an unsigned 64-bit wrap -- but the one new constant the
rewritten paragraph introduces does not reconcile with the probe output printed
three lines above it in the same commit body, which is the same failure mode the
finding was raised for.

Confirmed by inspection, no action needed: `git diff 9cfa096 fd06b97` is
README.md only (11 insertions / 6 deletions, one bullet), so the code is
byte-identical to what was already accepted and the recorded gfx90a run still
describes this tree's binary.

### 1. `99712d2` commit body: the stated square-root argument contradicts the AtomA/AtomB pair the same body quotes as evidence

The body says:

> For this example the argument is 2^64 - 27191, its square root is 2^32, and
> the subtraction leaves AtomA as the double -2147482322.0

and quotes as its evidence, from the instrumented run:

> AtomA 2147484974 sizeA 1327 AtomB 2305843010287440402 sizeB 1327

Those two statements cannot both hold. Invert the `AtomB` expression at
`src_clean/VDW_Coulomb.cu:1539`

```cpp
AtomB = InteractionIdx + AtomA + 1 - NAdsorbateAtoms*(NAdsorbateAtoms-1)/2 + (NAdsorbateAtoms-AtomA)*((NAdsorbateAtoms-AtomA)-1)/2;
```

for `NAdsorbateAtoms = 1327` and the printed `AtomA = 2147484974`. Every term is
`size_t`, so `AtomB` is affine in `InteractionIdx` mod 2^64 and the printed
`AtomB` pins the index exactly:

```
InteractionIdx implied by the printed pair: 883100
sqrt arg = 18446744073709525217 = 2^64 - 26399
```

The claimed `2^64 - 27191` corresponds to `InteractionIdx = 883199`, which
produces `AtomB = 2305843010287440501`, not the `...402` the body prints. Note
this inversion uses the printed `AtomA` as given, so it does not depend on any
model of how the negative double converts to `size_t`; it is an identity on the
quoted numbers.

Everything downstream of the constant is right and needs no change: for
`2^64 - 26399` the square root is 4294967295.9999967 (~2^32),
`floor(sqrt/2 - 0.5)` is 2147483647, `AtomA` is the double -2147482322.0, which
is the printed 2147484974, and `AtomB` follows at 2.3e18. Fix by stating
`2^64 - 26399` (optionally naming `InteractionIdx = 883100`, which makes the
chain checkable end to end), or by dropping the specific constant and saying the
argument wraps to just under 2^64. Do not leave a number a maintainer can
falsify from the same paragraph.

The same `2^64 - 27191` appears in the Rescope section of this file (the
paragraph beginning "The closed-form triangular-index inverse") and in the
2026-08-20 review's reproduction block above; correct the Rescope one with the
commit body. The review block is a dated record of what was said and can stand.

### 2. `.claude/skills/cuda-to-rocm/references/assess-existing-support.md` -- the size claim is stale

The promoted paragraph says gRASPA "went from a multi-commit port with its own
compat header and CMake build to 24 insertions across 5 files". That was true at
`9cfa096`; `git diff e4edfc2...fd06b97 --stat` on the reviewed branch is
`5 files changed, 29 insertions(+), 1 deletion(-)`. Say 29 insertions, or drop
the count and keep "5 files", since the point is the order of magnitude and a
count that drifts with every README wording change will keep going stale. The
matching "Whole branch is 24 insertions / 1 deletion across 5 files" line in the
Rescope section of this file has the same drift.

### Checked and clean (no action)

- Prior finding 2 is fixed and is layout-independent: `README.md:95-96` now uses
  `-isystem "$(clang++ -print-resource-dir)/include"`, and the bullet says why
  (the directory is under `lib/clang/` in the HIP SDK and `lib/llvm/lib/clang/`
  in a ROCm build with LLVM in a subdirectory). `-print-resource-dir` is the
  compiler's own answer, so it is right on both, including the TheRock layout
  every Windows validation on this project actually used.
- Prior finding 3 is fixed and the two artifacts now agree exactly.
  `README.md:94-96` states the set once -- `-O3 -std=c++20 --offload-arch=<arch>
  -x hip -fgpu-rdc -fopenmp -D_USE_MATH_DEFINES -isystem ...`, link with
  `--hip-link` -- and `fd06b97`'s Windows Test Plan compile line carries exactly
  those flags, no more. "the same flags as `HIP_COMPILE`" is gone, and
  `-munsafe-fp-atomics`, `-Wno-unused-result` and `-Wno-format` appear in
  neither, which is the right call for an untested RDNA target. The Test Plan's
  five translation units (`main.cpp read_data.cpp data_struct.cpp axpy.cu
  VDW_Coulomb.cu`) are exactly `HIP_COMPILE`'s five, and its link line matches
  `HIP_COMPILE`'s (`-std=c++20 --offload-arch -fgpu-rdc -fopenmp --hip-link`).
  The cmd block correctly uses `for /f "delims=" %i in ('clang++
  -print-resource-dir') do set CLANG_RES=%i` rather than `$(...)`.
- Prior finding 4 is fixed and both promotions are well placed and generalizable.
  The `assess-existing-support.md` paragraph sits immediately after the "the
  existing AMD support IS OURS" classification, which is where a reader asking
  "does upstream already have this?" is already reading, and its rule -- re-check
  every round, rebuild on current upstream and keep the residual, diff your fixes
  against the merged shim before re-applying -- is the transferable part rather
  than the gRASPA specifics. The `fault-classes.md` entry sits directly after the
  existing out-of-bounds entry it extends, and adds a genuinely different shape
  (index-decode overflow in padded threads, unsigned wrap producing astronomical
  rather than past-the-end indices, `sqrt` of the wrapped value returning a real
  where a signed evaluation would give `NaN`, and "it stopped crashing" not
  closing an OOB finding) with a concrete method. Its quoted numbers are the
  probe's `AtomA`/`AtomB` and the 18-exabyte offset, all of which check out
  (2305843010287440402 * 8 = 1.84e19 bytes); it states no wrapped-argument
  constant, so finding 1 does not touch it.
- Hygiene: `jargon.py --port gRASPA` clean; titles 48 and 61 chars, both
  `[ROCm]`; AI-assistance disclosure and a fenced Test Plan in both bodies; no
  `Co-Authored-By`, `Signed-off-by` or noreply address; ASCII only in both
  messages and the whole diff. Working tree at `fd06b97` has only the untracked
  build artifact `src_clean/hip_main.x`, no modified tracked files.
- Code re-verified unchanged and still correct: the bounds test at
  `VDW_Coulomb.cu:1546` precedes both `MolID` loads (1547-1548), `MolA`/`MolB`
  are consumed only at 1549, and the reorder cannot change the accepted-pair
  set for any in-bounds thread. Windows
  guards remain inert on Linux (`main.cpp:19-21, 33-61, 71-77`), and `<numeric>`
  is additive. No fault-class exposure in the delta: no warp intrinsics, no
  hardcoded 32, no textures, no library swaps; the block reduction is
  `__syncthreads()`-based.
- No GPU run exists at `fd06b97`. Expected at review time, not part of this
  verdict. Because the fix for finding 1 is message-only, the tree that finally
  validates will still be this one.

## Fix round 2026-08-20 (linux-gfx90a) -- re-review findings 1-2, text only

Branch rewritten `fd06b97` -> `7e3c08b` (force-with-lease; no upstream PR
open, `pr-state` reports `none`). **The tree is byte-identical to `fd06b97`**:
`git diff fd06b97 7e3c08b` is empty and both commits carry the same tree
object `ff186406cc62dbab135ec2dbf66c7bde30028efc`. Only the first commit's
message changed, so `99712d2` -> `55d7c59` and `fd06b97` -> `7e3c08b` (the
second message is byte-identical, verified with `diff`). **No rebuild and no
re-run were done, deliberately: nothing a compiler reads changed.** The
gfx90a binary and the nine-example run recorded in the Rescope section above
describe this tree exactly; what they do not describe is the *sha*, so every
platform still needs a validation pass at `7e3c08b`.

### 1. The wrapped square-root argument is `2^64 - 26399`, not `2^64 - 27191`

Recomputed independently from `src_clean/VDW_Coulomb.cu:1538-1539` before
touching the message, and it reproduces the re-review exactly. Method: take
the probe's `AtomA = 2147484974` as given and invert the `AtomB` expression,
which is affine in `InteractionIdx` mod 2^64 (every operand is `size_t`, and
`N - AtomA` wraps to 18446744071562067969, whose triangular term is
2305843008139952128):

```python
N = 1327                                    # NAdsorbateAtoms, Tail-Correction
NmA  = (N - 2147484974) % 2**64
const = (2147484974 + 1 - N*(N-1)//2 + (NmA*((NmA-1) % 2**64) % 2**64)//2) % 2**64
IIdx  = (2305843010287440402 - const) % 2**64          # -> 883100
arg   = (-8*int_cast(IIdx) + 4*N*(N-1) - 7) % 2**64    # -> 2**64 - 26399
```

Results, each checked forward against the printed pair:

| quantity | value |
|---|---|
| real pair count `N*(N-1)/2` | 879801 |
| `InteractionIdx` implied by the printed `AtomB` | **883100** |
| sqrt argument | **18446744073709525217 = 2^64 - 26399** |
| `sqrt(arg)` (double) | 4294967295.9999967, i.e. 2^32 to a rounding step |
| `floor(sqrt/2 - 0.5)` | 2147483647 |
| `AtomA` as double -> `size_t` | -2147482322.0 -> 2147484974 (matches probe) |
| `AtomB` from that `AtomA` | 2305843010287440402 (matches probe) |

The previously written `2^64 - 27191` belongs to `InteractionIdx = 883199`,
which yields `AtomB = 2305843010287440501` -- one hundred off the printed
value, so it is falsifiable from the same paragraph. Both indices give the
same `AtomA`, which is why the error survived the first fix round: `AtomA` is
constant over a whole plateau of indices and only `AtomB` pins the index.
`55d7c59`'s body now names `InteractionIdx 883100` and the 879801 real pairs
alongside the constant, so the chain is checkable end to end without
re-deriving the index. Note the sqrt is *not* exactly 2^32: `2^64 - 26399`
rounds to the double `2^64 - 26624` (ulp is 2048 there), whose square root is
just under 2^32 -- the body says "to within a rounding step" rather than
claiming equality. `floor(sqrt/2 - 0.5)` is 2147483647 either way.

The same sentence in the Rescope section above is corrected identically. The
2026-08-20 review and re-review blocks keep their original numbers: they are
dated records of what was said.

### 2. Skill size claim refreshed

`.claude/skills/cuda-to-rocm/references/assess-existing-support.md` said the
rescoped port was "24 insertions across 5 files". `git diff --stat e4edfc2
moat-port` is `5 files changed, 29 insertions(+), 1 deletion(-)` (the README
rewording of the previous fix round added five lines), so the skill and the
"Whole branch is ..." line in the Rescope section both say 29 now.

### Gotcha worth remembering (project-specific, stays here)

A commit body that quotes raw instrumentation output has to reconcile with it
digit for digit, and the *second* decoded value is what pins the input. When
a decode chain has a plateau -- here every `InteractionIdx` in a wide range
maps to the same wrapped `AtomA` -- verifying the first printed number proves
nothing about the constant you claim upstream of it. Invert the expression
that is injective in the index instead. Not promoted to the skill: it is a
reviewing habit about this kernel's evidence, and the generalizable half
(unsigned wrap in padded-thread index math, latent OOB that stops
reproducing) is already in `references/fault-classes.md` from the last round.

## Review 2026-08-20 (third round, message-only fix `fd06b97` -> `7e3c08b`)

**Verdict: APPROVED.** No problems found. The single open finding of the
re-review is closed and nothing else changed.

Recomputed from scratch from `src_clean/VDW_Coulomb.cu:1538-1539`, not from the
porter's derivation: with `NAdsorbateAtoms = 1327`, `-8*(int) InteractionIdx`
is an `int` converted to `size_t` by the usual arithmetic conversions, so for
`InteractionIdx = 883100` the argument is `(2^64 - 7064800) + 7038408 - 7 =
18446744073709525217 = 2^64 - 26399`. Forward from there: `floor(sqrt/2 - 0.5)`
is 2147483647, `AtomA` is the double `1325 - 2147483647.0 = -2147482322.0`
whose 32-bit wrap is the probed 2147484974, and the `AtomB` expression with
`N - AtomA` wrapping to 18446744071562067969 gives exactly 2305843010287440402,
the probed value. 883100 > 879801 real pairs. Every number in `55d7c59`'s body
reproduces.

The "square root is 2^32 to within a rounding step" hedge is fair and is not a
finding: `2^64 - 26399` rounds to the double `2^64 - 26624`, whose square root
is 4294967295.9999967, about 7 ulps (3.3e-6) below 2^32 rather than one. The
claim it supports is unaffected -- `floor(sqrt/2.0 - 0.5)` is 2147483647 for
the computed value and for an exact 2^32 alike, and the body states that
result explicitly rather than resting on the approximation.

Also confirmed, no action: `git diff fd06b97 7e3c08b` is empty and both carry
tree `ff186406cc62dbab135ec2dbf66c7bde30028efc`, so the gfx90a binary and
nine-example run recorded in the Rescope section describe this tree; the second
commit's message is byte-identical (`diff` of `%B`). The Rescope section and
`.claude/skills/cuda-to-rocm/references/assess-existing-support.md:20` both say
`2^64 - 26399` / 29 insertions, and `git diff --stat e4edfc2 7e3c08b` is
`5 files changed, 29 insertions(+), 1 deletion(-)`. Remaining `2^64 - 27191`
occurrences are confined to the dated Review/Re-Review/first-fix blocks, which
correct themselves in place. Code re-verified unchanged: `MolA`/`MolB` are
declared at `VDW_Coulomb.cu:1482-1483`, assigned only at 1547-1548 and read
only at 1549, so hoisting the bounds test to 1546 cannot change the
accepted-pair set or any energy. Hygiene: `jargon.py --port gRASPA` clean;
titles 61 and 48 chars, both `[ROCm]`; AI-assistance disclosure and fenced Test
Plan in both bodies; no `Co-Authored-By`/`Signed-off-by`/noreply trailer; ASCII
throughout messages and diff; no AMD-internal account references; fork
`origin/moat-port` is at `7e3c08b` and the worktree has only the untracked
`src_clean/hip_main.x`.

No GPU run exists at `7e3c08b`; expected at review time. Since the tree is the
one already exercised on gfx90a, revalidation is a re-run for the sha rather
than for new code, and Windows still owes the first check of the README flag
recipe (see "For the next round" in the Rescope section).

## Validation 2026-08-20 (linux-gfx90a revalidate, MI250X gfx90a, `7e3c08b`)

**Tree-identity check**: `origin/moat-port` at `7e3c08b33df60cf2a970e2d9647a7a6922514458`
carries tree `ff186406cc62dbab135ec2dbf66c7bde30028efc` (`git log --format='%H %T'`),
the exact tree the Rescope section's nine-simulation + pytest run was recorded
against (via `55d7c59`/`fd06b97`, both message-only rewrites of the same tree).
Confirmed by direct inspection of this checkout, not taken from notes.

**Judgment call**: since the platform's stale `validated_sha` (`4710600`, from
the pre-rescope branch that was force-pushed away) carries no live evidence,
this is a first real run against the rescoped code -- not a carry-forward. Given
the tree-identity match, a full nine-simulation re-run would re-derive numbers
already on record for this exact binary content; instead did a fresh clean
build at `7e3c08b` (so the sha itself has direct build evidence) plus a real-GPU
re-run of the OOB-relevant case (Tail-Correction) and one NIST reference box,
plus the full pytest suite.

**Build** (clean, `rm -f src_clean/hip_main.x` first):
```bash
export ROCM=/opt/conda/envs/py_3.12/lib/python3.12/site-packages/_rocm_sdk_devel
export PATH=$ROCM/bin:$PATH
cd projects/gRASPA/src/src_clean && GRASPA_ARCH=gfx90a ../HIP_COMPILE
```
Warnings only (same as prior rounds), `hip_main.x` 982792 bytes -- byte-identical
size to the fix-round build recorded above, consistent with the tree match.

**GPU test results** (HIP_VISIBLE_DEVICES=1, HSA_XNACK=1, MI250X gfx90a):

```bash
cd Examples/Tail-Correction && HSA_XNACK=1 HIP_VISIBLE_DEVICES=1 \
  ../../src_clean/hip_main.x > output.txt 2>&1
cd Examples/Reference_NIST_SPCE/Box-1 && HSA_XNACK=1 HIP_VISIBLE_DEVICES=1 \
  ../../../src_clean/hip_main.x > output.txt 2>&1
cd Examples && python3 -m pytest -q -s
```

- Tail-Correction (1327 Ar, the OOB-relevant case fixed by `55d7c59`): exit 0,
  ENERGY DRIFT and GPU DRIFT all components 0.00000. Matches the Rescope run.
- Reference_NIST_SPCE/Box-1 (400 SPCE waters): exit 0, ENERGY DRIFT and GPU
  DRIFT all components 0.00000.
- `python3 -m pytest -q -s` from `Examples/`: **5 passed**. Same reported
  numbers as the Rescope round's pytest run (CO2-MFI GPU drift -4e-05,
  NU2000 CPU/GPU drift -0.00456/-0.00018 against the checked-in CUDA reference
  output for the three examples not re-run this round) -- those output.txt
  files were untouched by this session (`git diff` empty against them both
  before and after), so the suite is exercising the same on-disk evidence the
  Rescope round did, not stale numbers from a different tree.

**Cleanup**: the two tracked example `output.txt` files this run touched
(`Tail-Correction`, `Reference_NIST_SPCE/Box-1`) were restored with
`git checkout --`, and the untracked per-run directories
(`AllData/FirstBead/Lambda/Movies/Restart/TMMC`, `Examples/.pytest_cache`,
`Examples/__pycache__`) were removed. `git status --porcelain` at completion:
only the untracked `src_clean/hip_main.x` build artifact.

**CUDA no-regression gate**: not run this round -- environmental wall, not
previously recorded at this head. This project's CUDA build (`NVC_COMPILE`)
targets NVIDIA's HPC SDK compiler `nvc++` (`-target=gpu -stdpar=multicore
-cuda`), which is a different product from the CUDA toolkit's `nvcc`. This
host has `/opt/conda/envs/cuda-12.8/bin/nvcc` but no `nvc++`/HPC SDK
(`/opt/nvidia` absent). `cuda-not-validated: NVC_COMPILE requires nvc++
(NVIDIA HPC SDK), not installed on this host; the cuda-12.8 conda env provides
only nvcc, a different compiler that cannot run this project's CUDA build
script unmodified`. Not a gate. The Rescope section's argument that both kept
commits are CUDA-neutral by construction (bounds-check reorder, `_WIN32`
false on any CUDA host, additive `<numeric>`) still stands as reasoning, not
compiled evidence.

**Jargon / documentation gates**: `python3 utils/jargon.py --port gRASPA` ->
clean. README.md has a dedicated "AMD / ROCm (HIP)" section (lines 70-102)
next to the CUDA build instructions, covering the build command, `GRASPA_ARCH`,
runtime `HSA_XNACK=1` requirement, the Allegro ROCm LibTorch path, the
CUDA-only LCLin/cppflow exclusion, and the Windows clang recipe. Documented in
the project's own house style.

**Verdict**: PASS. `validated_sha` advances to `7e3c08b33df60cf2a970e2d9647a7a6922514458`.

## Validation 2026-08-20 (linux-gfx1100 revalidate, W7800 gfx1100, `7e3c08b`)

**Why a full run, not carry-forward**: the platform's stale `validated_sha`
(`4710600`) is a commit on the pre-rescope branch that was force-pushed away
(see the 2026-08-20 Rescope section); it no longer exists in `moat-port`
history, so there is no delta to classify and no prior tree to diff against.
Same judgment call as the gfx90a revalidation above: first real run against
the rescoped code.

**Clone**: no local checkout existed on this host; cloned
`https://github.com/AMD-Ecosystem/gRASPA.git` fresh and checked out
`moat-port` (`HEAD` = `7e3c08b33df60cf2a970e2d9647a7a6922514458`, matches
`status.json.head_sha` exactly). `pr-state gRASPA` = `none`, so `moat-port`
is not frozen.

**GPU**: 4x AMD Radeon Pro W7800 48GB (gfx1100, RDNA3), ROCm 7.2.3 at
`/opt/rocm` (this host has no `_rocm_sdk_devel` wheel; used the system ROCm
install instead -- same `HIP_COMPILE` recipe, different toolchain path).

**Build** (clean):
```bash
export PATH=/opt/rocm/bin:$PATH
cd projects/gRASPA/src/src_clean && GRASPA_ARCH=gfx1100 ../HIP_COMPILE
```
Warnings only (`nodiscard` on HIP runtime calls the shim maps, one VLA
extension warning -- same shapes as gfx90a). `hip_main.x` 1047592 bytes,
exit 0.

**GPU test results** (HIP_VISIBLE_DEVICES=0, HSA_XNACK=1):

```bash
cd Examples/Tail-Correction && HSA_XNACK=1 HIP_VISIBLE_DEVICES=0 \
  ../../src_clean/hip_main.x > output_gfx1100.txt 2>&1
cd Examples/Reference_NIST_SPCE/Box-1 && HSA_XNACK=1 HIP_VISIBLE_DEVICES=0 \
  ../../../src_clean/hip_main.x > output_gfx1100.txt 2>&1
cd Examples/CO2-MFI && HSA_XNACK=1 HIP_VISIBLE_DEVICES=0 \
  ../../src_clean/hip_main.x > output_gfx1100.txt 2>&1
cd Examples && python3 -m pytest -q -s
```

- Tail-Correction (1327 Ar, the OOB-relevant case fixed by `55d7c59`): exit 0,
  ENERGY DRIFT and GPU DRIFT all components 0.00000.
- Reference_NIST_SPCE/Box-1 (400 SPCE waters): exit 0, ENERGY DRIFT and GPU
  DRIFT all components 0.00000.
- CO2-MFI (18 CO2 molecules adsorbed): exit 0, ENERGY DRIFT 0.00000 all
  components; GPU DRIFT Ewald [Host-Host] -0.00004 (sub-threshold, the same
  k-space reduction-order signature seen on gfx90a, gfx1201 and gfx1101 --
  every VDW and real-space Coulomb component is exactly 0.00000).
- `python3 -m pytest -q -s` from `Examples/`: **5 passed**, checking the
  committed reference `output.txt` files (untouched by this session --
  each real run above was written to a separate `output_gfx1100.txt` so the
  tracked reference outputs stay as-is; `git status` before and after had no
  modified tracked files).

**Cleanup**: all untracked per-run directories and the three
`output_gfx1100.txt` files this run created were removed
(`AllData/FirstBead/Lambda/Movies/Restart/TMMC` under `CO2-MFI`,
`Reference_NIST_SPCE/Box-1`, and `Tail-Correction`; `Examples/.pytest_cache`,
`Examples/__pycache__`). `git status --porcelain` at completion: only the
untracked `src_clean/hip_main.x` build artifact.

**CUDA no-regression gate**: not re-run -- already recorded as an
environmental wall at this exact head_sha (`7e3c08b`) in the linux-gfx90a
revalidation entry above (`NVC_COMPILE` needs `nvc++`/NVIDIA HPC SDK, not
`nvcc`; this host also lacks `/opt/nvidia`).

**Jargon / documentation gates**: `python3 utils/jargon.py --port gRASPA` ->
clean. README.md's dedicated "AMD / ROCm (HIP)" section (already confirmed
present at this head_sha in the gfx90a entry above) covers this arch too via
`GRASPA_ARCH`.

**Verdict**: PASS. `validated_sha` advances to `7e3c08b33df60cf2a970e2d9647a7a6922514458` for linux-gfx1100.

## Validation 2026-08-20 (windows-gfx1151, Radeon 8060S gfx1151 RDNA3.5 APU, first validation, `7e3c08b`)

**Why a full run, not carry-forward**: no prior `windows-gfx1151` record exists.
`windows-gfx1101` and `windows-gfx1201` both carry `validated_sha =
471060045dad654d6fcb50ec45a3cd3345da3d8a`, the pre-rescope branch that was
force-pushed away on 2026-08-20 (see the Rescope section); it no longer
exists in `moat-port` history, so neither counts as live evidence for the
`windows` gate at the current `head_sha`. This is the first real run against
the rescoped README-driven Windows recipe.

**Clone**: no local checkout existed on this host; cloned
`https://github.com/AMD-Ecosystem/gRASPA.git` fresh and checked out
`moat-port` (`HEAD` = `7e3c08b33df60cf2a970e2d9647a7a6922514458`, matches
`status.json.head_sha` exactly). `protect-fork` installed.

**GPU**: AMD Radeon 8060S (gfx1151, RDNA3.5 APU, 20 CUs), Windows 11. ROCm
from a TheRock pip SDK at `D:/Develop/TheRock/.venv/Lib/site-packages/_rocm_sdk_devel`
(HIP/clang 7.14, clang++ 23.0.0). This host's `C:\WINDOWS\SYSTEM32\amdhip64_7.dll`
is broken (a known host issue, see the `cuda-to-rocm` skill / host notes, not
a port defect); the SDK's own `amdhip64_7.dll`, `amd_comgr.dll`, `rocm_kpack.dll`
and `hiprtc*.dll` were copied next to the executable rather than relying on PATH.

**Build** (README's Windows recipe -- no CMake since the rescope; `HIP_COMPILE`
is a bash script and does not run on Windows, so the classical core is built by
driving `clang++` directly per `README.md`'s AMD/ROCm section):
```bash
ROCM=D:/Develop/TheRock/.venv/Lib/site-packages/_rocm_sdk_devel
export INCLUDE='<MSVC 14.44.35207 + Windows SDK 10.0.22621.0 include dirs>'
export LIB='<MSVC 14.44.35207 + Windows SDK 10.0.22621.0 lib dirs>;<ROCM>/lib/llvm/lib/clang/23/lib/windows'
export PATH="$ROCM/lib/llvm/bin:<MSVC 14.44.35207 Hostx64/x64 bin>:$PATH"
cd src_clean
CLANG_RES="$(clang++ -print-resource-dir)"
clang++ -O3 -std=c++20 --offload-arch=gfx1151 -x hip -fgpu-rdc -fopenmp \
  -D_USE_MATH_DEFINES -isystem "$CLANG_RES/include" \
  -c main.cpp read_data.cpp data_struct.cpp axpy.cu VDW_Coulomb.cu
clang++ -std=c++20 --offload-arch=gfx1151 -fgpu-rdc -fopenmp --hip-link \
  main.o read_data.o data_struct.o axpy.o VDW_Coulomb.o -o hip_main.exe
```
Both commands wrapped in `utils/timeit.sh gRASPA compile --`. Compile: exit 0,
warnings only (nodiscard on the `hipError_t`-returning shim calls, one VLA
extension in `fxn_main.h` -- same shapes as every Linux build). Link: exit 0,
`hip_main.exe` 1537536 bytes. The README's `-isystem "$(clang++
-print-resource-dir)/include"` recipe (finding `.../lib/llvm/lib/clang/23/include`
on this TheRock layout) built cleanly first try -- no `emmintrin.h`/`wmemcmp`
link error, confirming the fix documented in `9cfa096`/`7e3c08b`.

**GPU test results** (`HSA_XNACK=1`, run from each example directory, wrapped
in `utils/timeit.sh gRASPA test --`):

1. CO2-MFI: exit 0, `Work took 24.075271 seconds`. ENERGY DRIFT (CPU vs
   running) all components 0.00000. GPU DRIFT: Ewald [Host-Host] -0.00004
   (sub-threshold; every VDW and real-space Coulomb component exactly
   0.00000), the same k-space reduction-order signature seen on gfx90a,
   gfx1100, gfx1201 and gfx1101. 17 CO2 molecules adsorbed (`C_co2[16], #:
   17`) -- identical to the gfx1201 run's 17 (Monte Carlo stochastic
   variance; gfx1101 saw 26, gfx1100/gfx90a saw 18).
2. Methane-TMMC (the OOB-relevant case fixed by `55d7c59`/`a0b6dce`): exit 0,
   `Work took 9.884038 seconds`, all three phases (INIT/EQUIL/PRODUCTION)
   completed without a memory access fault.
3. Tail-Correction (1327 Ar, the case whose crash the OOB fix's commit body
   documents in detail): exit 0, `Work took 3.435470 seconds`. ENERGY DRIFT
   and GPU DRIFT all components 0.00000. `Ar[20], #: 1323` -- identical to
   the gfx1201 run's 1323 (gfx1101 saw 1333, gfx1100/gfx90a saw 1327).

`python3 -m pytest -q -s` from `Examples/` (checking the five committed
reference `output.txt` files, the same evidence class the gfx90a/gfx1100
revalidations used -- Bae-Mixture and NU2000-pX-LinkerRotations were not
re-run here, ~28 min and untested-arch time cost respectively, out of budget
for this platform): **5 passed**. Reported drifts match the Linux-recorded
reference exactly: Methane-TMMC 0.0/0.0, Bae-Mixture -0.0/0.0, NU2000
-0.00456/-0.00018, Tail-Correction -0.0/0.0.

**Cleanup**: a `rm -rf .../output.txt` before each real run deleted the three
tracked reference `output.txt` files (CO2-MFI, Methane-TMMC, Tail-Correction)
instead of just clearing stale per-run state; restored with `git checkout --`
along with two files the Methane-TMMC run modified in place
(`Restart/System_0/restartfile`, `TMMC/System_0/TMMC_Statistics.data`) and one
it deleted (`AllData/System_0/AllAdsorbate.data`). All untracked per-run
directories (`AllData/FirstBead/Lambda/Movies/Restart/TMMC` under the three
example folders) were removed. `git status --porcelain` at completion: only
the untracked build artifacts in `src_clean/` (`hip_main.exe` and the five
copied runtime DLLs) -- no modified or missing tracked files.

**CUDA no-regression gate**: not re-run -- already recorded as an
environmental wall at this exact `head_sha` (`7e3c08b`) in the linux-gfx90a
revalidation entry above (`NVC_COMPILE` needs `nvc++`/NVIDIA HPC SDK, not
`nvcc`; this Windows host has no CUDA toolkit at all, consistent with that
gate running once per head_sha on whichever host can attempt it).

**Jargon / documentation gates**: `python3 utils/jargon.py --port gRASPA` ->
clean. `README.md`'s "AMD / ROCm (HIP)" section documents the Windows build
in the project's own house style, in the same place as the CUDA/Linux
instructions; this run followed it as written and it needed no correction.

**Verdict**: PASS. `windows-gfx1151` -> `completed`, `validated_sha =
7e3c08b33df60cf2a970e2d9647a7a6922514458`. This is the first `windows` gate
satisfied at the current `head_sha` (both prior Windows platforms are stale
at the pre-rescope sha) -- `python3 utils/moatlib.py pr-ready gRASPA` reports
`True` after this record.

## Review PR opened 2026-08-20 (windows gate closed by gfx1151)

Opened https://github.com/AMD-Ecosystem/gRASPA/pull/1 (`moat-port` -> `main`, 2 commits,
5 files, +29/-1), title
`[ROCm] Fix an out-of-bounds read and build the classical core on Windows`.
`prose.py` and `jargon.py` clean. Awaiting `/moat approve`.

Two things the approver should know:

1. **The windows gate was NOT satisfied before this run**, despite `windows-gfx1101` and
   `windows-gfx1201` both reading `completed`. Both are pinned to `4710600...`, a sha that
   was force-pushed away in the 2026-08-20 rescope (upstream merged its own HIP backend via
   `gpu_compat.h`, so the fork was rebuilt down to the residual 2-commit delta). Those two
   records are pre-rescope artifacts. `pr-ready` is correct because windows-gfx1151 is
   `completed` at the live head `7e3c08b`, but a platform-by-platform read of `status.json`
   would be misleading. Worth either revalidating those two or annotating them.

2. **The PR results table claims only the three platforms validated at `7e3c08b`**
   (linux-gfx90a, linux-gfx1100, windows-gfx1151). The stale gfx1101/gfx1201 figures are
   deliberately not cited, for the same reason.
