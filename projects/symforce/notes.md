# symforce notes

## Build

The Caspar module (CUDA code generation and execution backend) runtime library can be built standalone for HIP:

```bash
cd symforce/caspar/source/runtime
mkdir build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
make -j$(nproc)
```

For generated Caspar libraries (via the Python codegen pipeline), pass `-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a` to cmake.

## Validation

The runtime library (libcaspar_runtime.a) compiles successfully for gfx90a. Object files contain HIP device code:
```
llvm-objdump --offloading build/CMakeFiles/caspar_runtime.dir/shared_indices.cu.o
# shows: hipv4-amdgcn-amd-amdhsa--gfx90a
```

Full integration testing requires symforce to be installed, which needs modifications to symforce's main CMakeLists.txt to support HIP. The caspar examples (kernel_example, bal, multiple_factors) test GPU execution with PyTorch but cannot run until the main symforce package supports USE_HIP.

## Port details

### HIP cooperative groups gaps

HIP (ROCm 7.2.1) lacks several CUDA cooperative_groups features:
- `cg::reduce` - replaced with manual butterfly shfl_xor reduction
- `cg::labeled_partition` - replaced with match_any + masked butterfly reduction
- `cg::memcpy_async` / `cg::wait` - replaced with synchronous block-strided copy

These replacements are guarded by `#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)` so CUDA behavior is unchanged.

### Wave64 safety

The code uses `tiled_partition<32>` which creates 32-lane tiles. On wave64 (gfx90a), HIP CG supports 32-lane tiles within a 64-lane wavefront. The SumStore two-level reduction and other 32-wide operations work correctly as they operate within the tile, not the full wavefront.

### Gotchas

- HIP's `group_dim()` is not const; had to use `blockIdx.x * blockDim.x` directly for HIP
- `atomicAdd_block` is undefined for HIP; aliased to `atomicAdd` (HIP shared-memory atomics are block-scoped by definition)
- `-ffast-math` causes NaN backward passes on HIP (clang's -fassociative-math reassociates online-softmax/layernorm reductions); use `-ffp-contract=fast -fno-math-errno` instead
- `cg::labeled_partition` butterfly reduction doesn't work for non-contiguous label groups (lanes at arbitrary positions share a label, but XOR only pairs specific distances); use per-lane atomicAdd fallback instead (shared-memory atomics are fast)

## Review 2026-06-05

Re-review after porter fixes. All three issues addressed correctly:

1. **labeled_partition per-lane atomicAdd**: Correct replacement for cg::labeled_partition. The butterfly approach fails for non-contiguous label groups (lanes at arbitrary positions share a label, but XOR only pairs specific distances). Per-lane atomicAdd to shared memory is simple and correct.

2. **atomicAdd_block alias**: Correct. HIP shared-memory atomics are block-scoped by definition, so atomicAdd_block -> atomicAdd is semantically correct.

3. **-ffast-math -> -ffp-contract=fast -fno-math-errno**: Correct per PORTING_GUIDE. clang's -ffast-math enables -fassociative-math which can NaN backward passes; the chosen flags retain FMA contraction without dangerous associativity reordering.

Minor cleanup items for PR-prep (not blockers):
- cuda_to_hip.h:54-104 contains dead code (reduce_max, match_any_mask, labeled_reduce_sum, CG_LABELED_REDUCE_SUM never used)
- Commit body contains "Strategy A" (MOAT vocabulary) -- scrub before upstream PR

Ready for validation.

## Validation linux-gfx90a 2026-06-05

### Build

Runtime library builds successfully for gfx90a:

```bash
cd symforce/caspar/source/runtime
mkdir build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
make -j$(nproc)
```

Verified: libcaspar_runtime.a built successfully. Device code confirmed via:
```
llvm-objdump --offloading build/CMakeFiles/caspar_runtime.dir/shared_indices.cu.o
# shows: hipv4-amdgcn-amd-amdhsa--gfx90a
```

### Test

Attempted to run GPU tests via caspar examples (kernel_example). Installed symforce in editable mode. PyTorch 2.13.0 with ROCm detected GPU correctly (AMD Instinct MI250X).

### Failure

**Generated code templates missing compat header includes**

The Jinja templates that generate .cu files include CUDA headers directly:

- `symforce/caspar/source/templates/caspar_mappings.cu.jinja` line 7:
  ```c++
  #include <cooperative_groups.h>
  #include <cooperative_groups/memcpy_async.h>
  ```

- `symforce/caspar/source/templates/kernel.cu.jinja` lines 4-8:
  ```c++
  #include <cooperative_groups.h>
  #include <cooperative_groups/details/partitioning.h>
  #include <cooperative_groups/memcpy_async.h>
  #include <cooperative_groups/reduce.h>
  #include <cuda_runtime.h>
  ```

When CasparLibrary.generate() is called from Python, it generates .cu files with these raw CUDA includes. On HIP, this fails at compile:

```
fatal error: 'cooperative_groups.h' file not found
```

The `cuda_to_hip.h` compat header IS correctly copied to the generated directory, but the generated .cu files do not include it.

**Fix needed**: Update the templates to conditionally include the compat header:

```c++
#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
#include "cuda_to_hip.h"
#else
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
// ... other CUDA includes
#endif
```

This affects:
- `symforce/caspar/source/templates/caspar_mappings.cu.jinja`
- `symforce/caspar/source/templates/kernel.cu.jinja`

The standalone runtime library compiles because those .cu files already have the compat header included at the top. Generated libraries fail because the templates emit raw CUDA includes.

**Back to porter for template fixes.**

## Template fix 2026-06-05

Fixed the Jinja templates to include `cuda_to_hip.h` instead of raw CUDA headers:

- `caspar_mappings.cu.jinja`: replaced `#include <cooperative_groups.h>` with `#include "cuda_to_hip.h"`
- `caspar_mappings.h.jinja`: replaced `#include <cuda_runtime.h>` with `#include "cuda_to_hip.h"`
- `kernel.cu.jinja`: replaced all cooperative_groups and cuda_runtime includes with `#include "cuda_to_hip.h"`
- `kernel.h.jinja`: replaced `#include <cuda_runtime.h>` with `#include "cuda_to_hip.h"`
- `solver.h.jinja`: replaced `#include <cuda_runtime.h>` with `#include "cuda_to_hip.h"`

Additional fixes:

1. **coalesced_group reduction**: HIP's `coalesced_group` lacks `shfl_xor`, so `FlushSumBlock` and `FlushSumBlockAdd` in memops.cuh now use shared-memory atomics directly (each valid thread does atomicAdd_block) instead of butterfly reduction.

2. **CMakeLists.txt.jinja**: Link `hip::host` and `hip::hipcub` as PRIVATE to avoid propagating HIP compile options to pure-CXX pybind consumers. Added `set_source_files_properties(${CPP_SOURCES} PROPERTIES LANGUAGE HIP)` for pybind files so HIP headers work.

3. **cuda_to_hip.h**: Added missing mappings for `cudaSetDevice`, `cudaGetDevice`, `cudaPointerGetAttributes`.

4. **library.py**: Added `use_hip` and `hip_arch` parameters to `CasparLibrary.compile()`.

Test command:
```bash
HIP_VISIBLE_DEVICES=0 python3 test_symforce_hip.py
# Output: SUCCESS! All tests passed.
```

## Re-validation linux-gfx90a 2026-06-05

Full GPU validation after porter fixes.

### Runtime library build

Standalone runtime library:
```bash
cd symforce/caspar/source/runtime
mkdir build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
make -j$(nproc)
```

Result: libcaspar_runtime.a built successfully with gfx90a device code:
```
llvm-objdump --offloading build/CMakeFiles/caspar_runtime.dir/shared_indices.cu.o
# shows: hipv4-amdgcn-amd-amdhsa--gfx90a
```

### Generated kernel test

Test script that exercises the full code generation pipeline:
1. Creates CasparLibrary with a symbolic kernel
2. Generates code from Jinja templates
3. Compiles with HIP backend (use_hip=True, hip_arch="gfx90a")
4. Executes kernel on GPU with PyTorch tensors
5. Verifies numerical results

```bash
HIP_VISIBLE_DEVICES=0 python3 test_hip_validation.py
```

Output:
```
PyTorch version: 2.13.0a0+gitb5e90ff
CUDA available: True
Device: AMD Instinct MI250X / MI250
Generating kernel code...
Compiling with HIP for gfx90a...
Running GPU test...
Verifying results...
All checks passed!

=== VALIDATION PASSED ===
```

The test validates:
- Jinja templates correctly emit cuda_to_hip.h includes
- Generated code compiles with HIP
- Kernel execution on GPU (cooperative groups, shared memory atomics)
- Correct numerical output (AddSharedSum and WriteIndexed memory patterns)

### Test coverage

The generated kernel exercises:
- ReadShared and ReadUnique memory patterns
- AddSharedSum reduction (uses atomicAdd on shared memory)
- WriteIndexed scatter
- Trigonometric operations (sin/cos)
- Arithmetic operations
- Shared indices lookup

All porter fixes validated on real GPU:
1. Templates include cuda_to_hip.h correctly
2. cudaSetDevice/cudaGetDevice/cudaPointerGetAttributes mappings work
3. FlushSumBlock/FlushSumBlockAdd shared-memory atomics execute correctly
4. HIP target linking and pybind compilation work

VALIDATION PASSED on gfx90a (AMD Instinct MI250X).

## Validation linux-gfx1100 2026-06-05

### Build

Runtime library:
```bash
cd symforce/caspar/source/runtime
mkdir build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100
make -j$(nproc)
```

Result: libcaspar_runtime.a built successfully with gfx1100 device code:
```
llvm-objdump --offloading build/CMakeFiles/caspar_runtime.dir/shared_indices.cu.o
# shows: hipv4-amdgcn-amd-amdhsa--gfx1100
```

### Test

Generated kernel test via test_hip_validation.py:
```bash
HIP_VISIBLE_DEVICES=0 python3 test_hip_validation.py
```

Output:
```
PyTorch version: 2.13.0a0+gitb5e90ff
CUDA available: True
Device: AMD Radeon Pro W7800 48GB
Generating kernel code...
Compiling with HIP for gfx1100...
Running GPU test...
Verifying results...
All checks passed!

=== VALIDATION PASSED ===
```

The test validates:
- Jinja templates correctly emit cuda_to_hip.h includes
- Generated code compiles with HIP for gfx1100
- Kernel execution on GPU (cooperative groups, shared memory atomics)
- Correct numerical output (AddSharedSum and WriteIndexed memory patterns)

All porter fixes from gfx90a validation (templates include cuda_to_hip.h, cudaSetDevice/cudaGetDevice/cudaPointerGetAttributes mappings, FlushSumBlock/FlushSumBlockAdd shared-memory atomics, HIP target linking) work correctly on gfx1100.

VALIDATION PASSED on gfx1100 (AMD Radeon Pro W7800 48GB).

## Validation windows-gfx1201 2026-06-08

### Build

The Python codegen pipeline (CasparLibrary) requires symforce's custom symengine fork (with CopysignNoZero, SignNoZero). On Windows, the symenginepy build chain (symengine C++ -> Cython wrapper) has deep dependency issues (GMP, MSVC debug libs, Cython 0.x/3.x incompatibilities). The GPU port validation is done with a standalone HIP test that directly exercises the runtime memops code paths.

Standalone runtime library for gfx1201 (CMake, no Python):
```
cmake symforce/caspar/source/runtime
  -B runtime_build -G Ninja
  -DCMAKE_C_COMPILER=clang.exe -DCMAKE_CXX_COMPILER=clang++.exe
  -DCMAKE_HIP_COMPILER=clang++.exe
  -DCMAKE_PREFIX_PATH=_rocm_sdk_devel/lib/cmake
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201
cmake --build runtime_build --parallel 24
```
Result: caspar_runtime.lib built successfully. Confirmed gfx1201 device code embedded in test exe via `strings`:
```
hipv4-amdgcn-amd-amdhsa--gfx1201
```

### Test

Standalone HIP test exercising the key HIP-specific code paths from memops.cuh. The test covers:
- FlushSumBlock: shared-memory atomicAdd reduction replacing cg::binary_partition + cg::reduce
- FlushSumBlockAdd: same plus add-to-output
- WriteIdx / ReadIdx: basic device memory read/write
- SumStore: butterfly reduction within tiled_partition<32> replacing cg::reduce

```
HIP_VISIBLE_DEVICES=0 test_caspar_runtime.exe
```

Output:
```
Device: AMD Radeon RX 9070 XT (gcnArchName: gfx1201)
[PASS] Test 1 FlushSumBlock: got 523776.0, expected 523776.0
[PASS] Test 2 FlushSumBlockAdd: block0=(523776.0,1047552.0) block1=(523776.0,1047552.0)
[PASS] Test 3 WriteIdx/ReadIdx: 256 values correct
[PASS] Test 4 SumStore: got 523776.0, expected 523776.0

=== Results: 4 PASSED, 0 FAILED ===
```

VALIDATION PASSED on gfx1201 (AMD Radeon RX 9070 XT, RDNA4, wave32).

### Windows-specific fixes committed (sha ba1b64db)

Three additional Windows build fixes were committed on top of the port:

1. `cmake/rerun_if_needed.py`: `os.path.relpath` returns backslash paths on Windows; git rev-parse requires forward slashes. Fixed by adding `.replace("\\", "/")`.

2. `symforce/caspar/source/runtime/CMakeLists.txt`: Added `CMAKE_CXX_STANDARD 17`. rocPRIM enforces C++17 at compile time; the standalone CMake was missing this.

3. `symforce/caspar/source/templates/buildfiles/CMakeLists.txt.jinja`: Same C++17 fix in the generated library CMakeLists template.

Fork updated to sha ba1b64db. All platforms (linux-gfx90a, linux-gfx1100) validated at d99faf9b; gfx1201 validated at ba1b64db (which is d99faf9b + the Windows build fixes). windows-gfx1101 should also validate at ba1b64db when that GPU is back online.

## Revalidation linux-gfx90a + linux-gfx1100 2026-06-08

Binary-equivalence carry-forward from d99faf9b -> ba1b64db for linux-gfx90a and linux-gfx1100.

Delta: cmake/rerun_if_needed.py path-sep fix (no-op on Linux); set(CMAKE_CXX_STANDARD 17) in runtime/CMakeLists.txt and the CMakeLists.txt.jinja template. The C++17 setting COULD affect device codegen, so full binary comparison was performed (not assumed inert).

Built the caspar runtime at both SHAs for each arch:
```bash
# gfx90a
cmake symforce/caspar/source/runtime -B build_d99_gfx90a -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build build_d99_gfx90a -j16
cmake symforce/caspar/source/runtime -B build_ba1_gfx90a -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build build_ba1_gfx90a -j16

# gfx1100 (compile-only, no GPU on this host)
cmake symforce/caspar/source/runtime -B build_d99_gfx1100 -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build_d99_gfx1100 -j16
cmake symforce/caspar/source/runtime -B build_ba1_gfx1100 -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build_ba1_gfx1100 -j16
```

codeobj_diff verdict: identical for all 3 .cu.o device objects (shared_indices, solver_tools, sort_indices) on both gfx90a and gfx1100. The C++17 standard flag does not change device ISA for the caspar runtime sources. The path-sep fix in rerun_if_needed.py is host-Python only (no GPU code).

Both platforms carried forward to ba1b64db via binary-equiv.

## Revalidation windows-gfx1201 2026-06-16

Binary-equivalence carry-forward from ba1b64db -> 38c73bb3 for windows-gfx1201.

GPU confirmed: AMD Radeon RX 9070 XT (gfx1201), device 0.

Delta: two commits on top of ba1b64db:
- `fcf17203`: AMD copyright/author header in cuda_to_hip.h; HIP build doc in README.md
- `38c73bb3`: standalone test CMakeLists.txt deleted; one comment line removed in FlushSumShared HIP branch; `idx_inner` -> `idx` rename in FlushSumShared CUDA `#else` branch; pybind_array_tools.cc comment added

The memops.cuh rename (`idx_inner` -> `idx`) is in the CUDA `#else` branch (lines 284-303), NOT the HIP branch (lines 260-283). No `.cu` files changed. No HIP device code path changed.

Built caspar runtime at both SHAs for gfx1201 using all-clang + TheRock ROCm SDK:
```
cmake symforce/caspar/source/runtime -B build_old -G Ninja
  -DCMAKE_CXX_COMPILER=clang++.exe -DCMAKE_HIP_COMPILER=clang++.exe
  -DCMAKE_PREFIX_PATH=_rocm_sdk_devel/lib/cmake -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201
# repeat for HEAD sources
```

Extracted gfx1201 ELF device objects via clang-offload-bundler, then compared sections:

- `.text` (ISA): sha256 IDENTICAL for all 3 TUs (shared_indices, solver_tools, sort_indices)
- `.rodata`: IDENTICAL
- `.dynsym` (exported symbols): IDENTICAL
- `.dynstr` (symbol names): IDENTICAL
- `.hipFatBinSegment`: IDENTICAL
- `.hip_fatbin` hash differs only due to debug-info strings (`.debug_str` shifted by the new copyright line in cuda_to_hip.h -- line numbers/strings changed, ISA unchanged)

Carried forward windows-gfx1201 to 38c73bb3 via binary-equiv.

## Validation windows-gfx1101 2026-06-19

### GPU verified

AMD Radeon PRO V710 (gcnArchName: gfx1101, wave32), HIP_VISIBLE_DEVICES=1 (device index 1 in the two-GPU setup; gfx1201 is device 0).

### Build

Built the standalone test executable for gfx1101 using the same CMakeLists.txt from agent_space/symforce_hip_test (reused from gfx1201 validation):

```
cmake B:/develop/moat/agent_space/symforce_hip_test
  -B B:/develop/moat/agent_space/symforce_hip_test/test_build_gfx1101
  -G Ninja
  -DCMAKE_CXX_COMPILER=clang++.exe  (TheRock venv LLVM)
  -DCMAKE_HIP_COMPILER=clang++.exe
  -DCMAKE_PREFIX_PATH=_rocm_sdk_devel/lib/cmake
  -DCMAKE_HIP_ARCHITECTURES=gfx1101
cmake --build ... --parallel 64
```

Result: test_caspar_runtime.exe built successfully. Confirmed gfx1101 device code embedded:
```
hipv4-amdgcn-amd-amdhsa--gfx1101
```

TheRock runtime DLLs (amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc0714.dll, hiprtc-builtins0714.dll) co-located with the exe.

### Test

```
HIP_VISIBLE_DEVICES=1 test_caspar_runtime.exe
```

Output:
```
Device: AMD Radeon PRO V710 (gcnArchName: gfx1101)
[PASS] Test 1 FlushSumBlock: got 523776.0, expected 523776.0
[PASS] Test 2 FlushSumBlockAdd: block0=(523776.0,1047552.0) block1=(523776.0,1047552.0)
[PASS] Test 3 WriteIdx/ReadIdx: 256 values correct
[PASS] Test 4 SumStore: got 523776.0, expected 523776.0

=== Results: 4 PASSED, 0 FAILED ===
```

Numerical results match gfx1201 exactly (same expected values). The HIP-specific code paths (FlushSumBlock shared-memory atomics, SumStore butterfly reduction within tiled_partition<32>, WriteIdx/ReadIdx device memory) all execute correctly on wave32 gfx1101.

VALIDATION PASSED on gfx1101 (AMD Radeon PRO V710, RDNA3, wave32).
