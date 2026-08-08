## Strategy A: pure CMake (preferred, minimal footprint)

Goal: only `.cu`/`.hip` translation units see the HIP toolchain; host C++ is untouched; the diff stays small.

1. Add one CUDA-to-HIP compat header (e.g. `src/.../cuda_to_hip.h`). On ROCm it aliases the CUDA spellings the project uses to their HIP equivalents and includes the HIP runtime; on NVIDIA it is a no-op include of the CUDA runtime. Everywhere else keep the plain CUDA spelling (`cudaXxx`, `curand*`, `cublas*`). This header is the only file that knows about HIP.

       #pragma once
       #if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
       #include <hip/hip_runtime.h>
       #define cudaMalloc        hipMalloc
       #define cudaFree          hipFree
       #define cudaMemcpy        hipMemcpy
       #define cudaStream_t      hipStream_t
       #define cudaError_t       hipError_t
       #define cudaSuccess       hipSuccess
       // ... only the symbols the project actually uses
       #else
       #include <cuda_runtime.h>
       #endif

   Use hipify's mapping tables as the authoritative cuda->hip name source when adding aliases: `torch/utils/hipify/cuda_to_hip_mappings.py` in a pytorch checkout lists 3000+ symbol mappings.

2. In CMake, gate the language on a HIP option instead of renaming files:

       option(USE_HIP "Build with HIP for AMD GPUs" OFF)
       if(USE_HIP)
         # Do NOT pin CMAKE_HIP_ARCHITECTURES to a literal arch (e.g. gfx90a) here.
         # enable_language(HIP) already does the right thing: it honors an explicit
         # -DCMAKE_HIP_ARCHITECTURES, else auto-detects the host GPU(s) via
         # rocm_agent_enumerator, else errors (FATAL_ERROR "Failed to find a default
         # HIP architecture"). A pin BEFORE this call preempts that auto-detect, so a
         # non-gfx90a user who omits -D silently builds gfx90a objects that fail to
         # load at runtime ("no kernel image"); a pin AFTER it is dead code (the call
         # already resolved the arch or errored). Either way another host still
         # overrides with -DCMAKE_HIP_ARCHITECTURES=<arch> and never edits this file.
         enable_language(HIP)
         set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE HIP)
         set_target_properties(<tgt> PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
       else()
         enable_language(CUDA)
       endif()

   Marking the existing `.cu` files `LANGUAGE HIP` keeps the diff minimal and the NVIDIA build intact. Configure with `-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a` (add `-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++` if CMake does not find it). Because the target reads `${CMAKE_HIP_ARCHITECTURES}`, one commit builds for any AMD target with only `-DCMAKE_HIP_ARCHITECTURES=<arch>` and no source change, so validating on another architecture needs no commit of its own. Pass every architecture you can test at planning time so the first bring-up is right.

   **Outside CMake you write that rule yourself, and the README has to say what you wrote.**
   A meson/Makefile project has no `enable_language(HIP)` to get this right for you, so the
   tempting shapes -- `option('amd_gfx', value: 'gfx90a')`, or `if gfx == ''` then
   `gfx = 'gfx90a'` -- reintroduce exactly the silent default the CMake note above warns
   about: a user on another card builds objects that load nowhere. Give the option an EMPTY
   default, autodetect by taking the first non-`gfx000` line of `rocm_agent_enumerator`, and
   hard-`error()` naming the flag to set when detection yields nothing. Two traps when you
   do: `run_command()` on a `find_program()` result needs meson >= 1.2.0, so guard the
   autodetect branch on the meson version and remember the error path is what pre-1.2.0
   users hit; and you cannot rehearse that error path by hiding the GPU, because
   `rocm_agent_enumerator` reads the sysfs topology and ignores `HIP_VISIBLE_DEVICES` /
   `ROCR_VISIBLE_DEVICES` (and a `find_program` fallback to the absolute
   `/opt/rocm/bin/rocm_agent_enumerator` defeats hiding it from `PATH`). Then re-read the
   README block you wrote about the flag: lc0 dropped its gfx90a fallback in a review round
   and left the README claiming "defaulting to gfx90a" through two further porter rounds
   until a validator's documentation check caught it. Any review round that changes what a
   build option DOES carries a documentation edit with it.

   **Document a clean-environment-safe build: pass `-DCMAKE_PREFIX_PATH=/opt/rocm`.** When the project's CMake calls `find_package(hip)` / `find_package(hipcub)` / `find_package(rocThrust)` (most Strategy A ports do, to link `hip::device`/`hip::host`), CMake locates ROCm's config packages by deriving the `/opt/rocm` prefix from `/opt/rocm/bin` being on `PATH`. Our dev hosts and the gfx90a container have ROCm on `PATH`, so the build "just works" for us; a clean ROCm container (e.g. `rocm/dev-ubuntu`) does not, and the exact documented command fails with `hip_DIR-NOTFOUND` (or hipcub/rocThrust NOTFOUND). Setting `-DCMAKE_HIP_COMPILER` by absolute path is NOT enough (it points the HIP language at the compiler but does not seed the package search), and `ROCM_PATH=/opt/rocm` alone is NOT enough either (both proven experimentally). Always pass `-DCMAKE_PREFIX_PATH=/opt/rocm` (append `;/opt/rocm` when the command already sets a prefix for another dependency, e.g. a vendored gtsam/pangolin install) in BOTH the recipe you run and every documented build block (README, install guide, notes.md), plus a one-line "or put `/opt/rocm/bin` on `PATH`" note. Watch the silent variant: `find_package(hip QUIET)` does not error on a clean container, it disables the GPU path and builds a CPU-only library. Builds that use only `enable_language(HIP)` with no roc/hip `find_package` and no `hip::` link are not affected. (visionaray #53 surfaced this while standing up a clean-container CI job.)

3. Guard genuinely divergent code with `#if defined(USE_HIP)`; keep such guards rare. Dispatch sites that accept either backend use `#if defined(USE_CUDA) || defined(USE_HIP)`.

This is how colmap was ported (PR 4420 plus follow-ups): one compat header, `.cu` marked `LANGUAGE HIP`, a few guarded fixes. PyTorch validated that this isolates HIP: on an MI250 build only the HIP translation units receive `-x hip`; host files are untouched.

### Large libraries: `--offload-compress` to fit the host image under the 2 GiB x86-64 relocation reach

A heavily-templated library (cudf: rocThrust/rocPRIM `DeviceReduce`/`DeviceScan`/`DeviceRadixSort` instantiated over a full type x op matrix) embeds an enormous device code object PER translation unit. The HIP offload bundle is stored UNCOMPRESSED by default, so a single reduce `.o` can reach ~200 MB (~159 MB of which is the `.hip_fatbin`). Linking many such TUs into one shared object can push the HOST image past the +/-2 GiB reach of x86-64 32-bit PC-relative relocations -- the link fails with `R_X86_64_PC32` overflow first, then (if you reach for `-mcmodel=large`, which only fixes data refs) `R_X86_64_TLSGD`/`TLSLD`/`__tls_get_addr` and `.eh_frame` "PC offset too large", neither of which the large code model covers. nvcc fatbins are compact, so the CUDA build never hits this; it is a ROCm-specific build-scale wall, not a HIP-correctness one.

Fix: add `--offload-compress` to the HIP compile options (`target_compile_options(<tgt> PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-compress>)`). A documented ROCm clang HIP flag (7.2.1+: "Compress offload device binaries (HIP only)") that stores the embedded device bundle in the compressed CCOB format; the HIP runtime decompresses it on load. Measured on a cudf rocPRIM reduce TU @ gfx90a: object 207.8 MB -> 52.1 MB (3.99x), `.hip_fatbin` 158.6 MB -> 2.89 MB (54.9x) -- the device ISA matrix is hugely redundant so it compresses ~55x. This let cudf's full reductions + aggregation + groupby + join surface link into ONE ~540 MB libcudf.so with no multi-.so split and no `-mcmodel=large`. For a non-RDC build (no `-fgpu-rdc` device-link step) the per-TU compile is the only place the bundle is formed, so there is no separate device-link to also flag. Prefer this BEFORE splitting a library or capping the template matrix. Also build a single target arch (no multi-arch fatbin) and add `-ffunction-sections -fdata-sections` + `--gc-sections` to prune unreachable host instantiations. (cudf.)

### Relocatable device code: the HIP device link never sees inside a `.a`

A project that declares `__device__`/`__constant__` globals or device function-pointer
tables in HEADERS and defines them in a separate `.cu` needs relocatable device code on
HIP, exactly as it needs `CUDA_SEPARABLE_COMPILATION` on NVIDIA. The symptom is a link
error naming a `__device__` member, e.g.
`lld: error: undefined hidden symbol: MetricTaker::operator()(thrust::tuple<...>) const`.
Turning on `-fgpu-rdc` plus `HIP_SEPARABLE_COMPILATION ON` is necessary but often not
sufficient, and the missing piece is not documented anywhere obvious:

**The HIP device link only sees device objects passed DIRECTLY on the link line. Objects
inside a static archive are invisible to it.** A project built as many small
`add_library(... STATIC)` targets therefore performs a device link that finds no device
objects at all, and the failure mutates into undefined `__hip_fatbin_*` /
`__hip_gpubin_handle_*` symbols rather than a clear diagnostic. nvcc's device linker does
pull device code out of archives, so the CUDA build never shows this.

Fix by restructuring the HIP build only: make each GooFit-style component an
`add_library(<name> OBJECT ...)` with `POSITION_INDEPENDENT_CODE ON`, record the target in
a global property, and gather every object library into ONE shared library that performs a
single device link spanning all translation units. That resolves every cross-TU
`__device__`/`__constant__` global and every device function-pointer table at once. Keep it
inside `if(GOOFIT_DEVICE STREQUAL HIP)`-style guards so the CUDA and CPU builds keep their
archives. Note also that IPO/LTO cannot be combined with HIP relocatable device code, so
disable `INTERPROCEDURAL_OPTIMIZATION` on that path.

Two smaller companions from the same port: mark BOTH `.cu` files and any `.cpp` that
includes rocThrust as `LANGUAGE HIP` (CMake otherwise drops the `.cu` objects from the
device link once the project enabled HIP rather than CUDA), and delete host globals that
take the address of a `__device__` function (`device_function_ptr p = device_Step;` at
namespace scope) -- that is invalid on HIP and dead code on CUDA. (GooFit; first seen on
RXMesh, where only the `-fgpu-rdc` half was needed.)

## The shim-header method: a port with zero source edits

The plain Strategy A shape adds `#include "cuda_to_hip.h"` to every `.cu`/`.cuh` that needs
it. That works, but it touches every source file, and the diff a maintainer reviews is
dominated by include lines rather than by the port.

Better, where the project includes CUDA headers by name: put SHADOW HEADERS named after the
CUDA ones in a `hip_compat/` directory, and prepend that directory to the include path on
the HIP build only:

    target_include_directories(<tgt> BEFORE PRIVATE ${CMAKE_SOURCE_DIR}/hip_compat)

`hip_compat/cuda_runtime.h` includes `hip/hip_runtime.h` and defines the aliases;
`cublas_v2.h` maps to hipBLAS; `cufft.h`/`cufftXt.h` to hipFFT. CUDA-only headers the port
does not need (`mma.h`, the CUDA-samples `helper_cuda.h`/`helper_functions.h`) become empty
stubs. The CUDA build never sees the directory, so its include resolution is untouched.

TurboFNO went from 38 edited source files (53 files changed) to **zero source edits** (21
files changed), leaving the `.cu`/`.cuh` tree byte-identical to upstream. That was verified
behaviour-preserving: bit-identical GPU results, and all 10 device code objects byte-identical
across two architectures.

Prefer this when the project includes CUDA headers by name. It does not apply where sources
use CUDA symbols without including a CUDA header, or where the build already injects a
compat header with `-include`.

**A sibling-relative include cannot be shadowed by the include path, and that is the one
real hole in this method.** A quoted `#include "math.cuh"` inside `vendor/include/pkg/`
resolves against the includING FILE's own directory before any `-I` is consulted, so a
`hip_compat/pkg/math.cuh` never wins no matter how early the directory is prepended. The
fix costs one line and keeps the source tree untouched: force-include the shadow copy, and
let the header's own upstream include guard turn the original into a no-op.

    target_compile_options(<tgt> PRIVATE
      $<$<COMPILE_LANGUAGE:HIP>:-include${HIP_COMPAT_DIR}/pkg/math.cuh>)

Give the shadow file the SAME include guard macro as the header it replaces; that is what
makes the later sibling-relative include collapse. Note the cost this pulls in: a
force-included header creates no dependency edge, so wipe the build directory after editing
one (see fault-classes, "Headers, includes and build"). Diagnose it by reading the "In file
included from" chain in the first error and checking whether the includer sits in the same
directory as the header it names. (Quest)

## Build hygiene

- **Do not pin `--offload-arch` or `CMAKE_HIP_ARCHITECTURES` in the committed build.** Pass
  the arch at build time so another architecture reuses the recipe with only
  `--offload-arch=<arch>`; `enable_language(HIP)` auto-detects. A pinned arch is the single
  most common reason a port builds on the machine it was written on and nowhere else.
  (TurboFNO, LC-framework)
- **hipify-perl, when that is the mechanism rather than CMake HIP language:** run it
  synchronously -- `-inplace` in a backgrounded or `&&`-chained loop silently skips files --
  and always re-grep the whole tree for `cudaMalloc|cudaSuccess|include <cub|include <cuda.h`
  before compiling. Un-hipified files surface as "undeclared identifier cudaMalloc". Note
  that hipify prepends `#include "hip/hip_runtime.h"`, which breaks a g++ CPU reference
  build, so build that from a separate non-hipified copy. (LC-framework)
