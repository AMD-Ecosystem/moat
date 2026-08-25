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

## Vendored CUDA-samples helpers: keep them off the AMD build entirely

Many CUDA projects vendor `helper_cuda.h` / `helper_math.h` / `helper_string.h` from a
pre-2017 CUDA samples drop. Those copies carry "Please refer to the NVIDIA end user license
agreement (EULA)", so they are proprietary-marked files, and the rule is that nothing AMD
builds may compile, execute, or take them as a build input -- and that the port's diff must
not modify them either (editing one makes the diff a derivative work of it). Never delete a
licence or notice file to get there.

Two moves, both needed:

1. **Write your own substitute.** Grep first: in practice the only symbol a project uses from
   `helper_cuda.h` is `checkCudaErrors` (plvs: 36 sites; Velvet: 9). Write it INDEPENDENTLY --
   own name and namespace, a non-template function taking `hipError_t` exactly, your own
   message format, no `DEVICE_RESET` dance. Do not paraphrase the sample's `check<T>()`
   template or its `"CUDA error at %s:%d code=%d(%s)"` string; a HIP block that mirrors them
   is the same derivative problem one indirection away. Velvet's first port did exactly that
   and it had to be rewritten.
2. **Keep the directory off the HIP include path**, so an accidental include fails to resolve
   instead of compiling. Either wrap the entry (`if(NOT USE_HIP) target_include_directories(...
   Velvet/External/cuda) endif()`, Velvet) or put a shadow header in a `hip_compat/` directory
   prepended only on the HIP path (`include_directories(BEFORE ...)` inside `if(USE_HIP)`,
   plvs/alien). The CUDA build's include resolution is untouched either way.

**Prove it; do not infer it from the source.** `#if !defined(USE_HIP)` guards around the
include lines are not evidence the file is not a build input, and CMake's own
`HIP.includecache` lists textual `#include` lines it never resolved (it does not evaluate
`#if`), so it will show `helper_cuda.h` on a build that never opened it. What settles it is a
per-TU include trace with the build's real flags, taken from
`build/CMakeFiles/<tgt>.dir/flags.make`:

    hipcc -x hip -fsyntax-only -H <the build's exact flags> <tu> 2>&1 | grep -c helper_cuda

Expect 0 for EVERY translation unit, plus an empty `grep -rl "<vendor dir>" build/`.

**The CUDA path can usually be cleaned too.** NVIDIA republished all three headers under
BSD 3-Clause in `NVIDIA/cuda-samples` under `Common/`, and they are API supersets of the old
drops, so replacing the vendored copies with those releases verbatim is normally a no-op for
the CUDA build. Verify rather than assume: compile the TUs that include them with `nvcc` (and
the host TUs with the host compiler) at both the base and the new sha and diff the results, and
`-H`-trace one TU to confirm the swapped files are actually opened so the check is not vacuous.
Record the upstream commit and the sha256 of each file you copied in. (plvs, Velvet.)

## Never mark the host `.cpp` files `LANGUAGE HIP` to dodge the flag leak

The tempting shortcut when a mixed CXX/HIP target fails to compile is to mark *every* source
`LANGUAGE HIP` so one set of flags applies uniformly. Do not. That defines `__HIPCC__` for
the whole program, and CUDA projects routinely key host-only behaviour off `__CUDACC__` --
which the port then extends to `__HIPCC__`, correctly, for the device path. The classic
shape is a macro that suppresses default member initializers, because a
`__device__ __constant__` copy of the struct cannot be dynamically initialized:

    #if defined(__CUDACC__) || defined(__HIPCC__)
        #define HOST_INIT(val)
    #else
        #define HOST_INIT(val) = val
    #endif

    struct SimParams { int numSubsteps HOST_INIT(2); glm::vec3 gravity HOST_INIT(...); };

On NVIDIA the `.cpp` files are plain C++ and never see `__CUDACC__`, so the host copy keeps
its defaults. Compile them as HIP and the host copy silently loses every initializer. Velvet
shipped that way from its first commit: the application ran, rendered and reported a healthy
GPU, but the solver read zero substeps, zero iterations and zero gravity, so nothing ever
moved. It survived four platform validations because each used a standalone synthetic kernel
test rather than the real binary. Grep for `__CUDACC__` and `__CUDA_ARCH__` in host headers
before choosing an all-HIP target, and treat a hit as a hard blocker on that shortcut.

The failure the shortcut was dodging has its own fix, above: the HIP compile flags leak
because `hip::hipcub` and `roc::rocthrust` reach `hip::device` transitively
(`roc::rocprim_hip` -> `hip::device` in `rocprim-targets.cmake`), and `hip::device`'s
interface is built by

    function(hip_add_interface_compile_flags TARGET)
      set_property(TARGET ${TARGET} APPEND PROPERTY
        INTERFACE_COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:${_HIP_SHELL}${ARGN}>")
    endfunction()

called with `-x hip` and `--offload-arch=<arch>`. Note the genex is `COMPILE_LANGUAGE:CXX`,
so it hits exactly the plain C++ sources and never the HIP ones -- the opposite of what the
name suggests. Those three packages are header only, so take their include directories and
link `hip::host` (which carries only `__HIP_PLATFORM_AMD__=1` and the `amdhip64` import
library, no compile flags) for the runtime:

    set_source_files_properties(${CU_SOURCES} PROPERTIES LANGUAGE HIP)
    foreach(rocm_target hip::hipcub roc::rocthrust roc::rocprim_hip)
      if(TARGET ${rocm_target})
        get_target_property(inc ${rocm_target} INTERFACE_INCLUDE_DIRECTORIES)
        if(inc)
          target_include_directories(<tgt> SYSTEM PRIVATE ${inc})
        endif()
      endif()
    endforeach()
    target_link_libraries(<tgt> PRIVATE hip::host)

Before splitting, check two things or you trade a silent-zeros bug for a silent-no-op bug:
no header that a `.cpp` includes may contain a `<<<>>>` launch (a launch macro gated on
`__CUDACC__ || __HIPCC__` expands to nothing in the host units, so the launch disappears),
and no variable whose initializer the macro suppresses may be an `inline`/`extern` variable
that both language groups emit (the linker folds one COMDAT and may pick the uninitialized
one). Velvet passed both: launches live only in the two `.cu` files, and the `inline`
`Global::simParams` is included by no `.cu`.

Verify the split landed rather than assuming: the build log should show "Building CXX
object" for the host files (Velvet: 11 CXX against 2 HIP, where all 13 previously said HIP),
the recorded compile line for a host object should carry neither `-x hip` nor
`--offload-arch`, and a temporary `#if defined(__HIPCC__) #error` probe in one host file
should compile clean.

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

## Windows: CMake's HIP link rule mangles MSVC-style flags and paths

On Windows, CMake's HIP link path routes through `hipcc.exe`, which invokes `clang.exe
--driver-mode=g++` for `--hip-link`. That GCC driver then receives CMake's MSVC-style
response file and flags, and misreads them two ways: it treats bare MSVC linker flags
(`/machine:x64`, `/subsystem:console`) as file paths, and it applies GNU backslash-escaping
to the response file, eating every backslash so `D:\path\torch.lib` arrives as
`D:pathtorch.lib`. CMake also injects `-fuse-ld=lld-link`, which conflicts with `--hip-link`.

**This has nothing to do with `-fgpu-rdc`.** A project with no relocatable device code at all
still hits it if it links HIP through CMake on Windows. Do not conclude you are safe because
you are not using RDC -- that mistake was made once already and cost a round.

Fix: interpose a wrapper on the link step, set as `CMAKE_HIP_LINK_EXECUTABLE`. The canonical
copy is `assets/hip_link_win.py` in this skill. It strips `-fuse-ld=*`, converts `-Xlinker`
pairs and bare MSVC `/flags` to `-Wl,` form, and rewrites `-lXXX.lib` to `-Xlinker XXX.lib`
so lld-link resolves them via `LIB`. It is ~60 lines and carries no project-specific state.

**Vendor a copy into the fork** (conventionally `cmake/hip_link_win.py`) rather than
referencing this repository: the project's CMake calls it, and the work is meant to be
upstreamed, so a pull request cannot depend on a file that only exists here. Fix bugs in the
canonical copy first, then re-copy into the forks that carry it.

Guard the override to Windows. Related and separate: the clang-cl `/Fo` bundler bug, where the
CMake HIP compile rule must emit `-o <OBJECT>` instead of `/Fo<OBJECT>` because the driver
hands the same path to `clang-offload-bundler` as both input and output and Windows refuses to
rewrite a memory-mapped file (ROCm/TheRock#5615). A project can need the compile-rule fix, the
link wrapper, or both -- they are independent. (alien, origin of the wrapper; LichtFeld-Studio,
which needs the link half.)
