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

## Install and consume the port, not just build it

A project that ships `install()` rules and a `<Name>Config.cmake` has a second interface the
in-tree tests never touch, and it is routinely left CUDA-only by a port that otherwise works:
the exported config `find_dependency(CUDAToolkit)` unconditionally, the exported link
interface names a HIP imported target with no `find_package` that defines it, a replacement
for a CUDA-only dependency installs its export set without a config file to find it by, and
the shim headers that resolve `cuda_runtime.h` / `curand_kernel.h` in the INSTALLED headers
are neither installed nor on the interface include path. Every one of these is invisible
until someone consumes the installation, so **install to a scratch prefix and build a
five-line consumer project against it as part of the port**. `configure_file(... @ONLY)`
makes the config-template fix a one-liner with no CMakeLists change:

    if("@USE_HIP@")
        find_dependency(hip REQUIRED)
    else()
        find_dependency(CUDAToolkit REQUIRED)
    endif()

Quote the substitution: when the option is not defined (the subproject built standalone) it
expands to nothing, and a bare `if()` is a CMake error while `if("")` is false. Note also
that `cmake --install --prefix` does NOT override a prefix a project baked into its install
rules at configure time; reconfigure with `-DCMAKE_INSTALL_PREFIX` instead of concluding the
install is broken. Write the downstream snippet into the docs only after the consumer
actually configures, builds and runs. (HEonGPU)

## Every optional target the project can build, not just the one you build

`BUILD_TESTS` is on while you work, so the test targets get the full treatment -- source
compiled as HIP, the `USE_HIP` definition, the shim include directory -- and `BUILD_EXAMPLES`
and `BUILD_BENCHMARKS`, which default OFF, quietly do not. A half-converted branch of that
kind is worse than no branch at all, because it reads as tested: on HEonGPU the example
targets linked the HIP runtime but were still compiled as plain C++, so the first project
header pulled in `cuda_runtime.h` and then rocThrust, and the benchmark targets still linked
`CUDA::cudart` unconditionally and failed at configure time. **Turn every option the project
documents ON once and build it**, then run one binary from each group; that is minutes, and
it is the only way the claim in your commit message is true.

### OpenMP on a target you just switched to HIP: two runtimes, one binary

An imported target found for another language only half applies to a HIP target, and both
halves matter. `FindOpenMP.cmake` guards `OpenMP::OpenMP_CXX`'s `INTERFACE_COMPILE_OPTIONS`
with `$<COMPILE_LANGUAGE:CXX>` (it sets `INTERFACE_LINK_OPTIONS` only for Fujitsu and
IntelLLVM, so on GCC or Clang there are none), while `INTERFACE_LINK_LIBRARIES` -- the C++
compiler's OpenMP runtime, `libgomp` under GCC -- is unguarded and reaches the link line of
any target that links it. So on a source switched to `LANGUAGE HIP` the flag disappears and
the GNU runtime does not.

Compiled with no `-fopenmp`, the `#pragma omp` is ignored, the link against `libgomp` is
clean, and the binary runs SERIALLY with `omp_get_thread_num` returning 0 -- no diagnostic
at any stage. That is the default outcome and the one to watch for; the loud
undefined-`__kmpc_*` link error belongs to the different, intermediate state where the
compile got the flag and the link did not. Put `${OpenMP_CXX_FLAGS}` on both
`$<COMPILE_LANGUAGE:HIP>` and `$<LINK_LANGUAGE:HIP>`, written the way `FindOpenMP` itself
does it (`$<$<COMPILE_LANGUAGE:HIP>:SHELL:${OpenMP_CXX_FLAGS}>`), since the variable is a
space-separated string and a compiler whose OpenMP flag is more than one token would
otherwise arrive as a single argument.

**Then remove `OpenMP::OpenMP_CXX` from that branch instead of keeping it as well.** With
both, the binary carries two OpenMP runtimes: `readelf -d` shows `libgomp.so.1` and
`libomp.so` together in DT_NEEDED. It can look fine, because ROCm ships
`lib/llvm/lib/libgomp.so.1` as a symlink to its own `libomp.so`, so while DT_RUNPATH
reaches ROCm first both names resolve to one runtime. Put GNU's `libgomp` ahead of it --
any distro library path does -- and both load. The clang-compiled object's `__kmpc_*` calls
can only come from LLVM's runtime, so it still forks a real team, but `omp_get_thread_num`,
which both runtimes export, now answers from `libgomp`, which knows nothing about that
team. Measured on HEonGPU with a minimal HIP+OpenMP binary linked the same way: four
distinct OS thread ids, and `omp_get_thread_num` reporting `0 0 0 0`. Code that uses the
thread id to pick a per-thread resource then has every thread pick element 0 (in HEonGPU's
multi-stream example, one stream shared by the whole team) while still printing correct
results.

While you are there, drop any `LINKER_LANGUAGE CXX` left on the target from when it was
C++. The reason is not that device code goes missing -- a non-RDC HIP object carries its
own device image and `g++ obj.o -lamdhip64` links and runs it correctly -- but that
`$<LINK_LANGUAGE:HIP>` is false under a CXX link, so the OpenMP link flag above silently
never applies. (HEonGPU)

## Submodules pinned to commits that do not exist

**A port that edits a git submodule in place is lost the moment the clone is deleted, and
the branch it leaves behind cannot even be cloned.** HEonGPU pins GPU-NTT, GPU-FFT and
RNGonGPU as submodules. A porter edited each submodule's working tree, committed the
resulting gitlinks in the parent, and never pushed the submodule commits anywhere. The
gitlinks then named three SHAs that existed in no repository on earth. Two later sessions
each rebuilt the same submodule port from scratch before discovering this, and the second
one saved its reconstruction only as patch files in gitignored scratch space, so it was
lost a third time.

Check for it early: `git ls-tree HEAD <submodule-path>` in the port branch, then
`git submodule update --init`. If the update cannot find the commit, the branch is
unbuildable and no amount of local success proves otherwise.

Forking every submodule is usually not available (fork creation is admin-only, and a chain
of forks has to be re-pointed in `.gitmodules` and kept in sync). The durable shape that
needs no new repositories is to **reset the gitlinks to the real upstream commits and carry
the submodule changes as patch files in the parent repo**, applied by whatever script the
project already runs at configure time:

    git submodule update --init --recursive
    if [ "${1:-OFF}" = "ON" ]; then                       # only for AMD builds
        for name in ...; do
            patch="$here/patches/$name.patch"
            git -C "$here/$name" apply --reverse --check "$patch" 2>/dev/null && continue
            git -C "$here/$name" apply "$patch"
        done
    fi

The reverse-check is what makes repeated configures idempotent. Pass the AMD flag in from
CMake so an NVIDIA build sees the submodules untouched. Add `ignore = dirty` to each entry
in `.gitmodules`: applying the patches permanently dirties those working trees, and without
this every later `git status` reads as an unclean tree and trips the integrity gate, while
a genuine gitlink bump still shows. This also states the honest upstream position -- these
changes belong in the submodules' own repositories, and the patch files are exactly the
diffs to send there. (HEonGPU)

## Windows: a fetched ROCm-DS dependency has to be built static

ROCm-DS libraries -- hipMM (the RMM port) and the rapids_logger it fetches transitively --
are built shared by default and neither one works that way under an MSVC-style toolchain.
Two independent defects, both found porting HEonGPU, both in the dependency rather than in
the port:

- **They export nothing.** `rmm/detail/export.hpp` defines `RMM_EXPORT` as
  `__attribute__((visibility("default")))` under `__GNUC__` and as nothing otherwise (its
  comment says only glibc is supported); rapids_logger's macro is unconditionally the ELF
  attribute with no `#else`. The DLLs themselves build and link, so the failure surfaces
  far downstream at the **first executable link**, as `undefined symbol:
  rmm::cuda_stream_view::...` / `rapids_logger::logger::log(...)` for symbols that are
  plainly present in the import library. If a shared build is genuinely required, the fix
  is `set_target_properties(<tgt> PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)` on each of
  them (measured to link cleanly), but see below for why static is better.
- **A GNU-only linker option reaches an MSVC-style driver.** rapids_logger applies
  `target_link_options(rapids_logger PRIVATE "LINKER:--exclude-libs,libspdlog")` whenever
  `RAPIDS_LOGGER_HIDE_ALL_SPDLOG_SYMBOLS` is ON, which is its default under
  `BUILD_SHARED_LIBS`. CMake's `LINKER:` prefix translates only the *spelling* of the
  driver's flag-passing convention, never the flag itself, so `--exclude-libs libspdlog`
  arrives verbatim; lld-link warns that it does not know `--exclude-libs` and then treats
  the bare word `libspdlog` as an input file name and fails. Generalize this: any GNU/ELF
  linker idiom behind `LINKER:` (`--exclude-libs`, `--version-script`, `-Bsymbolic`,
  `--as-needed`) passes through untranslated and is a Windows build break waiting to
  happen.

Setting `BUILD_SHARED_LIBS` OFF for the directory scope that fetches them disposes of
both at once -- the export macros stop mattering, and rapids_logger's
`cmake_dependent_option` forces its symbol hiding off for a static build -- and it also
avoids a third problem, that Windows has no RPATH: with shared dependencies every
executable needs `rmm.dll`, `rapids_logger.dll`, `spdlog.dll` and `fmt.dll` copied next to
it before it will even start, which breaks `gtest_discover_tests` during the build itself
(exit `0xc0000135` right after a successful link).

Set it as a **normal variable in that directory scope**, not a cache entry. RMM declares
`option(BUILD_SHARED_LIBS "Build RMM shared libraries" ON)` in its own CMakeLists, and an
`option()` writes the cache, so fetching RMM silently flips every *later* fetch in the
project to shared as well -- that is how a project's googletest turns into `gtest.dll`
halfway through a port. Under CMP0077 NEW (any dependency requiring CMake 3.13+) an
`option()` defers to an existing normal variable and creates no cache entry at all, so the
scope-local `set(BUILD_SHARED_LIBS OFF)` keeps both the dependency and everything fetched
after it on the default static path. Guard the whole thing on `MSVC` (true for clang-cl,
and the correct predicate for "MSVC-style link driver") so Linux and the CUDA path are
untouched. (HEonGPU)
