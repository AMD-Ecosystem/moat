## Validation policy

- A port is validated only when the project's real test suite builds and passes on a real AMD GPU for the target arch, with no regression in non-GPU tests.
- Coverage is expressed as GATES -- wave64, wave32, windows -- and a gate is satisfied by ANY arch carrying that attribute. No arch leads: whichever host picks the project up ports it, and the rest validate the same branch independently and in parallel.
- A validation should not introduce a NEW fork commit for anything non-essential; if your arch needs no code change, leave the branch untouched. When a fix IS genuinely necessary (e.g. configurable arch), put it in a NEW commit on top -- NEVER amend a commit any arch has validated: amending orphans its `validated_sha`, the regression guard can no longer classify the delta, and every passed arch is forced back to a full revalidation. A new commit is cheap by comparison: `advance_head` carries validation forward automatically for documentation-only, comment/format-only and CI-config deltas, and a `codeobj_diff` binary-equivalence check can carry forward the rest.
- A CPU-only docker build (image `rocm/dev-ubuntu-24.04:7.2.4-complete`) proves the code compiles and links under ROCm. It cannot observe any Fault-class bug above, since no GPU runs, so it is never a validation gate. Do NOT wire it into the fork's GitHub Actions: a workflow that proves nothing and emails failure noise at whoever watches the fork is pure cost. (Not because of revalidation churn -- the regression guard deliberately classifies CI-config deltas as inert, so a yml commit moves the fork HEAD without forcing revalidation. The rule stands because the workflows gate nothing.) Disable Actions on the fork instead; run a CPU-only docker build locally if you want a manual compile check.
- gfx90a and gfx942 are CDNA (wave64); the gfx11xx/gfx12xx parts are RDNA (wave32). A change that passes on one width can still fail on the other via the warp-size class, which is why each arch validates on its own hardware rather than inheriting a result.
- PR-prep gate -- nvcc CUDA-build check: before opening any PR whose claim is "the CUDA/NVIDIA build is unchanged", compile the CUDA path (`USE_HIP=OFF`) with nvcc on this GPU-less host (nvcc from the dedicated conda env `/opt/conda/envs/cuda-12.8/bin/nvcc`, created if missing with `conda create -y -n cuda-12.8 -c nvidia cuda-toolkit=12.8` -- the full `cuda-toolkit`, not compiler-only `cuda-nvcc`, because the check must LINK; host gcc 13 works. ALWAYS pin the arch, e.g. `-DCMAKE_CUDA_ARCHITECTURES=80`: `native` autodetection on a host with no NVIDIA GPU silently degrades to an ancient arch, and `atomicAdd(double*)` "no instance of overloaded function" is that fingerprint, not a real failure). nvcc compiles without an NVIDIA GPU, so this is available on a ROCm-only host, and GPU validation on AMD cannot see a broken CUDA path -- "notes.md says the CUDA path is preserved" is NOT the same as having compiled it. State it honestly in the PR ("compile-checked with nvcc, not run"). REQUIRED when the port adds a CUDA/HIP-split backend (e.g. `Foo_cuda.cpp` vs `Foo_hip.cpp`): that CUDA file is never compiled during HIP-only porting and can be badly broken. The check must reach the LINK stage (build one real target/demo, not compile-only of a single TU) -- the undefined-reference class (e.g. explicit instantiations that do not match real call-site signatures) only surfaces at link. If the fix is HIP-binary-equivalent (codeobj_diff identical), it carries forward with no GPU re-validation.
  - Setup: point `-I` at the project's deps (vcpkg include dir, plus any CUDA-Samples headers it uses -- `helper_cuda.h`, `helper_math.h`, `helper_string.h`) and select the project's non-MATLAB/Python build macro. For a Thrust/CUB project also install `cuda-cccl`: on CUDA 13.x they ship there under `include/cccl/{thrust,cub}` (nvcc finds them automatically, but a host-compiler OpenMP-backend check needs that path on `-I` explicitly, and stdgpu's CMake wants `THRUST_INCLUDE_DIR` pointed at it).
  - For a header-only or template library the changed headers only compile when instantiated: CMake-configure the CUDA backend to generate its config headers, then `nvcc -c` a small TU that `template class`-instantiates the affected containers, rather than compiling headers alone.
  - The class this catches that a HIP-only build cannot: an unconditional device-header include reaching host translation units. Used on 8 projects in one six-week window; it caught a template-shadow regression in Velvet and an stdgpu regression of exactly that include class. (stdgpu, SCAMP, cuSZ, mahout, lc0, cuPDLPx, TIGRE, Velvet)
  - Also REQUIRED when the port moves `__device__` function BODIES into a header (a workaround reached for when a porter does not know HIP has `-fgpu-rdc`), because a host TU that reaches that header then has to parse device code. HEonGPU shipped that break past two review rounds and a full 20/20 GPU validation; the nvcc check found it in one run, together with a cuRAND include the compat header put in front of the CUDA runtime headers for one of the library's four host TUs. See "Headers, includes and build" in references/fault-classes.md for both. (HEonGPU)
  - A dependency that FetchContent-clones with full history can eat the whole budget (HEonGPU: RMM pulls NVIDIA/cccl). Do not conclude the check is unaffordable: read the pinned sha out of the dependency's own version file (rapids-cmake keeps it in `rapids-cmake/cpm/versions.json`), `git fetch --depth 1` that one commit, and point the build at the checkout. For a rapids-cmake project set BOTH `-DCPM_<Name>_SOURCE=` and `-DFETCHCONTENT_SOURCE_DIR_<NAME>=`, since CPM reads the first and plain FetchContent the second -- and mind that the two spell the name DIFFERENTLY. CPM looks up `CPM_${CPM_ARGS_NAME}_SOURCE` (`CPM_0.40.0.cmake:686`), i.e. the package name exactly as `CPMAddPackage` received it and case-sensitive, while CMake upper-cases the name for `FETCHCONTENT_SOURCE_DIR_<NAME>`. So on a lower-case package (`spdlog`, `fmt`) `-DCPM_SPDLOG_SOURCE=` is silently ignored and the full clone happens anyway -- the exact wall the recipe exists to avoid. Read the real spelling out of the dependency's own declaration: for rapids-cmake that is the key in `rapids-cmake/cpm/versions.json` (the same file the pinned sha comes from), otherwise the first argument of the project's `CPMAddPackage` call. It happened to be moot for HEonGPU only because rapids-cmake names the package `CCCL`. Same commit as the pin, so no version drift, and the check becomes a repeatable script. (HEonGPU)
## Platforms

A platform is `<os>-<gfx>`, and the set is open: whatever GPU your host reports is a
platform, and the gates it satisfies follow from its name -- the wavefront width from the
architecture family, `windows` from the OS. Nothing needs adding anywhere first. The only
thing that must be known in advance is the wavefront width of a family, because guessing
it wrong mis-sizes shared memory silently; an unrecognised family is refused with the
one-line fix (`[wave]` in `config/arches.toml`).

Ones seen so far, for orientation rather than as a roster:

- gfx90a: MI200-class CDNA2, wavefront 64. Satisfies wave64.
- gfx942: MI300-class CDNA3, wavefront 64. Additive evidence alongside gfx90a; exercises fp8 paths gfx90a cannot.
- gfx1100: RDNA3 (Radeon), wavefront 32. Watch warp-size assumptions and RDNA occupancy.
- gfx1201: RDNA4 (RX 9070 XT), wavefront 32. On Windows it satisfies wave32 and windows together; the same GPU on Linux satisfies wave32 alone.
- gfx1101, gfx1151: wavefront 32. Records already made against them still satisfy gates -- a validation does not stop being true because a machine changed.

## Windows: use TheRock ROCm, not the Windows HIP SDK

**Build and run against a full ROCm distribution from TheRock (its PyTorch wheels and the
venv around them), never the Windows HIP SDK.** The HIP SDK is a compiler-and-headers
package: its runtime and library set are narrower than a full ROCm, and a build that picks
up its `amdhip64` runtime -- or mixes its libraries with TheRock's -- fails at RUNTIME in
ways that look exactly like a broken port. Symptoms are misleading and widespread rather
than localized: a process exiting 127, "DLL"/"cannot load"/"image not found", or
`hipErrorLaunchFailure` (719) on the FIRST kernel launch.

Triage that class as an environment fault, not a GPU or port fault. Fix the environment
once -- confirm which ROCm the process actually loaded -- and do NOT rebuild in response to
a DLL-load error; rebuilding a correct binary against a broken runtime is the single
biggest time sink on Windows. A port is not "broken on Windows" until it has been run
against a full ROCm distribution.

## Windows: PATH does not beat System32 -- copy the runtime DLLs next to the executable

Building against TheRock is not enough, because on Windows the *loader*, not the linker,
picks the runtime. `amdhip64_7.dll` and `amd_comgr*.dll` also ship in the system
Adrenalin driver under `C:\WINDOWS\System32`, and the loader searches the executable's
own directory, then System32, and only then `PATH`. Putting TheRock's `bin` on `PATH` --
which is what a build environment script naturally does -- therefore does **not** win: a
binary linked entirely against TheRock still runs on the driver's copy.

When the driver's copy is a different vintage than the device bitcode the build used, the
runtime's internal blit/transfer kernels fail to JIT-link (`undefined hidden symbol:
__ockl_dm_init_v1`, `__amd_streamOpsWrite`) and the device transfer manager is never
created. The visible symptom is narrow and easy to misread as a hardware limitation:
`hipMemGetInfo` returns `hipErrorInvalidValue` with `free=0 total=0` while
`hipDeviceTotalMem` still works, and kernels may hang or the process may crash at exit.

The collision is structural, not a broken driver, and updating either side will not clear
it. Measured on the gfx1151 host 2026-08-13: System32's `amdhip64_7.dll` is "amdhip64 **7.2**
Runtime" (10.0.3679.0) paired with `amd_comgr_3.dll` (Build Type: Driver), while TheRock's
is "amdhip64 **7.14** Runtime" paired with `amd_comgr0713.dll` (Build Type: TheRock). The
`_7` suffix is a *major*-version SONAME, so every ROCm 7.x maps onto the same filename, and
the Adrenalin driver ships `amdhip64.dll`, `amdhip64_6.dll` and `amdhip64_7.dll` together.
Note TheRock renames its comgr (`amd_comgr0713`) so that one does not collide, but does not
rename `amdhip64_7`. Known upstream: ROCm/TheRock issues 2019 and 4755, and llama.cpp 17429
is the same fault.

Fix, and make it part of the run procedure rather than the build: copy TheRock's
`amdhip64_7.dll`, `amd_comgr*.dll` and `rocm_kpack.dll` from
`<venv>/Lib/site-packages/_rocm_sdk_core/bin/` into the directory holding every test
executable before running the suite. `rocm_kpack.dll` is a transitive dependency of
`amdhip64_7.dll` (confirmed with `dumpbin /dependents`, run under
`MSYS2_ARG_CONV_EXCL="*"`); copying only the first two leaves the loader failing on a DLL
name that appears nowhere in the build or link lines. Python/PyTorch
processes escape this without help because `rocm_sdk.preload_libraries()` does
`ctypes.CDLL(<absolute path>)` before anything else resolves -- which is why a torch
extension can work on a host where a native CMake project fails.

Two traps when triaging this:

- A minimal reproducer that contains **no device code** does not reproduce it. The failure
  needs the runtime to load a code object, so a ten-line program that only calls
  `hipMemGetInfo` succeeds against the very driver that fails the real binary. Such a
  program looks like clean isolation and proves nothing. Reproduce against a binary that
  actually launches a kernel.
- The API that reports the error is often not the API that caused it. Code in the shape
  `someCall(...); CHECK(hipGetLastError());` reports the *sticky* error, so the blamed
  line may be innocent. Check the call's own return value before naming a culprit.

Do not work around this in the port. Sizing a memory pool from
`hipDeviceProp_t::totalGlobalMem` when `hipMemGetInfo` fails is a source change to shared
code that buys nothing once the right DLLs are in place.

Related, on an integrated APU: `hipMemGetInfo` reports the whole HUMA pool (tens of GB),
so a constructor that eagerly allocates a fraction of *reported* device memory can ask for
far more than is usable and hang or fail. Suspect that shape when an APU validation stalls
at first allocation.

The quieter version of the same shape is a *slowdown*, not a stall, and it is easy to
misread as a port defect. Measured porting HEonGPU on windows-gfx1151 (Radeon 8060S,
72 GB unified): its default memory pool asks for 90% of reported device memory, which on
an APU is 90% of the whole machine (~65 GB). The reservation succeeds, but context
creation goes 1.0 s -> 10.5 s and a heavy test 21 s -> 53 s. An initial pool at or below
10% of memory, or a fixed 1-2 GB, matches no pool at all; 50% is already ~2x slower than
10%, so there is no safe middle fraction. Before blaming the kernels when an APU run is
uniformly ~2x slow, check whether the library reserves a *fraction of device memory* at
startup.

The fix belongs to the application, not the port: pass the library's own pool
configuration (HEonGPU: `MemoryPoolConfig::initial_device_fraction` /
`initial_device_bytes` into `context->generate()`). Do not change a shared default that is
correct for discrete cards, and do not special-case APUs in shared code -- document the
sizing where the project documents its defaults.

## Windows: "missing" HIP CMake packages are almost always a half-expanded rocm-sdk-devel

A CMake-HIP project (`find_package(hip/hiprand/rocthrust)`, `enable_language(HIP)`
with `CMAKE_HIP_COMPILER` = TheRock's `clang-cl.exe`) can hit failures with nothing to
do with the port's own code. Found porting HEonGPU (a pure-CMake math library, not a
torch extension) to windows-gfx1151; not yet seen on a torch-extension port because
those consume HIP through torch's own build glue, not CMake's native HIP language.

**A broken install, NOT a missing wheel. Diagnose before you work around.** An earlier
version of this entry claimed TheRock's Windows wheels ship no `hip`/`hiprand`/
`rocthrust` CMake packages and told you to hand-write shims. That was wrong, and it cost
a full validation round. TheRock ships all of them; the local install had failed to
expand. Corrected 2026-08-13 on the same host that produced the false claim.

`rocm-sdk-devel` is distributed as a `_devel.tar` that expands on first use, and most of
its entries are relative **symlinks** into the sibling runtime packages. Creating those
needs `SeCreateSymbolicLinkPrivilege` -- Developer Mode on, or an elevated shell. Without
it the expansion still "succeeds": the verbatim/hardlinkable files land under `bin/`, and
`cmake/`, `include/`, `lib/`, `libexec/`, `share/` are left **empty**. That is what an
empty `lib/cmake/hip/` actually means. TheRock's own `docs/development/windows_support.md`
states the requirement ("Symlink support is recommended. If symlink support is not
enabled, enable developer mode and/or grant your account the 'Create symbolic links'
permission").

Worse, it does not self-heal: the expander deletes the tarball as its **last** step, and
`get_devel_root()` then short-circuits on "`__init__.py` exists and no tarball". So
**re-running `rocm-sdk init` does nothing** -- it believes it finished.

Diagnose with the SDK's own check, not by eyeballing directories:

```
python -m rocm_sdk test          # healthy: "OK"; broken here was 1 failure + 12 errors
```

Put the venv's `Scripts/` on PATH first, or `testCLIUsesDevelRootPath` errors spuriously
(it shells out to a bare `hipconfig`).

Repair, from an **elevated** shell -- all four packages pinned to one version, because the
devel tree symlinks into its siblings and cannot be mixed:

```
python -m pip uninstall -y rocm rocm-sdk-core rocm-sdk-devel rocm-sdk-libraries-<gfx>
# then DELETE the leftover _rocm_sdk_* trees by hand: pip removes only files listed in
# each RECORD, and an expanded devel tree is not in any RECORD
python -m pip install --index-url https://rocm.nightlies.amd.com/v2/<gfx>/ \
    "rocm==$V" "rocm-sdk-core==$V" "rocm-sdk-devel==$V" "rocm-sdk-libraries-<gfx>==$V"
python -m rocm_sdk init && python -m rocm_sdk test
```

A healthy tree then has `lib/cmake/{hip,hip-lang,hiprand,rocprim,rocthrust}/*-config.cmake`,
rocThrust's ~731 headers under `include/thrust/`, and `bin/hipcc.exe`. Point the build at
`python -m rocm_sdk path --root`. Note the per-arch index (`/v2/<gfx>/`) carries release
families the general `whl-multi-arch` index has already moved past, which is how you pin a
version matching the rest of the fleet.

**Never hand-author CMake configs into the SDK tree.** A session on this host wrote a
`hip-lang-config.cmake` stub directly into `site-packages/_rocm_sdk_core/lib/cmake/`; two
months later another session found it, saw a plausible dated file, and recorded it as
"real and reusable, not mine". Fabricated files in a package directory are indistinguishable
from shipped ones -- `pip uninstall` leaves them behind, since they are in no RECORD, so
they even survive a reinstall. If a shim is genuinely unavoidable, put it in scratch space
and pass `-D<name>_DIR=`; check any suspect file against the owning package's RECORD
(`cut -d, -f1 <dist-info>/RECORD | grep lib/cmake`) before believing it.

**Two independent bugs in CMake's own (not ROCm's) HIP-language support for Windows +
clang-cl**, CMake 3.31:
1. `Platform/Windows-MSVC.cmake`'s MSVC-version-detection fallback chain checks
   `CMAKE_{C,CXX,Fortran,CUDA}_SIMULATE_VERSION` but has **no `HIP` branch**. This is
   invisible in a real project (`LANGUAGES C CXX HIP ...`) because `CXX` is processed
   first and its `SIMULATE_VERSION` covers HIP's evaluation too -- but CMake's OWN
   internal ABI-detection scratch project enables *only* HIP, hits every branch unset,
   and fails with `MSVC compiler version not detected properly` inside a `try_compile`
   you did not write. `-DCMAKE_HIP_COMPILER_FORCED=1` "fixes" this by skipping ABI
   detection, but that ALSO skips whatever normally seeds `-x hip` into the HIP compile
   rule -- `.cu`/`.hip` sources then silently compile as plain C++ (`-TP`, no `-x hip`)
   and every device intrinsic (`threadIdx`, `__syncthreads`, ...) is "undeclared
   identifier", which reads exactly like a much worse problem than it is. Real fix: keep
   `CMAKE_HIP_COMPILER_FORCED=1` (still needed to dodge the detection bug) AND add
   `-x hip` explicitly to `CMAKE_HIP_FLAGS` (it lands after the rule's fixed `-TP` on
   the command line, and the last language-selecting flag wins).
2. The `MSVC_RUNTIME_LIBRARY` target property has no entry for the `HIP` language in
   this CMake version at all (`Help/prop_tgt/MSVC_RUNTIME_LIBRARY.rst` documents C,
   CXX, CUDA, OBJC, OBJCXX, Fortran only) -- `CMake Error ... MSVC_RUNTIME_LIBRARY
   value 'MultiThreadedDLL' not known for this HIP compiler` at generate time, on the
   first HIP target CMake evaluates. Not a CMP0091 policy issue (tried OLD and NEW,
   both fail identically) -- the module-level compatibility table
   (`CMAKE_HIP_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_*`, set by
   `Windows-Clang.cmake`'s `__windows_compiler_clang(HIP)`) exists but the property's
   internal consumer does not route HIP through it. Fix: `-DCMAKE_MSVC_RUNTIME_LIBRARY=""`
   (empty, not merely unset) disables the property mechanism -- but understand that it
   disables it for **every** language, and nothing then supplies a CRT flag at all: no
   `/MD`, no `-D_DLL`, in `CMAKE_C_FLAGS`, `CMAKE_CXX_FLAGS` or `CMAKE_HIP_FLAGS` (read
   `CMAKE_CXX_FLAGS_RELEASE` out of `CMakeCache.txt` and confirm; an earlier session on
   this same host recorded that `Windows-Clang.cmake` still bakes `-D_DLL -D_MT` into the
   `_INIT` values, and that is simply not true for CMake 3.31 + clang-cl). clang-cl's
   no-flag default is the STATIC CRT, so every object you build gets `MT_StaticRelease`
   while conda-forge/vcpkg dependencies are `MD_DynamicRelease`, and lld-link then refuses
   the link with `/failifmismatch: mismatch detected for 'RuntimeLibrary'` naming one of
   your objects and one of theirs. Pass the CRT flag yourself to all three languages:
   `-DCMAKE_HIP_FLAGS="-x hip /MD ..."`, `-DCMAKE_CXX_FLAGS="... -EHsc -MD"`,
   `-DCMAKE_C_FLAGS="... -MD"`. Fixing only one language just moves the mismatch to the
   next pair, so read the two object names in each error and keep going until they agree.
   Note that `-Dgtest_force_shared_crt=ON` does NOT solve this for a FetchContent
   googletest: googletest only rewrites an existing `/MD` into `/MT`, so where no CRT flag
   exists there is nothing for the option to rewrite. (HEonGPU)
3. **clang-cl does not link the compiler-rt builtins.** Host code using 128-bit integer
   arithmetic -- ordinary in crypto and modular-arithmetic libraries -- links fine on
   Linux (libgcc/compiler-rt is implicit) and fails on Windows with
   `lld-link: error: undefined symbol: __udivti3` (or `__umodti3`, `__divti3`). The
   library ships with the toolchain: add `<rocm>/lib/llvm/lib/clang/<ver>/lib/windows` to
   `LIB` and `clang_rt.builtins-x86_64.lib` to `CMAKE_EXE_LINKER_FLAGS` and
   `CMAKE_SHARED_LINKER_FLAGS`. This is toolchain plumbing, not a port defect -- do not
   put it in the fork. (HEonGPU)

**A leading-slash MSVC flag passed from Git Bash is mangled into a path.** `-DCMAKE_CXX_FLAGS="/DWIN32 ..."`
reaches the compiler as `C:/Program Files/Git/DWIN32` (MSYS argument conversion), and the
build fails with `clang-cl: error: no such file or directory: 'C:/Program'`. Use the dash
spellings, which clang-cl accepts identically: `-MD`, `-EHsc`, `-DWIN32`. Do not reach for
`MSYS_NO_PATHCONV=1`/`MSYS2_ARG_CONV_EXCL='*'` as a blanket escape -- it is inherited by
every child process, so a project script that runs `git -C /d/path/...` inside the
configure step stops resolving its own paths and fails with a much more confusing error.
(HEonGPU)

**A Linux-only host API can be the entire Windows blocker in an otherwise portable port**,
and it will not surface until the first Windows build, however many Linux architectures
have already validated. The recurring set is `<sys/sysinfo.h>` + `sysinfo()` (replace with
`GlobalMemoryStatusEx()`/`MEMORYSTATUSEX::ullAvailPhys` under `#ifdef _WIN32`, leaving the
POSIX branch untouched), the BSD typedefs `u_int32_t`/`u_int64_t` (the Microsoft headers
have only the standard `uint32_t` spellings, which are the same types on Linux, so just
change them), and GNU-style `-g`/`-O3` appended to `CMAKE_<LANG>_FLAGS_*` (unused
arguments for clang-cl, which already receives `/Zi` and `/O2 /Ob2`, and *fatal* in any
subproject that compiles with warnings as errors -- googletest does; guard them with
`if(NOT MSVC)`). Grep for the whole set at once before the first Windows build rather than
discovering them one failed compile at a time. (HEonGPU)

**Windows is LLP64, so `unsigned long` is 32 bits -- and the dangerous instances are
inside third-party C APIs, not in the port's own declarations.** Grepping the project for
`long` finds nothing when the truncation lives in GMP's `mpz_*_ui` family, NTL's
`conv(ZZ, long)`/`to_long`, or any other library whose "unsigned integer" entry points are
typed `unsigned long`. On Linux those carry 64 bits and the code is correct; on Windows
every value above 2^32 is silently cut to its low half, with no warning, because the
implicit conversion is legal. Crypto, big-integer and modular-arithmetic libraries are
saturated with this shape.

What makes it expensive is that the symptom looks like a GPU or codegen fault. In HEonGPU
the moduli reaching `mpz_mul_ui` were 30-55 bits, so every CRT constant was built for the
wrong modulus chain: fifteen test binaries returned wrong numbers on `windows-gfx1151`
while the identical source passed on three Linux architectures, one of them the same
wavefront width. Nothing threw, nothing crashed, and the results were bit-for-bit
reproducible run to run.

Two things make it cheap to find instead. First, when Windows alone computes wrong values,
grep for the `_ui`/`_si`/`long` entry points of every C library on the host path *before*
suspecting the device: `mpz_.*_ui`, `to_long`, `static_cast<long>`, `%lu`. Second,
bisect host versus device with a ten-line probe rather than by reading kernels -- replicate
the library's host arithmetic in a standalone `clang-cl` program and check it against
Python or `__uint128_t` truth, and run the same operations in a trivial kernel. Both coming
back clean is what proves the fault is in the surrounding host plumbing.

The fix is data-model-independent, not `#ifdef`-ed, so the LP64 result stays bit identical:
`mpz_import(rop, 1, -1, sizeof(uint64_t), 0, 0, &value)` then `mpz_mul`/`mpz_mod`/
`mpz_fdiv_q` instead of the `_ui` call, and `NTL::ZZFromBytes`/`NTL::BytesFromZZ` over the
8-byte little-endian representation instead of `conv(ZZ, long)`/`to_long`. Note that
`mpz_set_ui(x, 1)` and other small literals are fine -- only values that can exceed 2^32
matter. (HEonGPU)

**Acquiring a Windows dev package neither TheRock nor vcpkg ships, when the usual
mirrors are unreachable.** GMP/OpenSSL/ZLIB had no reachable source on this network
(`ftpmirror.gnu.org`/`ftp.gnu.org`/`gmplib.org` all timed out; `github.com`, `pypi.org`,
`conda.anaconda.org` did not -- check which specific domains are blocked before
concluding the network is down). conda-forge ships MSVC-toolchain win-64 builds of
common C/C++ libraries (compatible with clang-cl, since both are MS-ABI); fetch one
directly with `curl` against `api.anaconda.org`'s redirect, no `conda`/`vcpkg`
installation required: `.conda` files are zip archives containing `pkg-*.tar.zst`,
and Windows's built-in `tar.exe` extracts `.zst` natively. A library conda-forge does
not ship for `win-64` either (NTL had none, checked explicitly via
`api.anaconda.org/package/conda-forge/<name>`'s platform list) may still have its own
historical Windows/MSVC fallback worth finding before assuming a from-scratch port is
needed: NTL's `dosify`/`ResetFeatures`/`mach_desc.win`/`NTL_WINPACK` (see HEonGPU
notes.md) turned what looked like "write a Windows build system for a 76-file C++
library" into copying four pre-existing pieces and compiling directly with clang-cl.

**A `core.autocrlf` line-ending mismatch can make an in-tree `git apply` of a
committed patch fail with a real (non-whitespace) hunk mismatch**, on Windows only,
if the patch file and the tree it applies to were checked out under different
`core.autocrlf` values at different times in the same working copy (e.g. because
something upstream of the patch step -- like `git submodule update --init`, invoked
from inside a build script -- ran before a config change and something else ran
after). Both sides being CRLF (or both LF) is what matters, not which value. Symptom:
`patch does not apply` at a real hunk, not just a "trailing whitespace" warning.
Fix locally without touching the tracked patch: `git config core.autocrlf false`
(both in the repo and in the affected submodules) then force every affected file to
be re-materialized from the index (`rm <file>; git checkout -- <file>` -- a plain
`git checkout -- .` does not always rewrite a file whose working-tree bytes already
happen to match under the stale convention).

## Windows: static initializers in TheRock's DLLs may never run

**A C++ test that gates on `torch::cuda::is_available()` can fail on Windows against a
port that works perfectly.** TheRock's PyTorch Windows build links `torch_hip.dll` with
`lld-link`, which emits no `.CRT$XCU` section; `_DllMainCRTStartup`'s CRT init pointer is
NULL, `_initterm` never runs, and so `REGISTER_CUDA_HOOKS`'s static initializers never
execute. `CUDAHooksRegistry` in `torch_cpu.dll` stays empty, `getCUDAHooks()` returns a
no-op stub, and `is_available()` answers false on a machine whose GPU is fine.

The tell is the split: Python `import torch` reports devices correctly
(`torch._C._cuda_getDeviceCount() > 0`), because the ctypes path does not go through the
registry, while the standalone C++ executable says there is no GPU. Kernel tests that do
not call `is_available()` first pass on the same run. If those three things are true
together, this is the cause and the port is not at fault.

Do not read it as "the project does not work on Windows" -- that mistake cost
LichtFeld-Studio a wrongly-suggested gate waiver, on a port where 320 of 914 tests were
passing on real gfx1101 hardware. It is a defect in a third-party build, so it is a
deferred bug report (`therock-windows-lld-link-crt-xcu`, registered against
LichtFeld-Studio -- `deferred.py list` finds it across refs) plus a workaround, not a
property of the platform. Worth trying, none of them yet tested: `/WHOLEARCHIVE` or
a forced reference into the hooks translation unit so the linker cannot drop it; calling
`_initterm` on the DLL's CRT section directly; spawning a thread after load, since the
TLS callbacks that do exist fire on `DLL_THREAD_ATTACH` and `DLL_PROCESS_ATTACH` calls
`DisableThreadLibraryCalls`.

## Two GPUs visible to one process can crash the runtime

On ROCm 7.14 a process that could see a mixed RDNA3 + RDNA4 pair crashed in the HIP
runtime before any kernel ran. It presents as a launch-time failure of a port that is
correct, so it is easy to spend an afternoon on the wrong suspect. If a machine has
more than one GPU and they are not the same architecture, pin `HIP_VISIBLE_DEVICES`
to exactly one per process and see whether the fault survives.

Do not carry an index in a script. Which index a card holds depends on what is
installed in that machine, so a pinned `HIP_VISIBLE_DEVICES=1` copied from older
notes silently selects a different card, or none. Read the device list at the time
you use it (`rocminfo`, `hipInfo`).

## One architecture gets wrong numbers while the others pass

A clean build that produces wrong results on exactly one architecture -- an iterative
solver, an LM/Newton fit, an FP regression head -- is usually floating-point
accumulation divergence rather than a port bug, and RDNA3.5 (gfx1151) is where it has
shown up. Record the error magnitude and stop rather than chasing it deep: the
comparison that matters is against the other architectures, not against a fix.

## A low-CU integrated GPU can outrun a fixed upstream test timeout without any fault

A 20-CU integrated APU (gfx1151) can be genuinely, correctly ~40x slower than the
datacenter/desktop cards the rest of the fleet validates on for the single heaviest
kernel sequence in a suite, and a project's own `TIMEOUT` on `add_test`/
`gtest_discover_tests` is usually sized against those faster cards. Found porting
HEonGPU: `ctest` reported `TFHE_Gate_Boots` as `***Timeout` at the project's hardcoded
30-second per-test limit (`test/CMakeLists.txt`, identical on the CUDA path), while
`linux-gfx942` finishes the entire 20-test suite in 13-15s. Running the same
executable directly, outside ctest's harness, showed it was not hung or wrong -- it
passed in 112.9s, `[ PASSED ]` from gtest's own assertions.

Diagnose by running the flagged executable directly before concluding anything: `rc=0`
plus a `[ PASSED ]`/correct-value line after the harness's cutoff means "too slow for
this budget," not "broken." A hang or wrong answer looks different -- no completion
line ever appears, or it appears with wrong values.

Getting a clean harness-level pass count (worth doing when the dispatch's bar is an
exact N/N matching Linux) may need a throwaway local bump of the test's own `TIMEOUT`
property: ctest's own `--timeout <seconds>` CLI flag does **not** override an
explicit per-test `TIMEOUT` set via `set_tests_properties`/`gtest_discover_tests` --
confirmed, `ctest --timeout 180` against a `TIMEOUT 30` property still printed
`***Timeout 30.07 sec`. Edit the property, reconfigure (cheap -- CMake only
regenerates the test files, no recompilation, if no source changed), rebuild only the
one affected test target so `gtest_discover_tests`'s post-build discovery step
re-runs and picks up the new property, run ctest, then `git checkout --` the file
before completion and verify `git status --porcelain` is empty. This is not a port
fix and must not ship -- it exists only to prove the harness-level count, the same way
the CUDA-arch pin is a throwaway for the CUDA no-regression gate.

## Diagnosing a suspected AMD fault before escalating

Two patterns that each cost a deep investigation before the real cause was found.

- **A "data-dependent, later-data-corrupts-earlier, per-tile" corruption signature is the fingerprint of a REPRODUCER bug, not a codegen fault.** cuSZ chased a suspected miscompile to a BLOCKED state and an IR bisect; the actual cause was the test input -- `np.arange(..., dtype=float32) * (python float)` promotes to float64, so `.tofile()` wrote 8 bytes per element and the tool read the stream as f32. Validate the byte width and dtype of any binary test input before escalating to an ISA bisect or a ROCm bug report. (cuSZ)
- **Triangulate single- against double-precision before blaming the wavefront.** When a warp-collective rewrite shows SP divergence, run a second GPU variant and compare both to the DP oracle. If both GPU variants diverge from DP identically at the same positions, and the DP path is bit-identical to the CPU oracle, it is floating-point reassociation at a comparison boundary -- not a wave-size fault. SCAMP used this to clear a ~0.5 divergence at 10/8093 positions as a threshold-boundary artifact. (SCAMP)
- **When a probe for an already-fixed defect newly fails on a second arch, suspect the probe's own inputs before the arch.** HEonGPU's Barrett-shift fix (61/62-bit moduli) verified clean on gfx90a; the identical probe strategy on gfx1100 showed 19998/20000 bad at those widths and looked like a wave32 regression the first arch had missed. The probe generated its test moduli as `2^bits - small_offset`, pathologically close to an exact power of two; the library's HOST-side `bit_generator()` computes `bit = (T1)(log2(value) + 1)` in plain `double`, and at 61-62 bits a value within roughly 1024 of a power of two rounds `log2` up to the boundary in double precision, over-counting the bit width by one and mis-sizing the unrelated Barrett `mu` constant -- identical behavior on any host, any back end, not a HIP or wavefront effect. Redrawing the probe's moduli uniformly across each target bit width (not clustered at the boundary) reproduced 0/20000 bad on both arches. Before concluding a fix regressed on a new arch, check whether the SECOND probe run's inputs, not just its outputs, differ from the first. (HEonGPU)
