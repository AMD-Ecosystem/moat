## Validation policy

- A port is validated only when the project's real test suite builds and passes on a real AMD GPU for the target arch, with no regression in non-GPU tests.
- Coverage is expressed as GATES -- wave64, wave32, windows -- and a gate is satisfied by ANY arch carrying that attribute. No arch leads: whichever host picks the project up ports it, and the rest validate the same branch independently and in parallel.
- A validation should not introduce a NEW fork commit for anything non-essential; if your arch needs no code change, leave the branch untouched. When a fix IS genuinely necessary (e.g. configurable arch), put it in a NEW commit on top -- NEVER amend a commit any arch has validated: amending orphans its `validated_sha`, the regression guard can no longer classify the delta, and every passed arch is forced back to a full revalidation. A new commit is cheap by comparison: `advance_head` carries validation forward automatically for documentation-only, comment/format-only and CI-config deltas, and a `codeobj_diff` binary-equivalence check can carry forward the rest.
- `codeobj_diff` false-positive class -- `__FILE__` (or any compiler-embedded absolute path) inside a check/log macro that gets forwarded into a template argument (e.g. a `TORCH_CHECK(...)`-style macro fed `__FILE__` directly, as in `CUDA_CHECK` -> `torchCheckMsgImpl<char[N], ...>`) makes the mangled symbol name encode the *length of the build path*, not anything about the code. Comparing two builds done from two different absolute checkout directories (e.g. old sha in `agent_space/Foo-old`, new sha in `projects/Foo/src`) then reports the exported-symbol set as `differ` even when nothing changed -- the two `char[N]` array lengths just differ because the paths differ in length. Diagnostic: demangle the differing symbol (`c++filt`); if the only difference is a `char[N]` literal-length template parameter, it is this artifact, not a real change. Fix: rebuild BOTH shas from the identical absolute path (checkout the old sha in place, build to a separate `--build-base`/output dir, then checkout the new sha back) and re-diff; if it now reports `identical`, the original `differ` was purely a path-length artifact. (CuMesh, linux-gfx1100 revalidation 2026-08-11)
- A CPU-only docker build (image `rocm/dev-ubuntu-24.04:7.2.4-complete`) proves the code compiles and links under ROCm. It cannot observe any Fault-class bug above, since no GPU runs, so it is never a validation gate. Do NOT wire it into the fork's GitHub Actions: a workflow that proves nothing and emails failure noise at whoever watches the fork is pure cost. (Not because of revalidation churn -- the regression guard deliberately classifies CI-config deltas as inert, so a yml commit moves the fork HEAD without forcing revalidation. The rule stands because the workflows gate nothing.) Disable Actions on the fork instead; run a CPU-only docker build locally if you want a manual compile check.
- gfx90a and gfx942 are CDNA (wave64); the gfx11xx/gfx12xx parts are RDNA (wave32). A change that passes on one width can still fail on the other via the warp-size class, which is why each arch validates on its own hardware rather than inheriting a result.
- PR-prep gate -- nvcc CUDA-build check: before opening any PR whose claim is "the CUDA/NVIDIA build is unchanged", compile the CUDA path (`USE_HIP=OFF`) with nvcc on this GPU-less host (nvcc from the dedicated conda env `/opt/conda/envs/cuda-12.8/bin/nvcc`, created if missing with `conda create -y -n cuda-12.8 -c nvidia cuda-toolkit=12.8` -- the full `cuda-toolkit`, not compiler-only `cuda-nvcc`, because the check must LINK; host gcc 13 works. ALWAYS pin the arch, e.g. `-DCMAKE_CUDA_ARCHITECTURES=80`: `native` autodetection on a host with no NVIDIA GPU silently degrades to an ancient arch, and `atomicAdd(double*)` "no instance of overloaded function" is that fingerprint, not a real failure). nvcc compiles without an NVIDIA GPU, so this is available on a ROCm-only host, and GPU validation on AMD cannot see a broken CUDA path -- "notes.md says the CUDA path is preserved" is NOT the same as having compiled it. State it honestly in the PR ("compile-checked with nvcc, not run"). REQUIRED when the port adds a CUDA/HIP-split backend (e.g. `Foo_cuda.cpp` vs `Foo_hip.cpp`): that CUDA file is never compiled during HIP-only porting and can be badly broken. The check must reach the LINK stage (build one real target/demo, not compile-only of a single TU) -- the undefined-reference class (e.g. explicit instantiations that do not match real call-site signatures) only surfaces at link. If the fix is HIP-binary-equivalent (codeobj_diff identical), it carries forward with no GPU re-validation.
  - Setup: point `-I` at the project's deps (vcpkg include dir, plus any CUDA-Samples headers it uses -- `helper_cuda.h`, `helper_math.h`, `helper_string.h`) and select the project's non-MATLAB/Python build macro. For a Thrust/CUB project also install `cuda-cccl`: on CUDA 13.x they ship there under `include/cccl/{thrust,cub}` (nvcc finds them automatically, but a host-compiler OpenMP-backend check needs that path on `-I` explicitly, and stdgpu's CMake wants `THRUST_INCLUDE_DIR` pointed at it).
  - For a header-only or template library the changed headers only compile when instantiated: CMake-configure the CUDA backend to generate its config headers, then `nvcc -c` a small TU that `template class`-instantiates the affected containers, rather than compiling headers alone.
  - The newest toolkit is not the whole check: a host TU that touches `__half` can be ill-formed on every CUDA release the project supports and fine on the one installed here, because 12.2 exposed `__half`'s integer converts to host compilers (11.0 through 12.1 hide them, and a 12.2-or-newer build hides them again if it defines `__CUDA_FP16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__`). When the port touches half-typed host code, also compile that translation unit (or a repro of its template shape) against an older toolkit's headers -- `conda create -n cuda-12.1-hdr -c nvidia cuda-cudart-dev=12.1 cuda-cccl=12.1` is enough for a `g++ -std=c++17 -I<env>/include` check, no nvcc needed. See "A ROCm-only branch inside a shared template still COMPILES on the CUDA path" in fault-classes.md for what it catches. (marian-dev)
  - The class this catches that a HIP-only build cannot: an unconditional device-header include reaching host translation units. Used on 8 projects in one six-week window; it caught a template-shadow regression in Velvet and an stdgpu regression of exactly that include class. (stdgpu, SCAMP, cuSZ, mahout, lc0, cuPDLPx, TIGRE, Velvet)
  - A torch `CUDAExtension` cannot get this gate the same way: `torch/extension.h` pulls in the fleet's ONLY installed PyTorch, a ROCm dev build with no CUDA-flavored counterpart on hand, so the check dies in the ambient install's headers before it can judge the port. See "CUDA gate for torch-extension ports" below for the fingerprints, the diagnosis, and the `cuda-not-validated` recording convention. (FaithC, accelerated-scan)
  - A CUDA-path configure failure at `find_package(CUDA <ver> REQUIRED)` ("did not find CUDAConfig.cmake / cuda-config.cmake") on a project whose `cmake_minimum_required` floor is >= 3.27 is a CMake/toolchain incompatibility, not a port regression, if the project still calls the legacy `FindCUDA` module (as opposed to pure `enable_language(CUDA)` + `FindCUDAToolkit`, which is unaffected). CMake 3.27 added policy CMP0146 ("the FindCUDA module is removed"); once the project's minimum-required version reaches or exceeds a policy's introduction version, that policy defaults to NEW and the module self-disables (its file is found on disk but returns immediately) -- confirm with `--debug-find-pkg=CUDA`, which shows "The file was found at .../FindCUDA.cmake" right above the "did not find" error. `-DCMAKE_POLICY_DEFAULT_CMP0146=OLD` does NOT recover it once minimum-required is at/above the policy version (verified with a two-line reproducer CMakeLists.txt); only an in-project `cmake_policy(SET CMP0146 OLD)` before the `find_package(CUDA...)` call works, which would be an edit to code the port does not own. Confirm pre-existing by building the identical upstream base sha with the identical cmake/nvcc/gcc versions: if it fails identically, record `cuda-not-validated` with the CMP0146 explanation, not a gate. (arbor)
  - Fault class -- clang-only compiler flag reaches nvcc unconditionally: a flag added for the HIP build gets appended to a SHARED `extra_compile_args`/flag list without an `if IS_HIP` (or `USE_ROCM`) guard, and hipcc (clang-based, accepts GCC/clang-style flags like `-fvisibility=hidden` directly on its own command line) never complains, so the port looks clean until nvcc sees the same flag and rejects it outright (`nvcc fatal : Unknown option '-fvisibility=hidden'`) because nvcc's own flags are a distinct namespace from the host compiler's -- GCC/clang-style flags must reach nvcc wrapped as `-Xcompiler=<flag>` (or via `--compiler-options`), never bare. `asm("trap;")` swapped for `__builtin_trap()` is the device-code-syntax instance of the same underlying mistake (HIP/clang accepts it, nvcc's frontend does not); the compiler-flag instance found in CuMesh (`setup.py` appending `-fvisibility=hidden -fvisibility-inlines-hidden` straight into a torch `CUDAExtension`'s `"nvcc"` key for both backends) is the build-system-level version. Diagnostic: if a CUDA no-regression compile fails and the flag/construct in question does not appear anywhere in the pre-port upstream source at all (`git show <upstream_sha>:<file> | grep <token>` empty), it is unambiguously new port code, not pre-existing breakage -- no need to also build upstream to compare. The preventive compiler-key rule and three-way host mapping belong in `strategy-b-torch.md`; apply them while editing the extension rather than waiting for this gate to expose the mistake.
    - Verify the wrap POSITIVELY, not just that the build stopped failing: a mis-wrapped flag is silently accepted and dropped, so the build goes green while the intent (here, symbol hiding) is gone. Confirm the effect in the artifact -- `nm -D --defined-only <ext>.so | grep <ns>` should show only the module entry point while `nm --defined-only` shows the namespace's symbols as local. In CuMesh's `_cubvh` that was 1 exported (`PyInit__cubvh`) against 309 local `cubvh::` symbols.
    - When a build-flag fix is meant to touch only the CUDA path, prove the ROCm path is untouched by DIFFING the generated flag lists instead of arguing from the diff: monkeypatch `setuptools.setup` to print each extension's `extra_compile_args` and exit, run it either side of the change under `BUILD_TARGET=rocm`, and diff. An identical dump is a much stronger claim than "the guard looks right", and it is what tells a validator the already-passed archs face a no-op. (CuMesh)
  - Torch-extension (Strategy B) CUDA no-regression setup: a bare conda `cuda-toolkit` env has no python/torch, so `CUDAExtension`'s build path (which needs `torch.utils.cpp_extension`) is unreachable from it directly. Make a throwaway venv with a matching-major.minor CUDA-enabled torch wheel instead (e.g. conda toolkit is 12.8 -> `pip install --index-url https://download.pytorch.org/whl/cu128 torch==<latest>+cu128`; PyPI's default `pip install torch` pulls whatever CUDA minor is current and can mismatch the installed toolkit). Point `CUDA_HOME` at the toolkit env and prepend its `bin/` to `PATH`; set `TORCH_CUDA_ARCH_LIST` (e.g. `8.0`) to pin the arch the same way `-DCMAKE_CUDA_ARCHITECTURES` does for CMake projects. The nvidia-channel `cuda-toolkit` conda package puts `cuda.h`/`cuda_runtime.h` under `targets/x86_64-linux/include/`, not directly under `include/`; a host-compiler TU that `#include`s them directly (not through nvcc) needs that path added to `CPLUS_INCLUDE_PATH`/`C_INCLUDE_PATH` explicitly. Run `setup.py build_ext` from the extension's own directory (its `sources` list is relative to cwd, not to `__file__`). The CUDA-only torch wheel is the reliable lever because `torch.version.hip` is `None`; merely unsetting `ROCM_HOME`/`ROCM_PATH` is not, since torch can rediscover ROCm from its SDK package, `hipcc` on `PATH`, or `/opt/rocm`. If the project has an explicit backend override such as `BUILD_TARGET=cuda`, set it as an additional guard. (CuMesh)
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

## Windows: `cmd.exe /c "<script>.bat"` from Git Bash silently does nothing

Git Bash's MSYS layer rewrites any argument that looks like a Unix path, and `/c` (cmd's
own switch, not a path) gets mangled into a drive letter. `cmd.exe /c "build.bat"`
launched from Git Bash then opens an interactive shell and returns immediately -- exit 0,
a log containing only the `Microsoft Windows [Version ...]` banner, no compiler output,
in well under a second. That is a silent no-op, not a build result: a real build takes
real time, so a suspiciously fast "pass" from a `cmd.exe /c` wrapper on this host is the
tell. Fix: prefix the invocation with `MSYS2_ARG_CONV_EXCL="*"` (or `MSYS2_ARG_CONV_EXCL=/c`)
to stop the rewrite; `MSYS2_ARG_CONV_EXCL="*" cmd.exe /c "build.bat"` runs the script.

## Windows: keep one clang driver across every CMake language, and check the CMake version

With CMake's `HIP` language on Windows, all three of `CMAKE_C_COMPILER`,
`CMAKE_CXX_COMPILER` and `CMAKE_HIP_COMPILER` must use the SAME clang driver. Mixing
drivers fails before one TU compiles, with an error that does not name the cause:
`CXX=clang-cl` with `HIP=clang++` makes CMake apply MSVC link rules to a GNU-driver link
line, giving `clang++: error: no such file or directory: '/machine:x64'`.

Which driver to pick depends on the CMake version, so check it first:

- **CMake 4.x: either works.** All-`clang-cl` is fine, and is what a project whose other
  dependencies are MSVC-built will want. (UCM's recorded gfx1151 recipe; re-confirmed on
  dietgpu with CMake 4.3.2.)
- **CMake 3.31: `clang-cl` cannot be the HIP compiler.** `Windows-Clang-HIP.cmake` never
  populates `CMAKE_HIP_SIMULATE_VERSION`, so configure dies inside `Windows-MSVC.cmake`
  with "MSVC compiler version not detected properly". Use the GNU-driver
  `clang`/`clang++` for all three instead -- it still targets the MSVC ABI, so MSVC-built
  dependencies and vendored submodules link normally. (Proven on dietgpu.)

Read that failure as "this CMake is too old", not "clang-cl cannot host HIP". A newer
CMake is one `pip install cmake` away and TheRock's own venv already carries 4.3.2, so
reach for it before restructuring a build around the older constraint.

Note the split with PyTorch extension ports, where `clang-cl` IS the required host
compiler (see strategy-b-torch.md): torch drives the compile itself and never goes
through CMake's `HIP` language.

## Windows: stage the ROCm runtime next to the .exe, kpack included

On a host whose System32 `amdhip64_7.dll` is broken, `PATH` does not save you -- Windows
resolves System32 first -- so copy the SDK runtime into the directory holding each test
executable. Copy `rocm_kpack.dll` along with it: `amdhip64_7.dll` imports it, and when it
is absent the process dies with a bare `error while loading shared libraries: ?` and exit
127, naming no module at all. That empty `?` is the fingerprint of a missing transitive
import, not of a bad port.

`llvm-objdump -p <binary> | grep "DLL Name"` walks the import table and finds the gap in
seconds; run it on the DLLs too, not just the .exe, since the missing module is usually
one level down.

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

## Windows torch-extension CUDA-compat shims can rot across torch versions

A hand-written `cuda.h`/`cuda_runtime.h` shim (added so an unhipified vendored header
like `#include <ATen/cuda/CUDAContext.h>` still compiles under HIP on Windows, where
TheRock's SDK ships no CUDA compat headers at all) is only complete against the specific
torch build it was validated with. Torch's own `c10/cuda/CUDAStream.h` is not a stable,
minimal surface: a later torch nightly can add a method (`is_capturing()`, using
`cudaStreamCaptureStatus`/`cudaStreamCaptureStatusNone`/`cudaStreamIsCapturing`) that an
older validated build never exercised, and the shim silently stops covering it. The
symptom is `error: unknown type name 'cudaStreamCaptureStatus'; did you mean
'hipStreamCaptureStatus'` deep inside a real torch header the port does not own, which
reads like a toolchain wall but is not one -- it is pure host-side header compilation, no
GPU or driver involved, so it is not host-specific and would reproduce on any Windows arch
building against that same (or a newer, similarly-shaped) torch release. Confirm by
diffing the shim's declared symbol set against what the failing header line actually
needs; if the shim is missing a handful of names it is a narrow, describable gap, not
grounds for a waiver ("if your own notes say an X fix would let this reach completed, you
are looking at a defect").

A second, distinct symptom can appear in the same file once the first class of error is
fixed or tolerated past clang's `-ferror-limit`: "redefinition of ..." across `c10/hip/*`
headers, when a `.hip` file includes both the vendored header's raw
`ATen/cuda/CUDAContext.h` chain AND an explicit `ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h`
include added earlier (to make `at::cuda::getCurrentCUDAStream()` resolve). On a torch
version where `ATen/cuda/*` is genuinely non-hip-aware content rather than a HIP-aliased
masquerading header, both trees land in one translation unit and collide -- a structural
conflict between two real headers, not a one-line shim fix, and worth escalating to the
porter rather than patching live during validation. (CuMesh, windows-gfx1151, torch
2.12.0+rocm7.14.0a20260519 vs. the 2.9.1 build the port was last validated against)

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

## RCCL collectives need the whole multi-GPU host at once

NCCL->RCCL is a mechanical swap (RCCL is API-compatible and ships in ROCm), but
validating it is unlike validating a kernel: collectives are a multi-GPU feature, so
the test process must hold two or more same-architecture GPUs SIMULTANEOUSLY. Two
consequences for scheduling on a multi-GPU host:

- Parallelising ports across a host's GPUs -- one build or validation pinned per
  device with `HIP_VISIBLE_DEVICES` -- is suspended for the duration of a
  collectives run. The run owns every device it uses; another port's job landing on
  a sibling GPU mid-run contends with it, and both results are suspect.
- The one-device-per-process pin the mixed-arch section above prescribes must be
  LIFTED for the collectives process: give it exactly the same-architecture devices
  it should use (read the device list at run time, never a remembered index), and
  keep mixed-architecture pairs out of one process regardless.

A single-GPU host cannot exercise the collectives path at all; record that as the
arch's limitation rather than letting a collectives-off run stand in for it
silently. Read the device list before concluding that, though: a host remembered
as single-GPU may not be one. What the run has to show is the library announcing
itself and the communicators constructing, plus a training job spanning every
device converging and its model decoding correctly -- a job that starts and does
not crash proves only that the collectives were never called. (marian-dev, on
four gfx1100)

## A passing assertion COUNT is not a list of what was tested

Catch2 stops iterating a test case's remaining SECTIONs once one of them throws,
so a single unported path that aborts hides every section after it. marian-dev's
operator suite reported "284 of 287 assertions; the 3 failures are the sparse
path" across three platforms and two reviews. Fixing the sparse path turned the
same suite into 603 assertions and immediately failed three of the newly reached
ones, in an unrelated operation that had been broken on ROCm from the start.

When a suite has a known-failing section, its pass count is a LOWER BOUND on what
went untested, and "the failures are isolated to X" describes what ran, not the
port. Compare the assertion count against the CUDA build's, or filter the failing
section out and watch whether the total moves. A deferral that makes a test case
abort quietly defers the rest of the file with it. (marian-dev)

## One architecture gets wrong numbers while the others pass

A clean build that produces wrong results on exactly one architecture -- an iterative
solver, an LM/Newton fit, an FP regression head -- is usually floating-point
accumulation divergence rather than a port bug, and RDNA3.5 (gfx1151) is where it has
shown up. Record the error magnitude and stop rather than chasing it deep: the
comparison that matters is against the other architectures, not against a fix.

## `classify` needs the fork already cloned at `projects/<name>/src`

`moatlib.py classify <name> <old_sha> <new_sha>` diffs the fork's git history, so it
reads `projects/<name>/src` as a checkout, not a description. In a fresh worktree
(the common case: a validator dispatched to revalidate rarely has yesterday's build
directory lying around) that path does not exist yet, and the tool fails closed --
`class=unknown arch_independent=False (classification failed -> revalidate)` -- which
is indistinguishable from a genuine indeterminate verdict. Clone the fork's port
branch into `projects/<name>/src` FIRST (or point it at an existing checkout), then
classify; do not read an `unknown` verdict as evidence the delta is functional before
checking whether the clone was even there. (SCAMP)

## Diagnosing a suspected AMD fault before escalating

Two patterns that each cost a deep investigation before the real cause was found.

- **A "data-dependent, later-data-corrupts-earlier, per-tile" corruption signature is the fingerprint of a REPRODUCER bug, not a codegen fault.** cuSZ chased a suspected miscompile to a BLOCKED state and an IR bisect; the actual cause was the test input -- `np.arange(..., dtype=float32) * (python float)` promotes to float64, so `.tofile()` wrote 8 bytes per element and the tool read the stream as f32. Validate the byte width and dtype of any binary test input before escalating to an ISA bisect or a ROCm bug report. (cuSZ)
- **Triangulate single- against double-precision before blaming the wavefront.** When a warp-collective rewrite shows SP divergence, run a second GPU variant and compare both to the DP oracle. If both GPU variants diverge from DP identically at the same positions, and the DP path is bit-identical to the CPU oracle, it is floating-point reassociation at a comparison boundary -- not a wave-size fault. SCAMP used this to clear a ~0.5 divergence at 10/8093 positions as a threshold-boundary artifact. (SCAMP)

## CUDA gate for torch-extension ports: the host's installed PyTorch itself can be the wall

On a host whose only installed PyTorch is a ROCm dev build (e.g. built from `/var/lib/jenkins/pytorch` with `USE_ROCM=1`), nvcc-compiling a torch C++/CUDA extension fails for reasons that have nothing to do with the port. No full `CUDAExtension` link is reachable without downloading a genuine CUDA-flavored torch wheel, so the closest available check is a raw `nvcc -c` of the `.cu` against `torch.utils.cpp_extension.include_paths()` -- and that dies in the ambient install's headers. Two fingerprints of the same root cause:

- The shipped `torch/include` tree was generated for the ROCm build and is missing CUDA-only generated headers, e.g. `#include <c10/cuda/impl/cuda_cmake_macros.h>: No such file or directory` (that file is emitted by cmake only when `USE_CUDA=1`).
- `torch/headeronly/util/complex.h` guards `#include <thrust/complex.h>` with `#if defined(__HIPCC__) || defined(__HIPCC__)` -- a duplicated-token typo, evidently meant `__CUDACC__ || __HIPCC__` -- so under nvcc the include is skipped while `c10/util/complex.h`/`complex_math.h` still reference `thrust::complex` unconditionally under `__CUDACC__`, cascading into ~100 "identifier thrust is undefined" errors on ANY project that includes `torch/extension.h`, regardless of that project's own code (observed on dev build `2.14.0a0+gitb6b444c`).

Neither is a defect in the port: they are defects in the ambient PyTorch install, present identically whether you build the pristine upstream or the ROCm branch. Diagnose it as environmental by checking that the errors bottom out in `torch/headeronly`/`c10` (not the port's own files) and that `grep -n '__builtin_trap\|__trap\|__HIP\|hip[A-Z]\|amdgcn\|USE_ROCM'` over the port's own changed sources is empty. Do not chase this by patching the installed torch headers (that is fixing the environment, not the port). Record `cuda-not-validated: <the missing-header or duplicated-guard error>` and, if there is time budget left, substitute a source-level check: diff the port's CUDA-facing code (e.g. the non-ROCm branch of a dual-path file) against upstream and confirm it is byte-for-byte the same logic, only guarded differently -- that is as much passthrough evidence as a real nvcc pass would give without one. (FaithC, accelerated-scan)
- **A cascading wall of SegFault/OOM/"no kernel image available" partway through a long serial suite can be one poisoned physical card, not a port defect -- check `rocm-smi --showpids` against `--showmeminfo vram` before concluding anything.** arrayfire on an 8x MI300X host: a full `ctest` run started clean, crashed (SegFault) on an early binary, and every binary after it degraded -- more SegFaults, a "no kernel image is available for execution on the device" (hipErrorNoBinaryForGpu) on a JIT op, then explicit "Device out of memory". `rocm-smi --showmeminfo vram` showed that GPU pinned near its full capacity; `rocm-smi --showpids` showed *no process attached to it at all*, including after `kill -9` of every test binary -- the VRAM was orphaned at the KFD/driver level, not held by a live allocator, and a process crash should not leak GPU memory past its own exit. Re-running the identical suite (fresh JIT disk cache) on a different, verified-idle card on the same host, same build, same commit, came back 131/132 clean with no crash, no OOM, and no "no kernel image" anywhere. The disagreement between the two `rocm-smi` views is the tell: if `--showmeminfo` says a card is nearly full but `--showpids` shows nothing on it, that card's state cannot be trusted for this session -- switch cards to disambiguate before spending time on the port. (arrayfire)

## HIP + C++23 on an older toolchain: `isfinite cannot overload` (and how to stay forward-compatible)

Symptom, on the first HIP translation unit compiled at `-std=c++23`:

```
__clang_cuda_math_forward_declares.h:93:17: error: __device__ function 'isfinite'
  cannot overload __host__ __device__ function 'isfinite'
<MSVC>\include\cmath:672:17: note: previous declaration is here
  672 | _CLANG_BUILTIN1(isfinite)
```

**Cause.** C++23 (P0533R9) makes the standard math functions `constexpr`. HIP treats a
`constexpr` function as implicitly `__host__ __device__`, so clang's
`__clang_cuda_math_forward_declares.h`, which declares `isfinite`/`isinf`/`isnan`/`isnormal`
`__device__`-only, can no longer overload them. Every HIP TU fails identically -- they all
pull the same `__clang_hip_runtime_wrapper.h` chain.

**Already fixed upstream**: llvm/llvm-project#201563 (merged 2026-06-09) includes the
forward-declares BEFORE `<cmath>`. Reported at ROCm/llvm-project#285 and #2669, both closed.
Do NOT file this again.

**Which toolchains have it** (verified 2026-08-24 by reading the shipped header, not by
version arithmetic -- ROCm cherry-picks, so commit ancestry misleads):

| Toolchain | Has fix? |
|---|---|
| ROCm 7.14 (`7.14.0a20260612`, the last of the old nightly stream) | no |
| ROCm 10.0 (`release/therock-10.0`, llvm pin 2026-07-28) | **no** -- it will ship without it |
| ROCm 10.1 nightly (`10.1.0a20260823` and later) | **yes** |
| ROCm/llvm-project `amd-compiler-2026-06` .. `-09`, and `main` | no |
| ROCm/llvm-project `amd-staging` | yes |

To check any toolchain in one line, read the order in its own header:
`grep -n -E '#include <cmath>|__clang_cuda_math_forward_declares' $(clang++ -print-resource-dir)/include/__clang_hip_runtime_wrapper.h`
-- forward-declares on the LOWER line number means fixed.

**Preferred fix: upgrade.** ROCm 10.1 nightlies carry it. Note the index moved: the old
`rocm.nightlies.amd.com` is frozen at `7.14.0a20260612`; current builds are at
`https://nightly.repo.amd.com/rocm/whl-next/`. Package layout changed too --
`rocm-sdk-devel` is now a thin metapackage whose copy of this header is a SYMLINK into
`_rocm_sdk_core`, and device packages renamed `rocm-sdk-libraries-gfx*` ->
`rocm-sdk-device-gfx*`.

**If you must stay on an older ROCm**, use `assets/HipCxx23Shim.cmake`. It generates a copy
of the INSTALLED wrapper with the two includes reordered and puts it on the include path
ahead of the resource dir -- the upstream fix, applied locally, with the language standard
untouched.

Measured alternatives, all tried against ROCm 7.14 on gfx1151:

| Approach | Compiles at C++23 | Keeps C++23 language | Keeps C++23 STL |
|---|---|---|---|
| `-std=c++20` pin | yes | **no** | **no** |
| `-D_CONSTEXPR23=` | **no** (the STL redefines it) | -- | -- |
| `-D_HAS_CXX23=0` | yes | yes | **no** |
| generated wrapper shim on `-I` | yes | yes | yes |

Prefer the shim. Both `-std=c++20` and `-D_HAS_CXX23=0` leave HIP TUs and host TUs with
DIFFERENT standard-library configurations, which is an ODR hazard whenever a shared type
crosses that boundary -- and MSVC has no `detect_mismatch` for `_HAS_CXX*`, so it fails
silently at runtime rather than loudly at link. That matters most in exactly the projects
that hit this, since they are C++23 project-wide.

**Forward compatibility is the whole design of the module**, because a shim that keeps
applying after the toolchain is fixed would shadow a good header with a stale copy:
generated from the installed header at configure time and never checked in; returns empty if
a plain probe already compiles; returns empty if the installed header is already in the fixed
order; and re-probes WITH the shim, discarding it and warning if it did not actually help.

Verified in four directions on this host: ROCm 7.14 at C++23 -> shim generated and a real HIP
target builds; a simulated ROCm 10.1 toolchain (real 10.1 header via `-resource-dir`) ->
"no shim needed", nothing generated; a C++20 project -> "no shim needed"; and a negative
control with the shim disabled -> the same target fails on the same header, proving the pass
was not vacuous.

Nothing here is guarded on Windows. MSVC's STL is merely the first to implement P0533R9;
ROCm/llvm-project#285 notes that libstdc++ doing so will break Linux identically. A
`WIN32`-only guard, or a ROCm version check, would miss that. Probe for the capability.

## `jargon.py --port` alone misses an in-flight fix round

`utils/jargon.py --port <name>` resolves its range from `obj.get("fork_branch") or
PORT_BRANCH`, and nothing in `moatlib.py` ever writes `fork_branch` -- so it always
defaults to `moat-port`. Once a fix round is staged on `moat-fix-<pr#>` (the published
`moat-port` is frozen while a PR is open), `--port` silently scans the FROZEN branch and
never sees the round's new commits; it can report "clean" while genuinely-added jargon
sits unscanned on the staging branch. Scan the real range explicitly instead: `python3
utils/jargon.py -C projects/<name>/src --commits <base>..<fix-branch> --diff
<base>..<fix-branch>`. Two things make this fail open rather than loud: a shallow single-branch
clone of the fork usually has no local `main`, and `scan_commits`/`scan_diff` (unlike
`port_range`'s explicit range-count check) report zero hits on an unresolvable git range
instead of erroring -- so an unfetched `main` produces a false "clean", not a crash. Fetch
it first (`git fetch --depth 50 origin main:main`) and confirm the range resolves to a
nonzero commit count before trusting a "clean" result. (mahout)

## A cached build-script decision from a wrong env var survives fixing the env var

If a `build.rs` reads an env var (e.g. `ROCM_PATH`) to emit `cargo:rustc-link-search`, but
does not also declare `cargo:rerun-if-env-changed=<that var>`, cargo has no reason to
re-run the script when only the env var changes -- so a build attempted once with a wrong
value (e.g. an old `ROCM_PATH` copied from another host's notes, pointing at a path that
doesn't exist on this one) caches the wrong `-L` search path in `target/debug/build/<crate>-<hash>/output`
and keeps reusing it even after the env var is corrected. The compile step can appear to
pass (a `.rlib`/static-archive target does not need the search path), while the FIRST
step that actually links a final binary -- a `cargo test` binary, not `cargo build` of a
lib -- fails with `unable to find library -l<name>`, which looks like a missing runtime
library rather than a stale cache. Fix: `cargo clean -p <crate>` (not just re-running with
the right env) before rebuilding. Generalizes beyond ROCm: always check `env | grep -i
<toolkit>_path` against what the build script actually expects before trusting old notes
from a different host generation -- ROCm/CUDA install layouts (`/opt/rocm` vs a
TheRock/pip `_rocm_sdk_devel` tree vs a conda env) are not portable between hosts even
within the same fleet. (mahout)
