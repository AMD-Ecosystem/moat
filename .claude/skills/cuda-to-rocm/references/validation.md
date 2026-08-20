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
  - A torch `CUDAExtension` cannot get this gate the same way: `torch/extension.h` pulls in the fleet's ONLY installed PyTorch, a ROCm dev build with no CUDA-flavored counterpart on hand, so the check dies in the ambient install's headers before it can judge the port. See "CUDA gate for torch-extension ports" below for the fingerprints, the diagnosis, and the `cuda-not-validated` recording convention. (FaithC, accelerated-scan)
  - A CUDA-path configure failure at `find_package(CUDA <ver> REQUIRED)` ("did not find CUDAConfig.cmake / cuda-config.cmake") on a project whose `cmake_minimum_required` floor is >= 3.27 is a CMake/toolchain incompatibility, not a port regression, if the project still calls the legacy `FindCUDA` module (as opposed to pure `enable_language(CUDA)` + `FindCUDAToolkit`, which is unaffected). CMake 3.27 added policy CMP0146 ("the FindCUDA module is removed"); once the project's minimum-required version reaches or exceeds a policy's introduction version, that policy defaults to NEW and the module self-disables (its file is found on disk but returns immediately) -- confirm with `--debug-find-pkg=CUDA`, which shows "The file was found at .../FindCUDA.cmake" right above the "did not find" error. `-DCMAKE_POLICY_DEFAULT_CMP0146=OLD` does NOT recover it once minimum-required is at/above the policy version (verified with a two-line reproducer CMakeLists.txt); only an in-project `cmake_policy(SET CMP0146 OLD)` before the `find_package(CUDA...)` call works, which would be an edit to code the port does not own. Confirm pre-existing by building the identical upstream base sha with the identical cmake/nvcc/gcc versions: if it fails identically, record `cuda-not-validated` with the CMP0146 explanation, not a gate. (arbor)
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

## Windows: an unpinned `find_package(Python)` silently builds against the wrong interpreter, and an unpinned build type silently wants a debug lib that does not exist

A CMake project that calls `find_package(Python REQUIRED COMPONENTS Interpreter
Development)` with no hint, on a Windows host that has more than one Python discoverable
via PATH or the registry, does not necessarily pick the venv you activated. It can silently
resolve a different interpreter -- e.g. a system Python 3.13 instead of the venv's cp312 --
and link the extension against that interpreter's import library, producing a
`_kernels.cp313-win_amd64.pyd` when the test harness needs a cp312 build. Nothing in the
configure or build log flags this; the target filename (`cp3XX-win_amd64.pyd`) is the tell,
and only shows up in `build.ninja`, not the configure summary. Fix by pinning
`-DPython_EXECUTABLE=<venv>/Scripts/python.exe` explicitly rather than trusting discovery.

Separately, an unset `CMAKE_BUILD_TYPE` on a single-config Windows generator (Ninja) is not
"no build type" -- CMake and MSVC toolchain files commonly default it to `Debug`. A pybind
target's `Python::Python`/`Development` component then links against the DEBUG import
library (`python312_d.lib`), which a normal (non-debug) CPython install does not ship. The
failure surfaces at the LINK step, after every object has compiled cleanly:
`lld-link: error: could not open 'python312_d.lib': no such file or directory`. It reads
like a missing dependency, not a build-type default; fix by pinning
`-DCMAKE_BUILD_TYPE=Release` explicitly. Both knobs are validator-environment CMake
invocation flags, not fork changes -- record them in `notes.md` so the next Windows session
does not rediscover them, the same way the amdclang++-vs-clang-cl and `-fuse-ld=lld-link`
Windows CMake gotchas already are.

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

## CUDA no-regression gate: an old pinned dependency can wall it before your own code is even reached

A project that vendors an external NVIDIA library at a specific pinned tag (RAPIDS/RAFT,
a CUB/Thrust snapshot, anything fetched by CPM/FetchContent) can fail its OWN CMake
configure against a modern CUDA toolkit, before the port's `.cu` files are ever compiled.
Quest pins RAFT to `branch-24.02` (~early 2024); against CUDA 12.8 + CMake 3.31, RAFT's
CMakeLists still links the legacy `CUDA::nvToolsExt` imported target, but
`FindCUDAToolkit` no longer creates it -- NVTX became header-only
(`nvtx3/nvToolsExt.h`) and the shared-library target it backed was retired. Configure
dies inside `_deps/raft-src/cpp/CMakeLists.txt`, nowhere near anything the port touched.

Tell this apart from a real CUDA regression by WHERE the error sits: a fault in a
`_deps/<external>-src/` path, or any file this port did not edit, is the dependency's
own version skew against the toolchain you happened to use to test -- it would equally
block an unmodified upstream build under the same CUDA-toolkit + CMake combination, so
it says nothing about the port. A fault inside the project's own source, especially one
that traces to a HIP-only define/branch leaking into the CUDA leg, is the real gate and
must go back to the porter. Do not try to work around the dependency's own build system
(stubbing an imported target, patching its CMakeLists) to force the CUDA leg further --
that fix belongs upstream in the dependency, not in this port, and patching it locally
proves nothing about the port either. Record it as `cuda-not-validated: <precise error
and where it sits>` and say plainly how far the pinned-arch mechanism WAS exercised
(e.g. `enable_language(CUDA)` succeeding with the pin surviving to the cache is real,
new evidence even when the wall stops you one step later) versus what remains
unexercised (no compile line was ever seen for the port's own kernels). Don't let the
record imply more than what was actually run.

## A path only some GPUs reach: force it, with an override that cannot be ignored

When a kernel picks between two code paths from a device property -- multiprocessor
count, occupancy, wavefront size, available LDS -- every machine exercises exactly one
of them and the other is untested there. Quest's decode takes a single-block kernel over
a partitioned one when `batch_size * num_kv_heads >= blocks_per_cu * cu_count`: true on a
20-CU part, false on 70- and 304-CU ones, so a repair to the single-block path was
invisible to two of the three platforms that validated it and to any NVIDIA CI.

An environment-variable override that forces the choice is the cheap fix, but only if it
cannot be ignored in silence. Otherwise a green run under the override is
indistinguishable from a run that never took the forced path, and the evidence the
override exists to produce is worth nothing. Three ways it no-ops, all of which shipped
in Quest's first version of the knob:

- an unrecognized value returning "no override" instead of failing, so `Single` for
  `single`, or any typo, reads as a pass;
- caching the `getenv` in a function-local `static`, so the in-process
  `monkeypatch.setenv` that a "test-visible override" invites reads the first value the
  process ever saw -- read it per call instead, this is host-side setup, not an inner
  loop;
- a pre-existing fallback further down that leaves the requested path anyway (Quest's
  `if (new_batch_size == batch_size) { tmp_size = 0; }` reverts a requested split to the
  single-block shape).

Hard-error on all three, using the project's own exception type so it surfaces in the
test harness (`throw std::invalid_argument` reaches Python as `ValueError` through
pybind11). Then drive the override from a test in the tree -- a pytest that re-runs the
affected suites once per path as subprocesses -- so the coverage is automatic instead of
something a validator must remember from a commit body, and document the knob where the
build is documented. Do not restrict such a knob to the HIP build without a reason: the
coverage gap comes from the device, not the platform, and the NVIDIA leg has it too.

Confirm independently that the forced paths really do differ before trusting the
override: `AMD_LOG_LEVEL=3` plus a grep for a kernel name only one path launches is
enough (in Quest, forced split launched `MergeStatesKernel` six times over the decode
suite, forced single-block never launched it). A guard that no test can reach -- Quest's
split-requested-but-unpartitioned error, unreachable at batch size 1 -- can be checked by
inverting its condition, rebuilding, observing the throw, then restoring and rebuilding.

## Diagnosing a suspected AMD fault before escalating

Two patterns that each cost a deep investigation before the real cause was found.

- **A "data-dependent, later-data-corrupts-earlier, per-tile" corruption signature is the fingerprint of a REPRODUCER bug, not a codegen fault.** cuSZ chased a suspected miscompile to a BLOCKED state and an IR bisect; the actual cause was the test input -- `np.arange(..., dtype=float32) * (python float)` promotes to float64, so `.tofile()` wrote 8 bytes per element and the tool read the stream as f32. Validate the byte width and dtype of any binary test input before escalating to an ISA bisect or a ROCm bug report. (cuSZ)
- **Triangulate single- against double-precision before blaming the wavefront.** When a warp-collective rewrite shows SP divergence, run a second GPU variant and compare both to the DP oracle. If both GPU variants diverge from DP identically at the same positions, and the DP path is bit-identical to the CPU oracle, it is floating-point reassociation at a comparison boundary -- not a wave-size fault. SCAMP used this to clear a ~0.5 divergence at 10/8093 positions as a threshold-boundary artifact. (SCAMP)

## CUDA gate for torch-extension ports: the host's installed PyTorch itself can be the wall

On a host whose only installed PyTorch is a ROCm dev build (e.g. built from `/var/lib/jenkins/pytorch` with `USE_ROCM=1`), nvcc-compiling a torch C++/CUDA extension fails for reasons that have nothing to do with the port. No full `CUDAExtension` link is reachable without downloading a genuine CUDA-flavored torch wheel, so the closest available check is a raw `nvcc -c` of the `.cu` against `torch.utils.cpp_extension.include_paths()` -- and that dies in the ambient install's headers. Two fingerprints of the same root cause:

- The shipped `torch/include` tree was generated for the ROCm build and is missing CUDA-only generated headers, e.g. `#include <c10/cuda/impl/cuda_cmake_macros.h>: No such file or directory` (that file is emitted by cmake only when `USE_CUDA=1`).
- `torch/headeronly/util/complex.h` guards `#include <thrust/complex.h>` with `#if defined(__HIPCC__) || defined(__HIPCC__)` -- a duplicated-token typo, evidently meant `__CUDACC__ || __HIPCC__` -- so under nvcc the include is skipped while `c10/util/complex.h`/`complex_math.h` still reference `thrust::complex` unconditionally under `__CUDACC__`, cascading into ~100 "identifier thrust is undefined" errors on ANY project that includes `torch/extension.h`, regardless of that project's own code (observed on dev build `2.14.0a0+gitb6b444c`).

Neither is a defect in the port: they are defects in the ambient PyTorch install, present identically whether you build the pristine upstream or the ROCm branch. Diagnose it as environmental by checking that the errors bottom out in `torch/headeronly`/`c10` (not the port's own files) and that `grep -n '__builtin_trap\|__trap\|__HIP\|hip[A-Z]\|amdgcn\|USE_ROCM'` over the port's own changed sources is empty. Do not chase this by patching the installed torch headers (that is fixing the environment, not the port). Record `cuda-not-validated: <the missing-header or duplicated-guard error>` and, if there is time budget left, substitute a source-level check: diff the port's CUDA-facing code (e.g. the non-ROCm branch of a dual-path file) against upstream and confirm it is byte-for-byte the same logic, only guarded differently -- that is as much passthrough evidence as a real nvcc pass would give without one. (FaithC, accelerated-scan)
