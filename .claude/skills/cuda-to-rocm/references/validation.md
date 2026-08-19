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
## Prove a GPU test RAN; do not infer it from a green result or from wall time

A GPU test that never touched the GPU reports PASS. This is common enough to plan for:
a test-fixture helper compiled to an empty stub without a windowing toolkit, a
`skipIf(not is_available())` that evaluates true because the runtime is not visible in
that shell, an entire suite gated behind a `GPU_ENABLED` that a headless configure turned
off. The suite says "100% tests passed" and the port is unvalidated.

Wall time is the obvious heuristic and it is NOT reliable. Use kernel dispatches:

    AMD_LOG_LEVEL=3 ./the_test --gtest_filter='TheGpuTest.*' 2>&1 \
      | grep -oE 'YourKernelName|OtherKernelName' | sort | uniq -c

The HIP runtime names every kernel it launches. Zero dispatches of your kernels means the
body did not run, whatever the result line says. This is direct evidence, cheap, and it
works on any project without instrumenting the code.

colmap is the source. Its GPU SIFT tests run through a `RunGpuTest` helper that calls
`RunThreadWithOpenGLContext`, which is an empty inline when the GUI is disabled -- so a
headless build silently skipped every GPU test body and reported 145 tests passing. Worse
for the timing heuristic: after the fix, the GPU matcher tests take 0-3 ms in the headless
build and ~200 ms in the GUI build, and the difference is entirely Qt constructing an
offscreen context. Timing would have labelled the real run a skipped one and the skipped
run a real one. The dispatch count was right both times.

Two follow-ons worth taking:

- When a project's test infrastructure has such a stub, FIXING it is porting work, not
  scope creep, and it is not platform-specific: the same trap was hiding the same thing
  from the CUDA build. Add a test that asserts the body ran, so the no-op cannot return.
- The dispatch log also tells you WHICH backend ran. A project with both a compute and a
  GLSL/CPU fallback will happily pass its whole suite on the fallback; seeing your kernel
  names is what distinguishes "the port works" from "the fallback works".
- When a second arch validates the same fixed test inputs, DIFF the dispatch counts against
  the first arch's recorded numbers rather than just checking they are nonzero. If the test
  data and algorithm are deterministic (colmap's `sift_test` runs fixed images through fixed
  SIFT parameters), the per-kernel counts are architecture-independent even though wavefront
  width differs -- gfx1100 (wave32) and gfx90a (wave64) produced the identical kernel-for-
  kernel, count-for-count table for colmap. An exact match is stronger corroboration than a
  second "it's nonzero" check; a mismatch with no code difference between the two validations
  would be a signal worth chasing, not a footnote.

### Closing a "it might silently fall back" risk: trace it, do not reason about it

Dispatch counts prove the backend ran on the run you measured. When the worry is that some
ORDER of events could flip a global backend flag -- "if the fallback initializes first, the
compute flag gets cleared and the suite still goes green" -- the answer is a trace, and a
Release build with no debug info is enough for one. Statics like `Foo::_UseBackend` and
ordinary member functions are in the symbol table, so gdb can breakpoint the decision
function, breakpoint each backend's constructor, and put a hardware watchpoint on the flag:

    break *0x<addr of the decision fn>
    commands
    silent
    printf "decide(arg=%d) flag=%d\n", (int)$rdi, *(int*)0x<addr of flag>
    continue
    end
    watch *(int*)0x<addr of flag>

Counts answer the question outright: how many times the decision function was entered and
with what argument, how many times each backend was constructed, and every value the flag
ever took. In colmap that turned "did not materialise on the machine we happened to use"
into "the clearing line did not run once in this binary, and the flag was written exactly
once, `0 -> 1`".

**Then be careful what you claim from it, because this is where a trace gets oversold.** A
trace covers the paths the binary you ran actually took, with the options it ran with. It
is evidence for "not reached in this configuration"; it is never evidence for
"unreachable". colmap's first write-up made exactly that jump -- "the clearing line is
unreachable, because the only caller always requests the compute backend" -- and review
disproved it with one probe: a public option (`darkness_adaptivity`, combined with the
default negative GPU index) makes the caller omit the flag that selects the compute
backend, and the supposedly dead line runs. Upgrading "not reached here" to "cannot happen"
takes an argument from the CODE, not a bigger trace. In colmap the argument that did hold
was that the API re-asserts the flag on every freshly constructed object that requests the
compute backend, so no earlier user can leave it cleared for a later one -- true on every
GPU and every driver, and it is what actually closed the risk. Trace to find the mechanism;
read the code to bound it.

**When that argument holds, ask what ELSE the failing path wrote.** Proving flag A cannot
persist closes the risk you named, not the failure mode. The initialization that clears A
usually sets siblings in the same function, and one of those is what reaches the next user.
In colmap the failed GLSL init also zeroes a second global meaning "this process has no
usable GL", it is sticky (the init function returns immediately once it is 0 and never
retries), and every later extractor -- including the compute ones the structural argument
had just protected -- early-returns on it and is constructed as a null. There is no fallback
on that path at all, so the consequence is a hard failure with nothing extracted, not the
silent degrade the risk was written about; a project WITHOUT a fallback fails loudly instead
of passing green, which is the better of the two symptoms and still a live defect. So:
enumerate every write on the failure path, not just the flag you suspected, and for each ask
who resets it and who reads it later. Sticky-on-failure, process-wide, and consulted by a
user that would otherwise have worked is the shape that bites; the state that bites is
rarely the state the risk named.

Two traps that each produce a confidently wrong answer:

- **Reading a global AFTER the inferior exits gives the ELF initial value, not the last
  live value.** gdb falls back to the executable's `.data`/`.bss` image once the process is
  gone, and prints it without complaint. In colmap that printed exactly the cleared-flag
  state the reviewer had predicted, from a run where the flag was never cleared. Read the
  flag at a breakpoint inside the live process, or watch it.
- **`LD_PRELOAD` on a GL/driver entry point does not intercept a Qt (or any
  `GetProcAddress`-style) caller.** Qt resolves GL through `QOpenGLFunctions` /
  `eglGetProcAddress`, so the preloaded symbol is bypassed while a directly-linked caller in
  the same process IS intercepted. If you use a preload shim to fake an environment, prove
  it fires with a positive control first; a breakpoint needs no such control.

(colmap)

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

## A GPU test that hangs only under `ctest -jN` is usually the software GL stack

A test that passes standalone and hangs in the suite is not a GPU fault and rarely a port
fault. Get the stack before theorising -- `sudo gdb -p <pid> -batch -ex "bt"`, since
`ptrace_scope` normally blocks a plain attach -- and read which library the top frames are
in. colmap's `feature/sift_test` hung under `xvfb-run -a ctest -jN` while passing 5 out of
5 standalone; the stack was `RunGpuTest -> ~QApplicationPrivate -> ~QXcbIntegration ->
XCloseDisplay -> libGLX_mesa -> libgallium -> pthread_clockjoin`, i.e. Mesa llvmpipe
deadlocking on its own worker threads while closing the X display with other GL clients
live on it. Nothing COLMAP or ROCm wrote appears in the trace, and the Qt frames name the
trigger: a `QApplication` built and destroyed per test.

**It is a race, so one green run proves nothing -- and a string of green ones does not
establish a rate either.** Counts from one host at `-j16`, in the order they were taken: 2
hangs in 2 initial `-j8`/`-j16` attempts; then 4 runs giving 1 full pass (159/159, 8.76 s)
and 3 timeouts; then, in a later session holding constant every variable those runs could
name (same machine and 64 cores, same Mesa build, same build directory, same command), 8
runs giving 7 passes (8.7-9.5 s) and 1 timeout, with the hang between clean runs. Those
sessions do not fit a single frequency -- P(<= 1 hang in 8) is about 4e-4 if three in four
hang -- so the frequency is unstable across conditions nobody has isolated. The only
differences on record are GPU visibility (`HIP_VISIBLE_DEVICES` pinned to one card in the
last session) and background machine load, and the stack implicates neither. Report the
counts you observed and the conditions you held; do not publish a rate, and do not let a
green run retire the hazard.

The shape generalizes to any suite whose GPU tests build a windowing-toolkit context per
test: many short-lived GL clients on one Xvfb, and the teardown races. Isolate the two
variables separately before blaming either -- running the one test alone against a
PRE-EXISTING shared display distinguishes "shared display" from "concurrent clients"; in
colmap it was the latter. The reliable workaround is lowering `-j`: colmap ran the whole
159-test suite in 11.6-12.3 s at `-j4` with no hang observed, which is also the shape of
the evidence -- the suite is not slow, one test is stuck. Note that ctest reports the hang
as `Timeout` against its default 1500 s cap, which reads like a slow test rather than a
deadlock: a suite wall time sitting exactly at the cap with one test blamed is the tell.
(colmap)

## One architecture gets wrong numbers while the others pass

A clean build that produces wrong results on exactly one architecture -- an iterative
solver, an LM/Newton fit, an FP regression head -- is usually floating-point
accumulation divergence rather than a port bug, and RDNA3.5 (gfx1151) is where it has
shown up. Record the error magnitude and stop rather than chasing it deep: the
comparison that matters is against the other architectures, not against a fix.

## Diagnosing a suspected AMD fault before escalating

Two patterns that each cost a deep investigation before the real cause was found.

- **A "data-dependent, later-data-corrupts-earlier, per-tile" corruption signature is the fingerprint of a REPRODUCER bug, not a codegen fault.** cuSZ chased a suspected miscompile to a BLOCKED state and an IR bisect; the actual cause was the test input -- `np.arange(..., dtype=float32) * (python float)` promotes to float64, so `.tofile()` wrote 8 bytes per element and the tool read the stream as f32. Validate the byte width and dtype of any binary test input before escalating to an ISA bisect or a ROCm bug report. (cuSZ)
- **Triangulate single- against double-precision before blaming the wavefront.** When a warp-collective rewrite shows SP divergence, run a second GPU variant and compare both to the DP oracle. If both GPU variants diverge from DP identically at the same positions, and the DP path is bit-identical to the CPU oracle, it is floating-point reassociation at a comparison boundary -- not a wave-size fault. SCAMP used this to clear a ~0.5 divergence at 10/8093 positions as a threshold-boundary artifact. (SCAMP)

## CUDA gate for torch-extension ports: the host's installed PyTorch itself can be the wall

On a host whose only installed PyTorch is a ROCm dev build (e.g. built from `/var/lib/jenkins/pytorch` with `USE_ROCM=1`), nvcc-compiling a torch C++/CUDA extension fails for reasons that have nothing to do with the port. No full `CUDAExtension` link is reachable without downloading a genuine CUDA-flavored torch wheel, so the closest available check is a raw `nvcc -c` of the `.cu` against `torch.utils.cpp_extension.include_paths()` -- and that dies in the ambient install's headers. Two fingerprints of the same root cause:

- The shipped `torch/include` tree was generated for the ROCm build and is missing CUDA-only generated headers, e.g. `#include <c10/cuda/impl/cuda_cmake_macros.h>: No such file or directory` (that file is emitted by cmake only when `USE_CUDA=1`).
- `torch/headeronly/util/complex.h` guards `#include <thrust/complex.h>` with `#if defined(__HIPCC__) || defined(__HIPCC__)` -- a duplicated-token typo, evidently meant `__CUDACC__ || __HIPCC__` -- so under nvcc the include is skipped while `c10/util/complex.h`/`complex_math.h` still reference `thrust::complex` unconditionally under `__CUDACC__`, cascading into ~100 "identifier thrust is undefined" errors on ANY project that includes `torch/extension.h`, regardless of that project's own code (observed on dev build `2.14.0a0+gitb6b444c`).

Neither is a defect in the port: they are defects in the ambient PyTorch install, present identically whether you build the pristine upstream or the ROCm branch. Diagnose it as environmental by checking that the errors bottom out in `torch/headeronly`/`c10` (not the port's own files) and that `grep -n '__builtin_trap\|__trap\|__HIP\|hip[A-Z]\|amdgcn\|USE_ROCM'` over the port's own changed sources is empty. Do not chase this by patching the installed torch headers (that is fixing the environment, not the port). Record `cuda-not-validated: <the missing-header or duplicated-guard error>` and, if there is time budget left, substitute a source-level check: diff the port's CUDA-facing code (e.g. the non-ROCm branch of a dual-path file) against upstream and confirm it is byte-for-byte the same logic, only guarded differently -- that is as much passthrough evidence as a real nvcc pass would give without one. (FaithC, accelerated-scan)
