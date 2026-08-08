## Validation policy

- A port is validated only when the project's real test suite builds and passes on a real AMD GPU for the target arch, with no regression in non-GPU tests.
- Coverage is expressed as GATES -- wave64, wave32, windows -- and a gate is satisfied by ANY arch carrying that attribute. No arch leads: whichever host picks the project up ports it, and the rest validate the same branch independently and in parallel.
- A validation must not introduce a NEW fork commit for anything non-essential (CI YAML, formatting, comments). Advancing head_sha forces every already-passed arch back to `revalidate` for a zero-GPU-effect change -- pure churn. If your arch needs no code change, leave the commit untouched; amend in only a genuinely necessary build/source fix (e.g. configurable arch).
- A CPU-only docker build (image `rocm/dev-ubuntu-24.04:7.2.4-complete`) proves the code compiles and links under ROCm. It cannot observe any Fault-class bug above, since no GPU runs, so it is never a validation gate. Do NOT wire it into the fork's GitHub Actions: a yml change bumps the fork HEAD sha and forces every platform to revalidate (churn), and the run just fails and emails. Disable Actions on the fork instead; run a CPU-only docker build locally if you want a manual compile check.
- gfx90a and gfx942 are CDNA (wave64); the gfx11xx/gfx12xx parts are RDNA (wave32). A change that passes on one width can still fail on the other via the warp-size class, which is why each arch validates on its own hardware rather than inheriting a result.
- PR-prep gate -- nvcc CUDA-build check: before opening any PR whose claim is "the CUDA/NVIDIA build is unchanged", compile the CUDA path (`USE_HIP=OFF`) with nvcc on this GPU-less host (conda `cuda-nvcc`; full recipe in memory `moat-cuda-compile-check-on-rocm-host`). nvcc compiles without an NVIDIA GPU, so this is available on a ROCm-only host, and GPU validation on AMD cannot see a broken CUDA path -- "notes.md says the CUDA path is preserved" is NOT the same as having compiled it. State it honestly in the PR ("compile-checked with nvcc, not run"). REQUIRED when the port adds a CUDA/HIP-split backend (e.g. `Foo_cuda.cpp` vs `Foo_hip.cpp`): that CUDA file is never compiled during HIP-only porting and can be badly broken. The check must reach the LINK stage (build one real target/demo, not compile-only of a single TU) -- the undefined-reference class (e.g. explicit instantiations that do not match real call-site signatures) only surfaces at link. If the fix is HIP-binary-equivalent (codeobj_diff identical), it carries forward with no GPU re-validation.
  - Setup: point `-I` at the project's deps (vcpkg include dir, plus any CUDA-Samples headers it uses -- `helper_cuda.h`, `helper_math.h`, `helper_string.h`) and select the project's non-MATLAB/Python build macro. For a Thrust/CUB project also install `cuda-cccl`: on CUDA 13.x they ship there under `include/cccl/{thrust,cub}` (nvcc finds them automatically, but a host-compiler OpenMP-backend check needs that path on `-I` explicitly, and stdgpu's CMake wants `THRUST_INCLUDE_DIR` pointed at it).
  - For a header-only or template library the changed headers only compile when instantiated: CMake-configure the CUDA backend to generate its config headers, then `nvcc -c` a small TU that `template class`-instantiates the affected containers, rather than compiling headers alone.
  - The class this catches that a HIP-only build cannot: an unconditional device-header include reaching host translation units. Used on 8 projects in one six-week window; it caught a template-shadow regression in Velvet and an stdgpu regression of exactly that include class. (stdgpu, SCAMP, cuSZ, mahout, lc0, cuPDLPx, TIGRE, Velvet)
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
ever took. In colmap this turned a "did not materialise on the machine we happened to use"
into "the clearing line is unreachable, because the only caller always requests the compute
backend" -- a statement that holds on every GPU and every GL driver rather than on one host.

Two traps that each produce a confidently wrong answer:

- **Reading a global AFTER the inferior exits gives the ELF initial value, not the last
  live value.** gdb falls back to the executable's `.data`/`.bss` image once the process is
  gone, and prints it without complaint. In colmap that printed exactly the fallback state
  the reviewer had predicted, from a run where the flag was never cleared. Read the flag at
  a breakpoint inside the live process, or watch it.
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
`data/deferred.json` bug report (`therock-windows-lld-link-crt-xcu`) plus a workaround,
not a property of the platform. Worth trying, none of them yet tested: `/WHOLEARCHIVE` or
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
in. colmap's `feature/sift_test` hung twice out of two under `xvfb-run -a ctest -jN` and
passed 5 out of 5 standalone; the stack was `XCloseDisplay -> libGLX_mesa -> libgallium ->
pthread_join`, i.e. Mesa llvmpipe deadlocking on its own worker threads while closing the X
display with other GL clients live on it. Nothing COLMAP or ROCm wrote appears in the
trace.

The shape generalizes to any suite whose GPU tests build a windowing-toolkit context per
test: many short-lived GL clients on one Xvfb, and the teardown races. Isolate the two
variables separately before blaming either -- running the one test alone against a
PRE-EXISTING shared display distinguishes "shared display" from "concurrent clients"; in
colmap it was the latter. Fix by lowering `-j`: colmap hung at `-j8` and `-j16` and ran the
whole 159-test suite in 11.6 s at `-j4`, which is also the shape of the evidence -- the
suite is not slow, one test is stuck. Note that ctest reports this as `Timeout` against its default
1500 s cap, which reads like a slow test rather than a deadlock: a suite wall time sitting
exactly at the cap with one test blamed is the tell. (colmap)

## One architecture gets wrong numbers while the others pass

A clean build that produces wrong results on exactly one architecture -- an iterative
solver, an LM/Newton fit, an FP regression head -- is usually floating-point
accumulation divergence rather than a port bug, and RDNA3.5 (gfx1151) is where it has
shown up. Record the error magnitude and stop rather than chasing it deep: the
comparison that matters is against the other architectures, not against a fix.

## codeobj_diff needs both builds from the SAME absolute source path

`utils/codeobj_diff.py` compares device ISA byte-for-byte after stripping addresses, but it
cannot strip a source PATH that got compiled into the binary as a string literal. A device
TU that calls `assert()` embeds `__FILE__` -- the compiler's absolute path to that source
file -- into `.rodata`/`.strtab`, unconditionally, even in a Release build with no debug
info. Building the "old" sha in a second `git worktree`/clone at a different absolute path
than the "new" build changes that embedded string for every TU with an `assert()`, so
`codeobj_diff` reports `differ` on binaries whose actual instruction stream is unchanged --
a false positive that looks exactly like a real regression.

The tell: `roc-obj-ls` reports the identical device-code offset and size on both binaries,
and the byte-level divergence is confined to `.dynstr`/`.rodata`/`.strtab` string-table
sizes, not instruction bytes; `strings` on the extracted code object shows the only diff is
an absolute path prefix, not project-relative content. Fix by building both shas from the
SAME absolute source-tree path: checkout sha A in place, build to `-B dirA`, checkout sha B
in the SAME tree, build to `-B dirB`, then diff `dirA` vs `dirB`. Never stand up a second
worktree/clone at a different path for this comparison. (alien, gfx90a revalidate of a pure
header-file-move delta: cross-path compare said `differ` on all 4 GPU executables; the
same-path rebuild said `identical`, matching the sibling gfx1100/gfx1201 carry-forwards
already recorded for the identical source delta.)

## Diagnosing a suspected AMD fault before escalating

Two patterns that each cost a deep investigation before the real cause was found.

- **A "data-dependent, later-data-corrupts-earlier, per-tile" corruption signature is the fingerprint of a REPRODUCER bug, not a codegen fault.** cuSZ chased a suspected miscompile to a BLOCKED state and an IR bisect; the actual cause was the test input -- `np.arange(..., dtype=float32) * (python float)` promotes to float64, so `.tofile()` wrote 8 bytes per element and the tool read the stream as f32. Validate the byte width and dtype of any binary test input before escalating to an ISA bisect or a ROCm bug report. (cuSZ)
- **Triangulate single- against double-precision before blaming the wavefront.** When a warp-collective rewrite shows SP divergence, run a second GPU variant and compare both to the DP oracle. If both GPU variants diverge from DP identically at the same positions, and the DP path is bit-identical to the CPU oracle, it is floating-point reassociation at a comparison boundary -- not a wave-size fault. SCAMP used this to clear a ~0.5 divergence at 10/8093 positions as a threshold-boundary artifact. (SCAMP)
