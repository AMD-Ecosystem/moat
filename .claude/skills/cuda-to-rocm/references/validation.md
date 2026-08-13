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

## Windows: put the whole CMake build on the GNU-style clang driver

**A project using CMake's `HIP` language does not configure on Windows with a `clang-cl`
host compiler.** Both mixes fail before one TU compiles, and neither error names the real
cause:

- `CXX=clang-cl` + `HIP=clang++` -- CMake applies MSVC link rules to a GNU-driver link
  line, giving `clang++: error: no such file or directory: '/machine:x64'`.
- `CXX=clang-cl` + `HIP=clang-cl` -- `Windows-Clang-HIP.cmake` never populates
  `CMAKE_HIP_SIMULATE_VERSION`, so configure dies inside `Windows-MSVC.cmake` with
  "MSVC compiler version not detected properly".

Point `CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER` and `CMAKE_HIP_COMPILER` all at the SDK's
GNU-driver `clang`/`clang++`. It still targets the MSVC ABI, so MSVC-built dependencies
and vendored submodules link normally. (Proven on dietgpu, CMake 3.31 + clang 23.)

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

## Diagnosing a suspected AMD fault before escalating

Two patterns that each cost a deep investigation before the real cause was found.

- **A "data-dependent, later-data-corrupts-earlier, per-tile" corruption signature is the fingerprint of a REPRODUCER bug, not a codegen fault.** cuSZ chased a suspected miscompile to a BLOCKED state and an IR bisect; the actual cause was the test input -- `np.arange(..., dtype=float32) * (python float)` promotes to float64, so `.tofile()` wrote 8 bytes per element and the tool read the stream as f32. Validate the byte width and dtype of any binary test input before escalating to an ISA bisect or a ROCm bug report. (cuSZ)
- **Triangulate single- against double-precision before blaming the wavefront.** When a warp-collective rewrite shows SP divergence, run a second GPU variant and compare both to the DP oracle. If both GPU variants diverge from DP identically at the same positions, and the DP path is bit-identical to the CPU oracle, it is floating-point reassociation at a comparison boundary -- not a wave-size fault. SCAMP used this to clear a ~0.5 divergence at 10/8093 positions as a threshold-boundary artifact. (SCAMP)
