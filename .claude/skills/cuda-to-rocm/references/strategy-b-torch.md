## Strategy B: pytorch extension

Torch hipifies extension sources at build time. Do not add a compat header and do not hand-rename symbols.

- Building a `CUDAExtension` on a ROCm torch automatically runs `torch.utils.hipify` on the extension's `.cu`/`.cuh` sources and links the HIP runtime (`amdhip64`, `c10_hip`, `torch_hip`). See `torch/utils/cpp_extension.py`.
- Keep sources in CUDA spelling. In a hipified TU that is correct under both hipify generations (see below), so it needs no guard. Fix only what hipify cannot (warp size, see the fault classes) in source, guarded by `USE_ROCM`.
- Build against a ROCm torch. If the tree was hipified once and is stale after edits, re-run the project's hipify step before rebuilding (a known incremental-build gotcha: edits to `.cu` can recompile the stale hipified mirror unless you re-hipify first).
- For projects shipping their own `.cu` plus a setup.py, the change is often just: build against a ROCm torch and fix the fault classes below.

### hipify generation: v1 renames, v2 masquerades

Which generation runs decides what the `c10`/`at` CUDA classes are called, so it is worth
knowing before writing a stream or allocator call.

- **v1** renamed them: only `c10::hip::getCurrentHIPStream` existed after hipification.
- **v2** (pytorch#174087, `torch/utils/hipify/version.py` bumped 1.0.0 -> 2.0.0) STOPS
  renaming them. The CUDA spellings stay public and are the masquerading API, while the
  hip-spelled symbols become `#ifdef USE_ROCM`.

**In an ordinary Strategy B extension this costs you nothing: write the CUDA spelling.**
v1 renames it, v2 keeps it, and no guard is needed either way. That is the whole reason
Strategy B keeps sources in CUDA spelling.

**The trap is a TU that hipify never sees.** A host `.cpp` deliberately routed around
hipify -- through hipcc for a Windows link fix, say -- or a CMake/`USE_HIP` port where torch
source-hipify never runs at all, gets no translation. Hard-coding either spelling there
breaks the other generation. Detect it at build time and branch:

    Version(torch.utils.hipify.__version__) >= "2.0.0"   ->  pass -DTORCH_HIPIFY_V2

    #if defined(TORCH_HIPIFY_V2)
    ... c10::cuda::getCurrentCUDAStream ...
    #else
    ... c10::hip::getCurrentHIPStream ...
    #endif

Use a neutral define; never a moat-named one, since it ends up in the upstream diff.

**Do not reach for `_WIN32` or a ROCm version as a proxy for the generation.** They
correlate only by accident in this fleet and can anti-correlate -- a newer ROCm shipping an
older torch -- so a guard keyed on the wrong axis silently breaks the other platform's
build. That fault class, with the two regressions that produced it, is in
`fault-classes.md` under "Types, dispatch and platform limits". (aihwkit)

### CMake-driven torch extension: `find_package(Torch)` takes the GPU architecture away from you

A torch extension built through CMake rather than `CUDAExtension` (`find_package(Torch)` +
`pybind11_add_module`, the Strategy-B environment with Strategy-A mechanics) has an
architecture-selection trap that silently ships a module with no code object for the user's
GPU. Two independent things go wrong, and both are ORDERING bugs -- the fix is where the
lines sit, not what they say.

**1. Enabling the language decides the architecture, so decide it first.**
`CMakeDetermineHIPCompiler.cmake` caches `CMAKE_HIP_ARCHITECTURES` from
`rocm_agent_enumerator` the moment `enable_language(HIP)` runs and the variable is not
already defined (`elseif(NOT DEFINED CMAKE_HIP_ARCHITECTURES)`). Anything that sets it
AFTER that line is dead code. `CMakeDetermineCUDACompiler.cmake` does the same for
`CMAKE_CUDA_ARCHITECTURES` (`if("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")`), which is why
upstream projects put `set(CMAKE_CUDA_ARCHITECTURES native)` above `project()`. Restructuring
a `project(... LANGUAGES CUDA CXX)` into `project(... LANGUAGES CXX)` + a `USE_HIP`-branched
`enable_language()` moves that line below the decision point and QUIETLY BREAKS THE NVIDIA
BUILD -- on Quest the CUDA path lost `native` and would have fallen back to nvcc's default,
which ptxas rejects for the `cp.async`/`mma.sync` the vendored flashinfer kernels emit. Both
architecture defaults, and any compiler pins, belong above `project()` under `if(NOT USE_HIP)`;
`option(USE_HIP ...)` can precede `project()` so the guard is available there.

**2. `find_package(Torch)` then overwrites it with PyTorch's OWN list.**
`Caffe2/public/LoadHIP.cmake` does `set(CMAKE_HIP_ARCHITECTURES ${PYTORCH_ROCM_ARCH})` -- a
NORMAL variable that shadows your cache entry for the rest of the directory scope (and it
calls `enable_language(HIP)` itself). `PYTORCH_ROCM_ARCH` there comes from the environment,
else from `rocm_agent_enumerator` (`Caffe2/public/utils.cmake`, `torch_hip_get_arch_list`).
So `set_target_properties(mod PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")`
written after `find_package(Torch)` re-applies the multi-arch list the torch WHEEL was built
for, and `-DCMAKE_HIP_ARCHITECTURES=<yours>` has no effect at all. Snapshot the resolved
target into a private variable before `find_package(Torch)` and apply THAT:

    if(NOT CMAKE_HIP_ARCHITECTURES AND DEFINED ENV{PYTORCH_ROCM_ARCH})
      string(REPLACE " " ";" _proj_hip_archs "$ENV{PYTORCH_ROCM_ARCH}")
      set(CMAKE_HIP_ARCHITECTURES "${_proj_hip_archs}" CACHE STRING "HIP architectures" FORCE)
    endif()
    enable_language(HIP)
    set(<PROJ>_HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES})
    list(REMOVE_DUPLICATES <PROJ>_HIP_ARCHITECTURES)   # enumerator repeats per device
    ...
    find_package(Torch REQUIRED)
    ...
    set_target_properties(mod PROPERTIES HIP_ARCHITECTURES "${<PROJ>_HIP_ARCHITECTURES}")

That gives the precedence a user expects: `-DCMAKE_HIP_ARCHITECTURES` wins, then
`PYTORCH_ROCM_ARCH`, then the local GPUs.

**`CACHE ... FORCE` on that line is load bearing, and a plain `set()` is a trap that passes
every test you are likely to run.** The branch in `CMakeDetermineHIPCompiler.cmake` that
writes the cache entry is an `elseif(NOT DEFINED CMAKE_HIP_ARCHITECTURES)`, so setting a
NORMAL variable ahead of `enable_language(HIP)` both supplies the value AND suppresses the
only thing that would have persisted it: `CMakeCache.txt` ends up with no
`CMAKE_HIP_ARCHITECTURES` entry at all and the architecture lives for exactly one configure.
The build works, the compile line is right, the tests pass. Then any later configure in that
directory without the variable still in the environment -- and ninja re-runs cmake by itself
whenever a listfile changes -- resolves it to empty and hard-fails with `HIP_ARCHITECTURES
is empty for target "<tgt>"`. Writing it to the cache is what makes the value survive the
configure that discovered it. `FORCE` cannot clobber a user's `-DCMAKE_HIP_ARCHITECTURES`,
because the enclosing `if(NOT CMAKE_HIP_ARCHITECTURES ...)` has already excluded that case.
Note the failure mode this replaces is a REGRESSION the port introduces: before the fix
torch's own `LoadHIP.cmake` re-read the environment every configure, so the build directory
never went empty. Test it explicitly -- configure with the variable set, then reconfigure the
SAME build directory with `env -u PYTORCH_ROCM_ARCH`, and require both a zero exit and the
original architecture on the compile line. (Quest)

**Do not settle this by reading the CMake. Read the generated compile line.** Configure with
`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` and
`grep -o -- "--offload-arch=[^ \"]*" build/compile_commands.json | sort -u`, once per
precedence case, and confirm the shipped binary with
`llvm-objdump --offloading <mod>.so`. On Quest, `-DCMAKE_HIP_ARCHITECTURES=gfx1201` was
producing `gfx90a,gfx942,gfx950,gfx1100` and no gfx1201; nothing in the CMake said so, and
the only symptom a user sees is `hipErrorNoBinaryForGpu` at import. Note that the bug HIDES
on a host whose GPU is in the torch wheel's list -- which is most build machines -- so the
compile-line check is the gate, not a successful local run. (Quest)

### Dependency environment for PyTorch projects (the install path is part of the port)

For a PyTorch-ecosystem project the in-repo `.cu` (above) is only half the bring-up: the project also `pip install`s a dependency graph that defaults to CUDA wheels, and getting a ROCm environment that actually EXERCISES your ported code is its own recurring task. The failure modes repeat across the whole 3D-vision / point-cloud / VLA / world-model space, so they generalize. (AMD's rocm3d -- Andy Luo and David Li, 2026, andyluo7.github.io -- catalogs these as reusable recipes for that domain; the dependency-family patterns below are the transferable core. Note the model: these are environment substitutions to get a project RUNNING on ROCm, not upstreamed source ports -- MOAT's deliverable is still the in-repo HIP port + multi-arch PR, so confirm your ported code is the code under test, not a swapped-in prebuilt wheel.)

- A pip install silently overwrites the ROCm torch. An unfiltered `pip install -r requirements.txt` reinstalls the CUDA `torch`/`torchvision`/`xformers`/`flash-attn`/`triton` wheels on top of your ROCm ones, after which the project "has no GPU". Strip the CUDA-default pins first -- `grep -vEi '^(torch|torchvision|torchaudio|xformers|triton|flash.attn|cupy|gsplat)' requirements.txt | pip install -r /dev/stdin` -- then install the ROCm builds explicitly.
- Wheel name != import name. AMD ships some ROCm wheels on its own index (`pip install <pkg> --extra-index-url=https://pypi.amd.com/<rocm-ver>/simple/`); the distribution name can differ from the import (`amd_gsplat` imports as `gsplat`). Pin the ROCm index so pip does not silently prefer a CUDA wheel from default PyPI.
- Build source extensions AGAINST the installed ROCm torch: `PYTORCH_ROCM_ARCH=<arch> pip install <submodule> --no-build-isolation`. `--no-build-isolation` is load-bearing -- without it pip builds in a fresh isolated env and pulls a CUDA torch to compile against, yielding a CUDA extension that then fails at import on ROCm.
- A common dep usually already has a ROCm port -- consume it, do not re-port it. Apply the existing-AMD-support test (above) to each DEPENDENCY, not just the top repo: `spconv` -> `spconv_rocm`, `torch_scatter`/`torch_cluster` -> the pyg-rocm-build wheels, `gsplat` -> AMD's `amd_gsplat`. (So a dependency being un-portable at the source level does not by itself block a downstream consumer -- a dep-level ROCm substitute may exist.) Consume these via the `_deps/` mechanism; they are environment, not deliverables.
- Import success != correct backend, and that gap is a VALIDATION trap. A project with several attention paths (PyTorch SDPA, FlashAttention-2 Triton, AITER Triton/CK) will often `import` fine and "run" by silently falling back to SDPA -- exercising NONE of the kernel you care about. Enable the AMD path explicitly (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` for flash-attn; install `aiter` for the AITER paths) and then CONFIRM the runtime backend (logs/profiler, steady-state iteration time, a produced artifact), not just that the import worked. This binds straight to MOAT's Validation policy: when swapped ROCm wheels carry the kernels, a run can "pass" on a CPU/SDPA fallback while testing nothing of your port -- prove the code under test is YOUR ported code and that it ran on the GPU.
- Repo-local native op with no ROCm wheel = an upstreamable port, not a swap. When the project ships its own CUDA op that no ROCm package provides (e.g. a deformable-attention `.cu`), the clean pattern is: build the forward path as a HIP extension, register it as a PyTorch custom op with a pure-PyTorch fallback for any unported path, and validate by numerical diff against the reference (tol ~1e-6) plus a latency check. This IS the MOAT source contribution and the upstream PR; the dependency recipes above are only the environment around it.
