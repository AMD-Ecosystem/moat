## Strategy B: pytorch extension

Torch hipifies extension sources at build time. Do not add a compat header and do not hand-rename symbols.

- Building a `CUDAExtension` on a ROCm torch automatically runs `torch.utils.hipify` on the extension's `.cu`/`.cuh` sources and links the HIP runtime (`amdhip64`, `c10_hip`, `torch_hip`). See `torch/utils/cpp_extension.py`.
- Keep sources in CUDA spelling. In a hipified TU that is correct under both hipify generations (see below), so it needs no guard. Fix only what hipify cannot (warp size, see the fault classes) in source, guarded by `USE_ROCM`.
- Build against a ROCm torch. If the tree was hipified once and is stale after edits, re-run the project's hipify step before rebuilding (a known incremental-build gotcha: edits to `.cu` can recompile the stale hipified mirror unless you re-hipify first).
- For projects shipping their own `.cu` plus a setup.py, the change is often just: build against a ROCm torch and fix the fault classes below.

### Keep compiler flags in the right command-line namespace

Torch's `CUDAExtension` uses the `"nvcc"` entry in `extra_compile_args` for both nvcc on
CUDA builds and hipcc on ROCm builds. That does not make their flag syntaxes interchangeable:
hipcc accepts GCC/clang host flags such as `-fvisibility=hidden` directly, while nvcc needs
the same flag wrapped as `-Xcompiler=-fvisibility=hidden`. The `"cxx"` entry is a separate
host-compiler command line and keeps the bare spelling.

Map a shared host flag according to all three relevant cases: pass it bare to hipcc, wrap
it for nvcc with a GCC/clang host, and drop it for nvcc with MSVC when there is no MSVC
equivalent. Derive the wrapped list from the bare list so the two cannot drift. Do not use
only an `IS_HIP` guard: that misses the Windows CUDA case and can hand cl.exe an unsupported
`-f...` option. Validate the mapping on each backend by inspecting the generated command
lines, and verify the intended artifact effect when a flag can be accepted but ignored.
(CuMesh)

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

### hipify scope for a vendored library: the `includes` pattern, not the walk

Known first half: `torch/utils/hipify/hipify_python.py`'s `matched_files_iter` PRUNES `third_party/` from its file walk (it re-adds only `third_party/nvfuser`), so a vendored library's `.cu` sources are reported "[ignored, not to be hipified]" and the raw CUDA is handed to the compiler. The standard workaround is to run `hipify_python.hipify()` yourself, rooted at the vendored library so the prune does not apply.

Less known second half, and the one that bites: **which HEADERS get rewritten is decided by the `includes` pattern, not by the walk.** `hipify()` extends `all_files` with every file under `header_include_dirs` that matches `includes`, and the include-rewrite hook then recursively hipifies any `#include`d header whose basename appears in `all_files`. Root your own pass at the library's `src/` with `includes=[<lib>/src/*]` and its `include/` tree stays raw -- even though you passed the include dir in `header_include_dirs`.

That asymmetry is why this hides on Linux and detonates elsewhere. torch's own pass uses `project_directory=<extension root>` with `includes=[<root>/*]`, and that pattern DOES match `third_party/<lib>/include/...` even though the walk pruned it -- so torch quietly hipifies the vendored headers for you. A hand-rolled pass rooted narrower does not, and the platform that needs the hand-rolled pass is the only platform that breaks.

The symptom is not a missing symbol, it is two header trees in one TU. A vendored header that includes `<ATen/cuda/CUDAContext.h>` stays raw while the source's `#ifdef USE_ROCM` include pulls `<ATen/hip/...>`, and those trees are alternatives that were never meant to coexist: torch's hipified `c10/hip/HIPStream.h` opens `namespace c10::cuda {` and declares the same `class CUDAStream` that `c10/cuda/CUDAStream.h` declares. You get a redefinition cascade across `c10/hip/HIPStream.h`, `HIPException.h`, `HIPFunctions.h`, `HIPDeviceAssertionHost.h`, plus `unknown type name 'cudaStreamCaptureStatus'` from `c10/cuda/CUDAStream.h`'s `is_capturing()`, which uses graph-capture symbols HIP has no equivalent for.

**Fix the scope, not the symptom.** Add the library's include dir to `includes` (`includes=[<lib>/src/*, <lib>/include/*]`, rooted at `<lib>`) so hipify emits `api_hip.h` carrying `<ATen/hip/HIPContext.h>` and rewrites the including source to match. Adding the missing capture symbols to a CUDA-compat shim looks like the fix and is not: it silences the first error and leaves the redefinition cascade untouched. Keep the patterns tight -- a bare `<lib>/*` would drag a nested Eigen or other vendored headers into hipify's scope.

Two diagnostics worth carrying:

- If you see "redefinition" of `c10/hip/*` classes, look UP the include chain for a raw `ATen/cuda` include, not for a missing symbol. `ATen/cuda/*` and `ATen/hip/*` are mutually exclusive in a single TU.
- Prefer making hipify's WALK find your sources over listing them in `extra_files`. The walk unix-normalizes each path; the `extra_files` append does not, and `preprocess_file_and_save_result` tests membership against the normalized form -- so on Windows an `extra_files` entry outside the walk root can be silently skipped.

(CuMesh, whose bundled `cubvh` failed exactly this way on Windows while Linux had built it for months.)

### Dependency environment for PyTorch projects (the install path is part of the port)

For a PyTorch-ecosystem project the in-repo `.cu` (above) is only half the bring-up: the project also `pip install`s a dependency graph that defaults to CUDA wheels, and getting a ROCm environment that actually EXERCISES your ported code is its own recurring task. The failure modes repeat across the whole 3D-vision / point-cloud / VLA / world-model space, so they generalize. (AMD's rocm3d -- Andy Luo and David Li, 2026, andyluo7.github.io -- catalogs these as reusable recipes for that domain; the dependency-family patterns below are the transferable core. Note the model: these are environment substitutions to get a project RUNNING on ROCm, not upstreamed source ports -- MOAT's deliverable is still the in-repo HIP port + multi-arch PR, so confirm your ported code is the code under test, not a swapped-in prebuilt wheel.)

- A pip install silently overwrites the ROCm torch. An unfiltered `pip install -r requirements.txt` reinstalls the CUDA `torch`/`torchvision`/`xformers`/`flash-attn`/`triton` wheels on top of your ROCm ones, after which the project "has no GPU". Strip the CUDA-default pins first -- `grep -vEi '^(torch|torchvision|torchaudio|xformers|triton|flash.attn|cupy|gsplat)' requirements.txt | pip install -r /dev/stdin` -- then install the ROCm builds explicitly.
- Wheel name != import name. AMD ships some ROCm wheels on its own index (`pip install <pkg> --extra-index-url=https://pypi.amd.com/<rocm-ver>/simple/`); the distribution name can differ from the import (`amd_gsplat` imports as `gsplat`). Pin the ROCm index so pip does not silently prefer a CUDA wheel from default PyPI.
- Build source extensions AGAINST the installed ROCm torch: `PYTORCH_ROCM_ARCH=<arch> pip install <submodule> --no-build-isolation`. `--no-build-isolation` is load-bearing -- without it pip builds in a fresh isolated env and pulls a CUDA torch to compile against, yielding a CUDA extension that then fails at import on ROCm.
- A common dep usually already has a ROCm port -- consume it, do not re-port it. Apply the existing-AMD-support test (above) to each DEPENDENCY, not just the top repo: `spconv` -> `spconv_rocm`, `torch_scatter`/`torch_cluster` -> the pyg-rocm-build wheels, `gsplat` -> AMD's `amd_gsplat`. (So a dependency being un-portable at the source level does not by itself block a downstream consumer -- a dep-level ROCm substitute may exist.) Consume these via the `_deps/` mechanism; they are environment, not deliverables.
- Import success != correct backend, and that gap is a VALIDATION trap. A project with several attention paths (PyTorch SDPA, FlashAttention-2 Triton, AITER Triton/CK) will often `import` fine and "run" by silently falling back to SDPA -- exercising NONE of the kernel you care about. Enable the AMD path explicitly (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` for flash-attn; install `aiter` for the AITER paths) and then CONFIRM the runtime backend (logs/profiler, steady-state iteration time, a produced artifact), not just that the import worked. This binds straight to MOAT's Validation policy: when swapped ROCm wheels carry the kernels, a run can "pass" on a CPU/SDPA fallback while testing nothing of your port -- prove the code under test is YOUR ported code and that it ran on the GPU.
- Repo-local native op with no ROCm wheel = an upstreamable port, not a swap. When the project ships its own CUDA op that no ROCm package provides (e.g. a deformable-attention `.cu`), the clean pattern is: build the forward path as a HIP extension, register it as a PyTorch custom op with a pure-PyTorch fallback for any unported path, and validate by numerical diff against the reference (tol ~1e-6) plus a latency check. This IS the MOAT source contribution and the upstream PR; the dependency recipes above are only the environment around it.
