## Strategy B: pytorch extension

Torch hipifies extension sources at build time. Do not add a compat header and do not hand-rename symbols.

- Building a `CUDAExtension` on a ROCm torch automatically runs `torch.utils.hipify` on the extension's `.cu`/`.cuh` sources and links the HIP runtime (`amdhip64`, `c10_hip`, `torch_hip`). See `torch/utils/cpp_extension.py`.
- Keep sources in CUDA spelling. In a hipified TU that is correct under both hipify generations (see below), so it needs no guard. Fix only what hipify cannot (warp size, see the fault classes) in source, guarded by `USE_ROCM`.
- Build against a ROCm torch. If the tree was hipified once and is stale after edits, re-run the project's hipify step before rebuilding (a known incremental-build gotcha: edits to `.cu` can recompile the stale hipified mirror unless you re-hipify first).
- For projects shipping their own `.cu` plus a setup.py, the change is often just: build against a ROCm torch and fix the fault classes below.

### Never commit the hipified mirror; add it to .gitignore instead

Torch hipify writes its output into the working tree on every build, usually as SIBLING FILES next to the CUDA sources. They are build output that happens to land in the source tree, and `git add -A` after a build sweeps all of them in.

`torch/utils/hipify/hipify_python.get_hip_file_path` (reached from `cpp_extension.py` via `hipify(..., is_pytorch_extension=True)`) decides each output path, and it does NOT always append `_hip`. It rewrites `cuda`->`hip`, `CUDA`->`HIP` and `THC`->`THH` and always rewrites a `.cu` extension to `.hip`. It appends `_hip` to the stem ONLY when that rewrite left both the directory and the full filename unchanged; otherwise the mirror is simply the renamed path. So where the output lands depends on whether the source path already contains the word:

| source | hipified output |
| --- | --- |
| `src/kernel.cu` | `src/kernel.hip` |
| `src/kernel.cuh` | `src/kernel_hip.cuh` |
| `src/util.h` | `src/util_hip.h` |
| `src/cuda/kernel.cuh` | `src/hip/kernel.cuh` -- new directory, filename unchanged |
| `src/cuda_kernels/kernel.cuh` | `src/hip_kernels/kernel.cuh` -- substring, not a whole component |
| `src/cuda_utils.cuh` | `src/hip_utils.cuh` -- same directory, no `_hip` suffix |
| `src/THCFoo.h` | `src/THHFoo.h` |

The directory rewrite is a plain substring `replace` over the whole dirpath string, not a match on path components, so `cuda_kernels/` and `my_cuda/` are rewritten exactly like `cuda/` is. A `cuda`-containing directory or filename is common in CUDA projects, so the renaming rows are not exotic. Copying a suffix-only ignore list into such a project produces a list that misses the entire mirror.

A mirror appears only where hipify actually CHANGES the content. `hipify_python.preprocessor` returns `[skipped, no changes]` and writes no file when the translated output is byte-identical to the input and the dirpath did not move; `preprocess_file_and_save_result` is only the wrapper that calls it and prints the status, so look in `preprocessor` for the skip. So the table predicts WHERE a mirror lands, not THAT one exists: a source hipify leaves alone produces nothing at all. That is why faster-gaussian-splatting's 17 `.h` files yielded only 10 `_hip.h`, and why its `bindings.cpp` yielded no mirror despite being an eligible input. Deriving the ignore list from the table therefore over-predicts, which is harmless; what it must not do is under-predict, so verify against a real build rather than against a count.

Committing them is worse than noise. A maintainer opening the PR sees a machine translation of their own kernels sitting beside the originals, dwarfing the real diff -- faster-gaussian-splatting pushed 26 such files, 5,883 of 5,971 added lines, for an 88-line port. It is also a staleness trap that defeats the integrity gate: hipify rewrites the files in place, so a from-clean build leaves `git status` clean and everything looks fine, but the next source edit dirties TRACKED files, which then either get committed out of sync with the sources or read as a dirty fork.

So in any Strategy B port, before the first commit, put in the project's `.gitignore`, in its existing style:

    *.hip
    *_hip.h
    *_hip.cuh

Those three cover the suffix case only, which is the whole mirror only when no source path contains `cuda`/`CUDA`/`THC`. Derive the rest from the table: a directory path CONTAINING `cuda`/`CUDA`/`THC` anywhere in it puts its mirror in a sibling directory with that substring rewritten, so ignore that directory (`src/hip/` for `src/cuda/`, `src/hip_kernels/` for `src/cuda_kernels/`); a `cuda`/`CUDA`/`THC` in a filename gives a mirror with the rewritten name in the same directory, so ignore that name (`hip_utils.cuh`, or `hip_*` if there are many and the project has no hand-written ones).

Do not trust the derivation on its own -- verify. `git clean -fdx`, full build, and `git status --porcelain` must print nothing: no tracked file modified AND no generated file newly untracked. The second half is the part that catches this; a tree where the artifacts are tracked passes the first half. It also catches a rename you did not predict, which is why it is the check that matters rather than the pattern list. If the project genuinely hand-writes a `.hip` file or already has a `hip/` directory (Strategy B ports normally do not), narrow the pattern rather than dropping it. (faster-gaussian-splatting)

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

### Dependency environment for PyTorch projects (the install path is part of the port)

For a PyTorch-ecosystem project the in-repo `.cu` (above) is only half the bring-up: the project also `pip install`s a dependency graph that defaults to CUDA wheels, and getting a ROCm environment that actually EXERCISES your ported code is its own recurring task. The failure modes repeat across the whole 3D-vision / point-cloud / VLA / world-model space, so they generalize. (AMD's rocm3d -- Andy Luo and David Li, 2026, andyluo7.github.io -- catalogs these as reusable recipes for that domain; the dependency-family patterns below are the transferable core. Note the model: these are environment substitutions to get a project RUNNING on ROCm, not upstreamed source ports -- MOAT's deliverable is still the in-repo HIP port + multi-arch PR, so confirm your ported code is the code under test, not a swapped-in prebuilt wheel.)

- A pip install silently overwrites the ROCm torch. An unfiltered `pip install -r requirements.txt` reinstalls the CUDA `torch`/`torchvision`/`xformers`/`flash-attn`/`triton` wheels on top of your ROCm ones, after which the project "has no GPU". Strip the CUDA-default pins first -- `grep -vEi '^(torch|torchvision|torchaudio|xformers|triton|flash.attn|cupy|gsplat)' requirements.txt | pip install -r /dev/stdin` -- then install the ROCm builds explicitly.
- Wheel name != import name. AMD ships some ROCm wheels on its own index (`pip install <pkg> --extra-index-url=https://pypi.amd.com/<rocm-ver>/simple/`); the distribution name can differ from the import (`amd_gsplat` imports as `gsplat`). Pin the ROCm index so pip does not silently prefer a CUDA wheel from default PyPI.
- Build source extensions AGAINST the installed ROCm torch: `PYTORCH_ROCM_ARCH=<arch> pip install <submodule> --no-build-isolation`. `--no-build-isolation` is load-bearing -- without it pip builds in a fresh isolated env and pulls a CUDA torch to compile against, yielding a CUDA extension that then fails at import on ROCm.
- A common dep usually already has a ROCm port -- consume it, do not re-port it. Apply the existing-AMD-support test (above) to each DEPENDENCY, not just the top repo: `spconv` -> `spconv_rocm`, `torch_scatter`/`torch_cluster` -> the pyg-rocm-build wheels, `gsplat` -> AMD's `amd_gsplat`. (So a dependency being un-portable at the source level does not by itself block a downstream consumer -- a dep-level ROCm substitute may exist.) Consume these via the `_deps/` mechanism; they are environment, not deliverables.
- Import success != correct backend, and that gap is a VALIDATION trap. A project with several attention paths (PyTorch SDPA, FlashAttention-2 Triton, AITER Triton/CK) will often `import` fine and "run" by silently falling back to SDPA -- exercising NONE of the kernel you care about. Enable the AMD path explicitly (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` for flash-attn; install `aiter` for the AITER paths) and then CONFIRM the runtime backend (logs/profiler, steady-state iteration time, a produced artifact), not just that the import worked. This binds straight to MOAT's Validation policy: when swapped ROCm wheels carry the kernels, a run can "pass" on a CPU/SDPA fallback while testing nothing of your port -- prove the code under test is YOUR ported code and that it ran on the GPU.
- Repo-local native op with no ROCm wheel = an upstreamable port, not a swap. When the project ships its own CUDA op that no ROCm package provides (e.g. a deformable-attention `.cu`), the clean pattern is: build the forward path as a HIP extension, register it as a PyTorch custom op with a pure-PyTorch fallback for any unported path, and validate by numerical diff against the reference (tol ~1e-6) plus a latency check. This IS the MOAT source contribution and the upstream PR; the dependency recipes above are only the environment around it.
