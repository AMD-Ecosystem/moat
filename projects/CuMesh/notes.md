# CuMesh notes

## Build

PyTorch extension (Strategy B). Build with a ROCm PyTorch environment:

```bash
cd projects/CuMesh/src
GPU_ARCHS=gfx90a pip install . --no-build-isolation -v
```

For multi-arch or other targets:
```bash
GPU_ARCHS="gfx90a;gfx1100" pip install . --no-build-isolation
```

## Port summary

Four files changed:

1. `src/dtypes.cuh`: Added `__host__` qualifier to Vec3f/QEM methods (hipCUB DeviceSegmentedReduce requires host-callable constructors)
2. `src/clean_up.cu`: Replaced `::cuda::std::tuple` with `rocprim::tuple` for int3_decomposer (CCCL unavailable on HIP)
3. `setup.py`: Gated NVCC-only flags (--extended-lambda, -U__CUDA_NO_HALF_*) behind IS_HIP check
4. `setup.py`: Updated C++ standard to C++20 (required by PyTorch 2.13+ headers)

No warp-size issues -- no warp intrinsics or hardcoded warpSize in the codebase.

## Validation

Tested on MI250X (gfx90a), ROCm 7.2:

- simplify.py: Mesh decimation 69451 -> 9949 faces
- fill_holes.py: Hole filling works
- remove_duplicate_faces.py: Duplicate removal works
- unify_orientations.py: Orientation unification works
- uv_unwrap.py: Fast clustering + xatlas integration (90 clusters)
- remesh.py: Dual contouring + BVH projection works
- cuBVH: BVH construction and queries work
- Atlas (xatlas): CPU-only module, unaffected by port

## Gotchas

- rocprim::tuple has an explicit constructor; use `rocprim::tie()` instead of braced init `{a, b, c}`
- PyTorch 2.13+ headers require C++20 for the `requires` keyword
- Build generates _hip.h/_hip.cpp files via torch hipify; these are gitignored build artifacts

## Review 2026-06-05

**Reviewer**: MOAT reviewer agent

**Verdict**: Approve (review-passed)

**Checklist**:
- Port strategy: Correct (Strategy B for PyTorch extension, torch hipifies at build time)
- Fault classes: None apply (no warp intrinsics, no textures, no resource handles)
- Minimal footprint: 4 files changed, all changes are HIP-guarded or additive
- Build system: Correct (`IS_HIP` gating, `GPU_ARCHS` for arch selection)
- Testing: All example scripts validated on gfx90a
- Backward compatibility: CUDA path preserved, `__host__` qualifiers valid for CUDA
- Commit hygiene: Title prefixed [ROCm], mentions Claude, no noreply trailer

No problems found.

## Validation 2026-06-05

**Platform**: linux-gfx90a (MI250X, ROCm 7.2, PyTorch 2.13.0a0+gitb5e90ff)

**Build command**:
```bash
cd /var/lib/jenkins/moat/projects/CuMesh/src
HIP_VISIBLE_DEVICES=1 GPU_ARCHS=gfx90a pip install . --no-build-isolation -v
```

**Test suite**: Example scripts in `examples/` directory

**Results**: All tests PASS

1. `simplify.py`: Mesh decimation 69451 -> 9830 faces PASS
2. `fill_holes.py`: Hole filling 69451 -> 69594 faces PASS
3. `remove_duplicate_faces.py`: Duplicate removal (no duplicates found) PASS
4. `unify_orientations.py`: Orientation unification PASS
5. `uv_unwrap.py`: Fast clustering (90 clusters) + xatlas UV unwrapping PASS
6. `remesh.py`: Dual contouring + BVH projection -> 204916 faces PASS

All GPU operations (mesh simplification, hole filling, remeshing, BVH construction/queries) executed successfully on gfx90a.

**Platform**: linux-gfx1100 (gfx1100, ROCm 7.2, PyTorch 2.13.0a0+gitb5e90ff)

**Build command**:
```bash
cd /var/lib/jenkins/moat/projects/CuMesh/src
GPU_ARCHS=gfx1100 pip install . --no-build-isolation -v
```

**Test suite**: Example scripts in `examples/` directory

**Results**: All tests PASS

1. `simplify.py`: Mesh decimation 69451 -> 9895 faces PASS
2. `fill_holes.py`: Hole filling 69451 -> 69594 faces PASS
3. `remove_duplicate_faces.py`: Duplicate removal (no duplicates found) PASS
4. `unify_orientations.py`: Orientation unification PASS
5. `uv_unwrap.py`: Fast clustering (90 clusters) + xatlas UV unwrapping PASS
6. `remesh.py`: Dual contouring + BVH projection -> 204916 faces PASS

All GPU operations (mesh simplification, hole filling, remeshing, BVH construction/queries) executed successfully on gfx1100.

## Windows build notes

### IS_HIP detection on Windows

`torch.utils.cpp_extension` sets `IS_HIP_EXTENSION` based on whether `ROCM_HOME` is set in the environment. On Windows, always set `ROCM_HOME` to the TheRock SDK devel directory:

```bat
set ROCM_HOME=B:\develop\TheRock\external-builds\pytorch\.venv\Lib\site-packages\_rocm_sdk_devel
set DISTUTILS_USE_SDK=1
```

### MSVC vs hipcc (clang) for PyTorch bindings

MSVC-compiled objects referencing `c10::ValueError(SourceLocation, string)` via inherited constructors (`using Error::Error`) generate direct `dllimport` symbol references that do not match what c10.dll exports on Windows. The fix is to compile such sources via hipcc (clang) using `.cu` wrapper includes.

Affected files routed through hipcc via `.cu` wrappers:
- `src/ext.cpp` -> `src/ext_winhip.cu`
- `third_party/xatlas/xatlas.cpp` -> `third_party/xatlas/xatlas_winhip.cu`
- `third_party/xatlas/binding.cpp` -> `third_party/xatlas/binding_winhip.cu`
- `third_party/cubvh/src/bindings.cpp` -> `third_party/cubvh/src/bindings_winhip.cu`

### hip_cuda_compat/ shim headers

TheRock's Windows SDK does not include CUDA compat headers. The `hip_cuda_compat/` directory provides minimal shims:
- `cuda.h` - `cudaMalloc` template, `cudaStream_t`, `cudaError_t` aliases, stream APIs
- `cuda_runtime.h`, `cuda_runtime_api.h` - forward to HIP headers
- `cuda_fp16.h`, `cublas_v2.h`, `cublasLt.h`, `cusparse.h`, `cusolverDn.h` - alias to ROCm equivalents
- `hipsolver/hipsolver.h` - `cusolverDnHandle_t = hipsolverDnHandle_t`
- `eigen_hip_compat.h` - `using std::fill_n` etc for Eigen SparseCore in HIP device compilation
- `c10/cuda/impl/cuda_cmake_macros.h` - `TORCH_CUDA_CPP_API = __declspec(dllimport)` stub

### thrust::cuda::par on Windows

`thrust::cuda::par` is unavailable in TheRock's Windows SDK. In `third_party/cubvh/include/gpu/spcumc.cuh`, replaced all occurrences with `THRUST_CUDA_PAR` macro that expands to `thrust::hip::par` when `USE_ROCM` is defined.

### HIPStreamMasqueradingAsCUDA include

hipified code calling `at::hip::getCurrentHIPStreamMasqueradingAsCUDA()` requires an explicit include of `ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h` in `third_party/cubvh/src/api_gpu.cu` on Windows -- the indirect includes don't pull it in.

### cubvh submodule

The Windows fixes are on branch `moat-windows` of `AMD-Ecosystem/cubvh`. The `.gitmodules` is updated accordingly.

### Build commands (Windows)

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set HIP_VISIBLE_DEVICES=0
set PYTORCH_ROCM_ARCH=gfx1201
set GPU_ARCHS=gfx1201
set ROCM_HOME=B:\develop\TheRock\external-builds\pytorch\.venv\Lib\site-packages\_rocm_sdk_devel
set DISTUTILS_USE_SDK=1
cd B:\develop\moat\projects\CuMesh\src
B:\develop\TheRock\external-builds\pytorch\.venv\Scripts\python.exe setup.py build_ext --inplace
```

Note: vcvars64.bat must be called first to get MSVC link.exe on PATH before Git's link.exe.

## Validation 2026-06-07

**Platform**: windows-gfx1201 (AMD Radeon RX 9070 XT, gfx1201, ROCm 7.14, TheRock PyTorch)

**GPU arch**: gfx1201 (HIP_VISIBLE_DEVICES=0, device index 0)

**Build command**: `python setup.py build_ext --inplace` via vcvars64.bat wrapper with ROCM_HOME, DISTUTILS_USE_SDK=1

**Fork HEAD**: 50df2a0d9121cf600d2b188f158121b6129fff94 (added Windows HIP support on top of validated port)

**Test suite**: 6 example scripts in `examples/`

**Results**: All tests PASS

1. `fill_holes.py`: Hole filling 34834 -> 34838 vertices, 69451 -> 69594 faces PASS
2. `remesh.py`: Dual contouring + BVH projection -> 204916 faces PASS
3. `remove_duplicate_faces.py`: Duplicate removal (no duplicates found in bunny) PASS
4. `simplify.py`: Mesh decimation 69451 -> 9865 faces PASS
5. `unify_orientations.py`: Orientation unification PASS
6. `uv_unwrap.py`: xatlas UV unwrapping -> 45018 vertices, 69451 faces PASS

All GPU operations (BVH construction, mesh simplification, hole filling, remeshing) executed successfully on gfx1201.

Linux platforms (gfx90a, gfx1100) carried forward: all new files are Windows-only (`hip_cuda_compat/` shims, `*_winhip.cu` wrappers); `setup.py` Windows additions gated behind `IS_WINDOWS` check; Linux build path unchanged.

## cubvh submodule consolidation (2026-06-19)

Re-pinned the cubvh submodule from the TRELLIS variant (`AMD-Ecosystem/cubvh@moat-windows`,
which carried a `cumesh::` namespace wrap) to the canonical ROCm branch
(`AMD-Ecosystem/cubvh@moat-port` = clean fork of ashawkey/cubvh + the deeper-stack fix),
`.gitmodules` branch updated accordingly. Submodule now at d1e7224.

The namespace wrap was replaced by building `cumesh._cubvh` with
`-fvisibility=hidden -fvisibility-inlines-hidden`: exported `cubvh::` symbols drop
189 -> 7 (residual 7 = HIP kernel stubs hipcc forces visible). CPython imports
extensions RTLD_LOCAL, and TRELLIS.2 imports only `cumesh` and never sets
RTLD_GLOBAL (checked: microsoft/TRELLIS.2 has no setdlopenflags/RTLD_GLOBAL), so
hidden visibility fully isolates the symbols without any cubvh source edit.

The Windows+HIP pybind wrapper moved out of the submodule into CuMesh's own
`src/cubvh_bindings_winhip.cu` (it depends on CuMesh's
`hip_cuda_compat/eigen_hip_compat.h`), keeping cubvh pristine/upstream-mergeable.

Validated gfx90a: full CuMesh build (all 3 extensions) + icosphere unsigned/signed
distance (max err 0.0045 vs analytic, all finite). gfx1100/gfx1201 -> revalidate.

Once followers revalidate, the `AMD-Ecosystem/cubvh@moat-windows` branch can be deleted
(no longer referenced).

## Validation 2026-06-19

**Platform**: linux-gfx1100 (AMD Radeon Pro W7800, gfx1100, ROCm 7.2, PyTorch 2.13.0a0+gitb5e90ff)

**SHA validated**: a35c791

**Delta from previous validated_sha (50df2a0)**: 1 commit -- cubvh submodule re-pinned from
moat-windows (fca73b9) to moat-port (d1e7224); setup.py adds `-fvisibility=hidden
-fvisibility-inlines-hidden` to the `_cubvh` extension; `src/cubvh_bindings_winhip.cu`
wrapper file added (Windows+HIP only, not compiled on Linux).

**codeobj_diff verdict**: `differ` (different cubvh code + visibility flags change exported
symbols) -- full GPU revalidation required.

**Build command**:
```bash
GPU_ARCHS=gfx1100 pip install /var/lib/jenkins/moat/projects/CuMesh/src --no-build-isolation -v
```

**Test suite**: 6 example scripts in `examples/` (HIP_VISIBLE_DEVICES=0)

**Results**: All 6 tests PASS

1. `simplify.py`: Mesh decimation 69451 -> 9367 faces PASS
2. `fill_holes.py`: Hole filling 34834 -> 34838 vertices, 69451 -> 69594 faces PASS
3. `remove_duplicate_faces.py`: Duplicate removal (no duplicates found) PASS
4. `unify_orientations.py`: Orientation unification PASS
5. `uv_unwrap.py`: Fast clustering (90 clusters) + xatlas UV unwrapping -> 45110 vertices PASS
6. `remesh.py`: Dual contouring + BVH projection -> 204916 faces PASS

All GPU operations (mesh simplification, hole filling, remeshing, BVH construction/queries)
executed successfully on gfx1100.
**Platform**: windows-gfx1201 (AMD Radeon RX 9070 XT, gfx1201, ROCm 7.14, TheRock PyTorch)

**GPU arch**: gfx1201 (HIP_VISIBLE_DEVICES=0, single GPU, device index 0)

**Fork HEAD**: 3a64e4a (submodule bump to cubvh@91d693a with Windows stream API fix)

**Delta from prior validated_sha (50df2a0) to revalidate target (a35c791)**: functional -- new
`src/cubvh_bindings_winhip.cu`, `setup.py` visibility flags, cubvh submodule repinned
from `fca73b9` (moat-windows) to `d1e7224` (moat-port). Required full GPU revalidation.

**Additional fix committed this session**: the new `AMD-Ecosystem/cubvh@moat-port` at `d1e7224`
had a build bug on Windows: `api_gpu.cu` calls `at::cuda::getCurrentCUDAStream()`, which
the old submodule worked around via torch's hipify pass (converting calls to
`at::hip::getCurrentHIPStreamMasqueradingAsCUDA()`). The new cubvh is compiled directly
by hipcc without torch hipify, so the unresolved `at::cuda` symbol broke the build.
Fix: inject `at::cuda::getCurrentCUDAStream()` as an inline wrapper over
`c10::hip::getCurrentHIPStreamMasqueradingAsCUDA()` in `api_gpu.cu` under `#ifdef USE_ROCM`,
before `api_gpu.h` opens `namespace at::cuda`. Committed to `AMD-Ecosystem/cubvh@moat-port`
(91d693a) and submodule bumped in CuMesh (3a64e4a). Linux path unchanged (torch hipify
substitutes calls before hipcc sees the file; the extra function is dead code there).

**Build command**: `python setup.py build_ext --inplace --force` via vcvars64.bat wrapper
with `ROCM_HOME`, `DISTUTILS_USE_SDK=1`, `GPU_ARCHS=gfx1201`, `HIP_VISIBLE_DEVICES=0`

**Test suite**: 6 example scripts in `examples/`

**Results**: All tests PASS

1. `fill_holes.py`: Hole filling 34834 -> 34838 vertices, 69451 -> 69594 faces PASS
2. `remesh.py`: Dual contouring + BVH projection -> 102396 vertices, 204916 faces PASS
3. `remove_duplicate_faces.py`: Duplicate removal (no duplicates found in bunny) PASS
4. `simplify.py`: Mesh decimation 69451 -> 9303 faces PASS
5. `unify_orientations.py`: Orientation unification PASS
6. `uv_unwrap.py`: xatlas UV unwrapping -> 45018 vertices, 69451 faces PASS

All GPU operations (BVH construction, mesh simplification, hole filling, remeshing) executed
successfully on gfx1201.

## Validation 2026-06-19 (linux-gfx90a revalidate)

**Platform**: linux-gfx90a (MI250X, gfx90a, ROCm 7.2, PyTorch 2.13.0a0+gitb5e90ff)

**Revalidate trigger**: head moved from a35c791 to 3a64e4a (cubvh submodule bump from
d1e7224 to 91d693a: Windows hipcc at::cuda::getCurrentCUDAStream() wrapper).

**Delta classification**: functional -- `advance-head` classified as unknown/non-arch-independent,
requiring full GPU revalidation.

**Build result at 3a64e4a**: FAILED. The new cubvh 91d693a added an at::cuda::getCurrentCUDAStream()
wrapper in api_gpu.cu under #ifdef USE_ROCM, declared with return type
c10::hip::HIPStreamMasqueradingAsCUDA. After torch's hipify pass generated api_gpu.hip,
c10::hip::getCurrentHIPStreamMasqueradingAsCUDA() returns c10::cuda::CUDAStream (the base
class), not the derived HIPStreamMasqueradingAsCUDA. The implicit conversion failed.

**Fix**: Changed return type to c10::cuda::CUDAStream in AMD-Ecosystem/cubvh@moat-port (dac9199)
and switched include from ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h to c10/hip/HIPStream.h
(where getCurrentHIPStreamMasqueradingAsCUDA is actually defined). CuMesh submodule bumped
to dac9199 in commit 236445c. The wrapper is dead code on Linux (hipify already rewrites all
call sites) but must compile cleanly after the hipify pass. On Windows, CUDAStream and
HIPStreamMasqueradingAsCUDA have identical layout (derived IS-A base), so behavior is unchanged.

**Build command**:
```bash
cd /var/lib/jenkins/moat/projects/CuMesh/src
GPU_ARCHS=gfx90a pip install . --no-build-isolation -v
```

**Test suite**: 6 example scripts in examples/ (HIP_VISIBLE_DEVICES=0)

**Results**: All 6 tests PASS

1. `simplify.py`: Mesh decimation 69451 -> 9793 faces PASS
2. `fill_holes.py`: Hole filling 34834 -> 34838 vertices, 69451 -> 69594 faces PASS
3. `remove_duplicate_faces.py`: Duplicate removal (no duplicates found) PASS
4. `unify_orientations.py`: Orientation unification PASS
5. `uv_unwrap.py`: Fast clustering (90 clusters) + xatlas UV unwrapping -> 45110 vertices PASS
6. `remesh.py`: Dual contouring + BVH projection -> 204916 faces PASS

All GPU operations executed successfully on gfx90a.

**SHA validated**: 236445c (cubvh submodule at dac9199)
## Hipify integration fix 2026-06-19 (porter, windows-gfx1201)

Corrected the cubvh/CuMesh Windows hipify integration. Bumped the cubvh submodule to
the shim-revert commit (AMD-Ecosystem/cubvh@0ebdc65) and reworked how the bundled cubvh
.cu sources are hipified on Windows. No manual stream shim anywhere.

Root cause (Windows-only): torch's CUDAExtension hipifies the extension's .cu sources,
but `torch/utils/hipify/hipify_python.py` `matched_files_iter` deliberately PRUNES the
entire `third_party/` subtree from its file walk (it removes "third_party" and only
re-adds "third_party/nvfuser"). cubvh lives at `third_party/cubvh/`, so its sources are
never in `all_files`; `preprocess_file_and_save_result` then returns
`[ignored, not to be hipified]` with `hipified_path=None`, and cpp_extension falls back
to compiling the RAW `api_gpu.cu`. The raw .cu calls `at::cuda::getCurrentCUDAStream()`,
which the torch (hipify v1) headers on this host do not provide -> 8 "no member named
getCurrentCUDAStream in namespace 'at::cuda'" errors. (A second, independent reason it
fails on Windows: the extra_files fallback append in `hipify()` is not unix-path
normalized, so even the explicitly-listed source misses the `all_files` membership test
on Windows backslash paths.)

This is why the standalone cubvh build works but CuMesh did not: standalone lists
`src/api_gpu.cu` (NOT under third_party/), so hipify walks it and emits `src/api_gpu.hip`;
CuMesh lists `third_party/cubvh/src/api_gpu.cu`, which the prune drops.

Why no manual shim / no version guard: hipify's OUTPUT is already correct on both
hipify versions -- it rewrites `at::cuda::getCurrentCUDAStream()` ->
`at::hip::getCurrentHIPStreamMasqueradingAsCUDA()` and `cudaStream_t` -> `hipStream_t`.
The only problem was that the hipified `.hip` was never generated/used for the bundled
cubvh on Windows. A manual `at::cuda::getCurrentCUDAStream` wrapper in cubvh's source
is both unnecessary (hipify rewrites the call sites) and harmful (it redefines the
public symbol on the hipify version that provides it, breaking that build), so it was
removed from cubvh (commit 0ebdc65).

Fix (setup.py): on Windows+HIP, `_hipify_cubvh_sources()` runs hipify with
`project_directory` pointed at the cubvh `src` dir (no third_party prune at that root),
which emits `api_gpu.hip` / `bvh.hip` with the correct stream rewrite, and substitutes
the `.hip` paths into the cubvh extension's source list. `_BuildExt` registers `.hip`
with MSVC's `_cpp_extensions` so those outputs compile (PyTorch BuildExtension only
registers .cu/.cuh; the missing .hip is the documented Windows regression fixed by
pytorch/pytorch#187665, merged to pytorch main via pytorchbot (commit c451e5413efc,
2026-06-18) -- the GitHub API still shows that PR CLOSED/mergedAt=null, but the change
is live on main; the TheRock 2.9.1 torch on this host predates it, so `_BuildExt` is
still required here). The `.hip` files are gitignored build
artifacts; the cubvh submodule source stays pristine and upstream-mergeable. Linux is
unaffected: torch hipifies cubvh in place there via the normal clang path; these
additions are gated on `IS_WINDOWS and IS_HIP`.

### Build + validation 2026-06-19 (windows-gfx1201)

Host: AMD Radeon RX 9070 XT (gfx1201), ROCm 7.14 (TheRock), torch 2.9.1+rocm7.14,
hipify 1.0.0 (v1), HIP_VISIBLE_DEVICES=0.

Clean build (deleted build/ and all stale .hip):
```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set HIP_VISIBLE_DEVICES=0
set GPU_ARCHS=gfx1201
set ROCM_HOME=B:\develop\TheRock\external-builds\pytorch\.venv\Lib\site-packages\_rocm_sdk_devel
set DISTUTILS_USE_SDK=1
cd B:\develop\moat\projects\CuMesh\src
python setup.py build_ext --inplace --force
```
Built all 3 pyd (`_C`, `_cubvh`, `_cumesh_xatlas`), exit 0. Proof the hipified path is
used: ninja compiles `third_party\cubvh\src\api_gpu.hip` (NOT api_gpu.cu); the freshly
regenerated api_gpu.hip has 9 `getCurrentHIPStreamMasqueradingAsCUDA` calls and 0
`namespace at { namespace cuda` shim blocks. With the cubvh shim removed, the raw .cu
would NOT compile on v1, so the clean build proves the hipified source is now compiled.

6 example scripts (run from examples/, PYTHONPATH=src root, HIP_VISIBLE_DEVICES=0), all PASS:
1. fill_holes.py: 34834 -> 34838 vertices, 69451 -> 69594 faces PASS
2. remesh.py: dual contouring + BVH projection -> 102396 vertices, 204916 faces PASS
3. remove_duplicate_faces.py: no duplicates found PASS
4. simplify.py: 34834 -> 5001 vertices, 69451 -> 9930 faces PASS
5. unify_orientations.py: orientation unification PASS
6. uv_unwrap.py: 90 clusters + xatlas -> 45018 vertices, 69451 faces PASS

Note: the `cmdclass` is now `_BuildExt` (was stock `BuildExtension`). Linux (hipify v2)
revalidation is handled separately on a Linux host.

## Revalidation 2026-06-19 (linux-gfx90a carry-forward)

**Platform**: linux-gfx90a (MI250X, gfx90a, ROCm 7.2)

**Revalidate trigger**: head moved from 236445c to ed3e794 (gfx1201 porter committed
"[ROCm] Compile hipified cubvh sources on Windows; drop stream shim").

**Delta 236445c -> ed3e794**:
- `setup.py`: adds `_BuildExt` class (Windows MSVC `.hip` extension registration,
  gated on `sys.platform == "win32"`) and `_hipify_cubvh_sources()` function (gated
  on `IS_WINDOWS and IS_HIP`); on Linux both are no-ops. Also refactors cubvh
  sources/include_dirs into variables (functionally equivalent); `cmdclass` changed
  to `_BuildExt` (no-op on Linux -- `_BuildExt.build_extensions` is a passthrough
  unless `sys.platform == "win32"`).
- `third_party/cubvh`: submodule pin dac9199 -> 0ebdc65 (removes the manual
  `at::cuda::getCurrentCUDAStream()` shim from api_gpu.cu; the hipify pass on Linux
  already rewrites all call sites, so the shim was dead code there).

**Method**: binary-equivalence check. Built CuMesh at both 236445c (old build already
in agent_space/cumesh_build_old) and ed3e794 (GPU_ARCHS=gfx90a, no-build-isolation,
output to agent_space/cumesh_build_new), then ran utils/codeobj_diff.py on the 3 .so
files.

**codeobj_diff results**:
- `_C.so`: identical (GPU ISA + 1008 exported symbols)
- `_cubvh.so`: identical (GPU ISA + 66 exported symbols)
- `_cumesh_xatlas.so`: indeterminate from roc-obj-ls (exit 255, "No kernel section
  found" -- CPU-only module), but byte-for-byte identical (md5: 7c6648cede4920b7c52203b55138319a)

**Verdict**: carry-forward. All GPU device code and exported symbols are identical.
The indeterminate on _cumesh_xatlas is a roc-obj-ls false negative on a CPU-only
binary; the byte identity proves no change. Delta is Windows-only; Linux build output
is unchanged.

**Action**: linux-gfx90a advanced to completed at ed3e794 via carry-forward.

## Revalidation 2026-06-19 (linux-gfx1100 carry-forward)

**Platform**: linux-gfx1100 (AMD Radeon Pro W7800, gfx1100, ROCm 7.2, PyTorch 2.13.0a0+gitb5e90ff)

**Revalidate trigger**: head moved from a35c791 to ed3e794 (three commits: cubvh Windows
stream fix, Linux build return-type fix, and Windows hipify integration with pristine cubvh).

**Delta a35c791 -> ed3e794**:
- 3a64e4a: cubvh submodule bump to 91d693a (Windows stream API fix -- Windows-only)
- 236445c: cubvh submodule bump to dac9199, correct getCurrentCUDAStream return type in
  api_gpu.cu under USE_ROCM -- this shim is dead code on Linux (hipify rewrites all call
  sites before hipcc sees the file)
- ed3e794: setup.py adds `_BuildExt` (IS_WINDOWS gated) and `_hipify_cubvh_sources()`
  (IS_WINDOWS and IS_HIP gated); cubvh submodule pin dac9199 -> 0ebdc65 (removes the
  stream shim entirely from cubvh source; shim was dead code on Linux)

**Method**: binary-equivalence check. Built CuMesh at ed3e794 (GPU_ARCHS=gfx1100,
no-build-isolation) and compared against the a35c791 build installed in site-packages
(built 2026-06-19T14:53 during the prior gfx1100 GPU validation run).

**codeobj_diff results**:
- `_C.so`: identical (GPU ISA + 1007 exported symbols)
- `_cubvh.so`: identical (GPU ISA + 64 exported symbols)
- `_cumesh_xatlas.so`: indeterminate from roc-obj-ls (CPU-only module), but md5-identical
  (70cbbc6ab9a456f7c7911bee5c25a573)

**Verdict**: carry-forward. All GPU device code and exported symbols are identical on
gfx1100. The Linux delta removes dead code (stream shim that hipify already rewrites)
and adds Windows-only IS_WINDOWS-gated setup.py additions that are no-ops on Linux.
The hipify path for vendored cubvh sources works correctly on Linux: torch hipify
walks third_party/cubvh/ and emits api_gpu.hip / bvh.hip with correct stream rewrites.

**Action**: linux-gfx1100 advanced to completed at ed3e794 via binary-equiv carry-forward.

## Validation 2026-06-19 (linux-gfx90a revalidate, cubvh e5a657a)

**Platform**: linux-gfx90a (MI250X, gfx90a, ROCm 7.2, PyTorch 2.13.0a0+git8f9a6c8)

**SHA validated**: 0cdf194 (cubvh submodule at e5a657a, consolidated single-commit ROCm port)

**Revalidate trigger**: head moved from ed3e794 to 0cdf194. Delta: cubvh submodule bumped
from 0ebdc65 to e5a657a (the consolidated single-commit cubvh ROCm port that is now cubvh's
upstream PR #33 head, rebased on cubvh upstream main 7855c00). This changes the compiled
cubvh kernel sources (api_gpu.cu, bvh.cu, spcumc.cuh, bvh.cuh) -- a functional change,
not carry-forward eligible.

**Method**: full GPU revalidation. Removed stale cubvh object files (api_gpu.o, bindings_hip.o,
bvh.o) to force recompile from the new e5a657a sources, then rebuilt and ran all 6 example
scripts on HIP_VISIBLE_DEVICES=0.

**Build command**:
```bash
# Remove stale cubvh objects from previous build (0ebdc65)
rm -f build/temp.linux-x86_64-cpython-312/third_party/cubvh/src/*.o
GPU_ARCHS=gfx90a pip install . --no-build-isolation -v
```

Build: PASS. All 3 extensions compiled (_C, _cubvh, _cumesh_xatlas). Warnings only (nodiscard,
non-trivially-copyable memcpy, abstract non-virtual destructor) -- non-fatal.

**Test suite**: 6 example scripts in examples/ (HIP_VISIBLE_DEVICES=0)

**Results**: All 6 tests PASS

1. `simplify.py`: Mesh decimation 34834 -> 4998 vertices, 69451 -> 9920 faces PASS
2. `fill_holes.py`: Hole filling 34834 -> 34838 vertices, 69451 -> 69594 faces PASS
3. `remove_duplicate_faces.py`: Duplicate removal (no duplicates found) PASS
4. `unify_orientations.py`: Orientation unification (34834 vertices, 69451 faces) PASS
5. `uv_unwrap.py`: Fast clustering (90 clusters) + xatlas UV unwrapping -> 45110 vertices PASS
6. `remesh.py`: Dual contouring + BVH projection -> 102396 vertices, 204916 faces PASS

All GPU operations (BVH construction, mesh simplification, hole filling, remeshing) executed
successfully on gfx90a with the consolidated cubvh ROCm port (e5a657a).

## Revalidation 2026-06-19 (windows-gfx1201 carry-forward, cubvh e5a657a)

**Platform**: windows-gfx1201 (AMD Radeon RX 9070 XT, gfx1201, ROCm 7.14, TheRock PyTorch)

**GPU arch**: gfx1201 (HIP_VISIBLE_DEVICES=0 confirmed: AMD Radeon RX 9070 XT)

**SHA validated**: 0cdf194 (cubvh submodule at e5a657a)

**Delta from previous validated_sha (ed3e794)**: 1 commit -- cubvh submodule bumped from
0ebdc65 to e5a657a. Inspection: only `readme.md` and `setup.py` comment text changed in
cubvh between 0ebdc65 and e5a657a; `src/` kernel sources (`api_gpu.cu`, `bvh.cu`, headers)
are byte-for-byte identical (`git diff 0ebdc65..e5a657a -- src/` produces no output).

**Method**: binary-equivalence via PE .hip_fat section SHA256. Built at 0cdf194 (vcvars64 +
ROCM_HOME + GPU_ARCHS=gfx1201, `python setup.py build_ext --inplace --force`), compared
with prior build at ed3e794 (saved to agent_space/cumesh_old_gfx1201/).

**PE .hip_fat section SHA256 hashes (gfx1201 device code)**:

- `_C.pyd`: cedc7e743b6623ade5eafb29cbc1bd88cd5fc2f0b38c16580286ab9b99463fa8 (old == new)
- `_cubvh.pyd`: 8ca9c59364d3b651999f05aca2ff3a5895200d4165affd0fdc5707d5be11ab38 (old == new)
- `_cumesh_xatlas.pyd`: no .hip_fat (CPU-only); file sizes identical (724480 bytes); same PE
  imports/exports; byte diff is PE timestamp only.

**Verdict**: carry-forward. All GPU device code is bit-identical on gfx1201. The delta is
documentation/comment-only in the compiled cubvh sources; no GPU re-run required.

**Action**: windows-gfx1201 advanced to completed at 0cdf194 via binary-equiv carry-forward.
## Revalidation 2026-06-19 (linux-gfx1100 carry-forward, cubvh e5a657a)

**Platform**: linux-gfx1100 (AMD Radeon Pro W7800, gfx1100, ROCm 7.2, PyTorch 2.13.0a0+gitb5e90ff)

**Revalidate trigger**: head moved from ed3e794 to 0cdf194. Delta: one commit -- cubvh
submodule pointer bumped from 0ebdc65 to e5a657a (the consolidated single-commit cubvh ROCm
port). Source delta in cubvh: only readme.md and cubvh's own setup.py changed; no device
sources (.cu/.cuh/.h) changed.

**Method**: binary-equivalence check. Confirmed delta (git diff 0ebdc65..e5a657a --name-only:
readme.md, setup.py only). CuMesh hipifies the vendored cubvh sources via its own setup.py
(not cubvh's); hipify correctly said "[skipped, already hipified]" for all cubvh .hip files
during build, and "ninja: no work to do" for all extensions (device objects unchanged).
Compared output .so files from GPU_ARCHS=gfx1100 build at 0cdf194 vs the prior ed3e794
build in agent_space/CuMesh-cobj-new-gfx1100 using utils/codeobj_diff.py.

**Build command**:
```bash
cd /var/lib/jenkins/moat/projects/CuMesh/src
git checkout 0cdf194 && git submodule update --init --recursive third_party/cubvh
GPU_ARCHS=gfx1100 python setup.py build_ext --build-lib /var/lib/jenkins/moat/agent_space/CuMesh-0cdf194-gfx1100
```

**codeobj_diff results** (ed3e794 vs 0cdf194):
- `_C.so`: identical (device ISA + 1007 exported symbols)
- `_cubvh.so`: identical (device ISA + 64 exported symbols)
- `_cumesh_xatlas.so`: indeterminate (roc-obj-ls exit 255, no GPU kernel section -- CPU-only
  xatlas module); byte-identical (md5: 70cbbc6ab9a456f7c7911bee5c25a573)

**Verdict**: carry-forward. All GPU device code and exported symbols are identical on gfx1100.
The cubvh submodule change from 0ebdc65 to e5a657a touches only readme.md and cubvh's own
setup.py; CuMesh's gfx1100 device code is completely unchanged.

**Action**: linux-gfx1100 advanced to completed at 0cdf194 via binary-equiv carry-forward.

## Validation 2026-06-19 (windows-gfx1101)

**Platform**: windows-gfx1101 (AMD Radeon PRO V710, gfx1101, ROCm 7.14, TheRock PyTorch)

**GPU arch**: gfx1101 (HIP_VISIBLE_DEVICES=1, device index 1; device 0 = gfx1201)

**Fork HEAD**: e5ae38f (squashed single-commit ROCm port)

**Build command**:
```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set HIP_VISIBLE_DEVICES=1
set GPU_ARCHS=gfx1101
set ROCM_HOME=B:\develop\TheRock\external-builds\pytorch\.venv\Lib\site-packages\_rocm_sdk_devel
set DISTUTILS_USE_SDK=1
cd B:\develop\moat\projects\CuMesh\src
python setup.py build_ext --inplace --force
```

Built all 3 pyd extensions (_C, _cubvh, _cumesh_xatlas) with --offload-arch=gfx1101. Warnings
only (nodiscard, non-trivially-copyable memcpy, abstract non-virtual destructor); exit 0.

**Test suite**: 6 example scripts in `examples/` (HIP_VISIBLE_DEVICES=1, PYTHONPATH=src root)

**Results**: All 6 tests PASS

1. `fill_holes.py`: Hole filling 34834 -> 34838 vertices, 69451 -> 69594 faces PASS
2. `remesh.py`: Dual contouring + BVH projection -> 102396 vertices, 204916 faces PASS
3. `remove_duplicate_faces.py`: Duplicate removal (no duplicates found in bunny) PASS
4. `simplify.py`: Mesh decimation 69451 -> 9845 faces PASS
5. `unify_orientations.py`: Orientation unification (34834 vertices, 69451 faces) PASS
6. `uv_unwrap.py`: Fast clustering (90 clusters) + xatlas UV unwrapping -> 45018 vertices PASS

All GPU operations (BVH construction, mesh simplification, hole filling, remeshing) executed
successfully on gfx1101. No TDR or wedge events observed.

## cubvh submodule repointed to upstream (2026-06-23)

ashawkey/cubvh#33 (our ROCm port) merged upstream (merge 757b913b on main), so
third_party/cubvh no longer tracks the fork. Repointed in d5c1355:
- .gitmodules url: AMD-Ecosystem/cubvh -> ashawkey/cubvh
- .gitmodules branch: moat-port -> main
- gitlink: e5a657a (jeffdaily moat-port) -> 757b913b (upstream merge commit)

757b913b's tree is identical to the previously pinned e5a657a (it is the merge
parent), so the cubvh content CuMesh builds is byte-for-byte unchanged; AMD
platforms carried forward (source-class), no GPU re-run. This supersedes the
earlier moat-windows/moat-port submodule notes above -- CuMesh now consumes
upstream cubvh directly. The PR no longer depends on a personal cubvh fork.
