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

`torch.utils.cpp_extension` sets `IS_HIP_EXTENSION` only when it finds a ROCm home and
`torch.version.hip` is non-null. `ROCM_HOME` is therefore necessary for this TheRock
Windows environment, but it is not by itself the backend selector. Set it to the TheRock
SDK devel directory when building with the ROCm torch:

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

## Validation 2026-08-09 (linux-gfx90a revalidate -> CUDA gate FAILED)

**Platform**: linux-gfx90a (MI250X, index 1 of 4, gfx90a)

**Revalidate trigger**: head moved from e5ae38f to d5c1355 (1 commit: cubvh submodule
repointed AMD-Ecosystem/cubvh@moat-port -> ashawkey/cubvh@main after the ROCm PR merged
upstream).

**Delta classification**: `python3 utils/moatlib.py classify CuMesh e5ae38f d5c1355` returned
`unknown` (submodule gitlink changes are not source-classifiable). Manual proof instead:
`git diff e5ae38f..d5c1355 --stat` touches only `.gitmodules` (url/branch text) and the
`third_party/cubvh` gitlink (`e5a657a` -> `757b913b`). `git rev-parse e5a657a^{tree}` and
`git rev-parse 757b913b^{tree}` are the IDENTICAL tree `f16858ee411cf768bdbfce2a443d3c09669d9ddf`
(757b913b is the upstream merge commit of our e5a657a ROCm PR, so its tree is byte-identical
to what gfx90a already validated at e5ae38f). CuMesh's own tree is otherwise unchanged.
This matches exactly the carry-forward already recorded for linux-gfx1100 and both windows
archs at this same delta. gfx90a's ROCm build/GPU-test side would carry forward cleanly.

**CUDA no-regression gate: FAILED.** Never previously recorded for this project (checked:
zero prior "CUDA gate" entries in this file), so it was run now rather than skipped.
`nvcc` (conda `cuda-12.8` env, `-gencode=arch=compute_80,code=sm_80`, CUDA-enabled
torch 2.11.0+cu128 in a scratch venv) compiling the `cumesh._cubvh` extension:

```
nvcc fatal   : Unknown option '-fvisibility=hidden'
```

**Root cause**: `setup.py`'s `cumesh._cubvh` extension appends
`["-fvisibility=hidden", "-fvisibility-inlines-hidden"]` unconditionally to BOTH `cxx` and
`nvcc` `extra_compile_args` (added in the 2026-06-19 "cubvh submodule consolidation" commit,
to keep the hidden-visibility cubvh symbols isolated -- see that note above). Under HIP,
the `"nvcc"` key's flags go to `hipcc`, which is clang-based and accepts `-fvisibility=`
directly, so the HIP/ROCm build never saw a problem. Real `nvcc` treats `"nvcc"` key flags
as its own command line, not the host compiler's, and rejects a raw GCC/clang-style flag
outright -- it must be wrapped `-Xcompiler=-fvisibility=hidden` (or `--compiler-options
-fvisibility=hidden`) to reach the host compiler. Confirmed this is new code, not
pre-existing breakage: `git show 12289e1:setup.py | grep fvisibility` (upstream base, before
the port) returns nothing -- the flag does not exist anywhere in the pre-port file, so there
is no upstream error to compare against and no ambiguity about which side introduced it.

This is a genuine CUDA regression (same fault class as the sibling `asm("trap;")` /
`__builtin_trap()` case noted in the validator brief: valid under HIP/clang, rejected by
nvcc), not an environmental wall. Fix belongs in `setup.py`: wrap those two flags as
`-Xcompiler=-fvisibility=hidden -Xcompiler=-fvisibility-inlines-hidden` for the `"nvcc"` key
(or gate them `if IS_HIP` for that key and add the `-Xcompiler=` wrapped form for the CUDA
`else` branch); the `"cxx"` key is fine as-is (that host-compiler list is shared and
`-fvisibility=hidden` is valid GCC/clang syntax there either way).

**Build environment for this check** (documented for repeat use): `python3.12 -m venv`
scratch env, `pip install --index-url https://download.pytorch.org/whl/cu128
torch==2.11.0+cu128` (matches the conda `cuda-12.8` toolkit major.minor), then:
```bash
export CUDA_HOME=/opt/conda/envs/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.0
export CPLUS_INCLUDE_PATH=/opt/conda/envs/cuda-12.8/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=/opt/conda/envs/cuda-12.8/targets/x86_64-linux/include:$C_INCLUDE_PATH
cd projects/CuMesh/src && python setup.py build_ext --build-lib <scratch-dir>
```
Note: the nvidia-channel conda `cuda-toolkit` package puts `cuda.h`/`cuda_runtime.h` etc.
under `targets/x86_64-linux/include/`, not directly under `include/` -- the host compiler
(g++, for non-`.cu` translation units that still `#include <cuda.h>`) needs that path added
explicitly; `nvcc` itself finds its own headers regardless.

**Verdict**: validation-failed. The ROCm/gfx90a side is carry-forward-clean, but the CUDA
no-regression gate -- required before marking any arch `completed` at this head_sha -- fails
with a build-breaking regression in `setup.py`. Sending back to the porter; this is a
one-line-class fix (wrap the two flags for the nvcc branch), not a design issue.

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

## Port fix 2026-08-09 (linux-gfx90a, CUDA no-regression failure)

Fixes the `nvcc fatal : Unknown option '-fvisibility=hidden'` recorded in
"Validation 2026-08-09". New commit 4440182 on top of d5c1355 (NOT an amend --
d5c1355 is the validated_sha of linux-gfx1100, windows-gfx1101 and
windows-gfx1201, and it is already published in upstream PR #36).

**Fix**: `setup.py` now builds the visibility flags once and derives the nvcc
form from them, instead of appending the same bare list to both compiler keys:

```python
if IS_WINDOWS and not IS_HIP:
    visibility_flags = []
    visibility_nvcc_flags = []
else:
    visibility_flags = ["-fvisibility=hidden", "-fvisibility-inlines-hidden"]
    visibility_nvcc_flags = (
        visibility_flags if IS_HIP
        else [f"-Xcompiler={flag}" for flag in visibility_flags]
    )
```

Chose `-Xcompiler=` over an `if IS_HIP` gate because the flags are WANTED on the
CUDA path too (they are what keeps cubvh's symbols from interposing), and because
`-Xcompiler=` is already this file's idiom for host-compiler flags on the nvcc key
(see the `-Xcompiler=/std:c++17` block for Windows CUDA). The guard is three-way,
not two: MSVC has no `-fvisibility` equivalent, so a Windows CUDA build drops the
flags rather than making cl.exe emit D9002 on every TU.

**Verified both ways** (the point being that one build cannot see the other's
breakage):

1. CUDA. Negative control first: with `setup.py` stashed back to d5c1355 the
   build fails with the exact recorded error on both `api_gpu.o` and `bvh.o`, so
   the repro environment is faithful. With the fix it compiles clean. nvcc 12.8,
   torch 2.11.0+cu128 in a scratch venv, `-gencode=arch=compute_80,code=sm_80`
   (recipe as documented in the 2026-08-09 validation note above). Confirmed the
   flag is not merely accepted-and-dropped: the resulting `_cubvh.so` exports only
   `PyInit__cubvh` while all 309 `cubvh::` symbols are local, i.e. `-Xcompiler=`
   really delivered the hiding.
2. ROCm. Flag lists generated under `BUILD_TARGET=rocm` are IDENTICAL before and
   after this commit (diffed by dumping each extension's `extra_compile_args`),
   so the compiled output cannot differ; no `-Xcompiler` leaks into any hipcc
   line. Rebuilt on MI250X (gfx90a, ROCm 7.2.53211, torch 2.14.0a0) and re-ran the
   six examples: simplify 69451 -> 9890 faces, fill_holes -> 69594, duplicate
   removal and orientation unification unchanged at 69451, remesh -> 204916,
   uv_unwrap 90 clusters. All pass.

Note the simplify result is 9890 faces here vs 9949 recorded on 2026-06-19. That
is not attributable to this commit -- the ROCm flag lists are provably identical
across it -- and the run is on a newer torch/ROCm than the earlier note; QEM
decimation to a face target is order-sensitive. Flagging it so the next validator
does not read the difference as a regression from this change.

**Pre-existing jargon finding, deliberately NOT fixed.** `python3
utils/jargon.py --port CuMesh` exits 1, but both hits are in d5c1355's commit
message ("jeffdaily/cubvh@moat-port"), not in the new commit. Clearing them means
rewriting a commit that three archs have validated and that is already public in
upstream PR #36. That is a person's call, not a porter's -- raising it rather than
force-pushing over published, validated history.

## Review 2026-08-09

**Reviewer**: MOAT reviewer agent (local-branch mode, `moat-port` vs `main`, base 12289e1)

**Verdict**: Request Changes

The round's commit (4440182) is correct and I could not fault it. The verdict rests on one
source line on the fork and two problems in the lesson this branch publishes to every
agent, all three fixable with new commits and none of them requiring history to be touched.
Detail on the four questions the fix was to be judged against is at the end, since the
validator needs it. A settled decision about the published commit messages is recorded
below the findings so it is not re-raised.

### Fork branch

**1. `src/cubvh_bindings_winhip.cu:1` carries an added AMD copyright line.**

```
// Copyright (c) 2026 Advanced Micro Devices, Inc.
```

The standing rule is to add no copyright or author line unless the project's house style
clearly carries per-company parallel lines for outside contributions. CuMesh's does not:
`LICENSE` and the only two files with a notice (`src/hash/api.h:1-10`,
`src/remesh/api.h:1-10`) all carry the sole author's copyright, MIT, single owner. The
port's own twelve other new files (`src/ext_winhip.cu`, `third_party/xatlas/*_winhip.cu`,
all of `hip_cuda_compat/`) carry none, so the line is inconsistent inside the port as well
as against the project. Remove it. The cost is nil: a comment-only delta is
carry-forward-eligible, so no arch re-runs.

**2. `simplify.py`'s face count has never been pinned down and the record now spans 6.5%.**

The attribution of 9890-against-9949 to order-sensitive QEM rather than to this commit is
sound, and I confirmed the mechanical half of it independently (see below): the ROCm flag
lists are identical across 4440182, so the binaries cannot differ. But the wider claim is
untested. This file records 9949, 9830, 9895, 9865, 9367, 9303, 9793, 9920, 9845, 9930 and
now 9890 for `target_faces = 10000` (`examples/simplify.py:17`), across four archs and
across trees that were proven binary-identical to each other. Nobody has established
whether that spread is run-to-run nondeterminism on a fixed binary, or something
environmental moving it, and no CUDA baseline for `simplify` exists anywhere in the record,
so "matches upstream" has never been checked for the one operation with a visible numeric
output. The cheap experiment is two consecutive runs of `examples/simplify.py` against one
installed build: if the count moves, it is inherent to the algorithm and should be written
down as such; if it is stable per binary, something else is moving it and that is worth
finding before the port is called done.

### Skill lesson (published to every agent when this branch merges)

**3. `.claude/skills/cuda-to-rocm/references/validation.md:18` ends with a false
instruction: "unset `ROCM_HOME` so `IS_HIP_EXTENSION` resolves False".**

It does not. `IS_HIP_EXTENSION = bool(ROCM_HOME is not None and torch.version.hip is not
None)`, and `_find_rocm_home()` falls back through `_rocm_sdk_core`, `hipcc` on PATH, and
`/opt/rocm` when neither `ROCM_HOME` nor `ROCM_PATH` is set. Checked on this host:

```
$ env -u ROCM_HOME -u ROCM_PATH python3 -c "import torch.utils.cpp_extension as c; print(c.ROCM_HOME, c.IS_HIP_EXTENSION)"
/opt/rocm-7.2.1 True
```

An agent following that clause on a ROCm host would get a HIP build and misread the result.
What actually makes the recipe work is the thing the entry already prescribes, the
CUDA-only torch wheel in the scratch venv, whose `torch.version.hip` is None. Drop the
clause or replace it with the real lever; for a project with CuMesh's `BUILD_TARGET`
switch, `BUILD_TARGET=cuda` forces it directly (`setup.py:79-80`). The same imprecision is
in this file at the "IS_HIP detection on Windows" note above, which is worth correcting in
the same edit.

**4. The porter-facing half of the lesson is filed where only a validator will read it.**

All five bullets sit under validation.md's PR-prep nvcc-check gate. The diagnostic half
belongs there. The half that would have PREVENTED the bug does not: that the `"nvcc"` and
`"cxx"` keys of `extra_compile_args` are different command lines, that a GCC/clang flag on
the nvcc key needs `-Xcompiler=`, and that the guard is three-way. That is something a
porter learns while editing a torch extension's `setup.py`, and `references/strategy-b-torch.md`
is the file they open; it currently says nothing about compile flags at all beyond
`strategy-b-torch.md:8`. Put the rule there and leave the pointer in validation.md, per the
filing rule. The porter who wrote the original unconditional append would not have opened
validation.md.

### Settled: the published commit messages stand (maintainer decision, 2026-08-09)

Not a finding, and not open. `python3 utils/jargon.py --port CuMesh` exits 1 on `moat-port`
and `moat` in commit `d5c1355`'s message. The maintainer has decided that line stays: the
commit is public in upstream PR #36 and three architectures validated at it, and the
porter's refusal to rewrite published, validated history was correct. The allowlist in
`config/jargon.toml` will not be extended to clear it either. Both escapes are closed
deliberately, so a rewrite, a squash and an allowlist entry are all off the table rather
than merely unattempted.

Known consequence, recorded as state: `upstream.py --review` scans the whole branch's
commit messages, so CuMesh cannot open a review PR through the tooling while that line
stands, and the review PR is the only route to the upstream PR.

Two further observations about the same frozen commits, recorded so a later reader does not
mistake them for new problems and reach for the same closed remedies. `d5c1355`'s title is
77 characters against the 72 limit. `e5ae38f`'s Test Plan runs `python test/simplify.py`
and five siblings, and there is no `test/` directory at any commit on this branch or at the
base (`git ls-tree main` lists `examples`); the scripts are `examples/simplify.py` and so
on.

One substantive fact about the branch that its own commit messages understate, useful to
anyone reading PR #36 rather than something to act on here: `e5ae38f` says every
AMD-specific change is gated behind USE_ROCM / IS_HIP_EXTENSION / HIP-platform macros and
that the NVIDIA build compiles the same sources as before, but three changes touch the CUDA
path ungated. The `_cubvh` visibility flags apply to the CUDA build too (`setup.py:231,239`,
which is what the nvcc failure exposed); `-std=c++17` becomes `-std=c++20` for Linux CUDA
(`setup.py:113-120` against the base file's `else` branch); and `third_party/cubvh` moves
off the maintainer's own `JeffreyXiang/cubvh@trellis.2` pin onto `ashawkey/cubvh@main`, so
the NVIDIA build compiles different cubvh sources and the one commit that fork carried over
its merge-base ("wrap cubvh API with cumesh namespace to avoid symbol conflicts", ce92267)
is dropped, its job taken over by hidden visibility. All three are defensible on the
merits: C++20 is inside the documented floor (`README.md:15` requires CUDA >= 12.4) and
helps CUDA users on newer torch as much as ROCm ones, the visibility flags are wanted on
CUDA, and `README.md:71` already credits ashawkey as cubvh's origin. The reasoning for the
namespace-wrap substitution lives in `setup.py:228-230` and in the 2026-06-19 note above.

### On the four questions this round was to be judged against

Recorded because the validator inherits them.

- **`-Xcompiler=` over an `IS_HIP` gate**: right call, and the idiom claim checks out. The
  `-Xcompiler=/std:c++17` block is upstream's own, at lines 42-45 of the base `setup.py`
  (`git show main:setup.py`), not something the port introduced, so the file's convention
  argument holds. Gating the flags off for CUDA would have left the CUDA build without the
  symbol isolation that replaced the dropped namespace wrap, which is the wrong trade.
- **The three-way guard**: correct on all four arms. I dumped each extension's
  `extra_compile_args` at d5c1355 and 4440182 with `setuptools.setup` and `CUDAExtension`
  stubbed, forcing `platform.system` for the Windows arms. Linux+HIP and Windows+HIP are
  byte-identical across the commit; Linux CUDA changes only the two nvcc-key entries from
  bare to `-Xcompiler=`-wrapped and leaves the cxx key bare, which is right for g++;
  Windows CUDA drops both, leaving a flag list that is upstream's plus the port's unrelated
  additions. No configuration that can use the flags loses them, and MSVC is never handed
  a `-f` flag on either key.
- **Additive on CUDA**: yes for this commit. The ROCm claim is not merely asserted, it is
  reproducible: `BUILD_TARGET=rocm` produces identical dumps either side, so the three
  completed archs face a provable no-op and this commit cannot be the cause of any numeric
  difference. (The port's other CUDA-path changes predate this commit; see finding 2.)
- **Evidence that the flag takes effect**: the right evidence. `cubvh::` symbols come
  mostly from `bvh.cu` and `api_gpu.cu`, the two translation units on the nvcc key, so a
  `-Xcompiler=` that failed to arrive would leave them at default visibility and exported,
  as the ROCm-side 189-to-7 measurement in the 2026-06-19 note shows happens without the
  flags. 309 local `cubvh::` symbols against `PyInit__cubvh` alone in the dynamic table
  distinguishes "delivered" from "accepted and ignored", which is what was asked.

Fault classes: none apply and I re-checked rather than inheriting the plan's claim. No
`__shfl*`, `__ballot`, `__activemask` or `warpSize` anywhere in `src/` or in the vendored
cubvh; no texture objects, `cudaArray` or `tex2D`; no hardcoded 32 in a lane context. The
CUB-to-hipCUB swap goes through torch hipify, and the one place it cannot reach
(`src/clean_up.cu:245-255`, `rocprim::tuple` for the `int3_decomposer`) is guarded on
`__HIP_PLATFORM_AMD__` and leaves the CCCL spelling on the CUDA side. Fork tree is clean;
no `Co-Authored-By` trailer, no non-ASCII, no AMD-internal account or host references in
any message or added line.

## Port follow-up 2026-08-11 (linux-gfx942)

Resolved the actionable findings from the 2026-08-09 review at fork HEAD
`89e63244d859861fee80901f144fa8b004c6dabe`, which is present on `origin/moat-port`.
The fork commit removes the added company copyright notice from
`src/cubvh_bindings_winhip.cu`; it is comment-only and the fork tree is clean. The
control-plane change adds the preventive `CUDAExtension` compiler-key rule to
`references/strategy-b-torch.md`, leaves the diagnostic pointer in `validation.md`, and
corrects the false claim that unsetting or setting `ROCM_HOME` alone determines
`IS_HIP_EXTENSION`. The matching Windows note above now states both required inputs:
a discovered ROCm home and non-null `torch.version.hip`.

**Platform and toolchain**: one process pinned to device 0 of eight AMD Instinct MI300X
GPUs (`gfx942:sramecc+:xnack-`). PyTorch `2.14.0a0+git7d05abc`, HIP
`7.14.60850`, AMD clang `23.0.0git` (`46fcb339`), Python 3.12.

**Build command**:

```bash
utils/timeit.sh CuMesh compile -- bash -lc \
  'cd projects/CuMesh/src && HIP_VISIBLE_DEVICES=0 GPU_ARCHS=gfx942 \
  python3 -m pip install . --no-build-isolation -v'
```

Build PASS in 118.603 seconds. hipcc compiled the GPU translation units with
`--offload-arch=gfx942`; all three extensions (`_C`, `_cubvh`, `_cumesh_xatlas`) built and
installed. The compiler emitted only the previously documented warnings (ignored
`nodiscard` HIP results, non-trivial `memcpy`, abstract non-virtual destructor, and
visibility/VLA warnings).

The first example invocation stopped before any GPU work because `trimesh` was not
installed (`ModuleNotFoundError`). This was an example-environment dependency, not a
source or runtime failure. Installed `trimesh==5.0.0` with:

```bash
python3 -m pip install trimesh
```

**Fixed-binary simplification experiment**: both runs loaded the same installed extension,
`/opt/conda/envs/py_3.12/lib/python3.12/site-packages/cumesh/_C.cpython-312-x86_64-linux-gnu.so`,
and ran consecutively on the same pinned GPU:

```bash
utils/timeit.sh CuMesh test -- bash -lc \
  'cd projects/CuMesh/src/examples && for run in 1 2; do \
  echo RUN=$run; HIP_VISIBLE_DEVICES=0 python3 simplify.py; done'
```

Both runs PASS. Run 1 produced 4,951 vertices / 9,833 faces; run 2 produced 4,926
vertices / 9,778 faces. Therefore the previously recorded output spread is genuine
run-to-run nondeterminism in the simplifier even for a fixed binary, device, input, and
environment; the face count is not a stable regression oracle. The invariant is successful
completion near the requested 10,000-face target with a valid mesh.

**Complete GPU example suite**:

```bash
cd projects/CuMesh/src/examples
HIP_VISIBLE_DEVICES=0 python3 simplify.py
HIP_VISIBLE_DEVICES=0 python3 fill_holes.py
HIP_VISIBLE_DEVICES=0 python3 remove_duplicate_faces.py
HIP_VISIBLE_DEVICES=0 python3 unify_orientations.py
HIP_VISIBLE_DEVICES=0 python3 remesh.py
HIP_VISIBLE_DEVICES=0 python3 uv_unwrap.py
```

All six PASS (each exit 0) in one wrapped 27.004-second run:

1. `simplify.py`: 34,834 vertices / 69,451 faces -> 4,970 vertices / 9,863 faces.
2. `fill_holes.py`: 34,834 / 69,451 -> 34,838 / 69,594.
3. `remove_duplicate_faces.py`: 34,834 / 69,451 -> 34,834 / 69,451.
4. `unify_orientations.py`: 34,834 / 69,451 -> 34,834 / 69,451.
5. `remesh.py`: BVH construction, dual contouring and projection -> 102,396 vertices /
   204,916 faces.
6. `uv_unwrap.py`: 90 clusters, xatlas charts/packing -> 45,110 vertices / 69,451 faces.

Integrity gate: `git -C projects/CuMesh/src status --porcelain` is empty; generated hipify,
build, wheel, cache and example-output files are ignored artifacts. Local fork HEAD and
`origin/moat-port` both resolve to `89e63244d859861fee80901f144fa8b004c6dabe`.
`python3 utils/jargon.py --port CuMesh` still reports only the two settled hits in frozen
commit `d5c1355` (`moat-port` and `moat` on the same message line); the maintainer decision
recorded in the review above explicitly leaves that published, validated history unchanged.

## Review 2026-08-11 (linux-gfx942)

**Reviewer**: MOAT reviewer agent (local-branch mode, `moat-port` vs `main`, base
`12289e1062f0603f2f0d0771b02e1395d247f26f`)

**Verdict**: Approve

No actionable findings at fork HEAD `89e63244d859861fee80901f144fa8b004c6dabe`
or in the accompanying Strategy B and validation guidance.

## Validation 2026-08-11 (linux-gfx942, independent validator)

**Verdict**: validation-failed. The fresh ROCm build and all six real GPU examples pass
on gfx942, but the final documentation gate fails: the upstream-visible README documents
only a CUDA-enabled PyTorch installation and CUDA Toolkit prerequisite, with no ROCm
prerequisite, backend selection, or `GPU_ARCHS=gfx942` source-build recipe. Per the
validator role this must return to the porter; the validator did not edit fork source.

**Commit and platform**: validated fork HEAD
`89e63244d859861fee80901f144fa8b004c6dabe`; local `HEAD` and `origin/moat-port` resolve
to that exact SHA. One process was pinned with `HIP_VISIBLE_DEVICES=0` to an AMD Instinct
MI300X HF reporting `gfx942:sramecc+:xnack-`. Python 3.12.13, PyTorch
`2.14.0a0+git7d05abc`, HIP `7.14.60850`, and AMD clang `23.0.0git`
(`46fcb339fb61119b337f973c7ca9e710a319fdd0`).

**Independent clean build**: used a new out-of-tree build base so no porter/reviewer
objects or installed extensions were reused:

```bash
utils/timeit.sh CuMesh compile -- bash -lc \
  'cd /var/lib/jenkins/moat/projects/CuMesh/src && \
  HIP_VISIBLE_DEVICES=0 GPU_ARCHS=gfx942 python3 setup.py build \
  --build-base /var/lib/jenkins/moat/agent_space/CuMesh-validator-gfx942-89e63244 \
  --force'
```

PASS in 112.604 seconds. All three extensions (`_C`, `_cubvh`, and the CPU-only
`_cumesh_xatlas`) compiled and linked in the fresh build directory. Every HIP translation
unit was compiled with `--offload-arch=gfx942`. The warnings were the already-recorded
ignored `nodiscard` HIP results, non-trivial `memcpy`, abstract non-virtual destructor,
visibility, and VLA warnings; there were no errors.

**Real GPU suite**: copied the examples to the scratch build directory, set `PYTHONPATH`
to the fresh build's `lib.linux-x86_64-cpython-312`, verified the imported `cumesh` path,
and ran the project's complete six-example suite:

```bash
utils/timeit.sh CuMesh test -- bash -lc 'set -euo pipefail
export HIP_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1
export PYTHONPATH=/var/lib/jenkins/moat/agent_space/CuMesh-validator-gfx942-89e63244/lib.linux-x86_64-cpython-312
cd /var/lib/jenkins/moat/agent_space/CuMesh-validator-gfx942-89e63244/examples-run
python3 -c "import cumesh, torch; print(cumesh.__file__); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).gcnArchName)"
for script in simplify.py fill_holes.py remove_duplicate_faces.py unify_orientations.py remesh.py uv_unwrap.py; do
  python3 "$script"
done'
```

All 6/6 PASS in 29.719 seconds:

1. `simplify.py`: 34,834 vertices / 69,451 faces -> 4,894 vertices / 9,713 faces.
2. `fill_holes.py`: 34,834 / 69,451 -> 34,838 / 69,594.
3. `remove_duplicate_faces.py`: unchanged at 34,834 / 69,451.
4. `unify_orientations.py`: unchanged at 34,834 / 69,451.
5. `remesh.py`: BVH construction, sparse grid, dual contouring, and projection ->
   102,396 vertices / 204,916 faces.
6. `uv_unwrap.py`: 90 clusters, xatlas charts/packing -> 45,110 vertices / 69,451 faces.

The CPU-only xatlas extension compiled and its full chart/pack path ran inside
`uv_unwrap.py`. Against upstream, `third_party/xatlas` differs only by two Windows-HIP
wrapper files; its Linux CPU sources are unchanged, so this is also the non-GPU
no-regression evidence. CuMesh has no formal test runner beyond these examples.

**CUDA no-regression gate**: not repeated. The CUDA compile/link gate passed with nvcc
12.8 at commit `4440182` as recorded in the port-fix note above. The complete fork delta
from `4440182` to this HEAD removes only two comment lines from the Windows-HIP binding;
no CUDA or build behavior changed, so the recorded gate carries forward under the
validator shortcut.

**Final gates**:

- Integrity PASS: `git -C projects/CuMesh/src status --porcelain` is empty after the
  build and tests; all generated files are ignored. The cubvh submodule is at
  `757b913bfbf19ed65e3a379d159391a8e29efa0f` as pinned.
- Jargon reports the same two hits in frozen commit `d5c1355` (`moat-port` and `moat`).
  The maintainer decision recorded in the 2026-08-09 review explicitly leaves that
  already-published history unchanged, so this is a settled exception rather than a new
  validator finding.
- Documentation FAIL: `README.md` says PyTorch must have CUDA support and requires CUDA
  Toolkit >= 12.4, then gives only `pip install CuMesh --no-build-isolation`. It contains
  no `ROCm` or `HIP` occurrence and does not document `BUILD_TARGET=rocm`, `GPU_ARCHS`,
  or the ROCm source-build command. Add the ROCm path next to the CUDA build in the
  project's house style, in a new fork commit, then revalidate linux-gfx942 at that HEAD.
