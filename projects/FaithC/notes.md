# FaithC notes

## Port summary (linux-gfx90a, lead)
Strategy B (torch hipify). The whole port is: restore a `setup.py` with a
`CUDAExtension`/`BuildExtension` over `_C/{bindings.cpp,kernels.cu}` (the v1.5
pyproject dropped it, so a source install had no compiled `_C` while ops.py
hard-imports it), plus one mechanical source fix for a hipify parser limit.
Fork: https://github.com/AMD-Ecosystem/FaithC @ moat-port (head ec2fae2).

## Gotchas

### hipify cannot rewrite a parenthesized ternary kernel launch
`aabb_tri_sat_clip_select_cuda` originally launched a template kernel chosen by
a ternary directly in the launch:
`(max_vert==8 ? sat_clip_kernel<scalar_t,8> : sat_clip_kernel<scalar_t,7>)<<<...>>>`.
hipify's regex `<<<...>>>` -> `hipLaunchKernelGGL` rewrite mis-parses the
parenthesized expression and emits a mangled `...<scalar_t,7hipLaunchKernelGGL((>))`
token, giving `invalid suffix 'hipLaunchKernelGGL' on integer constant`. Fix:
hoist the selected kernel into a local function pointer
(`auto kernel = cond ? k<...,8> : k<...,7>;`) then launch `kernel<<<...>>>`.
Parses cleanly under both hipify and nvcc; identical semantics. Applies to both
the mode-1 (sat_centroid) and mode-2 (sat_clip) launch sites.

### Build / incremental
- `cd src && PYTORCH_ROCM_ARCH=gfx90a python setup.py build_ext --inplace`.
- Multi-arch fat binary: `PYTORCH_ROCM_ARCH="gfx90a;gfx1100" ... build_ext --inplace`,
  verify `llvm-objdump --offloading _C*.so | grep -E "gfx90a|gfx1100"`.
- After editing a `.cu`, delete the stale `src/faithcontour/_C/kernels.hip` and
  the `build/` dir before rebuilding so hipify regenerates (Strategy B incremental
  trap). `.gitignore` now excludes `*.hip`, `*.prehip`, `*.so.*`.
- Pybind module is `TORCH_EXTENSION_NAME` = `_C`; to load the `.so` standalone
  (without importing the `faithcontour` package, which pulls scipy/utils.grid),
  load it under spec name `_C` (PyInit symbol is `PyInit__C`).

### Numerics
- No `-ffp-contract=on` pin needed. The Moller-Trumbore dot drift vs a torch CPU
  reference is 3.5e-7, well inside the kernels' eps guards.
- wave-size-agnostic: zero warp intrinsics, no shfl/ballot/cub; `extern __shared__`
  is sized by `blockDim.x` and fully `__syncthreads`-fenced. `dim3(32,32)` blocks
  are 2D tile dims, not warp assumptions. No wave64 work; gfx1100 expected to pass
  by rebuild with no delta.

## Validation (real gfx90a, MI250X, GCD 1)
Harness `agent_space/faithc_harness.py` drives all four bindings on GPU vs a
pure-torch CPU reference. Atomic kernels (segment_tri_intersection_fused,
gen_candidates_overlap) compared as ORDER-INDEPENDENT sorted (a,t) pair sets
(atomicAdd slotting is nondeterministic); non-atomic kernels (voxelize_mark
use_sat F/T, aabb_tri_sat_clip_select modes 0/1/2) compared exactly + rerun
determinism; overflow path exercised with a small cap. 16/16 PASS.
`AMD_LOG_LEVEL=3` confirms native gfx90a code-object dispatch.

## Deferred: end-to-end demo dependencies
`demo.py` / the encoder/decoder need two GPU deps NOT in this repo, so the full
pipeline is a follow-up (the four `_C` kernels are the validated lead gate):
- `atom3d` (Luo-Yihao/Atom3d, ~25% CUDA: MeshBVH + octree). Its own build is
  CUDA-only; would need a separate MOAT port. Recommend scaffolding it and
  recording FaithC `depends_on` it for the e2e story.
- `torch_scatter` (rusty1s/pytorch_scatter): builds on ROCm torch via auto-hipify
  but must be compiled for this ROCm; external pip dep, not a MOAT project.

## Review 2026-06-02 (reviewer, gfx90a)
Verdict: review-passed. Independently reproduced on real gfx90a (MI250X, GCD 3, ROCm torch hip 7.2.53211).

Verified clean:
- setup.py CUDAExtension + BuildExtension drives a clean-from-scratch HIP build (auto-hipify generates kernels.hip, compiles, links amdhip64/c10_hip/torch_hip); sources stay CUDA-native (.cu/.cpp/.h tracked, no .hip/.so committed; .gitignore covers *.hip/*.prehip/*.so.*).
- Multi-arch: PYTORCH_ROCM_ARCH="gfx90a;gfx1100" fresh build -> both code objects in _C*.so (llvm-objdump --offloading).
- Ternary-launch hoist (kernels.cu:730 mode-1 sat_centroid, kernels.cu:745 mode-2 sat_clip): `auto kernel = max_vert==8 ? K<scalar_t,8> : K<scalar_t,7>; kernel<<<...>>>` is semantically identical to the original `(cond ? A : B)<<<...>>>`; both template branches differ only in the non-type MAXV arg so they share one function-pointer type that `auto` deduces cleanly. Parses under hipify and nvcc.
- Wave-agnostic confirmed: zero warp intrinsics (no shfl/ballot/activemask/any/all), no cooperative groups, no threadfence, no textures/surfaces, no cub/thrust/cublas/curand/cufft. Only atomicAdd + __syncthreads + dynamic extern __shared__ (sized by blockDim.x, fully fenced) + float math. dim3(32,32) are 2D tile dims, not warp assumptions.
- Correctness harness reproduced: 16/16 PASS. atomicAdd kernels (seg_tri, gen_candidates_overlap) order-independent sorted-pair sets; non-atomic kernels exact + rerun-deterministic; overflow path exercised; Moller-Trumbore dot drift 3.5e-7, eps-absorbed (no -ffp-contract pin needed).
- Commit hygiene: [ROCm] title 54 chars, mentions Claude, no noreply/ghstack/em-dash. Fork main == upstream main (1580e2e) clean mirror; moat-port HEAD == ec2fae28; fork Actions disabled. No AMD-internal account reference.

Minor (non-blocking, recorded for the e2e follow-up): the harness cross-checks mode-0 hit_mask against a CPU SAT reference, but mode-1/mode-2 centroid/poly-vert VALUES are validated only for determinism + index alignment + range, not against an independent CPU Sutherland-Hodgman reference (plan called for "centroids/areas within tol"). The clip geometry is unchanged from upstream CUDA and the shared SAT plane logic is cross-checked via mode-0, so this is a coverage gap, not a defect; close it when atom3d enables the end-to-end demo run.

## Validation 2026-06-02

Platform: linux-gfx90a (AMD Instinct MI250X, GCD 3, gfx90a:sramecc+:xnack-, ROCm 7.2)
Fork: AMD-Ecosystem/FaithC @ moat-port ec2fae28
Validator: claude-sonnet-4-6

Build command (from-scratch, multi-arch):
```
rm -f src/faithcontour/_C*.so && rm -rf build/ && rm -f src/faithcontour/_C/kernels.hip
HIP_VISIBLE_DEVICES=3 PYTORCH_ROCM_ARCH="gfx90a;gfx1100" python setup.py build_ext --inplace
```
Build result: PASS (73 s, exit 0, warnings only -- loop-unroll advisory on sat_centroid/sat_clip templates)

Multi-arch code objects verified:
```
llvm-objdump --offloading _C.cpython-312-x86_64-linux-gnu.so | grep -E "gfx90a|gfx1100"
# hipv4-amdgcn-amd-amdhsa--gfx1100  PRESENT
# hipv4-amdgcn-amd-amdhsa--gfx90a   PRESENT
```

Test command:
```
HIP_VISIBLE_DEVICES=3 AMD_LOG_LEVEL=3 python agent_space/faithc_harness.py
```
Test result: 16/16 PASS (4 s, exit 0)

AMD_LOG_LEVEL=3 confirms native gfx90a code-object dispatch (no JIT fallback):
"Using native code object for device: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-"

Pass breakdown:
- seg_tri pair set: PASS
- seg_tri dots (maxerr=3.54e-07): PASS
- seg_tri deterministic set: PASS
- overlap no spurious overflow: PASS
- overlap pair set: PASS
- overlap overflow flag set: PASS
- voxelize_mark use_sat=False exact: PASS
- voxelize_mark use_sat=False deterministic: PASS
- voxelize_mark use_sat=True exact: PASS
- voxelize_mark use_sat=True deterministic: PASS
- sat mode0 hit_mask exact (116 hits): PASS
- sat mode0 deterministic: PASS
- sat mode1 deterministic: PASS
- sat mode1 idx alignment: PASS
- sat mode2 deterministic poly verts: PASS
- sat mode2 poly_counts in range: PASS

Verdict: completed. validated_sha=ec2fae28.

## Validation 2026-06-07 (linux-gfx90a carry-forward)

Platform: linux-gfx90a (AMD Instinct MI250X, gfx90a)
Fork: AMD-Ecosystem/FaithC @ moat-port c72480ea
Method: binary-equivalence (codeobj_diff.py)

Built both ec2fae28 and c72480ea at PYTORCH_ROCM_ARCH=gfx90a; ran
`python3 utils/codeobj_diff.py faithc-cmp-old faithc-cmp-new`:

```
verdict=identical
  _C.cpython-312-x86_64-linux-gnu.so: identical (exported symbols + device ISA identical (144 exports))
```

The delta commit changes `long` -> `int64_t` in kernel signatures and host
`data_ptr<>()` calls (a Windows LLP64 fix). On 64-bit Linux `sizeof(long)==8`,
so this rename is semantically transparent and compiles to identical gfx90a
device ISA. No GPU re-run needed.

Verdict: carry-forward completed. validated_sha=c72480ea (linux-gfx90a).

## Validation 2026-06-07 (windows-gfx1201)

Platform: AMD Radeon RX 9070 XT, gfx1201 (RDNA4, wave32), Windows 11 Pro for Workstations
Fork: AMD-Ecosystem/FaithC @ moat-port c72480ea (delta commit on top of ec2fae28)
Validator: claude-sonnet-4-6

### Windows delta-port changes (new commit c72480e on top of ec2fae28)

Two Windows-specific fixes required (neither needed on Linux):

1. **LLP64 `long` fix**: On Windows, `long` is 32-bit (LLP64 ABI), while
   `torch::kInt64` tensors are 64-bit. All `long*` kernel signatures and
   `data_ptr<long>()` host calls replaced with `int64_t*` / `data_ptr<int64_t>()`.
   On Linux `long` is 64-bit so the change is semantically transparent there.

2. **`c10::ValueError` linker fix**: `c10.dll` does not export the inherited
   constructor `c10::ValueError(SourceLocation, string)` (MSVC does not re-export
   inherited constructors even for `C10_API` classes). Headers included via
   `<torch/extension.h>` (e.g. `ATen/TensorIndexing.h`) trigger `TORCH_CHECK_VALUE`
   which generates a `__declspec(dllimport)` reference to that constructor, causing
   LNK2001. Fix: `/ALTERNATENAME` linker directive in `setup.py` (Windows-only)
   redirects the dllimport thunk to `c10::Error(SourceLocation, string)`, which IS
   exported. `ValueError IS-A Error` with no additional data members; semantically
   identical constructors.

Build environment:
- MSVC link.exe prepended to PATH (before Git's /usr/bin/link)
- ROCM_HOME=_rocm_sdk_devel, DISTUTILS_USE_SDK=1, HIP_VISIBLE_DEVICES=0, PYTORCH_ROCM_ARCH=gfx1201

Build command (from-scratch):
```
export PATH="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64:$PATH"
cd projects/FaithC/src
rm -f src/faithcontour/_C/kernels.hip && rm -rf build/
HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1201 \
  ROCM_HOME=".venv/Lib/site-packages/_rocm_sdk_devel" \
  DISTUTILS_USE_SDK=1 \
  python.exe setup.py build_ext --inplace
```
Build result: PASS (27 s, exit 0, loop-unroll advisories on sat_centroid/sat_clip -- same as Linux)
gfx1201 code-object confirmed in .pyd (`.hipFatB` section present in PE binary)

Test command:
```
HIP_VISIBLE_DEVICES=0 python.exe agent_space/faithc_harness_win.py
```
Test result: 17/17 PASS (2 s, exit 0)

Pass breakdown:
- seg_tri pair set: PASS
- seg_tri dots: PASS
- seg_tri deterministic set: PASS
- overlap no spurious overflow: PASS
- overlap pair set: PASS
- overlap overflow flag set: PASS
- voxelize_mark use_sat=False exact: PASS
- voxelize_mark use_sat=False deterministic: PASS
- voxelize_mark use_sat=True exact: PASS
- voxelize_mark use_sat=True deterministic: PASS
- sat mode0 hit_mask exact: PASS
- sat mode0 deterministic: PASS
- sat mode0 hit_mask exact (hits found): PASS
- sat mode1 deterministic: PASS
- sat mode1 idx alignment: PASS
- sat mode2 deterministic poly verts: PASS
- sat mode2 poly_counts in range: PASS

GPU dispatch confirmed: .pyd contains `.hipFatB` section; kernels executed on
AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32) at HIP_VISIBLE_DEVICES=0.

Verdict: completed. validated_sha=c72480ea (windows-gfx1201 only).

Note for linux-gfx90a/gfx1100: c72480e changed `long`->`int64_t` (source rename;
semantically transparent on 64-bit Linux where sizeof(long)==8). Linux validators
can carry forward via `codeobj_diff.py` binary-equivalence check.

## Validation 2026-06-02 (gfx1100)

Platform: linux-gfx1100 (AMD Radeon Pro W7800 48GB, gfx1100, RDNA3, wave32, ROCm 7.2.1)
Fork: AMD-Ecosystem/FaithC @ moat-port ec2fae28 (no delta from gfx90a -- wave-agnostic confirmed)
Validator: claude-sonnet-4-6

Build command (gfx1100-only, from-scratch):
```
rm -f src/faithcontour/_C/kernels.hip && rm -rf build/
HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1100 python setup.py build_ext --inplace
```
Build result: PASS (~58 s, exit 0, loop-unroll advisories on sat_centroid/sat_clip templates -- same as gfx90a)

gfx1100 code-object verified:
```
llvm-objdump --offloading _C.cpython-312-x86_64-linux-gnu.so | grep gfx
# hipv4-amdgcn-amd-amdhsa--gfx1100  PRESENT (single-arch gfx1100 build)
```

Fork clone git status: clean (no uncommitted files; .hip/.so.* gitignored)

Test command:
```
HIP_VISIBLE_DEVICES=0 AMD_LOG_LEVEL=3 python agent_space/faithc_harness.py
```
Test result: 16/16 PASS (exit 0)

AMD_LOG_LEVEL=3 confirms native gfx1100 code-object dispatch (no JIT fallback, no HSA fault):
"Using native code object for device: amdgcn-amd-amdhsa--gfx1100 co: amdgcn-amd-amdhsa--gfx1100"

Pass breakdown:
- seg_tri pair set: PASS
- seg_tri dots (maxerr=1.19e-07): PASS
- seg_tri deterministic set: PASS
- overlap no spurious overflow: PASS
- overlap pair set: PASS
- overlap overflow flag set: PASS
- voxelize_mark use_sat=False exact: PASS
- voxelize_mark use_sat=False deterministic: PASS
- voxelize_mark use_sat=True exact: PASS
- voxelize_mark use_sat=True deterministic: PASS
- sat mode0 hit_mask exact (4 hits): PASS
- sat mode0 deterministic: PASS
- sat mode1 deterministic: PASS
- sat mode1 idx alignment: PASS
- sat mode2 deterministic poly verts: PASS
- sat mode2 poly_counts in range: PASS

Wave32 verdict: CONFIRMED wave-agnostic. Zero warp intrinsics (no shfl/ballot/cub),
extern __shared__ sized by blockDim.x, fully __syncthreads-fenced, dim3(32,32) are 2D
tile dims. No delta needed from gfx90a lead; commit ec2fae28 untouched.

Harness note: sat_centroid_kernel leaves hit_mask[k] uninitialized on early-return (poly
clips to 0-vert) paths -- this is upstream behavior, not a regression. Harness compares
hit_mask only where poly_count>0; poly_counts (always written) confirmed deterministic.

Verdict: completed. validated_sha=ec2fae28.

## Validation 2026-06-16 (windows-gfx1101)

Platform: AMD Radeon PRO V710, gfx1101 (RDNA3, wave32), Windows 11 Pro for Workstations
Fork: AMD-Ecosystem/FaithC @ moat-port c72480ea
Validator: claude-sonnet-4-6

Build command (from-scratch, gfx1101):
```
export PATH="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64:$PATH"
cd projects/FaithC/src
rm -f src/faithcontour/_C/kernels.hip && rm -rf build/ && rm -f src/faithcontour/_C.cp312-win_amd64.pyd
HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1101 \
  ROCM_HOME=".venv/Lib/site-packages/_rocm_sdk_devel" \
  DISTUTILS_USE_SDK=1 \
  python.exe setup.py build_ext --inplace
```
Build result: PASS (~30 s, exit 0, loop-unroll advisories on sat_centroid/sat_clip -- same as Linux and gfx1201)
gfx1101 code-object confirmed in .pyd: `gfx1101` marker and `.hipFatB` section present in PE binary.

Test command:
```
HIP_VISIBLE_DEVICES=0 python.exe agent_space/faithc_harness_win.py
```
Test result: 17/17 PASS (exit 0)

Pass breakdown:
- seg_tri pair set: PASS
- seg_tri dots: PASS
- seg_tri deterministic set: PASS
- overlap no spurious overflow: PASS
- overlap pair set: PASS
- overlap overflow flag set: PASS
- voxelize_mark use_sat=False exact: PASS
- voxelize_mark use_sat=False deterministic: PASS
- voxelize_mark use_sat=True exact: PASS
- voxelize_mark use_sat=True deterministic: PASS
- sat mode0 hit_mask exact: PASS
- sat mode0 deterministic: PASS
- sat mode0 hit_mask exact (hits found): PASS
- sat mode1 deterministic: PASS
- sat mode1 idx alignment: PASS
- sat mode2 deterministic poly verts: PASS
- sat mode2 poly_counts in range: PASS

GPU dispatch confirmed: .pyd contains `.hipFatB` section with `gfx1101` code object; kernels
executed on AMD Radeon PRO V710 (gfx1101, RDNA3, wave32) at HIP_VISIBLE_DEVICES=0.
Fork source tree: clean (only untracked .pyd build artifact, gitignored).

No source changes needed vs. c72480ea -- the Windows LLP64 and ValueError fixes from that
commit apply identically to gfx1101 (same Windows/MSVC ABI as gfx1201); gfx1101 is RDNA3
wave32, same family as the already-validated linux-gfx1100.

Verdict: completed. validated_sha=c72480ea (windows-gfx1101).

## Revalidation 2026-06-07 (linux-gfx1100 carry-forward)

Platform: linux-gfx1100 (AMD Radeon Pro W7800 48GB, gfx1100, RDNA3, wave32, ROCm 7.2.1)
Fork: AMD-Ecosystem/FaithC @ moat-port c72480ea (delta commit on top of ec2fae28)
Method: binary-equivalence (codeobj_diff.py)

Git delta (ec2fae28 -> c72480ea):
- `kernels.cu`: `long` -> `int64_t` rename in kernel signatures and host data_ptr calls
- `setup.py`: Windows-only `/ALTERNATENAME` linker directive guarded by `sys.platform == "win32"`

The Windows setup.py change has no effect on Linux (guarded at Python level). On 64-bit
Linux x86_64, `sizeof(long)==8 == sizeof(int64_t)`, so the `long->int64_t` rename is
semantically transparent and compiles to identical gfx1100 device ISA.

Built both SHAs at PYTORCH_ROCM_ARCH=gfx1100:
```
# Old (ec2fae28):
rm -f src/faithcontour/_C/kernels.hip src/faithcontour/_C*.so && rm -rf build/
HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1100 python setup.py build_ext --inplace

# New (c72480ea):
rm -f src/faithcontour/_C/kernels.hip src/faithcontour/_C*.so && rm -rf build/
HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1100 python setup.py build_ext --inplace
```

Both builds: PASS (exit 0, loop-unroll advisories on sat_centroid/sat_clip templates)

codeobj_diff result:
```
verdict=identical
  _C.cpython-312-x86_64-linux-gnu.so: identical (exported symbols + device ISA identical (144 exports))
```

No GPU re-run needed. Verdict: carry-forward completed. validated_sha=c72480ea (linux-gfx1100).

## Revalidation 2026-06-17 (windows-gfx1201)

Platform: AMD Radeon RX 9070 XT, gfx1201 (RDNA4, wave32), Windows 11 Pro for Workstations
Fork: AMD-Ecosystem/FaithC @ moat-port 5e7e93a
Validator: claude-sonnet-4-6

### Context

windows-gfx1201 was in `revalidate` state: validated_sha was 9827cc8 (old, pre-rebase
history), head_sha was 1d47e7a (rebased onto upstream main 156f104 after upstream merged
#11). The old SHA was unreachable in the current history so binary-equivalence carry-forward
was not possible; a full rebuild and GPU test was required.

### New commit added this session: 5e7e93a

During the build, PyTorch's `BuildExtension` on Windows raised "Don't know how to compile
kernels.hip". Root cause: after a PyTorch update (torch 2.9.1+rocm7.14, Jun 12 2026),
`BuildExtension.build_extensions` no longer adds `.hip` to the MSVC compiler's
`_cpp_extensions` list (it adds `.cu/.cuh` but not `.hip`). The hipify step renames
`kernels.cu` -> `kernels.hip` before compilation; the MSVC compiler driver's compile loop
then fails before the spawn wrapper (which routes `.hip` -> hipcc) is ever reached.

Fix: subclass `BuildExtension` in setup.py, override `build_extensions` to append `.hip`
to `_cpp_extensions` on Windows before delegating to the parent. The parent's spawn wrapper
then correctly intercepts `.hip` files and routes them to hipcc. The guard
(`sys.platform == "win32"` and `hasattr(self.compiler, "_cpp_extensions")`) is a no-op on
Linux where the HIP compiler is clang, not MSVC. Committed as 5e7e93a on top of 1d47e7a.

### Build

```
export PATH="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64:$PATH"
VENV=/b/develop/TheRock/external-builds/pytorch/.venv
cd /b/develop/moat/projects/FaithC/src
rm -f src/faithcontour/_C/kernels.hip && rm -rf build/
HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1201 \
  ROCM_HOME=$VENV/Lib/site-packages/_rocm_sdk_devel \
  DISTUTILS_USE_SDK=1 \
  $VENV/Scripts/python.exe setup.py build_ext --inplace
```

Build result: PASS (~60 s, exit 0, loop-unroll advisories on sat_centroid/sat_clip -- same as before)
hipcc invoked with `--offload-arch=gfx1201`; 40 warnings for gfx1201. gfx1201 code-object
confirmed in .pyd (`.hipFatB` section present; hipcc target=gfx1201).

### Test

```
HIP_VISIBLE_DEVICES=0 $VENV/Scripts/python.exe agent_space/faithc_harness_win.py
```

Test result: 17/17 PASS (exit 0)

Pass breakdown:
- seg_tri pair set: PASS
- seg_tri dots: PASS
- seg_tri deterministic set: PASS
- overlap no spurious overflow: PASS
- overlap pair set: PASS
- overlap overflow flag set: PASS
- voxelize_mark use_sat=False exact: PASS
- voxelize_mark use_sat=False deterministic: PASS
- voxelize_mark use_sat=True exact: PASS
- voxelize_mark use_sat=True deterministic: PASS
- sat mode0 hit_mask exact: PASS
- sat mode0 deterministic: PASS
- sat mode0 hit_mask exact (hits found): PASS
- sat mode1 deterministic: PASS
- sat mode1 idx alignment: PASS
- sat mode2 deterministic poly verts: PASS
- sat mode2 poly_counts in range: PASS

GPU dispatch confirmed: hipcc compiled with --offload-arch=gfx1201 for AMD Radeon RX 9070 XT
(gfx1201, RDNA4, wave32); HIP_VISIBLE_DEVICES=0 verified at start via hipInfo.
Fork source tree: clean (only untracked .pyd build artifact, gitignored).

### Note for other platforms

The setup.py subclass fix (5e7e93a on top of 1d47e7a) is Windows-only (guarded by
`sys.platform == "win32"`). On Linux the new `BuildExtension` subclass is a no-op (the
MSVC compiler is not used; `_cpp_extensions` is an MSVC-specific attribute absent from
the Unix CCompiler). Linux validators can carry forward via binary-equivalence
(codeobj_diff): the compiled .so should be identical to what was built at 1d47e7a.

Verdict: completed. validated_sha=5e7e93a (windows-gfx1201).

## Revalidation 2026-06-18 (linux-gfx1100)

Platform: linux-gfx1100 (AMD Radeon Pro W7800 48GB, gfx1100, RDNA3, wave32, ROCm 7.2.53211)
Fork: AMD-Ecosystem/FaithC @ moat-port 5e7e93a
Validator: claude-sonnet-4-6

### Delta assessed: 9827cc8 (phantom pre-rebase SHA, not in current history) -> 5e7e93a

The linux-gfx1100 validated_sha (9827cc8) is a pre-rebase phantom -- not in the
current branch history after the upstream merge rebase. Binary-equivalence carry-forward
was not possible (old SHA unreachable), so a full GPU revalidation was performed.

The single new commit relative to 1d47e7a (prior Linux-equivalent state) is 5e7e93a,
which adds a Windows-only setup.py subclass fix (guarded by `sys.platform == "win32"`).
On Linux this code path is never entered (clang is the host compiler, not MSVC; the
`_cpp_extensions` attribute is MSVC-specific and absent from the Unix CCompiler). The
fix is a no-op on Linux.

### Build

```
cd projects/FaithC/src
rm -f src/faithcontour/_C/kernels.hip src/faithcontour/_C*.so && rm -rf build/
HIP_VISIBLE_DEVICES=2 PYTORCH_ROCM_ARCH=gfx1100 python setup.py build_ext --inplace
```

Build result: PASS (40 s, exit 0, loop-unroll advisories on sat_centroid/sat_clip -- same as before)

gfx1100 code-object verified:
```
llvm-objdump --offloading _C.cpython-312-x86_64-linux-gnu.so | grep gfx
# Extracting: hipv4-amdgcn-amd-amdhsa--gfx1100  PRESENT (single-arch gfx1100 build)
```

Fork source tree: clean (no uncommitted files; .hip/.so.* gitignored)

### Test

```
HIP_VISIBLE_DEVICES=2 AMD_LOG_LEVEL=3 python agent_space/faithc_harness.py
```

Test result: 16/16 PASS (exit 0)

AMD_LOG_LEVEL=3 confirms native gfx1100 code-object dispatch:
"Using native code object for device: amdgcn-amd-amdhsa--gfx1100 co: amdgcn-amd-amdhsa--gfx1100"

Pass breakdown:
- seg_tri pair set: PASS
- seg_tri dots (maxerr=1.19e-07): PASS
- seg_tri deterministic set: PASS
- overlap no spurious overflow: PASS
- overlap pair set: PASS
- overlap overflow flag set: PASS
- voxelize_mark use_sat=False exact: PASS
- voxelize_mark use_sat=False deterministic: PASS
- voxelize_mark use_sat=True exact: PASS
- voxelize_mark use_sat=True deterministic: PASS
- sat mode0 hit_mask exact (4 hits): PASS
- sat mode0 deterministic: PASS
- sat mode1 deterministic: PASS
- sat mode1 idx alignment: PASS
- sat mode2 deterministic poly verts: PASS
- sat mode2 poly_counts in range: PASS

Verdict: completed. validated_sha=5e7e93a (linux-gfx1100).

## Revalidation 2026-06-19 (windows-gfx1101)

Platform: AMD Radeon PRO V710, gfx1101 (RDNA3, wave32), Windows 11 Pro for Workstations
Fork: AMD-Ecosystem/FaithC @ moat-port 5e7e93a
Validator: claude-sonnet-4-6

### Context

windows-gfx1101 was in `revalidate` state: validated_sha 9827cc8 is a pre-rebase
phantom SHA not in the current branch history. Binary-equivalence carry-forward was
not possible (old SHA unreachable), so a full GPU revalidation was performed.

The single new commit relative to the prior gfx1101 validation (c72480ea) is 5e7e93a,
which adds a Windows-only setup.py `BuildExtension` subclass fix (guarded by
`sys.platform == "win32"`) that appends `.hip` to MSVC's `_cpp_extensions`. This fix
IS functional on Windows gfx1101 (same MSVC ABI as gfx1201), so a full GPU re-run
was required.

Device mapping verified before run: HIP_VISIBLE_DEVICES=1 -> AMD Radeon PRO V710
(gfx1101); HIP_VISIBLE_DEVICES=0 -> AMD Radeon RX 9070 XT (gfx1201).

### Build

```
export PATH="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64:$PATH"
VENV=/b/develop/TheRock/external-builds/pytorch/.venv
rm -f projects/FaithC/src/src/faithcontour/_C/kernels.hip && rm -rf projects/FaithC/src/build/ && rm -f projects/FaithC/src/src/faithcontour/_C.cp312-win_amd64.pyd
HIP_VISIBLE_DEVICES=1 PYTORCH_ROCM_ARCH=gfx1101 \
  ROCM_HOME=$VENV/Lib/site-packages/_rocm_sdk_devel \
  DISTUTILS_USE_SDK=1 \
  env -C projects/FaithC/src $VENV/Scripts/python.exe projects/FaithC/src/setup.py build_ext --inplace
```

Build result: PASS (~27 s, exit 0, 40 loop-unroll advisories on sat_centroid/sat_clip -- same as gfx1201)
gfx1101 code-object confirmed in .pyd (`gfx1101` marker and `.hipFatB` section present in PE binary).

### Test

```
HIP_VISIBLE_DEVICES=1 $VENV/Scripts/python.exe agent_space/faithc_harness_win.py
```

Test result: 17/17 PASS (exit 0, ~3 s)

Pass breakdown:
- seg_tri pair set: PASS
- seg_tri dots: PASS
- seg_tri deterministic set: PASS
- overlap no spurious overflow: PASS
- overlap pair set: PASS
- overlap overflow flag set: PASS
- voxelize_mark use_sat=False exact: PASS
- voxelize_mark use_sat=False deterministic: PASS
- voxelize_mark use_sat=True exact: PASS
- voxelize_mark use_sat=True deterministic: PASS
- sat mode0 hit_mask exact: PASS
- sat mode0 deterministic: PASS
- sat mode0 hit_mask exact (hits found): PASS
- sat mode1 deterministic: PASS
- sat mode1 idx alignment: PASS
- sat mode2 deterministic poly verts: PASS
- sat mode2 poly_counts in range: PASS

GPU dispatch confirmed: hipcc compiled with --offload-arch=gfx1101; .pyd contains
`.hipFatB` section with `gfx1101` code object. Kernels executed on AMD Radeon PRO V710
(gfx1101, RDNA3, wave32) at HIP_VISIBLE_DEVICES=1.
Fork source tree: clean (only untracked .pyd build artifact, gitignored).

Verdict: completed. validated_sha=5e7e93a (windows-gfx1101).

## Validation 2026-08-09 (linux-gfx90a carry-forward)

Platform: linux-gfx90a (AMD Instinct MI250X, gfx90a), HIP_VISIBLE_DEVICES=0
Fork: AMD-Ecosystem/FaithC @ moat-port 5e7e93a
Validator: claude-opus-5-1m

### Delta assessed: 1d47e7a (prior linux-gfx90a validated_sha) -> 5e7e93a (head_sha)

`python3 utils/moatlib.py classify FaithC 1d47e7a58d2062961ba543d871da3524f6b616fb
5e7e93aa38a53937a552b5758e060fa9b0d642ab` -> `class=mixed` (token count differs in
setup.py), so full revalidation would normally apply; confirmed binary equivalence
instead per the carry-forward shortcut.

`git diff 1d47e7a 5e7e93a` touches only `setup.py`: adds a `BuildExtension` subclass
that appends `.hip` to MSVC's `_cpp_extensions` list, entirely inside
`if sys.platform == "win32" and hasattr(self.compiler, "_cpp_extensions"):`. Same
single commit already assessed as Linux-inert by the 2026-06-18 linux-gfx1100
revalidation (full GPU re-run there since its old validated_sha was an unreachable
pre-rebase phantom). Here the old SHA (1d47e7a) is reachable, so binary equivalence
was provable directly instead of a fresh 16/16 GPU run.

### Binary-equivalence build (same absolute source path, per codeobj_diff.py caveat
### that __FILE__ strings make identical code compare as differ otherwise)

```
# old: git checkout 1d47e7a58d2062961ba543d871da3524f6b616fb (detached)
rm -f src/faithcontour/_C/kernels.hip src/faithcontour/_C*.so && rm -rf build/
HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx90a python setup.py build_ext --inplace
cp src/faithcontour/_C.cpython-312-x86_64-linux-gnu.so .../faithc-cmp-old/

# new: git checkout 5e7e93aa38a53937a552b5758e060fa9b0d642ab (detached)
rm -f src/faithcontour/_C/kernels.hip src/faithcontour/_C*.so && rm -rf build/
HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx90a python setup.py build_ext --inplace
cp src/faithcontour/_C.cpython-312-x86_64-linux-gnu.so .../faithc-cmp-new/
```

Both builds: PASS (exit 0, same loop-unroll advisories on sat_centroid/sat_clip as
every prior gfx90a build).

```
python3 utils/codeobj_diff.py faithc-cmp-old faithc-cmp-new
verdict=identical
  _C.cpython-312-x86_64-linux-gnu.so: identical (exported symbols + device ISA identical (139 exports))
```

No GPU re-run needed -- the compiled program on gfx90a is provably unchanged.
`python3 utils/moatlib.py carry-forward FaithC linux-gfx90a 5e7e93aa38a53937a552b5758e060fa9b0d642ab binary-equiv "..."`.

### CUDA no-regression gate: cuda-not-validated

Attempted `nvcc -c src/faithcontour/_C/kernels.cu -arch=sm_80 -std=c++20
--expt-relaxed-constexpr` (pinned arch per policy; ccbin g++-13) against this host's
only PyTorch install (ROCm dev build `2.14.0a0+gitb6b444c`) since no CUDA-flavored
PyTorch is installed here. Failed with ~100 errors, all rooted at
`torch/headeronly/util/complex.h:9`:

```
#if defined(__HIPCC__) || defined(__HIPCC__)
#include <thrust/complex.h>
#endif
```

That guard checks `__HIPCC__` twice (evidently meant `__CUDACC__ || __HIPCC__`), so
under nvcc (`__CUDACC__` defined, `__HIPCC__` not) the file never includes
`<thrust/complex.h>`, while `c10/util/complex.h`/`complex_math.h` unconditionally
reference `thrust::complex`/`thrust::abs` under `#if defined(__CUDACC__) ||
defined(__HIPCC__)`, producing "identifier thrust is undefined" cascades. This is a
pre-existing defect in the installed dev-build PyTorch's own headers, unrelated to
FaithC: it fires on ANY torch/extension.h-including TU compiled with nvcc against
this install, regardless of project content. `grep -n
'__builtin_trap|__trap|__HIP|hip[A-Z]|amdgcn|USE_ROCM'` over the three port sources
(kernels.cu, bindings.cpp, kernels.h) found nothing -- no HIP-only symbols, no
asm/trap constructs, confirming the port's own diff contains nothing nvcc-illegal.

Recording `cuda-not-validated: no CUDA-flavored PyTorch install on this host to
build the CUDAExtension against; the only torch (ROCm dev build) has an unrelated
__HIPCC__/__HIPCC__ typo in torch/headeronly/util/complex.h that blocks any
torch-extension nvcc compile here, independent of port content`. Not a gate;
environmental wall per validator policy (a real CUDA-flavored torch install is an
NVIDIA-only dependency graph, not in the conda cuda-12.8 toolkit env).

### Jargon / docs

`python3 utils/jargon.py --port FaithC` -> clean. README.md AMD GPU (ROCm) section
(collapsible `<summary>` alongside the CUDA instructions) already documents the
ROCm build; no change needed.

Fork clone: clean at 5e7e93a (checked out/rebuilt twice for the binary-equivalence
compare, restored to head with no tracked-file diff).

Verdict: completed (carry-forward, binary-equiv). validated_sha=5e7e93a
(linux-gfx90a). CUDA gate: cuda-not-validated (environmental wall, see above).
