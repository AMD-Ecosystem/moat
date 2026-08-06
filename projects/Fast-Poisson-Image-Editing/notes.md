# Fast-Poisson-Image-Editing notes

## Summary

fpie (Fast Poisson Image Editing): parallel Poisson image-editing / seamless-cloning solver
with many interchangeable backends behind one Python API. The CUDA backend
(`fpie/core/cuda/`, pybind11 module `core_cuda`) is a Jacobi stencil/reduction solver. Ported
to ROCm/HIP for gfx90a via Strategy A (compat header + `.cu` marked `LANGUAGE HIP`, CUDA path
untouched, all HIP behind `USE_HIP`). No prior AMD support existed (genuine fresh port).

## Build classification: pure CMake (Strategy A)

NOT a pytorch extension. setup.py (CMakeBuild) drives a plain `cmake && make`; the CUDA backend
is `pybind11_add_module(core_cuda solver.cc equ.cu grid.cu utils.cu)` linked to `cudart`
(fpie/core/cuda/CMakeLists.txt). pybind11 is git-cloned by the top-level CMakeLists at configure
time (gitignored).

## Files changed (port)

- `fpie/core/cuda/cuda_to_hip.h` (NEW): the only HIP-aware file. On USE_HIP includes
  <hip/hip_runtime.h> and `#define`s the 11 cudaXxx runtime symbols to hip; else includes the
  CUDA headers (<cuda.h>, <cuda_runtime.h>, <driver_functions.h>). Vector types need no alias.
- `fpie/core/cuda/utils.h`: include the compat header instead of the raw CUDA headers; guard the
  custom `operator+=(float3&,float3)` with `#if !defined(USE_HIP)` (fault class below).
- `fpie/core/cuda/CMakeLists.txt`: `option(USE_HIP OFF)`; HIP branch does `enable_language(HIP)`,
  defaults CMAKE_HIP_ARCHITECTURES to gfx90a only when unset, marks the .cu `LANGUAGE HIP`,
  `pybind11_add_module(... NO_EXTRAS ...)`, sets HIP_ARCHITECTURES, defines USE_HIP; else branch
  is the original CUDA path verbatim.
- `CMakeLists.txt` (top): gate the cuda subdir on USE_HIP (find_package(CUDA) fails on a ROCm
  host), else keep the original find_package(CUDA) guard.

USE_HIP defaults OFF everywhere, so a stock NVIDIA/CUDA build is unchanged.

## CUDA surface (small, fully enumerated)

Runtime: cudaMalloc, cudaFree, cudaMemcpy, cudaMemset, cudaMemcpy{Host->Device,Device->Host},
cudaDeviceSynchronize, cudaGetDeviceCount, cudaGetDeviceProperties, cudaDeviceProp, cudaError_t.
Headers: <cuda.h>, <cuda_runtime.h>, <driver_functions.h>. Vector types int4/float3 +
make_int4/make_float3 (native in HIP). Kernel model only. NO warp primitives, NO textures/
surfaces, NO cuBLAS/cuFFT/cuRAND/Thrust/CUB, NO atomics.

## Fault classes hit / not hit

- HIP vector-type operator collision (HIT, the only source fix): HIP's `float3`
  (`HIP_vector_type<float,3>`) ships a componentwise member `operator+=`, so the project's free
  `operator+=(float3&,float3)` (utils.h:11) made EVERY `+=` ambiguous (14 errors in equ.cu, 2 in
  grid.cu). CUDA's float3 is a plain struct with no operators, so it still needs the free one.
  Fix: `#if !defined(USE_HIP)` around just that operator; HIP's built-in has identical semantics.
  The project's other free operators (`+`, `-`, scalar `*`, scalar `/`, `int4*int`) do NOT
  collide (HIP provides only `*`/`/` as vector-vector friends, not the float3*scalar overloads),
  so they were left untouched.
- pybind11 LTO + HIP link (HIT, build-only): pybind11_add_module injects `-flto=auto
  -fno-fat-lto-objects` into compile+link flags; the HIP link step does not finalize LTO, so the
  module shipped with no exported `PyInit_core_cuda` -> ImportError ("dynamic module does not
  define module export function"). `INTERPROCEDURAL_OPTIMIZATION OFF` on the target is NOT
  enough (pybind11 sets the flags directly, not via the CMake IPO property). Fix: pass
  `NO_EXTRAS` to pybind11_add_module on the HIP target (drops the LTO flags). After: `PyInit_*`
  exported, import works.
- driver_functions.h (HANDLED): not present in ROCm; included only on the CUDA side of the
  compat header (it provided pitched-ptr helpers this code never uses).
- Out-of-bounds stencil reads (NOT triggered): grid.cu reads `X-m3, X-3, X+3, X+m3` neighbors
  without clamping, but the Python layer guarantees a >=1px inactive (mask==0) border on every
  side (process.py: zero-out edge then crop x.min()-1 .. x.max()+2), and kernels only write where
  mask is set, so all neighbors are in-bounds. equ.cu gathers neighbors via index array `A` and
  guards id==0; slot 0 is allocated. Confirmed by reading the host code AND by the grid solver
  running bit-exact on real image data on gfx90a. No clamp needed.
- Warp size / wave64 (NOT triggered): the error reduction uses NO warp primitive and assumes NO
  lane count. error_sum_equ_kernel (equ.cu:162) reduces via __shared__ + a single-thread serial
  loop; error_grid_kernel (grid.cu:73) reduces per-block by thread (0,0) and globally by block
  (0,0), both serial. Hence run-to-run deterministic on wave64 and no kWarpSize constant needed.

## Build (gfx90a, GPU 1)

    cmake <src> -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=$(which python3)
    make -j core_cuda

`ldd core_cuda*.so` -> libamdhip64.so.7 (HIP runtime, not cudart). For a follower arch, only
`-DCMAKE_HIP_ARCHITECTURES=<arch>` changes; no source edit.

## Validation (REAL GPU, HIP_VISIBLE_DEVICES=1, gfx90a / MI250X)

Sample blend = the project's canonical test1 image set (tests/data.txt: DIP2018 test1
src/mask/tgt), placement `-h1 -150 -w1 -50`, 5000 Jacobi iterations, gradient=max.

- equ solver:  max|cuda - numpy| = 0.0 (bit-identical output image), deterministic (run1==run2),
  abs residual [0,0,0].
- grid solver: max|cuda - numpy| = 0.0 (bit-identical), deterministic, abs residual [0,0,0]
  (numpy reference residual ~0.06; cuda converged at least as well).
- End-to-end CLI (fpie.cli.main, the `fpie` console-script path) with `-b cuda` for both
  `--method equ` and `--method grid` writes valid 427x770 blended images (~17-20 ms/run).
- Upstream smoke suite `pytest tests/test_smoke.py`: 7 passed, 1 skipped (the skipped one is the
  OpenMP-vs-numpy CPU comparison; openmp not installed into the validation venv). No regression.

Harness: agent_space/validate_fpie.py (compares cuda vs numpy + determinism).

Gotcha: `python3 -m fpie.cli` does NOTHING (exits 0, no output) because cli.py has no
`if __name__=="__main__"` block -- it only defines `main()`. Use the `fpie` console script or
call `fpie.cli.main()` (as the validation harness does). Not a port issue.

## Env notes

- cv2 (opencv-python-headless) pulls numpy 2.x which breaks numba 0.64 (needs numpy<2.3); pin
  `numpy==1.26.2` so numba + cv2 + the core path all import.
- taichi not installed (optional backend); not needed for the cuda port validation.

## Validation 2026-05-30 (gfx1100, ROCm 7.2.1)

Platform: 2x AMD Radeon Pro W7800 48GB, gfx1100 (RDNA3, wave32), ROCm 7.2.1.
Fork branch: `moat-port` @ `6a41e0b33c9bd6a54c1d69f0cda3354318cd2cff` (no source change from gfx90a).

**Build commands (wrapped with timeit.sh):**

```
cmake /var/lib/jenkins/moat/projects/Fast-Poisson-Image-Editing/src \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -B /var/lib/jenkins/moat/agent_space/fpie_build_gfx1100

utils/timeit.sh Fast-Poisson-Image-Editing compile -- \
  cmake --build /var/lib/jenkins/moat/agent_space/fpie_build_gfx1100 --target core_cuda -j$(nproc)
```

Build: PASS. Only warnings are `nodiscard` on `hipGetDeviceProperties` (pre-existing CUDA style, harmless).

**gfx1100 code-object evidence:**

```
roc-obj-ls core_cuda.cpython-312-x86_64-linux-gnu.so
# -> hipv4-amdgcn-amd-amdhsa--gfx1100 (two code objects, one per kernel file)
# No gfx90a code objects present.

nm -D core_cuda.cpython-312-x86_64-linux-gnu.so | grep PyInit
# -> 0000000000022940 T PyInit_core_cuda  (NO_EXTRAS fix confirmed working)
```

**.so size:** 434 KB (not tiny; LTO pitfall avoided via NO_EXTRAS).

**Import check:**

```python
from fpie.process import ALL_BACKEND
# -> ['numpy', 'numba', 'taichi-cpu', 'taichi-gpu', 'gcc', 'openmp', 'cuda']
```

**GPU validation (HIP_VISIBLE_DEVICES=0, test1 dataset, 5000 Jacobi iters, gradient=max, mask_on_tgt=(-150,-50)):**

```
utils/timeit.sh Fast-Poisson-Image-Editing test -- \
  python3 /var/lib/jenkins/moat/agent_space/validate_fpie_gfx1100.py
```

Results:
- equ solver: max|cuda - numpy| = 0.0 (bit-exact vs numpy reference), residual=[0,0,0], output shape 427x770.
- grid solver: max|cuda - numpy| = 0.0 (bit-exact vs numpy reference), residual=[0,0,0] (numpy residual ~0.06; HIP converged better).
- Determinism (run1 vs run2): max|run1-run2| = 0 for both equ and grid (bit-exact, wave32 serial reductions as expected).
- Output matches the gfx90a reference: bit-exact in all channels (both solvers).

**Non-GPU smoke suite (pytest tests/test_smoke.py):** 8 passed, 0 skipped, 0 failed. (gfx90a had 7 passed + 1 skipped for OpenMP; here OpenMP is installed so all 8 pass -- no regression.)

**Fork:** NOT touched. No source change needed, no fork push, head_sha unchanged at `6a41e0b`.

**Result: PASS.**

## Validation 2026-05-31 (windows-gfx1151, ROCm 7.14.0a TheRock)

Platform: AMD Radeon 8060S Graphics (gfx1151, RDNA3.5 APU), Windows 11. AMD clang
23.0.0git (rocm-sdk 7.14.0a20260531). Validate-first follower: NO source change
(fork HEAD 6a41e0b, same sha as gfx90a + gfx1100). Build script: agent_space/fpie_build.sh.

### Build (Windows all-clang)
enable_language(HIP) forces the all-clang toolchain on Windows. Built only the
core_cuda target against venv-gsplat Python 3.13:
```
cmake -S <src> -B build-win-gfx1151 -G Ninja \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_HIP_COMPILER=clang++ \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DCMAKE_PREFIX_PATH=<rocm-root> \
  -DPYTHON_EXECUTABLE=<py> -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_HIP_STANDARD=17 -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_FLAGS="-DNOMINMAX -DWIN32_LEAN_AND_MEAN" -DCMAKE_BUILD_TYPE=Release
cmake --build build-win-gfx1151 --target core_cuda -j16   # [5/5] core_cuda.cp313-win_amd64.pyd
```
Windows-only build-invocation deltas (NOT source changes): all-clang trio;
POLICY_VERSION_MINIMUM=3.5 (min 3.5 vs cmake 4.3); HIP_STANDARD=17; NOMINMAX. The
gfx1151 code object is embedded (strings -> gfx1151 only); .pyd 374 KB (NO_EXTRAS
LTO pitfall avoided).

### Runtime / harness
process.py does `from fpie import core_cuda`, so the .pyd must sit at fpie/core_cuda.*.pyd
(directly under fpie/, NOT fpie/core/cuda/). TheRock runtime deployed beside it
(amdhip64_7.dll + amd_comgr + rocm_kpack.dll). Deps: numpy + opencv-python-headless
(no numba/taichi/openmp needed for the cuda+numpy path).

### Results (HIP_VISIBLE_DEVICES=0)
- core_cuda init: "Found 1 CUDA devices, Device 0: AMD Radeon(TM) 8060S Graphics, 20 SMs".
- cuda vs numpy (agent_space/validate_fpie_win.py, 64x64 random src/tgt, random mask,
  500 Jacobi iters):
    Equ  : max|cuda-numpy| = 0 (bit-identical), deterministic (run1==run2) -> PASS
    Grid : max|cuda-numpy| = 0 (bit-identical), deterministic               -> PASS
- Upstream smoke suite `pytest tests/test_smoke.py`: 7 passed, 1 skipped (the skip is
  the OpenMP-vs-numpy CPU test; openmp not built). Identical to gfx90a + gfx1100.

RESULT: PASS. Matches gfx90a + gfx1100 exactly. windows-gfx1151 -> completed.

## Validation 2026-06-03 (windows-gfx1151) -- binary-equiv carry-forward to 4c6411b

Revalidate triggered by the fork advancing 6a41e0b -> 4c6411b (a rebase onto newer
upstream + resha; the recorded validated_sha 7c8dbe8 was an orphaned one-commit-era
amend). gfx1151 had GPU-validated the core_cuda backend at 6a41e0b (the 2026-05-31
record above: cuda-vs-numpy bit-identical, 7/1 smoke). Confirmed binary-equivalence
on this host rather than re-running the GPU:

Delta 6a41e0b..4c6411b (5 files): README.md + docs/{backend,get_start}.rst (docs);
fpie/core/cuda/CMakeLists.txt (default-arch refactor: `if(NOT CMAKE_HIP_ARCHITECTURES)`
+ REMOVE_DUPLICATES, and dropping MOAT jargon from a comment); setup.py (+_hip_cmake_args
for `pip install` HIP autodetect). The .cu kernels (equ/grid/utils) are UNTOUCHED. The
CMakeLists default-arch branch is dead code for the validated build (which passes
`-DCMAKE_HIP_ARCHITECTURES=gfx1151` explicitly; REMOVE_DUPLICATES is a no-op on one
arch), and setup.py is off the direct `cmake --build --target core_cuda` validated path.

Proof (rebuilt at 4c6411b via agent_space/fpie_build.sh, all-clang gfx1151):
- Extracted the embedded device code object from both .pyd: llvm-objcopy --dump-section
  .hip_fat, clang-offload-bundler --unbundle hipv4-amdgcn-amd-amdhsa--gfx1151.
- llvm-objdump -d of the two code objects: ISA BYTE-IDENTICAL (844 instruction lines
  match exactly; the only textual diff is the objdump "file format" header echoing the
  temp filename).
- llvm-nm: identical kernel symbol table (iter_equ_kernel, error_equ_kernel,
  copy_X_equ_kernel, error_sum_equ_kernel, iter_shared_equ_kernel).
- .hipFatB fatbin wrapper section md5-identical; .hip_fat raw bytes differed only in
  non-semantic code-object metadata (ELF note), not instructions.

method=binary-equiv. validated_sha -> 4c6411b. No GPU re-run. All three platforms
now completed @ 4c6411b.

## Validation 2026-06-03 (gfx1100) -- binary-equiv carry-forward to 4c6411b (own-arch confirm)

Closing a documentation gap: gfx1100 had been set completed @ 4c6411b folded into the
gfx1151 entry above ("all three platforms"), but the per-arch binary-equivalence proof
recorded there extracted only the gfx1151 code object. This entry records the first-hand
gfx1100 confirmation, run on the gfx1100 Jenkins host (2x W7800, ROCm 7.2.1).

gfx1100 GPU-validated core_cuda at 6a41e0b (the 2026-05-30 record above: cuda-vs-numpy
bit-exact, determinism bit-exact, 8/0 smoke). Delta 6a41e0b..4c6411b is docs + a
CMakeLists default-arch refactor + a setup.py pip-HIP-autodetect addition; `git diff
6a41e0b 4c6411b -- '*.cu' '*.cuh' '*.cpp' '*.h'` is EMPTY (zero device source change).

Proof (built both shas via git worktrees, identical cmake invocation, gfx1100):
- Same cmake line as the GPU validation: `-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release`,
  `--target core_cuda`. Both built clean.
- utils/codeobj_diff.py OLD.so NEW.so -> verdict=identical (exported symbols + device
  ISA identical, 92 exports).
- Corroboration: 2 gfx1100 code objects per .so (matches original); device ISA disasm
  sha256 identical (268a634d...f4df06c on both); exported-symbol tables differ ONLY in
  three auto-generated __hip_cuid_* BSS GUIDs (per-TU compilation markers, non-semantic,
  unreferenced by name -- same non-semantic-metadata class as the gfx1151 ELF-note diff).

method=binary-equiv. gfx1100 completed @ 4c6411b is now backed by an own-arch artifact.
No GPU re-run. PR #25 stands as-is. (Builds were throwaway worktrees under agent_space.)

## Validation 2026-06-04 (windows-gfx1101 + windows-gfx1201, one FAT .pyd) -- follower, NO source change

validated_sha: 4c6411b (zero-churn followers). Host = dual-GPU Windows workstation
(memory windows-gfx1101-gfx1201-host). ROCm 7.14 / TheRock pip SDK; venv Python 3.12.

### Multi-arch fat build (one .pyd, both GPUs)
CMake reads CMAKE_HIP_ARCHITECTURES; one configure with a LIST emits both archs into the
core_cuda pybind module. Script: agent_space/fpie-win/build.sh.
```
ROCM=.../_rocm_sdk_devel ; PY=.../venv/Scripts/python.exe
cmake -S . -B build-win -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx1101;gfx1201" \
  -DCMAKE_C/CXX/HIP_COMPILER=$ROCM/lib/llvm/bin/clang(++).exe -DCMAKE_PREFIX_PATH=$ROCM \
  -DPYTHON_EXECUTABLE=$PY -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_HIP_STANDARD=17 -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_FLAGS="-DNOMINMAX -DWIN32_LEAN_AND_MEAN" -DCMAKE_BUILD_TYPE=Release
cmake --build build-win --target core_cuda -j64   # -> core_cuda.cp312-win_amd64.pyd
```
Deps: `pip install opencv-python-headless pytest` into the venv (numpy already present).

### Runtime
The .pyd must sit at fpie/core_cuda.*.pyd (process.py does `from fpie import core_cuda`).
Copy TheRock amdhip64_7.dll/amd_comgr/rocm_kpack/hiprtc into fpie/ (beside the .pyd; beats
System32 amdhip64 -- the dietgpu lesson), and `os.add_dll_directory()` for fpie/ + the three
_rocm_sdk_{core,devel,libraries}/bin. Run the harness with `python -u` (the C++ core prints a
device banner via printf; without -u, Python's buffered stdout is interleaved/lost).

### Validation (real GPU; numpy backend is the per-pixel oracle)
agent_space/fpie-win/validate_fpie_win.py: 64x64 random src/tgt, random mask, 500 iters,
Equ + Grid solvers, cuda vs numpy, plus cuda determinism (run1 vs run2):
| | gfx1101 (dev0) | gfx1201 (dev1) |
|--|----------------|----------------|
| Equ  max\|cuda-numpy\| | 0 (bit-identical), det 0 | 0 (bit-identical), det 0 |
| Grid max\|cuda-numpy\| | 0 (bit-identical), det 0 | 0 (bit-identical), det 0 |
| pytest tests/test_smoke.py | 7 passed, 1 skipped | (CPU suite, device-independent) |

ALL_BACKEND = ['numpy','cuda'] (core_cuda imported). Bit-identical to numpy on both RDNA3 and
RDNA4; the openmp skip is expected (openmp backend not built). Matches gfx90a/gfx1100/gfx1151.
State: windows-gfx1101 + windows-gfx1201 port-ready -> completed (validated_sha 4c6411b,
fork unchanged). All five platforms terminal. Upstream PR #25 already merged.
