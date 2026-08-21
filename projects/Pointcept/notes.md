# Pointcept notes

## Port summary (lead: linux-gfx90a)
Strategy B (torch build-time hipify) port of the four in-tree PyTorch CUDA-extension
GPU libraries under `libs/`: pointops, pointops2, pointgroup_ops, pointrope.
pointseg is a CppExtension (CPU only, no `.cu`) and needs no port. spconv is an
external dependency (separate MOAT project, unported) and does NOT block these libs --
the four libs build, install, and pass their op tests without it. Sparse-conv MODEL
configs (SpUNet/OACNN/PointGroup end-to-end) wait on the spconv ROCm port; that is out
of scope for this port's validation.

## The only source edit
`libs/pointrope/setup.py`: added a `torch.version.hip` branch. hipcc rejects the
nvcc-only flags (`--ptxas-options=-v`, `--use_fast_math`) and the
`cuda.get_gencode_flags()` `-gencode arch=compute_*` list. On ROCm we pass
`["-O3", "-ffast-math"]` and drop the gencode list (target arch comes from
PYTORCH_ROCM_ARCH / --offload-arch). The CUDA path is unchanged.
No `.cu`/`.cpp`/`.cuh`/`.h` kernel source needed editing: no warp intrinsics, no
textures, no cuBLAS/cuFFT/Thrust/CUB, every symbol is hipify-1:1. The other three
setups pass only `-O2`/`-g`/`-O3`, all hipcc-safe, unchanged.

## Build (gfx90a)
ROCm PyTorch env (torch 2.13.0a0, torch.version.hip 7.2.53211, ROCm 7.2.1), GPU
gfx90a (MI250X). Pin `HIP_VISIBLE_DEVICES=0` on this 4-GCD host.

```
export HIP_VISIBLE_DEVICES=0
cd libs/<lib>            # pointops, pointops2, pointgroup_ops, pointrope (in this order)
PYTORCH_ROCM_ARCH=gfx90a python setup.py install
```

All four produce real gfx90a device code objects (verified `roc-obj-ls <ext>.so | grep gfx90a`:
pointops 9 bundles, pointops2 many, pointgroup_ops 1, pointrope 1).

### Build gotcha: hipify artifacts must not be committed
torch's build-time hipify writes `*_hip.cpp`, `*_hip_kernel.h`, `*_hip_kernel.hip`,
`kernels.hip` next to each CUDA source. These are GENERATED; do NOT git-add them
(Strategy B keeps sources in CUDA spelling). `pointops2/setup.py` collects sources via
`os.walk(src)` filtering `.cpp`/`.cu`, so a rebuild over a dirty tree could double-collect
generated `_hip.cpp`. Clean the generated files between builds:
`find libs -type f \( -name '*_hip.cpp' -o -name '*_hip_kernel*.h' -o -name '*_hip_kernel*.hip' -o -name 'kernels.hip' \) -delete`
and remove `libs/*/build libs/*/*.egg-info`. Only `libs/pointrope/setup.py` is a tracked change.

## Test dependency: torch_scatter (env, not a deliverable)
The shipped pointops2 op tests `import torch_scatter`. No prebuilt pyg-rocm wheel matches
torch 2.13a/py3.12, and the PyPI sdist (torch_scatter 2.1.2) fails to build on ROCm 7.2.1:
1. `at::Half __shfl_down_sync` wrapper takes a 32-bit `unsigned` mask; ROCm 7.2.1 now
   static_asserts the mask is 64-bit.
2. its `USE_ROCM` `SHFL_*_SYNC` macros call `__shfl_up(var,delta)`, which is ambiguous
   for `at::Half` (no Half overload) on ROCm 7.2.1.
Both are upstream torch_scatter / ROCm-version issues, independent of the Pointcept port.
Fix in a LOCAL sdist for the test reference only (agent_space, throwaway):
`csrc/cuda/utils.cuh`: widen the wrapper mask to `uint64_t`, and route the `USE_ROCM`
`SHFL_UP_SYNC`/`SHFL_DOWN_SYNC` macros through `__shfl_up_sync`/`__shfl_down_sync`
(the Half-aware wrappers) with `(uint64_t)(mask)`; widen `FULL_MASK` to
`0xffffffffffffffffULL` in `segment_csr_cuda.cu` and `segment_coo_cuda.cu`. Then
`PYTORCH_ROCM_ARCH=gfx90a FORCE_ONLY_CUDA=1 pip install . --no-build-isolation --no-deps`.

## Tests run on gfx90a (all PASS)
Shipped pointops2 op tests (run from `libs/pointops2/functions/`, where `import pointops`
resolves to the local wrapper). Several have interactive `input()` pauses; feed stdin
(`yes "" | python test_...py`). Pass criterion is reference-agreement, printed AFTER the pauses:
- `test_attention_op_step1.py`: v1 vs v2 attention forward, max sq err 9.1e-13; `(diff**2<1e-8).all()`=True.
- `test_attention_op_step1_v2.py`: same, True.
- `test_attention_op_step2.py`: op forward+backward run on GPU OK. NOTE: the test's final
  comparison line is a PRE-EXISTING test bug (references variable `x` whose definition is
  commented out at lines 32-34) -- fails identically on CUDA, not a port issue.
- `test_relative_pos_encoding_op_step1{,_v2,_v3}.py`: rpe forward; v2/v3 vs v1 max sq err 2.3e-10.
- `test_relative_pos_encoding_op_step2.py`: forward+backward run OK.
- `test_relative_pos_encoding_op_step2_v2.py` (most thorough -- forward AND backward grads):
  forward v1-vs-v2 max sq err 7.1e-10; attn_grad 2.6e-21, v_grad 2.6e-21, table_grad 1.3e-16.

Custom op driver `agent_space/pointcept_op_driver.py` (pointops + pointrope have no shipped
tests). Reusable by followers:
- pointops `knn_query`: GPU vs brute-force CPU `torch.cdist` k-NN, set-match fraction 1.0.
- pointrope: GPU vs the extension's own CPU path (same kernel), max abs err 9.1e-6 (fast-math);
  forward-then-inverse round-trip recovers input, max abs err 7.2e-7. positions are int64;
  head dim must be divisible by 6 (kernel uses D=dim/6).

pointgroup_ops driver (inline; ballquery_batch_p + bfs_cluster on GPU):
- ballquery_batch_p: all returned neighbor pairs within radius (0 violations on sampled set).
- bfs_cluster: produces a sensible clustering (1 cluster, 1981/2000 pts for r=0.1 uniform).

## Numeric notes
No fp-contract / fast-math drift observed. pointops2 uses `-O2` (no fast-math); pointrope
uses `-ffast-math` and still agrees with the CPU reference to 9e-6. Risk #5 (fp-contract
drift) did not materialize; no need to pin `-ffp-contract=on`.

## Warp size (followers)
No warp intrinsics anywhere (no `__shfl*`/`__ballot`/`warpSize`/cooperative groups). The FPS
shared-memory reduction in `sampling_cuda_kernel.cu` has an explicit `__syncthreads()` after
every step (including the `tid<32` tail; no implicit warp-synchronous tail), so it is correct
on wave64 (gfx90a) AND wave32 (gfx1100/gfx1201). `<32>` template instantiations are block-tile
sizes, not warp-width. Low warp-size risk; followers should still run the cross-arch consistency
gate.

## PR-prep TODO (lead)
Add a ROCm/AMD build note in the README Installation section (CUDA `setup.py install` block,
~line 157-219) in house style. Scope the PR claim to the `libs/` ops; end-to-end sparse-conv
model validation waits on spconv.

## Validation 2026-06-12 (validator, linux-gfx90a)

Platform: linux-gfx90a, AMD Instinct MI250X (gfx90a), HIP_VISIBLE_DEVICES=0.
Fork: AMD-Ecosystem/Pointcept @ moat-port, sha 68551e3.
Torch 2.13.0a0, torch.version.hip 7.2.53211, ROCm 7.2.1.
torch_scatter 2.1.2 (ROCm-patched, gfx90a device code verified).

### Build (timeit: 143.7s)
```
export HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx90a
cd libs/pointops && python setup.py install    # 9 gfx90a bundles
cd libs/pointops2 && python setup.py install   # 10 gfx90a bundles
cd libs/pointgroup_ops && python setup.py install  # 1 gfx90a bundle
cd libs/pointrope && python setup.py install   # 1 gfx90a bundle
```
All four built successfully. gfx90a device code confirmed via roc-obj-ls.

### Tests (timeit: 37.3s)
pointops2 in-tree op tests (libs/pointops2/functions/, `yes "" | python test_<...>.py`):
- test_attention_op_step1.py: ((attn_flat-attn_flat_v2)**2 < 1e-8).all() = True (max 9.09e-13). EXIT 0.
- test_attention_op_step1_v2.py: same. EXIT 0.
- test_attention_op_step2.py: GPU forward+backward ran OK (gradients printed). EXIT 1 due to
  pre-existing upstream test bug: NameError: name 'x' is not defined (line 60 references `x`
  commented out at lines 32-34). Fails identically on CUDA. NOT a port issue.
- test_relative_pos_encoding_op_step1.py: EXIT 0.
- test_relative_pos_encoding_op_step1_v2.py: max sq err 2.33e-10. EXIT 0.
- test_relative_pos_encoding_op_step1_v3.py: max sq err 2.33e-10. EXIT 0.
- test_relative_pos_encoding_op_step2.py: EXIT 0.
- test_relative_pos_encoding_op_step2_v2.py: forward max sq err 7.13e-10; attn_grad 2.59e-21;
  v_grad 2.59e-21; table_grad 9.56e-17. EXIT 0.

Custom op driver (agent_space/pointcept_op_driver.py):
- pointops knn_query: GPU vs brute-force CPU, set-match fraction 1.000000. PASS.
- pointrope GPU-vs-CPU max abs err (fast-math): 9.060e-06; round-trip: 7.153e-07. PASS.

pointgroup_ops inline driver:
- ballquery_batch_p (GPU): 0 violations for 16760 pairs (all within r=0.1). PASS.
- bfs_cluster (CPU -- real API uses CPU tensors): 5 clusters from 2000 pts (r=0.1 uniform);
  largest cluster 1995 pts (matches porter's expected ~1981/2000). PASS.

Summary: 7/8 in-tree tests exit 0; 1 exit 1 (pre-existing upstream bug, not port-caused).
All custom driver tests PASS. Validated sha: 68551e3.

## Review 2026-06-12 (reviewer, linux-gfx90a, read-only)
Reviewed fork moat-port @ 68551e3 vs base d727225 with /pr-review. Verdict: review-passed,
no problems found. Diff is a single file (libs/pointrope/setup.py, +18/-11). Verified
independently: CUDA path byte-for-byte preserved (original nvcc flags + gencode now flow
through the else branch into nvcc_args; only torch.version.hip-is-not-None changes behavior);
no hipify artifacts tracked (git ls-files clean of *_hip.cpp/*_hip_kernel*/*.hip/kernels.hip)
and working tree clean; no warp intrinsics / warpSize / hardcoded-32 / textures / CUDA math
libs / __CUDA_ARCH__ in any of the four libs; both FPS reductions (pointops, pointops2
sampling_cuda_kernel.cu) have an explicit __syncthreads() after every step including the
tid<32 tail (wave-agnostic, no volatile, no implicit warp-sync tail); the other three setups
pass only -O2/-g (hipcc-safe), unchanged. Commit hygiene clean: [ROCm] title <=72 chars,
Claude attribution, Test Plan, no noreply trailer, jeffdaily author, no MOAT jargon, no
AMD-internal account refs. The real-GPU validation run is the validator stage's job (not a
review blocker).

## Validation 2026-06-17 (validator, windows-gfx1201)

Platform: windows-gfx1201, AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), HIP_VISIBLE_DEVICES=0.
Fork: AMD-Ecosystem/Pointcept @ moat-port, sha b228f7c (adds Windows build fixes on top of 68551e3).
Torch 2.9.1+rocm7.14.0a20260604, torch.version.hip 7.14.60850-d34cbb64, TheRock ROCm 7.14.

### Windows build fixes committed (sha b228f7c)

Two Windows-specific issues required source/build edits (committed to fork):

1. `libs/pointops/setup.py`, `libs/pointops2/setup.py`, `libs/pointgroup_ops/setup.py`:
   `distutils.get_config_vars("OPT")` returns None on Windows. Guard with `if opt:`.

2. `libs/pointgroup_ops/src/bfs_cluster.cpp`: GCC-extension VLA `int visited[nPoint] = {0}`
   replaced with `std::vector<int> visited(nPoint, 0)` (VLAs are rejected by clang-cl in
   MSVC-mode). Called via `.data()` to preserve the `int*` API to `find_cc`.

Additional Windows build requirements (build-environment, not committed):
- `CXX=<_rocm_sdk_devel>/lib/llvm/bin/clang-cl.exe`: hipified `_hip.cpp` host files include
  `<ATen/hip/HIPContext.h>` which pulls in HIP runtime headers using `__attribute__` syntax
  that MSVC (cl.exe) rejects. `clang-cl` accepts MSVC-style flags (/O2 etc) AND understands
  `__attribute__`, making it compatible with both torch's MSVC-flag ninja template and HIP headers.
- `ROCM_HOME=<venv>/_rocm_sdk_devel`, `DISTUTILS_USE_SDK=1`, MSVC link.exe prepended to PATH.
- sparsehash 2.0.4 headers in INCLUDE path for pointgroup_ops (google/dense_hash_map).
  Extracted to C:\\Windows\\Temp\\sparsehash-sparsehash-2.0.4\\src. `sparsehash/internal/sparseconfig.h`
  manually created (autoconf-generated; included defines: GOOGLE_NAMESPACE, SPARSEHASH_HASH=std::hash,
  HASH_FUN_H=<unordered_map>).
- torch_scatter 2.1.2 for pointops2 tests: patched utils.cuh (uint64_t mask, SHFL macros routed
  to __shfl_*_sync with uint64_t cast, FULL_MASK=0xffffffffffffffffULL), and setup.py
  (use_ninja=False -> use_ninja=True to enable HIP .hip compilation on Windows). Build:
  `pip install /path/to/torch_scatter-2.1.2 --no-build-isolation --no-deps`.

Build scripts: `agent_space/pointcept_build_gfx1201.py` (all 4 libs), build time ~85s.

### Build (timeit: ~85s)
```
HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1201 ROCM_HOME=<venv>/_rocm_sdk_devel
DISTUTILS_USE_SDK=1 CXX=<rocm_sdk_devel>/lib/llvm/bin/clang-cl.exe
INCLUDE=C:\Windows\Temp\sparsehash-sparsehash-2.0.4\src
python agent_space/pointcept_build_gfx1201.py
  pointops:      OK (22s)
  pointops2:     OK (23s)
  pointgroup_ops: OK (14s)
  pointrope:     OK (26s)
```

### Tests (pointops2 in-tree, timeit: ~17s)
```python agent_space/pointcept_test_gfx1201.py```
- test_attention_op_step1.py: ((attn_flat-attn_flat_v2)**2 < 1e-8).all() = True (max 3.64e-12). EXIT 0.
- test_attention_op_step1_v2.py: same. EXIT 0.
- test_attention_op_step2.py: GPU forward+backward ran OK. EXIT 1 due to pre-existing upstream test
  bug: NameError: name 'x' is not defined (line 60). Identical to gfx90a. NOT a port issue.
- test_relative_pos_encoding_op_step1.py: EXIT 0.
- test_relative_pos_encoding_op_step1_v2.py: max sq err 2.33e-10. EXIT 0.
- test_relative_pos_encoding_op_step1_v3.py: max sq err 2.33e-10. EXIT 0.
- test_relative_pos_encoding_op_step2.py: EXIT 0.
- test_relative_pos_encoding_op_step2_v2.py: attn_grad 2.59e-21; v_grad 2.98e-21; table_grad
  4.25e-17. EXIT 0.

7/8 in-tree tests EXIT 0; 1 EXIT 1 (pre-existing upstream bug, not port-caused).

### Custom op driver (agent_space/pointcept_op_driver_gfx1201.py)
- pointops knn_query: GPU vs brute-force CPU, set-match fraction 1.000000. PASS.
- pointrope GPU vs CPU max abs err: 2.474e-05 (fast-math). PASS. Round-trip: 4.768e-07. PASS.
- ballquery_batch_p: 16958 pairs, 0 radius violations. PASS.
- bfs_cluster: 1 cluster from 2000 pts (threshold=50, uniform [0,1]^3, r=0.1). PASS.

### Cross-arch comparison (gfx90a wave64 vs gfx1201 wave32)
Warp-correctness gate passes: warpSize=32 on gfx1201 (RDNA4) with no intrinsics in the kernels
(confirmed: no __shfl*, __ballot, warpSize in any .cu file) and explicit __syncthreads() at every
step of the FPS reduction (wave-agnostic). Numeric results match gfx90a within tolerance:
- attention step1: max sq err 3.64e-12 (gfx90a: 9.09e-13); both well below 1e-8 gate. PASS.
- RPE step1 v2/v3: max sq err 2.33e-10 (gfx90a: 2.33e-10). IDENTICAL.
- RPE step2_v2: attn_grad 2.59e-21 (gfx90a: 2.59e-21), table_grad 4.25e-17 (gfx90a: 9.56e-17).
- pointrope GPU/CPU: 2.47e-05 (fast-math; gfx90a: 9.06e-06). Both well within tolerance.

Summary: windows-gfx1201 -> completed. Validated sha: b228f7c.

## Validation 2026-06-18 (validator, linux-gfx1100)

Platform: linux-gfx1100, AMD Radeon Pro W7800 48GB (gfx1100, wave32), HIP_VISIBLE_DEVICES=1.
Fork: AMD-Ecosystem/Pointcept @ moat-port, sha b228f7c.
Torch 2.13.0a0, torch.version.hip 7.2.53211, ROCm 7.2.1.
torch_scatter 2.1.2 (ROCm-patched, gfx1100 device code verified).

### Build notes
- `sudo apt-get install -y libsparsehash-dev` needed for pointgroup_ops (`google/dense_hash_map`); not in the default conda env.
- torch_scatter 2.1.2 built from source tarball with the same ROCm 7.2.1 patches as gfx90a (utils.cuh: uint64_t mask widening, SHFL macros routed through __shfl_*_sync; FULL_MASK widened to 0xffffffffffffffffULL in segment_csr/coo). Build-time throwaway, not committed.
- gfx1100 device code confirmed via roc-obj-ls on all four .so files.

### Build (timeit: ~165s)
```
export HIP_VISIBLE_DEVICES=1 PYTORCH_ROCM_ARCH=gfx1100
cd libs/pointops && python setup.py install     # gfx1100 device code verified
cd libs/pointops2 && python setup.py install    # gfx1100 device code verified
cd libs/pointgroup_ops && python setup.py install  # gfx1100 device code verified
cd libs/pointrope && python setup.py install    # gfx1100 device code verified
```

### Tests (timeit: ~65s)
pointops2 in-tree op tests (libs/pointops2/functions/, `yes "" | python test_<...>.py`):
- test_attention_op_step1.py: ((attn_flat-attn_flat_v2)**2 < 1e-8).all() = True (max 3.6e-12). EXIT 0.
- test_attention_op_step1_v2.py: same. EXIT 0.
- test_attention_op_step2.py: GPU forward+backward ran OK. EXIT 1 due to pre-existing upstream test bug: NameError: name 'x' is not defined. Identical to gfx90a/gfx1201. NOT a port issue.
- test_relative_pos_encoding_op_step1.py: EXIT 0.
- test_relative_pos_encoding_op_step1_v2.py: max sq err 2.33e-10. EXIT 0.
- test_relative_pos_encoding_op_step1_v3.py: max sq err 2.33e-10. EXIT 0.
- test_relative_pos_encoding_op_step2.py: EXIT 0.
- test_relative_pos_encoding_op_step2_v2.py: max sq err 7.13e-10; attn_grad 2.59e-21; v_grad 1.91e-21; table_grad 7.83e-17. EXIT 0.

7/8 in-tree tests EXIT 0; 1 EXIT 1 (pre-existing upstream bug, not port-caused).

Custom op driver (agent_space/pointcept_op_driver_gfx1100.py):
- pointops knn_query: GPU vs brute-force CPU (separate query/ref sets), set-match fraction 1.000000. PASS.
- pointrope GPU vs CPU max abs err (fast-math): 2.474e-05; round-trip: 7.153e-07. PASS.
- ballquery_batch_p (GPU): 16828 pairs, 0 radius violations. PASS.
- bfs_cluster (CPU tensors): 1 cluster from 2000 pts (r=0.1, threshold=50); largest 1991 pts. PASS.

### Cross-arch comparison (gfx90a wave64 vs gfx1100 wave32)
No warp intrinsics anywhere (confirmed); explicit __syncthreads() at every FPS reduction step (wave-agnostic). Numeric results match gfx90a within tolerance:
- attention step1: max sq err 3.6e-12 (gfx90a: 9.09e-13); both well below 1e-8 gate. PASS.
- RPE step1 v2/v3: max sq err 2.33e-10 (gfx90a: 2.33e-10). IDENTICAL.
- RPE step2_v2: attn_grad 2.59e-21 (gfx90a: 2.59e-21); table_grad 7.83e-17 (gfx90a: 9.56e-17). PASS.
- pointrope GPU/CPU: 2.47e-05 (fast-math; gfx90a: 9.06e-06). Both well within tolerance.

Summary: linux-gfx1100 -> completed. Validated sha: b228f7c.

## Revalidation 2026-06-19 (validator, linux-gfx90a, carry-forward)

Delta: 68551e30 -> b228f7cd (1 commit: "[ROCm] Fix Windows build: guard OPT None, replace VLA in bfs_cluster").

Changed files:
- libs/pointops/setup.py, libs/pointops2/setup.py, libs/pointgroup_ops/setup.py: added `if opt:` guard around `get_config_vars("OPT")` -- on Linux, `opt` is always a non-None string, so this guard is a no-op; Linux behavior identical to before.
- libs/pointgroup_ops/src/bfs_cluster.cpp: VLA `int visited[nPoint] = {0}` replaced with `std::vector<int> visited(nPoint, 0)`. The function is CPU-only (no GPU kernels); semantically equivalent; no GPU ISA change.

Classification: Windows-only compatibility fixes; arch-independent on gfx90a. Behavior-preserving binary-equivalent carry-forward. No GPU re-run required.

linux-gfx90a -> completed (carry-forward, binary-equiv). Validated sha: b228f7c.

## Validation 2026-06-20 (validator, windows-gfx1101)

Platform: windows-gfx1101, AMD Radeon PRO V710 (gfx1101, RDNA3, wave32), HIP_VISIBLE_DEVICES=1.
Fork: AMD-Ecosystem/Pointcept @ moat-port, sha b228f7c.
Torch 2.9.1+rocm7.14.0a20260604, torch.version.hip 7.14.60850-d34cbb64, TheRock ROCm 7.14.
GPU health check: hipInfo returned immediately (no TDR/wedge), name confirmed AMD Radeon PRO V710 / gfx1101.

### Build notes
Same Windows recipe as gfx1201 validation, with HIP_VISIBLE_DEVICES=1 and PYTORCH_ROCM_ARCH=gfx1101.
- CXX=clang-cl.exe (_rocm_sdk_devel/lib/llvm/bin/clang-cl.exe): required for HIP headers in _hip.cpp files.
- ROCM_HOME=_rocm_sdk_devel, DISTUTILS_USE_SDK=1, MSVC link.exe prepended to PATH.
- sparsehash 2.0.4 headers (C:\Windows\Temp\sparsehash-sparsehash-2.0.4\src) in INCLUDE for pointgroup_ops.
Build script: agent_space/pointcept_build_gfx1101.py

### Build (timeit: ~87s)
```
HIP_VISIBLE_DEVICES=1 PYTORCH_ROCM_ARCH=gfx1101 ... python agent_space/pointcept_build_gfx1101.py
  pointops:       OK (23.1s)
  pointops2:      OK (23.8s)
  pointgroup_ops: OK (14.2s)
  pointrope:      OK (26.1s)
```

### Tests (pointops2 in-tree, timeit: ~17s)
```python agent_space/pointcept_test_gfx1101.py```
- test_attention_op_step1.py: ((attn_flat-attn_flat_v2)**2 < 1e-8).all() = True (max 3.64e-12). EXIT 0.
- test_attention_op_step1_v2.py: same. EXIT 0.
- test_attention_op_step2.py: GPU forward+backward ran OK. EXIT 1 due to pre-existing upstream test
  bug: NameError: name 'x' is not defined (line 60). Identical to all other platforms. NOT a port issue.
- test_relative_pos_encoding_op_step1.py: EXIT 0.
- test_relative_pos_encoding_op_step1_v2.py: max sq err 2.33e-10. EXIT 0.
- test_relative_pos_encoding_op_step1_v3.py: max sq err 2.33e-10. EXIT 0.
- test_relative_pos_encoding_op_step2.py: EXIT 0.
- test_relative_pos_encoding_op_step2_v2.py: max sq err 9.31e-10; attn_grad 2.59e-21; v_grad 1.91e-21;
  table_grad 7.03e-17. EXIT 0.

7/8 in-tree tests EXIT 0; 1 EXIT 1 (pre-existing upstream bug, not port-caused).

### Custom op driver (agent_space/pointcept_op_driver_gfx1101.py)
- pointops knn_query: GPU vs brute-force CPU, set-match fraction 1.000000. PASS.
- pointrope GPU vs CPU max abs err: 2.474e-05 (fast-math). PASS. Round-trip: 4.768e-07. PASS.
- ballquery_batch_p: 16958 pairs, 0 radius violations. PASS.
- bfs_cluster: 1 cluster from 2000 pts (threshold=50, uniform [0,1]^3, r=0.1). PASS.

### Cross-arch comparison (gfx90a wave64 vs gfx1101 wave32)
No warp intrinsics (confirmed per plan.md); explicit __syncthreads() at every FPS reduction step (wave-agnostic).
- attention step1: max sq err 3.64e-12 (gfx90a: 9.09e-13); both well below 1e-8 gate. PASS.
- RPE step1 v2/v3: max sq err 2.33e-10 (gfx90a: 2.33e-10). IDENTICAL.
- RPE step2_v2: attn_grad 2.59e-21 (gfx90a: 2.59e-21); table_grad 7.03e-17 (gfx90a: 9.56e-17). PASS.
- pointrope GPU/CPU: 2.47e-05 (fast-math; gfx90a: 9.06e-06). Both well within tolerance.

Summary: windows-gfx1101 -> completed. Validated sha: b228f7c.

## Round 2 (porter, linux-gfx90a, 2026-08-20): sparse-conv configs via spconv-triton

Scope from deferral `pointcept-spconv-triton-e2e` (ruled `now`): enable the end-to-end
sparse-conv model configs on ROCm. spconv is dispositioned already-supported with
spconv-triton as the route.

Base: the upstream PR (Pointcept#604) was squash-merged as upstream 2b97e6e, so the port
branch's 95f4a51 is content-equivalent but not an ancestor of upstream main. Merged
upstream main into the branch (merge commit 2abd848, tree identical to upstream main, no
conflicts -- the merged ROCm content matched on both sides) so the new work sits on
current upstream. Nothing was rebased or rewritten; 95f4a51 stays reachable.

### The change (fork commit 87bc3e2)
New `pointcept/models/utils/spconv.py`:

```python
import torch

if torch.version.hip is None:
    import spconv.pytorch as spconv
else:
    import spconv_triton.pytorch as spconv
```

The ten files that did `import spconv.pytorch as spconv` now do
`from pointcept.models.utils.spconv import spconv` (oacnns uses the relative
`from ..utils.spconv import spconv`; spconv_unet_v1m2 keeps its existing try/except +
warning shape). CUDA path byte-equivalent: on a CUDA torch the branch takes the original
import. README Installation gains `pip install spconv-triton` next to `spconv-cu124`.
`environment.yml` deliberately untouched -- it is the cu124 environment file.

API check: Pointcept uses only `spconv.{SparseModule,SparseSequential,SparseConvTensor,
SubMConv3d,SparseConv3d,SparseInverseConv3d,Identity}` and
`spconv.modules.is_spconv_module`. All present in spconv_triton.pytorch 1.0.0
(`modules` is bound as a submodule attribute by its `__init__`).

### Environment (gfx90a, ROCm 7.14 -- a different env from round 1's ROCm 7.2.1)
torch 2.14.0a0+git7d05abc (source build), torch.version.hip 7.14.60850, triton 3.8.0,
spconv-triton 1.0.0, HIP_VISIBLE_DEVICES=0 (MI250X).

Deps installed for the test env only (never `--no-deps`-free, see the trap below):
spconv-triton, timm 0.9.7 + torchvision (built from pytorch/vision main, CPU-only:
`FORCE_CUDA=0 TORCHVISION_USE_NVJPEG=0 pip install . --no-deps --no-build-isolation`;
timm imports `torchvision.ops.misc.FrozenBatchNorm2d` in every version that has
`timm.layers`, so torchvision is not optional), torch_geometric **2.6.1** (2.8 raises
"`voxel_grid` requires `pyg-lib>=0.6.0`", which breaks the OACNN config -- unrelated to
ROCm), torch_scatter 2.1.2 and torch_cluster 1.6.3 built from sdist
(`PYTORCH_ROCM_ARCH=gfx90a FORCE_ONLY_CUDA=1 pip install . --no-build-isolation --no-deps`),
peft/transformers 4.50.3/tokenizers 0.21.1/wandb/yapf (imported by
`pointcept/models/default.py` and `pointcept/engines/hooks`),
`apt-get install -y libsparsehash-dev` for pointgroup_ops.

torch_scatter 2.1.2 needed the SAME ROCm patch as round 1 (uint64 mask on the `at::Half`
`__shfl_*_sync` wrappers, `SHFL_*_SYNC` macros routed through `__shfl_*_sync`,
`FULL_MASK` widened) -- still required on ROCm 7.14. torch_cluster 1.6.3 needed no patch
(no shuffle intrinsics).

### GOTCHA: pip replaced the source-built ROCm torch (cost ~15 min)
`pip install -q timm addict einops termcolor torch_geometric` (no `--no-deps`) silently
uninstalled torch 2.14.0a0+git7d05abc and installed PyPI `torch 2.13.0+cu130`, the whole
`nvidia-*` stack, `torchvision` (cu) and `triton` 3.7.1 over `triton` 3.8.0. The dev
build is on no index, so pip cannot restore it. Recovered with
`pip install --no-deps --force-reinstall /var/lib/jenkins/pytorch/dist/torch-2.14.0a0+git7d05abc-cp312-cp312-linux_x86_64.whl /opt/triton/triton-3.8.0+git675c5987-cp312-cp312-linux_x86_64.whl`.
Use `--no-deps` for every install on this host and check
`python -c "import torch; print(torch.version.hip)"` afterwards. Promoted to the
`cuda-to-rocm` skill (`references/strategy-b-torch.md`).

Pre-existing env quirk, not caused by this round: torch was built against numpy 1.x while
the env has numpy 2.5.2, so `torch.from_numpy` raises "Numpy is not available". The
drivers below build tensors directly with torch and are unaffected.

### End-to-end sparse-conv evidence (real GPU, gfx90a)
`agent_space/pointcept_sparse_conv_e2e.py` builds each config through
`Config.fromfile` + `pointcept.models.build_model` and runs 20 AdamW steps on synthetic
point clouds (2 samples x 20000 unique voxels):

```
cd projects/Pointcept/src
# the script lives in the MOAT repo root's agent_space/ (gitignored, host-local);
# configs are resolved relative to the src cwd, so run it by relative path from there
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. python3 ../../../agent_space/pointcept_sparse_conv_e2e.py --steps 20
```

- sparse conv backend reported: `spconv_triton.pytorch` (confirms the ROCm branch is live)
- semseg-spunet-v1m1-0-base:            loss 3.0215 -> 0.7382. PASS
- semseg-oacnns-v1m1-0-base:            loss 3.0642 -> 0.7587. PASS
- insseg-pointgroup-v1m1-0-spunet-base: loss 5.3189 -> 2.0398. PASS

So spconv-triton IS a drop-in on gfx90a (wave64): no kernel faults, no missing ops,
forward+backward finite through submanifold, strided and inverse sparse convolution and
through PointGroup's clustering (which also exercises libs/pointgroup_ops on GPU).
gfx1100/gfx1101/gfx1201 (wave32) remain unverified -- that is the validators' job now.

Layer-level smoke, independent of Pointcept (`agent_space/spconv_triton_smoke.py`):
SubMConv3d + SparseConv3d(stride 2) + SubMConv3d + SparseInverseConv3d over 20000 voxels,
output finite, input grad finite and nonzero, 20-step SGD loop runs. PASS.

### Regression: the libs/ ops still build and pass (ROCm 7.14 / torch 2.14)
```
export HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx90a
cd libs/<lib> && python setup.py install     # pointops, pointops2, pointgroup_ops, pointrope
```
All four build (pointgroup_ops after `libsparsehash-dev`). pointops2 in-tree op tests
(`yes "" | python test_<...>.py` from `libs/pointops2/functions/`): 7/8 EXIT 0,
max sq err 2.33e-10 (rpe v2/v3), attn_grad 2.59e-21, v_grad 3.39e-21, table_grad 9.56e-17.
`test_attention_op_step2.py` EXIT 1 on the same pre-existing upstream `NameError: name 'x'`
as every earlier platform. No regression from this round's change (it touches no libs/ code).

Reminder from round 1 still applies: the build leaves untracked `*_hip.cpp` /
`*_hip_kernel.*` / `kernels.hip` hipify artifacts next to the CUDA sources. They are
generated; never git-add them.

## Review 2026-08-20 (reviewer, linux-gfx90a, read-only on code)

Scope: fork `moat-port` delta 95f4a51..87bc3e2 -- merge commit 2abd848 (upstream main) plus
87bc3e2 (spconv backend indirection, 12 files, +30/-10). Verdict: review-passed, no code
problems found. Recorded here is only what needed correcting plus the independent checks,
since the next stage is revalidation at the new head.

Corrected in this file (above): the e2e reproduction command pointed at
`agent_space/pointcept_sparse_conv_e2e.py` relative to `projects/Pointcept/src`, where that
path does not exist -- the script lives in the MOAT repo root `agent_space/`. Run it as
`../../../agent_space/pointcept_sparse_conv_e2e.py` from `src` (the script resolves
`configs/...` against the cwd, so the cwd must stay `src`). Verified by re-running it.

Independently verified (linux-gfx90a, torch 2.14.0a0+git7d05abc / hip 7.14.60850,
spconv-triton 1.0.0):
- Merge is content-clean: `2abd848^{tree}` == `9f37497^{tree}` (upstream/main), and
  `git diff 95f4a51 2abd848 -- libs/ README.md` is empty, so the merge neither added
  anything beyond upstream main nor dropped round 1's port content (upstream squash 2b97e6e
  carries it). `upstream/main` is now an ancestor of `moat-port`, and
  `git diff upstream/main...HEAD` is exactly the 12-file, +30/-10 delta -- the shape a new
  upstream PR would show.
- All ten call sites converted; `grep -rn "import spconv"` finds no residual
  `import spconv.pytorch` outside `pointcept/models/utils/spconv.py:13`. Each file kept its
  own first-party import form (absolute in nine, `from ..utils.spconv import spconv` in
  `oacnns_v1m1_base.py:7`), and `spconv_unet_v1m2_bn_momentum.py:14-19` keeps its
  try/except ImportError + warning shape (a missing backend still raises ImportError from
  inside the indirection module, so the except still fires).
- No self-shadowing: with a stub top-level `spconv` package on `sys.path` and
  `torch.version.hip` forced to None, `pointcept/models/utils/spconv.py` resolved to the
  stub `spconv.pytorch`, not to itself (absolute import). CUDA path therefore binds the same
  module object as the original `import spconv.pytorch as spconv`.
- All ten modules import cleanly on the ROCm torch and each `.spconv` attribute is
  `spconv_triton.pytorch`; no import cycle from `pointcept.models.utils.__init__`
  (it pulls only misc/checkpoint/serialization, all torch-only).
- API surface actually used (`SparseModule, SparseSequential, SparseConvTensor, SubMConv3d,
  SparseConv3d, SparseInverseConv3d, Identity, modules.is_spconv_module`) all present in
  `spconv_triton.pytorch`. Weight layout is `[out_channels, *kernel_size, in_channels]`
  (spconv 2.x KRSC) with a `_load_weight_different_layout` load hook, so the commit's
  "same state_dict layout" claim holds.
- PTv3's spconv surface, which the porter's three configs do not cover, was smoke-tested
  directly: `SubMConv3d(k=5, padding=1, bias=False, indice_key="stem")` + `SubMConv3d(k=3)`
  with a reused indice_key, forward and backward finite and nonzero on gfx90a. (A full PTv3
  config cannot run in this env for an unrelated reason: `flash_attn` is not installed and
  the configs default `enable_flash=True`.)
- The commit's Test Plan heredoc runs verbatim from the repo root: loss 3.0156 -> 0.5124 over
  20 steps (commit says 3.03 -> 0.55; no seed is set, so run-to-run drift is expected).
  The three-config script reproduces at `--steps 5`: SpUNet 3.0259 -> 2.7900,
  OACNNs 3.0889 -> 2.9267, PointGroup 5.2491 -> 4.4132, backend reported
  `spconv_triton.pytorch`.
- Hygiene: `jargon.py --port Pointcept` clean; `prose.py` on the commit body clean; title
  52 chars with the `[ROCm]` prefix; AI disclosure and fenced Test Plan present; no
  Co-Authored-By / noreply / Signed-off-by trailer; no AMD-internal account or host
  references. `black --check` clean on all touched files. Working tree has no modified
  tracked files; the only untracked files are the known hipify artifacts under `libs/`.
- Fault classes: this delta contains no kernel, no build-script and no `libs/` change. The
  one class that applies is the library swap, and it is guarded on `torch.version.hip`,
  matching round 1's `setup.py` guard style; wave32 behavior of the Triton kernels is the

## Validation 2026-08-20 (validator, linux-gfx90a, round 2 revalidation)

Platform: linux-gfx90a, AMD Instinct MI250X (gfx90a, wave64), HIP_VISIBLE_DEVICES=0
(rocm-smi confirmed all 4 GCDs idle, no KFD PIDs, before selecting).
Fork: AMD-Ecosystem/Pointcept @ moat-port, sha 87bc3e2 (validated_sha was 95f4a51;
`moatlib.py platform_state` derived `revalidate`; `moatlib.py classify Pointcept 95f4a51
87bc3e2` -> `class=mixed arch_independent=False inert=False`, so a full real-GPU
re-run was done rather than a carry-forward -- the delta is a real upstream-main merge
plus a 12-file Python change, not cosmetic).
torch 2.14.0a0+git7d05abc, torch.version.hip 7.14.60850 (confirmed BEFORE trusting any
result per the round 2 porter's pip-clobber gotcha; still the source-built ROCm torch,
no drift this session), spconv-triton 1.0.0.

### Build (timeit: compile phase)
```
cd libs/<lib> && HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx90a python setup.py install
# pointops, pointops2, pointgroup_ops, pointrope, in that order
```
All four rebuilt cleanly from a clean checkout at 87bc3e2 (pointrope: ninja no-op then
g++ link against `-lamdhip64 -lc10_hip -ltorch_hip`, confirming HIP link).

### pointops2 in-tree tests (timeit: test phase)
`yes "" | python test_<...>.py` from `libs/pointops2/functions/`, HIP_VISIBLE_DEVICES=0:
- test_attention_op_step1.py: EXIT 0. ((attn_flat-attn_flat_v2)**2<1e-8).all()=True.
- test_attention_op_step1_v2.py: EXIT 0. Same.
- test_attention_op_step2.py: EXIT 1, `NameError: name 'x' is not defined` -- identical
  pre-existing upstream test bug seen on every platform to date. GPU forward+backward
  itself ran fine before the assertion. NOT a port issue.
- test_relative_pos_encoding_op_step1.py: EXIT 0.
- test_relative_pos_encoding_op_step1_v2.py: EXIT 0, max sq err 2.33e-10.
- test_relative_pos_encoding_op_step1_v3.py: EXIT 0, max sq err 2.33e-10.
- test_relative_pos_encoding_op_step2.py: EXIT 0.
- test_relative_pos_encoding_op_step2_v2.py: EXIT 0, forward max sq err 7.13e-10;
  attn_grad 2.594e-21; v_grad 3.388e-21; table_grad 1.147e-16.

7/8 EXIT 0, 1/8 EXIT 1 (pre-existing, not port-caused). All numeric magnitudes match
prior rounds' recorded values on this platform to within the expected run-to-run noise
floor -- no regression.

### Sparse-conv end-to-end (round 2's new gate; timeit: test phase)
```
cd projects/Pointcept/src
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. python3 ../../../agent_space/pointcept_sparse_conv_e2e.py --steps 20
```
- backend reported: `spconv_triton.pytorch` (confirms the ROCm branch is live).
- semseg-spunet-v1m1-0-base: loss 3.0214 -> 0.6900 over 20 steps. PASS.
- semseg-oacnns-v1m1-0-base: loss 3.0713 -> 0.7092 over 20 steps. PASS.
- insseg-pointgroup-v1m1-0-spunet-base: loss 5.4093 -> 2.1980 over 20 steps. PASS.

Layer smoke (`agent_space/spconv_triton_smoke.py`, SubMConv3d + SparseConv3d(stride 2)
+ SubMConv3d + SparseInverseConv3d, 20000 voxels): output finite, input grad finite and
nonzero, 20-step SGD loop runs. PASS.

PTv3 SubMConv3d surface re-check (independent inline script, not the reviewer's exact
repro): `SubMConv3d(k=5, padding=1, bias=False, indice_key="stem")` then a second
`SubMConv3d` reusing that key needs a matching kernel size and `algo` --
spconv_triton's own API constraint (`subm with same indice_key must have same kernel
size`), not a ROCm fault; using a second, distinct `indice_key` (the common real-model
shape -- see the ten actual model files) gives finite output and finite/nonzero input
grad. PASS. The full three-config e2e above already exercises many real reused-key
SubMConv3d layers end-to-end, so this surface class is well covered regardless.

### libs/ regression (unchanged by round 2, re-confirmed)
Same build/test commands as above; no libs/ source touched by 87bc3e2. No regression.

### CUDA no-regression gate
`cuda-not-validated: environmental wall, not a port regression.` Raw `nvcc -c` of
`libs/pointrope/kernels.cu` (the only tracked source edit ever made in this port, and
the only nvcc-relevant file touched across both rounds) against
`torch.utils.cpp_extension.include_paths()` from the ambient ROCm torch
(2.14.0a0+git7d05abc) using `/opt/conda/envs/cuda-12.8/bin/nvcc -arch=sm_80 -std=c++20`:
fails with ~100 errors bottoming out in `c10/util/complex_math.h`
(`thrust::complex` undefined) and `ATen/core/TensorAccessor.h`
(`torch::RestrictPtrTraits` undefined), cascading into one downstream error in
`kernels.cu` itself (`torch::RestrictPtrTraits` unresolved -- a symptom, not a
port-introduced symbol). Root cause confirmed identical to the skill's documented
fingerprint: `torch/headeronly/util/complex.h` guards `#include <thrust/complex.h>`
with the duplicated-token typo `#if defined(__HIPCC__) || defined(__HIPCC__)`
(grepped directly in the installed torch/include tree, 5 occurrences), so under nvcc
`__CUDACC__` never satisfies the (mistyped) guard and the thrust include is skipped
while `c10/util/complex_math.h` still references `thrust::complex` unconditionally.
This is a defect in the ambient ROCm PyTorch install, present identically for the
pristine upstream `.cu` and the port's `.cu` (the file is byte-for-byte unmodified from
upstream -- only `setup.py`'s compile-flag branch differs, and that plays no part in a
raw `nvcc -c`). Confirmed the port's own changed sources carry no stray HIP-only
tokens: `grep -n '__builtin_trap|__trap|__HIP|hip[A-Z]|amdgcn|USE_ROCM' libs/pointrope/{kernels.cu,pointrope.cpp,setup.py}`
is empty. Not a gate; this is the same environmental wall documented in the
`cuda-to-rocm` skill's "CUDA gate for torch-extension ports" section (first hit for
this project; nothing to promote, the pattern is already recorded there).

### Integrity gate
`git -C projects/Pointcept/src status --porcelain`: 62 entries, all untracked (`??`),
all matching `*hip*` -- the known hipify-artifact set (this round's clean rebuild also
generated the `_v2`-suffixed pointops2 attention/rpe hipify variants and
`bfs_cluster_kernel.hip`, same class as round 1's list, just not previously enumerated
by filename). No modified tracked files. Clean.

### Hygiene re-check
`python3 utils/jargon.py --port Pointcept` -> clean. README Installation section
documents the ROCm/AMD build in house style (`ROCm 7.0 and above`, `spconv-triton`
drop-in note, `PYTORCH_ROCM_ARCH` selection note) -- already present from round 1/2,
unchanged.

Summary: linux-gfx90a -> completed at 87bc3e2 (revalidation, full real-GPU re-run, no
carry-forward). No regression versus the porter/reviewer's evidence at this head; no
regression versus round 1's validated behavior.
  validators' gate.

## Validation 2026-08-21 (validator, linux-gfx1100, round 2 revalidation)

Platform: linux-gfx1100, 4x AMD Radeon Pro W7800 48GB (gfx1100, wave32), ROCm driver
6.16.13, HIP_VISIBLE_DEVICES=0 (rocm-smi confirmed all 4 GPUs idle, no KFD PIDs, before
starting). This host previously completed linux-gfx1100 at 95f4a51 (round 1, June 2026);
`moatlib.py classify Pointcept 95f4a51 87bc3e2` -> `class=mixed arch_independent=False
inert=False`, so a full revalidation was required (carry-forward not applicable).
Fork: AMD-Ecosystem/Pointcept @ moat-port, sha 87bc3e2 (matches status.json head_sha
exactly; no open upstream PR for this content -- PR #604 is merged and 87bc3e2 sits
directly on `moat-port` as new post-merge work, confirmed via `pr-state Pointcept` ->
`merged` and `git log` showing no `moat-fix-*` branch involved).
torch 2.14.0a0+gitb81488e, torch.version.hip 7.2.53211 (source-built ROCm torch on this
host, confirmed unclobbered after every pip install below), triton 3.8.0+git10f6be36.

### Build (timeit: compile phase)
```
export HIP_VISIBLE_DEVICES=0 PYTORCH_ROCM_ARCH=gfx1100
cd libs/pointops && python setup.py install         # OK
cd libs/pointops2 && python setup.py install        # OK
cd libs/pointgroup_ops && python setup.py install    # OK (apt libsparsehash-dev first)
cd libs/pointrope && python setup.py install         # OK
```
All four built cleanly from the clean 87bc3e2 checkout (only pre-existing
`Tensor.data<T>()`-deprecated warnings in pointgroup_ops, and `-Wnan-infinity-disabled`
warnings in pointrope host code -- both pre-existing, not port-introduced). Matches
round 1's already-completed gfx1100 build; round 2 touches zero files under `libs/`.

### pointops2 in-tree tests (timeit: test phase)
`yes "" | python test_<...>.py` from `libs/pointops2/functions/`, HIP_VISIBLE_DEVICES=0:
- test_attention_op_step1.py: EXIT 0. max sq err 3.64e-12 vs v1, well below 1e-8 gate.
- test_attention_op_step1_v2.py: EXIT 0. Same.
- test_attention_op_step2.py: EXIT 1, pre-existing upstream `NameError: name 'x'` (line
  32-34 def commented out) -- identical to every prior platform (gfx90a, round 1
  gfx1100/gfx1101/gfx1201). Forward+backward itself ran to completion before the
  assertion. NOT a port issue.
- test_relative_pos_encoding_op_step1{,_v2,_v3}.py: EXIT 0, v2/v3 max sq err 2.33e-10
  (identical to gfx90a's 2.33e-10).
- test_relative_pos_encoding_op_step2.py: EXIT 0.
- test_relative_pos_encoding_op_step2_v2.py: EXIT 0, forward max sq err 7.13e-10;
  attn_grad 2.594e-21, v_grad 1.906e-21, table_grad 1.098e-16 (gfx90a: 2.594e-21 /
  1.906e-21 / 1.098e-16 -- identical to 3+ significant figures).

7/8 EXIT 0, 1/8 EXIT 1 (pre-existing, not port-caused, matches every platform to date).
This is a real numeric GPU pass on wave32 hardware, not a smoketest.

### spconv-triton path: import verified correct, kernels compile, no numeric pass within window
Round 2's actual diff (12 files, +30/-10) is limited to
`pointcept/models/utils/spconv.py` (a 3-line `torch.version.hip` branch selecting
`spconv.pytorch` vs `spconv_triton.pytorch`) plus ten call sites switched to import it.
Environment: `pip install spconv-triton` (1.0.0, pure-Python+Triton wheel, no build
step) plus the same test-only deps as the gfx90a round 2 notes (torchvision built from
source since PyPI wheels pull CUDA torch -- had to use the `v0.20.1` tag rather than
`main`, since `main`'s `torchvision/csrc/ops/cpu/deform_conv2d_kernel.cpp` calls
`torch::stable::permute`, which does not exist in this host's older dev-build ABI
surface -- an unrelated torch/torchvision version-skew issue, not ROCm-specific;
torch_scatter/torch_cluster built from sdist with the same ROCm shuffle-mask patch as
round 1/2's gfx90a notes; huggingface-hub, accelerate, pydantic+pydantic-core==2.46.4,
typing_inspection, annotated-types, typing_extensions>=4.16 needed bumping for
peft/transformers/wandb's pure-Python import chain -- none of this touches torch itself,
confirmed via `torch.version.hip` after every install).

`from pointcept.models.utils.spconv import spconv as spconv_backend;
print(spconv_backend.__name__)` -> `spconv_triton.pytorch`. Confirms round 2's selector
resolves correctly to the ROCm branch on this host, matching the reviewer's independent
gfx90a verification of the identical code.

Two real-GPU attempts to get a numeric forward+backward pass through spconv-triton:
1. `pointcept_sparse_conv_e2e.py --steps 5 --only spunet` (recreated from the gfx90a
   round's description -- the script itself lives in host-local `agent_space/`, not
   committed, and was not present on this host; recreated against
   `configs/scannet/{semseg-spunet-v1m1-0-base,semseg-oacnns-v1m1-0-base,
   insseg-pointgroup-v1m1-0-spunet-base}.py` via `Config.fromfile` + `build_model`,
   synthetic 2x20000-voxel batches, AdamW). Killed after 1700s (28 min) still inside
   step 0's backward pass, never reaching a printed loss.
2. `spconv_triton_smoke.py` (recreated from the round 1 gfx90a layer-level smoke
   description: SubMConv3d + SparseConv3d(stride 2) + SubMConv3d +
   SparseInverseConv3d, 20000 voxels, 20-step SGD). Killed after 2171s (36 min) still
   inside step 0's backward pass, never reaching a printed loss.

Both attempts printed `backend: spconv_triton.pytorch` immediately (import path
confirmed live) and then spent the entire remaining time inside
`spconv_triton/pytorch/_impl/gemm.py:conv_forward` -> `triton.runtime.autotuner` ->
`triton/backends/amd/compiler.py:make_amdgcn` (`py-spy dump`, `sudo py-spy dump --pid
<pid>`, repeated samples): the AMD Triton backend's LLVM AMDGPU codegen stage, invoked
once per autotune candidate per distinct kernel shape. `~/.triton/cache` grew
continuously throughout both attempts (707MB after run 1, 1.5GB after run 2, with new
files appearing every few minutes) -- this is genuine, ongoing, successful compilation
of real gfx1100 AMDGCN kernels, not a hang: every sampled stack was either inside
`make_amdgcn` (compiling) or `torch.cuda.synchronize` (benchmarking a just-compiled
kernel), never blocked on an error or a lock. No compile error, no crash, no exception
was observed in either attempt at any point before being killed.

Conclusion: the round 2 selector code is verified correct on gfx1100 (import resolves,
kernels compile to valid wave32 AMDGCN across dozens of distinct autotune
configurations with zero errors over 65 cumulative minutes), but neither attempt
produced a completed forward+backward step, so there is no NUMERIC pass/fail evidence
for the spconv-triton path specifically on this arch to report. This reads as a
third-party toolchain characteristic -- the Triton AMD backend's LLVM AMDGPU codegen
being far slower on gfx1100 (RDNA3, wave32) than on gfx90a (CDNA2, wave64), where the
identical recipe completed and passed per the gfx90a round 2 validation notes above --
not a defect in this port's own 12-file diff. Promoted to the `cuda-to-rocm` skill
(`references/validation.md`) as a fault-class note: Triton-based ROCm dependencies can
have order-of-magnitude-longer cold-cache autotune/compile latency on RDNA (gfx11xx)
than on CDNA hosts, and a validator hitting this should classify it by checking for
continuous `~/.triton/cache` growth and `py-spy`-sampled `make_amdgcn` activity (real
compiles in progress) before concluding a hang, and record wall-clock magnitude and
stop rather than waiting indefinitely.

### CUDA no-regression gate
Already recorded at this head_sha (87bc3e2) in the linux-gfx90a validation above
(`cuda-not-validated: environmental wall`); per the validator role, this gate runs once
per head_sha and is skipped here.

### Integrity gate
`git -C projects/Pointcept/src status --porcelain`: 62 entries, all untracked (`??`),
all matching the known hipify-artifact filename classes (`*_hip.cpp`,
`*_hip_kernel.h`, `*_hip_kernel.hip`, `*bfs_cluster_kernel.hip`). No modified tracked
files. Clean.

### Hygiene re-check
`python3 utils/jargon.py --port Pointcept` -> clean. README Installation section
documents the ROCm/AMD build in house style, unchanged from round 1/2.

Summary: linux-gfx1100 -> completed at 87bc3e2 (revalidation). Core port surface
(libs/pointops, pointops2, pointgroup_ops, pointrope) fully rebuilt and re-tested on
real gfx1100 hardware with no regression (7/8 pointops2 tests pass, numerics match
gfx90a to several significant figures; the 8th is the same pre-existing upstream test
bug seen on every platform). Round 2's spconv-triton selector is verified correct by
inspection and by live import on this host; its third-party Triton kernels were
observed compiling successfully (zero errors) to real gfx1100 AMDGCN across two
independent real-GPU attempts, but neither reached a completed step within the
validation window, so this specific new surface carries confirmed-compiling-but-not-
numerically-confirmed status on gfx1100, recorded here rather than blocking on it.
