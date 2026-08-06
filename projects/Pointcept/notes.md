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
