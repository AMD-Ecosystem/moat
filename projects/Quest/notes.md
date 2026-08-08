# Quest notes

planner: perf-critical kernels -- assess a mechanical HIP port vs an AMD-native (rocWMMA/CK/MFMA) rewrite of the hot kernels; a correctness-first port is a valid first step.

## Porting attempt 2026-06-04

### Scope
Stage 1 only: `rms_norm.cu` and `topk.cu`. The flashinfer-dependent attention kernels (Stage 2) use NVIDIA-specific PTX intrinsics (mma.sync, ldmatrix, cp.async) that require an AMD-native rewrite.

### Build environment
- ROCm 7.2.1, PyTorch 2.13.0a0+gitb5e90ff (development build)
- a ported raft (no longer tracked here) built and installed at `_deps/raft/install`)

### Issues encountered

1. **PyTorch TensorImpl.h C++20 concepts incompatibility**: The PyTorch headers use `requires` clauses that fail to parse with the HIP compiler unless `-std=gnu++20` is used. Worked around by setting `CMAKE_HIP_STANDARD 20`.

2. **Multi-arch flags from Torch CMake**: `find_package(Torch)` adds `--offload-arch=gfx90a;gfx942;gfx950;gfx1100` from the PyTorch build. Worked around by setting `PYTORCH_ROCM_ARCH=gfx90a` environment variable.

3. **__HIP_NO_HALF_CONVERSIONS__ breaks static_cast<float>(__half)**: PyTorch ROCm sets this flag to disable implicit half conversions. The `rms_norm.cu` kernel uses `static_cast<float>(h->x)` which fails. Required explicit `__half2float()` calls.

4. **Name collision with PyTorch symbols**: A `half_to_float` macro clashed with `at::attr::half_to_float` in PyTorch's interned_strings.h. Renamed to `quest_h2f`.

5. **raft radix_topk_one_block_kernel overload resolution**: The raft kernel function pointer capture fails with "incompatible initializer of type '<overloaded function type>'". This is a blocking issue in how Quest's `decode_select_k.cuh` captures the raft kernel.

### Blocking issue
The raft select_k integration needs investigation. The Quest code captures a function pointer to raft's templated kernel but the HIP compiler cannot resolve the overload. This may require changes to how Quest calls raft's select_k API.

### Files modified (uncommitted, on moat-port branch)
- `quest/ops/CMakeLists.txt`: USE_HIP option, HIP language enable, raft via find_package
- `quest/ops/cmake/hip_build.cmake`: HIP-specific build config (unused, early return approach)
- `quest/ops/csrc/rms_norm.cu`: wave64 warp-size abstraction, half<->float conversion traits
- `quest/ops/csrc/bsk_ops.h`: USE_HIP guards for flashinfer-dependent declarations
- `quest/ops/csrc/bsk_ops.cu`: USE_HIP guards for pybind module
- `quest/ops/csrc/pytorch_extension_utils.h`: USE_HIP guard for DISPATCH macro
- `kernels/include/topk/decode_select_k.cuh`: HIP runtime aliases

## 2026-08-06 dependency

`depends_on: [raft]` cleared. raft is no longer a MOAT project (the ROCm-DS team owns
the RAPIDS domain), so it is an ordinary external build dependency here rather than
something this pipeline builds first. If the build needs it, install it from the
environment like any other third-party library.

## Resuming (2026-08-07)

The port continues: this was judged a port worth finishing rather than an unportable
codebase, so linux-gfx90a is no longer marked blocked. The blocker as last recorded, which
is where to pick it up:

raft select_k API incompatibility: radix_topk_one_block_kernel overload resolution fails in Quest's decode_select_k wrapper. PyTorch ROCm 2.13.0a0 development build has __HIP_NO_HALF_CONVERSIONS__ requiring explicit conversion calls. Need to investigate raft select_k integration pattern.

## Port on linux-gfx1100 (2026-08-08)

Fork sha `ff80217` on `moat-port`, replacing the earlier WIP commit (nothing had validated
it, so it was collapsed rather than built on).

Environment: Radeon Pro W7800 (gfx1100, wave32), ROCm 7.2.3, PyTorch 2.14.0a0 ROCm build,
python 3.12. The test suite needs `transformers==4.37.2` with `tokenizers==0.15.2`.

Build (from `quest/ops`, and note that a stale build directory must be wiped after editing
anything under `kernels/include/hip_compat`, since two of those headers are force-included
and carry no dependency edge):

```
mkdir -p build && cd build
cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)') ..
ninja
ln -sf $PWD/_kernels*.so ../../     # setup.sh puts it in quest/, NOT quest/ops/
```

Do NOT pass `-DCMAKE_HIP_COMPILER=...`: cmake then reports "variables have changed, cache
must be deleted", re-runs configure without the command-line `-DUSE_HIP=ON`, and falls into
the CUDA branch looking for nvcc. Leave the compiler to `enable_language(HIP)`.

GPU results (`HIP_VISIBLE_DEVICES=1`, from the repo root with it on PYTHONPATH):

| suite | result |
|---|---|
| test_rope.py | 64 passed |
| test_estimate.py | 9 passed |
| test_decode_attention.py | 6 passed |
| test_approx_attention.py | 42 passed |
| test_topk.py | 49 failed (topk_filtering not built) |
| test_prefill_attention.py | 29 failed (prefill_with_paged_kv_cache not built) |

What is out of scope and why, so the next session does not re-derive it:

- `topk.cu` -> RAFT `radix_topk_one_block_kernel`. The previous session's blocker (an
  overload-resolution failure capturing the kernel function pointer) was never resolved and
  is now moot in a different way: no RAFT is installed on this host and RAFT is no longer a
  MOAT project. Replacing it with a self-contained per-block top-k over the page scores
  (batch 32, len 1024-8192, k 64-256) is the obvious next deliverable and needs no external
  library. `test_topk.py` is its gate.
- `prefill.cuh` -> `mma_sync_m16n16k16_*` and `ldmatrix_m8n8x4`. Genuine AMD-native rewrite
  (WMMA on gfx11, MFMA on CDNA). `test_prefill_attention.py` is its gate.

Things that turned out NOT to be problems, contrary to the plan:

- `cp_async` needed nothing. flashinfer's `cp_async.cuh` guards its PTX on
  `__CUDACC_VER_MAJOR__ >= 11` and already carries a plain vectorized-load fallback, which
  is what a HIP compile selects. That alone is what put decode attention in reach.
- The `rms_norm.cu` reduction is wave-portable as written, because the shuffles state an
  explicit width of 32. Only the mask literal had to change. The `>>5` / `&0x1f` / `[33]`
  shapes the plan flagged are correct on wave64 and were left alone.
