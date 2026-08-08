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

## Review 2026-08-08

Reviewed fork sha `ff80217` (moat-port vs `01c1623`) on linux-gfx1100. Verdict:
changes-requested. Review PR (never merged, feedback lives as line comments):
https://github.com/AMD-Ecosystem/Quest/pull/1

The shim-header approach is sound and the analysis behind it checks out; the defects are
all in the build file and the README.

1. `quest/ops/CMakeLists.txt:72` -- `-DCMAKE_HIP_ARCHITECTURES` is INERT. `find_package(Torch)`
   pulls `Caffe2/public/LoadHIP.cmake:107`, `set(CMAKE_HIP_ARCHITECTURES ${PYTORCH_ROCM_ARCH})`,
   a normal variable that shadows the cache entry, so `set_target_properties(... HIP_ARCHITECTURES
   "${CMAKE_HIP_ARCHITECTURES}")` re-applies torch's list. Measured: configuring with
   `-DCMAKE_HIP_ARCHITECTURES=gfx1201` produces `--offload-arch=gfx90a,gfx942,gfx950,gfx1100`
   and no gfx1201; `PYTORCH_ROCM_ARCH=gfx908` with no `-D` produces gfx908 alone. So the env
   var is the only working knob, the reverse of what the README says, and any user on an arch
   outside torch's default list gets hipErrorNoBinaryForGpu. The comment "Override Torch's
   multi-arch flags" says the opposite of what the line does.
2. `CMakeLists.txt:17` -- the `PYTORCH_ROCM_ARCH` / FATAL_ERROR block is dead. `enable_language(HIP)`
   on line 14 already defines `CMAKE_HIP_ARCHITECTURES` (CMakeDetermineHIPCompiler.cmake:296-334,
   via rocm_agent_enumerator, else CMake's own fatal error). Cache after configure reads
   `gfx1100;gfx1100;gfx1100;gfx1100` even with PYTORCH_ROCM_ARCH=gfx908 set. Hoist above
   `enable_language(HIP)`.
3. `CMakeLists.txt:29` -- same ordering bug on the CUDA leg, and it changes the NVIDIA build:
   `enable_language(CUDA)` runs first and CMakeDetermineCUDACompiler.cmake:261-267 caches nvcc's
   default arch, so `set(CMAKE_CUDA_ARCHITECTURES native)` never applies. Upstream had it before
   `project()` where it did. Likely a hard failure, not a slow path: flashinfer emits cp.async /
   mma.sync PTX that ptxas rejects below sm_80. Read off the CMake module; no nvcc on this host.
4. `CMakeLists.txt:4` -- the upstream `gcc-11`/`g++-11` compiler pins were deleted rather than
   guarded, contradicting the commit message's "nothing the NVIDIA build compiles or includes
   changes". `option(USE_HIP)` precedes `project()`, so `if(NOT USE_HIP)` restores them.
5. `README.md:69` -- the PYTORCH_ROCM_ARCH sentence is backwards (follows from 1).
6. `README.md:60` -- does not say the end-to-end path is unavailable. QuestAttention.py:117,127,151
   reach `_kernels.prefill_with_paged_kv_cache` and `_kernels.topk_filtering`, so scripts/passkey.sh
   and everything under evaluation/ -- the section immediately below this text -- cannot run.
7. `quest/ops/csrc/bsk_ops.{cu,h}` -- gratuitous reordering (apply_rope_in_place moved in both
   files, the guarded m.defs relocated to the bottom) turns a four-line additive diff into a
   whole-file reshuffle. Guard in place.

Checked and correct, so nobody re-derives it:

- Force-include mechanism: both shadow headers reuse the original guard macros
  (`FLASHINFER_MATH_CUH_`, `VEC_DTYPES_CUH_`), so the sibling-relative include of the original
  collapses. vec_dtypes.cuh differs from the vendored file ONLY in the vec_cast dispatch.
- math.cuh replacement is symbol-complete and numerically appropriate: ex2.approx -> exp2f
  (`v_exp_f32`), rcp.approx -> `__frcp_rn` (correctly rounded); both at least as accurate as the
  approximate instructions, against a 5e-3 reference. shfl.sync.bfly with clamp operand 0x1f is
  a 32-lane segment, which `__shfl_xor(x, mask, 32)` reproduces on both wavefront widths.
- rms_norm.cu mask change is required: amd_warp_sync_functions.h:307 static_asserts
  `sizeof(MaskT) == 8`, and torch's Caffe2Targets.cmake puts `-DHIP_ENABLE_WARP_SYNC_BUILTINS`
  on every extension compile.
- **Item 3 adjudicated: the porter is right, the plan was wrong.** Every shuffle in
  `blockReduceSum` states width 32, so `lane`/`wid` keep their meaning on wave64 and
  `shared[NUM][33]` still bounds `blockDim.x/32 <= 32` entries. Rewriting to `warpSize` would
  have been a regression for no gain. Same for the in-scope decode path: decode_attn.cuh:102,158
  reduce over offsets < bdx with `static_assert(bdx <= 32)` at 824/923/1118.
- Scoped-out suites are honest: 49 + 29 failures re-run on gfx1100, all 78 `AttributeError:
  module 'quest._kernels' has no attribute ...`. Exported list is exactly the six ops the
  README names.
- Empty `cuda/pipeline` shim is correct: included by permuted_smem.cuh / decode.cuh /
  decode_attn.cuh, but no `cuda::pipeline` object is declared in the compiled sources.
- Commit hygiene clean: `[ROCm]` title 50 chars, Claude named, no noreply trailer, jargon clean,
  no non-ASCII and no em-dash in added lines, fork tree clean.
- Lesson promotion (6d42e1d) is correctly filed and item 3's outcome is reflected accurately:
  fault-classes.md states that an explicit width is already wave-agnostic and that rewriting it
  to `warpSize` is a regression risk, which is the correction the plan needed.
