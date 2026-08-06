# pytorch_scatter -- porting notes

## Disposition: SKIP (already-supported)

rusty1s/pytorch_scatter has mature, AUTHORITATIVE, merged upstream ROCm/HIP
support. No MOAT port is warranted. Recorded via
`triage.py skip rusty1s/pytorch_scatter --reason already-supported`.

This mirrors the two sibling PyG extensions just skipped (mmcv,
pytorch_cluster), but the evidence below is specific to pytorch_scatter and was
verified directly, not assumed.

## Evidence

### Build-time ROCm path in setup.py (Strategy B, torch CUDAExtension)
- setup.py:19  `WITH_CUDA = CUDA_HOME is not None or torch.version.hip`
- setup.py:37-38 strips stale generated `hip` files on rebuild
- setup.py:78-82 under `if torch.version.hip:` adds `('USE_ROCM', None)` to
  define_macros and undefs `__HIP_NO_HALF_CONVERSIONS__`
- setup.py:120-122 hipify abs-path work-around

So a `CUDAExtension` built against a ROCm torch auto-hipifies the `.cu`/`.cuh`
sources and links the HIP runtime. This is exactly the Strategy B build path
MOAT would otherwise add -- it is already present.

### ROCm-guarded device source (the fault classes are already handled)
- csrc/cuda/utils.cuh:9-47 -- USE_ROCM-guarded warp-shuffle shim:
  `warp_mask_t = unsigned long long` on ROCm (64-bit mask), `at::Half`
  overloads of `__shfl_up_sync`/`__shfl_down_sync`/`__shfl_up`/`__shfl_down`,
  a `__ldg(at::Half*)` shim, and `SHFL_UP_SYNC`/`SHFL_DOWN_SYNC` macros that
  resolve to the MASKLESS `__shfl_up`/`__shfl_down` on ROCm (avoiding the HIP
  masked-sync mask-mismatch abort) and to the `_sync` forms on CUDA.
- csrc/cuda/atomics.cuh:157 -- USE_ROCM-guarded `atomAdd(at::Half*)` decimal
  CAS path; BF16 atomAdd present for both backends.
- csrc/version.cpp:10-31 -- USE_ROCM -> `<hip/hip_version.h>` / `HIP_VERSION`.

### Wave64 correctness of the warp reductions (the one thing worth scrutiny)
The segment kernels use 32-relative warp reductions, which on a 64-lane CDNA
wavefront treat each wavefront as two 32-lane segments:
- segment_csr_cuda.cu -- reduction loop `for(i=TB/2; i>0; i/=2)` with TB a
  template param <= 32; SHFL_DOWN over the TB-sized sub-group, lane 0 writes.
  Sub-warp-relative, never reads beyond its TB lanes -> wave64-safe.
- segment_coo_cuda.cu:27-52 -- `lane_idx = row_idx & 31`, `for(i=1;i<32;i*=2)`
  prefix scan via SHFL_UP, accumulation gated by
  `lane_idx >= i && row_idx/D == (row_idx-i)/D` (segment contiguity), and the
  partial result is flushed with `Reducer::atomic_write` at every segment
  boundary (`lane_idx==31 || row_idx/D != (row_idx+1)/D || idx != next_idx`).
  Because the cross-lane aggregation is bounded to the 32-lane segment and any
  carry across the wavefront's two halves is completed by the atomic flush,
  the kernel is correct on wave64 as written. This is the AMD-validated,
  wheel-shipped behavior; MOAT does not need to re-derive it.

No cub/thrust usage in the device sources (greps empty). No textures, no
managed memory, no library swaps.

### Merged AMD/ROCm-led PR history (authoritative)
- #325 "enable ROCm build; add BF16 for ROCm and CUDA" -- author org **ROCm**,
  branch `ROCm:enable_rocm`, MERGED 2022-09-30. Foundational: setup.py
  torch.version.hip path, USE_ROCM, the SHFL macro shim, BF16.
- #411 "[ROCm] fixes ambiguous calls to shfl* ... c10::Half to __half" --
  author **ashwinma** (AMD), MERGED 2024-01-23. The at::Half shfl overloads.
- #498 "Add ROCm 6.4.3+ support" -- author **Looong01**, MERGED 2025-08-15.
- Our own **jeffdaily** PR #493 "[ROCm] do not use __shfl sync functions" was
  **CLOSED** 2025-07-22 (superseded), confirming the maintained path is the
  merged one, not a MOAT delta.

### Maintained ROCm wheels
- Looong01/pyg-rocm-build ships prebuilt torch-scatter ROCm wheels; latest
  release 2026-04-21, ROCm 5.7 up to 7.2.2, PyTorch 2.11, Python 3.10-3.14.

## Conclusion
Authoritative (ROCm-org + AMD-engineer authored, merged) upstream support plus
maintained wheels. Per PORTING_GUIDE "assess existing AMD support" this is a
skip; duplicating it would re-port already-merged, AMD-owned work. State left
unclaimed; the already-supported disposition gates re-adoption.
