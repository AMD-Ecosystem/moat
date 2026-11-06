# modular notes

## Intake screen (2026-11-05, gap-matrix seed)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict: ready, upstream; marked third-party-closed).

### Licence: mixed, per file

`utils/licenses.py check modular/modular` returns UNPARSED (GitHub NOASSERTION), so the files were read directly:

- https://github.com/modular/modular/blob/main/LICENSE: "Mojo and its contributions are licensed under the Apache License v2.0 with LLVM Exceptions ... Portions of the MAX source code and its binary distribution are licensed under the MAX License ... The remainder of the source is under the Apache License v2.0 with LLVM Exceptions."
- `Licenses/README.md`: "Different parts of the MAX source are licensed under either the APACHE_LICENSE or the MAX_LICENSE. Refer to each file's comments for the correct license."
- `Licenses/MAX_LICENSE` (Modular Inc, "a Qualcomm Company", last modified 2026-10-21): a limited source licence; derivative works may be distributed in object code form only, and any MAX portion stays under the MAX terms. Not an open source licence (tier 4 in substance).

Recorded as `Apache-2.0-WITH-LLVM-exception AND LicenseRef-MAX-License`, which tiers as 4 (the worst part binds). Which files fall under which licence was not enumerated; that would matter only if the project were taken up, and mixed per-part licensing is a person's decision.

Tooling note: `licenses.tier_of` checks ` WITH ` before ` AND `, so the spaced form `Apache-2.0 WITH LLVM-exception AND LicenseRef-MAX-License` splits on WITH first and returns tier 1, silently dropping the MAX part. The hyphenated tier-1 identifier was used to avoid that; the ordering is worth fixing on trunk.

### AMD support: native, upstream

- Upstream active (pushed 2026-11-05), not archived.
- MAX/Mojo targets AMD GPUs natively: `max/docs/site/system-requirements.mdx` says "we regularly verify model compatibility on NVIDIA B200 and AMD MI355X GPUs"; Linux only (Windows via WSL).
- Native HIP runtime driver in-tree: `MLRT/lib/Driver/DeviceContext/HIP/` (HIPDriver, HIPDeviceContext, HIPDeviceGraphBuilder, HIPCompilationDevice). AMD-specific Mojo kernels: 78 paths under `max/kernels/src/**/amd` (e.g. `linalg/matmul/gpu/amd/amd_4wave_matmul.mojo`, `linalg/arch/amd/block_scaled_mma.mojo`).
- CUDA surface for a CUDA-to-HIP port: effectively none. Only two `.cu` files exist: `max/kernels/benchmarks/misc/cutlass_benchmark.cu` (an out-of-tree CUTLASS comparison benchmark) and `max/kernels/test/gpu/device_context/data/vec_add.cu` (test data). Kernels are written in Mojo and compiled per vendor by the Mojo/MAX compiler, which is not open source.
- AMD work in flight upstream by contributors: #6736 RDNA3 W4A16 path, #6474 MXFP4 dequant fix on MI300X, #4883 gfx906. No gap a MOAT port could fill.

## Recommendation

Decline, reason `already-supported` (`not-a-target` would also fit): MAX/Mojo supports AMD GPUs natively with an in-tree HIP driver, AMD-specific kernels and MI355X verification, and there is no CUDA source to port; parts of the tree are also under the source-available MAX License (tier 4).
