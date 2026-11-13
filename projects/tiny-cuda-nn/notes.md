# tiny-cuda-nn notes

## Intake screen (2026-11-05, gap-matrix seed; matrix verdict: partial, community fork)

### Licence
- Top level `LICENSE.txt` is BSD-3-Clause (NVIDIA copyright), and has been since the initial commit 99c9a8be (2021-04-21); README "License and Citation" says "licensed under the BSD 3-clause license". GitHub reports NOASSERTION. Recorded `license_spdx` BSD-3-Clause.
- `python3 utils/licenses.py scan-nvidia agent_space/tiny-cuda-nn-screen` (depth-1 clone of the default branch) flags 4 files with the NVIDIA proprietary header ("NVIDIA CORPORATION and its licensors retain all intellectual property and proprietary rights ... without an express license agreement from NVIDIA CORPORATION is strictly prohibited"):
  - `bindings/torch/tinycudann/__init__.py` (added 2a3de3e6, 2022-02-14 "Add PyTorch extension")
  - `bindings/torch/tinycudann/modules.py` (added 2a3de3e6, 2022-02-14)
  - `dependencies/stbi/stbi_wrapper.cpp`, `stbi_wrapper.h` (added b3d5cae3, 2022-02-12)
- These are NOT stale pre-relicence headers: the repo was BSD-3 from day one, and these files were added ten months later already carrying the proprietary boilerplate. Every other source file read (src/, include/, `bindings/torch/tinycudann/bindings.cpp`) carries the BSD-3 header. The likeliest explanation is NVIDIA's internal template pasted by mistake, but the text on the files says otherwise and it has stood uncorrected for 4.5 years.
- Real hits, on code the port needs: `modules.py` is the PyTorch loader and must change for ROCm (it gates on `torch.cuda.get_device_capability`, iterates `ALL_COMPUTE_CAPABILITIES` and imports `tinycudann_bindings._<cc>_C`; a HIP build needs a gfx-keyed or single module name). `__init__.py` is a one-line re-export. The stbi wrapper is used only by `samples/mlp_learning_an_image` and `benchmarks/image` (the main C++ sample, which a validator would run).
- Submodules: `dependencies/cutlass` (NVIDIA/cutlass, BSD-3), `dependencies/fmt` (MIT), `dependencies/cmrc` (MIT). CUTLASS has no HIP equivalent; the fully-fused MLP's CUTLASS path would need a rocWMMA/CK reimplementation (planner's concern).
- Result against jeff's bar for NVIDIA-owned repos (clean file by file): FAILS. Two of the four hits are on files the port must edit or ship.

### Duplicate effort
- Upstream: no open or closed PRs matching hip/rocm/amd; README has no AMD mention. Upstream active (pushed 2026-09-29), not archived.
- No `ROCm/` or `AMD-Ecosystem/` repo of this name.
- Community ROCm ports (none is a GitHub fork; all separately named, licence NOASSERTION):
  - `amdadvtech/tiny-hip-nn` (organization account, created 2022-01, last push 2023-12): hipified port for MI200/gfx90a, mlp-learning-an-image sample only, fp32 matrix cores; README lists ROCm 5.x runtime gaps. Stale; AMD-adjacent but not authoritative.
  - `ZJLi2013/tiny-rocm-nn` (personal, created 2024-09, last push 2026-04, 4 stars): HIP + hipBLAS + rocWMMA port, MI300X benchmarks.
  - `Painter3000/tiny-rdna4-nn` (personal, created 2026-07, last push 2026-08): gfx1201/RDNA4 rocWMMA fused-MLP research port.
  - All three are non-authoritative community ports per `assess-existing-support.md`: hints only, no code to inherit.
- The `amd` and `AMD-AGI` orgs are behind SAML SSO for this token, so an AMD-internal effort there could not be checked.

### Viability
- Genuine CUDA: 19 `.cu`, fully-fused MLP on CUTLASS/WMMA, hash-grid encodings, PyTorch extension. High value (used by instant-ngp, nerfstudio and many NeRF/3DGS pipelines); the fused MLP needs an AMD-native (rocWMMA/MFMA, wave32 and wave64) rewrite rather than a mechanical port.

### Recommendation
Decline, `license-blocked`, unless jeff rules otherwise below. Worth porting on technical merit, but it fails the file-by-file bar on a file the port must modify.

### Question for jeff
tiny-cuda-nn is BSD-3 at top level (since the initial commit; README says BSD-3), but `bindings/torch/tinycudann/{__init__,modules}.py` and `dependencies/stbi/stbi_wrapper.{cpp,h}` carry NVIDIA's "proprietary rights ... use without an express license agreement is strictly prohibited" header, added in Feb 2022 after the BSD release, not stale from before it. The port must edit `modules.py` (its CUDA compute-capability loader). Do you (a) treat these four headers as covered by the repo-wide BSD-3 grant and take it up, (b) take it up confined to the BSD-3 files, writing a new BSD-3 HIP loader instead of editing `modules.py` and leaving the stbi sample untouched, or (c) decline as license-blocked?
