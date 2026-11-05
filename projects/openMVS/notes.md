# openMVS notes

## Intake screen (2026-11-05, windows-gfx1151 host, agent)

Source of the lead: third-party "AMD AI Software Gap Matrix" seed (verdict unknown, AMD path none). Treated as a lead; findings below are from a shallow clone of `develop` @ `efa0b619366c5e343ef9c4279fc943cf53c082f6` (2026-11-03) under `agent_space/openMVS-screen`.

**Recommendation: fork (take up).** Native CUDA-to-HIP port of the three MVS kernels (PatchMatch depth maps, confidence, mesh refinement) plus the small driver-API utility layer, validated with the project's own CPU-vs-GPU MVS tests. No AMD effort exists anywhere; the maintainer has publicly said GPU acceleration is NVIDIA-only.

### Licence

- `python3 utils/licenses.py check cdcseacave/openMVS` -> `license=AGPL-3.0 tier=3` (strong copyleft: contributing needs a person's approval, recorded per project). Read `LICENSE` (GNU AGPL v3 text) and `COPYRIGHT.md` directly; recorded with `set-license-spdx openMVS AGPL-3.0 --evidence https://github.com/cdcseacave/openMVS/blob/develop/LICENSE`. This is a fact, not a gate at intake: tier 3 waits for a person before anything goes upstream.
- `COPYRIGHT.md` lists per-part third parties: SeaCave (Boost), histogram/ACRANSAC (MPL-2.0), ceres (BSD), lmfit (BSD-2), loopy-belief-propagation (BSD-3), Eigen (MPL-2.0 only), OpenCV (BSD), Boost, CGAL (GPL/LGPL), halfmesh (MIT). Documented per-part licensing, not ambiguity.
- Not listed in `COPYRIGHT.md`: the in-repo vcpkg overlay `ports/siftgpu/` carries a vendored SiftGPU (`ports/siftgpu/source/LICENSE`, UNC Chapel Hill, "educational, research and non-profit purposes") -- the same non-commercial component COLMAP vendors (tier 4 read on its own). Its `ports/siftgpu/vcpkg.json` mislabels it `"license": "BSD-3-Clause"`. Contribution-only scope means this does not block contributing; it does mean a binary built with the `siftgpu` feature is not something AMD may ship or depend on without a separate legal answer. Same `contribute-only` reasoning as colmap's notes.
- No git submodules (`.gitmodules` absent). `python3 utils/licenses.py scan-nvidia agent_space/openMVS-screen` -> "no NVIDIA proprietary licence text". Not an NVIDIA-owned repo.

### Duplicate effort

- Upstream docs: `grep -rniE 'rocm|\bhip\b|amd|gfx[0-9]' README* docs/` -> no hits.
- `gh pr list --repo cdcseacave/openMVS --state all --search "hip OR rocm OR amd"` -> only #859 (closed 2022, multi-GPU depth maps, CUDA; a false match). No branches matching hip/rocm/amd.
- Issues: #1085 "Can we use GPU other then CUDA of processing?" (closed). The maintainer (cdcseacave) answered: "OpenMVS GPU acceleration specifically targets NVIDIA GPUs through CUDA (no native OpenCL/ROCm/Metal support)" and pointed at the CPU path. A Metal backend has since been added (`libs/MVS/PatchMatchMetal.{h,mm,metal}`), which shows the maintainer accepts non-CUDA GPU backends.
- `gh search repos openmvs --owner AMD-Ecosystem` and `--owner ROCm` -> none. `gh search repos "openmvs rocm"` -> none. Web search (openMVS ROCm/HIP/AMD) -> nothing; OpenDroneMap states GPU densification needs an NVIDIA card.
- Duplicate: none.

### Viability and CUDA surface

- Upstream live: not archived, pushed 2026-11-05, ~4.2k stars, default branch `develop` (scaffold recorded with `--branch develop`).
- CUDA is an optional, first-class CMake language path: `OpenMVS_USE_CUDA` (default ON, forced OFF on macOS), `CHECK_LANGUAGE(CUDA)` / `ENABLE_LANGUAGE(CUDA)`, `CMAKE_CUDA_SEPARABLE_COMPILATION ON` + `CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON` (relocatable device code -- watch for the Windows `-fgpu-rdc` bundler fault on the laptop), `--expt-relaxed-constexpr`, links `CUDA::cuda_driver CUDA::cudart CUDA::curand`, defines `_USE_CUDA`. Also enables vcpkg manifest feature `cuda`, which pulls `ceres[cuda]` and `siftgpu[cuda]`. Falls back to CPU silently when CUDA is not found.
- No NVRTC and no PTX/module loading (`cuModule*` absent). The driver API is used only for memory/array management in `libs/Common/UtilCUDA.{h,cpp}` (~475 lines): `cuMemAlloc/Free`, `cuMemcpyHtoD/DtoH/2D`, `cuArray3DCreate/Destroy`, `cuMemsetD*`, `cuMemAllocHost`, `cuCtxSynchronize`, `cuGetErrorName/String`. All have HIP equivalents (`hipMalloc`-family / `hipDrv*` / `hipArray3DCreate`).
- Kernels (first-party, AGPL): `libs/MVS/PatchMatchCUDA.cu` (34 KB), `SceneRefineCUDA.cu` (52 KB), `ConfidenceCUDA.cu` (17 KB) plus `.inl`/`CUDA/*.h` helpers; ~105 KB total. Vendored SiftGPU `ports/siftgpu/source/src/ProgramCU.cu` (69 KB) is optional (vcpkg feature `siftgpu`).
- Fault-class flags for the planner: `cudaTextureObject_t` + `tex2D` (40/20 uses, cudaArray-backed -- texture pitch/format rules); `curand` device API (`curandState`, maps to hipRAND/rocRAND); `warpSize` (18 uses) and `__ballot_sync(0xffffffffu, ...)` + `__shfl_up_sync(0xffffffffu, ...)` in `SceneRefineCUDA.cu:199,273,282` -- 32-bit lane masks and a 32-bit ballot result, the wave64 hazard class; `CMAKE_CUDA_ARCHITECTURES` parsing that `FATAL_ERROR`s on non-numeric archs (needs a HIP branch). (Issue #1130's SM-to-cores table is gone from current `develop`: no `MapSMtoCores`/`ConvertSMVer` in `libs/`.)
- GPU tests exist: `apps/Tests` registers `CommonUnitTests`, `SFMPipelineTest`, `MVSPipelineTest` via CTest; `TestsMVS.cpp` runs a CPU and a CUDA refinement on the same synthetic fixture and scores both against ground truth (`#ifdef _USE_CUDA`, `SEACAVE::CUDA::isEnabled()`). Upstream CI builds with `-DOpenMVS_USE_CUDA=OFF`, so the GPU path is never CI-tested upstream.

### Dependencies and relation to COLMAP

- No hard MOAT dependency: the core port needs only HIP runtime + hipRAND/rocRAND. `depends_on` stays `[]`.
- Optional GPU deps (not MOAT projects): ceres-solver's `cuda` feature (cuBLAS/cuSOLVER dense solvers; would need hipBLAS/hipSOLVER or simply the CPU ceres on ROCm) and SiftGPU's `cuda` feature. Recommend the port scope the `cuda` vcpkg feature so a ROCm build uses plain `ceres` and leaves `siftgpu` on its GL/CPU path unless ported.
- COLMAP (MOAT project, BSD-3, `AMD-Ecosystem/colmap`) is the sibling in the same pipeline (COLMAP SfM -> openMVS densify/mesh/texture; openMVS imports COLMAP scenes). Shared patterns: (1) both have a PatchMatch MVS CUDA kernel built on texture objects + curand; COLMAP's merged HIP patch_match_stereo port (colmap/colmap#4420) is the nearest reference for texture/curand/HIP-CMake handling. (2) Both vendor the same UNC SiftGPU with `ProgramCU.cu`; the three SiftGPU faults found on colmap's parked `rocm-sift-gpu` work (CuTexObj rule-of-five double destroy, ComputeDOG OOB read, tex2D pitch alignment needing a linear binding) apply directly if openMVS's SiftGPU is ever in scope -- do that once and reuse. Shared SiftGPU work is optional and should not gate the core port.

### Proposed work (for the planner)

1. Add a HIP build path alongside CUDA in CMake (`LANGUAGE HIP` on the three `.cu` files or an `OpenMVS_USE_HIP` option, rocRAND/hipRAND, HIP driver-API shim for `UtilCUDA`), keeping the CUDA path untouched.
2. Port `PatchMatchCUDA`, `ConfidenceCUDA`, `SceneRefineCUDA`; fix wave64 masks/ballots in `SceneRefineCUDA.cu`.
3. Run `MVSPipelineTest` (CPU vs GPU on the synthetic fixture) plus a real DensifyPointCloud run on wave64, wave32 and Windows.
4. Docs change upstream describing the ROCm build. Tier-3 licence clearance by a person before anything goes upstream.

### Questions for jeff

None blocking. Recorded for the decision: openMVS is AGPL-3.0 (tier 3), so a licence clearance is needed before any upstream PR, and its optional SiftGPU overlay carries the same non-commercial licence as COLMAP's.
