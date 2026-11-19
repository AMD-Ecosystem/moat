# ComfyUI-3D-Pack notes

## Intake screen (2026-11-05, windows-gfx1151 host, intake agent)

Seed: third-party "AMD AI Software Gap Matrix" (verdict not-yet, AMD path none). Treated as a lead only.

Recommendation: **decline**, `SKIP_REASON` = `license-blocked`. Not a CUDA port: a Python aggregator of ComfyUI nodes. All of its in-tree CUDA is vendored third-party code, and most of the GPU work it needs sits in dependencies that MOAT has already declined or cannot redistribute. What is left would be a dependency-assembly job, and the critical pieces of it are licence-blocked.

### Licence

- Top level: `LICENSE` is MIT (tier 1). GitHub's field agrees. Recorded as `license_spdx` MIT with the evidence URL.
- Per file it is NOT clean. `python3 utils/licenses.py scan-nvidia agent_space/ComfyUI-3D-Pack-screen` (shallow clone of main @ 2025-11-17 merge of #545) finds **47 files carrying NVIDIA proprietary licence text** ("Any use, reproduction, disclosure or distribution ... without an express license agreement from NVIDIA ... is strictly prohibited", some tagged `SPDX-License-Identifier: LicenseRef-NvidiaProprietary`). I read the headers; they are real licence text, not copyright lines:
  - FlexiCubes copies (`MVs_Algorithms/FlexiCubes/*`, `Gen_3D_Modules/{CRM,CRM_T2I_V2,CRM_T2I_V3}/util/flexicubes*.py, tables.py`, `TRELLIS/.../flexicubes/*`, `Stable3DGen/.../mesh/flexicube.py, tables.py`): the 2023 NVIDIA proprietary header.
  - EG3D/StyleGAN3 torch_utils under `Gen_3D_Modules/Hunyuan3D_V1/svrm/ldm/modules/rendering_neus/` (`third_party/ops/bias_act.{cu,cpp,h,py}`, `custom_ops.py`, `dnnlib/*`, `misc.py`, `grid_sample_gradfix.py`, `utils/ray_*.py`, `renderer.py`, `synthesizer.py`): `LicenseRef-NvidiaProprietary`.
  - InstantMesh geometry/renderer (`Gen_3D_Modules/InstantMesh/models/geometry/**`, `models/renderer/**`, `utils/mesh_util.py`) and `CharacterGen/Stage_3D/material/tets/generate_tets.py`: same NVIDIA proprietary header.
- Other vendored licences (recursed into `Gen_3D_Modules/*`; no `.gitmodules`):
  - `Hunyuan3D_V2/`, `Hunyuan3D_2_1/hy3dpaint/` (and `Checkpoints/Diffusers/tencent/*`): Tencent Hunyuan 3D Community License, territory-restricted (does not apply in the EU, UK, South Korea). Two of the in-tree `.cu` files (`custom_rasterizer/.../rasterizer_gpu.cu`) are under it.
  - `PartCrafter/`, `TripoSG/`, `TriplaneGaussian/.../snowflake`, `.../pointclouds/LICENSE_POINTNET`: own LICENSE files (not read in full; not the deciding factor).
  - Most other module directories carry no licence file of their own and fall under the top-level MIT only by inclusion, which is not established provenance (they are copies of separately licensed upstream research repos).

### In-tree CUDA (about 180 KB, not the matrix's 106 KB)

| File | Origin / licence |
| --- | --- |
| `Hunyuan3D_V1/.../third_party/ops/bias_act.cu` | NVIDIA EG3D/StyleGAN3, NVIDIA proprietary |
| `Hunyuan3D_V1/.../third_party/ops/grid_sample.cu`, `gridsample_cuda.cu` (duplicates, 31 KB each) | double-backward grid_sample extension, no licence header; sits inside the NVIDIA-licensed `third_party/` tree |
| `Hunyuan3D_V2/.../custom_rasterizer/.../rasterizer_gpu.cu`, `Hunyuan3D_2_1/.../rasterizer_gpu.cu` | Tencent Hunyuan Community License |
| `StableFast3D/sf3d/.slangtorch_cache/texture_baker/.../texture_baker_cuda.cu` (98 KB) | generated slangtorch cache, not source; slangtorch emits CUDA only (no HIP target, cf. 3dgrut skip) |

None of this is the pack author's own CUDA.

### GPU dependencies and their MOAT status

From `requirements.txt`, `_Pre_Builds/_Build_Scripts/{dependencies.txt,build_config.yaml}` and imports:

| Dependency | Used by | MOAT status / licence |
| --- | --- | --- |
| nvdiffrast (NVlabs) | 14 module dirs incl. `mesh_processer`, DiffRastMesh, CRM*, InstantMesh, TRELLIS, Unique3D, MV_Adapter, LGM, CharacterGen | **skip license-blocked** (NVIDIA Source Code License, non-commercial) |
| diff-gaussian-rasterization (ashawkey fork) | LGM, TriplaneGaussian, GaussianSplatting | Inria/MPII Gaussian-Splatting License (non-commercial); not a MOAT project; simple-knn skipped not-a-target |
| spconv + cumm | TRELLIS, Stable3DGen | **skip already-supported** (spconv-triton drop-in) |
| pytorch3d | several | MOAT project `pytorch3d`, stage review-passed, upstream PR facebookresearch/pytorch3d#2039 |
| torch-scatter | build list | **skip already-supported** (upstream ROCm support) |
| nerfacc | NeRF | **skip ported-elsewhere** |
| slangtorch | StableFast3D | CUDA-only codegen; shader-slang/slang skipped not-a-target |
| diso (SarahWeiii) | TRELLIS-family meshing | CUDA extension, GitHub licence NOASSERTION, not screened |
| xformers, triton | many | upstream ROCm builds exist |
| TRELLIS code (vendored copy) | TRELLIS, Stable3DGen | related MOAT project `TRELLIS.2` (awaiting-fork) covers Microsoft's successor |

So this is a dependency-assembly job, not a port, and the two most-used GPU dependencies (nvdiffrast, diff-gaussian-rasterization) are non-commercial licences that MOAT has already declined or would decline. A ROCm-only subset that avoids nvdiffrast, the Inria rasterizer and the NVIDIA-licensed files would drop most of the pack's node families.

### Duplicate effort

- Upstream PRs (all states) matching hip/rocm/amd: none. Branches: `main`, `dev`, `feature/update-unet-models` -- none AMD.
- Issues: #537 "Advice/tips for AMD cards" (2025-09), closed with the maintainer-side reply "I don't think it's possible to run most of the models on AMD right now". #31 (hipconfig symlink, 2024) closed.
- README / DOCKER_INSTRUCTIONS / `_Pre_Builds/README.md`: no AMD/ROCm/HIP mention.
- No ROCm or AMD-Ecosystem repo of that name; web search finds no ROCm port of the pack (only general ComfyUI-on-ROCm images and a separate TRELLIS.2-on-ROCm guide).
- Duplicate: none.

### Upstream activity

MIT, 3.9k stars, 25 open issues, not archived. Last commit 2025-11-17 (DenisKochetov, #545); before that 2025-08-07. Effectively low-activity maintenance by a co-maintainer for about a year; the build pipeline pins CUDA wheels (torch 2.7.0 cu12.x) from a prebuilt-wheel HF repo.

### Why decline rather than take up

1. Licence: 47 NVIDIA-proprietary files plus Tencent territory-restricted code inside an MIT top level; under the approved policy an NVIDIA-encumbered repo may be ported only if it is clean file by file, and this one is not.
2. The real GPU dependencies are licence-blocked (nvdiffrast) or non-commercial (Inria rasterizer), and MOAT has already declined them; the rest are already supported elsewhere.
3. No CUDA of the pack's own to port; the in-tree `.cu` files are vendored third-party or generated code.

If a person wants AMD coverage for 3D generation in ComfyUI anyway, the better target is the individual model repos with clean licences (TRELLIS.2 is already in the queue), not this aggregator.
