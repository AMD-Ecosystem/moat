# MOAT: Moat Obliteration via Automated Translation

MOAT ports popular CUDA GitHub projects to ROCm/HIP, one repo at a time. Coverage is expressed as gates -- wave64, wave32, windows -- so a port is proven once each gate has an architecture that validated it on real hardware. It is driven by Claude: intake screens a candidate, a planner analyses it, a porter applies the change on a fork in the AMD-Ecosystem org, a reviewer checks it, and a validator runs the real tests on AMD GPUs. Submitting the result upstream, and everything that follows with a maintainer, stays with a person. This repo is the control plane; it tracks progress and holds the porting knowledge in the `cuda-to-rocm` skill.

## How it works

Each project gets a folder under `projects/` holding its plan, notes, and a per-platform status file. A fresh Claude CLI run in this repo detects its AMD architecture, finds the next actionable project, and continues the pipeline. No platform leads: whichever host picks the project up first does the porting, and the rest validate the same fork branch independently and in parallel, since the AMD targets share one unified ROCm port.

A platform is an architecture on an operating system, and it carries the gates implied by both -- its wavefront size, which is fixed by the architecture, and its OS. The same GPU on a different OS is a different platform covering different gates. A gate is satisfied once any one platform carrying that attribute has validated, so the gates can be covered by as few as two hosts.

## Licence

MIT, matching the ROCm projects this work targets. See [LICENSE](LICENSE). The ports
themselves live on forks and stay under each upstream project's own licence; nothing in
this repository changes that.

## Scope and honesty

The project list below is a best-effort ranked union of targeted GitHub searches, not a census of every CUDA repo (GitHub search caps results per query and misses repos whose dominant language is not Cuda). Ports aim to be minimally invasive: for pure CMake projects we prefer `enable_language(HIP)` plus a single cuda-to-hip compat header; for pytorch extensions we rely on torch's build-time hipify. A CPU-only build smoketest proves compilation only; correctness is gated on real-GPU test runs. See the `cuda-to-rocm` skill.

Projects we will not port (already ported, already supported, can't be ported, or not a real target) are recorded with reasons in `data/dispositions.json` and kept out of the actionable list; `utils/triage.py` manages those decisions.

## Projects

<!-- MOAT:TABLE:START -->
**Coverage is proven per gate, not per machine.** A gate is met once *any* AMD GPU
with that property has run the project's real test suite. Which specific card did it
is recorded in each project's status file.

| gate | what it covers |
|---|---|
| `wave64` | data-center cards -- Instinct MI200/MI300 class (CDNA), 64 threads per wavefront |
| `wave32` | desktop and workstation cards -- Radeon RX / PRO (RDNA), 32 threads per wavefront |
| `windows` | the port builds and runs on Windows, not only Linux |

A wavefront is AMD's analogue of an NVIDIA warp, which is always 32 threads. Code that
silently assumes 32 is the single most common way a CUDA port breaks on AMD, so the two
widths are proven separately rather than assumed to follow from each other.

| cell | meaning | | outcome | meaning |
|---|---|---|---|---|
| ✅ | proven on the current code | | 🟣 | contribution merged upstream |
| 🔄 | proven earlier; the code has moved since | | 🟢 | pull request open |
| 🔧 | in progress | | 🔴 | pull request closed |
| ⬜ | not started | | ⚖️ | licence bars contributing the port |
| 🚫 | blocked, with a reason recorded | | ⚪ | set aside, with the reason recorded |
| 🎫 | waived for this project, with maintainer approval | | — | nothing recorded |
| — | nothing recorded | | | |

The project name links upstream.

| Project | `wave64` | `wave32` | `windows` | Outcome |
| --- | :---: | :---: | :---: | --- |
| [3DGS-LM](https://github.com/lukasHoel/3DGS-LM) ([fork](https://github.com/AMD-Ecosystem/3DGS-LM/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#15](https://github.com/lukasHoel/3DGS-LM/pull/15) |
| [3DUNDERWORLD-SLS-GPU_CPU](https://github.com/theICTlab/3DUNDERWORLD-SLS-GPU_CPU) ([fork](https://github.com/AMD-Ecosystem/3DUNDERWORLD-SLS-GPU_CPU/tree/moat-port)) | 🔄 | ✅ | 🔄 | 🟢 [#33](https://github.com/theICTlab/3DUNDERWORLD-SLS-GPU_CPU/pull/33) |
| [3P-ADMM-PC2](https://github.com/Samarvivian/3P-ADMM-PC2) ([fork](https://github.com/AMD-Ecosystem/3P-ADMM-PC2/tree/moat-port)) | 🔄 | ✅ | ✅ | 🟣 [#10](https://github.com/Samarvivian/3P-ADMM-PC2/pull/10) |
| [accelerated-scan](https://github.com/proger/accelerated-scan) ([fork](https://github.com/AMD-Ecosystem/accelerated-scan/tree/moat-port)) | 🔄 | ✅ | ✅ | 🟢 [#17](https://github.com/proger/accelerated-scan/pull/17) |
| [aihwkit](https://github.com/IBM/aihwkit) ([fork](https://github.com/AMD-Ecosystem/aihwkit/tree/moat-port)) | 🔄 | ✅ | ✅ | 🟢 [#770](https://github.com/IBM/aihwkit/pull/770) |
| [alien](https://github.com/chrxh/alien) ([fork](https://github.com/AMD-Ecosystem/alien/tree/moat-port)) | ✅ | ✅ | 🔄 | 🟣 [#710](https://github.com/chrxh/alien/pull/710) |
| [amgcl](https://github.com/ddemidov/amgcl) ([fork](https://github.com/AMD-Ecosystem/amgcl/tree/moat-port)) | ✅ | ✅ | ✅ | 🟣 [#315](https://github.com/ddemidov/amgcl/pull/315) |
| [anari-visionaray](https://github.com/szellmann/anari-visionaray) ([fork](https://github.com/AMD-Ecosystem/anari-visionaray/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#10](https://github.com/szellmann/anari-visionaray/pull/10) |
| [arbor](https://github.com/arbor-sim/arbor) ([fork](https://github.com/AMD-Ecosystem/arbor/tree/moat-port)) | 🔄 | ✅ | ✅ | 🟣 [#2512](https://github.com/arbor-sim/arbor/pull/2512) |
| [arrayfire](https://github.com/arrayfire/arrayfire) ([fork](https://github.com/AMD-Ecosystem/arrayfire/tree/moat-port)) | 🔄 | ✅ | ✅ | 🟢 [#3708](https://github.com/arrayfire/arrayfire/pull/3708) |
| [AutoDock-GPU](https://github.com/ccsb-scripps/AutoDock-GPU) ([fork](https://github.com/AMD-Ecosystem/AutoDock-GPU/tree/moat-port)) | 🔄 | ✅ | ✅ | 🟢 [#320](https://github.com/ccsb-scripps/AutoDock-GPU/pull/320) |
| [bam](https://github.com/ZaidQureshi/bam) ([fork](https://github.com/AMD-Ecosystem/bam/tree/moat-port)) | 🚫 | 🚫 | 🚫 | ⏸ on hold |
| [barney](https://github.com/NVIDIA/barney) ([fork](https://github.com/AMD-Ecosystem/barney/tree/moat-port)) | ✅ | ✅ | ✅ | 🟣 [#46](https://github.com/NVIDIA/barney/pull/46) |
| [baspacho](https://github.com/facebookresearch/baspacho) ([fork](https://github.com/AMD-Ecosystem/baspacho/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#10](https://github.com/facebookresearch/baspacho/pull/10) |
| [bellhopcuda](https://github.com/A-New-BellHope/bellhopcuda) ([fork](https://github.com/AMD-Ecosystem/bellhopcuda/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#65](https://github.com/A-New-BellHope/bellhopcuda/pull/65) |
| [brian2cuda](https://github.com/brian-team/brian2cuda) ([fork](https://github.com/AMD-Ecosystem/brian2cuda/tree/moat-port)) | ✅ | ✅ | 🔄 | 🟢 [#327](https://github.com/brian-team/brian2cuda/pull/327) |
| [catboost](https://github.com/catboost/catboost) ([fork](https://github.com/AMD-Ecosystem/catboost-moat/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#3111](https://github.com/catboost/catboost/pull/3111) |
| [colmap](https://github.com/colmap/colmap) ([fork](https://github.com/AMD-Ecosystem/colmap/tree/moat-port)) | 🔧 | ⬜ | ⬜ | — |
| [CPM.cu](https://github.com/OpenBMB/CPM.cu) | 🚫 | — | — | ⚪ not-portable |
| [CubbyFlow](https://github.com/utilForever/CubbyFlow) ([fork](https://github.com/AMD-Ecosystem/CubbyFlow/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#145](https://github.com/utilForever/CubbyFlow/pull/145) |
| [cuBQL](https://github.com/NVIDIA/cuBQL) ([fork](https://github.com/AMD-Ecosystem/cuBQL/tree/moat-port)) | ✅ | ✅ | ✅ | 🟣 [#35](https://github.com/NVIDIA/cuBQL/pull/35) |
| [cubvh](https://github.com/ashawkey/cubvh) ([fork](https://github.com/AMD-Ecosystem/cubvh/tree/moat-port)) | ✅ | ✅ | ✅ | 🟣 [#33](https://github.com/ashawkey/cubvh/pull/33) |
| [cuda-efficient-features](https://github.com/fixstars/cuda-efficient-features) ([fork](https://github.com/AMD-Ecosystem/cuda-efficient-features/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#3](https://github.com/fixstars/cuda-efficient-features/pull/3) |
| [CUDA-L2](https://github.com/deepreinforce-ai/CUDA-L2) | 🚫 | — | — | ⚪ not-portable |
| [CUDA-ScanMatcher-ICP](https://github.com/botforge/CUDA-ScanMatcher-ICP) ([fork](https://github.com/AMD-Ecosystem/CUDA-ScanMatcher-ICP/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#10](https://github.com/botforge/CUDA-ScanMatcher-ICP/pull/10) |
| [cuda_voxelizer](https://github.com/Forceflow/cuda_voxelizer) | ⬜ | ⬜ | ⬜ | — |
| [cudaKDTree](https://github.com/ingowald/cudaKDTree) ([fork](https://github.com/AMD-Ecosystem/cudaKDTree/tree/moat-port)) | ✅ | ✅ | ✅ | 🟣 [#40](https://github.com/ingowald/cudaKDTree/pull/40) |
| [CudaSift](https://github.com/Celebrandil/CudaSift) ([fork](https://github.com/AMD-Ecosystem/CudaSift/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#97](https://github.com/Celebrandil/CudaSift/pull/97) |
| [CuMesh](https://github.com/JeffreyXiang/CuMesh) ([fork](https://github.com/AMD-Ecosystem/CuMesh/tree/moat-port)) | 🔄 | ✅ | ✅ | 🟢 [#36](https://github.com/JeffreyXiang/CuMesh/pull/36) |
| [cuPDLP-C](https://github.com/COPT-Public/cuPDLP-C) ([fork](https://github.com/AMD-Ecosystem/cuPDLP-C/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#41](https://github.com/COPT-Public/cuPDLP-C/pull/41) |
| [cuPDLPx](https://github.com/MIT-Lu-Lab/cuPDLPx) ([fork](https://github.com/AMD-Ecosystem/cuPDLPx/tree/moat-port)) | ✅ | ✅ | 🔄 | 🟣 [#94](https://github.com/MIT-Lu-Lab/cuPDLPx/pull/94) |
| [cupoch](https://github.com/neka-nat/cupoch) ([fork](https://github.com/AMD-Ecosystem/cupoch/tree/moat-port)) | ✅ | ✅ | ✅ | 🟣 [#143](https://github.com/neka-nat/cupoch/pull/143) |
| [CuRast](https://github.com/m-schuetz/CuRast) ([fork](https://github.com/AMD-Ecosystem/CuRast/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#2](https://github.com/m-schuetz/CuRast/pull/2) |
| [cuSZ](https://github.com/szcompressor/cuSZ) ([fork](https://github.com/AMD-Ecosystem/cuSZ/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#91](https://github.com/szcompressor/cuSZ/pull/91) |
| [CV-CUDA](https://github.com/CVCUDA/CV-CUDA) ([fork](https://github.com/AMD-Ecosystem/CV-CUDA/tree/moat-port)) | 🔄 | ✅ | 🚫 | 🟢 [#293](https://github.com/CVCUDA/CV-CUDA/pull/293) |
| [DEM-Engine](https://github.com/projectchrono/DEM-Engine) ([fork](https://github.com/AMD-Ecosystem/DEM-Engine/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#69](https://github.com/projectchrono/DEM-Engine/pull/69) |
| [dgSPARSE-Lib](https://github.com/dgSPARSE/dgSPARSE-Lib) ([fork](https://github.com/AMD-Ecosystem/dgSPARSE-Lib/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#39](https://github.com/dgSPARSE/dgSPARSE-Lib/pull/39) |
| [dietgpu](https://github.com/facebookresearch/dietgpu) ([fork](https://github.com/AMD-Ecosystem/dietgpu/tree/moat-port)) | ✅ | ✅ | 🔄 | — |
| [diff-surfel-rasterizations](https://github.com/xbillowy/diff-surfel-rasterizations) ([fork](https://github.com/AMD-Ecosystem/diff-surfel-rasterizations/tree/moat-port)) | ⬜ | ⬜ | ⬜ | — |
| [diff-surfel-tracing](https://github.com/xbillowy/diff-surfel-tracing) ([fork](https://github.com/AMD-Ecosystem/diff-surfel-tracing/tree/moat-port)) | ⬜ | ⬜ | ⬜ | — |
| [DiffPhysDrone](https://github.com/HenryHuYu/DiffPhysDrone) ([fork](https://github.com/AMD-Ecosystem/DiffPhysDrone/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#45](https://github.com/HenryHuYu/DiffPhysDrone/pull/45) |
| [DynOSAM](https://github.com/ACFR-RPG/DynOSAM) | ⬜ | ⬜ | ⬜ | — |
| [egg.c](https://github.com/d0rc/egg.c) ([fork](https://github.com/AMD-Ecosystem/egg.c/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#8](https://github.com/d0rc/egg.c/pull/8) |
| [ElasticFusion](https://github.com/mp3guy/ElasticFusion) ([fork](https://github.com/AMD-Ecosystem/ElasticFusion/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [EnvGS](https://github.com/zju3dv/EnvGS) ([fork](https://github.com/AMD-Ecosystem/EnvGS/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [espresso](https://github.com/espressomd/espresso) ([fork](https://github.com/AMD-Ecosystem/espresso/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [evogp](https://github.com/EMI-Group/evogp) ([fork](https://github.com/AMD-Ecosystem/evogp/tree/moat-port)) | ✅ | ✅ | ✅ | 🟣 [#12](https://github.com/EMI-Group/evogp/pull/12) |
| [faiss](https://github.com/facebookresearch/faiss) ([fork](https://github.com/AMD-Ecosystem/faiss/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [FaithC](https://github.com/Luo-Yihao/FaithC) ([fork](https://github.com/AMD-Ecosystem/FaithC/tree/moat-port)) | 🔄 | ✅ | ✅ | 🟢 [#12](https://github.com/Luo-Yihao/FaithC/pull/12) |
| [Fast-Poisson-Image-Editing](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing) ([fork](https://github.com/AMD-Ecosystem/Fast-Poisson-Image-Editing/tree/moat-port)) | ✅ | ✅ | ✅ | 🟣 [#25](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing/pull/25) |
| [faster-gaussian-splatting](https://github.com/nerficg-project/faster-gaussian-splatting) ([fork](https://github.com/AMD-Ecosystem/faster-gaussian-splatting/tree/moat-port)) | 🔄 | 🔄 | 🔄 | — |
| [FastGeodis](https://github.com/masadcv/FastGeodis) ([fork](https://github.com/AMD-Ecosystem/FastGeodis/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#68](https://github.com/masadcv/FastGeodis/pull/68) |
| [fdtd3d](https://github.com/zer011b/fdtd3d) ([fork](https://github.com/AMD-Ecosystem/fdtd3d/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [ffpa-attn](https://github.com/xlite-dev/ffpa-attn) ([fork](https://github.com/AMD-Ecosystem/ffpa-attn/tree/moat-port)) | 🔄 | ✅ | ✅ | 🟣 [#268](https://github.com/xlite-dev/ffpa-attn/pull/268) |
| [FLAMEGPU2](https://github.com/FLAMEGPU/FLAMEGPU2) ([fork](https://github.com/AMD-Ecosystem/FLAMEGPU2/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [FlashKDA](https://github.com/MoonshotAI/FlashKDA) | ⬜ | ⬜ | ⬜ | — |
| [FlashMoE](https://github.com/osayamenja/FlashMoE) | 🚫 | — | — | ⚪ not-portable |
| [FlashRT](https://github.com/flashrt-project/FlashRT) | 🚫 | — | — | ⚪ not-portable |
| [foldmason](https://github.com/steineggerlab/foldmason) ([fork](https://github.com/AMD-Ecosystem/foldmason/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [foldseek](https://github.com/steineggerlab/foldseek) ([fork](https://github.com/AMD-Ecosystem/foldseek/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [gaussian_splatting](https://github.com/joeyan/gaussian_splatting) ([fork](https://github.com/AMD-Ecosystem/gaussian_splatting/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#37](https://github.com/joeyan/gaussian_splatting/pull/37) |
| [gdtk](https://github.com/gdtk-uq/gdtk) ([fork](https://github.com/AMD-Ecosystem/gdtk/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [GOMC](https://github.com/GOMC-WSU/GOMC) ([fork](https://github.com/AMD-Ecosystem/GOMC/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [GooFit](https://github.com/GooFit/GooFit) ([fork](https://github.com/AMD-Ecosystem/GooFit/tree/moat-port)) | 🔧 | ⬜ | ⬜ | ⏸ on hold |
| [gpu4pyscf](https://github.com/pyscf/gpu4pyscf) ([fork](https://github.com/AMD-Ecosystem/gpu4pyscf/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [GPU_IPC](https://github.com/KemengHuang/GPU_IPC) ([fork](https://github.com/AMD-Ecosystem/GPU_IPC/tree/moat-port)) | 🔧 | 🔧 | ⬜ | — |
| [Gpufit](https://github.com/gpufit/Gpufit) ([fork](https://github.com/AMD-Ecosystem/Gpufit/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#149](https://github.com/gpufit/Gpufit/pull/149) |
| [GPUMD](https://github.com/brucefan1983/GPUMD) ([fork](https://github.com/AMD-Ecosystem/GPUMD/tree/moat-port)) | ✅ | ✅ | ✅ | 🟣 [#1538](https://github.com/brucefan1983/GPUMD/pull/1538) |
| [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR) ([fork](https://github.com/AMD-Ecosystem/gpuRIR/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [gRASPA](https://github.com/snurr-group/gRASPA) ([fork](https://github.com/AMD-Ecosystem/gRASPA/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [gtsam_points](https://github.com/koide3/gtsam_points) ([fork](https://github.com/AMD-Ecosystem/gtsam_points/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#99](https://github.com/koide3/gtsam_points/pull/99) |
| [h2o4gpu](https://github.com/h2oai/h2o4gpu) | ⬜ | ⬜ | ⬜ | — |
| [HEonGPU](https://github.com/Alisah-Ozcan/HEonGPU) ([fork](https://github.com/AMD-Ecosystem/HEonGPU/tree/moat-port)) | 🔧 | 🚫 | ⬜ | — |
| [icicle](https://github.com/ingonyama-zk/icicle) ([fork](https://github.com/AMD-Ecosystem/icicle/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [k2](https://github.com/k2-fsa/k2) ([fork](https://github.com/AMD-Ecosystem/k2/tree/moat-port)) | 🔄 | ✅ | ✅ | 🟣 [#1353](https://github.com/k2-fsa/k2/pull/1353) |
| [kaldi](https://github.com/kaldi-asr/kaldi) ([fork](https://github.com/AMD-Ecosystem/kaldi/tree/moat-port)) | ✅ | ✅ | 🚫 | 🟢 [#4986](https://github.com/kaldi-asr/kaldi/pull/4986) |
| [kaldifeat](https://github.com/csukuangfj/kaldifeat) ([fork](https://github.com/AMD-Ecosystem/kaldifeat/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [KittenGpuLBVH](https://github.com/jerry060599/KittenGpuLBVH) ([fork](https://github.com/AMD-Ecosystem/KittenGpuLBVH/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#5](https://github.com/jerry060599/KittenGpuLBVH/pull/5) |
| [LC-framework](https://github.com/burtscher/LC-framework) ([fork](https://github.com/AMD-Ecosystem/LC-framework/tree/moat-port)) | ✅ | ✅ | ⬜ | — |
| [lc0](https://github.com/LeelaChessZero/lc0) ([fork](https://github.com/AMD-Ecosystem/lc0/tree/moat-port)) | 🔄 | ✅ | 🚫 | 🟢 [#2420](https://github.com/LeelaChessZero/lc0/pull/2420) |
| [LEAP](https://github.com/llnl/LEAP) ([fork](https://github.com/AMD-Ecosystem/LEAP/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [libSGM](https://github.com/fixstars/libSGM) ([fork](https://github.com/AMD-Ecosystem/libSGM/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#89](https://github.com/fixstars/libSGM/pull/89) |
| [LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio) ([fork](https://github.com/AMD-Ecosystem/LichtFeld-Studio/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [LiteGS](https://github.com/MooreThreads/LiteGS) ([fork](https://github.com/AMD-Ecosystem/LiteGS/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [llm-awq](https://github.com/mit-han-lab/llm-awq) ([fork](https://github.com/AMD-Ecosystem/llm-awq/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [llm.c](https://github.com/karpathy/llm.c) ([fork](https://github.com/AMD-Ecosystem/llm.c/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#854](https://github.com/karpathy/llm.c/pull/854) |
| [llmq](https://github.com/IST-DASLab/llmq) | 🚫 | — | — | ⚪ not-portable |
| [mahout](https://github.com/apache/mahout) ([fork](https://github.com/AMD-Ecosystem/mahout/tree/moat-port)) | ✅ | ✅ | 🔄 | 🟢 [#1399](https://github.com/apache/mahout/pull/1399) |
| [marian-dev](https://github.com/marian-nmt/marian-dev) ([fork](https://github.com/AMD-Ecosystem/marian-dev/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM) ([fork](https://github.com/AMD-Ecosystem/MASt3R-SLAM/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [mcx](https://github.com/fangq/mcx) ([fork](https://github.com/AMD-Ecosystem/mcx/tree/moat-port)) | 🔄 | ✅ | ✅ | 🟣 [#264](https://github.com/fangq/mcx/pull/264) |
| [metaeuk](https://github.com/soedinglab/metaeuk) ([fork](https://github.com/AMD-Ecosystem/metaeuk/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [mHC.cu](https://github.com/AndreSlavescu/mHC.cu) ([fork](https://github.com/AMD-Ecosystem/mHC.cu/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [MMseqs2](https://github.com/soedinglab/MMseqs2) ([fork](https://github.com/AMD-Ecosystem/MMseqs2/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [MPPI-Generic](https://github.com/ACDSLab/MPPI-Generic) ([fork](https://github.com/AMD-Ecosystem/MPPI-Generic/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [mumax3](https://github.com/mumax/3) ([fork](https://github.com/AMD-Ecosystem/mumax3/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#400](https://github.com/mumax/3/pull/400) |
| [ntransformer](https://github.com/xaskasdf/ntransformer) ([fork](https://github.com/AMD-Ecosystem/ntransformer/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [OCTproZ](https://github.com/spectralcode/OCTproZ) ([fork](https://github.com/AMD-Ecosystem/OCTproZ/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [ohm](https://github.com/csiro-robotics/ohm) ([fork](https://github.com/AMD-Ecosystem/ohm/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [oneflow](https://github.com/Oneflow-Inc/oneflow) ([fork](https://github.com/AMD-Ecosystem/oneflow/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [op43dgs](https://github.com/LetianHuang/op43dgs) ([fork](https://github.com/AMD-Ecosystem/op43dgs/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [Open3D](https://github.com/isl-org/Open3D) ([fork](https://github.com/AMD-Ecosystem/Open3D/tree/moat-port)) | ✅ | ✅ | 🔄 | 🟢 [#7509](https://github.com/isl-org/Open3D/pull/7509) |
| [opencv](https://github.com/opencv/opencv) ([fork](https://github.com/AMD-Ecosystem/opencv/tree/moat-port)) | ⬜ | ⬜ | ⬜ | — |
| [opencv_contrib](https://github.com/opencv/opencv_contrib) ([fork](https://github.com/AMD-Ecosystem/opencv_contrib/tree/moat-port)) | 🔄 | ✅ | ✅ | 🟢 [#4147](https://github.com/opencv/opencv_contrib/pull/4147) |
| [PhoenixOS](https://github.com/SJTU-IPADS/PhoenixOS) | 🚫 | ⬜ | ⬜ | ⏸ on hold |
| [plumed2](https://github.com/plumed/plumed2) ([fork](https://github.com/AMD-Ecosystem/plumed2/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [plvs](https://github.com/luigifreda/plvs) ([fork](https://github.com/AMD-Ecosystem/plvs/tree/moat-port)) | ✅ | ✅ | 🚫 | — |
| [Pointcept](https://github.com/Pointcept/Pointcept) ([fork](https://github.com/AMD-Ecosystem/Pointcept/tree/moat-port)) | ✅ | ✅ | ✅ | 🟣 [#604](https://github.com/Pointcept/Pointcept/pull/604) |
| [popsift](https://github.com/alicevision/popsift) ([fork](https://github.com/AMD-Ecosystem/popsift/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#186](https://github.com/alicevision/popsift/pull/186) |
| [prismatic](https://github.com/prism-em/prismatic) ([fork](https://github.com/AMD-Ecosystem/prismatic/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [pytorch3d](https://github.com/facebookresearch/pytorch3d) | ✅ | ✅ | ✅ | 🟣 [#2039](https://github.com/facebookresearch/pytorch3d/pull/2039) |
| [qrack](https://github.com/unitaryfoundation/qrack) ([fork](https://github.com/AMD-Ecosystem/qrack/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [Quest](https://github.com/mit-han-lab/Quest) ([fork](https://github.com/AMD-Ecosystem/Quest/tree/moat-port)) | 🔧 | 🔧 | ⬜ | — |
| [QUICK](https://github.com/merzlab/QUICK) ([fork](https://github.com/AMD-Ecosystem/QUICK/tree/moat-port)) | 🔄 | ✅ | 🚫 | — |
| [rmagine](https://github.com/uos/rmagine) ([fork](https://github.com/AMD-Ecosystem/rmagine/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [rmcl](https://github.com/uos/rmcl) ([fork](https://github.com/AMD-Ecosystem/rmcl/tree/moat-port)) | ⬜ | ⬜ | ⬜ | — |
| [RWKV-CUDA](https://github.com/BlinkDL/RWKV-CUDA) ([fork](https://github.com/AMD-Ecosystem/RWKV-CUDA/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [RXMesh](https://github.com/owensgroup/RXMesh) ([fork](https://github.com/AMD-Ecosystem/RXMesh/tree/moat-port)) | 🔄 | ✅ | ✅ | 🟢 [#73](https://github.com/owensgroup/RXMesh/pull/73) |
| [SCAMP](https://github.com/zpzim/SCAMP) ([fork](https://github.com/AMD-Ecosystem/SCAMP/tree/moat-port)) | 🔄 | ✅ | ✅ | 🟢 [#145](https://github.com/zpzim/SCAMP/pull/145) |
| [SpargeAttn](https://github.com/thu-ml/SpargeAttn) | 🚫 | — | — | ⚪ not-portable |
| [spconv](https://github.com/traveller59/spconv) | ⬜ | ⬜ | ⬜ | — |
| [splatad](https://github.com/carlinds/splatad) ([fork](https://github.com/AMD-Ecosystem/splatad/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#24](https://github.com/carlinds/splatad/pull/24) |
| [sppark](https://github.com/supranational/sppark) ([fork](https://github.com/AMD-Ecosystem/sppark/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#82](https://github.com/supranational/sppark/pull/82) |
| [stdgpu](https://github.com/stotko/stdgpu) ([fork](https://github.com/AMD-Ecosystem/stdgpu/tree/moat-port)) | 🔄 | ✅ | ✅ | 🟣 [#484](https://github.com/stotko/stdgpu/pull/484) |
| [symforce](https://github.com/symforce-org/symforce) ([fork](https://github.com/AMD-Ecosystem/symforce/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#465](https://github.com/symforce-org/symforce/pull/465) |
| [TIGRE](https://github.com/CERN/TIGRE) ([fork](https://github.com/AMD-Ecosystem/TIGRE/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#747](https://github.com/CERN/TIGRE/pull/747) |
| [tiny-vllm](https://github.com/jmaczan/tiny-vllm) ([fork](https://github.com/AMD-Ecosystem/tiny-vllm/tree/moat-port)) | 🔄 | ✅ | ✅ | 🟣 [#2](https://github.com/jmaczan/tiny-vllm/pull/2) |
| [torch-linear-assignment](https://github.com/ivan-chai/torch-linear-assignment) ([fork](https://github.com/AMD-Ecosystem/torch-linear-assignment/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#31](https://github.com/ivan-chai/torch-linear-assignment/pull/31) |
| [TornadoVM](https://github.com/beehive-lab/TornadoVM) | ⬜ | ⬜ | ⬜ | — |
| [tsne-cuda](https://github.com/CannyLab/tsne-cuda) | ⬜ | ⬜ | ⬜ | — |
| [TTT3R](https://github.com/Inception3D/TTT3R) ([fork](https://github.com/AMD-Ecosystem/TTT3R/tree/moat-port)) | ✅ | ✅ | ✅ | — |
| [TurboFNO](https://github.com/shixun404/TurboFNO) ([fork](https://github.com/AMD-Ecosystem/TurboFNO/tree/moat-port)) | ✅ | ✅ | ✅ | 🟣 [#3](https://github.com/shixun404/TurboFNO/pull/3) |
| [unified-cache-management](https://github.com/ModelEngine-Group/unified-cache-management) ([fork](https://github.com/AMD-Ecosystem/unified-cache-management/tree/moat-port)) | ✅ | ✅ | 🔄 | 🟢 [#1021](https://github.com/ModelEngine-Group/unified-cache-management/pull/1021) |
| [Velvet](https://github.com/vitalight/Velvet) ([fork](https://github.com/AMD-Ecosystem/Velvet/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#9](https://github.com/vitalight/Velvet/pull/9) |
| [visionaray](https://github.com/szellmann/visionaray) ([fork](https://github.com/AMD-Ecosystem/visionaray/tree/moat-port)) | 🔄 | ✅ | ✅ | 🟣 [#51](https://github.com/szellmann/visionaray/pull/51) |
| [yalm](https://github.com/andrewkchan/yalm) ([fork](https://github.com/AMD-Ecosystem/yalm/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#12](https://github.com/andrewkchan/yalm/pull/12) |
| [YarnBall](https://github.com/jerry060599/YarnBall) ([fork](https://github.com/AMD-Ecosystem/YarnBall/tree/moat-port)) | ✅ | ✅ | ✅ | 🟢 [#5](https://github.com/jerry060599/YarnBall/pull/5) |
| [ZhiLight](https://github.com/zhihu/ZhiLight) ([fork](https://github.com/AMD-Ecosystem/ZhiLight/tree/moat-port)) | ✅ | ✅ | 🎫 | 🟢 [#81](https://github.com/zhihu/ZhiLight/pull/81) |
<!-- MOAT:TABLE:END -->

## Layout

See `projects/README.md` for the per-project files, the `cuda-to-rocm` skill for porting strategy and fault classes, and `CLAUDE.md` for how a Claude CLI drives the pipeline.
