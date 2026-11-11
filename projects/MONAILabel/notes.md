# MONAILabel notes

## Intake (2026-11-11)

### Fork

The AMD fork already exists: AMD-Ecosystem/MONAILabel (GitHub fork of Project-MONAI/MONAILabel), default branch `amd-develop`. Jeff Daily asked the AMD-Ecosystem admins for write access on 2026-11-11; until it is granted nobody in MOAT can push `moat-port` there, so the stage is left at intake for Jeff (no `awaiting-fork`, no `release-forks`). Whether MOAT builds on `amd-develop` or ports independently from upstream `main` is Jeff's ruling; the assessment below is the input. `fork_default_branch` was scaffolded as upstream's `main`.

### Licence

- Apache-2.0 (LICENSE), tier 1, `utils/licenses.py check Project-MONAI/MONAILabel`.
- `scan-nvidia` on a shallow upstream clone (agent_space/MONAILabel-screen): no NVIDIA proprietary licence text. No submodules. The OHIF v3 plugin under `plugins/ohifv3` pulls OHIF (MIT) at build time.

### Duplicate effort

- AMD-Ecosystem/MONAILabel `amd-develop` is the authoritative AMD port (AMD staff; ships an `amd-monailabel` wheel and a ROCm docs site). ROCm-LS has only archived MONAI-family repos (ROCm-LS/monai, MONAIwip, model-zoo).
- Upstream PR Project-MONAI/MONAILabel#1925 (OPEN, opened 2026-10-15 by nilapate from `AMD-Ecosystem:fix/issue-1924`, +162/-15, 4 files): "Fix /gpu endpoint and gpu_memory_map() to support AMD ROCm and degrade gracefully", fixes #1924. Review decision CHANGES_REQUESTED by maintainer ericspod ("minor changes, otherwise good"); ericspod then asked the other maintainers to look ("good to go", 2026-10-22) and noted he cannot push to the PR because the AMD-Ecosystem fork does not allow maintainer edits. mergeStateStatus BLOCKED. This is the only piece of the AMD work offered upstream, and MOAT should not compete with it: if MOAT touches the same code, it should build on #1925.
- MOAT `MONAI` project (fork AMD-Ecosystem/MONAI, stage planning, torch-extension) is related: see Dependencies.

### GPU surface (upstream main c6c7b46, 2026-10-22)

- No CUDA source (`.cu`/`.cuh`/`.cpp`) at all. Pure Python on PyTorch + MONAI. This is a `lane: validate`-shaped project: build on ROCm torch, fix GPU-detection and packaging, run the GPU tests, document ROCm support.
- NVIDIA-specific paths: `monailabel/utils/others/generic.py` `gpu_memory_map()` and `monailabel/endpoints/logs.py` `/gpu` shell out to `nvidia-smi` (the #1925 fix); `monailabel/scribbles/utils.py` `learn_and_apply_gmm_monai()` requires `torch.utils.cpp_extension.CUDA_HOME` and MONAI's JIT-compiled `GaussianMixtureModel` CUDA extension. `CUDA_HOME` is normally unset on a ROCm torch, so that scribbles path raises `OSError` on ROCm today, and the AMD fork did not touch it. `monai.networks.blocks.CRF` (scribbles) also uses MONAI's permutohedral-lattice extension when available.
- Optional deps with GPU/native code: SAM-2 (has an optional CUDA connected-components extension), numpymaxflow (native, not GPU), itk, openslide.
- Tests: `runtests.sh --unittests` (pytest under `tests/unit`), plus integration tests that start the server. Upstream CI `pythonapp.yml` runs on `windows-latest` and `ubuntu-latest`.
- Upstream officially supports Ubuntu and Windows ("MONAI Label supports both Ubuntu and Windows OS with GPU/CUDA enabled", docs/source/installation.rst) and ships `monailabel.bat`. Active, not archived.

### Dependencies

- Soft dependency on MOAT `MONAI`: MONAILabel needs the `monai` package, which runs on ROCm in pure-Python mode, but the scribbles GMM/CRF paths need MONAI's C++/CUDA extensions, which are in the MONAI port's surface. Not recorded with `set-deps` because the core server, infer and train paths work without them; validate those paths against MOAT's MONAI when it lands.

### Viability

Yes. The AMD fork already runs it on Instinct; upstream already supports Windows; nothing needs a kernel port. The work is GPU detection on Radeon/Windows, the scribbles `CUDA_HOME` gate, and validation.

## AMD fork assessment

Compared `AMD-Ecosystem/MONAILabel@amd-develop` (0370aa6, 2026-09-30) with `Project-MONAI/MONAILabel@main` (c6c7b46, 2026-10-22):

- **Ahead/behind:** 56 commits ahead (12 merges), 6 behind. Merge base 6ed8f8c, 2026-02-04. The fork is close to upstream because upstream is quiet.
- **Size:** 50 files, +2324 / -308 (21 added, 28 modified, 1 renamed). Port code proper is small: the core commit d7cd49a "[ROCm] Add AMD ROCm GPU support to MONAILabel" touched 3 files (+52/-15): `generic.py` `gpu_memory_map()` falling back to `rocm-smi --showmeminfo vram --csv`, `/gpu` endpoint dispatch, Dockerfile ROCm base. That is essentially what became upstream #1925.
- **Everything else:** ROCm docs site (`docs/sphinx`, `docs/install`, RTD config), README rewrite with Instinct quick-start, unit tests added to lift coverage 77% -> 80%, a validation report (`validation/`, MI350X gfx950), OHIF v3 plugin fixes for a pinned OHIF commit (`plugins/ohifv3/...`, workbox build), Dockerfile rebuilt on `amd-monai`, and fixes for unit-test failures in their environment.
- **Noise and upstream-incompatible changes:** package rename `monailabel` -> `amd-monailabel` (setup.cfg, wheel globs); `requirements.txt` drops `monai[...]` in favour of a separately installed `amd-monai` and moves test-only deps into runtime requirements; a key rename `image_path` -> `img_path` across `sam2/infer.py`, `tasks/infer/basic_infer.py`, `transform/cache.py`, `transform/writer.py`, `datastore/utils/convert.py` and sample apps (looks tied to `amd-monai`; it would break stock MONAI users); "remove CUDA/Nvidia ref" commit; AMD modification-notice headers on upstream files; CI `pre-build` job removed.
- **GPUs claimed:** Instinct only: README "validated on AMD Instinct MI300X, MI325X, and MI355X", compatibility table ROCm 10.0.0 / Ubuntu 24.04 / Python 3.12 / MI300X, MI325X, MI355X; `AMDGPU_TARGETS=gfx942` or `gfx950` in the install steps.
- **Wave64 assumptions:** none possible in code (no kernels). Arch appears only as gfx942/gfx950 install instructions and wheel device extras.
- **Linux-only pieces:** the ROCm path relies on `rocm-smi` (Linux; not present on Windows and absent from pip-only ROCm installs unless the SMI package is added; being superseded by `amd-smi`), Docker/Ubuntu-only install docs. Upstream's own Windows support is untouched.
- **Offered upstream:** only the GPU-detection fix, as #1925 (open, blocked on maintainer edit access and the requested changes). The rest is downstream product work.

### Radeon (wave32) and Windows risks

- GPU detection: `rocm-smi` card numbering can differ from HIP device order on hosts with an iGPU plus a Radeon dGPU, and `rocm-smi` is missing on Windows, where `gpu_memory_map()` falls back to the hard-coded `{0: 4300}` default. A backend-agnostic query (`torch.cuda.mem_get_info`, or `amd-smi`) would serve both; any change here should go through #1925, not a parallel PR.
- Scribbles GMM path: the `CUDA_HOME` gate and MONAI's JIT GMM extension block it on ROCm (Linux and Windows) until MOAT's MONAI port covers the extension.
- Windows ROCm torch (TheRock) is required; SAM-2 and numpymaxflow wheels on Windows are unverified.
- Large models (SAM-2, segmentation bundles) on a 16 GB Radeon or the gfx1151 APU: watch memory, and do not run large probes on the laptop.

### Recommendation

`fork` (record via `set-intake`). Low-risk validate-lane port: no kernels, upstream already supports Windows, and an AMD PR (#1925) is already in front of the maintainers. MOAT's value is wave32 and Windows validation plus the ROCm gaps the AMD fork left (Windows GPU detection, scribbles `CUDA_HOME`).

## Rulings (2026-11-11, Jeff Daily)

- Base: independent port from the tip of upstream Project-MONAI/MONAILabel `main` (record the exact commit when porting starts). AMD-Ecosystem/MONAILabel `amd-develop` and upstream PR #1925 are reference material only. Keep the port minimal: GPU discovery through torch instead of depending on `rocm-smi`; make the scribbles GMM path work when `CUDA_HOME` is unset on ROCm torch; correct device ordering on hosts with an integrated GPU plus a Radeon card. Leave out AMD's `amd-monailabel` rename, the `image_path` -> `img_path` key rename, `amd-monai` requirements and product docs.
- Windows: in scope (upstream already supports Windows); no waiver.
- Dependency: `depends_on: MONAI` (the GMM extension comes from MONAI; MOAT's independent MONAI port goes first).
- Delivery: Jeff's call later. AMD's #1925 is open upstream for the same code; nothing is posted upstream or to the AMD team until Jeff decides.
- The fork is AMD-Ecosystem/MONAILabel; `moat-port` cannot be pushed until Jeff's write access arrives (requested 2026-11-11).
