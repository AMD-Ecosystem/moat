# Clair3 notes

## Intake (2026-11-11)

### Fork

The AMD fork already exists: AMD-Ecosystem/Clair3 (GitHub fork of HKU-BAL/Clair3), default branch `main`, with AMD work also on `amd-develop`, `rocm-enablement` and `fix/pypy-venv-pip-shadowing`. Jeff Daily asked the AMD-Ecosystem admins for write access on 2026-11-11; until it is granted nobody in MOAT can push `moat-port` there, so the stage is left at intake for Jeff (no `awaiting-fork`, no `release-forks`). Whether MOAT builds on the AMD fork or works independently from upstream `main` is Jeff's ruling; the assessment below is the input.

### Licence

- GitHub reports no licence (`licenses.py check` -> UNPARSED/NONE). Read `LICENSE.md` directly: it is the BSD 3-Clause licence text ("Copyright 2021 The University of Hong Kong, Department of Computer Science"; the three standard conditions and disclaimer). Recorded as `BSD-3-Clause`.
- `scan-nvidia` on a shallow upstream clone (agent_space/Clair3-screen): no NVIDIA proprietary licence text. No submodules.
- Vendored C in `src/`: `csv.c` (MIT, Jan Doczy), `levenshtein.c` (MIT, Titus Wormer), `khash.h`/`kvec.h` (attractivechaos, MIT), `preprocess/realign/ssw.c` (MIT, Boston College). `src/medaka_bamiter.*` and `preprocess/medaka_utils.py` carry no header but are derived from Oxford Nanopore's medaka, which is MPL-2.0 upstream. None of this is GPU code and a ROCm change would not touch it; noted so nobody is surprised later. Build-time downloads: samtools/htslib (MIT/BSD) and longphase (GPL-3.0, a separate executable).

### Duplicate effort: ROCm support is already upstream

- HKU-BAL/Clair3 PR #480 "Support AMD Instinct GPUs: fix GPU detection and document ROCm installation" (prateek9623, from `AMD-Ecosystem:amd-develop`, +319/-13, 4 files) MERGED 2026-10-26. Released in v2.0.5 (2026-11-06): `Dockerfile.amd` (ROCm PyTorch from AMD's pip index, default `GFX_ARCH=gfx950`, gfx942 documented) and a `rocm-smi` fallback in `clair3/CallVariantsFromCffiGPU.py`. README: "Native GPU support on Linux (NVIDIA CUDA and AMD Instinct ROCm) and Apple Silicon".
- PR #486 (AMD-Ecosystem `fix/pypy-venv-pip-shadowing`) merged 2026-11-06. The AMD team upstreams here and the maintainers accept it.
- So Linux/Instinct ROCm support is mature, upstream, and AMD's own. Per `assess-existing-support.md` that is normally a skip (`already-supported`); what is left is the coverage it does not claim.

### GPU surface (upstream main 78caea9, 2026-11-06)

- No CUDA source. Clair3 has no GPU code of its own: the CFFI layer (`libclair3.so`) is plain C linking htslib, and every GPU touchpoint is the `torch.cuda.*` Python API (models converted from TensorFlow, `convert_tf_checkpoint_to_torch.py`). `lane: validate` shape.
- Only GPU-specific code: GPU discovery and VRAM sizing in `clair3/CallVariantsFromCffiGPU.py` (nvidia-smi, then rocm-smi), `--device=cuda:N`, `CUDA_VISIBLE_DEVICES` worker pinning.
- Tests: a single unit test file (`tests/test_likelihood_math.py`); real validation is an end-to-end call on the quick-demo data (`run_clair3.sh` on the ONT/PacBio/Illumina demos with `--use_gpu`) compared with the CPU path or the published VCF.
- Platforms upstream: Linux (x86_64, aarch64) and macOS/Apple Silicon. Not Windows: the pipeline is bash (`run_clair3.sh`, `scripts/clair3*.sh`), GNU parallel, pypy, samtools/htslib, whatshap/longphase, all Linux/macOS tooling. Active, not archived.

### Viability

Yes on Radeon Linux (ROCm torch + rocm-smi already work for Instinct; Radeon needs a run and a docs line). Windows is not a Clair3 platform at all, so a Windows gate is a waiver candidate, not a port.

## AMD fork assessment

Compared `AMD-Ecosystem/Clair3@main` (e297294, 2026-09-17) with `HKU-BAL/Clair3@main` (78caea9, 2026-11-06):

- **Ahead/behind:** 2 commits ahead (b68f6f6 "[ROCm] AMD Instinct (ROCm) enablement for Clair3" plus its merge), 28 behind. Merge base 789067a, 2026-09-15.
- **Size and noise:** 34 files, +433 / -553. Most of that is not ROCm work: b68f6f6 was evidently authored against an older tree and committed onto 789067a, so it silently reverts newer upstream changes (`shared/version.py` 2.0.3 -> 2.0.2, `run_clair3.py` -45, README -464/+..., `MergeVcf.py`, `SortVcf.py`, the C sources). The fork's `main` is a stale, superseded branch and should not be built on.
- **The real AMD work** is `amd-develop` (678f964, "Support AMD Instinct GPUs through ROCm PyTorch", 2026-10-21): it is an ancestor of upstream `main`, i.e. fully merged as #480. `fix/pypy-venv-pip-shadowing` is merged as #486. `rocm-enablement` is the stale b68f6f6.
- **GPUs claimed:** Instinct only: `Dockerfile.amd` header "AMD Instinct / ROCm", `GFX_ARCH` default gfx950 (MI350X/MI355X), gfx942 (MI300X) documented; docs `gpu_quick_start.md` Options 4 and 5 are "AMD Instinct (ROCm)".
- **Wave64 assumptions:** none possible (no kernels); arch appears only in the wheel device extras.
- **Linux-only pieces:** `rocm-smi` (Linux; absent on Windows; superseded by `amd-smi`; its presence in the pip-only ROCm install that `Dockerfile.amd` uses is unverified), Docker device passthrough (`/dev/kfd`, `/dev/dri`), and the whole Clair3 pipeline.
- **Offered upstream:** yes, and merged (#480, #486).

### Radeon (wave32) and Windows risks

- `rocm-smi` card numbering versus HIP device order on hosts with an iGPU plus a Radeon dGPU: Clair3 pins workers by physical id into `CUDA_VISIBLE_DEVICES`, so a mismatch sends work to the wrong device or the iGPU. Worth checking on a Radeon host.
- Docs and `Dockerfile.amd` name only Instinct targets; a Radeon user needs `device-gfx1100`/`gfx1201` extras and a docs line. That docs change is the likely upstream deliverable.
- Model numerics on RDNA: compare GPU VCF with the CPU VCF on the demo data, not only exit status.
- Windows: not an upstream platform even on CPU; waiver candidate (an agent may only suggest it).

### Recommendation

`fork` (record via `set-intake`), validate-lane and small: ROCm is already upstream and AMD's own, so the only work is a Radeon (wave32) end-to-end validation, the device-index check, and a docs/Dockerfile line naming Radeon targets, sent upstream the way #480 was. If Jeff judges upstream's Instinct support sufficient, the alternative is a decline with reason `already-supported`. Windows should be a waiver.
