# tsne-cuda notes

## Intake (2026-08-07)

**What it is:** `CannyLab/tsne-cuda`, a CUDA implementation of the FIt-SNE algorithm with a Python (sklearn-compatible) binding. Active upstream: not archived, last push 2026-07-22, 1.9k stars, 18 open issues.

### Licence

`utils/licenses.py check CannyLab/tsne-cuda` reports `BSD-3-Clause`, tier 1 (GitHub parsed it cleanly, no NOASSERTION). Confirmed by reading the files directly rather than trusting the field alone: top-level `LICENSE`, `packaging/conda/LICENSE.txt`, and `src/python/LICENSE.txt` are all identical BSD-3-Clause text (Regents of the University of California, 2018). One git submodule, `third_party/cxxopts` (pinned at `c713b44`), carries its own MIT licence. No unlicensed vendored code found. `utils/licenses.py`'s NVIDIA-proprietary text-marker scan (`scan_nvidia`, matches licence TEXT not copyright lines) returned zero hits across the tree -- CUDA-specific code is present but not under an NVIDIA proprietary licence.

Recorded: `status.json.license_spdx = BSD-3-Clause` (tier 1, cleared to contribute).

### Duplicate effort

- No `tsne-cuda` (or fork of `CannyLab/tsne-cuda`) in AMD-Ecosystem or ROCm orgs (`gh search repos`/`gh repo list` against both, empty).
- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/ INSTALL.md` in the upstream tree: no hits (false-positive filtered for "chip"/"ship"/"whip"/"amdahl"). No "notable forks" section.
- RAPIDS cuML's `TSNE` implementation is explicitly derived from this project ("The CUDA implementation in RAPIDS cuML is derived from the excellent CannyLabs open source implementation") -- but that is a *downstream consumer*, not an AMD port, and there is no evidence RAPIDS cuML itself has ROCm support to point to instead. Not a duplicate-effort finding, just worth knowing tsne-cuda is the origin of a well-known derivative.

No existing or parked AMD effort found for this project.

### Viability

Genuine CUDA project: 18 `.cu` files (`src/kernels/*.cu` -- `apply_forces`, `attr_forces`, `rep_forces`, `nbodyfft`, `perplexity_search` -- plus `src/util/*.cu`, `src/fit_tsne.cu`, `src/ext/pymodule_ext.cu` for the Python extension). CMakeLists links `CUDA::cublas`, `CUDA::cufft`, `CUDA::cufftw`, `CUDA::cusparse`, uses Thrust (`src/util/thrust_utils.cu`), and hard-requires GPU FAISS (`find_package(FAISS REQUIRED)`, linked via `${FAISS_LIBRARIES}`) -- FAISS is not optional, the build fails without it. cuBLAS/cuFFT/cuSPARSE/Thrust all have direct hip*/roc* substitutes covered by the standard hipify path. The interactive ZMQ visualization path is already compiled out upstream (`-DNO_ZMQ`), so no extra third-party dependency there.

**MOAT dependency:** `faiss` is an existing MOAT project (`AMD-Ecosystem/faiss`) already `completed` on `linux-gfx90a`. This is a real, satisfiable dependency, not a blocker -- recorded with `set-deps tsne-cuda faiss`. A planner should build tsne-cuda against the validated FAISS ROCm port per its notes.md "Install as a dependency" section.

Not archived; a PR upstream has a real destination.

### Recommendation

Fork: the licence is clear (tier 1, no ambiguity, no NVIDIA-proprietary text, submodule licence checked), no duplicate AMD effort exists, the CUDA surface is real and standard (cuBLAS/cuFFT/cuSPARSE/Thrust + GPU FAISS, all with known hip*/roc* substitutes), and its one MOAT dependency (`faiss`) is already validated and completed. Recommending the planner take this up.
