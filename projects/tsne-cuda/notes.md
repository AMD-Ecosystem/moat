# tsne-cuda notes

## Intake 2026-08-06 (agent: intake, arch: linux-gfx90a)

Candidate: CannyLab/tsne-cuda, "GPU Accelerated t-SNE for CUDA with Python bindings", 1939 stars, MOAT candidate priority 5.775. Verdict: take it up. State set to `awaiting-fork`.

### Licence: BSD-3-Clause (tier 1, cleared to contribute)

`utils/licenses.py check` reports BSD-3-Clause / tier 1, and the repo's own LICENSE text confirms it: "BSD 3-Clause License, Copyright (c) 2018, Regents of the University of California". The same text is duplicated verbatim at `packaging/conda/LICENSE.txt` and `src/python/LICENSE.txt`, so the Python wheel and conda package carry the identical grant. Nothing mixed or per-part.

Recursed into vendored code: the only submodule is `third_party/cxxopts` (MIT, Copyright (c) 2014 Jarryd Beck). It is used only by the standalone C++ CLI (`src/exe/main.cu`), which upstream deliberately no longer builds, so it does not enter the shipped library at all. MIT is tier 1 too, so this does not complicate anything.

`licenses.scan_nvidia` over the full checkout (including the submodule) returns no matches: no file carries NVIDIA proprietary licence text.

Recorded in `upstream.json`: `license_spdx=BSD-3-Clause`, `approval_scope=contribute-only`. `moatlib.py license-gate tsne-cuda` -> license-ok=True.

### Duplicate effort: none found

- No `tsne`/`t-SNE` repo in AMD-Ecosystem, none in ROCm.
- `gh search repos tsne-cuda` returns only CannyLab/tsne-cuda plus unrelated academic CUDA t-SNE toys and PyTorch reimplementations. No AMD or ROCm fork of this upstream exists anywhere visible.
- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README.md INSTALL.md docs/` in upstream: zero substantive hits (only Doxygen boilerplate words containing "ship"). Upstream has no "notable forks" section and no AMD port to point at.

Adjacent-but-not-duplicate: RAPIDS cuML ships its own `cuml.TSNE` (both Barnes-Hut and an FIt-SNE/FFT-interpolation path derived from the same literature as this repo). cuML is dispositioned in MOAT as `ported-elsewhere` ("RAPIDS: the ROCm-DS team owns this domain"). That does not make this project redundant: `tsnecuda` is a separate, separately-packaged library (conda-forge `tsnecuda`, pip `tsnecuda`) with its own API and its own user base, and it is not part of the RAPIDS stack. If ROCm-DS lands cuML on ROCm, AMD users gain *a* t-SNE, not *this* one. Coordination note rather than a race: nobody else is working on this repo.

### Viability: good. Small, self-contained, and the hard dependency is already ported.

CUDA surface is genuine and small: 20 `.cu` files plus headers, roughly 5.6k lines total, CMake (`enable_language(CUDA)`, `find_package(CUDAToolkit)`), building one shared library `libtsnecuda.so` that the Python package loads by ctypes (`numpy.ctypeslib.load_library('libtsnecuda', ...)` in `src/python/tsnecuda/TSNE.py`). No pybind11/torch extension machinery, so the Python side needs essentially nothing beyond the library being built.

Libraries linked, each with a direct ROCm counterpart, and the live call surface is narrower than the link line suggests:

- **FAISS GPU** (`faiss::IndexIVFFlat`, `faiss::gpu::StandardGpuResources`, `index_cpu_to_gpu`) for the kNN step. This is the one dependency that would be a hard port on its own -- and MOAT already owns it: `projects/faiss` is `completed` on linux-gfx90a, linux-gfx1100, windows-gfx1101/1201/1151 at `validated_sha=ab1dcf71`, and its notes.md has an "## Install as a dependency" recipe (`-DFAISS_ENABLE_ROCM=ON`). Recorded via `moatlib.py set-deps tsne-cuda faiss`. This single fact is most of the viability case.
- **cuBLAS**: live use is `cublasSgemv` in `src/util/reduce_utils.cu` plus handle create/destroy. hipBLAS is 1:1.
- **cuFFT**: `cufftCreate` / `cufftMakePlanMany` / `cufftExecR2C` / `cufftExecC2R` / `cufftDestroy` in `src/fit_tsne.cu` and `src/kernels/nbodyfft.cu`. hipFFT is 1:1 on these entry points. Note `CUDA::cufftw` is on the link line but no source references the FFTW-compat API -- that link can simply be dropped rather than needing a hipFFT FFTW shim.
- **cuSPARSE**: live use is only `cusparseCreateMatDescr` / `SetMatType` / `SetMatIndexBase` in `fit_tsne.cu`; the real sparse work (`cusparseXcsrsort`, `cusparseXcsrgeamNnz`, `cusparseScsrgeam`, `cusparseSgthr`) is entirely commented out in `src/util/math_utils.cu`. So the descriptor is effectively vestigial and hipSPARSE covers it trivially.
- **Thrust** (heavily: ~284 `thrust::device_vector`, plus transform / reduce / transform_reduce / sort / iterators) -> rocThrust, the well-trodden path.
- OpenMP, pthread: unchanged.

Wave-size risk looks low, which matters for the wave64/wave32 gates. Grep finds no `__shfl*`, no `__ballot*`, no cooperative groups, no tensor-core/wgmma use anywhere in `src/`. Upstream's own CMakeLists says as much ("The kernels use no architecture-specific intrinsics ... no warp shuffle/vote, no cooperative groups, no tensor-core / wgmma"). `GpuOptions` in `src/include/options.h` already reads `warpSize` from the device at runtime and the old `warp_size != 32` bail-out is commented out.

Two things the planner should look at rather than assume:

1. `GpuOptions(int device)` in `src/include/options.h` branches launch parameters on `device_properties.major` (==8 Ampere/Ada, >=7, >=6, >=5, >=3). Under HIP, `major` is the gcnArch major, so gfx90a lands in the `>=7` branch and RDNA in `>=7` too. That is not a crash, but the block sizes it picks were swept on an A100 and are AMD-arbitrary. Needs a deliberate HIP branch, or at least a documented decision to keep the defaults.
2. There is no GPU test target to run. `src/test/test.cu` and `src/include/test/*.h` exist but CMakeLists does not build them (upstream dropped gtest/gflags along with the CLI), and `src/python/tsnecuda/test/` contains only `__init__.py`. Validation will therefore be a numerical-quality check (embedding quality on a labeled dataset such as MNIST, e.g. kNN accuracy / trustworthiness, cross-checked against sklearn or a CUDA baseline) rather than `ctest`. Worth planning explicitly; it is the main open question for the validator.

Build type: cmake. `find_package(CUDAToolkit REQUIRED)` and a hand-rolled `cmake/Modules/FindFAISS.cmake` that looks for `faiss/Index.h`, `faiss/gpu/GpuIndex.h` and `libfaiss` -- it will find the MOAT ROCm FAISS install as-is, since faiss's ROCm build keeps the same headers and library name.

### Upstream is alive; a PR has a destination

Not archived. HEAD at intake is `244db8f13811a2abc94e5fe931d7616ef72a06b4`, "4.0.1: modernize Dockerized wheel packaging + pip index (#141)", dated 2026-07-22, i.e. two weeks before intake, and it is a merged numbered PR, so upstream is accepting outside contributions. This is a normal upstream-PR outcome, not fork-only.

Recent upstream work (the CMakeLists comments about aarch64/Grace, CUDA 13, dropping MKL/BLAS, dropping ZMQ and the CLI) shows a maintainer actively simplifying the build for portability. That is a favorable context for an "add ROCm support" PR.
