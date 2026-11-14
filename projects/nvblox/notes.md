# nvblox notes

## Intake screen (2026-11-05, gap-matrix seed; matrix verdict: unknown)

### Licence
- Top level `LICENSE.md` is Apache-2.0. GitHub reports NOASSERTION (the file starts with a `# License {#license}` markdown heading). Recorded `license_spdx` Apache-2.0.
- `python3 utils/licenses.py scan-nvidia agent_space/nvblox-screen` (depth-1 clone of `public`) flags 116 files with NVIDIA's proprietary header ("NVIDIA CORPORATION and its licensors retain all intellectual property and proprietary rights ... without an express license agreement from NVIDIA CORPORATION is strictly prohibited"). Every hit read; they fall in three groups:
  - `nvblox_torch/` (the PyTorch bindings): all 30 C++/CUDA files under `nvblox_torch/cpp/` (e.g. `src/py_mapper.cu`, `py_nvblox.cu`, `py_scene.cu`, `py_sensor.cu`, `sdf_query.cu`, `include/nvblox_torch/*.h`, `cuda_stream.cpp`) and 59 Python files under `nvblox_torch/nvblox_torch/` (package, tests, examples), plus 2 in `nvblox_torch/internal_tests/`. Real licence text on code a ROCm port of the torch bindings would have to edit.
  - The Bazel build: top-level `BUILD.bazel`, `MODULE.bazel`, `bazel/toolchains/*`, `bazel/third_party/stdgpu/*`, and one `BUILD.bazel` in each of `nvblox/{include,src,tests,examples,experiments,thirdparty}` and `executables/`, `executables/tests/`.
  - Tooling: `ci/*.py`, `ci/install_bazel.sh`, `docker/torch_install_helper.py`, `docs/helpers.py`, `docs/_ext/*`, `hooks/check_for_copyright.py`, `python/scripts/...`, `python/test/...`.
- The core library is clean: the `.cu/.h/.cpp` sources under `nvblox/include`, `nvblox/src`, `nvblox/tests`, `executables/` carry the Apache-2.0 header (91 `.cu` outside `nvblox_torch`), and the CMake build (`CMakeLists.txt`, `nvblox/CMakeLists.txt`, `cmake/`, `nvblox/thirdparty/*/*.cmake`) is not flagged.
- Not stale: upstream's own pre-commit hook `hooks/check_for_copyright.py` accepts exactly two headers, the proprietary one and the Apache-2.0 one, so the split is a deliberate, maintained dual-header regime rather than a leftover. This is real mixed per-part licensing.
- Result against jeff's bar for NVIDIA-owned repos (clean file by file): FAILS for the repository as a whole and for `nvblox_torch`; PASSES for a port confined to the core C++/CUDA library built with CMake (no edit to any Bazel file, `nvblox_torch/` untouched and unbuilt).

### Dependencies
- stdgpu (MOAT project `stdgpu`, fork AMD-Ecosystem/stdgpu; our HIP fixes merged upstream as stotko/stdgpu#484 on 2026-08-06; stdgpu has had a HIP backend of its own). nvblox fetches stdgpu via FetchContent pinned at 71a5aef (2024-02-11), 63 commits behind stdgpu master, and applies six local patches (`nvblox/thirdparty/stdgpu/*.patch`, including CUDA 12.6/13.x namespace fixes and an exposed `occupied` array). A port must bump the pin past #484 (or use `USE_SYSTEM_STDGPU`) and re-derive the patches against it. Not recorded with `set-deps` yet: that belongs to the adoption, since stdgpu is already completed for the gates. Known hazard: the gfx1151 laptop starves stdgpu's cross-wavefront spinlock retry loops under extreme contention, which nvblox's GPU hash will exercise.
- Other thirdparty: eigen, gflags, glog, gtest, sqlite, benchmark (standard permissive, fetched via CMake).

### Duplicate effort
- No upstream PRs or issues matching amd/rocm/hip; README has no AMD mention. No `ROCm/nvblox` or `AMD-Ecosystem/nvblox`; GitHub search for nvblox rocm/hip finds nothing. Upstream active (pushed 2026-10-23, 1.3k stars), not archived. The `amd` and `AMD-AGI` orgs are behind SAML SSO for this token, so an AMD-internal effort there could not be checked.

### Viability
- Genuine CUDA (91 `.cu` in the core, GPU hash on stdgpu, TSDF/ESDF integrators, mesh, CUDA streams/events). Robotics mapping library used by Isaac ROS and cuRobo; a real gap. Viable technically.

### Recommendation
Stop and ask (recorded as a decline, `license-blocked`, pending jeff's ruling). The licensing is genuinely mixed per part, which is a person's call; if jeff accepts a core-only scope the case flips to take-up.

### Question for jeff
nvblox is Apache-2.0 at top level and its core library (`nvblox/include`, `nvblox/src`, tests, executables, the whole CMake build; 91 `.cu`) carries the Apache-2.0 header, but `nvblox_torch/` (30 C++/CUDA files including `py_mapper.cu`, `sdf_query.cu`, plus 59 Python files), every Bazel build file and the CI/doc tooling carry NVIDIA's "proprietary rights ... strictly prohibited" header, and upstream's own copyright hook accepts both headers, so the split is deliberate. Do you (a) take it up scoped to the Apache-2.0 core library via CMake, never editing a proprietary-headered file and leaving `nvblox_torch` unported, (b) decline as license-blocked because the repository is not clean file by file, or (c) something else? Either take-up path needs stdgpu's pin bumped past our merged stotko/stdgpu#484.
