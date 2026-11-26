# spconv_cpp notes

## Origin

Taken up by jeff's 2026-11-23 ruling on autoware_universe's planner question Q1 (option A,
recorded in `projects/autoware_universe/notes.md` on `port/autoware_universe`, "## Rulings
on the planner's questions"): `autowarefoundation/spconv_cpp` becomes its own project, a
pre-generated C++ snapshot of spconv and cumm that would be regenerated with a HIP cumm
backend, depending on the MOAT projects `spconv` and `cumm`. autoware_universe adds
`spconv_cpp` to its own `depends_on` only when its stage 4 starts.

## Intake -- 2026-11-23, linux-gfx942 (fleet job, headless, no GPU used)

Recommendation: **FORK**, with the work held behind `spconv` and `cumm` (both
`awaiting-fork`). Upstream: https://github.com/autowarefoundation/spconv_cpp, default
branch `main`, screened at `aa1021e68` (2026-04-14, "feat: support NVIDIA DRIVE AGX
Thor"). Not archived, not a fork, 25 stars.

### What it is

A distribution snapshot, not a source project: the output of
`python -m spconv.gencode --inference_only=True` (spconv 2.3.8) plus cumm's `tensorview`
headers, with CMake + CPack files that build two Debian packages (`cumm`, header-only
INTERFACE target; `spconv`, a shared library). README names cumm 0.5.3 / spconv 2.3.8 /
CUDA 13.0, SM 7.5 to 12.0; `HOW_TO_UPDATE.md` documents regeneration from cumm `v0.8.2`
and spconv `v2.3.8` (the README and `cumm/version.txt` say 0.5.3; the planner should pin
which cumm the snapshot really came from). Autoware consumes it for its spconv TensorRT
plugins, the bevfusion voxeliser/backbone and ptv3's backbone.

Tree: 22,439 files, 196 MB. `spconv/include/spconvlib` 19,759 headers and
`spconv/src/spconvlib` 2,559 sources (858 `.cu`, 154 `.cc`, the rest `.h`/`.txt`), all
generated and almost all without any header; `cumm/include/tensorview` 100 files, copied
from cumm. `spconv/CMakeLists.txt` is `project(spconv LANGUAGES CXX CUDA)`;
`cumm/CMakeLists.txt` uses `find_package(CUDAToolkit)` and links `CUDA::cudart`.

### Licence -- Apache-2.0 top level; recorded as cumm's compound expression, tier 3 by the gate

    python3 utils/licenses.py check autowarefoundation/spconv_cpp      # Apache-2.0, tier 1 (top level only)
    python3 utils/licenses.py scan-nvidia agent_space/spconv_cpp-screen  # clean: no NVIDIA proprietary licence text

Recorded with `set-license-spdx`:
`Apache-2.0 AND BSD-3-Clause AND MIT AND BSL-1.0 AND Unlicense AND GPL-3.0-or-later WITH GCC-exception-3.1`
(evidence: `LICENSE` at `aa1021e68`), the same expression as cumm's record, because the
snapshot vendors cumm's headers, including the four GPL-headed ones. `license-gate
spconv_cpp` is therefore closed (tier 3) until a person runs `record-license-clearance`.

Full scan (every tracked file grepped for copyright, SPDX, licence text and
"taken/adapted/copied from" provenance; no submodules; three licence files):

- `LICENSE` and `spconv/LICENSE`: identical Apache-2.0 ("Copyright 2019-2021 Yan Yan").
  `cumm/LICENSE`: Apache-2.0 ("Copyright 2021 Yan Yan"). README: "under the same license as
  spconv (Apache 2.0)". The generated `spconvlib` tree carries no per-file headers; it is
  spconv's generator output under spconv's Apache-2.0.
- Everything that is not Apache-2.0 lives in `cumm/include/tensorview/`, and each one is a
  file cumm's own intake already tabulated (`projects/cumm/notes.md` on `port/cumm`,
  "### Licence"):

| path (under `cumm/include/tensorview/`) | licence | port modifies? |
|---|---|---|
| `core/nvrtc_std.h`, `core/nvrtc/core.h`, `core/nvrtc/limits.h`, `core/nvrtc/type_traits.h` | **GPL-3.0-or-later WITH GCC-exception-3.1** (libstdc++, "Copyright (C) 2008-2021 Free Software Foundation"); `type_traits.h:492` also has a CCCL libcudacxx snippet (Apache-2.0 WITH LLVM-exception) | no, if regenerated from the ported cumm under jeff's cumm condition (see below) |
| `cuda/driver.h` and 11 `gemm/arch/*`, `gemm/dtypes/*` files | Apache-2.0 / BSD-3-Clause, "Copyright (c) 2017-2021, NVIDIA CORPORATION" (CUTLASS / TensorRT origin; `gemm/dtypes/float8.h` has `SPDX-License-Identifier: BSD-3-Clause`) | replaced by the ported cumm's copies |
| `core/cc17.h`, a block in `tensor.h` | BSD-3-Clause (PyTorch/Caffe2 notice); `tensor.h` also "Copyright 2019-2020 Kakao Brain" | same |
| `thirdparty/nlohmann/json.hpp` | MIT (nlohmann json 3.x, with its MIT/BSL third-party snippets) | no |
| `thirdparty/tsl/robin_*.h` | MIT (Thibaut Goetghebuer-Planchon) | no |
| `prettyprint.h`, `core/mp_helper.h` | BSL-1.0 | no |
| `core/arrayops/linalg.h` | Unlicense ("from https://github.com/sgorsten/linalg") | no |
| `profile/cuda_profiler.h:84` | a few lines "adapted from cpython pytime.c" (PSF-2.0) | no |
| Stack Overflow snippets: `core/array.h`, `pybind.h`, `utility/tuplehash.h`, and `spconv/src/.../SpconvOps_count_bits.cu` | CC BY-SA by origin, a few lines each (the last is spconv's popcount snippet, also in spconv's record) | no |

What the clearance decision is about: the only new question beside cumm's is whether the
same GPL-with-exception headers may be redistributed a second time inside this snapshot.
jeff cleared cumm on 2026-11-23 with the condition that the port does not modify those
four files and puts a hipRTC path in new files. A regenerated spconv_cpp would copy the
ported cumm's headers verbatim, so the same condition holds here by construction.

### Duplicate effort and upstream state

- Upstream PRs (all states): none mentions HIP/ROCm/AMD. Branches `main`,
  `perf/yank-out-thrust` (draft PR #12 "perf: remove Thrust", closed 2026-05-26). Issues:
  #2, #3, #6, none about AMD. Docs grep (`README.md`, `HOW_TO_UPDATE.md`, both inner
  READMEs) for amd/rocm/hip/gfx: no hits beyond `amd64`.
- `gh search repos spconv --owner AMD-Ecosystem` and `--owner ROCm`: none. `spconv_cpp`
  search: only upstream and `Zhao-Qihao/spconv_cpp` (an older CUDA 12.3 copy of the same
  snapshot). Ten GitHub forks of upstream: none with AMD work found by name.
- `autowarefoundation` org issues/PRs mentioning ROCm or HIP: none relevant (one 2023
  Debian-packaging issue).
- AMD work exists only at the spconv/cumm level, and jeff ruled on it for those projects
  on 2026-11-23 (proceed independently of `ZJLi2013/spconv_rocm` / `ZJLi2013/cumm-rocm`).
  Found here as well: `HYGON-AI/spconv-das` (Apache-2.0, Hygon HCU downstream of
  `ZJLi2013/spconv_rocm` branch `rocm`, pushed 2026-11-19) and `jiaqiwang969/spconv-rocm`
  (1 KB, README only, no licence). Neither ships a pre-generated C++ snapshot. No
  duplicate at the spconv_cpp level.
- Activity: 9 PRs from 6 contributors, 7 merged; last merged PR #10 (2026-03-19), last commit 2026-04-14, last push 2026-05-26. Issue
  replies come from contributors, not reliably from maintainers (#3, a doc fix, has no
  reply). Active enough to accept a PR.

### Viability -- yes, but only after spconv and cumm

- Genuine CUDA: 858 generated `.cu` files; 3,617 files with inline `asm`, 2,515 with
  `mma.sync`/`ldmatrix`/`cp.async` (tensor-core PTX from cumm's GEMM generator), 391 with
  `__CUDA_ARCH__` dispatch.
- Generated CUDA belongs to its generator, so hand-hipifying the 22k files would be the
  wrong port. The work is: (1) a HIP backend in cumm's code generator and spconv's
  `gencode` (MOAT `cumm` and `spconv`), (2) regenerate this snapshot with it, (3) a HIP
  branch in the two CMakeLists (CUDA language, `CUDA::cudart`, the arch list) and the CPack
  metadata, (4) build and run on AMD. Under jeff's spconv scope ruling (MFMA and WMMA
  kernel families required), the regenerated snapshot inherits those kernels; the
  `--inference_only=True` subset is what Autoware needs.
- Hard dependencies: `spconv`, `cumm` (recorded at scaffold with `--deps`). `ccimport` is
  needed only to regenerate and is dispositioned already-supported.
- Cost is small once the generators exist; until then nothing here can start. The
  selector already holds it on `depends_on`.
- Upstream destination: a regenerated snapshot plus CMake HIP branch is a large
  generated-file diff; whether autowarefoundation would take it as a PR or prefer the
  HIP build as a separate snapshot is a question for the planner and, later, the
  maintainers.

### Recorded

    python3 utils/moatlib.py scaffold autowarefoundation/spconv_cpp --branch main --deps spconv cumm
    python3 utils/moatlib.py set-license-spdx spconv_cpp "Apache-2.0 AND BSD-3-Clause AND MIT AND BSL-1.0 AND Unlicense AND GPL-3.0-or-later WITH GCC-exception-3.1" --evidence https://github.com/autowarefoundation/spconv_cpp/blob/aa1021e686fd5dac91288a772ca8b1b76062a7e4/LICENSE
    python3 utils/moatlib.py set-intake spconv_cpp fork ...
    python3 utils/moatlib.py set-state spconv_cpp linux-gfx942 awaiting-fork --agent intake

### Questions for jeff

1. Licence clearance (`record-license-clearance spconv_cpp`): the snapshot redistributes
   cumm's four GPL-3.0-or-later WITH GCC-exception-3.1 headers. Same condition as cumm
   (the port does not modify them; regeneration copies the ported cumm's headers)?
2. Fork `AMD-Ecosystem/spconv_cpp` (the adopt decision; the work waits on spconv and cumm
   either way).

## Planning -- 2026-11-26, linux-gfx90a (fleet job, headless, no GPU used, nothing built)

Plan written to `plan.md`; surface accounting in `surface.json` (802 components: 779 generated by `surface.py`, 23 added; 665 scoped out with reasons, 664 of them the generated NVIDIA tensor-op kernel libraries; `surface.py check spconv_cpp` clean). Built on cumm's plan (`port/cumm` `1614d936`). spconv's planner held the lock on `port/spconv` (`2edafec1`) with no plan written yet, so the interface needed from spconv is written in `plan.md` as requirements ("Interface this port needs from cumm and spconv").

Read-only inputs, cloned into the job directory: upstream `aa1021e68` at `projects/spconv_cpp/src` (blob-less full history), spconv `263d6b474` (= v2.3.8), cumm `d971d02a0` plus tags, pccm `860790f`. Read-only `gh search code` / `gh api` on autoware_universe for the consumer calls.

Decisions, in brief:
- Build class `cmake` (`set-ext-type spconv_cpp cmake`); sources are generated by pccm/ccimport, so the port has a regeneration half (Python, output committed) and a CMake HIP half.
- Strategy: a second, regenerated tree under `rocm/` (`rocm/cumm`, `rocm/spconv`, `rocm/tools`, `rocm/test`), same CMake package and target names, CUDA tree byte-identical. Reasons: kernel sets differ by backend; the CUDA tree carries hand edits a regeneration loses; its `tensorview` is cumm v0.5.3 while the port's is v0.8.2+.
- pccm's `gen_cmake` (`pccm/core/__init__.py:2386-2489`) writes only the `nvcc` bucket (under `COMPILE_LANGUAGE:CUDA`) and the generating OS's `g++`/`cl` bucket, plus absolute include paths; cumm's HIP flags live in the `clang++` bucket, so they never reach generated CMake. The regeneration script therefore normalises the per-target CMake files to HIP and writes `hip_flags.cmake` from cumm's exported flag set. pccm stays unmodified, as with cumm.
- Hard dependencies unchanged: `depends_on: [spconv, cumm]` (already recorded). Regeneration-time tools, not MOAT dependencies: pccm and ccimport from PyPI (ccimport dispositioned already-supported). ROCm components used by the build: amdhip64 (`hip::host`), hipBLASLt, rocThrust/hipCUB/rocPRIM (headers, through the regenerated spconv ops). Optional: OpenMP (`cumm::cumm_cpu` links it when found, as upstream).

Facts pinned while planning (for the porter):
- Upstream after the generated initial commit is hand-maintained: all 1,548 per-target CMake files rewritten (`969fe3e0`, `e45eba5a`, `111ffdf6`), `SpconvOps_sort_1d_by_key{,_split}_allocator_v2.cu` rewritten Thrust -> `cub::DeviceRadixSort` and `CompileInfo` (11,0) added (`aa1021e6`), `reverse(uint64_t)` made to throw (`5ce0f623`), `arch_defs.h` SM list (`1b23b70d`).
- `cumm/include/tensorview` = cumm v0.5.3 (`2bb4166`) + `core/printf2.h` `<cstdint>` + `cuda/arch_defs.h` SM entries. spconv v2.3.8 `setup.py` pins `cumm>=0.7.11,<0.8.0`; `HOW_TO_UPDATE.md` says v0.8.2; README and `cumm/version.txt` say 0.5.3.
- libspconv has no NVRTC: both tuners' `cached_get_nvrtc_params` throw "not implemented in c++, must be overrided in python!!!". No hipRTC work in this port.
- `ConvTunerSimple_get_all_available.cc:59-63` drops every SIMT fp16 kernel when `arch >= (7,0)`; on HIP the CC tuple of gfx90a is 9.0, so with no AMD-family match there is no algorithm (spconv requirement 5 in `plan.md`).
- Autoware consumers: `autoware_tensorrt_plugins` (`implicit_gemm_plugin.cpp`, `indice_conv_plugin.cpp`, `get_indices_pairs{,_implicit_gemm}_plugin.cpp`; fp16/fp32 only, `PLUGIN_ASSERT`) and `autoware_bevfusion/lib/preprocess/preprocess_kernel.cu` (`Point2Voxel` 3D); CMake `find_package(cumm)` + `find_package(spconv)`, `spconv::spconv`.
- `surface.json` keeps the CUDA census's generated file paths as per-directory counts (`cuda.file_groups`): the full 13,362-path list is 1.9 MB, over `check.py`'s 1 MB file gate. Recorded under `removed_from_floor`; no component removed.

### Questions for jeff (none blocks the porter, who waits on cumm and spconv anyway)

1. Windows gate. spconv_cpp is a Linux-only packaging repository (Debian packages, a Jetson cross toolchain) and its only consumer, autoware_universe, has your approved windows waiver. The CMake HIP build itself should work on Windows with the ROCm SDK. Options: A (planned) validate on windows-gfx1151 last, build and harness only; B suggest a windows waiver on the same grounds as autoware_universe (an agent may only suggest; an unapproved suggestion blocks).
2. Debian package names for the ROCm build: `cumm-rocm7` / `spconv-rocm7` (per your per-ROCm-major naming rule), with `Conflicts:` on the CUDA `cumm`/`spconv` packages, keeping the CMake package and target names (`cumm::cumm`, `spconv::spconv`) so Autoware's `find_package` lines work unchanged. Agree, or another name?
3. Upstream vehicle (for later, after validation): the ROCm tree is a large generated addition (estimate 3k to 8k files, 30 to 80 MB). One upstream PR adding `rocm/`, or ask the maintainers whether they prefer a separate branch or repository. A moat-checkup question, recorded now so it is not a surprise.
4. int8 inference on HIP follows cumm's question 1; Autoware's spconv plugins use fp16/fp32 only. No separate decision needed here unless you want int8 for libspconv specifically.

Scoped out with a reason (not a partial-port cut; listed for visibility): `extras/arm64-toolchain.cmake` (NVIDIA SBSA/Jetson cross compilation; no ROCm aarch64 SDK for these GPU targets; the file is unchanged).

Recorded:

    python3 utils/moatlib.py set-state spconv_cpp linux-gfx90a planning --agent planner
    python3 utils/moatlib.py set-ext-type spconv_cpp cmake
    python3 utils/surface.py generate spconv_cpp      # then scoped-out/added entries, file list collapsed
    python3 utils/surface.py check spconv_cpp         # fully accounted for
    python3 utils/moatlib.py set-state spconv_cpp linux-gfx90a planned --agent planner
