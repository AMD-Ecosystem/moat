# sonar notes

## Intake (2026-08-06, linux-gfx90a)

Verdict: **decline, `already-supported`**. Sonar ships first-class, actively maintained ROCm support covering every architecture MOAT gates on. There is no port to do.

### Licence

`AGPL-3.0`, tier 3 (strong copyleft). GitHub's field and the LICENSE file agree: the file is the verbatim GNU Affero General Public License v3, 19 November 2007, 662 lines, no addenda. Recorded in `upstream.json.license_spdx`.

No submodules (`.gitmodules` absent). Two nested licence files, both permissive and consistent with the top-level AGPL: `aphrodite/third_party/flash_linear_attention/LICENSE` and `csrc/punica/LICENSE`.

**NVIDIA proprietary licence TEXT is present in two files** (matched against `config/licenses.toml` `tier3.nvidia_proprietary.text_markers`, not a copyright grep):

- `aphrodite/multimodal/evs.py`
- `aphrodite/model_executor/models/radio.py`

Both carry the "NVIDIA CORPORATION and its licensors retain all intellectual property ... strictly prohibited" block directly under an `SPDX-License-Identifier: Apache-2.0` line, an internally contradictory header inherited verbatim from vLLM upstream. Both are pure-Python model/multimodal code with no CUDA in them, so they sit outside anything a port would touch. Recorded rather than escalated because the project is declined; if Sonar is ever revisited for a different reason, this needs a person's decision first.

Scope note: the licence read above is about CONTRIBUTING upstream. It says nothing about using, shipping, or depending on Sonar -- AGPL-3.0 makes that a separate and much heavier question.

### Duplicate effort -- this is the decline

Sonar is [Aphrodite Engine](https://github.com/PygmalionAI/aphrodite-engine) renamed (the Python package is still `aphrodite`, the CLI is still `aphrodite serve`), and Aphrodite is itself a downstream fork of vLLM. README:11 says so outright, and the ROCm code arrives through `[sync]` commits from vLLM.

ROCm is a supported platform, not an afterthought:

- `README.md:40` -- "The installer supports Linux x86-64 (NVIDIA CUDA, AMD ROCm, or CPU)"; `docs/.../installation.md:127` has an "AMD ROCm" section with `APHRODITE_TARGET_DEVICE=rocm`.
- `CMakeLists.txt:52` -- `HIP_SUPPORTED_ARCHS` covers gfx906, gfx908, gfx90a, gfx942, gfx950, gfx1250, gfx1030, gfx1100-1103, gfx1150-1153, gfx1200, gfx1201. That is 17 architectures and it is a superset of every arch MOAT's wave64/wave32/windows gates could be satisfied by.
- `aphrodite/platforms/rocm.py` (1055 lines), `cmake/hipify.py`, and a dedicated `csrc/rocm/` tree of hand-written HIP kernels: `attention.cu` (3719 lines), `skinny_gemms.cu`, `skinny_gemms_int4.cu`, `q_gemm_rdna3.cu`, `q_gemm_rdna3_wmma.cu`, `moe_q_gemm_rdna3.cu`, `qdq_4_rdna3.cuh` -- i.e. RDNA WMMA paths, not just a CDNA port.
- AITER integration throughout: `rocm_aiter_fa.py`, `rocm_aiter_unified_attn.py`, `rocm_aiter_mla.py`, `rocm_aiter_mla_sparse.py`, `rocm_aiter_moe.py`, `rocm_aiter_fusion.py`.
- Per-model AMD implementations under `aphrodite/models/*/amd/`: deepseek_v32, deepseek_v4, kimi_k3, minimax_m3, and inkling (which includes Gluon kernels written specifically for gfx950: `rel_mha_decode_gfx950.py`, `rel_mha_extend_gfx950.py`).
- 66 tuned fused-MoE configs plus ~60 tuned block-FP8 GEMM configs for MI300X, MI308X, MI325X, MI350X/MI350_OAM, MI355X/MI355_OAM, and Radeon R9700 -- these only exist if someone ran the tuner on that hardware.
- ~40 ROCm-specific test files under `tests/` (`tests/rocm/`, `tests/kernels/**/test_rocm_*`, `tests/models/inkling/rocm/`, `tests/weight_loading/models-amd.txt`).
- CI builds ROCm wheels: `.github/scripts/build_rocm_wheel.sh`, `ensure_rocm_builder.sh`, referenced from `release-wheel.yml` and `platform-wheel.yml`. Three ROCm Dockerfiles including `Dockerfile.rocm_gfx1250`.

Freshness: the newest commits touching `aphrodite/platforms/rocm.py` and `csrc/rocm/` are 2026-08-03 ("Enable gfx1250 ROCm architecture") and 2026-07-30 ("[ROCm] Add AITER FP8 ViT encoder attention"), against a repo last pushed 2026-08-04. This is current, not a stale branch.

By `assess-existing-support.md` this is the first bucket -- "mature ROCm/HIP support upstream -> skip (disposition already-supported)". It is also AMD's own work arriving via vLLM, which the same reference says we do not duplicate.

No fork of this upstream exists in AMD-Ecosystem or ROCm, and neither org has a repo named `sonar`. That absence is not a gap: the AMD effort here lives in vLLM upstream and is synced down, so a fork of Sonar would have nothing to add.

### Viability (recorded for completeness)

It does genuinely use CUDA -- 205 `.cu` and 139 `.cuh` files, ~3.9 MB of CUDA, built as a torch extension via CMake -- so it would have been a real target on the size test alone. But the hand-written GPU surface is already dual-targeted through `cmake/hipify.py` plus the `csrc/rocm/` native-HIP tree, which is precisely the porting work MOAT would otherwise do. No MOAT-project dependencies. Upstream is active, not archived (pushed 2026-08-04, 1822 stars).

The one thing worth carrying away: if AMD wants to move the needle on this engine, the leverage is in vLLM (and AITER), not in Sonar. Sonar consumes both.
