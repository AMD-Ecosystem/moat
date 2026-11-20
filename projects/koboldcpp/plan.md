# koboldcpp port plan

Planner run 2026-11-20 on linux-gfx950, against upstream `concedo` @ 6d2fedf8624e (the fork's only branch, identical to upstream's) and `concedo_experimental` @ ee5ff66bd1e0. This is a validation project by Jeff Daily's ruling of 2026-11-10 (notes.md): validate upstream's own HIP build on every required gate and fix what breaks on the fork. **Three questions need Jeff's ruling before a porter starts; they are at the top of notes.md, "Planning (2026-11-20)".** This plan states the recommended answer to each and marks where the work depends on it.

## Project

- Upstream: https://github.com/LostRuins/koboldcpp (AGPL-3.0, tier 3; any upstream offer needs a recorded licence clearance first).
- Upstream branches: `concedo` (default, release branch), `concedo_experimental` (development; all 30 most recent merged PRs, including the two ROCm CI PRs #1823 and #2349, target it), `upstream` (a mirror of ggml-org/llama.cpp master).
- Fork: https://github.com/AMD-Ecosystem/koboldcpp, which carries only `concedo`. No `moat-port` yet.
- Record problem: `status.json` has `fork_default_branch: "main"`, which does not exist on the fork or upstream. Fixing it is `set-base-branch`, a person's ruling (question 2).

## Existing AMD support

**Mature first-party HIP build upstream, Linux only.** Decision: validate and improve it; do not re-port.

- `make LLAMA_HIPBLAS=1` is the documented ROCm build (README "Manual build" table, Linux section). Makefile lines 302-372 compile every `ggml/src/ggml-cuda/*.cu`, the template instances, and koboldcpp's three legacy `.cu` files with `hipcc -x hip --offload-arch=...`, and link hipBLAS + rocBLAS. The block is actively maintained (SDK-layout-aware hipcc/hipconfig discovery, `rocm-check`, `ROCM_LIB_PATH`).
- Release CI `.github/workflows/kcpp-build-release-linux-rocm.yaml` builds `KCPP_PORTABLE=1 ./koboldcpp.sh dist` in `rocm/dev-ubuntu-22.04:7.14.0-full` and uploads `koboldcpp-linux-x64-rocm` to the `rocm-rolling` release.
- Windows: no documented or released ROCm build. README points Windows AMD users at Vulkan. Two undocumented routes exist in the tree (see Risks R1, R2).
- Community: YellowRoseCx/koboldcpp-rocm, about 20 releases behind, non-authoritative; a hint for the Windows build at most, never code to inherit.
- Upstream does not link platform forks instead of merging; an upstream PR is a valid vehicle once a licence clearance is recorded.
- Performance kernels: the vendored ggml backend already has AMD-native paths (MFMA on CDNA, WMMA on RDNA3/4, `__builtin_amdgcn_sdot4`/`sudot4` dp4a). Mechanical-vs-rewrite does not arise; no rewrite is planned.

## Build classification

`make` (`set-ext-type koboldcpp make`). Evidence: README build table and Windows section; `CMakeLists.txt` lines 1-8 say it is "ONLY intended for CUBLAS builds on Windows Visual Studio" and unsupported otherwise. Not a torch extension: no `setup.py`/`CUDAExtension`, and `transformers[torch]` in `requirements.txt` serves the model-conversion scripts only. The Python layer is a ctypes launcher (`koboldcpp.py` line 953 loads `koboldcpp_hipblas.so/.dll`).

## Port strategy

Strategy A-shaped, already implemented upstream: sources keep CUDA spelling and `ggml/src/ggml-cuda/vendors/hip.h` (plus the inline shims at the top of each `otherarch/*-cuda.cu`) maps them to HIP. The port work is fixes inside that existing structure, not a new compat layer:

1. Linux: build and run at the fork head on wave64 and wave32; fix what breaks.
2. Windows: make one Windows ROCm route build and run (question 1; recommended route is the existing CMake MSVC/clang branch with TheRock).
3. Keep the CUDA path byte-unchanged for Linux `make LLAMA_CUBLAS=1` and the Windows CMake CUDA build (nvcc compile on the mi355 host before presenting).

## CUDA surface inventory

`surface.py` census: 299 files, 240 kernels, 331 launches, 111 warp-intrinsic sites, 193 cuBLAS sites, 25 NCCL, 8 CUB, 18 driver-API sites.

| area | what | HIP status |
|---|---|---|
| `ggml/src/ggml-cuda/` (71 `.cu` + 120 template instances) | vendored llama.cpp backend, synced to llama.cpp aa588a34 (2026-10-30) on `concedo`; `concedo_experimental` has 634 more commits, 91 files changed in the GPU dirs | upstream llama.cpp HIP path, handles CDNA1-4 (gfx950 = `GGML_CUDA_CC_CDNA4`), RDNA1-4, gfx1150/1151 |
| koboldcpp patches to that backend | `GGML_CUDA_FORCE_MMQ` defined in `mmq.cuh` with a runtime `g_mul_mat_q` switch (`--usecuda mmq/nommq`); pool allocator rewrite in `ggml-cuda.cu` (frees the largest buffer when nothing fits, nommq only); revert of llama.cpp's "BF16 -> F32 compute on AMD without fast BF16 when ne11 > 32"; FA instance subset (`fattn.cu` mirrors 5 vec pairs); `flash_attn_mask_to_KV_max` launched with chevrons | shared code; must be exercised on HIP (test plan T1, T2) |
| `otherarch/ggml_v3-cuda.cu` (10.3k lines, 63 kernels) | legacy GGML v3 backend with its own HIP shim and `__dp4a` | logical 32-wide warps (`WARP_SIZE 32`, `__shfl_xor(..., 32)`): wave64-safe; `__dp4a` falls back to scalar on gfx942/gfx950/gfx12 (correct, slower) |
| `otherarch/ggml_v2-cuda.cu`, `ggml_v2-cuda-legacy.cu` (10 kernels) | legacy GGML v2 hipBLAS SGEMM/GemmEx + dequant | inline shim; no warp-width assumptions beyond width-32 shuffles |
| cuBLAS | hipBLAS/rocBLAS via `vendors/hip.h` and the otherarch shims | rocBLAS; `HIP_VERSION >= 60500000` gates the compute-type enums |
| NCCL (`GGML_USE_NCCL`) | multi-GPU allreduce | not enabled by the Makefile; out of the HIP build as upstream ships it |
| CUB (`GGML_CUDA_USE_CUB`) | argsort/top-k | CUDA-only define (`common.cuh:118`); HIP uses the fallback kernels |
| driver API (VMM) | `cuMemCreate` pool | disabled on HIP by `-DGGML_HIP_NO_VMM` |
| HIP graphs | `GGML_HIP_GRAPHS` | not defined by the Makefile; off, as upstream ships it |

## Risk list

- **R1 Windows CMake route is broken as written.** `CMakeLists.txt:217` builds `ggml-rocm` from `${GGML_SOURCES_CUDA}` instead of the `GGML_SOURCES_ROCM` list assembled just above it (empty or CUDA-only content when the HIP branch is taken); `LLAMA_CUBLAS` defaults ON; ROCm is located with a PowerShell glob of `C:/Program Files/AMD/ROCm/` (the legacy HIP SDK), not a TheRock SDK; sources are marked `LANGUAGE CXX` with `-xhip`, which needs the ROCm clang as the CXX compiler (cuda-to-rocm `windows.md`, one-clang-driver entries).
- **R2 Windows Makefile route mixes ABIs.** Under w64devkit, `CXX` is MinGW g++ while `HCXX` is hipcc targeting the MSVC ABI; the `koboldcpp_hipblas.dll` link combines both. Expected not to link; also `HIPLDFLAGS` passes `-Wl,-rpath=`, which lld-link rejects.
- **R3 Windows runtime loading.** `koboldcpp.py:1019` adds `%HIP_PATH%\bin` to the DLL search path; a TheRock SDK does not set `HIP_PATH`, and the HIP runtime, rocBLAS and its `library/` kernels must be found or staged (windows.md, "Stage the ROCm runtime next to the .exe").
- **R4 Portable target list.** The `LLAMA_PORTABLE` default `GPU_TARGETS` (Makefile:328) lacks gfx950, gfx1150 and gfx1151 (and gfx1103, gfx1152). The release CI container has no GPU, so the published `koboldcpp-linux-x64-rocm` is expected to carry no code object for MI350/MI355 or Strix Halo. Verify with `roc-obj-ls`/`llvm-objdump --offloading` on the rolling binary before fixing; adding a target must stay within what the CI's rocBLAS ships.
- **R5 BF16 compute-type revert on AMD.** koboldcpp keeps BF16 compute for any batch on AMD GPUs without fast BF16 hardware, where llama.cpp switches to F32 above ne11 = 32. Check numerics and speed with a BF16 GGUF under `nommq` on an arch without fast BF16 (gfx90a/gfx1030 if available; record which archs were covered).
- **R6 Pool allocator rewrite.** `pool_size -= size` after freeing a buffer of `b.size` bytes (`ggml-cuda.cu`, nommq-only path) looks like an accounting slip in shared code; not HIP-specific, but `hipFree` is synchronizing and the path only runs in `nommq` mode. Exercise it (long nommq generation with growing batch), record, and report as a shared-code finding rather than a HIP fix.
- **R7 FA instance subset.** HIP builds only the 5 fattn-vec pairs koboldcpp enables; `--quantkv` combinations outside them must fall back cleanly, not abort. Test each `--quantkv` value with FA on.
- **R8 Legacy `__dp4a` scalar fallback** on gfx942/gfx950/gfx12 in `ggml_v3-cuda.cu`: correctness-neutral; only a perf item. No change unless legacy-format speed matters (it should not gate).
- **R9 Fast-moving base.** `concedo_experimental` moved 634 commits in three weeks and re-syncs llama.cpp every few days. Validate at one recorded fork head; expect sync rounds.
- Fault classes checked and clear at plan time: no hard-coded wave-32 lane math outside logical width-32 shuffles (legacy files); no textures; no managed memory (except opt-in `GGML_V3_HIP_UMA`); host flags passed to hipcc are `-O3 -fno-finite-math-only` (no fast-math leak into device code).

## File-by-file change list

Expected, to be confirmed by the porter's builds; anything not listed that breaks gets fixed the same way and recorded in notes.md.

- `Makefile`: add missing gfx targets to the `LLAMA_PORTABLE` list (R4) if verified; Windows only if question 1 chooses the Makefile route.
- `CMakeLists.txt` (if question 1 = CMake route): `ggml-rocm` source list (R1); ROCm root from `ROCM_PATH`/`HIP_PATH`/TheRock `rocm-sdk path --root` before the HIP SDK glob; `GPU_TARGETS`/`CMAKE_HIP_ARCHITECTURES` pass-through; whatever the clang-only Windows link needs, guarded to `WIN32 AND LLAMA_HIPBLAS`.
- `koboldcpp.py`: Windows ROCm DLL directory discovery beyond `HIP_PATH` (R3), guarded to the hipBLAS library on Windows.
- `README.md`: the ROCm build and run line for each platform that works, and the `--usecuda` "Nvidia Only" wording (it also drives ROCm).
- No kernel changes are expected on Linux. A kernel fix, if a test forces one, must hold on both wavefront widths.

## Build commands

Linux (wave64 and wave32), from the fork clone, CPU jobs capped to physical cores:

```
make clean
make -j$MAX_JOBS LLAMA_HIPBLAS=1 GPU_TARGETS=<gfx>          # host arch only, as validators build
make -j$MAX_JOBS LLAMA_HIPBLAS=1 LLAMA_PORTABLE=1           # once per round: the release target list compiles
```

The release recipe (`KCPP_CUDA=rocm KCPP_PORTABLE=1 ./koboldcpp.sh dist`) needs conda/mamba and builds Vulkan too; run it once on one Linux host per round as the packaging check, not per validator.

Windows: depends on question 1. Recommended: TheRock SDK, `cmake -G Ninja -DLLAMA_CUBLAS=OFF -DLLAMA_HIPBLAS=ON -DCMAKE_C_COMPILER=<rocm clang> -DCMAKE_CXX_COMPILER=<rocm clang++> -DGPU_TARGETS=gfx1151`, then `cmake --build`.

CUDA no-regression (mi355 host, `/opt/conda/envs/cuda-12.8/bin/nvcc`): `make -j$MAX_JOBS LLAMA_CUBLAS=1` to completion; if CMake is touched, also a CMake CUDA configure+build on Linux.

## Test plan

Upstream has no GPU test suite. The gate is two validator-side harnesses (never committed to the fork) plus the non-GPU tests.

- **T1 kernel level: llama.cpp `test-backend-ops` against koboldcpp's ggml.** Check out ggml-org/llama.cpp at the sync commit of the fork head (second parent of the latest "Merge branch 'upstream'" commit; aa588a34 for `concedo` 6d2fedf8), overlay koboldcpp's `ggml/include`, `ggml/src/ggml-cuda/` and `ggml/src/*.c[pp]` (keep llama.cpp's CMake files, which koboldcpp deleted), build `-DGGML_HIP=ON -DGPU_TARGETS=<gfx> -DGGML_HIP_NO_VMM=ON` with the same FA instance defines, and run `test-backend-ops -b ROCm0` (mode `test`). Record pass/fail/skip counts. A failing op is cross-checked against pristine llama.cpp at the same commit, to separate koboldcpp patches from inherited behaviour. If the overlay cannot build within the budget, fall back to pristine llama.cpp at the sync commit and say T1 then covers inherited code only.
- **T2 end to end, HIP against CPU.** Same model, same prompt, greedy (`temperature 0`, `top_k 1`, fixed seed) through the HTTP API (`/api/v1/generate`, or `--prompt` mode), `--usecuda` vs `--usecpu`: first 64 tokens identical or explained by a logit near-tie. Matrix: a Q4_K_M and a Q8_0 GGUF (small Llama/Qwen class) under `mmq` and `nommq`; a BF16 GGUF under `nommq` (R5); FA with `--quantkv` f16, q8_0, q4_0, bf16 (R7); one legacy GGML v3 and one v2 model (otherarch files); `--benchmark` for tokens/s.
- **T3 modality smoke** (one run each, output sane and the GPU actually used, kernel dispatches counted or VRAM observed): image generation (small SD1.5 or SDXL-turbo GGUF), whisper (tiny/base), TTS (OuteTTS or Qwen3-TTS), embeddings, `--mmproj` vision. Music (ACE-Step) and two-GPU `--tensor_split` where hardware and models allow; otherwise recorded as owed with the reason (scope per question 3).
- **Non-GPU regression:** `python3 tests/test_koboldcpp.py`, `python3 tests/test_numeric_parsing.py`, `python3 tests/test_anthropic_tool_choice.py`; `tests/test_autoguess.py` where network and `transformers` are available. Compare with the same tests at the base commit.
- Platforms: wave64 (gfx942 or gfx950), wave32 (gfx1100/gfx1201 Linux), Windows (gfx1151) last, after both Linux gates pass at the same head.

## Open questions

See notes.md, "Planning (2026-11-20)": (1) Windows route, (2) PR base branch and the wrong `fork_default_branch`, (3) how deep each modality must validate. Later, not blocking a porter: the AGPL-3.0 licence clearance before anything goes upstream.
