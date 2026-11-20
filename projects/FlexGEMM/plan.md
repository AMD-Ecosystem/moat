# FlexGEMM port plan

Planner run 2026-11-20 on linux-gfx942 (fleet job, no GPU used, nothing built except a
preprocessor-only check of a torch header). Upstream clone inspected at `main` 6dd94a8.

## Project

- Upstream: https://github.com/JeffreyXiang/FlexGEMM (MIT), default branch `main`, head
  6dd94a8 (2026-04-22, unchanged since the November re-screen).
- Fork: https://github.com/AMD-Ecosystem/FlexGEMM, `main` = upstream 6dd94a8. The porter
  creates `moat-port` from it.
- Why it is in the pipeline: TRELLIS.2 (`depends_on = ["CuMesh", "FlexGEMM"]`) owes all
  three gates. Jeff ruled `fork` on 2026-11-04, reversing the August `already-supported`
  recommendation, because upstream's ROCm support covers Linux on MI300X only. Nothing
  covers Windows or wave32 (intake re-screen in notes.md).
- Target branch: `main`, because that is what TRELLIS.2's `setup.sh` clones. Upstream's
  `dev/all_triton` ("FlexGEMM 2.0", PR #28, a pure-Triton rewrite) has not been merged to
  `main`. If it is merged later, that is a sync round, not part of this plan.

## Existing AMD support

**Improvable: upstream itself is the authoritative AMD port, on Linux and CDNA only.**

- JeffreyXiang/FlexGEMM#18 (ZJLi2013, merged 2026-04-20) is the upstream ROCm enablement:
  a `__syncwarp` stand-in in `migemm_neighmap_pp.cu` and `allow_tf32 = not torch.version.hip`.
  Its evidence is MI300X with ROCm 6.4.3, torch 2.6 and Triton 3.2. It is authoritative
  because the maintainer merged it, and it is the base this port builds on, not something
  to redo.
- `setup.py` already selects ROCm (`IS_HIP_EXTENSION`, `BUILD_TARGET`, `GPU_ARCHS` ->
  `--offload-arch`). The Triton configs have a `'hip'` platform list and an `MI300X`
  device list, and `autotune_cache.json` ships MI300X autotune results.
- `ATLAS-0321/FlexGEMM-rocm` (a duplicated `kernels/hip/` tree, never offered upstream) is
  a community fork and not authoritative. Ignore it.
- Delivery: an upstream PR to JeffreyXiang/FlexGEMM. Upstream merges platform work into
  the tree; it does not link forks. Open PR #31 (rwfsmith, Windows/MSVC build, no
  maintainer response since 2026-08-21) edits the same `setup.py` Windows block. The port
  must stay mergeable with or without #31, so keep the `setup.py` change small and
  explain the overlap in the PR body.
- Port versus rewrite: the GEMM work is Triton (`flex_gemm/kernels/triton/`), which already
  compiles natively for AMD through Triton's AMD backend, with MFMA/WMMA. There is no
  CUTLASS, no PTX and no tensor-core intrinsic in the CUDA extension. Nothing needs an
  AMD-native rewrite. Performance tuning for RDNA (autotune configs and cache entries) is
  out of scope for correctness: optional follow-up evidence, not a gate.

## Build classification

**torch-extension** (`setup.py`):

- `setup.py:2` `from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION`
- `setup.py:55-70` one `CUDAExtension(name="flex_gemm.kernels.cuda", sources=[6 .cu + ext.cpp])`
- `pyproject.toml` build requires `torch`. Runtime needs `torch>=2.4`, `triton>=3.2`
  (`triton-windows` on Windows) and `filelock`.

Recorded with `moatlib.py set-ext-type FlexGEMM torch-extension` (it was
`setuptools-cuda-extension` from intake, a non-canonical spelling).

## Port strategy

**Strategy B (torch hipify).** Linux needs no source change. Torch hipifies the six `.cu`
files at build time, and PR #18 is the only device-code adjustment the extension
needed. The work is in `setup.py`:

1. a shared, non-ROCm fix for the C++ standard (current torch refuses `-std=c++17`);
2. a Windows+HIP flag set, because the existing Windows branch hands nvcc-only
   `-Xcompiler=` flags to hipcc;
3. a sensible default target on Windows, where `--offload-arch=native` fails.

After that comes validation on wave64, wave32 and windows, plus a dependency recipe
for TRELLIS.2.

## CUDA surface inventory

The surface census (`surface.json`) gives 11 CUDA files, 22 kernels, 31 launches and 3
warp-intrinsic sites. In detail:

| Area | Files | CUDA features | HIP mapping / risk |
|---|---|---|---|
| hashmap | `hash/hash.cu`, `hash/hash.cuh` (~580 lines) | 32- and 64-bit `atomicCAS` open-addressing insert/lookup, Murmur3 | `atomicCAS(unsigned long long*)` and `uint32_t` exist in HIP. Keys are initialised with `torch::full(max)` and values are written on insert, so nothing reads uninitialised memory. No risk. |
| serialize | `serialize/api.cu`, `z_order.h`, `hilbert.h`, `utils.h` (~740 lines) | integer bit twiddling, `__host__ __device__` helpers, host loops with `#pragma omp parallel for` | Pure integer code. The omp pragmas sit in hipcc-compiled TUs without `-fopenmp`, so they are serial on both vendors already (unchanged behaviour). No risk. |
| grid_sample | `grid_sample/grid_sample.cu` (~360 lines) | hashmap lookups for trilinear/nearest neighbours | No intrinsics. No risk. |
| spconv neighbour maps | `spconv/subm_neighbor_map.cu`, `sparse_neighbor_map.cu`, `migemm_neighmap_pp.cu`, `utils.h` (~1,400 lines) | dynamic `extern __shared__` (BLOCK_SIZE x V x 4 B), `int4`/`int2` vector loads, `__ffs`, `__popc`, `warpSize`, `__syncwarp` | Dynamic LDS is at most 256 x 32 x 4 = 32 KB, because the 32-bit gray code caps V at 32. That is under 64 KB on CDNA and RDNA. `reduce_code_kernel` (see Risks) is the only warp-level code. It indexes by runtime `warpSize`, so it is correct on wave32 and wave64. |
| bindings | `ext.cpp` | pybind11 module, host compiler | Host compiler only (cl/clang-cl on Windows, g++ on Linux). |

Not present anywhere (intake grep, rechecked): cuBLAS, cuFFT, cuRAND, cuSPARSE, cuSOLVER,
cuDNN, Thrust, CUB, CUTLASS, textures/surfaces, `__shfl*`, `__ballot*`, inline PTX,
cooperative groups, the driver API, NVRTC, pinned or managed memory, explicit
streams/events. Every launch uses `<<<grid, block[, shmem]>>>` on the default stream, on
both vendors.

**Triton surface (not in the CUDA census; it is the larger half of the project):**

- `flex_gemm/kernels/triton/spconv/`: 12 kernel modules, ~3.3k lines. These are
  implicit, masked-implicit and split-K variants of sparse conv and of submanifold-conv
  fwd/bwd. `config.py` selects autotune lists through `get_autotune_config` (device name
  substring first, then platform `'hip'`/`'cuda'`). `allow_tf32` is False on HIP, so dots
  use `input_precision='ieee'`.
- `flex_gemm/kernels/triton/grid_sample/`: `indice_weighed_sum_fwd/bwd.py` and their
  config.
- `flex_gemm/utils/autotuner.py`: a custom autotuner whose cache is keyed by
  `torch.cuda.get_device_name()` and persisted to `~/.flex_gemm/autotune_cache.json`.
- Python ops: `flex_gemm/ops/` (spconv, submanifold conv, grid_sample, serialize). Its
  `EXPLICIT_GEMM` path is im2col plus torch matmul (hipBLAS/hipBLASLt on ROCm) and serves
  as the in-repo numerical reference.

## Risk list

1. **C++ standard (all platforms, shared code, confirmed).** `setup.py` passes
   `-std=c++17` (`/std:c++17`, `-Xcompiler=/std:c++17` on Windows). From torch 2.14 on,
   `torch/csrc/api/include/torch/all.h` has `#error C++20 or later compatible compiler
   is required` (pytorch#178150, in v2.14.0, not in v2.13.0). It was confirmed on this
   host with the installed torch 2.16: `g++ -std=c++17 -E` of `<torch/extension.h>`
   hits the `#error` and `-std=c++20` is clean. Every current TheRock torch
   is >= 2.14, so **upstream `main` does not build against current ROCm torch on any
   OS** (and not against CUDA torch >= 2.14 either). `BuildExtension` appends its own
   standard only when the user passed none (`append_std17_if_no_std_present`, which now
   appends c++20; on Windows CUDA `win_cuda_flags(...) + ['-std=c++20']`). Fix: drop the
   explicit standard flags so torch supplies the one its headers need. Older torch
   keeps getting c++17 and torch >= 2.14 gets c++20. Bumping to a hard c++20 (what #31
   does on Windows only) would also break a CUDA 11.x nvcc with old torch.
2. **Windows+HIP flags.** `extra_compile_args['nvcc']` goes straight to hipcc on a ROCm
   torch (`win_hip_flags` = `COMMON_HIPCC_FLAGS + COMMON_HIP_FLAGS + cflags + arch`, with
   no filtering). `-Xcompiler=/EHsc` and the other `-Xcompiler=` options are nvcc
   syntax, and clang rejects them. This is the gsplat/pytorch3d class ("guard win32 device
   flags on the HIP backend"). Guard them on `IS_HIP`.
3. **Windows default target.** `GPU_ARCHS` defaults to `native`, and
   `--offload-arch=native` fails with Windows hipcc (cuda-to-rocm windows.md,
   "hipcc.exe and hiprtc traps"). Because an `--offload-arch` is present, torch's
   `_get_rocm_arch_flags` also skips `PYTORCH_ROCM_ARCH`. On Windows+HIP with no
   `GPU_ARCHS`, emit no `--offload-arch` so torch derives the targets from
   `PYTORCH_ROCM_ARCH`. Linux keeps `native`, upstream's tested behaviour.
4. **Windows host-compiler traps (not yet seen for this project; porter confirms).**
   - `/openmp` in `cxx`: under clang-cl it pulls in `libomp`, which TheRock may not ship.
     `ext.cpp` has no OpenMP, so on Windows+HIP it is safe to drop if the link fails.
   - CuMesh on the same toolchain hit a dllimport mismatch for `c10::ValueError` with a
     cl.exe-compiled binding TU. FlexGEMM's `ext.cpp` holds only pybind `m.def`s, and
     `TORCH_CHECK` lives in the hipcc-compiled `.cu` files, so it should not apply. If it
     does, CuMesh's `.cu` wrapper for the binding TU is the known fix.
   - hipcc `-c` static CRT (`-fms-runtime-lib=dll`) and Windows `.hip` registration:
     current torch handles both (`self.compiler._cpp_extensions += ['.cu', '.cuh',
     '.hip']`). Check the laptop's torch version before adding anything.
   - The hipify output path: sources sit under `flex_gemm/kernels/cuda/`, so hipify
     writes `flex_gemm/kernels/hip/**.hip` (the directory rewrite rule, strategy-b-torch.md).
     They are not under `third_party/`, so the CuMesh Windows prune does not apply. Add
     the generated `flex_gemm/kernels/hip/` to `.gitignore` only if the porter finds
     upstream leaves it untracked by accident. Do not commit hipified output.
   - `sparse_neighbor_map.cu` `data_ptr<T>()` without `template` (issue #19, PR #31)
     breaks nvcc on Windows. Clang accepts it, because the tensor's type is not
     dependent, so it is not ours to change. Leave it to #31.
5. **Triton on Windows AMD.** `triton-windows` (3.9.0.post30 seen 2026-11) carries the AMD
   backend. The precedent is spconv-triton forward+backward on gfx1151 (Pointcept) and
   ffpa-attn on gfx1101/gfx1201. Risks:
   - cold-compile time: dozens of kernels times autotune candidates. Expect tens of
     minutes on RDNA, so use a persistent `TRITON_CACHE_DIR` and a long window;
   - LDS overrun: the `'hip'` spconv list includes `B1=256,B2=256,BK=16, num_stages=2`,
     which is 64 KB of operand tiles for fp32 inputs, exactly the RDNA limit before any
     epilogue scratch. ffpa-attn saw Triton-Windows pick a 72 KB config on gfx1101 and
     poison the device. FlexGEMM's autotuner `_benchmark` does not catch
     `OutOfResources`. If a fp32 config fails, the fix is a guarded prune in FlexGEMM's
     autotuner or config list, a real port change that the porter makes then. fp16 (what
     the tests and TRELLIS.2 use) is 32 KB and safe.
6. **wave32 / RDNA Triton.** Upstream has no RDNA evidence at all. The `'hip'` configs
   carry `waves_per_eu` (accepted on gfx11/12). Only the `MI300X` device list uses
   `kpack`/`matrix_instr_nonkdim`, and it is selected by device name only. The platform
   name check `get_platform_name()` works on RDNA. Shipped autotune cache keys are exact
   device names (`AMD Instinct MI300X VF`), so every other AMD part starts cold. That is
   correct but slow, a performance note.
7. **`reduce_code_kernel` warp step (deferred item `flexgemm-wave-barrier-syncwarp`).**
   PR #18's `#define __syncwarp(...) __builtin_amdgcn_wave_barrier()` replaces HIP's own
   `__syncwarp()`. Torch passes `-DHIP_ENABLE_WARP_SYNC_BUILTINS=1` to extension builds,
   and HIP's version is `fence(release, wavefront); wave_barrier; fence(acquire,
   wavefront)`, so the macro swaps in a weaker form (the builtin is `IntrNoMem` in LLVM
   and orders no memory). The reduction ORs values from the same segment and is
   idempotent, so only a load hoisted above another lane's previous store could
   change a result. Jeff's 2026-10-09 ruling was defer until FlexGEMM is a TRELLIS.2
   dependency AND a wave32 card shows a real fault. The first half now holds, the
   second has not happened. **This plan keeps the ruling: no change.** The validators
   exercise the kernel on wave32 and wave64 (it runs in every `MASKED_IMPLICIT_GEMM*`
   neighbour-cache build, see the Test plan), and a fault there sends the porter to
   delete the macro (HIP's built-in `__syncwarp` is the stronger form) and the item back
   to Jeff.
8. **Upstream drift.** `dev/all_triton` (2.0) and PR #35 (Triton 3.9 fixes on that
   branch) show Triton 3.9 rejects some 2.0 kernels. `main`'s kernels compare `uint32`
   neighbour values with `0xffffffff`, which is legal. If the installed Triton (3.9.0 on
   this host's image) rejects a `main` kernel, it is a real finding for the porter, and
   #35's approach is the model.
9. Not applicable (checked): texture pitch or rule-of-five on handles (none), library swaps
   (none), CC-keyed dispatch (none; Triton configs key on the device-name string),
   unzeroed allocations (every `torch::empty` output is fully written before it is read).

## File-by-file change list

Expected fork delta is about 10 lines, all in `setup.py`:

1. `setup.py`, shared, no `[ROCm]` prefix (commit 1, "setup.py: let torch choose the C++
   standard"): remove `-std=c++17` from the Linux `cxx` and `nvcc` lists, and remove
   `/std:c++17` and `-Xcompiler=/std:c++17` from the Windows lists. Keep every other flag.
2. `setup.py`, `[ROCm]` (commit 2, "[ROCm] setup.py: Windows HIP flags and default
   target"): in the Windows branch, when `IS_HIP`, give hipcc
   `["-O3"] + cc_flag` with no `-Xcompiler=` options. Keep `cxx` as-is unless the link
   proves `/openmp` must go (risk 4). When `IS_HIP` and `GPU_ARCHS` is unset on Windows,
   set `cc_flag = []` so torch uses `PYTORCH_ROCM_ARCH`. The CUDA branches stay
   byte-identical apart from commit 1.
3. `README.md` (commit 3, optional, `[ROCm]`): a short "AMD GPUs (ROCm)" install note
   giving the documented command and the `GPU_ARCHS` / `PYTORCH_ROCM_ARCH` knob, plus
   `triton-windows` on Windows. The README has no install-per-platform section today,
   so keep it to a few lines and no device-selection env vars in front of commands.
4. Anything a validator finds (risks 4, 5, 7, 8) is added as a new commit with its
   evidence, never an amend.

No change to `.cu`/`.cuh`, Triton kernels, `ops/`, tests or `autotune_cache.json` is
planned.

## Build commands

Linux (any gfx). On a fleet host, build against the image's ROCm torch but install into
the job's own venv (`$FLEET_VENV`), never into the image's shared Python environment:

```
cd projects/FlexGEMM/src
GPU_ARCHS=<gfx> MAX_JOBS=<physical cores> \
  pip install . --no-build-isolation --no-deps
python -c "import flex_gemm, flex_gemm.kernels.cuda as k; print(k.__file__)"
```

`--no-build-isolation` because `setup.py` imports torch at top level, and an isolated
env would have none. `--no-deps` because a pip resolve of `torch>=2.4` / `triton` can
replace the ROCm torch (strategy-b-torch.md). Run the GPU tests from outside the source
tree so the installed package is the one exercised.

Windows (TheRock torch venv, `triton-windows` installed into it):

```
call "...\VC\Auxiliary\Build\vcvars64.bat"
set ROCM_HOME=<venv>\Lib\site-packages\_rocm_sdk_devel
set DISTUTILS_USE_SDK=1
set GPU_ARCHS=gfx1151
pip install . --no-build-isolation --no-deps
pip install --no-deps triton-windows filelock
```

(the CuMesh Windows recipe; see its notes "Windows build notes").

CUDA no-regression compile (before presenting to Jeff): on the mi355 fleet host,
`/opt/conda/envs/cuda-12.8/bin/nvcc` with a CUDA torch, `BUILD_TARGET=cuda pip install .
--no-build-isolation --no-deps`, or record `cuda-not-validated: <error>` per
moat-porting if only a ROCm torch is available there. Also diff the flag lists on both
sides of the change (strategy-b-torch.md: monkeypatch `setuptools.setup` to print
`extra_compile_args`) for `BUILD_TARGET=cuda` on Linux and with `platform.system`
patched to Windows, to show the CUDA flags changed only by the dropped `-std`.

## Test plan

Upstream's `tests/` are standalone scripts (`python tests/<file>.py`, run from `tests/`
or with `tests/` on `sys.path` for `utils.py`). Some assert; the conv scripts print
error and timing tables against an in-repo reference and assert nothing. Validators
run all of them as follows and record the commands, the printed tables and pass/fail
in notes.md.

**A. Asserting tests (run unchanged, every platform):**

| Script | Exercises | Pass criterion |
|---|---|---|
| `tests/hashmap.py` | CUDA ext hashmap insert/lookup, 4 key/value dtype pairs, RES 1..1024 | its `assert torch.all(values == values_)` |
| `tests/serialize.py` | z-order / hilbert encode+decode | its assert |
| `tests/spconv/out_coords.py` | hashmap and expand-unique output-coord builders vs torch | its asserts |
| `tests/spconv/neighbor_cache.py` | sparse-conv neighbour maps, every algorithm's cache (incl. masked -> `reduce_code_kernel`) vs torch | its `torch.equal` asserts |
| `tests/submconv/neighbor_cache.py` | submanifold neighbour maps and caches vs torch | its asserts |
| `tests/grid_sample.py` | CUDA-ext neighbour lookup + Triton weighted sum vs `grid_sample_3d_torch` | printed err; pass if err_max(min(abs,rel)) <= 1e-3 for fp32 (record values) |

**B. Conv correctness tables (fwd and bwd, sparse and submanifold):**
`tests/spconv/fwd.py`, `tests/spconv/bwd.py`, `tests/submconv/fwd.py`,
`tests/submconv/bwd.py`. The reference is FlexGEMM's own `EXPLICIT_GEMM` (torch matmul),
and `tests/spconv/fwd.py` also has a dense `torch.nn.Conv3d` column. Each Triton algorithm
(`igemm`, `igemmk`, `migemm`, `migemmk`) is compared with it. These scripts
`import spconv.pytorch` at top level for a third-party comparison column (traveller59
spconv, CUDA-only). Two ways, in order of preference:
1. `pip install --no-deps spconv-triton` (Triton drop-in for the spconv 2.x API) and run
   unchanged. If its API rejects `algo=spconv.ConvAlgo.MaskSplitImplicitGemm`, use (2).
2. A validator-side driver (in `agent_space/`, its full text recorded in notes.md) that
   loads each script's source, deletes the `import spconv...` line and the `'spconv':`
   entry from `kernel_functions`, optionally replaces the `config` list (below), and
   `exec`s it. No fork edit.

Pass criterion: for every config row, each Triton algorithm's `err_max` (fp16; the
column is min(abs, rel) in per-mille) is no worse than 2x the `dense` column on
`spconv/fwd.py`, and `err_mean` <= 5 per-mille on every script. Record the full tables.
A row that is far off on one algorithm and fine on the others is a port finding, not
noise.

Sizes: the scripts go up to RES=512 (dense reference = 137 GB, which OOMs and is caught
by `benchmark_kernel`). On a discrete GPU run the configs unchanged. **On the Windows
gfx1151 APU, cap the configs at RES <= 64** through the driver in B.2: big-tensor tests
there starve system RAM (standing rule). Record the cap.

**C. Examples and flex_gemm-only benchmarks (smoke, every platform):**
`examples/spconv.py`, `examples/submconv.py` (run to completion, finite output);
`tests/submconv/inf_ablation.py`, `tests/submconv/train_ablation.py` (flex_gemm only,
need matplotlib; run to completion, record timings; on the APU cap the sizes the same way).
`tests/submconv/inf_benchmark.py` and `train_benchmark.py` import torchsparse, fvdb,
warpconvnet and spconv, which are third-party CUDA libraries with no ROCm builds. Run
their FlexGEMM columns through the B.2 driver with those imports and columns removed;
the third-party columns are not FlexGEMM code.

**D. Cold-compile handling.** Set a persistent `TRITON_CACHE_DIR` and
`~/.flex_gemm` per host before the first run, and expect long first runs (B's bwd
scripts autotune many kernels). On a fleet host, start long runs with `setsid nohup`, an
exit marker and foreground polls (moat-porting, Fleet hosts: background commands die at
2 h). Do not read a slow first run as a hang without the `py-spy` check
(strategy-b-torch.md).

**E. Gates.** wave64 (gfx90a/gfx942), wave32 (gfx1100/gfx1201 Linux, or the Windows
RDNA host), windows (gfx1151 or another Windows RDNA host; one Windows RDNA run
satisfies wave32 and windows). Each validator writes `## Install as a dependency` for its
OS in notes.md from its own validated build, which is what TRELLIS.2 consumes and what
clears `dep-doc-gaps`.

**Non-GPU regression set:** none in the repo. CPU-side behaviour that must not change:
`import flex_gemm` on a machine without a GPU must still only fail where upstream fails,
and the CUDA build's compile flags must differ only by the dropped `-std` (flag-list
diff above).

## Open questions

None blocking. Recorded for the porter and for Jeff's information:

- The deferred `flexgemm-wave-barrier-syncwarp` stays deferred per Jeff's ruling (risk 7).
  Only a wave32/wave64 failure in the neighbour-cache or `migemm*` rows reopens it.
- The C++-standard fix is a shared bug fix that also helps CUDA users on torch >= 2.14. It
  overlaps textually with open PR #31. If #31 merges first, the porter syncs and drops
  the now-redundant Windows half.
- RDNA performance (autotune configs, cache entries for gfx11/gfx12) is not a gate.
  Record cold and warm timings so a later tuning pass has a baseline.
